# NBA Best Ball Points Prediction

Predict a player's performance in a popular NBA best-ball fantasy contest (`dk_pts`) for
**each game** of an upcoming season, using only information available before the
season starts.

A chain of Bayesian component models — availability, minutes-as-team-composition, and the
box-score components of `dk_pts` — is fitted in Stan on 30 seasons of NBA box scores. Whole
seasons are drawn from their joint posterior (`make simulate-season`), ranked into a draft
board, and played out in a simulated snake draft against an ADP field under real contest
structures (`make strategy-sweep`), which closes the chain from a box score to a roster.

This file is the overview. `CLAUDE.md` is the working reference for conventions and measured
facts; `docs/*-plan.md` hold the detailed designs.

---

## 1. Introduction

### The target

Scoring is a linear function of the box score, plus a nonlinear bonus:

```
dk_pts = 1.0·PTS + 0.5·FG3M + 1.25·REB + 1.5·AST + 2.0·STL + 2.0·BLK − 0.5·TOV + bonus
bonus  = 1.5 for a double-double, 4.5 for a triple-double
```

Implemented once, in [src/data/preprocess.py](src/data/preprocess.py) (`compute_dk_pts`).

Full contest rules are in [docs/dk_best_ball_rules.md](docs/dk_best_ball_rules.md): 16
players drafted in a snake draft, frozen for the season, with the best 7 by slot scoring
each week and four elimination rounds ending 4/4.

### The prediction-time constraint

This is the defining constraint of the project. **The draft happens after the preseason and
before the opener**, so at prediction time we know the **schedule**, the **season-start
rosters** (definitively which team each player is on), every team's and player's
**previous-season** statistics, and the **current season's preseason box scores**. We do
*not* know within-season trades, regular-season minutes, injuries, or form.

The information set is therefore a cross-season join: **current-season roster membership ×
prior-season statistics × current-season preseason statistics**. Two consequences shape
everything downstream. Minutes weights in any roster aggregate must come from season S−1,
since season-S regular-season minutes are unknown. And every feature except the
schedule-derived ones is constant within a player-season — the only per-game variation
available is opponent, home/away and rest.

**The preseason term was added 2026-08-13 and it is a change of problem statement, not a
feature.** Contest entry is open from the offseason through the opener, and drafting late
mainly reduces the risk of a season-altering injury landing between draft night and opening
night — so the offseason information set is no longer the one to build for. Preseason data
enters as **additional columns on existing heads**, difference-coded against the
prior-season features so that a coefficient of zero recovers the pre-2026-08-13 head
exactly; **four head groups carry a block today** — availability, marginal minutes, the
minutes composition (through its prior share) and ten of the eleven component rate heads — and
`stan.availability.preseason` / `stan.minutes.preseason` /
`stan.composition.preseason.adopt` / `stan.components.preseason` are exact rollbacks. One operational consequence is
load-bearing: two of the availability block's columns are read over the preseason's **tail**,
so **a complete preseason is now a production precondition** and the October runbook's
Oct 17–20 draft window is not advisory. See
[docs/preseason-plan.md](docs/preseason-plan.md).

### Where the variance actually is

Measured on 254,167 player-games (2014-15 → 2023-24) by `make variance-budget`
([src/eda/variance_budget.py](src/eda/variance_budget.py)). Within-player-season residual sd
is **9.445** dk_pts against a total sd of 14.566.

| Source | Share of variance | basis |
|---|---|---|
| Player-season identity | **57.96%** | of total per-game variance |
| **Own minutes played** | **46.40%** — *unknowable in advance* | of within-player residual |
| Opponent × season | 0.691% | of within-player residual |
| Home / away | 0.034% | of within-player residual |

Never quote a row without its basis: identity is 58% *of the total*, everything else is a
share of the remaining 42%. The opponent rows are in-sample ANOVAs on contemporaneous
opponent identity — ceilings, not achievable gains.

**So roughly 90% of attainable skill is getting each player's season-level rate right, and
1–2% is per-game modulation.** That budget is why this project spends its effort on
availability, minutes and component rates rather than on matchup features, and why the
deliverable is a *distribution* rather than a sharper point estimate.

### This project

This is the AI-assisted version 2 of a
[hand-coded effort using the R stack](https://github.com/danfosterfire/nba_stats). It

1. replaces an R/tidyverse stack with a Python one, for fun and learning;
2. refines the per-box-score-component Bayesian models (Stan);
3. adds simulated drafts and seasons, to use the full posteriors and test drafting
   strategies rather than learning them live;
4. communicates process and findings through a Streamlit dashboard.

---

## 2. Methods

### Data

30 seasons (1996-97 → 2025-26), ~380 MB of raw CSVs pulled from `nba_api` by
[src/data/fetch.py](src/data/fetch.py) and cleaned by
[src/data/preprocess.py](src/data/preprocess.py) into 731,906 regular-season player-games.
Four capture programs feed the availability side: the 20-season box-score inactive/DNP
backfill ([src/data/boxscore_status.py](src/data/boxscore_status.py), 2006-07+), the NBA's
daily injury-report PDFs ([src/data/injury_reports.py](src/data/injury_reports.py)), the
ESPN status feed ([src/data/injuries.py](src/data/injuries.py)), and DraftKings /
FantasyPros ADP boards ([src/data/adp_draftkings.py](src/data/adp_draftkings.py),
[src/data/adp_fantasypros.py](src/data/adp_fantasypros.py)).

Three of those sources are **not backfillable** and are on a deadline: the injury-report
PDFs age out of the CDN after ~7 months, the ESPN feed has no history, and the DraftKings
draft board is login-gated with zero Wayback presence and open only in October. `make
daily-capture` must stay on a cron; see [docs/adp-plan.md](docs/adp-plan.md).

Season-level features are assembled by [src/eda/season_matrix.py](src/eda/season_matrix.py)
into **Tier A** (30 seasons, 10,900 qualified player-seasons × 150 box-score columns) and
**Tier B** (13 seasons, adding tracking and hustle families, a strict column superset).
Each has an unfiltered twin (`season_matrix_roster_*`) for roster aggregation, because the
PCA's `GP≥20 & MIN≥10` filter drops real teammates who consume real minutes.

The full season-level EDA pipeline runs as `make eda`; each stage is one module in
[src/eda/](src/eda/) writing one artifact.

### Train / validation / test

**A temporal walk-forward by target season.** Every row is a season predicted from the
season before it, so the 30 data seasons give 29 target seasons, and the split is a
suffix of them:

| Split | Target seasons | Role |
|---|---|---|
| Train | 1997-98 → 2021-22 | fitting |
| Validation | 2022-23, 2023-24 | **variant selection — the only thing selection may read** |
| Test | 2024-25, 2025-26 | confirmation, quoted in Results |

`test_seasons: 2` is the knob; the seasons are derived by sorting the labels present and
taking the last two, never hard-coded.

**Since 2026-08-05 this is enforced by the code rather than by discipline.**
[src/models/held_out.py](src/models/held_out.py) makes the test split a *capability*:
`split_seasons` — the one function every head goes through — hands back a guarded frame
that **raises** when read, so `train, _ = split_seasons(...)` stays legal and
`score(model, test)` does not. Sweeps call `selection_split`, which never materializes the
held-out rows at all. The only thing that unlocks them is
[src/final_evaluation.py](src/final_evaluation.py) (`make final-evaluation`), which reads
which arm shipped from the artifact rather than re-deciding, refits on train **plus**
validation, and scores test once.

The rule had failed before that: the games-played head's Gate D was specified with the
incumbent's *test* figures as its bars, run on test, and settled which model ships — on a
margin a paired bootstrap could not distinguish from zero, which reversed when re-decided
on validation. Nothing in the code objected, because nothing in the code knew. **The
sweeps no longer emit a test column at all**, which also removed a confound: the test side
used to refit on train + validation at double the sampler iterations, so a val/test
disagreement conflated the evaluation rows with the training data and the chain length and
was never the replication check it looked like. That change cut roughly half the sampler
time across the Stan heads.

Anything fitted from data — spline knots, imputation means, the conversion floor's
shrinkage constant, the composition head's dispersion bin edges — is estimated on the
fitting half alone.

Three honest caveats, kept here rather than in a footnote because they are the places the
discipline is not clean:

- ~~**One comparison still has not gone through validation.**~~ ✅ **Closed 2026-08-08.**
  [availability.py](src/models/availability.py) was the last, and the caveat named the two
  decisions it left on the test split: the playoff-workload feature block and the choice of
  a GLM over a GBM and ridge. Both were re-decided on validation and **both stand — but the
  ladder's ordering reversed**, with the GBM going from third to first on mean CRPS. It does
  not take the head, because a paired bootstrap cannot distinguish its margin from zero and
  because it is *worse* than the GLM on the lowest realized-games quartile. That the caveat
  was pointing at a real reversal is the argument for having written it down.
  [stan_composition.py](src/models/stan_composition.py) was the other outstanding case —
  converted in code on 2026-08-05 with an artifact that still carried held-out columns — and
  that closed on **2026-08-08** when the head was re-run; nothing about its verdict reversed.
  [component_rates.py](src/models/component_rates.py) *was* the worst case, with no split
  guard at all because it defined its own `split_seasons`; that is fixed and pinned by a
  test.
- **The EDA layer is pooled over all 30 seasons, test included.** Nothing there is fitted,
  so no held-out score is inflated — but specification decisions came out of it
  (persistence splitting on attempts vs conversions, "shrink conversion percentages hard",
  minutes weighting, and the shot-attempt reparameterization itself). That is model design
  informed by data that includes the test window, and no split guard can see it.
- **The numbers the simulator is *given* rather than scored on are calibrated per fit
  window.** ⚠️ **This bullet said "the four numbers the simulator will consume as direct
  inputs" until 2026-08-16 and named the wrong four**, which is a correction worth keeping
  rather than a typo: two of them — the game-level minutes dispersion and the ten-game block
  variance inflation — are **diagnostics**, measured and reported and never read by a draw,
  and the injected per-(player, season) σ, which *is* read on every minutes draw, was not on
  the list at all. What the draw actually consumes is **three** numbers plus one at the
  contest layer: the residual copula, the per-game bonus overdispersion (which does double
  duty as the copula's inversion scale), the injected σ — graded by role since 2026-08-16 —
  and Gate C's `rho` below. `docs/sim-inputs-plan.md` holds the inventory and the dashboard's
  "Inputs beyond the heads" page draws it split by role.

  The windowed three carry a `fit_window` column and the consumers default to `train_val`;
  the full-window figures move by less than the precision they are quoted at,
  which is why the leak would never have announced itself. The injected σ is not among them —
  it is a config constant, not an artifact measured per window. **A third window, `train`, was
  added 2026-08-08**, because `train_val` is clean for a *test*-split readout and not for a
  *validation* one: it contains 2022-23 and 2023-24, which is exactly what the realized
  backtest scores against. Which window to consume is decided by what the number will be
  scored against, not by which is widest — and the same rule now governs the persisted
  posteriors, which are namespaced by window (`make posteriors`).
  - ⚠️ **There is a fifth such number and it is worse than the other four: Gate C's `rho`, the
    draft field's rank-noise rotation.** It is not a constant read from an artifact but is
    *solved by bisection per season* against that season's realized model-versus-market skill
    gap — so it is fitted on the rows the backtest then scores, which is a tighter loop than
    a fit-window choice. It is also solved **per arm**, which is why `sim_lift` is not
    comparable across two models: the 2026-08-15 counterfactual reads 0.399071 → **0.362309**
    and 0.395650 → **0.311690**, so the two arms' simulated worlds are literally not the same
    world. That the rotation *falls* for a better model is independent corroboration rather
    than a defect, and the realized readout — which is priced by pairing, not by the simulated
    bar — is unaffected. `docs/preseason-plan.md` P5 and session 6b.

### The output contract — twelve components, and `dk_pts` falls out

**`dk_pts` is deterministic given the components and is never predicted directly.** Twelve
quantities are modelled per player-game, built by
[src/features/targets.py](src/features/targets.py):

| Component | Likelihood | Exposure / trials |
|---|---|---|
| `min` (given availability) | successes / trials | **game length** — 48 in regulation; 53/58/… in OT |
| `fga` `fta` `reb` `ast` `stl` `blk` `tov` | count (negative binomial) | `min` |
| `fg3a` \| `fga` | successes / trials — the three-point **share** | `fga` |
| `fg2m` `fg3m` `ftm` | successes / trials | `fg2a` / `fg3a` / `fta` |
| *`fg2a`* | **derived**, `fga − fg3a` | — |

Only eight of the twelve reach the scoring function; `min` and the attempt counts matter
solely through the exposure and trials they supply. Reassembly is exact and linear:
`pts = 2·fg2m + 3·fg3m + ftm`, then `compute_dk_pts`.

> **Total attempts are the count and the three-point mix is a share of them.** A
> three-point attempt *substitutes* for a two, so fitting `fg2a` and `fg3a` as independent
> counts left that dependence for the residual copula to carry. Modelling `fga` and
> `fg3a | fga` enforces it by construction, keeps the posterior factorization exact, and
> is better specified — shot-mix *shares* persist like counts (0.886) while conversion
> percentages do not (0.500). Adopted 2026-08-03; `fg2a` becomes derived, exactly as `pts`
> already is, and the head count stays at eleven. See
> [docs/shot-attempt-basis-plan.md](docs/shot-attempt-basis-plan.md).

Two things force this decomposition rather than a single `dk_pts` head. First, the
double-double bonus is a **simultaneous threshold** on five components, so
`E[bonus] ≠ bonus(E[x])` and the deliverable is a *joint draw*, not twelve marginals.
Second, context effects **cancel across components** in the DK sum — `teammate_assist_supply`
moves 2.106 dk_pts of gross component movement into −0.254 net, an 8.30× cancellation
(`make context-value`), and the opponent main effect cancels 1.98× (`make opponent`). A
single `dk_pts` head fits the residue.

The trials denominator for minutes is derived rather than assumed:
[src/features/game_length.py](src/features/game_length.py) recovers every game's length from
summed team minutes ÷ 5 — validated by the two teams being independent estimates that
disagree on **0 of 37,986 games**. 5.93% of games go to overtime, so truncating at 48 would
censor the top of the minutes distribution in exactly the games where stars play most.

### The chain, and why the heads are fitted separately

The generative structure is a chain of conditionals:

```
availability → min | available → counts | min → makes | attempts
```

With disjoint parameter blocks and independent priors the joint posterior **factorizes
exactly**, so separate fits recover the identical posterior a single joint model would.
This is an identity, not an approximation, and it is what makes full Bayesian inference
affordable here — the prior attempt's `megamodel.stan` paid the joint-fit price for a
coupling that was not in the model and had to run on 1% of the data.

A second identity does the rest of the work: for `y_g ~ Poisson(m_g·e^η)` with η constant
within a player-season, the product over games factors into a Poisson on the season total
times a multinomial free of η. So ~10,000 player-season rows recover the coefficients of
731,906 player-game rows.

The correlation the simulator needs therefore enters at **draw time**, not fit time: one
shared `min` draw per player-game pushed through all eleven component heads as exposure
(minutes being 46.4% of within-player residual variance), plus a Gaussian copula on the
remainder. [src/eda/residual_correlation.py](src/eda/residual_correlation.py) writes that
matrix, conditioned on minutes, as an explicit simulator input.

### The fitted heads — what ships today

Four `.stan` sources in [src/stan/](src/stan/) serve every head, which is the
factorization argument as code:

| Source | Serves | Driver | In `make stan` |
|---|---|---|---|
| `betabinomial_glm.stan` | availability, minutes, the 4 conversion heads, **overtime onset** | [stan_availability.py](src/models/stan_availability.py), [stan_minutes.py](src/models/stan_minutes.py), [stan_components.py](src/models/stan_components.py), [stan_game_length.py](src/models/stan_game_length.py) | **yes** |
| `negbinomial_glm.stan` | the 7 count heads | [stan_components.py](src/models/stan_components.py) | **yes** |
| `composition_glm.stan` | the team-game minutes allocation | [stan_composition.py](src/models/stan_composition.py) | **yes** |
| `betageometric_duration.stan` | **overtime depth**, and the absence-spell length | [stan_game_length.py](src/models/stan_game_length.py), [stan_games_played.py](src/models/stan_games_played.py) | **yes** (game length only) |

Shared plumbing — compilation, sampling, and diagnostic extraction into a CSV rather than a
scrollback buffer — is in [src/models/stan_utils.py](src/models/stan_utils.py).
`make stan` runs all five: `stan-availability`, `stan-minutes`, `stan-game-length`,
`stan-components`, `stan-composition` — cheapest first, so a plumbing failure surfaces in
seconds rather than after the composition's nine hours, and with the composition last
because it imports the minutes head and measures itself against it.

**Availability** ([src/models/availability.py](src/models/availability.py) is the point-MLE
reference, [stan_availability.py](src/models/stan_availability.py) the Bayesian port) is a
beta-binomial GLM over games played out of team games. Games played is the largest lever on
the season total and the least persistent quantity in the project, so the head shrinks hard
toward a league/age baseline and emits a distribution.

**Since 2026-08-13 it also carries a ten-column preseason block, adopted against a gate it
did not clear.** P2's bar was a conjunction — a validation CRPS interval clear of zero with
the boundary held, **and** the rolling-origin harness agreeing — written because twice before
on this head a block won validation and shrank 4–6× rolling. It failed in the direction the
bar did not anticipate: validation could not resolve the arm (**−0.102 [−0.322, +0.121]** on
the draftable population) while the rolling harness passed both halves at **10 of 10**
origins (**−0.254 [−0.344, −0.162]**), on 4.6× the rows with 2.4× the precision, and
validation's interval contains the rolling estimate. The gate is recorded as failed, because
a bar re-read after seeing which side an arm landed on is not a bar; the ship is an owner
decision taken with that in view. What the block buys on this head is **calibration** rather
than accuracy — all seven arms carrying a preseason column improve `boundary_tail_error` at
both readings — and the arm that shipped is the wider one the *fitting half* preferred,
which reverses P1's "use a smaller block". `docs/preseason-plan.md` P2.

**It is a two-component mixture, and that head is selected on tail calibration rather than
on CRPS** — the only head in the project that is. A disrupted season is a *different event*,
not an extreme draw of a per-game rate, so it gets its own component with its own mean and
dispersion and a weight `π` that carries covariates: the arm can say **who** is at risk,
which a wider frailty can only refuse to. Against the single-component head it halves the
boundary error (**0.0120** against 0.0201) and ties CRPS (+0.011, interval spanning zero),
which is a ship only because the objective was stated first —
`docs/availability-window-plan.md` §7 and §8. Two things fall out of it that a metric table
does not show. The main component's dispersion spread widens from 1.54× to **1.97×** across
prior-minutes buckets, almost entirely because a *star's* ρ falls by a fifth once his
disrupted seasons live somewhere else. And the port check's reference has to be the mixture's
own point MLE, since the two likelihoods estimate different parameters.

**And the head's largest calibration error is at a unit it cannot see, which is a limit of
the target rather than of the likelihood.** A beta-binomial asserts the season's ~82 games
are exchangeable trials; absences come in spells, so they are not. But permuting a
played/missed vector leaves `gp` exactly where it was, so clustering and frailty enter its
variance only through `C + ρ(n − C)` and are **not separately identified** — no arm on the
likelihood axis could have found this, and five already-fitted non-exchangeable arms agree,
the marginal-neutral one reproducing CRPS, PIT and the tail error to every decimal. At the
**scoring period** DK actually seats a lineup in, the assumption understates a star's chance
of three consecutive dead weeks by **9.1×**. What pays for it is the simulator's **layout
step**, one layer below the likelihood, which was shipped on a judgement rather than a gate
and has now been re-measured and rebuilt. `allocate_spells` alone recovered **81.5%** of the
pooled gap and left a residual whose sign flipped by role. That residual turned out to be two
mechanisms: the **44.17%** of missed games that are tenure edge blocks, whose length and
position an interior-fitted beta-geometric at a random start gets wrong, and an overflow
branch that collapsed a heavily-absent row to one giant block on **41.28%** of fringe rows
against **2.32%** of star rows. Neither fix ships alone — removing the branch on its own
takes the pooled longest-dead-run recovery to **0.2707**, because the accident was doing most
of the clumping — and together they land the recovery within 4% of 1.0 pooled and in
**[0.852, 1.039]** across role. `sim.availability.layout = tenure_merge`,
`make availability-exchangeability`, `docs/availability-window-plan.md` §11 and §13.

**Minutes** in the shipped chain is [stan_minutes.py](src/models/stan_minutes.py): the
marginal `min | available`, fitted season-collapsed as successes out of real game length,
selecting `logit(own) + spline`. What it supplies the simulator is the season-level
**spread** — the one thing the composition head below is structurally unable to produce.

**Since 2026-08-13 it carries a five-column preseason block and fits from 2004-05, and this
is the head where the preseason is worth the most.** The columns are the *season-centred*
preseason minutes delta plus four age-split missing indicators; the window is cut because the
preseason panel begins at 2004-05 and this head used to fit from 1997-98, so 8,306 fitting
rows become 6,152. Unlike availability it cleared both halves of its bar outright —
validation CRPS **−4.789 [−8.08, −1.59]** minutes and the rolling harness **−7.940
[−9.41, −6.44]** at 12 of 13 origins, the first block in the project whose *rolling* reading
is the larger one. Integrated over `beta` the block is worth **−5.911** CRPS minutes against
a control fitted on the same rows with the columns removed (136.958 against 142.869), so the
increment survives the posterior and grows slightly under it. Centred rather than raw because
preseason minutes are compressed by an amount that varies with the calendar and this head has
no year term to absorb it; centring also takes the season-total bias from −36.68 to **−10.95**,
better than the pre-block head's own −19.24. `docs/preseason-plan.md` P3.

~~⚠️ **The block narrowed this head's season-level ρ from 0.05025 to 0.041894**, ~9%, and
this head ships *for* its season-level spread.~~ ✅ **Re-read 2026-08-14, and the narrowing is
an improvement rather than a cost.** At the season unit the post-block head is better on every
row of `make minutes-unification`: CRPS **136.60** (was 144.35), MAE **190.21** (was 200.12),
R² **0.8947** (was 0.8829), bias **−11.91** (was −14.09) and PIT KS **0.0668** (was 0.0735),
on a predictive sd that did narrow, **277.23** against 302.75. Sharper *and* better calibrated
is the signature of a real covariate, not of a head that lost its spread. The consequence is
downstream and it is a reversal — see the injection stake below.

**Minutes as a team-game composition**
([stan_composition.py](src/models/stan_composition.py)) is the second minutes head, and it
does something the marginal one cannot: it allocates each team-game's `5 × game_length`
minutes among the players who played, by decomposing the multinomial into sequential
binomial trials ordered by prior-season minutes share, with the per-player cap enforced
through the trials (`m_k = min(U, R_k)`) rather than checked afterwards. That gets **both**
per-game constraints — the exact team total and the individual cap — where the marginal head
gets only the cap, and it makes teammate-absence redistribution a *fitted* quantity.

**Both minutes heads ship, and `make minutes-unification` settled which half each one owns.**
This file used to say they "compose rather than compete", with the marginal head owning the
season-level mean and the game-level dispersion. Two of those three claims were wrong. The
game-level dispersion is a *data measurement* that happens to live in `stan_minutes` — the
fitted object never appears in it — and the composition fits its own, role-graded from
**0.14019** for fringe players to **0.0735171** for stars. And the composition now *beats* the
marginal head on the season-level **mean**: scored at the season unit on the 742 validation
player-seasons both cover, MAE **184.541** against **190.21** and R² **0.902836** against
**0.8947**. Its bias is **7.9673** against **−11.91** — the two miss in opposite directions
and the composition's is now the larger of the two in absolute terms, where before the blend
it was the smaller (+2.41 against −14.09). (Pre-blend the same row read MAE 200.28 and R²
0.8848, so the composition trailed on both.)

⚠️ **The season-unit figures in this section and the next are the last reading against an
un-blended composition, and they are being re-measured.** The composition adopted its own
preseason block on 2026-08-14 (`docs/preseason-plan.md` P5), so the earlier note that "the
composition carries no preseason block and its numbers are bit-identical" is withdrawn — it
was true only of the 2026-08-14 morning reading, when every figure that moved moved because
the *marginal* head improved. Against the pre-preseason marginal head the MAE/R²/bias row
read 200.12, 0.8829 and −14.09, with the composition leading on MAE rather than trailing it.

What survives is the season-level **spread**. Summed composition draws give a season-total
predictive sd of **60.5757** minutes against the marginal head's **277.23** — **4.58×** too
narrow, CRPS **155.941** against **136.60** — a paired-bootstrap gap of **+19.3382**,
interval **[+13.0465, +25.8738]** — because draws that are iid across games cannot manufacture
season-level heterogeneity. ⚠️ **The sharpest form of that no longer holds**: the composition
now *clears* the season-unit no-fit carry-forward floor (**155.941** against **161.29**),
where before the preseason blend it did not (170.06 against the same 161.29) — on the same
draws that clear its own per-team-game floor decisively (**4.26174** against **4.47013**). Beating a
no-fit floor on CRPS and carrying 4.6× too little spread are compatible, and PIT KS
**0.331968** against the floor's 0.1242 is what separates them — so the "same head, two
units, opposite verdicts" line survives as a statement about *spread* and not about the
floor. (The gap was +33.45 [+26.15, +41.335] at 4.29× before the blend, and +25.70
[+18.96, +33.25] at 4.68× before the marginal head's own block.)

**That gap is a missing parameter, not a ceiling — measured, and it matters for what gets
built next.** A *shared* effect cannot fix it: a season term is a league-wide shift, and
against a head that allocates every minute in the league it has **0.000000%** of the residual
variance to reach. But a **per-(player, season)** effect is not shared, and injecting one
into the existing posterior — `σ·z` per player-season per draw, shared across that player's
games, re-run through the head's own allocation — moves the season-total predictive sd from
60.58 to **237.911** at σ = **0.375** and the CRPS to **130.692**, which now **beats** the
marginal head (**−5.91125**, interval **[−10.3858, −1.50137]**) while keeping the team
constraint exact, at a season-unit calibration that is effectively identical to it: PIT KS
**0.0665499** against 0.0668. MAE barely moves, so it buys spread and not fit.

**The caveat that made that a bound rather than a score is closed, and the two grids now
agree exactly.** σ was read off validation, which is the split it is scored against — so the
same grid was re-run on the last two *training* seasons (1,145 player-seasons). On the
pre-blend head its optimum was 0.450 against validation's 0.375, one step apart; on the
blended head **both grids put the optimum at 0.375** (train CRPS **108.834** at 0.450 against
its own optimum, validation **132.437** at 0.450 against **130.692** at 0.375). Two grids on
disjoint rows agreeing exactly is stronger evidence than the step-apart version it replaces,
and it is why **σ moved to 0.375 on 2026-08-14** — its *input* changed, rather than the
constant being re-decided.

✅ **And the injection stake reverses back, decisively.** Before the composition's own
preseason block, σ = 0.450 **lost** to the marginal head at +6.26 [+0.92, +11.49]. It now
**ties** (**−4.16592**, interval **[−8.4861, +0.0114191]**, PIT KS **0.0952291**) and the
shipped σ = 0.375 **wins** with an interval clear of zero. **On whether that retires the marginal head, see below — the
answer turns out not to depend on which head predicts better.**

✅ **That arm has now been measured and fitted, at the pilot window, and it is worth more than
the gap.** The preseason enters this head as a blend into `w_share` rather than as a column on
`beta`, because `w_share` also reaches the **offset** and the **allocation order**, which no
coefficient touches. `make composition-preseason` scored it on the head's own no-fit floor
(**−0.19972 [−0.21661, −0.18225]** CRPS minutes per player-game, nothing fitted) and `make
composition-preseason-fit` then fitted it against a same-window control: **−0.20883 [−0.22129,
−0.19693]**, retention **1.040**, so the increment *grows* under the posterior. At the season
unit — the unit this whole comparison lives at — the floor's tie becomes **−14.62649 [−19.48503,
−9.54922]** once the head is fitted, through the **mean** rather than the spread.

✅ **And it survives the window the head actually fits — the increment grows a third time, and
it now beats the shipped head rather than a pilot control.** Session 4d takes the arm to the
covered window (2004-05 on, the cut the panel forces since it starts there and this head fits
from 1996-97) and adds a **third** arm at 1996-97 carrying no preseason column, which is the
shipped head and the only thing a ship decision can be read against. The increment reaches
**−0.23418 [−0.24551, −0.22314]** CRPS minutes per player-game on the draft pool, retention
**1.115**; against what ships today it is **−0.25409 [−0.26520, −0.24315]** per player-game and
**−17.27296 [−22.32569, −11.92164]** per player-season. The coverage cut is worth **−0.01991**
of that on its own — P3's direction, but **8.5%** of the increment rather than the quarter that
round paid, so the block is not mostly window. Season MAE falls **17.70** minutes while the
predictive sd *narrows*, so it is the mean and the injected σ is untouched. Three fits, 7.76 h,
0 divergences. **Still not shipped**: `stan.composition.preseason` configures the measurement
target only, and adopting it means a `first_season` of 2004-05 on the head plus `make
posteriors --groups composition`, which is P5. `docs/preseason-plan.md` sessions 4b–4d.

**So the injection ships**, applied by `minutes_unification.rehydrate_composition` — a
consumer gets the effect by loading the head rather than by remembering to apply it, and 0.0
recovers the un-injected head exactly.

**And since 2026-08-16 it is graded by role**, because one number was missing in *both*
directions at once: at the shared σ fringe player-seasons were still **1.73×**
under-dispersed while stars were **over**-dispersed at **0.86×**. The raw season-total
deficit is nearly role-flat (4.42 fringe to 3.84 star at σ = 0) — what is not flat is where a
constant *logit-scale* σ lands after it has been through the allocation and summed to a
season. `sim.minutes.player_season_sigma_by_role = [0.600, 0.375, 0.375, 0.300]` over the
composition's own `rho_bin`, a **2.00×** spread, with `player_season_sigma = 0.375` kept as
the shared rung the grading was selected against. Each bucket's grid was searched
coordinate-wise on the **training** seasons and confirmed on validation, all four agreed to a
grid step, and **two of the four did not move** — the shipped change is fringe and star.
Fringe `sd_ratio` goes **1.7276 → 1.2757** and its low PIT tail **0.1535 → 0.0833**, and the
star bucket comes back the other way, **0.8601 → 1.0444**; the win against the marginal head
widens from **−5.9112 [−10.3858, −1.5014]** to **−6.4504 [−10.9991, −2.0419]**; the
team-season total's sd across draws stays exactly 0. Independent
corroboration: the quadrature line's converged *fitted* σ on the same axis reads 0.60898 /
0.51013 / 0.46440 / 0.29075 at a 2.09× spread, so two instruments sharing no arithmetic agree
on both **ends** to within a grid step. Deleting the config key restores the pre-2026-08-16
draw bit-for-bit. `make minutes-role-sigma`,
[docs/draw-time-calibration-plan.md](docs/draw-time-calibration-plan.md).

*(Two provenance corrections closed the same day, both of the "correct behaviour, wrong
record" kind. This line read σ = 0.450 until 2026-08-16 and was stale — σ moved to 0.375 on
2026-08-14 when the composition's preseason blend made the train and validation grids agree.
And `minutes_unification.SHIPPED_PS_SIGMA` read 0.450 while config read 0.375 for two days;
it is the fallback for a missing key, so nothing misbehaved and the live value was always
the config's, but the reader it misleads is the one about to change the injection. It is
0.375 now and a test reads `configs/default.yaml` to keep the two together.)*

`composition_glm.stan` also carries the effect as an **optional fitted parameter** (`sigma_u`,
with `U_n = 0` nesting the shipped head exactly), and `make posteriors` persists its scale and
takes precedence over the constant. It is built but **not fitted**: at 12.0× the shipped arm's
cost it did not converge inside the budget, and four sampler-side remedies — centring, sharing
`rho`, dropping `rho`, and a dense metric — all made it worse rather than better, so the cost
is intrinsic to a 2,204-parameter hierarchical posterior rather than a configuration mistake.
`make composition-effects` is the ladder and `docs/potential-to-dos.md` is the write-up. What
a fit would still buy is a σ estimated jointly with the coefficients and a predictive that
integrates over σ's posterior instead of plugging one in.

**Since 2026-08-16 there is a third representation of that same parameter, and unlike the
second one it converges: marginalize the latents instead of sampling them.** Integrating
each unit's `u_i` out with per-unit Gauss–Hermite quadrature inside the model block leaves
**~35 parameters at any window** rather than 2,204 at the pilot and 12,307 at the full one —
so the funnel cannot exist and `dense_e` is back inside `choose_metric`'s 20×-warmup rule. `Q = 0` nests
the sampled-latent head exactly and both blocks stay in the file, because they are two routes
to one posterior. The measured catch is that the node count is a property of the node
**placement**: fixed nodes are 2.6 nats out at Q = 61, while nodes placed at each unit's own
Newton-fitted likelihood mode and curvature reach **2.5e-6 nats at Q = 21**.

At probe scale, with the sampled latent refitted beside it on the same 26,039 rows, it reads
R̂ **1.0138** / min ESS **719** against that arm's **1.1317** / **25**, and the two agree on
the effect — `sigma_u` **0.47927** marginalized against **0.48095** sampled. Cost is **21.9×**
the shipped control at Q = 21, and **1.85× the latent arm** — but per-fit wall clock is the
wrong metric here, because the two arms are not producing the same thing: at **0.898** ESS per
draw against **0.031**, the marginal arm delivers **15.5×** the effective samples per second.
Feasibility follows that, not the 1.85×. The pilot window is one arm overnight (**~11.5 h** at
a 700+300 budget its own sampling efficiency justifies); the **full** window is **~74 h**
serially, so `reduce_sum` over units is a prerequisite for it rather than an optimization.

The role-graded arm converges better still (R̂ **1.0069**) at the same cost, and its σ grades
monotonically — **0.609** fringe to **0.291** star, a **2.09×** spread that matches the fitted
per-game ρ's 1.91× in axis, direction and size.

⛔ **The line is parked as of 2026-08-16, and not on a failed gate — every gate above was
passing.** Three days of sampler buys a σ that is *estimated* rather than plugged in, but not
the clean all-posterior simulator: the injected σ is one of five draw-time inputs the simulator
is *given* rather than fits, so retiring it leaves four. The replacement targets the same
measured defect for hours instead of days —
[docs/draw-time-calibration-plan.md](docs/draw-time-calibration-plan.md), grading the injected
σ by role at draw time, with the fitted values above as its independent reference. The
quadrature block stays in the file, inert at `Q = 0`.
`docs/composition-quadrature-plan.md`.

✅ **That replacement was built, measured and shipped the same day, and it vindicated the
parking twice over.** It cost 13 minutes of numpy against the quadrature arm's three days,
and the two agree: 0.600 / 0.375 / 0.375 / 0.300 from the draw-time grid against 0.609 /
0.510 / 0.464 / 0.291 fitted, both **ends** within a grid step and the spread 2.00× against
2.09×. So the fitted σ's remaining value is the two things a plugged-in constant still cannot
do — estimate σ jointly with `beta`, and integrate the predictive over σ's posterior — and
neither is the reason the gradient was worth having. See the injection section above.

**Six independent routes agree on the effect size**, which is why a plugged-in σ is
defensible in the meantime: 0.375 (validation CRPS grid), ≈0.41 (calibration — the ratio of
the head's residual sd to its predictive sd), **0.450** (train CRPS grid, the shipped value),
0.4776 and 0.4809 (fitted in Stan, non-centred and centred), and **0.4375** (the marginal
profile above, at fixed `beta` and a shared σ). Two of those share no arithmetic with the
others, and the last is the only one that does the integral rather than plugging a constant
in — so it corroborates the size without being a posterior.

**And the constraint is not just a cost — it is a dynamic the contest is sensitive to.** A
team's season minutes are a fixed pot, so teammates' season totals are negatively correlated:
a fixed sum over K players forces mean pairwise **r = −1/(K−1)**, which at the measured
**16.05**-player roster size is **−0.0664**. Over **963** single-team validation
player-seasons the composition sits on it at **−0.0503828**. The marginal head reads
**+0.0007** (−0.0001 before the preseason block — a null either way) and puts a **946.1**-minute
predictive sd on a team season total that is physically fixed. That is invisible in every marginal metric and lands on two strategy axes
directly: a same-team stack's minutes are *anti*-correlated rather than independent, and
handcuffing a starter with his backup is a hedge that exists only if the model carries the
sign.

The composition also covers **1,111** validation player-seasons against the marginal head's
742 — the **369** rookies and low-minute players the `≥ 200 prior minutes` filter drops, who
are draftable.

⚠️ **"Which head supplies the simulator's minutes" was never actually open, and this file
said otherwise until 2026-08-14.** It used to read: *"`stan_minutes` ships today because it is
the only head with the right season-level spread today, and the simulator's minutes draw is
the open design question rather than a settled blend."* That is wrong about the code.
`src/sim/` imports neither `StanMinutes` nor `rehydrate_minutes`, and `sim/season.py` never
looks up `artifacts["minutes"]` — it consumes `availability`, `composition`,
`game_length_ot`, `game_length_depth`, `gp_duration` and the eleven component heads. **The
simulator's minutes have come from the composition plus the injected σ all along.**

What `sim/season.py` takes from the `stan_minutes` *module* is not the fitted head: `beta_shapes`
is arithmetic four other modules also use, `game_level_dispersion` is a **data measurement**
in which the `StanMinutes` object never appears, and two Gate bars are read from *artifacts*.

**So retiring the marginal head would mean ceasing to fit it, and the case for keeping it does
not depend on which head predicts better.** It is the `independent_comparator` in
`stan_composition`'s own ladder — the control that never trains on the composition window,
and the thing today's **−0.4186** headline is measured against — and it is the season-unit
reference the injected σ is calibrated against, which matters more now that σ has moved. It
costs **514 s** to persist against the composition's **7,357 s**, so there is no compute
argument either. A composition that beats it makes the comparator *more* valuable, not less.
The one development that would genuinely retire it is a **fitted** `sigma_u`, which removes
the need for a plugged-in σ and hence for a reference to plug it in against —
`docs/potential-to-dos.md`.

**Components** ([stan_components.py](src/models/stan_components.py)) fits the seven negative
binomial counts and four beta-binomial conversions, each against a mandatory no-fit floor.
Its head lists come from [component_rates.py](src/models/component_rates.py), which models
total attempts as a count and the three-point mix as a share — see the output contract above.

**Since 2026-08-15 ten of the eleven component heads carry a preseason block**, and the
eleventh is the only head in the whole preseason round that a paired instrument measured as
*worse* with one. `make components-preseason`
([components_preseason.py](src/models/components_preseason.py)) armed **all eleven** — not the
six P1's ΔR² screen short-listed — and held them to the same two-reading bar the other heads
were held to. That widening is the round's strongest finding: **the screen's sign did not
survive on a single one of the three heads it called actively harmful**, and two of those three
(`fta`, `fg2m|fg2a`) clear the real bar outright while a fourth excluded head, `fg3a|fga`,
turns out to be the third-largest result in the round. On the shipped `own_delta_shrunk` arm
**every one of the eleven clears the rolling half** and **seven clear validation**; no head
anywhere in the round has an interval clear of zero on the wrong side.

The shipped column is the **volume-shrunk** delta plus the four age-split missing indicators —
five columns per head, with `k` fitted per head on the fitting half and read from the artifact
that fitted it rather than pinned in code. **Centring is not used here**, unlike `stan_minutes`:
it loses on five heads with intervals clear of zero, because the compression it corrects is a
property of *levels* and a per-36 rate has already divided the exposure out.

⚠️ **`fg3m|fg3a` is opted out by `stan_components.PRESEASON_EXCLUDE`**, on three agreeing
instruments — P1's attribution (the head's apparent gain was entirely the shared indicator
pair, its own delta **−0.00237**), 6b's pooled point MLE (**+0.0069**, the only positive of the
eleven), and the posterior control (**+0.02914**, worse on CRPS, NLL *and* PIT KS at once). A
conversion delta is a logit of a percentage over ~15 preseason attempts and shooting percentage
is the least persistent quantity in the box score, so the block adds variance and no signal.
The rollback is exact and checked: the head reproduces its pre-block fit at NLL **3.1676**
against **3.1677**, on the full 8,630-row window rather than the cut one.
`stan.components.preseason: false` is the exact rollback for the other ten.
See `docs/preseason-plan.md` session 6b.

**Game length** ([stan_game_length.py](src/models/stan_game_length.py), `make
stan-game-length`) is the one input a *forward* simulation cannot look up. Both minutes heads
need a length, and every backtest so far read it from `game_length.parquet` because the games
had already happened. It is two heads on two existing sources — a beta-binomial on whether a
game goes to overtime, collapsed to ~30 season cells, and a beta-geometric on how deep — and
it is the only head besides `min` that ships a season term. Four fits and **0.3 s** of
sampler time, the cheapest in the project by three orders of magnitude. Its draw is taken
**once per game and shared by both teams**, because overtime is a property of the game.

**Season terms** ([src/models/season_terms.py](src/models/season_terms.py), `make
season-terms`) is a 108-fit ablation asking whether any head needs a year trend or a year
random effect to track league-wide era movement. [src/eda/season_effects.py](src/eda/season_effects.py)
measures the league series it would be correcting for.

### Simulation — built

[src/sim/season.py](src/sim/season.py) (`make simulate-season`) writes the layer's output
contract: a `player × scoring_period × sim` tensor of `dk_pts` and a `uint8` games-played
twin, per season. Per-game draws happen *inside* it and are summed into the twenty scoring
periods immediately, so nothing downstream ever materializes a `player × game × sim` array.
Its four prerequisites are [posteriors.py](src/models/posteriors.py) (`make posteriors`,
twenty heads), [scoring_periods.py](src/features/scoring_periods.py),
[stan_game_length.py](src/models/stan_game_length.py) and
[minutes_unification.py](src/models/minutes_unification.py).
[docs/simulations-plan.md](docs/simulations-plan.md) holds the specification and the Gate A
readout; the specification is pinned by measurements rather than guesses:

- **Draw, never plug in.** `E[min]` and `E[gp]` are wrong inputs to a threshold bonus. That
  now includes the length of the game itself: `stan_game_length` draws it, once per game and
  shared by both teams.
- **Minutes come from both heads, and that is measured rather than assumed**
  (`make minutes-unification`): the per-game allocation and its role-graded dispersion from
  the composition, and the season-level **spread** from the marginal head, injected as a
  per-(player, season) σ that has been **graded by role** since 2026-08-16
  (`make minutes-role-sigma`). ⚠️ **Both block-inflation figures are diagnostics and this
  bullet used to imply otherwise**: the **2.43×** ten-game figure from
  `make serial-correlation` is a *target*, not something the draw reads — the serial
  structure the simulator has is *produced* by those season-constant σ shocks — and the
  **4.65×** game-level figure is likewise a check on the composition's draws rather than an
  input to them.
- **Sequential structure goes on minutes and nowhere else.** There is no shooting hot hand:
  both field-goal conversion heads are measured nulls, so those eleven heads stay collapsed.
- **Bonus overdispersion is unit-specific** — 0.10 at the season unit, **0.025** at the
  player-game unit the simulator will actually draw at
  ([src/features/targets.py](src/features/targets.py), `make component-targets`).
- **Availability absences are a mixture, not a Markov chain.** A constant hazard matches the
  mean spell length and misses both tails, so the spell length is modelled as a
  beta-geometric — a geometric hazard with a Beta frailty integrated out, which beats the
  geometric by 11,278 log-likelihood points at one extra parameter.
- **The games-played process is a tenure decomposition, not one chain over the schedule.**
  A departure is an absorbing hitting time, not a low recovery rate, so a recurrent chain
  relocates it to the player's first absence and over-predicts the left tail. The head is
  **entry index × exit index × a within-tenure two-state chain**
  ([docs/games-played-plan.md](docs/games-played-plan.md), `make games-played`).

### Ranking, drafting and tournaments — built

The reason the simulation layer exists, and the chain is now closed end to end. Simulated
seasons give a posterior over each player's season trajectory; that becomes a draft ranking,
which is executed in a simulated snake draft against an ADP-based opponent model, and the
resulting portfolios are replayed both inside the simulated world and against realized box
scores under real tournament structures. Four stages, four `make` targets:

| stage | target | what it does |
|---|---|---|
| the market | `make adp` | [src/features/adp.py](src/features/adp.py) builds a point-in-time-safe ADP panel — ADP is a forecast of the same target, so it is the most leakage-prone input in the repo and gets three separate dates per row |
| the contest | `make bracket` | [src/sim/bracket.py](src/sim/bracket.py) seats the best 7 of 16 by slot in each period and runs the four-round chain with its cascading tie-break, wildcards and payouts |
| the draft | `make draft` | [src/sim/draft.py](src/sim/draft.py) runs the snake against a field drawn from the DK-recalibrated consensus plus rank noise |
| the sweep | `make strategy-sweep` | [src/sim/strategy.py](src/sim/strategy.py) — **24 strategies × 5 tournament structures × 2 validation seasons at 500 simulated worlds each**, paired inside the world, plus the realized readout. All five captured structures are swept under one stake-parity entries rule; every stake is simulated, and which contests to actually enter is an open decision |

[dashboard/economics.py](dashboard/economics.py) derives the tournament structures from
`data/raw/dk_best_ball_tournament_*.csv`: five real tournaments, Round 1 a zero-consolation
knockout in every one, and rake expressed as a break-even edge hurdle (+10.45% to +17.60%)
because that is the unit a measured edge can be compared in.

ADP belongs in *this* layer, not in the prediction layer — that is a settled decision, so the
blend weight sweeps a real axis rather than a weight the model already absorbed.

**And the layer ships a tool, not only a backtest.** [dashboard/draft_room.py](dashboard/draft_room.py)
is a live recommender that prices each candidate inside a completed roster — one click per
pick, against a season's simulated field — and it is the one file in `dashboard/` allowed to
import from `src/`. It runs both as page 9 of the dashboard and standalone under `make
draft-room`, because draft night is a thirty-second clock. See
[docs/simulations-plan.md](docs/simulations-plan.md).

### What was tried and deprioritized

An LSTM/Transformer trunk over prior-season game logs ([src/models/lstm.py](src/models/lstm.py),
[transformer.py](src/models/transformer.py), [src/train.py](src/train.py)) is retained but
deprioritized. Held out, the prior-season game *sequence* is worth **+0.0059 R²** over four
season aggregates, and +0.0074 above its own shuffled-order null — the trunk is mostly
re-deriving a season mean the season matrix already holds.

---

## 3. Results

Headlines only. Every figure is reproduced by the `make` target named beside it and lands in
`outputs/`. **This file is audited**: `make docs-audit` re-derives each quoted result from
its artifact and **exits non-zero** on disagreement, so a headline copied here and never
refreshed fails the build rather than quietly misleading. Sampler timings are the one
exception — presence-checked, not value-checked, because they measure the machine rather
than the model.

**The availability head is the largest measured win.** `make availability-model` /
`make season-total`. Scored by CRPS in games on validation, the beta-binomial GLM reads
**10.006** against a GBM's **9.876**, ridge's 10.004 and a league/age baseline's 13.387. The
GBM leads on the mean and the GLM ships anyway, for two measured reasons: a paired bootstrap
over the 883 rows puts the gap at **−0.1297** with a 95% interval of **[−0.3154, +0.0672]**,
and by realized-games quartile the GBM is **+0.333** CRPS *worse* on the seasons that fell
apart — the population the head exists for. On the actual
deliverable it is worth **−210 dk_pts of season-total MAE** against assuming a full season
(610.8 → 400.5), with bias falling from +523.3 to −3.1. The oracles settle which half of the
error dominates: perfect games played gives MAE 214.4 against perfect rate's 261.9.
(Both ladders were held-out measurements until they moved to validation — the season total
on 2026-08-05, where it read 646.3 → 435.1 and 221.3 against 302.7, and the CRPS row on
2026-08-08, where it read GLM **10.795** against GBM 10.888, ridge 10.896 and 13.614.)

**The component rate side is nearly saturated from prior-season information alone.**
`make component-rates` / `make stan-components`. A no-fit floor — prior per-36 rate × actual
minutes, no fitting — scores validation R² **0.81–0.95**, and the best fitted head beats it by
+0.0013 to +0.0334. Every head is quoted against that floor; `ftm|fta` does not clear it at
all. (These moved from the held-out seasons to validation on 2026-08-05, where they read
0.82–0.94 and +0.0019 to +0.0203.)

⚠️ **Saturated against *prior-season* information, which is not the same as saturated — and
`make components-preseason` is where that distinction became a number.** Six preseason games
are not prior-season information. Measured at each head's own distributional unit rather than
by R², the preseason block is worth **more than the entire fitted head is worth over
arithmetic** on three of the eleven heads: on `reb` fitting buys **0.1507** CRPS rebounds over
the floor and the block buys **1.0929** more (**7.25×**), on `fg3a|fga` the ratio is **3.77×**
and on `fga` **1.17×**. **Six of eleven armed heads clear the two-reading gate outright and
ten of eleven ship**, the other four on a rolling half that passes at 9–13 of 13 origins
against a validation half that cannot resolve them on 706 rows — the same owner decision the
availability head's block was taken under.

**The block survives the posterior**: `stan_components` fits each armed head a same-window
`__no_preseason` control, and **ten of eleven hold at a median retention of 0.985**, four of
them *growing*. The eleventh, `fg3m|fg3a`, goes the other way and is rolled back — the only
measured-worse result in the entire preseason round. The R² screen that picked the original
short list was wrong in both directions: `reb` was its *smallest* clearing count head and has
the largest block-to-fit ratio here, while `ftm|fta` was its **largest** increment and is a tie
at both readings. `docs/preseason-plan.md` session 6b.

**The specification that matters is scale, not curvature — except where the likelihood
changes the answer.** Putting the player's own prior rate in on the log scale is worth
almost everything; linear-in-raw-rate inside `exp()` is unusable (`fg3a` held-out R²
**−19.00**). Under the negative binomial, though, splines are not a refinement but the
difference between a model and a failure on the skewed heads (`blk` 0.649 → 0.832), which
reverses what the Poisson fits implied. (That pair read 0.679 → 0.858 on the held-out
seasons, before the sweep moved to validation on 2026-08-06, and 0.673 → 0.831 before the
preseason block cut the fitting window on 2026-08-15.)

**A head that "failed" by two parts in a thousand did not fail.** `fta` was recorded as
falling below its no-fit floor, making the whole free-throw family a null; on the
validation split it clears by **+0.0198** — **+0.0144** before the preseason block, so the
reversal widened — and only `ftm|fta` still fails. The reversal is
the fourth of its kind since the held-out split was locked, and all four turned on test
margins under 1%.

**The minutes composition beats the independent draw on the independent draw's own metric.**
`make stan-composition`, **re-run 2026-08-14 with the preseason-blended offset, fitting
2004-05 on**. On validation, CRPS **4.26174** against the independent comparator's
**4.68034** (**−8.94%**), while also hitting the team total exactly where the independent
draw misses by **33.6451** minutes per team-game. The plan predicted a wash and budgeted for
arguing on capability instead. Two sub-results: the pure binomial decomposition is *worse*
than the no-fit floor (PIT KS **0.182**) — dispersion is the difference between a model and a
failure again — and dispersion is genuinely role-graded, fitted at **0.140** for fringe
players against **0.074** for stars, a **1.91×** spread that cuts calibration error by
**40.9%**. The comparator row is the control: it never trains on the composition window and
reproduced to six decimals when the head moved off the held-out split on 2026-08-08, as did
the selected arm's rank — nothing about the verdict reversed.

*(Before the blend was adopted this table read CRPS 4.4945 against 4.7842, **−6.06%**, a
33.89-minute team miss, binomial PIT KS 0.192, and dispersion 0.1768 against 0.0855 for a
2.07× spread cutting calibration error by 35%. Every figure improved, and **the dispersion
fell across every bucket**, which is the mechanism working as specified rather than a
separate result — a better offset leaves less residual overdispersion for the beta-binomial
to carry, so the graded spread narrows even as calibration improves. `betabinom_ot_graded`
is selected again, which the changed offset did not oblige.)*

**And the same head loses to the same comparator at the season unit, which is why both
minutes heads ship.** `make minutes-unification`. Summed to season totals on the 742
validation player-seasons both heads cover, the composition reads CRPS **170.06** against the
marginal head's **136.60** — a paired-bootstrap gap of **+33.45** minutes, interval
**[+26.15, +41.335]** — and does not clear the no-fit carry-forward floor's **161.29** at that
unit. The mean is not what fails: MAE **200.28** against **190.21**, R² **0.8848** against
**0.8947**, bias **+2.41** against **−11.91**. The predictive **spread** is, at **4.29×** too
narrow (sd **64.65** against **277.23**, PIT KS **0.3341** against **0.0668**), because
iid-across-games draws cannot make season-level heterogeneity. **A head is only a model at
the unit it was scored at**, and this is the cleanest demonstration of that in the repo: one
posterior, two units, opposite verdicts against the same two floors. The follow-up measured
in the same target — an injected per-player-season effect at σ = 0.375 gets the gap to
**+5.57 [−0.07, +11.06]**, while a league-wide season term has **0.000000%** of the residual
variance to reach — is in §2's minutes section, because it changes what gets built rather than
what shipped.

**Re-read 2026-08-14 against the preseason-armed marginal head, and the injection's verdict
reverses.** Every figure in the paragraph above moved because the *marginal* head improved;
the composition carries no preseason block and reproduced bit-for-bit. Before the block the
gap was **+25.70 [+18.96, +33.25]** at 4.68× and the shipped σ = 0.450 injection **tied** the
marginal head (**−1.49 [−6.14, +3.22]**); it now **loses** at **+6.26 [+0.92, +11.49]**. σ
itself is unmoved — its grid never contained the marginal head — but "retiring the marginal
head is a live prospect" is withdrawn.

**The 3PA/2PA substitution is best handled by reparameterization — re-measured
un-handicapped, and now shipped.** `make stan-substitution` for the measurement;
`component_rates.COUNT_HEADS` for the adoption. Modelling `fga` as a count
and `fg3a | fga` as a beta-binomial share on `fga` trials beats two independent count heads
by **−0.501041 nats** per player-season on validation, with each head fitted at its own
selected variant and both arms swept — a legitimate comparison because the coordinate change
is a bijection with unit Jacobian. The originally recorded −0.793 / **−0.771**
had both arms pinned at `log_own`, where `fg3a` scored R² **0.3719** against **0.9046**
for the spline it actually selected, so the canonical arm was handicapped; removing the
handicap costs 0.306 nats of the margin and the result survives anyway. ⚠️ **That handicapped
figure now reads −0.7218**, because `make stan-components` rewrites it and the preseason block
moved ten of the eleven heads underneath it — while the un-handicapped **−0.501041** is
computed inside `stan_component_substitution_sweep.csv`, which only `make stan-substitution`
rewrites and which has not re-run. **The two files therefore sit on opposite sides of the
block and may no longer be differenced**, which retired the floating-point identity check
between them; `docs/shot-attempt-basis-plan.md` works it through. The gate itself is decided
inside the sweep alone and is unaffected. **The strongest
version is that the basis beats the model**: the reparameterized *no-fit floor* beats the
canonical basis's *fitted* configuration by **−0.440841**. (The gate went validation-only on
2026-08-06 with `src/models/held_out.py`; the test column it used to carry read −0.493549 and
is kept as a record in `docs/shot-attempt-basis-plan.md`.) Adopting it also retired
this project's worst misspecification: `fg3a` scored **−19.00** R² under a linear
predictor, where the `fga` that replaces it scores 0.9527 and clears the highest floor of
any count head.

**No head ships a season term, and the ceiling on ever needing one is ≤5% of MAE.** `make
season-terms`. An oracle that rescales each scored season by its own realized league total
— the ceiling on any trend, year effect or manual override — is worth a median **1.29%** of
base MAE across heads, and at most 4.97%. A trend moves bias in both directions across the
count heads rather than removing it, so its apparent win on the season total is
cross-component cancellation. The minutes head is the single exception and adopts a year
effect. What a year effect *is* worth is joint spread: **+10.4%** on a 15-man roster's
season-total sd, against +0.6% from shared coefficient uncertainty.

**The drafting edge is large in the simulated world and the realized readout cannot confirm
it — which is the result, not a caveat.** `make strategy-sweep`. Against a symmetric-field
null, the arm that ships lifts its Round-1 advance probability by **0.236915** in the 600k
Shootaround's simulated worlds and by **0.199380** on the two validation seasons replayed
against realized box scores. ~~⚠️ That is **not** attributable to the preseason block, because
four things moved in one pass and the previous `strategy_*.csv` was overwritten.~~ ✅ **Closed
2026-08-15 by a paired counterfactual** (`make preseason-contest`), which refits **all four**
head groups with the block off and freezes σ at 0.375 in both arms. Essentially **all** of the
chain's Gate A gain is the block — the counterfactual lands within **0.11** and **0.49**
dk_pts of the recorded pre-block season-total MAE — and the realized lift is higher with the
block in **9** of **10** season × tournament cells, **+0.095962** at the 600k, with the tenth
cell at exactly **0.000000**: no cell moves *against* the block. The *simulated*
side resolves nothing at a bar of **0.074835**, but its `adp` control — a board identical
across arms — moved **−0.00515** against the 24-strategy mean of **+0.04585**, so unlike the
availability mixture's null the gain is not the world getting easier.
The second number is not a smaller version of the first: the
simulated side pools 500 drawn worlds per season and the realized side has exactly one, so
its intervals cover most of the table and it selected nothing. ⚠️ **Gate D no longer fails
outright, and the change is one cell.** The two buy-in tiers select materially different
rosters in **1** of **6** paired comparisons in the shipped arm — 2022-23's tier-aware
`bracket_ev` — against **0** of 6 in the counterfactual and 0 of 6 before the component block.
One of six, on the arm the block improved, with the same season's other tier-aware comparison
and both of 2023-24's staying below the bar, is not support for "draft differently for a bigger
field"; it is a single cell crossing a threshold. The conclusion stands and the count does
not.

⚠️ **Both headline numbers moved on 2026-08-16 when the injected σ was graded by role, and
neither move is evidence for the grading.** The pair read 0.230387 / 0.197293 before the
chain was re-run. On the **simulated** side that is not a comparison at all: Gate C's `rho` is
solved per arm and moved 0.362309 → **0.358192** and 0.311690 → **0.317198**, so the two runs
score in different worlds — the caveat §2 already carries. On the **realized** side, which
*is* comparable because both arms face the same box scores, the shipped strategy improves in
only **4 of 10** season × tournament cells at a mean of **−0.0027**, and the +0.002087 at the
600k is one season up (0.131040 → 0.144217) and one down (0.263546 → 0.254543). The `adp`
control moved **exactly 0.000000**, as it must — an ADP board does not read our tensor. **So
the contest layer neither confirms nor contradicts the graded σ**, which is the expected
result at N = 2 seasons and is why that calibration was decided at the season unit where it
was measured. `docs/draw-time-calibration-plan.md` §9.

One consequence for the paragraph above: `preseason_block_contest.csv` is a **captured** pair
(`captured_at` 2026-08-15) and today's chain re-run did not touch it, so every figure it
carries still describes what it measured. But its `preseason` arm is no longer the shipped
chain — re-taking that counterfactual against today's arm would be a fresh two-pass run, and
nothing here has done it.

*(Superseded by the component block's chain re-run on 2026-08-15, and kept beside the
corrections: the simulated lift read **0.2358** and the realized **0.204098**; the P5
counterfactual's realized delta at the 600k was **+0.102767** across 10 of 10 cells with an
`adp` control of **−0.006490** against a 24-strategy mean of **+0.034429**; `reb`'s block bought
**0.8213** for a ratio of **5.45×** against `fga`'s **1.07×**, both read on the declared primary
arm rather than the shipped one; and the autodraft twin led by **+0.00841967** while giving up
**0.105298**.)*

**The edge is not an artifact of a too-simple field, and automation's price is the
objective, not the executor.** `make draft-sim-need` / `make strategy-sweep-need` /
`make strategy-sweep`. Giving the field lineup reasoning is a measured null twice over: a
joint Gate B calibration fits the slot-reaching lean at **0** picks (the observed ADP
curve carries none, degrading fastest in the elite region), and against a *stipulated*
8-pick lean the shipped arm's simulated lift **rises** to **0.306** — slot-reaching pays
value for shape, so the fitted pure-ADP field is the harder opponent and stays shipped.
On the execution axis, submitting our best feasible ranking to DK's own autodraft is
identical to clicking it under DK's 8G/8F/3C caps — and beats the uncapped click by
**+0.00128856** (600k, resolved) — while giving up **0.0602481** of simulated lift against the
shipped per-pick objective, which no static board can express. *(Those two read +0.000303134
and 0.0706785 before the 2026-08-16 chain re-run; both are simulated-side figures, so the
per-arm `rho` caveat above applies to them as well.)* The 30-second-clock
fallback is safe; the objective is the half worth defending. See
[docs/simulations-plan.md](docs/simulations-plan.md), "The field with lineup reasoning,
and the execution axis".

**The sampler behaved.** 37 component fits with 0 divergences and every fit clearing every
convergence bar, 54 season-term fits with 0
divergences and 0 treedepth saturation, and the availability port reproduces the point MLE
of its own likelihood with the MLE inside the 95% credible interval for 45 of 45 terms —
where the *reference* is load-bearing: scored against the single-component MLE the 2026-08-12
mixture read 19 of 24, which measures the likelihood change rather than the port. Cost is
concentrated
entirely in the spline variants. Dropping the test side halved the component fit count from
74 and cut sampler time from 305.0 to **137.4** minutes *while* raising every selection fit
to full-length chains — which incidentally fixed the one fit that used to miss its R̂ bar.
1,880 tests pass (`.venv/bin/python -m pytest tests/`).

---

## 4. Discussion

The variance budget in §1 says the season-level rate is ~90% of attainable skill, and the
results bear that out from an uncomfortable direction: the rate side is close to saturated
by a no-fit carry-forward, while availability — the largest lever on the season total — is
the *least* persistent quantity measured here (r = 0.317 year over year). The oracle
comparison makes this concrete. Perfect knowledge of games played is worth more than perfect
knowledge of the rate. Effort spent on richer rate features is spent against a floor that is
already 0.81–0.95 R²; effort spent on the availability distribution and on the joint
structure between components is not.

That is why the deliverable is a joint draw rather than a set of marginals, and it is where
the modelling and the contest meet. The bonus is a simultaneous threshold on five
components; a best-ball lineup is a max over 16 players each week; and a tournament payout is
convex to the point of being a zero-consolation knockout in Round 1. All three are functions
of the *shape* of the distribution, not of its mean. A model that improves marginal CRPS by
1% and gets the correlation structure wrong is worth less here than one that does the
reverse — which is the argument for the Bayesian posterior (every player shares
coefficients, so one draw moves the whole board together) and for the composition head (a
team's minutes are zero-sum, so a teammate's absence has to land somewhere).

The honest caveat is unchanged and worth restating. **Five games of the current season
settle 86% of the season total** (R² 0.859, `make target-profile`), against +0.0086 R² for
the entire team-context block. The prior-season-only constraint costs roughly two orders of
magnitude more than any feature in this repo can buy back. It is a product requirement — the
draft happens before the season — and everything here is built for that frame, but it sets
the ceiling.

Known limits, logged so they are not rediscovered: mid-season trades are not modelled (13.6%
of players appeared for 2+ teams in 2023-24); 14.7% of season-start roster minutes have no
usable prior-season row and are imputed rather than dropped; and the free-throw family does
not currently clear its floor.

---

## 5. Repository map

```
src/data/       fetch, preprocess, availability capture (box-score status, injury
                reports, ESPN feed), ADP capture, and `capture_calendar.py`, which
                writes what all four of those archives hold as an artifact rather
                than a printout
src/features/   component targets, game length, team context, opponent, availability, ADP,
                and scoring periods — the NBA week grid DK's tournament rounds sit on
src/eda/        the season-level analysis pipeline — one module per artifact
src/models/     the Stan heads (availability, minutes, composition, components,
                game length, season terms) plus the sklearn references they are checked
                against, and `posteriors.py`, which persists every fitted head's thinned
                draws and design recipe so nothing downstream has to refit —
                `model_cards.py` turns those into the flat tables the dashboard reads,
                since the dashboard may not open a pickle that can score a frame
src/stan/       four .stan sources for twenty-plus heads
src/sim/        the simulation and drafting layer — numpy over the posterior artifacts,
                so nothing here needs CmdStan. `season.py` writes THE tensor,
                `bracket.py` runs the four-round contest, `draft.py` the snake against
                an ADP field, `strategy.py` the sweep, and `draft_room.py` the live
                recommender's engine
dashboard/      data visualizations over the artifacts — a multipage Streamlit shell over
                the Overview, the PCA player-style fingerprint, the model detail pages
                (one renderer, seven blocks, a class table), the inputs the simulator is
                *given* rather than fits, the tournament & strategy page and the live
                draft board, which also ships standalone as `make draft-room`;
                reads artifacts only, never refits
docs/           plan docs — predictions, availability, minutes composition, ADP,
                simulations, EDA, provenance, dashboard, contest rules
```

## 6. Quick start

```bash
make install          # create .venv and install requirements
make fetch            # pull raw data from nba_api (long)
make eda              # the full season-level EDA sweep
make stan             # fit the availability, minutes and component heads
make dashboard        # the Streamlit views at http://localhost:8501
make test             # pytest
```

The Stan heads need a CmdStan toolchain, which pip does not manage:

```bash
.venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Always use `.venv`, never the system Python. Two guards run over the documentation itself:
`make docs-audit` re-derives every quoted *result* in the plan docs from its artifact and
fails on a mismatch — sampler timings are presence-checked instead, since they measure the
machine rather than the model, and `make dashboard-audit` reports drift between the decision registry
and the docs it distills. See `CLAUDE.md` for conventions.
