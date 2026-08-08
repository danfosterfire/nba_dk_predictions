# NBA Best Ball Points Prediction

Predict a player's performance in a popular NBA best-ball fantasy contest (`dk_pts`) for
**each game** of an upcoming season, using only information available before the
season starts.

A chain of Bayesian component models — availability, minutes-as-team-composition, and the
box-score components of `dk_pts` — is fitted in Stan on 30 seasons of NBA box scores. Those
fits are built; the layers they exist to serve are not yet. Drawing whole seasons from their
joint posterior, ranking players from those draws, and backtesting drafting strategies under
real contest rules are the next three stages, and are labelled as planned below.

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

This is the defining constraint of the project. Before the season starts we know the
**schedule**, the **season-start rosters** (definitively which team each player is on), and
every team's and player's **previous-season** statistics. We do *not* know within-season
trades, current-season minutes, injuries, or form.

The information set is therefore a cross-season join: **current-season roster membership ×
prior-season statistics**. Two consequences shape everything downstream. Minutes weights in
any roster aggregate must come from season S−1, since season-S minutes are unknown. And
every feature except the schedule-derived ones is constant within a player-season — the only
per-game variation available is opponent, home/away and rest.

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

- **One comparison still has not gone through validation.**
  [availability.py](src/models/availability.py) is not converted, so the playoff-workload
  feature block and the choice of a GLM over a GBM and ridge remain test-set decisions.
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
- **The four numbers the simulator will consume as direct inputs are now calibrated on
  train + validation**, because they are *given* to the simulator rather than scored by it:
  the residual copula, the game-level minutes dispersion, the block variance inflation and
  the bonus overdispersion. Their artifacts carry both windows under a `fit_window` column
  and the consumers default to `train_val`; the full-window figures move by less than the
  precision they are quoted at, which is why the leak would never have announced itself.

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

Three `.stan` sources in [src/stan/](src/stan/) serve every head, which is the
factorization argument as code:

| Source | Serves | Driver | In `make stan` |
|---|---|---|---|
| `betabinomial_glm.stan` | availability, minutes, the 4 conversion heads | [stan_availability.py](src/models/stan_availability.py), [stan_minutes.py](src/models/stan_minutes.py), [stan_components.py](src/models/stan_components.py) | **yes** |
| `negbinomial_glm.stan` | the 7 count heads | [stan_components.py](src/models/stan_components.py) | **yes** |
| `composition_glm.stan` | the team-game minutes allocation | [stan_composition.py](src/models/stan_composition.py) | **yes** |

Shared plumbing — compilation, sampling, and diagnostic extraction into a CSV rather than a
scrollback buffer — is in [src/models/stan_utils.py](src/models/stan_utils.py).
`make stan` runs all four: `stan-availability`, `stan-minutes`, `stan-components`,
`stan-composition` — in that order, because the composition imports the minutes head and
measures itself against it.

**Availability** ([src/models/availability.py](src/models/availability.py) is the point-MLE
reference, [stan_availability.py](src/models/stan_availability.py) the Bayesian port) is a
beta-binomial GLM over games played out of team games. Games played is the largest lever on
the season total and the least persistent quantity in the project, so the head shrinks hard
toward a league/age baseline and emits a distribution.

**Minutes** in the shipped chain is [stan_minutes.py](src/models/stan_minutes.py): the
marginal `min | available`, fitted season-collapsed as successes out of real game length,
selecting `logit(own) + spline`. It supplies two of the three numbers the simulator needs —
the season-level mean, and separately the **game-level** dispersion, which a season total
cannot identify on its own.

**Minutes as a team-game composition**
([stan_composition.py](src/models/stan_composition.py)) is the second minutes head, and it
does something the marginal one cannot: it allocates each team-game's `5 × game_length`
minutes among the players who played, by decomposing the multinomial into sequential
binomial trials ordered by prior-season minutes share, with the per-player cap enforced
through the trials (`m_k = min(U, R_k)`) rather than checked afterwards. That gets **both**
per-game constraints — the exact team total and the individual cap — where the marginal head
gets only the cap, and it makes teammate-absence redistribution a *fitted* quantity. The two
heads compose rather than compete: `stan_minutes` still owns the season-level mean and the
game-level dispersion, neither of which the composition produces.

**Components** ([stan_components.py](src/models/stan_components.py)) fits the seven negative
binomial counts and four beta-binomial conversions, each against a mandatory no-fit floor.
Its head lists come from [component_rates.py](src/models/component_rates.py), which models
total attempts as a count and the three-point mix as a share — see the output contract above.

**Season terms** ([src/models/season_terms.py](src/models/season_terms.py), `make
season-terms`) is a 108-fit ablation asking whether any head needs a year trend or a year
random effect to track league-wide era movement. [src/eda/season_effects.py](src/eda/season_effects.py)
measures the league series it would be correcting for.

### Simulation — planned

Not built. [docs/simulations-plan.md](docs/simulations-plan.md) holds the specification,
which is pinned by measurements rather than guesses:

- **Draw, never plug in.** `E[min]` and `E[gp]` are wrong inputs to a threshold bonus.
- **Three minutes numbers, which compose rather than substitute**: the season-level mean
  from the minutes head, the **game-level** dispersion (4.65× binomial, measured separately
  because a season total cannot separate per-game from per-season noise), and the **2.43×**
  ten-game block variance inflation from `make serial-correlation` for serial dependence
  between games.
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

### Ranking, drafting and tournaments — planned

Also not built, and the reason the simulation layer exists. Simulated seasons produce a
posterior over each player's season trajectory; that becomes a draft ranking, which is
executed in a simulated snake draft against an ADP-based opponent model, and the resulting
portfolios are replayed against realized box scores under real tournament structures.

The pieces already in place are the market side and the contest economics.
[src/features/adp.py](src/features/adp.py) builds a point-in-time-safe ADP panel — ADP is a
forecast of the same target, so it is the most leakage-prone input in the repo and gets
three separate dates per row. [dashboard/economics.py](dashboard/economics.py) derives the
tournament structures from `data/raw/dk_best_ball_tournament_*.csv`: five real tournaments,
Round 1 a zero-consolation knockout in every one, and rake expressed as a break-even edge
hurdle (+10.45% to +17.60%) because that is the unit a measured edge can be compared in.

ADP belongs in *this* layer, not in the prediction layer — that is a settled decision, so the
blend weight sweeps a real axis rather than a weight the model already absorbed.

### What was tried and deprioritized

An LSTM/Transformer trunk over prior-season game logs ([src/models/lstm.py](src/models/lstm.py),
[transformer.py](src/models/transformer.py), [src/train.py](src/train.py)) is retained but
deprioritized. Held out, the prior-season game *sequence* is worth **+0.0059 R²** over four
season aggregates, and +0.0074 above its own shuffled-order null — the trunk is mostly
re-deriving a season mean the season matrix already holds.

---

## 3. Results

Headlines only. Every figure is reproduced by the `make` target named beside it and lands in
`outputs/`. **This file is audited**: `make docs-audit` re-derives each quoted figure from
its artifact and **exits non-zero** on disagreement, so a headline copied here and never
refreshed fails the build rather than quietly misleading.

**The availability head is the largest measured win.** `make availability-model` /
`make season-total`. Scored by CRPS in games, the
beta-binomial GLM reads **10.795** against a GBM's 10.888, ridge's 10.896 and a league/age
baseline's 13.614 — gradient boosting does not beat a 19-feature GLM. On the actual
deliverable it is worth **−210 dk_pts of season-total MAE** against assuming a full season
(610.8 → 400.5), with bias falling from +523.3 to −3.1. The oracles settle which half of the
error dominates: perfect games played gives MAE 214.4 against perfect rate's 261.9.
(The season-total ladder moved from the held-out seasons to validation on 2026-08-05, where
it had read 646.3 → 435.1 and 221.3 against 302.7; the head-vs-baseline CRPS row is still a
held-out measurement, because `make availability-model` has not moved yet.)

**The component rate side is nearly saturated from prior-season information alone.**
`make component-rates` / `make stan-components`. A no-fit floor — prior per-36 rate × actual
minutes, no fitting — scores validation R² **0.81–0.95**, and the best fitted head beats it by
+0.0013 to +0.0334. Every head is quoted against that floor; `ftm|fta` does not clear it at
all. (These moved from the held-out seasons to validation on 2026-08-05, where they read
0.82–0.94 and +0.0019 to +0.0203.)

**The specification that matters is scale, not curvature — except where the likelihood
changes the answer.** Putting the player's own prior rate in on the log scale is worth
almost everything; linear-in-raw-rate inside `exp()` is unusable (`fg3a` held-out R²
**−19.00**). Under the negative binomial, though, splines are not a refinement but the
difference between a model and a failure on the skewed heads (`blk` 0.673 → 0.831), which
reverses what the Poisson fits implied. (That pair read 0.679 → 0.858 on the held-out
seasons, before the sweep moved to validation on 2026-08-06.)

**A head that "failed" by two parts in a thousand did not fail.** `fta` was recorded as
falling below its no-fit floor, making the whole free-throw family a null; on the
validation split it clears by **+0.0144** and only `ftm|fta` still fails. The reversal is
the fourth of its kind since the held-out split was locked, and all four turned on test
margins under 1%.

**The minutes composition beats the independent draw on the independent draw's own metric.**
`make stan-composition`, fitted on all 30 seasons. On validation, CRPS **4.4945** against the
independent comparator's 4.7842 (**−6.06%**), while also hitting the team total exactly where
the independent draw misses by **33.89** minutes per team-game. The plan predicted a wash and
budgeted for arguing on capability instead. Two sub-results: the pure binomial decomposition
is *worse* than the no-fit floor (PIT KS **0.192**) — dispersion is the difference between a
model and a failure again — and dispersion is genuinely role-graded, fitted at **0.177** for
fringe players against **0.085** for stars, a **2.07×** spread that cuts calibration error by
**35%**. The comparator row is the control: it never trains on the composition window and
reproduced to six decimals when the head moved off the held-out split on 2026-08-08, as did
the selected arm's rank — nothing about the verdict reversed.

**The 3PA/2PA substitution is best handled by reparameterization — re-measured
un-handicapped, and now shipped.** `make stan-substitution` for the measurement;
`component_rates.COUNT_HEADS` for the adoption. Modelling `fga` as a count
and `fg3a | fga` as a beta-binomial share on `fga` trials beats two independent count heads
by **−0.501041 nats** per player-season on validation, with each head fitted at its own
selected variant and both arms swept — a legitimate comparison because the coordinate change
is a bijection with unit Jacobian. The originally recorded −0.793 / −0.771
had both arms pinned at `log_own`, where `fg3a` scored R² **0.3719** against **0.9046**
for the spline it actually selected, so the canonical arm was handicapped; removing the
handicap costs 0.306 nats of the margin and the result survives anyway. **The strongest
version is that the basis beats the model**: the reparameterized *no-fit floor* beats the
canonical basis's *fitted* configuration by **−0.440841**. (The gate went validation-only on
2026-08-06 with `src/models/held_out.py`; the test column it used to carry read −0.493549 and
is kept as a record in `docs/shot-attempt-basis-plan.md`.) Adopting it also retired
this project's worst misspecification: `fg3a` scored **−19.00** R² under a linear
predictor, where the `fga` that replaces it scores 0.9489 and clears the highest floor of
any count head.

**No head ships a season term, and the ceiling on ever needing one is ≤5% of MAE.** `make
season-terms`. An oracle that rescales each scored season by its own realized league total
— the ceiling on any trend, year effect or manual override — is worth a median **1.29%** of
base MAE across heads, and at most 4.97%. A trend moves bias in both directions across the
count heads rather than removing it, so its apparent win on the season total is
cross-component cancellation. The minutes head is the single exception and adopts a year
effect. What a year effect *is* worth is joint spread: **+10.4%** on a 15-man roster's
season-total sd, against +0.2% from shared coefficient uncertainty.

**The sampler behaved.** 37 component fits with 0 divergences and every fit clearing every
convergence bar, 54 season-term fits with 0
divergences and 0 treedepth saturation, and the availability port reproduces the point MLE
with the MLE inside the 95% credible interval for 21 of 21 terms. Cost is concentrated
entirely in the spline variants. Dropping the test side halved the component fit count from
74 and cut sampler time from 305.0 to **137.4** minutes *while* raising every selection fit
to full-length chains — which incidentally fixed the one fit that used to miss its R̂ bar.
779 tests pass (`.venv/bin/pytest tests/`).

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
                reports, ESPN feed), ADP capture
src/features/   component targets, game length, team context, opponent, availability, ADP
src/eda/        the season-level analysis pipeline — one module per artifact
src/models/     the Stan heads (availability, minutes, composition, components,
                season terms) plus the sklearn references they are checked against
src/stan/       three .stan sources for eleven-plus heads
dashboard/      nine-tab project walkthrough; reads artifacts only, never refits
docs/           plan docs — predictions, availability, minutes composition, ADP,
                simulations, EDA, provenance, dashboard, contest rules
```

## 6. Quick start

```bash
make install          # create .venv and install requirements
make fetch            # pull raw data from nba_api (long)
make eda              # the full season-level EDA sweep
make stan             # fit the availability, minutes and component heads
make dashboard        # Streamlit walkthrough at http://localhost:8501
make test             # pytest
```

The Stan heads need a CmdStan toolchain, which pip does not manage:

```bash
.venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Always use `.venv`, never the system Python. Two guards run over the documentation itself:
`make docs-audit` re-derives every quoted figure in the plan docs from its artifact and
fails on a mismatch, and `make dashboard-audit` reports drift between the decision registry
and the docs it distills. See `CLAUDE.md` for conventions.
