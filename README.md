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
- **The four numbers the simulator will consume as direct inputs are calibrated per fit
  window**, because they are *given* to the simulator rather than scored by it: the residual
  copula, the game-level minutes dispersion, the block variance inflation and the bonus
  overdispersion. Their artifacts carry a `fit_window` column and the consumers default to
  `train_val`; the full-window figures move by less than the precision they are quoted at,
  which is why the leak would never have announced itself. **A third window, `train`, was
  added 2026-08-08**, because `train_val` is clean for a *test*-split readout and not for a
  *validation* one: it contains 2022-23 and 2023-24, which is exactly what the realized
  backtest scores against. Which window to consume is decided by what the number will be
  scored against, not by which is widest — and the same rule now governs the persisted
  posteriors, which are namespaced by window (`make posteriors`).

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

**Minutes** in the shipped chain is [stan_minutes.py](src/models/stan_minutes.py): the
marginal `min | available`, fitted season-collapsed as successes out of real game length,
selecting `logit(own) + spline`. What it supplies the simulator is the season-level
**spread** — the one thing the composition head below is structurally unable to produce.

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
**0.1768** for fringe players to **0.0855** for stars. And the composition matches the
marginal head on the season-level **mean**: scored at the season unit on the 742 validation
player-seasons both cover, MAE **200.28** against **200.12** and R² **0.8848** against
**0.8829**, with a bias of **+2.41** against **−14.09**, so it is the less biased of the two.

What survives is the season-level **spread**. Summed composition draws give a season-total
predictive sd of **64.65** minutes against the marginal head's **302.75** — **4.68×** too
narrow, CRPS **170.06** against **144.35** with a paired-bootstrap interval of
**[+18.96, +33.25]** — because draws that are iid across games cannot manufacture
season-level heterogeneity. The sharpest form of it is that at the season unit the
composition does not clear the no-fit carry-forward floor (170.06 against **161.29**) on the
same draws that clear its own per-team-game floor decisively (**4.4945** against 4.6776).
Same head, same posterior, opposite verdicts at two units.

**That gap is a missing parameter, not a ceiling — measured, and it matters for what gets
built next.** A *shared* effect cannot fix it: a season term is a league-wide shift, and
against a head that allocates every minute in the league it has **0.000000%** of the residual
variance to reach. But a **per-(player, season)** effect is not shared, and injecting one
into the existing posterior — `σ·z` per player-season per draw, shared across that player's
games, re-run through the head's own allocation — moves the season-total predictive sd from
64.65 to **239.45** at σ = **0.375** and the CRPS to **142.17**, which *ties* the marginal
head (**−2.18**, interval **[−6.96, +2.85]**) while keeping the team constraint exact — and
at σ = 0.45 the season-unit calibration passes it outright, PIT KS **0.0659** against 0.0735.
MAE barely moves, so it buys spread and not fit.

**The caveat that made that a bound rather than a score is now closed.** σ was read off
validation, which is the split it is scored against — so the same grid was re-run on the last
two *training* seasons (1,145 player-seasons) and its optimum is interior at **σ = 0.450**
(CRPS **117.07** on those rows), one grid step from validation's 0.375 and worth 0.4 CRPS
minutes between them. Two grids on disjoint rows agreeing to a step is the evidence that the
figure was never moved by the evaluation data. At σ = 0.450 the validation reading is CRPS **142.87** against 144.35 — gap
**−1.49**, interval **[−6.14, +3.22]**, a tie — with the better PIT KS of the two and the team
constraint still exact. So the injection is shippable today with a σ that owes the evaluation
rows nothing, and retiring the marginal head is a live prospect rather than a closed one.

**So the injection ships**, as `sim.minutes.player_season_sigma = 0.450`, applied by
`minutes_unification.rehydrate_composition` — a consumer gets the effect by loading the head
rather than by remembering to apply it, and 0.0 recovers the un-injected head exactly.

`composition_glm.stan` also carries the effect as an **optional fitted parameter** (`sigma_u`,
with `U_n = 0` nesting the shipped head exactly), and `make posteriors` persists its scale and
takes precedence over the constant. It is built but **not fitted**: at 12.0× the shipped arm's
cost it did not converge inside the budget, and four sampler-side remedies — centring, sharing
`rho`, dropping `rho`, and a dense metric — all made it worse rather than better, so the cost
is intrinsic to a 2,204-parameter hierarchical posterior rather than a configuration mistake.
`make composition-effects` is the ladder and `docs/potential-to-dos.md` is the write-up. What
a fit would still buy is a σ estimated jointly with the coefficients and a predictive that
integrates over σ's posterior instead of plugging one in.

**Five independent routes agree on the effect size**, which is why a plugged-in σ is
defensible in the meantime: 0.375 (validation CRPS grid), ≈0.41 (calibration — the ratio of
the head's residual sd to its predictive sd), **0.450** (train CRPS grid, the shipped value),
0.4776 and 0.4809 (fitted in Stan, non-centred and centred). Two of those share no arithmetic
with the others.

**And the constraint is not just a cost — it is a dynamic the contest is sensitive to.** A
team's season minutes are a fixed pot, so teammates' season totals are negatively correlated:
a fixed sum over K players forces mean pairwise **r = −1/(K−1)**, which at the measured
**16.05**-player roster size is **−0.0664**. Over **963** single-team validation
player-seasons the composition sits on it at **−0.0509**. The marginal head reads
**−0.0001** and puts a **1,022.9**-minute predictive sd on a team season total that is
physically fixed. That is invisible in every marginal metric and lands on two strategy axes
directly: a same-team stack's minutes are *anti*-correlated rather than independent, and
handcuffing a starter with his backup is a hedge that exists only if the model carries the
sign.

The composition also covers **1,111** validation player-seasons against the marginal head's
742 — the **369** rookies and low-minute players the `≥ 200 prior minutes` filter drops, who
are draftable. So `stan_minutes` ships today because it is the only head with the right
season-level spread today, and the simulator's minutes draw is the open design question
rather than a settled blend.

**Components** ([stan_components.py](src/models/stan_components.py)) fits the seven negative
binomial counts and four beta-binomial conversions, each against a mandatory no-fit floor.
Its head lists come from [component_rates.py](src/models/component_rates.py), which models
total attempts as a count and the three-point mix as a share — see the output contract above.

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
  the composition, the season-level **spread** from the marginal head, and the **2.43×**
  ten-game block variance inflation from `make serial-correlation` for serial dependence
  between games. The **4.65×** game-level figure is a *diagnostic* to check the composition's
  draws against, not an input to them.
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

**And the same head loses to the same comparator at the season unit, which is why both
minutes heads ship.** `make minutes-unification`. Summed to season totals on the 742
validation player-seasons both heads cover, the composition reads CRPS **170.06** against the
marginal head's **144.35** — a paired-bootstrap gap of **+25.70** minutes, interval
**[+18.96, +33.25]** — and does not clear the no-fit carry-forward floor's **161.29** at that
unit. The mean is not what fails: MAE **200.28** against **200.12**, R² **0.8848** against
**0.8829**, bias **+2.41** against **−14.09**. The predictive **spread** is, at **4.68×** too
narrow (sd **64.65** against **302.75**, PIT KS **0.3341** against **0.0735**), because
iid-across-games draws cannot make season-level heterogeneity. **A head is only a model at
the unit it was scored at**, and this is the cleanest demonstration of that in the repo: one
posterior, two units, opposite verdicts against the same two floors. The follow-up measured
in the same target — an injected per-player-season effect closes the gap to a **tie** at
σ = 0.375, while a league-wide season term has **0.000000%** of the residual variance to
reach — is in §2's minutes section, because it changes what gets built rather than what
shipped.

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
season-total sd, against +0.5% from shared coefficient uncertainty.

**The drafting edge is large in the simulated world and the realized readout cannot confirm
it — which is the result, not a caveat.** `make strategy-sweep`. Against a symmetric-field
null, the arm that ships lifts its Round-1 advance probability by **0.2107** in the 600k
Shootaround's simulated worlds and by **0.1268** on the two validation seasons replayed
against realized box scores. The second number is not a smaller version of the first: the
simulated side pools 500 drawn worlds per season and the realized side has exactly one, so
its intervals cover most of the table and it selected nothing. **Gate D fails, and that is
also a result** — the two buy-in tiers do not select materially different rosters in **0**
of **6** paired comparisons, under a tier-blind ranking or a tier-aware objective, so
"draft differently for a bigger field" is not a strategy this simulator can support.

**The edge is not an artifact of a too-simple field, and automation's price is the
objective, not the executor.** `make draft-sim-need` / `make strategy-sweep-need` /
`make strategy-sweep`. Giving the field lineup reasoning is a measured null twice over: a
joint Gate B calibration fits the slot-reaching lean at **0** picks (the observed ADP
curve carries none, degrading fastest in the elite region), and against a *stipulated*
8-pick lean the shipped arm's simulated lift **rises** to **0.306** — slot-reaching pays
value for shape, so the fitted pure-ADP field is the harder opponent and stays shipped.
On the execution axis, submitting our best feasible ranking to DK's own autodraft is
identical to clicking it under DK's 8G/8F/3C caps — and beats the uncapped click by
**+0.0091** (600k, resolved) — while giving up **0.092** of simulated lift against the
shipped per-pick objective, which no static board can express. The 30-second-clock
fallback is safe; the objective is the half worth defending. See
[docs/simulations-plan.md](docs/simulations-plan.md), "The field with lineup reasoning,
and the execution axis".

**The sampler behaved.** 37 component fits with 0 divergences and every fit clearing every
convergence bar, 54 season-term fits with 0
divergences and 0 treedepth saturation, and the availability port reproduces the point MLE
with the MLE inside the 95% credible interval for 24 of 24 terms. Cost is concentrated
entirely in the spline variants. Dropping the test side halved the component fit count from
74 and cut sampler time from 305.0 to **137.4** minutes *while* raising every selection fit
to full-length chains — which incidentally fixed the one fit that used to miss its R̂ bar.
1,536 tests pass (`.venv/bin/python -m pytest tests/`).

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
