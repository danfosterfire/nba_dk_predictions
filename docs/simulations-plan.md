# Simulations Plan: Drafting and Scoring Best-Ball Portfolios

This is a planning doc, not a measurement report. It records direction, not results — update it
in place as pieces get built, the way `availability-plan.md` was.

**Status: specification settled 2026-08-08, nothing built.** The prediction layer is complete
and its heads all ship; this layer is the reason they exist. Before 2026-08-08 this doc named
four things to build and said nothing about the order, the interfaces between them, or the
artifacts — it was the only plan doc in `docs/` a session could not start work from. That
planning session settled six decisions, found one hard prerequisite nobody had noticed, and
put the whole thing on a calendar. All three are recorded below.

## Purpose

Closes the loop `~/Documents/nba_stats` never finished: a draft simulator, a season/tournament
outcome simulator, and a backtesting harness, so draft strategies (ADP-blended ranking,
exposure caps, posterior-sampled diversification, stacking) can be **compared on real historical
seasons** instead of learned live. Consumes `docs/predictions-plan.md`'s posterior draws.

**Unlike every layer before it, this one goes live this year.** The 2026-27 season is a
production run, not a research exercise — see "The deadline" below.

---

## What was settled on 2026-08-08

Six decisions, taken in a planning session against the economics derived in
`dashboard/economics.py` and the artifact survey recorded under "The prerequisite nobody
noticed". Each has a registry entry in `dashboard/decisions.py`.

| # | Decision | Why |
|---|---|---|
| 1 | **Two strategies, two tiers**: 10 entries at $20 (`600k_shootaround`) and 4 entries at $52 (`20k_spin_move`) | Near-equal stake ($200 / $208) across structures whose objectives differ in *shape*, not just scale. Comparing them is itself a result |
| 2 | **The test split is a pure readout** | The strategy is frozen on validation and written to an artifact; the test backtest is run once, reported, and changes **nothing** — not the strategy, not the stake, not the entry decision |
| 3 | **Reactive live-pick is the primary draft mode** | Fewer, higher-conviction entries drafted manually. Ranking-submission is the fallback for any fast draft that outruns the clock. This resolves a contradiction the doc used to carry in two places |
| 4 | **Draft state arrives through a local draft-room UI** | A Streamlit page over the precomputed board, one click per pick. Works at a 30-second clock and needs no external access. Reading the DK page is not on the critical path |
| 5 | **The in-draft objective is payout-weighted EV over the full bracket** | Simulate all four rounds against an ADP field and pick the player maximizing expected payout. It is the actual money objective, and only modestly more work once the bracket sim exists |
| 6 | **Strategy tuning runs on simulated truth with model-error injection** | The only surface with enough resolution to rank strategies — but the injection is mandatory, for the reason in "Why an uninjected simulated backtest cannot price ADP" |

---

## The deadline, and the ordering it forces

The DraftKings 2026-27 board was open and carrying ADP on **2026-07-28**, roughly 12 weeks
before the season. Drafts happen before October. That is the binding constraint on everything
below, and it inverts the natural build order: a **defensible draftable board and a working
draft room must exist before the full strategy sweep does**, not fall out of it.

Two production inputs were checked on 2026-08-08:

- ✅ **2026-27 rosters are live.** `commonteamroster` returns them (19 players for BOS on the
  probe, with `POSITION` null for unsigned/two-way players — the DK board covers that gap).
- ❌ **The 2026-27 regular-season schedule is not published.** `ScheduleLeagueV2` returns 20
  rows for 2026-27 — 19 preseason games plus the 12/11/2026 NBA Cup final that
  `docs/dk_best_ball_rules.md` says does not score — against 1,400 rows for 2025-26. The
  schedule is normally released in mid-August. **Poll for it**; the simulator cannot run a
  production season without per-team game dates.

Because the schedule is not yet available and the backtest seasons are, **every backtest piece
can be built now and the production run is schedule-gated at the very end.** That is the
ordering the to-do list follows.

---

## The prerequisite nobody noticed: no head persists its posterior ✅ built 2026-08-08

`make stan` fits eleven-plus heads and writes metrics, diagnostics and per-row predictions.
**It writes no coefficient draws.** The single exception is `stan_composition`'s crash
checkpoint, which pickles `alpha_draws` / `beta_draws` / `rho_draws` / `scaler` / `features`
per arm — incidentally, to survive a 9.9-hour loop, not as a consumable artifact.

So "simulate from the joint posterior" used to mean *refit*: ~137 minutes for the component
heads and ~2.7 hours for one composition arm, every run. That is not a thing a draft room can
do, and it is not a thing a strategy sweep can do a hundred times.

**Step zero of this layer is therefore a posterior artifact**, and it is a small module rather
than a research question. `make posteriors` (`src/models/posteriors.py`) writes one file per
head to `data/features/posteriors/<head>.pkl` plus a `manifest.csv`, carrying:

- **thinned coefficient draws** (`alpha_draws`, `beta_draws`, and whichever dispersion the
  family carries — `rho_draws`, `phi_draws`, `kappa_draws`), 1,000 draws by default, thinned
  across the whole posterior via `stan_utils.thin` — never sliced off the front, for the reason
  `season_terms._draw_components` already records;
- the **design recipe** needed to score an arbitrary frame: an ordered list of *fitted* steps
  (imputation means, `log1p` / `logit` transforms, the fitted `SplineTransformer` and its
  knots, the dispersion bin edges), the feature list, the fitted `StandardScaler`, and the
  selected variant name;
- **provenance**: the fit window (`train`, `train_val`, or `full`), the CmdStan version, the
  git SHA, the timestamp, and the sampler budget.

`stan_composition._checkpoint` was the precedent for the mechanics; this generalizes it to
every head, promotes it from a crash artifact to a contract, and gives it a loader.
**`make posteriors` is now the only thing in this layer that needs a CmdStan toolchain, and
everything downstream is numpy.**

**Eighteen heads:** availability, minutes, the seven counts, the four conversions, the
composition, and the games-played process's four (`gp_entry`, `gp_exit`, `gp_onset`,
`gp_duration`). Each is refitted once at the variant its own sweep selected — read from that
sweep's artifact rather than re-decided, the rule `src/final_evaluation.py` follows.
**Twenty since 2026-08-09**, when the game-length head added `game_length_ot` and
`game_length_depth` — they cost 0.3 s between them and go first in the group order, so a
failure in the plumbing surfaces before any expensive head runs.

### The recipe is captured from the real ladder, and verified against it every run

Each head's variant ladder lives in its own module. `posteriors.py` **calls those functions**
for the transformed frames and the feature list and only expresses the fitted state they leave
implicit; every spline is refitted from the ladder's own returned training frame, which still
carries the pre-spline column. What is unavoidably duplicated is the *order* of the steps,
which is how a recipe drifts silently — so **every artifact is verified at build time**: the
recipe applied to raw probe rows must reproduce the head's own standardized design matrix and
the head's own predictions. A ladder that changes shape fails the build.

That build-time check is the gate this item was given, and it holds exactly: the recipe path
and the head path reproduce each other **bit for bit** (max design error and max prediction
error both `0.00e+00` over 400 probe rows), against bars of 1e-9 and 1e-8. Not approximately —
the two paths do the same arithmetic on the same doubles, so anything above floating-point
noise would be a real defect rather than a tolerance question.

**The reference is captured at fit time, not read from `outputs/predictions/`, and that is
forced rather than convenient.** Only two heads persist per-row predictions at all, and both
were produced by a *different fit* — `make stan` fits on `train` and scores validation, while
these artifacts fit on `train_val`. Comparing against those CSVs would measure the change of
training window, not the fidelity of the recipe. So each artifact stores its own probe: 400
raw validation rows spanning the frame (`thin`, applied to rows), the head's own standardized
design matrix on them, and the head's own reported mean. The round-trip then rebuilds all
three from the pickle alone, and the artifact is self-verifying forever after — a consumer can
call `roundtrip()` to find out whether the thing on disk still does what it claims.

**It also earned its keep immediately.** The first version derived which columns had been
imputed from the head's *feature list*, which is right for the count and conversion ladders and
wrong for the composition one: that ladder discards `impute`'s per-column flags in favour of a
single `design_missing` indicator, because a rookie loses all 18 design columns at once and 18
identical flags are a degenerate subspace the sampler pays for in treedepth. So the feature
list carried no `__miss` column, the recipe was built with an empty imputation step, and every
rookie row would have gone into the simulator as `NaN` → 0 after standardization. The columns
were all present and every shape was right. Only the numbers were wrong, which is the failure
mode this check exists for; the flags are now read off the ladder's own transformed frame.

### One window per consumer — the default reversed from `train_val` to `train`

This doc's config block originally specified `fit_window: train_val`, on the grounds that it
matches the four simulator inputs already calibrated that way. **That was wrong and the
default is now `train`.** The analogy does not transfer. Those four — the residual copula,
the game-level minutes dispersion, the block variance inflation and the bonus overdispersion
— are *given* to the simulator and never scored against realized data; they set the shape of
the noise. The posterior coefficients generate the board, and "Realized truth — the honest
readout" below replays portfolios drafted from that board against **real 2022-23 and 2023-24
box scores**. Those are the validation seasons, and at `train_val` every one of those rows is
in the fit: **883 of 10,361** availability rows (8.5%) and **773 of 9,403** component rows
(8.2%). Per row the leverage on a ~12-parameter GLM is tiny; as a class it is exactly the
shape of the four sub-1% reversals already logged here, and the realized readout is the one
figure in this layer whose whole job is to be believable.

**A second, independent argument points the same way.** Gate C calibrates the error injection
to the model's *measured* out-of-sample miss — availability CRPS 10.006 games, component R²
0.81–0.95 against the no-fit floors, season-total MAE 400.5 dk_pts. Every one of those figures
was measured by scoring **train-fitted** heads on validation. If the simulator draws its truth
from `train_val` heads, the injected error describes a differently-fitted model from the one
generating the world, and the sweep's `α` is being priced against a mismatch. `train` keeps the
generating model and the injection target the same object.

`train_val` is not discarded — it moves to the consumer it actually fits. All three windows
are wanted at once, so artifacts are **namespaced by window**,
`data/features/posteriors/<window>/<head>.pkl`:

| window | consumer | why that one |
|---|---|---|
| **`train`** (default) | the realized 2022-23 / 2023-24 backtest and the strategy sweep | the only window that has not seen the seasons being scored |
| `train_val` | the one-shot test readout on 2024-25 / 2025-26 (build item 10) | a held-out figure should describe the model that would actually deploy — the rule `src/final_evaluation.py` already follows |
| `full` | the 2026-27 production board (build item 9) | reads the held-out seasons, so it goes through `held_out.assert_unlocked` |

The flat layout this replaced let a second window silently overwrite the first, which is the
opposite of what "fitting one posterior per fit window is what makes it affordable to widen
the realized backtest later" asks for.

**The four simulator inputs now emit the same third window** — ✅ closed 2026-08-08, the same
day it was opened. Moving the coefficients to `train` did *not* on its own make the realized
backtest clean: `stan_minutes_dispersion.csv`, `serial_correlation.csv`,
`residual_correlation.csv` and `bonus_calibration.csv` carried only `full` and `train_val`
rows, so a 2022-23 / 2023-24 backtest would have held clean coefficients alongside a noise
*shape* calibrated on the very seasons it was scoring. Second-order — those four are variance
and dependence parameters rather than means, which is the original argument for calibrating
them wide — but it capped how clean the readout could honestly be called.

The fix was one shared helper, which is why it was worth doing immediately rather than
deferring to `src/sim/season.py`. `src/data/preprocess.py` owns `FIT_WINDOWS`, and three of the
four modules already looped over it, so adding `TRAIN_WINDOW` made them emit the third row
without touching them. `fit_window` drops **twice** `TEST_SEASONS` for `train` — the same
two-step `held_out.selection_split` performs, expressed as one count — and `held_out_seasons`
now reports per window so each module can print what it actually excluded. Only
`stan_minutes.run` had the pair hard-coded and needed changing.

Three windows, one per consumer, and the rule is the same everywhere: **which window to consume
is decided by what the number will be scored against, not by which is widest.**

**The head zoo is less uniform than it looks, and the build found that out the hard way.**
The spell-duration head (`BetaGeometricHead`) differs from the other seventeen in three
separate places: it standardizes *inline* inside `shapes` and exposes no `_design`; it has no
`predict_mean`, `predict_p` or `mu_draws`, so its mean exists only as `a / (a + b)` from the
per-draw shapes; and that path runs through `games_played.beta_shapes`, which **clips** `mu`
— so the clip is part of the reported mean rather than a guard on the way to it. The first
run raised on the missing `_design` rather than writing a plausible artifact, which is the
behaviour wanted, and the fix stores the clip bounds explicitly. One honest caveat follows:
for that head alone the *design* half of the round-trip is written out here rather than taken
from the head, so it is weaker than the other seventeen. Its *prediction* half still goes
through `shapes` and `beta_shapes`, which is the half that would catch a wrong scaler anyway.
Both its arms — with covariates and intercept-only, where `features = []` leaves no scaler at
all — are pinned by tests.

Three things the artifact records that are easy to leave out:

- **which mean the head reports.** The three beta-binomial heads inside the games-played
  process report a *plug-in* mean (`sigmoid` at the posterior-mean coefficients) while every
  other beta-binomial head reports the posterior mean of `mu`. Storing which one applies is
  the difference between a round-trip that checks something and one that checks the loader
  against itself; a test pins that the two differ by more than the tolerance.
- **the fit window, enforceable.** `require_window(artifacts, window)` lets a consumer refuse
  the wrong one. The leak it prevents is quiet in both directions: a backtest scoring
  *validation* with `train_val` heads has read 2022-23 and 2023-24 through the coefficients,
  and one scoring *test* with `full` heads has read 2024-25 and 2025-26 the same way. No
  split guard can see either, because the guards sit on frames rather than on parameters.
  `--window full` itself goes through `held_out.assert_unlocked`.
- **the year random effect is refused, not silently dropped.** No head in `make stan` enables
  one, and `YearTerm.shift` is a fresh `N(0, 1)` per posterior draw shared across rows rather
  than a stored coefficient — so an artifact cannot carry it as written. The builder raises
  rather than writing a head whose predictive is quietly narrower than the fitted one.

**This also unblocks the walk-forward question.** Fitting one posterior per fit window is what
makes it affordable to widen the realized backtest later without re-deciding anything.

**Cost, measured on the `train_val` set.** **8.46 h** of sampler time for all eighteen heads
at full-length chains, of which the composition head alone is **301.2 min** on 683,453 rows —
so the other seventeen come to 3.44 h between them. Every head converged (max R̂ **1.0083**,
**0 divergences** anywhere) and every one round-trips, worst prediction error **1.17e-15**
against a 1e-8 bar.

That 36:1 cost ratio between one head and the other seventeen is why each artifact is written
the moment it is built and the manifest merges by head: `--groups components` re-does one
family without touching the rest, and a crash in the expensive head costs one head rather than
the run. It is also why the two windows can be built in parallel on separate `--groups` — the
manifest re-reads at every flush rather than snapshotting at start, so concurrent runs take
the union instead of the last writer clobbering the first.

**Loading is free, which is the whole point.** All eighteen artifacts load in **0.02 s** with
`cmdstanpy` made unimportable, and `require_window` refuses a set fitted at the wrong window.

---

## The second prerequisite: game length is a random variable forward, not a lookup ✅ built 2026-08-09

Raised 2026-08-08, after the build order was first written. **Every backtest so far has read
`game_length` from `data/features/game_length.parquet`, because in a replay the games already
happened.** In a forward simulation — the production 2026-27 run, and every simulated-truth
season the strategy sweep draws — nothing knows how long a game will be. Both minutes heads
need it: the composition head allocates exactly `5 × game_length` minutes per team-game, and
the marginal head uses `game_length` as its binomial trials. So the simulator needs a
game-length draw, and it belongs upstream of everything else in the chain.

### What already exists, and why it is in the wrong place and the wrong form

`stan_composition.py` (lines 1074–1096) already carries `fit_ot_tail` and
`sample_game_length`: a **two-parameter point-MLE geometric tail**, fitted on train seasons
and checked against held-out ones, reading `p_any_ot = 0.0608` and `p_more_ot = 0.1408`. It
works — its held-out PPC reads 2,322 observed regulation games against 2,310.5 predicted, 120
against 128.4 at 1OT, and 18 against 18.1 at 2OT.

Three things are wrong with leaving it there:

- **It is a point estimate.** Every other simulator input is a posterior, and `make posteriors`
  now covers eighteen heads. A hardcoded pair of floats is the one input that does not go
  through the artifact contract.
- **It lives in the wrong module.** The simulator would have to import `stan_composition` — a
  9.9-hour head — to get a game-length draw.
- **It is unconditional, and there is a real season trend it cannot see.** Measured on 35,546
  regular-season games: the logit slope is **−0.00893 per season** (se 0.00264, z = **−3.38**,
  p = 7.3e-4), taking the fitted rate from **0.0670** in 1996-97 to **0.0526** in 2025-26 and
  extrapolating to **0.0521** for 2026-27. Pooled over the last five seasons the rate is
  **0.0504** against **0.0613** over the first twenty-five. **The train window's 0.0608
  therefore overstates the 2026-27 rate by about 17% relative.**

### What the data says about the shape — the geometric is right, and two parameters are enough

Continuation probability by depth over all 35,546 regular-season games: **0.1382** at
1OT→2OT, **0.1438** at 2OT→3OT, **0.1190** at 3OT→4OT. Essentially constant, so a geometric
in depth covers 2OT, 3OT and 4OT for free and there is no depth dependence to model. Depth
counts are 33,433 / 1,821 / 250 / 37 / 5 and nothing exceeds 4OT.

Season-to-season dispersion in the OT rate is only **1.11×** binomial, so the era movement is
a **trend rather than a wander** — precisely the distinction `src/eda/season_effects.py` exists
to draw. That matters because this project's recorded position is that no head ships a season
term; this would be the second exception after the minutes head, and unlike the marginal cases
the evidence is a clean monotone trend against near-binomial season noise.

### Size it honestly before building it

Expected extra minutes per game is `5 × p / (1 − 0.1382)`: **0.345** min at the pooled rate,
**0.302** at the trend-extrapolated 2026-27 rate. That is **0.72%** against **0.63%** of
regulation — so getting the trend right is worth **0.09% of total minutes**. Nobody should
build this expecting a mean effect.

**The reason to build it properly is the tail, not the mean.** Overtime is where 40+ minute
games come from: 1,650 player-games exceed 48 minutes and the observed maximum is 63.0. Under
a best-ball weekly max and a threshold bonus, a star's ceiling week is the thing that decides
a 2-of-12 pod, and an OT frequency 17% too high inflates every star's simulated ceiling. That
is a shape error in exactly the statistic the tournament objective is most sensitive to.

### The design — a Stan head with a full posterior, and **no new `.stan` file**

Two existing sources already cover it, which is the same factorization argument the rest of
the project runs on:

| Piece | Likelihood | Source | Collapses to |
|---|---|---|---|
| does a game go to OT | beta-binomial | `betabinomial_glm.stan` | OT games out of games, per season cell |
| how many overtimes | beta-geometric | `betageometric_duration.stan` | one row per OT game's depth |

The beta-binomial's overdispersion parameter **is** the trend-versus-wander answer, measured
rather than asserted, and the beta-geometric's Beta frailty is the natural fix for the one
place the plain geometric misses — it over-predicts 3OT+ by 3 games in 2,460. It is the same
device the absence-spell process already uses one level down.

Prediction-time-legal covariates, in ladder order against a mandatory no-fit floor (the league
constant rate, carried forward):

1. **floor** — pooled league rate, no fitting.
2. **season trend** — the measured −0.00893 logit slope. Expected to win.
3. **+ matchup** — `|prior-season net rating difference|` between the two scheduled teams,
   since evenly matched teams are likelier to be tied at the buzzer. Known before the season
   given the schedule, and available from `team_estimated_metrics_*.csv`. Speculative; expect
   a null and record it as one.

**This is the cheapest head in the project by a wide margin** — roughly 30 collapsed rows and
2–4 parameters against the composition head's 631k rows. Seconds of sampler time. That is an
argument for doing it properly rather than for expanding its scope: it must not become a
research project, and the ladder above is the whole of it.

### What was built, and what it measured — 2026-08-09

`src/models/stan_game_length.py`, `make stan-game-length`. **Four Stan fits, 0.3 s of
sampler time, 0 divergences, max R̂ 1.0048** — the cheapest head in the project by three
orders of magnitude, as sized. No new `.stan` file: `betabinomial_glm.stan` on 26 season
cells and `betageometric_duration.stan` on four collapsed depth rows. The head classes were
reused too, not rewritten — `stan_games_played.BetaBinomialHead` and `BetaGeometricHead` are
these likelihoods with different data.

**Both gates pass, and only one of them is close.**

| arm | fit games | onset nats/game | complete nats/game | pred. OT rate | Σ\|obs − pred\| over the 4 OT classes |
|---|---|---|---|---|---|
| `floor` (the incumbent) | 30,626 | −0.216288 | −0.239145 | 0.06077 | **22.97** |
| **`season_trend`** ✅ | 30,626 | **−0.216068** | **−0.239122** | 0.05545 | **9.42** |
| `season_trend_covered` | 8,289 | −0.216384 | −0.239438 | 0.05003 | 35.48 |
| `season_trend_matchup` | 8,289 | −0.216655 | −0.239708 | 0.04940 | 38.49 |

Observed on the 2,460 validation games is an OT rate of **0.056098**. The class counts are
2,322 / 120 / 18 / 0 against the incumbent's 2,310.5 / 128.4 / 18.1 / 2.96 and the trend
arm's **2,323.6 / 117.4 / 15.8 / 3.1** — so the head is **2.4× closer** on the statistic the
gate names, and the whole of that comes from the OT *rate* rather than from the depth shape.

**The log-likelihood gate passes on a margin that is not a result.** +0.000023 nats per game
on the complete model, 95% paired bootstrap **[−0.000918, +0.000908]**. That is honest and it
is expected: 94% of games are regulation, and moving `p` from 0.0608 to 0.0555 against a
truth of 0.0561 is worth almost nothing per game. The counts are where the win is, which is
why the plan specified them as the gate. Selection reads the **onset half** (+0.000219), since
every fitted arm shares one depth head and the arms differ in nothing else.

**Three findings the plan did not anticipate.**

- **The matchup arm is a null, as predicted, but the reason is not the one expected.**
  `|prior-season net rating difference|` fits a coefficient of **−0.0119** per rating point —
  the *right sign*, since evenly matched teams should be likelier to be tied — and still
  loses to its own same-window control by **−0.000270** nats per game. `team_estimated_metrics_*.csv`
  starts at 2014-15, so the arm can only fit 8,289 of 30,626 training games, and the
  `season_trend_covered` control exists to separate "the covariate is worthless" from "seven
  seasons is not enough to fit a trend on". The control answers that plainly: fitted on the
  short window the *trend itself* degrades from −0.00701 to −0.02901 logit per season and the
  class error triples to 35.48. **The covariate costs a little; the window costs a lot.**
- **The plan's stated reason for the Beta frailty is backwards, and it is corrected here.**
  "It over-predicts 3OT+ by 3 games in 2,460" describes a *validation* over-prediction, and a
  frailty puts **more** mass in the tail, not less — the beta-geometric predicts 3.1 there
  against the plain geometric's 3.0, so it is marginally worse at exactly the miss it was
  motivated by. What the frailty actually fixes is the opposite miss on the fitting half,
  where the geometric **under**-predicts 3OT: 37 observed against **31.7** geometric and
  **34.5** beta-geometric on 1,861 overtime games. On validation the plain geometric is ahead
  by 0.00350 nats per overtime game over 138 of them — about one 2OT game's worth of evidence,
  so the two are not distinguishable there. **The frailty ships anyway, on a reason that is
  not fit**: it *nests* the geometric (κ → ∞, fitted at **κ = 37.7**) and it is the only form
  of the depth model that carries a posterior, which is the entire point of moving this out of
  `fit_ot_tail`.
- **The trend-versus-wander answer is a trend, but `rho` is weakly identified and must not be
  quoted as a measurement.** With the slope in, the residual season dispersion is
  **2.49e-4 [1.66e-5, 7.14e-4]**, i.e. 1.22× binomial at the posterior median — but on 26
  cells, against the uniform prior `betabinomial_glm.stan` deliberately puts on `rho`, that
  posterior leans upward. A Pearson dispersion around the fitted trend reads **0.91**, i.e.
  *under*-dispersed. Read the fitted `rho` as an upper bound on the wander, not as a
  measurement of it. The direction of the conclusion is unaffected: the era movement is in the
  slope, which extrapolates, and not in a residual spread, which does not.

**The fitted slope is −0.00701 logit per season, not −0.00893.** The plan's figure is the
full-window fit over 30 seasons; this one is fitted on `train` (1996-97 → 2021-22) as the
split requires. Extrapolated, this fit reads **0.0542** for 2026-27 against the floor's
0.0608 — the same ~11% relative overstatement the plan flagged at 17% off the wider window.
The production board refits at the `full` window through `make posteriors`.

**The draw is checked as a draw, not only as a set of probabilities.** `simulated_slate`
redraws the whole validation season 500 times *through `sample_game_length` itself*, one
posterior draw per slate, and the sampled class counts reproduce the analytic ones
(2,324.2 / 117.0 / 15.7 / 3.0 against 2,323.6 / 117.4 / 15.8 / 3.1). That is the only check on
the sampler: a length drawn per team-game, a frailty applied per game instead of per season, or
a wrong grid step would leave every probability in the module correct and every simulated
season wrong. The 90% interval on the validation OT count, *with* the season frailty, is
**[111, 162]** against 138 observed.

**Artifacts.** `outputs/predictions/stan_game_length_{metrics,ppc,depth,diagnostics}.csv`,
plus `game_length_ot.pkl` / `game_length_depth.pkl` under
`data/features/posteriors/<window>/` — **twenty heads** now, not eighteen. Both round-trip at
0.00e+00 design error and 2.22e-16 prediction error.

### What moved out of `stan_composition`

`fit_ot_tail`, `sample_game_length` and `ot_tail_check` are **deleted**, and
`stan_composition_ot_tail.csv` with them; `src/docs_audit.py`'s five claims on those figures
re-derive from the new head's artifacts instead, and the values did not move — the floor
reads p_any **0.0608** and p_more **0.1408** on the same 30,626 games, because it is the same
function on the same rows. Registered as `withdrawn`.

**One clause of the specification was not followed, deliberately.** The plan says
`stan_composition` should afterwards *import* the new head. It does not. Nothing in that
module calls a game-length draw — it consumes the **realized** `game_length` column on every
row it fits or scores, and only a forward simulation needs the draw — so the import would be
dead code, and re-emitting the OT artifact from a 9.9-hour target rather than a 7-second one
is worse provenance, not better. `stan-game-length` still runs ahead of `stan-composition` in
the `stan` aggregate, for the ordinary reason: cheapest first, so a plumbing failure surfaces
in seconds.

---

## The third prerequisite: is the marginal minutes head still needed? ✅ settled 2026-08-09

**Yes. `make minutes-unification` ran the gate and the composition lost at the season unit,
so both heads ship — but not for the reason the sentence being tested gave.** The audit
killed one of its three claims and the gate killed a second; what survives is the third,
and it is not the one anybody would have bet on. The measured verdict is under "What the
gate found" below, and everything above it is the reasoning as it stood on 2026-08-08.

Raised 2026-08-08. `README.md` said the two minutes heads "compose rather than compete" —
`stan_composition` owning the per-game allocation and `stan_minutes` still owning "the
season-level mean and the game-level dispersion, neither of which the composition produces."
**Checked against the code, that sentence asserts three things and only one of them holds.**

### What `stan_minutes` actually exports, and who consumes it

Production consumers are two: `posteriors.py` persists a `StanMinutes` fit as one of the
eighteen heads, so it sits inside the simulator's artifact contract; and `season_terms.py`
runs the year-effect ablation on it. Everything else is measurement or presentation —
`stan_composition.independent_comparator` refits it as the control its headline −6.06% is
measured against, plus `dashboard/tabs/minutes.py` and `src/docs_audit.py`.

### Claim 1 — "owns the game-level dispersion". **False as stated.**

`stan_minutes.game_level_dispersion` (line 368) takes `targets` and `lengths`, computes each
player-season's own realized share as `mu`, and fits a dispersion to that. **The `StanMinutes`
object never appears.** Every Stan fit in the file could be deleted and it would still return
4.65× — it is a data measurement that happens to live in that module.

It is also the wrong number for the simulator if minutes are drawn from the composition. The
selected composition arm fits its dispersion **role-graded over four bins** — rho **0.1768**
for fringe players down to **0.0855** for stars, a spread already credited with cutting
calibration error 35% — against a single pooled figure whose own docstring calls it
"in-sample against each player-season's own mean; a floor". The two sit on different
parameterizations (the composition's rho disperses its sequential binomial *trials*, not
`game_length`), so they are not the same number — but they are the same kind of quantity and
**only one of them can govern the draw.**

### Claim 2 — "owns the season-level mean". **Untested, and that is the whole gate.**

The heads score at different units, which is why their metrics tables do not look comparable:
minutes at season-total (CRPS 143.9, MAE 199.6, R² 0.883), composition per-team-game (CRPS
4.494, MAE 6.33, R² 0.474). The composition's per-game predictions **sum to a season total by
construction**, so it can produce the season mean. Whether it is better or worse at that unit
has never been measured.

> Measured 2026-08-09, and **this claim is false too**: the composition matches the marginal
> head on the season-level mean and is the *less biased* of the two. What it cannot produce
> is the season-level **spread** — which is a different quantity, was never what the sentence
> said, and is the only reason the marginal head survives. See "What the gate found".

### Claim 3 — the year effect. **Genuinely load-bearing, and the reason not to retire it yet.**

`season_terms` selected the `year` arm for `min`: val MAE **199.03** against the base arm's
199.72, `sigma_year` **0.0231**. It is the only head in the project that ships a season term,
and a year effect is worth **+10.4%** on a 15-man roster's season-total sd — which matters
here more than anywhere else, because roster-level spread is what decides a 2-of-12 knockout.
**The composition cannot carry one today**: `composition_glm.stan` has no `S` / `year_z` /
`sigma_year` block, unlike `betabinomial_glm.stan` where `S = 0` disables it exactly. Retiring
`stan_minutes` right now would drop the project's only era correction.

> **Never became load-bearing, because the gate settled it first.** The year block was to be
> ported into `composition_glm.stan` only if the composition won; it did not, `stan_minutes`
> ships, and its `year` arm ships with it unchanged. `composition_glm.stan` still carries no
> `S` / `year_z` / `sigma_year` block and does not need one — which also saves the
> season-terms ablation re-run, a refit of the project's most expensive head.

### Two things that cut toward the composition, and one that is a non-argument

- **Coverage favours the composition.** Its own docstring: a rookie cannot be dropped because
  the sum must be complete, so no-prior players get an expanding-window draft-bucket share
  prior instead of the `>= 200 prior minutes` filter every other head applies. Broader
  coverage — and rookies are draftable, which matters for this layer specifically.
- **Cost is a non-argument either way.** `make stan-minutes` is **0.341 h** across its four
  arms against the composition's **9.92 h**, so retiring the target saves twenty minutes. The
  composition refits a minutes model internally as `comparator/val` (358 s) regardless, so the
  code stays even if the target goes.

### The gate, and what each outcome means

Score the composition's season-total sums against the minutes head's season-total predictions
on the same validation rows, same metric.

- **Composition wins or ties** → port the optional year block into `composition_glm.stan` by
  copying the `S = 0` device from `betabinomial_glm.stan`, re-run the season-terms ablation
  for it, and retire `stan_minutes` from the production chain — keeping it as a **frozen
  comparator artifact** so the −6.06% claim stays reproducible without a refit.
- **Composition loses at the season unit** → keep both, and replace `README.md`'s "compose
  rather than compete" with the measured reason.

**Two changes follow regardless of the outcome**, and they are already reflected above: the
simulator draws minutes from the composition, and the 4.65× moves from *simulator input* to
*diagnostic* — a number the composition's draws are checked against rather than one they are
built from.

### What the gate found ✅ `make minutes-unification`, 2026-08-09

> **Rendered since 2026-08-10** on the dashboard's Minutes page (`make dashboard`,
> `/minutes`), which draws this table, the sigma sweep below it and the teammate-coupling
> table as three named blocks — the two-unit reversal as a ratio to each unit's own no-fit
> floor, since 4.4945 CRPS minutes per player-game and 170.06 per season cannot share an
> axis. See [dashboard-plan.md](dashboard-plan.md#step-5b-as-built--the-minutes-page).

**The composition loses, so both heads ship — and every part of the reasoning above about
*why* was wrong.** Scored at the season unit on the **742** validation player-seasons both
heads cover, at the `train` fit window, 1,000 posterior draws each:

| arm | CRPS | MAE | R² | bias | PIT KS | predictive sd |
|---|---|---|---|---|---|---|
| `minutes_head` | **144.35** | 200.12 | 0.8829 | −14.09 | 0.0735 | **302.75** |
| `composition_sum` | 170.06 | 200.28 | 0.8848 | +2.41 | 0.3341 | **64.65** |
| `carry_forward` (no-fit floor) | 161.29 | 213.65 | 0.8532 | +24.14 | 0.1242 | 330.84 |
| `composition_sum_all_rows` | 149.72 | 175.97 | 0.9156 | −0.00 | 0.3575 | 55.86 |

Paired bootstrap over the 742 rows, composition − minutes: **+25.70** CRPS minutes, 95%
interval **[+18.96, +33.25]**, P(Δ<0) = **0.0%**. Not a margin this repo cannot resolve —
the four sub-1% reversals already logged here are the reason that sentence needs saying.

**The mean was never the problem. The spread is.** MAE 200.28 against 200.12 and R² 0.8848
against 0.8829 are a tie, and the composition is the *less biased* of the two (+2.41 against
−14.09). Its season-total predictive sd is **64.65** minutes against **302.75** — **4.68×**
too narrow — and its PIT KS is **0.3341** against **0.0735**. Summing draws that are iid
across games cannot manufacture season-level heterogeneity: per-game noise averages down by
~1/√G while a season-level multiplier passes through in full. That is the same identity
`stan_minutes`' docstring already states from the other direction, arriving here as a
capability gap rather than as a caveat.

**A *shared* dispersion knob cannot fix it, because the constraint forbids what it would
buy.** A team's season minutes are fixed at `5 × Σ game_length`, so summing a team's players
over a season is a *constant* across draws — measured as a predictive sd of **0.00** minutes.
The head cannot give every player on a roster a correlated "he played more than expected this
year" season, because the minutes have to come from a teammate. Its aggregate bias of
−0.00 over the complete 1,111-row player set is the same fact wearing a different hat: the
league's minutes are fully allocated, so summing every unit recovers the total exactly, by
construction and not by skill.

> **Read that precisely: it rules out a *shared* effect, not every effect.** An earlier draft
> of this section said the constraint "forbids the fix" without the qualifier, which is wrong
> and was corrected the same day. A per-**(player, season)** effect is not shared, and
> "Could a season term fix the composition instead?" below measures what it buys: the
> composition ties the marginal head. The distinction is the whole result.

**The sharpest form of the result is the floor comparison.** At the season unit the
composition does not clear the no-fit carry-forward floor — 170.06 against **161.29** — on
the same draws that clear its own per-team-game floor decisively (4.4945 against 4.6776).
Same head, same posterior, opposite verdicts at two units. Per `CLAUDE.md` a head that does
not clear its floor is not a model; the honest reading is narrower and more useful: **the
unit is part of the claim**, and a head is only a model at the unit it was fitted and scored
at.

Two things that cut the composition's way and do not change the verdict. It covers **1,111**
validation player-seasons against the marginal head's 742 — the **369** rookies and
low-minute players the `≥ 200 prior minutes & ≥ 10 games` filter drops, who are draftable —
and it scores better on that wider set than either head does on the narrow one (CRPS 149.72,
R² 0.9156), because the extra rows are low-minute players who are easier to predict. That is
a coverage advantage, not a win, and it is reported as its own row rather than folded into
the comparison.

Two guards on the measurement itself, both reported in the artifact. The two frames disagree
about realized season minutes by a mean of **1.78** and a max of **8** minutes, because the
marginal head builds from `component_targets.parquet` and the composition from
`availability_panel.parquet`; scoring both arms against a single shared target moves the gap
from +25.70 to **+25.72**, so that is not what carries it. And nothing here was refitted —
both heads are rehydrated around their persisted `make posteriors` draws and score through
their **own** `predict_samples`, so the gate costs seconds rather than the composition's
9.92 h and neither arm is a differently-fitted model from the one it is compared against.

### Could a season term fix the composition instead? **No — measured, and it is not close.**

The obvious follow-up, since the minutes head is the one place in this project that ships a
season term and the composition carries none. `make minutes-unification` answers it from the
head's own residuals without fitting anything, and the answer is no on three independent
counts.

**A year term has no variance to reach.** A trend or a year random effect is a *league-wide
shift shared by every row in a posterior draw* (`stan_utils.YearTerm`), so the only variance
it can explain is that of the league-wide mean residual across seasons. For a head that
allocates every minute in the league that quantity is **0.000** minutes against a residual sd
of **243.50** — **0.000000%** of the variance to be explained. That is zero by construction
rather than by accident: the residuals sum to zero within each season because the minutes are
fully allocated, so there is no league-wide level for a league-wide term to move.

**The constraint blocks it a second way.** Even granting a shift, adding the same `delta` to
every player's η in a team-game re-tilts the stick-breaking allocation toward the top of the
rotation and leaves the team total at exactly `5 × Σ game_length`. A shared shift is
*definitionally* a re-allocation here, and re-allocation is not spread.

**And it is the wrong size by an order of magnitude.** The deviation that needs modelling —
realized against predicted season minutes, per player-season, on the log scale — has an sd of
**0.2836** over the 867 validation rows clearing 200 realized minutes. The minutes head's
fitted `sigma_year` is **0.0231**. Even if a league-wide term could apply, it is roughly
**12×** too small to be the object in question.

### The effect that *does* work is indexed by (player, season) — and it ties

Measured in the same target, and it is not a small effect. Inject `σ·z[unit, draw]` into the
linear predictor — one standard normal per player-season per posterior draw, shared across
that player's games — and re-run the head's **own** sequential allocation. That is exactly
the predictive a fitted random effect produces once integrated over, the way `YearTerm`
already treats an unobserved future level. Scored on the same 742 gate rows:

| σ | CRPS | Δ vs marginal | 95% interval | predictive sd | MAE | PIT KS | verdict |
|---|---|---|---|---|---|---|---|
| 0.000 | 170.06 | +25.70 | [+18.96, +33.25] | 64.65 | 200.28 | 0.3341 | loses |
| 0.200 | 149.79 | +5.44 | [−0.17, +11.79] | 141.67 | 199.99 | 0.1872 | ties |
| 0.300 | 143.46 | −0.89 | [−5.97, +4.68] | 197.65 | 199.98 | 0.1237 | ties |
| **0.375** | **142.17** | **−2.18** | **[−6.96, +2.85]** | **239.45** | 200.38 | 0.0808 | **ties** |
| 0.450 | 142.87 | −1.49 | [−6.14, +3.22] | 280.87 | 200.51 | **0.0659** | ties |
| 0.600 | 148.92 | +4.57 | [+0.15, +9.18] | 359.57 | 201.35 | 0.1251 | loses |

**So the 4.68× is a missing parameter, not a ceiling.** The optimum is interior, it sits at a
σ that brackets both data-implied figures above (log-ratio 0.2836, logit-share 0.4934), the
predictive sd lands on the 243.50 residual sd rather than overshooting, and MAE moves by less
than a minute across the whole sweep — it buys spread and not fit, which is precisely the
diagnosis. At σ = 0.45 the PIT KS of **0.0659** is *better calibrated at the season unit than
the marginal head's 0.0735*, while the team constraint still holds exactly.

**The caveat is load-bearing and the result should not be quoted without it.** σ is read off
validation CRPS, so this is a **tuned upper bound on what the parameterization can reach, not
a score.** A real fit estimates σ on train and re-estimates `beta` alongside it, and could
land better or worse. What the sweep legitimately settles is the structural question — the
constraint permits the spread — and that was the thing in doubt.

Two costs still to size. It is ~10,000 parameters on a 683k-row head that already costs
9.92 h and adapts a `dense_e` metric over ~25, so the sampler profile changes completely and
a non-centered parameterization is mandatory rather than optional. And the effect is only
partly identified for the players who matter most: a star's season share is pinned by the cap
on many nights, so the frailty and the cap compete to explain the same rows — visible in the
sweep as the redraw-clamp count rising from 5 to 71 across it.

### Why the zero-sum dynamic is a requirement, not a nicety

The constraint is not only a cost the composition pays; it is a **mechanism the contest is
sensitive to**, and no marginal metric can see whether a head has it. On 963 single-team
validation player-seasons:

| head | mean pairwise r between teammates | predictive sd of the team's season total |
|---|---|---|
| composition | **−0.0509** | 79.0 min |
| marginal head | **−0.0001** | **1,022.9** min |

A fixed team total over K players forces mean pairwise `r = −1/(K−1)`, which at the measured
roster size of 16.05 is **−0.0664**. The composition sits on it. The marginal head reads
essentially zero, and puts a **1,022.9-minute** predictive sd on a team season total that is
*physically fixed at ~19,810* — it assigns real probability to outcomes that cannot happen.
(Neither figure is 0.00 because dropping traded players leaves a subset of each roster and a
subset of a fixed-sum set has no fixed sum; the exact figure is the 0.00 on the complete
team-game blocks. The **ratio** is what carries.)

Two strategy axes depend on the sign directly, and both are in the sweep's config:

- **stacking.** A same-team pair's minutes are *anti*-correlated. Under independent draws
  they read as unrelated, so a minutes-driven stack is mispriced — plausibly with the wrong
  sign on its minutes component, on an axis `src/sim/strategy.py` explicitly sweeps.
- **handcuffing.** Drafting a starter's backup is a hedge that *only exists* if the model
  carries the negative correlation. A head without it cannot discover the strategy, so the
  sweep would silently never propose it.

This is also why "take the spread from the marginal head and the allocation from the
composition" is not the clean composition it sounds like — see below.

### What ships, and what the simulator has to do about it

- **Both heads stay.** `stan_minutes` is not retired, `make stan-minutes` stays in the
  chain, and it stays in `make posteriors`' head list.
- **No year block is ported into `composition_glm.stan`**, and the season-terms ablation is
  not re-run for it — that work was conditional on the composition winning. The `year` arm
  ships where it already did.
- **`README.md`'s "the two heads compose rather than compete" is replaced by the measured
  split**, which is what that sentence was reaching for and got wrong in two of three parts:
  the composition owns the per-game **allocation** and the coverage, and the marginal head
  owns the season-level **spread**. Neither owns "the season-level mean" — they tie on it.
- **`src/sim/season.py`'s minutes draw is the one thing this gate did NOT settle**, and the
  σ sweep is why. Three options, in the order they should be tried:

  1. **Inject the per-player-season effect into the composition's draw** (recommended). It
     needs no refit — the injection runs on the existing posterior and is ~10 lines in
     `src/sim/season.py` — it keeps the team constraint and the zero-sum dynamic exactly, and
     it ties the marginal head at the season unit. **The one honest piece of work it owes is
     σ**: 0.375 was read off validation and must be re-estimated on `train` before it can be
     used, or the sweep is tuned on the split it is later scored against.
  2. **Fit σ as a Stan parameter** and re-run the gate. Strictly better if it works, and it
     would reopen supersession — a composition carrying both the right season-level spread
     and the exact constraint is strictly more than the marginal head has. It costs a refit
     of the project's most expensive head, so it is a schedule decision against October, not
     a technical one.
  3. **Blend the two heads.** Listed last because it is *not* the clean composition it
     sounds like. Independent per-player season multipliers drawn from the marginal head are
     renormalized away by the allocation step, since the composition distributes a fixed pot
     — so the spread does not survive the blend, and scaling after allocation breaks the
     constraint instead. Anyone reaching for this should read the teammate-coupling table
     first: the marginal head's independence is not a neutral simplification, it is the
     thing that erases stacking and handcuffing.

  **Option 2 was chosen on 2026-08-09** and is specified below and built as item 3d.

---

## Fitting σ: the player-season effect as a Stan parameter

Chosen 2026-08-09 over the injection, because the injection's σ is tuned on the split it is
scored against and a fitted σ is not. The whole specification is below; build item 3d carries
the session prompt.

### What goes in `composition_glm.stan`, and the device that makes it safe

One optional block, added the way `betabinomial_glm.stan` adds its year effect:

```
eta = logit_prior + alpha + X * beta + sigma_u * u_z[unit_idx]
```

with `u_z` and `sigma_u` **zero-length when `U_n = 0`**, so the disabled model is not "the
same model with a small coefficient" but literally the current parameter space, priors and
likelihood. That is the `S = 0` device, and the file already uses the same trick internally
for `n_rho_par` on the binomial arm, so it is a house pattern rather than an import. **A test
pins the nesting**, exactly as the year block's does — it is the only thing standing between
"we added a parameter" and "we silently changed the shipped head."

`unit_idx` is per row and indexes **(player, season)**, not player: a player's minutes role is
a property of the season he is in, and a career-long effect would be absorbed by
`logit_share_lag1` and the offset.

### Two technical constraints that are not obvious, and one measured

**`dense_e` has to go on these arms.** The head's docstring records that the dense metric was
chosen because "at ~25 parameters the dense adaptation is free" — it took a one-season probe
from treedepth 8–9 and 645 s under `diag_e` to treedepth 4 and 65 s. That argument does not
survive the random effect. The full training window carries **12,307** player-season units
against 631,158 rows, so a dense metric is a 12,332² mass matrix — ~1.2 GB and a Cholesky of
it per adaptation window. **Use `diag_e` for any arm with the effect**, and expect the
treedepth win to be partly given back. This is the single largest cost risk in the item.

**Non-centered, and the reason is the sparse tail rather than convention.** Rows per unit run
median **57**, p10 **11**, minimum **1**. Well-informed units prefer a centered
parameterization and the one-game units are a funnel under it, so the mix argues for
non-centered as the default. It is not free — at 57 observations non-centered is the worse
choice — so **divergences are the diagnostic, not an assumption**, and a centered arm is the
first thing to try if they appear.

**The cap and the effect compete for the same rows.** A star's allocation is pinned by
`m_k = min(U, R_k)` on many nights, so `u` and the cap explain overlapping variation — the
`offset_clipped` indicator already marks exactly those rows. The injection sweep showed the
symptom: redraw clamps rose from 5 to 71 across it. If `sigma_u` comes back implausibly small
for stars, grading it by `w_share` bin the way `rho` already is (`n_rho = 4`) is the natural
second arm, not a new idea.

### The feature block — what is missing, what is on disk, and what it is worth

**The composition's 25 features are all properties of the player alone.** Prior availability,
prior minutes, workload, durability, age, playoff workload, his own prior share, four data
indicators, and two overtime terms. A head whose entire job is dividing a fixed team pot
among teammates carries **nothing about the teammates**. Team information enters only through
the offset, which is *last season's* composition.

`data/features/team_context_tierA.parquet` already holds the block, built leave-one-out and
point-in-time safe: `role_crowding` (minutes-weighted archetype similarity between a player
and his teammates — deliberately nonlinear, so it does not centroid-collapse the way a mean
of PC scores would), `teammate_usage_max/sum/load`, `teammate_assist_supply`,
`teammate_spacing`, `team_pace`, `n_teammates`, plus `reliability` and `roster_coverage` for
the rows it describes badly. It covers 11,928 player-seasons against the composition's wider
frame, so it will leave holes on exactly the rookies the composition refuses to drop — the
`design_missing` indicator already handles that shape and should be reused rather than
duplicated.

**Size the expectation honestly before building it.** Measured in a scratch session on 9,793
train player-seasons, with the deviation defined as `logit(realized share) − logit(prior
share)`, sd **0.674**:

| signal | correlation with the deviation |
|---|---|
| the player's own lag-1 deviation | **−0.201** — mean reversion, not persistence |
| departed teammates' prior share | +0.041 |
| arriving players' prior share | −0.063 |
| net minutes opened up | +0.088 |

In-sample R² on the deviation is **0.040** from own history alone and **0.052** adding the
team columns, so a crude team block is worth about **+1.2 points of R²** and everything
together explains **5.2%**. The signs are all right, which says the construction is sound;
the magnitudes say the deviation is overwhelmingly unforecastable from pre-season information
— the same wall the whole project runs into, where availability persists at r = 0.317 and
five games of the real season settle 86% of the season total.

Two caveats point in opposite directions and are the reason to test rather than assume. The
baseline is the **carry-forward floor, not the fitted model**, and the shipped β already
carries `minutes_per_game_lag2/3`, age and durability, which overlap with the mean-reversion
signal — so the incremental value over what ships is probably *below* 4%. But the team
columns above are a sum of shares, and `role_crowding` is the archetype-weighted version: a
departing star should matter far more to his positional replacement than to the roster
average, and a sum cannot see that. **These four correlations are a scratch measurement and
item 3d must re-derive them into its artifact**, so they stop being prose.

**ADP is off limits**, and that is a settled decision rather than an oversight
(`docs/adp-plan.md`; registry `adp-in-strategy-layer`). It would be the best available predictor of
a minutes-share change, and spending it here would leave the strategy layer's blend weight
sweeping an axis the model had already absorbed.

### The ladder, and the pilot that keeps it affordable

Four arms, and the incumbent is **not refitted** — its posterior is on disk:

| arm | mean function | effect | asks |
|---|---|---|---|
| `betabinom_ot_graded` | shipped | — | the incumbent, read from `posteriors/<window>/composition.pkl` |
| `+ps` | shipped | `sigma_u` | does the effect close the season-unit gap? |
| `+team` | shipped + team context | — | do features predict the deviation on their own? |
| `+ps_team` | shipped + team context | `sigma_u` | do features *reduce* `sigma_u`? |

**Run the pilot window first.** `stan.composition.first_season` is the knob and `probe_timing`
is the Gate A that already exists. At `2018-19` the training frame is **97,587** rows and
**2,204** units against the full window's 631,158 and 12,307 — 6.5× and 5.6× smaller. Get the
arm *ordering* there, then commit the selected arm to the full window. That is exactly the
path this head already took once, from pilot to Gate E.

### Gates

| gate | pass condition | why this bar |
|---|---|---|
| **P1** | `U_n = 0` reproduces the current posterior **exactly** — same parameter space, same draws at the same seed | The only thing separating "added a parameter" from "silently changed the shipped head". Test-pinned, mirroring the year block's |
| **P2** | re-running `make minutes-unification`, the composition **ties or beats** the marginal head at the season unit — paired-bootstrap interval straddling or below zero — with team-sum error still exactly 0 | The gate this whole line of work exists to move. The injection already showed a tie is reachable at σ ≈ 0.375 |
| **P3** | per-team-game CRPS does not regress past the incumbent's **4.4945** | The head's existing win is not to be traded for the new one. Both units, or it is not an improvement |
| **P4** | for `+ps_team` against `+ps`: the team block must **reduce fitted `sigma_u`** | The mechanism check. A feature block that improves CRPS *without* shrinking `sigma_u` is explaining something other than the player-season deviation, and the claim would not be supported |
| **P5** | `sigma_u` lands near the injection's **0.375–0.45** | A free replication. A fit that comes back at 0.05 or 1.5 disagrees with a measurement taken on the same rows, and that disagreement is a bug until explained |

Selection reads validation only, as everywhere.

### The fallback, and why it must be written down before starting

**If the fit blows the budget or will not converge, ship the injection with σ estimated on
`train`.** That is option 1, it needs no refit, `player_season_effect_sweep` already
implements the arithmetic, and it is *already measured* to tie. The only work it owes is
re-estimating σ on the fitting half instead of reading it off validation. Drafts happen
before October and a board that exists beats a board that is still sampling.

### What was built — 2026-08-09

The head-level **capability** is complete, tested and pinned; the **commitment** is a
compute decision the measurements below are meant to inform. Both halves are reported here
because separating them is the whole point of running a pilot.

**`composition_glm.stan` carries the block, and `U_n = 0` nests exactly.** Gate P1 is an
identity rather than an assertion about source text, and it holds: at `sigma_u = 0` the
effect model's log density exceeds the `U_n = 0` model's by **exactly** `−½Σz²` and nothing
else, so the likelihood, the offset, the priors on `alpha`/`beta` and the dispersion term
are untouched and the disabled arm is literally the shipped head. Pinned by
`tests/test_stan_composition.py::test_U_n_zero_nests_exactly_inside_the_player_season_model`,
mirroring the year block's. Two source-text tests pin the zero-length declarations and the
non-centred form beside it.

`PlayerSeasonTerm` in `stan_composition.py` is the Python half, and it is `YearTerm` one
index down: the fitted `u_z` are **never stored**, because they describe player-seasons that
are over, and the predictive integrates over a fresh `z ~ N(0, 1)` per (unit, posterior
draw), shared across that unit's games. `_eta_base` deliberately stays the *deterministic*
predictor — two consumers depend on that, `posteriors._finish` (whose round-trip reference
would otherwise be non-reproducible) and `player_season_effect_sweep` (which injects its own
σ on top and would otherwise double-count a fitted one).

**Three deliberate departures from the specification above, each with its reason.**

- **It is `make composition-effects`, not `make stan-composition`.** The plan says item 3d
  edits that head, and it does — the capability lives in `stan_composition.py`. But the
  *run* writes `outputs/predictions/composition_effects_*.csv` rather than overwriting
  `stan_composition_*.csv`, because that artifact is the incumbent's record and
  `make docs-audit` re-derives eleven quoted figures from it. A three-arm partial run at a
  pilot window would have failed the gate on bookkeeping rather than on a measurement.
- **The ladder is four arms, not three: a same-window `base` control was added.** The plan
  says not to refit the incumbent, and its full-window posterior is untouched — but a
  *pilot-window* arm ordering is uninterpretable against a full-window baseline, which is
  exactly what `stan_game_length`'s `season_trend_covered` control exists to prevent. `base`
  is the shipped specification on this window and nothing more.
- **The team block gets its own `team_missing` indicator rather than reusing
  `design_missing`.** The instruction was to reuse it, and the reason not to is measured:
  the two mark different rows. `design_missing` flags a composition row with no
  availability-design row at all; the team block additionally misses **4.14%** of pilot
  training rows that *do* have a design, because `team_context_tierA.parquet` is built off
  the season matrix's qualified frame (it covers 93.55% of pilot train rows and 94.20% of
  val). Folding them together would leave those rows imputed to the training mean with
  nothing for `beta` to correct on. What the instruction was *about* is honoured exactly:
  one indicator for the whole block, not five identical ones, which is the degenerate
  subspace `design_missing` exists to avoid.

**The four scratch correlations are re-derived into an artifact, and they hold.**
`outputs/predictions/composition_effects_deviation.csv`, measured on **8,570** full-window
training player-seasons with a real prior and ≥ 200 realized minutes (the plan's scratch
figure was 9,793 on a looser qualification):

| signal | re-derived | plan's scratch figure |
|---|---|---|
| the player's own lag-1 deviation | **−0.2519** | −0.201 |
| departed teammates' prior share | **+0.0411** | +0.041 |
| arriving players' prior share | **−0.0105** | −0.063 |
| net minutes opened up | **+0.0672** | +0.088 |

In-sample R² on the deviation runs **0.0634** from own history, **0.0699** adding roster
churn, and **0.0750** adding the five-column team block — so **the block is worth +1.2
points of R²**, which is what the plan predicted to the decimal, and the whole thing
explains **7.5%** of a deviation whose sd is **0.5954**. The team block *alone* reads
**0.0065**. Every sign matches, so the construction is sound; the magnitudes are the
finding, and they are the same wall the rest of the project runs into.

The measurement is taken on the **full window's** fitting half rather than the pilot's,
deliberately: the deviation is a property of the data and not of whichever window the head
is fitted on, and restricting it to four training seasons would cut it to ~1,500
player-seasons and make the re-derivation incomparable to the prose it replaces.

**`make posteriors` carries `sigma_u`, and only `sigma_u`.** `_finish` gained the capability
rather than being routed around: the scale is thinned on the *same* draw index as
`alpha`/`beta` (so a consumer cannot apply draw 700's spread to draw 3's coefficients), the
term object is copied rather than shared, and both the artifact `extras` and the manifest
carry it — a head whose predictive is quietly narrower than its fit is the one failure a
round-trip on the mean cannot see. The year-effect refusal stays exactly as it was, and now
names this as the worked example. The team block travels as a new `join` recipe step
carrying its own block and train means, so a rebuilt `team_context_tierA.parquet` cannot
silently change what a persisted posterior scores.

**`player_season_effect_sweep` reads σ from the artifact when one is there**, marked
`sigma_source = "fitted"` beside the injected grid, and the grid stays as the calibration
check — an interior optimum is how you find out whether a fitted σ landed in the right
place.

### The fallback is no longer a paragraph — it is measured and runnable ✅ 2026-08-09

The fallback above says: if the fit blows the budget, ship the injection with σ estimated on
`train` instead of validation. **That has now been done**, because it is minutes of numpy
rather than hours of sampler and because a board that exists beats a board that is still
sampling.

`minutes_unification.estimate_sigma_on_train` runs the identical grid — same arithmetic,
same metric, same code path — over the last two **training** seasons (2020-21 and 2021-22,
**1,145** player-seasons). It reads:

| σ | train CRPS | train PIT KS | train predictive sd |
|---|---|---|---|
| 0.000 | 139.89 | 0.3401 | 54.06 |
| 0.300 | 119.51 | 0.1832 | 144.75 |
| 0.375 | 117.45 | 0.1443 | 174.22 |
| **0.450** | **117.07** | **0.1112** | 203.38 |
| 0.600 | 119.55 | 0.1103 | 258.61 |

**σ_train = 0.450, against the validation grid's 0.375, and the optimum is interior on both
sides.** The agreement is the whole point: it says the figure the injection was criticised
for tuning on the evaluation split was not, in fact, moved by that split — the two grids
disagree by one step, and 0.375 and 0.450 are separated by 0.4 CRPS minutes on train and 0.7
on validation. So the injection can ship with a σ that owes the evaluation rows nothing.

At **σ = 0.450** the validation reading is CRPS **142.87** against the marginal head's
144.35 — a paired gap of **−1.49**, interval **[−6.14, +3.22]**, i.e. a **tie** — with PIT
KS **0.0659** against the marginal head's 0.0735, so it is the better-calibrated of the two
at the season unit, and the team constraint still holds exactly. That is the shippable
configuration today, and `src/sim/season.py` can consume it without waiting for a refit.

**It is a fallback and not the answer**, for the reason the plan gives: a fitted `sigma_u` is
estimated jointly with `beta` and could land better or worse, and only the fit can shrink σ
in response to features (Gate P4). But it removes the schedule risk from the item entirely.

### Cost — Gate A, and why the arm ordering is what the pilot buys

The random effect is expensive, and the probe says so plainly. Both arms on **one training
season** of the pilot window — identical rows, identical iterations (200 + 200 × 4 chains):

| arm | metric | wall clock | max R̂ | min ESS | divergences | treedepth-saturated | fitted σ_u |
|---|---|---|---|---|---|---|---|
| `base` | `dense_e` | **125 s** | 1.0172 | 400 | 0 | 0 | — |
| `ps` (non-centred) | `diag_e` | **1,496 s** | 1.0948 | **35** | **0** | 17 | **0.4776** |
| `ps_centered` | `diag_e` | 2,510 s | 1.1067 | 27 | **0** | **212** | 0.4809 |

**12.0× on the same 26,039 rows and 605 units**, which is the cost risk the plan named as the
largest in the item, arriving exactly where it was predicted: `dense_e` took that one-season
probe from treedepth 8–9 to treedepth 4 when the head had ~25 parameters, and 12,307
player-season units at the full window make a dense metric a 12,332-square mass matrix
(~1.2 GB and a Cholesky per adaptation window). The treedepth win is given back in full.

**And the `ps` fit did not converge at that budget** — R̂ 1.0948 against a 1.01 bar and a
minimum ESS of 35 against 400 — so the honest reading is that 1,496 s is a *lower* bound on
what a usable fit costs, not an estimate of one. The diagnosis is mixing rather than
geometry: **zero divergences** with 17 treedepth-saturated draws and a step size of 0.0094 is
a sampler taking very long trajectories through a poorly conditioned diagonal metric, not one
falling into a funnel.

### The centred parameterization is a measured null, and that closes a door

The plan names a centred arm as "the first thing to try if [divergences] appear", so
`ps_centered` was built and probed on the identical rows. **It is worse, and the evidence
that carries is not the wall clock.** The 2,510 s ran alongside the ladder's own Gate A
probe and is therefore contended, so the timing is not a clean comparison — but
**treedepth saturation is a property of the geometry rather than of the machine, and it goes
from 17 draws to 212, a 12.5× increase**. R̂ and ESS move the wrong way too. Both arms report
**zero divergences**, in both parameterizations.

That combination rules something out. A funnel produces divergences and is what the centred
form is dangerous for; zero divergences in both, with saturation rising sharply under
centring, says the posterior is not funnelling in either coordinate system — it is a long,
poorly conditioned ridge that NUTS is walking slowly. **So the parameterization is not the
lever**, and the plan's stated first response does not apply here. The premise it rested on
also does not survive the pilot window: it justified non-centred by "p10 11, minimum 1", but
at this window only **1.8%** of units carry a single row and the median is 49 — a
well-informed set, which is the regime centred is supposed to prefer.

**What is left to try is structural or mechanical, not a re-coordinatization.** The two
candidates, neither tested: `rho` and `sigma_u` may be competing for the same within-unit
overdispersion — the graded `rho` disperses the sequential binomial trials over four
prior-share bins while `sigma_u` shifts the same rows, and a ridge between them is exactly
what a collapsed step size with no divergences looks like. And `reduce_sum` threading is
untried and costs nothing in the posterior: the likelihood is one vectorized call plus
**1,487** scalar truncation calls at the pilot window, on a machine running 4 chains over 14
cores.

**A fourth estimate of σ_u came out of it, and this one is parameterization-independent.**
0.4809 centred against 0.4776 non-centred, 0.450 from the train injection grid and 0.375 from
the validation one. The first three shared arithmetic; these two share only the model, so the
agreement is a stronger check than the earlier ones.

Extrapolated by rows **and** units (a random-effect fit's cost is not linear in rows alone),
with the 1.63× correction this head's own Gate A already measured: **~6.3 h per random-effect
arm at the pilot window**, and **~15.7 h** for the four-arm ladder. At the full window a
single such arm is **~38 h**. The consequence for the schedule is worth stating without
hedging: **the full-window fit is not a same-day operation**, and the pilot exists precisely
so the arm ordering can be settled without paying for it.

**The one number that came back for free is the replication, and it is a good one.** The
under-converged one-season fit puts `sigma_u` at **0.4776**, against **0.375** from the
validation injection grid and **0.450** from the train injection grid — three estimates by
three different routes (a Stan parameter, a validation-scored grid, a train-scored grid)
inside a band of 0.10. Gate P5 asked for the fit to land near 0.375–0.45 and it does. Read it
as corroboration of the effect *size* rather than as a fitted value to ship: the chains had
not mixed, and a converged fit will move it.

### Where it lands downstream

- **`make posteriors` has to carry `sigma_u`, and only `sigma_u`.** The fitted `u_z` are
  useless for a season that has not happened — the predictive integrates over a **fresh** `z`
  per posterior draw, exactly as `YearTerm.shift` does. `posteriors._finish` already
  **raises** on a head fitted with a year random effect for precisely this reason; this item
  turns that refusal into a supported capability for the composition rather than routing
  around it.
- **`minutes_unification.player_season_effect_sweep` becomes the consumer.** It already does
  the right arithmetic; after the fit it reads σ from the artifact instead of sweeping a grid,
  and the sweep stays as the calibration check.
- **Supersession gets re-taken, not re-argued.** If P2 passes, the composition carries both
  the season-level spread and the exact team constraint, which is strictly more than the
  marginal head has. Re-run `make minutes-unification` and let it decide, the way
  `src/final_evaluation.py` reads which arm shipped rather than re-deciding.
- `docs/minutes-composition-plan.md` gets the head-level record when it lands; this section is
  the driver and the gate.

---

## The output contract that collapses the compute problem

**The drafting layer never sees a player-game.** Best ball scores by *scoring period* — the
best 7 of your 16 by slot, summed over the games in that week — so the simulator's deliverable
to everything downstream is a single tensor:

```
sim_tensor[player, scoring_period, sim]  ->  dk_pts     float32
```

Round 1 is 17 weeks and Rounds 2–4 are one double week each, so there are **20 scoring
periods**. At ~550 draftable players and 2,000 sims that is 550 × 20 × 2,000 × 4 bytes ≈
**88 MB** — small enough to hold in memory for the whole strategy sweep and small enough to
load into a draft room in under a second.

Per-game draws still happen, because the double-double bonus is a per-game threshold on five
components simultaneously and `E[bonus] ≠ bonus(E[x])`. They happen *inside* the simulator and
are summed into periods immediately. Nothing outside the simulator ever materializes a
`player × game × sim` array.

**Fixing this contract is the difference between a draft sweep that runs in minutes and one
that runs in hours**, and it is what makes the sub-second in-draft recompute achievable.

A second, smaller tensor rides alongside it and must not be forgotten:

```
availability[player, scoring_period, sim]  ->  games played that period    uint8
```

Because a zero-game week is not the same as a bad week for lineup selection, and because the
frozen-roster risk in Rounds 2–4 is entirely a games-played story.

---

## Contest mechanics — ground truth is `docs/dk_best_ball_rules.md`

- **Roster**: 16 players, at least 2 NBA teams.
- **Weekly lineup**: 7 starters auto-selected as the highest scorers by eligible slot — 2 G,
  2 F, 1 C, 2 UTIL (G/F/C) — 9 bench, bench points do not count. The scoring period is the date
  of the first game through the last game of that week's game set; a rescheduled or suspended
  game counts for the period it is **played** in.
- **Scoring**: matches `compute_dk_pts` exactly. Verified against the rules doc; **no scoring
  changes needed.**
- **Draft**: snake, order randomized once the lobby fills, 16 rounds, one live pod of **12
  entries** in Round 1 for every tournament.
- **Auto-draft**: queue first, then pre-draft ranking, with default caps of **8 G / 8 F / 3 C**
  unless overridden, plus an exclusion list. Those caps are DK's own defaults — `nba_stats`'s
  `make_pick` re-implemented them rather than inventing them, and the opponent model should use
  them as given.
- **Tournament**: 4 rounds, one frozen roster throughout, **no redraft**. Round 1 is 17 weeks
  (10/20–2/14) cumulative; Rounds 2–4 are each one double week (2/15–3/7, 3/8–3/21, 3/22–4/4).
  The NBA Cup championship game (12/11/2026) does not score. Ties break on best single week,
  cascading down through the round's weeks, then on best individual player score, cascading
  the same way.

### Scoring periods are NBA weeks — derive, do not hand-enter ✅ built 2026-08-08

`ScheduleLeagueV2` carries `weekNumber` and `weekName` natively. Checked on 2025-26: the week
numbering runs Monday–Sunday, partitions game dates with **zero** dates in more than one week,
and Week 17 closes 2026-02-12 against DK's stated Round-1 close of 2/14. DK's rounds are NBA
week ranges.

`make scoring-periods` (`src/features/scoring_periods.py`) writes one row per
`(season, game_id)` — its period index, the week's Monday/Sunday bounds, and its tournament
round — to `data/features/scoring_periods.parquet`. **35,546 games over 30 seasons.**

**`weekNumber` only exists from 2017-18**, which the probe did not reach: it is identically
zero for every earlier season, so 21 of the 30 need a derivation. The one that ships anchors
each game date to its Monday and **dense-ranks the Mondays that carry games**, and it
reproduces the NBA's own numbering on **10,749 of 10,749 games across all nine seasons that
publish one** — every game, every season, agreement 1.0000.

**Dense-ranking rather than counting elapsed calendar weeks is the whole trick, and 2019-20
is why.** The NBA suspended for four months and resumed in the bubble, and its own numbering
calls the restart weeks 22–24 — consecutive with March — where `(date − first) // 7` gives
41–43. An elapsed-weeks index reproduces the other eight seasons and breaks that one. Ranking
occupied weeks reproduces all nine, because in a normal season no week between the first and
last game is empty, so the two definitions coincide exactly where they can and differ only
where the NBA itself skips.

The three edge cases are owned here once rather than per use:

- **Played, not scheduled.** The period comes from the realized `game_date`, never a
  scheduled one. This is currently a distinction without a difference — `ScheduleLeagueV2`
  serves the *realized* schedule, so the two dates agree on **7,380 of 7,380** checked games
  — and it is implemented anyway, because it stops being free the moment the production board
  reads a *forward* schedule, which is when a silently wrong period would cost most.
- **The NBA Cup final scores nowhere**, and it is *doubly* excluded: the NBA does not count it
  as a regular-season game either, so it carries game-id prefix `006` rather than `002` and
  never enters `game_logs.parquet`. Both are asserted, because the label is the only one of
  the two that survives the NBA changing its mind about the box score. Note the rules copy
  dates it to 12/11/2026 while 2025-26's was played 2025-12-16 — one more reason the module
  identifies it from the schedule rather than from a hard-coded date.
- **The all-star gap** breaks week adjacency (Week 17's games stop 2/12, Week 18's open 2/19)
  and is handled by *not* handling it: a calendar-anchored grid does not notice, because the
  break moves no Monday. It is reported as `break_gap_days`, not corrected.

**Round 1 = 17 weeks and Rounds 2–4 = one double week each is asserted for all 27 full-length
seasons.** 2025-26 reads R1 = 17 weeks / 819 games, then 2 weeks each at 87, 105 and 108
games. Round 2's 87 is the all-star break landing inside it — Week 18 carries 36 games against
a normal ~52 — which is what makes DK's "double week" label fit a two-week window.

Two things the build found that the probe could not:

- **Three seasons cannot carry the structure at all**: the 1998-99 (14 weeks) and 2011-12 (19
  weeks) lockouts and 2020-21 (21 weeks). DK's own rules already say what happens — a season
  shortened before the end of Round 1 refunds every contest, and one shortened later voids the
  rounds that did not close — so a short round is recorded `complete = False` rather than
  quietly scored as a full one. The assertion keeps its teeth by checking what cannot vary: a
  round never runs *longer* than its published length, and an incomplete round is always the
  last one a season reaches.
- **The round map is specified in weeks, not dates**, and that is forced. DK's published
  windows belong to a single season's rules — the same copy dating the Cup final to
  12/11/2026 — and slide against any other season's grid. Anchored onto 2025-26 they drift
  **+1 day at Round 1 and −6 days at Rounds 2–4**: under a week, so the map is anchored, but
  enough to move which week the last double week lands on. So `sim.scoring_periods.round_weeks`
  (`[17, 2, 2, 2]`) drives the map, the calendar comparison is printed as drift rather than
  asserted, and the knob is there for when real per-season DK windows are obtained. One
  consequence worth stating: **DK's Round 4 ends before the NBA season does**, leaving 2
  unscored weeks in 2025-26.

---

## The two target tournaments

Derived live by `dashboard/economics.py` from `data/raw/dk_best_ball_tournament_*.csv`.

| | `600k_shootaround` | `20k_spin_move` |
|---|---|---|
| entry fee | $20 | $52 |
| **entries this year** | **10** ($200) | **4** ($208) |
| max per player | 150 | 12 |
| field | 35,280 | 432 |
| rake / break-even hurdle | 14.97% / **+17.60%** | 10.97% / **+12.32%** |
| R1 → R2 | 2 of 12 | 2 of 12 |
| R2 → R3 | 1 of 12, 11 cash ≥$30 | 2 of 6, 4 cash ≥$80 |
| R3 → R4 | 1 of 10, 9 cash ≥$100 | 2 of 6, 4 cash ≥$250 |
| R4 | 49 paid, min $1,000 | 8 paid, min $750 |
| first prize | $200,000 = **10,000×** | $5,000 = **96×** |
| P(reach R4) at random | 0.139% | 1.852% |
| E[entries reaching R4] | 0.014 | 0.074 |

**Round 1 is the only zero-consolation round, and that reframes the objective.** Surviving it
guarantees a cash in both tournaments — `600k_shootaround` pays 11 of 12 R2 entries at $30
minimum on a $20 entry, and `20k_spin_move` pays or advances all 6 at $80 minimum on a $52
entry. So `P(any return) = P(top 2 of 12)` exactly, and everything past Round 1 sets the *size*
of the return rather than its sign. The earlier framing of this doc — "convex, therefore chase
the tail" — is right only for `600k_shootaround`, and only above the R2 floor.

**The two tiers differ in what strategy should buy.** `20k_spin_move`'s path is three
successive shallow cuts (2/12 → 2/6 → 2/6) into a nearly flat final table, so it rewards
**survival probability** and durability. `600k_shootaround`'s path narrows brutally after
Round 1 (2/12 → 1/12 → 1/10) into a 10,000× top prize, so above the R2 floor it rewards
**correlated upside and differentiation from the field**. Confirming that the sweep actually
selects different rosters for the two is Gate D.

**Field quality is unmeasured and is a live risk.** A $1 field plausibly holds far more
autodraft entries than a $450 one, in the opposite direction from the rake math. The opponent
model must let field composition vary by tier rather than assume one field everywhere, even
though nothing calibrates it today.

---

## Architecture

Modules, each writing one artifact, in dependency order. Everything below the model line is
numpy over the posterior artifact — **only `make posteriors` and `make stan-game-length` need
CmdStan.**

```
make posteriors ✅   src/models/posteriors.py      thinned draws + design recipe per head
                                                   -> data/features/posteriors/<head>.pkl
                                                      + manifest.csv

make stan-game-length ✅ src/models/stan_game_length.py does a game go to OT, and how deep
                                                   -> outputs/predictions/stan_game_length_*.csv
                                                      + two posteriors/ entries. Replaced
                                                      stan_composition.fit_ot_tail, which
                                                      is deleted

make scoring-periods ✅ src/features/scoring_periods.py NBA week grid -> DK round windows
                                                   -> data/features/scoring_periods.parquet

make draft-pool ✅   src/features/draft_pool.py    the board: player x season, DK position
                                                   eligibility, team, ADP, prior-season row
                                                   -> data/features/draft_pool.parquet
                                                      + draft_pool_position_audit.csv

make minutes-unification ✅ src/models/minutes_unification.py
                                                   both minutes heads at the SEASON unit, on
                                                   the rows both cover. Refits nothing —
                                                   rehydrates each head around its persisted
                                                   draws, so it needs no CmdStan
                                                   -> outputs/predictions/minutes_unification.csv

make simulate-season ✅ src/sim/season.py           THE tensor: player x scoring_period x
                                                   sim dk_pts plus a uint8 games-played
                                                   twin. Per-game draws happen INSIDE it
                                                   -> data/features/sim_tensor_<season>.npz
                                                      + outputs/predictions/sim_season_gate_a.csv

make draft-sim ✅    src/sim/draft.py              a 12-entry, 16-round snake over one
                                                   engine with two modes — reactive
                                                   (primary) and ranking-submission. The
                                                   opponent model is a REGISTRY; the field
                                                   drafts off the DK-recalibrated consensus
                                                   -> outputs/predictions/draft_{gate_b,
                                                      adp_curve,field,reactive}.csv
make bracket ✅      src/sim/bracket.py            best 7 of 16 by slot per period, the
                                                   4-round chain, the cascading tie-break,
                                                   wildcards and payouts. Every structural
                                                   number from dashboard/economics.py
                                                   -> outputs/predictions/bracket_{structure,
                                                      null,entries}.csv
make strategy-sweep ✅ src/sim/strategy.py         the sweep: Gate C's error injection,
                                                   22 strategies x 2 tiers x 2 seasons,
                                                   paired on the simulated season, plus
                                                   the realized readout
                                                   -> outputs/predictions/strategy_{gate_c,
                                                      injection,null,sweep,paired,gate_d,
                                                      realized,shipped}.csv

make draft-room-prep ✅ src/sim/draft_room.py      the engine: the cached reference field,
                                                   the null check and Gate E
                                                   -> data/features/draft_room_field_<season>.npz
                                                      + outputs/predictions/draft_room_{gate_e,
                                                      null,stability,picks}.csv
make draft-room ✅   dashboard/draft_room.py       the page. One click per pick; loads the
                                                   artifact above rather than rebuilding it
```

`src/sim/` is a new package, parallel to `src/models/` and `src/eda/`, because these are
neither models nor analyses. Conventions carry over unchanged: `python -m src.sim.<module>`
entry points, the duplicated `yaml.safe_load` idiom in `__main__`, `mkdir(parents=True,
exist_ok=True)` before every write, `f"... {n:,} ... → {dest}"` progress lines.

### `src/sim/season.py` — the season simulator

Assembly, not invention. Every piece already exists and is measured:

| Step | Source | Number |
|---|---|---|
| **how long the game is** | **`stan_game_length.sample_game_length`** ✅ | fitted OT rate **0.0542** for 2026-27 at the `train` window; **once per game, shared by both teams** |
| which games he plays | `stan_games_played.sequences` | entry × exit × within-tenure chain, all gates pass |
| minutes, team-constrained | `stan_composition.simulate_minutes` | CRPS 4.4945 vs 4.7842 independent |
| minutes, game-level noise | the composition's **own** role-graded rho | 0.1768 fringe → 0.0855 star. `stan_minutes_dispersion.csv`'s **4.65×** is a *diagnostic* to check draws against, **not** an input — see "The third prerequisite" |
| **minutes, season-level spread** | **`stan_minutes`, and it cannot come from the composition** | summed composition draws are **4.68×** too narrow at the season unit (sd 64.65 against 302.75, PIT KS 0.3341 against 0.0735); the team constraint pins a team's season minutes to a constant, so no dispersion parameter buys it back. `make minutes-unification` |
| minutes, serial dependence | `serial_correlation.csv` | **2.43×** ten-game block inflation |
| the eleven component heads | `season_terms._draw_components` | already materializes `fga → fg3a\|fga → fg2a → makes` |
| cross-component dependence | `residual_correlation.csv` | Gaussian copula, mean +0.013, max 0.157 |
| the bonus | `targets.expected_bonus` / `compute_dk_pts` | overdispersion **0.025** at the player-game unit |

Six rules the assembly must not violate, all of them already argued elsewhere and repeated
here because this is the module that could quietly break them:

1. **Draw, never plug in.** `E[min]` and `E[gp]` are wrong inputs to a threshold bonus.
2. **One shared `min` draw per player-game** feeds all eleven heads as exposure. Minutes are
   46.4% of within-player residual variance; this is where the correlation comes from.
3. **Sequential structure goes on minutes and nowhere else.** Both conversion heads are
   measured nulls for a hot hand.
4. **One posterior draw moves the whole board.** Players share `β`, so the sim index must be
   the *outer* loop over posterior draws — not resampled per player. That shared-`β` sweep is
   the cross-player correlation this layer wants, and drawing it per player destroys it.
5. **One game-length draw per *game*, shared by both teams.** Overtime is a property of the
   game, not of a team or a player: every player on the floor gets the extra minutes together.
   Drawing it per team-game would silently destroy that, and it is a real source of the
   correlated upside the tournament objective rewards — a same-team stack, which the strategy
   layer explicitly considers, shares its overtimes.
6. **Minutes come from BOTH heads, and taking them from the composition alone is the failure
   this layer is most likely to ship.** The composition owns the per-game allocation and the
   coverage; `stan_minutes` owns the season-level spread, which the composition is
   structurally unable to produce. A board built from the composition alone is calibrated
   per game and **4.68× too confident per season** — and the season-total spread is exactly
   what a 2-of-12 cut is decided on, so the error lands squarely on the objective. Measured
   by `make minutes-unification`; see "The third prerequisite".

Consumers default to the `train` fit window — see "One window per consumer": the realized
backtest scores 2022-23 and 2023-24, which `train_val` fits on.

### What was built, and what Gate A found — 2026-08-09

`src/sim/season.py`, `make simulate-season`. **Two tensors, 386 x 20 x 2,000 and
387 x 20 x 2,000, 77 MB each, at 76 s per season of pure numpy and no CmdStan anywhere.** The
per-sim loop is the whole design: one posterior draw per simulated season shared by every
player and every head, a game-length draw per game shared by both teams, an availability
draw laid out into spells, one minutes draw pushed through all eleven component heads as
exposure, and `compute_dk_pts` on the drawn integer box score — scattered into the twenty
scoring periods with a single `bincount`, so no `player x game x sim` array is ever
materialized.

**Two structural choices were forced by the identities rather than chosen.** The component
heads are fitted at the *season* unit, so drawing them per game needs the negative
binomial's own Poisson-Gamma representation — a season-level `Gamma(phi, 1/phi)` frailty per
(player, sim), which *is* the fitted head's season-total spread, plus per-game Poisson noise
around `rate x minutes x frailty`. The conversion heads factorize the same way and more
cleanly: `p ~ Beta(a, b)` once per (player, sim), then `Binomial(attempts_g, p)` per game
sums to **exactly** `BetaBinomial(sum attempts, mu, rho)`. That is also what "sequential
structure goes on minutes and nowhere else" asks for, since both field-goal conversion heads
are measured nulls for a hot hand.

#### Gate A, in full

Each row against the artifact that set it. Validation only; the two seasons are reported
separately because pooling them would hide the one figure that moves.

| check | 2022-23 | 2023-24 | bar | artifact |
|---|---|---|---|---|
| season-total dk_pts MAE | **402.14** | **407.89** | 400.46 | `season_total_metrics.csv` |
| …CRPS | **280.49** | **281.03** | 287.26 | " |
| …R² | 0.6481 | 0.6589 | 0.7073 | " |
| …bias | −21.93 | −63.34 | −3.06 | " |
| games played CRPS | **9.6754** | **9.7829** | 10.0057 | `stan_games_played_metrics.csv` |
| …bias, in games | **+0.127** | **−0.363** | — | " |
| …pooled GP pmf total variation | **0.0602** | **0.0588** | — | `stan_games_played_gp_pmf.csv` |
| bonus per played game | 0.1733 | 0.1702 | 0.1559 / 0.1626 realized | `bonus_calibration.csv` |
| …on **realized** minutes | **0.1535** | **0.1477** | 0.1559 / 0.1626 | `component_targets.parquet` |
| season minutes sd, given GP | **322.05** | **319.32** | 302.75 | `minutes_unification.csv` |

**Three of the four gate rows pass and the fourth is traced out of this module.** Season
totals land on the incumbent's MAE within 2%, and *better* than it on CRPS — the deliverable
is a distribution and that is the distributional metric. Games played reproduces the
availability head it was handed rather than approximating it: CRPS **below** the head's own
10.0057 (the simulator integrates over the posterior draw where the head's published figure
is scored per row), a bias of a tenth of a game, and a pooled pmf within 0.06 total variation
of the persisted one on a mean of 55.8 games against 55.6.

**The bonus is +11% high in 2022-23 and +5% in 2023-24, and the cause is upstream.** Running
the identical `draw_components` call on **realized** minutes and realized played games gives
0.1535 against a realized 0.1559 and 0.1477 against 0.1626 — i.e. the component chain,
its season/game frailty split and its copula are calibrated on the bonus to within 1.5% and
9% respectively, in the *low* direction. Everything above that comes from the minutes the
simulator draws, and the next row says why.

#### The diagnostic that earned its keep: the composition's game-level minutes dispersion

`docs/simulations-plan.md` demoted `stan_minutes_dispersion.csv`'s **4.65x** from a simulator
*input* to "a diagnostic to check the composition's draws against". Run for the first time,
it fails — and not because of anything in `src/sim/`:

| source | implied game-level overdispersion |
|---|---|
| realized 2022-23 minutes | **4.22** |
| the simulator's draws | **8.42** |
| the composition head's own draws, on **realized** availability, sigma = 0 | **7.70** |
| …with the shipped sigma = 0.45 | 7.87 |

So the shipped composition head puts roughly **1.8x** too much game-to-game spread on a
player's minutes, measured against his own realized season share, and the simulator inherits
it almost exactly. The injected player-season effect is not the cause (7.70 at sigma = 0),
and neither is the availability draw (the middle row conditions on realized availability).
This is compatible with everything already measured about the head — its per-team-game CRPS
of 4.4945 and its PIT are statements about the *allocation*, not about a player's dispersion
around his own season mean — and it is exactly the miss the diagnostic was kept for. It is
also the mechanism behind the bonus row: the bonus is convex in minutes, so an over-dispersed
minutes draw over-produces double-doubles.

**Two more diagnostics, both reported rather than gated.**

- **Serial dependence is not consumed, and that is now a measured decision rather than an
  omission.** `serial_correlation.csv`'s **2.43x** ten-game block inflation is a named
  simulator input; the simulator produces **1.40** / **1.52**, because the composition's
  draws are iid across games once availability is fixed and only the spell process clusters
  them. The mechanism to close it is ~3 lines — a `sigma_block * z[player, block]` term in
  the linear predictor — and it is **deliberately not shipped**, because it would add
  variance to a minutes draw that is already 1.8x too dispersed at the game level. The two
  live in one variance budget and the block term should be fitted against the dispersion
  miss, not on top of it.
- **The copula needed inverting, and conflating the two matrices was a real bug.**
  `residual_correlation.csv` measures the correlation of *Pearson residuals*; under a
  lognormal frailty of variance `v = 0.025`, `corr(resid_a, resid_b) = R_ab * v *
  sqrt(mu_a mu_b) / sqrt((1+v mu_a)(1+v mu_b))`, so the frailty correlation that produces a
  given residual correlation is roughly **ten times** it. The first build handed the copula
  the residual matrix directly and imposed a tenth of the intended dependence — every cell
  present, every shape right, only the numbers wrong. Inverting the relation at the
  population mean per-game count and projecting back to a valid correlation matrix takes the
  simulated off-diagonal mean from **−0.002** to **+0.017** against a target of **+0.022**,
  with a maximum cell error of 0.057. **4 of the 21 count pairs saturate**, which is itself a
  finding: the measured residual coupling sits at the ceiling a frailty of this variance can
  produce, so the bonus overdispersion and the residual correlation are close to two views of
  one per-game "big night" factor rather than two independent inputs.

#### Two wiring faults Gate A caught, both invisible in the output

Both would have produced a completely plausible board.

- **The availability denominator.** Taking the panel as it stands gives a traded player rows
  on *both* teams and a denominator of **92.6** games against the head's **82.0**, so the
  head's rate ran against ~13% too many opportunities. `features.availability.season_availability`
  already settles this — a traded player is attributed wholly to his **last team** — and
  reproducing that convention puts the two within one game on 432 of 433 players.
- **The players the availability head has no row for.** It is a lag-1 design, so 106 of 539
  rostered players in 2022-23 are outside its frame. Scoring them at the head's *intercept*
  put them at **58.4** simulated games against a realized **30.1**, and because the minutes
  allocation is zero-sum that moved ~**29,500** minutes a season off the players the tensor
  scores — a season-total dk_pts bias of **−90.8**. They now get the expanding-window
  empirical rate of no-design player-seasons in the earlier seasons selection may read
  (**0.4303** for 2022-23), which is the same point-in-time device
  `stan_composition.rookie_share_priors` already uses for their minutes share. The bias falls
  to **−21.9**.

#### What the artifact carries, and the honest caveats

`data/features/sim_tensor_<season>.npz`: `dk_pts` (float32), `games_played` (uint8),
`player_id`, `season_minutes`, `prior_minutes`, the round map for the twenty slots, and the
provenance a consumer needs to refuse the wrong one — fit window, sim count, posterior draw
count, seed, the composition variant, and the player-season sigma with its source.

- **The tensor scores 386 of 539 rostered players.** The missing 153 have no component-head
  design row (`>= 200 prior minutes`), and they are *kept in the minutes allocation* —
  dropping them would hand their minutes to their teammates — but cannot be scored. Pricing
  them is item 4's open question and belongs with the 2026 draft class the board build
  already flagged.
- **The season total is scored over DK's window, not the schedule.** Round 4 closes before
  the NBA season does, so the tensor carries 20 of ~24 weeks and both sides of the Gate A
  comparison are restricted to it. Games played is scored over the whole schedule, because
  that is the availability head's own denominator.
- **`offset_clipped` is the one design column supplied from an expectation rather than a
  draw.** The fitted head reads it off the realized allocation; a forward season has none, so
  it is evaluated at the deterministic proportional allocation. It marks ~1% of rows and
  carries a correction coefficient.
- **A feasibility repair fires on 0.12–0.18% of simulated team-games.** A team-game allocates
  `5 x game_length` minutes under a per-player cap of `game_length`, so it needs five
  available players; below that the highest-ranked absentees are promoted and the count is
  reported.
- **Mid-season trades are still not modelled**, inherited from the prediction layer.

### `src/sim/draft.py` — the draft simulator

A snake draft over 12 entries and 16 rounds. Opponents autodraft off recalibrated-DK ADP with
rank noise, subject to DK's real 8G/8F/3C caps. Two modes over one engine:

- **reactive** (primary) — a pick function sees the current board state and returns a ranked
  recommendation. This is what the draft room calls and what the strategy sweep exercises.
- **ranking-submission** (fallback) — a static pre-draft ranking plus position limits and an
  exclusion list, executed by DK's documented autodraft logic. Needed because a 30-second
  clock can outrun a human, and needed for the opponent model regardless.

**The field model has a real calibration target, which the plan previously assumed it did
not.** Observed ADP *is* the field's realized aggregate behaviour, so simulating many drafts
and measuring the resulting average draft position must reproduce the observed ADP curve. That
is Gate B, and it is what "calibrated to reproduce ADP" means concretely.

### What was built, and what Gate B found — 2026-08-09

`src/sim/draft.py`, `make draft-sim`. **Gate B passes wide — pooled mean absolute rank gap
5.922 picks against the 17.0 bar — and the interesting result is what the fit could not
resolve.**

| arm | 2022-23 | 2023-24 | pooled | grid spread over sd ∈ [0, 30] |
|---|---|---|---|---|
| `constant`, best sd | **0.00** | **4.00** | — | **0.110** / **0.117** picks |
| **`tiered`, best sd** ✅ | **4.00** | **4.00** | **4.00** | 29.449 / 29.535 picks |
| tiered MAE (fit region) | 6.294 | 5.551 | **5.922** | bar **17.0** |
| tiered MAE (all ADP'd) | 8.309 | 9.856 | 9.082 | bar 17.0 |

🔴 **A constant rank noise is not identified by a mean-ADP target, and that is a property of
the target rather than a failure of the fit.** Under symmetric noise of any size `E[pick]` is
the board rank for any interior player, so the whole grid from sd = 0 to sd = 30 moves the
objective by **0.110 picks** in 2022-23 and **0.117** in 2023-24 — a tenth of a pick across a
range whose top is plainly absurd — and the two seasons put its optimum in different places
(**0.00** and **4.00**) inside that band. Every grid point is run from the same seed, so those
are paired comparisons and the flatness is the objective rather than Monte Carlo error. The
plan's instruction to *fit* `rank_noise_sd` rather than choose it was worth following for
exactly this reason: choosing a plausible value would have concealed that the data never spoke.

**A rank-dependent shape is identified, because it is pinned at both ends at once.** The
tiered arm scales `docs/adp-plan.md`'s measured tier disagreement (5.1 picks in rounds 1–2
against 30.8 in rounds 9+) to mean 1 and fits one scalar over it, so the ladder stays
one-dimensional. Its optimum is interior — 5.966 at sd = 0, falling monotonically to
**5.922** at 4.00, then 6.138, 7.244, 10.03 and 35.41 — and **both validation seasons land on
4.00 independently**, which is the replication since they share no rows. The mechanism is
visible in the elite tier: the observed consensus #1 goes at **1.05**, and at its optimum the
tiered field puts him at **1.469** against the constant arm's **2.479** at the same scale.

**The selected arm beats the no-noise floor by +0.044 picks, which is not a result**, and it
is not why a noise of zero cannot ship. `field_diversity` is:

| | roster overlap between two drafts | distinct players drafted |
|---|---|---|
| `rank_noise_sd = 0` | **100%** | **1.000×** one draft's worth |
| shipped (`tiered`, 4.00) | **15.2%** | **1.133×** |

At zero noise every draft plays out identically, so a seat holds one roster in every simulated
season and the 35,280-entry field `make bracket` scores is **twelve rosters repeated**. Every
marginal statistic about it is fine. The joint object is not a field, and no ADP curve can see
it — which is the sharpest argument in this section for capturing real pick logs.

🔴 **The fit excludes the transfer map's terminal plateau, and that is about identification
rather than about passing.** The fitted isotonic recalibration takes 253 consensus ranks to 58
values, and both validation boards run off the end of it into one flat value — **160.14**, held
by **33** players in 2022-23 and **60** in 2023-24. Inside that plateau the target carries no
ordering information at all, so no field can reproduce it and any noise that shortens the tail
scores better, which drags an unrestricted fit toward implausibly large values. Gate B is
reported on **both** populations and passes on both (5.922 and 9.082 against 17.0), so the
restriction never rescues the gate.

**Three further things the build settled.**

- **A monotone recalibration cannot reorder a board**, so using the DK-recalibrated consensus
  rather than the raw one changes the ADP *values* and not a single pick. That is not an
  argument against it — the values are Gate B's target, they are the units the blend `α` will
  sweep in, and `docs/adp-plan.md`'s binding stands — but the C/F/G bias it was partly
  motivated by is *not* corrected by the shipped one-dimensional map, which is why the
  position offset was measured at −0.29 picks and left out. Noise therefore goes on the
  **rank**: the map's 55-wide plateau would make fifty-five players exchangeable under noise
  applied to the value.
- **DK's 8 G / 8 F / 3 C caps bind autodraft and not a person.** The rules are explicit —
  "the only way to override them once the draft starts is to make a manual selection" — so the
  reactive seat runs under `manual_config` and the eleven opponents keep the caps. Applying
  them to our own seat would silently forbid a roster a human may draft.
- **The caps do not imply a minimum**, and `require_legal_lineup` is the separate guard that
  does. 8 G + 8 F + 0 C satisfies every DK cap and seats no centre, and
  `bracket.best_lineup` returns a plausible six-man total for it without raising. A test pins
  both directions on a board whose centres all rank behind the last pick.

**The reactive mode is exercised rather than described**, and its latency is item 7's
headline in advance: ranked by marginal lineup value over the full remaining pool at
`n_sims = 500`, one recompute costs roughly **0.76 s mean and 1.5 s max** — wall clock, so it
measures the machine rather than the model. Gate E's bar is 1,000 ms.
So the draft room is on the edge and the two levers the plan already named — a partial sort,
and not re-ranking the deep tail — are needed rather than optional.

**Artifacts.** `outputs/predictions/draft_{gate_b,adp_curve,field,reactive}.csv`. The shipped
field is read from `draft_gate_b.csv` by `selected_field`, because `configs/default.yaml`
carries `rank_noise_sd: null` — a literal number there would be a second source of truth for a
figure the artifact owns, which is the rule `src/final_evaluation.py` already follows.

### 🎯 Real pick logs are the calibration this layer is missing

Raised 2026-08-09, out of Gate B. The measurement above says plainly that **the aggregate ADP
curve cannot identify how noisy a field is** — it constrains the mean, and a field is a joint
object. Entering a few cheap 12-entry pods early and recording the pick order would identify
directly what no amount of further ADP work can:

- **the noise level**, from the *variance* of a player's pick position rather than its mean —
  one board, twenty pods, and `rank_noise_sd` is measured instead of fitted against a flat
  objective. **The binding sample size is the number of pods, not the number of picks**: a
  drafted player leaves the board, so each pod observes him exactly once and twenty pods give
  twenty observations of his pick position however many picks that is. Twenty 12-man pods are
  worth far more than a handful of larger ones;
- **whether the noise is private or shared** — twelve drafters with independent opinions, or
  a room that collectively moves off the printed board. Both reproduce the same mean ADP and
  they are not the same field: private noise makes rosters diverge, a shared shift moves
  everyone together. It is a *between*-pod statistic and separable by attenuation. Under
  private noise a player goes when the first of twelve drafters to over-rate him gets the
  clock, which is an order statistic over twelve draws and concentrates hard, so his pick
  position varies much less across pods than the per-drafter sd; a shared shift passes through
  one-for-one. Predict the across-pod pick sd from the fitted private model and compare it to
  the realized one. This is the same identity the minutes head turns on — per-game noise
  averages down by ~1/√G while a season-level multiplier does not, which is why summed
  composition draws came out 4.68× too narrow;
- **the shape**, i.e. whether the tiered hypothesis is right at all. The elite tier is nearly
  deterministic in the observed ADP and the deep rounds are close to uncorrelated; a pick log
  says whether that is one noise scale varying with rank or two different behaviours;
- **positional runs**, which are the largest thing the current model does not have. Real
  drafters take a centre because they need one, and runs at a position are a well-known draft
  dynamic that pure ADP-plus-noise cannot generate. `AdpNeedAware` is built and switched off
  precisely because nothing calibrates `need_weight`;
- **the autodraft share**, which is `field_composition`'s uncalibrated knob and the tier risk
  this doc already logs. An entry that is autodrafting is visible in a pick log as a seat that
  never deviates from the board.

What to record, per pod: the **board** in force (a DK pre-draft-rankings CSV captured the same
day), then one row per pick — `pick_number`, `seat`, `player`, and whether the pick was manual
or autodrafted where it can be told. Twenty pods at a $1–3 entry is ~$40 and 3,840 picks, which
is a large sample for a two-parameter noise model. **The board capture matters as much as the
picks**: a pick log without the contemporaneous board measures the sum of the field's noise and
the board's drift, and DK's board cannot be backfilled (see `docs/adp-plan.md`).

This is a **data-capture deadline like the October board**, not a modelling task: drafts happen
before October and a pod not entered is not recoverable afterwards.

### `src/sim/bracket.py` — rounds, advancement, ties, payouts ✅ built 2026-08-09

Reads the tournament spec from the two CSVs via `dashboard.economics` — round count, pod size,
advance count and cash table — so pointing this at the real 2026-27 numbers is a data change.
It must get three things exactly right:

- **the cascading tie-break**, because the 2-vs-3 boundary in a 12-entry pod over 17 weeks will
  be close often, and the rules are explicit: best single week, then second-best, down through
  the round, then best individual player score, cascading the same way;
- **wildcards**, which fill any shortfall from the highest-scoring non-advancing entries;
- **the selected survivor field in Rounds 2–4.** Opponents there are not an ADP field — they
  are the population that already cleared a 2-of-12 cut. Simulating the whole bracket gets this
  for free; scoring rounds independently against a fresh ADP field would systematically
  overstate continuation value.

#### The weekly lineup is an assignment problem, and first-fit is wrong by 2 points in 188

The specification above did not name the lineup itself, and it is the part with a wrong
answer sitting in easy reach. A week starts **2 G / 2 F / 1 C / 2 UTIL** out of 16, and a
dual-eligible player seated in the first slot he fits can lock a better player out
altogether. On the roster `tests/test_bracket.py` pins, a first-fit assigner scores **186**
against the true **188**: it seats a G/F dual at guard, which fills both guard seats and
both UTIL seats with guards and strands the fifth guard, so the lineup reaches down to a
23-point centre instead. The error is **one-sided** — it can only understate — and silent.

**No solver is needed, and that is a structural fact rather than an optimization.** A
lineup's value depends on *which* seven players are picked and never on where they sit, so
the question is which 7-subsets can be seated at all — and those are exactly the independent
sets of a **transversal matroid**. Greedy is optimal on a matroid, so sorting the sixteen by
score and keeping every player whose addition preserves seatability is provably the maximum,
in sixteen vectorized steps with no dependency. Seatability is Hall's condition, which over
three position types is **eight inequalities**: for every subset `A` of {G, F, C}, the
players eligible only within `A` must not outnumber `capacity(A) + 2` UTIL seats. `A =
{G,F,C}` is the roster-size constraint and `A = {}` refuses a player with no position, so
the eight cover everything.

DK ships single-position players (`dk-is-single-position-and-the-map-is-86-percent`), so
this costs nothing today — and it is what keeps the rejected dual convention a column swap
(`dual_g` / `dual_f` / `dual_c`) rather than a rewrite.

#### The gate: the symmetric-field null, and the two defects it caught

The bracket has no gate in the table below, so it was given one that is known in closed
form. In a field where every entry is drawn from the same process, each one advances at
`n_advance / pod_size` and is worth exactly **`-rake`**, because a field of identical
entries must collect the whole prize pool and nothing more. That single identity exercises
the pod sizes, the advance chain, the wildcard fill and every cash band simultaneously.

All five captured tournaments reconcile to **1e-16**, and `make bracket` reproduces it by
simulation on exchangeable entries as well as deriving it analytically. It found two real
defects the day it was written, neither of which any marginal check would have shown:

- **The per-player tie-break level was asymmetric.** Our entries carried their real
  per-player contributions and the field carried a column of zeros, which is not a *missing*
  tie-break but a *winning* one — lexicographically, `(-300, -250, …)` sorts ahead of
  `(-0.0,)` every time. Because dk_pts are quarter-point multiples, exact ties are common
  rather than exotic, so every one of them went to us: **+68%** on P(reach round 4), **+27%**
  on round 3, and a null ROI of **+71%** where the truth is −11%. The level is now
  all-or-nothing — used when both populations carry it, and otherwise falling through to the
  weekly cascade and then to the random break, which is the right answer for two identical
  rosters anyway.
- **A transcription error in the prize CSV.** `15k_and_one` appeared to pay 24 of its 42
  finalists for $13,200 against a stated $15,000 pool, while the other four reconciled to
  the cent. **Pod sizes for rounds 2+ are inferred** — `economics.advance_table` takes each
  round's pod to be the largest place it pays or advances, since the CSV records payouts and
  not contest sizes — so the first question was whether the inference was wrong. A brute
  force over every pod-size assignment consistent with the CSV found **none** that closed
  the gap, which identified the rows rather than the pods. Corrected at source the same day
  (42 paid places, $9,771). The inference now has three independent corroborations at 5 of 5
  where it previously had two at 4 of 5: the chain stays integral, each final round's field
  equals its paid places, and the payouts reconcile to the pool. `economics`'s own
  `final_field_equals_paid` evidence moved from 4 to 5, and its test with it.

#### Progression is dealt and ranked, and the model that used to sit here is withdrawn

The whole contest is played out at its real field size. Each round shuffles the survivors,
deals them into real pods, ranks each pod by the cascade above, and carries the top
`n_advance` forward; our own entries are simply *in* the field at known rows, which is what
DK does with them. An entry's place is its place.

**That is a reversal, and the thing it reverses is instructive.** The first version sized
the field by a config knob (3,000) and cut it by pods — and `600k_shootaround` advances 1 in
720 across three cuts, so Round 4 was decided against **four** surviving entries standing in
for a 49-entry final table. The null read ROI **+0.72** against an exact −0.1497. The fix
taken at the time was to keep the small field and replace the survivor *subset* with a
per-entry survival **weight**, drawing each entry's place parametrically from it. That
removed the degeneracy and cost two further bugs, both caught by the same null:

- a **binomial** pod-mate count where a pod is dealt *without* replacement. Invisible at
  Round 1, where 11 pod-mates come from tens of thousands; decisive at the final round,
  where the pod *is* the surviving field. Worth **+0.08 of ROI** on `20k_spin_move`;
- an **off-by-one** on whether an entry joins the field or occupies one of its slots, which
  left last place reachable **0.016** of the time against 0.125 in a pod of 8.

Sizing the field correctly dissolves the original problem instead of managing it: 35,280 →
5,880 → 490 → 49 and 432 → 72 → 24 → 8 are all real populations. So the weights, the place
distribution and the renormalization step are gone. **The wrong lesson from the degeneracy
was that the survivor population needs a model; the right one is that the field size is a
structural number.**

Two quantities that were Monte Carlo estimates are now **exact identities**, and both are
reported per run: every round's survivor count equals the published field size, and the
payouts sum to the prize pool — measured at **0.00e+00** for all five tournaments. The
tie-break also moved to where the rules put it, *within the contest being decided* rather
than over a global ordering of the field.

**Wildcards** are DK's own mechanism and are dormant in production: every captured chain
divides exactly, which `verify_chain` asserts for all 20 rounds, so the pods always deliver
the target field. The path is tested directly, because the case DK documents it for —
Round-1 contests that did not all fill — is one a production run could meet.

#### The field is the tournament's real entry count, and all five are simulated

**35,280 / 17,640 / 14,688 / 432 / 216**, read from the captured metadata rather than
chosen. It is a structural number like the others, and it is load-bearing: the final round
is one contest of everyone who reached it, so the field size *is* the last pod, and sizing
it by hand is what produced every problem in the section above.

All five tournaments are simulated, not only the two being entered. They cost **one scoring
pass** between them — the field is drafted once per season at the largest tournament's size
and each contest takes the prefix it needs, which is valid because entries are exchangeable
and every field size is a multiple of the 12-entry Round-1 pod. Four structures the money is
not going into are four more chances for a structural bug to surface, and they span the
shapes: `88k_alley_oop` is 216 entries into a 4-man final table, `600k_shootaround` is
35,280 into 49.

#### The benchmark entry has to be held out of the field

A "best available" entry — the board's top sixteen, undraftable in a real pod — is carried
as a reference to show the bracket discriminates at all. **It cannot sit in the field while
the null is measured**, and that is not fastidiousness: the prize pool is fixed, so a strong
entry's winnings come out of everyone else's. In `88k_alley_oop`, 216 entries with a $20,000
top prize on an $88,000 pool, one entry that always reaches the final table moved every
other entry's ROI by more than **twenty points** — the null read −0.2954 against −0.0947
until the benchmark was pulled out and substituted into a single seat for its own run.

#### The null sample has to be drawn at random, and its interval resampled over entries

Two defects in the *check* rather than in the bracket, both found by the same disagreement
and both worth recording because they are the shape of mistake that makes a gate lie.

**The sample was a prefix of the field.** `placeholder_field` lays rows out as
`pod * 12 + seat`, so a prefix that does not end on a pod boundary over-weights early draft
seats. A 250-row prefix of `20k_spin_move`'s 432 entries read ROI **−0.05** against the
exact −0.11, while the whole field read the identity — the sample was biased, not the
simulator.

**And the interval could not have caught it**, because it resampled *sims*. Two entries
differ by their rosters, which are fixed across sims, so a sim-resample sees none of the
variability that actually separates entries and reports an interval far too narrow. Both are
now correct: the sample is a uniform random subset and the bootstrap resamples entries.

The general lesson is the one this layer keeps re-learning: **an identity is a stronger check
than an estimate**. The survivor counts and the payout total are exact and caught real bugs;
the per-entry ROI is an estimate and needed two fixes of its own before it could be believed.

#### 🔴 `draft_pool.parquet` is the wrong frame to derive the split from

The first version of `make bracket` took its seasons from the draft pool and ran a full
backtest on **2024-25 — a test season — without raising.** The pool carries the live 2026-27
production board, so its last two labels are 2025-26 and 2026-27 and `selection_split` hands
back 2023-24 and 2024-25 as "validation", one season forward of the project's.

**The guard did not object because the guard believes the frame it is handed.** That is a
different failure from the one `src/models/held_out.py` was built for: the capability check
is sound, and it was fed a frame whose last two seasons are not the last two target seasons.
The pool is right to carry 2026-27 — that is the production board — so the fix is on the
consumer, and the split now comes from the same component design `make simulate-season`
builds its tensors against, which is the only frame that can be right here since the bracket
scores those tensors. Pinned by a test that re-locks the guard first, since `conftest`
unlocks the suite.

#### What is a placeholder, and what is final

`placeholder_field` builds the opponents with Gumbel-noised twelve-entry snake drafts over
the board. **Its first version drew each entry independently and that was badly wrong** —
mean board rank of a pick **9.4**, against the ~96 a twelve-man draft implies, so every
entry held the same top ten and the rosters were largely identical. Drafting twelve entries
against a shared board consumes 192 players and makes rosters disjoint within a pod, which
is what a real pod looks like; legality is enforced during the draft (an entry whose
remaining picks equal the positions it still owes is restricted to them) rather than
repaired afterwards.

Build item 6 (`src/sim/draft.py`) replaces the *ranking* it reads — DK's recalibrated ADP
consensus rather than the simulator's own projection — plus DK's autodraft caps and the
`rank_noise_sd` Gate B fits. The draft mechanism stays. Everything above it — lineups, ties,
advancement, wildcards, payouts — is final.

### `src/sim/strategy.py` — the sweep

A strategy is a config object, so exploration is a table rather than a rewrite:

```yaml
ranking:      model_mean | model_quantile:<q> | adp | blend
alpha:        scalar, or per-round (rounds 1-2 vs 9+ — see below)
position_caps: {G: 8, F: 8, C: 3}          # DK defaults; overridable
exposure_caps: per-player share across the portfolio's entries
stacking:     same-team pair bonus, 0 = off
objective:    bracket_ev | p_advance | expected_score
n_entries:    10 (600k_shootaround) | 4 (20k_spin_move)
```

**Select on P(advance) lift, report ROI.** The two are not equally measurable. ROI is dominated
by rare deep runs — `600k_shootaround` reaches Round 4 on 0.139% of entries — so its Monte
Carlo error is enormous. `P(top 2 of 12)` is a 16.67% event and resolves orders of magnitude
faster on the same simulation budget. Since surviving Round 1 is exactly the condition for any
return at all, the lift in `P(top 2 of 12)` over an ADP-drafted entry is both the statistic the
sweep can resolve and a defensible headline. ROI against the break-even hurdle is reported
alongside it, with its interval.

`docs/adp-plan.md` binds two things on the blend: use the **DK-recalibrated** consensus, not a
raw one (a monotone recalibration cuts cross-validated error from 24.0 to 17.0 picks, and DK
drafts centers 11.9 picks earlier because category-league ADP discounts them for FT%), and
expect `α` to want to vary by round, since disagreement is 5.1 picks in rounds 1–2 against 30.8
in rounds 9+ — which is where 9 of the 16 roster spots are filled.

### What was built, and what Gates C and D found — 2026-08-09

`src/sim/strategy.py`, `make strategy-sweep`. **22 strategies × 2 tiers × 2 validation
seasons at 500 simulated worlds each, in 4.8 minutes of numpy.** Gate C passes on the two
targets it can hit and misses two rows in a stated direction; Gate D **fails, and that is the
result** — the two tiers do not select materially different rosters, under a tier-blind
ranking or a tier-aware objective.

#### 🔴 Gate C's premise was half wrong, and fixing it changed what gets injected

The plan's argument is that a world drawn from the model's own posterior is one where the
model is calibrated by construction, so ADP can only add noise and `α → 0` for reasons that
have nothing to do with the market. **The conclusion is right and the stated mechanism is
not.** The mechanism is a claim about *magnitude* — that the simulated world is too easy —
and it had never been measured. Measured, it does not hold:

| | 2022-23 | 2023-24 | bar | artifact |
|---|---|---|---|---|
| season-total MAE, **uninjected** | **414.97** | **398.96** | 400.46 | `season_total_metrics.csv` |
| availability CRPS, **uninjected** | **10.0935** | **9.9387** | 10.0057 | `stan_games_played_metrics.csv` |
| season-total R², uninjected | 0.5520 | 0.5994 | 0.7073 | `season_total_metrics.csv` |
| dk_pts-per-game R², uninjected | 0.7018 | 0.7374 | 0.81–0.95 | `stan_component_metrics.csv` |

The uninjected world already reproduces the model's measured out-of-sample miss on two of
the four rows and is **harder** than reality on the other two. That is what a head which
shrinks hard is supposed to deliver — its predictive spread is about the size of its real
error — and Gate A had half-said it already, with the simulator's season-total CRPS coming in
*better* than the incumbent's against realized data.

**What the uninjected world gets wrong is not the size of the error but the standing of the
two rankers.** On the priced players of the validation seasons, Spearman against realized
season totals reads model **0.7044 / 0.7090** against market **0.7564 / 0.7250** — the market
is *ahead* by **+0.052** and **+0.016**. In a world drawn from the model's posterior the model
leads by **−0.111** and **−0.118**, because there it is the unbiased efficient predictor and
ADP is a strictly noisier view of the same thing. That is a swing of **0.16 to 0.13 Spearman
points**, and no amount of extra *noise* closes it: noise is what the model already has too
much of relative to the market.

So the injection **rotates** the error onto the market-visible direction at fixed magnitude
rather than adding error on top of it, and it has two solved parameters rather than plugged-in
ones: `g` holds the season-total MAE on its bar and `ρ` puts the simulated skill gap on the
realized one. Both land exactly (MAE **400.4586** against 400.4586; gap **+0.0521** and
**+0.0160** against +0.0521 and +0.0160).

**`ρ` has an independent second route and the two agree.** Solved from the skill gap it is
**0.4292 / 0.3999**; measured directly as `corr(market disagreement, model residual)` on
realized data it is **0.3083 / 0.3360**, i.e. the market sees **9.5% / 11.3%** of the
variance of the model's miss. The two share no arithmetic — one is a difference of rank
correlations, the other a correlation between a disagreement and a residual — so agreement
inside 0.1 is evidence rather than bookkeeping.

**Two Gate C rows are not met, both in the conservative direction**, and they are reported
rather than tuned away. Season-total R² reads **0.5428 / 0.5733** against 0.7073 and the
per-game rate R² reads **0.5280 / 0.5811** against the count heads' 0.81–0.95 floor band. The
injected world is *harder* than reality: at the same MAE, truth has less between-player spread
than a real season does, so ranking players is harder there than here. A harder world
understates every strategy's lift, and since the *relative* standing of model and market is
pinned it should leave `α` roughly where it belongs.

**The band itself had to be re-derived rather than typed in.** 0.81–0.95 is the **count**
heads' no-fit carry-forward floor. Read off the *selected* rows of the same file it becomes
[0.13, 0.96], because three of the four conversion heads score under 0.35 — a bar no simulated
world could fail. A test pins which rows it comes from.

#### 🔴 The board had to be restricted to priceable players, and it is the largest single correction here

`make simulate-season` scores 386 of 539 rostered players and pads the rest with **zeros** so a
draft can still run into them. A model-ranked strategy never takes one. Measured on every run
by drafting thirty pods of the fitted field on the *unrestricted* board, the ADP field takes
**1.2556** and **1.1861** of them per sixteen-man entry across the two seasons, and **73.06%**
and **74.72%** of its entries hold at least one — a roster spot that scores nothing all season.
That is a coverage hole in the tensor arriving as a handicap on one side of the comparison, and
it is worth far more than any axis the sweep measures. In a reduced-budget diagnostic run
before the fix, our own **pure-ADP** entry read `P(top 2 of 12) = 0.285` against an exact
0.1667 and `model_mean` read 0.43 — a diagnostic rather than a reproducible figure, kept
because it is the size of the thing. `make bracket` sees the same defect from the other end,
where the best-available benchmark reads `p_advance = 1.0` in all five tournaments.

So both sides now draft the priceable board: **347 of 448** rows in 2022-23 and **359 of 464**
in 2023-24, of which **16** and **21** carry ADP. The cost is stated rather than hidden — who is
on the board at pick *k* changes, so the field's picks are slightly better than a real field's —
and it is smaller and more honest than scoring a real player at zero. The right fix is upstream:
pricing those 153 players is the open question item 4 left.

**The guard that says the correction worked is the symmetric-field null**, run on the injected
field before any strategy is scored: an entry drawn from the field reaches Round 1 at
**0.166667** against an exact 0.166667, error **5e-11**. Every lift below is measured against
that number.

#### The sweep, paired — because the unpaired intervals cover the whole table

Unpaired, the top of the table is `+0.211 [+0.143, +0.282]` and the eighth row is
`+0.119 [+0.038, +0.209]`: nothing is separated from anything. That is a property of the
*level*, not of the differences — a simulated season kind to one strategy is kind to all of
them, since every entry is scored on the same drawn world. Differencing inside the sim removes
the common term, and then almost everything resolves. Lift in `P(top 2 of 12)` against
`model_mean`, pooled over 1,000 worlds:

| arm | axis | 600k_shootaround | 20k_spin_move |
|---|---|---|---|
| `lineup_value_blend30` ✅ | objective | **+0.1082 [+0.0954, +0.1214]** | **+0.0720 [+0.0594, +0.0845]** |
| `lineup_value` | objective | +0.0744 [+0.0636, +0.0858] | +0.0491 [+0.0389, +0.0597] |
| `bracket_ev_blend30` | objective | +0.0494 [+0.0372, +0.0608] | +0.0501 [+0.0368, +0.0632] |
| `blend_a15` | alpha | +0.0354 [+0.0287, +0.0426] | +0.0387 [+0.0314, +0.0468] |
| `blend_late` | alpha_by_round | +0.0343 [+0.0255, +0.0433] | +0.0473 [+0.0383, +0.0572] |
| `blend_a70` | alpha | +0.0223 [+0.0094, +0.0351] | +0.0402 [+0.0282, +0.0527] |
| `blend_caps_dk` | position_caps | +0.0160 [+0.0058, +0.0258] | +0.0187 [+0.0092, +0.0278] |
| `bracket_ev` | objective | **−0.0225 [−0.0339, −0.0111]** | +0.0070 [−0.0042, +0.0185] |
| `model_q90` | ranking | −0.0075 [−0.0163, +0.0014] | −0.0199 [−0.0297, −0.0098] |
| `adp` | ranking | **−0.0576 [−0.0733, −0.0423]** | **−0.0987 [−0.1146, −0.0828]** |

**Six things this settles.**

- **`α > 0` pays, and it is resolved.** Every blend arm is at or above `model_mean` and pure
  `adp` loses decisively in both tiers. The blend is worth a further **+0.0338 [+0.0237,
  +0.0446]** *on top of* the best objective (`lineup_value_blend30` against `lineup_value`), so
  it is not a substitute for the in-draft pricing but an addition to it.
- **Which `α` is not resolved, and the two tiers disagree about it.** 600k peaks at α = 0.15
  and 20k at α = 0.70, with α = 0.30 and 0.50 unresolved against zero at 600k. The α *axis* has
  a sign; its *location* does not, on this budget. That is Gate B's finding about
  `rank_noise_sd` arriving one layer up, and for the same reason: the objective is flat near
  its optimum.
- **The per-round direction `docs/adp-plan.md` predicts is confirmed in one tier and not the
  other.** Against its own reverse control, `blend_late` (α 0.15/0.35/0.65 over rounds 1-2, 3-8,
  9+) minus `blend_early` (0.65/0.35/0.15) reads **+0.0340 [+0.0266, +0.0423]** at 20k —
  resolved — and **+0.0050 [−0.0025, +0.0133]** at 600k. Leaning on the market in the deep
  rounds is right where it resolves and never wrong.
- **Stacking is a measured loss.** Against its own uncapped twin, `blend_stack12` costs
  **−0.0179 / −0.0431** of lift and buys **+0.0077 / −0.0401** of `P(any of N)`. That is the
  sign the zero-sum minutes constraint implies — teammates' season minutes are anti-correlated
  at a measured mean pairwise **r = −0.0509** — and it is the answer to an axis the plan flagged
  as possibly mispriced with the wrong sign.
- **Exposure caps are a real trade with a measured price on both sides.** Against the same twin,
  a 40% cap costs **−0.0461 [−0.0538, −0.0385]** of per-entry lift and buys **+0.0236 [+0.0142,
  +0.0332]** of `P(any of N)`. Neither dominates. The selection criterion is per-entry lift, so
  the cap can never be *selected* by it — which is exactly why the portfolio statistic is
  reported beside it rather than instead of it.
- **The EV objective trades survival for money, in the tournament whose money is in the tail.**
  `bracket_ev` is the *worst* resolved arm at 600k on lift (−0.0225) and the *best* on ROI
  (**+61.8** against the shipped arm's +21.5). `select-on-p-advance-report-roi` says select on
  the first; the disagreement is a real decision and it is recorded rather than smoothed. The
  draft room measured the same split from the other end — "a P(advance)-maximal roster is a
  chalk roster".

**What ships is `lineup_value_blend30` for both tiers**: rank by the marginal weekly-lineup
value of `src/sim/draft_room.py`'s matroid exchange, blended 30% into the DK-recalibrated ADP
rank, no exposure cap, no stacking, uncapped positions. Written to
`outputs/predictions/strategy_shipped.csv`, which build item 10 reads rather than re-deciding.

#### Gate D fails: the two tiers do not select different rosters

| season | comparison | cross-tier overlap | within-tier | verdict |
|---|---|---|---|---|
| 2022-23 | shipped (tier-blind) | 0.480 | 0.517 / 0.396 | not different |
| 2022-23 | `bracket_ev` (tier-aware) | 0.713 | 0.704 / 0.792 | not different |
| 2022-23 | `bracket_ev_blend30` (tier-aware) | 0.442 | 0.404 / 0.458 | not different |
| 2023-24 | shipped (tier-blind) | 0.430 | 0.415 / 0.312 | not different |
| 2023-24 | `bracket_ev` (tier-aware) | 0.544 | 0.524 / 0.583 | not different |
| 2023-24 | `bracket_ev_blend30` (tier-aware) | 0.292 | 0.271 / 0.323 | not different |

In every comparison the two tiers' portfolios overlap each other about as much as each
overlaps itself, and both tiers select the **same** strategy. **The comparison names its own
mechanism**, which is what makes the answer interpretable: a `ranking` strategy is tier-blind
by construction — the board key knows nothing about which payout table it is drafting into — so
under those arms Gate D can only fail. The `bracket_ev` arms *are* tier-aware, pricing each
candidate against that tournament's own pods, advance counts and cash bands, and they fail it
too.

**The mechanism is Round 1.** Both tournaments cut 2 of 12 in the round that decides whether
there is any return at all, and 83% of entries are gone there whichever one they entered.
Everything the economics say about the tiers — a 10,000× top prize against a flat final table —
is a statement about the 17% of paths that survive, so it moves the objective very little.
`two-strategies-two-tiers` said comparing the two tiers is itself a result; the result is that
**the two portfolios can share a board**, and the practical consequence for October is that
there is one board to build rather than two. (`only_a` / `only_b` in the artifact are confounded
by entry count — 10 entries touch more players than 4 — so `cross_overlap` against `within` is
the fair reading.)

#### The realized readout — one good season, one wash

Same portfolios, replayed against real 2022-23 and 2023-24 box scores. **A readout, not a
selector.**

| season | tier | shipped `P(top 2 of 12)` | lift | an ADP entry |
|---|---|---|---|---|
| 2022-23 | 600k_shootaround | 0.4015 [0.1719, 0.6397] | **+0.235** | 0.1438 |
| 2023-24 | 600k_shootaround | 0.1855 [0.0237, 0.3943] | **+0.019** | 0.2545 |
| 2022-23 | 20k_spin_move | 0.3382 [0.0002, 0.7501] | **+0.172** | 0.3427 |
| 2023-24 | 20k_spin_move | 0.1412 [0.0102, 0.3739] | **−0.026** | 0.2045 |

2022-23 is a good season and 2023-24 is a wash in which the ADP entry beat us in both tiers.
The intervals resample the **field** and the entries, not the season — a season cannot be
resampled, there are two of them, and the honest statement is that **the realized edge is not
distinguishable from zero.** What the readout is for is catching a strategy broken in a way the
simulated world cannot see, and nothing here is broken: the shipped arm clears the ADP baseline
in one season of each tier and trails it in the other, which is what N = 2 looks like.

#### Three caveats that bound every number above

- 🔴 **The injection acts on the season-level rate and leaves the model's knowledge of the
  *shape* exact.** All three of Gate C's named targets are level statistics — a season total, a
  games-played CRPS, a rate R² — so an injection calibrated to them cannot perturb what the model
  knows about the weekly distribution, the double-double threshold, or the cross-component
  correlation. The shipped objective uses all of that, and truth is drawn from the same joint.
  So the simulated lift is an **upper bound** on a real one.
- 🔴 **The field drafts strictly by ADP with rank noise and does no lineup reasoning at all.**
  A real drafter balances positions. The edge measured here is over that field, not over a
  room of humans, and `docs/simulations-plan.md` already names real pick logs as the missing
  calibration.
- **`ρ` is measured on the same two seasons the sweep scores.** It is a simulator *input*,
  calibrated the way the other four are, and validation is the split selection may read — but
  every `α` below inherits the sampling error of two seasons of ~200 priced players.

**ROI is reported and is not a level to act on.** 600k reads **+21.5 [+5.6, +46.8]** against a
+17.60% hurdle and 20k reads **+2.98 [+0.83, +5.54]** against +12.32%. Both clear the hurdle by
orders of magnitude, which is itself the reason not to believe them: they inherit the three
caveats above, and 600k's ROI is the figure `make bracket` and `make draft-room-prep` both
already record as not estimable at any affordable budget (the null's E[payout] reads **−15%**
at this field size, against **−0.0%** for 20k). The lift in `P(top 2 of 12)` is the number to
read.

---

## The backtest, and why its two halves do different jobs

### Simulated truth — the tuning surface

Draw a season from the posterior, call it truth, draft against it, score the bracket. Unlimited
resolution, and the only surface with enough power to separate strategies.

**Why an uninjected simulated backtest cannot price ADP.** In a world drawn from the model's own
posterior, the model is perfectly calibrated by construction. ADP can then only add noise, so
the sweep will drive `α → 0` for reasons that have nothing to do with whether the market knows
something. The same failure hits every strategy that hedges model error: exposure caps,
differentiation, shrinkage toward consensus.

So the truth draw is **perturbed to reproduce the model's measured out-of-sample miss** before
anything is scored against it. The calibration targets are already on disk: the availability
head's validation CRPS (10.006 games), the component heads' validation R² against their no-fit
floors (0.81–0.95), and the season-total MAE (400.5 dk_pts). The injected world must reproduce
those, not the model's in-sample calibration. **An uninjected sweep is not a conservative
version of this — it is a sweep that answers a different question**, and its α is not
transportable.

This is the one genuinely new piece of statistical machinery in the layer, and it deserves its
own gate (Gate C).

### Realized truth — the honest readout

Replay simulated portfolios against real box scores. Two seasons: **2022-23 and 2023-24**, the
validation split, both of which happen to carry FantasyPros ADP.

N = 2 seasons of correlated pods will not distinguish `α = 0.3` from `α = 0.5`, and the plan
should not pretend otherwise. Its job is to catch a strategy that is broken in a way the
simulated world cannot see, and to put an honest — wide — interval on the measured edge.

**A cheap widening exists if it turns out to be needed — but it is half the size this plan
thought.** ADP was said to also cover 2014-15, 2017-18, 2018-19 and 2019-20. Under
`adp.training_rows` **only 2014-15 survives**: in the other three, every archived snapshot
postdates the season's first game, so the board those seasons drafted on was never captured
(`make draft-pool`, and see "Point-in-time costs four of the nine ADP seasons"). The widening
is therefore **three seasons, not six**. What it buys is unchanged in kind: the fitted heads are
in-sample on 2014-15, but the **no-fit carry-forward floor is out-of-sample by construction**
and scores only 0.001–0.03 R² below the fitted heads, so a floor-ranked backtest is a legitimate
robustness check on strategy *shape* even though it cannot price the fitted model's edge. Build
it only if the two-season result is ambiguous, and expect less from it than the original note
promised.

### The test split — a pure readout, mechanized as one

Settled 2026-08-08: the test seasons (2024-25, 2025-26) get **one** backtest run before going
live, and it changes **nothing**. Not the strategy, not the stake, not the entry decision.

The rule is enforced the way `src/models/held_out.py` enforces it for the model heads, because
prose already failed once here:

- the strategy sweep goes through `selection_split` and never materializes the test rows;
- the shipped strategy is written to an artifact by the validation sweep, and the test runner
  **reads which strategy shipped rather than re-deciding**, exactly as `src/final_evaluation.py`
  does;
- the test runner is a **report generator** — ROI distribution, P(advance), P(cash), worst-case
  drawdown across the 10 + 4 entries — and emits no ranking, no selection and no recommendation;
- it runs inside `held_out.unlocked("pre-season risk readout")` so the unlock is visible in the
  log.

---

## The live draft room

`dashboard/draft_room.py`, a Streamlit page separate from the walkthrough app. It loads the
precomputed sim tensor and the draft pool, shows the board, and takes **one click per pick** to
mark a player gone. No external access, no OCR, no page scraping on the critical path.

**Since 2026-08-10 it ships twice off one `render()`** — still its own app under `make
draft-room`, which is what draft night launches, and also page 9 of the dashboard. That is
`docs/dashboard-plan.md`'s step 7, and the reason the standalone launch stays is blast
radius rather than speed: measured in Chrome, the page paints in 3.17 s cold against the
standalone launch's 3.63 s and in **0.31 s** on a return, because `st.cache_resource` holds
the reference field for the life of the process. What a separate process buys is that
nothing else can raise, block or allocate inside a thirty-second clock.

**Latency is the design constraint and it should be treated as a gate.** A 30-second fast-draft
clock means a recompute budget well under a second. That is achievable because the expensive
work is precomputed: the in-draft calculation is the marginal bracket EV of adding each
remaining player to the current roster, which is array math over a `16 × 20 × n_sims` slice.
Two things make it fit:

- **drop to `n_sims = 500` in-draft** and keep 2,000 for the sweep. The in-draft decision is a
  ranking of candidates, not an estimate of a level;
- **best-7-by-slot is a partial sort**, not a full one — never recompute the whole lineup
  selection when one player is added.

If the budget cannot be met, the fallback is a precomputed static ranking with an exclusion
list, which is the ranking-submission mode already being built.

Reading the DK draft page directly — via the browser extension or a pasted pick log — is worth
exploring for 8-hour slow drafts, where the clock is not the constraint. It is explicitly not
on the critical path and should not gate the draft room shipping.

### What was built, and what Gate E found ✅ 2026-08-09

`dashboard/draft_room.py` is the page and `src/sim/draft_room.py` is everything that computes
anything; `make draft-room` runs the first and `make draft-room-prep` runs the second.
**Gate E passes with five times the headroom the plan budgeted for: 112 ms mean, 185 ms p95
and 200 ms worst over the full remaining pool of 359 priceable players at `n_sims = 500`,
against a 1,000 ms bar.** The fallback — a static ranking plus exclusion list in DK's
pre-draft-rankings format — stays built as `draft.export_ranking` and is now genuinely a
fallback rather than a likely outcome.

**Both levers the plan named were required, and the second is exact rather than a partial
sort.** `make draft-sim` had measured the marginal-lineup-value recompute at 0.76 s mean and
1.5 s max — over the bar before the bracket EV was added on top. `n_sims = 500` is the first
lever and unchanged. The second is sharper than "a partial sort": `bracket.best_lineup` is
matroid greedy, and for a matroid the max-weight basis of `S + c` is either the old basis or
a single exchange out of it, so with the basis computed once per (period, sim) a candidate's
lift is

```
lift = max(0, score(c) − threshold[mask(c)])
```

one subtraction over `[candidate, period, sim]` instead of a 16-step greedy over a
`[candidate, period, sim, 17]` gather. **Which player he displaces is Hall's condition, not
the lowest score** — the man he replaces must relieve every tight constraint at once, and
displacing the lineup's cheapest starter outright would let a fifth guard evict a centre.
`tests/test_draft_room.py` pins the identity against `best_lineup` itself over 400 random
rosters, on single-position and dual-eligible masks both, because a wrong threshold still
returns a ranked table.

#### The objective, and the two things it needed that the plan did not name

Decision 5's payout-weighted bracket EV needs a *field* and a *finished roster*, and neither
falls out of the tensor.

- **The field is drafted, once, and cached.** 100 twelve-seat pods of `src/sim/draft.py`'s
  fitted field, scored on the same tensor and the same sims as our own entry, written to
  `data/features/draft_room_field_<season>.npz`. It costs ~13 s to build and under a second
  to reload, which is why launching a room is a second rather than a minute. Rounds 2–4 face
  the **survivor** population, obtained by reweighting that same field by its own advance
  probability rather than by dealing it — `bracket.py`'s argument that an independent field
  overstates continuation value, reached analytically because a recompute cannot deal 35,280
  entries inside a 30-second clock.
- **A pick is priced inside a *completed* roster** — what we hold, the candidate, and the
  best available at each pick we have left. Scored as the roster stands at pick 3 our entry
  is so far below a field of complete rosters that P(top 2 of 12) is zero for every
  candidate and the ranking has no resolution at all. The completion **fills the 2 G / 2 F /
  1 C slate before taking best available**, which is `bracket.top_roster`'s convention and
  is load-bearing rather than tidy: the completion is the baseline every candidate is
  measured against, and deferring the centre to the last forced pick left a
  replacement-level centre in the base and priced every centre on the board against it. On
  2022-23's opening pick that put **five centres in the top seven and dropped Dončić to
  eighth**; filling the slate first returns the board to Jokić, Giannis, Dončić, Embiid,
  Tatum.

**The symmetric-field null is what caught the one real error in the arithmetic.** Running
the field's own entries through the same path a candidate goes through must reproduce
`bracket.symmetric_null` exactly — `P(advance round 1) = n_advance / pod_size`, and
`E[payout] = total_prizes / total_entries`. The first version read P(advance) 44% high in
round 3, because the naive survival function is a **right-endpoint Riemann sum** of
`∫(1−q)^(P−1) dq` and biases every round's advance rate by about `1 / (2 n_eff)`. With the
midpoint plotting position — ties split down the middle, which is also what the rules'
cascade does — the reach probabilities come back exact to **2e-8** at every round.

#### 🔴 The EV level does not converge, and the ranking inherits it

**All five captured structures are priced, and running the null on all of them turns one
caveat into a pattern.** The room loads a reference for every tournament rather than only
the two being entered — all five run a 12-entry Round-1 pod, so the draft is identical and
a reference is a reweighting of a field already drafted and scored. The EV error then
tracks one thing:

| tournament | E[payout], analytic | from the room's field | error | effective entries by round |
|---|---|---|---|---|
| `88k_alley_oop` | $407.4074 | **$407.4048** | **−0.0%** | 1,200 / 304 / 122 / 47 |
| `20k_spin_move` | $46.2963 | **$46.2838** | **−0.03%** | 1,200 / 304 / 122 / 47 |
| `15k_and_one` | $0.8503 | $0.8278 | −2.7% | 1,200 / 304 / 45 / 10 |
| `50k_four_pt_play` | $3.4041 | $3.2624 | −4.2% | 1,200 / 304 / 62 / 13 |
| `600k_shootaround` | $17.0068 | **$14.0892** | **−17.2%** | 1,200 / 304 / 38 / **6** |

The ordering is the ratio of the final table's size to the population that reaches it. The
two structures with shallow cuts (2/6 → 2/6) into a small final table reproduce the null
**exactly**; the ones that funnel a whole field into a 42-, 49- or 68-seat table do not,
and `600k_shootaround` is worst because two thirds of its EV sits in a 49-seat table reached
by 0.139% of entries and topped by a 10,000× prize — six effective entries to resolve it.
**So the EV is trustworthy where the money is spread and untrustworthy where it is
concentrated**, which is a sharper statement than "the EV is a level with a known bias" and
it is only visible because all five are priced.

**Tripling the field does not fix the worst case and does not stabilize its ranking**:
300 pods buy −7.7% for 43 s of build time and the rank correlation across two fields
*falls* from 0.86 to 0.78. This is `600k_shootaround`'s ROI is not estimable at any
affordable simulation budget, arriving in the in-draft objective.

P(top 2 of 12) is **0.166667 against 0.166667 in all five**, which is the other half of the
same point: Round 1 is a 2-of-12 cut in every captured structure, so the statistic that
resolves is also the statistic that does not depend on which tournament you are pricing.

What that costs the recommendation is measured rather than assumed, by redrawing the field
from a second seed and re-ranking the same board states:

| objective | keeps its top pick | top-3 overlap | top-5 overlap | rank correlation |
|---|---|---|---|---|
| `bracket_ev` (`600k_shootaround`) | **87.5%** | **0.750** | 0.950 | **0.8771** |
| `bracket_ev` (`20k_spin_move`) | 87.5% | 1.000 | 0.975 | 0.9915 |
| `p_advance` | **100%** | **1.000** | **1.000** | **0.9979** |
| `lineup_value` | 100% | 1.000 | 1.000 | 1.0000 |

So the EV ships as the objective, because that is decision 5 and it is the money question —
and `p_advance` ships beside it on every row, because it is the statistic that resolves and
the one `select-on-p-advance-report-roi` already says the sweep selects on.

**Which of the two you read is a real choice rather than a formality, and it is larger than
this doc assumed.** Measured over eight board states per season, the two objectives name the
same top pick on **50% to 88%** of them, with a top-3 overlap of **0.33 to 0.62** and a rank
correlation of **0.43 to 0.66**. They are not two views of one quantity: `p_advance` is the
*sign* of the return and is blind to everything above the cut, while the EV is the *size* and
carries the 10,000× top prize that is most of `600k_shootaround`'s money. A
P(advance)-maximal roster is a chalk roster, and chalk is the wrong shape for the tournament
whose whole prize is in the tail. The practical reading, until Gate C prices the model error:
take `p_advance` when the EV lead is inside the field noise, and let the EV break ties among
candidates whose P(advance) is level. **The bracket-EV
pick differs from the marginal-lineup-value pick on 25% to 50% of the sixteen rounds**,
which is the answer to whether substituting the objective was worth doing at all.

#### The pick log is the one artifact in this project that cannot be regenerated

Added 2026-08-09, after the room shipped. Every other figure here is a `make` target away;
a pod is played once and the order the board came off in is gone the moment the tab
closes. So the room writes `outputs/draft_logs/draft_log_<season>_<session>.csv` **after
every pick** rather than on a button — a live draft is exactly where a closed tab costs
something unrecoverable, and 192 rows of CSV is microseconds — with a download button
beside it. Note that `outputs/` is gitignored, as `data/raw/` is: the log lands with the DK
boards and the injury snapshots in the set of captures that are not backfillable and not
versioned, and it wants the same backup they do.

It records the snake — `board_index` replays the whole draft through `replay`, and the
board columns join to `draft_pool.parquet` without a name match — plus, on each row, **what
the room advised at the moment the pick was made**. That is the part that cannot be
reconstructed afterwards: re-ranking from a finished log would score each pick against a
board state that did not exist when it was made. On our own rows `cost_vs_best` is
therefore what overriding the model cost by the model's own reckoning, and twenty real
drafts of it is the only honest record of whether a human under a 30-second clock helps or
hurts. On an opponent's row the same columns price what the field took against what our
board wanted, which is the disagreement the whole strategy rests on — so they are kept, and
`followed` is null there rather than `False`.

This is the same capture `🎯 Real pick logs are the calibration this layer is missing` asks
for, with the board attached by construction rather than remembered separately.

#### Injury notes are on the page, and deliberately nowhere near the ranking

Added 2026-08-09. **The room is the first thing in this project to show a drafter something
the model has not seen.** Nothing in the pipeline consumes either injury feed today:
`src/data/injuries.py` writes the ESPN log and no module reads it, and
`src/data/injury_reports.py`'s only consumer is `src/eda/report_calibration.py`, which
measures `P(play | designation)` as a study rather than as a feature. So the availability
head knows how much a player missed *last* season and cannot know he had surgery in June —
which `docs/availability-plan.md` already names as the one genuinely new input, blocked
until the daily capture spans an offseason boundary.

That gap is the reason to show it and also the reason to keep it out of the value path. The
feeds describe **today**, so on a backtest board today's status *is* the resolved outcome,
and folding either into a ranking would be the leak `point-in-time-discipline` forbids — one
no split guard could see, because the guards sit on frames rather than on displayed text. So
the notes are attached to rows for a human to read, `evaluate` never receives them, and a
test pins that the columns stay out of both the ranking and the pick log.

Three things the build had to get right:

- **The two feeds answer different questions and are not blended.** The NBA report is
  published about an hour before a game, so between June and October it carries no player
  rows at all — its last report naming anybody is **2026-06-13**, 57 days before the
  snapshot beside it. ESPN is the feed that is alive in the offseason, which is when a
  best-ball draft happens: **148 players as of 2026-08-03**, 67 of them on a validation
  board. Both are shown with their capture date and age, because a merged status would hide
  which one said it.
- **Neither feed carries a player id**, so this is one of the name joins
  `docs/model-development-notes.md` allows, and it gets the second guard that rule demands:
  uniqueness on *both* sides. A key naming two board rows, or two rows inside one feed, is
  reported as `ambiguous` and attached to nobody. Attaching a wrong note is worse than
  attaching none — a drafter who passes on a healthy star because the room labelled him Out
  has lost the pick, and no downstream number would ever show it. Team is deliberately not
  used as a third field: the feeds print `"Brooklyn Nets"` where the board prints `"BKN"`,
  and inventing a thirty-row lookup to disambiguate a case that does not currently occur
  (0 ambiguous on both validation boards) is how a join acquires a silent failure mode.
- **A feed describing another season is flagged as one.** On a 2023-24 practice board an
  August 2026 snapshot is not stale, it is about a different season, and the page says so
  above the recommendation. The test is crude on purpose — a season label spans two calendar
  years and a capture outside both is elsewhere — because `adp-freeze-rule` records what
  happened the last time this project inferred a season from a month.

The badge goes **on the button** rather than in a column beside it, since the button is
what the eye is already on; the panel under the table carries the full text, because "Out,
right Achilles, back ~April" and "Day-To-Day, sore calf" are the same badge and not remotely
the same pick.

**One caveat has to be read before any EV level is believed, and it is not this layer's to
fix.** The field drafts off ADP while our board is the model's own projection, so a
best-available roster reaches round 4 far more often than an ADP entry — `make bracket`'s
benchmark entry already reads `p_advance = 1.0` in all five tournaments. Every EV the room
prints inherits that. It is exactly what Gate C's error injection exists to price, and until
item 8 runs, **the room's EV is a ranking device and not money.** The page says so on screen.

---

## Data the layer needs, and where it comes from

- ~~**DK position eligibility** — `data/raw/team_rosters_*.csv` carries `POSITION` with
  DK-shaped dual eligibility (`G-F`, `F-C`, `C-F`, `F-G`)…~~ ✅ **Built 2026-08-09 as `make
  draft-pool`, and the validation reversed the assumption** — see below. `POSITION` is still the
  right source and is still complete for all 30 seasons with zero nulls; what was wrong is that
  its duals are DK-shaped.
- **ADP** — `data/features/adp_panel.parquet`, 15,012 rows. It *holds* nine seasons, but under
  `adp.training_rows` only **five** are point-in-time legal: 2014-15, 2022-23, 2023-24, 2025-26
  and 2026-27. **Both validation seasons survive**, which is the coverage that matters — but the
  "cheap widening" above gains one season rather than four. See below.
- **Schedule** — realized game dates from `data/processed/game_logs.parquet` for backtests;
  `ScheduleLeagueV2` for the production season, **once it is published**.
- **Tournament structure** — the two `dk_best_ball_tournament_*.csv` files, through
  `dashboard.economics`. Re-verify against the live 2026-27 contests before any backtest
  number is treated as load-bearing.

### 🔴 DraftKings is single-position, and this plan assumed otherwise

Measured 2026-08-09 by `make draft-pool`, which was told to validate the NBA.com → DK mapping
rather than trust it. It does not survive.

**Both DK boards print exactly one of `G` / `F` / `C` for every player — 1,640 rows across two
seasons, zero duals, zero slashes.** NBA.com hands a dual to 18.2% of rostered players; DK hands
out none. That is not a quirk of one file: it reproduces independently on both boards, and DK's
own label is **99.85% stable** across them (1 change in 667 shared ids), so the label is a
settled per-player attribute rather than something the board recomputes.

The two sources are therefore two *opinions*, not a coarse view and a fine one. Joined on the
persistent DK id through `adp_dk_id_map.parquet` — never on a name — against the contemporaneous
2025-26 roster (n = 501):

| | agreement |
|---|---|
| DK's letter lies inside NBA.com's position set | **92.61%** |
| DK's letter equals NBA.com's **primary** letter | **86.63%** |
| …restricted to NBA singles (`G`, `F`, `C`) | 90.98% |
| …restricted to NBA duals (`G-F`, `F-C`, …) | **100.00%** inside the pair, 67.03% on the primary |

Read the last two rows together. Where NBA.com commits to one letter, DK contradicts it **9.02%**
of the time. Where NBA.com says "tweener", DK always picks one of the two it named — but which
one is close to a coin toss, and on `G-F` it is 16 G against 19 F on 35 players. **Every
disagreement is between adjacent classes; there is not one G↔C swap in either board.**

**What ships.** `position` is a single letter for every row, because that is the shape DK uses:
DK's own where a board exists, NBA.com's primary otherwise. The alternative — granting both
letters of a dual — is better on one error and much worse on the other. It never *misses* DK's
letter (100% containment) but hands a second slot to the 18.2% NBA.com calls tweeners, and a
spurious eligibility inflates every lineup it touches. Primary-only misassigns ~13% of players
symmetrically, which is noise in *which* slot a player fills; dual grants systematic extra
flexibility, which is upward bias in every simulated score and in exactly the direction that
makes a strategy look profitable when it is not. **Between symmetric noise and optimistic bias,
take the noise.** `dual_g` / `dual_f` / `dual_c` carry the rejected convention anyway, so the
sensitivity run is a column swap — and they are NBA.com's set **union** the shipped letter,
because NBA.com's set on its own is not a superset of what ships. It names nothing at all for
the 2026 draft class (205 rows of the production board) and it names the *other* class wherever
DK contradicts a single NBA.com letter, so a bare swap would leave 219 players eligible nowhere
and 310 eligible somewhere they cannot play. Unioning keeps the difference to the one thing the
convention is about: whether a tweener gets his second slot. `assert_pool` pins it.

A *fitted* majority map was also rejected, deliberately: it scores 87.2% against the primary
rule's 86.6%, and the entire difference is flipping `G-F` to `F` on a 19-vs-16 split. Fitting a
coin toss on 35 observations to buy 0.6 points is not a map, it is noise with a lookup table.

**The caveat this leaves.** Backtest seasons get mapped positions (86.6% right) and the
production season gets DK's own (right by definition), so the backtest understates lineup fit
relative to the live board. That is the conservative direction — it makes the measured edge
smaller — but a strategy tuned on how *awkward* rosters are to fill is tuned partly on an
artifact.

**Is validating against the boards a split read?** The only boards that exist are Oct-2025 (a
**test** season) and Jul-2026 (production), so this plan's own instruction cannot be followed
without touching 2025-26. It is not a violation: a position label is not a target-season outcome,
carries nothing about 2025-26 scoring, and `held_out.py` guards frames so a *score* cannot be
read, not roster attributes. The audit reports the same measurement against **train-only** roster
rows so the argument is checkable rather than asserted — restricted to rosters through 2021-22
the 2026-27 board reads **95.42%** containment and **88.55%** primary agreement against the
contemporaneous 92.61% / 86.63%. The finding does not come from the held-out rows and does not
move when they are removed.

### What else the board build found

- **The pool is season-start rosters, not the roster CSV.** `team_rosters_<season>.csv` is a
  *current-status* snapshot — the 2025-26 file carries `HOW_ACQUIRED = "Signed on 03/04/26"` —
  so using it for membership would put February signings in an October draft pool. Membership is
  `team_context.season_start_roster` (first appearance inside the team's first 10 games), which
  is this project's existing point-in-time definition; the CSV is read for `POSITION` only, which
  is a static attribute rather than a season outcome.
- **2026-27's pool is the DK board itself**, 942 players, because that season has no game log.
  That is the authoritative answer rather than a fallback: DK's player pool *is* the draftable
  set. When `commonteamroster` is fetched for 2026-27, its `POSITION` nulls (unsigned and
  two-way) resolve through the same board and `position_source` records how many needed it.
- 🔴 **The 2026 draft class is kept, not dropped.** 162 board rows carry no `player_id` because
  they have never played an NBA game, and `adp_dk_id_map` correctly reports them as
  `no_nba_history` rather than as a join failure. They are nonetheless draftable and several go
  early — **AJ Dybantsa at ADP 41.8** is a fourth-round pick. Dropping them would break the draft
  simulator for a reason unrelated to whether the model can score them, because the field takes
  them at their ADP regardless and *who is still on the board at pick k* is what a snake draft
  turns on. They carry a negative surrogate id (`-dk_player_id`) that can never collide with an
  `nba_api` id, flagged `has_nba_id`. **How to price them is still open** and belongs to item 4.
- **132 players (1.0%) are dropped as unslottable** — no `POSITION` in any roster file and no DK
  board row, a coverage hole in the 1996-2007 files. Keeping them would be worse than dropping
  them: eligible at no class, a player can still be drafted, consumes a roster spot and can never
  be started, silently shrinking a 16-man roster. Worst season is 1996-97 at 26; both validation
  seasons lose at most 4.

### 🔴 Point-in-time costs four of the nine ADP seasons

This plan's coverage list was panel *presence*, not legality. Under `adp.training_rows` — a row
is filled only from a board observed at or before the season's first game, the qualifying
*observation* rather than the qualifying value — only **five** seasons survive: 2014-15, 2022-23,
2023-24, 2025-26 and 2026-27. In 2017-18, 2018-19, 2019-20 and 2024-25 *every* archived snapshot
postdates the first game, which `docs/adp-plan.md` already predicted for two of them ("2016-17
and 2024-25 are recoverable only from mid-season snapshots") without following it through to what
the point-in-time rule then does.

Two consequences. **Both validation seasons survive** — 424 ADP'd players over 912 pool rows —
which is the coverage the realized backtest needs. And **2024-25, a test season, carries no legal
ADP at all**, which item 10's risk readout has to account for: half the test window cannot be
drafted against a contemporaneous market at all.

---

## Gates

Every gate is judged on validation or on simulated truth. None reads the test split.

| Gate | Pass condition | Why this bar |
|---|---|---|
| **A** ✅ ⚠️ | the season simulator reproduces the **marginals it was built from**: season-total dk_pts distribution against `season_total_metrics.csv`, GP pmf against `stan_games_played_gp_pmf.csv`, and per-game bonus rate against `bonus_calibration.csv` | An assembly bug is silent. Every input head is already calibrated, so a simulator that misses a marginal it was handed has a wiring fault, not a modelling one. **Run 2026-08-09**: season totals pass (MAE 402.14 / 407.89 against 400.46, CRPS 280.49 / 281.03 against 287.26), games played passes (CRPS 9.6754 / 9.7829 against the head's 10.0057, bias +0.127 / −0.363), minutes spread passes given games played (322.05 / 319.32 against 302.75); the **bonus is +11% / +5% high and is traced out of the module** — on realized minutes the same draw reads 0.1535 / 0.1477 against 0.1559 / 0.1626, so it is the composition head's 1.8x game-level minutes over-dispersion. Two wiring faults were caught and fixed |
| **B** ✅ ⚠️ | simulated drafts reproduce the **observed ADP curve** — mean absolute rank gap under the 17.0-pick recalibration error, so the field model is no worse than the market proxy it consumes | The field model's only real calibration target. Failing it means the opponent model is not a field. **Run 2026-08-09**: passes wide, pooled **5.922** picks on the fit region and **9.082** over every ADP'd player. The caveat is the fit rather than the gate — a **constant** rank noise is not identified by a mean-ADP target at all (the whole sd 0–30 grid spans **0.110** / **0.117** picks and the two seasons disagree about its optimum inside that band), while a rank-dependent shape is, with both validation seasons landing on **sd = 4.00** independently. What separates the shipped arm from a zero-noise field is not the 0.044-pick margin but `field_diversity`: at sd = 0 two drafts share **100%** of a seat's roster |
| **C** ✅ ⚠️ | the **error-injected** simulated world reproduces the model's measured out-of-sample miss: availability CRPS ≈ 10.006 games, component R² in 0.81–0.95, season-total MAE ≈ 400.5 dk_pts | Without this the sweep cannot price ADP, exposure caps, or any other hedge against model error. **Run 2026-08-09, and the premise did not survive contact**: the *uninjected* world already reproduces the miss in magnitude (MAE **414.97 / 398.96** against 400.46; availability CRPS **10.0935 / 9.9387** against 10.0057) and is *harder* than reality on the other two rows. What it gets backwards is the two rankers' relative standing — the market leads the model by **+0.052 / +0.016** Spearman on realized data and *trails* by **−0.111 / −0.118** in a world drawn from the model's posterior. So the injection rotates the error onto the market-visible direction at fixed magnitude, with `g` solved from the MAE bar and `ρ` from the skill gap; both land exactly, and `ρ` is corroborated to within 0.1 by an independent route. Season-total R² (**0.543 / 0.573** against 0.7073) and rate R² (**0.528 / 0.581** against 0.81–0.95) are **not met, in the conservative direction** |
| **D** 🔴 | the sweep selects **materially different** rosters for the two tiers | If the $20 and $52 strategies converge, either the objective is not doing its job or the tier difference is smaller than the economics imply. Either way it needs to be known before entering. **Run 2026-08-09: it fails, and the finding is the tier difference.** Both tiers select the same strategy, and in all six comparisons the cross-tier roster overlap sits inside the within-tier band — under a tier-blind ranking (0.480 against 0.517 / 0.396) *and* under a `bracket_ev` objective that reads each tournament's own pods and cash bands (0.713 against 0.704 / 0.792). The mechanism is Round 1: both cut 2 of 12 in the only zero-consolation round, so 83% of paths end identically and the tail economics move the objective very little. **One board serves both tiers** |
| **E** ✅ | in-draft recompute **under 1.0 s** at `n_sims = 500` on the full remaining pool | The 30-second clock. Failing it drops the draft room to ranking-submission mode. **Run 2026-08-09**: passes with 5× of headroom — **112 ms mean, 185 ms p95, 200 ms max** over 359 candidates, against 1,000 ms. Both levers the plan named were needed and both are exact rather than approximate: `n_sims = 500`, and best-7-by-slot as **one matroid exchange** per candidate. The same page also reproduces the symmetric-field null's P(advance) to **2e-8** |
| **F** ✅ | measured edge, expressed as **lift in P(top 2 of 12)** over an ADP-drafted entry, is reported with a bootstrap interval against the break-even hurdle (+17.60% / +12.32%) | Not a pass/fail on the edge itself — a requirement that the number is quoted in comparable units with its uncertainty, rather than as a point estimate. **Run 2026-08-09.** The ADP baseline is exact rather than estimated (`n_advance / pod_size` = 0.166667, reproduced by the injected field to 5e-11), so the interval on the lift is the interval on `P(advance)` shifted. Simulated: **+0.211 [+0.143, +0.282]** and **+0.199 [+0.091, +0.328]**; realized on two seasons: **+0.235 / +0.019** and **+0.172 / −0.026**, i.e. not distinguishable from zero. ROI **+21.5 [+5.6, +46.8]** against +17.60% and **+2.98 [+0.83, +5.54]** against +12.32%, both carrying the three caveats under "What was built" |

---

## Config

```yaml
sim:
  posterior_draws: 1000          # thinned across the whole posterior, never sliced
  fit_window: train              # NOT train_val — see "One window per consumer" below
  n_sims: 2000                   # strategy sweep; sim s uses posterior draw s % 1000
  n_sims_draft: 500              # in-draft; a ranking, not a level
  scoring_periods: 20            # R1's 17 weeks + three double weeks
  pod_size: 12
  seed: 0

  tournaments:
    600k_shootaround: {entries: 10}
    20k_spin_move:    {entries: 4}

  n_sims_draft: 500              # in-draft; a ranking, not a level
  pod_size: 12

  draft_room:
    field_drafts: 100            # 12-seat pods in the cached reference field. 100 gives
                                 # `20k_spin_move`'s null to -0.03%; `600k_shootaround`
                                 # reads -17% and does NOT converge — 300 pods buy -7.7%
                                 # and make the RANKING less stable, not more
    objective: bracket_ev        # or `p_advance` / `lineup_value`; all three ship
    seat: 0                      # DK randomizes it, so the page owns it in practice

  field:
    adp_source: dk_recalibrated  # docs/adp-plan.md: never the raw consensus
    noise_model: null            # fitted by Gate B — `tiered` beat `constant`
    rank_noise_sd: null          # fitted by Gate B, not chosen. Both read from
                                 # outputs/predictions/draft_gate_b.csv's selected row
    position_caps: {G: 8, F: 8, C: 3}   # DK's own autodraft defaults — MAXIMA, and they
                                 # imply no minimum: 8G + 8F + 0C seats no centre
    require_legal_lineup: true   # the separate guard that does. Not a DK rule
    need_weight: 0.0             # `adp_need`'s lean toward owed positions, in picks
    n_drafts: 400                # drafts per Gate B grid point
    composition:                 # UNCALIBRATED, per tier, resolved through `default`
      default: {adp: 1.0}

  error_injection:
    enabled: true                # Gate C. Disabling it changes the question being asked
    targets: [availability_crps, component_r2, season_total_mae]
```

The game-length head sits under `stan:` with the other heads, not under `sim:` — it is a
fitted model, and only its draws are a simulator input:

```yaml
stan:
  game_length:
    # Depth counts 33,433 / 1,821 / 250 / 37 / 5 over 35,546 regular-season games; nothing
    # past 4OT. Continuation is 0.1382 / 0.1438 / 0.1190 by depth, so a geometric in depth
    # covers 2OT-4OT for free and there is no depth dependence to fit.
    arms: [floor, season_trend, season_trend_matchup]
    # Regular season only, matching every other fitting frame and the contest window.
    season_type: regular
    # Half-normal scale on the beta-geometric concentration, mirroring games_played.
    kappa_scale: 25.0
    # Added at build time: the matchup arm's cells. A beta-binomial at one game per row is
    # a Bernoulli and its dispersion is unidentified, so the per-game gap is binned.
    matchup_bins: 10
```

`season_trend_covered` is not listed and does not need to be — the sweep inserts it whenever
`season_trend_matchup` is present, because the matchup arm is uninterpretable without a
same-window control and an optional control is a control that gets switched off.

---

## Tests

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.

- **lineup selection** ✅ `tests/test_bracket.py` — best 7 by slot from a hand-built 16 with
  known scores, including the UTIL fallback and a dual-eligibility player who must be placed
  to maximize the total rather than greedily. The dual case is written as a *comparison*: a
  first-fit assigner is implemented in the test file so the margin (186 against 188) is
  visible rather than asserted. A companion pins that a roster with one forward seats only
  six, since that failure returns a plausible-looking score.
- **tie-breaks** ✅ — two entries with identical round totals and different weekly maxima, then
  identical weekly maxima and different best-player scores, cascading; plus that only the
  *tied span* is re-ordered and the entries around it are untouched, and that the per-player
  level is all-or-nothing across the two populations.
- **bracket arithmetic** ✅ — the advance chain reproduces `economics.advance_table` field
  sizes exactly for all five tournaments, and the symmetric-field null returns `-rake` for
  all five. Wildcards are forced by asking for a bigger next round than the pods deliver,
  since the captured chains all divide exactly.
- **scoring-period bucketing** — a postponed game scores in the period it is played; the NBA Cup
  final scores nowhere; every game date lands in exactly one period.
- **the split guard** — the strategy sweep raises if it reaches the test seasons, pinned the way
  `component_rates`' guard is pinned.
- **posterior round-trip** ✅ `tests/test_posteriors.py` — a head's saved draws and design
  recipe reproduce its stored predictions to numerical tolerance, without refitting. Run with
  the heads' draws *injected* rather than sampled, because the fit is exactly the part the
  gate is not about and injecting exercises the identical prediction path in milliseconds on a
  machine with no CmdStan. Covers both drift modes — a dropped step (the recipe names the
  missing column) and a corrupted fitted value (every column present, every shape right, and
  only the numbers wrong), which is the silent one. `make posteriors` runs the same
  `roundtrip()` on every real head and raises, so the fitted version is a build gate rather
  than an untested claim.
- **draw order** — `fga → fg3a|fga → fg2a → makes` is materialized in that order; reordering it
  must fail loudly.
- **the game-length draw** ✅ `tests/test_stan_game_length.py` — 20 tests, no sampler. The
  three that matter are about the *draw* rather than the fit, because its failures are all
  silent: one length **per game** and not per team-game (the shape is pinned); the season
  frailty **shared across the slate**, checked in both directions since a frailty drawn per
  game reproduces the marginal and shows binomial spread; and the 48/53/58 grid. A fourth
  pins that handing the draw a *cell* frame raises instead of truncating — the wiring error
  that would otherwise produce a plausible season from the wrong frame — and a fifth pins
  that `posterior_inputs` (from the pickles) equals `draw_inputs` (from live heads), since
  two producers of one contract is how a simulator ends up drawing from something subtly
  different from what was fitted. `tests/test_stan_composition.py` pins the *absence* of
  `fit_ot_tail`, because the failure there is reintroduction.
- **the draft room** ✅ `tests/test_draft_room.py` — 22 tests, no tensor and no field draft.
  Three are load-bearing and the rest are guards. **The exchange identity** is checked
  against `bracket.best_lineup` itself over 400 random rosters, single-position and
  dual-eligible both, since the whole latency argument rests on it being exact rather than
  close — and a wrong threshold still returns a ranked table. **The symmetric-field null** is
  reproduced on a hand-built two-round tournament whose answer is arithmetic, which catches a
  wrong survivor reweighting, a wrong plotting position and a payout indexed off by one at
  once. **The plotting position** is pinned directly, because it is the one term that moves no
  shape and every level. Plus: a base roster that cannot seat seven raises instead of pricing
  against six; `survival` reads each sim against its own field, since pooling would let a
  high-scoring sim's field beat a low-scoring sim's entry; the completion fills the slate
  first and never names a player already gone; one click advances the snake and `replay`
  rebuilds the state from the log; and an unpriceable player is excluded rather than valued
  at zero. `tests/test_dashboard.py` pins the import exemption in both directions — one named
  file, bounded at `src.sim`.

---

## Open questions and risks

- **The 2026-27 schedule is not published.** Blocks the production run only. Poll `ScheduleLeagueV2`.
- **Field skill by tournament tier is unmeasured**, and will bias any opponent model that
  assumes one field composition across a $20 and a $52 contest. Since 2026-08-09 it is at
  least *visible*: `sim.field.composition` is per tournament with a `default`, and
  `make draft-sim` writes the resolved mix and seat assignment per tier to
  `draft_field.csv` with `calibrated = False` on every row.
- 🎯 **Real 12-entry pick logs are the missing calibration, and they are on the same
  clock as the October board.** Gate B measured that a mean-ADP curve constrains the field's
  *mean* and says essentially nothing about its noise — 0.110 to 0.117 picks across the whole
  sd grid. Twenty cheap pods (~$40, 3,840 picks) with the contemporaneous DK board captured
  alongside would identify the noise level from pick *variance*, test the tiered shape, expose
  positional runs — the largest dynamic ADP-plus-noise cannot generate, and the reason
  `AdpNeedAware` ships switched off — and reveal the autodraft share the tier risk above turns
  on. Not backfillable: a pod not entered is gone. See "Real pick logs are the calibration
  this layer is missing".
- **`make bracket` still drafts its field with `placeholder_field`.** `src/sim/draft.py`
  exists and Gate B passes, but repointing the bracket invalidates every figure in
  `bracket_*.csv` and needs a full re-run, so it is sequenced with item 8 rather than done
  in passing. The consequence today is that the bracket's field ranks by the **simulator's own
  projection** rather than by market ADP, which makes it stronger and more correlated than a
  real pod.
- ~~**Gate E is not free.**~~ ✅ **Closed 2026-08-09.** The marginal-lineup-value recompute
  did measure 0.76 s mean and 1.5 s max, over the bar before the bracket EV went on top. The
  first lever named — best-7-by-slot without a re-solve — turned out to be available in an
  **exact** form rather than as a partial sort, and it alone is worth the whole margin: Gate
  E now reads **112 ms mean and 200 ms max**. The second lever, not re-ranking the deep tail,
  was **not needed and is not implemented**, so the room ranks every legal priceable player
  on every pick. That is the better outcome — a screen would have been a heuristic over an
  objective whose top few candidates are already inside its own field noise.
- 🔴 **The draft room's EV level is optimistic for the same reason the bracket's benchmark
  entry is.** The reference field drafts off ADP while the room's board is the model's own
  projection, so a best-available roster reaches round 4 far more often than an ADP entry
  does. Gate C's error injection is what prices it; until item 8 runs, the room's EV is a
  ranking device rather than money, and the page says so on screen.
- **The DK ADP capture deadline is live** — an early-to-mid October 2026 board is the second
  anchor the recalibration needs, and it cannot be backfilled. See `docs/adp-plan.md`.
- **No-redraft risk is not in the ranking.** A Round-1 pick who is traded or suffers a
  season-ending injury is frozen dead weight through Rounds 2–4. The simulator captures this if
  the spell process runs the full season, but whether the *ranking* should carry an explicit
  durability term distinct from expected value is open.
- **Mid-season trades are not modelled** — 13.6% of players appeared for 2+ teams in 2023-24.
  A known limit inherited from the prediction layer.
- **N = 2 realized seasons** is the ceiling on the honest edge estimate. Named, not solved.
- ~~**The four simulator inputs emit no `train` row.**~~ ✅ **Closed 2026-08-08.** They now
  emit all three windows off one shared `FIT_WINDOWS` in `src/data/preprocess.py`. The
  consumer-side obligation remains and is not enforceable by the artifact: a backtest scored
  on 2022-23 / 2023-24 must *ask* for `train`, and the programmatic defaults are still
  `train_val`. `src/sim/season.py` is where that has to be gotten right.
- **The fitted `sigma_u` is a compute booking, not a task.** Gate A measures ~6.3 h per
  random-effect arm at the pilot window and ~38 h at the full one, and the short-chain probe
  did not converge — so the real figure is higher. Item 3d's capability is built and its
  fallback (σ = 0.450, estimated on `train`) is shippable, so nothing downstream is blocked;
  what is blocked is *retiring the marginal minutes head*, which needs a converged
  full-window fit. Whoever books the machine should run the pilot ladder first and read the
  arm ordering, per "The ladder, and the pilot that keeps it affordable".
- 🔴 **The composition head is 1.8x too dispersed at the game level**, measured by
  `make simulate-season`'s Gate A: implied overdispersion **7.70** from the head's own draws
  on realized availability against **4.22** realized, at sigma = 0 so the injected
  player-season effect is not the cause. It over-produces the double-double bonus by ~11%
  and inflates every star's single-game ceiling, which is the statistic a 2-of-12 pod is most
  sensitive to. It does **not** show up in the head's own per-team-game CRPS or PIT, which
  are statements about the allocation rather than about a player's spread around his own
  season mean.
- **The 2.43x ten-game block inflation is not consumed**, and cannot be until the row above
  is settled: the simulator reads 1.40 / 1.52, and adding the ~3-line block term would put
  more variance into a minutes draw that is already too wide at the game level.
- 🔴 **153 of 539 rostered players are in the minutes allocation but not in the tensor**, for
  want of a component-head design row. They cannot be dropped (the allocation is zero-sum)
  and cannot be scored. Pricing them is the same open question as the 2026 draft class —
  and since 2026-08-09 it has a price tag: `make strategy-sweep` measures on every run that
  the ADP field drafts **1.2556** and **1.1861** of them per sixteen-man entry across the two
  validation seasons, with **73.06%** and **74.72%** of entries holding at least one, so every
  one is a roster spot scoring zero all season. Left in, it read as model edge
  worth more than every strategy axis combined (a pure-ADP entry of our own scored
  `P(top 2 of 12) = 0.285` against an exact 0.1667). The sweep restricts the board to
  priceable players on **both** sides as a stopgap, at the cost of changing who is on the
  board at pick *k*; a floor projection for those players would remove the stopgap.
- 🔴 **The error injection perturbs the season-level rate and cannot perturb the model's
  knowledge of the distribution's shape.** Gate C's three named targets are all level
  statistics, so nothing in the injection touches what the model knows about the weekly
  distribution, the double-double threshold or the cross-component copula — and the shipped
  objective uses all three, in a world drawn from the same joint. Every simulated lift in
  `make strategy-sweep` is therefore an **upper bound**. Closing it means a target expressed
  on the *shape* rather than on the level; nothing on disk currently supplies one.
- **`ρ`, the injection's market-visibility parameter, is measured on the two seasons the
  sweep scores.** Legal (validation is the split selection may read) and consistent with how
  the other four simulator inputs are calibrated, but every `α` the sweep selects inherits
  the sampling error of two seasons of ~200 priced players. A third ADP-legal season is
  2014-15 and it is the same cheap widening "Realized truth" already prices.
- **`α`'s *location* is not identified and its *sign* is.** Every blend arm beats the pure
  model and pure ADP loses to it, resolved; but 600k peaks at α = 0.15 and 20k at α = 0.70,
  with α = 0.30 and 0.50 unresolved against zero at 600k. This is Gate B's `rank_noise_sd`
  finding one layer up — a flat objective near its optimum — and real pick logs would not fix
  it, because the flatness is in the payout rather than in the field.
- **Tournament structures may change** for the live 2026-27 contests. Re-verify the metadata
  and prize CSVs before treating any backtest result as load-bearing. `make bracket`'s
  symmetric-field null is the check that will catch a bad transcription — it already caught
  one — and it should be run against the 2026-27 structures the day they are captured.
- **`600k_shootaround`'s ROI is not estimable at any affordable simulation budget**, and
  `make bracket` now says so rather than printing a point estimate. It reaches round 4 on
  0.139% of entries and pays 10,000x at the top of it, so the expected number of top prizes
  in a run is order one and the ROI swings by tens of percent between runs — the same
  observation `select-on-p-advance-report-roi` makes, arriving before the sweep it governs
  exists. P(reach round r) *is* resolved at every round, which is what the null is read on;
  the ROI carries a bootstrap interval and a `roi_covers_analytic` flag.
- **Scoring a real-sized field is the layer's first genuinely slow step.** 35,280 entries x
  20 periods x 250 sims is 176M lineup solves per season, and the field's per-period array
  is 0.71 GB. It is already amortized across the five tournaments — one draft and one
  scoring pass per season, each contest taking the prefix it needs — and the same trick is
  what item 8 needs: the sweep cannot rebuild the field per strategy, so the field's period
  scores want to be built once and reused across the whole table. `sim.bracket.n_sims` is
  the memory knob rather than a precision one.
- **`600k_shootaround`'s per-entry ROI still will not resolve, and that is structural.**
  The dealt bracket removed every modelling bias, but not the Monte Carlo problem: an entry
  reaches round 4 on 0.139% of tries and the top step is 10,000x, so a subsample's ROI
  swings by tens of percent. The *aggregate* identities are exact, which is what makes the
  wiring checkable at all; the per-entry number carries a bootstrap interval and
  `roi_covers_analytic`. Read the null on P(reach round r), which is exact by construction.
- **Ties are common rather than exotic, because dk_pts are quarter-point multiples.** A
  17-week round total lands on a coarse grid, so a large field ties constantly and the
  cascade is a hot path rather than a formality — which is why `rank_within_pod` re-orders
  only the tied span and never the whole row.
- **This doc is not yet in `make docs-audit`.** Add it once it carries measured figures rather
  than specification — see `docs/docs-audit.md`.

---

## The build order — one session per item, with its opening prompt

Ordered so the **live-draft path closes at item 7**. Items 2, 3 and 3b depend on nothing and can
run in any order alongside item 1. **Item 3c must follow 3b** — both were expected to edit
`stan_composition`, though 3c in the event did not — and both must land before item 4, which
imports whatever they settle. **Item 3d follows 3c** and does edit that head, so nothing else
may be in flight on it. Items 1, 2, 3, 3b, 3c, 4, 5, 6 and 7 are done — **the live-draft path
is closed and a draft can be run today**; **item 8 is next**, and it is what makes the numbers
the room prints mean money rather than only rank.

**Item 3d's capability landed 2026-08-09 and its full-window commitment did not** — the
distinction is spelled out under item 3d itself. Item 4 is **not** blocked by that: it
consumes minutes from the composition plus a per-player-season effect, and the injection's
σ = 0.450 (estimated on `train`) supplies one today without a refit.

**Paste the framing prompt below first, then the item's own prompt.** The framing carries
everything common — what to read, the conventions, the split rule, the deadline, and how to
verify — which is why no item prompt repeats any of it.

### The framing prompt — paste at the start of every session

> We are building the tournament simulation and drafting layer of this project. It is the last
> layer; the Bayesian prediction heads underneath it are all fitted and shipped.
>
> Read first, in this order: `CLAUDE.md`, `docs/project-spec.md`,
> `docs/dk_best_ball_rules.md`, and `docs/simulations-plan.md`. The last one is the spec for
> this work — its "What was settled on 2026-08-08" table, its artifact contract, and its gates
> are binding, and if you think one of them is wrong, say so before building rather than
> quietly deviating.
>
> Five things that are easy to get wrong here:
>
> 1. **The split.** Selection reads validation (2022-23, 2023-24) and nothing else.
>    `src/models/held_out.py` makes the test seasons a capability that raises when read — go
>    through `selection_split`, never around it. This project's one recorded discipline failure
>    was a gate specified with test figures as its bars.
> 2. **The deadline is real.** The 2026-27 season is a production run, not a research exercise,
>    and drafts happen before October. Prefer the working version now over the elegant version
>    later, and say plainly when you are trading one for the other.
> 3. **Conventions are not negotiable.** Entry points run as `python -m src.<module>` from the
>    repo root with a matching `Makefile` target in `.PHONY`; config is
>    `cfg = yaml.safe_load(open("configs/default.yaml"))` duplicated in `__main__` blocks and
>    must not be refactored into a helper; `mkdir(parents=True, exist_ok=True)` before every
>    write; print `f"... {n:,} ... → {dest}"` progress lines; tests use plain `assert` with
>    synthetic builders and no fixtures or classes.
> 4. **Never reimplement what exists.** `preprocess.compute_dk_pts` is the only DK scoring
>    function. `fetch._slug` / `_season_start_year` own season keys. `dashboard.economics` owns
>    every tournament structural number. Reuse, do not reproduce.
> 5. **Load-bearing decisions get registry entries.** Add or update `dashboard/decisions.py`
>    alongside the edit to `docs/simulations-plan.md`, using the closed status vocabulary — a
>    reversal becomes `withdrawn` and keeps its entry.
>
> Always use `.venv`, never system Python. Verify before you finish:
> `make test`, `make docs-audit` (a gate — it must exit zero) and `make dashboard-audit` (a
> report). Report what actually happened, including failures.
>
> Your task for this session:

### 1. `make posteriors` — persist the fitted posteriors ✅ built 2026-08-08

> Read `CLAUDE.md`, `docs/project-spec.md` and `docs/simulations-plan.md` (the section "The
> prerequisite nobody noticed"). Build `src/models/posteriors.py` + `make posteriors`, which
> loads each fitted Stan head and writes one artifact per head to
> `data/features/posteriors/<head>.pkl`: thinned coefficient draws (1,000, via
> `stan_utils.thin`, spread across the whole posterior — never sliced off the front), the design
> recipe needed to score an arbitrary frame (feature list, fitted scaler, spline knots,
> imputation means, selected variant), and provenance (fit window, CmdStan version, git SHA,
> timestamp). `stan_composition._checkpoint` is the precedent for the mechanics — generalize it
> from a crash artifact to a contract and give it a loader. Cover every head in `make stan` plus
> `stan-games-played`. Add a round-trip test: saved draws + recipe reproduce that head's stored
> predictions to numerical tolerance **without refitting**. Default the fit window to
> `train_val`. This is the one target that needs CmdStan; everything downstream is numpy.

### 2. `make scoring-periods` — the DK week grid

> Read `docs/simulations-plan.md` ("Scoring periods are NBA weeks") and
> `docs/dk_best_ball_rules.md`. Build `src/features/scoring_periods.py` + `make scoring-periods`
> writing `data/features/scoring_periods.parquet`: one row per (season, game_id) with its
> scoring period and tournament round. Read the NBA week grid from
> `nba_api.stats.endpoints.ScheduleLeagueV2` (`weekNumber` / `weekName`) for seasons that have
> it and derive it from realized `game_date` in `data/processed/game_logs.parquet` otherwise;
> map DK's published round windows onto that grid. Own three edge cases once: a postponed game
> scores in the period it is **played**, the NBA Cup final does not score, and the all-star gap
> breaks week adjacency. Assert every game date lands in exactly one period. Verify Round 1 = 17
> weeks and Rounds 2–4 = one double week each. Flip the `scoring-periods-are-nba-weeks` registry
> entry from `open` to `built`.

### 3. `make draft-pool` — the board, with DK position eligibility ✅ built 2026-08-09

**13,105 player-seasons over 31 seasons**, 942 of them the 2026-27 production board. The
validation it was told to run **reversed this plan's position assumption** — DK is
single-position — and turned up two more things worth carrying forward: the 2026 draft class has
no `player_id` and must be kept anyway, and point-in-time discipline costs four of the nine ADP
seasons. All three are written up under "Data the layer needs" above.


> Read `docs/simulations-plan.md` ("Data the layer needs") and `docs/adp-plan.md`. Build
> `src/features/draft_pool.py` + `make draft-pool` writing `data/features/draft_pool.parquet`:
> one row per (season, player) with team, **DK position eligibility** (G / F / C flags, duals
> allowed), ADP from `adp_panel.parquet`, and the prior-season key the model heads need. Take
> positions from `data/raw/team_rosters_*.csv` (`POSITION`, 30 seasons, zero nulls, already
> DK-shaped duals) and **validate the NBA.com → DK mapping** against the two
> `data/raw/dk_draft_rankings/*.csv` boards, which carry DK's own positions for 698 and 942
> players — report the disagreement rate rather than assuming the mapping. Fall back to the DK
> board for the 2026-27 rows whose `POSITION` is null (unsigned / two-way). Reuse
> `adp_dk_id_map.parquet` for the id join; never name-match where an id exists.

### 3b. `make stan-game-length` — the overtime head ✅ built 2026-08-09

Added 2026-08-08, after the rest of the order was written; built 2026-08-09. The measured
outcome is under "The second prerequisite" above — both gates pass, the matchup arm is the
predicted null, and one of the plan's stated *reasons* turned out to be backwards. The
opening prompt is kept below as written, because the record of what a session was asked for
is worth as much as the record of what it found.

> Build `src/models/stan_game_length.py` plus a `make stan-game-length` target: the Bayesian
> game-length head, with a full posterior, that the forward simulator draws from.
>
> Read "The second prerequisite: game length is a random variable forward, not a lookup" in
> `docs/simulations-plan.md` first — it carries the measurements, the ladder and the sizing.
>
> Why it exists: every backtest so far reads `game_length` from
> `data/features/game_length.parquet` because the games already happened. In a forward
> simulation nothing knows how long a game will be, and both minutes heads need it — the
> composition head allocates exactly `5 × game_length` per team-game and the marginal head
> uses it as binomial trials.
>
> A point-MLE version already exists in the wrong module and the wrong form:
> `stan_composition.fit_ot_tail` / `sample_game_length` (around line 1074), two floats fitted
> on train seasons. Replace it, do not duplicate it. Afterwards `stan_composition` imports
> this head — the same ordering constraint it already has on `stan_minutes` — and its own
> versions are deleted with a `withdrawn` registry entry recording the move.
>
> **No new `.stan` file.** Two existing sources cover it: `betabinomial_glm.stan` for whether
> a game goes to overtime (OT games out of games, collapsed to season cells) and
> `betageometric_duration.stan` for how many (one row per OT game's depth). The
> beta-binomial's overdispersion parameter is the trend-versus-wander answer measured rather
> than asserted, and the beta-geometric's Beta frailty is the natural fix for the plain
> geometric's one miss — it over-predicts 3OT+ by 3 games in 2,460.
>
> Arm ladder, against a mandatory no-fit floor:
>   1. floor — pooled league rate, no fitting;
>   2. season trend — the measured logit slope of −0.00893 per season (se 0.00264, z = −3.38).
>      Expected to win, and it would be the second head in the project to ship a season term;
>   3. + matchup — `|prior-season net rating difference|` between the two scheduled teams,
>      from `team_estimated_metrics_*.csv`, legal pre-season given the schedule. Speculative;
>      expect a null and record it as one.
>
> Facts already measured, so do not re-derive them: 35,546 regular-season games; depth counts
> 33,433 / 1,821 / 250 / 37 / 5 with nothing past 4OT; continuation probability 0.1382 /
> 0.1438 / 0.1190 by depth, so a geometric in depth is right; season dispersion 1.11×
> binomial; fitted rate 0.0670 in 1996-97 falling to 0.0526 in 2025-26 and 0.0521 for
> 2026-27, against the train window's 0.0608.
>
> **Size it honestly and do not let it grow.** The trend is worth about 0.09% of total
> minutes, so this is not a mean-effects story — it is a tail story. Overtime is where 40+
> minute games come from (1,650 player-games exceed 48 minutes, maximum 63.0), and under a
> best-ball weekly max plus a threshold bonus an OT frequency 17% too high inflates every
> star's simulated ceiling, which is the statistic a 2-of-12 pod is most sensitive to. The
> ladder above is the whole scope. This is roughly 30 collapsed rows and 2–4 parameters —
> the cheapest head in the project, seconds of sampler time.
>
> Gate: reproduce the held-out OT-class counts at least as well as the incumbent point
> estimate does (2,322 / 120 / 18 / 0 observed against 2,310.5 / 128.4 / 18.1 / 2.96
> predicted over 2,460 games), and beat the no-fit floor on held-out log-likelihood per game.
> Selection reads validation only.
>
> Then register it in `make posteriors` so its draws land in
> `data/features/posteriors/game_length_*.pkl` alongside the other eighteen heads, and add a
> `sample_game_length(rng, n_games, draws)` entry point that the simulator calls **once per
> game, shared by both teams** — overtime is a property of the game, and drawing it per
> team-game would destroy a correlation a same-team stack depends on.

### 3c. `make minutes-unification` — does the composition retire the marginal head? ✅ built 2026-08-09

Added 2026-08-08 — see "The third prerequisite" above, which now carries the measured
verdict. **The composition lost at the season unit, so both heads ship**, no year block was
ported into `composition_glm.stan`, and the season-terms ablation was not re-run for it. The
one thing item 4 must carry forward is that the simulator's minutes draw takes the
allocation from the composition and the season-level spread from `stan_minutes`.

The sequencing note was right and the collision it predicted did not happen: 3b landed
first, `stan_composition` gave up `fit_ot_tail` there, and this item turned out not to touch
that module at all — the gate lives in its own module and reads persisted posteriors.

> Settle whether `stan_minutes` is still needed in the production chain, or whether
> `stan_composition` supersedes it.
>
> Read "The third prerequisite: is the marginal minutes head still needed?" in
> `docs/simulations-plan.md` first — it carries the audit of what `stan_minutes` exports, who
> consumes it, and which of `README.md`'s three claims survive.
>
> **Check first that item 3b (`make stan-game-length`) has landed**, since it also edits
> `stan_composition` and the two changes will collide otherwise.
>
> Two of the three things the README says `stan_minutes` owns do not hold, and they are
> already established — do not re-derive them:
>   - `game_level_dispersion` (stan_minutes.py:368) is a pure DATA measurement. It reads
>     targets and lengths, computes each player-season's own realized share as mu, and fits a
>     dispersion. The StanMinutes object never appears, so the 4.65x figure does not depend on
>     the fit at all.
>   - The composition's selected arm already fits a role-graded game-level dispersion (rho
>     0.1768 fringe to 0.0855 star over four bins) which is what governs a draw FROM the
>     composition. Different parameterization from the 4.65x, same kind of quantity, and only
>     one can govern the draw.
>
> **The gate is the untested claim**: score the composition's season-total sums against the
> minutes head's season-total predictions on the same validation rows and the same metric
> (CRPS and MAE in minutes, plus PIT KS). The composition's per-game predictions sum to a
> season total by construction; whether that beats the season-collapsed head at the season
> unit has never been measured. Write the comparison to
> `outputs/predictions/minutes_unification.csv`.
>
> If the composition WINS OR TIES:
>   - add the optional year block to `src/stan/composition_glm.stan`, copying the `S = 0`
>     device verbatim from `betabinomial_glm.stan` — zero-length `year_z` and `sigma_year`, so
>     `S = 0` reproduces the current posterior EXACTLY, and add a test pinning that;
>   - re-run the season-terms ablation for the composition arm. `season_terms` selected the
>     `year` arm for `min` (val MAE 199.03 against base 199.72, sigma_year 0.0231) and it is
>     the only head in the project shipping a season term, worth +10.4% on a 15-man roster's
>     season-total sd. That spread is what a 2-of-12 knockout is decided on, so the era
>     correction must survive the move rather than be dropped with it;
>   - retire `stan_minutes` from the production chain and from `make posteriors`' head list,
>     keeping it as a FROZEN comparator artifact so the composition's -6.06% headline stays
>     reproducible without a refit. Do not delete the module — `stan_composition` imports
>     `StanMinutes`, `beta_shapes` and `minutes_variants` for its `comparator/val` arm.
>
> If the composition LOSES at the season unit:
>   - keep both, and replace README.md's "the two heads compose rather than compete" with the
>     measured reason. As written that sentence asserts three things and only one holds.
>
> Either outcome: the simulator draws minutes from the composition, and 4.65x moves from
> "simulator input" to "diagnostic". Update the season-simulator input table, the registry
> entries `minutes-head-supersession-is-open` and `game-level-dispersion-is-not-a-fit`, and
> the README's minutes section.
>
> Cost is not an argument either way and should not drive the decision: `make stan-minutes` is
> 0.341 h across four arms against the composition's 9.92 h, so retiring the target saves
> twenty minutes.

### 3d. `make composition-effects` — fit the player-season effect ⚠️ capability built 2026-08-09, commitment open

Added 2026-08-09, out of item 3c's follow-up. **The capability landed and the commitment did
not**, and the two should not be conflated:

- ✅ **Built and pinned.** The optional `sigma_u` block in `composition_glm.stan` with
  `U_n = 0` nesting the shipped head *exactly* (Gate P1, checked as an identity),
  `PlayerSeasonTerm`, the team-context block and its join, `make composition-effects`,
  `make posteriors` carrying `sigma_u`, and `player_season_effect_sweep` reading it.
- ✅ **Measured.** The four deviation correlations re-derived into an artifact; the team
  block worth **+1.2 points of R²** on a deviation that is 7.5% predictable; Gate A's cost;
  and `sigma_u` **0.4776** from a one-season fit, replicating the injection.
- ✅ **The fallback shipped.** σ estimated on `train` is **0.450** and ties the marginal head
  on validation, so the schedule risk is gone.
- ⏳ **Not done.** The pilot ladder has not been run to completion and no full-window arm
  exists, so Gates P2 and P3 are not yet answered by a *fitted* head. At ~6.3 h per
  random-effect arm at the pilot window and ~38 h at the full one, that is a compute booking
  rather than a session.
- ⛔ **Gate P4 is not being run, and that is a decision rather than a shortfall.** `ps_team`
  was cut from the ladder on 2026-08-09, leaving `base → ps → team` at ~7.5 h against ~15 h.
  P4 asks whether the team block reduces fitted `sigma_u`; the question it is a proxy for
  was already answered without a sampler, on 8,570 training player-seasons — the block is
  worth **+1.2 points of R²** on a deviation that is **7.5%** predictable in total. A 6.3 h
  fit cannot resolve that better than the regression did, so the arm buys a null that is
  already priced. `team` stays, because it is ~30 min under `dense_e` and prices the same
  block at the per-team-game unit, which the deviation regression does not reach.

**How many full-window arms this justifies is its own decision, and the answer is not
three.** `make posteriors` fits the composition once per fit window and the repo keeps three
(`train`, `train_val`, `full`), so turning `player_season_effect: true` on and rebuilding all
of them is ~114 h. The recommendation is **at most one, at `train`, and only if the pilot's
`ps` arm clears P2 and P3** — the injection at σ = 0.450 already ships a tie, the one thing a
fit adds that the injection cannot is shrinking `sigma_u` in response to features, and that
lever is measured at ~nil. What a converged full-window fit *would* buy is retiring the
marginal minutes head, which is worth real money on two strategy axes (see "Why the zero-sum
dynamic is a requirement"); that is one fit, not three.

It edits `src/stan/composition_glm.stan` and `stan_composition.py`, so nothing else may be in
flight on that head. Its full specification is "Fitting σ: the player-season effect as a Stan
parameter" above — read that first; the prompt below does not repeat it. Note that the target
name in the prompt is stale: the ladder runs as **`make composition-effects`**, for the
reasons under "What was built".

> Fit a per-**(player, season)** random effect in the minutes composition head, and sweep a
> team-context feature block alongside it in the same run.
>
> Read "Fitting σ: the player-season effect as a Stan parameter" in
> `docs/simulations-plan.md` first. It carries the design, the two non-obvious technical
> constraints, the arm ladder, all five gates and the fallback, and none of it is repeated
> here. Then read "The third prerequisite" above it for why this exists at all.
>
> **The one-line why.** `make minutes-unification` found the composition's season totals
> **4.68x too narrow** (predictive sd 64.65 against the marginal head's 302.75) and traced it
> to a missing parameter rather than a ceiling: injecting `sigma * z` per player-season per
> posterior draw into the *existing* posterior moves the sd to 239.45 and the CRPS to 142.17,
> which ties the marginal head. That injection tuned sigma on validation, which is the split
> it is scored against. This item fits it instead.
>
> **Do not refit the incumbent.** `betabinom_ot_graded`'s posterior is on disk at
> `data/features/posteriors/<window>/composition.pkl` and is the comparison baseline. Three
> new arms only: `+ps`, `+team`, `+ps_team`.
>
> **Run the pilot window before the full one.** `stan.composition.first_season = "2018-19"`
> cuts the training frame from 631,158 rows and 12,307 units to 97,587 and 2,204. Get the arm
> ordering there, commit the selected arm to the full window, and use the existing
> `probe_timing` Gate A to abort before a run you cannot afford. This head took exactly that
> path once already.
>
> Four things that are easy to get wrong, all of them measured rather than guessed:
>
> 1. **`U_n = 0` must reproduce the current posterior EXACTLY** — zero-length `u_z` and
>    `sigma_u`, the `S = 0` device copied verbatim from `betabinomial_glm.stan`. Pin it with
>    a test, the way the year block's nesting is pinned. This is Gate P1 and it is the only
>    thing separating a new parameter from a silently changed shipped head.
> 2. **Drop `dense_e` on the random-effect arms.** `StanComposition.fit` hard-codes
>    `metric="dense_e"`, chosen when the head had ~25 parameters. At 12,307 units that is a
>    12,332-square mass matrix — ~1.2 GB and a Cholesky per adaptation window. Use `diag_e`
>    and expect to give back part of the treedepth win the dense metric bought.
> 3. **Non-centered by default, but treat it as testable.** Rows per unit run median 57,
>    p10 11, minimum 1: the dense units would prefer centered and the sparse tail funnels
>    under it. Divergences are the diagnostic, and a centered arm is the first response.
> 4. **ADP is off limits here** — `docs/adp-plan.md` settles that it belongs to the strategy
>    layer, and spending it in the prediction layer would leave the blend weight sweeping an
>    axis the model had already absorbed.
>
> The team block is already built and point-in-time safe:
> `data/features/team_context_tierA.parquet`, headed by `role_crowding` (minutes-weighted
> archetype similarity, leave-one-out) plus `teammate_usage_max/sum/load` and `n_teammates`.
> It covers 11,928 player-seasons against a wider composition frame, so it will leave holes
> on exactly the rookies this head refuses to drop — reuse the `design_missing` indicator
> rather than minting a second one. **Also re-derive the four scratch correlations the plan
> doc quotes** (own lag-1 deviation −0.201, departed +0.041, arrived −0.063, net opened
> +0.088; joint in-sample R² 0.040 → 0.052) into the run's artifact, so they stop being prose.
> Expect the block to be worth about a point of R² on the deviation and record it as a null
> if it is not — the deviation is only ~5% predictable from pre-season information at all,
> and that is the finding, not a disappointment.
>
> **Gates P1–P5 are in the plan doc.** The two that decide the item: re-running `make
> minutes-unification` with the new posterior, the composition must **tie or beat** the
> marginal head at the season unit with team-sum error still exactly 0 (**P2**), while not
> regressing past the incumbent's per-team-game CRPS of **4.4945** (**P3**). `sigma_u` landing
> near the injection's 0.375–0.45 (**P5**) is a free replication; a wildly different value is
> a bug until explained. For `+ps_team`, the team block must **reduce** fitted `sigma_u`
> (**P4**) — a block that improves CRPS without shrinking it is explaining something else.
>
> **The fallback is not optional and belongs in the run's notes.** If the fit blows the budget
> or will not converge, ship the injection with sigma estimated on `train` instead of
> validation. `minutes_unification.player_season_effect_sweep` already implements the
> arithmetic and it is already measured to tie. Drafts happen before October.
>
> Then: teach `make posteriors` to carry `sigma_u` (and **only** `sigma_u` — the fitted `u_z`
> are useless for a season that has not happened, and the predictive draws a fresh `z` per
> posterior draw the way `YearTerm.shift` does; `posteriors._finish` currently raises on
> exactly this case and should gain the capability rather than be routed around), repoint
> `player_season_effect_sweep` to read sigma from the artifact, and **re-run `make
> minutes-unification` to re-take the supersession decision rather than re-arguing it**.

### 4. `make simulate-season` — the tensor, and Gate A ✅ built 2026-08-09

**Two 77 MB tensors, three of Gate A's four rows passing, and the fourth traced out of the
module.** The measured outcome is under "What was built, and what Gate A found" above. Three
things are worth carrying into item 5 and beyond: the composition head is **1.8x too
dispersed at the game level** against a player's own realized season share, which is a
property of the shipped head rather than of the assembly and is what over-produces the bonus;
the **2.43x serial-dependence input is deliberately not consumed** until that is settled,
because both live in one variance budget; and the residual copula had to be **inverted**
before use, since the frailty correlation that produces a given residual correlation is about
ten times it.


> Read `docs/simulations-plan.md` ("The output contract", "`src/sim/season.py`") and
> `docs/predictions-plan.md`. Create the `src/sim/` package and build `src/sim/season.py` +
> `make simulate-season`, writing `data/features/sim_tensor_<season>.npz`: a
> `player x scoring_period x sim` float32 tensor of dk_pts plus a `uint8` games-played twin.
> Assemble, do not invent — `stan_game_length.sample_game_length` once per game shared by both
> teams, `stan_games_played.sequences` for which games, `stan_composition.simulate_minutes`
> with the composition's **own** role-graded rho plus the 2.43x block inflation for minutes,
> `season_terms._draw_components` for the eleven heads in the order `fga → fg3a|fga → fg2a →
> makes`, the conditioned matrix from `residual_correlation.csv` as the copula, and
> `compute_dk_pts` for scoring. **Minutes come from BOTH heads** — `make minutes-unification`
> measured that summed composition draws are 4.68x too narrow at the season unit and that the
> team constraint forbids fixing it inside the composition, so the season-level spread has to
> come from `stan_minutes`; a board built from the composition alone is calibrated per game
> and far too confident per season, which is exactly what a 2-of-12 cut is decided on. The
> 4.65x game-level figure is a **diagnostic to check draws against, not an input**. Rules that
> must not be violated: draw never plug in; one shared `min` draw per player-game feeds all
> eleven heads; sequential structure goes on minutes only; **the posterior draw is the outer
> loop**, shared across all players, because that shared `β` is the cross-player correlation
> this layer exists for. **Gate A**: the simulator must reproduce the marginals it was handed —
> season-total dk_pts against `season_total_metrics.csv`, the GP pmf against
> `stan_games_played_gp_pmf.csv`, per-game bonus rate against `bonus_calibration.csv`, and the
> season-total minutes spread against `minutes_unification.csv`'s 302.75. A miss here is a
> wiring fault, not a modelling one.

### 5. `make bracket` — lineups, ties, advancement, payouts ✅ built 2026-08-09

**The measured outcome is under "`src/sim/bracket.py`" above.** Five things are worth
carrying into item 6. The weekly lineup is an **assignment problem** and first-fit
understates it one-sidedly. The **symmetric-field null** is this layer's gate and it caught
four separate defects — a tie-break asymmetry worth +68% on P(reach round 4), a
transcription error in the prize CSV, and two bugs in a parametric progression model that
was itself withdrawn. **Progression is dealt and ranked at the tournament's real field
size**, which is what made the survivor count and the payout total exact identities rather
than estimates. **`draft_pool.parquet` is the wrong frame to derive the split from** — it
carries the 2026-27 production board and silently shifts validation onto a test season. And
a strong reference entry has to be held **out** of the field it is measured against, because
a fixed pool means its winnings come out of everyone else's.

> Read `docs/simulations-plan.md` ("`src/sim/bracket.py`", "Tests") and
> `docs/dk_best_ball_rules.md`. Build `src/sim/bracket.py`: best-7-of-16 by slot per scoring
> period (2 G / 2 F / 1 C / 2 UTIL, dual eligibility placed to maximize the total rather than
> greedily), Round-1 to Round-4 advancement, wildcards, and the **cascading tie-break** — best
> single week, then second-best down through the round, then best individual player score,
> cascading the same way. Read every structural number (round count, pod size, advance count,
> cash table) from `dashboard.economics`, never hardcoded. The Round 2–4 field must be the
> **selected survivor population**, not a fresh ADP field; simulating the whole bracket gets
> this for free and scoring rounds independently would overstate continuation value. Tests:
> hand-built lineups with known scores, both tie-break cascades, and the advance chain
> reproducing `economics.advance_table` field sizes exactly for all five tournaments.

### 6. `make draft-sim` — the snake draft and the ADP field, and Gate B ✅ built 2026-08-09

**Gate B passes wide and the fit is the finding** — see "What was built, and what Gate B
found" above. Four things carry into items 7 and 8. The **mean-ADP target does not identify a
constant rank noise** (0.110–0.117 picks across the whole grid), which is why real pick logs
are now a named capture rather than a nicety. The **opponent model is a registry**, so a more
realistic drafter is a class with one or two methods and not a second draft loop. **DK's caps
bind autodraft and not a manual pick**, so our own seat runs uncapped. And the reactive
recompute measures roughly **0.76 s mean / 1.5 s max** at `n_sims = 500` over the full
remaining pool, so Gate E is not free.

Two things it deliberately did **not** do, both stated rather than quietly deferred:
`bracket.placeholder_field` is **not** repointed at this module — the swap is `draft_field`
into one call site, but it invalidates every figure in `bracket_*.csv` and wants a full
`make bracket` re-run, which belongs with item 8 where the field is rebuilt per strategy
anyway. And the in-draft objective is **marginal lineup value**, not the payout-weighted
bracket EV decision 5 names; `recommend`'s `value` argument is the seam item 7 substitutes it
through.

> Read `docs/simulations-plan.md` ("`src/sim/draft.py`") and `docs/adp-plan.md`. Build
> `src/sim/draft.py`: a 12-entry, 16-round snake draft over one engine with two modes —
> **reactive** (a pick function sees board state and returns a ranked recommendation; this is
> primary) and **ranking-submission** (static ranking + position limits + exclusion list,
> executed by DK's documented autodraft logic: queue → ranking → 8G/8F/3C caps). Opponents
> autodraft off the **DK-recalibrated** consensus from `adp_transfer.parquet` — never the raw
> consensus — with rank noise. **Gate B**: simulate many drafts, measure the resulting average
> draft position, and require it to reproduce the observed ADP curve within the 17.0-pick
> recalibration error. Fit `rank_noise_sd` to that target rather than choosing it. Let field
> composition vary by tournament tier even though nothing calibrates that yet.

### 7. `dashboard/draft_room.py` — the live recommender, and Gate E ✅ built 2026-08-09

**Gate E passes at 112 ms mean / 200 ms max against a 1,000 ms bar, so a live draft is
possible** — see "What was built, and what Gate E found" above. Four things carry into item 8.
Best-7-by-slot is **one matroid exchange** and it is exact, which is a tool the sweep can use
too. The in-draft objective now *is* payout-weighted bracket EV, so `recommend`'s `value` seam
is filled. **The EV level does not converge on `600k_shootaround`** — −17% against the
symmetric null at the shipped field size, and tripling the field makes the *ranking* less
stable rather than more — while P(top 2 of 12) is exact and reproduces across fields, which is
the same split `select-on-p-advance-report-roi` already predicted. And a pick can only be
priced inside a **completed** roster, which is a plug-in the sweep will have to make a
decision about rather than inherit.

One deviation from the conventions, taken deliberately and registered as
`draft-room-imports-src-sim`: the page imports `src.sim`, which `dashboard/README.md`
otherwise forbids. The rule exists so a view cannot refit; the alternative here was
reimplementing `bracket.best_lineup` and `draft.legal_mask`, which is the drift the rule
prevents arriving the other way round. The invariant is narrowed by a test rather than
waived — one named file, bounded at `src.sim`.

> Read `docs/simulations-plan.md` ("The live draft room"). Build a Streamlit page separate from
> the walkthrough app: load the precomputed sim tensor and draft pool, show the board, take
> **one click per pick** to mark a player gone, and return a ranked recommendation by marginal
> bracket EV. **Gate E is a hard latency bar: under 1.0 s per recompute on the full remaining
> pool.** Two things make it fit — drop to `n_sims = 500` in-draft (the decision is a ranking,
> not a level) and use a partial sort for best-7-by-slot rather than recomputing the whole
> lineup selection per candidate. Measure and report the actual latency. If the bar cannot be
> met, fall back to exporting a static ranking + exclusion list in DK's pre-draft-rankings CSV
> format, which is the ranking-submission mode item 6 already built.

### 8. `make strategy-sweep` — the sweep, error injection, and Gates C and D ✅ built 2026-08-09

**The measured outcome is under "What was built, and what Gates C and D found" above.** Five
things carry into items 9, 10 and 11. **Gate C's premise was half wrong** — the uninjected
world already reproduces the model's miss in magnitude and gets the *market's relative skill*
backwards instead, so the injection rotates rather than adds. **The board had to be restricted
to priceable players on both sides**, because the tensor's 153 unscored players cost the ADP
field 1.26 roster spots an entry and cost a model-ranked strategy nothing. **Nothing in the
sweep table resolves unpaired** and almost everything does once differenced inside the
simulated season. **Gate D fails**: one board serves both tiers. And **the shipped strategy is
`lineup_value_blend30`**, written to `strategy_shipped.csv` for item 10 to read rather than
re-decide.

> Read `docs/simulations-plan.md` ("`src/sim/strategy.py`", "The backtest"). Build
> `src/sim/strategy.py` + `make strategy-sweep`. A strategy is a config object (ranking source,
> `α` overall and per-round, position caps, exposure caps, stacking, objective, entry count), so
> the sweep is a table. **Gate C first, because it gates the sweep's validity**: perturb the
> simulated truth to reproduce the model's *measured* out-of-sample miss — availability CRPS
> 10.006 games, component R² 0.81–0.95 against the no-fit floors, season-total MAE 400.5 dk_pts
> — and verify it does. Without it the sweep drives `α → 0` for reasons that have nothing to do
> with the market. **Select on lift in P(top 2 of 12), report ROI** against the break-even
> hurdle with a bootstrap interval; ROI alone is dominated by 0.139%-probability deep runs and
> will not resolve. Sweep both tiers (10 entries at $20, 4 at $52). **Gate D**: confirm the two
> tiers select materially different rosters. Go through `selection_split` — the sweep must
> raise if it reaches the test seasons, pinned by a test the way `component_rates`' guard is.
> Write the shipped strategy to an artifact. Then replay against realized 2022-23 / 2023-24 and
> report it with honest, wide intervals as a readout, not a selector.

### 9. Production run for 2026-27 ⏳ schedule-gated

> Read `docs/simulations-plan.md` ("The deadline"). Poll
> `nba_api.stats.endpoints.ScheduleLeagueV2(season="2026-27")` — as of 2026-08-08 it returns 20
> rows (19 preseason plus the 12/11/2026 Cup final) against 1,400 for 2025-26, so the regular
> season is not published. Once it is: refresh `make fetch` for 2026-27 rosters and schedule,
> refit the heads on all 30 seasons, rebuild the posterior artifacts at the full window, build
> the 2026-27 draft pool and sim tensor, and produce the board. Flip
> `production-schedule-not-published` from `blocked` to resolved.

### 10. The test-split risk readout — once, before going live

> Read `docs/simulations-plan.md` ("The test split") and `src/final_evaluation.py`. Extend the
> final evaluation to backtest the **already-shipped** strategy on 2024-25 and 2025-26 inside
> `held_out.unlocked("pre-season risk readout")`. It must read which strategy shipped from the
> artifact rather than re-deciding, and emit only a risk report — ROI distribution, P(advance),
> P(cash), worst-case drawdown across the 10 + 4 entries. **It changes nothing**: not the
> strategy, not the stake, not the entry decision. Run it once.

### 11. Dashboard and docs refresh

> Read `docs/dashboard-plan.md`, `docs/provenance-plan.md` and `docs/docs-audit.md`. Repoint
> `dashboard/tabs/simulations.py::CHAIN` and `dashboard/tabs/drafting.py` at the real artifacts
> — both currently render "not built" warnings. Flip the 2026-08-08 registry entries from
> `settled` / `open` to `built` with their artifacts in `reproduce`. Update README §2's
> "Simulation — planned" and "Ranking, drafting and tournaments — planned" sections and §3
> Results with the measured figures. **Add `docs/simulations-plan.md` to `src/docs_audit.py`**
> now that it carries measured figures rather than specification. Run `make docs-audit` and
> `make dashboard-audit`.

## Doc and registry obligations

Six decisions from 2026-08-08 are registered in `dashboard/decisions.py` under topics
`simulations` and `drafting`. As pieces get built: flip each entry's `status` from `settled` to
`built` with its artifact in `reproduce`, update this doc in place, and repoint
`dashboard/tabs/simulations.py::CHAIN` and `dashboard/tabs/drafting.py` at the real artifacts —
both tabs currently render "not built" warnings that will otherwise go stale.
