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

## The second prerequisite: game length is a random variable forward, not a lookup

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

Once it ships, `stan_game_length` owns the game-length draw, `stan_composition` **imports** it
rather than defining its own — the same ordering constraint the composition already has on
`stan_minutes` — and `fit_ot_tail` / `sample_game_length` are withdrawn from `stan_composition`
with a registry entry recording the move.

---

## The third prerequisite: is the marginal minutes head still needed?

Raised 2026-08-08. `README.md` says the two minutes heads "compose rather than compete" —
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

### Claim 3 — the year effect. **Genuinely load-bearing, and the reason not to retire it yet.**

`season_terms` selected the `year` arm for `min`: val MAE **199.03** against the base arm's
199.72, `sigma_year` **0.0231**. It is the only head in the project that ships a season term,
and a year effect is worth **+10.4%** on a 15-man roster's season-total sd — which matters
here more than anywhere else, because roster-level spread is what decides a 2-of-12 knockout.
**The composition cannot carry one today**: `composition_glm.stan` has no `S` / `year_z` /
`sigma_year` block, unlike `betabinomial_glm.stan` where `S = 0` disables it exactly. Retiring
`stan_minutes` right now would drop the project's only era correction.

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

make stan-game-length src/models/stan_game_length.py  does a game go to OT, and how deep
                                                   -> outputs/predictions/stan_game_length_*.csv
                                                      + a posteriors/ entry. Replaces
                                                      stan_composition.fit_ot_tail

make scoring-periods ✅ src/features/scoring_periods.py NBA week grid -> DK round windows
                                                   -> data/features/scoring_periods.parquet

make draft-pool      src/features/draft_pool.py    the board: player x season, DK position
                                                   eligibility, team, ADP, prior-season row
                                                   -> data/features/draft_pool.parquet

make simulate-season src/sim/season.py             THE tensor
                                                   -> data/features/sim_tensor_<season>.npz

make draft-sim       src/sim/draft.py              snake draft vs an ADP field
make bracket         src/sim/bracket.py            4 rounds, advancement, ties, payouts
make strategy-sweep  src/sim/strategy.py           the sweep -> outputs/predictions/strategy_*.csv
make draft-room      dashboard/draft_room.py       the live recommender
```

`src/sim/` is a new package, parallel to `src/models/` and `src/eda/`, because these are
neither models nor analyses. Conventions carry over unchanged: `python -m src.sim.<module>`
entry points, the duplicated `yaml.safe_load` idiom in `__main__`, `mkdir(parents=True,
exist_ok=True)` before every write, `f"... {n:,} ... → {dest}"` progress lines.

### `src/sim/season.py` — the season simulator

Assembly, not invention. Every piece already exists and is measured:

| Step | Source | Number |
|---|---|---|
| **how long the game is** | **`stan_game_length`** | **5.94%** of games go to OT; see below |
| which games he plays | `stan_games_played.sequences` | entry × exit × within-tenure chain, all gates pass |
| minutes, team-constrained | `stan_composition.simulate_minutes` | CRPS 4.4945 vs 4.7842 independent |
| minutes, game-level noise | the composition's **own** role-graded rho | 0.1768 fringe → 0.0855 star. `stan_minutes_dispersion.csv`'s **4.65×** is a *diagnostic* to check draws against, **not** an input — see "The third prerequisite" |
| minutes, serial dependence | `serial_correlation.csv` | **2.43×** ten-game block inflation |
| the eleven component heads | `season_terms._draw_components` | already materializes `fga → fg3a\|fga → fg2a → makes` |
| cross-component dependence | `residual_correlation.csv` | Gaussian copula, mean +0.013, max 0.157 |
| the bonus | `targets.expected_bonus` / `compute_dk_pts` | overdispersion **0.025** at the player-game unit |

Five rules the assembly must not violate, all of them already argued elsewhere and repeated
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

Consumers default to the `train_val` fit window, matching the four numbers already calibrated
that way.

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

### `src/sim/bracket.py` — rounds, advancement, ties, payouts

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

**A cheap widening exists if it turns out to be needed.** ADP also covers 2014-15, 2017-18,
2018-19 and 2019-20. The fitted heads are in-sample on those, but the **no-fit carry-forward
floor is out-of-sample by construction** and scores only 0.001–0.03 R² below the fitted heads.
A floor-ranked backtest across six seasons is a legitimate robustness check on strategy
*shape*, even though it cannot price the fitted model's edge. Build it only if the two-season
result is ambiguous.

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

---

## Data the layer needs, and where it comes from

- **DK position eligibility** — `data/raw/team_rosters_*.csv` carries `POSITION` with DK-shaped
  dual eligibility (`G-F`, `F-C`, `C-F`, `F-G`) for all 30 seasons with **zero** nulls on the
  historical files. Without it no lineup can be filled at all. The two
  `data/raw/dk_draft_rankings/*.csv` boards carry DK's *own* positions for 698 and 942 players,
  which is what the NBA.com → DK mapping is validated against. 2026-27 rosters carry nulls for
  unsigned players; the DK board covers them.
- **ADP** — `data/features/adp_panel.parquet`, 15,012 rows. Coverage is 2014-15, 2017-18,
  2018-19, 2019-20, 2022-23, 2023-24, 2024-25, 2025-26 (both sources) and 2026-27 (DK only).
  Both validation seasons are covered, which is the coverage that matters.
- **Schedule** — realized game dates from `data/processed/game_logs.parquet` for backtests;
  `ScheduleLeagueV2` for the production season, **once it is published**.
- **Tournament structure** — the two `dk_best_ball_tournament_*.csv` files, through
  `dashboard.economics`. Re-verify against the live 2026-27 contests before any backtest
  number is treated as load-bearing.

---

## Gates

Every gate is judged on validation or on simulated truth. None reads the test split.

| Gate | Pass condition | Why this bar |
|---|---|---|
| **A** | the season simulator reproduces the **marginals it was built from**: season-total dk_pts distribution against `season_total_metrics.csv`, GP pmf against `stan_games_played_gp_pmf.csv`, and per-game bonus rate against `bonus_calibration.csv` | An assembly bug is silent. Every input head is already calibrated, so a simulator that misses a marginal it was handed has a wiring fault, not a modelling one |
| **B** | simulated drafts reproduce the **observed ADP curve** — mean absolute rank gap under the 17.0-pick recalibration error, so the field model is no worse than the market proxy it consumes | The field model's only real calibration target. Failing it means the opponent model is not a field |
| **C** | the **error-injected** simulated world reproduces the model's measured out-of-sample miss: availability CRPS ≈ 10.006 games, component R² in 0.81–0.95, season-total MAE ≈ 400.5 dk_pts | Without this the sweep cannot price ADP, exposure caps, or any other hedge against model error |
| **D** | the sweep selects **materially different** rosters for the two tiers | If the $20 and $52 strategies converge, either the objective is not doing its job or the tier difference is smaller than the economics imply. Either way it needs to be known before entering |
| **E** | in-draft recompute **under 1.0 s** at `n_sims = 500` on the full remaining pool | The 30-second clock. Failing it drops the draft room to ranking-submission mode |
| **F** | measured edge, expressed as **lift in P(top 2 of 12)** over an ADP-drafted entry, is reported with a bootstrap interval against the break-even hurdle (+17.60% / +12.32%) | Not a pass/fail on the edge itself — a requirement that the number is quoted in comparable units with its uncertainty, rather than as a point estimate |

---

## Config

```yaml
sim:
  posterior_draws: 1000          # thinned across the whole posterior, never sliced
  fit_window: train              # NOT train_val — see "One window per consumer" below
  n_sims: 2000                   # strategy sweep
  n_sims_draft: 500              # in-draft; a ranking, not a level
  scoring_periods: 20            # R1's 17 weeks + three double weeks
  pod_size: 12
  seed: 0

  tournaments:
    600k_shootaround: {entries: 10}
    20k_spin_move:    {entries: 4}

  field:
    adp_source: dk_recalibrated  # docs/adp-plan.md: never the raw consensus
    rank_noise_sd: null          # fitted by Gate B, not chosen
    position_caps: {G: 8, F: 8, C: 3}   # DK's own autodraft defaults

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
```

---

## Tests

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.

- **lineup selection** — best 7 by slot from a hand-built 16 with known scores, including the
  UTIL fallback and a dual-eligibility player who must be placed to maximize the total rather
  than greedily.
- **tie-breaks** — two entries with identical round totals and different weekly maxima, then
  identical weekly maxima and different best-player scores, cascading.
- **bracket arithmetic** — the advance chain must reproduce `economics.advance_table` field
  sizes exactly for all five tournaments.
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

---

## Open questions and risks

- **The 2026-27 schedule is not published.** Blocks the production run only. Poll `ScheduleLeagueV2`.
- **Field skill by tournament tier is unmeasured**, and will bias any opponent model that
  assumes one field composition across a $20 and a $52 contest.
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
- **Tournament structures may change** for the live 2026-27 contests. Re-verify the metadata
  and prize CSVs before treating any backtest result as load-bearing.
- **This doc is not yet in `make docs-audit`.** Add it once it carries measured figures rather
  than specification — see `docs/docs-audit.md`.

---

## The build order — one session per item, with its opening prompt

Ordered so the **live-draft path closes at item 7**. Items 2, 3 and 3b depend on nothing and can
run in any order alongside item 1. **Item 3c must follow 3b** — both edit `stan_composition`
— and both must land before item 4, which imports whatever they settle.

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

### 3. `make draft-pool` — the board, with DK position eligibility

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

### 3b. `make stan-game-length` — the overtime head ⛔ blocks 4

Added 2026-08-08, after the rest of the order was written — see "The second prerequisite"
above. Independent of items 1–3 and can run alongside them, but must land before item 4, and
must be added to `make posteriors`' head list once it exists.

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

### 3c. `make minutes-unification` — does the composition retire the marginal head? ⛔ blocks 4

Added 2026-08-08 — see "The third prerequisite" above. **Sequence it after 3b**, which is
already in flight and edits the same module (`stan_composition` gives up `fit_ot_tail` there
and may give up the comparator here). Must land before item 4, because it settles what the
simulator's minutes draw imports.

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

### 4. `make simulate-season` — the tensor, and Gate A

> Read `docs/simulations-plan.md` ("The output contract", "`src/sim/season.py`") and
> `docs/predictions-plan.md`. Create the `src/sim/` package and build `src/sim/season.py` +
> `make simulate-season`, writing `data/features/sim_tensor_<season>.npz`: a
> `player x scoring_period x sim` float32 tensor of dk_pts plus a `uint8` games-played twin.
> Assemble, do not invent — `stan_games_played.sequences` for which games, `stan_composition
> .simulate_minutes` plus the 4.65x game-level dispersion and 2.43x block inflation for minutes,
> `season_terms._draw_components` for the eleven heads in the order `fga → fg3a|fga → fg2a →
> makes`, the conditioned matrix from `residual_correlation.csv` as the copula, and
> `compute_dk_pts` for scoring. Four rules that must not be violated: draw never plug in; one
> shared `min` draw per player-game feeds all eleven heads; sequential structure goes on minutes
> only; **the posterior draw is the outer loop**, shared across all players, because that shared
> `β` is the cross-player correlation this layer exists for. **Gate A**: the simulator must
> reproduce the marginals it was handed — season-total dk_pts against `season_total_metrics.csv`,
> the GP pmf against `stan_games_played_gp_pmf.csv`, per-game bonus rate against
> `bonus_calibration.csv`. A miss here is a wiring fault, not a modelling one.

### 5. `make bracket` — lineups, ties, advancement, payouts

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

### 6. `make draft-sim` — the snake draft and the ADP field, and Gate B

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

### 7. `dashboard/draft_room.py` — the live recommender, and Gate E 🎯 live-draft ready

> Read `docs/simulations-plan.md` ("The live draft room"). Build a Streamlit page separate from
> the walkthrough app: load the precomputed sim tensor and draft pool, show the board, take
> **one click per pick** to mark a player gone, and return a ranked recommendation by marginal
> bracket EV. **Gate E is a hard latency bar: under 1.0 s per recompute on the full remaining
> pool.** Two things make it fit — drop to `n_sims = 500` in-draft (the decision is a ranking,
> not a level) and use a partial sort for best-7-by-slot rather than recomputing the whole
> lineup selection per candidate. Measure and report the actual latency. If the bar cannot be
> met, fall back to exporting a static ranking + exclusion list in DK's pre-draft-rankings CSV
> format, which is the ranking-submission mode item 6 already built.

### 8. `make strategy-sweep` — the sweep, error injection, and Gates C and D

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
