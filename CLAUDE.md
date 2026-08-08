# NBA Deep Learning

Predicts a player's DraftKings fantasy points (dk_pts) for each game of an upcoming season,
plus the running season total. **See `README.md` for the problem definition, the measured
variance budget, and the modeling design** — this file covers conventions and hard-won
facts you should not re-derive.

## Prediction-time constraint (drives everything)

Before the season starts we know: (1) the season schedule, (2) **season-start rosters** —
definitively which team each player is on, plus team identity — and (3) *previous-season*
stats for every team and player. We do **not** know within-season roster changes
(mid-season trades), current-season minutes, injuries, or form.

The information set is therefore a **cross-season join: current-season roster membership ×
prior-season statistics.**

Consequences:
- Team composition aggregates the **season-S roster** described by **season S-1 stats** —
  not the S-1 roster. Same for opponents.
- **Minutes weights must come from S-1**, since season-S minutes are unknown.
- Rookies/returnees are on the known roster but have no S-1 stats — impute
  (`bio_draft_number`) or exclude *and renormalize*; dropping them silently biases
  aggregates toward veterans.
- Team identity is known, so team embeddings/fixed effects are legitimate inputs.
- Every feature except schedule-derived ones is still **constant within a player-season**.
  The only per-game variation available is opponent, home/away, and rest.
- Season-start rosters are derivable from disk: a player's team in his **earliest game of
  season S**. Known before the season, so not leakage in a backtest.
- Only mid-season churn is irreducible — 78/572 players (13.6%) appeared for 2+ teams in
  2023-24. Out of scope for now.

## Project layout

```
src/data/      fetch, preprocess, dataset, injuries, injury_reports, boxscore_status,
               adp_draftkings, adp_fantasypros
src/features/  rolling stats, matchup context, team context, targets, availability,
               game_length, encoding, adp
src/eda/       season_matrix, pca, archetypes, context_value, persistence, aging,
               target, feature_diagnostics, availability, report_calibration, adp_profile,
               serial_correlation, variance_budget, residual_correlation
               (season-level analysis pipeline)
src/models/    lstm, transformer, multihead, xgboost baseline, availability,
               season_total, component_rates, games_played,
               stan_utils, stan_availability, stan_minutes, stan_components,
               stan_composition, stan_games_played, season_terms
src/stan/      betabinomial_glm.stan, negbinomial_glm.stan — TWO files for fourteen-plus
               heads, because availability / minutes / conversions / onset / entry /
               exit are the same likelihood with different data, and the counts are the
               other one. Both carry an OPTIONAL year random effect that `S = 0`
               disables EXACTLY — zero-length parameter vectors, identical posterior.
               composition_glm.stan is the third and stands apart: the team-game
               minutes allocation, a multinomial decomposed into binomial trials.
               betageometric_duration.stan is the fourth: absence-spell length, a
               geometric hazard with a Beta frailty integrated out — the same device
               one level down. `H_open = 0` disables its in-progress offset exactly,
               the same way `S = 0` disables the year effect
src/train.py   training loop
src/evaluate.py test-split metrics
src/predict.py  inference entry point
configs/       default.yaml — all hyperparams and paths
data/          raw → processed → features pipeline
outputs/       checkpoints, prediction CSVs, eda reports, stan/ (compiled binaries,
               gitignored — cmdstanpy builds them from a copy of src/stan/)
dashboard/     nine-tab project walkthrough over the precomputed artifacts.
               app/theme/charts/layout/artifacts + decisions (the registry),
               economics (tournament derivations), audit (make dashboard-audit),
               tabs/ (one render(ctx) each). decisions/economics/audit import no
               streamlit; nothing in the package imports src/ — see dashboard/README.md
docs/          eda-plan (season-level EDA spec), availability-plan (games played /
               minutes), games-played-plan (the spell process: entry x exit x a
               within-tenure chain, and why a plain full-window chain fails),
               adp-plan (market proxy, sourcing + where ADP belongs),
               predictions-plan, minutes-composition-plan, simulations-plan,
               shot-attempt-basis-plan (Gate 0 of the fga x fg3a|fga
               reparameterization — measured, adoption specified, NOT taken),
               dk_best_ball_rules,
               provenance-plan (every figure gets a make target — and the six
               corrections that fell out of building them), dashboard-plan
```

## Pipeline (run in order)

```bash
python -m src.data.fetch           # pull raw game logs from nba_api
python -m src.data.preprocess      # clean + add season column → data/processed/game_logs.parquet
python -m src.train                # train model → outputs/checkpoints/best_model.pt
                                   #   also saves scaler/feature_cols to data/features/
python -m src.evaluate             # test MAE/RMSE → outputs/predictions/test_metrics.csv
python -m src.predict              # inference → outputs/predictions/next_season_predictions.csv
```

`src/features/encode.py` (rolling aggregate features) is retained for experimentation but is no longer part of the primary training pipeline.

### Season-level EDA pipeline

```bash
make season-matrix    # → season_matrix_tier{A,B} + season_matrix_roster_tier{A,B} + coverage_report.csv
make pca              # → pca_{tier}_{mode}_{scores,loadings,variance} + pickled artifacts
make archetypes       # → archetypes_*, archetype_profiles_*, team_composition_*, kmeans_*, gmm_*
make team-context     # → team_context_tier{A,B}.parquet  (roster(S) × stats(S-1))
make context-value    # → outputs/eda/team_context_value_tier{A,B}.csv
make opponent         # → opponent_{pca,profile_cols,matchup,profile}_tier* + outputs/eda/opponent_matchup_*
make persistence      # → outputs/eda/persistence.csv
make aging            # → outputs/eda/aging_curves.csv
make target-profile   # → outputs/eda/target_{profile.csv,season_totals,trajectories}
make feature-diagnostics  # → outputs/eda/feature_diagnostics.csv + feature_correlation_tier*
make game-length      # → game_length.parquet (48 + 5k per game) + coverage csv
make serial-correlation   # → outputs/eda/serial_correlation.csv
                      #    lag-k autocorrelation + block variance inflation per component
make variance-budget  # → outputs/eda/variance_budget.csv
                      #    the five headline shares, both denominators, both nulls, the
                      #    residual sd — and the corrected own-minutes row
make residual-correlation # → outputs/eda/residual_correlation.csv
                      #    the 11x11 conditional matrix the simulator's copula takes as
                      #    an INPUT, in long form, plus the raw basis for contrast
make season-effects   # → outputs/eda/season_effects_{league_rates,summary,
                      #    carry_forward_bias}.csv — per quantity, is the league's era
                      #    movement an extrapolable TREND or an unforecastable SHOCK, and
                      #    what does ignoring it cost? No head carries a season term today.
make availability     # → availability_{panel,features}.parquet  (both roster windows,
                      #    plus the three-way status and the missed-reason split)
make availability-profile # → outputs/eda/availability_profile.csv
make report-calibration   # → report_transfer.parquet + outputs/eda/report_calibration.csv
                          #    P(play | injury-report designation, reason)
make eda              # all of the above, in dependency order
make dashboard        # nine-tab project walkthrough over the precomputed artifacts
make dashboard-audit  # registry drift report — a report, not a gate; exits 0 with findings
make docs-audit       # every quoted figure in the plan docs vs its artifact — a GATE
```

**`make docs-audit` is the guard against prose drifting away from its artifact**, which has
happened twice, both times silently and both times in a table that was *partially* refreshed:
the season-total R² column, and the report-calibration block. `src/docs_audit.py` holds one
`Claim` per quoted figure and runs three checks — the figure equals its artifact **to the
precision it is quoted at** (`22.7` to ±0.05, `0.0635` to ±0.00005); the quoted string still
appears in the doc, so a claim cannot rot into describing nothing; and coverage, so "how much
of this doc is audited" is a number. Unlike `dashboard-audit` this one **exits non-zero** —
a doc contradicting its artifact is a defect, not a preference — and it is also a `pytest`
test. Missing artifacts are *skipped*, so a fresh checkout without `make eda` is clean.
**Rebuilding an artifact will fail it until the docs are updated. That is the point.**
It guards the artifact→prose direction tightly and the prose→artifact direction loosely;
see the module docstring for exactly what it cannot catch.

**It covers eight docs — `README.md`, `availability-plan`, `minutes-composition-plan`,
`predictions-plan`, `games-played-plan`, `shot-attempt-basis-plan`, `adp-plan` and
`CLAUDE.md` — with 2,230 claims and one builder per doc.**
Coverage of measured figures: 80% (README), 79% (shot-attempt basis), 64% (composition),
63% (predictions), 62% (adp), 60% (CLAUDE.md), 55% (availability), 46% (games played) —
`make docs-audit` prints them live, so treat the
printout rather than this line as current. The uncovered remainder is prose-only figures
(`docs/provenance-plan.md` lists all fourteen), costing estimates, and counts of things
rather than measurements.
**583 of the claims are superseded values held for the record**, which is the number that
grows fastest as heads move off the test split: each conversion retires a measurement
without deleting it. The season-term conversion alone added **161** — the largest single
jump so far, because that ablation quotes four arms across thirteen heads and five
downstream tables, so a change of split retires figures by the table rather than by the
line.

**`README.md` was added last and is the doc the guard fits best**, which is why its coverage
is the highest: it holds no measurements of its own, only a selection of headlines copied
from `CLAUDE.md` and the plan docs, and it is the most-read and least-maintained file in the
repo — the exact conditions under which a figure goes stale unnoticed. Two things its builder
does that the others do not: it claims **roundings** (`58%` against 57.96%, `86%` against an
R² of 0.859), because an overview should round and `implied_tolerance` already handles that
correctly; and it claims **shipped constants against their fitted optima** (the bonus
overdispersions 0.10 and 0.025 against `bonus_calibration.csv`'s `analysis == "fitted"`
rows), so a re-calibration that moves an optimum away from the constant fails here rather
than passing silently. The tournament break-even hurdles have no `outputs/` artifact —
`dashboard/economics.py` derives them at render time — so they are claimed against the
checked-in raw boards, with `_break_even_hurdle` duplicating one line of `economics.py`
rather than importing it, since nothing in `src/` imports the dashboard package. A test pins
the two copies together.

**A block quoted in two docs is claimed from both against the one artifact**, because
"current in one doc and stale in the other" is the failure that has already happened twice
here (the season-total R² column, the report-calibration block). `_regime_claims` and
`_season_term_claims` / `_season_term_summary_claims` are shared builders for exactly that
reason — `CLAUDE.md` carries a summary of the season-term verdict and the plan doc carries
it in full, so the summary claims the subset it quotes rather than being forced to carry
every cell.

**Some quoted figures must NOT agree with the artifact, and `Claim(historical=True)` is how
they survive.** Two kinds: a superseded value preserved beside its correction ("corrected
2026-07-30 from 0.664 / 0.838 / 0.922"), and a scratch-session measurement kept beside the
promoted one — `docs/adp-plan.md` is built on the second, quoting ρ **0.8704** on 218 pairs in
its planning section and **0.8675** on 226 in its implementation section, *both correct*. A
historical claim is excluded from the value check and still presence-checked, so the failure
mode it guards is **deletion**, not drift. There are 56 of them. Without the flag the only
options are to "correct" a reversal out of existence or to leave it unprotected.

**Extending it to the three new docs found drift in all three**, which is the argument for
having built it: the serial-correlation table in `predictions-plan.md` (twelve rows, refreshed
in `CLAUDE.md` and not here), the roster-coverage and residual-correlation figures corrected in
two other files and missed here, the ADP position offset (−0.3 → **−2.0**, a real change on the
larger matched set), and — the second occurrence of the exact failure named above — the
**report-calibration block, stale in `CLAUDE.md` while current in `availability-plan.md`**. It
is now claimed from both docs against the one artifact so that cannot recur.

**The dashboard reads artifacts and nothing else — that is an invariant, not a
convention.** It never refits, and there is **no import from `src/`** anywhere in the
package; a test walks it with `ast` and fails if one appears. The nine tabs follow the
project end to end rather than mirroring `src/eda/`: problem · data collection · EDA ·
availability · minutes · components · simulations · drafting · decision log.
`.streamlit/config.toml` sets `headless = true`, without which Streamlit's first-run email
prompt makes `make dashboard` exit 255 instead of serving.

**Every figure on the dashboard is read from an artifact a `make` target produced.** Where
none exists, the panel renders `layout.pending_marker(...)` naming the target rather than a
typed number, and `make dashboard-audit` counts the markers. It is **0** today, because all
ten items in `docs/provenance-plan.md` landed first. The only exception to the rule is an
`incident` entry, which carries a date and a doc reference instead of a number.

`make dashboard-audit` runs four checks — every cited artifact exists (also a `pytest`
test), no source doc has a git commit newer than an entry's `reviewed` date, no artifact on
disk goes unreferenced by both tabs and registry, and no pending markers remain. A weekly
launchd job (`com.nba-deep-learning.dashboard-audit`, Mondays 09:00) appends it to
`outputs/dashboard_audit.log`. **The orphan check is the one that earns its keep**: nine
artifact families had accumulated unreachable from the dashboard purely because nothing was
looking, and removing a tab silently re-creates that — so bring it back to zero deliberately,
either by rendering the family or by naming it in a registry entry with its make target.

**Verify dashboard changes with Streamlit's `AppTest`, not with `curl`.** A request to port
8501 returns 200 from the HTML shell even when the script raises on every tab; `AppTest`
executes `app.py` for real, and because `st.tabs` renders all its children, an exception in
any tab surfaces.

### Availability data capture

```bash
make daily-capture     # injury-reports + injuries — MUST be on a cron; see the Makefile
make boxscore-status   # 2006-07 → 2025-26 inactive lists + DNP reasons, ~8-14 h, resumable
make availability-model  # baselines + CRPS/PIT → outputs/predictions/availability_*.csv
make season-total      # composes gp × rate → outputs/predictions/season_total_*.csv
                       #   what the availability head is worth on the actual deliverable
make component-rates   # → component_rate_metrics.csv
                       #   the 11 component heads vs the mandatory no-fit floor
```

### Stan heads

```bash
make stan-availability # port of the point-MLE beta-binomial + the posterior it buys
make stan-minutes      # min | available, trials = real game length (NEVER 48)
make stan-components   # 8 NB count heads + 3 beta-binomial conversion heads
make stan-composition  # the team-game minutes COMPOSITION — the per-game allocation
                       #   (zero-sum + the cap); see docs/minutes-composition-plan.md
make stan              # all four, in chain order — composition AFTER stan-minutes,
                       #   which it imports from and measures itself against
make games-played      # the games-played spell process, numpy only — the collapse, the
                       #   spell classes, the closed-form beta-geometric fits and Gate 0's
                       #   empirical-hazard Monte Carlo. NO Stan, so a process class can
                       #   be rejected in seconds rather than in NUTS hours.
make stan-games-played # the fitted arms. Imports stan_availability's head as its
                       #   permanent floor, so that target runs first. Held OUT of the
                       #   `stan` aggregate until Gate D passes.
make stan-substitution # Gate 0 of docs/shot-attempt-basis-plan.md — the `fga` count x
                       #   `fg3a | fga` share against the two independent attempt
                       #   counts, both arms un-handicapped. 16 fits and its OWN
                       #   artifact, because `substitution_arm` lives inside
                       #   `stan-components` and cannot be refreshed without its 209
                       #   minutes. An ablation, so not in `stan`.
make season-terms      # does any head need a season term, and which kind? A trend
                       #   covariate and a year random effect per head, plus the
                       #   season × role arm the availability era effect calls for.
                       #   An ABLATION over the shipped heads, so also not in `stan`;
                       #   it reads their selected specs from their artifacts.
```

Fitted **separately**, one model per head, because the chain
`availability → min | available → counts | min → makes | attempts` factorizes the joint
posterior exactly when the parameter blocks are distinct. Sources live in `src/stan/`;
cmdstanpy compiles a **copy** into `outputs/stan/` so no binary and no generated `.hpp`
enters the repo. Needs a CmdStan toolchain, which pip does not manage:

```bash
.venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

**`daily-capture` is the only thing in this repo with a deadline.** Both its sources are
current-status feeds that cannot be backfilled: the NBA injury-report PDFs age out of the
CDN after ~7 months (a 403 thereafter, forever) and the ESPN feed has no history at all.
A day the cron does not run is a day permanently lost. `make injury-reports` is idempotent
and re-parsing is offline (`--reparse`), because the archived PDFs — not the parsed CSVs —
are the artifact worth keeping.

### ADP capture

```bash
make adp-draftkings    # ingest manually-downloaded DK boards + the DK ID → player_id map
make adp-fantasypros   # live consensus capture (--backfill for Wayback, --reparse offline)
make adp-panel         # → adp_panel.parquet, point-in-time safe
make adp-profile       # → adp_transfer.parquet + outputs/eda/adp_profile.csv
make adp               # all four, in order
make adp-status        # coverage for both sources, no requests
```

> ⏰ **TODO — 2026-08-01: run `make adp-fantasypros -- --backfill`** (i.e.
> `.venv/bin/python -m src.data.adp_fantasypros --backfill`). Only **23 of 259** archived
> snapshots are on disk. The sweep was blocked on 2026-07-28 by Wayback throttling
> (**HTTP 498** — its rate-limit code, returned after the planning-session sweep), which
> clears on its own. The run is resumable and skips anything already archived, so
> re-running costs nothing and a partial run is safe to repeat. **Delete this note once it
> has completed** and update the season-coverage table in `docs/adp-plan.md`.

**DraftKings ADP is the second thing in this repo with a deadline, and it is a manual
one.** The board is login-gated, has **zero** Wayback snapshots, exposes no API, and is
live only while contests are open (~Oct). It cannot be scraped or backfilled: a board not
downloaded from the draft lobby while it is open is gone permanently. Two are captured
(2025-10-17, 2026-07-28); the load-bearing one still to get is an **early-to-mid October
2026** board, timing-matched to the 2025 anchor. **See `docs/adp-plan.md`.**

**Tier A** = 30 seasons, box-score families, 10,900 player-seasons × 150 features.
**Tier B** = 13 seasons (2013-14+), adds tracking/hustle, 5,077 × 287, strict column
superset of Tier A. Artifacts that later modeling consumes go to `data/features/`;
human-readable analysis reports go to `outputs/eda/`.

## Python environment

Always use the project virtual environment (`.venv`) for running Python or installing packages — never the system Python.

```bash
# Create (first time only)
python -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

All `python`, `pip`, and `pytest` commands must be prefixed with the venv activation or run via the venv's binaries directly (`.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/pytest`).

## Tests

```bash
.venv/bin/pytest tests/
```

## Key config knobs (configs/default.yaml)

- `model.type`: `lstm` or `transformer`
- `features.sequence_length`: how many prior-season games to use as input (default 10)
- `features.target_stats`: `[dk_pts]`
- `data.seasons`: which NBA seasons to pull (need ≥3 for a train/val/test split)

## Train / val / test split

Temporal walk-forward by season. With seasons [S1, S2, S3, S4]:
- Train:  predict S3 games using S2 stats (and any earlier pairs)
- Val:    predict S3 games using S2 stats (second-to-last pair)
- Test:   predict S4 games using S3 stats (last pair)

The scaler is fit only on the prior-season games that feed into training, preventing leakage.

**Do not use the test data at any point during data preparation, evaluating 
modeling decisions/alternatives, model fitting, or evaluating simulated draft 
strategies on backtested seasons. Only use the validation data for these purposes. 
The test data should only be used for a final end-of-project step testing the 
error rate of the entire workflow.** (Note: This constraint was less rigorously 
imposed earlier in the project, and several existing findings relate to the 
test data, though with likely-minimal leakage impacts. This constraint is 
intended to avoid further leakage as the project develops.)

### ✅ The rule is now a capability, not a convention — `src/models/held_out.py`

**A rule that lives only in prose gets followed until it is inconvenient, and this one
already failed.** The games-played head's Gate D was *specified* with the incumbent's test
figures as its bars, run on the test split, and settled which model ships — on a 0.0013
CRPS margin a paired bootstrap could not distinguish from zero, which **reversed** when
re-decided on validation. Nothing in the code objected, because nothing in the code knew.

So reaching the test split now **raises** unless something has explicitly unlocked it:

- **`availability.split_seasons` is the choke point** every head goes through, and the guard
  lives *there* rather than at each call site — so a new head cannot forget to add a guard it
  never had to add. It returns `(kept, _HeldOut(held))`, and `_HeldOut` raises on
  `__getitem__` / `to_numpy` / `merge` but not on `len()` or `.columns`. The guard is on
  **use**, not on carving: `train, _ = split_seasons(...)` and `len(test)` stay legal because
  a module has to be able to separate the rows to know what to exclude.
- **`held_out.selection_split(design)` → `(train, validation)`** is what a sweep calls. The
  test seasons are never materialized, so there is nothing to read by accident.
- **`held_out.final_split(design)` → `(full_train, test)`** is guarded, and
  `src/final_evaluation.py` (`make final-evaluation`) is the only thing that unlocks it.
  It reads which arm shipped **from the artifact** rather than re-deciding, refits on train
  **plus** validation, and scores test once. Registered heads: `availability`,
  `games_played`, `season_total`.
- `tests/conftest.py` unlocks for the suite (unit tests exercise the split machinery on
  synthetic four-season frames); `tests/test_held_out.py` re-locks and tests the lock, and
  **AST-walks every converted module** to assert it names `selection_split` and never
  `split_seasons`.

**Why the final evaluation refits rather than predicting test from the training fit.** A test
figure should describe the model that would actually deploy, which has seen train *and*
validation. Predicting test from a train-only fit is a *lower bound* on deployed performance,
not an estimate of it. The cost is that a val-versus-test comparison is confounded — different
rows, different training data — which is exactly why the sweeps no longer emit a test column
at all. **A val/test gap was never a replication check here**, and the project stopped
pretending it was on 2026-08-05.

**Converted heads, and what it cost.** `stan_availability`, `stan_minutes`,
`stan_components`, `stan_composition`, `stan_games_played`, `season_terms`, `season_total`,
`component_rates`, and `eda.season_effects`'s `carry_forward_bias`. Roughly **half the
sampler time** across the Stan heads, since the test side was the majority of the fits at
double the iterations — and selection was raised to full-length chains everywhere, because
the short/long split was a second confound sitting inside a comparison being read as a
replication.

- **`component_rates` was the one head with no guard at all**, because it defined its own
  `split_seasons` rather than importing the shared one. Six identical lines routing around
  the lock. A test now pins that it cannot come back.
- **Four findings reversed on the move, and all four had passed on test.** Gate E of the
  games-played plan (✅ by 0.03 dk_pts on test → ❌ by 6.34 on validation); `fg3a|fga` under
  sklearn (clears its floor on test, fails on validation); the games-played Gate D that
  started this; and — added 2026-08-06 when `stan_components` was re-run — **`fta`, which
  failed its no-fit floor by 0.0024 on test and clears it by 0.0144 on validation**,
  retiring "the whole free-throw family fails". **Three survived unchanged**, which is the
  useful contrast: the availability
  Stan port still reproduces the MLE on 21/21 coefficients, the `reb` alpha-trap pair
  reproduces 0.928 / 0.662 to a thousandth across the split change, and the substitution
  arm's validation margin reproduces at −0.771 across a half-length-to-full-length change
  in every one of its fits.
  - **Every one of the four reversals was a sub-1% margin on test that a paired bootstrap
    could not have called.** That is the pattern worth carrying into the remaining
    conversions: the split move does not overturn findings with real margins, it overturns
    the ones that were never distinguishable from zero and were being reported as verdicts.
- **✅ `stan_composition` is converted AND its artifact regenerated (2026-08-08)** — the
  last code/artifact disagreement in the repo is closed, and **nothing reversed**: same
  selected arm, same ordering, same gate outcomes, with no arm moving more than 0.0019 CRPS
  across a doubling of chain length.
  - **The deferral argument was wrong, and the way it was wrong is the lesson.** It rested
    on "the run buys no decision" — true, and beside the point. Held-out figures sitting in
    an artifact are quoted from the docs and rendered on the dashboard as the head's
    performance, and they become the bar a successor arm is measured against. That is
    precisely how the games-played Gate D acquired test-set bars. **A stale artifact is not
    inert.**
  - **⚠️ It also rested on a cost estimate that was itself a `CLAUDE.md`-style false
    correction.** On 2026-08-06 the module's "the conversion roughly halves it" was
    "corrected" to "saves ~40%, 12.6–14.7 h", by assuming the validation fits would take 2×
    as long once selection moved to full-length chains. Measured: they took **1.33×**
    (7.35 h → 9.78 h), and `betabinom/val` ran *faster* at double the iterations (9,358 s
    against 9,615 s), because longer warmup adapts a better step size and buys fewer
    leapfrog steps. Total **9.92 h** against 21.13 h — a **53%** saving, so the original
    "halves it" was right and the correction was the error. **Sampler cost is not linear in
    the knob you are turning**, and an estimate from an untested proportionality is worth
    less than the run it replaces.
- **`make availability-model` has NOT been converted** and still scores the held-out split,
  so the head-vs-GBM-vs-ridge CRPS ladder (10.795 / 10.888 / 10.896 / 13.614) and the
  playoff-workload ablation remain held-out measurements. That is the one outstanding piece.

---

## Scope: regular season only — playoffs are features, never targets

**Every fitting frame is regular season only.** `preprocess.load_raw` defaults to
`season_type="regular"`; asking for playoff rows is explicit and deliberate. Three
independent reasons, in descending order of how decisive they are:

- **The contest is over before the playoffs begin.** `docs/dk_best_ball_rules.md`: Round 1
  runs 10/20–2/14 and Round 4 ends **4/4**, ahead of a mid-April playoff start. Predicting
  playoff games is out of scope by the rules of the product, not by modelling convenience.
- **Playoff minutes are a role interaction with a *sign change*, not a level shift.** ✅
  `make availability-profile` (`measurement == "playoff_scope"`, unweighted per the recorded
  exception). Measured on 5,762 player-seasons with ≥20 regular-season games and a playoff
  appearance, median playoff-to-regular MPG ratio by **that season's** role: **bench (<12 mpg)
  0.505**, rotation (12–24) 0.761, **starter (24+) 1.054** — 65.8% of starters play *more*,
  and 0 rows fall outside the three buckets. A pooled playoff indicator would fit one
  coefficient to a −50% effect and a +5% effect at once, and bench players are numerous enough
  to drag it toward compression — distorting exactly the star minutes the model most needs
  right.
- **Availability shifts too**: the share of a playoff team's ≥20-game regular-season
  players who appear at all runs **0.711 / 0.905 / 0.958** across those same buckets.
  Rotations shorten, which is a different availability process from the one
  `docs/availability-plan.md` models.
  - **⚠️ This supersedes the recorded 0.664 / 0.838 / 0.922**, which used a per-(player, team)
    denominator — one row for every team a player appeared for. That records a player traded
    away from a playoff team in February as having "not appeared" for it, which is roster
    churn misread as unavailability, and it double-counts him. Keying on his **last** team is
    the construction this repo uses everywhere else. The recorded figures overstated how much
    rotations shorten; the ordering, which is the finding, is unchanged and slightly sharper.
  - The denominator is scoped to players on a **playoff team**, so "his team missed the
    playoffs" cannot be read as "he did not dress".

**The playoff logs are still worth their fetch, as prior-season workload** — ✅ **built.**
`features.availability.playoff_workload` / `attach_workload` emit `playoff_games`,
`playoff_minutes`, `playoff_mpg`, `made_playoffs`, `playoff_minutes_share`,
`total_minutes_incl_playoffs`, `career_minutes` and `career_seasons` on
`availability_features.parquet`. 41.8% of player-seasons made the playoffs, mean 8.3 games
and 194 minutes (max 26 games), and `total_minutes` undercounts those players' mileage by
**10.8%**. Timing needs no special handling: season S-1's playoffs end in June and season
S opens in October, so the lag-1 values pass `assert_point_in_time` like everything else.
`game_length.py` likewise covers both season types, since length is a property of a game
rather than a target.

**Missing playoff logs raise rather than zero-fill** — "nobody made the playoffs in 30
seasons" and "the logs were never fetched" would otherwise be the same frame, which is the
`not_rostered`-vs-`unknown` collapse in a new costume.

## The component targets — the model's output contract

**`dk_pts` is deterministic given the components, and is never predicted directly.** Twelve
quantities are modeled per player-game, each with its own likelihood. Attempts and minutes
enter only as **exposure** and **trials** — they contribute nothing to DK scoring themselves.

| Component | Distribution | Exposure / trials |
|---|---|---|
| `min` (given availability) | successes / trials | trials = **game length**: 48, or 53/58/… in OT — *not* a count |
| `fga` | count — negative binomial | `min` |
| `fg3a` \| `fga` | successes / trials — the three-point **share of attempts** | `fga` |
| *`fg2a`* | **derived**, `fga − fg3a` — not a head | — |
| `fta` | count, but **arrives in pairs** — model *trips* and double | `min` |
| `fg2m` | successes / trials | `fg2a` |
| `fg3m` | successes / trials | `fg3a` |
| `ftm` | successes / trials | `fta` |
| `reb` | count | `min` |
| `ast` | count | `min` |
| `stl` | count | `min` |
| `blk` | count | `min` |
| `tov` | count | `min` |

**The shot-attempt basis is `fga` × `fg3a | fga`, adopted 2026-08-03 — seven counts and
four conversions, still eleven heads.** A three-point attempt *substitutes* for a two, so
`fg2a` and `fg3a` are not independent counts; modelling total attempts and the three-point
*mix* enforces the substitution by construction and keeps the posterior factorization
exact. `fg2a` becomes derived, exactly as `pts` already is, and is still the trials for
`fg2m | fg2a`. See `docs/shot-attempt-basis-plan.md`.
**The draw order is therefore `fga → fg3a | fga → fg2a = fga − fg3a → makes`** — a
conversion head's own draw becomes a later head's trials, which is an edge the two-count
basis never had. `season_terms._draw_components` materializes it; nothing else may
reorder it.

Availability sits **upstream** of all twelve: it gates whether the player-game exists at all
(`docs/availability-plan.md`), and `min` is drawn conditional on availability.

**The trials denominator for `min` is derivable exactly — do not truncate at 48.**
`make game-length` (`src/features/game_length.py` → `data/features/game_length.parquet`,
one row per game with `game_length`, `n_overtimes`, `n_periods`, `reliable`). Five players
are on the court at every moment, so a team's summed minutes are exactly `5 × game length`,
and the logs carry minutes to the second (98.4% of rows non-integer). **The two teams are
two independent estimates of the same quantity, and that is the validation** — measured
over all 37,986 games (regular + playoffs, 1996-97 → 2025-26): **0 disagreements**, 0
unreliable rows, worst rounding residual **0.617 min** against a 2.5 min decision boundary.
5.93% of games go to overtime (1,942 single, 264 double, 41 triple, 6 quadruple), so
truncating at 48 discards ~6% of games and censors the top of the minutes distribution
in exactly the games where stars play most — 1,650 player-games exceed 48 minutes, and
the observed maximum is 63.0.

**The specification is feasible on every row, and it is asserted rather than reported** — ✅
`make game-length` (`minutes_feasibility` / `assert_feasible`, `analysis == "feasibility"` in
`game_length_coverage.csv`). Joining `game_length` to `component_targets.parquet` covers
**100.0%** of 731,906 player-games and yields **zero** rows with `min > game_length`; the
maximum `min / game_length` ratio is exactly **1.0000** (a player who played every minute), so
the bound is tight rather than merely satisfied. `min ~ Binomial(game_length, ·)` is well posed
everywhere, with no clipping and no boundary hack.

**A nonzero violation count is a build failure, not a metric** — reporting it as a percentage
would be the same mistake as reporting an unmatched rate. `assert_feasible` has two arms because
they fail differently and both are otherwise silent: a violation means the binomial support is
wrong, and an *unmatched* player-game means the join is broken and the fitting frame quietly
shrank. Game ids go through `boxscore_status.pad_game_id` on both sides, since the logs carry
`int64` and the box-score files zero-pad — a test feeds it an int on one side and a padded
string on the other.

**Build it from the raw logs, never from `component_targets.parquet`.** `preprocess.clean`
drops players below `data.min_games`, and a team sum over a filtered frame is missing whole
players' minutes — it understates the length silently while still snapping to a plausible
grid point. `derive_game_length` flags any game whose residual exceeds `SNAP_TOLERANCE`
rather than snapping it anyway; `tests/test_game_length.py` pins that case.

Given a joint draw, reassemble `pts = 2·fg2m + 3·fg3m + ftm` and pass it to
`preprocess.compute_dk_pts` with `fg3m`, `reb`, `ast`, `stl`, `blk`, `tov`. Only **eight**
components reach the scoring function — `fg2m`, `fg3m`, `ftm`, `reb`, `ast`, `stl`, `blk`,
`tov`; `min` and the three attempt counts matter solely through the exposure and trials they
supply to those eight. The double-double / triple-double bonus is a threshold on
`pts`/`reb`/`ast`/`stl`/`blk` simultaneously, which is why **the deliverable is a joint draw,
not twelve marginals**. Full rationale and the fitting strategy: `docs/predictions-plan.md`.

---

## Established facts — do not re-derive

✅ **`make variance-budget`** (`src/eda/variance_budget.py` → `outputs/eda/variance_budget.csv`,
25 rows). Measured on 254,167 player-games (2014-15 → 2023-24; the recorded 254,187 predates a
data refresh). Within-player-season residual sd is **9.445 dk_pts**, total sd 14.566.

| Source | Share of variance | basis |
|---|---|---|
| Player-season identity | 57.96% | of **total** per-game variance |
| **Own minutes played** | **46.40%** — *unknowable in advance* | of within-player residual |
| Opponent × season | 0.691% | of within-player residual |
| Opponent × archetype × season, above a shuffled null | +0.308% to +0.962% | of within-player residual |
| Home / away | 0.034% | of within-player residual |

**Never quote a row without its basis.** Identity is 58% *of the total*; everything else is a
share of the remaining 42%. The artifact carries `basis` per row so the mix-up is impossible.

- **⚠️ Own minutes is 46.4%, not the 18.6% recorded until 2026-07-29.** The old figure
  conditioned the residual on the **raw minutes level** pooled across players, which is
  attenuated by construction: a 30-minute game is *below* average for a 34-mpg starter and far
  *above* it for an 18-mpg reserve, so their residuals cancel inside the cell. Conditioning on
  the minutes **deviation** — the contrast the residual is defined by — gives 46.40%, and 46.19%
  under a nonparametric fit, against a saturated per-player-season bound of 59.40%. The
  mechanism is on disk too: dk_pts per extra minute runs **0.852** at the bottom mpg tier to
  **1.230** at the top, so no single function of the level can represent it.
  `own_minutes_raw_level` still ships and reproduces 18.650%, so the superseded construction
  stays legible rather than being overwritten. **This strengthens the shared-`min` draw** the
  simulator is built on — minutes are a much larger common factor than the project believed.
- The opponent rows are **in-sample ANOVAs on contemporaneous opponent identity** — ceilings,
  not achievable gains. Both `opponent.variance_ceiling` and
  `feature_diagnostics.cell_importance` are run on identical rows and the gap ships as its own
  row (0.0050 pp / 0.0024 pp), so the one-off and its generalization cannot drift.
- The main effects are on the full window (254,167); the interaction needs a prior-season
  archetype and is on 198,509, where opponent × season reads 0.793% instead of 0.691%. Both
  frames are emitted, each with its own `n_games`.
- **The opponent effect in dk_pts units**: season-averaged sd of the opponent cell means is
  **0.867** against a residual sd of 9.445 — the same fact as "0.69% of variance", in units a
  reader can size.

**~90% of attainable skill is the season-level rate; ~1–2% is per-game modulation.** Budget
effort accordingly, and do not expect team composition to carry the model.

- **Predict components, not dk_pts directly.** Effects cancel *across components* in the DK
  sum. ✅ Both halves are now targets: `make context-value` writes `gross_dk_movement` /
  `net_dk_movement` / `cancellation_ratio` per own-team feature, and `make opponent` writes
  `opponent_sd` / `opponent_sd_corrected` / `dk_weighted_opponent_sd` per outcome with the
  triple on the `dk_pts` row.
  - Team context: `teammate_assist_supply` moves ast/36 by **−0.681**, reb/36 by **+0.503**
    and blk/36 by **+0.115** per sd *in component units*, which DK-weights to **2.106 gross
    against −0.254 net — an 8.30× cancellation** (`role_crowding` 8.22×,
    `teammate_spacing` 5.01×, `team_pace` 4.74×).
    **⚠️ The per-component numbers recorded until 2026-07-29 (−0.361 / +0.218 / +0.191) are
    the partial *correlations*, not per-sd movements.** A correlation is dimensionless and
    cannot be DK-weighted, which is why the gross/net pair could not be rebuilt from them.
    The ratios were right; the sentence describing their construction was not. The artifact
    now carries both, as `r_<outcome>` and `effect_<outcome>`.
  - Opponent: **1.803 gross against 0.911 net, a 1.98× cancellation.**
    **⚠️ This supersedes the recorded 1.105-against-0.785 (1.41×), which no single
    construction reproduces** — four internally consistent ones give 2.01× (raw per-season
    sd), 1.98× (with the cell-mean sampling variance removed, the shipped figure), 2.11×
    (pooled over seasons) and 1.84× (a ridge fit on the opponent's prior profile). The ratio
    is robust *because* each is internally consistent, so 1.41× means gross and net were
    taken from different bases. `cross_component_cancellation` now takes both from one
    `sd_col` and a test pins the invariance. The correction makes the argument **stronger**.
  - `net_dk_movement` must agree in sign with `r_dk_pts_per_game` — but only for a feature
    that *has* a sign. `role_crowding` (r = −0.005) and `team_pace` (+0.008) are recorded
    nulls where both signs are noise, so features below `SIGN_CHECK_MIN_R = 0.02` are
    reported as unchecked rather than failing the guard.
  - The double-double bonus is a threshold on five components, so `E[bonus] ≠ bonus(E[x])` —
    a single dk_pts head cannot represent it. This cross-component argument is what carries
    the decomposition; the old rate-vs-minutes cancellation argument was an era artifact.
- **Decompose `pts` further, into shot classes — FT / 2PT / 3PT, attempts and makes.**
  `pts` is not a primitive count, it is a *weighted sum* of three of them. Write the
  identity in whichever basis you are actually holding, and never mix them:

  ```
  pts = 2·fgm  + 1·fg3m + ftm      # stored columns — fgm ALREADY INCLUDES threes,
                                   #   so a made three needs only 1 more point
      = 2·fg2m + 3·fg3m + ftm      # derived fg2m = fgm - fg3m — the form the heads
                                   #   reassemble with, and the intuitive one
  ```

  Both are exact on all 731,906 player-games and are the same expression collected
  differently. `2·fgm + 3·fg3m + ftm` is the tempting error — it pays a three 2 + 3 = 5
  and matches `pts` on only 59.97% of games, exactly those with `fg3m = 0`. Two measured
  reasons to model the parts, both strongest for regression/GLM:
  - **The weighting is the entire source of `pts`'s overdispersion.** Within-player
    var/mean is 2.16–2.31 for `pts` in *every* minutes bucket, but 0.86–1.08 for `fgm`,
    `fg2m`, `fg3m`, `fga`, `fg2a` and `fg3a`. A 2× coefficient squares into the variance
    and doubles var/mean while leaving the mean alone. So a Poisson head is badly
    misspecified on `pts` and correctly specified on the shot classes — decomposing does
    not merely help, it removes the misspecification rather than patching it with a
    negative binomial.
  - **Attempts persist; percentages barely do.** Year over year with season absorbed:
    `fg3a` 0.908, `fga` 0.862, `fta` 0.854 against `fg3_pct` 0.500, `fg_pct` 0.435,
    `ft_pct` 0.365, `ts_pct` 0.302, `efg_pct` 0.281. Split each class into
    **attempts × efficiency**, spend the model's capacity on attempts, and shrink the
    percentages hard toward player/league means — a ~2× persistence gap says which side
    carries prior-season signal. This applies to *conversion* percentages only; **share**
    percentages persist like counts — see the persistence bullets below.

  Free throws are the exception that stays overdispersed (`ftm` 1.70–1.91, `fta`
  1.95–2.08) because they **arrive in pairs**: 64.6% of nonzero `fta` in 24+ minute games
  is even, and `X = 2·Poisson` has var/mean exactly 2. Model *trips to the line* and
  double them if a Poisson FT head is wanted.

  Nothing downstream breaks: reassembly is linear and exact, so the bonus machinery is
  unaffected — rebuild `pts` from the classes before applying `expected_bonus`, whose
  thresholds are on `pts`, not on shot classes. Under DK weights the scoring contribution
  is `ftm + 2·fg2m + 3.5·fg3m` (3.5, because DK pays 0.5 per made three on top of the 3
  points — this is the `fg2m` basis, so the coefficient on threes is 3, not 1). The
  classes are not independent — 3PA substitutes for 2PA — so they need the same joint
  treatment the bonus already requires, not seven marginal fits.

  There is ample data: `fga/fgm/fg3a/fg3m/fta/ftm` are in every game log and season
  matrix, `player_shot_locations` adds zone-level attempts, and `sco_pct_*` adds shot mix.
  `fg2m`/`fg2a` are not stored and must be derived (`fgm - fg3m`, `fga - fg3a`), since
  `FGM`/`FGA` **include** threes — the same fact that makes the two identities above look
  inconsistent when they are not. Reproduce with `make target-profile`
  (`src/eda/target.py::SCORING_PARTS`) and `outputs/eda/persistence.csv`.
- **The opponent main effect beats the interaction ~17× on dk_pts** out of sample (0.368%
  vs +0.022%, held out on 2024-25/25-26 with prior-season-only inputs). The old guidance
  ("encode opponent as an interaction, not a team-quality term") compared an in-sample
  ceiling with an achievable gain and is **withdrawn**. Build the main effect properly.
- **Keep the interaction for the components, not the aggregate.** +0.203% on `blk_per36`
  (nearly doubling its main effect), +0.05–0.07% on pts/reb/ast, −0.016% on tov, and only
  +0.022% surviving into dk_pts. The fitted style axis for blocks is the big/guard
  dimension (+0.74 height, +0.71 blk/36, −0.55 3PA/36) — the predicted rim-protection
  mechanism. Rank barely matters (+0.018/+0.022/+0.025% at rank 1/2/3); rank 2 is already
  effectively rank 1.
- **"Above a shuffled null" is null-dependent — always state which marginal you permute.**
  Same data, same cells: +0.969% permuting opponent within season, +0.311% permuting
  archetype. With ~2,700 cells over 198k rows, expected chance R² is ~1.4% against a raw
  statistic of 2.31%, so the headline was largely cell count.
- **Minutes-weight any regression on per-36 rates.** `sd(pts_per36)` is 42.4 in sub-5-minute
  games against 6.6 above 24 minutes; unweighted, garbage time dominates and the opponent
  signal vanishes under it (`reb_per36` main effect reads 0.014% unweighted, 0.168%
  weighted). `src/features/opponent.py::minutes_weights`.
- **Linear DR is degenerate for team aggregation.** Minutes-weighted mean of PC scores is
  *exactly* the PCA projection of the mean stat line (verified to 3.6e-15) — averaging and
  projection commute. Value lives only in the nonlinear step (clustering / similarity).
- **The mean centroid collapses.** 336 team-season pairs sit in the closest 1% of mean-PC
  distance while allocating >55% of minutes to different archetypes.
- **kNN over player-seasons leaks identity.** 18.8% of player-seasons have another season
  of the *same player* as nearest neighbour; 37.7% within the 5 nearest. Exclude
  same-`player_id` neighbours if using kNN at all.
- **Two DR budgets.** Player's own stats: n=10,900, p=150 — barely reduce. Team
  composition: n=**892 team-seasons** — reduce hard, 2–4 dims per side.
- **Own-team features must be leave-one-out**, else they encode P's own style.
- **Never use same-game teammate stats** — contemporaneous with the outcome.
- **NO head carries a season term, the league moves, and after the ablation that is the
  RIGHT answer — settled 2026-07-31.** `make season-effects` (`src/eda/season_effects.py`)
  measures the league; `make season-terms` (`src/models/season_terms.py`) tests whether any
  head wants a term for it. A season *fixed* effect is unusable at prediction time —
  there is no dummy for a season that has not happened — so the two candidates are a
  **year-on-year trend** and a **year-level random effect**, and they are complementary
  rather than alternatives: **subtracting a linear trend shifts the *mean* of the
  year-over-year changes and leaves their *variance* exactly unchanged**, since
  `diff(a + b·x)` is the constant `b`. A trend fixes bias; only a year effect addresses
  spread. Pinned by a test.
  - **The three-point MIX is the only quantity where a trend is worth extrapolating** —
    `fg3a_pct`, the three-point share of attempts, has trend R² **0.93** at **+3.58%/season**
    against `stl` at R² **0.03**. Everything else is shock.
    **⚠️ True of the league SERIES and false of the HEAD** — see the ablation bullet below.
    - **The shot-attempt basis sharpened this, and the decomposition is the point.** The
      retired `fg3a` *count* series read R² 0.93 at **+4.07%/season**; splitting it into
      volume × mix shows that **+0.47%/season is total shot volume** (`fga`, R² 0.84, a 1.14×
      band over 30 seasons) and **+3.58%/season is the mix**. The three-point revolution is
      almost entirely a change in *which* shots are taken, not in how many.
  - **`fta` is the sharpest shock case and it is refereeing**: a 1.21× band, 4.3% yoy sd,
    past ±5% in 9 of 29 transitions — **+7.6% in 2004-05** (hand-checking crackdown) and
    **+8.6% in 2025-26**. The three-point mix's worst year is **−24.7%**, the 1997-98
    three-point line moving back.
  - **The cost is measured on the no-fit floor, and the lag is CHECKABLE rather than
    asserted.** The floor carries prior per-36 forward, so it lags any league move by exactly
    one season — which predicts that a season's bias carries the **opposite** sign to that
    season's league move. On validation it does, in **13 of 14** cells, correlating at
    **−0.944**. `fta` is the case in both directions: **−4.4%** against the league's +7.3%
    rise into 2022-23, then **+10.7%** against its −7.5% fall into 2023-24. No fitted head
    corrects it. ✅ `make season-effects` (`league_yoy_pct` / `opposes_league_move` in
    `season_effects_carry_forward_bias.csv`).
    - **⚠️ Never quote the pooled `all` column on its own — it reports a lag as a level, and
      that is how the retired reading went wrong.** On the held-out seasons this bullet said
      **`fta` −7.0%** (**−10.7%** in 2025-26), `blk` **+6.2%** in *both*. On validation `fta`
      pools to **+2.9%** while swinging −4.4% → +10.7%, and **`blk` reverses sign** (+4.3% →
      −5.4%, pooling to −0.8%) against a claim that it was "drift, not noise". The reversal
      is the lag working correctly, not a defect: `blk`'s league rate moved **−1.5%** into
      2022-23 and **+10.6%** into 2023-24, so the floor was owed opposite-signed errors.
      `blk`'s trend R² is **0.118** on a **−0.14%**/season slope, so there was never drift
      to see, and two adjacent seasons moving the same way is what noise looks like half
      the time. The
      *relationship* is the finding and it is sharper on the deciding split: the retired
      cells oppose their league move in only **10 of 14** (r = **−0.865**).
  - **This outranks the shared-β correlation the Stan work was built for.** A league shift is
    perfectly correlated across every player, so it does not diversify: −7% on free throws is
    −7% on a whole roster's free-throw points, against **+0.2%** for shared-β on a 15-man
    roster. ✅ **Now measured, and the gap is ~95× rather than the recorded order of
    magnitude** — a year effect widens a 15-man roster's season-total dk_pts spread by
    **+11.6%** and the whole 791-player board's by **+278%** (against shared-β's +6.4%).
  - **Rule changes are announced in the summer**, so a manual league-level override is
    legitimate point-in-time information — unlike anything drawn from inside the season.
    ✅ The path is live and empty: `stan.season_terms.league_override` in
    `configs/default.yaml`, `{component: {season: multiplier}}`.
  - **⚠️ Two confounds sit inside the window and they are different SHAPES.** ✅
    `make season-effects` (`regime_tests` → `season_effects_regimes.csv`). COVID (2019-20,
    2020-21) is a *transient regime* and gets an indicator that is zero for the forecast
    season; the Player Participation Policy (2023-24) is a *permanent* rule change and gets a
    break. Modelling COVID as a break would put the whole post-2021 series on the wrong
    intercept. **COVID is undetectable at league-rate level — 0 of 17 series significant**
    (`gp_share [30+ mpg]` +0.22%, p = 0.91), because the shortened schedules cancel in a
    rate. **The policy break is real and role-graded**: a level-only break reads **−4.63%**
    on `gp_share [30+ mpg]` at **p = 0.008** and −4.43% on `gp_share [all]` (p = 0.024),
    against **+5.37%** and *not* significant for `gp_share [<12 mpg]` — the sign flips
    across role, which is the availability plan's era gradient arriving as a dated policy
    step from a different estimator.
    - **Read `regime_seasons` before `next_season_shift_pct`.** Every break arm has only
      **3** post-break seasons, so the level+slope form fits its slope on three points and
      then moves the one-season-ahead forecast by up to **+22.1%** (`gp_share [<12 mpg]`)
      and **+18.8%** (`stl`). That is noise. **The break test disqualifies extrapolating a
      trend across 2023-24 rather than supplying a corrected one**; `policy_break_level` is
      the stable form and ships beside it.
  - **The year shocks are NOT one common factor, so a simulator draws one per head.** ✅
    `make season-effects` (`year_shock_correlation` → `season_effects_shock_correlation.csv`,
    136 pairs over 30 seasons). Detrended log league rates correlate at a mean of **+0.011**
    — no common factor at all — but mean |r| is **0.309** and 18.4% of pairs exceed 0.5, so
    they are not independent either: the structure is in specific **pairs**, the largest
    now being `fga`–`reb` at **+0.838**. Detrending is load-bearing — two series that both
    drift upward would otherwise correlate through their trends, which is drift and not
    shock. `stan_utils.YearTerm` takes a `stream` per head for this.
    - **⚠️ The recorded strongest pair, `fg2a`–`fg3a` at −0.833, was the 3PA/2PA
      substitution appearing in the year-shock structure — and adopting the shot-attempt
      basis removed it from this table too.** That is the same coupling, measured a third
      independent way (after the residual copula and the joint NLL), and it is now gone by
      construction rather than carried. Its removal also flips the *mean* from −0.009 to
      +0.011: one large negative pair was holding the average below zero, which is a useful
      reminder that a mean over 136 pairs is not a robust summary. The conclusion is
      unchanged — no common factor, structure in specific pairs — and the largest surviving
      pair is a plain positive one between two volume series.
- **✅ THE ABLATION RAN — `make season-terms`. No head ships a season term, and neither the
  2026-08-04 shot-attempt rerun nor the 2026-08-07 move to validation-only overturns that.**
  **54 fits, 0 divergences, max R̂ 1.0105, 79.3 min** — half the fits and 40% of the compute
  of the 108-fit both-splits run, because the test side is no longer fitted at all.
  - **⚠️ READ THIS BEFORE QUOTING THE SELECTED ARMS.** On the July head list almost every
    head selected `base`. Since the shot-attempt basis **11 of 13 select a season term** —
    `trend` ×5, `year` ×3, `trend_year` ×3, `base` only for `reb` and `stl`. That looks like
    a reversal and is not: **10 of the 13 margins are under 1% of the selection metric**
    (`fg3m|fg3a` 0.01%, `fg3a|fga` 0.05%, `ftm|fta` 0.09%, `gp` 0.11%, `min` 0.20%,
    `fga` 0.26%, `blk` 0.35%, `fta` 0.79%, plus the two `base` heads at exactly 0). An
    ablation whose winner flips across a change that does not touch most of its heads is
    **measuring noise**, and that is a stronger statement of "the terms are worth nothing"
    than the original run could make — it now has a replication test behind it, and it
    failed.
    - Only two heads move on a margin worth a second look: **`ast` at 2.21%** (`trend_year`)
      and **`fg2m|fg2a` at 1.88%** (`trend`). Neither is a shot-attempt head, so neither is
      explained by the basis change.
    - **The margin is read on each head's OWN selection metric** — `val_nll` for conversions,
      `val_crps` for everything else. `src/docs_audit.py` inferred it from whether `val_nll`
      was populated, and the 2026-08-07 run started writing `val_nll` for the count heads,
      silently switching them onto a column they are not selected on. The tell was `stl`
      reporting a 0.008% margin while selecting `base` — an arm that is the minimum by
      construction can only have margin zero. Key on `kind`, never on which columns happen
      to be filled.
    - **⭐ The 2026-08-07 run reproduced every per-head validation figure to five decimals
      and all 13 selected arms — and that is a DETERMINISM check, not a replication.** The
      validation fits were always validation fits at the same seed and the same 500/500
      budget, so dropping the test side could not move them. It proves the conversion
      changed nothing it should not have; it is *not* independent evidence for the verdict.
  Four arms per head (`base`, `trend`,
  `year`, `trend_year`) on each head's already-selected spec, plus `trend_x_role` and
  `trend_x_role_year` for availability; selected on validation (2022-23/23-24), every arm
  quoted against its no-fit floor. Full verdict in `docs/predictions-plan.md`.
  - **The trend is refuted most sharply on the one quantity that predicted it.** ⚠️ This
    bullet describes the RETIRED `fg3a` count head and its figures are preserved as the
    record of that basis — `fg3a` is no longer fitted, and its successor `fg3a|fga` selects
    `year` on a 0.05% margin. `fg3a`
    selected **`base`** (val CRPS **33.247** against trend 35.443 and year 34.309), a
    trend flipped its held-out bias from **−3.74% to +8.44%**, and a year effect drove it to
    **−9.01%** by absorbing drift as shocks and then zeroing them. Two measured mechanisms:
    the three-point climb **decelerated** — +3.58%/season over 30 seasons but **+1.61%/season
    over the last six** — and the head's dominant feature `log(fg3a_p36_lag1)` already
    carries the league level, so a trend adds a second correction on top of one already
    there. **The trend worsened held-out bias on 6 of 8 count heads**, and where it won on
    test it won on heads with no era story (`stl`, trend R² 0.03). That is overfitting.
    - ⚠️ **On validation the count is 3 of 7 and the heads are different ones** (`fga`, `reb`
      and `stl` worsen; `fta`, `ast`, `blk` and `tov` improve). Do not read that as the trend
      being rehabilitated — the basis and the split both changed, and these are sub-1% bias
      moves on heads whose whole season-term margin is under 1%. What survives is that the
      trend moves bias in **both directions across heads**, which is what the cancellation
      argument below rests on.
  - **⭐ A trend's apparent win on season-total dk_pts is cross-component cancellation, and
    the 2026-08-07 validation run RESTORES that reading after the test column had briefly
    undermined it.** On the 773-row validation frame all-trend scores MAE **105.76** against
    base's **105.71** — it does not win at all, it is 0.05 dk_pts *worse* — while flipping
    bias from **−16.44** to **+10.21**. The component biases behind it move in *both*
    directions, so what the aggregate does is those errors cancelling in the DK sum: same
    mechanism as the 8.30× cancellation on `teammate_assist_supply`, producing a false
    positive for a season term.
    - **⚠️ This closes the item that was flagged here as needing a human decision.** The
      test-only table read `trend` **106.06** / **−13.57** against base **108.56** /
      **−36.89**, i.e. `trend` improving *both* MAE and bias, which the cancellation story
      could not explain. On the split that is allowed to decide, the anomaly is gone and the
      original July shape is back. **No decision is owed** — the mechanism was never in
      doubt, only the arithmetic on one held-out frame.
    - Coverage is **0.45 / 0.76 / 0.91** against a nominal 0.50 / 0.80 / 0.95 on every arm,
      and no season term fixes it. That is the independent-draw composition missing the
      residual copula, not an argument about season terms either way.
    The season-total table is **uniform-arm** — confirmation, never selection — and since
    2026-08-07 it and the per-head table read the same split.
  - **⭐ The oracle bounds the whole question at ≤5% of MAE, and this is the number to
    remember.** `oracle_league` rescales each scored season by its own realized total — a
    perfect per-season league multiplier, and therefore the ceiling on a trend, a year effect
    and a manual override alike. As a share of base MAE: `fta` **4.97%**, `reb` 2.58%, `tov`
    **2.36%**, `ast` 1.29%, `blk` **1.17%**, `stl` 0.36%, `fga` −0.00% — **median 1.29%**.
    `fga` is negative because a perfect rescale can cost a fraction on a head whose league
    level barely moves, which is the cleanest statement that there is nothing there to win.
    - **⚠️ Quote the magnitude, never the ORDER.** On `test_mae` this read `stl` **3.11%**,
      `blk` 2.32%, `fta` **2.16%**, `reb` 1.71%, `fga` **0.81%**, `ast` 0.58%, `tov`
      **0.01%**, median **1.71%** — `stl` fell from first to sixth and `fta` rose from third
      to first across two scored seasons, while the median moved 1.71% → 1.29%. A ranking
      that reorders itself like that is season-specific accident. The one substantive change
      is the ceiling's height: quote **≤5%**, not ~3%.
  - **The year effect behaves exactly as designed, and recovers the league independently.**
    Median validation ΔR² vs base is **−0.00027** across thirteen heads — the mean-zero
    property surviving contact with data. And `sigma_year`, fitted by NUTS on player-season
    rows, lands within 20% of the *directly measured* league yoy sd on **3 of 7** count
    heads (`blk` 0.85×, `tov` 0.89×, `reb` 1.07×), and within a factor of 1.4 on all seven.
    **The shot-mix head is the tell**, at **1.84×** on the one quantity with a real secular
    trend: with no trend term the year effect absorbs *drift* as a sequence of shocks and
    then zeroes it at prediction time.
    - **⚠️ Every σ moved on 2026-08-07 and it was a latent DEFECT, not sampler noise.** The
      old `year_*` columns were written from the **test arm's** fit (trained on 27 seasons)
      into a row whose CRPS came from the **validation** arm (25) — one row, two models,
      visible in `year_n_train_seasons`. Nothing leaked; the σ compared against the league
      series simply was not the σ of the model the rest of the row described. The recorded
      **5 of 8 within 20%** (`blk` 0.99×, `tov` 0.94×, `fta` 0.87×, `stl` 0.82×, `reb`
      1.18×, `fg3a` 2.36×) is that contaminated column. Fitting on two fewer seasons shrinks
      σ on six of seven heads, pushing `fta` and `stl` out of the band.
  - **Mean-zero on the linear predictor is NOT mean-zero on the response.** Under a log link
    `E[exp(σz)] = exp(σ²/2)`, so integrating the year effect *raises* every predicted count.
    Measured at **1.000–1.008** across the heads — real, negligible, and reported rather than
    assumed (`YearTerm.response_multiplier`).
  - **What the year effect IS worth is joint spread, and only that.** Roster season-total
    dk_pts sd: **+7.95%** at 12 players, **+10.4% at 15**, +21.0% at 30, +83.5% at 150 and
    **+254%** across all 773 — against shared-β's +0.2% / +0.2% / +0.3% / +1.1% / +6.4%.
    Treat as an **upper bound**: `fga`, `ast` and `fg3a|fga` have a fitted σ above their
    measured league movement, so some of it is player heterogeneity. Give the simulator σ
    from `season_effects_summary.csv` (`yoy_sd_pct`), not the fitted value. **⚠️ Supersedes
    +8.9% / +11.6% / +22.1% / +91.1% / +278% on the 791-player test board** — the whole-board
    row is keyed on board size, so it is a different row rather than a moved value, and the
    shared-β column is a different board and a different head, so ~50× is indicative rather
    than a like-for-like division.
  - **The minutes head adopts a season term, and it is the one selection that reproduced
    across a change that should not touch it** — `logit_own_spline` + `year` survived both
    the shot-attempt basis change and the split conversion to four decimals, against 9 heads
    that flipped on the first. That invariance is what distinguishes a real selection from
    the noise-dominated ones above. `year` wins at val **143.81** against base 144.09.
    (Retired held-out column: **146.54** against **147.02**.)
    - **It also falsifies a recorded hypothesis.** `docs/availability-plan.md` proposed the
      head's bias as "the signature an era effect would leave"; a trend makes it **worse by
      10.8 minutes** (**−13.0** on `year` and −14.2 on `base` against **−25.0** on `trend`),
      so it is shrinkage toward a 30-season mean, **not** an era effect. ⚠️ The recorded
      **−38.2 / −56.6 / −41.0** are the held-out column and the *level* is withdrawn — on
      validation the FLOOR is the biased one at **+23.9** while the fitted arms run −13 to
      −27. What reproduces is the **gap** (fitted arms sit 27–51 minutes below the
      carry-forward on both splits), and the falsification survives because "a trend makes
      it worse" is a statement about the trend, not about which seasons it was measured on.
  - **The availability season × role interaction is a validation NULL.** `trend_x_role` and
    `trend_x_role_year` are the **worst two arms of seven on validation** (10.078, 10.087
    against base 10.007) — the most expensive arms at 26 features, buying a loss. The
    selected arm, `trend`, is worth **0.010 games** of validation CRPS — nothing. The era
    effect is real in the league series and does not transfer into a better forecast.
    - **⚠️ Its finding WAS a val/test disagreement and that half is now unreproducible.**
      Those same two arms were the **best two on test** (10.742, 10.736 against base
      10.797), the exact false-positive shape this repo has shipped before, caught only
      because selection never read test. `src/models/held_out.py` now stops the test column
      being computed at all, so the contrast is preserved as the record of why the lock
      exists rather than as a re-runnable measurement. The verdict does not depend on it.
  - **Every arm under-predicts the bonus by 7–13%** (`base` **−11.7%**, `trend` **−7.6%**,
    `year` **−12.8%**) and **that level is mostly not the season term**: this composition
    draws the eleven heads independently given realized minutes, omitting the positive
    cross-component dependence a simultaneous threshold needs. It argues for the residual
    copula, not against a season term. ⚠️ Supersedes −14.8% / −10.2% / −16.5% (a 12–19%
    shortfall) on the test board; the ordering is identical and the gap narrowed ~3 points.
  - **`metric="dense_e"` is what made this Bayesian ablation affordable.** On the `blk` spline
    base: **13.4 s against 236.6 s**, treedepth saturation **0 against 35**, same posterior.
    This is the cheaper form of the QR-whitened basis this file already recommends for
    spline variants, and the ablation applies it to *every* arm so the metric cannot confound
    the contrast.
- **ALWAYS absorb season when regressing on 30 pooled seasons.** This has already produced
  one false finding. Pooled, `teammate_spacing` correlates +0.315 with next-season per-36
  points and `team_pace` +0.336; with season fixed effects they are +0.035 and +0.091.
  Spacing, pace and scoring all roughly doubled over the sample, so anything built from
  them tracks anything rising. `src/eda/context_value.py::season_dummies` does this.
- **`role_crowding` is a settled null** — retested under the corrected construction.
  r = +0.048 vs next-season per-36 pts with season absorbed (+0.020 once P's own prior
  per-36 style is controlled), −0.005 vs per-game dk_pts, ΔR² = +0.00001. The old
  endogeneity explanation ("prior-season minutes already absorbed it") is **falsified** —
  crowding on a season-S lineup that did not exist in S-1 is still a null. Remaining
  explanations: rosters are built to avoid redundancy, and `teammate_usage_load` measures
  the same mechanism directly and far better. Do not revive archetype-similarity crowding.
- **`teammate_usage_load` is the strongest own-team feature** (ΔR² = +0.0066 alone; the
  whole block is +0.0086, of which the usage family is +0.0080 and
  crowding/spacing/pace is +0.0000). It is the minutes-weighted mean of teammates' prior
  usage × 5. It exists *only* under the corrected construction — on the superseded S-1
  roster it measures +0.0001. Prefer it to the raw `teammate_usage_sum`, which grows with
  roster size on the inclusive frame.
- **Team context is built on roster(S) × stats(S-1)**, rosters derived from a player's
  first appearance inside his team's first 10 games (`features.team_context.
  roster_window_games`). 10 games → ~15-man roster, 96% of realized team minutes; 0 →
  11 players, 84%.
- **14.7% of season-start roster minutes have no usable S-1 row** — 8.7% true rookies, 5.1%
  sub-threshold, 0.9% returnees; p90 = 29.0% per team-season, max 50.4% (CLE 1997-98). Never
  drop them. ✅ `make context-value` →
  `outputs/eda/roster_coverage_profile_tier{A,B}.csv`.
  `season_matrix_roster_tier*.parquet` is the unfiltered twin built for this
  (14,569 / 6,942 rows vs 10,900 / 5,077); rows carry `stats_source` and `reliability`,
  team-seasons carry `roster_coverage`.
  - **⚠️ The recorded 15.9% / 9.4% / 5.4% / 1.1% is the *whole-season* roster, not the
    season-start one.** Measured across seven window sizes, window 82 gives
    15.79 / 9.16 / 5.52 / 1.11 — a match — while the shipped
    `features.team_context.roster_window_games = 10` gives 14.71 / 8.73 / 5.07 / 0.91. The gap
    is mechanical: **rookies and players absent the previous season arrive late**, so widening
    the window pulls in disproportionately many of them. The season-start figure is the one
    that describes a real bias, because it is the only roster knowable before the season and
    therefore the only one `team_context` aggregates. Both windows are emitted.
  - **Weight by realized season-S minutes, never by the S-1 minutes the aggregate uses.** A
    rookie has zero of the latter, so the circular weighting reports every roster as fully
    covered — the failure mode reading as a success.
    `team_context.coverage_report` existed for this and was never called; it now is.
  - **"No usable S-1 row" is relative to the *qualified* matrix**, so the four-way split cannot
    come from `stats_source`, which only knows prior / stale / rookie against the *inclusive*
    frame. `context_value.description_source` derives all four from the two matrices and its
    head-count mix reproduces `stats_source` to 0.000%.
  - Head counts and minutes are both reported and neither substitutes: a rookie is **14.1% of
    roster rows and 8.7% of roster minutes**.
- **Prior-season reliability is `0.924 · m/(m+66)`** in total minutes, fitted on 11,272
  consecutive-season pairs. Applied as a multiplier in z-space, which *is* shrinkage to the
  league mean. Staleness costs a further ×0.96 per season (lag-1…4: 0.921/0.881/0.853/0.826).
  Rates stabilize fast — the 200-minute qualification threshold is already at 0.75.
- **Bonus overdispersion is 0.10 for season-mean counts and 0.025 for per-game counts, and
  the two must not be interchanged.** ✅ `make component-targets`
  (`targets.bonus_calibration` → `outputs/eda/bonus_calibration.csv`) re-runs the
  calibration, so "do not change it without re-calibrating" is a target rather than an
  instruction. `BONUS_OVERDISPERSION` is the variance of the shared per-game Gamma frailty in
  `expected_bonus` — it makes each category's marginal negative binomial *and* induces the
  positive dependence the bonus needs. It is **not** a dispersion of `dk_pts` and is not
  fitted by any head.
  - **The season-unit claim reproduces exactly**: on 11,938 player-seasons with ≥200 season
    minutes, bias at 0.10 is **+0.0009** dk_pts/game (recorded +0.001) and independent
    sampling reads **22.7%** low (recorded 23%). The fitted optimum is **0.0968**.
  - **⚠️ "Good fit across minutes buckets" is withdrawn.** At 0.10 the per-mpg-bucket bias
    runs **−0.0142 at 12–18 mpg against +0.0291 at 30–48** — a 0.0433 spread that *cancels*
    to +0.0009. One scalar frailty cannot absorb minutes variation whose relative size differs
    by bucket, so the aggregate is right and the buckets are not.
  - **⚠️ The simulator must use `BONUS_GAME_OVERDISPERSION = 0.025`.** At the player-game unit
    (per-36 rate × that game's actual minutes — how `expected_dk_pts` is called and how the
    simulator will draw) minutes are no longer hidden inside the frailty, so the fitted value
    is **0.0248** — and at that value the fit holds in *every* minutes bucket (bias −0.0004 to
    +0.0007) rather than only in aggregate. Using 0.10 per game over-predicts the bonus by
    **+0.036 dk_pts/game for 30+ minute players**, who are exactly the ones it is worth most
    for. 0.10 stays as the shipped constant because it is correct for its documented unit and
    no production code reads it yet.
  - The recorded level pair (0.098 vs 0.127 on 11,627 player-seasons) came from an unnamed
    filter of roughly ≥250 season minutes; at the project's standing ≥200 threshold it is
    0.0951 vs 0.1231. The ratio and the bias, which are what the claim rests on, are
    unaffected. The docstring's "(2014-15 → 2025-26)" was also wrong — only 6,067
    player-seasons exist in that window.

### Stage-4 diagnostics — persistence, aging, target shape, feature diagnostics

All four run on the **inclusive** roster frame, season-absorbed and minutes-weighted.
Reproduce with `make persistence` / `make aging` / `make target-profile` /
`make feature-diagnostics`.

- **Persistence splits on *share* vs *conversion*, not on "percentage".** Season-absorbed,
  minutes-weighted, 11,272 pairs. Shot-mix and usage **shares** persist nearly as well as
  counts — `sco_pct_fga_3pt` 0.886, `usg_pct_fg3a` 0.870, `usg_pct_reb` 0.854,
  `adv_ast_pct` 0.845, `adv_usg_pct` 0.786 — while **conversion** percentages do not
  (`fg3_pct` 0.500 … `efg_pct` 0.281). Shares are first-class features; only the
  conversion side gets shrunk. Stickiest counts: `reb` 0.941, `oreb` 0.924, `ast` 0.920,
  `fg3a` 0.908, `dreb` 0.906, `blk` 0.902. Tier B tracking is equally sticky —
  `drv_drives` 0.922, `pu_pull_up_fga` 0.918, `pass_potential_ast` 0.913,
  `hus_contested_shots_2pt` 0.909 — so the tracking families are worth their 13-season cost.
- **Whole families are noise; do not feed them.** Clutch (`clu_*`, median r 0.25, every
  column ≤ 0.55), the rating columns (`def_rating` 0.116, `net_rating` 0.180, `off_rating`
  0.285, `e_pace` 0.212), `plus_minus` 0.477, `sl_backcourt_*` ≈ 0.01, and every tracking
  `*_pct` conversion column (`drv_drive_fg_pct` 0.162, `pu_pull_up_fg3_pct` 0.115). These
  are also the largest **era gaps** — `adv_e_pace` reads 0.574 pooled against 0.212
  within-season, `def_opp_pts_paint` 0.715 vs 0.425 — so a *pooled* ranking promotes
  exactly the columns worth dropping.
- **Minutes weighting changes the feature *ranking*, not just regression coefficients.**
  Median **+0.214** r across the 72 minutes-weighted Tier A columns (mean +0.189). Worst:
  `stl` 0.743 weighted vs 0.401 unweighted, `tov` 0.790/0.456, `blk` 0.902/0.626,
  `plus_minus` 0.477/0.154. Unweighted, the low-count defensive stats look like noise when
  they are among the stickiest things a player has.
- **Games played is the least persistent quantity in the project — r = 0.317.**
  Season-absorbed and minutes-weighted over the same 11,272 pairs, against `min` per game
  0.779, `dk_pts` per game 0.869, `min_total` 0.640, `dk_pts_total` 0.760. `min`/`gp` are
  held *out* of `persistence.csv` as volume columns; reproduce with
  `lagged_pairs`/`pair_weights`/`demean_within` from `src/eda/persistence.py` over
  `season_matrix_roster_tierA.parquet`. Games played is simultaneously the **largest lever
  on the season total and the least predictable input** — shrink the availability head hard
  toward a league/age baseline rather than toward the player's own prior GP.
  **See `docs/availability-plan.md`**, reproduced by `make availability` +
  `make availability-profile` → `outputs/eda/availability_profile.csv`. Internal ceiling is
  **R² ≈ 0.24 in-sample**; GP is **~20× overdispersed** against a binomial (22.7× full
  window, 19.8× appearance; 26.7% of established rotation players below 60 games), so the
  head must emit a *distribution*, not a point estimate. Two nulls: a 3-year availability
  average does not beat 1 year (r 0.392 vs 0.398) and longest absence spell persists at
  r = 0.090 — there is no durability latent to extract. Prior **MPG predicts next-season GP
  as well as prior GP does** (R² 0.159 each).
- **Do not minutes-weight the availability head.** The rule everywhere else in this project
  is the opposite, so this is the exception to state explicitly: minutes weighting suppresses
  per-36 rates measured over a few garbage-time minutes, which is real measurement error.
  Games played has none — "he played 12 games" is exact — so weighting down-weights exactly
  the injured seasons the head exists to predict. It halves the ceiling (R² 0.236 → 0.116)
  and flips the MPG-vs-GP ordering. `availability_profile.csv` reports both columns; use the
  unweighted one for availability and the weighted one for rates.
- **The availability head is a beta-binomial GLM, and the simulator is not built.** Held out
  on 2024-25/2025-26 (10,361 train / 911 test), CRPS in games: **GLM 10.795**, GBM 10.888,
  ridge 10.896, league/age baseline 13.614. (Before the playoff-workload block: 10.914 /
  11.04 / 10.98 / 13.614.) Gradient boosting does not beat a 19-feature GLM, so
  the plan's own decision rule says stop. `src/models/availability.py`, `make
  availability-model`. Two things confirm the distribution is the working part: the fitted
  dispersion lands at **20–30× implied overdispersion**, independently recovering the ~20×
  in `availability_profile.csv`, and PIT is near-uniform (KS 0.08–0.11 against the
  baseline's 0.17). Held-out R² 0.268 is **not** comparable to the 0.236 ceiling — that one
  is in-sample and season-absorbed.
- **The availability head is worth ~210 dk_pts of season-total MAE, and availability is
  measurably the larger half of the error.** `make season-total`,
  `src/models/season_total.py` — composes `season_total = gp × dk_per_game_played` holding
  a **fixed** rate model (ridge, validation R² 0.783 on the rate) and varying only the
  games-played treatment, so the contrast is the head and nothing else. Selected on
  **validation** (2022-23/23-24), 9,421 train / 873 rows:

  | GP treatment | MAE | RMSE | R² | bias | CRPS |
  |---|---|---|---|---|---|
  | full season (naive) | 610.8 | 751.8 | 0.374 | **+523.3** | — |
  | prior GP carried forward | 417.9 | 560.5 | 0.652 | +19.4 | — |
  | league/age baseline | 461.6 | 565.9 | 0.645 | +12.9 | 329.1 |
  | **beta-binomial head** | **400.5** | **514.0** | **0.707** | **−3.1** | **287.3** |
  | *spell process* (Gate E) | *406.8* | *520.3* | *0.700* | *−16.6* | *291.8* |
  | *oracle rate* | *261.9* | *367.0* | *0.851* | *−41.3* | — |
  | *oracle GP* | *214.4* | *287.5* | *0.908* | *+2.0* | — |

  ⚠️ **This whole table was a TEST evaluation until 2026-08-05** and read 646.3 / 475.0 /
  476.0 / **435.1** / 302.7 / 221.3 MAE on 10,294 train / 896 test. It is the module where a
  games-played treatment is *chosen* — six of them are scored against each other and Gate E
  asks it to arbitrate — so it had to move to the split that is allowed to decide. Nothing
  about the finding reverses: the head still beats every non-oracle treatment, and
  `oracle_gp` still beats `oracle_rate`. **One ordering does flip**: `league_age` (461.6) is
  now clearly worse than `prior_gp` (417.9), where on test they were a tie (476.0 vs 475.0).
  - ⚠️ **The R² column was separately corrected 2026-07-30**, when it read 0.10 / 0.47 /
    0.55 / 0.59 / 0.78 / 0.88 against the test table above and no single construction
    reproduced that set. The diagnosis is worth keeping because the failure mode is generic:
    `full_season`, `prior_gp` and `oracle_gp` never touch the availability head, so no change
    to that head can move their R² — yet `full_season` was the most wrong, by 0.041. Since
    `r2 = 1 − SSE/ss_tot` and equal RMSE pins SSE, only `ss_tot` could differ, and solving per
    row gave six mutually inconsistent denominators (0.954× to 1.033×). The column had been
    hand-typed and never refreshed when the rest of the table was.

  −210.3 MAE (−34.4%) against assuming a full season and −17.4 against carrying prior GP
  forward. **The oracles settle which half dominates**: perfect games played gives 214.4
  against perfect rate's 261.9, so availability carries 186.1 dk_pts of the remaining error
  and the rate 138.5 — the plan's thesis, now a measurement rather than an assertion. (The
  test table put the same gap at 213.8 against 132.5; the *ordering* is what the thesis
  rests on and it survives the move, with a narrower margin.)
  `oracle_gp` is invariant to the head by construction, which makes it the check that a
  change to the GP treatment moved only what it should.
  Two calibration notes: the naive treatment's **+523.3 bias** is most of its error (it
  assumes 82 games for everyone), and the head's **−3.1** is near zero, which is the
  distribution doing its job. CRPS is on the pushforward of the GP pmf through `k → k·rate`
  — `crps_from_atoms`, the O(K) kernel reduction, pinned against the literal double sum in
  `tests/test_season_total.py`.
- **⭐ Gate E of the games-played plan RAN here on 2026-08-05 and the spell process FAILS
  it** — 406.8 MAE and 291.8 CRPS against the incumbent's 400.5 / 287.3. **It had been
  recorded as a ✅ on test with a margin of 0.03 dk_pts** (435.1053 against a 435.1352 bar),
  which is seven parts in a hundred thousand and was never evidence of anything. That is the
  third gate in that head to reverse on moving off the test split.
  `season_total.gate_e` now reads its bars out of the incumbent's own row on whichever table
  it is scoring, so a hard-coded threshold cannot smuggle a test-set number back in.
  **It does not speak to the `hybrid` arm**, whose games-played pmf is the incumbent's by
  construction and which would therefore tie Gate E exactly — the same marginal-gate
  category error as Gate D, one level down.
- **Gains shrink on established rotation players, but the ordering holds** — 582.9 naive →
  451.3 head (−22.6%), against −34.4% overall, because regulars miss less time. Do not
  quote the aggregate figure as if it applied to the players a DFS user cares about most.
  (Superseded test figures: 651.3 → 499.4, −23.3%.)
- **Fit the beta-binomial with an analytic gradient, and check the denominator first.**
  Two failures here were silent, not loud. (1) 13 player-seasons (0.12%) have
  `gp > team_games` — traded players whose two teams' schedules overlap — and since the
  log-likelihood is a *sum*, those rows make it non-finite at **every** ρ. Use
  `n = max(team_games, gp)`. (2) A joint L-BFGS-B fit over 17 parameters with *numeric*
  gradients does not converge: the objective is ~1e5 (a sum over 10,000 log-densities) while
  a finite-difference step moves it by ~1e-3, so it stops on gradient noise and returns
  coefficients near zero — a fitted model with R² −0.09. Supply the analytic gradient
  (`a + b` is free of μ, so its digamma terms cancel) and alternate β with ρ.
- **Splitting absences by *reason* is NOT the null the plan expected — settled on the full
  backfill.** `make boxscore-status` completed 2026-07-28 (25,706 of 25,709 games,
  2006-07 → 2025-26), mean `status_coverage` 0.696 and ~99.8% per season from 2006-07 on.
  Re-measured on **7,673 season pairs** (full window, in-sample, season-absorbed,
  unweighted): prior `gp_share` alone R² 0.2353 → +`missed_games` 0.2365 → **+ the 8-way
  reason split 0.2648**, against a shuffled-row null of 0.2363 (sd 0.00041). That is
  **+0.0285 above chance, ≈69 sd**, and it confirms the provisional 2-pair reading
  (+0.0266) rather than shrinking it. The aggregate `missed_games` remains worth
  **+0.0012** — the split is essentially the entire effect. The carrying reasons invert the
  intuition: `missed_scratch` −0.353, `missed_inactive` −0.233, `missed_not_rostered`
  −0.164 against next-season `gp_share`, while **`missed_injury` is +0.034** — the
  *rotation* reasons predict availability and the injury reason does not, matching the
  longest-spell null (0.090). **`missed_scratch`
  persists at 0.469**, more than any other availability column including `gp_share` itself
  (0.317), which is the strongest single statement of the plan's "much of what looks like
  availability is rotation status". `missed_injury` counts only injury-flagged players who
  **dressed**, so it partly marks "was a rotation player"; the genuinely unavailable are
  `inactive`, for which the endpoint states no reason at all.
  **`models.availability.FEATURE_COLS` does not consume any of these columns yet** — that
  is the one evidence-backed feature change outstanding on the head.
- **Playoff workload earns its place on the head, but by *selection*, not fatigue — every
  sign is the opposite of the mechanism it was built for.** `make availability-model`
  (`workload_ablation` → `outputs/predictions/availability_workload_ablation.csv`), held
  out on 2024-25/2025-26, everything fixed but the feature list:

  | variant | features | CRPS | vs baseline | held-out R² |
  |---|---|---|---|---|
  | baseline | 15 | 10.914 | — | 0.268 |
  | **+ playoff workload** | **19** | **10.795** | **−0.119** | **0.283** |
  | + playoff only | 18 | 10.817 | −0.097 | 0.281 |
  | + `career_minutes` only | 16 | 10.883 | −0.031 | 0.271 |

  Worth **+6.2 dk_pts of season-total MAE** (441.3 → 435.1; −9.5 more on established
  rotation players, 508.9 → 499.4), with `oracle_gp` unchanged at 221.3 — the internal check
  that only the GP treatment moved. ⚠️ **That trio is measured against the superseded TEST
  season-total table** and is kept because it is the only record of the pre-workload run;
  the ablation itself has not been re-measured on validation, since `make availability-model`
  still scores the held-out split. In-sample the block is **+0.0178 above its own shuffled
  null** (0.2998 vs 0.2820, sd 0.0002).

  **The direction is the finding.** Every single-season playoff column predicts *better*
  next-season availability — `playoff_mpg` r = +0.282 raw, +0.055 controlling for
  `gp_share` and MPG; `playoff_minutes_share` +0.108 controlled. Playoff participation
  marks a good player on a good team, and that selection effect beats fatigue outright.
  The only column pointing the way fatigue would is **`career_minutes`, at −0.067**
  controlled — cumulative mileage, which is what `make aging` implied when it put the
  availability arc at −54% peak-to-37. Same inversion as `missed_injury`.
  - **`total_minutes_incl_playoffs` is a measured null and is deliberately not a feature.**
    It was built on the true observation that `total_minutes` undercounts real mileage;
    swapping it in *lowers* in-sample R² (0.2843 → **0.2816**), because folding playoff
    minutes into the total mixes team quality into a clean regular-season workload measure.
    Keep the two effects in separate columns. Pinned by a test so it does not get "fixed"
    back in.
  - The GBM's margin narrowed but the ordering held — GLM 10.795 vs GBM 10.888, against
    10.914/11.04 before, so the block helped the GBM slightly more (−0.152 vs −0.119). The
    plan's stopping rule still says stop.
- **Nonlinear terms are a null for games played and NOT a null for minutes — and the split
  between those two is the useful part.** `make availability-model`
  (`nonlinearity_ablation` + `minutes_nonlinearity_probe` →
  `outputs/predictions/availability_{nonlinearity,minutes_nonlinearity}.csv`). Cubic
  B-splines with linear extrapolation (knots from **training** quantiles only) or
  quadratics, over the nine continuous columns; `age_sq` is already in the linear baseline.

  | variant | p | val CRPS | test CRPS | selected |
  |---|---|---|---|---|
  | **linear** | 19 | **10.006** | 10.795 | **✓** |
  | quadratic | 27 | 10.037 | *10.749* | |
  | spline k=4 | 54 | 10.041 | *10.761* | |
  | spline k=5 | 63 | 10.054 | *10.751* | |

  **The test column prefers every curved variant and none of them replicate.** This is the
  methodological trap worth remembering: a *paired* bootstrap on the 911 test rows puts the
  quadratic gain at **−0.047, 95% CI [−0.079, −0.015], P(Δ<0) = 99.7%** — and it is still a
  false positive, because a paired interval says a difference is consistent *within one
  sample*, not that the sample was representative. Select on a validation split; quote test
  for confirmation only. The GBM arm corroborates from a different direction: a fully
  nonparametric learner on the same features still loses to the linear GLM.
- **For minutes per game the same test replicates, and it is *not* the age arc.** Ridge
  probe (the minutes head is not built), predicting next-season MPG: linear R² val 0.6768 /
  test 0.6670, quadratic **0.6914 / 0.6746**, spline k=4 0.6930 / 0.6736 — both splits move
  the same way. Splining one column at a time attributes it:

  | column splined | Δ val R² | Δ test R² | replicates |
  |---|---|---|---|
  | **`minutes_per_game_lag1`** | **+0.0124** | **+0.0077** | **yes** |
  | `total_minutes_lag1` | +0.0008 | +0.0011 | yes |
  | `age` | −0.0008 | −0.0007 | no |
  | `career_minutes_lag1` | +0.0005 | −0.0006 | no |
  | everything else | ≤ +0.0007 | ≤ 0 | no |

  So the intuitive story — young players ramp up, prime players play heavy minutes, veterans
  get load-managed — is real (`make aging`: MPG peaks at 27 and falls to 0.515 by 37) but
  **already absorbed by `age + age_sq`**; a spline on age is actively *worse*. The
  nonlinearity that pays is a **floor at the bottom of the prior-MPG range, not a ceiling at
  the top**. Mean next-season MPG against prior: 4.3 → 10.5 (**+6.1**), 9.3 → 12.4 (+3.1),
  15.1 → 15.8 (+0.7), then roughly parallel decline of −2.0 from 26 mpg up. The local slope
  *rises* 0.39 → ~1.0 with prior MPG. Part of that is survivorship — a 4-mpg player needs a
  next-season row to appear at all — the same bias that makes cross-sectional age curves
  worthless here.
- **The component rate side has a no-fit floor that is nearly the whole model, and every
  proposed head must be quoted against it.** `make component-rates`
  (`src/models/component_rates.py` → `outputs/predictions/component_rate_metrics.csv`).
  **`carry_forward` = prior per-36 rate × actual minutes / 36, no fitting at all**, scores
  validation R² **0.81–0.95**, and the best of seven fitted variants beats it by only
  **+0.0013 to +0.0334**. A component head that does not clear it is not a model. Every
  output row carries `beats_floor` and `run` warns when no variant clears it for a head,
  because that is also the signature of the regularization trap below.
  ⚠️ **Measured on the held-out seasons until 2026-08-05, where it read 0.82–0.94 and
  +0.0019 to +0.0203.** This module was the only head in the project with **no guard at
  all** — it defined its own `split_seasons` rather than importing the shared one, so
  `src/models/held_out.py`'s lock, which lives inside `availability.split_seasons`, never
  saw it. The copy was behaviourally identical; deleting it changed nothing but the guard.
  The finding is unchanged and the floor is if anything harder to beat.
- **For the component count heads the answer is *scale*, not curvature: put the own prior
  rate in on the log scale.** Season-collapsed Poisson/NB (`y_season ~ Poisson(M·e^{Xβ})`,
  M = season minutes), 10,194 player-seasons with a ≥200-minute prior season, 29 seasons,
  scored on **validation** (2022-23/23-24), 8,630 fit / 773 rows. R² on the season total:

  | head | no-fit floor | linear | **log(own)** | spline(own) | + age×own |
  |---|---|---|---|---|---|
  | `fga` | 0.9514 | 0.9544 | 0.9586 | 0.9589 | **0.9591** |
  | `reb` | 0.9505 | 0.9181 | **0.9514** | 0.9497 | 0.9512 |
  | `ast` | 0.9195 | 0.8569 | 0.9222 | **0.9255** | 0.9225 |
  | `blk` | 0.8103 | 0.6510 | 0.7748 | **0.8291** | 0.7799 |
  | `fta` | 0.8765 | 0.7993 | 0.8922 | 0.8817 | **0.8932** |
  | `stl` | 0.8386 | 0.8556 | 0.8682 | 0.8693 | **0.8693** |
  | `tov` | 0.8966 | 0.9091 | 0.9171 | 0.9162 | **0.9173** |

  > ⚠️ **This table was a TEST evaluation until 2026-08-05** and read, in the same layout:
  > `fga` 0.9464 / 0.9455 / 0.9513 / 0.9517 / 0.9522, `reb` 0.9424 / 0.9278 / 0.9441 /
  > 0.9436 / 0.9442, `ast` 0.9197 / 0.8601 / 0.9229 / 0.9262 / 0.9236, `blk` 0.8407 /
  > 0.6375 / 0.8204 / 0.8605 / 0.8228, `fta` 0.8673 / 0.8449 / 0.8689 / 0.8692 / 0.8720,
  > `stl` 0.8194 / 0.8170 / 0.8369 / 0.8397 / 0.8338, `tov` 0.8845 / 0.8828 / 0.8915 /
  > 0.8913 / 0.8916. **Every conclusion in this block survives** — `linear` is still
  > catastrophic on `blk`, `log(own)` still recovers nearly all of it, splines still pay
  > only where the prior is skewed. Two things sharpened: `blk`'s spline gain grows from
  > +0.040 to **+0.054**, and `stl`'s from +0.003 to +0.001 while its *linear* arm now
  > *beats* its floor.
  >
  > ⚠️ **Refreshed 2026-08-03 for the shot-attempt basis.** The `fg2a` row
  > (0.9194 / 0.9089 / 0.9245 / 0.9248 / 0.9260) and the `fg3a` row
  > (0.9036 / 0.5197 / 0.8791 / 0.9088 / 0.8784) are retired with the two-count basis, and
  > `fga` replaces both — both sets of figures are the retired basis on the retired split.
  > **`fga` is still the best-behaved count head in the project**: floor **0.9514**, the
  > highest of the seven, and `linear` costs it nothing at all (it is +0.0030 *above* the
  > floor) against 0.5197 for the `fg3a` it replaced, because a total is much less skewed
  > than its three-point part.
  >
  > ⚠️ **The `+ age×own` column was corrected 2026-07-31, and the correction is a change of
  > *variant*, not of value.** It previously read 0.9437 / 0.9260 / 0.9249 / 0.9083 / 0.8558 /
  > 0.8708 / 0.8381 / 0.8918, which no row of `component_rate_metrics.csv` reproduces — those
  > are the interaction added on top of the **spline**, and the artifact ships `log_own_inter`,
  > the interaction on top of **`log(own)`**. The other four columns match the artifact
  > exactly on all eight heads, so this was one column typed from a variant that was later
  > dropped, in a table that was otherwise refreshed — the same shape as the season-total R²
  > failure. **The conclusion is unchanged and slightly stronger**: read against its own base
  > the interaction is worth ≤ +0.0031, and it is *negative* on two heads either way.

  A **log link wants a multiplicative predictor**: `log E[rate] = β·log(prior rate)` makes
  the model `rate ∝ prior_rate^β`, which is the right shape. Linear-in-raw-rate inside
  `exp()` is badly misspecified, catastrophically so for the zero-heavy skewed heads
  (`fg3a` 0.520, `blk` 0.638). `log1p(own)` recovers nearly all of it in **one term**.
  - **Splines add a real but small further gain, concentrated where the prior is most
    skewed** — `blk` **+0.054** over `log(own)` (and the retired `fg3a` +0.030), against
    ≤ +0.003 on five of the other six. Spend flexibility there only.
  - **The `age × own` and `mpg × own` interactions are a null once the scale is right** —
    ≤ +0.005 over `log(own)`, and *negative* for `reb` and `ast`. Component-specific aging is real
    (`make aging`) but does not survive as an interaction here.
  - The floor being this strong is the sharpest available statement of "attempts persist"
    (`fg3a` 0.908 in `persistence.csv`), and it means the rate side is close to saturated
    from prior-season information alone — consistent with `oracle_gp` 214.4 beating
    `oracle_rate` 261.9 on the season total.
  - **⚠️ These figures replaced an earlier set that was an artifact.** At `alpha=1.0` the
    linear baseline read R² 0.662 on `reb` and splines "improved" it by +0.15, with
    interactions worth −0.10 NLL. All of that was over-regularization (see the sklearn trap
    under "Data quirks"), and the log-scale test that would have caught it was run at the
    same broken alpha.
  - **Walk-forward PCA of the whole 156-column season matrix is worth ~nothing over three
    raw context columns.** Refitting scaler + PCA for every target season on S-1 and earlier
    only (never pooled — a pooled basis leaks the future invisibly), 10 components replacing
    `mpg`/`total_minutes`/`gp` while the head's own prior rate stays raw: `pca` lands within
    **±0.003** of `log_own` on every head (`reb` 0.9442 vs 0.9441, `fg2a` 0.9247 vs 0.9245,
    `ast` 0.9237 vs 0.9229), and `pca_spline` within ±0.003 of `log_own_spline` except
    `fg3a` +0.002 and `blk` +0.003. The 121 style/tracking columns add nothing once you
    have the player's own prior rate and his minutes — the same "rate side is saturated"
    conclusion from a second direction. PCA interactions (`log(own) × pc1/pc2`) are also a
    null. Use the cheap raw spec.
  - **`ftm|fta` is a head where *nothing* beats the floor** — best fitted NLL 3.0569
    against the floor's 3.0541. Free-throw percentage is pure player skill with no context
    to add, so an empirical-Bayes shrink of the prior is already optimal. `fg2m|fg2a` gains
    +0.079 NLL and `fg3m|fg3a` +0.028 (both from the PCA/spline variants), so the
    conversion side is worth ~1–2% of its NLL at most. (Recorded on the held-out split:
    3.1021 against 3.0822, +0.072 and +0.032.)
    - **⚠️ On validation `fg3a|fga` joins it under sklearn, and the Stan head clears the same
      floor by only +0.0034.** The shot-mix head's best Poisson/logistic variant reads 4.6317
      against a floor of **4.6191** — it *fails* — while `stan_components` clears that
      identical floor at **4.6157**. Both instruments now agree the head is *marginal*, where
      the recorded reading had Stan clearing by +0.0391 on test and made the sklearn failure
      look like an artifact of the coarser probe. The instrument difference is still real —
      sklearn's `spline_own` basis is on raw `own` while `stan_components` splines on
      `logit(own)`, which strictly nests the right scale — but it is worth 0.0160 NLL, not
      the 0.04 the test column implied. **Do not quote this head as clearing comfortably.**
  - **The conversion floor has to be a *shrunk* carry-forward, and that is a fact about
    proportions.** A player who went 0-for-3 from three has a prior 3P% of exactly 0.000;
    carrying it onto 200 attempts gives a beta-binomial NLL of **1.3e9** and makes the
    benchmark meaningless. The floor is therefore
    `p = (made + k·league_mean)/(attempts + k)` with `k` and `league_mean` fitted on train
    only — one shrinkage constant, no features. This is `CLAUDE.md`'s own "shrink conversion
    percentages hard" rule showing up as a benchmark requirement.
- **Fit the Stan heads SEPARATELY, not as one joint model — the posterior factorizes exactly.**
  The generative structure is a chain of conditionals: availability → `min | available` →
  counts `| min` → makes `| attempts`. With distinct parameter blocks and independent priors
  **the joint posterior factorizes into independent blocks**, so eleven separate fits recover
  the *identical* posterior — an identity of the same kind as the season-collapse identity, not
  an approximation. A joint fit only differs if you deliberately add shared coefficients or a
  correlated multivariate random effect.
  - **`megamodel.stan` is the proof, from this project's own history.** Every head there has
    its own `beta_*` with an independent prior and `nplayers`/`playerid` are commented out —
    no parameter is shared between any two heads. It was thirteen independent GLMs in one file,
    paying the full joint-fit price, which is why it ran on `sample_frac(0.01)`. 99% of the data
    was given up for a coupling that was not in the model.
  - **The correlation the simulator needs enters at draw time, not fit time.** Draw `min` once
    per player-game and push it through all eleven heads as exposure — minutes is **46.4%** of
    within-player residual variance (see the corrected variance budget above), overwhelmingly
    the largest common factor. Beyond that, residual cross-component correlation is **small**.
    ✅ **`make residual-correlation`** (`src/eda/residual_correlation.py` →
    `outputs/eda/residual_correlation.csv`, 246 rows, 1.2 s) writes the whole matrix in long
    form, because the copula needs the matrix rather than a summary of it. Measured on
    `serial_correlation.py`'s frame — 592,796 player-games, 9,052 player-seasons — the
    off-diagonals average **+0.0070** across the eleven heads and **+0.0225** across the seven
    counts, with a max of **+0.1329** (`fga`–`reb`). Minimum eigenvalue **+0.7853**, so it is
    PSD and usable as a copula with no nearest-PSD correction. Impose it at simulation time if
    step 2 misses; do not fit jointly.
    - **⚠️ Every row is emitted under two `fit_window` values and the simulator must consume
      `train_val`, not `full`.** Nothing here is fitted, so nothing was scored on a held-out
      split — which is exactly why the matrix could quietly be measured over the two seasons
      the heads hold out. It is a simulator *input*, not a finding the simulator reads about,
      so calibrating it on 2024-25/2025-26 would tune the simulator on the seasons it is
      later backtested against. **The figures quoted in this file and in `docs/` are the
      `full` window**, because that is the whole-sample description; anything the simulator
      is *given* takes `train_val`. The same axis applies to `serial_correlation.csv`,
      `bonus_calibration.csv` and the minutes head's game-level ρ.
    - **⭐ Adopting the shot-attempt basis shrank the copula's largest coupling by a third and
      improved its conditioning.** The two-count basis had to carry `fg3a`–`fg2a` at
      **−0.1248**; the shipped basis carries `fga`–`fg3a|fga` at **−0.0836**, and the minimum
      eigenvalue rises from +0.7559 to **+0.7853**. The substitution *identity* is gone by
      construction — one more three is exactly one fewer two — and what remains is a genuine
      residual relation between shot volume and shot mix, which is a different fact. The old
      pair still ships, under `basis == "legacy_two_count_basis"`, so the artifact records
      what the change bought rather than going quiet where the finding used to be.
    - The recorded summaries (+0.013, max 0.157, −0.110 over "101,588 games") were measured on
      **2021-22 onward** — 101,482 rows on the same frame, which is where the 0.157 maximum
      comes from. All three are confirmed in substance; the exact values were population-
      specific and the population was never stated.
    - **⚠️ The `raw` basis is 13× larger and is not a copula input.** Off-diagonals average
      **+0.090** unconditioned against +0.007 conditioned, and the largest raw cell is
      `fga`–`fta` at **+0.470** — two shot-volume counts both scaling with the minutes they
      were accumulated over, not a dependence. Building the copula on it would impose 13× the
      intended coupling *on top of* the shared minutes draw that produced it, so
      `minutes_conditioned` is an explicit column rather than a filename convention.
  - **Handle the 3PA/2PA substitution by reparameterizing into the chain**, not by coupling two
    Poissons: model `fga` as the count and `fg3a | fga` as a binomial *share*. That enforces the
    substitution by construction, keeps the factorization exact, and is better specified —
    `sco_pct_fga_3pt` persists at 0.886, i.e. shot-mix shares persist like counts.
  - Eleven small models are independently diagnosable and parallel; in one joint model
    divergences in the `blk` block degrade every other block's sampler. And at simulation time
    **never plug in `E[min]` — draw it**, since the bonus is a threshold.
- **✅ Built 2026-07-29 — `make stan`, cmdstanpy + CmdStan 2.39.0. Two `.stan` files serve
  every head**, which is the factorization argument as code rather than as prose:
  `src/stan/betabinomial_glm.stan` is availability *and* minutes *and* the three conversion
  heads (same likelihood, different `y`/`n`), and `src/stan/negbinomial_glm.stan` is the eight
  counts. Sources are checked in; cmdstanpy compiles a **copy** into `outputs/stan/` so no
  binary is committed and `src/stan/` stays free of generated `.hpp` files.
  - **The availability port reproduces the MLE, and that is a defined check rather than a
    hopeful comparison.** An L2 penalty of `l2` on standardized coefficients *is* a
    `normal(0, 1/sqrt(2·l2))` prior, so the posterior **mode** is exactly the penalized
    optimum `BetaBinomialGLM` finds — `stan_utils.prior_sd_for_l2` is that identity. Measured
    on 9,478 train / 883 **validation**: CRPS **10.0063** (Stan plug-in) / **10.0071**
    (posterior) against the MLE's **10.0057**, ρ **0.2808** vs **0.2806**, max coefficient gap
    **0.00335**, largest gap **0.085 posterior sd**, and the MLE inside the 95% credible
    interval for **21/21** terms. R̂ **1.0019**, min ESS 2,314, **0 divergences**, 195 s
    wall clock over 4 chains.
    - **⚠️ This block was measured on the held-out seasons until 2026-08-05 and read
      10.7947 / 10.7953 / 10.7952, ρ 0.2759 vs 0.2757, gap 0.0127 / 0.095 sd, R̂ 1.0025,
      min ESS 2,402, 254 s.** The port check moved to validation with every other head
      (`src/models/held_out.py`); the figures are not comparable, since both the fitting
      frame and the scored rows changed. **What the check asserts is unchanged and slightly
      sharper**: the coefficient gap fell 3.8× and the MLE is still inside the interval for
      every term, which is the whole content of "this is a port". The held-out reading is
      taken once, by `make final-evaluation`, through this module's own `fit_and_score`.
  - **Do not argue for the posterior on marginal CRPS — it is a wash by construction and the
    argument is the joint.** At ~10^4 rows against 20 parameters the posterior is sharp.
    What it buys is `Var_θ(Σ_i E[Y_i|θ])`: every player shares β, so one draw moves the whole
    board together, and that term is **exactly 0** under any point estimate.
  - **⚠️ But it is worth almost nothing on one roster, and the size of the portfolio is what
    decides.** The independent term grows as **sqrt(N)** and the shared-β term as **N**, so
    their ratio scales as sqrt(N). Measured on the 883-player **validation** board
    (`stan_availability.board_correlation`, random subsets, not extrapolated):

    | players | independent sd | shared-β sd | inflation |
    |---|---|---|---|
    | 12 | 69.9 | 4.2 | **+0.2%** |
    | 15 | 78.0 | 5.0 | +0.2% |
    | 30 | 110.3 | 8.8 | +0.3% |
    | 150 | 246.4 | 39.1 | +1.2% |
    | **883 (whole board)** | 597.8 | 222.8 | **+6.7%** |

    So it is real for **board-wide exposure across many lineups** and near-irrelevant for a
    single 15-man team. Quoting the 223-game figure as if it applied to one roster is the
    over-claim to avoid — it was made and corrected in the session that built this.
    - **The board is a SIMULATOR INPUT, so validation is where it belongs**, not merely
      where the lock put it. "How much does my whole board move together" is a number the
      simulator is *given*; calibrating it on the seasons the simulator is later backtested
      against is the leakage the split cannot catch, exactly as for `game_level_dispersion`
      and the residual copula. **⚠️ The recorded 911-player row (604.0 / 219.1 / +6.4%, and
      69.4 / 4.0 at 12 players) was the held-out board** and is superseded rather than
      wrong. The conclusion is untouched: a fraction of a percent on one roster, several
      percent across the board, and the ratio still scaling as sqrt(N).
  - **⚠️ The integrated predictive is NOT necessarily wider per player, and expecting it to be
    is a trap this session fell into.** By the law of total variance the mixture adds
    `Var_θ(E[Y|θ])` but replaces `Var(Y|θ̄)` with `E_θ[Var(Y|θ)]`, and
    `n·μ(1−μ)·[1+(n−1)ρ]` is **concave in μ**, so Jensen pushes the other way. Measured:
    **+0.046 against −0.082 games²**, i.e. the mixture is marginally *narrower*. Verified
    against the pmf to 1.3e-10. The marginal width is a red herring; the covariance is not.
  - **Two numerical facts, both of which cost a run.** (1) `1 - inv_logit(eta)` is exactly 0
    in double precision by `eta ≈ 37`, making a beta shape parameter 0 and rejecting the whole
    target; `s * inv_logit(-eta)` is algebraically identical and survives to `eta ≈ 745`. The
    `.stan` uses the latter and a test pins it. (2) **Always pass `inits`** — Stan's default
    uniform(−2, 2) on the unconstrained scale puts the starting linear predictor near
    `2·sqrt(K)`, which at K = 19 is already in the saturation region. Every head inits at the
    intercept-only solution with zero slopes.
  - **`n = max(team_games, gp)` matters more under HMC than under L-BFGS-B**, not less: a
    non-finite target poisons the trajectory rather than merely stopping an optimizer.
    `stan_availability.assert_binomial_support` re-checks it and a test pins it. The *other*
    recorded trap — L-BFGS-B stopping on finite-difference noise at a ~1e5 objective — simply
    does not exist here, since Stan differentiates exactly.
  - **B-spline bases are badly conditioned for HMC.** The spline variants sample at treedepth
    8 (255 leapfrog steps per iteration) with a step size of 0.011, against treedepth 3–4 for
    the linear ones — **721 s against 171 s** for the same data on the minutes head. Valid,
    just expensive; an orthogonalized (QR-whitened) basis is the fix if spline variants ever
    become the shipped spec. (The recorded **899** s / **189** s pair is the same contrast
    on the retired test refits, which ran at double the iterations.)
- **✅ The minutes head is built — `make stan-minutes`, and it is the FIRST head to use the
  real trials denominator.** `min` is successes out of **actual game length**, never 48:
  `data/features/game_length.parquet` supplies it, 5.93% of games go to overtime, and the
  prior attempt's `normal(μ,σ) T[0,48]` needed a `min == 48 → 47.9` fudge *and* discarded
  every overtime game. Season-collapsed on 9,804 player-seasons (8,306 fit / 742 validation
  on 2022-23 and 2023-24), `y` = season minutes, `n` = summed game length **over the games he
  played** — so it is the conditional `min | available` and composes with the availability
  head rather than double-counting absences. **0 rows clamped by rounding**, realized share
  max 0.9096.

  | variant | val CRPS | val R² | val MAE | val bias | selected |
  |---|---|---|---|---|---|
  | `carry_forward` (no-fit floor) | 161.45 | 0.8536 | 213.13 | **+23.91** | |
  | linear | 144.71 | 0.8826 | 199.54 | −3.26 | |
  | `logit(own)` | 145.45 | 0.8819 | 200.90 | −2.97 | |
  | `logit(own)` + quadratic | 144.54 | 0.8827 | 200.84 | −13.82 | |
  | **`logit(own)` + spline** | **143.93** | **0.8835** | 199.60 | −14.00 | **✓** |

  Clears the floor by **+0.0299 R² and −17.5 minutes of CRPS**. Max R̂ **1.0054**,
  **0 divergences** over 4 fits, 1,227 s total.
  - ⚠️ **This table was a TEST evaluation until 2026-08-06** and read, as
    `val CRPS / test CRPS / test R²`: floor 161.45 / **168.24** / **0.8166**, linear
    **144.62** / **147.18** / **0.8565**, `logit(own)` **145.44** / **147.35** / 0.8565,
    quadratic **144.83** / **147.21** / **0.8574**, spline **144.13** / **146.85** /
    **0.8572** — clearing the floor by **+0.0407** R² and **−21.4** minutes over 8 fits, max
    R̂ **1.0093**, **2,183** s, with held-out bias running from the floor's **−5.69** to
    **−33.1** (linear) and **−40.9** (spline). **Two things retired at once and only one of
    them is the split.** The `test_*` columns went because `sweep` no longer writes them;
    the *old* `val_crps` values went because selection was raised from 500/500 to
    full-length chains in the same change, so even the surviving column is a different
    measurement. Only the floor's **161.45** is arithmetic all the way down and reproduces
    to the digit.
  - **The selected variant reproduced, which is what makes this head's selection readable
    at all.** `logit_own_spline` is still chosen, and it now wins by 0.61 CRPS over the
    quadratic against 0.69 before — a margin that survived both a change of scored rows and
    a doubling of chain length. Contrast the season-term ablation, where **9 of 13 heads
    flipped their selected arm** on margins under 1%: an ablation whose winner moves under a
    change that should not touch it is measuring noise, and this one does not move.
  - **The specification answer is the OPPOSITE of the count heads', and that is the finding.**
    There, scale is everything and curvature is nearly nothing. Here the logit scale is a
    **dead wash** and slightly *worse* on both metrics (val R² 0.8819 against linear's
    0.8826, val CRPS 145.45 against 144.71), while curvature is what pays. Do not generalize
    "put it on the link's scale" from the counts to the minutes head.
  - **The season-level ρ is NOT the number the simulator needs, and they differ by more than
    the fit does.** Fitted season-level ρ = **0.05025**; game-level ρ measured separately
    against each player-season's own mean over 713,947 player-games = **0.0776**, i.e.
    **4.65× binomial** at a 48-minute game. A season total cannot separate a per-game random
    effect from a per-season one — iid game noise is diluted by ~1/G while a shared season
    multiplier passes through in full. Drawing per-game minutes from the season-level ρ would
    make every simulated game far too close to the player's average.
    `stan_minutes.game_level_dispersion` reports it; it is in-sample and therefore a *floor*.
    - **The simulator needs THREE minutes numbers and they compose, they do not substitute.**
      (1) the season-level mean from this head, (2) the **game-level** dispersion 4.65×
      binomial for the marginal spread of a single game, and (3) the **2.43× ten-game block
      variance inflation** from `make serial-correlation` for the serial dependence *between*
      games. Using only (2) gives independent draws with the right marginal and too little
      variance in any aggregate; using only (3) gets the clustering right and each game wrong.
  - **⚠️ "The fitted heads carry a −33 to −41 minute bias against the floor's −5.7" is
    WITHDRAWN — that was the held-out column, and on validation the FLOOR is the biased
    one.** The floor over-predicts by **+23.91** minutes while the fitted arms run **−3.26**
    (linear) to **−14.00** (spline). So the recorded explanation — "the floor is unbiased
    because it does not shrink" — is false on the split that selects.
    - **What reproduces is the GAP, not the level.** The fitted arms sit **−27.2** (linear)
      to **−37.9** (spline) minutes *below* the floor on validation, against −27.1 to −35.3
      on the retired held-out column. Shrinkage moves every arm roughly the same distance
      down from the carry-forward; where that lands in absolute terms is a property of the
      seasons being scored. So the calibration defect is **relative**, it still compounds
      through the eleven component heads that take these minutes as exposure, and it is
      still worth correcting before the simulator consumes it — but do not quote a signed
      level for it, and do not read "the fitted heads under-predict minutes" as a standing
      fact.
    - The two columns are not a controlled contrast in any case: different scored rows,
      different training data and different chain lengths, which is exactly the confound
      `src/models/held_out.py` was written to stop being read as a replication.
    - This also **removes the era-effect candidate** the bullet used to carry. A bias whose
      sign flips between two adjacent season pairs is not the signature of a monotone
      role-graded era trend; `make season-terms` had already falsified that from the other
      direction, finding a trend makes the bias *worse* by 15.5 minutes.
- **✅ The eleven component heads are built in Stan — `make stan-components`, 37 fits, 0
  divergences, max R̂ 1.0076, 137.4 min. Two results overturn what `make component-rates`
  measured with sklearn, one settles a standing recommendation, and one — the free-throw
  family — REVERSED on the move to validation.** 10,194 player-seasons, 8,630 fit / 773
  validation on 2022-23 and 2023-24. Validation R² on the season total:

  | head | no-fit floor | linear | `log(own)` | `log(own)` + spline | selected |
  |---|---|---|---|---|---|
  | `fga` | 0.9514 | 0.9489 | 0.9581 | **0.9584** | spline |
  | `reb` | 0.9505 | 0.8889 | **0.9513** | 0.9511 | `log_own` |
  | `ast` | 0.9195 | 0.6418 | 0.9198 | **0.9255** | spline |
  | `tov` | 0.8966 | 0.9048 | **0.9170** | 0.9169 | `log_own` |
  | `blk` | 0.8103 | **−0.2744** | **0.6730** | **0.8309** | spline |
  | `fta` | 0.8765 | 0.5741 | **0.8909** | 0.8893 | `log_own` |
  | `stl` | 0.8386 | 0.8492 | 0.8674 | **0.8695** | spline |

  > ⚠️ **This table was a TEST evaluation until 2026-08-06** and read, in the same layout:
  > `fga` 0.9464 / 0.9396 / 0.9501 / 0.9505, `reb` 0.9424 / 0.9095 / 0.9439 / 0.9428,
  > `ast` 0.9197 / 0.6615 / 0.9223 / 0.9240, `tov` 0.8845 / 0.8823 / 0.8929 / 0.8926,
  > `blk` 0.8407 / −1.393 / 0.6794 / 0.8579, `fta` 0.8673 / 0.8171 / 0.8649 / 0.8648,
  > `stl` 0.8194 / 0.8113 / 0.8390 / 0.8413, on 9,403 train / 791 test. **Every conclusion
  > survives except the free-throw one, which reverses** — see the `fta` bullet. The fit
  > count halves from **74** because the test side is no longer fitted at all, and the cost
  > falls from **305.0** min to 137.4 *despite* every fit now running full-length chains:
  > the old test side was the majority of the compute at double the iterations.

  - **⚠️ `selected` and `beats_floor` used to read DIFFERENT SPLITS. They no longer do, and
    that is the semantic change to notice in the artifact.** The variant was chosen on
    validation and then certified against its no-fit floor on *test*, so a head could be
    selected on one split and marked a failure on another — which is exactly what happened
    to `fta`. `_finalize` now computes both from `val_r2` / `val_nll`, making the floor
    comparison the honest one: does the variant I would ship beat arithmetic on the rows I
    chose it with? Certification against the held-out seasons is `src/final_evaluation.py`'s
    job and nothing else's.
  - **⭐ `fta` CLEARS its floor on validation, so "the whole free-throw family fails" is
    WITHDRAWN.** `log_own` scores **0.8909** against a floor of **0.8765** — **+0.0144**,
    and it is the selected variant — where on test it read 0.8649 against 0.8673 and failed
    by **0.0024**. Two parts in a thousand was never evidence of anything, and it was being
    read as a finding about free throws in general. **`ftm|fta` still fails, and is now the
    only head in the project that fails its floor at every variant.** That was always the
    better-founded half of the claim: free-throw *percentage* has a pure-player-skill
    argument that trips to the line never had, which is precisely why `fta` joining it
    deserved more suspicion than it got.
  - **⭐ The shot-attempt basis removed this project's worst misspecification case.** The
    retired `fg3a` row read floor 0.9036, **linear −19.00**, `log_own` **0.3719**, spline
    0.9046 — a head that *failed its floor at two of three variants*. `fga` replaces it and
    is the best-behaved count head in the project: floor **0.9514** (the highest), selected
    **0.9584**, and **linear costs it only 0.0025 R²** (0.9489) against `fg3a`'s −19.00. A
    total is far less skewed than its three-point part, so the scale barely matters. `blk`
    at **−0.2744** is now the worst linear case, and the only one left that is negative at
    all. The retired `fg2a` row (0.9194 / 0.9018 / 0.9241 / 0.9241) is unremarkable
    either way.
  - **⚠️ `log(own)` alone is NOT sufficient under a negative binomial, and that contradicts
    the Poisson result.** `make component-rates` has `log_own` at 0.7748 (`blk`) — both
    figures now on validation, so they are directly comparable — and under NB the same spec
    collapses to **0.6730**, far below the 0.8103 floor, with only the spline recovering it
    (**0.8309**). The mechanism is the likelihood, not the
    data: NB2's `var = μ + μ²/φ` down-weights large counts relative to Poisson, so the fit is
    driven by the low-count mass — exactly where the log-scale relation is most curved. **The
    "splines are worth ≤ +0.003 outside `fg3a`/`blk`" guidance is Poisson-specific.** Under NB
    the validation split picks the spline for **four** heads (`fga`, `ast`, `blk`, `stl`),
    and for `blk` it is not a refinement but the difference between a model and a
    failure. (The retired `fg3a` made the same point harder still, at 0.8791 under Poisson
    against **0.3719** under NB.)
  - **`linear` is catastrophic, far beyond what the Poisson fit showed** — `blk` **−0.2744**
    validation R² against 0.638 under sklearn, and the retired `fg3a` read **−19.00**.
    Linear-in-raw-rate
    inside `exp()` is not merely misspecified, it is unusable. The strongest available
    statement of "the specification is scale, not curvature." **It fails the no-fit floor on
    5 of the 7 count heads** — every one but `stl` and `tov` — and `fta` is the second-worst
    at **0.5741**, a head where every other variant is fine.
  - Conversion heads, validation beta-binomial NLL per row (lower better), all **four**
    selecting `logit(own)` + spline: the new shot-mix head `fg3a|fga` **4.6157** vs floor
    **4.6191** (**+0.0034**), `fg2m|fg2a` **3.8046** vs floor 3.8442 (**+0.0396**),
    `fg3m|fg3a` **3.1677** vs 3.1851 (+0.0173), `ftm|fta` 3.0804 vs **3.0541** (−0.0263,
    fails as predicted). Same ordering as the sklearn run and as the retired test reading
    (4.6137 / 4.6528, 3.7249 / 3.7770, 3.2407 / 3.2614, 3.1313 / 3.0822 — gains of +0.0391,
    +0.0521, +0.0208 and −0.0491).
    **The shot-mix head still clears its floor, but by +0.0034 rather than the +0.0391 the
    test column showed** — an order of magnitude narrower, and the one place where reading
    the old column overstated a head-level result. It is still the head-level confirmation
    of Gate 0, and `logit_own` alone still does *not* clear the floor, so the spline is
    load-bearing there exactly as it is for `blk`.
- **✅ The 3PA/2PA reparameterization survives an un-handicapped re-measurement — Gate 0,
  `make stan-substitution`, 2026-08-03. Full design and adoption spec:
  `docs/shot-attempt-basis-plan.md`.** Modelling `fga` as the count and `fg3a | fga` as a
  binomial *share* beats two independent count heads by **−0.501041 nats** on validation,
  per player-season, on the joint density of `(fg2a, fg3a)`: 10.505159 → 10.004118. Arm A is
  fitted at each head's own selected variant and arm B is swept, so neither side is
  handicapped. 8 fits, max R̂ **1.0047**, **0** divergences, 26.8 min. The comparison is
  legitimate because `(fg2a, fg3a) ↔ (fga, fg3a)` is a **bijection with unit Jacobian on the
  integers** — the same point in different coordinates — so the two joint log-densities are
  directly comparable.
  - **⚠️ Validation-only since the 2026-08-06 re-run, and two things retired with the test
    column.** Adoption was decided on a test margin of **−0.493549**, checked against arm A's
    *best-of-16* configuration (**10.476413**) which arm B beat by −0.491910. Neither can be
    recomputed: `_arm_a_grid` reads the grid out of `stan_component_metrics.csv`'s `test_nll`,
    and adoption removed `fg2a`/`fg3a` from that file while the split move removed the
    column. **The loss costs the argument 0.001640 nats** — that is the entire distance
    between the grid minimum and arm A's *own selected* pair, which the gate still fits, so
    "wins against arm A's best-of-16" and "wins against arm A as it would be fitted" were
    never different claims here. Full retired block in `docs/shot-attempt-basis-plan.md`.
  - **⚠️ The recorded −0.771 / −0.793 was measured against a straw man.** `substitution_arm`
    fits *every* head at `log_own`, but `fg3a` selects `log_own_spline`, and at `log_own` it
    reads R² 0.3719 with `beats_floor = False` against 0.9046 for the spline it ships. Arm B
    was also pinned at one variant rather than swept. They are a different, weaker
    experiment, not a stale value.
    - **⭐ The decomposition is now entirely on validation, where it used to need the test
      column.** Handicapped arm A scores **10.797078**; fitting `fg3a` at the spline it
      selects brings it to **10.505159**, so **the handicap is 0.291919 nats**, and sweeping
      arm B is worth a further **−0.021832**. So −0.771128 → **−0.501041** exactly, and
      **65%** of the recorded margin survives its own correction. (The retired test reading
      of the same three: 0.305646, −0.487010, −0.006539.)
    - **The regression check is now floating-point identity rather than a comparison to a
      recorded figure.** The gate's arm B pinned at `(log_own, logit_own)` and
      `substitution_arm`'s arm B are the same quantity reached by different code in
      different modules, and they agree at **10.025950** to **1.78e-15**. It replaces the
      retired check against 9.991042, which compared this gate's *test* side to a July
      number; both sides of the new one are current and on the same rows.
    - **⚠️ `stan_component_substitution.csv` WAS rebuilt on 2026-08-06, validation only**, so
      the sentence that stood here — that both recorded figures remain true of it, which was
      not rebuilt — is now half retired. The **validation** margin reproduces: 10.797 →
      10.026 at **−0.771**, unchanged to the precision quoted even though every fit went from
      half-length to full-length chains. That is a free replication of the arm, and the
      strongest reason to believe the half-iteration selection side was never the weak link.
      The **test** row is gone with the split.
      - **⚠️ And the last copy of the test figures went with the Gate 0 re-run on
        2026-08-06.** They had survived inside `stan_component_substitution_sweep.csv` — arm
        A's `log_own+log_own` grid cell at **10.783699** and arm B's `log_own` + `logit_own`
        heads summing to **9.991042**, differing by the recorded −0.792657 — and
        `docs_audit.subst_test` read them from there specifically so a superseded measurement
        stayed *value-checked* rather than merely quoted. `make stan-substitution` is now
        validation-only too, so that file no longer has a test half either. **−0.793 and
        0.792657 are presence-checked records now**, and the distinction that paragraph drew
        is worth keeping precisely because this is the case where it ran out: preserving a
        figure in an artifact only works until the artifact is legitimately rebuilt, and then
        the honest move is to demote the claim rather than freeze the run.
  - **⭐ The sharpest result is that the coordinate change beats the fitting, and it is
    SHARPER on validation than on the test column it replaces.** At their **no-fit floors** —
    prior per-36 rate × minutes for the count, shrunk carry-forward for the share, no
    features anywhere — the two bases score **11.174424** against **10.064318**, a
    **−1.110105** gap. So arm B's floor beats arm A's *fitted* configuration by
    **−0.440841** — 88% of the total margin — and arm B's own fitted heads add only
    **−0.060200** on top of their floor. This is not a better model of shot attempts; it is
    the same information in coordinates where the dependence is structural instead of
    residual. Same shape as "the rate side is nearly saturated by a carry-forward": when the
    floor is that strong, the **parameterization** is where the leverage is. (The retired
    test reading was 11.024027 / 10.085599 / −0.938427, with −0.390814 against the
    best-of-16 and −0.101096 of own fitting. **The floors themselves are arithmetic and
    reproduce to the digit** — `stan_component_metrics.csv` independently agrees on both
    arm-B rows — so the widening is the *fitted* side moving, not the benchmark.)
  - **The share head needs its spline to clear its floor, and this is the head where reading
    the test column would have shipped a failure.** `logit_own` reads 4.636018 on validation
    against a floor of 4.619109 — *below* it — while `logit_own_spline` clears at 4.615622.
    On test `logit_own` **did** clear (4.611402 against a 4.652797 floor), so a test-only
    reading would have selected a variant that fails its floor on the split that selects.
    `linear` fails on both. For `fga` the spline is near-irrelevant by contrast (5.388496 vs
    5.389932 on validation). The result is independently reproduced by `make stan-components`,
    whose own `fg3a|fga` rows are fitted through a different code path and land within 1e-4.
  - **⚠️ ADOPTED 2026-08-03 — this bullet used to say "still an ablation, not the shipped
    spec" and that has been false since the day it was written.** `component_rates.COUNT_HEADS`
    is `fga, fta, reb, ast, stl, blk, tov` and `fg2a`/`fg3a` are no longer count heads; read
    the head list rather than any prose. The head count stayed **11** (`fg2a` became
    *derived*, like `pts`), and `selected_specs`'s fallback to `log_own` — the silent-degrade
    site this bullet warned about — is now `stan_components.LEGACY_ARM_A_SPECS`, pinned
    constants recording what the pre-adoption artifact selected, so Gate 0 stays re-runnable
    instead of quietly re-acquiring the handicap it exists to remove. Verified 2026-08-06.
  - **`conversion_variants` takes `own=` explicitly, and the hazard it was written against
    turned out not to be real.** The worry was that the `{made}_pct_lag1` convention would
    resolve to `fg3a_pct_lag1` and silently fit the head on three-point *shooting* accuracy
    instead of the attempt *mix* (shares persist at 0.886, conversion percentages at 0.500).
    Adoption falsified it: `season_totals` builds `fg3a_pct = fg3a / fga`, which **is** the
    mix, while shooting percentage stays `fg3m_pct`. Keep `own=` for explicitness. The
    refactored arm reproduced the recorded **9.991042** to nine decimals, which is how we
    know the fix changed nothing else.
- **Sampler cost is concentrated entirely in the spline variants.** 6 of 37 fits saturated
  treedepth, *all* of them spline arms; the slowest fit is 19.6 min (`fg3m|fg3a` spline)
  against 1.0–2.2 min for a linear or `log_own` count head. **37/37 cleared every convergence
  bar** — max R̂ **1.0076**, 0 divergences.
  - **Raising the selection side to full-length chains removed the one fit that used to fail
    a bar, and that is the clearest evidence the old short/long split cost something real.**
    `fg2m|fg2a/logit_own_spline/val` read R̂ **1.0118** against a 1.01 bar with ESS 450; at
    full length it reads R̂ 1.0040 with ESS 987. The recorded defence of it — that the
    variant it chose won by 0.017 NLL, far outside the sampling noise — was the right call
    on the evidence available, but it was an argument for *tolerating* a diagnostic failure
    in a fit that decides something. There is now none to tolerate.
- **`time.perf_counter()` does NOT advance while macOS is asleep, so the timings survive a
  suspended run.** Worth recording because the opposite was assumed during this build: the
  July components run spanned a ~7 h machine sleep (10 h 11 m elapsed) and reported 208.6 min
  of compute with a maximum single fit of 19.8 min — no inflated row anywhere.
  `stan_utils.diagnostics` needs no sleep-correction. (That 208.6 is a record of *that* run;
  the shot-attempt refit totalled 305.0 min, and the current validation-only heads 137.4.)
  - **It reproduced on the 2026-08-06 refit, and this time the arithmetic closes.** That run
    was awake 07:06–08:40 and 16:48–17:12 with a **7.8 h** sleep in between: 606 min elapsed,
    of which ~469 were asleep, leaving **136.9 min** of awake wall clock against **137.4 min**
    of reported compute. The sleep is visible in the log as a 488.6-minute gap between two
    fit starts, of which only 19.6 min is the fit — so *elapsed* time is the misleading
    number and the artifact's is the true one, which is the whole point.
- **✅ The team-game minutes COMPOSITION SHIPS — Gate E taken at the full window
  2026-08-04, and it is now part of `make stan`.** It beats the independent draw on the
  independent draw's own metric. `make stan-composition`,
  `docs/minutes-composition-plan.md`. Each team-game's `5 × game_length` minutes are
  allocated among the K players who played by decomposing the multinomial into
  **sequential binomial trials**, ordered by prior-season minutes share, with the
  per-player cap enforced through the **trials** (`m_k = min(U, R_k)` — remaining
  capacity) rather than checked afterwards. Fitted on all 30 seasons (train 631,158 rows /
  61,252 team-games) and scored on **validation** (2022-23/23-24, 52,295 player-rows /
  4,920 team-games) since the 2026-08-08 refit:

  | variant | val CRPS | val PIT KS |
  |---|---|---|
  | `carry_forward` (floor) | 4.6776 | 0.0496 |
  | `binomial` | **4.9388** — *fails the floor* | **0.1919** |
  | `betabinom` | 4.5417 | 0.0496 |
  | `betabinom_ot` | 4.5431 | 0.0494 |
  | **`betabinom_ot_graded`** (selected) | **4.4945** | 0.0428 |
  | `independent_comparator` | 4.7842 | 0.0483 |

  ⚠️ **This table was a TEST evaluation until 2026-08-08**, when the artifact was
  regenerated to match code that had been validation-only since 2026-08-05. It read, as
  `val CRPS / test CRPS / test PIT KS`: floor 4.6776 / 4.8576 / 0.0354, `binomial` 4.9394 /
  **4.9732** / **0.1948**, `betabinom` 4.5422 / 4.5893 / 0.0405, `betabinom_ot` 4.5430 /
  4.5848 / 0.0414, selected **4.4926** / **4.5592** / 0.0393, comparator 4.7842 / **4.9140**
  / 0.0769. **Nothing reversed** — same selected arm, same ordering, same gate outcomes.
  Full retired block in `docs/minutes-composition-plan.md`.

  - **⭐ The selection replicated across a doubling of chain length, and that is the reason
    to believe it.** The validation side used to fit at `select_warmup`/`select_samples` and
    now runs full-length, so every fitted row is a fresh measurement — and **no arm moved by
    more than 0.0019 CRPS**. `carry_forward` and `independent_comparator` reproduced to six
    decimals, the first because it is arithmetic and the second because it never trains on
    the composition window. Contrast the season-term ablation, where 9 of 13 heads flipped
    their selected arm under a change that should not have touched them.
  - **`binomial`'s calibration failure now has a validation twin**, which it did not before:
    the old schema wrote `test_pit_ks` and no `val_pit_ks`, so the sharpest statement of the
    arm's failure was a held-out number. It reads **0.1919** on validation against the
    floor's 0.0496, reproducing the retired 0.1948 almost exactly.
  - **⚠️ Gate A under-predicts, but the multiplier is NOT a constant — do not treat 1.63× as
    one.** The one-pass sweep extrapolated **8.3 h** and took **9.78 h**, a **1.17×** miss,
    against the two-pass run's 12.8 h → **20.9 h** and **1.63×**. Per-row cost is
    **superlinear in rows** — 5.95 ms/row on the 26k-row probe against **14.83 ms/row** on
    the 631k-row `betabinom` fit — because more data sharpens the posterior, shrinks the
    step size and buys more leapfrog steps per iteration on top of an already-linear
    per-gradient cost. The miss shrank because the retired figure was inflated by a *second*
    pass the extrapolation modelled badly, not because the sampler got more predictable.
    **Treat Gate A as a lower bound**, and note that `stan_games_played.probe_timing`
    hard-codes `raw_hours * 1.63` from the retired measurement — which is now conservative
    rather than calibrated, and is deliberately left that way.
  - **Raising selection to full-length chains removed this head's only diagnostic failure.**
    Max R̂ is **1.00935** with **0** divergences over 6 fits and every fit converged. The
    fit that failed the 1.01 bar in the two-pass run was `binomial/val` at R̂ 1.0113 with
    ESS 421 — a *selection* fit running at half length; at full length it reads R̂ 1.00649
    with ESS 917. Same pattern as `fg2m|fg2a` in `stan_components`.

  - **This resolves the fork `docs/predictions-plan.md` left open.** That doc's warning
    box said a Dirichlet-multinomial gets the team total exactly but cannot bound any
    individual at `game_length`, and "**neither form gets both**". The sequential
    decomposition **does**: trials-as-remaining-capacity gives the cap, the deterministic
    last step gives the total. Both are asserted on every simulated draw.
  - **−0.2898 minutes of CRPS against the incumbent** (4.4945 vs 4.7842, **−6.06%**) — the
    plan predicted a wash and budgeted for arguing on capability instead. It won outright.
    And the capability gap is there too: the independent draw misses the team total by
    **33.89 minutes per team-game** where the composition is exact, and carries a **−0.5521**
    minute bias against the composition's exact zero. (The pilot read −0.406 and the retired
    test column −0.3548, i.e. **−7.2%**, at 36.87 minutes and a −1.2856 bias; the win shrinks
    as the frame changes and is nowhere near a wash on any of them.)
  - **The pure decomposition is worse than the no-fit floor**, and this is the sharpest
    result: the `binomial` arm reads 4.9388 against 4.6776 with PIT KS 0.1919 against
    0.0496 — far too tight, exactly as the measured game-level ρ (4.65× binomial)
    predicted. The dispersion is not a refinement, it is the difference between a model
    and a failure. Same shape as the NB-vs-Poisson finding on the count heads.
  - **The offset IS the floor**, so both share one code path: `logit(w_k / Σ_{j≥k} w_j ×
    R_k / m_k)` on renormalized prior shares means `β = 0` is prior-shares-carried-forward.
    That is **already a redistribution model** — a missing teammate shrinks the
    renormalizer and scales everyone else up — so `β` fits *deviations* from proportional
    redistribution, which is the "who absorbs the minutes" question as a fitted quantity.
  - **✅ ρ is graded by prior-share quartile, and role grading is real** — fitted
    **0.1768 / 0.1300 / 0.1115 / 0.0855** from fringe to star, a **2.07×** spread against
    a single shared **0.1211**. A 34-mpg starter's allocation step is genuinely steadier
    than a reserve's. `betabinom_ot_graded` differs from its twin in the **dispersion
    alone** — same features, same mean function — so the contrast is clean.
    **⚠️ Every ρ is larger at full window and the spread is narrower** (the pilot read
    0.1480 / 0.1125 / 0.0874 / 0.0613 against a shared 0.0970, a 2.41× spread): 26 seasons
    of rotation practice raise the dispersion everywhere and compress the fringe-to-star
    ratio. The grading is still real and still monotone; it is less extreme. (The two-pass
    run read 0.1751 / 0.1285 / 0.1099 / 0.0839 against 0.1195 at 2.09× — every value rose by
    ~0.0016 at full chain length and the ordering is untouched.)
    - **The calibration fix is the point, not the CRPS.** Realized/simulated variance
      ratio by tier goes **1.2224 / 0.8430 / 0.6589 / 0.6285** shared →
      **1.0465 / 0.8132 / 0.7071 / 0.8136** graded: mean |ratio − 1| falls **0.2730 →
      0.1782**, a **35%** cut, with the extremes improving most.
      **⚠️ The pilot's 0.1055 and its near-exact 0.988 star tier oversold this** — at full
      window the star tier lands at 0.814 and the cut is 35% rather than 59%. And **three of
      four tiers sit below 1**, so the head is mildly over-dispersed in aggregate, which
      is a better-posed target than chasing q2. (Retired test reading: 0.2928 → 0.1796, a
      39% cut, with tiers 1.3466 / 0.8089 / 0.7691 / 0.5973 → 1.0845 / 0.7687 / 0.8217 /
      0.7757.)
    - **⚠️ q2 still gets *worse* (0.843 → 0.813), and it is structural.** The fitted ρ is the
      dispersion of a **sequential step**; the ratio is measured on a player's
      **marginal** minutes. Because the order is prior-share *descending*, a low-share
      player breaks his stick last and inherits the accumulated remainder variation from
      everyone ahead of him — so grading step dispersion does not map one-to-one onto
      marginal variance by tier, and a tier can be pushed off a mark it happened to hit.
    - `n_rho = 1` is the shared model **exactly**, so one code path serves both and the
      graded arm strictly generalizes. A test pins the identity at the simulator level.
      **Bin edges come from train quantiles only** — leakage here would be especially
      quiet, since ρ never touches the mean.
  - **The OT interaction is real but tiny, and it LOSES validation** to plain `betabinom`
    by 0.0014 CRPS. It survives only because the graded arm is built on it. Starters take
    **0.6013** of team minutes in regulation and **0.6423** in OT; the head simulates
    **0.5912** → **0.6415**, reproducing the +4.1 pp shift as +5.0 pp with a −1.0 pp level
    *undershoot*. (The retired test reading had the level going the other way — observed
    0.5882 → 0.6314 against a simulated 0.5998 → 0.6464, a +1.2 pp overshoot — so the sign
    of the level error is a property of the scored seasons, not a standing bias.)
  - **Game length itself is a two-parameter geometric tail**: p_any = 0.0608, p_more =
    0.1408, which covers 3OT/4OT for free. It predicts 128.4 single-OT games
    against 120 observed — the form holds, but it overpredicts OT by ~7% on recent
    seasons, which belongs in the season-effects ledger.
  - **Rookies cannot be filtered here, unlike every other head** — the sum must be
    complete — so no-prior players (15.2% of rows, 12.3% of minutes) get an
    expanding-window draft-bucket share prior and sit last in the order.
- **Four numerical traps cost this head an hour of dead warmup, and three are new to this
  repo.** ✅ `make stan-composition`; the probe went from **60+ min without one draw** to
  **65 s**. Each is a case where the algebraically identical form is not the
  computationally identical form — the same lesson as `s * inv_logit(-eta)`, four more times.
  - **Never compute a tail as `1 − head`.** A tiny tail rounds the head mass to exactly
    1.0, `log1m(1)` is `-inf`, and **every chain died at initialization** — while the true
    target contribution (pmf over tail, both tiny) is O(1) and perfectly finite. Sum the
    **tail upward in log space**. A test pins it on a case where the head form returns −inf.
  - **Never call `beta_binomial_lcdf`/`lccdf` inside the gradient.** Stan routes it through
    the generalized hypergeometric (`grad_F32`); ~400 truncation rows made one gradient
    cost seconds. `optimize(iter=30)` took **>600 s** with it and **0.8 s** with an
    explicit pmf-ratio recurrence.
  - **Use `metric="dense_e"`.** Residual linear correlation held NUTS at treedepth 8–9
    under the default diagonal metric; dense drops it to 4 and the probe fit from
    **645 s to 65 s**. At ~25 parameters the dense adaptation is free. `stan_utils.sample`
    now takes `metric`; every other head keeps the default.
  - **Per-column imputation flags are a degenerate subspace when missingness is
    block-structured.** A rookie loses every design column at once, so `impute()`'s 18
    flags were exact copies — C(18,2) = 153 duplicate standardized columns. One
    `design_missing` indicator carries the same information. The other heads never hit
    this because they *filter* the no-prior rows.
  - A fifth, model-level rather than numerical: **the carry-forward offset can demand more
    than the cap** (~1% of rows — a star whose played-set prior shares sum well below 5),
    so it saturates at 0.93 with an `offset_clipped` indicator rather than at `1 − 1e-3`,
    which asserted "47.95 of 48 minutes" with a near-zero beta shape parameter.
- **Availability is strongly autocorrelated, and clustering is 42% of the overdispersion —
  not the sixth this file recorded until 2026-08-05.** `serial_structure` on 942,597
  transitions (appearance window): `P(play|played) = 0.905`, `P(play|missed) = 0.308`, so
  lag-1 ρ = **0.597** and a 2-state Markov chain inflates variance `(1+ρ)/(1-ρ)` = **3.96×**
  — against the **22.7×** measured. And the simple chain is falsified specifically: a
  constant hazard implies **geometric** spells, which matches the mean (3.25) and misses both
  tails — observed 0.483 of spells are 1 game against 0.308 predicted, and 0.0635 are 10+
  against 0.0365. Absences are a **mixture**.
  - **⚠️ "Clustering is ~a sixth of it" is withdrawn, and it was wrong twice over.** ✅
    `make availability-profile` (`key == "transition_rotation"` / `"markov_rotation"`).
    **(1) The composition is additive, not multiplicative**: with a player-season frailty on
    the mean and Markov clustering within, `inflation = C + ρ·(n − C)`, returning
    `1 + (n−1)ρ` at `C = 1` as it must — so dividing 22.7 by 3.96 is not a decomposition of
    anything. **(2) The two figures are measured on different frames**: the 3.96× runs on the
    appearance window over *all* players, the 22.7× on the full window over *established
    rotation* players. Matched, the same population reads `P(play|played) = **0.9443**`,
    `P(play|missed) = **0.1333**`, giving **C = 9.58**.
  - **So there is no dispersion hole to fill; there is a surplus to avoid.** The residual
    frailty a chain must not double-count is `(22.70 − 9.581)/(82 − 9.581) = **0.181**`
    against the incumbent's fitted **0.2757**, and stacking that ρ on the measured clustering
    predicts **29.55** against 22.70 — a **30.2%** overshoot. This is why the games-played
    spell process was decided *in writing, before its sweep ran* to ship on tail calibration
    with GP CRPS as a non-regression bar. The unsuffixed profile rows are untouched and still
    correct measurements of what they measure; only the ratio taken between them was wrong.
  - Build the process for the season-total joint distribution and the preseason initial
    state, not for GP CRPS. **See `docs/games-played-plan.md` and the entry below.**
- **✅ The games-played spell process is built — `make games-played` + `make
  stan-games-played`, 2026-08-05. NO ARM SHIPS; the incumbent stands. But the reason Gate D
  rejects the best arm is a defect in Gate D.** Full design and results:
  **`docs/games-played-plan.md`**. Everything below is measured on **validation** — the
  first pass decided Gate D on the test split, which reversed when re-decided correctly, and
  `src/models/held_out.py` now makes that impossible rather than discouraged.
  - **The frame is `entry index × exit index × a within-tenure two-state chain`, because a
    plain full-window chain FAILS at its own ceiling.** *A departure is an absorbing hitting
    time, not a low recovery rate* — a waived player's cell has `r̂ ≈ 0`, so a recurrent
    chain makes him absorbing from his **first absence** rather than from the game he was
    cut. Gate 0 measures it with per-cell empirical hazards, the most generous
    parameterization available: the plain chain over-predicts `P(GP<41)` by **21.9%**,
    **z = +5.29** against the sampling error of the observed proportion, where the tenure
    decomposition misses by **3.8%** (**z = +0.93**).
  - **⭐ The process class is right and the TENURE is the bottleneck.** The oracle-tenure arm
    scores validation CRPS **7.2265** against its own floor's **10.0992** on the same rows
    (the incumbent reads **10.0057** on the full validation frame) — 28% better on the
    deciding split. Given the observed tenure the within-tenure chain is far better than the
    season-level beta-binomial, and every bit of that is destroyed by having to predict entry
    and exit from preseason covariates. **The bottleneck is knowing when a player joins and
    leaves a roster**, which is mid-season churn and already out of scope. That also says
    where *not* to spend: more flexibility on the absence process cannot recover what the
    tenure factors lose.
  - **⭐ Gate D is a MARGINAL gate and cannot pass a marginal-neutral arm.** The `hybrid` arm
    draws its games-played count *from* the incumbent's pmf, so its marginal is identical
    rather than approximate — CRPS 10.0057 against 10.0057, PIT 0.0939 against 0.0939, tail
    0.0406 against 0.0406, to every decimal. All three of Gate D's criteria are marginal and
    its tail test is a strict inequality, so the hybrid ties every bar and fails on the tie.
    **That is a category error in the instrument, not a verdict on the arm** — and it is the
    reason the head's decision is a judgement rather than a gate outcome.
  - **The fitted arms lose on their merits**: `duration_covariates` scores **10.1625**
    against the incumbent's 10.0057, a gap of **+0.1568** (paired bootstrap over 883 rows,
    95% CI [+0.0737, +0.2393], P(better) = 0.1%), `full_window` +0.2647, `three_state`
    +0.2983, `calibrated_fallback` +0.0149 and it also loses PIT. `three_state` posts the *best* tail of any arm (0.0041)
    while being worst on CRPS — the tail alone is a noisy criterion on 359 rotation rows.
  - **⭐ The spell SHAPE is where the arms differ, and no marginal metric can see it.** CRPS,
    MAE and `P(GP<41)` are all functions of a pmf over a count: two models with identical
    games-played distributions can scatter absences as coin flips or block them into a
    fortnight and score the same. Simulated spell lengths against observed, mean absolute
    relative error: **hybrid 0.5310**, `full_window` 1.7408, `duration_covariates` **1.8105**,
    `calibrated_fallback` **5.0082**. The hybrid is within 2% on both P(T=1) and P(≥10 games)
    — the statistic a Round 1 knockout turns on — and its weakness is the extreme tail
    (+157% on month-long absences).
  - **A constant recovery hazard is disqualifying for a best-ball simulator.** The
    calibrated fallback's geometric spells have a mean length of **10.06 games against an
    observed 3.09**, P(T=1) of 0.1171 against 0.4924, and it understates a season-ending
    absence by 99% at the fitted interior rate. Use the beta-geometric.
  - **The collapse is exact and it is what makes this affordable.** 1,297,766 transitions →
    **32,944** binomial rows, a **39.4×** reduction with an *identical* posterior, so
    `docs/availability-plan.md`'s premise that a game-level fit would be "a much larger fit"
    is wrong — it is the same size. The matched-frame clustering the fallback inverts is
    **9.5806**.
  - **Gate B passes**: the onset head scores **−0.330658** per at-risk transition against a
    shrunk carry-forward's **−0.350239** (**+0.019580**). **Gate C passes**: the simulated
    hazard-by-streak curve keeps falling past a streak of 20, which is the check that the
    frailty has not collapsed to a point mass — a failure invisible in the GP marginal.
    **Gate E has not run**: `season_total.py` still evaluates on the held-out split.
  - **The duration head is beta-geometric and fits INTERIOR spells only** —
    `src/stan/betageometric_duration.stan`, the same Beta-frailty device
    `betabinomial_glm.stan` uses one level down, beating the geometric by **11,278**
    log-likelihood points at one extra parameter. Interior only because three independent
    measurements say the edge spells are roster mechanics: `not_rostered` shares of 58.7% /
    46.2% against 1.0% for interior spells; **82.7%** agreement between the structural proxy
    and `not_rostered`; and the length-biased renewal identity failing in the same direction,
    with in-progress spells **2.2×** longer than it allows. The cell the structural proxy
    cannot see — missed outside the appearance window while still rostered — is **16.74%**
    of missed games, and reaching it is what the three-state arm was for.
  - **Ignoring censoring fails silently both available ways.** Dropping censored spells
    understates `P(T ≥ 26)` by **2.16×**; treating them as complete still understates it.
    With proper censoring the fitted `a` = **0.721** < 1, so the beta-geometric has **no
    finite mean** and a polynomial tail — the simulator does not care, but `E[T]` must never
    be quoted from that fit.
  - **The onset hazard's 14.3× fall with played streak is SORTING, not state dependence.**
    High-hazard players break their streaks early, so long streaks are populated by
    low-hazard players — the frailty the beta-binomial marginalizes generates the whole curve
    for free, which is what lets the collapse keep it.
  - **⚠️ The tail target is itself an estimate.** 339 rotation rows on test and 359 on
    validation put standard errors of ~0.02 on the observed tail probabilities — the same
    order as the differences between arms. Do not read a tail comparison as sharper than that.
- **The injury-report transfer function is measured, and the designation scale is NOT
  monotone.** 12,338 of 12,406 archive rows joined to a realized box-score outcome (99.5%,
  **0.0% unmatched names**), 142 game dates inside 2025-26, 532 players. `P(play)`:
  **Out 0.002, Doubtful 0.030, Questionable 0.498, Probable 0.914, Available 0.855** —
  `Available` plays *less* than `Probable`. That inversion is reason mix, not noise in the
  labels: excluding G-League rows the scale reads 0.001 / 0.022 / 0.553 / 0.919 / **0.903**,
  and the residual 1.6 pp sits inside a ~1.6 pp standard error. **Treat Probable and
  Available as one designation**, and condition on `reason_category` rather than on the
  five-level scale. `make report-calibration`, `src/eda/report_calibration.py` →
  `data/features/report_transfer.parquet` (20 cells) + `outputs/eda/report_calibration.csv`.
  - **⚠️ This block was refreshed 2026-07-31 and had read 12,007 of 12,406 (96.8%) with
    Doubtful 0.027 / Questionable 0.500 / Available 0.852.** `docs/availability-plan.md`
    carried the same block and was refreshed on 2026-07-30 when the box-score backfill closed
    331 previously uncovered rows; this copy was not. It is the second time this exact block
    has gone stale in one doc while being current in another, which is why it is now claimed
    in `src/docs_audit.py` from **both** docs against the one artifact.
- **`Out` is near-deterministic and sticky; `Questionable` is a coin flip that resolves.**
  Out → 89.8% inactive / 9.4% dnp / 0.2% played, and **98.5% of Out designations are
  unchanged** in the next day's report. Questionable is unchanged only 38.9% of the time,
  and a *stale* Questionable is worth about what a fresh one is (p_play 0.475 at lead 1 vs
  0.512 at lead 0) — which is the encouraging read for a preseason snapshot, since that is
  read weeks ahead. Minutes barely move: a Questionable who plays gets 23.5 against a
  Probable's 24.7, so there is **no meaningful minutes haircut** to model — the designation
  acts on the play/not-play margin, not on workload.
- **The stated reason disambiguates `inactive` directly — no Basketball-Reference scrape
  needed.** The plan wanted BBRef transaction logs to separate "unavailable because hurt"
  from "unavailable because not on the team". The PDFs state it: `Out|Not With Team` is
  **12.5% `absent`** from the box score and `Out|Trade Pending` **14.3%**, against 0.2% for
  `Out|Injury/Illness`. Roster mechanics are the only reasons that generate `absent` at all.
  This is forward-only (the archive starts 2025-12-29), so BBRef remains the only option for
  history — but for the preseason snapshot the cheaper source is sufficient.
- **Basketball-Reference cannot supply historical injury data — checked, not assumed.**
  `/leagues/NBA_2024_transactions.html` is 323 KB with **zero** occurrences of "injur"
  (the formal Injured List ended in 2005); `/friv/injuries.fcgi` is a 38-row *current
  status* page with no history, so it carries the same point-in-time prohibition as the
  ESPN feed; `*/gamelog/` is `Disallow`ed. It *is* a legitimate dated source of **roster
  movement**, which is the right way to disambiguate `inactive` (hurt vs. not on the team)
  — a roster-membership source, not an injury one.
- **Point-in-time discipline is the largest correctness risk in the availability head.**
  Never fill a historical row from a current-status source: the ESPN feed and
  Basketball-Reference describe *today*, so a 2019 row filled from them encodes the resolved
  outcome, which is the target. Only dated-at-publication sources may fill history — the NBA
  report PDFs, transaction logs, and the box-score backfill. `return_date` is a **forecast
  made on the snapshot date** and must never be overwritten with the realized return; read
  the ESPN log only through `injuries.snapshot_as_of`, and every training row through
  `models.availability.assert_point_in_time`.
- **An unmatched rate does NOT validate a join — it can be bought by fabricating rows.**
  The single most dangerous measurement in this repo, because it reads as a success metric
  and rewards the worst failure mode. Measured while building the ADP id map: a
  "same surname + same first initial" rule scored **0.0% unmatched** and was wrong on
  **11 of its 12 non-exact matches** — `Cameron Boozer` → **Carlos Boozer**,
  `Darryn Peterson` → **Drew Peterson**, `RJ Davis` → **Ricky Davis**, `Javante McCoy` →
  **Jelani McCoy**. Every fabricated match *improves* the score, so the metric is
  monotonically increasing in exactly the error it is supposed to detect. Rules:
  - **List every non-exact match and read them.** A fuzzy tier small enough to eyeball is
    a feature, not a limitation; if it is too large to read, it is too large to trust.
    ✅ **`make adp-panel`** (`adp.match_audit` → `outputs/eda/adp_match_audit.csv`, 57 rows)
    emits all **16** surviving fuzzy matches with both names, the rule and the seasons-apart
    gap, over 3,591 unique (source, name, season) rows. All 16 read correctly.
  - **Report unmatched alongside the count and method of fuzzy matches**, never alone.
  - ✅ **The rejected rule now runs forever, as an ablation beside the metric it discredits.**
    Re-run on the same rows it makes **31** matches, **23** of which the cascade refuses, and
    it scores **0.00% unmatched against the cascade's 0.50%**. `Cameron Boozer` → Carlos
    Boozer and `Darryn Peterson` → Drew Peterson come straight back out, alongside
    `Mikel Brown Jr.` → Moses Brown, `Baba Miller` → Brandon Miller and
    `Dillon Mitchell` → Davion Mitchell. A *better* unmatched rate on a rule that fabricates
    matches, in the same file as the rate, is the point: if someone later simplifies the
    cascade back toward it, the audit says so on the next run.
  - **Two independent guards beat one clever rule.** A first name must be a genuine
    *prefix* (≥3 chars) **and** the candidate must have played within ~3 seasons of the
    row. Either alone still invents people: prefix-only matched `Mikel Brown Jr.` (a 2026
    rookie) to **Mike Brown**, last seen 1996-97.
  - **Keep "no such entity" apart from "join failed."** 162 DK pool entries have no
    `player_id` because they have never played an NBA game — reported as `no_nba_history`,
    not `unmatched`, the same distinction `report_calibration.py` draws between `absent`
    and `unmatched`. Collapsing them turns "the 2026 draft class exists" into a fake defect
    and hides real ones.
  - Corollary, now **audited and passed**: `report_calibration.py`'s "0.0% unmatched
    names" holds up. It has no fuzzy tier at all — the join is exact-match on `name_key`
    and scoped to a single `game_id`, so the cross-era failures above cannot occur. The
    residual risk is a **same-game name collision**, measured at **1 cell in 797,473
    status rows over 20 seasons** (game `0021200757`, two different `Chris Johnson`s).
    Those rows are now `outcome = "ambiguous"`, excluded from `OUTCOMES` and reported in
    `coverage`; `share_ambiguous` is **0.0%** on the current archive overlap.
- **Name-based joins exist in exactly two places; everything else keys on ids.**
  `report_calibration.py` (`game_id` + `name_key`) and the ADP modules (`player_key`).
  `preprocess.py` and `dataset.py` only ever group by `player_id`. **`injuries.py` and
  `injury_reports.py` store names and never join** — the ESPN feed carries no
  `player_id` at all and the PDFs carry none either, so any future consumer of those
  archives inherits the whole problem. Give them the ADP cascade rather than a fresh
  `merge`, and route the result through the same audit.
- **`normalize_name` strips digits, which makes `P0`…`P39` a single key.** A synthetic
  test fixture built that way silently exercised a 40-way collision and *passed*, because
  the join attached one arbitrary player's outcome to all 40 report rows and the assertion
  was on the row count. Synthetic names in tests must stay distinct **after**
  normalization — use letters.
- **Reconstructing absences: key `played` on `(player_id, team_id, game_id)`.** A `game_id`
  belongs to *both* teams, so a two-key merge credits a traded player with games he played
  for his new team to his old one, and stretches his appearance window to season's end.
- **The panel's `status` column replaces the roster-window bracket with a measurement**
  wherever `make boxscore-status` has run: `played` / `dnp` (dressed and available, not
  used — a *rotation* fact, not a health one) / `inactive` / `not_rostered` / `unknown`.
  Keep `not_rostered` and `unknown` apart: coverage is tracked **per game**, so only inside
  a backfilled game does a missing box-score row mean "not on that roster". Collapsing them
  makes a half-finished backfill read as a league-wide roster collapse. Game ids are ints in
  the game logs and zero-padded 10-char strings in the box-score files — normalize with
  `boxscore_status.pad_game_id` or the merge matches nothing, silently, while still
  returning a full panel.
- **Reliability is per column, not one curve.** `r(m) = r_inf · m/(m+m0)` refitted for every
  column: median `m0` **364** minutes (p10 112, p90 ≥ 2000) against the global 66, and
  minutes-for-r=0.75 spans 192 (`fg3a`) to > 20,000 (clutch), median **1,416**. The global
  `0.924 · m/(m+66)` was fitted on eight *style* stats — the fast-stabilizing end — so it
  over-credits the median column. Per-column `reliability_r_inf` / `reliability_m0` /
  `minutes_for_r75` are in `persistence.csv`; prefer them where a column-specific shrink is
  cheap. Lag decay is also column-specific: `usg_pct_reb` 0.854 → 0.843 over lags 1→4
  against `bas_pts` 0.861 → 0.698, so the flat 0.96/season staleness decay is too fast for
  shares and too slow for volume.
- **Aging lives in availability, not in rates.** Delta-method, era-adjusted,
  minutes-weighted, indexed to age 23: per-36 DK-linear peaks at **26** (1.041) and is still
  0.895 at 34; minutes/game peaks at 27 (1.111) then falls to 0.792 at 34 and **0.515 at
  37**; games played peaks at 28 (1.054), 0.952 at 34. A ±15% rate arc against a −54%
  availability arc — put age in the availability head.
- **Cross-sectional age curves are worthless here.** Mean per-36 DK-linear reads 31.1 at 19,
  31.9 at 23, 31.6 at 30, 31.4 at 34 and **34.3 at 39** — flat, then rising. That is pure
  survivorship. `cross_sectional_mean` sits beside `cumulative` in the output only to make
  the gap visible; never build on it.
- **Aging is component- and archetype-specific.** At 34: `reb/36` 0.977, `blk/36` 1.011,
  `stl/36` 0.962 — flat — against `pts/36` 0.844, `ast/36` 0.867, `fg3m/36` 0.872. Big-man
  counting stats barely age; skill and scoring do. By archetype, the 3-point-heavy cluster
  is 1.029 at 34 while "low scoring volume, low usage" is 0.805. A single dk_pts age curve
  averages two opposite shapes — another reason for component heads.
- **Zero-inflation is a minutes artifact, not a property of the target.** `dk_pts` is 0 in
  35.4% of sub-5-minute games and **0.0% above 18 minutes** — conditional on minutes the
  zeros are Poisson zeros and there is nothing to zero-inflate. But in 30–48 minute games
  `blk` is still 0 in 58.6% of games, `fg3m` 42.2%, `stl` 33.8%: those three heads are
  genuinely low-count and misspecified under MSE at *any* minutes level.
- **Dispersion is worst at *low* usage, not low minutes.** Within-player var/mean: `fga`
  1.62 below 0.15 usage against 1.32 above 0.30, `reb` 1.72 vs 1.22, `pts` 3.28 vs 2.64. The
  Poisson specification is tightest where the minutes are. If a dispersion term is added,
  key it on **usage**, not minutes.
- **Five games settle 86% of the season total.** Extrapolating the first-k mean over the
  games actually played — so this isolates the *rate*, holding availability known — gives
  R² 0.859 / 0.905 / 0.942 / 0.974 at k = 5 / 10 / 20 / 41 and MAE 238 / 193 / 148 / 94
  dk_pts. This is the measured price of the prior-season-only constraint.
- **The season total's two factors, on the log scale where they are exactly additive** — ✅
  `make target-profile` (`analysis == "season_total_decomposition"`). On 12,996 player-seasons
  with **≥10 games**, log(season total) is **84.54%** explained by log(per-game rate) alone and
  **73.43%** by log(games) alone (games: mean 56.0, sd 20.8, p10 23, p90 80).
  - The two **overlap rather than partitioning** — they correlate at **+0.585**, which is why
    the shares sum past 1. Fitting both returns R² exactly **1.0000**, since
    `log(total) = log(rate) + log(games)` is arithmetic; that row is a free check that the
    decomposition is of the right identity.
  - **The ≥10-game floor is load-bearing.** `preprocess.clean` filters on a player's *career*
    games, not his season, so the unfiltered frame carries one- and two-game seasons whose
    per-game rate is noise: it reads 78.5% / 81.7% and **reverses which factor dominates**.
    `SEASON_TOTAL_MIN_GAMES` names it and a test pins that it changes the answer.
- **The raw feature matrix is singular, not merely collinear.** Tier A rank 144/149, Tier B
  277/285, condition number infinite. Sixteen Tier A columns have exactly infinite VIF
  because they are literal cross-family duplicates — `adv_def_rating` == `def_def_rating`,
  `adv_dreb_pct` == `def_dreb_pct`, `usg_pct_blk` == `def_pct_blk`, `usg_pct_stl` ==
  `def_pct_stl`, `usg_pct_dreb` == `def_pct_dreb` — and others are exact complements
  (`sco_pct_fga_2pt` + `sco_pct_fga_3pt` = 1). **Any unpenalized GLM/OLS on the raw matrix
  fails outright**; ridge/elastic-net or a pruned set is mandatory. 104/149 columns have
  VIF > 10 (median 92); pruning at |r| ≥ 0.95 collapses 74 Tier A columns into 22 clusters
  (largest 9 — the whole rebound family), leaving **97 of 149**, and 152 Tier B columns into
  50, leaving 183 of 285.
- **Key team features on `team_id`, never `team_abbreviation`.** `team_id` has 30 categories,
  min 317 rows, no cold start. The abbreviation has **36** categories, min 24 rows, with
  NOK/VAN/CHH/NOH/SEA spanning as few as 2 seasons — relocations split their own history.
  `archetype` (9 Tier A / 6 Tier B, min 540/377) and `draft_bucket` (5, min 1,456) are safe
  to one-hot.
- **The prior-season game *sequence* is worth ~+0.6 pp R².** Held out on the last two
  seasons (10,215 train / 889 test): four season aggregates (mean dk, sd dk, mean minutes,
  games) give R² **0.7599**; adding five order features (dk and minutes slope, lag-1
  autocorrelation, last-10 gap, half-to-half gap) gives **0.7657**; the same five computed
  on **shuffled** game order give 0.7584. So +0.0059 over aggregates and +0.0074 above its
  own null — real, and ~0.8% relative. The LSTM/Transformer trunk over prior-season game
  logs is mostly re-deriving a season mean `season_matrix.py` already holds.
  **Decision (do not relitigate — this has come up repeatedly in planning): deprioritize
  the LSTM/Transformer trunk** relative to the component/availability/joint-correlation
  modeling work in `docs/predictions-plan.md`. Revisit only against new evidence that
  per-game sequence structure matters more than measured here.
- **There is no shooting hot hand, so the successes/trials heads collapse for free — the
  serial structure is all in the exposure.** `make serial-correlation`
  (`src/eda/serial_correlation.py` → `outputs/eda/serial_correlation.csv`), 592,796
  player-games / 9,052 player-seasons (played, `min ≥ 5`, ≥40 games). Pearson residuals
  against each player-season's own rate with `min` as exposure — so minutes are already
  conditioned out of the count rows — against a null that permutes game order **within**
  player-season. That null is load-bearing: it carries the ≈ −0.016 bias that within-season
  demeaning induces, without which `stl` and `tov` read as negative dependence when both are
  mildly positive.

  | component | lag-1 excess | 10-game block variance inflation |
  |---|---|---|
  | **`min`** | **+0.294** | **2.43×** |
  | `min` detrended | +0.212 | 1.71× |
  | **`fg3a\|fga`** — the shot **mix** | **+0.101** | **1.57×** |
  | `fga` | +0.061 | 1.38× |
  | `ast`, `fta`, `reb`, `blk`, `stl`, `tov` | +0.008 … +0.039 | 1.07–1.22× |
  | `ftm\|fta` | +0.012 | 1.10× |
  | **`fg2m\|fg2a`** | **+0.002** | **1.03×** |
  | **`fg3m\|fg3a`** | **−0.002** | **1.01×** |

  - **⚠️ "Conversion head" stopped being a synonym for "shooting head" when the shot-attempt
    basis landed, and the summary line was corrected for it.** `fg3a | fga` is a
    beta-binomial like the makes, but it measures shot **mix**, not accuracy — and it is the
    *largest* non-minutes dependence in the table (**+0.101**, z = 94, block inflation
    **1.57×**), above the `fga` count it splits. Shot selection drifts within a season the
    way minutes do; shooting accuracy does not. Quoting one max over all four conversion
    heads would report that drift as a hot hand and simultaneously hide that the three
    *shooting* heads are still clean nulls (max |excess| **0.0122**).

  The answer splits on the **attempts vs conversion** line `persistence.csv` already found
  at the season level, now confirmed at the game level — with the mix share sitting on the
  attempts side of it despite being a conversion head by likelihood. Both field-goal
  *shooting* rows are nulls (z = 1.9 and −1.3 on ~600k pairs), so constant-θ-within-season —
  exactly what the binomial collapse assumes — is what the data looks like. What *is*
  dependent is the exposure side: minutes at 2.43×, shot volume at 1.38× and shot mix at
  1.57× **on top of** minutes. Decay is
  slower than AR(1) (minutes reads 0.278/0.212/0.170/0.113 at lags 1/2/3/5 against AR(1)'s
  0.278/0.078/0.022), and removing a within-season linear trend drops lag-1 to 0.196 — so
  roughly a third is slow role drift and two-thirds a shock with a 3–5 game e-folding.
  Rotation churn and injury ramps, not shooting form. **Put the sequential model on minutes,
  beside the availability spell process, and leave the shooting heads collapsed** — but note
  that `fg3a | fga` is now the one head besides minutes with a serial story worth a second
  look, which is new information the two-count basis could not surface.
  Block inflation is the decision-relevant column: it is the factor by which an
  independent-draws simulator understates the variance of an aggregate.
- **Archetypes are a partition of a continuum, not discovered clusters.** k-means silhouette
  decreases monotonically in k and peaks at 0.181 (Tier A) / 0.241 (Tier B) at k=4 — no k
  shows real separation. GMM BIC picks k=9 / k=6, which is what is fitted. Use the **soft**
  membership vector; a hard archetype label claims structure the data does not have.
- **The shuffled-null helper reproduces the one-off it generalizes.**
  `feature_diagnostics.cell_importance` gives +0.9619% permuting opponent and +0.3080%
  permuting archetype against `opponent.variance_ceiling`'s +0.9570% / +0.3056% — gaps of
  0.005 / 0.002 pp. Route any future importance ranking through it so a score always ships
  with its own chance level.

### Data quirks

- `MIN` in season files is **per-game**, not a total (except where `per_mode` says
  otherwise). Total minutes = `MIN * GP`.
- **Per-36 rule:** `MIN` always carries the same basis as the stats beside it, so
  `stat / MIN * 36` is correct for both `PerGame` *and* `Totals`. `PerMinute` is `stat * 36`.
  `Per100Possessions` has no minutes basis and raises.
- Season CSVs are already **one row per player**, deduped across trades; a traded player is
  attributed wholly to his **last** team.
- ~~**`preprocess.load_raw` turns playoff logs into pseudo-seasons**~~ — ✅ **fixed
  2026-07-29.** It globbed `game_logs_*.csv`, which also matches the 30
  `game_logs_playoffs_*.csv` files added by the availability work, and derived `season`
  from the filename slug — so a playoff file became `playoffs-1996-97`, doubling the
  season count for every `groupby(["player_id", "season"])`. `load_raw` now takes
  `season_type` (`"regular"` default / `"playoffs"` / `"all"`), parses the prefix off the
  slug so both types share one `season` label, and tags rows with `season_type`, which
  `clean` carries through. An unknown value raises rather than returning an empty frame.
  Rebuilding gives back exactly the pre-existing 731,906 rows, so the fix restored intent
  rather than changing data.
- **`component_targets.parquet` is a filtered frame, not the raw log.** `preprocess.clean`
  drops players below `data.min_games` (786,765 raw rows → 731,906). Fine for per-player
  analysis; wrong for anything that aggregates a whole team-game, which is why
  `game_length.py` reads raw.
- `player_shot_locations` has a **two-row MultiIndex header**; every other family is flat.
- Empty `LeagueDashPlayerStats` responses are **poisoned server-side cache entries** keyed
  on the full query, not rate limiting. Backoff does not help; cycling `per_mode_detailed`
  does. `fetch.py` handles this and records the winning mode in `data/raw/_fetch_manifest.csv`.
- Player `"Four Factors"` is genuinely unsupported by the endpoint — its stats live in
  Advanced. Team four-factors works fine.
- Coverage: core families 30 seasons; tracking/`pt_shot` 13 (2013-14+); estimated 12
  (2014-15+); hustle 11 (2015-16+, and **2015-16 covers only 147 players** — a partial
  first-season rollout, not a fetch failure); **box-score inactive lists 20 (2006-07+)**.
- **The inactive list is a third coverage boundary: 2006-07.** Probed on both sides —
  0 rows for 2005-06, 6 on 2006-07 opening night. Earlier seasons still separate played
  from dressed-but-scratched (the traditional `COMMENT` field goes back further) but
  cannot see who was inactive.
- **Two box-score endpoints have version traps, and they fail in opposite ways.**
  `BoxScoreTraditionalV2` returns **0 rows** from 2025-26 on — loud, obvious, route to V3
  (the field is lower-case `comment`). `BoxScoreSummaryV2` is the dangerous one: after
  **2025-04-10** it still returns 200 with a populated `GameSummary` and silently drops
  the `InactivePlayers` result set, so a backfill built on it records "nobody was
  inactive" rather than failing. Measured: 223 of 228 games in 2025-26 came back empty.
  **Use `BoxScoreSummaryV3`** (`boxScoreSummary.{home,away}Team.inactives`, team id on the
  parent block) — it covers 2006-07 → 2025-26 and agrees with V2 exactly where V2 works.
- **A third V3 failure mode: `nba_api`'s own parser raises on a stub payload, and the guard
  in `_inactive_rows` cannot catch it.** Three games (`0022500259`–`0022500261`, all
  2025-11-19) return a `boxScoreSummary` whose `arena`, `teamId` and `inactives` are all
  `null`. `nba_api/stats/endpoints/_parsers/boxscoresummaryv3.py::get_arena_info_data`
  calls `arena.get("arenaId")` unguarded, so it throws `AttributeError: 'NoneType' object
  has no attribute 'get'` **inside the constructor** — before `get_dict()` is ever reached.
  The `payload.get("boxScoreSummary") or {}` guard therefore never runs, and the error text
  reads like a bug in this repo when it is upstream. Confirmed there is nothing to recover:
  the raw JSON (via `NBAStatsHTTP().send_api_request`, which bypasses the parser) is a stub,
  and V2 returns 0 `InactivePlayers` for the same games because 2025-11-19 is inside its
  silent-failure window. Only the *traditional* side survives (28 dressed, 19 played).
  **Left unfetched deliberately.** Recording the dressed rows without the inactive list
  would mislabel those games' inactives as `not_rostered` — the exact collapse the panel's
  per-game coverage tracking exists to prevent. With no status rows the game is
  `status_covered = 0` and every row is `unknown`, which is correct. 3 of 25,709 games.
- **The backfill manifest is append-only, so counting `error` rows overstates the damage.**
  A retried game keeps its old failure row and gains a new success row. After the second
  pass the manifest holds 77 error rows against **3** games that actually lack data — dedupe
  to games with no successful attempt (`set(all) - set(error.isna())`) before reporting.
  Every transient network fault (connection resets, read timeouts — 68 of them, clustered in
  2012-13 and 2013-14) cleared on one retry.
- **NBA injury-report PDFs**: `CreationDate` matches the filename key exactly, a 403 means
  the key does not exist (aged out, or no games — the All-Star break is absent from the
  middle of a live window), and neither is retryable. The PDF is landscape with a flipped
  text matrix, so **y increases downward** and rows sort ascending. Cells are left-aligned
  and wrap: assign a chunk to the *last* column starting at or before it, never by
  midpoints. A wrapped reason is typeset **centred** on its row (first line above the
  player name, continuation below) and can **wrap across a page break**, landing at the
  top of the next page as a tail of the previous page's last row. A team that missed the
  filing deadline gets a row with `NOT YET SUBMITTED` in the *reason* column and no
  player — it needs its own anchor or it corrupts the nearest player's row.
- The 7 `team_stats_*` families and `team_estimated_metrics` feed
  `src/features/opponent.py`. `team_stats_opponent`'s `OPP_*` columns are *what opponents
  recorded against this team* — already the per-component opponent profile. Never rebuild
  this by aggregating player rows.
- **Team files break the per-36 rule.** `fetch_team_stats` uses `PerMode="PerGame"` and the
  counting stats are per game, but `MIN` and `POSS` are **season totals** (`MIN` ≈ 3966 =
  48.4 × 82). Normalize per-100 possessions with `POSS / GP`, never `stat / MIN * 36`.
- **⚠️ `sklearn`'s two regularization conventions are opposite, and with exposure weights
  the difference is ~7 orders of magnitude.** `PoissonRegressor` (and every
  `_GeneralizedLinearRegressor`) minimizes `deviance / (2·Σw) + alpha·‖coef‖²` — the data
  term is **averaged by the weight sum**. Fitting a rate with `sample_weight = minutes`
  makes `Σw ≈ 10⁷`, so the default-looking `alpha=1.0` is an enormous penalty and shrinks
  every coefficient to nearly zero. Measured on the `reb` season-total head: R² **0.662 at
  alpha=1.0 against 0.928 at alpha ≤ 0.01**, and it fails *quietly* — the fit converges,
  coefficients are finite, and a flexible basis partially compensates, so a spline or
  interaction looks like it is buying real signal. `LogisticRegression` is the reverse: its
  objective is `Σ w·logloss + ‖coef‖²/(2C)`, **not** averaged, so with the same weights
  `C=1.0` is *weak*. **Always sanity-check a weighted GLM against a no-fit baseline** — for
  the component heads, `prior rate × actual minutes`, which scores R² 0.81–0.95 and would
  have exposed this immediately.
  - ✅ **The trap is now a curve and a permanent regression guard**: `make component-rates`
    (`alpha_sensitivity`, `analysis == "alpha_sensitivity"` — 98 rows over 7 count heads × 2
    variants × a 1e-8…10 grid, with `floor_r2` on every row). `reb` on the raw-rate spec reads
    **0.9181 at alpha=1e-8**, **0.9267 at 0.01** and **0.6619 at alpha=1.0** — and the middle
    pair reproduces the recorded 0.928 / 0.662 to a thousandth **across the move from the
    held-out split to validation**, which is the strongest available statement that this is a
    property of the penalty and not of any particular rows. At alpha=10, **14 of 14 fits fall
    below the no-fit floor**, the visual statement of why the floor is mandatory. (Recorded
    held-out: 0.9278 / **0.9322** / **0.6620**, and 16 of 16 — the 16 dating from the retired
    eight-head basis.)
  - **It hurts the shipped spec too, not just the misspecified one.** Median R² loss from
    `POISSON_ALPHA` to alpha=1.0 is **0.200** on `linear` and **0.305** on `log_own`, worst
    `blk` at **0.497**. Every head's best alpha on the grid is ≤ 0.01. (Recorded: 0.228 /
    0.289, worst `fg3a` at 0.595 — a retired head.)
  - **The guard compares each head against its own optimum, not against the floor.** `blk`
    loses to the floor at *every* alpha under `log_own` because it needs a spline — a
    documented modelling finding, not the penalty misbehaving — so a floor crossing alone
    cannot be the trigger.
- **The season matrix serves the PCA, not roster aggregation.** Its `GP≥20 & MIN≥10` filter
  keeps garbage-time per-36 outliers out of the PCA, but it also drops real teammates who
  consume real minutes. Build roster aggregates on `season_matrix_roster_tier*.parquet`
  (the unfiltered twin) with reliability shrinkage, never on the qualified matrix.

## Conventions

- No shared config helper exists; the idiom is
  `cfg = yaml.safe_load(open("configs/default.yaml"))` in `__main__` blocks, duplicated in
  several modules. Match it — do not refactor it away.
- Entry points run as `python -m src.<module>` from the repo root, with a matching
  `Makefile` target listed in `.PHONY`.
- Reuse `src/data/preprocess.py::compute_dk_pts` verbatim; never reimplement DK scoring.
- Reuse `src/data/fetch.py::_slug` / `_season_start_year` for season-key handling.
- `Path(...).mkdir(parents=True, exist_ok=True)` before every write; print
  `f"... {n:,} ... → {dest}"` progress lines.
- Artifact save/load mirrors `src/features/encode.py::save_artifacts` / `load_artifacts`.
- When a module defines classes that get **pickled**, its `__main__` block must import
  `run` through the package path (`from src.eda.pca import run as _run`) — otherwise
  classes pickle as `__main__.Foo` and cannot be loaded from any other process.
- Tests use plain `assert` with synthetic builders, no fixtures or classes (mirroring
  `tests/test_preprocess.py`).
- **When a load-bearing decision is taken, reversed, or measured, add or update its entry in
  `dashboard/decisions.py`** alongside the `CLAUDE.md` / plan-doc edit. The registry is what
  the dashboard's decision log renders, and it carries `source` and `reviewed` so
  `make dashboard-audit` can flag entries whose source doc has moved since. Statuses come
  from a closed vocabulary — a reversal becomes `withdrawn` and keeps its entry rather than
  being deleted, because the reversals are the most useful thing on that page.
