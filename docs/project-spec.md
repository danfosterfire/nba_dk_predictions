# NBA DK Predictions

Predicts a player's DraftKings NBA Best Ball fantasy points (dk_pts) for each 
game of an upcoming season, plus the running season total. Uses predictions to 
simulate historical and upcoming seasons and develop drafting strategies.


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

**The draft happens after the preseason and before the opener.** At prediction time we know:
(1) the season schedule, (2) **season-start rosters** — definitively which team each player
is on, plus team identity — (3) *previous-season* stats for every team and player, and (4)
the **current season's preseason box scores**. We do **not** know within-season roster
changes (mid-season trades), current-season *regular-season* minutes, injuries, or form.

The information set is therefore a **cross-season join: current-season roster membership ×
prior-season statistics × current-season preseason statistics.**

**Item (4) was added 2026-08-13** and is a change of problem statement rather than a new
feature — `docs/preseason-plan.md` is the whole of it. Preseason quantities enter as
**additional columns on existing heads**, difference-coded against the prior-season
equivalent on each head's own link scale, so that a coefficient of zero recovers the
pre-2026-08-13 head exactly. **Four head groups carry a block today** — availability,
marginal minutes, the minutes composition (through its prior share rather than a coefficient)
and **ten of the eleven component rate heads**, the eleventh opted out by
`stan_components.PRESEASON_EXCLUDE` as the round's one measured-worse result.
`stan.availability.preseason`, `stan.minutes.preseason`,
`stan.composition.preseason.adopt` and `stan.components.preseason` are exact rollbacks.
Three rules ride with it:

- **A preseason quantity is a forecast covariate, never a substitute observation.** Nothing
  from the preseason enters any head's likelihood as a target row — different coaching
  objectives, different effort, exhibition opponents.
- **Every preseason figure is quoted on the season-start-roster population.** Pooled over
  everyone who appeared in season S, a missing preseason row mostly means a January signing,
  and the same block reads 6.2× larger there than on the draft pool it is applied to.
- **A complete preseason is a production precondition.** Two shipped availability columns are
  read over the preseason's *tail*, so the runbook's Oct 17–20 draft window is load-bearing.

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
dashboard/     data visualizations over the precomputed artifacts — today the PCA
               player-style fingerprint. app/pca/charts/theme/artifacts, plus three
               files that are NOT the dashboard and are load bearing elsewhere:
               decisions (the registry), economics (tournament derivations), audit
               (make dashboard-audit). Everything but app/artifacts imports no
               streamlit; nothing in the package imports src/ — see dashboard/README.md
docs/          eda-plan (season-level EDA spec), availability-plan (games played /
               minutes), games-played-plan (the spell process: entry x exit x a
               within-tenure chain, and why a plain full-window chain fails),
               adp-plan (market proxy, sourcing + where ADP belongs),
               predictions-plan, minutes-composition-plan, simulations-plan,
               shot-attempt-basis-plan (the fga x fg3a|fga reparameterization —
               measured, adopted 2026-08-03, shipped), dk_best_ball_rules,
               provenance-plan (every figure gets a make target — and the six
               corrections that fell out of building them), dashboard-plan
```



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

## Train / validate / test split

There are 30 seasons of data available for this project: from the 1996-97 season through
the 2025-26 season. The project objective is to develop a drafting strategy before the
beginning of the 2026-27 season.

**The split is a temporal walk-forward by target season.** Every row is a season predicted
from the season before it, so 30 data seasons give **29 target seasons**, and the split is
a suffix of them:

- **Train — target seasons 1997-98 → 2021-22.** Exploratory analysis, model selection, and
  model fitting.
- **Validate — 2022-23 and 2023-24.** The only split model selection may read. Fit on
  train, predict validate, compare predictions against observed values, and settle
  hyperparameter and specification choices here. Also the window for developing and
  backtesting drafting strategies.
- **Test — 2024-25 and 2025-26. Do not use at any point during data preparation,
  evaluating modeling decisions/alternatives, model fitting, or evaluating simulated draft
  strategies. Reserved for a final end-of-project measurement of the whole workflow.**
- **Production.** Once the pipeline is settled and all modeling decisions are complete,
  refit on all 30 seasons, predict 2026-27 performance, and apply the drafting rules to
  2026-27 contests.

`test_seasons: 2` in `configs/default.yaml` is the knob. The seasons themselves are derived
by sorting the labels present and taking the last two — never hard-coded.

**The split is enforced by the code, not by discipline.** `src/models/held_out.py` makes the
test split a capability. `selection_split(design) -> (train, validation)` is what every
sweep calls, and it never materializes the test rows at all. `final_split` is guarded and
raises when read; `src/final_evaluation.py` (`make final-evaluation`) is the only thing that
unlocks it, and it refits on train **plus** validation before scoring test once. Anything
fitted from data — spline knots, imputation means, shrinkage constants, dispersion bin edges
— is estimated on the fitting half alone.

Detailed notes on the current status of this split are in
./docs/train-validate-test-split.md.

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
| `fta` | count — negative binomial | `min` |
| `fg2m` | successes / trials | `fg2a` |
| `fg3m` | successes / trials | `fg3a` |
| `ftm` | successes / trials | `fta` |
| `reb` | count — negative binomial | `min` |
| `ast` | count — negative binomial | `min` |
| `stl` | count — negative binomial | `min` |
| `blk` | count — negative binomial | `min` |
| `tov` | count — negative binomial | `min` |

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

