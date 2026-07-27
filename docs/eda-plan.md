# NBA Season-Level EDA: PCA, Archetypes, and a Streamlit Explorer

> **Status:** All stages complete. Stages 0–2 landed first, plus feature work beyond
> this plan's original scope; stages 3–4 (the dashboard and the four diagnostic
> modules) landed 2026-07-27. Written 2026-07-26, revised as
> measurements landed; revised again the same day to broaden Stage 4 so it serves three
> future model families instead of assuming one. See [Sequencing](#sequencing) for current
> status, [What changed](#what-changed-since-this-was-written) for the findings that alter
> the remaining specs, and [Serving three model families](#serving-three-model-families)
> for that broadening. **README.md is the authority on the modeling design**; this file
> covers the EDA and dashboard pipeline only — not fitting or evaluating any model.

## Serving three model families

The eventual modeling work will try three families against the same component targets —
regularized/hierarchical regression (a GLM per head, with player/team partial pooling),
gradient-boosted trees, and the deep multi-head net already sketched in
`src/models/multihead.py` — then combine them in an ensemble. That reshapes what "done"
means for the remaining EDA stages:

- **Ensembling happens at the component level** (per-36 rate, minutes, `P(play)`), not on
  raw `dk_pts` — the DK-weighted sum and the joint nonlinear bonus are shared machinery
  applied once, after combining base models' component predictions. Every EDA output
  needs to stay at the component level, which the plan already does (`target.py`, the
  opponent/team-context work) — nothing to change there, just to keep front of mind.
- **Stage 4 was scoped as if only the NN existed.** `target.py`'s distributional
  diagnostics were originally framed as validating `multihead.py`'s Poisson choice; the
  same diagnostics decide a GLM's family (Poisson/NegBin/Tweedie/log-normal) and a GBM's
  objective. One measurement, three consumers — reframed below rather than duplicated.
- **Each family has a failure mode the others don't**, which the current plan doesn't
  check for: GLMs need collinearity diagnostics before choosing raw-plus-regularization
  vs. a reduced feature set; GBMs will find spurious splits on the 892-team-season
  composition block unless importances are read against a null, not raw gain; the NN's
  sequence trunk over prior-season game logs might just be re-deriving what
  `season_matrix.py` already aggregates for free. A new module,
  `src/eda/feature_diagnostics.py` (Stage 4, item 7 below), covers all three.

Fitting, tuning, or evaluating any of the three families is out of scope for this
document — see README's Implementation plan table for that.

## What changed since this was written

Four measured findings modify the specs below. Two of them contradict what was originally
planned, and both remaining stages are affected.

1. **Absorb season in every pooled regression.** Pooled across 30 seasons, `teammate_spacing`
   correlates +0.315 with next-season per-36 points; with season fixed effects, +0.035.
   Spacing, pace and scoring all roughly doubled over the sample, so anything built from
   them tracks anything rising. This already produced one false finding that reached the
   README. **This directly hits Stage 4's `persistence.py`**, whose core statistic is a
   pooled season-*t* vs season-*t+1* correlation — as originally specified it will report
   era drift as persistence. Use `src/eda/context_value.py::season_dummies`.
2. **Minutes-weight anything computed on per-36 rates.** `sd(pts_per36)` is 42.4 in
   sub-5-minute games against 6.6 above 24 minutes. Unweighted, garbage time dominates:
   the opponent main effect on `reb_per36` reads 0.014% unweighted and 0.168% weighted.
   **Hits `persistence.py` and `target.py`.** Use
   `src/features/opponent.py::minutes_weights`.
3. **"Above a shuffled null" is null-dependent.** The same interaction statistic is +0.97%
   or +0.31% on identical data depending only on which marginal the null permutes. Always
   state the construction, and prefer held-out evaluation to in-sample ANOVA.
4. **Two consumers want two row sets.** The `GP≥20 & MIN≥10` filter is correct for the PCA
   and wrong for roster aggregation — 15.9% of roster minutes belong to players it drops.
   Stage 1 now also emits an unfiltered twin; see below.

## Context

The repo currently trains an LSTM/Transformer on `game_logs_*.csv` alone — a player's last N games of the prior season → next-season per-game `dk_pts`. Everything else that was scraped is unused: the only raw family anything in `src/`, `tests/`, or `notebooks/` reads is `game_logs_*.csv` (via `preprocess.load_raw`). The other **~30 season-level families** — advanced, usage, scoring, misc, bio, clutch, shot locations, 9 tracking families, hustle, estimated metrics, and 7 team families — are fetched and sitting on disk (284 MB, 30 seasons, 1996-97 → 2025-26, all complete including 2025-26) but entirely unconsumed.

The goal is to move from "predict per-game dk_pts from prior-season game logs" to "predict per-game dk_pts **and running season total** from prior seasons + own-team composition + opposing-team composition." Team composition is the missing input, and there is no way to express it without first having a compact, meaningful description of what kind of player someone is. That is what the PCA is for: reduce ~80–180 correlated season stats to a handful of interpretable style axes, cluster those into archetypes, and then represent any team as a minutes-weighted distribution over archetypes.

This plan builds that foundation, plus the supporting EDA that tells us which prior-season signals are actually worth feeding a model.

### Data facts established during exploration

- **Season coverage:** core families 30 seasons; tracking + `pt_shot` 13 (2013-14+); hustle 11 (2015-16+); estimated metrics 12.
- **Season files are already one row per player** — deduped across trades (`TEAM_COUNT` marks traded players; 2023-24 has 572 rows / 572 unique `PLAYER_ID`). No aggregation needed.
- **Joins are clean.** All families cover 399/399 qualified players in 2023-24 on `PLAYER_ID`.
- **`MIN` in season files is per-game**, not total (`PerMode="PerGame"`). Total minutes = `MIN * GP`.
- **Two data defects — both diagnosed against the live API (see below).**
- **`player_shot_locations_*` has a two-row MultiIndex header**; every other family is flat.
- Most families carry a long tail of `*_RANK` columns plus `TEAM_COUNT` / `NICKNAME` that must be dropped.

### Fetch-failure diagnosis (completed — live API probes)

The initial hypothesis was a rate limit during the first scrape. It is not, but retrying *is* the fix — for a different reason, and the difference changes the repair.

**Root cause: poisoned server-side cache entries keyed on the full query string.** Evidence:

- Only the `Defense` measure is affected. For 2021-22, `Base`/`Advanced`/`Scoring`/`Usage`/`Misc` all return 605 rows; `Defense` returns 0.
- **Empty responses return in ~0.1s; real ones take ~2s.** The empty result never reaches the backend — it is a cached negative.
- Backoff does not help: 6 attempts with 2→30s backoff produced 24 consecutive empty responses.
- **Changing `per_mode_detailed` does help**, because it changes the cache key. 2023-24 `Defense`+`PerGame` → 0 rows in 0.15s; `Defense`+`Totals` → **572 rows** in 2.1s.

**All 6 broken seasons are recoverable** by cycling `per_mode`, verified end-to-end:

| Season | Recovered via | Rows |
|---|---|---|
| 2000-01 | PerGame | 441 |
| 2002-03 | PerMinute | 428 |
| 2004-05 | PerGame | 464 |
| 2005-06 | Totals | 458 |
| 2015-16 | PerGame | 476 |
| 2023-24 | Totals | 572 |

All six carry `GP`, `MIN`, `DEF_RATING`, `DEF_WS`, `STL`, `BLK`, `DREB`, so every variant normalizes to a common basis.

**Two further findings:**

- **Latent bug in `_skip_or_fetch`** (`src/data/fetch.py:72-77`): it only tests `dest.exists()`. A header-only file counts as fetched, so re-running the pipeline would have skipped these five files permanently. This is why the corruption survived; it must be fixed or the backfill cannot run.
- **Player `Four Factors` is genuinely unsupported** — it fails with `KeyError: 'LeagueDashPlayerStats'` (the response omits the result set entirely) under every `per_mode`, a different failure mode from the cache poisoning. No loss: `EFG_PCT`, `TM_TOV_PCT`, `OREB_PCT`, and `FTA_RATE` are all carried by the Advanced family. It should be removed from `PLAYER_STAT_MEASURES` so it stops printing errors.
- Scanning every CSV for header-only files found **exactly these five** — no other family is affected.

### Decisions taken

- **Both feature tiers.** Tier A = 30 seasons, box-score-derived (~80 features, ~12k player-seasons). Tier B = 13 seasons, + full tracking/hustle (~180 features, ~5.5k player-seasons).
- **Both era treatments.** Era-adjusted (within-season z-scores) is the default and the one that feeds modeling; pooled-raw is kept so the dashboard can show era drift (the 3-point revolution) as a visible trajectory.
- **Full EDA sweep** — PCA + archetypes + persistence + aging + target analysis.
- **EDA is model-family-agnostic.** Every remaining diagnostic (persistence, aging,
  target distribution, and the new feature-diagnostics module) is built to feed
  regression/GLM, GBM, and the deep multi-head net alike — see
  [Serving three model families](#serving-three-model-families).

---

## Conventions to follow

Matching what already exists (there is **no shared config helper** — the idiom is duplicated in 6 places, and this work should not refactor that):

```python
cfg = yaml.safe_load(open("configs/default.yaml"))          # __main__ blocks
def run(cfg_path: str = "configs/default.yaml") -> ...:      # callable API
```

- Entry points run as `python -m src.<module>` from repo root; add matching `Makefile` targets next to `features:` and list them in `.PHONY`.
- Reuse `src/data/fetch.py::_slug` / `_season_start_year` for season-key handling, and `preprocess.load_raw`'s reverse (`slug.replace("_", "-")`) so season keys are `"2023-24"` and join to `game_logs.parquet`.
- Reuse `src/data/preprocess.py::compute_dk_pts` verbatim — do not reimplement the DraftKings formula.
- `Path(...).mkdir(parents=True, exist_ok=True)` before every write; `print(f"... {n:,} ... → {dest}")` progress lines.
- Artifact save/load mirrors `src/features/encode.py::save_artifacts` / `load_artifacts`.

## Dependencies to add

`requirements.txt` needs `streamlit`, `plotly`, and `pyarrow` (parquet writes; not currently installed). Install with `.venv/bin/pip install -r requirements.txt`.

---

## Implementation

### 0. Repair the fetch layer and backfill the defense family

Must land first — the season matrix should be built over complete data, not around a hole.

In `src/data/fetch.py`:

- **Fix `_skip_or_fetch`** to treat a file with a header but zero data rows as *not* fetched, so corrupt files self-heal on re-run instead of being skipped forever. Cheap check: `dest.stat().st_size` plus a one-line read, avoiding a full parse of 284 MB of CSVs.
- **Add per_mode fallback to `fetch_player_stats`.** On an empty (0-row) response, retry the same query across `["PerGame", "Totals", "Per100Possessions", "PerMinute"]` until one returns rows, then record which mode won. Backoff is *not* the mechanism — varying the cache key is — so retries can be immediate with a small courtesy delay.
- **Warn loudly on an empty result.** The current `_save` prints a contented `Saved 0 rows`; that is how this slipped through. It should print a `WARNING` when `len(df) == 0`.
- **Remove `"Four Factors"` from `PLAYER_STAT_MEASURES`** (verified unsupported for players under every per_mode; team four-factors is unaffected and stays).
- **Write a backfill manifest** to `data/raw/_fetch_manifest.csv` recording `(family, season, per_mode, rows)` so the season-matrix loader knows the basis of each file and can normalize correctly.

Then run the backfill for the six seasons — 2000-01, 2002-03, 2004-05, 2005-06, 2015-16, 2023-24 — writing real files into `data/raw/`. Expected rows: 441 / 428 / 464 / 458 / 476 / 572, already verified against the live API.

### 1. `src/eda/season_matrix.py` — the foundation everything else consumes ✅ done

Builds one row per `(player_id, season)`.

- **Loader** with a family registry: `(filename_prefix, column_prefix, first_year, header_rows)`. Drops `*_RANK`, `TEAM_COUNT`, `NICKNAME`, `GROUP_SET`. Prefixes columns per family (`adv_`, `usg_`, `sco_`, `misc_`, `clu_`, `sl_`, `drv_`, `pass_`, `hus_`, …) so nothing collides — several families repeat `GP`/`MIN`/`FGM`.
- **Shot locations** read with `header=[0,1]` and flattened to `sl_restricted_area_fga` style names.
- **Keep the `player_stats_defense` family** (repaired in Stage 0, so all 30 seasons are available). `DEF_WS` / `DEF_WS_RAW` are unique to it. Because backfilled seasons may arrive in a different `per_mode`, the loader normalizes counting stats to per-36 using the recorded mode from the backfill manifest: `Totals → /(MIN*GP)*36`, `PerGame → /MIN*36`, `PerMinute → *36`. Rate columns (`DEF_RATING`, `*_PCT`) are mode-invariant and pass through.
- **Rate normalization:** counting stats → per-36 (`stat / MIN * 36`, valid since both are per-game). Percentages, frequencies, and `*_PCT` shares pass through untouched. Keep `MIN`, `GP`, and `MIN*GP` as explicit volume columns held *out* of the PCA — otherwise PC1 collapses to "minutes played" and tells us nothing.
- **Target join:** aggregate raw `game_logs_*.csv` per `(player_id, season)` using `compute_dk_pts`, attaching `dk_pts_total`, `dk_pts_per_game`, `dk_pts_std`, `games_played`. Done directly from raw CSVs rather than `data/processed/game_logs.parquet` — that directory is currently empty, so this must not depend on `preprocess` having been run.
- **Qualification filter:** `GP >= 20 and MIN >= 10`, configurable. Unqualified players produce wild per-36 outliers (a 12-minute season with 2 threes).
- **Outputs:** `data/features/season_matrix_tierA.parquet`, `season_matrix_tierB.parquet`, plus `data/features/coverage_report.csv` (season × family availability, feeding the dashboard's data-quality tab).
- **Added since:** an unfiltered twin per tier, `season_matrix_roster_tier{A,B}.parquet` (14,569 / 6,942 rows vs 10,900 / 5,077 qualified). Same columns, every player who took the floor. The qualification filter serves the PCA; roster aggregation needs everyone and down-weights thin seasons instead. See finding 4 above.

### 2. `src/eda/pca.py` ✅ done

- Two standardization modes: `within_season` (z-score inside each season — era-neutral) and `pooled` (one z-score across all seasons — era becomes a visible axis).
- Missing values imputed to the within-season median before fitting.
- Fit `sklearn.decomposition.PCA`, retain components to ~90% cumulative variance.
- **Outputs per (tier × mode):** `pca_{tier}_{mode}_scores.parquet` (PC scores keyed by player_id/season, plus passthrough identity + volume columns), `_loadings.parquet`, `_variance.csv`, and pickled `PCA` + scaler objects via the `encode.py` artifact pattern.

### 3. `src/eda/archetypes.py` — the bridge to the team-composition goal ✅ done

- k-means and GMM over the leading ~10 era-adjusted PCs; sweep `k = 4..14` reporting silhouette and BIC so the choice is evidence-based rather than asserted.
- Auto-name each cluster from its highest-|z| stats (e.g. "high-usage rim pressure", "low-usage 3&D wing"), with names stored in a small editable mapping rather than hardcoded in logic.
- **`team_composition_{tier}.parquet`:** for each `(team, season)`, the minutes-weighted distribution over archetypes, plus roster-level aggregates (mean height/age, pace, usage concentration via HHI). This is the artifact the eventual model consumes for both own-team and opponent-team inputs.
- **Added since:** the fitted KMeans/GMM are persisted (`kmeans_tier*.pkl`, `gmm_tier*.pkl`, `cluster_meta_tier*.pkl`) and `assign_archetypes()` places arbitrary player-seasons — including sub-threshold ones — in the same space, with optional reliability shrinkage applied in z-space before projection.

### 4. `src/eda/persistence.py` — which prior-season stats actually carry signal, for any model family ⬜ planned

The most directly load-bearing analysis for the modeling goal — it decides the candidate
feature list for every family the modeling stage will try (GLM, GBM, and the multi-head
net alike).

- For every numeric feature, correlate season *t* against season *t+1* for the same player; report `r`, `n_pairs`, ranked.
- **Absorb season, and report both.** A pooled correlation across 30 seasons conflates persistence with era drift (finding 1). Report the pooled and within-season figures side by side; where they diverge sharply, the pooled one is measuring the calendar.
- **Weight by minutes** for any per-36 column (finding 2), otherwise the ranking is dominated by garbage-time noise rather than by which stats are stable.
- Reliability as a function of games played (split-half correlation by GP bucket), to find the minimum sample where a stat stabilizes. **Partly done already**: on eight style stats over 11,272 consecutive-season pairs, reliability fits `0.924 · m/(m+66)` in total minutes, and lag-*k* persistence runs 0.921 / 0.881 / 0.853 / 0.826 at lags 1–4. Generalize that to every column rather than re-deriving it.
- Expected outcome: usage/rebound/assist rates are sticky, `FG3_PCT` and `PLUS_MINUS` are
  noise. That directly determines which features are worth giving any of the three model
  families — a GLM's feature list, a GBM's candidate columns, and an NN's input width are
  all cut from the same ranking.
- Output: `outputs/eda/persistence.csv`.

### 5. `src/eda/aging.py` ⬜ planned

- **Delta method** (mean within-player change between consecutive seasons, bucketed by age) to avoid the survivorship bias that corrupts naive age curves — weak players leave the league, so raw cross-sectional curves overstate late-career performance.
- Curves for per-36 `dk_pts` and its components, overall and per archetype.
- Consumer already waiting: `team_context.py` decays returnees by a flat 0.96/season. A real age-adjusted decay replaces that constant.
- **Consumption differs by family**, worth keeping the output shape general enough for
  all three: a GLM wants the curve as a fitted spline term, a GBM just wants raw age as a
  column (it finds the shape itself), and the NN wants it as a feature or small embedding
  alongside archetype. Don't commit the output format to one of these.
- Output: `outputs/eda/aging_curves.csv`.

### 6. `src/eda/target.py` ⬜ planned

- Per-game `dk_pts` **and each component's** distribution: skew, zero-inflation from
  DNP/garbage-time games, and **variance-vs-mean by minutes and usage bucket** — run per
  component (`pts`, `reb`, `ast`, ...), not just on the aggregate. The three model
  families each read this differently: it picks the GLM family (Poisson vs. NegBin vs.
  Tweedie vs. log-normal) per component, the GBM objective per component, and it
  validates — or is the evidence to change — `multihead.py`'s existing Poisson-with-
  minutes-offset choice. One measurement, three decisions.
- Running season total: cumulative trajectories, and how well the first *k* games predict the final total (the "running season total" half of the goal).
- Output: `outputs/eda/target_profile.csv`.

### 7. `src/eda/feature_diagnostics.py` — failure modes specific to each future model family ⬜ planned

Doesn't rank features by signal (that's `persistence.py`); checks for the three ways
picking a model family can go wrong that the rest of the plan doesn't cover.

- **Collinearity.** Correlation matrix, VIF, and condition number over the raw
  within-season z-scored feature set (~150 Tier A / ~287 Tier B columns). Several
  families report near-duplicate `GP`/`MIN` definitions — flag clusters so a future GLM
  can choose raw-plus-regularization vs. a reduced set with evidence instead of by
  assumption.
- **Cardinality / effective-n per categorical** (`team_id`, archetype label,
  draft-slot bucket): category counts and minimum per-category *n*, flagging cold-start
  entities (team relocations — SEA→OKC, NJN→BKN — and expansion teams). Decides
  target-encoding vs. one-hot for a GBM and embedding-table size for an NN.
- **A reusable shuffled-null importance helper**, generalizing the permutation pattern
  already built once for opponent (`opponent.py::variance_ceiling`, and finding 3 in
  "What changed") into a shared function any future GBM/NN importance ranking can call —
  so an importance score always ships with its own chance level instead of being read
  against a fixed threshold.
- **Sequence-vs-aggregate ablation.** Does the raw prior-season game-log sequence carry
  anything (order, within-season trend, autocorrelation) beyond what
  `season_matrix.py`'s per-season aggregates already contain? Cheap to check now, and it
  decides whether the NN's LSTM trunk over game logs is earning its complexity or just
  re-deriving a season mean at a much higher cost than reading it off the aggregate.
- Output: `outputs/eda/feature_diagnostics.csv` (collinearity + cardinality reports), plus
  the importance-null helper importable from `src/eda/feature_diagnostics.py` for reuse
  wherever a future model reports feature importance.

### 8. `dashboard/app.py` — Streamlit explorer ⬜ planned

Reads the precomputed parquet artifacts (fast startup, no refitting on load) with a `ROOT` `sys.path` bootstrap matching the notebook convention. Tabs:

1. **Coverage** — season × family availability heatmap, surfacing the defense-family holes and tracking-era boundary. Now also worth showing `roster_coverage` per team-season: the share of roster weight that is observed rather than imputed.
2. **PCA** — scree plot, interactive biplot with player-name hover, colorable by archetype / era / team / age, loadings table, and a single-player trajectory through PC space across their career. Tier and era-mode are sidebar toggles.
3. **Archetypes** — cluster profile bars, cluster rosters, team-composition stacked bars by season.
4. **Persistence** — the ranked YoY table plus a *t* vs *t+1* scatter for any selected stat. Show pooled and within-season *r* together (finding 1).
5. **Aging** — age curves, filterable by archetype.
6. **Target** — `dk_pts` distributions and heteroscedasticity plots.

Three tabs worth adding, for artifacts built since this plan was written:

7. **Team context** — the `context_value` table (partial *r* per feature × outcome, ΔR²), and per-team-season roster composition by `stats_source`.
8. **Opponent** — the five defensive-profile PCs with their loadings, teams plotted in that space by season, and the held-out matchup table.
9. **Feature diagnostics** — collinearity heatmap, cardinality/cold-start table, and the
   sequence-vs-aggregate ablation result. The one tab that isn't about a finding but about
   which future model families can safely use which inputs.

Load the `dataviz` skill before writing any chart code so the visuals are consistent and readable in both light and dark themes.

### 9. Config and wiring ✅ done

New top-level `eda:` block in `configs/default.yaml` (min GP/minutes, tier definitions, PCA variance target, cluster-k range, output dirs). New `Makefile` targets: `season-matrix`, `pca`, `archetypes`, `eda` (runs the full sweep), `dashboard` (`.venv/bin/streamlit run dashboard/app.py`), all added to `.PHONY`.

Since extended with `team-context`, `context-value`, `opponent`, `component-targets`,
`persistence`, `aging`, `target-profile`, `feature-diagnostics` and `dashboard`, plus
`features.team_context` / `features.opponent` and `eda.{persistence,aging,target,
feature_diagnostics}` config blocks. All targets are in `.PHONY`; `eda` runs the full
sweep in dependency order.

---

## Work completed outside this plan

Built while stages 3–4 were pending, because the modeling design needed them. Specified in
README.md, listed here so this file is not read as the whole picture:

| Module | What it is |
|---|---|
| `src/features/targets.py` | Component targets and the calibrated expected bonus |
| `src/features/team_context.py` | LOO own-team encoder on roster(S) × stats(S-1) |
| `src/features/opponent.py` | Team defensive profile + low-rank bilinear matchup |
| `src/eda/context_value.py` | What own-team context is worth, above the player's own prior season |
| `src/models/multihead.py` | Availability × per-36 rate heads with a differentiable bonus |

---

## Sequencing

Staged so there is something inspectable early rather than one big drop:

0. ✅ **Stage 0** — repair `fetch.py` and backfill the six defense seasons. Everything downstream reads this data.
1. ✅ **Stage 1** — `season_matrix.py` + tests + coverage report. Nothing downstream is trustworthy until the join and per-36 logic are verified.
2. ✅ **Stage 2** — `pca.py` + `archetypes.py`, with printed variance/loading summaries for a sanity read.
3. ✅ **Stage 3** — Streamlit dashboard, all nine tabs, over the precomputed artifacts.
4. ✅ **Stage 4** — `persistence.py`, `aging.py`, `target.py`, `feature_diagnostics.py`, and
   their dashboard tabs (including the new feature-diagnostics tab).

## Verification

Current state: **144 tests pass** (`.venv/bin/pytest tests/`). Stages 0–2 verified as
below; the checks for stages 3–4 stand as written.

- **Stage 0 checks** (before anything else):
  - All 30 `player_stats_defense_*.csv` files exist and every one has >400 data rows.
  - `_skip_or_fetch` returns `False` for a synthetic header-only file and `True` for one with rows — the regression that let this persist.
  - `data/raw/_fetch_manifest.csv` records a `per_mode` for each backfilled season.
- **Tests** in `tests/test_season_matrix.py`, matching the existing plain-`assert`, synthetic-builder style (no fixtures/classes, mirroring `tests/test_preprocess.py`):
  - one row per `(player_id, season)`; no `*_RANK` / `TEAM_COUNT` columns survive.
  - per-36 conversion is arithmetically correct on a hand-checked row.
  - shot-location MultiIndex flattening produces the expected column names.
  - Tier A spans 30 seasons and Tier B exactly 13; Tier B is a column superset of Tier A.
  - qualification filter excludes a synthetic 5-GP player.
  - PCA round-trip: `inverse_transform(transform(X))` reconstructs within tolerance at full rank.
- **End-to-end:** `make eda` from a clean `data/features/`, then confirm each artifact exists with expected row counts (~12k Tier A, ~5.5k Tier B).
- **Domain sanity check on the archetypes** — the strongest signal that the pipeline is correct. Known centers (Jokić, Embiid) must not land in a guard cluster; known 3&D wings should group together. Print a labeled sample per cluster for inspection rather than asserting correctness silently.
- **Dashboard:** `make dashboard`, confirm all nine tabs render and the tier/era toggles switch the underlying artifact.
- **Regression:** `.venv/bin/pytest tests/` still passes — this work is purely additive and touches no existing module.
- **Stage 4 specifically:** `persistence.py` must report pooled *and* within-season correlations. If a stat's pooled *r* is far above its within-season *r*, that stat is tracking the era, and saying so is the useful output. A ranking built on pooled *r* alone would put the 3-point columns at the top for the wrong reason.
- **`feature_diagnostics.py` specifically:** the shuffled-null importance helper must
  reproduce the opponent finding it generalizes from — feeding it the same
  opponent-within-season permutation should recover ~+0.97% (finding 3, "What changed").
  If it doesn't, the generalized version disagrees with the one-off it was built from.
