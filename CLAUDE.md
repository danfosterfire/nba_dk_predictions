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
src/data/      fetch, preprocess, dataset
src/features/  rolling stats, matchup context, team context, targets, encoding
src/eda/       season_matrix, pca, archetypes, context_value, persistence, aging,
               target, feature_diagnostics  (season-level analysis pipeline)
src/models/    lstm, transformer, multihead, xgboost baseline
src/train.py   training loop
src/evaluate.py test-split metrics
src/predict.py  inference entry point
configs/       default.yaml — all hyperparams and paths
data/          raw → processed → features pipeline
outputs/       checkpoints, prediction CSVs, eda reports
dashboard/     Streamlit explorer over the precomputed artifacts
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
make eda              # all of the above, in dependency order
make dashboard        # Streamlit explorer (9 tabs over the precomputed artifacts)
```

The dashboard reads artifacts only — it never refits. `.streamlit/config.toml` sets
`headless = true`, without which Streamlit's first-run email prompt makes
`make dashboard` exit 255 instead of serving.

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

---

## Established facts — do not re-derive

Measured on 254,187 player-games (2014-15 → 2023-24). Within-player-season residual sd is
**9.4 dk_pts**.

| Source | Share of variance |
|---|---|
| Player-season identity (of total per-game variance) | 58.0% |
| Own minutes played (of within-player residual) | 18.6% — *unknowable in advance* |
| Opponent × season | 0.69% |
| Opponent × archetype × season, above a shuffled null | +0.31% to +0.97% |
| Home / away | 0.03% |

The opponent rows are **in-sample ANOVAs on contemporaneous opponent identity** — ceilings,
not achievable gains. Reproduce with `src/features/opponent.py::variance_ceiling`.

**~90% of attainable skill is the season-level rate; ~1–2% is per-game modulation.** Budget
effort accordingly, and do not expect team composition to carry the model.

- **Predict components, not dk_pts directly.** Effects cancel *across components* in the DK
  sum. Opponent: per-component weighted opponent sd sums to 1.105 vs 0.785 measured on
  dk_pts directly. Team context is worse — `teammate_assist_supply` moves ast/36 −0.361,
  reb/36 +0.218, blk/36 +0.191 per sd, which is 2.11 dk_pts gross against 0.25 net, an
  **8.3× cancellation** (`role_crowding` 8.2×, `teammate_spacing` 5.0×, `team_pace` 4.7×).
  The double-double bonus is a threshold on five components, so `E[bonus] ≠ bonus(E[x])` —
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
- **15.9% of roster minutes have no usable S-1 row** — 9.4% true rookies, 5.4%
  sub-threshold, 1.1% returnees; p90 = 32.9% per team-season, max 54.1%. Never drop them.
  `season_matrix_roster_tier*.parquet` is the unfiltered twin built for this
  (14,569 / 6,942 rows vs 10,900 / 5,077); rows carry `stats_source` and `reliability`,
  team-seasons carry `roster_coverage`.
- **Prior-season reliability is `0.924 · m/(m+66)`** in total minutes, fitted on 11,272
  consecutive-season pairs. Applied as a multiplier in z-space, which *is* shrinkage to the
  league mean. Staleness costs a further ×0.96 per season (lag-1…4: 0.921/0.881/0.853/0.826).
  Rates stabilize fast — the 200-minute qualification threshold is already at 0.75.
- **Bonus overdispersion is calibrated to 0.10** against 11,627 player-seasons. Independent
  sampling is 23% too low. Do not change it without re-running that calibration.

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
- **Games played is the least persistent quantity in the project — r = 0.316.**
  Season-absorbed and minutes-weighted over the same 11,272 pairs, against `min` per game
  0.779, `dk_pts` per game 0.869, `min_total` 0.640, `dk_pts_total` 0.760. `min`/`gp` are
  held *out* of `persistence.csv` as volume columns; reproduce with
  `lagged_pairs`/`pair_weights`/`demean_within` from `src/eda/persistence.py` over
  `season_matrix_roster_tierA.parquet`. Games played is simultaneously the **largest lever
  on the season total and the least predictable input** — shrink the availability head hard
  toward a league/age baseline rather than toward the player's own prior GP.
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
  dk_pts. Separately, log(season total) is 84.5% explained by log(per-game rate) alone and
  73.4% by log(games) alone (games: sd 20.8 on a mean of 56, p10 23, p90 80). This is the
  measured price of the prior-season-only constraint.
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
- `player_shot_locations` has a **two-row MultiIndex header**; every other family is flat.
- Empty `LeagueDashPlayerStats` responses are **poisoned server-side cache entries** keyed
  on the full query, not rate limiting. Backoff does not help; cycling `per_mode_detailed`
  does. `fetch.py` handles this and records the winning mode in `data/raw/_fetch_manifest.csv`.
- Player `"Four Factors"` is genuinely unsupported by the endpoint — its stats live in
  Advanced. Team four-factors works fine.
- Coverage: core families 30 seasons; tracking/`pt_shot` 13 (2013-14+); estimated 12
  (2014-15+); hustle 11 (2015-16+, and **2015-16 covers only 147 players** — a partial
  first-season rollout, not a fetch failure).
- The 7 `team_stats_*` families and `team_estimated_metrics` feed
  `src/features/opponent.py`. `team_stats_opponent`'s `OPP_*` columns are *what opponents
  recorded against this team* — already the per-component opponent profile. Never rebuild
  this by aggregating player rows.
- **Team files break the per-36 rule.** `fetch_team_stats` uses `PerMode="PerGame"` and the
  counting stats are per game, but `MIN` and `POSS` are **season totals** (`MIN` ≈ 3966 =
  48.4 × 82). Normalize per-100 possessions with `POSS / GP`, never `stat / MIN * 36`.
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
