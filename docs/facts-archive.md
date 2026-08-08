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
