
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
- **⭐ The availability head is a beta-binomial GLM and the simulator is not built — but on
  validation the GBM leads the ladder, and it is the ORDERING that reversed, not the
  decision.** Scored on 2022-23/2023-24 (9,478 train / 883 validation), CRPS in games:
  **GBM 9.876**, ridge 10.004, **GLM 10.006**, league/age 13.387. `src/models/availability.py`,
  `make availability-model`.
  - **⚠️ This was the last held-out measurement in the project and it read GLM 10.795 / GBM
    10.888 / ridge 10.896 / league-age 13.614 on 10,361 train / 911 test until 2026-08-08.**
    (Pre-workload-block: 10.914 / 11.04 / 10.98 / 13.614.) The recorded conclusion —
    "gradient boosting does not beat a 19-feature GLM, so the plan's own decision rule says
    stop" — is **false as stated** on the split that decides.
  - **The decision stands because the reversal is not distinguishable from zero.** ✅
    `make availability-model` (`availability.ladder_comparison` →
    `availability_ladder_comparison.csv`), a paired bootstrap over the same 883 rows against
    the shipped head: GBM **−0.1297** CRPS, 95% CI **[−0.3154, +0.0672]**, P(better) 0.906;
    ridge **−0.0014**, CI [−0.0546, +0.0516]. Only the league/age baseline separates
    (+3.3811, CI [+2.8567, +3.9299]). The ridge's 0.0014 is the same order as the
    **0.0013**-CRPS margin that decided the games-played Gate D, reversed, and wrote
    `src/models/held_out.py`. Treating this one as a verdict would be that mistake with the
    sign flipped.
  - **⭐ Where the GBM wins is the argument against it.** By *realized* games-played
    quartile, the GBM is **+0.333** CRPS worse than the GLM in q1 and −0.063 / −0.452 /
    −0.366 better in q2/q3/q4; the ridge runs **+0.251** / +0.109 / −0.008 / −0.393. Both
    challengers beat the GLM on seasons that went normally and lose on the seasons that fell
    apart — the population the head exists for. Quote the quartile row before the mean.
  - Two things confirm the distribution is the working part: the fitted
    dispersion lands at **20–30× implied overdispersion**, independently recovering the ~20×
    in `availability_profile.csv`, and PIT is near-uniform (KS 0.07–0.11 against the
    baseline's 0.15). Validation R² 0.374 is **not** comparable to the 0.236 ceiling — that
    one is in-sample and season-absorbed. (Retired held-out reading: KS 0.08–0.11 against
    0.17, R² 0.268.)
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
  (`workload_ablation` → `outputs/predictions/availability_workload_ablation.csv`), on
  **validation** (2022-23/2023-24), everything fixed but the feature list:

  | variant | features | CRPS | vs baseline | val R² |
  |---|---|---|---|---|
  | baseline | 15 | 10.104 | — | 0.363 |
  | **+ playoff workload** | **19** | **10.006** | **−0.098** | **0.374** |
  | + playoff only | 18 | 10.015 | −0.089 | 0.373 |
  | + `career_minutes` only | 16 | 10.085 | −0.018 | 0.365 |

  ⚠️ **This decided a feature block on the held-out split until 2026-08-08**, where it read
  10.914 / **10.795** / 10.817 / 10.883 CRPS and 0.268 / **0.283** / 0.281 / 0.271 R² at a
  gain of **−0.119**. **It survives the re-decision unchanged** — same sign, same ordering
  of all four variants, same order of magnitude — which is the contrast worth drawing with
  the model ladder above it: a block worth a tenth of a game does not care which split
  measured it, and a 0.13-game gap between two models does.

  ⚠️ Worth **+6.2 dk_pts of season-total MAE** (441.3 → 435.1; −9.5 more on established
  rotation players, 508.9 → 499.4), with `oracle_gp` unchanged at 221.3 — the internal check
  that only the GP treatment moved. **That trio is measured against the superseded TEST
  season-total table** and is kept because it is the only record of the pre-workload run.
  It has no validation twin and will not get one from the current code: `season_total.py`
  holds the availability head fixed at the shipped feature list, so nothing composes a
  baseline-feature variant through it. In-sample the block is **+0.0178 above its own
  shuffled null** (0.2998 vs 0.2820, sd 0.0002) — in-sample, so untouched by the split.

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
  - ⚠️ **The recorded "the GBM's margin narrowed but the ordering held" is withdrawn.** On
    test the block helped the GBM slightly more than the GLM (−0.152 vs −0.119), leaving
    GLM 10.795 against GBM 10.888 from 10.914/11.04 before. On validation the margin it was
    tracking has crossed zero — see the ladder bullet above. The stopping rule still says
    stop, but for a different reason: nothing beats the GLM *distinguishably*.
- **Nonlinear terms are a null for games played and NOT a null for minutes — and the split
  between those two is the useful part.** `make availability-model`
  (`nonlinearity_ablation` + `minutes_nonlinearity_probe` →
  `outputs/predictions/availability_{nonlinearity,minutes_nonlinearity}.csv`). Cubic
  B-splines with linear extrapolation (knots from **training** quantiles only) or
  quadratics, over the nine continuous columns; `age_sq` is already in the linear baseline.

  | variant | p | val CRPS | val R² | selected |
  |---|---|---|---|---|
  | **linear** | 19 | **10.006** | **0.374** | **✓** |
  | quadratic | 27 | 10.037 | 0.370 | |
  | spline k=4 | 54 | 10.041 | 0.367 | |
  | spline k=5 | 63 | 10.054 | 0.366 | |

  - **⭐ Every validation figure here reproduced to five decimals across the 2026-08-08
    conversion, and that is a DETERMINISM check.** `held_out.selection_split` hands back
    exactly the frames this ablation's own private inner split used to carve, so dropping
    the test side could not move them — it proves the conversion changed nothing it should
    not have, and is not independent evidence for the verdict.
  - **⚠️ The retired test column preferred every curved variant** — 10.795 (linear) against
    **10.749** / **10.761** / **10.751** — and this is the methodological trap worth
    remembering: a *paired* bootstrap on the 911 test rows put the quadratic gain at
    **−0.047, 95% CI [−0.079, −0.015], P(Δ<0) = 99.7%**, and it was still a false positive,
    because a paired interval says a difference is consistent *within one sample*, not that
    the sample was representative. The GBM arm corroborates the null from a different
    direction: a fully nonparametric learner on the same features does not beat the linear
    GLM by a distinguishable margin either.
- **For minutes per game curvature is real, and it is *not* the age arc.** Ridge
  probe (the minutes head is not built), predicting next-season MPG on validation: linear
  R² **0.6768**, quadratic **0.6914**, spline k=4 **0.6930**. Splining one column at a time
  attributes it:

  | column splined | Δ val R² |
  |---|---|
  | **`minutes_per_game_lag1`** | **+0.0124** |
  | `total_minutes_lag1` | +0.0008 |
  | `career_year` | +0.0007 |
  | `age` | **−0.0008** |
  | `playoff_minutes_share_lag1` | −0.0016 |

  - **⚠️ "The same test replicates" is WITHDRAWN — the test column is gone and it was never
    a replication.** It read 0.6670 / **0.6746** / 0.6736 by variant and +0.0077 /
    +0.0011 / −0.0006 / −0.0007 by column, with a `replicates` flag requiring both columns
    to move the same way. Those two columns differed in *training data* as well as in scored
    rows — the confound `src/models/held_out.py` exists to stop being read as agreement — so
    the flag was asserting something it could not see. The validation column is unchanged to
    four decimals. **What actually carries the finding is the contrast between the two
    TARGETS on one frame**: the identical experiment, same rows, same code, is a null for
    games played and is not for minutes per game.

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
