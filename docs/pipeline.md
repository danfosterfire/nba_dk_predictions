
# Pipeline (run in order)

```bash
python -m src.data.fetch           # pull raw game logs from nba_api
python -m src.data.preprocess      # clean + add season column → data/processed/game_logs.parquet
python -m src.train                # train model → outputs/checkpoints/best_model.pt
                                   #   also saves scaler/feature_cols to data/features/
python -m src.evaluate             # test MAE/RMSE → outputs/predictions/test_metrics.csv
python -m src.predict              # inference → outputs/predictions/next_season_predictions.csv
```

`src/features/encode.py` (rolling aggregate features) is retained for experimentation but is no longer part of the primary training pipeline.

## Season-level EDA pipeline

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
make preseason        # → preseason.parquet + outputs/eda/preseason_coverage.csv
                      #    current-season preseason games as FORECAST COVARIATES — the
                      #    only current-season observation the project may read, and never
                      #    a target row. Its logs are backfillable, so this stays OUT of
                      #    `make daily-capture`; one fetch after the final preseason game
                      #    is enough. See docs/preseason-plan.md
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
make dashboard        # data visualizations over the precomputed artifacts
                      #    (today: the PCA player-style fingerprint)
make dashboard-config # .streamlit/config.toml regenerated from dashboard/theme.THEMES, so
                      #    the page chrome and the chart surfaces are one palette; a test
                      #    parses the checked-in file back and fails if they have drifted
make dashboard-audit  # registry drift report — a report, not a gate; exits 0 with findings
make docs-audit       # every quoted figure in the plan docs vs its artifact — a GATE
```

### Availability data capture

```bash
make daily-capture     # injury-reports + injuries + capture-calendar — MUST be on a
                       #    cron; see the Makefile
make capture-calendar  # → outputs/eda/capture_{calendar,programs}.csv. The same coverage
                       #    `capture-status` and `adp-status` print, as an artifact the
                       #    dashboard's page 7 draws. Reads disk only, makes no requests —
                       #    a printout cannot be drawn, diffed or checked by anything
make boxscore-status   # 2006-07 → 2025-26 inactive lists + DNP reasons, ~8-14 h, resumable
make availability-model  # baselines + CRPS/PIT → outputs/predictions/availability_*.csv
make season-total      # composes gp × rate → outputs/predictions/season_total_*.csv
                       #   what the availability head is worth on the actual deliverable
make component-rates   # → component_rate_metrics.csv
                       #   the 11 component heads vs the mandatory no-fit floor
```


**`daily-capture` is the only thing in this repo with a deadline.** Both its sources are
current-status feeds that cannot be backfilled: the NBA injury-report PDFs age out of the
CDN after ~7 months (a 403 thereafter, forever) and the ESPN feed has no history at all.
A day the cron does not run is a day permanently lost. `make injury-reports` is idempotent
and re-parsing is offline (`--reparse`), because the archived PDFs — not the parsed CSVs —
are the artifact worth keeping.

### Stan heads

```bash
make stan-availability # port of the point-MLE beta-binomial + the posterior it buys
make stan-availability-mixture
                       # the LOW-AVAILABILITY MIXTURE's port check — the same head with
                       #   a second component for the disrupted season, scored against
                       #   the point-MLE arm `make availability-window` selected. Its own
                       #   artifacts, deliberately: `stan-availability` writes the figures
                       #   `make docs-audit` pins, and a port check must not move them
make stan-minutes      # min | available, trials = real game length (NEVER 48)
make stan-components   # 8 NB count heads + 3 beta-binomial conversion heads
make stan-composition  # the team-game minutes COMPOSITION — the per-game allocation
                       #   (zero-sum + the cap); see docs/minutes-composition-plan.md
make stan-game-length  # does a game go to OVERTIME, and how deep. The one thing a
                       #   forward simulation cannot look up: both minutes heads need a
                       #   length, and in a replay it comes from the parquet. Two
                       #   existing .stan sources, ~30 collapsed rows, 0.3 s of sampler
                       #   — the cheapest head in the project, which is why it runs
                       #   first in the aggregate. Replaced stan_composition.fit_ot_tail.
make stan              # all five, cheapest first so a plumbing failure surfaces in
                       #   seconds, and composition AFTER stan-minutes, which it imports
                       #   from and measures itself against
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
make posteriors        # PERSIST the fits: 20 heads refitted once at the variant their
                       #   own sweep selected, each writing thinned draws + the design
                       #   recipe + provenance to
                       #   data/features/posteriors/<window>/<head>.pkl.
                       #   `make stan` throws its coefficient draws away, so without
                       #   this the simulation layer has to refit to draw anything.
                       #   WINDOW=train by default — the backtest scores the validation
                       #   seasons, which train_val has already read; `--window full`
                       #   fits on the held-out seasons and is guarded. Budget most of a
                       #   day, and use `--groups <family>` to redo one family. Not in
                       #   `stan`: it consumes those artifacts rather than being one.
```

### The simulation layer

Everything here is numpy over the posterior pickles. Only `make posteriors` above needs a
sampler.

```bash
make scoring-periods   # one row per (season, game_id): its scoring period and its DK
                       #   tournament round → data/features/scoring_periods.parquet.
                       #   A best-ball lineup is scored WEEKLY, so every weekly max,
                       #   round total and advancement cut downstream aggregates over a
                       #   period — this is the only module that says what one is.
                       #   DK's periods are NBA weeks: ScheduleLeagueV2 carries
                       #   `weekNumber` from 2017-18, and the 21 older seasons get a
                       #   derivation that reproduces it on 10,749 of 10,749 games.
                       #   Owns three edge cases once — a postponed game scores in the
                       #   period PLAYED, the NBA Cup final scores nowhere, and the
                       #   all-star gap moves no Monday. Schedules cache to data/raw, so
                       #   a rebuild needs no network; REFRESH=1 re-pulls them.

make draft-pool        # the board: one row per (season, player) with team, DK position
                       #   eligibility, ADP and the prior-season key the heads score him
                       #   from → data/features/draft_pool.parquet. 13,105 player-seasons
                       #   over 31 seasons, 942 of them the 2026-27 production board.
                       #   IT REVERSED A PLAN ASSUMPTION: DK is SINGLE-position — both
                       #   boards print exactly one of G/F/C for all 1,640 rows, zero
                       #   duals — so NBA.com's `G-F` duals are not DK-shaped. Validated
                       #   on the persistent DK id, DK's letter equals NBA.com's primary
                       #   on 86.63% of players and lies inside its set on 92.61%
                       #   → outputs/eda/draft_pool_position_audit.csv. Membership is
                       #   season-start rosters from the game logs, never the roster CSV,
                       #   which is a current-status snapshot carrying February signings.

make minutes-unification
                       # the two minutes heads scored against each other at the SEASON
                       #   unit, on the 742 validation player-seasons both cover
                       #   → outputs/predictions/minutes_unification.csv. Settles whether
                       #   the composition supersedes the marginal head; it does NOT, so
                       #   both ship. The mean is a tie (MAE 200.28 against 200.12) and
                       #   the SPREAD is not: summed composition draws are 4.68x too
                       #   narrow at the season unit, and the team constraint forbids
                       #   fixing it inside the head — a team's season minutes have a
                       #   predictive sd of 0.00 across draws. REFITS NOTHING: it
                       #   rehydrates each head around its persisted `make posteriors`
                       #   draws and calls the head's own predict path, so it costs
                       #   seconds against the composition's 9.92 h and needs no CmdStan.
                       #   It also runs the same injection grid on TRAINING rows, which is
                       #   what makes the injection shippable: sigma_train = 0.450 against
                       #   the validation grid's 0.375.

make minutes-window    # the marginal minutes head's fitting window x dispersion ladder,
                       #   the same question `make availability-window` asked one head
                       #   over → outputs/predictions/minutes_window_era.csv,
                       #   minutes_window_break.csv, minutes_window.csv,
                       #   minutes_window_rolling.csv, minutes_window_stake.csv.
                       #   Point MLE for the ladder and a rehydrated posterior for the
                       #   stake, so no CmdStan and no refit of either minutes head.
                       #   THREE FINDINGS. It rebuilds availability-window-plan §6's era
                       #   series through each head's OWN design rows, which corrects the
                       #   sd contraction from -15.2% to -9.0%, moves the break from
                       #   2014-15 to 2010-11, and finds the contraction has REVERTED
                       #   since 2019-20. The ladder's window wins on validation and does
                       #   NOT replicate on a 13-origin rolling harness (-0.079
                       #   [-0.59, +0.43]) while role-graded rho does, 13 of 13 origins at
                       #   a 3.03x spread — the largest in the project. And the stake is a
                       #   NULL: sim.minutes.player_season_sigma = 0.450 stands, because a
                       #   short window makes the marginal head a HARDER reference to tie
                       #   rather than an easier one.

make availability-regime # the last two open axes on the availability head's fitting rule
                       #   → outputs/predictions/availability_{regime,shrinkage,
                       #   regime_confirmation}.csv. (1) an explicit COVID-regime indicator
                       #   or exclusion, swept against lookback; (2) shrinkage toward the
                       #   long-window fit. Point MLE for selection, ~6 min, no CmdStan.
                       #   BOTH ARE NULLS, and the round's reusable part is the
                       #   INSTRUMENT: the walk-forward harness is blind to a regime that
                       #   sits in its last three fitting seasons — no arm is active at
                       #   more than 2 of 13 origins, and those two are the trough seasons
                       #   themselves, so it ranks the arms backwards. The fix is a
                       #   CONTAMINATION harness that injects the block into ten ordinary
                       #   origins, with a same-size ordinary block as the control. It also
                       #   rebuilds availability-window-plan §5b's spliced arm and
                       #   WITHDRAWS that round's diagnosis of it: fitted jointly, the same
                       #   coefficient partition does not beat the transplant.

make availability-exchangeability
                       # the head's TRIALS assumption — availability-window-plan §11 →
                       #   outputs/predictions/availability_{clustering,exchangeability}.
                       #   csv. numpy only, seconds, NOTHING IS FITTED, and that is the
                       #   finding: `gp` is invariant to the arrangement, so C and rho
                       #   enter its variance only through C + rho*(n - C) and no
                       #   likelihood over gp can separate them. Five already-fitted
                       #   non-exchangeable arms agree; `hybrid` reproduces CRPS, PIT and
                       #   the tail error to every decimal. So the instrument is a ladder
                       #   at the SCORING PERIOD holding gp fixed at its realized value:
                       #   observed vs `allocate_spells` (ships) vs uniform placement
                       #   (what the beta-binomial asserts). The assumption understates a
                       #   star's P(3 consecutive dead periods) by 9.1x and the shipped
                       #   layout already pays 63-82% of it. What is left is the TENURE
                       #   half — 44.17% of missed games are edge blocks the layout gives
                       #   neither the right shape nor the right position.

make availability-no-prior
                       # what the players the head has NO ROW FOR realize —
                       #   availability-window-plan §8a → outputs/predictions/
                       #   availability_no_prior.csv. Descriptive, seconds. Built to settle
                       #   a decision that was taken and never implemented, and it
                       #   WITHDRAWS it: the role bucket carries DISPERSION, which spans
                       #   1.1557x across draft buckets, while the population's LEVEL spans
                       #   3.3260x. Applied as specified the rule would hand a lottery
                       #   top-5 pick a narrower rho than he realizes. `role_bins`'
                       #   lowest-bucket fallback stands and is now measured rather than
                       #   assumed.
                       # §8b is the LADDER on the axis that survived →
                       #   availability_no_design_level.csv. Four pooling KEYS over one
                       #   estimator, so `pooled` reproduces the shipped scalar exactly.
                       #   `tenure_draft` ships: CRPS 9.8689 against the incumbent's
                       #   14.4551 on validation, and it beats the runner-up too, because
                       #   a draft bucket is a 3.17x gradient for a FIRST appearance and a
                       #   near-flat for a RETURN. Consumed by
                       #   sim.availability.no_design_level.

make weekly-scores     # Gate A at the unit a LINEUP is set at: observed against
                       #   simulated dk_pts per player per scoring period, on train and
                       #   validation → outputs/predictions/weekly_score_{index,period,
                       #   ecdf,calibration,quantile}.csv + weekly_score_sample.parquet,
                       #   and page 8 of the dashboard. `make simulate-season` scores the
                       #   season total, the games-played pmf, the per-game bonus rate and
                       #   the minutes spread — nothing scored dk_pts at the week, which
                       #   is where DK seats the best 7 of 16. NO re-simulation: the
                       #   tensor's second axis already IS the scoring period, so the whole
                       #   target is a reduction plus `make model-cards`' binning helpers
                       #   by import, in ~2 s. Reads four tensors: the two validation
                       #   seasons plus 2018-19 and 2021-22, which are the last two
                       #   TRAINING seasons carrying DK's whole four-round structure —
                       #   2020-21 has no Round 4 at all and 2019-20's is the bubble, and
                       #   a season with an empty slot is REFUSED rather than scored as
                       #   zeros. Build the training pair first with
                       #   `python -m src.sim.season --season 2018-19 --season 2021-22`
                       #   (~78 s and ~80 MB each). `--draws` moves the simulated-season
                       #   budget and `--no-write` gates without writing.

make composition-effects
                       # item 3d — the per-(player, season) random effect fitted as
                       #   `sigma_u` in composition_glm.stan, and a team-context block
                       #   swept alongside it. Four arms (`base` `ps` `ps_team` `team`)
                       #   at the PILOT window
                       #   → outputs/predictions/composition_effects_{metrics,season,
                       #   deviation,diagnostics}.csv. NEEDS CmdStan and is EXPENSIVE:
                       #   `dense_e` is not viable at 12,307 player-season units so the
                       #   random-effect arms drop to `diag_e`, and Gate A probes the `ps`
                       #   arm rather than a plain one for exactly that reason. Writes its
                       #   own artifacts rather than stan_composition_*.csv, which is the
                       #   incumbent's record and is quoted by `make docs-audit`. The
                       #   deviation table lands BEFORE any sampling, so an aborted run
                       #   still leaves it.

make mixture-value     # what the availability mixture is worth in the CONTEST, as a
                       #   paired counterfactual
                       #   → outputs/predictions/availability_mixture_contest.csv.
                       #   REPORTS two arms; it does not run them. Running them is two
                       #   passes over five targets differing in one config key
                       #   (`stan.availability.mixture`), ~1 h each because only
                       #   `--groups availability` of `make posteriors` is refitted, with
                       #   `python -m src.sim.mixture_value --capture {single,mixture}`
                       #   freezing each pass into outputs/predictions/mixture_arms/.
                       #   `--capture` REFUSES when the config key and the arm name
                       #   disagree. Read the `resolution` block before the `contest`
                       #   one: 500 worlds per season resolves a lift gap of 0.0876, and
                       #   the simulated lift is SELF-SCORED — each arm is measured in a
                       #   world it generated, which is why the `adp` control row (a
                       #   board identical across arms) is what the null rests on.
```

**After `make posteriors`, nothing else in the simulation layer needs CmdStan.** That is the
point of it — `src/sim/` is numpy over the pickles, and a draft room loading a board cannot
wait on a sampler.

Fitted **separately**, one model per head, because the chain
`availability → min | available → counts | min → makes | attempts` factorizes the joint
posterior exactly when the parameter blocks are distinct. Sources live in `src/stan/`;
cmdstanpy compiles a **copy** into `outputs/stan/` so no binary and no generated `.hpp`
enters the repo. Needs a CmdStan toolchain, which pip does not manage:

```bash
.venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```



### ADP capture

```bash
make adp-draftkings    # ingest manually-downloaded DK boards + the DK ID → player_id map
make adp-fantasypros   # live consensus capture (--backfill for Wayback, --reparse offline)
make adp-panel         # → adp_panel.parquet, point-in-time safe
make adp-profile       # → adp_transfer.parquet + outputs/eda/adp_profile.csv
make adp               # all four, in order
make adp-status        # coverage for both sources, no requests
```

