
# Pipeline (run in order)

**Every recipe runs with `PYTHONUNBUFFERED=1`**, set once with `export` at the top of the
`Makefile` rather than per target. Python buffers stdout whenever it is not writing to a
terminal, which under `make` is almost always — so without it the progress lines this project
prints by convention (`f"... {n:,} ... → {dest}"`) appear only when a buffer fills or the
process exits, and a multi-hour target that is working looks exactly like one that is hung.
That is not hypothetical: one session had the component ladder, `make posteriors` and `make
strategy-sweep` all run blind and needed `/usr/bin/sample` to establish the sweep was alive,
on a day the sweep took 53 minutes against a documented 4.8. It is global rather than a list
of the slow targets because unbuffered stdout costs the fast ones nothing, and a list is one
more place to forget a new target. Adopted 2026-08-15, after four sessions of exporting it by
hand.

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
make preseason-value  # → outputs/eda/preseason_value.csv   (needs `make preseason`)
                      #    the P1 gate: redundancy, incremental signal against each head's
                      #    OWN metric, and the missingness census. TRAIN SEASONS ONLY — it
                      #    never materializes validation, because a screen that spends the
                      #    selection split leaves the arm it is screening nothing to select
                      #    on. Every figure is quoted on the season-start-roster population
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

make components-preseason
                       # preseason-plan session 6b: the preseason block as NESTED arms on
                       #   ALL ELEVEN rate heads →
                       #   outputs/predictions/components_preseason{,_rolling,_shrinkage}
                       #   .csv. Point MLE on each head's SHIPPED variant (read from
                       #   stan_component_metrics.csv), so no CmdStan; under five minutes.
                       #   Needs `make preseason`, `make preseason-value` and
                       #   `make stan-components`.
                       #   ELEVEN, NOT SIX. P1's DR2 screen short-listed six and called
                       #   three others actively harmful. The screen's SIGN did not survive
                       #   on any of the three, and two of them (`fta`, `fg2m|fg2a`) clear
                       #   the real bar outright; a fourth excluded head, `fg3a|fga`, is the
                       #   third-largest result in the round. A screen is a filter, never a
                       #   veto.
                       #   SIX OF ELEVEN CLEAR the conjunction on the declared primary —
                       #   `fga`, `fg3a|fga`, `ast`, `reb`, `fta`, `fg2m|fg2a`. On the
                       #   fitting-half-promoted shrunk arm, which is what ships, ALL
                       #   ELEVEN clear the rolling half (9-13 of 13 origins) and SEVEN
                       #   clear validation. No head anywhere in the round has an interval
                       #   clear of zero on the wrong side.
                       #   THREE THINGS TO KNOW. Read against the no-fit floor rather than
                       #   zero, the block is worth MORE than the whole fitted head on
                       #   `reb` (7.25x), `fg3a|fga` (3.77x) and `fga` (1.17x). The volume
                       #   term wants an empirical-Bayes shrink here, not P1's additive
                       #   one, with `k` fitted per head (20 to 320 pseudo-minutes). And
                       #   SEASON-CENTRING LOSES on this family — it is a device for levels,
                       #   and a per-36 rate has already divided the exposure out.
                       #   TEN OF ELEVEN SHIP, adopted 2026-08-15 via `make stan-components`
                       #   (`stan.components.preseason`). Retention under the posterior is
                       #   0.985 median, four heads GROWING. `fg3m|fg3a` is rolled back by
                       #   `stan_components.PRESEASON_EXCLUDE` — the only measured-worse
                       #   result in the whole preseason round, on three agreeing
                       #   instruments. Priced end to end by `make preseason-contest`,
                       #   which now flips FOUR config keys.
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

### The held-out seasons — two targets, and they may only run in this order

```bash
make final-evaluation  # the ONE reading. Refits the shipped spec on train+validation and
                       #   scores test once: `availability`, `games_played`,
                       #   `season_total`, and `chain` — the deliverable, which builds a
                       #   board from the train_val posterior, drafts the SHIPPED strategy
                       #   and replays it against realized box scores. The three heads are
                       #   minutes; the chain is hours and needs
                       #   `make posteriors WINDOW=train_val` first, so run it alone:
                       #     .venv/bin/python -m src.final_evaluation chain
                       #   The artifact merges BY HEAD, so that does not retract the rest.
                       #   The three heads were taken 2026-08-21; the chain is a separate
                       #   run — docs/final-evaluation-plan.md.
make posteriors-production
                       # the production fit: the same 20 heads at the `full` window, for
                       #   the upcoming season's board. Guarded twice — `--production` has
                       #   to be typed AND final_evaluation.csv has to already exist,
                       #   because deploying before measuring leaves no honest measurement
                       #   to take. Budget most of a day.
make production-check  # is the chain ready to price a season it has never seen? Reads
                       #   disk only, costs a second. Two halves: the MODEL half (20 heads
                       #   at `full`, matching the `train` specification) is finishable
                       #   today; the SEASON half cannot be finished early, and a missing
                       #   row there before October is the NBA schedule.
```

### Scoring a season nobody has played

```bash
make rosters           # the upcoming season's rosters, ALWAYS re-fetched — it is the
                       #   forward MEMBERSHIP rule and it changes with every signing up to
                       #   the opener, so unlike every other raw file it must not be
                       #   cached. Teams come from the published schedule when there is no
                       #   game log yet. `make rosters SEASON=2026-27`, ~30 calls.
make forward-rehearsal # build a PLAYED season's design without its game log — from the
                       #   roster snapshot and the schedule — and compare against the
                       #   design as it is built today. Part A is the mechanics and has an
                       #   exact right answer; part B is the population and is a BOUND,
                       #   because a retrospective snapshot is contaminated in a direction
                       #   the file does not record. The runbook's own advice, since the
                       #   real October window is too short to debug a join in. Part D
                       #   compares the composition's forward per-player frame against
                       #   the shipped `head_frame` path the same way. numpy.
make forward-board     # §6h's acceptance test: the forward inputs pushed through the
                       #   SIMULATOR to a board ranking on a played season, against the
                       #   retrospective board from the same `train` posteriors — with a
                       #   retro-vs-retro run at seed+1 as the noise floor the gap is
                       #   judged against. SEASON= and SIMS= override 2023-24 / 500.
```

`src/features/forward_design.py` also holds `synthetic_game_log`, which is the production
path rather than a target: the published schedule crossed with the roster snapshot, which
every existing builder then consumes unchanged because they all aggregate that one file.

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
                       #   all-star gap moves no Monday. Schedules cache to data/raw/nbastats, so
                       #   a rebuild needs no network; REFRESH=1 re-pulls them. A season
                       #   nobody has played enters only when NAMED — `--forward
                       #   2026-27` appends its triples from the synthetic game log, so
                       #   the grid carries the same game ids the roster grid will,
                       #   filler games included.

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

make minutes-role-sigma
                       # grades the injected per-(player, season) sigma by ROLE, over the
                       #   composition's own `rho_bin` → outputs/predictions/
                       #   minutes_role_sigma.csv. A DRAW-TIME calibration: no sampler, no
                       #   head refitted, ~13 min of numpy over the persisted posteriors.
                       #   A single sigma was missing in BOTH directions at once — at the
                       #   shared 0.375 fringe player-seasons were still 1.73x
                       #   under-dispersed while stars were OVER-dispersed at 0.86x — so
                       #   each bucket's CRPS grid is searched coordinate-wise on the last
                       #   two TRAINING seasons and confirmed on validation, which never
                       #   chooses. Ships [0.600, 0.375, 0.375, 0.300], a 2.00x spread;
                       #   two of the four buckets keep the shared value. Fringe sd_ratio
                       #   1.7276 -> 1.2757 and the win against the marginal head widens
                       #   to -6.4504 [-10.9991, -2.0419]. Deleting
                       #   `sim.minutes.player_season_sigma_by_role` restores the
                       #   pre-2026-08-16 draw bit-for-bit. It also PRINTS a warning when
                       #   config and the search disagree, since the search re-derives the
                       #   vector and never reads the key.

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

make minutes-preseason # preseason-plan P3: the preseason block as a NESTED arm on the
                       #   marginal minutes head → outputs/predictions/
                       #   minutes_preseason{,_rolling,_shrinkage}.csv. Same point-MLE
                       #   machinery as `minutes-window`, so no CmdStan; needs
                       #   `make preseason` and `make preseason-value` first.
                       #   THE GATE PASSES on both halves — validation -4.789
                       #   [-8.08, -1.59] and a 13-origin rolling harness at -7.940
                       #   [-9.41, -6.44], 12 of 13 — so the composition's conditional
                       #   opens. THREE THINGS TO KNOW. Every arm INCLUDING the reference
                       #   fits the covered window (2004-05 on), because this head fits
                       #   from 1997-98 and the missing indicator would be an era dummy on
                       #   a quarter of its rows; that cut alone is worth 1.19 CRPS. The
                       #   rolling reading is LARGER than validation, the first time in
                       #   this repo. And the shipped column is the delta CENTRED within
                       #   season: preseason minutes are compressed, the level is nuisance,
                       #   and centring both wins and repairs a bias the raw delta creates.

make availability-preseason
                       # preseason-plan P2: the same block as a NESTED arm on the
                       #   AVAILABILITY head that ships — the two-component mixture →
                       #   outputs/predictions/availability_preseason{,_rolling,_effects,
                       #   _block}.csv. Eight arms, point MLE, ~1 h, no CmdStan. The block
                       #   goes to two places, `beta` and the disruption weight `pi`,
                       #   because availability-window-plan §14d established those are
                       #   different questions. THE GATE FAILS, and it fails on the
                       #   reading that cannot resolve it: validation CRPS -0.102
                       #   [-0.322, +0.121] on the draftable population spans zero while the
                       #   rolling harness reads -0.254 [-0.344, -0.162] at 10 OF 10 origins
                       #   with the boundary held. The bar is a conjunction and it fails, so
                       #   nothing is ported — but it was written against the opposite
                       #   failure (§12e, §14f) and has no clause for this one. §14f's own
                       #   diagnostic says why: the rolling interval is 2.4x NARROWER on
                       #   4.6x the rows, so what validation lacks is resolution rather than
                       #   the effect being smaller. TWO THINGS TO KNOW. The pooled reading
                       #   passes at both readings and 6.2x of it (2.8x rolling) is
                       #   mid-season signings, which is P1's population finding at this
                       #   head's own unit. And the SHIPPED head's low-tail error changes
                       #   SIGN between the two populations at both readings — under-
                       #   predicting P(GP<10) pooled, over-predicting it on the draft pool
                       #   — potential-to-dos item 9.

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
                       # P4(a) of preseason-plan rides in the same target →
                       #   availability_no_prior_preseason.csv. Two extra axes on the SAME
                       #   ladder: a preseason minutes-share key, and the population the
                       #   rates are POOLED from. Both FAIL the gate on the draft pool and
                       #   nothing ships. Pooled, the preseason key passes at both
                       #   readings — the fourth instrument to show the population
                       #   restriction is load-bearing.

make composition-preseason
                       # Session 4b of preseason-plan, gate 1 →
                       #   outputs/predictions/composition_preseason.csv. The composition is
                       #   the one head where two of three routes are unreachable by a
                       #   coefficient: `w_share` is the feature OWN, the offset
                       #   (`logit_prior`) AND the allocation order (`order_frame`), and no
                       #   coefficient touches the last two. Blends a preseason minutes share
                       #   into `w_share` and scores it through the head's OWN no-fit floor,
                       #   which sets eta = 0 and isolates them — 48 s, no CmdStan, no
                       #   fit of the head. PASSES at the head's own selection unit
                       #   (-0.19972 [-0.21661, -0.18225] CRPS minutes per player-game) and
                       #   TIES at the season unit, which is this head's standing lesson.
                       #   The attribution says it is the OFFSET: `order_only` is worth
                       #   3.75% of the margin, which reverses half of what P3 predicted.

make composition-preseason-fit
                       # Sessions 4c and 4d: the half the screen above does not settle.
                       #   `beta` can correct an offset the floor cannot, so the increment
                       #   could shrink under a fit — or grow, as P3's own did. Fits of the
                       #   shipped variant (a same-window `base` control and the
                       #   blended-offset arm at k = 80), plus BOTH frames' no-fit floors at
                       #   the same 200 predictive draws, which makes the retention —
                       #   fitted increment / floor increment — a within-artifact ratio.
                       #   The bar is frozen in `report()` before the run.
                       #   4c ran the PILOT window (2018-19 on, ~30 min) →
                       #     composition_preseason_fit.csv, and docs-audit re-derives ~45
                       #     figures from it, so a later round must not overwrite it.
                       #   4d runs the head's own window, which the module CUTS to the
                       #     preseason panel's coverage (2004-05) for the two gate arms —
                       #     P3's rule — and adds `base_full_window` at 1996-97 carrying no
                       #     preseason column, to price that cut on its own. ~6.3 h for the
                       #     three; `stan.composition.preseason.label` namespaces its
                       #     artifacts → composition_preseason_fit_covered*.csv.
                       #   NOT part of `make stan`; does not write stan_composition_*.csv.

make rookie-priors     # P4(b): does a no-prior player's own preseason beat his draft
                       #   bucket? → outputs/predictions/rookie_priors.csv. The incumbent
                       #   is stan_composition.rookie_share_priors, the bio_draft_number
                       #   imputation behind his `w_share`. Three arms over one estimator,
                       #   with the blend's `k` chosen on an inner carve of the FITTING
                       #   half. The draft bucket is an ANTI-MODEL for rates (R2 -0.043 to
                       #   +0.046) and the shrunk preseason clears on five of eight targets
                       #   at 19 of 19 rolling origins; the minutes SHARE — the only target
                       #   with a live consumer — is a tie. numpy only, ~10 s.

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
                       #   Two further arms since 2026-08-16 — `mq` and `mq_graded`,
                       #   the same effect MARGINALIZED by per-unit quadrature rather
                       #   than sampled (docs/composition-quadrature-plan.md). Those
                       #   keep `dense_e` at every window, because their parameter block
                       #   is ~35 wide rather than one per unit, and Gate A scales them
                       #   by rows alone for the same reason.
                       #   ⚠️ `stan.composition.effects.label` suffixes every artifact
                       #   this target writes and defaults to "". Set it for any round
                       #   that is not re-running the 2026-08-09 ladder: `_flush` merges
                       #   by ARM NAME, and `make docs-audit` re-derives the pilot `base`
                       #   arm's CRPS from composition_effects_metrics.csv.

make composition-quadrature-check
                       # does the marginal path compute the integral it claims to, at
                       #   FULL scale? Production driver → Stan `log_prob` on the real
                       #   pilot frame (97,587 rows / 2,062 units / 1,487 truncation
                       #   rows), against an independent numpy evaluation of the same
                       #   integral written from the data dict alone
                       #   → outputs/predictions/composition_quadrature_check.csv.
                       #   NEEDS CmdStan but does NOT sample: two log-posterior
                       #   evaluations, seconds once the frame is built. RAISES on
                       #   disagreement. The unit suite checks the same claim on tiny
                       #   synthetic units; this is the half that cannot live there, and
                       #   it is the cheapest guard on the one approximation in the whole
                       #   representation. Run it after any change to
                       #   composition_glm.stan's `Q > 0` block.

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

make preseason-contest # what the PRESEASON BLOCK is worth in the contest, as a paired
                       #   counterfactual — the same device one round over
                       #   → outputs/predictions/preseason_block_contest.csv.
                       #   REPORTS two arms; it does not run them. Running them is two
                       #   passes over five targets differing in the FOUR keys that ARE
                       #   the block (`stan.availability.preseason`,
                       #   `stan.minutes.preseason`,
                       #   `stan.composition.preseason.adopt`,
                       #   `stan.components.preseason`), ~4.5 h each because all
                       #   four head groups are refitted and the composition is 2-3 h of
                       #   it. `--capture {base,preseason}` freezes each pass into
                       #   outputs/predictions/preseason_arms/. Run the COUNTERFACTUAL
                       #   first, so the shipped arm is what disk ends on. `--capture`
                       #   REFUSES unless ALL FOUR keys agree with the arm name — a pass
                       #   with the block half on is neither arm.
                       #   ⚠️ IT OVERWRITES IN PLACE. The 2026-08-15 four-key pass replaced
                       #   P5's three-key one in this same file, so preseason-plan's "P5
                       #   closes" section is a RECORD and session 6b holds the live
                       #   figures. The `base` capture is byte-identical across the two
                       #   passes, which is the only reason the components' own share can
                       #   be recovered by differencing the two deltas (-2.27088 and
                       #   -3.84536 of season-total MAE).
                       #   ⚠️ ONLY THREE OF THE FOUR HEAD GROUPS CAN REACH THIS. `src/sim/`
                       #   never loads the marginal minutes head, so P3's block — the
                       #   largest of the three by its own gate — is structurally
                       #   invisible here; a test pins the import fact and the `reach`
                       #   block reports which windows moved. ⚠️ The `reach` block has NO
                       #   `components` row: `reach_rows` reports `refit_landed` for
                       #   availability, minutes and composition only, so the family this
                       #   round shipped has no per-head trace in the artifact that prices
                       #   it — all four CONFIG KEYS do show 0→1. Read `resolution` before
                       #   `contest`, then `board`: this block arrives as the allocation
                       #   MEAN rather than as shape, so a ranking is what it can move.
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

