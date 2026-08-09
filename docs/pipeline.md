
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
make dashboard-audit  # registry drift report — a report, not a gate; exits 0 with findings
make docs-audit       # every quoted figure in the plan docs vs its artifact — a GATE
```

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


**`daily-capture` is the only thing in this repo with a deadline.** Both its sources are
current-status feeds that cannot be backfilled: the NBA injury-report PDFs age out of the
CDN after ~7 months (a 403 thereafter, forever) and the ESPN feed has no history at all.
A day the cron does not run is a day permanently lost. `make injury-reports` is idempotent
and re-parsing is offline (`--reparse`), because the archived PDFs — not the parsed CSVs —
are the artifact worth keeping.

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
make posteriors        # PERSIST the fits: 18 heads refitted once at the variant their
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

