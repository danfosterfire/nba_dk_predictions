PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: venv install fetch preprocess features train evaluate predict test clean \
        season-matrix pca archetypes eda team-context component-targets context-value \
        opponent persistence aging target-profile feature-diagnostics dashboard \
        dashboard-audit docs-audit \
        availability availability-profile injury-reports injuries daily-capture \
        boxscore-status availability-model capture-status report-calibration \
        season-total adp adp-draftkings adp-fantasypros adp-panel adp-profile \
        adp-status game-length serial-correlation component-rates \
        variance-budget residual-correlation season-effects \
        stan stan-availability stan-minutes stan-components stan-composition \
        stan-substitution season-terms games-played stan-games-played \
        stan-game-length posteriors minutes-unification composition-effects \
        scoring-periods draft-pool simulate-season final-evaluation

venv:
	/opt/homebrew/bin/python3.14 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

fetch:
	$(PYTHON) -m src.data.fetch

preprocess:
	$(PYTHON) -m src.data.preprocess

# ── Daily capture ─────────────────────────────────────────────────────────────
# Both sources are current-status only and CANNOT be backfilled: the NBA report PDFs
# age out of the CDN after ~7 months, and the ESPN feed has no history at all. Every
# day `daily-capture` does not run is a day permanently lost. Schedule it:
#
#   crontab -e
#   30 18 * * *  cd /path/to/nba_deep_learning && make daily-capture >> data/raw/daily_capture.log 2>&1
#
# 6:30 PM local is after the 5:00 PM ET report is published. Re-running the same day is
# a no-op, so a missed day self-heals on the next run for anything still retained.
injury-reports:
	$(PYTHON) -m src.data.injury_reports

injuries:
	$(PYTHON) -m src.data.injuries --daily

daily-capture: injury-reports injuries

# Which days were captured, which had no report to capture, and which were MISSED.
# The PDF gaps are recoverable until they age out; the ESPN gaps never are.
capture-status:
	$(PYTHON) -m src.data.injury_reports --status
	@echo
	$(PYTHON) -m src.data.injuries --status

# The 2006-07 → 2025-26 inactive-list and DNP-reason backfill. ~24,600 games, 8-14 h.
# Resumable per game — kill it and re-run.
boxscore-status:
	$(PYTHON) -m src.data.boxscore_status

# ── ADP ───────────────────────────────────────────────────────────────────────
# The DK board is login-gated, has ZERO Wayback presence, and is live only while
# contests are open (~Oct). It CANNOT be scraped or backfilled: download the CSV by hand
# from the draft lobby into data/raw/dk_draft_rankings/ and let `adp-draftkings` ingest
# it. See docs/adp-plan.md. FantasyPros is backfillable and has ~11 months of slack,
# because its board freezes between draft seasons.
adp-draftkings:
	$(PYTHON) -m src.data.adp_draftkings

# Live capture. Add --backfill for the 12-season Wayback sweep (slow, rate-limited),
# or --reparse to rebuild from the archive offline.
adp-fantasypros:
	$(PYTHON) -m src.data.adp_fantasypros

adp-panel:
	$(PYTHON) -m src.features.adp

adp-profile:
	$(PYTHON) -m src.eda.adp_profile

# Ingest both sources, build the panel, measure the transfer function.
adp: adp-draftkings adp-fantasypros adp-panel adp-profile

adp-status:
	$(PYTHON) -m src.data.adp_draftkings --status
	@echo
	$(PYTHON) -m src.data.adp_fantasypros --status

features:
	$(PYTHON) -m src.features.encode

season-matrix:
	$(PYTHON) -m src.eda.season_matrix

pca:
	$(PYTHON) -m src.eda.pca

archetypes:
	$(PYTHON) -m src.eda.archetypes

team-context:
	$(PYTHON) -m src.features.team_context

component-targets:
	$(PYTHON) -m src.features.targets

game-length:
	$(PYTHON) -m src.features.game_length

context-value:
	$(PYTHON) -m src.eda.context_value

opponent:
	$(PYTHON) -m src.features.opponent

persistence:
	$(PYTHON) -m src.eda.persistence

aging:
	$(PYTHON) -m src.eda.aging

target-profile:
	$(PYTHON) -m src.eda.target

feature-diagnostics:
	$(PYTHON) -m src.eda.feature_diagnostics

serial-correlation:
	$(PYTHON) -m src.eda.serial_correlation

# League-level season effects: is each quantity's era movement an extrapolable TREND or an
# unforecastable SHOCK, and what does ignoring it cost? No head carries a season term today.
season-effects:
	$(PYTHON) -m src.eda.season_effects

# ── Provenance: figures that were prose-only until docs/provenance-plan.md ────
# The variance budget needs the archetypes (for the interaction's lagged style label) and
# the raw game logs; the residual correlation needs component_targets. Both sit after
# `component-targets` and `opponent` in `make eda`.
variance-budget:
	$(PYTHON) -m src.eda.variance_budget

# The conditional cross-component correlation matrix — a simulator INPUT, not a summary.
residual-correlation:
	$(PYTHON) -m src.eda.residual_correlation

availability:
	$(PYTHON) -m src.features.availability

availability-profile:
	$(PYTHON) -m src.eda.availability

season-total:
	$(PYTHON) -m src.models.season_total

component-rates:
	$(PYTHON) -m src.models.component_rates

availability-model:
	$(PYTHON) -m src.models.availability

# ── Stan heads ────────────────────────────────────────────────────────────────
# Fitted SEPARATELY, one model per head, because the chain availability -> min |
# available -> counts | min -> makes | attempts factorizes the joint posterior exactly
# when the parameter blocks are distinct. Sources are in src/stan/; cmdstanpy compiles
# them into outputs/stan/, which .gitignore already covers, so no binary is committed.
#
# Requires cmdstanpy plus a CmdStan toolchain:
#   .venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
stan-availability:
	$(PYTHON) -m src.models.stan_availability

# The games-played spell process — docs/games-played-plan.md. `games-played` is the numpy
# reference and Gate 0: the collapse, the spell classes, the closed-form beta-geometric
# fits and the empirical-hazard Monte Carlo that rejects the plain full-window chain. It
# has NO Stan dependency and runs in seconds, which is the point — a process class is
# cheaper to reject in numpy than in NUTS.
games-played:
	$(PYTHON) -m src.models.games_played

# The fitted arms. Imports `stan_availability`'s head as its permanent floor, so that
# target has to be ahead of it — the same ordering constraint `stan-composition` has on
# `stan-minutes`. Held OUT of the `stan` aggregate until Gate D passes, matching how
# `stan-substitution` and `season-terms` are held out.
stan-games-played: games-played
	$(PYTHON) -m src.models.stan_games_played

stan-minutes:
	$(PYTHON) -m src.models.stan_minutes

stan-components:
	$(PYTHON) -m src.models.stan_components

# Gate 0 of docs/shot-attempt-basis-plan.md — the `fga` count x `fg3a | fga` share
# reparameterization against the two independent attempt counts, with BOTH arms
# un-handicapped: arm A at each head's own selected variant, arm B swept for real.
# Sixteen fits, and its own artifact, because `substitution_arm` lives inside
# `stan-components` and cannot be refreshed without that target's 209 minutes.
# Reads stan_component_metrics.csv, so run `stan-components` first.
stan-substitution:
	$(PYTHON) -m src.models.stan_components --gate0

# The game-length head — docs/simulations-plan.md, "The second prerequisite: game length is
# a random variable forward, not a lookup". Whether a game goes to overtime
# (betabinomial_glm.stan on ~30 season cells) x how deep (betageometric_duration.stan on
# four collapsed depth rows). NO new .stan source, ~2-4 parameters, seconds of sampler
# time: the cheapest head in the project.
#
# `stan-composition` imports its floor and its draw, so this target has to be ahead of it —
# the same ordering constraint the composition already has on `stan-minutes`.
stan-game-length:
	$(PYTHON) -m src.models.stan_game_length

# The team-game minutes composition (docs/minutes-composition-plan.md). Gates A-E all
# pass at the full window, so it is now part of the `stan` aggregate.
stan-composition:
	$(PYTHON) -m src.models.stan_composition

# Order matters: the composition imports `StanMinutes`, `minutes_variants` and
# `game_level_dispersion` from the minutes head, and its `independent_comparator` refits
# that head as the incumbent it is measured against. `stan-minutes` therefore has to be
# ahead of it, not merely present.
#
# BUDGET MOST OF A DAY. `stan-composition` alone took 9.9 h of sampler time at the full
# window on 2026-08-08, down from ~21 h before it went validation-only (its Gate A
# extrapolated 8.3 h and under-predicted by 1.17x, because per-row cost is superlinear in
# rows; the two-pass run missed by 1.63x, so the multiplier is not a constant — treat the
# gate as a lower bound). `stan-components` is ~2.3 h on top. Each composition arm
# checkpoints to outputs/checkpoints/stan_composition/ as it completes, so a crash costs
# one arm rather than the run. Sleeping the machine mid-run is safe: the sampler suspends
# and resumes, and perf_counter does not advance while asleep, so the reported cost stays
# honest while elapsed wall clock does not.
stan: stan-availability stan-minutes stan-game-length stan-components stan-composition

# ── The simulation layer's step zero ──────────────────────────────────────────
# `make stan` writes metrics, diagnostics and per-row predictions and THROWS THE
# COEFFICIENT DRAWS AWAY, so simulating from the joint posterior has meant refitting.
# This refits each head ONCE at its shipped variant and persists the thinned draws, the
# design recipe and the provenance to data/features/posteriors/<head>.pkl — after which
# `make posteriors` is the only target in the simulation layer that needs a CmdStan
# toolchain and everything downstream is numpy.
#
# BUDGET MOST OF A DAY, for the same reason `stan` does: eighteen fits, of which the
# composition head is one and is hours on its own. Each artifact is written the moment it
# is built and the manifest merges by head, so
#
#   $(PYTHON) -m src.models.posteriors --groups components
#
# re-does one family without touching the rest.
#
# Artifacts are namespaced by fit window — data/features/posteriors/<window>/ — because all
# three windows are wanted at once and have different consumers:
#   train      DEFAULT. The realized backtest scores 2022-23 / 2023-24, and `train_val`
#              fits on them.
#   train_val  the one-shot test readout, which should describe the model that would
#              actually deploy — the rule src/final_evaluation.py already follows.
#   full       the 2026-27 production board. Reads the held-out seasons, so it is guarded
#              by src/models/held_out.py.
WINDOW ?= train

posteriors:
	$(PYTHON) -m src.models.posteriors --window $(WINDOW)

# Does the composition supersede the marginal minutes head? README.md claimed the two
# "compose rather than compete", with the marginal head still owning the season-level mean
# and the game-level dispersion. This scores both at the SEASON unit on the same validation
# player-seasons — the comparison neither head's own metrics table could make, because they
# publish at different units.
#
# Needs `make posteriors` and nothing else: it rehydrates both heads around their persisted
# draws and calls their own `predict_samples`, so it costs seconds rather than the
# composition's 9.92 h, and no arm is a differently-fitted model from the one it is compared
# against. No CmdStan.
minutes-unification:
	$(PYTHON) -m src.models.minutes_unification

# Item 3d: fit the per-(player, season) random effect `make minutes-unification` measured
# the need for, and sweep a team-context block alongside it. Four arms — a same-window
# `base` control plus `ps`, `team`, `ps_team` — at the PILOT window by default, because the
# ordering is what the pilot buys and the full-window commitment is a separate decision.
#
# NEEDS CmdStan, and it is expensive: `dense_e` is forced off on the random-effect arms
# (12,307 units at the full window would be a 12,332-square mass matrix), so part of the
# treedepth win the dense metric bought is given back. Gate A probes the `ps` arm rather
# than a plain one and aborts before a run that will not fit the budget.
#
# Deliberately NOT part of `make stan`, and deliberately not writing
# outputs/predictions/stan_composition_*.csv: that artifact is the incumbent's record and
# `make docs-audit` re-derives eleven quoted figures from it.
composition-effects:
	$(PYTHON) -m src.models.composition_effects

# One row per (season, game_id): its scoring period and its DK tournament round. A
# best-ball lineup is scored weekly, so every weekly max, round total and advancement
# cut downstream is an aggregate over a period, and this is the only module that says
# what a period is. DK's periods are NBA weeks: ScheduleLeagueV2 carries `weekNumber`
# from 2017-18 on, and older seasons get a derivation that reproduces it exactly on all
# nine seasons that publish one. Owns three edge cases once — a postponed game scores in
# the period it is PLAYED in, the NBA Cup final scores nowhere, and the all-star gap
# breaks week adjacency without moving a Monday. Schedules cache to data/raw, so a
# rebuild does not need the endpoint; `REFRESH=1` re-pulls them.
scoring-periods:
	$(PYTHON) -m src.features.scoring_periods $(if $(REFRESH),--refresh,)

# One row per (season, player): team, DK position eligibility, ADP, and the prior-season
# key the heads score him from. This is the board the draft simulator picks from, and
# eligibility is the load-bearing half — a best-ball week starts 2 G / 2 F / 1 C / 2 UTIL,
# so it decides which slots a player can fill and therefore every weekly max downstream.
#
# 🔴 IT REVERSES A PLAN ASSUMPTION. docs/simulations-plan.md said team_rosters_*.csv
# carries "DK-shaped dual eligibility". It does not: both DK boards print exactly ONE of
# G / F / C for all 1,640 rows, and DK's label is 99.85% stable across them. Validated on
# the persistent DK id (never a name), DK's letter equals NBA.com's primary on 86.63% of
# players and lies inside NBA.com's position set on 92.61%. So `position` ships a single
# letter — DK's own where a board exists, NBA.com's primary otherwise — and the rejected
# dual convention rides along as `dual_*` so a sensitivity run is a column swap.
draft-pool:
	$(PYTHON) -m src.features.draft_pool

# ── The simulation layer (src/sim/) ───────────────────────────────────────────
# THE tensor: player x scoring_period x sim dk_pts plus a uint8 games-played twin, one
# .npz per season. Per-game draws happen INSIDE the module and are summed into the 20
# scoring periods immediately, because the bonus is a per-game threshold on five components
# and E[bonus] != bonus(E[x]) — nothing downstream ever materializes a player x game x sim
# array. It is a staircase rather than one step: +1.5 for a double-double and +3 more for a
# triple-double, stacking to 4.5, so the convexity is sharper than either alone. Numpy only: it reads `make posteriors`' pickles and needs no
# CmdStan. Defaults to the two VALIDATION seasons; `--season` and `--n-sims` override.
simulate-season:
	$(PYTHON) -m src.sim.season

# Does any head need a season term, and which kind? A trend covariate and a year-level
# random effect for every head, plus the season x role interaction the availability era
# effect calls for, scored on held-out CRPS, interval coverage and season-total dk_pts.
# Reads each head's already-selected spec from the `stan-components` / `stan-minutes`
# artifacts, so run those first or it falls back to the right-scale spec and says so.
# Deliberately NOT in the `stan` aggregate: it is an ablation over the shipped heads
# rather than one of them.
season-terms:
	$(PYTHON) -m src.models.season_terms

report-calibration:
	$(PYTHON) -m src.eda.report_calibration

# Full EDA sweep, in dependency order
eda: season-matrix pca archetypes team-context context-value opponent \
     component-targets game-length variance-budget residual-correlation \
     persistence aging target-profile \
     feature-diagnostics serial-correlation \
     availability availability-profile season-effects report-calibration \
     adp-panel adp-profile

# Through `$(PYTHON) -m`, not `.venv/bin/streamlit`: the venv's console scripts carry an
# absolute shebang from the directory the venv was created in, so they broke when the repo
# was renamed off `nba_deep_learning`. Module invocation reads the interpreter from
# `$(PYTHON)` and survives a rename.
dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

# Registry drift report — see dashboard/README.md. A report, not a gate: it exits 0
# with findings on purpose, because failing on a doc edit trains people to ignore it.
# `python -m dashboard.audit` rather than `src.<module>`: the audit is about the
# dashboard, not the data pipeline.
#
# Runs weekly under launchd, appending to outputs/dashboard_audit.log:
#
#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nba-deep-learning.dashboard-audit.plist
#   launchctl kickstart -p gui/$(id -u)/com.nba-deep-learning.dashboard-audit   # run now
#
# Weekly rather than daily because nothing here has a deadline — unlike daily-capture,
# whose sources cannot be backfilled. Reviewing the log is a two-minute job, which is
# the whole point: the alternative is a re-read of six plan docs.
dashboard-audit:
	$(PYTHON) -m dashboard.audit

# ── The one reading of the held-out seasons ───────────────────────────────────
# Every sweep, ablation and gate in this project selects on VALIDATION;
# src/models/held_out.py locks the test split and raises on anything that reaches it.
# This target is the only thing that unlocks it: it takes the already-selected spec,
# refits on train+validation and scores test ONCE.
#
# Do not run it to check whether a validation result "held up". If a number from here
# changes a modelling decision, the split is spent and the estimate is no longer unbiased.
final-evaluation:
	$(PYTHON) -m src.final_evaluation

# Every quoted figure in the plan docs, checked against the artifact behind it. Unlike
# dashboard-audit this one is a GATE — it exits non-zero on a disagreement, because a doc
# contradicting its artifact is an unambiguous defect. Missing artifacts are skipped, so a
# fresh checkout without `make eda` does not report a wall of red.
#
# Rebuilding an artifact with new data will make this fail until the docs are updated.
# That is the intended behaviour: it is the alarm that has been missing twice.
docs-audit:
	$(PYTHON) -m src.docs_audit

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

predict:
	$(PYTHON) -m src.predict

test:
	$(PYTHON) -m pytest tests/ -v

# Run the full pipeline end-to-end
pipeline: fetch preprocess features train evaluate

clean:
	rm -rf data/raw/* data/processed/* data/features/* outputs/checkpoints/* outputs/predictions/*
