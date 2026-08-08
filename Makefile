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
        final-evaluation

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
stan: stan-availability stan-minutes stan-components stan-composition

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

dashboard:
	.venv/bin/streamlit run dashboard/app.py

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
	.venv/bin/pytest tests/ -v

# Run the full pipeline end-to-end
pipeline: fetch preprocess features train evaluate

clean:
	rm -rf data/raw/* data/processed/* data/features/* outputs/checkpoints/* outputs/predictions/*
