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
        stan stan-availability stan-minutes stan-components stan-composition

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

stan-minutes:
	$(PYTHON) -m src.models.stan_minutes

stan-components:
	$(PYTHON) -m src.models.stan_components

# The team-game minutes composition PILOT (docs/minutes-composition-plan.md) —
# deliberately NOT in the `stan` aggregate until the pilot's gates pass and the
# full-window decision is recorded there.
stan-composition:
	$(PYTHON) -m src.models.stan_composition

stan: stan-availability stan-minutes stan-components

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
