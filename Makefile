PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

# Python buffers stdout whenever it is not writing to a terminal, which under `make` is
# almost always. That turns this project's progress-line convention
# (`f"... {n:,} ... → {dest}"`) into nothing at all until a buffer fills or the process
# exits, so a multi-hour target that is working looks exactly like one that is hung — and
# the long ones here run 45 min to 2.7 h. It has cost real diagnostic time: one session had
# the component ladder, `posteriors` and `strategy-sweep` all run blind and needed
# `/usr/bin/sample` to confirm the sweep was alive, on a day the sweep took 53 minutes
# against the 4.8 the docs quoted. `export` applies it to every recipe rather than to a
# hand-maintained list of the slow ones, because unbuffered stdout costs the fast targets
# nothing and a list is one more thing to forget to add a target to.
export PYTHONUNBUFFERED = 1

.PHONY: venv install fetch preprocess features train evaluate predict test clean \
        season-matrix pca archetypes eda team-context component-targets context-value \
        opponent persistence aging target-profile feature-diagnostics dashboard \
        dashboard-audit dashboard-config docs-audit \
        availability availability-profile injury-reports injuries daily-capture \
        boxscore-status availability-model availability-window \
        availability-weighting availability-regime availability-exchangeability \
        availability-no-prior availability-absence availability-preseason \
        rookie-priors \
        capture-status \
        capture-calendar \
        report-calibration \
        season-total adp adp-draftkings adp-fantasypros adp-panel adp-profile \
        adp-status game-length preseason preseason-value serial-correlation \
        component-rates components-preseason \
        variance-budget residual-correlation season-effects \
        stan stan-availability stan-availability-mixture stan-minutes \
        stan-components stan-composition \
        stan-substitution season-terms games-played stan-games-played \
        stan-game-length posteriors model-cards minutes-unification minutes-window \
        minutes-preseason composition-effects composition-preseason \
        composition-preseason-fit \
        scoring-periods draft-pool simulate-season weekly-scores bracket draft-sim \
        draft-sim-need draft-room draft-room-prep strategy-sweep strategy-sweep-need \
        pick-log-stake mixture-value preseason-contest final-evaluation

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

# The calendar runs last and makes no requests: whatever captured, record what is now on
# disk. A cron that stops firing is only visible in the artifact it stops refreshing.
daily-capture: injury-reports injuries capture-calendar

# Which days were captured, which had no report to capture, and which were MISSED.
# The PDF gaps are recoverable until they age out; the ESPN gaps never are.
capture-status:
	$(PYTHON) -m src.data.injury_reports --status
	@echo
	$(PYTHON) -m src.data.injuries --status

# The same coverage as `capture-status` and `adp-status`, for all four programs, as an
# artifact rather than a printout — the dashboard's page 7 draws it. Reads disk only.
capture-calendar:
	$(PYTHON) -m src.data.capture_calendar

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

# Current-season preseason games as forecast covariates (docs/preseason-plan.md, P0).
# The logs are BACKFILLABLE — one fetch after the final preseason game is enough, so this
# deliberately stays out of `make daily-capture`.
preseason:
	$(PYTHON) -m src.features.preseason

# P1, the gate that decides which heads get a preseason arm — redundancy, incremental
# signal against each head's own metric, and the missingness census. TRAIN SEASONS ONLY:
# it never materializes validation, because a screen that spends the selection split
# leaves P2 nothing to select on. Needs `make preseason` first.
preseason-value:
	$(PYTHON) -m src.eda.preseason_value

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

# Session 6b of docs/preseason-plan.md: the preseason block as NESTED arms on the five count
# heads P1's gate cleared (`ast`, `fga`, `stl`, `tov`, `reb`) plus `ftm|fta`. P1's own text
# says its bar was an R^2 screen on a point estimate and is a filter for what is worth
# fitting, never evidence that anything ships — so this re-asks it at the P2/P3 bar: a
# validation CRPS interval clear of zero on the DRAFTABLE population AND the rolling-origin
# harness agreeing.
#
# `blk`, `fta`, `fg2m|fg2a` and `fg3m|fg3a` are P1's recorded nulls and get no arm. Every
# arm — the reference included — fits the COVERED window only (2004-05 onward), since this
# design fits from 1997-98 and a missing-preseason indicator would read as an era dummy on
# the pre-2005 rows; the full-window incumbent rides as a context row so the restriction's
# own cost is visible. Point MLE on each head's SHIPPED variant (read from
# stan_component_metrics.csv), so no CmdStan and an arm can be rejected before any sampler
# time is spent. Needs `make preseason`, `make preseason-value` and `make stan-components`.
components-preseason:
	$(PYTHON) -m src.models.components_preseason

availability-model:
	$(PYTHON) -m src.models.availability

# The fitting-window x season-trend x dispersion x LIKELIHOOD ladder for the availability
# head, scored on TAIL COVERAGE rather than only CRPS — the shipped head misses both ends
# of its own distribution and no marginal metric it has ever been gated on can see that.
# The fourth axis was added 2026-08-11 once the mechanism was measured: `a` and `b` are both
# functions of `(mu, rho)`, so the frailty's boundary behaviour and its variance are the
# SAME parameter, which is why moving `rho` could only halve the miss. It varies the frailty
# with the other three axes held at the shipped arm, and every arm reproduces the incumbent
# at its own nesting parameter values. Point MLE, no CmdStan, minutes: an arm earns a Stan
# port here, it does not ship from here.
availability-window:
	$(PYTHON) -m src.models.availability_window

# Four ways to SPEND the old seasons rather than keep or discard them wholesale:
# regularization crossed with lookback (the confound in the ladder above), separate
# windows for the mean and the dispersion, exponential season decay, and per-coefficient-
# block windowing. All scored on the rolling harness — validation is never read.
availability-weighting:
	$(PYTHON) -m src.models.availability_weighting

# The two axes `docs/availability-window-plan.md` §9 leaves open, run as one round because
# they are the same complaint from two sides. (1) RECENCY IS NOT REPRESENTATIVENESS: an
# explicit regime indicator for 2019-20 -> 2021-22, or excluding those seasons, swept
# against lookback. (2) SHRINKAGE toward the long-window fit — the blocks fitted jointly
# under different priors rather than transplanted, which is the principled version of the
# spliced arm that failed in §5b. The plain walk-forward harness is structurally BLIND to
# axis (1) — the regime block is the last three fitting seasons, so most origins produce a
# bit-identical fit — so the selector is a contamination harness that injects the block
# into a non-regime origin, with a same-size ordinary block as its control. Validation is
# read once, at the end, on the point MLE AND on the shipped mixture.
availability-regime:
	$(PYTHON) -m src.models.availability_regime

# The last axis `docs/availability-window-plan.md` §9 leaves open: the head's trials are
# NOT exchangeable — absences come in spells — and no arm on the likelihood axis touched
# it. Nothing here fits anything, because `gp` is INVARIANT to the arrangement, so no
# likelihood over it can separate clustering from frailty (they enter the variance only
# through `C + rho*(n - C)`). The instrument is therefore a ladder at the SCORING PERIOD,
# holding gp fixed at its realized value and varying only the layout: observed against
# `allocate_spells` (what ships) against uniformly-placed absences (what the beta-binomial
# asserts). numpy only, no CmdStan, seconds.
availability-exchangeability:
	$(PYTHON) -m src.models.availability_exchangeability

# Two independent attacks on the boundary defect, crossed as ONE 2x2 so their impacts can
# be told apart. (1) The COVARIATE block: §11b measured that a missed game is four
# processes with opposite role signatures and the head sees none of it, so this adds last
# season's absence COMPOSITION as shares. (2) The LIKELIHOOD: a compound counting process,
# `missed = sum of K spells` with K beta-binomial and the spell length its own
# beta-geometric, which gives "zero onsets all year" and "one absorbing event, early"
# different parameters. `lambda = 1` nests the incumbent exactly, and the round profiles
# it rather than trusting a free fit that stops at the corner. Point MLE, numpy, minutes.
#
# THREE ROUNDS, selected by `features.availability.absence.rounds`. The 2x2 above is
# `crossed`, and it measured the block against `betabinom` — which is not the head that
# ships. `mixture` is §14: the same block crossed against the shipped two-component head,
# with the block on `beta` alone in one arm and on `beta` AND the disruption weight `pi` in
# the other, because those are different questions. The rounds write DISJOINT artifacts, so
# `rounds: [mixture]` re-runs §14 without touching the five files `make docs-audit`
# re-derives §12 from.
#
# `population` is §15a (`potential-to-dos.md` item 9) and it asks a question about a
# SHIPPED head: §7 selected the two-component `mixture` over `betabinom` on a boundary
# error measured on all 883 validation rows, and 772 of those are on an October roster —
# the only population this head is ever applied to. Every arm is fitted ONCE and the
# population is a mask on the SCORED rows, so a column difference is the same head on a
# subset rather than a head refitted for it. Validation and §4b's rolling harness both,
# because a sign flip on a shipped head's selection metric is what §10e's replicate-or-fail
# rule exists for. ~30 min.
availability-absence:
	$(PYTHON) -m src.models.availability_absence

# P2 of docs/preseason-plan.md: the preseason block as a NESTED arm on the availability head
# that SHIPS — the two-component mixture, three_point_era window, role-graded rho. Eight
# arms, and the block goes to two different places: the mean function `beta` and the
# disrupted-season weight `pi`, which §14d established are different questions. The declared
# primary is ONE column plus P1's age-split indicator, because P1 measured every column of
# its own block beating the block that contains them on this target.
#
# Unlike P3 nothing is cut for coverage: the shipped window starts 2012-13 and the panel
# starts 2004-05. What IS load-bearing is the POPULATION — P1's first reading here was 6x
# too large because it pooled mid-season signings — so every arm is scored on both and the
# verdict is read on the season-start roster. The bar is BOTH validation and the
# rolling-origin harness; §12e and §14f are two blocks that won validation on these exact
# rows and shrank 4-6x rolling. Point MLE, numpy, ~1 h. Needs `make preseason`.
availability-preseason:
	$(PYTHON) -m src.models.availability_preseason

# What the players the head has NO ROW FOR actually realize — rookies and returning
# veterans, who reach the simulator through `sim/season.no_design_availability` rather than
# through the design. Two rounds, in the order they ran.
#
# §8a settled `docs/availability-window-plan.md` §8 decision 2, which specified grading
# their ROLE BUCKET in three classes. The bucket carries dispersion, not level, so the
# decision is only worth implementing if the classes differ in dispersion — and they do not.
#
# §8b is the ladder on the axis that survived. The arms are pooling KEYS over ONE estimator
# — the realized `gp / team_games` of rows carrying that key over seasons strictly before
# the target — so `pooled` reproduces the shipped scalar exactly and the comparison is a
# mean function against a mean function. Scored through the beta-binomial the simulator
# itself applies, at the fringe bucket's rho. `tenure_draft` ships. numpy only, seconds.
#
# P4(a) of docs/preseason-plan.md rides along in the same target rather than taking its own:
# it is the SAME ladder on the SAME rows with a preseason minutes-share key joined on, and a
# second target would have to rebuild the panel, the design and the classification to ask a
# question one merge away. It adds two axes — the key, and the population the rates are
# POOLED from, since the consumer is only ever applied to rostered players and the estimator
# has always pooled over January signings too. Both readings are quoted on the draftable
# population per P1 decision 5. Needs `make preseason`; skipped with a message without it.
#
# `run_recency` is §15b (`potential-to-dos.md` item 11) and rides along for the same reason:
# it crosses that population axis with the pool's DEPTH — all seasons before the target
# against the last 10 or 5 — on the shipped key. P4(a) left a hypothesis that the all-rows
# estimator's near-zero validation bias is a CANCELLATION between a population error and an
# era drift rather than accuracy, and this is what separates them. It does: the drift is
# real and removing it makes CRPS monotonically WORSE, because a shallow pool starves
# `MIN_CELL` and `graded_share` falls 0.7926 -> 0.5524. Nothing ships. numpy only.
availability-no-prior:
	$(PYTHON) -m src.models.availability_no_prior

# P4(b): the OTHER thing a no-prior player gets from his draft slot. `stan_composition.
# rookie_share_priors` hands him an expanding-window mean of what past players in his draft
# bucket realized, which becomes his `w_share` — the composition's prior minutes share and,
# through `order_frame`, his place in the allocation order. This asks whether his own
# preseason beats that, on the share and on the seven per-36 rates plus the 3PA mix.
#
# Three arms over ONE estimator, the §8b discipline: the bucket mean, his preseason reading,
# and the two blended by preseason volume with `k` chosen on an inner carve of the FITTING
# half. `k = 0` and `k = inf` are the two endpoint arms exactly. numpy only, ~1 min.
rookie-priors:
	$(PYTHON) -m src.models.rookie_priors

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

# The low-availability mixture's port check — docs/availability-window-plan.md §7h. A
# SEPARATE target from `stan-availability` on purpose: that one writes the head's shipped
# metrics, which every quoted port figure in the docs is audited against, and this answers
# a different question (does the Stan mixture reproduce the point-MLE arm §7c selected)
# into its own artifacts. Held out of the `stan` aggregate for the same reason
# `stan-substitution` is.
stan-availability-mixture:
	$(PYTHON) -m src.models.stan_availability --mixture-check

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

# The dashboard-shaped view of every fitted head — one flat artifact per block of a model
# page: the index, the coefficients, the features, their correlations, the predictive ECDF
# ribbon, the binned calibration density and a bounded scatter sample.
# The pages CANNOT read data/features/posteriors/*.pkl themselves: unpickling imports
# src.models.posteriors, which the dashboard's ast-based purity guard cannot see because it
# walks static imports only, and the object carries a fitted StandardScaler plus the ordered
# design steps — the capability to score an arbitrary frame, which is exactly the drift the
# rule exists to prevent. So this emitter stands between them and the dashboard reads only
# its output.
#
# Reads the `train` window and nothing else, deliberately without a WINDOW knob: at
# train_val the validation rows were IN the fit, and a "validation" histogram drawn from
# those coefficients is an in-sample picture wearing the wrong label. Every emitted row
# carries `split` in {train, validation}; there is no test column.
#
# Cheap and NO CmdStan — ~10 s over the persisted posteriors, no refit and no Stan sampler.
# It fails the build rather than writing a wrong artifact: each head's design matrix is
# re-derived twice, once through the persisted recipe and once through the head's own
# variant ladder, and the two must agree to 1e-9 — and the 200-draw predictive it draws
# through each head's OWN predict_samples must reproduce that head's reported mean to 5%,
# which is the failure a design check structurally cannot see. See docs/model-cards-plan.md.
#
# There is deliberately no rebuild-one-head flag, unlike `make posteriors`: at ten seconds
# for all twenty a partial run buys nothing and would leave the artifacts describing three
# heads. To debug one head's check, `--check <heads>` builds and verifies without writing:
#
#   $(PYTHON) -m src.models.model_cards --check composition
model-cards:
	$(PYTHON) -m src.models.model_cards

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

# The fitting-window x dispersion ladder for the MARGINAL minutes head — the same question
# `make availability-window` asked one head over, and the one `docs/availability-window-
# plan.md` §9 item 1 called the largest open stake in that line of work, because it can
# revise a SHIPPED decision. Four steps: rebuild §6's era series through each head's own
# design rows (its own caveat says they were measured on a rotation filter), scan for the
# break, ladder window x dispersion on validation, confirm on a rolling origin over the
# fitting half, and re-run the composition's injected-sigma grid against each window's arm.
#
# Point MLE for the ladder and a rehydrated posterior for the stake, so no CmdStan and no
# refit of either minutes head — minutes, not the composition's 9.92 h. Needs
# `make posteriors` for step 4 only.
minutes-window:
	$(PYTHON) -m src.models.minutes_window

# P3 of docs/preseason-plan.md: the preseason block as a NESTED arm on the marginal minutes
# head, which P1's gate moved ahead of availability because the block is worth 2.5x more
# here on the population it would be used on (+0.0492 R^2 against +0.0198). Same point-MLE
# machinery as `minutes-window`, so no CmdStan and an arm can be rejected before any
# sampler time is spent.
#
# Every arm — the reference included — fits the COVERED window only (2004-05 onward), since
# this head fits from 1997-98 and a missing-preseason indicator would otherwise read as an
# era dummy on a quarter of the training rows. The bar is stated in the module docstring and
# is BOTH validation and the rolling-origin harness; the composition head is priced only if
# it passes. Needs `make preseason` and `make preseason-value`.
minutes-preseason:
	$(PYTHON) -m src.models.minutes_preseason

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

# Session 4b of docs/preseason-plan.md, gate 1. `w_share` enters this head THREE ways — as
# the feature OWN, as the offset (`logit_prior`), and as the allocation ORDER (`order_frame`)
# — and NO COEFFICIENT reaches the last two, which is what the round's house pattern cannot
# get at. This blends a preseason minutes share into `w_share` and scores it through the
# head's OWN no-fit floor, which sets eta = 0 and so isolates exactly those two routes: the
# whole question is answered with no CmdStan and no fit, and a losing arm never costs an hour.
#
# `k -> inf` is the incumbent EXACTLY and `k` is selected on an inner carve of the fitting
# half. Both units, because `make minutes-unification` is the standing demonstration that
# this head's verdict belongs to a unit. Needs `make preseason`. ~6 min.
composition-preseason:
	$(PYTHON) -m src.models.composition_preseason

# Session 4b's FIT — the half the screen above explicitly does not settle. `FloorComposition`
# sets eta = 0, so it can say a better offset helps but not whether `beta` would have
# absorbed the help; P3's own increment GREW when integrated over `beta` and this head's
# per-team-game level is exactly what a fitted intercept is good at soaking up, so both
# directions have a precedent.
#
# Fits of the SHIPPED variant — a same-window `base` control and the blended-offset arm at
# k = 80 — plus both frames' own no-fit floors at the same 200 predictive draws, so the
# retention (fitted increment / floor increment) is a within-artifact ratio rather than a
# comparison across two rounds' draw budgets.
#
# ⚙️ SESSION 4D promoted this off the pilot window, and the promotion brings a THIRD arm.
# The preseason panel starts at 2004-05 and this head fits from 1996-97, so the two gate
# arms are cut to the covered window — P3's rule, since a missing-preseason indicator on a
# pre-2005 row is an era dummy — and `base_full_window` fits 1996-97 carrying no preseason
# column, which is what prices that cut on its own. P3 measured the same cut at 1.19 CRPS
# minutes BEFORE any preseason column existed, a quarter of that round's increment.
#
# Deliberately NOT part of `make stan`, and deliberately not writing
# outputs/predictions/stan_composition_*.csv: that artifact is the incumbent's record and
# `make docs-audit` re-derives eleven quoted figures from it. The same argument applies one
# level up — a labelled run (`stan.composition.preseason.label`) keeps 4c's PILOT artifact
# intact, since docs-audit re-derives ~45 figures from that one too.
#
# ⚠️ Cost: ~1.9 h per covered-window arm and ~2.6 h for the full-window control, so ~6.3 h
# for the three. NOT the 9.92 h "per arm" an earlier handoff quoted — that figure is the
# WHOLE four-variant `make stan-composition` sweep (2.01 + 2.60 + 2.50 + 2.67 h, plus probe
# and comparator), not one fit. Each arm checkpoints as it lands. Needs `make preseason`.
composition-preseason-fit:
	$(PYTHON) -m src.models.composition_preseason_fit

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

# Gate A at the unit the LINEUP is set at. `make simulate-season` scores the season total,
# the games-played pmf, the per-game bonus rate and the season-total minutes spread —
# nothing scores dk_pts at the scoring period, which is where DK seats the best 7 of 16 and
# therefore where every weekly max, round total and elimination cut downstream comes from.
# A head is only a model at the unit it was scored at; this is that check one level down
# from Gate A's own headline row.
#
# NO re-simulation: the tensor's second axis already IS the scoring period, so the whole
# target is a reduction plus the model-card binning helpers, by import, and runs in ~2 s.
# It reads four tensors — the two validation seasons plus 2018-19 and 2021-22, which are
# the last two TRAINING seasons carrying DK's whole four-round structure (2020-21 has no
# Round 4 at all and 2019-20's is the Orlando bubble). Build the training pair first:
#
#   $(PYTHON) -m src.sim.season --season 2018-19 --season 2021-22
#
# `--draws` moves the number of simulated seasons behind each panel and `--no-write` gates
# without writing, which is how that budget was measured rather than assumed.
weekly-scores:
	$(PYTHON) -m src.sim.weekly

# The contest itself: best 7 of 16 by slot each scoring period, the four-round advance
# chain, the cascading tie-break, wildcards and payouts. Every structural number — round
# count, pod size, advance count, cash table — is read from dashboard/economics.py, which
# derives it from the two captured DK CSVs, so pointing this at the live 2026-27 contests
# is a data change and not a code change.
#
# The lineup is an ASSIGNMENT problem, not a greedy fill: seating a dual-eligible player in
# the first slot he fits can lock a better player out, and it never raises. Seatable
# 7-subsets are the independent sets of a transversal matroid, so sorting by score and
# keeping every player who preserves seatability is exactly optimal.
#
# Its own check is the symmetric-field null, which is known in closed form: an exchangeable
# entry advances at n_advance/pod_size and is worth exactly -rake, because the prize pool is
# paid out in full. That exercises the pod sizes, the advance chain, the wildcard fill and
# every cash band at once — and it is what caught a transcription slip in the 15k_and_one
# prize CSV. `--n-field` sets deep-round resolution; see configs/default.yaml.
bracket:
	$(PYTHON) -m src.sim.bracket

# The snake draft: 12 entries, 16 rounds, one engine and two modes. REACTIVE is primary —
# a pick function sees the board and ranks the remaining players by their MARGINAL LINEUP
# VALUE on the sim tensor, so positional scarcity is priced by the same matroid that
# decides a real week. RANKING-SUBMISSION is the fallback and is DK's documented autodraft
# verbatim: queue first, then the pre-draft ranking, under 8G/8F/3C caps, with an exclusion
# list that yields only when a needed position would otherwise go unfilled.
#
# Opponents autodraft off the DK-RECALIBRATED consensus (draft_pool.adp_dk_scale), never
# the raw one — docs/adp-plan.md measured that DK takes centers 11.9 picks earlier because
# category-league ADP discounts them for FT%, and uncorrected that reads as model edge on
# one position. The opponent model is a REGISTRY: a strategy supplies static keys and an
# optional roster-aware bonus, and the engine owns availability, caps and legality.
#
# Gate B fits `rank_noise_sd` rather than choosing it: simulate many drafts, take each
# player's mean pick over the drafts he went in (DK's own definition of an ADP), and score
# it against the curve the field consumed, against the recalibration's own 17.0-pick error.
draft-sim:
	$(PYTHON) -m src.sim.draft

# The `adp_need` field: disciplined ADP consensus merged with lineup reasoning — every
# seat leans toward the starting slots (2 G / 2 F / 1 C) it still owes, by `need_weight`
# picks per owed slot. (noise, need) are fitted JOINTLY on Gate B's mean-ADP target, with
# need_weight = 0 nesting the shipped pure-ADP field exactly, so the artifact answers a
# question the base calibration could not ask: does the fitted rank noise stand in for
# lineup reasoning the field model omits? Writes draft_gate_b_need.csv, which is what
# `selected_field` resolves when the composition seats `adp_need`.
draft-sim-need:
	$(PYTHON) -m src.sim.draft --opponent adp_need

# The live draft room. `draft-room-prep` is the offline half: it drafts and scores each
# season's reference field ONCE into data/features/draft_room_field_<season>.npz — the
# population the recommender's `q` is read from — checks it against the symmetric-field
# null, and measures GATE E, which is the 1.0 s per-recompute bar a 30-second fast-draft
# clock implies. `draft-room` is the page, and it loads that artifact rather than
# rebuilding it, so launching a room is a second rather than a minute.
#
# Two things make the recompute fit, and both are in docs/simulations-plan.md: n_sims
# drops to 500 in-draft because the decision is a RANKING of candidates rather than an
# estimate of a level, and best-7-by-slot is ONE matroid exchange per candidate rather
# than a re-solve — exact, not approximate, and pinned against bracket.best_lineup.
#
# The objective is decision 5's: payout-weighted EV over all four rounds, with the
# survivor population of rounds 2-4 obtained by reweighting the same field rather than by
# dealing it. Read draft_room_null.csv before trusting an EV level and
# draft_room_stability.csv before trusting a close call between two candidates.
draft-room-prep:
	$(PYTHON) -m src.sim.draft_room

# Module invocation for the same reason `dashboard` uses it: the venv's console scripts
# carry an absolute shebang and do not survive the repo being renamed.
draft-room:
	$(PYTHON) -m streamlit run dashboard/draft_room.py

# The strategy sweep: a table over ranking source, blend weight (overall and per round),
# position caps, exposure caps, stacking, in-draft objective and entry count, scored on
# simulated truth and read out against realized 2022-23 / 2023-24.
#
# GATE C runs first because it gates the sweep's validity. The plan's premise is that a
# world drawn from the model's own posterior is too easy, so alpha goes to zero for reasons
# that have nothing to do with the market; measured, the premise is half right. The
# MAGNITUDE of the model's miss is already reproduced without any injection — 419.6 dk_pts
# of season-total MAE against a measured 400.5. What is wrong is the two rankers' relative
# standing: in that world the model leads ADP by +0.11 Spearman, while on realized
# validation seasons the MARKET leads by +0.06. So the injection ROTATES the error onto the
# market-visible direction at a fixed magnitude rather than adding noise on top of it, and
# `rho` is solved from that gap. An independent route — the correlation between market
# disagreement and the model's realized error — agrees to within a step.
#
# Selection is LIFT IN P(top 2 of 12), which is exact for the ADP baseline (n_advance /
# pod_size) and resolves orders of magnitude faster than ROI; ROI rides alongside with a
# bootstrap interval against the break-even hurdle. GATE D asks whether the two tiers
# actually select different rosters and reports the mechanism, since a ranking strategy is
# tier-blind by construction and only the bracket-EV arms read the payout table.
strategy-sweep:
	$(PYTHON) -m src.sim.strategy

# The same sweep against the `adp_need` field — disciplined consensus with lineup
# reasoning. The calibration SELECTS need_weight = 0 (the observed ADP curve carries no
# slot-reaching, and it degrades fastest in the elite region), so the fitted field is the
# shipped field bitwise and a sweep at it would measure nothing; this target runs the
# ROBUSTNESS PROBE instead, stipulating an 8-pick lean — the strongest within ~0.14 picks
# of the selected fit — with the noise scale still read from the calibration artifact.
# A different measurement, not a re-decision: artifacts carry the `_adp_need_w8` suffix
# and the shipped strategy_*.csv set is untouched. Requires `make draft-sim-need` first.
strategy-sweep-need:
	$(PYTHON) -m src.sim.strategy --field adp_need --need-weight 8

# The pick-log stake, priced at the stake it would actually be: 20 entries at $1 in
# 15k_and_one — the likely first real entries, whose purpose is capturing pick-log data
# (see `real-pick-logs-are-the-missing-field-calibration`) — drafted three ways on the
# same injected worlds: DK autodraft on the submittable board, the draft room's own
# bracket-EV objective, and the reference tiers' shipped lineup-value arm. Gaps are
# paired on the world and reported in per-entry survival, portfolio P(any advance), and
# DOLLARS on the $20 at risk. Answers "what does clicking 320 picks buy over submitting
# a ranking" for the stake where that trade is actually live.
pick-log-stake:
	$(PYTHON) -m src.sim.strategy --pick-log-stake

# What the availability head's tail-calibration win is worth in the contest — as a PAIRED
# counterfactual rather than a re-read. `docs/availability-window-plan.md` §7k asked this
# once and had to discard the answer, because the sweep it compared against predated the
# window round and moved three things at once. This target reports two arms of the same
# chain, captured under the same code; it does not run them.
#
# Running them is two passes over five targets, differing in ONE config key. Only the
# availability group of `make posteriors` is refitted, which is what makes a pass an hour:
#
#   # in configs/default.yaml: stan.availability.mixture: false
#   $(PYTHON) -m src.models.posteriors --window train --groups availability
#   make simulate-season bracket draft-sim strategy-sweep
#   $(PYTHON) -m src.sim.mixture_value --capture single
#   # then the same five with `mixture: true` and `--capture mixture`
#   make mixture-value
#
# `--capture` refuses when the config key and the arm name disagree, since capturing the
# artifacts of one arm under the other's name is precisely the confound being removed.
# READ `resolution` BEFORE the contest block: 500 worlds per season resolves a lift gap of
# ~0.09, and the simulated lift is SELF-SCORED — each arm is measured in a world it
# generated, so only the realized rows share a truth across arms.
mixture-value:
	$(PYTHON) -m src.sim.mixture_value

# What the PRESEASON BLOCK is worth in the contest — `mixture-value`'s device one round
# over, and the measurement `docs/preseason-plan.md` P5 could not take. That run moved the
# composition's posterior, `sim.minutes.player_season_sigma`, the ADP field and Gate C's
# injection in one pass and overwrote the previous `strategy_*.csv`, so its 0.1890 -> 0.2358
# lift is the chain under the new heads and not the block's own contribution. This target
# reports two arms captured under the same code; it does not run them.
#
# Running them is two passes over five targets, differing in the THREE keys that are the
# block. All three head groups are refitted, which is what makes a pass ~3.5 h — the
# composition alone is 2 h:
#
#   # in configs/default.yaml: stan.availability.preseason: false,
#   #   stan.minutes.preseason: false, stan.composition.preseason.adopt: false
#   $(PYTHON) -m src.models.posteriors --window train \
#       --groups availability,minutes,composition
#   make simulate-season bracket draft-sim strategy-sweep
#   $(PYTHON) -m src.sim.preseason_contest --capture base
#   # then the same five with all three back to true and `--capture preseason`
#   make preseason-contest
#
# Run the COUNTERFACTUAL first so the shipped arm is what disk ends on. `--capture` refuses
# unless all three keys agree with the arm name: a pass with the block half on is neither
# arm, and it is the one mistake that leaves every number in the artifact plausible.
#
# ONLY TWO OF THE THREE HEADS CAN REACH THIS READOUT. `src/sim/` never loads the marginal
# minutes head — it takes `beta_shapes` (arithmetic) from that module and nothing else — so
# P3's block, the largest of the three by its own gate, is structurally invisible here. Its
# key is flipped and its posterior refitted anyway so the arm name means what it says; the
# `reach` block reports which windows actually moved, and a test pins the import fact.
#
# READ `resolution` BEFORE the contest block: 500 worlds per season resolves a lift gap of
# ~0.09, and the simulated lift is SELF-SCORED. Then read `board` — this block arrives as
# the allocation MEAN rather than as shape, so a ranking is the thing it can move.
preseason-contest:
	$(PYTHON) -m src.sim.preseason_contest

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

# `.streamlit/config.toml` rendered from `dashboard/theme.py`, so the page chrome and
# the chart surfaces are one palette rather than two copies of it. Run it after editing
# `THEMES`; a test parses the checked-in file back against the module and fails if the
# two have drifted, which is what makes this a regeneration rather than a suggestion.
dashboard-config:
	$(PYTHON) -m dashboard.theme

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
