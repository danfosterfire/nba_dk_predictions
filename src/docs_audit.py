"""`make docs-audit` — check every quoted figure in the plan docs against its artifact.

This repo has been bitten twice by prose drifting away from the artifact behind it, both
times silently and both times in a table that had been *partially* refreshed:

- the season-total R² column, hand-typed and never recomputed when the MAE side moved;
- the report-calibration block, left behind when the box-score backfill closed 331
  previously uncovered rows.

Neither was catchable by reading, because a stale number looks exactly like a fresh one.
This module makes "the docs agree with the artifacts" a number.

## The three checks

Each `Claim` names a literal string as it appears in a doc, and a lookup into an artifact.

1. **Value** — the quoted figure equals the artifact value, to the precision it is quoted
   at. `"22.7"` is checked to ±0.05, `"0.0635"` to ±0.00005. A figure quoted to two
   decimals is not held to four.
2. **Presence** — the quoted string still appears in the doc. Without this the registry
   rots the moment somebody edits the prose, which is the same failure one level up.
   Proven live: reverting this doc by accident during development lit up 39 stale claims,
   naming every corrected figure that had been lost.
3. **Coverage** — how many measured figures in each doc *no* claim covers, so "how much of
   this document is audited" is a number rather than an impression.

Value mismatches are **failures** and exit non-zero: a doc disagreeing with an artifact is
an unambiguous defect, not a matter of taste. Missing artifacts are *skipped* rather than
failed, so a fresh checkout without `make eda` does not report a wall of red. Coverage is
reported and never fails.

## Historical claims — figures that must NOT agree with the artifact

Some quoted figures are **deliberately superseded** and must survive exactly as written:

- a corrected value preserved beside its correction ("corrected 2026-07-30 from 0.664 /
  0.838 / 0.922"), which is the record of a reversal;
- a scratch-session measurement kept beside the promoted one, which `docs/adp-plan.md` is
  built on — it quotes what was measured before planning *and* what implementing it
  changed, on different populations, and **both are correct**.

`Claim(historical=True)` covers these. It is **excluded from the value check** — the whole
point is that it disagrees — but still **presence-checked**, so the record cannot be
silently deleted while the registry keeps passing. That is the asymmetry that matters: the
failure mode for a live figure is "it drifted", and for a historical one it is "somebody
tidied it away". A historical claim still names the artifact that superseded it, and the
report prints the pair, so the reversal stays legible from the audit alone.

Without this field the two kinds are indistinguishable to the presence check, and the only
alternative — leaving them unclaimed — protects nothing.

## Cost figures — sampler wall clock is not a result

The second population that is presence-checked but never value-checked, added 2026-08-09
and keyed on the **artifact column** rather than marked per claim: anything derived from
`COST_COLUMNS` (`wall_clock_s`, `probe_hours`, `fit_seconds`), which today is 31 claims —
every wall clock, every "sampler minutes", every share-of-sweep ratio.

Every other figure here is a property of the data and reproduces exactly at a fixed seed.
A timing is a property of the *machine* and of whatever else is running on it. Re-running
`make stan-minutes` at identical data and seed reproduced every statistical figure to the
digit and moved its four timings by up to 32%, purely because another head was sampling on
the other cores; the gate failed on six figures, none of which carried information about
the model.

The cost of getting this wrong is worse than the noise. It leaves the project in a position
where the only way to make a **gate** pass is to spend sampler hours re-measuring a number
nobody consumes — precisely the refit that was aborted, deliberately, on 2026-08-09. A gate
that can be satisfied only by burning compute on a non-result teaches people to stop
trusting the gate, which costs more than a stale timing ever could.

Two deliberate limits. The rule is **narrow**: a claim reading `max_rhat` or `divergences`
from the very same diagnostics CSV stays a hard failure, which is where an overlong run
from bad geometry actually surfaces now that the timing does not. And it is **automatic**:
because it keys on the column that was read, a timing claim added later inherits it without
anyone remembering to mark it — which is the failure mode a per-claim flag would have.

## What this does and does not catch

It catches the two failures that have actually happened here:

- **Artifact drift** — a target is re-run, its numbers move, the prose does not. The value
  check fires.
- **Claim rot** — the prose is edited and the registry is not, so a claim stops describing
  anything. The presence check fires.

It does **not** catch a doc edited to a different-but-plausible number that still appears
somewhere else in the same file: the value check compares the *registry's* quoted string to
the artifact, not the doc's own text, and presence is satisfied by any occurrence. Anchoring
each claim to a line would close that, at the cost of a registry that churns on every edit.
The honest summary is that this guards the artifact→prose direction tightly and the
prose→artifact direction loosely, and the loose direction is the one a human review catches.

Run with `make docs-audit`, or `python -m src.docs_audit`.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

AVAIL = "docs/availability-plan.md"
COMP = "docs/minutes-composition-plan.md"
PRED = "docs/predictions-plan.md"
ADP = "docs/adp-plan.md"
SHOT = "docs/shot-attempt-basis-plan.md"
README = "README.md"
GAMES = "docs/games-played-plan.md"
MWIN = "docs/minutes-window-plan.md"        # the marginal minutes head's window ladder
# The availability head's window / season-term / dispersion / likelihood round. Only its
# §7l block is claimed — the paired contest counterfactual — because that is the section
# with a live artifact behind it. The ladders in §3-§7 write their own CSVs and are a
# larger job; entering the doc here at all is what makes adding them incremental.
AWIN = "docs/availability-window-plan.md"

# `CLAUDE.md` held the established-facts section until 2026-08-08, when it was split
# across the five docs below and reduced to a router. Nothing claims `CLAUDE.md` now:
# it carries no measurements, so there is nothing to drift. The figures did not move
# as one block — the split was by subject — which is why `_established_facts` names a
# destination per section rather than taking one doc for the whole builder.
FACTS = "docs/facts-archive.md"            # measured facts, not to be re-derived
NOTES = "docs/model-development-notes.md"  # findings from model selection and fitting
QUIRKS = "docs/data-quirks.md"             # raw-data and library behaviour
SPEC = "docs/project-spec.md"              # the spec every workflow reads
SPLIT = "docs/train-validate-test-split.md"
SIMS = "docs/simulations-plan.md"           # the simulation and drafting layer

WEEK_INDEX = "outputs/predictions/weekly_score_index.csv"
WEEK_PERIOD = "outputs/predictions/weekly_score_period.csv"
WEEK_QUANTILE = "outputs/predictions/weekly_score_quantile.csv"
SIM_GATE_A = "outputs/predictions/sim_season_gate_a.csv"

GP_GATE = "outputs/predictions/stan_games_played_gate.csv"
GP_COLLAPSE = "outputs/predictions/stan_games_played_collapse.csv"
GP_SPELLS = "outputs/predictions/stan_games_played_spells.csv"
GP_METRICS = "outputs/predictions/stan_games_played_metrics.csv"
GP_GATES = "outputs/predictions/stan_games_played_gates.csv"
GP_PROCESS = "outputs/predictions/spell_process.csv"
GP_DIAG = "outputs/predictions/stan_games_played_diagnostics.csv"
GP_SHAPE = "outputs/predictions/stan_games_played_spell_shape.csv"

PROFILE = "outputs/eda/availability_profile.csv"
METRICS = "outputs/predictions/availability_metrics.csv"
ABLATION = "outputs/predictions/availability_workload_ablation.csv"
LADDER = "outputs/predictions/availability_ladder_comparison.csv"
NONLIN = "outputs/predictions/availability_nonlinearity.csv"
MIN_NONLIN = "outputs/predictions/availability_minutes_nonlinearity.csv"
STAN_MIN_M = "outputs/predictions/stan_minutes_metrics.csv"
STAN_MIN_D = "outputs/predictions/stan_minutes_dispersion.csv"
STAN_MIN_G = "outputs/predictions/stan_minutes_diagnostics.csv"
STAN_AV_M = "outputs/predictions/stan_availability_metrics.csv"
STAN_AV_D = "outputs/predictions/stan_availability_diagnostics.csv"
STAN_AV_B = "outputs/predictions/stan_availability_board.csv"
SEASON_TOTAL = "outputs/predictions/season_total_metrics.csv"
SEASON_TOTAL_GATE_E = "outputs/predictions/season_total_gate_e.csv"
REPORT_CAL = "outputs/eda/report_calibration.csv"
COMP_M = "outputs/predictions/stan_composition_metrics.csv"
COMP_D = "outputs/predictions/stan_composition_diagnostics.csv"
COMP_P = "outputs/predictions/stan_composition_ppc.csv"
COMP_J = "outputs/predictions/stan_composition_joint_nll.csv"
# The OT tail moved out of `stan-composition` on 2026-08-09 — `src/models/stan_game_length.py`
# owns it now, and the pooled pair survives there as that head's no-fit floor. The claims
# below still live in `docs/minutes-composition-plan.md`, because that is where the
# composition's own record of them is; only the artifact behind them moved.
GL_M = "outputs/predictions/stan_game_length_metrics.csv"
GL_PPC = "outputs/predictions/stan_game_length_ppc.csv"
# The two minutes heads scored at the SEASON unit against each other — `make
# minutes-unification`. One row per arm plus one paired-bootstrap row, which is why the
# delta columns are blank on the arm rows and vice versa.
MIN_UNIF = "outputs/predictions/minutes_unification.csv"
# The marginal head's window x dispersion ladder — `make minutes-window`. The era file
# carries two row shapes in one table: per-season cells (`block` blank) and pooled era
# blocks (`season` blank), so a lookup keyed on one axis can never hit the other.
MWIN_ERA = "outputs/predictions/minutes_window_era.csv"
MWIN_BREAK = "outputs/predictions/minutes_window_break.csv"
MWIN_LADDER = "outputs/predictions/minutes_window.csv"
MWIN_ROLL = "outputs/predictions/minutes_window_rolling.csv"
MWIN_STAKE = "outputs/predictions/minutes_window_stake.csv"
# The availability mixture's PAIRED contest counterfactual — `make mixture-value`. One
# long table over five blocks, keyed on (block, measure, key), carrying BOTH arms in the
# `single` and `mixture` columns. Claiming a delta therefore reads one row rather than
# differencing two artifacts, which is what stops a half-refreshed pair passing.
MIXVAL = "outputs/predictions/availability_mixture_contest.csv"
AREG = "outputs/predictions/availability_regime.csv"
ASHRINK = "outputs/predictions/availability_shrinkage.csv"
ARCONF = "outputs/predictions/availability_regime_confirmation.csv"
ACLUST = "outputs/predictions/availability_clustering.csv"
AEXCH = "outputs/predictions/availability_exchangeability.csv"
# §12's crossed round — `make availability-absence`. Four artifacts, and the split is by
# what each one protects: the 2x2's arm rows, the same 2x2 read as effects with the
# interaction as its own row, the compound's `lambda` profile, and the block's own
# composition. The interaction file is separate rather than folded into the arm table
# because an effect is a difference between two rows and cannot be a column on either.
# §8b's level ladder — `make availability-no-prior`. Two analyses in one file: `level_arm`
# (an arm x split summary) and `level_pairwise` (every ordered pair of arms). The pairwise
# rows are a separate analysis rather than more columns because the runner-up comparison is
# the one that settled the design, and a delta between two arms cannot be a column on either.
ANOP_L = "outputs/predictions/availability_no_design_level.csv"
AABS = "outputs/predictions/availability_absence.csv"
AABS_I = "outputs/predictions/availability_absence_interaction.csv"
AABS_L = "outputs/predictions/availability_absence_lambda.csv"
AABS_B = "outputs/predictions/availability_absence_block.csv"
AABS_R = "outputs/predictions/availability_absence_rolling.csv"
#: §5b's block-window arm, rebuilt inside `make availability-regime` as the
#: control the whole shrinkage comparison is read against.
SPLICE = "splice8__intercept_workload"
GL_DEPTH = "outputs/predictions/stan_game_length_depth.csv"
GL_D = "outputs/predictions/stan_game_length_diagnostics.csv"
COMP_RHO = "outputs/predictions/stan_composition_dispersion.csv"

SERIAL = "outputs/eda/serial_correlation.csv"
RESID = "outputs/eda/residual_correlation.csv"
RATES = "outputs/predictions/component_rate_metrics.csv"
STAN_C_M = "outputs/predictions/stan_component_metrics.csv"
STAN_C_S = "outputs/predictions/stan_component_substitution.csv"
SHOT_SWEEP = "outputs/predictions/stan_component_substitution_sweep.csv"
SHOT_D = "outputs/predictions/stan_component_substitution_sweep_diagnostics.csv"
STAN_C_D = "outputs/predictions/stan_component_diagnostics.csv"
SEASON_EFF = "outputs/eda/season_effects_summary.csv"
SEASON_REGIME = "outputs/eda/season_effects_regimes.csv"
SHOCK_CORR = "outputs/eda/season_effects_shock_correlation.csv"
TERM_M = "outputs/predictions/season_term_metrics.csv"
TERM_TOTAL = "outputs/predictions/season_term_season_total.csv"
TERM_SPREAD = "outputs/predictions/season_term_roster_spread.csv"
TERM_SIGMA = "outputs/predictions/season_term_sigma_vs_league.csv"
TERM_BONUS = "outputs/predictions/season_term_bonus.csv"
SEASON_BIAS = "outputs/eda/season_effects_carry_forward_bias.csv"
SEASON_RATES = "outputs/eda/season_effects_league_rates.csv"
ROSTER_A = "outputs/eda/roster_coverage_profile_tierA.csv"
VARIANCE = "outputs/eda/variance_budget.csv"
PERSIST = "outputs/eda/persistence.csv"
TARGET = "outputs/eda/target_profile.csv"
AGING = "outputs/eda/aging_curves.csv"
GAME_LEN = "outputs/eda/game_length_coverage.csv"
BONUS = "outputs/eda/bonus_calibration.csv"
CONTEXT_A = "outputs/eda/team_context_value_tierA.csv"
OPPONENT_A = "outputs/eda/opponent_matchup_tierA.csv"
ADP_PROFILE = "outputs/eda/adp_profile.csv"
ADP_AUDIT = "outputs/eda/adp_match_audit.csv"
DIAGNOSTICS = "outputs/eda/feature_diagnostics.csv"
MATRIX_A = "data/features/season_matrix_tierA.parquet"
# The tournament economics have no `outputs/` artifact — `dashboard/economics.py`
# derives them at render time from these checked-in raw boards, which are therefore
# the artifact. `table()` reads any CSV path, so no new plumbing is needed.
TOURNAMENTS = "data/raw/dk_best_ball_tournament_metadata.csv"
# The drafting layer, quoted in `README.md` only. `docs/simulations-plan.md` carries the
# same figures at far greater length and is deliberately *not* in this registry — the
# overview is the document that goes stale, which is the argument the README builder makes
# about itself, so the claims live where the drift risk is.
STRATEGY_SWEEP = "outputs/predictions/strategy_sweep.csv"
STRATEGY_SHIPPED = "outputs/predictions/strategy_shipped.csv"
STRATEGY_GATE_D = "outputs/predictions/strategy_gate_d.csv"
STRATEGY_PAIRED = "outputs/predictions/strategy_paired.csv"
# The field-robustness pair, 2026-08-11: the joint (noise, need) calibration and the
# sweep against the stipulated 8-pick lean its suffix names.
GATE_B_NEED = "outputs/predictions/draft_gate_b_need.csv"
SHIPPED_NEED = "outputs/predictions/strategy_shipped_adp_need_w8.csv"


# ── Claims ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Claim:
    """One quoted figure, and where it comes from.

    `quoted` is the literal text in the doc — including its thousands separators, `%`
    or `×` — because that string is what the presence check looks for and what the
    precision is inferred from.

    `historical` marks a figure the doc keeps *because* it was superseded — a corrected
    value beside its correction, or a scratch-session measurement beside the promoted one.
    Those are exempt from the value check and still presence-checked; see the module
    docstring.
    """

    doc: str
    quoted: str
    artifact: str
    actual: Callable[[], float]
    label: str = ""
    tol: float | None = None
    historical: bool = False

    def value(self) -> float:
        """The quoted string as a number, with separators and units stripped."""
        return parse_quoted(self.quoted)

    def tolerance(self) -> float:
        return self.tol if self.tol is not None else implied_tolerance(self.quoted)


_CLEAN = str.maketrans({",": "", "×": "", "%": "", "−": "-", "−": "-"})


def parse_quoted(text: str) -> float:
    return float(text.translate(_CLEAN).strip())


def implied_tolerance(text: str) -> float:
    """Half a unit in the last quoted decimal place — the honest reading of a rounding.

    `"22.7"` tolerates ±0.05; `"12,406"` tolerates ±0.5. A figure is only wrong here if
    no correctly-rounded value could have produced it.
    """
    body = text.translate(_CLEAN).strip()
    decimals = len(body.split(".")[1]) if "." in body else 0
    return 0.5 * 10 ** (-decimals)


def is_percent(text: str) -> bool:
    return "%" in text


# ── Artifact access ───────────────────────────────────────────────────────────

_CACHE: dict[str, pd.DataFrame | None] = {}


def table(rel: str) -> pd.DataFrame | None:
    """Read an artifact once, or return None if it has not been built.

    Parquet as well as CSV, because several ADP facts (the DK board's pool size, the
    persistent-id map's coverage) live only in `data/features/*.parquet` — there is no
    summary CSV in front of them and inventing one to satisfy the auditor would be the
    tail wagging the dog.
    """
    if rel not in _CACHE:
        path = ROOT / rel
        if not path.exists():
            _CACHE[rel] = None
        elif path.suffix == ".parquet":
            _CACHE[rel] = pd.read_parquet(path)
        else:
            _CACHE[rel] = pd.read_csv(path)
    return _CACHE[rel]


# ── Cost figures: presence-checked, never value-checked ──────────────────────
#
# **Sampler wall clock is not a result, and auditing it as one makes this gate lie.**
# Every other figure here is a property of the data and reproduces exactly at a fixed
# seed. A timing is a property of the *machine* and of whatever else was running on it:
# re-running `make stan-minutes` on 2026-08-08 with identical data and seed reproduced
# every statistical figure to the digit and moved its four timings by up to 32%, purely
# because the composition head was sampling on the other cores. The audit failed on six
# figures, none of which carried any information.
#
# The cost of getting this wrong is worse than the noise. It puts the project in a
# position where the only way to make a GATE pass is to spend sampler hours re-measuring
# a number nobody consumes — which is exactly the refit that was aborted, deliberately,
# on 2026-08-09. A gate that can be satisfied only by burning compute on a non-result
# trains people to stop trusting the gate.
#
# So a figure derived from any of these columns is **presence-checked**: the quoted
# string must still appear in the doc, so a cost claim cannot be tidied away, but the
# value is not compared. The rule lives here, once, keyed on the artifact column rather
# than marked per claim — a timing claim added later inherits it without anyone
# remembering to. Overlong runs from bad geometry surface in the diagnostics
# (`treedepth_saturated`, `divergences`), which ARE value-checked and are the honest
# place to catch them.
COST_COLUMNS = frozenset({"wall_clock_s", "probe_hours", "fit_seconds"})

# Columns touched while evaluating the claim currently under check. `check_values`
# clears it before each `actual()` and reads it after.
_columns_read: set[str] = set()


def _note_columns(*columns: str) -> None:
    _columns_read.update(c for c in columns if isinstance(c, str))


def touched_cost_column() -> set[str]:
    return set(_columns_read) & COST_COLUMNS


class MissingColumn(KeyError):
    """A claim asks for a column its artifact no longer has.

    Distinct from "no such row", which is a fresh-checkout condition and is *skipped*.
    A missing column means the artifact's **schema** changed under the claim — a head
    renaming `test_r2` to `val_r2`, say — and the claim has stopped describing anything.
    That has to fail loudly: skipping it would leave dozens of figures reading as audited
    while checking nothing, which is the exact failure `Claim(historical=True)` exists to
    make impossible.
    """


def _one(frame: pd.DataFrame | None, column: str, **where) -> float:
    _note_columns(column)
    if frame is None:
        return float("nan")
    for col in (*where, column):
        if col not in frame.columns:
            raise MissingColumn(col)
    mask = pd.Series(True, index=frame.index)
    for col, val in where.items():
        mask &= frame[col] == val
    hit = frame[mask]
    return float(hit[column].iloc[0]) if len(hit) else float("nan")


def prof(measurement: str, key: str, metric: str, window: str = "full") -> float:
    return _one(table(PROFILE), "value", measurement=measurement, window=window,
                key=key, metric=metric)


def prof_n(measurement: str, key: str, metric: str, window: str = "full") -> float:
    return _one(table(PROFILE), "n", measurement=measurement, window=window,
                key=key, metric=metric)


def metric(rel: str, model: str, name: str, group: str = "all") -> float:
    return _one(table(rel), "value", model=model, metric=name, group=group)


def treatment(name: str, which: str, group: str = "all") -> float:
    return _one(table(SEASON_TOTAL), "value", treatment=name, metric=which, group=group)


def cal(key: str, name: str, measurement: str = "calibration") -> float:
    return _one(table(REPORT_CAL), "value", measurement=measurement, key=key,
                metric=name)


def cell(rel: str, column: str, **where) -> float:
    return _one(table(rel), column, **where)


def total(rel: str, column: str) -> float:
    _note_columns(column)
    frame = table(rel)
    return float(frame[column].sum()) if frame is not None else float("nan")


def rows(rel: str) -> float:
    frame = table(rel)
    return float(len(frame)) if frame is not None else float("nan")


def max_of(rel: str, column: str) -> float:
    _note_columns(column)
    frame = table(rel)
    return float(frame[column].max()) if frame is not None else float("nan")


def nunique(rel: str, column: str) -> float:
    """Distinct values in a column — for a doc that quotes the *size* of a grid.

    `rows()` is the wrong tool where the artifact is long-format: the sweep is one row per
    strategy x tier x season, so its arm count is a `nunique` and not a length.
    """
    _note_columns(column)
    frame = table(rel)
    return float(frame[column].nunique()) if frame is not None else float("nan")


def mean_abs_dev(rel: str, column: str, centre: float, **where) -> float:
    """Mean |value - centre| over the matching rows — a calibration summary that is
    itself a quoted figure, so it has to come from the artifact like any other."""
    _note_columns(column)
    frame = table(rel)
    if frame is None:
        return float("nan")
    mask = pd.Series(True, index=frame.index)
    for col, val in where.items():
        mask &= frame[col] == val
    hit = frame[mask]
    return float((hit[column] - centre).abs().mean()) if len(hit) else float("nan")


def _windowed(rel: str, column: str, fit_window: str = "full", **where) -> float:
    """`_one`, but pinned to a fit window when the artifact carries that axis.

    Several producers now emit every row twice — once over every season (`full`) and once
    excluding the seasons the heads hold out (`train_val`) — because figures that are
    *simulator inputs* must not be calibrated on the backtest seasons. `_one` returns
    `.iloc[0]`, so an unfiltered lookup against such an artifact silently takes whichever
    row sorts first. The windows differ by less than most quoted precisions, which is
    precisely why the wrong read would not announce itself.

    Written defensively: artifacts that have not yet gained the axis are looked up
    unchanged, so this is safe to apply ahead of a regeneration.
    """
    frame = table(rel)
    if frame is not None and "fit_window" in frame.columns:
        return _one(frame, column, fit_window=fit_window, **where)
    return _one(frame, column, **where)


def serial(component: str, column: str, fit_window: str = "full") -> float:
    """One serial-correlation cell, for a named fit window.

    `serial_correlation.py` emits every row twice — once over every season (`full`) and
    once excluding the two the heads hold out (`train_val`), because the block inflation
    is a simulator *input* and calibrating it on the backtest seasons would tune the
    simulator on the seasons it is scored against. **The window has to be named here.**
    Without it `_one` takes `.iloc[0]`, i.e. whichever row sorts first, which is the same
    silent-ambiguity failure the composition PPC lookups were fixed for. The prose in
    `docs/` quotes the full-window figures, so that is the default; the
    two windows differ by up to 0.002 on `lag1_excess`, which is inside most quoted
    precisions and therefore exactly the kind of wrong-row read that would not announce
    itself.
    """
    return _windowed(SERIAL, column, fit_window, component=component)


def resid(column: str, kind: str | None = None,
          basis: str = "minutes_conditioned", fit_window: str = "full") -> float:
    """Off-diagonal summaries of the residual correlation matrix.

    Both ordered pairs are in the artifact, so a mean over off-diagonals is already
    symmetric-weighted. `kind` restricts to the seven counts, which is the population the
    prose quotes — the notes report the all-eleven mean beside it and the two differ by
    nearly 2×, so the restriction is load-bearing rather than cosmetic.
    """
    frame = table(RESID)
    if frame is None:
        return float("nan")
    off = frame[(frame["basis"] == basis)
                & (frame["component_a"] != frame["component_b"])]
    if "fit_window" in off.columns:
        off = off[off["fit_window"] == fit_window]
    if kind is not None:
        off = off[(off["kind_a"] == kind) & (off["kind_b"] == kind)]
    return float(off["r"].mean() if column == "mean" else off["r"].max())


def resid_min_eig(basis: str = "minutes_conditioned",
                  fit_window: str = "full") -> float:
    """Smallest eigenvalue of the conditioned matrix — the copula's usability gate.

    Recomputed from the long form here rather than read from a column, because the
    artifact does not carry one: it is a property of the whole matrix, not of a pair.
    """
    frame = table(RESID)
    if frame is None:
        return float("nan")
    sub = frame[frame["basis"] == basis]
    if "fit_window" in sub.columns:
        sub = sub[sub["fit_window"] == fit_window]
    wide = sub.pivot(index="component_a", columns="component_b", values="r")
    wide = wide.reindex(index=wide.columns)
    return float(np.linalg.eigvalsh(wide.fillna(0.0).to_numpy(float)).min())


def rate(head: str, variant: str, column: str = "r2",
         analysis: str = "variant_sweep") -> float:
    return _one(table(RATES), column, analysis=analysis, head=head, variant=variant)


def stan_c(head: str, variant: str, column: str) -> float:
    return _one(table(STAN_C_M), column, head=head, variant=variant)


# `subst_test` lived here until 2026-08-06. It recovered the handicapped substitution
# margin's TEST half from the Gate 0 sweep after `stan_component_substitution.csv` went
# validation-only, so that a superseded figure stayed value-checked rather than becoming a
# quotation. Then `make stan-substitution` went validation-only too and the last copy went
# with it. The lesson is worth the comment: preserving a superseded measurement by pointing
# at a second artifact only holds until that artifact is legitimately rebuilt, and the
# answer then is to demote the claim to `historical=True`, not to freeze the run.


def season_eff(quantity: str, column: str) -> float:
    return _one(table(SEASON_EFF), column, quantity=quantity)


def term(head: str, arm: str, column: str) -> float:
    """One season-term ablation cell. `arm` is never optional: `base` and `trend` are
    different models and quoting a number without naming the arm is the same mistake as
    quoting a variance-budget share without its basis."""
    return _one(table(TERM_M), column, head=head, arm=arm)


def _alpha_loss(variant: str, head: str | None = None) -> float:
    """R2 lost between the shipped alpha and alpha=1.0, median over heads or for one head.

    The alpha curve is a regression guard rather than a finding, so what the docs quote is
    a summary of it — and a summary of a guard has to be derived from the guard, not typed
    beside it, or it stops tracking the thing it describes.
    """
    frame = table(RATES)
    if frame is None:
        return float("nan")
    sweep = frame[(frame["analysis"] == "alpha_sensitivity")
                  & (frame["variant"] == variant)]
    lo = sweep[sweep["alpha"] == sweep["alpha"].min()].set_index("head")["r2"]
    hi = sweep[sweep["alpha"] == 1.0].set_index("head")["r2"]
    loss = (lo - hi).dropna()
    return float(loss[head]) if head else float(loss.median())


def oracle_gain(head: str) -> float:
    """The perfect-league-override ceiling as a FRACTION of the base arm's MAE.

    A fraction rather than a percentage because the quoted strings carry `%`, which the
    value check re-multiplies. This is the number that bounds every form of season term.

    Read from `val_mae` since 2026-08-07. The ablation no longer scores the held-out
    seasons at all (`src/models/held_out.py`), so the ceiling is now measured on the rows
    that select — which is the right frame for it in any case: a bound on what a season
    term could buy is only useful beside the arms it bounds.
    """
    base = term(head, "base", "val_mae")
    return (base - term(head, "oracle_league", "val_mae")) / base


def term_total(arm: str, column: str) -> float:
    return _one(table(TERM_TOTAL), column, arm=arm)


def term_spread(n_players: int, arm: str,
                column: str = "inflation_vs_base") -> float:
    return _one(table(TERM_SPREAD), column, n_players=n_players, arm=arm)


def term_sigma(head: str, column: str) -> float:
    return _one(table(TERM_SIGMA), column, head=head)


def term_bonus(arm: str, column: str = "bonus_bias_pct") -> float:
    return _one(table(TERM_BONUS), column, arm=arm)


def regime(series: str, column: str, arm: str = "policy_break_level") -> float:
    """One regime-test row. `arm` defaults to the LEVEL-only break, not the level+slope
    one: with three post-break seasons the slope arm's extrapolation is noise, so quoting
    it without naming the arm would be the same class of error as quoting a variance-budget
    share without its basis."""
    return _one(table(SEASON_REGIME), column, series=series, arm=arm)


def carry_bias(component: str, season: str) -> float:
    return _one(table(SEASON_BIAS), "bias_pct", component=component, season=season)


def carry_league_move(component: str, season: str) -> float:
    """That season's own league move, joined onto the bias cell it should oppose.

    Read from the bias artifact rather than from `season_effects_league_rates.csv` so the
    doc's side-by-side table is checked against the *joined* pair. A claim that took the
    two halves from two files could pass while the join between them was broken, which is
    the failure the `opposes_league_move` column exists to make visible.
    """
    return _one(table(SEASON_BIAS), "league_yoy_pct", component=component, season=season)


def _lag_cells() -> pd.DataFrame:
    """The per-season bias cells — the `all` rows carry no league move by construction."""
    t = table(SEASON_BIAS)
    return t[t["opposes_league_move"].notna()]


def _lag_opposing(total: bool = False) -> float:
    cells = _lag_cells()
    return float(len(cells)) if total else float(cells["opposes_league_move"].sum())


def _lag_corr() -> float:
    cells = _lag_cells()
    return float(np.corrcoef(cells["bias_pct"], cells["league_yoy_pct"])[0, 1])


def roster(key: str, metric: str = "share_of_minutes",
           window: str = "season_start", scope: str = "league") -> float:
    return _one(table(ROSTER_A), "value", window=window, scope=scope, key=key,
                metric=metric)


def budget(source: str, column: str = "share_of_variance", **where) -> float:
    """A variance-budget row. `null_construction` disambiguates the two shuffled nulls,
    which share a `source` and differ by 3× — quoting one without naming the marginal is
    a recorded past error, so the accessor makes naming it the only way to reach them."""
    return _one(table(VARIANCE), column, source=source, **where)


def persist(feature: str, column: str = "r_within_season",
            tier: str = "A") -> float:
    """One persistence row. `minutes_weighted` is a *property* of the column rather than a
    choice — counts are weighted and percentages are not — so it is never filtered on."""
    return _one(table(PERSIST), column, tier=tier, feature=feature)


def ladder(model: str, column: str, group: str = "all") -> float:
    """One cell of the availability ladder's paired comparison.

    `group` is required to reach a quartile row rather than defaulting into it, because
    "the GBM is 0.13 better" and "the GBM is 0.33 worse where it matters" are both true of
    this artifact and the whole point is that they are different rows.
    """
    return _one(table(LADDER), column, model=model, group=group)


def adp(section: str, metric: str) -> float:
    return _one(table(ADP_PROFILE), "value", section=section, metric=metric)


# ── The claim registry ────────────────────────────────────────────────────────

def _c(quoted: str, artifact: str, actual: Callable[[], float], label: str,
       doc: str = AVAIL, tol: float | None = None,
       historical: bool = False) -> Claim:
    return Claim(doc=doc, quoted=quoted, artifact=artifact, actual=actual,
                 label=label, tol=tol, historical=historical)


def _availability() -> list[Claim]:
    """`docs/availability-plan.md` — the availability, minutes and season-total heads."""
    C: list[Claim] = []
    add = C.append

    # ── header ────────────────────────────────────────────────────────────────
    add(_c("284", PROFILE, lambda: rows(PROFILE), "availability_profile row count"))
    # The pre-`*_rotation` count, kept beside its replacement: the 24 new rows measure the
    # transition structure on the frame the overdispersion figure uses, and the whole point
    # of that addition is that the two were never the same frame.
    add(_c("260", PROFILE, lambda: rows(PROFILE),
           "availability_profile row count before the rotation keys", historical=True))
    add(_c("1,314,238", PROFILE, lambda: prof("window_bracket", "panel", "rows"),
           "panel player-games"))

    # ── the internal ceiling ──────────────────────────────────────────────────
    ceiling = [("prior_gp_share", "0.159", "0.079"), ("prior_mpg", "0.159", "0.057"),
               ("gp_lags_1_3", "0.184", "0.092"), ("gp_and_mpg_lag1", "0.214", "0.104"),
               ("gp_and_mpg_lags_1_3", "0.222", "0.111"),
               ("plus_age_and_career", "0.236", "0.116")]
    for key, unw, wtd in ceiling:
        add(_c(unw, PROFILE,
               lambda k=key: prof("predictor_r2", k, "r2_in_sample_unweighted"),
               f"ceiling {key} unweighted"))
        add(_c(wtd, PROFILE,
               lambda k=key: prof("predictor_r2", k, "r2_in_sample_weighted"),
               f"ceiling {key} weighted"))
    add(_c("7,276", PROFILE,
           lambda: prof_n("predictor_r2", "prior_gp_share", "r2_in_sample_unweighted"),
           "ceiling population"))

    # ── persistence cross-check ───────────────────────────────────────────────
    for key, val in [("gp_share", "0.317"), ("minutes_per_game", "0.779"),
                     ("total_minutes", "0.641")]:
        add(_c(val, PROFILE, lambda k=key: prof("persistence", k, "r_within_weighted"),
               f"persistence {key}"))

    # ── the two nulls ─────────────────────────────────────────────────────────
    add(_c("0.392", PROFILE, lambda: prof("multiyear", "gp_share_mean3", "r_unweighted"),
           "3-year average, unweighted"))
    add(_c("0.398", PROFILE, lambda: prof("multiyear", "gp_share_lag1", "r_unweighted"),
           "1-year prior, unweighted"))
    add(_c("0.090", PROFILE,
           lambda: prof("persistence", "longest_spell", "r_within_weighted"),
           "longest spell persistence"))
    add(_c("0.088", PROFILE,
           lambda: prof("persistence", "long_spells", "r_within_weighted"),
           "long spells persistence"))
    add(_c("0.407", PROFILE, lambda: prof("persistence", "n_spells", "r_within_weighted"),
           "spell count persistence"))

    # ── overdispersion ────────────────────────────────────────────────────────
    rot = "rotation_players"
    add(_c("5,267", PROFILE, lambda: prof("overdispersion", rot, "n"),
           "rotation population, full window"))
    add(_c("0.803", PROFILE, lambda: prof("overdispersion", rot, "mean_gp_share"),
           "mean gp_share"))
    add(_c("0.209", PROFILE, lambda: prof("overdispersion", rot, "sd_gp_share"),
           "observed sd"))
    add(_c("0.044", PROFILE, lambda: prof("overdispersion", rot, "binomial_sd"),
           "binomial sd"))
    add(_c("22.7", PROFILE, lambda: prof("overdispersion", rot, "variance_ratio"),
           "overdispersion, full window"))
    add(_c("19.8", PROFILE,
           lambda: prof("overdispersion", rot, "variance_ratio", "appearance"),
           "overdispersion, appearance window"))
    add(_c("5,639", PROFILE, lambda: prof("overdispersion", rot, "n", "appearance"),
           "rotation population, appearance window"))
    hist = [("115", ["gp_0_8", "gp_8_16"]), ("206", ["gp_16_24", "gp_24_32"]),
            ("404", ["gp_32_40", "gp_40_48"]), ("985", ["gp_48_56", "gp_56_64"]),
            ("2,400", ["gp_64_72", "gp_72_80"]), ("1,157", ["gp_80_88"])]
    for quoted, keys in hist:
        add(_c(quoted, PROFILE,
               lambda ks=keys: sum(prof("overdispersion", k, "count") for k in ks),
               f"GP histogram bin {quoted}"))
    add(_c("26.7%", PROFILE,
           lambda: prof("overdispersion", rot, "share_below_60_games"),
           "share below 60 games"))
    add(_c("9.6%", PROFILE, lambda: prof("overdispersion", rot, "share_below_41_games"),
           "share below 41 games"))

    # ── spells ────────────────────────────────────────────────────────────────
    add(_c("68,530", PROFILE,
           lambda: prof("spell_distribution", "all", "n_spells", "appearance"),
           "absence spells, appearance window"))
    add(_c("48.3%", PROFILE,
           lambda: prof("spell_distribution", "all", "share_single_game", "appearance"),
           "single-game spells, appearance"))
    for key, quoted in [("p50", "2"), ("p75", "3"), ("p90", "7"), ("p99", "26")]:
        add(_c(quoted, PROFILE,
               lambda k=key: prof("spell_distribution", k, "spell_games", "appearance"),
               f"spell {key}, appearance"))
    add(_c("36.2%", PROFILE,
           lambda: prof("spell_distribution", "all",
                        "share_missed_games_in_spells_ge10", "appearance"),
           "missed games in 10+ spells, appearance"))
    add(_c("42.6%", PROFILE,
           lambda: prof("spell_distribution", "all", "share_single_game"),
           "single-game spells, full"))
    add(_c("72.4%", PROFILE,
           lambda: prof("spell_distribution", "all",
                        "share_missed_games_in_spells_ge10"),
           "missed games in 10+ spells, full"))

    # ── carryover ─────────────────────────────────────────────────────────────
    carry = [("0", "7,697", "0.647", "0.714"), ("1-2", "1,454", "0.629", "0.681"),
             ("3-5", "507", "0.573", "0.634"), ("6-10", "458", "0.573", "0.625"),
             ("11-20", "440", "0.530", "0.592"), ("21+", "716", "0.457", "0.516")]
    for key, n, play, gp in carry:
        add(_c(play, PROFILE,
               lambda k=key: prof("carryover", k, "next_season_start_play_rate"),
               f"carryover {key} play rate"))
        add(_c(gp, PROFILE,
               lambda k=key: prof("carryover", k, "next_season_gp_share"),
               f"carryover {key} gp share"))
        add(_c(n, PROFILE,
               lambda k=key: prof_n("carryover", k, "next_season_gp_share"),
               f"carryover {key} n"))
    add(_c("0.1007", PROFILE,
           lambda: prof("carryover", "incremental", "r2_prior_gp_share"),
           "carryover base R2"))
    add(_c("0.1018", PROFILE,
           lambda: prof("carryover", "incremental", "r2_plus_trailing_missed"),
           "carryover + trailing R2"))

    # ── the two roster windows ────────────────────────────────────────────────
    add(_c("959,069", PROFILE,
           lambda: prof("window_bracket", "panel", "rows", "appearance"),
           "appearance window rows"))
    add(_c("0.768", PROFILE,
           lambda: prof("window_bracket", "panel", "played_rate", "appearance"),
           "appearance played rate"))
    add(_c("0.560", PROFILE, lambda: prof("window_bracket", "panel", "played_rate"),
           "full window played rate"))
    add(_c("6.27", PROFILE,
           lambda: prof("window_bracket", "season_edge", "mean_trailing_missed"),
           "mean trailing missed"))
    add(_c("14,569", PROFILE,
           lambda: prof_n("window_bracket", "season_edge", "mean_trailing_missed"),
           "player-seasons"))

    # ── serial structure ──────────────────────────────────────────────────────
    ap = "appearance"
    add(_c("942,597", PROFILE,
           lambda: prof_n("serial_structure", "transition", "p_play_given_played", ap),
           "transitions"))
    add(_c("0.905", PROFILE,
           lambda: prof("serial_structure", "transition", "p_play_given_played", ap),
           "P(play | played)"))
    add(_c("0.308", PROFILE,
           lambda: prof("serial_structure", "transition", "q_play_given_missed", ap),
           "P(play | missed)"))
    add(_c("0.597", PROFILE,
           lambda: prof("serial_structure", "transition", "lag1_autocorrelation", ap),
           "lag-1 autocorrelation"))
    add(_c("3.96", PROFILE,
           lambda: prof("serial_structure", "markov",
                        "clustering_variance_inflation", ap),
           "Markov variance inflation"))
    add(_c("0.764", PROFILE,
           lambda: prof("serial_structure", "markov", "stationary_play_rate", ap),
           "stationary play rate"))
    add(_c("0.483", PROFILE,
           lambda: prof("serial_structure", "spell_shape", "share_single_observed", ap),
           "spells of 1, observed"))
    add(_c("0.0635", PROFILE,
           lambda: prof("serial_structure", "spell_shape", "share_ge10_observed", ap),
           "spells of 10+, observed"))
    add(_c("0.0365", PROFILE,
           lambda: prof("serial_structure", "spell_shape", "share_ge10_geometric", ap),
           "spells of 10+, geometric"))
    add(_c("3.25", PROFILE,
           lambda: prof("serial_structure", "spell_shape", "mean_spell_observed", ap),
           "mean spell"))

    # ── the MATCHED frame, which is a different population AND a different window ──
    # The rows above are the appearance window over all players; these are the full window
    # over established rotation players — the frame `overdispersion` quotes 22.7x on. The
    # two were being divided into each other, which is why both now ship.
    add(_c("0.9443", PROFILE,
           lambda: prof("serial_structure", "transition_rotation",
                        "p_play_given_played"),
           "P(play | played), matched population"))
    add(_c("0.1333", PROFILE,
           lambda: prof("serial_structure", "transition_rotation",
                        "q_play_given_missed"),
           "P(play | missed), matched population"))
    add(_c("9.58", PROFILE,
           lambda: prof("serial_structure", "markov_rotation",
                        "clustering_variance_inflation"),
           "clustering C, matched population"))
    add(_c("9.581", PROFILE,
           lambda: prof("serial_structure", "markov_rotation",
                        "clustering_variance_inflation"),
           "clustering C, matched population (3dp)"))
    # The additive identity, which is arithmetic on two artifact values rather than a
    # third measurement: rho = (I - C)/(n - C) and the stacked inflation C + rho*(n - C).
    add(_c("0.181", PROFILE,
           lambda: ((prof("overdispersion", "rotation_players", "variance_ratio")
                     - prof("serial_structure", "markov_rotation",
                            "clustering_variance_inflation"))
                    / (82.0 - prof("serial_structure", "markov_rotation",
                                   "clustering_variance_inflation"))),
           "residual frailty rho the chain must not double-count"))
    add(_c("29.55", PROFILE,
           lambda: (prof("serial_structure", "markov_rotation",
                         "clustering_variance_inflation")
                    + 0.2757 * (82.0 - prof("serial_structure", "markov_rotation",
                                            "clustering_variance_inflation"))),
           "stacking the incumbent rho on the measured clustering"))
    add(_c("30.2%", PROFILE,
           lambda: ((prof("serial_structure", "markov_rotation",
                          "clustering_variance_inflation")
                     + 0.2757 * (82.0 - prof("serial_structure", "markov_rotation",
                                             "clustering_variance_inflation")))
                    / prof("overdispersion", "rotation_players", "variance_ratio")
                    - 1.0),
           "how far stacking overshoots the measured overdispersion"))

    # ── the absence-reason decomposition ──────────────────────────────────────
    add(_c("7,673", PROFILE,
           lambda: prof("decomposition", "coverage", "usable_pairs"),
           "usable season pairs"))
    add(_c("0.696", PROFILE,
           lambda: prof("decomposition", "coverage", "mean_status_coverage"),
           "mean status coverage"))
    incremental = [("0.2353", "r2_prior_gp_share"),
                   ("0.2365", "r2_plus_missed_games"),
                   ("0.2648", "r2_plus_reason_split"),
                   ("0.2363", "r2_reason_split_shuffled_null"),
                   ("0.00041", "r2_reason_split_null_sd"),
                   ("0.0285", "r2_reason_split_above_null")]
    for quoted, name in incremental:
        add(_c(quoted, PROFILE,
               lambda n=name: prof("decomposition", "incremental", n),
               f"reason split {name}"))
    reasons = [("missed_scratch", "−0.353", "0.469"),
               ("missed_inactive", "−0.233", "0.208"),
               ("missed_not_rostered", "−0.164", "0.136"),
               ("missed_injury", "+0.034", "0.168"),
               ("missed_games", "−0.339", "0.233")]
    for key, corr, persistence in reasons:
        add(_c(corr.lstrip("+"), PROFILE,
               lambda k=key: prof("decomposition", k, "r_vs_next_gp_share"),
               f"{key} vs next gp_share"))
        add(_c(persistence, PROFILE,
               lambda k=key: prof("decomposition", k, "r_persistence_of_column"),
               f"{key} persistence"))

    # ── report calibration ────────────────────────────────────────────────────
    add(_c("12,338", REPORT_CAL, lambda: cal("all", "scored_rows", "coverage"),
           "report rows scored"))
    add(_c("12,406", REPORT_CAL, lambda: cal("all", "report_rows", "coverage"),
           "report rows total"))
    add(_c("142", REPORT_CAL, lambda: cal("all", "game_dates", "coverage"),
           "report game dates"))
    add(_c("532", REPORT_CAL, lambda: cal("all", "players", "coverage"),
           "report players"))
    designations = [
        ("Out", "8,539", "0.002", "0.094", "0.898", "0.006", "11.0"),
        ("Doubtful", "499", "0.030", "0.222", "0.747", "0.000", "16.7"),
        ("Questionable", "1,788", "0.498", "0.199", "0.303", "0.001", "23.5"),
        ("Probable", "602", "0.914", "0.066", "0.020", "0.000", "24.7"),
        ("Available", "910", "0.855", "0.118", "0.024", "0.003", "23.8"),
    ]
    for name, n, play, dnp, inactive, absent, mins in designations:
        key = f"{name}|all"
        for quoted, col in [(n, "n"), (play, "p_played"), (dnp, "p_dnp"),
                            (inactive, "p_inactive"), (absent, "p_absent"),
                            (mins, "mean_min_given_played")]:
            add(_c(quoted, REPORT_CAL, lambda k=key, c=col: cal(k, c),
                   f"{name} {col}"))
    add(_c("98.5%", REPORT_CAL, lambda: cal("Out", "p_unchanged", "revision"),
           "Out unchanged next day"))
    add(_c("0.475", REPORT_CAL, lambda: cal("Questionable", "p_played", "revision"),
           "stale Questionable p_play"))
    add(_c("12.5%", REPORT_CAL,
           lambda: cal("Out|Not With Team", "p_absent", "by_reason"),
           "Out|Not With Team absent"))
    add(_c("14.3%", REPORT_CAL,
           lambda: cal("Out|Trade Pending", "p_absent", "by_reason"),
           "Out|Trade Pending absent"))

    # ── playoff scope ─────────────────────────────────────────────────────────
    playoffs = [("bench_lt12", "0.505", "0.711"),
                ("rotation_12_24", "0.761", "0.905"),
                ("starter_24plus", "1.054", "0.958")]
    for key, ratio, appearance in playoffs:
        add(_c(ratio, PROFILE,
               lambda k=key: prof("playoff_scope", k, "median_playoff_to_regular_mpg"),
               f"playoff MPG ratio, {key}"))
        add(_c(appearance, PROFILE,
               lambda k=key: prof("playoff_scope", k, "playoff_appearance_rate"),
               f"playoff appearance rate, {key}"))

    # ── stage E baselines ─────────────────────────────────────────────────────
    # Validation-only since 2026-08-08, when the last held-out head was converted. The
    # ORDERING reversed on the move — the GBM leads on mean CRPS and the GLM still ships —
    # so the retired column is preserved below and the paired interval that settles it is
    # claimed as its own block.
    C += _availability_ladder_claims(AVAIL)
    add(_c("15.5%", METRICS,
           lambda: metric(METRICS, "beta_binomial", "predicted_share_below_41",
                          "rotation"), "GLM predicted below 41"))
    add(_c("35.2%", METRICS,
           lambda: metric(METRICS, "beta_binomial", "predicted_share_below_60",
                          "rotation"), "GLM predicted below 60"))
    add(_c("10.0%", METRICS,
           lambda: metric(METRICS, "beta_binomial", "observed_share_below_41",
                          "rotation"), "observed below 41"))
    add(_c("32.6%", METRICS,
           lambda: metric(METRICS, "beta_binomial", "observed_share_below_60",
                          "rotation"), "observed below 60"))
    add(_c("24.1%", METRICS,
           lambda: metric(METRICS, "league_age", "predicted_share_below_41",
                          "rotation"), "baseline predicted below 41"))
    add(_c("43.9%", METRICS,
           lambda: metric(METRICS, "league_age", "predicted_share_below_60",
                          "rotation"), "baseline predicted below 60"))
    for quoted, label in [("15.0%", "GLM predicted below 41"),
                          ("34.8%", "GLM predicted below 60"),
                          ("11.8%", "observed below 41"),
                          ("36.9%", "observed below 60"),
                          ("24.8%", "baseline predicted below 41"),
                          ("45.0%", "baseline predicted below 60"),
                          ("0.268", "held-out R2")]:
        add(_c(quoted, METRICS, lambda: float("nan"),
               f"pre-lock held-out tail: {label}", historical=True))

    # ── workload ablation ─────────────────────────────────────────────────────
    C += _workload_ablation_claims(AVAIL)

    # ── minutes nonlinearity probe ────────────────────────────────────────────
    # The `test_r2` / `test_vs_linear` columns went with the conversion, and with them the
    # `replicates` flag. The val column reproduced to four decimals, which is a determinism
    # check on the move rather than evidence for the finding.
    probes = [("linear", "0.6768"), ("quadratic", "0.6914"), ("spline_k4", "0.6930")]
    for name, val in probes:
        add(_c(val, MIN_NONLIN,
               lambda n=name: cell(MIN_NONLIN, "val_r2", scope="variant", name=n),
               f"MPG probe {name} val R2"))
    add(_c("0.0124", MIN_NONLIN,
           lambda: cell(MIN_NONLIN, "val_vs_linear", scope="column",
                        name="minutes_per_game_lag1"), "prior-MPG spline, val"))
    add(_c("0.0008", MIN_NONLIN,
           lambda: cell(MIN_NONLIN, "val_vs_linear", scope="column",
                        name="total_minutes_lag1"), "total-minutes spline, val"))
    add(_c("−0.0008", MIN_NONLIN,
           lambda: cell(MIN_NONLIN, "val_vs_linear", scope="column", name="age"),
           "age spline, val"))
    for quoted, label in [("0.6670", "linear test R2"), ("0.6746", "quadratic test R2"),
                          ("0.6736", "spline_k4 test R2"),
                          ("0.0077", "prior-MPG spline, test")]:
        add(_c(quoted, MIN_NONLIN, lambda: float("nan"),
               f"pre-lock held-out MPG probe: {label}", historical=True))

    # ── the minutes head ──────────────────────────────────────────────────────
    # Validation-only since `make stan-minutes` was re-run under the held-out lock
    # (2026-08-06). `bias` is claimed per arm rather than left in prose because the *level*
    # of the bias moved with the split while the fitted-minus-floor **gap** reproduced, and
    # a prose range cannot express that distinction — see `_MINUTES_PRE_LOCK`.
    variants = [("carry_forward", "161.45", "0.8536", "213.13", "+23.91"),
                ("linear", "144.71", "0.8826", "199.54", "−3.26"),
                ("logit_own", "145.45", "0.8819", "200.90", "−2.97"),
                ("logit_own_quadratic", "144.54", "0.8827", "200.84", "−13.82"),
                ("logit_own_spline", "143.93", "0.8835", "199.60", "−14.00")]
    for name, crps, r2, mae, bias in variants:
        add(_c(crps, STAN_MIN_M,
               lambda n=name: cell(STAN_MIN_M, "val_crps", variant=n),
               f"minutes {name} val CRPS"))
        add(_c(r2, STAN_MIN_M,
               lambda n=name: cell(STAN_MIN_M, "val_r2", variant=n),
               f"minutes {name} val R2"))
        add(_c(mae, STAN_MIN_M,
               lambda n=name: cell(STAN_MIN_M, "val_mae", variant=n),
               f"minutes {name} val MAE"))
        add(_c(bias, STAN_MIN_M,
               lambda n=name: cell(STAN_MIN_M, "val_bias", variant=n),
               f"minutes {name} val bias"))
    add(_c("+0.0299", STAN_MIN_M,
           lambda: (cell(STAN_MIN_M, "val_r2", variant="logit_own_spline")
                    - cell(STAN_MIN_M, "val_r2", variant="carry_forward")),
           "minutes R2 gain over floor"))
    add(_c("−17.5", STAN_MIN_M,
           lambda: (cell(STAN_MIN_M, "val_crps", variant="logit_own_spline")
                    - cell(STAN_MIN_M, "val_crps", variant="carry_forward")),
           "minutes CRPS gain over floor"))
    add(_c("1,503", STAN_MIN_G, lambda: total(STAN_MIN_G, "wall_clock_s"),
           "minutes head wall clock"))
    add(_c("1.0054", STAN_MIN_G, lambda: max_of(STAN_MIN_G, "max_rhat"),
           "minutes head max R-hat"))
    add(_c("0.05025", STAN_MIN_D,
           lambda: cell(STAN_MIN_D, "rho", metric="season_level_rho"),
           "season-level rho"))
    add(_c("0.0776", STAN_MIN_D,
           lambda: cell(STAN_MIN_D, "rho", metric="game_level_rho"),
           "game-level rho"))
    add(_c("4.65", STAN_MIN_D,
           lambda: cell(STAN_MIN_D, "implied_overdispersion", metric="game_level_rho"),
           "game-level overdispersion"))
    add(_c("713,947", STAN_MIN_D,
           lambda: cell(STAN_MIN_D, "n_player_games", metric="game_level_rho"),
           "game-level population"))
    C += _minutes_pre_lock_claims(AVAIL)

    # ── the Stan availability port ────────────────────────────────────────────
    # Five arms since 2026-08-12, not four: the head fits a 2012-13 window with a
    # role-graded rho AND a two-component mixture, so all three point MLEs are refitted on
    # the windowed rows and scored beside it. Their CRPS figures also reproduce
    # `availability_window.csv`'s selected arms, which is a third-party check on the port.
    ports = [("beta_binomial", "9.8444", "0.3889", "0.0632", "0.2627"),
             ("beta_binomial_role_rho", "9.8247", "0.3889", "0.0588", "0.2627"),
             ("mixture_mle", "9.8237", "0.3844", "0.0631", "0.2245"),
             ("stan_plug_in", "9.8195", "0.3851", "0.0643", "0.2261"),
             ("stan_posterior", "9.8239", "0.3847", "0.0643", "0.2261")]
    for model, crps, r2, ks, rho in ports:
        for quoted, name in [(crps, "crps_games"), (r2, "r2_gp_share"),
                             (ks, "pit_ks_distance"), (rho, "dispersion_rho")]:
            add(_c(quoted, STAN_AV_M,
                   lambda m=model, n=name: metric(STAN_AV_M, m, n),
                   f"stan {model} {name}"))
    add(_c("14.3770", STAN_AV_M,
           lambda: metric(STAN_AV_M, "stan_posterior", "mae_games"), "stan posterior MAE"))
    add(_c("14.3610", STAN_AV_M,
           lambda: metric(STAN_AV_M, "stan_plug_in", "mae_games"), "stan plug-in MAE"))
    add(_c("14.3785", STAN_AV_M,
           lambda: metric(STAN_AV_M, "mixture_mle", "mae_games"), "mixture MLE MAE"))
    add(_c("14.4211", STAN_AV_M,
           lambda: metric(STAN_AV_M, "beta_binomial", "mae_games"), "MLE MAE"))
    add(_c("1.0073", STAN_AV_D, lambda: cell(STAN_AV_D, "max_rhat"), "stan R-hat"))
    add(_c("1,399", STAN_AV_D, lambda: cell(STAN_AV_D, "min_ess_bulk"), "stan min ESS"))
    add(_c("366", STAN_AV_D, lambda: cell(STAN_AV_D, "wall_clock_s"),
           "stan wall clock"))
    add(_c("332.588", STAN_AV_B,
           lambda: cell(STAN_AV_B, "shared_beta_sd", n_players=883),
           "shared-beta sd, full board"))
    # The single-component role-graded head this replaced on 2026-08-12. Presence only.
    for quoted, label in [("9.8136", "plug-in CRPS"), ("9.8155", "posterior CRPS"),
                          ("0.3894", "plug-in R2"), ("0.3893", "posterior R2"),
                          ("0.0679", "plug-in PIT KS"), ("0.0690", "posterior PIT KS"),
                          ("0.2595", "stan rho"), ("0.09352", "coefficient gap"),
                          ("1.658", "largest gap in sds"), ("1.0050", "R-hat"),
                          ("2,382", "min ESS"), ("94", "wall clock"),
                          ("297", "board shared-beta sd"), ("12.3%", "board inflation"),
                          ("0.3147", "MLE rho, <12 mpg"), ("0.3176", "rho, <12 mpg"),
                          ("0.2688", "MLE rho, 12-24"), ("0.2698", "rho, 12-24"),
                          ("0.2573", "MLE rho, 24-30"), ("0.2532", "rho, 24-30"),
                          ("0.2142", "MLE rho, 30+ mpg"), ("0.2064", "rho, 30+ mpg"),
                          ("1.54×", "rho spread"), ("1.47×", "MLE rho spread")]:
        add(_c(quoted, STAN_AV_M, lambda: float("nan"),
               f"pre-mixture single-component availability head: {label}",
               historical=True))
    # The full-window, shared-rho head, kept beside the windowed one it became. Presence
    # only: the point of keeping them is that the window is a trade and the board term is
    # the side of it that got worse.
    for quoted, label in [("10.0057", "MLE CRPS"), ("10.0063", "plug-in CRPS"),
                          ("10.0071", "posterior CRPS"), ("0.3736", "R2"),
                          ("0.0939", "PIT KS"), ("0.2806", "MLE rho"),
                          ("0.2808", "stan rho"), ("0.00335", "coefficient gap"),
                          ("0.085", "largest gap in sds"), ("1.0019", "R-hat"),
                          ("2,314", "min ESS"), ("196", "wall clock"),
                          ("222.8", "board shared-beta sd"), ("6.7%", "board inflation")]:
        add(_c(quoted, STAN_AV_M, lambda: float("nan"),
               f"pre-window full-sample availability head: {label}", historical=True))
    # The held-out port check, preserved beside the validation one it became. Presence-only:
    # these are the pre-lock figures, and the point of keeping them is that a reversal is the
    # most useful thing in this file.
    for quoted, label in [("10.7952", "MLE CRPS"), ("10.7947", "plug-in CRPS"),
                          ("10.7953", "posterior CRPS"), ("0.2831", "MLE R2"),
                          ("0.2832", "stan R2"), ("0.0963", "MLE PIT KS"),
                          ("0.0952", "plug-in PIT KS"), ("0.2757", "MLE rho"),
                          ("0.2759", "stan rho"), ("0.0127", "coefficient gap"),
                          ("0.095", "largest gap in sds"), ("1.0025", "R-hat"),
                          ("2,402", "min ESS"), ("254", "wall clock"),
                          ("911", "held-out board size"), ("219.1", "board shared-beta sd"),
                          ("604.0", "board independent sd"), ("6.4%", "board inflation")]:
        add(_c(quoted, STAN_AV_M, lambda: float("nan"),
               f"pre-lock held-out availability port: {label}", historical=True))

    # ── the season total ──────────────────────────────────────────────────────
    C += _season_total_claims(AVAIL)
    add(_c("0.469", PROFILE,
           lambda: prof("decomposition", "missed_scratch", "r_persistence_of_column"),
           "missed_scratch persistence"))

    C += _regime_claims(AVAIL)
    return C


# The six-row ladder plus the spell-process arm, on VALIDATION since 2026-08-05. The whole
# table was a test evaluation until then; the superseded values ride along as historical
# claims so a later editor cannot quietly delete the reversal.
SEASON_TOTAL_ROWS = [("full_season", "610.8", "751.8", "0.374", "523.3"),
                     ("prior_gp", "417.9", "560.5", "0.652", "19.4"),
                     ("league_age", "461.6", "565.9", "0.645", "12.9"),
                     ("beta_binomial", "400.5", "514.0", "0.707", "−3.1"),
                     ("spell_process", "406.6", "520.0", "0.700", "−15.9"),
                     ("oracle_rate", "261.9", "367.0", "0.851", "−41.3"),
                     ("oracle_gp", "214.4", "287.5", "0.908", "2.0")]

SEASON_TOTAL_HISTORICAL = ["646.3", "831.2", "0.141", "541.9", "475.0", "638.5",
                           "0.493", "41.9", "476.0", "595.8", "0.559", "23.2",
                           "435.1", "570.8", "0.595", "6.1", "302.7", "427.7",
                           "0.773", "5.3", "221.3", "304.2", "0.885", "−33.2",
                           "651.3", "499.4", "213.8", "132.5", "−32.7%", "−39.9",
                           "−23.3%", "316.9", "340.8", "211.1"]


# The four-way model ladder, on VALIDATION since 2026-08-08 — the last held-out head to be
# converted. The ORDERING reversed: the GBM leads on mean CRPS where the GLM led on test.
# The retired figures ride along as historical claims so a later editor cannot delete the
# one reversal in this project whose paired interval was actually computed.
LADDER_ROWS = [("gbm", "9.876", "14.20", "0.381", "0.070", "20.1"),
               ("ridge", "10.004", "14.29", "0.376", "0.107", "23.1"),
               ("beta_binomial", "10.006", "14.46", "0.374", "0.094", "23.7"),
               ("league_age", "13.387", "18.74", "−0.057", "0.151", "29.7")]

# The retired held-out ladder, listed per doc because each quotes a different slice of it:
# the plan doc carries the whole table, `docs/model-development-notes.md` the CRPS row plus
# the pre-block figures, and `README.md` and `docs/train-validate-test-split.md` only the
# CRPS row. A union list would presence-check figures into docs
# that never had them, which reports a stale claim where nothing is stale.
LADDER_HISTORICAL = ("10.795", "10.888", "10.896", "13.614", "15.39", "15.41", "18.94",
                     "0.283", "0.262", "0.275", "−0.084", "0.096", "0.079", "0.103",
                     "0.174", "23.3", "19.9", "22.7", "29.5", "10.914", "10.98", "11.04",
                     "0.126", "0.093", "10,361", "911")


LADDER_SCOPES = ("table", "summary", "headline")


def _availability_ladder_claims(doc: str, scope: str = "table",
                                historical: tuple[str, ...] = LADDER_HISTORICAL
                                ) -> list[Claim]:
    """The availability model ladder and the paired interval that settles it.

    Shared between `docs/model-development-notes.md`, `README.md`,
    `docs/availability-plan.md` and `docs/train-validate-test-split.md` for the reason
    `_season_total_claims` is: four docs quote one artifact, and a block going stale in one
    while staying current in another is the failure this file has already caught twice.

    The paired-bootstrap block is claimed at least as hard as the CRPS row, because the
    ordering reversed on the split move and the *interval* is the entire reason the head did
    not change with it. A doc that quoted the new ordering without the interval would be
    reporting a reversal as a verdict — so `headline` still carries the GBM's interval even
    though it carries nothing else.

    `scope` names how much of the block a doc actually quotes: `table` is the full plan-doc
    treatment, `summary` is the CRPS row with the intervals and quartiles, `headline` is the
    CRPS row alone. Claiming more than a doc quotes reports a stale claim where nothing is
    stale, which is noise in the one report that has to stay readable.
    """
    if scope not in LADDER_SCOPES:
        raise ValueError(f"scope must be one of {LADDER_SCOPES}; got {scope!r}")
    C: list[Claim] = []

    def add(*a, **k):
        C.append(_c(*a, **k, doc=doc))

    for model, crps, mae, r2, ks, od in LADDER_ROWS:
        add(crps, METRICS, lambda m=model: metric(METRICS, m, "crps_games"),
            f"{model} CRPS")
        if scope != "table":
            continue
        for quoted, name in [(mae, "mae_games"), (r2, "r2_gp_share"),
                             (ks, "pit_ks_distance"), (od, "implied_overdispersion")]:
            add(quoted, METRICS, lambda m=model, n=name: metric(METRICS, m, n),
                f"{model} {name}")

    if scope != "headline":
        for model, delta, lo, hi, p, share in [
                ("gbm", "−0.1297", "−0.3154", "+0.0672", "0.906", "62.4%"),
                ("ridge", "−0.0014", "−0.0546", "+0.0516", "0.515", "60.2%"),
                ("league_age", "+3.3811", "+2.8567", "+3.9299", "0.000", "28.5%")]:
            add(delta, LADDER, lambda m=model: ladder(m, "delta_vs_reference"),
                f"{model} vs the shipped head")
            add(lo, LADDER, lambda m=model: ladder(m, "ci_lo"), f"{model} CI low")
            add(hi, LADDER, lambda m=model: ladder(m, "ci_hi"), f"{model} CI high")
            add(p, LADDER, lambda m=model: ladder(m, "p_better"), f"{model} P(better)")
            if scope == "table":
                add(share, LADDER, lambda m=model: ladder(m, "share_rows_better"),
                    f"{model} share of rows better")

        # The quartile rows are the substantive reason the GLM keeps the head, so they are
        # claimed rather than left as prose beside a claimed mean.
        for model, quartiles in [("gbm", ["+0.333", "−0.063", "−0.452", "−0.366"]),
                                 ("ridge", ["+0.251", "+0.109", "−0.008", "−0.393"])]:
            for i, quoted in enumerate(quartiles, start=1):
                add(quoted, LADDER,
                    lambda m=model, g=f"gp_q{i}": ladder(m, "delta_vs_reference", g),
                    f"{model} delta in gp_q{i}")

    for quoted in historical:
        add(quoted, METRICS, lambda: float("nan"),
            f"pre-lock held-out availability ladder: {quoted}", historical=True)
    return C


# The notes quote the CRPS row, the pre-block figures and the workload note; `README.md`
# and `docs/train-validate-test-split.md` quote only the CRPS row.
LADDER_HISTORICAL_NOTES = ("10.795", "10.888", "10.896", "13.614", "10.914", "10.98",
                            "11.04", "−0.152", "10,361", "911", "0.268")
LADDER_HISTORICAL_README = ("10.795", "10.888", "10.896", "13.614")


# The playoff/mileage block, on VALIDATION since 2026-08-08. Unlike the ladder beside it
# this decision survived the move unchanged — same sign, same ordering of all four variants
# — which is the contrast the docs draw between a block worth a tenth of a game and a
# model gap of 0.13 that was never distinguishable from zero.
ABLATION_ROWS = [("baseline", "10.104", "0.363"),
                 ("plus_playoff_workload", "10.006", "0.374"),
                 ("plus_playoff_only", "10.015", "0.373"),
                 ("plus_career_minutes_only", "10.085", "0.365")]

ABLATION_HISTORICAL = ["10.914", "10.795", "10.817", "10.883",
                       "0.268", "0.283", "0.281", "0.271", "−0.119"]


def _workload_ablation_claims(doc: str, gain: bool = True) -> list[Claim]:
    """The playoff/mileage workload ablation, claimed from whichever doc quotes it."""
    C: list[Claim] = []

    def add(*a, **k):
        C.append(_c(*a, **k, doc=doc))

    for variant, crps, r2 in ABLATION_ROWS:
        add(crps, ABLATION, lambda v=variant: cell(ABLATION, "crps_games", variant=v),
            f"ablation {variant} CRPS")
        add(r2, ABLATION, lambda v=variant: cell(ABLATION, "r2_gp_share", variant=v),
            f"ablation {variant} R2")
    if gain:
        add("−0.098", ABLATION,
            lambda: (cell(ABLATION, "crps_games", variant="plus_playoff_workload")
                     - cell(ABLATION, "crps_games", variant="baseline")),
            "playoff-workload CRPS gain")
    for quoted in ABLATION_HISTORICAL:
        add(quoted, ABLATION, lambda: float("nan"),
            f"pre-lock held-out workload ablation: {quoted}", historical=True)
    return C


def _season_total_claims(doc: str, rotation: bool = True) -> list[Claim]:
    """The season-total ladder, claimed from whichever doc quotes it.

    Shared for the reason `_regime_claims` is: `docs/model-development-notes.md` and
    `docs/availability-plan.md` carry the same table against the same artifact, and the
    one failure this repo has already shipped twice is a block going stale in one doc
    while staying current in the other.

    `rotation` is off for docs that quote only the headline rows.
    """
    C: list[Claim] = []
    add = C.append
    for name, mae, rmse, r2, bias in SEASON_TOTAL_ROWS:
        for quoted, which in [(mae, "mae_dk_total"), (rmse, "rmse_dk_total"),
                              (r2, "r2_dk_total"), (bias, "bias_dk_total")]:
            add(_c(quoted, SEASON_TOTAL,
                   lambda n=name, w=which: treatment(n, w),
                   f"season total {name} {which}", doc=doc))
    if rotation:
        add(_c("582.9", SEASON_TOTAL,
               lambda: treatment("full_season", "mae_dk_total", "rotation"),
               "rotation naive MAE", doc=doc))
        add(_c("451.3", SEASON_TOTAL,
               lambda: treatment("beta_binomial", "mae_dk_total", "rotation"),
               "rotation head MAE", doc=doc))
    return C


# The minutes head's pre-lock held-out column, kept beside the validation one it became when
# `make stan-minutes` was re-run under `src/models/held_out.py` on 2026-08-06. Two kinds of
# figure retire together here and it is worth knowing why: the `test_*` columns because the
# sweep no longer writes them, and the *old* `val_crps` values because selection was raised
# from 500/500 to full-length chains at the same time, so even the surviving column is a
# different measurement. Only the floor's 161.45 is arithmetic and reproduced exactly.
_MINUTES_PRE_LOCK = (
    ("144.62", "linear val CRPS, short chains"),
    ("145.44", "logit_own val CRPS, short chains"),
    ("144.83", "quadratic val CRPS, short chains"),
    ("144.13", "spline val CRPS, short chains"),
    ("168.24", "floor test CRPS"),
    ("147.18", "linear test CRPS"),
    ("147.35", "logit_own test CRPS"),
    ("147.21", "quadratic test CRPS"),
    ("146.85", "spline test CRPS"),
    ("0.8166", "floor test R2"),
    ("0.8565", "linear and logit_own test R2"),
    ("0.8574", "quadratic test R2"),
    ("0.8572", "spline test R2"),
    ("+0.0407", "R2 gain over floor, test"),
    ("−21.4", "CRPS gain over floor, test"),
    ("2,183", "wall clock over 8 fits"),
    ("1.0093", "max R-hat over 8 fits"),
    ("899", "spline wall clock, test refit"),
    ("189", "linear wall clock, test refit"),
    ("−33.1", "linear test bias"),
    ("−40.9", "spline test bias"),
    ("−5.69", "floor test bias"),
)


def _minutes_pre_lock_claims(doc: str, keep: tuple[str, ...] | None = None
                             ) -> list[Claim]:
    """The retired minutes column, presence-checked from whichever doc still quotes it.

    Shared for the reason `_season_total_claims` is: `docs/model-development-notes.md` and
    `docs/availability-plan.md` carry the same block, `docs/predictions-plan.md` quotes a
    subset, and one of them going stale while the others stay current is the failure this
    module has already caught twice. `keep` narrows the set for a doc that only quotes part
    of it, so a claim cannot rot into describing nothing.
    """
    return [_c(quoted, STAN_MIN_M, lambda: float("nan"),
               f"pre-lock held-out minutes head: {label}", doc=doc, historical=True)
            for quoted, label in _MINUTES_PRE_LOCK
            if keep is None or quoted in keep]


def _regime_claims(doc: str) -> list[Claim]:
    """The two regime confounds, tested rather than flagged.

    Shared by `docs/availability-plan.md` and the notes against the one artifact. The
    Participation Policy arm is the one that matters: it is the era finding arriving as a
    dated policy step, from a different estimator than the era table.
    """
    C: list[Claim] = []
    add = C.append
    ppp = [("gp_share [30+ mpg]", "−4.63%", "0.008", "−2.90%"),
           ("gp_share [all]", "−4.43%", "0.024", "−2.78%"),
           ("gp_share [24-30]", "−4.92%", "0.066", "−3.08%"),
           ("gp_share [12-24]", "−3.46%", "0.180", "−2.16%"),
           ("gp_share [<12 mpg]", "+5.37%", "0.280", "+3.30%")]
    for name, effect, p, shift in ppp:
        add(_c(effect, SEASON_REGIME,
               lambda s=name: regime(s, "regime_effect_pct") / 100.0,
               f"PPP level break {name}", doc=doc))
        add(_c(p, SEASON_REGIME, lambda s=name: regime(s, "p_value"),
               f"PPP break p-value {name}", doc=doc))
        add(_c(shift, SEASON_REGIME,
               lambda s=name: regime(s, "next_season_shift_pct") / 100.0,
               f"PPP one-season-ahead shift {name}", doc=doc))
    add(_c("+0.22%", SEASON_REGIME,
           lambda: regime("gp_share [30+ mpg]", "regime_effect_pct",
                          "covid_indicator") / 100.0, "COVID effect, 30+ mpg", doc=doc))
    add(_c("0.91", SEASON_REGIME,
           lambda: regime("gp_share [30+ mpg]", "p_value", "covid_indicator"),
           "COVID p-value, 30+ mpg", doc=doc))
    # The fragility pair: a slope fitted on three seasons, extrapolated one further.
    add(_c("+22.1%", SEASON_REGIME,
           lambda: regime("gp_share [<12 mpg]", "next_season_shift_pct",
                          "policy_break") / 100.0, "level+slope shift, fringe", doc=doc))
    add(_c("+18.8%", SEASON_REGIME,
           lambda: regime("stl", "next_season_shift_pct", "policy_break") / 100.0,
           "level+slope shift, stl", doc=doc))
    add(_c("3", SEASON_REGIME,
           lambda: regime("gp_share [all]", "regime_seasons"), "post-break seasons",
           doc=doc))
    return C


_CARRY_BIAS_RETIRED = (
    ("−3.1%", "fta 2024-25"), ("−10.7%", "fta 2025-26"), ("−7.0%", "fta pooled"),
    ("−10.0%", "stl 2024-25"), ("+0.7%", "stl 2025-26"), ("−4.6%", "stl pooled"),
    ("−1.3%", "fg3a 2025-26"), ("−4.2%", "fg3a pooled"),
    ("−1.8%", "tov 2025-26"), ("−3.2%", "tov pooled"),
    ("−1.1%", "ast 2024-25"), ("−4.8%", "ast 2025-26"), ("−2.9%", "ast pooled"),
    ("+6.5%", "blk 2024-25"), ("+6.0%", "blk 2025-26"), ("+6.2%", "blk pooled"),
    # The retired lag agreement is "10 of 14" in the prose, which is not a single number
    # and so cannot be a `quoted`. The correlation beside it is unique to that sentence, so
    # protecting it protects the count: delete the sentence and this claim goes stale.
    ("−0.865", "retired lag correlation"),
    ("791", "retired held-out rows"),
)


def _carry_bias_claims(doc: str,
                       components: tuple[str, ...] = ("fta", "ast", "blk", "stl",
                                                      "tov", "reb", "fga"),
                       retired: tuple[str, ...] | None = None) -> list[Claim]:
    """The no-fit floor's season bias, and the lag test that explains it.

    Shared by `docs/predictions-plan.md` and `docs/facts-archive.md` against the one
    artifact, for the reason this file keeps re-learning: the same block current in one doc
    and stale in the other is the failure that has already happened twice here. The archive
    carries only the
    two components it names, so `components` narrows the cell claims and `retired` narrows
    the superseded ones, while the lag test — which both docs quote in full — is claimed
    either way.

    **Each cell is claimed beside its league move on purpose.** The finding is not that any
    component is biased by some amount; it is that the bias *opposes* that season's league
    move, which is what makes it a one-season lag rather than a level. Claiming the bias
    alone would let the pair drift apart silently, and the retired reading is the proof that
    a bias quoted without its league move gets read as a standing property of the component.
    """
    C: list[Claim] = []
    add = C.append
    cells = {
        #          league 2022-23, bias 2022-23, league 2023-24, bias 2023-24, pooled
        "fta": ("+7.3%", "−4.4%", "−7.5%", "+10.7%", "+2.9%"),
        "ast": ("+2.5%", "−2.2%", "+5.6%", "−6.2%", "−4.2%"),
        "blk": ("−1.5%", "+4.3%", "+10.6%", "−5.4%", "−0.8%"),
        "stl": ("−4.6%", "+5.1%", "+2.7%", "−3.5%", "+0.8%"),
        "tov": ("+2.7%", "−0.6%", "−3.8%", "+5.6%", "+2.5%"),
        "reb": ("−2.5%", "+3.8%", "+0.4%", "−0.0%", "+1.9%"),
        "fga": ("−0.0%", "+1.4%", "+0.9%", "+0.9%", "+1.1%"),
    }
    for comp in components:
        m22, b22, m23, b23, pooled = cells[comp]
        for season, move, bias in [("2022-23", m22, b22), ("2023-24", m23, b23)]:
            add(_c(move, SEASON_BIAS,
                   lambda c=comp, s=season: carry_league_move(c, s) / 100.0,
                   f"league move {comp} {season}", doc=doc))
            add(_c(bias, SEASON_BIAS,
                   lambda c=comp, s=season: carry_bias(c, s) / 100.0,
                   f"carry-forward bias {comp} {season}", doc=doc))
        add(_c(pooled, SEASON_BIAS, lambda c=comp: carry_bias(c, "all") / 100.0,
               f"carry-forward bias {comp} pooled", doc=doc))

    # The lag test itself. "13 of 14" is two claims because the denominator is the thing
    # that would move if a head were added or retired, and a ratio quoted as a fraction of
    # a changed denominator is the same silent staleness one level down.
    add(_c("13", SEASON_BIAS, _lag_opposing, "cells opposing the league move", doc=doc))
    add(_c("14", SEASON_BIAS, lambda: _lag_opposing(total=True),
           "per-season bias cells", doc=doc))
    add(_c("−0.944", SEASON_BIAS, _lag_corr, "bias vs league move correlation", doc=doc))
    # `blk`'s trend profile is what disqualifies the retired "drift, not noise" reading, so
    # it is claimed here beside the reversal rather than in the drift table it comes from.
    add(_c("0.118", SEASON_EFF, lambda: season_eff("blk", "trend_r2"),
           "blk trend R2, the no-drift check", doc=doc))
    add(_c("−0.14%", SEASON_EFF,
           lambda: season_eff("blk", "trend_pct_per_season") / 100.0,
           "blk trend slope", doc=doc))

    # ── the retired held-out reading ──────────────────────────────────────────
    # Presence-checked only: the artifact was legitimately rebuilt on validation, so these
    # rows no longer exist anywhere. That is the same demotion `subst_test` took — a
    # superseded figure stays value-checked only for as long as some artifact still holds
    # it, and freezing a run to keep one checkable is the wrong trade.
    for quoted, label in _CARRY_BIAS_RETIRED:
        if retired is None or quoted in retired:
            add(_c(quoted, SEASON_BIAS, lambda: float("nan"),
                   f"retired held-out carry-forward bias: {label}", doc=doc,
                   historical=True))
    return C


def _minutes_unification_claims(doc: str) -> list[Claim]:
    """The season-unit head-to-head between the two minutes heads, claimable from any doc.

    A shared builder for the same reason `_regime_claims` and `_season_term_claims` are: the
    block is quoted in `README.md` and in `docs/simulations-plan.md`, and "current in one and
    stale in the other" is the failure that has already happened twice here. The plan doc is
    not in the audit yet — build item 11 adds it — so today this has one caller and the
    second is why it is a function.

    The verdict-bearing pair is claimed from **both** sides of each comparison rather than
    only from the gap, because a headline that quotes 170.06 against 144.35 and a gap of
    +25.70 can go stale in three independent places.
    """
    C: list[Claim] = []

    def add(quoted, column, arm, label, **kw):
        C.append(_c(quoted, MIN_UNIF, lambda: cell(MIN_UNIF, column, arm=arm), label,
                    doc=doc, **kw))

    COMP, MINS = "composition_sum", "minutes_head"
    FLOOR, ALL = "carry_forward", "composition_sum_all_rows"
    DELTA = "composition_minus_minutes"

    # The verdict: CRPS at the season unit, both arms and the floor they are read against.
    add("170.06", "crps_minutes", COMP, "composition summed to season totals, CRPS")
    add("144.35", "crps_minutes", MINS, "marginal minutes head, season-total CRPS")
    add("161.29", "crps_minutes", FLOOR,
        "the season-unit no-fit carry-forward floor the composition fails to clear")

    # The mean is a tie — three figures on each side, because "a tie" is the claim.
    add("200.28", "mae_minutes", COMP, "composition season-total MAE")
    add("200.12", "mae_minutes", MINS, "marginal head season-total MAE")
    add("0.8848", "r2_minutes", COMP, "composition season-total R2")
    add("0.8829", "r2_minutes", MINS, "marginal head season-total R2")
    add("+2.41", "bias_minutes", COMP, "composition season-total bias")
    add("−14.09", "bias_minutes", MINS, "marginal head season-total bias")

    # The spread is not, and this is what keeps both heads in the chain.
    add("64.65", "predictive_sd", COMP, "composition season-total predictive sd")
    add("302.75", "predictive_sd", MINS, "marginal head season-total predictive sd")
    add("0.3341", "pit_ks", COMP, "composition season-total PIT KS")
    add("0.0735", "pit_ks", MINS, "marginal head season-total PIT KS")
    add("0.00", "team_season_sd", ALL,
        "a team's season minutes are fixed across draws — the structural half")

    # The paired bootstrap, which is what makes the gap a verdict rather than a margin.
    add("+25.70", "crps_delta", DELTA, "paired-bootstrap CRPS gap, composition − minutes")
    add("+18.96", "ci_lo", DELTA, "bootstrap interval, lower")
    add("+33.25", "ci_hi", DELTA, "bootstrap interval, upper")
    add("4.68", "sd_ratio_minutes_over_composition", DELTA,
        "how much narrower the composition's season total is")

    # Coverage: the composition's genuine advantage, reported as its own row.
    add("742", "n", MINS, "player-seasons both heads cover")
    add("1,111", "n", ALL, "player-seasons the composition covers")
    C.append(_c("369", MIN_UNIF,
                lambda: cell(MIN_UNIF, "n", arm=ALL) - cell(MIN_UNIF, "n", arm=MINS),
                "rows only the composition reaches — the `>= 200 prior minutes` drop",
                doc=doc))

    # The injected per-player-season effect: whether the narrow season total is a ceiling
    # or a missing parameter. Claimed at the sweep's best sigma, plus the sigma itself,
    # because "it ties at 0.375" goes stale if either half moves.
    PS = "composition_sum_plus_player_season_effect"
    # **The unit is part of the lookup, not decoration.** Since 2026-08-09 the same arm name
    # also carries the fallback's `ps_sigma_on_train` rows — the identical grid scored on
    # TRAINING player-seasons, so every sigma value appears twice. Without the unit filter
    # `ps()` would be ambiguous and `best_sigma` would silently return the train grid's
    # optimum, whose CRPS is lower for the ordinary reason that it is in-sample. This is the
    # same silent-ambiguity failure the composition PPC lookups were fixed for.
    UNIT = "ps_effect_sweep"

    def ps(column: str, sigma: float = 0.375) -> float:
        return cell(MIN_UNIF, column, arm=PS, sigma=sigma, unit=UNIT)

    def best_sigma() -> float:
        """The sweep's own CRPS-minimising sigma, so `0.375` cannot rot into a stale label."""
        frame = table(MIN_UNIF)
        arm = frame[(frame["arm"] == PS) & (frame["unit"] == UNIT)]
        return float(arm.loc[arm["crps_minutes"].idxmin(), "sigma"])

    C += [
        _c("0.375", MIN_UNIF, best_sigma,
           "the sweep's CRPS-optimal injected effect size", doc=doc),
        _c("142.17", MIN_UNIF, lambda: ps("crps_minutes"),
           "composition + injected player-season effect, CRPS at the sweep optimum",
           doc=doc),
        _c("239.45", MIN_UNIF, lambda: ps("predictive_sd"),
           "the season-total spread the injection recovers", doc=doc),
        _c("−2.18", MIN_UNIF, lambda: ps("crps_delta"),
           "injected arm against the marginal head", doc=doc),
        _c("−6.96", MIN_UNIF, lambda: ps("ci_lo"), "injected arm interval, lower", doc=doc),
        _c("+2.85", MIN_UNIF, lambda: ps("ci_hi"), "injected arm interval, upper", doc=doc),
        _c("0.0659", MIN_UNIF, lambda: ps("pit_ks", 0.45),
           "best PIT KS in the sweep — better calibrated than the marginal head", doc=doc),
    ]

    # The fallback made shippable: the SAME grid on training rows, which is what removes the
    # injection's one load-bearing caveat. Claimed from both grids, because "they agree to a
    # step" is the claim and it goes stale if either side moves.
    TRAIN_UNIT = "ps_sigma_on_train"

    def on_train(column: str, sigma: float = 0.45) -> float:
        return cell(MIN_UNIF, column, arm=PS, sigma=sigma, unit=TRAIN_UNIT)

    def best_train_sigma() -> float:
        frame = table(MIN_UNIF)
        arm = frame[(frame["arm"] == PS) & (frame["unit"] == TRAIN_UNIT)]
        return float(arm.loc[arm["crps_minutes"].idxmin(), "sigma"])

    C += [
        _c("0.450", MIN_UNIF, best_train_sigma,
           "the injection's sigma re-estimated on TRAIN — the fallback's shippable figure",
           doc=doc),
        _c("117.07", MIN_UNIF, lambda: on_train("crps_minutes"),
           "train-grid CRPS at its own optimum", doc=doc),
        _c("1,145", MIN_UNIF, lambda: on_train("n"),
           "training player-seasons the fallback's sigma is estimated over", doc=doc),
        _c("142.87", MIN_UNIF, lambda: ps("crps_minutes", 0.45),
           "validation CRPS at the train-estimated sigma", doc=doc),
        _c("−1.49", MIN_UNIF, lambda: ps("crps_delta", 0.45),
           "the train-estimated sigma against the marginal head", doc=doc),
        _c("−6.14", MIN_UNIF, lambda: ps("ci_lo", 0.45),
           "that arm's interval, lower", doc=doc),
        _c("+3.22", MIN_UNIF, lambda: ps("ci_hi", 0.45),
           "that arm's interval, upper", doc=doc),
    ]

    # The zero-sum dynamic: the composition sits on the identity a fixed team total forces
    # and the marginal head does not, which is invisible in every marginal metric.
    def couple(column: str, arm: str) -> float:
        return cell(MIN_UNIF, column, arm=arm, unit="teammate_coupling")

    C += [
        _c("−0.0509", MIN_UNIF, lambda: couple("r_teammates", COMP),
           "composition teammate correlation", doc=doc),
        _c("−0.0001", MIN_UNIF, lambda: couple("r_teammates", MINS),
           "marginal head teammate correlation — the failure", doc=doc),
        _c("−0.0664", MIN_UNIF, lambda: couple("r_implied_by_fixed_sum", COMP),
           "what a fixed team total forces at the measured roster size", doc=doc),
        _c("1,022.9", MIN_UNIF, lambda: couple("team_season_sum_sd", MINS),
           "the marginal head's spread on a physically fixed team total", doc=doc),
        _c("16.05", MIN_UNIF, lambda: couple("roster_size", COMP),
           "mean single-team roster size the coupling is measured over", doc=doc),
        _c("963", MIN_UNIF, lambda: couple("n", COMP),
           "single-team player-seasons the coupling is measured on", doc=doc),
    ]
    return C


def _composition() -> list[Claim]:
    """`docs/minutes-composition-plan.md` — the team-game minutes allocation.

    Refreshed for the full-window refit (Gate E, 2026-08-04). Two things are audited that
    are not just "the table": the `independent_comparator` row, which is INVARIANT to the
    window by construction and is therefore the control on the whole refit, and the pilot
    figures the doc keeps in its before/after table, which are marked historical.

    The `binomial` row is the one worth auditing hardest: it is the arm that FAILS, and
    "the pure decomposition is too tight" is the head's sharpest claim.
    """
    C: list[Claim] = []
    add = C.append
    SEL = "betabinom_ot_graded"

    # ── the validation ladder, re-measured 2026-08-08 ─────────────────────────
    # The sweep is validation-only since the held-out lock (src/models/held_out.py) and
    # the artifact was regenerated on 2026-08-08, so `test_*` no longer exists here. The
    # retired test column is preserved wholesale at the foot of this builder.
    comp = [("carry_forward", "4.6776", "0.4442", "0.0496"),
            ("binomial", "4.9388", "0.4752", "0.1919"),
            ("betabinom", "4.5417", "0.4699", "0.0496"),
            ("betabinom_ot", "4.5431", "0.4697", "0.0494"),
            (SEL, "4.4945", "0.4741", "0.0428"),
            ("independent_comparator", "4.7842", "0.4024", "0.0483")]
    for name, val_crps, val_r2, pit in comp:
        for quoted, column in [(val_crps, "val_crps"), (val_r2, "val_r2"),
                               (pit, "val_pit_ks")]:
            add(_c(quoted, COMP_M,
                   lambda n=name, c=column: cell(COMP_M, c, variant=n),
                   f"composition {name} {column}", doc=COMP))

    # Gate C / Gate D, as differences rather than as retyped numbers.
    add(_c("−0.1832", COMP_M,
           lambda: (cell(COMP_M, "val_crps", variant=SEL)
                    - cell(COMP_M, "val_crps", variant="carry_forward")),
           "composition vs the floor, val", doc=COMP))
    add(_c("−0.2898", COMP_M,
           lambda: (cell(COMP_M, "val_crps", variant=SEL)
                    - cell(COMP_M, "val_crps", variant="independent_comparator")),
           "composition vs the incumbent, val", doc=COMP))
    add(_c("−6.06%", COMP_M,
           lambda: (cell(COMP_M, "val_crps", variant=SEL)
                    / cell(COMP_M, "val_crps", variant="independent_comparator") - 1.0),
           "composition gain as a percentage, val", doc=COMP))
    add(_c("−3.92%", COMP_M,
           lambda: (cell(COMP_M, "val_crps", variant=SEL)
                    / cell(COMP_M, "val_crps", variant="carry_forward") - 1.0),
           "composition gain over the floor, percentage, val", doc=COMP))
    add(_c("−0.5521", COMP_M,
           lambda: cell(COMP_M, "val_bias", variant="independent_comparator"),
           "comparator bias, val", doc=COMP))

    # Gate A — the gate that did not behave, so both sides of the miss are audited.
    # The one-pass sweep re-measured the miss at 1.17x, well inside the 1.63x the
    # two-pass run recorded; see the retired block below and `stan_games_played`.
    add(_c("8.3", COMP_M, lambda: cell(COMP_M, "probe_hours", variant="binomial"),
           "composition Gate A extrapolation", doc=COMP))
    add(_c("9.78", COMP_D, lambda: _comp_sweep_seconds() / 3600,
           "composition actual sweep hours", doc=COMP))
    add(_c("1.17", COMP_D,
           lambda: (_comp_sweep_seconds() / 3600
                    / cell(COMP_M, "probe_hours", variant="binomial")),
           "how far Gate A under-predicted", doc=COMP))
    add(_c("155", COMP_D,
           lambda: cell(COMP_D, "wall_clock_s", label="probe/one-season"),
           "probe seconds", doc=COMP))
    add(_c("14.83", COMP_D,
           lambda: cell(COMP_D, "wall_clock_s", label="betabinom/val") / 631158 * 1000,
           "full-window ms per row", doc=COMP))
    add(_c("1.00935", COMP_D, lambda: max_of(COMP_D, "max_rhat"),
           "composition max R-hat", doc=COMP))
    add(_c("0", COMP_D, lambda: total(COMP_D, "divergences"),
           "composition divergences", doc=COMP))
    add(_c("6", COMP_D, lambda: rows(COMP_D), "composition fits", doc=COMP))

    # The per-arm cost shares behind the fallback-order correction.
    for arm, minutes, share in [("binomial", "120.5", "20.5%"),
                                ("betabinom", "156.0", "26.6%"),
                                ("betabinom_ot", "150.2", "25.6%"),
                                (SEL, "160.2", "27.3%")]:
        add(_c(minutes, COMP_D, lambda a=arm: _comp_arm_seconds(a) / 60,
               f"{arm} sampler minutes", doc=COMP))
        add(_c(share, COMP_D,
               lambda a=arm: _comp_arm_seconds(a) / _comp_sweep_seconds(),
               f"{arm} share of sweep time", doc=COMP))
    add(_c("586.9", COMP_D, lambda: _comp_sweep_seconds() / 60,
           "total sweep sampler minutes", doc=COMP))

    # The team-sum asymmetry — the capability the model exists for, so both sides are
    # audited rather than only the headline. Every lookup names its variant: the PPC file
    # carries two arms, and an unfiltered lookup silently takes whichever sorts first.
    add(_c("33.89", COMP_P,
           lambda: cell(COMP_P, "simulated", variant=SEL,
                        analysis="team_sum_abs_error", group="independent"),
           "comparator team-sum error", doc=COMP))
    for quoted, column, group in [("0.6013", "observed", "regulation/composition"),
                                  ("0.6423", "observed", "overtime/composition"),
                                  ("0.5912", "simulated", "regulation/composition"),
                                  ("0.6415", "simulated", "overtime/composition"),
                                  ("0.5923", "simulated", "regulation/independent"),
                                  ("0.6280", "simulated", "overtime/independent")]:
        add(_c(quoted, COMP_P,
               lambda c=column, g=group: cell(COMP_P, c, variant=SEL,
                                              analysis="starter_share", group=g),
               f"starter share {column} {group}", doc=COMP))

    # The graded-vs-shared calibration table — the point of the graded arm, so both
    # columns are audited rather than only the improved one.
    ratios = [("betabinom_ot", "q1_fringe", "1.2224"),
              ("betabinom_ot", "q2", "0.8430"),
              ("betabinom_ot", "q3", "0.6589"),
              ("betabinom_ot", "q4_star", "0.6285"),
              (SEL, "q1_fringe", "1.0465"),
              (SEL, "q2", "0.8132"),
              (SEL, "q3", "0.7071"),
              (SEL, "q4_star", "0.8136")]
    for arm, tier, quoted in ratios:
        add(_c(quoted, COMP_P,
               lambda a=arm, t=tier: cell(COMP_P, "ratio", variant=a,
                                          analysis="variance_ratio", group=t),
               f"variance ratio {arm} {tier}", doc=COMP))
    for arm, quoted in [("betabinom_ot", "0.2730"), (SEL, "0.1782")]:
        add(_c(quoted, COMP_P,
               lambda a=arm: mean_abs_dev(COMP_P, "ratio", 1.0, variant=a,
                                          analysis="variance_ratio"),
               f"mean |ratio-1| {arm}", doc=COMP))
    add(_c("35%", COMP_P,
           lambda: (1.0 - mean_abs_dev(COMP_P, "ratio", 1.0, variant=SEL,
                                       analysis="variance_ratio")
                    / mean_abs_dev(COMP_P, "ratio", 1.0, variant="betabinom_ot",
                                   analysis="variance_ratio")),
           "calibration error cut by grading", doc=COMP))

    # The fitted dispersions themselves — the mechanism, and the sharpest single
    # statement that role grading is real.
    for b, quoted in [("1", "0.1768"), ("2", "0.1300"), ("3", "0.1115"), ("4", "0.0855")]:
        add(_c(quoted, COMP_RHO,
               lambda i=int(b): cell(COMP_RHO, "rho", variant=SEL, bin=i),
               f"graded rho bin {b}", doc=COMP))
    add(_c("0.1211", COMP_RHO,
           lambda: cell(COMP_RHO, "rho", variant="betabinom_ot", bin=1),
           "shared rho", doc=COMP))
    add(_c("2.07", COMP_RHO,
           lambda: (cell(COMP_RHO, "rho", variant=SEL, bin=1)
                    / cell(COMP_RHO, "rho", variant=SEL, bin=4)),
           "graded rho spread", doc=COMP))

    for arm, quoted in [("composition", "32.862"), ("independent", "37.984")]:
        add(_c(quoted, COMP_J,
               lambda a=arm: cell(COMP_J, "mean_joint_nll", split="val", arm=a),
               f"composition joint NLL {arm}", doc=COMP))

    # The OT tail is invariant to the window by construction — the floor gets 1996-97 to
    # 2021-22 either way — so these confirm rather than change. They are re-derived from
    # `stan-game-length` since 2026-08-09; the values did not move, only the producer.
    add(_c("0.0608", GL_M, lambda: cell(GL_M, "p_any_ot", variant="floor"),
           "OT tail p_any", doc=COMP))
    add(_c("0.1408", GL_M, lambda: cell(GL_M, "p_more_ot", variant="floor"),
           "OT tail p_more", doc=COMP))
    add(_c("30,626", GL_M, lambda: cell(GL_M, "n_fit_games", variant="floor"),
           "OT tail training games", doc=COMP))
    add(_c("128.4", GL_PPC,
           lambda: cell(GL_PPC, "predicted", variant="floor", **{"class": "1OT"}),
           "OT tail predicted 1OT", doc=COMP))
    add(_c("120", GL_PPC,
           lambda: cell(GL_PPC, "observed", variant="floor", **{"class": "1OT"}),
           "OT tail observed 1OT", doc=COMP))

    # ── the retired TEST column, held for the record ──────────────────────────
    # Gate E was taken on these on 2026-08-04, before `src/models/held_out.py` locked the
    # split. The sweep no longer fits or scores test, so no artifact can back them and
    # they are presence-checked only — which makes DELETION the failure this guards, not
    # drift. They are kept because the head's whole verdict was once quoted from them.
    for quoted, label in [("4.8576", "retired test CRPS, carry_forward"),
                          ("4.9732", "retired test CRPS, binomial"),
                          ("4.5893", "retired test CRPS, betabinom"),
                          ("4.5848", "retired test CRPS, betabinom_ot"),
                          ("4.5592", "retired test CRPS, selected"),
                          ("4.9140", "retired test CRPS, comparator"),
                          ("0.3678", "retired test R2, carry_forward"),
                          ("0.4259", "retired test R2, binomial"),
                          ("0.4247", "retired test R2, betabinom"),
                          ("0.4255", "retired test R2, betabinom_ot"),
                          ("0.4244", "retired test R2, selected"),
                          ("0.3301", "retired test R2, comparator"),
                          ("0.0354", "retired test PIT KS, carry_forward"),
                          ("0.1948", "retired test PIT KS, binomial"),
                          ("0.0405", "retired test PIT KS, betabinom"),
                          ("0.0414", "retired test PIT KS, betabinom_ot"),
                          ("0.0393", "retired test PIT KS, selected"),
                          ("0.0769", "retired test PIT KS, comparator"),
                          ("−0.2985", "retired test gain vs the floor"),
                          ("−0.3548", "retired test gain vs the incumbent"),
                          ("−7.2%", "retired test gain, percentage"),
                          ("−1.2856", "retired test comparator bias"),
                          ("36.87", "retired test comparator team-sum error"),
                          ("0.5882", "retired test starter share, observed regulation"),
                          ("0.6314", "retired test starter share, observed overtime"),
                          ("0.5998", "retired test starter share, simulated regulation"),
                          ("0.6464", "retired test starter share, simulated overtime"),
                          ("1.3466", "retired test variance ratio, shared q1"),
                          ("0.8089", "retired test variance ratio, shared q2"),
                          ("0.7691", "retired test variance ratio, shared q3"),
                          ("0.5973", "retired test variance ratio, shared q4"),
                          ("1.0845", "retired test variance ratio, graded q1"),
                          ("0.7687", "retired test variance ratio, graded q2"),
                          ("0.8217", "retired test variance ratio, graded q3"),
                          ("0.7757", "retired test variance ratio, graded q4"),
                          ("0.2928", "retired test mean |ratio-1|, shared"),
                          ("0.1796", "retired test mean |ratio-1|, graded"),
                          ("39%", "retired test calibration cut"),
                          ("0.1751", "retired two-pass graded rho, fringe"),
                          ("0.1285", "retired two-pass graded rho, q2"),
                          ("0.1099", "retired two-pass graded rho, q3"),
                          ("0.0839", "retired two-pass graded rho, star"),
                          ("0.1195", "retired two-pass shared rho"),
                          ("2.09", "retired two-pass rho spread"),
                          ("33.614", "retired test joint NLL, composition"),
                          ("38.828", "retired test joint NLL, independent"),
                          ("256.9", "retired test OT tail predicted 1OT"),
                          ("222", "retired test OT tail observed 1OT"),
                          ("12.8", "retired two-pass Gate A extrapolation"),
                          ("20.9", "retired two-pass sweep hours"),
                          ("1.63", "retired two-pass Gate A under-prediction"),
                          ("15.23", "retired two-pass ms per row"),
                          ("1.0113", "retired two-pass max R-hat"),
                          ("191.8", "retired two-pass binomial sampler minutes"),
                          ("506.3", "retired two-pass betabinom sampler minutes"),
                          ("1,251.8", "retired two-pass total sampler minutes")]:
        add(_c(quoted, COMP_M, lambda: float("nan"), label, doc=COMP, historical=True))

    # The pilot figures the doc keeps in its before/after table. Superseded by the
    # full-window refit and preserved beside it, so presence-checked and value-exempt.
    for quoted, label in [("−0.406", "pilot gain vs the incumbent"),
                          ("−0.312", "pilot gain vs the floor"),
                          ("0.1480", "pilot graded rho, fringe"),
                          ("0.1125", "pilot graded rho, q2"),
                          ("0.0874", "pilot graded rho, q3"),
                          ("0.0613", "pilot graded rho, star"),
                          ("0.0970", "pilot shared rho"),
                          ("2.41", "pilot rho spread"),
                          ("0.1055", "pilot mean |ratio-1|"),
                          ("0.988", "pilot star-tier variance ratio"),
                          ("59%", "pilot calibration cut")]:
        add(_c(quoted, COMP_M, lambda: float("nan"), label, doc=COMP, historical=True))
    return C


def _comp_arm_seconds(arm: str) -> float:
    """Sampler seconds for one sweep arm, both splits."""
    frame = table(COMP_D)
    if frame is None:
        return float("nan")
    hit = frame[frame["label"].isin([f"{arm}/val", f"{arm}/test"])]
    _note_columns("wall_clock_s")
    return float(hit["wall_clock_s"].sum())


def _comp_sweep_seconds() -> float:
    """Sampler seconds for the four-arm sweep, excluding the probe and the comparator."""
    return sum(_comp_arm_seconds(a) for a in
               ("binomial", "betabinom", "betabinom_ot", "betabinom_ot_graded"))


def _predictions() -> list[Claim]:
    """`docs/predictions-plan.md` — the layer above the per-player marginals.

    This doc quotes **two** runs of the same eleven heads: a Poisson/sklearn one
    (`component_rate_metrics.csv`) and a negative-binomial/Stan one
    (`stan_component_metrics.csv`), and it is explicit that the second *overturns* the
    first's guidance. They are claimed against separate artifacts and must never be
    cross-checked against each other.
    """
    C: list[Claim] = []

    def add(quoted, artifact, actual, label, **kw):
        C.append(_c(quoted, artifact, actual, label, doc=PRED, **kw))

    # ── scope: regular season only ────────────────────────────────────────────
    add("66%", PROFILE,
        lambda: prof("playoff_scope", "starter_24plus",
                     "share_playing_more_in_playoffs"),
        "starters playing more in the playoffs")
    add("5,762", PROFILE,
        lambda: prof("playoff_scope", "all", "n_with_playoff_appearance"),
        "playoff-scope population")
    for key, ratio, appearance in [("bench_lt12", "0.505", "0.711"),
                                   ("rotation_12_24", "0.761", "0.905"),
                                   ("starter_24plus", "1.054", "0.958")]:
        add(ratio, PROFILE,
            lambda k=key: prof("playoff_scope", k, "median_playoff_to_regular_mpg"),
            f"playoff MPG ratio, {key}")
        add(appearance, PROFILE,
            lambda k=key: prof("playoff_scope", k, "playoff_appearance_rate"),
            f"playoff appearance rate, {key}")
    # The superseded per-(player, team) denominator, kept beside its correction.
    for key, old in [("bench_lt12", "0.664"), ("rotation_12_24", "0.838"),
                     ("starter_24plus", "0.922")]:
        add(old, PROFILE,
            lambda k=key: prof("playoff_scope", k, "playoff_appearance_rate"),
            f"superseded appearance rate, {key}", historical=True)

    # ── what's already decided ────────────────────────────────────────────────
    add("210.3", SEASON_TOTAL,
        lambda: (treatment("full_season", "mae_dk_total")
                 - treatment("beta_binomial", "mae_dk_total")),
        "availability head vs a full season")

    # ── the collapse ──────────────────────────────────────────────────────────
    add("731,906", GAME_LEN,
        lambda: cell(GAME_LEN, "player_games", analysis="feasibility", season="all"),
        "player-games in the fitting frame")

    # ── is there a hot streak? ────────────────────────────────────────────────
    # The whole table, because this is a doc that had a partially-refreshed one before.
    serial_rows = [("min", "0.278", "−0.016", "+0.294", "2.43"),
                   ("fg3a|fga", "0.085", "−0.016", "+0.101", "1.57"),
                   ("fga", "0.045", "−0.016", "+0.061", "1.38"),
                   ("ast", "0.024", "−0.016", "+0.039", "1.22"),
                   ("fta", "0.015", "−0.016", "+0.031", "1.18"),
                   ("reb", "0.014", "−0.015", "+0.030", "1.17"),
                   ("blk", "0.007", "−0.016", "+0.023", "1.13"),
                   ("stl", "−0.003", "−0.016", "+0.013", "1.08"),
                   ("tov", "−0.006", "−0.014", "+0.008", "1.07"),
                   ("ftm|fta", "−0.014", "−0.026", "+0.012", "1.10"),
                   ("fg2m|fg2a", "−0.015", "−0.018", "+0.002", "1.03"),
                   ("fg3m|fg3a", "−0.019", "−0.017", "−0.002", "1.01")]
    for comp, lag1, null, excess, block in serial_rows:
        add(lag1, SERIAL, lambda c=comp: serial(c, "lag1"), f"serial {comp} lag-1")
        add(null, SERIAL, lambda c=comp: serial(c, "lag1_null"),
            f"serial {comp} null")
        add(excess.lstrip("+"), SERIAL, lambda c=comp: serial(c, "lag1_excess"),
            f"serial {comp} excess")
        add(block, SERIAL, lambda c=comp: serial(c, "block_inflation"),
            f"serial {comp} block inflation")
    add("592,796", SERIAL, lambda: max_of(SERIAL, "n_pairs") + 9052,
        "serial-correlation player-games")
    # The pre-refresh cells, preserved in the ⚠️ note beside the table. The established-facts
    # section had already been refreshed and this table had not, which is the drift the
    # audit found.
    for comp, column, old in [("min", "lag1_excess", "+0.297"),
                              ("min", "lag1_null", "−0.017"),
                              ("fg3a", "lag1", "0.061"),
                              ("fg3a", "lag1_excess", "+0.077"),
                              ("ftm|fta", "lag1", "−0.008"),
                              ("ftm|fta", "lag1_null", "−0.023"),
                              ("ftm|fta", "lag1_excess", "+0.015"),
                              ("ftm|fta", "block_inflation", "1.11")]:
        add(old.lstrip("+"), SERIAL, lambda c=comp, k=column: serial(c, k),
            f"superseded serial {comp} {column}", historical=True)
    for comp, z in [("fg3m|fg3a", "−1.3"), ("fg2m|fg2a", "1.9")]:
        add(z, SERIAL, lambda c=comp: serial(c, "lag1_z"), f"serial {comp} z")
    for lag, quoted in [("lag1", "0.278"), ("lag2", "0.212"), ("lag3", "0.170"),
                        ("lag5", "0.113")]:
        add(quoted, SERIAL, lambda l=lag: serial("min", l), f"minutes {lag}")
    add("0.196", SERIAL, lambda: serial("min_detrended", "lag1"),
        "detrended minutes lag-1")
    # The AR(1) comparator is φ and its powers, so it is derived from the same column
    # rather than stored — which is why the prose and the table must agree on φ.
    for power, quoted in [(2, "0.078"), (3, "0.022")]:
        add(quoted, SERIAL, lambda k=power: serial("min", "lag1") ** k,
            f"AR(1) at lag {power}, prose")
    for power, quoted in [(2, "0.0775"), (3, "0.0216")]:
        add(quoted, SERIAL, lambda k=power: serial("min", "lag1") ** k,
            f"AR(1) at lag {power}, table")
    add("0.2785", SERIAL, lambda: serial("min", "lag1"), "AR(1) phi")

    # ── the deferred per-game question ────────────────────────────────────────
    add("0.0776", STAN_MIN_D,
        lambda: cell(STAN_MIN_D, "rho", metric="game_level_rho"), "game-level rho")
    add("4.65", STAN_MIN_D,
        lambda: cell(STAN_MIN_D, "implied_overdispersion", metric="game_level_rho"),
        "game-level overdispersion")
    add("0.8835", STAN_MIN_M,
        lambda: cell(STAN_MIN_M, "val_r2", variant="logit_own_spline"),
        "minutes head val R2")
    add("0.8536", STAN_MIN_M,
        lambda: cell(STAN_MIN_M, "val_r2", variant="carry_forward"),
        "minutes no-fit floor")
    # The per-game costing extrapolates these two, so they have to be the fits the artifact
    # actually holds. Since the lock there is only a `/val` fit per variant.
    for variant, quoted in [("logit_own_spline", "955"), ("linear", "175")]:
        add(quoted, STAN_MIN_G,
            lambda v=variant: cell(STAN_MIN_G, "wall_clock_s", label=f"{v}/val"),
            f"minutes {variant} wall clock")
    # This doc quotes only the figures it uses, not the whole retired table. `+0.041` is its
    # own rounding of `+0.0407` and belongs to this doc alone, so it is not in the shared set.
    C += _minutes_pre_lock_claims(PRED, keep=("0.8572", "0.8166", "899", "189", "−21.4"))
    add("+0.041", STAN_MIN_M, lambda: float("nan"),
        "pre-lock held-out minutes head: R2 gain over floor, rounded", historical=True)

    # ── the composition alternative, summarised back into this doc ────────────
    # Validation figures since the 2026-08-08 refit; the retired test column lives in
    # `docs/minutes-composition-plan.md` and is claimed there. `independent_comparator`
    # is invariant to the window and reproduced to six decimals. See `_composition`.
    for variant, quoted in [("betabinom_ot_graded", "4.4945"),
                            ("carry_forward", "4.6776"),
                            ("independent_comparator", "4.7842"),
                            ("binomial", "4.9388")]:
        add(quoted, COMP_M, lambda v=variant: cell(COMP_M, "val_crps", variant=v),
            f"composition {variant} val CRPS")
    add("0.1919", COMP_M,
        lambda: cell(COMP_M, "val_pit_ks", variant="binomial"),
        "composition binomial PIT KS")
    add("33.89", COMP_P,
        lambda: cell(COMP_P, "simulated", variant="betabinom_ot_graded",
                     analysis="team_sum_abs_error",
                     group="independent"), "comparator team-sum error")
    add("−0.2898", COMP_M,
        lambda: (cell(COMP_M, "val_crps", variant="betabinom_ot_graded")
                 - cell(COMP_M, "val_crps", variant="independent_comparator")),
        "composition CRPS gain")
    for tier, quoted in [("q1_fringe", "1.05"), ("q4_star", "0.81")]:
        add(quoted, COMP_P,
            lambda t=tier: cell(COMP_P, "ratio", variant="betabinom_ot_graded",
                                analysis="variance_ratio", group=t),
            f"composition variance ratio {tier}")

    # ── why not an explicit lagged term ───────────────────────────────────────
    add("583,744", SERIAL, lambda: serial("min", "n_pairs"), "AR(1) decay pairs")
    for lag, quoted in [("lag1", "0.2785"), ("lag2", "0.2116"), ("lag3", "0.1698"),
                        ("lag5", "0.1131"), ("lag10", "0.0339")]:
        add(quoted, SERIAL, lambda l=lag: serial("min", l), f"AR decay {lag}")
    add("2.432", SERIAL, lambda: serial("min", "block_inflation"),
        "measured block inflation")
    add("1.706", SERIAL, lambda: serial("min_detrended", "block_inflation"),
        "detrended block inflation")

    # ── season effects ────────────────────────────────────────────────────────
    effects = [("fg3a_pct", "2.64", "+3.58", "0.93", "6.19"),
               ("fta", "1.21", "−0.55", "0.63", "4.36"),
               ("blk", "1.14", "−0.14", "0.12", "3.70"),
               ("ast", "1.30", "+0.73", "0.64", "3.08"),
               ("stl", "1.18", "−0.08", "0.03", "3.03"),
               ("tov", "1.16", "−0.33", "0.59", "2.89"),
               ("fga", "1.14", "+0.47", "0.84", "1.52"),
               ("fg3m_pct", "1.08", "+0.10", "0.27", "2.06"),
               ("reb", "1.11", "+0.25", "0.59", "1.48"),
               ("fg2m_pct", "1.20", "+0.61", "0.84", "1.46"),
               ("ftm_pct", "1.08", "+0.19", "0.77", "1.06"),
               ("minutes_share", "1.09", "−0.25", "0.81", "0.92"),
               ("gp_share [<12 mpg]", "1.48", "−0.71", "0.48", "9.04"),
               ("gp_share [12-24]", "1.30", "−0.54", "0.64", "4.21"),
               ("gp_share [24-30]", "1.25", "−0.39", "0.45", "3.85"),
               ("gp_share [all]", "1.24", "−0.51", "0.71", "2.86"),
               ("gp_share [30+ mpg]", "1.21", "−0.49", "0.74", "2.84")]
    for quantity, band, trend, r2, sd in effects:
        add(band, SEASON_EFF, lambda q=quantity: season_eff(q, "max_over_min"),
            f"season effect {quantity} band")
        add(trend.lstrip("+"), SEASON_EFF,
            lambda q=quantity: season_eff(q, "trend_pct_per_season"),
            f"season effect {quantity} trend")
        add(r2, SEASON_EFF, lambda q=quantity: season_eff(q, "trend_r2"),
            f"season effect {quantity} trend R2")
        add(sd, SEASON_EFF, lambda q=quantity: season_eff(q, "yoy_sd_pct"),
            f"season effect {quantity} yoy sd")
    add("−24.7%", SEASON_EFF,
        lambda: season_eff("fg3a_pct", "worst_yoy_pct") / 100.0,
        "three-point mix worst year")
    add("+3.58%", SEASON_EFF,
        lambda: season_eff("fg3a_pct", "trend_pct_per_season") / 100.0,
        "three-point mix trend, prose")
    add("4.4%", SEASON_EFF, lambda: season_eff("fta", "yoy_sd_pct") / 100.0,
        "fta yoy sd, prose")
    # The six `fta` swings past ±5%, which are the case that prompted the whole section.
    for season, quoted in [("2004-05", "+7.6%"), ("2011-12", "−7.8%"),
                           ("2017-18", "−6.0%"), ("2022-23", "+7.3%"),
                           ("2023-24", "−7.5%"), ("2025-26", "+8.6%")]:
        add(quoted, SEASON_RATES,
            lambda sn=season: _one(table(SEASON_RATES), "yoy_pct", quantity="fta",
                                   season=sn) / 100.0,
            f"fta yoy {season}")
    C.extend(_carry_bias_claims(PRED))

    # ── the rate side ─────────────────────────────────────────────────────────
    # Poisson/sklearn arm — the one the NB block below overturns. Kept apart on purpose.
    #
    # `fg3a` was RETIRED as a count head when the shot-attempt basis landed (2026-08-03),
    # so `component_rate_metrics.csv` no longer carries its rows. Its figures stay in the
    # prose because the finding they support — linear-in-raw-rate inside `exp()` is
    # catastrophically misspecified on a skewed count — is what motivated the whole
    # scale-not-curvature rule, and `blk` still demonstrates it live. They are marked
    # historical: presence-checked so they cannot be deleted, value-exempt because the row
    # they measured no longer exists. Leaving them unmarked would let them decay into the
    # module's "missing artifact" skip, which is meant for a fresh checkout, not for a
    # claim that has quietly stopped describing anything.
    # `blk` is the live demonstration and `fg3a` is the retired one; since 2026-08-05 the
    # live figures are also on a different split from the recorded ones, so this block
    # carries three generations of the same claim and only the newest is value-checked.
    add("0.651", RATES, lambda: rate("blk", "linear"), "poisson blk linear R2")
    add("0.775", RATES, lambda: rate("blk", "log_own"), "poisson blk log_own R2")
    add("0.054", RATES,
        lambda: rate("blk", "log_own_spline") - rate("blk", "log_own"),
        "poisson blk spline gain")
    add("0.7748", RATES, lambda: rate("blk", "log_own"),
        "poisson blk log_own R2, 4dp")
    for quoted in ("0.520", "0.879", "0.030", "0.8791",     # the retired fg3a head
                   "0.637", "0.820", "0.040", "0.8204"):    # blk on the retired split
        add(quoted, RATES, lambda: float("nan"),
            f"superseded poisson figure: {quoted}", historical=True)
    # "the best of seven fitted variants beats the floor by +0.0019 to +0.0228" — the
    # range over the seven count heads, which is what makes the floor binding.
    count_heads = ("fga", "fta", "reb", "ast", "stl", "blk", "tov")
    fitted = ("linear", "log_own", "log_own_spline", "log_own_inter", "pca",
              "pca_spline", "pca_inter")
    def _best_gain(head: str) -> float:
        return max(rate(head, v) for v in fitted) - rate(head, "carry_forward")
    add("0.0013", RATES, lambda: min(_best_gain(h) for h in count_heads),
        "smallest gain over the no-fit floor")
    add("0.0334", RATES, lambda: max(_best_gain(h) for h in count_heads),
        "largest gain over the no-fit floor")
    add("214.4", SEASON_TOTAL, lambda: treatment("oracle_gp", "mae_dk_total"),
        "oracle GP MAE")
    add("261.9", SEASON_TOTAL, lambda: treatment("oracle_rate", "mae_dk_total"),
        "oracle rate MAE")
    # The conversion heads' fitted variants are named differently from the count heads'
    # in the sklearn run — `spline_own` / `inter` / `pca_inter`, no `log_own` — so the
    # "best fitted" minimum has to enumerate the right five.
    conversion_variants = ("linear", "spline_own", "inter", "pca", "pca_inter")
    for head, quoted in [("fg2m|fg2a", "0.079"), ("fg3m|fg3a", "0.028")]:
        add(quoted, RATES,
            lambda h=head: rate(h, "carry_forward", "nll")
            - min(rate(h, v, "nll") for v in conversion_variants),
            f"poisson {head} NLL gain")

    # ── the built Stan block (negative binomial) ──────────────────────────────
    add("9.8195", STAN_AV_M, lambda: metric(STAN_AV_M, "stan_plug_in", "crps_games"),
        "stan availability CRPS")
    add("9.8237", STAN_AV_M,
        lambda: metric(STAN_AV_M, "mixture_mle", "crps_games"), "mixture MLE CRPS")
    add("0.2261", STAN_AV_M,
        lambda: metric(STAN_AV_M, "stan_plug_in", "dispersion_rho"), "stan rho")
    add("0.2245", STAN_AV_M,
        lambda: metric(STAN_AV_M, "mixture_mle", "dispersion_rho"), "mixture MLE rho")
    add("1.0073", STAN_AV_D, lambda: cell(STAN_AV_D, "max_rhat"), "stan R-hat")
    add("366", STAN_AV_D, lambda: cell(STAN_AV_D, "wall_clock_s"),
        "stan wall clock")
    add("0.5%", STAN_AV_B,
        lambda: cell(STAN_AV_B, "inflation", n_players=15) - 1.0,
        "board inflation, 15 players")
    add("14.8%", STAN_AV_B,
        lambda: cell(STAN_AV_B, "inflation", n_players=883) - 1.0,
        "board inflation, all 883")
    # The single-component role-graded head this replaced on 2026-08-12.
    for quoted in ("9.8136", "9.8444", "0.2595", "0.2627", "1.0050", "94", "12.3%"):
        add(quoted, STAN_AV_M, lambda: float("nan"),
            f"pre-mixture single-component availability head: {quoted}", historical=True)
    for quoted in ("10.7947", "10.7952", "0.2759", "0.2757", "1.0025", "254",
                   "6.4%", "911"):
        add(quoted, STAN_AV_M, lambda: float("nan"),
            f"pre-lock held-out availability port: {quoted}", historical=True)
    # The full-window, shared-rho head this replaced on 2026-08-11.
    for quoted in ("10.0063", "10.0057", "0.2808", "0.2806", "1.0019", "196", "6.7%"):
        add(quoted, STAN_AV_M, lambda: float("nan"),
            f"pre-window full-sample availability head: {quoted}", historical=True)
    add("−17.5", STAN_MIN_M,
        lambda: (cell(STAN_MIN_M, "val_crps", variant="logit_own_spline")
                 - cell(STAN_MIN_M, "val_crps", variant="carry_forward")),
        "minutes CRPS gain over floor")
    add("+0.030", STAN_MIN_M,
        lambda: (cell(STAN_MIN_M, "val_r2", variant="logit_own_spline")
                 - cell(STAN_MIN_M, "val_r2", variant="carry_forward")),
        "minutes R2 gain over floor")
    add("137.4", STAN_C_D, lambda: total(STAN_C_D, "wall_clock_s") / 60,
        "component sampler minutes")
    add("305.0", STAN_C_D, lambda: total(STAN_C_D, "wall_clock_s") / 60,
        "component sampler minutes, both splits", historical=True)
    add("1.0076", STAN_C_D, lambda: max_of(STAN_C_D, "max_rhat"),
        "component max R-hat")
    add("1.0118", STAN_C_D, lambda: max_of(STAN_C_D, "max_rhat"),
        "component max R-hat, half-length selection fits", historical=True)
    # Two retirements are layered here and they are NOT the same thing. `fg3a` is a retired
    # count HEAD (shot-attempt basis, 2026-08-04) whose rows left the artifact; `test_r2` is
    # a retired COLUMN (held-out lock, 2026-08-06) that left every head at once. Both are
    # carried as historical, so a reader can see which basis and which split a figure came
    # from rather than finding one number where two measurements used to be.
    for head, quoted, retired in [("blk", "0.6730", False), ("fg3a", "0.3719", True)]:
        add(quoted, STAN_C_M, lambda h=head: stan_c(h, "log_own", "val_r2"),
            f"NB {head} log_own R2", historical=retired)
    add("0.6794", STAN_C_M, lambda: stan_c("blk", "log_own", "val_r2"),
        "NB blk log_own R2, on test", historical=True)
    for head, quoted, retired in [("blk", "0.8309", False), ("fg3a", "0.9046", True)]:
        add(quoted, STAN_C_M,
            lambda h=head: stan_c(h, "log_own_spline", "val_r2"),
            f"NB {head} spline R2", historical=retired)
    add("0.8579", STAN_C_M, lambda: stan_c("blk", "log_own_spline", "val_r2"),
        "NB blk spline R2, on test", historical=True)
    add("−19.00", STAN_C_M, lambda: stan_c("fg3a", "linear", "val_r2"),
        "NB fg3a linear R2", historical=True)
    for quoted, variant, label in [("0.9514", "carry_forward", "fga floor"),
                                   ("0.9489", "linear", "fga linear R2"),
                                   ("0.9581", "log_own", "fga log_own R2"),
                                   ("0.9584", "log_own_spline", "fga selected R2")]:
        add(quoted, STAN_C_M, lambda v=variant: stan_c("fga", v, "val_r2"),
            f"NB {label}")
    for quoted, variant, label in [("0.9464", "carry_forward", "fga floor"),
                                   ("0.9396", "linear", "fga linear R2"),
                                   ("0.9501", "log_own", "fga log_own R2"),
                                   ("0.9505", "log_own_spline", "fga selected R2")]:
        add(quoted, STAN_C_M, lambda v=variant: stan_c("fga", v, "val_r2"),
            f"NB {label}, on test", historical=True)
    add("−0.2744", STAN_C_M, lambda: stan_c("blk", "linear", "val_r2"),
        "NB blk linear R2")
    add("−1.393", STAN_C_M, lambda: stan_c("blk", "linear", "val_r2"),
        "NB blk linear R2, on test", historical=True)
    # The reversal this conversion turned up: `fta` failed its floor by 0.0024 on test and
    # clears it by 0.0144 on validation. Both readings are claimed — the live pair against
    # the artifact, the test pair for the record — because the finding it retired ("the
    # whole free-throw family fails") is only legible next to the numbers that produced it.
    add("0.8909", STAN_C_M, lambda: stan_c("fta", "log_own", "val_r2"),
        "NB fta selected R2")
    add("0.8765", STAN_C_M, lambda: stan_c("fta", "carry_forward", "val_r2"),
        "NB fta floor")
    add("0.8649", STAN_C_M, lambda: stan_c("fta", "log_own", "val_r2"),
        "NB fta best fitted R2, on test", historical=True)
    add("0.8673", STAN_C_M, lambda: stan_c("fta", "carry_forward", "val_r2"),
        "NB fta floor, on test", historical=True)
    # The validation half comes from `stan_component_substitution.csv`, which the 2026-08-06
    # refit rewrote and which reproduced it to the quoted precision. The test half is
    # history: it outlived that refit inside the Gate 0 sweep, and then that sweep went
    # validation-only too. Presence-checked only — there is no artifact left to check it
    # against, which is the honest state rather than a gap.
    add("−0.771", STAN_C_S,
        lambda: cell(STAN_C_S, "reparam_minus_canonical", split="val",
                     arm="two_counts"), "substitution gain, val")
    add("−0.793", SHOT_SWEEP, lambda: float("nan"),
        "substitution gain, test", historical=True)
    for arm, quoted in [("two_counts", "10.797"), ("fga_x_fg3a_share", "10.026")]:
        add(quoted, STAN_C_S,
            lambda a=arm: cell(STAN_C_S, "mean_joint_nll", split="val", arm=a),
            f"substitution joint NLL val/{arm}")
    for arm, quoted in [("two_counts", "10.784"), ("fga_x_fg3a_share", "9.991")]:
        add(quoted, SHOT_SWEEP, lambda: float("nan"),
            f"substitution joint NLL test/{arm}", historical=True)

    # ── the residual copula ───────────────────────────────────────────────────
    add("0.0225", RESID, lambda: resid("mean", kind="count"),
        "residual off-diagonal mean, 7 counts")
    add("0.1329", RESID, lambda: resid("max"), "residual max off-diagonal")

    # ── ADP, as summarised back into this doc ─────────────────────────────────
    add("14.7%", ROSTER_A, lambda: roster("undescribed"),
        "roster minutes undescribed")
    add("8.7%", ROSTER_A, lambda: roster("rookie"), "rookie minutes")
    add("5.1%", ROSTER_A, lambda: roster("sub_threshold"), "sub-threshold minutes")
    add("0.9%", ROSTER_A, lambda: roster("returnee"), "returnee minutes")
    add("24.4", ADP_PROFILE, lambda: adp("ladder", "consensus raw"),
        "consensus raw error")
    add("17.3", ADP_PROFILE,
        lambda: adp("ladder", "+ monotone (isotonic) rescale"),
        "isotonic recalibrated error")
    add("−2.0", ADP_PROFILE,
        lambda: (adp("ladder", "+ isotonic + C/F/G offset")
                 - adp("ladder", "+ monotone (isotonic) rescale")),
        "position offset gain")
    # The planning-session value on 218 pairs, kept beside its correction. It is still the
    # live figure in `docs/adp-plan.md`'s scratch section, on that population.
    add("−0.3", ADP_PROFILE,
        lambda: (adp("ladder", "+ isotonic + C/F/G offset")
                 - adp("ladder", "+ monotone (isotonic) rescale")),
        "superseded position offset gain", historical=True)
    add("−2.01", ADP_PROFILE,
        lambda: (adp("ladder", "+ isotonic + C/F/G offset")
                 - adp("ladder", "+ monotone (isotonic) rescale")),
        "position offset gain, 2dp")
    add("−0.29", ADP_PROFILE,
        lambda: (adp("ladder", "+ isotonic + C/F/G offset")
                 - adp("ladder", "+ monotone (isotonic) rescale")),
        "superseded position offset gain, 2dp", historical=True)
    add("11.6%", ADP_PROFILE,
        lambda: ((adp("ladder", "+ monotone (isotonic) rescale")
                  - adp("ladder", "+ isotonic + C/F/G offset"))
                 / adp("ladder", "+ monotone (isotonic) rescale")),
        "position offset as a share of recalibrated error")
    add("−7.1", ADP_PROFILE,
        lambda: (adp("ladder", "+ monotone (isotonic) rescale")
                 - adp("ladder", "consensus raw")),
        "monotone step gain")

    # ── the season-term ablation ──────────────────────────────────────────────
    # The full block. `docs/facts-archive.md` carries a summary of the same verdict and
    # claims the overlapping figures against the SAME artifact — deliberately, because a block going
    # stale in one doc while current in the other is this repo's recorded failure mode and
    # it has already happened twice.
    C += _season_term_claims(PRED)
    return C


def _term_margin(head: str) -> float:
    """Validation gain of the selected arm over `base`, as a FRACTION of base.

    Returned as a fraction because `check_values` scales percent-quoted claims itself.

    The selection metric differs by head kind — `season_terms._finalize` is called with
    `val_nll` for the conversion sweep and `val_crps` for the counts, minutes and
    availability — so this keys on **`kind`**, which is what decides it.

    **It used to key on whether `val_nll` was populated, and the 2026-08-07 re-run broke
    that.** The count sweep now writes `val_nll` where it previously left the column NaN,
    so the "is it populated" heuristic silently switched the count heads onto a metric they
    are not selected on. The visible symptom was `stl` reporting a 0.0081% margin while its
    selected arm was `base` — an arm that is the minimum by construction can only have a
    margin of zero, so a nonzero one meant the margin and the selection were reading
    different columns. Keying on `kind` cannot drift that way: a column can gain values,
    but a conversion head cannot stop being a conversion head.
    """
    frame = table(TERM_M)
    if frame is None:
        return float("nan")
    # Restricted to the four ablation arms on purpose: `oracle_league` is a CEILING, not
    # a candidate, and letting it into the minimum would inflate every margin here — which
    # is the opposite of the point, since the point is that the margins are tiny.
    arms = ("base", "trend", "year", "trend_year")
    blk = frame[(frame["head"] == head) & (frame["arm"].isin(arms))]
    col = "val_nll" if (blk["kind"] == "conversion").all() else "val_crps"
    by_arm = blk.set_index("arm")[col]
    base = float(by_arm["base"])
    return abs(base - float(by_arm.min())) / abs(base)


def _season_term_summary_claims(doc: str) -> list[Claim]:
    """The subset of the verdict `docs/facts-archive.md` quotes, against the same artifact.

    A subset rather than the whole block because the archive is a summary and does not
    carry every cell — forcing it to would make the two docs the same document. What it
    does carry is claimed here, so the two cannot drift apart on the figures they share.
    """
    C: list[Claim] = []
    add = C.append
    for head, quoted in [("fg3a|fga", "0.05%"), ("fg3m|fg3a", "0.01%"),
                         ("ftm|fta", "0.09%"), ("gp", "0.11%"), ("min", "0.20%"),
                         ("fga", "0.26%"), ("blk", "0.35%"), ("fta", "0.79%"),
                         ("fg2m|fg2a", "1.88%"), ("ast", "2.21%")]:
        add(_c(quoted, TERM_M, lambda h=head: _term_margin(h),
               f"season-term selection margin, {head}", doc=doc))
    # `fg3a` is a RETIRED head — the shot-attempt basis replaced it with `fga` x `fg3a|fga`
    # on 2026-08-03, so none of these have a row to check any more. They are historical
    # rather than deleted because this head is the whole argument against a trend, and the
    # argument is only legible with its figures attached. Marking them historical also
    # takes them off the schema: a value-checked claim on a retired head degrades to a
    # `no-such-row` skip today and to a hard `schema-change` failure the moment its column
    # is renamed, which is a failure about the artifact's shape rather than about the doc.
    for quoted, label in [("33.247", "fg3a base val CRPS"),
                          ("35.443", "fg3a trend val CRPS"),
                          ("34.309", "fg3a year val CRPS"),
                          ("−3.74%", "fg3a base bias"),
                          ("+8.44%", "fg3a trend bias"),
                          ("−9.01%", "fg3a year bias")]:
        add(_c(quoted, TERM_M, lambda: float("nan"), label, doc=doc, historical=True))
    # The oracle ceiling, now on `val_mae`. Its ORDERING reshuffled completely across the
    # split move while its SIZE did not, which is the same noise argument the margins make.
    for head, quoted in [("fta", "4.97%"), ("reb", "2.58%"), ("tov", "2.36%"),
                         ("ast", "1.29%"), ("blk", "1.17%"), ("stl", "0.36%")]:
        add(_c(quoted, TERM_M, lambda h=head: oracle_gain(h),
               f"oracle ceiling {head}", doc=doc))
    for quoted, label in [("3.11%", "oracle ceiling stl"), ("2.32%", "oracle ceiling blk"),
                          ("2.16%", "oracle ceiling fta"), ("1.71%", "oracle ceiling reb"),
                          ("0.81%", "oracle ceiling fga"), ("0.58%", "oracle ceiling ast"),
                          ("0.01%", "oracle ceiling tov")]:
        add(_c(quoted, TERM_M, lambda: float("nan"),
               f"{label}, held-out reading", doc=doc, historical=True))
    add(_c("105.71", TERM_TOTAL, lambda: term_total("base", "mae"),
           "season total base MAE", doc=doc))
    add(_c("105.76", TERM_TOTAL, lambda: term_total("trend", "mae"),
           "season total trend MAE", doc=doc))
    add(_c("−16.44", TERM_TOTAL, lambda: term_total("base", "bias"),
           "season total base bias", doc=doc))
    add(_c("+10.21", TERM_TOTAL, lambda: term_total("trend", "bias"),
           "season total trend bias", doc=doc))
    for quoted, label in [("106.06", "season total trend MAE"),
                          ("108.56", "season total base MAE"),
                          ("−13.57", "season total trend bias"),
                          ("−36.89", "season total base bias")]:
        add(_c(quoted, TERM_TOTAL, lambda: float("nan"),
               f"{label}, held-out reading", doc=doc, historical=True))
    # `sigma_year` moved because the old column was the TEST arm's fit (27 training
    # seasons) sitting in a row whose CRPS came from the validation arm (25). Same head,
    # two different models, one row. Now both come from the validation fit.
    for head, sigma in [("blk", "0.85×"), ("tov", "0.89×"), ("reb", "1.07×")]:
        add(_c(sigma, TERM_SIGMA, lambda h=head: term_sigma(h, "ratio_to_yoy_sd"),
               f"sigma/league ratio {head}", doc=doc))
    for quoted, label in [("0.99×", "blk"), ("0.94×", "tov"), ("0.87×", "fta"),
                          ("0.82×", "stl"), ("1.18×", "reb"), ("2.36×", "fg3a")]:
        add(_c(quoted, TERM_SIGMA, lambda: float("nan"),
               f"sigma/league ratio {label}, test-arm reading", doc=doc, historical=True))
    # The whole-board row is keyed on the board SIZE, so the validation board's 773 is a
    # different lookup from the test board's 791 — a stale key here would degrade to a
    # silent `no-such-row` skip rather than a mismatch, which is why it is requoted.
    for n, quoted in [(12, "+7.95%"), (15, "+10.4%"), (30, "+21.0%"),
                      (150, "+83.5%"), (773, "+254%")]:
        add(_c(quoted, TERM_SPREAD, lambda k=n: term_spread(k, "year") - 1.0,
               f"year roster spread inflation, {n} players", doc=doc))
    for quoted, label in [("+8.9%", "12"), ("+11.6%", "15"), ("+22.1%", "30"),
                          ("+91.1%", "150"), ("+278%", "791")]:
        add(_c(quoted, TERM_SPREAD, lambda: float("nan"),
               f"year roster spread inflation, {label} players, test board",
               doc=doc, historical=True))
    for arm, val in [("base", "144.09"), ("year", "143.81")]:
        add(_c(val, TERM_M, lambda a=arm: term("min", a, "val_crps"),
               f"minutes {arm} val CRPS", doc=doc))
    for quoted, label in [("147.02", "base"), ("146.54", "year")]:
        add(_c(quoted, TERM_M, lambda: float("nan"),
               f"minutes {label} test CRPS", doc=doc, historical=True))
    add(_c("−13.0", TERM_M, lambda: term("min", "year", "val_bias"),
           "minutes year val bias", doc=doc))
    add(_c("−25.0", TERM_M, lambda: term("min", "trend", "val_bias"),
           "minutes trend val bias", doc=doc))
    for quoted, label in [("−38.2", "year"), ("−56.6", "trend")]:
        add(_c(quoted, TERM_M, lambda: float("nan"),
               f"minutes {label} bias, held-out reading", doc=doc, historical=True))
    for arm, quoted in [("base", "−11.7%"), ("trend", "−7.6%"), ("year", "−12.8%")]:
        add(_c(quoted, TERM_BONUS, lambda a=arm: term_bonus(a) / 100.0,
               f"bonus bias {arm}", doc=doc))
    for quoted, label in [("−14.8%", "base"), ("−10.2%", "trend"), ("−16.5%", "year")]:
        add(_c(quoted, TERM_BONUS, lambda: float("nan"),
               f"bonus bias {label}, held-out reading", doc=doc, historical=True))
    return C


def _season_term_claims(doc: str) -> list[Claim]:
    """The `make season-terms` verdict, in full, for `docs/predictions-plan.md`."""
    C: list[Claim] = []
    add = C.append

    # `fg3a` — the head that refuted the trend, and the reason the verdict is what it is.
    # RETIRED with the two-count basis on 2026-08-03 and no longer fitted, so every figure
    # here is a record of that basis rather than a live measurement. Held historical for the
    # same reason the doc still carries the block: this is the sharpest evidence against a
    # trend the project ever produced, and its successor `fg3a|fga` cannot restate it — the
    # mix head selects `year` on a margin too small to mean anything.
    for arm, val, test, bias in [("base", "33.247", "33.723", "−3.74%"),
                                 ("trend", "35.443", "33.900", "+8.44%"),
                                 ("year", "34.309", "35.408", "−9.01%"),
                                 ("trend_year", "36.108", "34.706", "+11.48%")]:
        for quoted, what in [(val, "val CRPS"), (test, "test CRPS"), (bias, "bias")]:
            add(_c(quoted, TERM_M, lambda: float("nan"),
                   f"fg3a {arm} {what}", doc=doc, historical=True))
    add(_c("38.747", TERM_M, lambda: float("nan"),
           "fg3a floor CRPS", doc=doc, historical=True))

    # The oracle ceiling — the single number that bounds the whole question. On `val_mae`
    # since 2026-08-07. `fga` is quoted as a NEGATIVE gain and needs its own claim: a
    # perfect per-season rescale making MAE very slightly worse is a real property of a
    # head whose seasons barely move, not a defect, and rounding it to "0.0%" would hide
    # the sign.
    for head, quoted in [("fta", "4.97%"), ("reb", "2.58%"), ("tov", "2.36%"),
                         ("ast", "1.29%"), ("blk", "1.17%"), ("stl", "0.36%"),
                         ("fga", "−0.00%")]:
        add(_c(quoted, TERM_M, lambda h=head: oracle_gain(h),
               f"oracle ceiling {head}", doc=doc))
    for quoted, label in [("3.11%", "stl"), ("2.32%", "blk"), ("2.16%", "fta"),
                          ("1.71%", "reb"), ("0.81%", "fga"), ("0.58%", "ast"),
                          ("0.01%", "tov")]:
        add(_c(quoted, TERM_M, lambda: float("nan"),
               f"oracle ceiling {label}, held-out reading", doc=doc, historical=True))

    # Season-total dk_pts, uniform-arm, now on the 773-row VALIDATION frame.
    for arm, mae, bias, crps in [("base", "105.71", "−16.44", "76.05"),
                                 ("trend", "105.76", "+10.21", "75.67"),
                                 ("year", "106.52", "−23.40", "76.68"),
                                 ("trend_year", "105.00", "+9.22", "75.46")]:
        add(_c(mae, TERM_TOTAL, lambda a=arm: term_total(a, "mae"),
               f"season total {arm} MAE", doc=doc))
        add(_c(bias, TERM_TOTAL, lambda a=arm: term_total(a, "bias"),
               f"season total {arm} bias", doc=doc))
        add(_c(crps, TERM_TOTAL, lambda a=arm: term_total(a, "crps"),
               f"season total {arm} CRPS", doc=doc))
    for mae, bias, crps, arm in [("108.56", "−36.89", "78.27", "base"),
                                 ("106.06", "−13.57", "75.91", "trend"),
                                 ("110.56", "−45.87", "79.92", "year"),
                                 ("105.92", "−12.89", "76.06", "trend_year")]:
        for quoted, what in [(mae, "MAE"), (bias, "bias"), (crps, "CRPS")]:
            add(_c(quoted, TERM_TOTAL, lambda: float("nan"),
                   f"season total {arm} {what}, held-out reading",
                   doc=doc, historical=True))

    # `sigma_year` against the independently measured league movement. Every value moved,
    # and NOT because the sampler is noisy: the old `year_*` columns were written from the
    # TEST arm's fit (27 training seasons) into a row whose CRPS came from the validation
    # arm (25). `year_n_train_seasons` is the column that shows it.
    for head, sigma, league, ratio in [("blk", "3.15%", "3.70%", "0.85"),
                                       ("tov", "2.56%", "2.89%", "0.89"),
                                       ("fta", "3.23%", "4.36%", "0.74"),
                                       ("stl", "2.30%", "3.03%", "0.76"),
                                       ("reb", "1.58%", "1.48%", "1.07"),
                                       ("fga", "2.10%", "1.52%", "1.38"),
                                       ("ast", "4.32%", "3.08%", "1.40"),
                                       ("fg3a|fga", "11.42%", "6.19%", "1.84")]:
        add(_c(sigma, TERM_SIGMA, lambda h=head: term_sigma(h, "sigma_year_pct") / 100.0,
               f"sigma_year {head}", doc=doc))
        add(_c(league, TERM_SIGMA,
               lambda h=head: term_sigma(h, "league_yoy_sd_pct") / 100.0,
               f"league yoy sd {head}, beside sigma", doc=doc))
        add(_c(ratio, TERM_SIGMA, lambda h=head: term_sigma(h, "ratio_to_yoy_sd"),
               f"sigma/league ratio {head}", doc=doc))
    for sigma, league, ratio, head in [("3.66%", "3.70%", "0.99", "blk"),
                                       ("2.71%", "2.89%", "0.94", "tov"),
                                       ("3.78%", "4.36%", "0.87", "fta"),
                                       ("2.49%", "3.03%", "0.82", "stl"),
                                       ("1.75%", "1.48%", "1.18", "reb"),
                                       ("3.03%", "2.42%", "1.25", "fg2a"),
                                       ("4.46%", "3.08%", "1.45", "ast"),
                                       ("15.43%", "6.53%", "2.36", "fg3a")]:
        for quoted, what in [(sigma, "sigma_year"), (league, "league yoy sd"),
                             (ratio, "ratio")]:
            add(_c(quoted, TERM_SIGMA, lambda: float("nan"),
                   f"{what} {head}, test-arm reading", doc=doc, historical=True))

    # Roster spread — what the year effect is actually worth. Keyed on board SIZE, so the
    # validation board's 773 is a different row from the test board's 791.
    for n, quoted in [(12, "+7.95%"), (15, "+10.4%"), (30, "+21.0%"),
                      (150, "+83.5%"), (773, "+254%")]:
        add(_c(quoted, TERM_SPREAD, lambda k=n: term_spread(k, "year") - 1.0,
               f"year roster spread inflation, {n} players", doc=doc))
    add(_c("594", TERM_SPREAD, lambda: term_spread(15, "base", "total_sd"),
           "base roster sd, 15 players", doc=doc))
    add(_c("655", TERM_SPREAD, lambda: term_spread(15, "year", "total_sd"),
           "year roster sd, 15 players", doc=doc))
    for quoted, label in [("+8.9%", "12 players"), ("+11.6%", "15 players"),
                          ("+22.1%", "30 players"), ("+91.1%", "150 players"),
                          ("+278%", "791 players"), ("557", "base sd, 15"),
                          ("621", "year sd, 15")]:
        add(_c(quoted, TERM_SPREAD, lambda: float("nan"),
               f"roster spread {label}, test board", doc=doc, historical=True))

    # Minutes — the one head that adopts a season term. `val_crps` reproduced to four
    # decimals across the conversion; `val_bias` is a genuinely new quantity, since the old
    # bare `bias` column was the held-out one.
    for arm, val, bias in [("carry_forward", "161.45", "+23.9"),
                           ("base", "144.09", "−14.2"),
                           ("trend", "144.44", "−25.0"),
                           ("year", "143.81", "−13.0")]:
        add(_c(val, TERM_M, lambda a=arm: term("min", a, "val_crps"),
               f"minutes {arm} val CRPS", doc=doc))
        add(_c(bias, TERM_M, lambda a=arm: term("min", a, "val_bias"),
               f"minutes {arm} val bias", doc=doc))
    for test, bias, arm in [("168.24", "−5.69", "carry_forward"),
                            ("147.02", "−41.05", "base"),
                            ("148.78", "−56.59", "trend"),
                            ("146.54", "−38.19", "year")]:
        for quoted, what in [(test, "test CRPS"), (bias, "bias")]:
            add(_c(quoted, TERM_M, lambda: float("nan"),
                   f"minutes {arm} {what}, held-out reading", doc=doc, historical=True))

    # Availability — the role-interaction arms. Their whole finding was a val/test
    # DISAGREEMENT, so the test half is now historical and the doc keeps it as the record
    # of a false positive that selection caught.
    for arm, val in [("carry_forward", "13.387"), ("base", "10.007"), ("trend", "9.997"),
                     ("year", "10.015"), ("trend_year", "10.005"),
                     ("trend_x_role", "10.078"), ("trend_x_role_year", "10.087")]:
        add(_c(val, TERM_M, lambda a=arm: term("gp", a, "val_crps"),
               f"availability {arm} val CRPS", doc=doc))
    for quoted, arm in [("13.614", "carry_forward"), ("10.797", "base"),
                        ("10.765", "trend"), ("10.813", "year"), ("10.757", "trend_year"),
                        ("10.742", "trend_x_role"), ("10.736", "trend_x_role_year")]:
        add(_c(quoted, TERM_M, lambda: float("nan"),
               f"availability {arm} test CRPS", doc=doc, historical=True))

    # The bonus, whose LEVEL is the missing copula and whose ORDERING is the bias story.
    for arm, quoted in [("base", "−11.7%"), ("trend", "−7.6%"),
                        ("year", "−12.8%"), ("trend_year", "−6.9%")]:
        add(_c(quoted, TERM_BONUS, lambda a=arm: term_bonus(a) / 100.0,
               f"bonus bias {arm}", doc=doc))
    for quoted, arm in [("−14.8%", "base"), ("−10.2%", "trend"),
                        ("−16.5%", "year"), ("−9.9%", "trend_year")]:
        add(_c(quoted, TERM_BONUS, lambda: float("nan"),
               f"bonus bias {arm}, held-out reading", doc=doc, historical=True))
    return C


def _adp() -> list[Claim]:
    """`docs/adp-plan.md` — the market proxy, its sourcing, and where it belongs.

    **This doc is the reason `Claim.historical` exists.** It is written in two halves that
    quote the same quantities on different populations, and both are correct: "What I
    measured before planning" is a scratch-session run on the **218** pairs a looser join
    produced, and "What implementing it changed" is `adp_profile.csv` on the **226** pairs
    the guarded cascade recovers. Correcting the first half to the artifact would delete
    the record of what the implementation changed, which is the doc's whole point — so the
    planning figures are `historical`, presence-checked and never value-checked.
    """
    C: list[Claim] = []

    def add(quoted, artifact, actual, label, **kw):
        C.append(_c(quoted, artifact, actual, label, doc=ADP, **kw))

    DK_BOARD = "data/features/adp_draftkings.parquet"
    DK_MAP = "data/features/adp_dk_id_map.parquet"
    TRANSFER = "data/features/adp_transfer.parquet"

    def board(season: str, column: str, how: str = "count") -> float:
        frame = table(DK_BOARD)
        if frame is None:
            return float("nan")
        rows = frame[frame["season"] == season]
        if how == "pool":
            return float(rows["pool_size"].max())
        vals = rows[column].dropna()
        return {"count": lambda: float(len(vals)),
                "min": lambda: float(vals.min()),
                "max": lambda: float(vals.max()),
                "p25": lambda: float(vals.quantile(0.25)),
                "p50": lambda: float(vals.quantile(0.50)),
                "p75": lambda: float(vals.quantile(0.75))}[how]()

    def shared_ids() -> set:
        frame = table(DK_BOARD)
        if frame is None:
            return set()
        return (set(frame[frame["season"] == "2025-26"]["dk_player_id"])
                & set(frame[frame["season"] == "2026-27"]["dk_player_id"]))

    def audit(metric: str, rule: str = "cascade") -> float:
        return _one(table(ADP_AUDIT), "value", section="summary", rule=rule,
                    metric=metric)

    # ── the two DK boards ─────────────────────────────────────────────────────
    for season, pool, with_adp in [("2025-26", "698", "249"), ("2026-27", "942", "202")]:
        add(pool, DK_BOARD, lambda s=season: board(s, "adp", "pool"),
            f"DK {season} pool size")
        add(with_adp, DK_BOARD, lambda s=season: board(s, "adp"),
            f"DK {season} rows with ADP")
    add("35.7%", DK_BOARD,
        lambda: board("2025-26", "adp") / board("2025-26", "adp", "pool"),
        "DK 2025-26 ADP share")
    add("21.4%", DK_BOARD,
        lambda: board("2026-27", "adp") / board("2026-27", "adp", "pool"),
        "DK 2026-27 ADP share")
    ranges = [("2025-26", [("1.053", "min"), ("186.3", "max"), ("63.0", "p25"),
                           ("126.1", "p50"), ("178.6", "p75")]),
              ("2026-27", [("1.139", "min"), ("184.1", "max"), ("53.2", "p25"),
                           ("101.1", "p50"), ("153.7", "p75")])]
    for season, stats in ranges:
        for quoted, how in stats:
            add(quoted, DK_BOARD, lambda s=season, h=how: board(s, "adp", h),
                f"DK {season} ADP {how}")
    add("667", DK_BOARD, lambda: float(len(shared_ids())), "shared DK ids")
    add("100.0%", DK_BOARD, lambda: _dk_name_agreement(), "DK id name agreement")
    # The five named ADPs are the evidence that the July file is a 2026-27 board rather
    # than a re-serve of 2025-26, so they are worth pinning individually.
    for name, season, quoted in [("AJ Dybantsa", "2026-27", "41.8"),
                                 ("Cameron Boozer", "2026-27", "47.2"),
                                 ("Darryn Peterson", "2026-27", "53.2"),
                                 ("Tyrese Haliburton", "2026-27", "20.5"),
                                 ("Damian Lillard", "2026-27", "85.4")]:
        add(quoted, DK_BOARD,
            lambda n=name, s=season: _one(table(DK_BOARD), "adp", player_name=n,
                                          season=s),
            f"DK ADP for {name}")
    add("178", DK_BOARD, lambda: _dk_team_changes(), "DK team changes between boards")
    add("175", DK_BOARD, lambda: _dk_both_boards_adp(), "players with ADP on both boards")
    add("13.4%", DK_BOARD, lambda: _dk_no_prior_adp_share(),
        "2026-27 ADP'd players with no prior-year DK ADP")
    add("811", DK_MAP,
        lambda: float((table(DK_MAP)["match_method"] != "no_nba_history").sum())
        if table(DK_MAP) is not None else float("nan"), "matchable DK ids")
    add("162", DK_MAP,
        lambda: float((table(DK_MAP)["match_method"] == "no_nba_history").sum())
        if table(DK_MAP) is not None else float("nan"),
        "DK pool entries with no NBA history")

    # ── the match audit ───────────────────────────────────────────────────────
    add("0.50%", ADP_AUDIT, lambda: audit("unmatched_rate_cascade"),
        "cascade unmatched rate")

    # ── what implementing it changed — the auditable half ─────────────────────
    add("226", ADP_PROFILE, lambda: adp("agreement", "n_pairs"), "matched pairs")
    add("0.8675", ADP_PROFILE, lambda: adp("agreement", "spearman"),
        "Spearman, implemented")
    add("23.1", ADP_PROFILE, lambda: adp("agreement", "mean_abs_rank_gap"),
        "mean absolute rank gap, implemented")
    add("+13.88", ADP_PROFILE, lambda: adp("position_bias", "C:mean_rank_gap"),
        "centre bias, implemented")
    add("24.38", ADP_PROFILE, lambda: adp("ladder", "consensus raw"),
        "consensus raw, implemented")
    add("17.32", ADP_PROFILE,
        lambda: adp("ladder", "+ monotone (isotonic) rescale"),
        "isotonic rescale, implemented")
    add("5.09", ADP_PROFILE, lambda: adp("tier_gap", "R1-2:mean_abs_rank_gap"),
        "rounds 1-2 gap, implemented")
    add("32.31", ADP_PROFILE, lambda: adp("tier_gap", "R9+:mean_abs_rank_gap"),
        "rounds 9+ gap, implemented")
    add("1", TRANSFER,
        lambda: float(table(TRANSFER)["n_anchors"].iloc[0])
        if table(TRANSFER) is not None else float("nan"), "anchor count")

    # ── the planning-session half, kept for the record ────────────────────────
    scratch = [
        ("0.8704", lambda: adp("agreement", "spearman"), "Spearman"),
        ("0.870", lambda: adp("agreement", "spearman"), "Spearman, short version"),
        ("21.9", lambda: adp("agreement", "mean_abs_rank_gap"), "mean rank gap"),
        ("38.1%", lambda: adp("agreement", "share_gap_over_20"), "gaps over 20"),
        ("16.5%", lambda: adp("agreement", "share_gap_over_40"), "gaps over 40"),
        ("+11.86", lambda: adp("position_bias", "C:mean_rank_gap"), "centre bias"),
        ("+0.16", lambda: adp("position_bias", "F:mean_rank_gap"), "forward bias"),
        ("−4.78", lambda: adp("position_bias", "G:mean_rank_gap"), "guard bias"),
        ("24.02", lambda: adp("ladder", "consensus raw"), "consensus raw"),
        ("21.89", lambda: adp("ladder", "consensus order vs DK order"),
         "ordering only"),
        ("21.24", lambda: adp("ladder", "+ linear rescale"), "linear rescale"),
        ("16.99", lambda: adp("ladder", "+ monotone (isotonic) rescale"), "isotonic"),
        ("20.78", lambda: adp("ladder", "+ linear + C/F/G offset"), "linear + offset"),
        ("16.70", lambda: adp("ladder", "+ isotonic + C/F/G offset"),
         "isotonic + offset"),
        ("14.4", lambda: adp("ladder", "+ isotonic, uncensored only"), "uncensored"),
        ("5.1", lambda: adp("tier_gap", "R1-2:mean_abs_rank_gap"), "rounds 1-2 gap"),
        ("9.7", lambda: adp("tier_gap", "R3-4:mean_abs_rank_gap"), "rounds 3-4 gap"),
        ("10.5", lambda: adp("tier_gap", "R5-8:mean_abs_rank_gap"), "rounds 5-8 gap"),
        ("30.8", lambda: adp("tier_gap", "R9+:mean_abs_rank_gap"), "rounds 9+ gap"),
        ("87.6%", lambda: adp("agreement", "n_pairs") / 249.0, "matched share"),
        ("0.8911", lambda: adp("agreement", "spearman"), "Pearson r"),
        ("14.2", lambda: adp("agreement", "median_abs_rank_gap"), "median rank gap"),
        ("24.0", lambda: adp("ladder", "consensus raw"), "consensus raw, short version"),
        ("17.0", lambda: adp("ladder", "+ monotone (isotonic) rescale"),
         "isotonic, short version"),
        ("11.9", lambda: adp("position_bias", "C:mean_rank_gap"),
         "centre bias, short version"),
    ]
    for quoted, fn, label in scratch:
        add(quoted, ADP_PROFILE, fn, f"planning-session {label}", historical=True)

    # ── the roster-coverage exception ─────────────────────────────────────────
    add("14.7%", ROSTER_A, lambda: roster("undescribed"),
        "roster minutes undescribed")
    add("8.7%", ROSTER_A, lambda: roster("rookie"), "rookie minutes")
    add("5.1%", ROSTER_A, lambda: roster("sub_threshold"), "sub-threshold minutes")
    add("0.9%", ROSTER_A, lambda: roster("returnee"), "returnee minutes")

    return C


def _dk_team_changes() -> float:
    frame = table("data/features/adp_draftkings.parquet")
    if frame is None:
        return float("nan")
    a = frame[frame["season"] == "2025-26"].set_index("dk_player_id")["team"]
    b = frame[frame["season"] == "2026-27"].set_index("dk_player_id")["team"]
    shared = a.index.intersection(b.index)
    return float((a[shared] != b[shared]).sum())


def _dk_both_boards_adp() -> float:
    frame = table("data/features/adp_draftkings.parquet")
    if frame is None:
        return float("nan")
    with_adp = frame.dropna(subset=["adp"])
    return float(len(set(with_adp[with_adp["season"] == "2025-26"]["dk_player_id"])
                    & set(with_adp[with_adp["season"] == "2026-27"]["dk_player_id"])))


def _dk_name_agreement() -> float:
    frame = table("data/features/adp_draftkings.parquet")
    if frame is None:
        return float("nan")
    a = frame[frame["season"] == "2025-26"].set_index("dk_player_id")["player_name"]
    b = frame[frame["season"] == "2026-27"].set_index("dk_player_id")["player_name"]
    shared = a.index.intersection(b.index)
    return float((a[shared] == b[shared]).mean())


def _dk_no_prior_adp_share() -> float:
    frame = table("data/features/adp_draftkings.parquet")
    if frame is None:
        return float("nan")
    with_adp = frame.dropna(subset=["adp"])
    prior = set(with_adp[with_adp["season"] == "2025-26"]["dk_player_id"])
    now = with_adp[with_adp["season"] == "2026-27"]["dk_player_id"]
    return float((~now.isin(prior)).mean())


def _established_facts() -> list[Claim]:
    """The established-facts section — the source of truth the plan docs defer to.

    It lived in `CLAUDE.md` until 2026-08-08 and now lives across `docs/`, split by
    subject rather than moved as one block. So this is the one builder that is not
    one-doc-one-function: `into(...)` names the destination for the section that
    follows, and `add(..., doc=X)` overrides it for the individual figures that landed
    somewhere other than the rest of their section.

    Only the *figures* are claimed. The project-layout, pipeline, conventions and dashboard
    sections carry no measurements and need none, which is why these docs' coverage
    denominators are smaller than their length suggests.
    """
    C: list[Claim] = []
    here = FACTS

    def into(doc: str) -> None:
        """Set the destination for the claims that follow, until the next `into`."""
        nonlocal here
        here = doc

    def add(quoted, artifact, actual, label, doc=None, **kw):
        C.append(_c(quoted, artifact, actual, label, doc=doc or here, **kw))

    def context(feature: str, column: str) -> float:
        return _one(table(CONTEXT_A), column, feature=feature)

    def opp(outcome: str, column: str) -> float:
        return _one(table(OPPONENT_A), column, outcome=outcome)

    def bonus(unit: str, od: float, column: str, bucket: str = "all") -> float:
        return _windowed(BONUS, column, analysis="calibration", unit=unit,
                         bucket=bucket, overdispersion=od)

    def age_ratio(metric: str, age: int) -> float:
        return _one(table(AGING), "cumulative_ratio", tier="A", metric=metric,
                    archetype=-1, age=age)

    def glen(column: str, season: str = "all",
             analysis: str = "feasibility") -> float:
        return _one(table(GAME_LEN), column, analysis=analysis, season=season,
                    season_type="regular")

    # ── the variance budget ───────────────────────────────────────────────────
    into(FACTS)
    add("57.96%", VARIANCE, lambda: budget("player_season_identity"),
        "player-season identity share")
    add("46.40%", VARIANCE, lambda: budget("own_minutes"), "own minutes share")
    add("0.691%", VARIANCE, lambda: budget("opponent_x_season"),
        "opponent x season share")
    add("0.034%", VARIANCE, lambda: budget("home_away"), "home/away share")
    add("9.445", VARIANCE,
        lambda: budget("within_player_season_residual_sd", "value"), "residual sd")
    add("14.566", VARIANCE, lambda: budget("dk_pts_sd", "value"), "total sd")
    add("254,167", VARIANCE, lambda: budget("own_minutes", "n_games"),
        "variance-budget population")
    add("198,509", VARIANCE,
        lambda: budget("opponent_x_archetype_x_season", "n_games"),
        "archetype-panel population")
    add("18.6%", VARIANCE, lambda: budget("own_minutes_raw_level"),
        "superseded own-minutes share", historical=True)
    add("18.650%", VARIANCE, lambda: budget("own_minutes_raw_level"),
        "superseded own-minutes construction, reproduced")
    add("46.19%", VARIANCE, lambda: budget("own_minutes_nonparametric"),
        "own minutes, nonparametric")
    add("59.40%", VARIANCE, lambda: budget("own_minutes_saturated"),
        "own minutes, saturated bound")
    add("0.852", VARIANCE,
        lambda: budget("minutes_slope_bottom_mpg_tier", "value"),
        "dk_pts per minute, bottom tier")
    add("1.230", VARIANCE, lambda: budget("minutes_slope_top_mpg_tier", "value"),
        "dk_pts per minute, top tier")
    add("0.867", VARIANCE, lambda: budget("opponent_effect_sd", "value"),
        "opponent effect sd in dk_pts")
    add("0.793%", VARIANCE,
        lambda: budget("opponent_x_season_archetype_panel"),
        "opponent x season on the archetype panel")
    for null, above, ceiling, gap in [
            ("shuffle_opponent_within_season", "0.962%", "0.9570%", "0.0050"),
            ("shuffle_archetype_within_season", "0.308%", "0.3056%", "0.0024")]:
        add(above, VARIANCE,
            lambda n=null: budget("opponent_x_archetype_x_season_above_null",
                                  null_construction=n),
            f"interaction above {null}")
        # The ceilings and the two cell-importance figures below went to the notes
        # rather than to the archive with the rest of the budget.
        add(ceiling, VARIANCE,
            lambda n=null: budget(
                "opponent_x_archetype_x_season_above_null_variance_ceiling",
                null_construction=n),
            f"variance ceiling, {null}", doc=NOTES)
        add(gap, VARIANCE,
            lambda n=null: budget("null_reproduction_gap", null_construction=n) * 100,
            f"reproduction gap, {null}")
    add("0.9619%", VARIANCE,
        lambda: budget("opponent_x_archetype_x_season_above_null",
                       null_construction="shuffle_opponent_within_season"),
        "cell_importance, shuffle opponent", doc=NOTES)
    add("0.3080%", VARIANCE,
        lambda: budget("opponent_x_archetype_x_season_above_null",
                       null_construction="shuffle_archetype_within_season"),
        "cell_importance, shuffle archetype", doc=NOTES)

    # ── playoff scope and workload ────────────────────────────────────────────
    into(NOTES)
    add("5,762", PROFILE,
        lambda: prof("playoff_scope", "all", "n_with_playoff_appearance"),
        "playoff-scope population")
    for key, ratio, appearance in [("bench_lt12", "0.505", "0.711"),
                                   ("rotation_12_24", "0.761", "0.905"),
                                   ("starter_24plus", "1.054", "0.958")]:
        add(ratio, PROFILE,
            lambda k=key: prof("playoff_scope", k, "median_playoff_to_regular_mpg"),
            f"playoff MPG ratio, {key}")
        add(appearance, PROFILE,
            lambda k=key: prof("playoff_scope", k, "playoff_appearance_rate"),
            f"playoff appearance rate, {key}")
    for key, old in [("bench_lt12", "0.664"), ("rotation_12_24", "0.838"),
                     ("starter_24plus", "0.922")]:
        add(old, PROFILE,
            lambda k=key: prof("playoff_scope", k, "playoff_appearance_rate"),
            f"superseded appearance rate, {key}", historical=True)
    add("65.8%", PROFILE,
        lambda: prof("playoff_scope", "starter_24plus",
                     "share_playing_more_in_playoffs"),
        "starters playing more")

    # ── game length and feasibility ───────────────────────────────────────────
    into(SPEC)
    add("731,906", GAME_LEN, lambda: glen("player_games"), "feasibility population")
    add("100.0%", GAME_LEN, lambda: glen("join_coverage"), "feasibility coverage")
    add("1.0000", GAME_LEN, lambda: glen("max_min_over_length"), "max minutes ratio")
    add("1,650", GAME_LEN, lambda: glen("player_games_above_regulation"),
        "player-games above 48 minutes")
    add("63.0", GAME_LEN, lambda: glen("max_minutes"), "observed max minutes")
    # The derivation section is one row per season x season_type with no "all" row, so
    # the headline figures are aggregates over it.
    add("37,986", GAME_LEN, lambda: _derivation("games", "sum"),
        "games in the derivation")
    add("0.617", GAME_LEN, lambda: _derivation("max_abs_residual", "max"),
        "worst rounding residual")
    add("5.93%", GAME_LEN, lambda: _derivation("ot_rate", "weighted"),
        "overtime rate")

    # ── cross-component cancellation ──────────────────────────────────────────
    into(FACTS)
    for feature, gross, net, ratio in [
            ("teammate_assist_supply", "2.106", "−0.254", "8.30"),
            ("role_crowding", None, None, "8.22"),
            ("teammate_spacing", None, None, "5.01"),
            ("team_pace", None, None, "4.74")]:
        add(ratio, CONTEXT_A, lambda f=feature: context(f, "cancellation_ratio"),
            f"{feature} cancellation ratio")
        if gross:
            add(gross, CONTEXT_A, lambda f=feature: context(f, "gross_dk_movement"),
                f"{feature} gross DK movement")
            add(net, CONTEXT_A, lambda f=feature: context(f, "net_dk_movement"),
                f"{feature} net DK movement")
    for outcome, quoted in [("ast_per36", "−0.681"), ("reb_per36", "+0.503"),
                            ("blk_per36", "+0.115")]:
        add(quoted, CONTEXT_A,
            lambda o=outcome: context("teammate_assist_supply", f"effect_{o}"),
            f"teammate_assist_supply effect on {outcome}")
    for outcome, quoted in [("ast_per36", "−0.361"), ("reb_per36", "+0.218"),
                            ("blk_per36", "+0.191")]:
        add(quoted, CONTEXT_A,
            lambda o=outcome: context("teammate_assist_supply", f"r_{o}"),
            f"superseded {outcome} figure (a correlation)", historical=True)
    add("1.803", OPPONENT_A, lambda: opp("dk_pts", "gross_dk_movement"),
        "opponent gross DK movement")
    add("0.911", OPPONENT_A, lambda: opp("dk_pts", "net_dk_movement"),
        "opponent net DK movement")
    add("1.98", OPPONENT_A, lambda: opp("dk_pts", "cancellation_ratio"),
        "opponent cancellation ratio")
    add("+0.203%", OPPONENT_A,
        lambda: opp("blk_per36", "interaction_above_null"),
        "blk interaction above null")
    add("+0.022%", OPPONENT_A, lambda: opp("dk_pts", "interaction_above_null"),
        "dk_pts interaction above null")
    add("0.368%", OPPONENT_A, lambda: opp("dk_pts", "r2_main_effect"),
        "dk_pts opponent main effect")
    add("0.0066", CONTEXT_A, lambda: context("teammate_usage_load", "delta_r2"),
        "teammate_usage_load delta R2")
    add("0.00001", CONTEXT_A, lambda: context("role_crowding", "delta_r2"),
        "role_crowding delta R2")
    add("−0.005", CONTEXT_A,
        lambda: context("role_crowding", "r_dk_pts_per_game"),
        "role_crowding vs dk_pts")
    add("+0.008", CONTEXT_A, lambda: context("team_pace", "r_dk_pts_per_game"),
        "team_pace vs dk_pts")

    # ── roster coverage ───────────────────────────────────────────────────────
    into(FACTS)
    add("14.7%", ROSTER_A, lambda: roster("undescribed"),
        "roster minutes undescribed")
    add("8.7%", ROSTER_A, lambda: roster("rookie"), "rookie minutes")
    add("5.1%", ROSTER_A, lambda: roster("sub_threshold"), "sub-threshold minutes")
    add("0.9%", ROSTER_A, lambda: roster("returnee"), "returnee minutes")
    add("14.1%", ROSTER_A, lambda: roster("rookie", "share_of_headcount"),
        "rookie headcount share")
    add("29.0%", ROSTER_A,
        lambda: roster("undescribed", "p90", scope="team_season_distribution"),
        "p90 undescribed per team-season")
    add("50.4%", ROSTER_A,
        lambda: roster("undescribed", "max", scope="team_season_distribution"),
        "worst team-season")
    for key, quoted in [("undescribed", "15.79"), ("rookie", "9.16"),
                        ("sub_threshold", "5.52"), ("returnee", "1.11")]:
        add(quoted, ROSTER_A,
            lambda k=key: roster(k, window="whole_season") * 100,
            f"whole-season roster {key}")
    add("14,569", PROFILE,
        lambda: prof_n("window_bracket", "season_edge", "mean_trailing_missed"),
        "inclusive matrix rows")

    # ── persistence ───────────────────────────────────────────────────────────
    into(NOTES)
    # `persistence.csv` keys on the prefixed column name, not the bare stat, because
    # `blk` exists in four families — the same reason the notes warn that
    # `adv_def_rating == def_def_rating`.
    persistence = [("bas_fg3a", "0.908"), ("bas_fga", "0.862"), ("bas_fta", "0.854"),
                   ("bas_fg3_pct", "0.500"), ("bas_fg_pct", "0.435"),
                   ("bas_ft_pct", "0.365"), ("adv_ts_pct", "0.302"),
                   ("adv_efg_pct", "0.281"), ("bas_reb", "0.941"),
                   ("bas_oreb", "0.924"), ("bas_ast", "0.920"),
                   ("bas_dreb", "0.906"), ("bas_blk", "0.902"),
                   ("bas_stl", "0.743"), ("bas_tov", "0.790"),
                   ("bas_plus_minus", "0.477"), ("sco_pct_fga_3pt", "0.886"),
                   ("usg_pct_fg3a", "0.870"), ("usg_pct_reb", "0.854"),
                   ("adv_ast_pct", "0.845"), ("adv_usg_pct", "0.786"),
                   ("adv_def_rating", "0.116"), ("adv_net_rating", "0.180"),
                   ("adv_off_rating", "0.285")]
    # Three of these rows are quoted in the archive's share-vs-conversion fact rather
    # than in the notes' persistence table, so they are claimed from there.
    persist_in_facts = {"bas_fga", "bas_fg_pct", "adv_ts_pct"}
    for feature, quoted in persistence:
        add(quoted, PERSIST, lambda f=feature: persist(f),
            f"persistence {feature}",
            doc=FACTS if feature in persist_in_facts else NOTES)
    for feature, quoted in [("bas_stl", "0.401"), ("bas_tov", "0.456"),
                            ("bas_blk", "0.626"), ("bas_plus_minus", "0.154")]:
        add(quoted, PERSIST,
            lambda f=feature: persist(f, "r_within_unweighted"),
            f"persistence {feature}, unweighted")
    for feature, quoted in [("drv_drives", "0.922"), ("pu_pull_up_fga", "0.918"),
                            ("pass_potential_ast", "0.913"),
                            ("hus_contested_shots_2pt", "0.909")]:
        add(quoted, PERSIST, lambda f=feature: persist(f, tier="B"),
            f"persistence {feature}, tier B")
    add("11,272", PERSIST, lambda: persist("bas_reb", "n_pairs"),
        "persistence pairs")
    add("0.574", PERSIST, lambda: persist("adv_e_pace", "r_pooled"),
        "e_pace pooled")
    add("0.212", PERSIST, lambda: persist("adv_e_pace"), "e_pace within-season")

    # ── availability ──────────────────────────────────────────────────────────
    into(NOTES)
    add("0.317", PROFILE, lambda: prof("persistence", "gp_share", "r_within_weighted"),
        "games-played persistence")
    add("22.7", PROFILE,
        lambda: prof("overdispersion", "rotation_players", "variance_ratio"),
        "GP overdispersion, full window")
    add("19.8", PROFILE,
        lambda: prof("overdispersion", "rotation_players", "variance_ratio",
                     "appearance"), "GP overdispersion, appearance window")
    add("26.7%", PROFILE,
        lambda: prof("overdispersion", "rotation_players", "share_below_60_games"),
        "share below 60 games")
    add("0.392", PROFILE, lambda: prof("multiyear", "gp_share_mean3", "r_unweighted"),
        "3-year average")
    add("0.398", PROFILE, lambda: prof("multiyear", "gp_share_lag1", "r_unweighted"),
        "1-year prior")
    add("0.090", PROFILE,
        lambda: prof("persistence", "longest_spell", "r_within_weighted"),
        "longest spell persistence")
    add("0.236", PROFILE,
        lambda: prof("predictor_r2", "plus_age_and_career",
                     "r2_in_sample_unweighted"), "availability ceiling")
    add("0.116", PROFILE,
        lambda: prof("predictor_r2", "plus_age_and_career",
                     "r2_in_sample_weighted"), "availability ceiling, weighted")
    add("0.159", PROFILE,
        lambda: prof("predictor_r2", "prior_mpg", "r2_in_sample_unweighted"),
        "prior MPG ceiling")
    C += _availability_ladder_claims(NOTES, scope="summary",
                                     historical=LADDER_HISTORICAL_NOTES)
    add("0.374", METRICS, lambda: metric(METRICS, "beta_binomial", "r2_gp_share"),
        "validation R2 of the shipped head")

    # ── the season total ──────────────────────────────────────────────────────
    into(NOTES)
    C += _season_total_claims(NOTES)
    add("210.3", SEASON_TOTAL,
        lambda: (treatment("full_season", "mae_dk_total")
                 - treatment("beta_binomial", "mae_dk_total")),
        "head vs a full season")
    # The playoff-workload ablation's season-total figures were never re-measured — its
    # own artifact is still a test evaluation — so all four are historical rather than
    # derived from the current ladder. Deleting them would erase the only record of the
    # pre-workload run.
    for quoted, label in [("441.3", "season total before playoff workload"),
                          ("508.9", "rotation MAE before playoff workload"),
                          ("+6.2", "workload gain on the season total"),
                          ("−9.5", "workload gain on rotation players")]:
        add(quoted, ABLATION, lambda: float("nan"), label, historical=True)

    # ── the component rate heads (Poisson / sklearn) ──────────────────────────
    into(NOTES)
    # `fg2a` and `fg3a` are RETIRED count heads (shot-attempt basis, 2026-08-03) and their
    # rows are gone from the artifact; `fga` replaces both. The retired rows stay in the
    # doc beside the new one, value-exempt, because the table is the record of what the
    # two-count basis measured.
    poisson = [("fga", "0.9514", "0.9544", "0.9586", "0.9589", "0.9591"),
               ("reb", "0.9505", "0.9181", "0.9514", "0.9497", "0.9512"),
               ("ast", "0.9195", "0.8569", "0.9222", "0.9255", "0.9225"),
               ("blk", "0.8103", "0.6510", "0.7748", "0.8291", "0.7799"),
               ("fta", "0.8765", "0.7993", "0.8922", "0.8817", "0.8932"),
               ("stl", "0.8386", "0.8556", "0.8682", "0.8693", "0.8693"),
               ("tov", "0.8966", "0.9091", "0.9171", "0.9162", "0.9173")]
    for head, floor, linear, log_own, spline, inter in poisson:
        for quoted, variant in [(floor, "carry_forward"), (linear, "linear"),
                                (log_own, "log_own"), (spline, "log_own_spline"),
                                (inter, "log_own_inter")]:
            add(quoted, RATES, lambda h=head, v=variant: rate(h, v),
                f"poisson {head} {variant}")
    # The superseded TEST table, kept whole beside the validation one. Two supersessions
    # are stacked here and they are different: the `fg2a` / `fg3a` rows are the retired
    # two-count BASIS, the rest are the retired SPLIT.
    for quoted in ("0.9464", "0.9455", "0.9513", "0.9517", "0.9522",
                   "0.9424", "0.9278", "0.9441", "0.9436", "0.9442",
                   "0.9194", "0.9089", "0.9245", "0.9248", "0.9260",
                   "0.9197", "0.8601", "0.9229", "0.9262", "0.9236",
                   "0.9036", "0.5197", "0.8791", "0.9088", "0.8784",
                   "0.8407", "0.6375", "0.8204", "0.8605", "0.8228",
                   "0.8673", "0.8449", "0.8689", "0.8692", "0.8720",
                   "0.8194", "0.8170", "0.8369", "0.8397", "0.8338",
                   "0.8845", "0.8828", "0.8915", "0.8913", "0.8916"):
        add(quoted, RATES, lambda: float("nan"),
            f"pre-lock held-out component rate: {quoted}", historical=True)
    add("3.0569", RATES,
        lambda: min(rate("ftm|fta", v, "nll")
                    for v in ("linear", "spline_own", "inter", "pca", "pca_inter")),
        "ftm|fta best fitted NLL")
    add("3.0541", RATES, lambda: rate("ftm|fta", "carry_forward", "nll"),
        "ftm|fta floor NLL")
    add("4.6317", RATES,
        lambda: min(rate("fg3a|fga", v, "nll")
                    for v in ("linear", "spline_own", "inter", "pca", "pca_inter")),
        "fg3a|fga best fitted NLL under sklearn")
    add("4.6191", RATES, lambda: rate("fg3a|fga", "carry_forward", "nll"),
        "fg3a|fga floor NLL under sklearn")
    for quoted in ("3.1021", "3.0822"):
        add(quoted, RATES, lambda: float("nan"),
            f"pre-lock held-out ftm|fta NLL: {quoted}", historical=True)
    # The alpha-sensitivity block is the `sklearn` regularization quirk, so it went to
    # `data-quirks.md`; only the unpenalized reference figure stayed with the notes.
    for alpha, quoted, where in [(1e-8, "0.9181", NOTES), (0.01, "0.9267", QUIRKS),
                                 (1.0, "0.6619", QUIRKS)]:
        add(quoted, RATES,
            lambda a=alpha: _one(table(RATES), "r2", analysis="alpha_sensitivity",
                                 head="reb", variant="linear", alpha=a),
            f"reb alpha sensitivity at {alpha}", doc=where)
    for quoted in ("0.9322", "0.6620"):
        add(quoted, RATES, lambda: float("nan"),
            f"pre-lock held-out reb alpha figure: {quoted}", historical=True,
            doc=QUIRKS)
    add("14", RATES,
        lambda: float((~table(RATES)[
            (table(RATES)["analysis"] == "alpha_sensitivity")
            & (table(RATES)["alpha"] == 10.0)]["beats_floor"]).sum()),
        "fits below the floor at alpha=10")
    add("0.200", RATES, lambda: _alpha_loss("linear"),
        "median alpha=1.0 loss, linear", doc=QUIRKS)
    add("0.305", RATES, lambda: _alpha_loss("log_own"),
        "median alpha=1.0 loss, log_own")
    add("0.497", RATES, lambda: _alpha_loss("log_own", head="blk"),
        "worst alpha=1.0 loss", doc=QUIRKS)

    # ── the Stan heads ────────────────────────────────────────────────────────
    into(NOTES)
    # `fg2a` / `fg3a` are RETIRED count heads (shot-attempt basis, 2026-08-04) and their
    # rows are gone from the artifact; `fga` replaces both. They stay in the doc's table as
    # the record of the retired basis, value-exempt but presence-checked.
    # The live table, on validation. `retired` marks a head the shot-attempt basis removed;
    # the superseded TEST reading of every surviving head follows below, so the two axes a
    # figure can move along — which basis, which split — stay separately legible.
    stan_counts = [("fga", "0.9514", "0.9489", "0.9581", "0.9584", False),
                   ("reb", "0.9505", "0.8889", "0.9513", "0.9511", False),
                   ("fg2a", "0.9194", "0.9018", "0.9241", "0.9241", True),
                   ("ast", "0.9195", "0.6418", "0.9198", "0.9255", False),
                   ("fg3a", "0.9036", "−19.00", "0.3719", "0.9046", True),
                   ("tov", "0.8966", "0.9048", "0.9170", "0.9169", False),
                   ("blk", "0.8103", "−0.2744", "0.6730", "0.8309", False),
                   ("fta", "0.8765", "0.5741", "0.8909", "0.8893", False),
                   ("stl", "0.8386", "0.8492", "0.8674", "0.8695", False)]
    for head, floor, linear, log_own, spline, retired in stan_counts:
        for quoted, variant in [(floor, "carry_forward"), (linear, "linear"),
                                (log_own, "log_own"), (spline, "log_own_spline")]:
            add(quoted, STAN_C_M,
                lambda h=head, v=variant: stan_c(h, v, "val_r2"),
                f"NB {head} {variant}", historical=retired)
    stan_counts_on_test = [("fga", "0.9464", "0.9396", "0.9501", "0.9505"),
                           ("reb", "0.9424", "0.9095", "0.9439", "0.9428"),
                           ("ast", "0.9197", "0.6615", "0.9223", "0.9240"),
                           ("tov", "0.8845", "0.8823", "0.8929", "0.8926"),
                           ("blk", "0.8407", "−1.393", "0.6794", "0.8579"),
                           ("fta", "0.8673", "0.8171", "0.8649", "0.8648"),
                           ("stl", "0.8194", "0.8113", "0.8390", "0.8413")]
    for head, floor, linear, log_own, spline in stan_counts_on_test:
        for quoted, variant in [(floor, "carry_forward"), (linear, "linear"),
                                (log_own, "log_own"), (spline, "log_own_spline")]:
            add(quoted, STAN_C_M,
                lambda h=head, v=variant: stan_c(h, v, "val_r2"),
                f"NB {head} {variant}, on test", historical=True)
    conversions = [("fg3a|fga", "4.6157", "4.6191"),
                   ("fg2m|fg2a", "3.8046", "3.8442"),
                   ("fg3m|fg3a", "3.1677", "3.1851"),
                   ("ftm|fta", "3.0804", "3.0541")]
    for head, fitted, floor in conversions:
        add(fitted, STAN_C_M,
            lambda h=head: stan_c(h, "logit_own_spline", "val_nll"),
            f"NB {head} fitted NLL")
        add(floor, STAN_C_M,
            lambda h=head: stan_c(h, "carry_forward", "val_nll"),
            f"NB {head} floor NLL")
    for head, fitted, floor in [("fg3a|fga", "4.6137", "4.6528"),
                                ("fg2m|fg2a", "3.7249", "3.7770"),
                                ("fg3m|fg3a", "3.2407", "3.2614"),
                                ("ftm|fta", "3.1313", "3.0822")]:
        add(fitted, STAN_C_M,
            lambda h=head: stan_c(h, "logit_own_spline", "val_nll"),
            f"NB {head} fitted NLL, on test", historical=True)
        add(floor, STAN_C_M,
            lambda h=head: stan_c(h, "carry_forward", "val_nll"),
            f"NB {head} floor NLL, on test", historical=True)
    add("+0.0034", STAN_C_M,
        lambda: stan_c("fg3a|fga", "carry_forward", "val_nll")
        - stan_c("fg3a|fga", "logit_own_spline", "val_nll"),
        "fg3a|fga NLL gain")
    add("+0.0396", STAN_C_M,
        lambda: stan_c("fg2m|fg2a", "carry_forward", "val_nll")
        - stan_c("fg2m|fg2a", "logit_own_spline", "val_nll"),
        "fg2m|fg2a NLL gain")
    add("+0.0173", STAN_C_M,
        lambda: stan_c("fg3m|fg3a", "carry_forward", "val_nll")
        - stan_c("fg3m|fg3a", "logit_own_spline", "val_nll"),
        "fg3m|fg3a NLL gain")
    add("−0.0263", STAN_C_M,
        lambda: stan_c("ftm|fta", "carry_forward", "val_nll")
        - stan_c("ftm|fta", "logit_own_spline", "val_nll"),
        "ftm|fta NLL gain")
    for quoted, label in [("+0.0391", "fg3a|fga"), ("+0.0521", "fg2m|fg2a"),
                          ("+0.0208", "fg3m|fg3a"), ("−0.0491", "ftm|fta")]:
        add(quoted, STAN_C_M,
            lambda h=label: stan_c(h, "carry_forward", "val_nll")
            - stan_c(h, "logit_own_spline", "val_nll"),
            f"{label} NLL gain, on test", historical=True)
    add("137.4", STAN_C_D, lambda: total(STAN_C_D, "wall_clock_s") / 60,
        "component sampler minutes")
    add("305.0", STAN_C_D, lambda: total(STAN_C_D, "wall_clock_s") / 60,
        "component sampler minutes, both splits", historical=True)
    add("1.0076", STAN_C_D, lambda: max_of(STAN_C_D, "max_rhat"),
        "component max R-hat")
    add("1.0118", STAN_C_D, lambda: max_of(STAN_C_D, "max_rhat"),
        "component max R-hat, half-length selection fits", historical=True)
    # The handicapped margins, kept in the doc beside their correction. The validation one
    # remains exactly true of `stan_component_substitution.csv` — a weaker experiment, not a
    # stale value — so it stays value-checked. Its test twin does not: it survived that
    # file's conversion inside the Gate 0 sweep, and the 2026-08-06 re-run of that sweep took
    # the last copy. Demoted rather than frozen, because freezing a run to keep a claim
    # checkable is the tail wagging the dog.
    add("−0.771", STAN_C_S,
        lambda: cell(STAN_C_S, "reparam_minus_canonical", split="val",
                     arm="two_counts"), "substitution gain, val (handicapped)")
    add("−0.793", SHOT_SWEEP, lambda: float("nan"),
        "substitution gain, test (handicapped)", historical=True)
    add("0.792657", SHOT_SWEEP, lambda: float("nan"),
        "the handicapped test margin, unsigned", historical=True)
    # The four joint-NLL cells the doc used to quote are gone from the prose, replaced by
    # Gate 0's. `10.797` was deliberately NOT re-pointed: the string survives elsewhere in
    # this doc as an unrelated availability CRPS, so a presence check on it would pass for
    # the wrong reason — the exact false-negative `check_presence` exists to avoid.
    # Gate 0 is validation-only since the 2026-08-06 re-run. The notes carry the summary
    # and `docs/shot-attempt-basis-plan.md` carries the full retired test block, so the two
    # test figures kept here are the two the summary sentence actually names — the margin the
    # adoption was decided on and the best-of-16 it was checked against. Both presence-only.
    def shot(arm: str) -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        sub = frame[(frame["analysis"] == "joint") & (frame["split"] == "val")
                    & (frame["arm"] == arm) & frame["selected"].astype(bool)]
        return float(sub["mean_nll"].iloc[0])

    def shot_head(name: str, variant: str, column: str = "mean_nll") -> float:
        arm = "two_counts" if name in ("fg2a", "fg3a") else "fga_x_fg3a_share"
        return cell(SHOT_SWEEP, column, analysis="head", split="val", arm=arm,
                    head=name, variant=variant)

    def shot_floor_total(names, variants) -> float:
        return sum(shot_head(n, v, "floor_nll") for n, v in zip(names, variants))

    add("10.505159", SHOT_SWEEP, lambda: shot("two_counts"), "gate 0 arm A joint NLL")
    add("10.004118", SHOT_SWEEP, lambda: shot("fga_x_fg3a_share"),
        "gate 0 arm B joint NLL")
    add("−0.501041", SHOT_SWEEP,
        lambda: shot("fga_x_fg3a_share") - shot("two_counts"), "gate 0 margin")
    add("−0.493549", SHOT_SWEEP, lambda: float("nan"),
        "gate 0 margin, on test — what adoption was decided on", historical=True)
    add("10.476413", SHOT_SWEEP, lambda: float("nan"),
        "gate 0 arm A best-of-16, on test", historical=True)
    # The handicap decomposition, live on validation since the 2026-08-06 re-run. Its test
    # predecessors sit beside it in the doc and are presence-checked only.
    add("10.797078", STAN_C_S,
        lambda: cell(STAN_C_S, "mean_joint_nll", split="val", arm="two_counts"),
        "gate 0 handicapped arm A")
    add("0.291919", SHOT_SWEEP,
        lambda: cell(STAN_C_S, "mean_joint_nll", split="val",
                     arm="two_counts") - shot("two_counts"),
        "gate 0 handicap in nats, on validation")
    add("−0.021832", SHOT_SWEEP,
        lambda: shot("fga_x_fg3a_share") - shot_head("fga", "log_own")
        - shot_head("fg3a|fga", "logit_own"),
        "gate 0 value of sweeping arm B")
    add("−0.771128", STAN_C_S,
        lambda: cell(STAN_C_S, "reparam_minus_canonical", split="val",
                     arm="two_counts"), "gate 0 recorded margin, full precision")
    add("65%", SHOT_SWEEP,
        lambda: (shot("fga_x_fg3a_share") - shot("two_counts"))
        / cell(STAN_C_S, "reparam_minus_canonical", split="val", arm="two_counts"),
        "gate 0 share of the recorded margin surviving the correction")
    # Claimed from both artifacts against one string: they are the same quantity computed by
    # different code, and the whole point is that they cannot drift apart.
    add("10.025950", SHOT_SWEEP,
        lambda: shot_head("fga", "log_own") + shot_head("fg3a|fga", "logit_own"),
        "gate 0 arm B pinned")
    add("10.025950", STAN_C_S,
        lambda: cell(STAN_C_S, "mean_joint_nll", split="val", arm="fga_x_fg3a_share"),
        "…and substitution_arm computes it identically")
    for quoted, label in [("0.305646", "the handicap"),
                          ("−0.487010", "margin before arm B was swept"),
                          ("−0.006539", "what sweeping arm B added")]:
        add(quoted, SHOT_SWEEP, lambda: float("nan"),
            f"gate 0 {label}, on test", historical=True)
    # The headline: the coordinate change beats the fitting. On validation the gap is wider
    # than the test figures it replaces, so the finding survives the move with room to spare.
    ARM_A_F = (("fg2a", "fg3a"), ("log_own", "log_own_spline"))
    ARM_B_F = (("fga", "fg3a|fga"), ("log_own", "logit_own"))
    add("11.174424", SHOT_SWEEP, lambda: shot_floor_total(*ARM_A_F),
        "gate 0 arm A no-fit floor total")
    add("10.064318", SHOT_SWEEP, lambda: shot_floor_total(*ARM_B_F),
        "gate 0 arm B no-fit floor total")
    add("−1.110105", SHOT_SWEEP,
        lambda: shot_floor_total(*ARM_B_F) - shot_floor_total(*ARM_A_F),
        "gate 0 floor-to-floor gain")
    add("−0.440841", SHOT_SWEEP,
        lambda: shot_floor_total(*ARM_B_F) - shot("two_counts"),
        "gate 0 arm B floor vs arm A fitted")
    add("−0.060200", SHOT_SWEEP,
        lambda: shot("fga_x_fg3a_share") - shot_floor_total(*ARM_B_F),
        "gate 0 what arm B's own fitting adds")
    # The share head's floor failure on validation — the reason it needs its spline. This is
    # the pair the split move protected: on test `logit_own` cleared this floor.
    for quoted, name, variant, column in [
            ("4.636018", "fg3a|fga", "logit_own", "mean_nll"),
            ("4.619109", "fg3a|fga", "logit_own", "floor_nll"),
            ("4.615622", "fg3a|fga", "logit_own_spline", "mean_nll"),
            ("5.388496", "fga", "log_own_spline", "mean_nll"),
            ("5.389932", "fga", "log_own", "mean_nll")]:
        add(quoted, SHOT_SWEEP,
            lambda n=name, v=variant, c=column: shot_head(n, v, c),
            f"gate 0 {name}@{variant} {column}")
    add("1.0047", SHOT_D, lambda: max_of(SHOT_D, "max_rhat"), "gate 0 max R-hat")
    add("26.8", SHOT_D, lambda: total(SHOT_D, "wall_clock_s") / 60,
        "gate 0 sampler minutes")

    # the availability port and the board decomposition
    for model, crps, rho in [("beta_binomial", "9.8444", "0.2627"),
                             ("mixture_mle", "9.8237", "0.2245"),
                             ("stan_plug_in", "9.8195", "0.2261"),
                             ("stan_posterior", "9.8239", "0.2261")]:
        add(crps, STAN_AV_M, lambda m=model: metric(STAN_AV_M, m, "crps_games"),
            f"stan {model} CRPS")
        add(rho, STAN_AV_M,
            lambda m=model: metric(STAN_AV_M, m, "dispersion_rho"),
            f"stan {model} rho")
    add("1.0073", STAN_AV_D, lambda: cell(STAN_AV_D, "max_rhat"), "stan R-hat")
    add("1,399", STAN_AV_D, lambda: cell(STAN_AV_D, "min_ess_bulk"), "stan min ESS")
    add("366", STAN_AV_D, lambda: cell(STAN_AV_D, "wall_clock_s"),
        "stan wall clock")
    board_rows = [(12, "69.060", "6.499", "0.4%"), (15, "76.810", "7.669", "0.5%"),
                  (30, "108.652", "13.370", "0.8%"), (150, "242.604", "58.432", "2.9%"),
                  (883, "589.169", "332.588", "14.8%")]
    for n, indep, shared, infl in board_rows:
        add(indep, STAN_AV_B,
            lambda k=n: cell(STAN_AV_B, "independent_sd", n_players=k),
            f"board independent sd, {n}")
        add(shared, STAN_AV_B,
            lambda k=n: cell(STAN_AV_B, "shared_beta_sd", n_players=k),
            f"board shared-beta sd, {n}")
        add(infl, STAN_AV_B,
            lambda k=n: cell(STAN_AV_B, "inflation", n_players=k) - 1.0,
            f"board inflation, {n}")
    for quoted in ("10.7952", "10.7947", "10.7953", "0.2757", "0.2759", "1.0025",
                   "2,402", "254", "0.0127", "0.095", "69.4", "4.0", "604.0",
                   "219.1", "6.4%", "911"):
        add(quoted, STAN_AV_M, lambda: float("nan"),
            f"pre-lock held-out availability port: {quoted}", historical=True)
    # The full-window, shared-rho head this replaced on 2026-08-11. Its board row is the
    # side of the window trade that got worse, so it is kept rather than dropped.
    for quoted in ("10.0063", "10.0071", "10.0057", "0.2806", "0.2808", "0.00335",
                   "0.085", "1.0019", "2,314", "196", "69.9", "4.2", "597.8",
                   "222.8", "6.7%"):
        add(quoted, STAN_AV_M, lambda: float("nan"),
            f"pre-window full-sample availability head: {quoted}", historical=True)
    # The single-component role-graded head the mixture replaced on 2026-08-12. Same rule:
    # the board term is the side of THIS trade that got wider, so both rows are kept.
    for quoted in ("9.8136", "9.8155", "0.2595", "0.09352", "1.658", "1.0050", "2,382",
                   "94", "68.0", "6.1", "75.8", "7.2", "107.2", "12.3", "239.4",
                   "52.5", "581.4", "297.2", "12.3%", "2.4%"):
        add(quoted, STAN_AV_M, lambda: float("nan"),
            f"pre-mixture single-component availability head: {quoted}", historical=True)

    # the minutes head — validation only since the 2026-08-06 re-run under the lock
    minutes = [("carry_forward", "161.45", "0.8536", "213.13", "+23.91"),
               ("linear", "144.71", "0.8826", "199.54", "−3.26"),
               ("logit_own", "145.45", "0.8819", "200.90", "−2.97"),
               ("logit_own_quadratic", "144.54", "0.8827", "200.84", "−13.82"),
               ("logit_own_spline", "143.93", "0.8835", "199.60", "−14.00")]
    for name, crps, r2, mae, bias in minutes:
        add(crps, STAN_MIN_M, lambda n=name: cell(STAN_MIN_M, "val_crps", variant=n),
            f"minutes {name} val CRPS")
        add(r2, STAN_MIN_M, lambda n=name: cell(STAN_MIN_M, "val_r2", variant=n),
            f"minutes {name} val R2")
        add(mae, STAN_MIN_M, lambda n=name: cell(STAN_MIN_M, "val_mae", variant=n),
            f"minutes {name} val MAE")
        add(bias, STAN_MIN_M, lambda n=name: cell(STAN_MIN_M, "val_bias", variant=n),
            f"minutes {name} val bias")
    # The fitted-minus-floor bias gap, which is what reproduced across the split move while
    # the level did not. Derived from two cells rather than quoted flat, so it cannot drift
    # away from the rows it is a difference of.
    for name, quoted in [("linear", "−27.2"), ("logit_own_spline", "−37.9")]:
        add(quoted, STAN_MIN_M,
            lambda n=name: (cell(STAN_MIN_M, "val_bias", variant=n)
                            - cell(STAN_MIN_M, "val_bias", variant="carry_forward")),
            f"minutes {name} bias below the floor")
    add("0.05025", STAN_MIN_D,
        lambda: cell(STAN_MIN_D, "rho", metric="season_level_rho"),
        "season-level rho")
    add("0.0776", STAN_MIN_D,
        lambda: cell(STAN_MIN_D, "rho", metric="game_level_rho"), "game-level rho")
    add("4.65", STAN_MIN_D,
        lambda: cell(STAN_MIN_D, "implied_overdispersion", metric="game_level_rho"),
        "game-level overdispersion")
    add("713,947", STAN_MIN_D,
        lambda: cell(STAN_MIN_D, "n_player_games", metric="game_level_rho"),
        "game-level population")
    add("1,503", STAN_MIN_G, lambda: total(STAN_MIN_G, "wall_clock_s"),
        "minutes head wall clock")
    add("1.0054", STAN_MIN_G, lambda: max_of(STAN_MIN_G, "max_rhat"),
        "minutes head max R-hat")
    add("+0.0299", STAN_MIN_M,
        lambda: (cell(STAN_MIN_M, "val_r2", variant="logit_own_spline")
                 - cell(STAN_MIN_M, "val_r2", variant="carry_forward")),
        "minutes R2 gain over floor")
    add("−17.5", STAN_MIN_M,
        lambda: (cell(STAN_MIN_M, "val_crps", variant="logit_own_spline")
                 - cell(STAN_MIN_M, "val_crps", variant="carry_forward")),
        "minutes CRPS gain over floor")
    add("955", STAN_MIN_G,
        lambda: cell(STAN_MIN_G, "wall_clock_s", label="logit_own_spline/val"),
        "spline wall clock")
    add("175", STAN_MIN_G,
        lambda: cell(STAN_MIN_G, "wall_clock_s", label="linear/val"),
        "linear wall clock")
    C += _minutes_pre_lock_claims(NOTES)

    # the composition, at full window (Gate E, 2026-08-04). `independent_comparator` never
    # trains on the composition window and scores identical rows, so it is invariant and is
    # the control on the refit. Pilot figures the doc keeps beside the new ones are marked
    # historical: presence-checked, value-exempt.
    for variant, val, pit in [
            ("carry_forward", "4.6776", "0.0496"),
            ("binomial", "4.9388", "0.1919"),
            ("betabinom", "4.5417", "0.0496"),
            ("betabinom_ot", "4.5431", "0.0494"),
            ("betabinom_ot_graded", "4.4945", "0.0428"),
            ("independent_comparator", "4.7842", "0.0483")]:
        add(val, COMP_M, lambda v=variant: cell(COMP_M, "val_crps", variant=v),
            f"composition {variant} val CRPS")
        add(pit, COMP_M, lambda v=variant: cell(COMP_M, "val_pit_ks", variant=v),
            f"composition {variant} PIT KS")
    add("−0.2898", COMP_M,
        lambda: (cell(COMP_M, "val_crps", variant="betabinom_ot_graded")
                 - cell(COMP_M, "val_crps", variant="independent_comparator")),
        "composition gain vs the incumbent")
    add("−6.06%", COMP_M,
        lambda: (cell(COMP_M, "val_crps", variant="betabinom_ot_graded")
                 / cell(COMP_M, "val_crps", variant="independent_comparator") - 1.0),
        "composition gain, percentage")
    add("−0.5521", COMP_M,
        lambda: cell(COMP_M, "val_bias", variant="independent_comparator"),
        "comparator bias")
    add("8.3", COMP_M, lambda: cell(COMP_M, "probe_hours", variant="binomial"),
        "composition Gate A extrapolation")
    add("9.78", COMP_D, lambda: _comp_sweep_seconds() / 3600,
        "composition actual sweep hours")
    add("1.17", COMP_D,
        lambda: (_comp_sweep_seconds() / 3600
                 / cell(COMP_M, "probe_hours", variant="binomial")),
        "how far Gate A under-predicted")
    add("14.83", COMP_D,
        lambda: cell(COMP_D, "wall_clock_s", label="betabinom/val") / 631158 * 1000,
        "full-window ms per row")
    add("33.89", COMP_P,
        lambda: cell(COMP_P, "simulated", variant="betabinom_ot_graded",
                     analysis="team_sum_abs_error",
                     group="independent"), "comparator team-sum error")
    for quoted, column, group in [("0.6013", "observed", "regulation/composition"),
                                  ("0.6423", "observed", "overtime/composition"),
                                  ("0.5912", "simulated", "regulation/composition"),
                                  ("0.6415", "simulated", "overtime/composition")]:
        add(quoted, COMP_P,
            lambda c=column, g=group: cell(COMP_P, c, variant="betabinom_ot_graded",
                                           analysis="starter_share", group=g),
            f"starter share {column} {group}")
    for arm, tier, quoted in [("betabinom_ot", "q1_fringe", "1.2224"),
                              ("betabinom_ot", "q2", "0.8430"),
                              ("betabinom_ot", "q3", "0.6589"),
                              ("betabinom_ot", "q4_star", "0.6285"),
                              ("betabinom_ot_graded", "q1_fringe", "1.0465"),
                              ("betabinom_ot_graded", "q2", "0.8132"),
                              ("betabinom_ot_graded", "q3", "0.7071"),
                              ("betabinom_ot_graded", "q4_star", "0.8136")]:
        add(quoted, COMP_P,
            lambda a=arm, t=tier: cell(COMP_P, "ratio", variant=a,
                                       analysis="variance_ratio", group=t),
            f"composition variance ratio {arm} {tier}")
    for arm, quoted in [("betabinom_ot", "0.2730"),
                        ("betabinom_ot_graded", "0.1782")]:
        add(quoted, COMP_P,
            lambda a=arm: mean_abs_dev(COMP_P, "ratio", 1.0, variant=a,
                                       analysis="variance_ratio"),
            f"composition mean |ratio-1| {arm}")
    add("35%", COMP_P,
        lambda: (1.0 - mean_abs_dev(COMP_P, "ratio", 1.0,
                                    variant="betabinom_ot_graded",
                                    analysis="variance_ratio")
                 / mean_abs_dev(COMP_P, "ratio", 1.0, variant="betabinom_ot",
                                analysis="variance_ratio")),
        "composition calibration cut")
    for b, quoted in [(1, "0.1768"), (2, "0.1300"), (3, "0.1115"), (4, "0.0855")]:
        add(quoted, COMP_RHO,
            lambda i=b: cell(COMP_RHO, "rho", variant="betabinom_ot_graded", bin=i),
            f"composition graded rho bin {b}")
    add("0.1211", COMP_RHO,
        lambda: cell(COMP_RHO, "rho", variant="betabinom_ot", bin=1),
        "composition shared rho")
    add("2.07", COMP_RHO,
        lambda: (cell(COMP_RHO, "rho", variant="betabinom_ot_graded", bin=1)
                 / cell(COMP_RHO, "rho", variant="betabinom_ot_graded", bin=4)),
        "composition graded rho spread")
    # The retired test / two-pass figures the notes keep beside their replacements. The
    # full retired block lives in `docs/minutes-composition-plan.md`; this is the subset
    # the summary quotes, claimed from here so it cannot go stale in one doc only.
    for quoted, label in [("4.5592", "retired test CRPS, selected"),
                          ("4.9140", "retired test CRPS, comparator"),
                          ("0.1948", "retired test PIT KS, binomial"),
                          ("−0.3548", "retired test gain vs the incumbent"),
                          ("−7.2%", "retired test gain, percentage"),
                          ("−1.2856", "retired test comparator bias"),
                          ("36.87", "retired test comparator team-sum error"),
                          ("20.9", "retired two-pass sweep hours"),
                          ("1.63", "retired two-pass Gate A under-prediction"),
                          ("12.8", "retired two-pass Gate A extrapolation"),
                          ("−0.406", "pilot gain vs the incumbent"),
                          ("0.1480", "pilot graded rho, fringe"),
                          ("0.1125", "pilot graded rho, q2"),
                          ("0.0874", "pilot graded rho, q3"),
                          ("0.0613", "pilot graded rho, star"),
                          ("0.0970", "pilot shared rho"),
                          ("2.41", "pilot rho spread"),
                          ("0.1055", "pilot mean |ratio-1|"),
                          ("0.988", "pilot star-tier ratio"),
                          ("59%", "pilot calibration cut")]:
        add(quoted, COMP_M, lambda: float("nan"), label, historical=True)
    add("0.0608", GL_M, lambda: cell(GL_M, "p_any_ot", variant="floor"),
        "OT tail p_any")
    add("0.1408", GL_M, lambda: cell(GL_M, "p_more_ot", variant="floor"),
        "OT tail p_more")
    add("128.4", GL_PPC,
        lambda: cell(GL_PPC, "predicted", variant="floor", **{"class": "1OT"}),
        "OT tail predicted 1OT")

    # ── serial and residual correlation ───────────────────────────────────────
    into(NOTES)
    for comp, excess, block in [("min", "+0.294", "2.43"),
                                ("fg3a|fga", "+0.101", "1.57"),
                                ("fga", "+0.061", "1.38"), ("ftm|fta", "+0.012",
                                                            "1.10"),
                                ("fg2m|fg2a", "+0.002", "1.03"),
                                ("fg3m|fg3a", "−0.002", "1.01")]:
        add(excess, SERIAL, lambda c=comp: serial(c, "lag1_excess"),
            f"serial {comp} excess")
        add(block, SERIAL, lambda c=comp: serial(c, "block_inflation"),
            f"serial {comp} block inflation")
    add("+0.212", SERIAL, lambda: serial("min_detrended", "lag1_excess"),
        "detrended minutes excess")
    add("1.71", SERIAL, lambda: serial("min_detrended", "block_inflation"),
        "detrended block inflation")
    add("592,796", SERIAL, lambda: max_of(SERIAL, "n_pairs") + 9052,
        "serial-correlation player-games")
    for lag, quoted in [("lag1", "0.278"), ("lag2", "0.212"), ("lag3", "0.170"),
                        ("lag5", "0.113")]:
        add(quoted, SERIAL, lambda l=lag: serial("min", l), f"minutes {lag}")
    add("0.196", SERIAL, lambda: serial("min_detrended", "lag1"),
        "detrended minutes lag-1")
    add("+0.0070", RESID, lambda: resid("mean"), "residual mean, all 11")
    add("+0.0225", RESID, lambda: resid("mean", kind="count"),
        "residual mean, 7 counts")
    add("+0.1329", RESID, lambda: resid("max"), "residual max")
    # The retired pair now lives under its own basis label — it is a contrast against the
    # two-count basis, not a cell of the shipped matrix.
    add("−0.1248", RESID,
        lambda: _windowed(RESID, "r", component_a="fg3a", component_b="fg2a",
                          basis="legacy_two_count_basis"),
        "3PA/2PA substitution, legacy basis")
    add("−0.0836", RESID,
        lambda: _windowed(RESID, "r", component_a="fga", component_b="fg3a|fga",
                          basis="minutes_conditioned"),
        "volume/mix coupling, shipped basis")
    add("+0.7853", RESID, resid_min_eig, "copula min eigenvalue")
    add("+0.090", RESID, lambda: resid("mean", basis="raw"), "raw residual mean")
    add("+0.470", RESID, lambda: resid("max", basis="raw"), "raw residual max")
    for quoted, label in [("+0.013", "mean"), ("0.157", "max"), ("−0.110", "3PA/2PA")]:
        add(quoted, RESID, lambda: resid("mean", kind="count"),
            f"superseded residual {label} (2021-22 onward)", historical=True)

    # ── report calibration ────────────────────────────────────────────────────
    into(NOTES)
    add("12,338", REPORT_CAL, lambda: cal("all", "scored_rows", "coverage"),
        "report rows scored")
    add("12,007", REPORT_CAL, lambda: cal("all", "scored_rows", "coverage"),
        "superseded scored-row count", historical=True)
    add("96.8%", REPORT_CAL, lambda: cal("all", "match_rate", "coverage"),
        "superseded join rate", historical=True)
    for quoted, label in [("0.027", "Doubtful"), ("0.500", "Questionable"),
                          ("0.852", "Available")]:
        add(quoted, REPORT_CAL,
            lambda k=label: cal(f"{k}|all", "p_played"),
            f"superseded {label} p_played", historical=True)
    add("99.5%", REPORT_CAL, lambda: cal("all", "match_rate", "coverage"),
        "report join rate")
    add("12,406", REPORT_CAL, lambda: cal("all", "report_rows", "coverage"),
        "report rows total")
    add("142", REPORT_CAL, lambda: cal("all", "game_dates", "coverage"),
        "report game dates")
    add("532", REPORT_CAL, lambda: cal("all", "players", "coverage"),
        "report players")
    for name, quoted in [("Out", "0.002"), ("Doubtful", "0.030"),
                         ("Questionable", "0.498"), ("Probable", "0.914"),
                         ("Available", "0.855")]:
        add(quoted, REPORT_CAL, lambda k=f"{name}|all": cal(k, "p_played"),
            f"{name} p_played")
    add("98.5%", REPORT_CAL, lambda: cal("Out", "p_unchanged", "revision"),
        "Out unchanged next day")
    add("38.9%", REPORT_CAL,
        lambda: cal("Questionable", "p_unchanged", "revision"),
        "Questionable unchanged next day")
    add("0.475", REPORT_CAL,
        lambda: cal("Questionable|lead_1", "p_played"),
        "stale Questionable p_play")
    add("0.512", REPORT_CAL,
        lambda: cal("Questionable|lead_0", "p_played"),
        "fresh Questionable p_play")
    add("89.8%", REPORT_CAL, lambda: cal("Out|all", "p_inactive"), "Out inactive")
    add("9.4%", REPORT_CAL, lambda: cal("Out|all", "p_dnp"), "Out dnp")
    add("23.5", REPORT_CAL,
        lambda: cal("Questionable|all", "mean_min_given_played"),
        "Questionable minutes given played")
    add("24.7", REPORT_CAL,
        lambda: cal("Probable|all", "mean_min_given_played"),
        "Probable minutes given played")
    # The G-League-excluded scale, which is what makes the Probable/Available inversion
    # reason mix rather than label noise.
    for name, quoted in [("Out", "0.001"), ("Doubtful", "0.022"),
                         ("Questionable", "0.553"), ("Probable", "0.919"),
                         ("Available", "0.903")]:
        add(quoted, REPORT_CAL,
            lambda k=name: cal(f"{k}|ex_gleague", "p_played", "by_reason"),
            f"{name} p_played, excluding G-League")
    add("1.6", REPORT_CAL,
        lambda: (cal("Probable|ex_gleague", "p_played", "by_reason")
                 - cal("Available|ex_gleague", "p_played", "by_reason")) * 100,
        "the residual Probable/Available gap in pp")
    add("12.5%", REPORT_CAL,
        lambda: cal("Out|Not With Team", "p_absent", "by_reason"),
        "Out|Not With Team absent")
    add("14.3%", REPORT_CAL,
        lambda: cal("Out|Trade Pending", "p_absent", "by_reason"),
        "Out|Trade Pending absent")

    # ── bonus calibration ─────────────────────────────────────────────────────
    into(FACTS)
    add("22.7%", BONUS,
        lambda: -bonus("player_season", 0.0, "relative_bias"),
        "independent sampling reads low")
    add("+0.0009", BONUS, lambda: bonus("player_season", 0.10, "bias"),
        "shipped overdispersion bias")
    add("0.0968", BONUS,
        lambda: _windowed(BONUS, "overdispersion", analysis="fitted",
                     unit="player_season"), "fitted season-unit optimum")
    add("0.0248", BONUS,
        lambda: _windowed(BONUS, "overdispersion", analysis="fitted",
                     unit="player_game"), "fitted game-unit optimum")
    add("11,938", BONUS, lambda: bonus("player_season", 0.10, "n"),
        "bonus calibration population")
    add("0.0951", BONUS,
        lambda: bonus("player_season", 0.0, "expected_mean_bonus"),
        "independent expected bonus")
    add("0.1231", BONUS,
        lambda: bonus("player_season", 0.0, "realized_mean_bonus"),
        "realized bonus")

    # ── aging ─────────────────────────────────────────────────────────────────
    into(NOTES)
    for metric_name, age, quoted in [("dk_linear_per36", 26, "1.041"),
                                     ("dk_linear_per36", 34, "0.895"),
                                     ("minutes_per_game", 27, "1.111"),
                                     ("minutes_per_game", 34, "0.792"),
                                     ("minutes_per_game", 37, "0.515"),
                                     ("games_played", 28, "1.054"),
                                     ("games_played", 34, "0.952")]:
        add(quoted, AGING, lambda m=metric_name, a=age: age_ratio(m, a),
            f"aging {metric_name} at {age}")

    # ── target profile ────────────────────────────────────────────────────────
    into(NOTES)
    def decomp(bucket: str, column: str) -> float:
        return _one(table(TARGET), column, analysis="season_total_decomposition",
                    bucket_kind="log_factor", bucket=bucket)

    add("84.54%", TARGET, lambda: decomp("log_rate", "r2"),
        "log(total) on log(rate)")
    add("73.43%", TARGET, lambda: decomp("log_games", "r2"),
        "log(total) on log(games)")
    add("1.0000", TARGET, lambda: decomp("log_rate_and_games", "r2"),
        "the identity, as a free check")
    add("+0.585", TARGET, lambda: decomp("log_rate_vs_log_games", "r"),
        "the two factors correlate")
    add("12,996", TARGET, lambda: decomp("log_rate", "n"),
        "decomposition population")

    # ── the absence-reason decomposition ──────────────────────────────────────
    into(NOTES)
    add("7,673", PROFILE, lambda: prof("decomposition", "coverage", "usable_pairs"),
        "usable season pairs")
    add("0.696", PROFILE,
        lambda: prof("decomposition", "coverage", "mean_status_coverage"),
        "mean status coverage")
    for quoted, name in [("0.2353", "r2_prior_gp_share"),
                         ("0.2365", "r2_plus_missed_games"),
                         ("0.2648", "r2_plus_reason_split"),
                         ("0.2363", "r2_reason_split_shuffled_null"),
                         ("0.00041", "r2_reason_split_null_sd"),
                         ("0.0285", "r2_reason_split_above_null")]:
        add(quoted, PROFILE, lambda n=name: prof("decomposition", "incremental", n),
            f"reason split {name}")
    for key, corr, persistence in [("missed_scratch", "−0.353", "0.469"),
                                   ("missed_inactive", "−0.233", None),
                                   ("missed_not_rostered", "−0.164", None),
                                   ("missed_injury", "+0.034", None)]:
        add(corr, PROFILE,
            lambda k=key: prof("decomposition", k, "r_vs_next_gp_share"),
            f"{key} vs next gp_share")
        if persistence:
            add(persistence, PROFILE,
                lambda k=key: prof("decomposition", k, "r_persistence_of_column"),
                f"{key} persistence")
    add("0.0012", PROFILE,
        lambda: (prof("decomposition", "incremental", "r2_plus_missed_games")
                 - prof("decomposition", "incremental", "r2_prior_gp_share")),
        "missed_games incremental")

    # ── the serial availability structure ─────────────────────────────────────
    into(NOTES)
    ap = "appearance"
    add("942,597", PROFILE,
        lambda: prof_n("serial_structure", "transition", "p_play_given_played", ap),
        "transitions")
    add("0.905", PROFILE,
        lambda: prof("serial_structure", "transition", "p_play_given_played", ap),
        "P(play | played)")
    add("0.308", PROFILE,
        lambda: prof("serial_structure", "transition", "q_play_given_missed", ap),
        "P(play | missed)")
    add("0.597", PROFILE,
        lambda: prof("serial_structure", "transition", "lag1_autocorrelation", ap),
        "lag-1 autocorrelation")
    add("3.96", PROFILE,
        lambda: prof("serial_structure", "markov",
                     "clustering_variance_inflation", ap),
        "Markov variance inflation")
    add("0.483", PROFILE,
        lambda: prof("serial_structure", "spell_shape", "share_single_observed", ap),
        "spells of 1, observed")
    add("0.0635", PROFILE,
        lambda: prof("serial_structure", "spell_shape", "share_ge10_observed", ap),
        "spells of 10+, observed")
    add("0.0365", PROFILE,
        lambda: prof("serial_structure", "spell_shape", "share_ge10_geometric", ap),
        "spells of 10+, geometric")
    add("3.25", PROFILE,
        lambda: prof("serial_structure", "spell_shape", "mean_spell_observed", ap),
        "mean spell")

    # The matched frame, claimed from the notes as well as from the plan doc against the
    # one artifact — this exact block has already gone stale in one doc while current in
    # another, twice, which is why both sides are pinned.
    add("0.9443", PROFILE,
        lambda: prof("serial_structure", "transition_rotation", "p_play_given_played"),
        "P(play | played), matched population")
    add("0.1333", PROFILE,
        lambda: prof("serial_structure", "transition_rotation", "q_play_given_missed"),
        "P(play | missed), matched population")
    add("9.58", PROFILE,
        lambda: prof("serial_structure", "markov_rotation",
                     "clustering_variance_inflation"),
        "clustering C, matched population")
    add("0.181", PROFILE,
        lambda: ((prof("overdispersion", "rotation_players", "variance_ratio")
                  - prof("serial_structure", "markov_rotation",
                         "clustering_variance_inflation"))
                 / (82.0 - prof("serial_structure", "markov_rotation",
                                "clustering_variance_inflation"))),
        "residual frailty rho")
    add("29.55", PROFILE,
        lambda: (prof("serial_structure", "markov_rotation",
                      "clustering_variance_inflation")
                 + 0.2757 * (82.0 - prof("serial_structure", "markov_rotation",
                                         "clustering_variance_inflation"))),
        "stacked inflation")
    add("30.2%", PROFILE,
        lambda: ((prof("serial_structure", "markov_rotation",
                       "clustering_variance_inflation")
                  + 0.2757 * (82.0 - prof("serial_structure", "markov_rotation",
                                          "clustering_variance_inflation")))
                 / prof("overdispersion", "rotation_players", "variance_ratio") - 1.0),
        "stacking overshoot")

    # ── the nonlinearity ablations ────────────────────────────────────────────
    into(NOTES)
    # Validation-only since 2026-08-08. Every val figure reproduced to five decimals on the
    # move, because `selection_split` returns exactly the frames this ablation's own inner
    # split used to carve — a determinism check, not a replication.
    for variant, val, r2 in [("linear", "10.006", "0.374"),
                             ("quadratic", "10.037", "0.370"),
                             ("spline_k4", "10.041", "0.367"),
                             ("spline_k5", "10.054", "0.366")]:
        add(val, NONLIN,
            lambda v=variant: cell(NONLIN, "val_crps_games", variant=v),
            f"nonlinearity {variant} val CRPS")
        add(r2, NONLIN,
            lambda v=variant: cell(NONLIN, "val_r2_gp_share", variant=v),
            f"nonlinearity {variant} val R2")
    for quoted in ("10.795", "10.749", "10.761", "10.751"):
        add(quoted, NONLIN, lambda: float("nan"),
            f"pre-lock held-out nonlinearity CRPS: {quoted}", historical=True)
    for name, val in [("linear", "0.6768"), ("quadratic", "0.6914"),
                      ("spline_k4", "0.6930")]:
        add(val, MIN_NONLIN,
            lambda n=name: cell(MIN_NONLIN, "val_r2", scope="variant", name=n),
            f"MPG probe {name} val R2")
    for column, quoted in [("minutes_per_game_lag1", "0.0124"),
                           ("total_minutes_lag1", "0.0008"),
                           ("career_year", "0.0007"),
                           ("age", "−0.0008"),
                           ("playoff_minutes_share_lag1", "−0.0016")]:
        add(quoted, MIN_NONLIN,
            lambda c=column: cell(MIN_NONLIN, "val_vs_linear", scope="column",
                                  name=c), f"{column} spline, val")
    # `0.0005` is deliberately absent: it was `career_minutes_lag1`'s *validation* delta,
    # which is still live in the artifact — the doc simply stopped quoting that row. A
    # historical claim protects a superseded figure from deletion and would be the wrong
    # instrument for a current one.
    for quoted in ("0.6670", "0.6746", "0.6736", "0.0077", "0.0011", "−0.0007",
                   "−0.0006"):
        add(quoted, MIN_NONLIN, lambda: float("nan"),
            f"pre-lock held-out MPG probe: {quoted}", historical=True)

    # ── the workload ablation ─────────────────────────────────────────────────
    C += _workload_ablation_claims(NOTES)

    # ── the season total, derived ─────────────────────────────────────────────
    into(NOTES)
    add("287.3", SEASON_TOTAL, lambda: treatment("beta_binomial", "crps_dk_total"),
        "head season-total CRPS")
    add("329.1", SEASON_TOTAL, lambda: treatment("league_age", "crps_dk_total"),
        "baseline season-total CRPS")
    add("186.1", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 - treatment("oracle_gp", "mae_dk_total")),
        "availability's share of the remaining error")
    add("138.5", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 - treatment("oracle_rate", "mae_dk_total")),
        "the rate's share of the remaining error")
    add("−34.4%", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 / treatment("full_season", "mae_dk_total") - 1.0),
        "head vs a full season, relative")
    add("−17.4", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 - treatment("prior_gp", "mae_dk_total")),
        "head vs carrying prior GP forward")
    add("−22.6%", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total", "rotation")
                 / treatment("full_season", "mae_dk_total", "rotation") - 1.0),
        "rotation-player gain, relative")
    # Gate E, which ran here for the first time on 2026-08-05 and failed.
    add("406.6", SEASON_TOTAL, lambda: cell(SEASON_TOTAL_GATE_E, "mae"),
        "Gate E spell-process MAE")
    add("291.6", SEASON_TOTAL, lambda: cell(SEASON_TOTAL_GATE_E, "crps"),
        "Gate E spell-process CRPS")
    # The superseded TEST ladder, preserved beside the validation one. Presence-only, so
    # the failure this guards is deletion of the reversal rather than drift in it.
    for quoted in ("646.3", "475.0", "476.0", "435.1", "302.7", "221.3", "213.8",
                   "132.5", "651.3", "499.4", "−23.3%", "441.3", "508.9",
                   "435.1053", "435.1352"):
        add(quoted, SEASON_TOTAL, lambda: float("nan"),
            f"pre-lock held-out season total: {quoted}", historical=True)

    # ── target dispersion and the first-k ladder ──────────────────────────────
    into(NOTES)
    def dist(metric_name: str, bucket: str, column: str,
             kind: str = "minutes") -> float:
        return _one(table(TARGET), column, analysis="distribution",
                    metric=metric_name, bucket_kind=kind, bucket=bucket)

    add("35.4%", TARGET, lambda: dist("dk_pts", "0-5", "zero_share"),
        "dk_pts zeros below 5 minutes")
    add("0.0%", TARGET, lambda: dist("dk_pts", "30-48", "zero_share"),
        "dk_pts zeros above 30 minutes")
    add("58.6%", TARGET, lambda: dist("blk", "30-48", "zero_share"),
        "blk zeros in 30-48 minute games")
    add("42.2%", TARGET, lambda: dist("fg3m", "30-48", "zero_share"),
        "fg3m zeros in 30-48 minute games")
    add("33.8%", TARGET, lambda: dist("stl", "30-48", "zero_share"),
        "stl zeros in 30-48 minute games")
    for k, r2, mae in [("5", "0.859", "238"), ("10", "0.905", "193"),
                       ("20", "0.942", "148"), ("41", "0.974", "94")]:
        add(r2, TARGET,
            lambda b=k: _one(table(TARGET), "r2_extrapolated", analysis="season_total",
                             bucket_kind="first_k_games", bucket=b),
            f"first-{k} season-total R2")
        add(mae, TARGET,
            lambda b=k: _one(table(TARGET), "mae", analysis="season_total",
                             bucket_kind="first_k_games", bucket=b),
            f"first-{k} season-total MAE")

    # ── aging by component and the cross-sectional trap ───────────────────────
    into(NOTES)
    for metric_name, quoted in [("reb_per36", "0.977"), ("blk_per36", "1.011"),
                                ("stl_per36", "0.962"), ("pts_per36", "0.844"),
                                ("ast_per36", "0.867"), ("fg3m_per36", "0.872")]:
        add(quoted, AGING, lambda m=metric_name: age_ratio(m, 34),
            f"aging {metric_name} at 34")
    for age, quoted in [(19, "31.1"), (23, "31.9"), (30, "31.6"), (34, "31.4"),
                        (39, "34.3")]:
        add(quoted, AGING,
            lambda a=age: _one(table(AGING), "cross_sectional_mean", tier="A",
                               metric="dk_linear_per36", archetype=-1, age=a),
            f"cross-sectional mean at {age}")

    # ── the opponent cancellation family ──────────────────────────────────────
    into(FACTS)
    # Only the shipped construction is on disk. The other three ratios the archive quotes
    # (2.01x raw per-season sd, 2.11x pooled, 1.84x ridge) were measured once and are
    # listed in `docs/provenance-plan.md` as prose-only.
    for quoted, label in [("1.105", "gross"), ("0.785", "net"), ("1.41", "ratio")]:
        add(quoted, OPPONENT_A, lambda: opp("dk_pts", "cancellation_ratio"),
            f"superseded opponent {label}", historical=True)

    # ── the ADP match audit ───────────────────────────────────────────────────
    into(FACTS)
    def audit(metric_name: str, rule: str = "cascade") -> float:
        return _one(table(ADP_AUDIT), "value", section="summary", rule=rule,
                    metric=metric_name)

    # The two headline rates are quoted in the notes' name-matching block; the rule
    # comparison below stayed with the archive.
    add("3,591", ADP_AUDIT, lambda: audit("rows_audited"), "rows audited", doc=NOTES)
    add("0.50%", ADP_AUDIT, lambda: audit("unmatched_rate_cascade"),
        "cascade unmatched rate", doc=NOTES)
    add("0.00%", ADP_AUDIT,
        lambda: audit("unmatched_rate_surname_initial", "surname_initial"),
        "rejected rule's unmatched rate")
    add("57", ADP_AUDIT, lambda: rows(ADP_AUDIT), "match audit rows")

    # The season-term verdict, claimed against the SAME artifact `docs/predictions-plan.md`
    # claims it from. Both, deliberately: this file's copy of a block going stale while the
    # plan doc's stayed current is the exact failure that has already happened twice here.
    C += _season_term_summary_claims(FACTS)

    # Same arrangement for the carry-forward bias, restricted to the two components this
    # file names. It was the plan doc's alone until 2026-08-08, which is precisely the
    # unguarded shape described above — the block lived in both docs and only one was
    # checked.
    C += _carry_bias_claims(FACTS, components=("fta", "blk"),
                            retired=("−7.0%", "−10.7%", "+6.2%", "−0.865"))

    # The regime block, restricted to what this file's summary quotes — the availability
    # plan carries the full five-row table and claims all of it.
    add("−4.63%", SEASON_REGIME,
        lambda: regime("gp_share [30+ mpg]", "regime_effect_pct") / 100.0,
        "PPP level break, 30+ mpg")
    add("0.008", SEASON_REGIME, lambda: regime("gp_share [30+ mpg]", "p_value"),
        "PPP break p-value, 30+ mpg")
    add("−4.43%", SEASON_REGIME,
        lambda: regime("gp_share [all]", "regime_effect_pct") / 100.0,
        "PPP level break, all")
    add("0.024", SEASON_REGIME, lambda: regime("gp_share [all]", "p_value"),
        "PPP break p-value, all")
    add("+5.37%", SEASON_REGIME,
        lambda: regime("gp_share [<12 mpg]", "regime_effect_pct") / 100.0,
        "PPP level break, fringe")
    add("+0.22%", SEASON_REGIME,
        lambda: regime("gp_share [30+ mpg]", "regime_effect_pct",
                       "covid_indicator") / 100.0, "COVID effect, 30+ mpg")
    add("0.91", SEASON_REGIME,
        lambda: regime("gp_share [30+ mpg]", "p_value", "covid_indicator"),
        "COVID p-value, 30+ mpg")
    add("+22.1%", SEASON_REGIME,
        lambda: regime("gp_share [<12 mpg]", "next_season_shift_pct",
                       "policy_break") / 100.0, "level+slope shift, fringe")
    add("+18.8%", SEASON_REGIME,
        lambda: regime("stl", "next_season_shift_pct", "policy_break") / 100.0,
        "level+slope shift, stl")
    add("+0.838", SHOCK_CORR, lambda: _strongest_shock_pair(), "strongest shock pair")
    add("+0.011", SHOCK_CORR, lambda: table(SHOCK_CORR)["corr"].mean(),
        "mean shock correlation")
    add("0.309", SHOCK_CORR, lambda: table(SHOCK_CORR)["corr"].abs().mean(),
        "mean |shock correlation|")
    add("136", SHOCK_CORR, lambda: rows(SHOCK_CORR), "shock correlation pairs")
    add("18.4%", SHOCK_CORR,
        lambda: (table(SHOCK_CORR)["corr"].abs() > 0.5).mean(),
        "share of pairs above |r| = 0.5")

    return C


def _strongest_shock_pair() -> float:
    frame = table(SHOCK_CORR)
    return float(frame.loc[frame["corr"].abs().idxmax(), "corr"])


def _derivation(column: str, how: str) -> float:
    frame = table(GAME_LEN)
    if frame is None:
        return float("nan")
    rows = frame[frame["analysis"] == "derivation"]
    if how == "sum":
        return float(rows[column].sum())
    if how == "max":
        return float(rows[column].max())
    return float((rows[column] * rows["games"]).sum() / rows["games"].sum())


def _break_even_hurdle(tournament: str) -> float:
    """A tournament's break-even edge hurdle, as a percentage.

    `1/(1 − rake) − 1` with `rake = 1 − prizes/(entries × fee)`, which collapses to
    `entries × fee / prizes − 1`. Mirrors `dashboard.economics.rake` /
    `break_even_hurdle` rather than importing them: nothing in `src/` imports the
    dashboard package, and the dashboard's own invariant is that the dependency never
    runs the other way either. The formula is one line and both copies are pinned by
    tests, which is cheaper than coupling the two packages.
    """
    frame = table(TOURNAMENTS)
    if frame is None:
        return float("nan")
    hit = frame[frame["type"] == tournament]
    if not len(hit):
        return float("nan")
    row = hit.iloc[0]
    pool = float(row["total_entries"]) * float(row["entry_fee_per_team"])
    return pool / float(row["total_prizes"]) - 1.0


def _readme() -> list[Claim]:
    """`README.md` — the project overview, in scientific-paper form.

    Its figures are a **selection** from the established facts and the plan docs rather than new
    measurements, and that is precisely why it needs claiming. A headline copied once
    into an overview and never refreshed is the exact failure this module was built for,
    and the README is the most-read and least-maintained document in the repo — the two
    drift incidents on record (the season-total R2 column, the report-calibration block)
    both happened in files under far more active editing than this one.

    Two things are different here from the plan-doc builders:

    - **Roundings are claimed, not skipped.** The README quotes `58%` for 57.96% and
      `86%` for an R2 of 0.859, because an overview should round. `implied_tolerance`
      already handles this correctly — a figure is wrong only if no correctly-rounded
      value could have produced it — so rounding costs no strictness worth having and
      leaving them unclaimed would exempt the most-read numbers in the repo.
    - **Shipped constants are claimed against their fitted optima.** The bonus
      overdispersions 0.10 and 0.025 are constants in `features/targets.py`, not
      measurements; claiming them against `bonus_calibration.csv`'s fitted values is
      what makes a re-calibration that moves the optimum away from the shipped constant
      show up as a failure here rather than silently.

    Three figures are deliberately left unclaimed because no artifact holds them: the
    whole-block team-context delta R2 (`+0.0086`), mid-season churn (`13.6%`), and the
    approximate skill split (`~90%`). They are unclaimed in the established facts for the
    same reason, and coverage reports them rather than hiding them.
    """
    C: list[Claim] = []

    def add(quoted, artifact, actual, label, **kw):
        C.append(_c(quoted, artifact, actual, label, doc=README, **kw))

    def context(feature: str, column: str) -> float:
        return _one(table(CONTEXT_A), column, feature=feature)

    def opp(outcome: str, column: str) -> float:
        return _one(table(OPPONENT_A), column, outcome=outcome)

    def glen(column: str) -> float:
        return _one(table(GAME_LEN), column, analysis="feasibility", season="all",
                    season_type="regular")

    def comp_m(variant: str, column: str) -> float:
        return cell(COMP_M, column, variant=variant)

    def first_k(column: str, k: str = "5") -> float:
        return _one(table(TARGET), column, analysis="season_total",
                    bucket_kind="first_k_games", bucket=k)

    def bonus_optimum(unit: str) -> float:
        return _one(table(BONUS), "overdispersion", analysis="fitted", unit=unit,
                    bucket="all")

    # ── introduction: the variance budget ─────────────────────────────────────
    add("254,167", VARIANCE, lambda: budget("own_minutes", "n_games"),
        "variance-budget population")
    add("9.445", VARIANCE,
        lambda: budget("within_player_season_residual_sd", "value"), "residual sd")
    add("14.566", VARIANCE, lambda: budget("dk_pts_sd", "value"), "total sd")
    add("57.96%", VARIANCE, lambda: budget("player_season_identity"),
        "player-season identity share")
    add("46.40%", VARIANCE, lambda: budget("own_minutes"), "own minutes share")
    add("0.691%", VARIANCE, lambda: budget("opponent_x_season"),
        "opponent x season share")
    add("0.034%", VARIANCE, lambda: budget("home_away"), "home/away share")
    # The two roundings in the "never quote a row without its basis" sentence. The
    # second is derived, because "the remaining 42%" is only true as the complement.
    add("58%", VARIANCE, lambda: budget("player_season_identity"),
        "identity share, rounded")
    add("42%", VARIANCE, lambda: 1.0 - budget("player_season_identity"),
        "the residual share, as the complement")
    add("46.4%", VARIANCE, lambda: budget("own_minutes"),
        "own minutes share, rounded")

    # ── methods: data ─────────────────────────────────────────────────────────
    add("731,906", GAME_LEN, lambda: glen("player_games"), "cleaned player-games")
    add("10,900", MATRIX_A, lambda: rows(MATRIX_A), "Tier A player-seasons")

    # ── methods: the output contract ──────────────────────────────────────────
    add("2.106", CONTEXT_A,
        lambda: context("teammate_assist_supply", "gross_dk_movement"),
        "teammate_assist_supply gross DK movement")
    add("−0.254", CONTEXT_A,
        lambda: context("teammate_assist_supply", "net_dk_movement"),
        "teammate_assist_supply net DK movement")
    add("8.30", CONTEXT_A,
        lambda: context("teammate_assist_supply", "cancellation_ratio"),
        "teammate_assist_supply cancellation ratio")
    add("1.98", OPPONENT_A, lambda: opp("dk_pts", "cancellation_ratio"),
        "opponent cancellation ratio")
    add("37,986", GAME_LEN, lambda: _derivation("games", "sum"),
        "games in the game-length derivation")
    add("5.93%", GAME_LEN, lambda: _derivation("ot_rate", "weighted"), "overtime rate")

    # ── methods: the simulation specification ─────────────────────────────────
    add("4.65", STAN_MIN_D,
        lambda: cell(STAN_MIN_D, "implied_overdispersion", metric="game_level_rho"),
        "game-level minutes overdispersion")
    add("2.43", SERIAL, lambda: serial("min", "block_inflation"),
        "minutes block variance inflation")
    add("0.10", BONUS, lambda: bonus_optimum("player_season"),
        "season-unit bonus overdispersion, against its fitted optimum")
    add("0.025", BONUS, lambda: bonus_optimum("player_game"),
        "game-unit bonus overdispersion, against its fitted optimum")

    # ── methods: contest economics ────────────────────────────────────────────
    add("10.45%", TOURNAMENTS, lambda: _break_even_hurdle("88k_alley_oop"),
        "lowest break-even edge hurdle")
    add("17.60%", TOURNAMENTS, lambda: _break_even_hurdle("600k_shootaround"),
        "highest break-even edge hurdle")

    # ── methods: the drafting layer's shape ───────────────────────────────────
    # The grid is a *size*, not a score, and it is claimed for the same reason the row
    # counts above are: an arm added to `STRATEGIES` or a change to `sim.n_worlds` moves
    # it, and nothing else in this file would notice.
    add("24", STRATEGY_SWEEP, lambda: nunique(STRATEGY_SWEEP, "strategy"),
        "strategies in the sweep")
    add("500", STRATEGY_SWEEP, lambda: max_of(STRATEGY_SWEEP, "n_sims"),
        "simulated worlds per season in the sweep")

    # ── methods: what was deprioritized ───────────────────────────────────────
    add("0.0059", DIAGNOSTICS,
        lambda: cell(DIAGNOSTICS, "delta_sequence", analysis="sequence_ablation"),
        "sequence features over season aggregates")
    add("0.0074", DIAGNOSTICS,
        lambda: cell(DIAGNOSTICS, "delta_above_null", analysis="sequence_ablation"),
        "sequence features above their shuffled null")

    # ── results: availability ─────────────────────────────────────────────────
    # The overview quotes the CRPS column and the paired interval, not the full ladder —
    # `full=False` claims exactly that subset rather than forcing the README to carry every
    # cell of a table it deliberately summarizes.
    C += _availability_ladder_claims(README, scope="headline",
                                     historical=LADDER_HISTORICAL_README)
    add("−0.1297", LADDER, lambda: ladder("gbm", "delta_vs_reference"),
        "GBM vs the shipped head")
    add("−0.3154", LADDER, lambda: ladder("gbm", "ci_lo"), "GBM CI low")
    add("+0.0672", LADDER, lambda: ladder("gbm", "ci_hi"), "GBM CI high")
    add("+0.333", LADDER, lambda: ladder("gbm", "delta_vs_reference", "gp_q1"),
        "GBM delta on the worst games quartile")
    for name, which, quoted in [("full_season", "mae_dk_total", "610.8"),
                                ("beta_binomial", "mae_dk_total", "400.5"),
                                ("full_season", "bias_dk_total", "523.3"),
                                ("beta_binomial", "bias_dk_total", "−3.1"),
                                ("oracle_gp", "mae_dk_total", "214.4"),
                                ("oracle_rate", "mae_dk_total", "261.9")]:
        add(quoted, SEASON_TOTAL, lambda n=name, w=which: treatment(n, w),
            f"season total {name} {which}")
    add("−210", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 - treatment("full_season", "mae_dk_total")),
        "head vs a full season, rounded")
    for quoted in ("646.3", "435.1", "221.3", "302.7"):
        add(quoted, SEASON_TOTAL, lambda: float("nan"),
            f"pre-lock held-out season total: {quoted}", historical=True)

    # ── results: the component floor ──────────────────────────────────────────
    count_heads = ("fga", "fta", "reb", "ast", "stl", "blk", "tov")
    fitted = ("linear", "log_own", "log_own_spline", "log_own_inter", "pca",
              "pca_spline", "pca_inter")
    add("0.81", RATES,
        lambda: min(rate(h, "carry_forward") for h in count_heads),
        "weakest no-fit floor across the count heads")
    add("0.0013", RATES,
        lambda: min(max(rate(h, v) for v in fitted) - rate(h, "carry_forward")
                    for h in count_heads),
        "smallest gain over the no-fit floor")
    add("0.0334", RATES,
        lambda: max(max(rate(h, v) for v in fitted) - rate(h, "carry_forward")
                    for h in count_heads),
        "largest gain over the no-fit floor")
    add("−19.00", STAN_C_M, lambda: stan_c("fg3a", "linear", "val_r2"),
        "fg3a under a linear predictor", historical=True)
    # Rounded to 3dp here on purpose — this is the overview, and `implied_tolerance`
    # handles the rounding. The 3dp form is what makes the README's copy independently
    # checkable rather than a transcription of the notes' 4dp table.
    add("0.673", STAN_C_M, lambda: stan_c("blk", "log_own", "val_r2"),
        "blk log_own R2, 3dp")
    add("0.831", STAN_C_M, lambda: stan_c("blk", "log_own_spline", "val_r2"),
        "blk spline R2, 3dp")
    add("0.679", STAN_C_M, lambda: stan_c("blk", "log_own", "val_r2"),
        "blk log_own R2, 3dp, on test", historical=True)
    add("0.858", STAN_C_M, lambda: stan_c("blk", "log_own_spline", "val_r2"),
        "blk spline R2, 3dp, on test", historical=True)
    add("+0.0144", STAN_C_M,
        lambda: stan_c("fta", "log_own", "val_r2") - stan_c("fta", "carry_forward",
                                                            "val_r2"),
        "fta clears its floor — the reversal the split move produced")
    # The two figures behind "the substitution arm's canonical side was handicapped":
    # `substitution_arm` fits every head at log_own, and that is the variant on which
    # `fg3a` fails its own floor. Claimed so the caveat cannot rot into a bare assertion.
    add("0.3719", STAN_C_M, lambda: stan_c("fg3a", "log_own", "val_r2"),
        "fg3a at log_own — the variant the substitution arm used", historical=True)
    add("0.9046", STAN_C_M, lambda: stan_c("fg3a", "log_own_spline", "val_r2"),
        "fg3a at its selected spline variant", historical=True)

    # ── results: the minutes composition ──────────────────────────────────────
    SEL = "betabinom_ot_graded"
    add("4.4945", COMP_M, lambda: comp_m(SEL, "val_crps"), "composition val CRPS")
    add("4.7842", COMP_M, lambda: comp_m("independent_comparator", "val_crps"),
        "independent comparator val CRPS")
    add("−6.06%", COMP_M,
        lambda: (comp_m(SEL, "val_crps")
                 / comp_m("independent_comparator", "val_crps") - 1.0),
        "composition CRPS gain, as a percentage")
    add("33.89", COMP_P,
        lambda: cell(COMP_P, "simulated", variant=SEL, analysis="team_sum_abs_error",
                     group="independent"),
        "comparator team-sum error")
    add("0.192", COMP_M, lambda: comp_m("binomial", "val_pit_ks"),
        "binomial arm PIT KS, 3dp")
    add("0.177", COMP_RHO, lambda: cell(COMP_RHO, "rho", variant=SEL, bin=1),
        "fringe-tier dispersion, 3dp")
    add("0.085", COMP_RHO, lambda: cell(COMP_RHO, "rho", variant=SEL, bin=4),
        "star-tier dispersion, 3dp")
    add("2.07", COMP_RHO,
        lambda: (cell(COMP_RHO, "rho", variant=SEL, bin=1)
                 / cell(COMP_RHO, "rho", variant=SEL, bin=4)),
        "graded dispersion spread")
    add("35%", COMP_P,
        lambda: (1.0 - mean_abs_dev(COMP_P, "ratio", 1.0, variant=SEL,
                                    analysis="variance_ratio")
                 / mean_abs_dev(COMP_P, "ratio", 1.0, variant="betabinom_ot",
                                analysis="variance_ratio")),
        "calibration error cut by grading")
    # The graded dispersion at 4dp as well as 3dp: the minutes section quotes it to four
    # places to distinguish it from the 4.65x game-level figure it is NOT, so the precision
    # is doing work and the claim has to match it.
    add("0.1768", COMP_RHO, lambda: cell(COMP_RHO, "rho", variant=SEL, bin=1),
        "fringe-tier dispersion, 4dp")
    add("0.0855", COMP_RHO, lambda: cell(COMP_RHO, "rho", variant=SEL, bin=4),
        "star-tier dispersion, 4dp")
    # The per-team-game floor, which the composition clears on the same draws that fail the
    # season-unit one. The pair is the whole "a head is only a model at the unit it was
    # scored at" claim, so both halves are claimed rather than the contrast asserted.
    add("4.6776", COMP_M, lambda: comp_m("carry_forward", "val_crps"),
        "the composition's per-team-game no-fit floor, which it does clear")

    # ── results: the two minutes heads at the season unit ─────────────────────
    C.extend(_minutes_unification_claims(README))

    # ── results: substitution and season terms ────────────────────────────────
    # The README's claim used to be that the gain *replicates across splits*. Gate 0 is
    # validation-only since 2026-08-06, so there is no second split to replicate on and the
    # word is gone from the prose. What replaces it is a different and better-founded
    # replication: the margin survived a doubling of chain length. The test figures stay in
    # the README as the record of what the adoption was decided on, presence-checked only.
    add("−0.771", STAN_C_S,
        lambda: cell(STAN_C_S, "reparam_minus_canonical", split="val",
                     arm="two_counts"),
        "substitution gain, val (handicapped)")
    add("−0.793", SHOT_SWEEP, lambda: float("nan"),
        "substitution gain, test (handicapped)", historical=True)
    add("−0.493549", SHOT_SWEEP, lambda: float("nan"),
        "un-handicapped substitution gain, test", historical=True)

    def shot_joint(arm: str) -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        sub = frame[(frame["analysis"] == "joint") & (frame["split"] == "val")
                    & (frame["arm"] == arm) & frame["selected"].astype(bool)]
        return float(sub["mean_nll"].iloc[0])

    def shot_floor(name: str, variant: str) -> float:
        arm = "two_counts" if name in ("fg2a", "fg3a") else "fga_x_fg3a_share"
        return cell(SHOT_SWEEP, "floor_nll", analysis="head", split="val", arm=arm,
                    head=name, variant=variant)

    add("−0.501041", SHOT_SWEEP,
        lambda: shot_joint("fga_x_fg3a_share") - shot_joint("two_counts"),
        "un-handicapped substitution gain, val")
    add("−0.440841", SHOT_SWEEP,
        lambda: (shot_floor("fga", "log_own") + shot_floor("fg3a|fga", "logit_own")
                 - shot_joint("two_counts")),
        "reparameterized floor vs canonical fitted")
    # The oracle bounds every form of season term, so both ends of the range the
    # README quotes are claimed — the "~3%" ceiling and the median.
    term_heads = ("fga", "fta", "reb", "ast", "stl", "blk", "tov")
    add("5%", TERM_M, lambda: max(oracle_gain(h) for h in term_heads),
        "the league-oracle ceiling")
    add("1.29%", TERM_M,
        lambda: float(pd.Series([oracle_gain(h) for h in term_heads]).median()),
        "the league-oracle median")
    add("4.97%", TERM_M, lambda: max(oracle_gain(h) for h in term_heads),
        "the league-oracle maximum")
    add("10.4%", TERM_SPREAD,
        lambda: term_spread(15, "year") - 1.0,
        "year-effect roster spread at 15 players")
    add("0.5%", STAN_AV_B,
        lambda: cell(STAN_AV_B, "inflation", n_players=15) - 1.0,
        "shared-beta roster spread at 15 players")

    # ── results: the drafting layer ───────────────────────────────────────────
    # The pair the README's headline turns on. They are two columns of one row, so a
    # re-sweep that moved only the realized side — the one with N = 2 seasons behind it —
    # would show up here as a single disagreement rather than as a silently updated story.
    def shipped(column: str) -> float:
        return cell(STRATEGY_SHIPPED, column, tournament="600k_shootaround")

    add("0.1890", STRATEGY_SHIPPED, lambda: shipped("sim_lift"),
        "shipped arm's simulated advance lift, 600k")
    add("0.1713", STRATEGY_SHIPPED, lambda: shipped("realized_lift"),
        "shipped arm's realized advance lift, 600k")
    # Gate D's failure is a *count of zero*, which is the one shape of result that decays
    # silently: a sweep that started separating the tiers would leave the prose true-looking
    # and wrong. Both ends are claimed, so the denominator cannot drift either.
    add("6", STRATEGY_GATE_D, lambda: rows(STRATEGY_GATE_D),
        "Gate D paired comparisons")
    add("0", STRATEGY_GATE_D, lambda: total(STRATEGY_GATE_D, "materially_different"),
        "Gate D comparisons that separate the tiers")

    # ── results: the field, and the execution axis (2026-08-11) ──────────────
    # The fitted need weight is a zero the same way Gate D's count is: a recalibration
    # that started selecting a positive lean would leave the README's "measured null"
    # claim true-looking and wrong, so the selected row is claimed directly.
    add("0", GATE_B_NEED,
        lambda: cell(GATE_B_NEED, "need_weight", season="pooled", selected=True),
        "fitted field lineup-reasoning lean, picks")
    add("0.306", SHIPPED_NEED,
        lambda: cell(SHIPPED_NEED, "sim_lift", tournament="600k_shootaround"),
        "shipped arm's simulated lift against the stipulated need-aware field, 600k")
    add("+0.0144", STRATEGY_PAIRED,
        lambda: cell(STRATEGY_PAIRED, "gap", tournament="600k_shootaround",
                     metric="p_advance", baseline="blend_a30",
                     strategy="autodraft_blend_a30"),
        "autodraft twin over the uncapped click of the same ranking, 600k")

    def sweep_mean_lift(arm: str) -> float:
        frame = table(STRATEGY_SWEEP)
        if frame is None:
            return float("nan")
        hit = frame[(frame["tournament"] == "600k_shootaround")
                    & (frame["strategy"] == arm)]
        return float(hit["lift_vs_null"].mean()) if len(hit) else float("nan")

    add("0.0975", STRATEGY_SHIPPED,
        lambda: shipped("sim_lift") - sweep_mean_lift("autodraft_blend_a30"),
        "lift given up by autodrafting instead of the shipped objective, 600k")

    # ── discussion ────────────────────────────────────────────────────────────
    add("0.317", PROFILE,
        lambda: prof("persistence", "gp_share", "r_within_weighted"),
        "games-played persistence")
    add("86%", TARGET, lambda: first_k("r2_extrapolated"),
        "first-5-games season-total share, rounded")
    add("0.859", TARGET, lambda: first_k("r2_extrapolated"),
        "first-5-games season-total R2")
    add("14.7%", ROSTER_A, lambda: roster("undescribed"),
        "roster minutes with no usable prior-season row")

    # ── the availability head's trials assumption (§2, availability) ──────────
    # Two figures only, and deliberately the two that carry opposite halves of the
    # sentence: the size of the error and the size of what already pays for it. An
    # overview that quoted the first without the second would read as an open defect.
    add("9.1×", AEXCH,
        lambda: 1.0 / cell(AEXCH, "exchangeable_ratio", analysis="period_gap",
                           population="30+ mpg", metric="p_dead_run"),
        "star P(3 consecutive dead periods), exchangeable understatement")
    add("81.5%", AEXCH,
        lambda: cell(AEXCH, "recovered_share", analysis="period_gap", population="all",
                     metric="p_dead_period"),
        "pooled share of the exchangeability gap the shipped layout recovers")

    return C


def _shot_basis() -> list[Claim]:
    """`docs/shot-attempt-basis-plan.md` — Gate 0 of the shot-attempt reparameterization.

    Every figure in that doc comes from one artifact, `stan_component_substitution_sweep`,
    except the arm-A grid, which is read out of `stan_component_metrics.csv` at zero fits
    and is therefore claimed against *that* file. The two are audited together on purpose:
    the doc's headline is a comparison between them, and a comparison whose two sides come
    from different runs is exactly the drift this module exists to catch.
    """
    C: list[Claim] = []
    add = C.append

    def head(name: str, variant: str, column: str = "mean_nll") -> float:
        arm = "two_counts" if name in ("fg2a", "fg3a") else "fga_x_fg3a_share"
        return cell(SHOT_SWEEP, column, analysis="head", split="val", arm=arm,
                    head=name, variant=variant)

    def joint(arm: str) -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        sub = frame[(frame["analysis"] == "joint") & (frame["split"] == "val")
                    & (frame["arm"] == arm) & frame["selected"].astype(bool)]
        return float(sub["mean_nll"].iloc[0])

    def retired(quoted: str, label: str) -> None:
        """A test-split figure the doc keeps as a record, with nothing left to check it.

        These were value-checked until 2026-08-06, when `make stan-substitution` was re-run
        validation-only. The doc keeps them in one clearly-marked block; the audit keeps
        them presence-checked, so the failure they still guard is deletion.
        """
        add(_c(quoted, SHOT_SWEEP, lambda: float("nan"), label, doc=SHOT,
               historical=True))

    # ── the per-factor table, every variant against its own no-fit floor ──────
    per_factor = [
        ("fg2a", "log_own", "5.263114"),
        ("fg3a", "log_own_spline", "5.242045"),
        ("fga", "linear", "5.407505"),
        ("fga", "log_own", "5.389932"),
        ("fga", "log_own_spline", "5.388496"),
        ("fg3a|fga", "linear", "4.912709"),
        ("fg3a|fga", "logit_own", "4.636018"),
        ("fg3a|fga", "logit_own_spline", "4.615622"),
    ]
    for name, variant, quoted in per_factor:
        add(_c(quoted, SHOT_SWEEP,
               lambda h=name, v=variant: head(h, v),
               f"{name}@{variant} validation NLL", doc=SHOT))
    # The floors are pure arithmetic — a shrunk carry-forward with `k` fitted on train — so
    # they are the one part of this gate that reproduces to the digit across a refit, and
    # `stan_component_metrics.csv` independently agrees on the two arm-B rows.
    floors = [("fg2a", "log_own", "5.271028"),
              ("fg3a", "log_own_spline", "5.903396"),
              ("fga", "log_own", "5.445210"),
              ("fg3a|fga", "logit_own", "4.619109")]
    for name, variant, quoted in floors:
        add(_c(quoted, SHOT_SWEEP,
               lambda h=name, v=variant: head(h, v, "floor_nll"),
               f"{name} no-fit floor", doc=SHOT))

    # ── the verdict ──────────────────────────────────────────────────────────
    add(_c("10.505159", SHOT_SWEEP, lambda: joint("two_counts"),
           "arm A joint NLL", doc=SHOT))
    add(_c("10.004118", SHOT_SWEEP, lambda: joint("fga_x_fg3a_share"),
           "arm B joint NLL", doc=SHOT))
    add(_c("−0.501041", SHOT_SWEEP,
           lambda: joint("fga_x_fg3a_share") - joint("two_counts"),
           "reparameterization margin", doc=SHOT))

    # The handicap decomposition, now entirely on validation — it needed the test column
    # until 2026-08-06. `pinned` is arm B at the single variant `substitution_arm` used, so
    # the rows below split the recorded margin into "what un-handicapping arm A cost" and
    # "what sweeping arm B added on top", and the two must sum back to it.
    def pinned() -> float:
        return head("fga", "log_own") + head("fg3a|fga", "logit_own")

    def handicapped(arm: str) -> float:
        return cell(STAN_C_S, "mean_joint_nll", split="val", arm=arm)

    add(_c("10.797078", STAN_C_S, lambda: handicapped("two_counts"),
           "arm A with both heads at log_own — the handicapped arm", doc=SHOT))
    add(_c("0.291919", SHOT_SWEEP,
           lambda: handicapped("two_counts") - joint("two_counts"),
           "the handicap, in nats — on validation", doc=SHOT))
    add(_c("−0.771128", STAN_C_S,
           lambda: cell(STAN_C_S, "reparam_minus_canonical", split="val",
                        arm="two_counts"),
           "the recorded handicapped margin, full precision", doc=SHOT))
    add(_c("−0.021832", SHOT_SWEEP, lambda: joint("fga_x_fg3a_share") - pinned(),
           "what sweeping arm B added", doc=SHOT))
    add(_c("65%", SHOT_SWEEP,
           lambda: (joint("fga_x_fg3a_share") - joint("two_counts"))
           / cell(STAN_C_S, "reparam_minus_canonical", split="val", arm="two_counts"),
           "share of the recorded margin that survives the correction", doc=SHOT))
    # The regression check: the gate's pinned arm B and `substitution_arm`'s arm B are the
    # same quantity computed by different code in different modules. Claimed from both
    # artifacts against the one quoted string, so they cannot drift apart silently.
    add(_c("10.025950", SHOT_SWEEP, pinned,
           "arm B pinned at (log_own, logit_own)", doc=SHOT))
    add(_c("10.025950", STAN_C_S, lambda: handicapped("fga_x_fg3a_share"),
           "…and `substitution_arm` computes it identically", doc=SHOT))

    # ── the headline: the coordinate change beats the fitting ─────────────────
    def floor_total(names: tuple[str, str], variants: tuple[str, str]) -> float:
        return sum(head(n, v, "floor_nll") for n, v in zip(names, variants))

    arm_a_floor = (("fg2a", "fg3a"), ("log_own", "log_own_spline"))
    arm_b_floor = (("fga", "fg3a|fga"), ("log_own", "logit_own"))
    add(_c("11.174424", SHOT_SWEEP, lambda: floor_total(*arm_a_floor),
           "arm A no-fit floor total", doc=SHOT))
    add(_c("10.064318", SHOT_SWEEP, lambda: floor_total(*arm_b_floor),
           "arm B no-fit floor total", doc=SHOT))
    add(_c("−1.110105", SHOT_SWEEP,
           lambda: floor_total(*arm_b_floor) - floor_total(*arm_a_floor),
           "floor-to-floor gain from the coordinate change", doc=SHOT))
    add(_c("−0.440841", SHOT_SWEEP,
           lambda: floor_total(*arm_b_floor) - joint("two_counts"),
           "arm B's floor against arm A fitted", doc=SHOT))
    add(_c("−0.060200", SHOT_SWEEP,
           lambda: joint("fga_x_fg3a_share") - floor_total(*arm_b_floor),
           "what arm B's own fitting adds", doc=SHOT))

    # ── population and sampler ────────────────────────────────────────────────
    add(_c("773", SHOT_SWEEP, lambda: head("fg2a", "log_own", "n"),
           "validation rows", doc=SHOT))
    add(_c("18", SHOT_SWEEP, lambda: rows(SHOT_SWEEP), "sweep artifact rows", doc=SHOT))
    add(_c("1.0047", SHOT_D, lambda: max_of(SHOT_D, "max_rhat"), "max R-hat", doc=SHOT))
    add(_c("0", SHOT_D, lambda: total(SHOT_D, "divergences"), "divergences", doc=SHOT))
    add(_c("26.8", SHOT_D, lambda: total(SHOT_D, "wall_clock_s") / 60,
           "sampler minutes", doc=SHOT))
    add(_c("8", SHOT_D, lambda: rows(SHOT_D), "fits", doc=SHOT))

    # ── the retired test half, kept as a record ───────────────────────────────
    # Every figure below was value-checked until the 2026-08-06 validation-only re-run.
    # They stay in the doc's "what the re-run retired" block and stay presence-checked.
    for quoted, label in [
            ("5.229492", "fg2a@log_own"), ("5.248561", "fg3a@log_own_spline"),
            ("5.426520", "fga@linear"), ("5.379640", "fga@log_own"),
            ("5.376766", "fga@log_own_spline"), ("4.929134", "fg3a|fga@linear"),
            ("4.611402", "fg3a|fga@logit_own"),
            ("4.607737", "fg3a|fga@logit_own_spline"),
            ("5.292842", "fg2a floor"), ("5.731185", "fg3a floor"),
            ("5.432802", "fga floor"), ("4.652797", "fg3a|fga floor"),
            ("10.478052", "arm A joint"), ("9.984503", "arm B joint"),
            ("−0.493549", "the margin the adoption was decided on"),
            ("10.476413", "arm A best-of-16"),
            ("−0.491910", "arm B against the best-of-16"),
            ("0.001640", "what the best-of-16 bought over arm A's selected pair"),
            ("0.305646", "the handicap, in nats"),
            ("−0.487010", "margin before arm B was swept"),
            ("−0.006539", "what sweeping arm B added"),
            ("9.991042", "the refactored share arm's regression check"),
            ("11.024027", "arm A no-fit floor total"),
            ("10.085599", "arm B no-fit floor total"),
            ("−0.938427", "floor-to-floor gain"),
            ("−0.390814", "arm B's floor against arm A's best fitted"),
            ("−0.101096", "what arm B's own fitting adds"),
            ("791", "test rows"), ("52", "sweep artifact rows"),
            ("16", "fits"), ("1.0087", "max R-hat"), ("46.5", "sampler minutes"),
            ("−0.793", "the recorded handicapped margin"),
            ("0.792657", "the recorded handicapped margin, unsigned")]:
        retired(quoted, f"retired test-split figure: {label}")

    # The validation half of the handicapped pair is still live, in the one artifact that
    # still carries it.
    add(_c("−0.771", STAN_C_S,
           lambda: cell(STAN_C_S, "reparam_minus_canonical", split="val",
                        arm="two_counts"),
           "the recorded handicapped margin, val", doc=SHOT))
    add(_c("0.3719", STAN_C_M,
           lambda: cell(STAN_C_M, "val_r2", head="fg3a", variant="log_own"),
           "fg3a at log_own — the handicap", doc=SHOT, historical=True))
    add(_c("0.9046", STAN_C_M,
           lambda: cell(STAN_C_M, "val_r2", head="fg3a", variant="log_own_spline"),
           "fg3a at its shipped spline", doc=SHOT, historical=True))
    add(_c("−0.1248", RESID,
           lambda: _windowed(RESID, "r", basis="legacy_two_count_basis",
                             component_a="fg3a", component_b="fg2a"),
           "the substitution off-diagonal, legacy basis", doc=SHOT))
    add(_c("−0.0836", RESID,
           lambda: _windowed(RESID, "r", basis="minutes_conditioned",
                             component_a="fga", component_b="fg3a|fga"),
           "the substitution off-diagonal, shipped basis", doc=SHOT))
    add(_c("+0.7853", RESID, lambda: resid_min_eig(),
           "copula min eigenvalue", doc=SHOT))
    add(_c("+0.0070", RESID, lambda: resid("mean"),
           "conditioned off-diagonal mean", doc=SHOT))
    add(_c("0.886", PERSIST, lambda: persist("sco_pct_fga_3pt"),
           "shot-mix share persistence", doc=SHOT))
    add(_c("731,906", GAME_LEN, lambda: cell(GAME_LEN, "player_games", analysis="feasibility",
                             season="all", season_type="regular"), "player-games", doc=SHOT))
    return C


def h_label(arm: str) -> str:
    return "arm A" if arm == "two_counts" else "arm B"


def gp_gate(arm: str, metric: str, column: str = "simulated") -> float:
    return _one(table(GP_GATE), column, arm=arm, metric=metric)


def gp_gate_z(arm: str) -> float:
    """The gate's own statistic: how many standard errors of the OBSERVED proportion the
    simulated left tail sits from it. Recomputed here rather than read, because the
    artifact carries the two proportions and the row count and the z is arithmetic on
    them — so a claim on it is a claim on the gate's logic and not only on its output."""
    o = gp_gate(arm, "p_below_41", "observed")
    s = gp_gate(arm, "p_below_41", "simulated")
    n = _one(table(GP_GATE), "n", arm=arm, metric="p_below_41")
    return (s - o) / np.sqrt(o * (1 - o) / n)


def gp_collapse(column: str, window: str = "full") -> float:
    return _one(table(GP_COLLAPSE), column, window=window)


def gp_spell(column: str, **where) -> float:
    return _one(table(GP_SPELLS), column, **where)


def gp_metric(variant: str, column: str) -> float:
    return _one(table(GP_METRICS), column, variant=variant)


def gp_gates(gate: str, column: str, arm: str | None = None) -> float:
    """One gate row. `arm` is required for Gate D, which has one row per candidate —
    the selected fitted arm and the closed-form fallback — because quoting a Gate D
    figure without naming which candidate it belongs to is the same class of error as
    quoting a variance-budget share without its basis."""
    where = {"gate": gate}
    if arm is not None:
        where["arm"] = arm
    return _one(table(GP_GATES), column, **where)


def gp_process(component: str, statistic: str) -> float:
    return _one(table(GP_PROCESS), "value", component=component, statistic=statistic)


def _games_played() -> list[Claim]:
    """`docs/games-played-plan.md` — the spell process."""
    C: list[Claim] = []

    def add(quoted: str, artifact: str, actual: Callable[[], float], label: str,
            historical: bool = False) -> None:
        C.append(_c(quoted, artifact, actual, label, doc=GAMES,
                    historical=historical))

    # ── the collapse ──────────────────────────────────────────────────────────
    add("32,944", GP_COLLAPSE, lambda: gp_collapse("collapsed_binomial_rows"),
        "collapsed binomial rows")
    add("1,297,766", GP_COLLAPSE, lambda: gp_collapse("transitions"),
        "full-window transitions")
    add("942,597", GP_COLLAPSE, lambda: gp_collapse("transitions", "appearance"),
        "appearance-window transitions")
    add("39.4", GP_COLLAPSE, lambda: gp_collapse("collapse_ratio"), "collapse ratio")
    add("28.6", GP_COLLAPSE, lambda: gp_collapse("collapse_ratio", "appearance"),
        "appearance collapse ratio")
    for window, on, at_risk, hazard, rec, at_miss in [
            ("full", "75,838", "727,246", "0.1043", "75,496", "570,520"),
            ("appearance", "68,530", "719,938", "0.0952", "68,530", "222,659")]:
        add(on, GP_COLLAPSE, lambda w=window: gp_collapse("onsets", w),
            f"{window} onsets")
        add(at_risk, GP_COLLAPSE, lambda w=window: gp_collapse("at_risk_played", w),
            f"{window} at-risk transitions")
        add(hazard, GP_COLLAPSE, lambda w=window: gp_collapse("onset_hazard", w),
            f"{window} onset hazard")
        add(rec, GP_COLLAPSE, lambda w=window: gp_collapse("recoveries", w),
            f"{window} recoveries")
        add(at_miss, GP_COLLAPSE, lambda w=window: gp_collapse("at_risk_missed", w),
            f"{window} at-risk missed")
    add("0.1323", GP_COLLAPSE, lambda: gp_collapse("recovery_hazard"),
        "full recovery hazard")
    add("0.3078", GP_COLLAPSE,
        lambda: gp_collapse("recovery_hazard", "appearance"),
        "appearance recovery hazard")

    # ── spell classes ─────────────────────────────────────────────────────────
    for name, spells, missed, mean, share_s, share_m, nr in [
            ("interior", "68,530", "222,659", "3.25", "82.8%", "38.5%", "0.010"),
            ("left_truncated", "6,966", "189,668", "27.23", "8.4%", "32.8%", "0.587"),
            ("right_censored", "7,308", "165,501", "22.65", "8.8%", "28.6%", "0.462")]:
        add(spells, GP_SPELLS, lambda k=name: gp_spell("spells", spell_class=k),
            f"{name} spells")
        add(missed, GP_SPELLS,
            lambda k=name: gp_spell("missed_games", spell_class=k),
            f"{name} missed games")
        add(mean, GP_SPELLS, lambda k=name: gp_spell("mean_length", spell_class=k),
            f"{name} mean length")
        add(share_s, GP_SPELLS,
            lambda k=name: gp_spell("share_of_spells", spell_class=k),
            f"{name} share of spells")
        add(share_m, GP_SPELLS,
            lambda k=name: gp_spell("share_of_missed", spell_class=k),
            f"{name} share of missed games")
        add(nr, GP_SPELLS,
            lambda k=name: gp_spell("not_rostered_share", spell_class=k),
            f"{name} not_rostered share")
    add("82,804", GP_SPELLS,
        lambda: gp_spell("n_spells", treatment="proper_censoring"), "total spells")

    # ── the censoring bias, and the two derived columns that were corrected ───
    # Not named `treatment` — that is a module-level accessor, and assigning to the name
    # anywhere in this function would make every reference to it local.
    for how, a, b, p26 in [("drop_censored", "1.088", "1.323", "0.0398"),
                           ("censored_as_complete", "0.917", "1.252", "0.0598"),
                           ("proper_censoring", "0.721", "0.978", "0.0861")]:
        add(a, GP_SPELLS, lambda t=how: gp_spell("a", treatment=t), f"{how} a")
        add(b, GP_SPELLS, lambda t=how: gp_spell("b", treatment=t), f"{how} b")
        add(p26, GP_SPELLS, lambda t=how: gp_spell("p_ge_26", treatment=t),
            f"{how} P(T>=26)")
    add("16.03", GP_SPELLS,
        lambda: gp_spell("mean_spell", treatment="drop_censored"),
        "drop_censored E[T]")
    add("2.16", GP_SPELLS,
        lambda: (gp_spell("p_ge_26", treatment="proper_censoring")
                 / gp_spell("p_ge_26", treatment="drop_censored")),
        "how far dropping censored spells understates the tail")
    add("1.44", GP_SPELLS,
        lambda: (gp_spell("p_ge_26", treatment="proper_censoring")
                 / gp_spell("p_ge_26", treatment="censored_as_complete")),
        "how far treating censored as complete understates the tail")
    # The superseded derived columns, kept beside their corrections. The FITTED
    # parameters reproduced exactly across two implementations; only the arithmetic on
    # top of them moved, which is what makes these a reversal worth preserving.
    for quoted, label in [("5.84", "planning E[T], drop_censored"),
                          ("7.68", "planning E[T], censored as complete"),
                          ("9.95", "planning E[T], proper censoring"),
                          ("0.0377", "planning P(T>=26), drop_censored"),
                          ("0.0549", "planning P(T>=26), censored as complete"),
                          ("0.0741", "planning P(T>=26), proper censoring")]:
        add(quoted, GP_SPELLS,
            lambda: gp_spell("p_ge_26", treatment="proper_censoring"), label,
            historical=True)

    # ── the duration candidates ───────────────────────────────────────────────
    add("126,171.9", GP_SPELLS, lambda: gp_spell("nll", model="beta_geometric"),
        "beta-geometric nll")
    add("252,347.7", GP_SPELLS, lambda: gp_spell("aic", model="beta_geometric"),
        "beta-geometric AIC")
    add("137,450.3", GP_SPELLS, lambda: gp_spell("nll", model="geometric"),
        "geometric nll")
    add("274,902.6", GP_SPELLS, lambda: gp_spell("aic", model="geometric"),
        "geometric AIC")
    add("1.762", GP_SPELLS, lambda: gp_spell("a", model="beta_geometric"),
        "fitted a on interior spells")
    add("1.926", GP_SPELLS, lambda: gp_spell("b", model="beta_geometric"),
        "fitted b on interior spells")
    add("0.4777", GP_SPELLS, lambda: gp_spell("p_eq_1", model="beta_geometric"),
        "beta-geometric P(T=1)")
    add("0.4829", GP_SPELLS, lambda: gp_spell("p_eq_1", model="observed"),
        "observed P(T=1)")
    add("0.0635", GP_SPELLS, lambda: gp_spell("p_ge_10", model="observed"),
        "observed P(T>=10)")
    add("0.0106", GP_SPELLS, lambda: gp_spell("p_ge_26", model="observed"),
        "observed P(T>=26)")
    add("11,278", GP_SPELLS,
        lambda: gp_spell("nll", model="geometric") - gp_spell("nll",
                                                              model="beta_geometric"),
        "log-likelihood the geometric gives up")
    # Quoted in README.md too — the same figure from two docs against the one artifact,
    # because "current in one doc and stale in the other" is the failure that has already
    # happened twice in this repo.
    C.append(_c("11,278", GP_SPELLS,
                lambda: gp_spell("nll", model="geometric")
                - gp_spell("nll", model="beta_geometric"),
                "log-likelihood the geometric gives up", doc=README))

    # ── left truncation ───────────────────────────────────────────────────────
    add("−1.7042", GP_SPELLS,
        lambda: gp_spell("open_shift", analysis="left_truncation"),
        "fitted in-progress offset")
    add("0.1367", GP_SPELLS,
        lambda: gp_spell("mu_open_fitted", analysis="left_truncation"),
        "fitted in-progress mu")
    add("0.3027", GP_SPELLS,
        lambda: gp_spell("mu_open_predicted", analysis="left_truncation"),
        "size-biased identity's prediction")
    add("2.2", GP_SPELLS,
        lambda: (gp_spell("mu_open_predicted", analysis="left_truncation")
                 / gp_spell("mu_open_fitted", analysis="left_truncation")),
        "how far in-progress spells exceed renewal")

    # ── the hazard-by-streak curve ────────────────────────────────────────────
    for streak, hazard, at_risk in [("1", "0.2976", "81,525"), ("2", "0.1912", "55,387"),
                                    ("3", "0.1386", "43,601"), ("4", "0.1067", "36,694"),
                                    ("5", "0.0926", "32,125"),
                                    ("6-10", "0.0672", "120,539"),
                                    ("11-20", "0.0463", "142,519"),
                                    ("21-41", "0.0338", "133,322"),
                                    ("42-10000", "0.0208", "74,226")]:
        add(hazard, GP_SPELLS,
            lambda s=streak: gp_spell("hazard", analysis="hazard_by_streak", streak=s),
            f"onset hazard at streak {streak}")
        add(at_risk, GP_SPELLS,
            lambda s=streak: gp_spell("at_risk", analysis="hazard_by_streak", streak=s),
            f"at-risk transitions at streak {streak}")
    add("14.3", GP_SPELLS,
        lambda: (gp_spell("hazard", analysis="hazard_by_streak", streak="1")
                 / gp_spell("hazard", analysis="hazard_by_streak", streak="42-10000")),
        "how far the onset hazard falls across the streak range")
    for quoted, label in [("0.3164", "planning hazard at streak 1"),
                          ("0.0265", "planning hazard past 41"),
                          ("11.9", "planning hazard fall")]:
        add(quoted, GP_SPELLS,
            lambda: gp_spell("hazard", analysis="hazard_by_streak", streak="1"),
            label, historical=True)

    # ── the structural proxy against the box-score status ─────────────────────
    add("82.7%", GP_SPELLS,
        lambda: gp_spell("value", analysis="status_agreement", statistic="agreement"),
        "structural proxy vs not_rostered agreement")
    add("16.74%", GP_SPELLS,
        lambda: gp_spell("value", analysis="status_agreement",
                         statistic="missed_outside_window_while_rostered"),
        "missed outside the window while still rostered")
    add("98.5%", GP_SPELLS,
        lambda: gp_spell("value", analysis="status_agreement",
                         statistic="not_rostered_persistence"),
        "not_rostered per-game persistence")
    add("422,219", GP_SPELLS,
        lambda: gp_spell("n", analysis="status_agreement", statistic="agreement"),
        "missed games from 2006-07 with covered status")
    add("16.75%", GP_SPELLS,
        lambda: gp_spell("value", analysis="status_agreement",
                         statistic="missed_outside_window_while_rostered"),
        "planning rounding of the rostered-but-absent cell", historical=True)

    # ── Gate 0 ────────────────────────────────────────────────────────────────
    add("4,667", GP_GATE, lambda: _one(table(GP_GATE), "n",
                                       arm="tenure_decomposition",
                                       metric="p_below_41"),
        "Gate 0 population")
    for arm, mean, sd, over, p41, p60 in [
            ("full_window_chain", "0.7988", "0.2399", "29.37", "0.1356", "0.3189"),
            ("tenure_decomposition", "0.8132", "0.2197", "26.05", "0.1155",
             "0.3009")]:
        add(mean, GP_GATE, lambda a=arm: gp_gate(a, "mean_gp_share"), f"{arm} mean")
        add(sd, GP_GATE, lambda a=arm: gp_gate(a, "sd_gp_share"), f"{arm} sd")
        add(over, GP_GATE, lambda a=arm: gp_gate(a, "overdispersion"),
            f"{arm} overdispersion")
        add(p41, GP_GATE, lambda a=arm: gp_gate(a, "p_below_41"), f"{arm} P(GP<41)")
        add(p60, GP_GATE, lambda a=arm: gp_gate(a, "p_below_60"), f"{arm} P(GP<60)")
    add("0.8070", GP_GATE,
        lambda: gp_gate("tenure_decomposition", "mean_gp_share", "observed"),
        "observed mean")
    add("0.2120", GP_GATE,
        lambda: gp_gate("tenure_decomposition", "sd_gp_share", "observed"),
        "observed sd")
    add("23.66", GP_GATE,
        lambda: gp_gate("tenure_decomposition", "overdispersion", "observed"),
        "observed overdispersion")
    add("0.1112", GP_GATE,
        lambda: gp_gate("tenure_decomposition", "p_below_41", "observed"),
        "observed P(GP<41)")
    add("0.3107", GP_GATE,
        lambda: gp_gate("tenure_decomposition", "p_below_60", "observed"),
        "observed P(GP<60)")
    add("21.9%", GP_GATE,
        lambda: (gp_gate("full_window_chain", "p_below_41")
                 / gp_gate("full_window_chain", "p_below_41", "observed") - 1.0),
        "plain chain tail over-prediction")
    add("3.8%", GP_GATE,
        lambda: (gp_gate("tenure_decomposition", "p_below_41")
                 / gp_gate("tenure_decomposition", "p_below_41", "observed") - 1.0),
        "tenure decomposition tail error")
    add("+5.29", GP_GATE, lambda: gp_gate_z("full_window_chain"),
        "plain chain z on the left tail")
    add("+0.93", GP_GATE, lambda: gp_gate_z("tenure_decomposition"),
        "tenure decomposition z on the left tail")
    add("0.0046", GP_GATE,
        lambda: np.sqrt(gp_gate("tenure_decomposition", "p_below_41", "observed")
                        * (1 - gp_gate("tenure_decomposition", "p_below_41",
                                       "observed"))
                        / _one(table(GP_GATE), "n", arm="tenure_decomposition",
                               metric="p_below_41")),
        "standard error of the observed left tail")

    # ── the arm ladder, VALIDATION only ───────────────────────────────────────
    # There is no test column any more: `src/models/held_out.py` locks the held-out split
    # and the sweep never reaches it. The end-of-project figures live in
    # `final_evaluation.csv` and are claimed only once that has been run.
    ladder = [("floor", "10.0057", "10.0057", "0.0939", "23.7251", "0.0406"),
              ("within_tenure", "7.23495", "10.0992", "0.1259", "11.4603", "0.0092"),
              ("full_window", "10.2797", "10.0057", "0.0755", "22.1686", "0.0496"),
              ("three_state", "10.3466", "10.0501", "0.1204", "20.3687", "0.0043"),
              ("duration_covariates", "10.1676", "10.0057", "0.0672", "22.1055",
               "0.0535"),
              ("calibrated_fallback", "10.0207", "10.0057", "0.1017", "23.7251",
               "0.0440"),
              ("hybrid", "10.0057", "10.0057", "0.0939", "23.7251", "0.0406")]
    for arm, val, floor, pit, od, tail in ladder:
        add(val, GP_METRICS, lambda a=arm: gp_metric(a, "val_crps"), f"{arm} val CRPS")
        add(floor, GP_METRICS, lambda a=arm: gp_metric(a, "floor_val_crps"),
            f"{arm} floor")
        add(pit, GP_METRICS, lambda a=arm: gp_metric(a, "val_pit_ks"), f"{arm} PIT KS")
        add(od, GP_METRICS, lambda a=arm: gp_metric(a, "val_implied_overdispersion"),
            f"{arm} implied overdispersion")
        add(tail, GP_METRICS, lambda a=arm: gp_metric(a, "val_tail_error"),
            f"{arm} tail error")
    for arm, quoted in [("within_tenure", "−2.8642"), ("full_window", "+0.2739"),
                        ("three_state", "+0.2965"),
                        ("duration_covariates", "+0.1619"),
                        ("calibrated_fallback", "+0.0149")]:
        add(quoted, GP_METRICS, lambda a=arm: gp_metric(a, "crps_vs_floor"),
            f"{arm} vs its floor")

    # ── the gates, all on validation ──────────────────────────────────────────
    add("0.78", GP_GATES, lambda: gp_gates("A", "extrapolated_hours"),
        "Gate A corrected estimate")
    add("−0.350239", GP_GATES, lambda: gp_gates("B", "floor_loglik_per_transition"),
        "Gate B floor")
    add("−0.330662", GP_GATES, lambda: gp_gates("B", "head_loglik_per_transition"),
        "Gate B fitted head")
    add("+0.019577", GP_GATES, lambda: gp_gates("B", "gain"), "Gate B gain")
    add("48.1", GP_GATES, lambda: gp_gates("B", "floor_shrinkage_k"),
        "Gate B floor shrinkage")
    add("0.0741", GP_GATES, lambda: gp_gates("B", "floor_league_rate"),
        "Gate B league onset rate")
    for arm, crps, pit, p41, p60, tail in [
            ("duration_covariates", "10.1676", "0.0672", "0.1597", "0.3734", "0.0535"),
            ("calibrated_fallback", "10.0207", "0.1017", "0.1499", "0.3643", "0.0440"),
            ("hybrid", "10.0057", "0.0939", "0.1553", "0.3521", "0.0406")]:
        add(crps, GP_GATES, lambda a=arm: gp_gates("D", "crps", a), f"Gate D {arm} CRPS")
        add(pit, GP_GATES, lambda a=arm: gp_gates("D", "pit_ks", a),
            f"Gate D {arm} PIT KS")
        add(p41, GP_GATES, lambda a=arm: gp_gates("D", "predicted_below_41", a),
            f"Gate D {arm} P(GP<41)")
        add(p60, GP_GATES, lambda a=arm: gp_gates("D", "predicted_below_60", a),
            f"Gate D {arm} P(GP<60)")
        add(tail, GP_GATES, lambda a=arm: gp_gates("D", "tail_error", a),
            f"Gate D {arm} tail error")
    add("0.1003", GP_GATES, lambda: gp_gates("D", "observed_below_41", "hybrid"),
        "Gate D observed P(GP<41)")
    add("0.3259", GP_GATES, lambda: gp_gates("D", "observed_below_60", "hybrid"),
        "Gate D observed P(GP<60)")

    # ── the spell shape: the metric no marginal score can see ─────────────────
    shape = [("observed", "0.4924", "0.0581", "0.0079", "3.0857", None),
             ("hybrid", "0.4904", "0.0572", "0.0202", "3.8220", "0.5310"),
             ("calibrated_fallback", "0.1171", "0.3544", "0.0799", "10.0594", "5.0082"),
             ("full_window", "0.4556", "0.1018", "0.0409", "4.8658", "1.6746"),
             ("duration_covariates", "0.4471", "0.1063", "0.0416", "4.9746", "1.7373")]
    for arm, p1, p10, p26, mean, err in shape:
        for quoted, col in ((p1, "p_eq_1"), (p10, "p_ge_10"), (p26, "p_ge_26"),
                            (mean, "mean_spell")):
            add(quoted, GP_SHAPE,
                lambda a=arm, c=col: _one(table(GP_SHAPE), c, arm=a),
                f"spell shape {arm} {col}")
        if err:
            add(err, GP_SHAPE,
                lambda a=arm: _one(table(GP_SHAPE), "mean_abs_rel_error", arm=a),
                f"spell shape {arm} mean abs rel error")

    # ── the fallback's parameters ─────────────────────────────────────────────
    # `rho_markov` is historical because the row it read no longer exists, and the reason is
    # worth stating: `spell_process.csv` carries the *shipping* arm's parameters, and the
    # calibrated block is written only when the fallback is that arm. It was, on the test
    # split. On validation no arm ships and the file describes `duration_covariates`, so the
    # claim went from passing to skipping without anything drifting. Demote rather than
    # freeze the run — the same call the retired `subst_test` comment above describes — and
    # the figure stays presence-checked, so the retired block in the plan doc that carries it
    # cannot be tidied away. `clustering_C` stays live: the fallback rows are written
    # whichever arm ships, because the calibration is reported either way.
    add("0.8110", GP_PROCESS, lambda: float("nan"), "fallback rho_markov",
        historical=True)
    add("9.5806", GP_PROCESS, lambda: gp_process("fallback", "clustering_C"),
        "fallback clustering C")

    # ── the sampler ───────────────────────────────────────────────────────────
    add("13", GP_DIAG, lambda: float(len(table(GP_DIAG))), "fits in the sweep")
    return C


def _games_played_in_notes() -> list[Claim]:
    """The games-played block as `docs/model-development-notes.md` quotes it.

    A block quoted in two docs is claimed from both, because "current in one doc and stale
    in the other" is the failure that has already happened twice here. The notes carry a
    summary and `docs/games-played-plan.md` the full version, so this claims the subset the
    summary quotes rather than forcing it to carry every cell.
    """
    C: list[Claim] = []

    def add(quoted: str, artifact: str, actual: Callable[[], float],
            label: str, doc: str = NOTES) -> None:
        C.append(_c(quoted, artifact, actual, label, doc=doc))

    add("+5.29", GP_GATE, lambda: gp_gate_z("full_window_chain"),
        "plain chain z on the left tail")
    add("+0.93", GP_GATE, lambda: gp_gate_z("tenure_decomposition"),
        "tenure decomposition z")
    add("21.9%", GP_GATE,
        lambda: (gp_gate("full_window_chain", "p_below_41")
                 / gp_gate("full_window_chain", "p_below_41", "observed") - 1.0),
        "plain chain tail over-prediction")
    add("3.8%", GP_GATE,
        lambda: (gp_gate("tenure_decomposition", "p_below_41")
                 / gp_gate("tenure_decomposition", "p_below_41", "observed") - 1.0),
        "tenure decomposition tail error")
    add("7.23495", GP_METRICS, lambda: gp_metric("within_tenure", "val_crps"),
        "oracle-tenure CRPS")
    add("10.0992", GP_METRICS, lambda: gp_metric("within_tenure", "floor_val_crps"),
        "oracle-tenure floor")
    add("10.0057", GP_METRICS, lambda: gp_metric("floor", "val_crps"),
        "incumbent CRPS")
    add("10.1676", GP_METRICS, lambda: gp_metric("duration_covariates", "val_crps"),
        "best fitted arm CRPS")
    add("+0.1619", GP_METRICS,
        lambda: gp_metric("duration_covariates", "crps_vs_floor"),
        "best fitted arm vs floor")
    add("0.0406", GP_METRICS, lambda: gp_metric("floor", "val_tail_error"),
        "incumbent tail error")
    add("0.5310", GP_SHAPE,
        lambda: _one(table(GP_SHAPE), "mean_abs_rel_error", arm="hybrid"),
        "hybrid spell-shape error")
    add("1.7373", GP_SHAPE,
        lambda: _one(table(GP_SHAPE), "mean_abs_rel_error", arm="duration_covariates"),
        "arm A spell-shape error")
    add("5.0082", GP_SHAPE,
        lambda: _one(table(GP_SHAPE), "mean_abs_rel_error", arm="calibrated_fallback"),
        "fallback spell-shape error")
    add("9.5806", GP_PROCESS, lambda: gp_process("fallback", "clustering_C"),
        "matched clustering C")
    add("1,297,766", GP_COLLAPSE, lambda: gp_collapse("transitions"), "transitions")
    add("32,944", GP_COLLAPSE, lambda: gp_collapse("collapsed_binomial_rows"),
        "collapsed rows")
    add("39.4", GP_COLLAPSE, lambda: gp_collapse("collapse_ratio"), "collapse ratio")
    add("16.74%", GP_SPELLS,
        lambda: gp_spell("value", analysis="status_agreement",
                         statistic="missed_outside_window_while_rostered"),
        "rostered-but-absent share")
    add("82.7%", GP_SPELLS,
        lambda: gp_spell("value", analysis="status_agreement", statistic="agreement"),
        "structural proxy agreement")
    add("11,278", GP_SPELLS,
        lambda: (gp_spell("nll", model="geometric")
                 - gp_spell("nll", model="beta_geometric")),
        "beta-geometric over geometric")
    add("2.16", GP_SPELLS,
        lambda: (gp_spell("p_ge_26", treatment="proper_censoring")
                 / gp_spell("p_ge_26", treatment="drop_censored")),
        "censoring understatement")
    add("0.721", GP_SPELLS, lambda: gp_spell("a", treatment="proper_censoring"),
        "censored fit a")
    add("14.3", GP_SPELLS,
        lambda: (gp_spell("hazard", analysis="hazard_by_streak", streak="1")
                 / gp_spell("hazard", analysis="hazard_by_streak",
                            streak="42-10000")),
        "hazard fall across the streak range")
    return C


def _train_validate_test() -> list[Claim]:
    """`docs/train-validate-test-split.md` — the split discipline and what it reversed.

    This doc quotes the availability ladder as the record of the reversal that motivated
    the lock, so the block is claimed here as well as from the notes: "current in one doc
    and stale in the other" is the failure this file exists to catch, and a reversal
    narrative going stale is the worst version of it.

    Its remaining figures are deliberately unclaimed. The sampler-cost measurements
    (7.35 h against 9.78 h, 9,358 s against 9,615 s) were taken once during the
    conversion and have no artifact to re-derive them from, and the reversal margins it
    cites in passing are claimed from the docs that own those blocks.
    """
    C: list[Claim] = _availability_ladder_claims(
        SPLIT, scope="headline",
        historical=("10.795", "10.888", "10.896", "13.614"))
    C.append(_c("−0.771", STAN_C_S,
                lambda: cell(STAN_C_S, "reparam_minus_canonical", split="val",
                             arm="two_counts"),
                "substitution margin held across the chain-length change", doc=SPLIT))
    return C


def _tail_standard_error(metric: str) -> float:
    """`sqrt(p(1-p)/n)` on the rotation subpopulation — how well the Gate D *target* is
    known. Comparable in size to the differences between candidates, which is why the doc
    refuses to quote "closer to observed" as though the truth were known exactly."""
    frame = table(STAN_AV_M)
    if frame is None:
        return float("nan")
    p = _one(frame, "value", model="beta_binomial", group="rotation", metric=metric)
    n = _one(frame, "n", model="beta_binomial", group="rotation", metric=metric)
    return float(np.sqrt(p * (1 - p) / n))


# ── The weekly unit — Gate A one level below the season total ─────────────────

def _week(column: str, period_type: str = "week", split: str = "train") -> float:
    """One reading off `weekly_score_index.csv`, keyed the way the artifact is."""
    return _one(table(WEEK_INDEX), column, period_type=period_type, split=split)


def _week_extreme(column: str, largest: bool = True) -> float:
    """The widest or narrowest reading across the four facets.

    The doc quotes several of these as spans — "KS distances span 0.0265-0.0639" — because
    four facets of one comparison are read together and a span is what a reader takes from
    the table. Claiming both ends means a facet that moves in either direction fails.
    """
    frame = table(WEEK_INDEX)
    if frame is None:
        return float("nan")
    values = frame[column].astype(float)
    return float(values.max() if largest else values.min())


def _week_spread_ratio(largest: bool = True) -> float:
    """Simulated over observed sd, pooled over every row and draw, across the facets.

    Derived here rather than emitted because it is a ratio of two columns the artifact
    already carries, and the doc's claim is about the ratio. `pooled_sd` is deliberately
    the numerator: `point_sd` is narrower by construction and is the substitution this
    figure exists to avoid.
    """
    frame = table(WEEK_INDEX)
    if frame is None:
        return float("nan")
    ratio = frame["pooled_sd"].astype(float) / frame["observed_sd"].astype(float)
    return float(ratio.max() if largest else ratio.min())


def _week_line_gap(largest: bool = True) -> float:
    """How far a binned quartile line sits from its own level, worst per facet.

    The page computes this in `dashboard/weekly.py` from the same rows; re-derived here
    from the artifact so the doc's span is checked against the file rather than the view.
    """
    frame = table(WEEK_QUANTILE)
    if frame is None:
        return float("nan")
    lines = frame[frame["panel"] == "quantile"]
    gap = (lines["y"].astype(float) - lines["level"].astype(float)).abs()
    worst = gap.groupby([lines["period_type"], lines["split"]]).max()
    return float(worst.max() if largest else worst.min())


def _week_period_bias(slot: int, split: str = "validation") -> float:
    """Per-scoring-period bias, pooled across the split's seasons by row count.

    Row-weighted rather than a mean of means, which is what `weekly.profile_panel` draws
    and why each season's own `n` ships on the artifact.
    """
    frame = table(WEEK_PERIOD)
    if frame is None:
        return float("nan")
    part = frame[(frame["split"] == split) & (frame["slot"] == slot)]
    return float(np.average(part["bias"].astype(float),
                            weights=part["n"].astype(float)))


def _season_total_bias(largest: bool = True) -> float:
    """Gate A's own season-total bias, across the four simulated seasons."""
    frame = table(SIM_GATE_A)
    if frame is None:
        return float("nan")
    values = frame[frame["check"] == "season_total_dk"]["bias"].astype(float)
    return float(values.max() if largest else values.min())


def _weekly() -> list[Claim]:
    """`docs/simulations-plan.md`'s weekly Gate A row — the fifth bar, added 2026-08-10.

    Everything here is a *reading* rather than a gate, which is why it is claimed at all:
    the row counts, the season-total reconstruction and the draw budget are re-derived by
    `make weekly-scores` itself and fail the build, exactly as `docs/model-cards-plan.md`
    argues for its own figures. The metric table, the spreads and the per-period profile
    are checked by nothing else, and they are what a reader takes away.
    """
    facets = (("week", "train", "13,022", "52.74", "50.02", "29.93", "−2.72", "0.3985",
               "20.34"),
              ("week", "validation", "13,141", "53.40", "51.05", "28.72", "−2.35",
               "0.4658", "19.40"),
              ("double_week", "train", "2,298", "93.24", "88.43", "50.85", "−4.81",
               "0.4542", "34.41"),
              ("double_week", "validation", "2,319", "98.51", "95.52", "53.10", "−2.99",
               "0.4225", "36.03"))
    columns = ("n", "observed_mean", "predicted_mean", "mae", "bias", "r2", "crps")
    C: list[Claim] = []
    for period_type, split, *quoted in facets:
        for text, column in zip(quoted, columns):
            C.append(_c(text, WEEK_INDEX,
                        (lambda c=column, p=period_type, s=split: _week(c, p, s)),
                        f"weekly {period_type}/{split} {column}", doc=SIMS))

    C += [
        _c("30,780", WEEK_INDEX, lambda: total(WEEK_INDEX, "n"),
           "player-periods scored", doc=SIMS),
        # The spread, which is what a max over sixteen players is most sensitive to.
        _c("0.927", WEEK_INDEX, lambda: _week_spread_ratio(largest=False),
           "narrowest simulated/observed sd ratio", doc=SIMS),
        _c("0.969", WEEK_INDEX, lambda: _week_spread_ratio(largest=True),
           "widest simulated/observed sd ratio", doc=SIMS),
        _c("29.40", WEEK_INDEX, lambda: _week("point_sd"),
           "one-week train point-prediction sd", doc=SIMS),
        _c("49.05", WEEK_INDEX, lambda: _week("observed_sd"),
           "one-week train observed sd", doc=SIMS),
        # Zero weeks — the feature a season total averages away completely, and the row the
        # `tenure_merge` layout was aimed at. Quoted as percentages, and `check_values`
        # scales a `%` claim itself.
        _c("20.7%", WEEK_INDEX, lambda: _week("zero_share"),
           "one-week train observed zero share", doc=SIMS),
        _c("19.9%", WEEK_INDEX,
           lambda: _week("zero_share", split="validation"),
           "one-week validation observed zero share", doc=SIMS),
        _c("17.95%", WEEK_INDEX, lambda: _week("predicted_zero_share"),
           "one-week train simulated zero share", doc=SIMS),
        _c("19.07%", WEEK_INDEX,
           lambda: _week("predicted_zero_share", split="validation"),
           "one-week validation simulated zero share", doc=SIMS),
        # Calibration, read as a distance and never as a verdict.
        _c("0.0208", WEEK_INDEX, lambda: _week_extreme("ks", largest=False),
           "narrowest KS distance", doc=SIMS),
        _c("0.0582", WEEK_INDEX, lambda: _week_extreme("ks", largest=True),
           "widest KS distance", doc=SIMS),
        _c("0.1150", WEEK_QUANTILE, lambda: _week_line_gap(largest=False),
           "narrowest quantile-line gap", doc=SIMS),
        _c("0.1607", WEEK_QUANTILE, lambda: _week_line_gap(largest=True),
           "widest quantile-line gap", doc=SIMS),
        # The only bars in the target, and both are on the budget rather than the model.
        _c("0.0079", WEEK_INDEX, lambda: _week_extreme("ecdf_band_mc"),
           "worst ribbon half-sample disagreement", doc=SIMS),
        _c("0.0022", WEEK_INDEX, lambda: _week_extreme("ks_mc"),
           "worst KS half-sample disagreement", doc=SIMS),
        *[_c(quoted, WEEK_INDEX, lambda: float("nan"),
             f"pre-grading half-sample bar reading, {quoted}", doc=SIMS, historical=True)
          for quoted in ("0.0074", "0.0056")],
        # The pre-grading facet column, quoted in the note beside the live table for the
        # same reason the layout round quoted its own: the claim is the movement.
        *[_c(quoted, WEEK_INDEX, lambda: float("nan"),
             f"pre-grading weekly reading, {quoted}", doc=SIMS, historical=True)
          for quoted in ("−2.87", "−2.14", "−5.11", "−2.64", "20.35", "19.46", "34.47",
                         "36.11", "49.87", "51.26", "88.13", "95.87", "0.923", "0.971",
                         "0.0171", "0.0600", "0.1066", "0.1417", "19.00%",
                         "−1.88", "−3.03", "−2.28", "−1.60", "−1.19")],
        # The pre-`tenure_merge` readings, quoted in the prose beside the live ones because
        # the layout round's whole downstream claim is the movement rather than the level.
        # Presence-checked: an artifact holds one value per row, not its history.
        *[_c(quoted, WEEK_INDEX, lambda: float("nan"),
             f"pre-layout weekly reading, {quoted}", doc=SIMS, historical=True)
          for quoted in ("−2.93", "−2.26", "−3.61", "−1.21", "20.39", "19.63",
                         "16.9%", "18.2%", "0.920", "0.954", "0.0265", "0.0639")],
        # Where the season-total bias actually sits, week by week.
        _c("−2.31", WEEK_PERIOD, lambda: _week_period_bias(0),
           "validation bias in week 1", doc=SIMS),
        _c("−3.38", WEEK_PERIOD, lambda: _week_period_bias(1),
           "validation bias in week 2", doc=SIMS),
        _c("−2.55", WEEK_PERIOD, lambda: _week_period_bias(2),
           "validation bias in week 3", doc=SIMS),
        _c("−1.78", WEEK_PERIOD, lambda: _week_period_bias(12),
           "validation bias in week 13", doc=SIMS),
        _c("−1.31", WEEK_PERIOD, lambda: _week_period_bias(16),
           "validation bias in week 17", doc=SIMS),
        # The pre-`tenure_merge` profile, quoted beside the live one because the finding is
        # that the SHAPE went away — a flat −2 where there used to be a monotone ramp.
        *[_c(quoted, WEEK_PERIOD, lambda: float("nan"),
             f"pre-layout weekly bias profile, {quoted}", doc=SIMS, historical=True)
          for quoted in ("−5.28", "−4.99", "−3.27", "−1.08", "−0.70")],
        # Gate A's own season-total bias, so the weekly row is read against it.
        _c("−26.5", SIM_GATE_A, lambda: _season_total_bias(largest=True),
           "smallest season-total bias", doc=SIMS),
        # The same row's earlier readings, kept in the prose because the bullet's argument is
        # that a −69 dk_pts fault dwarfs everything measured on the head since. They are
        # presence-checked: an artifact holds one value per row, not its history.
        _c("−21.9", SIM_GATE_A, lambda: float("nan"),
           "season-total bias before the tenure_merge layout", doc=SIMS, historical=True),
        _c("−21.2", SIM_GATE_A, lambda: float("nan"),
           "season-total bias at the role-graded dispersion fix", doc=SIMS,
           historical=True),
        _c("−73.4", SIM_GATE_A, lambda: _season_total_bias(largest=False),
           "largest season-total bias", doc=SIMS),
    ]
    return C


def _minutes_window() -> list[Claim]:
    """`docs/minutes-window-plan.md` — the marginal minutes head's window x dispersion
    ladder, the era series rebuilt through each head's own rows, and the injection stake.

    Claimed at three scopes, because the doc's three sections fail differently. The era
    series is a **correction** of figures another doc quoted, so both the old and the new
    values are here — the rotation-filter row is live rather than historical, since
    `make minutes-window` re-derives §6's own population deliberately so the two reconcile
    in one table. The ladder and the harness are claimed as a matched pair on the axis the
    round turns on: a window margin that replicated and a dispersion margin that did not
    would be the same table with the opposite conclusion, so both intervals are audited.
    """
    C: list[Claim] = []

    def add(quoted: str, artifact: str, actual, label: str, **kw) -> None:
        C.append(_c(quoted, artifact, actual, label, doc=MWIN, **kw))

    def era(population: str, column: str, **where) -> float:
        return cell(MWIN_ERA, column, population=population, **where)

    def arm(rel: str, name: str, column: str) -> float:
        return cell(rel, column, arm=name)

    # ── §1: the two series, rebuilt through each head's own design rows ───────
    endpoints = [
        ("rotation_filter", "0.1918", "0.1626", "−15.2%", "0.1574", "0.0151",
         "10.4", "−7.4%"),
        ("minutes_head", "0.2004", "0.1824", "−9.0%", "0.1075", "0.0134",
         "8.0", "−6.2%"),
        ("composition_head", "0.2277", "0.2053", "−9.8%", "0.1134", "0.0105",
         "10.8", "−9.5%"),
    ]
    for pop, sd0, sd1, change, p0, p1, fold, mean in endpoints:
        for quoted, column in ((sd0, "sd_rate_first"), (sd1, "sd_rate_last"),
                               (change, "sd_rate_change"),
                               (p0, "p_workhorse_first"), (p1, "p_workhorse_last"),
                               (fold, "p_workhorse_fold"),
                               (mean, "mean_rate_change")):
            add(quoted, MWIN_ERA,
                lambda p=pop, c=column: era(p, c, block="endpoints"),
                f"{pop} endpoint {column}")

    # The pooled blocks, which are what "flat afterwards" turned out to be wrong about.
    for quoted, block, column in (("0.1942", "pre_2014_15", "sd_rate"),
                                  ("0.8851", "pre_2014_15", "sd_logit"),
                                  ("0.4902", "pre_2014_15", "mean_rate"),
                                  ("0.1027", "pre_2014_15", "p_workhorse"),
                                  ("0.1663", "post_2014_first_half", "sd_rate"),
                                  ("0.7410", "post_2014_first_half", "sd_logit"),
                                  ("0.4714", "post_2014_first_half", "mean_rate"),
                                  ("0.0154", "post_2014_first_half", "p_workhorse"),
                                  ("0.1706", "post_2014_second_half", "sd_rate"),
                                  ("0.7635", "post_2014_second_half", "sd_logit"),
                                  ("0.4739", "post_2014_second_half", "mean_rate"),
                                  ("0.0148", "post_2014_second_half", "p_workhorse"),
                                  ("0.0151", "post_2014_15", "p_workhorse")):
        add(quoted, MWIN_ERA,
            lambda b=block, c=column: era("minutes_head", c, block=b),
            f"minutes_head {block} {column}")

    # The reversion itself, as the two per-season endpoints of the rising run.
    add("0.1563", MWIN_ERA, lambda: era("minutes_head", "sd_rate", season="2019-20"),
        "minutes_head sd trough 2019-20")
    add("1.0789", MWIN_ERA,
        lambda: era("composition_head", "sd_logit", season="2023-24"),
        "composition_head sd_logit at a twelve-season high")

    # ── §1: the breakpoint scan ───────────────────────────────────────────────
    for pop, stat, f, shift in (("minutes_head", "mean_rate", "79.30", "−0.0187"),
                                ("minutes_head", "sd_rate", "107.15", "−0.0280"),
                                ("minutes_head", "p_workhorse", "203.07", "−0.0957"),
                                ("composition_head", "sd_rate", "119.90", "−0.0238"),
                                ("rotation_filter", "p_workhorse", "203.62", "−0.0952")):
        add(f, MWIN_BREAK,
            lambda p=pop, s=stat: cell(MWIN_BREAK, "sup_f", population=p, statistic=s),
            f"sup-F {pop} {stat}")
        add(shift, MWIN_BREAK,
            lambda p=pop, s=stat: cell(MWIN_BREAK, "shift", population=p, statistic=s),
            f"break shift {pop} {stat}")
    for pop, stat, p95 in (("minutes_head", "mean_rate", "9.28"),
                           ("minutes_head", "sd_rate", "8.84"),
                           ("minutes_head", "p_workhorse", "8.82"),
                           ("composition_head", "sd_rate", "9.41"),
                           ("rotation_filter", "p_workhorse", "9.04")):
        add(p95, MWIN_BREAK,
            lambda p=pop, s=stat: cell(MWIN_BREAK, "null_p95", population=p,
                                       statistic=s),
            f"MC null 95th {pop} {stat}")

    # The composition's own concentration series, which lives in the same artifact as a
    # fourth population — same question, same unit of time, different statistic.
    for quoted, season in (("0.1275", "1996-97"), ("0.1157", "2023-24"),
                           ("0.1131", "2018-19")):
        add(quoted, MWIN_ERA,
            lambda s=season: era("composition_team_game", "hhi", season=s),
            f"team-game HHI {season}")

    # ── §2: the ladder ────────────────────────────────────────────────────────
    ladder_arms = [
        ("carry_forward", "161.289", "+17.062", "+11.17", "+23.16", "0.1242", "330.84",
         "0.0623", None),
        ("full__shared", "144.228", None, None, None, "0.0737", "302.04", "0.0501",
         "1.00"),
        ("full__role", "142.555", "−1.672", "−2.55", "−0.80", "0.0652", "297.46",
         "0.0501", "2.12"),
        ("post_break__shared", "142.407", "−1.821", "−2.86", "−0.77", "0.0554",
         "284.24", "0.0442", "1.00"),
        ("post_break__role", "140.622", "−3.606", "−4.88", "−2.24", "0.0360", "275.06",
         "0.0442", "2.72"),
        ("three_point_era__shared", "141.836", "−2.392", "−3.63", "−1.08", "0.0587",
         "284.70", "0.0445", "1.00"),
        ("three_point_era__role", "139.554", "−4.674", "−6.16", "−3.11", "0.0420",
         "273.92", "0.0445", "2.92"),
        ("post_2014__shared", "141.540", "−2.687", "−4.19", "−1.14", "0.0537", "277.06",
         "0.0421", "1.00"),
        ("post_2014__role", "138.911", "−5.317", "−7.02", "−3.61", "0.0392", "265.67",
         "0.0421", "3.03"),
    ]
    for name, crps, delta, lo, hi, pit, sd, rho, spread in ladder_arms:
        add(crps, MWIN_LADDER, lambda n=name: arm(MWIN_LADDER, n, "val_crps"),
            f"ladder {name} CRPS")
        add(pit, MWIN_LADDER, lambda n=name: arm(MWIN_LADDER, n, "val_pit_ks"),
            f"ladder {name} PIT KS")
        add(sd, MWIN_LADDER, lambda n=name: arm(MWIN_LADDER, n, "predictive_sd"),
            f"ladder {name} predictive sd")
        add(rho, MWIN_LADDER, lambda n=name: arm(MWIN_LADDER, n, "rho"),
            f"ladder {name} rho")
        if spread is not None:
            add(spread, MWIN_LADDER,
                lambda n=name: arm(MWIN_LADDER, n, "rho_spread"),
                f"ladder {name} rho spread")
        if delta is not None:
            add(delta, MWIN_LADDER,
                lambda n=name: arm(MWIN_LADDER, n, "crps_vs_incumbent"),
                f"ladder {name} vs incumbent")
            add(lo, MWIN_LADDER,
                lambda n=name: arm(MWIN_LADDER, n, "crps_vs_incumbent_lo"),
                f"ladder {name} interval low")
            add(hi, MWIN_LADDER,
                lambda n=name: arm(MWIN_LADDER, n, "crps_vs_incumbent_hi"),
                f"ladder {name} interval high")

    # The graded dispersion, bucket by bucket — the round's headline, so every cell.
    for name, low, mid, high, star in (
            ("full__role", "0.06130", "0.06055", "0.05306", "0.02894"),
            ("post_break__role", "0.06203", "0.05288", "0.04481", "0.02283"),
            ("three_point_era__role", "0.06249", "0.05392", "0.04439", "0.02140"),
            ("post_2014__role", "0.05997", "0.05157", "0.04096", "0.01982")):
        for quoted, bucket in ((low, "<12 mpg"), (mid, "12-24"), (high, "24-30"),
                               (star, "30+ mpg")):
            add(quoted, MWIN_LADDER,
                lambda n=name, b=bucket: arm(MWIN_LADDER, n, f"rho_{b}"),
                f"{name} rho {bucket}")

    # The two costs the CRPS column hides.
    for name, bias, cover in (("full__shared", "−14.78", "0.5768"),
                              ("post_2014__role", "−22.56", "0.5283")):
        add(bias, MWIN_LADDER, lambda n=name: arm(MWIN_LADDER, n, "val_bias"),
            f"ladder {name} season-total bias")
        add(cover, MWIN_LADDER, lambda n=name: arm(MWIN_LADDER, n, "coverage_50"),
            f"ladder {name} realized 50% coverage")
    add("0.9609", MWIN_LADDER,
        lambda: arm(MWIN_LADDER, "full__shared", "coverage_95"),
        "ladder full__shared realized 95% coverage")
    add("0.9528", MWIN_LADDER,
        lambda: arm(MWIN_LADDER, "post_2014__role", "coverage_95"),
        "ladder post_2014__role realized 95% coverage")

    # The reference check — the point MLE against the shipped Stan head's own figures.
    add("144.352", MIN_UNIF,
        lambda: cell(MIN_UNIF, "crps_minutes", arm="minutes_head",
                     unit="season_total"),
        "shipped Stan head season-unit CRPS")
    add("302.75", MIN_UNIF,
        lambda: cell(MIN_UNIF, "predictive_sd", arm="minutes_head",
                     unit="season_total"),
        "shipped Stan head season-unit predictive sd")

    # ── §3: the rolling-origin confirmation ───────────────────────────────────
    rolling_arms = [
        ("8__role", "155.668", "−1.778", "−2.36", "−1.20", "0.0414", "0.0482"),
        ("12__role", "155.708", "−1.738", "−2.20", "−1.28", "0.0406", "0.0502"),
        ("all__role", "156.070", "−1.376", "−1.73", "−1.00", "0.0440", "0.0530"),
        ("5__role", "156.303", "−1.143", "−1.87", "−0.40", "0.0426", "0.0463"),
        ("12__shared", "157.030", "−0.416", "−0.78", "−0.07", "0.0379", "0.0502"),
        ("3__role", "157.109", "−0.337", "−1.29", "+0.57", "0.0418", "0.0452"),
        ("8__shared", "157.367", "−0.079", "−0.59", "+0.43", "0.0356", "0.0482"),
        ("all__shared", "157.446", None, None, None, "0.0456", "0.0530"),
        ("5__shared", "157.555", "+0.109", "−0.56", "+0.76", "0.0419", "0.0463"),
        ("3__shared", "158.439", "+0.993", "+0.14", "+1.83", "0.0419", "0.0452"),
    ]
    for name, crps, delta, lo, hi, pit, rho in rolling_arms:
        add(crps, MWIN_ROLL, lambda n=name: arm(MWIN_ROLL, n, "crps"),
            f"rolling {name} CRPS")
        add(pit, MWIN_ROLL, lambda n=name: arm(MWIN_ROLL, n, "pit_ks"),
            f"rolling {name} PIT KS")
        add(rho, MWIN_ROLL, lambda n=name: arm(MWIN_ROLL, n, "mean_rho"),
            f"rolling {name} mean rho")
        if delta is not None:
            add(delta, MWIN_ROLL, lambda n=name: arm(MWIN_ROLL, n, "crps_vs_all"),
                f"rolling {name} vs all")
            add(lo, MWIN_ROLL, lambda n=name: arm(MWIN_ROLL, n, "crps_vs_all_lo"),
                f"rolling {name} interval low")
            add(hi, MWIN_ROLL, lambda n=name: arm(MWIN_ROLL, n, "crps_vs_all_hi"),
                f"rolling {name} interval high")
    for name, fit_rows in (("8__role", "2,690"), ("12__role", "3,972"),
                           ("all__role", "5,839"), ("5__role", "1,699"),
                           ("3__role", "1,024")):
        add(fit_rows, MWIN_ROLL, lambda n=name: arm(MWIN_ROLL, n, "mean_fit_rows"),
            f"rolling {name} mean fit rows")
    add("4,517", MWIN_ROLL, lambda: max_of(MWIN_ROLL, "n_scored"),
        "rolling scored rows")

    # ── §4: the stake ─────────────────────────────────────────────────────────
    for name, crps, pit, sd in (
            ("minutes__full__shared", "144.228", "0.0737", "302.04"),
            ("minutes__post_break__shared", "142.407", "0.0554", "284.24"),
            ("minutes__three_point_era__shared", "141.836", "0.0587", "284.70"),
            ("minutes__post_2014__shared", "141.540", "0.0537", "277.06"),
            ("minutes__post_2014__role", "138.911", "0.0392", "265.67")):
        add(crps, MWIN_STAKE, lambda n=name: arm(MWIN_STAKE, n, "crps_minutes"),
            f"stake {name} CRPS")
        add(pit, MWIN_STAKE, lambda n=name: arm(MWIN_STAKE, n, "pit_ks"),
            f"stake {name} PIT KS")
        add(sd, MWIN_STAKE, lambda n=name: arm(MWIN_STAKE, n, "predictive_sd"),
            f"stake {name} predictive sd")
    for name, lo, hi in (("tie_band__full__shared", "0.200", "0.525"),
                         ("tie_band__post_break__shared", "0.250", "0.525"),
                         ("tie_band__three_point_era__shared", "0.250", "0.525"),
                         ("tie_band__post_2014__shared", "0.250", "0.525"),
                         ("tie_band__post_2014__role", "0.300", "0.450")):
        add(lo, MWIN_STAKE, lambda n=name: arm(MWIN_STAKE, n, "sigma"),
            f"{name} lower edge")
        add(hi, MWIN_STAKE, lambda n=name: arm(MWIN_STAKE, n, "sigma_band_hi"),
            f"{name} upper edge")
    add("0.050", MWIN_STAKE,
        lambda: arm(MWIN_STAKE, "tie_band__full__shared", "grid_step"),
        "stake grid step")
    # The composition arms the doc reads the marginal head's PIT against.
    add("0.0659", MIN_UNIF,
        lambda: cell(MIN_UNIF, "pit_ks", unit="ps_effect_sweep", sigma=0.45),
        "composition PIT at the shipped sigma")
    add("0.0808", MIN_UNIF,
        lambda: cell(MIN_UNIF, "pit_ks", unit="ps_effect_sweep", sigma=0.375),
        "composition PIT at the validation grid optimum")
    add("64.65", MIN_UNIF,
        lambda: cell(MIN_UNIF, "predictive_sd", arm="composition_sum",
                     unit="season_total"),
        "composition un-injected predictive sd")
    return C


def _availability_window() -> list[Claim]:
    """`docs/availability-window-plan.md` §7l — the mixture's PAIRED contest counterfactual.

    Only §7l is claimed. The doc's earlier sections are a nine-part ladder over window,
    season term, dispersion and likelihood, and bringing them in is a separate job; entering
    the doc into the registry at all is what makes that job incremental rather than a
    decision.

    The block is worth auditing tightly for one reason that is specific to it: **it is the
    section most likely to be re-run and half-updated.** Its figures come from two arms
    captured hours apart, and the failure it records — comparing against a baseline of
    unknown vintage — is precisely the failure a stale half of this table would reproduce.
    So both arms of every pair are claimed, not just the deltas: a refreshed `mixture`
    column beside a stale `single` one fails here rather than reading as a new result.

    The `adp` control is claimed at both ends for the same reason. It is the row the whole
    "measured null rather than underpowered null" reading rests on, and a control that
    silently stopped being a control would leave the conclusion standing on nothing.
    """
    C: list[Claim] = []

    def add(quoted: str, actual, label: str, **kw) -> None:
        C.append(_c(quoted, MIXVAL, actual, label, doc=AWIN, **kw))

    def mv(block: str, measure: str, key: str, column: str = "mixture") -> float:
        return cell(MIXVAL, column, block=block, measure=measure, key=key)

    # ── the board: the order the drafting layer actually consumes ─────────────
    for season, spearman, moved in (("2022-23", "0.9990", "3.1979"),
                                    ("2023-24", "0.9993", "2.9219")):
        add(spearman, lambda s=season: mv("board", "spearman_mean_total", s),
            f"board rank correlation between the arms, {season}")
        add(moved, lambda s=season: mv("board", "mean_abs_rank_move_drafted", s),
            f"mean |rank move| over the drafted picks, {season}")
    # A `%` quote is scaled by `check_values`, so these hand back the share itself.
    add("98%", lambda: mv("board", "top100_overlap", "2022-23"),
        "top-100 overlap between the arms, 2022-23")
    add("99%", lambda: mv("board", "top100_overlap", "2023-24"),
        "top-100 overlap between the arms, 2023-24")

    # ── the draw: the shape that DOES move, graded by role ───────────────────
    # The star bucket at both ends, because the claim is the SIZE of the iron-man
    # correction and a delta alone would survive both figures drifting together.
    add("0.1023", lambda: mv("draw", "p_iron_man_strict", "2022-23 30+ mpg", "single"),
        "star P(gp>=75), single-component arm")
    add("0.0834", lambda: mv("draw", "p_iron_man_strict", "2022-23 30+ mpg"),
        "star P(gp>=75), mixture arm")
    add("0.0385", lambda: mv("draw", "p_iron_man_strict", "2023-24 30+ mpg", "single"),
        "star P(gp>=75), single-component arm, 2023-24")
    add("0.0316", lambda: mv("draw", "p_iron_man_strict", "2023-24 30+ mpg"),
        "star P(gp>=75), mixture arm, 2023-24")
    # The q10 split is the finding, so both signs are claimed: one number moving would
    # leave "splits by role" true-looking with the split gone.
    add("−29.49", lambda: mv("draw", "q10_total", "2022-23 <12 mpg", "delta"),
        "fringe season-total q10 move")
    add("+46.40", lambda: mv("draw", "q10_total", "2022-23 30+ mpg", "delta"),
        "star season-total q10 move")
    add("+52.75", lambda: mv("draw", "q10_total", "2023-24 30+ mpg", "delta"),
        "star season-total q10 move, 2023-24")

    # ── the contest, at the reference strategy in BOTH arms ──────────────────
    for measure, single, mixture, label in (
            ("sim_lift", "0.1631", "0.1890", "600k simulated lift"),
            ("realized_lift", "0.1881", "0.1713", "600k realized lift")):
        add(single, lambda m=measure: mv("contest", m, "600k_shootaround", "single"),
            f"{label}, single-component arm")
        add(mixture, lambda m=measure: mv("contest", m, "600k_shootaround"),
            f"{label}, mixture arm")
    add("0.3541", lambda: mv("contest", "sim_lift", "88k_alley_oop", "single"),
        "88k simulated lift at the reference strategy, single-component arm")
    add("0.4359", lambda: mv("contest", "sim_lift", "88k_alley_oop"),
        "88k simulated lift, mixture arm")

    # ── the control, and the resolution the null is read against ─────────────
    add("+0.0097", lambda: mv("strategy", "lift_delta_mean", "600k_shootaround"),
        "mean lift delta over the 24 strategies")
    add("0.0161", lambda: mv("strategy", "lift_delta_sd", "600k_shootaround"),
        "lift delta sd over the 24 strategies")
    add("20", lambda: mv("strategy", "lift_delta_positive", "600k_shootaround"),
        "strategies whose lift moved up")
    add("0.0397", lambda: mv("strategy", "adp_only_lift", "600k_shootaround", "single"),
        "the ADP control's lift, single-component arm")
    add("0.0490", lambda: mv("strategy", "adp_only_lift", "600k_shootaround"),
        "the ADP control's lift, mixture arm")
    add("+0.0093", lambda: mv("strategy", "adp_only_lift", "600k_shootaround", "delta"),
        "THE CONTROL: a board identical across arms, moving anyway")
    add("+1.0091", lambda: mv("strategy", "reference_lift_z", "600k_shootaround"),
        "the shipped arm's delta in sds of the across-strategy spread")
    add("0.9174", lambda: mv("strategy", "ordering_spearman", "600k_shootaround"),
        "strategy ordering between the arms")
    add("0.0876", lambda: mv("resolution", "min_detectable_lift_gap", "mixture"),
        "the 95% resolution on an arm-to-arm lift gap")

    # ── the verdicts, which is where a re-run would show a real change ───────
    # Split into two claims because the registry parses a bare number: "0 of 6" is a
    # count and a denominator, and both have to be claimed or the failure can decay by
    # the denominator moving underneath a zero that stays true.
    add("0", lambda: mv("verdict", "gate_d_materially_different", "all tournaments"),
        "Gate D comparisons separating the tiers, both arms", tol=0.5)
    add("6", lambda: mv("verdict", "gate_d_comparisons", "all tournaments"),
        "Gate D paired comparisons behind that zero", tol=0.5)
    for season, single, mixture in (("2022-23", "0.4189", "0.4268"),
                                    ("2023-24", "0.4234", "0.4164")):
        add(single, lambda s=season: mv("verdict", "injection_rho", s, "single"),
            f"injection rho, single-component arm, {season}")
        add(mixture, lambda s=season: mv("verdict", "injection_rho", s),
            f"injection rho, mixture arm, {season}")
    return C


def _availability_regime() -> list[Claim]:
    """`docs/availability-window-plan.md` §10 — the regime axis and the shrinkage family.

    A null round, which changes what is worth claiming. There is no shipped figure to
    protect, so the claims here protect the two things a null can silently lose:

    **The controls.** `regime_placebo` is what turns "the trough hurts" into a measurement
    rather than a statement about row count, and `splice8__intercept_workload` is what makes
    the shrinkage table comparable to §5b at all — it reproduces that round's 9.8983 and
    −0.0359 on a fresh run, which is the provenance the whole §10c comparison stands on. A
    control that quietly stopped controlling would leave both sections reading the same and
    meaning nothing.

    **Both sides of every reversal.** §10c withdraws §5b's co-adaptation diagnosis on one
    matched pair, so the pair is claimed at both ends and so is the interval; §10d's verdict
    is that the selected arms *lose*, so the selected arms' validation rows are claimed with
    their intervals rather than only the shipped head's.

    The identification block is claimed because §10d's largest single figure — the lag arm's
    +0.1610 — is only interpretable beside it, and a reader who lost the 1.000 would be left
    with a feature that looks like it was tested and failed.
    """
    C: list[Claim] = []

    def add(quoted: str, artifact: str, actual, label: str, **kw) -> None:
        C.append(_c(quoted, artifact, actual, label, doc=AWIN, **kw))

    def reg(arm: str, column: str, experiment: str) -> float:
        return cell(AREG, column, arm=arm, experiment=experiment)

    def shr(arm: str, column: str, experiment: str = "shrinkage") -> float:
        return cell(ASHRINK, column, arm=arm, experiment=experiment)

    def conf(arm: str, column: str) -> float:
        return cell(ARCONF, column, arm=arm)

    # ── §10a: the blindness, and the sign reversal it produces ────────────────
    add("2", AREG, lambda: max_of(AREG, "origins_active"),
        "most origins any regime arm is active at", tol=0.5)
    for quoted, arm, column in (
            ("−0.0198", "lb8__dummy_lag", "crps_vs_lookback_none"),
            ("−0.0341", "lb8__dummy_lag", "lookback_none_lo"),
            ("−0.0046", "lb8__dummy_lag", "lookback_none_hi"),
            ("−0.0169", "lb8__dummy_both", "crps_vs_lookback_none"),
            ("+0.0018", "lb8__w0.50", "crps_vs_lookback_none"),
            ("+0.0072", "lb8__exclude", "crps_vs_lookback_none"),
            ("+0.0081", "lb8__dummy_target", "crps_vs_lookback_none"),
            ("+0.0036", "lb8__dummy_target", "lookback_none_lo"),
            ("+0.0124", "lb8__dummy_target", "lookback_none_hi")):
        add(quoted, AREG, lambda a=arm, c=column: reg(a, c, "regime_x_lookback"),
            f"{arm} {column}")

    # ── §10b: the contamination harness and its control ───────────────────────
    contam = [("clean", "10.0121", None, None, None, "−0.0060"),
              ("contaminated", "10.0307", "+0.0186", "−0.0067", "+0.0428", "−0.0166"),
              ("contaminated__dummy_target", "9.9951", "−0.0171", "−0.0325", "−0.0027",
               "−0.0026"),
              ("contaminated__dummy_both", "9.9968", "−0.0154", "−0.0311", "−0.0005",
               "−0.0023"),
              ("contaminated__dummy_lag", "9.9992", "−0.0129", "−0.0273", "+0.0001",
               "−0.0046"),
              ("contaminated__w0.25", "10.0132", "+0.0011", "−0.0074", "+0.0093",
               "−0.0096"),
              ("contaminated__w0.50", "10.0175", "+0.0054", "−0.0100", "+0.0202",
               "−0.0124")]
    for arm, crps, delta, lo, hi, bias in contam:
        add(crps, AREG, lambda a=arm: reg(a, "crps", "regime_contamination"),
            f"contamination CRPS, {arm}")
        add(bias, AREG, lambda a=arm: reg(a, "share_bias", "regime_contamination"),
            f"contamination predicted-share bias, {arm}")
        for quoted, column in ((delta, "crps_vs_reference"), (lo, "lo"), (hi, "hi")):
            if quoted is not None:
                add(quoted, AREG,
                    lambda a=arm, c=column: reg(a, c, "regime_contamination"),
                    f"contamination {column}, {arm}")
    add("8", AREG,
        lambda: reg("contaminated__dummy_target", "origins_won", "regime_contamination"),
        "origins the indicator wins", tol=0.5)
    # The core-block sensitivity: same sign, interval now covering zero.
    for quoted, column in (("−0.0125", "crps_vs_reference"), ("−0.0276", "lo"),
                           ("+0.0017", "hi")):
        add(quoted, AREG,
            lambda c=column: reg("contaminated__dummy_target", c,
                                 "regime_contamination_core"),
            f"core-block contamination {column}")
    # THE CONTROL. Both the regime block's cost against the placebo and the placebo's own
    # null against clean, because the reading is the pair.
    for quoted, arm, column in (("9.7426", "inject_regime", "crps"),
                                ("+0.0402", "inject_regime", "crps_vs_reference"),
                                ("+0.0163", "inject_regime", "lo"),
                                ("+0.0653", "inject_regime", "hi"),
                                ("9.7085", "clean", "crps"),
                                ("+0.0061", "clean", "crps_vs_reference"),
                                ("−0.0107", "clean", "lo"),
                                ("+0.0232", "clean", "hi"),
                                ("9.7024", "inject_placebo", "crps")):
        add(quoted, AREG,
            lambda a=arm, c=column: reg(a, c, "regime_placebo_vs_placebo"),
            f"placebo control {column}, {arm}")

    # ── §10c: the shrinkage ladder, and the §5b arms it reproduces ────────────
    add("9.9478", ASHRINK, lambda: shr("long_only", "crps"),
        "the untruncated fit — §5b's (all, all) reproduced")
    add("9.8983", ASHRINK, lambda: shr(SPLICE, "crps"),
        "§5b's block winner reproduced")
    add("9.9342", ASHRINK, lambda: shr("shrink8__laminf", "crps"),
        "§5b's `short__none` reference reproduced")
    add("−0.0359", ASHRINK,
        lambda: shr(SPLICE, "crps") - shr("shrink8__laminf", "crps"),
        "§5b's headline block gap, re-derived on one run")
    add("9.918054", ASHRINK, lambda: shr("shrink8__lam0", "crps"),
        "the lambda = 0 endpoint")
    add("9.917984", ASHRINK, lambda: shr("short8__only", "crps"),
        "the plain short-window fit it reproduces")
    for quoted, arm, column in (
            ("9.8939", "shrink5__free_drift__laminf", "crps"),
            ("−0.0539", "shrink5__free_drift__laminf", "crps_vs_reference"),
            ("−0.0750", "shrink5__free_drift__laminf", "lo"),
            ("−0.0322", "shrink5__free_drift__laminf", "hi"),
            ("10", "shrink5__free_drift__laminf", "origins_won"),
            ("9.8973", "shrink5__free_drift__lam1024", "crps"),
            ("−0.0496", SPLICE, "crps_vs_reference"),
            ("12", SPLICE, "origins_won"),
            ("9.9025", "shrink5__lam256", "crps"),
            ("9.9038", "shrink8__free_drift__laminf", "crps"),
            ("9.9180", "short8__only", "crps"),
            ("−0.0136", "shrink8__laminf", "crps_vs_reference")):
        add(quoted, ASHRINK, lambda a=arm, c=column: shr(a, c),
            f"shrinkage {column}, {arm}",
            **({"tol": 0.5} if column == "origins_won" else {}))
    # The lambda sweep at short = 8, quoted as a monotone run: the claim is that the knob
    # has an INTERIOR optimum, which a single endpoint could not carry.
    for quoted, lam in (("9.9181", "lam0"), ("9.9175", "lam1"), ("9.9164", "lam4"),
                        ("9.9145", "lam16"), ("9.9117", "lam64"), ("9.9088", "lam256"),
                        ("9.9102", "lam1024")):
        add(quoted, ASHRINK, lambda x=lam: shr(f"shrink8__{x}", "crps"),
            f"global shrinkage at short = 8, {lam}")
    # THE REVERSAL, at both ends and with its interval.
    for quoted, arm, column in (("+0.0055", "shrink8__free_drift__laminf",
                                 "crps_vs_reference"),
                                ("−0.0017", "shrink8__free_drift__laminf", "lo"),
                                ("+0.0131", "shrink8__free_drift__laminf", "hi"),
                                ("4", "shrink8__free_drift__laminf", "origins_won"),
                                ("−0.0044", "shrink5__free_drift__laminf",
                                 "crps_vs_reference"),
                                ("−0.0175", "shrink5__free_drift__laminf", "lo"),
                                ("+0.0081", "shrink5__free_drift__laminf", "hi")):
        add(quoted, ASHRINK,
            lambda a=arm, c=column: shr(a, c, "shrinkage_vs_splice"),
            f"joint fit against the splice, {column}, {arm}",
            **({"tol": 0.5} if column == "origins_won" else {}))

    # ── §10d: the one validation reading ──────────────────────────────────────
    val = [("shipped__2012_role_rho", "9.8247", None, None, None, "0.0588", "0.0178"),
           ("regime__w0.50", "9.8192", "−0.0056", "−0.0289", "+0.0195", "0.0692",
            "0.0198"),
           ("regime__w0.25", "9.8225", "−0.0023", "−0.0421", "+0.0402", "0.0758",
            "0.0212"),
           ("shrink__global__2012__lam256", "9.8363", "+0.0116", "−0.0053", "+0.0278",
            "0.0611", "0.0183"),
           ("regime__exclude", "9.8367", "+0.0119", "−0.0490", "+0.0780", "0.0830",
            "0.0231"),
           ("regime__dummy_target", "9.8423", "+0.0175", "−0.0326", "+0.0693", "0.0779",
            "0.0233"),
           ("regime__core_exclude", "9.8534", "+0.0286", "−0.0254", "+0.0894", "0.0822",
            "0.0223"),
           ("shrink__global__lb5__lam256", "9.9046", "+0.0798", "+0.0258", "+0.1304",
            "0.0759", "0.0143"),
           ("shrink__free_drift__2012__laminf", "9.9001", "+0.0753", "+0.0368", "+0.1139",
            "0.0605", "0.0180"),
           ("shrink__free_drift__lb5__laminf", "9.9393", "+0.1146", "+0.0444", "+0.1821",
            "0.0854", "0.0122"),
           ("regime__dummy_both", "9.9350", "+0.1103", "+0.0122", "+0.2095", "0.0799",
            "0.0133"),
           ("regime__dummy_lag", "9.9857", "+0.1610", "+0.0472", "+0.2807", "0.0833",
            "0.0113"),
           ("mixture__shipped_2012", "9.8237", None, None, None, "0.0631", "0.0108"),
           ("mixture__regime_exclude", "9.8709", "+0.0472", "−0.0221", "+0.1261",
            "0.0931", "0.0180"),
           ("mixture__regime_dummy_target", "9.8925", "+0.0688", "+0.0145", "+0.1258",
            "0.0886", "0.0161")]
    for arm, crps, delta, lo, hi, pit, boundary in val:
        add(crps, ARCONF, lambda a=arm: conf(a, "val_crps"), f"validation CRPS, {arm}")
        add(pit, ARCONF, lambda a=arm: conf(a, "val_pit_ks"), f"validation PIT KS, {arm}")
        add(boundary, ARCONF, lambda a=arm: conf(a, "boundary_tail_error"),
            f"validation boundary error, {arm}")
        for quoted, column in ((delta, "crps_vs_shipped"), (lo, "crps_vs_shipped_lo"),
                               (hi, "crps_vs_shipped_hi")):
            if quoted is not None:
                add(quoted, ARCONF, lambda a=arm, c=column: conf(a, c),
                    f"validation {column}, {arm}")

    # ── the identification block the lag arm's reading depends on ─────────────
    for quoted, column in (("1.000", "lag_implies_target_train"),
                           ("885", "n_regime_lag_train"),
                           ("1,285", "n_regime_target_train"),
                           ("433", "n_regime_lag_val"),
                           ("0", "n_regime_target_val")):
        add(quoted, ARCONF,
            lambda c=column: cell(ARCONF, c, arm="shipped__2012_role_rho"),
            f"regime identification, {column}",
            **({"tol": 0.5} if column == "n_regime_target_val" else {}))
    return C


def _availability_exchangeability() -> list[Claim]:
    """`docs/availability-window-plan.md` §11 — the exchangeable-trials assumption.

    The round has no fitted arm, so what needs protecting is unusual: the claims here are
    all *ratios between arms of the same ladder*, and every one of them is only meaningful
    because the three arms carry identical games played. That equality is asserted inside
    `layout_ladder` and pinned by `tests/test_availability_exchangeability.py` rather than
    claimed here, since it is a `0.0` and not a figure.

    Three groups, and each protects a different half of the verdict:

    **The decomposition** (§11b). `edge_share` is what makes §11d's residual a *named*
    mechanism rather than a shrug, and it is the number the `potential-to-dos.md` entry is
    built on. The spell-shape rows are claimed because §11e result 4 turns them into a
    standing licence to pool — a licence that would quietly expire if the shape gradient
    grew and nobody noticed.

    **The `gp` margin** (§11a). Every arm, including the two whose CRPS is a loss, because
    the argument is that *all five* agree; a table that lost its losing rows would read as
    cherry-picking. `hybrid`'s three columns are claimed against the floor's own values,
    which is the marginal-neutrality proof.

    **The period ladder** (§11c/§11d). Both ends of every gap and the recovered share, for
    §10c's reason: the verdict is "materially wrong AND already mostly paid for", so an
    audit that protected only the first half would let the second half rot.
    """
    C: list[Claim] = []

    def add(quoted: str, artifact: str, actual, label: str, **kw) -> None:
        C.append(_c(quoted, artifact, actual, label, doc=AWIN, **kw))

    def clu(analysis: str, population: str, column: str) -> float:
        return cell(ACLUST, column, analysis=analysis, population=population)

    def gpm(arm: str, column: str) -> float:
        return cell(ACLUST, column, analysis="gp_margin", arm=arm)

    def lay(population: str, arm: str, column: str) -> float:
        return cell(AEXCH, column, analysis="period_layout", population=population,
                    arm=arm)

    def gap(population: str, metric: str, column: str) -> float:
        return cell(AEXCH, column, analysis="period_gap", population=population,
                    metric=metric)

    # ── §11b: where the non-exchangeability comes from ────────────────────────
    # The `edge_share` rows are claimed alongside their two components deliberately: the
    # aggregate is nearly flat across role and the components are not, which is the finding,
    # and an audit that protected only the aggregate would let the split rot back into it.
    add("85,341", ACLUST, lambda: clu("missed_decomposition", "all", "missed_games"),
        "missed games on the fitting rows", tol=0.5)
    for quoted, population, column in (
            ("55.83%", "all", "interior_share"),
            ("44.17%", "all", "edge_share"),
            ("20.68%", "all", "edge_not_rostered_share"),
            ("23.50%", "all", "edge_still_rostered_share"),
            ("46.81%", "all", "not_rostered_share_of_edge"),
            ("0.88%", "all", "interior_not_rostered_share"),
            ("48.67%", "<12 mpg", "interior_share"),
            ("51.33%", "<12 mpg", "edge_share"),
            ("36.46%", "<12 mpg", "edge_not_rostered_share"),
            ("14.86%", "<12 mpg", "edge_still_rostered_share"),
            ("71.04%", "<12 mpg", "not_rostered_share_of_edge"),
            ("57.78%", "12-24", "interior_share"),
            ("22.41%", "12-24", "edge_not_rostered_share"),
            ("19.80%", "12-24", "edge_still_rostered_share"),
            ("53.09%", "12-24", "not_rostered_share_of_edge"),
            ("6.93%", "24-30", "edge_not_rostered_share"),
            ("35.29%", "24-30", "edge_still_rostered_share"),
            ("16.41%", "24-30", "not_rostered_share_of_edge"),
            ("60.48%", "30+ mpg", "interior_share"),
            ("39.52%", "30+ mpg", "edge_share"),
            ("2.75%", "30+ mpg", "edge_not_rostered_share"),
            ("36.77%", "30+ mpg", "edge_still_rostered_share"),
            ("6.95%", "30+ mpg", "not_rostered_share_of_edge")):
        add(quoted, ACLUST,
            lambda p=population, c=column: clu("missed_decomposition", p, c),
            f"{column} of missed games, {population}")
    add("22,211", ACLUST,
        lambda: clu("missed_decomposition", "<12 mpg", "missed_games"),
        "missed games, fringe bucket", tol=0.5)
    # §11b's target-convention note quotes four figures from a scratch measurement rather
    # than from an artifact — the multi-team denominator comparison has no `make` target
    # behind it, so it is presence-checked as `historical` rather than value-checked. The
    # rule in `docs/docs-audit.md` is that an unbacked figure is either claimed this way or
    # not quoted; silently leaving it unclaimed is the failure mode.
    for quoted in ("165.75", "0.3006", "0.6132", "0.6879", "0.050%"):
        add(quoted, ACLUST, lambda: float("nan"),
            f"multi-team target convention, {quoted} (scratch, no target)",
            historical=True)
    # The two multiples §11b leads on, each the ratio of a pair already claimed above.
    add("13.3×", ACLUST,
        lambda: (clu("missed_decomposition", "<12 mpg", "edge_not_rostered_share")
                 / clu("missed_decomposition", "30+ mpg", "edge_not_rostered_share")),
        "not-rostered edge share, fringe over star")
    add("2.5×", ACLUST,
        lambda: (clu("missed_decomposition", "30+ mpg", "edge_still_rostered_share")
                 / clu("missed_decomposition", "<12 mpg", "edge_still_rostered_share")),
        "still-rostered edge share, star over fringe")
    shape = [("all", "3.8587", "3.0660", "0.4906", "0.0551", "0.4871", "3.8703"),
             ("<12 mpg", "5.5839", "3.1844", "0.4530", "0.0571", "0.4577", "4.5722"),
             ("12-24", "4.4525", "2.8811", "0.4862", "0.0471", "0.4849", "4.5069"),
             ("24-30", "2.6325", "3.3177", "0.5116", "0.0652", "0.4971", "3.1097"),
             ("30+ mpg", "2.7621", "3.2261", "0.5367", "0.0669", "0.5291", "2.6223")]
    for population, rate, mean, p1, p10, mu, kappa in shape:
        for quoted, column in ((rate, "spells_per_season"), (mean, "mean_spell"),
                               (p1, "p_spell_1"), (p10, "p_spell_ge10"),
                               (mu, "bg_mu"), (kappa, "bg_kappa")):
            add(quoted, ACLUST,
                lambda p=population, c=column: clu("spell_shape", p, c),
                f"interior spell {column}, {population}")

    # ── §11a: the gp margin, every arm including the losses ───────────────────
    margin = [("full_window", "10.2797", "+0.2739", "0.0755", "0.0496"),
              ("three_state", "10.3466", "+0.2965", "0.1204", "0.0043"),
              ("duration_covariates", "10.1676", "+0.1619", "0.0672", "0.0535"),
              ("calibrated_fallback", "10.0207", "+0.0149", "0.1017", "0.0440"),
              ("hybrid", "10.0057", "0.0000", "0.0939", "0.0406")]
    for arm, crps, delta, pit, tail in margin:
        for quoted, column in ((crps, "val_crps"), (delta, "crps_vs_floor"),
                               (pit, "val_pit_ks"), (tail, "val_tail_error")):
            add(quoted, ACLUST, lambda a=arm, c=column: gpm(a, c),
                f"gp-margin {column}, {arm}")

    # ── §11c: the period ladder, both ends of every gap ───────────────────────
    ladder = [("all", "observed", "0.2447", "4.4634", "0.4607"),
              ("all", "clustered", "0.2273", "4.7862", "0.3555"),
              ("all", "exchangeable", "0.1509", "1.6862", "0.1729"),
              ("<12 mpg", "clustered", "0.4738", "10.7233", "0.6469"),
              ("<12 mpg", "observed", "0.4531", "8.0224", "0.6642"),
              ("24-30", "observed", "0.1749", "3.4643", "0.4643"),
              ("24-30", "clustered", "0.1253", "2.3189", "0.2351"),
              ("24-30", "exchangeable", "0.0637", "0.6909", "0.0540"),
              ("30+ mpg", "observed", "0.1202", "2.3218", "0.2816"),
              ("30+ mpg", "clustered", "0.0892", "1.7370", "0.1526"),
              ("30+ mpg", "exchangeable", "0.0361", "0.4602", "0.0308")]
    for population, arm, dead, run, run3 in ladder:
        for quoted, column in ((dead, "p_dead_period"), (run, "longest_dead_run"),
                               (run3, "p_dead_run")):
            add(quoted, AEXCH, lambda p=population, a=arm, c=column: lay(p, a, c),
                f"period {column}, {population} {arm}")
    add("751", AEXCH,
        lambda: lay("all", "observed", "player_seasons"),
        "single-team validation player-seasons in the period ladder", tol=0.5)

    # ── §11d: what the shipped layout recovers ────────────────────────────────
    for quoted, population, metric in (("81.5%", "all", "p_dead_period"),
                                       ("111.6%", "all", "longest_dead_run"),
                                       ("63.4%", "all", "p_dead_run"),
                                       ("123.9%", "<12 mpg", "p_dead_period"),
                                       ("168.0%", "<12 mpg", "longest_dead_run"),
                                       ("91.4%", "<12 mpg", "p_dead_run"),
                                       ("87.8%", "12-24", "p_dead_period"),
                                       ("117.0%", "12-24", "longest_dead_run"),
                                       ("74.8%", "12-24", "p_dead_run"),
                                       ("55.4%", "24-30", "p_dead_period"),
                                       ("58.7%", "24-30", "longest_dead_run"),
                                       ("44.2%", "24-30", "p_dead_run"),
                                       ("63.1%", "30+ mpg", "p_dead_period"),
                                       ("68.6%", "30+ mpg", "longest_dead_run"),
                                       ("48.6%", "30+ mpg", "p_dead_run")):
        add(quoted, AEXCH,
            lambda p=population, m=metric: gap(p, m, "recovered_share"),
            f"recovered share, {population} {metric}")
    # The three multiples §11c leads on, each derived from a pair already claimed above.
    for quoted, population, metric in (("2.65×", "all", "longest_dead_run"),
                                       ("3.3×", "30+ mpg", "p_dead_period"),
                                       ("5.0×", "30+ mpg", "longest_dead_run"),
                                       ("9.1×", "30+ mpg", "p_dead_run")):
        add(quoted, AEXCH,
            lambda p=population, m=metric: 1.0 / gap(p, m, "exchangeable_ratio"),
            f"exchangeable understatement, {population} {metric}")

    # ── §13: the tenure factor and the overflow policy ────────────────────────
    # The 2x2's four `recovered_share` columns are claimed for EVERY arm rather than only
    # the winner, because the section's argument is the interaction: `merge` alone is a
    # catastrophe and `tenure` alone moves the sign flip instead of closing it, and a table
    # that kept only `tenure_merge` would read as a single-factor result.
    def prof(population: str, bucket: str, column: str) -> float:
        return cell(ACLUST, column, analysis="edge_profile", population=population,
                    missed_share_bin=bucket)

    def over(population: str, arm: str) -> float:
        return cell(AEXCH, "overflow_rate", analysis="overflow_incidence",
                    population=population, arm=arm)

    def spell(population: str, arm: str, column: str) -> float:
        return cell(AEXCH, column, analysis="layout_spell_shape", population=population,
                    arm=arm)

    # §13a — what the edge fraction is conditional on, which is the conditioning choice
    for quoted, bucket, column in (
            ("0.1321", "0%-10%", "mean_edge_frac"), ("0.7075", "0%-10%", "p_no_edge"),
            ("0.0472", "0%-10%", "p_all_edge"),
            ("0.1581", "10%-25%", "mean_edge_frac"), ("0.4953", "10%-25%", "p_no_edge"),
            ("0.0165", "10%-25%", "p_all_edge"),
            ("0.2296", "25%-50%", "mean_edge_frac"), ("0.3338", "25%-50%", "p_no_edge"),
            ("0.0293", "25%-50%", "p_all_edge"),
            ("0.5679", "50%-101%", "mean_edge_frac"), ("0.0829", "50%-101%", "p_no_edge"),
            ("0.1434", "50%-101%", "p_all_edge")):
        add(quoted, ACLUST, lambda b=bucket, c=column: prof("all", b, c),
            f"edge profile {column}, missed share {bucket}")
    for quoted, bucket in (("848", "0%-10%"), ("850", "10%-25%"), ("716", "25%-50%"),
                           ("844", "50%-101%")):
        add(quoted, ACLUST, lambda b=bucket: prof("all", b, "player_seasons"),
            f"edge profile rows, missed share {bucket}", tol=0.5)
    add("3,258", ACLUST, lambda: prof("all", "all", "player_seasons"),
        "fitting player-seasons with a missed game", tol=0.5)
    # Role moves the END rather than the amount — the pair that says so, and its multiple.
    for quoted, population, column in (("0.2156", "<12 mpg", "mean_pre_frac"),
                                       ("0.0519", "30+ mpg", "mean_pre_frac"),
                                       ("0.1546", "<12 mpg", "mean_post_frac"),
                                       ("0.1936", "30+ mpg", "mean_post_frac")):
        add(quoted, ACLUST, lambda p=population, c=column: prof(p, "all", c),
            f"edge profile {column}, {population}")
    add("4.2×", ACLUST,
        lambda: prof("<12 mpg", "all", "mean_pre_frac") / prof("30+ mpg", "all",
                                                               "mean_pre_frac"),
        "leading-block share, fringe over star")

    # §13b — the overflow branch, and that it is role-graded rather than a guard
    for quoted, population, arm in (("14.74%", "all", "clustered"),
                                    ("41.28%", "<12 mpg", "clustered"),
                                    ("15.21%", "12-24", "clustered"),
                                    ("3.77%", "24-30", "clustered"),
                                    ("2.32%", "30+ mpg", "clustered"),
                                    ("7.99%", "all", "tenure"),
                                    ("23.82%", "<12 mpg", "tenure"),
                                    ("8.11%", "12-24", "tenure"),
                                    ("1.23%", "24-30", "tenure"),
                                    ("1.06%", "30+ mpg", "tenure")):
        add(quoted, AEXCH, lambda p=population, a=arm: over(p, a),
            f"overflow rate, {population} {arm}")
    add("17.8×", AEXCH,
        lambda: over("<12 mpg", "clustered") / over("30+ mpg", "clustered"),
        "overflow rate, fringe over star")

    # §13c — the 2x2, every arm on every arrangement-sensitive metric
    grid = [("all", "p_dead_period", "0.8146", "0.4735", "1.1043", "0.9740"),
            ("all", "longest_dead_run", "1.1162", "0.2707", "1.2124", "0.9612"),
            ("all", "p_dead_run", "0.6343", "0.5579", "0.9282", "0.9186"),
            ("<12 mpg", "p_dead_period", "1.2389", "0.2945", "1.5370", "1.1286"),
            ("<12 mpg", "longest_dead_run", "1.6804", "0.0073", "1.5027", "0.9656"),
            ("<12 mpg", "p_dead_run", "0.9138", "0.6865", "1.2793", "1.2125"),
            ("12-24", "p_dead_period", "0.8781", "0.5022", "1.1787", "1.0571"),
            ("12-24", "longest_dead_run", "1.1695", "0.2963", "1.2734", "1.0391"),
            ("12-24", "p_dead_run", "0.7482", "0.6393", "1.0104", "1.0095"),
            ("24-30", "p_dead_period", "0.5542", "0.4534", "0.8589", "0.8289"),
            ("24-30", "longest_dead_run", "0.5870", "0.3762", "0.8984", "0.8518"),
            ("24-30", "p_dead_run", "0.4415", "0.4143", "0.7047", "0.7047"),
            ("30+ mpg", "p_dead_period", "0.6306", "0.5804", "0.8766", "0.8428"),
            ("30+ mpg", "longest_dead_run", "0.6858", "0.5100", "0.9537", "0.8829"),
            ("30+ mpg", "p_dead_run", "0.4858", "0.5032", "0.8396", "0.8350")]
    for population, metric, shipped, merge, tenure, both in grid:
        for quoted, column in ((shipped, "recovered_share"),
                               (merge, "merge_recovered_share"),
                               (tenure, "tenure_recovered_share"),
                               (both, "tenure_merge_recovered_share")):
            add(quoted, AEXCH,
                lambda p=population, m=metric, c=column: gap(p, m, c),
                f"{column}, {population} {metric}")
    # §13d — the confirmation the arm was NOT selected on, so all five arms are claimed
    realized = [("observed", "6.5433", "4.6009", "0.1015", "0.0269"),
                ("clustered", "7.4263", "4.0539", "0.0562", "0.0224"),
                ("merge", "8.8215", "3.4127", "0.0676", "0.0060"),
                ("tenure", "6.1638", "4.8842", "0.0937", "0.0387"),
                ("tenure_merge", "6.6575", "4.5220", "0.0932", "0.0285")]
    for arm, rate, mean, p10, p30 in realized:
        for quoted, column in ((rate, "spells_per_season"), (mean, "mean_spell"),
                               (p10, "p_spell_ge10"), (p30, "p_spell_ge30")):
            add(quoted, AEXCH, lambda a=arm, c=column: spell("all", a, c),
                f"realized {column}, {arm}")
    return C



def _availability_absence() -> list[Claim]:
    """`docs/availability-window-plan.md` §12 — the absence block crossed with the compound.

    The round is mostly a **null**, and a null needs its losing rows protected more than a
    win does: the whole point of recording it is that nobody rebuilds the arm, and an audit
    that only guarded the surviving figure would let the reason rot away from under it.

    Four groups.

    **The block** (§12a). What the four named kinds actually cover, and the mask that
    removes nothing at the shipped window — asserted in the run rather than assumed, so it
    is claimed rather than printed.

    **The 2x2** (§12b). Every arm's CRPS, selector and body error, including `mixture`,
    which is carried as a context row precisely because it never trains on the block and
    reproduces §7c — a control that would stop meaning anything if it were left unclaimed.
    Both margin families, because the verdict is "real on CRPS, small on the boundary" and
    an audit protecting one half would let the other drift.

    **The profile** (§12c/§12d). `lambda`'s grid with `rho` beside it, because §11a's
    identification argument is the finding and `rho` is where it is visible. The
    pinned-duration row is claimed on *every* column: it holds the best `boundary_tail_error`
    in the document while being 314 log-likelihood points worse, and that pair is the sixth
    firing of §4's warning.
    """
    C: list[Claim] = []

    def add(quoted: str, artifact: str, actual, label: str, **kw) -> None:
        C.append(_c(quoted, artifact, actual, label, doc=AWIN, **kw))

    def arm(name: str, column: str) -> float:
        return cell(AABS, column, arm=name)

    def eff(metric: str, effect: str, column: str) -> float:
        return cell(AABS_I, column, metric=metric, effect=effect)

    def prof(name: str, column: str) -> float:
        return cell(AABS_L, column, arm=name)

    def blk(statistic: str) -> float:
        return cell(AABS_B, "value", statistic=statistic)

    # ── §12a: what the block is made of ───────────────────────────────────────
    add("99.24%", AABS_B, lambda: blk("kinds_share_of_missed"),
        "share of missed games in the four named kinds")
    add("4,027", AABS_B, lambda: blk("n_fitting_rows"),
        "fitting rows at the shipped window", tol=0.5)
    add("0.00%", AABS_B, lambda: blk("share_fitting_rows_masked"),
        "share of fitting rows removed by the status-coverage mask")

    # ── §12b: the 2x2, every arm including the two that are the reference twice ──
    ladder = [
        ("betabinom", "24", "−16,239.16", "9.8125", "0.0667", "0.0201", "0.0107", "0.0253"),
        ("betabinom__absence_mix", "28", "−16,220.27", "9.7515", "0.0588",
         "0.0184", "0.0136", "0.0236"),
        ("compound", "27", None, "9.8122", None, None, None, None),
        ("compound__absence_mix", "31", "−16,220.26", "9.7513", None, None, None, None),
        ("mixture", "35", "−16,174.52", "9.8237", "0.0631", None, "0.0047", "0.0235"),
    ]
    for name, params, ll, crps, pit, boundary, body, shoulder in ladder:
        add(params, AABS, lambda n=name: arm(n, "n_params"), f"{name} parameters", tol=0.5)
        add(crps, AABS, lambda n=name: arm(n, "val_crps"), f"{name} validation CRPS")
        for quoted, column in ((ll, "train_loglik"), (pit, "val_pit_ks"),
                               (boundary, "boundary_tail_error"), (body, "body_error"),
                               (shoulder, "shoulder_error")):
            if quoted is not None:
                add(quoted, AABS, lambda n=name, c=column: arm(n, c), f"{name} {column}")
    # `mixture`'s selector is quoted at §7c's rounding of the same 0.010850, which is what
    # makes the control readable beside that table; the tolerance is widened by one digit
    # rather than the figure being requoted to a different number for the same arm.
    add("0.0109", AABS, lambda: arm("mixture", "boundary_tail_error"),
        "mixture boundary error, the context row", tol=0.0001)

    # A margin and its two bounds are three claims, on this file's standing convention: the
    # interval is what makes the margin a finding rather than a prompt, so it cannot be
    # protected as a decoration on the point estimate.
    margins = [("betabinom__absence_mix", "crps_vs_betabinom",
                ("−0.0610", "−0.1148", "−0.0047"), "CRPS"),
               ("betabinom__absence_mix", "boundary_vs_betabinom",
                ("−0.00174", "−0.00240", "−0.00106"), "boundary"),
               ("compound", "crps_vs_betabinom",
                ("−0.00035", "−0.00109", "+0.00037"), "CRPS"),
               ("compound", "boundary_vs_betabinom",
                ("−0.0000036", "−0.0000110", "+0.0000036"), "boundary"),
               ("compound__absence_mix", "crps_vs_betabinom",
                ("−0.0613", "−0.1149", "−0.0050"), "CRPS"),
               ("compound__absence_mix", "boundary_vs_betabinom",
                ("−0.00173", "−0.00239", "−0.00105"), "boundary"),
               ("mixture", "crps_vs_betabinom",
                ("+0.0112", "−0.0280", "+0.0511"), "CRPS"),
               ("mixture", "boundary_vs_betabinom",
                ("−0.00891", "−0.00994", "−0.00420"), "boundary")]
    for name, column, (point, lo, hi), kind in margins:
        add(point, AABS, lambda n=name, c=column: arm(n, c), f"{name} {kind} margin")
        add(lo, AABS, lambda n=name, c=column: arm(n, f"{c}_lo"),
            f"{name} {kind} margin, lower bound")
        add(hi, AABS, lambda n=name, c=column: arm(n, f"{c}_hi"),
            f"{name} {kind} margin, upper bound")

    # The effects table, and the interaction it exists for. The interaction gets both bounds
    # because "clears zero and is two orders of magnitude below the main effect it is an
    # interaction with" is the whole reading, and half of it is the interval.
    for quoted, metric, effect, column in (
            ("−0.00177", "boundary_tail_error", "absence_mix | betabinom", "delta"),
            ("+0.00294", "body_error", "absence_mix | betabinom", "delta"),
            ("−0.00167", "shoulder_error", "absence_mix | betabinom", "delta"),
            ("+0.0000150", "boundary_tail_error", "interaction", "delta"),
            ("+0.0000092", "boundary_tail_error", "interaction", "delta_lo"),
            ("+0.0000200", "boundary_tail_error", "interaction", "delta_hi")):
        add(quoted, AABS_I, lambda m=metric, e=effect, c=column: eff(m, e, c),
            f"{metric} effect, {effect} ({column})")

    # The two shares §12b leads on, each a ratio of a pair already claimed above. They are
    # the sentence "real but small": the block closes a twentieth of what the shipped
    # mixture closes, and quoting either number without the other would be the finding.
    add("8.6%", AABS,
        lambda: (arm("betabinom__absence_mix", "boundary_vs_betabinom")
                 / -arm("betabinom", "boundary_tail_error")),
        "block's share of the reference boundary error")
    add("44.3%", AABS,
        lambda: (arm("mixture", "boundary_vs_betabinom")
                 / -arm("betabinom", "boundary_tail_error")),
        "mixture's share of the reference boundary error")
    # The multi-start spread, which is the reason the profile exists: §7b reads this column
    # the other way round on `mixture` and `finite_mix`, and 95 on an arm sitting on its own
    # bound is what makes "MLE" and "stuck optimizer" indistinguishable from the fit alone.
    add("95.18", AABS, lambda: arm("compound", "start_loglik_spread"),
        "compound's multi-start log-likelihood spread")
    add("0.003", AABS,
        lambda: abs(arm("compound", "train_loglik") - arm("betabinom", "train_loglik")),
        "compound's train log-likelihood against the reference", tol=0.0005)
    add("91%", AABS,
        lambda: (arm("betabinom__absence_mix", "boundary_tail_error")
                 / arm("betabinom", "boundary_tail_error")),
        "boundary error remaining after the block")
    add("18.9", AABS,
        lambda: (arm("betabinom__absence_mix", "train_loglik")
                 - arm("betabinom", "train_loglik")),
        "training log-likelihood the block buys")

    # ── §12c/§12d: the profile, and the row that is the trap ──────────────────
    profile = [("compound_lambda1", "1.000", "0.2586", "9.8125", "0.0201", "0.0107"),
               ("compound_lambda0.25", "1.042", "0.2320", "9.9366", "0.0274", "0.0282"),
               ("compound_lambda0.1", "1.170", "0.1922", "9.9198", "0.0229", "0.0297"),
               ("compound_lambda0", "1.238", "0.1772", "9.9144", "0.0214", "0.0302")]
    for name, spell, rho, crps, boundary, body in profile:
        for quoted, column in ((spell, "mean_spell"), (rho, "rho_weighted"),
                               (crps, "val_crps"), (boundary, "boundary_tail_error"),
                               (body, "body_error")):
            add(quoted, AABS_L, lambda n=name, c=column: prof(n, c),
                f"profile {name} {column}")
    add("−16,553.57", AABS_L,
        lambda: prof("compound_lambda0_pinned", "train_loglik"),
        "pinned-duration train log-likelihood")
    for quoted, column in (("3.129", "mean_spell"), ("0.0393", "rho_weighted"),
                           ("9.9018", "val_crps"), ("0.0846", "val_pit_ks"),
                           ("0.0087", "boundary_tail_error"),
                           ("0.0325", "body_error"), ("0.0478", "shoulder_error"),
                           ("0.0378", "point_mass_error"),
                           ("+0.0118", "err_below_10")):
        add(quoted, AABS_L,
            lambda c=column: prof("compound_lambda0_pinned", c),
            f"pinned-duration {column}")
    # ── §12e: the rolling confirmation, which is where the CRPS win goes ──────
    #
    # Both readings of the block are claimed, and the reversal is the reason. A registry
    # holding only the validation margin would let the doc keep saying "the second-largest
    # margin ever measured on this head" long after the harness that failed to reproduce it
    # had drifted.
    def roll(name: str, column: str) -> float:
        return cell(AABS_R, column, arm=name)

    add("2,871", AABS_R, lambda: roll("betabinom", "n_scored"),
        "rolling-harness scored rows", tol=0.5)
    rolling = [("betabinom", "10.1100", "0.0263", None, None, None),
               ("betabinom__absence_mix", "10.0995", "0.0259",
                ("−0.0105", "−0.0395", "+0.0183"),
                ("−0.000414", "−0.000767", "−0.000036"), "4 / 7"),
               ("compound", "10.1101", "0.0263",
                ("+0.000143", "+0.000023", "+0.000267"),
                ("+0.0000001", "−0.0000010", "+0.0000013"), None),
               ("compound__absence_mix", "10.0996", "0.0259",
                ("−0.0103", "−0.0394", "+0.0184"),
                ("−0.000404", "−0.000757", "−0.000027"), None),
               ("mixture", "10.1148", "0.0184",
                ("+0.0048", "−0.0195", "+0.0318"),
                ("−0.00786", "−0.00821", "−0.00752"), None)]
    for name, crps, boundary, crps_ci, boundary_ci, _won in rolling:
        add(crps, AABS_R, lambda n=name: roll(n, "crps"), f"rolling {name} CRPS")
        add(boundary, AABS_R, lambda n=name: roll(n, "boundary_tail_error"),
            f"rolling {name} boundary error")
        for interval, column, kind in ((crps_ci, "crps_vs_betabinom", "CRPS"),
                                       (boundary_ci, "boundary_vs_betabinom", "boundary")):
            if interval is None:
                continue
            point, lo, hi = interval
            add(point, AABS_R, lambda n=name, c=column: roll(n, c),
                f"rolling {name} {kind} margin")
            add(lo, AABS_R, lambda n=name, c=column: roll(n, f"{c}_lo"),
                f"rolling {name} {kind} margin, lower bound")
            add(hi, AABS_R, lambda n=name, c=column: roll(n, f"{c}_hi"),
                f"rolling {name} {kind} margin, upper bound")
    add("+0.00101", AABS_R,
        lambda: roll("betabinom__absence_mix", "shoulder_vs_betabinom"),
        "rolling block shoulder margin, the sign that flips")
    # The two multiples §12e leads on, each a ratio of a pair already claimed above.
    add("5.8×", AABS,
        lambda: (arm("betabinom__absence_mix", "crps_vs_betabinom")
                 / roll("betabinom__absence_mix", "crps_vs_betabinom")),
        "block CRPS margin, validation over rolling")
    add("19×", AABS_R,
        lambda: (roll("mixture", "boundary_vs_betabinom")
                 / roll("betabinom__absence_mix", "boundary_vs_betabinom")),
        "mixture over block, boundary margin on the rolling rows", tol=0.5)
    add("4.2×", AABS,
        lambda: (arm("betabinom__absence_mix", "boundary_vs_betabinom")
                 / roll("betabinom__absence_mix", "boundary_vs_betabinom")),
        "block boundary margin, validation over rolling")

    add("15.2%", AABS_L,
        lambda: prof("compound_lambda0_pinned", "rho_vs_nesting"),
        "pinned-duration rho against the nesting row")
    # Both halves of the D1 verdict, as intervals. The point estimates alone would read as
    # "better boundary, worse CRPS" — the intervals are what turn that into a failure on
    # both counts, and they are the reason the row is recorded rather than pursued.
    for quoted, column in (("+0.0892", "crps_vs_nesting"),
                           ("+0.0028", "crps_vs_nesting_lo"),
                           ("+0.1817", "crps_vs_nesting_hi"),
                           ("−0.0108", "boundary_vs_betabinom"),
                           ("−0.0232", "boundary_vs_betabinom_lo"),
                           ("+0.0057", "boundary_vs_betabinom_hi")):
        add(quoted, AABS_L, lambda c=column: prof("compound_lambda0_pinned", c),
            f"pinned-duration {column}")
    add("31.5%", AABS_L,
        lambda: 1.0 - prof("compound_lambda0", "rho_vs_nesting"),
        "rho collapse from the corner to lambda = 0")
    return C


def _availability_no_design_level() -> list[Claim]:
    """`docs/availability-window-plan.md` §8b — the no-design availability LEVEL.

    Three groups, and the reason each is here rather than left to prose.

    **The ladder** — every arm on both splits, not only the winner. The incumbent's
    validation R² of −0.0865 is the sharpest statement in the section (one rate is worse
    than that population's own mean) and it belongs to a *losing* arm, so an audit that
    guarded only `tenure_draft` would let it rot. `pooled`'s CRPS is claimed for the same
    reason the comparator row is claimed in the composition round: it is the control.

    **Both margin families, with their bounds.** Against the incumbent *and* against the
    runner-up, because the arms are nested keys and "the extra key earns its place" is a
    separate finding from "grading beats not grading" — §8b would still be worth shipping if
    the first held and the second did not, and it would be a different arm.

    **The mechanism table** (the rookie/returning rates by bucket) is NOT claimed here: it is
    a point-in-time cut of the pooling window quoted in prose, and the estimator that
    produces it is exercised by `tests/test_availability_no_prior.py` rather than persisted
    per cell. What the audit protects is the scored consequence of it.
    """
    C: list[Claim] = []

    def add(quoted: str, artifact: str, actual, label: str, **kw) -> None:
        C.append(_c(quoted, artifact, actual, label, doc=AWIN, **kw))

    def lvl(name: str, split: str, column: str) -> float:
        return cell(ANOP_L, column, analysis="level_arm", arm=name, split=split)

    def pair(name: str, reference: str, split: str, column: str) -> float:
        return cell(ANOP_L, column, analysis="level_pairwise", arm=name,
                    reference=reference, split=split)

    ladder = [("pooled", "14.4551", "22.7988", "−0.0865", "1.0073", "0.3531"),
              ("draft", "10.7612", "16.8625", "0.3269", "2.9957", "0.2399"),
              ("tenure", "14.1700", "22.2125", "−0.0549", "1.4973", "0.3512"),
              ("tenure_draft", "9.8689", "15.6068", "0.4316", "3.2529", "0.2087")]
    for name, crps, mae, r2, spread, rho in ladder:
        for quoted, column in ((crps, "crps"), (mae, "mae"), (r2, "r2_gp_share"),
                               (spread, "mu_spread"), (rho, "rho_residual")):
            add(quoted, ANOP_L, lambda n=name, c=column: lvl(n, "validation", c),
                f"validation {name} {column}")
    add("228", ANOP_L, lambda: lvl("pooled", "validation", "rows"),
        "validation rows the level arms are scored on", tol=0.5)
    add("2,529", ANOP_L, lambda: lvl("pooled", "rolling", "rows"),
        "rolling rows the level arms are scored on", tol=0.5)
    add("26", ANOP_L, lambda: lvl("pooled", "rolling", "n_seasons"),
        "rolling origins", tol=0.5)
    for quoted, name in (("15.0576", "pooled"), ("12.3164", "tenure_draft")):
        add(quoted, ANOP_L, lambda n=name: lvl(n, "rolling", "crps"),
            f"rolling {name} CRPS")
    add("0.3480", ANOP_L, lambda: lvl("tenure_draft", "rolling", "rho_residual"),
        "rolling residual rho under the shipped arm")

    # A margin and its two bounds are three claims, on this file's standing convention.
    margins = [("tenure_draft", "pooled", "validation",
                ("−4.5862", "−5.6200", "−3.5764")),
               ("tenure_draft", "pooled", "rolling",
                ("−2.7413", "−3.0221", "−2.4488")),
               ("tenure_draft", "draft", "validation",
                ("−0.8923", "−1.6575", "−0.1332")),
               ("tenure_draft", "draft", "rolling",
                ("−0.3713", "−0.5562", "−0.1928")),
               ("draft", "pooled", "validation",
                ("−3.6939", "−4.7499", "−2.6323")),
               ("tenure", "pooled", "validation",
                ("−0.2851", "−0.6803", "+0.1314"))]
    for name, reference, split, (point, lo, hi) in margins:
        for quoted, column in ((point, "crps_delta"), (lo, "crps_delta_lo"),
                               (hi, "crps_delta_hi")):
            add(quoted, ANOP_L,
                lambda n=name, r=reference, s=split, c=column: pair(n, r, s, c),
                f"{split} {name} against {reference} ({column})")
    # The origin counts, which are a different claim from the interval and the one the doc
    # leans on for "not one season carrying twenty-five". 18 of 26 is quoted precisely
    # because it is the WEAKER half and the prose says so.
    for quoted, reference in (("24", "pooled"), ("18", "draft")):
        add(quoted, ANOP_L,
            lambda r=reference: pair("tenure_draft", r, "rolling", "origins_won"),
            f"origins the shipped arm wins against {reference}", tol=0.5)
    add("26", ANOP_L,
        lambda: pair("tenure_draft", "pooled", "rolling", "origins"),
        "origins the level arms are compared at", tol=0.5)
    # `graded_share` on both splits. The validation one is the robustness claim — 1.000 means
    # the shipped decision does not depend on where `MIN_CELL` was put — and a claim is the
    # only thing that keeps it true after the pooling window next grows.
    add("10.7%", ANOP_L,
        lambda: 1.0 - lvl("tenure_draft", "rolling", "graded_share"),
        "rolling rows taking a coarser key than the shipped one")
    add("1.000", ANOP_L, lambda: lvl("tenure_draft", "validation", "graded_share"),
        "validation rows taking the shipped arm's own key")

    # The headline share, which is a ratio of two rows already claimed above. Quoted because
    # "31.7% of the incumbent's CRPS" is the sentence a reader takes away, and a ratio that
    # nothing derives is exactly the figure that survives its own numerator being refreshed.
    add("31.7%", ANOP_L,
        lambda: (lvl("pooled", "validation", "crps")
                 - lvl("tenure_draft", "validation", "crps"))
        / lvl("pooled", "validation", "crps"),
        "share of the incumbent's CRPS the shipped arm removes")

    # ── the simulator's readout, from Gate A's own artifact ──────────────────
    #
    # Only the SHIPPED column is claimed against a value. The `pooled` column is a
    # counterfactual run rather than a superseded reading — reproducible by setting
    # `sim.availability.no_design_level: pooled` and re-running `make simulate-season`, the
    # same status the layout arms' comparison columns carry — so it is presence-checked.
    def gate(season: str, check: str, column: str) -> float:
        return cell(SIM_GATE_A, column, season=season, check=check)

    for season, mae, crps, r2, bias in (
            ("2022-23", "397.36", "276.48", "0.6589", "−26.50"),
            ("2023-24", "398.45", "275.17", "0.6732", "−71.15")):
        for quoted, column in ((mae, "mae"), (crps, "crps"), (r2, "r2"), (bias, "bias")):
            add(quoted, SIM_GATE_A,
                lambda s=season, c=column: gate(s, "season_total_dk", c),
                f"{season} season-total {column} at the graded level")
    for season, share, realized, error in (("2022-23", "0.1015", "0.1057", "0.0294"),
                                           ("2023-24", "0.1079", "0.0992", "0.0472")):
        add(share, SIM_GATE_A,
            lambda s=season: gate(s, "no_design_team_minutes_share", "value"),
            f"{season} simulated no-design league minutes share")
        # The bar the per-team row is read against, so "right on average and wrong on all
        # thirty rosters" cannot drift into a claim about a number that has moved.
        add(realized, SIM_GATE_A,
            lambda s=season: gate(s, "no_design_team_minutes_share", "bar_value"),
            f"{season} realized no-design league minutes share")
        add(error, SIM_GATE_A,
            lambda s=season: gate(s, "no_design_team_minutes_share", "mae"),
            f"{season} per-team no-design minutes share error")
    for quoted in ("400.55", "400.00", "278.67", "276.41", "0.6516", "0.6721", "−22.89",
                   "−64.13", "9.5291", "9.5517", "0.0355", "0.0563", "0.0984", "0.1023"):
        C.append(_c(quoted, SIM_GATE_A, lambda: float("nan"),
                    f"pooled-scalar counterfactual reading, {quoted}", doc=AWIN,
                    historical=True))
    return C


def _build() -> tuple[Claim, ...]:
    """Every claim, in doc order. One builder per doc — the registry is long enough that
    a single function made it hard to see which doc a section belonged to.

    `_established_facts` is the one exception: the section it claims was split across five
    docs by subject in the 2026-08-08 reorganization, so it names a destination per section
    rather than taking one for the whole builder."""
    return tuple(_availability() + _composition() + _predictions() + _adp()
                 + _established_facts() + _readme() + _shot_basis() + _games_played()
                 + _games_played_in_notes() + _train_validate_test() + _weekly()
                 + _minutes_window() + _availability_window()
                 + _availability_regime() + _availability_exchangeability()
                 + _availability_absence() + _availability_no_design_level())


CLAIMS: tuple[Claim, ...] = _build()


# ── Checks ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Finding:
    check: str
    doc: str
    label: str
    detail: str


def doc_text(doc: str, root: Path = ROOT) -> str:
    path = root / doc
    return path.read_text() if path.exists() else ""


def check_values(claims: tuple[Claim, ...] = CLAIMS,
                 root: Path = ROOT) -> tuple[list[Finding], list[Finding]]:
    """Quoted figure against artifact. Returns (mismatches, skipped).

    Two kinds of claim are presence-checked rather than value-checked, and they are
    reported separately because they mean different things. `historical` claims are
    *superseded* — a corrected value kept beside its correction — and they are in the
    registry so a reversal cannot be tidied away. **Cost** claims are figures derived from
    a `COST_COLUMNS` column: sampler wall clock does not reproduce across machines or
    across whatever else is running, so comparing it to a stored constant fails for
    reasons that carry no information about the model. Both still have to appear in the
    doc.
    """
    bad, skipped = [], []
    for claim in claims:
        if claim.historical:
            skipped.append(Finding("superseded", claim.doc, claim.label,
                                   f"{claim.quoted} kept as the superseded value"))
            continue
        if not (root / claim.artifact).exists():
            skipped.append(Finding("missing-artifact", claim.doc, claim.label,
                                   f"`{claim.artifact}` not built"))
            continue
        _columns_read.clear()
        try:
            actual = float(claim.actual())
        except MissingColumn as exc:
            bad.append(Finding(
                "schema-change", claim.doc, claim.label,
                f"`{claim.artifact}` has no column {exc.args[0]!r} — the artifact's "
                f"schema moved under this claim, so it is checking nothing. Requote "
                f"from the column that replaced it and keep the old value as "
                f"Claim(historical=True)."))
            continue
        # Checked AFTER `actual()` rather than declared per claim, so the rule keys on
        # the artifact column that was actually read. A `MissingColumn` on a cost artifact
        # still fails loudly above: a schema change is a real defect whatever the column
        # measures.
        cost = touched_cost_column()
        if cost:
            skipped.append(Finding(
                "cost-figure", claim.doc, claim.label,
                f"{claim.quoted} presence-checked only — derived from "
                f"{', '.join(sorted(cost))}, which measures the machine rather than the "
                f"model (artifact says {actual:.6g})"))
            continue
        if actual != actual:                      # NaN — the lookup found no row
            skipped.append(Finding("no-such-row", claim.doc, claim.label,
                                   f"no row in `{claim.artifact}`"))
            continue
        if is_percent(claim.quoted):
            actual *= 100.0
        if abs(claim.value() - actual) > claim.tolerance():
            bad.append(Finding(
                "value-mismatch", claim.doc, claim.label,
                f"doc says {claim.quoted}, `{claim.artifact}` says "
                f"{actual:.6g} (tolerance ±{claim.tolerance():g})"))
    return bad, skipped


def check_presence(claims: tuple[Claim, ...] = CLAIMS,
                   root: Path = ROOT) -> list[Finding]:
    """The quoted string still appears in the doc.

    Without this the registry rots the moment somebody edits the prose — a claim that
    silently stops describing anything is exactly the drift this module exists to catch,
    one level up.
    """
    out = []
    cache: dict[str, str] = {}
    for claim in claims:
        if claim.doc not in cache:
            cache[claim.doc] = doc_text(claim.doc, root)
        if claim.quoted not in cache[claim.doc]:
            out.append(Finding("stale-claim", claim.doc, claim.label,
                               f"{claim.quoted!r} no longer appears in the doc"))
    return out


# A leading sign only counts when it is not itself part of a range or a hyphenated word,
# so `12–24` and `10-day` do not read as negative numbers.
_NUMBER = re.compile(r"(?<![\w.\-−–])[-−]?\d[\d,]*(?:\.\d+)?%?")


def numeric_literals(text: str) -> list[str]:
    """Every number in the doc that could plausibly be a measurement.

    Season labels (`2024-25`), ISO dates and bare years are dropped, along with fenced
    code and inline code spans. What is left is the denominator for coverage.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", " ", text)  # ISO dates
    text = re.sub(r"\b(19|20)\d{2}-\d{2}\b", " ", text)      # season labels
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)            # bare years
    text = re.sub(r"`[^`]*`", " ", text)                     # inline code
    return [m.group(0) for m in _NUMBER.finditer(text)]


def is_measurement(literal: str) -> bool:
    """Whether a literal looks like a measured quantity rather than prose counting.

    A decimal, a percentage, or a number large enough to carry a thousands separator.
    `"0.483"`, `"26.7%"` and `"12,406"` qualify; `"two"`, `"30"` seasons and `"82"`
    games do not. This is the population that actually drifts — every figure corrected
    in the 2026-07-30 sweep was one of these — so it is the honest denominator to
    report coverage against.
    """
    return "." in literal or "%" in literal or "," in literal


def coverage(doc: str, claims: tuple[Claim, ...] = CLAIMS,
             root: Path = ROOT) -> dict:
    """How much of a doc's numeric content any claim covers.

    Two denominators, because they answer different questions: `measurements` is the
    at-risk population and the number to drive to 100%, while `numbers` is every
    literal in the prose and will never get there — nor should it.
    """
    text = doc_text(doc, root)
    literals = numeric_literals(text)
    measured = [n for n in literals if is_measurement(n)]
    quoted = {c.quoted for c in claims if c.doc == doc}
    return {
        "doc": doc,
        "claims": sum(1 for c in claims if c.doc == doc),
        "historical": sum(1 for c in claims if c.doc == doc and c.historical),
        "distinct": len(quoted),
        "numbers": len(literals),
        "measurements": len(measured),
        "covered": sum(1 for n in measured if n in quoted),
        "share": (sum(1 for n in measured if n in quoted) / len(measured)
                  if measured else 0.0),
        "uncovered": sorted({n for n in measured if n not in quoted}),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def run(root: Path = ROOT) -> dict:
    bad, skipped = check_values(root=root)
    return {"value-mismatch": bad, "stale-claim": check_presence(root=root),
            "skipped": skipped,
            "coverage": [coverage(d, root=root)
                         for d in sorted({c.doc for c in CLAIMS})]}


def report(results: dict) -> str:
    lines = ["Docs audit", "=" * 70, ""]
    for cov in results["coverage"]:
        lines.append(f"{cov['doc']}")
        hist = (f", {cov['historical']} of them superseded values held for the record"
                if cov["historical"] else "")
        lines.append(f"  {cov['claims']} claims covering "
                     f"{cov['covered']}/{cov['measurements']} measured figures "
                     f"({cov['share']:.0%}), out of {cov['numbers']} numeric "
                     f"literals in the prose{hist}")
    lines.append("")

    for check, heading in [("value-mismatch", "Figures that disagree with their artifact"),
                           ("stale-claim", "Claims whose text is no longer in the doc")]:
        found = results[check]
        lines.append(f"{heading}: {len(found)}")
        for f in found:
            lines.append(f"  - [{f.doc}] {f.label}: {f.detail}")
        lines.append("")

    unbuilt = [f for f in results["skipped"]
               if f.check not in ("superseded", "cost-figure")]
    if unbuilt:
        lines.append(f"Skipped (artifact not built): {len(unbuilt)}")
        for f in unbuilt[:10]:
            lines.append(f"  - {f.label}: {f.detail}")
        if len(unbuilt) > 10:
            lines.append(f"  … and {len(unbuilt) - 10} more")
        lines.append("")

    superseded = [f for f in results["skipped"] if f.check == "superseded"]
    cost = [f for f in results["skipped"] if f.check == "cost-figure"]
    checked = len(CLAIMS) - len(results["skipped"])
    lines.append("-" * 70)
    lines.append(f"figures checked:  {checked} of {len(CLAIMS)}")
    lines.append(f"superseded:       {len(superseded)} (presence-checked, not value-checked)")
    lines.append(f"cost figures:     {len(cost)} (presence-checked — wall clock measures "
                 f"the machine)")
    lines.append(f"disagreements:    {len(results['value-mismatch'])}")
    lines.append(f"stale claims:     {len(results['stale-claim'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run()
    print(report(results))
    # A disagreement between a doc and its artifact is an unambiguous defect, so this
    # one exits non-zero — unlike `make dashboard-audit`, which reports registry drift
    # that is often just somebody editing a doc.
    sys.exit(1 if (results["value-mismatch"] or results["stale-claim"]) else 0)
