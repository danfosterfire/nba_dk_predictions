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

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

AVAIL = "docs/availability-plan.md"
COMP = "docs/minutes-composition-plan.md"
PRED = "docs/predictions-plan.md"
ADP = "docs/adp-plan.md"
CLAUDE = "CLAUDE.md"
SHOT = "docs/shot-attempt-basis-plan.md"
README = "README.md"

PROFILE = "outputs/eda/availability_profile.csv"
METRICS = "outputs/predictions/availability_metrics.csv"
ABLATION = "outputs/predictions/availability_workload_ablation.csv"
NONLIN = "outputs/predictions/availability_nonlinearity.csv"
MIN_NONLIN = "outputs/predictions/availability_minutes_nonlinearity.csv"
STAN_MIN_M = "outputs/predictions/stan_minutes_metrics.csv"
STAN_MIN_D = "outputs/predictions/stan_minutes_dispersion.csv"
STAN_MIN_G = "outputs/predictions/stan_minutes_diagnostics.csv"
STAN_AV_M = "outputs/predictions/stan_availability_metrics.csv"
STAN_AV_D = "outputs/predictions/stan_availability_diagnostics.csv"
STAN_AV_B = "outputs/predictions/stan_availability_board.csv"
SEASON_TOTAL = "outputs/predictions/season_total_metrics.csv"
REPORT_CAL = "outputs/eda/report_calibration.csv"
COMP_M = "outputs/predictions/stan_composition_metrics.csv"
COMP_D = "outputs/predictions/stan_composition_diagnostics.csv"
COMP_P = "outputs/predictions/stan_composition_ppc.csv"
COMP_J = "outputs/predictions/stan_composition_joint_nll.csv"
COMP_O = "outputs/predictions/stan_composition_ot_tail.csv"
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


def _one(frame: pd.DataFrame | None, column: str, **where) -> float:
    if frame is None:
        return float("nan")
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
    frame = table(rel)
    return float(frame[column].sum()) if frame is not None else float("nan")


def rows(rel: str) -> float:
    frame = table(rel)
    return float(len(frame)) if frame is not None else float("nan")


def max_of(rel: str, column: str) -> float:
    frame = table(rel)
    return float(frame[column].max()) if frame is not None else float("nan")


def mean_abs_dev(rel: str, column: str, centre: float, **where) -> float:
    """Mean |value - centre| over the matching rows — a calibration summary that is
    itself a quoted figure, so it has to come from the artifact like any other."""
    frame = table(rel)
    if frame is None:
        return float("nan")
    mask = pd.Series(True, index=frame.index)
    for col, val in where.items():
        mask &= frame[col] == val
    hit = frame[mask]
    return float((hit[column] - centre).abs().mean()) if len(hit) else float("nan")


def serial(component: str, column: str) -> float:
    return _one(table(SERIAL), column, component=component)


def resid(column: str, kind: str | None = None,
          basis: str = "minutes_conditioned") -> float:
    """Off-diagonal summaries of the residual correlation matrix.

    Both ordered pairs are in the artifact, so a mean over off-diagonals is already
    symmetric-weighted. `kind` restricts to the eight counts, which is the population the
    prose quotes — `CLAUDE.md` reports the all-eleven mean beside it and the two differ by
    nearly 2×, so the restriction is load-bearing rather than cosmetic.
    """
    frame = table(RESID)
    if frame is None:
        return float("nan")
    off = frame[(frame["basis"] == basis)
                & (frame["component_a"] != frame["component_b"])]
    if kind is not None:
        off = off[(off["kind_a"] == kind) & (off["kind_b"] == kind)]
    return float(off["r"].mean() if column == "mean" else off["r"].max())


def rate(head: str, variant: str, column: str = "r2",
         analysis: str = "variant_sweep") -> float:
    return _one(table(RATES), column, analysis=analysis, head=head, variant=variant)


def stan_c(head: str, variant: str, column: str) -> float:
    return _one(table(STAN_C_M), column, head=head, variant=variant)


def season_eff(quantity: str, column: str) -> float:
    return _one(table(SEASON_EFF), column, quantity=quantity)


def term(head: str, arm: str, column: str) -> float:
    """One season-term ablation cell. `arm` is never optional: `base` and `trend` are
    different models and quoting a number without naming the arm is the same mistake as
    quoting a variance-budget share without its basis."""
    return _one(table(TERM_M), column, head=head, arm=arm)


def oracle_gain(head: str) -> float:
    """The perfect-league-override ceiling as a FRACTION of the base arm's MAE.

    A fraction rather than a percentage because the quoted strings carry `%`, which the
    value check re-multiplies. This is the number that bounds every form of season term.
    """
    base = term(head, "base", "test_mae")
    return (base - term(head, "oracle_league", "test_mae")) / base


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
    add(_c("260", PROFILE, lambda: rows(PROFILE), "availability_profile row count"))
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
    baselines = [("beta_binomial", "10.795", "15.39", "0.283", "0.096", "23.3"),
                 ("gbm", "10.888", "15.41", "0.262", "0.079", "19.9"),
                 ("ridge", "10.896", "15.39", "0.275", "0.103", "22.7"),
                 ("league_age", "13.614", "18.94", "−0.084", "0.174", "29.5")]
    for model, crps, mae, r2, ks, od in baselines:
        for quoted, name in [(crps, "crps_games"), (mae, "mae_games"),
                             (r2, "r2_gp_share"), (ks, "pit_ks_distance"),
                             (od, "implied_overdispersion")]:
            add(_c(quoted, METRICS,
                   lambda m=model, n=name: metric(METRICS, m, n),
                   f"{model} {name}"))
    add(_c("15.0%", METRICS,
           lambda: metric(METRICS, "beta_binomial", "predicted_share_below_41",
                          "rotation"), "GLM predicted below 41"))
    add(_c("34.8%", METRICS,
           lambda: metric(METRICS, "beta_binomial", "predicted_share_below_60",
                          "rotation"), "GLM predicted below 60"))
    add(_c("11.8%", METRICS,
           lambda: metric(METRICS, "beta_binomial", "observed_share_below_41",
                          "rotation"), "observed below 41"))
    add(_c("36.9%", METRICS,
           lambda: metric(METRICS, "beta_binomial", "observed_share_below_60",
                          "rotation"), "observed below 60"))
    add(_c("24.8%", METRICS,
           lambda: metric(METRICS, "league_age", "predicted_share_below_41",
                          "rotation"), "baseline predicted below 41"))
    add(_c("45.0%", METRICS,
           lambda: metric(METRICS, "league_age", "predicted_share_below_60",
                          "rotation"), "baseline predicted below 60"))

    # ── workload ablation ─────────────────────────────────────────────────────
    ablations = [("baseline", "10.914", "0.268"),
                 ("plus_playoff_workload", "10.795", "0.283"),
                 ("plus_playoff_only", "10.817", "0.281"),
                 ("plus_career_minutes_only", "10.883", "0.271")]
    for variant, crps, r2 in ablations:
        add(_c(crps, ABLATION,
               lambda v=variant: cell(ABLATION, "crps_games", variant=v),
               f"ablation {variant} CRPS"))
        add(_c(r2, ABLATION,
               lambda v=variant: cell(ABLATION, "r2_gp_share", variant=v),
               f"ablation {variant} R2"))

    # ── minutes nonlinearity probe ────────────────────────────────────────────
    probes = [("linear", "0.6768", "0.6670"), ("quadratic", "0.6914", "0.6746"),
              ("spline_k4", "0.6930", "0.6736")]
    for name, val, test in probes:
        add(_c(val, MIN_NONLIN,
               lambda n=name: cell(MIN_NONLIN, "val_r2", scope="variant", name=n),
               f"MPG probe {name} val R2"))
        add(_c(test, MIN_NONLIN,
               lambda n=name: cell(MIN_NONLIN, "test_r2", scope="variant", name=n),
               f"MPG probe {name} test R2"))
    add(_c("0.0124", MIN_NONLIN,
           lambda: cell(MIN_NONLIN, "val_vs_linear", scope="column",
                        name="minutes_per_game_lag1"), "prior-MPG spline, val"))
    add(_c("0.0077", MIN_NONLIN,
           lambda: cell(MIN_NONLIN, "test_vs_linear", scope="column",
                        name="minutes_per_game_lag1"), "prior-MPG spline, test"))

    # ── the minutes head ──────────────────────────────────────────────────────
    variants = [("carry_forward", "161.45", "168.24", "0.8166"),
                ("linear", "144.62", "147.18", "0.8565"),
                ("logit_own", "145.44", "147.35", "0.8565"),
                ("logit_own_quadratic", "144.83", "147.21", "0.8574"),
                ("logit_own_spline", "144.13", "146.85", "0.8572")]
    for name, val, test, r2 in variants:
        add(_c(val, STAN_MIN_M,
               lambda n=name: cell(STAN_MIN_M, "val_crps", variant=n),
               f"minutes {name} val CRPS"))
        add(_c(test, STAN_MIN_M,
               lambda n=name: cell(STAN_MIN_M, "test_crps", variant=n),
               f"minutes {name} test CRPS"))
        add(_c(r2, STAN_MIN_M,
               lambda n=name: cell(STAN_MIN_M, "test_r2", variant=n),
               f"minutes {name} test R2"))
    add(_c("1,829", STAN_MIN_G, lambda: total(STAN_MIN_G, "wall_clock_s"),
           "minutes head wall clock"))
    add(_c("0.0495", STAN_MIN_D,
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

    # ── the Stan availability port ────────────────────────────────────────────
    ports = [("beta_binomial", "10.7952", "0.2831", "0.0963", "0.2757"),
             ("stan_plug_in", "10.7947", "0.2832", "0.0952", "0.2759"),
             ("stan_posterior", "10.7953", "0.2832", "0.0963", "0.2759")]
    for model, crps, r2, ks, rho in ports:
        for quoted, name in [(crps, "crps_games"), (r2, "r2_gp_share"),
                             (ks, "pit_ks_distance"), (rho, "dispersion_rho")]:
            add(_c(quoted, STAN_AV_M,
                   lambda m=model, n=name: metric(STAN_AV_M, m, n),
                   f"stan {model} {name}"))
    add(_c("1.0025", STAN_AV_D, lambda: cell(STAN_AV_D, "max_rhat"), "stan R-hat"))
    add(_c("2,402", STAN_AV_D, lambda: cell(STAN_AV_D, "min_ess_bulk"), "stan min ESS"))
    add(_c("254", STAN_AV_D, lambda: cell(STAN_AV_D, "wall_clock_s"),
           "stan wall clock"))
    add(_c("219", STAN_AV_B,
           lambda: cell(STAN_AV_B, "shared_beta_sd", n_players=911),
           "shared-beta sd, full board"))

    # ── the season total ──────────────────────────────────────────────────────
    totals = [("full_season", "646.3", "831.2", "0.141", "541.9"),
              ("prior_gp", "475.0", "638.5", "0.493", "41.9"),
              ("league_age", "476.0", "595.8", "0.559", "23.2"),
              ("beta_binomial", "435.1", "570.8", "0.595", "6.1"),
              ("oracle_rate", "302.7", "427.7", "0.773", "5.3"),
              ("oracle_gp", "221.3", "304.2", "0.885", "−33.2")]
    for name, mae, rmse, r2, bias in totals:
        for quoted, which in [(mae, "mae_dk_total"), (rmse, "rmse_dk_total"),
                              (r2, "r2_dk_total"), (bias, "bias_dk_total")]:
            add(_c(quoted, SEASON_TOTAL,
                   lambda n=name, w=which: treatment(n, w),
                   f"season total {name} {which}"))
    add(_c("651.3", SEASON_TOTAL,
           lambda: treatment("full_season", "mae_dk_total", "rotation"),
           "rotation naive MAE"))
    add(_c("499.4", SEASON_TOTAL,
           lambda: treatment("beta_binomial", "mae_dk_total", "rotation"),
           "rotation head MAE"))
    add(_c("0.469", PROFILE,
           lambda: prof("decomposition", "missed_scratch", "r_persistence_of_column"),
           "missed_scratch persistence"))

    C += _regime_claims(AVAIL)
    return C


def _regime_claims(doc: str) -> list[Claim]:
    """The two regime confounds, tested rather than flagged.

    Shared by `docs/availability-plan.md` and `CLAUDE.md` against the one artifact. The
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


def _composition() -> list[Claim]:
    """`docs/minutes-composition-plan.md` — the team-game minutes allocation pilot.

    The `binomial` row is the one worth auditing hardest: it is the arm that FAILS, and
    "the pure decomposition is too tight" is the pilot's sharpest claim.
    """
    C: list[Claim] = []
    add = C.append

    comp = [("carry_forward", "4.6331", "4.8194", "0.3679", "0.0178"),
            ("binomial", "4.9345", "4.9429", "0.4307", "0.1942"),
            ("betabinom", "4.5109", "4.5361", "0.4307", "0.0205"),
            ("betabinom_ot", "4.5099", "4.5353", "0.4306", "0.0202"),
            ("betabinom_ot_graded", "4.4561", "4.5078", "0.4295", "0.0221"),
            ("independent_comparator", "4.7842", "4.9140", "0.3301", "0.0769")]
    for name, val_crps, test_crps, test_r2, pit in comp:
        for quoted, column in [(val_crps, "val_crps"), (test_crps, "test_crps"),
                               (test_r2, "test_r2"), (pit, "test_pit_ks")]:
            add(_c(quoted, COMP_M,
                   lambda n=name, c=column: cell(COMP_M, c, variant=n),
                   f"composition {name} {column}", doc=COMP))
    add(_c("2.576", COMP_M, lambda: cell(COMP_M, "probe_hours", variant="binomial"),
           "composition Gate A extrapolation", doc=COMP))
    add(_c("171.6", COMP_D, lambda: total(COMP_D, "wall_clock_s") / 60,
           "composition sampler minutes", doc=COMP))
    add(_c("1.0093", COMP_D, lambda: max_of(COMP_D, "max_rhat"),
           "composition max R-hat", doc=COMP))
    add(_c("0", COMP_D, lambda: total(COMP_D, "divergences"),
           "composition divergences", doc=COMP))

    # The team-sum asymmetry — the capability the model exists for, so both sides
    # are audited rather than only the headline.
    # The PPC file now carries BOTH arms, so every claim names its variant — without
    # it a lookup silently takes whichever row sorts first, which is the class of
    # quiet mistake this module exists to catch.
    SEL = "betabinom_ot_graded"
    add(_c("36.87", COMP_P,
           lambda: cell(COMP_P, "simulated", variant=SEL,
                        analysis="team_sum_abs_error", group="independent"),
           "comparator team-sum error", doc=COMP))
    add(_c("0.5882", COMP_P,
           lambda: cell(COMP_P, "observed", variant=SEL, analysis="starter_share",
                        group="regulation/composition"),
           "observed starter share, regulation", doc=COMP))
    add(_c("0.6314", COMP_P,
           lambda: cell(COMP_P, "observed", variant=SEL, analysis="starter_share",
                        group="overtime/composition"),
           "observed starter share, overtime", doc=COMP))
    add(_c("0.6034", COMP_P,
           lambda: cell(COMP_P, "simulated", variant=SEL, analysis="starter_share",
                        group="regulation/composition"),
           "simulated starter share, regulation", doc=COMP))
    add(_c("0.6446", COMP_P,
           lambda: cell(COMP_P, "simulated", variant=SEL, analysis="starter_share",
                        group="overtime/composition"),
           "simulated starter share, overtime", doc=COMP))

    # The graded-vs-shared calibration table — the point of the graded arm, so both
    # columns are audited rather than only the improved one.
    ratios = [("betabinom_ot", "q1_fringe", "1.5900"),
              ("betabinom_ot", "q2", "0.9656"),
              ("betabinom_ot", "q3", "0.8961"),
              ("betabinom_ot", "q4_star", "0.7000"),
              (SEL, "q1_fringe", "1.2093"),
              (SEL, "q2", "0.8405"),
              (SEL, "q3", "0.9587"),
              (SEL, "q4_star", "0.9880")]
    for arm, tier, quoted in ratios:
        add(_c(quoted, COMP_P,
               lambda a=arm, t=tier: cell(COMP_P, "ratio", variant=a,
                                          analysis="variance_ratio", group=t),
               f"variance ratio {arm} {tier}", doc=COMP))
    for arm, quoted in [("betabinom_ot", "0.2571"), (SEL, "0.1055")]:
        add(_c(quoted, COMP_P,
               lambda a=arm: mean_abs_dev(COMP_P, "ratio", 1.0, variant=a,
                                          analysis="variance_ratio"),
               f"mean |ratio-1| {arm}", doc=COMP))

    # The fitted dispersions themselves — the mechanism, and the sharpest single
    # statement that role grading is real.
    graded_rho = [("1", "0.1480"), ("2", "0.1125"), ("3", "0.0874"), ("4", "0.0613")]
    for b, quoted in graded_rho:
        add(_c(quoted, COMP_RHO,
               lambda i=int(b): cell(COMP_RHO, "rho", variant=SEL, bin=i),
               f"graded rho bin {b}", doc=COMP))
    add(_c("0.0970", COMP_RHO,
           lambda: cell(COMP_RHO, "rho", variant="betabinom_ot", bin=1),
           "shared rho", doc=COMP))
    add(_c("2.41", COMP_RHO,
           lambda: (cell(COMP_RHO, "rho", variant=SEL, bin=1)
                    / cell(COMP_RHO, "rho", variant=SEL, bin=4)),
           "graded rho spread", doc=COMP))

    for arm, quoted in [("composition", "33.51"), ("independent", "38.83")]:
        add(_c(quoted, COMP_J,
               lambda a=arm: cell(COMP_J, "mean_joint_nll", split="test", arm=a),
               f"composition joint NLL {arm}", doc=COMP))

    add(_c("0.0608", COMP_O, lambda: cell(COMP_O, "p_any_ot", **{"class": "params"}),
           "OT tail p_any", doc=COMP))
    add(_c("0.1408", COMP_O, lambda: cell(COMP_O, "p_more_ot", **{"class": "params"}),
           "OT tail p_more", doc=COMP))
    add(_c("30,626", COMP_O, lambda: cell(COMP_O, "n_games", **{"class": "params"}),
           "OT tail training games", doc=COMP))
    add(_c("256.9", COMP_O, lambda: cell(COMP_O, "predicted", **{"class": "1OT"}),
           "OT tail predicted 1OT", doc=COMP))
    add(_c("222", COMP_O, lambda: cell(COMP_O, "observed", **{"class": "1OT"}),
           "OT tail observed 1OT", doc=COMP))

    return C


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
    add("211.1", SEASON_TOTAL,
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
                   ("fg3a", "0.065", "−0.015", "+0.080", "1.48"),
                   ("fg2a", "0.062", "−0.015", "+0.077", "1.46"),
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
    # The pre-refresh cells, preserved in the ⚠️ note beside the table. `CLAUDE.md` had
    # already been refreshed and this table had not, which is the drift the audit found.
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
    add("0.8572", STAN_MIN_M,
        lambda: cell(STAN_MIN_M, "test_r2", variant="logit_own_spline"),
        "minutes head test R2")
    add("0.8166", STAN_MIN_M,
        lambda: cell(STAN_MIN_M, "test_r2", variant="carry_forward"),
        "minutes no-fit floor")
    for variant, quoted in [("logit_own_spline", "752"), ("linear", "168")]:
        add(quoted, STAN_MIN_G,
            lambda v=variant: cell(STAN_MIN_G, "wall_clock_s", label=f"{v}/test"),
            f"minutes {variant} wall clock")

    # ── the composition alternative, summarised back into this doc ────────────
    for variant, quoted in [("betabinom_ot_graded", "4.5078"),
                            ("carry_forward", "4.8194"),
                            ("independent_comparator", "4.9140"),
                            ("binomial", "4.9429")]:
        add(quoted, COMP_M, lambda v=variant: cell(COMP_M, "test_crps", variant=v),
            f"composition {variant} test CRPS")
    add("0.1942", COMP_M,
        lambda: cell(COMP_M, "test_pit_ks", variant="binomial"),
        "composition binomial PIT KS")
    add("36.87", COMP_P,
        lambda: cell(COMP_P, "simulated", variant="betabinom_ot_graded",
                     analysis="team_sum_abs_error",
                     group="independent"), "comparator team-sum error")
    add("−0.406", COMP_M,
        lambda: (cell(COMP_M, "test_crps", variant="betabinom_ot_graded")
                 - cell(COMP_M, "test_crps", variant="independent_comparator")),
        "composition CRPS gain")
    for tier, quoted in [("q1_fringe", "1.21"), ("q4_star", "0.99")]:
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
    effects = [("fg3a", "2.97", "+4.07", "0.93", "6.53"),
               ("fta", "1.21", "−0.55", "0.63", "4.36"),
               ("blk", "1.14", "−0.14", "0.12", "3.70"),
               ("ast", "1.30", "+0.73", "0.64", "3.08"),
               ("stl", "1.18", "−0.08", "0.03", "3.03"),
               ("tov", "1.16", "−0.33", "0.59", "2.89"),
               ("fg2a", "1.32", "−0.87", "0.86", "2.42"),
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
    add("−24.4%", SEASON_EFF,
        lambda: season_eff("fg3a", "worst_yoy_pct") / 100.0, "fg3a worst year")
    add("+4.07%", SEASON_EFF,
        lambda: season_eff("fg3a", "trend_pct_per_season") / 100.0,
        "fg3a trend, prose")
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
    bias = [("fta", "−3.1", "−10.7", "−7.0"), ("stl", "−10.0", "+0.7", "−4.6"),
            ("fg3a", "−7.0", "−1.3", "−4.2"), ("tov", "−4.6", "−1.8", "−3.2"),
            ("ast", "−1.1", "−4.8", "−2.9"), ("blk", "+6.5", "+6.0", "+6.2")]
    for comp, s24, s25, both in bias:
        for quoted, season in [(s24, "2024-25"), (s25, "2025-26"), (both, "all")]:
            add(f"{quoted}%", SEASON_BIAS,
                lambda c=comp, s=season: carry_bias(c, s) / 100.0,
                f"carry-forward bias {comp} {season}")

    # ── the rate side ─────────────────────────────────────────────────────────
    # Poisson/sklearn arm — the one the NB block below overturns. Kept apart on purpose.
    for head, quoted in [("fg3a", "0.520"), ("blk", "0.637")]:
        add(quoted, RATES, lambda h=head: rate(h, "linear"),
            f"poisson {head} linear R2")
    for head, quoted in [("fg3a", "0.879"), ("blk", "0.820")]:
        add(quoted, RATES, lambda h=head: rate(h, "log_own"),
            f"poisson {head} log_own R2")
    for head, quoted in [("fg3a", "0.030"), ("blk", "0.040")]:
        add(quoted, RATES,
            lambda h=head: rate(h, "log_own_spline") - rate(h, "log_own"),
            f"poisson {head} spline gain")
    for head, quoted in [("fg3a", "0.8791"), ("blk", "0.8204")]:
        add(quoted, RATES, lambda h=head: rate(h, "log_own"),
            f"poisson {head} log_own R2, 4dp")
    # "the best of seven fitted variants beats the floor by +0.0019 to +0.0228" — the
    # range over the eight count heads, which is what makes the floor binding.
    count_heads = ("fg2a", "fg3a", "fta", "reb", "ast", "stl", "blk", "tov")
    fitted = ("linear", "log_own", "log_own_spline", "log_own_inter", "pca",
              "pca_spline", "pca_inter")
    def _best_gain(head: str) -> float:
        return max(rate(head, v) for v in fitted) - rate(head, "carry_forward")
    add("0.0019", RATES, lambda: min(_best_gain(h) for h in count_heads),
        "smallest gain over the no-fit floor")
    add("0.0228", RATES, lambda: max(_best_gain(h) for h in count_heads),
        "largest gain over the no-fit floor")
    add("221.3", SEASON_TOTAL, lambda: treatment("oracle_gp", "mae_dk_total"),
        "oracle GP MAE")
    add("302.7", SEASON_TOTAL, lambda: treatment("oracle_rate", "mae_dk_total"),
        "oracle rate MAE")
    # The conversion heads' fitted variants are named differently from the count heads'
    # in the sklearn run — `spline_own` / `inter` / `pca_inter`, no `log_own` — so the
    # "best fitted" minimum has to enumerate the right five.
    conversion_variants = ("linear", "spline_own", "inter", "pca", "pca_inter")
    for head, quoted in [("fg2m|fg2a", "0.061"), ("fg3m|fg3a", "0.037")]:
        add(quoted, RATES,
            lambda h=head: rate(h, "carry_forward", "nll")
            - min(rate(h, v, "nll") for v in conversion_variants),
            f"poisson {head} NLL gain")

    # ── the built Stan block (negative binomial) ──────────────────────────────
    add("10.7947", STAN_AV_M, lambda: metric(STAN_AV_M, "stan_plug_in", "crps_games"),
        "stan availability CRPS")
    add("10.7952", STAN_AV_M,
        lambda: metric(STAN_AV_M, "beta_binomial", "crps_games"), "MLE CRPS")
    add("0.2759", STAN_AV_M,
        lambda: metric(STAN_AV_M, "stan_plug_in", "dispersion_rho"), "stan rho")
    add("0.2757", STAN_AV_M,
        lambda: metric(STAN_AV_M, "beta_binomial", "dispersion_rho"), "MLE rho")
    add("1.0025", STAN_AV_D, lambda: cell(STAN_AV_D, "max_rhat"), "stan R-hat")
    add("254", STAN_AV_D, lambda: cell(STAN_AV_D, "wall_clock_s"),
        "stan wall clock")
    add("0.2%", STAN_AV_B,
        lambda: cell(STAN_AV_B, "inflation", n_players=15) - 1.0,
        "board inflation, 15 players")
    add("6.4%", STAN_AV_B,
        lambda: cell(STAN_AV_B, "inflation", n_players=911) - 1.0,
        "board inflation, all 911")
    add("−21.4", STAN_MIN_M,
        lambda: (cell(STAN_MIN_M, "test_crps", variant="logit_own_spline")
                 - cell(STAN_MIN_M, "test_crps", variant="carry_forward")),
        "minutes CRPS gain over floor")
    add("+0.041", STAN_MIN_M,
        lambda: (cell(STAN_MIN_M, "test_r2", variant="logit_own_spline")
                 - cell(STAN_MIN_M, "test_r2", variant="carry_forward")),
        "minutes R2 gain over floor")
    add("208.6", STAN_C_D, lambda: total(STAN_C_D, "wall_clock_s") / 60,
        "component sampler minutes")
    add("1.0118", STAN_C_D, lambda: max_of(STAN_C_D, "max_rhat"),
        "component max R-hat")
    for head, quoted in [("blk", "0.6794"), ("fg3a", "0.3719")]:
        add(quoted, STAN_C_M, lambda h=head: stan_c(h, "log_own", "test_r2"),
            f"NB {head} log_own R2")
    for head, quoted in [("blk", "0.8579"), ("fg3a", "0.9046")]:
        add(quoted, STAN_C_M,
            lambda h=head: stan_c(h, "log_own_spline", "test_r2"),
            f"NB {head} spline R2")
    add("−19.00", STAN_C_M, lambda: stan_c("fg3a", "linear", "test_r2"),
        "NB fg3a linear R2")
    add("−1.393", STAN_C_M, lambda: stan_c("blk", "linear", "test_r2"),
        "NB blk linear R2")
    add("0.8649", STAN_C_M, lambda: stan_c("fta", "log_own", "test_r2"),
        "NB fta best fitted R2")
    add("0.8673", STAN_C_M, lambda: stan_c("fta", "carry_forward", "test_r2"),
        "NB fta floor")
    add("−0.79", STAN_C_S,
        lambda: cell(STAN_C_S, "reparam_minus_canonical", split="test",
                     arm="two_counts"), "substitution gain, rounded")
    for split, quoted in [("val", "−0.771"), ("test", "−0.793")]:
        add(quoted, STAN_C_S,
            lambda s=split: cell(STAN_C_S, "reparam_minus_canonical", split=s,
                                 arm="two_counts"),
            f"substitution gain, {split}")
    for split, arm, quoted in [("val", "two_counts", "10.797"),
                               ("val", "fga_x_fg3a_share", "10.026"),
                               ("test", "two_counts", "10.784"),
                               ("test", "fga_x_fg3a_share", "9.991")]:
        add(quoted, STAN_C_S,
            lambda s=split, a=arm: cell(STAN_C_S, "mean_joint_nll", split=s, arm=a),
            f"substitution joint NLL {split}/{arm}")

    # ── the residual copula ───────────────────────────────────────────────────
    add("0.0121", RESID, lambda: resid("mean", kind="count"),
        "residual off-diagonal mean, 8 counts")
    add("0.1422", RESID, lambda: resid("max"), "residual max off-diagonal")

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
    # The full block. `CLAUDE.md` carries a summary of the same verdict and claims the
    # overlapping figures against the SAME artifact — deliberately, because a block going
    # stale in one doc while current in the other is this repo's recorded failure mode and
    # it has already happened twice.
    C += _season_term_claims(PRED)
    return C


def _season_term_summary_claims(doc: str) -> list[Claim]:
    """The subset of the verdict `CLAUDE.md` quotes, against the same artifact.

    A subset rather than the whole block because `CLAUDE.md` is a summary and does not
    carry every cell — forcing it to would make the two docs the same document. What it
    does carry is claimed here, so the two cannot drift apart on the figures they share.
    """
    C: list[Claim] = []
    add = C.append
    add(_c("33.247", TERM_M, lambda: term("fg3a", "base", "val_crps"),
           "fg3a base val CRPS", doc=doc))
    add(_c("35.443", TERM_M, lambda: term("fg3a", "trend", "val_crps"),
           "fg3a trend val CRPS", doc=doc))
    add(_c("34.309", TERM_M, lambda: term("fg3a", "year", "val_crps"),
           "fg3a year val CRPS", doc=doc))
    add(_c("−3.74%", TERM_M, lambda: term("fg3a", "base", "bias_pct") / 100.0,
           "fg3a base bias", doc=doc))
    add(_c("+8.44%", TERM_M, lambda: term("fg3a", "trend", "bias_pct") / 100.0,
           "fg3a trend bias", doc=doc))
    add(_c("−9.01%", TERM_M, lambda: term("fg3a", "year", "bias_pct") / 100.0,
           "fg3a year bias", doc=doc))
    for head, quoted in [("stl", "3.11%"), ("blk", "2.32%"), ("fta", "2.16%"),
                         ("reb", "1.71%"), ("fg3a", "0.92%"), ("ast", "0.58%"),
                         ("tov", "0.01%"), ("fg2a", "−0.06%")]:
        add(_c(quoted, TERM_M, lambda h=head: oracle_gain(h),
               f"oracle ceiling {head}", doc=doc))
    add(_c("108.88", TERM_TOTAL, lambda: term_total("trend", "mae"),
           "season total trend MAE", doc=doc))
    add(_c("109.96", TERM_TOTAL, lambda: term_total("base", "mae"),
           "season total base MAE", doc=doc))
    add(_c("+7.60", TERM_TOTAL, lambda: term_total("trend", "bias"),
           "season total trend bias", doc=doc))
    add(_c("−32.44", TERM_TOTAL, lambda: term_total("base", "bias"),
           "season total base bias", doc=doc))
    for head, sigma in [("blk", "0.99×"), ("tov", "0.94×"), ("fta", "0.87×"),
                        ("stl", "0.82×"), ("reb", "1.18×")]:
        add(_c(sigma, TERM_SIGMA, lambda h=head: term_sigma(h, "ratio_to_yoy_sd"),
               f"sigma/league ratio {head}", doc=doc))
    add(_c("2.36×", TERM_SIGMA, lambda: term_sigma("fg3a", "ratio_to_yoy_sd"),
           "sigma/league ratio fg3a", doc=doc))
    for n, quoted in [(12, "+15.0%"), (15, "+19.0%"), (30, "+37.7%"),
                      (150, "+138%"), (791, "+364%")]:
        add(_c(quoted, TERM_SPREAD, lambda k=n: term_spread(k, "year") - 1.0,
               f"year roster spread inflation, {n} players", doc=doc))
    for arm, val, test in [("base", "144.09", "147.02"), ("year", "143.81", "146.54")]:
        add(_c(val, TERM_M, lambda a=arm: term("min", a, "val_crps"),
               f"minutes {arm} val CRPS", doc=doc))
        add(_c(test, TERM_M, lambda a=arm: term("min", a, "test_crps"),
               f"minutes {arm} test CRPS", doc=doc))
    add(_c("−38.2", TERM_M, lambda: term("min", "year", "bias"),
           "minutes year bias", doc=doc))
    add(_c("−56.6", TERM_M, lambda: term("min", "trend", "bias"),
           "minutes trend bias", doc=doc))
    for arm, quoted in [("base", "−16.6%"), ("trend", "−12.0%"), ("year", "−18.5%")]:
        add(_c(quoted, TERM_BONUS, lambda a=arm: term_bonus(a) / 100.0,
               f"bonus bias {arm}", doc=doc))
    return C


def _season_term_claims(doc: str) -> list[Claim]:
    """The `make season-terms` verdict, in full, for `docs/predictions-plan.md`."""
    C: list[Claim] = []
    add = C.append

    # `fg3a` — the head that refutes the trend, and the reason the verdict is what it is.
    for arm, val, test, bias in [("base", "33.247", "33.723", "−3.74%"),
                                 ("trend", "35.443", "33.900", "+8.44%"),
                                 ("year", "34.309", "35.408", "−9.01%"),
                                 ("trend_year", "36.108", "34.706", "+11.48%")]:
        add(_c(val, TERM_M, lambda a=arm: term("fg3a", a, "val_crps"),
               f"fg3a {arm} val CRPS", doc=doc))
        add(_c(test, TERM_M, lambda a=arm: term("fg3a", a, "test_crps"),
               f"fg3a {arm} test CRPS", doc=doc))
        add(_c(bias, TERM_M, lambda a=arm: term("fg3a", a, "bias_pct") / 100.0,
               f"fg3a {arm} bias", doc=doc))
    add(_c("38.747", TERM_M, lambda: term("fg3a", "carry_forward", "test_crps"),
           "fg3a floor CRPS", doc=doc))

    # The oracle ceiling — the single number that bounds the whole question.
    for head, quoted in [("stl", "3.11%"), ("blk", "2.32%"), ("fta", "2.16%"),
                         ("reb", "1.71%"), ("fg3a", "0.92%"), ("ast", "0.58%"),
                         ("tov", "0.01%"), ("fg2a", "−0.06%")]:
        add(_c(quoted, TERM_M, lambda h=head: oracle_gain(h),
               f"oracle ceiling {head}", doc=doc))

    # Season-total dk_pts, uniform-arm and test-only.
    for arm, mae, bias, crps in [("base", "109.96", "−32.44", "82.70"),
                                 ("trend", "108.88", "+7.60", "80.57"),
                                 ("year", "113.64", "−51.71", "85.68"),
                                 ("trend_year", "110.09", "+15.25", "80.87")]:
        add(_c(mae, TERM_TOTAL, lambda a=arm: term_total(a, "mae"),
               f"season total {arm} MAE", doc=doc))
        add(_c(bias, TERM_TOTAL, lambda a=arm: term_total(a, "bias"),
               f"season total {arm} bias", doc=doc))
        add(_c(crps, TERM_TOTAL, lambda a=arm: term_total(a, "crps"),
               f"season total {arm} CRPS", doc=doc))

    # `sigma_year` against the independently measured league movement.
    for head, sigma, league, ratio in [("blk", "3.66%", "3.70%", "0.99"),
                                       ("tov", "2.71%", "2.89%", "0.94"),
                                       ("fta", "3.78%", "4.36%", "0.87"),
                                       ("stl", "2.49%", "3.03%", "0.82"),
                                       ("reb", "1.75%", "1.48%", "1.18"),
                                       ("fg2a", "3.03%", "2.42%", "1.25"),
                                       ("ast", "4.46%", "3.08%", "1.45"),
                                       ("fg3a", "15.43%", "6.53%", "2.36")]:
        add(_c(sigma, TERM_SIGMA, lambda h=head: term_sigma(h, "sigma_year_pct") / 100.0,
               f"sigma_year {head}", doc=doc))
        add(_c(league, TERM_SIGMA,
               lambda h=head: term_sigma(h, "league_yoy_sd_pct") / 100.0,
               f"league yoy sd {head}, beside sigma", doc=doc))
        add(_c(ratio, TERM_SIGMA, lambda h=head: term_sigma(h, "ratio_to_yoy_sd"),
               f"sigma/league ratio {head}", doc=doc))

    # Roster spread — what the year effect is actually worth.
    for n, quoted in [(12, "+15.0%"), (15, "+19.0%"), (30, "+37.7%"),
                      (150, "+138%"), (791, "+364%")]:
        add(_c(quoted, TERM_SPREAD, lambda k=n: term_spread(k, "year") - 1.0,
               f"year roster spread inflation, {n} players", doc=doc))
    add(_c("841", TERM_SPREAD, lambda: term_spread(15, "base", "total_sd"),
           "base roster sd, 15 players", doc=doc))
    add(_c("1,001", TERM_SPREAD, lambda: term_spread(15, "year", "total_sd"),
           "year roster sd, 15 players", doc=doc))

    # Minutes — the one head that adopts a season term.
    for arm, val, test, bias in [("carry_forward", "161.45", "168.24", "−5.69"),
                                 ("base", "144.09", "147.02", "−41.05"),
                                 ("trend", "144.44", "148.78", "−56.59"),
                                 ("year", "143.81", "146.54", "−38.19")]:
        add(_c(val, TERM_M, lambda a=arm: term("min", a, "val_crps"),
               f"minutes {arm} val CRPS", doc=doc))
        add(_c(test, TERM_M, lambda a=arm: term("min", a, "test_crps"),
               f"minutes {arm} test CRPS", doc=doc))
        add(_c(bias, TERM_M, lambda a=arm: term("min", a, "bias"),
               f"minutes {arm} bias", doc=doc))

    # The bonus, whose LEVEL is the missing copula and whose ORDERING is the bias story.
    for arm, quoted in [("base", "−16.6%"), ("trend", "−12.0%"),
                        ("year", "−18.5%"), ("trend_year", "−11.6%")]:
        add(_c(quoted, TERM_BONUS, lambda a=arm: term_bonus(a) / 100.0,
               f"bonus bias {arm}", doc=doc))
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


def _claude() -> list[Claim]:
    """`CLAUDE.md`'s established-facts section — the source of truth the plan docs defer to.

    Only the *figures* are claimed. The project-layout, pipeline, conventions and dashboard
    sections carry no measurements and need none, which is why this doc's coverage
    denominator is smaller than its length suggests.
    """
    C: list[Claim] = []

    def add(quoted, artifact, actual, label, **kw):
        C.append(_c(quoted, artifact, actual, label, doc=CLAUDE, **kw))

    def context(feature: str, column: str) -> float:
        return _one(table(CONTEXT_A), column, feature=feature)

    def opp(outcome: str, column: str) -> float:
        return _one(table(OPPONENT_A), column, outcome=outcome)

    def bonus(unit: str, od: float, column: str, bucket: str = "all") -> float:
        return _one(table(BONUS), column, analysis="calibration", unit=unit,
                    bucket=bucket, overdispersion=od)

    def age_ratio(metric: str, age: int) -> float:
        return _one(table(AGING), "cumulative_ratio", tier="A", metric=metric,
                    archetype=-1, age=age)

    def glen(column: str, season: str = "all",
             analysis: str = "feasibility") -> float:
        return _one(table(GAME_LEN), column, analysis=analysis, season=season,
                    season_type="regular")

    # ── the variance budget ───────────────────────────────────────────────────
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
        add(ceiling, VARIANCE,
            lambda n=null: budget(
                "opponent_x_archetype_x_season_above_null_variance_ceiling",
                null_construction=n),
            f"variance ceiling, {null}")
        add(gap, VARIANCE,
            lambda n=null: budget("null_reproduction_gap", null_construction=n) * 100,
            f"reproduction gap, {null}")
    add("0.9619%", VARIANCE,
        lambda: budget("opponent_x_archetype_x_season_above_null",
                       null_construction="shuffle_opponent_within_season"),
        "cell_importance, shuffle opponent")
    add("0.3080%", VARIANCE,
        lambda: budget("opponent_x_archetype_x_season_above_null",
                       null_construction="shuffle_archetype_within_season"),
        "cell_importance, shuffle archetype")

    # ── playoff scope and workload ────────────────────────────────────────────
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
    # `persistence.csv` keys on the prefixed column name, not the bare stat, because
    # `blk` exists in four families — the same reason `CLAUDE.md` warns that
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
    for feature, quoted in persistence:
        add(quoted, PERSIST, lambda f=feature: persist(f),
            f"persistence {feature}")
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
    for model, quoted in [("beta_binomial", "10.795"), ("gbm", "10.888"),
                          ("ridge", "10.896"), ("league_age", "13.614")]:
        add(quoted, METRICS, lambda m=model: metric(METRICS, m, "crps_games"),
            f"{model} CRPS")
    add("0.268", ABLATION,
        lambda: cell(ABLATION, "r2_gp_share", variant="baseline"),
        "held-out R2 before playoff workload")
    add("10.914", ABLATION,
        lambda: cell(ABLATION, "crps_games", variant="baseline"),
        "baseline CRPS")

    # ── the season total ──────────────────────────────────────────────────────
    totals = [("full_season", "646.3", "831.2", "0.141", "541.9"),
              ("prior_gp", "475.0", "638.5", "0.493", "41.9"),
              ("league_age", "476.0", "595.8", "0.559", "23.2"),
              ("beta_binomial", "435.1", "570.8", "0.595", "6.1"),
              ("oracle_rate", "302.7", "427.7", "0.773", "5.3"),
              ("oracle_gp", "221.3", "304.2", "0.885", "−33.2")]
    for name, mae, rmse, r2, bias in totals:
        for quoted, which in [(mae, "mae_dk_total"), (rmse, "rmse_dk_total"),
                              (r2, "r2_dk_total"), (bias, "bias_dk_total")]:
            add(quoted, SEASON_TOTAL,
                lambda n=name, w=which: treatment(n, w),
                f"season total {name} {which}")
    add("211.1", SEASON_TOTAL,
        lambda: (treatment("full_season", "mae_dk_total")
                 - treatment("beta_binomial", "mae_dk_total")),
        "head vs a full season")
    add("651.3", SEASON_TOTAL,
        lambda: treatment("full_season", "mae_dk_total", "rotation"),
        "rotation naive MAE")
    add("499.4", SEASON_TOTAL,
        lambda: treatment("beta_binomial", "mae_dk_total", "rotation"),
        "rotation head MAE")
    add("441.3", ABLATION,
        lambda: treatment("beta_binomial", "mae_dk_total")
        + 6.2, "season total before playoff workload", tol=0.15)
    add("508.9", SEASON_TOTAL,
        lambda: treatment("beta_binomial", "mae_dk_total", "rotation")
        + 9.5, "rotation MAE before playoff workload", tol=0.15)

    # ── the component rate heads (Poisson / sklearn) ──────────────────────────
    poisson = [("reb", "0.9424", "0.9278", "0.9441", "0.9436", "0.9442"),
               ("fg2a", "0.9194", "0.9089", "0.9245", "0.9248", "0.9260"),
               ("ast", "0.9197", "0.8601", "0.9229", "0.9262", "0.9236"),
               ("fg3a", "0.9036", "0.5197", "0.8791", "0.9088", "0.8784"),
               ("blk", "0.8407", "0.6375", "0.8204", "0.8605", "0.8228"),
               ("fta", "0.8673", "0.8449", "0.8689", "0.8692", "0.8720"),
               ("stl", "0.8194", "0.8170", "0.8369", "0.8397", "0.8338"),
               ("tov", "0.8845", "0.8828", "0.8915", "0.8913", "0.8916")]
    for head, floor, linear, log_own, spline, inter in poisson:
        for quoted, variant in [(floor, "carry_forward"), (linear, "linear"),
                                (log_own, "log_own"), (spline, "log_own_spline"),
                                (inter, "log_own_inter")]:
            add(quoted, RATES, lambda h=head, v=variant: rate(h, v),
                f"poisson {head} {variant}")
    add("3.0894", RATES,
        lambda: min(rate("ftm|fta", v, "nll")
                    for v in ("linear", "spline_own", "inter", "pca", "pca_inter")),
        "ftm|fta best fitted NLL")
    add("3.0822", RATES, lambda: rate("ftm|fta", "carry_forward", "nll"),
        "ftm|fta floor NLL")
    for alpha, quoted in [(1e-8, "0.9278"), (0.01, "0.9322"), (1.0, "0.6620")]:
        add(quoted, RATES,
            lambda a=alpha: _one(table(RATES), "r2", analysis="alpha_sensitivity",
                                 head="reb", variant="linear", alpha=a),
            f"reb alpha sensitivity at {alpha}")

    # ── the Stan heads ────────────────────────────────────────────────────────
    stan_counts = [("reb", "0.9424", "0.9095", "0.9439", "0.9428"),
                   ("fg2a", "0.9194", "0.9018", "0.9241", "0.9241"),
                   ("ast", "0.9197", "0.6615", "0.9223", "0.9240"),
                   ("fg3a", "0.9036", "−19.00", "0.3719", "0.9046"),
                   ("tov", "0.8845", "0.8823", "0.8929", "0.8926"),
                   ("blk", "0.8407", "−1.393", "0.6794", "0.8579"),
                   ("fta", "0.8673", "0.8171", "0.8649", "0.8648"),
                   ("stl", "0.8194", "0.8113", "0.8390", "0.8413")]
    for head, floor, linear, log_own, spline in stan_counts:
        for quoted, variant in [(floor, "carry_forward"), (linear, "linear"),
                                (log_own, "log_own"), (spline, "log_own_spline")]:
            add(quoted, STAN_C_M,
                lambda h=head, v=variant: stan_c(h, v, "test_r2"),
                f"NB {head} {variant}")
    conversions = [("fg2m|fg2a", "3.7249", "3.7770"),
                   ("fg3m|fg3a", "3.2407", "3.2614"),
                   ("ftm|fta", "3.1313", "3.0822")]
    for head, fitted, floor in conversions:
        add(fitted, STAN_C_M,
            lambda h=head: stan_c(h, "logit_own_spline", "test_nll"),
            f"NB {head} fitted NLL")
        add(floor, STAN_C_M,
            lambda h=head: stan_c(h, "carry_forward", "test_nll"),
            f"NB {head} floor NLL")
    add("0.0521", STAN_C_M,
        lambda: stan_c("fg2m|fg2a", "carry_forward", "test_nll")
        - stan_c("fg2m|fg2a", "logit_own_spline", "test_nll"),
        "fg2m|fg2a NLL gain")
    add("0.0208", STAN_C_M,
        lambda: stan_c("fg3m|fg3a", "carry_forward", "test_nll")
        - stan_c("fg3m|fg3a", "logit_own_spline", "test_nll"),
        "fg3m|fg3a NLL gain")
    add("−0.0491", STAN_C_M,
        lambda: stan_c("ftm|fta", "carry_forward", "test_nll")
        - stan_c("ftm|fta", "logit_own_spline", "test_nll"),
        "ftm|fta NLL gain")
    add("208.6", STAN_C_D, lambda: total(STAN_C_D, "wall_clock_s") / 60,
        "component sampler minutes")
    add("1.0118", STAN_C_D, lambda: max_of(STAN_C_D, "max_rhat"),
        "component max R-hat")
    # The handicapped margins, kept in the doc beside their correction. They remain
    # exactly true of `stan_component_substitution.csv` — a weaker experiment, not a
    # stale value — so they are value-checked rather than flagged historical.
    add("−0.771", STAN_C_S,
        lambda: cell(STAN_C_S, "reparam_minus_canonical", split="val",
                     arm="two_counts"), "substitution gain, val (handicapped)")
    add("−0.793", STAN_C_S,
        lambda: cell(STAN_C_S, "reparam_minus_canonical", split="test",
                     arm="two_counts"), "substitution gain, test (handicapped)")
    add("0.792657", STAN_C_S,
        lambda: -cell(STAN_C_S, "reparam_minus_canonical", split="test",
                      arm="two_counts"), "the handicapped test margin, unsigned")
    # The four joint-NLL cells the doc used to quote are gone from the prose, replaced by
    # Gate 0's. `10.797` was deliberately NOT re-pointed: the string survives elsewhere in
    # this doc as an unrelated availability CRPS, so a presence check on it would pass for
    # the wrong reason — the exact false-negative `check_presence` exists to avoid.
    def shot(split: str, arm: str) -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        sub = frame[(frame["analysis"] == "joint") & (frame["split"] == split)
                    & (frame["arm"] == arm) & frame["selected"].astype(bool)]
        return float(sub["mean_nll"].iloc[0])

    def shot_head(split: str, name: str, variant: str, column: str = "mean_nll") -> float:
        arm = "two_counts" if name in ("fg2a", "fg3a") else "fga_x_fg3a_share"
        return cell(SHOT_SWEEP, column, analysis="head", split=split, arm=arm,
                    head=name, variant=variant)

    def shot_floor_total(names, variants, split: str = "test") -> float:
        return sum(shot_head(split, n, v, "floor_nll") for n, v in zip(names, variants))

    def shot_grid_best() -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        return float(frame[frame["analysis"] == "arm_a_grid"]["mean_nll"].min())

    for split, canonical, reparam, margin in [
            ("val", "10.504935", "10.004153", "−0.500782"),
            ("test", "10.478052", "9.984503", "−0.493549")]:
        add(canonical, SHOT_SWEEP, lambda s=split: shot(s, "two_counts"),
            f"gate 0 arm A joint NLL, {split}")
        add(reparam, SHOT_SWEEP, lambda s=split: shot(s, "fga_x_fg3a_share"),
            f"gate 0 arm B joint NLL, {split}")
        add(margin, SHOT_SWEEP,
            lambda s=split: shot(s, "fga_x_fg3a_share") - shot(s, "two_counts"),
            f"gate 0 margin, {split}")
    add("10.476413", SHOT_SWEEP, shot_grid_best, "gate 0 arm A best-of-16")
    add("−0.491910", SHOT_SWEEP,
        lambda: shot("test", "fga_x_fg3a_share") - shot_grid_best(),
        "gate 0 margin against arm A's best-of-16")
    add("0.305646", SHOT_SWEEP,
        lambda: (cell(STAN_C_M, "test_nll", head="fg3a", variant="log_own")
                 - shot_head("test", "fg3a", "log_own_spline")),
        "gate 0 handicap in nats")
    add("−0.487010", SHOT_SWEEP,
        lambda: (shot_head("test", "fga", "log_own")
                 + shot_head("test", "fg3a|fga", "logit_own")
                 - shot("test", "two_counts")),
        "gate 0 margin before arm B was swept")
    add("−0.006539", SHOT_SWEEP,
        lambda: (shot("test", "fga_x_fg3a_share")
                 - shot_head("test", "fga", "log_own")
                 - shot_head("test", "fg3a|fga", "logit_own")),
        "gate 0 value of sweeping arm B")
    add("9.991042", SHOT_SWEEP,
        lambda: (shot_head("test", "fga", "log_own")
                 + shot_head("test", "fg3a|fga", "logit_own")),
        "gate 0 reproduces the recorded joint NLL")
    # The headline: the coordinate change beats the fitting.
    ARM_A_F = (("fg2a", "fg3a"), ("log_own", "log_own_spline"))
    ARM_B_F = (("fga", "fg3a|fga"), ("log_own", "logit_own"))
    add("11.024027", SHOT_SWEEP, lambda: shot_floor_total(*ARM_A_F),
        "gate 0 arm A no-fit floor total")
    add("10.085599", SHOT_SWEEP, lambda: shot_floor_total(*ARM_B_F),
        "gate 0 arm B no-fit floor total")
    add("−0.938427", SHOT_SWEEP,
        lambda: shot_floor_total(*ARM_B_F) - shot_floor_total(*ARM_A_F),
        "gate 0 floor-to-floor gain")
    add("−0.390814", SHOT_SWEEP,
        lambda: shot_floor_total(*ARM_B_F) - shot_grid_best(),
        "gate 0 arm B floor vs arm A best fitted")
    add("−0.101096", SHOT_SWEEP,
        lambda: shot("test", "fga_x_fg3a_share") - shot_floor_total(*ARM_B_F),
        "gate 0 what arm B's own fitting adds")
    # The share head's floor failure on validation — the reason it needs its spline.
    for quoted, name, variant, column, split in [
            ("4.636033", "fg3a|fga", "logit_own", "mean_nll", "val"),
            ("4.619109", "fg3a|fga", "logit_own", "floor_nll", "val"),
            ("4.615620", "fg3a|fga", "logit_own_spline", "mean_nll", "val"),
            ("5.388533", "fga", "log_own_spline", "mean_nll", "val"),
            ("5.390057", "fga", "log_own", "mean_nll", "val")]:
        add(quoted, SHOT_SWEEP,
            lambda n=name, v=variant, c=column, s=split: shot_head(s, n, v, c),
            f"gate 0 {name}@{variant} {column} {split}")
    add("1.0087", SHOT_D, lambda: max_of(SHOT_D, "max_rhat"), "gate 0 max R-hat")
    add("46.5", SHOT_D, lambda: total(SHOT_D, "wall_clock_s") / 60,
        "gate 0 sampler minutes")

    # the availability port and the board decomposition
    for model, crps, rho in [("beta_binomial", "10.7952", "0.2757"),
                             ("stan_plug_in", "10.7947", "0.2759"),
                             ("stan_posterior", "10.7953", "0.2759")]:
        add(crps, STAN_AV_M, lambda m=model: metric(STAN_AV_M, m, "crps_games"),
            f"stan {model} CRPS")
        add(rho, STAN_AV_M,
            lambda m=model: metric(STAN_AV_M, m, "dispersion_rho"),
            f"stan {model} rho")
    add("1.0025", STAN_AV_D, lambda: cell(STAN_AV_D, "max_rhat"), "stan R-hat")
    add("2,402", STAN_AV_D, lambda: cell(STAN_AV_D, "min_ess_bulk"), "stan min ESS")
    add("254", STAN_AV_D, lambda: cell(STAN_AV_D, "wall_clock_s"),
        "stan wall clock")
    board_rows = [(12, "69.4", "4.0", "0.2%"), (15, "77.6", "4.8", "0.2%"),
                  (30, "109.5", "8.4", "0.3%"), (150, "245.0", "37.2", "1.1%"),
                  (911, "604.0", "219.1", "6.4%")]
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

    # the minutes head
    minutes = [("carry_forward", "161.45", "168.24", "0.8166"),
               ("linear", "144.62", "147.18", "0.8565"),
               ("logit_own", "145.44", "147.35", "0.8565"),
               ("logit_own_quadratic", "144.83", "147.21", "0.8574"),
               ("logit_own_spline", "144.13", "146.85", "0.8572")]
    for name, val, test, r2 in minutes:
        add(val, STAN_MIN_M, lambda n=name: cell(STAN_MIN_M, "val_crps", variant=n),
            f"minutes {name} val CRPS")
        add(test, STAN_MIN_M,
            lambda n=name: cell(STAN_MIN_M, "test_crps", variant=n),
            f"minutes {name} test CRPS")
        add(r2, STAN_MIN_M, lambda n=name: cell(STAN_MIN_M, "test_r2", variant=n),
            f"minutes {name} test R2")
    add("0.0495", STAN_MIN_D,
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
    add("1,829", STAN_MIN_G, lambda: total(STAN_MIN_G, "wall_clock_s"),
        "minutes head wall clock")
    add("+0.0407", STAN_MIN_M,
        lambda: (cell(STAN_MIN_M, "test_r2", variant="logit_own_spline")
                 - cell(STAN_MIN_M, "test_r2", variant="carry_forward")),
        "minutes R2 gain over floor")
    add("752", STAN_MIN_G,
        lambda: cell(STAN_MIN_G, "wall_clock_s", label="logit_own_spline/test"),
        "spline wall clock")
    add("168", STAN_MIN_G,
        lambda: cell(STAN_MIN_G, "wall_clock_s", label="linear/test"),
        "linear wall clock")

    # the composition pilot
    for variant, val, test, pit in [("carry_forward", "4.6331", "4.8194", "0.0178"),
                                    ("binomial", "4.9345", "4.9429", "0.1942"),
                                    ("betabinom", "4.5109", "4.5361", "0.0205"),
                                    ("betabinom_ot", "4.5099", "4.5353", "0.0202"),
                                    ("betabinom_ot_graded", "4.4561", "4.5078",
                                     "0.0221"),
                                    ("independent_comparator", "4.7842", "4.9140",
                                     "0.0769")]:
        add(val, COMP_M, lambda v=variant: cell(COMP_M, "val_crps", variant=v),
            f"composition {variant} val CRPS")
        add(test, COMP_M, lambda v=variant: cell(COMP_M, "test_crps", variant=v),
            f"composition {variant} test CRPS")
        add(pit, COMP_M, lambda v=variant: cell(COMP_M, "test_pit_ks", variant=v),
            f"composition {variant} PIT KS")
    add("36.87", COMP_P,
        lambda: cell(COMP_P, "simulated", variant="betabinom_ot_graded",
                     analysis="team_sum_abs_error",
                     group="independent"), "comparator team-sum error")
    add("0.5882", COMP_P,
        lambda: cell(COMP_P, "observed", variant="betabinom_ot_graded",
                     analysis="starter_share",
                     group="regulation/composition"), "starter share, regulation")
    add("0.6314", COMP_P,
        lambda: cell(COMP_P, "observed", variant="betabinom_ot_graded",
                     analysis="starter_share",
                     group="overtime/composition"), "starter share, overtime")
    # Both arms of the graded-vs-shared calibration table, each pinned to its variant:
    # the PPC artifact carries two arms now, so an unfiltered lookup would silently
    # take whichever sorts first.
    for arm, tier, quoted in [("betabinom_ot", "q1_fringe", "1.5900"),
                              ("betabinom_ot", "q2", "0.9656"),
                              ("betabinom_ot", "q3", "0.8961"),
                              ("betabinom_ot", "q4_star", "0.7000"),
                              ("betabinom_ot_graded", "q1_fringe", "1.2093"),
                              ("betabinom_ot_graded", "q2", "0.8405"),
                              ("betabinom_ot_graded", "q3", "0.9587"),
                              ("betabinom_ot_graded", "q4_star", "0.9880")]:
        add(quoted, COMP_P,
            lambda a=arm, t=tier: cell(COMP_P, "ratio", variant=a,
                                       analysis="variance_ratio", group=t),
            f"composition variance ratio {arm} {tier}")
    for arm, quoted in [("betabinom_ot", "0.2571"),
                        ("betabinom_ot_graded", "0.1055")]:
        add(quoted, COMP_P,
            lambda a=arm: mean_abs_dev(COMP_P, "ratio", 1.0, variant=a,
                                       analysis="variance_ratio"),
            f"composition mean |ratio-1| {arm}")
    for b, quoted in [(1, "0.1480"), (2, "0.1125"), (3, "0.0874"), (4, "0.0613")]:
        add(quoted, COMP_RHO,
            lambda i=b: cell(COMP_RHO, "rho", variant="betabinom_ot_graded", bin=i),
            f"composition graded rho bin {b}")
    add("0.0970", COMP_RHO,
        lambda: cell(COMP_RHO, "rho", variant="betabinom_ot", bin=1),
        "composition shared rho")
    add("2.41", COMP_RHO,
        lambda: (cell(COMP_RHO, "rho", variant="betabinom_ot_graded", bin=1)
                 / cell(COMP_RHO, "rho", variant="betabinom_ot_graded", bin=4)),
        "composition graded rho spread")
    add("0.0608", COMP_O, lambda: cell(COMP_O, "p_any_ot", **{"class": "params"}),
        "OT tail p_any")
    add("0.1408", COMP_O, lambda: cell(COMP_O, "p_more_ot", **{"class": "params"}),
        "OT tail p_more")
    add("256.9", COMP_O, lambda: cell(COMP_O, "predicted", **{"class": "1OT"}),
        "OT tail predicted 1OT")

    # ── serial and residual correlation ───────────────────────────────────────
    for comp, excess, block in [("min", "+0.294", "2.43"), ("fg3a", "+0.080", "1.48"),
                                ("fg2a", "+0.077", "1.46"), ("ftm|fta", "+0.012",
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
    add("+0.0071", RESID, lambda: resid("mean"), "residual mean, all 11")
    add("+0.0121", RESID, lambda: resid("mean", kind="count"),
        "residual mean, 8 counts")
    add("+0.1422", RESID, lambda: resid("max"), "residual max")
    add("−0.1248", RESID,
        lambda: cell(RESID, "r", component_a="fg3a", component_b="fg2a",
                     basis="minutes_conditioned"), "3PA/2PA substitution")
    add("+0.112", RESID, lambda: resid("mean", basis="raw"), "raw residual mean")
    add("+0.492", RESID, lambda: resid("max", basis="raw"), "raw residual max")
    for quoted, label in [("+0.013", "mean"), ("0.157", "max"), ("−0.110", "3PA/2PA")]:
        add(quoted, RESID, lambda: resid("mean", kind="count"),
            f"superseded residual {label} (2021-22 onward)", historical=True)

    # ── report calibration ────────────────────────────────────────────────────
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
    add("22.7%", BONUS,
        lambda: -bonus("player_season", 0.0, "relative_bias"),
        "independent sampling reads low")
    add("+0.0009", BONUS, lambda: bonus("player_season", 0.10, "bias"),
        "shipped overdispersion bias")
    add("0.0968", BONUS,
        lambda: _one(table(BONUS), "overdispersion", analysis="fitted",
                     unit="player_season"), "fitted season-unit optimum")
    add("0.0248", BONUS,
        lambda: _one(table(BONUS), "overdispersion", analysis="fitted",
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

    # ── the nonlinearity ablations ────────────────────────────────────────────
    for variant, val, test in [("linear", "10.006", None),
                               ("quadratic", "10.037", "10.749"),
                               ("spline_k4", "10.041", "10.761"),
                               ("spline_k5", "10.054", "10.751")]:
        add(val, NONLIN,
            lambda v=variant: cell(NONLIN, "val_crps_games", variant=v),
            f"nonlinearity {variant} val CRPS")
        if test:
            add(test, NONLIN,
                lambda v=variant: cell(NONLIN, "test_crps_games", variant=v),
                f"nonlinearity {variant} test CRPS")
    for name, val, test in [("linear", "0.6768", "0.6670"),
                            ("quadratic", "0.6914", "0.6746"),
                            ("spline_k4", "0.6930", "0.6736")]:
        add(val, MIN_NONLIN,
            lambda n=name: cell(MIN_NONLIN, "val_r2", scope="variant", name=n),
            f"MPG probe {name} val R2")
        add(test, MIN_NONLIN,
            lambda n=name: cell(MIN_NONLIN, "test_r2", scope="variant", name=n),
            f"MPG probe {name} test R2")
    for column, quoted in [("minutes_per_game_lag1", ("0.0124", "0.0077")),
                           ("total_minutes_lag1", ("0.0008", "0.0011")),
                           ("age", ("−0.0008", "−0.0007")),
                           ("career_minutes_lag1", ("0.0005", "−0.0006"))]:
        add(quoted[0], MIN_NONLIN,
            lambda c=column: cell(MIN_NONLIN, "val_vs_linear", scope="column",
                                  name=c), f"{column} spline, val")
        add(quoted[1], MIN_NONLIN,
            lambda c=column: cell(MIN_NONLIN, "test_vs_linear", scope="column",
                                  name=c), f"{column} spline, test")

    # ── the workload ablation ─────────────────────────────────────────────────
    for variant, crps, r2 in [("baseline", "10.914", "0.268"),
                              ("plus_playoff_workload", "10.795", "0.283"),
                              ("plus_playoff_only", "10.817", "0.281"),
                              ("plus_career_minutes_only", "10.883", "0.271")]:
        add(crps, ABLATION, lambda v=variant: cell(ABLATION, "crps_games", variant=v),
            f"ablation {variant} CRPS")
        add(r2, ABLATION, lambda v=variant: cell(ABLATION, "r2_gp_share", variant=v),
            f"ablation {variant} R2")
    add("−0.119", ABLATION,
        lambda: (cell(ABLATION, "crps_games", variant="plus_playoff_workload")
                 - cell(ABLATION, "crps_games", variant="baseline")),
        "playoff-workload CRPS gain")

    # ── the season total, derived ─────────────────────────────────────────────
    add("316.9", SEASON_TOTAL, lambda: treatment("beta_binomial", "crps_dk_total"),
        "head season-total CRPS")
    add("340.8", SEASON_TOTAL, lambda: treatment("league_age", "crps_dk_total"),
        "baseline season-total CRPS")
    add("213.8", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 - treatment("oracle_gp", "mae_dk_total")),
        "availability's share of the remaining error")
    add("132.5", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 - treatment("oracle_rate", "mae_dk_total")),
        "the rate's share of the remaining error")
    add("−32.7%", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 / treatment("full_season", "mae_dk_total") - 1.0),
        "head vs a full season, relative")
    add("−39.9", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total")
                 - treatment("prior_gp", "mae_dk_total")),
        "head vs carrying prior GP forward")
    add("−23.3%", SEASON_TOTAL,
        lambda: (treatment("beta_binomial", "mae_dk_total", "rotation")
                 / treatment("full_season", "mae_dk_total", "rotation") - 1.0),
        "rotation-player gain, relative")

    # ── target dispersion and the first-k ladder ──────────────────────────────
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
    # Only the shipped construction is on disk. The other three ratios CLAUDE.md quotes
    # (2.01x raw per-season sd, 2.11x pooled, 1.84x ridge) were measured once and are
    # listed in `docs/provenance-plan.md` as prose-only.
    for quoted, label in [("1.105", "gross"), ("0.785", "net"), ("1.41", "ratio")]:
        add(quoted, OPPONENT_A, lambda: opp("dk_pts", "cancellation_ratio"),
            f"superseded opponent {label}", historical=True)

    # ── the ADP match audit ───────────────────────────────────────────────────
    def audit(metric_name: str, rule: str = "cascade") -> float:
        return _one(table(ADP_AUDIT), "value", section="summary", rule=rule,
                    metric=metric_name)

    add("3,591", ADP_AUDIT, lambda: audit("rows_audited"), "rows audited")
    add("0.50%", ADP_AUDIT, lambda: audit("unmatched_rate_cascade"),
        "cascade unmatched rate")
    add("0.00%", ADP_AUDIT,
        lambda: audit("unmatched_rate_surname_initial", "surname_initial"),
        "rejected rule's unmatched rate")
    add("57", ADP_AUDIT, lambda: rows(ADP_AUDIT), "match audit rows")

    # The season-term verdict, claimed against the SAME artifact `docs/predictions-plan.md`
    # claims it from. Both, deliberately: this file's copy of a block going stale while the
    # plan doc's stayed current is the exact failure that has already happened twice here.
    C += _season_term_summary_claims(CLAUDE)

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
    add("−0.833", SHOCK_CORR, lambda: _strongest_shock_pair(), "strongest shock pair")
    add("−0.009", SHOCK_CORR, lambda: table(SHOCK_CORR)["corr"].mean(),
        "mean shock correlation")
    add("0.307", SHOCK_CORR, lambda: table(SHOCK_CORR)["corr"].abs().mean(),
        "mean |shock correlation|")
    add("136", SHOCK_CORR, lambda: rows(SHOCK_CORR), "shock correlation pairs")
    add("20.6%", SHOCK_CORR,
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

    Its figures are a **selection** from `CLAUDE.md` and the plan docs rather than new
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
    approximate skill split (`~90%`). They are unclaimed in `CLAUDE.md` for the same
    reason, and coverage reports them rather than hiding them.
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

    # ── methods: what was deprioritized ───────────────────────────────────────
    add("0.0059", DIAGNOSTICS,
        lambda: cell(DIAGNOSTICS, "delta_sequence", analysis="sequence_ablation"),
        "sequence features over season aggregates")
    add("0.0074", DIAGNOSTICS,
        lambda: cell(DIAGNOSTICS, "delta_above_null", analysis="sequence_ablation"),
        "sequence features above their shuffled null")

    # ── results: availability ─────────────────────────────────────────────────
    for model, quoted in [("beta_binomial", "10.795"), ("gbm", "10.888"),
                          ("ridge", "10.896"), ("league_age", "13.614")]:
        add(quoted, METRICS, lambda m=model: metric(METRICS, m, "crps_games"),
            f"{model} CRPS")
    for name, which, quoted in [("full_season", "mae_dk_total", "646.3"),
                                ("beta_binomial", "mae_dk_total", "435.1"),
                                ("full_season", "bias_dk_total", "541.9"),
                                ("beta_binomial", "bias_dk_total", "6.1"),
                                ("oracle_gp", "mae_dk_total", "221.3"),
                                ("oracle_rate", "mae_dk_total", "302.7")]:
        add(quoted, SEASON_TOTAL, lambda n=name, w=which: treatment(n, w),
            f"season total {name} {which}")

    # ── results: the component floor ──────────────────────────────────────────
    count_heads = ("fg2a", "fg3a", "fta", "reb", "ast", "stl", "blk", "tov")
    fitted = ("linear", "log_own", "log_own_spline", "log_own_inter", "pca",
              "pca_spline", "pca_inter")
    add("0.82", RATES,
        lambda: min(rate(h, "carry_forward") for h in count_heads),
        "weakest no-fit floor across the count heads")
    add("0.0019", RATES,
        lambda: min(max(rate(h, v) for v in fitted) - rate(h, "carry_forward")
                    for h in count_heads),
        "smallest gain over the no-fit floor")
    add("0.0228", RATES,
        lambda: max(max(rate(h, v) for v in fitted) - rate(h, "carry_forward")
                    for h in count_heads),
        "largest gain over the no-fit floor")
    add("−19.00", STAN_C_M, lambda: stan_c("fg3a", "linear", "test_r2"),
        "fg3a under a linear predictor")
    add("0.679", STAN_C_M, lambda: stan_c("blk", "log_own", "test_r2"),
        "blk log_own R2, 3dp")
    add("0.858", STAN_C_M, lambda: stan_c("blk", "log_own_spline", "test_r2"),
        "blk spline R2, 3dp")
    # The two figures behind "the substitution arm's canonical side was handicapped":
    # `substitution_arm` fits every head at log_own, and that is the variant on which
    # `fg3a` fails its own floor. Claimed so the caveat cannot rot into a bare assertion.
    add("0.3719", STAN_C_M, lambda: stan_c("fg3a", "log_own", "test_r2"),
        "fg3a at log_own — the variant the substitution arm used")
    add("0.9046", STAN_C_M, lambda: stan_c("fg3a", "log_own_spline", "test_r2"),
        "fg3a at its selected spline variant")

    # ── results: the minutes composition ──────────────────────────────────────
    SEL = "betabinom_ot_graded"
    add("4.5078", COMP_M, lambda: comp_m(SEL, "test_crps"), "composition test CRPS")
    add("4.9140", COMP_M, lambda: comp_m("independent_comparator", "test_crps"),
        "independent comparator test CRPS")
    add("−8.3%", COMP_M,
        lambda: (comp_m(SEL, "test_crps")
                 / comp_m("independent_comparator", "test_crps") - 1.0),
        "composition CRPS gain, as a percentage")
    add("−0.406", COMP_M,
        lambda: comp_m(SEL, "test_crps") - comp_m("independent_comparator",
                                                  "test_crps"),
        "composition CRPS gain against the incumbent")
    add("36.87", COMP_P,
        lambda: cell(COMP_P, "simulated", variant=SEL, analysis="team_sum_abs_error",
                     group="independent"),
        "comparator team-sum error")
    add("0.194", COMP_M, lambda: comp_m("binomial", "test_pit_ks"),
        "binomial arm PIT KS, 3dp")
    add("0.148", COMP_RHO, lambda: cell(COMP_RHO, "rho", variant=SEL, bin=1),
        "fringe-tier dispersion, 3dp")
    add("0.061", COMP_RHO, lambda: cell(COMP_RHO, "rho", variant=SEL, bin=4),
        "star-tier dispersion, 3dp")
    add("2.41", COMP_RHO,
        lambda: (cell(COMP_RHO, "rho", variant=SEL, bin=1)
                 / cell(COMP_RHO, "rho", variant=SEL, bin=4)),
        "graded dispersion spread")
    add("59%", COMP_P,
        lambda: (1.0 - mean_abs_dev(COMP_P, "ratio", 1.0, variant=SEL,
                                    analysis="variance_ratio")
                 / mean_abs_dev(COMP_P, "ratio", 1.0, variant="betabinom_ot",
                                analysis="variance_ratio")),
        "calibration error cut by grading")

    # ── results: substitution and season terms ────────────────────────────────
    # Both splits, because the README's claim is that it *replicates* — quoting only the
    # test figure would leave the word "replicates" describing nothing auditable. The
    # handicapped pair stays claimed against the artifact it is still true of; the
    # un-handicapped pair is claimed against Gate 0's.
    for split, quoted in [("test", "−0.793"), ("val", "−0.771")]:
        add(quoted, STAN_C_S,
            lambda s=split: cell(STAN_C_S, "reparam_minus_canonical", split=s,
                                 arm="two_counts"),
            f"substitution gain, {split} (handicapped)")

    def shot_joint(split: str, arm: str) -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        sub = frame[(frame["analysis"] == "joint") & (frame["split"] == split)
                    & (frame["arm"] == arm) & frame["selected"].astype(bool)]
        return float(sub["mean_nll"].iloc[0])

    def shot_floor(name: str, variant: str) -> float:
        arm = "two_counts" if name in ("fg2a", "fg3a") else "fga_x_fg3a_share"
        return cell(SHOT_SWEEP, "floor_nll", analysis="head", split="test", arm=arm,
                    head=name, variant=variant)

    def shot_grid_min() -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        return float(frame[frame["analysis"] == "arm_a_grid"]["mean_nll"].min())

    for split, quoted in [("test", "−0.493549"), ("val", "−0.500782")]:
        add(quoted, SHOT_SWEEP,
            lambda s=split: (shot_joint(s, "fga_x_fg3a_share")
                             - shot_joint(s, "two_counts")),
            f"un-handicapped substitution gain, {split}")
    add("−0.491910", SHOT_SWEEP,
        lambda: shot_joint("test", "fga_x_fg3a_share") - shot_grid_min(),
        "substitution gain against arm A's best-of-16")
    add("11.024027", SHOT_SWEEP,
        lambda: shot_floor("fg2a", "log_own") + shot_floor("fg3a", "log_own_spline"),
        "canonical-basis no-fit floor")
    add("10.085599", SHOT_SWEEP,
        lambda: shot_floor("fga", "log_own") + shot_floor("fg3a|fga", "logit_own"),
        "reparameterized no-fit floor")
    add("−0.390814", SHOT_SWEEP,
        lambda: (shot_floor("fga", "log_own") + shot_floor("fg3a|fga", "logit_own")
                 - shot_grid_min()),
        "reparameterized floor vs canonical best fitted")
    add("−0.101096", SHOT_SWEEP,
        lambda: (shot_joint("test", "fga_x_fg3a_share")
                 - shot_floor("fga", "log_own") - shot_floor("fg3a|fga", "logit_own")),
        "what the reparameterized arm's own fitting adds")
    # The oracle bounds every form of season term, so both ends of the range the
    # README quotes are claimed — the "~3%" ceiling and the median.
    term_heads = ("fg2a", "fg3a", "fta", "reb", "ast", "stl", "blk", "tov")
    add("3%", TERM_M, lambda: max(oracle_gain(h) for h in term_heads),
        "the league-oracle ceiling")
    add("1.32%", TERM_M,
        lambda: float(pd.Series([oracle_gain(h) for h in term_heads]).median()),
        "the league-oracle median")
    add("19.0%", TERM_SPREAD,
        lambda: term_spread(15, "year") - 1.0,
        "year-effect roster spread at 15 players")
    add("0.2%", STAN_AV_B,
        lambda: cell(STAN_AV_B, "inflation", n_players=15) - 1.0,
        "shared-beta roster spread at 15 players")

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

    def head(split: str, arm: str, name: str, variant: str,
             column: str = "mean_nll") -> float:
        return cell(SHOT_SWEEP, column, analysis="head", split=split, arm=arm,
                    head=name, variant=variant)

    def joint(split: str, arm: str) -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        sub = frame[(frame["analysis"] == "joint") & (frame["split"] == split)
                    & (frame["arm"] == arm) & frame["selected"].astype(bool)]
        return float(sub["mean_nll"].iloc[0])

    def grid_best() -> float:
        frame = table(SHOT_SWEEP)
        if frame is None:
            return float("nan")
        return float(frame[frame["analysis"] == "arm_a_grid"]["mean_nll"].min())

    # ── the regression check, which is what licenses every other figure here ──
    add(_c("5.229492", SHOT_SWEEP, lambda: head("test", "two_counts", "fg2a", "log_own"),
           "fg2a@log_own reproduces the July fit", doc=SHOT))
    add(_c("5.248561", SHOT_SWEEP,
           lambda: head("test", "two_counts", "fg3a", "log_own_spline"),
           "fg3a@log_own_spline reproduces the July fit", doc=SHOT))
    add(_c("9.991042", SHOT_SWEEP,
           lambda: (head("test", "fga_x_fg3a_share", "fga", "log_own")
                    + head("test", "fga_x_fg3a_share", "fg3a|fga", "logit_own")),
           "refactored share arm reproduces the recorded joint NLL", doc=SHOT))

    # ── the per-factor table, both splits, every variant, against its floor ───
    per_factor = [
        ("two_counts", "fg2a", "log_own", "5.262971", "5.229492"),
        ("two_counts", "fg3a", "log_own_spline", "5.241963", "5.248561"),
        ("fga_x_fg3a_share", "fga", "linear", "5.407438", "5.426520"),
        ("fga_x_fg3a_share", "fga", "log_own", "5.390057", "5.379640"),
        ("fga_x_fg3a_share", "fga", "log_own_spline", "5.388533", "5.376766"),
        ("fga_x_fg3a_share", "fg3a|fga", "linear", "4.912507", "4.929134"),
        ("fga_x_fg3a_share", "fg3a|fga", "logit_own", "4.636033", "4.611402"),
        ("fga_x_fg3a_share", "fg3a|fga", "logit_own_spline", "4.615620", "4.607737"),
    ]
    for arm, name, variant, val, test in per_factor:
        for quoted, split in ((val, "val"), (test, "test")):
            add(_c(quoted, SHOT_SWEEP,
                   lambda s=split, a=arm, h=name, v=variant: head(s, a, h, v),
                   f"{h_label(arm)} {name}@{variant} {split} NLL", doc=SHOT))
    floors = [("fg2a", "log_own", "two_counts", "5.271028", "5.292842"),
              ("fg3a", "log_own_spline", "two_counts", "5.903396", "5.731185"),
              ("fga", "log_own", "fga_x_fg3a_share", "5.445210", "5.432802"),
              ("fg3a|fga", "logit_own", "fga_x_fg3a_share", "4.619109", "4.652797")]
    for name, variant, arm, val, test in floors:
        for quoted, split in ((val, "val"), (test, "test")):
            add(_c(quoted, SHOT_SWEEP,
                   lambda s=split, a=arm, h=name, v=variant: head(s, a, h, v,
                                                                  "floor_nll"),
                   f"{name} no-fit floor, {split}", doc=SHOT))

    # ── the verdict ──────────────────────────────────────────────────────────
    for split, canonical, reparam, margin in [
            ("val", "10.504935", "10.004153", "−0.500782"),
            ("test", "10.478052", "9.984503", "−0.493549")]:
        add(_c(canonical, SHOT_SWEEP, lambda s=split: joint(s, "two_counts"),
               f"arm A joint NLL, {split}", doc=SHOT))
        add(_c(reparam, SHOT_SWEEP, lambda s=split: joint(s, "fga_x_fg3a_share"),
               f"arm B joint NLL, {split}", doc=SHOT))
        add(_c(margin, SHOT_SWEEP,
               lambda s=split: joint(s, "fga_x_fg3a_share") - joint(s, "two_counts"),
               f"reparameterization margin, {split}", doc=SHOT))

    add(_c("10.476413", SHOT_SWEEP, grid_best, "arm A best-of-16", doc=SHOT))
    add(_c("−0.491910", SHOT_SWEEP,
           lambda: joint("test", "fga_x_fg3a_share") - grid_best(),
           "arm B against arm A's best-of-16", doc=SHOT))
    # The handicap decomposition: how much of the recorded margin was the straw man.
    add(_c("0.305646", SHOT_SWEEP,
           lambda: (head("test", "two_counts", "fg3a", "log_own_spline")
                    - cell(STAN_C_M, "test_nll", head="fg3a", variant="log_own")) * -1,
           "the handicap, in nats", doc=SHOT))
    add(_c("−0.487010", SHOT_SWEEP,
           lambda: (head("test", "fga_x_fg3a_share", "fga", "log_own")
                    + head("test", "fga_x_fg3a_share", "fg3a|fga", "logit_own")
                    - joint("test", "two_counts")),
           "margin before arm B was swept", doc=SHOT))
    add(_c("−0.006539", SHOT_SWEEP,
           lambda: (joint("test", "fga_x_fg3a_share")
                    - head("test", "fga_x_fg3a_share", "fga", "log_own")
                    - head("test", "fga_x_fg3a_share", "fg3a|fga", "logit_own")),
           "what sweeping arm B added", doc=SHOT))

    # ── the headline: the coordinate change beats the fitting ─────────────────
    def floor_total(arm: str, names: tuple[str, str], variants: tuple[str, str]) -> float:
        return sum(head("test", arm, n, v, "floor_nll")
                   for n, v in zip(names, variants))

    arm_a_floor = ("two_counts", ("fg2a", "fg3a"), ("log_own", "log_own_spline"))
    arm_b_floor = ("fga_x_fg3a_share", ("fga", "fg3a|fga"), ("log_own", "logit_own"))
    add(_c("11.024027", SHOT_SWEEP, lambda: floor_total(*arm_a_floor),
           "arm A no-fit floor total", doc=SHOT))
    add(_c("10.085599", SHOT_SWEEP, lambda: floor_total(*arm_b_floor),
           "arm B no-fit floor total", doc=SHOT))
    add(_c("−0.938427", SHOT_SWEEP,
           lambda: floor_total(*arm_b_floor) - floor_total(*arm_a_floor),
           "floor-to-floor gain from the coordinate change", doc=SHOT))
    add(_c("−0.390814", SHOT_SWEEP,
           lambda: floor_total(*arm_b_floor) - grid_best(),
           "arm B's floor against arm A's best fitted", doc=SHOT))
    add(_c("−0.101096", SHOT_SWEEP,
           lambda: joint("test", "fga_x_fg3a_share") - floor_total(*arm_b_floor),
           "what arm B's own fitting adds", doc=SHOT))

    # ── population and sampler ────────────────────────────────────────────────
    add(_c("791", SHOT_SWEEP,
           lambda: head("test", "two_counts", "fg2a", "log_own", "n"),
           "test rows", doc=SHOT))
    add(_c("773", SHOT_SWEEP,
           lambda: head("val", "two_counts", "fg2a", "log_own", "n"),
           "validation rows", doc=SHOT))
    add(_c("52", SHOT_SWEEP, lambda: rows(SHOT_SWEEP), "sweep artifact rows", doc=SHOT))
    add(_c("1.0087", SHOT_D, lambda: max_of(SHOT_D, "max_rhat"), "max R-hat", doc=SHOT))
    add(_c("0", SHOT_D, lambda: total(SHOT_D, "divergences"), "divergences", doc=SHOT))
    add(_c("46.5", SHOT_D, lambda: total(SHOT_D, "wall_clock_s") / 60,
           "sampler minutes", doc=SHOT))
    add(_c("16", SHOT_D, lambda: rows(SHOT_D), "fits", doc=SHOT))

    # The recorded (handicapped) margins, preserved beside their correction. These are
    # still exactly true of `stan_component_substitution.csv`, which this session does not
    # rebuild — so they are value-checked rather than flagged historical.
    for split, quoted in (("val", "−0.771"), ("test", "−0.793")):
        add(_c(quoted, STAN_C_S,
               lambda s=split: cell(STAN_C_S, "reparam_minus_canonical", split=s,
                                    arm="two_counts"),
               f"the recorded handicapped margin, {split}", doc=SHOT))
    add(_c("0.792657", STAN_C_S,
           lambda: -cell(STAN_C_S, "reparam_minus_canonical", split="test",
                         arm="two_counts"),
           "the recorded test margin, unsigned", doc=SHOT))
    add(_c("0.3719", STAN_C_M,
           lambda: cell(STAN_C_M, "test_r2", head="fg3a", variant="log_own"),
           "fg3a at log_own — the handicap", doc=SHOT))
    add(_c("0.9046", STAN_C_M,
           lambda: cell(STAN_C_M, "test_r2", head="fg3a", variant="log_own_spline"),
           "fg3a at its shipped spline", doc=SHOT))
    add(_c("−0.1248", RESID, lambda: cell(RESID, "r", basis="minutes_conditioned",
                             component_a="fg3a", component_b="fg2a"),
           "the substitution off-diagonal", doc=SHOT))
    add(_c("+0.0071", RESID, lambda: resid("mean"),
           "conditioned off-diagonal mean", doc=SHOT))
    add(_c("0.886", PERSIST, lambda: persist("sco_pct_fga_3pt"),
           "shot-mix share persistence", doc=SHOT))
    add(_c("731,906", GAME_LEN, lambda: cell(GAME_LEN, "player_games", analysis="feasibility",
                             season="all", season_type="regular"), "player-games", doc=SHOT))
    return C


def h_label(arm: str) -> str:
    return "arm A" if arm == "two_counts" else "arm B"


def _build() -> tuple[Claim, ...]:
    """Every claim, in doc order. One builder per doc — the registry is long enough that
    a single function made it hard to see which doc a section belonged to."""
    return tuple(_availability() + _composition() + _predictions() + _adp()
                 + _claude() + _readme() + _shot_basis())


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

    Historical claims are reported as `superseded` rather than checked — they are in the
    registry for the presence check, which is what stops a reversal being tidied away.
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
        actual = float(claim.actual())
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

    unbuilt = [f for f in results["skipped"] if f.check != "superseded"]
    if unbuilt:
        lines.append(f"Skipped (artifact not built): {len(unbuilt)}")
        for f in unbuilt[:10]:
            lines.append(f"  - {f.label}: {f.detail}")
        if len(unbuilt) > 10:
            lines.append(f"  … and {len(unbuilt) - 10} more")
        lines.append("")

    superseded = [f for f in results["skipped"] if f.check == "superseded"]
    checked = len(CLAIMS) - len(results["skipped"])
    lines.append("-" * 70)
    lines.append(f"figures checked:  {checked} of {len(CLAIMS)}")
    lines.append(f"superseded:       {len(superseded)} (presence-checked, not value-checked)")
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
