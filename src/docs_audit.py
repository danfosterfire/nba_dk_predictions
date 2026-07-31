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


# ── Claims ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Claim:
    """One quoted figure, and where it comes from.

    `quoted` is the literal text in the doc — including its thousands separators, `%`
    or `×` — because that string is what the presence check looks for and what the
    precision is inferred from.
    """

    doc: str
    quoted: str
    artifact: str
    actual: Callable[[], float]
    label: str = ""
    tol: float | None = None

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
    """Read an artifact once, or return None if it has not been built."""
    if rel not in _CACHE:
        path = ROOT / rel
        _CACHE[rel] = pd.read_csv(path) if path.exists() else None
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


# ── The claim registry ────────────────────────────────────────────────────────

def _c(quoted: str, artifact: str, actual: Callable[[], float], label: str,
       doc: str = AVAIL, tol: float | None = None) -> Claim:
    return Claim(doc=doc, quoted=quoted, artifact=artifact, actual=actual,
                 label=label, tol=tol)


def _build() -> tuple[Claim, ...]:
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

    return tuple(C)


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
    """Quoted figure against artifact. Returns (mismatches, skipped)."""
    bad, skipped = [], []
    for claim in claims:
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
        lines.append(f"  {cov['claims']} claims covering "
                     f"{cov['covered']}/{cov['measurements']} measured figures "
                     f"({cov['share']:.0%}), out of {cov['numbers']} numeric "
                     f"literals in the prose")
    lines.append("")

    for check, heading in [("value-mismatch", "Figures that disagree with their artifact"),
                           ("stale-claim", "Claims whose text is no longer in the doc")]:
        found = results[check]
        lines.append(f"{heading}: {len(found)}")
        for f in found:
            lines.append(f"  - [{f.doc}] {f.label}: {f.detail}")
        lines.append("")

    if results["skipped"]:
        lines.append(f"Skipped (artifact not built): {len(results['skipped'])}")
        for f in results["skipped"][:10]:
            lines.append(f"  - {f.label}: {f.detail}")
        if len(results["skipped"]) > 10:
            lines.append(f"  … and {len(results['skipped']) - 10} more")
        lines.append("")

    checked = len(CLAIMS) - len(results["skipped"])
    lines.append("-" * 70)
    lines.append(f"figures checked:  {checked} of {len(CLAIMS)}")
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
