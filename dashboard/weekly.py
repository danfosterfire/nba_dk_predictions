"""The weekly-scores page's pure layer — Gate A at the unit a lineup is set at.

`make weekly-scores` (`src/sim/weekly.py`) writes six flat artifacts comparing observed
against simulated `dk_pts` **per player per scoring period**, on the training and
validation splits. This module reshapes them for the view and computes nothing a model
would compute: every metric, distance and band on the page was written by the emitter.

**The frames it returns are the same frames a model page's blocks 5 and 6 take**, which is
deliberate and is why this file is short. `charts.fig_ecdf`, `fig_calibration`, `fig_qq`
and `fig_quantile_residual` are reused **unmodified** — the emitter cut the artifacts with
`src/models/model_cards.py`'s own binning helpers, so the ribbon, the binned density, the
QQ-uniform and the rank-transformed residual on this page are the same objects and the same
encodings a reader has already learned on the four model pages. Only the key column
differs: a model card is keyed by `head` and these are keyed by `period_type`.

## The one thing this page has to say before any figure

**Three of the twenty scoring periods are not a week.** DK's Round 1 is seventeen weekly
periods and Rounds 2, 3 and 4 are two weeks each, which the simulator collapses onto twenty
tensor slots. So a "double week" carries roughly twice the games and roughly twice the
`dk_pts`, and pooling the two into one distribution would put a right tail on the picture
that is a calendar fact rather than a model. Every panel here is faceted by
`period_type` for that reason, and `PERIOD_ORDER` fixes which comes first.
"""

import numpy as np
import pandas as pd

# ── The artifacts, and the target that writes them ────────────────────────────

INDEX_FILE = "weekly_score_index.csv"
PERIOD_FILE = "weekly_score_period.csv"
ECDF_FILE = "weekly_score_ecdf.csv"
CALIBRATION_FILE = "weekly_score_calibration.csv"
QUANTILE_FILE = "weekly_score_quantile.csv"
SAMPLE_FILE = "weekly_score_sample.parquet"

MAKE_WEEKLY = "make weekly-scores"
MAKE_SIMULATE = "make simulate-season"

#: Gate A's own artifact, read for the season-total row this page is the counterpart to.
GATE_A_FILE = "sim_season_gate_a.csv"
SEASON_TOTAL_CHECK = "season_total_dk"

#: The closed split vocabulary, shared with `model_cards` so one page cannot render a
#: split another refuses. The emitter carves both through `held_out.selection_split` and
#: the test seasons are absent by construction, not by filtering.
SPLITS = ("train", "validation")
SPLIT_LABELS = {"train": "Train", "validation": "Validation"}

#: Facet order: the seventeen one-week periods first, because that is what "weekly" means
#: on this page and the double weeks are the exception it has to declare.
WEEK = "week"
DOUBLE_WEEK = "double_week"
PERIOD_ORDER = (WEEK, DOUBLE_WEEK)

#: Mirrors `src/sim/weekly.KS_MC_TOL` / `ECDF_BAND_TOL`, which mirror `model_cards`' own,
#: the way `model_cards.KS_MC_TOL` mirrors the emitter's — a test holds all of them
#: together. A threshold typed into a caption is a claim about a build that can move
#: without it.
KS_MC_TOL = 0.02
ECDF_BAND_TOL = 0.02

#: What a calibrated scaled residual puts its binned quartile lines at.
QUANTILE_LEVELS = (0.25, 0.5, 0.75)


# ── Block 1 · the unit ────────────────────────────────────────────────────────

def period_label(index: pd.DataFrame, period_type: str) -> str:
    part = index[index["period_type"] == period_type]
    return str(part["period_label"].iloc[0]) if len(part) else period_type


def period_types(index: pd.DataFrame) -> list[str]:
    """The facets present, in `PERIOD_ORDER`, then anything the artifact adds."""
    have = set(index["period_type"])
    ordered = [p for p in PERIOD_ORDER if p in have]
    return ordered + sorted(have - set(ordered))


def facet_row(index: pd.DataFrame, period_type: str, split: str) -> pd.Series | None:
    part = index[(index["period_type"] == period_type) & (index["split"] == split)]
    return part.iloc[0] if len(part) else None


def structure(index: pd.DataFrame) -> pd.DataFrame:
    """What each facet actually is — periods, weeks, seasons, players, team-games.

    The table the page owes before any distribution: **the two facets are not two equal
    units**, and a reader comparing their spreads without knowing that is being misled. The
    same argument the model pages make about the fitting unit, one level up.
    """
    rows = []
    for period_type in period_types(index):
        for split in SPLITS:
            row = facet_row(index, period_type, split)
            if row is None:
                continue
            rows.append({
                "Period": row["period_label"],
                "Split": SPLIT_LABELS[split],
                "Seasons": str(row["seasons"]).replace(",", ", "),
                "Periods per season": int(row["n_periods"]),
                "Weeks each": int(row["weeks"]),
                "Player-seasons": int(row["n_player_seasons"]),
                "Team-games scored": int(row["team_games"]),
                "Rows": int(row["n"]),
            })
    return pd.DataFrame(rows)


# ── Block 2 · Gate A at this unit ─────────────────────────────────────────────

def gate_board(index: pd.DataFrame) -> pd.DataFrame:
    """The marginal metric set, per facet — the same arithmetic Gate A's season row uses.

    `src/sim/season.marginal_metrics` computes both, which is what makes the season-total
    row below a comparison rather than two definitions side by side.
    """
    rows = []
    for period_type in period_types(index):
        for split in SPLITS:
            row = facet_row(index, period_type, split)
            if row is None:
                continue
            rows.append({
                "Unit": row["period_label"], "Split": SPLIT_LABELS[split],
                "n": int(row["n"]),
                "Observed mean": float(row["observed_mean"]),
                "Predicted mean": float(row["predicted_mean"]),
                "MAE": float(row["mae"]), "Bias": float(row["bias"]),
                "R²": float(row["r2"]), "CRPS": float(row["crps"]),
            })
    return pd.DataFrame(rows)


def spread_board(index: pd.DataFrame) -> pd.DataFrame:
    """Three model spreads against the observed one, because only one of them compares.

    A best-ball week is a **max over sixteen players**, so the weekly *spread* decides more
    of a lineup's score than the weekly mean does. The emitter ships three of them and the
    distinction is load-bearing: `point_sd` is the spread of the per-row posterior means and
    is narrower than the data by construction — a mean over draws has averaged its own noise
    away — so printing it beside the observed sd would report a model far too narrow when
    nothing of the sort has been measured. `pooled_sd` is the marginal the model implies
    over every row and draw at once, and is the one the observed sd answers.
    """
    rows = []
    for period_type in period_types(index):
        for split in SPLITS:
            row = facet_row(index, period_type, split)
            if row is None:
                continue
            observed = float(row["observed_sd"])
            rows.append({
                "Unit": row["period_label"], "Split": SPLIT_LABELS[split],
                "Observed sd": observed,
                "Simulated sd (pooled)": float(row["pooled_sd"]),
                "Ratio": float(row["pooled_sd"]) / observed if observed else float("nan"),
                "Per player-period": float(row["predictive_sd"]),
                "Point prediction": float(row["point_sd"]),
                "Zero weeks observed": float(row["zero_share"]),
                "Zero weeks simulated": float(row["predicted_zero_share"]),
            })
    return pd.DataFrame(rows)


def season_total_row(gate: pd.DataFrame | None) -> pd.DataFrame:
    """Gate A's own season-total `dk_pts` row, per season — this page's counterpart.

    Read rather than restated. The page's whole framing is "the same tensor, at a unit Gate
    A does not cover", and a reader cannot evaluate that without the row it is beside.
    """
    if gate is None or "check" not in gate.columns:
        return pd.DataFrame()
    part = gate[gate["check"] == SEASON_TOTAL_CHECK]
    if part.empty:
        return part
    return pd.DataFrame({
        "Season": part["season"].astype(str),
        "n": part["n"].astype(int),
        "MAE": part["mae"].astype(float),
        "Bias": part["bias"].astype(float),
        "R²": part["r2"].astype(float),
        "CRPS": part["crps"].astype(float),
    }).sort_values("Season").reset_index(drop=True)


# ── Block 3 · the per-period profile ──────────────────────────────────────────

def profile_panel(period: pd.DataFrame, split: str) -> pd.DataFrame:
    """Observed and simulated mean `dk_pts` per player, slot by slot, pooled over seasons.

    The facets above pool seventeen weeks into one distribution, which is right for a
    calibration panel and cannot answer "does the simulator drift through the season". This
    is that second question. Pooled across the split's seasons by a **row-weighted** mean
    rather than a mean of means, so a season with more scorable players carries more of the
    point — the emitter's per-season rows carry their own `n` for exactly that.
    """
    part = period[period["split"] == split]
    if part.empty:
        return pd.DataFrame()
    out = []
    for slot, rows in part.groupby("slot"):
        weight = rows["n"].to_numpy(dtype=float)
        out.append({
            "slot": int(slot),
            "tournament_round": int(rows["tournament_round"].iloc[0]),
            "period_type": str(rows["period_type"].iloc[0]),
            "label": slot_label(int(slot), int(rows["tournament_round"].iloc[0]),
                                str(rows["period_type"].iloc[0])),
            "observed": float(np.average(rows["observed_mean"], weights=weight)),
            "predicted": float(np.average(rows["predicted_mean"], weights=weight)),
            "bias": float(np.average(rows["bias"], weights=weight)),
            "mae": float(np.average(rows["mae"], weights=weight)),
            "games": float(np.average(rows["observed_games"], weights=weight)),
            "n": int(weight.sum()),
            "seasons": int(len(rows)),
        })
    return pd.DataFrame(out).sort_values("slot").reset_index(drop=True)


def slot_label(slot: int, tournament_round: int, period_type: str) -> str:
    """`W1`…`W17` for the Round-1 weeks, `R2`/`R3`/`R4` for the double weeks.

    The axis has to carry the structure, because the last three ticks are twice the unit of
    the first seventeen and a bare 1…20 would say the opposite.
    """
    if period_type == WEEK:
        return f"W{slot + 1}"
    return f"R{tournament_round}"


def profile_table(panel: pd.DataFrame) -> pd.DataFrame:
    """The figure's table twin — the relief rule, since three of its ticks change unit."""
    return pd.DataFrame({
        "Period": panel["label"], "Round": panel["tournament_round"],
        "Games played": panel["games"],
        "Observed mean": panel["observed"], "Simulated mean": panel["predicted"],
        "Bias": panel["bias"], "MAE": panel["mae"], "Rows": panel["n"],
    }).round({"Games played": 2, "Observed mean": 2, "Simulated mean": 2,
              "Bias": 2, "MAE": 2})


# ── Blocks 4-6 · the three panels a model page already draws ──────────────────

def ecdf_panel(ecdf: pd.DataFrame, period_type: str, split: str) -> pd.DataFrame:
    part = ecdf[(ecdf["period_type"] == period_type) & (ecdf["split"] == split)]
    return part.sort_values("grid_index").reset_index(drop=True)


def band_distance(ecdf: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """How far each facet's observed ECDF sits from the median replicate season.

    **The distance is the reading and in-or-out is not**, which is `model_cards`'
    `band_distance` rule and holds here for the same arithmetic reason: at n ≈ 10⁴ the
    ribbon is one to two ECDF points wide and every honest model leaves it somewhere.
    `max_gap` is in ECDF units — 0.04 means the observed curve is never more than four
    percentage points of probability from the median simulated season.
    """
    rows = []
    for period_type in period_types(index):
        for split in SPLITS:
            part = ecdf_panel(ecdf, period_type, split)
            if part.empty:
                continue
            gap = (part["observed"] - part["q50"]).abs()
            inside = ((part["observed"] >= part["q2.5"])
                      & (part["observed"] <= part["q97.5"]))
            rows.append({
                "period_type": period_type, "split": split,
                "label": period_label(index, period_type),
                "split_label": SPLIT_LABELS[split],
                "max_gap": float(gap.max()), "mean_gap": float(gap.mean()),
                "inside_95": float(inside.mean()), "n_grid": int(len(part)),
                "n_rows": int(part["n_rows"].iloc[0]),
                "n_draws": int(part["n_draws"].iloc[0]),
                "value_at_max": float(part.loc[gap.idxmax(), "value"]),
            })
    return pd.DataFrame(rows)


def calibration_panel(calibration: pd.DataFrame, period_type: str,
                      split: str) -> pd.DataFrame:
    """The binned density, in the exact frame shape `charts.fig_calibration` takes."""
    part = calibration[(calibration["period_type"] == period_type)
                       & (calibration["split"] == split)]
    if part.empty:
        return part
    out = part.copy()
    out["x_center"] = (out["x_left"] + out["x_right"]) / 2.0
    out["y_center"] = (out["y_left"] + out["y_right"]) / 2.0
    return out.sort_values(["x_index", "y_index"]).reset_index(drop=True)


def sample_points(sample: pd.DataFrame, period_type: str, split: str) -> pd.DataFrame:
    """The bounded overlay — `fitted`/`observed` for one panel, `u`/`predicted_rank` for
    the other, taken at the same thinned rows so a point is the same player-period in both.
    """
    part = sample[(sample["period_type"] == period_type) & (sample["split"] == split)]
    return part.reset_index(drop=True)


def _quantile_part(quantile: pd.DataFrame, period_type: str, panel: str,
                   split: str) -> pd.DataFrame:
    part = quantile[(quantile["period_type"] == period_type)
                    & (quantile["panel"] == panel) & (quantile["split"] == split)]
    return part.sort_values(["x_index", "y_index"]).reset_index(drop=True)


def qq_panel(quantile: pd.DataFrame, period_type: str, split: str) -> pd.DataFrame:
    """The QQ-uniform curve, renamed off the artifact's generic `(x, y)` grammar."""
    part = _quantile_part(quantile, period_type, "qq", split)
    if part.empty:
        return part
    return pd.DataFrame({
        "expected": part["x"].to_numpy(dtype=float),
        "observed": part["y"].to_numpy(dtype=float),
        "lo": part["lo"].to_numpy(dtype=float),
        "hi": part["hi"].to_numpy(dtype=float),
        "n": part["n"].to_numpy(dtype=int),
    })


def residual_cells(quantile: pd.DataFrame, period_type: str, split: str) -> pd.DataFrame:
    part = _quantile_part(quantile, period_type, "residual", split)
    if part.empty:
        return part
    out = part.copy()
    out["x_center"] = out["x"]
    out["y_center"] = out["y"]
    return out


def quantile_lines(quantile: pd.DataFrame, period_type: str, split: str) -> pd.DataFrame:
    part = _quantile_part(quantile, period_type, "quantile", split)
    return part.sort_values(["level", "x_index"]).reset_index(drop=True)


def quantile_distance(quantile: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """The KS distance per facet, and how far the quartile lines sit from their own levels.

    **A distance, never a verdict** — the rule the model pages carry, unchanged. `line_gap`
    is the second reading and the one that says *where*: a facet can sit close to uniform
    overall and still drift across the predicted range, which is exactly what a single KS
    cannot see and the rank-transformed panel exists to show.
    """
    rows = []
    for period_type in period_types(index):
        for split in SPLITS:
            part = _quantile_part(quantile, period_type, "qq", split)
            if part.empty:
                continue
            lines = quantile_lines(quantile, period_type, split)
            gap = (float((lines["y"] - lines["level"]).abs().max()) if len(lines)
                   else float("nan"))
            rows.append({
                "period_type": period_type, "split": split,
                "label": period_label(index, period_type),
                "split_label": SPLIT_LABELS[split],
                "ks": float(part["ks"].iloc[0]), "n": int(part["n"].iloc[0]),
                "line_gap": gap, "n_bins": int(lines["x_index"].nunique()),
            })
    return pd.DataFrame(rows)


# ── Panel dictionaries, in the shape the shared figures take ──────────────────

def calibration_keys(index: pd.DataFrame) -> dict[str, str]:
    """`fig_calibration`'s `panel_labels`: one **row** per facet, two split columns.

    The figure builds `len(panel_labels)` rows by the two splits and shares one axis range
    down each row, which is exactly the grid this page wants — the two facets are on
    different `dk_pts` scales and must not share an axis, and train and validation within a
    facet must.
    """
    return {period_type: period_label(index, period_type)
            for period_type in period_types(index)}


def provenance(index: pd.DataFrame) -> str:
    """One sentence naming what the ribbon is made of, for the caption under it."""
    if index.empty:
        return ""
    row = index.iloc[0]
    return (f"Each band is cut from {int(row['sim_draws']):,} whole simulated seasons — "
            f"thinned across the {int(row['n_sims']):,} the tensor carries, over "
            f"{int(row['n_posterior_draws']):,} posterior draws at the "
            f"`{row['fit_window']}` fit window, which is the only window that has not read "
            f"the seasons on the validation side.")
