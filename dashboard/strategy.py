"""The tournament page's pure layer — contest structure, the sweep, the paired gaps.

Shapes the artifacts `make bracket` and `make strategy-sweep` wrote into the frames
`dashboard/views/tournament.py` draws. **No Streamlit import**, so every rule below is
exercised as a plain function in `tests/test_dashboard.py` rather than through a rendered
page, and no function here reads a file: the view opens the artifacts and hands frames in.

`dashboard/economics.py` is the sibling that owns the *contest* arithmetic — rake, the
break-even hurdle, the advance chain — derived from the two captured DraftKings CSVs. This
module never recomputes any of it; it joins that table to what the simulator measured.

## The one derivation that is this module's own, and why it needs one

The sweep selects on **lift in `P(top 2 of 12)`** and reports ROI alongside, for a measured
reason: `600k_shootaround` reaches Round 4 on 0.139% of entries, so ROI's Monte Carlo error
is enormous while a 16.67% event resolves orders of magnitude faster on the same budget.
That leaves the page with a units problem. The break-even hurdle is denominated in *return*
(+17.60% / +12.32%) and the sweep's headline is denominated in *survival*, so the hurdle
cannot simply be drawn on the lift axis.

`break_even_lift` converts it, under one stated assumption — that expected payout scales
with `P(advance)`. An exchangeable entry returns `1 − rake` of its fee at `p_null`, so
returning the whole fee needs `p_null / (1 − rake) = p_null·(1 + hurdle)`, i.e. a lift of
**`p_null · hurdle`**: +0.0293 at `600k_shootaround` and +0.0205 at `20k_spin_move`.

**The assumption is conservative, and that is measured rather than asserted.**
`payout_elasticity` reads the elasticity of the sweep's own ROI with respect to its own
survival, `log(payout ratio) / log(survival ratio)`, off every swept row of the selected
tier. Proportional means 1, and every tier reads a median well above it (5.40 at
`600k_shootaround` when first measured) — payout compounds through four cuts and the top
prize runs to 10,000×, so a strategy that survives twice as often is worth far more than
twice as much. The drawn line therefore sits *above* the lift a real break-even needs,
which is the direction a reference line should err in. The page prints the elasticity
beside the line rather than hiding the assumption inside it.
"""

import re

import numpy as np
import pandas as pd

# ── The artifacts, by the target that writes them ─────────────────────────────

STRUCTURE_FILE = "bracket_structure.csv"
SWEEP_FILE = "strategy_sweep.csv"
PAIRED_FILE = "strategy_paired.csv"
SHIPPED_FILE = "strategy_shipped.csv"
REALIZED_FILE = "strategy_realized.csv"

#: The field-robustness half of block 5. `draft_gate_b_need.csv` is the joint
#: (noise, need) calibration — it selects `need_weight = 0`, which is why the sweep
#: against a need-aware field had to *stipulate* its lean and says so in its suffix.
GATE_B_NEED_FILE = "draft_gate_b_need.csv"
SWEEP_NEED_FILE = "strategy_sweep_adp_need_w8.csv"
SHIPPED_NEED_FILE = "strategy_shipped_adp_need_w8.csv"

#: The pick-log stake readout — the 20 × $1 `15k_and_one` teams the pick-log plan would
#: enter, drafted by DK autodraft against the live optimizer on the same worlds.
PICK_LOG_STAKE_FILE = "strategy_pick_log_stake.csv"
PICK_LOG_PAIRED_FILE = "strategy_pick_log_paired.csv"
PICK_LOG_BASELINE = "autodraft_blend_a30"

MAKE_BRACKET = "make bracket"
MAKE_SWEEP = "make strategy-sweep"
MAKE_SIM_NEED = "make draft-sim-need"
MAKE_SWEEP_NEED = "make strategy-sweep-need"
MAKE_PICK_LOG = "make pick-log-stake"

#: The execution axis: the same opinion submitted as a static pre-draft board and
#: executed by DK's autodraft logic, against clicking every pick. The manual twin of the
#: autodraft arm, and the caps-only control that separates the executor from the caps.
EXECUTION_ARMS = ("autodraft_blend_a30", "blend_a30", "blend_caps_dk")

#: The two reference tiers — the pair the headline results are quoted at, and the pair
#: Gate D compares. Since 2026-08-11 the sweep drafts portfolios into **all five**
#: captured structures — the entry counts follow one stake-parity rule in
#: `configs/default.yaml` — so every tier carries a swept board and the strategy blocks
#: render for any of them. All stakes are simulated; no contest has been entered, and
#: which to enter is an open decision.
TARGET_TIERS = ("600k_shootaround", "20k_spin_move")

#: Facet order for the sweep, coarse-to-fine: who to rank by, how much market to blend,
#: whether that blend varies with the round, what the in-draft objective prices, then the
#: three portfolio-shape axes. Not a result order — a reading order.
AXIS_ORDER = ("ranking", "alpha", "alpha_by_round", "objective",
              "position_caps", "exposure", "stacking", "execution")

AXIS_LABELS = {
    "ranking": "Ranking source",
    "alpha": "Blend weight α",
    "alpha_by_round": "α by draft round",
    "objective": "In-draft objective",
    "position_caps": "Position caps",
    "exposure": "Exposure cap",
    "stacking": "Same-team stacking",
    "execution": "Execution (DK autodraft)",
}

METRIC_LABELS = {
    "p_advance": "P(top 2 of 12), per entry",
    "p_any_advance": "P(at least one entry advances), per portfolio",
}

#: What the two backtest halves are *for*. The distinction is the point of block 3: one
#: has unlimited resolution and is the only surface with the power to separate strategies,
#: the other has N = 2 seasons and cannot, and only one of them selected anything.
SURFACES = {
    "Simulated": "the tuning surface — seasons drawn from the posterior, "
                 "error-injected, and the only surface with the power to separate arms",
    "Realized": "the honest readout — the same portfolios replayed against real box "
                "scores. Two seasons, and it selected nothing",
}


def pretty_tournament(name: str) -> str:
    """`600k_shootaround` → `600k Shootaround`, leaving the buy-in tier lowercase.

    `str.title()` would render it `600K Shootaround`, which is not what DraftKings calls
    the contest and reads as a unit rather than a name.
    """
    return " ".join(word if re.fullmatch(r"\d+k", word) else word.capitalize()
                    for word in str(name).split("_"))


# ── Block 1 · the contest structure ───────────────────────────────────────────

def contest_summary(econ: pd.DataFrame, advance: pd.DataFrame) -> pd.DataFrame:
    """One row per tournament: the economics beside what surviving the ladder costs.

    `econ` is `economics.economics()` and `advance` is `economics.advance_table()`; both
    are derived from the captured DraftKings CSVs rather than from anything simulated.
    `p_reach_final` is the product of the advance rates, i.e. what an exchangeable entry's
    chance of reaching the final table is before any strategy is applied.
    """
    reach = (advance[advance["n_advance"] > 0]
             .groupby("tournament")["advance_rate"].prod().rename("p_reach_final"))
    rounds = advance.groupby("tournament")["round"].max().rename("n_rounds")
    first_cut = (advance[advance["round"] == 1]
                 .set_index("tournament")["advance_rate"].rename("r1_advance_rate"))
    out = (econ.set_index("tournament")
           .join([reach, rounds, first_cut]).reset_index())
    out["is_target"] = out["tournament"].isin(TARGET_TIERS)
    return out.sort_values("entry_fee_per_team").reset_index(drop=True)


def survival_frame(structure: pd.DataFrame, season: str) -> pd.DataFrame:
    """`P(reach round r)` for an exchangeable entry, one row per tournament × round.

    Read off `bracket_structure.csv`'s own analytic column rather than recomputed, so the
    curve on the page is the number `make bracket` checked its symmetric-field null
    against. The two are the same quantity, and having the page derive its own would be a
    second opinion nothing reconciles.
    """
    sub = structure[structure["season"].astype(str) == str(season)]
    if sub.empty:
        return pd.DataFrame(columns=["tournament", "round", "p_reach", "is_target"])
    out = sub[["tournament", "round", "p_reach_analytic", "cash_places",
               "n_advance", "pod_size"]].copy()
    out = out.rename(columns={"p_reach_analytic": "p_reach"})
    out["is_target"] = out["tournament"].isin(TARGET_TIERS)
    return out.sort_values(["tournament", "round"]).reset_index(drop=True)


def round_ladder(advance: pd.DataFrame, tournament: str) -> pd.DataFrame:
    """One tournament's four rounds as a table — pod, cut, cash, and the knockout flag."""
    sub = advance[advance["tournament"] == tournament].sort_values("round")
    out = pd.DataFrame({
        "Round": sub["round"].astype(int),
        "Field": sub["field_entries"].round().astype("Int64"),
        "Pod": sub["pod_size"].astype(int),
        "Advance": sub["n_advance"].astype(int),
        "Advance rate": sub["advance_rate"],
        "Cash places": sub["cash_places"].astype(int),
        "Min cash": sub["min_cash"],
        "Zero consolation": sub["zero_consolation"],
    })
    return out.reset_index(drop=True)


# ── Block 2 · the sweep, and the hurdle in the units the sweep resolves ───────

def break_even_lift(p_null: float, hurdle: float) -> float:
    """The lift in `P(advance)` that returns the entry fee, at proportional payout.

    An exchangeable entry advances at `p_null` and is worth exactly `1 − rake` of its fee.
    If expected payout scales with `P(advance)`, returning the whole fee needs
    `p_null · (1 + hurdle)`, since `1 + hurdle = 1/(1 − rake)` by construction in
    `economics.break_even_hurdle`. The lift is the difference.

    Proportional is the conservative end of the range this project measures — see
    `payout_elasticity`, which reads the sweep's own ROI as growing with survival at an
    exponent above 1 on every swept row.
    """
    p, h = float(p_null), float(hurdle)
    if not 0.0 < p < 1.0:
        raise ValueError(f"a null advance rate must sit in (0, 1); got {p_null!r}")
    if h <= -1.0:
        raise ValueError(f"a break-even hurdle must exceed −1; got {hurdle!r}")
    return p * h


def payout_elasticity(sweep: pd.DataFrame) -> pd.DataFrame:
    """How fast the sweep's own ROI grows with its own survival, per tournament.

    `log(payout ratio) / log(survival ratio)`, each ratio taken against the symmetric-field
    null the same row carries. 1 is proportional — the assumption `break_even_lift` makes —
    and anything above it means the drawn reference line is a conservative one.

    Rows that cannot form the ratio are dropped rather than clipped: a strategy sitting
    exactly on the null gives `log(1) = 0` in the denominator, and a payout ratio at or
    below zero has no logarithm.
    """
    frame = sweep.copy()
    survival = frame["p_advance"] / frame["p_advance_null"]
    payout = (1.0 + frame["roi"]) / (1.0 + frame["roi_null"])
    usable = (survival > 0) & (payout > 0) & (np.abs(survival - 1.0) > 1e-9)
    frame = frame[usable].copy()
    frame["elasticity"] = (np.log(payout[usable].to_numpy())
                           / np.log(survival[usable].to_numpy()))
    grouped = frame.groupby("tournament")["elasticity"]
    return pd.DataFrame({
        "tournament": grouped.median().index,
        "median_elasticity": grouped.median().to_numpy(),
        "share_above_proportional": grouped.apply(lambda s: float((s > 1.0).mean())
                                                  ).to_numpy(),
        "n_rows": grouped.size().to_numpy(),
    }).reset_index(drop=True)


def sweep_panel(sweep: pd.DataFrame, tournament: str) -> pd.DataFrame:
    """The swept arms for one tournament, ordered for a facet-by-axis dot plot.

    Facets follow `AXIS_ORDER`; inside a facet, arms are ordered by their mean lift across
    the swept seasons so the best arm on each axis is the top row of its own block. The
    ordering is recomputed per tournament rather than fixed, because the two tiers disagree
    about which `α` wins and a shared order would hide that.
    """
    sub = sweep[sweep["tournament"] == tournament].copy()
    if sub.empty:
        return sub.assign(axis_label=pd.Series(dtype=str),
                          axis_rank=pd.Series(dtype=int),
                          strategy_rank=pd.Series(dtype=int))
    order = {axis: i for i, axis in enumerate(AXIS_ORDER)}
    sub["axis_rank"] = sub["axis"].map(order).fillna(len(order)).astype(int)
    sub["axis_label"] = sub["axis"].map(AXIS_LABELS).fillna(sub["axis"])
    mean_lift = sub.groupby("strategy")["lift_vs_null"].mean()
    sub["strategy_rank"] = sub["strategy"].map(-mean_lift)
    return sub.sort_values(["axis_rank", "strategy_rank", "season"]).reset_index(drop=True)


def facets(panel: pd.DataFrame) -> list[tuple[str, str, list[str]]]:
    """`(axis, label, arms top-to-bottom)` per facet, in the panel's own row order."""
    out = []
    for axis in panel["axis"].drop_duplicates():
        block = panel[panel["axis"] == axis]
        arms = list(dict.fromkeys(block["strategy"]))
        out.append((axis, str(block["axis_label"].iloc[0]), arms))
    return out


# ── Block 3 · the two backtest surfaces ───────────────────────────────────────

def shipped_arm(shipped: pd.DataFrame, tournament: str) -> pd.Series | None:
    """The row `make strategy-sweep` froze for this tier, or None if it shipped none."""
    sub = shipped[shipped["tournament"] == tournament]
    return None if sub.empty else sub.iloc[0]


def surfaces_panel(sweep: pd.DataFrame, realized: pd.DataFrame, tournament: str,
                   arms: tuple[str, ...]) -> pd.DataFrame:
    """The same arms on both backtest surfaces, one row per surface × season × arm.

    The two halves are put on one `P(top 2 of 12)` axis on purpose. They differ by three
    orders of magnitude in resolution — the simulated side pools 500 drawn worlds per
    season and the realized side has exactly one — and an interval is the only honest way
    to show that. Reading the two point estimates against each other without the widths
    beside them is the mistake the block exists to prevent.
    """
    rows = []
    for surface, frame in (("Simulated", sweep), ("Realized", realized)):
        sub = frame[frame["tournament"] == tournament]
        for season in sorted(sub["season"].astype(str).unique()):
            for arm in arms:
                hit = sub[(sub["season"].astype(str) == season)
                          & (sub["strategy"] == arm)]
                if hit.empty:
                    continue
                row = hit.iloc[0]
                rows.append({
                    "surface": surface, "season": season, "strategy": arm,
                    "label": f"{surface} · {season}",
                    "p_advance": float(row["p_advance"]),
                    "lo": float(row["p_advance_lo"]), "hi": float(row["p_advance_hi"]),
                    "lift": float(row["lift_vs_null"]),
                    "p_null": float(row["p_advance_null"]),
                    "width": float(row["p_advance_hi"]) - float(row["p_advance_lo"]),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    surface_rank = {"Simulated": 0, "Realized": 1}
    out["surface_rank"] = out["surface"].map(surface_rank)
    return out.sort_values(["surface_rank", "season"]).reset_index(drop=True)


def resolution_gap(panel: pd.DataFrame) -> dict[str, float]:
    """How much wider the honest readout's intervals are than the tuning surface's.

    The single number that says why one half selected the strategy and the other did not.
    """
    widths = panel.groupby("surface")["width"].mean()
    sim, real = float(widths.get("Simulated", np.nan)), float(widths.get("Realized", np.nan))
    return {"simulated": sim, "realized": real,
            "ratio": real / sim if sim else float("nan")}


# ── Block 4 · the paired comparisons ──────────────────────────────────────────

def paired_panel(paired: pd.DataFrame, tournament: str, metric: str,
                 baseline: str) -> pd.DataFrame:
    """Every arm against one baseline, sorted by gap, with the unresolved ones flagged.

    **`crosses_zero` is derived from the interval this chart draws, not read from the
    artifact's `resolved` column.** The two agree today and a test pins that they do — but
    the styling has to follow the bar the reader is looking at, or a flag that drifted from
    its own interval would put a filled marker on a gap the chart visibly shows straddling
    zero.

    The self-comparison is dropped. A baseline against itself is a gap of exactly zero with
    a zero-width interval, which is arithmetic rather than a result, and leaving it in
    would put a row in the "does not resolve" count that never could.
    """
    sub = paired[(paired["tournament"] == tournament)
                 & (paired["metric"] == metric)
                 & (paired["baseline"] == baseline)
                 & (paired["strategy"] != baseline)].copy()
    if sub.empty:
        return sub.assign(crosses_zero=pd.Series(dtype=bool))
    sub["crosses_zero"] = (sub["gap_lo"] <= 0.0) & (sub["gap_hi"] >= 0.0)
    return sub.sort_values("gap", ascending=False).reset_index(drop=True)


def unresolved(panel: pd.DataFrame) -> tuple[int, int]:
    """`(gaps whose interval crosses zero, gaps in the panel)`."""
    return int(panel["crosses_zero"].sum()), int(len(panel))


# ── Block 5 · the field, and the execution ────────────────────────────────────

def need_calibration(gate_b_need: pd.DataFrame, probe_weight: float = 8.0) -> dict:
    """What the joint (noise, need) calibration found, as the numbers block 5 prints.

    The selected row is the fitted answer — `need_weight = 0` today, meaning the observed
    ADP curve carries no slot-reaching for the field model to imitate. The probe row is
    the stipulated lean the robustness sweep ran at, read from the same pooled grid so the
    two `mae_fit` figures are on identical draws. `mae_elite` rides along because the
    elite region is where a need lean degrades fastest, which is the evidence the fitted
    zero rests on.
    """
    pooled = gate_b_need[gate_b_need["season"] == "pooled"]
    if pooled.empty or not pooled["selected"].any():
        raise ValueError("draft_gate_b_need.csv carries no pooled selected row — "
                         "rerun `make draft-sim-need`")
    selected = pooled[pooled["selected"]].iloc[0]
    at_probe = pooled[pooled["need_weight"] == float(probe_weight)]
    probe = (at_probe.loc[at_probe["mae_fit"].idxmin()] if len(at_probe) else None)
    return {
        "fitted_need_weight": float(selected["need_weight"]),
        "rank_noise_sd": float(selected["rank_noise_sd"]),
        "mae_fit": float(selected["mae_fit"]),
        "mae_elite": float(selected["mae_elite"]),
        "passes": bool(selected["passes"]),
        "probe_weight": float(probe_weight) if probe is not None else float("nan"),
        "probe_mae_fit": float(probe["mae_fit"]) if probe is not None else float("nan"),
        "probe_mae_elite": (float(probe["mae_elite"]) if probe is not None
                            else float("nan")),
    }


def field_comparison(sweep: pd.DataFrame, sweep_need: pd.DataFrame, tournament: str,
                     arms: tuple[str, ...]) -> pd.DataFrame:
    """The same arms against the two fields, pooled over seasons — block 5's left half.

    One row per field × arm, with the interval endpoints averaged the same way the lift
    is, so the frame draws directly. The need field's rows exist only if its sweep was
    run; the caller decides what absence means.
    """
    rows = []
    for field, frame in (("Fitted ADP field", sweep),
                         ("Need-aware probe (w = 8)", sweep_need)):
        sub = frame[frame["tournament"] == tournament]
        for arm in arms:
            hit = sub[sub["strategy"] == arm]
            if hit.empty:
                continue
            rows.append({
                "field": field, "strategy": arm,
                "lift": float(hit["lift_vs_null"].mean()),
                "lo": float(hit["lift_lo"].mean()), "hi": float(hit["lift_hi"].mean()),
                "roi": float(hit["roi"].mean()),
                "n_seasons": int(hit["season"].nunique()),
            })
    return pd.DataFrame(rows)


def execution_panel(sweep: pd.DataFrame, tournament: str,
                    arms: tuple[str, ...] = EXECUTION_ARMS) -> pd.DataFrame:
    """The execution arms' pooled lifts for one tier — block 5's right half."""
    sub = sweep[sweep["tournament"] == tournament]
    rows = []
    for arm in arms:
        hit = sub[sub["strategy"] == arm]
        if hit.empty:
            continue
        rows.append({"strategy": arm,
                     "lift": float(hit["lift_vs_null"].mean()),
                     "lo": float(hit["lift_lo"].mean()),
                     "hi": float(hit["lift_hi"].mean())})
    return pd.DataFrame(rows)


def pick_log_panel(levels: pd.DataFrame, paired: pd.DataFrame,
                   baseline: str = PICK_LOG_BASELINE
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The pick-log stake, shaped for block 5: pooled levels and the paired gaps.

    Levels are pooled over the swept seasons per arm, with the portfolio's expected
    payout stated in dollars beside the stake so the two can be read against each other.
    Gaps come from the artifact already paired on the world; the baseline's
    self-comparison rows are dropped, for `paired_panel`'s reason — a gap of exactly
    zero with a zero-width interval is arithmetic, not a result.
    """
    pooled = (levels.groupby("strategy", as_index=False)
              .agg(autodraft=("autodraft", "first"),
                   n_entries=("n_entries", "first"),
                   entry_fee=("entry_fee", "first"),
                   stake=("stake", "first"),
                   p_advance=("p_advance", "mean"),
                   p_any_advance=("p_any_advance", "mean"),
                   ev_per_entry=("ev", "mean"),
                   n_seasons=("season", "nunique")))
    pooled["portfolio_ev"] = pooled["ev_per_entry"] * pooled["n_entries"]
    pooled = pooled.sort_values("p_any_advance", ascending=False).reset_index(drop=True)

    gaps = paired[paired["strategy"] != baseline].copy()
    gaps["crosses_zero"] = (gaps["gap_lo"] <= 0.0) & (gaps["gap_hi"] >= 0.0)
    return pooled, gaps


def pick_log_cost(gaps: pd.DataFrame, arm: str) -> dict[str, float]:
    """One arm's gaps over the autodraft baseline, keyed by metric — the tile values."""
    sub = gaps[gaps["strategy"] == arm]
    out: dict[str, float] = {}
    for row in sub.itertuples():
        out[str(row.metric)] = float(row.gap)
        out[f"{row.metric}_lo"] = float(row.gap_lo)
        out[f"{row.metric}_hi"] = float(row.gap_hi)
        out[f"{row.metric}_resolved"] = bool(not row.crosses_zero)
    return out


def autodraft_matches_caps(sweep: pd.DataFrame) -> bool:
    """Does the autodraft twin reproduce the caps-only arm exactly, everywhere?

    The identity block 5 leans on: executing a static ranking through DK's autodraft is
    the same computation as clicking it under DK's position caps, so the two arms must
    hold identical rosters and identical lifts on every tier × season. They reach that
    equality through two separate code paths, which is why it is checked from the
    artifact rather than assumed from the code.
    """
    a = sweep[sweep["strategy"] == "autodraft_blend_a30"]
    b = sweep[sweep["strategy"] == "blend_caps_dk"]
    if a.empty or b.empty or len(a) != len(b):
        return False
    keys = ["season", "tournament"]
    merged = a.set_index(keys)[["lift_vs_null"]].join(
        b.set_index(keys)[["lift_vs_null"]], rsuffix="_caps", how="inner")
    # Exact equality on purpose: the two arms hold the same rosters, so their scored
    # lifts are the same float, not merely close.
    return bool(len(merged) == len(a)
                and (merged["lift_vs_null"].to_numpy()
                     == merged["lift_vs_null_caps"].to_numpy()).all())
