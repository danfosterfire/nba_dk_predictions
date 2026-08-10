"""Tournament & strategy — the contest, the sweep, the two backtests, the paired gaps.

The page the project builds toward, and the one that needed no new pipeline work: every
figure on it reads something `make bracket` or `make strategy-sweep` had already written
and nothing had ever drawn. Four blocks, in the order a reader needs them:

1. **The contest structure.** Five real DraftKings best-ball tournaments, four elimination
   rounds, Round 1 a zero-consolation knockout in every one of them, and rake expressed as
   a break-even edge hurdle — because that is the unit a *measured* edge compares in.
2. **The sweep.** Twenty-two strategies over seven axes, drawn as lift over the
   symmetric-field null with intervals, faceted by axis, with the hurdle converted into
   the survival units the sweep resolves in (`strategy.break_even_lift`).
3. **Simulated versus realized.** The same portfolios on the tuning surface and on the
   honest readout, on one axis, so their *widths* can be compared. The block's job is to
   make it impossible to read the second as a confirmation of the first.
4. **The paired comparisons.** The unpaired intervals in block 2 cover most of the table
   because a kind simulated season is kind to every arm at once; differencing inside the
   world removes that, and then almost everything separates. **The arms that still do not
   are the useful ones**, and they are styled rather than buried.

Every number is read from an artifact. The pure layer is `dashboard/strategy.py` and the
contest arithmetic is `dashboard/economics.py`; nothing here computes a result, and
nothing in this package imports `src/`.

Replaced `dashboard/views/placeholder.py`, which held this row in `app.VIEWS` from the
multipage shell landing on 2026-08-10 until this page did.
"""

import pandas as pd
import streamlit as st

from dashboard import economics, shell, strategy
from dashboard.artifacts import ROOT, optional, predictions_dir, rel
from dashboard.charts import (fig_hurdle, fig_paired, fig_survival, fig_surfaces,
                              fig_sweep)

#: The market arm every block compares against. `adp` is not just another strategy — it
#: is what the field actually does, so it is the baseline a measured edge is an edge over.
MARKET_ARM = "adp"
#: The paired block's default baseline: the model's own point ranking, i.e. "what does
#: each axis buy over ranking players by their posterior mean?"
DEFAULT_BASELINE = "model_mean"


# ── Loading ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Reading the strategy artifacts…")
def load_strategy() -> dict[str, pd.DataFrame] | None:
    """The four `make strategy-sweep` artifacts blocks 2–4 draw."""
    directory = predictions_dir()
    frames = {}
    for key, name in (("sweep", strategy.SWEEP_FILE),
                      ("paired", strategy.PAIRED_FILE),
                      ("shipped", strategy.SHIPPED_FILE),
                      ("realized", strategy.REALIZED_FILE)):
        frame = optional(directory / name, target=strategy.MAKE_SWEEP)
        if frame is None:
            return None
        frames[key] = frame
    return frames


@st.cache_data(show_spinner="Reading the bracket artifacts…")
def load_contest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """`bracket_structure.csv` plus the two contest tables `economics.py` derives.

    The economics half is checked by path rather than caught by exception: a missing
    capture should name the file that is missing, not surface as a `FileNotFoundError`
    from inside a module whose whole job is arithmetic.
    """
    structure = optional(predictions_dir() / strategy.STRUCTURE_FILE,
                         target=strategy.MAKE_BRACKET)
    if structure is None:
        return None
    for source in (economics.METADATA, economics.PRIZES):
        if not (ROOT / source).exists():
            st.warning(f"`{source}` not found — the captured DraftKings tournament "
                       "tables are raw data, not a build product.")
            return None
    return structure, economics.economics(), economics.advance_table()


# ── Block 1 · the contest structure ───────────────────────────────────────────

def contest_tiles(row: pd.Series) -> None:
    tiles = [
        ("Entry fee", f"${row['entry_fee_per_team']:,.0f}", "Per entry, per season"),
        ("Field", f"{row['total_entries']:,.0f}",
         "Entries in the captured running of this tournament"),
        ("Rake", f"{row['rake']:.2%}",
         "Share of the buy-in pool the house keeps"),
        ("Break-even edge", f"{row['break_even_hurdle']:+.2%}",
         "1/(1−rake) − 1 — the edge that just returns the entry fee. Not the same "
         "number as the rake, and the units a measured edge compares in"),
        ("First prize", f"{row['first_prize_multiple']:,.0f}×",
         "Top prize as a multiple of the entry fee"),
        ("Reaches Round 4", f"{row['p_reach_final']:.3%}",
         "An exchangeable entry, before any strategy"),
    ]
    for col, (label, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, help=helptext)


def contest_block(structure: pd.DataFrame, econ: pd.DataFrame, advance: pd.DataFrame,
                  tournament: str, th: dict) -> None:
    summary = strategy.contest_summary(econ, advance)
    row = summary[summary["tournament"] == tournament]
    if len(row):
        contest_tiles(row.iloc[0])

    season = sorted(structure["season"].astype(str).unique())[-1]
    survival = strategy.survival_frame(structure, season)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.plotly_chart(
            fig_survival(survival, th, strategy.TARGET_TIERS,
                         strategy.pretty_tournament),
            width="stretch", key="survival", config={"displayModeBar": False})
        st.caption(
            "**Round 1 pays nothing at all.** Five of every six entries are gone in a "
            "round with no consolation, in all five captured structures — so "
            "`P(any return)` *is* `P(top 2 of 12)`, and everything past Round 1 sets "
            "the size of the return rather than its sign. The two tiers the sweep "
            "drafts into are highlighted; the other three are drawn but not swept. "
            "**20k Spin Move and 88k Alley Oop advance identically** (2 of 12, then 2 "
            "of 6, then 2 of 6), so their curves coincide exactly and the highlighted "
            "one is drawn on top — the table below lists both.")
    with right:
        st.plotly_chart(
            fig_hurdle(summary, th, strategy.TARGET_TIERS, strategy.pretty_tournament),
            width="stretch", key="hurdle", config={"displayModeBar": False})
        st.caption(
            "Rake in the units an edge is measured in. Losing 14.97% of the pool means "
            "beating the field by **17.60%** to break even, not by 14.97% — "
            "`1/(1−rake) − 1`. Every reference line further down this page comes from "
            "this column.")

    with st.expander("Table view — the round ladder, and the five captured contests"):
        st.markdown(f"**{strategy.pretty_tournament(tournament)}** · the four rounds")
        st.dataframe(
            strategy.round_ladder(advance, tournament), hide_index=True,
            width="stretch",
            column_config={
                "Advance rate": st.column_config.NumberColumn(format="%.3f"),
                "Min cash": st.column_config.NumberColumn(format="$%.2f")})
        st.markdown(f"**Every captured tournament** · survival to each round, {season}")
        st.dataframe(_survival_table(survival), hide_index=True, width="stretch")


def _survival_table(survival: pd.DataFrame) -> pd.DataFrame:
    wide = survival.pivot(index="tournament", columns="round", values="p_reach")
    wide.columns = [f"Reaches R{c}" for c in wide.columns]
    wide.index = [strategy.pretty_tournament(t) for t in wide.index]
    return wide.reset_index(names="Tournament")


# ── Block 2 · the sweep ───────────────────────────────────────────────────────

def sweep_block(sweep: pd.DataFrame, tournament: str, th: dict) -> None:
    panel = strategy.sweep_panel(sweep, tournament)
    if panel.empty:
        st.info("The sweep carries no rows for this tournament.")
        return
    hurdle = float(panel["break_even_hurdle"].iloc[0])
    p_null = float(panel["p_advance_null"].iloc[0])
    lift = strategy.break_even_lift(p_null, hurdle)
    elasticity = strategy.payout_elasticity(sweep)
    elastic = elasticity[elasticity["tournament"] == tournament]

    st.plotly_chart(
        fig_sweep(panel, strategy.facets(panel), th, lift),
        width="stretch", key="sweep", config={"displayModeBar": False})

    st.caption(
        f"**Selection is lift in `P(top 2 of 12)`, and that is a measurement decision.** "
        f"ROI is dominated by the rare deep runs block 1 priced, so its Monte Carlo error "
        f"is enormous; a {p_null:.2%} event resolves orders of magnitude faster on the "
        f"same budget. Since surviving Round 1 is exactly the condition for any return at "
        f"all, this is both the statistic the sweep can resolve and a defensible "
        f"headline. ROI rides alongside it in the table below and is not a level to act "
        f"on.")
    st.caption(
        f"**The break-even line is the hurdle converted into these units, under one "
        f"stated assumption.** An exchangeable entry advances at {p_null:.4f} and returns "
        f"`1 − rake`; if expected payout scaled with `P(advance)`, returning the whole "
        f"fee would need a lift of `{p_null:.4f} × {hurdle:.4f}` = **{lift:+.4f}**. "
        + (f"Payout does **not** scale proportionally, and the direction is the safe one: "
           f"the elasticity of this sweep's own ROI with respect to its own survival has "
           f"a median of **{float(elastic['median_elasticity'].iloc[0]):.2f}** and sits "
           f"above 1 on "
           f"{float(elastic['share_above_proportional'].iloc[0]):.0%} of "
           f"{int(elastic['n_rows'].iloc[0])} swept rows, so the drawn line is *above* "
           f"the lift a real break-even needs."
           if len(elastic) else ""))
    st.caption(
        "**These intervals are unpaired and will cover most of the table.** That is a "
        "property of the level rather than of the differences — a simulated season kind "
        "to one arm is kind to all of them, since every entry is scored on the same "
        "drawn world. Block 4 is where arms are separated from each other; this block is "
        "where they are separated from the null.")

    with st.expander("Table view — every swept arm"):
        st.dataframe(_sweep_table(panel), hide_index=True, width="stretch")


def _sweep_table(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Axis": panel["axis_label"],
        "Arm": panel["strategy"],
        "Season": panel["season"],
        "Lift": panel["lift_vs_null"],
        "Low": panel["lift_lo"],
        "High": panel["lift_hi"],
        "P(top 2 of 12)": panel["p_advance"],
        "ROI": panel["roi"],
        "Entries": panel["n_entries"],
    })
    return out.round({"Lift": 4, "Low": 4, "High": 4, "P(top 2 of 12)": 4, "ROI": 2})


# ── Block 3 · the two backtest surfaces ───────────────────────────────────────

def surfaces_block(sweep: pd.DataFrame, realized: pd.DataFrame, shipped: pd.DataFrame,
                   tournament: str, th: dict) -> None:
    arm = strategy.shipped_arm(shipped, tournament)
    if arm is None:
        st.info("No strategy was frozen for this tournament.")
        return
    arms = (str(arm["strategy"]), MARKET_ARM)
    panel = strategy.surfaces_panel(sweep, realized, tournament, arms)
    if panel.empty:
        st.info("Neither backtest surface carries this tournament.")
        return

    tiles = [
        ("Shipped arm", str(arm["strategy"]), f"Selected on {arm['selected_on']}"),
        ("Simulated lift", f"{float(arm['sim_lift']):+.3f}",
         "Pooled over the swept seasons, on the tuning surface"),
        ("Realized lift", f"{float(arm['realized_lift']):+.3f}",
         f"Against real box scores, {int(arm['realized_seasons'])} seasons"),
        ("Interval width", f"{strategy.resolution_gap(panel)['ratio']:.1f}×",
         "How much wider the realized intervals are than the simulated ones — the "
         "reason one surface selected the strategy and the other did not"),
    ]
    for col, (label, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, help=helptext)

    st.plotly_chart(
        fig_surfaces(panel, th, float(panel["p_null"].iloc[0]), arms),
        width="stretch", key="surfaces", config={"displayModeBar": False})

    left, right = st.columns(2, gap="large")
    left.caption(f"**Simulated** — {strategy.SURFACES['Simulated']}.")
    right.caption(f"**Realized** — {strategy.SURFACES['Realized']}.")
    st.caption(
        "The two halves share an axis so their **widths** can be compared, which is the "
        "point of putting them together. The realized intervals resample the field and "
        "the entries, never the season: a season cannot be resampled and there are two "
        "of them. So the honest statement is that the realized edge is **not "
        "distinguishable from zero** — what the readout is for is catching a strategy "
        "broken in a way the simulated world cannot see, and nothing here is broken.")

    with st.expander("Table view — both surfaces"):
        st.dataframe(_surfaces_table(panel), hide_index=True, width="stretch")


def _surfaces_table(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Surface": panel["surface"], "Season": panel["season"],
        "Arm": panel["strategy"], "P(top 2 of 12)": panel["p_advance"],
        "Low": panel["lo"], "High": panel["hi"],
        "Interval width": panel["width"], "Lift": panel["lift"],
    })
    return out.round(4)


# ── Block 4 · the paired comparisons ──────────────────────────────────────────

def paired_block(paired: pd.DataFrame, tournament: str, th: dict) -> None:
    baselines = sorted(paired["baseline"].unique())
    metrics = [m for m in strategy.METRIC_LABELS if m in set(paired["metric"])]
    pick_baseline, pick_metric, _ = st.columns([1.2, 1.6, 2])
    baseline = pick_baseline.selectbox(
        "Baseline", baselines,
        index=baselines.index(DEFAULT_BASELINE) if DEFAULT_BASELINE in baselines else 0,
        help="Every arm is differenced against this one inside each simulated season.")
    metric = pick_metric.selectbox(
        "Metric", metrics, format_func=lambda m: strategy.METRIC_LABELS.get(m, m),
        help="Per-entry survival, or the portfolio's chance that at least one entry "
             "advances. Hedging axes such as an exposure cap are priced by the second "
             "and can never be selected by the first.")

    panel = strategy.paired_panel(paired, tournament, metric, baseline)
    if panel.empty:
        st.info("No paired comparisons for this combination.")
        return
    crossing, total = strategy.unresolved(panel)

    tiles = [
        ("Does not resolve", f"{crossing} of {total}",
         "Gaps whose 95% interval covers zero — a decision this budget cannot make"),
        ("Worlds paired", f"{int(panel['n_worlds'].iloc[0]):,}",
         "Simulated seasons resampled by the paired bootstrap"),
        ("Best resolved gap",
         f"{float(panel.loc[~panel['crosses_zero'], 'gap'].max()):+.4f}"
         if (~panel["crosses_zero"]).any() else "—",
         "The largest gap whose interval clears zero"),
    ]
    for col, (label, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, help=helptext)

    st.plotly_chart(
        fig_paired(panel, th, baseline, strategy.METRIC_LABELS.get(metric, metric)),
        width="stretch", key="paired", config={"displayModeBar": False})

    st.caption(
        f"**{crossing} of {total} gaps do not resolve, and they are the most useful rows "
        "here.** An interval covering zero is a decision the simulation budget cannot "
        "make, not a decision made — so they are drawn in their own slot with an open "
        "marker rather than left to be mistaken for small effects. The α arms clustered "
        "there are the concrete case: the blend axis has a resolved *sign* and its "
        "*location* is unresolved, and the two tiers disagree about where it is.")
    st.caption(
        "Pairing removes the common term the block-2 intervals are dominated by. Every "
        "arm is scored on the same drawn worlds, so differencing inside the world and "
        "resampling worlds is the same paired bootstrap every model comparison in this "
        "project uses.")

    with st.expander("Table view — every paired gap"):
        st.dataframe(_paired_table(panel), hide_index=True, width="stretch")


def _paired_table(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Arm": panel["strategy"], "Gap": panel["gap"],
        "Low": panel["gap_lo"], "High": panel["gap_hi"],
        "P(gap < 0)": panel["p_gap_below_zero"],
        "Resolves": ~panel["crosses_zero"],
    })
    return out.round(4)


# ── Page ──────────────────────────────────────────────────────────────────────

def render() -> None:
    shell.compact_tiles()
    st.title("Tournament & strategy")
    st.caption(
        "What the contest actually pays, what twenty-two drafting strategies bought "
        "against it in simulation, and what two seasons of real box scores had to say "
        "about that.")

    # The two families are gated separately so a half-built pipeline still draws the half
    # it has: `artifacts.optional` has already named the missing target in a warning.
    contest = load_contest()
    frames = load_strategy()
    if contest is None and frames is None:
        return

    th = shell.current_theme()
    tiers = (sorted(frames["sweep"]["tournament"].unique(),
                    key=lambda t: strategy.TARGET_TIERS.index(t)
                    if t in strategy.TARGET_TIERS else 99)
             if frames is not None else list(strategy.TARGET_TIERS))

    with st.sidebar:
        st.header("Tournament")
        tournament = st.selectbox(
            "Tier", tiers, format_func=strategy.pretty_tournament,
            label_visibility="collapsed",
            help="The two tiers the sweep drafts into. They differ by 2.6× in entry fee "
                 "and by 100× in top-prize multiple — and select the same portfolio, "
                 "which is what Gate D found.")
        st.markdown("---")
        st.caption(
            f"Read from `{rel(predictions_dir())}` — reproduce with "
            f"`{strategy.MAKE_SWEEP}` and `{strategy.MAKE_BRACKET}`; the contest tables "
            f"are captured raw data. Nothing on this page is re-simulated.")

    if contest is not None:
        st.markdown("---")
        st.subheader("1 · The contest structure")
        st.caption(
            "Five real DraftKings best-ball tournaments, captured with their prize "
            "tables. Four elimination rounds, and Round 1 is a zero-consolation "
            "knockout in every one of them.")
        contest_block(*contest, tournament, th)

    if frames is None:
        return

    st.markdown("---")
    st.subheader("2 · The strategy sweep")
    st.caption(
        "Twenty-two strategies over seven axes, each drafted against a field of "
        "ADP-drafting opponents in simulated seasons drawn from the model's own "
        "posterior and error-injected to reproduce its measured out-of-sample miss.")
    sweep_block(frames["sweep"], tournament, th)

    st.markdown("---")
    st.subheader("3 · Simulated versus realized")
    st.caption(
        "The same portfolios on both backtest surfaces. One of them selected the "
        "strategy; the other is a readout that changed nothing, and the difference "
        "between those two jobs is visible in the interval widths.")
    surfaces_block(frames["sweep"], frames["realized"], frames["shipped"],
                   tournament, th)

    st.markdown("---")
    st.subheader("4 · The paired comparisons")
    st.caption(
        "Every arm differenced against one baseline **inside** each simulated season, "
        "which removes the common term that makes block 2's intervals overlap.")
    paired_block(frames["paired"], tournament, th)
