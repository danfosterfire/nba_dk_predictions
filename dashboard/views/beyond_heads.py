"""Inputs beyond the heads — ADP, the capture programs, the calibrated numbers, the layout.

Everything the simulator consumes that is **not a fitted coefficient**. That is a genuinely
distinct kind of input and it was invisible on this dashboard until this page: the model
pages draw posteriors, and none of these four families is one.

1. **ADP.** A market forecast of the same target the model predicts, which makes it the most
   leakage-prone input in the repo and the one with the strictest dating rule. The block is
   built around what that rule *costs* rather than around the correlation it buys.
2. **The capture programs.** Three of the four are perishable. This is the only block on the
   dashboard that is an operational alarm rather than a result — a gap in the calendar is
   not a worse measurement, it is a day that no longer exists — so it is drawn at the top of
   the block and states plainly which gaps can still be closed.
3. **The calibrated numbers, split by whether the draw reads them.** ⚠️ Until 2026-08-16
   this said "the four calibrated simulator inputs", and it was wrong in three ways: two of
   the four are **diagnostics** the draws are scored against rather than given, and the
   injected per-(player, season) sigma — consumed on every minutes draw — was not listed at
   all. Five rows in two groups now, each carrying where it enters the draw or what it is
   checked against. The three windowed ones still show their `fit_window` instead of hiding
   it, because which window to consume is decided by what the number will be scored against,
   and the differences are small enough that consuming the wrong one would never announce
   itself. `docs/sim-inputs-plan.md`.
4. **The availability layout.** Added 2026-08-12, and the only input here that is not a
   number. The availability head draws *how many* games a player misses and cannot say
   *where* they fall, because games played is invariant to the arrangement — so the layout
   is chosen at draw time. It earns a block because the unit it moves is the one DK scores:
   twenty periods, best 7 of 16 seated in each.

   **The block draws the arm that ships and the season it reproduces, and nothing else.**
   The ladder artifact behind it carries four drawn arms; three of them are an argument for
   the fourth, and an argument belongs in `docs/availability-window-plan.md` §13 and in the
   decision log. That is also why the block reads *levels* rather than the ladder's own
   `recovered_share`, which is a ratio against an arm this page does not draw.
   `test_no_rejected_layout_arm_reaches_the_page_source` pins it.

The pure layer is `dashboard/inputs.py`; nothing here computes a result, and nothing in this
package imports `src/`.
"""

import pandas as pd
import streamlit as st

from dashboard import inputs, shell
from dashboard.artifacts import eda_dir, features_dir, optional, predictions_dir, rel
from dashboard.charts import (fig_adp_lag, fig_block_inflation, fig_calendar,
                              fig_correlation, fig_ladder, fig_layout_exposure,
                              fig_metric_facets)

#: How far back the calendar reaches. Long enough to cover a season's worth of daily
#: captures and short enough that one missed day is still a visible cell.
CALENDAR_DAYS = 150


# ── Loading ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Reading the ADP artifacts…")
def load_adp() -> dict[str, pd.DataFrame] | None:
    """The panel `make adp-panel` wrote, plus the two `make adp-profile` reports."""
    panel = optional(features_dir() / inputs.PANEL_FILE, target=inputs.MAKE_ADP)
    profile = optional(eda_dir() / inputs.PROFILE_FILE, target=inputs.MAKE_ADP)
    audit = optional(eda_dir() / inputs.MATCH_AUDIT_FILE, target=inputs.MAKE_ADP)
    if panel is None or profile is None or audit is None:
        return None
    return {"panel": panel, "profile": profile, "audit": audit}


@st.cache_data(show_spinner="Reading the capture calendar…")
def load_capture() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    calendar = optional(eda_dir() / inputs.CALENDAR_FILE, target=inputs.MAKE_CALENDAR)
    programs = optional(eda_dir() / inputs.PROGRAMS_FILE, target=inputs.MAKE_CALENDAR)
    if calendar is None or programs is None:
        return None
    return calendar, programs


@st.cache_data(show_spinner="Reading the availability layout…")
def load_layout() -> dict[str, pd.DataFrame] | None:
    """The two artifacts `make availability-exchangeability` writes."""
    ladder = optional(predictions_dir() / inputs.LAYOUT_FILE, target=inputs.MAKE_LAYOUT)
    profile = optional(predictions_dir() / inputs.LAYOUT_PROFILE_FILE,
                       target=inputs.MAKE_LAYOUT)
    if ladder is None or profile is None:
        return None
    return {"layout": ladder, "profile": profile}


@st.cache_data(show_spinner="Reading the calibrated simulator inputs…")
def load_calibrated() -> dict[str, pd.DataFrame] | None:
    """The five artifacts, each from the target that calibrated it."""
    wanted = (("residual", eda_dir() / inputs.RESIDUAL_FILE, inputs.MAKE_RESIDUAL),
              ("serial", eda_dir() / inputs.SERIAL_FILE, inputs.MAKE_SERIAL),
              ("bonus", eda_dir() / inputs.BONUS_FILE, inputs.MAKE_BONUS),
              ("dispersion", predictions_dir() / inputs.DISPERSION_FILE,
               inputs.MAKE_DISPERSION),
              # The injected sigma's `shipped_configuration` row. Fifth because it is the
              # one consumed number that is a config constant rather than a measurement.
              ("unification", predictions_dir() / inputs.MIN_UNIF_FILE,
               inputs.MAKE_MIN_UNIF))
    frames = {}
    for key, path, target in wanted:
        frame = optional(path, target=target)
        if frame is None:
            return None
        frames[key] = frame
    return frames


# ── Block 1 · ADP ─────────────────────────────────────────────────────────────

def adp_block(frames: dict, th: dict) -> None:
    panel, profile, audit = frames["panel"], frames["profile"], frames["audit"]
    cost = inputs.point_in_time_cost(panel)
    agree = inputs.agreement(profile)
    match = inputs.match_summary(audit)
    cascade = match.get("cascade", {})
    ablation = match.get("surname_initial", {})

    tiles = [
        ("Seasons held", f"{cost['seasons_held']}",
         "Seasons with at least one captured board, from either source"),
        ("Point-in-time legal", f"{cost['seasons_legal']} of {cost['seasons_held']}",
         "Seasons with a board observed at or before the opener. The other "
         f"{cost['seasons_lost']} are dropped by `adp.training_rows` — "
         + ", ".join(cost["lost"])),
        ("Rows that survive", f"{cost['row_share']:.0%}",
         f"{cost['legal_rows']:,} of {cost['rows']:,} panel rows"),
        ("DK ↔ consensus ρ", f"{agree.get('spearman', float('nan')):.4f}",
         f"Spearman over the {int(agree.get('n_pairs', 0))} players on both boards — "
         "the one paired observation the whole recalibration rests on"),
        ("Mean rank gap", f"{agree.get('mean_abs_rank_gap', float('nan')):.1f} picks",
         "How far apart the two boards put the same player, before any correction"),
        ("Unmatched after the cascade",
         f"{cascade.get('unmatched_rate_cascade', float('nan')):.2%}",
         f"{int(cascade.get('unmatched', 0))} of "
         f"{int(cascade.get('matchable_rows', 0))} matchable rows. The "
         f"{int(cascade.get('no_nba_history', 0))} board entries who have never played an "
         "NBA game are counted separately, not as failures"),
    ]
    for col, (label, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, help=helptext)

    snaps = inputs.snapshots(panel)
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.plotly_chart(fig_adp_lag(inputs.datable(snaps), th),
                        width="stretch", key="adp_lag",
                        config={"displayModeBar": False})
        undrawn = inputs.undated(snaps)
        st.caption(
            f"**Three dates per row, and this is the axis they exist for.** Every capture "
            f"carries when it was *observed*, when the season it describes *began*, and "
            f"the gap between them; a board is usable only if that gap is negative. "
            f"{cost['seasons_lost']} of {cost['seasons_held']} seasons have no board on "
            f"the left of the line and are dropped entirely — not down-weighted, dropped, "
            f"because a frozen board observed in July still carries a *preseason value* "
            f"and this project counts the qualifying **observation** rather than the "
            f"qualifying value."
            + (f" One season is on the tiles above and **not** on this axis: "
               f"{', '.join(sorted(set(undrawn['season'])))} has not been played, so its "
               f"board has no tip-off to measure from. It is legal — a board cannot "
               f"postdate a season that has not started — and it is in the table below "
               f"rather than drawn at a zero it was never measured at."
               if len(undrawn) else ""))
    with right:
        st.plotly_chart(fig_ladder(inputs.ladder(profile), th),
                        width="stretch", key="adp_ladder",
                        config={"displayModeBar": False})
        st.caption(
            "**The consensus board is never used raw**, and the ladder is why: a "
            "category-league consensus discounts centers for free-throw percentage while "
            "DK Best Ball pays rebounds 1.25 and blocks 2.0 flat, so the same player sits "
            f"{inputs.position_bias(profile).set_index('position')['mean_rank_gap'].get('C', float('nan')):+.1f} "
            "picks apart on the two boards at center. The shipped rung is the monotone "
            "rescale; the position offset scores better and is **not** shipped, because it "
            "needs a position label the two sources disagree about on 9.02% of players.")

    st.warning(
        f"**An unmatched rate is not a sufficient check on a join.** The rejected "
        f"surname-plus-initial rule scores **{ablation.get('unmatched_rate_surname_initial', 0):.1%}** "
        f"unmatched — better than the shipped cascade's "
        f"{cascade.get('unmatched_rate_cascade', float('nan')):.2%} — by *fabricating* "
        f"{int(ablation.get('ablation_false_matches', 0))} of its "
        f"{int(ablation.get('ablation_matches', 0))} matches. Every invented match improves "
        f"the metric, so the metric is monotonically increasing in the error it is meant to "
        f"detect. The rule is kept running forever as an ablation, beside the number it "
        f"discredits.", icon=":material/warning:")

    with st.expander("Table view — coverage by season, the surviving fuzzy matches, and "
                     "what the rejected rule invents"):
        st.markdown("**Coverage by season** · `legal` is an *any*, not an *all*")
        st.dataframe(_coverage_table(inputs.season_coverage(panel)), hide_index=True,
                     width="stretch")
        st.markdown("**Every surviving non-exact match** · small enough to read, which is "
                    "the standard a fuzzy tier is held to")
        st.dataframe(inputs.fuzzy_matches(audit), hide_index=True, width="stretch")
        st.markdown("**What the rejected rule matched instead** · the ablation, still running")
        st.dataframe(inputs.ablation_false_matches(audit), hide_index=True,
                     width="stretch")
        st.markdown("**Where the two boards disagree** · by position, and by draft round")
        st.dataframe(inputs.position_bias(profile), hide_index=True, width="stretch")
        st.dataframe(inputs.tier_gap(profile), hide_index=True, width="stretch")


def _coverage_table(coverage: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Season": coverage["season"], "Sources": coverage["sources"],
        "Boards": coverage["n_snapshots"], "Legal boards": coverage["n_legal"],
        "Usable": coverage["legal"], "Earliest legal": coverage["earliest_legal"],
        "Rows": coverage["rows"], "Legal rows": coverage["legal_rows"],
    })


# ── Block 2 · the capture programs ────────────────────────────────────────────

def capture_block(calendar: pd.DataFrame, programs: pd.DataFrame, th: dict) -> None:
    alarm = inputs.calendar_alarm(programs)

    tiles = [
        ("Days captured", f"{alarm['captured']:,}",
         "Across all four programs, inside each one's own window"),
        ("Gaps still fetchable", f"{alarm['recoverable']}",
         "Missed days inside a retention window. `make daily-capture` closes them, and "
         "they stop being fetchable when they age out"),
        ("Days permanently lost", f"{alarm['lost']}",
         "Missed days with no refetch. Nothing closes these"),
        ("Most exposed", alarm["worst"] or "—",
         "The program holding the most permanently lost days"),
    ]
    for col, (label, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, help=helptext)

    grid = inputs.calendar_grid(calendar, programs, days=CALENDAR_DAYS)
    st.plotly_chart(
        fig_calendar(grid, inputs.row_labels(programs), th, inputs.STATE_ORDER,
                     inputs.STATE_LABELS),
        width="stretch", key="calendar", config={"displayModeBar": False})

    st.caption(
        f"The last {CALENDAR_DAYS} days of every program. **`nothing to capture` is not a "
        f"failure** — it is a day the source published nothing, which is most of the "
        f"offseason, and colouring it as a gap would cry wolf on dozens of days. **`missed` "
        f"is a run that did not happen**, and whether it can still be closed is a property "
        f"of the program rather than of the day, so it is written on the row label: the NBA "
        f"PDFs are fetchable until they age out of the CDN, and the ESPN feed keeps no "
        f"history at all.")
    st.caption(
        "The two ADP boards appear as the sparse *events* they are, and their empty "
        "stretches are drawn as nothing rather than as gaps — a board that opens in "
        "October has no schedule to have missed the rest of the year. FantasyPros is the "
        "one source of the four that is not on a deadline, because Wayback holds its "
        "history.")

    # The four source descriptions live inside the expander rather than on the surface.
    # They are the one genuinely *prose* thing on this page, and `dashboard/README.md` is
    # explicit that prose about the project belongs in `docs/` — so the main surface keeps
    # the alarm and the calendar, and what each archive is sits one level down with the
    # table it annotates. Caught by reading the rendered page, where four stacked bold
    # paragraphs had turned the block back into a document.
    with st.expander("Table view — what each program is, and every missed day"):
        for row in programs.itertuples(index=False):
            note = inputs.PROGRAM_NOTES.get(row.program)
            if note is not None:
                st.markdown(f"**{note.label}** — {note.what} {note.stake}")
        st.dataframe(inputs.program_table(programs), hide_index=True, width="stretch")
        gaps = inputs.gap_table(calendar)
        st.markdown(f"**Every missed day** · {len(gaps)} of them")
        st.dataframe(gaps, hide_index=True, width="stretch")


# ── Block 3 · the four calibrated inputs ──────────────────────────────────────

def calibrated_block(frames: dict, window: str, th: dict) -> None:
    table = inputs.calibrated_table(frames, window)
    panel = inputs.window_panel(frames)
    widest_label, widest = inputs.widest_relative_spread(panel)

    # ⚠️ Two groups, not one row of five. This block showed "the four calibrated simulator
    # inputs" until 2026-08-16, which put a number the draw never reads beside one it reads
    # on every player-game and left the injected sigma off the page entirely. The split is
    # the correction: `role` comes from `inputs.CALIBRATED` so the page cannot disagree with
    # the pure layer about which is which, and `where` is on every tile because "consumed"
    # is only meaningful if a reader can see the seam it enters at.
    consumed = inputs.by_role(table, inputs.CONSUMED)
    diagnostic = inputs.by_role(table, inputs.DIAGNOSTIC)

    st.markdown("**Consumed by the draw** — change one and the tensor changes.")
    for col, row in zip(st.columns(len(consumed)), consumed.itertuples(index=False)):
        col.metric(row.label, row.text,
                   help=f"{row.unit} — {row.what}\n\n**Enters at:** {row.where}")

    st.markdown("**Diagnostic only** — measured, reported, and never imposed.")
    for col, row in zip(st.columns(len(diagnostic)), diagnostic.itertuples(index=False)):
        col.metric(row.label, row.text,
                   help=f"{row.unit} — {row.what}\n\n**Checked by:** {row.where}")
    st.caption(
        "**The distinction is the block.** All five are calibrated; only three are *inputs*. "
        "A diagnostic is a target the drawn value is scored against — the game-level minutes "
        "dispersion lost input status by an explicit decision, because the composition fits "
        "its own role-graded rho and only one of the two can govern a draw, and the block "
        "inflation was never imposed at all: the serial structure the draw has is *produced* "
        "by the season-constant frailties. The injected sigma is the one that was missing "
        "from this page while being read on every minutes draw the simulator makes — and it "
        "is where the **marginal minutes head** enters the shipped chain: that head is "
        "fitted and carded and never drawn from, and its season-total spread is the target "
        "`make minutes-unification` calibrated this constant against, so its information "
        "reaches the simulator as this one number rather than as draws.")

    st.plotly_chart(
        fig_metric_facets(inputs.window_facets(panel), th, inputs.WINDOW_SLOTS,
                          columns=2),
        width="stretch", key="windows", config={"displayModeBar": False})
    st.caption(
        f"**Each windowed number at all three fit windows, which is the point of the "
        f"block.** The injected sigma is absent here deliberately — it is a config constant "
        f"rather than an artifact measured per window, and three identical rows would imply "
        f"it had been measured three times. The "
        f"windows differ by two seasons out of thirty, so the largest of the four — "
        f"{widest_label} — moves by **{widest:.1%}** across them and the rest by less. That "
        f"is exactly why the window has to be *chosen* rather than defaulted into: a number "
        f"that moved visibly would have been caught years ago, and one that does not is the "
        f"kind that gets consumed at the wrong window forever. "
        f"`residual_correlation.to_matrix` defaults to `{inputs.SAFE_WINDOW}` because it is "
        f"never *wrong* — it excludes the test seasons and nothing else — and "
        f"`make simulate-season` overrides it to `{inputs.SIM_WINDOW}`, because its backtest "
        f"scores 2022-23 and 2023-24, which are *inside* `{inputs.SAFE_WINDOW}`.")

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        square = inputs.copula_square(frames["residual"], window, mask_diagonal=True)
        limit = inputs.copula_limit(frames["residual"], window)
        st.plotly_chart(fig_correlation(square, th, height=460, limit=limit),
                        width="stretch", key="copula",
                        config={"displayModeBar": False})
        cells = inputs.copula_cells(frames["residual"], window)
        top = cells.iloc[0] if len(cells) else None
        st.caption(
            "**What is left after the shared minutes draw.** Minutes are 46.4% of "
            "within-player residual variance and every head gets the *same* minutes draw as "
            "exposure, so this matrix is the dependence that survives conditioning on them. "
            + (f"The largest cell in it is **{top['pair']}** at **{top['r']:+.3f}**, "
               f"which is why the scale runs to ±{limit:g} and the diagonal is blanked: on "
               f"a matrix pinned to ±1 every real cell here is the midpoint. "
               if top is not None else "")
            + "Only the **count block** is imposed: three of the four conversion heads are "
            "measured nulls for a per-game hot hand, so a season-level probability is the "
            "right factorization and there is no per-game residual left to couple.")
    with right:
        serial = inputs.block_inflation(frames["serial"], window)
        st.plotly_chart(fig_block_inflation(serial, th),
                        width="stretch", key="serial",
                        config={"displayModeBar": False})
        st.caption(
            "**Sequential structure goes on minutes and nowhere else**, and this is the "
            "measurement that decided it. Minutes carry a real ten-game block inflation; "
            "the shooting heads sit at one, which is the null a hot hand would have to "
            "beat. The one conversion head that is *not* a null is `fg3a|fga` — the "
            "three-point **mix** genuinely moves game to game — and its coupling is the "
            "one cell the simulator knowingly does not carry.")

    with st.expander("Table view — the four inputs, all three windows, and the two "
                     "unit-specific readings"):
        st.dataframe(_windows_table(panel), hide_index=True, width="stretch")
        st.markdown("**Bonus overdispersion is unit-specific** · the same threshold "
                    "calibrated at two units gives two answers, and the simulator draws at "
                    "the second")
        st.dataframe(_bonus_table(inputs.bonus_rows(frames["bonus"], window)),
                     hide_index=True, width="stretch")
        st.markdown("**The game-level minutes dispersion** · a diagnostic the composition's "
                    "draws are checked against, not an input to them")
        st.dataframe(inputs.minutes_dispersion(frames["dispersion"]), hide_index=True,
                     width="stretch")
        st.markdown("**The copula's largest cells** · each pair once")
        st.dataframe(inputs.copula_cells(frames["residual"], window, top=12),
                     hide_index=True, width="stretch")


def _windows_table(panel: pd.DataFrame) -> pd.DataFrame:
    wide = panel.pivot(index="label", columns="fit_window", values="value")
    wide = wide[[w for w in inputs.FIT_WINDOWS if w in wide.columns]]
    spread = panel.drop_duplicates("label").set_index("label")["relative_spread"]
    out = wide.reset_index(names="Input")
    out["Spread across windows"] = [spread.get(name, float("nan"))
                                    for name in out["Input"]]
    return out.round(6)


def _bonus_table(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Unit": rows["unit"],
        "Zeroes the bias at": rows["fitted"].round(5),
        "Shipped constant": rows["shipped_constant"],
        "Relative bias at the constant": rows["shipped_relative_bias"].round(4),
        "Rows": rows["n"],
    })


# ── Block 4 · the availability layout ─────────────────────────────────────────

def layout_block(frames: dict, th: dict) -> None:
    ladder, clustering = frames["layout"], frames["profile"]
    headline = inputs.layout_headline(ladder)
    scored = inputs.layout_rows_scored(ladder)

    for col, row in zip(st.columns(len(headline)), headline.itertuples(index=False)):
        col.metric(row.label, row.text, help=f"{row.unit} — {row.what}")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.plotly_chart(
            fig_layout_exposure(inputs.layout_exposure(ladder), th, inputs.LAYOUT_SLOTS),
            width="stretch", key="layout_exposure", config={"displayModeBar": False})
        st.caption(
            f"**The availability head says how many games a player misses; it cannot say "
            f"where they fall.** Permuting a played/missed vector leaves games played "
            f"exactly where it was, so the arrangement is chosen at draw time rather than "
            f"fitted — which is what puts it on this page. It matters because DraftKings "
            f"scores twenty periods and seats the best 7 of 16 in each, so what a roster is "
            f"exposed to is not how many games a player missed but how many *periods* he is "
            f"a guaranteed zero, and whether they come in a row. Read on "
            f"**{scored:,}** single-team validation player-seasons, with games played held "
            f"at its realized value on every row so nothing here is a marginal difference "
            f"wearing an arrangement's clothes.")
    with right:
        st.plotly_chart(
            fig_metric_facets(_edge_facet(inputs.layout_edge_profile(clustering)), th,
                              inputs.EDGE_SLOTS, columns=1, row_height=190),
            width="stretch", key="layout_edges", config={"displayModeBar": False})
        st.caption(
            "**What the draw conditions on.** A tenure edge block is a delayed first "
            "appearance or a season that ended early — one contiguous run at an end of the "
            "schedule, not an injury with a return. How much of a player-season's missed "
            "time sits in one is a function of *how much he missed*, which is why that is "
            "the key the block length is resampled within. Role is the second key, and it "
            "moves which **end**: the leading block runs from "
            f"{_ends_text(inputs.layout_edge_ends(clustering))}.")

    with st.expander("Table view — the spell lengths the layout realizes, and the two "
                     "draw keys"):
        st.markdown("**Absence-spell lengths** · the layout is selected on scoring-period "
                    "exposure, so this distribution is a statistic it was never scored "
                    "against — reproducing it is independent evidence rather than a "
                    "restatement")
        st.dataframe(_spell_table(inputs.layout_spell_shape(ladder)), hide_index=True,
                     width="stretch")
        st.markdown("**Draw key 1 · how much he missed** · sets how much of it is an edge "
                    "block")
        st.dataframe(inputs.layout_edge_profile(clustering).round(4), hide_index=True,
                     width="stretch")
        st.markdown("**Draw key 2 · role** · sets which end the block sits at")
        st.dataframe(inputs.layout_edge_ends(clustering).round(4), hide_index=True,
                     width="stretch")


def _edge_facet(profile: pd.DataFrame) -> pd.DataFrame:
    """The edge profile in `fig_metric_facets`' schema — one facet, four rows.

    Reused rather than given its own builder, for the reason `charts.py` states about the
    copula and the window panel: this is a small fixed set of rows compared inside a facet,
    which is that figure. Every row takes the same slot because no bin *ships* — the
    gradient across them is the mechanism, and highlight-and-gray here would answer a
    question ("which one is chosen?") that the figure is not about.
    """
    if profile.empty:
        return pd.DataFrame(columns=["metric_label", "head", "label", "value", "text",
                                     "reference"])
    return pd.DataFrame({
        "metric_label": "Share of a player-season's missed games in a tenure edge block",
        "head": inputs.EDGE_KEY,
        "label": [f"missed {b}" for b in profile["missed_share_bin"]],
        "value": profile["mean_edge_frac"].to_numpy(),
        "text": [f"{v:.3f}" for v in profile["mean_edge_frac"]],
        "reference": [None] * len(profile),
    })


def _ends_text(ends: pd.DataFrame) -> str:
    if ends.empty:
        return "a late signing at one end to a season-ending injury at the other"
    first, last = ends.iloc[0], ends.iloc[-1]
    return (f"**{first['mean_pre_frac']:.3f}** of a {first['population']} player's missed "
            f"games down to **{last['mean_pre_frac']:.3f}** of a {last['population']} "
            f"player's, while the trailing block runs the other way "
            f"({first['mean_post_frac']:.3f} to {last['mean_post_frac']:.3f})")


def _spell_table(shape: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Statistic": shape["label"],
        "What the simulator draws": shape["shipped"].round(4),
        "What actually happened": shape["observed"].round(4),
    })


# ── Page ──────────────────────────────────────────────────────────────────────

def render() -> None:
    shell.compact_tiles()
    st.title("Inputs beyond the heads")
    st.caption(
        "Everything the simulator consumes that is not a fitted coefficient: the market's "
        "own forecast, the archives that will not exist if nobody captures them, and the "
        "four numbers that are handed to the simulator rather than scored by it.")

    adp = load_adp()
    capture = load_capture()
    calibrated = load_calibrated()
    layout = load_layout()
    if adp is None and capture is None and calibrated is None and layout is None:
        return

    th = shell.current_theme()
    with st.sidebar:
        st.header("Fit window")
        window = st.selectbox(
            "Window", list(inputs.FIT_WINDOWS),
            index=list(inputs.FIT_WINDOWS).index(inputs.SIM_WINDOW),
            label_visibility="collapsed",
            help="Which window block 3 reads. `train` is what `make simulate-season` "
                 "consumes, because its backtest scores the two seasons `train_val` "
                 "contains. `full` reads the held-out seasons and is here to be compared "
                 "against, not consumed.")
        st.markdown("---")
        st.caption(
            f"Read from `{rel(eda_dir())}`, `{rel(features_dir())}` and "
            f"`{rel(predictions_dir())}` — reproduce with `{inputs.MAKE_ADP}`, "
            f"`{inputs.MAKE_CALENDAR}`, `{inputs.MAKE_RESIDUAL}`, `{inputs.MAKE_SERIAL}`, "
            f"`{inputs.MAKE_BONUS}`, `{inputs.MAKE_DISPERSION}` and "
            f"`{inputs.MAKE_LAYOUT}`. Nothing on this page is refitted.")

    if capture is not None:
        st.markdown("---")
        st.subheader("1 · The capture programs, and the ones on a deadline")
        st.caption(
            "First, because it is the only block here that is an alarm rather than a "
            "result. Three of these four archives cannot be rebuilt from anywhere: a day "
            "nobody captured is a day that no longer exists.")
        capture_block(*capture, th)

    if adp is not None:
        st.markdown("---")
        st.subheader("2 · ADP — the market's forecast, and what dating it costs")
        st.caption(
            "ADP is a forecast of the same target this project predicts, which makes it "
            "the most leakage-prone input in the repo. It gets three dates per row and a "
            "name-matching audit that refuses to be summarized by a coverage rate.")
        adp_block(adp, th)

    if calibrated is not None:
        st.markdown("---")
        st.subheader("3 · The four calibrated simulator inputs")
        st.caption(
            "Not coefficients and not scores: four numbers measured once and handed to the "
            "simulator. Each carries the fit window it was measured at, and the window is "
            "on screen because choosing it wrong is invisible in the output.")
        calibrated_block(calibrated, window, th)

    if layout is not None:
        st.markdown("---")
        st.subheader("4 · The availability layout — where the missed games fall")
        st.caption(
            "A fifth input, and the only one here that is not a number. The availability "
            "head draws how many games a player misses; where they land is a separate "
            "choice made at draw time, because games played is the same count whichever "
            "way the season is arranged.")
        layout_block(layout, th)
