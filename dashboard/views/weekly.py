"""Weekly scores — observed against simulated `dk_pts`, per player per scoring period.

**This page is Gate A at a unit Gate A does not cover.** `make simulate-season` scores four
things — the season total, the games-played pmf, the per-game bonus rate and the
season-total minutes spread — and none of them is `dk_pts` at the scoring period. That is
the unit the contest is decided at: DK seats the best 7 of a 16-man roster **per period**,
so every weekly max, every round total and every elimination cut is a function of the
weekly distribution rather than of the season total. This project has already paid once for
the lesson that a head is only a model at the unit it was scored at (`make
minutes-unification`, one posterior and two opposite verdicts at two units), and this is
that check one level down from Gate A's own headline row.

Six blocks:

1. **The unit** — what a period is, and the fact the rest of the page depends on: three of
   the twenty are double weeks, so the two facets are not two equal units.
2. **Gate A at this unit** — the same metric set `src/sim/season.py` reports, computed by
   the same function, beside Gate A's own season-total row.
3. **The spread** — three model spreads against the observed one, because a best-ball week
   is a max over sixteen players and only one of the three is comparable.
4. **Period by period** — the reading along the calendar the pooled panels cannot give.
5. **The observed ECDF over the predictive ribbon**, read as a distance.
6. **Observed against predicted, and the scaled quantile residual** — the same two figures
   a model page's block 6 draws, over a different predictive.

Blocks 5 and 6 reuse `charts.fig_ecdf`, `fig_calibration`, `fig_qq` and
`fig_quantile_residual` **unmodified**: `make weekly-scores` cuts its artifacts with
`src/models/model_cards.py`'s own binning helpers, so the encodings are the ones a reader
has already learned on the four model pages. The pure layer is `dashboard/weekly.py`.
Nothing here computes a model quantity and nothing in this package imports `src/`.
"""

import pandas as pd
import streamlit as st

from dashboard import shell
from dashboard import weekly as wk
from dashboard.artifacts import optional, predictions_dir, rel
from dashboard.charts import (fig_calibration, fig_ecdf, fig_period_profile, fig_qq,
                              fig_quantile_residual)


# ── Loading ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Reading the weekly scores…")
def load_weekly() -> dict[str, pd.DataFrame] | None:
    """The six artifacts `make weekly-scores` wrote, or `None` once one has been named."""
    directory = predictions_dir()
    frames = {}
    for key, name in (("index", wk.INDEX_FILE),
                      ("period", wk.PERIOD_FILE),
                      ("ecdf", wk.ECDF_FILE),
                      ("calibration", wk.CALIBRATION_FILE),
                      ("quantile", wk.QUANTILE_FILE),
                      ("sample", wk.SAMPLE_FILE)):
        frame = optional(directory / name, target=wk.MAKE_WEEKLY)
        if frame is None:
            return None
        frames[key] = frame
    return frames


@st.cache_data(show_spinner=False)
def load_gate_a() -> pd.DataFrame | None:
    """Gate A's own table — the season-total row this page is the counterpart to."""
    path = predictions_dir() / wk.GATE_A_FILE
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Block 1 · the unit ────────────────────────────────────────────────────────

def unit_block(index: pd.DataFrame) -> None:
    st.dataframe(wk.structure(index), hide_index=True, width="stretch")
    st.caption(
        "**Three of the twenty scoring periods are not a week, and the panels below are "
        "faceted for that reason.** DK's Round 1 is seventeen weekly periods and Rounds 2, "
        "3 and 4 are two weeks each, which the simulator collapses onto twenty tensor "
        "slots. A double week carries about twice the games and about twice the `dk_pts`, "
        "so pooling the two into one distribution would put a right tail on every picture "
        "here that is a calendar fact rather than a model — the same argument the model "
        "pages make about their fitting unit, one level up.")
    st.caption(
        "The seasons are the two validation seasons and **2018-19 and 2021-22**, which are "
        "the last two training seasons carrying DK's whole four-round structure. 2020-21 "
        "has no Round 4 at all — the season started on 21 December and ran out of weeks — "
        "and 2019-20's Round 4 is the Orlando bubble, where a fifth of the pool has an "
        "observed zero that is a schedule fact rather than an availability outcome. The "
        "test seasons are locked and appear nowhere on this page.")


# ── Block 2 · Gate A at this unit ─────────────────────────────────────────────

def gate_block(index: pd.DataFrame, gate: pd.DataFrame | None) -> None:
    board = wk.gate_board(index)
    st.dataframe(
        board.round({"Observed mean": 2, "Predicted mean": 2, "MAE": 2, "Bias": 2,
                     "R²": 4, "CRPS": 2}),
        hide_index=True, width="stretch")
    st.caption(
        "The same metric set every head in this project reports, computed by the **same "
        "function** `make simulate-season` computes its own Gate A rows with "
        "(`season.marginal_metrics`) — so the season-total row below is a comparison of two "
        "units rather than of two definitions. CRPS is the distributional one and is the "
        "reading that matters here: the deliverable is a draw, not a point estimate.")

    season = wk.season_total_row(gate)
    if not season.empty:
        with st.expander("Gate A's own row — the same tensor, summed to the season"):
            st.dataframe(season.round({"MAE": 2, "Bias": 2, "R²": 4, "CRPS": 2}),
                         hide_index=True, width="stretch")
            st.caption(
                f"From `{wk.GATE_A_FILE}`, one row per simulated season. It is a **season "
                f"total over twenty periods**, so its MAE is roughly twenty times a "
                f"week's; what is comparable across the two is R², and it is the number "
                f"that says how much of the weekly signal survives at the unit the contest "
                f"pays at.")


def spread_block(index: pd.DataFrame) -> None:
    board = wk.spread_board(index)
    tiles = []
    for _, row in board.iterrows():
        tiles.append((f"{row['Unit']} · {row['Split'].lower()}",
                      f"{row['Ratio']:.3f}×",
                      f"Simulated spread over observed spread, both pooled over every row "
                      f"and draw: {row['Simulated sd (pooled)']:.1f} against "
                      f"{row['Observed sd']:.1f} dk_pts"))
    for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 3)), tiles):
        col.metric(label, value, help=helptext)

    st.dataframe(
        board.round({"Observed sd": 2, "Simulated sd (pooled)": 2, "Ratio": 3,
                     "Per player-period": 2, "Point prediction": 2,
                     "Zero weeks observed": 3, "Zero weeks simulated": 3}),
        hide_index=True, width="stretch")
    st.caption(
        "**A best-ball week is a max over sixteen players, so the weekly spread decides "
        "more of a lineup's score than the weekly mean does.** Three model spreads ship and "
        "only one of them answers the observed column: *pooled* is the marginal the "
        "simulator implies over every row and draw at once. *Per player-period* is what it "
        "puts on one player's one week, and *point prediction* is the spread of the "
        "posterior means alone — narrower than the data **by construction**, because a mean "
        "over draws has averaged its own noise away, so a page printing that one beside the "
        "observed column would report a far too narrow model when nothing of the sort has "
        "been measured.")


# ── Block 3 · period by period ────────────────────────────────────────────────

def profile_block(period: pd.DataFrame, th: dict) -> None:
    panels = {wk.SPLIT_LABELS[split]: wk.profile_panel(period, split)
              for split in wk.SPLITS}
    panels = {name: part for name, part in panels.items()
              if part is not None and len(part)}
    if not panels:
        st.info("No per-period readout was cut.")
        return

    st.plotly_chart(fig_period_profile(panels, th, title=""), width="stretch",
                    key="weekly-profile", config={"displayModeBar": False})
    st.caption(
        "**The reading is the gap between the two lines, week by week.** The panels below "
        "pool seventeen weeks into one distribution, which is right for a calibration "
        "picture and cannot say whether the simulator drifts *through* a season. The "
        "dotted divider is where the unit changes: `W1`–`W17` are one week each and "
        "`R2`/`R3`/`R4` are whole rounds of two, so the step up at the right is the "
        "calendar. Each point is pooled over the split's seasons weighted by its own row "
        "count.")

    for name, part in panels.items():
        with st.expander(f"Table view — {name.lower()}, period by period"):
            st.dataframe(wk.profile_table(part), hide_index=True, width="stretch")


# ── Block 4 · the ribbon ──────────────────────────────────────────────────────

def split_panels(builder, cards_key: str, cards: dict, period_type: str) -> dict:
    """One facet's two splits, keyed by split label — the shape every shared figure takes.

    **One figure per period type rather than one figure of four panels**, and that was a
    rendered-PNG finding rather than a preference: at four columns the horizontal legend
    runs across the first two subplot titles, since both live in the strip above the plot
    area. Two columns is also the shape a model page draws, so the reader meets the same
    figure twice rather than a wider variant of it.
    """
    return {wk.SPLIT_LABELS[split]: frame
            for split in wk.SPLITS
            for frame in [builder(cards[cards_key], period_type, split)]
            if frame is not None and len(frame)}


def ribbon_block(cards: dict, th: dict) -> None:
    index = cards["index"]
    distance = wk.band_distance(cards["ecdf"], index)
    if distance.empty:
        st.info("No ECDF ribbon was cut.")
        return

    tiles = [(f"{row['label']} · {row['split_label'].lower()}",
              f"{row['max_gap']:.3f}",
              f"The furthest the observed ECDF sits from the median simulated season, in "
              f"ECDF units, at {row['value_at_max']:,.3g} dk_pts")
             for _, row in distance.iterrows()]
    for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 3)), tiles):
        col.metric(label, value, help=helptext)

    for period_type in wk.period_types(index):
        panels = split_panels(wk.ecdf_panel, "ecdf", cards, period_type)
        if not panels:
            continue
        label = wk.period_label(index, period_type)
        st.markdown(f"**{label}** — "
                    f"{int(wk.facet_row(index, period_type, 'train')['n_periods'])} "
                    f"periods a season")
        st.plotly_chart(
            fig_ecdf(panels, th, f"dk_pts in one {label.lower()} period", title=""),
            width="stretch", key=f"weekly-ecdf-{period_type}",
            config={"displayModeBar": False})
    st.caption(
        f"**Read the size of the miss, not in-or-out.** One replicate is a whole simulated "
        f"season, and its ECDF over this facet's player-periods is recorded — so the band "
        f"is the spread of *seasons* the model thinks it could have produced. At these "
        f"sample sizes it is one to two ECDF points wide and an honest model leaves it "
        f"somewhere; the useful statistic is the largest vertical distance from the median "
        f"replicate, tiled above. {wk.provenance(index)}")

    with st.expander("Table view — the distance, per facet"):
        st.dataframe(
            distance.rename(columns={
                "label": "Unit", "split_label": "Split",
                "max_gap": "Largest gap from median", "mean_gap": "Mean gap",
                "inside_95": "Inside 95% band", "n_grid": "Grid points",
                "n_rows": "Rows", "n_draws": "Simulated seasons",
                "value_at_max": "dk_pts at the largest gap"})
            .drop(columns=["period_type", "split"])
            .round({"Largest gap from median": 4, "Mean gap": 4, "Inside 95% band": 3,
                    "dk_pts at the largest gap": 2}),
            hide_index=True, width="stretch")


# ── Block 5 · predicted against observed, and the quantile residual ───────────

def calibration_block(cards: dict, th: dict) -> None:
    index = cards["index"]
    labels = wk.calibration_keys(index)
    cells = {(period_type, split): wk.calibration_panel(cards["calibration"],
                                                        period_type, split)
             for period_type in labels for split in wk.SPLITS}
    if all(frame is None or frame.empty for frame in cells.values()):
        st.info("No calibration density was cut.")
        return
    points = {(period_type, split): wk.sample_points(cards["sample"], period_type, split)
              for period_type in labels for split in wk.SPLITS}
    axes = {period_type: ("Simulated mean dk_pts", "Observed dk_pts")
            for period_type in labels}

    st.plotly_chart(
        # Height per row rather than a constant: `fig_calibration` derives its grid from
        # the panel count, and two rows squeezed into one row's height puts the second
        # subplot's title on the first's x-axis label — which `AppTest` cannot see.
        fig_calibration(cells, points, th, axes, labels, title="",
                        height=180 + 260 * len(labels)),
        width="stretch", key="weekly-calibration", config={"displayModeBar": False})
    st.caption(
        "A 30 × 30 binned density with a bounded subsample laid over it for texture — the "
        "same object and the same encoding a model page's block 6 draws, because "
        "`make weekly-scores` bins it with `make model-cards`' own helpers. One **row per "
        "period length** and one axis range down each row: the two units are on different "
        "`dk_pts` scales and must not share an axis, while train and validation inside a "
        "row must. The x axis is the mean over the simulated seasons, which is a point "
        "summary of a distribution the ribbon above shows in full.")
    st.caption(
        "**The dark band along the bottom of every panel is real and is the whole reason "
        "this unit is worth scoring.** About a fifth of player-periods score nothing at "
        "all — the player was hurt, rested or out of the rotation that week — and a season "
        "total averages that away completely. A best-ball lineup does not: it takes the "
        "best 7 of 16 in each period, so a zero week is survivable and a *cluster* of them "
        "is not.")


def quantile_block(cards: dict, th: dict) -> None:
    st.markdown("**Scaled quantile residuals**")
    index = cards["index"]
    distance = wk.quantile_distance(cards["quantile"], index)
    if distance.empty:
        st.info("No quantile residual was cut.")
        return

    tiles = [(f"KS · {row['label'].lower()} {row['split_label'].lower()}",
              f"{row['ks']:.3f}",
              f"How far the {int(row['n']):,} scaled residuals sit from uniform, at their "
              f"furthest point. A distance in probability units — never a pass or a fail")
             for _, row in distance.iterrows()]
    for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 3)), tiles):
        col.metric(label, value, help=helptext)

    for period_type in wk.period_types(index):
        panels = split_panels(wk.qq_panel, "quantile", cards, period_type)
        if not panels:
            continue
        st.markdown(f"**{wk.period_label(index, period_type)}**")
        st.plotly_chart(fig_qq(panels, th, title=""), width="stretch",
                        key=f"weekly-qq-{period_type}",
                        config={"displayModeBar": False})
    st.caption(
        "**DHARMa's residual, over the simulated seasons.** Each player-period's residual "
        "is its randomized quantile inside its own replicate distribution — `below + U·at`, "
        "randomized across the probability mass at the observed value because the plain "
        "quantile of a discrete predictive is not uniform even under a perfect model. "
        "Uniform iff calibrated **whatever the scale**, which is what lets a one-week panel "
        "and a double-week panel be read on the same axes at all. The envelope is the "
        "pointwise Beta band on each order statistic: about 5 of 100 points sit outside it "
        "under a correct model, so it is a sense of scale rather than a test.")

    drawn = False
    for period_type in wk.period_types(index):
        residual = split_panels(wk.residual_cells, "quantile", cards, period_type)
        if not residual:
            continue
        drawn = True
        lines = split_panels(wk.quantile_lines, "quantile", cards, period_type)
        overlay = split_panels(wk.sample_points, "sample", cards, period_type)
        st.markdown(f"**{wk.period_label(index, period_type)}** — the residual against "
                    f"where the simulator ranked the week")
        st.plotly_chart(
            fig_quantile_residual(residual, lines, overlay, th, wk.QUANTILE_LEVELS,
                                  title=""),
            width="stretch", key=f"weekly-quantile-{period_type}",
            config={"displayModeBar": False})
    if drawn:
        st.caption(
            "**Predicted is rank-transformed**, so both axes are [0, 1] by construction and "
            "the four facets are on one grid without being put there. The three lines are "
            "the binned 0.25 / 0.5 / 0.75 quantiles of the residual and are **flat at those "
            "levels iff calibrated**; the dashed references are the levels themselves, and "
            "a bin with too few rows for a quartile is a gap rather than a line through "
            "three points. The shading is the *departure* from an even spread rather than "
            "the mass — a calibrated residual fills this square evenly by construction — so "
            "red is where rows pile up and blue is where they thin out.")

    row = index.iloc[0]
    st.caption(
        f"**The KS distance is a distance.** At {int(distance['n'].max()):,} rows a strict "
        f"uniformity test rejects every model in this project, so it is tiled as a size and "
        f"never as a verdict — the same rule the ribbon above is read under. What *is* "
        f"gated is the budget behind it: `{wk.MAKE_WEEKLY}` re-reads both the ribbon and the "
        f"distance on two interleaved halves of the {int(row['sim_draws']):,} simulated "
        f"seasons and fails the build if either moves by more than {wk.KS_MC_TOL}. The worst "
        f"reading here is {float(index['ks_mc'].max()):.4f} on the distance and "
        f"{float(index['ecdf_band_mc'].max()):.4f} on the ribbon.")

    with st.expander("Table view — the distance, per facet"):
        st.dataframe(
            distance.rename(columns={
                "label": "Unit", "split_label": "Split", "ks": "KS distance from uniform",
                "n": "Rows", "line_gap": "Furthest quantile line from its level",
                "n_bins": "Bins with enough rows to draw"})
            .drop(columns=["period_type", "split"])
            .round({"KS distance from uniform": 4,
                    "Furthest quantile line from its level": 4}),
            hide_index=True, width="stretch")


# ── The page ──────────────────────────────────────────────────────────────────

def render() -> None:
    shell.compact_tiles()
    st.title("Weekly scores")

    cards = load_weekly()
    if cards is None:
        st.stop()
    index = cards["index"]
    if index.empty:
        st.warning(f"`{wk.INDEX_FILE}` carries no facets — run `{wk.MAKE_WEEKLY}`.")
        st.stop()
    th = shell.current_theme()

    with st.sidebar:
        st.caption(
            f"Read from `{rel(predictions_dir())}` — reproduce with `{wk.MAKE_WEEKLY}`, "
            f"which reduces the tensors `{wk.MAKE_SIMULATE}` wrote and simulates nothing "
            f"itself. The period axis is already in the tensor, so changing the unit is a "
            f"grouped sum.")

    st.caption(
        "Gate A scores the season total. **The contest does not.** DK seats the best 7 of a "
        "16-man roster in every scoring period, so a weekly max, a round total and an "
        "elimination cut are all functions of the *weekly* distribution — and nothing in "
        "this project scored `dk_pts` there. This page is that check, on the same tensor, "
        "on the training and validation splits.")
    st.markdown(
        f"### dk_pts per player per scoring period · "
        f"**{int(index['n'].sum()):,}** player-periods  \n"
        f"`{index['seasons'].iloc[0]}` and `{index['seasons'].iloc[-1]}` · "
        f"{int(index['sim_draws'].iloc[0]):,} simulated seasons a facet · "
        f"posteriors at the `{index['fit_window'].iloc[0]}` fit window")

    blocks = (
        ("The unit", "Twenty periods a season, and three of them are not a week.",
         lambda: unit_block(index)),
        ("Gate A at this unit", "The marginals the simulator was handed, one level below "
                                "the season total.",
         lambda: gate_block(index, load_gate_a())),
        ("The spread", "A best-ball week is a max, so this is the statistic it is most "
                       "sensitive to.",
         lambda: spread_block(index)),
        ("Period by period", "The same comparison read along the calendar instead of "
                             "across it.",
         lambda: profile_block(cards["period"], th)),
        ("Predictive calibration", "The observed weekly distribution against the one the "
                                   "simulator says it should have produced.",
         lambda: ribbon_block(cards, th)),
        ("Predicted against observed", "Where a week lands, and what it leaves behind on a "
                                       "scale both period lengths share.",
         lambda: (calibration_block(cards, th), quantile_block(cards, th))),
    )
    for number, (title, caption, body) in enumerate(blocks, start=1):
        st.markdown("---")
        st.subheader(f"{number} · {title}")
        st.caption(caption)
        body()
