"""Game length — the one input a forward simulation cannot look up.

Page 6 of `docs/dashboard-plan.md`'s expansion, and the third instantiation of the generic
renderer in `dashboard/views/model_page.py`. Two heads behind the selector: a beta-binomial
on whether a game reaches overtime, collapsed to one cell per season, and a beta-geometric
on how many extra periods it then plays. Both minutes heads need a game length, and every
backtest so far read it off `game_length.parquet` because the games had already happened.

**These are the two smallest heads in the project, and that shapes the page.** The onset
head fits 26 season cells and the depth head 4 depth cells, so several of the seven blocks
degrade honestly — the depth head has no design columns at all, and the onset head has one,
which is not a pair to correlate. The renderer says so rather than drawing an empty panel.

The consequence that mattered enough to add a block for: the onset head's **validation ECDF
is two grid points**, because two season cells is what the validation window collapses to.
Block 5's ribbon cannot say whether that head is right. What can is the unit the head is
actually consumed at — games — which `make stan-game-length` already writes as a
posterior-predictive count per game class, for the fitted arm and for its no-fit floor. That
is this page's own block, and it sits under block 1 for the same reason the box-score page's
does: it is the first question, not the seventh.
"""

import pandas as pd
import streamlit as st

from dashboard import model_cards as mc
from dashboard.artifacts import optional, predictions_dir
from dashboard.charts import fig_class_counts
from dashboard.views import model_page

CLASS_KEY = "game_length"

#: The split both game-length artifacts label their held-back rows with. `make
#: stan-game-length` writes `val`, not `validation`; the model cards' own vocabulary is the
#: long form, and the two files are not the same file.
LADDER_SPLIT = "val"


@st.cache_data(show_spinner=False)
def load_ladder() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The onset head's per-class predictive check and the depth head's depth ladder."""
    directory = predictions_dir()
    ppc = optional(directory / mc.GAME_LENGTH_PPC_FILE, target=mc.MAKE_STAN_GAME_LENGTH)
    depth = optional(directory / mc.GAME_LENGTH_DEPTH_FILE,
                     target=mc.MAKE_STAN_GAME_LENGTH)
    return ppc, depth


def floor_block(cards: dict, row: pd.Series, th: dict) -> None:
    """This page's own block: both heads read at the unit they are consumed at — games.

    Named rather than numbered, like the box-score page's. It is the readout block 5 cannot
    give the onset head, whose validation ECDF is two grid points wide.
    """
    ppc, depth = load_ladder()
    panel = mc.class_counts(row, ppc, depth, LADDER_SPLIT)
    if panel.empty:
        return
    head = str(row["head"])
    is_depth = head == "game_length_depth"

    st.markdown("---")
    st.subheader("Against the no-fit floor, in games")
    st.caption(
        "Both heads are quoted against a **no-fit floor**: for onset, the pooled league "
        "overtime rate with no season term; for depth, a plain geometric — a constant "
        "continuation hazard, which is what the beta-geometric integrates a Beta frailty "
        "out of. Counts are validation games, the split the ladder selected on."
        + ("" if is_depth else
           " `regulation` is `n_games` minus the other three by construction, for every arm "
           "alike, so it is carried in the table below and left off the figure — drawn, it "
           "is a 2,300-long bar that flattens the three classes the arms differ on."))

    drawn = panel[panel["drawn"]]
    st.plotly_chart(
        fig_class_counts(drawn, th,
                         value_label="overtime games" if is_depth else "games",
                         title=""),
        width="stretch", key=f"classes-{head}", config={"displayModeBar": False})
    grid = mc.band_distance(cards["ecdf"], head)
    validation = grid[grid["split"] == "validation"]
    st.caption(
        "The observed is the outlined bar and the two arms fill toward it — the ink is the "
        "encoding block 5's ribbon uses for its observed curve. Every bar prints its own "
        "count, and the table below carries all of them."
        + (f" **This is the readout block 5 cannot give**: that head's validation ribbon "
           f"below is drawn on {int(validation['n_grid'].iloc[0])} grid points, because "
           f"this head is fitted per {row['unit']} and the validation window is only "
           f"{int(row['n_validation']):,} of them. An ECDF over that many points is an "
           f"arithmetic shape rather than a calibration reading."
           if len(validation) else ""))

    nats = mc.depth_likelihood(depth, LADDER_SPLIT) if is_depth else None
    if nats is not None:
        tiles = [
            ("Geometric", f"{float(nats['geometric']):.4f}",
             "Nats per overtime game on validation — the no-fit floor"),
            ("Beta-geometric", f"{float(nats['beta_geometric']):.4f}",
             "Nats per overtime game on validation — the shipped head"),
            ("Overtime games", f"{int(nats['n_ot_games']):,}",
             "Validation games that reached overtime — everything the two figures above "
             "are scored on. The head itself is fitted on the training seasons"),
        ]
        for col, (name, value, helptext) in zip(st.columns(len(tiles)), tiles):
            col.metric(name, value, help=helptext)
        st.caption(
            "**The frailty is not what earns its keep here.** Per overtime game the two "
            "arms are a wash on validation, and on this few games they would have to be. "
            "The beta-geometric ships because the same Stan source serves the "
            "absence-spell duration head, where the extra parameter is decisive — see "
            "`docs/games-played-plan.md`.")

    label = "every depth" if is_depth else "every class, including regulation"
    with st.expander(f"Table view — {label}"):
        st.dataframe(mc.class_table(panel), hide_index=True, width="stretch")


def render() -> None:
    model_page.render(CLASS_KEY,
                      extra={1: floor_block, 3: model_page.stan_block})
