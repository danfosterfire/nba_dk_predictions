"""Box-score components — the eleven heads `dk_pts` is reassembled from.

Page 5 of `docs/dashboard-plan.md`'s expansion, and the second instantiation of the generic
renderer in `dashboard/views/model_page.py`. The seven blocks, the head selector and the
unit all come from there. What this module adds is the one thing this page owes that the
other three do not: **each head beside its own no-fit floor**.

That block is not decoration. A component head's entire claim is that fitting bought
something over a prior per-36 rate carried forward with no fitting at all, every head in
this project is quoted against that floor, and `ftm|fta` — free-throw percentage, which is
close to pure player skill — does not clear it. Rendering the ladder is how a page says so
instead of showing a fitted head with no reference point.

The other page-specific job is a naming hazard rather than a number. **`fg3a|fga` is a share
of attempts, not a shooting percentage**, and its own prior-season design column is called
`logit_fg3a_pct_lag1`, where `_pct_` means `fg3a / fga`. Three-point *shooting* percentage
is `fg3m / fg3a` and sits on a different head two rows down the same selector. That is the
confusion `stan_components.conversion_variants` takes an explicit `own=` parameter to
prevent in the fitting code; a page that reprinted the column name and said nothing would
hand it straight back. `model_cards.COMPONENT_BASIS` carries the claim and a test anchors it
to the shipped artifact, so a refit that renames the column fails rather than mislabels.
"""

import pandas as pd
import streamlit as st

from dashboard import model_cards as mc
from dashboard.artifacts import optional, predictions_dir
from dashboard.charts import fig_floor_margin
from dashboard.views import model_page

CLASS_KEY = "components"


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame | None:
    """The heads' own variant ladder — what each was selected from, and its floor."""
    return optional(predictions_dir() / mc.COMPONENT_METRICS_FILE,
                    target=mc.MAKE_STAN_COMPONENTS)


def floor_block(cards: dict, row: pd.Series, th: dict) -> None:
    """This page's own block, rendered under block 1.

    **Named rather than numbered**: the seven numbered blocks are the contract every model
    page keeps, and this is the box-score page's own. It sits under block 1 because "did
    this head need to exist" is the first question about a component head, not the seventh.
    """
    head = str(row["head"])
    st.caption(mc.basis_note(head))

    metrics = load_metrics()
    if metrics is None:
        return
    label = mc.metrics_key(row)
    board = mc.floor_board(metrics, cards["index"], CLASS_KEY)
    mine = board[board["head"] == head]
    if board.empty or mine.empty:
        st.warning(f"`{mc.COMPONENT_METRICS_FILE}` carries no ladder for `{label}` — run "
                   f"`{mc.MAKE_STAN_COMPONENTS}`.")
        return
    mine = mine.iloc[0]

    st.markdown("---")
    st.subheader("Against the no-fit floor")
    st.caption(
        "Every head on this page is quoted against a **no-fit floor** — for a count, the "
        "player's prior-season per-36 rate times his actual minutes, with nothing fitted; "
        "for a conversion, his prior-season percentage shrunk toward the league. The floor "
        "is a row of the same variant ladder that chose the shipped arm, so the comparison "
        "is one artifact rather than two runs.")

    tiles = [
        ("No-fit floor", f"{mine['floor_r2']:.4f}",
         f"Validation R² for `{mc.COMPONENT_FLOOR}` — no fitting at all"),
        ("This head", f"{mine['shipped_r2']:.4f}",
         f"Validation R² for `{mine['variant']}`, the arm its own ladder selected"),
        ("Margin", f"{mine['margin']:+.4f}", "What the fitting bought, in R²"),
        ("Clears the floor", "yes" if mine["clears"] else "no",
         "Whether the shipped arm scores above the floor on validation"),
    ]
    for col, (name, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(name, value, help=helptext)

    st.markdown(f"**`{label}`'s own ladder**")
    st.dataframe(mc.floor_ladder(metrics, label), hide_index=True, width="stretch")
    st.caption(
        "The floor first, then the arms, best R² down. Every column is scored on the "
        "**validation** split, which is the only split selection may read — the ladder that "
        "chose this head never materialized a test row.")

    st.plotly_chart(
        fig_floor_margin(board, th, highlight=head, title=""),
        width="stretch", key=f"floor-{head}", config={"displayModeBar": False})
    failing = board[~board["clears"]]
    st.caption(
        "**The zero line is the floor.** A bar's length is what fitting bought that head in "
        "validation R², and a bar on the left of the line is a head the floor beats — drawn "
        "outlined as well as reversed, since no value here may be reachable by colour "
        "alone. The head open above is in colour; the others are gray. "
        + (", ".join(f"`{r['label']}`" for _, r in failing.iterrows())
           + (" does not clear its floor" if len(failing) == 1
              else " do not clear their floors")
           + ", which is a finding rather than a defect: shrinking a prior percentage is "
             "already close to optimal for a quantity that is nearly pure player skill. "
           if len(failing) else "Every head on this page clears its floor. ")
        + "R² is each head's own, on its own response, so a height reads as *how much the "
          "fit added* and never as one head being better than another — a count head's R² "
          "is over a season total and a conversion head's is over a rate.")

    with st.expander("Table view — every head against its floor"):
        st.dataframe(mc.floor_table(board), hide_index=True, width="stretch")


def render() -> None:
    model_page.render(CLASS_KEY, extra={1: floor_block})
