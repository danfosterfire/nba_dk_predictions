"""Overview — page 1, and the one page of prose on this dashboard.

**Read `docs/dashboard-plan.md`, "Charter amendment 2026-08-10", before editing this
file.** The 2026-08-08 overhaul deleted a nine-tab project walkthrough for being
documentation rendered as an app, and that reasoning still stands; this page is exempted
from it because the *audience* changed. The walkthrough served the project architect and
lost to `docs/`. This page serves a portfolio reader who arrives at a URL with no context
and will not open a repository, and no document serves that reader, because they will not
read one.

The exemption is bounded, and the three bounds are the terms it was granted on:

1. **One page, one screen.** If it scrolls, it has become the walkthrough again. That is
   the whole reason the diagram is 150 px, the stage notes and the route blurbs are one
   line each, and there are five tiles rather than the eight this project could justify.
2. **Every number is read from an artifact.** Typed prose may say *what the project does*;
   it may not state a *result*. `dashboard/overview.py` is the mechanism — every figure on
   the page arrives from there as a lookup, so a re-run that moves a number moves the page
   and no headline can drift from the file that made it.
3. **No decision registry, no provenance links, no reversal log.** Those are what made the
   walkthrough a documentation surface. They stay in `dashboard/decisions.py` and `docs/`.

The route block is why this page was built last: it links into the eight pages that show
the work, and until they existed it would have been a table of contents for nothing.
"""

import pandas as pd
import streamlit as st

from dashboard import overview, shell
from dashboard.artifacts import eda_dir, optional, predictions_dir
from dashboard.charts import fig_pipeline

#: The route grid. Four across keeps each blurb on one line at laptop width; eight pages in
#: two rows is one of the things holding the page to a screen.
ROUTE_COLUMNS = 4

# Bound 1 is a *measurement*, not an intention, and this block is most of how it is met.
# Streamlit's defaults are laid out for a scrolling document: 6 rem of padding above the
# first element, a 2.5 rem `h1`, and 1 rem between every vertical block. On a page whose
# only hard constraint is that it end above the fold, that is 150-odd pixels spent on air
# before a word is read. Measured in Chrome at 1440x900, the page came in at 1,144 px with
# the defaults and 702 px with this block plus shorter copy.
#
# It is scoped to the main container so it cannot reach the sidebar, and it lives in the
# view rather than in `shell.py` because it is a layout choice for one page — a model page
# is *supposed* to scroll, and inheriting this would only make it scroll further.
#
# The delta rule is the second thing a rendering caught. A metric's delta does not wrap and
# does not ellipsize its own overflow honestly — at the default size, five tiles across put
# `-210.3 against assuming a full season` on screen as `-210.3 against assuming …`, and a
# comparison truncated mid-phrase is worse than no comparison. `stMetricDelta` was read off
# this Streamlit's frontend bundle, as `shell.TILE_CSS`'s test IDs were.
COMPACT_CSS = """
<style>
  [data-testid="stMainBlockContainer"] { padding-top: 3rem; padding-bottom: 1rem; }
  [data-testid="stMainBlockContainer"] h1 {
    font-size: 2.1rem; line-height: 1.3; padding: 0.2rem 0 0.2rem 0;
  }
  [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap: 0.65rem; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem; }
</style>
"""


@st.cache_data(show_spinner="Reading the headline artifacts…")
def load() -> dict[str, pd.DataFrame]:
    """Whichever of the eight sources are on disk, keyed by `overview.Source.key`.

    Partial by design. `optional()` names the missing file's `make` target on the page, and
    the readings that needed it drop out — a landing page should degrade to fewer tiles,
    not to a stack trace or to a screen of warnings with nothing under them.
    """
    roots = {overview.EDA: eda_dir(), overview.PREDICTIONS: predictions_dir()}
    frames = {}
    for source in overview.SOURCES:
        frame = optional(roots[source.directory] / source.filename,
                         target=source.target)
        if frame is not None:
            frames[source.key] = frame
    return frames


def intro() -> None:
    """What the problem is. Typed prose, and it states no result — see bound 2 above."""
    st.title("Drafting an NBA season, before it starts")
    st.markdown(
        "DraftKings best ball is a snake draft of 16 players, frozen before opening "
        "night: the best seven by roster slot score each week, and four elimination "
        "rounds cut the field to one. Nothing can be traded, started or benched "
        "afterwards, and all anyone knows on draft night is last season's box scores "
        "plus this season's rosters and schedule. **So what is worth having is not a "
        "sharper guess at each player's average, but an honest joint distribution over "
        "what a whole roster will do** — which is what this project fits, in Stan, one "
        "Bayesian head per box-score component, then drafts against real contests.")


def tiles(frames: dict[str, pd.DataFrame]) -> None:
    readings = overview.tile_readings(frames)
    if not readings:
        return
    shell.compact_tiles()
    for col, reading in zip(st.columns(len(readings)), readings):
        col.metric(reading.label, reading.value,
                   delta=reading.delta or None,
                   delta_color="inverse" if reading.delta.startswith("-") else "normal",
                   help=reading.note)
    st.caption(
        "Every figure on this page is read from an artifact a `make` target wrote, never "
        "typed — hover a tile for what it measures and which file it comes from.")


def pipeline(frames: dict[str, pd.DataFrame], th: dict) -> None:
    stages = overview.stage_readings(frames)
    if not stages:
        return
    wrapped = [stage._replace(note=overview.wrap(stage.note)) for stage in stages]
    st.plotly_chart(fig_pipeline(wrapped, th), width="stretch", key="pipeline",
                    config={"displayModeBar": False, "staticPlot": True})


def routes() -> None:
    """The way in. Eight page links in two rows, each with one line of what it holds.

    `shell.page` hands back the `st.Page` the entrypoint registered, because `st.page_link`
    accepts only those; outside the shell it returns None and the row degrades to its
    blurb. The links carry no numbers — a route label is exactly where a stray headline
    would try to creep back in past bound 2.
    """
    st.markdown("**Where the work is**")
    # A fresh `st.columns` per row rather than one grid filled column-major. A column is a
    # vertical stack, so under one grid a blurb that wraps to two lines in the first row
    # pushes only *its* column's second link down and the bottom row comes out ragged —
    # visible in a screenshot, invisible to `AppTest`, which sees eight page links either
    # way. Row by row, a wrap can cost alignment inside its own row and nowhere else.
    entries = list(overview.ROUTES)
    for start in range(0, len(entries), ROUTE_COLUMNS):
        row = entries[start:start + ROUTE_COLUMNS]
        for col, route in zip(st.columns(ROUTE_COLUMNS), row):
            target = shell.page(route.url_path)
            if target is None:
                col.markdown(f"**{route.url_path}**")
            else:
                col.page_link(target)
            col.caption(route.blurb)


def render() -> None:
    # The style block goes first so a missing-artifact warning from `load()` lands inside
    # the same compact chrome as everything below it.
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)
    th = shell.current_theme()
    frames = load()
    intro()
    tiles(frames)
    pipeline(frames, th)
    routes()
