"""Overview — page 1, and the one page of prose on this dashboard.

**Read `docs/dashboard-plan.md`, "Charter amendment 2026-08-10", before editing this
file.** The 2026-08-08 overhaul deleted a nine-tab project walkthrough for being
documentation rendered as an app, and that reasoning still stands; this page is exempted
from it because the *audience* changed. The walkthrough served the project architect and
lost to `docs/`. This page serves a portfolio reader who arrives at a URL with no context
and will not open a repository, and no document serves that reader, because they will not
read one.

The exemption is bounded, and the three bounds are the terms it was granted on:

1. **It opens above the fold and scrolls no further than one screen more.** Amended
   2026-08-10 from a flat one screen, when the page became a paper — four sections plus
   the diagram plus nine routes do not fit 900 px, and the alternative was cutting the
   diagram or the routes to pay for the prose. Still a browser measurement, still the only
   bound no other layer can see, and the ceiling is still hard: the walkthrough was nine
   tabs, and nothing that opens above the fold and ends inside two screens is that.
2. **Every number is read from an artifact.** Typed prose may say *what the project does*;
   it may not state a *result*. `dashboard/overview.py` is the mechanism — every figure on
   the page arrives from there as a lookup, so a re-run that moves a number moves the page
   and no headline can drift from the file that made it. Since the page is prose, the
   bound is also enforced one level down: a typed sentence carries **no digit**, which is
   why the contest's own rules are spelled in words.
3. **No decision registry, no provenance links, no reversal log.** Those are what made the
   walkthrough a documentation surface. They stay in `dashboard/decisions.py` and `docs/`.

The route block is why this page was built last: it links into the nine pages that show
the work, and until they existed it would have been a table of contents for nothing.
"""

import pandas as pd
import streamlit as st

from dashboard import overview, shell
from dashboard.artifacts import eda_dir, optional, predictions_dir
from dashboard.charts import fig_pipeline

#: Sections per row. Two puts a 60-to-75-character measure under each heading, which is a
#: readable column rather than the 150-character line a full-width paragraph would be at
#: 1440 px — and it halves what four sections of prose cost the page in height.
SECTION_COLUMNS = 2

#: Sections above the pipeline diagram. Introduction and Methods, so the figure lands where
#: it is being described and breaks the prose in the middle rather than trailing it. Both
#: carry typed sentences and therefore always render, so the split does not move on a
#: half-built repo.
SECTIONS_ABOVE_DIAGRAM = 2

#: The route grid. Nine pages divide evenly by three, and at three across a blurb stays on
#: one line — a four-across grid left a ragged row of one and cost the same height.
ROUTE_COLUMNS = 3

# Bound 1 is a *measurement*, not an intention, and this block is most of how it is met.
# Streamlit's defaults are laid out for a scrolling document: 6 rem of padding above the
# first element, a 2.5 rem `h1`, and 1 rem between every vertical block. On a page whose
# only hard constraint is where it ends, that is 150-odd pixels spent on air before a word
# is read. Measured in Chrome at 1440x900: the tile version came in at 1,144 px with the
# defaults and 702 px with this block plus shorter copy, and the paper below ships at
# 1,057 px under the amended bound.
#
# It is scoped to the main container so it cannot reach the sidebar, and it lives in the
# view rather than in `shell.py` because it is a layout choice for one page — a model page
# is *supposed* to scroll, and inheriting this would only make it scroll further.
#
# The `h4` rule is the section heading. Streamlit sizes headings for a document with one
# `h1` and a handful of sections under it; four of them on a landing page need to read as
# labels on a paragraph rather than as chapters, and the margin above each is what a
# heading costs four times over.
COMPACT_CSS = """
<style>
  [data-testid="stMainBlockContainer"] { padding-top: 3rem; padding-bottom: 1rem; }
  [data-testid="stMainBlockContainer"] h1 {
    font-size: 2.1rem; line-height: 1.3; padding: 0.2rem 0 0.2rem 0;
  }
  [data-testid="stMainBlockContainer"] h4 {
    font-size: 1.05rem; line-height: 1.3; padding: 0 0 0.15rem 0; font-weight: 600;
  }
  [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap: 0.65rem; }
  [data-testid="stMainBlockContainer"] p { margin-bottom: 0.25rem; }
</style>
"""


@st.cache_data(show_spinner="Reading the headline artifacts…")
def load() -> dict[str, pd.DataFrame]:
    """Whichever of the ten sources are on disk, keyed by `overview.Source.key`.

    Partial by design. `optional()` names the missing file's `make` target on the page, and
    the readings that needed it drop out — a landing page should degrade to a shorter
    paper, not to a stack trace or to a screen of warnings with nothing under them.
    """
    roots = {overview.EDA: eda_dir(), overview.PREDICTIONS: predictions_dir()}
    frames = {}
    for source in overview.SOURCES:
        frame = optional(roots[source.directory] / source.filename,
                         target=source.target)
        if frame is not None:
            frames[source.key] = frame
    return frames


def title() -> None:
    """The one line of typed prose outside a section, and it states no result."""
    st.title("Drafting an NBA season, before it starts")


def band(paragraphs: list) -> None:
    """One row of sections, side by side.

    Heading and paragraph go out in **one** `st.markdown` call rather than two, because two
    are two vertical blocks with the container's gap between them — which is a heading
    floating off its own text, four times over.
    """
    if not paragraphs:
        return
    columns = st.columns(max(len(paragraphs), SECTION_COLUMNS))
    for col, para in zip(columns, paragraphs):
        col.markdown(f"#### {para.heading}\n\n{para.text}")


def pipeline(frames: dict[str, pd.DataFrame], th: dict) -> None:
    stages = overview.stage_readings(frames)
    if not stages:
        return
    wrapped = [stage._replace(note=overview.wrap(stage.note)) for stage in stages]
    st.plotly_chart(fig_pipeline(wrapped, th), width="stretch", key="pipeline",
                    config={"displayModeBar": False, "staticPlot": True})


def provenance() -> None:
    """Bound 2, said once where a reader can see it.

    The tiles used to carry this per figure, in a `help` tooltip naming the artifact. Prose
    has no hover, and putting nine filenames in the text would be the provenance block
    bound 3 forbids — so the qualifiers that mattered (which split, which head family, how
    many seasons) moved into the sentences themselves and this is what is left.
    """
    st.caption(
        "Every figure above is read from an artifact a `make` target wrote, never typed "
        "into the page — a re-run that moves a number moves this paragraph.")


def routes() -> None:
    """The way in. Nine page links in three rows, each with one line of what it holds.

    `shell.page` hands back the `st.Page` the entrypoint registered, because `st.page_link`
    accepts only those; outside the shell it returns None and the row degrades to its
    blurb. The links carry no numbers — a route label is exactly where a stray headline
    would try to creep back in past bound 2.
    """
    st.markdown("**Where the work is**")
    # A fresh `st.columns` per row rather than one grid filled column-major. A column is a
    # vertical stack, so under one grid a blurb that wraps to two lines in the first row
    # pushes only *its* column's second link down and the bottom row comes out ragged —
    # visible in a screenshot, invisible to `AppTest`, which sees nine page links either
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
    paragraphs = overview.paragraphs(frames)
    title()
    band(paragraphs[:SECTIONS_ABOVE_DIAGRAM])
    pipeline(frames, th)
    band(paragraphs[SECTIONS_ABOVE_DIAGRAM:])
    provenance()
    routes()
