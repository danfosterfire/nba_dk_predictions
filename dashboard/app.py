"""The dashboard entrypoint — an `st.navigation` shell over the view pages.

Run it with `make dashboard`. The pages themselves are in `dashboard/views/`, one module
each behind a `render()`; this file owns only what is true of the whole surface: the page
config, the navigation, and the cross-page state in `dashboard/shell.py`.

## Why navigation rather than tabs

`st.tabs` would have been fewer lines and is the wrong container, for a reason that is
structural rather than cosmetic: **Streamlit executes the body of every tab on every
rerun.** Tabs are a client-side affordance — the Python inside each `with tab:` block runs
whether or not that tab is visible, and the inactive content is hidden with CSS. Nine tabs
would therefore mean that every interaction anywhere re-runs all nine, including whichever
one loads the 90 MB simulation tensor, and no amount of caching fixes it because the cost
is the *rendering* and not only the I/O.

`st.navigation` / `st.Page` runs only the selected page's script. The pages share one
server process, so `@st.cache_data` and `@st.cache_resource` are shared across them and a
tensor loaded by one page stays warm if the reader navigates away and back. That is the
only structure in which the live draft board can be a page at all, rather than a separate
app. See `docs/dashboard-plan.md`, "The expansion".

## What the entrypoint owes a page

The body of this file runs on **every** rerun; a `render()` runs only when its page is
selected. So anything that must survive navigation is rendered here, from
`dashboard/shell.py`, and a page reads it — today the light/dark appearance mode, which
would otherwise reset to the detected default each time the reader changed page. Sidebar
order follows that ownership: the navigation, then the shell's controls, then whatever the
page writes to `st.sidebar` for itself.

Nothing here imports from `src/`. The dashboard reads artifacts and nothing else, and
`tests/test_dashboard.py` pins it for every file in the package.
"""

import sys
from pathlib import Path
from typing import Callable, NamedTuple

# The repo root on the path so `dashboard.*` resolves: `streamlit run` puts the
# *script's* directory on sys.path, not the project root, so the package would not
# otherwise import. Nothing from `src/` is imported here — the dashboard reads
# artifacts and nothing else, and a test pins it.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard import shell
from dashboard.views import fingerprints, placeholder


class View(NamedTuple):
    """One entry in the sidebar. `url_path` is pinned so a link outlives a retitling."""

    render: Callable[[], None]
    title: str
    icon: str
    url_path: str


# Sidebar order. `docs/dashboard-plan.md` specifies seven more pages beyond these two;
# each arrives as one module in `dashboard/views/` and one row here, and nothing else in
# this file moves.
#
# The second row is a placeholder, and it is here for a measured reason rather than for
# looks: **Streamlit draws no navigation widget for a one-page app**, so a shell shipped
# with only the fingerprint view would be indistinguishable from the single-page script it
# replaced and could not be verified in a browser. It is the *next* page in the build
# order, so step 2 replaces this row rather than adding to it. See
# `dashboard/views/placeholder.py`.
VIEWS: tuple[View, ...] = (
    View(fingerprints.render, "Player fingerprints", ":material/radar:", "fingerprints"),
    View(placeholder.planned(
        "Tournament & strategy", "step 2 of the build order",
        "The contest structure, the strategy sweep, simulated against realized, and the "
        "paired comparisons whose intervals cross zero.",
        "what `make strategy-sweep` and `make bracket` have already written, plus the "
        "contest arithmetic in `dashboard/economics.py` — all on disk today, none of it "
        "drawn."),
        "Tournament & strategy", ":material/trophy:", "tournament"),
)


def pages() -> list[st.Page]:
    """`VIEWS` as `st.Page` objects. The first is the default, i.e. what `/` serves."""
    return [st.Page(view.render, title=view.title, icon=view.icon,
                    url_path=view.url_path, default=(i == 0))
            for i, view in enumerate(VIEWS)]


def main() -> None:
    st.set_page_config(page_title="NBA best ball", layout="wide", page_icon="🏀")
    page = st.navigation(pages())
    shell.appearance_control()
    page.run()


if __name__ == "__main__":
    main()
