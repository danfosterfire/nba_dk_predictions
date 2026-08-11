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
app — and since 2026-08-10 it is one. Measured in Chrome rather than argued: the room's
first paint is 3.17 s selected from the navigation against 3.63 s for a cold
`make draft-room`, and **0.31 s** on a return visit, because the ~40 MB reference field is
still in `st.cache_resource` from the first one. See `docs/dashboard-plan.md`, "The
expansion" and "Step 7, as built".

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

from dashboard import model_cards, shell
from dashboard.views import (availability, beyond_heads, components, draft_room,
                             fingerprints, game_length, minutes, tournament)


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
# **The navigation must keep at least two rows**, and that is measured rather than
# stylistic: Streamlit draws no navigation widget at all for a one-page app, so a shell
# that dropped back to one page would be indistinguishable in a browser from the
# single-page script it replaced. The second row held a placeholder from the shell
# landing on 2026-08-10 until the tournament page replaced it the same day, exactly as
# the build order specified.
def model_view(render: Callable[[], None], class_key: str) -> View:
    """A model-detail page's row, titled from `model_cards.CLASSES` rather than here.

    The four model pages are one renderer over one class table; taking the title, icon and
    `url_path` from that table means adding page 4, 5 or 6 is a view module and this one
    line, and that a page's name cannot disagree between the navigation and the page.
    """
    spec = model_cards.CLASSES[class_key]
    return View(render, spec.title, spec.icon, spec.url_path)


VIEWS: tuple[View, ...] = (
    View(fingerprints.render, "Player fingerprints", ":material/radar:", "fingerprints"),
    model_view(availability.render, availability.CLASS_KEY),
    model_view(minutes.render, minutes.CLASS_KEY),
    model_view(components.render, components.CLASS_KEY),
    model_view(game_length.render, game_length.CLASS_KEY),
    View(beyond_heads.render, "Inputs beyond the heads", ":material/inventory_2:",
         "inputs"),
    View(tournament.render, "Tournament & strategy", ":material/trophy:", "tournament"),
    # Last, and the one row whose page is also its own app: `make draft-room` launches
    # `dashboard/draft_room.py` directly for draft night. The row costs nothing until it is
    # selected — `views/draft_room.py` defers the import that reaches `src.sim`, so a
    # reader who never opens the room never loads the simulation layer.
    # The icon is a ranked list rather than a basketball because `model_cards.CLASSES`
    # already spends `sports_basketball` on the box-score page, and two identical icons in
    # one navigation is a row a reader has to read twice.
    View(draft_room.render, "Draft board", ":material/format_list_numbered:", "draft-room"),
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
