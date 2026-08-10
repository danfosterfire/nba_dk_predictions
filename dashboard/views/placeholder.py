"""A page that is specified but not built — scaffolding, and deliberately temporary.

**Streamlit renders no navigation widget for a one-page app.** That was measured, not
assumed: with a single `st.Page`, `st.navigation(position="sidebar")` sends
`Position.SIDEBAR` and the frontend draws nothing, so `[data-testid="stSidebarNav"]` is
absent from the DOM. A shell shipped with only the fingerprint view would therefore look
byte-for-byte like the single-page app it replaced, and neither the navigation nor the
cross-page appearance state in `dashboard/shell.py` could be verified in a browser at all.

So the shell ships with one placeholder alongside the one real page, and the placeholder
is the *next* page in `docs/dashboard-plan.md`'s build order rather than a lorem-ipsum
tab. Step 2 deletes it by replacing its row in `app.VIEWS` with a real view module; when
the expansion lands, this file goes with `docs/dashboard-build-prompts.md`.

Two rules a placeholder follows, both of them about not lying to a machine that is
checking:

- **It shows no numbers.** An unbuilt page has no artifact to read one from, and typing
  one would be the hand-typed claim the 2026-08-08 overhaul deleted a whole walkthrough
  over.
- **It names `make` targets, never artifact filenames.** `dashboard/audit.py`'s
  orphaned-artifact check counts an artifact as read when a string literal anywhere in
  `dashboard/` names it, so a placeholder listing `strategy_sweep.csv` would report a file
  as drawn that nothing draws — and the orphan count is how the plan decides what to build
  next. (Checked when this shipped: naming those files masked nothing today, because the
  registry already accounts for them. It would still have been a false signal.)
"""

from typing import Callable

import streamlit as st


def planned(title: str, step: str, summary: str, sources: str) -> Callable[[], None]:
    """A `render()` for a page the build order has specified and not yet built."""

    def render() -> None:
        st.title(title)
        st.caption(summary)
        st.info(f"**Not built yet — {step}.** This entry exists so the shell has a "
                "navigation to render and a second page to hold state across; it is "
                "replaced by the real view rather than added to.", icon=":material/hourglass:")
        st.markdown(f"**Will read** {sources}")
        st.markdown("The design is in `docs/dashboard-plan.md`, under "
                    "*The expansion — from one view to nine pages*.")

    return render
