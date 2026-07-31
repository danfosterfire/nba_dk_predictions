"""Shared Streamlit presentation pieces.

Two rules from `docs/dashboard-plan.md` are enforced here rather than remembered:

- **Altitude is the feature.** Each tab opens with its decisions and a handful of
  hero numbers; every table, matrix and per-column ranking goes behind an
  expander. `table_view` was already the accessible twin of a chart; `detail`
  extends the same idiom to "the detail behind a claim".
- **Every figure is read from an artifact a `make` target produced.** `provenance`
  prints that link under a panel, and `pending_marker` is what a panel renders
  instead of a typed number when no such artifact exists yet.

Badge colours below are Streamlit's own named chrome colours, deliberately not the
validated eight-slot series palette — a status chip is interface, not data, and
borrowing a data slot for it would imply an encoding that is not there.
"""

import pandas as pd
import streamlit as st

from dashboard.decisions import Decision, STATUSES, make_target

STATUS_COLOR: dict[str, str] = {
    "built": "green",
    "settled": "blue",
    "measured": "blue",
    "null": "gray",
    "withdrawn": "orange",
    "open": "violet",
    "blocked": "red",
    "deadline": "red",
    "incident": "orange",
}

STATUS_ICON: dict[str, str] = {
    "built": ":material/check_circle:",
    "settled": ":material/gavel:",
    "measured": ":material/straighten:",
    "null": ":material/block:",
    "withdrawn": ":material/undo:",
    "open": ":material/pending:",
    "blocked": ":material/lock:",
    "deadline": ":material/schedule:",
    "incident": ":material/warning:",
}


def note(text: str) -> None:
    st.caption(text)


def table_view(df: pd.DataFrame, label: str = "Table view") -> None:
    """The WCAG-clean twin every chart ships with — values never live in colour alone."""
    with st.expander(label):
        st.dataframe(df, width="stretch", hide_index=True)


def detail(label: str = "Detail"):
    """The altitude rule as a context manager: a table is one click below a claim."""
    return st.expander(label)


def stat_tiles(items: list[tuple[str, str, str]]) -> None:
    """A row of hero numbers — the right form when the story is one value."""
    cols = st.columns(len(items))
    for col, (label, value, helptext) in zip(cols, items):
        col.metric(label, value, help=helptext)


def status_badge(status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"{status!r} is not in the closed status vocabulary")
    st.badge(status, icon=STATUS_ICON[status], color=STATUS_COLOR[status])


def provenance(text: str) -> None:
    """The make-target → artifact link under a panel."""
    st.caption(f":material/link: {text}")


def pending_marker(target: str, figure: str) -> None:
    """Stand in for a figure no `make` target produces yet.

    Deliberately loud, and deliberately *not* the number: the provenance rule says
    a figure that cannot be reproduced does not go on the page, and a visible gap
    naming its target is more honest than a typed constant that looks identical to
    a live artifact read. `make dashboard-audit` counts these, so the count trends
    to zero.

    All ten items in `docs/provenance-plan.md` landed, so nothing calls this today —
    it stays because the next unbacked figure should hit a marker rather than a
    tempting hard-coded value.
    """
    st.warning(f":material/hourglass_top: **Pending provenance** — {figure} needs "
               f"`{target}`, which does not write it yet. Deliberately not typed in.")


def decision_card(d: Decision, show_topic: bool = False) -> None:
    """One registry entry, rendered at altitude.

    A `withdrawn` entry renders its replacement and what caught it, because for
    this reader those are the most valuable rows in the registry — the reversals
    are content, not embarrassment.
    """
    with st.container(border=True):
        left, right = st.columns([6, 1], vertical_alignment="top")
        with left:
            st.markdown(f"**{d.claim}**")
        with right:
            status_badge(d.status)

        st.markdown(d.because)

        if d.status == "withdrawn":
            if d.replaced_by:
                st.markdown(f":material/arrow_forward: **Replaced by** — {d.replaced_by}")
            if d.caught_by:
                st.markdown(f":material/search: **Caught by** — {d.caught_by}")
        if d.status == "blocked" and d.unblocks:
            st.markdown(f":material/key: **Unblocked by** — {d.unblocks}")
        if d.status == "deadline" and d.due:
            st.markdown(f":material/schedule: **Due** — {d.due}")

        bits = []
        if show_topic:
            bits.append(f"topic `{d.topic}`")
        if d.reproduce:
            bits.append(f"reproduce `{make_target(d)}`")
        else:
            # An incident carries a date and a doc reference instead of a number.
            bits.append("no live figure — dated diagnosis")
        bits.append(f"source `{d.source}`")
        bits.append(f"decided {d.date} · reviewed {d.reviewed}")
        st.caption(" · ".join(bits))


def decision_cards(entries, show_topic: bool = False) -> None:
    for d in entries:
        decision_card(d, show_topic=show_topic)


def tab_header(title: str, blurb: str) -> None:
    st.subheader(title)
    note(blurb)
