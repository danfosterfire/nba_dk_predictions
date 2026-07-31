"""Tab 9 · Decision log.

The whole registry, filterable — plus the three summary reads that only exist once
the decisions are structured: the status mix, the reversal thread, and the deadline
board.

For this reader the reversal thread is the most valuable single view on the page,
which is why `withdrawn` is a first-class status rather than a deletion.
"""

from datetime import date

import pandas as pd
import streamlit as st

from dashboard import decisions as D
from dashboard.artifacts import Ctx
from dashboard.charts import fig_bars
from dashboard.layout import (decision_card, decision_cards, detail, note,
                              provenance, stat_tiles, status_badge, tab_header,
                              table_view)

# Roughly "how settled is this", for ordering the status-mix chart. Not a scale —
# `incident` and `deadline` are kinds of entry, not degrees of resolution.
STATUS_ORDER = ["settled", "built", "measured", "null", "withdrawn", "incident",
                "deadline", "blocked", "open"]


def render(ctx: Ctx) -> None:
    tab_header(
        "Decision log",
        "Every load-bearing decision in the project, in one structured record. "
        "`CLAUDE.md` and the `docs/*-plan.md` files remain the source of truth — this "
        "is a distillation of them for browsing, and where the two disagree the docs "
        "are right and this is stale.")

    _status_mix(ctx)
    _reversals(ctx)
    _deadlines(ctx)
    _blocked_and_open(ctx)
    _browser(ctx)


# ── Status mix ────────────────────────────────────────────────────────────────

def _status_mix(ctx: Ctx) -> None:
    mix = D.status_mix()
    total = len(D.REGISTRY)
    resolved = sum(mix[s] for s in ("settled", "built", "measured", "null"))

    stat_tiles([
        ("Decisions recorded", f"{total}",
         f"Across {len(D.TOPICS)} topics. Each renders on its own tab as well as "
         f"here."),
        ("Resolved", f"{resolved / total:.0%}",
         f"{resolved} entries that are settled, built, measured or a recorded null."),
        ("Reversals", f"{mix['withdrawn']}",
         "Previously believed, then falsified by a better measurement. Kept, never "
         "deleted."),
        ("Still open or blocked", f"{mix['open'] + mix['blocked']}",
         "Known gaps with no answer yet."),
    ])

    frame = pd.DataFrame({
        "status": [s for s in STATUS_ORDER if mix[s]],
        "count": [mix[s] for s in STATUS_ORDER if mix[s]],
    })
    st.plotly_chart(
        fig_bars(frame, "status", ["count"], ctx.th,
                 "How much of this project is settled against open",
                 axis_title="entries", height=340),
        width="stretch")
    with detail("What each status means"):
        st.dataframe(
            pd.DataFrame({"status": list(D.STATUSES),
                          "meaning": list(D.STATUSES.values()),
                          "entries": [mix[s] for s in D.STATUSES]}),
            width="stretch", hide_index=True)
    note("The vocabulary is **closed**. `null` is for nulls that are *measurements* "
         "and therefore carry an artifact; a dated diagnosis of an external system is "
         "an `incident`, rendered without a live-number claim because re-deriving it "
         "would mean re-probing a third party to no purpose.")


# ── The reversal thread ───────────────────────────────────────────────────────

def _reversals(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The reversal thread")
    note("Every belief this project has overturned, with what replaced it and what "
         "caught it. This is the most useful view on the page: a record of catching "
         "your own false findings is worth more than a record of never recording "
         "one. In every case here the recorded value was the wrong one, and in every "
         "case the conclusion it supported survived — twice more strongly than "
         "before.")

    withdrawn = D.by_status("withdrawn")
    if not withdrawn:
        st.info("No reversals recorded.")
        return
    for d in sorted(withdrawn, key=lambda x: x.date, reverse=True):
        with st.container(border=True):
            head, badge = st.columns([6, 1], vertical_alignment="top")
            with head:
                st.markdown(f"~~{d.claim}~~")
            with badge:
                status_badge(d.status)
            st.caption(f"topic `{d.topic}` · believed because: {d.because}")
            st.markdown(f":material/arrow_forward: **Replaced by** — {d.replaced_by}")
            st.markdown(f":material/search: **Caught by** — {d.caught_by}")
            bits = [f"source `{d.source}`", f"corrected {d.date}"]
            if d.reproduce:
                bits.insert(0, f"reproduce `{D.make_target(d)}`")
            st.caption(" · ".join(bits))


# ── The deadline board ────────────────────────────────────────────────────────

def _deadlines(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The deadline board")
    note("Work that is permanently lost if it is not done by a date. This is the only "
         "part of the project where waiting has an irreversible cost.")

    entries = D.by_status("deadline")
    if not entries:
        st.info("No deadlines recorded.")
        return

    today = date.today()

    def days_left(d: D.Decision) -> float:
        try:
            return (date.fromisoformat(d.due) - today).days
        except ValueError:
            return float("-inf")            # standing/rolling — always at the top

    for d in sorted(entries, key=days_left):
        left = days_left(d)
        with st.container(border=True):
            head, badge = st.columns([6, 1], vertical_alignment="top")
            with head:
                st.markdown(f"**{d.claim}**")
            with badge:
                if left == float("-inf"):
                    st.badge("standing", icon=":material/autorenew:", color="orange")
                elif left < 0:
                    st.badge(f"{abs(int(left))}d overdue",
                             icon=":material/priority_high:", color="red")
                else:
                    st.badge(f"{int(left)}d left", icon=":material/schedule:",
                             color="red" if left < 30 else "orange")
            st.markdown(d.because)
            st.caption(f"due {d.due} · source `{d.source}` · topic `{d.topic}`")


# ── Blocked and open ──────────────────────────────────────────────────────────

def _blocked_and_open(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### What is not done")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Blocked")
        entries = D.by_status("blocked")
        if entries:
            decision_cards(entries, show_topic=True)
        else:
            st.caption("Nothing blocked.")
    with right:
        st.markdown("#### Open")
        entries = D.by_status("open")
        if entries:
            decision_cards(entries, show_topic=True)
        else:
            st.caption("Nothing open.")


# ── The browser ───────────────────────────────────────────────────────────────

def _browser(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Browse the whole registry")

    c1, c2, c3 = st.columns([2, 2, 3])
    topics = c1.multiselect("topic", list(D.TOPICS),
                            format_func=lambda t: D.TOPIC_LABELS[t])
    statuses = c2.multiselect("status", [s for s in STATUS_ORDER
                                         if D.status_mix()[s]])
    query = c3.text_input("search claims and reasons", "")

    tags = sorted({t for d in D.REGISTRY for t in d.tags})
    picked_tags = st.multiselect("tag", tags)

    hits = list(D.REGISTRY)
    if topics:
        hits = [d for d in hits if d.topic in topics]
    if statuses:
        hits = [d for d in hits if d.status in statuses]
    if picked_tags:
        hits = [d for d in hits if set(d.tags) & set(picked_tags)]
    if query:
        q = query.lower()
        hits = [d for d in hits
                if q in d.claim.lower() or q in d.because.lower()
                or q in d.id.lower() or q in d.replaced_by.lower()]

    st.caption(f"{len(hits)} of {len(D.REGISTRY)} entries")

    frame = pd.DataFrame([{
        "id": d.id,
        "topic": d.topic,
        "status": d.status,
        "claim": d.claim,
        "reproduce": D.make_target(d),
        "source": d.source,
        "decided": d.date,
        "reviewed": d.reviewed,
        "tags": ", ".join(d.tags),
    } for d in hits])
    table_view(frame, "Matching entries — table view")

    for d in hits:
        decision_card(d, show_topic=True)

    provenance("`dashboard/decisions.py` — distilled from `CLAUDE.md` and the "
               "`docs/*-plan.md` files. `make dashboard-audit` checks every "
               "`reproduce` artifact exists and flags entries whose source doc has "
               "moved since `reviewed`.")
