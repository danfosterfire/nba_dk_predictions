"""Cross-page state — the things a view must not own, because it does not always run.

`app.py` is the `st.navigation` entrypoint and its body executes on **every** rerun,
whichever page is selected; a view module's `render()` executes only when its own page
is. That asymmetry is the whole reason this module exists. Anything that has to survive
navigation is rendered by the entrypoint from here, and a view *reads* it rather than
declaring it — a control declared inside a view would be torn down the moment the reader
navigated away, taking its `st.session_state` entry with it, and the next page would come
up in whatever the detected default happened to be.

Today that is one thing, the light/dark appearance mode. It is genuinely global: the
palette is selected per mode rather than flipped (`theme.py`), so a reader who picked dark
on one page and got light on the next would be reading two different validated palettes in
one session.

The mode lives under a single key, `APPEARANCE`, seeded once from `detected_mode()` and
thereafter owned by the widget. `current_theme()` is what a view calls; it never touches
the key directly, so the storage can change without a view knowing.

`remember()` / `recall()` are the other half of the same asymmetry, for a control the
*entrypoint* cannot render on a page's behalf — see their own docstrings.
"""

from typing import TypeVar

import streamlit as st

from dashboard.theme import theme

MODES = ("light", "dark")
APPEARANCE = "appearance"

# Shadow keys are namespaced so they cannot collide with a widget key, which would put two
# owners on one entry — the same argument `appearance_control` makes about `index=`.
REMEMBERED = "remembered:"

V = TypeVar("V")

# Streamlit's metric tiles are sized for a three-tile hero row, and clip their own values
# past four across — caught in a browser on the fingerprint page's five-tile header, where
# the test IDs below were read off this Streamlit's frontend bundle rather than assumed.
#
# It lives here rather than in a view because two pages now want it, but it is **opt-in**
# rather than applied by the entrypoint: a page that has not been laid out yet should not
# silently inherit a type scale chosen for somebody else's header. No data reaches the
# block, so there is nothing for the HTML escape to matter to.
TILE_CSS = """
<style>
  [data-testid="stMetricValue"] { font-size: 1.4rem; line-height: 1.5rem; }
  [data-testid="stMetricLabel"] p { font-size: 0.75rem; }
  [data-testid="stMetric"] { padding: 0.2rem 0 0 0; }
</style>
"""


def compact_tiles() -> None:
    """Opt into the tile type scale that fits four or more metrics across a row."""
    st.markdown(TILE_CSS, unsafe_allow_html=True)


def detected_mode() -> str:
    """Follow Streamlit's own theme where it exposes one."""
    for get in (lambda: st.context.theme.type, lambda: st.get_option("theme.base")):
        try:
            value = get()
        except Exception:
            continue
        if value in MODES:
            return value
    return "light"


def mode() -> str:
    """The selected appearance, safe to call before the control has been rendered."""
    value = st.session_state.get(APPEARANCE)
    return value if value in MODES else detected_mode()


def current_theme() -> dict:
    """The palette every chart on every page is built against."""
    return theme(mode())


def recall(key: str, default: V) -> V:
    """What this control held before the reader last left the page, or `default`.

    Streamlit clears the state of every widget the current page did not render, so a
    control declared inside a `render()` comes back at its default after a navigation —
    measured, and the reason `appearance_control` lives in the entrypoint. A page whose
    controls *say what its other state means* cannot use that escape: the draft room's
    seat, season and tournament are only meaningful beside a pick log that is a plain
    session-state key and therefore does survive, so a silent reset to seat 1 would replay
    a real pod against the wrong roster rather than lose it.

    A plain key written by the page while it runs is not widget state and is not cleared,
    so the pair here is the page's own memory: `recall` seeds the widget, `remember` stores
    what it came back with. It is deliberately *not* an alternative to putting genuinely
    global state in this module — the appearance mode belongs to every page and is rendered
    by the entrypoint; these belong to one page and only have to outlive leaving it.
    """
    return st.session_state.get(f"{REMEMBERED}{key}", default)


def remember(key: str, value: V) -> V:
    """Store a control's value under `key` and hand it straight back."""
    st.session_state[f"{REMEMBERED}{key}"] = value
    return value


def appearance_control() -> str:
    """The sidebar mode switch, rendered by the entrypoint so it outlives a page.

    Seeded rather than defaulted: passing `index=` alongside a key Streamlit already
    holds a value for makes the widget argue with session state. Seeding the key once
    and then omitting `index=` leaves one owner.
    """
    if st.session_state.get(APPEARANCE) not in MODES:
        st.session_state[APPEARANCE] = detected_mode()
    with st.sidebar:
        st.header("Appearance")
        st.radio("Mode", list(MODES), key=APPEARANCE, horizontal=True,
                 label_visibility="collapsed",
                 help="Chart steps are selected per mode, not flipped. The choice is "
                      "held by the shell, so it survives moving between pages.")
        st.markdown("---")
    return st.session_state[APPEARANCE]
