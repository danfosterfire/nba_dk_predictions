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
"""

import streamlit as st

from dashboard.theme import theme

MODES = ("light", "dark")
APPEARANCE = "appearance"


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
