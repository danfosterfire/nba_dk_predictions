"""Project walkthrough — nine tabs following the build end to end.

Page config, sidebar and tab dispatch, and nothing else: the panels live in
`dashboard/tabs/`, the palette in `theme.py`, the figure builders in `charts.py`.

**Audience: the project architect, wanting a birds-eye view of the decisions.**
That sets the two rules the tabs are written to. *Altitude is the feature* — a tab
opens with its decisions and a handful of hero numbers, and every table and
per-column ranking goes behind an expander. *Reversals are content* — `withdrawn`
is a first-class status rather than a deletion, because this project's record of
catching its own false findings is among the most useful things on the page.

**Every figure here is read from an artifact a `make` target produced.** Nothing
refits, and there is no import from `src/` anywhere in the package. Run with
`make dashboard`.
"""

import sys
from pathlib import Path

# The repo root on the path so `dashboard.*` resolves: `streamlit run` puts the
# *script's* directory on sys.path, not the project root, so the package would not
# otherwise import. Nothing from `src/` is imported here or in any tab — the
# dashboard reads artifacts and nothing else, and a test pins it.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.artifacts import Ctx, pipeline_health
from dashboard.tabs import TABS, TAB_NAMES


def detected_mode() -> str:
    """Follow Streamlit's own theme where it exposes one."""
    for get in (lambda: st.context.theme.type, lambda: st.get_option("theme.base")):
        try:
            value = get()
        except Exception:
            continue
        if value in ("light", "dark"):
            return value
    return "light"


def sidebar() -> Ctx:
    from dashboard.theme import theme

    with st.sidebar:
        st.header("Appearance")
        default = detected_mode()
        appearance = st.radio("Mode", ["light", "dark"],
                              index=0 if default == "light" else 1,
                              horizontal=True, label_visibility="collapsed",
                              help="Chart steps are selected per mode, not flipped.")

        st.markdown("---")
        # Tier and era mode scope the coverage heatmap and nothing else now that
        # tab 3's figures are deferred. Leaving them under a global "Scope" heading
        # would imply the availability head has a tier, which it does not.
        st.header("Data scope")
        st.caption("Scopes the coverage panels on **Data collection** only. The head "
                   "tabs are fitted on one frame each and have no tier.")
        tier = st.radio("Tier", ["A", "B"], horizontal=True,
                        help="A: 30 seasons, box-score families. "
                             "B: 13 seasons (2013-14+), adds tracking and hustle.")
        mode = st.radio("Era mode", ["within_season", "pooled"],
                        help="within_season is era-neutral and feeds modeling; "
                             "pooled makes era a visible axis.")

        st.markdown("---")
        st.header("Pipeline health")
        health = pipeline_health()
        if health.empty:
            st.caption("No expected artifacts declared.")
        else:
            present = int(health["present"].sum())
            total = len(health)
            st.metric("Artifacts present", f"{present} / {total}",
                      help="Every artifact the decision registry names, checked on "
                           "disk. This is the same set `make dashboard-audit` checks.")
            missing = health[~health["present"]]
            if missing.empty:
                st.caption(":material/check_circle: Everything the registry expects "
                           "is on disk.")
            else:
                with st.expander(f"{len(missing)} missing", expanded=False):
                    for row in missing.itertuples():
                        st.markdown(f"`{row.artifact}` — run `{row.target}`")

        st.markdown("---")
        st.caption("Reads `data/features/`, `outputs/eda/` and "
                   "`outputs/predictions/`. Nothing is refitted here. "
                   "`make dashboard-audit` reports registry drift.")

    return Ctx(th=theme(appearance), appearance=appearance, tier=tier, mode=mode)


def main() -> None:
    st.set_page_config(page_title="NBA season projections — walkthrough",
                       layout="wide", page_icon="🏀")
    st.title("Predicting a season of DraftKings fantasy points")
    st.caption("A walkthrough of the build, tab by tab. Every figure is read from an "
               "artifact a `make` target produced — nothing on this page is typed in.")

    ctx = sidebar()
    for tab, (_, render) in zip(st.tabs(list(TAB_NAMES)), TABS):
        with tab:
            render(ctx)


if __name__ == "__main__":
    main()
