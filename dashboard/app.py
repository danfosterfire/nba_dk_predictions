"""Player style fingerprints — one player-season's PCA scores, read radially.

Pick a season and a player. The chart puts each of the first ten principal components on
its own spoke and the player's score for that component on the radius, in standard
deviations from the league mean, on an axis fixed at ±2 SD for every player and every
season — so two fingerprints differ in *shape*, never in scale.

**Click a spoke** and the panel beside the chart explains that component: its strongest
loadings, the rotation player-seasons at either end of the axis, and what the loadings
say. One component at a time, because ten panels around the chart left no room to read
any of them.

Everything on the page is read from what `make pca` wrote. The only typed content is the
ten component titles, which are an interpretation of the loadings and are pinned to a
direction by `pca.COMPONENTS[...].anchor` so a refit cannot silently invert one.

Run with `make dashboard`.
"""

import sys
from pathlib import Path

# The repo root on the path so `dashboard.*` resolves: `streamlit run` puts the
# *script's* directory on sys.path, not the project root, so the package would not
# otherwise import. Nothing from `src/` is imported here — the dashboard reads
# artifacts and nothing else, and a test pins it.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard import pca
from dashboard.artifacts import features_dir, optional, rel
from dashboard.charts import fig_loadings, fig_radar
from dashboard.theme import theme

TOP_LOADINGS = 10
NEIGHBOURS = 3
NO_OVERLAY = "— none —"

# Streamlit's metric tiles are sized for a three-tile hero row; five of them clipped
# their own values. A static style block is the smallest fix that keeps the tiles —
# no data reaches it, so there is nothing for the HTML escape to matter to.
TILE_CSS = """
<style>
  [data-testid="stMetricValue"] { font-size: 1.4rem; line-height: 1.5rem; }
  [data-testid="stMetricLabel"] p { font-size: 0.75rem; }
  [data-testid="stMetric"] { padding: 0.2rem 0 0 0; }
</style>
"""


# ── Loading ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Reading the PCA artifacts…")
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Scores, loadings and variance, oriented so every labelled axis points its way."""
    directory = features_dir()
    scores = optional(directory / pca.SCORES_FILE)
    loadings = optional(directory / pca.LOADINGS_FILE)
    variance = optional(directory / pca.VARIANCE_FILE)
    if scores is None or loadings is None or variance is None:
        return None
    scores, loadings, _ = pca.orient(scores, loadings)
    return scores, loadings, variance


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


def click_target(event) -> list:
    """The clicked points out of a `plotly_chart` selection, whatever shape it arrives in."""
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return []
    points = (selection.get("points") if hasattr(selection, "get")
              else getattr(selection, "points", None))
    return list(points or [])


# ── Panels ────────────────────────────────────────────────────────────────────

def header_tiles(row: pd.Series) -> None:
    tiles = [("Team", str(row["team_abbreviation"]), "Season-end team in the box scores"),
             ("Age", f"{row['age']:.0f}", "Age during the season"),
             ("Games", f"{row['gp']:.0f}", "Games played"),
             ("Minutes / g", f"{row['min']:.1f}", "Minutes per game"),
             ("DK pts / g", f"{row['dk_pts_per_game']:.1f}",
              "DraftKings fantasy points per game — the project's target")]
    for col, (label, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, help=helptext)


def component_panel(pc: str, loadings: pd.DataFrame, scores: pd.DataFrame,
                    fingerprint: pd.DataFrame, share: dict[str, float],
                    th: dict) -> None:
    """The one component the reader clicked, with room to actually read it."""
    component = pca.BY_PC[pc]
    row = fingerprint[fingerprint["pc"] == pc].iloc[0]
    pct = share.get(pc)

    st.markdown(f"### {pc.upper()} · {component.title}")
    if pct is not None:
        st.caption(f"{pct:.1%} of the matrix's variance")
    st.metric("This player-season", f"{row['sd']:+.1f} SD",
              help="Distance from the league mean on this component."
                   + (" Outside the ±2 SD axis, so the chart pins it."
                      if row["pinned"] else ""))

    frame = pca.top_loadings(loadings, pc, TOP_LOADINGS)
    st.plotly_chart(fig_loadings(frame, th, "Strongest loadings"),
                    width="stretch", key=f"loadings-{pc}")

    high, low = pca.exemplars(scores, pc)
    st.markdown(f"**Most positive** — {high}  \n**Most negative** — {low}")
    st.caption(f"Rotation seasons only: {pca.EXEMPLAR_MIN_MPG:.0f}+ minutes a game over "
               f"{pca.EXEMPLAR_MIN_GP}+ games. Unfiltered, the extremes are "
               "sub-500-minute players whose rate stats are noise.")

    st.markdown(component.reads)
    st.caption(f"Direction anchored on `{component.anchor}` loading positive, so a refit "
               "that flips the axis cannot silently invert this reading.")

    with st.expander("Table view — the loadings above"):
        st.dataframe(frame[["feature", "loading"]].round({"loading": 4}),
                     hide_index=True, width="stretch")


def neighbour_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.rename(columns={
        "player_name": "Player", "season": "Season", "team_abbreviation": "Team",
        "age": "Age", "gp": "GP", "min": "MPG", "dk_pts_per_game": "DK pts / g",
        "distance": "Distance"})
    return out[["Player", "Season", "Team", "Age", "GP", "MPG", "DK pts / g",
                "Distance"]].round({"Age": 0, "MPG": 1, "DK pts / g": 1,
                                    "Distance": 2})


# ── Page ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="NBA player style fingerprints", layout="wide",
                       page_icon="🏀")
    st.markdown(TILE_CSS, unsafe_allow_html=True)
    st.title("Player style fingerprints")
    st.caption("Where one player-season sits on the ten largest axes of variation in "
               "the season matrix — each spoke a principal component, each radius that "
               "component's score in standard deviations from the league mean.")

    loaded = load()
    if loaded is None:
        st.stop()
    scores, loadings, variance = loaded
    share = pca.variance_share(variance)

    with st.sidebar:
        st.header("Appearance")
        default = detected_mode()
        appearance = st.radio("Mode", ["light", "dark"],
                              index=0 if default == "light" else 1, horizontal=True,
                              label_visibility="collapsed",
                              help="Chart steps are selected per mode, not flipped.")
        st.markdown("---")
        st.header("Nearest neighbours")
        same_season = st.toggle(
            "Same season only", value=False,
            help="Off: the closest player-seasons anywhere in the 30 seasons. On: only "
                 "within the selected season. A player's own other seasons are always "
                 "excluded — they are usually his three nearest neighbours, which is "
                 "true and uninformative.")
        st.markdown("---")
        st.caption(
            f"Tier A, `within_season`: {scores['season'].nunique()} seasons, "
            f"{len(scores):,} qualified player-seasons, "
            f"{len(loadings):,} box-score features. Read from "
            f"`{rel(features_dir() / pca.SCORES_FILE)}` and its two siblings — "
            "reproduce with `make pca`. Nothing on this page is refitted.")

    seasons = sorted(scores["season"].unique(), reverse=True)
    pick_season, pick_player, _ = st.columns([1, 2, 3])
    season = pick_season.selectbox("Season", seasons, index=0)
    in_season = scores[scores["season"] == season].sort_values("player_name")
    player = pick_player.selectbox("Player", in_season["player_name"].tolist())

    row = in_season[in_season["player_name"] == player].iloc[0]
    th = theme(appearance)
    frame = pca.fingerprint(scores, int(row["player_id"]), season)
    table = pca.neighbors(scores, int(row["player_id"]), season, k=NEIGHBOURS,
                          same_season_only=same_season)

    st.subheader(f"{player} · {season}")
    header_tiles(row)
    st.markdown("---")

    chart, panel = st.columns([1.2, 1], gap="large")

    with chart:
        # Rendered above the chart rather than beside the neighbour table, so the
        # overlay's value is known before the figure it changes is built.
        options = [NO_OVERLAY] + [f"{r.Player} · {r.Season}"
                                  for r in neighbour_table(table).itertuples()]
        choice = st.selectbox(
            "Compare with a nearest neighbour", options, index=0,
            help="Draws that player-season's fingerprint over this one, unfilled.")
        overlay, overlay_name = None, ""
        if choice != NO_OVERLAY:
            match = table.iloc[options.index(choice) - 1]
            overlay = pca.fingerprint(scores, int(match["player_id"]),
                                      str(match["season"]))
            overlay_name = choice

        selected = st.session_state.get("component", pca.PC_NAMES[0])
        if selected not in pca.BY_PC:
            selected = pca.PC_NAMES[0]
        event = st.plotly_chart(
            fig_radar(frame, th, f"{player} · {season}", limit=pca.AXIS_LIMIT,
                      overlay=overlay, overlay_name=overlay_name,
                      selected=selected.upper()),
            width="stretch", key="radar", on_select="rerun", selection_mode="points",
            # Drawing the disc on cartesian axes brings plotly's zoom/pan modebar with
            # it, and neither gesture means anything on a fixed ±2 SD axis.
            config={"displayModeBar": False})

        clicked = pca.component_from_click(click_target(event))
        if clicked is not None and clicked != selected:
            # Written before the panel's selectbox is instantiated, which is what lets a
            # click and the selectbox drive one value instead of two that drift apart.
            st.session_state["component"] = clicked
            st.rerun()

        st.caption(":material/ads_click: Click a spoke to read that component beside "
                   "the chart.")

        pinned = frame[frame["pinned"]]
        if len(pinned):
            st.caption(
                ":material/push_pin: Outside ±2 SD and pinned, drawn as an open "
                "marker — " +
                ", ".join(f"**{r.label}** {r.sd:+.1f} SD" for r in pinned.itertuples())
                + ". A positive pin sits on the rim, a negative one on the centre.")
        else:
            st.caption("Every component falls inside ±2 SD — nothing is pinned.")

        with st.expander("Table view — every component score"):
            st.dataframe(
                frame[["label", "title", "sd", "pinned"]].rename(columns={
                    "label": "Component", "title": "Reads as", "sd": "Score (SD)",
                    "pinned": "Pinned"}).round({"Score (SD)": 1}),
                hide_index=True, width="stretch")

    with panel:
        st.selectbox(
            "Component", pca.PC_NAMES, key="component",
            format_func=lambda p: f"{p.upper()} · {pca.BY_PC[p].title}",
            help="The same choice the chart's spokes make, for anyone not using a "
                 "mouse.")
        component_panel(st.session_state["component"], loadings, scores, frame,
                        share, th)

    st.markdown("---")
    st.subheader("Nearest neighbours in PCA space")
    st.caption(
        f"Euclidean distance over the raw scores of PC1–PC{pca.N_COMPONENTS}, which is "
        "distance in the standardized feature space the PCA was fitted on — so the "
        "high-variance style axes dominate, as they should. "
        + (f"Restricted to {season}." if same_season else "Across all 30 seasons."))
    if len(table):
        st.dataframe(neighbour_table(table), hide_index=True, width="stretch")
    else:
        st.info("No comparable player-seasons under the current filter.")


if __name__ == "__main__":
    main()
