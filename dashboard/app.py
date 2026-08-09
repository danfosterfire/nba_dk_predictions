"""Player style fingerprints — one player-season's PCA scores, read radially.

Pick a season and a player. The centre chart puts each of the first ten principal
components on its own spoke and the player's score for that component on the radius, in
standard deviations from the league mean, on an axis fixed at ±2 SD for every player and
every season — so two fingerprints differ in *shape*, never in scale. Around it, the
loadings that define each component, in the same angular order. Below it, the three
player-seasons nearest this one in PCA space.

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

TOP_LOADINGS = 8
NEIGHBOURS = 3


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


# ── Panels ────────────────────────────────────────────────────────────────────

def loading_panel(pc: str, loadings: pd.DataFrame, scores: pd.DataFrame,
                  share: dict[str, float], th: dict) -> None:
    """One component: its title, its top loadings, and the seasons at either end."""
    component = pca.BY_PC[pc]
    frame = pca.top_loadings(loadings, pc, TOP_LOADINGS)
    pct = share.get(pc)
    heading = f"{pc.upper()} · {component.title}"
    if pct is not None:
        heading += f"  ({pct:.1%} of variance)"
    st.plotly_chart(fig_loadings(frame, th, heading),
                    width="stretch", key=f"loadings-{pc}")

    high, low = pca.exemplars(scores, pc)
    st.caption(f"**+** {high}  ·  **−** {low}")
    with st.expander("What this axis reads", expanded=False):
        st.markdown(component.reads)
        st.caption(f"Direction anchored on `{component.anchor}` loading positive.")
        st.dataframe(frame[["feature", "loading"]], hide_index=True,
                     width="stretch")


def header_tiles(row: pd.Series) -> None:
    tiles = [("Team", str(row["team_abbreviation"]), "Season-end team in the box scores"),
             ("Age", f"{row['age']:.0f}", "Age during the season"),
             ("Games", f"{row['gp']:.0f}", "Games played"),
             ("Minutes", f"{row['min']:.1f}", "Minutes per game"),
             ("DK pts / g", f"{row['dk_pts_per_game']:.1f}",
              "DraftKings fantasy points per game — the project's target")]
    for col, (label, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, help=helptext)


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

    st.markdown("---")
    left, centre, right = st.columns([1.15, 2.1, 1.15], gap="medium")

    # The panel columns follow the circle: components 1–5 run down the left of the
    # chart in order, 6–10 climb the right, so the right column reads 10 → 6.
    with left:
        for pc in pca.PC_NAMES[:5]:
            loading_panel(pc, loadings, scores, share, th)
    with right:
        for pc in reversed(pca.PC_NAMES[5:]):
            loading_panel(pc, loadings, scores, share, th)

    with centre:
        st.subheader(f"{player} · {season}")
        header_tiles(row)

        overlay, overlay_name = None, ""
        if len(table):
            options = ["— none —"] + [f"{r.Player} · {r.Season}"
                                      for r in neighbour_table(table).itertuples()]
            choice = st.selectbox(
                "Overlay a nearest neighbour", options, index=0,
                help="Draws that player-season's fingerprint over this one, unfilled.")
            if choice != "— none —":
                match = table.iloc[options.index(choice) - 1]
                overlay = pca.fingerprint(scores, int(match["player_id"]),
                                          str(match["season"]))
                overlay_name = choice

        st.plotly_chart(
            fig_radar(frame, th, f"{player} · {season}", limit=pca.AXIS_LIMIT,
                      overlay=overlay, overlay_name=overlay_name),
            width="stretch", key="radar")

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
                    "pinned": "Pinned"}).round({"Score (SD)": 2}),
                hide_index=True, width="stretch")

        st.subheader("Nearest neighbours in PCA space")
        st.caption(
            f"Euclidean distance over the raw scores of PC1–PC{pca.N_COMPONENTS}, which "
            "is distance in the standardized feature space the PCA was fitted on — so "
            "the high-variance style axes dominate, as they should. "
            + ("Restricted to this season."
               if same_season else "Across all seasons."))
        if len(table):
            st.dataframe(neighbour_table(table), hide_index=True, width="stretch")
        else:
            st.info("No comparable player-seasons under the current filter.")


if __name__ == "__main__":
    main()
