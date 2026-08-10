"""The dashboard's pure layer: palette rules, the PCA view's logic, registry, audit.

`theme.py`, `charts.py`, `pca.py`, `decisions.py`, `economics.py` and `audit.py` import
no Streamlit, which is what lets every rule below be exercised as a plain function
rather than through a rendered page.

A handful of tests read the real `data/features/pca_tierA_within_season_*` artifacts.
Those are the ones that keep the ten typed component titles honest — a title is an
interpretation of loadings that live on disk, so nothing but the artifact can confirm
the anchor feature still exists and still loads the way the title claims.
"""

import ast
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import audit, charts, decisions, economics, pca, theme

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"


# ── Synthetic builders ────────────────────────────────────────────────────────

def _scores(n: int = 40, seasons: tuple[str, ...] = ("2021-22", "2022-23")
            ) -> pd.DataFrame:
    """A scores frame shaped like the artifact: identity columns plus pc1…pc10."""
    rng = np.random.default_rng(0)
    rows = []
    for s, season in enumerate(seasons):
        for i in range(n):
            rows.append({"player_id": i, "player_name": f"Player {i:02d}",
                         "season": season, "team_abbreviation": "XYZ",
                         "age": 20.0 + (i % 15), "gp": 30 + (i % 50),
                         "min": 5.0 + (i % 30), "dk_pts_per_game": 5.0 + i,
                         **{f"pc{k}": float(rng.normal(0, 6 - 0.4 * k))
                            for k in range(1, 11)},
                         "_seed": s})
    return pd.DataFrame(rows).drop(columns="_seed")


def _loadings(flip: tuple[str, ...] = ()) -> pd.DataFrame:
    """Loadings carrying every anchor, with the named components sign-flipped."""
    features = [c.anchor for c in pca.COMPONENTS] + ["bas_pts", "adv_ast_ratio_x"]
    frame = pd.DataFrame({"feature": features})
    for k, c in enumerate(pca.COMPONENTS, start=1):
        column = np.linspace(0.30, 0.02, len(features))
        column[k - 1] = 0.5                       # the anchor loads hardest
        column[-1] = -0.4                         # and something loads negative
        if c.pc in flip:
            column = -column
        frame[c.pc] = column
    return frame


def _decision(**kw) -> decisions.Decision:
    base = dict(id="synthetic", topic="problem", claim="A claim.",
                because="A reason.", status="settled", source="CLAUDE.md",
                reviewed="2026-07-30", date="2026-07-30",
                reproduce="make thing → outputs/eda/thing.csv")
    return decisions.Decision(**{**base, **kw})


def _real():
    """The shipped decomposition, oriented — or a skip if `make pca` has not run."""
    paths = [FEATURES / name for name in
             (pca.SCORES_FILE, pca.LOADINGS_FILE, pca.VARIANCE_FILE)]
    if not all(p.exists() for p in paths):
        pytest.skip("PCA artifacts absent — run `make pca`")
    scores, loadings = pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
    variance = pd.read_csv(paths[2])
    return (*pca.orient(scores, loadings)[:2], variance)


# ── Palette rules ─────────────────────────────────────────────────────────────

def test_both_modes_supply_the_same_eight_slots_in_a_fixed_order():
    light, dark = theme.theme("light"), theme.theme("dark")
    assert len(light["series"]) == len(dark["series"]) == 8
    # slot order is the colour-blind-safety mechanism, so it must not be re-sorted
    assert light["series"][0] == "#2a78d6" and light["series"][1] == "#eb6834"
    assert dark["series"][0] == "#3987e5" and dark["series"][1] == "#d95926"


def test_the_all_pairs_cap_is_three():
    """Only the first three slots clear the CVD floors on every pair."""
    assert theme.ALL_PAIRS_CAP == 3


def test_sequential_ramp_is_one_hue_and_reverses_for_the_dark_surface():
    light = [c for _, c in theme.theme("light")["sequential"]]
    dark = [c for _, c in theme.theme("dark")["sequential"]]
    assert light == theme.BLUE_RAMP
    assert dark == list(reversed(theme.BLUE_RAMP))
    # near-zero recedes toward the surface in both modes
    assert light[0] == "#cde2fb" and dark[0] == "#0d366b"


def test_diverging_scale_is_two_hues_with_a_neutral_midpoint():
    for mode in ("light", "dark"):
        th = theme.theme(mode)
        stops = th["diverging"]
        assert [s[0] for s in stops] == [0.0, 0.5, 1.0]      # equal arms
        assert stops[1][1] == th["neutral"]                   # gray, not a hue
        assert stops[0][1] in th["series"] and stops[2][1] in th["series"]


def test_ordinal_ramp_stays_clear_of_the_surface():
    """An ordinal step must not sink into the background at either end."""
    assert "#cde2fb" not in theme.theme("light")["ordinal"]   # lighter than step 250
    assert "#0d366b" not in theme.theme("dark")["ordinal"]    # darker than step 600


def test_ordinal_colors_are_distinct_and_count_matched():
    for n in (2, 5, 10):
        colors = theme.ordinal_colors(theme.theme("light"), n)
        assert len(colors) == n
        assert len(set(colors)) == n


# ── Chrome ────────────────────────────────────────────────────────────────────

def test_apply_theme_pins_the_validated_surface_and_solid_hairlines():
    th = theme.theme("dark")
    fig = theme.apply_theme(go.Figure(), th)
    assert fig.layout.paper_bgcolor == th["surface"] == "#1a1a19"
    assert fig.layout.plot_bgcolor == th["surface"]
    # dashed grid reads as "threshold" when it is only a grid
    assert fig.layout.xaxis.griddash == "solid"
    assert fig.layout.yaxis.gridcolor == th["grid"]


def test_light_mode_pins_the_surface_the_palette_was_validated_against():
    assert theme.theme("light")["surface"] == "#fcfcfb"


def test_an_untitled_figure_gets_an_empty_title_rather_than_a_bare_font():
    """plotly.js renders a title object with a font and no text as "undefined".

    Only in a browser — kaleido draws nothing — so this is invisible to a PNG check
    and was found by screenshotting the running page.
    """
    th = theme.theme("light")
    assert theme.apply_theme(go.Figure(), th).layout.title.text == ""
    kept = theme.apply_theme(go.Figure(layout=dict(title="Kept")), th)
    assert kept.layout.title.text == "Kept"


def test_the_radar_is_untitled_and_a_loadings_panel_keeps_its_heading():
    th = theme.theme("light")
    assert charts.fig_radar(_fingerprint(), th, "P").layout.title.text == ""
    panel = charts.fig_loadings(pca.top_loadings(_loadings(), "pc1", 4), th, "PC1")
    assert panel.layout.title.text == "PC1"


# ── Component specification ───────────────────────────────────────────────────

def test_the_spec_declares_one_component_per_spoke_in_order():
    assert len(pca.COMPONENTS) == pca.N_COMPONENTS
    assert pca.PC_NAMES == tuple(f"pc{i}" for i in range(1, pca.N_COMPONENTS + 1))
    assert len(set(pca.PC_NAMES)) == pca.N_COMPONENTS


def test_every_component_carries_a_short_title_an_anchor_and_a_reading():
    for c in pca.COMPONENTS:
        assert c.title.strip() and len(c.title.split()) <= 5, c.pc
        assert c.anchor.strip() and c.reads.strip(), c.pc


def test_every_anchor_is_a_real_feature_in_the_shipped_loadings():
    """The typed titles' one dependency on the artifact, so it cannot rot silently."""
    _, loadings, _ = _real()
    features = set(loadings["feature"])
    for c in pca.COMPONENTS:
        assert c.anchor in features, f"{c.pc} anchors on missing {c.anchor}"


def test_the_shipped_decomposition_already_points_the_labelled_way():
    """Orientation is a guard, not a correction — today every flip is a no-op.

    If this ever fails the titles are still right and the guard did its job; it is
    here so a refit that flips an axis is *noticed* rather than silently corrected.
    """
    if not (FEATURES / pca.LOADINGS_FILE).exists():
        pytest.skip("PCA artifacts absent — run `make pca`")
    loadings = pd.read_parquet(FEATURES / pca.LOADINGS_FILE)
    assert set(pca.orientation(loadings).values()) == {1}


# ── Orientation ───────────────────────────────────────────────────────────────

def test_orientation_reads_the_anchor_sign():
    assert pca.orientation(_loadings())["pc3"] == 1
    assert pca.orientation(_loadings(flip=("pc3",)))["pc3"] == -1


def test_orient_flips_the_score_and_the_loading_together():
    """Flipping one without the other stops the bars explaining the radius."""
    scores, loadings = _scores(), _loadings(flip=("pc2",))
    before = scores["pc2"].to_numpy().copy()
    out_scores, out_loadings, signs = pca.orient(scores, loadings)
    assert signs["pc2"] == -1 and signs["pc1"] == 1
    assert np.allclose(out_scores["pc2"], -before)
    assert np.allclose(out_loadings["pc2"], -loadings["pc2"])
    assert np.allclose(out_scores["pc1"], scores["pc1"])       # untouched
    # and the anchor now loads positive, which is what the title claims
    anchor = pca.BY_PC["pc2"].anchor
    assert float(out_loadings.set_index("feature").at[anchor, "pc2"]) > 0


def test_orient_leaves_the_caller_s_frames_alone():
    scores, loadings = _scores(), _loadings(flip=("pc1",))
    original = scores["pc1"].to_numpy().copy()
    pca.orient(scores, loadings)
    assert np.allclose(scores["pc1"], original)


def test_a_missing_anchor_or_component_defaults_to_no_flip():
    thin = pd.DataFrame({"feature": ["something_else"], "pc1": [-0.9]})
    assert set(pca.orientation(thin).values()) == {1}


# ── Scaling ───────────────────────────────────────────────────────────────────

def test_sd_units_put_every_spoke_on_one_scale():
    """A raw PC1 of 6 and a raw PC10 of 6 are not the same distance from average."""
    scores = _scores()
    raw = pca.sd_scale(scores)
    assert raw["pc1"] > 2 * raw["pc10"]                # the artifact's own spread
    scaled = pca.in_sd_units(scores)
    assert np.allclose(scaled.std(ddof=1).to_numpy(), 1.0)
    assert np.allclose(scaled.mean().to_numpy(), 0.0, atol=1e-12)


def test_a_degenerate_component_does_not_divide_by_zero():
    scores = _scores()
    scores["pc7"] = 0.0
    assert pca.in_sd_units(scores)["pc7"].eq(0.0).all()


def test_clamp_pins_rather_than_rescales():
    values = pd.Series([-9.0, -1.5, 0.0, 1.5, 9.0])
    assert list(pca.clamp(values, 2.0)) == [-2.0, -1.5, 0.0, 1.5, 2.0]


# ── The fingerprint ───────────────────────────────────────────────────────────

def test_the_fingerprint_keeps_the_true_score_beside_the_pinned_radius():
    scores = _scores()
    scores.loc[0, "pc1"] = 400.0                       # far off the axis
    frame = pca.fingerprint(scores, int(scores.loc[0, "player_id"]),
                            scores.loc[0, "season"])
    assert list(frame["pc"]) == list(pca.PC_NAMES)
    row = frame[frame["pc"] == "pc1"].iloc[0]
    assert row["radius"] == pca.AXIS_LIMIT and row["sd"] > pca.AXIS_LIMIT
    assert bool(row["pinned"])
    assert frame["radius"].abs().max() <= pca.AXIS_LIMIT
    # the label a reader sees comes from the spec, not from the column name
    assert row["title"] == pca.BY_PC["pc1"].title


def test_the_fingerprint_refuses_a_player_season_it_does_not_have():
    try:
        pca.fingerprint(_scores(), 999, "1996-97")
    except KeyError:
        return
    raise AssertionError("an absent player-season must raise")


# ── Loadings ──────────────────────────────────────────────────────────────────

def test_top_loadings_are_sorted_signed_so_the_bars_diverge_around_zero():
    frame = pca.top_loadings(_loadings(), "pc1", 4)
    assert len(frame) == 4
    assert frame["loading"].is_monotonic_decreasing
    assert list(frame.columns) == ["feature", "loading", "pretty"]
    assert frame["pretty"].str.len().gt(0).all()


def test_top_loadings_balance_the_sign_rather_than_taking_the_top_by_magnitude():
    """PC1's eight largest are all positive; its negative end lands ninth.

    A plain top-`n` would show "rebounding big" and drop the "not a shooter" half of
    an axis defined by the opposition, so each side is guaranteed half the slots.
    """
    _, loadings, _ = _real()
    plain = loadings.reindex(
        loadings["pc1"].abs().sort_values(ascending=False).index).head(8)
    assert (plain["pc1"] > 0).all()                    # the rule that would fail

    frame = pca.top_loadings(loadings, "pc1", 8)
    assert (frame["loading"] > 0).sum() == 4 and (frame["loading"] < 0).sum() == 4
    assert any("fg3" in f or "3pt" in f for f in
               frame.loc[frame["loading"] < 0, "feature"])


def test_a_one_sided_component_backfills_rather_than_shrinking_the_panel():
    loadings = pd.DataFrame({"feature": [f"bas_f{i}" for i in range(6)],
                             "pc1": [0.5, 0.4, 0.3, 0.2, 0.1, -0.05]})
    frame = pca.top_loadings(loadings, "pc1", 5)
    assert len(frame) == 5
    assert (frame["loading"] < 0).sum() == 1           # only one negative exists


def test_feature_names_are_prettified_family_first():
    assert pca.pretty_feature("usg_pct_oreb") == "usage · % OREB"
    assert pca.pretty_feature("adv_ts_pct") == "advanced · TS%"
    assert pca.pretty_feature("sco_pct_pts_2pt_mr") == "scoring · % PTS 2PT mid-range"
    # a compound the one-token-at-a-time pass would render as "AST to"
    assert pca.pretty_feature("adv_ast_to") == "advanced · AST/TOV"
    assert pca.pretty_feature("adv_ast_ratio") == "advanced · AST ratio"
    # an unknown family passes through rather than losing its first token
    assert pca.pretty_feature("mystery_column") == "mystery column"


def test_a_click_resolves_to_a_component_by_its_spoke_label():
    assert pca.component_from_click([{"theta": "PC4", "point_index": 99}]) == "pc4"


def test_a_click_falls_back_to_the_point_index_when_theta_is_absent():
    for key in ("point_index", "pointIndex", "point_number", "pointNumber"):
        assert pca.component_from_click([{key: 2}]) == "pc3"


def test_the_repeated_closing_vertex_wraps_to_the_first_component():
    """The polygon repeats its first point to close itself, so index 10 is PC1."""
    assert pca.component_from_click([{"point_index": pca.N_COMPONENTS}]) == "pc1"


def test_an_empty_or_unrecognisable_selection_changes_nothing():
    assert pca.component_from_click([]) is None
    assert pca.component_from_click(None) is None
    assert pca.component_from_click([{"theta": "not a spoke"}]) is None


def test_variance_share_keys_on_the_pc_name_not_the_index():
    variance = pd.DataFrame({"component": [1, 2, 3],
                             "explained_variance_ratio": [0.4, 0.3, 0.2],
                             "cumulative": [0.4, 0.7, 0.9]})
    assert pca.variance_share(variance) == {"pc1": 0.4, "pc2": 0.3, "pc3": 0.2}


# ── Exemplars ─────────────────────────────────────────────────────────────────

def test_exemplars_ignore_seasons_too_thin_to_name_an_axis():
    """Without the filter every extreme is a 200-minute player's noise."""
    scores = _scores()
    scores["min"], scores["gp"] = 30.0, 70
    scores.loc[0, ["min", "gp"]] = [4.0, 6]            # a cup-of-coffee season
    scores.loc[0, "pc1"] = 1e6                         # and an absurd score
    high, _ = pca.exemplars(scores, "pc1")
    assert scores.loc[0, "player_name"] not in high


def test_exemplars_fall_back_rather_than_going_blank():
    scores = _scores()
    scores["min"], scores["gp"] = 1.0, 1               # nobody clears the bar
    high, low = pca.exemplars(scores, "pc1")
    assert high != "—" and low != "—" and high != low


def test_every_component_names_two_real_player_seasons():
    scores, _, _ = _real()
    for c in pca.COMPONENTS:
        high, low = pca.exemplars(scores, c.pc)
        assert " · " in high and " · " in low, c.pc
        assert high != low, c.pc


# ── Nearest neighbours ────────────────────────────────────────────────────────

def test_neighbours_exclude_the_player_s_own_other_seasons():
    """They are usually the three nearest, which is true and tells you nothing."""
    scores = _scores()
    target = scores.iloc[0]
    # the same player's other season sits on top of this one and must still be dropped
    same = (scores["player_id"] == target["player_id"]) & (scores["season"] != target["season"])
    scores.loc[same, list(pca.PC_NAMES)] = target[list(pca.PC_NAMES)].to_numpy()

    out = pca.neighbors(scores, int(target["player_id"]), target["season"])
    assert len(out) == 3
    assert (out["player_id"] != target["player_id"]).all()


def test_same_season_only_restricts_the_pool():
    scores = _scores()
    target = scores.iloc[0]
    out = pca.neighbors(scores, int(target["player_id"]), target["season"],
                        same_season_only=True)
    assert set(out["season"]) == {target["season"]}


def test_neighbours_are_sorted_and_carry_a_distance():
    scores = _scores()
    target = scores.iloc[0]
    out = pca.neighbors(scores, int(target["player_id"]), target["season"], k=5)
    assert len(out) == 5
    assert out["distance"].is_monotonic_increasing
    assert (out["distance"] > 0).all()


def test_distance_is_raw_not_standardized_so_the_big_axes_dominate():
    """Scaling each component to one SD first would be Mahalanobis distance.

    That gives PC10 — team pace, 1.9% of variance — the same say as PC1, which is not
    what "nearest in PCA space" means. Here candidate B is closer in raw space and
    candidate A is closer once every axis is stretched to unit variance; raw must win.
    """
    base = {f"pc{k}": 0.0 for k in range(1, 11)}
    ident = dict(player_name="x", season="2022-23", team_abbreviation="XYZ",
                 age=25.0, gp=70, min=30.0, dk_pts_per_game=20.0)
    rows = [{"player_id": 0, **ident, **base},
            {"player_id": 1, **ident, **base, "pc1": 1.0},        # A
            {"player_id": 2, **ident, **base, "pc10": 0.5}]       # B
    # spread pc1 wide and pc10 narrow, so the two metrics disagree
    for i, extra in enumerate([(20.0, 0.0), (-20.0, 0.0)], start=3):
        rows.append({"player_id": i, **ident, **base,
                     "pc1": extra[0], "pc10": extra[1]})
    scores = pd.DataFrame(rows)

    out = pca.neighbors(scores, 0, "2022-23", k=2)
    assert list(out["player_id"]) == [2, 1]

    scaled = pca.in_sd_units(scores)
    assert abs(scaled.loc[2, "pc10"]) > abs(scaled.loc[1, "pc1"])   # the reversal


def test_an_empty_pool_returns_an_empty_frame_rather_than_raising():
    scores = _scores(seasons=("2022-23",))
    scores["player_id"] = 7                     # every row is the same player
    out = pca.neighbors(scores, 7, "2022-23")
    assert out.empty


def test_neighbours_refuse_a_player_season_they_do_not_have():
    try:
        pca.neighbors(_scores(), 999, "1996-97")
    except KeyError:
        return
    raise AssertionError("an absent player-season must raise")


# ── Figures ───────────────────────────────────────────────────────────────────

def _fingerprint(pinned: bool = False) -> pd.DataFrame:
    scores = _scores()
    if pinned:
        scores.loc[0, "pc4"] = 1e4
    return pca.fingerprint(scores, int(scores.loc[0, "player_id"]),
                           scores.loc[0, "season"])


def test_the_score_to_radius_map_puts_the_centre_at_minus_two_and_the_rim_at_plus_two():
    assert charts.unit_radius(-2.0, 2.0) == 0.0
    assert charts.unit_radius(0.0, 2.0) == 0.5
    assert charts.unit_radius(2.0, 2.0) == charts.RIM == 1.0


def test_the_first_spoke_sits_at_the_top_and_the_rest_run_counterclockwise():
    """The side panel order follows the circle, so the direction is load bearing."""
    assert charts.spoke_angle(0, 10) == charts.ANGULAR_ROTATION == 90
    assert charts.spoke_angle(1, 10) == 126          # counterclockwise, to the left
    assert charts.ANGULAR_DIRECTION == "counterclockwise"


def test_the_radar_axis_is_fixed_so_two_fingerprints_differ_in_shape():
    th = theme.theme("light")
    for fig in (charts.fig_radar(_fingerprint(), th, "P"),
                charts.fig_radar(_fingerprint(pinned=True), th, "P")):
        assert list(fig.layout.xaxis.range) == [-charts.AXIS_EXTENT,
                                                charts.AXIS_EXTENT]
        # every plotted vertex is inside the rim however extreme the player is
        player = fig.data[-1]
        assert max(x * x + y * y for x, y in zip(player.x, player.y)) <= 1.0 + 1e-9
    # and circles stay circular when the container is not square
    assert fig.layout.yaxis.scaleanchor == "x" and fig.layout.yaxis.scaleratio == 1


def test_the_grid_is_shapes_so_nothing_but_a_data_point_can_be_clicked():
    """Streamlit reports a click as a trace point index; a grid trace would alias."""
    fig = charts.fig_radar(_fingerprint(), theme.theme("light"), "P")
    assert len(fig.data) == 2                            # the hit layer and the player
    rings = [s for s in fig.layout.shapes if s.type == "circle"]
    spokes = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(rings) == 4        # −2 is the centre, so four rings are drawable
    assert len(spokes) == pca.N_COMPONENTS
    assert len(fig.layout.annotations) == pca.N_COMPONENTS + 5   # spokes + ring labels


def test_the_radar_closes_its_polygon():
    player = charts.fig_radar(_fingerprint(), theme.theme("light"), "P").data[-1]
    assert len(player.x) == pca.N_COMPONENTS + 1          # first point repeated
    assert (player.x[0], player.y[0]) == (player.x[-1], player.y[-1])
    assert player.fill == "toself"


def test_a_generous_invisible_hit_layer_sits_under_the_visible_markers():
    """A click near a vertex has to land, and it must not steal the hover."""
    fig = charts.fig_radar(_fingerprint(), theme.theme("light"), "P")
    hit, player = fig.data[0], fig.data[-1]
    assert hit.marker.size == charts.HIT_MARKER > charts.SELECTED_MARKER
    assert hit.marker.color == "rgba(0,0,0,0)"
    assert hit.hoverinfo == "skip" and hit.showlegend is False
    # same order as the visible points, so either one resolves to the same component
    assert list(hit.x) == list(player.x)[:-1]


def test_a_pinned_spoke_is_drawn_open_and_hovers_its_true_score():
    fig = charts.fig_radar(_fingerprint(pinned=True), theme.theme("light"), "P")
    player = fig.data[-1]
    assert "circle-open" in player.marker.symbol
    assert "circle" in player.marker.symbol              # only the pinned one is open
    # the hover reads customdata, not the plotted radius, so a pinned spoke cannot
    # claim to be a 2.0
    assert "customdata[2]" in player.hovertemplate
    assert any(abs(float(c[2])) > pca.AXIS_LIMIT for c in player.customdata)


def test_the_hover_score_is_preformatted_to_one_decimal():
    """A d3 spec over a mixed-dtype customdata array is how 14 digits reach the screen."""
    player = charts.fig_radar(_fingerprint(), theme.theme("light"), "P").data[-1]
    assert ":.2f" not in player.hovertemplate
    for _, _, score in player.customdata:
        assert isinstance(score, str)
        whole, _, decimals = score.partition(".")
        assert len(decimals) == 1 and whole[0] in "+-"


def test_the_selected_spoke_is_enlarged_and_ringed():
    """The chart has to say which component the panel beside it is explaining."""
    fig = charts.fig_radar(_fingerprint(), theme.theme("light"), "P", selected="PC4")
    marker = fig.data[-1].marker
    index = pca.PC_NAMES.index("pc4")
    assert marker.size[index] == charts.SELECTED_MARKER
    assert all(s == charts.MARKER for i, s in enumerate(marker.size) if i != index)
    assert marker.line.width[index] == 2


def test_nothing_is_enlarged_when_no_spoke_is_selected():
    fig = charts.fig_radar(_fingerprint(), theme.theme("light"), "P")
    assert set(fig.data[-1].marker.size) == {charts.MARKER}


def test_an_overlay_takes_the_second_slot_and_turns_the_legend_on():
    th = theme.theme("light")
    plain = charts.fig_radar(_fingerprint(), th, "P")
    with_overlay = charts.fig_radar(_fingerprint(), th, "P",
                                    overlay=_fingerprint(), overlay_name="Q")
    assert plain.layout.showlegend is False
    assert with_overlay.layout.showlegend is True
    overlay = next(t for t in with_overlay.data if t.name == "Q")
    assert overlay.line.color == th["series"][1] and overlay.fill is None


def test_loading_bars_take_the_diverging_ends_not_two_categorical_slots():
    """A loading's sign is a direction on one axis, not a second category."""
    th = theme.theme("light")
    frame = pca.top_loadings(_loadings(), "pc1", 6)
    fig = charts.fig_loadings(frame, th, "PC1")
    colors = set(fig.data[0].marker.color)
    assert colors <= {th["diverging"][0][1], th["diverging"][-1][1]}
    assert len(colors) == 2                              # both signs are present
    assert fig.layout.showlegend is False


def test_a_loading_panel_grows_with_its_bar_count():
    th = theme.theme("light")
    short = charts.fig_loadings(pca.top_loadings(_loadings(), "pc1", 4), th, "t")
    tall = charts.fig_loadings(pca.top_loadings(_loadings(), "pc1", 10), th, "t")
    assert tall.layout.height > short.layout.height


def test_translucent_converts_a_hex_fill_without_touching_the_stroke():
    assert charts._translucent("#2a78d6", 0.22) == "rgba(42,120,214,0.22)"


# ── Wiring ────────────────────────────────────────────────────────────────────

def test_the_repo_root_is_on_the_path_for_package_imports():
    from dashboard import artifacts
    assert str(ROOT) == str(artifacts.ROOT)


# `draft_room.py` is the one page that reaches into `src/`, and the exemption is narrow
# on purpose — see the module's own docstring and `dashboard/README.md`. It is a live
# decision tool rather than a view, and what it needs is `bracket.best_lineup` and
# `draft.legal_mask`; the alternative to importing them is a second copy of the matroid
# that seats a weekly lineup and a second opinion about which players are legal, which is
# the drift this rule exists to prevent arriving through the other door.
SRC_IMPORTERS = {"draft_room.py": "src.sim"}


def _src_imports(path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        found += [n for n in names if n == "src" or n.startswith("src.")]
    return found


def test_the_dashboard_imports_nothing_from_src():
    """The invariant: it reads artifacts and nothing else.

    The PCA view reads three files `make pca` wrote; it never imports `src.eda.pca` to
    re-project anything, which is what stops the dashboard drifting from the fit.
    """
    offenders = []
    for path in sorted((ROOT / "dashboard").rglob("*.py")):
        if path.name in SRC_IMPORTERS:
            continue
        offenders += [f"{path.relative_to(ROOT)}: {name}" for name in _src_imports(path)]
    assert offenders == []


def test_the_one_exempt_page_reaches_no_further_than_the_simulation_layer():
    """The exemption is a boundary, not a hole.

    `src.sim` is numpy over the artifacts the pipeline wrote and imports no CmdStan, so a
    page built on it still cannot refit anything. An import of `src.models` or `src.data`
    would be a page that could, which is the thing the rule forbids.
    """
    for name, allowed in SRC_IMPORTERS.items():
        path = ROOT / "dashboard" / name
        assert path.exists(), name
        imports = _src_imports(path)
        assert imports, f"{name} is listed as exempt but imports nothing from src/"
        for module in imports:
            assert module == allowed or module.startswith(f"{allowed}."), \
                f"{name} imports {module}, outside the exempt {allowed}"


# The Streamlit surface, exhaustively. Everything else in the package stays testable
# without a runtime, and this is written as a denylist over the whole tree rather than an
# allowlist of pure modules so that a new file is pure *by default* — the expansion in
# `docs/dashboard-plan.md` adds eight more pages, and the failure mode worth guarding is a
# view's logic being written into the view instead of into a pure sibling.
STREAMLIT_SURFACE = {"app.py", "shell.py", "artifacts.py", "draft_room.py"}
VIEWS_DIR = "views"


def test_pure_modules_do_not_import_streamlit():
    """Only the entrypoint, the shell, the I/O layer and the pages touch Streamlit."""
    for path in sorted((ROOT / "dashboard").rglob("*.py")):
        if path.name in STREAMLIT_SURFACE or path.parent.name == VIEWS_DIR:
            continue
        assert "streamlit" not in path.read_text(), \
            f"{path.relative_to(ROOT)} must stay Streamlit-free"


def test_every_named_streamlit_surface_still_exists():
    """A denylist that names a deleted file silently stops guarding a real one."""
    for name in STREAMLIT_SURFACE:
        assert (ROOT / "dashboard" / name).exists(), name


def test_the_pure_layer_is_still_the_bigger_half():
    """The view holds layout; the logic it draws lives in a module a test can call.

    Not a style rule — `dashboard/pca.py` is 400-odd lines of orientation, scaling,
    neighbours and click resolution, all of it exercised directly above, and none of it
    reachable if it had been written inside `render()`.
    """
    pure = {p.name for p in (ROOT / "dashboard").glob("*.py")
            if p.name not in STREAMLIT_SURFACE}
    assert {"theme.py", "charts.py", "pca.py", "decisions.py", "economics.py",
            "audit.py"} <= pure


# ── The multipage shell ───────────────────────────────────────────────────────
#
# `app.VIEWS` is a plain tuple built without touching Streamlit — `st.Page` is only
# constructed inside `app.pages()` — so the navigation registry is testable as data.
# What a rendered page does is covered by the three verification layers in
# `dashboard/README.md`, not here.

def test_the_navigation_carries_at_least_two_entries():
    """Measured, not stylistic: Streamlit draws no navigation for a one-page app.

    With a single `st.Page`, `st.navigation(position="sidebar")` still sends
    `Position.SIDEBAR` and the frontend renders nothing — `[data-testid="stSidebarNav"]`
    is absent from the DOM. A shell that dropped back to one page would therefore be
    indistinguishable from the single-page script it replaced, and the cross-page state in
    `dashboard/shell.py` would have nothing to survive. So the shell ships a placeholder
    for the next page in the build order rather than shipping alone.
    """
    from dashboard import app
    assert len(app.VIEWS) >= 2


def test_every_page_has_a_unique_url_path_and_a_callable():
    """Duplicate paths are a `StreamlitAPIException` at nav-build time, i.e. in a browser."""
    from dashboard import app
    paths = [v.url_path for v in app.VIEWS]
    assert len(paths) == len(set(paths)), paths
    for view in app.VIEWS:
        assert callable(view.render), view.title
        assert view.title.strip() and view.icon.strip(), view
        assert "/" not in view.url_path and view.url_path == view.url_path.strip("/")


def test_the_first_page_is_the_real_one():
    """`pages()` makes index 0 the default, so `/` must not serve a placeholder."""
    from dashboard import app
    from dashboard.views import fingerprints
    assert app.VIEWS[0].render is fingerprints.render


def test_the_shell_offers_exactly_the_modes_the_palette_defines():
    """A mode the palette has no entry for is a `KeyError` inside `theme()`."""
    from dashboard import shell
    assert set(shell.MODES) == set(theme.THEMES)


# ── The registry ──────────────────────────────────────────────────────────────

def test_every_entry_uses_the_closed_status_vocabulary():
    for d in decisions.REGISTRY:
        assert d.status in decisions.STATUSES, (d.id, d.status)
        assert d.topic in decisions.TOPICS, (d.id, d.topic)


def test_every_entry_has_a_claim_a_reason_and_a_source_that_exists():
    for d in decisions.REGISTRY:
        assert d.claim.strip(), d.id
        assert d.because.strip(), d.id
        assert (ROOT / d.source).exists(), f"{d.id} cites missing {d.source}"


def test_entry_ids_are_unique():
    ids = [d.id for d in decisions.REGISTRY]
    assert len(ids) == len(set(ids))


def test_reviewed_and_date_are_iso_dates():
    for d in decisions.REGISTRY:
        date.fromisoformat(d.reviewed)
        date.fromisoformat(d.date)


def test_every_reproduce_artifact_exists_unless_the_status_exempts_it():
    """The anti-drift guard — the one audit check that is a hard test.

    An entry citing a target that was renamed or never built is exactly the drift
    `docs/dashboard-plan.md` names as its main risk, so this one fails the suite
    while the other three checks stay report-only.
    """
    assert audit.missing_artifacts() == []


def test_a_withdrawn_entry_keeps_its_replacement_and_what_caught_it():
    """Reversals are content — an entry that loses those fields loses the point."""
    for d in decisions.by_status("withdrawn"):
        assert d.replaced_by.strip(), d.id
        assert d.caught_by.strip(), d.id


def test_a_deadline_entry_names_its_date():
    for d in decisions.by_status("deadline"):
        assert d.due.strip(), d.id


def test_a_blocked_entry_names_the_unblocking_condition():
    for d in decisions.by_status("blocked"):
        assert d.unblocks.strip(), d.id


def test_incident_entries_carry_no_live_figure():
    """The deliberate exception to the provenance rule, and its boundary."""
    for d in decisions.by_status("incident"):
        assert not decisions.artifact_paths(d), (
            f"{d.id} is an incident but names an artifact — if it has a figure it is "
            f"not an incident")


def test_reproduce_parses_into_a_target_and_its_artifacts():
    d = _decision(reproduce="make stan-availability → outputs/predictions/a.csv, "
                            "outputs/predictions/b.csv")
    assert decisions.make_target(d) == "make stan-availability"
    assert decisions.artifact_paths(d) == ("outputs/predictions/a.csv",
                                           "outputs/predictions/b.csv")


def test_an_entry_with_no_reproduce_yields_no_artifacts():
    d = _decision(status="open", reproduce="")
    assert decisions.artifact_paths(d) == ()
    assert decisions.needs_artifact(d) is False


def test_status_mix_counts_every_status_in_the_vocabulary():
    mix = decisions.status_mix()
    assert set(mix) == set(decisions.STATUSES)
    assert sum(mix.values()) == len(decisions.REGISTRY)


def test_every_topic_has_at_least_one_entry():
    for topic in decisions.TOPICS:
        assert decisions.by_topic(topic), topic


# ── The audit ─────────────────────────────────────────────────────────────────

def test_the_staleness_check_flags_an_entry_older_than_its_source_doc():
    stale = _decision(id="stale", reviewed="2020-01-01")
    fresh = _decision(id="fresh", reviewed="2026-12-31")
    moved = lambda path, root: date(2026, 6, 1)          # noqa: E731 — injected stub

    flagged = audit.stale_entries((stale, fresh), ROOT, commit_date=moved)
    assert [f.subject for f in flagged] == ["stale"]
    assert "2026-06-01" in flagged[0].detail


def test_the_staleness_check_is_silent_when_git_knows_nothing_about_the_doc():
    d = _decision(reviewed="2020-01-01")
    assert audit.stale_entries((d,), ROOT, commit_date=lambda p, r: None) == []


def test_the_staleness_check_uses_commit_dates_not_mtime(tmp_path):
    """A fresh checkout resets mtime, so mtime would flag the whole registry."""
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("x")
    assert audit.last_commit_date("CLAUDE.md", tmp_path) is None


def test_the_orphan_check_flags_an_artifact_no_entry_and_no_view_references(tmp_path):
    art = tmp_path / "outputs" / "eda"
    art.mkdir(parents=True)
    (art / "referenced_by_registry.csv").write_text("a\n1\n")
    (art / "referenced_by_a_view.csv").write_text("a\n1\n")
    (art / "nobody_reads_this.csv").write_text("a\n1\n")

    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "some_view.py").write_text('path = features_dir() / "referenced_by_a_view.csv"\n')

    registry = (_decision(reproduce="make thing → outputs/eda/referenced_by_registry.csv"),)
    flagged = audit.orphaned_artifacts(root=tmp_path, registry=registry, package=pkg,
                                       dirs=["outputs/eda"])
    assert [f.subject for f in flagged] == ["outputs/eda/nobody_reads_this.csv"]


def test_the_orphan_check_credits_a_glob_for_a_whole_artifact_family(tmp_path):
    """A target that writes twenty files gets one registry line, not twenty."""
    art = tmp_path / "data" / "features"
    art.mkdir(parents=True)
    for name in ("pca_tierA_pooled.pkl", "pca_tierB_within_season_scores.parquet"):
        (art / name).write_text("x")
    (art / "unrelated.parquet").write_text("x")

    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "empty.py").write_text("x = 1\n")

    registry = (_decision(reproduce="make pca → data/features/pca_*"),)
    flagged = audit.orphaned_artifacts(root=tmp_path, registry=registry, package=pkg,
                                       dirs=["data/features"])
    assert [f.subject for f in flagged] == ["data/features/unrelated.parquet"]


def test_a_tier_parameterized_read_credits_the_whole_family(tmp_path):
    """`f"team_context_tier{tier}.parquet"` never contains the tier literally."""
    art = tmp_path / "data" / "features"
    art.mkdir(parents=True)
    (art / "team_context_tierA.parquet").write_text("x")
    (art / "team_context_tierB.parquet").write_text("x")

    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "view.py").write_text(
        'p = features_dir() / f"team_context_tier{tier}.parquet"\n')

    assert audit.orphaned_artifacts(root=tmp_path, registry=(), package=pkg,
                                    dirs=["data/features"]) == []


def test_the_missing_artifact_check_reports_a_renamed_target(tmp_path):
    registry = (_decision(id="renamed",
                          reproduce="make thing → outputs/eda/never_written.csv"),)
    flagged = audit.missing_artifacts(registry, tmp_path)
    assert [f.subject for f in flagged] == ["renamed"]
    assert "make thing" in flagged[0].detail


def test_the_missing_artifact_check_skips_the_unbacked_statuses(tmp_path):
    for status in decisions.UNBACKED_STATUSES:
        d = _decision(status=status, reproduce="")
        assert audit.missing_artifacts((d,), tmp_path) == []


def test_an_entry_that_owes_a_figure_and_names_none_is_flagged(tmp_path):
    d = _decision(status="measured", reproduce="")
    flagged = audit.missing_artifacts((d,), tmp_path)
    assert len(flagged) == 1
    assert "obliges" in flagged[0].detail


def test_the_pending_marker_count_is_the_provenance_gap(tmp_path):
    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "gated.py").write_text(
        "from dashboard.layout import pending_marker\n"
        "def render(ctx):\n"
        "    pending_marker('make variance-budget', 'the variance budget')\n")
    (pkg / "clean.py").write_text("def render(ctx):\n    pass\n")

    flagged = audit.pending_markers(package=pkg)
    assert len(flagged) == 1
    assert flagged[0].subject.endswith("gated.py:3")


def test_the_pending_marker_count_ignores_the_definition_itself(tmp_path):
    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "layout.py").write_text("def pending_marker(target, figure):\n    pass\n")
    assert audit.pending_markers(package=pkg) == []


def test_source_literals_read_fstring_parts_but_not_comments(tmp_path):
    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "m.py").write_text(
        '# mentions ghost_artifact.csv in a comment only\n'
        'a = "real_artifact.csv"\n'
        'b = f"prefixed_tier{tier}.parquet"\n'
        'c = "short.csv"\n')
    literals = audit.source_literals(pkg)
    assert "real_artifact.csv" in literals
    assert "prefixed_tier" in literals
    assert not any("ghost_artifact" in lit for lit in literals)
    assert "short.csv" not in literals                    # below MIN_LITERAL


def test_the_audit_report_renders_every_check_and_the_two_headline_counts():
    results = audit.run()
    text = audit.report(results)
    for heading in audit.HEADINGS.values():
        assert heading in text
    assert "typed constants pending:" in text
    assert "orphaned artifacts:" in text


# ── Tournament economics ──────────────────────────────────────────────────────

def test_rake_and_hurdle_match_hand_computed_values_on_a_synthetic_table():
    # 100 entries at $10 is a $1,000 pool; paying $800 keeps 20%.
    r = economics.rake(entries=100, fee=10, prizes=800)
    assert abs(r - 0.20) < 1e-12
    # Keeping 20% of the pool means beating the field by 25% to return the fee —
    # 1/0.8 = 1.25 — not by 20%. That gap is the whole reason for the hurdle unit.
    assert abs(economics.break_even_hurdle(r) - 0.25) < 1e-12


def test_a_zero_rake_contest_has_a_zero_hurdle():
    assert economics.break_even_hurdle(economics.rake(10, 5, 50)) == 0.0


def test_rake_rejects_an_empty_pool():
    try:
        economics.rake(entries=0, fee=10, prizes=100)
    except ValueError:
        return
    raise AssertionError("an empty buy-in pool must raise")


def test_the_metadata_loader_drops_the_spreadsheets_trailing_columns():
    meta = economics.load_metadata()
    assert not any(str(c).startswith("Unnamed:") for c in meta.columns)
    assert {"type", "total_entries", "entry_fee_per_team", "total_prizes"} <= set(
        meta.columns)


def test_the_five_real_tournaments_derive_their_own_economics():
    econ = economics.economics()
    assert len(econ) == 5
    # every rake is a plausible house cut, and the hurdle always exceeds it
    assert (econ["rake"] > 0).all() and (econ["rake"] < 0.30).all()
    assert (econ["break_even_hurdle"] > econ["rake"]).all()
    # the pool is the identity it is defined by
    assert (econ["buy_in_pool"]
            == econ["total_entries"] * econ["entry_fee_per_team"]).all()


def test_the_derived_advance_counts_reproduce_the_five_known_tournaments():
    adv = economics.advance_table()
    advancing = adv[adv["n_advance"] > 0]
    assert set(advancing["tournament"]) == set(economics.load_metadata()["type"])
    # the plan's stated range: 1-of-12 through 2-of-6
    pairs = set(zip(advancing["n_advance"], advancing["pod_size"]))
    assert (1, 12) in pairs and (2, 6) in pairs and (2, 12) in pairs
    assert advancing["advance_rate"].between(0, 1).all()


def test_round_one_is_a_zero_consolation_knockout_in_every_tournament():
    adv = economics.advance_table()
    r1 = adv[adv["round"] == 1]
    assert len(r1) == 5
    assert (r1["cash_places"] == 0).all()      # ranks 3-12 get nothing
    assert (r1["n_advance"] == 2).all()
    assert bool(r1["zero_consolation"].all())


def test_the_round_one_pod_size_is_confirmed_by_the_entry_counts():
    """The prize CSV does not carry it, so it needs a second, independent route."""
    assert economics.check_round_one_pod(round_one_pod=12) is True
    for wrong in (2, 3, 4, 6, 8, 10, 11, 13, 14, 20, 24):
        assert economics.check_round_one_pod(round_one_pod=wrong) is False


def test_integrality_alone_does_not_pin_the_round_one_pod():
    """Worth pinning, because it is the trap in this derivation.

    Any divisor of 12 gives a whole-number chain — halving the pod just doubles the
    next round's field. What actually identifies 12 is that all five tournaments
    then have a final round paying *exactly* its own field.
    """
    six = economics.round_one_pod_evidence(round_one_pod=6)
    twelve = economics.round_one_pod_evidence(round_one_pod=12)
    assert six["integral"] is True                    # necessary, not sufficient
    assert six["final_field_equals_paid"] == 0
    assert twelve["final_field_equals_paid"] == 5     # 4 before the 2026-08-09 fix
    assert twelve["final_field_equals_paid"] == twelve["n_tournaments"]


def test_the_payout_curve_expands_the_banded_prize_rows():
    curve = economics.payout_curve(tournament="88k_alley_oop")
    assert list(curve["place"]) == [1, 2, 3, 4]
    assert curve["cash"].is_monotonic_decreasing
    # and an unknown tournament yields an empty frame rather than raising
    assert economics.payout_curve(tournament="no_such_contest").empty


def test_payout_convexity_differs_by_an_order_of_magnitude_across_the_five():
    econ = economics.economics()
    multiples = econ["first_prize_multiple"]
    assert multiples.max() / multiples.min() > 10


# ── Artifact loading ──────────────────────────────────────────────────────────

def test_optional_returns_none_for_a_missing_path_without_raising(tmp_path):
    from dashboard.artifacts import optional
    assert optional(tmp_path / "nope.csv") is None


def test_optional_reads_a_real_table(tmp_path):
    from dashboard.artifacts import optional
    path = tmp_path / "small.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    frame = optional(path)
    assert frame is not None and list(frame["a"]) == [1, 2]
