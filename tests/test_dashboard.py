"""The dashboard's pure layer: palette rules, both views' logic, registry, audit.

`theme.py`, `charts.py`, `pca.py`, `strategy.py`, `decisions.py`, `economics.py` and
`audit.py` import no Streamlit, which is what lets every rule below be exercised as a
plain function rather than through a rendered page.

A handful of tests read the real artifacts on disk — the
`data/features/pca_tierA_within_season_*` files and the strategy / bracket families under
`outputs/predictions/`. Those are the ones that keep a *derived* quantity honest against
the artifact it is derived from: a component title is an interpretation of loadings that
live on disk, and the tournament page's "does this gap resolve" styling is derived from an
interval that has its own `resolved` column beside it. Nothing but the artifact can
confirm those still agree.
"""

import ast
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import (audit, charts, decisions, economics, model_cards, pca, strategy,
                       theme)

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"
PREDICTIONS = ROOT / "outputs" / "predictions"


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


# ── The tournament page's pure layer ──────────────────────────────────────────
#
# `dashboard/strategy.py` shapes what `make bracket` and `make strategy-sweep` wrote into
# the frames the page draws. The synthetic builders below carry the columns the real
# artifacts carry and nothing else; a handful of tests read the real files, and those are
# the ones that keep a derived quantity honest against the artifact it is derived from.

def _sweep(tournaments=("600k_shootaround", "20k_spin_move"),
           seasons=("2022-23", "2023-24")) -> pd.DataFrame:
    """A sweep frame: three axes, five arms, every column the page reads."""
    arms = [("ranking", "model_mean", 0.10), ("ranking", "adp", -0.02),
            ("alpha", "blend_a15", 0.17), ("alpha", "blend_a85", 0.05),
            ("objective", "lineup_value", 0.21)]
    hurdles = {"600k_shootaround": 0.176, "20k_spin_move": 0.1232}
    rows = []
    for t in tournaments:
        for s, season in enumerate(seasons):
            for axis, arm, lift in arms:
                # A second season perturbs the ordering, so a test can see that arms are
                # ranked on the mean across seasons rather than on whichever came first.
                value = lift + (0.01 if s else 0.0)
                rows.append({
                    "season": season, "tournament": t, "strategy": arm, "axis": axis,
                    "n_entries": 10, "p_advance": 1 / 6 + value,
                    "p_advance_lo": 1 / 6 + value - 0.05,
                    "p_advance_hi": 1 / 6 + value + 0.05,
                    "lift_vs_null": value, "lift_lo": value - 0.05,
                    "lift_hi": value + 0.05, "p_advance_null": 1 / 6,
                    "roi": 2.0 + 10 * value, "roi_null": -0.1497,
                    "break_even_hurdle": hurdles[t]})
    return pd.DataFrame(rows)


def _paired(tournament: str = "600k_shootaround") -> pd.DataFrame:
    """Gaps against one baseline: two clear of zero, two straddling it, plus the self."""
    rows = [("lineup_value", 0.10, 0.09, 0.11), ("blend_a15", 0.04, 0.01, 0.07),
            ("blend_a30", 0.01, -0.01, 0.03), ("blend_stack4", -0.001, -0.02, 0.02),
            ("adp", -0.06, -0.08, -0.04), ("model_mean", 0.0, 0.0, 0.0)]
    return pd.DataFrame([
        {"tournament": tournament, "metric": "p_advance", "baseline": "model_mean",
         "strategy": arm, "gap": gap, "gap_lo": lo, "gap_hi": hi,
         "p_gap_below_zero": 0.5, "resolved": not (lo <= 0 <= hi), "n_worlds": 1000}
        for arm, gap, lo, hi in rows])


def _advance() -> pd.DataFrame:
    """Two tournaments' round ladders, shaped like `economics.advance_table()`."""
    rows = []
    for t, rates in (("600k_shootaround", [(12, 2), (12, 1), (10, 1), (49, 0)]),
                     ("20k_spin_move", [(12, 2), (6, 2), (6, 2), (8, 0)])):
        field = 35280.0 if t == "600k_shootaround" else 432.0
        for rnd, (pod, adv) in enumerate(rates, start=1):
            rows.append({"tournament": t, "round": rnd, "field_entries": field,
                         "pod_size": pod, "n_advance": adv,
                         "advance_rate": adv / pod, "cash_places": 0 if rnd == 1 else 9,
                         "paid_share": 0.0, "min_cash": 0.0 if rnd == 1 else 30.0,
                         "zero_consolation": rnd == 1})
            if adv:
                field = field / pod * adv
    return pd.DataFrame(rows)


def _econ() -> pd.DataFrame:
    return pd.DataFrame([
        {"tournament": "600k_shootaround", "total_entries": 35280,
         "entry_fee_per_team": 20, "rake": 0.14966, "break_even_hurdle": 0.176,
         "first_prize": 200000.0, "first_prize_multiple": 10000.0},
        {"tournament": "20k_spin_move", "total_entries": 432,
         "entry_fee_per_team": 52, "rake": 0.109687, "break_even_hurdle": 0.1232,
         "first_prize": 5000.0, "first_prize_multiple": 96.15}])


def _artifact(name: str) -> pd.DataFrame:
    path = PREDICTIONS / name
    if not path.exists():
        pytest.skip(f"{name} absent — run `make strategy-sweep` / `make bracket`")
    return pd.read_csv(path)


def test_a_tournament_name_keeps_its_buy_in_tier_lowercase():
    """`str.title()` renders it `600K Shootaround`, which reads as a unit, not a name."""
    assert strategy.pretty_tournament("600k_shootaround") == "600k Shootaround"
    assert strategy.pretty_tournament("15k_and_one") == "15k And One"


def test_the_break_even_lift_is_the_hurdle_in_survival_units():
    """The derivation, end to end: lifting the null by it returns the whole entry fee.

    `1 + hurdle` is `1/(1 − rake)` by construction, so `p_null · (1 + hurdle)` is the
    advance rate at which an entry worth `1 − rake` of its fee becomes worth all of it.
    """
    rake = 0.14966
    hurdle = economics.break_even_hurdle(rake)
    p_null = 1 / 6
    assert strategy.break_even_lift(p_null, hurdle) == pytest.approx(p_null * hurdle)
    assert p_null + strategy.break_even_lift(p_null, hurdle) == \
        pytest.approx(p_null / (1 - rake))


def test_the_break_even_lift_refuses_a_null_that_is_not_a_probability():
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            strategy.break_even_lift(bad, 0.176)
    with pytest.raises(ValueError):
        strategy.break_even_lift(1 / 6, -1.0)


def test_payout_elasticity_reads_one_for_a_proportional_payout():
    """The assumption the reference line makes, exercised where it is exactly true."""
    frame = _sweep(tournaments=("600k_shootaround",))
    survival = frame["p_advance"] / frame["p_advance_null"]
    frame["roi"] = (1 + frame["roi_null"]) * survival - 1.0
    out = strategy.payout_elasticity(frame)
    assert float(out["median_elasticity"].iloc[0]) == pytest.approx(1.0)
    assert float(out["share_above_proportional"].iloc[0]) == 0.0


def test_payout_elasticity_reads_above_one_when_payout_compounds():
    frame = _sweep(tournaments=("600k_shootaround",))
    survival = frame["p_advance"] / frame["p_advance_null"]
    frame["roi"] = (1 + frame["roi_null"]) * survival ** 3 - 1.0
    out = strategy.payout_elasticity(frame)
    assert float(out["median_elasticity"].iloc[0]) == pytest.approx(3.0)
    assert float(out["share_above_proportional"].iloc[0]) == 1.0


def test_an_arm_sitting_exactly_on_the_null_is_dropped_not_divided_by():
    """`log(1)` in the denominator; the row has no elasticity rather than an infinite one."""
    frame = _sweep(tournaments=("600k_shootaround",))
    frame.loc[0, "p_advance"] = frame.loc[0, "p_advance_null"]
    out = strategy.payout_elasticity(frame)
    assert int(out["n_rows"].iloc[0]) == len(frame) - 1
    assert np.isfinite(out["median_elasticity"]).all()


def test_the_shipped_sweep_is_convex_in_survival_so_the_drawn_line_is_conservative():
    """The measurement the page prints beside the reference line, on the real artifact.

    Proportional is 1. Anything above it means the break-even lift the chart draws sits
    *above* the lift a real break-even needs, which is the direction to err in.
    """
    out = strategy.payout_elasticity(_artifact(strategy.SWEEP_FILE))
    assert (out["median_elasticity"] > 1.0).all()
    assert (out["share_above_proportional"] == 1.0).all()


def test_the_contest_summary_chains_the_advance_rates_into_a_final_reach():
    summary = strategy.contest_summary(_econ(), _advance())
    row = summary[summary["tournament"] == "600k_shootaround"].iloc[0]
    assert row["p_reach_final"] == pytest.approx((2 / 12) * (1 / 12) * (1 / 10))
    assert row["r1_advance_rate"] == pytest.approx(2 / 12)
    assert bool(row["is_target"])
    assert list(summary["entry_fee_per_team"]) == sorted(summary["entry_fee_per_team"])


def test_the_derived_reach_agrees_with_the_brackets_own_analytic_column():
    """Two routes to the same number: the economics chain, and what `make bracket` wrote.

    The page draws the artifact's column and tiles the derived one, so a disagreement
    would put two different survival probabilities on one screen.
    """
    structure = _artifact(strategy.STRUCTURE_FILE)
    summary = strategy.contest_summary(economics.economics(),
                                       economics.advance_table())
    season = sorted(structure["season"].astype(str).unique())[-1]
    final = strategy.survival_frame(structure, season)
    final = final[final["n_advance"] == 0].set_index("tournament")["p_reach"]
    for row in summary.itertuples():
        assert row.p_reach_final == pytest.approx(float(final[row.tournament]))


def test_the_survival_frame_filters_by_season_and_flags_the_swept_tiers():
    structure = _artifact(strategy.STRUCTURE_FILE)
    season = sorted(structure["season"].astype(str).unique())[0]
    frame = strategy.survival_frame(structure, season)
    assert set(frame.loc[frame["is_target"], "tournament"]) == set(strategy.TARGET_TIERS)
    assert (frame[frame["round"] == 1]["p_reach"] == 1.0).all()


def test_an_unknown_season_gives_an_empty_survival_frame_rather_than_raising():
    frame = strategy.survival_frame(_artifact(strategy.STRUCTURE_FILE), "1899-00")
    assert frame.empty and "p_reach" in frame.columns


def test_the_round_ladder_marks_round_one_as_the_zero_consolation_cut():
    ladder = strategy.round_ladder(_advance(), "600k_shootaround")
    assert list(ladder["Round"]) == [1, 2, 3, 4]
    assert bool(ladder.loc[0, "Zero consolation"])
    assert not ladder.loc[1:, "Zero consolation"].any()


def test_the_sweep_panel_facets_in_the_declared_reading_order():
    panel = strategy.sweep_panel(_sweep(), "600k_shootaround")
    seen = list(dict.fromkeys(panel["axis"]))
    assert seen == [a for a in strategy.AXIS_ORDER if a in set(seen)]
    assert set(panel["tournament"]) == {"600k_shootaround"}


def test_arms_are_ordered_by_their_mean_lift_across_seasons_not_by_the_first_row():
    frame = _sweep(tournaments=("600k_shootaround",))
    # Make `adp` win 2022-23 outright while still losing on the mean.
    frame.loc[(frame["strategy"] == "adp") & (frame["season"] == "2022-23"),
              "lift_vs_null"] = 0.90
    panel = strategy.sweep_panel(frame, "600k_shootaround")
    ranking = [arms for axis, _, arms in strategy.facets(panel) if axis == "ranking"][0]
    assert ranking == ["adp", "model_mean"]
    frame.loc[frame["strategy"] == "adp", "lift_vs_null"] = -0.5
    panel = strategy.sweep_panel(frame, "600k_shootaround")
    ranking = [arms for axis, _, arms in strategy.facets(panel) if axis == "ranking"][0]
    assert ranking == ["model_mean", "adp"]


def test_the_two_tiers_get_their_own_arm_ordering():
    """Fixed order would hide that they disagree about which `α` wins — they do."""
    sweep = _artifact(strategy.SWEEP_FILE)
    orders = {}
    for tier in strategy.TARGET_TIERS:
        panel = strategy.sweep_panel(sweep, tier)
        orders[tier] = [arms for axis, _, arms in strategy.facets(panel)
                        if axis == "alpha"][0]
    assert orders[strategy.TARGET_TIERS[0]] != orders[strategy.TARGET_TIERS[1]]


def test_facets_list_each_arm_once_in_row_order():
    panel = strategy.sweep_panel(_sweep(), "600k_shootaround")
    for _, label, arms in strategy.facets(panel):
        assert label and len(arms) == len(set(arms))
    assert sum(len(arms) for _, _, arms in strategy.facets(panel)) == 5


def test_both_backtest_surfaces_land_on_one_panel_with_the_tuning_side_first():
    sweep, realized = _sweep(), _sweep()
    realized["p_advance_lo"] -= 0.3
    realized["p_advance_hi"] += 0.3
    panel = strategy.surfaces_panel(sweep, realized, "600k_shootaround",
                                    ("model_mean", "adp"))
    assert list(dict.fromkeys(panel["surface"])) == ["Simulated", "Realized"]
    assert set(panel["strategy"]) == {"model_mean", "adp"}
    assert (panel["width"] == panel["hi"] - panel["lo"]).all()


def test_an_arm_missing_from_a_surface_is_skipped_rather_than_faked():
    sweep = _sweep()
    panel = strategy.surfaces_panel(sweep, sweep, "600k_shootaround",
                                    ("model_mean", "never_swept"))
    assert set(panel["strategy"]) == {"model_mean"}


def test_the_honest_readout_has_the_wider_intervals():
    """One number for why one surface selected the strategy and the other did not."""
    sweep = _artifact(strategy.SWEEP_FILE)
    realized = _artifact(strategy.REALIZED_FILE)
    shipped = _artifact(strategy.SHIPPED_FILE)
    tier = strategy.TARGET_TIERS[0]
    arm = strategy.shipped_arm(shipped, tier)
    panel = strategy.surfaces_panel(sweep, realized, tier,
                                    (str(arm["strategy"]), "adp"))
    assert strategy.resolution_gap(panel)["ratio"] > 1.0


def test_a_tournament_that_shipped_nothing_returns_no_arm():
    assert strategy.shipped_arm(_artifact(strategy.SHIPPED_FILE), "nope") is None


def test_the_paired_panel_drops_the_self_comparison():
    """A baseline against itself is a zero-width interval at zero — arithmetic, not a
    result, and it would sit in the unresolved count forever."""
    panel = strategy.paired_panel(_paired(), "600k_shootaround", "p_advance",
                                  "model_mean")
    assert "model_mean" not in set(panel["strategy"])
    assert list(panel["gap"]) == sorted(panel["gap"], reverse=True)


def test_crossing_zero_is_derived_from_the_interval_the_chart_draws():
    """Not read from `resolved`: the styling has to follow the bar the reader sees."""
    frame = _paired()
    frame.loc[frame["strategy"] == "blend_a30", "resolved"] = True   # a drifted flag
    panel = strategy.paired_panel(frame, "600k_shootaround", "p_advance", "model_mean")
    row = panel[panel["strategy"] == "blend_a30"].iloc[0]
    assert bool(row["crosses_zero"])
    assert strategy.unresolved(panel) == (2, 5)


def test_the_derived_flag_agrees_with_the_shipped_artifacts_own_resolved_column():
    """They do agree today, and this is what says so if a future run stops agreeing."""
    paired = _artifact(strategy.PAIRED_FILE)
    for tournament in paired["tournament"].unique():
        for metric in paired["metric"].unique():
            for baseline in paired["baseline"].unique():
                panel = strategy.paired_panel(paired, tournament, metric, baseline)
                assert (panel["crosses_zero"] == ~panel["resolved"].astype(bool)).all()


def test_some_gaps_do_not_resolve_which_is_the_block_s_reason_to_exist():
    paired = _artifact(strategy.PAIRED_FILE)
    panel = strategy.paired_panel(paired, strategy.TARGET_TIERS[0], "p_advance",
                                  "model_mean")
    crossing, total = strategy.unresolved(panel)
    assert 0 < crossing < total


# ── The tournament page's figures ─────────────────────────────────────────────

def _panel():
    frame = _sweep()
    return strategy.sweep_panel(frame, "600k_shootaround")


def test_five_tournaments_stay_inside_the_all_pairs_cap_by_graying_the_rest():
    th = theme.theme("light")
    names = ["15k_and_one", "20k_spin_move", "50k_four_pt_play", "600k_shootaround",
             "88k_alley_oop"]
    colors = charts._tier_colors(th, names, strategy.TARGET_TIERS)
    assert colors[names.index("600k_shootaround")] == th["series"][0]
    assert colors[names.index("20k_spin_move")] == th["series"][1]
    assert [c for c in colors if c == th["muted"]] == [th["muted"]] * 3
    assert len({c for c in colors if c != th["muted"]}) <= theme.ALL_PAIRS_CAP


def test_the_highlighted_tiers_take_their_slot_from_the_declared_order():
    """So the survival curve and the hurdle bars agree on which tier is which colour."""
    th = theme.theme("light")
    forward = charts._tier_colors(th, list(strategy.TARGET_TIERS),
                                  strategy.TARGET_TIERS)
    backward = charts._tier_colors(th, list(reversed(strategy.TARGET_TIERS)),
                                   strategy.TARGET_TIERS)
    assert forward == [th["series"][0], th["series"][1]]
    assert backward == [th["series"][1], th["series"][0]]


def test_the_survival_curve_is_logarithmic_and_draws_the_highlights_last():
    """`20k_spin_move` and `88k_alley_oop` share an advance chain exactly, so a gray
    curve drawn last would hide a highlighted one."""
    th = theme.theme("light")
    frame = strategy.survival_frame(_artifact(strategy.STRUCTURE_FILE), "2022-23")
    fig = charts.fig_survival(frame, th, strategy.TARGET_TIERS,
                              strategy.pretty_tournament)
    assert fig.layout.yaxis.type == "log"
    assert len(fig.data) == frame["tournament"].nunique()
    drawn = [t.name for t in fig.data]
    assert drawn[-2:] == [strategy.pretty_tournament(t)
                          for t in strategy.TARGET_TIERS]


def test_every_hurdle_bar_prints_its_own_value():
    """Three light-mode slots fall under 3:1 on the light surface, so a bar cannot be
    read by colour alone."""
    th = theme.theme("light")
    fig = charts.fig_hurdle(strategy.contest_summary(_econ(), _advance()), th,
                            strategy.TARGET_TIERS, strategy.pretty_tournament)
    bar = fig.data[0]
    assert len(bar.text) == len(bar.x)
    assert all("%" in t for t in bar.text)


def test_the_sweep_draws_one_subplot_per_axis_with_both_reference_lines_in_each():
    th = theme.theme("light")
    panel = _panel()
    facets = strategy.facets(panel)
    fig = charts.fig_sweep(panel, facets, th, 0.0293)
    assert len(fig.layout.shapes) == 2 * len(facets)
    assert {s.x0 for s in fig.layout.shapes} == {0.0, 0.0293}
    titles = [a.text for a in fig.layout.annotations]
    for _, label, _ in facets:
        assert label in titles


def test_the_sweep_nudges_its_two_seasons_off_the_shared_row():
    """Plotly offsets grouped bars and not grouped scatter, so two intervals at one
    category would sit exactly on top of each other."""
    th = theme.theme("light")
    panel = _panel()
    fig = charts.fig_sweep(panel, strategy.facets(panel), th, 0.0293)
    rows = sorted({round(float(y), 6) for trace in fig.data for y in trace.y})
    assert all(abs(y - round(y)) > 1e-6 for y in rows)
    assert len({round(abs(y - round(y)), 6) for y in rows}) == 1


def test_the_sweep_keeps_the_best_arm_on_the_top_row_of_its_facet():
    th = theme.theme("light")
    panel = _panel()
    facets = strategy.facets(panel)
    fig = charts.fig_sweep(panel, facets, th, 0.0293)
    axes = [fig.layout[f"yaxis{'' if i == 1 else i}"] for i in range(1, len(facets) + 1)]
    for axis, (_, _, arms) in zip(axes, facets):
        assert list(axis.ticktext) == arms
        assert axis.range[0] > axis.range[1]          # inverted: index 0 is the top row


def test_the_sweep_stays_inside_the_all_pairs_cap_with_its_reference_line():
    """Two seasons plus the break-even line is exactly three categorical slots."""
    th = theme.theme("light")
    panel = _panel()
    fig = charts.fig_sweep(panel, strategy.facets(panel), th, 0.0293)
    used = {t.marker.color for t in fig.data}
    used |= {s.line.color for s in fig.layout.shapes} - {th["axis"]}
    assert used <= set(th["series"][:theme.ALL_PAIRS_CAP])


def test_a_row_position_axis_carries_no_zero_line():
    """Row 0 is an arm, not an origin — plotly's zeroline drew a rule through the top
    row of every panel until this was turned off."""
    th = theme.theme("light")
    panel = _panel()
    sweep = charts.fig_sweep(panel, strategy.facets(panel), th, 0.0293)
    assert sweep.layout.yaxis.zeroline is False
    paired = charts.fig_paired(
        strategy.paired_panel(_paired(), "600k_shootaround", "p_advance",
                              "model_mean"), th, "model_mean", "P(top 2 of 12)")
    assert paired.layout.yaxis.zeroline is False


def test_the_two_surfaces_are_separated_by_a_rule_and_by_their_row_labels():
    th = theme.theme("light")
    sweep, realized = _sweep(), _sweep()
    panel = strategy.surfaces_panel(sweep, realized, "600k_shootaround",
                                    ("model_mean", "adp"))
    fig = charts.fig_surfaces(panel, th, 1 / 6, ("model_mean", "adp"))
    rules = [s for s in fig.layout.shapes if s.y0 == s.y1]
    assert len(rules) == 1
    assert [t for t in fig.layout.yaxis.ticktext if t.startswith("Simulated")]
    assert [t for t in fig.layout.yaxis.ticktext if t.startswith("Realized")]
    assert any(s.x0 == pytest.approx(1 / 6) for s in fig.layout.shapes)


def test_an_unresolved_gap_is_drawn_hollow_in_its_own_slot():
    th = theme.theme("light")
    panel = strategy.paired_panel(_paired(), "600k_shootaround", "p_advance",
                                  "model_mean")
    fig = charts.fig_paired(panel, th, "model_mean", "P(top 2 of 12)")
    by_name = {t.name: t for t in fig.data}
    assert set(by_name) == {"resolves", "does not resolve"}
    assert by_name["resolves"].marker.symbol == "circle"
    assert by_name["does not resolve"].marker.symbol == "circle-open"
    assert by_name["resolves"].marker.color == th["series"][0]
    assert by_name["does not resolve"].marker.color == th["series"][1]


def test_only_the_gaps_whose_interval_covers_zero_are_drawn_hollow():
    """The redundancy that matters: the hollow marker and the straddled zero line say
    the same thing, so neither has to be believed on its own."""
    th = theme.theme("light")
    panel = strategy.paired_panel(_paired(), "600k_shootaround", "p_advance",
                                  "model_mean")
    fig = charts.fig_paired(panel, th, "model_mean", "P(top 2 of 12)")
    hollow = [t for t in fig.data if t.marker.symbol == "circle-open"][0]
    for x, minus, plus in zip(hollow.x, hollow.error_x.arrayminus,
                              hollow.error_x.array):
        assert x - minus <= 0 <= x + plus


def test_a_panel_with_nothing_unresolved_draws_only_the_resolved_series():
    th = theme.theme("light")
    frame = _paired()
    frame.loc[frame["strategy"].isin(["blend_a30", "blend_stack4"]),
              ["gap", "gap_lo", "gap_hi"]] = [0.2, 0.15, 0.25]
    panel = strategy.paired_panel(frame, "600k_shootaround", "p_advance", "model_mean")
    fig = charts.fig_paired(panel, th, "model_mean", "P(top 2 of 12)")
    assert [t.name for t in fig.data] == ["resolves"]


def test_every_tournament_figure_carries_an_explicit_title():
    """A title object with a font and no text renders as the literal string "undefined"
    in a browser, and only in a browser."""
    th = theme.theme("light")
    panel = _panel()
    sweep, realized = _sweep(), _sweep()
    figures = [
        charts.fig_survival(
            strategy.survival_frame(_artifact(strategy.STRUCTURE_FILE), "2022-23"),
            th, strategy.TARGET_TIERS, strategy.pretty_tournament),
        charts.fig_hurdle(strategy.contest_summary(_econ(), _advance()), th,
                          strategy.TARGET_TIERS, strategy.pretty_tournament),
        charts.fig_sweep(panel, strategy.facets(panel), th, 0.0293),
        charts.fig_surfaces(
            strategy.surfaces_panel(sweep, realized, "600k_shootaround",
                                    ("model_mean", "adp")),
            th, 1 / 6, ("model_mean", "adp")),
        charts.fig_paired(
            strategy.paired_panel(_paired(), "600k_shootaround", "p_advance",
                                  "model_mean"), th, "model_mean", "P(top 2 of 12)"),
    ]
    for fig in figures:
        assert fig.layout.title.text is not None
        assert fig.layout.paper_bgcolor == th["surface"]


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
    assert {"theme.py", "charts.py", "pca.py", "strategy.py", "decisions.py",
            "economics.py", "audit.py"} <= pure


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
    `dashboard/shell.py` would have nothing to survive. The shell held the second row with
    a placeholder until the tournament page took it on 2026-08-10; the constraint outlives
    the placeholder, which is why this assertion does.
    """
    from dashboard import app
    assert len(app.VIEWS) >= 2


def test_every_navigation_row_is_a_real_view_module():
    """The placeholder is gone, and a new one would be a `views/placeholder.py` again.

    `pages()` hands `st.Page` a bare callable, so a row whose `render` came from anywhere
    but a view module — a closure, a lambda, a stub — would still navigate perfectly well
    and put something on a URL that no module owns.
    """
    from dashboard import app
    for view in app.VIEWS:
        module = getattr(view.render, "__module__", "")
        assert module.startswith("dashboard.views."), (view.title, module)
        assert view.render.__name__ == "render", view.title


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


# ── The model pages ───────────────────────────────────────────────────────────
#
# `dashboard/model_cards.py` reshapes the eight `make model-cards` artifacts into the seven
# blocks `views/model_page.py` draws. The coverage rule is the one the emitter's own tests
# follow: **one case per way this layer can be wrong silently**, because every one of those
# renders as a perfectly good-looking page — a spline basis scattered through an
# alphabetical grid, a discrete bar with no width, a colourbar labelling four panels that do
# not share a scale, a diagnostics cell reading `None` where the source carries nothing.
#
# The handful that read `outputs/predictions/` are the artifact-contract tests: they keep
# this module's *declarations* — which heads make a page, where each head's sampler row
# lives — honest against the artifacts they describe, which is the same job the PCA anchor
# tests do. They skip rather than fail without `make model-cards`.

def _index(**overrides) -> pd.DataFrame:
    """An index row shaped like the artifact, for one synthetic head."""
    base = dict(
        head="synthetic", label="synthetic", model_class="availability",
        class_label="Availability", unit="player-season", family="betabinomial",
        likelihood="beta-binomial", description="A head.", variant="base",
        n_features=3, n_terms=5, n_fit=900, n_validation=100, n_frame_rows=900,
        row_filter="", n_draws=1000, n_density_pairs=3, fit_window="train",
        first_season="1997-98", last_season="2021-22", response="mean_mu",
        dispersion="rho", max_rhat=1.001, divergences=0, converged=True,
        coefficient_scale="standardized", recipe_design_error=0.0,
        roundtrip_prediction_error=0.0, design_check="vacuous", verified=True,
        response_label="games played", predictive_draws=200, n_predictive_train=900,
        n_predictive_validation=100, predictive_rows_capped=False,
        predictive_weighted=False, fitted_source="head_predict",
        predictive_check="mean", predictive_bias=0.002, ecdf_band_mc=0.01,
        ecdf_band_gated=True, player_season_sigma=0.0, git_sha="abcdef123456",
        built_at="2026-08-10T00:00:00+00:00")
    return pd.DataFrame([{**base, **overrides}])


def _feature_rows(head: str = "synthetic") -> pd.DataFrame:
    """Two splits x three features, one of them a two-basis spline, one an imputation flag."""
    rows = []
    specs = [("log_x__s0", "log_x", 0, "linear", [0.0, 1.0, 2.0], 0.10),
             ("log_x__s1", "log_x", 1, "linear", [0.0, 1.0, 2.0], 0.10),
             ("age", "age", -1, "linear", [20.0, 25.0, 30.0], 0.0),
             ("x__miss", "missingness", -1, "discrete", [0.0, 1.0], 0.0)]
    for feature, family, basis, kind, edges, missing in specs:
        left = edges if kind == "discrete" else edges[:-1]
        right = edges if kind == "discrete" else edges[1:]
        for split, n, share in (("train", 900, missing), ("validation", 100, missing / 2)):
            for i, (lo, hi) in enumerate(zip(left, right)):
                rows.append({
                    "head": head, "feature": feature, "split": split, "bin_index": i,
                    "term_family": family, "basis_index": basis, "n": n, "n_finite": n,
                    "mean": 1.0, "sd": 0.5, "min": float(edges[0]), "q05": 0.1,
                    "q50": 0.5, "q95": 0.9, "max": float(edges[-1]),
                    "missing_share": share, "bin_kind": kind, "n_bins": len(left),
                    "bin_left": float(lo), "bin_right": float(hi),
                    "count": 10 * (i + 1), "density": 0.5 / len(left) * (i + 1)})
    return pd.DataFrame(rows)


def _correlations(head: str = "synthetic", constant: str = "x__miss") -> pd.DataFrame:
    features = ["log_x__s0", "log_x__s1", "age", "x__miss"]
    values = {("log_x__s0", "log_x__s1"): 0.9, ("log_x__s0", "age"): 0.4,
              ("log_x__s1", "age"): -0.2}
    rows = []
    for split in ("train", "validation"):
        pairs = []
        for i, x in enumerate(features):
            for y in features[i + 1:]:
                r = values.get((x, y), values.get((y, x), 0.05))
                if split == "validation" and constant in (x, y):
                    r = float("nan")
                pairs.append((abs(r) if np.isfinite(r) else -1.0, x, y))
        pairs.sort(key=lambda p: -p[0])
        rank = {(x, y): n + 1 for n, (score, x, y) in enumerate(pairs) if score >= 0}
        for i, x in enumerate(features):
            for j, y in enumerate(features):
                if x == y:
                    r = 1.0
                else:
                    r = values.get((x, y), values.get((y, x), 0.05))
                    if split == "validation" and constant in (x, y):
                        r = float("nan")
                position = rank.get((x, y)) or rank.get((y, x))
                rows.append({"head": head, "split": split, "feature_x": x,
                             "feature_y": y, "i": i, "j": j, "r": r,
                             "abs_r": abs(r), "n": 900,
                             "pair_rank": position or -1,
                             "top_pair": bool(position and position <= 2)})
    return pd.DataFrame(rows)


def _density(head: str = "synthetic") -> pd.DataFrame:
    rows = []
    for split, n in (("train", 900), ("validation", 100)):
        for i in range(3):
            for j in range(3):
                rows.append({"head": head, "feature_x": "log_x__s0",
                             "feature_y": "log_x__s1", "pair_rank": 1, "split": split,
                             "r": 0.9, "x_index": i, "y_index": j,
                             "x_left": float(i), "x_right": float(i + 1),
                             "y_left": float(j), "y_right": float(j + 1),
                             "count": 10 * (i + j + 1),
                             "density": 0.02 * (i + j + 1), "n": n})
    return pd.DataFrame(rows)


def _coefficients(head: str = "synthetic") -> pd.DataFrame:
    terms = [("(intercept)", "intercept", "intercept", -1, 0.80),
             ("log_x__s0", "log_x", "coefficient", 0, -0.10),
             ("log_x__s1", "log_x", "coefficient", 1, -0.55),
             ("age", "age", "coefficient", -1, 0.30),
             ("rho", "dispersion", "dispersion", -1, 0.28)]
    return pd.DataFrame([{
        "head": head, "term": term, "term_family": family, "term_role": role,
        "basis_index": basis, "n_draws": 1000, "mean": mean, "sd": 0.05,
        "q2.5": mean - 0.1, "q25": mean - 0.05, "q50": mean, "q75": mean + 0.05,
        "q97.5": mean + 0.1, "p_positive": 1.0 if mean > 0 else 0.0,
        "scaler_center": 0.0, "scaler_scale": 1.0}
        for term, family, role, basis, mean in terms])


def _ecdf(head: str = "synthetic", offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for split, n in (("train", 900), ("validation", 100)):
        for k in range(10):
            q50 = (k + 1) / 10
            rows.append({
                "head": head, "split": split, "grid_index": k, "value": float(k),
                "observed": min(1.0, q50 + (offset if split == "train" else 0.0)),
                "grid_kind": "quantile", "n_rows": n, "n_draws": 200,
                "q2.5": q50 - 0.05, "q10": q50 - 0.04, "q25": q50 - 0.02,
                "q50": q50, "q75": q50 + 0.02, "q90": q50 + 0.04,
                "q97.5": q50 + 0.05})
    return pd.DataFrame(rows)


def _calibration(head: str = "synthetic") -> pd.DataFrame:
    rows = []
    for panel in ("fitted_observed", "residual_fitted"):
        for split, n in (("train", 900), ("validation", 100)):
            for i, (count, y) in enumerate([(90, 0.0), (10, 10.0)]):
                rows.append({
                    "head": head, "split": split, "panel": panel, "x_index": i,
                    "y_index": i, "x_left": float(i), "x_right": float(i + 1),
                    "y_left": y, "y_right": y + 1.0, "count": count,
                    "density": count / 100.0, "n": n})
    return pd.DataFrame(rows)


def _sample(head: str = "synthetic") -> pd.DataFrame:
    return pd.DataFrame([
        {"head": head, "split": split, "row": i, "fitted": 1.0 + i,
         "observed": 2.0 + i, "residual": 1.0}
        for split in ("train", "validation") for i in range(5)])


def _cards() -> dict:
    return {"index": _index(), "coefficients": _coefficients(),
            "features": _feature_rows(), "correlations": _correlations(),
            "density": _density(), "ecdf": _ecdf(), "calibration": _calibration(),
            "sample": _sample()}


# ── Which heads make a page ───────────────────────────────────────────────────

def test_a_class_lists_its_heads_in_the_declared_order_not_the_artifact_order():
    """Entry, onset, duration, exit is how a tenure runs; no column carries that."""
    index = pd.DataFrame({"head": ["gp_exit", "availability", "gp_onset", "gp_entry",
                                   "gp_duration"]})
    assert model_cards.heads_of(index, "availability") == [
        "availability", "gp_entry", "gp_onset", "gp_duration", "gp_exit"]


def test_a_head_the_emitter_has_not_written_is_skipped_rather_than_raising():
    """A half-built `outputs/predictions/` draws the heads it has."""
    index = pd.DataFrame({"head": ["availability", "gp_entry"]})
    assert model_cards.heads_of(index, "availability") == ["availability", "gp_entry"]
    assert model_cards.heads_of(pd.DataFrame({"head": []}), "minutes") == []


def test_an_unknown_page_or_head_raises_by_name():
    with pytest.raises(KeyError):
        model_cards.model_class("nonsense")
    with pytest.raises(KeyError, match="model-cards"):
        model_cards.head_row(_index(), "absent")


def test_every_declared_page_carries_a_unique_url_path_and_an_intro():
    paths = [c.url_path for c in model_cards.CLASSES.values()]
    assert len(paths) == len(set(paths))
    for spec in model_cards.CLASSES.values():
        assert spec.heads and spec.intro.strip() and spec.icon.startswith(":material/")


# ── Block 1 ───────────────────────────────────────────────────────────────────

def test_the_specification_states_the_unit_and_reads_every_field_from_the_artifact():
    spec = model_cards.specification(_index().iloc[0])
    fields = dict(zip(spec["Field"], spec["Value"]))
    assert fields["Unit"] == "player-season"
    assert fields["Response"] == "games played"
    assert fields["Fitted seasons"] == "1997-98 → 2021-22"
    # The row filter is a row only where the head has one, so a page does not print
    # "Row filter: nan" for the sixteen heads that drop nothing.
    assert "Row filter" not in set(spec["Field"])
    filtered = model_cards.specification(_index(row_filter="fg3a > 0").iloc[0])
    assert dict(zip(filtered["Field"], filtered["Value"]))["Row filter"] == "fg3a > 0"


# ── Block 2 ───────────────────────────────────────────────────────────────────

def test_a_spline_family_stays_together_in_the_small_multiple_order():
    order = model_cards.feature_order(_feature_rows(), "synthetic")
    assert order.index("log_x__s1") == order.index("log_x__s0") + 1


def test_the_feature_table_carries_the_imputed_share_of_each_split_separately():
    """The composition imputes 17.0% of train and 12.6% of validation; one number hides it."""
    summary = model_cards.feature_summary(_feature_rows(), "synthetic")
    row = summary[summary["Feature"] == "log_x__s0"].iloc[0]
    assert row["n (Train)"] == 900 and row["n (Validation)"] == 100
    assert row["Imputed share"] == pytest.approx(0.10)


def test_a_discrete_feature_gets_a_drawable_bar_width():
    """Its edges are the values themselves, so left == right — and plotly refuses a
    zero-width or NaN bar outright, which is an exception rather than a bad picture."""
    panel = model_cards.histogram_panel(_feature_rows(), "synthetic")
    flag = panel[panel["feature"] == "x__miss"]
    assert (flag["bin_width"] > 0).all()
    assert flag["bin_width"].max() < 1.0        # under the gap, so two values stay two bars
    assert (panel[panel["feature"] == "age"]["bin_width"] == 5.0).all()


def test_the_histogram_panel_plots_shares_so_two_split_sizes_are_comparable():
    panel = model_cards.histogram_panel(_feature_rows(), "synthetic")
    assert {"density", "bin_center", "panel_index"} <= set(panel.columns)
    assert panel["panel_index"].is_monotonic_increasing


# ── Block 3 ───────────────────────────────────────────────────────────────────

def test_the_correlation_square_is_a_reshape_in_the_design_matrix_order():
    square = model_cards.correlation_square(_correlations(), "synthetic", "train")
    assert list(square.index) == list(square.columns)
    assert list(square.index) == ["log_x__s0", "log_x__s1", "age", "x__miss"]
    assert np.allclose(np.diag(square.to_numpy()), 1.0)


def test_a_column_constant_on_a_split_is_named_rather_than_dropped_from_the_axis():
    """Three real features are identically constant on validation; that is a finding."""
    assert model_cards.constant_features(_correlations(), "synthetic", "validation") == \
        ["x__miss"]
    assert model_cards.constant_features(_correlations(), "synthetic", "train") == []


def test_the_pair_menu_offers_only_pairs_that_have_a_density_behind_them():
    menu = model_cards.pair_menu(_correlations(), "synthetic")
    assert list(menu["pair_rank"]) == [1, 2]
    assert (menu["feature_x"] != menu["feature_y"]).all()
    assert "r = +0.90" in model_cards.pair_label(menu.iloc[0])


def test_the_density_panel_centres_its_cells_and_filters_to_one_split():
    cells = model_cards.density_panel(_density(), "synthetic", "log_x__s0", "log_x__s1",
                                      "validation")
    assert len(cells) == 9 and set(cells["split"]) == {"validation"}
    assert cells["x_center"].iloc[0] == pytest.approx(0.5)
    assert int(cells["n"].iloc[0]) == 100


# ── Block 4 ───────────────────────────────────────────────────────────────────

def test_the_coefficient_panel_leaves_the_intercept_and_dispersion_out():
    """They are not on the standardized slope scale, and the intercept would set the axis."""
    panel = model_cards.coefficient_panel(_coefficients(), "synthetic")
    assert set(panel["term"]) == {"log_x__s0", "log_x__s1", "age"}
    scalars = model_cards.scalar_terms(_coefficients(), "synthetic")
    assert set(scalars["term"]) == {"(intercept)", "rho"}


def test_the_panel_opens_on_its_strongest_family_with_the_bases_in_order():
    """Row 0 is the top of the figure — a panel that opens on its weakest term buries
    the answer, which is what the first cut did until it was rendered."""
    panel = model_cards.coefficient_panel(_coefficients(), "synthetic")
    assert list(panel["term"]) == ["log_x__s0", "log_x__s1", "age"]
    assert list(panel["basis_index"])[:2] == [0, 1]


def test_collapsing_a_family_keeps_its_widest_basis_and_says_so():
    panel = model_cards.coefficient_panel(_coefficients(), "synthetic", collapse=True)
    assert len(panel) == 2
    row = panel[panel["term_family"] == "log_x"].iloc[0]
    assert row["term"] == "log_x__s1"          # |−0.55| beats |−0.10|
    assert "widest of 2 bases" in row["label"]
    assert panel[panel["term_family"] == "age"].iloc[0]["label"] == "age"


def test_the_coefficient_table_twin_carries_the_interval_the_bar_was_drawn_from():
    panel = model_cards.coefficient_panel(_coefficients(), "synthetic")
    table = model_cards.coefficient_table(panel)
    assert list(table["Term"]) == list(panel["label"])
    assert np.allclose(table["2.5%"], panel["q2.5"], atol=1e-6)


# ── Block 5 ───────────────────────────────────────────────────────────────────

def test_the_ribbon_is_read_as_a_distance_with_coverage_as_the_footnote():
    """At n ~ 10^4 every head leaves the band somewhere, so in-or-out is not the reading."""
    distance = model_cards.band_distance(_ecdf(offset=0.03), "synthetic")
    train = distance[distance["split"] == "train"].iloc[0]
    validation = distance[distance["split"] == "validation"].iloc[0]
    assert train["max_gap"] == pytest.approx(0.03)
    assert validation["max_gap"] == pytest.approx(0.0)
    assert train["inside_95"] == pytest.approx(1.0)     # 0.03 is inside a +/-0.05 band
    # A curve 0.30 off a +/-0.05 band leaves it everywhere except where the ECDF saturates
    # at 1 — which is the shape of the real thing: coverage collapses, the distance does not.
    far = model_cards.band_distance(_ecdf(offset=0.30), "synthetic").iloc[0]
    assert far["inside_95"] < 0.2 and far["max_gap"] == pytest.approx(0.30)


def test_the_provenance_line_says_what_was_actually_drawn():
    line = model_cards.predictive_provenance(_index().iloc[0])
    assert "200 posterior draws" in line and "900 training rows" in line
    capped = model_cards.predictive_provenance(
        _index(predictive_rows_capped=True, predictive_weighted=True).iloc[0])
    assert "subsample" in capped and "multiplicity" in capped


# ── Block 6 ───────────────────────────────────────────────────────────────────

def test_the_binned_summary_weights_by_the_cell_counts_it_draws():
    """Unweighted, the two cells of the synthetic panel average to 5.0 rather than 1.0."""
    summary = model_cards.calibration_summary(_calibration(), "synthetic")
    row = summary[(summary["Panel"] == "Predicted against observed")
                  & (summary["Split"] == "Train")].iloc[0]
    assert row["Mean observed"] == pytest.approx(0.9 * 0.5 + 0.1 * 10.5)
    assert row["n"] == 900 and row["Cells"] == 2


def test_the_sample_overlay_is_filtered_to_its_own_split():
    points = model_cards.sample_points(_sample(), "synthetic", "validation")
    assert len(points) == 5 and set(points["split"]) == {"validation"}


# ── Block 7 ───────────────────────────────────────────────────────────────────

def _diagnostics(label: str = "stan_posterior") -> pd.DataFrame:
    return pd.DataFrame([{"label": label, "max_rhat": 1.002, "min_ess_bulk": 2313.0,
                          "min_ess_tail": 2516.0, "divergences": 0,
                          "treedepth_saturated": 0, "n_draws": 4000,
                          "wall_clock_s": 196.4, "converged": True,
                          "cmdstan": "cmdstan-2.39.0"}])


def _manifest(head: str = "synthetic") -> pd.DataFrame:
    return pd.DataFrame([{"head": head, "n_draws_before_thinning": 4000,
                          "fit_seconds": 324.9, "cmdstan": "cmdstan-2.39.0"}])


def test_a_heads_diagnostics_label_is_filled_from_its_own_index_row():
    """A head that ships a different variant follows its own row, not an edit here."""
    row = _index(head="gp_onset", variant="duration_covariates").iloc[0]
    assert model_cards.diagnostics_source(row) == (
        model_cards.GAMES_PLAYED_DIAGNOSTICS, "duration_covariates/val/onset")
    component = _index(head="fg3a_given_fga", label="fg3a|fga", model_class="components",
                       variant="logit_own_spline").iloc[0]
    assert model_cards.diagnostics_source(component) == (
        model_cards.COMPONENT_DIAGNOSTICS, "fg3a|fga/logit_own_spline/val")


def test_a_head_with_no_declared_diagnostics_source_raises_rather_than_drawing_nothing():
    with pytest.raises(KeyError, match="DIAGNOSTICS"):
        model_cards.diagnostics_source(_index(head="unknown", model_class="unknown")
                                       .iloc[0])


def test_the_two_sampler_runs_are_two_rows_and_neither_borrows_the_others_numbers():
    # A real head name, since the label a run is looked up under is derived from the row.
    runs = model_cards.sampler_runs(_index(head="availability").iloc[0],
                                    _manifest("availability"), _diagnostics())
    assert len(runs) == 2
    persisted, selection = runs.iloc[0], runs.iloc[1]
    # `make posteriors` records no ESS and `make stan` kept no draws. A blank in either
    # direction has to read as absent rather than as a measurement of zero.
    assert persisted["ESS bulk"] == model_cards.ABSENT
    assert persisted["Treedepth hits"] == model_cards.ABSENT
    assert selection["Git SHA"] == model_cards.ABSENT
    assert "none kept" in selection["Draws"] and "1,000 kept" in persisted["Draws"]
    assert persisted["Max R̂"] == "1.00100" and selection["Max R̂"] == "1.00200"
    assert persisted["Wall clock"] == "325 s" and selection["Wall clock"] == "196 s"


def test_a_missing_diagnostics_table_leaves_the_persisted_row_alone():
    runs = model_cards.sampler_runs(_index(head="availability").iloc[0], None, None)
    assert len(runs) == 1 and runs.iloc[0]["Draws"].endswith("kept")
    assert runs.iloc[0]["CmdStan"] == model_cards.ABSENT


def test_the_build_checks_say_which_of_them_could_have_failed():
    """`vacuous` marks the heads where check 2 compares a frame with itself."""
    checks = model_cards.build_checks(_index().iloc[0])
    kinds = dict(zip(checks["Check"], checks["Kind"]))
    assert kinds["Recipe against the head's own ladder"] == "vacuous"
    assert dict(zip(checks["Check"], checks["Reading"]))[
        "Drawn mean against reported mean"] == "+0.20%"
    uncheckable = model_cards.build_checks(
        _index(predictive_bias=float("nan"), ecdf_band_gated=False).iloc[0])
    readings = dict(zip(uncheckable["Check"], uncheckable["Reading"]))
    assert readings["Drawn mean against reported mean"] == "not checkable"
    assert dict(zip(uncheckable["Check"], uncheckable["Kind"]))[
        "Ribbon stability at the draw budget"] == "reported only"


# ── The figures ───────────────────────────────────────────────────────────────

def _model_figures(th: dict) -> list:
    cards = _cards()
    panel = model_cards.histogram_panel(cards["features"], "synthetic")
    square = model_cards.correlation_square(cards["correlations"], "synthetic", "train")
    cells = model_cards.density_panel(cards["density"], "synthetic", "log_x__s0",
                                      "log_x__s1", "train")
    ecdf = {model_cards.SPLIT_LABELS[s]: model_cards.ecdf_panel(cards["ecdf"],
                                                                "synthetic", s)
            for s in model_cards.SPLITS}
    grids = {(p, s): model_cards.calibration_panel(cards["calibration"], "synthetic", p, s)
             for p in model_cards.PANELS for s in model_cards.SPLITS}
    points = {(p, s): model_cards.sample_points(cards["sample"], "synthetic", s)
              for p in model_cards.PANELS for s in model_cards.SPLITS}
    return [
        charts.fig_features(panel, th, title="features"),
        charts.fig_correlation(square, th, title="corr"),
        charts.fig_joint(cells, th, "log_x__s0", "log_x__s1", title="joint"),
        charts.fig_coefficients(
            model_cards.coefficient_panel(cards["coefficients"], "synthetic"), th,
            title="coefficients"),
        charts.fig_ecdf(ecdf, th, "games played", title="ecdf"),
        charts.fig_calibration(grids, points, th, model_cards.PANEL_AXES,
                               model_cards.PANEL_LABELS, title="calibration"),
    ]


def test_each_feature_gets_a_filled_train_bar_and_a_stepped_validation_line():
    """Two splits, two marks — so the comparison survives without colour."""
    th = theme.theme("light")
    fig = charts.fig_features(
        model_cards.histogram_panel(_feature_rows(), "synthetic"), th)
    bars = [t for t in fig.data if t.type == "bar"]
    lines = [t for t in fig.data if t.type == "scatter"]
    assert len(bars) == len(lines) == 4                    # one per feature
    assert all(t.line.shape == "hvh" for t in lines)
    assert sum(t.showlegend for t in fig.data) == 2        # one legend entry per split


def test_the_correlation_heatmap_is_pinned_to_the_full_range_around_zero():
    """Pinned rather than scaled to the data, so two heads' heatmaps mean the same thing."""
    fig = charts.fig_correlation(
        model_cards.correlation_square(_correlations(), "synthetic", "train"),
        theme.theme("light"))
    heat = fig.data[0]
    assert (heat.zmin, heat.zmid, heat.zmax) == (-1.0, 0.0, 1.0)
    assert heat.colorscale == tuple(tuple(step) for step in theme.theme("light")["diverging"])
    assert fig.layout.yaxis.scaleanchor == "x"             # square cells


def test_a_density_leaves_its_empty_cells_transparent_rather_than_at_the_ramp_floor():
    """A zero would read as 'measured and low' instead of 'no rows landed here'."""
    cells = model_cards.density_panel(_density(), "synthetic", "log_x__s0", "log_x__s1",
                                      "train")
    fig = charts.fig_joint(cells.iloc[:4], theme.theme("light"), "x", "y")
    z = np.asarray(fig.data[0].z, dtype=float)
    assert np.isnan(z).any() and fig.data[0].hoverongaps is False


def test_the_coefficient_bars_take_the_two_ends_of_the_diverging_scale():
    """A coefficient's sign is a direction on one axis, not two categorical slots."""
    th = theme.theme("light")
    panel = model_cards.coefficient_panel(_coefficients(), "synthetic")
    fig = charts.fig_coefficients(panel, th)
    colors = list(fig.data[0].marker.color)
    assert set(colors) == {th["diverging"][0][1], th["diverging"][-1][1]}
    assert colors[-1] == th["diverging"][-1][1]            # `age` is positive
    # Row 0 at the top, and no zeroline on an axis whose 0 is a term rather than an origin.
    assert fig.layout.yaxis.range == (len(panel) - 0.5, -0.5)
    assert fig.layout.yaxis.zeroline is False
    assert fig.layout.xaxis.zeroline is True
    assert len(fig.layout.shapes) == len(panel)            # one interval rule per term


def test_the_ribbon_draws_three_nested_bands_under_one_observed_curve():
    th = theme.theme("light")
    panels = {model_cards.SPLIT_LABELS[s]: model_cards.ecdf_panel(_ecdf(), "synthetic", s)
              for s in model_cards.SPLITS}
    fig = charts.fig_ecdf(panels, th, "games played")
    names = [t.name for t in fig.data]
    assert names.count("95% band") == names.count("50% band") == 2
    assert names.count("observed") == 2 and names.count("median replicate") == 2
    assert sum(t.showlegend for t in fig.data) == 5        # one legend for both subplots
    observed = [t for t in fig.data if t.name == "observed"]
    assert all(t.line.color == th["ink"] for t in observed)


def test_the_four_calibration_panels_share_a_colourbar_only_because_each_is_relative():
    """Four panels on four absolute scales under one legend would label three of them
    wrongly — a 100-row validation panel puts far more share in a cell than a 900-row one."""
    th = theme.theme("light")
    cards = _cards()
    grids = {(p, s): model_cards.calibration_panel(cards["calibration"], "synthetic", p, s)
             for p in model_cards.PANELS for s in model_cards.SPLITS}
    points = {(p, s): model_cards.sample_points(cards["sample"], "synthetic", s)
              for p in model_cards.PANELS for s in model_cards.SPLITS}
    fig = charts.fig_calibration(grids, points, th, model_cards.PANEL_AXES,
                                 model_cards.PANEL_LABELS)
    heatmaps = [t for t in fig.data if t.type == "heatmap"]
    assert len(heatmaps) == 4
    assert sum(t.showscale for t in heatmaps) == 1
    for heat in heatmaps:
        assert np.nanmax(np.asarray(heat.z, dtype=float)) == pytest.approx(1.0)
        assert np.nanmax(np.asarray(heat.customdata, dtype=float)) < 1.0   # the raw share


def test_both_splits_of_a_calibration_panel_are_drawn_on_one_axis_range():
    """The emitter clips the density's tails into its end bins; an unclipped overlay
    setting the axis undoes that, which squashed a 2-to-9-game density into a sliver."""
    th = theme.theme("light")
    cards = _cards()
    grids = {(p, s): model_cards.calibration_panel(cards["calibration"], "synthetic", p, s)
             for p in model_cards.PANELS for s in model_cards.SPLITS}
    wild = pd.DataFrame([{"head": "synthetic", "split": "train", "row": 0,
                          "fitted": 900.0, "observed": 900.0, "residual": 0.0}])
    points = {(p, s): wild for p in model_cards.PANELS for s in model_cards.SPLITS}
    fig = charts.fig_calibration(grids, points, th, model_cards.PANEL_AXES,
                                 model_cards.PANEL_LABELS)
    axes = [fig.layout[f"xaxis{'' if i == 1 else i}"].range for i in range(1, 5)]
    assert axes[0] == axes[1] and axes[2] == axes[3]       # train against validation
    assert max(axes[0]) < 10                               # the 900 point did not set it


def test_every_model_figure_carries_an_explicit_title_and_the_pinned_surface():
    """A title object with a font and no text renders as the literal string "undefined"."""
    for mode in theme.THEMES:
        th = theme.theme(mode)
        for fig in _model_figures(th):
            assert fig.layout.title.text is not None
            assert fig.layout.paper_bgcolor == th["surface"]


# ── The shipped artifacts ─────────────────────────────────────────────────────

def _card(name: str) -> pd.DataFrame:
    path = PREDICTIONS / name
    if not path.exists():
        pytest.skip(f"{path} is missing; run `{model_cards.MAKE_CARDS}`")
    return (pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path))


def test_every_carded_head_belongs_to_exactly_one_model_page():
    """A twenty-first head should fail a test rather than vanish from the navigation."""
    index = _card(model_cards.INDEX_FILE)
    declared = [h for spec in model_cards.CLASSES.values() for h in spec.heads]
    assert len(declared) == len(set(declared)), "a head is on two pages"
    assert set(declared) == set(index["head"])
    for key, spec in model_cards.CLASSES.items():
        classes = set(index[index["head"].isin(spec.heads)]["model_class"])
        assert classes == {key}, (key, classes)


def test_every_heads_diagnostics_row_exists_in_the_table_it_names():
    """The label is derived per class from each head's own variant, so a refit at a
    different arm is the failure mode — and block 7 would render empty rather than wrong."""
    index = _card(model_cards.INDEX_FILE)
    tables = {}
    for _, row in index.iterrows():
        filename, label = model_cards.diagnostics_source(row)
        path = PREDICTIONS / filename
        if not path.exists():
            pytest.skip(f"{path} is missing; run `{model_cards.MAKE_STAN}`")
        table = tables.setdefault(filename, pd.read_csv(path))
        assert label in set(table["label"]), (row["head"], filename, label)


def test_every_shipped_pair_menu_has_its_density_on_disk():
    index = _card(model_cards.INDEX_FILE)
    correlations = _card(model_cards.CORRELATION_FILE)
    density = _card(model_cards.DENSITY_FILE)
    for _, row in index.iterrows():
        menu = model_cards.pair_menu(correlations, row["head"])
        assert len(menu) == int(row["n_density_pairs"]), row["head"]
        drawn = density[density["head"] == row["head"]]
        assert set(zip(menu["feature_x"], menu["feature_y"])) == \
            set(zip(drawn["feature_x"], drawn["feature_y"])), row["head"]


def test_the_shipped_index_lets_every_availability_head_render_all_seven_blocks():
    """One end-to-end pass over the real artifacts, block by block, without a runtime."""
    cards = {key: _card(name) for key, name in (
        ("index", model_cards.INDEX_FILE),
        ("coefficients", model_cards.COEFFICIENTS_FILE),
        ("features", model_cards.FEATURES_FILE),
        ("correlations", model_cards.CORRELATION_FILE),
        ("density", model_cards.DENSITY_FILE),
        ("ecdf", model_cards.ECDF_FILE),
        ("calibration", model_cards.CALIBRATION_FILE),
        ("sample", model_cards.SAMPLE_FILE))}
    th = theme.theme("light")
    for head in model_cards.heads_of(cards["index"], "availability"):
        row = model_cards.head_row(cards["index"], head)
        assert str(row["unit"]).strip() and str(row["response_label"]).strip()
        assert len(model_cards.specification(row)) >= 10
        order = model_cards.feature_order(cards["features"], head)
        assert len(order) == int(row["n_features"])
        charts.fig_features(model_cards.histogram_panel(cards["features"], head, order), th)
        charts.fig_correlation(
            model_cards.correlation_square(cards["correlations"], head, "train"), th)
        pair = model_cards.pair_menu(cards["correlations"], head).iloc[0]
        charts.fig_joint(
            model_cards.density_panel(cards["density"], head, pair["feature_x"],
                                      pair["feature_y"], "train"), th, "x", "y")
        panel = model_cards.coefficient_panel(cards["coefficients"], head)
        assert len(panel) == int(row["n_features"])
        charts.fig_coefficients(panel, th)
        distance = model_cards.band_distance(cards["ecdf"], head)
        assert len(distance) == 2 and (distance["max_gap"] < 0.5).all()
        assert len(model_cards.calibration_summary(cards["calibration"], head)) == 4
        assert len(model_cards.build_checks(row)) == 4
