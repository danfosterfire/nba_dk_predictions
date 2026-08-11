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
import tomllib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import (audit, charts, decisions, economics, inputs, model_cards,
                       overview, pca, strategy, theme, weekly)

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"
PREDICTIONS = ROOT / "outputs" / "predictions"
EDA = ROOT / "outputs" / "eda"


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


# ── The page chrome, which is the same palette ────────────────────────────────
# `.streamlit/config.toml` is generated by `make dashboard-config` from `theme.THEMES`.
# These pin the generator, the checked-in file, and the one line in it that has nothing
# to do with colour and breaks `make dashboard` outright when it goes missing.

def _config() -> dict:
    return tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())


def test_the_checked_in_config_is_exactly_what_the_palette_generates():
    """`make dashboard-config` is a no-op on a clean tree, or the chrome has drifted."""
    assert _config() == tomllib.loads(theme.config_toml())


def test_both_theme_tables_are_the_palette_rather_than_a_second_copy_of_it():
    cfg = _config()["theme"]
    for mode in theme.THEMES:
        assert cfg[mode] == theme.streamlit_theme(mode), mode


def test_every_chrome_colour_is_a_value_the_palette_already_defines():
    """A hand-typed hex here would put `dashboard/README.md`'s contrast figures out of
    date silently, since nothing would ever compare the two again."""
    for mode, palette in theme.THEMES.items():
        known = set(palette["series"]) | {v for v in palette.values()
                                          if isinstance(v, str)} | {theme.FONT}
        table = theme.streamlit_theme(mode)
        flat = {**{k: v for k, v in table.items() if isinstance(v, str)},
                **table["sidebar"]}
        for key, value in flat.items():
            assert value in known, (mode, key, value)


def test_the_page_ground_is_the_surface_the_figures_are_pinned_to():
    """The point of the whole step: a chart has no visible edge against its page."""
    for mode in theme.THEMES:
        painted = theme.apply_theme(go.Figure(), theme.theme(mode)).layout.paper_bgcolor
        assert theme.streamlit_theme(mode)["backgroundColor"] == painted


def test_the_sidebar_inverts_the_pair_so_a_widget_reads_as_raised_in_both_modes():
    for mode, palette in theme.THEMES.items():
        table = theme.streamlit_theme(mode)
        assert table["backgroundColor"] == palette["surface"]
        assert table["sidebar"]["backgroundColor"] == palette["plane"]
        assert table["sidebar"]["secondaryBackgroundColor"] == palette["surface"]


def test_the_diverging_midpoint_never_becomes_furniture():
    """`neutral` is a data colour — the midpoint of the diverging scale — so chrome
    borrowing it would make a reader read a zero value as a panel.

    Asserted on the roles rather than the colours, because in dark mode `neutral` and
    `axis` are the same hex and no comparison of values could tell them apart.
    """
    used = set(theme.CHROME_ROLES.values()) | set(theme.SIDEBAR_ROLES.values())
    assert "neutral" not in used
    assert used <= set(theme.THEMES["light"]) | {"accent", "font"}


def test_the_two_modes_declare_the_same_settings():
    light, dark = (theme.streamlit_theme(m) for m in ("light", "dark"))
    assert light.keys() == dark.keys()
    assert light["sidebar"].keys() == dark["sidebar"].keys()
    assert light != dark


def test_headless_survives_the_generated_config():
    """Without it Streamlit's first run stops at an interactive prompt and exits 255,
    which is a broken `make dashboard` rather than a wrong colour."""
    assert _config()["server"]["headless"] is True
    assert tomllib.loads(theme.config_toml())["server"]["headless"] is True


def test_the_config_sets_no_top_level_chart_colours():
    """They exist once for both modes while `SERIES` is selected per mode, so setting
    them would push one mode's eight slots onto the other."""
    top = _config()["theme"]
    assert not [k for k in top if k.startswith("chart")]


def test_the_font_stack_survives_a_toml_round_trip():
    """It carries its own double quotes, so it is written as a TOML literal string."""
    assert '"Segoe UI"' in theme.FONT
    assert _config()["theme"]["light"]["font"] == theme.FONT


# ── The one appearance control ────────────────────────────────────────────────

def test_the_appearance_is_streamlits_own_setting_and_nothing_else(monkeypatch):
    """The sidebar radio was retired on 2026-08-10 — see `shell.detected_mode()`.

    Two controls could disagree, and after the config half the disagreement got *more*
    visible rather than less. What settled it is a capability: `st.dataframe` renders to
    a canvas, so no CSS a page injects can repaint a table, while config can — and every
    model page puts a table twin beside every chart.
    """
    from dashboard import shell
    assert not hasattr(shell, "appearance_control")
    assert not hasattr(shell, "APPEARANCE")
    monkeypatch.setattr(shell, "detected_mode", lambda: "dark")
    assert shell.mode() == "dark"
    assert shell.current_theme() == theme.theme("dark")


def test_no_module_writes_an_appearance_key_into_session_state():
    """A leftover writer would put a second owner back on the mode without a widget."""
    for path in sorted((ROOT / "dashboard").rglob("*.py")):
        assert "appearance_control" not in path.read_text(), path.name


def test_the_entrypoint_renders_no_shell_control_before_the_page():
    """`app.main()` calling a control that no longer exists is an AttributeError in a
    browser and nowhere else, since nothing in the suite runs the entrypoint."""
    from dashboard import app
    src = (ROOT / "dashboard" / "app.py").read_text()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "shell"]
    assert {n.func.attr for n in calls} == {"publish_pages"}
    assert callable(app.main)


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
#
# **Keyed by path rather than by basename, since 2026-08-10.** The room joined the
# navigation as page 9, which gave it a row-owning sibling at `views/draft_room.py` — and
# under the old basename match that sibling would have inherited the exemption for free,
# widening a deliberately narrow hole by the act of naming a file. The wrapper is held to
# the ordinary rule; only the path below is exempt.
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
        if path.relative_to(ROOT / "dashboard").as_posix() in SRC_IMPORTERS:
            continue
        offenders += [f"{path.relative_to(ROOT)}: {name}" for name in _src_imports(path)]
    assert offenders == []


def test_the_exemption_is_a_path_so_a_second_file_cannot_inherit_it_by_name():
    """The room's navigation row lives in a file with the same basename as the room.

    `views/draft_room.py` owns page 9's row and `draft_room.py` is the page; under a
    basename match the wrapper would have been exempt the moment it was created, which is
    a widened hole nobody would have had to argue for. Both files exist, exactly one is
    exempt, and the key that exempts it names a directory.
    """
    assert list(SRC_IMPORTERS) == ["draft_room.py"]
    assert (ROOT / "dashboard" / "views" / "draft_room.py").exists()
    assert _src_imports(ROOT / "dashboard" / "views" / "draft_room.py") == []


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


# ── The draft room, which is both a page and its own app ──────────────────────
#
# Page 9 joined the navigation on 2026-08-10 and `make draft-room` still launches the same
# file directly, because draft night is a thirty-second clock and should not share a
# process with anything. Everything below is one of the two conditions that made the move
# allowed — see `docs/dashboard-plan.md`, "Page 9".

def test_only_the_standalone_entrypoint_sets_the_page_config():
    """`st.set_page_config` may be called once per process, so `render()` must not.

    Both launches reach the same body: `main()` sets the config and calls `render()`;
    `app.py` sets its own and calls `render()` through the view wrapper. A
    `set_page_config` left in the shared half raises `StreamlitAPIException` in the app —
    on the page, at navigation time, which is the one place no unit test looks.
    """
    tree = ast.parse((ROOT / "dashboard" / "draft_room.py").read_text())
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    assert {"render", "main"} <= set(functions)
    setters = {name for name, node in functions.items()
               for call in ast.walk(node)
               if isinstance(call, ast.Call)
               and getattr(call.func, "attr", "") == "set_page_config"}
    assert setters == {"main"}
    assert "render" in {call.func.id for call in ast.walk(functions["main"])
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}


def test_make_draft_room_still_launches_the_page_file_itself():
    """The standalone launch is a condition of the move, not a leftover convenience."""
    assert "streamlit run dashboard/draft_room.py" in (ROOT / "Makefile").read_text()


def test_the_draft_board_row_hands_off_to_the_page_module():
    """The row is owned by a `views/` module; the page it draws lives beside the package.

    Every other row's `render` *is* the page. This one delegates, which is what lets one
    file be both a page and an app — so the row is pinned here rather than left to the
    generic navigation tests, which cannot tell a delegation from a stub.
    """
    from dashboard import app
    rows = [v for v in app.VIEWS if v.url_path == "draft-room"]
    assert len(rows) == 1 and rows[0].title == "Draft board"
    assert rows[0].render.__module__ == "dashboard.views.draft_room"


def test_the_view_wrapper_defers_the_simulation_import_until_the_page_is_opened():
    """A reader who never opens the room never loads `src.sim`.

    `app.py` imports every view module before it draws anything, so a module-level import
    here would put `import src.sim.draft_room` — 0.89 s — on the startup path of every
    page. Run in a subprocess because this suite imports the simulation layer elsewhere,
    and `sys.modules` is per-process.
    """
    import subprocess
    import sys
    probe = ("import sys; import dashboard.views.draft_room as page; "
             "assert callable(page.render); "
             "assert not [m for m in sys.modules if m.startswith('src.')], "
             "sorted(m for m in sys.modules if m.startswith('src.'))")
    subprocess.run([sys.executable, "-c", probe], cwd=ROOT, check=True)


def test_a_remembered_control_survives_the_page_being_left(monkeypatch):
    """`shell.recall` / `shell.remember` are plain keys, which navigation does not clear.

    Streamlit clears widget state for a page the reader has left. The draft room's pick
    log is a plain key and *does* survive, so a control that resets while the log persists
    is worse than either — it replays a real pod against the wrong seat.
    """
    from dashboard import shell
    monkeypatch.setattr(shell.st, "session_state", {})
    assert shell.recall("seat", 1) == 1
    assert shell.remember("seat", 7) == 7
    assert shell.recall("seat", 1) == 7
    # Namespaced, so a shadow key and a widget key of the same name stay separate owners.
    assert list(shell.st.session_state) == ["remembered:seat"]


def test_a_remembered_choice_that_is_no_longer_on_offer_falls_back(monkeypatch):
    """A season whose tensor has gone must not take the room down with it."""
    from dashboard import draft_room as page
    from dashboard import shell
    seen = {}

    def selectbox(label, options, index=0, **kwargs):
        seen["index"] = index
        return options[index]

    monkeypatch.setattr(shell.st, "session_state", {})
    monkeypatch.setattr(page.st, "selectbox", selectbox)
    seasons = ["2022-23", "2023-24"]

    assert page.choice("Season board", seasons, "season", len(seasons) - 1) == "2023-24"
    assert seen["index"] == 1
    shell.remember("season", "2022-23")
    assert page.choice("Season board", seasons, "season", len(seasons) - 1) == "2022-23"
    shell.remember("season", "1996-97")
    assert page.choice("Season board", seasons, "season", len(seasons) - 1) == "2023-24"


# The Streamlit surface, exhaustively. Everything else in the package stays testable
# without a runtime, and this is written as a denylist over the whole tree rather than an
# allowlist of pure modules so that a new file is pure *by default* — the expansion in
# `docs/dashboard-plan.md` adds eight more pages, and the failure mode worth guarding is a
# view's logic being written into the view instead of into a pure sibling.
STREAMLIT_SURFACE = {"app.py", "shell.py", "artifacts.py", "draft_room.py"}
VIEWS_DIR = "views"


def _streamlit_imports(path) -> list[str]:
    """Imports of Streamlit in `path`, by AST rather than by grepping for the word.

    A substring match was the original guard and it stopped being right on 2026-08-10:
    `theme.py` renders `.streamlit/config.toml` from the palette, so it *names* Streamlit
    in a path, a function name and four paragraphs of comment while importing nothing.
    The rule the guard exists for is that the pure layer must be callable without a
    runtime, and that is an import question — which is also what
    `test_the_dashboard_imports_nothing_from_src` next door already asks.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        found += [n for n in names if n == "streamlit" or n.startswith("streamlit.")]
    return found


def test_pure_modules_do_not_import_streamlit():
    """Only the entrypoint, the shell, the I/O layer and the pages touch Streamlit."""
    for path in sorted((ROOT / "dashboard").rglob("*.py")):
        if path.name in STREAMLIT_SURFACE or path.parent.name == VIEWS_DIR:
            continue
        assert _streamlit_imports(path) == [], \
            f"{path.relative_to(ROOT)} must stay Streamlit-free"


def test_the_pure_layer_is_importable_without_a_streamlit_runtime():
    """The guard's actual promise, asserted rather than inferred from an import list."""
    import subprocess
    import sys
    pure = [p.stem for p in sorted((ROOT / "dashboard").glob("*.py"))
            if p.name not in STREAMLIT_SURFACE]
    code = ("import sys; sys.modules['streamlit'] = None\n"
            "import " + ", ".join(f"dashboard.{m}" for m in pure))
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


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
    """`pages()` makes index 0 the default, so `/` must not serve a placeholder.

    The fingerprint view held index 0 from the shell landing until the Overview took it on
    2026-08-10, which is the whole point of that page: a reader arriving at the bare URL
    with no context gets the one page written for them rather than a radial chart of a
    player-season they did not choose.
    """
    from dashboard import app
    from dashboard.views import overview as overview_page
    assert app.VIEWS[0].render is overview_page.render
    assert app.VIEWS[0].url_path == "overview"


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
        chain_role="games_played_count",
        chain_role_label="draws the games-played count", in_draw_path=True,
        chain_role_note="A note about the draw.",
        coefficient_scale="standardized", recipe_design_error=0.0,
        roundtrip_prediction_error=0.0, design_check="vacuous", verified=True,
        response_label="games played", predictive_draws=200, n_predictive_train=900,
        n_predictive_validation=100, predictive_rows_capped=False,
        predictive_weighted=False, fitted_source="head_predict",
        predictive_check="mean", predictive_bias=0.002, ecdf_band_mc=0.01,
        ecdf_band_gated=True, player_season_sigma=0.0,
        quantile_scope="drawn", quantile_reason="", quantile_ks_train=0.02,
        quantile_ks_validation=0.06, quantile_ks_mc=0.004, quantile_ks_gated=True,
        quantile_weighting="unweighted", git_sha="abcdef123456",
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
    for panel in model_cards.PANELS:
        for split, n in (("train", 900), ("validation", 100)):
            for i, (count, y) in enumerate([(90, 0.0), (10, 10.0)]):
                rows.append({
                    "head": head, "split": split, "panel": panel, "x_index": i,
                    "y_index": i, "x_left": float(i), "x_right": float(i + 1),
                    "y_left": y, "y_right": y + 1.0, "count": count,
                    "density": count / 100.0, "n": n})
    return pd.DataFrame(rows)


def _quantile(head: str = "synthetic", bins: int = 4, drift: float = 0.0) -> pd.DataFrame:
    """The three panels of one head's scaled residual, shaped like the artifact.

    `drift` tilts the quantile lines off their own levels, which is what a head that is
    calibrated on average and wrong at one end of its own fit looks like here.
    """
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for split, n, ks in (("train", 900, 0.02), ("validation", 100, 0.06)):
        stamp = {"head": head, "split": split, "ks": ks, "n": n}
        for k in range(bins):
            expected = (k + 1) / (bins + 1)
            rows.append({**stamp, "panel": "qq", "x_index": k, "y_index": -1,
                         "x": expected, "y": min(1.0, expected + 0.01),
                         "lo": max(0.0, expected - 0.05), "hi": min(1.0, expected + 0.05)})
        for i in range(bins):
            for j in range(bins):
                rows.append({**stamp, "panel": "residual", "x_index": i, "y_index": j,
                             "x": float((edges[i] + edges[i + 1]) / 2),
                             "y": float((edges[j] + edges[j + 1]) / 2),
                             "x_left": edges[i], "x_right": edges[i + 1],
                             "y_left": edges[j], "y_right": edges[j + 1],
                             "count": 10 + i + j,
                             "density": (10 + i + j) / (bins * bins * 12.0)})
        for i in range(bins):
            for level in model_cards.QUANTILE_LEVELS:
                rows.append({**stamp, "panel": "quantile", "x_index": i, "y_index": -1,
                             "x": float((edges[i] + edges[i + 1]) / 2),
                             "y": level + drift * i, "level": level,
                             "x_left": edges[i], "x_right": edges[i + 1],
                             "count": n // bins})
    return pd.DataFrame(rows)


def _sample(head: str = "synthetic") -> pd.DataFrame:
    return pd.DataFrame([
        {"head": head, "split": split, "row": i, "fitted": 1.0 + i,
         "observed": 2.0 + i, "u": 0.1 * (i + 1), "predicted_rank": 0.1 * (i + 2)}
        for split in ("train", "validation") for i in range(5)])


def _cards() -> dict:
    return {"index": _index(), "coefficients": _coefficients(),
            "features": _feature_rows(), "correlations": _correlations(),
            "density": _density(), "ecdf": _ecdf(), "calibration": _calibration(),
            "quantile": _quantile(), "sample": _sample()}


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


def test_the_specification_states_what_the_shipped_chain_does_with_the_head():
    """Beside the unit, and read from the same artifact for the same reason.

    A head being fitted, converged and carded says nothing about whether the simulator
    calls it — four of the twenty are never read at draw time — and a view that decided
    that for itself would go stale on the next refactor of `src/sim/season.py`.
    """
    spec = model_cards.specification(_index().iloc[0])
    fields = list(spec["Field"])
    # Literally beside it: the chain role is the row after the unit.
    assert fields.index("In the shipped chain") == fields.index("Unit") + 1
    assert dict(zip(fields, spec["Value"]))["In the shipped chain"] == \
        "draws the games-played count"


def test_the_chain_role_sentence_takes_its_auxiliary_from_the_draw_path_flag():
    """`in_draw_path` ships beside the label rather than being inferred from it, and the
    page uses it: a head that is read reads "it <verb>s", one that is not reads "it is
    not called at draw time". A view that lost the boolean would print the second as the
    first and say the opposite of the truth."""
    drawn = model_cards.chain_role_phrase(_index().iloc[0])
    assert drawn.startswith("**In a simulated season it draws the games-played count.**")
    idle = model_cards.chain_role_phrase(_index(
        head="gp_onset", chain_role="not_at_draw_time", in_draw_path=False,
        chain_role_label="not called at draw time",
        chain_role_note="It supplies a bar instead.").iloc[0])
    assert idle.startswith("**In a simulated season it is not called at draw time.**")
    assert "It supplies a bar instead." in idle


def test_a_page_built_before_the_chain_role_existed_loses_a_caption_not_prints_nan():
    """The index is regenerated by `make model-cards`, but a stale one on disk must
    degrade the way every other absent field does — a dash in the table and no sentence,
    never the four letters `nan`."""
    row = _index().iloc[0].drop(["chain_role_label", "in_draw_path", "chain_role_note"])
    assert model_cards.chain_role_phrase(row) == ""
    spec = model_cards.specification(row)
    assert dict(zip(spec["Field"], spec["Value"]))["In the shipped chain"] == "—"
    assert "nan" not in " ".join(spec["Value"])


def test_an_absent_field_reads_as_a_dash_rather_than_as_the_string_nan():
    """A missing CSV cell reads back as a float `nan` whose `str()` is four letters that
    print on the page and look like a value. The game-length depth head is fitted on depth
    cells and has no season span at all, which is the first row where one reached a
    caption — the same class of defect as a plotly title rendering as `undefined`."""
    row = _index(head="game_length_depth", first_season=float("nan"),
                 last_season=float("nan"), dispersion="").iloc[0]
    assert model_cards.season_span(row) == "—"
    spec = dict(zip(model_cards.specification(row)["Field"],
                    model_cards.specification(row)["Value"]))
    assert spec["Fitted seasons"] == "—" and spec["Dispersion"] == "—"
    assert "nan" not in " ".join(model_cards.specification(row)["Value"])
    # A head that does have seasons still prints them, and a one-sided span still reads.
    assert model_cards.season_span(_index().iloc[0]) == "1997-98 → 2021-22"
    assert model_cards.text(float("nan")) == "—" and model_cards.text("base") == "base"


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


def test_the_imputed_shares_line_survives_a_column_name_with_a_space_in_it():
    """`itertuples` renames any column that is not an identifier, so `Imputed share`
    arrives as `_10` and reading it back by name raises — on the heads that imputed
    something and only those, which is why nine heads and a whole page rendered fine."""
    summary = model_cards.feature_summary(_feature_rows(), "synthetic")
    assert model_cards.imputed_shares(summary) == "`log_x__s0` 10.00%, `log_x__s1` 10.00%"
    clean = summary.assign(**{"Imputed share": 0.0})
    assert model_cards.imputed_shares(clean) == ""


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
    # One subsample, four coordinates: two panels read the same rows, so a point in the
    # calibration density and a point in the residual panel are the same player-season.
    assert {"fitted", "observed", "u", "predicted_rank"} <= set(points.columns)


# ── Block 6 · the scaled quantile residual ────────────────────────────────────

def test_the_qq_panel_is_named_rather_than_handed_over_as_x_and_y():
    """A figure builder taking `x`/`y`/`lo`/`hi` cannot say which is which, and the axis
    titles are the entire content of a QQ plot."""
    panel = model_cards.qq_panel(_quantile(), "synthetic", "train")
    assert list(panel.columns) == ["expected", "observed", "lo", "hi", "n"]
    assert panel["expected"].is_monotonic_increasing
    assert (panel["lo"] <= panel["hi"]).all()
    assert model_cards.qq_panel(_quantile(), "synthetic", "test").empty


def test_the_residual_cells_carry_the_centres_the_heatmap_builder_expects():
    """Same frame shape `calibration_panel` returns, so one builder draws both densities."""
    cells = model_cards.residual_cells(_quantile(), "synthetic", "validation")
    assert np.allclose(cells["x_center"], (cells["x_left"] + cells["x_right"]) / 2)
    assert np.allclose(cells["y_center"], (cells["y_left"] + cells["y_right"]) / 2)
    assert set(cells["split"]) == {"validation"}


def test_the_ks_distance_is_the_reading_and_the_line_gap_says_where():
    """A head can sit close to uniform overall and drift across its own predicted range,
    which is exactly what one KS cannot see and the rank-transformed panel exists for."""
    flat = model_cards.quantile_distance(_quantile(), "synthetic")
    assert list(flat["split"]) == ["train", "validation"]
    assert flat.iloc[0]["ks"] == pytest.approx(0.02)
    assert flat.iloc[0]["line_gap"] == pytest.approx(0.0)
    assert flat.iloc[0]["n"] == 900

    drifting = model_cards.quantile_distance(_quantile(drift=0.05), "synthetic")
    assert drifting.iloc[0]["ks"] == pytest.approx(0.02)      # unchanged
    assert drifting.iloc[0]["line_gap"] == pytest.approx(0.15)


def test_a_head_out_of_scope_says_why_rather_than_rendering_an_empty_panel():
    """`make model-cards` declares it; the page prints the reason. An absent panel with no
    reason beside it is the same defect the emitter's own rule exists to prevent."""
    assert model_cards.quantile_note(_index().iloc[0]) == ""
    out = _index(quantile_scope="not_applicable",
                 quantile_reason="its response is a linear predictor").iloc[0]
    assert model_cards.quantile_note(out) == "its response is a linear predictor"
    assert model_cards.quantile_note(
        _index(quantile_scope="not_applicable", quantile_reason=float("nan")).iloc[0]
    ) == "no reason recorded"


def test_the_dashboards_quoted_budget_bar_is_the_emitters_own():
    """The page prints the bar in a caption; a second hand-typed constant would drift."""
    from src.models import model_cards as emitter

    assert model_cards.KS_MC_TOL == emitter.KS_MC_TOL
    assert model_cards.QUANTILE_LEVELS == emitter.QUANTILE_LEVELS
    assert (model_cards.QQ_PANEL, model_cards.RESIDUAL_PANEL,
            model_cards.LINE_PANEL) == emitter.QUANTILE_PANELS


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
    qq = {model_cards.SPLIT_LABELS[s]: model_cards.qq_panel(cards["quantile"],
                                                            "synthetic", s)
          for s in model_cards.SPLITS}
    residual = {model_cards.SPLIT_LABELS[s]: model_cards.residual_cells(
        cards["quantile"], "synthetic", s) for s in model_cards.SPLITS}
    lines = {model_cards.SPLIT_LABELS[s]: model_cards.quantile_lines(
        cards["quantile"], "synthetic", s) for s in model_cards.SPLITS}
    overlay = {model_cards.SPLIT_LABELS[s]: model_cards.sample_points(
        cards["sample"], "synthetic", s) for s in model_cards.SPLITS}
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
        charts.fig_qq(qq, th, title="qq"),
        charts.fig_quantile_residual(residual, lines, overlay, th,
                                     model_cards.QUANTILE_LEVELS, title="residual"),
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


def test_the_calibration_panels_share_a_colourbar_only_because_each_is_relative():
    """Panels on absolute scales under one legend would label the others wrongly — a
    100-row validation panel puts far more share in a cell than a 900-row one."""
    th = theme.theme("light")
    cards = _cards()
    grids = {(p, s): model_cards.calibration_panel(cards["calibration"], "synthetic", p, s)
             for p in model_cards.PANELS for s in model_cards.SPLITS}
    points = {(p, s): model_cards.sample_points(cards["sample"], "synthetic", s)
              for p in model_cards.PANELS for s in model_cards.SPLITS}
    fig = charts.fig_calibration(grids, points, th, model_cards.PANEL_AXES,
                                 model_cards.PANEL_LABELS)
    heatmaps = [t for t in fig.data if t.type == "heatmap"]
    assert len(heatmaps) == len(model_cards.PANELS) * len(model_cards.SPLITS)
    assert sum(t.showscale for t in heatmaps) == 1
    for heat in heatmaps:
        assert np.nanmax(np.asarray(heat.z, dtype=float)) == pytest.approx(1.0)
        assert np.nanmax(np.asarray(heat.customdata, dtype=float)) < 1.0   # the raw share


def test_the_calibration_grid_is_one_row_per_panel_rather_than_a_fixed_two_by_two():
    """It drew four panels until the residual half moved to the quantile figure; a
    hard-coded 2 x 2 would leave two empty cells and a figure twice as tall as its content,
    which `AppTest` counts identically either way."""
    th = theme.theme("light")
    cards = _cards()
    grids = {(p, s): model_cards.calibration_panel(cards["calibration"], "synthetic", p, s)
             for p in model_cards.PANELS for s in model_cards.SPLITS}
    fig = charts.fig_calibration(grids, {}, th, model_cards.PANEL_AXES,
                                 model_cards.PANEL_LABELS)
    panels = len(model_cards.PANELS) * len(model_cards.SPLITS)
    assert len(fig.layout.annotations) == panels
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == panels


def test_both_splits_of_a_calibration_panel_are_drawn_on_one_axis_range():
    """The emitter clips the density's tails into its end bins; an unclipped overlay
    setting the axis undoes that, which squashed a 2-to-9-game density into a sliver."""
    th = theme.theme("light")
    cards = _cards()
    grids = {(p, s): model_cards.calibration_panel(cards["calibration"], "synthetic", p, s)
             for p in model_cards.PANELS for s in model_cards.SPLITS}
    wild = pd.DataFrame([{"head": "synthetic", "split": "train", "row": 0,
                          "fitted": 900.0, "observed": 900.0, "u": 0.5,
                          "predicted_rank": 0.5}])
    points = {(p, s): wild for p in model_cards.PANELS for s in model_cards.SPLITS}
    fig = charts.fig_calibration(grids, points, th, model_cards.PANEL_AXES,
                                 model_cards.PANEL_LABELS)
    axes = [fig.layout[f"xaxis{'' if i == 1 else i}"].range
            for i in range(1, len(model_cards.PANELS) * len(model_cards.SPLITS) + 1)]
    assert axes[0] == axes[1]                              # train against validation
    assert max(axes[0]) < 10                               # the 900 point did not set it


def test_the_qq_panel_draws_the_envelope_the_diagonal_and_the_points_in_that_order():
    """The diagonal is the claim, so it cannot be inferred from the points — and the
    envelope is background rather than a third data series."""
    th = theme.theme("light")
    panels = {model_cards.SPLIT_LABELS[s]: model_cards.qq_panel(_quantile(), "synthetic", s)
              for s in model_cards.SPLITS}
    fig = charts.fig_qq(panels, th)
    names = [t.name for t in fig.data]

    assert names.count("95% pointwise envelope") == 2
    assert names.count("uniform") == names.count("observed residual") == 2
    assert sum(t.showlegend for t in fig.data) == 3        # one legend for both subplots
    # Every trace names its mode: plotly infers `lines+markers` under 20 points and takes
    # the marker colour from its own colorway, which is in no palette this project
    # validated — and a two-point QQ is exactly what `game_length_ot` ships.
    assert all(t.mode is not None for t in fig.data)
    observed = [t for t in fig.data if t.name == "observed residual"]
    assert all(t.mode == "markers" and t.marker.color == th["ink"] for t in observed)
    # Both axes are the unit square, pinned rather than scaled to the data, so two heads'
    # QQ panels mean the same thing.
    assert fig.layout.yaxis.range == (-0.02, 1.02)


def test_the_residual_panel_draws_its_quantile_lines_against_their_own_levels():
    """"Flat at 0.25 / 0.5 / 0.75" has to be readable off the panel, not asserted in a
    caption — so each empirical line has a dashed reference at its own level."""
    th = theme.theme("light")
    quantile = _quantile()
    cells = {model_cards.SPLIT_LABELS[s]: model_cards.residual_cells(quantile,
                                                                     "synthetic", s)
             for s in model_cards.SPLITS}
    lines = {model_cards.SPLIT_LABELS[s]: model_cards.quantile_lines(quantile,
                                                                     "synthetic", s)
             for s in model_cards.SPLITS}
    points = {model_cards.SPLIT_LABELS[s]: model_cards.sample_points(_sample(),
                                                                     "synthetic", s)
              for s in model_cards.SPLITS}
    fig = charts.fig_quantile_residual(cells, lines, points, th,
                                       model_cards.QUANTILE_LEVELS)

    empirical = [t for t in fig.data if t.name == "empirical quantile"]
    expected = [t for t in fig.data if t.name == "expected level"]
    assert len(empirical) == len(expected) == 2 * len(model_cards.QUANTILE_LEVELS)
    assert {tuple(t.y) for t in expected} == {(lv, lv) for lv in
                                              model_cards.QUANTILE_LEVELS}
    # Two colours, not five: `ALL_PAIRS_CAP` is three, and each line's level is legible
    # from the reference under it and from the hover rather than from its colour.
    assert len({t.line.color for t in empirical}) == 1
    assert all(t.line.dash == "dash" for t in expected)
    assert len([t for t in fig.data if t.type == "heatmap"]) == 2
    # One legend entry per series across both subplots, not one per line per panel.
    assert sum(bool(t.showlegend) for t in fig.data) == 2
    assert fig.layout.yaxis.range == (0, 1) and fig.layout.xaxis.range == (0, 1)


def test_a_head_with_no_drawable_quantile_bins_still_builds_both_figures():
    """`game_length_ot`'s two validation cells: a two-point QQ and no quartile lines."""
    th = theme.theme("light")
    tiny = _quantile(bins=2)
    empty = tiny[tiny["panel"] != "quantile"]
    fig = charts.fig_quantile_residual(
        {"Train": model_cards.residual_cells(empty, "synthetic", "train")},
        {"Train": model_cards.quantile_lines(empty, "synthetic", "train")},
        {}, th, model_cards.QUANTILE_LEVELS)
    assert not [t for t in fig.data if t.name == "empirical quantile"]
    assert len([t for t in fig.data if t.type == "heatmap"]) == 1


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


def test_every_shipped_head_renders_a_chain_role_sentence_on_its_page():
    """Every model class gets the column, not just the availability page — which is why it
    is emitted per head rather than added to one page's intro."""
    index = _card(model_cards.INDEX_FILE)
    for _, row in index.iterrows():
        label, note = str(row["chain_role_label"]), str(row["chain_role_note"])
        # No fallback fired anywhere: the dash and the four letters `nan` are what an
        # absent cell renders as, and both look like a value on the page.
        for cell in (label, note):
            assert cell.strip() and cell not in ("—", "nan"), (row["head"], cell)
        verb = "it" if bool(row["in_draw_path"]) else "it is"
        assert model_cards.chain_role_phrase(row) == \
            f"**In a simulated season {verb} {label}.** {note}", row["head"]
        assert dict(zip(model_cards.specification(row)["Field"],
                        model_cards.specification(row)["Value"]))[
            "In the shipped chain"] == label, row["head"]


def test_the_availability_intro_names_every_head_and_the_two_the_chain_takes():
    """The intro read *"Two ways of predicting the same quantity"*, which says two
    alternates where one ships — and the shipped chain takes **one head from each**
    approach. The anchor is the artifact's own `in_draw_path`: if a refactor of
    `src/sim/season.py` changes which of these five heads a season draw reads, the count
    below moves and this intro has to be rewritten rather than quietly going stale."""
    index = _card(model_cards.INDEX_FILE).set_index("head")
    spec = model_cards.CLASSES["availability"]
    heads = model_cards.heads_of(_card(model_cards.INDEX_FILE), "availability")
    drawn = [h for h in heads if bool(index.loc[h, "in_draw_path"])]
    assert drawn == ["availability", "gp_duration"]
    for head in heads:
        assert f"`{head}`" in spec.intro, head
    assert "Two ways of predicting the same quantity" not in spec.intro


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


def _draw_quantile_block(cards: dict, head: str, th: dict) -> pd.DataFrame:
    """Block 6's quantile half over one real head, exactly as the view assembles it.

    Returned so a caller can assert on the distance; called by all three end-to-end passes,
    so every carded head's residual panels are built at least once against the real
    artifact rather than against the synthetic one above.
    """
    qq = {model_cards.SPLIT_LABELS[s]: model_cards.qq_panel(cards["quantile"], head, s)
          for s in model_cards.SPLITS}
    charts.fig_qq({name: part for name, part in qq.items() if not part.empty}, th)
    charts.fig_quantile_residual(
        {model_cards.SPLIT_LABELS[s]: model_cards.residual_cells(cards["quantile"], head, s)
         for s in model_cards.SPLITS},
        {model_cards.SPLIT_LABELS[s]: model_cards.quantile_lines(cards["quantile"], head, s)
         for s in model_cards.SPLITS},
        {model_cards.SPLIT_LABELS[s]: model_cards.sample_points(cards["sample"], head, s)
         for s in model_cards.SPLITS},
        th, model_cards.QUANTILE_LEVELS)
    distance = model_cards.quantile_distance(cards["quantile"], head)
    assert len(distance) == 2, head
    assert distance["ks"].between(0.0, 1.0).all(), head
    return distance


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
        ("quantile", model_cards.QUANTILE_FILE),
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
        assert len(model_cards.calibration_summary(cards["calibration"], head)) == 2
        _draw_quantile_block(cards, head, th)
        assert len(model_cards.build_checks(row)) == 4


# ── The box-score page's own block · against the no-fit floor ─────────────────
#
# Pages 5 and 6 are the generic renderer plus one named block each, so what is new to test
# is that block: the ladder each head was selected from, its floor, and — on page 5 — the
# claim about *what each head is a rate of*, which is an interpretation and therefore
# carries an anchor the way `pca.COMPONENTS` does.

def _component_metrics(head: str = "ftm|fta", floor_r2: float = 0.34,
                       shipped_r2: float = 0.32) -> pd.DataFrame:
    """One head's four-arm ladder, floor first. Defaults to the head that fails its floor."""
    arms = [("carry_forward", 0, floor_r2, True, False),
            ("linear", 9, shipped_r2 - 0.02, shipped_r2 - 0.02 > floor_r2, False),
            ("logit_own", 9, shipped_r2 - 0.01, shipped_r2 - 0.01 > floor_r2, False),
            ("logit_own_spline", 14, shipped_r2, shipped_r2 > floor_r2, True)]
    return pd.DataFrame([{
        "head": head, "kind": "conversion", "variant": variant, "n_features": n,
        "val_r2": r2, "val_mae": 0.06, "val_nll": 3.05, "val_crps": 3.7,
        "val_pit_ks": 0.03, "val_dispersion": 0.01, "selected": selected,
        "beats_floor": beats}
        for variant, n, r2, beats, selected in arms])


def _component_index() -> pd.DataFrame:
    """Two carded component heads — one that clears its floor and one that does not."""
    return pd.concat([
        _index(head="fga", label="fga", model_class="components",
               variant="log_own_spline"),
        _index(head="ftm_given_fta", label="ftm|fta", model_class="components",
               variant="logit_own_spline")], ignore_index=True)


def test_every_component_head_declares_what_its_rate_is_a_rate_of():
    """Eleven heads, three roles, and the page states which — a count is not a share."""
    declared = model_cards.CLASSES["components"].heads
    assert set(declared) == set(model_cards.COMPONENT_BASIS)
    roles = {model_cards.component_basis(h).role for h in declared}
    assert roles == {model_cards.ROLE_COUNT, model_cards.ROLE_SHARE,
                     model_cards.ROLE_CONVERSION}
    # Exactly one share head, and it is not one of the three conversions.
    shares = [h for h in declared
              if model_cards.component_basis(h).role == model_cards.ROLE_SHARE]
    assert shares == [model_cards.SHARE_HEAD]


def test_an_undeclared_head_raises_rather_than_being_described_as_nothing():
    with pytest.raises(KeyError, match="declared basis"):
        model_cards.component_basis("minutes")


def test_the_share_head_is_named_a_share_and_the_shooting_percentage_head_is_not():
    """The whole point of the block: `fg3a|fga` is `fg3a / fga`, and the column that
    carries it is called `logit_fg3a_pct_lag1`, which looks exactly like a 3P% and is not
    one. A reader who takes the name at face value has the head's rationale backwards."""
    share = model_cards.basis_note(model_cards.SHARE_HEAD)
    assert "not a shooting percentage" in share
    assert "fg3a / fga" in share and "fg3m / fg3a" in share
    # Both column names appear, so the two quantities are separable on the page itself.
    assert "logit_fg3a_pct_lag1" in share and "logit_fg3m_pct_lag1" in share
    percentage = model_cards.basis_note(model_cards.SHOOTING_PCT_HEAD)
    assert "conversion" in percentage and "share of attempts" not in percentage


def test_the_ladder_puts_the_floor_first_and_blanks_its_own_beats_floor_flag():
    """`carry_forward`'s `beats_floor` is `True` in the artifact and means nothing — it is
    the floor. Rendered as a tick it would read as a fifth arm that cleared."""
    ladder = model_cards.floor_ladder(_component_metrics(), "ftm|fta")
    assert ladder.iloc[0]["Arm"].startswith(model_cards.COMPONENT_FLOOR)
    assert pd.isna(ladder.iloc[0]["Clears the floor"])
    assert ladder["Shipped"].sum() == 1
    # Below the floor row, the arms run best R² first.
    assert list(ladder["R²"][1:]) == sorted(ladder["R²"][1:], reverse=True)


def test_a_head_with_no_ladder_row_yields_an_empty_frame_rather_than_raising():
    assert model_cards.floor_ladder(_component_metrics(), "reb").empty


def test_the_board_carries_the_margin_over_each_heads_own_floor():
    metrics = pd.concat([_component_metrics("fga", floor_r2=0.95, shipped_r2=0.958),
                         _component_metrics("ftm|fta")], ignore_index=True)
    board = model_cards.floor_board(metrics, _component_index())
    assert list(board["head"]) == ["fga", "ftm_given_fta"]   # declared page order
    margins = dict(zip(board["head"], board["margin"]))
    assert margins["fga"] == pytest.approx(0.008)
    assert margins["ftm_given_fta"] == pytest.approx(-0.02)
    assert list(board["clears"]) == [True, False]


def test_the_board_skips_a_head_whose_ladder_is_not_on_disk_rather_than_raising():
    board = model_cards.floor_board(_component_metrics("ftm|fta"), _component_index())
    assert list(board["head"]) == ["ftm_given_fta"]


def test_the_margin_bars_put_the_head_that_fails_its_floor_across_the_zero_line():
    """Position carries the finding, because colour may not: the palette's relief rule.
    Two more routes ride along — the bar is outlined, and it prints its own value."""
    th = theme.theme("light")
    metrics = pd.concat([_component_metrics("fga", floor_r2=0.95, shipped_r2=0.958),
                         _component_metrics("ftm|fta")], ignore_index=True)
    board = model_cards.floor_board(metrics, _component_index())
    fig = charts.fig_floor_margin(board, th, highlight="fga")
    bar = fig.data[0]
    assert list(bar.y) == [0, 1]
    assert list(fig.layout.yaxis.ticktext) == ["fga", "ftm|fta"]
    assert bar.x[0] > 0 > bar.x[1]                            # sorted, failing head last
    assert fig.layout.xaxis.zeroline is True
    assert list(bar.marker.line.color) == [th["surface"], th["ink"]]
    assert list(bar.text) == ["+0.0080", "-0.0200"]
    assert bar.textposition == "outside"


def test_the_open_head_is_the_only_coloured_bar_on_the_board():
    """Eleven bars is well past `ALL_PAIRS_CAP`, so this is highlight-and-gray and the
    highlight is the head the reader currently has open."""
    th = theme.theme("dark")
    metrics = pd.concat([_component_metrics("fga", floor_r2=0.95, shipped_r2=0.958),
                         _component_metrics("ftm|fta")], ignore_index=True)
    board = model_cards.floor_board(metrics, _component_index())
    fig = charts.fig_floor_margin(board, th, highlight="ftm_given_fta")
    assert list(fig.data[0].marker.color) == [th["muted"], th["series"][0]]


# ── The game length page's own block · in games ───────────────────────────────

def _ppc(variant: str = "season_trend") -> pd.DataFrame:
    rows = []
    for arm, scale in (("floor", 1.06), (variant, 0.98)):
        for class_label, observed in (("regulation", 2322), ("1OT", 120), ("2OT", 18),
                                      ("3OT+", 0)):
            predicted = (2460 - 138 * scale if class_label == "regulation"
                         else observed * scale + 0.5)
            rows.append({"variant": arm, "class": class_label, "observed": observed,
                         "predicted": predicted, "n_games": 2460,
                         "abs_error": abs(predicted - observed)})
    return pd.DataFrame(rows)


def _depth_ladder() -> pd.DataFrame:
    rows = []
    for split, n in (("train", 1861), ("val", 138)):
        for depth, label, observed in ((1, "1OT", 120.0), (2, "2OT", 18.0),
                                       (3, "3OT", 0.0), (4, "4OT+", 0.0)):
            rows.append({"split": split, "depth": depth, "label": label, "n_ot_games": n,
                         "observed": observed, "geometric": observed + 1.0,
                         "beta_geometric": observed + 1.5,
                         "geometric_abs_error": 1.0, "beta_geometric_abs_error": 1.5})
        rows.append({"split": split, "depth": 0, "label": "nats/OT game", "n_ot_games": n,
                     "observed": float("nan"), "geometric": -0.407,
                     "beta_geometric": -0.411, "geometric_abs_error": float("nan"),
                     "beta_geometric_abs_error": float("nan")})
    return pd.DataFrame(rows)


def test_the_onset_block_reads_the_head_at_the_unit_it_is_consumed_at():
    """Its validation ECDF is two grid points — two season cells — so block 5 cannot say
    whether the head is right. Games can, and the ladder already writes them."""
    row = _index(head="game_length_ot", model_class="game_length",
                 variant="season_trend").iloc[0]
    panel = model_cards.class_counts(row, _ppc(), _depth_ladder())
    assert set(panel["series"]) == {"observed", "no-fit floor", "season_trend"}
    assert list(dict.fromkeys(panel["class_label"])) == list(model_cards.ONSET_CLASSES)


def test_regulation_is_carried_in_the_table_and_left_off_the_figure():
    """It is `n_games` minus the other three by construction, for every arm alike, so
    drawing it is a 2,300-long bar that flattens the classes the arms differ on."""
    row = _index(head="game_length_ot", model_class="game_length",
                 variant="season_trend").iloc[0]
    panel = model_cards.class_counts(row, _ppc(), _depth_ladder())
    assert not panel[panel["class_label"] == "regulation"]["drawn"].any()
    assert panel[panel["class_label"] != "regulation"]["drawn"].all()
    assert "regulation" in set(model_cards.class_table(panel)["Class"])


def test_the_depth_block_drops_the_row_that_is_a_likelihood_rather_than_a_depth():
    """`depth == 0` carries nats per overtime game and no observed count; drawn as a class
    it is an empty bar labelled `nats/OT game`."""
    row = _index(head="game_length_depth", model_class="game_length").iloc[0]
    panel = model_cards.class_counts(row, _ppc(), _depth_ladder())
    assert "nats/OT game" not in set(panel["class_label"])
    assert list(dict.fromkeys(panel["class_label"])) == ["1OT", "2OT", "3OT", "4OT+"]
    nats = model_cards.depth_likelihood(_depth_ladder())
    assert float(nats["geometric"]) == pytest.approx(-0.407)
    assert int(nats["n_ot_games"]) == 138


def test_the_class_block_reads_the_validation_split_of_both_ladders():
    """`make stan-game-length` writes `val`, not the cards' `validation` — two files, two
    vocabularies, and reading the training rows here would score the fit on its own data."""
    row = _index(head="game_length_depth", model_class="game_length").iloc[0]
    ladder = _depth_ladder()
    assert model_cards.depth_counts(ladder, "val")["count"].iloc[0] == 120.0
    train_only = ladder[ladder["split"] == "train"]
    assert model_cards.class_counts(row, _ppc(), train_only, "val").empty


def test_the_class_table_twin_reads_in_the_same_order_as_its_figure():
    """`pivot` sorts alphabetically, which put the fitted arm before the floor in the
    table and after it in the legend — two orders for one comparison."""
    row = _index(head="game_length_depth", model_class="game_length").iloc[0]
    panel = model_cards.class_counts(row, _ppc(), _depth_ladder())
    table = model_cards.class_table(panel)
    assert list(table.columns)[1:] == list(dict.fromkeys(panel["series"]))
    assert list(table["Class"]) == list(dict.fromkeys(panel["class_label"]))


def test_the_observed_counts_are_an_outlined_bar_and_the_two_arms_are_filled():
    """Observed is the target, not a third model, so it is a different *mark* as well as a
    different colour — ink is what block 5's ribbon uses for its observed curve. A solid
    ink bar was the first cut and read as the largest quantity on the chart."""
    th = theme.theme("light")
    row = _index(head="game_length_ot", model_class="game_length",
                 variant="season_trend").iloc[0]
    panel = model_cards.class_counts(row, _ppc(), _depth_ladder())
    fig = charts.fig_class_counts(panel[panel["drawn"]], th)
    by_name = {trace.name: trace for trace in fig.data}
    assert by_name["observed"].marker.color == "rgba(0,0,0,0)"
    assert by_name["observed"].marker.line.color == th["ink"]
    assert by_name["no-fit floor"].marker.color == th["series"][0]
    assert by_name["season_trend"].marker.color == th["series"][1]
    assert all(t.marker.line.width == 0 for t in fig.data if t.name != "observed")
    assert len(fig.data) <= theme.ALL_PAIRS_CAP
    assert all(list(trace.text) for trace in fig.data)        # every bar prints its value
    assert fig.layout.barmode == "group"


# ── What the two smallest heads in the project forced on the shared renderer ──
#
# Pages 5 and 6 changed `model_page.py` and `charts.py` rather than forking a renderer, so
# these three pin what changed. All three were caught by rendering the figure, and none of
# them is reachable from the availability page: two need a head with one design column and
# the third needs a grid short enough for plotly to guess at the mark.

def test_a_short_ribbon_declares_its_mark_so_plotly_cannot_pick_its_own_colours():
    """Plotly infers `lines+markers` for a trace of 20 points or fewer, and infers the
    marker colour from its own default colorway — so the onset head's two-point validation
    grid drew stray cyan and red dots that are in no palette this project validated."""
    th = theme.theme("light")
    short = _ecdf().groupby("split", group_keys=False).head(2)
    panels = {model_cards.SPLIT_LABELS[s]: model_cards.ecdf_panel(short, "synthetic", s)
              for s in model_cards.SPLITS}
    fig = charts.fig_ecdf(panels, th, "overtime games in the cell")
    assert all(len(trace.x) <= 4 for trace in fig.data)        # the short-grid regime
    assert all(trace.mode == "lines" for trace in fig.data)
    palette = set(th["series"]) | {th["ink"], charts._translucent(th["series"][0], 0.14),
                                   charts._translucent(th["series"][0], 0.20),
                                   charts._translucent(th["series"][0], 0.26)}
    drawn = {trace.line.color for trace in fig.data if trace.line.color}
    assert drawn <= palette


def test_a_feature_grid_never_lays_out_more_columns_than_it_has_features():
    """One design column in a four-wide grid is one histogram and three empty cells."""
    th = theme.theme("light")
    lone = _feature_rows()
    lone = lone[lone["feature"] == "age"]
    fig = charts.fig_features(model_cards.histogram_panel(lone, "synthetic"), th)
    assert fig.layout.xaxis.domain == (0.0, 1.0)               # full width, not a quarter
    # And a single-row grid leaves the legend room, which a taller one gets for free.
    assert fig.layout.margin.t > 48
    wide = charts.fig_features(model_cards.histogram_panel(_feature_rows(), "synthetic"),
                               th)
    assert wide.layout.xaxis.domain[1] < 0.5                   # four across, as before


# ── A lone design column is not a constant one ────────────────────────────────

def test_a_single_feature_head_reports_no_constant_column():
    """The game-length onset head's whole design is one season term. Its off-diagonal is
    empty by arithmetic, and reporting that as "correlates with nothing" states a fact
    about the column count as though it were a fact about the data."""
    lone = pd.DataFrame([{"head": "game_length_ot", "split": split,
                          "feature_x": "season_start_year",
                          "feature_y": "season_start_year", "i": 0, "j": 0, "r": 1.0,
                          "abs_r": 1.0, "n": 26, "pair_rank": -1, "top_pair": False}
                         for split in ("train", "validation")])
    assert model_cards.correlation_square(lone, "game_length_ot", "train").shape == (1, 1)
    assert model_cards.constant_features(lone, "game_length_ot", "validation") == []
    assert model_cards.pair_menu(lone, "game_length_ot").empty
    # And the real thing a constant column looks like still reports.
    assert model_cards.constant_features(_correlations(), "synthetic", "validation") == \
        ["x__miss"]


# ── The shipped artifacts, for pages 5 and 6 ──────────────────────────────────

def test_every_component_basis_is_anchored_to_a_real_design_family():
    """The interpretation this page makes — "this head is a share, that one is a shooting
    percentage" — is tied to the column the head was actually fed, so a refit that renames
    or drops it fails here rather than mislabelling the page."""
    features = _card(model_cards.FEATURES_FILE)
    for head, basis in model_cards.COMPONENT_BASIS.items():
        families = set(features[features["head"] == head]["term_family"])
        assert basis.own_family in families, (head, basis.own_family, sorted(families))
    # The two that are confusable are on two different heads, which is the whole claim.
    assert (model_cards.COMPONENT_BASIS[model_cards.SHARE_HEAD].own_family
            != model_cards.COMPONENT_BASIS[model_cards.SHOOTING_PCT_HEAD].own_family)


def test_the_shipped_component_ladder_still_has_one_head_below_its_floor():
    """`ftm|fta` failing its floor is quoted in `README.md` and drawn on the page. If a
    refit ever fixes it, the caption that says so has to move with it."""
    metrics = _card(model_cards.COMPONENT_METRICS_FILE)
    board = model_cards.floor_board(metrics, _card(model_cards.INDEX_FILE))
    assert len(board) == len(model_cards.CLASSES["components"].heads)
    failing = list(board[~board["clears"]]["head"])
    assert failing == ["ftm_given_fta"]
    assert board.loc[board["head"] == "ftm_given_fta", "margin"].iloc[0] < 0
    # And every clearing head's margin agrees in sign with the artifact's own flag.
    assert (board[board["clears"]]["margin"] > 0).all()


def test_the_shipped_game_length_ladder_draws_both_heads_at_the_games_unit():
    index = _card(model_cards.INDEX_FILE)
    ppc = _card(model_cards.GAME_LENGTH_PPC_FILE)
    depth = _card(model_cards.GAME_LENGTH_DEPTH_FILE)
    th = theme.theme("light")
    for head in model_cards.heads_of(index, "game_length"):
        row = model_cards.head_row(index, head)
        panel = model_cards.class_counts(row, ppc, depth)
        assert not panel.empty and panel["series"].nunique() == 3
        assert len(model_cards.class_table(panel)) >= 4
        charts.fig_class_counts(panel[panel["drawn"]], th)
    # The onset arm the page draws is the one the card says shipped, not a hard-coded name.
    variant = str(model_cards.head_row(index, "game_length_ot")["variant"])
    assert variant in set(ppc["variant"])


def test_the_shipped_index_lets_every_component_and_game_length_head_render():
    """The same end-to-end pass the availability page gets, over the two new pages —
    including the heads with one design column and with none at all."""
    cards = {key: _card(name) for key, name in (
        ("index", model_cards.INDEX_FILE),
        ("coefficients", model_cards.COEFFICIENTS_FILE),
        ("features", model_cards.FEATURES_FILE),
        ("correlations", model_cards.CORRELATION_FILE),
        ("ecdf", model_cards.ECDF_FILE),
        ("calibration", model_cards.CALIBRATION_FILE),
        ("quantile", model_cards.QUANTILE_FILE),
        ("sample", model_cards.SAMPLE_FILE))}
    th = theme.theme("light")
    widths = {}
    for class_key in ("components", "game_length"):
        for head in model_cards.heads_of(cards["index"], class_key):
            row = model_cards.head_row(cards["index"], head)
            assert str(row["unit"]).strip() and str(row["response_label"]).strip()
            order = model_cards.feature_order(cards["features"], head)
            widths[head] = len(order)
            assert len(order) == int(row["n_features"])
            if order:
                charts.fig_features(
                    model_cards.histogram_panel(cards["features"], head, order), th)
            square = model_cards.correlation_square(cards["correlations"], head, "train")
            if len(square) >= 2:
                charts.fig_correlation(square, th)
            panel = model_cards.coefficient_panel(cards["coefficients"], head)
            if not panel.empty:
                charts.fig_coefficients(panel, th)
            charts.fig_ecdf(
                {model_cards.SPLIT_LABELS[s]: model_cards.ecdf_panel(cards["ecdf"], head, s)
                 for s in model_cards.SPLITS}, th, str(row["response_label"]))
            _draw_quantile_block(cards, head, th)
            assert len(model_cards.build_checks(row)) == 4
    # The two degenerate widths pages 5 and 6 are the first to exercise: one column, and
    # none. Both reach the renderer's own branches rather than a figure builder.
    assert widths["game_length_ot"] == 1 and widths["game_length_depth"] == 0
    # `game_length_ot`'s validation split is two season cells, so its QQ is two points and
    # every bin of its residual panel is under `QUANTILE_MIN_ROWS`. The figure still builds
    # and the lines are simply absent — the case that would otherwise raise on an empty
    # frame or draw a quartile through two rows.
    assert model_cards.quantile_lines(cards["quantile"], "game_length_ot",
                                      "validation").empty


# ── The minutes page's own blocks · one posterior, two units ──────────────────
#
# Page 4 is the first model page whose own blocks are *comparisons* rather than readings of
# the open head, and every way that can go silently wrong is a case below: an arm mapped to
# the wrong head, a gap flipped in one of its two directions but not the other, two CRPS in
# different units drawn on one axis, and a sigma the page types rather than reads.

def _unification() -> pd.DataFrame:
    """A `minutes_unification.csv` shaped like the artifact — every unit it carries.

    The numbers are the shipped ones rounded, because the frame's *shape* is what these
    tests exercise and a reader comparing a case to the page should see the same story:
    the composition loses at the season unit, ties once the effect is injected, and sits on
    the teammate coupling the marginal head misses.
    """
    rows = []
    season = [("minutes_head", 144.35, 200.12, 0.8829, -14.09, 0.0735, 302.75, 742),
              ("composition_sum", 170.06, 200.28, 0.8848, 2.41, 0.3341, 64.65, 742),
              ("carry_forward", 161.29, 213.65, 0.8532, 24.14, 0.1242, 330.84, 742),
              ("composition_sum_all_rows", 149.72, 175.97, 0.9156, 0.0, 0.3575, 55.86,
               1111)]
    for arm, crps, mae, r2, bias, ks, sd, n in season:
        rows.append({"arm": arm, "unit": "season_total", "n": n, "crps_minutes": crps,
                     "mae_minutes": mae, "r2_minutes": r2, "bias_minutes": bias,
                     "pit_ks": ks, "predictive_sd": sd})
    rows.append({"arm": "composition_minus_minutes", "unit": "paired_bootstrap", "n": 742,
                 "crps_delta": 25.70, "ci_lo": 18.96, "ci_hi": 33.25,
                 "crps_delta_shared_target": 25.72, "n_bootstrap": 2000,
                 "sd_ratio_minutes_over_composition": 4.68, "verdict": "loses"})
    rows.append({"arm": "season_effect_headroom", "unit": "variance_decomposition",
                 "n": 1111, "resid_sd": 243.50, "share_of_resid_var_reachable": 0.0})
    sweep = [(0.0, 170.06, 25.70, 18.96, 33.25, 0.3341, 64.65, "loses"),
             (0.3, 143.46, -0.89, -5.97, 4.68, 0.1237, 197.65, "ties"),
             (0.375, 142.17, -2.18, -6.96, 2.85, 0.0808, 239.45, "ties"),
             (0.45, 142.87, -1.49, -6.14, 3.22, 0.0659, 280.87, "ties"),
             (0.6, 148.92, 4.57, 0.15, 9.18, 0.1251, 359.57, "loses")]
    for sigma, crps, delta, lo, hi, ks, sd, verdict in sweep:
        rows.append({"arm": "composition_sum_plus_player_season_effect",
                     "unit": "ps_effect_sweep", "n": 742, "sigma": sigma,
                     "crps_minutes": crps, "crps_delta": delta, "ci_lo": lo, "ci_hi": hi,
                     "pit_ks": ks, "predictive_sd": sd, "verdict": verdict,
                     "sigma_source": "injected_grid"})
    for sigma, crps, ks, sd in ((0.0, 139.89, 0.3401, 54.06), (0.3, 119.51, 0.1832, 144.75),
                                (0.375, 117.45, 0.1443, 174.21),
                                (0.45, 117.07, 0.1112, 203.38),
                                (0.6, 119.55, 0.1103, 258.61)):
        rows.append({"arm": "composition_sum_plus_player_season_effect",
                     "unit": "ps_sigma_on_train", "n": 1145, "sigma": sigma,
                     "crps_minutes": crps, "pit_ks": ks, "predictive_sd": sd,
                     "sigma_source": "train_grid", "seasons": "2020-21, 2021-22"})
    for arm, r, forced, sd, roster, n in (
            ("composition_sum", -0.0509, -0.0664, 79.0, 16.05, 963),
            ("minutes_head", -0.0001, -0.1060, 1022.9, 10.43, 626)):
        rows.append({"arm": arm, "unit": "teammate_coupling", "n": n, "r_teammates": r,
                     "r_implied_by_fixed_sum": forced, "team_season_sum_sd": sd,
                     "roster_size": roster})
    return pd.DataFrame(rows)


def _composition_ladder() -> pd.DataFrame:
    """The composition's own `make stan-composition` ladder, floor and comparator too."""
    return pd.DataFrame([
        {"variant": "carry_forward", "n_features": 0, "val_crps": 4.6776,
         "val_r2": 0.4442, "val_team_sum_abs": 0.0, "selected": False, "beats_floor": True},
        {"variant": "betabinom", "n_features": 23, "val_crps": 4.5417, "val_r2": 0.4699,
         "val_team_sum_abs": 0.0, "selected": False, "beats_floor": True},
        {"variant": "betabinom_ot_graded", "n_features": 25, "val_crps": 4.4945,
         "val_r2": 0.4741, "val_team_sum_abs": 0.0, "selected": True, "beats_floor": True},
        {"variant": "independent_comparator", "n_features": -1, "val_crps": 4.7842,
         "val_r2": 0.4024, "val_team_sum_abs": 33.89, "selected": False,
         "beats_floor": False}])


def _minutes_index() -> pd.DataFrame:
    """Two index rows shaped like the artifact, at the two units the page contrasts."""
    return pd.concat([
        _index(head="composition", label="minutes composition", model_class="minutes",
               unit="player-game inside a team-game", player_season_sigma=0.45),
        _index(head="minutes", label="min|available", model_class="minutes",
               unit="player-season", player_season_sigma=0.0)], ignore_index=True)


def test_the_two_units_are_drawn_against_their_own_floors_not_a_shared_one():
    """4.5 CRPS minutes per player-game and 170 per season cannot share an axis, so the
    figure plots the ratio to each unit's own floor — and the whole finding is that the
    same head lands on opposite sides of zero in the two panels."""
    board = model_cards.unit_board(_unification(), _composition_ladder(),
                                   _minutes_index())
    assert len(board) == 4
    composition = board[board["head"] == "composition"].set_index("unit")
    marginal = board[board["head"] == "minutes"].set_index("unit")
    assert composition.loc["fitted", "improvement"] > 0
    assert composition.loc["season", "improvement"] < 0
    assert marginal.loc["fitted", "improvement"] < 0
    assert marginal.loc["season", "improvement"] > 0
    # Each row is against the floor of its own unit, not of the other one.
    assert composition.loc["fitted", "floor_crps"] == 4.6776
    assert composition.loc["season", "floor_crps"] == 161.29
    assert (board["clears"] == (board["crps"] < board["floor_crps"])).all()


def test_the_marginal_head_is_read_from_the_comparator_arm_at_the_composition_unit():
    """The two heads meet at the per-game unit only because the composition's own ladder
    refits the marginal head as its control. Mapping that arm to the wrong head would put
    the composition's own number on both rows and the reversal would vanish."""
    board = model_cards.unit_board(_unification(), _composition_ladder(),
                                   _minutes_index())
    fitted = board[board["unit"] == "fitted"].set_index("head")
    assert fitted.loc["minutes", "arm"] == model_cards.COMPARATOR_ARM
    assert fitted.loc["composition", "arm"] == "betabinom_ot_graded"
    season = board[board["unit"] == "season"].set_index("head")
    assert season.loc["composition", "arm"] == model_cards.ARM_COMPOSITION
    assert season.loc["minutes", "arm"] == model_cards.ARM_MARGINAL


def test_the_unit_labels_are_read_from_the_index_rather_than_typed():
    index = _minutes_index()
    board = model_cards.unit_board(_unification(), _composition_ladder(), index)
    assert set(board[board["unit"] == "fitted"]["unit_label"]) == {
        "player-game inside a team-game"}
    assert set(board[board["unit"] == "season"]["unit_label"]) == {
        "player-season, summed"}
    assert set(board["label"]) == {"minutes composition", "min|available"}


def test_a_missing_ladder_leaves_the_season_half_of_the_board_alone():
    """A half-built `outputs/predictions/` draws the half it has rather than raising."""
    board = model_cards.unit_board(_unification(), None, _minutes_index())
    assert set(board["unit"]) == {"season"} and len(board) == 2
    assert model_cards.unit_board(None, None, _minutes_index()).empty
    assert model_cards.unit_table(pd.DataFrame()).empty


def test_the_paired_gap_flips_its_interval_ends_with_its_sign():
    """The artifact stores one direction and the page has a head selector, so the gap is
    re-oriented — and an interval whose ends are negated without being swapped would be
    backwards in exactly one of the two branches."""
    unification = _unification()
    forward = model_cards.season_gap(unification, model_cards.HEAD_COMPOSITION)
    reversed_ = model_cards.season_gap(unification, model_cards.HEAD_MARGINAL)
    assert forward["crps_delta"] == pytest.approx(25.70)
    assert reversed_["crps_delta"] == pytest.approx(-25.70)
    assert reversed_["ci_lo"] < reversed_["crps_delta"] < reversed_["ci_hi"]
    assert reversed_["ci_lo"] == pytest.approx(-forward["ci_hi"])
    assert reversed_["ci_hi"] == pytest.approx(-forward["ci_lo"])
    assert model_cards.season_gap(None) is None


def test_the_spread_panel_separates_where_the_predictive_sits_from_how_wide_it_is():
    """Drawing only the means would say the two heads are the same model, and drawing only
    the spread would not say that the fit is a tie. The block's claim needs both."""
    panel = model_cards.spread_panel(_unification(), _minutes_index())
    groups = dict(panel.groupby("metric")["group"].first())
    assert groups["mae_minutes"] == model_cards.GROUP_MEAN
    assert groups["predictive_sd"] == model_cards.GROUP_SPREAD
    wide = panel.pivot(index="metric", columns="head", values="value")
    assert wide.loc["mae_minutes"].max() / wide.loc["mae_minutes"].min() < 1.01
    assert (wide.loc["predictive_sd", "minutes"]
            / wide.loc["predictive_sd", "composition"]) > 4
    # The residual sd rides only on the panel that has a target value to be read against.
    reference = panel.set_index("metric")["reference"]
    assert reference.loc["predictive_sd"].notna().all()
    assert reference.drop("predictive_sd").isna().all()


def test_the_sweep_merges_two_grids_on_sigma_rather_than_stacking_them():
    """The grids score disjoint rows, so their CRPS levels are not comparable — merging on
    sigma is what lets the figure give them separate panels and still read one optimum
    against the other."""
    sweep = model_cards.sigma_sweep(_unification())
    assert list(sweep["sigma"]) == sorted(sweep["sigma"])
    assert sweep["val_n"].dropna().unique().tolist() == [742]
    assert sweep["train_n"].dropna().unique().tolist() == [1145]
    val_best = float(sweep.loc[sweep["val_crps"].idxmin(), "sigma"])
    train_best = float(sweep.loc[sweep["train_crps"].idxmin(), "sigma"])
    assert val_best == 0.375 and train_best == 0.45
    # Both optima are interior, which is what makes either one a measurement.
    for column in ("val_crps", "train_crps"):
        best = sweep[column].idxmin()
        assert 0 < best < len(sweep) - 1
    assert model_cards.sigma_sweep(None).empty


def test_a_tie_in_the_sweep_is_an_interval_that_straddles_zero():
    """The page draws the gaps through the tournament page's own paired builder, whose
    `crosses_zero` is the hollow marker. That flag has to agree with the verdict the
    artifact recorded, or the figure and the table would say different things."""
    sweep = model_cards.sigma_sweep(_unification())
    gaps = model_cards.sigma_gaps(sweep, shipped=0.45)
    assert list(gaps["crosses_zero"]) == [v == "ties" for v in gaps["verdict"]]
    assert ((gaps["gap_lo"] <= gaps["gap"]) & (gaps["gap"] <= gaps["gap_hi"])).all()
    shipped = gaps[gaps["sigma"] == 0.45].iloc[0]
    assert "shipped" in shipped["strategy"] and bool(shipped["crosses_zero"])
    assert "un-injected" in gaps[gaps["sigma"] == 0.0].iloc[0]["strategy"]
    assert model_cards.sigma_gaps(pd.DataFrame()).empty


def test_the_shipped_sigma_is_read_from_the_composition_card_not_typed():
    """`player_season_sigma` is what `make posteriors` recorded and what
    `rehydrate_composition` applies, so a page that typed 0.450 would keep printing it
    after the shipped value moved."""
    index = _minutes_index()
    assert model_cards.shipped_sigma(index) == 0.45
    assert model_cards.sigma_label(0.45, 0.45).endswith("shipped")
    assert model_cards.sigma_label(0.375, 0.45) == "σ = 0.375"
    moved = index.copy()
    moved.loc[moved["head"] == "composition", "player_season_sigma"] = 0.3
    assert model_cards.shipped_sigma(moved) == 0.3
    assert model_cards.shipped_sigma(pd.DataFrame({"head": []})) is None
    row = model_cards.sigma_row(model_cards.sigma_sweep(_unification()), 0.45)
    assert row is not None and float(row["val_pit_ks"]) == pytest.approx(0.0659)
    assert model_cards.sigma_row(model_cards.sigma_sweep(_unification()), 0.9) is None


def test_each_head_is_coupled_against_its_own_roster_size_not_a_shared_line():
    """−1/(K−1) is arithmetic, and K differs between the heads because the marginal head's
    prior-minutes filter drops real teammates. One shared reference line would be wrong for
    one of the two rows."""
    panel = model_cards.teammate_coupling(_unification(), _minutes_index())
    assert len(panel) == 2 and panel["forced"].nunique() == 2
    for _, row in panel.iterrows():
        assert row["forced"] == pytest.approx(-1.0 / (row["roster"] - 1), abs=1e-3)
    composition = panel[panel["head"] == "composition"].iloc[0]
    marginal = panel[panel["head"] == "minutes"].iloc[0]
    # The composition sits on its constraint; the marginal head reads independence.
    assert abs(composition["measured"] - composition["forced"]) < 0.02
    assert abs(marginal["measured"]) < abs(marginal["forced"]) / 10
    assert marginal["team_sd"] > 10 * composition["team_sd"]
    assert len(model_cards.coupling_table(panel)) == 2


def test_a_season_reading_goes_through_the_arm_map_and_survives_a_missing_column():
    unification = _unification()
    assert model_cards.season_reading(unification, "minutes", "crps_minutes") == 144.35
    assert model_cards.season_reading(unification, "composition", "pit_ks") == 0.3341
    assert model_cards.season_reading(unification, "minutes", "absent") is None
    assert model_cards.season_reading(None, "minutes", "crps_minutes") is None
    coverage = model_cards.coverage_row(unification)
    assert coverage is not None and int(coverage["n"]) == 1111


# ── The minutes page's four figures ───────────────────────────────────────────

def test_the_two_units_take_a_facet_each_so_neither_crps_sets_the_others_axis():
    board = model_cards.unit_board(_unification(), _composition_ladder(),
                                   _minutes_index())
    fig = charts.fig_unit_verdict(board, theme.theme("light"), model_cards.MINUTES_SLOTS)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 2 and all(len(t.x) == 2 for t in bars)
    assert {t.xaxis for t in bars} == {"x", "x2"}
    # The zero line is the floor, so it is drawn at reference weight rather than as a
    # gridline — the one encoding of the verdict that survives the relief rule.
    assert all(axis.zerolinewidth == 2 for axis in
               (fig.layout.xaxis, fig.layout.xaxis2))
    assert all("%" in t for trace in bars for t in trace.text)


def test_each_head_keeps_one_palette_slot_across_every_figure_on_the_page():
    """Three of the four are comparisons *between* the heads, so a colour that followed the
    selector would mean two different things on one screen."""
    th = theme.theme("light")
    unification, index = _unification(), _minutes_index()
    composition, marginal = th["series"][0], th["series"][1]
    verdict = charts.fig_unit_verdict(
        model_cards.unit_board(unification, _composition_ladder(), index), th,
        model_cards.MINUTES_SLOTS)
    spread = charts.fig_metric_facets(
        model_cards.spread_panel(unification, index), th, model_cards.MINUTES_SLOTS)
    coupling = charts.fig_coupling(
        model_cards.teammate_coupling(unification, index), th, model_cards.MINUTES_SLOTS)
    for fig in (verdict, spread):
        for trace in [t for t in fig.data if t.type == "bar"]:
            assert list(trace.marker.color) == [composition, marginal]
    measured = [t for t in coupling.data if t.mode == "markers"][-1]
    assert list(measured.marker.color) == [composition, marginal]
    # Two series is inside the all-pairs cap, which is why nothing here is grayed.
    assert len(model_cards.MINUTES_SLOTS) <= theme.ALL_PAIRS_CAP


def test_a_metric_facet_gets_its_own_axis_range_and_only_one_carries_a_reference():
    panel = model_cards.spread_panel(_unification(), _minutes_index())
    fig = charts.fig_metric_facets(panel, theme.theme("light"),
                                   model_cards.MINUTES_SLOTS)
    axes = [fig.layout[name].range for name in ("xaxis", "xaxis2", "xaxis3", "xaxis4")]
    assert all(r is not None for r in axes)
    assert len({tuple(r) for r in axes}) == 4
    references = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(references) == 1
    assert float(references[0].x0) == pytest.approx(243.50)
    assert any("residual sd" in (a.text or "") for a in fig.layout.annotations)


def test_the_two_sigma_grids_never_share_a_y_axis():
    """Their CRPS levels are on disjoint rows. One shared axis would invite exactly the
    comparison the two panels exist to forbid."""
    sweep = model_cards.sigma_sweep(_unification())
    fig = charts.fig_sigma_grids(sweep, theme.theme("light"), marginal_crps=144.35)
    curves = [t for t in fig.data if t.mode == "lines+markers"]
    assert len(curves) == 2
    assert {t.yaxis for t in curves} == {"y", "y2"}
    # Two axes, and neither is *linked* to the other — `shared_yaxes` would set `matches`
    # and put two disjoint row sets' CRPS on one scale, which is the whole hazard.
    assert fig.layout.yaxis2.matches is None
    left, right = (min(t.y) for t in curves), (max(t.y) for t in curves)
    assert max(right) - min(left) > 20   # the two levels really are far apart
    titles = [a.text for a in fig.layout.annotations]
    assert any("742" in t for t in titles) and any("1,145" in t for t in titles)
    # The marginal head's level is drawn once, on the split it was scored on.
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 1 and float(lines[0].y0) == pytest.approx(144.35)


def test_the_coupling_figure_draws_the_gap_rather_than_two_rival_series():
    panel = model_cards.teammate_coupling(_unification(), _minutes_index())
    fig = charts.fig_coupling(panel, theme.theme("light"), model_cards.MINUTES_SLOTS)
    dumbbells = [t for t in fig.data if t.mode == "lines"]
    assert len(dumbbells) == len(panel)
    for trace, (_, row) in zip(dumbbells, panel.iterrows()):
        assert list(trace.x) == [row["forced"], row["measured"]]
        assert trace.y[0] == trace.y[1]          # one row, not two series
    forced = [t for t in fig.data if t.mode == "markers"][0]
    assert forced.marker.color == "rgba(0,0,0,0)"   # the reference is hollow, like a PPC
    assert forced.marker.line.color == theme.theme("light")["ink"]


def test_a_bare_reference_line_draws_no_annotation():
    """`fig_paired`'s baseline label used to sit at the top of the paper, where it collided
    with the legend on any frame whose zero landed under one — visible on the minutes page's
    sigma sweep and latent on the tournament page. The axis title already names it."""
    sweep = model_cards.sigma_sweep(_unification())
    gaps = model_cards.sigma_gaps(sweep, shipped=0.45)
    fig = charts.fig_paired(gaps, theme.theme("light"), baseline="the marginal head",
                            unit="season-total CRPS minutes")
    assert "the marginal head" in fig.layout.xaxis.title.text
    assert not [a for a in fig.layout.annotations if "baseline" in (a.text or "")]
    assert [s for s in fig.layout.shapes if s.type == "line" and s.x0 == 0]


# ── The shipped artifacts, for page 4 ─────────────────────────────────────────

def test_the_shipped_unification_still_reverses_across_the_two_units():
    """The page's whole claim, held against the real artifacts: one posterior clears its
    floor at one unit and fails at the other, and the rival head does the reverse. If a
    refit ever changes that, the page's captions have to move with it."""
    board = model_cards.unit_board(_card(model_cards.MINUTES_UNIFICATION_FILE),
                                   _card(model_cards.COMPOSITION_METRICS_FILE),
                                   _card(model_cards.INDEX_FILE))
    assert len(board) == 4
    verdicts = {(r["head"], r["unit"]): bool(r["clears"]) for _, r in board.iterrows()}
    assert verdicts == {("composition", "fitted"): True,
                        ("composition", "season"): False,
                        ("minutes", "fitted"): False,
                        ("minutes", "season"): True}
    # Both units name a head this page actually carries, so neither row can go unlabelled.
    assert set(board["head"]) == set(model_cards.CLASSES["minutes"].heads)


def test_the_shipped_sigma_is_the_train_grids_own_optimum():
    """The load-bearing property of the shipped 0.450: it is read off the *training* rows,
    not the ones it is scored against. A refit that moved the train optimum without moving
    `player_season_sigma` would leave the page claiming a σ nothing selected."""
    sweep = model_cards.sigma_sweep(_card(model_cards.MINUTES_UNIFICATION_FILE))
    shipped = model_cards.shipped_sigma(_card(model_cards.INDEX_FILE))
    assert shipped is not None
    assert float(sweep.loc[sweep["train_crps"].idxmin(), "sigma"]) == pytest.approx(shipped)
    assert model_cards.sigma_row(sweep, shipped) is not None
    # And at that σ the gap against the marginal head is a tie, which is the claim tiled.
    row = model_cards.sigma_row(sweep, shipped)
    assert float(row["val_ci_lo"]) < 0 < float(row["val_ci_hi"])


def test_the_shipped_index_lets_both_minutes_heads_render_all_seven_blocks():
    cards = {key: _card(name) for key, name in (
        ("index", model_cards.INDEX_FILE),
        ("coefficients", model_cards.COEFFICIENTS_FILE),
        ("features", model_cards.FEATURES_FILE),
        ("correlations", model_cards.CORRELATION_FILE),
        ("ecdf", model_cards.ECDF_FILE),
        ("calibration", model_cards.CALIBRATION_FILE),
        ("quantile", model_cards.QUANTILE_FILE),
        ("sample", model_cards.SAMPLE_FILE))}
    th = theme.theme("light")
    for head in model_cards.heads_of(cards["index"], "minutes"):
        row = model_cards.head_row(cards["index"], head)
        assert str(row["unit"]).strip() and str(row["response_label"]).strip()
        order = model_cards.feature_order(cards["features"], head)
        assert len(order) == int(row["n_features"])
        charts.fig_features(
            model_cards.histogram_panel(cards["features"], head, order), th)
        charts.fig_correlation(
            model_cards.correlation_square(cards["correlations"], head, "train"), th)
        charts.fig_coefficients(
            model_cards.coefficient_panel(cards["coefficients"], head), th)
        charts.fig_ecdf(
            {model_cards.SPLIT_LABELS[s]: model_cards.ecdf_panel(cards["ecdf"], head, s)
             for s in model_cards.SPLITS}, th, str(row["response_label"]))
        _draw_quantile_block(cards, head, th)
        assert len(model_cards.build_checks(row)) == 4
    # **The composition ships a quantile residual**, which the step that built this panel
    # expected it not to: `predictive_check = none` is about its *fitted* value, and `u` is
    # a function of the draws and the observed, which its `predict_samples` puts on the
    # minutes scale. Its own `score_samples` computes the same statistic.
    assert model_cards.quantile_note(
        model_cards.head_row(cards["index"], "composition")) == ""
    assert not model_cards.qq_panel(cards["quantile"], "composition", "train").empty
    # The composition is the widest head in the project and the reason block 2 has a limit.
    assert len(model_cards.feature_order(cards["features"], "composition")) == 25
    # Its dispersion is role-graded, which is why block 4 tiles four of them and not one.
    scalars = model_cards.scalar_terms(cards["coefficients"], "composition")
    assert int((scalars["term_role"] == "dispersion").sum()) == 4


# ── Page 7 · inputs beyond the heads ──────────────────────────────────────────
#
# Three families that have nothing in common except that none of them is a fitted
# coefficient, so they get three groups of builders rather than one.

def _adp_panel() -> pd.DataFrame:
    """A panel shaped like `adp_panel.parquet`: long over players, dated per snapshot.

    Four seasons that between them cover every case the block has to handle — a season
    with a legal board and a late one beside it, a season with only late boards, a season
    with several legal boards, and a board for a season that has not been played.
    """
    snapshots = [
        # (season, source, as_of, season_start, lag, legal, players)
        ("2018-19", "fantasypros", "2019-09-02", "2018-10-16", 321.0, False, 3),
        ("2022-23", "fantasypros", "2022-10-02", "2022-10-18", -16.0, True, 4),
        ("2022-23", "fantasypros", "2022-10-15", "2022-10-18", -3.0, True, 4),
        ("2023-24", "fantasypros", "2023-10-17", "2023-10-24", -7.0, True, 2),
        ("2023-24", "fantasypros", "2024-08-03", "2023-10-24", 284.0, False, 2),
        ("2026-27", "draftkings", "2026-07-28", None, None, True, 5),
    ]
    rows = []
    for season, source, as_of, start, lag, legal, players in snapshots:
        for i in range(players):
            rows.append({"season": season, "snapshot_source": source,
                         "as_of_date": as_of, "season_start_date": start,
                         "snapshot_lag_days": lag,
                         "captured_before_season_start": legal,
                         "player_name": f"Player {i}", "adp": 1.0 + i})
    return pd.DataFrame(rows)


def _adp_profile() -> pd.DataFrame:
    rows = [("agreement", "n_pairs", 226.0), ("agreement", "spearman", 0.8675),
            ("agreement", "mean_abs_rank_gap", 23.115),
            ("ladder", "consensus raw", 24.378),
            ("ladder", "+ linear rescale", 21.377),
            ("ladder", inputs.SHIPPED_LADDER_ARM, 17.322),
            ("ladder", "+ isotonic + C/F/G offset", 15.316),
            ("position_bias", "C:mean_rank_gap", 13.88),
            ("position_bias", "C:n", 43.0),
            ("position_bias", "G:mean_rank_gap", -4.89),
            ("position_bias", "G:n", 97.0),
            ("tier_gap", "R9+:mean_abs_rank_gap", 32.31), ("tier_gap", "R9+:n", 134.0),
            ("tier_gap", "R1-2:mean_abs_rank_gap", 5.09), ("tier_gap", "R1-2:n", 23.0)]
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def _adp_audit() -> pd.DataFrame:
    columns = ["section", "source", "rule", "board_name", "matched_name", "board_season",
               "seasons_apart", "player_id", "metric", "value"]
    rows = [
        ("fuzzy_match", "draftkings", "prefix", "Alexandre Sarr", "Alex Sarr", "2025-26",
         0.0, 1.0, "", np.nan),
        ("ablation_match", "draftkings", "surname_initial", "Cameron Boozer",
         "Carlos Boozer", "2026-27", 12.0, 2.0, "agrees_with_cascade", 0.0),
        ("ablation_match", "draftkings", "surname_initial", "Alexandre Sarr", "Alex Sarr",
         "2025-26", 0.0, 1.0, "agrees_with_cascade", 1.0),
        ("summary", "", "cascade", "", "", "", np.nan, np.nan, "unmatched", 17.0),
        ("summary", "", "cascade", "", "", "", np.nan, np.nan, "matchable_rows", 3417.0),
        ("summary", "", "cascade", "", "", "", np.nan, np.nan,
         "unmatched_rate_cascade", 0.00498),
        ("summary", "", "surname_initial", "", "", "", np.nan, np.nan,
         "ablation_false_matches", 23.0),
        ("summary", "", "surname_initial", "", "", "", np.nan, np.nan,
         "unmatched_rate_surname_initial", 0.0),
    ]
    return pd.DataFrame(rows, columns=columns)


def _calendar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A calendar and its program table, with one gap of each kind."""
    rows = [
        ("injury_reports", "2026-08-08", "captured", False, 130.0),
        ("injury_reports", "2026-08-09", "nothing_to_capture", False, 0.0),
        ("injury_reports", "2026-08-10", "missed", True, 0.0),
        ("espn_injuries", "2026-08-08", "captured", False, 40.0),
        ("espn_injuries", "2026-08-09", "missed", False, 0.0),
        ("espn_injuries", "2026-08-10", "missed", False, 0.0),
        ("adp_fantasypros", "2026-07-28", "captured", False, 260.0),
        ("adp_draftkings", "2026-07-28", "captured", False, 942.0),
    ]
    calendar = pd.DataFrame(rows, columns=["program", "capture_date", "state",
                                           "recoverable", "records"])
    programs = pd.DataFrame([
        ("injury_reports", "daily", "window", 210, "2026-08-08", "2026-08-10",
         3, 1, 1, 1, 1, 0),
        ("espn_injuries", "daily", "never", 0, "2026-08-08", "2026-08-10",
         3, 1, 0, 2, 0, 2),
        ("adp_fantasypros", "event", "archive", 0, "2026-07-28", "2026-07-28",
         1, 1, 0, 0, 0, 0),
        ("adp_draftkings", "event", "never", 0, "2026-07-28", "2026-07-28",
         1, 1, 0, 0, 0, 0),
    ], columns=["program", "cadence", "recovery", "recovery_window_days", "window_start",
                "window_end", "n_days", "n_captured", "n_nothing_to_capture", "n_missed",
                "n_recoverable", "n_lost"])
    programs["records"] = 0.0
    return calendar, programs


def _calibrated() -> dict[str, pd.DataFrame]:
    """The four artifacts, at three windows, with the components' kinds carried."""
    counts = ["fga", "reb"]
    conversions = ["fg3a|fga", "fg3m|fg3a"]
    kinds = {**{c: "count" for c in counts}, **{c: "conversion" for c in conversions}}
    residual, serial, bonus, dispersion = [], [], [], []
    for shift, window in enumerate(inputs.FIT_WINDOWS):
        for a in counts + conversions:
            for b in counts + conversions:
                r = 1.0 if a == b else (0.12 if {a, b} == set(counts)
                                        else -0.05 + 0.001 * shift)
                residual.append({"fit_window": window, "component_a": a,
                                 "component_b": b, "kind_a": kinds[a],
                                 "kind_b": kinds[b], "r": r, "n_games": 1000,
                                 "basis": inputs.MINUTES_CONDITIONED,
                                 "minutes_conditioned": True})
                residual.append({"fit_window": window, "component_a": a,
                                 "component_b": b, "kind_a": kinds[a],
                                 "kind_b": kinds[b], "r": r + 0.5, "n_games": 1000,
                                 "basis": "raw", "minutes_conditioned": False})
        for component, kind, inflation in (("min", "minutes", 2.43 - 0.01 * shift),
                                           ("min_detrended", "minutes", 1.70),
                                           ("fga", "count", 1.37)):
            serial.append({"fit_window": window, "component": component, "kind": kind,
                           "block_inflation": inflation, "lag1": 0.28, "n_pairs": 5000})
        for unit, fitted in (("player_season", 0.0967), ("player_game", 0.0248)):
            bonus.append({"fit_window": window, "analysis": "fitted", "unit": unit,
                          "bucket": "all", "overdispersion": fitted + 0.0001 * shift,
                          "n": 1000, "relative_bias": 0.0, "is_shipped": False,
                          "is_independent": False})
            bonus.append({"fit_window": window, "analysis": "calibration", "unit": unit,
                          "bucket": "all", "overdispersion": 0.1, "n": 1000,
                          "relative_bias": 0.16, "is_shipped": True,
                          "is_independent": False})
        dispersion.append({"metric": "game_level_rho", "fit_window": window,
                           "rho": 0.078, "implied_overdispersion": 4.65 + 0.03 * shift,
                           "n_player_games": 700000.0, "note": ""})
    return {"residual": pd.DataFrame(residual), "serial": pd.DataFrame(serial),
            "bonus": pd.DataFrame(bonus), "dispersion": pd.DataFrame(dispersion)}


# ── Block 1 · what point-in-time safety costs ─────────────────────────────────

def test_a_season_is_usable_if_any_board_beat_the_opener_not_if_all_did():
    """`training_rows` filters rows, not seasons, so a season with one legal board and
    three late ones is a season the backtest can draft in. Reading the verdict as an `all`
    would throw away 2023-24, which has exactly that shape on the real panel."""
    coverage = inputs.season_coverage(_adp_panel()).set_index("season")
    assert bool(coverage.loc["2023-24", "legal"])
    assert coverage.loc["2023-24", "n_snapshots"] == 2
    assert coverage.loc["2023-24", "n_legal"] == 1
    assert coverage.loc["2023-24", "earliest_legal"] == "2023-10-17"
    # And a season whose only boards are late is not usable at all.
    assert not bool(coverage.loc["2018-19", "legal"])
    assert coverage.loc["2018-19", "legal_rows"] == 0


def test_the_cost_of_the_dating_rule_is_counted_in_seasons_and_in_rows():
    """Both, because they say different things: the row share is what the panel loses and
    the season count is what the *backtest* loses, and a season is not partly draftable."""
    cost = inputs.point_in_time_cost(_adp_panel())
    assert (cost["seasons_held"], cost["seasons_legal"], cost["seasons_lost"]) == (4, 3, 1)
    assert cost["lost"] == ("2018-19",)
    assert cost["legal_rows"] < cost["rows"]
    assert 0.0 < cost["row_share"] < 1.0


def test_a_board_for_an_unplayed_season_is_legal_and_has_no_position_on_the_axis():
    """It cannot postdate a season that has not started, so it is kept; it has no tip-off
    to measure from, so it is named in the table rather than drawn at a zero it never had."""
    snaps = inputs.snapshots(_adp_panel())
    undated = inputs.undated(snaps)
    assert list(undated["season"]) == ["2026-27"] and bool(undated["legal"].iloc[0])
    assert len(inputs.datable(snaps)) == len(snaps) - 1
    assert inputs.datable(snaps)["lag_days"].notna().all()


def test_the_ladder_keeps_the_order_it_was_measured_in_and_flags_what_ships():
    """Sorting by score would turn a ladder of stacked corrections into a menu, and put an
    arm that is not shipped at the top of it."""
    ladder = inputs.ladder(_adp_profile())
    assert list(ladder["arm"])[0] == "consensus raw"          # as measured, not as ranked
    assert ladder["mean_abs_rank_gap"].iloc[-1] < ladder["mean_abs_rank_gap"].iloc[0]
    shipped = ladder[ladder["shipped"]]
    assert list(shipped["arm"]) == [inputs.SHIPPED_LADDER_ARM]
    # The shipped rung is deliberately NOT the best one on the board.
    assert shipped["mean_abs_rank_gap"].iloc[0] > ladder["mean_abs_rank_gap"].min()


def test_the_rejected_matching_rule_scores_better_and_is_still_wrong():
    """The page's one warning, and the reason it is a warning: an unmatched rate improves
    with every fabricated match, so it is monotonically increasing in its own error."""
    summary = inputs.match_summary(_adp_audit())
    assert summary["surname_initial"]["unmatched_rate_surname_initial"] < \
        summary["cascade"]["unmatched_rate_cascade"]
    assert summary["surname_initial"]["ablation_false_matches"] > 0
    # And the fabrications are listed by name rather than counted, worst-dated first.
    false = inputs.ablation_false_matches(_adp_audit())
    assert list(false["board name"]) == ["Cameron Boozer"]
    assert float(false["seasons apart"].iloc[0]) == 12.0
    # The surviving fuzzy tier stays small enough to read.
    assert len(inputs.fuzzy_matches(_adp_audit())) == 1


# ── Block 2 · the calendar is an alarm ────────────────────────────────────────

def test_recoverability_rides_on_the_row_label_because_it_never_varies_along_a_row():
    """Encoding it in the cell would spend a colour the palette cannot spare on a fact that
    is constant across the row — and the states already take two slots and a neutral."""
    _, programs = _calendar()
    labels = inputs.row_labels(programs)
    assert len(labels) == len(programs)
    assert "recoverable until it ages out" in labels[0]
    assert "permanently lost" in labels[1]
    assert len(inputs.STATE_ORDER) == 3


def test_an_event_programs_empty_days_are_drawn_as_nothing_rather_than_as_gaps():
    """A board that opens in October has no schedule to have missed, so filling its row
    would report a year of failures a year."""
    calendar, programs = _calendar()
    grid = inputs.calendar_grid(calendar, programs, days=30)
    per_program = grid.groupby("program").size().to_dict()
    assert per_program["adp_draftkings"] == 1
    assert per_program["injury_reports"] == 3
    # Every cell carries a code the figure can map, and the row it belongs on.
    assert set(grid["code"]) <= set(range(len(inputs.STATE_ORDER)))
    assert set(grid["row"]) == set(range(len(programs)))


def test_the_alarm_separates_a_chore_from_an_incident():
    """One number would collapse "run the cron" and "these days no longer exist"."""
    _, programs = _calendar()
    alarm = inputs.calendar_alarm(programs)
    assert alarm["recoverable"] == 1 and alarm["lost"] == 2
    assert alarm["worst"] == inputs.PROGRAM_NOTES["espn_injuries"].label
    # And the gap table says which of the two each day is.
    calendar, _ = _calendar()
    gaps = inputs.gap_table(calendar)
    assert len(gaps) == 3 and gaps["Still fetchable"].sum() == 1


def test_the_calendar_row_is_matched_by_index_not_by_label_prefix():
    """`row_labels` decorates a program name with its policy, so matching a cell to a row
    by label means matching a prefix — and two programs sharing one would silently stack."""
    calendar, programs = _calendar()
    programs = programs.copy()
    programs.loc[programs["program"] == "adp_draftkings", "program"] = "injury_reports_v2"
    calendar = calendar.copy()
    calendar.loc[calendar["program"] == "adp_draftkings", "program"] = "injury_reports_v2"
    grid = inputs.calendar_grid(calendar, programs, days=30)
    rows = dict(zip(grid["program"], grid["row"]))
    assert rows["injury_reports"] != rows["injury_reports_v2"]


def test_every_capture_program_has_a_note_and_every_note_a_program():
    """`PROGRAM_NOTES` is the page's interpretation of the artifact, so it is anchored in
    both directions — a new program with no note would render as a blank row, and a note
    for a program nobody captures would describe a source that is not there."""
    _, programs = _calendar()
    assert set(programs["program"]) == set(inputs.PROGRAM_NOTES)
    assert set(programs["recovery"]) <= set(inputs.RECOVERY_LABELS)
    for note in inputs.PROGRAM_NOTES.values():
        assert note.label and note.what and note.stake


# ── Block 3 · the window is shown, not chosen for the reader ──────────────────

def test_the_copula_keeps_its_counts_ahead_of_its_conversions():
    """Only the count block is imposed on the simulator's draws, so reading it off the
    wrong corner of the matrix would be silent. The order comes from the artifact."""
    frames = _calibrated()
    order = inputs.copula_components(frames["residual"])
    kinds = inputs.copula_kinds(frames["residual"])
    seen_conversion = False
    for name in order:
        if kinds[name] == "conversion":
            seen_conversion = True
        else:
            assert not seen_conversion, order
    square = inputs.copula_square(frames["residual"])
    assert list(square.index) == list(square.columns) == order


def test_the_copula_scale_is_narrowed_and_its_diagonal_blanked_together():
    """A self-correlation is 1.0 by construction and the largest real cell is a tenth of
    that, so a scale that fits the diagonal renders every real cell as the midpoint."""
    frames = _calibrated()
    limit = inputs.copula_limit(frames["residual"])
    assert 0.0 < limit < 1.0
    cells = inputs.copula_cells(frames["residual"])
    assert limit >= cells["r"].abs().max()
    masked = inputs.copula_square(frames["residual"], mask_diagonal=True)
    assert masked.to_numpy().diagonal().tolist() == [np.nan] * len(masked) or \
        np.isnan(np.diag(masked.to_numpy())).all()
    # The unmasked square keeps the diagonal, because there it is a free check on the pivot.
    assert (np.diag(inputs.copula_square(frames["residual"]).to_numpy()) == 1.0).all()
    fig = charts.fig_correlation(masked, theme.theme("light"), limit=limit)
    assert (fig.data[0].zmin, fig.data[0].zmax) == (-limit, limit)


def test_a_pair_appears_once_on_the_cell_list_and_never_against_itself():
    cells = inputs.copula_cells(_calibrated()["residual"], top=20)
    assert len(cells) == len(set(cells["pair"]))
    assert not any(pair.split(" · ")[0] == pair.split(" · ")[1] for pair in cells["pair"])


def test_the_detrended_minutes_series_is_not_drawn_as_a_second_component():
    """It is the same series with the season trend removed, so drawing both puts one
    component on the chart twice and invites reading them as two heads."""
    frames = _calibrated()
    drawn = inputs.block_inflation(frames["serial"], "train")
    assert "min_detrended" not in set(drawn["component"])
    assert drawn["component"].iloc[0] == "min"                  # largest first
    assert "min_detrended" in set(
        inputs.block_inflation(frames["serial"], "train", detrended=True)["component"])


def test_the_bonus_overdispersion_is_reported_per_unit_against_the_shipped_constant():
    """Two units, two answers, and the simulator draws at the second. Reporting one would
    make the constant look either calibrated or badly wrong depending which."""
    rows = inputs.bonus_rows(_calibrated()["bonus"], "train").set_index("unit")
    assert set(rows.index) == {"player_season", "player_game"}
    assert rows.loc["player_season", "fitted"] > rows.loc["player_game", "fitted"] * 3
    assert (rows["shipped_constant"] == 0.1).all()
    # The season unit sits on the constant and the game unit does not, which is the point.
    assert abs(rows.loc["player_season", "fitted"] - 0.1) < 0.01
    assert abs(rows.loc["player_game", "fitted"] - 0.1) > 0.05


def test_every_calibrated_input_reads_at_every_window_and_moves_almost_not_at_all():
    """The block's argument: the three windows differ by two seasons out of thirty, so a
    number consumed at the wrong one would never announce itself in the output."""
    frames = _calibrated()
    panel = inputs.window_panel(frames)
    assert len(panel) == len(inputs.CALIBRATED) * len(inputs.FIT_WINDOWS)
    assert set(panel["fit_window"]) == set(inputs.FIT_WINDOWS)
    assert panel["value"].notna().all()
    assert (panel["relative_spread"] < 0.10).all()
    label, spread = inputs.widest_relative_spread(panel)
    assert label in set(panel["label"]) and 0.0 <= spread < 0.10


def test_the_two_windows_the_page_names_are_the_two_the_code_actually_uses():
    """`train_val` is the safe default and `train` is what the simulator overrides it to,
    and the page's whole claim is that both are right for different reasons."""
    assert inputs.SAFE_WINDOW in inputs.FIT_WINDOWS
    assert inputs.SIM_WINDOW in inputs.FIT_WINDOWS
    assert inputs.SAFE_WINDOW != inputs.SIM_WINDOW
    # And the simulator's window is the one the figure highlights, alone.
    assert list(inputs.WINDOW_SLOTS) == [inputs.SIM_WINDOW]


def test_an_unnamed_row_grays_out_rather_than_taking_a_colour_nobody_chose():
    """The fallback that lets one builder serve a fixed pairing and highlight-and-gray."""
    th = theme.theme("light")
    assert charts._head_colors(th, ["a", "b"], {"a": 0}) == [th["series"][0], th["muted"]]


def test_the_window_facets_reshape_into_the_builder_the_minutes_page_already_has():
    """One figure, two pages: a handful of rows compared inside each facet, facets in
    different units. A near-copy would be a second place for the palette rules to drift."""
    frames = _calibrated()
    facets = inputs.window_facets(inputs.window_panel(frames))
    assert set(facets.columns) >= {"metric_label", "head", "label", "value", "text",
                                   "reference"}
    assert facets["reference"].isna().all()          # none of these has a target value
    fig = charts.fig_metric_facets(facets, theme.theme("light"), inputs.WINDOW_SLOTS,
                                   columns=2)
    assert len(fig.data) == len(inputs.CALIBRATED)
    for trace in fig.data:
        assert len(set(trace.marker.color)) == 2     # the simulator's window, and gray


# ── Page 7's figures ──────────────────────────────────────────────────────────

def test_the_calendar_names_its_states_without_a_colourbar_over_three_integers():
    """A heatmap draws no legend and a continuous colourbar over three categories is a
    claim they are a scale, so the states arrive as marker traces with no points in them."""
    calendar, programs = _calendar()
    th = theme.theme("light")
    fig = charts.fig_calendar(inputs.calendar_grid(calendar, programs, days=30),
                              inputs.row_labels(programs), th, inputs.STATE_ORDER,
                              inputs.STATE_LABELS)
    heat = fig.data[0]
    assert not heat.showscale
    keys = [trace.name for trace in fig.data[1:]]
    assert keys == [inputs.STATE_LABELS[s] for s in inputs.STATE_ORDER]
    assert all(trace.mode == "markers" for trace in fig.data[1:])
    # Two categorical slots and a neutral — well inside ALL_PAIRS_CAP.
    used = [trace.marker.color for trace in fig.data[1:]]
    assert used == [th["series"][0], th["neutral"], th["series"][1]]
    assert len([c for c in used if c in th["series"]]) <= theme.ALL_PAIRS_CAP


def test_a_day_no_program_covers_stays_a_hole_rather_than_becoming_a_zero():
    """A painted cell would report "captured nothing" about a day nobody owed."""
    calendar, programs = _calendar()
    fig = charts.fig_calendar(inputs.calendar_grid(calendar, programs, days=30),
                              inputs.row_labels(programs), theme.theme("light"),
                              inputs.STATE_ORDER, inputs.STATE_LABELS)
    z = np.asarray(fig.data[0].z, dtype=float)
    assert np.isnan(z).any() and not fig.data[0].hoverongaps


def test_the_days_are_not_separated_but_the_rows_are():
    """At 150 days a gap between cells is as wide as a cell, so a run of captures came out
    as a barcode and a genuinely missing day looked like the gutter beside a present one."""
    calendar, programs = _calendar()
    fig = charts.fig_calendar(inputs.calendar_grid(calendar, programs, days=30),
                              inputs.row_labels(programs), theme.theme("light"),
                              inputs.STATE_ORDER, inputs.STATE_LABELS)
    assert fig.data[0].xgap == 0 and fig.data[0].ygap > 0


def test_the_adp_timeline_carries_legality_three_ways():
    """Position on the axis, colour, and marker shape — because whether a season exists at
    all turns on it, and three light-mode slots fall under 3:1 on the light surface."""
    snaps = inputs.datable(inputs.snapshots(_adp_panel()))
    fig = charts.fig_adp_lag(snaps, theme.theme("light"))
    assert len(fig.data) == 2
    legal, late = fig.data
    assert legal.marker.symbol != late.marker.symbol
    assert legal.marker.color != late.marker.color
    assert all(x <= 0 for x in legal.x) and all(x > 0 for x in late.x)
    assert "negative is before it" in fig.layout.xaxis.title.text
    # The reference at zero is drawn bare — the axis title already says what zero is.
    assert not fig.layout.annotations


def test_the_block_inflation_reference_is_drawn_over_the_bars_and_at_weight():
    """Below them it survives only in the gutters and reads as a dashed line, which
    `theme.py` bans; at hairline weight it is indistinguishable from the gridlines."""
    frames = _calibrated()
    fig = charts.fig_block_inflation(inputs.block_inflation(frames["serial"], "train"),
                                     theme.theme("light"))
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 1 and lines[0].x0 == 1.0
    assert lines[0].layer == "above" and lines[0].line.width >= 2
    assert lines[0].line.dash in (None, "solid")
    assert "independent draws" in fig.layout.xaxis.title.text


def test_every_bar_on_page_seven_prints_its_own_value():
    """The relief rule in its most literal form: no bar's length is the only route to its
    number, because three light-mode slots fall under 3:1 on the light surface."""
    th = theme.theme("light")
    frames = _calibrated()
    for fig in (charts.fig_ladder(inputs.ladder(_adp_profile()), th),
                charts.fig_block_inflation(
                    inputs.block_inflation(frames["serial"], "train"), th)):
        bars = [t for t in fig.data if isinstance(t, go.Bar)]
        assert bars and all(len(t.text) == len(t.x) for t in bars)
        assert all(t.textposition == "outside" and t.cliponaxis is False for t in bars)


def test_page_sevens_figures_carry_a_title_and_the_pinned_surface():
    """`apply_theme` sets `title.text` explicitly because a title object with a font and no
    text renders as the literal string "undefined" in a browser."""
    frames = _calibrated()
    calendar, programs = _calendar()
    for mode in theme.THEMES:
        th = theme.theme(mode)
        figs = [
            charts.fig_calendar(inputs.calendar_grid(calendar, programs, days=30),
                                inputs.row_labels(programs), th, inputs.STATE_ORDER,
                                inputs.STATE_LABELS),
            charts.fig_adp_lag(inputs.datable(inputs.snapshots(_adp_panel())), th),
            charts.fig_ladder(inputs.ladder(_adp_profile()), th),
            charts.fig_block_inflation(
                inputs.block_inflation(frames["serial"], "train"), th),
        ]
        for fig in figs:
            assert fig.layout.title.text is not None
            assert fig.layout.paper_bgcolor == th["surface"]
            assert fig.layout.plot_bgcolor == th["surface"]
            for trace in fig.data:
                if isinstance(trace, go.Scatter):
                    assert trace.mode                      # never plotly's own inference


# ── The shipped artifacts, for page 7 ─────────────────────────────────────────

def _eda(name: str) -> pd.DataFrame:
    path = EDA / name
    if not path.exists():
        pytest.skip(f"{name} not built")
    return pd.read_csv(path)


def test_the_shipped_panel_still_loses_four_of_its_nine_seasons_to_the_dating_rule():
    """Quoted on the page and in `docs/simulations-plan.md`. If a Wayback backfill ever
    recovers a season, the number moves and the prose has to move with it."""
    path = FEATURES / inputs.PANEL_FILE
    if not path.exists():
        pytest.skip("adp_panel.parquet not built")
    cost = inputs.point_in_time_cost(pd.read_parquet(path))
    assert cost["seasons_held"] == 9
    assert cost["seasons_legal"] == 5
    assert cost["lost"] == ("2017-18", "2018-19", "2019-20", "2024-25")
    # Both validation seasons survive, which is the coverage that actually matters.
    coverage = inputs.season_coverage(pd.read_parquet(path)).set_index("season")
    assert bool(coverage.loc["2022-23", "legal"]) and bool(coverage.loc["2023-24", "legal"])


def test_the_shipped_ladder_still_names_the_arm_that_ships():
    """The page flags the shipped rung rather than the best one, so the name has to exist
    in the artifact — a renamed rung would silently flag nothing."""
    ladder = inputs.ladder(_eda(inputs.PROFILE_FILE))
    assert int(ladder["shipped"].sum()) == 1
    assert ladder["mean_abs_rank_gap"].min() < \
        float(ladder.loc[ladder["shipped"], "mean_abs_rank_gap"].iloc[0])


def test_the_shipped_calendar_agrees_with_its_own_program_table():
    """The counts are read from the program table and the grid from the calendar, so the
    two files have to be one measurement. Both come from one `make capture-calendar` run."""
    calendar, programs = _eda(inputs.CALENDAR_FILE), _eda(inputs.PROGRAMS_FILE)
    assert set(programs["program"]) == set(inputs.PROGRAM_NOTES)
    for row in programs.itertuples(index=False):
        part = calendar[calendar["program"] == row.program]
        assert int((part["state"] == inputs.CAPTURED).sum()) == row.n_captured
        assert int((part["state"] == inputs.MISSED).sum()) == row.n_missed
        assert row.n_recoverable + row.n_lost == row.n_missed
    # Only a `window` program can hold a recoverable gap; the rest are gone when missed.
    windowed = set(programs.loc[programs["recovery"] == "window", "program"])
    assert set(programs.loc[programs["n_recoverable"] > 0, "program"]) <= windowed


def test_the_shipped_calibrated_inputs_read_at_all_three_windows():
    """The whole block: four numbers, three windows, and none of them moves enough to be
    noticed if it were consumed at the wrong one."""
    frames = {"residual": _eda(inputs.RESIDUAL_FILE), "serial": _eda(inputs.SERIAL_FILE),
              "bonus": _eda(inputs.BONUS_FILE)}
    path = PREDICTIONS / inputs.DISPERSION_FILE
    if not path.exists():
        pytest.skip("stan_minutes_dispersion.csv not built")
    frames["dispersion"] = pd.read_csv(path)
    panel = inputs.window_panel(frames)
    assert panel["value"].notna().all()
    assert (panel["relative_spread"] < 0.05).all()
    # The copula's own count block is the part the simulator imposes, and it is positive.
    assert inputs.copula_mean(frames["residual"], inputs.SIM_WINDOW) > 0
    # The shipped bonus constant is calibrated at the season unit and not at the game one.
    rows = inputs.bonus_rows(frames["bonus"], inputs.SIM_WINDOW).set_index("unit")
    assert abs(rows.loc["player_season", "fitted"] - 0.1) < 0.01
    assert rows.loc["player_game", "fitted"] < 0.05


# ── Page 1 · the Overview, and the three bounds it exists under ───────────────
#
# The charter amendment in `docs/dashboard-plan.md` grants this page an exemption from "the
# dashboard shows data, prose belongs in the docs" on three terms. Two of them are testable
# here and one is not: bound 1 (opens above the fold, ends inside one more screen) is a
# browser measurement and lives in the plan doc's verification section, bound 2 (every
# number read from an artifact) is what the `Spec` table below makes structural, and bound
# 3 (no registry, no provenance links, no reversal log) is a grep.
#
# Bound 2 gained a second half on 2026-08-10, when the five hero tiles became four sections
# of prose: a tile had nowhere to put a typed number and a sentence does, so a typed
# fragment is now held to carrying no digit at all.

def _overview_frames() -> dict:
    """A synthetic frame per source, shaped like the artifact and nothing like the data."""
    return {
        "coverage": pd.DataFrame({
            "analysis": ["feasibility"] * 4 + ["derivation"],
            "season": ["all", "2021-22", "2022-23", "2022-23", "all"],
            "season_type": ["regular", "regular", "regular", "playoff", "regular"],
            "player_games": [500.0, 200.0, 300.0, 40.0, 9.0]}),
        "budget": pd.DataFrame({
            "source": ["own_minutes", "home_away", "dk_pts_sd"],
            "share_of_variance": [0.464, 0.0003, float("nan")]}),
        "cards": pd.DataFrame({"head": ["a", "b", "c"], "divergences": [0, 0, 0]}),
        "season_total": pd.DataFrame({
            "treatment": ["beta_binomial", "full_season", "beta_binomial",
                          "oracle_gp", "oracle_rate"],
            "group": ["all", "all", "rotation", "all", "all"],
            "metric": ["mae_dk_total"] * 5,
            "value": [400.0, 610.0, 451.0, 214.0, 262.0]}),
        "components": pd.DataFrame({
            "head": ["fga", "blk", "fg3a", "ftm"],
            "kind": ["count", "count", "share", "share"],
            "variant": ["carry_forward"] * 4,
            "val_r2": [0.95, 0.81, 0.13, 0.30]}),
        "simulation": pd.DataFrame({"season": ["2022-23", "2023-24"],
                                    "n_sims": [2000, 2000]}),
        # Four rows, because the double-week facet is a second unit and the sentence reads
        # exactly one of them — a builder carrying only the row that is read cannot catch
        # a filter that stopped filtering.
        "weekly": pd.DataFrame({
            "period_type": ["week", "week", "double_week", "double_week"],
            "split": ["train", "validation", "train", "validation"],
            "zero_share": [0.2066, 0.1990, 0.1910, 0.1703],
            "predicted_zero_share": [0.1686, 0.1823, 0.1467, 0.1539]}),
        "sweep": pd.DataFrame({"strategy": ["adp", "model_mean", "adp"]}),
        # A rate that is exactly the field null plus the lift, so the tile's three printed
        # numbers can be checked to add up rather than merely to be present.
        "shipped": pd.DataFrame({
            "tournament": [overview.HEADLINE_TIER, "20k_spin_move"],
            "realized_p_advance": [1 / 6 + 0.125, 0.2397],
            "realized_lift": [0.125, 0.0730],
            "realized_seasons": [2, 2]}),
        "bracket": pd.DataFrame({"tournament": ["a", "b", "c", "a"]}),
    }


def test_every_source_names_the_make_target_that_writes_it():
    """`optional()` reports a missing artifact by naming its target; a blank one reports
    nothing useful, and this page reads more artifacts than any other."""
    assert len(overview.SOURCES) == len({s.key for s in overview.SOURCES})
    for source in overview.SOURCES:
        assert source.directory in (overview.EDA, overview.PREDICTIONS), source
        assert source.filename.endswith((".csv", ".parquet")), source
        assert source.target.startswith("make "), source


def _overview_text(frames: dict) -> str:
    """Everything a reader would see on the page, as one string.

    The sections and the diagram together, because a figure that moved from a sentence into
    a diagram box has not left the page and a test that watched only one half would call
    that a loss.
    """
    return " ".join([para.text for para in overview.paragraphs(frames)]
                    + [stage.figure for stage in overview.stage_readings(frames)])


def test_every_reading_declares_the_sources_it_reads():
    """Bound 2, structurally: a figure cannot reach the page except through a `Spec`.

    The needs are what let a half-built repo drop a reading instead of raising, so a spec
    that under-declares would take the page down on exactly the machine it was meant to
    protect. Every declared key has to be a real source.
    """
    keys = {source.key for source in overview.SOURCES}
    assert overview.PROSE_SPECS, "the paper states no result at all"
    for spec in overview.STAGE_SPECS + overview.PROSE_SPECS:
        assert spec.needs, spec
        assert set(spec.needs) <= keys, spec.needs


def test_the_paper_runs_introduction_methods_results_discussion():
    """`README.md`'s structure, at a landing page's length.

    Two to four sentences each is the brief and it is the whole defence of the rewrite: the
    complaint the five tiles drew was that they were numbers with no argument, and a
    section that grows past four sentences has started being the walkthrough's chapter
    again rather than a paragraph.
    """
    assert [s.heading for s in overview.SECTIONS] == [
        "Introduction", "Methods", "Results", "Discussion"]
    for section in overview.SECTIONS:
        assert 2 <= len(section.body) <= 4, section.heading


def test_typed_prose_carries_no_digit():
    """Bound 2 at the sentence, which is where it has to bite once the page is prose.

    A hero tile had nowhere to put a typed number — its value came from a `Spec` and its
    label was a label. A paragraph has room for one mid-sentence, which is exactly how the
    walkthrough's claims drifted from the documents that made them. So a digit on this page
    means "read from an artifact", and the contest's own rules are spelled in words:
    *sixteen players* is a rule and `46.4%` is a measurement.
    """
    for section in overview.SECTIONS:
        assert not any(ch.isdigit() for ch in section.heading), section.heading
        for fragment in section.body:
            if isinstance(fragment, str):
                assert not any(ch.isdigit() for ch in fragment), fragment


def test_a_missing_artifact_costs_its_own_sentences_and_no_others():
    """The half-built repo. Ten CSVs, and no one of them may blank the page."""
    frames = _overview_frames()
    full = _overview_text(frames)
    assert len(overview.paragraphs(frames)) == len(overview.SECTIONS)
    assert len(overview.stage_readings(frames)) == len(overview.STAGE_SPECS)
    typed = [fragment for section in overview.SECTIONS for fragment in section.body
             if isinstance(fragment, str)]
    for key in list(frames):
        short = {k: v for k, v in frames.items() if k != key}
        text = _overview_text(short)
        assert text, f"dropping {key} emptied the page"
        assert len(text) < len(full), f"dropping {key} cost the page nothing"
        # The typed half does not depend on an artifact and must survive all ten losses.
        for sentence in typed:
            assert sentence in text, (key, sentence[:40])
    assert overview.stage_readings({}) == []


def test_a_section_of_pure_lookups_loses_its_heading_rather_than_standing_empty():
    """`Results` is every-sentence-a-`Spec`, and a heading over nothing is worse than an
    absent section. The other three keep typed sentences and survive a bare repo, which is
    the asymmetry the drop rule has to get right rather than dropping on a count."""
    bare = overview.paragraphs({})
    assert [para.heading for para in bare] == ["Introduction", "Methods", "Discussion"]
    assert all(para.text.strip() for para in bare)


def test_every_figure_on_the_page_carries_a_digit_from_a_frame():
    """The bound stated as an assertion: no lookup and no stage may be a typed constant.

    Weak on its own — a hard-coded string has digits too — which is why it sits beside the
    reading tests below, where each figure is checked to *move with* its frame.
    """
    frames = _overview_frames()
    readings = [spec.build(frames) for spec in overview.PROSE_SPECS]
    for reading in readings + overview.stage_readings(frames):
        figure = getattr(reading, "text", None) or reading.figure
        assert any(ch.isdigit() for ch in figure), reading


def test_the_sentences_read_the_values_their_frames_hold():
    """Every figure in the prose, against the frame it came out of.

    Read off the joined text rather than off each `Reading`, because the page is a
    paragraph now: a sentence that built correctly and never made it into its section
    would pass a per-reading check and show the reader nothing.
    """
    text = _overview_text(_overview_frames())
    assert "**46.4%**" in text                      # introduction: the minutes share
    assert "**3** persisted fits" in text           # methods: the sampler
    assert "**0** divergences" in text
    assert "**400.0**" in text and "**610.0**" in text     # results: the ladder's two ends
    assert "**0.81–0.95**" in text                  # results: the count heads' floor
    assert "**29.2%**" in text and "**16.7%**" in text     # results: rate against the null
    assert "**2** validation seasons" in text
    assert "**214.0**" in text and "**262.0**" in text     # discussion: the two oracles
    assert "**19.9%**" in text and "**18.2%**" in text     # discussion: scoreless weeks


def test_the_scoreless_share_reads_one_period_type_and_one_split():
    """Three of the twenty scoring slots are **double** weeks carrying about twice the
    games, so the two period types are two units — pooling them would report a calendar
    fact as a model miss. The split is the one every other figure on the page is read on."""
    weekly = _overview_frames()["weekly"]
    assert overview.scoreless_weeks(weekly) == (0.1990, 0.1823)
    with pytest.raises(overview.MissingRow):
        overview.scoreless_weeks(weekly[weekly["period_type"] == "nothing"])


def test_the_two_oracles_are_read_as_a_pair_from_the_same_ladder():
    """The discussion sentence's whole claim is that these two disagree with the intuition
    about which half is hard, so both come off `season_total_metrics.csv` — the same file
    and the same `group`/`metric` filter the Results sentence uses."""
    metrics = _overview_frames()["season_total"]
    assert overview.season_total_mae(metrics, overview.ORACLE_GP_TREATMENT) == 214.0
    assert overview.season_total_mae(metrics, overview.ORACLE_RATE_TREATMENT) == 262.0


def test_the_floor_band_is_the_count_heads_and_not_the_conversions():
    """0.81-0.95 is the *count* heads' floor. Read off the conversions too it opens to
    [0.13, 0.96], and a band that wide is not a floor — `docs/simulations-plan.md` had to
    re-derive this once already, which is why the filter is pinned rather than assumed."""
    components = _overview_frames()["components"]
    assert overview.floor_band(components) == (0.81, 0.95)
    with pytest.raises(overview.MissingRow):
        overview.floor_band(components[components["kind"] == "nothing"])


def test_the_field_null_is_derived_so_the_three_numbers_add_up():
    """The Results sentence prints a rate and a null in the same breath. Typing 1/6 in
    beside a rate read from an artifact is how the two stop agreeing after a re-run."""
    got = overview.advance(_overview_frames()["shipped"])
    assert abs(got["rate"] - got["lift"] - got["null"]) < 1e-12
    assert abs(got["null"] - 1 / 6) < 1e-3


def test_the_pooled_row_is_not_counted_as_a_season():
    """`game_length_coverage.csv` carries an `all` row beside the per-season ones, and the
    stage prints a season count — so the pooled row is one off from a wrong number."""
    coverage = _overview_frames()["coverage"]
    assert overview.seasons(coverage) == 2
    assert overview.player_games(coverage) == 500.0


def test_a_lookup_that_finds_no_row_raises_rather_than_printing_nan():
    """`str()` of a missing cell is the four letters `nan`, which are truthy and print on
    the page. On a landing page there is no honest fallback, so this one is loud."""
    with pytest.raises(overview.MissingRow):
        overview.season_total_mae(_overview_frames()["season_total"], "no_such_arm")


def test_stage_notes_are_short_enough_to_stay_inside_their_own_box():
    """Plotly does not wrap an annotation and does not clip it either — an over-long note
    simply runs out over the box's border, and only a rendering shows it. `NOTE_WIDTH` is
    sized for the narrowest checked viewport; one wrapped line per note is the budget that
    keeps the diagram at `STAGE_HEIGHT`."""
    for stage in overview.stage_readings(_overview_frames()):
        assert overview.wrap(stage.note).count("<br>") == 0, stage.note
        assert len(stage.note) <= overview.NOTE_WIDTH, stage.note


def test_wrap_breaks_a_long_note_rather_than_truncating_it():
    assert overview.wrap("", 10) == ""
    wrapped = overview.wrap("one two three four five six", 10)
    assert wrapped.split("<br>") == ["one two", "three four", "five six"]


def test_every_route_points_at_a_real_page_and_states_no_result():
    """Bound 3, and the place a stray headline would try to creep back in.

    The blurbs are the only prose on the page that sits next to a link, which makes them
    the natural home for "the composition is 4.68x too narrow". They are held to naming
    what a page holds, and every one of them has to name a page that exists.
    """
    from dashboard import app
    paths = {view.url_path for view in app.VIEWS}
    assert len(overview.ROUTES) == len(app.VIEWS) - 1
    for route in overview.ROUTES:
        assert route.url_path in paths, route.url_path
        assert route.url_path != "overview", "the page does not link to itself"
        assert route.blurb.strip() and route.blurb[0].isupper(), route
        assert not any(ch.isdigit() for ch in route.blurb), route


def test_the_overview_holds_no_registry_no_provenance_link_and_no_reversal_log():
    """Bound 3 as a grep over both halves of the page.

    The three things named are what made the deleted walkthrough a documentation surface:
    an import of `decisions`, a link into `docs/`, and the withdrawn-entry log. Comments
    and docstrings are exempt and have to be — both modules cite the charter they live
    under in their own headers — so the check runs over the string literals that could
    actually reach a reader, which is every literal that is not a docstring.
    """
    for name in ("overview.py", "views/overview.py"):
        tree = ast.parse((ROOT / "dashboard" / name).read_text())
        imported = {n.module or "" for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom)}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                     for a in n.names}
        assert not any("decisions" in m for m in imported), name
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)
                      and isinstance(node.body[0].value.value, str)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in docstrings:
                continue
            assert "docs/" not in node.value, (name, node.value[:60])
            assert "withdrawn" not in node.value, (name, node.value[:60])


def test_the_pipeline_diagram_draws_one_box_per_stage_and_an_arrow_between():
    """Four arrows for five boxes: the chain is the point, and a missing head would leave
    five tiles that merely happen to be adjacent."""
    th = theme.theme("light")
    stages = overview.stage_readings(_overview_frames())
    fig = charts.fig_pipeline(stages, th)
    assert len(fig.layout.shapes) == len(stages)
    arrows = [a for a in fig.layout.annotations if a.showarrow]
    labels = [a for a in fig.layout.annotations if not a.showarrow]
    assert len(arrows) == len(stages) - 1
    assert len(labels) == 3 * len(stages)
    printed = {a.text for a in labels}
    for stage in stages:
        assert {stage.title, stage.figure} <= printed


def test_the_diagram_spends_no_colour_slot_on_a_sequence_that_is_not_a_scale():
    """Five steps is past `ALL_PAIRS_CAP` and, more to the point, the steps are not a
    scale — five hues would be an encoding that decodes to nothing. Every box takes the
    same neutral fill, and every figure prints itself, so the relief rule is trivial."""
    th = theme.theme("light")
    fig = charts.fig_pipeline(overview.stage_readings(_overview_frames()), th)
    assert {s.fillcolor for s in fig.layout.shapes} == {th["neutral"]}
    assert not fig.data, "the diagram plots no data, so it holds no traces"
    assert fig.layout.paper_bgcolor == th["surface"]


def test_the_diagram_leaves_room_for_the_outer_boxes_own_borders():
    """Zero horizontal margin puts the first and last box's 1px border on the paper edge,
    where it is clipped — caught in a PNG, invisible in the figure spec."""
    fig = charts.fig_pipeline(overview.stage_readings(_overview_frames()),
                              theme.theme("dark"))
    assert fig.layout.xaxis.range[0] < 0.0 and fig.layout.xaxis.range[1] > 1.0
    assert fig.layout.margin.l == 0 and fig.layout.margin.t < 10


def test_an_empty_diagram_does_not_divide_by_zero():
    """`stage_readings` returns [] on a repo with none of the five sources built."""
    fig = charts.fig_pipeline([], theme.theme("light"))
    assert fig.layout.shapes == () and fig.layout.annotations == ()


# ── The entrypoint hands its pages to whoever links to them ───────────────────

def test_the_page_registry_is_keyed_by_the_declared_path_not_the_rewritten_one():
    """Streamlit rewrites the *default* page's `url_path` to `""` so it can serve `/`.

    The Overview is the default page and the registry is what its sibling links come out
    of, so reading the key back off the `StreamlitPage` would lose one row — and it would
    lose whichever row is first, which is the one nothing links to, so nothing would fail.
    """
    from dashboard import app, shell

    class FakePage:
        def __init__(self, path):
            self._declared = path
            self.url_path = ""       # what Streamlit reports for the default page

    built = {view.url_path: FakePage(view.url_path) for view in app.VIEWS}
    shell.publish_pages(built)
    for view in app.VIEWS:
        assert shell.page(view.url_path) is built[view.url_path], view.url_path
    assert shell.page("no-such-page") is None
    shell.publish_pages({})
    assert shell.page("overview") is None


def test_the_entrypoint_publishes_the_pages_it_navigates_with():
    """`st.page_link` accepts only a page `st.navigation` was handed, so the two calls have
    to see the same objects. Building `pages()` twice would validate and route to the
    registered twin, which works by accident until a `url_path` changes."""
    source = (ROOT / "dashboard" / "app.py").read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [c for c in ast.walk(main) if isinstance(c, ast.Call)]
    assert sum(1 for c in calls if getattr(c.func, "id", "") == "pages") == 1
    assert any(getattr(c.func, "attr", "") == "publish_pages" for c in calls)


# ── The weekly-scores page ────────────────────────────────────────────────────
#
# The pure layer over `make weekly-scores`' six artifacts. Everything asserted here is a
# *reshape* — the emitter computed every metric, distance and band — so what is worth
# pinning is the small set of reshapes whose failure is silent: a facet quietly pooled, a
# spread statistic quietly swapped for the wrong one of three, a per-period point quietly
# averaged rather than row-weighted, and a threshold quoted in a caption that has drifted
# from the build that enforces it.

def _weekly_index() -> pd.DataFrame:
    """Two facets x two splits, in `weekly_score_index.csv`'s shape."""
    rows = []
    for period_type, label, periods, weeks in (("week", "One week", 17, 1),
                                               ("double_week", "Double week", 3, 2)):
        for split, seasons in (("train", "2018-19,2021-22"),
                               ("validation", "2022-23,2023-24")):
            rows.append({
                "period_type": period_type, "period_label": label, "split": split,
                "seasons": seasons, "n_seasons": 2, "n_periods": periods, "weeks": weeks,
                "n_players": 550, "n_player_seasons": 766, "team_games": 1689,
                "response_label": "dk_pts in one scoring period",
                "sim_draws": 500, "n_sims": 2000, "n_posterior_draws": 1000,
                "fit_window": "train", "ecdf_band_mc": 0.004, "ecdf_band_gated": True,
                "ks": 0.05, "ks_mc": 0.001, "ks_gated": True, "quantile_seed": 1,
                "n": 13022, "mae": 30.0, "rmse": 38.0, "r2": 0.4, "bias": -2.9,
                "crps": 20.0, "observed_mean": 52.7, "observed_sd": 49.0,
                "predicted_mean": 49.8, "point_sd": 28.9, "predictive_sd": 32.8,
                "pooled_sd": 45.3, "zero_share": 0.21, "predicted_zero_share": 0.17})
    return pd.DataFrame(rows)


def test_the_facets_are_ordered_with_the_weekly_one_first():
    """The page is called Weekly scores, so a double week is the exception it declares."""
    index = _weekly_index()
    assert weekly.period_types(index) == ["week", "double_week"]
    assert weekly.period_label(index, "double_week") == "Double week"


def test_an_unknown_facet_sorts_after_the_two_the_page_knows():
    """A new period length must appear rather than vanish — the panels are the contract."""
    index = pd.concat([_weekly_index(),
                       _weekly_index().head(1).assign(period_type="triple_week")])
    assert weekly.period_types(index) == ["week", "double_week", "triple_week"]


def test_the_structure_table_states_that_the_two_facets_are_not_equal_units():
    """The one thing the page owes before any distribution.

    Seventeen one-week periods against three two-week ones: a reader comparing their
    spreads without that is being misled, which is the same argument the model pages make
    about their fitting unit one level up.
    """
    table = weekly.structure(_weekly_index())
    assert list(table["Periods per season"]) == [17, 17, 3, 3]
    assert list(table["Weeks each"]) == [1, 1, 2, 2]
    assert set(table["Split"]) == {"Train", "Validation"}


def test_the_spread_board_compares_the_pooled_sd_and_not_the_point_prediction():
    """**The one substitution that would report a defect that was never measured.**

    `point_sd` is the spread of the per-row posterior means and is narrower than the data
    by construction — a mean over draws has averaged its own noise away. Printed as "the
    simulated spread" it reads as a model far too narrow. `pooled_sd` is the marginal the
    simulator implies over every row and draw, which is the one the observed sd answers.
    """
    board = weekly.spread_board(_weekly_index())
    assert board["Ratio"].iloc[0] == pytest.approx(45.3 / 49.0)
    assert board["Point prediction"].iloc[0] == 28.9
    assert board["Simulated sd (pooled)"].iloc[0] == 45.3


def test_the_gate_board_carries_the_same_metric_names_gate_a_reports():
    board = weekly.gate_board(_weekly_index())
    assert {"MAE", "Bias", "R²", "CRPS"} <= set(board.columns)
    assert len(board) == 4


def test_the_season_total_row_is_read_from_gate_a_rather_than_restated():
    gate = pd.DataFrame({"season": ["2023-24", "2022-23"],
                         "check": ["season_total_dk", "season_total_dk"],
                         "n": [387, 386], "mae": [407.9, 402.1], "bias": [-63.3, -21.9],
                         "r2": [0.659, 0.648], "crps": [281.0, 280.5]})
    table = weekly.season_total_row(gate)
    assert list(table["Season"]) == ["2022-23", "2023-24"]
    assert weekly.season_total_row(None).empty
    assert weekly.season_total_row(gate[gate["check"] == "nothing"]).empty


def _weekly_period() -> pd.DataFrame:
    """Two seasons of unequal size, so a mean of means and a row-weighted mean differ."""
    rows = []
    for season, n, observed in (("2022-23", 100, 40.0), ("2023-24", 300, 60.0)):
        rows.append({"split": "validation", "season": season, "slot": 0,
                     "tournament_round": 1, "period_type": "week",
                     "period_label": "One week", "weeks": 1, "team_games": 50,
                     "start": "2022-10-17", "end": "2022-10-23",
                     "observed_mean": observed, "observed_games": 2.0,
                     "predicted_mean": observed - 2.0, "n": n, "mae": 30.0, "rmse": 38.0,
                     "r2": 0.4, "bias": -2.0, "crps": 20.0})
    return pd.DataFrame(rows)


def test_a_period_point_is_row_weighted_across_its_seasons():
    """A season with more scorable players carries more of the point.

    A mean of means would read 50.0 here; the weighted mean is 55.0, and the emitter ships
    each season's own `n` for exactly this.
    """
    panel = weekly.profile_panel(_weekly_period(), "validation")
    assert panel["observed"].iloc[0] == pytest.approx(55.0)
    assert panel["n"].iloc[0] == 400 and panel["seasons"].iloc[0] == 2


def test_an_absent_split_gives_an_empty_profile_rather_than_raising():
    assert weekly.profile_panel(_weekly_period(), "train").empty


def test_the_period_axis_labels_carry_the_unit_change():
    """`W1`..`W17` then `R2`/`R3`/`R4`: a bare 1..20 would say the last three are weeks."""
    assert weekly.slot_label(0, 1, "week") == "W1"
    assert weekly.slot_label(16, 1, "week") == "W17"
    assert weekly.slot_label(17, 2, "double_week") == "R2"
    assert weekly.slot_label(19, 4, "double_week") == "R4"


def _weekly_ecdf() -> pd.DataFrame:
    grid = np.linspace(0.0, 100.0, 8)
    rows = []
    for period_type in ("week", "double_week"):
        for split in ("train", "validation"):
            for i, value in enumerate(grid):
                share = (i + 1) / len(grid)
                rows.append({"period_type": period_type, "split": split,
                             "grid_index": i, "value": value,
                             "observed": share, "q50": share - 0.02,
                             "q2.5": share - 0.05, "q97.5": share + 0.05,
                             "grid_kind": "quantile", "n_rows": 13022, "n_draws": 500})
    return pd.DataFrame(rows)


def test_the_band_reading_is_a_distance_from_the_median_replicate():
    """The rule `model_cards.band_distance` already carries, at a second unit.

    At these sample sizes the ribbon is one to two ECDF points wide and an honest model
    leaves it somewhere, so in-or-out would report failure everywhere. `inside_95` ships
    beside the distance as a footnote rather than as the reading.
    """
    distance = weekly.band_distance(_weekly_ecdf(), _weekly_index())
    assert len(distance) == 4
    assert distance["max_gap"].max() == pytest.approx(0.02)
    assert (distance["inside_95"] == 1.0).all()


def test_the_calibration_grid_is_one_row_per_facet_and_two_split_columns():
    """`fig_calibration` builds `len(panel_labels)` rows by the two splits.

    The two facets are on different `dk_pts` scales and must not share an axis range;
    train and validation inside a facet must, which is the whole reason they are drawn
    side by side.
    """
    keys = weekly.calibration_keys(_weekly_index())
    assert list(keys) == ["week", "double_week"]
    assert keys["double_week"] == "Double week"


def test_the_caption_threshold_mirrors_the_build_that_enforces_it():
    """A hand-typed bar in a caption is a claim about a build that can move without it.

    The same pinning `model_cards.KS_MC_TOL` gets against the emitter's, one page over.
    """
    from src.models.model_cards import ECDF_BAND_TOL, KS_MC_TOL

    assert weekly.KS_MC_TOL == KS_MC_TOL
    assert weekly.ECDF_BAND_TOL == ECDF_BAND_TOL


def test_the_weekly_page_names_its_own_artifacts_so_the_audit_credits_them():
    """`audit.py` counts an artifact as read when a string literal in `dashboard/` names it.

    Six new files landed in `outputs/predictions/` with this page; if the pure layer held
    them as anything but literals the orphan count would rise and the page would still
    render perfectly.
    """
    from src.sim.weekly import ARTIFACTS

    named = {weekly.INDEX_FILE, weekly.PERIOD_FILE, weekly.ECDF_FILE,
             weekly.CALIBRATION_FILE, weekly.QUANTILE_FILE, weekly.SAMPLE_FILE}
    assert named == set(ARTIFACTS)


def test_the_split_vocabulary_is_the_one_the_model_pages_use():
    assert weekly.SPLITS == model_cards.SPLITS
    assert weekly.SPLIT_LABELS == model_cards.SPLIT_LABELS


def test_the_weekly_page_sits_between_the_inputs_and_the_contest():
    """Position is the argument: heads and their inputs above, the contest that consumes
    the tensor below. The pinned `url_path`s did **not** move when it was inserted, which
    is why they are declared rather than derived from the order."""
    from dashboard import app
    paths = [view.url_path for view in app.VIEWS]
    assert paths.index("weekly") == paths.index("inputs") + 1
    assert paths.index("tournament") == paths.index("weekly") + 1
    assert paths[-1] == "draft-room"


def test_a_legend_over_subplot_titles_is_lifted_clear_of_them():
    """Caught by a rendered PNG, which is the only layer that can see it.

    `apply_theme` puts a horizontal legend at y=1.02 and `make_subplots` writes its titles
    into the same paper-referenced strip, so the ECDF ribbon's five entries ran straight
    through the first subplot's title. Nothing in the trace says so.
    """
    from dashboard.charts import LEGEND_ABOVE_TITLES, fig_ecdf

    panel = _weekly_ecdf()
    panel = panel[(panel["period_type"] == "week") & (panel["split"] == "train")]
    panel = panel.assign(**{"q10": panel["q2.5"], "q90": panel["q97.5"],
                            "q25": panel["q2.5"], "q75": panel["q97.5"]})
    fig = fig_ecdf({"Train": panel, "Validation": panel}, theme.theme("light"), "dk_pts")
    assert fig.layout.legend.y == LEGEND_ABOVE_TITLES > 1.02
    assert fig.layout.margin.t > 48
