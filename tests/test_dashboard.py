"""The dashboard's pure layer: palette rules and figure builders.

`dashboard/app.py` guards its rendering behind `if __name__ == "__main__"`, which
`streamlit run` satisfies and an import does not — so the builders can be exercised
here without a Streamlit runtime. Nothing below touches a real artifact; the point
is that the colour rules the dataviz method fixes are enforced in code rather than
maintained by hand.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent


def _app():
    spec = importlib.util.spec_from_file_location("dashboard_app",
                                                  ROOT / "dashboard" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = _app()


# ── Synthetic builders ────────────────────────────────────────────────────────

def _bars_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({"feature": [f"f{i}" for i in range(n)],
                         "a": np.linspace(0.1, 0.9, n),
                         "b": np.linspace(0.9, 0.1, n)})


def _scatter_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "pc1": rng.normal(size=n), "pc2": rng.normal(size=n),
        "age": rng.integers(20, 38, n).astype(float),
        "archetype_name": rng.choice(list("abcdefghi"), n),
        "player_name": [f"P{i}" for i in range(n)],
    })


# ── Palette rules ─────────────────────────────────────────────────────────────

def test_both_modes_supply_the_same_eight_slots_in_a_fixed_order():
    light, dark = app.theme("light"), app.theme("dark")
    assert len(light["series"]) == len(dark["series"]) == 8
    # slot order is the colour-blind-safety mechanism, so it must not be re-sorted
    assert light["series"][0] == "#2a78d6" and light["series"][1] == "#eb6834"
    assert dark["series"][0] == "#3987e5" and dark["series"][1] == "#d95926"


def test_the_all_pairs_cap_is_three():
    """Only the first three slots clear the CVD floors on every pair."""
    assert app.ALL_PAIRS_CAP == 3


def test_sequential_ramp_is_one_hue_and_reverses_for_the_dark_surface():
    light = [c for _, c in app.theme("light")["sequential"]]
    dark = [c for _, c in app.theme("dark")["sequential"]]
    assert light == app.BLUE_RAMP
    assert dark == list(reversed(app.BLUE_RAMP))
    # near-zero recedes toward the surface in both modes
    assert light[0] == "#cde2fb" and dark[0] == "#0d366b"


def test_diverging_scale_is_two_hues_with_a_neutral_midpoint():
    for mode in ("light", "dark"):
        th = app.theme(mode)
        stops = th["diverging"]
        assert [s[0] for s in stops] == [0.0, 0.5, 1.0]      # equal arms
        assert stops[1][1] == th["neutral"]                   # gray, not a hue
        assert stops[0][1] in th["series"] and stops[2][1] in th["series"]


def test_ordinal_ramp_stays_clear_of_the_surface():
    """An ordinal step must not sink into the background at either end."""
    assert "#cde2fb" not in app.theme("light")["ordinal"]     # lighter than step 250
    assert "#0d366b" not in app.theme("dark")["ordinal"]      # darker than step 600


def test_ordinal_colors_are_distinct_and_count_matched():
    for n in (2, 5, 10):
        colors = app.ordinal_colors(app.theme("light"), n)
        assert len(colors) == n
        assert len(set(colors)) == n


# ── Chrome ────────────────────────────────────────────────────────────────────

def test_apply_theme_pins_the_validated_surface_and_solid_hairlines():
    th = app.theme("dark")
    fig = app.apply_theme(go.Figure(), th)
    assert fig.layout.paper_bgcolor == th["surface"] == "#1a1a19"
    assert fig.layout.plot_bgcolor == th["surface"]
    # dashed grid reads as "threshold" when it is only a grid
    assert fig.layout.xaxis.griddash == "solid"
    assert fig.layout.yaxis.gridcolor == th["grid"]


def test_light_mode_pins_the_surface_the_palette_was_validated_against():
    assert app.theme("light")["surface"] == "#fcfcfb"


# ── Figure builders ───────────────────────────────────────────────────────────

def test_bars_assign_slots_in_order_and_separate_fills_with_the_surface():
    th = app.theme("light")
    fig = app.fig_bars(_bars_frame(), "feature", ["a", "b"], th, "t")
    assert len(fig.data) == 2
    assert fig.data[0].marker.color == th["series"][0]
    assert fig.data[1].marker.color == th["series"][1]
    # a 2px surface gap, not a border drawn around the marks
    assert all(t.marker.line.color == th["surface"] and t.marker.line.width == 2
               for t in fig.data)
    assert fig.layout.showlegend is True


def test_a_single_bar_series_carries_no_legend_box():
    fig = app.fig_bars(_bars_frame(), "feature", ["a"], app.theme("light"), "t")
    assert fig.layout.showlegend is False


def test_emphasis_paints_one_category_and_mutes_the_rest():
    th = app.theme("light")
    df = _bars_frame(4)
    fig = app.fig_bars(df, "feature", ["a"], th, "t", emphasis="f2")
    colors = list(fig.data[0].marker.color)
    assert colors == [th["muted"], th["muted"], th["series"][0], th["muted"]]


def test_scatter_emphasis_never_exceeds_the_all_pairs_cap():
    th = app.theme("light")
    df = _scatter_frame()
    fig = app.fig_scatter(df, "pc1", "pc2", th, "t", color_by="archetype_name",
                          highlight=list("abcdefghi"))
    highlighted = [t for t in fig.data if t.name != "everything else"]
    assert len(highlighted) == app.ALL_PAIRS_CAP
    assert [t.marker.color for t in highlighted] == th["series"][:app.ALL_PAIRS_CAP]


def test_scatter_mutes_the_unhighlighted_population():
    th = app.theme("light")
    fig = app.fig_scatter(_scatter_frame(), "pc1", "pc2", th, "t",
                          color_by="archetype_name", highlight=["a"])
    rest = next(t for t in fig.data if t.name == "everything else")
    assert rest.marker.color == th["muted"]
    assert rest.marker.opacity < 0.5


def test_a_continuous_key_uses_the_sequential_ramp_and_has_no_pair_limit():
    th = app.theme("light")
    fig = app.fig_scatter(_scatter_frame(), "pc1", "pc2", th, "t", color_by="age",
                          continuous=True)
    assert len(fig.data) == 1
    assert list(fig.data[0].marker.colorscale) == [
        (s[0], s[1]) for s in th["sequential"]]
    assert fig.data[0].marker.showscale is True


def test_a_single_series_scatter_uses_slot_one_and_no_legend():
    th = app.theme("light")
    fig = app.fig_scatter(_scatter_frame(), "pc1", "pc2", th, "t")
    assert fig.data[0].marker.color == th["series"][0]
    assert fig.layout.showlegend is False


def test_heatmap_picks_sequential_for_magnitude_and_diverging_for_polarity():
    th = app.theme("light")
    z = pd.DataFrame(np.random.default_rng(0).normal(size=(4, 5)),
                     index=list("abcd"), columns=list("vwxyz"))
    mag = app.fig_heatmap(z, th, "t", "c")
    pol = app.fig_heatmap(z, th, "t", "c", diverging=True, zmid=0.0)
    assert list(mag.data[0].colorscale) == [(s[0], s[1]) for s in th["sequential"]]
    assert list(pol.data[0].colorscale) == [(s[0], s[1]) for s in th["diverging"]]
    assert pol.data[0].zmid == 0.0
    assert mag.data[0].zmid is None


def test_lines_are_two_px_with_markers_ringed_in_the_surface():
    th = app.theme("dark")
    df = pd.DataFrame({"age": range(20, 30), "a": range(10), "b": range(10, 20)})
    fig = app.fig_lines(df, "age", {"a": "A", "b": "B"}, th, "t")
    assert len(fig.data) == 2
    for i, trace in enumerate(fig.data):
        assert trace.line.width == 2
        assert trace.marker.size >= 8
        assert trace.marker.line.color == th["surface"]
        assert trace.line.color == th["series"][i]


def test_lines_direct_label_the_endpoint_rather_than_every_point():
    th = app.theme("light")
    df = pd.DataFrame({"age": range(20, 30), "a": range(10)})
    fig = app.fig_lines(df, "age", {"a": "A"}, th, "t", label_last=True)
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "A"


def test_lines_skip_a_series_the_frame_does_not_carry():
    th = app.theme("light")
    df = pd.DataFrame({"age": range(5), "a": range(5)})
    fig = app.fig_lines(df, "age", {"a": "A", "missing": "M"}, th, "t")
    assert len(fig.data) == 1


# ── Wiring ────────────────────────────────────────────────────────────────────

def test_the_module_declares_nine_tabs():
    assert len(app.TABS) == 9
    assert app.TABS[0] == "Coverage" and app.TABS[-1] == "Feature diagnostics"


def test_every_tab_name_has_a_renderer():
    renderers = ["tab_coverage", "tab_pca", "tab_archetypes", "tab_persistence",
                 "tab_aging", "tab_target", "tab_team_context", "tab_opponent",
                 "tab_feature_diagnostics"]
    assert len(renderers) == len(app.TABS)
    for name in renderers:
        assert callable(getattr(app, name))


def test_the_repo_root_is_on_the_path_for_src_imports():
    assert str(ROOT) == str(app.ROOT)
