import numpy as np
import pandas as pd

from src.eda.aging import (
    ALL_ARCHETYPES,
    build_curves,
    build_deltas,
    curve,
    dk_linear_per36,
    metric_cols,
    peak_age,
    prepare,
)

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2010, 2020)]


# ── Synthetic builders ────────────────────────────────────────────────────────

def _matrix(rows: list[dict]) -> pd.DataFrame:
    """A season-matrix-shaped frame with every component column present."""
    df = pd.DataFrame(rows)
    for col in ("bas_pts", "bas_fg3m", "bas_reb", "bas_ast",
                "bas_stl", "bas_blk", "bas_tov"):
        if col not in df:
            df[col] = 0.0
    for col, default in (("min", 30.0), ("gp", 70.0), ("min_total", 2100.0)):
        if col not in df:
            df[col] = default
    return df


def _career(player_id: int, start_age: int, values: list[float],
            first_season: int = 0, minutes: float = 2000.0) -> list[dict]:
    """One player's consecutive seasons, `values` going into bas_pts."""
    return [{"player_id": player_id, "season": SEASONS[first_season + i],
             "age": float(start_age + i), "bas_pts": v, "min_total": minutes}
            for i, v in enumerate(values)]


def _deltas(rows: list[dict]) -> pd.DataFrame:
    """A deltas-shaped frame, the input `curve` consumes."""
    return pd.DataFrame(rows)


# ── DK recombination ──────────────────────────────────────────────────────────

def test_dk_linear_per36_applies_the_dk_weights():
    df = pd.DataFrame([{"bas_pts": 20.0, "bas_fg3m": 2.0, "bas_reb": 8.0, "bas_ast": 4.0,
                        "bas_stl": 1.0, "bas_blk": 0.5, "bas_tov": 3.0}])
    expected = 20 + 0.5 * 2 + 1.25 * 8 + 1.5 * 4 + 2 * 1 + 2 * 0.5 - 0.5 * 3
    assert abs(float(dk_linear_per36(df).iloc[0]) - expected) < 1e-9


def test_prepare_survives_the_games_played_name_collision():
    """`gp` renames onto `games_played`, which the target join already supplies."""
    m = _matrix(_career(1, 25, [20.0]))
    m["games_played"] = 68.0
    prepared = prepare(m)
    assert list(prepared.columns).count("games_played") == 1
    assert prepared["games_played"].iloc[0] == 70.0     # from gp, not the target join
    assert "dk_linear_per36" in prepared
    assert "pts_per36" in metric_cols(prepared)


# ── Delta construction ────────────────────────────────────────────────────────

def test_build_deltas_labels_each_change_with_the_earlier_age():
    prepared = prepare(_matrix(_career(1, 25, [20.0, 24.0, 21.0])))
    d = build_deltas(prepared, SEASONS, metric_cols(prepared))
    assert sorted(d["age"]) == [25.0, 26.0]
    by_age = d.set_index("age")["delta_pts_per36"]
    assert abs(by_age[25.0] - 4.0) < 1e-9
    assert abs(by_age[26.0] - (-3.0)) < 1e-9


def test_build_deltas_floors_a_fractional_age():
    prepared = prepare(_matrix(_career(1, 25, [20.0, 22.0])))
    prepared.loc[0, "age"] = 25.9
    assert build_deltas(prepared, SEASONS, metric_cols(prepared))["age"].iloc[0] == 25.0


# ── Era absorption ────────────────────────────────────────────────────────────

def test_era_adjustment_removes_a_league_wide_gain_that_hits_every_age():
    """Everyone improves +2 a season regardless of age; the aging shape is flat."""
    rows = []
    for p in range(40):
        rows.append({"age": 22.0 + (p % 12), "season": "s1", "player_id": p,
                     "weight": 1.0, "delta_x": 2.0, "level_x": 20.0})
    out = curve(_deltas(rows), "x", 22, 33, min_players=1, anchor_age=23)
    assert np.allclose(out["mean_delta"], 2.0)
    assert np.allclose(out["mean_delta_era_adj"], 0.0, atol=1e-9)


def test_era_adjustment_keeps_the_age_shape_it_should_not_remove():
    """Same league-wide +2, but with a real age gradient underneath it."""
    rows = []
    for p in range(120):
        age = 22.0 + (p % 12)
        rows.append({"age": age, "season": "s1", "player_id": p, "weight": 1.0,
                     "delta_x": 2.0 + (27.0 - age) * 0.5, "level_x": 20.0})
    out = curve(_deltas(rows), "x", 22, 33, min_players=1, anchor_age=23).set_index("age")
    # young players still improve faster than old ones after the era is absorbed
    assert out.loc[22, "mean_delta_era_adj"] > out.loc[32, "mean_delta_era_adj"]
    assert abs((out.loc[22, "mean_delta_era_adj"] - out.loc[23, "mean_delta_era_adj"])
               - 0.5) < 1e-9


# ── Integrating the curve ─────────────────────────────────────────────────────

def _peaked_deltas(peak: int = 27, n_per_age: int = 40) -> pd.DataFrame:
    """+1 a year up to `peak`, -1 a year after it."""
    rows = []
    for age in range(20, 37):
        for p in range(n_per_age):
            rows.append({"age": float(age), "season": f"s{age % 3}",
                         "player_id": age * 1000 + p, "weight": 1.0,
                         "delta_x": 1.0 if age < peak else -1.0, "level_x": 30.0})
    return pd.DataFrame(rows)


def test_cumulative_curve_peaks_where_the_deltas_change_sign():
    out = curve(_peaked_deltas(peak=27), "x", 20, 36, min_players=1, anchor_age=23)
    assert float(out.loc[out["cumulative"].idxmax(), "age"]) == 27.0


def test_cumulative_is_zero_at_the_anchor_age():
    out = curve(_peaked_deltas(), "x", 20, 36, min_players=1, anchor_age=23)
    assert abs(float(out.set_index("age").loc[23, "cumulative"])) < 1e-9


def test_cumulative_ratio_is_one_at_the_anchor_age():
    out = curve(_peaked_deltas(), "x", 20, 36, min_players=1, anchor_age=23)
    assert abs(float(out.set_index("age").loc[23, "cumulative_ratio"]) - 1.0) < 1e-9


def test_min_players_per_age_drops_thin_ages():
    d = _peaked_deltas(n_per_age=40)
    d = d[~((d["age"] == 30.0) & (d.groupby("age").cumcount() >= 3))]
    out = curve(d, "x", 20, 36, min_players=25, anchor_age=23)
    assert 30.0 not in set(out["age"])
    assert 29.0 in set(out["age"])


# ── The bias the delta method exists to avoid ─────────────────────────────────

def test_a_decline_shared_by_every_age_is_absorbed_as_era_drift():
    """A property worth knowing, not a bug: the era-adjusted curve is *relative*.

    If literally everyone declines at the same rate at every age, that is
    observationally identical to the league deflating, and `mean_delta_era_adj`
    reports zero. The absolute movement is still in `mean_delta`.
    """
    rows = [{"age": 24.0 + (p % 12), "season": "s1", "player_id": p, "weight": 1.0,
             "delta_x": -1.0, "level_x": 30.0} for p in range(120)]
    out = curve(pd.DataFrame(rows), "x", 24, 35, min_players=1, anchor_age=24)
    assert np.allclose(out["mean_delta"], -1.0)
    assert np.allclose(out["mean_delta_era_adj"], 0.0, atol=1e-9)


def test_delta_method_declines_where_the_cross_section_rises():
    """Survivorship, planted: decline accelerates with age, but only the best survive.

    The cross-sectional mean climbs with age purely because weak players have left,
    while the integrated delta curve falls. That gap is the whole reason this module
    does not just group by age.
    """
    rows = []
    for p in range(200):
        ability = float(p)                    # p=199 is the best player
        # A player survives past 28 only if he is in the top quarter.
        last_age = 36 if p >= 150 else 28
        for age in range(24, last_age + 1):
            decline = -0.2 * (age - 23)       # accelerating with age
            rows.append({"age": float(age), "season": f"s{age % 4}", "player_id": p,
                         "weight": 1.0, "delta_x": decline,
                         "level_x": ability - 1.0 * (age - 24)})
    out = curve(pd.DataFrame(rows), "x", 24, 35, min_players=1,
                anchor_age=24).set_index("age")

    assert out.loc[35, "cross_sectional_mean"] > out.loc[24, "cross_sectional_mean"]
    assert out.loc[35, "cumulative"] < out.loc[24, "cumulative"]
    # The era-adjusted deltas are relative to the panel's average change, so the
    # gradient — not the level — is what carries the accelerating decline. The curve
    # is stepped rather than smooth because each synthetic season absorbs its own
    # mean, hence the tolerance.
    assert (np.diff(out["mean_delta_era_adj"].to_numpy()) <= 1e-9).all()


# ── Stratifying by archetype ──────────────────────────────────────────────────

def test_build_curves_emits_an_overall_row_plus_one_per_archetype():
    d = _peaked_deltas(peak=27, n_per_age=30)
    d["archetype"] = (d["player_id"] % 2).astype(int)
    d["archetype_name"] = np.where(d["archetype"] == 0, "bigs", "guards")

    out = build_curves(d, ["x"], 20, 36, min_players=1, anchor_age=23)
    assert set(out["archetype_name"]) == {ALL_ARCHETYPES, "bigs", "guards"}
    assert peak_age(out, "x") == 27.0
    assert peak_age(out, "x", "guards") == 27.0
