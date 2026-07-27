import numpy as np
import pandas as pd

from src.data.preprocess import compute_dk_pts
from src.features.targets import (
    BONUS_CATEGORIES,
    COMPONENTS,
    DK_WEIGHTS,
    SHOT_CLASSES,
    add_shot_classes,
    bonus_part,
    build_component_targets,
    dk_from_components,
    dk_scoring_from_shot_classes,
    expected_bonus,
    expected_dk_pts,
    linear_part,
    pts_from_shot_classes,
    season_totals,
)


def _games(n: int = 4, **overrides) -> pd.DataFrame:
    base = {"pts": 20.0, "reb": 5.0, "ast": 4.0, "stl": 1.0, "blk": 0.5,
            "tov": 2.0, "fg3m": 2.0, "min": 30.0}
    df = pd.DataFrame({k: [v] * n for k, v in base.items()})
    for k, v in overrides.items():
        df[k] = v
    df["player_id"] = 1
    df["season"] = "2023-24"
    df["game_id"] = range(n)
    return df


# ── Recombination matches the single source of truth ─────────────────────────

def test_dk_from_components_matches_compute_dk_pts():
    df = _games(6, pts=[20.0, 9.0, 11.0, 12.0, 30.0, 0.0],
                reb=[5.0, 10.0, 11.0, 12.0, 10.0, 0.0],
                ast=[4.0, 10.0, 3.0, 11.0, 10.0, 0.0],
                stl=[1.0, 1.0, 10.0, 10.0, 2.0, 0.0],
                blk=[0.5, 0.0, 1.0, 10.0, 1.0, 0.0])
    assert np.allclose(dk_from_components(df), compute_dk_pts(df))


def test_linear_part_uses_the_documented_weights():
    df = _games(1)
    expected = sum(df[c].iloc[0] * w for c, w in DK_WEIGHTS.items())
    assert abs(linear_part(df).iloc[0] - expected) < 1e-12
    # 20 + 2(0.5) + 5(1.25) + 4(1.5) + 1(2) + 0.5(2) + 2(-0.5)
    #  = 20 + 1 + 6.25 + 6 + 2 + 1 - 1
    assert abs(linear_part(df).iloc[0] - 35.25) < 1e-12


def test_bonus_part_thresholds():
    df = _games(5,
                pts=[9.0, 10.0, 10.0, 10.0, 10.0],
                reb=[9.0, 9.0, 10.0, 10.0, 10.0],
                ast=[9.0, 9.0, 9.0, 10.0, 10.0],
                stl=[0.0, 0.0, 0.0, 0.0, 10.0],
                blk=[0.0, 0.0, 0.0, 0.0, 10.0])
    assert bonus_part(df).tolist() == [0.0, 0.0, 1.5, 4.5, 4.5]


def test_bonus_ignores_non_bonus_categories():
    """Turnovers and threes do not count toward a double-double."""
    df = _games(1, pts=10.0, reb=10.0, tov=15.0, fg3m=12.0)
    assert bonus_part(df).iloc[0] == 1.5
    assert set(BONUS_CATEGORIES) == {"pts", "reb", "ast", "stl", "blk"}


# ── Target construction ──────────────────────────────────────────────────────

def test_build_component_targets_adds_rates_and_played_flag():
    df = build_component_targets(_games(2, min=[36.0, 18.0], pts=[20.0, 10.0]))
    assert df["played"].tolist() == [1, 1]
    # at exactly 36 minutes the per-36 rate equals the raw count
    assert abs(df["pts_per36"].iloc[0] - 20.0) < 1e-12
    # at 18 minutes it doubles
    assert abs(df["pts_per36"].iloc[1] - 20.0) < 1e-12
    assert np.allclose(df["dk_linear"] + df["dk_bonus"], df["dk_pts"])


def test_zero_minute_games_get_no_rate_rather_than_a_zero_rate():
    """A DNP has no rate; filling 0 would train the rate heads toward zero."""
    df = build_component_targets(_games(2, min=[30.0, 0.0], pts=[20.0, 0.0]))
    assert df["played"].tolist() == [1, 0]
    assert df["pts_per36"].notna().iloc[0]
    assert pd.isna(df["pts_per36"].iloc[1])


def test_season_totals_sums_components_and_counts_games():
    df = build_component_targets(_games(10))
    out = season_totals(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["games_played"] == 10
    assert abs(row["pts_sum"] - 200.0) < 1e-9
    assert abs(row["min_sum"] - 300.0) < 1e-9


# ── Expected bonus ───────────────────────────────────────────────────────────

def test_expected_bonus_is_near_zero_for_a_low_volume_player():
    lam = np.array([[8.0, 2.0, 1.5, 0.6, 0.3]])
    assert expected_bonus(lam, n_samples=4000, seed=0)[0] < 0.10


def test_expected_bonus_is_large_for_a_triple_double_averager():
    lam = np.array([[27.0, 12.0, 11.0, 1.5, 0.8]])
    assert expected_bonus(lam, n_samples=4000, seed=0)[0] > 2.0


def test_expected_bonus_increases_with_expected_counts():
    lam = np.array([[10.0, 5.0, 4.0, 1.0, 0.5],
                    [20.0, 8.0, 6.0, 1.0, 0.5],
                    [28.0, 12.0, 11.0, 2.0, 1.0]])
    out = expected_bonus(lam, n_samples=4000, seed=0)
    assert out[0] < out[1] < out[2]


def test_expected_bonus_is_bounded_by_the_payout():
    lam = np.array([[60.0, 40.0, 30.0, 20.0, 15.0]])
    assert 4.0 < expected_bonus(lam, n_samples=2000, seed=0)[0] <= 4.5


def test_overdispersion_raises_the_expected_bonus():
    """Positive correlation between components makes multi-category games likelier."""
    lam = np.array([[20.0, 9.0, 8.0, 1.5, 0.7]])
    independent = expected_bonus(lam, overdispersion=1e-9, n_samples=8000, seed=1)[0]
    correlated = expected_bonus(lam, overdispersion=0.10, n_samples=8000, seed=1)[0]
    assert correlated > independent


def test_expected_bonus_rejects_wrong_shape():
    try:
        expected_bonus(np.zeros((3, 4)))
        raise AssertionError("expected ValueError for the wrong column count")
    except ValueError as exc:
        assert "expected (n, 5)" in str(exc)


def test_expected_bonus_chunking_does_not_change_results():
    rng = np.random.default_rng(0)
    lam = np.abs(rng.normal(8, 4, size=(50, 5)))
    a = expected_bonus(lam, n_samples=512, seed=5, chunk=1000)
    b = expected_bonus(lam, n_samples=512, seed=5, chunk=7)
    # different chunkings consume the RNG differently, so compare in aggregate
    assert abs(a.mean() - b.mean()) < 0.05


# ── Full expectation ─────────────────────────────────────────────────────────

def test_expected_dk_pts_scales_with_minutes():
    rates = pd.DataFrame({c: [v] for c, v in
                          {"pts": 22.0, "fg3m": 2.0, "reb": 6.0, "ast": 4.0,
                           "stl": 1.0, "blk": 0.5, "tov": 2.0}.items()})
    low = expected_dk_pts(np.array([18.0]), rates, n_samples=2000, seed=0)[0]
    high = expected_dk_pts(np.array([36.0]), rates, n_samples=2000, seed=0)[0]
    assert high > low
    # the linear part is exactly proportional; only the bonus breaks proportionality
    assert high < 2.5 * low


def test_expected_dk_pts_scales_by_availability():
    rates = pd.DataFrame({c: [10.0] for c in COMPONENTS})
    full = expected_dk_pts(np.array([30.0]), rates, n_samples=1000, seed=0)[0]
    half = expected_dk_pts(np.array([30.0]), rates, play_prob=np.array([0.5]),
                           n_samples=1000, seed=0)[0]
    assert abs(half - full * 0.5) < 1e-9


# ── Shot classes: the pts decomposition ──────────────────────────────────────

def _shot_games(n: int = 4, **overrides) -> pd.DataFrame:
    """Games whose shot columns are internally consistent with `pts`.

    FGM/FGA include threes, so 7 field goals here means 5 twos and 2 threes:
    2*7 + 2 + 4 = 20 points.
    """
    base = {"pts": 20.0, "fgm": 7.0, "fga": 15.0, "fg3m": 2.0, "fg3a": 5.0,
            "ftm": 4.0, "fta": 5.0, "reb": 5.0, "ast": 4.0, "stl": 1.0,
            "blk": 0.5, "tov": 2.0, "min": 30.0}
    df = pd.DataFrame({k: [v] * n for k, v in base.items()})
    for k, v in overrides.items():
        df[k] = v
    df["player_id"] = 1
    df["season"] = "2023-24"
    df["game_id"] = range(n)
    return df


def test_add_shot_classes_subtracts_threes_out_of_field_goals():
    df = add_shot_classes(_shot_games(1))
    assert df["fg2m"].iloc[0] == 5.0      # 7 field goals, 2 of them threes
    assert df["fg2a"].iloc[0] == 10.0     # 15 attempts, 5 of them from three


def test_the_pts_identity_holds_in_both_bases():
    """`2*fgm + fg3m + ftm` and `2*fg2m + 3*fg3m + ftm` are the same expression."""
    df = add_shot_classes(_shot_games(5, pts=[20.0, 3.0, 31.0, 0.0, 12.0],
                                      fgm=[7.0, 1.0, 11.0, 0.0, 5.0],
                                      fg3m=[2.0, 1.0, 5.0, 0.0, 2.0],
                                      ftm=[4.0, 0.0, 4.0, 0.0, 0.0]))
    stored = 2 * df["fgm"] + df["fg3m"] + df["ftm"]
    assert np.allclose(stored, df["pts"])
    assert np.allclose(pts_from_shot_classes(df), df["pts"])


def test_the_tempting_wrong_form_double_counts_every_three():
    """`2*fgm + 3*fg3m + ftm` pays a made three 2 + 3 = 5, and only agrees at fg3m = 0."""
    df = add_shot_classes(_shot_games(2, fg3m=[2.0, 0.0], fgm=[7.0, 8.0],
                                      pts=[20.0, 20.0], ftm=[4.0, 4.0]))
    wrong = 2 * df["fgm"] + 3 * df["fg3m"] + df["ftm"]
    assert wrong.iloc[0] == df["pts"].iloc[0] + 2 * df["fg3m"].iloc[0]   # overstated
    assert wrong.iloc[1] == df["pts"].iloc[1]                            # no threes


def test_add_shot_classes_is_a_no_op_when_the_source_columns_are_absent():
    df = add_shot_classes(pd.DataFrame({"pts": [10.0]}))
    assert "fg2m" not in df and "fg2a" not in df


def test_shot_classes_stay_out_of_the_dk_sum():
    """They decompose `pts`; adding them to DK_WEIGHTS would double-count it."""
    for c in SHOT_CLASSES:
        assert c not in DK_WEIGHTS or c == "fg3m"     # fg3m earns its own 0.5
    df = add_shot_classes(_shot_games(3))
    assert np.allclose(dk_from_components(df), compute_dk_pts(df))


def test_dk_scoring_from_shot_classes_matches_the_dk_scoring_terms():
    """A three is worth 3.5 in this basis: 3 points plus DK's 0.5 bonus."""
    df = add_shot_classes(_shot_games(4, pts=[20.0, 3.0, 31.0, 12.0],
                                      fgm=[7.0, 1.0, 11.0, 5.0],
                                      fg3m=[2.0, 1.0, 5.0, 2.0],
                                      ftm=[4.0, 0.0, 4.0, 0.0]))
    expected = 1.0 * df["pts"] + 0.5 * df["fg3m"]
    assert np.allclose(dk_scoring_from_shot_classes(df), expected)


def test_build_component_targets_emits_shot_class_columns_and_rates():
    out = build_component_targets(_shot_games(2, min=[36.0, 18.0]))
    for c in SHOT_CLASSES:
        assert c in out and f"{c}_per36" in out
    # 5 twos in 18 minutes is 10 per 36
    assert abs(out["fg2m_per36"].iloc[1] - 10.0) < 1e-9
    assert np.allclose(pts_from_shot_classes(out), out["pts"])
