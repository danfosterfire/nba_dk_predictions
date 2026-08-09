import numpy as np
import pandas as pd

from src.data.preprocess import compute_dk_pts
from src.features.targets import (
    BONUS_CATEGORIES,
    COMPONENTS,
    DK_WEIGHTS,
    SHOT_CLASSES,
    add_shot_classes,
    bonus_calibration,
    bonus_frames,
    bonus_part,
    build_component_targets,
    dk_from_components,
    dk_scoring_from_shot_classes,
    expected_bonus,
    expected_dk_pts,
    linear_part,
    pts_from_shot_classes,
    season_totals,
    zero_bias_overdispersion,
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


# ── Bonus calibration ────────────────────────────────────────────────────────

def _calibration_games(n_players: int = 40, games: int = 60, seed: int = 0,
                       frailty: float = 0.10) -> pd.DataFrame:
    """Player-games drawn from the very model `expected_bonus` assumes.

    Counts are Poisson at `rate x minutes / 36` under a shared per-game Gamma frailty of
    variance `frailty`, so the calibration has a *known* right answer: recovering `frailty`
    from the realized bonus is the whole check, and it cannot be passed by accident.
    """
    rng = np.random.default_rng(seed)
    # Several season labels, assigned round-robin so each player still has exactly one
    # player-season. `bonus_calibration` measures every fit window, and the narrowest —
    # `train` — drops twice `TEST_SEASONS`, so the frame needs more than four season
    # labels or the window is empty and raises.
    seasons = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
    rows = []
    for p in range(n_players):
        mpg = rng.uniform(14.0, 36.0)
        base = np.array([mpg * 0.60, mpg * 0.22, mpg * 0.14, mpg * 0.035, mpg * 0.02])
        g = rng.gamma(1.0 / frailty, frailty, size=games)
        draws = rng.poisson(base[None, :] * g[:, None])
        for i in range(games):
            rows.append({"player_id": p, "season": seasons[p % len(seasons)],
                         "game_id": i,
                         "min": mpg, "fg3m": 1.0,
                         **dict(zip(BONUS_CATEGORIES, draws[i].astype(float)))})
    df = pd.DataFrame(rows)
    df["tov"] = 2.0
    df["dk_bonus"] = bonus_part(df)
    return df


def test_bonus_frames_share_a_population_and_carry_expected_counts():
    df = _calibration_games(n_players=6, games=40)
    season, game = bonus_frames(df, min_season_minutes=200)
    assert len(season) == 6
    assert set(game["player_id"]) == set(season["player_id"])
    for c in BONUS_CATEGORIES:
        assert f"expected_{c}" in season.columns and f"expected_{c}" in game.columns
    # At constant minutes the two units agree row-for-row on the expected counts.
    merged = game.merge(season[["player_id", "expected_pts"]], on="player_id",
                        suffixes=("_game", "_season"))
    assert np.allclose(merged["expected_pts_game"], merged["expected_pts_season"])


def test_the_season_minutes_filter_drops_thin_player_seasons():
    df = _calibration_games(n_players=4, games=6)      # ~150 minutes each
    season, game = bonus_frames(df, min_season_minutes=200)
    assert season.empty and game.empty
    season, _ = bonus_frames(df, min_season_minutes=1)
    assert len(season) == 4


def test_calibration_recovers_the_overdispersion_it_was_generated_with():
    """The check that makes `BONUS_OVERDISPERSION` a measurement, not an instruction."""
    df = _calibration_games(n_players=60, games=70, frailty=0.10, seed=3)
    cal = bonus_calibration(df, min_season_minutes=200,
                            grid=[0.0, 0.05, 0.10, 0.15, 0.20],
                            game_grid=[0.10], n_samples=400)
    fitted = cal[(cal["analysis"] == "fitted") & (cal["unit"] == "player_season")
                 & (cal["fit_window"] == "full")]
    assert len(fitted) == 1
    assert abs(float(fitted["overdispersion"].iloc[0]) - 0.10) < 0.04


def test_independent_sampling_reads_low_and_the_bias_rises_with_overdispersion():
    df = _calibration_games(n_players=40, games=60, frailty=0.10, seed=5)
    cal = bonus_calibration(df, min_season_minutes=200, grid=[0.0, 0.10, 0.30],
                            game_grid=[0.10], n_samples=400)
    # Scoped to one fit window: `bonus_calibration` emits every row under both `full` and
    # `train_val`, so an unscoped count is two of everything and the "exactly one
    # independent row" invariant reads as broken when it is not.
    allrows = cal[(cal["analysis"] == "calibration") & (cal["bucket"] == "all")
                  & (cal["unit"] == "player_season")
                  & (cal["fit_window"] == "full")].sort_values("overdispersion")
    assert allrows["is_independent"].sum() == 1
    assert allrows["is_shipped"].sum() == 1
    assert float(allrows["bias"].iloc[0]) < 0            # independent reads low
    assert list(allrows["bias"]) == sorted(allrows["bias"])


def test_calibration_carries_a_bucket_break_alongside_the_aggregate():
    df = _calibration_games(n_players=30, games=50)
    cal = bonus_calibration(df, min_season_minutes=200, grid=[0.10],
                            game_grid=[0.10], n_samples=200)
    rows = cal[(cal["analysis"] == "calibration") & (cal["fit_window"] == "full")]
    assert (rows["bucket"] == "all").sum() == 2          # one per unit, one window
    assert (rows["bucket"] != "all").sum() > 0
    # Buckets must partition their unit exactly — no row counted twice or dropped.
    for unit in ("player_season", "player_game"):
        sub = rows[(rows["unit"] == unit) & (rows["overdispersion"] == 0.10)]
        total = int(sub[sub["bucket"] == "all"]["n"].iloc[0])
        assert int(sub[sub["bucket"] != "all"]["n"].sum()) == total


def test_zero_bias_overdispersion_returns_none_without_a_sign_change():
    rows = [{"unit": "player_season", "bucket": "all", "overdispersion": od,
             "bias": 0.01, "n": 10, "realized_mean_bonus": 0.1}
            for od in (0.0, 0.1, 0.2)]
    assert zero_bias_overdispersion(rows, "player_season") is None


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
