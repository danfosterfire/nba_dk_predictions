"""Tests for the league-level season-effect measurements.

Synthetic builders throughout, with league series constructed to have a *known* trend and a
*known* shock, so the decomposition is checked against a right answer rather than against
whatever the real data happens to do.
"""

import numpy as np
import pandas as pd

from src.eda.season_effects import (availability_rates, carry_forward_bias,
                                    drift_vs_shock, league_rates)


def _targets(seasons, per36, minutes_per_game=30.0, games=50, players=4):
    """Player-games whose league FTA/36 is exactly `per36[season]`.

    Every other component is held flat so a test that moves `fta` cannot accidentally be
    reading a different column.
    """
    rows = []
    for season, rate in zip(seasons, per36):
        for p in range(players):
            for g in range(games):
                rows.append({
                    "player_id": 100 + p, "season": season, "season_type": "regular",
                    "played": 1, "game_id": g, "min": minutes_per_game,
                    "fta": rate * minutes_per_game / 36.0,
                    "fg2a": 8.0, "fg3a": 4.0, "reb": 5.0, "ast": 3.0,
                    "stl": 1.0, "blk": 0.5, "tov": 2.0,
                    "ftm": 0.75 * rate * minutes_per_game / 36.0,
                    "fg2m": 4.0, "fg3m": 1.4,
                })
    return pd.DataFrame(rows)


def test_league_rate_is_totals_over_totals_not_a_mean_of_player_rates():
    """A mean of per-player rates moves with roster churn; the league rate must not.

    Two players with very different minutes and very different rates: the league value has
    to be minutes-weighted, so it sits nearer the high-minutes player.
    """
    rows = []
    for g in range(20):
        rows.append({"player_id": 1, "season": "2020-21", "season_type": "regular",
                     "played": 1, "game_id": g, "min": 36.0, "fta": 10.0,
                     "fg2a": 1, "fg3a": 1, "reb": 1, "ast": 1, "stl": 1, "blk": 1,
                     "tov": 1, "ftm": 5.0, "fg2m": 1, "fg3m": 1})
        rows.append({"player_id": 2, "season": "2020-21", "season_type": "regular",
                     "played": 1, "game_id": g, "min": 4.0, "fta": 0.0,
                     "fg2a": 1, "fg3a": 1, "reb": 1, "ast": 1, "stl": 1, "blk": 1,
                     "tov": 1, "ftm": 0.0, "fg2m": 1, "fg3m": 1})
    out = league_rates(pd.DataFrame(rows))
    fta = out[out["quantity"] == "fta"]["rate"].iloc[0]
    # totals-over-totals: 200 FTA over 800 minutes -> 9.0 per 36
    assert np.isclose(fta, 200.0 / 800.0 * 36.0)
    # the unweighted mean of the two players' rates would be 5.0 — materially different
    assert not np.isclose(fta, 5.0)


def test_league_rates_exclude_playoffs_and_unplayed_rows():
    """Pooling playoffs would put a step in every season proportional to how far teams
    advanced, which is a composition artifact rather than a league rate."""
    base = _targets(["2020-21"], [4.0], games=10, players=2)
    playoff = base.assign(season_type="playoffs", fta=base["fta"] * 5)
    absent = base.assign(played=0, fta=base["fta"] * 99)
    out = league_rates(pd.concat([base, playoff, absent], ignore_index=True))
    assert np.isclose(out[out["quantity"] == "fta"]["rate"].iloc[0], 4.0)


def test_a_pure_trend_is_recognised_as_extrapolable():
    seasons = [f"{y}-{str(y + 1)[2:]}" for y in range(2000, 2020)]
    rate = 2.0 * 1.05 ** np.arange(len(seasons))         # exactly +5% a season
    out = drift_vs_shock(league_rates(_targets(seasons, rate)))
    fta = out[out["quantity"] == "fta"].iloc[0]
    assert np.isclose(fta["trend_pct_per_season"], 5.0, atol=0.05)
    assert fta["trend_r2"] > 0.999
    assert bool(fta["trend_worth_extrapolating"])
    assert fta["recommendation"] == "trend + year effect"


def test_a_pure_shock_is_not_mistaken_for_a_trend():
    rng = np.random.default_rng(0)
    seasons = [f"{y}-{str(y + 1)[2:]}" for y in range(2000, 2026)]
    rate = 4.0 * np.exp(rng.normal(0, 0.05, len(seasons)))   # no drift, 5% shocks
    out = drift_vs_shock(league_rates(_targets(seasons, rate)))
    fta = out[out["quantity"] == "fta"].iloc[0]
    assert abs(fta["trend_pct_per_season"]) < 1.0
    assert not bool(fta["trend_worth_extrapolating"])
    assert fta["recommendation"] == "year effect only"
    assert fta["yoy_sd_pct"] > 3.0


def test_a_trend_removes_the_yoy_MEAN_and_leaves_the_yoy_SD_untouched():
    """The identity the whole recommendation rests on.

    `diff(a + b*x)` is the constant `b`, so detrending shifts the mean of the year-over-year
    changes and cannot change their variance. That is why a trend fixed effect and a
    year-level random effect are complementary rather than alternatives — if this were
    false, one of the two would be redundant.
    """
    rng = np.random.default_rng(1)
    n = 30
    y = np.log(3.0) + 0.04 * np.arange(n) + rng.normal(0, 0.05, n)
    fitted = np.polyval(np.polyfit(np.arange(n), y, 1), np.arange(n))

    # The variance is untouched EXACTLY — this is the identity, not an approximation.
    assert np.isclose(np.diff(y).std(ddof=1), np.diff(y - fitted).std(ddof=1))

    # The drift is what goes. Not to exactly zero: the mean of a differenced series
    # telescopes to (r[-1] - r[0])/(n-1), and OLS residuals sum to zero without their
    # endpoints being equal. So the right claim is "essentially all of it".
    raw, detrended = np.diff(y).mean(), np.diff(y - fitted).mean()
    assert abs(raw - 0.04) < 0.02
    assert abs(detrended) < 0.05 * abs(raw)


def test_trend_and_shock_are_reported_separately_when_both_are_present():
    rng = np.random.default_rng(2)
    seasons = [f"{y}-{str(y + 1)[2:]}" for y in range(1996, 2026)]
    rate = 2.0 * 1.04 ** np.arange(len(seasons)) * np.exp(rng.normal(0, 0.06,
                                                                    len(seasons)))
    out = drift_vs_shock(league_rates(_targets(seasons, rate)))
    fta = out[out["quantity"] == "fta"].iloc[0]
    # The trend is recovered despite the noise...
    assert abs(fta["trend_pct_per_season"] - 4.0) < 1.5
    assert bool(fta["trend_worth_extrapolating"])
    # ...and the shock survives it, which is the point.
    assert fta["level_resid_sd_pct"] > 3.0


def test_drift_vs_shock_skips_a_series_too_short_to_fit():
    out = drift_vs_shock(pd.DataFrame({
        "quantity": ["x"] * 3, "season": ["a", "b", "c"], "rate": [1.0, 2.0, 3.0]}))
    assert out.empty


def _design(seasons, rate_by_season, minutes=1000.0, players=30):
    """Player-seasons where every player's realized fta/36 equals the league rate, and the
    lag-1 column is the previous season's rate — so carry-forward's error is exactly the
    league move."""
    rows = []
    prior = dict(zip(seasons[1:], rate_by_season[:-1]))
    for season, rate in zip(seasons, rate_by_season):
        if season not in prior:
            continue
        for p in range(players):
            row = {"player_id": p, "season": season, "total_minutes": minutes}
            for c in ["fg2a", "fg3a", "fta", "reb", "ast", "stl", "blk", "tov"]:
                value = rate if c == "fta" else 5.0
                lag = prior[season] if c == "fta" else 5.0
                row[c] = value * minutes / 36.0
                row[f"{c}_p36_lag1"] = lag
            rows.append(row)
    return pd.DataFrame(rows)


def test_carry_forward_bias_recovers_a_known_league_move():
    """If the league rises 10%, carrying the prior season forward under-predicts by ~9.1%.

    Under-prediction because the bias is expressed against the *realized* total:
    (old − new)/new = (1/1.1) − 1 = −9.09%.
    """
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    rates = [4.0, 4.0, 4.0, 4.0, 4.4]                 # +10% in the final season only
    bias = carry_forward_bias(_design(seasons, rates), test_seasons=1)
    row = bias[(bias["component"] == "fta") & (bias["season"] == "2025-26")].iloc[0]
    assert np.isclose(row["bias_pct"], 100 * (4.0 / 4.4 - 1), atol=0.01)
    assert row["bias_pct"] < 0


def test_carry_forward_bias_is_zero_for_a_flat_league():
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    bias = carry_forward_bias(_design(seasons, [4.0] * 5), test_seasons=1)
    row = bias[(bias["component"] == "fta") & (bias["season"] == "2025-26")].iloc[0]
    assert abs(row["bias_pct"]) < 1e-9


def test_availability_rates_report_overall_and_role_split():
    """The era effect is role-graded, so a single league mean would hide it."""
    rows = []
    for season, star, fringe in [("2004-05", 0.90, 0.45), ("2024-25", 0.75, 0.42)]:
        for p in range(40):
            rows.append({"window": "full", "season": season, "player_id": p,
                         "gp": 60, "minutes_per_game": 34.0, "gp_share": star})
            rows.append({"window": "full", "season": season, "player_id": 100 + p,
                         "gp": 40, "minutes_per_game": 8.0, "gp_share": fringe})
    out = availability_rates(pd.DataFrame(rows))
    star_rows = out[out["role"] == "30+ mpg"].set_index("season")["rate"]
    fringe_rows = out[out["role"] == "<12 mpg"].set_index("season")["rate"]
    assert np.isclose(star_rows["2004-05"], 0.90)
    assert np.isclose(star_rows["2024-25"], 0.75)
    # The star decline (-0.15) is far steeper than the fringe one (-0.03) — the signature.
    assert (star_rows["2004-05"] - star_rows["2024-25"]) > (
        fringe_rows["2004-05"] - fringe_rows["2024-25"])


def test_availability_rates_exclude_the_appearance_window_twin():
    """`availability_features.parquet` carries both roster constructions; mixing them would
    double-count every player-season."""
    rows = []
    for window in ("full", "appearance"):
        for p in range(20):
            rows.append({"window": window, "season": "2020-21", "player_id": p,
                         "gp": 60, "minutes_per_game": 30.0,
                         "gp_share": 0.8 if window == "full" else 0.99})
    out = availability_rates(pd.DataFrame(rows))
    assert np.isclose(out[out["role"] == "all"]["rate"].iloc[0], 0.8)
    assert int(out[out["role"] == "all"]["n"].iloc[0]) == 20
