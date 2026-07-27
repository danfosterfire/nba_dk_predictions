import numpy as np
import pandas as pd

from src.eda.persistence import (
    demean_within,
    fit_reliability,
    is_per36,
    lagged_pairs,
    measure,
    minutes_for_reliability,
    pair_weights,
    persistence_row,
    weighted_corr,
)

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23"]


# ── Synthetic builders ────────────────────────────────────────────────────────

def _frame(rows: list[dict]) -> pd.DataFrame:
    """A minimal season-matrix-shaped frame: identity, volume, and stat columns."""
    df = pd.DataFrame(rows)
    df["season_start_year"] = [int(s.split("-")[0]) for s in df["season"]]
    return df


def _drifting_frame(n_players: int = 60) -> pd.DataFrame:
    """A stat with zero year-over-year signal that nonetheless rises every season.

    Each player's value is his own rank scrambled independently each year — no
    persistence at all — plus a large league-wide trend. Pooled, the trend makes
    consecutive seasons correlate; within season, nothing is left.
    """
    rng = np.random.default_rng(7)
    rows = []
    for si, season in enumerate(SEASONS):
        for p in range(n_players):
            rows.append({
                "player_id": p, "season": season, "min_total": 1500.0,
                "bas_pts": 10.0 * si + rng.normal(),      # era drift, no persistence
                "adv_usg_pct": 0.2 + 0.001 * p,           # constant per player
            })
    return _frame(rows)


# ── Column classification ─────────────────────────────────────────────────────

def test_is_per36_reads_through_the_family_prefix():
    for col in ["bas_pts", "bas_fg3a", "def_dreb", "sl_restricted_area_fga",
                "clu_pts", "misc_pts_paint"]:
        assert is_per36(col), col
    for col in ["adv_usg_pct", "adv_ast_ratio", "def_def_rating", "adv_pace",
                "bio_player_height_inches", "bio_draft_number", "spd_avg_speed"]:
        assert not is_per36(col), col


# ── Weighted statistics ───────────────────────────────────────────────────────

def test_weighted_corr_matches_numpy_when_unweighted():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.6 * x + rng.normal(size=200)
    assert abs(weighted_corr(x, y) - np.corrcoef(x, y)[0, 1]) < 1e-12


def test_weighted_corr_follows_the_weights():
    """Two subpopulations with opposite signs — the weights decide which one wins."""
    x = np.array([-2.0, -1.0, 1.0, 2.0, -2.0, -1.0, 1.0, 2.0])
    y = np.array([-2.0, -1.0, 1.0, 2.0, 2.0, 1.0, -1.0, -2.0])
    heavy_positive = np.array([10.0] * 4 + [1.0] * 4)
    heavy_negative = np.array([1.0] * 4 + [10.0] * 4)
    assert weighted_corr(x, y, heavy_positive) > 0.7
    assert weighted_corr(x, y, heavy_negative) < -0.7


def test_weighted_corr_is_nan_when_a_side_is_constant():
    assert np.isnan(weighted_corr(np.ones(50), np.arange(50.0)))


def test_demean_within_removes_group_means():
    v = np.array([1.0, 3.0, 10.0, 20.0])
    g = np.array(["a", "a", "b", "b"])
    assert np.allclose(demean_within(v, g), [-1.0, 1.0, -5.0, 5.0])


def test_demean_within_uses_the_weighted_group_mean():
    v = np.array([0.0, 10.0])
    g = np.array(["a", "a"])
    w = np.array([9.0, 1.0])          # weighted mean is 1.0, not 5.0
    assert np.allclose(demean_within(v, g, w), [-1.0, 9.0])


def test_pair_weights_are_dominated_by_the_worse_measured_season():
    """2,400 minutes paired with 90 must not weigh like two full seasons."""
    w = pair_weights([2400.0, 2400.0], [2400.0, 90.0])
    assert w[0] > 10 * w[1]
    # harmonic mean, then normalized to mean 1
    raw = np.array([2400.0, 2 * 2400 * 90 / (2400 + 90)])
    assert np.allclose(w, raw / raw.mean())


def test_pair_weights_zero_out_a_missing_season():
    assert pair_weights([1000.0, 0.0], [1000.0, 1000.0])[1] == 0.0


# ── Season pairing ────────────────────────────────────────────────────────────

def test_lagged_pairs_never_pairs_across_a_missing_season():
    frame = _frame([
        {"player_id": 1, "season": "2019-20", "min_total": 100.0, "bas_pts": 1.0},
        {"player_id": 1, "season": "2021-22", "min_total": 100.0, "bas_pts": 2.0},
    ])
    assert len(lagged_pairs(frame, SEASONS, lag=1)) == 0
    lag2 = lagged_pairs(frame, SEASONS, lag=2)
    assert len(lag2) == 1
    assert lag2.iloc[0]["season"] == "2019-20"
    assert lag2.iloc[0]["season_next"] == "2021-22"


def test_lagged_pairs_chains_consecutive_seasons():
    frame = _frame([{"player_id": 1, "season": s, "min_total": 100.0, "bas_pts": float(i)}
                    for i, s in enumerate(SEASONS)])
    pairs = lagged_pairs(frame, SEASONS, lag=1)
    assert len(pairs) == 3
    assert list(pairs["bas_pts"]) == [0.0, 1.0, 2.0]
    assert list(pairs["bas_pts_next"]) == [1.0, 2.0, 3.0]


# ── The correction that matters ───────────────────────────────────────────────

def test_pooled_persistence_reads_era_drift_that_within_season_does_not():
    """The finding that motivated absorbing season, reproduced on planted data."""
    pairs = lagged_pairs(_drifting_frame(), SEASONS, lag=1)
    row = persistence_row(pairs, "bas_pts", weighted=True)
    assert row["r_pooled"] > 0.9            # the trend, mistaken for persistence
    assert abs(row["r_within_season"]) < 0.2  # nothing left once the era is absorbed


def test_a_genuinely_stable_stat_survives_season_absorption():
    pairs = lagged_pairs(_drifting_frame(), SEASONS, lag=1)
    row = persistence_row(pairs, "adv_usg_pct", weighted=False)
    assert row["r_pooled"] > 0.99 and row["r_within_season"] > 0.99


# ── Reliability curve ─────────────────────────────────────────────────────────

def test_fit_reliability_recovers_a_planted_curve():
    minutes = np.array([50.0, 150.0, 400.0, 800.0, 1500.0, 2500.0])
    r_inf, m0 = 0.90, 66.0
    by_bin = pd.DataFrame({"mean_minutes": minutes, "n": np.full(len(minutes), 500.0),
                           "r": r_inf * minutes / (minutes + m0)})
    fit_inf, fit_m0 = fit_reliability(by_bin)
    assert abs(fit_inf - r_inf) < 0.02
    assert abs(fit_m0 - m0) < 10.0


def test_fit_reliability_declines_with_too_few_bins():
    by_bin = pd.DataFrame({"mean_minutes": [100.0, 900.0], "n": [50.0, 50.0],
                           "r": [0.5, 0.9]})
    assert np.isnan(fit_reliability(by_bin)[0])


def test_minutes_for_reliability_inverts_the_curve():
    m = minutes_for_reliability(0.924, 66.0, target=0.75)
    assert abs(0.924 * m / (m + 66.0) - 0.75) < 1e-9
    # a stat whose ceiling is below the target never reaches it, at any minutes
    assert np.isnan(minutes_for_reliability(0.60, 66.0, target=0.75))


# ── End to end ────────────────────────────────────────────────────────────────

def test_measure_ranks_the_stable_stat_above_the_drifting_one():
    table = measure(_drifting_frame(), SEASONS, min_pairs=10, max_lag=2)
    ranked = list(table["feature"])
    assert ranked.index("adv_usg_pct") < ranked.index("bas_pts")

    drifting = table.set_index("feature").loc["bas_pts"]
    assert drifting["era_gap"] > 0.7          # pooled far above within-season
    assert drifting["minutes_weighted"]       # bas_pts is a per-36 column
    assert not table.set_index("feature").loc["adv_usg_pct"]["minutes_weighted"]


def test_measure_drops_features_with_too_few_pairs():
    frame = _drifting_frame(n_players=5)
    assert measure(frame, SEASONS, min_pairs=200).empty
