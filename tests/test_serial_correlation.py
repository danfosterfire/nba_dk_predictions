import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (FULL_WINDOW, TRAIN_VAL_WINDOW,
                                 fit_window)
from src.eda.serial_correlation import (
    COUNT_COMPONENTS,
    block_inflation,
    conversion_residuals,
    count_residuals,
    lag_autocorr,
    measure,
    minutes_residuals,
    player_season_frame,
    shuffled_null,
)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _panel(counts: dict[str, np.ndarray], minutes: np.ndarray,
           n_seasons: int = 1) -> pd.DataFrame:
    """A component-targets-shaped frame, one player-season per block of games.

    Column names use letters, never digits — `normalize_name` collisions are a recorded
    trap elsewhere in this repo and synthetic ids should stay distinct regardless.
    """
    per = len(minutes) // n_seasons
    rows = {"min": minutes, "played": np.ones(len(minutes), int)}
    rows |= {k: v for k, v in counts.items()}
    df = pd.DataFrame(rows)
    df["player_id"] = np.repeat(np.arange(n_seasons), per)[: len(df)]
    df["season"] = "2021-22"
    df["game_date"] = pd.to_datetime("2021-11-01") + pd.to_timedelta(
        np.tile(np.arange(per), n_seasons)[: len(df)], unit="D")
    df["ps"] = df["player_id"]
    df["game_index"] = df.groupby("ps").cumcount()
    return df


def _iid_poisson(n: int, rate: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(rate, n).astype(float)


def _ar_counts(n: int, rate: float, rho: float, seed: int = 0) -> np.ndarray:
    """Counts driven by an AR(1) latent multiplier — genuine serial dependence."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + rng.normal(0, np.sqrt(1 - rho**2))
    return rng.poisson(rate * np.exp(0.5 * z - 0.125)).astype(float)


# ── Residual construction ─────────────────────────────────────────────────────

def test_count_residuals_are_centred_and_unit_scale_under_poisson():
    n = 4000
    y = _iid_poisson(n, 8.0)
    df = _panel({"reb": y}, np.full(n, 30.0))
    resid, valid = count_residuals(df, "reb")
    assert valid.all()
    assert abs(resid.mean()) < 0.05
    assert 0.9 < resid.std() < 1.1


def test_count_residuals_divide_out_minutes():
    """A player whose rate is constant but whose minutes vary has no rate residual."""
    minutes = np.tile([10.0, 40.0], 200)
    df = _panel({"reb": minutes * 0.2}, minutes)      # exactly 0.2 reb per minute
    resid, valid = count_residuals(df, "reb")
    assert np.allclose(resid[valid], 0.0, atol=1e-8)


def test_conversion_residuals_are_valid_only_where_attempts_exist():
    att = np.array([0.0, 3.0, 0.0, 5.0] * 50)
    made = np.where(att > 0, att * 0.4, 0.0)
    df = _panel({"fg3a": att, "fg3m": made}, np.full(len(att), 30.0))
    _, valid = conversion_residuals(df, "fg3m", "fg3a")
    assert not valid[att == 0].any()
    assert valid[att == 5.0].all()


def test_minutes_detrending_removes_a_linear_role_change():
    n = 200
    minutes = np.linspace(10.0, 40.0, n)              # pure trend, no noise
    df = _panel({"reb": np.full(n, 5.0)}, minutes)
    plain, _ = minutes_residuals(df)
    detrended, _ = minutes_residuals(df, detrend=True)
    assert np.abs(plain).max() > 10.0
    assert np.allclose(detrended, 0.0, atol=1e-8)


# ── Lag pairing ───────────────────────────────────────────────────────────────

def _reference_lag_corr(resid, ps, valid, lag):
    """Obvious O(n) reference: walk the valid rows and pair only within a season."""
    r = [resid[i] for i in range(len(resid)) if valid[i]]
    p = [ps[i] for i in range(len(ps)) if valid[i]]
    a, b = [], []
    for i in range(lag, len(r)):
        if p[i] == p[i - lag]:
            a.append(r[i])
            b.append(r[i - lag])
    return np.corrcoef(a, b)[0, 1], len(a)


def test_lags_never_pair_across_a_player_season_boundary():
    """Vectorized pairing must match an explicit loop that respects the boundary."""
    rng = np.random.default_rng(7)
    ps = np.repeat(np.arange(20), 50)
    resid = rng.normal(size=1000)
    valid = rng.random(1000) > 0.2      # some rows drop out, as conversions do
    for lag in [1, 2, 5]:
        got_r, got_n = lag_autocorr(resid, ps, valid, lag)
        ref_r, ref_n = _reference_lag_corr(resid, ps, valid, lag)
        assert got_n == ref_n
        assert np.isclose(got_r, ref_r)


def test_pairs_are_lost_at_every_season_boundary_not_just_the_first():
    """20 seasons x 50 games at lag 1 gives 49 pairs each, never 999."""
    ps = np.repeat(np.arange(20), 50)
    _, n_pairs = lag_autocorr(np.arange(1000, dtype=float), ps, np.ones(1000, bool), 1)
    assert n_pairs == 20 * 49


def test_lag_pairs_consecutive_valid_games_for_conversions():
    """For a conversion head, "the previous game" means the previous game with an
    attempt — a night with no threes is not evidence about shooting form."""
    ps = np.zeros(6, int)
    resid = np.array([1.0, 99.0, 2.0, 99.0, 3.0, 99.0])
    valid = np.array([True, False, True, False, True, False])
    r, n_pairs = lag_autocorr(resid, ps, valid, 1)
    assert n_pairs == 2              # (1,2) and (2,3); the 99s never enter
    assert np.isclose(r, 1.0)


def test_higher_lags_respect_the_boundary_too():
    ps = np.array([0] * 10 + [1] * 10)
    resid = np.arange(20, dtype=float)
    _, n_pairs = lag_autocorr(resid, ps, np.ones(20, bool), 5)
    assert n_pairs == 10             # 5 usable pairs in each season


# ── Block inflation and the null ──────────────────────────────────────────────

def test_block_inflation_is_one_for_independent_games():
    n = 20000
    resid = np.random.default_rng(1).normal(size=n)
    ps = np.repeat(np.arange(n // 100), 100)
    assert 0.8 < block_inflation(resid, ps, np.ones(n, bool)) < 1.2


def test_block_inflation_exceeds_one_when_games_cluster():
    rng = np.random.default_rng(2)
    n, rho = 20000, 0.6
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = rho * resid[i - 1] + rng.normal(0, np.sqrt(1 - rho**2))
    ps = np.repeat(np.arange(n // 100), 100)
    assert block_inflation(resid, ps, np.ones(n, bool)) > 1.5


def test_shuffling_within_player_season_destroys_dependence():
    rng = np.random.default_rng(3)
    n, rho = 20000, 0.6
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = rho * resid[i - 1] + rng.normal(0, np.sqrt(1 - rho**2))
    ps = np.repeat(np.arange(n // 100), 100)
    valid = np.ones(n, bool)
    observed, _ = lag_autocorr(resid, ps, valid, 1)
    null_mean, null_sd = shuffled_null(
        resid, ps, valid, lambda r, p, v: lag_autocorr(r, p, v, 1)[0], replicates=4)
    assert observed > 0.4
    assert abs(null_mean) < 0.05
    assert null_sd >= 0


def test_the_null_carries_the_demeaning_bias_the_raw_statistic_has():
    """Residuals taken about a player-season's own mean are negatively correlated by
    construction (~ -1/(G-1)). The null must reproduce that, or a genuinely independent
    component reads as negative dependence."""
    n, per = 12000, 60
    y = _iid_poisson(n, 6.0, seed=4)
    df = _panel({"reb": y}, np.full(n, 30.0), n_seasons=n // per)
    resid, valid = count_residuals(df, "reb")
    ps = df["ps"].to_numpy()
    observed, _ = lag_autocorr(resid, ps, valid, 1)
    null_mean, _ = shuffled_null(
        resid, ps, valid, lambda r, p, v: lag_autocorr(r, p, v, 1)[0], replicates=4)
    assert observed < 0                              # the bias, not a real effect
    assert abs(observed - null_mean) < 0.02          # and the null absorbs it


# ── End to end ────────────────────────────────────────────────────────────────

def test_measure_separates_an_ar_component_from_an_independent_one():
    n, per = 12000, 60
    df = _panel({"reb": _iid_poisson(n, 6.0, seed=5),
                 "ast": _ar_counts(n, 6.0, 0.7, seed=6)},
                np.full(n, 30.0), n_seasons=n // per)
    # measure() walks its own component list, so drive the two helpers directly.
    ps = df["ps"].to_numpy()
    flat, v1 = count_residuals(df, "reb")
    auto, v2 = count_residuals(df, "ast")

    flat_excess = (lag_autocorr(flat, ps, v1, 1)[0]
                   - shuffled_null(flat, ps, v1,
                                   lambda r, p, v: lag_autocorr(r, p, v, 1)[0],
                                   replicates=4)[0])
    auto_excess = (lag_autocorr(auto, ps, v2, 1)[0]
                   - shuffled_null(auto, ps, v2,
                                   lambda r, p, v: lag_autocorr(r, p, v, 1)[0],
                                   replicates=4)[0])
    assert abs(flat_excess) < 0.03
    assert auto_excess > 0.15
    assert block_inflation(auto, ps, v2) > block_inflation(flat, ps, v1)


def test_player_season_frame_filters_and_indexes():
    n = 200
    df = pd.DataFrame({
        "player_id": np.repeat([1, 2], n // 2),
        "season": "2021-22",
        "game_date": pd.to_datetime("2021-11-01") + pd.to_timedelta(
            np.tile(np.arange(n // 2), 2), unit="D"),
        "min": np.r_[np.full(n // 2, 30.0), np.full(n // 2, 2.0)],  # p2 is garbage time
        "played": 1,
        "reb": 5.0,
    })
    out = player_season_frame(df, min_minutes=5.0, min_games=40)
    assert set(out["player_id"]) == {1}
    assert out["ps"].nunique() == 1
    assert list(out["game_index"]) == list(range(len(out)))


def test_measure_returns_a_row_per_component_with_its_own_chance_level():
    """Every component gets a chance level; none may come back NaN. Makes are binomial
    draws on their own attempts and minutes vary, because a fixture where either is
    deterministic has zero residual variance and hides that."""
    rng = np.random.default_rng(8)
    n, per = 6000, 60
    counts = {c: _iid_poisson(n, 8.0 if c == "fga" else 4.0, seed=i)
              for i, c in enumerate(COUNT_COMPONENTS)}
    # The chain, in order: the share splits `fga`, `fg2a` is the remainder, then the makes
    # are drawn on their own attempts. `fg2a` is derived here exactly as the head list
    # says it is derived in the model.
    for made, att, p in [("fg3a", "fga", 0.36), ("fg2m", "fg2a", 0.5),
                         ("fg3m", "fg3a", 0.36), ("ftm", "fta", 0.78)]:
        if att not in counts:
            counts["fg2a"] = np.maximum(counts["fga"] - counts["fg3a"], 0.0)
        counts[made] = rng.binomial(counts[att].astype(int), p).astype(float)
    minutes = rng.uniform(12.0, 38.0, n)
    df = _panel(counts, minutes, n_seasons=n // per)

    table = measure(df, lags=[1, 2])
    assert len(table) == 13                        # 7 counts + 4 conversions + 2 minutes
    for col in ["lag1", "lag1_null", "lag1_excess", "block_inflation"]:
        assert col in table.columns
    assert table["lag1_null"].notna().all()
    assert table["block_inflation"].notna().all()
    # Independent by construction, so every excess should sit near zero.
    assert table["lag1_excess"].abs().max() < 0.05


# ── The fit window: block inflation is a simulator input ─────────────────────

def _multi_season_frame(seasons: int = 5, blocks: int = 20,
                        games: int = 60) -> pd.DataFrame:
    """A played-games frame spanning several seasons, for the window axis."""
    n = blocks * games
    minutes = np.full(n, 28.0)
    counts = {c: _iid_poisson(n, 4.0, seed=i)
              for i, c in enumerate(COUNT_COMPONENTS)}
    counts |= {"fg3a": _iid_poisson(n, 3.0, seed=90),
               "fg2a": _iid_poisson(n, 6.0, seed=91),
               "fg2m": _iid_poisson(n, 3.0, seed=92),
               "fg3m": _iid_poisson(n, 1.0, seed=93),
               "ftm": _iid_poisson(n, 2.0, seed=94)}
    df = _panel(counts, minutes, n_seasons=blocks)
    labels = [f"20{10 + i:02d}-{11 + i:02d}" for i in range(seasons)]
    df["season"] = [labels[p % seasons] for p in df["ps"]]
    return df


def test_fit_window_narrows_the_frame_without_changing_the_component_set():
    df = _multi_season_frame()
    full = measure(fit_window(df, FULL_WINDOW))
    train_val = measure(fit_window(df, TRAIN_VAL_WINDOW))
    assert list(full["component"]) == list(train_val["component"])
    assert train_val["n_pairs"].sum() < full["n_pairs"].sum()


def test_holding_out_more_seasons_than_exist_raises_rather_than_emptying():
    """An empty fitting half makes every statistic NaN, which reads as a null."""
    df = _multi_season_frame(seasons=2)
    with pytest.raises(ValueError, match="would be empty"):
        fit_window(df, TRAIN_VAL_WINDOW)
