import numpy as np
import pandas as pd

from src.eda.target import (
    bucket_labels,
    dispersion_stats,
    first_k_predictiveness,
    game_index,
    profile,
    season_totals,
    trajectories,
)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _games(rows: list[dict]) -> pd.DataFrame:
    """A component-targets-shaped frame: keys, a date, minutes and dk_pts."""
    df = pd.DataFrame(rows)
    if "game_date" not in df:
        df["game_date"] = pd.to_datetime("2023-10-01") + pd.to_timedelta(
            df.groupby(["player_id", "season"]).cumcount(), unit="D")
    if "game_id" not in df:
        df["game_id"] = np.arange(len(df))
    if "min" not in df:
        df["min"] = 30.0
    return df


def _poisson_games(rate: float = 4.0, n_players: int = 40, n_games: int = 40,
                   seed: int = 3) -> pd.DataFrame:
    """Every player-season drawn from the *same* Poisson — no player heterogeneity."""
    rng = np.random.default_rng(seed)
    rows = [{"player_id": p, "season": "2023-24", "dk_pts": float(v)}
            for p in range(n_players) for v in rng.poisson(rate, n_games)]
    return _games(rows)


# ── Dispersion ────────────────────────────────────────────────────────────────

def test_poisson_data_reads_as_var_over_mean_near_one():
    stats = dispersion_stats(_poisson_games(rate=4.0), "dk_pts")
    assert abs(stats["var_over_mean"] - 1.0) < 0.15
    assert abs(stats["var_over_mean_within"] - 1.0) < 0.15
    assert abs(stats["dispersion_alpha_within"]) < 0.05


def test_within_separates_player_heterogeneity_from_game_to_game_noise():
    """The distinction the likelihood choice turns on, planted.

    Players differ enormously, but each one is a constant. The marginal var/mean is
    huge; the within-player-season figure — the one that decides the likelihood — is
    zero, because no player's own count varies at all.
    """
    rows = [{"player_id": p, "season": "2023-24", "dk_pts": float(p)}
            for p in range(1, 41) for _ in range(20)]
    stats = dispersion_stats(_games(rows), "dk_pts")
    assert stats["var_over_mean"] > 5.0
    assert abs(stats["var_over_mean_within"]) < 1e-9


def test_dispersion_declines_to_report_on_a_thin_slice():
    stats = dispersion_stats(_games([{"player_id": 1, "season": "s", "dk_pts": 1.0}]),
                             "dk_pts")
    assert stats == {"n": 1}


def test_zero_share_and_skew_are_reported():
    rows = [{"player_id": p, "season": "2023-24", "dk_pts": 0.0 if p % 4 else 10.0}
            for p in range(80)]
    stats = dispersion_stats(_games(rows), "dk_pts")
    assert abs(stats["zero_share"] - 0.75) < 1e-9
    assert stats["skew"] > 0


# ── Buckets ───────────────────────────────────────────────────────────────────

def test_bucket_labels_are_half_open_with_an_inclusive_top():
    v = pd.Series([0.0, 4.9, 5.0, 11.0, 47.0, 48.0, 60.0])
    labels = bucket_labels(v, [0, 5, 12, 48])
    assert list(labels) == ["0-5", "0-5", "5-12", "5-12", "12-48", "12-48", "12-48"]


def test_bucket_labels_pass_nan_through():
    assert pd.isna(bucket_labels(pd.Series([np.nan]), [0, 5, 10]).iloc[0])


def test_profile_emits_an_overall_row_and_one_per_minutes_bucket():
    g = _poisson_games()
    g["min"] = np.tile([3.0, 20.0], len(g) // 2)
    out = profile(g, metrics=["dk_pts"], minutes_buckets=[0, 5, 48])
    assert set(out["bucket_kind"]) == {"all", "minutes"}
    assert set(out[out["bucket_kind"] == "minutes"]["bucket"]) == {"0-5", "5-48"}


# ── Season totals and the running trajectory ──────────────────────────────────

def test_game_index_orders_by_date_inside_each_player_season():
    g = _games([{"player_id": 1, "season": "s", "dk_pts": float(i)} for i in range(5)])
    g = g.iloc[::-1].reset_index(drop=True)         # feed it out of order
    idx = game_index(g)
    assert list(idx[g["dk_pts"] == 0.0]) == [1]
    assert list(idx[g["dk_pts"] == 4.0]) == [5]


def test_season_totals_first_k_mean_uses_only_the_prefix():
    dk = [10.0] * 5 + [30.0] * 15
    g = _games([{"player_id": 1, "season": "s", "dk_pts": v} for v in dk])
    out = season_totals(g, ks=[5, 10]).iloc[0]
    assert out["games"] == 20
    assert abs(out["dk_pts_total"] - sum(dk)) < 1e-9
    assert abs(out["first5_mean"] - 10.0) < 1e-9
    assert abs(out["first10_mean"] - 20.0) < 1e-9


def test_first_k_is_perfect_when_every_player_is_a_constant():
    rows = []
    for p in range(60):
        rows += [{"player_id": p, "season": "s", "dk_pts": float(p)} for _ in range(40)]
    totals = season_totals(_games(rows), ks=[5])
    out = first_k_predictiveness(totals, ks=[5]).iloc[0]
    assert out["r"] > 0.9999
    assert out["mae"] < 1e-6
    assert out["n"] == 60


def test_first_k_excludes_seasons_barely_longer_than_the_prefix():
    rows = []
    for p in range(60):
        n = 40 if p % 2 else 7           # 7 games is not long enough to score k=5
        rows += [{"player_id": p, "season": "s", "dk_pts": float(p)} for _ in range(n)]
    totals = season_totals(_games(rows), ks=[5])
    assert first_k_predictiveness(totals, ks=[5]).iloc[0]["n"] == 30


def test_trajectories_are_cumulative_and_increase_with_the_final_decile():
    rows = []
    for p in range(100):
        rows += [{"player_id": p, "season": "s", "dk_pts": float(p % 10) + 1.0}
                 for _ in range(30)]
    traj = trajectories(_games(rows), n_deciles=5)
    for _, g in traj.groupby("decile"):
        assert g.sort_values("game_index")["mean_cumulative"].is_monotonic_increasing
    last = traj.sort_values("game_index").groupby("decile").tail(1)
    assert last.sort_values("decile")["mean_cumulative"].is_monotonic_increasing
