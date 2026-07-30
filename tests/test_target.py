import numpy as np
import pandas as pd

from src.eda.target import (
    bucket_labels,
    dispersion_stats,
    first_k_predictiveness,
    game_index,
    profile,
    season_total_decomposition,
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


# ── The season total's two log factors ────────────────────────────────────────

def _totals(spec: list[tuple[int, float]]) -> pd.DataFrame:
    """`season_totals` over player-seasons given as (games, dk_pts per game)."""
    rows = []
    for p, (n, rate) in enumerate(spec):
        rows += [{"player_id": p, "season": "s", "dk_pts": rate} for _ in range(n)]
    return season_totals(_games(rows), ks=[5])


def test_the_two_log_factors_together_reproduce_the_identity_exactly():
    """log(total) = log(rate) + log(games) is arithmetic, so both together must give R²=1."""
    rng = np.random.default_rng(0)
    spec = [(int(rng.integers(10, 82)), float(rng.uniform(3.0, 45.0))) for _ in range(300)]
    out = season_total_decomposition(_totals(spec), min_games=10)
    both = out[out["bucket"] == "log_rate_and_games"]["r2"].iloc[0]
    assert abs(both - 1.0) < 1e-9


def test_a_fixed_games_count_hands_the_whole_total_to_the_rate():
    """With games constant, log(games) has no variance and explains nothing."""
    rng = np.random.default_rng(1)
    spec = [(50, float(rng.uniform(3.0, 45.0))) for _ in range(200)]
    out = season_total_decomposition(_totals(spec), min_games=10).set_index("bucket")
    assert out.loc["log_rate", "r2"] > 0.9999
    assert abs(out.loc["log_games", "r2"]) < 1e-9


def test_a_fixed_rate_hands_the_whole_total_to_games_played():
    rng = np.random.default_rng(2)
    spec = [(int(rng.integers(10, 82)), 20.0) for _ in range(200)]
    out = season_total_decomposition(_totals(spec), min_games=10).set_index("bucket")
    assert out.loc["log_games", "r2"] > 0.9999
    assert abs(out.loc["log_rate", "r2"]) < 1e-9


def test_the_games_floor_changes_the_answer_which_is_why_it_is_reported():
    """Two-game seasons carry a noisy rate; the floor is load-bearing, not cosmetic."""
    rng = np.random.default_rng(3)
    spec = [(int(rng.integers(10, 82)), float(rng.uniform(10.0, 40.0))) for _ in range(200)]
    spec += [(2, float(rng.uniform(0.5, 60.0))) for _ in range(200)]
    totals = _totals(spec)
    floored = season_total_decomposition(totals, min_games=10).set_index("bucket")
    unfiltered = season_total_decomposition(totals, min_games=1).set_index("bucket")
    assert floored.loc["log_rate", "r2"] != unfiltered.loc["log_rate", "r2"]
    assert int(floored[floored.index == "log_rate"]["n"].iloc[0]) == 200


def test_the_spread_row_carries_games_in_games_not_as_a_share():
    spec = [(n, 20.0) for n in ([20] * 50 + [60] * 50)]
    out = season_total_decomposition(_totals(spec), min_games=10)
    spread = out[out["bucket_kind"] == "spread"].iloc[0]
    assert spread["metric"] == "games"
    assert abs(spread["mean"] - 40.0) < 1e-9
    assert spread["p10"] == 20.0 and spread["p90"] == 60.0


def test_decomposition_rows_are_additive_and_do_not_disturb_the_existing_sections():
    """The extension must ADD rows, never rewrite the schema the other sections use."""
    spec = [(40, 20.0), (60, 25.0)] * 30
    totals = _totals(spec)
    pred = first_k_predictiveness(totals, ks=[5])
    decomp = season_total_decomposition(totals, min_games=10)
    combined = pd.concat([pred, decomp], ignore_index=True)
    assert set(pred.columns) <= set(combined.columns)
    assert (combined["analysis"] == "season_total").sum() == len(pred)
    assert (combined["analysis"] == "season_total_decomposition").sum() == len(decomp)


def test_decomposition_returns_empty_rather_than_nonsense_on_a_thin_frame():
    assert season_total_decomposition(_totals([(40, 20.0)]), min_games=10).empty


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
