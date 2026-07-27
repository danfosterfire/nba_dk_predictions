import numpy as np
import pandas as pd

from src.features.opponent import (
    BilinearMatchup,
    _cell_share,
    _is_team_rate,
    fit_bilinear,
    is_home,
    lag_profiles,
    minutes_weights,
    nxt_map,
    opponent_abbreviation,
    profile_cols,
    standardize_within_season,
    to_per100,
    variance_explained,
)


# ── Matchup string parsing ───────────────────────────────────────────────────

def test_opponent_abbreviation_handles_home_and_away():
    m = pd.Series(["LAL @ NOP", "MEM vs. DEN", "BOS vs. WAS"])
    assert opponent_abbreviation(m).tolist() == ["NOP", "DEN", "WAS"]


def test_is_home_flags_vs_not_at():
    m = pd.Series(["LAL @ NOP", "MEM vs. DEN"])
    assert is_home(m).tolist() == [0.0, 1.0]


# ── Per-100 normalization ────────────────────────────────────────────────────

def _team_seasons() -> pd.DataFrame:
    # MIN and POSS are season totals in these files; the counting stats are per game.
    return pd.DataFrame({
        "team_id": [1, 2],
        "season": ["2023-24", "2023-24"],
        "gp": [82, 82],
        "poss": [8200.0, 7380.0],          # 100.0 and 90.0 possessions per game
        "opp_pts": [110.0, 99.0],
        "def_rating": [110.0, 110.0],
        "opp_fg_pct": [0.47, 0.47],
    })


def test_to_per100_uses_possessions_per_game_not_minutes():
    out = to_per100(_team_seasons())
    # 110 pts/game at 100 poss/game is 110 per 100; 99 at 90 poss/game is also 110.
    assert np.allclose(out["opp_pts"].to_numpy(), [110.0, 110.0])


def test_to_per100_leaves_rate_columns_alone():
    out = to_per100(_team_seasons())
    assert np.allclose(out["def_rating"].to_numpy(), [110.0, 110.0])
    assert np.allclose(out["opp_fg_pct"].to_numpy(), [0.47, 0.47])


def test_is_team_rate_classifies_ratings_and_percentages():
    assert _is_team_rate("DEF_RATING") and _is_team_rate("OPP_FG_PCT")
    assert _is_team_rate("PACE") and _is_team_rate("OPP_FTA_RATE")
    assert not _is_team_rate("OPP_PTS") and not _is_team_rate("OPP_REB")


def test_profile_cols_holds_out_volume_and_identity():
    cols = profile_cols(to_per100(_team_seasons()))
    assert "team_id" not in cols and "season" not in cols
    assert "poss" not in cols and "gp" not in cols and "poss_per_game" not in cols
    assert "opp_pts" in cols


# ── Era standardization ──────────────────────────────────────────────────────

def test_standardize_within_season_removes_era_level():
    """A league-wide rise must not read as every team getting worse defensively."""
    df = pd.DataFrame({
        "team_id": [1, 2, 1, 2],
        "season": ["2003-04", "2003-04", "2023-24", "2023-24"],
        "opp_pts": [90.0, 94.0, 110.0, 114.0],       # +20 league-wide, same spread
    })
    out = standardize_within_season(df, ["opp_pts"])
    early = out[out["season"] == "2003-04"]["opp_pts"].to_numpy()
    late = out[out["season"] == "2023-24"]["opp_pts"].to_numpy()
    assert np.allclose(early, late)
    assert abs(early.mean()) < 1e-12


def test_standardize_within_season_fills_a_constant_column_with_zero():
    df = pd.DataFrame({"team_id": [1, 2], "season": ["2023-24"] * 2,
                       "opp_pts": [100.0, 100.0]})
    out = standardize_within_season(df, ["opp_pts"])
    assert np.allclose(out["opp_pts"].to_numpy(), 0.0)


# ── Lagging ──────────────────────────────────────────────────────────────────

def test_lag_profiles_shifts_a_team_forward_one_season():
    seasons = ["2021-22", "2022-23", "2023-24"]
    prof = pd.DataFrame({"team_id": [1, 1, 1], "season": seasons, "opp_pc1": [0.1, 0.2, 0.3]})
    out = lag_profiles(prof, seasons).sort_values("season")
    assert out["season"].tolist() == ["2022-23", "2023-24"]
    assert out["profile_season"].tolist() == ["2021-22", "2022-23"]
    assert np.allclose(out["opp_pc1"].to_numpy(), [0.1, 0.2])


def test_lag_profiles_keys_on_team_id_so_relocations_keep_their_history():
    """SEA→OKC keeps its team_id; joining on abbreviation would lose the prior season."""
    seasons = ["2007-08", "2008-09"]
    prof = pd.DataFrame({"team_id": [1610612760], "season": ["2007-08"], "opp_pc1": [0.9]})
    out = lag_profiles(prof, seasons)
    assert out["team_id"].tolist() == [1610612760]
    assert out["season"].tolist() == ["2008-09"]


def test_nxt_map_does_not_pair_across_a_gap():
    assert nxt_map(["2019-20", "2020-21"]) == {"2019-20": "2020-21"}
    assert "2020-21" not in nxt_map(["2019-20", "2020-21"])


# ── Weighting ────────────────────────────────────────────────────────────────

def test_minutes_weights_normalize_to_mean_one_and_ignore_bad_rows():
    w = minutes_weights([10.0, 30.0, np.nan, -1.0])
    assert abs(w.mean() - 1.0) < 1e-12
    assert w[2] == 0.0 and w[3] == 0.0
    assert w[1] == 3 * w[0]


def test_variance_explained_is_weighted_when_weights_are_given():
    y = np.array([0.0, 2.0, 100.0])
    pred = np.array([0.0, 2.0, 0.0])
    # the wild low-minutes row is down-weighted away, so the fit on the rest is perfect
    assert abs(variance_explained(y, pred, w=np.array([1.0, 1.0, 0.0])) - 1.0) < 1e-12
    # unweighted, that one row dominates and the same prediction looks terrible
    assert variance_explained(y, pred) < 0.1


def test_cell_share_is_one_when_cells_determine_the_outcome():
    y = np.array([1.0, 1.0, 5.0, 5.0])
    assert abs(_cell_share(y, np.array(["a", "a", "b", "b"])) - 1.0) < 1e-12
    assert abs(_cell_share(y, np.array(["a", "b", "a", "b"]))) < 1e-12


# ── The bilinear fit ─────────────────────────────────────────────────────────

def _planted(n=4000, d_s=4, d_o=3, seed=0, noise=0.1):
    """y = alpha·o + (u·s)(v·o) + noise, with a known rank-1 interaction."""
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, d_s))
    O = rng.normal(size=(n, d_o))
    u = np.array([1.0, -0.5, 0.0, 0.0])[:d_s]
    v = np.array([0.8, 0.0, 0.6])[:d_o]
    alpha = np.array([0.3, -0.2, 0.1])[:d_o]
    y = O @ alpha + (S @ u) * (O @ v) + rng.normal(scale=noise, size=n)
    return S, O, y, u, v, alpha


def test_fit_bilinear_recovers_a_planted_rank_one_interaction():
    S, O, y, u, v, _ = _planted()
    fit = fit_bilinear(S, O, y, rank=1, ridge=1e-6, seed=0)
    # u and v are identified only up to a shared scale, so compare the outer product.
    planted = np.outer(u, v)
    recovered = np.outer(fit.U[0], fit.V[0])
    assert np.allclose(recovered, planted, atol=0.05)


def test_fit_bilinear_recovers_the_main_effect_alongside_the_interaction():
    S, O, y, _, _, alpha = _planted()
    fit = fit_bilinear(S, O, y, rank=1, ridge=1e-6, seed=0)
    assert np.allclose(fit.alpha, alpha, atol=0.05)


def test_fit_bilinear_predicts_better_than_the_main_effect_alone():
    S, O, y, _, _, _ = _planted()
    fit = fit_bilinear(S, O, y, rank=1, ridge=1e-6, seed=0)
    main_only = BilinearMatchup(U=np.zeros_like(fit.U), V=fit.V, alpha=fit.alpha,
                                intercept=fit.intercept, rank=1)
    assert variance_explained(y, fit.predict(S, O)) > \
           variance_explained(y, main_only.predict(S, O)) + 0.1


def test_fit_bilinear_normalizes_v_so_refits_are_comparable():
    S, O, y, _, _, _ = _planted()
    fit = fit_bilinear(S, O, y, rank=2, ridge=1e-6, seed=0)
    assert np.allclose(np.linalg.norm(fit.V, axis=1), 1.0)


def test_bilinear_interaction_has_one_column_per_rank():
    S, O, y, _, _, _ = _planted()
    fit = fit_bilinear(S, O, y, rank=2, ridge=1e-6, seed=0)
    assert fit.interaction(S, O).shape == (len(S), 2)


def test_fit_bilinear_finds_nothing_when_the_pairing_is_broken():
    """The shuffled null: same marginals, no player↔opponent correspondence."""
    S, O, y, _, _, _ = _planted(noise=1.0)
    rng = np.random.default_rng(1)
    S_shuf = S[rng.permutation(len(S))]
    fit = fit_bilinear(S_shuf, O, y, rank=1, ridge=1.0, seed=0)
    gain = variance_explained(y, fit.predict(S_shuf, O)) - variance_explained(
        y, fit.intercept + O @ fit.alpha)
    assert gain < 0.01


def test_fit_bilinear_respects_sample_weights():
    """Down-weighted rows must not steer the fit."""
    S, O, y, u, v, _ = _planted(n=2000, seed=3)
    corrupt = np.concatenate([y[:1000], np.full(1000, 50.0)])
    w = np.concatenate([np.ones(1000), np.zeros(1000)])
    fit = fit_bilinear(S, O, corrupt, rank=1, ridge=1e-6, seed=0, w=w)
    assert np.allclose(np.outer(fit.U[0], fit.V[0]), np.outer(u, v), atol=0.1)
