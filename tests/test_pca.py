import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.eda.pca import (
    PooledScaler,
    WithinSeasonScaler,
    fit_pca,
    load_artifacts,
    n_components_for,
    pca_feature_cols,
    save_artifacts,
    top_loadings,
)


def _matrix(n_per_season: int = 40, seasons=("2021-22", "2022-23", "2023-24")) -> pd.DataFrame:
    """A season matrix with a deliberate era trend in one feature."""
    rng = np.random.default_rng(0)
    rows = []
    for si, season in enumerate(seasons):
        for i in range(n_per_season):
            rows.append({
                "player_id": 100 + i,
                "player_name": f"Player {i}",
                "season": season,
                "season_start_year": 2021 + si,
                "team_id": 1,
                "team_abbreviation": "LAL",
                "age": 25.0 + (i % 8),
                "gp": 70,
                "min": 28.0,
                "min_total": 1960.0,
                # two correlated style features plus one that drifts with era
                "bas_pts": 18.0 + rng.normal(0, 4),
                "bas_fga": 14.0 + rng.normal(0, 3),
                "bas_fg3a": 2.0 + 2.0 * si + rng.normal(0, 0.5),
                "adv_usg_pct": 0.20 + rng.normal(0, 0.04),
                "bio_draft_year": 2015 + si,
                "dk_pts_total": 2000.0, "dk_pts_per_game": 30.0,
                "dk_pts_std": 8.0, "games_played": 70,
            })
    return pd.DataFrame(rows)


STYLE = ["bas_pts", "bas_fga", "bas_fg3a", "adv_usg_pct"]


# ── Feature selection ────────────────────────────────────────────────────────

def test_pca_features_exclude_volume_identity_target_and_draft_year():
    feats = pca_feature_cols(_matrix())
    assert set(feats) == set(STYLE)
    # draft_year is calendar, not style — it would hand PC1 to the clock
    assert "bio_draft_year" not in feats
    for c in ["min", "gp", "min_total", "age", "dk_pts_per_game", "season_start_year"]:
        assert c not in feats


# ── Within-season standardization ────────────────────────────────────────────

def test_within_season_scaler_centers_each_season_independently():
    m = _matrix()
    X = WithinSeasonScaler().fit_transform(m, STYLE)
    z = pd.DataFrame(X, columns=STYLE).assign(season=m["season"].values)
    for _, g in z.groupby("season"):
        assert np.abs(g[STYLE].mean().to_numpy()).max() < 1e-10
        assert np.abs(g[STYLE].std().to_numpy() - 1.0).max() < 1e-6


def test_within_season_scaler_removes_the_era_trend():
    """bas_fg3a rises every season by construction; era-neutral z must flatten it."""
    m = _matrix()
    raw_by_season = m.groupby("season")["bas_fg3a"].mean()
    assert raw_by_season.is_monotonic_increasing

    X = WithinSeasonScaler().fit_transform(m, STYLE)
    z = pd.DataFrame(X, columns=STYLE).assign(season=m["season"].values)
    assert np.abs(z.groupby("season")["bas_fg3a"].mean().to_numpy()).max() < 1e-10


def test_pooled_scaler_keeps_the_era_trend_visible():
    m = _matrix()
    X = PooledScaler().fit_transform(m, STYLE)
    z = pd.DataFrame(X, columns=STYLE).assign(season=m["season"].values)
    assert z.groupby("season")["bas_fg3a"].mean().is_monotonic_increasing


def test_within_season_scaler_imputes_to_the_season_median():
    m = _matrix()
    m.loc[0, "bas_pts"] = np.nan
    season = m.loc[0, "season"]
    median = m[m["season"] == season]["bas_pts"].median()

    scaler = WithinSeasonScaler().fit(m, STYLE)
    X = scaler.transform(m)
    mean = scaler.means.loc[season, "bas_pts"]
    std = scaler.stds.loc[season, "bas_pts"]
    assert abs(X[0, STYLE.index("bas_pts")] - (median - mean) / std) < 1e-9


def test_scalers_emit_no_nan_for_a_column_absent_all_season():
    """Tier B has no hustle data for 2013-14; those columns are wholly NaN there.

    The two modes handle it differently, and both are right: within-season has no
    season statistics to borrow and lands on 0 (that season's league average in
    z-space), while pooled legitimately falls back to the global median, which
    z-scores to a small offset from the all-season mean.
    """
    m = _matrix()
    m.loc[m["season"] == "2021-22", "bas_pts"] = np.nan
    gap = (m["season"] == "2021-22").to_numpy()
    col = STYLE.index("bas_pts")

    for scaler in (WithinSeasonScaler(), PooledScaler()):
        X = scaler.fit_transform(m, STYLE)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    within = WithinSeasonScaler().fit_transform(m, STYLE)
    assert np.abs(within[gap, col]).max() == 0.0

    pooled = PooledScaler().fit_transform(m, STYLE)
    assert len(np.unique(pooled[gap, col])) == 1        # every gap row gets the same fill
    assert np.abs(pooled[gap, col]).max() < 1.0         # and it sits near the global mean


def test_zero_variance_column_does_not_divide_by_zero():
    m = _matrix()
    m["bas_pts"] = 15.0
    for scaler in (WithinSeasonScaler(), PooledScaler()):
        X = scaler.fit_transform(m, STYLE)
        assert not np.isnan(X).any() and not np.isinf(X).any()


# ── Fitting ──────────────────────────────────────────────────────────────────

def test_n_components_for_variance_target():
    ratios = np.array([0.5, 0.25, 0.15, 0.10])
    assert n_components_for(ratios, 0.50) == 1
    assert n_components_for(ratios, 0.70) == 2
    assert n_components_for(ratios, 0.90) == 3
    assert n_components_for(ratios, 1.00) == 4


def test_pca_round_trip_reconstructs_at_full_rank():
    """inverse_transform(transform(X)) must return X when no components are dropped."""
    m = _matrix()
    X = WithinSeasonScaler().fit_transform(m, STYLE)
    pca = PCA(random_state=42).fit(X)
    assert np.abs(pca.inverse_transform(pca.transform(X)) - X).max() < 1e-9


def test_fit_pca_outputs_are_aligned_and_shaped():
    m = _matrix()
    pca, scaler, scores, loadings, variance, k = fit_pca(m, "within_season", 0.90)

    assert len(scores) == len(m)
    assert [f"pc{i+1}" for i in range(k)] == [c for c in scores.columns if c.startswith("pc")]
    # identity, volume and target columns ride along for later joins
    for c in ["player_id", "season", "min_total", "dk_pts_per_game"]:
        assert c in scores.columns
    assert len(loadings) == len(STYLE)
    assert len(variance) == len(STYLE)
    assert variance["cumulative"].iloc[-1] > 0.999
    assert variance["cumulative"].iloc[k - 1] >= 0.90


def test_top_loadings_ranks_by_absolute_weight():
    loadings = pd.DataFrame({"feature": ["a", "b", "c"], "pc1": [0.1, -0.9, 0.4]})
    top = top_loadings(loadings, "pc1", 2)
    assert top["feature"].tolist() == ["b", "c"]


def test_artifacts_round_trip_through_pickle(tmp_path):
    m = _matrix()
    pca, scaler, *_ = fit_pca(m, "within_season", 0.90)
    save_artifacts(pca, scaler, STYLE, tmp_path, "tierA_within_season")

    loaded_pca, loaded_scaler, feats = load_artifacts(tmp_path, "tierA_within_season")
    assert feats == STYLE
    assert np.allclose(loaded_pca.components_, pca.components_)
    assert np.allclose(loaded_scaler.transform(m), scaler.transform(m))
