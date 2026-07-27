"""PCA over the season matrix — reduce ~130/~270 correlated stats to style axes.

Two standardization modes, both kept because they answer different questions:

    within_season  z-score inside each season. Era-neutral: a player is described
                   relative to his own league. This is what feeds archetypes and
                   modeling.
    pooled         one z-score across all 30 seasons. Era becomes a visible axis,
                   so the 3-point revolution shows up as a trajectory.

Volume columns (MIN, GP, total minutes) are held out by `feature_cols` — with
them in, PC1 is just "minutes played" and says nothing about style.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.eda.season_matrix import ID_COLS, VOLUME_COLS, feature_cols

# Calendar, not style: in pooled mode this correlates almost perfectly with
# season and would hand PC1 to the clock instead of to playing style.
PCA_EXCLUDE = ("bio_draft_year",)

TARGET_COLS = ["dk_pts_total", "dk_pts_per_game", "dk_pts_std", "games_played"]


def pca_feature_cols(matrix: pd.DataFrame) -> list[str]:
    """Style features eligible for the PCA."""
    return [c for c in feature_cols(matrix) if c not in PCA_EXCLUDE]


# ── Standardization ───────────────────────────────────────────────────────────

@dataclass
class WithinSeasonScaler:
    """Impute to the season median, then z-score inside each season.

    Fitted statistics are stored per season so a transform is reproducible; a
    season not seen during fit cannot be transformed.
    """

    features: list[str] | None = None
    medians: pd.DataFrame | None = None
    means: pd.DataFrame | None = None
    stds: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, cols: list[str], season_col: str = "season") -> "WithinSeasonScaler":
        self.features = list(cols)
        grouped = df.groupby(season_col)[cols]
        self.medians = grouped.median()
        filled = df[cols].fillna(df.groupby(season_col)[cols].transform("median"))
        by_season = filled.assign(**{season_col: df[season_col].values}).groupby(season_col)
        self.means = by_season[cols].mean()
        self.stds = by_season[cols].std()
        return self

    def transform(self, df: pd.DataFrame, season_col: str = "season") -> np.ndarray:
        cols, seasons = self.features, df[season_col]
        med = self.medians.reindex(seasons).set_axis(df.index)[cols]
        mean = self.means.reindex(seasons).set_axis(df.index)[cols]
        std = self.stds.reindex(seasons).set_axis(df.index)[cols]
        z = (df[cols].fillna(med) - mean) / std.replace(0.0, 1.0)
        # A column absent for a whole season has no median or mean to borrow;
        # 0 is that season's league average in z-space, the neutral position.
        return z.fillna(0.0).to_numpy(dtype=float)

    def fit_transform(self, df: pd.DataFrame, cols: list[str], season_col: str = "season") -> np.ndarray:
        return self.fit(df, cols, season_col).transform(df, season_col)


@dataclass
class PooledScaler:
    """Impute to the global median, then one z-score across every season."""

    features: list[str] | None = None
    medians: pd.Series | None = None
    scaler: StandardScaler | None = None

    def fit(self, df: pd.DataFrame, cols: list[str], season_col: str = "season") -> "PooledScaler":
        self.features = list(cols)
        self.medians = df[cols].median()
        self.scaler = StandardScaler().fit(df[cols].fillna(self.medians))
        return self

    def transform(self, df: pd.DataFrame, season_col: str = "season") -> np.ndarray:
        filled = df[self.features].fillna(self.medians)
        return np.nan_to_num(self.scaler.transform(filled), nan=0.0)

    def fit_transform(self, df: pd.DataFrame, cols: list[str], season_col: str = "season") -> np.ndarray:
        return self.fit(df, cols, season_col).transform(df, season_col)


SCALERS = {"within_season": WithinSeasonScaler, "pooled": PooledScaler}


# ── Fitting ───────────────────────────────────────────────────────────────────

def n_components_for(variance_ratio: np.ndarray, target: float) -> int:
    """Smallest component count reaching `target` cumulative explained variance."""
    return int(np.searchsorted(np.cumsum(variance_ratio), target) + 1)


def fit_pca(matrix: pd.DataFrame, mode: str, variance_target: float = 0.90,
            seed: int = 42) -> tuple[PCA, object, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Standardize, fit a full PCA, and return scores/loadings/variance.

    The PCA is fitted at full rank so the scree plot is complete, then truncated
    to the components needed for `variance_target`.
    """
    cols = pca_feature_cols(matrix)
    scaler = SCALERS[mode]().fit(matrix, cols)
    X = scaler.transform(matrix)

    pca = PCA(random_state=seed).fit(X)
    k = n_components_for(pca.explained_variance_ratio_, variance_target)
    pc_names = [f"pc{i + 1}" for i in range(k)]

    scores = pd.DataFrame(pca.transform(X)[:, :k], columns=pc_names, index=matrix.index)
    passthrough = [c for c in ID_COLS + VOLUME_COLS + TARGET_COLS if c in matrix.columns]
    scores = pd.concat([matrix[passthrough], scores], axis=1)

    loadings = pd.DataFrame(pca.components_[:k].T, index=cols, columns=pc_names)
    loadings.index.name = "feature"

    variance = pd.DataFrame({
        "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    })
    return pca, scaler, scores, loadings.reset_index(), variance, k


# ── Artifacts ─────────────────────────────────────────────────────────────────

def save_artifacts(pca: PCA, scaler: object, features: list[str],
                   out_dir: str | Path, tag: str) -> None:
    """Mirror of src/features/encode.py::save_artifacts, namespaced per tier/mode."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"pca_{tag}.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(out_dir / f"pca_scaler_{tag}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / f"pca_features_{tag}.pkl", "wb") as f:
        pickle.dump(features, f)


def load_artifacts(out_dir: str | Path, tag: str) -> tuple[PCA, object, list[str]]:
    out_dir = Path(out_dir)
    with open(out_dir / f"pca_{tag}.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(out_dir / f"pca_scaler_{tag}.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(out_dir / f"pca_features_{tag}.pkl", "rb") as f:
        features = pickle.load(f)
    return pca, scaler, features


def top_loadings(loadings: pd.DataFrame, pc: str, n: int = 6) -> pd.DataFrame:
    """The n features that define a component, strongest absolute weight first."""
    out = loadings.reindex(loadings[pc].abs().sort_values(ascending=False).index)
    return out[["feature", pc]].head(n)


def run(cfg: dict) -> dict[str, Path]:
    """Fit PCA for every (tier × mode) and write scores, loadings and variance."""
    features_dir = Path(cfg["data"]["features_dir"])
    pca_cfg = cfg["eda"]["pca"]
    target = pca_cfg["variance_target"]
    seed = cfg["training"]["seed"]

    written: dict[str, Path] = {}
    for tier in ("A", "B"):
        src = features_dir / f"season_matrix_tier{tier}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found — run `make season-matrix` first")
        matrix = pd.read_parquet(src)

        for mode in pca_cfg["modes"]:
            tag = f"tier{tier}_{mode}"
            pca, scaler, scores, loadings, variance, k = fit_pca(matrix, mode, target, seed)

            scores.to_parquet(features_dir / f"pca_{tag}_scores.parquet", index=False)
            loadings.to_parquet(features_dir / f"pca_{tag}_loadings.parquet", index=False)
            variance.to_csv(features_dir / f"pca_{tag}_variance.csv", index=False)
            save_artifacts(pca, scaler, pca_feature_cols(matrix), features_dir, tag)
            written[tag] = features_dir / f"pca_{tag}_scores.parquet"

            n_feat = len(pca_feature_cols(matrix))
            print(f"\nTier {tier} / {mode}: {len(matrix):,} player-seasons, {n_feat} features "
                  f"→ {k} PCs for {target:.0%} variance")
            for i in range(min(3, k)):
                pc = f"pc{i + 1}"
                share = variance.loc[i, "explained_variance_ratio"]
                top = top_loadings(loadings, pc, 5)
                terms = ", ".join(f"{r.feature} {r[pc]:+.2f}" for _, r in top.iterrows())
                print(f"  {pc} ({share:.1%}): {terms}")
            print(f"  → {features_dir / f'pca_{tag}_scores.parquet'}")

    return written


if __name__ == "__main__":
    # Import through the package path before running: under `python -m`, classes
    # defined here would otherwise pickle as `__main__.WithinSeasonScaler` and be
    # unloadable from any other process.
    from src.eda.pca import run as _run

    cfg = yaml.safe_load(open("configs/default.yaml"))
    _run(cfg)
