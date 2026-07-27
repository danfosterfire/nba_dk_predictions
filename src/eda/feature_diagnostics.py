"""Failure modes specific to each model family the modeling stage will try.

`persistence.py` ranks features by signal. This module does not rank anything — it
checks the four ways *choosing a model family* can go wrong, which nothing else in
the plan covers. Three families are on the table (regularized/hierarchical GLM,
gradient-boosted trees, the deep multi-head net) and each breaks differently:

1. **Collinearity** — a GLM has to decide between raw-plus-regularization and a
   reduced feature set, and several families in the season matrix report
   near-duplicate columns (`def_dreb` and `bas_dreb` are the same number). VIF,
   condition number, and |r|-threshold clusters, over the within-season z-scored
   feature set. Trees and nets do not care; a GLM's coefficients do.

2. **Cardinality and effective n** — `team_id`, archetype and draft-slot bucket
   decide target-encoding vs. one-hot for a GBM and embedding-table size for the NN.
   Relocations (SEA→OKC, NJN→BKN) and expansion teams are genuine cold starts, and
   the inverse-Herfindahl effective category count says how much of the nominal
   cardinality is real.

3. **A shuffled-null importance helper** (`above_null`, `cell_importance`,
   `importance_table`) — generalized from `opponent.py::variance_ceiling` so that any
   future importance number ships with its own chance level. With ~2,700 cells over
   198k rows the expected chance R² is ~1.4% against a raw statistic of 2.31%, so an
   importance read against a fixed threshold instead of against its own null is
   mostly measuring cell count. The API **requires naming the permuted marginal**,
   because the same statistic on the same data is +0.97% or +0.31% depending only on
   that choice. `run` re-derives the +0.97% figure through the generalized helper and
   against `variance_ceiling` itself, so the two cannot silently drift apart.

4. **Sequence-vs-aggregate ablation** — the NN's LSTM trunk reads a player's
   prior-season game log in order. Does order carry anything `season_matrix.py`'s
   aggregates do not already have for free? Three held-out fits: aggregates alone,
   aggregates plus order-dependent features, and the same order-dependent features
   recomputed on **shuffled** game order. The third is the null: it has the identical
   parameter count, so any gap between it and the second is information in the
   ordering rather than in the extra degrees of freedom.

Output: `outputs/eda/feature_diagnostics.csv` (all four, keyed by an `analysis`
column) plus `feature_correlation_tier{A,B}.parquet` for the dashboard heatmap.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.eda.pca import SCALERS, pca_feature_cols
from src.features.opponent import _ridge, minutes_weights, variance_explained
from src.features.team_context import draft_bucket

KEYS = ["player_id", "season"]


# ── The reusable shuffled-null helper ─────────────────────────────────────────

def cell_means_r2(y: np.ndarray, labels: np.ndarray,
                  weights: np.ndarray | None = None) -> float:
    """Share of the variance of y explained by per-cell means — a one-way ANOVA.

    Generalizes `opponent.py::_cell_share` to weighted observations. In sample and by
    construction: this is a ceiling, and its whole point is to be compared against a
    permutation of itself rather than against zero.
    """
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    df = pd.DataFrame({"y": y, "w": w, "g": np.asarray(labels)})
    num = (df["y"] * df["w"]).groupby(df["g"]).transform("sum")
    den = df["w"].groupby(df["g"]).transform("sum")
    fitted = num / den.replace(0.0, np.nan)

    mean = float(np.average(y, weights=w)) if w.sum() > 0 else np.nan
    tss = float((w * (y - mean) ** 2).sum())
    rss = float((w * (y - fitted.to_numpy()) ** 2).sum())
    return 1.0 - rss / tss if tss > 0 else np.nan


def join_keys(keys: dict[str, np.ndarray]) -> np.ndarray:
    """Combine several key columns into one cell label per row."""
    parts = [pd.Series(np.asarray(v)).astype(str).to_numpy() for v in keys.values()]
    out = parts[0]
    for p in parts[1:]:
        out = np.char.add(np.char.add(out, "|"), p)
    return out


def permute(values: np.ndarray, within: np.ndarray | None,
            rng: np.random.Generator) -> np.ndarray:
    """Shuffle `values`, optionally only inside strata of `within`.

    Permuting within a stratum preserves that stratum's marginal as well as the
    global one — "which opponent, holding the season fixed" rather than "which
    opponent, ignoring that the league changed".
    """
    v = np.asarray(values)
    if within is None:
        return rng.permutation(v)
    out = v.copy()
    within = np.asarray(within)
    for s in np.unique(within):
        m = within == s
        out[m] = rng.permutation(v[m])
    return out


def above_null(statistic, values: dict[str, np.ndarray], permuted: str,
               within: np.ndarray | None = None, n_shuffles: int = 5,
               seed: int = 42) -> dict:
    """Any statistic, minus its own chance level under permuting one named input.

    `statistic` takes the `values` dict and returns a float, so this covers a cell-mean
    ANOVA, a refitted model's held-out score, or a GBM's gain — whatever the caller
    wants an honest chance level for. `permuted` names which entry gets shuffled and
    is **required**: the same statistic on the same data moves by 3x depending on that
    choice, so an unlabelled "above a shuffled null" is not a number anyone can use.
    """
    rng = np.random.default_rng(seed)
    observed = statistic(values)
    nulls = [statistic({**values, permuted: permute(values[permuted], within, rng)})
             for _ in range(n_shuffles)]
    nulls = np.asarray(nulls, dtype=float)
    return {
        "statistic": observed,
        "null_mean": float(np.mean(nulls)),
        "null_sd": float(np.std(nulls, ddof=1)) if n_shuffles > 1 else np.nan,
        "above_null": observed - float(np.mean(nulls)),
        "permuted": permuted,
        "n_shuffles": n_shuffles,
    }


def cell_importance(y: np.ndarray, keys: dict[str, np.ndarray], permuted: str,
                    within: np.ndarray | None = None,
                    weights: np.ndarray | None = None, n_shuffles: int = 5,
                    seed: int = 42) -> dict:
    """`cell_means_r2` over the joined keys, against a named permutation of one key."""
    if permuted not in keys:
        raise KeyError(f"{permuted!r} is not one of the keys {list(keys)}")

    def statistic(v: dict) -> float:
        return cell_means_r2(y, join_keys(v), weights)

    out = above_null(statistic, dict(keys), permuted, within, n_shuffles, seed)
    out["n"] = int(len(y))
    out["n_cells"] = int(len(np.unique(join_keys(keys))))
    out["keys"] = "+".join(keys)
    return out


def importance_table(y: np.ndarray, keys: dict[str, np.ndarray],
                     within: np.ndarray | None = None, within_key: str | None = None,
                     weights: np.ndarray | None = None, n_shuffles: int = 5,
                     seed: int = 42) -> pd.DataFrame:
    """`cell_importance` once per key — the disagreement between them *is* the finding.

    Reporting a single "above a shuffled null" figure hides that the choice of
    marginal moved the opponent × archetype × season statistic from +0.96% to +0.32%
    on identical data. Reporting the whole table makes that visible for free.

    `within_key` names a key to stratify by; it is skipped as a permutation target,
    since permuting a variable within its own levels is the identity.
    """
    if within_key is not None:
        within = keys[within_key]
    targets = [k for k in keys if k != within_key]
    return pd.DataFrame([
        cell_importance(y, keys, k, within, weights, n_shuffles, seed) for k in targets
    ])


# ── 1. Collinearity ───────────────────────────────────────────────────────────

def standardized_features(matrix: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Within-season z-scored feature block — the same one the PCA is fitted on."""
    cols = pca_feature_cols(matrix)
    X = SCALERS["within_season"]().fit_transform(matrix, cols)
    keep = [i for i in range(X.shape[1]) if np.isfinite(X[:, i]).all() and X[:, i].std() > 0]
    return X[:, keep], [cols[i] for i in keep]


def correlation_matrix(X: np.ndarray, cols: list[str]) -> pd.DataFrame:
    R = np.corrcoef(X, rowvar=False)
    return pd.DataFrame(np.nan_to_num(R, nan=0.0), index=cols, columns=cols)


def vif_from_correlation(R: pd.DataFrame, tol: float = 1e-8) -> pd.Series:
    """VIF_i = 1/(1-R²_i), which for standardized data is the i-th diagonal of R⁻¹.

    The block is rank-deficient by construction — `def_dreb` *is* `bas_dreb` — so a
    plain inverse raises. A plain pseudo-inverse is worse than raising: on an exactly
    singular R it returns the *minimum-norm* solution, which hands a perfectly
    duplicated column a VIF near zero and buries the very features this exists to
    find. So the null space is separated out explicitly: any feature loading on a
    near-zero eigenvector is exactly collinear with something and gets `inf`, while
    the invertible part is inverted normally.
    """
    vals, vecs = np.linalg.eigh(R.to_numpy())
    null = vals < tol
    keep = ~null

    out = np.full(len(vals), np.inf)
    if keep.any():
        V, L = vecs[:, keep], vals[keep]
        out = np.einsum("ij,j,ij->i", V, 1.0 / L, V)
    if null.any():
        # Squared loading on the null space: nonzero means "this column is a linear
        # combination of others", which is an infinite variance inflation.
        out = np.where((vecs[:, null] ** 2).sum(axis=1) > tol, np.inf, out)
    return pd.Series(out, index=R.index)


def _spectrum(R: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Eigenvalues of R and the rank tolerance, matching `np.linalg.matrix_rank`."""
    eig = np.linalg.eigvalsh(R.to_numpy())
    tol = float(eig.max()) * len(eig) * np.finfo(float).eps
    return eig, tol


def condition_number(R: pd.DataFrame) -> float:
    """sqrt(λ_max/λ_min) of the correlation matrix. Above ~30 is conventionally severe.

    Infinite when the block is rank-deficient, which is the honest answer: a GLM
    cannot be fitted on it at all without regularization or dropping columns. The
    companion `matrix_rank` in the summary says how many directions are missing, so
    "singular" does not have to be the end of the report.
    """
    eig, tol = _spectrum(R)
    lo = float(eig.min())
    return float(np.sqrt(eig.max() / lo)) if lo > tol else np.inf


def matrix_rank(R: pd.DataFrame) -> int:
    eig, tol = _spectrum(R)
    return int((eig > tol).sum())


def correlation_clusters(R: pd.DataFrame, threshold: float = 0.95) -> pd.Series:
    """Connected components of the graph "these two features correlate above |r|"."""
    adj = (R.abs().to_numpy() >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return pd.Series(labels, index=R.index)


def collinearity(matrix: pd.DataFrame, threshold: float = 0.95,
                 vif_flag: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-feature VIF and cluster membership, plus the block-level summary."""
    X, cols = standardized_features(matrix)
    R = correlation_matrix(X, cols)

    off = R.abs().to_numpy().copy()
    np.fill_diagonal(off, 0.0)
    partner_idx = off.argmax(axis=1)

    clusters = correlation_clusters(R, threshold)
    sizes = clusters.value_counts()
    table = pd.DataFrame({
        "name": cols,
        "vif": vif_from_correlation(R).to_numpy(),
        "max_abs_corr": off.max(axis=1),
        "max_corr_partner": [cols[i] for i in partner_idx],
        "cluster_id": clusters.to_numpy(),
        "cluster_size": [int(sizes[c]) for c in clusters],
    })
    summary = {
        "n_features": len(cols),
        "matrix_rank": matrix_rank(R),
        "condition_number": condition_number(R),
        "n_vif_above_flag": int((table["vif"] >= vif_flag).sum()),
        "n_clusters": int((sizes > 1).sum()),
        "largest_cluster": int(sizes.max()),
    }
    return table.sort_values("vif", ascending=False).reset_index(drop=True), R, summary


# ── 2. Cardinality and cold starts ────────────────────────────────────────────

def cardinality(values: pd.Series, seasons: pd.Series,
                min_n: int = 50) -> dict:
    """Category counts, effective count, and which categories are cold starts.

    `n_effective` is the inverse Herfindahl `1/Σ pᵢ²` — how many categories the data
    behaves as if it has. A nominal 36 team ids that is effectively 30 is telling you
    the six extras are relocations with a handful of seasons each.
    """
    v = values.dropna()
    counts = v.value_counts()
    p = counts / counts.sum()
    span = seasons.loc[v.index].groupby(v).nunique()
    thin = counts[counts < min_n]

    return {
        "n": int(len(v)),
        "n_categories": int(len(counts)),
        "n_effective": float(1.0 / (p ** 2).sum()),
        "min_n": int(counts.min()),
        "median_n": float(counts.median()),
        "n_below_threshold": int(len(thin)),
        "min_seasons_spanned": int(span.min()),
        # The rarest categories, which are the ones a target encoder will overfit and
        # an embedding table will never train: relocations, defunct franchises,
        # shortened seasons.
        "cold_start": ", ".join(str(c) for c in counts.nsmallest(5).index),
    }


def cardinality_report(matrix: pd.DataFrame, archetypes: pd.DataFrame | None,
                       min_n: int = 50) -> pd.DataFrame:
    """Cardinality for every categorical a model would embed or target-encode."""
    df = matrix.copy()
    if archetypes is not None:
        df = df.merge(archetypes[KEYS + ["archetype"]], on=KEYS, how="left")
    if "bio_draft_number" in df:
        df["draft_bucket"] = draft_bucket(df["bio_draft_number"])

    rows = []
    for col in ("team_id", "team_abbreviation", "archetype", "draft_bucket", "season"):
        if col not in df:
            continue
        rows.append({"analysis": "cardinality", "name": col,
                     **cardinality(df[col], df["season"], min_n)})
    return pd.DataFrame(rows)


# ── 3. Reproducing the opponent finding through the generalized helper ────────

def opponent_null_reproduction(cfg: dict, tier: str = "A") -> pd.DataFrame:
    """Re-derive the quoted opponent figures through `cell_importance`.

    The verification the plan asks for: feeding the generalized helper the same
    opponent-within-season permutation must recover ~+0.97%. `variance_ceiling` is run
    on the identical panel alongside it, so a divergence between the one-off and its
    generalization shows up as a number rather than as silence.
    """
    from src.features.opponent import build_panel, nxt_map, variance_ceiling

    fd_cfg = cfg["eda"].get("feature_diagnostics", {})
    window = fd_cfg.get("null_window", ["2014-15", "2023-24"])
    n_shuffles = fd_cfg.get("n_shuffles", 5)
    seed = cfg["training"]["seed"]
    seasons = cfg["data"]["seasons"]
    features_dir = Path(cfg["data"]["features_dir"])

    games, *_ = build_panel(seasons, cfg["data"]["raw_dir"], features_dir, tier,
                            seed=seed)
    arch = pd.read_parquet(features_dir / f"archetypes_tier{tier}.parquet",
                           columns=KEYS + ["archetype"])
    arch["season"] = arch["season"].map(nxt_map(seasons))
    games = games.merge(arch.dropna(subset=["season"]), on=KEYS, how="inner")

    lo, hi = window
    in_window = [s for s in seasons if lo <= s <= hi]
    games = games[games["season"].isin(in_window)]

    keys = {
        "opponent": games["opponent_team_id"].astype(str).to_numpy(),
        "archetype": games["archetype"].astype(str).to_numpy(),
        "season": games["season"].to_numpy(),
    }
    y = games["dk_pts"].to_numpy(dtype=float)   # already within-player-season residuals

    table = importance_table(y, keys, within_key="season",
                             n_shuffles=n_shuffles, seed=seed)
    table.insert(0, "analysis", "null_reproduction")
    table = table.rename(columns={"permuted": "name"})
    table["window"] = f"{lo}..{hi}"

    # The one-off this generalizes from, on exactly the same rows.
    ceil = variance_ceiling(games, games["archetype"], n_shuffles=n_shuffles, seed=seed)
    table["variance_ceiling_above_null"] = [
        ceil["opponent_x_archetype_x_season"] - ceil[f"null_shuffle_{k}"]
        if f"null_shuffle_{k}" in ceil else np.nan
        for k in table["name"]
    ]
    table["reproduction_gap"] = table["above_null"] - table["variance_ceiling_above_null"]
    return table


# ── 4. Sequence vs. aggregate ─────────────────────────────────────────────────

def _group_slope(df: pd.DataFrame, keys: list[str], x: str, y: str) -> pd.Series:
    """Per-group OLS slope of y on x, computed with groupby sums rather than apply."""
    g = [df[k] for k in keys]
    dx = df[x] - df.groupby(keys)[x].transform("mean")
    dy = df[y] - df.groupby(keys)[y].transform("mean")
    num = (dx * dy).groupby(g).sum()
    den = (dx ** 2).groupby(g).sum()
    return num / den.replace(0.0, np.nan)


def _group_autocorr(df: pd.DataFrame, keys: list[str], y: str, lag: int = 1) -> pd.Series:
    """Per-group lag-k autocorrelation of y, in the frame's existing row order."""
    prev = df.groupby(keys)[y].shift(lag)
    sub = df.loc[prev.notna(), keys + [y]].assign(_prev=prev.dropna())
    g = [sub[k] for k in keys]
    da = sub[y] - sub.groupby(keys)[y].transform("mean")
    db = sub["_prev"] - sub.groupby(keys)["_prev"].transform("mean")
    num = (da * db).groupby(g).sum()
    den = np.sqrt((da ** 2).groupby(g).sum() * (db ** 2).groupby(g).sum())
    return num / den.replace(0.0, np.nan)


AGGREGATE_FEATURES = ["mean_dk", "sd_dk", "mean_min", "games"]
SEQUENCE_FEATURES = ["slope_dk", "slope_min", "autocorr_dk", "recent_gap", "half_gap"]


def sequence_features(games: pd.DataFrame, recent_games: int = 10,
                      shuffle_seed: int | None = None) -> pd.DataFrame:
    """Per player-season aggregates, plus five order-dependent summaries.

    With `shuffle_seed` set, game order is permuted *inside* each player-season before
    the order-dependent features are computed. The aggregates are invariant to that by
    construction, so the shuffled arm is the same model with the same parameter count
    fitted on order features that cannot contain order — exactly the null the
    ablation needs.
    """
    df = games[KEYS + ["game_date", "game_id", "dk_pts", "min"]].copy()
    df = df.sort_values(KEYS + ["game_date", "game_id"])
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        df = df.sample(frac=1.0, random_state=rng.integers(1 << 31))
        df = df.sort_values(KEYS, kind="stable")
    df["game_index"] = df.groupby(KEYS).cumcount() + 1

    out = df.groupby(KEYS).agg(mean_dk=("dk_pts", "mean"), sd_dk=("dk_pts", "std"),
                               mean_min=("min", "mean"), games=("dk_pts", "size"))
    out["slope_dk"] = _group_slope(df, KEYS, "game_index", "dk_pts")
    out["slope_min"] = _group_slope(df, KEYS, "game_index", "min")
    out["autocorr_dk"] = _group_autocorr(df, KEYS, "dk_pts")

    tail = df[df["game_index"] > df.groupby(KEYS)["game_index"].transform("max") - recent_games]
    out["recent_gap"] = tail.groupby(KEYS)["dk_pts"].mean() - out["mean_dk"]

    half = df.groupby(KEYS)["game_index"].transform("max") / 2.0
    second = df[df["game_index"] > half].groupby(KEYS)["dk_pts"].mean()
    first = df[df["game_index"] <= half].groupby(KEYS)["dk_pts"].mean()
    out["half_gap"] = second - first
    return out.reset_index()


def zscore_within_season(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Era-neutral scaling, so a temporal split does not need season dummies it
    cannot extend to the held-out seasons."""
    out = df.copy()
    g = out.groupby("season")[cols]
    out[cols] = ((out[cols] - g.transform("mean")) /
                 g.transform("std").replace(0.0, np.nan)).fillna(0.0)
    return out


def sequence_ablation(games: pd.DataFrame, seasons: list[str], test_seasons: int = 2,
                      recent_games: int = 10, ridge: float = 10.0,
                      seed: int = 42) -> dict:
    """Held-out R² from aggregates, aggregates + order, and order-on-shuffled-order."""
    nxt = {s: seasons[i + 1] for i, s in enumerate(seasons) if i + 1 < len(seasons)}

    real = sequence_features(games, recent_games)
    fake = sequence_features(games, recent_games, shuffle_seed=seed)
    outcome = (games.groupby(KEYS)["dk_pts"].mean().rename("outcome").reset_index())

    def panel(feat: pd.DataFrame) -> pd.DataFrame:
        p = feat.copy()
        p["season"] = p["season"].map(nxt)
        p = p.dropna(subset=["season"]).merge(outcome, on=KEYS, how="inner")
        cols = AGGREGATE_FEATURES + SEQUENCE_FEATURES + ["outcome"]
        p[cols] = p[cols].apply(pd.to_numeric, errors="coerce")
        return zscore_within_season(p.dropna(subset=cols), cols)

    p_real, p_fake = panel(real), panel(fake)
    # The two arms must be scored on identical rows, or the comparison is between
    # two different test sets rather than between two feature sets.
    common = p_real.merge(p_fake[KEYS], on=KEYS, how="inner")[KEYS]
    p_real = p_real.merge(common, on=KEYS)
    p_fake = p_fake.merge(common, on=KEYS)

    held = set(seasons[-test_seasons:])
    tr, te = ~p_real["season"].isin(held), p_real["season"].isin(held)
    y_tr = p_real.loc[tr, "outcome"].to_numpy(float)
    y_te = p_real.loc[te, "outcome"].to_numpy(float)

    def fit_score(panel_df: pd.DataFrame, cols: list[str]) -> float:
        X = panel_df[cols].to_numpy(float)
        X = np.hstack([np.ones((len(X), 1)), X])
        beta = _ridge(X[tr.to_numpy()], y_tr, ridge)
        return variance_explained(y_te, X[te.to_numpy()] @ beta)

    agg = fit_score(p_real, AGGREGATE_FEATURES)
    seq = fit_score(p_real, AGGREGATE_FEATURES + SEQUENCE_FEATURES)
    null = fit_score(p_fake, AGGREGATE_FEATURES + SEQUENCE_FEATURES)
    return {
        "analysis": "sequence_ablation", "name": "next_season_dk_pts_per_game",
        "n": int(len(p_real)), "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "r2_aggregate": agg, "r2_sequence": seq, "r2_sequence_shuffled": null,
        "delta_sequence": seq - agg, "delta_above_null": seq - null,
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(cfg: dict) -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = cfg["eda"].get("feature_diagnostics", {})
    threshold = fd.get("collinearity_threshold", 0.95)
    vif_flag = fd.get("vif_flag", 10.0)
    min_n = fd.get("min_category_n", 50)
    seq_cfg = fd.get("sequence", {})
    seed = cfg["training"]["seed"]

    blocks = []
    for tier in ("A", "B"):
        src = features_dir / f"season_matrix_tier{tier}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found — run `make season-matrix` first")
        matrix = pd.read_parquet(src)

        table, R, summary = collinearity(matrix, threshold, vif_flag)
        table.insert(0, "analysis", "collinearity")
        table.insert(1, "tier", tier)
        blocks.append(table)
        blocks.append(pd.DataFrame([{"analysis": "collinearity_summary", "tier": tier,
                                     "name": f"tier{tier}", **summary}]))

        R.to_parquet(out_dir / f"feature_correlation_tier{tier}.parquet")

        arch_path = features_dir / f"archetypes_tier{tier}.parquet"
        arch = pd.read_parquet(arch_path) if arch_path.exists() else None
        card = cardinality_report(matrix, arch, min_n)
        card.insert(1, "tier", tier)
        blocks.append(card)

        print(f"\nTier {tier}: {summary['n_features']} within-season z-scored features, "
              f"rank {summary['matrix_rank']}")
        print(f"  condition number {summary['condition_number']:,.0f}; "
              f"{summary['n_vif_above_flag']} features with VIF >= {vif_flag:g}; "
              f"{summary['n_clusters']} clusters at |r| >= {threshold} "
              f"(largest {summary['largest_cluster']})")
        dup = table[table["cluster_size"] > 1].head(8)
        print("  worst offenders (a GLM must choose; a tree does not care):")
        print(dup[["name", "vif", "max_abs_corr", "max_corr_partner", "cluster_size"]]
              .round(3).to_string(index=False))
        print("  categoricals:")
        print(card[["name", "n_categories", "n_effective", "min_n",
                    "n_below_threshold", "cold_start"]].round(1).to_string(index=False))

    # ── the shuffled-null helper, checked against the one-off it generalizes ──
    repro = opponent_null_reproduction(cfg, "A")
    blocks.append(repro)
    print(f"\nShuffled-null reproduction ({int(repro['n'].iloc[0]):,} player-games, "
          f"{int(repro['n_cells'].iloc[0]):,} cells, window {repro['window'].iloc[0]}) — "
          "opponent x archetype x season,\n% of within-player-season residual variance:")
    show = repro.assign(**{c: repro[c] * 100 for c in
                           ("statistic", "null_mean", "above_null",
                            "variance_ceiling_above_null", "reproduction_gap")})
    print(show[["name", "statistic", "null_mean", "above_null",
                "variance_ceiling_above_null", "reproduction_gap"]]
          .rename(columns={"name": "permuted"}).round(3).to_string(index=False))
    print("  ^ the two nulls disagree by ~3x on identical data. That is why the helper "
          "makes\n    naming the permuted marginal mandatory.")

    # ── does the game-log sequence beat the season aggregate? ────────────────
    ct = features_dir / "component_targets.parquet"
    if ct.exists():
        games = pd.read_parquet(ct, columns=KEYS + ["game_date", "game_id",
                                                    "dk_pts", "min"])
        abl = sequence_ablation(games, cfg["data"]["seasons"],
                                seq_cfg.get("test_seasons", 2),
                                seq_cfg.get("recent_games", 10), seed=seed)
        blocks.append(pd.DataFrame([abl]))
        print(f"\nSequence vs. aggregate ({abl['n_train']:,} train / {abl['n_test']:,} "
              "held-out player-seasons), held-out R² on next-season dk_pts/game:")
        print(f"  aggregates only                     {abl['r2_aggregate']:.4f}")
        print(f"  + order-dependent features          {abl['r2_sequence']:.4f} "
              f"({abl['delta_sequence']:+.4f})")
        print(f"  same features on shuffled order     {abl['r2_sequence_shuffled']:.4f}")
        print(f"  information in the ordering         {abl['delta_above_null']:+.4f}")
    else:
        print(f"\nSkipping sequence ablation — {ct} not found "
              "(run `make component-targets`)")

    out = pd.concat(blocks, ignore_index=True)
    dest = out_dir / "feature_diagnostics.csv"
    out.to_csv(dest, index=False)
    print(f"\nFeature diagnostics: {len(out):,} rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
