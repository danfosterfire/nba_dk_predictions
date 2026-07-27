"""Cluster the era-adjusted PC scores into player archetypes.

The archetypes are the bridge to the team-composition goal: once every player-season
carries an archetype, a team is expressible as a minutes-weighted distribution over
archetypes, which is the input the eventual model needs for both own-team and
opponent-team context.

k is chosen from a silhouette/BIC sweep rather than asserted — see `sweep_k`.

The fitted KMeans/GMM are persisted so that players *outside* the qualified season
matrix can be placed in the same archetype space — see `assign_archetypes`. Roster
aggregation needs this: a season-S roster contains real teammates who failed the
`GP>=20 & MIN>=10` filter, and dropping them biases the aggregate toward veterans.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.eda.pca import load_artifacts

# Readable phrases for the features that end up naming clusters. Several columns
# describe the same concept (adv_reb_pct and usg_pct_reb are both rebounding), so
# labels repeat deliberately — the namer dedupes by label to avoid "rebounding,
# rebounding, rebounding".
STAT_LABELS: dict[str, str] = {
    # volume and role
    "adv_usg_pct": "usage", "adv_e_usg_pct": "usage", "usg_pct_pts": "usage",
    "bas_pts": "scoring volume", "bas_fgm": "scoring volume", "bas_fga": "shot volume",
    # rebounding
    "adv_reb_pct": "rebounding", "usg_pct_reb": "rebounding", "bas_reb": "rebounding",
    "adv_oreb_pct": "offensive rebounding", "usg_pct_oreb": "offensive rebounding",
    "bas_oreb": "offensive rebounding",
    "adv_dreb_pct": "defensive rebounding", "bas_dreb": "defensive rebounding",
    # playmaking
    "adv_ast_pct": "playmaking", "bas_ast": "playmaking", "usg_pct_ast": "playmaking",
    "adv_ast_ratio": "playmaking", "pass_potential_ast": "playmaking",
    "sco_pct_ast_fgm": "assisted shots", "sco_pct_uast_fgm": "self-created shots",
    # shot profile
    "bas_fg3a": "3-point volume", "bas_fg3m": "3-point volume",
    "sco_pct_fga_3pt": "3-point rate", "sl_above_the_break_3_fga": "above-break 3s",
    "sl_restricted_area_fga": "rim attempts", "sco_pct_pts_paint": "paint scoring",
    "misc_pts_paint": "paint scoring", "sl_mid_range_fga": "mid-range volume",
    "bas_fta": "free-throw rate", "usg_pct_fta": "free-throw rate",
    "sco_pct_pts_ft": "free-throw rate",
    # efficiency
    "adv_ts_pct": "shooting efficiency", "adv_efg_pct": "shooting efficiency",
    "bas_fg_pct": "shooting efficiency", "bas_fg3_pct": "3-point accuracy",
    # defense and disruption
    "bas_blk": "shot blocking", "usg_pct_blk": "shot blocking",
    "bas_stl": "steals", "usg_pct_stl": "steals",
    "def_def_rating": "defensive rating", "def_def_ws": "defensive win shares",
    # mistakes
    "adv_tm_tov_pct": "turnovers", "bas_tov": "turnovers", "usg_pct_tov": "turnovers",
    "bas_pf": "fouling", "usg_pct_pf": "fouling",
    # physique and context
    "bio_player_height_inches": "height", "bio_player_weight": "weight",
    "adv_net_rating": "net rating", "bas_plus_minus": "on-court margin",
    "adv_pace": "pace", "clu_pts": "clutch scoring",
    # tracking (Tier B)
    "poss_touches": "touches", "poss_time_of_poss": "time of possession",
    "cs_catch_shoot_fga": "catch-and-shoot volume", "pu_pull_up_fga": "pull-up volume",
    "drv_drives": "drives", "poss_paint_touches": "paint touches",
    "pnt_paint_touches": "paint touches", "post_post_touches": "post touches",
    "elb_elbow_touch_pts": "elbow touches", "hus_contested_shots": "contested shots",
    "hus_screen_assists": "screen assists", "hus_deflections": "deflections",
    "spd_avg_speed": "average speed", "spd_dist_miles": "distance covered",
}

# Hand-written names for specific clusters, applied after auto-naming. Keyed by
# (tier, cluster id) for the configured k and seed — edit freely, the auto-name
# is only a starting point.
ARCHETYPE_NAME_OVERRIDES: dict[tuple[str, int], str] = {}


# ── Choosing k ────────────────────────────────────────────────────────────────

def sweep_k(X: np.ndarray, k_min: int, k_max: int, seed: int = 42,
            sample_size: int = 5000) -> pd.DataFrame:
    """Score k-means and GMM across a range of k.

    Silhouette is computed on a subsample — it is O(n²) and the ranking across k
    is stable well below the full 10k rows.
    """
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        gm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full").fit(X)
        n = min(sample_size, len(X))
        rows.append({
            "k": k,
            "kmeans_silhouette": silhouette_score(X, km.labels_, sample_size=n, random_state=seed),
            "kmeans_inertia": km.inertia_,
            "gmm_silhouette": silhouette_score(X, gm.predict(X), sample_size=n, random_state=seed),
            "gmm_bic": gm.bic(X),
        })
        print(f"  k={k:2}  kmeans silhouette={rows[-1]['kmeans_silhouette']:.4f}  "
              f"gmm BIC={rows[-1]['gmm_bic']:,.0f}  gmm silhouette={rows[-1]['gmm_silhouette']:.4f}")
    return pd.DataFrame(rows)


def choose_k(sweep: pd.DataFrame, configured: int | None = None) -> tuple[int, str]:
    """Pick k from the sweep, preferring GMM BIC over silhouette.

    Silhouette rewards well-separated globular clusters. Player style is a
    continuum, so silhouette declines monotonically across the whole range and
    would always return `k_min` — it measures how un-blobby the data is, not how
    many archetypes there are. GMM BIC does show an interior optimum, so that is
    the selector. The BIC curve is shallow (fractions of a percent between
    neighbouring k), so treat the result as a reasonable default and override it
    in config when a coarser or finer vocabulary suits the downstream task.
    """
    if configured:
        return int(configured), "configured"
    monotonic = sweep["kmeans_silhouette"].is_monotonic_decreasing
    k = int(sweep.loc[sweep["gmm_bic"].idxmin(), "k"])
    note = "min GMM BIC"
    if monotonic:
        note += "; silhouette declines monotonically so it is not used"
    return k, note


# ── Naming ────────────────────────────────────────────────────────────────────

def cluster_profiles(z: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean standardized value of every feature per cluster (cluster × feature)."""
    prof = z.assign(archetype=labels).groupby("archetype").mean()
    prof.index.name = "archetype"
    return prof


def name_cluster(profile: pd.Series, n_terms: int = 3, threshold: float = 0.35) -> str:
    """Name a cluster from its strongest deviations, e.g. "high usage, low rebounding".

    Only labelled features are considered, and each concept is used once so the
    name does not repeat "rebounding" three times from three correlated columns.
    """
    ranked = profile.reindex(profile.abs().sort_values(ascending=False).index)
    terms, seen = [], set()
    for feat, z in ranked.items():
        label = STAT_LABELS.get(feat)
        if label is None or label in seen or abs(z) < threshold:
            continue
        seen.add(label)
        terms.append(f"{'high' if z > 0 else 'low'} {label}")
        if len(terms) == n_terms:
            break
    return ", ".join(terms) if terms else "league-average profile"


def name_clusters(profiles: pd.DataFrame, tier: str) -> dict[int, str]:
    names = {int(c): name_cluster(profiles.loc[c]) for c in profiles.index}
    for (t, cid), override in ARCHETYPE_NAME_OVERRIDES.items():
        if t == tier and cid in names:
            names[cid] = override
    return names


# ── Team composition ──────────────────────────────────────────────────────────

# Players whose role is not in dispute, printed after clustering so the grouping
# can be eyeballed. Centers must not share a cluster with primary guards, and the
# 3&D wings should sit together.
SANITY_PLAYERS = {
    "centers": ["Nikola Jokić", "Joel Embiid", "Rudy Gobert", "DeAndre Jordan"],
    "lead guards": ["Chris Paul", "Stephen Curry", "Trae Young", "Allen Iverson"],
    "3&D wings": ["Mikal Bridges", "Robert Covington", "P.J. Tucker", "Danny Green"],
}


def print_sanity_check(labeled: pd.DataFrame) -> None:
    """Show which archetype each known role lands in, most recent season first."""
    print("\n  Domain sanity check (archetype ids per known role):")
    for role, players in SANITY_PLAYERS.items():
        hits = []
        for name in players:
            rows = labeled[labeled["player_name"] == name]
            if rows.empty:
                continue
            latest = rows.loc[rows["season"].idxmax()]
            hits.append(f"{name.split()[-1]}={int(latest['archetype'])}")
        print(f"    {role:12} {'  '.join(hits) if hits else '(none present)'}")


ROSTER_STATS = ("adv_usg_pct", "adv_pace", "bio_player_height_inches")


def team_composition(labeled: pd.DataFrame, k: int) -> pd.DataFrame:
    """Minutes-weighted archetype distribution and roster aggregates per (team, season).

    `labeled` must carry the archetype labels, `min_total`, `age` and the raw
    ROSTER_STATS columns.

    Two caveats inherited from the season files: a traded player's whole season is
    attributed to his *last* team, and only qualified players (the GP/MIN filter)
    are counted — so `roster_size` and `minutes_covered` are reported to make the
    denominator visible rather than implied.
    """
    df = labeled.rename(columns={"min_total": "weight"})

    shares = df.pivot_table(index=["team_abbreviation", "season"], columns="archetype",
                            values="weight", aggfunc="sum", fill_value=0.0)
    shares = shares.div(shares.sum(axis=1), axis=0)
    # Reindex so every team-season carries a column per archetype, including
    # archetypes it happens to have no minutes at.
    shares = shares.reindex(columns=range(k), fill_value=0.0)
    shares.columns = [f"share_arch{c}" for c in shares.columns]

    def _weighted(g: pd.DataFrame, col: str) -> float:
        if col not in g:
            return np.nan
        v, w = g[col], g["weight"]
        ok = v.notna() & w.notna() & (w > 0)
        return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan

    aggs = []
    for (team, season), g in df.groupby(["team_abbreviation", "season"]):
        # Usage concentration: each player's share of the roster's usage-minutes,
        # squared and summed (Herfindahl). 1/n is perfectly even, 1 is one player.
        u = (g.get("adv_usg_pct", pd.Series(dtype=float)).fillna(0) * g["weight"]).clip(lower=0)
        aggs.append({
            "team_abbreviation": team, "season": season,
            "roster_size": len(g), "minutes_covered": float(g["weight"].sum()),
            "mean_age": _weighted(g, "age"),
            "mean_height_inches": _weighted(g, "bio_player_height_inches"),
            "pace": _weighted(g, "adv_pace"),
            "usage_hhi": float(((u / u.sum()) ** 2).sum()) if u.sum() > 0 else np.nan,
            "dk_pts_per_game_total": float(g["dk_pts_per_game"].sum()),
        })

    return pd.DataFrame(aggs).merge(shares.reset_index(), on=["team_abbreviation", "season"])


# ── Cluster artifacts ─────────────────────────────────────────────────────────

def save_cluster_artifacts(kmeans: KMeans, gmm: GaussianMixture, n_pcs: int,
                           names: dict[int, str], out_dir: str | Path, tier: str) -> None:
    """Mirror of src/eda/pca.py::save_artifacts for the fitted clusterers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"kmeans_tier{tier}.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(out_dir / f"gmm_tier{tier}.pkl", "wb") as f:
        pickle.dump(gmm, f)
    with open(out_dir / f"cluster_meta_tier{tier}.pkl", "wb") as f:
        pickle.dump({"n_pcs": n_pcs, "k": int(gmm.n_components), "names": names}, f)


def load_cluster_artifacts(out_dir: str | Path, tier: str) -> tuple[KMeans, GaussianMixture, dict]:
    out_dir = Path(out_dir)
    with open(out_dir / f"kmeans_tier{tier}.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(out_dir / f"gmm_tier{tier}.pkl", "rb") as f:
        gmm = pickle.load(f)
    with open(out_dir / f"cluster_meta_tier{tier}.pkl", "rb") as f:
        meta = pickle.load(f)
    return kmeans, gmm, meta


def assign_archetypes(matrix: pd.DataFrame, features_dir: str | Path, tier: str,
                      reliability: np.ndarray | None = None) -> pd.DataFrame:
    """Place arbitrary player-seasons in the fitted archetype space.

    `matrix` must carry the tier's season-matrix columns; rows need not have been in
    the fitted set. Returns the `pc*`, `archetype` and `gmm_p*` columns aligned to
    `matrix.index`.

    `reliability` (optional, one value per row in [0, 1]) shrinks the standardized
    features toward the league average before projection. Zero minutes of evidence
    should not produce a confident style read, and z-space is exactly where "shrink
    toward the league mean" is a plain multiplication. Feeding a shrunken row through
    the *same* PCA and GMM keeps every player in one comparable space.
    """
    features_dir = Path(features_dir)
    pca, scaler, _ = load_artifacts(features_dir, f"tier{tier}_within_season")
    kmeans, gmm, meta = load_cluster_artifacts(features_dir, tier)

    z = scaler.transform(matrix)
    if reliability is not None:
        z = z * np.asarray(reliability, dtype=float)[:, None]

    n_pcs = meta["n_pcs"]
    X = pca.transform(z)[:, :n_pcs]

    out = pd.DataFrame(X, columns=[f"pc{i + 1}" for i in range(n_pcs)], index=matrix.index)
    out["archetype"] = kmeans.predict(X)
    out["gmm_archetype"] = gmm.predict(X)
    for i, col in enumerate(gmm.predict_proba(X).T):
        out[f"gmm_p{i}"] = col
    return out


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    """Cluster each tier's era-adjusted PCs and write archetype + team artifacts."""
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    arch_cfg = cfg["eda"]["archetypes"]
    seed = cfg["training"]["seed"]

    written: dict[str, Path] = {}
    for tier in ("A", "B"):
        tag = f"tier{tier}_within_season"          # archetypes use the era-neutral space
        scores_path = features_dir / f"pca_{tag}_scores.parquet"
        if not scores_path.exists():
            raise FileNotFoundError(f"{scores_path} not found — run `make pca` first")

        scores = pd.read_parquet(scores_path)
        matrix = pd.read_parquet(features_dir / f"season_matrix_tier{tier}.parquet")
        _, scaler, feat_names = load_artifacts(features_dir, tag)

        n_pcs = min(arch_cfg["n_pcs"], sum(c.startswith("pc") for c in scores.columns))
        X = scores[[f"pc{i + 1}" for i in range(n_pcs)]].to_numpy()

        print(f"\nTier {tier}: sweeping k over {len(X):,} player-seasons × {n_pcs} PCs")
        sweep = sweep_k(X, arch_cfg["k_min"], arch_cfg["k_max"], seed)
        k, why = choose_k(sweep, arch_cfg["k"])
        print(f"  → k={k} ({why})")

        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        gm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full").fit(X)

        z = pd.DataFrame(scaler.transform(matrix), columns=feat_names, index=matrix.index)
        profiles = cluster_profiles(z, km.labels_)
        names = name_clusters(profiles, tier)

        labeled = scores.copy()
        labeled["archetype"] = km.labels_
        labeled["archetype_name"] = [names[int(c)] for c in km.labels_]
        labeled["gmm_archetype"] = gm.predict(X)
        for i, col in enumerate(gm.predict_proba(X).T):
            labeled[f"gmm_p{i}"] = col

        # ── artifacts ────────────────────────────────────────────────────────
        dest = features_dir / f"archetypes_tier{tier}.parquet"
        labeled.to_parquet(dest, index=False)
        written[f"archetypes_{tier}"] = dest

        prof_out = profiles.reset_index()
        prof_out.insert(1, "archetype_name", [names[int(c)] for c in profiles.index])
        prof_out.to_parquet(features_dir / f"archetype_profiles_tier{tier}.parquet", index=False)
        sweep.to_csv(out_dir / f"archetype_sweep_tier{tier}.csv", index=False)
        # Persisted so roster aggregation can place sub-threshold players — who are
        # real teammates consuming real minutes — in this same space.
        save_cluster_artifacts(km, gm, n_pcs, names, features_dir, tier)

        # Roster aggregates use the raw (un-z-scored) stats so "mean height 79.4in"
        # and "pace 99.2" read in real units.
        roster = matrix[["player_id", "season", *(c for c in ROSTER_STATS if c in matrix)]]
        comp = team_composition(labeled.merge(roster, on=["player_id", "season"], how="left"), k)
        comp_dest = features_dir / f"team_composition_tier{tier}.parquet"
        comp.to_parquet(comp_dest, index=False)
        written[f"team_composition_{tier}"] = comp_dest

        # ── printed sanity read ──────────────────────────────────────────────
        print(f"\n  Tier {tier} archetypes (k={k}):")
        for cid in sorted(names):
            members = labeled[labeled["archetype"] == cid]
            # Distinct players — the same name across five seasons is one data point
            # for eyeballing purposes.
            sample = (members.sort_values("min_total", ascending=False)
                      .drop_duplicates("player_name")["player_name"].head(5).tolist())
            print(f"    {cid}: {names[cid]}")
            print(f"       n={len(members):,}  e.g. {', '.join(sample)}")
        print_sanity_check(labeled)
        print(f"  → {dest}")
        print(f"  → {comp_dest}  ({len(comp):,} team-seasons)")

    return written


if __name__ == "__main__":
    # Imported through the package path so the pickled clusterers resolve against
    # `src.eda.archetypes` rather than `__main__` — see CLAUDE.md.
    from src.eda.archetypes import run as _run

    cfg = yaml.safe_load(open("configs/default.yaml"))
    _run(cfg)
