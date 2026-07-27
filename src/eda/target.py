"""What the target actually looks like — per component, and as a running season total.

One measurement, three consumers. The mean-variance relationship of each component
picks a GLM's family (Poisson / negative binomial / Tweedie / log-normal), a GBM's
objective, and either validates or overturns the Poisson-with-minutes-offset choice
already sitting in `src/models/multihead.py`. Nothing here fits or evaluates a model;
it measures the thing all three would be fitting.

## The number that decides the likelihood

`var / mean` on a component pooled over every game is always far above 1, but almost
all of that is *players differing from each other* — player-season identity is 58% of
total per-game variance, and no likelihood choice fixes or should fix that. The
decision-relevant quantity is the dispersion of a player's counts around **his own**
season mean at a **fixed exposure**, so every bucket reports both:

    var_over_mean          marginal, inside the bucket
    var_over_mean_within   after removing each player-season's own mean

If `var_over_mean_within` is ~1, a Poisson head with a minutes offset is correctly
specified. Above 1, it is not, and `dispersion_alpha_within` — solving
`var = mu + alpha·mu²` — is the negative-binomial parameter that would fix it.

## Buckets

Minutes is the exposure the rate heads are offset by, so the mean-variance
relationship is traced along it. Usage is the player's role, joined from the season
matrix; it is *same-season* usage, which would be leakage in a feature but is exactly
right here, where the job is to describe the target rather than predict it.

## Coverage caveat

Built on `component_targets.parquet`, which comes from `preprocess.clean` and
therefore carries `min_games=20` — players with fewer than 20 games in a season are
absent. Zero-*count* inflation (a game with no blocks) is fully represented; zero-
*game* inflation (a fringe player's DNPs) is under-represented, and the season-total
half of this module is about players who played a real season anyway.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.targets import COMPONENTS, SHOT_CLASSES, add_shot_classes

# dk_pts is continuous and carries the double-double bonus, so its var/mean is not a
# Poisson diagnostic — it is reported for the distribution shape and the season-total
# work, and the likelihood question is answered by the components.
#
# `pts` is itself a weighted sum of counts — `pts = 2*fgm + fg3m + ftm`, exactly — so it
# is profiled alongside the shot classes it decomposes into. The 2x weight on field
# goals squares into the variance, which is most of why `pts` looks overdispersed while
# its parts do not. Attempts are carried next to makes because the attempts/efficiency
# split is the one a regression actually wants.
SCORING_PARTS = list(dict.fromkeys(["fgm", "fga", *SHOT_CLASSES]))
METRICS = list(dict.fromkeys(["dk_pts", *COMPONENTS, *SCORING_PARTS]))

MINUTES_BUCKETS = [0, 5, 12, 18, 24, 30, 48]
USAGE_BUCKETS = [0.0, 0.15, 0.20, 0.25, 0.30, 1.0]
FIRST_K_GAMES = [5, 10, 20, 41, 60]

KEYS = ["player_id", "season"]


# ── Distribution statistics ───────────────────────────────────────────────────

def dispersion_stats(df: pd.DataFrame, col: str) -> dict:
    """Mean-variance relationship and shape for one column of one bucket.

    The `_within` figures remove each player-season's own mean first, which is what
    separates "players differ" from "a given player's game-to-game count is
    overdispersed". Only the latter is a statement about the likelihood.
    """
    v = pd.to_numeric(df[col], errors="coerce")
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return {"n": int(len(v))}

    mean, var = float(v.mean()), float(v.var(ddof=1))
    out = {
        "n": int(len(v)),
        "mean": mean,
        "var": var,
        "var_over_mean": var / mean if mean > 0 else np.nan,
        # var = mu + alpha*mu^2 — alpha 0 is Poisson, alpha > 0 is negative binomial.
        "dispersion_alpha": (var - mean) / mean ** 2 if mean > 0 else np.nan,
        "zero_share": float((v == 0).mean()),
        "skew": float(v.skew()),
        "excess_kurtosis": float(v.kurtosis()),
        "p90": float(v.quantile(0.90)),
        "max": float(v.max()),
    }

    sub = df.loc[v.index, KEYS].assign(_v=v.to_numpy())
    counts = sub.groupby(KEYS)["_v"].transform("size")
    sub = sub[counts >= 2]
    if len(sub) >= 30:
        resid = sub["_v"] - sub.groupby(KEYS)["_v"].transform("mean")
        n_groups = sub.groupby(KEYS).ngroups
        dof = max(len(sub) - n_groups, 1)
        var_w = float((resid ** 2).sum() / dof)
        out["var_within"] = var_w
        out["var_over_mean_within"] = var_w / mean if mean > 0 else np.nan
        out["dispersion_alpha_within"] = (var_w - mean) / mean ** 2 if mean > 0 else np.nan
        out["n_within"] = int(len(sub))
    return out


def bucket_labels(values: pd.Series, edges: list[float]) -> pd.Series:
    """Half-open bins as readable `lo-hi` strings, with the top edge inclusive."""
    idx = np.clip(np.digitize(pd.to_numeric(values, errors="coerce"), edges[1:-1]),
                  0, len(edges) - 2)
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    out = pd.Series([labels[i] for i in idx], index=values.index, dtype=object)
    return out.where(pd.to_numeric(values, errors="coerce").notna())


def profile(games: pd.DataFrame, metrics: list[str] = None,
            minutes_buckets: list[float] = MINUTES_BUCKETS,
            usage_buckets: list[float] = USAGE_BUCKETS) -> pd.DataFrame:
    """Distribution of every metric, overall and inside each minutes/usage bucket."""
    metrics = metrics or [m for m in METRICS if m in games.columns]
    df = games.copy()
    df["_minutes_bucket"] = bucket_labels(df["min"], minutes_buckets)
    if "usg_pct" in df:
        df["_usage_bucket"] = bucket_labels(df["usg_pct"], usage_buckets)

    rows = []
    for metric in metrics:
        rows.append({"analysis": "distribution", "metric": metric,
                     "bucket_kind": "all", "bucket": "all",
                     **dispersion_stats(df, metric)})
        for kind, col in (("minutes", "_minutes_bucket"), ("usage", "_usage_bucket")):
            if col not in df:
                continue
            for bucket, g in df.dropna(subset=[col]).groupby(col, sort=True):
                rows.append({"analysis": "distribution", "metric": metric,
                             "bucket_kind": kind, "bucket": bucket,
                             **dispersion_stats(g, metric)})
    return pd.DataFrame(rows)


# ── Running season total ──────────────────────────────────────────────────────

def game_index(games: pd.DataFrame) -> pd.Series:
    """1-based position of each game inside its player-season, by date."""
    order = games.sort_values(KEYS + ["game_date", "game_id"])
    return order.groupby(KEYS).cumcount().add(1).reindex(games.index)


def season_totals(games: pd.DataFrame, ks: list[int] = FIRST_K_GAMES) -> pd.DataFrame:
    """Per player-season: the realized total, and the mean over each first-k prefix."""
    df = games.copy()
    df["game_index"] = game_index(df)

    out = df.groupby(KEYS).agg(
        games=("dk_pts", "size"),
        dk_pts_total=("dk_pts", "sum"),
        dk_pts_mean=("dk_pts", "mean"),
        dk_pts_std=("dk_pts", "std"),
        minutes_total=("min", "sum"),
    ).reset_index()

    for k in ks:
        prefix = (df[df["game_index"] <= k].groupby(KEYS)["dk_pts"].mean()
                  .rename(f"first{k}_mean"))
        out = out.merge(prefix.reset_index(), on=KEYS, how="left")
    return out


def first_k_predictiveness(totals: pd.DataFrame, ks: list[int] = FIRST_K_GAMES) -> pd.DataFrame:
    """How well the first k games predict the final total.

    The predictor is the deliberately naive one — the first-k mean, extrapolated over
    the games actually played — so the number reads as "how much of the season total
    is already settled by game k", not as a model score. Player-seasons shorter than
    k+5 games are excluded: their prefix *is* most of their season and would flatter
    the estimate.
    """
    rows = []
    for k in ks:
        col = f"first{k}_mean"
        if col not in totals:
            continue
        g = totals[(totals["games"] >= k + 5) & totals[col].notna()]
        if len(g) < 30:
            continue
        pred = g[col] * g["games"]
        actual = g["dk_pts_total"]
        r = float(np.corrcoef(pred, actual)[0, 1])
        ss_res = float(((actual - pred) ** 2).sum())
        ss_tot = float(((actual - actual.mean()) ** 2).sum())
        rows.append({
            "analysis": "season_total", "metric": "dk_pts",
            "bucket_kind": "first_k_games", "bucket": str(k), "n": int(len(g)),
            "r": r, "r2_of_r": r ** 2,
            "r2_extrapolated": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "mae": float((actual - pred).abs().mean()),
            "mean": float(actual.mean()),
        })
    return pd.DataFrame(rows)


def trajectories(games: pd.DataFrame, n_deciles: int = 10,
                 max_games: int = 82) -> pd.DataFrame:
    """Mean cumulative dk_pts by game index, split by final-total decile.

    Precomputed for the dashboard: the raw per-player curves are ~700k rows, while
    their decile envelope is a few hundred and says the same thing.
    """
    df = games.copy()
    df["game_index"] = game_index(df)
    df = df[df["game_index"] <= max_games]
    df = df.sort_values(KEYS + ["game_index"])
    df["cumulative_dk_pts"] = df.groupby(KEYS)["dk_pts"].cumsum()

    totals = df.groupby(KEYS)["dk_pts"].sum().rename("final_total")
    decile = pd.qcut(totals, n_deciles, labels=False, duplicates="drop").rename("decile")
    df = df.merge(pd.concat([totals, decile], axis=1).reset_index(), on=KEYS, how="left")

    out = df.groupby(["decile", "game_index"]).agg(
        n=("cumulative_dk_pts", "size"),
        mean_cumulative=("cumulative_dk_pts", "mean"),
        p25=("cumulative_dk_pts", lambda s: s.quantile(0.25)),
        p75=("cumulative_dk_pts", lambda s: s.quantile(0.75)),
        mean_final=("final_total", "mean"),
    ).reset_index()
    return out[out["n"] >= 20]


# ── Orchestration ─────────────────────────────────────────────────────────────

def load_games(cfg: dict) -> pd.DataFrame:
    """Component targets joined to the player's same-season usage rate."""
    features_dir = Path(cfg["data"]["features_dir"])
    src = features_dir / "component_targets.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run `make component-targets` first")
    games = pd.read_parquet(src)

    # `build_component_targets` already writes these, but the artifact on disk may
    # predate that; deriving through the same helper keeps one definition of "a
    # two-pointer is fgm minus fg3m" rather than a second copy that can drift.
    add_shot_classes(games)

    matrix_path = features_dir / "season_matrix_tierA.parquet"
    if matrix_path.exists():
        usg = pd.read_parquet(matrix_path, columns=["player_id", "season", "adv_usg_pct"])
        games = games.merge(usg.rename(columns={"adv_usg_pct": "usg_pct"}),
                            on=KEYS, how="left")
    return games


def run(cfg: dict) -> Path:
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t_cfg = cfg["eda"].get("target", {})
    minutes_buckets = t_cfg.get("minutes_buckets", MINUTES_BUCKETS)
    usage_buckets = t_cfg.get("usage_buckets", USAGE_BUCKETS)
    ks = t_cfg.get("first_k_games", FIRST_K_GAMES)

    games = load_games(cfg)
    print(f"\nTarget profile over {len(games):,} player-games, "
          f"{games['season'].nunique()} seasons")

    dist = profile(games, minutes_buckets=minutes_buckets, usage_buckets=usage_buckets)
    totals = season_totals(games, ks)
    pred = first_k_predictiveness(totals, ks)
    traj = trajectories(games)

    out = pd.concat([dist, pred], ignore_index=True)
    dest = out_dir / "target_profile.csv"
    out.to_csv(dest, index=False)

    totals_dest = out_dir / "target_season_totals.parquet"
    totals.to_parquet(totals_dest, index=False)
    traj_dest = out_dir / "target_trajectories.parquet"
    traj.to_parquet(traj_dest, index=False)

    # ── printed read: the likelihood decision ────────────────────────────────
    overall = dist[(dist["bucket_kind"] == "all")]
    show = ["metric", "n", "mean", "var", "var_over_mean", "var_over_mean_within",
            "dispersion_alpha_within", "zero_share", "skew"]
    print("\nPer-component distribution (pooled). `_within` removes each player-season's "
          "own mean —\nthat is the column the likelihood choice turns on, not the "
          "marginal one:")
    print(overall[show].round(3).to_string(index=False))

    print("\nMean-variance by minutes played (var_over_mean_within; 1.0 = Poisson is "
          "correctly specified):")
    pivot = dist[dist["bucket_kind"] == "minutes"].pivot_table(
        index="bucket", columns="metric", values="var_over_mean_within")
    order = [f"{minutes_buckets[i]:g}-{minutes_buckets[i + 1]:g}"
             for i in range(len(minutes_buckets) - 1)]
    print(pivot.reindex([o for o in order if o in pivot.index])[
        [m for m in METRICS if m in pivot.columns]].round(2).to_string())

    print("\nHow much of the season total is settled by game k "
          f"({len(totals):,} player-seasons):")
    print(pred[["bucket", "n", "r", "r2_extrapolated", "mae", "mean"]]
          .round(3).to_string(index=False))

    print(f"\n→ {dest}")
    print(f"→ {totals_dest}  ({len(totals):,} player-seasons)")
    print(f"→ {traj_dest}  ({len(traj):,} decile × game-index rows)")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
