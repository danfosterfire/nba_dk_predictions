"""Age curves for per-36 dk_pts and its components, by the delta method.

A cross-sectional curve — mean production at each age — is corrupted by
survivorship: weak 34-year-olds are out of the league, so the surviving ones look
strong and the curve turns *up* at the end. The delta method instead averages the
**within-player change** between consecutive seasons and integrates it, so every
point on the curve is estimated from players who were present at both ends of it.
Both are written out (`cross_sectional_mean` next to `cumulative`) because the gap
between them is the bias, and showing it is more useful than asserting it.

Two corrections carry over from the rest of the EDA:

**Absorb the season.** League-wide scoring, pace and three-point volume all rose over
the sample, so a raw mean delta is positive at every age for reasons that have
nothing to do with aging. `mean_delta_era_adj` is measured against the league's own
change that season pair; `mean_delta` keeps the drift in, and the two are reported
side by side.

**Minutes-weight.** These are per-36 rates, and a rate from a 6-minute-a-night season
is mostly noise. Pairs are weighted by the harmonic mean of their two seasons'
minutes (`persistence.pair_weights`).

The output is deliberately long and un-fitted — one row per (tier, metric,
archetype, age) — because its three eventual consumers want different things from
it: a GLM wants a spline basis, a GBM just wants raw age as a column, and the NN
wants a feature or embedding. Committing to a fitted functional form here would
serve one of them and get in the other two's way. The nearest consumer is
`src/features/team_context.py`, which decays a stale description by a flat
0.96/season; `cumulative_ratio` is the age-aware replacement for that constant.

Residual bias the delta method does not remove: a player who declines gets fewer
minutes rather than leaving outright, and minutes weighting then discounts his
decline. `min` and `gp` are carried as metrics precisely so that channel is visible
instead of hidden.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.archetypes import assign_archetypes, load_cluster_artifacts
from src.eda.persistence import demean_within, lagged_pairs, pair_weights
from src.features.targets import DK_WEIGHTS

# Per-36 components, as named in the season matrix (mirrors context_value.COMPONENT_COLS).
COMPONENT_COLS = {
    "pts_per36": "bas_pts", "fg3m_per36": "bas_fg3m", "reb_per36": "bas_reb",
    "ast_per36": "bas_ast", "stl_per36": "bas_stl", "blk_per36": "bas_blk",
    "tov_per36": "bas_tov",
}

# Volume, not style. Carried because "how much does an older player still play?" is
# as much of the aging question as his rates, and it is the channel by which the
# minutes weighting could otherwise hide a decline.
VOLUME_METRICS = {"minutes_per_game": "min", "games_played": "gp"}

ALL_ARCHETYPES = "all"


def dk_linear_per36(df: pd.DataFrame) -> pd.Series:
    """DK-weighted sum of the per-36 components — the linear part of dk_pts, per 36.

    Only the linear part: the double-double bonus is a per-*game* threshold on five
    counts, so it has no per-36 expression at all (`E[bonus] != bonus(E[x])`). The
    weights come from `src/features/targets.py`, never re-typed here.
    """
    return sum(df[col] * DK_WEIGHTS[name.removesuffix("_per36")]
               for name, col in COMPONENT_COLS.items() if col in df)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename the season-matrix columns to metric names and add dk_linear_per36."""
    out = frame.copy()
    rename = {v: k for k, v in COMPONENT_COLS.items() if v in out}
    rename |= {v: k for k, v in VOLUME_METRICS.items() if v in out}
    if all(c in out for c in COMPONENT_COLS.values()):
        out["dk_linear_per36"] = dk_linear_per36(out)

    # The season matrix already carries a `games_played` from the game-log target
    # join, which is the same count `gp` holds. Renaming onto it would produce two
    # columns of one name and every later lookup would return a frame, not a series.
    clash = [c for c in out.columns if c in set(rename.values()) and c not in rename]
    return out.drop(columns=clash).rename(columns=rename)


def metric_cols(prepared: pd.DataFrame) -> list[str]:
    names = ["dk_linear_per36", *COMPONENT_COLS, *VOLUME_METRICS]
    return [c for c in names if c in prepared.columns]


# ── Archetype labels ──────────────────────────────────────────────────────────

def label_archetypes(frame: pd.DataFrame, features_dir: str | Path,
                     tier: str) -> pd.DataFrame:
    """Place every row of the inclusive frame in the fitted archetype space.

    The persisted clusterers exist for exactly this: `archetypes_tier*.parquet` covers
    only the qualified matrix, and stratifying the curves on that would restrict every
    per-archetype curve to players who cleared `GP>=20 & MIN>=10`. Shrinkage in
    z-space (as in `team_context.describe_roster`) keeps a 40-minute season from
    being handed a confident role.
    """
    from src.features.team_context import reliability

    rel = reliability(frame["min_total"])
    assigned = assign_archetypes(frame, features_dir, tier, reliability=rel)
    _, _, meta = load_cluster_artifacts(features_dir, tier)
    names = meta["names"]

    out = frame.copy()
    out["archetype"] = assigned["archetype"].astype(int).to_numpy()
    out["archetype_name"] = [names.get(int(c), str(c)) for c in out["archetype"]]
    return out


# ── The delta method ──────────────────────────────────────────────────────────

def build_deltas(prepared: pd.DataFrame, seasons: list[str],
                 metrics: list[str]) -> pd.DataFrame:
    """One row per consecutive-season pair: age at t, weight, and each metric's change.

    Age is the player's age in the *earlier* season, floored to an integer, so a row
    labelled 27 is the change from 27 to 28.
    """
    pairs = lagged_pairs(prepared, seasons, lag=1)
    out = pd.DataFrame({
        "player_id": pairs["player_id"].to_numpy(),
        "season": pairs["season"].to_numpy(),
        "age": np.floor(pd.to_numeric(pairs["age"], errors="coerce")).to_numpy(),
        "weight": pair_weights(pairs["min_total"], pairs["min_total_next"]),
    })
    for col in ("archetype", "archetype_name"):
        if col in pairs:
            out[col] = pairs[col].to_numpy()

    for m in metrics:
        if f"{m}_next" not in pairs.columns:
            continue
        out[f"delta_{m}"] = (pd.to_numeric(pairs[f"{m}_next"], errors="coerce")
                             - pd.to_numeric(pairs[m], errors="coerce")).to_numpy()
        out[f"level_{m}"] = pd.to_numeric(pairs[m], errors="coerce").to_numpy()
    return out.dropna(subset=["age"])


def _weighted_mean_se(v: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean and its standard error, using Kish's effective sample size."""
    if len(v) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.average(v, weights=w))
    var = float(np.average((v - mean) ** 2, weights=w))
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    return mean, float(np.sqrt(var / n_eff)) if n_eff > 1 else np.nan


def curve(deltas: pd.DataFrame, metric: str, age_min: int, age_max: int,
          min_players: int, anchor_age: int) -> pd.DataFrame:
    """Mean change per age for one metric, integrated into a level curve.

    `mean_delta_era_adj` — and therefore `cumulative` — is *relative* to the league's
    average change that season. A decline shared equally by every age is
    observationally identical to the league deflating and is absorbed to zero; the
    absolute movement stays visible in `mean_delta`. What the era-adjusted curve
    measures is the aging *shape*, which is the part that transfers across eras.
    """
    col = f"delta_{metric}"
    if col not in deltas:
        return pd.DataFrame()

    d = deltas[["age", "season", "player_id", "weight", col]].copy()
    d["level"] = deltas.get(f"level_{metric}", np.nan)
    d = d[np.isfinite(d[col]) & (d["weight"] > 0)]
    if d.empty:
        return pd.DataFrame()

    # Era adjustment first, over the whole panel: a season pair's league-wide change
    # is the same number at every age, so it must be removed before slicing by age.
    d["delta_adj"] = demean_within(d[col].to_numpy(), d["season"].to_numpy(),
                                   d["weight"].to_numpy())

    rows = []
    for age in range(age_min, age_max + 1):
        g = d[d["age"] == age]
        n_players = g["player_id"].nunique()
        if n_players < min_players:
            continue
        w = g["weight"].to_numpy()
        raw, _ = _weighted_mean_se(g[col].to_numpy(), w)
        adj, se = _weighted_mean_se(g["delta_adj"].to_numpy(), w)
        cross = (np.average(g["level"], weights=w)
                 if np.isfinite(g["level"]).all() and len(g) else np.nan)
        rows.append({"age": age, "n_pairs": len(g), "n_players": int(n_players),
                     "mean_delta": raw, "mean_delta_era_adj": adj, "se": se,
                     "cross_sectional_mean": cross})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Integrate: the row at age a holds the change from a to a+1, so the level at a
    # is the sum of every change strictly below it.
    level = np.concatenate([[0.0], np.cumsum(out["mean_delta_era_adj"].to_numpy())[:-1]])
    anchor = np.interp(anchor_age, out["age"], level) if len(out) else 0.0
    out["cumulative"] = level - anchor

    # A multiplier a consumer can apply directly, e.g. in place of team_context's flat
    # 0.96/season staleness decay. Anchored on the cross-sectional level at the anchor
    # age, which is the only place the delta curve has an absolute scale.
    base = np.interp(anchor_age, out["age"], out["cross_sectional_mean"].to_numpy())
    out["cumulative_ratio"] = (base + out["cumulative"]) / base if base else np.nan
    out.insert(0, "metric", metric)
    return out


def build_curves(deltas: pd.DataFrame, metrics: list[str], age_min: int, age_max: int,
                 min_players: int, anchor_age: int, by_archetype: bool = True) -> pd.DataFrame:
    """Curves for every metric, overall and (optionally) per archetype."""
    frames = []
    for metric in metrics:
        overall = curve(deltas, metric, age_min, age_max, min_players, anchor_age)
        if not overall.empty:
            overall.insert(1, "archetype", -1)
            overall.insert(2, "archetype_name", ALL_ARCHETYPES)
            frames.append(overall)

        if not by_archetype or "archetype" not in deltas:
            continue
        for (arch, name), g in deltas.groupby(["archetype", "archetype_name"]):
            sub = curve(g, metric, age_min, age_max, min_players, anchor_age)
            if sub.empty:
                continue
            sub.insert(1, "archetype", int(arch))
            sub.insert(2, "archetype_name", name)
            frames.append(sub)

    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def peak_age(curves: pd.DataFrame, metric: str,
             archetype_name: str = ALL_ARCHETYPES) -> float:
    """Age at which the integrated curve for `metric` is highest."""
    g = curves[(curves["metric"] == metric) & (curves["archetype_name"] == archetype_name)]
    return float(g.loc[g["cumulative"].idxmax(), "age"]) if len(g) else np.nan


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(cfg: dict) -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    a_cfg = cfg["eda"].get("aging", {})
    age_min = a_cfg.get("age_min", 19)
    age_max = a_cfg.get("age_max", 40)
    min_players = a_cfg.get("min_players_per_age", 25)
    anchor_age = a_cfg.get("anchor_age", 23)
    seasons = cfg["data"]["seasons"]

    tables = []
    for tier in ("A", "B"):
        # Inclusive frame: the ages the curve is least certain about (19-21, 37+) are
        # exactly where the qualification filter removes the most players.
        src = features_dir / f"season_matrix_roster_tier{tier}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found — run `make season-matrix` first")
        frame = pd.read_parquet(src)
        frame = label_archetypes(frame, features_dir, tier)

        prepared = prepare(frame)
        metrics = metric_cols(prepared)
        deltas = build_deltas(prepared, seasons, metrics)
        curves = build_curves(deltas, metrics, age_min, age_max, min_players, anchor_age)
        if curves.empty:
            print(f"Tier {tier}: no age points cleared min_players_per_age={min_players}")
            continue
        curves.insert(0, "tier", tier)
        tables.append(curves)

        overall = curves[curves["archetype_name"] == ALL_ARCHETYPES]
        print(f"\nTier {tier}: {len(deltas):,} consecutive-season pairs, "
              f"ages {int(overall['age'].min())}-{int(overall['age'].max())}, "
              f"{curves['archetype_name'].nunique() - 1} archetypes")
        print("  peak age (integrated, era-adjusted): " +
              ", ".join(f"{m} {peak_age(curves, m):.0f}" for m in metrics
                        if np.isfinite(peak_age(curves, m))))

        show = ["age", "n_pairs", "n_players", "mean_delta", "mean_delta_era_adj",
                "cumulative", "cross_sectional_mean"]
        dk = overall[overall["metric"] == "dk_linear_per36"]
        if not dk.empty:
            print("  dk_linear_per36 — note cross_sectional_mean turning up at the top "
                  "where cumulative does not; that gap is the survivorship bias:")
            print(dk[show].round(3).to_string(index=False))

            # The consumer waiting on this: team_context decays a stale description by
            # a flat 0.96/season regardless of age.
            late = dk[dk["age"] >= 30]
            if len(late) > 1:
                ratio = late["cumulative_ratio"].to_numpy()
                step = np.mean(ratio[1:] / np.where(ratio[:-1] == 0, np.nan, ratio[:-1]))
                print(f"  implied per-season retention past 30: {step:.3f} "
                      f"(team_context currently uses a flat 0.96)")

    out = pd.concat(tables, ignore_index=True)
    dest = out_dir / "aging_curves.csv"
    out.to_csv(dest, index=False)
    print(f"\nAging curves: {len(out):,} tier × metric × archetype × age rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
