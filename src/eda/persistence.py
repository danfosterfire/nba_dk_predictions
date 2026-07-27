"""Which prior-season stats actually carry year-over-year signal.

The most directly load-bearing analysis for the modeling goal: everything the model
sees about a player is a season S-1 number used to predict season S, so a stat whose
own value does not survive a year is noise no matter how well it describes the past.
This ranks every column of the season matrix by how much of itself it keeps.

Three corrections separate this from a naive `corr(x_t, x_{t+1})`, and each one moves
the answer:

**Absorb the season.** A pooled correlation across 30 seasons conflates persistence
with era drift — three-point volume roughly doubled over the sample, so *any* column
that rose tracks *any* other column that rose. Both figures are reported side by
side; where they diverge sharply the pooled one is measuring the calendar, and saying
so is the useful output.

**Minutes-weight the per-36 columns.** `sd(pts_per36)` is 42.4 in sub-5-minute games
against 6.6 above 24 minutes. Unweighted, a ranking of per-36 stability is a ranking
of who got garbage time. Weights come from the harmonic mean of the pair's two
seasons — a correlation is only as well measured as its worse half. Rate columns
(`*_PCT`, ratings) are left unweighted, and `r_within_unweighted` is reported for
every column so the effect of the choice is visible rather than assumed.

**Report reliability as a curve, not a threshold.** For each column the observed
lag-1 correlation is traced across prior-minutes bins and fitted to
`r(m) = r_inf · m / (m + m0)`, the empirical-Bayes reliability of a mean measured over
`m` minutes. `src/features/team_context.py` already uses this shape with
`r_inf = 0.924, m0 = 66`, fitted on eight style stats; this generalizes it to every
column so a consumer can shrink each feature by its own curve.

Built on the **inclusive** roster frame, not the qualified matrix. The reliability
curve is unidentified without low-minute rows — the `GP>=20 & MIN>=10` filter starts
at ~200 minutes, already at r ≈ 0.75 — and minutes weighting is exactly what makes
those rows safe to include.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.season_matrix import _is_rate, feature_cols
from src.features.opponent import minutes_weights

# Minutes bins for the reliability curve. Dense at the bottom because that is where
# r(m) actually bends; above ~2,000 minutes every stat has flattened out.
RELIABILITY_BINS = [0, 100, 250, 500, 1000, 1500, 2000, 2500, 4000]

# Candidate half-saturation constants for the curve fit, log-spaced. r_inf is solved
# in closed form at each one, so this is a one-dimensional search.
M0_GRID = np.geomspace(2.0, 2000.0, 80)


def is_per36(col: str) -> bool:
    """True if a season-matrix column holds a per-36 rate, so needs minutes weighting.

    `season_matrix` lowercases and prefixes every column (`bas_pts`, `adv_usg_pct`),
    so the family prefix is stripped before reusing its own rate classifier — the
    single definition of what did and did not get divided by minutes.
    """
    stem = col.split("_", 1)[-1].upper()
    return not _is_rate(stem)


# ── Weighted statistics ───────────────────────────────────────────────────────

def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Pearson correlation under observation weights. NaN if either side is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    if len(x) < 3:
        return np.nan
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(np.average((x - mx) * (y - my), weights=w) / np.sqrt(vx * vy))


def demean_within(v: np.ndarray, groups: np.ndarray,
                  w: np.ndarray | None = None) -> np.ndarray:
    """Subtract each group's (weighted) mean — season fixed effects, absorbed.

    Equivalent to putting season dummies in the regression, which
    `src/eda/context_value.py::season_dummies` does explicitly. Here the panel is one
    column at a time and there is nothing else in the design, so demeaning is the
    same estimator at a fraction of the cost.
    """
    s = pd.Series(np.asarray(v, dtype=float)).reset_index(drop=True)
    g = pd.Series(np.asarray(groups)).reset_index(drop=True)
    if w is None:
        return (s - s.groupby(g).transform("mean")).to_numpy()
    wt = pd.Series(np.asarray(w, dtype=float)).reset_index(drop=True)
    num = (s * wt).groupby(g).transform("sum")
    den = wt.groupby(g).transform("sum")
    return (s - (num / den.replace(0.0, np.nan))).to_numpy()


def pair_weights(minutes_t: np.ndarray, minutes_next: np.ndarray) -> np.ndarray:
    """Weight for a season pair: the harmonic mean of its two total-minute counts.

    A year-over-year correlation is only as well measured as its worse half — 2,400
    minutes paired with 90 carries the noise of the 90 — and the harmonic mean is the
    summary dominated by the smaller of the two. `minutes_weights` then puts it on
    the mean-1 scale the rest of the project's weighted regressions use.
    """
    a = np.asarray(minutes_t, dtype=float)
    b = np.asarray(minutes_next, dtype=float)
    a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
    b = np.where(np.isfinite(b) & (b > 0), b, 0.0)
    both = (a > 0) & (b > 0)
    h = np.zeros(len(a))
    h[both] = 2.0 * a[both] * b[both] / (a[both] + b[both])
    return minutes_weights(h)


# ── Season pairing ────────────────────────────────────────────────────────────

def lagged_pairs(frame: pd.DataFrame, seasons: list[str], lag: int = 1) -> pd.DataFrame:
    """Pair each player's season t with his season t+lag.

    Pairing is on the season *index* rather than the raw label, so a player who
    missed a year does not silently pair across the gap as if it were consecutive.
    """
    order = {s: i for i, s in enumerate(seasons)}
    df = frame[frame["season"].isin(order)].copy()
    df["season_index"] = df["season"].map(order)

    nxt = df.copy()
    nxt["season_index"] = nxt["season_index"] - lag
    return df.merge(nxt, on=["player_id", "season_index"], suffixes=("", "_next"))


# ── Reliability curve ─────────────────────────────────────────────────────────

def reliability_by_bin(pairs: pd.DataFrame, col: str,
                       bins: list[float] = RELIABILITY_BINS) -> pd.DataFrame:
    """Observed lag-1 correlation of `col` inside each prior-minutes bin.

    Unweighted *within* a bin: the bin already holds minutes roughly fixed, and
    weighting inside it would fight the very variation the curve is fitted to. Season
    is still absorbed, because era drift inflates a correlation at every minutes level.
    """
    x = pairs[col].to_numpy(dtype=float)
    y = pairs[f"{col}_next"].to_numpy(dtype=float)
    m = pairs["min_total"].to_numpy(dtype=float)
    season = pairs["season"].to_numpy()

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(m) & (m > 0)
    x, y, m, season = x[ok], y[ok], m[ok], season[ok]

    idx = np.digitize(m, bins[1:-1])
    rows = []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() < 30:
            continue
        r = weighted_corr(demean_within(x[sel], season[sel]),
                          demean_within(y[sel], season[sel]))
        rows.append({"bin_lo": bins[b], "bin_hi": bins[b + 1],
                     "mean_minutes": float(m[sel].mean()), "n": int(sel.sum()), "r": r})
    return pd.DataFrame(rows).dropna(subset=["r"]) if rows else pd.DataFrame()


def fit_reliability(by_bin: pd.DataFrame) -> tuple[float, float]:
    """Fit `r(m) = r_inf · m / (m + m0)` to per-bin correlations, weighted by bin n.

    `m0` is searched over a log grid; at each candidate `r_inf` has a closed form, so
    this is a one-dimensional least-squares problem with no optimizer. Fewer than
    three usable bins cannot pin down two parameters, so the fit is declined.
    """
    if len(by_bin) < 3:
        return np.nan, np.nan
    m = by_bin["mean_minutes"].to_numpy(dtype=float)
    r = by_bin["r"].to_numpy(dtype=float)
    n = by_bin["n"].to_numpy(dtype=float)

    best = (np.inf, np.nan, np.nan)
    for m0 in M0_GRID:
        f = m / (m + m0)
        denom = float((n * f * f).sum())
        if denom <= 0:
            continue
        r_inf = float(np.clip((n * f * r).sum() / denom, 0.0, 1.0))
        sse = float((n * (r - r_inf * f) ** 2).sum())
        if sse < best[0]:
            best = (sse, r_inf, float(m0))
    return best[1], best[2]


def minutes_for_reliability(r_inf: float, m0: float, target: float = 0.75) -> float:
    """Total minutes at which the fitted curve reaches `target`, NaN if it never does."""
    if not np.isfinite(r_inf) or not np.isfinite(m0) or r_inf <= target:
        return np.nan
    return float(target * m0 / (r_inf - target))


# ── The measurement ───────────────────────────────────────────────────────────

def persistence_row(pairs: pd.DataFrame, col: str, weighted: bool) -> dict:
    """Pooled, within-season and unweighted lag-1 persistence for one column."""
    x = pairs[col].to_numpy(dtype=float)
    y = pairs[f"{col}_next"].to_numpy(dtype=float)
    season = pairs["season"].to_numpy()
    w = (pair_weights(pairs["min_total"], pairs["min_total_next"]) if weighted
         else np.ones(len(pairs)))

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, season, w = x[ok], y[ok], season[ok], w[ok]
    if len(x) < 3:
        return {"feature": col, "n_pairs": int(len(x)), "minutes_weighted": weighted}

    dx, dy = demean_within(x, season, w), demean_within(y, season, w)
    return {
        "feature": col,
        "n_pairs": int(len(x)),
        "minutes_weighted": weighted,
        "r_pooled": weighted_corr(x, y, w),
        "r_within_season": weighted_corr(dx, dy, w),
        "r_within_unweighted": weighted_corr(demean_within(x, season),
                                             demean_within(y, season)),
    }


def measure(frame: pd.DataFrame, seasons: list[str], cols: list[str] | None = None,
            max_lag: int = 4, min_pairs: int = 200,
            bins: list[float] = RELIABILITY_BINS) -> pd.DataFrame:
    """Rank every numeric feature by how much of itself it keeps a year later."""
    cols = cols or feature_cols(frame)
    pairs = {lag: lagged_pairs(frame, seasons, lag) for lag in range(1, max_lag + 1)}

    rows = []
    for col in cols:
        if f"{col}_next" not in pairs[1].columns:
            continue
        weighted = is_per36(col)
        row = persistence_row(pairs[1], col, weighted)
        if row["n_pairs"] < min_pairs:
            continue
        row["era_gap"] = row["r_pooled"] - row["r_within_season"]

        # Staleness: the same statistic at longer lags. team_context decays a stale
        # description by a flat 0.96/season; this is where a per-column rate lives.
        for lag in range(1, max_lag + 1):
            p = pairs[lag]
            row[f"r_lag{lag}"] = (persistence_row(p, col, weighted).get("r_within_season")
                                  if f"{col}_next" in p.columns else np.nan)

        by_bin = reliability_by_bin(pairs[1], col, bins)
        r_inf, m0 = fit_reliability(by_bin) if len(by_bin) else (np.nan, np.nan)
        row["reliability_r_inf"] = r_inf
        row["reliability_m0"] = m0
        row["reliability_bins"] = len(by_bin)
        row["minutes_for_r75"] = minutes_for_reliability(r_inf, m0)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("r_within_season", ascending=False).reset_index(drop=True)


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(cfg: dict) -> Path:
    """Measure persistence for both tiers and write one ranked table."""
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    p_cfg = cfg["eda"].get("persistence", {})
    max_lag = p_cfg.get("max_lag", 4)
    min_pairs = p_cfg.get("min_pairs", 200)
    bins = p_cfg.get("reliability_bins", RELIABILITY_BINS)
    seasons = cfg["data"]["seasons"]

    tables = []
    for tier in ("A", "B"):
        # The inclusive twin: the reliability curve needs the low-minute rows the
        # qualification filter removes, and minutes weighting is what makes them safe.
        src = features_dir / f"season_matrix_roster_tier{tier}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found — run `make season-matrix` first")
        frame = pd.read_parquet(src)

        table = measure(frame, seasons, max_lag=max_lag, min_pairs=min_pairs, bins=bins)
        table.insert(0, "tier", tier)
        tables.append(table)

        print(f"\nTier {tier}: {len(table)} features over "
              f"{int(table['n_pairs'].max()):,} consecutive-season pairs")
        show = ["feature", "n_pairs", "r_pooled", "r_within_season", "era_gap",
                "reliability_r_inf", "reliability_m0", "minutes_for_r75"]
        print("  most persistent:")
        print(table.head(12)[show].round(3).to_string(index=False))
        print("  least persistent:")
        print(table.tail(8)[show].round(3).to_string(index=False))

        # The whole reason both figures are reported: where they disagree, the pooled
        # number is the calendar rather than the player.
        era = table.reindex(table["era_gap"].abs().sort_values(ascending=False).index)
        print("  largest pooled-vs-within gaps (era drift wearing a persistence costume):")
        print(era.head(8)[["feature", "r_pooled", "r_within_season", "era_gap"]]
              .round(3).to_string(index=False))

        lags = [f"r_lag{k}" for k in range(1, max_lag + 1) if f"r_lag{k}" in table]
        print("  mean persistence by lag: " +
              ", ".join(f"{c} {table[c].mean():.3f}" for c in lags))

        # This is a generalization of a fit that already exists, so show it landing on
        # the same place: team_context uses r(m) = 0.924 * m/(m+66) for every column.
        per36 = table[table["minutes_weighted"]]
        print(f"  fitted curve across the {len(per36)} per-36 columns: "
              f"median r_inf {per36['reliability_r_inf'].median():.3f}, "
              f"median m0 {per36['reliability_m0'].median():.0f} "
              f"(team_context's single fit: r_inf 0.924, m0 66)")

    out = pd.concat(tables, ignore_index=True)
    dest = out_dir / "persistence.csv"
    out.to_csv(dest, index=False)
    print(f"\nPersistence: {len(out):,} feature × tier rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
