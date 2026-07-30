"""Measure the consensus → DraftKings ADP transfer function, and profile the panel.

Turns the numbers in `docs/adp-plan.md` from a planning-session scratch script into a
reproducible target. Two questions:

1. **How far is a Yahoo/ESPN consensus from DK's actual board?** Measured on paired
   observations — a DK board and a consensus board for the same season. Answer as of
   writing: Spearman 0.870, mean absolute rank gap 21.9 picks, and DK drafting centers
   ~11.9 picks earlier because category-league ADP discounts them for FT% while DK Best
   Ball pays rebounds 1.25 and blocks 2.0 flat.

2. **How much correction can the anchors support?** A ladder from "use consensus raw"
   through a monotone recalibration to a position-offset model, all cross-validated.
   Measured: 24.02 → 16.99 picks for the monotone step, and only −0.29 more for the C/F/G
   offset. One anchor identifies a shape, not a player-level model — so the fitted
   artifact is deliberately ~one degree of freedom.

**`n_anchors` is written beside the fit and must be reported with any result that uses
it.** With one paired season the map is *fitted* but not *validated*: cross-validation
within a season measures how well the shape generalizes across players, not across
seasons. That distinction is exactly the one this repo has been burned by before.

Usage:
    python -m src.eda.adp_profile
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

# A DK board and a consensus board count as the same observation if captured within this
# many days. Both are frozen-ish over short spans, and the real anchors are 12 days apart.
PAIR_WINDOW_DAYS = 45
MIN_PAIRS = 40
N_FOLDS = 5
SEED = 0

TIER_EDGES = [0, 24, 48, 96, 10_000]
TIER_LABELS = ["R1-2", "R3-4", "R5-8", "R9+"]


def paired_observations(panel: pd.DataFrame,
                        window_days: int = PAIR_WINDOW_DAYS) -> pd.DataFrame:
    """DK ADP alongside consensus ADP for the same player and season.

    Uses the consensus capture nearest in time to the DK capture, so a season with many
    consensus snapshots contributes one comparison rather than several correlated ones.
    """
    dk = panel[(panel.snapshot_source == "draftkings") & panel.adp.notna()
               & panel.player_id.notna()].copy()
    cons = panel[(panel.source_detail == "avg") & panel.adp.notna()
                 & panel.player_id.notna()].copy()
    if dk.empty or cons.empty:
        return pd.DataFrame()

    out = []
    for (season, dk_day), g in dk.groupby(["season", "as_of_date"]):
        same = cons[cons.season == season]
        if same.empty:
            continue
        gap = (pd.to_datetime(same.as_of_date) - pd.Timestamp(dk_day)).dt.days.abs()
        best = same.as_of_date.iloc[gap.argmin()]
        if gap.min() > window_days:
            continue
        c = same[same.as_of_date == best]
        merged = g.merge(c[["player_id", "adp", "position"]], on="player_id",
                         suffixes=("_dk", "_cons"))
        merged["consensus_date"] = best
        merged["pair_gap_days"] = int(gap.min())
        out.append(merged)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame())


# ── The ladder ────────────────────────────────────────────────────────────────

def _cv_mae(x: np.ndarray, y: np.ndarray, fit_predict) -> float:
    errs = []
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(x):
        errs.append(np.abs(fit_predict(tr, te) - y[te]))
    return float(np.mean(np.concatenate(errs)))


def transfer_ladder(pairs: pd.DataFrame) -> pd.DataFrame:
    """Cross-validated MAE for each way of turning consensus ADP into DK ADP."""
    x = pairs["adp_cons"].to_numpy(float)
    y = pairs["adp_dk"].to_numpy(float)
    pos = pairs["position_dk"].to_numpy()

    def linear(tr, te):
        return np.polyval(np.polyfit(x[tr], y[tr], 1), x[te])

    def iso(tr, te):
        return IsotonicRegression(out_of_bounds="clip").fit(x[tr], y[tr]).predict(x[te])

    def _with_offset(base_fn):
        def inner(tr, te):
            base_all = base_fn(np.arange(len(x)), np.arange(len(x)))
            resid = y - base_all
            off = {p: resid[tr][pos[tr] == p].mean() if (pos[tr] == p).any() else 0.0
                   for p in set(pos)}
            return base_all[te] + np.array([off.get(p, 0.0) for p in pos[te]])
        return inner

    rows = [
        ("consensus raw", float(np.mean(np.abs(x - y)))),
        ("consensus order vs DK order",
         float(np.mean(np.abs(pd.Series(x).rank() - pd.Series(y).rank())))),
        ("+ linear rescale", _cv_mae(x, y, linear)),
        ("+ monotone (isotonic) rescale", _cv_mae(x, y, iso)),
        ("+ linear + C/F/G offset", _cv_mae(x, y, _with_offset(linear))),
        ("+ isotonic + C/F/G offset", _cv_mae(x, y, _with_offset(iso))),
    ]
    uncensored = ~pairs["adp_censored"].to_numpy(bool)
    if uncensored.sum() > MIN_PAIRS:
        xu, yu = x[uncensored], y[uncensored]
        rows.append(("+ isotonic, uncensored only",
                     _cv_mae(xu, yu, lambda tr, te: IsotonicRegression(
                         out_of_bounds="clip").fit(xu[tr], yu[tr]).predict(xu[te]))))
    return pd.DataFrame(rows, columns=["method", "cv_mae_picks"])


def agreement(pairs: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    gap = pairs["adp_cons"].rank() - pairs["adp_dk"].rank()
    return {
        "n_pairs": len(pairs),
        "spearman": float(spearmanr(pairs.adp_dk, pairs.adp_cons).statistic),
        "mean_abs_rank_gap": float(gap.abs().mean()),
        "median_abs_rank_gap": float(gap.abs().median()),
        "share_gap_over_20": float((gap.abs() > 20).mean()),
        "share_gap_over_40": float((gap.abs() > 40).mean()),
        "share_censored": float(pairs.adp_censored.mean()),
    }


def position_bias(pairs: pd.DataFrame) -> pd.DataFrame:
    """Signed rank gap by DK position. Positive = DK drafts the position earlier.

    The mechanism is a scoring-system difference, not noise: Yahoo/ESPN ADP is
    category-league ADP, where centers are discounted for wrecking FT% and supplying no
    3PM or assists. Uncorrected, it reads as model edge on exactly one position.
    """
    p = pairs.assign(gap=pairs["adp_cons"].rank() - pairs["adp_dk"].rank())
    return (p.groupby("position_dk")["gap"]
            .agg(mean_rank_gap="mean", median_rank_gap="median", n="size")
            .reset_index().rename(columns={"position_dk": "position"}))


def tier_gaps(pairs: pd.DataFrame) -> pd.DataFrame:
    """Disagreement by consensus tier — it is concentrated in the late rounds."""
    p = pairs.assign(
        gap=(pairs["adp_cons"].rank() - pairs["adp_dk"].rank()).abs(),
        tier=pd.cut(pairs["adp_cons"], TIER_EDGES, labels=TIER_LABELS))
    return (p.groupby("tier", observed=True)["gap"]
            .agg(mean_abs_rank_gap="mean", n="size").reset_index())


def fit_transfer(pairs: pd.DataFrame) -> pd.DataFrame:
    """The shipped artifact: a monotone map from consensus ADP onto DK's scale."""
    x = pairs["adp_cons"].to_numpy(float)
    y = pairs["adp_dk"].to_numpy(float)
    model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    grid = np.arange(1, int(np.ceil(x.max())) + 1, dtype=float)
    return pd.DataFrame({
        "adp_consensus": grid,
        "adp_dk_fitted": model.predict(grid),
        "n_anchors": pairs["season"].nunique(),
        "n_pairs": len(pairs),
        "fitted_on_seasons": "|".join(sorted(pairs["season"].unique())),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def run(features_dir: str | Path = "data/features",
        output_dir: str | Path = "outputs/eda") -> pd.DataFrame:
    features_dir, output_dir = Path(features_dir), Path(output_dir)
    panel = pd.read_parquet(features_dir / "adp_panel.parquet")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = paired_observations(panel)
    if len(pairs) < MIN_PAIRS:
        print(f"  insufficient_pairs: {len(pairs)} paired observations "
              f"(need {MIN_PAIRS}). A DK board and a consensus board for the same season "
              f"are required; the consensus source is frozen at the previous season until "
              f"its drafts open, so a new DK board has no counterpart until then.")
        return pairs

    stats = agreement(pairs)
    ladder = transfer_ladder(pairs)
    bias = position_bias(pairs)
    tiers = tier_gaps(pairs)
    transfer = fit_transfer(pairs)

    transfer.to_parquet(features_dir / "adp_transfer.parquet", index=False)
    report = output_dir / "adp_profile.csv"
    pd.concat([
        pd.DataFrame({"section": "agreement", "metric": list(stats),
                      "value": [stats[k] for k in stats]}),
        ladder.assign(section="ladder").rename(
            columns={"method": "metric", "cv_mae_picks": "value"}),
        bias.assign(section="position_bias").melt(
            id_vars=["section", "position"], var_name="metric", value_name="value"
        ).assign(metric=lambda d: d.position + ":" + d.metric).drop(columns="position"),
        tiers.assign(section="tier_gap").melt(
            id_vars=["section", "tier"], var_name="metric", value_name="value"
        ).assign(metric=lambda d: d.tier.astype(str) + ":" + d.metric).drop(columns="tier"),
    ], ignore_index=True)[["section", "metric", "value"]].to_csv(report, index=False)

    print(f"  {stats['n_pairs']:,} paired observations over "
          f"{pairs.season.nunique()} season(s): {', '.join(sorted(pairs.season.unique()))}")
    print(f"  Spearman {stats['spearman']:.4f}   mean |rank gap| "
          f"{stats['mean_abs_rank_gap']:.1f} picks   "
          f"{stats['share_gap_over_20']:.1%} differ by >20")
    print()
    for _, r in ladder.iterrows():
        print(f"    {r.method:34s} {r.cv_mae_picks:6.2f}")
    print()
    print("  signed rank gap by DK position (+ = DK drafts earlier):")
    for _, r in bias.iterrows():
        print(f"    {r.position}: {r.mean_rank_gap:+6.2f}  (n={int(r.n)})")
    print()
    print("  mean |rank gap| by consensus tier:")
    for _, r in tiers.iterrows():
        print(f"    {r.tier:5s} {r.mean_abs_rank_gap:6.2f}  (n={int(r.n)})")
    n_anchors = int(transfer.n_anchors.iloc[0])
    print()
    print(f"  Wrote transfer function (n_anchors={n_anchors}) → "
          f"{features_dir / 'adp_transfer.parquet'}")
    if n_anchors < 2:
        print("  ⚠ n_anchors=1: the map is FITTED, not validated. Cross-validation here "
              "measures generalization across players within one season, not across "
              "seasons. Report any downstream result both with and without it.")
    print(f"  Wrote report → {report}")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure the consensus → DraftKings ADP transfer function.")
    parser.add_argument("--features-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(features_dir=args.features_dir
        or cfg.get("data", {}).get("features_dir", "data/features"),
        output_dir=cfg.get("eda", {}).get("output_dir", "outputs/eda"))
