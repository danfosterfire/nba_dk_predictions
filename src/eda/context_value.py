"""Measure what own-team context is worth, above the player's own prior season.

The question every team-context feature has to answer: given that we already know
what P did last season, does knowing who he plays *with* next season tell us
anything more? Anything that survives that control is real; anything that does not
is a restatement of P's own prior form.

Protocol (unchanged from the measurement it supersedes, so the numbers are
comparable):

    controls   P's own season S-1 dk_pts/game, minutes/game and usage
    features   own-team context for P's season-S roster
    outcomes   P's season-S per-36 components, minutes/game, per-game dk_pts

Reported per (feature, outcome): the partial correlation after residualizing both
sides on the controls, and — for per-game dk_pts — the incremental R² of adding the
feature to the control regression.

The split between rate outcomes and the minutes outcome is the point. Team context
acts on them with opposite signs, so a feature can be strongly predictive of both
and still look like nothing against per-game dk_pts, which is their product.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONTROL_COLS = ["prior_dk_pts_per_game", "prior_min", "prior_usg"]

# Per-36 components, as named in the season matrix.
COMPONENT_COLS = {
    "pts_per36": "bas_pts", "fg3m_per36": "bas_fg3m", "reb_per36": "bas_reb",
    "ast_per36": "bas_ast", "stl_per36": "bas_stl", "blk_per36": "bas_blk",
    "tov_per36": "bas_tov",
}

CONTEXT_FEATURES = [
    "role_crowding", "teammate_usage_max", "teammate_usage_sum", "teammate_usage_load",
    "teammate_assist_supply", "teammate_spacing", "team_pace", "roster_coverage",
]


# ── Regression helpers ────────────────────────────────────────────────────────

def _design(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residuals of y after least-squares regression on `controls` plus an intercept."""
    A = _design(controls)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def r_squared(y: np.ndarray, X: np.ndarray) -> float:
    A = _design(X)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / tss if tss > 0 else np.nan


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Correlation of x and y once both are stripped of what the controls explain."""
    rx, ry = residualize(x, controls), residualize(y, controls)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


# ── Panel assembly ────────────────────────────────────────────────────────────

def build_panel(matrix: pd.DataFrame, context: pd.DataFrame,
                seasons: list[str]) -> pd.DataFrame:
    """One row per player-season transition: S-1 controls, S context, S outcomes.

    `context` is already keyed on the season being predicted — the corrected
    construction describes P's season-S roster, so no lagging happens here.
    """
    order = {s: i for i, s in enumerate(seasons)}
    nxt = {s: seasons[i + 1] for s, i in order.items() if i + 1 < len(seasons)}

    prior = matrix[["player_id", "season", "dk_pts_per_game", "min", "adv_usg_pct"]].copy()
    prior = prior.rename(columns={"dk_pts_per_game": "prior_dk_pts_per_game",
                                  "min": "prior_min", "adv_usg_pct": "prior_usg"})
    prior["season"] = prior["season"].map(nxt)
    prior = prior.dropna(subset=["season"])

    outcome_cols = ["player_id", "season", "dk_pts_per_game", "min",
                    *COMPONENT_COLS.values()]
    outcomes = matrix[[c for c in outcome_cols if c in matrix.columns]].copy()
    outcomes = outcomes.rename(columns={v: k for k, v in COMPONENT_COLS.items()})
    outcomes = outcomes.rename(columns={"min": "minutes"})

    panel = outcomes.merge(prior, on=["player_id", "season"], how="inner")
    panel = panel.merge(context, on=["player_id", "season"], how="inner")
    return panel.dropna(subset=CONTROL_COLS + ["dk_pts_per_game", "minutes"])


# ── The measurement ───────────────────────────────────────────────────────────

def season_dummies(seasons: pd.Series) -> np.ndarray:
    """One-hot season indicators, first level dropped (the intercept carries it).

    Mandatory on a 30-season panel. Floor spacing and pace both roughly doubled over
    the sample, and so did scoring, so *any* feature built from them correlates with
    *any* rising outcome unless season is absorbed. Pooled estimates here are era
    trends wearing a team-context costume.
    """
    d = pd.get_dummies(seasons, drop_first=True)
    return d.to_numpy(dtype=float)


def control_matrix(sub: pd.DataFrame, season_fe: bool) -> np.ndarray:
    base = sub[CONTROL_COLS].to_numpy(dtype=float)
    if not season_fe:
        return base
    return np.column_stack([base, season_dummies(sub["season"])])


def measure(panel: pd.DataFrame, features: list[str] = None,
            season_fe: bool = True) -> pd.DataFrame:
    """Partial correlation of every context feature against every outcome."""
    features = features or [f for f in CONTEXT_FEATURES if f in panel.columns]
    outcomes = [c for c in list(COMPONENT_COLS) + ["minutes", "dk_pts_per_game"]
                if c in panel.columns]

    rows = []
    for feat in features:
        sub = panel.dropna(subset=[feat])
        controls = control_matrix(sub, season_fe)
        x = sub[feat].to_numpy(dtype=float)
        base_r2 = r_squared(sub["dk_pts_per_game"].to_numpy(dtype=float), controls)
        with_r2 = r_squared(sub["dk_pts_per_game"].to_numpy(dtype=float),
                            np.column_stack([controls, x]))
        row = {"feature": feat, "n": len(sub),
               "r2_controls": base_r2, "r2_with_feature": with_r2,
               "delta_r2": with_r2 - base_r2}
        for out in outcomes:
            ok = sub[out].notna().to_numpy()
            row[f"r_{out}"] = partial_corr(x[ok], sub[out].to_numpy(dtype=float)[ok],
                                           controls[ok])
        rows.append(row)
    return pd.DataFrame(rows)


def block_value(panel: pd.DataFrame, features: list[str] = None,
                season_fe: bool = True) -> dict:
    """Incremental R² of the whole context block on per-game dk_pts."""
    features = features or [f for f in CONTEXT_FEATURES if f in panel.columns]
    sub = panel.dropna(subset=features)
    controls = control_matrix(sub, season_fe)
    y = sub["dk_pts_per_game"].to_numpy(dtype=float)
    base = r_squared(y, controls)
    full = r_squared(y, np.column_stack([controls, sub[features].to_numpy(dtype=float)]))
    return {"n": len(sub), "r2_controls": base, "r2_with_block": full,
            "delta_r2": full - base}


def run(cfg: dict, tier: str = "A") -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_parquet(features_dir / f"season_matrix_tier{tier}.parquet")
    context = pd.read_parquet(features_dir / f"team_context_tier{tier}.parquet")
    panel = build_panel(matrix, context, cfg["data"]["seasons"])

    show = ["feature", "n", "r_pts_per36", "r_minutes", "r_dk_pts_per_game", "delta_r2"]
    print(f"\nTier {tier}: {len(panel):,} player-season transitions")

    for season_fe in (False, True):
        table = measure(panel, season_fe=season_fe)
        block = block_value(panel, season_fe=season_fe)
        label = "with season fixed effects" if season_fe else "pooled (no season control)"
        print(f"\n── {label} ──")
        print(f"Controls alone: R² = {block['r2_controls']:.4f}   "
              f"+ context block: R² = {block['r2_with_block']:.4f} "
              f"(+{block['delta_r2']:.4f}, n={block['n']:,})")
        print(table[[c for c in show if c in table]].round(4).to_string(index=False))
        if season_fe:
            dest = out_dir / f"team_context_value_tier{tier}.csv"
            table.to_csv(dest, index=False)

    print(f"\n→ {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    for t in ("A", "B"):
        run(cfg, t)
