"""Component-wise targets for dk_pts, and the arithmetic to put them back together.

dk_pts is a linear function of the box score plus a nonlinear bonus, so it is both
easier and more informative to predict the parts:

    dk_pts = [ P(play) · minutes ] × Σ wᵢ·rateᵢ  +  E[bonus]

Three reasons this beats a single dk_pts head, all measured on this dataset:

1. Team context cancels *across components*. Per sd, `teammate_assist_supply` moves
   ast/36 by −0.361, reb/36 by +0.218 and blk/36 by +0.191; DK-weighting those gives
   2.11 dk_pts of gross component movement against 0.25 net — an 8.3x cancellation.
   role_crowding cancels 8.2x, teammate_spacing 5.0x, team_pace 4.7x. A single
   dk_pts head sees a small fraction of what the component heads see.
2. Opponent effects cancel the same way: the per-component weighted opponent sd sums
   to 1.105 dk_pts against 0.785 measured on dk_pts directly.
3. The double-double bonus is a threshold on five components at ≥10, so
   E[bonus] ≠ bonus(E[components]). A dk_pts regression cannot represent it at all;
   component predictions plus a distribution can.

`compute_dk_pts` in src/data/preprocess.py stays the single source of truth for the
scoring rule — `dk_from_components` is asserted against it in the tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import compute_dk_pts

# The linear part of DraftKings NBA scoring.
DK_WEIGHTS: dict[str, float] = {
    "pts": 1.0, "fg3m": 0.5, "reb": 1.25, "ast": 1.5,
    "stl": 2.0, "blk": 2.0, "tov": -0.5,
}
COMPONENTS = list(DK_WEIGHTS)

# Categories counted toward the double/triple-double bonus, and its payout by count.
BONUS_CATEGORIES = ["pts", "reb", "ast", "stl", "blk"]
BONUS_THRESHOLD = 10
BONUS_BY_COUNT = {0: 0.0, 1: 0.0, 2: 1.5, 3: 4.5, 4: 4.5, 5: 4.5}

RATE_SUFFIX = "_per36"
EXPOSURE_MINUTES = 36.0

# Shot classes, for modelling 2s, 3s and free throws separately. Deliberately *not* in
# DK_WEIGHTS: `pts` already carries them, and adding them to the DK sum would
# double-count. They are a decomposition *of* `pts`, offered alongside it.
SHOT_CLASSES = ["fg2m", "fg3m", "ftm", "fg2a", "fg3a", "fta"]


# ── Recombination ─────────────────────────────────────────────────────────────

def linear_part(df: pd.DataFrame) -> pd.Series:
    """Σ wᵢ·xᵢ over the box-score components."""
    return sum(df[c] * w for c, w in DK_WEIGHTS.items())


def bonus_part(df: pd.DataFrame) -> pd.Series:
    """The double/triple-double bonus — a step function, not a linear term."""
    cats = (df[BONUS_CATEGORIES] >= BONUS_THRESHOLD).sum(axis=1)
    return cats.map(BONUS_BY_COUNT)


def dk_from_components(df: pd.DataFrame) -> pd.Series:
    """Reassemble dk_pts from realized components. Matches `compute_dk_pts` exactly."""
    return linear_part(df) + bonus_part(df)


# ── Shot classes ──────────────────────────────────────────────────────────────

def add_shot_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Add `fg2m` / `fg2a` in place — two-pointers are not stored anywhere.

    `FGM` and `FGA` **include** threes, so two-point makes are `fgm - fg3m`, not `fgm`.
    This is the single most common arithmetic slip in this dataset, which is why the
    subtraction lives in one function instead of being retyped at each call site.
    """
    if {"fgm", "fg3m"} <= set(df.columns):
        df["fg2m"] = df["fgm"] - df["fg3m"]
    if {"fga", "fg3a"} <= set(df.columns):
        df["fg2a"] = df["fga"] - df["fg3a"]
    return df


def pts_from_shot_classes(df: pd.DataFrame) -> pd.Series:
    """`pts = 2·fg2m + 3·fg3m + ftm` — exact, and the basis the rate heads reassemble in.

    Equivalent to `2·fgm + fg3m + ftm` on the stored columns; the two differ only in
    whether threes have already been counted once inside `fgm`. Asserted against `pts`
    itself in the tests, both ways.
    """
    return 2.0 * df["fg2m"] + 3.0 * df["fg3m"] + df["ftm"]


# DraftKings pays 1.0 per point plus a further 0.5 per made three, so in the shot-class
# basis a three is worth 3 + 0.5 = 3.5.
DK_SHOT_CLASS_WEIGHTS: dict[str, float] = {"fg2m": 2.0, "fg3m": 3.5, "ftm": 1.0}


def dk_scoring_from_shot_classes(df: pd.DataFrame) -> pd.Series:
    """The scoring share of dk_pts (`1.0·pts + 0.5·fg3m`), from the shot classes.

    Everything else in the DK sum — rebounds, assists, steals, blocks, turnovers and the
    bonus — is untouched by this decomposition.
    """
    return sum(df[c] * w for c, w in DK_SHOT_CLASS_WEIGHTS.items())


# ── Expected bonus ────────────────────────────────────────────────────────────

# Calibrated against 11,627 player-seasons of realized bonus (2014-15 → 2025-26):
# independent sampling is 23% too low (0.098 vs 0.127 dk_pts/game realized), and this
# value brings the mean bias to +0.001 with good fit across minutes buckets.
BONUS_OVERDISPERSION = 0.10


def expected_bonus(expected_counts: np.ndarray, overdispersion: float = BONUS_OVERDISPERSION,
                   n_samples: int = 512, seed: int = 42,
                   chunk: int = 20_000) -> np.ndarray:
    """Monte-Carlo E[bonus] from expected counts for the five bonus categories.

    `expected_counts` is (n, 5) in BONUS_CATEGORIES order.

    Components are positively correlated — a big night is big everywhere — so they are
    drawn as Poisson counts sharing a per-game frailty g ~ Gamma(k, 1/k) with mean 1
    and variance `overdispersion`. That gives negative-binomial marginals *and* the
    positive dependence that independent sampling would miss, which matters because
    the bonus rewards several categories clearing 10 in the same game.

    Chunked over rows: the sample tensor is n_samples × chunk × 5.
    """
    counts = np.asarray(expected_counts, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != len(BONUS_CATEGORIES):
        raise ValueError(f"expected (n, {len(BONUS_CATEGORIES)}), got {counts.shape}")

    rng = np.random.default_rng(seed)
    k = 1.0 / max(overdispersion, 1e-9)
    payout = np.array([BONUS_BY_COUNT[i] for i in range(len(BONUS_CATEGORIES) + 1)])
    out = np.empty(len(counts))

    for start in range(0, len(counts), chunk):
        block = counts[start:start + chunk]
        g = rng.gamma(k, 1.0 / k, size=(n_samples, len(block), 1))
        draws = rng.poisson(np.clip(block[None, :, :] * g, 0, None))
        cats = (draws >= BONUS_THRESHOLD).sum(axis=-1)
        out[start:start + chunk] = payout[cats].mean(axis=0)
    return out


def expected_dk_pts(minutes: np.ndarray, rates_per36: pd.DataFrame,
                    play_prob: np.ndarray | None = None, **bonus_kw) -> np.ndarray:
    """Expected dk_pts from predicted minutes and per-36 rates.

    Counts are exposure-scaled rates (`rate * minutes / 36`), the linear part is their
    weighted sum, and the bonus comes from `expected_bonus`. Scaled by P(play) when
    given, since a DNP scores zero.
    """
    exposure = np.asarray(minutes, dtype=float) / EXPOSURE_MINUTES
    counts = {c: rates_per36[c].to_numpy(dtype=float) * exposure for c in COMPONENTS}
    linear = sum(counts[c] * w for c, w in DK_WEIGHTS.items())
    bonus = expected_bonus(np.column_stack([counts[c] for c in BONUS_CATEGORIES]), **bonus_kw)
    total = linear + bonus
    return total if play_prob is None else total * np.asarray(play_prob, dtype=float)


# ── Target construction ───────────────────────────────────────────────────────

def build_component_targets(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Per-game component targets, plus minutes, a played flag and per-36 rates.

    Expects lowercased game-log columns (see src/data/preprocess.py::RENAME).
    Rates are left NaN for zero-minute rows rather than filled — a player who did not
    play has no rate, and the rate heads must be masked there, not trained on a zero.
    """
    df = game_logs.copy()
    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    df = df.dropna(subset=["min"] + COMPONENTS)

    df["played"] = (df["min"] > 0).astype(int)
    df["dk_pts"] = compute_dk_pts(df)
    df["dk_linear"] = linear_part(df)
    df["dk_bonus"] = bonus_part(df)

    # The shot-class decomposition of `pts`, carried alongside the DK components so 2s,
    # 3s and free throws can be modelled separately. `pts` is a *weighted sum* of these,
    # and the 2x weight on field goals is what makes `pts` look overdispersed while the
    # classes themselves are near-Poisson — see CLAUDE.md.
    add_shot_classes(df)

    exposure = (df["min"] / EXPOSURE_MINUTES).where(df["min"] > 0)
    for c in COMPONENTS + [c for c in SHOT_CLASSES if c in df.columns]:
        df[c + RATE_SUFFIX] = df[c] / exposure
    return df


def season_totals(targets: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, season) totals — the "running season total" half of the goal."""
    agg = {"dk_pts": ["sum", "mean", "std"], "min": ["sum", "mean"], "played": "sum"}
    agg |= {c: "sum" for c in COMPONENTS}
    out = targets.groupby(["player_id", "season"]).agg(agg)
    out.columns = ["_".join(c).rstrip("_") for c in out.columns]
    return out.rename(columns={"played_sum": "games_played"}).reset_index()


def run(cfg: dict) -> Path:
    from src.data.preprocess import clean, load_raw

    raw = load_raw(cfg["data"]["raw_dir"])
    df = clean(raw, min_games=cfg["data"]["min_games"])
    targets = build_component_targets(df)

    out_dir = Path(cfg["data"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "component_targets.parquet"
    targets.to_parquet(dest, index=False)

    check = np.abs(dk_from_components(targets) - targets["dk_pts"]).max()
    print(f"Component targets: {len(targets):,} player-games → {dest}")
    print(f"  recombination error vs compute_dk_pts: {check:.2e}")
    print(f"  bonus share of total dk_pts: "
          f"{targets['dk_bonus'].sum() / targets['dk_pts'].sum():.2%}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
