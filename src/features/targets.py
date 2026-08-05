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

from src.data.preprocess import (FIT_WINDOWS, FULL_WINDOW, TRAIN_VAL_WINDOW,
                                 compute_dk_pts, fit_window, held_out_seasons)

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

# Variance of the shared per-game frailty `g` in `expected_bonus` — NOT a dispersion of
# dk_pts, and not fitted by any head. It does two jobs at once: it makes each category's
# marginal negative binomial (`var = mu + overdispersion*mu^2`), and because `g` is shared
# across the five categories it induces the positive dependence the bonus needs.
#
# **This value is calibrated at the player-SEASON unit** — expected counts are a
# player-season's mean per-game counts — where it must also stand in for the minutes
# variation that a season mean hides. Re-run with `make component-targets`
# (`bonus_calibration` → `outputs/eda/bonus_calibration.csv`), which reproduces the two
# recorded facts on 11,938 qualifying player-seasons: aggregate bias **+0.0009** dk_pts/game
# and independent sampling **22.7%** too low.
#
# Two things that artifact adds, and the second is a correction:
#   - Fitted exactly, the season-unit optimum is **0.0968**, i.e. 0.10 to two figures.
#   - The aggregate fit is good but the **per-mpg-bucket fit is not** — at 0.10 the bias
#     runs -0.014 at 12-18 mpg against +0.029 at 30-48, a 0.043 spread that cancels to
#     +0.001. One scalar frailty cannot absorb minutes variation whose *relative* size
#     differs by bucket.
# **The simulator draws per game and must therefore use ~0.025, not this value** — see
# `BONUS_GAME_OVERDISPERSION`.
BONUS_OVERDISPERSION = 0.10

# The same quantity for per-GAME expected counts (per-36 rate x that game's actual minutes),
# which is how `expected_dk_pts` is called and how the simulator will draw. Minutes are no
# longer hidden inside the frailty, so the residual overdispersion is ~4x smaller — and at
# this value the fit holds across every minutes bucket (bias -0.0004 to +0.0007 dk_pts/game)
# rather than only in aggregate. Using 0.10 per game over-predicts the bonus by +0.036
# dk_pts/game for 30+ minute players, who are exactly the ones the bonus is worth most for.
BONUS_GAME_OVERDISPERSION = 0.025


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


# ── Bonus calibration ─────────────────────────────────────────────────────────
#
# `BONUS_OVERDISPERSION` above is the one constant in this project that `CLAUDE.md`
# forbids changing without re-running a calibration — and until this section existed there
# was no target that re-ran it. Two units, because they do not agree and the gap is the
# finding:
#
#   player_season   expected counts are the player-season's mean per-game counts. This is
#                   the unit the shipped 0.10 was fitted on, and the frailty absorbs both
#                   game-to-game count overdispersion *and* the minutes variation the
#                   season mean hides.
#   player_game     expected counts are the player-season per-36 rate x that game's actual
#                   minutes — the way `expected_dk_pts` is actually called. Minutes are no
#                   longer hidden, so there is less for the frailty to absorb and the same
#                   0.10 over-predicts.
#
# Both are reported at every grid point, with an mpg / minutes bucket break, because "it
# holds across buckets" is a claim and not a courtesy.

# Season minutes a player-season needs before its per-game rate is a meaningful prediction.
# The project's standing qualification threshold — reliability ~0.75 (`CLAUDE.md`), and the
# same value `models.component_rates.MIN_PRIOR_MINUTES` and the Stan minutes head use.
BONUS_MIN_SEASON_MINUTES = 200

BONUS_MPG_BUCKETS = [0.0, 12.0, 18.0, 24.0, 30.0, 48.0]
BONUS_OVERDISPERSION_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30]

# The per-game arm is ~60x the rows, so it gets three points rather than the whole grid:
# independent sampling, its own calibrated value, and the season-unit value — which is the
# comparison that shows why the two must not be interchanged.
BONUS_GAME_GRID = [0.0, BONUS_GAME_OVERDISPERSION, BONUS_OVERDISPERSION]

# Monte-Carlo draws for the per-game arm. The quantity is a mean over ~700k rows, so the
# per-row MC sd of ~0.6/sqrt(256) averages down to ~4e-5 — four orders below the +/-0.001
# bias the calibration is judged on. Halving the shipped default halves a 65-second pass.
BONUS_CALIBRATION_SAMPLES = 256

# |bias| above which `run` warns. The recorded calibration lands at +0.001.
BONUS_BIAS_TOLERANCE = 0.005


def _buckets(values: pd.Series, edges: list[float]) -> pd.Series:
    """Half-open `lo-hi` labels, top edge inclusive — mirrors `eda.target.bucket_labels`.

    Written here rather than imported because `src/eda` depends on `src/features`, never
    the other way round.
    """
    idx = np.clip(np.digitize(pd.to_numeric(values, errors="coerce"), edges[1:-1]),
                  0, len(edges) - 2)
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    return pd.Series([labels[i] for i in idx], index=values.index, dtype=object)


def bonus_frames(targets: pd.DataFrame,
                 min_season_minutes: float = BONUS_MIN_SEASON_MINUTES
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player-season, player-game) calibration frames over the same qualifying seasons.

    Both carry `expected_*` columns for the five bonus categories and a `realized` bonus,
    so the two units differ only in what the expected counts are conditioned on. Scoped to
    played games: a DNP scores zero on both sides and would only dilute the bias.
    """
    played = targets[targets["min"] > 0]
    season = played.groupby(["player_id", "season"], as_index=False).agg(
        **{c: (c, "sum") for c in BONUS_CATEGORIES},
        total_minutes=("min", "sum"), games=("min", "size"),
        realized=("dk_bonus", "mean"))
    season = season[season["total_minutes"] >= min_season_minutes].reset_index(drop=True)
    season["mpg"] = season["total_minutes"] / season["games"]
    for c in BONUS_CATEGORIES:
        season[f"rate_{c}"] = season[c] / season["total_minutes"] * EXPOSURE_MINUTES
        season[f"expected_{c}"] = season[f"rate_{c}"] * season["mpg"] / EXPOSURE_MINUTES

    rate_cols = ["player_id", "season"] + [f"rate_{c}" for c in BONUS_CATEGORIES]
    game = played.merge(season[rate_cols], on=["player_id", "season"], how="inner")
    for c in BONUS_CATEGORIES:
        game[f"expected_{c}"] = game[f"rate_{c}"] * game["min"] / EXPOSURE_MINUTES
    game = game.rename(columns={"dk_bonus": "realized"})
    return season, game


def _expected_columns(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame[f"expected_{c}"].to_numpy(dtype=float)
                            for c in BONUS_CATEGORIES])


def _calibration_row(unit: str, bucket: str, overdispersion: float,
                     realized: np.ndarray, predicted: np.ndarray) -> dict:
    r, p = float(np.mean(realized)), float(np.mean(predicted))
    return {"analysis": "calibration", "unit": unit, "bucket": bucket,
            "overdispersion": overdispersion, "n": int(len(realized)),
            "realized_mean_bonus": r, "expected_mean_bonus": p, "bias": p - r,
            "relative_bias": (p - r) / r if r > 0 else np.nan,
            "is_shipped": overdispersion == BONUS_OVERDISPERSION,
            "is_independent": overdispersion == 0.0}


def calibration_rows(frame: pd.DataFrame, unit: str, bucket_col: str,
                     edges: list[float], overdispersion: float,
                     n_samples: int = BONUS_CALIBRATION_SAMPLES, seed: int = 42) -> list[dict]:
    """One `all` row plus one row per bucket, from a single Monte-Carlo pass.

    `expected_bonus` returns a per-row expectation, so the bucket break costs nothing
    beyond the grouping — which is why every grid point can afford to carry it.
    """
    predicted = expected_bonus(_expected_columns(frame), overdispersion=overdispersion,
                               n_samples=n_samples, seed=seed)
    realized = frame["realized"].to_numpy(dtype=float)
    rows = [_calibration_row(unit, "all", overdispersion, realized, predicted)]
    labels = _buckets(frame[bucket_col], edges)
    for bucket in labels.dropna().unique():
        m = (labels == bucket).to_numpy()
        rows.append(_calibration_row(unit, str(bucket), overdispersion,
                                     realized[m], predicted[m]))
    return rows


def zero_bias_overdispersion(rows: list[dict], unit: str) -> dict | None:
    """Interpolate the grid's `all` rows to the overdispersion that zeroes the bias.

    E[bonus] rises monotonically in the frailty variance, so a sign change between two grid
    points brackets the optimum and a linear interpolation between them is enough — the
    grid spacing (0.025) is already finer than the calibration's own tolerance.
    """
    pts = sorted(((r["overdispersion"], r["bias"], r) for r in rows
                  if r["unit"] == unit and r["bucket"] == "all"), key=lambda t: t[0])
    for (od_lo, b_lo, lo), (od_hi, b_hi, _) in zip(pts, pts[1:]):
        if b_lo <= 0 <= b_hi and b_hi != b_lo:
            od = od_lo + (od_hi - od_lo) * (-b_lo) / (b_hi - b_lo)
            return {"analysis": "fitted", "unit": unit, "bucket": "all",
                    "overdispersion": od, "n": lo["n"],
                    "realized_mean_bonus": lo["realized_mean_bonus"],
                    "expected_mean_bonus": np.nan, "bias": 0.0, "relative_bias": 0.0,
                    "is_shipped": False, "is_independent": False}
    return None


def bonus_calibration(targets: pd.DataFrame,
                      min_season_minutes: float = BONUS_MIN_SEASON_MINUTES,
                      grid: list[float] = None, game_grid: list[float] = None,
                      n_samples: int = BONUS_CALIBRATION_SAMPLES,
                      seed: int = 42) -> pd.DataFrame:
    """The whole calibration record: both units, the grid, the buckets, the fitted value.

    The player-season arm sweeps the full grid (a few seconds a point). The player-game arm
    is ~60x more rows, so it gets `game_grid` — independent sampling, the shipped value, and
    one point near its own optimum, which is enough to bracket the fit and to show what the
    bucket biases do between the two.
    """
    grid = list(BONUS_OVERDISPERSION_GRID if grid is None else grid)
    if BONUS_OVERDISPERSION not in grid:
        grid = sorted(grid + [BONUS_OVERDISPERSION])
    game_grid = list(BONUS_GAME_GRID if game_grid is None else game_grid)

    out = []
    for window in FIT_WINDOWS:
        rows = _calibration_window(fit_window(targets, window), min_season_minutes,
                                   grid, game_grid, n_samples, seed)
        frame = pd.DataFrame(rows)
        frame.insert(0, "fit_window", window)
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def _calibration_window(targets: pd.DataFrame, min_season_minutes: float,
                        grid: list[float], game_grid: list[float],
                        n_samples: int, seed: int) -> list[dict]:
    """Both units' grids plus their fitted optima, on one window's rows.

    Split out of `bonus_calibration` when the fit window became a second axis. The
    overdispersion is a **simulator input** — the variance of the shared per-game Gamma
    frailty `expected_bonus` draws — so fitting it on every season would calibrate the
    bonus on the seasons the simulator is later scored against. Nothing here is a fit in
    the train/test sense, which is exactly why no split guard would have caught it.
    """
    rows = []
    season, game = bonus_frames(targets, min_season_minutes)
    for od in grid:
        rows += calibration_rows(season, "player_season", "mpg", BONUS_MPG_BUCKETS,
                                 od, n_samples, seed)
    for od in game_grid:
        rows += calibration_rows(game, "player_game", "min", BONUS_MPG_BUCKETS,
                                 od, n_samples, seed)
    for unit in ("player_season", "player_game"):
        fitted = zero_bias_overdispersion(rows, unit)
        if fitted is not None:
            rows.append(fitted)
    return rows


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

    # ── the bonus calibration, re-run rather than asserted ───────────────────
    b_cfg = cfg.get("features", {}).get("bonus_calibration", {})
    cal = bonus_calibration(
        targets,
        b_cfg.get("min_season_minutes", BONUS_MIN_SEASON_MINUTES),
        b_cfg.get("overdispersion_grid", BONUS_OVERDISPERSION_GRID),
        b_cfg.get("game_overdispersion_grid", BONUS_GAME_GRID),
        b_cfg.get("n_samples", BONUS_CALIBRATION_SAMPLES),
        b_cfg.get("seed", 42))
    eda_dir = Path(cfg["eda"]["output_dir"])
    eda_dir.mkdir(parents=True, exist_ok=True)
    cal_dest = eda_dir / "bonus_calibration.csv"
    cal.to_csv(cal_dest, index=False)

    show = ["unit", "bucket", "overdispersion", "n", "realized_mean_bonus",
            "expected_mean_bonus", "bias", "relative_bias"]
    allrows = cal[(cal["analysis"] == "calibration") & (cal["bucket"] == "all")
                  & (cal["fit_window"] == FULL_WINDOW)]
    print(f"\nBonus calibration — E[bonus] against realized, dk_pts/game "
          f"(player-seasons with >= "
          f"{b_cfg.get('min_season_minutes', BONUS_MIN_SEASON_MINUTES):g} season minutes):")
    print(allrows[show].round(4).to_string(index=False))

    for unit in ("player_season", "player_game"):
        ship = allrows[(allrows["unit"] == unit) & allrows["is_shipped"]]
        indep = allrows[(allrows["unit"] == unit) & allrows["is_independent"]]
        if ship.empty or indep.empty:
            continue
        bias = float(ship["bias"].iloc[0])
        print(f"\n  {unit}: shipped overdispersion {BONUS_OVERDISPERSION:g} biases "
              f"{bias:+.4f} dk_pts/game; independent sampling reads "
              f"{indep['relative_bias'].iloc[0]:+.1%}")
        tol = b_cfg.get("bias_tolerance", BONUS_BIAS_TOLERANCE)
        if unit == "player_season" and abs(bias) > tol:
            print(f"  ⚠️  |bias| {abs(bias):.4f} exceeds {tol:g} at the unit the shipped "
                  "value was fitted on.\n      BONUS_OVERDISPERSION needs re-fitting — see "
                  "the `fitted` rows for where it now lands.")

    # The aggregate bias is a *sum* over buckets, so it can be ~0 while every bucket is
    # wrong in an ordered way. That is exactly what happens, which is why the bucket break
    # is printed rather than filed.
    buckets = cal[(cal["analysis"] == "calibration") & (cal["bucket"] != "all")
                  & (cal["fit_window"] == FULL_WINDOW)]
    order = [f"{BONUS_MPG_BUCKETS[i]:g}-{BONUS_MPG_BUCKETS[i + 1]:g}"
             for i in range(len(BONUS_MPG_BUCKETS) - 1)]
    for unit in ("player_season", "player_game"):
        sub = buckets[buckets["unit"] == unit]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="bucket", columns="overdispersion", values="bias")
        piv = piv.reindex([o for o in order if o in piv.index])
        label = "mpg" if unit == "player_season" else "minutes played"
        print(f"\n  {unit} — bias by {label} bucket, by overdispersion:")
        print(piv.round(4).to_string())
    ship_buckets = buckets[buckets["is_shipped"] & (buckets["unit"] == "player_season")]
    if len(ship_buckets):
        spread = float(ship_buckets["bias"].max() - ship_buckets["bias"].min())
        print(f"\n  Across mpg buckets the shipped value's bias spans {spread:.4f} "
              f"dk_pts/game even though the\n  aggregate is "
              f"{float(allrows[allrows['is_shipped'] & (allrows['unit'] == 'player_season')]['bias'].iloc[0]):+.4f}"
              " — the bucket errors cancel rather than being small.\n  One scalar frailty "
              "cannot absorb minutes variation whose RELATIVE size differs by bucket.")

    fitted = cal[(cal["analysis"] == "fitted") & (cal["fit_window"] == FULL_WINDOW)]
    if len(fitted):
        print("\n  zero-bias overdispersion by unit: " + ", ".join(
            f"{r.unit} {r.overdispersion:.3f}" for r in fitted.itertuples()))
        print("  The two units disagree by design, and the simulator draws per GAME: an "
              "overdispersion\n  fitted on season-mean counts is also standing in for the "
              "minutes variation that\n  per-game counts already carry. Do not reuse one "
              "for the other.")

    # ── the window the optimum is calibrated on ──────────────────────────────
    # The overdispersion is a simulator INPUT, so measuring it over the held-out seasons
    # would tune the bonus on the seasons the simulator is later scored against. Nothing
    # here is a fit in the train/test sense, so no split guard covers it.
    held = ", ".join(held_out_seasons(targets))
    opt = cal[cal["analysis"] == "fitted"].pivot_table(
        index="unit", columns="fit_window", values="overdispersion")
    if set(FIT_WINDOWS).issubset(opt.columns):
        opt = opt[FIT_WINDOWS]
        opt["delta"] = opt[TRAIN_VAL_WINDOW] - opt[FULL_WINDOW]
        print(f"\n  Fitted optimum by fit window ({held} held out of `{TRAIN_VAL_WINDOW}`) "
              f"— the simulator draws at\n  the player-GAME unit, so `{TRAIN_VAL_WINDOW}` "
              f"of that row is the one to ship:")
        print(opt.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nBonus calibration: {len(cal):,} rows → {cal_dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
