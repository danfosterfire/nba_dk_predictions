"""Is there a hot streak? Serial dependence in the twelve per-game components.

The fitting strategy in `docs/predictions-plan.md` collapses the game-level likelihood
to player-season totals. That collapse is exact for the *mean structure* — for a
log-link Poisson with minutes exposure, and for a binomial with constant success
probability, the game-level likelihood factors into a season-total term times a
multinomial (resp. multivariate-hypergeometric) term free of the coefficients. What it
discards is precisely that second factor: the within-season allocation of the season
total across games. Serial dependence lives only there.

So the question this module answers is not "does a lag term improve prediction" — under
the prediction-time constraint no per-game lag is *available* at prediction time. It is
"**would a simulator that draws games independently misstate the spread of a simulated
season**", which is what the draft-strategy layer consumes.

⚠️ **What this module writes is a DIAGNOSTIC, not a simulator input**, and saying so here is
the point: nothing in `src/sim/` reads `serial_correlation.csv`. The block inflation is a
target Gate A scores the drawn tensor against, and the serial structure the draw actually has
is *produced* by the season-constant per-(player, season) σ shocks
(`sim.minutes.player_season_sigma_by_role`). Earlier prose in this repo described the 2.43×
as "consumed"; `docs/sim-inputs-plan.md` carries the consumed-vs-diagnostic inventory.

Two statistics, because they answer different halves of that:

**Lag-k autocorrelation of Pearson residuals.** Residuals are taken against each
player-season's own rate with `min` as exposure, so minutes are conditioned out of the
count rows and what remains is variation in *rate*, not in playing time.

**Block-sum variance inflation.** `Var(sum over a 10-game block) / (10 x per-game
variance)`, which is 1 under independence and larger when games cluster. This is the
decision-relevant number: it is the factor by which an independent-draws simulator
understates the variance of an aggregate.

Both ship with a **shuffled null** — game order permuted *within* player-season, which
preserves every marginal, every player-season mean, and the downward bias that
within-season demeaning induces in both statistics. Per `CLAUDE.md`, a score without its
own chance level is not a finding; here the null is doing real work, since the raw lag-1
figure for `stl` is *negative* and still represents positive dependence once the -0.017
demeaning bias is accounted for.

The measured answer, and it splits exactly along the attempts/conversion line that
`persistence.csv` already found at the season level: **there is no shooting hot hand**
(`fg3m|fg3a` excess -0.001, `fg2m|fg2a` +0.001, both dead nulls on ~600k pairs), while
the *exposure* side is strongly dependent (`min` 2.43x block inflation, shot volume
~1.45x on top of minutes). Conversion heads collapse for free; minutes is where a
sequential model has to go.

## Two fit windows, because the block inflation is a simulator INPUT

Nothing here is fitted, so nothing here is scored on a held-out split — which is exactly
why this artifact could quietly be measured over every season including the two the heads
hold out. The block inflation is not a finding the simulator reads *about*, it is a number
the simulator will be *given*, so calibrating it on 2024-25/2025-26 would tune the
simulator on the seasons it is later backtested against. Every row is emitted **three
times**, under `fit_window` in {`full`, `train_val`, `train`}, and **which one to consume is
decided by what the number will be scored against, not by which is widest**: `train` for the
realized 2022-23 / 2023-24 backtest, since `train_val` contains those seasons; `train_val`
for the one-shot test readout; `full` for production. `full` also stays because the
difference is worth having on disk rather than assumed, and because the prose in `CLAUDE.md`
and `docs/` quotes the full-window figures.

The third window was added on 2026-08-08, when `make posteriors` started emitting
coefficients per window and it became possible for a backtest to hold clean coefficients
alongside a noise shape calibrated on the very seasons it was scoring.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import (FIT_WINDOWS, FULL_WINDOW, TRAIN_VAL_WINDOW,
                                 TRAIN_WINDOW, fit_window, held_out_seasons)

# Counts modelled with `min` as exposure. Excludes the makes, which are conditioned on
# their own attempts and handled as conversions below.
# Mirrors `component_rates.COUNT_HEADS` / `CONVERSION_HEADS` — the shot-attempt basis,
# so these are the heads that are actually fitted. `fg2a` is derived (`fga - fg3a`) and
# is not a component here; it survives only as the legacy contrast in
# `residual_correlation.LEGACY_SUBSTITUTION_PAIR`.
COUNT_COMPONENTS = ["fga", "fta", "reb", "ast", "stl", "blk", "tov"]

# (made, attempted) — the successes/trials heads, in chain order.
CONVERSIONS = [("fg3a", "fga"), ("fg2m", "fg2a"), ("fg3m", "fg3a"), ("ftm", "fta")]

LAGS = [1, 2, 3, 5, 10, 20]
BLOCK_GAMES = 10
NULL_REPLICATES = 8

# A Pearson residual is only approximately standard normal once the expected count has
# some mass; below this the discreteness dominates and the correlation is attenuated.
MIN_EXPECTED = 0.5

# Garbage-time games carry real measurement error in the rate (CLAUDE.md), and the
# minutes floor is the cheap version of the minutes weighting used elsewhere.
MIN_MINUTES = 5.0
MIN_GAMES = 40


def player_season_frame(targets: pd.DataFrame, min_minutes: float = MIN_MINUTES,
                        min_games: int = MIN_GAMES) -> pd.DataFrame:
    """Played games in chronological order, restricted to substantial player-seasons.

    Sorted by (player_id, season, game_date) and stamped with a dense `ps` index and a
    within-season game counter, both of which the lag helpers key on so that a lag never
    reaches across a player-season boundary.
    """
    df = targets[(targets["played"] == 1) & (targets["min"] >= min_minutes)].copy()
    df = df.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    df["n_games"] = df.groupby(["player_id", "season"])["min"].transform("size")
    df = df[df["n_games"] >= min_games].reset_index(drop=True)
    key = df["player_id"].astype(str) + "_" + df["season"].astype(str)
    df["ps"] = pd.factorize(key)[0]
    df["game_index"] = df.groupby("ps").cumcount()
    return df


def count_residuals(df: pd.DataFrame, component: str) -> tuple[np.ndarray, np.ndarray]:
    """Pearson residuals against the player-season rate, with `min` as exposure.

    The expectation is `min_g * (sum_g y_g / sum_g min_g)`, i.e. the player-season's own
    per-minute rate. This is the residual the collapsed likelihood cannot see.
    """
    y = df[component].to_numpy(dtype=float)
    tot_y = df.groupby("ps")[component].transform("sum").to_numpy(dtype=float)
    tot_m = df.groupby("ps")["min"].transform("sum").to_numpy(dtype=float)
    expected = df["min"].to_numpy(dtype=float) * np.divide(
        tot_y, tot_m, out=np.zeros_like(tot_y), where=tot_m > 0)
    valid = expected > MIN_EXPECTED
    resid = np.zeros(len(df))
    np.divide(y - expected, np.sqrt(expected), out=resid, where=valid)
    return resid, valid


def conversion_residuals(df: pd.DataFrame, made: str,
                         attempted: str) -> tuple[np.ndarray, np.ndarray]:
    """Pearson residuals for a successes/trials head at the player-season rate.

    Only games with at least one attempt carry information, so `valid` excludes the
    rest — and the lag helpers then pair *consecutive valid games*, which is the right
    comparison: a night with no threes attempted is not evidence about shooting form.
    """
    att = df[attempted].to_numpy(dtype=float)
    mk = df[made].to_numpy(dtype=float)
    tot_a = df.groupby("ps")[attempted].transform("sum").to_numpy(dtype=float)
    tot_m = df.groupby("ps")[made].transform("sum").to_numpy(dtype=float)
    p = np.divide(tot_m, tot_a, out=np.full(len(df), np.nan), where=tot_a > 0)
    var = att * p * (1 - p)
    valid = (att > 0) & (var > MIN_EXPECTED) & np.isfinite(p)
    resid = np.zeros(len(df))
    np.divide(mk - att * p, np.sqrt(np.where(valid, var, 1.0)), out=resid, where=valid)
    return resid, valid


def minutes_residuals(df: pd.DataFrame,
                      detrend: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Minutes about the player-season mean, optionally with a linear trend removed.

    The detrended variant separates two mechanisms that both produce positive
    autocorrelation: slow role change over a season (a trend) and short-range shocks
    (rotation churn, injury ramps, blowout clusters). Reporting both is what licenses
    the claim about which one dominates.
    """
    mn = df["min"].to_numpy(dtype=float)
    centred = mn - df.groupby("ps")["min"].transform("mean").to_numpy(dtype=float)
    valid = np.ones(len(df), dtype=bool)
    if not detrend:
        return centred, valid
    idx = df["game_index"].to_numpy(dtype=float)
    di = idx - df.groupby("ps")["game_index"].transform("mean").to_numpy(dtype=float)
    num = df.assign(_a=di * centred).groupby("ps")["_a"].transform("sum").to_numpy()
    den = df.assign(_b=di * di).groupby("ps")["_b"].transform("sum").to_numpy()
    slope = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return centred - slope * di, valid


def lag_autocorr(resid: np.ndarray, ps: np.ndarray, valid: np.ndarray,
                 lag: int = 1) -> tuple[float, int]:
    """Pooled correlation between a residual and its k-th predecessor.

    Pairs must lie in the same player-season and both be valid. Note the pairing is on
    *position among valid rows*, not on raw row offset, so for the conversion heads a
    lag of 1 means "the previous game in which he attempted one", which is the
    meaningful notion of consecutive for those.
    """
    keep = valid
    r, p = resid[keep], ps[keep]
    if len(r) <= lag:
        return float("nan"), 0
    same = p[lag:] == p[:-lag]
    a, b = r[lag:][same], r[:-lag][same]
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan"), int(same.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(same.sum())


def block_inflation(resid: np.ndarray, ps: np.ndarray, valid: np.ndarray,
                    block: int = BLOCK_GAMES) -> float:
    """Var(block sum) / (block x per-game var), median over player-seasons.

    1.0 under independence. Only complete blocks count, and the per-game variance is the
    player-season's own — so this is a pure statement about clustering, with the level
    and the marginal spread divided out.
    """
    d = pd.DataFrame({"ps": ps[valid], "r": resid[valid]})
    if d.empty:
        return float("nan")
    d["block"] = d.groupby("ps").cumcount() // block
    d = d[d.groupby(["ps", "block"])["r"].transform("size") == block]
    if d.empty:
        return float("nan")
    block_sums = d.groupby(["ps", "block"])["r"].sum()
    within = d.groupby("ps")["r"].var(ddof=1)
    between = block_sums.groupby("ps").var(ddof=1)
    ok = between.notna() & within.notna() & (within > 0)
    if not ok.any():
        return float("nan")
    return float((between[ok] / (block * within[ok])).median())


def shuffled_null(resid: np.ndarray, ps: np.ndarray, valid: np.ndarray,
                  stat, replicates: int = NULL_REPLICATES,
                  seed: int = 0) -> tuple[float, float]:
    """Mean and sd of a statistic under game order permuted within player-season.

    Permuting *within* the player-season is the whole point: it destroys order while
    holding the player-season composition, its mean, and its marginal spread fixed, so
    the difference is attributable to sequence and nothing else.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(replicates):
        order = np.lexsort((rng.random(len(ps)), ps))
        vals.append(stat(resid[order], ps[order], valid[order]))
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _series(df: pd.DataFrame) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """(name, kind, residual, valid) for every component plus the two minutes variants."""
    out = [("min", "minutes", *minutes_residuals(df)),
           ("min_detrended", "minutes", *minutes_residuals(df, detrend=True))]
    out += [(c, "count", *count_residuals(df, c)) for c in COUNT_COMPONENTS]
    out += [(f"{m}|{a}", "conversion", *conversion_residuals(df, m, a))
            for m, a in CONVERSIONS]
    return out


def measure(df: pd.DataFrame, lags: list[int] = LAGS,
            block: int = BLOCK_GAMES) -> pd.DataFrame:
    """One row per component: lag-k autocorrelations, nulls, and block inflation."""
    ps = df["ps"].to_numpy()
    rows = []
    for name, kind, resid, valid in _series(df):
        obs, n_pairs = lag_autocorr(resid, ps, valid, 1)
        null_mean, null_sd = shuffled_null(
            resid, ps, valid, lambda r, p, v: lag_autocorr(r, p, v, 1)[0])
        blk = block_inflation(resid, ps, valid, block)
        blk_null, _ = shuffled_null(
            resid, ps, valid, lambda r, p, v: block_inflation(r, p, v, block))
        row = {
            "component": name, "kind": kind, "n_pairs": n_pairs,
            "lag1": obs, "lag1_null": null_mean, "lag1_null_sd": null_sd,
            "lag1_excess": obs - null_mean,
            "lag1_z": (obs - null_mean) / null_sd if null_sd > 0 else float("nan"),
            "block_observed": blk, "block_shuffled": blk_null,
            "block_inflation": blk / blk_null if blk_null else float("nan"),
        }
        for k in lags:
            row[f"lag{k}"] = lag_autocorr(resid, ps, valid, k)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg: dict) -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    df = player_season_frame(targets)
    print(f"Serial correlation: {len(df):,} player-games, "
          f"{df['ps'].nunique():,} player-seasons "
          f"(played, min >= {MIN_MINUTES:g}, >= {MIN_GAMES} games)")

    windows = []
    for window in FIT_WINDOWS:
        sub = fit_window(df, window)
        measured = measure(sub)
        measured.insert(0, "fit_window", window)
        windows.append(measured)
        print(f"  {window:9s} {len(sub):,} player-games, "
              f"{sub['ps'].nunique():,} player-seasons")
    table = pd.concat(windows, ignore_index=True)
    full = table[table["fit_window"] == FULL_WINDOW]

    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "serial_correlation.csv"
    table.to_csv(dest, index=False)

    show = ["component", "kind", "lag1", "lag1_null", "lag1_excess", "lag1_z",
            "block_inflation"]
    ordered = full.sort_values("block_inflation", ascending=False)
    print(ordered[show].to_string(index=False,
                                 float_format=lambda v: f"{v:.4f}"))

    # "Conversion head" is no longer a synonym for "shooting head". Under the
    # shot-attempt basis `fg3a | fga` is a beta-binomial like the makes but measures shot
    # MIX, not accuracy — and mix drifts within a season while accuracy does not. Lumping
    # them into one max would report the share's dependence as if it were a hot hand and
    # hide that the shooting heads are still clean nulls.
    shooting = full[(full["kind"] == "conversion")
                    & full["component"].str.startswith(("fg2m", "fg3m", "ftm"))]
    print(f"\n  shooting conversion heads — max |excess| "
          f"{shooting['lag1_excess'].abs().max():.4f}, max block inflation "
          f"{shooting['block_inflation'].max():.3f}: no shooting hot hand, so the "
          f"makes|attempts heads collapse for free.")
    mix = full[full["component"] == "fg3a|fga"]
    if len(mix):
        m = mix.iloc[0]
        print(f"  the shot-MIX share fg3a|fga is NOT one of them — excess "
              f"{m['lag1_excess']:+.4f} (z {m['lag1_z']:.0f}), block inflation "
              f"{m['block_inflation']:.3f}. Shot selection drifts within a season the way "
              f"minutes\n  do; shooting accuracy does not. It is a conversion head by "
              f"likelihood, not by subject matter.")
    mins = full[full["component"] == "min"].iloc[0]
    det = full[full["component"] == "min_detrended"].iloc[0]
    print(f"  minutes — lag1 {mins['lag1']:.3f} raw against {det['lag1']:.3f} "
          f"detrended, block inflation {mins['block_inflation']:.2f}x: the sequential "
          f"model belongs here, not on twelve heads.")

    # The block inflation is a simulator input, so the window it was calibrated on is a
    # correctness property and not a footnote. Held out: whatever `fit_window` drops.
    pivot = table.pivot(index="component", columns="fit_window",
                        values="block_inflation")[FIT_WINDOWS]
    pivot["widest_minus_narrowest"] = pivot[FULL_WINDOW] - pivot[TRAIN_WINDOW]
    print("\nBlock variance inflation by fit window — a number the simulator is GIVEN must "
          "not be\ncalibrated on the seasons it is later scored against, so WHICH window "
          "to consume is decided\nby what it will be scored against, not by which is "
          "widest:")
    for window in FIT_WINDOWS:
        held = ", ".join(held_out_seasons(df, window=window)) or "nothing"
        print(f"  {window:9s} holds out {held}")
    print(pivot.sort_values(FULL_WINDOW, ascending=False)
          .to_string(float_format=lambda v: f"{v:.4f}"))
    worst = pivot["widest_minus_narrowest"].abs().idxmax()
    print(f"  largest move across windows: {worst} "
          f"{pivot.loc[worst, FULL_WINDOW]:.4f} -> {pivot.loc[worst, TRAIN_WINDOW]:.4f} "
          f"({-pivot.loc[worst, 'widest_minus_narrowest']:+.4f})")
    print(f"Serial correlation: {len(table):,} component rows "
          f"({len(FIT_WINDOWS)} fit windows) → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
