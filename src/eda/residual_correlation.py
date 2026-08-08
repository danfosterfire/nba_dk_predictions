"""Residual cross-component correlation — the copula the simulator needs as an input.

`docs/predictions-plan.md` and `docs/simulations-plan.md` both settle where cross-component
correlation enters: **at draw time, not fit time.** Minutes are drawn once per player-game
and pushed through all eleven heads as exposure, which captures the single largest common
factor (**46.4%** of within-player residual variance — `make variance-budget`, the
`own_minutes` row). Whatever is left over is imposed with a Gaussian copula on the residuals.

That copula needs **the matrix**, not a summary of it, which is why this module exists as a
target rather than as three numbers in a docstring. Long form (`component_a`, `component_b`,
`r`, `n_games`, `basis`) specifically because the simulator reads it programmatically: a wide
matrix with component names as columns is a schema that breaks the moment a head is added.

## Two bases, and confusing them inflates the coupling 16x

    minutes_conditioned   Pearson residuals against each player-season's own rate, with
                          `min` as exposure for the counts and attempts as trials for the
                          conversions. This is the residual each head actually emits, and
                          the one the copula takes.
    raw                   The realized quantity itself, minutes NOT conditioned out —
                          counts as counts, a conversion head as `made / attempts`.

Measured over 592,796 player-games: the off-diagonal mean is **+0.007** conditioned against
**+0.090** raw (+0.023 against +0.226 over the seven counts alone), and the largest raw cell
is `fga`–`fta` at **+0.470** — which is not a cross-component dependence at all, it is two
shot-volume counts both scaling with the minutes they were accumulated over. A simulator
built on the raw matrix would impose ~13x the intended coupling *on top of* the shared
minutes draw that already produced it. So `minutes_conditioned` is carried as an explicit
column rather than left implicit in a filename.

## What it says

Conditional on minutes the residual coupling is **small**: off-diagonals average +0.007
across the eleven heads (+0.023 across the seven counts alone) with a maximum of +0.133
(`fga`–`reb`). The 3PA/2PA substitution is **gone from the matrix by construction** — the
shot-attempt basis models `fga` and then `fg3a | fga` as a share, so the identity is
enforced rather than carried. What the old two-count basis had to carry at **-0.125**
(`fg3a`-`fg2a`, kept as the `legacy_two_count_basis` contrast) is now **-0.084** between
`fga` and the mix, a genuine volume/mix relation rather than the substitution, and the
minimum eigenvalue improves from +0.756 to **+0.785**. So step 2 of the simulator's
correlation plan is close to sufficient, and this matrix is the measured licence for that
claim rather than an assumption.

The frame is `serial_correlation.py`'s, and the residual helpers are imported from it rather
than reimplemented — the two artifacts then sit on identical rows, and "the residual against
the player-season's own rate with minutes as exposure" has one definition in the project.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import (FIT_WINDOWS, FULL_WINDOW, TRAIN_VAL_WINDOW,
                                 fit_window, held_out_seasons)
from src.eda.serial_correlation import (
    CONVERSIONS,
    COUNT_COMPONENTS,
    MIN_GAMES,
    MIN_MINUTES,
    conversion_residuals,
    count_residuals,
    player_season_frame,
)

# The eleven non-minutes heads, in the order the generative chain reaches them. `min` is
# excluded by construction: it is the exposure every one of these is conditioned on, so its
# correlation with them is what the shared draw already carries.
COMPONENTS = COUNT_COMPONENTS + [f"{m}|{a}" for m, a in CONVERSIONS]

# The coupling the reparameterization replaces the old pair with. Under the shot-attempt
# basis `fg3a` is a SHARE of `fga`, so this cell is "does the three-point mix correlate
# with total shot volume" — a real residual dependence the copula still has to carry,
# unlike the substitution, which is now structural.
SUBSTITUTION_PAIR = ("fg3a|fga", "fga")

# The pair the old two-count basis had to carry in the copula, kept as an explicit
# contrast so the artifact still records what adopting the reparameterization bought.
# `fg2a` is DERIVED under the new basis (`fga - fg3a`) and is not among the modelled
# eleven, so this is emitted under its own `basis` label rather than into the matrix —
# `to_matrix` would otherwise have to be lenient about names, which is the silent filter
# this module deliberately no longer has.
LEGACY_SUBSTITUTION_PAIR = ("fg3a", "fg2a")
LEGACY_BASIS = "legacy_two_count_basis"

MINUTES_CONDITIONED = "minutes_conditioned"
RAW = "raw"


# ── Residual series, in each basis ────────────────────────────────────────────

def conditioned_series(df: pd.DataFrame) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """(name, kind, residual, valid) per head, minutes/attempts conditioned out.

    Reuses `serial_correlation`'s residual helpers verbatim. `valid` is theirs too, which
    matters: a night with no threes attempted is not evidence about three-point shooting,
    and a component whose expected count is below half is dominated by discreteness.
    """
    out = [(c, "count", *count_residuals(df, c)) for c in COUNT_COMPONENTS]
    out += [(f"{m}|{a}", "conversion", *conversion_residuals(df, m, a))
            for m, a in CONVERSIONS]
    return out


def raw_series(df: pd.DataFrame) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """The same heads in their realized units, with nothing conditioned out.

    Counts are the counts; a conversion head is its realized rate `made / attempts`, which
    is the honest unconditional analogue of a successes-out-of-trials residual. Carried only
    for the contrast in the module docstring — never as a copula input.
    """
    out = []
    for c in COUNT_COMPONENTS:
        out.append((c, "count", df[c].to_numpy(dtype=float),
                    np.ones(len(df), dtype=bool)))
    for made, att in CONVERSIONS:
        a = df[att].to_numpy(dtype=float)
        valid = a > 0
        rate = np.divide(df[made].to_numpy(dtype=float), a,
                         out=np.zeros(len(df)), where=valid)
        out.append((f"{made}|{att}", "conversion", rate, valid))
    return out


SERIES_BUILDERS = {MINUTES_CONDITIONED: conditioned_series, RAW: raw_series}


# ── The matrix ────────────────────────────────────────────────────────────────

def correlation_long(series: list[tuple[str, str, np.ndarray, np.ndarray]],
                     basis: str) -> pd.DataFrame:
    """Full symmetric pairwise matrix in long form, one row per ordered pair.

    Both `(a, b)` and `(b, a)` are emitted so a plain `pivot` returns the whole matrix
    rather than a triangle a consumer has to mirror. Pairs are **pairwise complete** — the
    rows where both components are valid — and `n_games` records that per cell, because the
    conversion heads are valid on a third of the rows the counts are.
    """
    kinds = {name: kind for name, kind, _, _ in series}
    rows = []
    for (na, _, ra, va), (nb, _, rb, vb) in itertools.combinations_with_replacement(series, 2):
        both = va & vb
        n = int(both.sum())
        if na == nb:
            r = 1.0
        elif n < 2 or ra[both].std() == 0 or rb[both].std() == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(ra[both], rb[both])[0, 1])
        rows.append({"component_a": na, "component_b": nb, "kind_a": kinds[na],
                     "kind_b": kinds[nb], "r": r, "n_games": n, "basis": basis,
                     "minutes_conditioned": basis == MINUTES_CONDITIONED})
        if na != nb:
            rows.append({"component_a": nb, "component_b": na, "kind_a": kinds[nb],
                         "kind_b": kinds[na], "r": r, "n_games": n, "basis": basis,
                         "minutes_conditioned": basis == MINUTES_CONDITIONED})
    return pd.DataFrame(rows)


def to_matrix(long: pd.DataFrame, basis: str | None = None,
              window: str = TRAIN_VAL_WINDOW) -> pd.DataFrame:
    """Pivot the long form back to a square matrix, in `COMPONENTS` order.

    **Strict about names in both directions.** This used to filter with
    `[c for c in COMPONENTS if c in wide.index]`, which silently dropped any component the
    artifact did not carry — so a head-list change would quietly shrink the copula instead
    of failing, and `substitution_r` would degrade to NaN with nothing to say why. A matrix
    that is missing a head is not a smaller copula, it is a wrong one.

    `window` **defaults to `train_val`, not to `full`**, because this function is the
    copula's programmatic entry point and the default is what an unthinking caller gets.
    A simulator handed the full-window matrix is calibrated on the seasons it is scored
    against; the leakage is invisible in the output, which is precisely the argument for
    making the safe window the default rather than the documented option. Pass
    `window=FULL_WINDOW` deliberately to reproduce the prose figures. A frame with no
    `fit_window` column at all is treated as one window, so older artifacts and the
    synthetic frames in the tests still pivot.
    """
    sub = long if basis is None else long[long["basis"] == basis]
    if "fit_window" in sub.columns:
        sub = sub[sub["fit_window"] == window]
        if sub.empty:
            raise ValueError(
                f"no rows for fit_window {window!r}; expected one of {FIT_WINDOWS}. "
                "Rebuild with `make residual-correlation`.")
    sub = sub[sub["component_a"].isin(COMPONENTS) | sub["component_b"].isin(COMPONENTS)]
    wide = sub.pivot(index="component_a", columns="component_b", values="r")
    missing = [c for c in COMPONENTS if c not in wide.index]
    unknown = [c for c in wide.index if c not in COMPONENTS]
    if missing or unknown:
        raise ValueError(
            f"residual correlation basis {basis!r} does not match COMPONENTS: "
            f"missing {missing}, unrecognised {unknown}. Rebuild with "
            f"`make residual-correlation` after any head-list change.")
    return wide.loc[COMPONENTS, COMPONENTS]


def legacy_substitution_long(df: pd.DataFrame) -> pd.DataFrame:
    """The old two-count basis's substitution cell, kept as a contrast row.

    Adopting the shot-attempt basis removes this coupling **by construction**, so the cell
    stops existing among the modelled eleven. Dropping it silently would make the copula's
    own artifact stop recording the reason the copula got smaller, so it is emitted under
    its own `basis` label — outside `COMPONENTS`, and therefore outside the matrix.
    """
    a, b = LEGACY_SUBSTITUTION_PAIR
    series = [(a, "count", *count_residuals(df, a)),
              (b, "count", *count_residuals(df, b))]
    long = correlation_long(series, LEGACY_BASIS)
    long["minutes_conditioned"] = True
    return long


def summarize(long: pd.DataFrame, basis: str,
              window: str = TRAIN_VAL_WINDOW) -> dict:
    """Off-diagonal mean, extremes, the substitution cell, and the copula's PSD check.

    `min_eigenvalue` is the one a consumer must not skip: a pairwise-complete matrix is not
    guaranteed positive semi-definite, and a Gaussian copula cannot be built from one that
    is not. Reported rather than corrected, so a future frame that breaks it says so —
    and it has to be checked **per window**, since dropping two seasons re-estimates every
    cell and PSD is a property of the whole matrix rather than of any one of them.
    """
    R = to_matrix(long, basis, window)
    off = R.where(~np.eye(len(R), dtype=bool)).stack()
    counts = [c for c in COUNT_COMPONENTS if c in R.index]
    off_counts = R.loc[counts, counts].where(~np.eye(len(counts), dtype=bool)).stack()
    a, b = SUBSTITUTION_PAIR
    return {
        "basis": basis,
        "fit_window": window,
        "n_components": len(R),
        "n_pairs": int(len(off) // 2),
        "off_diagonal_mean": float(off.mean()),
        "off_diagonal_mean_counts_only": float(off_counts.mean()),
        "off_diagonal_max": float(off.max()),
        "off_diagonal_max_pair": "-".join(off.idxmax()),
        "off_diagonal_min": float(off.min()),
        "off_diagonal_min_pair": "-".join(off.idxmin()),
        "substitution_r": float(R.loc[a, b]),
        # What the old two-count basis had to carry here, on the same rows. Under the
        # shot-attempt basis it is removed by construction, so the pair of numbers side by
        # side is the measurement of what adopting the reparameterization bought the copula.
        "legacy_substitution_r": _legacy_r(long, window),
        "min_eigenvalue": float(np.linalg.eigvalsh(R.fillna(0.0).to_numpy()).min()),
        "n_games_min": int(_scope(long, basis, window)["n_games"].min()),
        "n_games_max": int(_scope(long, basis, window)["n_games"].max()),
    }


def _scope(long: pd.DataFrame, basis: str, window: str) -> pd.DataFrame:
    sub = long[long["basis"] == basis]
    return sub[sub["fit_window"] == window] if "fit_window" in sub.columns else sub


def _legacy_r(long: pd.DataFrame, window: str = TRAIN_VAL_WINDOW) -> float:
    a, b = LEGACY_SUBSTITUTION_PAIR
    hit = _scope(long, LEGACY_BASIS, window)
    hit = hit[(hit["component_a"] == a) & (hit["component_b"] == b)]
    return float(hit["r"].iloc[0]) if len(hit) else float("nan")


def measure(df: pd.DataFrame) -> pd.DataFrame:
    """Both modelled bases, stacked, plus the legacy substitution contrast."""
    return pd.concat([correlation_long(build(df), basis)
                      for basis, build in SERIES_BUILDERS.items()]
                     + [legacy_substitution_long(df)], ignore_index=True)


def measure_windows(df: pd.DataFrame,
                    windows: list[str] = None) -> pd.DataFrame:
    """`measure` per fit window, tagged — the copula's calibration window is a choice.

    The matrix is a **simulator input**, not a summary the simulator reads about, so
    measuring it over the held-out seasons would tune the copula on the seasons the
    simulator is later backtested against. Nothing here is fitted, so no split guard
    would ever have caught that.
    """
    out = []
    for window in (FIT_WINDOWS if windows is None else windows):
        sub = measure(fit_window(df, window))
        sub.insert(0, "fit_window", window)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def run(cfg: dict) -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    df = player_season_frame(targets)
    print(f"Residual correlation: {len(df):,} player-games, "
          f"{df['ps'].nunique():,} player-seasons "
          f"(played, min >= {MIN_MINUTES:g}, >= {MIN_GAMES} games)")

    table = measure_windows(df)
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "residual_correlation.csv"
    table.to_csv(dest, index=False)

    for basis in SERIES_BUILDERS:
        s = summarize(table, basis, FULL_WINDOW)
        print(f"\n── {basis} ({FULL_WINDOW}) ──")
        print(to_matrix(table, basis, FULL_WINDOW).round(3).to_string())
        print(f"  off-diagonal mean {s['off_diagonal_mean']:+.4f} "
              f"({s['off_diagonal_mean_counts_only']:+.4f} over the "
              f"{len(COUNT_COMPONENTS)} counts alone), "
              f"max {s['off_diagonal_max']:+.4f} at {s['off_diagonal_max_pair']}, "
              f"min {s['off_diagonal_min']:+.4f} at {s['off_diagonal_min_pair']}")
        psd = "PSD" if s["min_eigenvalue"] > 0 else "NOT PSD — a copula cannot use it"
        print(f"  {SUBSTITUTION_PAIR[0]}-{SUBSTITUTION_PAIR[1]} "
              f"{s['substitution_r']:+.4f} (the legacy "
              f"{LEGACY_SUBSTITUTION_PAIR[0]}-{LEGACY_SUBSTITUTION_PAIR[1]} pair the "
              f"two-count basis had to carry: {s['legacy_substitution_r']:+.4f}, now "
              f"removed by construction);\n  min eigenvalue "
              f"{s['min_eigenvalue']:+.4f} ({psd})")

    cond = summarize(table, MINUTES_CONDITIONED, FULL_WINDOW)
    raw = summarize(table, RAW, FULL_WINDOW)
    ratio = raw["off_diagonal_mean"] / cond["off_diagonal_mean"]
    print(f"\nThe raw matrix's off-diagonals average {ratio:.0f}x the conditioned ones "
          f"({raw['off_diagonal_mean']:+.3f} vs {cond['off_diagonal_mean']:+.3f}), and its "
          f"largest cell is\n  {raw['off_diagonal_max_pair']} at "
          f"{raw['off_diagonal_max']:+.3f} — the same free throws counted twice, not a "
          "dependence. Impose the\n  conditioned matrix on top of a shared minutes draw; "
          "imposing the raw one double-counts minutes.")

    # ── the window the copula is actually calibrated on ──────────────────────
    held = ", ".join(held_out_seasons(df))
    print(f"\nThe copula is a simulator INPUT, so its calibration window matters and no "
          f"split guard\ncovers it — nothing here is fitted. `{TRAIN_VAL_WINDOW}` holds "
          f"out {held}, and it is the\nmatrix to consume ({to_matrix.__name__} defaults "
          f"to it):")
    keys = ["off_diagonal_mean", "off_diagonal_mean_counts_only", "off_diagonal_max",
            "off_diagonal_min", "substitution_r", "min_eigenvalue"]
    both = pd.DataFrame([summarize(table, MINUTES_CONDITIONED, w) for w in FIT_WINDOWS])
    contrast = both.set_index("fit_window")[keys].T
    contrast["delta"] = contrast[TRAIN_VAL_WINDOW] - contrast[FULL_WINDOW]
    print(contrast.to_string(float_format=lambda v: f"{v:+.4f}"))
    sim = both[both["fit_window"] == TRAIN_VAL_WINDOW].iloc[0]
    if sim["min_eigenvalue"] <= 0:
        raise ValueError(
            f"the {TRAIN_VAL_WINDOW} conditioned matrix is NOT positive semi-definite "
            f"(min eigenvalue {sim['min_eigenvalue']:+.4f}) — a Gaussian copula cannot "
            "be built from it. The full-window matrix being PSD is not a substitute: the "
            "simulator consumes this one.")
    print(f"  {TRAIN_VAL_WINDOW} min eigenvalue {sim['min_eigenvalue']:+.4f} — still PSD, "
          f"so the copula is usable on the window it should be calibrated on.")
    print(f"\nResidual correlation: {len(table):,} pair rows "
          f"({len(FIT_WINDOWS)} fit windows) → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
