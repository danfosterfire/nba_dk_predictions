"""Residual cross-component correlation — the copula the simulator needs as an input.

`docs/predictions-plan.md` and `docs/simulations-plan.md` both settle where cross-component
correlation enters: **at draw time, not fit time.** Minutes are drawn once per player-game
and pushed through all eleven heads as exposure, which captures the single largest common
factor (18.6% of within-player residual variance). Whatever is left over is imposed with a
Gaussian copula on the residuals.

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
**+0.112** raw (+0.012 against +0.191 over the eight counts alone), and the largest raw cell
is `fg2a`–`fta` at **+0.492** — which is not a cross-component dependence at all, it is two
shot-volume counts both scaling with the minutes they were accumulated over. A simulator
built on the raw matrix would impose ~16x the intended coupling *on top of* the shared
minutes draw that already produced it. So `minutes_conditioned` is carried as an explicit
column rather than left implicit in a filename.

## What it says

Conditional on minutes the residual coupling is **small**: off-diagonals average +0.007
across the eleven heads (+0.012 across the eight counts alone) with a maximum of +0.142
(`fg2a`–`reb`). The 3PA/2PA substitution shows up exactly where the reparameterization
argument predicts, at **-0.125** — and it is handled by modelling `fga` and then `fg3a | fga`
as a share, not by a copula. So step 2 of the simulator's correlation plan is close to
sufficient, and this matrix is the measured licence for that claim rather than an assumption.

The frame is `serial_correlation.py`'s, and the residual helpers are imported from it rather
than reimplemented — the two artifacts then sit on identical rows, and "the residual against
the player-season's own rate with minutes as exposure" has one definition in the project.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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

# The pair the reparameterization exists to handle, checked by name so a rename cannot
# silently drop the check.
SUBSTITUTION_PAIR = ("fg3a", "fg2a")

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


def to_matrix(long: pd.DataFrame, basis: str | None = None) -> pd.DataFrame:
    """Pivot the long form back to a square matrix, in `COMPONENTS` order."""
    sub = long if basis is None else long[long["basis"] == basis]
    wide = sub.pivot(index="component_a", columns="component_b", values="r")
    names = [c for c in COMPONENTS if c in wide.index]
    return wide.loc[names, names]


def summarize(long: pd.DataFrame, basis: str) -> dict:
    """Off-diagonal mean, extremes, the substitution cell, and the copula's PSD check.

    `min_eigenvalue` is the one a consumer must not skip: a pairwise-complete matrix is not
    guaranteed positive semi-definite, and a Gaussian copula cannot be built from one that
    is not. Reported rather than corrected, so a future frame that breaks it says so.
    """
    R = to_matrix(long, basis)
    off = R.where(~np.eye(len(R), dtype=bool)).stack()
    counts = [c for c in COUNT_COMPONENTS if c in R.index]
    off_counts = R.loc[counts, counts].where(~np.eye(len(counts), dtype=bool)).stack()
    a, b = SUBSTITUTION_PAIR
    return {
        "basis": basis,
        "n_components": len(R),
        "n_pairs": int(len(off) // 2),
        "off_diagonal_mean": float(off.mean()),
        "off_diagonal_mean_counts_only": float(off_counts.mean()),
        "off_diagonal_max": float(off.max()),
        "off_diagonal_max_pair": "-".join(off.idxmax()),
        "off_diagonal_min": float(off.min()),
        "off_diagonal_min_pair": "-".join(off.idxmin()),
        "substitution_r": float(R.loc[a, b]) if a in R.index and b in R.columns else np.nan,
        "min_eigenvalue": float(np.linalg.eigvalsh(R.fillna(0.0).to_numpy()).min()),
        "n_games_min": int(long[long["basis"] == basis]["n_games"].min()),
        "n_games_max": int(long[long["basis"] == basis]["n_games"].max()),
    }


def measure(df: pd.DataFrame) -> pd.DataFrame:
    """Both bases, stacked."""
    return pd.concat([correlation_long(build(df), basis)
                      for basis, build in SERIES_BUILDERS.items()], ignore_index=True)


def run(cfg: dict) -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    df = player_season_frame(targets)
    print(f"Residual correlation: {len(df):,} player-games, "
          f"{df['ps'].nunique():,} player-seasons "
          f"(played, min >= {MIN_MINUTES:g}, >= {MIN_GAMES} games)")

    table = measure(df)
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "residual_correlation.csv"
    table.to_csv(dest, index=False)

    for basis in SERIES_BUILDERS:
        s = summarize(table, basis)
        print(f"\n── {basis} ──")
        print(to_matrix(table, basis).round(3).to_string())
        print(f"  off-diagonal mean {s['off_diagonal_mean']:+.4f} "
              f"({s['off_diagonal_mean_counts_only']:+.4f} over the eight counts alone), "
              f"max {s['off_diagonal_max']:+.4f} at {s['off_diagonal_max_pair']}, "
              f"min {s['off_diagonal_min']:+.4f} at {s['off_diagonal_min_pair']}")
        psd = "PSD" if s["min_eigenvalue"] > 0 else "NOT PSD — a copula cannot use it"
        print(f"  {SUBSTITUTION_PAIR[0]}-{SUBSTITUTION_PAIR[1]} substitution "
              f"{s['substitution_r']:+.4f}; min eigenvalue {s['min_eigenvalue']:+.4f} ({psd})")

    cond = summarize(table, MINUTES_CONDITIONED)
    raw = summarize(table, RAW)
    ratio = raw["off_diagonal_mean"] / cond["off_diagonal_mean"]
    print(f"\nThe raw matrix's off-diagonals average {ratio:.0f}x the conditioned ones "
          f"({raw['off_diagonal_mean']:+.3f} vs {cond['off_diagonal_mean']:+.3f}), and its "
          f"largest cell is\n  {raw['off_diagonal_max_pair']} at "
          f"{raw['off_diagonal_max']:+.3f} — the same free throws counted twice, not a "
          "dependence. Impose the\n  conditioned matrix on top of a shared minutes draw; "
          "imposing the raw one double-counts minutes.")
    print(f"\nResidual correlation: {len(table):,} pair rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
