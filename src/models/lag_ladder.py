"""The lag-recovery ladder's gate — `docs/rookie-rates-plan.md` §5b, §4.

`component_rates.build_design` widens from *is the immediately-prior season big enough* to
*is any prior season constructible*, by imputing an unusable lag-1 block from the nearest
usable one. §7b chose the **shape** of that ladder in carry-forward R²; this module runs
the **gate** that decides which of its four rungs ship.

## What is and is not fitted here

**Nothing.** The eleven shipped heads are read off `data/features/posteriors/train/` and
score the recovered rows with the coefficients they already have. That is not a shortcut,
it is the claim under test: §3 constraint 4 says the ladder widens the veteran heads'
*scoring* population and never their fitting population, and the only way to demonstrate
that is to score with a posterior fitted before the ladder existed. The `train` window is
the right one for the same reason every selection reading uses it — validation is
2022-23/2023-24 and those seasons are not in it.

## The two gates, stated in §4 before any result

1. **Against the unserved status quo**, validation paired-bootstrap CRPS interval entirely
   below zero. The status quo for these rows is *no prediction at all*: a unit missing from
   the design is missing from the tensor, so it contributes a literal zero to every draw,
   and its predictive is a point mass at 0 whose CRPS is `|y - 0| = y`. That is a low bar
   on purpose — which is precisely why it is only half of the gate.
2. **Inside the shipped no-fit floor's published band** (0.81-0.95 on the count heads), so
   a rung cannot be admitted on the strength of beating nothing. This is `carry_forward`
   evaluated on the ladder's own imputed column, which is `recent_shrunk_rate` by
   construction — the estimator §7b measured, read here on the rows that actually ship.

## The unit, and where it departs from §4's wording

§4 says "at the player-game unit". These heads are **season-collapsed** — the collapse is
the argument `docs/predictions-plan.md` makes for fitting 10,194 player-seasons instead of
731,906 player-games — so a player-game predictive does not exist without inventing a
per-game dispersion nothing has fitted. The reading is therefore at the head's own scoring
unit, the season total, which is the unit `stan_component_metrics.csv` and
`components_preseason.csv` already report and the only one a figure here is comparable to.
The bootstrap still resamples rows, which is what the pairing was for.

Usage:
    python -m src.models.lag_ladder
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.component_rates import (CONVERSION_HEADS, COUNT_HEADS, LADDER_RUNGS,
                                        TEST_SEASONS, carry_forward, lag_ladder,
                                        rate_columns)
from src.models.held_out import selection_split
from src.models.minutes_unification import paired_bootstrap, verdict
from src.models.posteriors import load_all, posteriors_dir
from src.models.stan_components import _beta_shapes
from src.models.stan_utils import crps_from_samples
from src.sim.season import artifact_name

SEED = 42
BOOTSTRAP_REPS = 2000

# The window whose posteriors do the scoring. `train` ends at 2021-22, so the validation
# seasons the gate reads are outside it — the same discipline every selection reading in
# the project follows, and the reason the gate can refit nothing.
SCORING_WINDOW = "train"

# The shipped no-fit floor's published band on the count heads (`component_rates`'s module
# docstring, and README §3). Gate 2 is membership of this interval, not a one-sided test:
# a rung that scored ABOVE it would mean the imputed rows are easier than the population
# the band was measured on, which is a reason to look rather than to ship.
FLOOR_BAND = (0.81, 0.95)

# `veteran` is rung 0 — not a gated rung, but scored beside them so every figure has the
# shipped bar next to it rather than a remembered one.
GROUPS = ("veteran",) + LADDER_RUNGS


def head_list() -> list[tuple[str, str, str | None]]:
    """`(label, made/component, attempted)` for all eleven heads, counts first."""
    return ([(c, c, None) for c in COUNT_HEADS]
            + [(f"{m}|{a}", m, a) for m, a in CONVERSION_HEADS])


# ── The predictive, drawn from coefficients nothing here fitted ───────────────

def count_samples(artifact, frame: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """`(draws x rows)` season totals from the persisted negative-binomial head.

    One negative-binomial draw per posterior draw rather than a plug-in predictive at the
    posterior mean: the artifact carries 1,000 thinned draws and the whole point of
    persisting them is that a consumer does not have to collapse them.
    """
    mu = artifact.mu_draws(frame)
    exposure = frame[artifact.extras["exposure"]].to_numpy(dtype=float)
    mean = np.clip(mu * exposure[None, :], 1e-9, None)
    phi = np.asarray(artifact.draws["phi_draws"], dtype=float)[:, None]
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(np.broadcast_to(phi, mean.shape),
                                 phi / (phi + mean)).astype(float)


def conversion_samples(artifact, frame: pd.DataFrame, attempted: str,
                       seed: int = SEED) -> np.ndarray:
    """`(draws x rows)` makes from the persisted beta-binomial head, on realized trials."""
    p = artifact.mu_draws(frame)
    rho = np.asarray(artifact.draws["rho_draws"], dtype=float)[:, None]
    n = np.rint(frame[attempted].to_numpy(dtype=float)).astype(int)
    a, b = _beta_shapes(p, np.broadcast_to(rho, p.shape))
    rng = np.random.default_rng(seed)
    return rng.binomial(n[None, :], rng.beta(a, b)).astype(float)


def head_scores(artifact, frame: pd.DataFrame, component: str,
                attempted: str | None, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """`(per-row CRPS of the fitted head, per-row CRPS of the unserved status quo)`.

    The status quo is a **point mass at zero**, so its CRPS is the realized value itself.
    Written out rather than special-cased inside the bootstrap, because it is the honest
    statement of what happens to one of these players today: he is not in the tensor, so
    every draw scores him at zero.
    """
    if attempted is None:
        y = frame[component].to_numpy(dtype=float)
        samples = count_samples(artifact, frame, seed)
    else:
        n = np.rint(frame[attempted].to_numpy(dtype=float))
        y = np.minimum(np.rint(frame[component].to_numpy(dtype=float)), n)
        samples = conversion_samples(artifact, frame, attempted, seed)
    return crps_from_samples(samples, y), np.abs(y)


def live_rows(frame: pd.DataFrame, attempted: str | None) -> pd.DataFrame:
    """Rows a head can score — a conversion head needs at least one attempt."""
    if attempted is None:
        return frame
    return frame[frame[attempted].to_numpy(dtype=float) > 0]


# ── The two gates ─────────────────────────────────────────────────────────────

def _row(gate: str, group: str, head: str, metric: str, value: float, n: int) -> dict:
    return {"gate": gate, "group": group, "head": head, "metric": metric,
            "value": float(value), "n": int(n)}


def population(plain: pd.DataFrame, wide: pd.DataFrame,
               val: pd.DataFrame) -> list[dict]:
    """How many rows each rung is worth, in the design and on the validation split.

    Counted on the DESIGN rather than on played player-seasons, which is what
    `lag_recovery.csv`'s census counts: the two differ by the age and target-minutes
    conditions `build_design` applies on top, so reporting them apart keeps a reader from
    reading one table's `n` into the other's.
    """
    rows = [_row("population", "all", "", "n_design_today", len(plain), len(plain)),
            _row("population", "all", "", "n_design_ladder", len(wide), len(wide)),
            _row("population", "all", "", "n_validation", len(val), len(val))]
    for group in GROUPS:
        rows += [_row("population", group, "", "n_design",
                      int((wide["lag_rung"] == group).sum()), len(wide)),
                 _row("population", group, "", "n_validation",
                      int((val["lag_rung"] == group).sum()), len(val))]
    return rows


def crps_gate(artifacts: dict, val: pd.DataFrame, seed: int = SEED) -> list[dict]:
    """Gate 1, per head per rung: the fitted head against scoring the row at zero."""
    rows = []
    for label, component, attempted in head_list():
        art = artifacts[artifact_name(label)]
        for group in GROUPS:
            frame = live_rows(val[val["lag_rung"] == group], attempted)
            if len(frame) < 2:
                rows.append(_row("crps", group, label, "n_live", len(frame), len(frame)))
                continue
            fitted, unserved = head_scores(art, frame, component, attempted, seed)
            delta = paired_bootstrap(fitted, unserved, n_boot=BOOTSTRAP_REPS, seed=seed)
            rows += [
                _row("crps", group, label, "val_crps", fitted.mean(), len(frame)),
                _row("crps", group, label, "unserved_crps", unserved.mean(), len(frame)),
                _row("crps", group, label, "crps_delta", delta["crps_delta"], len(frame)),
                _row("crps", group, label, "ci_lo", delta["ci_lo"], len(frame)),
                _row("crps", group, label, "ci_hi", delta["ci_hi"], len(frame)),
                _row("crps", group, label, "n_live", len(frame), len(frame)),
                _row("crps", group, label, "wins", float(verdict(delta) == "wins"),
                     len(frame)),
            ]
    return rows


def band_gate(val: pd.DataFrame) -> list[dict]:
    """Gate 2 per rung: `carry_forward` on the ladder's own column, against the band.

    `carry_forward` reads `{component}_p36_lag1`, which for a recovered row IS the shrunk
    nearest-usable rate the ladder wrote there — so this is `recent_shrunk_rate` scored on
    the rows that would actually ship, rather than a second implementation of it.

    ## The gate is the MEAN over the count heads, and the per-head reading is diagnosis

    Read per head, the band rejects **the shipped design itself**: on the same 773
    validation veteran rows only 5 of the 7 count heads land inside [0.81, 0.95], because
    `fga` (0.951) and `reb` (0.950) score *above* it. A test the incumbent fails is not a
    test of anything, and the reason is that the published band is a summary of a pooled
    per-head spread rather than a tolerance every head sits in on every subpopulation.
    So the gate is the rung's mean R2 over the count heads — §4's own domain, "0.81-0.95
    **on the count heads**" — with the veteran rung carried alongside as the bar it is
    read against, and the per-head cells kept in the artifact because a rung that fails
    on one head is a different finding from one that fails on all of them.

    `rate_mean_r2` is the same average over §7b's **nine rate targets** rather than the
    seven count heads. It is reported and not gated: it is the figure §7b's table quotes,
    so a reader can put a rung's gate reading next to the measurement that shaped it, and
    it runs higher because `fga`/`fg2a`/`fg3a` are the attempt columns the project has
    always predicted best.
    """
    rows = []
    for group in GROUPS:
        frame = val[val["lag_rung"] == group]
        if len(frame) < 2:
            continue
        per_head = {}
        for component in rate_columns():
            y = frame[component].to_numpy(dtype=float)
            pred = carry_forward(frame, component)
            ok = np.isfinite(y) & np.isfinite(pred)
            denom = ((y[ok] - y[ok].mean()) ** 2).sum()
            r2 = (float("nan") if denom <= 0
                  else float(1.0 - ((y[ok] - pred[ok]) ** 2).sum() / denom))
            per_head[component] = r2
            rows += [
                _row("band", group, component, "floor_r2", r2, int(ok.sum())),
                _row("band", group, component, "in_band",
                     float(FLOOR_BAND[0] <= r2 <= FLOOR_BAND[1]), int(ok.sum())),
            ]
        counts = float(np.mean([per_head[c] for c in COUNT_HEADS]))
        rows += [
            _row("band", group, "", "count_mean_r2", counts, len(frame)),
            _row("band", group, "", "rate_mean_r2",
                 float(np.mean([per_head[c] for c in rate_columns()])), len(frame)),
            _row("band", group, "", "mean_in_band",
                 float(FLOOR_BAND[0] <= counts <= FLOOR_BAND[1]), len(frame)),
        ]
    return rows


def verdicts(table: pd.DataFrame) -> pd.DataFrame:
    """One admitted/rejected row per rung — the conjunction §4 stated in advance."""
    rows = []
    for group in LADDER_RUNGS:
        crps = table[(table["gate"] == "crps") & (table["group"] == group)
                     & (table["metric"] == "wins")]
        band = table[(table["gate"] == "band") & (table["group"] == group)
                     & (table["metric"] == "mean_in_band")]
        cells = table[(table["gate"] == "band") & (table["group"] == group)
                      & (table["metric"] == "in_band")
                      & (table["head"].isin(COUNT_HEADS))]
        n = int(table[(table["group"] == group)
                      & (table["metric"] == "n_live")]["n"].max() or 0)
        won = int(crps["value"].sum())
        in_band = bool(len(band) and band["value"].iloc[0] == 1.0)
        passes = bool(len(crps) and won == len(crps) and in_band)
        rows += [_row("verdict", group, "", "heads_winning_crps", won, n),
                 _row("verdict", group, "", "heads_scored_crps", len(crps), n),
                 _row("verdict", group, "", "count_mean_in_band", float(in_band), n),
                 _row("verdict", group, "", "count_heads_in_band",
                      float(cells["value"].sum()), n),
                 _row("verdict", group, "", "count_heads_scored", len(cells), n),
                 _row("verdict", group, "", "admitted", float(passes), n)]
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> Path:
    from src.models.stan_components import PRESEASON, head_design

    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    from src.models.component_rates import build_design
    # EVERY rung, whatever the config says: this is the measurement the config key is
    # written from, so it cannot be allowed to read it.
    ladder = lag_ladder(cfg, rungs=LADDER_RUNGS)
    plain = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    wide = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"],
                        ladder=ladder)

    preseason = bool(cfg.get("stan", {}).get("components", {}).get("preseason", PRESEASON))
    design = head_design(cfg, preseason, design=wide)
    train, val = selection_split(design, TEST_SEASONS)

    print("Lag-recovery ladder — the §4 gate")
    print(f"  The test split is LOCKED — seasons go through `held_out.selection_split`.")
    print(f"  scoring posteriors: data/features/posteriors/{SCORING_WINDOW}/ — NOTHING is "
          f"fitted here")
    print(f"  {len(plain):,} design rows today -> {len(wide):,} with every rung "
          f"(+{len(wide) - len(plain):,})")
    print(f"  {len(val):,} validation rows "
          f"({', '.join(sorted(val['season'].unique()))})")
    for group in GROUPS:
        print(f"    {group:<16} {int((val['lag_rung'] == group).sum()):>6,} validation "
              f"rows, {int((wide['lag_rung'] == group).sum()):>6,} in the design")

    artifacts = load_all(posteriors_dir(cfg, SCORING_WINDOW),
                         heads=[artifact_name(h) for h, _, _ in head_list()])
    rows = population(plain, wide, val) + crps_gate(artifacts, val) + band_gate(val)
    table = pd.DataFrame(rows)
    table = pd.concat([table, verdicts(table)], ignore_index=True)

    _report(table)
    dest = out_dir / "lag_ladder.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} ladder gate rows → {dest}")
    return dest


def _report(t: pd.DataFrame) -> None:
    def val_of(gate: str, group: str, head: str, metric: str) -> float:
        hit = t[(t["gate"] == gate) & (t["group"] == group) & (t["head"] == head)
                & (t["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    def mean_of(gate: str, group: str, metric: str) -> float:
        hit = t[(t["gate"] == gate) & (t["group"] == group) & (t["metric"] == metric)]
        return float(hit["value"].mean()) if len(hit) else float("nan")

    print("\nGate 1 — the fitted head against scoring the row at zero (validation CRPS)")
    print(f"  {'rung':<16}{'n':>6}{'head CRPS':>12}{'unserved':>11}"
          f"{'delta':>11}{'heads winning':>15}")
    for group in GROUPS:
        n = t[(t["group"] == group) & (t["metric"] == "n_live")]["n"]
        won = t[(t["gate"] == "crps") & (t["group"] == group) & (t["metric"] == "wins")]
        if not len(won):
            print(f"  {group:<16}{int(n.max()) if len(n) else 0:>6}   (too few rows)")
            continue
        print(f"  {group:<16}{int(n.max()):>6}"
              f"{mean_of('crps', group, 'val_crps'):>12.4f}"
              f"{mean_of('crps', group, 'unserved_crps'):>11.4f}"
              f"{mean_of('crps', group, 'crps_delta'):>11.4f}"
              f"{int(won['value'].sum()):>10} of {len(won)}")

    print("\nGate 2 — carry-forward R2 on the ladder's own column, against the shipped "
          f"floor band {FLOOR_BAND[0]}-{FLOOR_BAND[1]}")
    print(f"  {'rung':<16}" + "".join(f"{c:>8}" for c in COUNT_HEADS)
          + f"{'counts':>9}{'9 rates':>9}{'gate':>7}")
    for group in GROUPS:
        band = t[(t["gate"] == "band") & (t["group"] == group)
                 & (t["metric"] == "mean_in_band")]
        if not len(band):
            continue
        cells = "".join(f"{val_of('band', group, c, 'floor_r2'):>8.3f}"
                        for c in COUNT_HEADS)
        print(f"  {group:<16}{cells}"
              f"{val_of('band', group, '', 'count_mean_r2'):>9.4f}"
              f"{val_of('band', group, '', 'rate_mean_r2'):>9.4f}"
              f"{'in' if band['value'].iloc[0] == 1.0 else 'OUT':>7}")
    print("  the per-head cells are diagnosis, not the gate: on these same veteran rows "
          "only\n  5 of 7 count heads sit inside the band, `fga` and `reb` being ABOVE it.")

    print("\nThe verdict, per rung — §4's conjunction, stated before the result")
    for group in LADDER_RUNGS:
        admitted = val_of("verdict", group, "", "admitted")
        print(f"  {group:<16} {'ADMITTED' if admitted == 1.0 else 'rejected'}"
              f"   CRPS {int(val_of('verdict', group, '', 'heads_winning_crps'))}"
              f"/{int(val_of('verdict', group, '', 'heads_scored_crps'))} heads, "
              f"count-head mean R2 "
              f"{val_of('band', group, '', 'count_mean_r2'):.4f} "
              f"({'in' if val_of('verdict', group, '', 'count_mean_in_band') == 1.0 else 'OUT OF'}"
              f" band)")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
