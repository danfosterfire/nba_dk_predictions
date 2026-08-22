"""The eleven rookie rate heads, fitted, and §4's per-head ship gate —
`docs/rookie-rates-plan.md` §5d.

`make stan-rookie`. One artifact, `outputs/predictions/rookie_rate_metrics.csv`, in
`stan_component_metrics.csv`'s shape with the gate's columns appended.

## What Session 3 left and this session decides

§7d built the true-rookie design and eleven **no-fit floors** — P4(b)'s volume-shrunk
preseason blend wrapped in each head's own likelihood — and measured that the floor beats
the estimator shipping today on CRPS on 11 of 11 heads. It fitted nothing. This module puts
a sampler on the design and answers the only question left: **does a fitted head beat that
floor**, and therefore does a head ship fitted or ship its floor as a plug-in.

The ship rule is not "scored versus unscored". Every unit must carry all eleven quantities
to enter the tensor at all, so a head that fails still ships — as the floor estimator. The
gate decides *which arm*, never *whether*.

## The gate, stated in §4 before any result

A conjunction, and both halves are computed as code rather than read after the fact:

1. **Validation.** The selected variant's paired-bootstrap CRPS interval against the floor
   lies entirely below zero, on the **draftable season-start-roster** population — 108 rows,
   which is small, and the reason there is a second half at all.
2. **The rolling-origin harness on the fitting half.** Same sign, interval below zero, and a
   majority of origins won.

`components_preseason`'s bar verbatim, one family over, and for the reason that module
states: `docs/availability-window-plan.md` §12e and §14f record two blocks that won a
validation reading and shrank 4-6x rolling. On 108 rows that risk is larger, not smaller.

**The rolling half is Stan too, not a point-MLE stand-in.** `components_preseason` and
`minutes_preseason` confirm with cheap point fits because their rolling harness would
otherwise cost hours; here it costs minutes (see `METRIC`), so both halves of the gate are
the same estimator against the same floor and "agreeing" means what it says.

## Three variants, and why the ladder is shorter than the veteran one

`count_variants` walks raw -> log -> spline because a veteran head's own feature is a
prior-season rate that has to be put on the link scale first. A rookie head's own feature is
**already** on it: `rookie_rates.with_levels` builds `log1p(per-36)` for a count and a logit
for a conversion, because a level centred against the population is what replaces the
veteran delta. So the scale rung does not exist and §5d's ladder is the other two:

- `linear` — the shrunk level, the four missing indicators, the four slot indicators,
  years-since-draft, and age/age_sq.
- `slot_interaction` — plus slot x years-since-draft. **This is §5c's shipped feature list
  exactly**, and it is built by removing columns from `rookie_rates.head_features` rather
  than by restating it, so the two cannot drift.
- `slot_interaction_spline` — curvature on the level, the basis replacing the linear term.
  With linear extrapolation the basis contains the linear function, so the arm strictly
  nests the one below it and the contrast is curvature alone.

P4(b) set the expectation the first two rungs exist to test: **slot alone is an anti-model
for rates**, so it earns its place in interaction or not at all.

**The spline arm is the one rung that is not literally zero-recovering, and that is named
rather than hidden.** A B-spline basis evaluated at a level of exactly 0 is a constant
vector, not a zero one, so a row with no preseason reading contributes a constant shift
instead of nothing. The four missing indicators partition exactly those rows and absorb it,
so no row's prediction depends on a level nobody measured — which is the property that
mattered — but the column itself is not zero and this should not claim it is.

## The fitting rows, and why they are not all 1,218

Both arms fit on `train` minus the first covered season. The floor's dispersion needs a
bucket prior, and `bucket_priors` declines to emit one for 2004-05 because nothing precedes
it; giving the Stan head 57 rows the floor cannot have would make the comparison a
comparison of two fitting populations. `rookie_rates.scorable_rows` is that restriction and
it is applied once, to both.

## Where this departs from `stan_component_metrics.csv`'s shape

Two things added, both because the reading needs them:

- **`population`**, `all` or `draftable`. §4 reads the gate on the draftable subpopulation
  (P1 decision 5) and §7d quotes both, and one row per (head, variant) could carry only one.
- **the gate block** — the paired-bootstrap interval, the rolling harness and the verdict.
  `selected` and `beats_floor` keep their `_finalize` meaning: an R2 (count) or NLL
  (conversion) comparison, head-local. They are **not** the gate. The gate is a CRPS
  interval, `passes`, and it can disagree with `beats_floor` — a head can improve a point
  estimate without its predictive interval clearing zero on 108 rows.

The floor's row carries `variant = "no_fit_floor"` rather than `carry_forward`: the
estimator is not a carry-forward and a shared name would invite reading the two families'
floor rows as the same arithmetic.

Usage:
    python -m src.models.stan_rookie
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.preseason_value import attach_season_start_roster, covered_seasons
from src.models import rookie_rates as rr
from src.models.component_rates import COUNT_HEADS, TEST_SEASONS, add_spline
from src.models.held_out import selection_split
from src.models.minutes_unification import paired_bootstrap, verdict
from src.models.stan_components import (SPLINE_KNOTS, StanConversion, StanCount, _iters,
                                        merge_heads, score_conversion, score_count)
from src.models.stan_utils import crps_from_samples

#: The bootstrap replicate count every paired reading in this project uses
#: (`minutes_window.BOOTSTRAP_REPS`, `lag_ladder.BOOTSTRAP_REPS`).
BOOTSTRAP_REPS = 2000

#: **A dense mass matrix, and it is not a tuning preference — it is a 30x speedup.**
#: The slot block is four indicators, years-since-draft and their four products, which are
#: strongly linearly correlated by construction (a product is zero wherever its indicator
#: is), and `age`/`age_sq` add a second such pair. On `fga` the shipped diagonal metric
#: saturated treedepth on 793 of 4,000 draws and took 90.5 s; `dense_e` saturates 0, takes
#: 3.0 s, and lands on the same answer (R2 0.9633 against 0.9631, CRPS 22.82 against 22.71).
#: This is exactly the case `stan_utils.sample` documents the knob for — cheap at these
#: dimensions — and it is what makes the rolling half of the gate affordable in Stan.
METRIC = "dense_e"

#: The population §4 reads the verdict on. `all` is scored beside it because the gap between
#: the two is what says whether a head is being carried by rows no board ever prices.
DECISION_POPULATION = "draftable"

#: Smallest fitting frame an origin may have. The design carries 57-94 rookies per season,
#: so this puts the first origin at 2010-11 (six covered seasons, 350 rows) — below it a
#: 16-to-22 feature head is fitting more parameters than a season has rookies.
MIN_FIT_ROWS = 300

#: The floor arm every fitted head is read against — §7d's `shrunk` blend, never the
#: `draft_bucket` incumbent, which that session measured as an anti-model on five heads.
FLOOR_VARIANT = "no_fit_floor"

#: The ladder, in nesting order, so the report, the artifact and the doc order alike.
VARIANTS = ("linear", "slot_interaction", "slot_interaction_spline")

SEED = 42


# ── The variant ladder ────────────────────────────────────────────────────────

def variants(train: pd.DataFrame, test: pd.DataFrame, component: str,
             n_knots: int = SPLINE_KNOTS
             ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """§5d's three rungs, each defined by subtraction from `rookie_rates.head_features`.

    Subtraction rather than restatement: `head_features` is what Session 6 persists and what
    the forward design (§5g) builds, so a rung assembled independently here could ship a
    head fitted on columns the recipe does not carry.

    No imputation step, unlike `count_variants`. The rookie design has no NaN to fill — every
    block is built to be exactly 0 where its information is absent — so an `impute` call
    would add missing flags that are constant zero and buy nothing.
    """
    full = rr.head_features(component)
    level = rr.shrunk_column(component)
    interactions = set(rr.SLOT_INTERACTION_COLS)
    out = {"linear": (train, test, [c for c in full if c not in interactions]),
           "slot_interaction": (train, test, list(full))}
    tr, te, basis = add_spline(train, test, [level], n_knots)
    out["slot_interaction_spline"] = (tr, te, [c for c in full if c != level] + basis)
    return {name: out[name] for name in VARIANTS}


# ── Predictives: the fitted arm and the floor, on identical rows ──────────────

def live_rows(frame: pd.DataFrame, attempted: str | None) -> pd.DataFrame:
    """Rows a head can score — a conversion head needs at least one realized attempt.

    Applied by the **caller**, once, before either arm sees the frame. Both
    `stan_components._score_conv` and `rookie_rates.conversion_floor_predictive` apply the
    same restriction internally, so handing them a frame that is already restricted makes
    those no-ops and makes the row-for-row pairing the bootstrap assumes a fact rather than
    a hope.
    """
    if attempted is None:
        return frame.reset_index(drop=True)
    return frame[frame[attempted].to_numpy(dtype=float) > 0].reset_index(drop=True)


def realized(frame: pd.DataFrame, component: str,
             attempted: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    """`(y, n)` on the head's own scale — `n` is None for a count head.

    `min(made, attempted)` for a conversion, which is the same clamp `StanConversion.fit`
    and every floor in the project applies: the box score occasionally records more makes
    than attempts and a beta-binomial cannot take that row.
    """
    if attempted is None:
        return frame[component].to_numpy(dtype=float), None
    n = np.rint(frame[attempted].to_numpy(dtype=float))
    return np.minimum(np.rint(frame[component].to_numpy(dtype=float)), n), n


def floor_predictive(train: pd.DataFrame, test: pd.DataFrame, priors: pd.DataFrame,
                     component: str, attempted: str | None,
                     constants: rr.RookieConstants, seed: int = SEED
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """`(point prediction, draws x rows samples, dispersion)` for §7d's floor on `test`.

    Recomputed here rather than joined in from `rookie_rate_floors.csv`, because the gate
    needs the **per-row** predictive the paired bootstrap resamples and because this frame
    is not that artifact's frame — the caller has already cut it to live rows. A silently
    misaligned join is the one failure mode a paired test cannot detect.
    """
    k = constants.volume[component]
    if attempted is None:
        _, mu, samples, phi = rr.count_floor_predictive(train, test, priors, component, k,
                                                        rr.FLOOR_ARM, seed)
        return mu, samples, phi
    _, _, p, samples, rho = rr.conversion_floor_predictive(
        train, test, priors, component, attempted, k, *constants.conversion[component],
        rr.FLOOR_ARM, seed)
    return p, samples, rho


def fitted_predictive(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                      component: str, attempted: str | None, label: str, chains: int,
                      iters: dict, seed: int = SEED
                      ) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """`(point prediction, draws x rows samples, dispersion, diagnostics)` for one arm.

    Predictions handed back rather than metrics, so one fit serves both populations and both
    the level and the paired reading. A second fit at the same seed would be bit-identical —
    that is the property `merge_heads` rests on — but paying for it would double the sweep.
    """
    if attempted is None:
        model = StanCount(features, component, name=label, chains=chains, seed=seed,
                          metric=METRIC, **iters).fit(train)
        return (model.predict_mean(test), model.predict_samples(test, seed), model.phi,
                model.diagnostics)
    model = StanConversion(features, component, attempted, name=label, chains=chains,
                           seed=seed, metric=METRIC, **iters).fit(train)
    return (model.predict_p(test), model.predict_samples(test, seed), model.rho,
            model.diagnostics)


def score(frame: pd.DataFrame, component: str, attempted: str | None, point: np.ndarray,
          samples: np.ndarray, dispersion: float, mask: np.ndarray | None = None,
          seed: int = SEED) -> dict:
    """One arm's metrics, optionally restricted to a subpopulation.

    Restricting the *scoring* and not the *fit*: a head fitted on draftable rows alone would
    be a different head, and P1 decision 5's restriction is about which rows a figure is
    quoted on rather than about which rows teach the coefficients.
    """
    keep = np.ones(len(frame), dtype=bool) if mask is None else mask
    y, n = realized(frame, component, attempted)
    if attempted is None:
        return score_count(y[keep], point[keep], samples[:, keep], dispersion, seed)
    return score_conversion(y[keep], n[keep], point[keep], samples[:, keep], dispersion,
                            seed)


def per_row_crps(frame: pd.DataFrame, component: str, attempted: str | None,
                 samples: np.ndarray) -> np.ndarray:
    """The CRPS `score_*` averages, kept per row — what the paired bootstrap resamples."""
    y, _ = realized(frame, component, attempted)
    return crps_from_samples(samples, y)


def population_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """`all` and `draftable`, over an already-live frame."""
    return {"all": np.ones(len(frame), dtype=bool),
            DECISION_POPULATION: frame["on_season_start_roster"].to_numpy(float) > 0}


# ── Half one: the validation sweep ────────────────────────────────────────────

def _metrics(scored: dict) -> dict:
    """`score_*`'s dict under `stan_component_metrics.csv`'s `val_` prefix."""
    return {f"val_{k}": v for k, v in scored.items()}


def validation_rows(train: pd.DataFrame, val: pd.DataFrame, priors: pd.DataFrame,
                    constants: rr.RookieConstants, component: str, attempted: str | None,
                    label: str, kind: str, chains: int, iters: dict, n_knots: int,
                    seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """One head: the floor and three fitted arms, scored on both populations.

    The fits are shared across populations — a head is fitted once and scored twice — which
    is `components_preseason.rolling_confirmation`'s arrangement and the reason the
    draftable reading the verdict is taken on costs nothing extra.
    """
    te = live_rows(val, attempted)
    masks = population_masks(te)
    arms: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, float, int]] = {}
    diagnostics = []

    point, samples, dispersion = floor_predictive(train, te, priors, component, attempted,
                                                  constants, seed)
    arms[FLOOR_VARIANT] = (te, point, samples, dispersion, 0)
    for name, (v_tr, v_te, features) in variants(train, te, component, n_knots).items():
        point, samples, dispersion, diag = fitted_predictive(
            v_tr, v_te, features, component, attempted, f"{label}/{name}/rookie", chains,
            iters, seed)
        diagnostics.append(diag)
        arms[name] = (v_te, point, samples, dispersion, len(features))

    floor_crps = per_row_crps(te, component, attempted, arms[FLOOR_VARIANT][2])
    rows = []
    for name, (frame, point, samples, dispersion, n_features) in arms.items():
        crps = (floor_crps if name == FLOOR_VARIANT
                else per_row_crps(frame, component, attempted, samples))
        for population, mask in masks.items():
            row = {"head": label, "kind": kind, "variant": name,
                   "n_features": n_features, "population": population,
                   "n_scored": int(mask.sum()),
                   **_metrics(score(frame, component, attempted, point, samples,
                                    dispersion, mask, seed))}
            if name == FLOOR_VARIANT:
                row.update({"crps_vs_floor": 0.0, "crps_vs_floor_lo": 0.0,
                            "crps_vs_floor_hi": 0.0, "verdict_vs_floor": "floor"})
            else:
                delta = paired_bootstrap(crps[mask], floor_crps[mask],
                                         n_boot=BOOTSTRAP_REPS, seed=seed)
                row.update({"crps_vs_floor": delta["crps_delta"],
                            "crps_vs_floor_lo": delta["ci_lo"],
                            "crps_vs_floor_hi": delta["ci_hi"],
                            "verdict_vs_floor": verdict(delta)})
            rows.append(row)
    return rows, diagnostics


# ── Half two: the rolling-origin harness, fitting half only ───────────────────

def origins(train: pd.DataFrame, min_fit_rows: int = MIN_FIT_ROWS) -> list[int]:
    """Season start years that have `min_fit_rows` covered rookies strictly before them."""
    years = train["season_start_year"].to_numpy(dtype=int)
    return [int(y) for y in sorted(np.unique(years)) if int((years < y).sum())
            >= min_fit_rows]


def rolling_rows(train: pd.DataFrame, priors: pd.DataFrame,
                 constants: rr.RookieConstants, component: str, attempted: str | None,
                 label: str, kind: str, chains: int, iters: dict, n_knots: int,
                 seed: int = SEED, min_fit_rows: int = MIN_FIT_ROWS
                 ) -> tuple[list[dict], list[dict]]:
    """Walk-forward over the fitting half: one fit per (origin, variant), floor beside it.

    **This is half the gate, not a post-hoc check.** Every row touched is a fitting-half row
    and each origin fits on every covered season before it, so nothing here reads validation.
    Origins are the independent replicates, which is what makes a win count over them the
    multiplicity-robust statement a single pooled interval is not.
    """
    years = train["season_start_year"].to_numpy(dtype=int)
    pooled: dict[tuple[str, str], dict[str, list]] = {}
    diagnostics = []

    for origin in origins(train, min_fit_rows):
        fit_rows = train[years < origin].reset_index(drop=True)
        score_rows = live_rows(train[years == origin], attempted)
        if score_rows.empty:
            continue
        masks = population_masks(score_rows)
        arms = {FLOOR_VARIANT: (score_rows,) + floor_predictive(
            fit_rows, score_rows, priors, component, attempted, constants, seed)}
        for name, (v_tr, v_te, features) in variants(fit_rows, score_rows, component,
                                                     n_knots).items():
            point, samples, dispersion, diag = fitted_predictive(
                v_tr, v_te, features, component, attempted,
                f"{label}/{name}/origin{origin}", chains, iters, seed)
            diagnostics.append(diag)
            arms[name] = (v_te, point, samples, dispersion)

        for name, (frame, _point, samples, _dispersion) in arms.items():
            crps = per_row_crps(frame, component, attempted, samples)
            for population, mask in masks.items():
                slot = pooled.setdefault((name, population),
                                         {"crps": [], "origin": [], "n_fit": []})
                slot["crps"].append(crps[mask])
                slot["origin"].append(np.full(int(mask.sum()), origin))
                slot["n_fit"].append(len(fit_rows))
        print(f"    origin {origin}: fitted {len(fit_rows):,} rows, scored "
              f"{len(score_rows):,} ({int(masks[DECISION_POPULATION].sum()):,} draftable)")

    rows = []
    for population in ("all", DECISION_POPULATION):
        floor = np.concatenate(pooled[(FLOOR_VARIANT, population)]["crps"])
        origin_of = np.concatenate(pooled[(FLOOR_VARIANT, population)]["origin"])
        for name in (FLOOR_VARIANT,) + VARIANTS:
            crps = np.concatenate(pooled[(name, population)]["crps"])
            delta = paired_bootstrap(crps, floor, n_boot=BOOTSTRAP_REPS, seed=seed)
            won = sum(1 for o in np.unique(origin_of)
                      if crps[origin_of == o].mean() < floor[origin_of == o].mean())
            rows.append({
                "head": label, "kind": kind, "variant": name, "population": population,
                "n_rolling": int(len(crps)),
                "n_origins": int(len(np.unique(origin_of))),
                "mean_fit_rows": float(np.mean(pooled[(name, population)]["n_fit"])),
                "rolling_crps": float(crps.mean()),
                "rolling_crps_vs_floor": delta["crps_delta"],
                "rolling_lo": delta["ci_lo"], "rolling_hi": delta["ci_hi"],
                "origins_won": int(won),
                "rolling_verdict": "floor" if name == FLOOR_VARIANT else verdict(delta)})
    return rows, diagnostics


# ── The gate, as code ─────────────────────────────────────────────────────────

def selection_metric(kind: str) -> tuple[str, bool]:
    """`(column, higher_is_better)` — `stan_components._finalize`'s rule, unchanged.

    R2 for a count and NLL for a conversion, deliberately **not** CRPS: the shipped sweep
    picks its variant this way, and giving this family its own criterion would make a
    `selected` column here mean something different from the one beside it. The CRPS
    interval is the *gate*, and the two answering different questions is the point —
    `beats_floor` says the point estimate improved, `passes` says the predictive did.
    """
    return ("val_r2", True) if kind == "count" else ("val_nll", False)


def finalize(table: pd.DataFrame) -> pd.DataFrame:
    """`selected`, `beats_floor`, the two gate halves, `passes` and `ships`.

    `selected` is head-local and decided on the **decision population**, then carried onto
    that head's `all` rows so one variant per head is marked however the table is filtered.
    `beats_floor` is per (head, population), which is `_finalize`'s own per-block meaning.
    """
    out = table.copy()
    for col in ("selected", "beats_floor", "val_pass", "rolling_pass", "passes"):
        out[col] = False
    out["ships"] = FLOOR_VARIANT

    for head, block in out.groupby("head"):
        column, higher = selection_metric(str(block["kind"].iloc[0]))
        for population, part in block.groupby("population"):
            floor = part[part["variant"] == FLOOR_VARIANT][column]
            if floor.empty:
                continue
            better = ((part[column] > floor.iloc[0]) if higher
                      else (part[column] < floor.iloc[0]))
            out.loc[part.index, "beats_floor"] = better
            out.loc[part[part["variant"] == FLOOR_VARIANT].index, "beats_floor"] = True

        decision = block[(block["population"] == DECISION_POPULATION)
                         & (block["variant"] != FLOOR_VARIANT)]
        if decision.empty:
            continue
        best = decision[column].idxmax() if higher else decision[column].idxmin()
        winner = str(out.loc[best, "variant"])
        out.loc[block[block["variant"] == winner].index, "selected"] = True

        row = out.loc[best]
        val_pass = bool(row["crps_vs_floor_hi"] < 0.0)
        rolling_pass = bool(row["rolling_hi"] < 0.0
                            and row["origins_won"] * 2 > row["n_origins"])
        passes = bool(val_pass and rolling_pass)
        selected_rows = block[block["variant"] == winner].index
        out.loc[selected_rows, "val_pass"] = val_pass
        out.loc[selected_rows, "rolling_pass"] = rolling_pass
        out.loc[selected_rows, "passes"] = passes
        out.loc[block.index, "ships"] = winner if passes else FLOOR_VARIANT
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def fitting_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """`(train, validation, covered seasons)` for the rookie family.

    `rookie_rates.head_design` is the design **with** the fitted shrink applied, read from
    `rookie_rate_floors.csv` — so this module cannot fit a head on a block whose constants
    came from a different run of Session 3.
    """
    design = rr.head_design(cfg)
    window_games = int(cfg.get("features", {}).get("team_context", {})
                       .get("roster_window_games", 10))
    design = attach_season_start_roster(design, list(cfg["data"]["seasons"]),
                                        cfg["data"]["raw_dir"], window_games)
    eda_dir = Path(cfg["evaluation"].get("eda_dir", "outputs/eda"))
    covered = covered_seasons(pd.read_csv(eda_dir / "preseason_coverage.csv"))
    train, val = selection_split(design, TEST_SEASONS)
    return train, val, covered


def head_priors(history: pd.DataFrame, component: str, attempted: str | None,
                covered: list[str]) -> pd.DataFrame:
    """One head's expanding draft-bucket prior table, `rookie_rates`' own builder.

    Built over the **whole** train+validation history for the reason `rookie_rates.floor_rows`
    does: the window is expanding and reads seasons strictly before the target, so a
    validation season's prior is knowable in September and a training season's prior cannot
    see one. Handing it the already-restricted fitting frame instead would delete the first
    covered season from the history and so delete the *second* season's prior too —
    the restriction would eat one more season on every pass.
    """
    target = f"{component}_p36" if attempted is None else f"{component}_pct"
    return rr.bucket_prior_table(history, target, covered)


def fitting_rows(train: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """The rows BOTH arms fit on — `rookie_rates.floor_rows`' own restriction.

    Per head rather than once, because `bucket_priors` emits a season only when its target
    has history, and a head is entitled to a different answer even though in practice every
    head loses exactly the first covered season.
    """
    return train[rr.scorable_rows(train, priors)].reset_index(drop=True)


def run(cfg: dict, heads: tuple[str, ...] | None = None) -> Path:
    """The full sweep, or `heads` alone merged into the artifact already on disk.

    `merge_heads`' contract holds here for the reason it holds for the veteran family: the
    eleven heads are fitted separately, selection is head-local, and refitting one leaves
    the other ten bit-identical at a fixed seed.
    """
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    constants = rr.rookie_constants(out_dir / "rookie_rate_floors.csv")
    cfg_stan = cfg.get("stan", {})
    chains = int(cfg_stan.get("chains", 4))
    seed = int(cfg_stan.get("seed", SEED))
    iters = _iters(cfg_stan)
    n_knots = int(cfg_stan.get("components", {}).get("spline_knots", SPLINE_KNOTS))

    train_all, val, covered = fitting_frames(cfg)
    val_live = val.reset_index(drop=True)
    history = pd.concat([train_all, val_live], ignore_index=True)
    print("Rookie rate heads — the eleven fitted arms and §4's per-head ship gate")
    print(f"  The test split is LOCKED — this sweep fits and scores VALIDATION only "
          f"(src/models/held_out.py).")

    # Both arms fit the same rows: the floor needs an expanding bucket prior and the first
    # covered season has none, so handing the sampler rows the floor cannot have would make
    # this a comparison of two fitting populations rather than of two estimators.
    reference = fitting_rows(train_all, head_priors(history, COUNT_HEADS[0], None, covered))
    draftable = int((val_live["on_season_start_roster"] > 0).sum())
    print(f"  {len(reference):,} fit / {len(val_live):,} score "
          f"({', '.join(sorted(val_live['season'].unique()))} as validation); "
          f"{draftable:,} of the validation rows are draftable, which is the population\n"
          f"  §4 reads the verdict on.")
    print(f"  {len(train_all) - len(reference):,} training rows dropped: {covered[0]} has "
          f"no expanding bucket prior, so the floor cannot score it and neither may the "
          f"sampler.")
    print(f"  metric={METRIC} on every fit — the slot block's four products are collinear "
          f"with their own\n  indicators and a diagonal mass matrix saturates treedepth "
          f"(90.5 s and 793 saturations on\n  `fga` against 3.0 s and 0, same answer).")
    roll = origins(reference)
    print(f"  {len(roll)} rolling origins on the FITTING half ({roll[0]}-{roll[-1]}), "
          f">= {MIN_FIT_ROWS:,} fitting rows each.\n")

    blocks, diagnostics = [], []
    for label, component, attempted in rr.head_list():
        if heads is not None and label not in heads:
            continue
        kind = "count" if attempted is None else "conversion"
        priors = head_priors(history, component, attempted, covered)
        train = fitting_rows(train_all, priors)
        print(f"  {label} ({kind}) — {len(train):,} fitting rows; validation, then "
              f"{len(roll)} rolling origins")
        val_block, val_diag = validation_rows(train, val_live, priors, constants,
                                              component, attempted, label, kind, chains,
                                              iters, n_knots, seed)
        roll_block, roll_diag = rolling_rows(train, priors, constants, component,
                                             attempted, label, kind, chains, iters,
                                             n_knots, seed)
        diagnostics += val_diag + roll_diag
        blocks.append(pd.DataFrame(val_block).merge(
            pd.DataFrame(roll_block), on=["head", "kind", "variant", "population"],
            how="left"))

    fresh = finalize(pd.concat(blocks, ignore_index=True))
    dest = out_dir / "rookie_rate_metrics.csv"
    existing = pd.read_csv(dest) if dest.exists() else None
    table = merge_heads(existing, fresh)
    _report(table)
    unconverged = [d for d in diagnostics if not d.get("converged", True)]
    if unconverged:
        print(f"\n/!\\  {len(unconverged)} of {len(diagnostics)} fits did not converge: "
              + ", ".join(sorted(d["label"] for d in unconverged)))
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} rookie head rows → {dest}")
    return dest


#: Short labels for the report only — the artifact carries `VARIANTS` verbatim.
SHORT = {"linear": "linear", "slot_interaction": "+slot x yrs",
         "slot_interaction_spline": "+spline"}


def _report(t: pd.DataFrame) -> None:
    decision = t[t["population"] == DECISION_POPULATION]
    print(f"\nValidation · {DECISION_POPULATION} — the ladder against §7d's no-fit floor "
          f"(CRPS, lower is better)")
    print(f"  {'head':<11}{'floor':>9}"
          + "".join(f"{SHORT[v]:>20}" for v in VARIANTS))
    for label, _, _ in rr.head_list():
        block = decision[decision["head"] == label]
        if block.empty:
            continue
        cells = ""
        for name in VARIANTS:
            row = block[block["variant"] == name]
            if row.empty:
                cells += " " * 20
                continue
            mark = "*" if row["crps_vs_floor_hi"].iloc[0] < 0 else " "
            cells += (f"{row['val_crps'].iloc[0]:>10.4f}"
                      f"{row['crps_vs_floor'].iloc[0]:>+9.4f}{mark}")
        floor = block[block["variant"] == FLOOR_VARIANT]["val_crps"]
        print(f"  {label:<11}{floor.iloc[0] if len(floor) else float('nan'):>9.4f}{cells}")
    print("  each fitted cell is its CRPS then the paired delta against the floor; `*` "
          "marks an\n  interval entirely below zero.")

    print("\nThe gate, per head — §4's conjunction, stated before the result")
    print(f"  {'head':<11}{'ships':<17}{'validation, draftable':>28}"
          f"{'rolling origins, fitting half':>34}")
    for label, _, _ in rr.head_list():
        block = decision[(decision["head"] == label) & decision["selected"]]
        if block.empty:
            continue
        row = block.iloc[0]
        ships = f"FITTED {SHORT[row['variant']]}" if row["passes"] else "floor"
        print(f"  {label:<11}{ships:<17}"
              f"{row['crps_vs_floor']:>+9.4f} [{row['crps_vs_floor_lo']:+.4f}, "
              f"{row['crps_vs_floor_hi']:+.4f}]"
              f"{row['rolling_crps_vs_floor']:>+11.4f} "
              f"[{row['rolling_lo']:+.4f}, {row['rolling_hi']:+.4f}] "
              f"{int(row['origins_won']):>2}/{int(row['n_origins'])}")
    shipped = decision[decision["selected"] & decision["passes"]]
    print(f"\n  {len(shipped)} of {decision['head'].nunique()} heads ship FITTED; the rest "
          f"ship §7d's floor estimator as a plug-in.\n  Every unit carries all eleven "
          f"quantities either way — the gate decides which arm, never whether.")


if __name__ == "__main__":
    from src.models.stan_rookie import run as _run

    _run(yaml.safe_load(open("configs/default.yaml")))
