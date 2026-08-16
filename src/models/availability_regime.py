"""Recency against representativeness, and shrinkage toward the long-window fit.

`make availability-regime` → four rolling artifacts plus one validation confirmation.

## The two axes, and why they are one round

`docs/availability-window-plan.md` §9 leaves exactly two live items on the availability
head's fitting window, and §5b handed both of them the same motive: **every optimization
that round failed by leaning on the COVID trough.** A shorter effective window, a geometric
decay and an eight-season intercept all beat the plain window on the rolling harness and
all lost on validation, because the most recent training seasons are the trough and the
validation seasons partially recovered.

| item | axis | the instrument |
|---|---|---|
| §5.7 / §9.3 | **recency ≠ representativeness** | an explicit regime indicator for 2019-20 → 2021-22, or excluding those seasons |
| §5.5(c) / §9.4 | **shrinkage toward the long window** | blocks fitted jointly under different priors, not transplanted |

They are one round because they are the same complaint from two sides. A lookback says
"old rows are wrong"; the trough says "some *recent* rows are wrong". Shrinkage says
"neither — old rows are informative about most coefficients and misleading about a few, so
price them per coefficient instead of keeping or discarding them wholesale."

Nothing here rebuilds §5's (a), (b) or (d) — split `beta`/`rho` windows, exponential decay
and per-block *windowing* are measured nulls in `availability_weighting.py` and are not
re-run. §5b's spliced arm is carried as one **control row** in the shrinkage table, because
the whole claim of axis (2) is that shrinkage is the principled version of that arm and a
claim like that is only readable beside the thing it replaces.

## The blindness this round had to design around, and the shape of the fix

**The plain rolling harness cannot see the regime axis.** Origins run 2009 → 2021 on the
fitting half, and the regime block is 2019, 2020, 2021. An arm that excludes or indicates
those seasons produces a *bit-identical* fit to the incumbent at every origin whose fitting
rows end before 2019 — that is 11 of the 13. The two that remain (2020 and 2021) are
exactly the two origins §4b already caught the season trend's apparent gain hiding in.

So `regime_x_lookback` is run anyway, because §5.7 asks for the cross with lookback and
because the blindness should be a *measured* count rather than an assertion — every arm
carries `origins_active`, the number of origins at which its fitting frame differed from
the incumbent's at all. But it cannot be the selector.

`regime_contamination` is the selector, and it changes one thing: it **injects the regime
block into a non-regime origin's fitting rows**. For origin `o` ≤ 2018 the arms are

- `clean` — fit on seasons `< o`. The target: the fit you would have had.
- `contaminated` — fit on seasons `< o` **plus** 2019-21. The failure mode, staged.
- an instrument on top of `contaminated` — does it recover `clean`?

That is anachronistic by construction: the injected seasons are in the origin's future, so
this is an **ablation of a mechanism and not a forecast**, and no arm here may be read as a
walk-forward score. It buys 10 replicates of the situation the production fit is actually
in — fitting set contains the trough, target season does not — where the walk-forward
harness has **zero**, because the first non-regime target season after the trough is
2022-23 and that is validation. `regime_placebo` is its control: the same injection with a
*non-regime* block of the same size (2016-18), which separates "the injected rows are
unrepresentative" from "the injected rows are out of order".

Note also that `exclude` is not an arm in the contamination harness. Excluding the injected
block *is* `clean`, exactly, so it would be the reference wearing a second name. What the
contamination harness can test is whether an instrument that **keeps the rows** gets back
what exclusion gets for free.

## Selection reads the fitting half; validation is read once, at the end

`confirm_on_validation` is the only function here that touches a validation row, it runs
after every recipe above is fixed, and it scores through `availability_window.score_arm` so
its numbers sit directly beside `availability_window.csv` and
`availability_weighting_confirmation.csv`.

**It confirms on two likelihoods, not one.** The rolling selection runs on
`RoleGradedBetaBinomial` — the point-MLE arm §4 selected, which is what §5b optimized and
what makes this table comparable to that one, and which fits in 0.1 s against the mixture's
40 s. But the head that *ships* is the two-component mixture (§7i), and a row-weighting or
coefficient-shrinkage recipe that helps a single-component head has no guarantee of
surviving a likelihood whose second component already exists to absorb disrupted seasons.
So the confirmation carries the shipped mixture on the same arms. If a recipe wins on the
beta-binomial and dies on the mixture, that is the finding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import digamma
from sklearn.preprocessing import StandardScaler

from src.eda.season_effects import ROLE_LABELS
from src.features.availability import build_panel, season_availability
from src.models.availability import (FEATURE_COLS, RHO_MAX, RHO_MIN, _ab, _sigmoid,
                                     build_design, season_start_dates)
from src.models.availability_weighting import (WeightedBetaBinomialGLM, _collect, _pool,
                                               block_indices, fit_dispersion_weighted,
                                               weighted_neg_loglik)
from src.models.availability_window import (FIRST_ORIGIN, LIKELIHOOD_WINDOW, MIN_ROLE_ROWS,
                                            WINDOWS, MixtureFrailty, RoleGradedBetaBinomial,
                                            restrict_window, score_arm)
from src.models.held_out import selection_split
from src.models.season_terms import season_start_year

# ── The regime ────────────────────────────────────────────────────────────────
#
# Target season START years. 2019-20 → 2021-22 is the block §5.7 names: the bubble, the
# 72-game season, and the season whose COVID protocols produced the deepest trough in the
# series. `team_games` is already per-row, so the *schedule* half of those seasons is
# handled by the likelihood — what is left is the availability RATE, which on the head's
# own fitting rows reads 0.6739 / 0.6334 / 0.6052 against a twenty-season level near 0.710.
REGIME_YEARS: tuple[int, ...] = (2019, 2020, 2021)

# The sensitivity, and it is not cosmetic. The league series does not single out 2019-20 at
# all: at 0.6739 it sits between 2017-18 (0.6762) and 2018-19 (0.6704), both of which are
# ordinary seasons on the far side of the sup-F break §2 puts at 2017-18. So "the COVID
# trough" as a distinct regime is arguably the last two seasons only, and an arm that
# excludes 2019-20 with them is throwing away a representative season. Both are fitted.
REGIME_CORE: tuple[int, ...] = (2020, 2021)

# The placebo block for `regime_contamination`: the three non-regime seasons immediately
# before it, same size, same anachronism, ordinary league level. Injecting these instead
# isolates "unrepresentative" from "out of chronological order".
PLACEBO_YEARS: tuple[int, ...] = (2016, 2017, 2018)

# Partial exclusion. The weight family is one knob whose endpoints are the two things §5.7
# proposes: `1.0` is the incumbent and `0.0` is dropping the seasons, so "exclude" is not a
# separate arm but a limit of this one.
REGIME_WEIGHTS: tuple[float, ...] = (1.0, 0.50, 0.25, 0.0)

REGIME_TARGET_COL = "regime_target"
REGIME_LAG_COL = "regime_lag"

# Lookbacks, matching `availability_weighting.LOOKBACK_GRID` so the cross in
# `regime_x_lookback` is readable beside that module's tables.
LOOKBACK_GRID: tuple[int | None, ...] = (None, 12, 8, 5, 3)

# ── Shrinkage ─────────────────────────────────────────────────────────────────
#
# The prior precision on `beta_j - beta_long_j`, swept. `0.0` reproduces the plain
# short-window fit EXACTLY and `inf` pins the block to the long-window estimate exactly, so
# both endpoints of the knob are arms that already exist and the ladder is a one-parameter
# extension rather than a new family. Absolute rather than per-row, for the same reason
# `availability_weighting.L2_GRID` is: the log-likelihood it is weighed against is a sum.
LAMBDA_GRID: tuple[float, ...] = (0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, np.inf)

# Short windows the shrinkage is applied at. 8 is `availability_weighting`'s rolling
# optimum; 5 is where PIT and boundary error were still improving in §4b.
SHRINK_LOOKBACKS: tuple[int, ...] = (8, 5)

# The blocks §5b measured as drifting — the level and the workload relationship, five of
# twenty columns. In the block arms these are left FREE (`lambda = 0`) and everything else
# is shrunk, which is the same partition its spliced arm used and the reason that arm is
# carried here as a control.
DRIFTING_BLOCKS: tuple[str, ...] = ("intercept", "workload")

SHRINK_REFERENCE = "long_only"
SPLICE_ARM = "splice8__intercept_workload"
CONTAM_REFERENCE = "clean"
REGIME_REFERENCE = "lball__none"

# Origins used by the contamination harness: non-regime, and far enough in that the fitting
# rows are a real fit. Sharing `FIRST_ORIGIN` keeps it comparable to every other harness in
# this line of work.
CONTAM_ORIGINS = tuple(range(FIRST_ORIGIN, min(REGIME_YEARS)))
# The placebo needs its whole block to sit strictly after the origin, or it would inject
# the season it is scoring.
PLACEBO_ORIGINS = tuple(o for o in CONTAM_ORIGINS if o < min(PLACEBO_YEARS))


# ── Regime terms ──────────────────────────────────────────────────────────────

def add_regime_terms(frames: list[pd.DataFrame], kind: str,
                     years: tuple[int, ...] = REGIME_YEARS
                     ) -> tuple[list[pd.DataFrame], list[str]]:
    """Regime indicators on the target season, the lag season, or both.

    Two different claims, and they are not the same instrument:

    - **`target`** says the disrupted seasons had a lower availability *level*, so the
      trough belongs in a coefficient instead of in the intercept. It is a bump where
      `season_terms`' trend is a slope, which is the shape §5.2 recorded the trend as being
      wrong about.
    - **`lag`** says the disrupted seasons produced unrepresentative *features* — a player
      whose `gp_share_lag1` came out of the bubble looks more fragile than he is. This one
      is live on validation rather than only in the fitting half: 2022-23 is predicted from
      2021-22.

    Both are known before opening night — a season's own schedule disruption is not the
    target regressed on itself — and `predict_frame` is what decides what the target
    indicator is *set to* at scoring time.
    """
    names = {"target": [REGIME_TARGET_COL], "lag": [REGIME_LAG_COL],
             "both": [REGIME_TARGET_COL, REGIME_LAG_COL]}[kind]
    out = []
    for frame in frames:
        copy = frame.copy()
        year = season_start_year(frame)
        if REGIME_TARGET_COL in names:
            copy[REGIME_TARGET_COL] = np.isin(year, years).astype(float)
        if REGIME_LAG_COL in names:
            copy[REGIME_LAG_COL] = np.isin(year - 1, years).astype(float)
        out.append(copy)
    return out, names


def predict_frame(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The scored frame with the **target** regime indicator forced to 0.

    Production predicts an ordinary season, so that is what every arm is scored as
    predicting — including at the two origins that are themselves regime seasons, where
    this makes the arm *worse* rather than better. Setting the indicator to its realized
    value there would be scoring the head on a fact it will never have for 2026-27.

    The **lag** indicator is left at its realized value, because the prior season is
    observed before the draft. That asymmetry is the whole difference between the two.
    """
    if REGIME_TARGET_COL not in names:
        return frame
    out = frame.copy()
    out[REGIME_TARGET_COL] = 0.0
    return out


def regime_weights(rows: pd.DataFrame, weight: float,
                   years: tuple[int, ...] = REGIME_YEARS) -> np.ndarray:
    """`weight` on the regime seasons, 1 elsewhere. `0.0` is exclusion, `1.0` is a no-op."""
    return np.where(np.isin(season_start_year(rows), years), float(weight), 1.0)


def fit_weighted_graded(rows: pd.DataFrame, w: np.ndarray, l2: float = 1.0,
                        features: list[str] | None = None,
                        scaler: StandardScaler | None = None) -> RoleGradedBetaBinomial:
    """A role-graded head whose mean and dispersion both carry per-row weights.

    The same construction `availability_weighting.fit_decayed` uses, with the weights
    supplied rather than derived from a decay — kept here rather than folded into that
    function so the four measured arms in §5b are not re-run by a refactor.

    Rows at weight 0 contribute nothing to either half, so a weight arm at 0.0 and a fit on
    the filtered frame differ only in the scaler, which is fitted on the unweighted rows.
    That is deliberate: standardization is a property of the design, not of the arm.
    """
    features = list(features or FEATURE_COLS)
    mean_fit = WeightedBetaBinomialGLM(l2=l2, features=features, weights=w,
                                       scaler=scaler).fit(rows)
    graded = RoleGradedBetaBinomial(l2=l2, features=features)
    graded.scaler, graded.beta = mean_fit.scaler, mean_fit.beta
    y, n = rows["gp"].to_numpy(), rows["team_games"].to_numpy()
    mu = graded.predict_mean(rows)
    graded.rho = fit_dispersion_weighted(y, n, mu, w)
    graded.pooled_rho = float(graded.rho)
    buckets = RoleGradedBetaBinomial._buckets(rows)
    graded.rho_by_role = {}
    for label in ROLE_LABELS:
        mask = (buckets == label) & (w > 0)
        if mask.sum() >= MIN_ROLE_ROWS:
            graded.rho_by_role[label] = fit_dispersion_weighted(
                y[mask], n[mask], mu[mask], w[mask])
    return graded


# ── Shrinkage toward the long-window fit ──────────────────────────────────────

class ShrunkBetaBinomialGLM(RoleGradedBetaBinomial):
    """A short-window fit under a Gaussian prior centred on the long-window coefficients.

    §5.5(c), and §5b's diagnosis of why its spliced arm failed: *"a spliced coefficient
    vector is not a fit of anything"* — the blocks are not orthogonal, so coefficients
    estimated on eight seasons are co-adapted to each other and dropping five of them into
    a vector estimated on twenty-five breaks that. Here the blocks are **fitted jointly**
    under different priors instead of transplanted:

        -loglik(short rows) + l2 * ||beta[1:]||^2 + sum_j lambda_j * (beta_j - anchor_j)^2

    Two endpoints, which is what makes this one knob rather than a new family:

    - **`lambda = inf`** pins that coefficient to the long-window estimate **exactly**, by
      dropping it from the optimization rather than by a large finite penalty.
    - **`lambda = 0`** is the plain short-window fit's *objective*, reached to the
      optimizer's tolerance rather than bit for bit: the two differ only in where L-BFGS-B
      starts, since this one warm-starts at the anchor. Measured on 3,000 rows the
      penalized objectives agree to 2.6e-4 out of 12,195 while individual coefficients move
      by up to 2.4e-3 — the lag block is collinear by construction, so the surface is flat
      in exactly that direction and the gap is the flat direction rather than a second
      optimum. `tests/test_availability_regime.py` pins the objective, not the vector,
      because the objective is the thing the claim is about.

    And the second endpoint is what separates this from the splice even at its own limit.
    With the stable blocks pinned and the drifting blocks free, the drifting blocks are
    **re-estimated in the presence of** the pinned ones; the spliced arm imported them from
    a fit where the other fifteen columns were different numbers. Same coefficient
    partition, different estimator, and §5b says the estimator is the whole complaint.

    The anchor is only meaningful in the coordinates it was estimated in, so the long
    window's `scaler` is supplied rather than refitted — the discipline `block_window`
    already carries.

    **The penalty reaches the intercept**, where `l2` does not. Shrinking an intercept
    toward zero is meaningless and shrinking one toward another fit's intercept is not, and
    §5b measured the intercept as the single largest drifting block, so leaving it out
    would exempt the coefficient the axis is most about.
    """

    name = "beta_binomial_shrunk"

    def __init__(self, anchor: np.ndarray, lam: np.ndarray, scaler: StandardScaler,
                 l2: float = 1.0, features: list[str] | None = None,
                 max_rounds: int = 25):
        super().__init__(l2=l2, features=features, max_rounds=max_rounds)
        self.anchor = np.asarray(anchor, dtype=float)
        self.lam = np.asarray(lam, dtype=float)
        self.fixed_scaler = scaler

    def fit(self, train: pd.DataFrame) -> "ShrunkBetaBinomialGLM":
        self.scaler = self.fixed_scaler
        X = self._design(train)
        y = train["gp"].to_numpy()
        n = train["team_games"].to_numpy()
        ones = np.ones(len(train))

        pinned = ~np.isfinite(self.lam)
        free = ~pinned
        lam_free = self.lam[free]
        anchor = self.anchor
        ridge = np.zeros(X.shape[1])
        ridge[1:] = self.l2

        def expand(theta: np.ndarray) -> np.ndarray:
            beta = anchor.copy()
            beta[free] = theta
            return beta

        def objective(theta: np.ndarray, rho: float) -> tuple[float, np.ndarray]:
            beta = expand(theta)
            mu = _sigmoid(X @ beta)
            gap = beta - anchor
            value = (weighted_neg_loglik(y, n, mu, rho, ones)
                     + float(ridge @ (beta * beta))
                     + float(self.lam[free] @ (gap[free] * gap[free])))
            a, b = _ab(mu, rho)
            r = float(np.clip(rho, RHO_MIN, RHO_MAX))
            scale = (1.0 - r) / r
            dmu = scale * (digamma(y + a) - digamma(a) - digamma(n - y + b) + digamma(b))
            grad = -(X.T @ (dmu * mu * (1.0 - mu))) + 2.0 * ridge * beta
            grad = grad[free] + 2.0 * lam_free * gap[free]
            return value, np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        theta = anchor[free].copy()
        rho = 0.23
        best = None
        for _ in range(self.max_rounds):
            best = minimize(objective, theta, args=(rho,), jac=True, method="L-BFGS-B")
            theta = best.x
            new_rho = fit_dispersion_weighted(y, n, _sigmoid(X @ expand(theta)), ones)
            if abs(new_rho - rho) < 1e-5:
                rho = new_rho
                break
            rho = new_rho

        self.beta, self.rho = expand(theta), float(rho)
        self.converged = bool(best.success)
        self.n_pinned = int(pinned.sum())
        # The dispersion goes on the SHORT rows in every shrinkage arm, including the one
        # that pins every coefficient. That is what makes the ladder a contrast in the mean
        # function alone — `availability_weighting` already measured `rho`'s own window and
        # found the two effects additive but the second one worth nothing once `beta` is
        # windowed, so re-crossing it here would be re-running a null.
        self.pooled_rho = float(self.rho)
        mu = self.predict_mean(train)
        buckets = self._buckets(train)
        self.rho_by_role = {}
        for label in ROLE_LABELS:
            mask = buckets == label
            if mask.sum() >= MIN_ROLE_ROWS:
                self.rho_by_role[label] = fit_dispersion_weighted(
                    y[mask], n[mask], mu[mask], ones[mask])
        return self


def lambda_vector(features: list[str], value: float,
                  free_blocks: tuple[str, ...] = ()) -> np.ndarray:
    """A per-column prior precision: `value` everywhere, 0 on the blocks left free."""
    lam = np.full(len(features) + 1, float(value))
    if free_blocks:
        idx = block_indices(list(features))
        for block in free_blocks:
            lam[idx[block]] = 0.0
    return lam


def fit_shrunk(long_rows: pd.DataFrame, short_rows: pd.DataFrame, value: float,
               free_blocks: tuple[str, ...] = (), l2: float = 1.0,
               features: list[str] | None = None,
               anchor: RoleGradedBetaBinomial | None = None) -> ShrunkBetaBinomialGLM:
    """The long-window fit as a prior, the short-window rows as the likelihood."""
    features = list(features or FEATURE_COLS)
    anchor = anchor or RoleGradedBetaBinomial(l2=l2, features=features).fit(long_rows)
    lam = lambda_vector(features, value, free_blocks)
    return ShrunkBetaBinomialGLM(anchor=anchor.beta, lam=lam, scaler=anchor.scaler,
                                 l2=l2, features=features).fit(short_rows)


# ── Pooling helpers ───────────────────────────────────────────────────────────

def _fit_rows(train: pd.DataFrame, year: np.ndarray, origin: int,
              lookback: int | None) -> pd.DataFrame:
    floor = -np.inf if lookback is None else origin - lookback
    return train[(year < origin) & (year >= floor)]


def _enough(rows: pd.DataFrame) -> bool:
    return len(rows) >= MIN_ROLE_ROWS * len(ROLE_LABELS)


def feasible_lookbacks(train: pd.DataFrame, origins: list[int],
                       lookbacks: tuple[int | None, ...],
                       arms: dict[str, dict] | None = None) -> tuple[int | None, ...]:
    """Lookbacks whose every (origin, arm) cell clears the row guard.

    A pooled paired bootstrap requires every arm to have scored the **same rows**, so an
    arm that silently skips one origin is not comparable to one that did not — it is a
    different sample wearing the same reference. Rather than pool ragged vectors, a
    lookback that any arm cannot fit anywhere is dropped whole and the drop is printed.

    It bites in exactly one place, and it is the place the axis is about: at the last
    origin a three-season lookback that also excludes the regime block is one season of
    rows.
    """
    year = season_start_year(train)
    keep = []
    for lookback in lookbacks:
        ok = True
        for origin in origins:
            rows = _fit_rows(train, year, origin, lookback)
            ok = ok and _enough(rows)
            for spec in (arms or {}).values():
                if spec.get("kind") == "weight" and spec.get("weight") == 0.0:
                    ok = ok and _enough(rows[regime_weights(rows, 0.0, spec["years"]) > 0])
        if ok:
            keep.append(lookback)
    return tuple(keep)


def _with_extras(table: pd.DataFrame, per_arm: dict, extras: dict[str, dict]
                 ) -> pd.DataFrame:
    """Join per-arm scalars that `_pool` has no slot for.

    `_pool` reads `meta` from the FIRST origin only, which is the right thing for a knob
    that is constant across origins and the wrong thing for a count that accumulates over
    them — `origins_active` being the one this round needs, since it is the whole statement
    of what the walk-forward harness can and cannot see.

    The predictive **share bias** is joined the same way. It is the diagnostic the
    contamination harness is actually about: a fitting set that contains the trough should
    push the predicted availability level *down*, and a CRPS column reports whether that
    cost anything without ever saying whether it happened.
    """
    for column, values in extras.items():
        table[column] = table["arm"].map(values)
    return table


def _within_lookback(per_arm: dict, seed: int) -> dict[str, dict[str, float]]:
    """Each arm paired against the incumbent **at its own lookback**.

    `_pool` takes one reference for the whole experiment, which for a crossed grid means
    every cell's delta carries the lookback's effect as well as the arm's. That is exactly
    the confound §5b's `l2_by_lookback` was built to remove one axis over, so the crossed
    table gets both columns: `crps_vs_reference` against `lball__none` for comparability
    with `availability_weighting.csv`, and this one for reading the regime axis alone.
    """
    from src.models.availability_weighting import paired_bootstrap
    out: dict[str, dict[str, float]] = {}
    for name, d in per_arm.items():
        lb = name.split("__")[0]
        base = per_arm.get(f"{lb}__none")
        if base is None:
            continue
        arm = np.concatenate(d["crps"])
        reference = np.concatenate(base["crps"])
        mean, lo, hi = paired_bootstrap(arm, reference, seed=seed)
        out[name] = {"crps_vs_lookback_none": mean,
                     "lookback_none_lo": lo, "lookback_none_hi": hi}
    return out


def _share_bias(per_arm: dict) -> dict[str, float]:
    return {name: float(np.concatenate(d["pred_share"]).mean()
                        - np.concatenate(d["obs_share"]).mean())
            for name, d in per_arm.items()}


def _collect_shares(per_arm: dict, name: str, model, score: pd.DataFrame) -> None:
    slot = per_arm[name]
    slot.setdefault("pred_share", []).append(model.predict_mean(score))
    slot.setdefault("obs_share", []).append(
        score["gp"].to_numpy() / score["team_games"].to_numpy())


# ── Experiment A: the cross §5.7 asks for, and the blindness it measures ──────

def _regime_arms(l2: float) -> dict[str, dict]:
    """`{name: spec}` for the regime axis. One place, so every harness runs the same arms."""
    arms: dict[str, dict] = {}
    for w in REGIME_WEIGHTS:
        label = "none" if w == 1.0 else ("exclude" if w == 0.0 else f"w{w:.2f}")
        arms[label] = {"kind": "weight", "weight": w, "years": REGIME_YEARS}
    for kind in ("target", "lag", "both"):
        arms[f"dummy_{kind}"] = {"kind": "dummy", "dummy": kind, "years": REGIME_YEARS}
    arms["core_exclude"] = {"kind": "weight", "weight": 0.0, "years": REGIME_CORE}
    arms["core_dummy_target"] = {"kind": "dummy", "dummy": "target", "years": REGIME_CORE}
    for spec in arms.values():
        spec["l2"] = l2
    return arms


def build_regime_arm(spec: dict, fit_rows: pd.DataFrame, score: pd.DataFrame,
                     features: list[str] | None = None):
    """`(model, scored frame, features, active)` for one regime arm on one fitting set.

    `active` is whether the arm's fitting frame differed from the incumbent's *at all* —
    the count that makes the walk-forward harness's blindness a measurement.
    """
    features = list(features or FEATURE_COLS)
    l2, years = spec["l2"], spec["years"]
    in_regime = np.isin(season_start_year(fit_rows), years)
    if spec["kind"] == "weight":
        w = regime_weights(fit_rows, spec["weight"], years)
        model = fit_weighted_graded(fit_rows, w, l2=l2, features=features)
        return model, score, features, bool(in_regime.any() and spec["weight"] != 1.0)
    (tr, sc), names = add_regime_terms([fit_rows, score], spec["dummy"], years)
    cols = features + names
    model = RoleGradedBetaBinomial(l2=l2, features=cols).fit(tr)
    active = bool(np.any(tr[names].to_numpy(dtype=float) != 0.0))
    return model, predict_frame(sc, names), cols, active


def experiment_regime_x_lookback(train, max_games, origins, seed, l2) -> pd.DataFrame:
    """Regime handling crossed with lookback, on the plain walk-forward harness.

    §5.7 asks for the cross explicitly — "a separate axis from lookback and should be swept
    as one" — and this is it. Read `origins_active` before reading anything else in the
    table: an arm whose fitting rows never contained a regime season is the incumbent under
    a different name, and it will post the incumbent's CRPS to sixteen decimals.
    """
    year = season_start_year(train)
    specs = _regime_arms(l2)
    lookbacks = feasible_lookbacks(train, origins, LOOKBACK_GRID, specs)
    dropped = [lb for lb in LOOKBACK_GRID if lb not in lookbacks]
    if dropped:
        print(f"  dropped lookbacks {dropped} — some arm cannot clear "
              f"{MIN_ROLE_ROWS * len(ROLE_LABELS)} rows at every origin.")
    per_arm: dict = {}
    active: dict[str, int] = {}
    for origin in origins:
        score = train[year == origin]
        for lookback in lookbacks:
            rows = _fit_rows(train, year, origin, lookback)
            lb = "all" if lookback is None else str(lookback)
            for label, spec in specs.items():
                name = f"lb{lb}__{label}"
                model, sc, cols, is_active = build_regime_arm(spec, rows, score)
                _collect(per_arm, name, model, sc, max_games, origin,
                         {"lookback": lb, "regime_arm": label,
                          "n_fit": len(rows), "n_regime_rows": int(
                              np.isin(season_start_year(rows), spec["years"]).sum())})
                _collect_shares(per_arm, name, model, sc)
                active[name] = active.get(name, 0) + int(is_active)
    table = _pool(per_arm, REGIME_REFERENCE, "regime_x_lookback", seed)
    within = _within_lookback(per_arm, seed)
    extras = {"origins_active": active, "share_bias": _share_bias(per_arm)}
    for column in ("crps_vs_lookback_none", "lookback_none_lo", "lookback_none_hi"):
        extras[column] = {name: d[column] for name, d in within.items()}
    return _with_extras(table, per_arm, extras)


# ── Experiment B: the contamination harness, which can see the axis ───────────

def experiment_regime_contamination(train, max_games, seed, l2,
                                    inject_years: tuple[int, ...] = REGIME_YEARS,
                                    experiment: str = "regime_contamination"
                                    ) -> pd.DataFrame:
    """Stage the production situation: trough in the fitting rows, ordinary target season.

    Ten replicates of a situation the walk-forward harness has zero of. Every arm is fitted
    on `seasons < origin` **plus** the injected block, and scored on the origin season; the
    reference is the same fit without the block.

    The injected block is in the origin's future, so this is an **ablation and not a
    forecast** — no row here is a walk-forward score and none of it may be quoted as one.
    What it can say is whether an instrument that keeps the contaminated rows recovers the
    fit you would have had without them, which is exactly §5.7's question and exactly what
    the walk-forward design cannot reach.
    """
    year = season_start_year(train)
    # Every arm is defined against the block that was actually injected, so B2 is the same
    # ladder with a different regime definition rather than a second set of arms. `exclude`
    # is absent because excluding the injected block IS `clean`, exactly — the question here
    # is only whether an instrument that KEEPS the rows recovers what exclusion gets free.
    specs = {k: {**v, "years": inject_years}
             for k, v in _regime_arms(l2).items()
             if k not in ("exclude",) and not k.startswith("core_")}
    inject = train[np.isin(year, inject_years)]
    per_arm: dict = {}
    for origin in CONTAM_ORIGINS:
        score = train[year == origin]
        base = train[year < origin]
        if score.empty or not _enough(base):
            continue
        clean = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS)).fit(base)
        _collect(per_arm, CONTAM_REFERENCE, clean, score, max_games, origin,
                 {"regime_arm": "clean", "n_fit": len(base), "n_injected": 0})
        _collect_shares(per_arm, CONTAM_REFERENCE, clean, score)
        rows = pd.concat([base, inject], ignore_index=True)
        for label, spec in specs.items():
            name = "contaminated" if label == "none" else f"contaminated__{label}"
            model, sc, cols, _ = build_regime_arm(spec, rows, score)
            _collect(per_arm, name, model, sc, max_games, origin,
                     {"regime_arm": label, "n_fit": len(rows), "n_injected": len(inject)})
            _collect_shares(per_arm, name, model, sc)
        print(f"  origin {origin}: {len(base):,} clean / +{len(inject):,} injected "
              f"/ {len(score):,} scored")
    table = _pool(per_arm, CONTAM_REFERENCE, experiment, seed)
    return _with_extras(table, per_arm, {"share_bias": _share_bias(per_arm)})


def experiment_regime_placebo(train, max_games, seed, l2) -> pd.DataFrame:
    """The control for B: inject an ORDINARY block of the same size, out of order.

    `contaminated` differs from `clean` in two ways at once — the injected rows are
    unrepresentative, and they are anachronistic and add ~1,300 rows. This arm holds the
    second constant. If the placebo moves CRPS as much as the regime block does, the
    contamination harness is measuring row count and season order rather than the trough,
    and axis (1) has no result here.
    """
    year = season_start_year(train)
    regime = train[np.isin(year, REGIME_YEARS)]
    placebo = train[np.isin(year, PLACEBO_YEARS)]
    per_arm: dict = {}
    for origin in PLACEBO_ORIGINS:
        score = train[year == origin]
        base = train[year < origin]
        if score.empty or not _enough(base):
            continue
        for name, block in (("clean", None), ("inject_regime", regime),
                            ("inject_placebo", placebo)):
            rows = base if block is None else pd.concat([base, block], ignore_index=True)
            model = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS)).fit(rows)
            _collect(per_arm, name, model, score, max_games, origin,
                     {"n_fit": len(rows),
                      "n_injected": 0 if block is None else len(block)})
            _collect_shares(per_arm, name, model, score)
    table = _pool(per_arm, "clean", "regime_placebo", seed)
    # The control's own contrast, which is what the claim actually is: not "the regime
    # block hurts and the placebo does not", two statements against a third arm, but
    # "regime minus placebo", one statement with an interval on it.
    direct = _pool(per_arm, "inject_placebo", "regime_placebo_vs_placebo", seed)
    out = pd.concat([_with_extras(table, per_arm, {"share_bias": _share_bias(per_arm)}),
                     _with_extras(direct, per_arm, {"share_bias": _share_bias(per_arm)})],
                    ignore_index=True)
    return out


# ── Experiment C: shrinkage toward the long-window fit ────────────────────────

def experiment_shrinkage(train, max_games, origins, seed, l2) -> pd.DataFrame:
    """§5.5(c). One knob from "keep every season" to "keep the last eight", per block.

    Four families of arm, and the first two are the endpoints of the third:

    - `long_only` — every season, the incumbent, and the reference.
    - `shortN__only` — the plain N-season window, its own scaler and its own dispersion.
    - `shrinkN__lam*` — every coefficient shrunk toward the long-window estimate.
    - `shrinkN__free_drift__lam*` — §5b's five drifting columns free, the other fifteen
      shrunk. At `lam = inf` this is its spliced arm's coefficient partition fitted
      jointly instead of transplanted, which is the comparison the axis exists for.

    `splice8__intercept_workload` is carried as a control — §5b's arm, rebuilt, so the
    shrinkage rows are read against the thing they claim to be the principled version of
    rather than against a remembered figure.
    """
    year = season_start_year(train)
    idx = block_indices(list(FEATURE_COLS))
    drift_cols = np.concatenate([idx[b] for b in DRIFTING_BLOCKS])
    # Every arm has to cover every origin or the pooled pairing is comparing two samples,
    # so feasibility is settled once up front rather than by a `continue` inside the loop.
    lookbacks = feasible_lookbacks(train, origins, SHRINK_LOOKBACKS, arms={})
    if set(lookbacks) != set(SHRINK_LOOKBACKS):
        print(f"  dropped short lookbacks {sorted(set(SHRINK_LOOKBACKS) - set(lookbacks))}")
    per_arm: dict = {}
    for origin in origins:
        score = train[year == origin]
        long_rows = _fit_rows(train, year, origin, None)
        if score.empty or not _enough(long_rows):
            continue
        anchor = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS)).fit(long_rows)
        _collect(per_arm, SHRINK_REFERENCE, anchor, score, max_games, origin,
                 {"short_lookback": "all", "lam": np.nan, "free_drift": False,
                  "n_short": len(long_rows)})
        for lookback in SHRINK_LOOKBACKS:
            short_rows = _fit_rows(train, year, origin, lookback)
            if not _enough(short_rows):
                continue
            plain = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS)).fit(short_rows)
            _collect(per_arm, f"short{lookback}__only", plain, score, max_games, origin,
                     {"short_lookback": lookback, "lam": np.nan, "free_drift": False,
                      "n_short": len(short_rows)})
            for free in ((), DRIFTING_BLOCKS):
                tag = "free_drift__" if free else ""
                for value in LAMBDA_GRID:
                    model = fit_shrunk(long_rows, short_rows, value, free_blocks=free,
                                       l2=l2, anchor=anchor)
                    _collect(per_arm,
                             f"shrink{lookback}__{tag}lam{value:g}", model, score,
                             max_games, origin,
                             {"short_lookback": lookback, "lam": value,
                              "free_drift": bool(free), "n_short": len(short_rows)})
        # §5b's spliced arm, rebuilt at its own lookback so the control is the arm that
        # failed rather than a description of it.
        splice_rows = _fit_rows(train, year, origin, 8)
        if _enough(splice_rows):
            short = WeightedBetaBinomialGLM(l2=l2, features=list(FEATURE_COLS),
                                            scaler=anchor.scaler).fit(splice_rows)
            spliced = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS))
            spliced.scaler = anchor.scaler
            beta = anchor.beta.copy()
            beta[drift_cols] = short.beta[drift_cols]
            spliced.beta = beta
            y, n = splice_rows["gp"].to_numpy(), splice_rows["team_games"].to_numpy()
            mu = spliced.predict_mean(splice_rows)
            ones = np.ones(len(splice_rows))
            spliced.rho = fit_dispersion_weighted(y, n, mu, ones)
            spliced.pooled_rho = float(spliced.rho)
            buckets = RoleGradedBetaBinomial._buckets(splice_rows)
            spliced.rho_by_role = {
                label: fit_dispersion_weighted(y[buckets == label], n[buckets == label],
                                               mu[buckets == label], ones[buckets == label])
                for label in ROLE_LABELS if (buckets == label).sum() >= MIN_ROLE_ROWS}
            _collect(per_arm, SPLICE_ARM, spliced, score, max_games,
                     origin, {"short_lookback": 8, "lam": np.nan, "free_drift": True,
                              "n_short": len(splice_rows)})
        print(f"  origin {origin}: {len(long_rows):,} long rows")
    table = _pool(per_arm, SHRINK_REFERENCE, "shrinkage", seed)
    # A second pooling against §5b's spliced arm, because "shrinkage is the principled
    # version of the splice" is a claim about the gap between those two specifically and
    # `long_only` cannot carry an interval on it. The matched pair —
    # `shrink8__free_drift__laminf` against `splice8__intercept_workload`, same coefficient
    # partition, same two windows, joint fit against transplant — is the one row that tests
    # the mechanism rather than the recipe.
    versus = _pool(per_arm, SPLICE_ARM, "shrinkage_vs_splice", seed)
    return pd.concat([table, versus], ignore_index=True)


# ── The one validation reading ────────────────────────────────────────────────

def _shipped_window(train: pd.DataFrame) -> pd.DataFrame:
    return restrict_window(train, WINDOWS[LIKELIHOOD_WINDOW])


def confirm_on_validation(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
                          seed: int, l2: float, best_global: tuple[int, float],
                          best_free: tuple[int, float], best_regime: str,
                          mixture: bool = True) -> pd.DataFrame:
    """The one validation reading, on both likelihoods, after the recipes are fixed.

    Every regime arm sits **on top of the shipped window** (`three_point_era`, 2012-13+),
    because that is the head's fitting rule and the axis is a question about which of those
    rows to trust rather than a replacement for the window. The shrinkage arms use the full
    fitting half as the long window and the shipped window as the short one, which is the
    production configuration of §5.5(c): keep every season as a prior, fit on the recent
    ones.

    `mixture` runs the shipped two-component likelihood on the reference and on whatever
    the rolling harness selected. It is ~40 s a fit against the point MLE's 0.1 s, which is
    why it is three arms and not thirty — and why selection never touched it.
    """
    shipped = _shipped_window(train)
    rows: list[dict] = []
    per_row: dict[str, np.ndarray] = {}

    def add(name: str, model, fit_rows, score_rows, cols, kind: str) -> None:
        row, scores = score_arm(name, model, fit_rows, score_rows, cols, max_games, seed)
        row["experiment"] = "validation_confirmation"
        row["likelihood"] = kind
        rows.append(row)
        per_row[name] = scores

    # ── the point-MLE arms ────────────────────────────────────────────────────
    reference = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS)).fit(shipped)
    add("shipped__2012_role_rho", reference, shipped, val, list(FEATURE_COLS), "betabinom")

    for label, spec in _regime_arms(l2).items():
        if label == "none":
            continue
        model, sc, cols, _ = build_regime_arm(spec, shipped, val)
        fit_rows = shipped[regime_weights(shipped, 0.0, spec["years"]) > 0] \
            if spec["kind"] == "weight" and spec["weight"] == 0.0 else shipped
        add(f"regime__{label}", model, fit_rows, sc, cols, "betabinom")

    # Each shrinkage family is confirmed twice: at the **shipped window** as the short
    # rows, which is the production configuration of §5.5(c), and at the short lookback the
    # rolling harness actually selected, so the confirmation is of the selected recipe and
    # not only of a nearby one. They are different arms and both are quoted.
    year = season_start_year(train)
    for (lookback, value), free, tag in (((best_global), (), "global"),
                                         ((best_free), DRIFTING_BLOCKS, "free_drift")):
        model = fit_shrunk(train, shipped, value, free_blocks=free, l2=l2)
        add(f"shrink__{tag}__2012__lam{value:g}", model, shipped, val,
            list(FEATURE_COLS), "betabinom")
        short = train[year > year.max() - lookback]
        model = fit_shrunk(train, short, value, free_blocks=free, l2=l2)
        add(f"shrink__{tag}__lb{lookback}__lam{value:g}", model, short, val,
            list(FEATURE_COLS), "betabinom")

    # ── the shipped likelihood, on the reference and the two challengers ──────
    if mixture:
        add("mixture__shipped_2012", MixtureFrailty(l2=l2, features=list(FEATURE_COLS))
            .fit(shipped), shipped, val, list(FEATURE_COLS), "mixture")
        spec = _regime_arms(l2)[best_regime]
        if spec["kind"] == "dummy":
            (tr, va), names = add_regime_terms([shipped, val], spec["dummy"], spec["years"])
            cols, va = list(FEATURE_COLS) + names, predict_frame(va, names)
        else:
            tr = shipped[regime_weights(shipped, spec["weight"], spec["years"]) > 0] \
                if spec["weight"] == 0.0 else shipped
            cols, va = list(FEATURE_COLS), val
        add(f"mixture__regime_{best_regime}",
            MixtureFrailty(l2=l2, features=cols).fit(tr), tr, va, cols, "mixture")
        # Exclusion is carried whatever the harness selected, because it is the arm §5.7
        # names first and the one that needs no new parameter — a null on it is the
        # cheapest possible finding and the most likely to be reached for again.
        if best_regime != "exclude":
            excluded = shipped[regime_weights(shipped, 0.0, REGIME_YEARS) > 0]
            add("mixture__regime_exclude",
                MixtureFrailty(l2=l2, features=list(FEATURE_COLS)).fit(excluded), excluded,
                val, list(FEATURE_COLS), "mixture")
    out = pd.DataFrame(rows)
    for column, value in regime_identification(shipped, val).items():
        out[column] = value
    # Paired **within likelihood**, against that likelihood's own shipped arm. Across the
    # two it would be measuring the mixture, which §7 already settled — the question here
    # is whether a recipe survives the head it would be applied to, and a beta-binomial
    # challenger scored against the mixture's CRPS could not answer it either way.
    from src.models.availability_weighting import paired_bootstrap
    references = {"betabinom": "shipped__2012_role_rho",
                  "mixture": "mixture__shipped_2012"}
    stats = []
    for _, row in out.iterrows():
        reference = per_row.get(references.get(row["likelihood"], ""))
        if reference is None:
            stats.append((np.nan, np.nan, np.nan))
            continue
        stats.append(paired_bootstrap(per_row[row["arm"]], reference, seed=seed))
    out[["crps_vs_shipped", "crps_vs_shipped_lo", "crps_vs_shipped_hi"]] = stats
    out["beats_shipped"] = out["crps_vs_shipped_hi"] < 0.0
    return out


def regime_identification(fit_rows: pd.DataFrame, val: pd.DataFrame) -> dict[str, float]:
    """Whether the target and lag indicators are separable in the head's own fitting rows.

    They are not, and it is structural rather than a small-sample accident: the disrupted
    seasons are **consecutive**, so inside the shipped window `regime_lag = 1` implies
    `regime_target = 1` on every row — the lag flag marks target seasons 2020-21 and
    2021-22, which are the two deepest seasons in the trough. Fitted there and applied to
    2022-23, where the lag is set but the target is not, it extrapolates a trough level
    onto a season that partly recovered.

    Carried on the confirmation table rather than left as a remark, because the lag arm's
    validation reading is the one figure in this round most likely to be misread as a
    feature that failed instead of a coefficient that was never identified.
    """
    (tr, va), names = add_regime_terms([fit_rows, val], "both")
    target, lag = tr[REGIME_TARGET_COL].to_numpy(), tr[REGIME_LAG_COL].to_numpy()
    return {"n_regime_target_train": float(target.sum()),
            "n_regime_lag_train": float(lag.sum()),
            "lag_implies_target_train": float(target[lag == 1].mean()) if lag.any() else 0.0,
            "n_regime_lag_val": float(va[REGIME_LAG_COL].sum()),
            "n_regime_target_val": float(va[REGIME_TARGET_COL].sum())}


def _best_regime(contam: pd.DataFrame) -> str:
    """The instrument that best recovers `clean` on the contamination harness.

    Ranked on CRPS among the arms that actually keep the contaminated rows, so `clean`
    itself — which is the reference and is exclusion by construction — cannot be
    "selected" back into a recipe that has nothing to do.
    """
    sub = contam[contam["arm"].str.startswith("contaminated__")]
    if sub.empty:
        return "dummy_target"
    return str(sub.sort_values("crps").iloc[0]["arm"]).removeprefix("contaminated__")


def _best_shrink(shrink: pd.DataFrame, free_drift: bool) -> tuple[int, float]:
    """`(short lookback, lambda)` of the family's best rolling arm."""
    shrink = shrink[shrink["experiment"] == "shrinkage"]
    sub = shrink[(shrink["free_drift"] == free_drift) & shrink["lam"].notna()]
    if sub.empty:
        return (8, 0.0)
    best = sub.sort_values("crps").iloc[0]
    return int(best["short_lookback"]), float(best["lam"])


def run(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    cfg_av = cfg.get("features", {}).get("availability", {})
    seed = int(cfg_av.get("seed", 42))
    l2 = float(cfg_av.get("glm_l2", 1.0))

    panel = build_panel(seasons, raw_dir)
    design = build_design(season_availability(panel, "full"), seasons, raw_dir,
                          season_start_dates(panel))
    train, val = selection_split(design)
    max_games = int(design["team_games"].max())
    years = np.asarray(sorted(np.unique(season_start_year(train))))
    origins = [int(y) for y in years if y >= FIRST_ORIGIN]

    print(f"Availability regime + shrinkage: {len(train):,} fitting-half rows, "
          f"{len(origins)} walk-forward origins ({origins[0]}–{origins[-1]}).")
    print(f"  Regime block: {REGIME_YEARS} (core {REGIME_CORE}); "
          f"placebo block {PLACEBO_YEARS}.")
    print("  Validation is NOT read until the confirmation at the end.")

    print("\n── A. regime x lookback, plain walk-forward (read `origins_active` first) ──")
    regime = experiment_regime_x_lookback(train, max_games, origins, seed, l2)
    blind = regime[regime["origins_active"] == 0]
    print(f"  {len(blind):,} of {len(regime):,} arms are the incumbent refitted — their "
          f"fitting rows never\n  contained a regime season, so this harness cannot see "
          f"them at all.")
    best = regime[regime["origins_active"] > 0].sort_values("crps")
    if not best.empty:
        r = best.iloc[0]
        print(f"  best ACTIVE arm: {r['arm']}  CRPS {r['crps']:.4f} "
              f"({r['crps_vs_reference']:+.4f} vs reference, active at "
              f"{int(r['origins_active'])} origins)")

    print("\n── B. contamination: the trough injected into a non-regime origin ──")
    contam = experiment_regime_contamination(train, max_games, seed, l2)
    for _, r in contam.iterrows():
        print(f"  {r['arm']:<34} CRPS {r['crps']:.4f} ({r['crps_vs_reference']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}])  share bias {r['share_bias']:+.4f}")

    print("\n── B2. the same, with the CORE block (2020-21, 2021-22) only ──")
    contam_core = experiment_regime_contamination(
        train, max_games, seed, l2, inject_years=REGIME_CORE,
        experiment="regime_contamination_core")
    for _, r in contam_core.iterrows():
        print(f"  {r['arm']:<34} CRPS {r['crps']:.4f} ({r['crps_vs_reference']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}])  share bias {r['share_bias']:+.4f}")

    print("\n── C. the placebo control: an ordinary block, same size, out of order ──")
    placebo = experiment_regime_placebo(train, max_games, seed, l2)
    for _, r in placebo.iterrows():
        print(f"  {r['arm']:<34} CRPS {r['crps']:.4f} ({r['crps_vs_reference']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}])  share bias {r['share_bias']:+.4f}")

    regime_out = pd.concat([regime, contam, contam_core, placebo], ignore_index=True)
    regime_dest = out_dir / "availability_regime.csv"
    regime_out.to_csv(regime_dest, index=False)
    print(f"\nWrote {len(regime_out):,} arms across 4 experiments → {regime_dest}")

    print("\n── D. shrinkage toward the long-window fit ──")
    shrink = experiment_shrinkage(train, max_games, origins, seed, l2)
    for _, r in shrink[shrink["experiment"] == "shrinkage"].head(8).iterrows():
        print(f"  {r['arm']:<34} CRPS {r['crps']:.4f} ({r['crps_vs_reference']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}])  {int(r['origins_won'])}/"
              f"{int(r['n_origins'])} origins")
    shrink_dest = out_dir / "availability_shrinkage.csv"
    shrink.to_csv(shrink_dest, index=False)
    print(f"\nWrote {len(shrink):,} arms → {shrink_dest}")

    best_global = _best_shrink(shrink, free_drift=False)
    best_free = _best_shrink(shrink, free_drift=True)
    # Selected on the contamination harness, which is the only instrument here that can
    # see the axis at all — `regime_x_lookback` is blind at most of its own cells.
    best_regime = _best_regime(contam)
    print(f"\nRolling optima: global (lookback {best_global[0]}, lambda {best_global[1]:g}), "
          f"free-drift (lookback {best_free[0]}, lambda {best_free[1]:g}), "
          f"regime arm {best_regime}")

    print("\n── the ONE validation reading, on both likelihoods ──")
    confirm = confirm_on_validation(train, val, max_games, seed, l2, best_global, best_free,
                                    best_regime)
    for _, r in confirm.iterrows():
        print(f"  [{r['likelihood']:<9}] {r['arm']:<34} CRPS {r['val_crps']:.4f} "
              f"({r['crps_vs_shipped']:+.4f} [{r['crps_vs_shipped_lo']:+.4f}, "
              f"{r['crps_vs_shipped_hi']:+.4f}])  PIT {r['val_pit_ks']:.4f}  "
              f"boundary {r['boundary_tail_error']:.4f}")
    confirm_dest = out_dir / "availability_regime_confirmation.csv"
    confirm.to_csv(confirm_dest, index=False)
    print(f"\nWrote {len(confirm):,} arms → {confirm_dest}")

    return {"availability_regime": regime_dest,
            "availability_shrinkage": shrink_dest,
            "availability_regime_confirmation": confirm_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
