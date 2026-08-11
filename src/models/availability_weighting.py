"""Four ways to spend old seasons on the availability head, scored on the rolling harness.

`make availability-weighting` → `outputs/predictions/availability_weighting.csv`, one row
per (experiment, arm), with an `experiment` column.

## The question this answers

`docs/availability-window-plan.md` §4b found the thing worth optimizing: over 13 rolling
origins, **CRPS is optimal at a lookback of 8 seasons while PIT KS and boundary error keep
improving monotonically down to 3**. The mean function wants rows; the dispersion and the
shape want *recent* rows. A single truncation point is being asked to serve two estimands
whose optima point in opposite directions, and every arm in that table pays for one with the
other.

None of the four experiments below is a new model. They are four ways of *spending* the old
seasons rather than keeping or discarding them wholesale.

| experiment | knob | what it would show |
|---|---|---|
| `l2_by_lookback` | `l2` × lookback | whether the interior optimum at 8 is real or a penalty artifact |
| `split_window` | separate `beta` and `rho` lookbacks | whether decoupling beats any single window |
| `decay` | `w = lambda^(target - s)` | whether a smooth weight beats a hard edge |
| `block_window` | which coefficients get the short window | *where* the drift lives |

Everything is scored on the **fitting half only**, walking an origin the way
`availability_window.rolling_confirmation` does, so the 883 validation rows stay unspent and
remain available as a confirmation of whatever this selects.

## The confound `l2_by_lookback` exists to remove, and its direction

`BetaBinomialGLM`'s objective is `-loglik + l2 * ||beta||^2` where the log-likelihood is a
**sum** over rows, not a mean. So the likelihood term scales with N and the penalty does not:
holding `l2` fixed while shortening the window makes the penalty relatively **stronger**, not
weaker. A 1,155-row fit at `l2 = 1.0` is carrying roughly 5.7x the shrinkage of a 6,630-row
fit at the same nominal value. The turnaround at lookbacks 5 and 3 is therefore confounded
with over-regularization, and the honest reading of the 8-season optimum has to come from a
grid that sweeps both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize, minimize_scalar
from scipy.special import digamma
from scipy.stats import betabinom
from sklearn.preprocessing import StandardScaler

from src.eda.season_effects import ROLE_LABELS
from src.features.availability import build_panel, season_availability
from src.models.availability import (EPS, FEATURE_COLS, RHO_MAX, RHO_MIN, WORKLOAD_COLS,
                                     BetaBinomialGLM, _ab, _sigmoid, build_design,
                                     season_start_dates)
from src.models.availability_window import (FIRST_ORIGIN, MIN_ROLE_ROWS,
                                            RoleGradedBetaBinomial, _origin_scores,
                                            paired_bootstrap)
from src.models.held_out import selection_split
from src.models.season_terms import season_start_year

_BIG = 1e12

# Absolute L2 values swept against the summed log-likelihood. Spanning two orders of
# magnitude either side of the incumbent's 1.0, because the confound above means the
# *effective* shrinkage at a 3-season lookback is already several times the nominal figure.
L2_GRID = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)

LOOKBACK_GRID: tuple[int | None, ...] = (None, 12, 8, 5, 3)

# Geometric decay per season of age. 1.0 is uniform weighting and reproduces the `all`
# lookback EXACTLY, which is what makes this a generalization rather than an alternative.
# The effective lookback of a decay is ~1/(1 - lambda), so 0.85 ~ 6.7 seasons.
DECAY_GRID = (1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60)

# Coefficient blocks for `block_window`. Splitting the design this way asks *where* the
# league movement lands: a drifting intercept is a level shift, a drifting `gp_share` slope
# would mean the persistence of availability itself changed, which is a different claim.
BLOCKS: dict[str, tuple[str, ...]] = {
    "intercept": (),
    "gp_share": ("gp_share_lag1", "gp_share_lag2", "gp_share_lag3"),
    "minutes": ("minutes_per_game_lag1", "minutes_per_game_lag2",
                "minutes_per_game_lag3", "total_minutes_lag1"),
    "absence": ("trailing_missed_lag1", "end_play_rate_lag1", "n_spells_lag1",
                "longest_spell_lag1"),
    "age": ("age", "age_sq", "career_year", "n_prior_seasons"),
    "workload": tuple(WORKLOAD_COLS),
}

SHORT_BLOCK_LOOKBACK = 8
LONG_BLOCK_LOOKBACK = None


# ── A weighted beta-binomial GLM ──────────────────────────────────────────────

def weighted_neg_loglik(y, n, mu, rho, w) -> float:
    """`-sum(w * logpmf)`, guarded the way `availability._neg_loglik` is.

    Same guard and the same reason: an unguarded nan makes the numeric gradient nan too and
    L-BFGS-B stops wherever it had reached, which looks exactly like a fitted model.
    """
    a, b = _ab(mu, rho)
    ll = betabinom.logpmf(y, n, a, b)
    return _BIG if not np.all(np.isfinite(ll)) else -float((w * ll).sum())


def fit_dispersion_weighted(y, n, mu, w) -> float:
    """MLE of rho holding the mean fixed, on weighted rows.

    This is the function the `split_window` experiment turns on: it lets `rho` be estimated
    from a different set of rows than `beta` was, which is the whole point.
    """
    lo, hi = np.log(RHO_MIN / (1 - RHO_MIN)), np.log(RHO_MAX / (1 - RHO_MAX))
    best = minimize_scalar(lambda t: weighted_neg_loglik(y, n, mu, _sigmoid(t), w),
                           bounds=(lo, hi), method="bounded")
    return float(np.clip(_sigmoid(best.x), RHO_MIN, RHO_MAX))


class WeightedBetaBinomialGLM(BetaBinomialGLM):
    """`BetaBinomialGLM` with per-row weights and an optionally supplied scaler.

    Two additions, both needed by exactly one experiment:

    - **`weights`** makes `decay` possible. A truncation window is the special case
      `w in {0, 1}`, so `decay(lambda=1.0)` must reproduce the `all` lookback exactly — a
      test pins that.
    - **`scaler`** makes `block_window` possible. Splicing a coefficient from one fit into
      another is only meaningful if both were estimated in the same standardized
      coordinates, so the short-window fit is handed the long window's scaler rather than
      fitting its own.
    """

    name = "beta_binomial_weighted"

    def __init__(self, l2: float = 1.0, features=None, max_rounds: int = 25,
                 weights: np.ndarray | None = None, scaler: StandardScaler | None = None):
        super().__init__(l2=l2, features=features, max_rounds=max_rounds)
        self.weights = weights
        self.fixed_scaler = scaler

    def fit(self, train: pd.DataFrame) -> "WeightedBetaBinomialGLM":
        raw = train[self.features].to_numpy(dtype=float)
        self.scaler = self.fixed_scaler or StandardScaler().fit(raw)
        X = self._design(train)
        y = train["gp"].to_numpy()
        n = train["team_games"].to_numpy()
        w = (np.ones(len(train)) if self.weights is None
             else np.asarray(self.weights, dtype=float))

        def objective(beta: np.ndarray, rho: float) -> tuple[float, np.ndarray]:
            mu = _sigmoid(X @ beta)
            value = weighted_neg_loglik(y, n, mu, rho, w) + self.l2 * float(beta[1:] @ beta[1:])
            a, b = _ab(mu, rho)
            r = float(np.clip(rho, RHO_MIN, RHO_MAX))
            scale = (1.0 - r) / r
            dmu = scale * (digamma(y + a) - digamma(a) - digamma(n - y + b) + digamma(b))
            grad = -(X.T @ (w * dmu * mu * (1.0 - mu)))
            grad[1:] += 2.0 * self.l2 * beta[1:]
            return value, np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        beta = np.zeros(X.shape[1])
        beta[0] = np.log(max(y.sum(), 1) / max(n.sum() - y.sum(), 1))
        rho = 0.23
        for _ in range(self.max_rounds):
            best = minimize(objective, beta, args=(rho,), jac=True, method="L-BFGS-B")
            beta = best.x
            new_rho = fit_dispersion_weighted(y, n, _sigmoid(X @ beta), w)
            if abs(new_rho - rho) < 1e-5:
                rho = new_rho
                break
            rho = new_rho
        self.beta, self.rho = beta, float(rho)
        self.converged = bool(best.success)
        return self


def refit_dispersion_on(model, rows: pd.DataFrame, role_graded: bool):
    """Re-estimate the dispersion of an already-fitted mean function on different rows.

    The `split_window` experiment in one function. The mean function keeps every row it was
    given; only `rho` is moved onto the recent window, which is what §4b's divergent optima
    ask for. Returns the same object, mutated, because the mean function is unchanged and
    copying a fitted scaler to express that would be noise.
    """
    y, n = rows["gp"].to_numpy(), rows["team_games"].to_numpy()
    mu = model.predict_mean(rows)
    ones = np.ones(len(rows))
    model.rho = fit_dispersion_weighted(y, n, mu, ones)
    if role_graded:
        buckets = RoleGradedBetaBinomial._buckets(rows)
        model.pooled_rho = float(model.rho)
        model.rho_by_role = {}
        for label in ROLE_LABELS:
            mask = buckets == label
            if mask.sum() >= MIN_ROLE_ROWS:
                model.rho_by_role[label] = fit_dispersion_weighted(
                    y[mask], n[mask], mu[mask], ones[mask])
    return model


def decay_weights(rows: pd.DataFrame, origin: int, lam: float) -> np.ndarray:
    """`lambda^(origin - season)`, so the most recent fitted season carries weight 1."""
    age = origin - season_start_year(rows)
    return np.power(float(lam), age.astype(float))


def block_indices(features: list[str]) -> dict[str, np.ndarray]:
    """Design-matrix column indices per block. Column 0 is the intercept."""
    out = {"intercept": np.array([0])}
    for name, cols in BLOCKS.items():
        if name == "intercept":
            continue
        out[name] = np.array([features.index(c) + 1 for c in cols if c in features])
    return out


# ── The four experiments ──────────────────────────────────────────────────────

def _fit_rows(train: pd.DataFrame, year: np.ndarray, origin: int,
              lookback: int | None) -> pd.DataFrame:
    floor = -np.inf if lookback is None else origin - lookback
    return train[(year < origin) & (year >= floor)]


def _pool(per_arm: dict, reference_key: str, experiment: str, seed: int) -> pd.DataFrame:
    pooled = {k: {m: np.concatenate(v) for m, v in d.items() if m != "meta"}
              for k, d in per_arm.items()}
    reference = pooled[reference_key]["crps"]
    grid = np.linspace(0, 1, 101)
    rows = []
    for name, d in pooled.items():
        y, n, org, u = d["y"], d["n"], d["origin"], d["pit"]
        mean, lo, hi = paired_bootstrap(d["crps"], reference, seed=seed)
        wins = sum(1 for o in np.unique(org)
                   if d["crps"][org == o].mean() < reference[org == o].mean())
        rows.append({
            "experiment": experiment, "arm": name,
            "n_origins": int(len(np.unique(org))), "n_scored": int(len(y)),
            "crps": float(d["crps"].mean()),
            "crps_vs_reference": mean, "lo": lo, "hi": hi,
            "origins_won": wins,
            "pit_ks": float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u) - grid))),
            "err_below_10": float(d["p_below_10"].mean() - (y < 10).mean()),
            "err_below_41": float(d["p_below_41"].mean() - (y < 41).mean()),
            "err_below_60": float(d["p_below_60"].mean() - (y < 60).mean()),
            "err_full_schedule": float(d["p_full"].mean() - (y == n).mean()),
            **{k: v for k, v in per_arm[name]["meta"][0].items()},
        })
    out = pd.DataFrame(rows)
    out["boundary_error"] = (out["err_below_10"].abs() + out["err_full_schedule"].abs()) / 2
    out["body_error"] = (out["err_below_41"].abs() + out["err_below_60"].abs()) / 2
    return out.sort_values("crps").reset_index(drop=True)


def _collect(per_arm: dict, name: str, model, score: pd.DataFrame, max_games: int,
             origin: int, meta: dict) -> None:
    scored = _origin_scores(model, score, max_games)
    scored["origin"] = np.full(len(score), origin)
    slot = per_arm.setdefault(name, {"meta": [meta]})
    for key, values in scored.items():
        slot.setdefault(key, []).append(values)


def fit_decayed(rows: pd.DataFrame, origin: int, lam: float, l2: float = 1.0
                ) -> RoleGradedBetaBinomial:
    """A role-graded head whose mean AND dispersion are fitted under geometric decay.

    Both halves take the same weights, so a decay arm differs from the incumbent in the
    weighting and in nothing else — and `lam = 1.0` makes every weight 1, which is why it
    reproduces the untruncated fit exactly rather than approximately.
    """
    w = decay_weights(rows, origin, lam)
    model = WeightedBetaBinomialGLM(l2=l2, features=list(FEATURE_COLS), weights=w).fit(rows)
    graded = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS))
    graded.scaler, graded.beta = model.scaler, model.beta
    y, n = rows["gp"].to_numpy(), rows["team_games"].to_numpy()
    mu = graded.predict_mean(rows)
    graded.rho = fit_dispersion_weighted(y, n, mu, w)
    graded.pooled_rho = float(graded.rho)
    buckets = RoleGradedBetaBinomial._buckets(rows)
    graded.rho_by_role = {}
    for label in ROLE_LABELS:
        mask = buckets == label
        if mask.sum() >= MIN_ROLE_ROWS:
            graded.rho_by_role[label] = fit_dispersion_weighted(
                y[mask], n[mask], mu[mask], w[mask])
    return graded


def experiment_l2_by_lookback(train, max_games, origins, seed) -> pd.DataFrame:
    """Is the 8-season optimum real, or is it where fixed `l2` happens to bite?"""
    year = season_start_year(train)
    per_arm: dict = {}
    for origin in origins:
        score = train[year == origin]
        for lookback in LOOKBACK_GRID:
            rows = _fit_rows(train, year, origin, lookback)
            if len(rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
                continue
            for l2 in L2_GRID:
                lb = "all" if lookback is None else str(lookback)
                name = f"lb{lb}__l2_{l2:g}"
                model = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS)).fit(rows)
                _collect(per_arm, name, model, score, max_games, origin,
                         {"lookback": lb, "l2": l2,
                          "l2_per_1000_rows": l2 / (len(rows) / 1000.0)})
    return _pool(per_arm, "lball__l2_1", "l2_by_lookback", seed)


def experiment_split_window(train, max_games, origins, seed) -> pd.DataFrame:
    """Fit `beta` on one window and `rho` on another — the divergent optima, exploited."""
    year = season_start_year(train)
    per_arm: dict = {}
    for origin in origins:
        score = train[year == origin]
        for beta_lb in (None, 12, 8, 5):
            beta_rows = _fit_rows(train, year, origin, beta_lb)
            if len(beta_rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
                continue
            base = RoleGradedBetaBinomial(features=list(FEATURE_COLS)).fit(beta_rows)
            for rho_lb in LOOKBACK_GRID:
                rho_rows = _fit_rows(train, year, origin, rho_lb)
                if len(rho_rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
                    continue
                model = RoleGradedBetaBinomial(features=list(FEATURE_COLS))
                model.scaler, model.beta = base.scaler, base.beta
                model.l2, model.features = base.l2, base.features
                refit_dispersion_on(model, rho_rows, role_graded=True)
                bl = "all" if beta_lb is None else str(beta_lb)
                rl = "all" if rho_lb is None else str(rho_lb)
                _collect(per_arm, f"beta{bl}__rho{rl}", model, score, max_games, origin,
                         {"beta_lookback": bl, "rho_lookback": rl})
    return _pool(per_arm, "betaall__rhoall", "split_window", seed)


def experiment_decay(train, max_games, origins, seed) -> pd.DataFrame:
    """A smooth weight against a hard edge. `lambda = 1.0` IS the `all` lookback."""
    year = season_start_year(train)
    per_arm: dict = {}
    for origin in origins:
        score = train[year == origin]
        rows = train[year < origin]
        if len(rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
            continue
        for lam in DECAY_GRID:
            graded = fit_decayed(rows, origin, lam)
            _collect(per_arm, f"decay_{lam:g}", graded, score, max_games, origin,
                     {"lam": lam, "effective_lookback": np.inf if lam >= 1.0 else 1 / (1 - lam)})
    return _pool(per_arm, "decay_1", "decay", seed)


def experiment_block_window(train, max_games, origins, seed) -> pd.DataFrame:
    """Which coefficients need the short window, and which can keep every season?

    Both fits use the **long** window's scaler, so a coefficient from one is meaningful in
    the other's design. Each arm is the long-window vector with exactly one block replaced
    by the short-window estimate, so a block that matters shows up as its own row.
    """
    year = season_start_year(train)
    idx = block_indices(list(FEATURE_COLS))
    per_arm: dict = {}
    for origin in origins:
        score = train[year == origin]
        long_rows = _fit_rows(train, year, origin, LONG_BLOCK_LOOKBACK)
        short_rows = _fit_rows(train, year, origin, SHORT_BLOCK_LOOKBACK)
        if min(len(long_rows), len(short_rows)) < MIN_ROLE_ROWS * len(ROLE_LABELS):
            continue
        long_model = RoleGradedBetaBinomial(features=list(FEATURE_COLS)).fit(long_rows)
        short_model = WeightedBetaBinomialGLM(features=list(FEATURE_COLS),
                                              scaler=long_model.scaler).fit(short_rows)
        combos = {"none": np.array([], dtype=int), **idx,
                  "intercept+workload": np.concatenate([idx["intercept"], idx["workload"]]),
                  "all_blocks": np.arange(len(long_model.beta))}
        for name, columns in combos.items():
            spliced = RoleGradedBetaBinomial(features=list(FEATURE_COLS))
            spliced.scaler, spliced.l2 = long_model.scaler, long_model.l2
            spliced.features = long_model.features
            beta = long_model.beta.copy()
            if len(columns):
                beta[columns] = short_model.beta[columns]
            spliced.beta = beta
            refit_dispersion_on(spliced, short_rows, role_graded=True)
            _collect(per_arm, f"short__{name}", spliced, score, max_games, origin,
                     {"block": name, "n_columns": int(len(columns))})
    return _pool(per_arm, "short__none", "block_window", seed)


# ── The selected recipe, as a fittable head ───────────────────────────────────

class BlockWindowedBetaBinomial(RoleGradedBetaBinomial):
    """Slopes on every season; the drifting blocks and `rho` on a recent window.

    What `block_window` selected. The league moved the *level* of availability and the
    workload relationship, not the age curve or the persistence of absence — so discarding
    seasons wholesale throws away information that the stable blocks were using. This head
    keeps every season for 15 of its 20 coefficients and windows the 5 that drift.

    Fitted in two passes sharing one scaler, because a coefficient estimated in one set of
    standardized coordinates is meaningless in another.
    """

    name = "beta_binomial_block_windowed"

    def __init__(self, l2: float = 1.0, features=None, max_rounds: int = 25,
                 short_blocks: tuple[str, ...] = ("intercept", "workload"),
                 short_lookback: int = SHORT_BLOCK_LOOKBACK):
        super().__init__(l2=l2, features=features, max_rounds=max_rounds)
        self.short_blocks = short_blocks
        self.short_lookback = short_lookback

    def fit(self, train: pd.DataFrame) -> "BlockWindowedBetaBinomial":
        year = season_start_year(train)
        recent = train[year > year.max() - self.short_lookback]
        long_model = RoleGradedBetaBinomial(l2=self.l2, features=self.features).fit(train)
        short_model = WeightedBetaBinomialGLM(l2=self.l2, features=self.features,
                                              scaler=long_model.scaler).fit(recent)

        idx = block_indices(list(self.features))
        columns = np.concatenate([idx[b] for b in self.short_blocks]) if self.short_blocks \
            else np.array([], dtype=int)
        beta = long_model.beta.copy()
        beta[columns] = short_model.beta[columns]

        self.scaler, self.beta = long_model.scaler, beta
        self.n_short_rows, self.n_long_rows = len(recent), len(train)
        refit_dispersion_on(self, recent, role_graded=True)
        return self


def confirm_on_validation(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
                          seed: int) -> pd.DataFrame:
    """The one validation reading, taken after the rolling harness has selected.

    This is the only place in this module that touches validation, and it is deliberately
    last: everything above chose the recipe on the fitting half, so these rows are still an
    out-of-sample check rather than the thing that was optimised against. Three arms — the
    shipped head, the previous best from the window ladder, and what `block_window`
    selected — scored through `availability_window.score_arm` so the numbers are directly
    comparable to `availability_window.csv`.
    """
    from src.models.availability_window import score_arm

    year = season_start_year(train)
    recent8 = train[year > year.max() - SHORT_BLOCK_LOOKBACK]
    arms = {
        "incumbent__all_seasons_shared_rho":
            (BetaBinomialGLM(features=list(FEATURE_COLS)), train),
        "window_ladder__2012_role_rho":
            (RoleGradedBetaBinomial(features=list(FEATURE_COLS)),
             train[year >= 2012]),
        "rolling_best__lookback8_role_rho":
            (RoleGradedBetaBinomial(features=list(FEATURE_COLS)), recent8),
        "rolling_best__lookback8_l2_16":
            (RoleGradedBetaBinomial(l2=16.0, features=list(FEATURE_COLS)), recent8),
        "selected__block_windowed":
            (BlockWindowedBetaBinomial(features=list(FEATURE_COLS)), train),
    }
    rows = []
    first_val = int(season_start_year(val).min())
    for lam in (0.90, 0.85, 0.80):
        model = fit_decayed(train, first_val, lam)
        row, _ = score_arm(f"selected__decay_{lam:g}", model, train, val,
                           list(FEATURE_COLS), max_games, seed)
        row["experiment"] = "validation_confirmation"
        rows.append(row)
    for name, (model, fit_rows) in arms.items():
        model.fit(fit_rows)
        row, _ = score_arm(name, model, fit_rows, val, list(FEATURE_COLS), max_games, seed)
        row["experiment"] = "validation_confirmation"
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    seed = int(cfg.get("features", {}).get("availability", {}).get("seed", 42))

    panel = build_panel(seasons, raw_dir)
    design = build_design(season_availability(panel, "full"), seasons, raw_dir,
                          season_start_dates(panel))
    train, val = selection_split(design)
    max_games = int(design["team_games"].max())
    years = np.asarray(sorted(np.unique(season_start_year(train))))
    origins = [int(y) for y in years if y >= FIRST_ORIGIN]

    print(f"Availability weighting: {len(train):,} fitting-half rows, "
          f"{len(origins)} rolling origins ({origins[0]}–{origins[-1]}).")
    print("  Validation is NOT read here — it stays available to confirm whatever this "
          "selects.")

    tables = []
    for label, fn in [("l2_by_lookback", experiment_l2_by_lookback),
                      ("split_window", experiment_split_window),
                      ("decay", experiment_decay),
                      ("block_window", experiment_block_window)]:
        print(f"\n── {label} ──")
        table = fn(train, max_games, origins, seed)
        best = table.iloc[0]
        print(f"  best: {best['arm']}  CRPS {best['crps']:.4f}  "
              f"({best['crps_vs_reference']:+.4f} vs reference, "
              f"{int(best['origins_won'])}/{int(best['n_origins'])} origins)")
        tables.append(table)

    out = pd.concat(tables, ignore_index=True)
    dest = out_dir / "availability_weighting.csv"
    out.to_csv(dest, index=False)
    print(f"\nWrote {len(out):,} arms across 4 experiments → {dest}")

    print("\n── validation confirmation (the ONE reading, taken after selection) ──")
    confirm = confirm_on_validation(train, val, max_games, seed)
    for _, r in confirm.iterrows():
        print(f"  {r['arm']:<38} CRPS {r['val_crps']:.4f}  PIT {r['val_pit_ks']:.4f}  "
              f"P(<10) err {r['err_below_10']:+.4f}  P(full) err {r['err_full_schedule']:+.4f}")
    cdest = out_dir / "availability_weighting_confirmation.csv"
    confirm.to_csv(cdest, index=False)
    print(f"\nWrote {len(confirm):,} arms → {cdest}")
    return {"availability_weighting": dest, "availability_weighting_confirmation": cdest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
