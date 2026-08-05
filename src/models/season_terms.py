r"""Does any head need a season term, and which kind? — the ablation.

Every head in this project — availability, minutes, all eleven components — is fitted on
30 pooled seasons with **no season term**, against the repo's own standing rule ("ALWAYS
absorb season when regressing on 30 pooled seasons"). The reason the rule was never
applied is the prediction-time constraint: **a season fixed effect for season S does not
exist when forecasting S.** So the question is not "absorb season or not" but which of two
*usable* forms each head needs, and this module measures it.

`make season-effects` already measured the league series. This does not re-derive it; it
takes those findings as the hypothesis and tests them where it counts — on held-out CRPS,
calibration, and season-total dk_pts.

## The two forms, and why they are complementary rather than alternatives

Subtracting a linear trend shifts the **mean** of the year-over-year changes and leaves
their **variance** exactly unchanged, because `diff(a + b*x)` is the constant `b`. So:

- a **year-on-year trend** is a covariate. It enters the design matrix, costs one column,
  changes the mean, and can only help where the league moves smoothly in one direction.
  Measured, that is `fg3a` and nothing else (trend R^2 0.93 at +4.07%/season, against
  `stl` at 0.03).
- a **year-level random effect** is a parameter block. It is mean-zero at prediction time
  by construction and contributes its *variance*, so it widens the predictive rather than
  shifting it. It is the only thing that can touch the irreducible year-to-year spread.

That asymmetry sets the whole evaluation. **CRPS, PIT and interval coverage are where a
year effect has to earn its place**, not R^2 or MAE — at prediction time it is mean-zero,
so it widens the predictive and cannot shift it. It has one *legitimate* second-order
route to point accuracy, worth naming so it is not mistaken for the first: in the fit it
absorbs season, so `beta` is estimated **within** season, which is the repo's own standing
rule ("ALWAYS absorb season when regressing on 30 pooled seasons"). A small R^2 move is
therefore expected and good; a large one means the term is soaking up something that is
not league movement. A trend that improves *everything* is the opposite failure: with two
held-out seasons, a trend fitted on 28 and extrapolated 1-2 forward can fit the last two
by accident.

## What is fitted

Four arms per head, a 2x2 over the two forms, on top of that head's **already-selected**
specification (`stan_component_metrics.csv`, `stan_minutes_metrics.csv`) so the contrast
is the season term and nothing else:

    base | trend | year | trend_year

Availability gets two more, because `docs/availability-plan.md` finds its era effect is
**role-graded** rather than a level shift — heavy-minute players lost -0.101 of games-played
share from 2004-2010 to 2023-2025 against -0.037 for fringe players:

    trend_x_role | trend_x_role_year

Role is bucketed on **prior-season** MPG, which is known before the season starts, so the
interaction is legitimate point-in-time information. Bucketing on the target season's MPG
would be the target leaking into the design.

## Two confounds that sit inside the window, and the third path that is not a fit

`make season-effects` (`season_effects_regimes.csv`) tests both. 2019-20 and 2020-21 are a
COVID health-protocol regime; the Player Participation Policy arrives in 2023-24 and the
break is significant on exactly the series the availability plan predicted. A smooth trend
extrapolated across a policy discontinuity is actively wrong, which is why the trend arm
here is reported *beside* the break test rather than instead of it.

And rule changes are **announced in the summer**, before opening night, so a manual
league-level multiplier is legitimate point-in-time information in a way that nothing drawn
from inside the season is. `LEAGUE_OVERRIDE` keeps that path open and `oracle_override`
prices its ceiling: what a *perfect* preseason announcement would have been worth. That
ceiling bounds the whole question — if it is small, no season term of any kind can be large.

## Sampler note

Every fit here uses `metric="dense_e"`. The selected specs are B-spline bases, whose
columns are strongly correlated, and a diagonal metric cannot absorb that: measured on the
`blk` spline arm, **236.6 s at treedepth-saturation 35 against 13.4 s at 0** — a 17.7x
difference for the same posterior. `CLAUDE.md` records that an orthogonalized basis is the
fix if spline variants ever become the shipped spec; this is the cheaper equivalent, and it
is applied to **every** arm so the metric cannot confound the base-versus-season contrast.

Usage:
    python -m src.models.season_terms
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.features.targets import (BONUS_CATEGORIES, BONUS_GAME_OVERDISPERSION,
                                  expected_bonus)
from src.models.availability import (FEATURE_COLS, LeagueAgeBaseline, crps as gp_crps,
                                     pit_values, split_seasons as split_availability)
from src.models.component_rates import (CONVERSION_HEADS, COUNT_HEADS, DERIVED_COUNTS,
                                        build_design as build_component_design,
                                        split_seasons)
from src.models.stan_availability import StanAvailability, availability_design
from src.models.stan_components import (StanConversion, StanCount, conversion_floor,
                                        conversion_variants, count_floor,
                                        count_variants, score_conversion, score_count)
from src.models.stan_minutes import (FloorMinutes, StanMinutes,
                                     build_design as build_minutes_design,
                                     score as score_minutes, variants as minutes_variants)
from src.models.stan_utils import (crps_from_samples, diagnostics_frame, ks_uniform,
                                   pit_from_samples)

SEASON_TREND = "season_trend"
YEAR_COLUMN = "season"
METRIC = "dense_e"

# Central predictive intervals whose realized coverage is checked. 0.5 and 0.8 are where a
# draft product actually operates; 0.95 is where an under-dispersed predictive is most
# visibly broken.
COVERAGE_LEVELS = (0.5, 0.8, 0.95)

# Fallback specs when the selection artifacts are absent. `log_own` / `logit_own` are the
# right scale for their links; they are NOT the selected spec for every head, and the
# fallback prints a warning saying so.
DEFAULT_COUNT_SPEC = "log_own"
DEFAULT_CONVERSION_SPEC = "logit_own"
DEFAULT_MINUTES_SPEC = "logit_own_spline"

# Posterior draws pushed through the season-total composition. Eleven heads x draws x rows
# of predictive samples, so this is a memory knob; 400 puts Monte Carlo error on a coverage
# rate well inside a point of the sampling error on 791 test rows.
COMPOSITION_DRAWS = 400
BONUS_SAMPLES = 128

# ── The manual league-level override ─────────────────────────────────────────────────
#
# Rule changes and points of emphasis are ANNOUNCED before opening night — the 2004-05
# hand-checking crackdown and the 2021-22 non-basketball-moves emphasis were both public in
# advance — so a league-level multiplier set by hand is legitimate under this project's
# point-in-time discipline, unlike anything drawn from inside the season. It is deliberately
# NOT a fitted quantity: the whole point is that a human read a rule change the data cannot
# yet see.
#
# `{component: {season: multiplier}}`, applied to the predicted season total. Empty by
# default, which is exactly today's behaviour. `oracle_override` below measures the ceiling
# on what filling it in could ever be worth.
LEAGUE_OVERRIDE: dict[str, dict[str, float]] = {}


# ── Trend and role terms ──────────────────────────────────────────────────────

def season_start_year(frame: pd.DataFrame, col: str = "season") -> np.ndarray:
    """`"2019-20"` -> 2019. The season label is the only shared time key across designs."""
    return frame[col].astype(str).str.slice(0, 4).astype(int).to_numpy()


def add_trend(train: pd.DataFrame, frames: list[pd.DataFrame]
              ) -> tuple[list[pd.DataFrame], list[str]]:
    """Seasons since the training mean — a covariate that extrapolates by construction.

    Centering on **train** matters for the same reason the scaler is fitted on train: the
    held-out seasons must sit outside the fitted range, because that is what extrapolating
    one season forward means. Centering on the pooled frame would quietly put the test
    seasons inside the support and make the trend look better than it is.
    """
    centre = float(season_start_year(train).mean())
    out = []
    for frame in frames:
        copy = frame.copy()
        copy[SEASON_TREND] = season_start_year(frame) - centre
        out.append(copy)
    return out, [SEASON_TREND]


def add_role_terms(train: pd.DataFrame, frames: list[pd.DataFrame],
                   mpg_col: str = "minutes_per_game_lag1"
                   ) -> tuple[list[pd.DataFrame], list[str]]:
    """Prior-MPG role dummies and their interaction with the trend.

    **Prior-season** MPG, never the target season's: role has to be knowable before the
    season starts or the interaction is the outcome regressed on itself. The buckets are
    `src/eda/season_effects.ROLE_EDGES`, the same ones `docs/availability-plan.md` measured
    the -0.101 versus -0.037 split on, so the fitted interaction is comparable to it.

    The lowest bucket is the reference level, so `trend` keeps its meaning as that bucket's
    slope and each interaction reads as the extra slope for a heavier role.
    """
    names: list[str] = []
    out = []
    for frame in frames:
        copy = frame.copy()
        bucket = pd.cut(copy[mpg_col].to_numpy(dtype=float), ROLE_EDGES, labels=ROLE_LABELS)
        names = []
        for label in ROLE_LABELS[1:]:
            flag = f"role_{label.replace(' ', '_').replace('<', 'lt')}"
            copy[flag] = (bucket == label).astype(float)
            copy[f"{flag}__x__trend"] = copy[flag] * copy[SEASON_TREND]
            names += [flag, f"{flag}__x__trend"]
        out.append(copy)
    return out, names


# ── Arms ──────────────────────────────────────────────────────────────────────

def season_arms(train: pd.DataFrame, test: pd.DataFrame, features: list[str]
                ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str], str | None]]:
    """The 2x2 over the two usable forms, on top of a fixed base specification.

    Returns `(train, test, features, year_column)` per arm. `year_column=None` is the
    disabled year term, which makes the Stan parameter vectors zero-length and the model
    identical to the one without it — so `base` here IS the shipped head, not a re-fit of
    something near it.
    """
    (tt, et), trend_names = add_trend(train, [train, test])
    return {
        "base": (train, test, list(features), None),
        "trend": (tt, et, features + trend_names, None),
        "year": (train, test, list(features), YEAR_COLUMN),
        "trend_year": (tt, et, features + trend_names, YEAR_COLUMN),
    }


def availability_arms(train: pd.DataFrame, test: pd.DataFrame,
                      features: list[str] | None = None
                      ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str], str | None]]:
    """The 2x2 plus the two role-interaction arms the availability finding calls for."""
    features = list(features or FEATURE_COLS)
    arms = season_arms(train, test, features)
    (tt, et), _ = add_trend(train, [train, test])
    (rt, rte), role_names = add_role_terms(tt, [tt, et])
    role_features = features + [SEASON_TREND] + role_names
    arms["trend_x_role"] = (rt, rte, role_features, None)
    arms["trend_x_role_year"] = (rt, rte, role_features, YEAR_COLUMN)
    return arms


# ── Calibration ───────────────────────────────────────────────────────────────

def coverage_from_samples(samples: np.ndarray, y: np.ndarray,
                          levels: tuple[float, ...] = COVERAGE_LEVELS) -> dict:
    """Realized coverage of central predictive intervals — the year effect's real test.

    A mean-zero term cannot move R^2 or MAE, so if it is doing anything useful it shows up
    here: an under-dispersed predictive achieves less than nominal coverage, and adding the
    year effect should push each level toward its nominal value from below.
    """
    y = np.asarray(y, dtype=float)
    out = {}
    for level in levels:
        lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        out[f"coverage_{int(level * 100)}"] = float(((y >= lo) & (y <= hi)).mean())
        out[f"width_{int(level * 100)}"] = float((hi - lo).mean())
    return out


def coverage_from_pmf(pmf: np.ndarray, y: np.ndarray,
                      levels: tuple[float, ...] = COVERAGE_LEVELS) -> dict:
    """The same quantity for a head whose predictive is an explicit pmf grid."""
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    y = np.asarray(y, dtype=int)
    out = {}
    for level in levels:
        lo = np.argmax(cdf >= (1 - level) / 2, axis=1)
        hi = np.argmax(cdf >= 1 - (1 - level) / 2, axis=1)
        out[f"coverage_{int(level * 100)}"] = float(((y >= lo) & (y <= hi)).mean())
        out[f"width_{int(level * 100)}"] = float((hi - lo).mean())
    return out


def _bias(pred: np.ndarray, y: np.ndarray) -> dict:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    total = y.sum()
    return {"bias": float((pred - y).mean()),
            "bias_pct": float(100 * (pred - y).sum() / total) if total else np.nan}


# ── The league-level override ─────────────────────────────────────────────────

def oracle_override(pred: np.ndarray, y: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    """The CEILING on any league-level correction: rescale each season to its own truth.

    One multiplier per held-out season, `sum(y) / sum(pred)`, applied to every player in
    that season. It uses the realized season, so it is an oracle and can never be a model —
    that is the point. It bounds what a manual preseason override, a fitted trend, or any
    other league-level term could possibly buy, because a *single* league-wide scalar per
    season is the most general form all three take.

    If a head's fitted arms fall well short of this, the league movement is real and the
    fitted forms cannot reach it. If the oracle itself is worth little, the whole season-term
    question is closed for that head regardless of form.
    """
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    out = pred.copy()
    for season in np.unique(seasons):
        m = seasons == season
        denom = pred[m].sum()
        if denom > 0:
            out[m] = pred[m] * (y[m].sum() / denom)
    return out


def apply_override(pred: np.ndarray, component: str, seasons: np.ndarray,
                   table: dict[str, dict[str, float]] | None = None) -> np.ndarray:
    """Apply a hand-set, announced-in-advance league multiplier. A no-op when unset."""
    table = LEAGUE_OVERRIDE if table is None else table
    per_season = table.get(component, {})
    if not per_season:
        return np.asarray(pred, float)
    out = np.asarray(pred, float).copy()
    for season, multiplier in per_season.items():
        out[seasons == season] *= float(multiplier)
    return out


# ── Selected base specifications ──────────────────────────────────────────────

def selected_specs(predictions_dir: Path) -> tuple[dict[str, str], str]:
    """Each head's validation-selected spec, read from the artifacts that chose it.

    Read rather than re-derived so this ablation sits on the shipped specification instead
    of a second opinion about it. Missing artifacts fall back to the right-scale spec and
    say so — a silent fallback would make `fg3a` and `blk` look like season-term failures
    when they are really spec failures (`log_own` alone is below their floor under NB).
    """
    specs: dict[str, str] = {}
    path = predictions_dir / "stan_component_metrics.csv"
    if path.exists():
        table = pd.read_csv(path)
        for head, block in table[table["selected"]].groupby("head"):
            specs[str(head)] = str(block["variant"].iloc[0])
        # A *stale* artifact is the dangerous case, not a missing one. If it was written
        # before a head-list change it resolves for the old heads and silently drops back
        # to `log_own` for the new ones — and `log_own` is exactly the spec Gate 0 showed
        # is below its floor on the skewed attempt heads. Name the heads rather than
        # letting the fallback happen quietly.
        expected = set(COUNT_HEADS) | {f"{m}|{a}" for m, a in CONVERSION_HEADS}
        missing = sorted(expected - set(specs))
        if missing:
            print(f"  /!\\  {path} selects no variant for {missing} — falling back to "
                  f"{DEFAULT_COUNT_SPEC}/{DEFAULT_CONVERSION_SPEC}. If the head lists "
                  f"changed, re-run `make stan-components` before trusting this.")
    minutes_spec = DEFAULT_MINUTES_SPEC
    mpath = predictions_dir / "stan_minutes_metrics.csv"
    if mpath.exists():
        m = pd.read_csv(mpath)
        chosen = m[m["selected"]]
        if len(chosen):
            minutes_spec = str(chosen["variant"].iloc[0])
    return specs, minutes_spec


# ── Sweeps ────────────────────────────────────────────────────────────────────

def _iters(cfg_stan: dict) -> dict:
    """One sampler budget for every arm on both splits.

    The component sweep uses shorter chains for selection and longer ones for the test
    column; here both splits get the same budget, because the comparison being made is
    between ARMS and a budget that differs by split would put a second difference inside it.

    **Consequence worth stating rather than discovering later: the `base` arm's test column
    is NOT directly comparable to `stan_component_metrics.csv`.** That artifact's test
    fits run at 1000/1000 and these run at the selection budget, so small differences
    between the two tables are sampler noise, not a changed model. Within this table every
    arm shares the budget, which is what the ablation needs.
    """
    return {"warmup": int(cfg_stan.get("select_warmup", 500)),
            "samples": int(cfg_stan.get("select_samples", 500))}


def sweep_counts(train, val, test, full_train, specs, cfg_stan, n_knots
                 ) -> tuple[pd.DataFrame, list[dict], dict]:
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics, keep = [], [], {}

    for component in COUNT_HEADS:
        spec = specs.get(component, DEFAULT_COUNT_SPEC)
        floor_val = count_floor(train, val, component, seed)
        floor_test = count_floor(full_train, test, component, seed)
        floor_samples = _count_floor_samples(full_train, test, component, seed)
        y_test = test[component].to_numpy(float)
        rows.append({"head": component, "kind": "count", "arm": "carry_forward",
                     "spec": "none", "n_features": 0,
                     "val_r2": floor_val["r2"], "val_crps": floor_val["crps"],
                     **{f"test_{k}": v for k, v in floor_test.items()},
                     **_bias(np.clip(_carry(test, component), 0, None), y_test),
                     **coverage_from_samples(floor_samples, y_test)})

        val_specs = count_variants(train, val, component, n_knots)[spec]
        test_specs = count_variants(full_train, test, component, n_knots)[spec]
        for arm, (v_tr, v_te, v_features, year) in season_arms(
                val_specs[0], val_specs[1], val_specs[2]).items():
            v_model = StanCount(v_features, component, name=f"{component}/{arm}/val",
                                chains=chains, seed=seed, year_column=year, metric=METRIC,
                                **_iters(cfg_stan)).fit(v_tr)
            diagnostics.append(v_model.diagnostics)
            v_y = v_te[component].to_numpy(float)
            v = score_count(v_y, v_model.predict_mean(v_te),
                            v_model.predict_samples(v_te, seed), v_model.phi, seed)

            t_tr, t_te, t_features, _ = season_arms(
                test_specs[0], test_specs[1], test_specs[2])[arm]
            t_model = StanCount(t_features, component, name=f"{component}/{arm}/test",
                                chains=chains, seed=seed, year_column=year, metric=METRIC,
                                **_iters(cfg_stan)).fit(t_tr)
            diagnostics.append(t_model.diagnostics)
            t_pred = t_model.predict_mean(t_te)
            t_samples = t_model.predict_samples(t_te, seed)
            t = score_count(y_test, t_pred, t_samples, t_model.phi, seed)
            keep[(component, arm)] = (t_model, t_te)

            rows.append({"head": component, "kind": "count", "arm": arm, "spec": spec,
                         "n_features": len(t_features), "val_r2": v["r2"],
                         "val_crps": v["crps"],
                         **{f"test_{k}": val for k, val in t.items()},
                         **_bias(t_pred, y_test),
                         **coverage_from_samples(t_samples, y_test),
                         **{f"year_{k}": v2 for k, v2 in t_model.year.summary().items()}})

        # The oracle ceiling, on the base arm's own predictions.
        base_model, base_te = keep[(component, "base")]
        base_pred = base_model.predict_mean(base_te)
        seasons = test["season"].to_numpy()
        for label, pred in (
                ("oracle_league", oracle_override(base_pred, y_test, seasons)),
                # The hand-set path, live rather than commented out. With `LEAGUE_OVERRIDE`
                # empty this row is `base` exactly, which is the point: the row exists so
                # that filling in an announced rule change is a config edit and a re-run
                # rather than new code.
                ("manual_override", apply_override(base_pred, component, seasons))):
            rows.append({"head": component, "kind": "count", "arm": label, "spec": spec,
                         "n_features": len(base_model.features),
                         "test_r2": float(1 - ((y_test - pred) ** 2).sum()
                                          / ((y_test - y_test.mean()) ** 2).sum()),
                         "test_mae": float(np.abs(y_test - pred).mean()),
                         **_bias(pred, y_test)})
    return _finalize(pd.DataFrame(rows), "val_crps", "test_crps",
                     higher_is_better=False), diagnostics, keep


def _carry(frame: pd.DataFrame, component: str) -> np.ndarray:
    from src.models.component_rates import carry_forward
    return carry_forward(frame, component)


def _count_floor_samples(train, test, component, seed) -> np.ndarray:
    from src.models.component_rates import carry_forward, fit_nb_dispersion
    mu = np.clip(carry_forward(test, component), 1e-6, None)
    phi = fit_nb_dispersion(train[component].to_numpy(float),
                            np.clip(carry_forward(train, component), 1e-6, None))
    rng = np.random.default_rng(seed)
    p = phi / (phi + mu)
    return rng.negative_binomial(np.full((400, len(mu)), phi),
                                 np.repeat(p[None, :], 400, axis=0)).astype(float)


def sweep_conversions(train, val, test, full_train, specs, cfg_stan, n_knots
                      ) -> tuple[pd.DataFrame, list[dict], dict]:
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics, keep = [], [], {}

    for made, attempted in CONVERSION_HEADS:
        head = f"{made}|{attempted}"
        spec = specs.get(head, DEFAULT_CONVERSION_SPEC)
        floor_val = conversion_floor(train, val, made, attempted, seed)
        floor_test = conversion_floor(full_train, test, made, attempted, seed)
        rows.append({"head": head, "kind": "conversion", "arm": "carry_forward",
                     "spec": "none", "n_features": 0, "val_nll": floor_val["nll"],
                     "val_crps": floor_val["crps"],
                     **{f"test_{k}": v for k, v in floor_test.items()}})

        val_specs = conversion_variants(train, val, made, attempted, n_knots)[spec]
        test_specs = conversion_variants(full_train, test, made, attempted, n_knots)[spec]
        for arm, (v_tr, v_te, v_features, year) in season_arms(
                val_specs[0], val_specs[1], val_specs[2]).items():
            v_model = StanConversion(v_features, made, attempted,
                                     name=f"{head}/{arm}/val", chains=chains, seed=seed,
                                     year_column=year, metric=METRIC,
                                     **_iters(cfg_stan)).fit(v_tr)
            diagnostics.append(v_model.diagnostics)
            v = _score_conv(v_model, v_te, made, attempted, seed)

            t_tr, t_te, t_features, _ = season_arms(
                test_specs[0], test_specs[1], test_specs[2])[arm]
            t_model = StanConversion(t_features, made, attempted,
                                     name=f"{head}/{arm}/test", chains=chains, seed=seed,
                                     year_column=year, metric=METRIC,
                                     **_iters(cfg_stan)).fit(t_tr)
            diagnostics.append(t_model.diagnostics)
            t = _score_conv(t_model, t_te, made, attempted, seed)
            keep[(head, arm)] = (t_model, t_te)

            rows.append({"head": head, "kind": "conversion", "arm": arm, "spec": spec,
                         "n_features": len(t_features), "val_nll": v["nll"],
                         "val_crps": v["crps"],
                         **{f"test_{k}": val for k, val in t.items()},
                         **{f"year_{k}": v2 for k, v2 in t_model.year.summary().items()}})
    return _finalize(pd.DataFrame(rows), "val_nll", "test_nll",
                     higher_is_better=False), diagnostics, keep


def _score_conv(model: StanConversion, frame: pd.DataFrame, made: str, attempted: str,
                seed: int) -> dict:
    ok = frame[attempted].to_numpy(float) > 0
    live = frame[ok]
    n = np.rint(live[attempted].to_numpy(float)).astype(int)
    y = np.minimum(np.rint(live[made].to_numpy(float)).astype(int), n)
    p = model.predict_p(live)
    samples = model.predict_samples(live, seed)
    out = score_conversion(y.astype(float), n.astype(float), p, samples, model.rho, seed)
    out |= _bias(p * n, y.astype(float))
    out |= coverage_from_samples(samples, y.astype(float))
    return out


def sweep_minutes(train, val, test, full_train, spec, cfg_stan, n_knots
                  ) -> tuple[pd.DataFrame, list[dict]]:
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics = [], []

    floor_val = score_minutes(FloorMinutes().fit(train), val, "carry_forward", seed)
    floor_test = score_minutes(FloorMinutes().fit(full_train), test, "carry_forward", seed)
    floor_samples = FloorMinutes().fit(full_train).predict_samples(test, seed)
    y_test = test["successes"].to_numpy(float)
    rows.append({"head": "min", "kind": "minutes", "arm": "carry_forward", "spec": "none",
                 "n_features": 0, "val_crps": floor_val["crps_minutes"],
                 "test_crps": floor_test["crps_minutes"],
                 "val_r2": floor_val["r2_minutes"], "test_r2": floor_test["r2_minutes"],
                 "test_mae": floor_test["mae_minutes"],
                 "test_pit_ks": floor_test["pit_ks"], "bias": floor_test["bias_minutes"],
                 **coverage_from_samples(floor_samples, y_test)})

    val_specs = minutes_variants(train, val, n_knots)[spec]
    test_specs = minutes_variants(full_train, test, n_knots)[spec]
    for arm, (v_tr, v_te, v_features, year) in season_arms(
            val_specs[0], val_specs[1], val_specs[2]).items():
        v_model = StanMinutes(v_features, name=f"min/{arm}/val", chains=chains, seed=seed,
                              year_column=year, metric=METRIC,
                              **_iters(cfg_stan)).fit(v_tr)
        diagnostics.append(v_model.diagnostics)
        v = score_minutes(v_model, v_te, arm, seed)

        t_tr, t_te, t_features, _ = season_arms(
            test_specs[0], test_specs[1], test_specs[2])[arm]
        t_model = StanMinutes(t_features, name=f"min/{arm}/test", chains=chains, seed=seed,
                              year_column=year, metric=METRIC,
                              **_iters(cfg_stan)).fit(t_tr)
        diagnostics.append(t_model.diagnostics)
        t = score_minutes(t_model, t_te, arm, seed)
        samples = t_model.predict_samples(t_te, seed)
        rows.append({"head": "min", "kind": "minutes", "arm": arm, "spec": spec,
                     "n_features": len(t_features), "val_crps": v["crps_minutes"],
                     "test_crps": t["crps_minutes"], "val_r2": v["r2_minutes"],
                     "test_r2": t["r2_minutes"], "test_mae": t["mae_minutes"],
                     "test_pit_ks": t["pit_ks"], "bias": t["bias_minutes"],
                     **coverage_from_samples(samples, y_test),
                     **{f"year_{k}": v2 for k, v2 in t_model.year.summary().items()}})
    return _finalize(pd.DataFrame(rows), "val_crps", "test_crps",
                     higher_is_better=False), diagnostics


def sweep_availability(train, val, test, full_train, cfg_stan, l2
                       ) -> tuple[pd.DataFrame, list[dict]]:
    """The 2x2 plus the role interaction — the arm this head's own measurement calls for."""
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics = [], []
    max_games = int(pd.concat([full_train, test])["team_games"].max())

    val_arms = availability_arms(train, val)
    test_arms = availability_arms(full_train, test)
    y_test = test["gp"].to_numpy(int)
    n_test = test["team_games"].to_numpy(float)

    # The mandatory floor. For games played it is the league/age baseline rather than a
    # carry-forward: prior GP is the least persistent quantity in the project (r = 0.317),
    # which is exactly why `docs/availability-plan.md` shrinks toward a league/age curve
    # instead of toward the player's own prior. Its recorded CRPS is 13.614 games.
    floor_val = LeagueAgeBaseline().fit(train)
    floor_test = LeagueAgeBaseline().fit(full_train)
    floor_pmf = floor_test.predict_pmf(test, max_games)
    floor_pred = floor_test.predict_mean(test) * n_test
    rows.append({
        "head": "gp", "kind": "availability", "arm": "carry_forward",
        "spec": "league_age", "n_features": 0,
        "val_crps": float(gp_crps(floor_val.predict_pmf(val, max_games),
                                  val["gp"].to_numpy(int)).mean()),
        "test_crps": float(gp_crps(floor_pmf, y_test).mean()),
        "val_r2": _share_r2(val, floor_val), "test_r2": _share_r2(test, floor_test),
        "test_mae": float(np.abs(floor_pred - y_test).mean()),
        "test_pit_ks": ks_uniform(pit_values(floor_pmf, y_test, seed)),
        "test_dispersion": float(getattr(floor_test, "rho", np.nan)),
        **_bias(floor_pred, y_test.astype(float)),
        **coverage_from_pmf(floor_pmf, y_test)})

    for arm in val_arms:
        v_tr, v_te, v_features, year = val_arms[arm]
        v_model = StanAvailability(l2=l2, features=v_features, name=f"gp/{arm}/val",
                                   chains=chains, seed=seed, year_column=year,
                                   metric=METRIC, **_iters(cfg_stan)).fit(v_tr)
        diagnostics.append(v_model.diagnostics)
        v_pmf = v_model.predict_pmf(v_te, max_games)
        v_y = v_te["gp"].to_numpy(int)

        t_tr, t_te, t_features, _ = test_arms[arm]
        t_model = StanAvailability(l2=l2, features=t_features, name=f"gp/{arm}/test",
                                   chains=chains, seed=seed, year_column=year,
                                   metric=METRIC, **_iters(cfg_stan)).fit(t_tr)
        diagnostics.append(t_model.diagnostics)
        t_pmf = t_model.predict_pmf(t_te, max_games)
        pred = t_model.predict_mean(t_te) * n_test

        rows.append({
            "head": "gp", "kind": "availability", "arm": arm, "spec": "glm",
            "n_features": len(t_features),
            "val_crps": float(gp_crps(v_pmf, v_y).mean()),
            "test_crps": float(gp_crps(t_pmf, y_test).mean()),
            "val_r2": _share_r2(v_te, v_model), "test_r2": _share_r2(t_te, t_model),
            "test_mae": float(np.abs(pred - y_test).mean()),
            "test_pit_ks": ks_uniform(pit_values(t_pmf, y_test, seed)),
            "test_dispersion": float(t_model.rho),
            **_bias(pred, y_test.astype(float)),
            **coverage_from_pmf(t_pmf, y_test),
            **{f"year_{k}": v2 for k, v2 in t_model.year.summary().items()}})
    return _finalize(pd.DataFrame(rows), "val_crps", "test_crps",
                     higher_is_better=False), diagnostics


def _share_r2(frame: pd.DataFrame, model) -> float:
    y = (frame["gp"].to_numpy(float) / frame["team_games"].to_numpy(float))
    p = model.predict_mean(frame)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _finalize(table: pd.DataFrame, val_col: str, test_col: str,
              higher_is_better: bool) -> pd.DataFrame:
    """Mark the validation-selected arm per head, and whether it clears the no-fit floor.

    Two flags, kept apart for the reason `stan_components._finalize` keeps them apart:
    `selected` is a **validation** fact answering "which arm would I ship", `beats_floor`
    is a **test** fact answering "is this a model at all". This repo has already shipped
    one false positive whose paired bootstrap on test read [-0.079, -0.015] with
    P(delta<0) = 99.7% and did not replicate, so selection never reads a test column.

    **Selection is on CRPS here, not R^2, and that choice is forced by the question.** The
    component sweep selects count heads on R^2, which is right when the arms differ in
    their mean function. Half the arms here differ only in their *spread*: a year effect is
    mean-zero at prediction time, so selecting on R^2 would reject it by construction
    before measuring anything. CRPS scores the whole predictive and is the only criterion
    under which the two forms can compete on equal terms.

    Rows with no test metric — the oracle and manual-override arms, which are point
    predictions with no predictive distribution — get `beats_floor = NA` rather than
    `False`. Recording "the oracle fails the floor" because it has no CRPS would be a
    fabricated failure.
    """
    out = table.copy()
    out["selected"] = False
    out["beats_floor"] = pd.NA
    fixed = {"carry_forward", "oracle_league", "manual_override"}
    for head, block in out.groupby("head"):
        fitted = block[~block["arm"].isin(fixed) & block[val_col].notna()]
        if fitted.empty:
            continue
        best = (fitted[val_col].idxmax() if higher_is_better else fitted[val_col].idxmin())
        out.loc[best, "selected"] = True
        floor_rows = block[block["arm"] == "carry_forward"]
        if floor_rows.empty or test_col not in block:
            continue
        floor = floor_rows[test_col].iloc[0]
        better = (block[test_col] > floor) if higher_is_better else (block[test_col] < floor)
        scored = block[test_col].notna().to_numpy()
        out.loc[block.index[scored], "beats_floor"] = better.to_numpy()[scored]
        out.loc[floor_rows.index, "beats_floor"] = True
    return out


# ── The decisive downstream metric ────────────────────────────────────────────

DK_LINEAR_WEIGHTS = {"reb": 1.25, "ast": 1.5, "stl": 2.0, "blk": 2.0, "tov": -0.5}


def realized_season_dk(design: pd.DataFrame) -> np.ndarray:
    """Season-total dk_pts **excluding** the bonus, from realized component totals.

    Exact and linear in the eight scoring components, so it composes from season totals
    with no per-game pass: `pts = 2*fg2m + 3*fg3m + ftm`, then the DK weights. The bonus
    is a per-game threshold and is handled separately by `bonus_calibration`, because
    `E[bonus] != bonus(E[x])` and a season total genuinely cannot carry it.
    """
    pts = (2 * design["fg2m"].to_numpy(float) + 3 * design["fg3m"].to_numpy(float)
           + design["ftm"].to_numpy(float))
    total = pts + 0.5 * design["fg3m"].to_numpy(float)
    for col, weight in DK_LINEAR_WEIGHTS.items():
        total = total + weight * design[col].to_numpy(float)
    return total


def compose_season_dk(models: dict, frame: pd.DataFrame, arm: str, draws: int,
                      seed: int = 0) -> np.ndarray:
    """(draws x rows) season-total dk_pts, composed through the chain, not summed marginals.

    Attempts are drawn from the count heads and makes are then drawn **conditional on the
    drawn attempts**, which is the chain the output contract specifies. Drawing makes from
    realized attempts instead would leak the target and would also understate the spread,
    since attempt uncertainty is most of the uncertainty in points.

    The heads' year effects use independent streams (`YearTerm.stream`), which is what the
    measured cross-component shock correlation supports — mean **-0.009** over 136 pairs,
    with the large values confined to specific pairs rather than loaded on a common factor.
    """
    counts, made = _draw_components(models, frame, arm, draws, seed)
    pts = 2 * made["fg2m"] + 3 * made["fg3m"] + made["ftm"]
    total = pts + 0.5 * made["fg3m"]
    for col, weight in DK_LINEAR_WEIGHTS.items():
        total = total + weight * counts[col]
    return total


def _draw_components(models: dict, frame: pd.DataFrame, arm: str, draws: int,
                     seed: int) -> tuple[dict, dict]:
    """Aligned (draws x rows) samples of the seven counts and four make-counts.

    **Each head is scored on its OWN stored design frame, not on the raw one.** Every head
    carries a different set of derived columns — imputation flags, `log(own)`, a spline
    basis fitted on that head's own training quantiles — so the raw test frame does not
    have the columns any of them need. The stored frames are row-aligned with the raw one
    by construction (`impute` / `add_log` / `add_spline` all return copies in place), and
    that alignment is asserted rather than trusted, because a silent misalignment here
    would pair one player's rebounds with another's assists and still produce a plausible
    season total.

    `predictive_samples` is set on each model before drawing so every head returns exactly
    `draws` samples thinned across its **whole** posterior. Slicing `[:draws]` off a
    1000-sample block instead would take the first 40% of the draw order from the counts
    and the full span from the conversions — a silent mismatch in which part of the
    posterior each head contributes.
    """
    from src.models.stan_components import _beta_shapes

    def design_for(key) -> pd.DataFrame:
        model, stored = models[key]
        if len(stored) != len(frame) or not stored.index.equals(frame.index):
            raise ValueError(
                f"{key} was scored on a frame of {len(stored)} rows that does not align "
                f"with the {len(frame)}-row composition frame; the heads cannot be "
                f"composed row-wise")
        return model, stored

    rng = np.random.default_rng(seed)
    counts = {}
    for component in COUNT_HEADS:
        model, stored = design_for((component, arm))
        model.predictive_samples = draws
        counts[component] = model.predict_samples(stored, seed)

    made = {}

    def trials_for(name: str) -> np.ndarray:
        """The drawn trials for a conversion head, materializing derived counts in order.

        Under the shot-attempt basis the chain is
        `fga -> fg3a | fga -> fg2a = fga - fg3a -> makes`, so a conversion head's own draw
        (`fg3a`) becomes a later head's trials, and `fg2a` exists only as the difference.
        Neither edge existed in the two-count basis, where every `attempted` was already a
        fitted count — there this returns on the first branch and behaviour is identical.

        Clipped at zero because `fga` and `fg3a` are drawn from *different* posteriors and
        nothing forces `fg3a <= fga` on a given draw. It bites on a vanishing share of
        draws (the share head's mean is far from 1), and a negative trials count would be
        an error rather than a small bias.
        """
        if name in counts:
            return counts[name]
        if name in made:
            counts[name] = made[name]
            return counts[name]
        if name in DERIVED_COUNTS:
            total, part = DERIVED_COUNTS[name]
            counts[name] = np.maximum(trials_for(total) - trials_for(part), 0.0)
            return counts[name]
        raise KeyError(
            f"{name} is needed as trials but is neither a fitted count head, a conversion "
            f"head drawn earlier in the chain, nor a member of DERIVED_COUNTS — the head "
            f"lists and the draw order disagree")

    for m, attempted in CONVERSION_HEADS:
        model, stored = design_for((f"{m}|{attempted}", arm))
        p, rho = model.p_draws(stored, draws)
        a, b = _beta_shapes(p, rho[:, None])
        # Makes are drawn on the DRAWN attempts, not the realized ones: the chain is
        # `makes | attempts`, and conditioning on realized attempts would leak the target
        # and understate the spread, since attempt uncertainty is most of the uncertainty
        # in points.
        made[m] = rng.binomial(np.rint(trials_for(attempted)).astype(int),
                               rng.beta(a, b)).astype(float)
    return counts, made


PORTFOLIO_SIZES = (12, 15, 30, 150, None)      # None = the whole held-out board


def roster_spread(models: dict, frame: pd.DataFrame, arms: tuple[str, ...],
                  sizes: tuple = PORTFOLIO_SIZES, draws: int = COMPOSITION_DRAWS,
                  n_subsets: int = 200, seed: int = 0) -> pd.DataFrame:
    r"""How much does a season term widen a **drafted roster's** season total?

    This is the measurement the whole question turns on, and until now it was an
    assertion. `docs/predictions-plan.md` argues that season effects outrank the shared-β
    correlation the Stan work was built for, because a league shift is **perfectly
    correlated across every player** and therefore does not diversify away, where
    independent per-player error does. The shared-β term is measured
    (`stan_availability.board_correlation`): **+0.2% on a 15-man roster, +6.4% across all
    911**. The season term's equivalent number has never existed.

    It is computed the same way and is directly comparable: draw the joint season total
    per player per posterior draw, sum over a random roster, and take the sd of that sum
    across draws. The year effect enters as one shared `z` per draw, so its contribution
    to the roster total grows as **N** while independent error grows as **sqrt(N)** —
    which is exactly why the roster size has to be reported rather than one number.

    Subsets are drawn at random and averaged, so the figure is measured on real players
    rather than extrapolated under an equal-variance assumption.
    """
    rng = np.random.default_rng(seed)
    samples = {arm: compose_season_dk(models, frame, arm, draws, seed) for arm in arms
               if (COUNT_HEADS[0], arm) in models}
    if not samples:
        return pd.DataFrame()

    rows = []
    for size in sizes:
        if size is None or size >= len(frame):
            n_players, subsets = len(frame), [np.arange(len(frame))]
        else:
            n_players = size
            subsets = [rng.choice(len(frame), size, replace=False)
                       for _ in range(n_subsets)]
        base_sd = None
        for arm, block in samples.items():
            sd = float(np.mean([block[:, s].sum(axis=1).std(ddof=1) for s in subsets]))
            mean = float(np.mean([block[:, s].sum(axis=1).mean() for s in subsets]))
            if arm == "base":
                base_sd = sd
            rows.append({"arm": arm, "n_players": n_players, "n_subsets": len(subsets),
                         "expected_total_dk": mean, "total_sd": sd})
    out = pd.DataFrame(rows)
    base = out[out["arm"] == "base"].set_index("n_players")["total_sd"]
    out["inflation_vs_base"] = out.apply(
        lambda r: r["total_sd"] / base.get(r["n_players"], np.nan), axis=1)
    return out


def season_total_arms(models: dict, frame: pd.DataFrame, arms: tuple[str, ...],
                      draws: int = COMPOSITION_DRAWS, seed: int = 0) -> pd.DataFrame:
    """Season-total dk_pts per arm — the metric the whole question is decided on.

    Per-component R^2 is not the deliverable; a draftable season total is. Eleven heads
    each moving by a fraction of a percent can compound or cancel, and only composing them
    shows which.
    """
    y = realized_season_dk(frame)
    required = COUNT_HEADS + [f"{m}|{a}" for m, a in CONVERSION_HEADS]
    rows = []
    for arm in arms:
        present = [h for h in required if (h, arm) in models]
        # "This arm was never fitted" and "this arm is missing three of its eleven heads"
        # are different facts and used to be the same `continue`. A partially-migrated head
        # list lands squarely in the second case, and skipping it silently would drop the
        # arm from the table with no message — so an incomplete arm now raises.
        if not present:
            continue
        if len(present) != len(required):
            raise ValueError(
                f"arm {arm!r} has {len(present)} of {len(required)} heads fitted; "
                f"missing {sorted(set(required) - set(present))}. The head lists and the "
                f"fitted models disagree — refit rather than composing a partial chain")
        samples = compose_season_dk(models, frame, arm, draws, seed)
        pred = samples.mean(axis=0)
        rows.append({
            "arm": arm, "n": len(frame),
            "mae": float(np.abs(pred - y).mean()),
            "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
            "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "crps": float(crps_from_samples(samples, y).mean()),
            "pit_ks": ks_uniform(pit_from_samples(samples, y, seed)),
            **_bias(pred, y),
            **coverage_from_samples(samples, y)})
    return pd.DataFrame(rows)


def bonus_calibration(models: dict, frame: pd.DataFrame, realized: pd.DataFrame,
                      arms: tuple[str, ...], draws: int = 100,
                      seed: int = 0) -> pd.DataFrame:
    """Predicted against realized season bonus — the threshold half of the deliverable.

    The double-double bonus is a simultaneous threshold on five per-game counts, so it is
    convex in the rates and `E[bonus] != bonus(E[x])`. That makes it the one downstream
    quantity a mean-zero year effect can move on purpose rather than by accident: widening
    the predictive raises the probability of clearing a threshold.

    Drawn per posterior draw, converted to per-game rates by the player's realized games,
    and pushed through `expected_bonus` at `BONUS_GAME_OVERDISPERSION` — 0.025, the
    player-game value, not the 0.10 season-unit constant, per `CLAUDE.md`.
    """
    gp = np.clip(frame["gp"].to_numpy(float), 1, None)
    rows = []
    for arm in arms:
        if (COUNT_HEADS[0], arm) not in models:
            continue
        counts, made = _draw_components(models, frame, arm, draws, seed)
        pts = 2 * made["fg2m"] + 3 * made["fg3m"] + made["ftm"]
        per_game = np.stack([pts / gp, counts["reb"] / gp, counts["ast"] / gp,
                             counts["stl"] / gp, counts["blk"] / gp], axis=-1)
        flat = per_game.reshape(-1, len(BONUS_CATEGORIES))
        bonus = expected_bonus(flat, BONUS_GAME_OVERDISPERSION,
                               n_samples=BONUS_SAMPLES, seed=seed, chunk=5000)
        predicted = (bonus.reshape(per_game.shape[0], -1) * gp[None, :]).mean(axis=0)
        actual = realized["season_bonus"].to_numpy(float)
        rows.append({"arm": arm, "n": len(frame),
                     "predicted_bonus_per_game": float((predicted / gp).mean()),
                     "realized_bonus_per_game": float((actual / gp).mean()),
                     "bonus_mae": float(np.abs(predicted - actual).mean()),
                     **{f"bonus_{k}": v for k, v in _bias(predicted, actual).items()}})
    return pd.DataFrame(rows)


def realized_bonus(targets: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Season-total realized bonus per player-season, from the per-game logs.

    The bonus is per game, so it cannot be recovered from season totals — it has to be
    summed over the games as played.
    """
    played = targets[(targets["season_type"] == "regular") & (targets["played"] == 1)].copy()
    played["pts"] = (2 * played["fg2m"].to_numpy(float)
                     + 3 * played["fg3m"].to_numpy(float)
                     + played["ftm"].to_numpy(float))
    cats = (played[BONUS_CATEGORIES] >= 10).sum(axis=1)
    played["bonus"] = cats.map({0: 0.0, 1: 0.0, 2: 1.5, 3: 4.5, 4: 4.5, 5: 4.5})
    out = (played.groupby(["player_id", "season"], as_index=False)
           .agg(season_bonus=("bonus", "sum")))
    return frame[["player_id", "season"]].merge(out, on=["player_id", "season"],
                                                how="left").fillna({"season_bonus": 0.0})


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_stan = cfg.get("stan", {})
    cfg_terms = cfg_stan.get("season_terms", {}) or {}
    n_knots = int(cfg_stan.get("components", {}).get("spline_knots", 5))
    test_seasons = 2
    seed = int(cfg_stan.get("seed", 42))
    draws = int(cfg_terms.get("composition_draws", COMPOSITION_DRAWS))
    # The announced-rule-change path, read from config so filling it in is an edit rather
    # than a code change. Assigned to the module global because `apply_override` defaults
    # to it, which keeps the manual arm a one-liner at every call site.
    LEAGUE_OVERRIDE.update(cfg_terms.get("league_override", {}) or {})

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = build_component_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    full_train, test = split_seasons(design, test_seasons)
    train, val = split_seasons(full_train, test_seasons)

    specs, minutes_spec = selected_specs(out_dir)
    print(f"Season-term ablation: {len(design):,} player-seasons, "
          f"{design['season'].nunique()} target seasons")
    print(f"  {len(full_train):,} train / {len(test):,} test "
          f"({', '.join(sorted(test['season'].unique()))} held out)")
    print(f"  validation split: {len(train):,} fit / {len(val):,} select "
          f"({', '.join(sorted(val['season'].unique()))})")
    if specs:
        print(f"  base spec per head, read from stan_component_metrics.csv: "
              f"{', '.join(f'{h}={s}' for h, s in sorted(specs.items()))}")
    else:
        print(f"  /!\\  no stan_component_metrics.csv — falling back to "
              f"{DEFAULT_COUNT_SPEC}/{DEFAULT_CONVERSION_SPEC}. `fg3a` and `blk` are BELOW "
              f"their floor\n       under that spec for a documented curvature reason, so "
              f"read a season-term null on\n       those two as a spec artifact until "
              f"`make stan-components` has run.")
    print(f"  every fit uses metric={METRIC!r}: on the `blk` spline base it is 13.4 s "
          f"against 236.6 s\n  and treedepth saturation 0 against 35, for the same "
          f"posterior.")
    if LEAGUE_OVERRIDE:
        print(f"  manual league override ACTIVE for "
              f"{', '.join(sorted(LEAGUE_OVERRIDE))} — announced-in-advance rule changes "
              f"are\n  legitimate point-in-time information; see `LEAGUE_OVERRIDE`.")
    else:
        print("  manual league override empty, so `manual_override` == `base` by "
              "construction. The row\n  exists so filling in an announced rule change is a "
              "config edit, not new code.")
    print()

    counts, diag_counts, count_models = sweep_counts(
        train, val, test, full_train, specs, cfg_stan, n_knots)
    print("Count heads — held-out CRPS on the season total (lower is better):")
    print(counts.pivot_table(index="head", columns="arm", values="test_crps")
          .reindex(columns=["carry_forward", "base", "trend", "year", "trend_year"])
          .round(3).to_string())

    conversions, diag_conv, conv_models = sweep_conversions(
        train, val, test, full_train, specs, cfg_stan, n_knots)
    print("\nConversion heads — held-out beta-binomial NLL per row (lower is better):")
    print(conversions.pivot_table(index="head", columns="arm", values="test_nll")
          .reindex(columns=["carry_forward", "base", "trend", "year", "trend_year"])
          .round(4).to_string())

    minutes_design = build_minutes_design(cfg)
    m_full_train, m_test = split_availability(minutes_design, test_seasons)
    m_train, m_val = split_availability(m_full_train, test_seasons)
    minutes, diag_min = sweep_minutes(m_train, m_val, m_test, m_full_train,
                                      minutes_spec, cfg_stan, n_knots)
    print(f"\nMinutes head (spec {minutes_spec}) — CRPS in minutes:")
    print(minutes[["arm", "val_crps", "test_crps", "test_r2", "bias",
                   "coverage_80", "selected"]].round(3).to_string(index=False))

    av_design = availability_design(cfg)
    a_full_train, a_test = split_availability(av_design, test_seasons)
    a_train, a_val = split_availability(a_full_train, test_seasons)
    l2 = float(cfg.get("features", {}).get("availability", {}).get("glm_l2", 1.0))
    availability, diag_av = sweep_availability(a_train, a_val, a_test, a_full_train,
                                               cfg_stan, l2)
    print("\nAvailability head — CRPS in games, plus the role-interaction arms:")
    print(availability[["arm", "n_features", "val_crps", "test_crps", "test_r2", "bias",
                        "coverage_80", "selected"]].round(3).to_string(index=False))

    table = pd.concat([counts, conversions, minutes, availability], ignore_index=True)

    # ── the decisive downstream metric ────────────────────────────────────────
    models = {**count_models, **conv_models}
    arms = ("base", "trend", "year", "trend_year")
    totals = season_total_arms(models, test, arms, draws, seed)
    print("\n" + "=" * 78)
    print("SEASON-TOTAL dk_pts (bonus excluded, exact and linear in the eight "
          "scoring components)")
    print("=" * 78)
    print(totals[["arm", "mae", "rmse", "r2", "bias", "crps", "pit_ks",
                  "coverage_50", "coverage_80", "coverage_95"]].round(3).to_string(index=False))
    print("  Nominal coverage is 0.50 / 0.80 / 0.95. This is where a mean-zero year effect "
          "has to pay:\n  it cannot move MAE or R2 by construction, so if it does nothing "
          "here it does nothing.")

    spread = roster_spread(models, test, arms, draws=draws, seed=seed)
    if not spread.empty:
        print("\nRoster season-total SPREAD by portfolio size — the reason season effects "
              "matter at all:")
        print(spread.pivot_table(index="n_players", columns="arm", values="total_sd")
              .round(1).to_string())
        print(spread.pivot_table(index="n_players", columns="arm",
                                 values="inflation_vs_base").round(4).to_string())
        print("  A league shift is PERFECTLY correlated across players, so its "
              "contribution to a roster\n  total grows as N while independent error grows "
              "as sqrt(N). Compare against the shared-β\n  term measured on the same board "
              "in `stan_availability.board_correlation`: +0.2% on 15\n  players, +6.4% "
              "across all 911. `docs/predictions-plan.md` asserts season effects "
              "outrank\n  it by an order of magnitude; this table is that claim as a "
              "measurement.")

    bonus_actual = realized_bonus(targets, test)
    bonus = bonus_calibration(models, test, bonus_actual, arms, seed=seed)
    print("\nBonus-threshold calibration (per game, dk_pts):")
    print(bonus.round(4).to_string(index=False))
    print("  The bonus is a simultaneous threshold on five counts, so it is convex in the "
          "rates and is\n  the one downstream quantity a widening of the predictive moves "
          "on purpose.")

    sigma = sigma_against_league(
        table, Path(cfg["eda"]["output_dir"]) / "season_effects_summary.csv")
    if not sigma.empty:
        print("\nDoes the fitted sigma_year recover the league movement measured "
              "independently?")
        print(sigma[["head", "link", "sigma_year_pct", "league_yoy_sd_pct",
                     "league_level_resid_sd_pct", "ratio_to_yoy_sd",
                     "response_multiplier"]].round(3).to_string(index=False))
        print("  `sigma_year` is fitted by NUTS on player-season rows and knows nothing "
              "about\n  `make season-effects`, which measures the league rate directly as "
              "totals over totals.\n  On a LOG link the two are the same number in the "
              "same units, so the count rows are a\n  like-for-like check; the logit rows "
              "differ by 1/(1-p) and are a sanity check only.\n"
              "  `response_multiplier` is E[exp(sigma*z)] — the Jensen inflation a "
              "mean-zero term\n  puts on the RESPONSE scale, which is the one way it does "
              "move the mean.")

    _verdicts(table, totals)

    diag = diagnostics_frame(diag_counts + diag_conv + diag_min + diag_av)
    artifacts = {
        "metrics": (table, out_dir / "season_term_metrics.csv"),
        "sigma_vs_league": (sigma, out_dir / "season_term_sigma_vs_league.csv"),
        "season_total": (totals, out_dir / "season_term_season_total.csv"),
        "roster_spread": (spread, out_dir / "season_term_roster_spread.csv"),
        "bonus": (bonus, out_dir / "season_term_bonus.csv"),
        "diagnostics": (diag, out_dir / "season_term_diagnostics.csv"),
    }
    paths = {}
    for name, (frame, dest) in artifacts.items():
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")

    bad = diag[~diag["converged"]]
    print(f"\nSampler over {len(diag)} fits: max R-hat {diag['max_rhat'].max():.4f}, "
          f"{int(diag['divergences'].sum())} divergences, "
          f"{diag['wall_clock_s'].sum() / 60:.1f} min total")
    if len(bad):
        print(f"/!\\  {len(bad)} fits failed a convergence bar: "
              f"{', '.join(bad['label'].head(10))}")
    return paths


# Each head's own league series in `season_effects_summary.csv`. The conversion and
# minutes/availability heads sit on a logit link and their league series is a percentage
# change in the rate, so the two are the same quantity only up to a 1/(1-p) factor — the
# comparison is a sanity check on those rows and a like-for-like one on the counts.
LEAGUE_SERIES = {c: c for c in COUNT_HEADS} | {
    f"{m}|{a}": f"{m}_pct" for m, a in CONVERSION_HEADS} | {
    "min": "minutes_share", "gp": "gp_share [all]"}


def sigma_against_league(table: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    """Does the fitted `sigma_year` recover the league movement measured independently?

    A check with a defined answer rather than a hopeful comparison, in the same spirit as
    "the fitted dispersion lands at 20-30x, independently recovering the ~20x in
    `availability_profile.csv`". `sigma_year` is fitted on player-season rows by NUTS and
    knows nothing about `make season-effects`, which measures the league rate directly as
    totals over totals. On a **log** link a sigma of s is a fractional movement of s, so
    `100 * sigma_year` and `yoy_sd_pct` are the same number in the same units.

    Two reasons they should not match exactly, both of which the columns make visible:
    the year effect is the season shift *left over after* the covariates — chiefly the
    player's own prior rate, which already carries last season's league level — and on a
    logit link the units differ by 1/(1-p).
    """
    if not summary_path.exists():
        return pd.DataFrame()
    league = pd.read_csv(summary_path).set_index("quantity")
    rows = []
    for _, r in table[table["arm"] == "year"].iterrows():
        quantity = LEAGUE_SERIES.get(str(r["head"]))
        if quantity is None or quantity not in league.index:
            continue
        rows.append({
            "head": r["head"], "league_series": quantity,
            "link": "log" if r["kind"] == "count" else "logit",
            "sigma_year_pct": 100.0 * float(r.get("year_sigma_year", np.nan)),
            "league_yoy_sd_pct": float(league.loc[quantity, "yoy_sd_pct"]),
            "league_level_resid_sd_pct": float(
                league.loc[quantity, "level_resid_sd_pct"]),
            "response_multiplier": float(r.get("year_response_multiplier", np.nan))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ratio_to_yoy_sd"] = out["sigma_year_pct"] / out["league_yoy_sd_pct"]
    return out.sort_values("league_yoy_sd_pct", ascending=False).reset_index(drop=True)


def _verdicts(table: pd.DataFrame, totals: pd.DataFrame) -> None:
    """Print the three checks the ablation exists to make, each with its own decision rule."""
    print("\n" + "=" * 78)
    print("VERDICTS")
    print("=" * 78)

    chosen = table[table["selected"]]
    picked = chosen["arm"].value_counts()
    print(f"Validation-selected arm across {len(chosen)} heads: "
          f"{', '.join(f'{a} x{n}' for a, n in picked.items())}")

    # 1. A year effect must NOT move point accuracy.
    year_rows = table[table["arm"] == "year"].set_index("head")
    base_rows = table[table["arm"] == "base"].set_index("head")
    common = year_rows.index.intersection(base_rows.index)
    if len(common) and "test_r2" in table:
        dr2 = (year_rows.loc[common, "test_r2"] - base_rows.loc[common, "test_r2"]).dropna()
        print(f"\n1. Year effect vs base, held-out R2: median {dr2.median():+.5f}, "
              f"max |delta| {dr2.abs().max():.5f} ({dr2.abs().idxmax()})")
        print("   A year effect does TWO things and only one of them can move this "
              "number.\n"
              "     (a) at prediction time it is mean-zero, so it widens the predictive "
              "and cannot\n         move point accuracy at all;\n"
              "     (b) in the fit it absorbs season, so beta is estimated WITHIN season "
              "— which is\n         the repo's own standing rule and legitimately can "
              "move it, a little.\n"
              "   So a small move is expected and good; a large one means the term is "
              "soaking up\n   something that is not league movement.")

    # 2. A trend must help on `fg3a` and essentially nowhere else.
    trend_rows = table[table["arm"] == "trend"].set_index("head")
    common = trend_rows.index.intersection(base_rows.index)
    if len(common) and "test_crps" in table:
        d = (base_rows.loc[common, "test_crps"]
             - trend_rows.loc[common, "test_crps"]).dropna().sort_values(ascending=False)
        print(f"\n2. Trend vs base, held-out CRPS improvement (positive = trend helps):")
        print("   " + ", ".join(f"{h} {v:+.3f}" for h, v in d.items()))
        helped = [h for h, v in d.items() if v > 0]
        print(f"   helps {len(helped)} of {len(d)} heads. The league measurement says "
              f"`fg3a` and nothing\n   else; if it helps everywhere, suspect it is fitting "
              f"the last two seasons.")

    # 3. The oracle ceiling.
    oracle = table[table["arm"] == "oracle_league"].set_index("head")
    common = oracle.index.intersection(base_rows.index)
    if len(common) and "test_mae" in table:
        gain = ((base_rows.loc[common, "test_mae"] - oracle.loc[common, "test_mae"])
                / base_rows.loc[common, "test_mae"] * 100).dropna().sort_values(ascending=False)
        print(f"\n3. Oracle league override — the CEILING on any league-level term, "
              f"as % of MAE:")
        print("   " + ", ".join(f"{h} {v:.1f}%" for h, v in gain.items()))
        print("   A perfect preseason announcement, per season, per component. No fitted "
              "form can beat\n   this, so it bounds the entire question.")

    if not totals.empty and "base" in set(totals["arm"]):
        base = totals[totals["arm"] == "base"].iloc[0]
        print(f"\n4. Season-total dk_pts coverage, base arm: "
              f"{base['coverage_50']:.3f} / {base['coverage_80']:.3f} / "
              f"{base['coverage_95']:.3f} against a nominal 0.50 / 0.80 / 0.95.")
        for _, row in totals[totals["arm"] != "base"].iterrows():
            print(f"   {row['arm']:<12} {row['coverage_50']:.3f} / "
                  f"{row['coverage_80']:.3f} / {row['coverage_95']:.3f}   "
                  f"CRPS {row['crps']:.2f} vs {base['crps']:.2f}")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
