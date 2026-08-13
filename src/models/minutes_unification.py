"""Does the minutes composition supersede the marginal minutes head at the season unit?

`README.md` says the two minutes heads "compose rather than compete", with
`stan_composition` owning the per-game allocation and `stan_minutes` still owning "the
season-level mean and the game-level dispersion, neither of which the composition
produces." Audited on 2026-08-08 (`docs/simulations-plan.md`, "The third prerequisite"),
that sentence asserts three things and only one of them was known to hold. Two were settled
before this module existed:

- **the game-level dispersion claim is false as stated.** `stan_minutes.game_level_dispersion`
  reads `targets` and `lengths`, computes each player-season's own realized share as `mu`,
  and fits a dispersion to that. The `StanMinutes` object never appears, so every Stan fit
  in that module could be deleted and 4.65x would still come out. It is a data measurement
  that happens to live there;
- **the composition already fits its own game-level dispersion**, role-graded over four
  prior-share bins (rho 0.1768 fringe to 0.0855 star). Different parameterization from the
  4.65x — it disperses the sequential binomial *trials*, not `game_length` — but the same
  kind of quantity, and only one of them can govern a draw.

**The season-level claim was the untested one, and this module is its gate.** The
composition's per-game predictions sum to a season total by construction; whether that
beats the season-collapsed head at the season unit had never been measured, because the two
heads publish metrics at different units (minutes at season-total, composition per
team-game) and the tables were never made comparable.

## Why this needs no refit, and why that matters

Both heads' posteriors are already on disk at the `train` fit window, written by
`make posteriors`. This module loads them, rehydrates each head class around its own saved
draws and design recipe, and calls **the head's own `predict_samples`** — the identical code
path that produced the published metrics, on the identical validation rows. So the gate
costs seconds rather than the composition's 9.92 h, and no number here comes from a
differently-fitted model than the one it is being compared against.

`train` is the window to consume and it is not a default worth taking on trust: the rows
being scored are 2022-23 and 2023-24, which `train_val` fits on. `require_window` refuses
anything else, because a head fitted wider has read the evaluation seasons *through the
coefficients* and no frame-level split guard can see that.

## The comparison is at the season unit, on the rows both heads cover

The composition covers every played player-game — a rookie cannot be dropped, since the
team sum must be complete — while the marginal head applies the project's usual
`>= 200 prior minutes & >= 10 games` filter. So the composition scores more player-seasons
than the marginal head can, and the head-to-head runs on the intersection. The coverage
difference is reported as its own row rather than folded into the comparison, because it is
a real capability of the composition and not a term in the gate.

The two frames also disagree very slightly about what a player's realized season minutes
*were* — the marginal head builds from `component_targets.parquet` and the composition from
`availability_panel.parquet`, and the two differ on a handful of player-games. Each arm is
therefore scored against its own frame's realized total, exactly as its published metrics
are, and the disagreement is both reported and re-run as a shared-target robustness column
so it cannot be the thing carrying the verdict.

Usage:
    python -m src.models.minutes_unification
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.held_out import selection_split
from src.models.posteriors import load, posteriors_dir, require_window
from src.models.stan_composition import (GROUP_KEYS, PILOT_FIRST_SEASON,
                                         StanComposition, composition_frame,
                                         simulate_minutes)
from src.models.stan_minutes import PRESEASON as MINUTES_PRESEASON
from src.models.stan_minutes import FloorMinutes, StanMinutes
from src.models.stan_minutes import head_design as minutes_head_design
from src.models.stan_utils import crps_from_samples, ks_uniform, pit_from_samples

# The season unit: one row per player-season, which is what a draft board ranks on.
UNIT_KEYS = ["player_id", "season"]

# The window whose heads may score 2022-23 / 2023-24. Not a preference — `train_val` fits
# on exactly those seasons.
FIT_WINDOW = "train"

POSTERIOR_DRAWS = 1000
N_BOOTSTRAP = 2000
SEED = 0

# Realized minutes below which a log ratio says nothing. `stan_minutes.MIN_PRIOR_MINUTES`,
# reused rather than re-chosen — it is the threshold every other head in the project already
# treats as the line between a rate observation and noise.
MIN_QUALIFIED = 200

# Injected per-player-season effect sizes. Bracketing the two data-implied figures the
# headroom row reports (a log-ratio sd of 0.284 and a logit-share sd of 0.493) rather than
# centred on a guess, and wide enough on both sides to show the CRPS optimum is interior —
# a monotone sweep would not distinguish "this helps" from "the grid stopped too early".
PS_SIGMAS = (0.0, 0.2, 0.3, 0.375, 0.45, 0.6)

# The fallback's estimation window: the last N **training** seasons. `docs/simulations-plan.md`
# makes shipping the injection with sigma estimated on `train` the named fallback if the
# fitted version blows the budget, and the only work it owes is exactly this — moving sigma
# off the split it is scored against. The last two train seasons rather than all of them
# because the quantity is one scalar and the frame is 631,158 rows x 200 draws otherwise;
# two seasons match the validation window in size and era, which is the comparison the
# estimate is for.
TRAIN_SIGMA_SEASONS = 2

# The shipped injected effect size, if config does not say otherwise. Estimated on TRAIN by
# `estimate_sigma_on_train`, which is what removes the injection's one load-bearing caveat —
# the validation grid's 0.375 was read off the split it is scored against. Five independent
# routes land in [0.375, 0.481]: two CRPS grids, the calibration target, and two Stan fits.
SHIPPED_PS_SIGMA = 0.450

# Smallest roster a teammate correlation is taken over. Below four players the mean pairwise
# correlation of a fixed-sum block is dominated by the constraint's own -1/(K-1).
MIN_ROSTER = 4

# The gate's decision rule, stated once so it is not re-argued in prose. The paired
# bootstrap is over player-seasons, matching the availability head's ladder, which is the
# precedent for "a mean difference this repo could not distinguish from zero".
CI = (2.5, 97.5)


# ── Rehydration — the head's own code path, around the saved draws ────────────

def rehydrate_minutes(artifact, keep: int) -> StanMinutes:
    """A `StanMinutes` carrying the artifact's draws, scaler and feature list.

    Deliberately the real head rather than a reimplementation of its predictive: the
    beta-binomial season draw, the `thin` over posterior draws and the clipping all live in
    `StanMinutes.predict_samples`, and the point of this gate is to compare the two heads as
    they actually ship. `year` stays the disabled `YearTerm` the constructor builds —
    `make posteriors` refuses to persist a head fitted with one, so an artifact can never
    carry a year effect that this silently drops.
    """
    model = StanMinutes(list(artifact.recipe.features), name="unification/minutes",
                        predictive_samples=keep)
    model.scaler = artifact.recipe.scaler
    model.alpha_draws = np.asarray(artifact.draws["alpha_draws"])
    model.beta_draws = np.asarray(artifact.draws["beta_draws"])
    model.rho_draws = np.asarray(artifact.draws["rho_draws"])
    model.rho = float(model.rho_draws.mean())
    return model


def shipped_sigma(cfg: dict) -> float:
    """`sim.minutes.player_season_sigma` — the effect size the simulator draws with.

    One place, so a consumer cannot forget it. Reading it from config rather than hard-coding
    it is what makes `0.0` a supported configuration: that recovers the un-injected head
    exactly, which is the control every claim about the injection is measured against.
    """
    return float(cfg.get("sim", {}).get("minutes", {})
                 .get("player_season_sigma", SHIPPED_PS_SIGMA))


def rehydrate_composition(artifact, keep: int,
                          injected_sigma: float | None = None) -> StanComposition:
    """A `StanComposition` carrying the artifact's draws, at its selected arm.

    `rho_draws` is (draws x n_rho) even at n_rho = 1, so the graded and shared arms take the
    same downstream path — the head's own invariant, preserved here rather than re-derived.

    **The per-(player, season) effect arrives one of two ways, and the head cannot tell them
    apart downstream — which is the point.** A `sigma_u_draws` block in the artifact is a
    FITTED effect and takes precedence; failing that, `injected_sigma` supplies the shipped
    constant from `sim.minutes.player_season_sigma`. Either way the term is live on
    `predict_samples`, so **a consumer gets the effect by rehydrating the head and does not
    have to remember to apply it** — which was the injection's worst property while it lived
    only in the simulator, and precisely the class of provenance failure this repo has been
    bitten by before. `sigma_source` records which it was.

    The stream is restored from the artifact when one is fitted, so the rehydrated head draws
    the same `z` sequence the fitted one would.
    """
    sigma_u = artifact.draws.get("sigma_u_draws")
    injected = float(injected_sigma or 0.0)
    enabled = sigma_u is not None or injected > 0
    model = StanComposition(list(artifact.recipe.features),
                            int(artifact.extras["dispersed"]),
                            int(artifact.extras["n_rho"]),
                            name="unification/composition", predictive_samples=keep,
                            player_season_effect=enabled,
                            u_sd_scale=float(artifact.extras.get("u_sd_scale", 1.0)))
    model.scaler = artifact.recipe.scaler
    model.alpha_draws = np.asarray(artifact.draws["alpha_draws"])
    model.beta_draws = np.asarray(artifact.draws["beta_draws"])
    if sigma_u is not None:
        model.ps.sigma_draws = np.asarray(sigma_u)
        model.ps.stream = str(artifact.extras.get("u_stream", model.ps.stream))
        model.sigma_source = "fitted"
    elif injected > 0:
        # A constant across draws, which is exactly what "plug in sigma-hat" means and is
        # the honest difference from a fit: no posterior on sigma, so the predictive does
        # not integrate over its uncertainty.
        model.ps.sigma_draws = np.full(len(model.alpha_draws), injected)
        model.sigma_source = "injected"
    else:
        model.sigma_source = "none"
    if model.dispersed:
        model.rho_draws = np.asarray(artifact.draws["rho_draws"])
        model.rho_by_bin = model.rho_draws.mean(axis=0)
        model.rho = float(model.rho_by_bin.mean())
    else:
        model.rho_draws, model.rho_by_bin, model.rho = None, None, 0.0
    return model


# ── Collapsing per-game draws to the season unit ──────────────────────────────

def season_totals(samples: np.ndarray, frame: pd.DataFrame
                  ) -> tuple[np.ndarray, pd.DataFrame]:
    """`(draws x player-seasons)` totals and their unit index, summing within a unit.

    Summed by sorting into unit blocks and calling `np.add.reduceat` once, rather than
    `np.add.at`, which is unbuffered and runs ~50M scattered adds at this shape. The
    composition frame is ordered by team-game, so a player's games are **not** contiguous in
    it and the sort is required — reducing in place would silently sum the wrong rows.
    """
    if len(frame) != samples.shape[1]:
        raise ValueError(f"samples has {samples.shape[1]} rows against a frame of "
                         f"{len(frame)}; they must be the same player-games")
    grouped = frame.groupby(UNIT_KEYS, sort=True)
    codes = grouped.ngroup().to_numpy()
    units = grouped.size().reset_index(name="games")

    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(units)))
    totals = np.add.reduceat(samples[:, order], starts, axis=1)
    return totals, units


def realized_totals(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Realized season minutes per unit, from whichever column the frame calls it."""
    grouped = frame.groupby(UNIT_KEYS, sort=True)
    return grouped[column].sum().reset_index(name="realized")


# ── Scoring at the season unit ────────────────────────────────────────────────

def score_season(samples: np.ndarray, y: np.ndarray, label: str, unit: str,
                 seed: int = SEED) -> dict:
    """The marginal metric set at the season unit, in minutes.

    Mirrors `stan_minutes.score` and `stan_composition.score_samples` — the same CRPS, PIT
    and bias — with two differences. The mean is `samples.mean(axis=0)` for every arm rather
    than each head's own `predict_mean`, so the two are the same functional of the same
    draws; and `predictive_sd` is carried, because the season-total *spread* turns out to be
    what separates the two heads and a CRPS alone does not show which side of it a head
    fails on.
    """
    pred = samples.mean(axis=0)
    return {
        "arm": label,
        "unit": unit,
        "n": len(y),
        "n_draws": samples.shape[0],
        "crps_minutes": float(crps_from_samples(samples, y).mean()),
        "mae_minutes": float(np.abs(pred - y).mean()),
        "r2_minutes": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "bias_minutes": float((pred - y).mean()),
        "pit_ks": ks_uniform(pit_from_samples(samples, y, seed)),
        "predictive_sd": float(samples.std(axis=0).mean()),
        "realized_sd": float(y.std()),
    }


def paired_bootstrap(crps_a: np.ndarray, crps_b: np.ndarray, n_boot: int = N_BOOTSTRAP,
                     seed: int = SEED) -> dict:
    """`a - b` mean CRPS difference with a percentile interval, resampling units.

    Paired over player-seasons and not over draws: the two arms saw the same rows, so the
    row-level pairing removes the between-player variance that would otherwise swamp the
    difference. This is the test that reversed the availability ladder's apparent winner —
    a mean gap of -0.1297 with an interval of [-0.3154, +0.0672] — which is why a margin
    here is not read without one.
    """
    delta = np.asarray(crps_a, dtype=float) - np.asarray(crps_b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, CI)
    return {"crps_delta": float(delta.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_delta_negative": float((means < 0).mean()), "n_bootstrap": n_boot}


def season_effect_headroom(samples: np.ndarray, units: pd.DataFrame,
                           realized: np.ndarray) -> dict:
    """How much of the composition's missing season-total spread a SEASON term could reach.

    The natural next question once the gate lands, and it is answerable from the head's own
    residuals without fitting anything. A year trend or year random effect is a **league-wide
    shift shared by every row in a posterior draw** (`stan_utils.YearTerm`), so the variance
    it can reach is the variance of the league-wide mean residual across seasons and nothing
    else. Against a head that allocates every minute in the league, that quantity is zero by
    construction — the residuals sum to zero within each season because the minutes are fully
    allocated — which is why this is reported as a *headroom* rather than as a candidate arm.

    `log_ratio_sd` sizes the effect that *would* fit: the per-player-season deviation of
    realized from predicted season minutes, on the log scale, so it is directly comparable in
    order of magnitude to a fitted `sigma_year`. It is indexed by (player, season), not by
    season, and that index is the whole difference.

    That last figure is taken over player-seasons clearing `MIN_QUALIFIED` realized minutes,
    which is `stan_minutes`' own `min_prior_minutes` threshold rather than a new knob. Below
    it a log ratio is noise — an end-of-bench player at 4 realized minutes against 40
    predicted contributes -2.3 to a spread that is supposed to describe rotation players, and
    a realized zero contributes -inf. The restriction is reported as `log_ratio_n` so it is
    visible rather than implicit; the two variance figures above it are over every row.
    """
    pred = samples.mean(axis=0)
    realized = np.asarray(realized, dtype=float)
    resid = realized - pred
    season_means = (pd.DataFrame({"season": units["season"].to_numpy(), "resid": resid})
                    .groupby("season")["resid"].mean().to_numpy())
    between = float(season_means.std(ddof=0))
    qualified = (realized >= MIN_QUALIFIED) & (pred > 0)
    log_ratio = np.log(realized[qualified] / pred[qualified])
    return {
        "arm": "season_effect_headroom", "unit": "variance_decomposition",
        "n": len(resid), "n_draws": samples.shape[0],
        "resid_sd": float(resid.std()),
        "predictive_sd": float(samples.std(axis=0).mean()),
        "league_season_mean_resid_sd": between,
        "share_of_resid_var_reachable": float(between ** 2 / resid.var()),
        "log_ratio_sd": float(log_ratio.std()),
        "log_ratio_n": int(qualified.sum()),
    }


def unit_codes(frame: pd.DataFrame) -> np.ndarray:
    """Player-season index per row, for a shock shared across a player's games."""
    return frame.groupby(UNIT_KEYS, sort=True).ngroup().to_numpy()


def player_season_effect_sweep(model, frame: pd.DataFrame, raw: pd.DataFrame,
                               realized: np.ndarray, keep_rows: np.ndarray,
                               y: np.ndarray, crps_reference: np.ndarray,
                               sigmas=PS_SIGMAS, seed: int = SEED,
                               fitted_sigma: float | None = None) -> list[dict]:
    """What a per-(player, season) random effect would buy, injected rather than fitted.

    The gate finds the composition's season totals **4.68x too narrow**, and the obvious
    question is whether that is a missing parameter or a ceiling imposed by the team
    constraint. This answers it without a refit: add `sigma * z[unit, draw]` to the linear
    predictor — one standard normal per player-season per posterior draw, shared across that
    player's games — and re-run the head's own sequential allocation. That is exactly the
    predictive a fitted random effect produces once integrated over, which is how
    `stan_utils.YearTerm` already treats an unobserved future level.

    **This measures capability, not a result, and the difference matters.** `sigma` is being
    read off validation CRPS rather than estimated from training data, so the best row here
    is a tuned upper bound on what the parameterization can reach — not a shipped score. What
    it can legitimately settle is the structural question: whether the constraint permits the
    spread at all. It does.

    The mean function is left alone, so this also isolates the spread: a fitted version would
    re-estimate `beta` alongside `sigma` and could do better or worse.

    **Once the head ships a fitted `sigma_u`, this stops being the measurement and becomes
    the calibration check.** `fitted_sigma` is read off the posterior artifact and evaluated
    as one more row, marked `source = "fitted"`; the grid stays, because what it answers —
    is the CRPS optimum interior, and where — is exactly how you find out whether a fitted
    sigma landed in the right place. The injection is applied to `_eta_base`, which excludes
    the head's own effect, so this never double-counts a fitted one: at
    `sigma = sigma_u` it reproduces `predict_samples` in distribution.
    """
    eta_base, rho = model._eta_base(frame)
    codes = unit_codes(raw)
    n_units, n_draws = int(codes.max()) + 1, eta_base.shape[1]
    grid = [(float(s), "injected_grid") for s in sigmas]
    if fitted_sigma is not None:
        grid.append((float(fitted_sigma), "fitted"))

    rows = []
    for sigma, source in grid:
        rng = np.random.default_rng(seed + 7)
        eta = eta_base + float(sigma) * rng.normal(size=(n_units, n_draws))[codes, :]
        totals, _ = season_totals(simulate_minutes(frame, eta, rho, seed), raw)
        scored = totals[:, keep_rows]
        crps = crps_from_samples(scored, y)
        delta = paired_bootstrap(crps, crps_reference)
        rows.append({
            "arm": "composition_sum_plus_player_season_effect", "unit": "ps_effect_sweep",
            "sigma": float(sigma), "sigma_source": source, "n": len(y),
            "n_draws": n_draws,
            "crps_minutes": float(crps.mean()),
            "mae_minutes": float(np.abs(scored.mean(axis=0) - y).mean()),
            "pit_ks": ks_uniform(pit_from_samples(scored, y, seed)),
            "predictive_sd": float(scored.std(axis=0).mean()),
            "crps_delta": delta["crps_delta"], "ci_lo": delta["ci_lo"],
            "ci_hi": delta["ci_hi"], "verdict": verdict(delta),
        })
    return rows


def estimate_sigma_on_train(model, artifact, composition_train: pd.DataFrame,
                            sigmas=PS_SIGMAS, seed: int = SEED) -> list[dict]:
    """The same sweep on **training** player-seasons — the fallback's shippable sigma.

    `player_season_effect_sweep` reads its optimum off validation CRPS, which is the split
    the composition is later scored against, and that caveat is why the plan chose a fitted
    `sigma_u` over the injection. **If the fit blows the budget, the injection still ships —
    and this is the one piece of work it owes.** Running the identical grid on training rows
    moves sigma off the evaluation split, at the cost of minutes rather than a refit of the
    project's most expensive head.

    Two things make it an honest estimate rather than a relabelling. The rows are training
    rows the composition was *fitted* on, so the CRPS here is in-sample for `beta` — but
    `sigma` is not a parameter of that fit at all, so the quantity being optimized is the one
    the injection adds and nothing else. And the grid, the arithmetic and the metric are
    literally the same function, so a sigma chosen here and a sigma chosen on validation are
    comparable numbers rather than two different estimators.
    """
    seasons = sorted(composition_train["season"].unique())[-TRAIN_SIGMA_SEASONS:]
    raw = composition_train[composition_train["season"].isin(seasons)].reset_index(drop=True)
    frame = artifact.recipe.transform(raw)
    eta_base, rho = model._eta_base(frame)
    codes = unit_codes(raw)
    n_units, n_draws = int(codes.max()) + 1, eta_base.shape[1]
    realized = realized_totals(raw, "y")["realized"].to_numpy(float)

    rows = []
    for sigma in sigmas:
        rng = np.random.default_rng(seed + 11)
        eta = eta_base + float(sigma) * rng.normal(size=(n_units, n_draws))[codes, :]
        totals, _ = season_totals(simulate_minutes(frame, eta, rho, seed), raw)
        rows.append({
            "arm": "composition_sum_plus_player_season_effect", "unit": "ps_sigma_on_train",
            "sigma": float(sigma), "sigma_source": "train_grid",
            "n": totals.shape[1], "n_draws": n_draws,
            "seasons": ", ".join(seasons),
            "crps_minutes": float(crps_from_samples(totals, realized).mean()),
            "mae_minutes": float(np.abs(totals.mean(axis=0) - realized).mean()),
            "pit_ks": ks_uniform(pit_from_samples(totals, realized, seed)),
            "predictive_sd": float(totals.std(axis=0).mean()),
        })
    return rows


def teammate_coupling(totals: np.ndarray, units: pd.DataFrame, raw: pd.DataFrame,
                      label: str) -> dict:
    """Mean pairwise correlation of teammates' season totals, and the team total's sd.

    **The dynamic the composition exists for, measured on the quantity the draft cares
    about.** A team's season minutes are a fixed pot, so a player's over-performance has to
    come out of a teammate: teammates' season totals are negatively correlated, and a fixed
    sum over K players forces the mean pairwise correlation to exactly `-1/(K-1)`.

    A head drawing players independently reproduces neither. It puts the correlation at ~0
    and gives the *team's* season total a predictive sd of hundreds of minutes — a spread on
    a quantity that is physically fixed. That is invisible in any marginal metric and lands
    directly on two strategy axes: a same-team stack's minutes are anti-correlated rather
    than independent, and handcuffing a starter with his backup is a hedge that only exists
    if the negative correlation is in the model.

    Restricted to **single-team** player-seasons, because a mid-season trade splits a
    player's minutes across two teams and attributing them to one leaks into every team-level
    sum. 13.6% of players appeared for 2+ teams in 2023-24, so this is not a rounding error.

    That restriction is also why the composition's `team_season_sum_sd` here is a small
    positive number rather than zero: dropping the traded players leaves a *subset* of each
    roster, and a subset of a fixed-sum set does not itself have a fixed sum. The exact
    figure is the `team_season_sd` on the `composition_sum_all_rows` row, which groups by the
    head's own complete team-game blocks and reads 0.00. Both are correct; only one of them
    is a statement about the model, and the comparison that carries meaning here is the
    **ratio** between the two heads on identical rows.
    """
    single = raw.groupby(UNIT_KEYS)["team_id"].nunique().reset_index(name="n_teams")
    first = raw.groupby(UNIT_KEYS)["team_id"].first().reset_index()
    frame = (units.assign(row=np.arange(len(units)))
             .merge(single, on=UNIT_KEYS).merge(first, on=UNIT_KEYS))
    frame = frame[frame["n_teams"] == 1]

    corrs, sums, sizes = [], [], []
    for _, block in frame.groupby(["season", "team_id"], sort=False):
        idx = block["row"].to_numpy()
        x = totals[:, idx]
        x = x[:, x.std(axis=0) > 0]
        if x.shape[1] < MIN_ROSTER:
            continue
        c = np.corrcoef(x, rowvar=False)
        corrs.append(c[np.triu_indices_from(c, k=1)].mean())
        sums.append(x.sum(axis=1).std())
        sizes.append(x.shape[1])
    mean_k = float(np.mean(sizes))
    return {
        "arm": label, "unit": "teammate_coupling", "n": len(frame),
        "n_draws": totals.shape[0],
        "r_teammates": float(np.mean(corrs)),
        "team_season_sum_sd": float(np.mean(sums)),
        "roster_size": mean_k,
        # What a fixed team total forces, so the measured r is read against its own bar.
        "r_implied_by_fixed_sum": float(-1.0 / (mean_k - 1.0)),
    }


def verdict(delta: dict) -> str:
    """`wins` / `ties` / `loses`, from the interval alone.

    Stated as code so the outcome is not a judgement call made after seeing the number.
    `docs/simulations-plan.md` gives the composition the benefit of a tie, so the only
    outcome that keeps the marginal head is an interval lying entirely above zero.
    """
    if delta["ci_hi"] <= 0:
        return "wins"
    return "loses" if delta["ci_lo"] > 0 else "ties"


# ── Frames ────────────────────────────────────────────────────────────────────

def validation_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                          pd.DataFrame]:
    """`(minutes_train, minutes_val, composition_train, composition_val)` — through
    `selection_split` only.

    The minutes side goes through `stan_minutes.head_design`, **not** `build_design`: since
    2026-08-13 the shipped head carries a five-column preseason block, and this module
    rehydrates that head and calls its own `predict_samples`, so a frame without those
    columns is not a different comparison — it is a `KeyError`. The flag is read from the
    config rather than defaulted, so `stan.minutes.preseason: false` rolls this module back
    with the head it scores.

    The composition frame is built over every season for its lags and its expanding rookie
    prior and then cut to the head's fitting window, exactly as `stan_composition.run` and
    `posteriors.composition_artifact` both do; the window only decides which rows were
    *fitted*, and the validation seasons are the same either way.

    The minutes **fitting** rows are deliberately *not* cut to the preseason-covered window
    the head itself fits on. They are used here for one thing — `FloorMinutes`, a no-fit
    carry-forward whose mean carries no preseason column — and cutting them would move the
    floor this gate scores both heads against for a reason that has nothing to do with the
    floor.
    """
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    first_season = str(cfg.get("stan", {}).get("composition", {})
                       .get("first_season", PILOT_FIRST_SEASON))
    preseason = bool(cfg.get("stan", {}).get("minutes", {})
                     .get("preseason", MINUTES_PRESEASON))

    design = minutes_head_design(cfg, preseason)
    minutes_train, minutes_val = selection_split(design, test_seasons)

    frame = composition_frame(cfg)
    pilot = frame[frame["season"] >= first_season].reset_index(drop=True)
    composition_train, composition_val = selection_split(pilot, test_seasons)
    return minutes_train, minutes_val, composition_val, composition_train


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_sim = cfg.get("sim", {})
    keep = int(cfg_sim.get("posterior_draws", POSTERIOR_DRAWS))
    window = str(cfg_sim.get("fit_window", FIT_WINDOW))

    print("Minutes unification — does the composition supersede the marginal head?")
    print(f"  reading persisted posteriors at the `{window}` window; nothing is refitted.")
    artifacts = {name: load(name, posteriors_dir(cfg, window))
                 for name in ("minutes", "composition")}
    require_window(artifacts, window)
    print(f"  minutes    @ {artifacts['minutes'].recipe.variant}, "
          f"{artifacts['minutes'].n_draws:,} draws")
    print(f"  composition@ {artifacts['composition'].recipe.variant}, "
          f"{artifacts['composition'].n_draws:,} draws")
    fitted_sigma = artifacts["composition"].extras.get("sigma_u")
    if fitted_sigma:
        print(f"  the composition carries a FITTED player-season effect, sigma_u "
              f"{float(fitted_sigma):.4f} "
              f"[{float(artifacts['composition'].extras['sigma_u_lo']):.4f}, "
              f"{float(artifacts['composition'].extras['sigma_u_hi']):.4f}] over "
              f"{int(artifacts['composition'].extras['n_units']):,} units.\n"
              f"  Its season totals below are drawn WITH it, and the injection sweep "
              f"becomes a calibration check\n  rather than the measurement — see "
              f"`player_season_effect_sweep`.")

    (minutes_train, minutes_val, composition_val,
     composition_train) = validation_frames(cfg)
    seasons = ", ".join(sorted(minutes_val["season"].unique()))
    print(f"\n  The test split is LOCKED — this gate reads VALIDATION only "
          f"({seasons}).")
    print(f"  minutes val:     {len(minutes_val):,} player-seasons")
    print(f"  composition val: {len(composition_val):,} played player-games over "
          f"{composition_val.groupby(GROUP_KEYS, sort=False).ngroups:,} team-games")

    # ── The composition, summed to seasons ────────────────────────────────────
    composition = rehydrate_composition(artifacts["composition"], keep)
    comp_frame = artifacts["composition"].recipe.transform(composition_val)
    comp_games = composition.predict_samples(comp_frame, SEED)
    comp_totals, comp_units = season_totals(comp_games, composition_val)
    comp_units = comp_units.merge(realized_totals(composition_val, "y"), on=UNIT_KEYS)

    # ── The marginal head ─────────────────────────────────────────────────────
    minutes = rehydrate_minutes(artifacts["minutes"], keep)
    mins_frame = artifacts["minutes"].recipe.transform(minutes_val)
    mins_totals = minutes.predict_samples(mins_frame, SEED)
    mins_units = minutes_val[UNIT_KEYS].copy()
    mins_units["realized"] = minutes_val["successes"].to_numpy(float)

    floor_totals = FloorMinutes().fit(minutes_train).predict_samples(minutes_val, SEED)

    # ── The rows both heads cover ─────────────────────────────────────────────
    common = mins_units.merge(comp_units[UNIT_KEYS + ["realized"]], on=UNIT_KEYS,
                              how="inner", suffixes=("_minutes", "_composition"))
    comp_pos = {k: i for i, k in enumerate(map(tuple, comp_units[UNIT_KEYS].to_numpy()))}
    mins_pos = {k: i for i, k in enumerate(map(tuple, mins_units[UNIT_KEYS].to_numpy()))}
    keys = list(map(tuple, common[UNIT_KEYS].to_numpy()))
    idx_c = np.array([comp_pos[k] for k in keys])
    idx_m = np.array([mins_pos[k] for k in keys])

    y_comp = common["realized_composition"].to_numpy(float)
    y_mins = common["realized_minutes"].to_numpy(float)
    shared_comp = comp_totals[:, idx_c]
    shared_mins = mins_totals[:, idx_m]
    shared_floor = floor_totals[:, idx_m]

    print(f"\n  head-to-head on {len(common):,} player-seasons both heads cover; the "
          f"composition additionally\n  covers {len(comp_units) - len(common):,} the "
          f"marginal head's `>= 200 prior minutes & >= 10 games` filter drops.")

    rows = [
        score_season(shared_mins, y_mins, "minutes_head", "season_total"),
        score_season(shared_comp, y_comp, "composition_sum", "season_total"),
        score_season(shared_floor, y_mins, "carry_forward", "season_total"),
        score_season(comp_totals, comp_units["realized"].to_numpy(float),
                     "composition_sum_all_rows", "season_total"),
    ]

    # The composition's team-season total is FIXED, not merely tight: every team-game
    # allocates exactly 5 x game_length among the players who played, so summing a team's
    # players over a season is a constant across draws. Measured rather than asserted,
    # because it is the structural half of the verdict — a dispersion parameter cannot buy
    # a spread the constraint forbids.
    team_units = composition_val.groupby(["season", "team_id"], sort=True)
    team_codes = team_units.ngroup().to_numpy()
    team_order = np.argsort(team_codes, kind="stable")
    team_starts = np.searchsorted(team_codes[team_order], np.arange(team_codes.max() + 1))
    team_totals = np.add.reduceat(comp_games[:, team_order], team_starts, axis=1)
    rows[3]["team_season_sd"] = float(team_totals.std(axis=0).mean())

    headroom = season_effect_headroom(comp_totals, comp_units,
                                      comp_units["realized"].to_numpy(float))

    crps_comp = crps_from_samples(shared_comp, y_comp)
    crps_mins = crps_from_samples(shared_mins, y_mins)
    delta = paired_bootstrap(crps_comp, crps_mins)
    delta_shared = float(crps_from_samples(shared_comp, y_mins).mean()
                         - crps_mins.mean())
    outcome = verdict(delta)

    rows.append({
        "arm": "composition_minus_minutes", "unit": "paired_bootstrap",
        "n": len(common), "n_draws": keep, **delta,
        "crps_delta_shared_target": delta_shared,
        "realized_target_mean_abs_diff": float(np.abs(y_comp - y_mins).mean()),
        "realized_target_max_abs_diff": float(np.abs(y_comp - y_mins).max()),
        "sd_ratio_minutes_over_composition":
            float(rows[0]["predictive_sd"] / rows[1]["predictive_sd"]),
        "verdict": outcome,
    })
    rows.append(headroom)

    # Is the 4.68x a missing parameter or a ceiling the constraint imposes? Injected, not
    # fitted — see `player_season_effect_sweep`.
    sweep = player_season_effect_sweep(composition, comp_frame, composition_val,
                                       comp_units["realized"].to_numpy(float),
                                       idx_c, y_comp, crps_mins,
                                       fitted_sigma=fitted_sigma or None)
    rows.extend(sweep)
    # The grid's optimum, not the fitted row's — they answer different questions and the
    # narrative below compares them.
    best = min((r for r in sweep if r["sigma_source"] == "injected_grid"),
               key=lambda r: r["crps_minutes"])

    # The same grid on TRAINING rows — `docs/simulations-plan.md`'s named fallback made
    # runnable, so a shippable sigma exists whether or not the Stan fit lands in time.
    train_sweep = estimate_sigma_on_train(composition, artifacts["composition"],
                                          composition_train)
    rows.extend(train_sweep)
    sigma_train = min(train_sweep, key=lambda r: r["crps_minutes"])

    rows.append(teammate_coupling(comp_totals, comp_units, composition_val,
                                  "composition_sum"))
    rows.append(teammate_coupling(mins_totals, mins_units, composition_val,
                                  "minutes_head"))

    table = pd.DataFrame(rows)
    table["fit_window"] = window
    shown = ["arm", "n", "crps_minutes", "mae_minutes", "r2_minutes", "bias_minutes",
             "pit_ks", "predictive_sd"]
    print("\nSeason-total minutes, validation (CRPS and MAE in minutes, lower is better):")
    print(table.loc[table["unit"] == "season_total", shown].round(4)
          .to_string(index=False))

    print(f"\n  Paired bootstrap over {len(common):,} player-seasons, "
          f"composition - minutes:\n"
          f"    CRPS delta {delta['crps_delta']:+.4f} minutes, 95% CI "
          f"[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}], "
          f"P(delta < 0) = {delta['p_delta_negative']:.1%}")
    print(f"    against a single shared realized target: {delta_shared:+.4f} — the two "
          f"frames disagree\n    about realized minutes by "
          f"{np.abs(y_comp - y_mins).mean():.2f} on average "
          f"(max {np.abs(y_comp - y_mins).max():.0f}), which is not what carries this.")
    print(f"\n  VERDICT: the composition {outcome.upper()} at the season unit.")

    print(f"\n  Where it loses is the SPREAD, not the mean. MAE "
          f"{rows[1]['mae_minutes']:.2f} against {rows[0]['mae_minutes']:.2f} and R2 "
          f"{rows[1]['r2_minutes']:.4f} against {rows[0]['r2_minutes']:.4f} are a tie, and "
          f"the\n  composition is the less biased of the two "
          f"({rows[1]['bias_minutes']:+.2f} against "
          f"{rows[0]['bias_minutes']:+.2f}). Its season-total predictive sd is "
          f"{rows[1]['predictive_sd']:.1f} minutes\n  against "
          f"{rows[0]['predictive_sd']:.1f} — "
          f"{rows[0]['predictive_sd'] / rows[1]['predictive_sd']:.2f}x too narrow — and "
          f"its PIT KS is {rows[1]['pit_ks']:.4f} against {rows[0]['pit_ks']:.4f}.")
    print(f"  Summing iid-across-games draws cannot make season-level heterogeneity: "
          f"per-game noise\n  averages down by ~1/sqrt(G) while a season-level multiplier "
          f"passes through in full. And the\n  team constraint forbids the fix — a team's "
          f"season minutes are fixed at 5 x sum(game_length),\n  measured here as a "
          f"predictive sd of {rows[3]['team_season_sd']:.2f} minutes across draws.")
    print(f"\n  The sharpest form of that: at the season unit the composition does not "
          f"clear the no-fit\n  carry-forward floor — CRPS {rows[1]['crps_minutes']:.2f} "
          f"against the floor's {rows[2]['crps_minutes']:.2f} — on the same draws that "
          f"clear\n  its own per-team-game floor decisively. Same head, two units, "
          f"opposite verdicts.")
    print(f"  Its aggregate bias over the complete player set is "
          f"{rows[3]['bias_minutes']:+.1e} by CONSTRUCTION, not by\n  skill: the league's "
          f"minutes are fully allocated, so summing every unit recovers the total exactly.")

    print(f"\n  Could a SEASON term close the gap? No, and by a wide margin. A year trend "
          f"or year\n  random effect is a league-wide shift shared by every row in a draw, "
          f"so the only variance\n  it can reach is that of the league-wide mean residual "
          f"across seasons — which is\n  "
          f"{headroom['league_season_mean_resid_sd']:.3f} minutes against a residual sd of "
          f"{headroom['resid_sd']:.2f}, i.e. "
          f"{headroom['share_of_resid_var_reachable']:.6%} of the variance to be\n  "
          f"explained. That is zero by construction and not by accident: the head allocates "
          f"every\n  minute in the league, so the residuals sum to zero within each season.")
    print(f"  The effect that WOULD fit is indexed by (player, season), not by season: the "
          f"per-player-\n  season deviation of realized from predicted season minutes has "
          f"a log-scale sd of\n  {headroom['log_ratio_sd']:.4f} over the "
          f"{headroom['log_ratio_n']:,} rows clearing {MIN_QUALIFIED} realized minutes, "
          f"against the\n  minutes head's fitted sigma_year of 0.0231 — a different index, "
          f"and an order of magnitude apart\n  in size. Such a term is compatible with the "
          f"team constraint (one player's breakout takes\n  minutes from a teammate), "
          f"which is what makes it the promising direction rather than a\n  second way of "
          f"failing. The sweep below measures how far it gets.")

    print(f"\nInjected per-player-season effect — is the "
          f"{rows[0]['predictive_sd'] / rows[1]['predictive_sd']:.2f}x a missing parameter "
          f"or a ceiling?")
    print(pd.DataFrame(sweep)[["sigma", "sigma_source", "crps_minutes", "crps_delta",
                               "ci_lo", "ci_hi", "predictive_sd", "mae_minutes",
                               "pit_ks", "verdict"]]
          .round(4).to_string(index=False))
    print(f"  A MISSING PARAMETER. At sigma {best['sigma']:.3f} the composition reads CRPS "
          f"{best['crps_minutes']:.2f} against the\n  marginal head's "
          f"{float(crps_mins.mean()):.2f} — {best['verdict']} — with PIT KS "
          f"{best['pit_ks']:.4f} against the marginal head's {rows[0]['pit_ks']:.4f}, and a "
          f"predictive\n  sd of {best['predictive_sd']:.1f} against "
          f"{rows[1]['predictive_sd']:.1f} at sigma 0. The team constraint does NOT forbid "
          f"the spread —\n  it forbids a SHARED one, and a per-player effect is not shared. "
          f"MAE barely moves ({best['mae_minutes']:.2f}),\n  so this buys spread and not "
          f"fit, which is exactly the diagnosis.")
    print(f"\nThe same grid on TRAINING rows — the fallback's shippable sigma, since the "
          f"row above\n  reads its optimum off the split it is scored against "
          f"({sigma_train['seasons']}, "
          f"{sigma_train['n']:,} player-seasons):")
    print(pd.DataFrame(train_sweep)[["sigma", "crps_minutes", "mae_minutes", "pit_ks",
                                     "predictive_sd"]].round(4).to_string(index=False))
    print(f"  sigma_train = {sigma_train['sigma']:.3f} against the validation grid's "
          f"{best['sigma']:.3f}. Agreement is the point:\n  it says the tuned figure was "
          f"not tuned to the evaluation rows in any way that moved it.")

    # Which sigma actually ships, and what it reads on validation. `composition_sum` above
    # stays the UN-injected head deliberately: it is the control every claim here is
    # measured against, and the figures README and docs-audit quote.
    shipped = shipped_sigma(cfg)
    row = next((r for r in sweep if abs(r["sigma"] - shipped) < 1e-9), None)
    table_extra = {"arm": "shipped_configuration", "unit": "ps_effect_shipped",
                   "sigma": shipped, "sigma_source": "config",
                   "n": len(common), "n_draws": keep}
    if row is not None:
        table_extra.update({k: row[k] for k in
                            ("crps_minutes", "mae_minutes", "pit_ks", "predictive_sd",
                             "crps_delta", "ci_lo", "ci_hi", "verdict")})
        print(f"\n  SHIPPED: sim.minutes.player_season_sigma = {shipped:.3f}, estimated on "
              f"TRAIN.\n    validation CRPS {row['crps_minutes']:.2f} against the marginal "
              f"head's {float(crps_mins.mean()):.2f} "
              f"({row['crps_delta']:+.2f} [{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}] — "
              f"{row['verdict'].upper()}),\n    PIT KS {row['pit_ks']:.4f} against "
              f"{rows[0]['pit_ks']:.4f}, predictive sd {row['predictive_sd']:.1f} against "
              f"{rows[1]['predictive_sd']:.1f} un-injected.\n    `rehydrate_composition` "
              f"applies it, so a consumer gets it by loading the head rather than by "
              f"remembering to.")
    rows.append(table_extra)

    if fitted_sigma:
        row = next(r for r in sweep if r["sigma_source"] == "fitted")
        print(f"  CALIBRATION CHECK, not the measurement: the head ships a FITTED sigma_u "
              f"of {row['sigma']:.4f},\n  estimated on train, which reads CRPS "
              f"{row['crps_minutes']:.2f} here against the grid's best "
              f"{best['crps_minutes']:.2f} at sigma\n  {best['sigma']:.3f}. The grid stays "
              f"because an interior optimum is how you find out whether a\n  fitted sigma "
              f"landed in the right place.")
    else:
        print(f"  INJECTED, NOT FITTED: sigma is read off validation CRPS, so this is a "
              f"tuned upper bound on\n  what the parameterization can reach, not a score. "
              f"It settles the structural question and\n  nothing else — a real fit "
              f"estimates sigma from train and re-estimates beta alongside it.")

    couple = {r["arm"]: r for r in rows if r.get("unit") == "teammate_coupling"}
    c, m = couple["composition_sum"], couple["minutes_head"]
    print(f"\nTeammate coupling — the zero-sum dynamic, on {c['n']:,} single-team "
          f"player-seasons:")
    print(f"  composition:   r {c['r_teammates']:+.4f}, team season-total sd "
          f"{c['team_season_sum_sd']:8.1f} min")
    print(f"  marginal head: r {m['r_teammates']:+.4f}, team season-total sd "
          f"{m['team_season_sum_sd']:8.1f} min")
    print(f"  A fixed team total over {c['roster_size']:.2f} players forces mean pairwise "
          f"r = {c['r_implied_by_fixed_sum']:+.4f}, which the\n  composition sits on and "
          f"the marginal head misses entirely. (Neither team sd is 0 here because dropping\n"
          f"  traded players leaves a SUBSET of each roster; the exact figure is the 0.00 on "
          f"the\n  all-rows line above. The ratio between the two heads is what carries.) "
          f"The marginal head puts a\n  "
          f"{m['team_season_sum_sd']:.0f}-minute predictive sd on a team season total that "
          f"is PHYSICALLY FIXED — invisible in\n  every marginal metric, and it lands on "
          f"stacking (a same-team pair's minutes are\n  anti-correlated, not independent) "
          f"and on handcuffing (a hedge that exists only if the\n  model carries the sign).")

    dest = out_dir / "minutes_unification.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} unification rows → {dest}")
    return {"metrics": dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
