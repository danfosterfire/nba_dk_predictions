"""The strategy sweep — error injection (Gate C), the table, and the two backtests.

`docs/simulations-plan.md` ("`src/sim/strategy.py`", "The backtest") fixes what this owes.
A strategy is a **config object** — ranking source, `alpha` overall and per round, position
caps, exposure caps, stacking, objective, entry count — so exploring the space is a table
rather than a rewrite, and every arm goes through one draft loop and one scorer.

Everything above this module already exists and is measured: `make simulate-season` writes
the tensor, `make draft-sim` fits the field, `make bracket` owns the contest, and
`make draft-room-prep` owns the in-draft objective. This module composes them. The one
genuinely new piece of machinery is the error injection, and it has its own gate.

## Gate C, and the thing it measured that the plan did not expect

The plan's argument for injecting error is exactly right in its conclusion and wrong in its
premise, and the difference matters for what ships.

> A season drawn from the model's own posterior is a world where the model is perfectly
> calibrated by construction, so ADP can only add noise and the sweep will drive `alpha` to
> zero for reasons that have nothing to do with whether the market knows something.

The first clause is a statement about **magnitude** — that the simulated world is too easy —
and it is measurable directly: score the model's own posterior mean against a draw from its
own posterior and compare that to the model's measured out-of-sample miss. Run on the shipped
tensors, the uninjected world lands on the season-total MAE bar and on the availability CRPS
bar, and is *harder* than reality on the two R² rows (see `magnitude_check`). The simulator's
predictive spread is about the size of its real error, which is what a head that shrinks hard
is supposed to deliver, and Gate A had already half-said it: the simulator's season-total CRPS
against *realized* data came in **better** than the incumbent's.

So the uninjected world is not too easy. What it gets wrong is the **standing of the two
rankers**: its error is orthogonal to everything, so the model stays the unbiased efficient
predictor of it and ADP is a strictly noisier view of the same thing. On realized validation
seasons the market's Spearman against season totals is *above* the model's; in the model's own
posterior world the model leads by 0.11 to 0.12. Under that world no `alpha > 0` can pay for
itself whatever the market knows, and **more noise cannot fix it** — noise is what the model
already has too much of relative to the market.

The injection therefore **rotates** the error onto the market-visible direction while holding
its magnitude at the measured bar, rather than adding error on top of it. Two solved
parameters and nothing plugged in: `g` holds the season-total MAE on its bar and comes back
near 1.0 because the magnitude was already right, and `rho` puts the simulated market-minus-
model skill gap on the realized one. An independent route to `rho` — the correlation between
the market's disagreement and the model's realized residual — agrees to within 0.1.

**This is a stronger version of the plan's own claim, not a weakening of it.** An uninjected
sweep still answers a different question and its `alpha` is still not transportable — but the
reason is that the world has no market signal in it, not that the world is too kind.

## What the sweep selects on, and what it reports

`select-on-p-advance-report-roi`. Round 1 is a 2-of-12 cut in all five captured structures
and is the only zero-consolation round, so `P(top 2 of 12)` is exactly `P(any return)`, it is
a 16.67% event that resolves fast, and an ADP-drafted entry's value is known in closed form
(`n_advance / pod_size`) rather than estimated. ROI rides alongside with a bootstrap interval
against the break-even hurdle, and `600k_shootaround`'s ROI does not resolve at any
affordable budget — which `make bracket` and `make draft-room-prep` both already say.

**The portfolio is the unit, not the entry.** Exposure caps and differentiation cannot pay in
a per-entry mean: ten entries holding the same sixteen players have the same `P(advance)` as
one entry. What they change is `P(at least one advances)`, which is `1 - prod(1 - p)` *inside
a simulated season* and needs the per-sim advance probabilities `bracket_ev(per_sim=True)`
returns. Every hedge axis in the config is measured there.

## The two backtests do different jobs, and only one of them selects

Simulated truth is the tuning surface — unlimited resolution, and the only one with the power
to separate strategies. Realized 2022-23 / 2023-24 is a **readout**: N = 2 seasons of
correlated pods cannot distinguish `alpha = 0.3` from `alpha = 0.5` and this module does not
pretend otherwise. Its job is to catch a strategy broken in a way the simulated world cannot
see, and the intervals it reports are wide on purpose.

Usage:
    python -m src.sim.strategy
    python -m src.sim.strategy --season 2022-23 --n-sims 200      # fast
    python -m src.sim.strategy --no-objective-arm                 # skip the slow arm
"""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.stan_utils import crps_from_samples
from src.sim import draft, draft_room
from src.sim.bracket import (POSITIONS, ROSTER_SIZE, score_rosters, split_frame,
                             symmetric_null)
from src.sim.season import assert_season_allowed, scoring_slots, validation_seasons

SEED = 0

# Simulated seasons the sweep scores on. Our entries and the field share them, so every
# comparison is paired and resolves far faster than the nominal sample size suggests.
N_SIMS_SWEEP = 500

# Twelve-seat pods of the fitted field, drafted on the INJECTED world. Matches
# `sim.draft_room.field_drafts`, and the caveat is the same one `null_check` records: the
# advance rate is exact at this size and `600k_shootaround`'s EV level is not.
N_FIELD_DRAFTS = 100

# Bootstrap draws for every reported interval. Two-way — sims then entries — because the
# two carry different uncertainty: a simulated season moves every entry together, and an
# entry's roster is fixed across seasons.
N_BOOTSTRAP = 2000

# Gate C's rows: what is checked, which bar it is checked against, and the artifact that
# sets the bar. Named here so the gate cannot quietly drift onto a different artifact; the
# values themselves are always re-read at run time by `gate_c_bars`.
#
# `dk_rate_r2` is the one substitution and it is stated rather than hidden. The plan's third
# target is the **component** heads' R2 against their no-fit floors, and the tensor persists
# `dk_pts` and games played rather than the eleven components — re-drawing them would mean
# re-running the whole component chain. Season dk_pts per game played is the closest thing
# the artifact carries: it is a rate, it is the channel the injection acts on, and its band
# is the components' own. Read it as an analogue, not as the row itself.
GATE_C_ROWS = (
    ("season_total_mae", "season_total_mae", None,
     "season_total_metrics.csv", "beta_binomial/all"),
    ("season_total_crps", "season_total_crps", None,
     "season_total_metrics.csv", "beta_binomial/all"),
    ("season_total_r2", "season_total_r2", None,
     "season_total_metrics.csv", "beta_binomial/all"),
    ("availability_crps", "availability_crps", None,
     "stan_games_played_metrics.csv", "floor"),
    ("dk_rate_r2", None, ("component_r2_lo", "component_r2_hi"),
     "stan_component_metrics.csv", "selected heads (analogue)"),
)

# A season total below this is not a scale a multiplicative rate perturbation can act on —
# it is a player who barely appeared in that simulated world — so the injection leaves those
# cells alone rather than dividing by them. In dk_pts.
INJECT_FLOOR = 25.0
INJECT_MAX_RATIO = 4.0             # the largest rate perturbation the injection may apply

# Twelve-seat pods drafted on the UNRESTRICTED board, purely to measure how many
# zero-scoring players the ADP field takes. That figure is the argument for restricting the
# board at all, so it is measured on every run rather than remembered.
PRICEABLE_PROBE_DRAFTS = 30

# How the per-round `alpha` schedule is cut, in draft rounds. `docs/adp-plan.md` measures
# model-market disagreement at 5.1 picks in rounds 1-2 against 30.8 in rounds 9+, and rounds
# 9-16 fill 8 of the 16 roster spots — so a single `alpha` tuned on the whole draft is tuned
# mostly on the half where the two sources agree anyway.
ALPHA_ROUND_EDGES = (2, 8)


# ── 1. The strategy, as a config object ───────────────────────────────────────

RANKINGS = ("model_mean", "model_quantile", "adp", "blend")
OBJECTIVES = ("ranking", "lineup_value", "bracket_ev", "p_advance")


@dataclass(frozen=True)
class Strategy:
    """One row of the sweep. Everything that distinguishes one drafting policy.

    `ranking` names where the *value* ordering comes from and `objective` names how a pick
    is chosen given it — they are separate axes because a market blend is meaningful under
    every objective. `ranking = "blend"` mixes the model's ordering with the market's in
    **rank space**, which is the unit `alpha` is quoted in and the unit `rank_noise_sd` was
    fitted in, so the two can be read against each other.

    `alpha_rounds` is the per-round schedule as three numbers — rounds 1-2, 3-8 and 9-16 —
    rather than sixteen, because that is the shape `docs/adp-plan.md` measured and a
    sixteen-vector is sixteen chances to fit noise. `None` means the flat `alpha` applies
    throughout.
    """

    name: str
    ranking: str = "model_mean"
    alpha: float = 0.0
    alpha_rounds: tuple[float, float, float] | None = None
    quantile: float = 0.5
    position_caps: tuple[int, ...] = (ROSTER_SIZE,) * len(POSITIONS)
    exposure_cap: float = 1.0
    stacking: float = 0.0
    objective: str = "ranking"
    n_entries: int = 0                 # 0 = take the tier's own entry count
    axis: str = ""                     # which sweep axis this row varies; for reporting

    def __post_init__(self):
        if self.ranking not in RANKINGS:
            raise KeyError(f"unknown ranking {self.ranking!r}; registered: {RANKINGS}")
        if self.objective not in OBJECTIVES:
            raise KeyError(f"unknown objective {self.objective!r}; "
                           f"registered: {OBJECTIVES}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be a weight in [0, 1]; got {self.alpha}")
        if not 0.0 < self.exposure_cap <= 1.0:
            raise ValueError(f"exposure_cap is a share in (0, 1]; got {self.exposure_cap}")

    def alpha_at(self, round_index: int) -> float:
        """The market weight in force at draft round `round_index` (0-based)."""
        if self.ranking == "adp":
            return 1.0
        if self.ranking != "blend":
            return 0.0
        if self.alpha_rounds is None:
            return float(self.alpha)
        lo, hi = ALPHA_ROUND_EDGES
        tier = 0 if round_index < lo else (1 if round_index < hi else 2)
        return float(self.alpha_rounds[tier])

    def as_row(self) -> dict:
        return {"strategy": self.name, "axis": self.axis, "ranking": self.ranking,
                "alpha": self.alpha,
                "alpha_rounds": ("" if self.alpha_rounds is None
                                 else "/".join(f"{a:g}" for a in self.alpha_rounds)),
                "quantile": self.quantile,
                "position_caps": "/".join(f"{p}{c}" for p, c in
                                          zip(POSITIONS, self.position_caps)),
                "exposure_cap": self.exposure_cap, "stacking": self.stacking,
                "objective": self.objective}


def model_value(dk_pts: np.ndarray, ranking: str, quantile: float) -> np.ndarray:
    """The board's own opinion of a player's season, in dk_pts. Higher is better.

    `model_mean` is the posterior mean season total. `model_quantile` is an upper quantile
    of it, which is the cheapest possible expression of "chase the tail": under a
    zero-consolation knockout the entry that finishes third is worth the same as the entry
    that finishes twelfth, so a player's *ceiling* is worth more than his mean in a way a
    mean ranking cannot express. It reads the tensor's own sim axis, so the ceiling it names
    is the model's rather than a multiple of a standard deviation.
    """
    totals = dk_pts.sum(axis=1)                                # [player, sim]
    if ranking == "model_quantile":
        return np.quantile(totals, quantile, axis=1)
    return totals.mean(axis=1)


def rank_of(value: np.ndarray, better_is_high: bool = True) -> np.ndarray:
    """Dense 0-based ordering of `value`, as a float so it can be blended with a rank."""
    order = np.argsort(-value if better_is_high else value, kind="stable")
    out = np.empty(len(value), dtype=np.float64)
    out[order] = np.arange(len(value), dtype=np.float64)
    return out


# ── 2. Gate C — what the model's miss actually looks like, and injecting it ───

def market_signal(model_total: np.ndarray, adp: np.ndarray) -> tuple[np.ndarray,
                                                                    np.ndarray]:
    """`(d, has_adp)` — how much earlier the market takes a player than the model would.

    The market publishes an **ordering**, not a projection, so it has to be put on a value
    scale before it can be differenced against one. The scale used is the model's own: the
    player the market ranks *k*-th is assigned the *k*-th largest model season total, which
    is the isotonic map that changes the ordering not at all and the units completely. `d`
    is then `market value - model value`, standardized over the players the market prices.

    Deliberately not a regression of realized totals on ADP: that would fit the market's
    calibration on the same rows the disagreement is then measured on, and the quantity
    wanted here is *disagreement*, which is a property of two orderings alone.
    """
    has = np.isfinite(adp)
    d = np.zeros(len(model_total), dtype=np.float64)
    if has.sum() < 3:
        return d, has
    idx = np.nonzero(has)[0]
    order = idx[np.argsort(adp[idx], kind="stable")]
    implied = np.sort(model_total[idx])[::-1]
    gap = np.empty(len(idx), dtype=np.float64)
    gap[np.argsort(adp[idx], kind="stable")] = implied - model_total[order]
    d[idx] = (gap - gap.mean()) / max(gap.std(), 1e-12)
    return d, has


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware rank correlation, written out because the calibration calls it in a loop.

    The ties are not hypothetical: `adp_dk_scale`'s fitted isotonic map takes 253 consensus
    ranks to 58 values with a 55-wide plateau, so average ranks rather than ordinal ones
    are the difference between a correlation and a shuffled tail.
    """
    a = pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()
    b = pd.Series(np.asarray(y, dtype=float)).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def measure_market_share(model_total: np.ndarray, realized: np.ndarray,
                         adp: np.ndarray, played: np.ndarray) -> dict:
    """What the market is measurably worth against the model — the injection's two targets.

    Two quantities, both over the players the market prices and the season actually scored:

    - **`rho`**, `corr(d, realized - model)`. A positive value says the market is right
      about the *direction* the model is wrong in, which is the whole reason a blend can
      pay; zero says `alpha -> 0` is the correct answer rather than an artefact.
    - **`skill_gap`**, `spearman(market, realized) - spearman(model, realized)`. This is the
      quantity a blend weight is actually priced against, and it is the one the simulated
      world gets wrong by default: a world drawn from the model's posterior leaves the model
      an unbiased, efficient predictor of it and the market a strictly noisier view of the
      same thing, whatever `rho` says. Reproducing the *magnitude* of the miss does not fix
      that; reproducing the two rankers' measured relative skill does.

    Measured on validation, which is the split selection is allowed to read — and the same
    two seasons the sweep scores. That is the honest caveat and it belongs beside the
    numbers: both inherit the sampling error of two seasons of roughly 200 priced players,
    and every `alpha` the sweep selects inherits it in turn.
    """
    keep = np.isfinite(adp) & (played > 0)
    n = int(keep.sum())
    if n < 10:
        return {"n": n, "rho": 0.0, "rho_r2": 0.0, "spearman_model": np.nan,
                "spearman_market": np.nan, "skill_gap": 0.0,
                "mae_model": np.nan, "mae_market": np.nan}
    d, _ = market_signal(model_total, adp)
    err = realized - model_total
    rho = float(np.corrcoef(d[keep], err[keep])[0, 1])

    implied = np.zeros_like(model_total)
    idx = np.nonzero(np.isfinite(adp))[0]
    order = idx[np.argsort(adp[idx], kind="stable")]
    implied[order] = np.sort(model_total[idx])[::-1]
    s_model = spearman(model_total[keep], realized[keep])
    s_market = spearman(-adp[keep], realized[keep])
    return {"n": n, "rho": rho, "rho_r2": rho ** 2,
            "spearman_model": s_model, "spearman_market": s_market,
            "skill_gap": s_market - s_model,
            "mae_model": float(np.abs(err[keep]).mean()),
            "mae_market": float(np.abs(implied[keep] - realized[keep]).mean())}


def magnitude_check(dk_pts: np.ndarray, games_played: np.ndarray,
                    truth_sims: np.ndarray, model_total: np.ndarray | None = None,
                    predictive: np.ndarray | None = None) -> dict:
    """Score the model's belief against a world, however that world was made.

    This is the measurement the plan's premise turns on and it had never been taken. If the
    simulated world were "too easy", these figures would come in well inside the model's
    measured out-of-sample miss; each row is reported against the artifact that set its bar
    rather than collapsed into a verdict.

    **`model_total` and `predictive` are the model's belief and must be passed explicitly
    once the world has been perturbed.** Taking them from the array being scored is right
    for the uninjected world and silently wrong for the injected one: the injection moves
    each player's mean, so a mean recomputed from the injected tensor has already absorbed
    the market bias it was supposed to be missing, and the gate would report a smaller error
    than the model actually makes.

    `truth_sims` names which sims stand in for realized worlds. It is a **thinned** index
    rather than a prefix, for the reason `season_terms._draw_components` records: sim `s`
    uses posterior draw `s % n_draws`, so a prefix is one contiguous stretch of the
    posterior masquerading as a sample of worlds.
    """
    totals = dk_pts.sum(axis=1).astype(np.float64)             # [player, sim]
    gp = games_played.sum(axis=1).astype(np.float64)
    predictive = totals if predictive is None else np.asarray(predictive, dtype=float)
    model_total = (predictive.mean(axis=1) if model_total is None
                   else np.asarray(model_total, dtype=float))
    rate = totals / np.maximum(gp, 1e-9)
    model_rate = (predictive / np.maximum(gp, 1e-9)).mean(axis=1)

    mae, r2, rate_r2 = [], [], []
    for s in truth_sims:
        y = totals[:, s]
        mae.append(np.abs(model_total - y).mean())
        r2.append(1 - ((y - model_total) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        y = rate[:, s]
        rate_r2.append(1 - ((y - model_rate) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    crps = np.mean([crps_from_samples(predictive.T, totals[:, s]).mean()
                    for s in truth_sims])
    gp_crps = np.mean([crps_from_samples(gp.T, gp[:, s]).mean() for s in truth_sims])
    return {"n_players": int(totals.shape[0]), "n_truth_sims": int(len(truth_sims)),
            "season_total_mae": float(np.mean(mae)),
            "season_total_r2": float(np.mean(r2)),
            "season_total_crps": float(crps),
            "availability_crps": float(gp_crps),
            "dk_rate_r2": float(np.mean(rate_r2)),
            "model_mean_gp": float(gp.mean()),
            "predictive_sd": float(predictive.std(axis=1).mean())}


def inject_error(dk_pts: np.ndarray, d: np.ndarray, has_adp: np.ndarray,
                 rho: float, mae_target: float,
                 truth_sims: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rotate the world's error onto the market-visible direction, at a fixed magnitude.

    The uninjected world already carries a season-total deviation `e = T - E[T]` whose size
    is the model's measured miss (see `magnitude_check`) and whose direction is uncorrelated
    with everything. What it must also carry is a component the market can see, since
    otherwise no `alpha > 0` can pay whatever the market knows. So each player-world's
    deviation becomes

        u = g * (rho * sd(e) * d  +  sqrt(1 - rho^2) * e)

    with `d` standardized over the priced players. The mixing weights are the ones that make
    `corr(u, d) = rho` exactly among those players while leaving `var(u) = var(e)`, and `g`
    is then solved so the season-total MAE lands on its bar. **Players the market does not
    price keep `rho = 0` and full noise weight**, so the injection changes what their error
    correlates with for nobody and its size for no one — the alternative leaves the deep
    board with less error than the top of it, which is backwards.

    The perturbation is applied as a **season-level multiplicative rate** on the tensor: the
    weekly shape and the games-played twin are untouched and only the level moves. That is
    where the project's own variance budget puts the skill — roughly 90% of what is
    attainable is the season-level rate — and it is the one channel a pre-season market
    could plausibly be right about.
    """
    totals = dk_pts.sum(axis=1).astype(np.float64)             # [player, sim]
    model_total = totals.mean(axis=1)
    e = totals - model_total[:, None]
    s_e = float(e.std())

    w_market = np.where(has_adp, rho, 0.0)[:, None]
    w_noise = np.sqrt(np.maximum(1.0 - w_market ** 2, 0.0))
    u0 = w_market * s_e * d[:, None] + w_noise * e

    def achieved(g: float) -> tuple[np.ndarray, float]:
        target = np.maximum(model_total[:, None] + g * u0, 0.0)
        k = np.where(totals > INJECT_FLOOR,
                     np.clip(target / np.maximum(totals, INJECT_FLOOR),
                             0.0, INJECT_MAX_RATIO), 1.0)
        got = totals * k
        return k, float(np.mean([np.abs(model_total - got[:, s]).mean()
                                 for s in truth_sims]))

    lo, hi = 0.05, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if achieved(mid)[1] < mae_target:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    k, got_mae = achieved(g)

    truth = (dk_pts.astype(np.float32) * k[:, None, :].astype(np.float32))
    return truth, {"rho": float(rho), "scale_g": float(g), "sd_deviation": s_e,
                   "achieved_mae": got_mae, "target_mae": float(mae_target),
                   "n_priced": int(has_adp.sum()),
                   "clipped_cells": float(np.mean((k <= 0.0)
                                                  | (k >= INJECT_MAX_RATIO))),
                   "floored_cells": float(np.mean(totals <= INJECT_FLOOR))}


def simulated_skill_gap(truth: np.ndarray, model_total: np.ndarray, adp: np.ndarray,
                        has_adp: np.ndarray, truth_sims: np.ndarray) -> tuple[float, float,
                                                                              float]:
    """`(gap, model, market)` — the two rankers' skill in a world, on the priced players.

    The same three numbers `measure_market_share` takes on realized data, so the calibration
    target and the thing being calibrated are the identical statistic on the identical
    population and differ only in which season they call true.
    """
    totals = truth.sum(axis=1).astype(np.float64)
    s_model = float(np.mean([spearman(model_total[has_adp], totals[has_adp, s])
                             for s in truth_sims]))
    s_market = float(np.mean([spearman(-adp[has_adp], totals[has_adp, s])
                              for s in truth_sims]))
    return s_market - s_model, s_model, s_market


def calibrate_injection(dk_pts: np.ndarray, d: np.ndarray, has_adp: np.ndarray,
                        adp: np.ndarray, model_total: np.ndarray, gap_target: float,
                        mae_target: float, truth_sims: np.ndarray
                        ) -> tuple[np.ndarray, dict]:
    """Solve `rho` so the injected world reproduces the market's **measured** skill gap.

    The magnitude of the model's miss is already right in the uninjected world — that is
    `magnitude_check`'s finding and it is the plan's premise inverted. What is wrong is the
    *relative standing of the two rankers*: on the two validation seasons the market's
    Spearman against realized totals is **above** the model's, while in a world drawn from
    the model's own posterior the model leads by 0.15 to 0.18. A sweep run there prices
    `alpha` against a market that is a strictly noisier copy of the model, which is exactly
    the failure Gate C exists to prevent, and no amount of extra *noise* fixes it — noise
    is what the model already has too much of relative to the market.

    So `rho` is solved rather than plugged in: the mixing weight is monotone in the gap it
    produces, and one bisection puts the simulated gap on the realized one. `g` continues to
    hold the season-total MAE on its bar inside every evaluation, so the magnitude target is
    never traded for the skill target.

    **`rho` from the error correlation is kept as the independent check.** The two routes
    share no arithmetic — one is a correlation between a disagreement and a residual, the
    other a difference of two rank correlations — so agreement between them is evidence and
    disagreement is a question. Both are recorded.
    """
    def solve(rho: float) -> tuple[np.ndarray, dict, float]:
        truth, record = inject_error(dk_pts, d, has_adp, rho, mae_target, truth_sims)
        gap, s_model, s_market = simulated_skill_gap(truth, model_total, adp, has_adp,
                                                     truth_sims)
        record |= {"sim_spearman_model": s_model, "sim_spearman_market": s_market,
                   "sim_skill_gap": gap}
        return truth, record, gap

    lo, hi = 0.0, 0.95
    truth, record, gap = solve(hi)
    if gap < gap_target:                       # unreachable even at the top of the range
        record |= {"rho_source": "ceiling", "gap_target": float(gap_target)}
        return truth, record
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if solve(mid)[2] < gap_target:
            lo = mid
        else:
            hi = mid
    truth, record, _ = solve(0.5 * (lo + hi))
    record |= {"rho_source": "skill_gap", "gap_target": float(gap_target)}
    return truth, record


def component_r2_band(out_dir: Path) -> tuple[float, float]:
    """The 0.81-0.95 band, re-derived rather than typed in — Gate C's third bar.

    It is the **count heads' no-fit carry-forward floor**, which is what `README.md`'s "a
    no-fit floor ... scores validation R2 0.81-0.95" names. Read off the *selected* rows
    instead and the band comes back [0.13, 0.96], because three of the four conversion heads
    score under 0.35 at a unit where an R2 against a floor means something different. A bar
    that changes meaning depending on which rows are filtered is not a bar.
    """
    metrics = pd.read_csv(out_dir / "stan_component_metrics.csv")
    floor = metrics[(metrics["variant"] == "carry_forward")
                    & (metrics["kind"] == "count")]
    return float(floor["val_r2"].min()), float(floor["val_r2"].max())


def gate_c_bars(out_dir: Path) -> dict:
    """Every Gate C bar, re-read from its artifact rather than written down here."""
    totals = pd.read_csv(out_dir / "season_total_metrics.csv")
    bar = totals[(totals["treatment"] == "beta_binomial")
                 & (totals["group"] == "all")].set_index("metric")["value"]
    gp = pd.read_csv(out_dir / "stan_games_played_metrics.csv")
    lo, hi = component_r2_band(out_dir)
    return {"season_total_mae": float(bar["mae_dk_total"]),
            "season_total_r2": float(bar["r2_dk_total"]),
            "season_total_crps": float(bar["crps_dk_total"]),
            "availability_crps": float(gp.loc[gp["variant"] == "floor",
                                              "val_crps"].iloc[0]),
            "component_r2_lo": lo, "component_r2_hi": hi}


def gate_c(cfg: dict, season: str, dk_pts: np.ndarray, games_played: np.ndarray,
           scorable: np.ndarray, player_id: np.ndarray, board: pd.DataFrame,
           truth_sims: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Measure the miss, rotate its market-visible half in, and verify what came out.

    Returns the injected truth tensor (board-aligned, so it drops straight into
    `score_rosters`), the gate table, and the injection's own record.

    **Scored on the rows the tensor actually prices.** `draft.tensor_scores` pads the
    unscorable board rows with zeros so a draft can still run into them; measuring a gate
    over those rows would report a perfect fit on 62 players who are identically zero in
    every world. They keep their zeros in the truth tensor and take no part in the gate.
    """
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    bars = gate_c_bars(out_dir)
    idx = np.nonzero(np.asarray(scorable, dtype=bool))[0]
    dk, gp = dk_pts[idx], games_played[idx]

    predictive = dk.sum(axis=1).astype(np.float64)
    model_total = predictive.mean(axis=1)
    before = magnitude_check(dk, gp, truth_sims)
    realized = realized_totals(cfg, season, player_id[idx])
    adp = board_adp(board, player_id[idx])
    share = measure_market_share(model_total, realized["dk_total"].to_numpy(), adp,
                                 realized["gp"].to_numpy())
    d, has_adp = market_signal(model_total, adp)
    base_gap = simulated_skill_gap(dk, model_total, adp, has_adp, truth_sims)

    injected, record = calibrate_injection(dk, d, has_adp, adp, model_total,
                                           share["skill_gap"], bars["season_total_mae"],
                                           truth_sims)
    after = magnitude_check(injected, gp, truth_sims, model_total=model_total,
                            predictive=predictive)
    truth = np.array(dk_pts, dtype=np.float32, copy=True)
    truth[idx] = injected

    # The injected world's own market-visibility, measured the same way it was measured on
    # realized data — the round trip that says the rotation did what it claims.
    err = injected.sum(axis=1).astype(np.float64) - model_total[:, None]
    injected_rho = float(np.mean([np.corrcoef(d[has_adp], err[has_adp, s])[0, 1]
                                  for s in truth_sims]))

    rows = []
    for check, bar_key, band, artifact, source in GATE_C_ROWS:
        rows.append({
            "season": season, "check": check,
            "uninjected": before[check], "injected": after[check],
            "bar": bars[bar_key] if bar_key else np.nan,
            "bar_lo": bars[band[0]] if band else np.nan,
            "bar_hi": bars[band[1]] if band else np.nan,
            "artifact": artifact, "bar_source": source})
    rows.append({"season": season, "check": "market_skill_gap",
                 "uninjected": base_gap[0], "injected": record["sim_skill_gap"],
                 "bar": share["skill_gap"], "bar_lo": np.nan, "bar_hi": np.nan,
                 "artifact": "realized validation seasons",
                 "bar_source": "spearman(market) - spearman(model)"})
    record |= {"season": season, "market_rho_realized": share["rho"],
               "market_rho_injected": injected_rho,
               "market_r2_realized": share["rho_r2"], "n_adp_realized": share["n"],
               "spearman_model_realized": share["spearman_model"],
               "spearman_market_realized": share["spearman_market"],
               "skill_gap_realized": share["skill_gap"],
               "skill_gap_uninjected": base_gap[0],
               "spearman_model_uninjected": base_gap[1],
               "spearman_market_uninjected": base_gap[2],
               "mae_model_realized": share["mae_model"],
               "mae_market_realized": share["mae_market"],
               "predictive_sd": before["predictive_sd"],
               "model_mean_gp": before["model_mean_gp"],
               "n_scorable": int(len(idx))}
    return truth, pd.DataFrame(rows), record


# ── 3. Realized frames — the honest readout's target ──────────────────────────

def realized_totals(cfg: dict, season: str, player_id: np.ndarray) -> pd.DataFrame:
    """Realized season dk_pts inside DK's window, and games played over the schedule.

    The same two conventions `season.realized_frame` uses, and for the same reasons: DK's
    Round 4 closes before the NBA season does so the total is over slotted games only, while
    games played is over the whole regular season because that is the availability head's
    own denominator.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(
        features_dir / "component_targets.parquet",
        columns=["player_id", "season", "season_type", "game_id", "dk_pts", "played"])
    played = targets[(targets["season"] == season)
                     & (targets["season_type"] == "regular")
                     & (targets["played"] == 1)]
    slots = scoring_slots(features_dir, season)
    inside = set(slots.loc[slots["slot"] >= 0, "game_id"])
    out = pd.DataFrame({"player_id": player_id})
    out = out.merge(played[played["game_id"].isin(inside)]
                    .groupby("player_id", as_index=False).agg(dk_total=("dk_pts", "sum")),
                    on="player_id", how="left")
    out = out.merge(played.groupby("player_id", as_index=False).agg(gp=("played", "sum")),
                    on="player_id", how="left")
    return out.fillna({"dk_total": 0.0, "gp": 0.0})


def realized_tensor(cfg: dict, season: str, player_id: np.ndarray,
                    n_periods: int) -> np.ndarray:
    """`[player, period, 1]` of realized dk_pts — one world, the one that happened.

    Scored through the same `scoring_slots` grid the simulator writes its tensor on, so a
    portfolio can be replayed against reality by swapping one array and changing nothing
    else. The sim axis is length 1 and that is the whole point of the readout: two seasons
    is two worlds, and no amount of resampling manufactures a third.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(
        features_dir / "component_targets.parquet",
        columns=["player_id", "season", "season_type", "game_id", "dk_pts", "played"])
    rows = targets[(targets["season"] == season)
                   & (targets["season_type"] == "regular")
                   & (targets["played"] == 1)]
    slots = scoring_slots(features_dir, season).set_index("game_id")["slot"]
    rows = rows.assign(slot=rows["game_id"].map(slots))
    rows = rows[rows["slot"] >= 0]

    pos = {int(p): i for i, p in enumerate(player_id)}
    unit = rows["player_id"].map(pos)
    rows = rows[unit.notna()]
    out = np.zeros((len(player_id), n_periods, 1), dtype=np.float32)
    np.add.at(out, (rows["player_id"].map(pos).to_numpy(int),
                    rows["slot"].to_numpy(int), 0),
              rows["dk_pts"].to_numpy(np.float32))
    return out


def board_adp(board: pd.DataFrame, player_id: np.ndarray) -> np.ndarray:
    """The DK-recalibrated ADP for the tensor's players, `nan` where the market is silent.

    `docs/adp-plan.md` binds the source: never the raw consensus. A monotone recalibration
    cannot reorder a board, so what it buys here is the *scale* — and the scale is what a
    blend and a disagreement measure are both expressed in.
    """
    lookup = board.drop_duplicates("player_id").set_index("player_id")["adp_dk_scale"]
    return np.array([lookup.get(int(p), np.nan) for p in player_id], dtype=float)


# ── 4. The board both sides draft from ────────────────────────────────────────

def priceable_room(cfg: dict, room: draft_room.Room, seed: int = SEED,
                   n_field_drafts: int = N_FIELD_DRAFTS) -> tuple[draft_room.Room, dict]:
    """Restrict the board to the players the tensor can score — **for both sides**.

    🔴 This is a real restriction and it is here because the alternative is a bias that
    dwarfs everything the sweep measures. `make simulate-season` scores 386 of 539 rostered
    players; the rest have no component-head design row and are padded with **zeros** so a
    draft can still run into them. A strategy that ranks by the model never takes one. The
    ADP field takes **1.26 of them per sixteen-man entry**, with 73% of entries holding at
    least one, and every one of those is a roster spot that scores nothing all season.

    That is not model edge, it is a coverage hole in the tensor showing up as a handicap on
    one side of the comparison — and it is worth far more than any strategy axis here. Left
    in, it is also invisible: every marginal statistic looks fine and our entries simply win
    a lot. `make bracket` already sees the same thing from the other end, where the
    best-available benchmark reads `p_advance = 1.0` in all five tournaments.

    So the sweep runs on the priceable board, symmetrically. The cost is stated rather than
    hidden: **who is on the board at pick k changes**, by 101 of 448 rows in 2022-23 and 16
    of the 196 the market prices, so the field's picks are slightly better than a real
    field's. That is a smaller and more honest error than scoring a real player at zero, and
    the right fix is upstream — pricing those 153 players is the open question item 4 left.

    The room's cached field and its tournament references are rebuilt here, because they
    were drafted against the unrestricted board and a reference is only a reference to the
    population that produced it.
    """
    keep = np.nonzero(np.asarray(room.scorable, dtype=bool))[0]
    frame = room.frame.iloc[keep].copy().reset_index(drop=True)
    # Measured rather than asserted, because it is the whole argument for the restriction:
    # draft the ADP field on the UNRESTRICTED board and count how many of its picks the
    # tensor scores at zero. Thirty pods of the fitted field is a couple of seconds and
    # makes the figure reproducible from the shipped target.
    probe = draft.run_drafts(room.board, room.field_cfg, PRICEABLE_PROBE_DRAFTS,
                             np.random.default_rng(seed), pod_size=room.pod_size)
    unpriced = (~np.asarray(room.scorable, dtype=bool))[
        probe.roster.reshape(-1, ROSTER_SIZE)]
    dropped = {"n_dropped": int(len(room.frame) - len(frame)),
               "n_dropped_priced": int((~room.scorable
                                        & np.isfinite(room.board.adp)).sum()),
               "n_board": int(len(frame)),
               "n_priced": int(np.isfinite(room.board.adp[keep]).sum()),
               "field_unpriced_per_entry": float(unpriced.sum(axis=1).mean()),
               "field_entries_with_unpriced": float((unpriced.sum(axis=1) > 0).mean()),
               "probe_entries": int(unpriced.shape[0])}
    frame["board_rank"] = np.arange(len(frame), dtype=np.int32)
    board = draft.to_arrays(frame, room.season)

    from src.sim.bracket import position_masks
    masks = position_masks(frame)
    dk_pts = np.ascontiguousarray(room.dk_pts[keep])
    field_round = draft_room.build_field(frame, board, dk_pts, masks,
                                         room.round_of_period, room.field_cfg,
                                         n_field_drafts, seed, room.pod_size)
    out = draft_room.Room(
        season=room.season, frame=frame, board=board, dk_pts=dk_pts,
        scorable=np.ones(len(frame), dtype=bool), masks=masks,
        round_of_period=room.round_of_period,
        projection=dk_pts.sum(axis=1).mean(axis=1), pod_size=room.pod_size,
        field_cfg=room.field_cfg, field_round=field_round,
        refs={t: draft_room.field_reference(field_round, t) for t in room.refs},
        fit_window=room.fit_window)
    return out, dropped


# ── 5. Drafting a portfolio ───────────────────────────────────────────────────

def entry_seats(n_entries: int, pod_size: int) -> np.ndarray:
    """Which seat each entry drafts from, spread across the pod.

    DK randomizes the draft slot once the lobby fills, so a portfolio of ten entries holds
    ten different draft positions and a strategy measured only from seat 0 is measured on a
    board that never runs out of first-round talent. Spread deterministically rather than
    drawn, so the sweep is reproducible from its config alone.
    """
    if n_entries >= pod_size:
        return np.arange(n_entries) % pod_size
    return np.round(np.linspace(0, pod_size - 1, n_entries)).astype(int)


def strategy_keys(strategy: Strategy, value_rank: np.ndarray, adp_rank: np.ndarray,
                  round_index: int) -> np.ndarray:
    """The board key in force at one round — **lower is taken earlier**.

    A blend is a convex combination of two *ranks*, which keeps `alpha` in the units the
    disagreement was measured in (picks) and makes `alpha = 1` the market's board exactly
    and `alpha = 0` the model's. Blending values instead would be at the mercy of the
    recalibration's 55-wide plateaus, which is the same reason `src/sim/draft.py` puts its
    noise on the rank.
    """
    a = strategy.alpha_at(round_index)
    return a * adp_rank + (1.0 - a) * value_rank


def stacking_bonus(strategy: Strategy, teams: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Board places of credit for a teammate of somebody we already hold.

    In **picks**, so it reads against `alpha` and against `rank_cushion` on the same scale.
    A same-team pair is the one correlation a drafter can buy deliberately, and the sign is
    not obvious: they share the game's overtimes and its blowouts, which is upside a
    zero-consolation knockout pays for, and they share a fixed pot of team minutes, which
    `make minutes-unification` measured as a mean pairwise `r = -0.0509`. The simulator
    carries both, so the sweep can price the net rather than assume it.
    """
    if not strategy.stacking or not len(held):
        return np.zeros(len(teams), dtype=np.float64)
    mine = set(teams[held].tolist())
    return -float(strategy.stacking) * np.isin(teams, list(mine)).astype(np.float64)


def draft_portfolio(room: draft_room.Room, strategy: Strategy, tournament: str,
                    n_entries: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Draft `n_entries` twelve-seat pods under one strategy. Returns `[entry, 16]`.

    One `run_drafts` call per entry rather than one vectorized call for all of them, and
    that is forced twice over: each entry sits in a **different seat** (DK randomizes the
    slot), and **exposure caps couple the entries**, so entry `k`'s legal set depends on
    what entries `0..k-1` took. A pod is 192 argmaxes over ~450 players, so the loop costs
    nothing measurable next to the scoring.

    The opponents are `src/sim/draft.py`'s fitted field. Our own seat runs under
    `manual_config` — DK's 8 G / 8 F / 3 C bind autodraft and not a person — with the
    strategy's own caps substituted back in if it chose to carry any.
    """
    board = room.board
    value = model_value(room.dk_pts, strategy.ranking, strategy.quantile)
    value_rank = rank_of(value)
    adp_rank = board.rank.astype(np.float64)
    teams = room.frame["team"].to_numpy()
    seats = entry_seats(n_entries, room.pod_size)

    cap_count = int(np.ceil(strategy.exposure_cap * n_entries - 1e-9))
    used = np.zeros(board.n_players, dtype=np.int32)
    our_cfg = replace(draft.manual_config(room.field_cfg),
                      position_caps=tuple(strategy.position_caps))

    rosters = np.empty((n_entries, ROSTER_SIZE), dtype=np.int64)
    n_capped = 0
    for entry in range(n_entries):
        seat = int(seats[entry])

        def our_pick(state, our, allowed):
            nonlocal n_capped
            live = allowed[0] & room.scorable
            free = live & (used < cap_count)
            if free.any():
                live = free
            elif live.any():
                # The cap yields rather than raising, exactly as DK's own position caps do
                # when every position is full: an exposure cap is a preference about the
                # portfolio and a pick is compulsory.
                n_capped += 1
            if not live.any():
                raise ValueError("no priceable player is legal for this seat")
            base = (value_rank if strategy.objective == "ranking"
                    else _objective_rank(room, state, our, tournament, strategy, live))
            key = strategy_keys(strategy, base, adp_rank, state.round_index)
            key = key + stacking_bonus(strategy, teams, state.roster_of(our, 0))
            choice = int(np.argmin(np.where(live, key, np.inf)))
            used[choice] += 1
            return np.array([choice], dtype=np.int64)

        state = draft.run_drafts(board, room.field_cfg, 1, rng,
                                 pod_size=room.pod_size, seat_strategies=None,
                                 our_seat=seat, our_pick=our_pick, our_cfg=our_cfg)
        rosters[entry] = state.roster_of(seat, 0)

    return rosters, {"seats": seats.tolist(), "exposure_cap_count": cap_count,
                     "exposure_binding_picks": n_capped,
                     "max_exposure": float(used.max() / n_entries)}


def _objective_rank(room: draft_room.Room, state: draft.DraftState, seat: int,
                    tournament: str, strategy: Strategy,
                    live: np.ndarray) -> np.ndarray:
    """A value ordering from the in-draft objective, for the arms that use one.

    `src/sim/draft_room.py` owns every one of these — the completed-roster plug-in, the
    matroid exchange, the survivor reweighting — and this reads its table rather than
    recomputing any of it. Candidates the room does not rank (already gone, or unpriceable)
    are pushed behind every candidate it does, so the key stays total.
    """
    table, _ = draft_room.evaluate(room, state, seat, tournament,
                                   objective=strategy.objective,
                                   top=int(live.sum()))
    out = np.full(room.board.n_players, float(room.board.n_players), dtype=np.float64)
    out[table["board_rank"].to_numpy().astype(int)] = np.arange(len(table),
                                                               dtype=np.float64)
    return out


# ── 6. Scoring a portfolio ────────────────────────────────────────────────────

def portfolio_outcome(round_totals: np.ndarray, ref: draft_room.FieldReference,
                      rng: np.random.Generator, draws: int = N_BOOTSTRAP) -> dict:
    """P(top 2 of 12), P(any entry advances), ROI, and intervals on all three.

    **The portfolio is the unit.** A per-entry mean cannot see differentiation at all: ten
    identical entries have the same `P(advance)` as one. What they do not have is the same
    `P(at least one advances)`, which is `1 - prod(1 - p)` computed *inside* a simulated
    season and then averaged — so entries whose fortunes rise and fall together are worth
    strictly less than the same mean spread across independent ones. Every hedging axis in
    the config is priced there and nowhere else.

    The bootstrap is two-way and the order matters: resample **sims** first, because a
    simulated season moves every entry in the portfolio at once, then **entries** inside the
    resampled world. Resampling entries alone would report an interval that treats a
    portfolio's ten correlated entries as ten independent draws.
    """
    ev, p = draft_room.bracket_ev(round_totals, ref, per_sim=True)
    n_entries, n_sims = ev.shape
    fee = ref.entry_fee
    trace = p.mean(axis=0)                     # the portfolio's P(advance) per world

    def stats(e: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
        return (float(q.mean()),
                float((1.0 - np.prod(1.0 - q, axis=0)).mean()),
                float(e.mean()) / fee - 1.0)

    any_trace = 1.0 - np.prod(1.0 - p, axis=0)
    point = stats(ev, p)
    boot = np.empty((draws, 3))
    for b in range(draws):
        s = rng.integers(0, n_sims, n_sims)
        k = rng.integers(0, n_entries, n_entries)
        boot[b] = stats(ev[np.ix_(k, s)], p[np.ix_(k, s)])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return {"p_advance": point[0], "p_advance_lo": lo[0], "p_advance_hi": hi[0],
            "p_any_advance": point[1], "p_any_lo": lo[1], "p_any_hi": hi[1],
            "roi": point[2], "roi_lo": lo[2], "roi_hi": hi[2],
            "ev": float(ev.mean()), "entry_fee": fee, "n_entries": n_entries,
            "n_sims": n_sims, "trace": trace, "trace_any": any_trace}


def paired_gaps(traces: dict, baseline: str, tournament: str, metric: str,
                rng: np.random.Generator, draws: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Every strategy against one baseline, **paired on the simulated season**.

    The unpaired intervals in the sweep table are wide enough to cover most of the table at
    once, and that is a property of the *level* rather than of the differences: a simulated
    season that is kind to one strategy is kind to all of them, because every entry is
    scored on the same twenty weeks of the same drawn world. Differencing inside the sim
    removes that common term entirely, which is the same reason every model comparison in
    this repo is a paired bootstrap over rows rather than two independent intervals.

    `traces[(strategy, tournament)]` is one number per simulated season, seasons
    concatenated. Resampling those indices resamples worlds, which is the axis the
    uncertainty actually lives on.

    **Two metrics go through it, and the second is not decoration.** `p_advance` is the
    portfolio's mean `P(top 2 of 12)` and is what the sweep selects on. `p_any` is
    `P(at least one entry advances)`, and it is the only place an exposure cap or a
    differentiation rule can pay at all — capping exposure lowers every individual entry's
    chance by forcing it off the board's best players, so judging that axis on the selection
    criterion alone measures only its cost.
    """
    base = traces.get((baseline, tournament))
    if base is None:
        return pd.DataFrame()
    rows = []
    names = sorted({name for name, t in traces if t == tournament})
    for name in names:
        gap = traces[(name, tournament)] - base
        idx = rng.integers(0, len(gap), size=(draws, len(gap)))
        boot = gap[idx].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"tournament": tournament, "metric": metric, "baseline": baseline,
                     "strategy": name,
                     "gap": float(gap.mean()), "gap_lo": float(lo), "gap_hi": float(hi),
                     "p_gap_below_zero": float((boot < 0).mean()),
                     "resolved": bool(lo > 0.0 or hi < 0.0),
                     "n_worlds": int(len(gap))})
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


def roster_divergence(a: np.ndarray, b: np.ndarray) -> dict:
    """How different two portfolios are — Gate D's measurement.

    Mean pairwise roster overlap between the two, against the **within-portfolio** overlap
    as its control. That control is the whole measurement: two portfolios of ten entries
    each drawn from the same 450-player board will always share a lot of players, and the
    question is whether they share *more* with themselves than with each other.
    """
    def overlap(x: np.ndarray, y: np.ndarray, same: bool) -> float:
        vals = []
        for i in range(len(x)):
            for j in range(len(y)):
                if same and j <= i:
                    continue
                vals.append(len(set(x[i]) & set(y[j])) / ROSTER_SIZE)
        return float(np.mean(vals)) if vals else np.nan

    pool_a, pool_b = set(a.ravel().tolist()), set(b.ravel().tolist())
    return {"cross_overlap": overlap(a, b, False),
            "within_a": overlap(a, a, True), "within_b": overlap(b, b, True),
            "distinct_a": len(pool_a), "distinct_b": len(pool_b),
            "jaccard_pools": len(pool_a & pool_b) / max(len(pool_a | pool_b), 1),
            "only_a": len(pool_a - pool_b), "only_b": len(pool_b - pool_a)}


# ── 7. The table ──────────────────────────────────────────────────────────────

def strategy_table(objective_arm: bool = True) -> list[Strategy]:
    """The sweep, as rows. Every axis `docs/simulations-plan.md` names gets a ladder.

    Ordered so the cheap ranking arms run first: the objective arm calls
    `draft_room.evaluate` once per pick and costs ~100x what a board lookup does, so a
    failure in the plumbing surfaces in seconds — the same argument `make stan` makes for
    running the game-length head before the composition.
    """
    rows = [Strategy("model_mean", ranking="model_mean", axis="ranking"),
            Strategy("adp", ranking="adp", axis="ranking")]
    for q in (0.60, 0.75, 0.90):
        rows.append(Strategy(f"model_q{int(q * 100)}", ranking="model_quantile",
                             quantile=q, axis="ranking"))
    for a in (0.15, 0.30, 0.50, 0.70, 0.85):
        rows.append(Strategy(f"blend_a{int(a * 100)}", ranking="blend", alpha=a,
                             axis="alpha"))
    # Per-round schedules. The first pair is the direction `docs/adp-plan.md` predicts —
    # trust the model where the two agree, lean on the market in the deep rounds where they
    # do not — and the second is its reverse, which is the control: an axis that only ever
    # ships one direction is an axis nobody tested.
    for name, sched in (("blend_late", (0.15, 0.35, 0.65)),
                        ("blend_late_hard", (0.10, 0.40, 0.85)),
                        ("blend_early", (0.65, 0.35, 0.15))):
        rows.append(Strategy(name, ranking="blend", alpha=float(np.mean(sched)),
                             alpha_rounds=sched, axis="alpha_by_round"))
    for cap in (0.6, 0.4):
        rows.append(Strategy(f"blend_exposure{int(cap * 100)}", ranking="blend",
                             alpha=0.30, exposure_cap=cap, axis="exposure"))
    for s in (4.0, 12.0):
        rows.append(Strategy(f"blend_stack{int(s)}", ranking="blend", alpha=0.30,
                             stacking=s, axis="stacking"))
    rows.append(Strategy("blend_caps_dk", ranking="blend", alpha=0.30,
                         position_caps=tuple(draft.POSITION_CAPS[p] for p in POSITIONS),
                         axis="position_caps"))
    if objective_arm:
        for obj in ("lineup_value", "bracket_ev"):
            rows.append(Strategy(f"{obj}", ranking="model_mean", objective=obj,
                                 axis="objective"))
            rows.append(Strategy(f"{obj}_blend30", ranking="blend", alpha=0.30,
                                 objective=obj, axis="objective"))
    return rows


def sweep(cfg: dict, room: draft_room.Room, truth_dk: np.ndarray, season: str,
          strategies: list[Strategy], seed: int = SEED,
          n_field_drafts: int = N_FIELD_DRAFTS
          ) -> tuple[pd.DataFrame, dict, pd.DataFrame, tuple[dict, dict]]:
    """Every strategy against every entered tier, on one injected world.

    Returns the table, the drafted portfolios, the symmetric-field null check, and the two
    per-world traces the paired comparison is built from.

    The field is drafted and scored **once** and reused across the whole table, which is the
    only reason this is a table at all: `docs/simulations-plan.md` sizes a real field at
    176M lineup solves per season, and rebuilding it per strategy would put the sweep out of
    reach. Entries are exchangeable within a tier, so one field serves both.

    **`draft_room.null_check` runs on that field before any strategy does**, and it is the
    guard rather than a formality: an entry drawn from the field must reach round 1 at
    exactly `n_advance / pod_size`. Every lift the table reports is measured against that
    number, so if the scoring were tilted — a stale reference, a mis-scaled truth tensor, a
    board that no longer lines up with the tensor — the whole table would be tilted with it
    and every row would still look plausible.
    """
    entered = cfg.get("sim", {}).get("tournaments", {})
    masks = room.masks
    print(f"  drafting the reference field: {n_field_drafts} twelve-seat pods on the "
          f"INJECTED world")
    field_round = draft_room.build_field(room.frame, room.board, truth_dk, masks,
                                         room.round_of_period, room.field_cfg,
                                         n_field_drafts, seed, room.pod_size)
    refs = {t: draft_room.field_reference(field_round, t) for t in entered}
    nulls = {t: symmetric_null(t) for t in entered}
    null_table = pd.DataFrame([{"season": season,
                                **draft_room.null_check(field_round, refs[t])}
                               for t in entered])
    for row in null_table.itertuples():
        print(f"    null check {row.tournament:<19} P(top 2 of 12) "
              f"{row.p_advance_simulated:.6f} against {row.p_advance_analytic:.6f} "
              f"({row.p_advance_error:+.2e});  E[payout] "
              f"${row.ev_simulated:,.2f} against ${row.ev_analytic:,.2f} "
              f"({row.ev_relative_error:+.1%})")

    rows, portfolios, traces, any_traces = [], {}, {}, {}
    for strategy in strategies:
        for tournament, spec in entered.items():
            n_entries = strategy.n_entries or int(spec.get("entries", 1))
            rng = np.random.default_rng(seed)
            rosters, meta = draft_portfolio(room, strategy, tournament, n_entries, rng)
            scored = score_rosters(truth_dk, rosters, masks, room.round_of_period)
            out = portfolio_outcome(scored["round_total"].astype(np.float64),
                                    refs[tournament], np.random.default_rng(seed + 1))
            traces[(strategy.name, tournament)] = out.pop("trace")
            any_traces[(strategy.name, tournament)] = out.pop("trace_any")
            null = nulls[tournament]
            rows.append({"season": season, "tournament": tournament,
                         **strategy.as_row(), "n_entries": n_entries,
                         **out,
                         "p_advance_null": null["p_advance_round_1"],
                         "lift_vs_null": out["p_advance"] - null["p_advance_round_1"],
                         # The null is exact rather than estimated, so the interval on the
                         # lift is the interval on `p_advance` shifted — no second bootstrap
                         # and no extra uncertainty from the baseline.
                         "lift_lo": out["p_advance_lo"] - null["p_advance_round_1"],
                         "lift_hi": out["p_advance_hi"] - null["p_advance_round_1"],
                         "break_even_hurdle": null["break_even_hurdle"],
                         "roi_null": null["roi"],
                         "roi_beats_hurdle": bool(out["roi"] > 0.0),
                         "seats": ",".join(str(s) for s in meta["seats"]),
                         "max_exposure": meta["max_exposure"],
                         "exposure_binding_picks": meta["exposure_binding_picks"]})
            portfolios[(strategy.name, tournament)] = rosters
    return pd.DataFrame(rows), portfolios, null_table, (traces, any_traces)


# ── 8. Gates D and F, and the shipped artifact ────────────────────────────────

def select(table: pd.DataFrame, tournament: str) -> str:
    """Which strategy ships for one tier — highest lift in `P(top 2 of 12)`, pooled.

    Pooled over seasons rather than picked per season, because a per-season winner on two
    seasons is a coin toss dressed as a selection. Ties break toward the lower `alpha` and
    then toward the simpler row, so a strategy has to *earn* the market weight it carries.
    """
    sub = table[table["tournament"] == tournament]
    agg = (sub.groupby(["strategy", "alpha", "objective"], as_index=False)
           .agg(lift=("lift_vs_null", "mean"), p_any=("p_any_advance", "mean")))
    agg = agg.sort_values(["lift", "p_any", "alpha"], ascending=[False, False, True])
    return str(agg["strategy"].iloc[0])


def gate_d(portfolios: dict, shipped: dict, seasons: list[str],
           strategies: list[Strategy]) -> pd.DataFrame:
    """Do the two tiers actually select different rosters?

    Reported as a finding rather than a footnote, per `docs/simulations-plan.md`: if the $20
    and $52 strategies converge, either the objective is not doing its job or the tier
    difference is smaller than the economics imply, and either way it has to be known before
    entering.

    **The comparison has to name its own mechanism or the answer is uninterpretable.** A
    `ranking` strategy is *tier-blind by construction* — the board key knows nothing about
    which payout table it is drafting into — so under those arms the tiers can only diverge
    by selecting different rows of the table or by holding different numbers of entries. The
    tier-aware arms are the ones that read the structure: `bracket_ev` and `p_advance` price
    each candidate against that tournament's own pods, advance counts and cash bands. Both
    are reported, with `tier_aware` marking which is which, so "the tiers converge" can be
    read as the measurement it is rather than as a property of an objective that was never
    given the chance to differ.

    The control is the **within**-tier overlap. Two ten-entry portfolios drawn from the same
    350-player board share a lot of players whatever they are doing; the question is whether
    they share more with themselves than with each other.
    """
    tiers = list(shipped)
    aware = {s.name: s.objective in ("bracket_ev", "p_advance") for s in strategies}
    pairs = [("shipped", shipped[tiers[0]], shipped[tiers[1]])]
    pairs += [("tier_aware", s.name, s.name) for s in strategies
              if aware.get(s.name) and s.name != shipped[tiers[0]]]

    rows = []
    for season in seasons:
        for label, name_a, name_b in pairs:
            a = portfolios.get((season, name_a, tiers[0]))
            b = portfolios.get((season, name_b, tiers[1]))
            if a is None or b is None:
                continue
            d = roster_divergence(a, b)
            rows.append({"season": season, "comparison": label,
                         "tier_a": tiers[0], "tier_b": tiers[1],
                         "strategy_a": name_a, "strategy_b": name_b,
                         "same_strategy": name_a == name_b,
                         "tier_aware": bool(aware.get(name_a) and aware.get(name_b)),
                         **d,
                         "materially_different": bool(
                             d["cross_overlap"]
                             < min(d["within_a"], d["within_b"]) - 0.02)})
    return pd.DataFrame(rows)


# ── 9. The realized readout ───────────────────────────────────────────────────

def replay_realized(cfg: dict, room: draft_room.Room, season: str,
                    strategies: list[Strategy], portfolios: dict, seed: int = SEED,
                    n_field_drafts: int = N_FIELD_DRAFTS) -> pd.DataFrame:
    """Replay the portfolios against the season that actually happened.

    **A readout, not a selector.** One season is one world, so a portfolio's outcome here is
    a single draw and the only thing that can be resampled is the *field* — which is real
    uncertainty (DK randomizes the pod and the eleven opponents are not the same eleven
    twice) but is not season uncertainty. Two seasons is the ceiling on the honest edge
    estimate, and it is named rather than solved.

    **The portfolios are the sweep's own, not redrafted.** They would come back identical —
    same room, same seed, same board — because the board a drafter has in October is the
    **model view** either way and only the scoring changes. Passing them in makes that an
    identity rather than a coincidence, and it halves the run: under the tier-aware arms a
    redraft is sixteen `draft_room.evaluate` calls per entry.
    """
    entered = cfg.get("sim", {}).get("tournaments", {})
    truth = realized_tensor(cfg, season, room.frame["player_id"].to_numpy(),
                            len(room.round_of_period))
    field_round = draft_room.build_field(room.frame, room.board, truth, room.masks,
                                         room.round_of_period, room.field_cfg,
                                         n_field_drafts, seed, room.pod_size)
    refs = {t: draft_room.field_reference(field_round, t) for t in entered}

    rows = []
    for strategy in strategies:
        for tournament in entered:
            rosters = portfolios.get((strategy.name, tournament))
            if rosters is None:
                continue
            scored = score_rosters(truth, rosters, room.masks, room.round_of_period)
            out = portfolio_outcome(scored["round_total"].astype(np.float64),
                                    refs[tournament], np.random.default_rng(seed + 1))
            out.pop("trace"), out.pop("trace_any")
            null = symmetric_null(tournament)
            rows.append({"season": season, "tournament": tournament,
                         **strategy.as_row(), "n_entries": int(len(rosters)), **out,
                         "p_advance_null": null["p_advance_round_1"],
                         "lift_vs_null": out["p_advance"] - null["p_advance_round_1"],
                         "break_even_hurdle": null["break_even_hurdle"],
                         "n_worlds": 1})
    return pd.DataFrame(rows)


def ship(table: pd.DataFrame, realized: pd.DataFrame, cfg: dict,
         strategies: list[Strategy]) -> pd.DataFrame:
    """The artifact item 10 reads: which strategy ships for each tier, and on what.

    `src/final_evaluation.py`'s rule, applied one layer up — the test runner **reads which
    strategy shipped** rather than re-deciding it, so the held-out backtest cannot become a
    selection by accident. Everything needed to rebuild the policy is a column here, because
    an artifact that names a strategy but not its parameters is a strategy that has to be
    re-derived to be used.
    """
    by_name = {s.name: s for s in strategies}
    rows = []
    for tournament, spec in cfg.get("sim", {}).get("tournaments", {}).items():
        name = select(table, tournament)
        s = by_name[name]
        sub = table[(table["tournament"] == tournament) & (table["strategy"] == name)]
        rl = realized[(realized["tournament"] == tournament)
                      & (realized["strategy"] == name)]
        null = symmetric_null(tournament)
        rows.append({"tournament": tournament, **s.as_row(),
                     "n_entries": int(spec.get("entries", 1)),
                     "entry_fee": null["entry_fee"],
                     "break_even_hurdle": null["break_even_hurdle"],
                     "selected_on": "lift in P(top 2 of 12), pooled over validation seasons",
                     "sim_p_advance": float(sub["p_advance"].mean()),
                     "sim_lift": float(sub["lift_vs_null"].mean()),
                     "sim_p_any_advance": float(sub["p_any_advance"].mean()),
                     "sim_roi": float(sub["roi"].mean()),
                     "sim_roi_lo": float(sub["roi_lo"].mean()),
                     "sim_roi_hi": float(sub["roi_hi"].mean()),
                     "realized_p_advance": float(rl["p_advance"].mean())
                     if len(rl) else np.nan,
                     "realized_lift": float(rl["lift_vs_null"].mean())
                     if len(rl) else np.nan,
                     "realized_roi": float(rl["roi"].mean()) if len(rl) else np.nan,
                     "realized_seasons": int(rl["season"].nunique()) if len(rl) else 0})
    return pd.DataFrame(rows)


# ── 10. The entry point ────────────────────────────────────────────────────────

def truth_sim_index(n_sims: int, n_truth: int = 24) -> np.ndarray:
    """Which sims stand in for realized worlds — thinned, never a prefix.

    Sim `s` uses posterior draw `s % n_draws`, so a prefix of the sim axis is a contiguous
    stretch of the posterior masquerading as a sample of worlds. Same rule
    `stan_utils.thin` applies to draws, one level up.
    """
    return np.unique(np.linspace(0, n_sims - 1, min(n_truth, n_sims)).astype(int))


def run(cfg: dict, seasons: list[str] | None = None, n_sims: int | None = None,
        seed: int | None = None, objective_arm: bool = True,
        n_field_drafts: int | None = None) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_sim = cfg.get("sim", {})
    cfg_strategy = cfg_sim.get("strategy", {})
    n_sims = int(n_sims or cfg_strategy.get("n_sims", N_SIMS_SWEEP))
    n_field_drafts = int(n_field_drafts
                         or cfg_strategy.get("field_drafts", N_FIELD_DRAFTS))
    seed = int(SEED if seed is None else seed)

    design = split_frame(cfg)
    seasons = seasons or validation_seasons(design)
    for season in seasons:
        assert_season_allowed(season, design)

    strategies = strategy_table(objective_arm)
    entered = cfg_sim.get("tournaments", {})

    print("Strategy sweep — error injection (Gate C), the table, and both backtests")
    print(f"  The test split is LOCKED — seasons go through `held_out.selection_split`.")
    print(f"  {len(strategies)} strategies x {len(entered)} tiers x {len(seasons)} "
          f"seasons, {n_sims:,} simulated worlds each")
    print(f"  select on LIFT IN P(top 2 of 12); report ROI against the break-even hurdle")

    gate_c_rows, injection_rows, sweep_rows, realized_rows, null_rows = [], [], [], [], []
    portfolios: dict = {}
    pooled: dict = {}
    pooled_any: dict = {}
    for season in seasons:
        print(f"\n── {season} ──")
        full = draft_room.load_room(cfg, season, n_sims=n_sims)
        room, dropped = priceable_room(cfg, full, seed, n_field_drafts)
        print(f"  board restricted to the {dropped['n_board']:,} players the tensor "
              f"prices: {dropped['n_dropped']:,} dropped, {dropped['n_dropped_priced']:,} "
              f"of them carrying ADP. Unrestricted, the ADP field drafts "
              f"{dropped['field_unpriced_per_entry']:.2f} zero-scoring players per "
              f"sixteen-man entry and "
              f"{dropped['field_entries_with_unpriced'] * 100:.0f}% of its entries hold at "
              f"least one — a handicap on the field, not model edge "
              f"(see `priceable_room`)")
        games_played = board_games_played(features_dir, season, room, n_sims)
        truth_sims = truth_sim_index(n_sims)

        truth, gate, record = gate_c(cfg, season, room.dk_pts, games_played,
                                     room.scorable,
                                     room.frame["player_id"].to_numpy(), room.frame,
                                     truth_sims)
        gate_c_rows.append(gate)
        injection_rows.append(record | dropped)
        _report_gate_c(season, gate, record)

        table, folios, nulls, traces = sweep(cfg, room, truth, season, strategies, seed,
                                             n_field_drafts)
        sweep_rows.append(table)
        null_rows.append(nulls)
        for (name, tournament), rosters in folios.items():
            portfolios[(season, name, tournament)] = rosters
        # Seasons are concatenated along the world axis, so a paired resample draws from
        # both. Two seasons is not two independent worlds and the readout says so; what
        # this buys is a comparison between strategies inside each world, which is exactly
        # the term the unpaired interval cannot remove.
        for store, block in zip((pooled, pooled_any), traces):
            for key, trace in block.items():
                store[key] = (np.concatenate([store[key], trace]) if key in store
                              else trace)

        print(f"  replaying the same portfolios against realized {season}")
        realized_rows.append(replay_realized(cfg, room, season, strategies, folios,
                                             seed, n_field_drafts))

    table = pd.concat(sweep_rows, ignore_index=True)
    realized = pd.concat(realized_rows, ignore_index=True)
    shipped = {t: select(table, t) for t in entered}
    # `blend_early` is here as a **control** rather than as a contender: it reverses the
    # per-round schedule `docs/adp-plan.md` predicts, so `blend_late` minus `blend_early` is
    # a direct paired test of the direction rather than two readings off a common baseline.
    # `blend_a30` is the exposure and stacking arms' own uncapped twin — same ranking, same
    # alpha, hedge off. Comparing those axes to `model_mean` instead folds the blend's gain
    # into the hedge's, which is the difference between measuring an axis and measuring the
    # row it happens to sit in.
    baselines = ["model_mean", "adp", "blend_early", "blend_a30"]
    paired = pd.concat(
        [paired_gaps(store, base, t, metric, np.random.default_rng(seed + 2))
         for store, metric in ((pooled, "p_advance"), (pooled_any, "p_any_advance"))
         for t in entered
         for base in dict.fromkeys(baselines + [shipped[t]])],
        ignore_index=True)
    gate_d_table = gate_d(portfolios, shipped, seasons, strategies)
    ship_table = ship(table, realized, cfg, strategies)

    _report_sweep(table, shipped, entered)
    _report_paired(paired, shipped, entered, strategies)
    _report_gate_d(gate_d_table)
    _report_realized(realized, shipped)

    paths = {}
    for name, frame in (("strategy_gate_c", pd.concat(gate_c_rows, ignore_index=True)),
                        ("strategy_injection", pd.DataFrame(injection_rows)),
                        ("strategy_null", pd.concat(null_rows, ignore_index=True)),
                        ("strategy_sweep", table),
                        ("strategy_paired", paired),
                        ("strategy_gate_d", gate_d_table),
                        ("strategy_realized", realized),
                        ("strategy_shipped", ship_table)):
        dest = out_dir / f"{name}.csv"
        frame.to_csv(dest, index=False)
        print(f"Saved {len(frame):,} {name.replace('_', ' ')} rows → {dest}")
        paths[name] = dest
    return paths


def board_games_played(features_dir: Path, season: str, room: draft_room.Room,
                       n_sims: int) -> np.ndarray:
    """The games-played twin, lined up with the room's board rather than the tensor's.

    `draft.tensor_scores` pads the tensor out to the board and marks the rows it could not
    price; the twin has to travel the same way or Gate C's availability row would be
    measured on a different player set from its dk_pts rows.
    """
    with np.load(features_dir / f"sim_tensor_{season}.npz", allow_pickle=False) as z:
        gp, player_id = z["games_played"], z["player_id"]
    rows = {int(p): i for i, p in enumerate(player_id)}
    index = room.frame["player_id"].map(rows)
    out = np.zeros((len(room.frame), gp.shape[1], n_sims), dtype=np.uint8)
    ok = index.notna().to_numpy()
    out[ok] = gp[index[ok].to_numpy().astype(int), :, :n_sims]
    return out


def _report_gate_c(season: str, gate: pd.DataFrame, record: dict) -> None:
    print(f"\nGate C — does the simulated world carry the model's measured miss?")
    print(f"  measured on realized {season}, {record['n_adp_realized']} priced players:")
    print(f"    Spearman against realized season totals — model "
          f"{record['spearman_model_realized']:.4f}, market "
          f"{record['spearman_market_realized']:.4f}, gap "
          f"{record['skill_gap_realized']:+.4f}")
    print(f"    corr(market disagreement, model error) "
          f"{record['market_rho_realized']:+.4f} (R2 "
          f"{record['market_r2_realized']:.4f});  MAE model "
          f"{record['mae_model_realized']:.1f}, market "
          f"{record['mae_market_realized']:.1f}")
    print(f"  the UNINJECTED world has it backwards — model "
          f"{record['spearman_model_uninjected']:.4f}, market "
          f"{record['spearman_market_uninjected']:.4f}, gap "
          f"{record['skill_gap_uninjected']:+.4f}. That is the bias Gate C exists for, "
          f"measured rather than asserted.")
    print(f"  injection: rotate onto the market direction at rho="
          f"{record['rho']:+.4f} (solved from the skill gap; the error correlation says "
          f"{record['market_rho_realized']:+.4f} independently), scale "
          f"g={record['scale_g']:.4f} "
          f"({record['clipped_cells'] * 100:.2f}% of cells clipped, "
          f"{record['floored_cells'] * 100:.2f}% below the rate floor); the injected "
          f"world's own error correlation reads "
          f"{record['market_rho_injected']:+.4f}")
    for row in gate.itertuples():
        bar = (f"band [{row.bar_lo:.4f}, {row.bar_hi:.4f}]" if np.isnan(row.bar)
               else f"bar {row.bar:9.4f}")
        print(f"  {row.check:<20} uninjected {row.uninjected:10.4f}   injected "
              f"{row.injected:10.4f}   {bar}   [{row.artifact}]")


def _report_sweep(table: pd.DataFrame, shipped: dict, entered: dict) -> None:
    print("\nThe sweep — pooled over validation seasons, ordered by lift in "
          "P(top 2 of 12)")
    for tournament in entered:
        sub = table[table["tournament"] == tournament]
        agg = (sub.groupby(["strategy", "axis"], as_index=False)
               .agg(lift=("lift_vs_null", "mean"), llo=("lift_lo", "mean"),
                    lhi=("lift_hi", "mean"), p_any=("p_any_advance", "mean"),
                    roi=("roi", "mean"), lo=("roi_lo", "mean"), hi=("roi_hi", "mean"))
               .sort_values("lift", ascending=False))
        hurdle = float(sub["break_even_hurdle"].iloc[0])
        print(f"\n  {tournament}  (break-even hurdle {hurdle:+.2%}; an ADP-drafted entry's "
              f"P(top 2 of 12) is {float(sub['p_advance_null'].iloc[0]):.6f} exactly)")
        for row in agg.itertuples():
            mark = "*" if row.strategy == shipped[tournament] else " "
            print(f"  {mark} {row.strategy:<22} {row.axis:<14} "
                  f"lift {row.lift:+.4f} [{row.llo:+.4f}, {row.lhi:+.4f}]  "
                  f"P(any of N) {row.p_any:.4f}  "
                  f"ROI {row.roi:+.3f} [{row.lo:+.3f}, {row.hi:+.3f}]")


def _report_paired(paired: pd.DataFrame, shipped: dict, entered: dict,
                   strategies: list[Strategy]) -> None:
    axis = {s.name: s.axis for s in strategies}
    print("\nPaired on the simulated season — the comparison the unpaired intervals "
          "above cannot make")
    for tournament in entered:
        adv = paired[(paired["tournament"] == tournament)
                     & (paired["metric"] == "p_advance")]
        base = adv[adv["baseline"] == "model_mean"]
        if base.empty:
            continue
        rest = base[base["strategy"] != "model_mean"]
        print(f"\n  {tournament} — lift over `model_mean`, "
              f"{int(rest['resolved'].sum())} of {len(rest)} arms resolved on "
              f"{int(base['n_worlds'].iloc[0]):,} worlds")
        for row in rest.itertuples():
            print(f"    {row.strategy:<22} {axis.get(row.strategy, ''):<14} "
                  f"{row.gap:+.4f} [{row.gap_lo:+.4f}, {row.gap_hi:+.4f}]"
                  f"{'' if row.resolved else '   (not resolved)'}")

        late = adv[(adv["baseline"] == "blend_early")
                   & (adv["strategy"] == "blend_late")]
        if len(late):
            row = late.iloc[0]
            print(f"    the per-round DIRECTION, against its own reverse: blend_late "
                  f"- blend_early {row['gap']:+.4f} [{row['gap_lo']:+.4f}, "
                  f"{row['gap_hi']:+.4f}]"
                  f"{'  resolved' if row['resolved'] else '  not resolved'}")

        twin = "blend_a30"
        anyv = paired[(paired["tournament"] == tournament)
                      & (paired["metric"] == "p_any_advance")
                      & (paired["baseline"] == twin)]
        hedges = anyv[anyv["strategy"].map(lambda n: axis.get(n) in ("exposure",
                                                                    "stacking"))]
        if len(hedges):
            print(f"    the hedging axes against their own uncapped twin `{twin}` — "
                  f"per-entry lift is what the sweep selects on, P(any of N) is where "
                  f"they act:")
            for row in hedges.itertuples():
                lift = adv[(adv["baseline"] == twin)
                           & (adv["strategy"] == row.strategy)].iloc[0]
                print(f"      {row.strategy:<20} lift {lift['gap']:+.4f} "
                      f"[{lift['gap_lo']:+.4f}, {lift['gap_hi']:+.4f}]   "
                      f"P(any) {row.gap:+.4f} [{row.gap_lo:+.4f}, {row.gap_hi:+.4f}]")

        rivals = adv[(adv["baseline"] == shipped[tournament])
                     & (adv["strategy"] != shipped[tournament])]
        if len(rivals):
            best = rivals.iloc[0]
            print(f"    shipped `{shipped[tournament]}`: "
                  f"{int(rivals['resolved'].sum())} of {len(rivals)} rivals separated "
                  f"from it; the nearest is {best['strategy']} at {best['gap']:+.4f} "
                  f"[{best['gap_lo']:+.4f}, {best['gap_hi']:+.4f}]")


def _report_gate_d(table: pd.DataFrame) -> None:
    print("\nGate D — do the two tiers select materially different rosters?")
    if table.empty:
        print("  no comparable portfolios")
        return
    for row in table.itertuples():
        aware = "tier-aware objective" if row.tier_aware else "tier-BLIND ranking"
        print(f"  {row.season} [{row.comparison}] {row.strategy_a} ({row.tier_a}) vs "
              f"{row.strategy_b} ({row.tier_b}) — {aware}")
        print(f"    roster overlap across tiers {row.cross_overlap:.3f} against within-tier "
              f"{row.within_a:.3f} / {row.within_b:.3f};  "
              f"pool Jaccard {row.jaccard_pools:.3f}, "
              f"{row.only_a} players only in {row.tier_a}, {row.only_b} only in "
              f"{row.tier_b}")
        print(f"    materially different: {row.materially_different}")


def _report_realized(realized: pd.DataFrame, shipped: dict) -> None:
    print("\nThe realized readout — 2 seasons, wide intervals, and NOT a selector")
    for tournament, name in shipped.items():
        sub = realized[(realized["tournament"] == tournament)
                       & (realized["strategy"] == name)]
        base = realized[(realized["tournament"] == tournament)
                        & (realized["strategy"] == "adp")]
        for row in sub.itertuples():
            b = base[base["season"] == row.season]
            print(f"  {row.season} {tournament:<19} {name:<20} "
                  f"P(adv) {row.p_advance:.4f} [{row.p_advance_lo:.4f}, "
                  f"{row.p_advance_hi:.4f}]  lift {row.lift_vs_null:+.4f}  "
                  f"ROI {row.roi:+.3f} [{row.roi_lo:+.3f}, {row.roi_hi:+.3f}]"
                  + (f"   (an ADP entry: {float(b['p_advance'].iloc[0]):.4f})"
                     if len(b) else ""))
    print("  The interval above resamples the FIELD and the entries, not the season. "
          "N = 2 seasons is the ceiling on the honest edge estimate and no resampling "
          "manufactures a third.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", action="append", default=None,
                        help="target season; repeatable. Defaults to validation.")
    parser.add_argument("--n-sims", type=int, default=None)
    parser.add_argument("--field-drafts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-objective-arm", action="store_true",
                        help="skip the in-draft-objective arms, which call "
                             "`draft_room.evaluate` once per pick")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, seasons=args.season, n_sims=args.n_sims, seed=args.seed,
        objective_arm=not args.no_objective_arm, n_field_drafts=args.field_drafts)
