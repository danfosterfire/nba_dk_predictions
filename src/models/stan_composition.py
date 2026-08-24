"""The team-game minutes composition: a decomposed multinomial over who played.

The season minutes head (`make stan-minutes`) predicts each player's minutes
marginally. Two per-game facts it cannot represent: a team's minutes sum to exactly
`5 x game_length` (zero-sum — the mechanism behind teammate-absence redistribution),
and no player exceeds `game_length`. Each existing form gets one constraint exactly
and the other not at all. This head gets both **by construction**: each team-game's
N minutes are allocated among the K players who played by decomposing the multinomial
into sequential binomial trials (`src/stan/demo_decomposed_multinomial.stan` is the
pilot of the idea), ordered by prior-season minutes share, with the cap enforced
through the trials — `m_k = min(U, R_k)`, the remaining capacity.

Design and gates: `docs/minutes-composition-plan.md`. Settled there: one pooled fit
with N/U as per-row data (never per-OT-class fits), K = players who played (the
availability head owns the played margin), and a pilot window before any full-window
commitment.

**Game length is an INPUT to this head, and it is no longer defined here.** This module
used to carry `fit_ot_tail` / `sample_game_length` — a two-parameter point-MLE geometric
tail, parked here because the composition was the first thing that needed a game-length
class. `src/models/stan_game_length.py` owns it as of 2026-08-09: a Bayesian head with a
full posterior and a fitted season trend, seven seconds of sampler time rather than a
by-product of a nine-hour one. Nothing here changed — this head consumes the *realized*
`game_length` column on every row it fits or scores, and only a forward simulation needs
the draw.

## The offset is the floor, and the floor is already a redistribution model

Every step's linear predictor starts from the carry-forward conditional share:
`logit_prior = logit(w_k / tail_k x R_k / m_k)` with `w` the prior-share composition
renormalized over who played. At `alpha = beta = 0` the mean allocation is exactly
prior shares carried forward — the mandatory no-fit floor, sharing the fitted model's
code path — and a missing teammate shrinks the renormalizer, scaling every remaining
allocation up proportionally. `beta` fits *deviations* from proportional
redistribution.

## What is deliberately NOT here

- Serial structure: iid across games given features. The 2.43x block inflation stays
  with the residual serial process (`docs/predictions-plan.md`).
- DNP-CD: the played/not margin belongs to the availability head.
- A rookie can NOT be dropped — the sum must be complete — so no-prior players get an
  expanding-window draft-bucket share prior instead of the `>= 200 prior minutes`
  filter every other head uses. That is the structural difference from `stan_minutes`.

## VALIDATION-ONLY — code and artifact, since 2026-08-08

Converted 2026-08-05 with every other head: the test seasons are not fitted or scored here,
`sweep` emits `val_*` columns only, and `src/models/held_out.py` raises on anything that
reaches past validation. The artifact was regenerated on 2026-08-08, closing a three-day
window in which the code was converted and `outputs/predictions/stan_composition_*.csv`
still carried `test_*` columns. **Nothing about the head's verdict reversed** — same
selected arm, same ordering of all six rows, same gate outcomes. The retired test column is
preserved in `docs/minutes-composition-plan.md` under `Claim(historical=True)`.

**The re-run was deferred twice and then taken, and the deferral argument was wrong in a way
worth recording.** It rested on "the run buys no decision, and it costs 12–15 h". The first
half was true and beside the point: leaving held-out figures in an artifact leaves them in
the docs and on the dashboard, where they get quoted as the head's performance and become
the bar a successor arm is measured against — which is exactly how the games-played Gate D
acquired test-set bars. A stale artifact is not inert. **The second half was simply
wrong**, see below.

**Doubling chain length cost 33%, not 100%, and that is why the original "halves it" was
right.** The 2026-08-06 deferral corrected "the conversion roughly halves the cost" to
"saves ~40%, 12.6–14.7 h", by assuming the surviving validation fits would take twice as
long once selection moved from `select_warmup`/`select_samples` to full length. They did
not. Measured per fit: the four val fits went **7.35 h → 9.78 h**, a **1.33×** rise for a
2× iteration increase, and `betabinom/val` actually ran *faster* at double the iterations
(9,358 s against 9,615 s). Longer warmup buys a better-adapted step size, which buys fewer
leapfrog steps per iteration, which partly pays for the extra iterations. Total run
**9.92 h** against the 21.13 h two-pass run — a **53%** saving. The lesson is the one this
head keeps teaching: **sampler cost is not linear in the knob you are turning**, and an
estimate derived from an untested proportionality is worth less than the run it replaces.

**This head has no registered final evaluation.** `src/final_evaluation.py` registers
`availability`, `games_played` and `season_total` and not this one, so nothing currently
takes composition's held-out reading. Registering it is a full-window refit on
train+validation plus one scoring pass, which is a decision for whoever needs that number.

Usage:
    python -m src.models.stan_composition
"""

import pickle
import zlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml
from scipy.special import betaln, gammaln, roots_hermite
from scipy.stats import betabinom

from src.eda.availability import with_lags
from src.features.team_context import DRAFT_BUCKETS, UNDRAFTED_BUCKET
from src.models.availability import (EPS, FEATURE_COLS, RHO_MIN,
                                     fit_dispersion)
from src.models.held_out import selection_split
from src.models.component_rates import impute
from src.models.stan_availability import availability_design
from src.models.stan_minutes import (SPLINE_KNOTS, StanMinutes, beta_shapes,
                                     build_design as minutes_build_design,
                                     game_level_dispersion)
from src.models.stan_minutes import variants as minutes_variants
from src.models.stan_utils import (compile_model, crps_from_samples,
                                   diagnostics_frame, ks_uniform,
                                   pit_from_samples, posterior,
                                   prior_sd_for_l2, sample, standardized, thin,
                                   warn_if_unconverged)

MODEL = "composition_glm"
OWN = "logit_share_lag1"
PREDICTIVE_SAMPLES = 200
GLM_L2 = 1.0
INTERCEPT_SCALE = 5.0
RHO_INIT = 0.05
PILOT_FIRST_SEASON = "2018-19"
MAX_EXTRAPOLATED_HOURS = 24.0
TEST_SEASONS = 2

# Only reachable for the first two data seasons, which no pilot window ever fits —
# after that the expanding rookie prior always has observations.
FALLBACK_ROOKIE_SHARE = 0.15

# The stick-breaking offset's saturation bounds, and they are a MODEL choice, not a
# numerical guard. On ~1% of rows the proportional carry-forward demands MORE than
# the cap — a star on a roster whose played-set prior shares sum well below 5 gets
# ratio x R/m > 1 (measured: 6,511 of 665,318 non-last rows) — and an aggressive
# clip at 1-1e-3 asserts "he plays 47.95 of 48" with a near-zero beta shape
# parameter. The data routinely disagree by 5-15 minutes, the curvature at those
# rows is enormous, and the sampler's step size collapses: the first smoke run spent
# 60+ minutes inside warmup at max treedepth. 0.93 saturates at ~44.6 of 48 — a
# realistic "the cap binds" allocation — and the `offset_clipped` indicator lets
# beta learn the correction on exactly those rows.
OFFSET_CLIP = (0.02, 0.93)
# Bounds on the composition weight `w`. The lower bound keeps the stick-breaking
# offset finite for end-of-bench players; the upper mirrors RHO_MAX-style guard rails.
SHARE_CLIP = (EPS, 0.95)

# Ordered most-decision-relevant first. `sweep` iterates this order, so at full window
# the graded arm and its shared-rho twin — the pair the whole graded-vs-shared contrast
# rests on — land in the first third of the run rather than the last. Selection is
# order-independent (`mark_selection` uses `idxmin`), so only CSV row order moves.
FITTED_VARIANTS = ("betabinom_ot_graded", "betabinom_ot", "betabinom", "binomial")
GROUP_KEYS = ["game_id", "team_id"]

# The dispersion is graded over bins of this column — the prior-season minutes share
# that already orders the sequence and sets the offset. Bins rather than a functional
# form because the pilot measured the *pattern* (variance ratio 1.59 fringe -> 0.70
# star) and not a shape, and because a bin count is one honest knob.
RHO_BIN_COL = "w_share"
RHO_BINS = 4

# The simulator redraws a step that lands below the feasibility bound; past this many
# attempts it clamps to the bound and counts it. The bound binds only in
# short-rotation games (K = 6 occurs in 10 of 71,092 team-games).
MAX_REDRAWS = 50

# ── The per-(player, season) random effect ────────────────────────────────────

# The unit the effect is indexed by. NOT the player: a minutes role is a property of the
# season a player is in, and a career-long effect is already carried by `logit_share_lag1`
# and the stick-breaking offset.
UNIT_KEYS = ["player_id", "season"]

# ── The marginal (quadrature) representation of the same effect ───────────────
# `docs/composition-quadrature-plan.md`. Step 0's measurement, on the 2,062 pilot-window
# units that carry a likelihood row, of the WORST per-unit error in the marginal
# log-likelihood over sigma in [0.1, 1.0] — nodes placed at the no-fit floor and scored at
# the converged theta, which is the drift this driver actually faces:
#
#   scheme                             Q=11      Q=15      Q=21      Q=25      Q=41    Q=61
#   fixed nodes (the plan's first)        —         —    9.3e+0        —    4.1e+0  2.6e+0
#   this rule, inflate 1.0             2.6e+0   8.5e-1   7.6e-2   7.4e-3        —       —
#   this rule, inflate 1.2 (SHIPPED)   8.3e-1   6.0e-2   2.5e-6   4.9e-7        —       —
#   this rule, inflate 1.6             1.2e-2   3.1e-3   4.4e-4   1.1e-4        —       —
#
# Two things follow. The node count is a property of the PLACEMENT and not of the integrand
# — the same integral is 2.6 nats out at Q = 61 badly placed and 2.5e-6 at Q = 21 well
# placed, which is why `unit_laplace` is worth solving properly. And the tolerance the
# shipped pair clears (2.5e-6 nats on the worst unit, 2.3e-6 pooled over all 2,062, against
# per-unit log-likelihoods of order 100) is far under anything NUTS can resolve.
#
# ⚠️ `GH_INFLATE` is NOT a "wider is safer" knob and treating it as one makes the rule worse
# — see the 1.6 row, which is two orders of magnitude off the 1.2 row at every Q. Too-wide
# nodes step over a narrow spike exactly as badly as too-narrow ones miss its tails. 1.2 is
# a measured interior optimum, not a margin.
Q_NODES = 21
GH_INFLATE = 1.2

# Newton's step count and finite-difference width for the per-unit mode and curvature.
# Both are DRIVER-side constants on a 1-D concave-ish problem solved once at fixed data;
# nothing here enters a Stan gradient, so accuracy is nearly free.
LAPLACE_ITERS = 40
LAPLACE_H = 1e-3

# Guard rails on the fitted mode, and they are load-bearing rather than defensive.
#
# A unit with two rows and a dispersed likelihood is nearly FLAT in `u`, so its mode is not
# identified and Newton walks off — measured at 21.2 on a synthetic two-row unit. Left alone
# that misplaces the nodes catastrophically: at sigma = 0.65 the centre lands at 6.29 with a
# spread of 0.65, so every node sits in the far tail and the integral is 0.77 nats wrong
# against a dense grid.
#
# The fix is to let an uninformative unit BE uninformative. `CURV_MIN` is a floor at
# essentially zero rather than at prior strength, so `prec = curv + 1/sigma^2` collapses to
# the prior, the centre shrinks to ~0 and the nodes become the prior's own — which is the
# right placement for a unit the data says nothing about. Flooring curvature *up* to the
# prior's precision instead (the first thing tried here) keeps the wandered mode and is
# exactly how the misplacement above happened.
CURV_MIN = 1e-8
MODE_CLIP = 10.0

# Half-normal scale on `sigma_u`, on the LINEAR PREDICTOR scale. Deliberately loose. The
# two data-implied figures `make minutes-unification` reports are a log-ratio sd of 0.284
# and a logit-share sd of 0.493, and the injection sweep's CRPS optimum sat at 0.375-0.45 —
# so a half-normal(0, 1) is weakly informative by roughly a factor of two and the data
# decides. Tightening it around the injection's value would make Gate P5 ("the fit
# reproduces the injection") a foregone conclusion rather than a replication.
U_SD_SCALE = 1.0

# The team-context block, from `data/features/team_context_tierA.parquet`. Five columns,
# not the file's eleven: the composition head is the most expensive fit in the project and
# the deviation it is trying to explain is only ~5% predictable from pre-season information
# at all, so the block is headed by `role_crowding` — the minutes-weighted archetype
# similarity, which is the only column that can see that a departing star matters more to
# his positional replacement than to the roster average — plus the usage aggregates and the
# roster size. `teammate_assist_supply`, `teammate_spacing` and `team_pace` are production
# context rather than minutes-allocation context and are left out.
TEAM_COLS = ["role_crowding", "teammate_usage_max", "teammate_usage_sum",
             "teammate_usage_load", "n_teammates"]
TEAM_MISSING = "team_missing"

# When `dense_e` is worth adapting. TWO constraints, and the second is the binding one —
# which cost a wasted hour to learn on 2026-08-09 and is recorded so nobody re-learns it.
#
# **Memory** is quadratic in parameters and rules out only the widest window:
#
#   one-season probe    635 params      3.1 MB     0.09 GFLOP per Cholesky
#   pilot window      2,234 params     38.1 MB     3.72 GFLOP
#   full window      12,337 params  1,161.2 MB   625.90 GFLOP
#
# **Estimability** rules out all three. A dense metric estimates a P x P covariance from
# the WARMUP draws, so it needs draws on the order of the parameter count — and the
# random-effect arms have nothing like that:
#
#   635 params from 1,000 warmup draws   ->  1.57 draws per parameter
#   2,234 params from 1,000 warmup draws ->  0.45 draws per parameter
#
# At under one draw per parameter the adaptation is rank-deficient, CmdStan's regularization
# shrinks it back toward diagonal, and the extra cost buys nothing. Measured: the `ps` arm
# under `dense_e` ran past **an hour** on rows `diag_e` finished in 25 minutes, and was
# killed rather than finished. So the original instinct — no dense metric once the effect is
# on — was right, and the reasoning behind it (memory) was wrong; correcting the reasoning
# without correcting the rule made it worse. `DENSE_DRAWS_PER_PARAM` is what actually gates
# it, and at 1,000 warmup draws that means ~50 parameters, which is the regime the
# effect-free head lives in and the reason `dense_e` was measured to help there.
DENSE_METRIC_MAX_MB = 256.0
DENSE_DRAWS_PER_PARAM = 20.0


def choose_metric(n_params: int, warmup: int = 1000,
                  cap_mb: float = DENSE_METRIC_MAX_MB,
                  draws_per_param: float = DENSE_DRAWS_PER_PARAM) -> str:
    """`dense_e` only when the mass matrix both FITS and can be ESTIMATED.

    The estimability test is the one that bites: a dense metric adapted from fewer warmup
    draws than it has parameters is a rank-deficient covariance that CmdStan shrinks back
    toward diagonal, so it costs the Cholesky and buys none of the conditioning.
    """
    if (n_params ** 2) * 8 / 1024 ** 2 > cap_mb:
        return "diag_e"
    return "dense_e" if warmup >= draws_per_param * n_params else "diag_e"


def announce_metric(n_features: int, n_rho: int, warmup: int, n_units: int = 0,
                    n_sigma: int = 0) -> str:
    """The metric an arm will get, printed **before** the sampler starts.

    ⚠️ **A cost cliff, not a preference, and it is invisible in the artifact until the fit
    is over.** `choose_metric` grants `dense_e` only when `warmup >= 20 x parameters`, and
    this head's own probe measured NUTS held at treedepth 8-9 under `diag_e` against
    treedepth 4 under `dense_e` — roughly 10x the wall clock. A `composition_preseason_fit`
    attempt at `warmup: 500`, copied from the `effects` block, ran **32 minutes without
    completing its 500 warmup draws**, against `composition_effects`' **880 s** for the same
    arm at the same window for a whole 500+500 fit under `dense_e`.

    Nothing said so until it was killed: CmdStan writes no draw until warmup ends and its
    progress lines are buffered away, so the *only* early signal is the metric itself. This
    prints it in the first second.

    **A `diag_e` warning is not always actionable, which is why this warns rather than
    raises.** An arm carrying the per-(player, season) effect has `U_n` parameters — 2,204
    at the pilot window — and cannot reach `dense_e` at any warmup this project would run;
    for it `diag_e` is the correct metric and the long comment above `DENSE_METRIC_MAX_MB`
    is why. The warning is for the arms that could have had it and were configured out of
    it by a number nobody re-read.

    Lives here rather than in a caller because `choose_metric` is here: the two were in
    different modules for one day and the copy in `composition_preseason_fit` had no `U_n`
    term, so it would have promised `dense_e` to an arm that structurally cannot get it.
    """
    n_params = n_features + 1 + n_rho + n_units + n_sigma
    metric = choose_metric(n_params, warmup)
    print(f"    {n_params:,} parameters, {warmup} warmup draws → {metric}")
    if metric != "dense_e":
        needed = DENSE_DRAWS_PER_PARAM * n_params
        if n_units:
            print(f"    (expected: {n_units:,} player-season effects put `dense_e` out of "
                  f"reach at any warmup here, and `diag_e` is the right metric for it)")
        else:
            print(f"    /!\\  `diag_e` on this head is ~10x the wall clock of `dense_e` "
                  f"(treedepth 8-9 against 4 on its own probe). `dense_e` needs warmup >= "
                  f"{needed:.0f} and this run has {warmup}.")
    return metric


def unit_codes(frame: pd.DataFrame) -> np.ndarray:
    """0-based (player, season) index per row, for an effect shared across a unit's games."""
    return frame.groupby(UNIT_KEYS, sort=True).ngroup().to_numpy()


class PlayerSeasonTerm:
    r"""The per-(player, season) random effect, held by `StanComposition`.

    Disabled by default, in which case `data` returns the `U_n = 0` block — zero-length
    `u_z` and `sigma_u` — and `shift` returns zeros. The disabled head is then the model
    that existed before this block, exactly, rather than "the same model with a small
    coefficient". `stan_utils.YearTerm` is the pattern; the differences are the index and
    what sharing means.

    ## The fitted `u_z` are discarded, for the same reason `year_z` is

    A fitted `u_z[unit]` describes a player-season that is over. The seasons this project
    forecasts have no `u`, and carrying one forward would be a player-season fixed effect
    smuggled in — the exact thing that is unusable at prediction time. So the predictive
    integrates over a **fresh** `z ~ N(0, 1)`:

        eta_new = logit_prior + alpha + x'beta + sigma_u * z,  z ~ N(0, 1)

    one `z` per (unit, posterior draw), **shared across that unit's games within a draw**.
    That sharing is the whole mechanism: per-game noise averages down by ~1/sqrt(G) when
    summed to a season while a season-level shift passes through in full, which is the
    4.68x the gate measured. Drawing an independent `z` per game would reproduce the
    per-game marginal and buy none of the season-level spread.

    ## What it does NOT break

    The effect enters the linear predictor of a *step*, so the sequential allocation still
    hands the team exactly `5 x game_length` minutes and still caps every player at
    `game_length`. A player-season shifted up takes its minutes from a teammate, which is
    the dynamic the head exists for — unlike a shared shift, which is definitionally a
    re-allocation and buys no spread at all.
    """

    def __init__(self, enabled: bool = False, scale: float = U_SD_SCALE,
                 seed: int = 42, stream: str = "", centered: bool = False,
                 sampled: bool = True):
        self.enabled = bool(enabled)
        # Whether the LATENTS go to NUTS. `False` with `enabled=True` is the marginal
        # representation: `QuadratureTerm` supplies the likelihood, this class still owns
        # `sigma_u` and the predictive, and Stan is handed the `U_n = 0` block — which the
        # model rejects seeing alongside `Q > 0`, deliberately, since the two are
        # alternatives rather than layers.
        self.sampled = bool(sampled)
        self.scale, self.seed, self.stream = float(scale), int(seed), str(stream)
        # Same model, different coordinates for NUTS to walk. Non-centred by default because
        # the funnel it avoids produces WRONG answers where the centred form's failure mode
        # is merely slow ones; `centered=True` is the response to a collapsed step size with
        # treedepth saturation, which is what a well-informed unit under a non-centred
        # parameterization looks like. `shift` is identical either way — the predictive of a
        # unit that has not happened is `sigma_u * z` under both.
        self.centered = bool(centered)
        self.sigma_draws = np.zeros(0)
        self.n_units = 0

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng([self.seed, zlib.crc32(self.stream.encode())])

    def data(self, train: pd.DataFrame) -> dict:
        """The `U_n` / `unit_idx` / `u_sd_scale` keys. Stan has no optional data, so the
        disabled block is passed too — with `U_n = 0` and every index 0, which the model
        never reads."""
        if not self.enabled or not self.sampled:
            # The unit count is still recorded when the effect is on but marginalized, so
            # `summary` reports the same thing under both representations.
            self.n_units = (int(unit_codes(train).max()) + 1
                            if self.enabled and len(train) else 0)
            return {"U_n": 0, "unit_idx": [0] * len(train),
                    "u_sd_scale": float(self.scale), "u_centered": 0}
        codes = unit_codes(train)
        self.n_units = int(codes.max()) + 1 if len(codes) else 0
        return {"U_n": self.n_units, "unit_idx": (codes + 1).astype(int).tolist(),
                "u_sd_scale": float(self.scale), "u_centered": int(self.centered)}

    def absorb(self, fit) -> None:
        """Keep the `sigma_u` draws; `u_z` is deliberately never stored.

        Kept **two-dimensional**, `(draws x n_sigma)`, for the reason `rho_draws` already is:
        the graded and shared arms then take one downstream path and `n_sigma = 1` is the
        shared model exactly rather than a special case somebody has to remember.
        """
        if not self.enabled:
            self.sigma_draws = np.zeros(0)
            return
        draws = np.atleast_1d(fit.stan_variable("sigma_u"))
        self.sigma_draws = np.asarray(draws).reshape(len(draws), -1)

    @staticmethod
    def _unit_bins(frame: pd.DataFrame, codes: np.ndarray, n_units: int) -> np.ndarray:
        """Each unit's 0-based sigma bin, read off the row column it is graded over.

        `rho_bin` is a function of `w_share`, which is a (player, season) quantity — so it
        is constant within a unit by construction and the first row's value IS the unit's.
        """
        if "rho_bin" not in frame.columns or not n_units:
            return np.zeros(n_units, dtype=int)
        order = np.argsort(codes, kind="stable")
        first = np.searchsorted(codes[order], np.arange(n_units))
        return frame["rho_bin"].to_numpy(dtype=int)[order][first] - 1

    def shift(self, frame: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        """`(rows x draws)` fresh `sigma_u * z`, shared across a unit's rows per draw.

        `idx` is the same thinned posterior-draw index `_eta_base` used, so the sigma
        applied to a column is the sigma of the draw whose `alpha` and `beta` built it.

        With a graded `sigma_u` each unit takes the sigma of **its own** bin. A scalar or
        one-column `sigma_draws` — the shipped injection, and every artifact written before
        2026-08-16 — broadcasts to every unit, which is the same arithmetic as before.
        """
        idx = np.asarray(idx)
        if not self.enabled or self.sigma_draws.size == 0:
            return np.zeros((len(frame), len(idx)))
        codes = unit_codes(frame)
        n_units = int(codes.max()) + 1 if len(codes) else 0
        z = self._rng().standard_normal(size=(n_units, len(idx)))
        sigma = self.sigma_draws
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        if sigma.shape[1] == 1:
            per_unit = sigma[idx, 0][None, :]                 # broadcast over units
        else:
            per_unit = sigma[idx][:, self._unit_bins(frame, codes, n_units)].T
        return (per_unit * z)[codes, :]

    def summary(self) -> dict:
        if not self.enabled or self.sigma_draws.size == 0:
            return {"ps_effect": False, "sigma_u": 0.0, "sigma_u_sd": 0.0,
                    "sigma_u_lo": 0.0, "sigma_u_hi": 0.0, "n_units": 0,
                    "u_parameterization": "none"}
        draws = self.sigma_draws
        if draws.ndim == 1:
            draws = draws[:, None]
        # The headline stays the mean over bins, so a shared-sigma arm reports exactly what
        # it always did and the gate code needs no special case; the per-bin columns are
        # additive and are what the graded-vs-shared contrast is read from.
        pooled = draws.mean(axis=1)
        lo, hi = np.percentile(pooled, [2.5, 97.5])
        out = {"ps_effect": True,
               "sigma_u": float(pooled.mean()),
               "sigma_u_sd": float(pooled.std(ddof=1)),
               "sigma_u_lo": float(lo), "sigma_u_hi": float(hi),
               "n_units": int(self.n_units),
               "u_parameterization": "centered" if self.centered else "non_centered"}
        if draws.shape[1] > 1:
            for b in range(draws.shape[1]):
                out[f"sigma_u_bin{b + 1}"] = float(draws[:, b].mean())
        return out


# ── The marginal representation: quadrature over each unit's effect ───────────

def gauss_hermite(q: int) -> tuple[np.ndarray, np.ndarray]:
    """Abscissae and `log(w) + x^2` for `int e^{-x^2} g(x) dx ~ sum_q w_q g(x_q)`.

    The `e^{x^2}` is folded into the weight here rather than in Stan because it is data and
    because the two factors overflow and underflow in opposite directions — carrying them
    separately into a log-space accumulation is how a node quietly becomes a NaN.
    """
    x, w = roots_hermite(int(q))
    return x, np.log(w) + x ** 2


def _unit_arrays(train: pd.DataFrame) -> dict:
    """The unit-major view of the likelihood rows: order, offsets, bins.

    Rows carrying likelihood are the non-`is_last` ones. A unit whose every row is a
    deterministic remainder carries **no likelihood at all** and is simply absent — 142 of
    2,204 at the pilot window — which is correct rather than a hole: its marginal is
    `int phi(z) dz = 1` and contributes nothing to the target.
    """
    codes = unit_codes(train)
    live = np.flatnonzero(train["is_last"].to_numpy(dtype=int) == 0)
    order = live[np.argsort(codes[live], kind="stable")]
    unit_of_row = codes[order]
    _, first, lens = np.unique(unit_of_row, return_index=True, return_counts=True)
    bins = (train["rho_bin"].to_numpy(dtype=int)[order][first]
            if "rho_bin" in train.columns else np.ones(len(first), dtype=int))
    return {"rows": order, "starts": first, "lens": lens, "bins": bins,
            "unit_of_row": np.repeat(np.arange(len(first)), lens)}


def _unit_loglik(u: np.ndarray, eta0: np.ndarray, y: np.ndarray, m: np.ndarray,
                 s: np.ndarray, starts: np.ndarray, unit_of_row: np.ndarray
                 ) -> np.ndarray:
    """`(n_units x nodes)` summed beta-binomial log-pmf at per-unit shifts `u`.

    The driver's own copy of the model block's inner sum, used **only** to place the nodes.
    It omits the feasibility-bound normalization that 1.7% of rows carry, and that is a
    deliberate approximation rather than an oversight: the placement has to absorb the whole
    drift from the no-fit floor to the converged posterior (a mode shift of order 0.5), which
    step 0 measured Q = 21 handling at 2.5e-6 nats. A 1.7%-of-rows normalization is a far
    smaller perturbation than the one the node count is already sized for.
    """
    eta = eta0[:, None] + u[unit_of_row, :]
    p = 0.5 * (1.0 + np.tanh(0.5 * np.clip(eta, -400.0, 400.0)))
    a = s[:, None] * p
    b = s[:, None] * (1.0 - p)
    ll = (gammaln(m + 1.0)[:, None] - gammaln(y + 1.0)[:, None]
          - gammaln(m - y + 1.0)[:, None]
          + betaln(y[:, None] + a, (m - y)[:, None] + b) - betaln(a, b))
    return np.add.reduceat(ll, starts, axis=0)


def unit_laplace(eta0: np.ndarray, y: np.ndarray, m: np.ndarray, s: np.ndarray,
                 starts: np.ndarray, unit_of_row: np.ndarray,
                 iters: int = LAPLACE_ITERS, h: float = LAPLACE_H
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit mode and curvature of the log-likelihood in `u`, by safeguarded Newton.

    **This is where the node count comes from, and it is worth doing properly.** Step 0
    measured the same integral needing Q > 61 under fixed nodes, Q ~ 31 under the plan's
    "shrunken deviation + rough information" rule, and Q = 21 under this one. It is solved
    once, at fixed data, in the driver — a better constant, not an inner optimization, and
    it never enters a Stan gradient.

    Both outputs are **sigma-free**: they describe the LIKELIHOOD alone, and the model block
    combines them with the current `sigma_u` in closed form. That is what lets one placement
    stay accurate wherever the sampler takes sigma — step 0 checked 0.1 through 1.0.
    """
    n_units = len(starts)
    mode = np.zeros(n_units)
    probe = np.array([-h, 0.0, h])

    def at(u):
        return _unit_loglik(u, eta0, y, m, s, starts, unit_of_row)

    for _ in range(int(iters)):
        ll = at(mode[:, None] + probe[None, :])
        grad = (ll[:, 2] - ll[:, 0]) / (2 * h)
        curv = (ll[:, 2] - 2 * ll[:, 1] + ll[:, 0]) / h ** 2
        # A raw Newton step where the local curvature is not negative would run uphill
        # forever; those units get a bounded gradient step instead.
        concave = curv < -1e-10
        step = np.clip(np.where(concave, -grad / np.where(concave, curv, -1.0),
                                np.sign(grad) * 0.1), -1.0, 1.0)
        better = at((mode + step)[:, None])[:, 0] >= ll[:, 1]
        for _ in range(3):
            if better.all():
                break
            step = np.where(better, step, 0.5 * step)
            better = at((mode + step)[:, None])[:, 0] >= ll[:, 1]
        # Clipped every iteration, not once at the end: a nearly-flat unit accepts a long
        # run of tiny improvements and drifts, and a mode of 21 on the logit scale is a
        # likelihood with no maximum rather than a player who played 21 logits more.
        mode = np.clip(np.where(better, mode + step, mode), -MODE_CLIP, MODE_CLIP)
        if np.abs(np.where(better, step, 0.0)).max() < 1e-9:
            break

    ll = at(mode[:, None] + probe[None, :])
    curvature = -(ll[:, 2] - 2 * ll[:, 1] + ll[:, 0]) / h ** 2
    # A unit whose likelihood is flat or indefinite in `u` is one the data says nothing
    # about. Say so: a ~zero curvature makes `prec` the prior's own precision, which shrinks
    # the centre to 0 and puts the nodes exactly where an uninformative unit wants them.
    weak = ~(curvature > CURV_MIN)
    return np.where(weak, 0.0, mode), np.where(weak, CURV_MIN, curvature)


class QuadratureTerm:
    r"""The per-(player, season) effect, MARGINALIZED rather than sampled.

    `PlayerSeasonTerm` above hands the `u_i` to NUTS and hands back `sigma_u`. This hands
    NUTS **nothing**: each unit's `u_i` is integrated out inside the model block by
    Gauss-Hermite quadrature, so the posterior carries ~35 parameters at *any* fitting
    window instead of 2,204 at the pilot and 12,307 at the full one. Three things follow,
    and they are the whole case for the representation:

    * the funnel that made `ps` a hard geometry cannot exist, because there is no latent;
    * `dense_e` — which `choose_metric` grants only at warmup >= 20 x parameters, and which
      this head's own probe measures at ~10x the wall clock of `diag_e` — comes back;
    * the full window becomes reachable, and the full-window sigma is the one that would
      actually ship.

    The cost is `Q` evaluations of the likelihood per gradient instead of one. That is the
    trade the plan is: `Q = 21` against a parameter block that never converged.

    **Disabled by default**, in which case `data` returns the `Q = 0` block and the head is
    the model that existed before this class, exactly — the same device `PlayerSeasonTerm`
    and `stan_utils.YearTerm` use.

    The fitted `sigma_u` is per **sigma bin** from day one (`n_sigma = 1` nests the shared
    model). The bins are the head's own `rho_bin` — quantiles of `w_share`, already assigned
    per row and constant within a unit — because the injection's own per-role calibration
    found a ~2x gradient in what sigma wants, on the same axis and in the same direction as
    the fitted per-game `rho`. Reusing that column rather than minting a second one keeps
    "which bin is this player in" a single fact.
    """

    def __init__(self, enabled: bool = False, nodes: int = Q_NODES,
                 inflate: float = GH_INFLATE, n_sigma: int = 1,
                 scale: float = U_SD_SCALE):
        self.enabled = bool(enabled)
        self.nodes = int(nodes)
        self.inflate = float(inflate)
        self.n_sigma = int(n_sigma)
        self.scale = float(scale)
        self.n_units = 0
        self.placement: dict = {}

    def data(self, train: pd.DataFrame) -> dict:
        """The `Q` / `gh_*` / `u_*` keys. Stan has no optional data, so the disabled block
        is passed too — with `Q = 0` and every array empty, which the model never reads."""
        if not self.enabled:
            self.n_units = 0
            return {"Q": 0, "gh_x": [], "gh_log_w": [], "n_unit": 0, "u_start": [],
                    "u_len": [], "u_bin": [], "u_center": [], "u_curv": [],
                    "gh_inflate": self.inflate, "n_sigma": 1, "n_umap": 0, "u_row": []}

        arrays = _unit_arrays(train)
        rows = arrays["rows"]
        # Placed at the head's OWN NO-FIT FLOOR — eta = the stick-breaking offset, which is
        # alpha = beta = 0. Deliberately not read off a fitted posterior: placement changes
        # accuracy and never the estimand, so the only thing a fitted reference would buy is
        # a silent dependence on the artifact this head exists to replace.
        eta0 = train["logit_prior"].to_numpy(dtype=float)[rows]
        y = train["y"].to_numpy(dtype=float)[rows]
        m = train["m"].to_numpy(dtype=float)[rows]
        rho = _floor_dispersion_by_bin(train)
        bin_of_row = (train["rho_bin"].to_numpy(dtype=int)[rows] - 1
                      if "rho_bin" in train.columns else np.zeros(len(rows), dtype=int))
        s = (1 - rho[bin_of_row]) / rho[bin_of_row]

        center, curv = unit_laplace(eta0, y, m, s, arrays["starts"], arrays["unit_of_row"])
        gh_x, gh_log_w = gauss_hermite(self.nodes)
        self.n_units = len(arrays["starts"])
        self.placement = {"center": center, "curv": curv, "rho_floor": rho}
        n_bins = max(1, min(self.n_sigma, int(arrays["bins"].max())))
        # An EMPTY sigma bin is not a small problem. It would be sampled straight from its
        # half-normal prior, land in the artifact looking like a fitted quantity, and be
        # applied at predict time to whichever units fall in it there. The bins are train
        # quantiles of `w_share`, so all four are populated by construction — which is
        # exactly why a violation means something upstream changed rather than that the
        # window is small.
        present = set(np.minimum(arrays["bins"], n_bins).tolist())
        missing = sorted(set(range(1, n_bins + 1)) - present)
        if missing:
            raise ValueError(
                f"sigma bins {missing} carry no player-season unit, so their `sigma_u` "
                f"would be sampled from the prior and persisted as if it were fitted. "
                f"Check `rho_bin_edges` against this window's `{RHO_BIN_COL}`.")
        return {
            "Q": self.nodes,
            "gh_x": gh_x.tolist(),
            "gh_log_w": gh_log_w.tolist(),
            "n_unit": self.n_units,
            "u_start": (arrays["starts"] + 1).astype(int).tolist(),
            "u_len": arrays["lens"].astype(int).tolist(),
            "u_bin": (np.minimum(arrays["bins"], n_bins)).astype(int).tolist(),
            "u_center": center.tolist(),
            "u_curv": curv.tolist(),
            "gh_inflate": self.inflate,
            "n_sigma": n_bins,
            "n_umap": len(rows),
            "u_row": (rows + 1).astype(int).tolist(),
        }


def _floor_dispersion_by_bin(train: pd.DataFrame) -> np.ndarray:
    """Per-bin dispersion of the no-fit floor — the `rho` the node placement assumes.

    Fitted on the floor's own training steps, exactly as `FloorComposition` does, so the
    placement needs no posterior. It only has to be roughly right: `rho` enters the
    curvature through `1 / (1 + (m-1) rho)`, and the node count is already sized for a much
    larger perturbation than getting it a little wrong.
    """
    live = train[train["is_last"] == 0]
    p = 1.0 / (1.0 + np.exp(-live["logit_prior"].to_numpy(dtype=float)))
    y = live["y"].to_numpy(dtype=int)
    m = live["m"].to_numpy(dtype=int)
    if "rho_bin" not in live.columns:
        return np.array([fit_dispersion(y, m, p)])
    bins = live["rho_bin"].to_numpy(dtype=int)
    out = []
    for b in range(1, int(bins.max()) + 1):
        take = bins == b
        out.append(fit_dispersion(y[take], m[take], p[take]) if take.any() else RHO_INIT)
    return np.maximum(np.asarray(out, dtype=float), RHO_MIN)


# ── Frame construction ────────────────────────────────────────────────────────

def played_frame(panel: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    """Played player-games joined to their game length. Unfiltered, or the sum breaks.

    Built from `availability_panel.parquet`, never from `component_targets.parquet` /
    `game_logs.parquet` — both are `min_games`-filtered, and a team sum over a filtered
    frame is missing whole players' minutes (`CLAUDE.md`, game-length section).
    """
    for col in ("season", "game_id", "game_length", "n_overtimes", "reliable"):
        if col not in lengths.columns and col != "season":
            raise ValueError(f"game_length frame is missing {col}; run `make game-length`")

    reg = lengths[lengths["season_type"] == "regular"]
    played = panel[panel["played"] == 1]
    merged = played.merge(
        reg[["season", "game_id", "game_length", "n_overtimes", "reliable"]],
        on=["season", "game_id"], how="left")

    unmatched = int(merged["game_length"].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched:,} of {len(merged):,} played player-games have no game "
            f"length — check the merge keys and re-run `make game-length`.")
    if not merged["reliable"].all():
        bad = int((~merged["reliable"]).sum())
        print(f"  dropping {bad:,} player-games in games with unreliable length")
        merged = merged[merged["reliable"]]
    return merged.drop(columns=["reliable"])


def season_shares(played: pd.DataFrame) -> pd.DataFrame:
    """Realized minutes share per (player, season) — the unfiltered twin of
    `stan_minutes.minutes_targets`, on the same construction: only games he played,
    on both sides of the ratio."""
    out = (played.groupby(["player_id", "season"], as_index=False)
           .agg(minutes_played=("min", "sum"),
                length_played=("game_length", "sum"),
                games_played=("min", "size")))
    out["minutes_share"] = out["minutes_played"] / out["length_played"]
    return out


def share_lags(shares: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Prior share per (player, season): lag 1, falling back to lags 2-3 with a flag.

    `no_prior` marks rows with no share in any of the three prior seasons — the
    players every other head filters out and this one cannot.
    """
    lagged = with_lags(shares, seasons, ["minutes_share"], max_lag=3)
    l1, l2, l3 = (lagged[f"minutes_share_lag{k}"] for k in (1, 2, 3))
    lagged["prior_share"] = l1.fillna(l2).fillna(l3)
    lagged["share_stale"] = (l1.isna() & lagged["prior_share"].notna()).astype(float)
    lagged["no_prior"] = lagged["prior_share"].isna().astype(float)
    return lagged[["player_id", "season", "season_index", "minutes_share",
                   "prior_share", "share_stale", "no_prior"]]


def _draft_bucket(n: float) -> str:
    if pd.isna(n):
        return UNDRAFTED_BUCKET
    for lo, hi, name in DRAFT_BUCKETS:
        if lo <= n <= hi:
            return name
    return UNDRAFTED_BUCKET


def draft_numbers(features_dir: Path) -> pd.DataFrame:
    """Draft slot per (player, season) from the inclusive roster matrix, which covers
    every played player-season (verified: 0 missing)."""
    sm = pd.read_parquet(features_dir / "season_matrix_roster_tierA.parquet",
                         columns=["player_id", "season", "bio_draft_number"])
    sm = sm.rename(columns={"bio_draft_number": "draft_number"})
    sm["draft_bucket"] = sm["draft_number"].map(_draft_bucket)
    return sm


def rookie_share_priors(lagged: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Expanding-window mean realized share of no-prior players, per draft bucket.

    Point-in-time by construction: season S's prior averages seasons strictly before
    S, so it is knowable preseason. Season index 0 is excluded from the observations —
    with no earlier data *everyone* looks like a rookie there, and folding veterans
    into the bucket means would inflate every prior.
    """
    obs = lagged[(lagged["no_prior"] == 1) & (lagged["season_index"] >= 1)
                 & lagged["minutes_share"].notna()]
    rows = []
    for idx, season in enumerate(seasons):
        past = obs[obs["season_index"] < idx]
        overall = float(past["minutes_share"].mean()) if len(past) else np.nan
        for _, _, name in DRAFT_BUCKETS:
            rows.append({"season": season, "draft_bucket": name})
        rows.append({"season": season, "draft_bucket": UNDRAFTED_BUCKET})
        for row in rows[-(len(DRAFT_BUCKETS) + 1):]:
            in_bucket = past[past["draft_bucket"] == row["draft_bucket"]]
            row["rookie_share_prior"] = (float(in_bucket["minutes_share"].mean())
                                         if len(in_bucket) else overall)
            row["rookie_prior_n"] = int(len(in_bucket))
    out = pd.DataFrame(rows)
    out["rookie_share_prior"] = out["rookie_share_prior"].fillna(FALLBACK_ROOKIE_SHARE)
    return out


def order_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed within-team-season order: veterans by prior share descending, then
    no-prior players by draft slot, tie-broken by `player_id`. Computable preseason —
    every key is S-1 information — and each game's played subset inherits it.

    Sorts on `order_share` when the frame carries one and on `w_share` otherwise. The two
    are the same column for every caller but `composition_preseason`, which needs them
    separate to say whether a better prior share helps through the **offset** or through the
    **order** — `w_share` reaches the model both ways, and a single number that moved both
    could not tell them apart.
    """
    out = frame.copy()
    out["_w_neg"] = -(out["order_share"] if "order_share" in out else out["w_share"])
    out["_draft"] = out["draft_number"].fillna(99)
    out = out.sort_values(["season", "game_id", "team_id", "no_prior", "_w_neg",
                           "_draft", "player_id"]).reset_index(drop=True)
    out = out.drop(columns=["_w_neg", "_draft"])
    out["position"] = out.groupby(GROUP_KEYS, sort=False).cumcount()
    return out


def largest_remainder(frame: pd.DataFrame) -> np.ndarray:
    """Integer minutes per row, summing to exactly N per team-game, never above U.

    Floor everyone, then hand the deficit to the largest fractional remainders,
    skipping players already at the cap. A handful of games have a *negative* deficit
    (box-score minutes sum slightly past 5 x game_length — worst observed 3.08
    minutes), handled by removing from the smallest remainders instead. The frame must
    already be sorted into team-game blocks.
    """
    mins = frame["min"].to_numpy(dtype=float)
    caps = frame["U"].to_numpy(dtype=int)
    y = np.minimum(np.floor(mins), caps).astype(np.int64)
    frac = mins - np.floor(mins)

    starts = np.flatnonzero(frame["position"].to_numpy() == 0)
    bounds = np.append(starts, len(frame))
    targets = frame["N"].to_numpy(dtype=int)[starts]

    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        deficit = int(targets[i] - y[a:b].sum())
        if deficit == 0:
            continue
        order = np.argsort(-frac[a:b], kind="stable") if deficit > 0 \
            else np.argsort(frac[a:b], kind="stable")
        while deficit != 0:
            moved = False
            for j in order:
                if deficit > 0 and y[a + j] < caps[a + j]:
                    y[a + j] += 1
                    deficit -= 1
                    moved = True
                elif deficit < 0 and y[a + j] > 0:
                    y[a + j] -= 1
                    deficit += 1
                    moved = True
                if deficit == 0:
                    break
            if not moved:
                raise ValueError(f"cannot round team-game starting at row {a} to its "
                                 f"total — capacity exhausted")
    return y


def sequential_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """R, trials m = min(U, R), the feasibility bound, and the stick-breaking offset.

    All of these are **data** — functions of the observed allocation and the fixed
    order — which is what keeps the Stan likelihood fully vectorized. The simulator
    recomputes them sequentially from its own draws.
    """
    out = frame.copy()
    grp = out.groupby(GROUP_KEYS, sort=False)
    y = out["y"].to_numpy(dtype=np.int64)
    allocated_before = grp["y"].cumsum().to_numpy() - y
    n_total = out["N"].to_numpy(dtype=np.int64)
    caps = out["U"].to_numpy(dtype=np.int64)
    k_players = grp["y"].transform("size").to_numpy(dtype=np.int64)
    position = out["position"].to_numpy(dtype=np.int64)

    remaining = n_total - allocated_before
    trials = np.minimum(caps, remaining)
    players_after = k_players - position - 1
    lower = np.maximum(0, remaining - players_after * caps)
    out["R"] = remaining
    out["m"] = trials
    out["lo"] = lower
    out["is_last"] = (position == k_players - 1).astype(int)
    out["k_players"] = k_players

    if (y > trials).any():
        raise ValueError("y exceeds min(U, R) — the rounding or ordering is broken")
    if (y < lower).any():
        raise ValueError("y below the feasibility bound — the rounding is broken")

    w = out["w_share"].to_numpy(dtype=float)
    total_w = grp["w_share"].transform("sum").to_numpy()
    tail_w = total_w - (grp["w_share"].cumsum().to_numpy() - w)
    ratio = w / np.maximum(tail_w, 1e-12)
    raw = ratio * remaining / np.maximum(trials, 1)
    p_carry = np.clip(raw, *OFFSET_CLIP)
    out["stick_ratio"] = ratio
    out["offset_clipped"] = (raw != p_carry).astype(float)
    out["logit_prior"] = np.log(p_carry / (1 - p_carry))
    return out


def season_weights(shares: pd.DataFrame, seasons: list[str], features_dir: Path,
                   share_hook=None, drafts: pd.DataFrame | None = None
                   ) -> tuple[pd.DataFrame, list[str]]:
    """The per-(player, season) `w_share` block, and the columns it carries out.

    Extracted from `composition_frame` (2026-08-21) so the forward path
    (`features/forward_design.py`) builds the same columns through the same steps —
    duplicating the order of the steps is how a design drifts silently. Every key here is
    S-1 information, which is what makes the extraction legal: nothing in it needs the
    target season to have been played.

    `drafts` defaults to `draft_numbers(features_dir)`, which reads the season matrix and
    therefore has no row for a season nobody has played. The forward path passes the same
    frame extended with its own preseason-legal rows; every retrospective caller passes
    nothing and gets today's behaviour exactly.
    """
    lagged = share_lags(shares, seasons)
    if drafts is None:
        drafts = draft_numbers(features_dir)
    lagged = lagged.merge(drafts, on=["player_id", "season"], how="left")
    lagged["draft_bucket"] = lagged["draft_bucket"].fillna(UNDRAFTED_BUCKET)

    priors = rookie_share_priors(lagged, seasons)
    lagged = lagged.merge(priors[["season", "draft_bucket", "rookie_share_prior"]],
                          on=["season", "draft_bucket"], how="left")
    lagged["w_share"] = (lagged["prior_share"]
                         .fillna(lagged["rookie_share_prior"])
                         .fillna(FALLBACK_ROOKIE_SHARE)
                         .clip(*SHARE_CLIP))
    carried = ["player_id", "season", "w_share", "no_prior", "share_stale", "draft_number"]
    if share_hook is not None:
        lagged = share_hook(lagged)
        if lagged["w_share"].isna().any():
            raise ValueError("`share_hook` returned a NaN `w_share`")
        lagged["w_share"] = lagged["w_share"].clip(*SHARE_CLIP)
        if "order_share" in lagged:
            lagged["order_share"] = lagged["order_share"].clip(*SHARE_CLIP)
            carried.append("order_share")
    return lagged, carried


def composition_frame(cfg: dict, share_hook=None) -> pd.DataFrame:
    """One ordered row per played player-game, over every season.

    Built over the full history even when only a pilot window is fitted — the lags and
    the expanding rookie prior need the earlier seasons. Ordering is fixed within a
    team-season (prior share descending, no-prior players last by draft slot), so each
    game's played subset inherits it, and it is computable preseason.

    `share_hook`, if given, is called with the per-(player, season) frame **after**
    `w_share` is built and must return it with `w_share` replaced. It exists for
    `composition_preseason`, which asks whether a preseason reading improves that column —
    the one input that reaches BOTH the offset (`logit_prior`) and the allocation ORDER, so
    it cannot be tested as a feature. `None` is today's behaviour exactly and every other
    caller passes nothing.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    seasons = cfg["data"]["seasons"]

    panel = pd.read_parquet(features_dir / "availability_panel.parquet")
    lengths = pd.read_parquet(features_dir / "game_length.parquet")
    played = played_frame(panel, lengths)

    shares = season_shares(played)
    lagged, carried = season_weights(shares, seasons, features_dir, share_hook)

    frame = played.merge(lagged[carried], on=["player_id", "season"], how="left")
    if frame["w_share"].isna().any():
        raise ValueError("played rows with no composition weight — share_lags no "
                         "longer covers the panel")
    frame[OWN] = np.log(frame["w_share"] / (1 - frame["w_share"]))

    design = availability_design(cfg)
    frame = frame.merge(design[["player_id", "season"] + FEATURE_COLS],
                        on=["player_id", "season"], how="left")

    frame = order_frame(frame)

    frame["N"] = np.rint(5 * frame["game_length"].to_numpy(float)).astype(int)
    frame["U"] = np.rint(frame["game_length"].to_numpy(float)).astype(int)

    k = frame.groupby(GROUP_KEYS, sort=False)["player_id"].transform("size")
    infeasible = frame["N"] > k * frame["U"]
    if infeasible.any():
        raise ValueError(f"{int(infeasible.sum())} rows in team-games with N > K*U — "
                         f"the allocation is infeasible; check the played filter")

    frame["y"] = largest_remainder(frame)
    frame = sequential_columns(frame)

    sums = frame.groupby(GROUP_KEYS, sort=False)["y"].transform("sum")
    if (sums != frame["N"]).any():
        raise ValueError("rounded minutes do not sum to N — largest_remainder is broken")

    resid = (frame.groupby(GROUP_KEYS, sort=False)
             .agg(total=("min", "sum"), n=("N", "first")))
    absorbed = (resid["total"] - resid["n"]).abs()
    n_teams = frame.groupby(GROUP_KEYS, sort=False).ngroups
    print(f"  {len(frame):,} played player-rows over {n_teams:,} team-games; "
          f"rounding absorbed a max team-sum residual of {absorbed.max():.2f} min "
          f"(mean {absorbed.mean():.4f})")
    print(f"  no-prior rows: {frame['no_prior'].mean():.1%} of rows, "
          f"{frame.loc[frame['no_prior'] == 1, 'y'].sum() / frame['y'].sum():.1%} "
          f"of minutes")
    return frame


# ── The head's own path — the preseason-blended `w_share` ─────────────────────

# Adopted 2026-08-14 (`docs/preseason-plan.md` P5), on session 4d's `ship_margin`: the
# blended arm beats the head that was shipping by -0.25409 [-0.26520, -0.24315] CRPS
# minutes per player-game on the draft pool and -17.27296 [-22.32569, -11.92164] per
# player-season, with `team_sum_abs_error` exactly 0 on all six arms.
#
# ⚠️ This head takes the preseason through `w_share` and NOT through a column on `beta`,
# which is the opposite of the availability and minutes blocks. `w_share` reaches the model
# three ways — as the `OWN` feature, as the offset through `sequential_columns`, and as the
# allocation order through `order_frame` — and no coefficient reaches the last two. Session
# 4b's attribution put 103% of the screen's margin on the OFFSET and 3.75% on the ordering,
# so `route = offset_only` is what ships and the ordering stays on the incumbent share.
PRESEASON = True


def preseason_blend(cfg: dict, preseason: bool | None = None) -> tuple[float, str] | None:
    """`(k, route)` for the shipped blend, or `None` when the head carries no preseason.

    Read from `stan.composition.preseason` — the same block `make composition-preseason-fit`
    reads, deliberately, because `blend_k` and `route` are session 4b's decisions 2 and 3
    and the head and the round that measured it must not be able to disagree about them.
    The other keys in that block (`first_season`, `label`, `arms`) belong to the measurement
    target alone; `adopt` is the one this function turns on.
    """
    from src.models.composition_preseason import INCUMBENT_K
    from src.models.composition_preseason_fit import ROUTE, SELECTED_K

    pre_cfg = cfg.get("stan", {}).get("composition", {}).get("preseason", {})
    on = bool(pre_cfg.get("adopt", PRESEASON)) if preseason is None else bool(preseason)
    if not on:
        return None
    k = float(pre_cfg.get("blend_k", SELECTED_K))
    if k >= INCUMBENT_K:
        return None
    return k, str(pre_cfg.get("route", ROUTE))


def head_first_season(cfg: dict, preseason: bool | None = None) -> str:
    """This head's fitting window, cut to preseason coverage when the blend is on.

    `stan_minutes.first_covered_season`'s rule one head over: the panel's first intact-tail
    season is read off `preseason_coverage.csv` rather than typed, so the window is a
    property of the capture rather than a constant somebody has to remember to move — and it
    matters by exactly one season, since 2003-04 has 369 real preseason rows and is still
    excluded as `tail_missing`.

    Session 4d priced the cut on its own, with a third arm that fits below coverage because
    it carries no preseason column: `window_cost` is **-0.01991 [-0.02515, -0.01498]** CRPS
    minutes per player-game — the same direction as P3's cut and 8.5% of the increment
    rather than the quarter that round paid. So the cut helps here, and the block is not
    mostly window.
    """
    configured = str(cfg.get("stan", {}).get("composition", {})
                     .get("first_season", PILOT_FIRST_SEASON))
    if preseason_blend(cfg, preseason) is None:
        return configured
    from src.models.stan_minutes import first_covered_season

    return max(configured, first_covered_season(cfg))


def head_frame(cfg: dict, preseason: bool | None = None) -> pd.DataFrame:
    """`composition_frame` plus the blended `w_share` — **this head's path and no other's**.

    Separate from `composition_frame` for the reason `stan_minutes.head_design` is separate
    from `build_design`: that builder is how `composition_effects`, `minutes_window`,
    `rookie_priors` and both `composition_preseason` modules reach their rows, and a share
    that is structurally unavailable before 2004-05 must not enter any of them by accident.
    The measurement modules' `base` arms are *controls*, so a blend arriving under them
    silently would delete the very contrast they exist to draw.

    **`minutes_unification`, `model_cards` and `sim/season` are the exceptions, and they are
    the rule rather than violations of it**: each consumes this head's *persisted posterior*
    and needs the offset and the ordering that posterior was fitted on. Anything that scores
    or draws from the shipped head comes through here; anything that builds its own model on
    these rows goes through `composition_frame`.

    The hook is `composition_preseason.blend_hook` — the same function sessions 4b through 4d
    measured — so what ships is the arm that was scored rather than a second implementation
    of it.
    """
    return composition_frame(cfg, share_hook=shipped_share_hook(cfg, preseason))


def shipped_share_hook(cfg: dict, preseason: bool | None = None):
    """The `share_hook` the shipped head is fitted with, or `None` when the blend is off.

    Extracted from `head_frame` (2026-08-21) for the same reason as `season_weights`: the
    forward per-player frame (`features/forward_design.py`) must blend `w_share` through
    the code that was measured, not a second implementation of it. A forward season with
    no preseason rows yet — a 2026-27 frame built in August — comes back on the incumbent
    share for every player, because `blend_hook` joins `how="left"` and an unseen player
    carries weight 0; the same call in October picks the preseason up through the join
    with nothing changing here.
    """
    blend = preseason_blend(cfg, preseason)
    if blend is None:
        return None
    k, route = blend

    from src.models.composition_preseason import blend_hook, preseason_share

    panel_path = Path(cfg["data"]["features_dir"]) / "preseason.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} is missing and the composition head ships a preseason-blended "
            f"`w_share` — run `make preseason`, or set "
            f"`stan.composition.preseason.adopt: false` to fit the pre-2026-08-14 head "
            f"exactly.")
    pre = preseason_share(pd.read_parquet(panel_path))
    return blend_hook(pre, k, route)


# ── Feature variants ──────────────────────────────────────────────────────────

def rho_bin_edges(train: pd.DataFrame, n_bins: int = RHO_BINS) -> np.ndarray:
    """Interior quantile edges of the prior-share column, from **train** only.

    The same rule as the scaler and the spline knots: reading the held-out
    distribution to place the bins would leak it into the design, invisibly.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(train[RHO_BIN_COL].to_numpy(dtype=float), qs)


def assign_rho_bins(frames: list[pd.DataFrame], edges: np.ndarray
                    ) -> list[pd.DataFrame]:
    """1-based bin index per row. Values outside the training range land in the end
    bins rather than raising — a rookie share below every training quantile is a
    fringe player, which is what bin 1 means."""
    out = []
    for frame in frames:
        copy = frame.copy()
        values = copy[RHO_BIN_COL].to_numpy(dtype=float)
        copy["rho_bin"] = np.searchsorted(edges, values, side="right") + 1
        out.append(copy)
    return out


def variants(train: pd.DataFrame, test: pd.DataFrame, n_bins: int = RHO_BINS
             ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str], int, int]]:
    """The ladder from `docs/minutes-composition-plan.md`.

    `binomial` is the pure stick-breaking decomposition (the demo's model, with the
    cap fixed) and is expected to fail calibration — the measured game-level
    dispersion is 4.65x binomial. `betabinom` is the expected winner. `betabinom_ot`
    adds the per-game covariates: n_overtimes and its interaction with the own share,
    because OT minutes should tilt toward the players already playing most.

    `betabinom_ot_graded` differs from `betabinom_ot` in the dispersion ALONE — same
    features, same mean function, `n_rho` bins instead of one shared rho. That makes
    the pair a clean read on the pilot's one measured miscalibration (variance ratio
    1.59 fringe against 0.70 star) rather than a confounded comparison.
    """
    base = [c for c in FEATURE_COLS if c != "minutes_per_game_lag1"] + [OWN]
    tr, te = train.copy(), test.copy()
    for frame in (tr, te):
        frame["design_missing"] = frame["gp_share_lag1"].isna().astype(float)
    # `impute` would mint one flag per column, but a rookie loses every design
    # column at once, so the 18 flags are exact copies of each other — measured as
    # C(18,2) = 153 duplicate standardized column pairs, a degenerate subspace the
    # sampler pays for in treedepth. One `design_missing` indicator carries all of it.
    tr, te, _ = impute(tr, te, base)
    feats = base + ["design_missing", "no_prior", "share_stale", "offset_clipped"]

    # Bin edges from train only; every arm carries the column so the shared-rho arms
    # are the n_rho = 1 special case of the same code path rather than a separate one.
    edges = rho_bin_edges(tr, n_bins)
    tr, te = assign_rho_bins([tr, te], edges)

    out = {"binomial": (tr, te, list(feats), 0, 1),
           "betabinom": (tr, te, list(feats), 1, 1)}

    to, eo = tr.copy(), te.copy()
    for frame in (to, eo):
        frame["ot_x_own"] = frame["n_overtimes"].to_numpy(float) * frame[OWN].to_numpy(float)
    ot_feats = list(feats) + ["n_overtimes", "ot_x_own"]
    out["betabinom_ot"] = (to, eo, ot_feats, 1, 1)
    out["betabinom_ot_graded"] = (to, eo, list(ot_feats), 1, n_bins)
    return out


# ── Item 3d's ladder: the player-season effect and the team-context block ──────

def team_context(features_dir: Path, cols: list[str] = TEAM_COLS) -> pd.DataFrame:
    """The point-in-time-safe team block, one row per (player, season).

    Built leave-one-out by `make team-context`, so a player's own prior season never
    enters his own aggregate; every column is season S-1 statistics over the season-S
    roster, which is this project's information set exactly.
    """
    path = Path(features_dir) / "team_context_tierA.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no team context at {path}; run `make team-context`")
    return pd.read_parquet(path, columns=UNIT_KEYS + list(cols))


def attach_team_context(train: pd.DataFrame, val: pd.DataFrame, block: pd.DataFrame,
                        cols: list[str] = TEAM_COLS
                        ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Join the block, flag its holes with ONE indicator, impute from **train** means.

    **One `team_missing` flag for the whole block, not one per column** — the same
    decision `design_missing` embodies one block over: a player-season either has a team
    row or it has none of it, so five per-column flags would be five exact copies of each
    other and a degenerate subspace the sampler pays for in treedepth.

    It is nonetheless a **second** indicator rather than a reuse of `design_missing`, and
    that is a deliberate departure from the plan's instruction, measured rather than
    assumed: the two mark different rows. `design_missing` flags a composition row with no
    availability-design row at all; the team block additionally misses ~11% of rows that
    *do* have a design, because `team_context_tierA` is built off the season matrix's
    qualified frame. Folding them together would leave those rows silently imputed to the
    training mean with nothing for `beta` to correct on. The overlap is reported by
    `run` so the choice stays checkable.
    """
    tr = train.merge(block, on=UNIT_KEYS, how="left")
    te = val.merge(block, on=UNIT_KEYS, how="left")
    if len(tr) != len(train) or len(te) != len(val):
        raise ValueError("the team-context join duplicated rows — it must be unique on "
                         f"{UNIT_KEYS}")
    for frame in (tr, te):
        frame[TEAM_MISSING] = frame[cols[0]].isna().astype(float)
    coverage = {
        "team_rows_train": int((tr[TEAM_MISSING] == 0).sum()),
        "team_share_train": float(1 - tr[TEAM_MISSING].mean()),
        "team_share_val": float(1 - te[TEAM_MISSING].mean()),
        # The two indicators are not the same column, which is why both ship.
        "team_missing_and_design_missing": float(
            ((tr[TEAM_MISSING] == 1) & (tr["design_missing"] == 1)).mean()),
        "team_missing_with_design_present": float(
            ((tr[TEAM_MISSING] == 1) & (tr["design_missing"] == 0)).mean()),
    }
    means = {c: float(tr.loc[tr[TEAM_MISSING] == 0, c].mean()) for c in cols}
    for frame in (tr, te):
        for c in cols:
            frame[c] = frame[c].fillna(means[c])
    return tr, te, list(cols) + [TEAM_MISSING], {**coverage, "team_means": means}


class EffectArm(NamedTuple):
    """One arm of `effect_variants`, named so the ladder stops being read by index.

    A `NamedTuple` rather than a dataclass deliberately: it still unpacks positionally and
    still slices, so `built[arm][:6]` and the existing `built[a][5]` keep working while new
    call sites can say `built[a].quadrature`. The tuple grew from seven fields to nine when
    the marginal representation landed, and nine positional fields is where an index becomes
    a bug waiting to happen.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    features: list[str]
    dispersed: int
    n_rho: int
    player_season_effect: bool
    centered: bool
    quadrature: bool
    n_sigma: int


def effect_variants(train: pd.DataFrame, val: pd.DataFrame, block: pd.DataFrame,
                    base_variant: str = "betabinom_ot_graded", n_bins: int = RHO_BINS
                    ) -> tuple[dict, dict]:
    """The item-3d ladder, built on top of the shipped arm's frames and features.

    Four arms. `base` is the shipped specification refitted on whatever window is being
    run — a **same-window control**, not a refit of the incumbent, and it exists for the
    same reason `stan_game_length`'s `season_trend_covered` does: a pilot-window arm
    ordering is uninterpretable against a full-window baseline. The three that follow are
    the ones the plan names.

    | arm | mean function | effect | parameterization |
    |---|---|---|---|
    | `base` | shipped | — | — |
    | `ps` | shipped | `sigma_u` | non-centred |
    | `team` | shipped + team context | — | — |
    | `ps_team` | shipped + team context | `sigma_u` | non-centred |
    | `ps_centered` | shipped | `sigma_u` | **centred** |
    | `mq` | shipped | `sigma_u`, **marginalized** | quadrature, shared sigma |
    | `mq_graded` | shipped | `sigma_u` per `rho_bin`, marginalized | quadrature |

    `ps_centered` is the same model as `ps` in different coordinates, so it is a *sampler*
    arm rather than a modelling one — it is in the ladder because the two are not
    interchangeable in cost and the plan names the centred form as the first response when
    the non-centred one misbehaves. It must never be selected on CRPS against `ps`: they have
    the same posterior, and any difference between them is Monte Carlo error or a
    convergence failure.

    **`mq` stands in the same relation to `ps`** — same posterior, third representation —
    and the same rule applies with one addition: `mq` and `ps` disagreeing is a *bug*, not a
    result, and `docs/composition-quadrature-plan.md` step 4 exists to check it before any
    metric is read. What `mq` is actually for is the two things `ps` cannot do at all:
    converge, and reach the full window. `mq_graded` is the only arm here that is a
    different model from anything above it.

    Returns
    `{arm: (train, val, features, dispersed, n_rho, player_season_effect, centered,
      quadrature, n_sigma)}` and the join's coverage report.
    """
    tr, te, feats, dispersed, n_rho = variants(train, val, n_bins)[base_variant]
    tr_t, te_t, team_feats, coverage = attach_team_context(tr, te, block)
    with_team = list(feats) + team_feats
    return ({
        "base": EffectArm(tr, te, list(feats), dispersed, n_rho, False, False, False, 1),
        "ps": EffectArm(tr, te, list(feats), dispersed, n_rho, True, False, False, 1),
        "team": EffectArm(tr_t, te_t, with_team, dispersed, n_rho, False, False, False, 1),
        "ps_team": EffectArm(tr_t, te_t, list(with_team), dispersed, n_rho, True, False,
                             False, 1),
        "ps_centered": EffectArm(tr, te, list(feats), dispersed, n_rho, True, True,
                                 False, 1),
        "mq": EffectArm(tr, te, list(feats), dispersed, n_rho, False, False, True, 1),
        "mq_graded": EffectArm(tr, te, list(feats), dispersed, n_rho, False, False, True,
                               n_bins),
    }, coverage)


# ── Ragged arrays and the simulator ───────────────────────────────────────────

def ragged_arrays(frame: pd.DataFrame) -> dict:
    """Team-game block structure from the ordered frame, 0-based."""
    position = frame["position"].to_numpy(dtype=np.int64)
    starts = np.flatnonzero(position == 0)
    lens = np.append(starts[1:], len(frame)) - starts
    expected = frame["k_players"].to_numpy(dtype=np.int64)[starts]
    if not np.array_equal(lens, expected):
        raise ValueError("frame rows are not contiguous team-game blocks — do not "
                         "re-sort the composition frame")
    return {
        "starts": starts,
        "lens": lens,
        "n_total": frame["N"].to_numpy(dtype=np.int64)[starts],
        "caps": frame["U"].to_numpy(dtype=np.int64)[starts],
        "ratio": frame["stick_ratio"].to_numpy(dtype=float),
    }


def simulate_minutes(frame: pd.DataFrame, eta_base: np.ndarray,
                     rho: np.ndarray | None, seed: int = 0) -> np.ndarray:
    """(draws x rows) joint minutes draws — sequential, capped, summing to N exactly.

    `eta_base` is (rows x draws): the linear predictor *without* the stick-breaking
    offset, which is recomputed here per draw because it depends on the running
    remainder. `rho = None` is the binomial arm; otherwise it is (draws x n_rho) and
    each row takes the dispersion of its `rho_bin` — a frame without that column is
    treated as one shared bin, so the floor and the shared-rho arms need no special
    case. Beta-binomial steps are drawn as `p ~ Beta(a, b)` then `Binomial(m, p)` —
    exact, and orders of magnitude faster than a pmf grid at this shape. A step below
    the feasibility bound is redrawn and, past MAX_REDRAWS, clamped and counted loudly.
    """
    arrays = ragged_arrays(frame)
    starts, lens = arrays["starts"], arrays["lens"]
    n_total, caps, ratio = arrays["n_total"], arrays["caps"], arrays["ratio"]
    n_rows, n_draws = eta_base.shape
    max_len = int(lens.max())

    by_position = []
    for k in range(max_len):
        teams = np.flatnonzero(lens > k)
        by_position.append((teams, starts[teams] + k))

    if rho is not None:
        rho = np.atleast_2d(np.asarray(rho, dtype=float))
        if rho.shape[0] != n_draws:      # a single shared value broadcast per draw
            rho = rho.reshape(n_draws, -1)
    bins = (frame["rho_bin"].to_numpy(dtype=int) - 1 if "rho_bin" in frame.columns
            else np.zeros(n_rows, dtype=int))
    # A shared-rho model is handed the SAME frames as the graded one — every variant
    # carries `rho_bin` — so the width of `rho`, not the frame, decides whether the
    # bins are used. Without this the shared arms index past a length-1 rho.
    if rho is not None and rho.shape[1] == 1:
        bins = np.zeros(n_rows, dtype=int)

    rng = np.random.default_rng(seed)
    out = np.zeros((n_draws, n_rows), dtype=np.int64)
    clamped = 0
    for d in range(n_draws):
        remaining = n_total.copy()
        for k, (teams, rows) in enumerate(by_position):
            r_k = remaining[teams]
            u = caps[teams]
            last = lens[teams] == k + 1
            m = np.minimum(u, r_k)
            after = lens[teams] - k - 1
            lower = np.maximum(0, r_k - after * u)

            p_carry = np.clip(ratio[rows] * r_k / np.maximum(m, 1), *OFFSET_CLIP)
            eta = np.log(p_carry / (1 - p_carry)) + eta_base[rows, d]
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

            rho_k = None if rho is None else rho[d][bins[rows]]
            if rho_k is None:
                y_k = rng.binomial(m, mu)
            else:
                a, b = beta_shapes(mu, rho_k)
                y_k = rng.binomial(m, rng.beta(a, b))

            need = ~last & (y_k < lower)
            tries = 0
            while need.any() and tries < MAX_REDRAWS:
                if rho_k is None:
                    y_k[need] = rng.binomial(m[need], mu[need])
                else:
                    a, b = beta_shapes(mu[need], rho_k[need])
                    y_k[need] = rng.binomial(m[need], rng.beta(a, b))
                need = ~last & (y_k < lower)
                tries += 1
            if need.any():
                clamped += int(need.sum())
                y_k[need] = lower[need]

            y_k[last] = r_k[last]
            out[d, rows] = y_k
            remaining[teams] = r_k - y_k

    sums = np.add.reduceat(out, starts, axis=1)
    if not (sums == n_total[None, :]).all():
        raise AssertionError("a simulated team-game does not sum to N — the "
                             "sequential draw is broken")
    if (out > frame["U"].to_numpy(dtype=np.int64)[None, :]).any():
        raise AssertionError("a simulated player exceeds the game-length cap")
    if clamped:
        print(f"   /!\\  simulator clamped {clamped:,} step draws to the feasibility "
              f"bound (short-rotation games)")
    return out.astype(float)


# ── The head and its floor ────────────────────────────────────────────────────

class StanComposition:
    """The composition head. `dispersed = 0` is the pure stick-break binomial arm."""

    def __init__(self, features: list[str], dispersed: int, n_rho: int = 1,
                 l2: float = GLM_L2, name: str = "stan", chains: int = 4,
                 warmup: int = 1000, samples: int = 1000, seed: int = 42,
                 predictive_samples: int = PREDICTIVE_SAMPLES,
                 player_season_effect: bool = False,
                 u_sd_scale: float = U_SD_SCALE, u_centered: bool = False,
                 metric: str | None = None, quadrature: bool = False,
                 q_nodes: int = Q_NODES, gh_inflate: float = GH_INFLATE,
                 n_sigma: int = 1):
        self.features, self.dispersed, self.l2, self.name = features, dispersed, l2, name
        self.n_rho = int(n_rho)
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed
        self.predictive_samples = predictive_samples
        if player_season_effect and quadrature:
            raise ValueError("the sampled latent and the marginal quadrature are two "
                             "representations of the same effect — enable one")
        # Disabled by default, so every existing caller builds the head that shipped.
        # `ps` carries the effect at PREDICT time under both representations: the fitted
        # `sigma_u` lands in the same place either way, and a fresh `z` per (unit, draw) is
        # what the predictive integrates over regardless of how sigma was estimated.
        self.ps = PlayerSeasonTerm(player_season_effect or quadrature, u_sd_scale, seed,
                                   stream=name, centered=u_centered,
                                   sampled=not quadrature)
        self.quad = QuadratureTerm(quadrature, q_nodes, gh_inflate, n_sigma, u_sd_scale)
        # None = let `choose_metric` size it; a string forces it, which is what a probe
        # comparing the two metrics on identical data needs.
        self.metric = metric

    def stan_data(self, train: pd.DataFrame) -> dict:
        """The data dict `fit` passes to Stan — separate so diagnosis scripts and
        tests can build it without sampling."""
        (X,), self.scaler = standardized(train, [train], self.features)
        arrays = ragged_arrays(train)
        return {
            **self.ps.data(train), **self.quad.data(train),
            "G": len(arrays["starts"]), "P": len(train), "K": X.shape[1],
            "start": (arrays["starts"] + 1).tolist(),
            "len": arrays["lens"].tolist(),
            "y": train["y"].to_numpy(dtype=int).tolist(),
            "m": train["m"].to_numpy(dtype=int).tolist(),
            "lo": train["lo"].to_numpy(dtype=int).tolist(),
            "is_last": train["is_last"].to_numpy(dtype=int).tolist(),
            "N_total": arrays["n_total"].tolist(),
            "U": arrays["caps"].tolist(),
            "logit_prior": train["logit_prior"].to_numpy(dtype=float).tolist(),
            "X": X, "dispersed": int(self.dispersed),
            "n_rho": self.n_rho,
            "rho_bin": (train["rho_bin"].to_numpy(dtype=int).tolist()
                        if self.n_rho > 1 else [1] * len(train)),
            "beta_scale": prior_sd_for_l2(self.l2),
            "intercept_scale": INTERCEPT_SCALE,
        }

    def fit(self, train: pd.DataFrame) -> "StanComposition":
        data = self.stan_data(train)
        # alpha = 0 with zero slopes starts at the no-fit floor, which is a sane
        # allocation everywhere — the same reasoning as the intercept-only inits on
        # every other head.
        inits = {"alpha": 0.0, "beta": np.zeros(data["K"]).tolist()}
        if self.dispersed:
            inits["rho"] = [RHO_INIT] * self.n_rho
        if data["U_n"]:
            # Zero effects at a plausible scale — the same intercept-only reasoning every
            # other head inits by. 0.3 rather than 0 because the centred arm's prior is
            # `normal(0, sigma_u)` and a zero scale is not a starting point.
            inits["u_z"] = np.zeros(data["U_n"]).tolist()
            inits["sigma_u"] = [0.3]
        elif data["Q"]:
            # No latents to initialize — the whole point. Every bin starts at the same 0.3,
            # so the graded arm's init is the shared arm's and any spread it reports is
            # fitted rather than seeded.
            inits["sigma_u"] = [0.3] * int(data["n_sigma"])

        model = compile_model(MODEL)
        # dense_e, not the default diagonal metric: measured on the one-season probe,
        # the posterior's linear correlations hold NUTS at treedepth 8-9 under diag_e
        # (645s for 300+200 x 2 chains) and treedepth 4 under dense_e (65s). At
        # ~25 parameters the dense adaptation is free.
        #
        # **The random effect does not automatically put the dense metric out of reach**,
        # and treating it as if it did was a measured mistake: the cost is quadratic in
        # PARAMETERS, so a 605-unit probe wants a 3 MB matrix and only the 12,307-unit full
        # window wants 1.2 GB. `choose_metric` sizes it, and `DENSE_METRIC_MAX_MB` carries
        # the arithmetic. This matters because treedepth saturation — 791 of 800 draws on
        # the `ps_no_rho` diagnostic under `diag_e` — is precisely what the dense metric
        # fixed on this head before the effect existed.
        # The marginal representation adds `n_sigma` parameters and NOT `U_n` of them,
        # which is the whole reason `dense_e` is reachable for it at any window.
        metric = self.metric or choose_metric(
            int(data["K"]) + 1 + data["n_rho"] + int(data["U_n"])
            + (int(data["n_sigma"]) if data["Q"] else 0), self.warmup)
        fit, self.diagnostics = sample(
            model, data, chains=self.chains, warmup=self.warmup,
            samples=self.samples, seed=self.seed, label=self.name, inits=inits,
            metric=metric)
        self.diagnostics["metric"] = metric
        self.diagnostics["parameterization"] = (
            f"marginal_q{data['Q']}" if data["Q"] else
            "none" if not data["U_n"] else
            ("centered" if data["u_centered"] else "non_centered"))
        warn_if_unconverged(self.diagnostics)
        self.ps.absorb(fit)

        draws = posterior(fit, ["alpha", "beta"])
        self.alpha_draws = draws["alpha"].reshape(-1)
        self.beta_draws = draws["beta"].reshape(len(self.alpha_draws), -1)
        if self.dispersed:
            # (draws x n_rho) even when n_rho = 1, so the graded and shared arms take
            # the same downstream path.
            self.rho_draws = posterior(fit, ["rho"])["rho"].reshape(
                len(self.alpha_draws), -1)
            self.rho_by_bin = self.rho_draws.mean(axis=0)
            self.rho = float(self.rho_by_bin.mean())
        else:
            self.rho_draws, self.rho_by_bin, self.rho = None, None, 0.0
        return self

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    def _draw_index(self) -> np.ndarray:
        return thin(len(self.alpha_draws), self.predictive_samples)

    def _eta_base(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """The DETERMINISTIC linear predictor, without the player-season effect.

        Deliberately excludes the random effect even when one is fitted, because two
        consumers need exactly this: `posteriors._finish` stores it as the artifact's
        round-trip reference (a fresh `z` per call would make the gate non-reproducible),
        and `minutes_unification.player_season_effect_sweep` injects its own sigma on top.
        `predict_samples` is where the fitted effect enters.
        """
        idx = self._draw_index()
        eta = self._design(df) @ self.beta_draws[idx].T + self.alpha_draws[idx][None, :]
        rho = self.rho_draws[idx] if self.dispersed else None
        return eta, rho

    def predict_samples(self, df: pd.DataFrame, seed: int = 0) -> np.ndarray:
        eta, rho = self._eta_base(df)
        return simulate_minutes(df, eta + self.ps.shift(df, self._draw_index()),
                                rho, seed)

    def plug_in(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Posterior means: intercept, coefficients, and rho **per bin**."""
        rho = (self.rho_by_bin if self.dispersed
               else np.full(1, max(self.rho, RHO_MIN)))
        return float(self.alpha_draws.mean()), self.beta_draws.mean(axis=0), rho

    def rho_row(self, df: pd.DataFrame) -> np.ndarray:
        """Posterior-mean dispersion for each row of `df`, by its bin."""
        _, _, rho = self.plug_in()
        if len(rho) == 1 or "rho_bin" not in df.columns:
            return np.full(len(df), float(rho[0]))
        return rho[df["rho_bin"].to_numpy(dtype=int) - 1]


class FloorComposition:
    """The no-fit floor: prior shares renormalized over who played, no Stan.

    The mean is the offset alone; only the dispersion is fitted, on the floor's own
    training steps — the `FloorMinutes` pattern, so its CRPS is comparable and the
    contrast is about the mean function.
    """

    name = "carry_forward"

    def __init__(self, predictive_samples: int = PREDICTIVE_SAMPLES):
        self.predictive_samples = predictive_samples

    def fit(self, train: pd.DataFrame) -> "FloorComposition":
        live = train[train["is_last"] == 0]
        p = 1.0 / (1.0 + np.exp(-live["logit_prior"].to_numpy(float)))
        self.rho = fit_dispersion(live["y"].to_numpy(int), live["m"].to_numpy(int), p)
        return self

    def predict_samples(self, df: pd.DataFrame, seed: int = 0) -> np.ndarray:
        eta = np.zeros((len(df), self.predictive_samples))
        # One shared rho, deliberately: the floor answers "is the MEAN function worth
        # anything", so grading its dispersion too would blur the contrast the graded
        # arm is meant to isolate.
        rho = np.full((self.predictive_samples, 1), self.rho)
        return simulate_minutes(df, eta, rho, seed)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_samples(samples: np.ndarray, frame: pd.DataFrame, label: str,
                  seed: int = 0) -> dict:
    """Marginal metrics per player-game plus the one joint metric CRPS cannot see:
    the mean absolute team-sum error, exactly 0 for the composition by construction
    and the capability gap against any independent-draws model."""
    y = frame["y"].to_numpy(dtype=float)
    pred = samples.mean(axis=0)
    arrays = ragged_arrays(frame)
    sums = np.add.reduceat(samples, arrays["starts"], axis=1)
    team_abs = float(np.abs(sums - arrays["n_total"][None, :]).mean())
    return {
        "variant": label,
        "n": len(frame),
        "crps_minutes": float(crps_from_samples(samples, y).mean()),
        "r2_minutes": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "mae_minutes": float(np.abs(y - pred).mean()),
        "pit_ks": ks_uniform(pit_from_samples(samples, y, seed)),
        "bias_minutes": float((pred - y).mean()),
        "team_sum_abs_error": team_abs,
    }


def mark_selection(table: pd.DataFrame) -> pd.DataFrame:
    """Validation-selected fitted variant + the floor flag; baselines not selectable."""
    out = table.copy()
    fitted = out[out["variant"].isin(FITTED_VARIANTS)]
    if fitted.empty:
        raise ValueError("no fitted variants in the sweep table")
    out["selected"] = False
    out.loc[fitted["val_crps"].idxmin(), "selected"] = True
    floor = out[out["variant"] == "carry_forward"].iloc[0]
    out["beats_floor"] = ((out["val_crps"] < floor["val_crps"])
                          & (out["val_r2"] > floor["val_r2"]))
    out.loc[out["variant"] == "carry_forward", "beats_floor"] = True
    return out


# ── The independent comparator (the incumbent) ────────────────────────────────

def independent_comparator(cfg: dict, pilot: pd.DataFrame, val: pd.DataFrame,
                           cfg_stan: dict, seed: int) -> dict:
    """The season head + rho_game, drawn independently per player-game.

    This is the plan of record being compared against: `stan_minutes`'s selected
    variant (`logit_own_spline`) refit with nothing later than the evaluation split's
    first season, pushed to per-game draws with the measured game-level dispersion.
    Its trials are the same real game lengths, so the cap holds — what it cannot do
    is sum to N, and `team_sum_abs_error` is where that shows.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    design = minutes_build_design(cfg)
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    lengths = pd.read_parquet(features_dir / "game_length.parquet")

    out = {"samples": {}, "mu": {}}
    for split, frame in (("val", val),):
        eval_seasons = set(frame["season"].unique())
        later = {s for s in design["season"].unique()
                 if s >= min(eval_seasons)}
        head_train = design[~design["season"].isin(later)]
        head_eval = design[design["season"].isin(eval_seasons)]

        v = minutes_variants(head_train, head_eval, n_knots=SPLINE_KNOTS)
        tr, te, feats = v["logit_own_spline"]
        model = StanMinutes(feats, name=f"comparator/{split}",
                            chains=int(cfg_stan.get("chains", 4)), seed=seed,
                            warmup=int(cfg_stan.get("select_warmup", 500)),
                            samples=int(cfg_stan.get("select_samples", 500))).fit(tr)
        out[f"diagnostics_{split}"] = model.diagnostics
        mus, _ = model.mu_draws(te, model.predictive_samples)
        lookup = te[["player_id", "season"]].copy()
        lookup["mu_head"] = mus.mean(axis=0)

        rows = frame.merge(lookup, on=["player_id", "season"], how="left")
        covered = float(rows["mu_head"].notna().mean())
        mu = rows["mu_head"].fillna(rows["w_share"]).to_numpy(float)

        train_targets = targets[targets["season"].isin(
            set(head_train["season"].unique()))]
        rho_game = game_level_dispersion(train_targets, lengths)["rho"]

        rng = np.random.default_rng(seed)
        n = np.rint(frame["game_length"].to_numpy(float)).astype(int)
        keep = int(cfg_stan.get("composition", {})
                   .get("predictive_samples", PREDICTIVE_SAMPLES))
        a, b = beta_shapes(np.repeat(mu[None, :], keep, axis=0),
                           np.full((keep, 1), rho_game))
        out["samples"][split] = rng.binomial(n[None, :], rng.beta(a, b)).astype(float)
        out["mu"][split] = mu
        out[f"coverage_{split}"] = covered
        out[f"rho_game_{split}"] = float(rho_game)
    return out


# ── Sweep ─────────────────────────────────────────────────────────────────────

def _iters(cfg_stan: dict, fast: bool = False) -> dict:
    """`fast` survives for the Gate A probe alone.

    The sweep used to run its validation side short and its test side long; with the test
    side gone every sweep fit runs at full length, so `fast` is no longer a split budget —
    it is the probe's own knob, and `probe_timing` scales its extrapolation by the ratio.
    """
    if fast:
        return {"warmup": int(cfg_stan.get("select_warmup", 500)),
                "samples": int(cfg_stan.get("select_samples", 500))}
    return {"warmup": int(cfg_stan.get("warmup", 1000)),
            "samples": int(cfg_stan.get("samples", 1000))}


def _checkpoint(ckpt_dir: Path | None, label: str, row: dict,
                diagnostics: list[dict], models: dict) -> None:
    """Flush one completed arm to disk: its metric row, its two diagnostics, its draws.

    `run` writes its five CSVs only at the very end, and at full window the sweep is a
    ~14 h loop — so without this a crash in the last arm loses every fit before it.
    With it the PPC and joint-NLL tail of `run` can be re-driven from the
    pickles in minutes. Follows the `data.boxscore_status.flush_every` precedent:
    append as you go, and never let the flush itself take the run down.
    """
    if ckpt_dir is None:
        return
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in (("metrics", pd.DataFrame([row])),
                            ("diagnostics", pd.DataFrame(diagnostics))):
            dest = ckpt_dir / f"stan_composition_{name}.csv"
            frame.to_csv(dest, mode="a", header=not dest.exists(), index=False)
        # ~25 parameters x a few thousand draws — a few hundred KB per arm. The
        # frames are NOT pickled: they are rebuilt deterministically by `variants`.
        state = {
            side: {attr: getattr(models[label][side], attr, None)
                   for attr in ("alpha_draws", "beta_draws", "rho_draws",
                                "rho_by_bin", "rho", "scaler", "features",
                                "dispersed", "n_rho", "predictive_samples")}
            for side in ("val",)}
        with open(ckpt_dir / f"stan_composition_{label}.pkl", "wb") as fh:
            pickle.dump(state, fh)
        print(f"    checkpointed {label} → {ckpt_dir}")
    except Exception as exc:                                  # pragma: no cover
        print(f"    /!\\  checkpoint for {label} failed ({exc}); the run continues")


def sweep(train: pd.DataFrame, val: pd.DataFrame, cfg_stan: dict, comparator: dict,
          ckpt_dir: Path | None = None
          ) -> tuple[pd.DataFrame, list[dict], dict]:
    """Floor, ladder, incumbent — on the VALIDATION split only.

    **Halves the sampler cost — 21.13 h measured two-pass against 9.92 h measured one-pass,
    a 53% saving.** The test side was the expensive half twice over: it refit on
    train + validation (631k rows against ~590k) *and* ran at double the iterations, on a
    head whose per-row cost is superlinear in rows. Raising selection to full length spends
    only part of that back — the four val fits went 7.35 h → 9.78 h, a 1.33× rise for a 2×
    iteration increase, because longer warmup adapts a better step size and buys fewer
    leapfrog steps per iteration.

    Nothing it decided is lost. `betabinom_ot_graded` was already selected on `val_crps`,
    and the test column was labelled confirmation-only in the doc that took Gate E.
    """
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    keep = int(cfg_stan.get("composition", {})
               .get("predictive_samples", PREDICTIVE_SAMPLES))
    rows, diagnostics, models = [], [], {}

    floor_val = score_samples(
        FloorComposition(keep).fit(train).predict_samples(val, seed),
        val, "carry_forward", seed)
    rows.append(_row("carry_forward", 0, floor_val))

    for label, (v_tr, v_te, v_feats, dispersed, n_rho) in variants(train, val).items():
        model = StanComposition(v_feats, dispersed, n_rho, name=f"{label}/val",
                                chains=chains, seed=seed, predictive_samples=keep,
                                **_iters(cfg_stan)).fit(v_tr)
        diagnostics.append(model.diagnostics)
        v = score_samples(model.predict_samples(v_te, seed), v_te, label, seed)

        models[label] = {"val": model, "val_frame": v_te}
        rows.append(_row(label, len(v_feats), v))
        _checkpoint(ckpt_dir, label, rows[-1], diagnostics[-1:], models)

    rows.append(_row("independent_comparator", -1,
                     score_samples(comparator["samples"]["val"], val,
                                   "independent_comparator", seed)))

    return mark_selection(pd.DataFrame(rows)), diagnostics, models


def _row(label: str, n_features: int, v: dict) -> dict:
    return {"variant": label, "n_features": n_features,
            "val_crps": v["crps_minutes"], "val_r2": v["r2_minutes"],
            "val_mae": v["mae_minutes"], "val_pit_ks": v["pit_ks"],
            "val_bias": v["bias_minutes"],
            "val_team_sum_abs": v["team_sum_abs_error"]}


# ── Gate A: the timing probe ──────────────────────────────────────────────────

def probe_timing(train: pd.DataFrame, cfg_stan: dict, max_hours: float) -> dict:
    """One-season `betabinom` fit at probe iters, extrapolated linearly to the sweep.

    Aborts loudly past the budget rather than discovering it six hours in. Fallbacks,
    in order (docs/minutes-composition-plan.md): cut `binomial` from the ladder,
    shorten the chains, subsample **train** team-games — never val or test.

    The extrapolation now covers one pass, not two: since the sweep dropped its test side
    it fits each variant once, on `train`, at full length. **Gate A still under-predicts** —
    it read 12.8 h against an actual 20.9 h at the full window, a 1.63x miss, because
    per-row cost is superlinear in rows (more data sharpens the posterior, shrinks the step
    size and buys more leapfrog steps). Treat the number as a lower bound.
    """
    last = sorted(train["season"].unique())[-1]
    probe_frame = train[train["season"] == last]
    v = variants(probe_frame, probe_frame)
    tr, _, feats, dispersed, n_rho = v["betabinom"]
    model = StanComposition(feats, dispersed, n_rho, name="probe/one-season",
                            chains=int(cfg_stan.get("chains", 4)),
                            seed=int(cfg_stan.get("seed", 42)),
                            **_iters(cfg_stan, True)).fit(tr)
    seconds = model.diagnostics["wall_clock_s"]
    per_row = seconds / len(probe_frame)

    fast = _iters(cfg_stan, True)
    full = _iters(cfg_stan, False)
    iter_scale = (full["warmup"] + full["samples"]) / (fast["warmup"] + fast["samples"])
    n_variants = len(FITTED_VARIANTS)
    est = n_variants * len(train) * per_row * iter_scale
    hours = est / 3600
    print(f"  Gate A: probe fit {len(probe_frame):,} rows ({last}) in {seconds:.0f}s "
          f"-> sweep extrapolates to {hours:.1f}h "
          f"({per_row * 1000:.2f} ms/row at select iters)")
    if hours > max_hours:
        raise RuntimeError(
            f"Gate A: extrapolated sweep {hours:.1f}h exceeds the "
            f"{max_hours:.0f}h budget. Fallbacks, in order: cut `binomial` from the "
            f"ladder, shorten select chains, subsample TRAIN team-games (never "
            f"val/test). See docs/minutes-composition-plan.md.")
    return {"probe_season": last, "probe_rows": len(probe_frame),
            "probe_seconds": float(seconds), "extrapolated_hours": float(hours),
            "diagnostics": model.diagnostics}


# ── Posterior predictive checks ───────────────────────────────────────────────

def ppc(frame: pd.DataFrame, samples: np.ndarray, comparator_samples: np.ndarray,
        label: str) -> pd.DataFrame:
    """Three checks, long-form: team sums, starter share in OT, dispersion by tier."""
    arrays = ragged_arrays(frame)
    rows = []

    for source, s in (("composition", samples), ("independent", comparator_samples)):
        sums = np.add.reduceat(s, arrays["starts"], axis=1)
        rows.append({"analysis": "team_sum_abs_error", "group": source,
                     "observed": 0.0,
                     "simulated": float(np.abs(sums - arrays["n_total"][None, :]).mean()),
                     "n": int(sums.size)})

    starter = (frame["position"] < 5).to_numpy()
    ot = (frame["n_overtimes"] > 0).to_numpy()
    y = frame["y"].to_numpy(float)
    for game_class, mask in (("regulation", ~ot), ("overtime", ot)):
        sel = mask
        obs = float(y[sel & starter].sum() / max(y[sel].sum(), 1))
        for source, s in (("composition", samples), ("independent", comparator_samples)):
            sim_star = s[:, sel & starter].sum(axis=1)
            sim_all = np.maximum(s[:, sel].sum(axis=1), 1)
            rows.append({"analysis": "starter_share", "group": f"{game_class}/{source}",
                         "observed": obs,
                         "simulated": float((sim_star / sim_all).mean()),
                         "n": int(sel.sum())})

    tiers = pd.qcut(frame["w_share"], 4, labels=["q1_fringe", "q2", "q3", "q4_star"])
    mean_sim = samples.mean(axis=0)
    var_sim = samples.var(axis=0)
    for tier in tiers.cat.categories:
        sel = (tiers == tier).to_numpy()
        realized = float(((y[sel] - mean_sim[sel]) ** 2).mean())
        simulated = float(var_sim[sel].mean())
        rows.append({"analysis": "variance_ratio", "group": str(tier),
                     "observed": realized, "simulated": simulated,
                     "ratio": realized / max(simulated, 1e-9),
                     "n": int(sel.sum())})
    out = pd.DataFrame(rows)
    out.insert(0, "variant", label)
    return out


def joint_nll_table(models: dict, selected: str, comparator: dict,
                    frames: dict) -> pd.DataFrame:
    """Joint per-team-game NLL, composition vs independent — `substitution_arm` format.

    /!\\ NOT a unit-Jacobian bijection, unlike the 3PA/2PA case: the composition
    concentrates mass on the simplex slice the data always satisfies (its last step is
    deterministic), so it wins partly by knowing the constraint. Reported for
    contrast; the decision metrics are CRPS, the team-sum error and the PPCs.
    """
    rows = []
    for split in ("val",):
        frame = frames[split]
        model = models[selected][split]
        alpha, beta, _ = model.plug_in()
        eta = (model._design(frame) @ beta + alpha
               + frame["logit_prior"].to_numpy(float))
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        y = frame["y"].to_numpy(int)
        m = frame["m"].to_numpy(int)
        lo = frame["lo"].to_numpy(int)
        live = frame["is_last"].to_numpy(int) == 0

        a, b = beta_shapes(mu, np.maximum(model.rho_row(frame), RHO_MIN))
        nll = np.zeros(len(frame))
        nll[live] = -betabinom.logpmf(y[live], m[live], a[live], b[live])
        bound = live & (lo > 0)
        if bound.any():
            nll[bound] += np.log(np.clip(
                betabinom.sf(lo[bound] - 1, m[bound], a[bound], b[bound]), 1e-300, 1))
        arrays = ragged_arrays(frame)
        comp_nll = np.add.reduceat(np.nan_to_num(nll, nan=1e6), arrays["starts"])

        mu_i = np.clip(comparator["mu"][split], EPS, 1 - EPS)
        rho_g = comparator[f"rho_game_{split}"]
        n_i = np.rint(frame["game_length"].to_numpy(float)).astype(int)
        a_i, b_i = beta_shapes(mu_i, np.full_like(mu_i, rho_g))
        nll_i = -betabinom.logpmf(y, n_i, a_i, b_i)
        indep_nll = np.add.reduceat(np.nan_to_num(nll_i, nan=1e6, posinf=1e6),
                                    arrays["starts"])

        rows.append({"split": split, "arm": "composition",
                     "mean_joint_nll": float(comp_nll.mean()),
                     "n": len(arrays["starts"])})
        rows.append({"split": split, "arm": "independent",
                     "mean_joint_nll": float(indep_nll.mean()),
                     "n": len(arrays["starts"])})
    return pd.DataFrame(rows)


# `fit_ot_tail`, `sample_game_length` and `ot_tail_check` lived here until 2026-08-09 and
# are now `src/models/stan_game_length.py`. Three things were wrong with leaving them: they
# were a point estimate where every other simulator input is a posterior; the simulator
# would have had to import this nine-hour head to draw a game length; and they were
# unconditional, where a fitted season trend cuts the OT-class error on validation from
# 22.97 to 9.42 summed games. The pooled pair survives there as the head's mandatory no-fit
# floor and still reads p_any = 0.0608 / p_more = 0.1408 on the same 30,626 training games,
# so nothing about the incumbent's numbers moved — only who owns them. Registered as
# `withdrawn` in `dashboard/decisions.py`.


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_stan = cfg.get("stan", {})
    comp_cfg = cfg_stan.get("composition", {})
    configured = str(comp_cfg.get("first_season", PILOT_FIRST_SEASON))
    first_season = head_first_season(cfg)
    max_hours = float(comp_cfg.get("max_extrapolated_hours", MAX_EXTRAPOLATED_HOURS))
    seed = int(cfg_stan.get("seed", 42))
    blend = preseason_blend(cfg)

    print("Stan composition head — team-game minutes as a decomposed multinomial")
    print(f"  pilot window: {first_season} on (the frame spans every season for "
          f"lags and rookie priors)")
    if blend is not None:
        print(f"  preseason-blended `w_share`: k = {blend[0]:g} on the {blend[1]} route "
              f"(docs/preseason-plan.md P5).")
        if first_season > configured:
            print(f"  The window is cut from {configured} to {first_season} — the "
                  f"preseason panel's first\n  intact-tail season, read off "
                  f"`preseason_coverage.csv`. Session 4d priced that cut at\n  "
                  f"-0.01991 [-0.02515, -0.01498] CRPS minutes per player-game, in the "
                  f"arm's favour.")
    frame = head_frame(cfg)
    pilot = frame[frame["season"] >= first_season].reset_index(drop=True)

    train, val = selection_split(pilot, TEST_SEASONS)
    print("  The test split is LOCKED — this sweep fits and scores VALIDATION only\n"
          "  (src/models/held_out.py). The head's arm was already selected on "
          "`val_crps`.\n  NOTE: this head is not a registered `make final-evaluation` "
          "head, so nothing\n  currently takes its held-out reading — see the module "
          "docstring.")
    for name, part in (("train", train), ("val", val)):
        n_teams = part.groupby(GROUP_KEYS, sort=False).ngroups
        print(f"  {name}: {len(part):,} rows / {n_teams:,} team-games "
              f"({', '.join(sorted(part['season'].unique()))})")

    probe = probe_timing(train, cfg_stan, max_hours)

    print("\n  fitting the incumbent comparator (season head + rho_game, "
          "independent draws)...")
    comparator = independent_comparator(cfg, pilot, val, cfg_stan, seed)
    print(f"  comparator coverage: val {comparator['coverage_val']:.1%} of rows from "
          f"the season head\n  (the rest fall back to the carry-forward share)")

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]) / "stan_composition"
    table, diagnostics, models = sweep(train, val, cfg_stan, comparator, ckpt_dir)
    diagnostics.append(probe["diagnostics"])
    diagnostics.append(comparator["diagnostics_val"])

    print("\nVariant sweep (CRPS in minutes per player-game, lower is better):")
    print(table[["variant", "n_features", "val_crps", "val_r2", "val_pit_ks",
                 "val_bias", "val_team_sum_abs",
                 "selected", "beats_floor"]].round(4).to_string(index=False))
    selected = table.loc[table["selected"], "variant"].iloc[0]
    floor = table[table["variant"] == "carry_forward"].iloc[0]
    incumbent = table[table["variant"] == "independent_comparator"].iloc[0]
    chosen = table[table["variant"] == selected].iloc[0]
    print(f"\n  Selected on VALIDATION: {selected}. There is no test column.")
    print(f"  vs the floor:     CRPS {chosen['val_crps']:.4f} against "
          f"{floor['val_crps']:.4f} ({chosen['val_crps'] - floor['val_crps']:+.4f})")
    print(f"  vs the incumbent: CRPS {chosen['val_crps']:.4f} against "
          f"{incumbent['val_crps']:.4f} "
          f"({chosen['val_crps'] - incumbent['val_crps']:+.4f}); team-sum error "
          f"{chosen['val_team_sum_abs']:.2f} against "
          f"{incumbent['val_team_sum_abs']:.2f} minutes per team-game")
    if not bool(chosen["beats_floor"]):
        print("  /!\\  The selected variant does NOT clear the no-fit floor. Per "
              "CLAUDE.md that is not a model.")

    graded = models.get("betabinom_ot_graded")
    twin = models.get("betabinom_ot")
    # The shared-rho twin is the ONLY comparison that isolates the grading (same
    # features, same mean function), so it must never be cut from the ladder — but a
    # bare KeyError here would fire after the whole sweep and before any CSV is
    # written, which at full window is ~14 h of compute lost to a lookup.
    if graded is not None and graded["val"].rho_by_bin is not None and twin is not None:
        shared = twin["val"].rho
        bins = graded["val"].rho_by_bin
        print(f"\nFitted dispersion by prior-share bin (fringe -> star): "
              f"{', '.join(f'{r:.4f}' for r in bins)}")
        print(f"  against a single shared rho of {shared:.4f} — the graded arm differs "
              f"from `betabinom_ot`\n  in the dispersion ALONE, so the contrast is the "
              f"pilot's 1.59-to-0.70 variance ratio\n  and nothing else.")

    # PPC on the selected variant AND its shared-rho twin where both were fitted:
    # "did grading fix the tier miscalibration" is the whole question, and it is a
    # comparison, so both rows have to be on disk rather than one inferred from the
    # other.
    ppc_arms = [selected] + [a for a in ("betabinom_ot", "betabinom_ot_graded")
                             if a in models and a != selected]
    checks = pd.concat(
        [ppc(models[a]["val_frame"],
             models[a]["val"].predict_samples(models[a]["val_frame"], seed),
             comparator["samples"]["val"], a) for a in ppc_arms],
        ignore_index=True)
    print("\nPosterior predictive checks (validation split):")
    print(checks.round(4).to_string(index=False))

    joint = joint_nll_table(models, selected, comparator,
                            {"val": models[selected]["val_frame"]})
    print("\nJoint per-team-game NLL (plug-in) — composition vs independent:")
    print(joint.round(3).to_string(index=False))
    print("  /!\\  NOT a unit-Jacobian bijection (unlike the 3PA/2PA case): the\n"
          "  composition's last step is deterministic on the simplex slice, so it\n"
          "  wins partly by knowing the constraint. Contrast, not a headline.")

    print("\nGame length: this head consumes the REALIZED length on every row it fits or "
          "scores.\n  The forward draw moved to `make stan-game-length` on 2026-08-09 — "
          "see the module docstring.")

    diag = diagnostics_frame(diagnostics)
    rho_rows = []
    for label, holder in models.items():
        model = holder["val"]
        if model.rho_by_bin is None:
            continue
        for b, value in enumerate(model.rho_by_bin, start=1):
            rho_rows.append({"variant": label, "n_rho": model.n_rho, "bin": b,
                             "rho": float(value)})
    artifacts = {
        "metrics": (table.assign(probe_hours=probe["extrapolated_hours"]),
                    out_dir / "stan_composition_metrics.csv"),
        "diagnostics": (diag, out_dir / "stan_composition_diagnostics.csv"),
        "ppc": (checks, out_dir / "stan_composition_ppc.csv"),
        "dispersion": (pd.DataFrame(rho_rows),
                       out_dir / "stan_composition_dispersion.csv"),
        "joint_nll": (joint, out_dir / "stan_composition_joint_nll.csv"),
    }
    paths = {}
    for name, (df, dest) in artifacts.items():
        df.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(df):,} {name} rows → {dest}")
    print(f"\nSampler: max R-hat {diag['max_rhat'].max():.4f}, "
          f"{int(diag['divergences'].sum())} divergences over {len(diag)} fits, "
          f"{diag['wall_clock_s'].sum() / 60:.1f} min total")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
