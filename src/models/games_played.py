"""The games-played spell process — the point-MLE reference, in numpy.

`src/models/stan_availability.py` ships the season-level beta-binomial: games played out
of team games, held-out CRPS 10.795 on 911 rows. It says *how many* games a player misses
and nothing at all about *which*, and the simulator needs which — a weekly best-7-of-16
lineup is a max over a correlated set, so two players who miss the same week are a very
different roster from two who miss different weeks at identical marginal availability.

This module is the process that answers it, and it is deliberately **Stan-free** so Gate 0
runs on a machine with no CmdStan toolchain. `stan_games_played.py` fits the same
structure by NUTS and imports the simulator from here, exactly as `stan_availability.py`
is verified against `availability.py`.

## The frame: entry x exit x a within-tenure chain

A plain full-window two-state chain **does not work**, and Gate 0 measures the failure
rather than asserting it. A waived player's cell has a recovery hazard of about zero, so a
recurrent chain makes him absorbing **from his first onset** instead of from the game he
was actually cut: the departure gets relocated earlier in the season, dragging the mean
down and the left tail out. *A departure is an absorbing hitting time, not a low recovery
rate.*

The fix keeps all 30 seasons, because `in_appearance_window` identifies the tenure
structurally on every one of them:

    team_games = pre_tenure + tenure_games + post_tenure
    gp         = played games inside [first appearance, last appearance]

so the head is **entry index x exit index x within-tenure two-state chain**. Each factor
answers a separate question — when he joined, when he stopped, how often he missed while
there — and the decomposition reconstructs full-window `gp_share` with no residual.

Two things fall out for free, and both are load-bearing:

- **The within-tenure duration head needs no censoring branch.** A tenure opens and closes
  on games he played, so every absence inside it is interior. Censoring is entirely a
  property of the *edge* spells, which are exactly what the entry and exit heads model.
- **The initial state is known.** `collapse_transitions` is invariant to `s0`, so a plain
  chain needs a separate head for it; a tenure begins on an appearance by construction.

## Why the collapse is exact

Every feature the head uses is constant within a player-season, so for observed states the
game-level Bernoulli likelihood equals `h^a (1-h)^b r^c (1-r)^d` up to a factor free of the
parameters. Those four counts are sufficient — the same algebraic collapse the count heads
use, not an approximation. 1,297,766 full-window transitions reduce to 32,944 binomial
rows, a 39.4x collapse, and `tests/test_games_played.py` pins the identity first because it
is what makes the whole design legitimate.

## The dispersion budget is over-supplied, not short

`docs/availability-plan.md` reads clustering's 3.96x against the GP marginal's 22.7x and
concludes heterogeneity must supply the rest. Two corrections, both in
`docs/games-played-plan.md`: the composition is **additive**, `inflation = C + rho*(n - C)`,
not multiplicative; and the two figures are measured on different windows *and* different
populations. Matched, the same population's clustering is C = 9.58, so stacking the
incumbent's fitted rho on top of it predicts 29.6 against a target of 22.70 — a 30%
overshoot. `closed_form_calibration` inverts that identity, and it is both the documented
fallback if Gate D fails and the invariant every fitted arm should satisfy at its own C.

Usage:
    python -m src.models.games_played
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit

from src.features.availability import (CELL_KEYS, build_panel, collapse_transitions,
                                       spell_classes, tenure_frame)

# Guard rails on the beta-geometric, mirroring RHO_MIN/RHO_MAX's role on the
# beta-binomial: mu is a probability and kappa a concentration, and both objectives go
# non-finite at the open ends rather than merely uninteresting.
MU_MIN, MU_MAX = 1e-6, 1.0 - 1e-6
KAPPA_MIN, KAPPA_MAX = 1e-3, 1e7

# `rng.geometric` needs p > 0, and a frailty draw can land arbitrarily close to zero. The
# floor caps a simulated spell at 1e4 games, which every schedule truncates anyway.
Q_FLOOR = 1e-4

# Monte Carlo regularization on the simulated GP pmf, in pseudo-counts spread over the
# row's support. At 10,000 simulated seasons this moves any probability by <= 5e-5 —
# far below the Monte Carlo error it exists to remove, which is a pmf cell of exactly 0
# at an observed value whose true probability is small but positive. That would make the
# randomized PIT degenerate at a boundary and read as miscalibration.
MC_PSEUDO_COUNT = 0.5

# The population every tail figure in this project is quoted on: last season's regulars.
ROTATION_MIN_MPG = 20.0
ROTATION_MIN_GP_SHARE = 0.70
TAIL_THRESHOLDS = [41, 60]

GATE0_SIMS = 200


# ── The beta-geometric duration ───────────────────────────────────────────────
#
# A geometric per-game exit hazard with a Beta frailty integrated out analytically: the
# same device `betabinomial_glm.stan` uses for the season count, one level down. It is
# parameterized as mu = a/(a+b), which IS P(T = 1) — the first-game exit hazard, so a
# linear predictor on it reads directly — and kappa = a + b, with kappa -> infinity the
# plain geometric.

def beta_shapes(mu: np.ndarray, kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(np.asarray(mu, dtype=float), MU_MIN, MU_MAX)
    kappa = np.clip(np.asarray(kappa, dtype=float), KAPPA_MIN, KAPPA_MAX)
    return kappa * mu, kappa * (1.0 - mu)


def beta_geometric_logpmf(t: np.ndarray, mu: np.ndarray,
                          kappa: np.ndarray) -> np.ndarray:
    """log P(T = t) = log B(a+1, b+t-1) - log B(a, b)."""
    a, b = beta_shapes(mu, kappa)
    t = np.asarray(t, dtype=float)
    return betaln(a + 1.0, b + t - 1.0) - betaln(a, b)


def beta_geometric_logsf(t: np.ndarray, mu: np.ndarray,
                         kappa: np.ndarray) -> np.ndarray:
    """log P(T >= t) = log B(a, b+t-1) - log B(a, b) — right-censoring, one branch."""
    a, b = beta_shapes(mu, kappa)
    t = np.asarray(t, dtype=float)
    return betaln(a, b + t - 1.0) - betaln(a, b)


def beta_geometric_loglik(t: np.ndarray, mu: np.ndarray, kappa: np.ndarray,
                          censored: np.ndarray | None = None) -> np.ndarray:
    """One expression for both classes, and it is the Stan target verbatim.

    A censored row contributes `P(T >= t)` and an observed row `P(T = t)`; they differ
    only by whether the first shape parameter is incremented, so
    `lbeta(a + (1 - censored), b + t - 1) - lbeta(a, b)` covers the pair with no branch.
    """
    a, b = beta_shapes(mu, kappa)
    t = np.asarray(t, dtype=float)
    c = np.zeros_like(t) if censored is None else np.asarray(censored, dtype=float)
    return betaln(a + (1.0 - c), b + t - 1.0) - betaln(a, b)


def fit_beta_geometric(t: np.ndarray, w: np.ndarray | None = None,
                       censored: np.ndarray | None = None,
                       truncated: np.ndarray | None = None) -> dict:
    """MLE of (mu, kappa) — and of the in-progress offset when there is one to fit.

    **The objective is normalized by the weight sum before optimizing.** This project has
    already been bitten once by L-BFGS-B stopping on finite-difference noise: a raw sum
    over ~80,000 log-densities has magnitude ~1e5 while a finite-difference step moves it
    by ~1e-3, so the optimizer returns near-initial parameters and looks converged. Fitting
    the *mean* log-likelihood puts the objective at O(1) and removes the failure mode
    outright, and a Nelder-Mead polish confirms the optimum rather than trusting it.

    `truncated` marks spells already in progress at the timeline's first game. Their
    residual duration is beta-geometric with the frailty size-biased by 1/q — i.e.
    **Beta(a-1, b)** — so the offset is fitted freely and compared against that identity:
    agreement validates the renewal assumption, disagreement localizes it. It is fitted
    only when some row carries the flag; otherwise the parameter is unidentified and is
    not introduced at all, the same device `S = 0` uses for the year effect.
    """
    t = np.asarray(t, dtype=float)
    w = np.ones_like(t) if w is None else np.asarray(w, dtype=float)
    c = np.zeros_like(t) if censored is None else np.asarray(censored, dtype=float)
    tr = np.zeros_like(t) if truncated is None else np.asarray(truncated, dtype=float)
    fit_shift = bool(tr.any())
    total = float(w.sum())

    def negative(theta: np.ndarray) -> float:
        eta = theta[0] + (theta[2] * tr if fit_shift else 0.0)
        value = beta_geometric_loglik(t, expit(eta), np.exp(theta[1]), c)
        if not np.all(np.isfinite(value)):
            return 1e12
        return -float(np.dot(w, value)) / total

    share_one = float(np.dot(w, (t <= 1) & (c == 0)) / max(total, 1.0))
    start = np.array([logit(np.clip(share_one, 0.05, 0.95)), np.log(3.0), 0.0])
    best = minimize(negative, start, method="L-BFGS-B")
    best = minimize(negative, best.x, method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})

    mu = float(expit(best.x[0]))
    kappa = float(np.clip(np.exp(best.x[1]), KAPPA_MIN, KAPPA_MAX))
    a, b = beta_shapes(mu, kappa)
    out = {"mu": mu, "kappa": kappa, "a": float(a), "b": float(b),
           "nll": float(best.fun * total), "n_spells": int(total),
           "open_shift": float(best.x[2]) if fit_shift else np.nan}
    # E[T] = (a+b-1)/(a-1) exists only for a > 1. With proper right-censoring the fitted
    # `a` lands below 1 on the full window, so the mean is genuinely infinite — the
    # simulator does not care (a spell is truncated by the remaining schedule anyway) but
    # a quoted E[T] would be meaningless, and `make docs-audit` would happily pin it.
    out["mean_defined"] = bool(a > 1.0)
    out["mean_spell"] = float((a + b - 1.0) / (a - 1.0)) if a > 1.0 else np.nan
    return out


def duration_rows(spells: pd.DataFrame, features: list[str] | None = None,
                  design: pd.DataFrame | None = None) -> pd.DataFrame:
    """Spells collapsed to distinct rows with a multiplicity weight.

    Without covariates the likelihood depends on a spell only through
    `(length, censored, truncated)`, so ~83,000 spells reduce to a few hundred rows and
    the duration head costs nothing. With covariates the key gains the player-season, and
    the reduction is milder but still real. `tests/test_games_played.py` pins that the
    weighted likelihood equals the uncollapsed one.
    """
    keys = ["spell_games", "censored", "truncated"]
    frame = spells
    if features:
        if design is None:
            raise ValueError("covariate collapse needs the player-season design frame")
        frame = spells.merge(design[["season", "player_id"] + features],
                             on=["season", "player_id"], how="inner")
        keys = keys + ["season", "player_id"]
    out = (frame.groupby(keys, as_index=False).size()
           .rename(columns={"size": "w"}))
    if features:
        out = out.merge(design[["season", "player_id"] + features],
                        on=["season", "player_id"], how="left")
    return out.rename(columns={"spell_games": "t"})


def left_truncation_check(spells: pd.DataFrame) -> dict:
    """Fit the in-progress offset freely, then compare it against `Beta(a-1, b)`.

    By memorylessness the forward recurrence time of a geometric is geometric with the
    same hazard, so under a Beta(a, b) frailty the *residual* duration of a spell already
    in progress is beta-geometric with the frailty size-biased by `1/e` — i.e.
    **Beta(a-1, b)**. That costs zero parameters and is a prediction, so fitting the offset
    freely and comparing turns the renewal assumption into a check: agreement validates it,
    disagreement localizes it.

    **Fitted jointly with the interior spells, which is what identifies it.** On the
    left-truncated spells alone the base `mu` and the offset are the same parameter and the
    optimizer walks the shift to infinity — measured, and the reason this is not a
    two-line call. Right-censored spells are excluded: they are the *other* edge and the
    identity says nothing about them.

    The identity needs `a > 1`, since length-biasing by `1/q` is normalizable only then.
    The full-window fit lands at `a = 0.72`, so it is reported as undefined rather than
    computed anyway — the mean is infinite there and a "predicted" value would be noise
    dressed as a check.
    """
    usable = spells[spells["censored"] == 0]
    collapsed = duration_rows(usable)
    fit = fit_beta_geometric(collapsed["t"].to_numpy(), collapsed["w"].to_numpy(),
                             truncated=collapsed["truncated"].to_numpy())
    a, b = fit["a"], fit["b"]
    defined = a > 1.0
    predicted = (a - 1.0) / (a - 1.0 + b) if defined else np.nan
    fitted_open = float(expit(logit(np.clip(fit["mu"], MU_MIN, MU_MAX))
                              + fit["open_shift"]))
    return {**fit,
            "n_interior": int((usable["truncated"] == 0).sum()),
            "n_truncated": int((usable["truncated"] == 1).sum()),
            "mu_open_fitted": fitted_open,
            "mu_open_predicted": float(predicted),
            "identity_defined": bool(defined),
            "gap": float(fitted_open - predicted) if defined else np.nan}


def censoring_bias_table(spells: pd.DataFrame) -> pd.DataFrame:
    """The same spells fitted three ways — and two of the three fail silently.

    Failure mode 3 in the plan is "the duration head fits roster mechanics and gets called
    an injury model". This is the cheap check for it: emit the spell classes and what
    mishandling them costs, *before* fitting anything. Ignoring censored spells understates
    the extreme tail by ~2x and treating them as complete still understates it — neither
    raises, neither looks wrong, and both change the tail the head exists to get right.
    """
    rows = []
    for label, sel, censored in [
            ("drop_censored", spells["censored"] == 0, False),
            ("censored_as_complete", spells["spell_games"] > 0, False),
            ("proper_censoring", spells["spell_games"] > 0, True)]:
        part = spells[sel]
        collapsed = duration_rows(part)
        fit = fit_beta_geometric(
            collapsed["t"].to_numpy(), collapsed["w"].to_numpy(),
            collapsed["censored"].to_numpy() if censored else None)
        rows.append({"treatment": label, "n_spells": int(len(part)),
                     "a": fit["a"], "b": fit["b"], "mu": fit["mu"],
                     "kappa": fit["kappa"], "nll": fit["nll"],
                     "mean_defined": fit["mean_defined"],
                     "mean_spell": fit["mean_spell"],
                     "p_ge_26": float(np.exp(beta_geometric_logsf(
                         26, fit["mu"], fit["kappa"])))})
    return pd.DataFrame(rows)


def duration_candidates(spells: pd.DataFrame) -> pd.DataFrame:
    """Beta-geometric against the geometric null it has to beat, on interior spells.

    The geometric is what a constant-hazard two-state chain implies, and
    `make availability-profile` already falsifies it in both tails (0.483 of spells are a
    single game against 0.308 predicted; 0.0635 are 10+ against 0.0365). This puts a number
    on that: one extra parameter, and the log-likelihood moves by five figures.
    """
    collapsed = duration_rows(spells)
    t = collapsed["t"].to_numpy()
    w = collapsed["w"].to_numpy()
    total = float(w.sum())
    rows = []
    bg = fit_beta_geometric(t, w)
    # The geometric is the kappa -> infinity limit, fitted by its own one-parameter MLE:
    # the exit hazard is the reciprocal of the mean spell length.
    q = float(total / np.dot(w, t))
    geo_nll = -float(np.dot(w, np.log(q) + (t - 1.0) * np.log1p(-q)))
    for label, params, k, nll in [("beta_geometric", bg, 2, bg["nll"]),
                                  ("geometric", {"mu": q, "kappa": KAPPA_MAX}, 1,
                                   geo_nll)]:
        mu, kappa = params["mu"], params["kappa"]
        a, b = beta_shapes(mu, kappa)
        rows.append({
            "model": label, "n_params": k, "nll": nll, "aic": 2 * nll + 2 * k,
            "mu": mu, "kappa": kappa, "a": float(a), "b": float(b),
            "p_eq_1": float(np.exp(beta_geometric_logpmf(1, mu, kappa))),
            "p_ge_10": float(np.exp(beta_geometric_logsf(10, mu, kappa))),
            "p_ge_26": float(np.exp(beta_geometric_logsf(26, mu, kappa)))})
    rows.append({"model": "observed", "n_params": np.nan, "nll": np.nan,
                 "aic": np.nan, "mu": np.nan, "kappa": np.nan,
                 "a": np.nan, "b": np.nan,
                 "p_eq_1": float(np.dot(w, t == 1) / total),
                 "p_ge_10": float(np.dot(w, t >= 10) / total),
                 "p_ge_26": float(np.dot(w, t >= 26) / total)})
    return pd.DataFrame(rows)


# ── The simulator ─────────────────────────────────────────────────────────────

def simulate_gp(team_games: np.ndarray, params: dict, n_sims: int,
                seed: int = 0, force_endpoints: bool = True,
                return_played: bool = False):
    """(rows x n_sims) simulated games played, from the entry x exit x chain process.

    Vectorized over `rows * n_sims` chains with one loop over the game index, which is the
    only dimension that is genuinely sequential. `params` supplies beta shapes per row:
    `entry_a/entry_b` and `exit_a/exit_b` for the two tenure factors (or `pre`/`post`
    arrays to hold the tenure at its observed value, which is what arm A1 does),
    `onset_a/onset_b` for the per-player-season hazard frailty, and `dur_a/dur_b` for the
    spell-length frailty.

    **The onset frailty is drawn once per simulated player-season and the duration frailty
    once per spell**, which is exactly what the two likelihoods integrate out. The
    distinction is not cosmetic: `rng.beta(a, b)` without `size=` returns a *scalar*, and
    that bug silently gave a whole simulated population one shared hazard during planning.
    It does not announce itself — the marginal still looks plausible — so
    `tests/test_games_played.py` asserts the drawn vector's length and variance.

    `force_endpoints` distinguishes the shipped process (a tenure opens and closes on an
    appearance, so those two games are played by construction) from the plain full-window
    chain Gate 0 measures and rejects.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(team_games)
    T = np.repeat(np.asarray(team_games, dtype=np.int64), n_sims)
    n = len(T)

    def per_row(name: str) -> np.ndarray:
        return np.repeat(np.asarray(params[name], dtype=float), n_sims)

    if not force_endpoints:
        # A plain recurrent chain has no tenure at all: it runs over the whole schedule
        # and is free to end in either state. This is the arm Gate 0 rejects.
        entry_idx = np.zeros(n, dtype=np.int64)
        exit_idx = T - 1
    else:
        if "pre" in params:                   # oracle tenure — observed, not drawn
            pre = np.repeat(np.asarray(params["pre"], dtype=np.int64), n_sims)
            post = np.repeat(np.asarray(params["post"], dtype=np.int64), n_sims)
        else:
            pre = rng.binomial(np.maximum(T - 1, 0),
                               rng.beta(per_row("entry_a"), per_row("entry_b")))
            post = rng.binomial(np.maximum(T - 1 - pre, 0),
                                rng.beta(per_row("exit_a"), per_row("exit_b")))
        entry_idx = np.minimum(pre, T - 1)
        exit_idx = np.maximum(T - 1 - post, entry_idx)

    hazard = rng.beta(per_row("onset_a"), per_row("onset_b"))
    dur_a, dur_b = per_row("dur_a"), per_row("dur_b")

    def spell_lengths(mask: np.ndarray, cap: np.ndarray) -> np.ndarray:
        q = np.clip(rng.beta(dur_a[mask], dur_b[mask]), Q_FLOOR, 1.0)
        return np.minimum(rng.geometric(q), cap)

    absence_left = np.zeros(n, dtype=np.int64)
    if not force_endpoints and "initial_played" in params:
        # A plain chain can open in the missed state; the shipped process cannot.
        opens_missed = np.repeat(
            np.asarray(params["initial_played"], dtype=np.int64), n_sims) == 0
        if opens_missed.any():
            absence_left[opens_missed] = spell_lengths(opens_missed, T[opens_missed])

    # The last game index an absence may occupy. Under `force_endpoints` the tenure's final
    # game is played by construction, so a spell has to stop one short of it.
    last_absence = exit_idx - 1 if force_endpoints else exit_idx

    gp = np.zeros(n, dtype=np.int64)
    # Only materialized on request: (rows x sims x games) of int8 is ~15 MB at the
    # predictive's shape, which is fine for a diagnostic and pure waste on the hot path.
    sequence = (np.zeros((n, int(T.max())), dtype=np.int8) if return_played else None)
    for g in range(int(T.max())):
        live = (g >= entry_idx) & (g <= exit_idx)
        missing_now = live & (absence_left > 0)
        played_now = live & ~missing_now
        gp += played_now
        if sequence is not None:
            sequence[:, g] = played_now
        absence_left -= missing_now

        # The onset decision belongs to game g+1 and is taken here, immediately after a
        # played game — which is what makes "at risk" mean "the previous game was played"
        # and keeps the simulated at-risk count equal to the collapsed statistic.
        eligible = played_now & (g + 1 <= last_absence)
        if not eligible.any():
            continue
        # The hazard is constant within a simulated season. The measured post-return bump
        # — 2.0x / 1.5x / 1.2x at played streaks of 1 / 2 / 3, gone by game 4 — would need
        # the streak, which is per-game state the collapse discards by design. Adding it
        # here is an explicit-latent model, not a refinement of this one.
        fires = eligible & (rng.random(n) < hazard)
        if fires.any():
            absence_left[fires] = spell_lengths(fires, last_absence[fires] - g)
    if sequence is not None:
        return gp.reshape(n_rows, n_sims), sequence
    return gp.reshape(n_rows, n_sims)


def gp_pmf(sims: np.ndarray, max_games: int,
           pseudo_count: float = MC_PSEUDO_COUNT) -> np.ndarray:
    """(rows x max_games+1) Monte Carlo pmf over games played, lightly regularized.

    The regularization is stated in pseudo-counts rather than hidden as a probability
    floor: at 10,000 simulated seasons it moves any cell by <= 5e-5. It exists because a
    Monte Carlo pmf cell of exactly zero at the *observed* value makes the randomized PIT
    collapse onto a boundary, which reads as miscalibration and is an artifact of the
    draw count.
    """
    k = max_games + 1
    counts = np.zeros((sims.shape[0], k), dtype=float)
    clipped = np.clip(sims, 0, max_games)
    for row in range(sims.shape[0]):
        counts[row] = np.bincount(clipped[row], minlength=k)[:k]
    counts += pseudo_count / k
    return counts / counts.sum(axis=1, keepdims=True)


# ── Gate 0: does the process class reproduce the GP marginal at its ceiling? ───

def empirical_hazards(counts: pd.DataFrame) -> pd.DataFrame:
    """Per-cell observed onset and recovery rates, pooled where a cell has no exposure.

    The most generous parameterization available — no covariate block can beat a cell's
    own observed rates — which is what makes Gate 0 a statement about the **process class**
    rather than about any particular feature set. If it fails here it fails everywhere.
    """
    out = counts.copy()
    pooled_h = float(out["onsets"].sum() / max(out["at_risk_played"].sum(), 1))
    pooled_r = float(out["recoveries"].sum() / max(out["at_risk_missed"].sum(), 1))
    out["h"] = np.where(out["at_risk_played"] > 0,
                        out["onsets"] / out["at_risk_played"].replace(0, np.nan),
                        pooled_h)
    out["r"] = np.where(out["at_risk_missed"] > 0,
                        out["recoveries"] / out["at_risk_missed"].replace(0, np.nan),
                        pooled_r)
    out["h"] = out["h"].fillna(pooled_h).clip(MU_MIN, MU_MAX)
    out["r"] = out["r"].fillna(pooled_r).clip(MU_MIN, MU_MAX)
    return out


def _marginal_summary(gp: np.ndarray, team_games: np.ndarray) -> dict:
    share = gp / team_games
    mean = float(np.mean(share))
    sd = float(np.std(share, ddof=1))
    games = float(np.median(team_games))
    binomial_sd = float(np.sqrt(mean * (1 - mean) / games))
    return {"mean_gp_share": mean, "sd_gp_share": sd,
            "overdispersion": float(sd ** 2 / binomial_sd ** 2),
            "p_below_41": float(np.mean(gp < 41)),
            "p_below_60": float(np.mean(gp < 60))}


def gate_zero(panel: pd.DataFrame, population: set[tuple[str, int]],
              n_sims: int = GATE0_SIMS, seed: int = 42,
              single_team_only: bool = True) -> pd.DataFrame:
    """The Monte Carlo the plan's frame change rests on — reproduce it, do not assume it.

    Three arms on the same rotation player-seasons, all at their per-cell empirical
    ceiling:

    - `full_window_chain` — one recurrent two-state chain over the whole schedule. This is
      the arm that **fails**, and the failure is the reason the head is a tenure
      decomposition.
    - `appearance_chain` — the same chain confined to the tenure, scored against games
      played *within* the tenure. Where the plain chain works.
    - `tenure_decomposition` — observed entry and exit plus that chain, scored against
      full-window `gp_share`. The shipped process class.

    **Single-team player-seasons only, and this is not a detail.** The panel's process runs
    per (season, player, team) while games played is a per (player, season) quantity, so on
    a multi-team row `gp` counts one team's games against that team's whole schedule — a
    traded player reads as having missed half the season twice over. Measured: leaving them
    in moves the *observed* left tail from 0.109 to 0.237, which is large enough to hide
    the very overshoot this gate exists to detect. The gate would then pass the arm the
    plan rejects, for a reason that has nothing to do with the process class.
    """
    full = empirical_hazards(collapse_transitions(panel, "full"))
    inside = empirical_hazards(collapse_transitions(panel, "appearance"))
    tenure = tenure_frame(panel)

    keep = [(s, p) in population for s, p in zip(tenure["season"], tenure["player_id"])]
    tenure = tenure[keep]
    if single_team_only:
        multi = multi_team_seasons(panel)
        tenure = tenure[[(s, p) not in multi
                         for s, p in zip(tenure["season"], tenure["player_id"])]]
    frame = (tenure.merge(inside[CELL_KEYS + ["h", "r"]], on=CELL_KEYS, how="left")
             .merge(full[CELL_KEYS + ["h", "r", "initial_state"]], on=CELL_KEYS,
                    how="left", suffixes=("_in", "_full")))
    frame = frame.dropna(subset=["h_in", "h_full"]).reset_index(drop=True)

    T = frame["team_games"].to_numpy(np.int64)
    tenure_games = frame["tenure_games"].to_numpy(np.int64)
    gp = frame["gp"].to_numpy(np.int64)

    def geometric_spells(rate: np.ndarray) -> dict:
        # The plain chain's spell distribution IS geometric at the recovery hazard, which
        # is the kappa -> infinity limit of the duration head. Expressing it that way
        # keeps Gate 0 on the shipped simulator rather than on a second implementation.
        a, b = beta_shapes(rate, np.full(len(rate), KAPPA_MAX))
        return {"dur_a": a, "dur_b": b}

    arms = {}
    onset_full = beta_shapes(frame["h_full"].to_numpy(), np.full(len(frame), KAPPA_MAX))
    onset_in = beta_shapes(frame["h_in"].to_numpy(), np.full(len(frame), KAPPA_MAX))
    arms["full_window_chain"] = (
        {"onset_a": onset_full[0], "onset_b": onset_full[1],
         "initial_state": frame["initial_state"].to_numpy(),
         "initial_played": frame["initial_state"].to_numpy(),
         **geometric_spells(frame["r_full"].to_numpy())},
        False, T, gp)
    arms["appearance_chain"] = (
        {"onset_a": onset_in[0], "onset_b": onset_in[1],
         "pre": np.zeros(len(frame), np.int64),
         "post": np.zeros(len(frame), np.int64),
         **geometric_spells(frame["r_in"].to_numpy())},
        True, tenure_games, gp)
    arms["tenure_decomposition"] = (
        {"onset_a": onset_in[0], "onset_b": onset_in[1],
         "pre": frame["pre_tenure"].to_numpy(np.int64),
         "post": frame["post_tenure"].to_numpy(np.int64),
         **geometric_spells(frame["r_in"].to_numpy())},
        True, T, gp)

    rows = []
    for arm, (params, force, denom, observed) in arms.items():
        sims = simulate_gp(denom, params, n_sims, seed=seed, force_endpoints=force)
        obs = _marginal_summary(observed, denom)
        sim = _marginal_summary(sims.reshape(-1),
                                np.repeat(denom, sims.shape[1]))
        for metric in obs:
            rows.append({"arm": arm, "metric": metric, "observed": obs[metric],
                         "simulated": sim[metric], "n": len(frame),
                         "n_sims": n_sims})
    return pd.DataFrame(rows)


GATE0_SIGMA = 2.0


def gate_zero_verdict(gate: pd.DataFrame, sigma: float = GATE0_SIGMA) -> pd.DataFrame:
    """Pass/fail per arm on the left tail, in standard errors of the observed proportion.

    Two choices here, and both are made to stop the gate being a knob:

    **The statistic is `P(GP < 41)`, not the mean.** Every arm reproduces the mean to
    within a point — a chain running at a cell's own observed rates can hardly miss it — so
    a gate on the mean would pass the arm the plan rejects. The left tail is where a
    relocated departure shows up, and it is what Gate D is later judged on.

    **The bar is the sampling error of the observed proportion, not a chosen tolerance.**
    Monte Carlo error is negligible here (millions of simulated seasons); what is not
    negligible is that the observed tail is itself an estimate from a few thousand
    player-seasons. `sqrt(p(1-p)/n)` is that uncertainty, and asking the simulation to land
    inside two of them is the weakest bar that can still distinguish the arms — which is
    the right property for a gate whose job is to reject a process class rather than to
    rank two good ones.
    """
    tail = gate[gate["metric"] == "p_below_41"].copy()
    p, n = tail["observed"], tail["n"]
    tail["standard_error"] = np.sqrt(p * (1 - p) / n)
    tail["relative_error"] = (tail["simulated"] - p) / p
    tail["z"] = (tail["simulated"] - p) / tail["standard_error"]
    tail["passes"] = tail["z"].abs() <= sigma
    return tail[["arm", "observed", "simulated", "standard_error", "relative_error",
                 "z", "passes", "n"]]


# ── Gate C: the hazard-by-streak curve ────────────────────────────────────────

STREAK_BUCKETS = [0, 1, 2, 3, 4, 5, 10, 20, 41, 10_000]


def hazard_by_streak(panel: pd.DataFrame, window: str = "appearance",
                     population: set[tuple[str, int]] | None = None) -> pd.DataFrame:
    """Onset hazard against the number of consecutive games already played.

    `population` restricts to a set of (season, player_id) pairs. Gate C passes the
    **evaluation rows**, because the simulated curve is produced on those rows and putting
    an all-seasons observed curve beside a test-only simulated one would be comparing two
    populations in one table — the same error as quoting a variance-budget share without
    its basis.

    It falls ~12x from a one-game streak to past 41, which looks like state dependence and
    breaks the collapse if it is. It is not: a pure-frailty model with **zero** state
    dependence reproduces almost exactly that shape purely by sorting — players with high
    onset hazards break their streaks early, so long streaks are populated by low-hazard
    players. The frailty the beta-binomial marginalizes is precisely what generates this
    curve, so the collapse keeps it for free.

    Gate C reads this off the *simulated* seasons and checks the curve keeps falling past
    k = 20. A simulator that flattens there has lost the frailty.
    """
    from src.features.availability import _select_window

    df = _select_window(panel, window)
    if population is not None:
        df = df[[(s, p) in population
                 for s, p in zip(df["season"], df["player_id"])]]
    df = df.sort_values(CELL_KEYS + ["team_game_index"])
    played = df["played"].to_numpy(np.int64)
    cell_change = (df.groupby(CELL_KEYS, sort=False).cumcount() == 0).to_numpy()
    return _streak_table(played, cell_change)


def _streak_table(played: np.ndarray, cell_change: np.ndarray) -> pd.DataFrame:
    """Streak length before each transition, then the onset rate per streak bucket."""
    streak = np.zeros(len(played), dtype=np.int64)
    run = 0
    for i in range(len(played)):
        if cell_change[i]:
            run = 0
        run = run + 1 if played[i] else 0
        streak[i] = run
    # A transition exists only where the next row is the same cell.
    same_next = np.append(~cell_change[1:], False)
    at_risk = (streak > 0) & same_next
    onset = at_risk & (np.append(played[1:], 0) == 0)

    k = streak[at_risk]
    fired = onset[at_risk]
    labels = pd.cut(k, STREAK_BUCKETS, labels=streak_labels())
    out = (pd.DataFrame({"streak": labels, "onset": fired})
           .groupby("streak", observed=True, as_index=False)
           .agg(at_risk=("onset", "size"), onsets=("onset", "sum")))
    out["hazard"] = out["onsets"] / out["at_risk"].replace(0, np.nan)
    return out


def simulated_hazard_by_streak(team_games: np.ndarray, params: dict, n_sims: int,
                               seed: int = 0) -> pd.DataFrame:
    """The same curve, read off simulated seasons — the Gate C statistic.

    Re-simulates the played/missed *sequence* rather than reusing `simulate_gp`, which
    returns only the count. The duplication is the point: `simulate_gp` is the hot path
    and carries no per-game history, and threading one through it for a diagnostic would
    slow every predictive draw.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(team_games)
    T = np.repeat(np.asarray(team_games, np.int64), n_sims)
    n = len(T)

    def per_row(name: str) -> np.ndarray:
        return np.repeat(np.asarray(params[name], dtype=float), n_sims)

    if "pre" in params:
        pre = np.repeat(np.asarray(params["pre"], np.int64), n_sims)
        post = np.repeat(np.asarray(params["post"], np.int64), n_sims)
    else:
        pre = rng.binomial(np.maximum(T - 1, 0),
                           rng.beta(per_row("entry_a"), per_row("entry_b")))
        post = rng.binomial(np.maximum(T - 1 - pre, 0),
                            rng.beta(per_row("exit_a"), per_row("exit_b")))
    entry_idx = np.minimum(pre, T - 1)
    exit_idx = np.maximum(T - 1 - post, entry_idx)

    hazard = rng.beta(per_row("onset_a"), per_row("onset_b"))
    dur_a, dur_b = per_row("dur_a"), per_row("dur_b")

    absence_left = np.zeros(n, np.int64)
    streak = np.zeros(n, np.int64)
    at_risk = np.zeros((n, len(STREAK_BUCKETS) - 1), np.int64)
    onsets = np.zeros_like(at_risk)
    edges = np.array(STREAK_BUCKETS[1:-1])

    for g in range(int(T.max())):
        live = (g >= entry_idx) & (g <= exit_idx)
        missing_now = live & (absence_left > 0)
        played_now = live & ~missing_now
        absence_left -= missing_now
        streak = np.where(played_now, streak + 1, 0)

        eligible = played_now & (g + 1 <= exit_idx - 1)
        if not eligible.any():
            continue
        fires = eligible & (rng.random(n) < hazard)
        bucket = np.searchsorted(edges, streak, side="left")
        rows = np.flatnonzero(eligible)
        np.add.at(at_risk, (rows, bucket[rows]), 1)
        hit = np.flatnonzero(fires)
        np.add.at(onsets, (hit, bucket[hit]), 1)
        if fires.any():
            q = np.clip(rng.beta(dur_a[fires], dur_b[fires]), Q_FLOOR, 1.0)
            absence_left[fires] = np.minimum(rng.geometric(q),
                                             exit_idx[fires] - 1 - g)

    out = pd.DataFrame({"streak": streak_labels(),
                        "at_risk": at_risk.sum(axis=0),
                        "onsets": onsets.sum(axis=0)})
    out["hazard"] = out["onsets"] / out["at_risk"].replace(0, np.nan)
    return out


def status_agreement(panel: pd.DataFrame,
                     first_season: str = "2006-07") -> pd.DataFrame:
    """How well the structural tenure proxy agrees with the box-score `status`.

    The tenure decomposition identifies "not on the team" *structurally*, from
    `in_appearance_window`, which is what lets it run on all 30 seasons. From 2006-07 the
    box score says so directly, so the two can be cross-tabulated — and the interesting
    cell is not the agreement but the **disagreement in one particular direction**: games
    missed *outside* the appearance window while the player was **still rostered**. Those
    are season-ending and preseason injuries, they are the highest-value population in the
    whole head, and the structural proxy files them under "not on the team".

    `not_rostered` persistence rides along because it is what justifies treating a
    departure as absorbing rather than as a very low recovery rate.
    """
    sel = panel[(panel["season"] >= first_season) & (panel["status_covered"] == 1)]
    missed = sel[sel["played"] == 0]
    outside = (missed["in_appearance_window"] == 0).to_numpy()
    not_rostered = (missed["status"] == "not_rostered").to_numpy()

    ordered = sel.sort_values(CELL_KEYS + ["team_game_index"])
    flag = (ordered["status"] == "not_rostered").to_numpy()
    has_prior = (ordered.groupby(CELL_KEYS, sort=False).cumcount() > 0).to_numpy()
    prior_flag = np.roll(flag, 1)
    at_risk = has_prior & prior_flag

    rows = [
        ("agreement", float((outside == not_rostered).mean())),
        ("outside_window_and_not_rostered", float((outside & not_rostered).mean())),
        ("inside_window_and_rostered", float((~outside & ~not_rostered).mean())),
        # The cell the structural proxy cannot see, and the reason the three-state arm exists.
        ("missed_outside_window_while_rostered",
         float((outside & ~not_rostered).mean())),
        ("inside_window_and_not_rostered", float((~outside & not_rostered).mean())),
        ("not_rostered_persistence", float(flag[at_risk].mean())),
    ]
    return pd.DataFrame([{"analysis": "status_agreement", "statistic": k, "value": v,
                          "n": len(missed)} for k, v in rows])


def streak_labels() -> list[str]:
    """The streak buckets in order — the one definition both curves and the merge use.

    `hazard_by_streak` builds them through `pd.cut` (categorical) and
    `simulated_hazard_by_streak` through `searchsorted` (plain strings), so an outer merge
    on the two does not preserve order. Sorting on this list keeps the artifact and the
    printed table reading low-to-high instead of alphabetically.
    """
    return [f"{a + 1}-{b}" if b - a > 1 else str(b)
            for a, b in zip(STREAK_BUCKETS[:-1], STREAK_BUCKETS[1:])]


def order_by_streak(curve: pd.DataFrame) -> pd.DataFrame:
    order = {label: i for i, label in enumerate(streak_labels())}
    return (curve.assign(_i=curve["streak"].map(order)).sort_values("_i")
            .drop(columns="_i").reset_index(drop=True))


def curve_still_falling(curve: pd.DataFrame, after: str = "11-20") -> bool:
    """Gate C's second condition: the hazard keeps falling past a streak of 20.

    A simulator whose frailty has collapsed to a point mass produces a flat curve, which is
    the specific failure this catches — and it is invisible in the GP marginal, because a
    shared hazard and a distribution of hazards with the same mean give the same mean.
    """
    hazards = curve.set_index("streak")["hazard"]
    if after not in hazards.index:
        return False
    tail = hazards.loc[after:]
    return bool(len(tail) > 1 and tail.iloc[-1] < tail.iloc[0])


# ── The fallback: calibrate rather than fit ───────────────────────────────────

def closed_form_calibration(mu_star: np.ndarray, inflation_star: np.ndarray,
                            clustering: float, n: float) -> dict:
    """Option (b) in two lines: hazards that reproduce a given marginal and clustering.

    Given the incumbent's per-player predictive mean `mu*` and variance inflation `I*`, and
    a within-player clustering `C` measured from the transition rates, invert the additive
    identity `I = C + rho*(n - C)`:

        rho   = (I* - C) / (n - C)      # the residual frailty the chain must not double-count
        rho_M = (C - 1) / (C + 1)       # lag-1 autocorrelation reproducing C
        r     =      mu*  * (1 - rho_M) # recovery hazard
        h     = (1 - mu*) * (1 - rho_M) # onset hazard

    Exact, and it reproduces the incumbent's marginal **by construction** while getting the
    game-level clustering right. It is the documented branch if Gate D fails, and it is
    worth pinning as a test regardless of which arm wins, because it is the invariant every
    arm should satisfy at its own fitted `C`.

    **The identity is additive, not multiplicative.** "22.7 / 3.96, so the frailty must
    supply 5.7x" is the reading this function exists to make impossible.
    """
    rho = (np.asarray(inflation_star, dtype=float) - clustering) / (n - clustering)
    rho_markov = (clustering - 1.0) / (clustering + 1.0)
    mu_star = np.clip(np.asarray(mu_star, dtype=float), MU_MIN, MU_MAX)
    return {"rho_frailty": rho, "rho_markov": float(rho_markov),
            "recovery": mu_star * (1.0 - rho_markov),
            "onset": (1.0 - mu_star) * (1.0 - rho_markov)}


def variance_inflation(clustering: float, rho: float, n: float) -> float:
    """`C + rho*(n - C)` — returns `1 + (n-1)*rho` at C = 1, as it must."""
    return float(clustering + rho * (n - clustering))


def mean_matched_mu(mean_spell: np.ndarray, kappa: float,
                    horizon: int = 82) -> np.ndarray:
    """The beta-geometric `mu` whose **truncated** mean `E[min(T, horizon)]` is `mean_spell`.

    Truncated, and that is the whole point. The tempting identity is the untruncated one —
    `E[T] = (a+b-1)/(a-1)`, so `mu = (1 + (kappa-1)/E[T]) / kappa` — and it is **wrong here**
    by enough to matter: at the calibrated hazards it lands `a = 1.33`, where the mean is
    dominated by draws longer than a whole season. Those get cut off by the schedule, so the
    realized absence time is far below the target and the simulated play rate came out
    **0.8514 against a target of 0.8000**. Measured, not reasoned about.

    A season is 82 games, so what has to match is the mean of the *bounded* spell,
    `E[min(T, K)] = sum_{t=1..K} P(T >= t)`. That has no closed-form inverse, so it is
    tabulated on a grid of `mu` and interpolated — the truncated mean is monotone decreasing
    in `mu` (a higher first-game exit hazard means shorter spells), so the inverse is well
    defined. Cheap: one 400-point table shared by every row.

    This is the same class of error as the untruncated `E[T]` in the censoring table, one
    level down: a heavy-tailed distribution's mean is not a summary you can substitute into
    a bounded process.
    """
    mean_spell = np.asarray(mean_spell, dtype=float)
    grid = np.linspace(1.0 / kappa + 1e-3, MU_MAX, 400)
    t = np.arange(1, int(horizon) + 1)
    table = np.array([np.exp(beta_geometric_logsf(t, m, kappa)).sum() for m in grid])
    # `np.interp` needs ascending x; the truncated mean decreases in mu, so reverse both.
    return np.interp(np.clip(mean_spell, table.min(), table.max()),
                     table[::-1], grid[::-1])


def allocate_spells(gp: np.ndarray, team_games: np.ndarray, mu: float, kappa: float,
                    seed: int = 0) -> np.ndarray:
    """Given a games-played COUNT, lay the missed games out as realistic absence spells.

    This is the decoupling, and it removes the trade the other two arms are stuck with.
    The count comes from wherever it is best predicted — for now the incumbent's validated
    beta-binomial pmf — and this only decides *where* those absences fall. So the
    games-played marginal is not approximated, matched or calibrated: it is **exactly** the
    distribution the count was drawn from, because that is the distribution the count was
    drawn from. Every marginal metric (CRPS, PIT, MAE, the tail) is inherited unchanged.

    What it adds is the thing no marginal metric can see. Absences are partitioned into
    spells drawn from the fitted beta-geometric — the distribution that beats a geometric by
    11,278 log-likelihood points and lands within 7% of the observed P(spell >= 10 games)
    where a geometric is 43% low — and placed at random starts in the schedule. For a
    best-ball simulator the question is not "how many games" but "what is the chance he is
    gone for three weeks of Round 1", and only the spell shape answers it.

    Returns a (rows x max_games) 0/1 played matrix. The last spell is truncated to make the
    missed total exact, which biases the *simulated* spell distribution slightly short — an
    unavoidable consequence of conditioning on a fixed total, and the reason this reports
    its realized spell shape rather than assuming it inherits the head's.
    """
    rng = np.random.default_rng(seed)
    n = len(gp)
    width = int(np.max(team_games))
    played = np.zeros((n, width), dtype=np.int8)
    a, b = beta_shapes(np.full(1, mu), np.full(1, kappa))

    for i in range(n):
        T = int(team_games[i])
        played[i, :T] = 1
        missed = T - int(gp[i])
        if missed <= 0:
            continue
        # Draw spell lengths until they cover the missed total; truncate the last.
        lengths = []
        total = 0
        while total < missed:
            q = float(np.clip(rng.beta(a[0], b[0]), Q_FLOOR, 1.0))
            d = min(int(rng.geometric(q)), missed - total)
            lengths.append(d)
            total += d
        # Place them without overlap: choose starts among the available gaps.
        free = T - missed
        # `free + 1` slots between/around the played games, one per spell, sampled
        # without replacement so two spells never merge into one longer one.
        k = len(lengths)
        if k > free + 1:
            lengths = [missed]
            k = 1
        starts = np.sort(rng.choice(free + 1, size=k, replace=False))
        cursor = 0
        pos = 0
        for j, d in enumerate(lengths):
            pos = starts[j] + cursor
            played[i, pos:pos + d] = 0
            cursor += d
    return played


def spell_lengths_from(played: np.ndarray, team_games: np.ndarray) -> np.ndarray:
    """Absence-spell lengths read back off a simulated played/missed matrix."""
    out = []
    for i in range(len(played)):
        row = played[i, :int(team_games[i])]
        run = 0
        for v in row:
            if v == 0:
                run += 1
            elif run:
                out.append(run)
                run = 0
        if run:
            out.append(run)
    return np.array(out, dtype=int)


def simulate_calibrated(team_games: np.ndarray, mu_star: np.ndarray,
                        inflation_star: float | np.ndarray, clustering: float,
                        n_sims: int, seed: int = 0,
                        duration_kappa: float | None = None,
                        return_played: bool = False):
    """Option (b) as a running simulator: hazards inverted from a target marginal.

    Three exact identities make this reproduce what it is calibrated to, by construction
    rather than by fitting:

    - a two-state chain with `h = (1-mu)(1-rho_M)` and `r = mu(1-rho_M)` has stationary
      play rate exactly `mu` — the incumbent's per-player predictive mean, preserved;
    - its lag-1 autocorrelation is exactly `rho_M`, so setting `rho_M = (C-1)/(C+1)`
      reproduces a measured clustering `C` exactly;
    - a Beta frailty on `mu` with the residual `rho = (I* - C)/(n - C)` completes the
      additive variance identity to the target inflation `I*`.

    So it gets the **game-level** process — which games he misses, correlated the way the
    data says — while leaving the season marginal where the validated head already put it.
    That is the whole content of option (b): the marginal is not what is being bought.

    Unlike the fitted arms this runs a *recurrent* chain over the whole schedule, with no
    tenure factors. Gate 0 rejects that parameterization when its hazards come from each
    cell's own observed rates, because a waived player's near-zero recovery rate makes him
    absorbing from his first absence. It is not the same failure here: these hazards are
    derived from a *predicted* mean and are bounded away from zero by construction, so no
    cell can be accidentally absorbing.

    **`duration_kappa` is the hybrid**, and it is the one knob that changes the *shape* of
    an absence without touching the marginal. Left `None`, spells are geometric — the
    implicit consequence of a constant recovery hazard, and a distribution the data
    falsifies in both tails (P(spell >= 10 games) 43% low, P(>= 26) **99%** low, against
    68,530 observed spells). Set to the fitted concentration, spells are drawn from a
    beta-geometric whose mean is matched to `1/recovery`, so the stationary play rate is
    unchanged and only the tail moves. That matters for a best-ball simulator, where the
    question is not "how many games" but "what is the chance he is gone for three weeks of
    Round 1" — and a geometric answers that question with a number ~100x too small.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(team_games)
    T = np.repeat(np.asarray(team_games, dtype=np.int64), n_sims)
    n = len(T)

    cal = closed_form_calibration(mu_star, inflation_star, clustering,
                                  float(np.median(team_games)))
    rho_frailty = float(np.clip(np.mean(np.atleast_1d(cal["rho_frailty"])), 1e-6, 0.95))
    scale = (1.0 - rho_frailty) / rho_frailty
    mu = np.clip(np.repeat(np.asarray(mu_star, dtype=float), n_sims), MU_MIN, MU_MAX)
    drawn = rng.beta(mu * scale, (1.0 - mu) * scale)      # per simulated player-season

    rho_markov = cal["rho_markov"]
    onset = np.clip((1.0 - drawn) * (1.0 - rho_markov), MU_MIN, MU_MAX)
    recovery = np.clip(drawn * (1.0 - rho_markov), MU_MIN, MU_MAX)

    if duration_kappa is None:
        played = rng.random(n) < drawn                    # stationary initial state
        gp = played.astype(np.int64)
        seq = np.zeros((n, int(T.max())), dtype=np.int8) if return_played else None
        if seq is not None:
            seq[:, 0] = played
        for g in range(1, int(T.max())):
            u = rng.random(n)
            played = np.where(played, u >= onset, u < recovery)
            # Schedules differ in length (the 2011-12 and 2019-20 seasons, and a handful of
            # cancelled games), so a row stops accumulating at its own last game rather than
            # being simulated long and rescaled.
            gp += played & (g < T)
            if seq is not None:
                seq[:, g] = played & (g < T)
        if seq is not None:
            return gp.reshape(n_rows, n_sims), seq
        return gp.reshape(n_rows, n_sims)

    # The hybrid: the same alternating process, with the absence length drawn explicitly
    # from a mean-matched beta-geometric instead of implied by a constant hazard.
    mu_dur = mean_matched_mu(1.0 / recovery, duration_kappa)
    a_dur, b_dur = beta_shapes(mu_dur, np.full(n, duration_kappa))

    def spells(mask: np.ndarray) -> np.ndarray:
        q = np.clip(rng.beta(a_dur[mask], b_dur[mask]), Q_FLOOR, 1.0)
        return rng.geometric(q)

    absence_left = np.zeros(n, dtype=np.int64)
    opens_missed = rng.random(n) >= drawn                 # stationary initial state
    if opens_missed.any():
        absence_left[opens_missed] = spells(opens_missed)

    gp = np.zeros(n, dtype=np.int64)
    for g in range(int(T.max())):
        missing = absence_left > 0
        gp += (~missing) & (g < T)
        absence_left -= missing
        starts = (~missing) & (rng.random(n) < onset)
        if starts.any():
            absence_left[starts] = spells(starts)
    return gp.reshape(n_rows, n_sims)


# ── Frames ────────────────────────────────────────────────────────────────────

def multi_team_seasons(panel: pd.DataFrame) -> set[tuple[str, int]]:
    """(season, player_id) pairs that appeared for more than one team.

    12.2% of player-seasons, carrying **36.7%** of full-window missed games, because the
    full window counts a traded player as rostered all year for *both* teams. The panel's
    process runs per (season, player, team) while the incumbent's target runs per
    (player, season), so simulating per cell and summing double-counts those absences.
    Fitted rows exclude them; predicted rows do not, since every covariate is
    player-season level.
    """
    counts = panel.groupby(["season", "player_id"])["team_id"].nunique()
    return set(counts[counts > 1].index)


def process_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, player, team): tenure factors and the collapsed counts."""
    tenure = tenure_frame(panel)
    inside = collapse_transitions(panel, "appearance")[
        CELL_KEYS + ["onsets", "at_risk_played", "recoveries", "at_risk_missed"]]
    out = tenure.merge(inside, on=CELL_KEYS, how="left")
    multi = multi_team_seasons(panel)
    out["multi_team"] = [(s, p) in multi
                         for s, p in zip(out["season"], out["player_id"])]
    # The two tenure factors as successes out of trials, which is what the entry and exit
    # heads take. `team_games - 1` rather than `team_games`: a player must appear at least
    # once for the cell to exist at all, so the last game is not available to both.
    out["entry_trials"] = np.maximum(out["team_games"] - 1, 0)
    out["exit_trials"] = np.maximum(out["team_games"] - 1 - out["pre_tenure"], 0)
    return out


def rotation_population(design: pd.DataFrame) -> set[tuple[str, int]]:
    """Established rotation players, from an already-lagged design frame."""
    rot = design[(design["minutes_per_game_lag1"] >= ROTATION_MIN_MPG)
                 & (design["gp_share_lag1"] >= ROTATION_MIN_GP_SHARE)]
    return set(zip(rot["season"], rot["player_id"]))


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    cfg_gp = cfg.get("stan", {}).get("games_played", {})
    seed = int(cfg.get("stan", {}).get("seed", 42))
    n_sims = int(cfg_gp.get("gate0_sims", GATE0_SIMS))

    panel_path = features_dir / "availability_panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        stamp = pd.Timestamp(panel_path.stat().st_mtime, unit="s")
        print(f"Availability panel from cache ({stamp:%Y-%m-%d %H:%M}): "
              f"{len(panel):,} player-games")
    else:
        print("No cached availability panel; rebuilding from raw logs")
        panel = build_panel(seasons, raw_dir)

    print("\nGames-played spell process — the numpy reference (no Stan)")

    # ── the collapse ──────────────────────────────────────────────────────────
    collapse_rows = []
    for window in ("full", "appearance"):
        counts = collapse_transitions(panel, window)
        transitions = int(counts["at_risk_played"].sum()
                          + counts["at_risk_missed"].sum())
        collapse_rows.append({
            "window": window, "cells": len(counts),
            "collapsed_binomial_rows": 2 * len(counts),
            "transitions": transitions,
            "collapse_ratio": transitions / max(2 * len(counts), 1),
            "onsets": int(counts["onsets"].sum()),
            "at_risk_played": int(counts["at_risk_played"].sum()),
            "onset_hazard": float(counts["onsets"].sum()
                                  / max(counts["at_risk_played"].sum(), 1)),
            "recoveries": int(counts["recoveries"].sum()),
            "at_risk_missed": int(counts["at_risk_missed"].sum()),
            "recovery_hazard": float(counts["recoveries"].sum()
                                     / max(counts["at_risk_missed"].sum(), 1))})
    collapse = pd.DataFrame(collapse_rows)
    print("\nThe chain collapses to its sufficient statistics — exactly, not approximately:")
    print(collapse.round(4).to_string(index=False))

    # ── spell classes, and what mishandling them costs ────────────────────────
    spells = spell_classes(panel, "full")
    spells["not_rostered_games"] = (spells["not_rostered_share"]
                                    * spells["spell_games"])
    by_class = (spells.groupby("spell_class", as_index=False)
                .agg(spells=("spell_games", "size"),
                     missed_games=("spell_games", "sum"),
                     mean_length=("spell_games", "mean"),
                     not_rostered_games=("not_rostered_games", "sum")))
    by_class["share_of_spells"] = by_class["spells"] / by_class["spells"].sum()
    by_class["share_of_missed"] = (by_class["missed_games"]
                                   / by_class["missed_games"].sum())
    # Games-weighted, not the mean of per-spell shares: the question is what fraction of
    # the missed *games* in each class are roster mechanics rather than health.
    by_class["not_rostered_share"] = (by_class["not_rostered_games"]
                                      / by_class["missed_games"])
    print("\nSpell classes on the full window — and 61% of missed games sit in edge spells:")
    print(by_class.round(4).to_string(index=False))

    bias = censoring_bias_table(spells)
    print("\nWhat ignoring censoring costs, and both wrong answers fail silently:")
    print(bias.round(4).to_string(index=False))
    proper = bias[bias["treatment"] == "proper_censoring"].iloc[0]
    dropped = bias[bias["treatment"] == "drop_censored"].iloc[0]
    print(f"  Dropping censored spells understates P(T >= 26) by "
          f"{proper['p_ge_26'] / dropped['p_ge_26']:.2f}x.")
    if not bool(proper["mean_defined"]):
        print(f"  With proper censoring a = {proper['a']:.3f} < 1, so the fitted "
              f"beta-geometric has NO finite mean.\n  The simulator is unaffected — a "
              f"spell is truncated by the remaining schedule anyway — but E[T] must "
              f"never be quoted from that fit.")

    interior = spells[spells["spell_class"] == "interior"]
    candidates = duration_candidates(interior)
    print(f"\nDuration on the {len(interior):,} interior spells — beta-geometric against "
          f"the geometric null:")
    print(candidates.round(4).to_string(index=False))

    open_fit = left_truncation_check(spells)
    print(f"\nLeft truncation — the offset is FITTED and the identity PREDICTS it, so "
          f"agreement is a check:")
    print(f"  base mu {open_fit['mu']:.4f} (a = {open_fit['a']:.4f}, "
          f"b = {open_fit['b']:.4f}) on {open_fit['n_interior']:,} interior spells")
    print(f"  offset {open_fit['open_shift']:+.4f} on the logit scale gives the "
          f"in-progress mu = {open_fit['mu_open_fitted']:.4f}")
    print(f"  the size-biased Beta(a-1, b) identity predicts "
          f"{open_fit['mu_open_predicted']:.4f} — a gap of "
          f"{open_fit['gap']:+.4f}")
    if not open_fit["identity_defined"]:
        print("  /!\\  a <= 1, so the length-biased density is not normalizable and the "
              "identity does\n       not apply — which is itself the finding: the "
              "renewal assumption has broken down.")

    # ── Gate 0 ────────────────────────────────────────────────────────────────
    from src.eda.availability import with_lags
    from src.features.availability import season_availability

    frame = season_availability(panel, "full")
    lagged = with_lags(frame, seasons, ["minutes_per_game", "gp_share"], max_lag=1)
    population = rotation_population(lagged.dropna(subset=["gp_share_lag1"]))
    single_team = bool(cfg_gp.get("single_team_only", True))
    excluded = len(population & multi_team_seasons(panel)) if single_team else 0
    print(f"\nGate 0 — the process class at its ceiling, on {len(population):,} "
          f"established rotation player-seasons\n  less {excluded:,} multi-team ones: "
          f"`gp` is per player-season and the process is per (player, team), so a traded "
          f"player\n  reads as having missed half the season twice. Leaving them in moves "
          f"the OBSERVED left tail\n  from 0.111 to 0.237 — enough to hide the overshoot "
          f"this gate exists to detect.")
    gate = gate_zero(panel, population, n_sims, seed, single_team)
    print(gate.pivot_table(index="arm", columns="metric",
                           values=["observed", "simulated"]).round(4).to_string())
    verdict = gate_zero_verdict(gate)
    print("\nGate 0 verdict — on the LEFT TAIL, because every arm reproduces the mean:")
    print(verdict.round(4).to_string(index=False))
    if not bool(verdict.loc[verdict["arm"] == "tenure_decomposition",
                            "passes"].iloc[0]):
        print("  /!\\  The tenure decomposition FAILS Gate 0. Stop here — the process "
              "class is wrong,\n       and no covariate block fixes a process class.")
    if bool(verdict.loc[verdict["arm"] == "full_window_chain", "passes"].iloc[0]):
        print("  /!\\  The plain full-window chain PASSES, which contradicts the frame "
              "change in\n       docs/games-played-plan.md. Re-read the gate before "
              "trusting either result.")

    # ── the hazard-by-streak curve ────────────────────────────────────────────
    curve = hazard_by_streak(panel, "appearance")
    print("\nOnset hazard by played streak — the shape a frailty generates for free:")
    print(curve.round(4).to_string(index=False))

    agreement = status_agreement(panel)
    print("\nThe structural tenure proxy against the box-score status (2006-07 on):")
    print(agreement[["statistic", "value"]].round(4).to_string(index=False))
    print("  The interesting cell is the DISAGREEMENT in one direction: games missed "
          "outside the\n  appearance window while still ROSTERED. That is season-ending "
          "and preseason injury —\n  the highest-value population in the head — and the "
          "structural proxy files it under\n  'not on the team'. It is what the "
          "three-state arm exists to reach.")

    spells_out = pd.concat([
        by_class.assign(analysis="spell_class"),
        bias.assign(analysis="censoring_bias"),
        candidates.assign(analysis="duration_candidates"),
        curve.assign(analysis="hazard_by_streak"),
        pd.DataFrame([{**open_fit, "analysis": "left_truncation",
                       "treatment": "free_offset"}]),
        agreement,
    ], ignore_index=True)

    artifacts = {
        "gate": (gate, out_dir / "stan_games_played_gate.csv"),
        "spells": (spells_out, out_dir / "stan_games_played_spells.csv"),
        "collapse": (collapse, out_dir / "stan_games_played_collapse.csv"),
    }
    paths = {}
    for name, (data, dest) in artifacts.items():
        data.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(data):,} {name} rows → {dest}")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
