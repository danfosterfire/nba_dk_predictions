"""Two attacks on the availability head's boundary defect, crossed so they can be told apart.

`make availability-absence` → `availability_absence.csv`,
`availability_absence_interaction.csv`, `availability_absence_rolling.csv`.

## The defect

On validation the shipped head puts **5.66%** of player-seasons below ten games against an
observed **8.15%**, and **4.30%** at a full schedule against an observed **2.72%** — both
outside the 95% posterior-predictive band. `docs/availability-window-plan.md` §7 established
the mechanism and it is arithmetic rather than a hypothesis: under `a = mu(1-rho)/rho` and
`b = (1-mu)(1-rho)/rho` the frailty's **shape** and its **variance** are the same parameter,
so no setting of `rho` can put mass at one end without taking it from the other. §7's five
frailty arms varied the mixing distribution and the best of them halved the miss;
`logitnormal`, the arm that removes the divergence outright, failed the *other* way.

## The two axes, and why one ladder

They are different explanations of the same miss and the cheap one has never been tried,
so running them separately would leave the expensive one un-priced against it.

**Axis 1 — the covariate block.** §11b measured that a "missed game" is four processes, not
one: ~31% interior DNP (a healthy scratch), ~25% interior inactive (in-season injury), ~24%
a still-rostered edge block (preseason or season-ending injury), ~21% not-rostered (roster
churn). The mix *inverts* across role — a fringe player's interior absence is ~2:1 a coach's
decision and a star's is ~2:1 an injury. The head sees none of it: `FEATURE_COLS` carries
how MUCH he missed (`gp_share_lag1`, `trailing_missed_lag1`, `n_spells_lag1`,
`longest_spell_lag1`) and nothing about WHY. `availability.ABSENCE_MIX_COLS` is that block,
as **shares** of last season's missed games rather than counts — the counts sum to
`missed_games`, which is `team_games - gp`, which is `gp_share_lag1` on a different scale.

**Axis 2 — a compound counting process.** The head is a scalar latent rate pushed through a
binomial count, so the entire shape of the `gp` distribution has to be manufactured by one
number's mixing distribution. The two boundaries are not the same event: `P(gp = n)` is
"zero onsets all year" and `P(gp < 10)` is "one absorbing event, early". This arm gives them
different parameters:

    missed = sum_{j=1..K} L_j, truncated at n     gp = n - missed
    K ~ BetaBinom(n, h_i, rho_h)                  onsets; the covariates ride on h
    L ~ lambda*delta_1 + (1 - lambda)*BetaGeom(mu_d, kappa_d)

**The nesting is exact and it is the design constraint.** At `lambda = 1` every spell is one
game, `missed ~ BetaBinom(n, h, rho_h)`, and with `h = 1 - mu` that IS the incumbent by the
beta-binomial's `y -> n - y` symmetry. So `lambda` is a *bounded* parameter on [0, 1] rather
than a logit — the same device `BetaRectangularFrailty` uses for `theta`, for the reason
§7b gives: a nesting that has to be taken as a limit is a statement about floating point.
`assert_nests` holds it to the same 1e-8 as every other arm.

## The ladder

Five arms, crossed at the shipped window (`three_point_era`) with the season term and the
dispersion held at the shipped arm, exactly as §7 did:

| arm | likelihood | features |
|---|---|---|
| `betabinom` | the reference | `FEATURE_COLS` |
| `betabinom__absence_mix` | the reference | `+ ABSENCE_MIX_COLS` |
| `compound` | the counting process | `FEATURE_COLS` |
| `compound__absence_mix` | the counting process | `+ ABSENCE_MIX_COLS` |
| `mixture` | §7's selected arm, **what ships** | `FEATURE_COLS` (context row) |

Five is the whole grid. §7 refused to cross the likelihood axis with the full sweep because
19 arms was already the multiplicity problem §4b exists to answer.

## The second round — the block against the head that SHIPS (§14)

**§12 crossed the block against `betabinom`, and `betabinom` is not the head.** `mixture` is
(§7i), so the block's two margins were measured against a model nobody runs. This round is
`docs/potential-to-dos.md` item 8, and the question is redundancy rather than size:

| arm | what it answers |
|---|---|
| `mixture` | the incumbent, and the reference every margin is quoted against |
| `mixture__absence_mix` | does the block help the MEAN under a two-component head |
| `mixture__absence_mix_pi` | does knowing WHY he missed say WHO gets a disrupted season |
| `betabinom`, `betabinom__absence_mix` | the other level of the likelihood axis, so the
  **interaction** is a paired bootstrap rather than a subtraction across two artifacts |

`mixture` closes the boundary with a covariate-driven weight on a disrupted-season component,
so it already says *who* is at risk; the block says *why he missed last season*. If those are
the same information by two routes, the block's −0.0610 CRPS collapses here. The two covariate
lists are separate arms because they are separate questions — `pi_features` on `FrailtyGLM`
is what lets them move independently, and §7's note is that `PI_COLS` is deliberately short.

**`lambda` is profiled beside the ladder** (`availability_absence_lambda.csv`), and that is
not decoration. A free fit that stops at `lambda = 1` is either the MLE or an optimizer that
could not leave the corner it started in, and the two are indistinguishable in an arm table.
Pinning `lambda` and refitting everything around it separates them — and carries `rho`, which
is where §11a's `C + rho*(n - C)` identification argument becomes a number rather than a
derivation. The last profile row pins the duration at §11b's *measured* spell shape instead
of fitting it, because `lambda` alone does not parameterize spell length: the arm can send
`mu_d` to 1 and make the free branch degenerate at one game.

**The 2x2 is reported with its interaction explicit** (`availability_absence_interaction.csv`),
because "did it get better" is not the question. Each axis alone, each axis given the other,
and the difference between those two — which is what says whether the two are additive or
redundant. If the feature block closes most of the gap on its own, the boundary defect was
partly a missing-covariate problem wearing a functional-form costume.

## The rules that do not move

- The selector is `boundary_tail_error`; `body_error` sits beside it and is **never**
  averaged in. §4 found an arm that buys both boundaries by wrecking the middle, and the
  arms with the best boundary coverage there were the worst models.
- **D1** applies: an arm ships only if it improves tail calibration AND its paired-bootstrap
  CRPS interval excludes a material loss.
- A fresh winner is treated as failing until it replicates on the **rolling-origin** harness
  (§4b, §10e). Validation has reversed four arms that won on a single reading.
- Selection reads validation only. Nothing here calls `final_split`.
- Point MLE in numpy throughout. An arm that wins here earns a Stan port in a later session,
  exactly as `mixture` did; it does not ship from here.

Usage:
    python -m src.models.availability_absence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp

from src.models.availability import (ABSENCE_MIX_COLS, FEATURE_COLS, MIN_MIX_COVERAGE,
                                     MISSED_MIX_KINDS, attach_absence_mix, build_design,
                                     season_start_dates, _sigmoid)
from src.features.availability import build_panel, season_availability
from src.models.availability_window import (BOOTSTRAP_REPS, LIKELIHOOD_LOOKBACK,
                                            LIKELIHOOD_WINDOW, PI_COLS, WINDOWS, FrailtyGLM,
                                            BetaBinomFrailty, MixtureFrailty,
                                            MIN_ROLE_ROWS, _ab_row, _bb_dlogpmf_dmu,
                                            _origin_scores, _tail_errors, _tail_parts,
                                            assert_nests, paired_bootstrap, restrict_window,
                                            score_arm)
from src.models.games_played import (KAPPA_MAX, KAPPA_MIN, beta_geometric_logpmf,
                                     beta_geometric_logsf)
from src.models.held_out import selection_split
from src.models.season_terms import season_start_year
from src.eda.season_effects import ROLE_LABELS

# The interior absence-spell shape `make availability-exchangeability` already fitted on
# this head's own population — `availability_clustering.csv`, `spell_shape`, population
# `all`, mean spell 3.0660 games. It is the SEED for the compound's duration block, not a
# constraint on it: the arm fits `mu_d` and `kappa_d` freely from here and the table reports
# how far they move, because the spells this arm's `L` describes are the ones the head's
# `missed_games` is built from — edge blocks included — and those are not the interior
# spells the seed was fitted on.
SPELL_MU_SEED = 0.487148
SPELL_KAPPA_SEED = 3.870291

# Bounds on the duration block. `mu_d` is P(T = 1) and lives on a logit; `kappa_d` is a
# concentration on a log scale, floored and capped at `games_played`'s own guard rails so
# the two parameterizations cannot drift apart.
MU_D_BOUNDS = (-8.0, 8.0)
LOG_KAPPA_BOUNDS = (float(np.log(1e-2)), float(np.log(1e4)))

#: The four crossed arms plus §7's selected arm as a context row. `mixture` is what ships,
#: so every column in the table is readable against it without a second file.
LADDER_ARMS: tuple[str, ...] = ("betabinom", "betabinom__absence_mix",
                                "compound", "compound__absence_mix", "mixture")

#: The reference row every margin is quoted against — the same one §7c uses.
ABSENCE_REFERENCE = "betabinom"

# ── §14: the same block, crossed against the arm that actually ships ──────────
#
# §12 crossed the block against `betabinom` and the head that ships is `mixture`, so the
# block's two margins were measured against a head nobody runs. This round is the arm
# `docs/potential-to-dos.md` item 8 names, and the question is redundancy: `mixture` closes
# the boundary by giving the disrupted season its own component with a covariate-driven
# weight, so it already says WHO is at risk; the block says WHY he missed last season. Those
# are plausibly the same information reaching the same place by two routes.
#
# **The two covariate blocks are separate arms, and that is the round's substance.** The mean
# function and the disruption weight are different questions, and §7's note is that `PI_COLS`
# is deliberately short — nineteen more unpenalized parameters on 4,027 rows would measure
# the `l2` confound rather than the mechanism — so four more columns on `pi` is a decision to
# be made against that rather than a free extension.

#: §14's arms. The two `betabinom` rows are carried because the **interaction** needs both
#: levels of the likelihood axis: the block's effect under `mixture` minus its effect under
#: `betabinom` is what says whether the two are redundant, and it cannot be read from §12's
#: artifact because a paired bootstrap needs the four arms' per-row scores together.
MIXTURE_ARMS: tuple[str, ...] = ("mixture", "mixture__absence_mix",
                                 "mixture__absence_mix_pi",
                                 "betabinom", "betabinom__absence_mix")

#: The head that ships, and therefore what §14's margins are quoted against. D1 is a
#: statement about the arm being replaced, so quoting `betabinom` here would price the
#: candidates against a head that was retired in §7i.
MIXTURE_REFERENCE = "mixture"

#: Which §14 rows are context rather than candidates — the incumbent itself and the two
#: `betabinom` rows §12 already measured and this round only re-fits to pair the bootstrap.
MIXTURE_CONTEXT: tuple[str, ...] = ("mixture", "betabinom", "betabinom__absence_mix")

#: §14's effects. The second and third are the round's real question and are **not** the same
#: arm: whether the block helps the mean under a two-component head, and whether knowing WHY
#: he missed says WHO gets a disrupted season.
MIXTURE_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("absence_mix on beta | mixture", "mixture__absence_mix", "mixture"),
    ("absence_mix on beta+pi | mixture", "mixture__absence_mix_pi", "mixture"),
    ("absence_mix on pi | beta", "mixture__absence_mix_pi", "mixture__absence_mix"),
    ("absence_mix | betabinom", "betabinom__absence_mix", "betabinom"),
)

#: The redundancy test, as one number with an interval. A negative value means the block buys
#: MORE under the mixture than under the single-component head; a positive one the size of
#: §12's main effect means the mixture already had the block's information and the two are
#: two routes to the same correction.
MIXTURE_INTERACTIONS: tuple[tuple[str, str, str, str, str], ...] = (
    ("interaction", "mixture__absence_mix", "mixture",
     "betabinom__absence_mix", "betabinom"),
)

#: Which rounds `run` executes, and the config key that overrides it
#: (`features.availability.absence.rounds`). The two rounds write **disjoint** artifacts, so
#: a partial run cannot overwrite the other's rows — which is what makes this safe where
#: `composition_effects` needed a merge. §12's round costs the compound arms and an eight-row
#: profile; §14's costs four more mixture fits and its own rolling harness.
ROUNDS: tuple[str, ...] = ("crossed", "mixture")

#: `mixture`'s validation `boundary_tail_error` (§7c). The bar axis 1 has to clear on its
#: own for the compound arm to need re-motivating, and the level the round is read against.
MIXTURE_BOUNDARY = 0.0109

# The rolling harness's first origin. `status_coverage` is 0.0 before 2006-07, so the
# absence-mix shares are only defined for target seasons from 2007-08 on; at
# `LIKELIHOOD_LOOKBACK` = 8 the earliest origin whose whole fitting window carries them is
# 2007 + 8. Restricting the ORIGINS rather than dropping the rows is what keeps all five
# arms on identical rows — an arm fitted on a different population is a confound, not a
# comparison, and it is the one thing this harness exists to rule out.
ABSENCE_FIRST_ORIGIN = 2015


# ── The compound counting process ─────────────────────────────────────────────

def _onset_pmf_grid(n: np.ndarray, h: np.ndarray, rho: np.ndarray,
                    k: np.ndarray) -> np.ndarray:
    """`P(K = k)` for `K ~ BetaBinom(n, h, rho)` on a shared grid — the fast path.

    `availability_window._bb_pmf_grid` computes the same thing through
    `scipy.stats.betabinom.pmf`, which evaluates three log-betas per cell and costs ~30 ms
    on the (4,027 x 83) grid this arm needs at *every* objective evaluation. The
    successive-ratio identity

        P(k+1)/P(k) = ((n - k)/(k + 1)) * ((a + k)/(b + n - k - 1))

    turns the grid into a cumulative sum of logs, which is ~10x cheaper and agrees with
    `betabinom.pmf` to floating point. The tests pin the agreement rather than trusting it,
    because a fast path that is subtly wrong would show up as a likelihood that is merely
    slightly worse — indistinguishable from the null the compound arm is testing for.

    **The recursion is anchored by normalizing, not by evaluating `P(K = 0)`.** The closed
    form for that anchor is `betaln(a, n + b) - betaln(a, b)`, and when `rho` approaches its
    floor the shape parameters reach ~1e6, where those two log-betas are ~1e6 apart in
    magnitude and cancel to a number of order 1 — the difference then carries ~1e-10 of
    absolute error, which is enough to fail `assert_nests` at 300 rows. A beta-binomial's
    pmf sums to one over `0..n` exactly, so the anchor is recoverable from the ratios alone
    and the cancellation never happens.
    """
    a, b = _ab_row(h, rho)
    n = np.asarray(n, dtype=float)[:, None]
    kk = np.asarray(k, dtype=float)[None, :]
    a, b = a[:, None], b[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (np.log(np.maximum(n - kk, 1e-300)) - np.log(kk + 1.0)
                 + np.log(a + kk) - np.log(np.maximum(n - kk + b - 1.0, 1e-300)))
    log_pmf = np.zeros((len(a), len(k)))
    if len(k) > 1:
        log_pmf[:, 1:] = np.cumsum(ratio[:, :-1], axis=1)
    # Above the row's own schedule the count is impossible; the recursion's terms there are
    # garbage (a log of a floored zero) and are masked rather than clipped.
    log_pmf = np.where(kk <= n, log_pmf, -np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.nan_to_num(np.exp(log_pmf - logsumexp(log_pmf, axis=1, keepdims=True)),
                             nan=0.0, posinf=0.0, neginf=0.0)


class CompoundCountingFrailty(FrailtyGLM):
    """`missed = sum_{j<=K} L_j` truncated at `n`, with `K` beta-binomial and `L` a spell.

    The head's own generative story, written down. A beta-binomial on `gp` says the season's
    games are exchangeable trials of one latent rate; this says a season contains a
    *number of onsets*, each of which costs a *duration*, and lets the two carry different
    parameters. `P(gp = n)` then becomes "zero onsets" and `P(gp < 10)` becomes "one long
    absorbing spell", which are the two events the boundary defect is about and which a
    single mixing distribution has to trade against each other.

    **`lambda` is the nesting knob and is bounded, not transformed.** At `lambda = 1` every
    spell is exactly one game, `missed` is `BetaBinom(n, h, rho_h)` and — with `h = 1 - mu`
    — the beta-binomial's `y -> n - y` symmetry makes that the incumbent exactly. `extra`
    is `[lambda, logit(mu_d), log(kappa_d)]` and `_extra0` sets `lambda = 1`, so
    `assert_nests` is an equality at a finite, attainable parameter value like every other
    arm on this axis.

    **The duration parameters are row-constant, and that is a cost decision.** With one `L`
    for every row the K-fold convolution powers are a single `(n+1) x (n+1)` table shared by
    the whole design, and each row's pmf is its own `P(K)` vector against that table — two
    cumulative sums and a matrix product. Row-varying durations would make the table
    per-row and put a `(rows x n x n)` object inside the inner loop; §11b is also the
    measured argument against needing them, since the interior spell *shape* moves 15%
    across the whole role range while the *rate* moves 2.12x, and the rate is what `h`
    already carries.

    **Excess mass is piled at `missed = n`, not renormalized.** A player cannot miss more
    than his team plays, so `sum_j L_j > n` is not an impossible event to be conditioned
    away — it is a season that ended early, and it is exactly the `gp = 0` rows the low
    boundary is about. Renormalizing would spread that mass back across the support in
    proportion to draws the schedule already ruled out, which would move probability *out*
    of the tail this arm exists to fill. Piling also makes the pmf sum to one by
    construction rather than by cancellation.
    """

    name = "compound"

    def __init__(self, *args, lambda_fixed: float | None = None,
                 duration_fixed: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        #: Pin `lambda` and let everything else fit around it. `lambda_profile` uses this to
        #: turn "the optimizer stopped on the bound" into a profile likelihood, which is the
        #: only form of that statement that carries information.
        self.lambda_fixed = None if lambda_fixed is None else float(lambda_fixed)
        #: Pin the duration block at §11b's **measured** interior spell shape instead of
        #: fitting it. `lambda` alone does not parameterize spell length — the arm can send
        #: `mu_d` to 1 and make the free branch degenerate at one game — so this is the arm
        #: that is not allowed to evade the constraint, and it is the one that says what the
        #: head costs if the real spell process is imposed on it.
        self.duration_fixed = bool(duration_fixed)
        if self.lambda_fixed is not None:
            self.name = f"compound_lambda{self.lambda_fixed:g}"
            if self.duration_fixed:
                self.name += "_pinned"

    def _extra0(self) -> np.ndarray:
        # [lambda, logit(mu_d), log(kappa_d)] — lambda = 1 is the nesting point, where the
        # duration block is unidentified and its seed value therefore cannot matter.
        return np.array([1.0,
                         float(np.log(SPELL_MU_SEED / (1.0 - SPELL_MU_SEED))),
                         float(np.log(SPELL_KAPPA_SEED))])

    def _extra_bounds(self):
        # `_extra0` stays at the nesting point whatever `lambda` is pinned to, so a profile
        # arm still has to reproduce the incumbent under `assert_nests`.
        lam = ((0.0, 1.0) if self.lambda_fixed is None
               else (self.lambda_fixed, self.lambda_fixed))
        if not self.duration_fixed:
            return [lam, MU_D_BOUNDS, LOG_KAPPA_BOUNDS]
        seed = self._extra0()
        return [lam, (seed[1], seed[1]), (seed[2], seed[2])]

    def _fit_starts(self):
        if self.lambda_fixed is not None:
            start = self._extra0()
            start[0] = self.lambda_fixed
            return [start]
        # The nesting point plus two live spell mixtures, for the reason `_fit_starts`
        # documents one level up: at `lambda = 1` the duration block has no gradient at all
        # — every spell is one game whatever `mu_d` and `kappa_d` say — so a fit started
        # there can sit on the bound, report success, and reproduce the incumbent. That is
        # an optimizer artifact wearing a null's clothes, and it is the failure §7b measured
        # on `finite_mix`.
        starts = [self._extra0()]
        for lam in (0.5, 0.1):
            s = starts[0].copy()
            s[0] = lam
            starts.append(s)
        return starts

    # ── the spell distribution and its convolution powers ─────────────────────
    def _spell_pmf(self, extra: np.ndarray, m_max: int) -> np.ndarray:
        """`P(L = t)` on `t = 0..m_max`, with the beta-geometric's tail censored at m_max.

        A spell longer than the whole schedule is a season-ending block and lands at the
        truncation point however long it "really" is, so the residual `P(T >= m_max)` is
        placed at `m_max` rather than dropped. That keeps `L` a proper pmf, which is what
        makes the convolution table's rows proper too.
        """
        lam = float(np.clip(extra[0], 0.0, 1.0))
        mu_d = float(_sigmoid(np.clip(extra[1], *MU_D_BOUNDS)))
        kappa_d = float(np.clip(np.exp(np.clip(extra[2], *LOG_KAPPA_BOUNDS)),
                                KAPPA_MIN, KAPPA_MAX))
        pmf = np.zeros(m_max + 1)
        pmf[1] = lam
        if lam >= 1.0:
            return pmf
        t = np.arange(1, m_max)
        pmf[1:m_max] += (1.0 - lam) * np.exp(beta_geometric_logpmf(t, mu_d, kappa_d))
        pmf[m_max] += (1.0 - lam) * float(np.exp(beta_geometric_logsf(m_max, mu_d, kappa_d)))
        return pmf

    @staticmethod
    def _convolution_table(spell: np.ndarray) -> np.ndarray:
        """`conv[K, m] = P(sum of K iid spells = m)`, one table for every row.

        `conv[0]` is a point mass at zero — a season with no onsets misses no games — and
        each further row is one discrete convolution truncated back to the grid. The mass
        that falls off the end is not lost: every consumer reads it back as
        `1 - sum_{m < n}`, which is the piled truncation mass.
        """
        size = len(spell)
        conv = np.zeros((size, size))
        conv[0, 0] = 1.0
        for k in range(1, size):
            conv[k] = np.convolve(conv[k - 1], spell)[:size]
        return conv

    def _tables(self, extra: np.ndarray, m_max: int) -> tuple[np.ndarray, np.ndarray]:
        """`(conv, survival)` — the pmf table and `P(sum of K spells >= m)` beside it."""
        conv = self._convolution_table(self._spell_pmf(extra, m_max))
        survival = 1.0 - np.cumsum(conv, axis=1) + conv
        return conv, survival

    def _select(self, conv: np.ndarray, survival: np.ndarray, missed: np.ndarray,
                n: np.ndarray) -> np.ndarray:
        """The `(rows x K)` weights each row's `P(K)` vector is dotted against.

        A row observed at `missed < n` reads the pmf column; a row at the truncation point
        reads the survival column, which is where all the piled mass lives.
        """
        last = conv.shape[1] - 1
        at_bound = missed >= n
        interior = conv[:, np.clip(missed, 0, last)].T
        piled = survival[:, np.clip(n, 0, last)].T
        return np.where(at_bound[:, None], piled, interior)

    def _terms(self, y, n, eta, disp_row, extra, df):
        mu = _sigmoid(eta)
        h = 1.0 - mu
        m_max = int(np.max(n))
        k = np.arange(m_max + 1)
        p_k = _onset_pmf_grid(n, h, disp_row, k)
        conv, survival = self._tables(extra, m_max)
        weights = self._select(conv, survival, np.asarray(n - y, dtype=int),
                               np.asarray(n, dtype=int))

        prob = np.maximum((p_k * weights).sum(axis=1), 1e-300)
        # Only `h` depends on `eta`, so the gradient is the onset count's own score,
        # weighted by the convolution column and pushed through `dh/deta = -mu(1-mu)`.
        dlog_dh = _bb_dlogpmf_dmu(k[None, :], np.asarray(n, dtype=float)[:, None],
                                  h[:, None], disp_row[:, None])
        d_prob = (p_k * dlog_dh * weights).sum(axis=1)
        return np.log(prob), (d_prob / prob) * (-mu * (1.0 - mu))

    def _pmf(self, n, eta, disp_row, extra, k, df):
        m_max = int(len(k) - 1)
        p_k = _onset_pmf_grid(n, 1.0 - _sigmoid(eta), disp_row, np.arange(m_max + 1))
        conv, _ = self._tables(extra, m_max)
        missed = p_k @ conv

        out = np.zeros((len(n), m_max + 1))
        n_int = np.asarray(n, dtype=int)
        for value in np.unique(n_int):
            rows = n_int == value
            block = missed[rows][:, :value]
            # gp = n - missed, so the row reverses; whatever did not fit under the
            # schedule is the piled truncation mass and lands on gp = 0.
            out[np.flatnonzero(rows)[:, None], np.arange(1, value + 1)[None, :]] = \
                block[:, ::-1]
            out[rows, 0] = np.maximum(1.0 - block.sum(axis=1), 0.0)
        return out

    def shape_report(self, df: pd.DataFrame) -> dict:
        lam = float(np.clip(self.extra[0], 0.0, 1.0))
        mu_d = float(_sigmoid(np.clip(self.extra[1], *MU_D_BOUNDS)))
        kappa_d = float(np.clip(np.exp(np.clip(self.extra[2], *LOG_KAPPA_BOUNDS)),
                                KAPPA_MIN, KAPPA_MAX))
        mu = _sigmoid(self._design(df) @ self.beta)
        n = df["team_games"].to_numpy(dtype=float)
        disp_row = self._disp_row(df, self.dispersion)
        # The onset frailty is Beta on `h = 1 - mu`, so its shape parameters are the
        # incumbent's with the labels swapped: divergence at `gp = n` is `a_h < 1`. Reported
        # on the gp scale so the column means the same thing it does in §7c's table.
        a_h, b_h = _ab_row(1.0 - mu, disp_row)
        spell = self._spell_pmf(self.extra, int(n.max()))
        t = np.arange(len(spell))
        return {"diverges_at_one": float((a_h < 1).mean()),
                "diverges_at_zero": float((b_h < 1).mean()),
                "diverges_at_one_weighted": float((a_h < 1).mean()),
                "diverges_at_zero_weighted": float((b_h < 1).mean()),
                "lambda": lam, "lambda_at_bound": bool(lam >= 1.0 - 1e-9),
                "mu_d": mu_d, "kappa_d": kappa_d,
                "mu_d_seed": SPELL_MU_SEED, "kappa_d_seed": SPELL_KAPPA_SEED,
                "mean_spell": float((spell * t).sum()),
                "mean_onsets": float(((1.0 - mu) * n).mean()),
                "rho": ";".join(f"{r:.4f}" for r in self.dispersion)}


# The grid the compound's nesting parameter is profiled over. Anchored at **1.0**, which is
# the incumbent, and reaching 0.0, where every spell is a free beta-geometric and the arm is
# as far from a beta-binomial as it goes.
LAMBDA_GRID: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0)

#: The profile's arms as `(lambda, duration pinned at the measured spell shape)`. The last
#: row is the structurally honest one: no point mass, and the beta-geometric held at what
#: §11b actually measured rather than at whatever the `gp` likelihood prefers.
LAMBDA_ARMS: tuple[tuple[float, bool], ...] = (tuple((lam, False) for lam in LAMBDA_GRID)
                                               + ((0.0, True),))


def _d1(boundary_hi: float, crps_lo: float) -> bool:
    """§8's D1, as a predicate rather than as a reading.

    An arm ships if it **improves tail calibration** — the paired boundary margin's upper
    bound is clear of zero — **and** its CRPS is **non-inferior**, meaning the paired
    interval does not establish a loss. That second half is why `mixture` shipped on a CRPS
    of +0.011 with an interval spanning zero, and it is the guard that stops an arm trading
    real accuracy for tails. Both halves, or the arm does not ship.
    """
    if not np.isfinite(boundary_hi) or not np.isfinite(crps_lo):
        return False
    return bool(boundary_hi < 0.0 and crps_lo <= 0.0)


def _wins_crps_holds_boundary(boundary_lo: float, crps_hi: float) -> bool:
    """§14's bar — **D1 with its two halves swapped**, and the swap is the point.

    D1 was written for a round whose incumbent had a *bad* boundary, so it asks for a
    calibration gain and settles for CRPS non-inferiority. `mixture` already spent the
    boundary gain (0.0201 → 0.0109), so an arm bolted onto it has nothing left to buy there;
    what it has to buy is the CRPS the shipped head gave up. So this asks for the mirror
    image: a **CRPS interval clear of zero on the good side**, and a boundary margin whose
    interval does not establish a loss. `docs/potential-to-dos.md` item 8 states it in exactly
    those words, and reporting it as a column rather than leaving it to a reader is what stops
    the round being read against the wrong bar.

    Both predicates are carried on every row `crossed_ladder` emits, in both rounds, because
    an arm that passes one and fails the other is the interesting case and averaging them away
    would hide it. `lambda_profile` carries `d1_passes` only: its rows are a grid over a pinned
    parameter rather than candidates, and §12d reads both halves of that verdict off the
    intervals directly.
    """
    if not np.isfinite(boundary_lo) or not np.isfinite(crps_hi):
        return False
    return bool(crps_hi < 0.0 and boundary_lo <= 0.0)


def lambda_profile(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
                   arms: tuple[tuple[float, bool], ...] = LAMBDA_ARMS,
                   features: list[str] | None = None, l2: float = 1.0,
                   seed: int = 42) -> pd.DataFrame:
    """The compound's `lambda` pinned across a grid, with everything else refitted.

    **This is what makes a boundary fit a finding instead of a shrug.** A free fit that
    stops at `lambda = 1` is either the MLE or an optimizer that could not leave the corner
    it started in, and the two look identical in an output table. Pinning `lambda` and
    refitting `beta` and the four dispersions around it separates them: if the profile
    log-likelihood is monotone toward 1 the corner is the MLE, and if it is flat the two
    parameters are trading against each other.

    `rho` is carried on every row for that second reason. §11a's identification argument is
    that clustering and frailty enter a `gp` likelihood only through `C + rho*(n - C)`, so
    they are not separately identified — which predicts that lengthening the spells should
    pull `rho` down by however much the compounding put in, leaving the fit where it was.
    That prediction is a *number* here rather than an assertion, and it is checkable against
    a profile the arm cannot game.
    """
    features = list(FEATURE_COLS if features is None else features)
    rows: list[dict] = []
    scores: dict[str, np.ndarray] = {}
    tails: dict[str, dict[str, np.ndarray]] = {}
    y_val, n_val = val["gp"].to_numpy(), val["team_games"].to_numpy()
    for lam, pinned in arms:
        model = CompoundCountingFrailty(l2=l2, features=features, lambda_fixed=float(lam),
                                        duration_fixed=pinned).fit(train)
        assert_nests(model, train)
        row, per_row = score_arm(model.name, model, train, val, features, max_games, seed)
        scores[model.name] = per_row
        tails[model.name] = _tail_parts(model.predict_pmf(val, max_games), y_val, n_val)
        shape = model.shape_report(val)
        row.update({"lambda": float(lam),
                    "duration": "pinned_at_measured" if pinned else "free",
                    "train_loglik": model.train_loglik,
                    "train_loglik_incumbent": model.incumbent_loglik,
                    "mean_spell": shape["mean_spell"], "mu_d": shape["mu_d"],
                    "kappa_d": shape["kappa_d"],
                    "rho_weighted": float(model.rho),
                    "rho_by_role": ";".join(f"{r:.4f}" for r in model.dispersion)})
        rows.append(row)
        print(f"  lambda {lam:>4g} {'pinned' if pinned else 'free  '}  "
              f"trainLL {model.train_loglik:11.3f}  rho {model.rho:.4f}  "
              f"mean spell {shape['mean_spell']:.3f}  CRPS {row['val_crps']:.4f}  "
              f"boundary {row['boundary_tail_error']:.4f}")
    table = pd.DataFrame(rows)
    best = float(table["train_loglik"].max())
    table["loglik_vs_best"] = table["train_loglik"] - best
    table["is_profile_optimum"] = table["train_loglik"] >= best - 1e-9
    # The `rho` the profile is trading against, carried on every row so §11a's
    # `C + rho*(n - C)` prediction can be read off the table rather than recomputed.
    table["rho_vs_nesting"] = table["rho_weighted"] / float(
        table.loc[table["lambda"] == 1.0, "rho_weighted"].iloc[0])

    # **Every profile row gets the same two intervals the ladder's arms get**, against the
    # nesting row — which is the incumbent. The row that imposes the measured spell shape
    # posts the best boundary error anywhere on this head, and §4's standing warning is that
    # the arms with the best boundary coverage are the worst models. A boundary margin
    # quoted bare beside a CRPS margin quoted bare is exactly the reading that mistake is
    # made from, so D1 gets both halves as intervals.
    reference = table.loc[table["lambda"] == 1.0, "arm"].iloc[0]
    # The columns are named for `betabinom` rather than for `compound_lambda1`, and that is
    # the nesting identity rather than a shortcut: at `lambda = 1` this arm IS the incumbent
    # beta-binomial, asserted to 1e-8 by `assert_nests` on every row of the profile. Naming
    # them after the pinned arm would make the same margin read as a comparison against a
    # different model at each grid point.
    margins = _bootstrap_arms(tails, tuple(table["arm"]), reference, seed=seed,
                              suffix="betabinom")
    for name in table["arm"]:
        d, lo, hi = paired_bootstrap(scores[name], scores[reference], seed=seed)
        idx = table.index[table["arm"] == name][0]
        table.loc[idx, ["crps_vs_nesting", "crps_vs_nesting_lo",
                        "crps_vs_nesting_hi"]] = [d, lo, hi]
        for key, value in margins[name].items():
            table.loc[idx, key] = value
    table["d1_passes"] = [_d1(b, c) for b, c in zip(table["boundary_vs_betabinom_hi"],
                                                    table["crps_vs_nesting_lo"])]
    return table


# ── The crossed ladder ────────────────────────────────────────────────────────

def arm_spec(name: str) -> tuple[type, list[str], dict]:
    """`(likelihood class, feature list, constructor kwargs)` for one arm.

    The name is `likelihood[__block]`, and the block token says which covariate lists the
    absence composition joins:

    - `absence_mix` — the mean function `beta` only, which is §12's arm.
    - `absence_mix_pi` — `beta` **and** the mixture weight `pi`, which is §14's. It is
      refused on any other likelihood: `pi_features` reaches `FrailtyGLM`, so `betabinom`
      would accept it and silently ignore it, producing a row that reads as a third
      candidate and is a duplicate of the second.

    The kwargs dict is what keeps the two blocks independent while the ladder stays one call
    site, and it is empty for every arm §12 measured — which is why that round's five rows
    are reproduced bit for bit rather than merely closely.
    """
    likelihood, _, block = name.partition("__")
    cls = {"betabinom": BetaBinomFrailty, "compound": CompoundCountingFrailty,
           "mixture": MixtureFrailty}[likelihood]
    if block not in ("", "absence_mix", "absence_mix_pi"):
        raise ValueError(f"{name}: unknown covariate block {block!r}")
    features = list(FEATURE_COLS) + (list(ABSENCE_MIX_COLS) if block else [])
    kwargs: dict = {}
    if block == "absence_mix_pi":
        if cls is not MixtureFrailty:
            raise ValueError(f"{name}: only `mixture` has a `pi` for the block to ride on")
        kwargs["pi_features"] = list(PI_COLS) + list(ABSENCE_MIX_COLS)
    return cls, features, kwargs


def crossed_ladder(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
                   arms: tuple[str, ...] = LADDER_ARMS, l2: float = 1.0,
                   seed: int = 42, reference: str = ABSENCE_REFERENCE,
                   context: tuple[str, ...] = ("mixture",)) -> tuple[pd.DataFrame, dict, dict]:
    """The 2x2, scored on validation. Returns `(table, per-row CRPS, tail parts)`.

    The window, the season term and the dispersion are held at the shipped arm, so the only
    things that vary are the likelihood and the feature block — which is what makes the
    interaction below readable as an interaction.

    **`reference` is which arm every margin is quoted against, and it is a parameter because
    the two rounds on this axis have different incumbents.** §12 crossed the block against
    `betabinom` and quoted `betabinom`; §14 crosses it against the head that actually ships
    and has to quote `mixture`, because D1 is a statement about the arm being replaced. The
    column suffix follows the reference rather than being hard-coded, so a margin column can
    never name an arm it was not computed against.

    `context` names the arms that are carried for readability but are not candidates —
    `mixture` in §12, where it never trains on the block, and nothing in §14, where it is
    the reference.
    """
    cut = restrict_window(train, WINDOWS[LIKELIHOOD_WINDOW])
    y_val, n_val = val["gp"].to_numpy(), val["team_games"].to_numpy()
    rows: list[dict] = []
    per_row: dict[str, np.ndarray] = {}
    tails: dict[str, dict[str, np.ndarray]] = {}

    for name in arms:
        cls, features, kwargs = arm_spec(name)
        missing = [c for c in features if c not in cut.columns]
        if missing:
            raise ValueError(f"{name} needs {missing}; call `attach_absence_mix` first")
        if cut[features].isna().any().any():
            bad = [c for c in features if cut[c].isna().any()]
            raise ValueError(
                f"{name} has NaN in {bad} on the fitting window. The absence-mix shares "
                f"are masked where `status_coverage` is 0, so a window reaching before "
                f"2006-07 must be excluded rather than fitted.")
        model = cls(l2=l2, features=features, **kwargs).fit(cut)
        gap = assert_nests(model, cut)
        row, scores = score_arm(name, model, cut, val, features, max_games, seed)
        block = name.partition("__")[2]
        row.update({"likelihood": name.partition("__")[0],
                    "absence_mix": bool(block),
                    "absence_mix_on_pi": block == "absence_mix_pi",
                    "n_params": model.n_params, "nesting_loglik_gap": gap,
                    "train_loglik": model.train_loglik,
                    "train_loglik_at_nesting": model.nesting_loglik,
                    "train_loglik_incumbent": model.incumbent_loglik,
                    "selectable": name not in context,
                    "n_starts": model.n_starts,
                    "start_loglik_spread": model.start_spread})
        row.update({f"shape_{k}": v for k, v in model.shape_report(val).items()})
        rows.append(row)
        per_row[name] = scores
        tails[name] = _tail_parts(model.predict_pmf(val, max_games), y_val, n_val)
        print(f"  {name:<24} {model.n_params:>3} params  CRPS {row['val_crps']:.4f}  "
              f"PIT {row['val_pit_ks']:.4f}  boundary {row['boundary_tail_error']:.4f}  "
              f"body {row['body_error']:.4f}  shoulder {row['shoulder_error']:.4f}")

    suffix = _suffix(reference)
    ref_scores = per_row[reference]
    ref_boundary = next(r for r in rows if r["arm"] == reference)["boundary_tail_error"]
    for row in rows:
        d, lo, hi = paired_bootstrap(per_row[row["arm"]], ref_scores, seed=seed)
        row[f"crps_vs_{suffix}"] = d
        row[f"crps_vs_{suffix}_lo"] = lo
        row[f"crps_vs_{suffix}_hi"] = hi
        row[f"beats_{suffix}"] = bool(hi < 0.0)
        # D1, stated as a column rather than left to a reader: tail calibration has to
        # improve AND the CRPS interval has to exclude a material loss. Both halves, or the
        # arm does not ship — which is the rule `docs/availability-window-plan.md` §8 wrote
        # down before the arm it admitted was ported.
        row["improves_boundary"] = bool(row["boundary_tail_error"] < ref_boundary)
        row["beats_mixture_boundary"] = bool(row["boundary_tail_error"] < MIXTURE_BOUNDARY)
    boundary = _bootstrap_arms(tails, arms, reference, seed=seed)
    for row in rows:
        row.update(boundary[row["arm"]])
        row["d1_passes"] = _d1(row.get(f"boundary_vs_{suffix}_hi", np.nan),
                               row[f"crps_vs_{suffix}_lo"])
        # D1's mirror, on every row of both rounds. Against a reference that has already
        # spent the boundary gain there is nothing left for a new arm to buy there, so the
        # bar that matters is a CRPS interval clear of zero with the boundary held — and an
        # arm that passes one predicate and fails the other is the reading, not an anomaly.
        row["wins_crps_holds_boundary"] = _wins_crps_holds_boundary(
            row.get(f"boundary_vs_{suffix}_lo", np.nan), row[f"crps_vs_{suffix}_hi"])
    return pd.DataFrame(rows), per_row, tails


def _suffix(reference: str) -> str:
    """The column suffix a round's margins carry — the reference arm's own name.

    §12 quotes `betabinom` and §14 quotes `mixture`, and both write to the same schema of
    `<metric>_vs_<suffix>` columns. Derived rather than passed so a margin column cannot end
    up naming an arm it was not computed against, which is the one bookkeeping error a
    two-round file makes silently.
    """
    return reference.partition("__")[0]


def _bootstrap_arms(tails: dict, arms: tuple[str, ...], reference: str,
                    reps: int = BOOTSTRAP_REPS, seed: int = 42,
                    suffix: str | None = None) -> dict[str, dict]:
    """Every arm's three regional errors and its margin against the reference, on ONE
    resample of the rows per replicate.

    Sharing the index across arms is what makes the margins paired: `boundary_tail_error`
    is a non-linear statistic — two absolute values of differences of means — so it has to
    be recomputed on the resample rather than averaged from a per-row score, and if each arm
    drew its own rows the between-player variance would swamp the between-arm one.

    `suffix` overrides the column naming for the one caller whose reference row is not an
    arm of a ladder: `lambda_profile`'s reference is `compound_lambda1`, which *is* the
    incumbent's likelihood by the nesting identity, so its columns keep the incumbent's name.
    Everywhere else the suffix is derived from the reference and must be, so that a margin
    column cannot name an arm it was not computed against.
    """
    suffix = suffix or _suffix(reference)
    n_rows = len(tails[reference]["p_full"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_rows, size=(reps, n_rows))
    draws = {a: np.empty((reps, 3)) for a in arms}
    for r in range(reps):
        for a in arms:
            draws[a][r] = _tail_errors(tails[a], idx[r])

    names = ("boundary_tail_error", "body_error", "shoulder_error")
    out: dict[str, dict] = {}
    for a in arms:
        row = {}
        for j, metric in enumerate(names):
            row[f"{metric}_lo"] = float(np.percentile(draws[a][:, j], 2.5))
            row[f"{metric}_hi"] = float(np.percentile(draws[a][:, j], 97.5))
            if a != reference:
                d = draws[a][:, j] - draws[reference][:, j]
                short = metric.replace("_tail_error", "").replace("_error", "")
                row[f"{short}_vs_{suffix}"] = float(d.mean())
                row[f"{short}_vs_{suffix}_lo"] = float(np.percentile(d, 2.5))
                row[f"{short}_vs_{suffix}_hi"] = float(np.percentile(d, 97.5))
                row[f"beats_{suffix}_{short}"] = bool(np.percentile(d, 97.5) < 0.0)
        out[a] = row
    return out


#: The 2x2 read as effects. Each entry is `(label, arm, baseline)` and the effect is
#: `arm - baseline` on every metric — negative is better on all four, since three of them
#: are absolute calibration errors and the fourth is CRPS.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("absence_mix | betabinom", "betabinom__absence_mix", "betabinom"),
    ("absence_mix | compound", "compound__absence_mix", "compound"),
    ("compound | plain", "compound", "betabinom"),
    ("compound | absence_mix", "compound__absence_mix", "betabinom__absence_mix"),
)

#: The interaction each round reports, as `(label, arm, base, arm at the other level, base
#: at the other level)`. The effect is `(arm - base) - (other arm - other base)`, so a
#: near-zero value means the block buys the same thing whatever likelihood it is bolted to.
INTERACTIONS: tuple[tuple[str, str, str, str, str], ...] = (
    ("interaction", "compound__absence_mix", "compound",
     "betabinom__absence_mix", "betabinom"),
)


def interaction_table(per_row: dict, tails: dict, reps: int = BOOTSTRAP_REPS,
                      seed: int = 42, contrasts: tuple = CONTRASTS,
                      interactions: tuple = INTERACTIONS) -> pd.DataFrame:
    """The 2x2 with the interaction made explicit, on the selector and on CRPS.

    The question this round asks is not "did it get better" — it is **how much each axis
    buys alone, and whether the two are additive or redundant**. So each main effect is
    reported at both levels of the other axis, and the interaction

        [B(compound + mix) - B(compound)] - [B(betabinom + mix) - B(betabinom)]

    is reported as its own row with its own interval. A near-zero interaction means the two
    axes are additive and are fixing different things; a negative one means they compound;
    a positive one the size of the main effect means they are two routes to the same
    correction and only one of them is needed.

    Every effect is resampled on the **same** row indices within a replicate, for the reason
    `_bootstrap_arms` gives, and the four regional errors are recomputed on the resample
    rather than averaged, because they are not means of per-row scores.

    §14 reuses this with its own `contrasts` and `interactions`, where the crossed axis is
    the *likelihood the block is bolted to* rather than the counting process — same
    arithmetic, and the interaction answers whether the block's margin survives the
    two-component head.
    """
    arms = tuple(dict.fromkeys(
        [a for _, arm, base in contrasts for a in (arm, base)]
        + [a for spec in interactions for a in spec[1:]]))
    n_rows = len(tails[arms[0]]["p_full"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_rows, size=(reps, n_rows))
    regional = {a: np.empty((reps, 3)) for a in arms}
    crps_draws = {a: np.empty(reps) for a in arms}
    for r in range(reps):
        for a in arms:
            regional[a][r] = _tail_errors(tails[a], idx[r])
            crps_draws[a][r] = per_row[a][idx[r]].mean()

    metrics = (("boundary_tail_error", 0), ("body_error", 1), ("shoulder_error", 2),
               ("val_crps", None))

    def draws(arm: str, column: int | None) -> np.ndarray:
        return crps_draws[arm] if column is None else regional[arm][:, column]

    def point(arm: str, column: int | None) -> float:
        return float(per_row[arm].mean()) if column is None else float(
            _tail_errors(tails[arm], np.arange(n_rows))[column])

    rows: list[dict] = []
    for metric, column in metrics:
        specs = ([(label, arm, base, None, None) for label, arm, base in contrasts]
                 + list(interactions))
        for label, arm, base, other_arm, other_base in specs:
            if other_arm is not None:
                d = ((draws(arm, column) - draws(base, column))
                     - (draws(other_arm, column) - draws(other_base, column)))
                value = ((point(arm, column) - point(base, column))
                         - (point(other_arm, column) - point(other_base, column)))
                arm = base = None
            else:
                d = draws(arm, column) - draws(base, column)
                value = point(arm, column) - point(base, column)
            rows.append({
                "metric": metric, "effect": label,
                "arm": arm or "", "baseline": base or "",
                "arm_value": point(arm, column) if arm else np.nan,
                "baseline_value": point(base, column) if base else np.nan,
                "delta": value,
                "delta_lo": float(np.percentile(d, 2.5)),
                "delta_hi": float(np.percentile(d, 97.5)),
                "clears_zero": bool(np.percentile(d, 97.5) < 0.0),
            })
    return pd.DataFrame(rows)


# ── Rolling-origin confirmation ───────────────────────────────────────────────

def absence_rolling(train: pd.DataFrame, max_games: int,
                    arms: tuple[str, ...] = LADDER_ARMS, l2: float = 1.0,
                    lookback: int = LIKELIHOOD_LOOKBACK,
                    first_origin: int = ABSENCE_FIRST_ORIGIN,
                    seed: int = 42, reference: str = ABSENCE_REFERENCE) -> pd.DataFrame:
    """The 2x2 on §4b's harness — fitting half only, all arms on identical rows.

    §10e is the standing rule that a fresh winner is treated as failing until it replicates,
    and validation has reversed four arms that won on a single reading. This is the second
    reading, and it costs no validation rows: an origin walks across the fitting half, each
    arm fits on the `lookback` seasons before it and scores the origin season itself.

    The origins start later than §7's do, and the reason is the block rather than a choice:
    the absence-mix shares do not exist before the 2006-07 box-score backfill, so an origin
    whose fitting window reaches behind it would have to drop rows for two of the five arms.
    Restricting the origins keeps every arm on the same population, which is the only way
    the contrast stays a contrast.
    """
    years = season_start_year(train)
    origins = [int(y) for y in np.unique(years) if y >= first_origin]
    per_arm: dict[str, dict[str, list]] = {}

    for origin in origins:
        score = train[years == origin]
        fit_rows = train[(years < origin) & (years >= origin - lookback)]
        if score.empty or len(fit_rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
            continue
        for name in arms:
            cls, features, kwargs = arm_spec(name)
            if fit_rows[features].isna().any().any() or score[features].isna().any().any():
                raise ValueError(
                    f"origin {origin} has NaN absence-mix columns; `first_origin` must be "
                    f"at least {2007 + lookback} at lookback {lookback}")
            model = cls(l2=l2, features=features, **kwargs).fit(fit_rows)
            assert_nests(model, fit_rows)
            scored = _origin_scores(model, score, max_games)
            scored["origin"] = np.full(len(score), origin)
            slot = per_arm.setdefault(name, {})
            for key, values in scored.items():
                slot.setdefault(key, []).append(values)
        print(f"  origin {origin}: {len(fit_rows):,} fit / {len(score):,} scored")

    pooled = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in per_arm.items()}
    parts = {name: _rolling_parts(d) for name, d in pooled.items()}
    ref_scores = pooled[reference]["crps"]
    suffix = _suffix(reference)

    rows = []
    grid = np.linspace(0, 1, 101)
    for name, d in pooled.items():
        y, n, org = d["y"], d["n"], d["origin"]
        mean, lo, hi = paired_bootstrap(d["crps"], ref_scores, seed=seed)
        u = d["pit"]
        idx = np.arange(len(y))
        boundary, body, shoulder = _tail_errors(parts[name], idx)
        block = name.partition("__")[2]
        rows.append({
            "arm": name, "likelihood": name.partition("__")[0],
            "absence_mix": bool(block), "absence_mix_on_pi": block == "absence_mix_pi",
            "n_origins": int(len(np.unique(org))), "n_scored": int(len(y)),
            "crps": float(d["crps"].mean()), f"crps_vs_{suffix}": mean,
            f"crps_vs_{suffix}_lo": lo, f"crps_vs_{suffix}_hi": hi,
            "origins_won": sum(1 for o in np.unique(org)
                               if d["crps"][org == o].mean() < ref_scores[org == o].mean()),
            "pit_ks": float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u)
                                          - grid))),
            "err_below_10": float(d["p_below_10"].mean() - (y < 10).mean()),
            "err_full_schedule": float(d["p_full"].mean() - (y == n).mean()),
            "boundary_tail_error": boundary, "body_error": body,
            "shoulder_error": shoulder,
        })
    margins = _bootstrap_arms(parts, tuple(pooled), reference, seed=seed)
    for row in rows:
        row.update(margins[row["arm"]])
    return pd.DataFrame(rows).sort_values("boundary_tail_error").reset_index(drop=True)


def _rolling_parts(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`_tail_parts`, rebuilt from what `_origin_scores` pooled rather than from a pmf."""
    missed = d["n"].astype(int) - d["y"].astype(int)
    out = {"p_full": d["p_full"], "o_full": (d["y"] == d["n"]).astype(float),
           "p_low_shoulder": d["p_band_low_shoulder"],
           "o_low_shoulder": ((d["y"] >= 1) & (d["y"] < 10)).astype(float),
           "p_high_shoulder": d["p_band_high_shoulder"],
           "o_high_shoulder": ((missed >= 1) & (missed < 12)).astype(float)}
    for threshold in (10, 41, 60):
        out[f"p_{threshold}"] = d[f"p_below_{threshold}"]
        out[f"o_{threshold}"] = (d["y"] < threshold).astype(float)
    return out


# ── The block's own diagnostics ───────────────────────────────────────────────

def block_diagnostics(frame: pd.DataFrame, fitting: pd.DataFrame,
                      seasons: list[str]) -> pd.DataFrame:
    """What the covariate block is made of, as an artifact rather than a printout.

    Three things the round asserts rather than assumes, and none of them survives being
    left in a log line: how much of a season's missed games the four named kinds actually
    account for, how many fitting rows the status-coverage mask removes at the shipped
    window, and how often the four shares are all zero on a row that *did* miss games —
    which is the only case where the `missed_games == 0` convention could be confused with
    a real composition.
    """
    covered = frame[frame["status_coverage"].to_numpy(dtype=float) >= MIN_MIX_COVERAGE]
    with_absence = covered[covered["missed_games"] > 0]
    kinds = [f"missed_{k}" for k in MISSED_MIX_KINDS]
    named = float(with_absence[kinds].to_numpy(dtype=float).sum())
    total = float(with_absence["missed_games"].sum())
    all_zero = float((with_absence[kinds].to_numpy(dtype=float).sum(axis=1) == 0).mean())

    rows = [
        {"statistic": "kinds_share_of_missed", "value": named / total},
        {"statistic": "share_no_absences", "value": float((covered["missed_games"] == 0).mean())},
        {"statistic": "share_all_kinds_zero_with_absences", "value": all_zero},
        {"statistic": "n_fitting_rows", "value": float(len(fitting))},
        {"statistic": "n_fitting_rows_masked",
         "value": float(fitting[ABSENCE_MIX_COLS].isna().any(axis=1).sum())},
        # As a share too, because a count of zero is not a quotable figure — "0" appears in
        # every doc in the repo, so a presence check on it would protect nothing.
        {"statistic": "share_fitting_rows_masked",
         "value": float(fitting[ABSENCE_MIX_COLS].isna().any(axis=1).mean())},
        {"statistic": "first_covered_season_start_year",
         "value": float(frame.loc[frame["status_coverage"] >= MIN_MIX_COVERAGE,
                                  "season_start_year"].min())},
    ]
    for kind, col in zip(MISSED_MIX_KINDS, ABSENCE_MIX_COLS):
        rows.append({"statistic": f"mean_{kind}_share",
                     "value": float(fitting[col].mean())})
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def load_design(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    """`(train, validation, max_games, season frame)` with `ABSENCE_MIX_COLS` attached.

    The held-out split is never materialized: `selection_split` hands back the two frames
    the ladder is allowed to see and nothing else. The un-lagged season frame comes back
    too, because `block_diagnostics` reports on the block's *source* rather than on the
    design it becomes.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    seasons = cfg["data"]["seasons"]
    panel = build_panel(seasons, raw_dir)
    frame = season_availability(panel, "full")
    design = build_design(frame, seasons, raw_dir, season_start_dates(panel))
    design = attach_absence_mix(design, frame, seasons)
    train, val = selection_split(design)
    return train, val, int(design["team_games"].max()), frame


def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_av = cfg.get("features", {}).get("availability", {})
    seed = int(cfg_av.get("seed", 42))
    l2 = float(cfg_av.get("glm_l2", 1.0))
    rounds = tuple(cfg_av.get("absence", {}).get("rounds", ROUNDS))
    unknown = [r for r in rounds if r not in ROUNDS]
    if unknown:
        raise ValueError(f"unknown absence round(s) {unknown}; expected {list(ROUNDS)}")

    train, val, max_games, frame = load_design(cfg)
    cut = restrict_window(train, WINDOWS[LIKELIHOOD_WINDOW])
    masked = int(cut[ABSENCE_MIX_COLS].isna().any(axis=1).sum())
    print(f"Absence composition x compound counting process: {len(train):,} train / "
          f"{len(val):,} validation ({', '.join(sorted(val['season'].unique()))})")
    print("  The held-out split is LOCKED and is never materialized here.")
    print(f"  Shipped window ({LIKELIHOOD_WINDOW}): {len(cut):,} fitting rows, "
          f"{masked} dropped by the status-coverage mask.")
    print(f"  Rounds: {', '.join(rounds)}")

    written: dict[str, Path] = {}

    block = block_diagnostics(frame, cut, cfg["data"]["seasons"])
    block_dest = out_dir / "availability_absence_block.csv"
    block.to_csv(block_dest, index=False)
    print(f"  Wrote {len(block):,} block diagnostics → {block_dest}")
    written["availability_absence_block"] = block_dest

    if "crossed" in rounds:
        table, per_row, tails = crossed_ladder(train, val, max_games, l2=l2, seed=seed)
        dest = out_dir / "availability_absence.csv"
        table.to_csv(dest, index=False)
        print(f"\nWrote {len(table):,} arms → {dest}")
        written["availability_absence"] = dest

        print("\nThe 2x2 as effects, with the interaction explicit:")
        effects = interaction_table(per_row, tails, seed=seed)
        for _, r in effects[effects["metric"] == "boundary_tail_error"].iterrows():
            print(f"  boundary  {r['effect']:<24} {r['delta']:+.4f} "
                  f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]")
        eff_dest = out_dir / "availability_absence_interaction.csv"
        effects.to_csv(eff_dest, index=False)
        print(f"Wrote {len(effects):,} effect rows → {eff_dest}")
        written["availability_absence_interaction"] = eff_dest

        print("\nThe compound's nesting parameter, PROFILED — everything else refitted at "
              "each\n  pinned lambda, so a fit that stops at the corner can be told from a "
              "corner\n  that is the MLE. `rho` is carried because §11a predicts it absorbs "
              "the trade:")
        profile = lambda_profile(cut, val, max_games, l2=l2, seed=seed)
        prof_dest = out_dir / "availability_absence_lambda.csv"
        profile.to_csv(prof_dest, index=False)
        print(f"Wrote {len(profile):,} profile rows → {prof_dest}")
        written["availability_absence_lambda"] = prof_dest

        print(f"\nRolling-origin confirmation (lookback {LIKELIHOOD_LOOKBACK}, origins from "
              f"{ABSENCE_FIRST_ORIGIN}, fitting half only):")
        rolling = absence_rolling(train, max_games, l2=l2, seed=seed)
        roll_dest = out_dir / "availability_absence_rolling.csv"
        rolling.to_csv(roll_dest, index=False)
        print(f"\nWrote {len(rolling):,} arms × {int(rolling['n_scored'].max()):,} "
              f"scored rows → {roll_dest}")
        written["availability_absence_rolling"] = roll_dest

    if "mixture" in rounds:
        print(f"\n§14 — the same block crossed against the arm that SHIPS. Margins are "
              f"quoted\n  against `{MIXTURE_REFERENCE}`, and the block rides on `beta` alone "
              f"in one arm and on\n  `beta` AND `pi` in the other, because the mean function "
              f"and the disruption weight\n  are different questions:")
        m_table, m_per_row, m_tails = crossed_ladder(
            train, val, max_games, arms=MIXTURE_ARMS, l2=l2, seed=seed,
            reference=MIXTURE_REFERENCE, context=MIXTURE_CONTEXT)
        m_dest = out_dir / "availability_absence_mixture.csv"
        m_table.to_csv(m_dest, index=False)
        print(f"\nWrote {len(m_table):,} arms → {m_dest}")
        written["availability_absence_mixture"] = m_dest

        print("\nThe effects, with the redundancy interaction explicit — the block's margin\n"
              "  under `mixture` minus its margin under `betabinom`:")
        m_effects = interaction_table(m_per_row, m_tails, seed=seed,
                                     contrasts=MIXTURE_CONTRASTS,
                                     interactions=MIXTURE_INTERACTIONS)
        for metric in ("boundary_tail_error", "val_crps"):
            for _, r in m_effects[m_effects["metric"] == metric].iterrows():
                print(f"  {metric:<19} {r['effect']:<34} {r['delta']:+.5f} "
                      f"[{r['delta_lo']:+.5f}, {r['delta_hi']:+.5f}]")
        m_eff_dest = out_dir / "availability_absence_mixture_interaction.csv"
        m_effects.to_csv(m_eff_dest, index=False)
        print(f"Wrote {len(m_effects):,} effect rows → {m_eff_dest}")
        written["availability_absence_mixture_interaction"] = m_eff_dest

        print(f"\nRolling-origin confirmation of §14 (lookback {LIKELIHOOD_LOOKBACK}, "
              f"origins from {ABSENCE_FIRST_ORIGIN}, fitting half only). §10e: a fresh "
              f"winner\n  fails until it replicates, and §12's CRPS win did not.")
        m_rolling = absence_rolling(train, max_games, arms=MIXTURE_ARMS, l2=l2, seed=seed,
                                    reference=MIXTURE_REFERENCE)
        m_roll_dest = out_dir / "availability_absence_mixture_rolling.csv"
        m_rolling.to_csv(m_roll_dest, index=False)
        print(f"\nWrote {len(m_rolling):,} arms × {int(m_rolling['n_scored'].max()):,} "
              f"scored rows → {m_roll_dest}")
        written["availability_absence_mixture_rolling"] = m_roll_dest

    return written


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
