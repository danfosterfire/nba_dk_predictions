"""Fitting-window x dispersion ladder for the MARGINAL minutes head, and the era series
rebuilt through each head's own design rows.

`make minutes-window`. Five artifacts in `outputs/predictions/`:
`minutes_window_era.csv`, `minutes_window_break.csv`, `minutes_window.csv`,
`minutes_window_rolling.csv`, `minutes_window_stake.csv`.

## Why this exists

`docs/availability-window-plan.md` §6 measured the same defect shape on the minutes head
that §2 measured on availability — a cross-player dispersion that contracted and a
workhorse tail that all but vanished — and §9 item 1 named laddering it the largest open
stake in that line of work, because it can revise a **shipped** decision rather than only
add one. `make minutes-unification` ships the marginal head *solely* for its season-level
spread (predictive sd 302.75 minutes against the composition's 64.65). If that spread is an
average over a window whose dispersion contracted, a short-window refit should narrow the
predictive.

§6 also wrote down its own caveat, and honouring it is step one here. Those series were
built on a **rotation filter** (`gp >= 20`, `mpg >= 10`) rather than on either head's own
row filter, and the caveat's own reasoning splits the two claims: a fall of 10.4x is far too
large for a population definition to flip, a contraction of 15.2% is not. So `era_series`
rebuilds both through `stan_minutes.build_design` and
`stan_composition.composition_frame` before anything is fitted, and reports the rotation
population beside them so §6's figures reconcile rather than merely being replaced.

## What the ladder is, and what it is not

It is a **specification ladder on the point MLE**, exactly as `availability_window` is, for
the same reason: it runs in seconds and needs no CmdStan, so an arm can be rejected before
anyone spends sampler time. `PointMinutes` is the beta-binomial the Stan head fits, at its
posterior mode rather than integrated over — and the check that it is the right reference is
that `full__shared` at the head's selected variant reproduces the shipped head's own
season-unit figures, which `run` prints beside the artifact's.

Selection reads **validation and nothing else**. `selection_split` never materializes the
held-out rows, and every window is a restriction of the *training* half only. The
rolling-origin harness that confirms it walks the fitting half and touches neither.

## The metric that decides is not CRPS alone

The availability ladder's defect was tail coverage, so its metric set grew tails. This
head's defect — the reason it ships at all — is the **season-level predictive spread**, so
`score_arm` carries `predictive_sd` and interval coverage beside CRPS and PIT, and the stake
block scores that spread directly against the thing that consumes it.

## The stake, measured rather than argued

`injection_restake` re-runs `minutes_unification`'s injected per-(player, season) effect
against each window's marginal head as the reference. It loads the composition's persisted
posterior and never refits it: the composition's own draws do not depend on which marginal
arm they are compared to, so the sigma grid is simulated once and each arm is a different
reference in the same paired bootstrap.

**One structural fact frames that whole block, and it is worth stating before any number is
read.** `sim.minutes.player_season_sigma` is selected by the composition's *own* CRPS
optimum on training rows (`minutes_unification.estimate_sigma_on_train`) — the marginal head
appears nowhere in that estimator. So a narrower marginal predictive cannot move the shipped
sigma under the rule that chose it; what it can move is the **tie boundary**, i.e. how much
injected spread the composition needs before this repo can no longer tell the two heads
apart. Both are reported, and they are different questions.

Usage:
    python -m src.models.minutes_window
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import digamma
from sklearn.preprocessing import StandardScaler

from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.models.availability import (RHO_MAX, RHO_MIN, _ab, _neg_loglik, _sigmoid,
                                     fit_dispersion)
from src.models.availability_window import restrict_window
from src.models.held_out import selection_split
from src.models.minutes_unification import (UNIT_KEYS, paired_bootstrap, realized_totals,
                                            rehydrate_composition, season_totals,
                                            shipped_sigma, unit_codes, verdict)
from src.models.posteriors import load, posteriors_dir, require_window
from src.models.season_terms import season_start_year
from src.models.stan_composition import (GROUP_KEYS, PILOT_FIRST_SEASON, composition_frame,
                                         simulate_minutes)
from src.models.stan_minutes import (PREDICTIVE_SAMPLES, SPLINE_KNOTS, FloorMinutes,
                                     beta_shapes, minutes_targets, variants)
from src.models.stan_minutes import build_design as minutes_build_design
from src.models.stan_utils import crps_from_samples, ks_uniform, pit_from_samples

# The feature variant the ladder holds fixed. This is a **window** ladder, not a second
# feature selection, so every arm fits the variant `make stan-minutes` already selected on
# validation; `assert_shipped_variant` refuses to run against a stale constant.
VARIANT = "logit_own_spline"

# First season *start year* kept in the fitting half. `None` is the shipped head's 1997-98.
# `post_break` is where this head's OWN design rows put the break (`break_scan` below, and
# it is 2010-11 rather than the 2014-15 §6 read off the rotation filter); `three_point_era`
# is the window the availability head ships, carried so the two ladders are readable side by
# side; `post_2014` is §6's own claimed break, on the ladder so its premise gets tested
# rather than inherited.
WINDOWS: dict[str, int | None] = {
    "full": None,
    "post_break": 2010,
    "three_point_era": 2012,
    "post_2014": 2014,
}

# One dispersion for every player, or one per prior-MPG role bucket. The availability ladder
# found grading worth -0.017 CRPS and the composition fits a 2.07x spread on its own unit,
# so the question is live here even though this head has never asked it.
RHO_MODES = ("shared", "role")

REFERENCE_ARM = "full__shared"     # the shipped head: whole window, one dispersion

# Central predictive intervals whose realized coverage is reported, matching
# `availability_window.COVERAGE_LEVELS` so the two ladders read the same way.
COVERAGE_LEVELS = (0.5, 0.8, 0.95)

BOOTSTRAP_REPS = 2000

# Fewest training rows a role bucket needs before it gets its own dispersion, matching
# `availability_window.MIN_ROLE_ROWS`.
MIN_ROLE_ROWS = 100

# The workhorse threshold. 0.75 of a 48-minute game is 36 mpg, which is the event §6 names
# and the one a best-ball ceiling actually turns on.
WORKHORSE_SHARE = 0.75

# The prior-season column the role buckets cut on. `stan_availability.ROLE_COL`, reused
# rather than re-chosen — the two heads grade on the same axis or their spreads are not
# comparable figures.
ROLE_COL = "minutes_per_game_lag1"

# The rotation filter §6's series used, kept so its figures reconcile rather than being
# silently replaced by better ones.
ROTATION_MIN_GAMES, ROTATION_MIN_MPG = 20, 10.0

# Era blocks the per-season series is summarized into. Endpoint-to-endpoint is how §6 stated
# the contraction, and it is the fragile way to state a drift — a single first season
# against a single last one — so the pooled blocks are reported beside it.
ERA_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("pre_2014_15", "1996-97", "2013-14"),
    ("post_2014_15", "2014-15", "2023-24"),
    ("post_2014_first_half", "2014-15", "2018-19"),
    ("post_2014_second_half", "2019-20", "2023-24"),
    ("three_point_era", "2012-13", "2023-24"),
)

# The last season the era series runs to. §6 stopped at 2023-24 because 2024-25 and 2025-26
# are the held-out split, and a *descriptive* series over them would still be a look.
ERA_LAST_SEASON = "2023-24"

# Breakpoint scan: fewest seasons either side of a candidate break, and how many Monte-Carlo
# replicates calibrate the null. The max over breakpoints has no standard F distribution,
# which is why the null is simulated rather than looked up — the device
# `docs/availability-window-plan.md` §2 uses, one unit up (a season-level series rather than
# player-season rows).
BREAK_MIN_SIDE = 4
BREAK_NULL_REPS = 5000

# Lookback lengths in target seasons for the rolling harness, and the first origin.
# Identical to `availability_window`'s so the two confirmations are comparable: 12 is the
# longest that is genuine at the earliest origin, and the fitting half starts at 1997-98.
LOOKBACKS: tuple[int | None, ...] = (None, 12, 8, 5, 3)
FIRST_ORIGIN = 2009
ROLLING_REFERENCE = "all__shared"

# The injected per-(player, season) sigma grid for the stake. Denser than
# `minutes_unification.PS_SIGMAS` around the shipped 0.450, because the question here is
# *where* the tie boundary sits rather than where the optimum does, and a boundary read off
# a coarse grid is a grid artifact.
STAKE_SIGMAS: tuple[float, ...] = (0.0, 0.15, 0.2, 0.25, 0.3, 0.375, 0.45, 0.525, 0.6)

SEED = 0


# ── The era series, through each head's own design ────────────────────────────

def _dispersion_row(label: str, season: str, rate: np.ndarray) -> dict:
    """One (population, season) cell: location, spread, and the workhorse tail.

    `sd_logit` is carried beside the raw sd because the head's link is a logit and a raw sd
    near a bounded mean confounds spread with location — if the mean falls, a constant
    logit-scale spread shows up as a *smaller* raw one. Reporting both is what separates
    "the league got tighter" from "the league got lower".
    """
    clipped = np.clip(rate, 1e-3, 1 - 1e-3)
    return {"population": label, "season": season, "n": int(len(rate)),
            "mean_rate": float(rate.mean()), "sd_rate": float(rate.std(ddof=1)),
            "sd_logit": float(np.std(np.log(clipped / (1 - clipped)), ddof=1)),
            "p_workhorse": float((rate >= WORKHORSE_SHARE).mean())}


def era_series(populations: dict[str, pd.DataFrame], rate_col: str = "rate",
               last_season: str = ERA_LAST_SEASON) -> pd.DataFrame:
    """Per-season location and spread of each population's own rate.

    One row per (population, season). Every population supplies its own already-filtered
    frame with a `season` and a `rate` column, so this function never re-derives a
    population — which is the failure `docs/availability-window-plan.md` §2 warns about, one
    head over: reconstructing a head's rows by hand picked the wrong window in one attempt
    and fanned out on a merge in another, and neither raised.
    """
    rows = []
    for label, frame in populations.items():
        frame = frame[frame["season"] <= last_season]
        for season, block in frame.groupby("season", sort=True):
            rows.append(_dispersion_row(label, str(season),
                                        block[rate_col].to_numpy(float)))
    return pd.DataFrame(rows)


def era_blocks(series: pd.DataFrame, blocks=ERA_BLOCKS) -> pd.DataFrame:
    """Each population's series pooled into eras, plus its two endpoint seasons.

    §6 quoted the endpoints; this reports both, because they answer different questions and
    disagree here. An endpoint pair is two single seasons and inherits their sampling noise;
    a block average is the level. The block rows are `n`-weighted means of the per-season
    cells, so a season with more qualified players counts for more — and pooling the
    *within-season* spreads rather than the pooled rows keeps between-season level drift out
    of a dispersion figure.
    """
    rows = []
    for label, part in series.groupby("population", sort=True):
        part = part.sort_values("season")
        first, last = part.iloc[0], part.iloc[-1]
        rows.append({"population": label, "block": "endpoints",
                     "first_season": first["season"], "last_season": last["season"],
                     "n": int(part["n"].sum()),
                     "sd_rate_first": float(first["sd_rate"]),
                     "sd_rate_last": float(last["sd_rate"]),
                     "sd_rate_change": float(last["sd_rate"] / first["sd_rate"] - 1.0),
                     "p_workhorse_first": float(first["p_workhorse"]),
                     "p_workhorse_last": float(last["p_workhorse"]),
                     "p_workhorse_fold": float(first["p_workhorse"]
                                               / max(last["p_workhorse"], 1e-9)),
                     "mean_rate_first": float(first["mean_rate"]),
                     "mean_rate_last": float(last["mean_rate"]),
                     "mean_rate_change": float(last["mean_rate"]
                                               / first["mean_rate"] - 1.0)})
        for name, lo, hi in blocks:
            block = part[(part["season"] >= lo) & (part["season"] <= hi)]
            if block.empty:
                continue
            w = block["n"].to_numpy(float)
            rows.append({"population": label, "block": name,
                         "first_season": block["season"].iloc[0],
                         "last_season": block["season"].iloc[-1],
                         "n": int(w.sum()),
                         "sd_rate": float(np.average(block["sd_rate"], weights=w)),
                         "sd_logit": float(np.average(block["sd_logit"], weights=w)),
                         "mean_rate": float(np.average(block["mean_rate"], weights=w)),
                         "p_workhorse": float(np.average(block["p_workhorse"],
                                                         weights=w))})
    return pd.DataFrame(rows)


def sup_f(values: np.ndarray, min_side: int = BREAK_MIN_SIDE
          ) -> tuple[np.ndarray, np.ndarray]:
    """`(max F, index of the best break)` over every candidate split, row-wise.

    A two-sample F against a constant null at each candidate breakpoint, maximized over
    them. `min_side` keeps a break from being declared on three seasons, where the F is a
    statement about one outlier.

    Accepts a `(T,)` series or an `(R, T)` stack of them and returns scalars or `(R,)`
    arrays to match, so the Monte-Carlo null runs through **the same code** as the observed
    statistic rather than a faster copy of it. Two implementations of one scan is exactly
    how a critical value ends up calibrating something slightly different from what it is
    compared against.
    """
    v = np.atleast_2d(np.asarray(values, dtype=float))
    R, T = v.shape
    c1 = np.cumsum(v, axis=1)
    c2 = np.cumsum(v ** 2, axis=1)
    total_s, total_q = c1[:, -1], c2[:, -1]
    ss_restricted = total_q - total_s ** 2 / T

    idx = np.arange(min_side, T - min_side)
    n1 = idx.astype(float)[None, :]
    s1, q1 = c1[:, idx - 1], c2[:, idx - 1]
    n2 = float(T) - n1
    s2, q2 = total_s[:, None] - s1, total_q[:, None] - q1
    ss_unrestricted = (q1 - s1 ** 2 / n1) + (q2 - s2 ** 2 / n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(ss_unrestricted > 0,
                     (ss_restricted[:, None] - ss_unrestricted)
                     / (ss_unrestricted / (T - 2)), 0.0)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    best = f.argmax(axis=1)
    f_max, break_idx = f[np.arange(R), best], idx[best]
    if np.ndim(values) == 1:
        return float(f_max[0]), int(break_idx[0])
    return f_max, break_idx


def break_scan(series: pd.DataFrame, reps: int = BREAK_NULL_REPS,
               seed: int = 42) -> pd.DataFrame:
    """sup-F over every candidate breakpoint of each population's series, MC-calibrated.

    The **max** over breakpoints has no standard F distribution, so the critical value is
    simulated: `reps` draws of a constant-mean Gaussian series with the observed length and
    sd, each put through the same scan. That is §2's device applied one unit up — a
    season-level series of ~27 points rather than ~10,000 player-season rows — because the
    quantity whose break matters here is a *dispersion*, which has no row-level analogue.

    Three statistics per population, and they are allowed to disagree: the mean is where the
    league's level moved, `sd_rate` is where its spread moved, and `p_workhorse` is the tail
    the contest cares about. §2 found the two availability tails breaking twelve seasons
    apart, so a scan reporting one number would be hiding the interesting part.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for label, part in series.groupby("population", sort=True):
        part = part.sort_values("season")
        for column in ("mean_rate", "sd_rate", "p_workhorse"):
            values = part[column].to_numpy(float)
            f_obs, idx = sup_f(values)
            null, _ = sup_f(rng.normal(values.mean(), values.std(ddof=1),
                                       size=(reps, len(values))))
            left, right = values[:idx], values[idx:]
            rows.append({
                "population": label, "statistic": column, "n_seasons": len(values),
                "sup_f": float(f_obs),
                "break_season": part["season"].iloc[idx],
                "null_p95": float(np.percentile(null, 95)),
                "p_value": float((null >= f_obs).mean()),
                "mean_before": float(left.mean()), "mean_after": float(right.mean()),
                "shift": float(right.mean() - left.mean()),
            })
    return pd.DataFrame(rows)


def rotation_population(cfg: dict) -> pd.DataFrame:
    """§6's own population: every player-season clearing `gp >= 20` and `mpg >= 10`.

    Rebuilt here rather than cited, so the doc's figures can be reconciled against the two
    head-native series in the same table. It is deliberately NOT either head's rows — that
    is the whole point of the caveat this module opens by honouring.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    lengths = pd.read_parquet(features_dir / "game_length.parquet")
    frame = minutes_targets(targets, lengths)
    frame["mpg"] = frame["minutes_played"] / frame["games_played"]
    frame = frame[(frame["games_played"] >= ROTATION_MIN_GAMES)
                  & (frame["mpg"] >= ROTATION_MIN_MPG)]
    return frame.assign(rate=frame["minutes_share"])


def composition_population(frame: pd.DataFrame) -> pd.DataFrame:
    """The composition head's own rows, aggregated to the unit the marginal head uses.

    A player-game's allocation is a share of the team-game pot `5 x game_length`; summing
    both sides over a season and multiplying by 5 puts it back on the *per-game* share scale
    the marginal head fits, so "0.75 is a 36-mpg workhorse" means the same thing in both
    rows of the table. Without that rescaling the two populations would be compared at
    thresholds five times apart, which is a unit error wearing the shape of an era effect.
    """
    out = frame[["player_id", "season"]].copy()
    out["played"] = frame["y"].to_numpy(float)
    out["pot"] = 5.0 * frame["game_length"].to_numpy(float)
    grouped = out.groupby(["player_id", "season"], as_index=False)[["played", "pot"]].sum()
    grouped = grouped[grouped["pot"] > 0]
    grouped["rate"] = 5.0 * grouped["played"] / grouped["pot"]
    return grouped


def team_game_concentration(frame: pd.DataFrame,
                            last_season: str = ERA_LAST_SEASON) -> pd.DataFrame:
    """Per-season mean Herfindahl index of the within-team-game minutes allocation.

    The composition's era question is not the marginal head's: its unit is a *share* of a
    fixed pot, so a league-wide level drift divides out before the head sees it and only the
    **concentration** of the allocation can move. HHI is that quantity, and it is the series
    §6 reports for this head.
    """
    frame = frame[frame["season"] <= last_season]
    minutes = frame["y"].to_numpy(float)
    keys = frame.groupby(GROUP_KEYS, sort=False)
    totals = keys["y"].transform("sum").to_numpy(float)
    share = np.divide(minutes, totals, out=np.zeros_like(minutes), where=totals > 0)
    per_game = (pd.DataFrame({"season": frame["season"].to_numpy(),
                              "game": keys.ngroup().to_numpy(),
                              "sq": share ** 2, "one": 1.0})
                .groupby(["season", "game"], as_index=False)[["sq", "one"]].sum())
    return (per_game.groupby("season", as_index=False)
            .agg(hhi=("sq", "mean"), n_team_games=("game", "size"),
                 mean_players=("one", "mean")))


# ── The point-MLE head ────────────────────────────────────────────────────────

class PointMinutes:
    """The minutes head's beta-binomial at its posterior mode, on successes out of trials.

    `availability.BetaBinomialGLM` is the same likelihood and the same alternating fit, but
    it reads `gp` / `team_games` from the frame by name and this head's columns are
    `successes` / `trials`. Rather than loosen that class — it is the shipped availability
    reference and several modules fit through it — the fit is reproduced here against this
    head's columns, reusing `_ab`, `_neg_loglik` and `fit_dispersion` so the likelihood
    itself stays shared code rather than a second implementation of one density.

    The alternating MLE with an analytic gradient in `beta` is not a preference either: a
    joint numeric-gradient fit over ~25 parameters against a sum of ~8,000 log-densities
    stops on gradient noise with coefficients near zero, which is the failure
    `BetaBinomialGLM` documents and this head would reproduce exactly.
    """

    name = "point_minutes"

    def __init__(self, features: list[str], l2: float = 1.0, max_rounds: int = 25,
                 predictive_samples: int = PREDICTIVE_SAMPLES):
        self.features, self.l2, self.max_rounds = features, l2, max_rounds
        self.predictive_samples = predictive_samples

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(df[self.features].to_numpy(dtype=float),
                             nan=0.0, posinf=0.0, neginf=0.0)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(self._matrix(df))
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, train: pd.DataFrame) -> "PointMinutes":
        self.scaler = StandardScaler().fit(self._matrix(train))
        X = self._design(train)
        y = train["successes"].to_numpy(int)
        n = train["trials"].to_numpy(int)
        if (y > n).any():
            raise ValueError("successes exceed trials — see `stan_minutes.as_trials`")

        def objective(beta: np.ndarray, rho: float) -> tuple[float, np.ndarray]:
            mu = _sigmoid(X @ beta)
            value = _neg_loglik(y, n, mu, rho) + self.l2 * float(beta[1:] @ beta[1:])
            a, b = _ab(mu, rho)
            r = float(np.clip(rho, RHO_MIN, RHO_MAX))
            scale = (1.0 - r) / r
            # a + b = scale is free of mu, so its digamma terms cancel from d/dmu.
            dmu = scale * (digamma(y + a) - digamma(a) - digamma(n - y + b) + digamma(b))
            grad = -(X.T @ (dmu * mu * (1.0 - mu)))
            grad[1:] += 2.0 * self.l2 * beta[1:]
            return value, np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        beta = np.zeros(X.shape[1])
        beta[0] = np.log(max(y.sum(), 1) / max(n.sum() - y.sum(), 1))
        rho = 0.05                              # the head's own fitted value, as a start
        for _ in range(self.max_rounds):
            best = minimize(objective, beta, args=(rho,), jac=True, method="L-BFGS-B")
            beta = best.x
            new_rho = fit_dispersion(y, n, _sigmoid(X @ beta))
            if abs(new_rho - rho) < 1e-6:
                rho = new_rho
                break
            rho = new_rho
        self.beta, self.rho = beta, float(rho)
        self.converged = bool(best.success)
        return self

    def mu(self, df: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self._design(df) @ self.beta)

    def rho_for(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.rho)

    @property
    def rho_spread(self) -> float:
        return 1.0

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        return self.mu(df) * df["trials"].to_numpy(float)

    def predict_samples(self, df: pd.DataFrame, seed: int = SEED) -> np.ndarray:
        """(draws x rows) season minutes from the predictive.

        `p ~ Beta(a, b)` then `y ~ Binomial(n, p)`, which is `StanMinutes.predict_samples`
        with the posterior over `beta` collapsed to its mode — so the two differ by exactly
        the coefficient uncertainty and nothing else, which is what makes the reference
        check in `run` interpretable.
        """
        mu = np.repeat(self.mu(df)[None, :], self.predictive_samples, axis=0)
        rho = np.repeat(self.rho_for(df)[None, :], self.predictive_samples, axis=0)
        rng = np.random.default_rng(seed)
        a, b = beta_shapes(mu, rho)
        return rng.binomial(df["trials"].to_numpy(int)[None, :],
                            rng.beta(a, b)).astype(float)


class RoleGradedPointMinutes(PointMinutes):
    """`PointMinutes` with one dispersion per prior-MPG role bucket.

    The mean is fitted exactly as the shared arm's; `rho` is then re-fitted inside each
    bucket holding that mean fixed. Identical device to
    `availability_window.RoleGradedBetaBinomial`, on this head's columns — and it nests the
    shared arm exactly when every bucket agrees, which a test pins.

    Role is **prior-season** MPG, known before opening night, so this is not the target
    grading itself.
    """

    name = "point_minutes_role_rho"

    @staticmethod
    def _buckets(df: pd.DataFrame) -> np.ndarray:
        return np.asarray(pd.cut(df[ROLE_COL].to_numpy(dtype=float),
                                 ROLE_EDGES, labels=ROLE_LABELS).astype(str))

    def fit(self, train: pd.DataFrame) -> "RoleGradedPointMinutes":
        super().fit(train)
        self.pooled_rho = float(self.rho)
        y = train["successes"].to_numpy(int)
        n = train["trials"].to_numpy(int)
        mu = self.mu(train)
        buckets = self._buckets(train)
        self.rho_by_role = {}
        for label in ROLE_LABELS:
            mask = buckets == label
            if mask.sum() >= MIN_ROLE_ROWS:
                self.rho_by_role[label] = fit_dispersion(y[mask], n[mask], mu[mask])
        return self

    def rho_for(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.rho_by_role.get(b, self.pooled_rho)
                         for b in self._buckets(df)], dtype=float)

    @property
    def rho_spread(self) -> float:
        values = list(self.rho_by_role.values())
        return max(values) / min(values) if values else 1.0


class FittedArm:
    """A fitted arm bound to the validation frame its own spline basis was built on.

    Each window fits its spline knots on its own rows, so `variants` hands back a
    *differently transformed* validation frame per window. Downstream code asks every arm
    for draws on "the validation rows", and this is what makes that request unambiguous: the
    arm carries the frame, so no caller can pair one window's coefficients with another
    window's basis.
    """

    def __init__(self, name: str, model: PointMinutes, frame: pd.DataFrame):
        self.name, self.model, self.frame = name, model, frame

    def predict_samples(self, seed: int = SEED) -> np.ndarray:
        return self.model.predict_samples(self.frame, seed)


def fit_arm(train: pd.DataFrame, val: pd.DataFrame, window: str, rho_mode: str,
            n_knots: int = SPLINE_KNOTS, variant: str = VARIANT) -> FittedArm:
    """One (window x dispersion) arm, with its spline basis fitted on its own window.

    Anything estimated from data is estimated on the fitting half, and a window is a
    restriction of the fitting half — so the knots move with the window rather than being
    inherited from the full one. That is the rule `availability_window.build_arm` applies to
    its trend centring, for the same reason.
    """
    cut = restrict_window(train, WINDOWS[window])
    tr, va, features = variants(cut, val, n_knots)[variant]
    cls = PointMinutes if rho_mode == "shared" else RoleGradedPointMinutes
    return FittedArm(f"{window}__{rho_mode}", cls(features).fit(tr), va)


# ── Scoring ───────────────────────────────────────────────────────────────────

def interval_coverage(samples: np.ndarray, y: np.ndarray, level: float) -> float:
    """Realized coverage of the central `level` predictive interval, from draws.

    The sample-based twin of `availability_window.interval_coverage`, which reads a pmf
    grid. Minutes run 0..~4,000, so a grid is not an option here and the quantiles come off
    the draws instead.
    """
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha], axis=0)
    return float(((y >= lo) & (y <= hi)).mean())


def score_arm(name: str, samples: np.ndarray, val: pd.DataFrame, train: pd.DataFrame,
              n_features: int, rho: float, rho_spread: float,
              rho_by_role: dict[str, float] | None = None,
              seed: int = SEED) -> tuple[dict, np.ndarray]:
    """One fitted arm's row, and its per-row CRPS for the paired bootstrap."""
    y = val["successes"].to_numpy(float)
    pred = samples.mean(axis=0)
    scores = crps_from_samples(samples, y)
    row = {
        "arm": name,
        "n_train": len(train),
        "n_train_seasons": int(pd.Series(train["season"]).nunique()),
        "first_train_season": min(train["season"]),
        "n_features": n_features,
        "n_val": len(val),
        "val_crps": float(scores.mean()),
        "val_mae": float(np.abs(pred - y).mean()),
        "val_r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "val_bias": float((pred - y).mean()),
        "val_pit_ks": ks_uniform(pit_from_samples(samples, y, seed)),
        # The quantity the head SHIPS for. Carried in the ladder rather than derived later,
        # because an arm that improves CRPS by narrowing a predictive that is the only
        # reason this head exists is not obviously an improvement.
        "predictive_sd": float(samples.std(axis=0).mean()),
        "realized_sd": float(y.std()),
        "rho": float(rho),
        "rho_spread": float(rho_spread),
    }
    # The per-bucket dispersions themselves, not only their ratio. A spread of 3x could be
    # three buckets agreeing and one outlier, and the availability head quotes its own
    # graded `rho` bucket by bucket for exactly that reason.
    for label in ROLE_LABELS:
        row[f"rho_{label}"] = float((rho_by_role or {}).get(label, np.nan))
    for level in COVERAGE_LEVELS:
        row[f"coverage_{int(level * 100)}"] = interval_coverage(samples, y, level)
    return row, scores


def ladder(train: pd.DataFrame, val: pd.DataFrame, n_knots: int = SPLINE_KNOTS,
           variant: str = VARIANT, seed: int = SEED
           ) -> tuple[pd.DataFrame, dict[str, FittedArm]]:
    """Every (window x dispersion) arm plus the no-fit floor, scored on validation."""
    rows: list[dict] = []
    per_row: dict[str, np.ndarray] = {}
    arms: dict[str, FittedArm] = {}

    floor = FloorMinutes().fit(train)
    row, scores = score_arm("carry_forward", floor.predict_samples(val, seed), val,
                            train, 0, floor.rho, 1.0, seed=seed)
    rows.append(row)
    per_row["carry_forward"] = scores

    for window in WINDOWS:
        for rho_mode in RHO_MODES:
            arm = fit_arm(train, val, window, rho_mode, n_knots, variant)
            arms[arm.name] = arm
            row, scores = score_arm(arm.name, arm.predict_samples(seed), arm.frame,
                                    restrict_window(train, WINDOWS[window]),
                                    len(arm.model.features), arm.model.rho,
                                    arm.model.rho_spread,
                                    getattr(arm.model, "rho_by_role", None), seed)
            rows.append(row)
            per_row[arm.name] = scores
            print(f"  {arm.name:<28} {row['n_train']:>6,} rows / "
                  f"{row['n_train_seasons']:>2} seasons   CRPS {row['val_crps']:8.3f}   "
                  f"rho {row['rho']:.5f} ({row['rho_spread']:.2f}x)   "
                  f"predictive sd {row['predictive_sd']:7.2f}")

    reference = per_row[REFERENCE_ARM]
    floor_crps = float(next(r["val_crps"] for r in rows if r["arm"] == "carry_forward"))
    for row in rows:
        delta = paired_bootstrap(per_row[row["arm"]], reference, n_boot=BOOTSTRAP_REPS,
                                 seed=seed)
        row["crps_vs_incumbent"] = delta["crps_delta"]
        row["crps_vs_incumbent_lo"] = delta["ci_lo"]
        row["crps_vs_incumbent_hi"] = delta["ci_hi"]
        row["beats_incumbent"] = bool(delta["ci_hi"] < 0.0)
        # The incumbent's own row is a comparison against itself, where the interval is
        # degenerately [0, 0] and `verdict` would read it as a win. Naming it stops a table
        # from claiming the reference beat the reference.
        row["verdict"] = "reference" if row["arm"] == REFERENCE_ARM else verdict(delta)
        row["beats_floor"] = bool(row["val_crps"] <= floor_crps)
    return pd.DataFrame(rows), arms


# ── Rolling-origin confirmation, fitting half only ────────────────────────────

def rolling_confirmation(train: pd.DataFrame, n_knots: int = SPLINE_KNOTS,
                         variant: str = VARIANT, seed: int = SEED) -> pd.DataFrame:
    """Walk-forward over the fitting half: one fit per (origin, lookback, dispersion).

    The ladder scores eight arms on 742 validation player-seasons, which is a multiplicity
    problem and a power problem at once. This answers it **without spending validation
    twice** — every row it touches is a fitting-half row — and it reparameterizes the window
    into a **lookback length**, which is the thing that will still mean something when the
    production fit runs for 2026-27. An absolute first season is a fact about the past.
    """
    year = season_start_year(train)
    origins = [int(y) for y in sorted(np.unique(year)) if y >= FIRST_ORIGIN]
    per_arm: dict[str, dict[str, list]] = {}

    for origin in origins:
        score = train[year == origin]
        if score.empty:
            continue
        for lookback in LOOKBACKS:
            floor_year = -np.inf if lookback is None else origin - lookback
            fit_rows = train[(year < origin) & (year >= floor_year)]
            if len(fit_rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
                continue
            tr, sc, features = variants(fit_rows, score, n_knots)[variant]
            y = sc["successes"].to_numpy(float)
            for rho_mode in RHO_MODES:
                name = f"{'all' if lookback is None else lookback}__{rho_mode}"
                cls = PointMinutes if rho_mode == "shared" else RoleGradedPointMinutes
                model = cls(features).fit(tr)
                samples = model.predict_samples(sc, seed)
                slot = per_arm.setdefault(name, {})
                slot.setdefault("crps", []).append(crps_from_samples(samples, y))
                slot.setdefault("pit", []).append(pit_from_samples(samples, y, seed))
                slot.setdefault("sd", []).append(samples.std(axis=0))
                slot.setdefault("y", []).append(y)
                slot.setdefault("origin", []).append(np.full(len(sc), origin))
                slot.setdefault("n_fit", []).append(len(tr))
                slot.setdefault("rho", []).append(model.rho)
        print(f"  origin {origin}: scored {len(score):,} rows")

    pooled = {name: {k: np.concatenate(v) for k, v in d.items()
                     if k not in ("n_fit", "rho")}
              for name, d in per_arm.items()}
    reference = pooled[ROLLING_REFERENCE]["crps"]
    rows = []
    for name, d in pooled.items():
        origin = d["origin"]
        delta = paired_bootstrap(d["crps"], reference, n_boot=BOOTSTRAP_REPS, seed=seed)
        # Origins are the independent replicates, so a win count over them is the
        # multiplicity-robust statement a pooled interval is not.
        wins = sum(1 for o in np.unique(origin)
                   if d["crps"][origin == o].mean() < reference[origin == o].mean())
        rows.append({
            "arm": name,
            "lookback": name.split("__")[0],
            "rho_mode": name.split("__")[1],
            "n_origins": int(len(np.unique(origin))),
            "n_scored": int(len(d["y"])),
            "mean_fit_rows": float(np.mean(per_arm[name]["n_fit"])),
            "mean_rho": float(np.mean(per_arm[name]["rho"])),
            "crps": float(d["crps"].mean()),
            "crps_vs_all": delta["crps_delta"],
            "crps_vs_all_lo": delta["ci_lo"],
            "crps_vs_all_hi": delta["ci_hi"],
            "origins_won": wins,
            "pit_ks": ks_uniform(d["pit"]),
            "predictive_sd": float(d["sd"].mean()),
            "realized_sd": float(d["y"].std()),
        })
    return pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)


# ── The stake: what a narrower predictive does to the injection ───────────────

def injection_restake(arms: dict[str, FittedArm], minutes_val: pd.DataFrame,
                      composition_val: pd.DataFrame, artifact, keep: int,
                      shipped: float, sigmas: tuple[float, ...] = STAKE_SIGMAS,
                      seed: int = SEED) -> pd.DataFrame:
    """The composition's injected-sigma grid, re-scored against each window's marginal arm.

    `make minutes-unification` measured the injection against **one** reference — the
    shipped full-window marginal head — and read a tie at sigma 0.375 and again at the
    shipped 0.450. If a short window narrows the marginal predictive, the composition needs
    *less* injected spread before this repo can no longer tell the two apart, and the tie
    boundary moves. That is the quantity this function reports.

    The composition is simulated **once per sigma** and re-used across references, because
    its draws do not depend on which marginal arm they are compared to. Nothing is refitted:
    the persisted posterior is rehydrated and the head's own `simulate_minutes` runs, so
    every figure here comes from the same fitted object `minutes_unification` scores.

    The tie is reported as a **band** rather than as its lower edge alone, because the
    injection loses at both ends: too little spread and the composition is under-dispersed,
    too much and it is over-dispersed. A stronger reference narrows the band from both
    sides, and whether the shipped constant still sits inside it is the only part of this
    that is operational. Both edges are **grid** boundaries and ship with the grid step
    beside them rather than as continuous estimates.
    """
    composition = rehydrate_composition(artifact, keep)
    frame = artifact.recipe.transform(composition_val)
    eta_base, rho = composition._eta_base(frame)
    codes = unit_codes(composition_val)
    n_units, n_draws = int(codes.max()) + 1, eta_base.shape[1]

    comp_units = (composition_val.groupby(UNIT_KEYS, sort=True).size()
                  .reset_index(name="games")
                  .merge(realized_totals(composition_val, "y"), on=UNIT_KEYS))
    mins_units = minutes_val[UNIT_KEYS].copy()
    mins_units["realized"] = minutes_val["successes"].to_numpy(float)

    common = mins_units.merge(comp_units[UNIT_KEYS], on=UNIT_KEYS, how="inner")
    comp_pos = {k: i for i, k in enumerate(map(tuple, comp_units[UNIT_KEYS].to_numpy()))}
    mins_pos = {k: i for i, k in enumerate(map(tuple, mins_units[UNIT_KEYS].to_numpy()))}
    keys = list(map(tuple, common[UNIT_KEYS].to_numpy()))
    idx_c = np.array([comp_pos[k] for k in keys])
    idx_m = np.array([mins_pos[k] for k in keys])
    y_comp = comp_units["realized"].to_numpy(float)[idx_c]
    # Each arm is scored against its own frame's realized total, exactly as
    # `minutes_unification` does — the two frames disagree on a handful of player-games and
    # that disagreement is not allowed to become the thing carrying a verdict.
    y_mins = mins_units["realized"].to_numpy(float)[idx_m]

    references: dict[str, dict] = {}
    for name, arm in arms.items():
        samples = arm.predict_samples(seed)[:, idx_m]
        references[name] = {
            "crps": crps_from_samples(samples, y_mins),
            "predictive_sd": float(samples.std(axis=0).mean()),
            "pit_ks": ks_uniform(pit_from_samples(samples, y_mins, seed)),
        }

    rows = [{"arm": f"minutes__{name}", "unit": "season_total", "n": len(common),
             "n_draws": keep, "sigma": np.nan, "sigma_source": "marginal_head",
             "crps_minutes": float(ref["crps"].mean()),
             "pit_ks": ref["pit_ks"], "predictive_sd": ref["predictive_sd"]}
            for name, ref in references.items()]

    for sigma in sigmas:
        rng = np.random.default_rng(seed + 7)
        eta = eta_base + float(sigma) * rng.normal(size=(n_units, n_draws))[codes, :]
        totals, _ = season_totals(simulate_minutes(frame, eta, rho, seed), composition_val)
        scored = totals[:, idx_c]
        crps = crps_from_samples(scored, y_comp)
        base = {"arm": "composition_sum_plus_player_season_effect", "unit": "sigma_grid",
                "n": len(common), "n_draws": n_draws, "sigma": float(sigma),
                "sigma_source": "injected_grid",
                "crps_minutes": float(crps.mean()),
                "mae_minutes": float(np.abs(scored.mean(axis=0) - y_comp).mean()),
                "pit_ks": ks_uniform(pit_from_samples(scored, y_comp, seed)),
                "predictive_sd": float(scored.std(axis=0).mean())}
        for name, ref in references.items():
            delta = paired_bootstrap(crps, ref["crps"], n_boot=BOOTSTRAP_REPS, seed=seed)
            base[f"vs_{name}_delta"] = delta["crps_delta"]
            base[f"vs_{name}_lo"] = delta["ci_lo"]
            base[f"vs_{name}_hi"] = delta["ci_hi"]
            base[f"vs_{name}_verdict"] = verdict(delta)
        rows.append(base)
        print(f"  sigma {sigma:.3f}: composition CRPS {base['crps_minutes']:8.3f}, "
              f"predictive sd {base['predictive_sd']:7.2f}   "
              + "  ".join(f"vs {n} {base[f'vs_{n}_verdict']}" for n in references))

    grid = pd.DataFrame([r for r in rows if r["sigma_source"] == "injected_grid"])
    step = float(np.min(np.diff(np.sort(grid["sigma"].to_numpy()))))
    shipped_row = grid[np.isclose(grid["sigma"], shipped)]
    for name, ref in references.items():
        tie = grid[grid[f"vs_{name}_verdict"] != "loses"].sort_values("sigma")
        rows.append({
            "arm": f"tie_band__{name}", "unit": "tie_boundary",
            "n": len(common), "n_draws": keep,
            "sigma": float(tie["sigma"].iloc[0]) if len(tie) else np.nan,
            "sigma_band_hi": float(tie["sigma"].iloc[-1]) if len(tie) else np.nan,
            "sigma_source": "grid_boundary", "grid_step": step,
            "predictive_sd": ref["predictive_sd"],
            "crps_minutes": float(ref["crps"].mean()),
            "shipped_sigma": float(shipped),
            "shipped_verdict": (str(shipped_row[f"vs_{name}_verdict"].iloc[0])
                                if len(shipped_row) else "off_grid"),
        })
    return pd.DataFrame(rows)


# ── Guards ────────────────────────────────────────────────────────────────────

def assert_shipped_variant(out_dir: Path, variant: str = VARIANT) -> str:
    """The ladder's fixed variant is the one `make stan-minutes` selected, or this raises.

    A window ladder that silently laddered a *different* feature variant from the shipped
    head would produce a table of arms none of which is the incumbent, and the reference
    check in `run` would be comparing two things that differ in two ways at once. Reported
    as unverified rather than failed when the metrics artifact is absent, so a fresh
    checkout without `make stan-minutes` still runs.
    """
    path = Path(out_dir) / "stan_minutes_metrics.csv"
    if not path.exists():
        return f"{variant} (unverified — no stan_minutes_metrics.csv)"
    table = pd.read_csv(path)
    selected = str(table.loc[table["selected"], "variant"].iloc[0])
    if selected != variant:
        raise ValueError(
            f"the ladder holds `{variant}` fixed but `make stan-minutes` selected "
            f"`{selected}`. Update `minutes_window.VARIANT` — a window ladder on a variant "
            f"the head does not ship has no incumbent in it.")
    return f"{variant} (matches stan_minutes_metrics.csv)"


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_stan = cfg.get("stan", {})
    n_knots = int(cfg_stan.get("minutes", {}).get("spline_knots", SPLINE_KNOTS))
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    cfg_sim = cfg.get("sim", {})
    keep = int(cfg_sim.get("posterior_draws", 1000))
    # Named `fit_window` rather than `window`, because `window` is also what this module
    # calls a *fitting-window arm* and the two are different things — the posterior's fit
    # window is `train`, an arm's is `post_2014`.
    fit_window = str(cfg_sim.get("fit_window", "train"))

    variant = assert_shipped_variant(out_dir)
    print("Minutes window ladder — the fitting window and the dispersion of "
          "`min | available`")
    print(f"  variant held fixed at {variant}")

    design = minutes_build_design(cfg)
    train, val = selection_split(design, test_seasons)
    print(f"  {len(design):,} player-seasons; {len(train):,} fit / {len(val):,} select "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")
    print("  The held-out split is LOCKED (src/models/held_out.py) and is never "
          "materialized here.\n  Every window restricts the FITTING half only.")

    # ── 1. The era series, through each head's own design ─────────────────────
    print(f"\nStep 1 — rebuilding `docs/availability-window-plan.md` §6's series through "
          f"each head's OWN design\n  rows. Its own caveat says those were measured on a "
          f"rotation filter (`gp >= {ROTATION_MIN_GAMES}`, "
          f"`mpg >= {ROTATION_MIN_MPG:.0f}`)\n  rather than either head's row filter, and "
          f"that the fold in the workhorse tail is too large for\n  a population definition "
          f"to flip while the sd contraction is not.")
    comp_frame_all = composition_frame(cfg)
    populations = {
        "rotation_filter": rotation_population(cfg),
        "minutes_head": design.assign(rate=design["minutes_share"]),
        "composition_head": composition_population(comp_frame_all),
    }
    series = era_series(populations)
    blocks = era_blocks(series)
    # The composition's concentration measure joins the same file as a fourth population
    # rather than a sixth artifact: it is the same question at the same unit of time, and
    # the columns it does not share stay empty, which is already true of the block rows.
    hhi = team_game_concentration(comp_frame_all)
    era_dest = out_dir / "minutes_window_era.csv"
    pd.concat([series.assign(block=""), blocks.assign(season=""),
               hhi.assign(population="composition_team_game", block="")],
              ignore_index=True).to_csv(era_dest, index=False)

    print("\n  Endpoints — how §6 stated it — by population:")
    ends = blocks[blocks["block"] == "endpoints"]
    for _, r in ends.iterrows():
        print(f"    {r['population']:<17} sd {r['sd_rate_first']:.4f} -> "
              f"{r['sd_rate_last']:.4f} ({r['sd_rate_change']:+.1%}),  "
              f"P(rate >= {WORKHORSE_SHARE:.2f}) {r['p_workhorse_first']:.4f} -> "
              f"{r['p_workhorse_last']:.4f} ({r['p_workhorse_fold']:.1f}x),  "
              f"mean {r['mean_rate_change']:+.1%}")
    print("\n  Pooled era blocks (n-weighted over each block's seasons):")
    print(blocks[blocks["block"] != "endpoints"][
        ["population", "block", "n", "sd_rate", "sd_logit", "mean_rate", "p_workhorse"]
    ].round(4).to_string(index=False))

    print(f"\n  Composition concentration on its OWN unit — mean team-game HHI "
          f"{hhi['hhi'].iloc[0]:.4f} ({hhi['season'].iloc[0]})\n  -> "
          f"{hhi['hhi'].iloc[-1]:.4f} ({hhi['season'].iloc[-1]}), trough "
          f"{hhi['hhi'].min():.4f} in {hhi.loc[hhi['hhi'].idxmin(), 'season']}.")

    breaks = break_scan(series)
    break_dest = out_dir / "minutes_window_break.csv"
    breaks.to_csv(break_dest, index=False)
    print(f"\n  sup-F over every candidate breakpoint, against a "
          f"{BREAK_NULL_REPS:,}-replicate Monte-Carlo null:")
    print(breaks[["population", "statistic", "sup_f", "break_season", "null_p95",
                  "p_value", "shift"]].round(4).to_string(index=False))
    print(f"\nWrote {len(series):,} season cells + {len(blocks):,} block rows → {era_dest}")
    print(f"Wrote {len(breaks):,} breakpoint rows → {break_dest}")

    # ── 2. The ladder ─────────────────────────────────────────────────────────
    print(f"\nStep 2 — the window x dispersion ladder on the point MLE "
          f"({len(WINDOWS)} x {len(RHO_MODES)} arms plus the no-fit floor):")
    table, arms = ladder(train, val, n_knots, seed=SEED)
    dest = out_dir / "minutes_window.csv"
    table.to_csv(dest, index=False)
    print()
    print(table[["arm", "n_train", "n_train_seasons", "val_crps", "crps_vs_incumbent",
                 "crps_vs_incumbent_lo", "crps_vs_incumbent_hi", "val_pit_ks",
                 "predictive_sd", "rho", "rho_spread", "verdict"]]
          .round(4).to_string(index=False))
    print(f"\nWrote {len(table):,} arms → {dest}")

    graded = table[table["rho_spread"] > 1.0]
    print(f"\n  The graded dispersion, bucket by bucket "
          f"(`{ROLE_COL}`, {'/'.join(ROLE_LABELS)}):")
    for _, r in graded.iterrows():
        print(f"    {r['arm']:<26} "
              + "  ".join(f"{r[f'rho_{lab}']:.5f}" for lab in ROLE_LABELS)
              + f"   {r['rho_spread']:.2f}x")

    incumbent = table[table["arm"] == REFERENCE_ARM].iloc[0]
    print(f"\n  Reference check: `{REFERENCE_ARM}` reads CRPS {incumbent['val_crps']:.4f} "
          f"and predictive sd {incumbent['predictive_sd']:.2f},\n  against the shipped Stan "
          f"head's own season-unit figures in `minutes_unification.csv`. The two\n  differ "
          f"by the posterior over `beta`, which the point MLE collapses to its mode, and by "
          f"nothing\n  else — so an arm that moves either column here moves it there.")

    # ── 3. Rolling-origin confirmation ────────────────────────────────────────
    print(f"\nStep 3 — rolling-origin confirmation on the FITTING HALF ONLY "
          f"(origins from {FIRST_ORIGIN}, lookbacks {LOOKBACKS}):")
    rolling = rolling_confirmation(train, n_knots, seed=SEED)
    roll_dest = out_dir / "minutes_window_rolling.csv"
    rolling.to_csv(roll_dest, index=False)
    print()
    print(rolling[["arm", "mean_fit_rows", "n_scored", "crps", "crps_vs_all",
                   "crps_vs_all_lo", "crps_vs_all_hi", "origins_won", "pit_ks",
                   "predictive_sd", "mean_rho"]].round(4).to_string(index=False))
    print(f"\nWrote {len(rolling):,} arms × {int(rolling['n_scored'].max()):,} scored "
          f"rows → {roll_dest}")

    # The two readings put side by side, because the ladder crosses two axes and only the
    # harness can say which of them the fitting half reproduces. A window is matched to the
    # lookback with the closest fit-row count rather than to a nominal season count: the
    # rows are what a bias-variance trade actually spends.
    print("\n  Does the validation ladder replicate on the fitting half? The two axes are "
          "matched by\n  fit-row count, since rows rather than nominal seasons are what a "
          "window trades:")
    print(f"    {'axis':<34}{'validation':>22}{'rolling origin':>26}")
    for cut, lookback in (("three_point_era", "12"), ("post_2014", "8")):
        v = table[table["arm"] == f"{cut}__shared"].iloc[0]
        r = rolling[rolling["arm"] == f"{lookback}__shared"].iloc[0]
        print(f"    window {cut:<27}"
              f"{v['crps_vs_incumbent']:+8.3f} [{v['crps_vs_incumbent_lo']:+.2f}, "
              f"{v['crps_vs_incumbent_hi']:+.2f}]"
              f"{r['crps_vs_all']:+9.3f} [{r['crps_vs_all_lo']:+.2f}, "
              f"{r['crps_vs_all_hi']:+.2f}]  {r['origins_won']:>2}/{r['n_origins']}")
    v = table[table["arm"] == "full__role"].iloc[0]
    r = rolling[rolling["arm"] == "all__role"].iloc[0]
    print(f"    {'role-graded rho, window held':<34}"
          f"{v['crps_vs_incumbent']:+8.3f} [{v['crps_vs_incumbent_lo']:+.2f}, "
          f"{v['crps_vs_incumbent_hi']:+.2f}]"
          f"{r['crps_vs_all']:+9.3f} [{r['crps_vs_all_lo']:+.2f}, "
          f"{r['crps_vs_all_hi']:+.2f}]  {r['origins_won']:>2}/{r['n_origins']}")

    # ── 4. The stake ──────────────────────────────────────────────────────────
    print("\nStep 4 — the stake. `make minutes-unification` ships the marginal head SOLELY "
          "for its\n  season-level spread, so the question is whether a window changes that "
          "spread and, through\n  it, how much injected per-(player, season) sigma the "
          "composition needs to draw level.")
    artifacts = {"composition": load("composition", posteriors_dir(cfg, fit_window))}
    require_window(artifacts, fit_window)
    first_season = str(cfg_stan.get("composition", {})
                       .get("first_season", PILOT_FIRST_SEASON))
    pilot = comp_frame_all[comp_frame_all["season"] >= first_season].reset_index(drop=True)
    _, composition_val = selection_split(pilot, test_seasons)
    print(f"  composition posterior @ {artifacts['composition'].recipe.variant}, "
          f"{artifacts['composition'].n_draws:,} draws at the `{fit_window}` window; "
          f"nothing is refitted.")

    # The window axis at the shipped dispersion, plus whichever arm the ladder selected, so
    # the stake is read against the arm that would actually ship rather than only against
    # the axis being varied.
    fitted = table[table["arm"] != "carry_forward"]
    selected = str(fitted.loc[fitted["val_crps"].idxmin(), "arm"])
    stake_arms = {f"{w}__shared": arms[f"{w}__shared"] for w in WINDOWS}
    stake_arms[selected] = arms[selected]
    print(f"  references: {', '.join(stake_arms)} "
          f"(ladder-selected arm: {selected})")

    shipped = shipped_sigma(cfg)
    stake = injection_restake(stake_arms, val, composition_val,
                              artifacts["composition"], keep, shipped)
    stake_dest = out_dir / "minutes_window_stake.csv"
    stake.to_csv(stake_dest, index=False)

    heads = stake[stake["sigma_source"] == "marginal_head"]
    ties = stake[stake["sigma_source"] == "grid_boundary"]
    base_sd = float(heads.loc[heads["arm"] == "minutes__full__shared",
                              "predictive_sd"].iloc[0])
    print(f"\n  Each window's marginal head at the season unit, on the "
          f"{int(heads['n'].iloc[0]):,} rows both heads cover:")
    print(heads.assign(sd_vs_full=heads["predictive_sd"] / base_sd - 1.0)
          [["arm", "crps_minutes", "pit_ks", "predictive_sd", "sd_vs_full"]]
          .round(4).to_string(index=False))
    print(f"\n  The injected sigma BAND over which the composition is not distinguishable "
          f"from each arm\n  (grid step {ties['grid_step'].iloc[0]:.3f}); the injection "
          f"loses at both ends, under- then over-dispersed:")
    print(ties[["arm", "sigma", "sigma_band_hi", "shipped_sigma", "shipped_verdict",
                "predictive_sd", "crps_minutes"]].round(4).to_string(index=False))

    print(f"\n  The shipped constant is sim.minutes.player_season_sigma = {shipped:.3f}, "
          f"and it is selected by\n  the COMPOSITION's own CRPS optimum on training rows "
          f"(`minutes_unification.estimate_sigma_on_train`).\n  The marginal head appears "
          f"nowhere in that estimator, so a narrower marginal predictive cannot\n  move it "
          f"under the rule that chose it. What it moves is the tie band above — a different"
          f"\n  question, and the one this table answers.")
    print(f"\nWrote {len(stake):,} stake rows → {stake_dest}")

    return {"era": era_dest, "break": break_dest, "ladder": dest, "rolling": roll_dest,
            "stake": stake_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
