"""Predict a player's games played next season — as a distribution, not a number.

Games played is the largest lever on the season DK total and the least predictable input
in the project: it persists year over year at r = 0.316 against 0.779 for minutes per
game, and `log(season total)` is 73.4% explained by `log(games)` alone.

Two measured facts set the whole design, both from `make availability-profile`:

- **The internal ceiling is R² ≈ 0.24, in sample.** There is no durability latent to
  extract — a three-year availability average does not beat one year (r 0.392 vs 0.398),
  and injury severity persists at r = 0.090. So this module **starts with the baselines
  and stops there unless they are beaten**, per the plan's own instruction. The spell
  simulator is not built on spec.
- **Season GP is ~20× overdispersed against a binomial**, with a mode near 72-82 and a
  long left tail: 26.7% of established rotation players fall below 60 games and 9.6%
  below 41. A point estimate cannot represent that, and squared error against GP regresses
  everyone to ~65 games and never produces the tail. Every model here therefore emits a
  **beta-binomial over `gp` out of `team_games`**, and is scored with **CRPS and a PIT
  histogram**, with MAE and R² alongside only for continuity with the rest of the repo.

The beta-binomial is the natural likelihood for a bounded overdispersed count and gives
the distribution for free: with mean μ and dispersion ρ, the variance is
`n·μ(1-μ)·[1 + (n-1)ρ]`, so ρ *is* the overdispersion the profile measures — at n = 82 a
20× ratio is ρ ≈ 0.235. The point-prediction models (league/age, ridge, GBM) are wrapped
in the same family with ρ fitted on their own training residuals, so all four are scored
on identical footing and the comparison is about the mean function alone.

## Two rules this module exists to obey

**Do not minutes-weight.** The project-wide rule is the opposite, and this is the
documented exception (`CLAUDE.md`). Minutes weighting exists to stop a per-36 rate
measured over three garbage-time minutes from dominating a fit — real measurement error.
Games played has none: "he played 12 games" is exact. Weighting by minutes down-weights
precisely the injured seasons this head exists to predict, and halves the ceiling
(R² 0.236 → 0.116).

**Point-in-time discipline.** Every row carries `as_of_date`, and `assert_point_in_time`
refuses to return a design matrix where any row's `as_of_date` reaches its own season.
Features come from season S-1 and earlier only. The preseason injury snapshot the plan
specifies is *not* wired in yet and cannot be: the archives that could fill it
(`src/data/injury_reports.py`, `src/data/injuries.py`) begin in December 2025 and build
forward only, so every historical row would have to be filled from a current-status
source — which encodes the resolved outcome. The ablation that measures the snapshot's
value has to wait for the archive to cover a season boundary.

## Which rows every number here is measured on

**Validation**, since 2026-08-08. This module was the last head in the project still
scoring the held-out seasons, and it is the one where that mattered most: it does not
merely report, it *decides*. The four-way model ladder picks which mean function ships,
`workload_ablation` decides a feature block, and `nonlinearity_ablation` decides a basis.
Three decisions taken on the split that exists to measure them.

`split_seasons` still lives here — it is the choke point `src/models/held_out.py` wraps —
but nothing in this module calls it any more. `run` reaches the frames through
`held_out.selection_split`, so the test seasons are never materialized and the printed
ladder is a validation ladder.

Two functions lost a column in the move, and it is worth being plain about what that cost.
`nonlinearity_ablation` and `minutes_nonlinearity_probe` used to carve their own inner
validation split out of `train` and report val *and* test. The val column is unchanged —
`selection_split` hands back exactly the frames that inner split produced — but the test
column is gone, and with it the minutes probe's `replicates` flag. That flag was never the
replication check it looked like: the two columns differed in training data as well as in
evaluation rows, which is the confound `src/models/held_out.py` was written to stop being
read as agreement. What survives is what was always the honest half — the *contrast between
the two targets*, both measured on the same validation frame.

Usage:
    python -m src.models.availability
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize, minimize_scalar
from scipy.special import digamma
from scipy.stats import betabinom
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.eda.availability import load_ages, with_lags
from src.eda.feature_diagnostics import above_null
from src.features.availability import (attach_workload, build_panel,
                                       playoff_workload, season_availability)
from src.models.held_out import selection_split

# Season S-1 quantities. Lag 2 and 3 are kept because the profile's ladder shows them
# adding a little (R² 0.214 → 0.223), and filled from lag 1 when a player has no third
# prior season — requiring three would drop ~40% of rows to buy 0.009 R².
#
# `missed_games` is deliberately absent: it is `team_games - gp` and so carries the same
# information as `gp_share` on a different scale. The *reason* split of it is the
# interesting version, and that waits on the decomposition retest in
# `src/eda/availability.py::decomposition` rather than being fed on spec.
#
# The playoff-workload columns come from `features.availability.playoff_workload`. Season
# S-1's playoffs end in June and season S opens in October, so their lag-1 values are
# knowable at `as_of_date` with no special handling.
LAG_COLS = ["gp_share", "minutes_per_game", "total_minutes",
            "trailing_missed", "end_play_rate", "n_spells", "longest_spell",
            "playoff_games", "playoff_mpg", "playoff_minutes_share", "career_minutes"]

# The playoff/mileage block, held separate so the ablation in `workload_ablation` is a
# one-line contrast rather than a second hard-coded list.
WORKLOAD_COLS = ["playoff_games_lag1", "playoff_mpg_lag1",
                 "playoff_minutes_share_lag1", "career_minutes_lag1"]

# The un-lagged columns `attach_workload` has to have supplied for the above to exist.
WORKLOAD_SOURCE_COLS = [c.removesuffix("_lag1") for c in WORKLOAD_COLS]

FEATURE_COLS = [
    "gp_share_lag1", "gp_share_lag2", "gp_share_lag3",
    "minutes_per_game_lag1", "minutes_per_game_lag2", "minutes_per_game_lag3",
    "total_minutes_lag1", "trailing_missed_lag1", "end_play_rate_lag1",
    "n_spells_lag1", "longest_spell_lag1",
    "age", "age_sq", "career_year", "n_prior_seasons",
    *WORKLOAD_COLS,
]

# `total_minutes_incl_playoffs_lag1` is deliberately NOT here. It was built on the
# reasoning that `total_minutes` undercounts real mileage, which is true and predictively
# useless: swapping it in *lowers* in-sample R² (0.2843 → 0.2816), because folding playoff
# minutes into the total mixes team quality into what was a clean regular-season workload
# measure. Keep the two effects in separate columns.

class _HeldOut(pd.DataFrame):
    """A DataFrame that refuses to be read while the held-out split is locked.

    Subclassing rather than wrapping so it stays a DataFrame everywhere — the final
    evaluation unlocks and uses it exactly as before. Only the accessors that actually
    surface data are guarded; `len()` and `.columns` stay free, because reporting how many
    rows are held out is not the same as reading them.
    """

    _metadata: list = []

    @property
    def _constructor(self):
        return _HeldOut

    def _check(self):
        from src.models.held_out import assert_unlocked
        assert_unlocked("the held-out frame from `split_seasons`")

    def __getitem__(self, key):
        self._check()
        return pd.DataFrame(self)[key]

    def to_numpy(self, *a, **k):
        self._check()
        return pd.DataFrame(self).to_numpy(*a, **k)

    def merge(self, *a, **k):
        self._check()
        return pd.DataFrame(self).merge(*a, **k)


TEST_SEASONS = 2

# The head every downstream consumer holds: `stan_availability` ports it, `season_total`
# composes it, `stan_games_played` uses it as its permanent floor. The ladder is scored
# *against* it rather than against whichever row happens to have the lowest CRPS, because
# "is the leader distinguishable from the incumbent" is the question a selection has to
# answer and "who leads" is not.
SHIPPED_MODEL = "beta_binomial"

AGE_SHRINKAGE = 50.0      # pseudo-observations pulling each age toward the league mean
RIDGE_ALPHA = 100.0
EPS = 1e-3

# The population the profile quotes its tail figures on: last season's regulars. Reported
# separately because aggregate metrics hide them completely.
ROTATION_MIN_MPG = 20.0
ROTATION_MIN_GP_SHARE = 0.70
TAIL_THRESHOLDS = [41, 60]


# ── Design matrix ─────────────────────────────────────────────────────────────

def season_start_dates(panel: pd.DataFrame) -> pd.Series:
    """First game date of each season — what `as_of_date` has to precede."""
    return panel.groupby("season")["game_date"].min()


def build_design(frame: pd.DataFrame, seasons: list[str], raw_dir: str | Path,
                 starts: pd.Series) -> pd.DataFrame:
    """One row per (player, target season), features from S-1 and earlier only.

    `as_of_date` is the day before the target season's first game: the latest date at
    which this forecast is still a forecast. It is carried on every row so a model can
    later be trained at several prediction dates, which the plan flags as a product
    decision rather than a modelling one.

    The playoff/mileage block is attached here rather than by each caller. It lives
    outside `season_availability` because the panel is built from regular-season logs
    only — playoff games are a feature of S-1, never a row to fit (`CLAUDE.md`, "Scope") —
    and every consumer of this design matrix needs it, so doing it once here is what stops
    the next caller from failing on a missing column.
    """
    missing = [c for c in WORKLOAD_SOURCE_COLS if c not in frame.columns]
    if missing:
        frame = attach_workload(frame, playoff_workload(seasons, raw_dir), seasons)

    lagged = with_lags(frame, seasons, LAG_COLS, max_lag=3)

    ages = load_ages(seasons, raw_dir)
    lagged = (lagged.merge(ages, on=["season", "player_id"], how="left") if not ages.empty
              else lagged.assign(age=np.nan))
    lagged["age_sq"] = lagged["age"] ** 2
    lagged["career_year"] = (lagged.sort_values("season_index")
                             .groupby("player_id").cumcount())

    d = lagged.dropna(subset=["gp_share_lag1", "age", "gp"]).copy()
    d["n_prior_seasons"] = 1 + d["gp_share_lag2"].notna() + d["gp_share_lag3"].notna()
    for col in LAG_COLS:
        for lag in (2, 3):
            d[f"{col}_lag{lag}"] = d[f"{col}_lag{lag}"].fillna(d[f"{col}_lag1"])

    d["season_start_date"] = d["season"].map(starts)
    d["as_of_date"] = d["season_start_date"] - pd.Timedelta(days=1)
    d["gp"] = d["gp"].astype(int)
    # The binomial denominator is "games he could have played", which is his last team's
    # schedule *except* for the 13 traded players (0.12%) whose two teams' schedules
    # overlap enough that they played more games than either. `gp > n` makes
    # `betabinom.logpmf` non-finite, and because the likelihood is a sum, thirteen rows
    # take the whole fit down — silently, at every value of ρ.
    d["team_games"] = np.maximum(d["team_games"].astype(int), d["gp"])
    d = d[d["team_games"] > 0]
    return assert_point_in_time(d.reset_index(drop=True))


def assert_point_in_time(design: pd.DataFrame) -> pd.DataFrame:
    """Every training row's `as_of_date` must precede its own season's first game.

    The leakage assertion the plan asks for, in the code path rather than only in the
    test suite, because the failure it guards against is silent: a design matrix built
    from a current-status source looks completely normal and simply scores too well.
    """
    late = design["as_of_date"] >= design["season_start_date"]
    if late.any():
        bad = design.loc[late, ["player_id", "season", "as_of_date"]].head()
        raise ValueError(
            f"{int(late.sum())} rows have as_of_date at or after their season start:\n"
            f"{bad.to_string(index=False)}")
    if design["as_of_date"].isna().any():
        raise ValueError("rows with no as_of_date: season start dates are missing")
    return design


def split_seasons(design: pd.DataFrame, test_seasons: int = TEST_SEASONS
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal walk-forward: the last `test_seasons` target seasons are held out.

    **The second frame is the held-out split and is guarded.** `split_seasons` is the one
    choke point every head goes through to reach it, so the lock lives here rather than at
    each call site — a new head cannot forget to add it. Callers that only want the
    selection frames should use `held_out.selection_split`, which never materializes the
    held-out rows at all.

    The guard is on *use*, not on the split itself: carving the frame is how a module
    discovers what to exclude. `_HeldOut` therefore raises when the rows are read rather
    than when they are separated, so `train, _ = split_seasons(...)` stays legal and
    `score(model, test)` does not.
    """
    order = sorted(design["season"].unique())
    held = set(order[-test_seasons:])
    kept = design[~design["season"].isin(held)]
    return kept, _HeldOut(design[design["season"].isin(held)])


# ── The predictive distribution ───────────────────────────────────────────────

# Guard rails only, not a prior: at ρ → 1 the beta collapses to a two-point distribution
# on 0 and n, which is numerically nasty and never a useful forecast. The fitted values
# land near 0.2 (≈17× overdispersion at n = 82), so neither bound binds.
RHO_MIN, RHO_MAX = 1e-6, 0.95
_BIG = 1e12


def _sigmoid(x: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _ab(mu: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Beta-binomial shape parameters from mean and dispersion."""
    mu = np.clip(mu, EPS, 1 - EPS)
    rho = float(np.clip(rho, RHO_MIN, RHO_MAX))
    scale = (1.0 - rho) / rho
    return mu * scale, (1.0 - mu) * scale


def _neg_loglik(y: np.ndarray, n: np.ndarray, mu: np.ndarray, rho: float) -> float:
    """Negative log-likelihood, finite everywhere.

    Guarded because the optimizers reach for extreme shape parameters where
    `betabinom.logpmf` returns nan; an unguarded nan makes L-BFGS-B's numeric gradient
    nan too, and it stops at whatever point it had reached — a silent non-convergence
    that looks like a fitted model.
    """
    a, b = _ab(mu, rho)
    ll = betabinom.logpmf(y, n, a, b)
    return _BIG if not np.all(np.isfinite(ll)) else -float(ll.sum())


def fit_dispersion(y: np.ndarray, n: np.ndarray, mu: np.ndarray) -> float:
    """MLE of ρ holding the mean function fixed.

    Fitted per model on its own training residuals, which is what makes CRPS comparable
    across models with different mean functions: each gets the dispersion its own errors
    warrant rather than a shared guess.
    """
    lo, hi = np.log(RHO_MIN / (1 - RHO_MIN)), np.log(RHO_MAX / (1 - RHO_MAX))
    best = minimize_scalar(lambda t: _neg_loglik(y, n, mu, _sigmoid(t)),
                           bounds=(lo, hi), method="bounded")
    return float(np.clip(_sigmoid(best.x), RHO_MIN, RHO_MAX))


def predictive_pmf(n: np.ndarray, mu: np.ndarray, rho: float,
                   max_games: int | None = None) -> np.ndarray:
    """(rows × games+1) pmf over games played.

    Evaluated on one grid wide enough for the longest season; mass above a row's own
    `n` is zero, so the shortened seasons cost only unused columns.
    """
    max_games = int(max_games or n.max())
    k = np.arange(max_games + 1)[None, :]
    a, b = _ab(np.asarray(mu, dtype=float), rho)
    pmf = betabinom.pmf(k, np.asarray(n)[:, None], a[:, None], b[:, None])
    return np.nan_to_num(pmf)


def crps(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Discrete CRPS in games: Σ_k (F(k) − 1{y ≤ k})².

    The metric the plan asks for, and the reason it does: it scores the whole predicted
    distribution, so a model that gets the mean right and the left tail wrong is
    penalised for it where MAE and R² are not.
    """
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    k = np.arange(pmf.shape[1])[None, :]
    return np.sum((cdf - (np.asarray(y)[:, None] <= k)) ** 2, axis=1)


def pit_values(pmf: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Randomized PIT, uniform on [0,1] iff the distribution is calibrated.

    Randomized because the target is a count: the non-randomized PIT of a discrete
    distribution is not uniform even under a perfect model, so its histogram would show
    structure that is an artifact of discreteness rather than miscalibration.
    """
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    rows = np.arange(len(y))
    y = np.asarray(y, dtype=int)
    lower = np.where(y > 0, cdf[rows, np.maximum(y - 1, 0)], 0.0)
    mass = pmf[rows, y]
    return lower + np.random.default_rng(seed).random(len(y)) * mass


def pit_table(u: np.ndarray, model: str, bins: int = 10) -> pd.DataFrame:
    counts, edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
    return pd.DataFrame({"model": model, "bin_low": edges[:-1], "bin_high": edges[1:],
                         "count": counts, "share": counts / max(len(u), 1),
                         "expected_share": 1.0 / bins})


# ── Models ────────────────────────────────────────────────────────────────────

class AvailabilityModel:
    """A mean function over games-played share, plus a fitted dispersion.

    Subclasses implement `_fit_mean` / `_predict_mean`; everything distributional is
    shared so the four candidates differ only in the mean and are otherwise scored on
    identical terms.
    """

    name = "base"

    def fit(self, train: pd.DataFrame) -> "AvailabilityModel":
        y = train["gp"].to_numpy()
        n = train["team_games"].to_numpy()
        self._fit_mean(train, y / n)
        self.rho = fit_dispersion(y, n, self.predict_mean(train))
        return self

    def _fit_mean(self, train: pd.DataFrame, target: np.ndarray) -> None:
        raise NotImplementedError

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        return predictive_pmf(df["team_games"].to_numpy(), self.predict_mean(df),
                              self.rho, max_games)


class LeagueAgeBaseline(AvailabilityModel):
    """The README's standing instruction, made concrete: shrink hard toward a league/age
    baseline rather than the player's own prior games played.

    It ignores the player entirely. Given r = 0.316 and no durability latent, that is a
    serious contender rather than a straw man, and it is the bar the others have to clear
    before any of this is worth building.
    """

    name = "league_age"

    def __init__(self, shrinkage: float = AGE_SHRINKAGE):
        self.shrinkage = shrinkage

    def _fit_mean(self, train: pd.DataFrame, target: np.ndarray) -> None:
        self.global_mean = float(np.mean(target))
        ages = train["age"].round().astype(int)
        grouped = pd.DataFrame({"age": ages, "y": target}).groupby("age")["y"]
        counts, means = grouped.count(), grouped.mean()
        self.by_age = ((counts * means + self.shrinkage * self.global_mean)
                       / (counts + self.shrinkage))

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        ages = df["age"].round().astype(int)
        return ages.map(self.by_age).fillna(self.global_mean).to_numpy()


class RidgeAvailability(AvailabilityModel):
    """Ridge on the feature block. Penalised rather than plain OLS on principle — the
    lag columns are strongly collinear by construction (`gp_share_lag1..3`)."""

    name = "ridge"

    def __init__(self, alpha: float = RIDGE_ALPHA, features: list[str] | None = None):
        self.alpha = alpha
        self.features = features or FEATURE_COLS

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.features].to_numpy(dtype=float)

    def _fit_mean(self, train: pd.DataFrame, target: np.ndarray) -> None:
        self.scaler = StandardScaler().fit(self._design(train))
        self.model = Ridge(alpha=self.alpha).fit(
            self.scaler.transform(self._design(train)), target)

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(self.scaler.transform(self._design(df)))
        return np.clip(pred, EPS, 1 - EPS)


class BetaBinomialGLM(AvailabilityModel):
    """Games played out of team games, logit mean and dispersion fitted jointly.

    The likelihood that actually matches the target: bounded, overdispersed, integer.
    Unlike the wrapped point models it is not fitted to squared error at all, so its
    dispersion and its mean inform each other.
    """

    name = "beta_binomial"

    def __init__(self, l2: float = 1.0, features: list[str] | None = None,
                 max_rounds: int = 25):
        self.l2 = l2
        self.features = features or FEATURE_COLS
        self.max_rounds = max_rounds

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[self.features].to_numpy(dtype=float))
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, train: pd.DataFrame) -> "BetaBinomialGLM":
        self.scaler = StandardScaler().fit(train[self.features].to_numpy(dtype=float))
        X = self._design(train)
        y = train["gp"].to_numpy()
        n = train["team_games"].to_numpy()

        # Alternating MLE with an analytic gradient in β. A joint fit over 17 parameters
        # with numeric gradients does not work here: the objective is a sum of 10,000
        # log-densities, so its magnitude is ~1e5 while a finite-difference step moves it
        # by ~1e-3, and L-BFGS-B stops early on gradient noise — producing a fitted model
        # whose coefficients are essentially zero.
        def objective(beta: np.ndarray, rho: float) -> tuple[float, np.ndarray]:
            mu = _sigmoid(X @ beta)
            value = _neg_loglik(y, n, mu, rho) + self.l2 * float(beta[1:] @ beta[1:])
            a, b = _ab(mu, rho)
            scale = (1.0 - float(np.clip(rho, RHO_MIN, RHO_MAX))) / float(
                np.clip(rho, RHO_MIN, RHO_MAX))
            # a + b = scale is free of mu, so its digamma terms cancel from d/dmu.
            dmu = scale * (digamma(y + a) - digamma(a) - digamma(n - y + b) + digamma(b))
            grad = -(X.T @ (dmu * mu * (1.0 - mu)))
            grad[1:] += 2.0 * self.l2 * beta[1:]
            return value, np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        beta = np.zeros(X.shape[1])
        beta[0] = np.log(max(y.sum(), 1) / max(n.sum() - y.sum(), 1))
        rho = 0.23                            # the measured overdispersion, as a start
        for _ in range(self.max_rounds):
            best = minimize(objective, beta, args=(rho,), jac=True, method="L-BFGS-B")
            beta = best.x
            new_rho = fit_dispersion(y, n, _sigmoid(X @ beta))
            if abs(new_rho - rho) < 1e-5:
                rho = new_rho
                break
            rho = new_rho

        self.beta, self.rho = beta, float(rho)
        self.converged = bool(best.success)
        return self

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self._design(df) @ self.beta)


class GBMAvailability(AvailabilityModel):
    """Gradient boosting on the same features. Reported with a shuffled null, so the
    gain ships with its own chance level."""

    name = "gbm"

    def __init__(self, features: list[str] | None = None, seed: int = 42, **params):
        self.features = features or FEATURE_COLS
        self.params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
                       "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0,
                       "random_state": seed, **params}

    def _fit_mean(self, train: pd.DataFrame, target: np.ndarray) -> None:
        import xgboost as xgb
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(train[self.features].to_numpy(dtype=float), target)

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(df[self.features].to_numpy(dtype=float))
        return np.clip(pred, EPS, 1 - EPS)


def candidates(cfg_av: dict) -> list[AvailabilityModel]:
    return [
        LeagueAgeBaseline(cfg_av.get("age_shrinkage", AGE_SHRINKAGE)),
        RidgeAvailability(cfg_av.get("ridge", RIDGE_ALPHA)),
        BetaBinomialGLM(cfg_av.get("glm_l2", 1.0)),
        GBMAvailability(seed=cfg_av.get("seed", 42)),
    ]


# ── Evaluation ────────────────────────────────────────────────────────────────

def _row(model: str, group: str, metric: str, value: float, n: int) -> dict:
    return {"model": model, "group": group, "metric": metric, "n": n, "value": value}


def evaluate(model: AvailabilityModel, frame: pd.DataFrame, max_games: int,
             seed: int = 42) -> tuple[list[dict], pd.DataFrame]:
    """CRPS, PIT and the tail — plus MAE and R² for continuity.

    Takes whatever frame it is handed rather than naming it `test`: `run` scores
    validation, `src/final_evaluation.py` scores the held-out seasons once, and both go
    through this function so the two readings are the same measurement on different rows.
    """
    test = frame
    y = test["gp"].to_numpy()
    n = test["team_games"].to_numpy()
    mu = model.predict_mean(test)
    pmf = model.predict_pmf(test, max_games)
    scores = crps(pmf, y)
    pred_games = mu * n

    share, pred_share = y / n, mu
    ss_res = float(np.sum((share - pred_share) ** 2))
    ss_tot = float(np.sum((share - share.mean()) ** 2))

    rows = [
        _row(model.name, "all", "crps_games", float(scores.mean()), len(test)),
        _row(model.name, "all", "mae_games", float(np.abs(y - pred_games).mean()),
             len(test)),
        _row(model.name, "all", "rmse_games",
             float(np.sqrt(((y - pred_games) ** 2).mean())), len(test)),
        _row(model.name, "all", "r2_gp_share",
             1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan, len(test)),
        _row(model.name, "all", "dispersion_rho", float(model.rho), len(test)),
        _row(model.name, "all", "implied_overdispersion",
             float(1 + (np.median(n) - 1) * model.rho), len(test)),
    ]

    # The left tail is the point. P(GP < t) comes off the predicted distribution, so a
    # model that hits the mean by never predicting a lost season is caught here.
    cdf = np.cumsum(pmf, axis=1)
    rotation = ((test["minutes_per_game_lag1"] >= ROTATION_MIN_MPG)
                & (test["gp_share_lag1"] >= ROTATION_MIN_GP_SHARE)).to_numpy()
    for threshold in TAIL_THRESHOLDS:
        p_below = cdf[:, threshold - 1]
        observed = (y < threshold).astype(float)
        rows += [
            _row(model.name, "all", f"brier_below_{threshold}",
                 float(np.mean((p_below - observed) ** 2)), len(test)),
            _row(model.name, "rotation", f"predicted_share_below_{threshold}",
                 float(p_below[rotation].mean()) if rotation.any() else np.nan,
                 int(rotation.sum())),
            _row(model.name, "rotation", f"observed_share_below_{threshold}",
                 float(observed[rotation].mean()) if rotation.any() else np.nan,
                 int(rotation.sum())),
        ]
    if rotation.any():
        rows.append(_row(model.name, "rotation", "crps_games",
                         float(scores[rotation].mean()), int(rotation.sum())))

    u = pit_values(pmf, y, seed)
    # One number for calibration: how far the PIT's own CDF strays from uniform.
    grid = np.linspace(0, 1, 101)
    ks = float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u) - grid)))
    rows.append(_row(model.name, "all", "pit_ks_distance", ks, len(test)))

    predictions = test[["season", "player_id", "as_of_date"]].copy()
    predictions["model"] = model.name
    predictions["gp"] = y
    predictions["team_games"] = n
    predictions["predicted_gp"] = pred_games
    predictions["crps_games"] = scores
    predictions["pit"] = u
    return rows, predictions


def gbm_shuffled_null(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
                      n_shuffles: int = 5, seed: int = 42) -> dict:
    """The GBM's CRPS against its own chance level, permuting the target.

    Routed through `feature_diagnostics.above_null` so this score carries a chance level
    the same way every other importance number in the project does — and so the permuted
    marginal is named rather than left implicit.
    """
    def statistic(values: dict) -> float:
        shuffled = train.copy()
        shuffled["gp"] = values["gp"]
        model = GBMAvailability(seed=seed).fit(shuffled)
        return -float(crps(model.predict_pmf(val, max_games),
                           val["gp"].to_numpy()).mean())

    out = above_null(statistic, {"gp": train["gp"].to_numpy()}, "gp",
                     n_shuffles=n_shuffles, seed=seed)
    # Reported as CRPS (lower is better); the statistic is negated so "above null" reads
    # in the usual direction.
    return {"crps": -out["statistic"], "null_crps": -out["null_mean"],
            "null_sd": out["null_sd"], "improvement": out["above_null"],
            "n_shuffles": out["n_shuffles"]}


# Resamples for the ladder's paired interval. 4,000 puts the Monte Carlo error on a 95%
# endpoint well inside the last digit the docs quote.
BOOTSTRAP_DRAWS = 4000
GP_QUARTILES = 4


def ladder_comparison(predictions: pd.DataFrame, reference: str = "beta_binomial",
                      draws: int = BOOTSTRAP_DRAWS, seed: int = 42) -> pd.DataFrame:
    """Each candidate's CRPS against the shipped one, paired row by row.

    The ladder is a **selection**, so the margin between its top two rows decides which
    mean function ships, and a difference of means over 883 player-seasons is not evidence
    on its own. This is what says whether an ordering is real: the same rows scored by both
    models, resampled together, so the player-to-player variation that dominates CRPS
    cancels instead of being counted twice.

    It earns its place because the ordering *reversed* when this head moved off the test
    split on 2026-08-08 — the GBM went from third to first — and the interval is what turns
    that from a new verdict into a measured non-result.

    The realized-GP quartile rows are the second half of the answer and the more useful
    one. A mean over the whole frame hides *where* a model is better, and this head exists
    for the left tail: a candidate that wins overall by predicting ordinary seasons well
    and loses on the seasons that fell apart is not the one to ship, whatever its mean.
    """
    wide = predictions.pivot_table(index=["season", "player_id"], columns="model",
                                   values="crps_games")
    if reference not in wide.columns:
        raise ValueError(f"no {reference!r} row to compare against; got "
                         f"{sorted(wide.columns)}")
    gp = (predictions[predictions["model"] == reference]
          .set_index(["season", "player_id"])["gp"].reindex(wide.index))
    # Quartiles of the *realized* outcome, not of the prediction: the question is how a
    # model does on the seasons that actually went badly.
    quartile = pd.qcut(gp, GP_QUARTILES, labels=False, duplicates="drop")

    rng = np.random.default_rng(seed)
    n = len(wide)
    index = rng.integers(0, n, size=(draws, n))
    rows = []
    for model in sorted(wide.columns):
        delta = (wide[model] - wide[reference]).to_numpy()
        boot = delta[index].mean(axis=1)
        rows.append({
            "model": model, "group": "all", "n": n,
            "crps_games": float(wide[model].mean()),
            "delta_vs_reference": float(delta.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "p_better": float((boot < 0).mean()),
            "share_rows_better": float((delta < 0).mean()),
        })
        for q in sorted(pd.unique(quartile.dropna())):
            rows_q = (quartile == q).to_numpy()
            rows.append({
                "model": model, "group": f"gp_q{int(q) + 1}", "n": int(rows_q.sum()),
                "crps_games": float(wide[model].to_numpy()[rows_q].mean()),
                "delta_vs_reference": float(delta[rows_q].mean()),
                "ci_lo": np.nan, "ci_hi": np.nan, "p_better": np.nan,
                "share_rows_better": float((delta[rows_q] < 0).mean()),
            })
    out = pd.DataFrame(rows)
    out["reference"] = reference
    # An interval straddling zero is the whole point of computing one, so it is a column
    # rather than something a reader has to derive from two others.
    out["distinguishable"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    return out


def workload_ablation(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
                      seed: int = 42) -> pd.DataFrame:
    """Validation CRPS with and without the playoff/mileage block, on one fitted head.

    The plan's decision rule is a proper distributional metric, not in-sample R², so a
    feature block gets added only if it moves CRPS. Everything is held fixed except the
    feature list, so the contrast is the block and nothing else.

    Split three ways because the two halves of the block have opposite mechanisms:
    playoff participation marks a good player on a good team (selection), while
    cumulative mileage points the way fatigue would.

    **This decides a feature block, so it reads validation.** It decided one on the test
    split until 2026-08-08, which is the class of thing `src/models/held_out.py` exists to
    prevent — the block was adopted on a held-out CRPS margin of −0.119 games.
    """
    base = [c for c in FEATURE_COLS if c not in WORKLOAD_COLS]
    variants = {
        "baseline": base,
        "plus_playoff_workload": base + WORKLOAD_COLS,
        "plus_playoff_only": base + [c for c in WORKLOAD_COLS if c != "career_minutes_lag1"],
        "plus_career_minutes_only": base + ["career_minutes_lag1"],
    }
    rows = []
    for label, features in variants.items():
        model = BetaBinomialGLM(features=features).fit(train)
        model_rows, _ = evaluate(model, val, max_games, seed)
        keep = {r["metric"]: r["value"] for r in model_rows if r["group"] == "all"}
        rows.append({"variant": label, "n_features": len(features),
                     "crps_games": keep["crps_games"], "mae_games": keep["mae_games"],
                     "r2_gp_share": keep["r2_gp_share"],
                     "pit_ks_distance": keep["pit_ks_distance"]})
    out = pd.DataFrame(rows)
    baseline = float(out.loc[out["variant"] == "baseline", "crps_games"].iloc[0])
    out["crps_vs_baseline"] = out["crps_games"] - baseline
    return out


# The continuous columns where a nonlinear response is plausible a priori — age and
# career mileage for a career arc, prior MPG and prior availability for floor/ceiling
# compression. `age_sq` already exists in FEATURE_COLS, so `age` enters the linear
# baseline with a quadratic already fitted.
NONLINEAR_CANDIDATES = ["age", "career_minutes_lag1", "minutes_per_game_lag1",
                        "gp_share_lag1", "career_year", "total_minutes_lag1",
                        "playoff_mpg_lag1", "playoff_games_lag1",
                        "playoff_minutes_share_lag1"]
SPLINE_KNOTS = 5


def expand_basis(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
                 kind: str = "spline", n_knots: int = SPLINE_KNOTS
                 ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add a quadratic or natural-spline basis for `cols`, fitted on **train only**.

    Knot placement is estimated from the training quantiles and applied to test. Fitting
    the transformer on the pooled frame would leak the test distribution into the basis —
    a subtle version of exactly what `assert_point_in_time` guards against elsewhere.

    Cubic B-splines with `extrapolation="linear"` rather than raw polynomials: a degree-3
    polynomial extrapolates wildly past the data range, which for `career_minutes` (p90
    far above the median) is where the sparse rows live. Returns the frames plus the names
    of the columns that replace the originals.
    """
    if kind not in ("quadratic", "spline"):
        raise ValueError(f"kind must be 'quadratic' or 'spline'; got {kind!r}")
    tr, te = train.copy(), test.copy()
    names: list[str] = []
    for col in cols:
        if kind == "quadratic":
            name = f"{col}__sq"
            tr[name] = tr[col].to_numpy(float) ** 2
            te[name] = te[col].to_numpy(float) ** 2
            names.append(name)
            continue
        st = SplineTransformer(n_knots=n_knots, degree=3, extrapolation="linear",
                               include_bias=False)
        basis_tr = st.fit_transform(train[[col]].to_numpy(float))
        basis_te = st.transform(test[[col]].to_numpy(float))
        for j in range(basis_tr.shape[1]):
            name = f"{col}__s{j}"
            tr[name], te[name] = basis_tr[:, j], basis_te[:, j]
            names.append(name)
    return tr, te, names


def _nonlinear_variants(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
                        ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """The linear baseline plus quadratic and spline expansions of `cols`."""
    out = {"linear": (train, test, list(FEATURE_COLS))}
    quad_cols = [c for c in cols if c != "age"]      # age_sq is already in the baseline
    tr, te, names = expand_basis(train, test, quad_cols, kind="quadratic")
    out["quadratic"] = (tr, te, list(FEATURE_COLS) + names)
    for k in (4, 5):
        tr, te, names = expand_basis(train, test, cols, kind="spline", n_knots=k)
        keep = [c for c in FEATURE_COLS if c not in cols and c != "age_sq"]
        out[f"spline_k{k}"] = (tr, te, keep + names)
    return out


def nonlinearity_ablation(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
                          seed: int = 42, cols: list[str] | None = None) -> pd.DataFrame:
    """Does a nonlinear response in age / mileage / prior minutes buy anything?

    **Selection happens on validation, and there is no longer a test column to be tempted
    by.** That protocol was always this function's point; until 2026-08-08 it made the
    point while still printing the test column beside it, which is a weaker way to make it.
    The rows this fits and scores are unchanged — `held_out.selection_split` hands back
    exactly the frames the private inner split used to carve — so every validation figure
    is bit-for-bit what it was.

    The lesson the retired column taught is worth keeping in prose, because it is the
    sharpest case in the project. On the held-out rows a *paired* bootstrap put the
    quadratic expansion's CRPS gain at −0.047, 95% interval [−0.079, −0.015],
    P(Δ<0) = 99.7% — and it was still a false positive, because on validation the linear
    model wins outright. A paired interval says a difference is consistent *within one
    sample*; it says nothing about whether the sample was representative.

    The answer for **games played is a null** — the GBM arm reaching the same verdict from
    a different direction (a fully nonparametric learner on the same features still loses
    to the linear GLM) is the corroboration. For **minutes per game it is not a null**,
    and `minutes_per_game_lag1` carries essentially all of it; see `CLAUDE.md`.
    """
    cols = cols or [c for c in NONLINEAR_CANDIDATES if c in train.columns]
    variants = _nonlinear_variants(train, val, cols)

    rows = []
    for label, (v_tr, v_te, v_features) in variants.items():
        model = BetaBinomialGLM(features=v_features).fit(v_tr)
        model_rows, _ = evaluate(model, v_te, max_games, seed)
        keep = {r["metric"]: r["value"] for r in model_rows if r["group"] == "all"}
        rows.append({"variant": label, "n_features": len(v_features),
                     "val_crps_games": keep["crps_games"],
                     "val_r2_gp_share": keep["r2_gp_share"]})

    out = pd.DataFrame(rows)
    base_val = float(out.loc[out["variant"] == "linear", "val_crps_games"].iloc[0])
    out["val_vs_linear"] = out["val_crps_games"] - base_val
    out["selected"] = out["val_crps_games"] == out["val_crps_games"].min()
    return out


MINUTES_TARGET = "minutes_per_game"
MINUTES_RIDGE_ALPHA = 10.0


def _ridge_r2(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
              target: str) -> float:
    scaler = StandardScaler().fit(train[features].to_numpy(dtype=float))
    model = Ridge(alpha=MINUTES_RIDGE_ALPHA).fit(
        scaler.transform(train[features].to_numpy(dtype=float)),
        train[target].to_numpy(dtype=float))
    pred = model.predict(scaler.transform(test[features].to_numpy(dtype=float)))
    y = test[target].to_numpy(dtype=float)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def minutes_nonlinearity_probe(train: pd.DataFrame, val: pd.DataFrame,
                               cols: list[str] | None = None) -> pd.DataFrame:
    """The same nonlinearity question, asked of **minutes per game** instead of games.

    A probe rather than a head: the minutes model is not built (`docs/availability-plan.md`
    lists it as "not blocked, just not started"), and this exists because the nonlinearity
    answer *differs between the two targets* and that is worth knowing before it is built.
    A plain ridge stands in, so the R² here is not a claim about how good a minutes head
    can be — only about whether a curved response beats a straight one.

    Unlike the games-played arm this is **not** a null. The per-column rows say where it
    comes from, and it is not the career arc: a spline on `age` is *worse* than the
    existing `age + age_sq`, while `minutes_per_game_lag1` carries essentially all of the
    gain. The mechanism is a floor, not a ceiling — mean reversion from a low prior MPG is
    far steeper than from a high one.

    **The `replicates` column is gone with the test column, and it was never what its name
    claimed.** It required val and test to move the same way, but those two differed in
    training data as well as in scored rows, so agreement between them was not a
    replication and disagreement was not a refutation — the confound
    `src/models/held_out.py` was written to stop being read as one. The contrast that
    carries the finding is between the two *targets* on one frame: games played is a null
    here and minutes per game is not, and both are now measured on the same validation rows
    by the same code.
    """
    cols = cols or [c for c in NONLINEAR_CANDIDATES if c in train.columns]
    rows = []

    for label, (v_tr, v_te, v_features) in _nonlinear_variants(train, val, cols).items():
        rows.append({"scope": "variant", "name": label, "n_features": len(v_features),
                     "val_r2": _ridge_r2(v_tr, v_te, v_features, MINUTES_TARGET)})

    # One column at a time, so the gain is attributed rather than just observed.
    base_val = _ridge_r2(train, val, list(FEATURE_COLS), MINUTES_TARGET)
    for col in cols:
        drop = {col} | ({"age_sq"} if col == "age" else set())
        keep = [c for c in FEATURE_COLS if c not in drop]
        v_tr, v_te, names = expand_basis(train, val, [col], "spline")
        rows.append({"scope": "column", "name": col, "n_features": len(keep + names),
                     "val_r2": _ridge_r2(v_tr, v_te, keep + names, MINUTES_TARGET)})

    out = pd.DataFrame(rows)
    out["val_vs_linear"] = np.where(out["scope"] == "column",
                                    out["val_r2"] - base_val, np.nan)
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    cfg_av = cfg.get("features", {}).get("availability", {})
    test_seasons = int(cfg_av.get("test_seasons", TEST_SEASONS))
    seed = int(cfg_av.get("seed", 42))
    n_shuffles = int(cfg_av.get("n_shuffles", 5))

    panel = build_panel(seasons, raw_dir)
    frame = season_availability(panel, "full")
    design = build_design(frame, seasons, raw_dir, season_start_dates(panel))
    train, val = selection_split(design, test_seasons)
    max_games = int(design["team_games"].max())

    print(f"Availability design: {len(design):,} player-seasons, "
          f"{len(train):,} train / {len(val):,} validation "
          f"({', '.join(sorted(val['season'].unique()))})")
    print("  The test split is LOCKED (src/models/held_out.py). This module picks the mean "
          "function,\n  a feature block and a basis — three decisions — so it reads "
          "VALIDATION and nothing else.\n  The held-out reading is taken once, by "
          "`make final-evaluation`.")
    print(f"  as_of_date range {design['as_of_date'].min().date()} → "
          f"{design['as_of_date'].max().date()}, every row before its own season start")
    print("  Unweighted by design — see CLAUDE.md; minutes weighting halves the ceiling "
          "here.")

    rows, pit_frames, prediction_frames = [], [], []
    for model in candidates(cfg_av):
        model.fit(train)
        model_rows, predictions = evaluate(model, val, max_games, seed)
        rows += model_rows
        prediction_frames.append(predictions)
        pit_frames.append(pit_table(predictions["pit"].to_numpy(), model.name))

    metrics = pd.DataFrame(rows)
    table = (metrics[metrics.group == "all"]
             .pivot_table(index="model", columns="metric", values="value"))
    order = ["crps_games", "mae_games", "r2_gp_share", "pit_ks_distance",
             "implied_overdispersion"]
    print("\nValidation scores (CRPS in games, lower is better):")
    print(table[order].sort_values("crps_games").round(4).to_string())

    best = table["crps_games"].idxmin()
    baseline_crps = float(table.loc["league_age", "crps_games"])
    gain = baseline_crps - float(table.loc[best, "crps_games"])
    print(f"\nBest: {best} (CRPS {table.loc[best, 'crps_games']:.4f}) against the "
          f"league/age baseline's {baseline_crps:.4f} — {gain:+.4f} games")
    print("  R² here is out of sample and not season-absorbed, so it is NOT comparable to "
          "the profile's\n  in-sample season-absorbed ceiling of 0.236 — a different "
          "quantity, not a ceiling beaten.")

    ladder = ladder_comparison(pd.concat(prediction_frames, ignore_index=True),
                               reference=SHIPPED_MODEL, seed=seed)
    print(f"\nPaired against the shipped head ({SHIPPED_MODEL}) on the same rows — "
          f"the ORDERING is not the finding:")
    print(ladder[ladder["group"] == "all"]
          .drop(columns=["group", "reference"]).round(4).to_string(index=False))
    print(f"  A ladder is a selection, so a difference of means over {len(val):,} "
          f"player-seasons decides\n  nothing on its own. An interval that straddles zero "
          f"says the ordering is not evidence.")
    print("\n  The same deltas by REALIZED games-played quartile (negative = better than "
          "the shipped head):")
    print(ladder[ladder["group"] != "all"]
          .pivot_table(index="model", columns="group", values="delta_vs_reference")
          .round(3).to_string())
    print("  q1 is the population this head exists for. A candidate that wins overall on "
          "ordinary\n  seasons and loses on the seasons that fell apart is not the one to "
          "ship.")

    null = gbm_shuffled_null(train, val, max_games, n_shuffles, seed)
    rows += [_row("gbm", "null", "crps_shuffled_target", null["null_crps"], len(val)),
             _row("gbm", "null", "crps_improvement_over_null", null["improvement"],
                  len(val)),
             _row("gbm", "null", "null_sd", null["null_sd"], len(val))]
    print(f"GBM shuffled-target null: CRPS {null['null_crps']:.4f} "
          f"(sd {null['null_sd']:.4f}) — the GBM improves on chance by "
          f"{null['improvement']:.4f} games")

    print("\nLeft tail among established rotation players (predicted vs observed):")
    tail = (pd.DataFrame(rows).query("group == 'rotation'")
            .pivot_table(index="model", columns="metric", values="value"))
    tail_cols = [c for c in tail.columns if "share_below" in c]
    print(tail[sorted(tail_cols)].round(3).to_string())
    print("  A model that matches the mean by never predicting a lost season shows up "
          "here as a\n  predicted share far below the observed one.")

    ablation = workload_ablation(train, val, max_games, seed)
    print("\nPlayoff / mileage workload ablation (beta-binomial head, CRPS in games):")
    print(ablation.round(4).to_string(index=False))
    print("  Playoff participation predicts *better* next-season availability, not "
          "worse — selection\n  (good player on a good team) beats fatigue. "
          "`career_minutes` is the one column that\n  points the way fatigue would.")

    nonlinear = nonlinearity_ablation(train, val, max_games, seed)
    print("\nNonlinear response (quadratic / spline) on games played:")
    print(nonlinear.round(4).to_string(index=False))
    print("  A null: no curved variant beats linear on validation. The retired test column "
          "preferred\n  every one of them, and a paired bootstrap on those rows put the "
          "quadratic gain at\n  −0.047 [−0.079, −0.015] — a false positive with a "
          "convincing interval around it.\n  Games played stays linear.")

    minutes = minutes_nonlinearity_probe(train, val)
    print("\nThe same question on MINUTES PER GAME (ridge probe; the head is not built):")
    print(minutes.round(4).to_string(index=False))
    print("  Here it is NOT a null — the contrast between the two targets on the SAME rows "
          "is the\n  finding. And it is not the career arc: splining `age` is worse than "
          "the existing\n  age + age_sq, while `minutes_per_game_lag1` carries essentially "
          "all of it.")

    print("\nPIT calibration (share per decile; 0.100 is uniform):")
    pit = pd.concat(pit_frames, ignore_index=True)
    print(pit.pivot_table(index="model", columns="bin_low", values="share")
          .round(3).to_string())

    artifacts = {
        "metrics": (pd.DataFrame(rows), out_dir / "availability_metrics.csv"),
        "ladder": (ladder, out_dir / "availability_ladder_comparison.csv"),
        "ablation": (ablation, out_dir / "availability_workload_ablation.csv"),
        "nonlinearity": (nonlinear, out_dir / "availability_nonlinearity.csv"),
        "minutes_nonlinearity": (minutes,
                                 out_dir / "availability_minutes_nonlinearity.csv"),
        "pit": (pit, out_dir / "availability_pit.csv"),
        "predictions": (pd.concat(prediction_frames, ignore_index=True),
                        out_dir / "availability_predictions.csv"),
    }
    paths = {}
    for name, (frame, dest) in artifacts.items():
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")

    print("\nThe spell simulator is built only if these are beaten — see "
          "docs/availability-plan.md.")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
