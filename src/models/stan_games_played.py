"""The games-played head as a fitted spell process — entry x exit x a within-tenure chain.

`stan_availability.py` answers *how many* games a player misses. This answers *which*, and
the simulator needs which: the contest scores the best 7 of a 16-man roster each week, and
a max over a correlated set is not a function of the marginals. Two players who miss the
same fortnight are a very different roster from two who miss different fortnights at
identical marginal availability.

`games_played.py` is the numpy reference — the collapse, the spell classes, the closed-form
duration fits, the simulator and Gate 0 — and it runs without a CmdStan toolchain. This
module fits the same structure by NUTS. `evaluate`, `crps`, `pit_values`, `pit_table`,
and `build_design` are **imported** from `models.availability` rather than reimplemented,
so a metric difference between this head and the incumbent cannot be a
metric-implementation difference. The split comes from `held_out.selection_split`, which
never materializes the held-out rows at all.

## Four heads, three `.stan` files between them, and only one is new

| head | target | trials | source |
|---|---|---|---|
| entry | games before the first appearance | `team_games - 1` | `betabinomial_glm.stan` |
| exit | games after the last appearance | `team_games - 1 - pre` | `betabinomial_glm.stan` |
| onset | absences begun, within the tenure | at-risk transitions | `betabinomial_glm.stan` |
| duration | spell length | — | `betageometric_duration.stan` |

The first three are the *same likelihood with different data*, which is this project's
factorization argument as code — the same reason `betabinomial_glm.stan` already serves
availability, minutes and the three conversion heads. They are fitted separately because
the parameter blocks are distinct and the priors independent, so the joint posterior
factorizes exactly and separate fits recover the identical posterior a joint model would.

## What the arms are, and which of them can ship

- **A0 `floor`** — `BetaBinomialGLM` refit in-process. Not a candidate: it is the incumbent,
  and it stays permanently as the bar. Its CRPS must reproduce 10.795 or the harness is
  wrong before any challenger is read.
- **A1 `within_tenure`** — onset + duration with the tenure held at its **observed** value.
  An oracle, so it cannot ship; it exists so that a loss here proves the process class is
  wrong before A2-A4 are paid for.
- **A2 `full_window`** — + fitted entry and exit heads. The shippable 30-season head.
- **A3 `three_state`** — A2 with the tenure identified by the box-score `status` rather than
  structurally, on 2006-07+ only. Scored on its own rows *and* against A2 restricted to
  them, because a CRPS on a different row set is not comparable to 10.795.
- **A4 `duration_covariates`** — A2 with the feature block on the duration head.
- **B `calibrated_fallback`** — option (b): the same simulator with hazards *inverted from*
  the incumbent's marginal rather than fitted. Not selectable, because it reproduces that
  marginal by construction and choosing it on a marginal metric would be choosing it for
  the one thing it does not decide. It ships beside the fitted arms either way, since it is
  the invariant every one of them should satisfy at its own measured clustering.

**Selection reads the validation column only.** This repo has shipped one test-selected
false positive (`nonlinearity_ablation`, paired bootstrap P(delta<0) = 99.7%, did not
replicate) and caught a second (`trend_x_role`, the best two arms on test and the worst two
on validation) — **both in this same head**.

## The win condition is the tail, and that was decided in writing before the sweep

`docs/games-played-plan.md` settles it: the dispersion budget is **over-supplied**, not
short. Clustering on the matched population supplies C = 9.58 of the 22.70x GP
overdispersion, and the composition is additive — `inflation = C + rho*(n - C)` — so
stacking the incumbent's fitted rho on top predicts 29.6, a 30% overshoot. GP-marginal CRPS
is therefore expected to be a wash at best and is treated as a **non-regression bar**. The
head is judged on the left tail, where the incumbent reads 15.0% / 34.8% against an
observed 11.8% / 36.9%.

**If Gate D fails this lands in option (b), not in the bin.** That is the same simulator
with hazards set by `games_played.closed_form_calibration`, which inverts the variance
identity to reproduce the incumbent's marginal by construction while getting the game-level
clustering right. A documented branch, not a rescue.

## ⭐ What happened: the branch was taken, and it is the better model

**Gate D failed for the fitted arms and passed for the calibration, so option (b) is what
ships.** Measured 2026-08-05, 25 fits, 0 divergences, max R-hat 1.0071, 42.7 min:

- the selected fitted arm (`duration_covariates`) scores CRPS **10.8026** against the
  10.795 bar — a fail — while cutting the tail error to 0.0192;
- `calibrated_fallback` scores **10.7939**, PIT KS **0.0907**, tail error **0.0174** against
  the incumbent's 0.0264. All three bars clear.

**Two results are worth carrying out of this module.** First, the oracle arm: given the
*observed* tenure the within-tenure chain scores **7.0391** against the season-level
beta-binomial's 10.9870 on the same rows, so the process class is right by a wide margin and
the entire difficulty is predicting entry and exit from preseason covariates — mid-season
roster churn, already out of scope. Second, and the reason the fallback wins: it inherits a
marginal a validated head already calibrated and spends its own structure on the *shape*,
where the fitted arms had to earn both at once and paid for the tenure factors in accuracy.
**The coordinate change beats the fitting**, the same shape as the shot-attempt basis result.

So the fallback is not a consolation prize and is not selected on a marginal metric — it is
deliberately *not* selectable, because reproducing the incumbent's marginal by construction
is exactly the property that would make choosing it on CRPS circular.

Usage:
    python -m src.models.stan_games_played
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.availability import with_lags
from src.features.availability import (CELL_KEYS, collapse_transitions,
                                       spell_classes)
from src.models.availability import (EPS, FEATURE_COLS, RHO_MAX, RHO_MIN,
                                     AvailabilityModel, BetaBinomialGLM,
                                     evaluate, pit_table)
from src.models.held_out import assert_unlocked, selection_split
from src.models.games_played import (allocate_spells, beta_shapes,
                                     closed_form_calibration,
                                     curve_still_falling, duration_rows,
                                     fit_beta_geometric, gp_pmf,
                                     spell_lengths_from,
                                     hazard_by_streak, multi_team_seasons,
                                     process_frame, rotation_population,
                                     order_by_streak, simulate_calibrated, simulate_gp,
                                     simulated_hazard_by_streak,
                                     variance_inflation)
from src.models.stan_availability import availability_design
from src.models.stan_utils import (compile_model, diagnostics_frame, pi_block,
                                   posterior, prior_sd_for_l2, rho_block, sample,
                                   standardized, thin, warn_if_unconverged)

BINOMIAL_MODEL = "betabinomial_glm"
DURATION_MODEL = "betageometric_duration"

GLM_L2 = 1.0
INTERCEPT_SCALE = 5.0
KAPPA_SCALE = 25.0
TEST_SEASONS = 2

# Posterior draws pushed through the simulator, and simulated seasons per draw. The pmf's
# Monte Carlo error at 200 x 50 is ~0.003 on any single cell and far less on CRPS, which
# integrates over the whole distribution.
PREDICTIVE_DRAWS = 200
SIM_SEASONS = 50
# Draws simulated at once. Purely a memory knob: 20 x 50 x 911 chains of int64 is ~7 MB per
# working array, where all 200 at once would be ~73 MB across a dozen of them.
DRAW_CHUNK = 20

STATUS_FIRST_SEASON = "2006-07"
MIN_STATUS_COVERAGE = 0.9

# Arms in ladder order. `selectable` marks the ones that could actually ship: the floor is
# the incumbent and A1 holds the tenure at its observed value, so neither is a candidate
# however well it scores.
ARMS = {
    "floor": {"selectable": False, "oracle_tenure": False},
    "within_tenure": {"selectable": False, "oracle_tenure": True},
    "full_window": {"selectable": True, "oracle_tenure": False},
    "three_state": {"selectable": True, "oracle_tenure": False},
    "duration_covariates": {"selectable": True, "oracle_tenure": False},
}

# Arms whose games-played predictive is the incumbent's BY CONSTRUCTION, so a marginal
# metric cannot distinguish them from it and selecting on one would be circular.
MARGINAL_INHERITING_ARMS = ("hybrid",)

# Gate D's bars as `docs/games-played-plan.md` originally stated them — the incumbent's
# figures on the TEST split. They are kept only as a record: the gate now derives its bars
# from the incumbent's row on whichever split it is scoring, because a hard-coded 10.795 is
# a test-set number wearing a threshold's clothes, and using it settled the ship decision on
# a split that is not allowed to make it. See `_gate_d`.
HISTORICAL_TEST_BARS = {"crps": 10.795, "pit_ks": 0.096}


# ── The design ────────────────────────────────────────────────────────────────

def games_played_design(cfg: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """The incumbent's design frame, plus the three process targets per player-season.

    Built on `availability_design` deliberately: it is the frame that carries `as_of_date`
    and runs `assert_point_in_time`, so this head inherits the leakage guard rather than
    reimplementing it beside it, and its rows and denominators are exactly the ones the
    10.795 CRPS was measured on.

    Multi-team player-seasons keep their row and lose their process targets. They are
    excluded from every fit — the process runs per (player, team) and `gp` is per
    (player, season), so simulating per cell and summing double-counts a traded player's
    absences — and included in every evaluation, so CRPS stays comparable to the incumbent's
    on all 911 test rows. Prediction is unaffected because every covariate is
    player-season level.
    """
    design = availability_design(cfg)
    process = process_frame(panel)
    single = process[~process["multi_team"]]

    keep = ["season", "player_id", "pre_tenure", "post_tenure", "tenure_games",
            "entry_trials", "exit_trials", "onsets", "at_risk_played",
            "recoveries", "at_risk_missed"]
    out = design.merge(single[keep], on=["season", "player_id"], how="left")
    out["fittable"] = out["at_risk_played"].notna()

    # The two frames derive `team_games` independently — the design from
    # `season_availability` over the player's last team, this from the panel per cell — so
    # a disagreement means one of them has changed underneath the other.
    check = out[out["fittable"]].merge(
        single[["season", "player_id", "team_games"]], on=["season", "player_id"],
        how="left", suffixes=("", "_process"))
    gap = (check["team_games"] - check["team_games_process"]).abs()
    if (gap > 0).any():
        raise ValueError(
            f"{int((gap > 0).sum())} single-team rows disagree on team_games between the "
            f"availability design and the panel process frame (max gap {gap.max():.0f}) — "
            f"the two constructions have diverged.")
    return out


def onset_rate_lags(design: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Prior-season onset rate and its exposure — the onset head's no-fit floor.

    Gate B is "does the fitted onset head clear a carry-forward", and a carry-forward of a
    *proportion* has to be shrunk or it is meaningless: a player whose prior season carried
    two at-risk transitions and zero onsets has a prior rate of exactly 0.000, and a
    beta-binomial likelihood at p = 0 on a season of real exposure is non-finite. That is
    `component_rates`' conversion-floor lesson arriving as a benchmark requirement, so the
    floor is an empirical-Bayes shrink with one constant fitted on train only.
    """
    rates = design[["season", "player_id", "onsets", "at_risk_played"]].copy()
    lagged = with_lags(rates, seasons, ["onsets", "at_risk_played"], max_lag=1)
    return lagged[["season", "player_id", "onsets_lag1", "at_risk_played_lag1"]]


def rostered_tenure(panel: pd.DataFrame, first_season: str = STATUS_FIRST_SEASON,
                    min_coverage: float = MIN_STATUS_COVERAGE) -> pd.DataFrame:
    """A3's tenure: the window in which the box score says he was on the roster.

    The structural proxy — first to last appearance — cannot see the highest-value
    population in the whole head: the **16.75%** of full-window missed games that fall
    outside the appearance window while the player was *still rostered*. Those are
    season-ending and preseason injuries, and the proxy files them under "not on the team".
    `not_rostered` is 98.5%-per-game persistent, so treating it as absorbing is the right
    idealization and the difference is worth measuring rather than assuming.

    Restricted to seasons the backfill covers, and to cells whose coverage clears
    `min_coverage`. A cell measured where the backfill has not run would report "never
    rostered", which is a statement about the backfill.
    """
    sel = panel[panel["season"] >= first_season].copy()
    if sel.empty:
        return pd.DataFrame(columns=CELL_KEYS + ["pre_tenure", "post_tenure",
                                                 "tenure_games", "team_games"])
    coverage = (sel.groupby(CELL_KEYS, as_index=False)
                .agg(status_coverage=("status_covered", "mean")))
    sel = sel.merge(coverage, on=CELL_KEYS, how="left")
    sel = sel[sel["status_coverage"] >= min_coverage]
    if sel.empty:
        return pd.DataFrame(columns=CELL_KEYS + ["pre_tenure", "post_tenure",
                                                 "tenure_games", "team_games"])

    sel["rostered"] = (~sel["status"].isin(["not_rostered", "unknown"])).astype(int)
    on_roster = sel[sel["rostered"] == 1]
    span = (on_roster.groupby(CELL_KEYS, as_index=False)
            .agg(entry_index=("team_game_index", "min"),
                 exit_index=("team_game_index", "max")))
    schedule = (sel.groupby(CELL_KEYS, as_index=False)
                .agg(team_games=("team_game_index", lambda s: int(s.max()) + 1)))
    out = span.merge(schedule, on=CELL_KEYS, how="inner")
    out["pre_tenure"] = out["entry_index"]
    out["post_tenure"] = out["team_games"] - 1 - out["exit_index"]
    out["tenure_games"] = out["exit_index"] - out["entry_index"] + 1
    out["entry_trials"] = np.maximum(out["team_games"] - 1, 0)
    out["exit_trials"] = np.maximum(out["team_games"] - 1 - out["pre_tenure"], 0)
    return out


def rostered_process(panel: pd.DataFrame, tenure: pd.DataFrame) -> dict:
    """Onset counts and spells inside the *rostered* window rather than the appearance one.

    The same two-state chain on a wider interval: what the structural proxy files as a
    crude entry/exit factor for a player who tore an ACL in March becomes a modelled
    absence, because he stayed on the roster all along.

    Returns the counts frame and a *rewritten panel*, because the spells have to be
    re-extracted over the new window. `collapse_transitions` and `spell_classes` both
    dispatch on `in_appearance_window`, so the rostered window is expressed by rewriting
    that flag rather than by a third code path through both functions.
    """
    if tenure.empty:
        return {"counts": tenure, "panel": panel.iloc[:0]}
    sel = panel.merge(tenure[CELL_KEYS + ["entry_index", "exit_index"]],
                      on=CELL_KEYS, how="inner")
    inside = sel[(sel["team_game_index"] >= sel["entry_index"])
                 & (sel["team_game_index"] <= sel["exit_index"])].copy()
    inside["in_appearance_window"] = 1
    counts = collapse_transitions(inside, "appearance")
    merged = tenure.merge(
        counts[CELL_KEYS + ["onsets", "at_risk_played", "recoveries",
                            "at_risk_missed"]], on=CELL_KEYS, how="left")
    return {"counts": merged, "panel": inside}


# ── The heads ─────────────────────────────────────────────────────────────────

class BetaBinomialHead:
    """One beta-binomial GLM — the entry, exit and onset heads are this with different data.

    Deliberately not a subclass of `AvailabilityModel`: that class's contract is a *games
    played* predictive, and these three are components of one. `SpellProcess` is the thing
    that satisfies the contract.
    """

    def __init__(self, features: list[str], name: str, l2: float = GLM_L2,
                 chains: int = 4, warmup: int = 1000, samples: int = 1000,
                 seed: int = 42):
        self.features, self.name, self.l2 = features, name, l2
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed

    def fit(self, train: pd.DataFrame, y_col: str, n_col: str) -> "BetaBinomialHead":
        y = train[y_col].to_numpy(int)
        n = train[n_col].to_numpy(int)
        if (y > n).any() or (y < 0).any() or (n < 0).any():
            bad = int(((y > n) | (y < 0) | (n < 0)).sum())
            raise ValueError(
                f"{self.name}: {bad} rows violate 0 <= {y_col} <= {n_col}. The "
                f"beta-binomial is undefined there and the summed target is non-finite at "
                f"every rho, which under HMC poisons the trajectory rather than merely "
                f"stopping an optimizer.")
        (X,), self.scaler = standardized(train, [train], self.features)

        share = float(np.clip(y.sum() / max(n.sum(), 1), EPS, 1 - EPS))
        model = compile_model(BINOMIAL_MODEL)
        fit, self.diagnostics = sample(
            model,
            {"N": len(train), "K": X.shape[1], "X": X, "n": n.tolist(), "y": y.tolist(),
             "beta_scale": prior_sd_for_l2(self.l2),
             "intercept_scale": INTERCEPT_SCALE,
             "S": 0, "season_idx": [0] * len(train), "year_sd_scale": 0.25,
             # One dispersion for every row: `rho_block()` with no bins is the shared-rho
             # model exactly. Entry, exit and onset are counts over a schedule rather than
             # over a role, so there is no bin variable here to grade on.
             **rho_block(len(train)),
             # And no low-availability mixture: `pi_block()` with no design is `P = 0`,
             # which makes that block's parameters zero-length and this target the one
             # entry, exit and onset have always fitted. The spell process is the
             # STRUCTURAL alternative to a mixture, so it is the last head that would
             # want one bolted on.
             **pi_block(len(train))},
            chains=self.chains, warmup=self.warmup, samples=self.samples,
            seed=self.seed, label=self.name,
            # Every head here inits at the intercept-only solution with zero slopes:
            # Stan's uniform(-2, 2) default puts the starting linear predictor near
            # 2*sqrt(K), which at K = 19 is already in the saturation region. `rho` is a
            # vector[n_rho], so its init is a list even at length one.
            inits={"alpha": float(np.log(share / (1 - share))),
                   "beta": np.zeros(X.shape[1]).tolist(), "rho": [0.2]})
        warn_if_unconverged(self.diagnostics)

        draws = posterior(fit, ["alpha", "beta", "rho"])
        self.alpha_draws = draws["alpha"].reshape(-1)
        self.beta_draws = draws["beta"].reshape(len(self.alpha_draws), -1)
        self.rho_draws = draws["rho"].reshape(-1)
        self.rho = float(self.rho_draws.mean())
        return self

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    def shapes(self, df: pd.DataFrame, draw: int) -> tuple[np.ndarray, np.ndarray]:
        """Beta shape parameters per row at one posterior draw."""
        eta = self._design(df) @ self.beta_draws[draw] + self.alpha_draws[draw]
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        rho = float(np.clip(self.rho_draws[draw], RHO_MIN, RHO_MAX))
        scale = (1.0 - rho) / rho
        return np.clip(mu, EPS, 1 - EPS) * scale, (1.0 - np.clip(mu, EPS, 1 - EPS)) * scale

    def mean(self, df: pd.DataFrame) -> np.ndarray:
        eta = (self._design(df) @ self.beta_draws.mean(axis=0)
               + self.alpha_draws.mean())
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


class BetaGeometricHead:
    """The spell-duration head. `features = []` is the base arm and is legal in Stan."""

    def __init__(self, features: list[str], name: str, l2: float = GLM_L2,
                 kappa_scale: float = KAPPA_SCALE, chains: int = 4,
                 warmup: int = 1000, samples: int = 1000, seed: int = 42):
        self.features, self.name, self.l2 = list(features), name, l2
        self.kappa_scale = kappa_scale
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed

    def fit(self, rows: pd.DataFrame) -> "BetaGeometricHead":
        if self.features:
            (X,), self.scaler = standardized(rows, [rows], self.features)
        else:
            X, self.scaler = np.zeros((len(rows), 0)), None
        t = rows["t"].to_numpy(int)
        if (t < 1).any():
            raise ValueError(f"{self.name}: spell lengths below 1 — a spell is at least "
                             f"one missed game by construction")

        share_one = float(np.clip(
            np.dot(rows["w"].to_numpy(float), t == 1) / max(rows["w"].sum(), 1),
            EPS, 1 - EPS))
        model = compile_model(DURATION_MODEL)
        fit, self.diagnostics = sample(
            model,
            {"N": len(rows), "K": X.shape[1], "X": X, "t": t.tolist(),
             "censored": rows.get("censored", pd.Series(np.zeros(len(rows))))
                         .to_numpy(float).tolist(),
             "truncated": rows.get("truncated", pd.Series(np.zeros(len(rows))))
                          .to_numpy(float).tolist(),
             "w": rows["w"].to_numpy(float).tolist(),
             "beta_scale": prior_sd_for_l2(self.l2),
             "intercept_scale": INTERCEPT_SCALE, "kappa_scale": self.kappa_scale,
             "S": 0, "season_idx": [0] * len(rows), "year_sd_scale": 0.25},
            chains=self.chains, warmup=self.warmup, samples=self.samples,
            seed=self.seed, label=self.name,
            inits={"alpha": float(np.log(share_one / (1 - share_one))),
                   "beta": np.zeros(X.shape[1]).tolist(), "kappa": 3.7})
        warn_if_unconverged(self.diagnostics)

        draws = posterior(fit, ["alpha", "kappa"])
        self.alpha_draws = draws["alpha"].reshape(-1)
        self.kappa_draws = draws["kappa"].reshape(-1)
        self.beta_draws = (posterior(fit, ["beta"])["beta"]
                           .reshape(len(self.alpha_draws), -1) if self.features
                           else np.zeros((len(self.alpha_draws), 0)))
        self.mu = float(np.mean(1.0 / (1.0 + np.exp(-self.alpha_draws))))
        self.kappa = float(self.kappa_draws.mean())
        return self

    def shapes(self, df: pd.DataFrame, draw: int) -> tuple[np.ndarray, np.ndarray]:
        eta = np.full(len(df), self.alpha_draws[draw])
        if self.features:
            X = np.nan_to_num(df[self.features].to_numpy(dtype=float),
                              nan=0.0, posinf=0.0, neginf=0.0)
            eta = eta + self.scaler.transform(X) @ self.beta_draws[draw]
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        return beta_shapes(mu, np.full(len(df), self.kappa_draws[draw]))


class ShrunkOnsetFloor:
    """Gate B's bar: the prior-season onset rate carried forward, shrunk, with no features.

    `p = (onsets + k*league) / (at_risk + k)`, with `k` and the league mean fitted on train
    only. One constant, no covariates — and it has to be shrunk, because an unshrunk
    carry-forward of a proportion puts a player at exactly 0.000 and makes the benchmark
    infinite rather than merely strong.
    """

    name = "carry_forward"

    def fit(self, train: pd.DataFrame) -> "ShrunkOnsetFloor":
        self.league = float(train["onsets"].sum() / max(train["at_risk_played"].sum(), 1))
        grid = np.exp(np.linspace(np.log(0.5), np.log(500.0), 60))
        best, self.k = np.inf, grid[0]
        for k in grid:
            nll = -float(np.sum(_binomial_loglik(
                train["onsets"].to_numpy(float), train["at_risk_played"].to_numpy(float),
                self.predict(train, k))))
            if nll < best:
                best, self.k = nll, k
        return self

    def predict(self, df: pd.DataFrame, k: float | None = None) -> np.ndarray:
        k = self.k if k is None else k
        prior_on = df["onsets_lag1"].fillna(0.0).to_numpy(float)
        prior_at = df["at_risk_played_lag1"].fillna(0.0).to_numpy(float)
        return np.clip((prior_on + k * self.league) / (prior_at + k), EPS, 1 - EPS)


def _binomial_loglik(y: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    return y * np.log(p) + (n - y) * np.log1p(-p)


# ── The composed head ─────────────────────────────────────────────────────────

class SpellProcess(AvailabilityModel):
    """Entry x exit x within-tenure chain, composed into a predictive over games played.

    Subclasses `AvailabilityModel` on purpose: `evaluate` then scores it through exactly the
    same code path as the incumbent and the four original candidates, so the comparison is
    about the process and not about two implementations of CRPS.

    The pmf is Monte Carlo — there is no closed form for the games-played marginal of a
    tenure-truncated chain with two frailties — so it is built by simulating whole seasons.
    One posterior draw per block of simulated seasons, which is what makes the predictive
    the properly integrated one rather than a plug-in at the posterior mean, and which also
    preserves the shared-`beta` board correlation the Stan port exists for.
    """

    def __init__(self, heads: dict, name: str, oracle_tenure: bool = False,
                 draws: int = PREDICTIVE_DRAWS, sim_seasons: int = SIM_SEASONS,
                 seed: int = 42):
        self.heads, self.name = heads, name
        self.oracle_tenure = oracle_tenure
        self.draws, self.sim_seasons, self.seed = draws, sim_seasons, seed
        self._cache: tuple | None = None
        self.rho = np.nan

    def fit(self, train: pd.DataFrame) -> "SpellProcess":
        raise NotImplementedError("heads are fitted by `fit_heads` and composed here")

    def simulate(self, df: pd.DataFrame) -> np.ndarray:
        """(rows x draws*sim_seasons) games played, integrated over the posterior."""
        n_draws = len(self.heads["onset"].alpha_draws)
        idx = thin(n_draws, self.draws)
        team_games = df["team_games"].to_numpy(np.int64)

        blocks = []
        for lo in range(0, len(idx), DRAW_CHUNK):
            for d in idx[lo:lo + DRAW_CHUNK]:
                params = {}
                onset_a, onset_b = self.heads["onset"].shapes(df, d)
                dur_a, dur_b = self.heads["duration"].shapes(df, d)
                params.update(onset_a=onset_a, onset_b=onset_b,
                              dur_a=dur_a, dur_b=dur_b)
                if self.oracle_tenure:
                    params["pre"] = df["pre_tenure"].fillna(0).to_numpy(np.int64)
                    params["post"] = df["post_tenure"].fillna(0).to_numpy(np.int64)
                else:
                    entry_a, entry_b = self.heads["entry"].shapes(df, d)
                    exit_a, exit_b = self.heads["exit"].shapes(df, d)
                    params.update(entry_a=entry_a, entry_b=entry_b,
                                  exit_a=exit_a, exit_b=exit_b)
                blocks.append(simulate_gp(team_games, params, self.sim_seasons,
                                          seed=self.seed + int(d),
                                          force_endpoints=True))
        return np.concatenate(blocks, axis=1)

    def _ensure(self, df: pd.DataFrame, max_games: int) -> tuple:
        """Simulate once per (frame, grid) and serve both `predict_mean` and `predict_pmf`.

        `evaluate` calls the two in sequence on the same frame, and simulating a million
        seasons twice to answer one question would double the cost of every arm.
        """
        # `id` alone is not safe: a freed frame's address can be reused, and serving stale
        # simulations would be silent. The cheap content columns make a collision require
        # two frames that agree on row count, schedule total and endpoints.
        key = (id(df), len(df), max_games, int(df["team_games"].sum()),
               int(df["player_id"].iloc[0]) if len(df) else 0,
               int(df["player_id"].iloc[-1]) if len(df) else 0)
        if self._cache is None or self._cache[0] != key:
            sims = self.simulate(df)
            pmf = gp_pmf(sims, max_games)
            n = df["team_games"].to_numpy(float)
            # An "effective rho" so `implied_overdispersion` stays comparable to the
            # incumbent's fitted value. The process has no single dispersion parameter —
            # the spread comes from two frailties, the chain and the two tenure factors —
            # so this is read back off the simulated variance rather than fitted.
            mean_share = sims.mean(axis=1) / n
            var = sims.var(axis=1)
            binomial = n * mean_share * (1.0 - mean_share)
            inflation = float(np.mean(var / np.clip(binomial, 1e-9, None)))
            self.rho = float((inflation - 1.0) / max(np.median(n) - 1.0, 1.0))
            self._cache = (key, sims, pmf)
        return self._cache[1], self._cache[2]

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        sims, _ = self._ensure(df, int(df["team_games"].max()))
        return np.clip(sims.mean(axis=1) / df["team_games"].to_numpy(float),
                       EPS, 1 - EPS)

    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        _, pmf = self._ensure(df, max_games)
        return pmf


class CalibratedProcess(AvailabilityModel):
    """Option (b): the same simulator, hazards **calibrated** rather than fitted.

    The documented fallback if Gate D fails, and it ships either way — it is the invariant
    every fitted arm should satisfy at its own `C`, and having it scored beside them is
    what makes "the marginal was never the argument" checkable rather than asserted.

    It takes the incumbent's per-player predictive mean and inflation and a measured
    clustering, and inverts `inflation = C + rho*(n - C)` in two lines. So its GP marginal
    is the incumbent's **by construction** and what it adds is the game-level process:
    which games he misses, correlated the way the transition data says.
    """

    def __init__(self, incumbent: BetaBinomialGLM, clustering: float, name: str,
                 draws: int = PREDICTIVE_DRAWS, sim_seasons: int = SIM_SEASONS,
                 seed: int = 42):
        self.incumbent, self.clustering, self.name = incumbent, clustering, name
        # The same Monte Carlo budget the fitted arms get. It has to be the same, or the
        # comparison is partly a comparison of pmf resolution: a sparse MC pmf leaves zero
        # cells at observed values and inflates CRPS. Affordable because this chain has no
        # tenure factors and no spell machinery — it is four array ops per game index.
        self.n_sims = draws * sim_seasons
        self.seed = seed
        self.rho = incumbent.rho
        self._cache: tuple | None = None

    def fit(self, train: pd.DataFrame) -> "CalibratedProcess":
        return self

    def _ensure(self, df: pd.DataFrame, max_games: int) -> tuple:
        key = (id(df), len(df), max_games, int(df["team_games"].sum()))
        if self._cache is None or self._cache[0] != key:
            mu = self.incumbent.predict_mean(df)
            n = float(np.median(df["team_games"].to_numpy(float)))
            sims = simulate_calibrated(df["team_games"].to_numpy(np.int64), mu,
                                       1.0 + (n - 1.0) * self.incumbent.rho,
                                       self.clustering, self.n_sims, self.seed)
            self._cache = (key, sims, gp_pmf(sims, max_games))
        return self._cache[1], self._cache[2]

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        sims, _ = self._ensure(df, int(df["team_games"].max()))
        return np.clip(sims.mean(axis=1) / df["team_games"].to_numpy(float),
                       EPS, 1 - EPS)

    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        _, pmf = self._ensure(df, max_games)
        return pmf


class HybridProcess(AvailabilityModel):
    """The decoupled arm: the incumbent's marginal, the fitted head's spell shape.

    The other two challengers each have to earn the games-played marginal *and* the absence
    shape from one fitted object, and both pay for it — the fitted arms lose validation CRPS
    to the tenure factors, and the calibrated fallback buys its marginal with a geometric
    duration the data falsifies (P(spell >= 26 games) 99% low). This arm refuses the trade:
    the **count** is drawn from whichever distribution predicts it best — currently the
    incumbent's validated beta-binomial — and the fitted beta-geometric only decides *where*
    those absences fall.

    So its games-played predictive is **identically** the incumbent's, not an approximation
    of it, and `predict_pmf` returns exactly that. Every marginal metric is inherited, which
    means its row in a CRPS table is a duplicate of the floor's **by construction** — that
    is the correct result rather than a bug, and it is why this arm is not selectable on a
    marginal metric. What it adds is only visible in `spell_shape_table`.

    **One caveat measured rather than assumed**: spell length is not independent of the
    count it is conditioned on, because a player cannot miss 26 consecutive games unless he
    misses 26 games. Fed a plain Binomial(82, 0.78) the allocator produces
    `P(spell >= 26) = 0.0000`; fed the incumbent's ~23x-overdispersed pmf it produces
    0.0214. The heavy left tail of the marginal is what makes a long absence representable
    at all, which is a second reason not to adopt an arm with a worse marginal.
    """

    def __init__(self, incumbent: BetaBinomialGLM, mu_dur: float, kappa_dur: float,
                 name: str = "gp_hybrid", seed: int = 42):
        self.incumbent, self.name, self.seed = incumbent, name, seed
        self.mu_dur, self.kappa_dur = float(mu_dur), float(kappa_dur)
        self.rho = incumbent.rho

    def fit(self, train: pd.DataFrame) -> "HybridProcess":
        return self

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        return self.incumbent.predict_mean(df)

    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        return self.incumbent.predict_pmf(df, max_games)

    def sequences(self, df: pd.DataFrame, max_games: int, n_sims: int,
                  seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Draw counts from the incumbent's pmf, then lay them out as spells."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        pmf = self.predict_pmf(df, max_games)
        cdf = np.cumsum(pmf / pmf.sum(axis=1, keepdims=True), axis=1)
        team_games = df["team_games"].to_numpy(int)
        gp = np.concatenate([(rng.random(len(df))[:, None] > cdf).sum(axis=1)
                             for _ in range(n_sims)])
        tiled = np.tile(team_games, n_sims)
        gp = np.minimum(gp, tiled)
        return allocate_spells(gp, tiled, self.mu_dur, self.kappa_dur,
                               seed=rng.integers(1 << 31)), tiled


SPELL_STATS = [("p_eq_1", lambda x: (x == 1).mean()),
               ("p_ge_10", lambda x: (x >= 10).mean()),
               ("p_ge_26", lambda x: (x >= 26).mean()),
               ("mean_spell", lambda x: x.mean())]


def spell_shape_table(observed: np.ndarray, arms: dict) -> pd.DataFrame:
    """Observed against simulated absence-spell shape, per arm.

    **The metric no marginal score can see, and the reason this comparison exists.** CRPS,
    MAE and `P(GP<41)` are all functions of a pmf over a *count*: two models with identical
    games-played distributions can allocate those absences as 82 independent coin flips or
    as one 20-game block and score the same. For a best-ball simulator the difference is the
    whole question — a knockout on best-7-of-16 is decided by whether a player is gone for a
    fortnight, not by his season total.
    """
    rows = [{"arm": "observed", "n_spells": len(observed),
             **{k: float(f(observed)) for k, f in SPELL_STATS}}]
    for arm, lens in arms.items():
        row = {"arm": arm, "n_spells": len(lens),
               **{k: float(f(lens)) for k, f in SPELL_STATS}}
        for k, _ in SPELL_STATS:
            base = rows[0][k]
            row[f"{k}_rel_error"] = (row[k] / base - 1.0) if base else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = [f"{k}_rel_error" for k, _ in SPELL_STATS if k != "mean_spell"]
    out["mean_abs_rel_error"] = out[cols].abs().mean(axis=1)
    return out


def matched_transition_rates(panel: pd.DataFrame,
                             design: pd.DataFrame) -> tuple[float, float]:
    """`P(play | played)` and `P(play | missed)` on the frame the 22.7x is measured on.

    Full window, established rotation players. The *matching* is the whole point: the
    recorded 3.96x clustering figure runs on the appearance window over all players while
    the overdispersion it was being divided into runs here, so the two were never
    comparable and the ratio between them meant nothing.

    Computed here rather than read from `availability_profile.csv` because the fallback
    takes it as an *input*: reading it from an eda artifact would make this module's answer
    depend on whether `make availability-profile` had been run.
    """
    rotation = rotation_population(design)
    matched = collapse_transitions(
        panel[[(s, p) in rotation
               for s, p in zip(panel["season"], panel["player_id"])]], "full")
    p = float(1.0 - matched["onsets"].sum() / max(matched["at_risk_played"].sum(), 1))
    q = float(matched["recoveries"].sum() / max(matched["at_risk_missed"].sum(), 1))
    return p, q


def measured_clustering(panel: pd.DataFrame, design: pd.DataFrame) -> float:
    """`C = (1 + rho)/(1 - rho)` at the matched frame's lag-1 autocorrelation."""
    p, q = matched_transition_rates(panel, design)
    return (1.0 + (p - q)) / (1.0 - (p - q))


def fit_heads(train: pd.DataFrame, spell_rows: pd.DataFrame, features: list[str],
              duration_features: list[str], label: str, cfg_stan: dict,
              fast: bool, cache: dict, oracle_tenure: bool) -> dict:
    """Fit whatever the arm needs, reusing anything an earlier arm already fitted.

    The cache is keyed on what actually determines a fit — the head, its feature list, the
    split and the fitting frame's identity — so A1, A2 and A4 share one onset fit and one
    entry/exit pair between them instead of paying for three. At ~250 s a beta-binomial fit
    on 10,000 rows that is most of an hour.
    """
    iters = ({"warmup": int(cfg_stan.get("select_warmup", 500)),
              "samples": int(cfg_stan.get("select_samples", 500))} if fast else
             {"warmup": int(cfg_stan.get("warmup", 1000)),
              "samples": int(cfg_stan.get("samples", 1000))})
    chains = int(cfg_stan.get("chains", 4))
    seed = int(cfg_stan.get("seed", 42))
    frame_id = cache.setdefault("_frames", {}).setdefault(id(train), len(cache["_frames"]))

    def get(head: str, builder) -> object:
        key = (head, frame_id, label.split("/")[-1], tuple(features),
               tuple(duration_features), fast)
        if key not in cache:
            cache[key] = builder()
        return cache[key]

    heads = {}
    heads["onset"] = get("onset", lambda: BetaBinomialHead(
        features, f"{label}/onset", chains=chains, seed=seed, **iters
    ).fit(train, "onsets", "at_risk_played"))
    heads["duration"] = get("duration", lambda: BetaGeometricHead(
        duration_features, f"{label}/duration",
        kappa_scale=float(cfg_stan.get("games_played", {}).get("kappa_scale",
                                                               KAPPA_SCALE)),
        chains=chains, seed=seed, **iters).fit(spell_rows))
    if not oracle_tenure:
        heads["entry"] = get("entry", lambda: BetaBinomialHead(
            features, f"{label}/entry", chains=chains, seed=seed, **iters
        ).fit(train, "pre_tenure", "entry_trials"))
        heads["exit"] = get("exit", lambda: BetaBinomialHead(
            features, f"{label}/exit", chains=chains, seed=seed, **iters
        ).fit(train, "post_tenure", "exit_trials"))
    return heads


# ── Spell rows ────────────────────────────────────────────────────────────────

def spell_rows_for(panel: pd.DataFrame, frame: pd.DataFrame,
                   features: list[str] | None = None,
                   require_interior: bool = True) -> pd.DataFrame:
    """Within-tenure spells for the fitting rows, collapsed with a multiplicity weight.

    On the **appearance** tenure every spell is interior — the window opens and closes on a
    game he appeared in — so `censored` and `truncated` are identically zero and the
    duration head's edge branches are unused by construction rather than by omission. That
    is the tenure decomposition paying for itself a second time, and `require_interior`
    asserts it rather than trusting it.

    On A3's **rostered** tenure they are not, and that is the arm's whole point: a player
    who tore an ACL in March stays on the roster, so his absence runs to the end of the
    window and is genuinely right-censored. 3,205 of the spells inside the rostered window
    are edge spells the appearance window cannot see. So the branches in
    `betageometric_duration.stan` are live code, exercised by exactly the arm that was
    built to reach the population the structural proxy misses.
    """
    from src.models.games_played import duration_rows

    spells = spell_classes(panel, "appearance")
    keys = set(zip(frame["season"], frame["player_id"]))
    spells = spells[[(s, p) in keys
                     for s, p in zip(spells["season"], spells["player_id"])]]
    if require_interior and (spells["spell_class"] != "interior").any():
        n_bad = int((spells["spell_class"] != "interior").sum())
        raise ValueError(f"{n_bad} within-tenure spells are censored or truncated — the "
                         f"appearance window is supposed to make that impossible")
    return duration_rows(spells, features, frame if features else None)


# ── Gate A: the timing probe ──────────────────────────────────────────────────

def probe_timing(train: pd.DataFrame, spell_rows: pd.DataFrame, cfg_stan: dict,
                 max_hours: float) -> dict:
    """One onset fit on one season, extrapolated to the ladder — and it is a LOWER bound.

    `stan-composition`'s Gate A extrapolated 12.8 h against an actual 20.9 h, a **1.63x**
    under-prediction, because per-row sampler cost is superlinear in rows: more data
    sharpens the posterior, shrinks the step size and buys more leapfrog steps per
    iteration on top of an already-linear per-gradient cost. The same correction is applied
    here and the raw extrapolation is reported beside it, because a probe that quietly
    absorbs its own known bias is not a probe.

    **⚠️ 1.63 is deliberately retained as a CONSERVATIVE factor, not a current estimate.**
    When `stan-composition` went validation-only its one-pass sweep re-measured the same
    miss at **1.17x** (8.3 h extrapolated against 9.78 h actual, 2026-08-08), so the
    multiplier is a property of the *run shape* rather than of the sampler: the 1.63x was
    inflated by a second pass on a larger frame that the linear extrapolation modelled
    badly. Lowering the constant would make this gate more permissive, and the asymmetry
    says not to — aborting an affordable run costs one probe fit, while admitting an
    unaffordable one is discovered hours in. The stated figures above are the retired
    measurement and are kept because the constant is derived from them.
    """
    last = sorted(train["season"].unique())[-1]
    probe_frame = train[train["season"] == last]
    fast = {"warmup": int(cfg_stan.get("select_warmup", 500)),
            "samples": int(cfg_stan.get("select_samples", 500))}
    model = BetaBinomialHead(FEATURE_COLS, "probe/one-season",
                             chains=int(cfg_stan.get("chains", 4)),
                             seed=int(cfg_stan.get("seed", 42)), **fast
                             ).fit(probe_frame, "onsets", "at_risk_played")
    seconds = model.diagnostics["wall_clock_s"]
    per_row = seconds / max(len(probe_frame), 1)

    full = {"warmup": int(cfg_stan.get("warmup", 1000)),
            "samples": int(cfg_stan.get("samples", 1000))}
    iter_scale = (full["warmup"] + full["samples"]) / (fast["warmup"] + fast["samples"])
    # Distinct beta-binomial fits across the ladder: onset + entry + exit on the base
    # frame, the same three on A3's restricted frame, per split. The duration fits are
    # seconds (a few hundred collapsed rows against two parameters) except A4's, which is
    # counted as one more beta-binomial-sized fit.
    n_fits = 7
    est = n_fits * len(train) * per_row * (1.0 + iter_scale)
    raw_hours = est / 3600
    hours = raw_hours * 1.63
    print(f"  Gate A: probe fit {len(probe_frame):,} rows ({last}) in {seconds:.0f}s "
          f"-> {raw_hours:.1f}h linear, {hours:.1f}h after the 1.63x correction "
          f"`stan-composition` measured ({per_row * 1000:.2f} ms/row at select iters)")
    if hours > max_hours:
        raise RuntimeError(
            f"Gate A: corrected sweep estimate {hours:.1f}h exceeds the {max_hours:.0f}h "
            f"budget. Fallbacks, in order: drop `duration_covariates` from the ladder, "
            f"shorten the select chains, subsample TRAIN player-seasons (never val/test). "
            f"See docs/games-played-plan.md.")
    return {"probe_season": last, "probe_rows": len(probe_frame),
            "probe_seconds": float(seconds), "linear_hours": float(raw_hours),
            "extrapolated_hours": float(hours), "diagnostics": model.diagnostics}


# ── Gate B ────────────────────────────────────────────────────────────────────

def gate_b(train: pd.DataFrame, test: pd.DataFrame, head: BetaBinomialHead) -> dict:
    """Does the fitted onset head beat a shrunk carry-forward of the prior onset rate?

    Scored as held-out binomial log-likelihood per at-risk transition, which is the
    likelihood the head is fitted under. A head that does not clear its no-fit floor is
    not a model — `CLAUDE.md`'s standing rule, applied to the one component of this process
    that has a natural floor to clear.
    """
    floor = ShrunkOnsetFloor().fit(train)
    y = test["onsets"].to_numpy(float)
    n = test["at_risk_played"].to_numpy(float)
    exposure = max(n.sum(), 1.0)

    floor_ll = float(np.sum(_binomial_loglik(y, n, floor.predict(test)))) / exposure
    head_ll = float(np.sum(_binomial_loglik(
        y, n, np.clip(head.mean(test), EPS, 1 - EPS)))) / exposure
    return {"floor_shrinkage_k": float(floor.k), "floor_league_rate": floor.league,
            "floor_loglik_per_transition": floor_ll,
            "head_loglik_per_transition": head_ll,
            "gain": head_ll - floor_ll, "beats_floor": bool(head_ll > floor_ll),
            "n_rows": len(test), "n_transitions": int(n.sum())}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score(model, frame: pd.DataFrame, max_games: int, seed: int = 42) -> dict:
    rows, predictions = evaluate(model, frame, max_games, seed)
    flat = {(r["group"], r["metric"]): r["value"] for r in rows}
    return {"crps": flat[("all", "crps_games")],
            "mae": flat[("all", "mae_games")],
            "r2": flat[("all", "r2_gp_share")],
            "pit_ks": flat[("all", "pit_ks_distance")],
            "implied_overdispersion": flat[("all", "implied_overdispersion")],
            "rotation_crps": flat.get(("rotation", "crps_games"), np.nan),
            "predicted_below_41": flat.get(("rotation",
                                            "predicted_share_below_41"), np.nan),
            "observed_below_41": flat.get(("rotation",
                                           "observed_share_below_41"), np.nan),
            "predicted_below_60": flat.get(("rotation",
                                            "predicted_share_below_60"), np.nan),
            "observed_below_60": flat.get(("rotation",
                                           "observed_share_below_60"), np.nan),
            "_rows": rows, "_predictions": predictions}


def tail_error(scored: dict) -> float:
    """Mean absolute error on the two rotation tail probabilities — Gate D's win condition.

    One number so the arms can be ordered on the thing the head is actually for. CRPS
    integrates over the whole distribution and is dominated by the bulk, which is exactly
    why the plan decided in writing, before the sweep, that it is a non-regression bar
    rather than the metric.
    """
    return float(np.mean([
        abs(scored["predicted_below_41"] - scored["observed_below_41"]),
        abs(scored["predicted_below_60"] - scored["observed_below_60"])]))


# ── The three-state frame ─────────────────────────────────────────────────────

def three_state_design(panel: pd.DataFrame, design: pd.DataFrame,
                       first_season: str = STATUS_FIRST_SEASON,
                       min_coverage: float = MIN_STATUS_COVERAGE
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A3's design and its rewritten panel, with the tenure taken from `status`."""
    tenure = rostered_tenure(panel, first_season, min_coverage)
    if tenure.empty:
        return design.iloc[:0], panel.iloc[:0]
    process = rostered_process(panel, tenure)
    counts = process["counts"]
    multi = multi_team_seasons(panel)
    counts = counts[[(s, p) not in multi
                     for s, p in zip(counts["season"], counts["player_id"])]]

    keep = ["season", "player_id", "pre_tenure", "post_tenure", "tenure_games",
            "entry_trials", "exit_trials", "onsets", "at_risk_played"]
    base = design.drop(columns=[c for c in keep if c not in
                                ("season", "player_id")], errors="ignore")
    out = base.merge(counts[keep], on=["season", "player_id"], how="inner")
    out["fittable"] = out["at_risk_played"].notna()
    return out.reset_index(drop=True), process["panel"]


# ── The sweep ─────────────────────────────────────────────────────────────────

def _floor_scores(frames: dict, max_games: int, l2: float, seed: int) -> dict:
    """The incumbent refitted and scored on every row set an arm might be evaluated on.

    Two things this has to get right, and they are easy to get wrong in opposite ways.

    **Each split gets its own fit.** `frames` maps a split to a `(train, eval)` pair, and
    the validation-side floor is fitted on the *inner* frame — the same rows the arms'
    validation heads see. Fitting it on train+validation and scoring it on validation would
    give the floor a look at the selection split that no challenger gets, which is a leak
    in the bar rather than in a model and would therefore be invisible in every per-arm
    diagnostic.

    **Each row set gets its own score.** A1 and A3 are evaluated on restricted rows, so
    their CRPS is not comparable to 10.795 as written; scoring the incumbent on *the same
    rows* is the only honest comparison and it is what `crps_vs_floor` reports.
    """
    out = {}
    for name, (train, frame) in frames.items():
        out[name] = score(BetaBinomialGLM(l2).fit(train), frame, max_games, seed)
    return out


def sweep(base: pd.DataFrame, three_state: tuple[pd.DataFrame, pd.DataFrame],
          panel: pd.DataFrame, cfg_stan: dict, max_games: int,
          test_seasons: int = TEST_SEASONS,
          clustering: float | None = None) -> tuple[pd.DataFrame, list[dict], dict]:
    """Every arm, on the VALIDATION split only.

    The test seasons are not fitted, not scored and not returned. They were, until
    2026-08-05, and it cost more than compute: with both columns in one table the gate read
    the wrong one and settled which model ships on the split that is not allowed to decide.
    Removing the column removes the temptation, and `held_out.assert_unlocked` catches
    anything that reaches past it.

    It also removes a confound. The test side used to *refit* on train+validation at double
    the sampler iterations, so a val/test disagreement conflated three things — different
    evaluation rows, different training data and different chain lengths — and could not
    serve as the replication check it was being read as. The end-of-project number is taken
    by `src/final_evaluation.py`, which refits deliberately and once.

    Costs 63% less sampler time, because the test side was 12 of the 25 fits at twice the
    iterations.
    """
    seed = int(cfg_stan.get("seed", 42))
    l2 = float(cfg_stan.get("games_played", {}).get("glm_l2", GLM_L2))
    draws = int(cfg_stan.get("games_played", {}).get("predictive_draws",
                                                     PREDICTIVE_DRAWS))
    sims = int(cfg_stan.get("games_played", {}).get("sim_seasons", SIM_SEASONS))
    ts_design, ts_panel = three_state

    inner, val = selection_split(base, test_seasons)
    ts_inner, ts_val = (selection_split(ts_design, test_seasons)
                        if len(ts_design) else (ts_design, ts_design))

    rows, diagnostics, models, cache = [], [], {}, {}
    fittable = {"val": inner[inner["fittable"]]}
    ts_fittable = {"val": ts_inner[ts_inner["fittable"]]}

    # Every arm's floor is the incumbent refitted on that arm's own train and scored on
    # that arm's own rows — validation-side from `inner`, test-side from `full_train`.
    floors = {
        "base": _floor_scores({"val": (inner, val)}, max_games, l2, seed),
        "base_fittable": _floor_scores(
            {"val": (inner, val[val["fittable"]])}, max_games, l2, seed),
    }
    if len(ts_val):
        floors["three_state"] = _floor_scores({"val": (ts_inner, ts_val)},
                                              max_games, l2, seed)

    floor_row = floors["base"]
    rows.append({"variant": "floor", "n_features": len(FEATURE_COLS),
                 "n_val_rows": len(val),
                 "val_crps": floor_row["val"]["crps"],
                 "floor_val_crps": floor_row["val"]["crps"],
                 **_metric_columns(floor_row["val"], "val"),
                 "selectable": False})

    # `interior` is False only for the three-state arm, whose rostered tenure genuinely
    # admits edge spells — a season-ending injury while still on the roster.
    specs = [
        ("within_tenure", fittable, {"val": val[val["fittable"]]},
         panel, [], True, "base_fittable", True),
        ("full_window", fittable, {"val": val}, panel, [], False, "base", True),
    ]
    if len(ts_val):
        specs.append(("three_state", ts_fittable, {"val": ts_val},
                      ts_panel, [], False, "three_state", False))
    specs.append(("duration_covariates", fittable, {"val": val},
                  panel, list(FEATURE_COLS), False, "base", True))

    for (label, fit_frames, eval_frames, arm_panel, dur_feats, oracle, floor_key,
         interior) in specs:
        scored = {}
        for split in ("val",):
            train_frame = fit_frames[split]
            spell_rows = spell_rows_for(arm_panel, train_frame,
                                        dur_feats if dur_feats else None,
                                        require_interior=interior)
            heads = fit_heads(train_frame, spell_rows, list(FEATURE_COLS), dur_feats,
                              f"{label}/{split}", cfg_stan, fast=(split == "val"),
                              cache=cache, oracle_tenure=oracle)
            for head in heads.values():
                if head.diagnostics not in diagnostics:
                    diagnostics.append(head.diagnostics)
            model = SpellProcess(heads, f"gp_{label}", oracle_tenure=oracle,
                                 draws=draws, sim_seasons=sims, seed=seed)
            scored[split] = score(model, eval_frames[split], max_games, seed)
            models[f"{label}/{split}"] = {"model": model, "heads": heads,
                                          "train": train_frame,
                                          "eval": eval_frames[split]}
        f = floors[floor_key]
        rows.append({"variant": label,
                     "n_features": len(FEATURE_COLS) + len(dur_feats),
                     "n_val_rows": len(eval_frames["val"]),
                     "val_crps": scored["val"]["crps"],
                     "floor_val_crps": f["val"]["crps"],
                     **_metric_columns(scored["val"], "val"),
                     "selectable": ARMS[label]["selectable"]})

    # Arm B: the closed-form fallback, scored on the same rows as everything else. Not
    # selectable — it reproduces the incumbent's marginal by construction, so choosing it
    # on a marginal metric would be choosing it for the one thing it does not decide.
    if clustering is not None:
        # Same split discipline as the floor: the validation-side incumbent behind the
        # calibration is fitted on `inner`, never on train+validation.
        cal_val = CalibratedProcess(BetaBinomialGLM(l2).fit(inner), clustering,
                                    "gp_calibrated", draws, sims, seed)
        v = score(cal_val, val, max_games, seed)
        models["calibrated/val"] = {"model": cal_val, "heads": {},
                                    "train": inner, "eval": val}
        rows.append({"variant": "calibrated_fallback",
                     "n_features": len(FEATURE_COLS),
                     "n_val_rows": len(val),
                     "val_crps": v["crps"],
                     "floor_val_crps": floors["base"]["val"]["crps"],
                     **_metric_columns(v, "val"), "selectable": False})

    # The hybrid: the incumbent's marginal, the fitted head's spell shape. Its marginal row
    # duplicates the floor's by construction — that IS the result — so it is not selectable
    # and its case is made in `spell_shape_table` instead.
    interior = spell_classes(panel, "appearance")
    keys = set(zip(inner["season"], inner["player_id"]))
    train_spells = interior[[(a, b) in keys for a, b in
                             zip(interior["season"], interior["player_id"])]]
    collapsed = duration_rows(train_spells)
    dur = fit_beta_geometric(collapsed["t"].to_numpy(), collapsed["w"].to_numpy())
    hybrid = HybridProcess(BetaBinomialGLM(l2).fit(inner), dur["mu"], dur["kappa"],
                           seed=seed)
    models["hybrid/val"] = {"model": hybrid, "heads": {}, "train": inner,
                            "eval": val, "duration": dur}
    hv = score(hybrid, val, max_games, seed)
    rows.append({"variant": "hybrid", "n_features": len(FEATURE_COLS),
                 "n_val_rows": len(val),
                 "val_crps": hv["crps"],
                 "floor_val_crps": floors["base"]["val"]["crps"],
                 **_metric_columns(hv, "val"), "selectable": False})

    table = pd.DataFrame(rows)
    table["crps_vs_floor"] = table["val_crps"] - table["floor_val_crps"]
    candidates = table[table["selectable"]]
    table["selected"] = False
    if len(candidates):
        table.loc[candidates["val_crps"].idxmin(), "selected"] = True
    return table, diagnostics, models


_METRIC_KEYS = ("mae", "r2", "pit_ks", "implied_overdispersion", "rotation_crps",
                "predicted_below_41", "observed_below_41",
                "predicted_below_60", "observed_below_60")


def _metric_columns(scored: dict, split: str) -> dict:
    """Every metric, prefixed by split.

    Both splits are carried because the gates read **validation** — the table used to store
    the test side only, which is how the win condition (the tail) went unmeasured on the
    split that is allowed to decide, and how Gate D came to be settled on test.
    """
    out = {f"{split}_{k}": scored[k] for k in _METRIC_KEYS}
    out[f"{split}_tail_error"] = float(np.mean([
        abs(scored["predicted_below_41"] - scored["observed_below_41"]),
        abs(scored["predicted_below_60"] - scored["observed_below_60"])]))
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    cfg_stan = cfg.get("stan", {})
    cfg_gp = cfg_stan.get("games_played", {})
    seed = int(cfg_stan.get("seed", 42))
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", TEST_SEASONS))
    max_hours = float(cfg_gp.get("max_extrapolated_hours", 6.0))

    print("Stan games-played head — entry x exit x a within-tenure two-state chain")
    panel = pd.read_parquet(features_dir / "availability_panel.parquet")
    design = games_played_design(cfg, panel)
    design = design.merge(onset_rate_lags(design, seasons),
                          on=["season", "player_id"], how="left")
    max_games = int(design["team_games"].max())

    inner, val = selection_split(design, test_seasons)
    print(f"  {len(design):,} player-seasons. **The test split is LOCKED** — this sweep "
          f"fits and scores\n  VALIDATION only, and `held_out.assert_unlocked` raises on "
          f"anything that reaches past it.\n  The end-of-project number is taken once, by "
          f"`make final-evaluation`.")
    print(f"  {len(inner):,} fit / {len(val):,} select "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")
    print(f"  fittable (single-team): {int(design['fittable'].sum()):,} of "
          f"{len(design):,} rows ({design['fittable'].mean():.1%}). Multi-team seasons "
          f"keep their row\n  and lose their process targets — excluded from every fit, "
          f"included in every evaluation, so\n  CRPS stays comparable to the incumbent's "
          f"on all {len(val)} validation rows.")

    ts_design, ts_panel = three_state_design(panel, design)
    print(f"  three-state frame: {len(ts_design):,} player-seasons from "
          f"{STATUS_FIRST_SEASON} at >= {MIN_STATUS_COVERAGE:.0%} status coverage")

    probe = probe_timing(inner[inner["fittable"]],
                         spell_rows_for(panel, inner[inner["fittable"]]),
                         cfg_stan, max_hours)

    clustering = measured_clustering(panel, design)
    print(f"  matched-frame clustering C = {clustering:.4f} (full window, established "
          f"rotation players) — the\n  fallback's input, and the figure that makes the "
          f"dispersion budget a surplus rather than a hole.")

    table, diagnostics, models = sweep(design, (ts_design, ts_panel), panel,
                                       cfg_stan, max_games, test_seasons, clustering)
    diagnostics.append(probe["diagnostics"])

    print("\nArm ladder (CRPS in games, lower is better). Selection reads VALIDATION:")
    show = ["variant", "n_val_rows", "val_crps", "floor_val_crps", "crps_vs_floor",
            "val_pit_ks", "val_implied_overdispersion", "val_tail_error",
            "selectable", "selected"]
    print(table[[c for c in show if c in table.columns]].round(4).to_string(index=False))

    # ── Gate B ────────────────────────────────────────────────────────────────
    # On the VALIDATION side, like every other gate: the val-side holder is fitted on
    # `inner` and evaluated on `val`, so nothing here reads the test split.
    shipped = (table.loc[table["selected"], "variant"].iloc[0]
               if table["selected"].any() else "full_window")
    holder = models[f"{shipped}/val"]
    gate_b_row = gate_b(holder["train"], holder["eval"][holder["eval"]["fittable"]],
                        holder["heads"]["onset"])
    print(f"\nGate B — the onset head against a shrunk carry-forward, ON VALIDATION:")
    print(f"  floor (k = {gate_b_row['floor_shrinkage_k']:.1f}, league "
          f"{gate_b_row['floor_league_rate']:.4f}): "
          f"{gate_b_row['floor_loglik_per_transition']:.6f} per transition")
    print(f"  fitted head:                              "
          f"{gate_b_row['head_loglik_per_transition']:.6f} "
          f"({gate_b_row['gain']:+.6f})")
    if not gate_b_row["beats_floor"]:
        print("  /!\\  The onset head does NOT clear its no-fit floor. Per CLAUDE.md that "
              "is not a model.")

    # ── Gate C ────────────────────────────────────────────────────────────────
    # Both curves on the SAME rows: the simulated one is produced on the evaluation frame,
    # so the observed one is restricted to it rather than read off all 30 seasons.
    eval_rows = set(zip(holder["eval"]["season"], holder["eval"]["player_id"]))
    observed_curve = hazard_by_streak(panel, "appearance", eval_rows)
    sim_curve = _simulated_curve(holder, cfg_gp, seed)
    curve = order_by_streak(observed_curve.merge(
        sim_curve, on="streak", how="outer", suffixes=("_observed", "_simulated")))
    still_falling = curve_still_falling(sim_curve)
    print("\nGate C — the onset hazard by played streak, ON VALIDATION:")
    print(curve[["streak", "hazard_observed", "hazard_simulated"]].round(4)
          .to_string(index=False))
    print(f"  the simulated curve keeps falling past a streak of 20: {still_falling}")
    print("  It falls because of SORTING, not state dependence: high-hazard players break "
          "their streaks\n  early, so long streaks are populated by low-hazard players. "
          "The frailty the beta-binomial\n  marginalizes generates this for free, which is "
          "what lets the collapse keep it.")

    # ── Gate D ────────────────────────────────────────────────────────────────
    # Evaluated for BOTH candidates — the selected fitted arm and the closed-form
    # fallback — because option (b) is a real branch of this design rather than a
    # consolation prize, and the artifact has to record whether the thing that actually
    # ships cleared the gate. Reporting only the fitted arm's verdict and then quietly
    # shipping the other one would be the reverse of this repo's whole protocol.
    incumbent = table[table["variant"] == "floor"].iloc[0]
    verdicts = {shipped: _gate_d(table, shipped, incumbent)}
    for name, variant in (("calibrated", "calibrated_fallback"), ("hybrid", "hybrid")):
        if f"{name}/val" in models:
            verdicts[name] = _gate_d(table, variant, incumbent)

    print(f"\nGate D — the head gate, on "
          f"{int(table[table['variant'] == 'floor']['n_val_rows'].iloc[0])} VALIDATION "
          f"rows. The bars are the incumbent's own\n  figures on the same split, not "
          f"constants: CRPS and PIT are NON-REGRESSION bars and the\n  win condition is "
          f"the tail, decided in writing before the sweep ran.")
    for name, v in verdicts.items():
        print(f"  [{name}]")
        print(f"    CRPS    {v['crps']:.4f} against the incumbent's {v['crps_bar']:.4f}  "
              f"-> {'PASS' if v['crps_passes'] else 'FAIL'}")
        print(f"    PIT KS  {v['pit_ks']:.4f} against the incumbent's "
              f"{v['pit_ks_bar']:.4f}  -> {'PASS' if v['pit_passes'] else 'FAIL'}")
        print(f"    tail    P(GP<41) {v['predicted_below_41']:.4f} / P(GP<60) "
              f"{v['predicted_below_60']:.4f} against observed "
              f"{v['observed_below_41']:.4f} / {v['observed_below_60']:.4f}")
        print(f"            mean absolute tail error {v['tail_error']:.4f} against the "
              f"incumbent's {v['incumbent_tail_error']:.4f} "
              f"-> {'PASS' if v['tail_improves'] else 'FAIL'}")

    # Which model actually ships, and therefore which one Gate E measures. The fitted arm
    # has priority when it clears; otherwise the documented branch takes over — and only
    # if it clears too, since "the fallback ships" is not the same as "the fallback works".
    fitted_passes = verdicts[shipped]["passes"]
    cal_passes = verdicts.get("calibrated", {}).get("passes", False)
    ships = shipped if fitted_passes else ("calibrated" if cal_passes else shipped)
    gate_d = {**verdicts[ships], "ships": ships,
              "fitted_arm_passes": fitted_passes,
              "calibrated_passes": cal_passes}
    print(f"\n  SHIPS: {ships}.")
    if not fitted_passes:
        print("  The FITTED arm does not clear Gate D, which lands this in option (b) — "
              "the same\n  simulator with hazards inverted from the incumbent's marginal "
              "rather than fitted. A\n  documented branch, decided in writing before the "
              "sweep ran, and not a rescue.")
    if not verdicts[ships]["passes"]:
        print("  /!\\  Neither candidate clears Gate D. The head does not ship; the "
              "incumbent stands.")

    # ── The spell shape: the metric no marginal score can see ─────────────────
    shape = _spell_shape(models, panel, val, max_games, cfg_gp, seed)
    print("\nAbsence-spell shape on the VALIDATION rows — the statistic the contest "
          "actually turns on:")
    print(shape.round(4).to_string(index=False))
    print("  A knockout on best-7-of-16 is decided by whether a player is gone for a "
          "fortnight, not\n  by his season total — and CRPS cannot tell those apart. Two "
          "models with identical\n  games-played pmfs can scatter the absences or block "
          "them and score the same.")

    # ── The fallback, emitted whatever happens ────────────────────────────────
    fallback = _fallback_table(panel, design, inner, val, clustering, cfg_stan)
    print("\nOption (b), the closed-form fallback — emitted whatever Gate D says, because "
          "it is the\n  invariant every arm should satisfy at its own C:")
    print(fallback.round(4).to_string(index=False))

    process = _spell_process_table(models[f"{ships}/val"], table, gate_b_row, gate_d,
                                   curve, fallback, ships)

    diag = diagnostics_frame(diagnostics)
    predictions = score(models[f"{ships}/val"]["model"],
                        models[f"{ships}/val"]["eval"], max_games,
                        seed)["_predictions"]
    pit = pit_table(predictions["pit"].to_numpy(), f"gp_{ships}")
    coefficients = _coefficient_table(models, shipped)
    # Both Gate D verdicts land in the artifact, not only the one that shipped: "the
    # fitted arm failed and the fallback carried it" is the finding, and a table showing
    # one row would read as though only one candidate had ever been on the table.
    gates = pd.DataFrame(
        [{"gate": "A", **{k: v for k, v in probe.items() if k != "diagnostics"}},
         {"gate": "B", **gate_b_row},
         {"gate": "C", "still_falling": still_falling}]
        + [{"gate": "D", "ships": name == ships, **v} for name, v in verdicts.items()])

    # The games-played pmf, in long form, on the VALIDATION rows. `season_total.py` reads
    # it as an artifact rather than an import because that module runs without a CmdStan
    # toolchain; long form for the reason `residual_correlation.csv` is, since a column per
    # game index breaks the first time a season is not 82 games. Gate E therefore measures
    # the season-total contribution on validation too — the test-side pmf is written only by
    # `src/final_evaluation.py`.
    pmf_frame = _pmf_frame(models[f"{ships}/val"], max_games, ships)

    artifacts = {
        "metrics": (table.assign(probe_hours=probe["extrapolated_hours"]),
                    out_dir / "stan_games_played_metrics.csv"),
        "gp_pmf": (pmf_frame, out_dir / "stan_games_played_gp_pmf.csv"),
        "coefficients": (coefficients, out_dir / "stan_games_played_coefficients.csv"),
        "diagnostics": (diag, out_dir / "stan_games_played_diagnostics.csv"),
        "gates": (gates, out_dir / "stan_games_played_gates.csv"),
        "spell_shape": (shape, out_dir / "stan_games_played_spell_shape.csv"),
        "pit": (pit, out_dir / "stan_games_played_pit.csv"),
        "predictions": (predictions, out_dir / "stan_games_played_predictions.csv"),
        # Not optional: `dashboard/tabs/simulations.py::CHAIN` hard-codes this path as its
        # build check, and without it `make dashboard-audit`'s orphan check flags every
        # stan_games_played_* family as unreachable — trading one finding for six.
        "spell_process": (process, out_dir / "spell_process.csv"),
    }
    paths = {}
    for name, (data, dest) in artifacts.items():
        data.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(data):,} {name} rows → {dest}")
    print(f"\nSampler: max R-hat {diag['max_rhat'].max():.4f}, "
          f"{int(diag['divergences'].sum())} divergences over {len(diag)} fits, "
          f"{diag['wall_clock_s'].sum() / 60:.1f} min total")
    return paths


def _spell_shape(models: dict, panel: pd.DataFrame, val: pd.DataFrame,
                 max_games: int, cfg_gp: dict, seed: int) -> pd.DataFrame:
    """Simulated absence-spell lengths per arm, against the observed ones on the same rows.

    The observed side is the *validation* player-seasons' own within-tenure spells, so the
    comparison is population-matched — the same discipline Gate C needed.
    """
    n_sims = int(cfg_gp.get("shape_sims", 12))
    keys = set(zip(val["season"], val["player_id"]))
    interior = spell_classes(panel, "appearance")
    observed = interior[[(a, b) in keys for a, b in
                         zip(interior["season"], interior["player_id"])]
                        ]["spell_games"].to_numpy()

    arms: dict[str, np.ndarray] = {}
    hybrid = models.get("hybrid/val")
    if hybrid is not None:
        played, tiled = hybrid["model"].sequences(val, max_games, n_sims, seed)
        arms["hybrid"] = spell_lengths_from(played, tiled)

    cal = models.get("calibrated/val")
    if cal is not None:
        m = cal["model"]
        mu = m.incumbent.predict_mean(val)
        n = float(np.median(val["team_games"].to_numpy(float)))
        _, played = simulate_calibrated(
            val["team_games"].to_numpy(np.int64), mu,
            1.0 + (n - 1.0) * m.incumbent.rho, m.clustering, n_sims, seed,
            return_played=True)
        arms["calibrated_fallback"] = spell_lengths_from(
            played, np.tile(val["team_games"].to_numpy(int), n_sims))

    for label in ("full_window", "duration_covariates"):
        holder = models.get(f"{label}/val")
        if holder is None:
            continue
        heads, frame = holder["heads"], holder["eval"]
        d = int(np.argmin(np.abs(heads["onset"].alpha_draws
                                 - heads["onset"].alpha_draws.mean())))
        onset_a, onset_b = heads["onset"].shapes(frame, d)
        dur_a, dur_b = heads["duration"].shapes(frame, d)
        params = {"onset_a": onset_a, "onset_b": onset_b,
                  "dur_a": dur_a, "dur_b": dur_b}
        params["entry_a"], params["entry_b"] = heads["entry"].shapes(frame, d)
        params["exit_a"], params["exit_b"] = heads["exit"].shapes(frame, d)
        _, played = simulate_gp(frame["team_games"].to_numpy(np.int64), params,
                                n_sims, seed=seed, return_played=True)
        arms[label] = spell_lengths_from(
            played, np.tile(frame["team_games"].to_numpy(int), n_sims))

    return spell_shape_table(observed, arms)


def _gate_d(table: pd.DataFrame, variant: str, incumbent: pd.Series) -> dict:
    """Gate D for one arm, **on validation**: two non-regression bars and the tail.

    Two things changed here on 2026-08-05, and both were defects rather than preferences.

    **It reads the validation split.** `docs/games-played-plan.md` states Gate D's bars as
    the incumbent's *test* figures (10.795 / 0.096) and, three sections earlier, that
    selection reads validation and test confirms. Those conflict; the first implementation
    resolved it the wrong way and settled which model ships on the split that is not allowed
    to decide. Re-decided on validation the verdict reverses for both challengers — the
    tail, which is the declared win condition, flips sign on each.

    **The bars come from the incumbent's own row on the same split**, not from constants.
    A hard-coded 10.795 is a test-set number wearing a threshold's clothes; what the gate
    actually means is "does not regress against the head that already ships", and the only
    way to ask that on any split is to score the incumbent there too.
    """
    row = table[table["variant"] == variant].iloc[0]
    crps_bar = float(incumbent["val_crps"])
    pit_bar = float(incumbent["val_pit_ks"])
    out = {
        "arm": variant, "split": "validation",
        "crps": float(row["val_crps"]), "crps_bar": crps_bar,
        "crps_passes": bool(row["val_crps"] <= crps_bar),
        "pit_ks": float(row["val_pit_ks"]), "pit_ks_bar": pit_bar,
        "pit_passes": bool(row["val_pit_ks"] <= pit_bar),
        "predicted_below_41": float(row["val_predicted_below_41"]),
        "observed_below_41": float(row["val_observed_below_41"]),
        "predicted_below_60": float(row["val_predicted_below_60"]),
        "observed_below_60": float(row["val_observed_below_60"]),
        "tail_error": float(row["val_tail_error"]),
        "incumbent_tail_error": float(incumbent["val_tail_error"]),
        "tail_improves": bool(row["val_tail_error"] < incumbent["val_tail_error"]),
    }
    out["passes"] = bool(out["crps_passes"] and out["pit_passes"]
                         and out["tail_improves"])
    return out


def _pmf_frame(holder: dict, max_games: int, arm: str) -> pd.DataFrame:
    """The held-out games-played pmf, one row per (player-season, k) with mass above 0."""
    frame = holder["eval"]
    pmf = holder["model"].predict_pmf(frame, max_games)
    rows, cols = np.nonzero(pmf > 1e-9)
    return pd.DataFrame({
        "arm": arm,
        "season": frame["season"].to_numpy()[rows],
        "player_id": frame["player_id"].to_numpy()[rows],
        "team_games": frame["team_games"].to_numpy()[rows],
        "gp": cols,
        "p": pmf[rows, cols],
    })


def _simulated_curve(holder: dict, cfg_gp: dict, seed: int) -> pd.DataFrame:
    """Gate C's simulated hazard-by-streak curve, at the posterior mean."""
    frame = holder["eval"]
    heads, model = holder["heads"], holder["model"]
    d = int(np.argmin(np.abs(heads["onset"].alpha_draws
                             - heads["onset"].alpha_draws.mean())))
    onset_a, onset_b = heads["onset"].shapes(frame, d)
    dur_a, dur_b = heads["duration"].shapes(frame, d)
    params = {"onset_a": onset_a, "onset_b": onset_b, "dur_a": dur_a, "dur_b": dur_b}
    if model.oracle_tenure:
        params["pre"] = frame["pre_tenure"].fillna(0).to_numpy(np.int64)
        params["post"] = frame["post_tenure"].fillna(0).to_numpy(np.int64)
    else:
        params["entry_a"], params["entry_b"] = heads["entry"].shapes(frame, d)
        params["exit_a"], params["exit_b"] = heads["exit"].shapes(frame, d)
    return simulated_hazard_by_streak(frame["team_games"].to_numpy(np.int64), params,
                                      int(cfg_gp.get("sim_seasons", SIM_SEASONS)),
                                      seed)


def _fallback_table(panel: pd.DataFrame, design: pd.DataFrame, train: pd.DataFrame,
                    test: pd.DataFrame, clustering: float,
                    cfg_stan: dict) -> pd.DataFrame:
    """Option (b): hazards inverted from the incumbent's marginal and a measured C.

    Two lines and exact. It reproduces the incumbent's GP marginal **by construction**
    while getting the game-level clustering right, and it is worth emitting whichever arm
    wins because it is the invariant every arm should satisfy at its own fitted `C`.
    """
    l2 = float(cfg_stan.get("games_played", {}).get("glm_l2", GLM_L2))
    incumbent = BetaBinomialGLM(l2).fit(train)
    mu_star = incumbent.predict_mean(test)
    n = float(np.median(test["team_games"].to_numpy(float)))
    inflation_star = 1.0 + (n - 1.0) * incumbent.rho

    p, q = matched_transition_rates(panel, design)
    cal = closed_form_calibration(mu_star, inflation_star, clustering, n)
    stacked = variance_inflation(clustering, incumbent.rho, n)
    return pd.DataFrame([{
        "p_play_given_played": p, "q_play_given_missed": q,
        "clustering_C": clustering, "n_games": n,
        "incumbent_rho": incumbent.rho,
        "incumbent_inflation": inflation_star,
        "residual_rho": float(np.mean(cal["rho_frailty"])),
        "rho_markov": cal["rho_markov"],
        "mean_onset_hazard": float(np.mean(cal["onset"])),
        "mean_recovery_hazard": float(np.mean(cal["recovery"])),
        "stacked_inflation": stacked,
        "stacked_overshoot": stacked / inflation_star - 1.0}])


def _coefficient_table(models: dict, shipped: str) -> pd.DataFrame:
    """Posterior summaries for every head of the shipped arm.

    `not_rostered_share` of the fitting population rides along on the duration rows, per
    the plan's third failure mode: a duration head fitted on the full window is fitting
    roster mechanics, and the way to stop that being argued about is to report it beside
    every coefficient.
    """
    holder = models[f"{shipped}/val"]
    rows = []
    for head_name, head in holder["heads"].items():
        names = ["intercept"] + list(head.features)
        if head_name == "duration":
            draws = np.column_stack([head.alpha_draws, head.beta_draws]) \
                if head.features else head.alpha_draws[:, None]
            rows.append({"head": head_name, "term": "kappa",
                         "posterior_mean": float(head.kappa_draws.mean()),
                         "posterior_sd": float(head.kappa_draws.std(ddof=1)),
                         "q2_5": float(np.percentile(head.kappa_draws, 2.5)),
                         "q97_5": float(np.percentile(head.kappa_draws, 97.5))})
        else:
            draws = np.column_stack([head.alpha_draws, head.beta_draws])
            rows.append({"head": head_name, "term": "rho",
                         "posterior_mean": float(head.rho_draws.mean()),
                         "posterior_sd": float(head.rho_draws.std(ddof=1)),
                         "q2_5": float(np.percentile(head.rho_draws, 2.5)),
                         "q97_5": float(np.percentile(head.rho_draws, 97.5))})
        for j, term in enumerate(names[:draws.shape[1]]):
            rows.append({"head": head_name, "term": term,
                         "posterior_mean": float(draws[:, j].mean()),
                         "posterior_sd": float(draws[:, j].std(ddof=1)),
                         "q2_5": float(np.percentile(draws[:, j], 2.5)),
                         "q97_5": float(np.percentile(draws[:, j], 97.5))})
    return pd.DataFrame(rows)


def _spell_process_table(holder: dict, table: pd.DataFrame, gate_b_row: dict,
                         gate_d: dict, curve: pd.DataFrame,
                         fallback: pd.DataFrame, ships: str) -> pd.DataFrame:
    """`spell_process.csv` — what a simulator has to be **given**, in long form.

    The dashboard's registered slot, and deliberately a *parameter* table rather than a
    metrics one: a consumer building a season simulator needs the fitted hazards, the
    duration shape and the tenure factors, not the CRPS that justified them. The gate
    verdicts ride along so the page can say whether the process shipped.
    """
    heads = holder["heads"]
    label = ("calibrated_fallback" if ships == "calibrated"
             else (table.loc[table["selected"], "variant"].iloc[0]
                   if table["selected"].any() else "full_window"))
    chosen = table[table["variant"] == label].iloc[0]
    frame = holder["eval"]

    # The calibrated fallback has no fitted heads by construction — its hazards come from
    # inverting the variance identity — so it contributes the two-line calibration instead.
    rows: list[dict] = []
    if heads:
        rows += [
            {"component": "onset", "statistic": "mean_hazard",
             "value": float(np.mean(heads["onset"].mean(frame)))},
            {"component": "onset", "statistic": "dispersion_rho",
             "value": float(heads["onset"].rho)},
            {"component": "duration", "statistic": "mu_p_exit_first_game",
             "value": float(heads["duration"].mu)},
            {"component": "duration", "statistic": "kappa",
             "value": float(heads["duration"].kappa)},
            {"component": "duration", "statistic": "beta_a",
             "value": float(beta_shapes(heads["duration"].mu,
                                        heads["duration"].kappa)[0])},
            {"component": "duration", "statistic": "beta_b",
             "value": float(beta_shapes(heads["duration"].mu,
                                        heads["duration"].kappa)[1])},
        ]
        for name in ("entry", "exit"):
            if name in heads:
                rows += [{"component": name, "statistic": "mean_share",
                          "value": float(np.mean(heads[name].mean(frame)))},
                         {"component": name, "statistic": "dispersion_rho",
                          "value": float(heads[name].rho)}]
    else:
        rows += [{"component": "calibrated", "statistic": "rho_markov",
                  "value": float(fallback["rho_markov"].iloc[0])},
                 {"component": "calibrated", "statistic": "residual_rho",
                  "value": float(fallback["residual_rho"].iloc[0])},
                 {"component": "calibrated", "statistic": "mean_onset_hazard",
                  "value": float(fallback["mean_onset_hazard"].iloc[0])},
                 {"component": "calibrated", "statistic": "mean_recovery_hazard",
                  "value": float(fallback["mean_recovery_hazard"].iloc[0])}]
    rows += [
        {"component": "marginal", "statistic": "crps_games",
         "value": float(chosen["val_crps"])},
        {"component": "marginal", "statistic": "pit_ks", "value": float(chosen["val_pit_ks"])},
        {"component": "marginal", "statistic": "implied_overdispersion",
         "value": float(chosen["val_implied_overdispersion"])},
        {"component": "tail", "statistic": "predicted_share_below_41",
         "value": float(chosen["val_predicted_below_41"])},
        {"component": "tail", "statistic": "observed_share_below_41",
         "value": float(chosen["val_observed_below_41"])},
        {"component": "tail", "statistic": "predicted_share_below_60",
         "value": float(chosen["val_predicted_below_60"])},
        {"component": "tail", "statistic": "observed_share_below_60",
         "value": float(chosen["val_observed_below_60"])},
        {"component": "gate_b", "statistic": "onset_gain_over_floor",
         "value": float(gate_b_row["gain"])},
        {"component": "gate_d", "statistic": "passes",
         "value": float(gate_d["passes"])},
        {"component": "fallback", "statistic": "clustering_C",
         "value": float(fallback["clustering_C"].iloc[0])},
        {"component": "fallback", "statistic": "stacked_overshoot",
         "value": float(fallback["stacked_overshoot"].iloc[0])},
    ]
    for _, r in curve.iterrows():
        rows.append({"component": "hazard_by_streak",
                     "statistic": f"streak_{r['streak']}",
                     "value": float(r["hazard_simulated"])
                     if pd.notna(r.get("hazard_simulated")) else np.nan})
    out = pd.DataFrame(rows)
    out.insert(0, "arm", chosen["variant"])
    return out


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
