"""The minutes head: `min | available`, as successes out of the game's actual length.

The second link in the chain `availability -> min | available -> counts | min -> makes |
attempts`. Availability decides *whether* a player-game exists; this decides *how much*,
and its draw is then the exposure for all eleven component heads — which is why it is the
one head whose error propagates into every other one.

## Minutes are not a count, and the denominator is not 48

`min` is bounded above by the length of the game, so the likelihood is successes out of
trials with trials = that length. The prior attempt in `~/Documents/nba_stats` modelled it
as `normal(mu, sigma) T[0,48]` and paid for it twice: a `min == 48 -> 47.9` fudge against
the boundary, and a `filter(min < 48)` that **discarded every overtime game**.

`make game-length` (`data/features/game_length.parquet`) supplies the real denominator,
derived rather than fetched: five players are on the court at every moment, so a team's
summed minutes are exactly `5 x game length`. The two teams are two independent estimates
of the same quantity and they disagree on **0 of 37,986 games**. 5.93% of games go to
overtime, 1,650 player-games exceed 48 minutes and the observed maximum is 63.0 — so
truncating at 48 censors the top of the distribution in exactly the games where stars play
most. Joined to `component_targets.parquet` the specification is feasible on **100.0%** of
731,906 player-games with **zero** rows where `min > game_length`, maximum ratio exactly
1.0000. No clipping, no boundary hack.

## Season-collapsed, and what that does and does not estimate

Fitted on player-season rows: `y` = minutes played across the season, `n` = summed game
length over **the games he actually played**. That is the exact conditional the contract
asks for — minutes *given* availability — and it composes with the availability head as
`gp x minutes-per-played-game` rather than double-counting absences.

The collapse is the identity in `docs/predictions-plan.md`, so `beta` is what a
game-level fit would give. `rho` is **not**: a season total cannot distinguish a per-game
random effect from a per-season one, since iid per-game noise is diluted by ~1/G while a
shared season multiplier passes through in full. So the fitted `rho` here is season-level
heterogeneity, and `game_level_dispersion` measures the other quantity separately —
the simulator needs both, and conflating them would make simulated seasons far too tight.
Minutes are also the most serially dependent thing in the project (lag-1 excess +0.294,
10-game block variance inflation **2.43x**, `make serial-correlation`), which the
season-level `rho` absorbs in aggregate but which a per-game simulator has to reproduce
with an actual sequential process.

## Specification: the scale is the answer, and it is not age

Measured before this head existed (`availability.minutes_nonlinearity_probe`, and
replicated across validation *and* test, unlike the games-played arm which is a null):
a curved response in prior minutes pays, `minutes_per_game_lag1` carries essentially all
of it, and **splining `age` is actively worse** than the `age + age_sq` already in the
feature block. The mechanism is a floor at the bottom of the range, not a ceiling at the
top — mean next-season MPG runs 4.3 -> 10.5 (+6.1) against a parallel -2.0 decline above
26 mpg.

So the variants below spend flexibility on the prior-minutes term only, and the first
thing they try is **scale rather than curvature**: a logit link wants a logit-scale
predictor, exactly as the count heads' log link wants `log(prior rate)`. `logit_own` is
the one-term version of the fix; the quadratic and spline arms then ask whether curvature
buys anything *on top of* the right scale.

**Selection is on a validation split, and there is no test column at all.** This repo has
already produced one false positive whose paired bootstrap on test read [-0.079, -0.015]
with P(delta<0) = 99.7% and did not replicate; `src/models/held_out.py` now raises on the
held-out frame and `src/final_evaluation.py` reads it once, at the end.

Usage:
    python -m src.models.stan_minutes
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import SplineTransformer

from src.data.preprocess import (FIT_WINDOWS, FULL_WINDOW, TRAIN_VAL_WINDOW,
                                 TRAIN_WINDOW, fit_window, held_out_seasons)
from src.eda.availability import with_lags
from src.models.availability import (EPS, FEATURE_COLS, RHO_MAX, RHO_MIN,
                                     fit_dispersion)
from src.models.held_out import selection_split
from src.models.stan_availability import availability_design
from src.models.stan_utils import (YearTerm, compile_model, crps_from_samples,
                                   diagnostics_frame, ks_uniform,
                                   pit_from_samples, posterior,
                                   prior_sd_for_l2, rho_block, sample, standardized,
                                   thin, warn_if_unconverged)

MODEL = "betabinomial_glm"
OWN = "logit_share_lag1"
MIN_PRIOR_MINUTES = 200
MIN_GAMES = 10
SPLINE_KNOTS = 5
PREDICTIVE_SAMPLES = 1000
INTERCEPT_SCALE = 5.0
GLM_L2 = 1.0


# ── Target construction ───────────────────────────────────────────────────────

def minutes_targets(targets: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season): minutes played, and the trials denominator they came out of.

    Regular season only — every fitting frame in this project is, because the DK contest
    ends 4/4 and because playoff minutes are a role interaction whose *sign flips* with
    role (median playoff-to-regular MPG 0.505 bench / 0.761 rotation / 1.054 starter).
    Playoff minutes remain a prior-season workload feature and never a row to fit.

    Only games the player actually played contribute, on both sides of the ratio. That is
    what makes this the conditional `min | available` rather than a blend of minutes and
    absence, which the availability head already owns.
    """
    keys = ["season", "season_type", "game_id"]
    missing = [k for k in keys if k not in lengths.columns]
    if missing:
        raise ValueError(f"game_length frame is missing {missing}; run `make game-length`")

    played = targets[(targets["season_type"] == "regular") & (targets["played"] == 1)]
    merged = played.merge(lengths[keys + ["game_length", "reliable"]], on=keys, how="left")

    unmatched = int(merged["game_length"].isna().sum())
    if unmatched:
        # Loud rather than dropped. A missing length is a missing denominator, and
        # silently excluding those player-games would bias the rate upward for exactly the
        # players whose games went unmatched.
        raise ValueError(
            f"{unmatched:,} of {len(merged):,} played player-games have no game length. "
            f"`game_id` is int64 in the game logs and zero-padded in the box-score files — "
            f"check the key dtypes, and re-run `make game-length`.")
    if not merged["reliable"].all():
        bad = int((~merged["reliable"]).sum())
        print(f"  dropping {bad:,} player-games in games whose two teams' minutes "
              f"disagree (unreliable length)")
        merged = merged[merged["reliable"]]

    out = (merged.groupby(["player_id", "season"], as_index=False)
           .agg(minutes_played=("min", "sum"),
                length_played=("game_length", "sum"),
                games_played=("min", "size")))
    out["minutes_share"] = out["minutes_played"] / out["length_played"]
    return out


def as_trials(frame: pd.DataFrame) -> pd.DataFrame:
    """Integer successes/trials, with the rounding made explicit rather than assumed.

    Minutes are recorded to the second (98.4% of rows are non-integer), so the season
    total has to be rounded to enter a beta-binomial. The rounding is under one minute on
    a total of ~2,000 and cannot move any coefficient — but rounding *up* past the
    denominator would make the log-likelihood non-finite, which is the availability head's
    `gp > team_games` trap wearing different clothes. So the clamp is applied and counted.
    """
    out = frame.copy()
    out["trials"] = np.rint(out["length_played"].to_numpy(float)).astype(int)
    y = np.rint(out["minutes_played"].to_numpy(float)).astype(int)
    out["successes"] = np.minimum(y, out["trials"].to_numpy())
    out["rounding_clamped"] = (y > out["trials"].to_numpy()).astype(int)
    return out


def build_design(cfg: dict) -> pd.DataFrame:
    """Availability's feature block, joined to the minutes target and its own lag.

    Reusing `availability_design` is deliberate: it is the frame that carries `as_of_date`
    and runs `assert_point_in_time`, so the leakage guard covers this head too rather than
    being reimplemented beside it.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    seasons = cfg["data"]["seasons"]
    m_cfg = cfg.get("stan", {}).get("minutes", {})

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    lengths = pd.read_parquet(features_dir / "game_length.parquet")
    minutes = minutes_targets(targets, lengths)

    lagged = with_lags(minutes, seasons,
                       ["minutes_share", "minutes_played", "length_played",
                        "games_played"], max_lag=1)
    design = availability_design(cfg).merge(
        lagged, on=["player_id", "season"], how="inner")

    design = design.dropna(subset=["minutes_share_lag1", "minutes_share"])
    design = design[
        (design["total_minutes_lag1"] >= float(m_cfg.get("min_prior_minutes",
                                                         MIN_PRIOR_MINUTES)))
        & (design["games_played"] >= int(m_cfg.get("min_games", MIN_GAMES)))
        & (design["length_played"] > 0)]

    share = np.clip(design["minutes_share_lag1"].to_numpy(float), EPS, 1 - EPS)
    design[OWN] = np.log(share / (1 - share))
    return as_trials(design.reset_index(drop=True))


# ── The no-fit floor ──────────────────────────────────────────────────────────

def carry_forward(frame: pd.DataFrame) -> np.ndarray:
    """Prior minutes share x this season's realized game length. No fitting at all.

    The minutes analogue of `component_rates.carry_forward`, and mandatory for the same
    reason: a head that does not clear it is not a model. It conditions on realized
    exposure — the games he actually played — so it isolates the *rate* question and
    leaves availability to the head that owns it.
    """
    share = frame["minutes_share_lag1"].to_numpy(dtype=float)
    return np.clip(share, 0.0, 1.0) * frame["length_played"].to_numpy(dtype=float)


# ── Feature variants ──────────────────────────────────────────────────────────

def _spline(train: pd.DataFrame, frames: list[pd.DataFrame], col: str, n_knots: int
            ) -> tuple[list[pd.DataFrame], list[str]]:
    """Cubic B-spline basis, knots from **train** quantiles, linear extrapolation.

    Fitting the transformer on the pooled frame would leak the held-out distribution into
    the basis. Linear rather than polynomial extrapolation because a cubic runs away past
    the data range, which is where the sparse low-minutes rows live.
    """
    st = SplineTransformer(n_knots=n_knots, degree=3, extrapolation="linear",
                           include_bias=False)
    st.fit(train[[col]].to_numpy(dtype=float))
    names, out = [], []
    for frame in frames:
        basis = st.transform(frame[[col]].to_numpy(dtype=float))
        copy = frame.copy()
        names = [f"{col}__s{j}" for j in range(basis.shape[1])]
        for j, name in enumerate(names):
            copy[name] = basis[:, j]
        out.append(copy)
    return out, names


def variants(train: pd.DataFrame, test: pd.DataFrame, n_knots: int = SPLINE_KNOTS
             ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """Linear, right-scale, and two curvature arms on the prior-minutes term only.

    `age` is deliberately absent from every expansion. The career arc is real — MPG peaks
    at 27 and reaches 0.515 by 37 — but `age + age_sq` in the base block already fits it,
    and a spline on age measured *worse* on both splits.
    """
    base = list(FEATURE_COLS)
    without_own = [c for c in base if c != "minutes_per_game_lag1"]
    out = {"linear": (train, test, base),
           "logit_own": (train, test, without_own + [OWN])}

    sq = f"{OWN}__sq"
    tq, eq = train.copy(), test.copy()
    tq[sq] = tq[OWN] ** 2
    eq[sq] = eq[OWN] ** 2
    out["logit_own_quadratic"] = (tq, eq, without_own + [OWN, sq])

    (ts, es), names = _spline(train, [train, test], OWN, n_knots)
    out["logit_own_spline"] = (ts, es, without_own + names)
    return out


# ── The head ──────────────────────────────────────────────────────────────────

class StanMinutes:
    """Beta-binomial minutes head: successes = season minutes, trials = game length."""

    def __init__(self, features: list[str], l2: float = GLM_L2, name: str = "stan",
                 chains: int = 4, warmup: int = 1000, samples: int = 1000,
                 seed: int = 42, predictive_samples: int = PREDICTIVE_SAMPLES,
                 year_column: str | None = None, metric: str | None = None):
        self.features, self.l2, self.name = features, l2, name
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed
        self.predictive_samples = predictive_samples
        self.metric = metric
        self.year = YearTerm(year_column, seed=seed, stream=name)

    def fit(self, train: pd.DataFrame) -> "StanMinutes":
        (X,), self.scaler = standardized(train, [train], self.features)
        y = train["successes"].to_numpy(int)
        n = train["trials"].to_numpy(int)
        if (y > n).any():
            raise ValueError("successes exceed trials after rounding — see `as_trials`")

        share = float(np.clip(y.sum() / max(n.sum(), 1), EPS, 1 - EPS))
        model = compile_model(MODEL)
        fit, self.diagnostics = sample(
            model,
            {"N": len(train), "K": X.shape[1], "X": X, "n": n.tolist(), "y": y.tolist(),
             "beta_scale": prior_sd_for_l2(self.l2),
             "intercept_scale": INTERCEPT_SCALE, **self.year.data(train),
             # One dispersion for every row: `rho_block()` with no bins is the shared-rho
             # model exactly. The graded arm is availability's alone — see
             # docs/availability-window-plan.md and §6 for the same question here, which
             # is measured but not yet laddered.
             **rho_block(len(train))},
            chains=self.chains, warmup=self.warmup, samples=self.samples,
            seed=self.seed, label=self.name, metric=self.metric,
            # `rho` is a vector[n_rho] in the Stan source, so its init is a list even
            # when the vector has one entry.
            inits={"alpha": float(np.log(share / (1 - share))),
                   "beta": np.zeros(X.shape[1]).tolist(), "rho": [0.05]})
        warn_if_unconverged(self.diagnostics)

        draws = posterior(fit, ["alpha", "beta", "rho"])
        self.alpha_draws = draws["alpha"].reshape(-1)
        self.beta_draws = draws["beta"].reshape(len(self.alpha_draws), -1)
        self.rho_draws = draws["rho"].reshape(-1)
        self.rho = float(self.rho_draws.mean())
        self.year.absorb(fit)
        return self

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    def mu_draws(self, df: pd.DataFrame, keep: int) -> tuple[np.ndarray, np.ndarray]:
        idx = thin(len(self.alpha_draws), keep)
        # (rows x K) @ (K x draws) -> (rows x draws), then transposed to draws-major.
        eta = (self._design(df) @ self.beta_draws[idx].T
               + self.alpha_draws[idx][None, :] + self.year.shift(idx)[None, :])
        return 1.0 / (1.0 + np.exp(-np.clip(eta.T, -30, 30))), self.rho_draws[idx]

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        mus, _ = self.mu_draws(df, self.predictive_samples)
        return mus.mean(axis=0) * df["trials"].to_numpy(float)

    def predict_samples(self, df: pd.DataFrame, seed: int = 0) -> np.ndarray:
        """(draws x rows) minutes drawn from the posterior predictive.

        One beta-binomial draw per posterior draw, sampled as `p ~ Beta(a, b)` then
        `y ~ Binomial(n, p)` rather than through `scipy.stats.betabinom.rvs`, which is
        orders of magnitude slower at this shape. The support runs 0..~4,000 minutes, so
        an explicit pmf grid — what the availability head uses over 0..83 games — is not
        an option here.
        """
        mus, rhos = self.mu_draws(df, self.predictive_samples)
        rng = np.random.default_rng(seed)
        a, b = beta_shapes(mus, rhos[:, None])
        p = rng.beta(a, b)
        return rng.binomial(df["trials"].to_numpy(int)[None, :], p).astype(float)


def beta_shapes(mu: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(mu, EPS, 1 - EPS)
    rho = np.clip(rho, RHO_MIN, RHO_MAX)
    scale = (1.0 - rho) / rho
    return mu * scale, (1.0 - mu) * scale


class FloorMinutes:
    """The no-fit floor, wrapped in the same beta-binomial so CRPS is comparable.

    The mean is pure arithmetic — prior share carried forward — and only the dispersion is
    fitted, on the floor's own training residuals. That is the pattern `availability.py`
    already uses to score its point baselines: every candidate emits a distribution from
    the same family, so the comparison is about the mean function alone.
    """

    name = "carry_forward"

    def fit(self, train: pd.DataFrame) -> "FloorMinutes":
        mu = np.clip(carry_forward(train) / train["trials"].to_numpy(float),
                     EPS, 1 - EPS)
        self.rho = fit_dispersion(train["successes"].to_numpy(int),
                                  train["trials"].to_numpy(int), mu)
        return self

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        return carry_forward(df)

    def predict_samples(self, df: pd.DataFrame, seed: int = 0) -> np.ndarray:
        mu = np.clip(carry_forward(df) / df["trials"].to_numpy(float), EPS, 1 - EPS)
        rng = np.random.default_rng(seed)
        a, b = beta_shapes(np.repeat(mu[None, :], PREDICTIVE_SAMPLES, axis=0),
                           np.full((PREDICTIVE_SAMPLES, 1), self.rho))
        return rng.binomial(df["trials"].to_numpy(int)[None, :],
                            rng.beta(a, b)).astype(float)


# ── Game-level dispersion — a different quantity from the fitted rho ──────────

def game_level_dispersion(targets: pd.DataFrame, lengths: pd.DataFrame,
                          min_games: int = 20,
                          window: str = FULL_WINDOW) -> dict:
    """Within-player-season, per-game overdispersion of `min` against its own mean.

    The season-collapsed fit cannot see this: a shared season multiplier passes its full
    relative overdispersion into the season total, while iid game noise is diluted by
    ~1/G. The simulator draws minutes **per game**, so it needs this number and not the
    fitted season-level `rho` — using the season one would make every simulated game far
    too close to the player's average.

    Measured in-sample against each player-season's own realized share, which biases the
    estimate slightly *downward* (the mean is fitted from the same rows). It is reported
    as a floor on the game-level dispersion for that reason.

    `window` is the second in-sample question and a different one: this is a **simulator
    input**, so measuring it over every season would calibrate it on the seasons the
    simulator is later scored against. `train_val` is the one to consume; `full` ships
    beside it so the difference is measured rather than assumed.
    """
    keys = ["season", "season_type", "game_id"]
    played = targets[(targets["season_type"] == "regular") & (targets["played"] == 1)]
    merged = played.merge(lengths[keys + ["game_length"]], on=keys, how="inner")
    merged = merged[merged["game_length"] > 0]
    merged = fit_window(merged, window)

    grp = merged.groupby(["player_id", "season"])
    merged = merged.assign(n_games=grp["min"].transform("size"),
                           season_min=grp["min"].transform("sum"),
                           season_len=grp["game_length"].transform("sum"))
    merged = merged[merged["n_games"] >= min_games]

    mu = np.clip((merged["season_min"] / merged["season_len"]).to_numpy(float),
                 EPS, 1 - EPS)
    n = np.rint(merged["game_length"].to_numpy(float)).astype(int)
    y = np.minimum(np.rint(merged["min"].to_numpy(float)).astype(int), n)
    rho = fit_dispersion(y, n, mu)
    median_n = float(np.median(n))
    return {"metric": "game_level_rho", "fit_window": window, "rho": float(rho),
            "implied_overdispersion": float(1 + (median_n - 1) * rho),
            "n_player_games": int(len(merged)),
            "note": "in-sample against each player-season's own mean; a floor"}


# ── Evaluation ────────────────────────────────────────────────────────────────

def score(model, frame: pd.DataFrame, label: str, seed: int = 0) -> dict:
    y = frame["successes"].to_numpy(float)
    pred = model.predict_mean(frame)
    samples = model.predict_samples(frame, seed)
    u = pit_from_samples(samples, y, seed)
    return {
        "variant": label,
        "n": len(frame),
        "r2_minutes": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "mae_minutes": float(np.abs(y - pred).mean()),
        "crps_minutes": float(crps_from_samples(samples, y).mean()),
        "pit_ks": ks_uniform(u),
        "rho": float(getattr(model, "rho", np.nan)),
        "bias_minutes": float((pred - y).mean()),
    }


def sweep(train: pd.DataFrame, val: pd.DataFrame, cfg_stan: dict, n_knots: int
          ) -> tuple[pd.DataFrame, list[dict]]:
    """Every variant on the VALIDATION split. The test seasons are not touched.

    This used to fit each variant twice — once on `train` scored against `val`, once refit
    on `train + val` scored against `test` — and report both columns. Two things were wrong
    with that, and `src/models/held_out.py` now prevents both. The test column was an
    invitation to select on it, which the games-played head did and had to reverse; and the
    refit meant a val/test disagreement conflated the evaluation rows with the training data
    and the chain length, so it could not serve as the replication check it looked like.

    The held-out number is taken once, by `src/final_evaluation.py`.
    """
    seed = int(cfg_stan.get("seed", 42))
    fast = {"warmup": int(cfg_stan.get("select_warmup", 500)),
            "samples": int(cfg_stan.get("select_samples", 500))}
    full = {"warmup": int(cfg_stan.get("warmup", 1000)),
            "samples": int(cfg_stan.get("samples", 1000))}
    chains = int(cfg_stan.get("chains", 4))

    val_variants = variants(train, val, n_knots)

    rows, diagnostics = [], []
    floor_val = score(FloorMinutes().fit(train), val, "carry_forward", seed)
    rows.append({"variant": "carry_forward", "n_features": 0,
                 "val_crps": floor_val["crps_minutes"],
                 "val_r2": floor_val["r2_minutes"], "val_mae": floor_val["mae_minutes"],
                 "val_pit_ks": floor_val["pit_ks"], "val_rho": floor_val["rho"],
                 "val_bias": floor_val["bias_minutes"]})

    for label in val_variants:
        v_tr, v_te, v_features = val_variants[label]
        # Full-length chains now: with the test side gone there is no reason to run
        # selection short, and the shorter chains were a second confound in the old
        # val/test comparison.
        v_model = StanMinutes(v_features, name=f"{label}/val", chains=chains,
                              seed=seed, **full).fit(v_tr)
        v = score(v_model, v_te, label, seed)
        diagnostics.append(v_model.diagnostics)
        rows.append({"variant": label, "n_features": len(v_features),
                     "val_crps": v["crps_minutes"], "val_r2": v["r2_minutes"],
                     "val_mae": v["mae_minutes"], "val_pit_ks": v["pit_ks"],
                     "val_rho": v["rho"], "val_bias": v["bias_minutes"]})

    out = pd.DataFrame(rows)
    fitted = out[out["variant"] != "carry_forward"]
    best = fitted.loc[fitted["val_crps"].idxmin(), "variant"]
    out["selected"] = out["variant"] == best
    floor_r2 = float(out.loc[out["variant"] == "carry_forward", "val_r2"].iloc[0])
    floor_crps = float(out.loc[out["variant"] == "carry_forward", "val_crps"].iloc[0])
    out["beats_floor"] = (out["val_r2"] > floor_r2) & (out["val_crps"] < floor_crps)
    out.loc[out["variant"] == "carry_forward", "beats_floor"] = True
    return out, diagnostics


# `inner_split` used to live here — carve validation out of train, then `as_plain` the
# right-hand side because `split_seasons` guards it. `held_out.selection_split` is that
# function, generalized to start from the full design, and having both meant two ways to
# reach the same frames with only one of them named after the rule it enforces. Deleted
# 2026-08-06; `stan_games_played` and `stan_composition` were its only other callers and
# both now take `selection_split` directly.


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_stan = cfg.get("stan", {})
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    n_knots = int(cfg_stan.get("minutes", {}).get("spline_knots", SPLINE_KNOTS))

    print("Stan minutes head — successes out of actual game length, never 48")
    design = build_design(cfg)
    train, val = selection_split(design, test_seasons)

    clamped = int(design["rounding_clamped"].sum())
    print(f"  {len(design):,} player-seasons. The test split is LOCKED — selection reads "
          f"VALIDATION only\n  (src/models/held_out.py); the held-out number is taken once "
          f"by `make final-evaluation`.")
    print(f"  {len(train):,} fit / {len(val):,} select "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")
    print(f"  trials = summed game length over games played: median "
          f"{design['trials'].median():,.0f}, max {design['trials'].max():,.0f} minutes; "
          f"{clamped} rows clamped by rounding")
    print(f"  realized minutes share: mean {design['minutes_share'].mean():.4f}, "
          f"max {design['minutes_share'].max():.4f} (1.0 would be every minute of "
          f"every game)")

    table, diagnostics = sweep(train, val, cfg_stan, n_knots)
    print("\nVariant sweep (CRPS in minutes, lower is better):")
    print(table[["variant", "n_features", "val_crps", "val_r2", "val_mae",
                 "val_pit_ks", "selected", "beats_floor"]].round(4).to_string(index=False))
    selected = table.loc[table["selected"], "variant"].iloc[0]
    floor = table[table["variant"] == "carry_forward"].iloc[0]
    chosen = table[table["variant"] == selected].iloc[0]
    print(f"\n  Selected on VALIDATION: {selected}. There is no test column — this repo "
          f"has already\n  shipped one false positive selected on test and caught a "
          f"second.")
    print(f"  vs the no-fit floor: R2 {chosen['val_r2']:.4f} against "
          f"{floor['val_r2']:.4f} ({chosen['val_r2'] - floor['val_r2']:+.4f}), "
          f"CRPS {chosen['val_crps']:.3f} against {floor['val_crps']:.3f} "
          f"({chosen['val_crps'] - floor['val_crps']:+.3f} minutes)")
    if not bool(chosen["beats_floor"]):
        print("  /!\\  The selected variant does NOT clear the no-fit floor. Per "
              "CLAUDE.md that is not a\n       model — check the specification before "
              "reporting it as one.")

    targets = pd.read_parquet(Path(cfg["data"]["features_dir"])
                              / "component_targets.parquet")
    lengths = pd.read_parquet(Path(cfg["data"]["features_dir"]) / "game_length.parquet")
    game_rhos = [game_level_dispersion(targets, lengths, window=w) for w in FIT_WINDOWS]
    by_window = {r["fit_window"]: r for r in game_rhos}
    game_rho = by_window[FULL_WINDOW]
    season_rho = float(chosen["val_rho"])
    print(f"\nTwo dispersions, and they are different quantities:")
    print(f"  season-level rho (what this fit estimates): {season_rho:.5f}")
    print(f"  game-level rho   (what the simulator needs): {game_rho['rho']:.5f} "
          f"-> {game_rho['implied_overdispersion']:.2f}x binomial over "
          f"{game_rho['n_player_games']:,} player-games")
    print("  A season total cannot separate a per-game random effect from a per-season "
          "one:\n  iid game noise is diluted by ~1/G while a shared season multiplier "
          "passes through in full.\n  Drawing per-game minutes from the season-level rho "
          "would make every simulated game far\n  too close to the player's average.")
    print(f"\n  All three fit windows, because this is a simulator INPUT and calibrating "
          f"it on the seasons\n  the simulator is scored against is leakage the split "
          f"cannot catch. Which one to consume is\n  decided by what the number will be "
          f"scored against, not by which is widest:")
    consumer = {FULL_WINDOW: "production (2026-27)",
                TRAIN_VAL_WINDOW: "the one-shot test readout",
                TRAIN_WINDOW: "the realized 2022-23 / 2023-24 backtest"}
    for window in FIT_WINDOWS:
        row = by_window[window]
        held = ", ".join(held_out_seasons(targets, window=window)) or "nothing"
        print(f"    {window:9s} rho {row['rho']:.5f} -> "
              f"{row['implied_overdispersion']:.2f}x over "
              f"{row['n_player_games']:,} player-games; holds out {held}\n"
              f"              -> {consumer[window]}")

    diag = diagnostics_frame(diagnostics)
    artifacts = {
        "metrics": (table, out_dir / "stan_minutes_metrics.csv"),
        "diagnostics": (diag, out_dir / "stan_minutes_diagnostics.csv"),
        # The season-level row is tagged `train_val` in the sense the label carries
        # everywhere here — "excludes the held-out seasons" — which is still true and now
        # conservative: since the sweep became validation-only the selected variant fits on
        # `train` alone, a strict subset. The game-level rows carry both windows.
        "dispersion": (pd.DataFrame(game_rhos + [{"metric": "season_level_rho",
                                                  "fit_window": TRAIN_VAL_WINDOW,
                                                  "rho": season_rho}]),
                       out_dir / "stan_minutes_dispersion.csv"),
    }
    paths = {}
    for name, (frame, dest) in artifacts.items():
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")
    print(f"\nSampler: max R-hat {diag['max_rhat'].max():.4f}, "
          f"{int(diag['divergences'].sum())} divergences over {len(diag)} fits, "
          f"{diag['wall_clock_s'].sum():.0f}s total")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
