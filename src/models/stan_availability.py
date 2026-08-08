"""The availability head as a full Bayesian fit — the same likelihood, in Stan.

`src/models/availability.py::BetaBinomialGLM` is a validated point-MLE beta-binomial:
held-out CRPS 10.795 games on 911 test rows, 19 features, beating ridge (10.896), a GBM
(10.888) and a league/age baseline (13.614), and worth −211 dk_pts of season-total MAE.
This module ports **that exact likelihood** to Stan and draws `(beta, rho)` from the
posterior instead of fixing them at the optimum.

## Why, given the marginal metric will barely move

Stated plainly because the wrong justification is easy to reach for: at ~10,300 training
rows against 20 parameters the posterior is sharply concentrated, so posterior-mean
coefficients and the MLE agree closely and GP-marginal CRPS should improve very little. If
the case were argued on CRPS it would fail.

**The argument is the joint distribution across players.** Every player shares `beta`, so
one posterior draw shifts the whole board's availability together — a correctly calibrated
source of cross-player correlation that arrives free with the fit rather than as an
invented copula. For a draft portfolio "how wrong could my entire board be at once" is a
different and more important question than "how wrong is this one player", and only the
posterior answers it. `docs/predictions-plan.md` needs exactly that and nothing in this
repo previously supplied it.

## What this module checks, and the three rows it prints

The port is verified rather than assumed, by fitting both and scoring them with the *same*
code — `evaluate`, `crps` and `pit_values` are imported from the MLE module, not
reimplemented, so a metric difference cannot be a metric-implementation difference:

- `beta_binomial_mle` — the existing head, unchanged. The reference.
- `stan_plug_in` — posterior means substituted for the MLE's optimum, one beta-binomial
  per row. This is the row that should *match* the MLE, because the prior is set to
  `1/sqrt(2*l2)`, which makes the posterior mode exactly the penalized MLE.
- `stan_posterior` — the predictive distribution properly integrated over the posterior,
  `p(y) = mean_s BetaBinom(y | n, mu_s, rho_s)`. The row the simulator should consume.

**The integrated predictive is not necessarily *wider* per player, and expecting it to be
is a trap.** By the law of total variance the mixture adds `Var_th(E[Y|th])`, but it also
replaces `Var(Y|th_bar)` with `E_th[Var(Y|th)]`, and the conditional variance
`n*mu*(1-mu)*[1 + (n-1)*rho]` is **concave in mu** — so Jensen pushes the other way.
Measured on this design the two nearly cancel (+0.046 against -0.082 games^2, verified
against the pmf to 1.3e-10). The marginal width is a red herring; see
`board_correlation` for the quantity that is not.

## The trap that carries over, and the one that does not

**`n = max(team_games, gp)` is still required.** 13 traded player-seasons (0.12%) have
`gp > team_games`, because their two teams' schedules overlap. `beta_binomial_lpmf` is
undefined there, and since the log-likelihood is a sum those rows take the whole fit down
at *every* value of rho. Under HMC this is worse than under L-BFGS-B, not better: a
non-finite target poisons the trajectory rather than merely stopping an optimizer.
`build_design` already applies the fix, and `assert_binomial_support` re-checks it here
because a silent violation is fatal and cheap to rule out.

**The numeric-gradient failure does not carry over.** The MLE needs an analytic gradient
and an alternating fit because L-BFGS-B on a ~1e5-magnitude objective stops on
finite-difference noise. Stan differentiates the model exactly, so `beta` and `rho` are
sampled jointly with no alternation.

## Which rows the three numbers are measured on

**Validation**, since 2026-08-05. Every figure printed here — the CRPS triple, the board
correlation, the PIT deciles — describes the fit on `train` scored against the validation
seasons, and `src/models/held_out.py` raises on anything that reaches past them.

That is a demotion in what the numbers *are*, and worth stating plainly. A port check is
not a selection, so scoring it on the held-out seasons was never the failure mode the lock
was built for. But it is also not the end-of-project measurement, and a module that reads
test "just to see" is exactly how the test column stops feeling special — which is how the
games-played head came to settle a shipping decision on it. `board_correlation` is the
sharper case: it is a **simulator input**, a statement about how much a whole draft board
moves together, so calibrating it on the seasons the simulator is later backtested against
is leakage the split cannot catch, in the same way `game_level_dispersion` and the residual
copula already take `train_val`.

`fit_and_score` is the whole measurement, and `src/final_evaluation.py` calls **the same
function** on `(full_train, test)` when the workflow is finished. So the held-out reading is
not a reimplementation of this one; it is this one, run once, on the other frame.

Usage:
    python -m src.models.stan_availability
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import betabinom

from src.features.availability import build_panel, season_availability
from src.models.availability import (EPS, FEATURE_COLS, RHO_MAX, RHO_MIN,
                                     AvailabilityModel, BetaBinomialGLM,
                                     _sigmoid, build_design, evaluate,
                                     pit_table, season_start_dates)
from src.models.held_out import selection_split
from src.models.stan_utils import (YearTerm, compile_model, diagnostics_frame,
                                   posterior, prior_sd_for_l2, sample,
                                   standardized, thin, warn_if_unconverged)

MODEL = "betabinomial_glm"

# Draws kept for the posterior-predictive mixture. Each draw costs one (rows x games+1)
# beta-binomial evaluation, so this trades wall clock against Monte Carlo error in the
# predictive. 400 puts the MC error on a CRPS of ~10.8 games well below 0.001.
PREDICTIVE_DRAWS = 400
# Draws pushed through scipy at once. Purely a memory knob: 60 x 911 x 84 doubles is
# ~37 MB, where the full 400 at once would be ~245 MB.
CHUNK = 60

INTERCEPT_SCALE = 5.0


def assert_binomial_support(design: pd.DataFrame) -> pd.DataFrame:
    """`0 <= gp <= team_games` on every row, or the log-likelihood is non-finite.

    `build_design` already takes `n = max(team_games, gp)`, so this should never fire. It
    is here because the failure is silent in the data and catastrophic in the sampler:
    thirteen bad rows out of ten thousand make the target non-finite everywhere, and the
    symptom surfaces as an uninterpretable initialization error rather than as a data
    problem.
    """
    y = design["gp"].to_numpy(int)
    n = design["team_games"].to_numpy(int)
    bad = (y < 0) | (n < 0) | (y > n)
    if bad.any():
        rows = design.loc[bad, ["player_id", "season", "gp", "team_games"]].head()
        raise ValueError(
            f"{int(bad.sum())} rows violate 0 <= gp <= team_games — the beta-binomial "
            f"likelihood is undefined there and the summed target is non-finite at every "
            f"rho:\n{rows.to_string(index=False)}")
    return design


def availability_design(cfg: dict) -> pd.DataFrame:
    """The point-in-time-checked player-season design, from cache when it exists.

    `make availability` already writes `availability_panel.parquet` and
    `availability_features.parquet`; rebuilding them from 30 seasons of CSVs costs a
    couple of minutes and produces the same frames. The cache is used when present and
    the source is printed either way, because a silently stale artifact would move every
    number downstream of it.

    Shared with `stan_minutes.py`, which needs the identical feature block — the minutes
    head is `min | available`, the next link in the same chain.
    """
    from src.features.availability import load_artifacts

    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    seasons = cfg["data"]["seasons"]

    panel_path = features_dir / "availability_panel.parquet"
    try:
        panel, frame = load_artifacts(features_dir)
        frame = frame[frame["window"] == "full"].drop(columns=["window"])
        stamp = pd.Timestamp(panel_path.stat().st_mtime, unit="s")
        print(f"  availability artifacts from cache ({stamp:%Y-%m-%d %H:%M}) — "
              f"run `make availability` if the panel is stale")
    except FileNotFoundError:
        print("  no cached availability artifacts; rebuilding the panel from raw logs")
        panel = build_panel(seasons, raw_dir)
        frame = season_availability(panel, "full")

    design = build_design(frame, seasons, raw_dir, season_start_dates(panel))
    return assert_binomial_support(design)


class StanAvailability(AvailabilityModel):
    """Beta-binomial availability head, fitted by NUTS.

    Subclasses the MLE module's `AvailabilityModel` on purpose: `evaluate` then scores it
    through exactly the same code path as the other four candidates, so the comparison is
    about the fit and not about two implementations of CRPS.
    """

    def __init__(self, l2: float = 1.0, features: list[str] | None = None,
                 pmf_mode: str = "posterior", name: str | None = None,
                 chains: int = 4, warmup: int = 1000, samples: int = 1000,
                 seed: int = 42, predictive_draws: int = PREDICTIVE_DRAWS,
                 year_column: str | None = None, metric: str | None = None):
        if pmf_mode not in ("posterior", "plug_in"):
            raise ValueError(f"pmf_mode must be 'posterior' or 'plug_in'; got {pmf_mode!r}")
        self.l2 = l2
        self.features = features or FEATURE_COLS
        self.pmf_mode = pmf_mode
        self.name = name or f"stan_{pmf_mode}"
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed
        self.predictive_draws = predictive_draws
        self.metric = metric
        self.year = YearTerm(year_column, seed=seed, stream=self.name)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "StanAvailability":
        assert_binomial_support(train)
        (X,), self.scaler = standardized(train, [train], self.features)
        y = train["gp"].to_numpy(int)
        n = train["team_games"].to_numpy(int)

        data = {
            "N": len(train), "K": X.shape[1], "X": X,
            "n": n.tolist(), "y": y.tolist(),
            # The identity that makes this a port: an L2 penalty of `l2` on the
            # standardized coefficients IS a normal(0, 1/sqrt(2*l2)) prior, so the
            # posterior mode here is the MLE module's optimum rather than a nearby
            # quantity that happens to resemble it.
            "beta_scale": prior_sd_for_l2(self.l2),
            "intercept_scale": INTERCEPT_SCALE,
            **self.year.data(train),
        }
        share = float(np.clip(y.sum() / max(n.sum(), 1), EPS, 1 - EPS))
        inits = {"alpha": float(np.log(share / (1 - share))),
                 "beta": np.zeros(X.shape[1]).tolist(),
                 "rho": 0.23}      # the measured overdispersion, as a starting point

        model = compile_model(MODEL)
        fit, self.diagnostics = sample(
            model, data, chains=self.chains, warmup=self.warmup,
            samples=self.samples, seed=self.seed, label=self.name, inits=inits,
            metric=self.metric)
        warn_if_unconverged(self.diagnostics)

        draws = posterior(fit, ["alpha", "beta", "rho"])
        self.alpha_draws = draws["alpha"].reshape(-1)
        self.beta_draws = draws["beta"].reshape(len(self.alpha_draws), -1)
        self.rho_draws = draws["rho"].reshape(-1)

        # Posterior means, in the same layout as `BetaBinomialGLM.beta` (intercept first)
        # so the two coefficient vectors can be diffed element-wise.
        self.beta = np.r_[self.alpha_draws.mean(), self.beta_draws.mean(axis=0)]
        self.rho = float(self.rho_draws.mean())
        self.year.absorb(fit)
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    def mu_draws(self, df: pd.DataFrame, keep: int | None = None) -> tuple[np.ndarray,
                                                                          np.ndarray]:
        """(draws x rows) mean and the matching rho draws."""
        idx = thin(len(self.alpha_draws), keep or self.predictive_draws)
        # (rows x K) @ (K x draws) -> (rows x draws), then transposed to draws-major.
        eta = (self._design(df) @ self.beta_draws[idx].T
               + self.alpha_draws[idx][None, :] + self.year.shift(idx)[None, :])
        return _sigmoid(eta).T, self.rho_draws[idx]

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        """Posterior mean of mu, or mu at the posterior mean, depending on the mode.

        The distinction is small but real — `E[mu]` and `mu(E[beta])` differ by Jensen's
        inequality through `inv_logit` — and keeping them separate is what lets the
        plug-in row be a like-for-like comparison against the MLE.
        """
        if self.pmf_mode == "plug_in":
            return _sigmoid(self.beta[0] + self._design(df) @ self.beta[1:])
        return self.mu_draws(df)[0].mean(axis=0)

    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        n = df["team_games"].to_numpy(int)
        k = np.arange(int(max_games) + 1)
        if self.pmf_mode == "plug_in":
            return _plug_in_pmf(n, self.predict_mean(df), self.rho, k)

        # The predictive properly integrated over the posterior: a mixture of one
        # beta-binomial per draw, not one beta-binomial at the average parameter. This is
        # the whole reason for fitting in Stan, and it is strictly wider than the plug-in.
        mus, rhos = self.mu_draws(df)
        out = np.zeros((len(df), len(k)))
        for lo in range(0, len(mus), CHUNK):
            block, rho_block = mus[lo:lo + CHUNK], rhos[lo:lo + CHUNK]
            a, b = _shapes(block, rho_block[:, None])
            pmf = betabinom.pmf(k[None, None, :], n[None, :, None],
                                a[:, :, None], b[:, :, None])
            out += np.nan_to_num(pmf).sum(axis=0)
        return out / len(mus)


def _shapes(mu: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(mu, EPS, 1 - EPS)
    scale = (1.0 - np.clip(rho, RHO_MIN, RHO_MAX)) / np.clip(rho, RHO_MIN, RHO_MAX)
    return mu * scale, (1.0 - mu) * scale


def _plug_in_pmf(n: np.ndarray, mu: np.ndarray, rho: float, k: np.ndarray) -> np.ndarray:
    a, b = _shapes(np.asarray(mu, dtype=float), np.asarray(float(rho)))
    return np.nan_to_num(betabinom.pmf(k[None, :], n[:, None], a[:, None], b[:, None]))


# ── The port check ────────────────────────────────────────────────────────────

def coefficient_comparison(mle: BetaBinomialGLM, stan: StanAvailability,
                           features: list[str]) -> pd.DataFrame:
    """MLE optimum against the posterior, coefficient by coefficient.

    Both are fitted on the identical standardized design, so the vectors are directly
    comparable entry for entry. `z_from_mle` — how many posterior standard deviations the
    MLE sits from the posterior mean — is the column that matters: agreement means the
    port is faithful, and a large value on any single coefficient localizes a
    discrepancy that an aggregate norm would average away.
    """
    names = ["intercept"] + list(features)
    posterior_sd = np.r_[stan.alpha_draws.std(ddof=1),
                         stan.beta_draws.std(axis=0, ddof=1)]
    lo = np.r_[np.percentile(stan.alpha_draws, 2.5),
               np.percentile(stan.beta_draws, 2.5, axis=0)]
    hi = np.r_[np.percentile(stan.alpha_draws, 97.5),
               np.percentile(stan.beta_draws, 97.5, axis=0)]
    out = pd.DataFrame({
        "term": names, "mle": mle.beta, "posterior_mean": stan.beta,
        "posterior_sd": posterior_sd, "q2_5": lo, "q97_5": hi,
    })
    out["difference"] = out["posterior_mean"] - out["mle"]
    out["z_from_mle"] = out["difference"] / out["posterior_sd"].replace(0, np.nan)
    out["mle_inside_95"] = (out["mle"] >= out["q2_5"]) & (out["mle"] <= out["q97_5"])
    rho = pd.DataFrame([{
        "term": "rho", "mle": mle.rho, "posterior_mean": stan.rho,
        "posterior_sd": float(stan.rho_draws.std(ddof=1)),
        "q2_5": float(np.percentile(stan.rho_draws, 2.5)),
        "q97_5": float(np.percentile(stan.rho_draws, 97.5)),
    }])
    rho["difference"] = rho["posterior_mean"] - rho["mle"]
    rho["z_from_mle"] = rho["difference"] / rho["posterior_sd"]
    rho["mle_inside_95"] = (rho["mle"] >= rho["q2_5"]) & (rho["mle"] <= rho["q97_5"])
    return pd.concat([out, rho], ignore_index=True)


PORTFOLIO_SIZES = (12, 15, 30, 150, None)      # None = the whole held-out board


def board_correlation(stan: StanAvailability, frame: pd.DataFrame,
                      sizes: tuple = PORTFOLIO_SIZES, n_subsets: int = 200,
                      seed: int = 0) -> pd.DataFrame:
    r"""The thing the posterior buys that a point estimate cannot: whole-board covariance.

    Every player's mean is a function of one shared `beta`, so a posterior draw moves the
    entire board **together**. That is a genuine, correctly calibrated source of
    cross-player correlation which arrives free with the fit, rather than an invented
    copula — and for a draft portfolio "how wrong could my whole board be at once" is a
    different and more important question than "how wrong is this one player".

    Decomposed exactly by the law of total variance on the board total `T = sum_i Y_i`:

        Var(T) = E_th[ sum_i Var(Y_i | th) ]  +  Var_th( sum_i E[Y_i | th] )
                 \_____ independent _____/       \_____ shared beta _____/

    The first term is all a plug-in model has: with parameters fixed, what is left is
    independent across players and grows as **sqrt(N)**. The second is exactly zero under a
    point estimate and grows as **N**, because it is perfectly correlated across players.

    **So the size of the portfolio decides whether this matters at all, and it is reported
    across sizes rather than as one number.** The ratio of the two terms scales as sqrt(N):
    measured here it is a fraction of a percent on a 15-player roster and several percent
    across the whole board. Quoting only the board figure would badly oversell what the
    posterior does for a single draft roster; quoting only the roster figure would miss
    that it is the dominant term for board-wide exposure across many lineups.

    Subsets are drawn at random and averaged, so `inflation` is measured on real players
    rather than extrapolated from the full-board number under an equal-variance assumption.

    **`frame` is the validation board, not the held-out one.** This is a simulator input —
    "how much does my whole board move together" is a number the simulator is *given* — so
    measuring it on the seasons the simulator is later backtested against would be leakage
    of the kind the split cannot catch, exactly as for `stan_minutes.game_level_dispersion`
    and the residual copula. The board size therefore tracks the validation seasons' player
    count rather than the held-out one's.
    """
    mus, rhos = stan.mu_draws(frame)
    n = frame["team_games"].to_numpy(float)
    conditional = n * mus * (1.0 - mus) * (1.0 + (n - 1.0) * rhos[:, None])
    means = n * mus                                     # E[Y_i | theta], per draw per player

    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size is None or size >= len(frame):
            size, subsets = len(frame), [np.arange(len(frame))]
        else:
            subsets = [rng.choice(len(frame), size, replace=False)
                       for _ in range(n_subsets)]
        independent = np.array([np.sqrt(conditional[:, s].sum(axis=1).mean())
                                for s in subsets])
        shared = np.array([means[:, s].sum(axis=1).std(ddof=1) for s in subsets])
        total = np.hypot(independent, shared)
        rows.append({
            "n_players": size,
            "n_subsets": len(subsets),
            "expected_total_games": float(np.mean([means[:, s].sum(axis=1).mean()
                                                   for s in subsets])),
            "independent_sd": float(independent.mean()),
            "shared_beta_sd": float(shared.mean()),
            "total_sd": float(total.mean()),
            "inflation": float((total / independent).mean()),
        })
    return pd.DataFrame(rows)


# ── The measurement, on whichever pair of frames it is handed ─────────────────

def fit_and_score(train: pd.DataFrame, frame: pd.DataFrame, max_games: int,
                  cfg_stan: dict, l2: float, seed: int) -> dict:
    """Fit the MLE and the Stan head on `train`, score both plus the plug-in on `frame`.

    Split-agnostic on purpose. `run` hands it `(train, validation)`; when the workflow is
    finished `src/final_evaluation.py` hands it `(train + validation, test)`. The held-out
    number is therefore produced by *this* code rather than by a second implementation of
    it that could drift — the same reason the port check imports `evaluate` and `crps` from
    the MLE module instead of reimplementing them.
    """
    print("\nFitting the point MLE (the reference this ports)...")
    mle = BetaBinomialGLM(l2).fit(train)
    print(f"  converged={mle.converged}, rho={mle.rho:.4f}")

    print(f"\nFitting in Stan ({cfg_stan.get('chains', 4)} chains x "
          f"{cfg_stan.get('samples', 1000)} draws)...")
    stan = StanAvailability(
        l2=l2, pmf_mode="posterior", chains=int(cfg_stan.get("chains", 4)),
        warmup=int(cfg_stan.get("warmup", 1000)),
        samples=int(cfg_stan.get("samples", 1000)), seed=seed,
        predictive_draws=int(cfg_stan.get("predictive_draws",
                                          PREDICTIVE_DRAWS))).fit(train)
    d = stan.diagnostics
    print(f"  max R-hat {d['max_rhat']:.4f}, min ESS "
          f"{min(d['min_ess_bulk'], d['min_ess_tail']):.0f}, "
          f"{d['divergences']} divergences, {d['wall_clock_s']:.1f}s wall clock "
          f"({d['cmdstan']})")
    print(f"  posterior mean rho {stan.rho:.4f} against the MLE's {mle.rho:.4f}")

    # The plug-in view shares the fit; only the predictive differs.
    plug_in = StanAvailability(l2=l2, pmf_mode="plug_in", predictive_draws=1)
    plug_in.__dict__.update({k: v for k, v in stan.__dict__.items()
                             if k not in ("pmf_mode", "name")})

    rows, pit_frames, prediction_frames = [], [], []
    for model in (mle, plug_in, stan):
        model_rows, predictions = evaluate(model, frame, max_games, seed)
        rows += model_rows
        prediction_frames.append(predictions)
        pit_frames.append(pit_table(predictions["pit"].to_numpy(), model.name))

    return {
        "mle": mle, "stan": stan, "plug_in": plug_in,
        "metrics": pd.DataFrame(rows),
        "coefficients": coefficient_comparison(mle, stan, stan.features),
        "board": board_correlation(stan, frame),
        "pit": pd.concat(pit_frames, ignore_index=True),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "diagnostics": diagnostics_frame([stan.diagnostics]),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_av = cfg.get("features", {}).get("availability", {})
    cfg_stan = cfg.get("stan", {})
    test_seasons = int(cfg_av.get("test_seasons", 2))
    seed = int(cfg_stan.get("seed", cfg_av.get("seed", 42)))
    l2 = float(cfg_av.get("glm_l2", 1.0))

    design = availability_design(cfg)
    train, val = selection_split(design, test_seasons)
    max_games = int(design["team_games"].max())

    print(f"Stan availability: {len(design):,} player-seasons. The test split is LOCKED — "
          f"this port\n  check fits and scores VALIDATION only "
          f"(src/models/held_out.py); the held-out reading is\n  taken once, by "
          f"`make final-evaluation`, through this module's own `fit_and_score`.")
    print(f"  {len(train):,} fit / {len(val):,} score "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")
    print(f"  {len(FEATURE_COLS)} features, n = max(team_games, gp) — the 13 traded "
          f"player-seasons with gp > team_games would otherwise make the summed\n"
          f"  log-likelihood non-finite at every rho, which under HMC poisons the "
          f"trajectory rather than just stopping an optimizer.")

    scored = fit_and_score(train, val, max_games, cfg_stan, l2, seed)
    mle, stan, plug_in = scored["mle"], scored["stan"], scored["plug_in"]

    metrics = scored["metrics"]
    table = (metrics[metrics.group == "all"]
             .pivot_table(index="model", columns="metric", values="value"))
    order = ["crps_games", "mae_games", "r2_gp_share", "pit_ks_distance",
             "dispersion_rho", "implied_overdispersion"]
    print("\nValidation scores (CRPS in games, lower is better):")
    print(table[order].sort_values("crps_games").round(4).to_string())

    coefs = scored["coefficients"]
    worst = coefs.loc[coefs["z_from_mle"].abs().idxmax()]
    print(f"\nPort check — MLE optimum against the posterior ({len(coefs)} terms):")
    print(f"  max |posterior mean - MLE| = "
          f"{coefs['difference'].abs().max():.5f}")
    print(f"  largest gap in posterior sds: {worst['term']} at "
          f"z = {worst['z_from_mle']:+.3f}")
    print(f"  MLE inside the 95% credible interval for "
          f"{int(coefs['mle_inside_95'].sum())}/{len(coefs)} terms")
    print("  The prior is normal(0, 1/sqrt(2*l2)), so the posterior MODE is exactly the\n"
          "  penalized MLE — agreement here is a defined check, not a coincidence.")

    crps_mle = float(table.loc[mle.name, "crps_games"])
    crps_plug = float(table.loc[plug_in.name, "crps_games"])
    crps_post = float(table.loc[stan.name, "crps_games"])
    print(f"\n  CRPS: MLE {crps_mle:.4f} | Stan plug-in {crps_plug:.4f} "
          f"({crps_plug - crps_mle:+.4f}) | Stan posterior {crps_post:.4f} "
          f"({crps_post - crps_mle:+.4f})")
    print("  The marginal metric was never the argument — at ~10,000 rows against 20\n"
          "  parameters the posterior is sharp, so this is expected to be a wash.")

    board = scored["board"]
    print("\nWhat the posterior actually buys — shared-beta correlation, by portfolio size:")
    print(board.round(3).to_string(index=False))
    small = board.iloc[0]
    whole = board.iloc[-1]
    print(f"  The shared-beta term is EXACTLY ZERO under any point estimate. It grows as N "
          f"while the\n  independent term grows as sqrt(N), so the ratio scales as sqrt(N) "
          f"and the SIZE OF THE\n  PORTFOLIO decides whether it matters: assuming "
          f"independent marginals understates the\n  spread by "
          f"{small['inflation'] - 1:.1%} on {int(small['n_players'])} players and "
          f"{whole['inflation'] - 1:.1%} across all "
          f"{int(whole['n_players'])}.")
    print("  So this is real for board-wide exposure and near-irrelevant for one roster — "
          "do not\n  quote the board figure as if it applied to a 15-man team.")
    print("  Note it is a statement about the JOINT, not the marginal: per player the\n"
          "  integrated predictive is not necessarily wider, because Jensen on the concave\n"
          "  conditional variance pushes back against the parameter spread.")

    print("\nPIT calibration (share per decile; 0.100 is uniform):")
    pit = scored["pit"]
    print(pit.pivot_table(index="model", columns="bin_low", values="share")
          .round(3).to_string())

    artifacts = {
        "metrics": (metrics, out_dir / "stan_availability_metrics.csv"),
        "coefficients": (coefs, out_dir / "stan_availability_coefficients.csv"),
        "diagnostics": (scored["diagnostics"],
                        out_dir / "stan_availability_diagnostics.csv"),
        "board": (board, out_dir / "stan_availability_board.csv"),
        "pit": (pit, out_dir / "stan_availability_pit.csv"),
        "predictions": (scored["predictions"],
                        out_dir / "stan_availability_predictions.csv"),
    }
    paths = {}
    for name, (frame, dest) in artifacts.items():
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
