r"""The eleven component rate heads in Stan — eight counts, three conversions.

The third link in the chain: `counts | min` and `makes | attempts`. Together with the
availability and minutes heads this is the full output contract, and `dk_pts` is
reassembled deterministically from a joint draw rather than predicted directly.

| head | likelihood | exposure / trials |
|---|---|---|
| `fg2a` `fg3a` `fta` `reb` `ast` `stl` `blk` `tov` | negative binomial, log link | season minutes |
| `fg2m\|fg2a` `fg3m\|fg3a` `ftm\|fta` | beta-binomial, logit link | attempts |

## Eleven fits, not one joint model — and this is an identity, not a shortcut

The parameter blocks are disjoint and the priors independent, so the joint posterior
factorizes **exactly**: eleven separate fits recover the identical posterior a single
model would. `megamodel.stan` in the prior attempt is the cautionary case — every head had
its own `beta_*` with an independent prior and the player-effect terms were commented out,
so it was thirteen independent GLMs in one file paying the full joint-fit price. It ran on
`sample_frac(0.01)`. Ninety-nine percent of the data given up for a coupling that was not
in the model.

The correlation the simulator needs enters at **draw** time: one `min` draw pushed through
all eleven heads as exposure (minutes is 18.6% of within-player residual variance, by far
the largest common factor), then a Gaussian copula for the remainder if needed —
conditioning minutes out, the residual off-diagonals average **+0.013** with a max of
0.157. The one structural exception is the 3PA/2PA substitution at **-0.110**, and
`substitution_arm` below handles it by reparameterization rather than by coupling two
count heads.

## Season-collapsed, which is also an identity

For `y_g ~ Poisson(m_g e^eta)` with `eta` constant within a player-season,
`prod_g p(y_g) = Poisson(Y; M e^eta) x Multinomial(y; Y, m_g/M)` and the multinomial factor
is free of `eta`. The conversion heads collapse the same way with a hypergeometric factor.
So 10,194 player-season rows recover the coefficients of 731,906 player-game rows, and the
collapse is what makes eleven full Bayesian fits affordable at all.

## Two things every head here is quoted against

**The no-fit floor.** `carry_forward` — prior per-36 rate x actual minutes / 36, no
fitting whatsoever — scores held-out R^2 of **0.82-0.94**. A head that does not clear it is
not a model. `beats_floor` is on every output row. The conversion floor has to be a
*shrunk* carry-forward: a player who went 0-for-3 from three has a prior 3P% of exactly
0.000, and carrying that onto 200 attempts gives a beta-binomial NLL of 1.3e9.

**A validation split.** Variants are selected on a split carved out of train; the test
column is confirmation only. The measured expectation (`make component-rates`) is that
**scale beats curvature**: `log E[rate] = beta*log(prior rate)` makes the model
`rate ~ prior_rate^beta`, which is the right shape, while linear-in-raw-rate inside `exp()`
is badly misspecified — R^2 0.520 on `fg3a` and 0.638 on `blk`. Splines then add +0.030 and
+0.041 on those two heads only and <= +0.003 on the other six. The spline basis here is
built on `log(own)` rather than raw `own`, so it strictly nests `log_own` and the
comparison isolates curvature *given* the right scale.

Usage:
    python -m src.models.stan_components
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.availability import EPS, RHO_MAX, RHO_MIN, fit_dispersion
from src.models.availability import _neg_loglik as beta_binomial_nll
from src.models.component_rates import (BIO_COLS, CONTEXT_COLS, CONVERSION_HEADS,
                                        COUNT_HEADS, add_log, add_spline,
                                        build_design, carry_forward,
                                        carry_forward_conversion, fit_nb_dispersion,
                                        impute, nb_nll, split_seasons)
from src.models.stan_utils import (compile_model, crps_from_samples,
                                   diagnostics_frame, ks_uniform,
                                   pit_from_samples, posterior,
                                   prior_sd_for_l2, sample, standardized, thin,
                                   warn_if_unconverged)

COUNT_MODEL = "negbinomial_glm"
CONVERSION_MODEL = "betabinomial_glm"

SPLINE_KNOTS = 5
PREDICTIVE_SAMPLES = 1000

# Weakly informative on standardized features. A normal(0, 1) prior is an L2 penalty of
# 0.5, which against a log-likelihood summed over ~10^4 rows is negligible shrinkage — but
# it is not nothing, and that is the point: `CLAUDE.md` records that the raw feature matrix
# is *singular*, so an unpenalized GLM is not merely ill-conditioned, it fails outright.
# This block is small enough not to be singular, and the prior is insurance against the
# collinearity that is there (`total_minutes = mpg x gp` by construction).
BETA_SCALE = 1.0
INTERCEPT_SCALE = 5.0
# phi_inv = 0 is exactly Poisson, so the prior sits on the departure from it. Season
# totals land near phi ~ 10^2, i.e. phi_inv ~ 10^-2, where exponential(1) is essentially
# flat — weak exactly where the data is.
PHI_INV_SCALE = 1.0
CONVERSION_L2 = 1.0


# ── Feature variants ──────────────────────────────────────────────────────────

def count_variants(train: pd.DataFrame, test: pd.DataFrame, component: str,
                   n_knots: int = SPLINE_KNOTS
                   ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """Raw scale, log scale, and curvature on top of the log scale.

    Walk-forward PCA of the 156-column season matrix is deliberately absent: it lands
    within +/-0.003 of this cheap raw spec on every head, because the style and tracking
    families add nothing once you have the player's own prior rate and his minutes.
    """
    own = f"{component}_p36_lag1"
    base = [own] + CONTEXT_COLS + BIO_COLS
    tr, te, flags = impute(train, test, base)
    base = base + flags

    out = {"linear": (tr, te, list(base))}
    tl, el, log_names = add_log(tr, te, [own])
    log_base = [c for c in base if c != own] + log_names
    out["log_own"] = (tl, el, list(log_base))

    # Splined on log(own), not raw own. With linear extrapolation the basis contains the
    # linear function of log(own), so `log_own_spline` strictly NESTS `log_own` and the
    # contrast measures curvature given the right scale rather than curvature-vs-scale.
    ts, es, spline_names = add_spline(tl, el, log_names, n_knots)
    out["log_own_spline"] = (ts, es,
                             [c for c in log_base if c not in log_names] + spline_names)
    return out


def _logit_column(train: pd.DataFrame, test: pd.DataFrame, col: str
                  ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    name = f"logit_{col}"
    tr, te = train.copy(), test.copy()
    for frame, src in ((tr, train), (te, test)):
        p = np.clip(src[col].to_numpy(dtype=float), 1e-3, 1 - 1e-3)
        frame[name] = np.log(p / (1 - p))
    return tr, te, name


def conversion_variants(train: pd.DataFrame, test: pd.DataFrame, made: str,
                        attempted: str, n_knots: int = SPLINE_KNOTS
                        ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """The same scale-then-curvature ladder, on the logit that a logit link wants."""
    own = f"{made}_pct_lag1"
    vol = f"{attempted}_p36_lag1"
    base = [own, vol] + CONTEXT_COLS + BIO_COLS
    tr, te, flags = impute(train, test, base)
    base = base + flags

    out = {"linear": (tr, te, list(base))}
    tl, el, name = _logit_column(tr, te, own)
    logit_base = [c for c in base if c != own] + [name]
    out["logit_own"] = (tl, el, list(logit_base))
    ts, es, spline_names = add_spline(tl, el, [name], n_knots)
    out["logit_own_spline"] = (ts, es,
                               [c for c in logit_base if c != name] + spline_names)
    return out


# ── Heads ─────────────────────────────────────────────────────────────────────

class StanCount:
    """Negative-binomial season total with `log(minutes)` as an offset."""

    def __init__(self, features: list[str], component: str, name: str = "count",
                 chains: int = 4, warmup: int = 1000, samples: int = 1000,
                 seed: int = 42, predictive_samples: int = PREDICTIVE_SAMPLES):
        self.features, self.component, self.name = features, component, name
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed
        self.predictive_samples = predictive_samples

    def fit(self, train: pd.DataFrame) -> "StanCount":
        (X,), self.scaler = standardized(train, [train], self.features)
        y = train[self.component].to_numpy(float)
        exposure = train["total_minutes"].to_numpy(float)
        if (exposure <= 0).any():
            raise ValueError("non-positive exposure: log(minutes) is the offset")

        model = compile_model(COUNT_MODEL)
        fit, self.diagnostics = sample(
            model,
            {"N": len(train), "K": X.shape[1], "X": X,
             "y": np.rint(y).astype(int).tolist(), "exposure": exposure.tolist(),
             "beta_scale": BETA_SCALE, "intercept_scale": INTERCEPT_SCALE,
             "phi_inv_scale": PHI_INV_SCALE},
            chains=self.chains, warmup=self.warmup, samples=self.samples,
            seed=self.seed, label=self.name,
            inits={"alpha": float(np.log(max(y.sum(), 1) / exposure.sum())),
                   "beta": np.zeros(X.shape[1]).tolist(), "phi_inv": 0.05})
        warn_if_unconverged(self.diagnostics)

        draws = posterior(fit, ["alpha", "beta", "phi"])
        self.alpha_draws = draws["alpha"].reshape(-1)
        self.beta_draws = draws["beta"].reshape(len(self.alpha_draws), -1)
        self.phi_draws = draws["phi"].reshape(-1)
        self.phi = float(self.phi_draws.mean())
        return self

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    def mu_draws(self, df: pd.DataFrame, keep: int) -> tuple[np.ndarray, np.ndarray]:
        idx = thin(len(self.alpha_draws), keep)
        eta = self._design(df) @ self.beta_draws[idx].T + self.alpha_draws[idx][None, :]
        mu = np.exp(np.clip(eta, -30, 30)).T * df["total_minutes"].to_numpy(float)[None, :]
        return np.clip(mu, 1e-9, None), self.phi_draws[idx]

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        return self.mu_draws(df, self.predictive_samples)[0].mean(axis=0)

    def predict_samples(self, df: pd.DataFrame, seed: int = 0) -> np.ndarray:
        mu, phi = self.mu_draws(df, self.predictive_samples)
        rng = np.random.default_rng(seed)
        p = phi[:, None] / (phi[:, None] + mu)
        return rng.negative_binomial(phi[:, None] * np.ones_like(mu), p).astype(float)


class StanConversion:
    """Beta-binomial makes-out-of-attempts, sharing `betabinomial_glm.stan`."""

    def __init__(self, features: list[str], made: str, attempted: str,
                 name: str = "conversion", l2: float = CONVERSION_L2, chains: int = 4,
                 warmup: int = 1000, samples: int = 1000, seed: int = 42,
                 predictive_samples: int = PREDICTIVE_SAMPLES):
        self.features, self.made, self.attempted = features, made, attempted
        self.name, self.l2 = name, l2
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed
        self.predictive_samples = predictive_samples

    def fit(self, train: pd.DataFrame) -> "StanConversion":
        live = train[train[self.attempted] > 0]
        (X,), self.scaler = standardized(live, [live], self.features)
        y = np.rint(live[self.made].to_numpy(float)).astype(int)
        n = np.rint(live[self.attempted].to_numpy(float)).astype(int)
        y = np.minimum(y, n)

        share = float(np.clip(y.sum() / max(n.sum(), 1), EPS, 1 - EPS))
        model = compile_model(CONVERSION_MODEL)
        fit, self.diagnostics = sample(
            model,
            {"N": len(live), "K": X.shape[1], "X": X, "n": n.tolist(), "y": y.tolist(),
             "beta_scale": prior_sd_for_l2(self.l2),
             "intercept_scale": INTERCEPT_SCALE},
            chains=self.chains, warmup=self.warmup, samples=self.samples,
            seed=self.seed, label=self.name,
            inits={"alpha": float(np.log(share / (1 - share))),
                   "beta": np.zeros(X.shape[1]).tolist(), "rho": 0.01})
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

    def p_draws(self, df: pd.DataFrame, keep: int) -> tuple[np.ndarray, np.ndarray]:
        idx = thin(len(self.alpha_draws), keep)
        # (rows x K) @ (K x draws) -> (rows x draws), transposed to draws-major.
        eta = (self._design(df) @ self.beta_draws[idx].T
               + self.alpha_draws[idx][None, :]).T
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30))), self.rho_draws[idx]

    def predict_p(self, df: pd.DataFrame) -> np.ndarray:
        return self.p_draws(df, self.predictive_samples)[0].mean(axis=0)

    def predict_samples(self, df: pd.DataFrame, seed: int = 0) -> np.ndarray:
        p, rho = self.p_draws(df, self.predictive_samples)
        a, b = _beta_shapes(p, rho[:, None])
        rng = np.random.default_rng(seed)
        n = np.rint(df[self.attempted].to_numpy(float)).astype(int)
        return rng.binomial(n[None, :], rng.beta(a, b)).astype(float)


def _beta_shapes(mu: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(mu, EPS, 1 - EPS)
    rho = np.clip(rho, RHO_MIN, RHO_MAX)
    scale = (1.0 - rho) / rho
    return mu * scale, (1.0 - mu) * scale


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_count(y: np.ndarray, mu: np.ndarray, samples: np.ndarray, phi: float,
                seed: int = 0) -> dict:
    return {
        "r2": float(1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "mae": float(np.abs(y - mu).mean()),
        "nll": float(nb_nll(y, np.clip(mu, 1e-6, None), phi).mean()),
        "crps": float(crps_from_samples(samples, y).mean()),
        "pit_ks": ks_uniform(pit_from_samples(samples, y, seed)),
        "dispersion": float(phi),
    }


def score_conversion(y: np.ndarray, n: np.ndarray, p: np.ndarray, samples: np.ndarray,
                     rho: float, seed: int = 0) -> dict:
    realised = y / n
    return {
        "r2": float(1 - ((realised - p) ** 2).sum()
                    / ((realised - realised.mean()) ** 2).sum()),
        "mae": float(np.abs(realised - p).mean()),
        "nll": float(beta_binomial_nll(y.astype(int), n.astype(int), p, rho) / len(y)),
        "crps": float(crps_from_samples(samples, y).mean()),
        "pit_ks": ks_uniform(pit_from_samples(samples, y, seed)),
        "dispersion": float(rho),
    }


def count_floor(train: pd.DataFrame, test: pd.DataFrame, component: str,
                seed: int = 0) -> dict:
    """`carry_forward`, wrapped in an NB so its CRPS is comparable to a fitted head."""
    mu = np.clip(carry_forward(test, component), 1e-6, None)
    y = test[component].to_numpy(float)
    phi = fit_nb_dispersion(train[component].to_numpy(float),
                            np.clip(carry_forward(train, component), 1e-6, None))
    rng = np.random.default_rng(seed)
    p = phi / (phi + mu)
    samples = rng.negative_binomial(np.full((PREDICTIVE_SAMPLES, len(mu)), phi),
                                    np.repeat(p[None, :], PREDICTIVE_SAMPLES, axis=0)
                                    ).astype(float)
    return score_count(y, mu, samples, phi, seed)


def conversion_floor(train: pd.DataFrame, test: pd.DataFrame, made: str,
                     attempted: str, seed: int = 0) -> dict:
    """The shrunk carry-forward — a raw one is unusable, and that is a fact about
    proportions rather than a coding choice (see `carry_forward_conversion`)."""
    ok = test[attempted].to_numpy(float) > 0
    live = train[attempted].to_numpy(float) > 0
    p = carry_forward_conversion(train, test, made, attempted)[ok]
    rho = fit_dispersion(np.rint(train[made].to_numpy(float)[live]).astype(int),
                         np.rint(train[attempted].to_numpy(float)[live]).astype(int),
                         carry_forward_conversion(train, train, made, attempted)[live])
    y = np.rint(test[made].to_numpy(float)[ok]).astype(int)
    n = np.rint(test[attempted].to_numpy(float)[ok]).astype(int)
    y = np.minimum(y, n)
    a, b = _beta_shapes(np.repeat(p[None, :], PREDICTIVE_SAMPLES, axis=0),
                        np.full((PREDICTIVE_SAMPLES, 1), rho))
    rng = np.random.default_rng(seed)
    samples = rng.binomial(n[None, :], rng.beta(a, b)).astype(float)
    return score_conversion(y.astype(float), n.astype(float), p, samples, rho, seed)


# ── Sweeps ────────────────────────────────────────────────────────────────────

def _iters(cfg_stan: dict, fast: bool) -> dict:
    if fast:
        return {"warmup": int(cfg_stan.get("select_warmup", 500)),
                "samples": int(cfg_stan.get("select_samples", 500))}
    return {"warmup": int(cfg_stan.get("warmup", 1000)),
            "samples": int(cfg_stan.get("samples", 1000))}


def sweep_counts(train, val, test, full_train, cfg_stan, n_knots
                 ) -> tuple[pd.DataFrame, list[dict]]:
    """Every count head x variant on both splits. Selection reads validation only."""
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics = [], []

    for component in COUNT_HEADS:
        floor_val = count_floor(train, val, component, seed)
        floor_test = count_floor(full_train, test, component, seed)
        rows.append({"head": component, "kind": "count", "variant": "carry_forward",
                     "n_features": 0, "val_r2": floor_val["r2"],
                     "val_crps": floor_val["crps"],
                     **{f"test_{k}": v for k, v in floor_test.items()}})

        val_variants = count_variants(train, val, component, n_knots)
        test_variants = count_variants(full_train, test, component, n_knots)
        for label in val_variants:
            v_tr, v_te, v_features = val_variants[label]
            v_model = StanCount(v_features, component, name=f"{component}/{label}/val",
                                chains=chains, seed=seed,
                                **_iters(cfg_stan, True)).fit(v_tr)
            diagnostics.append(v_model.diagnostics)
            v_y = v_te[component].to_numpy(float)
            v = score_count(v_y, v_model.predict_mean(v_te),
                            v_model.predict_samples(v_te, seed), v_model.phi, seed)

            t_tr, t_te, t_features = test_variants[label]
            t_model = StanCount(t_features, component, name=f"{component}/{label}/test",
                                chains=chains, seed=seed,
                                **_iters(cfg_stan, False)).fit(t_tr)
            diagnostics.append(t_model.diagnostics)
            t_y = t_te[component].to_numpy(float)
            t = score_count(t_y, t_model.predict_mean(t_te),
                            t_model.predict_samples(t_te, seed), t_model.phi, seed)

            rows.append({"head": component, "kind": "count", "variant": label,
                         "n_features": len(t_features), "val_r2": v["r2"],
                         "val_crps": v["crps"],
                         **{f"test_{k}": val for k, val in t.items()}})
    return _finalize(pd.DataFrame(rows), "val_r2", "test_r2", higher_is_better=True), diagnostics


def sweep_conversions(train, val, test, full_train, cfg_stan, n_knots
                      ) -> tuple[pd.DataFrame, list[dict]]:
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics = [], []

    for made, attempted in CONVERSION_HEADS:
        head = f"{made}|{attempted}"
        floor_val = conversion_floor(train, val, made, attempted, seed)
        floor_test = conversion_floor(full_train, test, made, attempted, seed)
        rows.append({"head": head, "kind": "conversion", "variant": "carry_forward",
                     "n_features": 0, "val_nll": floor_val["nll"],
                     "val_crps": floor_val["crps"],
                     **{f"test_{k}": v for k, v in floor_test.items()}})

        val_variants = conversion_variants(train, val, made, attempted, n_knots)
        test_variants = conversion_variants(full_train, test, made, attempted, n_knots)
        for label in val_variants:
            v_tr, v_te, v_features = val_variants[label]
            v_model = StanConversion(v_features, made, attempted,
                                     name=f"{head}/{label}/val", chains=chains,
                                     seed=seed, **_iters(cfg_stan, True)).fit(v_tr)
            diagnostics.append(v_model.diagnostics)
            v = _score_conv(v_model, v_te, made, attempted, seed)

            t_tr, t_te, t_features = test_variants[label]
            t_model = StanConversion(t_features, made, attempted,
                                     name=f"{head}/{label}/test", chains=chains,
                                     seed=seed, **_iters(cfg_stan, False)).fit(t_tr)
            diagnostics.append(t_model.diagnostics)
            t = _score_conv(t_model, t_te, made, attempted, seed)

            rows.append({"head": head, "kind": "conversion", "variant": label,
                         "n_features": len(t_features), "val_nll": v["nll"],
                         "val_crps": v["crps"],
                         **{f"test_{k}": val for k, val in t.items()}})
    return _finalize(pd.DataFrame(rows), "val_nll", "test_nll",
                     higher_is_better=False), diagnostics


def _score_conv(model: StanConversion, frame: pd.DataFrame, made: str, attempted: str,
                seed: int) -> dict:
    ok = frame[attempted].to_numpy(float) > 0
    live = frame[ok]
    n = np.rint(live[attempted].to_numpy(float)).astype(int)
    y = np.minimum(np.rint(live[made].to_numpy(float)).astype(int), n)
    return score_conversion(y.astype(float), n.astype(float), model.predict_p(live),
                            model.predict_samples(live, seed), model.rho, seed)


def _finalize(table: pd.DataFrame, val_col: str, test_col: str,
              higher_is_better: bool) -> pd.DataFrame:
    """Mark the validation-selected variant per head, and whether it clears the floor.

    Two separate flags on purpose. `selected` answers "which variant would I ship", and it
    is decided on validation. `beats_floor` answers "is this a model at all", and it is a
    held-out fact about the shipped variant. A head can be selected and still fail the
    floor — `ftm|fta` is the known case, and reporting it as a win would be the error this
    column exists to prevent.
    """
    out = table.copy()
    out["selected"] = False
    out["beats_floor"] = False
    for head, block in out.groupby("head"):
        fitted = block[block["variant"] != "carry_forward"]
        if fitted.empty:
            continue
        best = (fitted[val_col].idxmax() if higher_is_better else fitted[val_col].idxmin())
        out.loc[best, "selected"] = True
        floor = block[block["variant"] == "carry_forward"][test_col].iloc[0]
        better = (block[test_col] > floor) if higher_is_better else (block[test_col] < floor)
        out.loc[block.index, "beats_floor"] = better
        out.loc[block[block["variant"] == "carry_forward"].index, "beats_floor"] = True
    return out


# ── The 3PA / 2PA substitution ────────────────────────────────────────────────

def add_substitution_columns(design: pd.DataFrame) -> pd.DataFrame:
    """`fga` and the prior three-point share, for the reparameterized arm.

    Per-36 rates share a denominator, so `fga_p36 = fg2a_p36 + fg3a_p36` exactly and the
    share is a ratio of the two. Nothing is re-derived from the game logs.
    """
    out = design.copy()
    out["fga"] = out["fg2a"].to_numpy() + out["fg3a"].to_numpy()
    prior = (out["fg2a_p36_lag1"].to_numpy(float) + out["fg3a_p36_lag1"].to_numpy(float))
    out["fga_p36_lag1"] = prior
    with np.errstate(invalid="ignore", divide="ignore"):
        out["fg3a_share_lag1"] = np.where(prior > 0,
                                          out["fg3a_p36_lag1"].to_numpy(float) / prior,
                                          np.nan)
    return out


def substitution_arm(train, val, test, full_train, cfg_stan, n_knots
                     ) -> tuple[pd.DataFrame, list[dict]]:
    """Two Poissons vs `fga` count x `fg3a | fga` share — compared as joint densities.

    3PA substitutes for 2PA: it is the one structural cross-component correlation the
    residual copula would otherwise have to carry (measured at **-0.110**, against an
    average off-diagonal of +0.013). Coupling two count heads would break the exact
    factorization the whole architecture rests on. Reparameterizing does not: model the
    total attempts as a count and the three-point *share* as a binomial, which enforces
    the substitution by construction and keeps the chain intact. It is also better
    specified — shot-mix shares persist like counts (`sco_pct_fga_3pt` 0.886) while
    conversion percentages do not.

    **The comparison is legitimate because the map is a bijection.** `(fg2a, fg3a)` and
    `(fga = fg2a + fg3a, fg3a)` are the same point in different coordinates, with unit
    Jacobian on the integers, so the two joint log-densities are directly comparable. Mean
    joint NLL over the test rows is therefore an apples-to-apples number and not a
    scale-mismatched one.
    """
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics = [], []

    for split, (tr, te) in {"val": (train, val), "test": (full_train, test)}.items():
        fast = split == "val"
        tr, te = add_substitution_columns(tr), add_substitution_columns(te)

        # Arm A — the canonical pair, two independent count heads.
        nll_a = np.zeros(len(te))
        for component in ("fg2a", "fg3a"):
            variants = count_variants(tr, te, component, n_knots)
            v_tr, v_te, features = variants["log_own"]
            model = StanCount(features, component, name=f"subst/{component}/{split}",
                              chains=chains, seed=seed,
                              **_iters(cfg_stan, fast)).fit(v_tr)
            diagnostics.append(model.diagnostics)
            nll_a += nb_nll(v_te[component].to_numpy(float),
                            np.clip(model.predict_mean(v_te), 1e-6, None), model.phi)

        # Arm B — total attempts as a count, three-point share as a binomial.
        variants = count_variants(tr, te, "fga", n_knots)
        v_tr, v_te, features = variants["log_own"]
        fga_model = StanCount(features, "fga", name=f"subst/fga/{split}", chains=chains,
                              seed=seed, **_iters(cfg_stan, fast)).fit(v_tr)
        diagnostics.append(fga_model.diagnostics)
        nll_b = nb_nll(v_te["fga"].to_numpy(float),
                       np.clip(fga_model.predict_mean(v_te), 1e-6, None), fga_model.phi)

        share_base = ["fg3a_share_lag1", "fga_p36_lag1"] + CONTEXT_COLS + BIO_COLS
        s_tr, s_te, flags = impute(tr, te, share_base)
        s_tr, s_te, name = _logit_column(s_tr, s_te, "fg3a_share_lag1")
        share_features = [c for c in share_base + flags
                          if c != "fg3a_share_lag1"] + [name]
        share_model = StanConversion(share_features, "fg3a", "fga",
                                     name=f"subst/fg3a|fga/{split}", chains=chains,
                                     seed=seed, **_iters(cfg_stan, fast)).fit(s_tr)
        diagnostics.append(share_model.diagnostics)
        n = np.rint(s_te["fga"].to_numpy(float)).astype(int)
        y = np.minimum(np.rint(s_te["fg3a"].to_numpy(float)).astype(int), n)
        live = n > 0
        share_nll = np.zeros(len(s_te))
        p = share_model.predict_p(s_te[live])
        share_nll[live] = -_beta_binomial_logpmf(y[live], n[live], p, share_model.rho)
        nll_b = nll_b + share_nll

        rows.append({"split": split, "arm": "two_counts",
                     "mean_joint_nll": float(nll_a.mean()), "n": len(te)})
        rows.append({"split": split, "arm": "fga_x_fg3a_share",
                     "mean_joint_nll": float(nll_b.mean()), "n": len(te)})

    out = pd.DataFrame(rows)
    pivot = out.pivot_table(index="split", columns="arm", values="mean_joint_nll")
    out = out.merge(
        (pivot["fga_x_fg3a_share"] - pivot["two_counts"]).rename("reparam_minus_canonical"),
        left_on="split", right_index=True)
    return out, diagnostics


def _beta_binomial_logpmf(y: np.ndarray, n: np.ndarray, p: np.ndarray,
                          rho: float) -> np.ndarray:
    from scipy.stats import betabinom
    a, b = _beta_shapes(np.asarray(p, dtype=float), np.asarray(float(rho)))
    return np.nan_to_num(betabinom.logpmf(y, n, a, b), nan=-1e6, neginf=-1e6)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_stan = cfg.get("stan", {})
    n_knots = int(cfg_stan.get("components", {}).get("spline_knots", SPLINE_KNOTS))
    test_seasons = 2

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    full_train, test = split_seasons(design, test_seasons)
    order = sorted(full_train["season"].unique())
    train, val = split_seasons(full_train, min(test_seasons, len(order) - 1))

    print(f"Stan component heads: {len(design):,} player-seasons, "
          f"{design['season'].nunique()} target seasons")
    print(f"  {len(full_train):,} train / {len(test):,} test "
          f"({', '.join(sorted(test['season'].unique()))} held out)")
    print(f"  validation split: {len(train):,} fit / {len(val):,} select "
          f"({', '.join(sorted(val['season'].unique()))})")
    print(f"  {len(COUNT_HEADS)} count heads + {len(CONVERSION_HEADS)} conversion heads, "
          f"fitted SEPARATELY — the chain factorizes the joint posterior exactly.\n")

    counts, diag_counts = sweep_counts(train, val, test, full_train, cfg_stan, n_knots)
    print("Count heads — held-out R^2 on the season total (higher is better):")
    print(counts.pivot_table(index="head", columns="variant", values="test_r2")
          .reindex(columns=["carry_forward", "linear", "log_own", "log_own_spline"])
          .round(4).to_string())

    conversions, diag_conv = sweep_conversions(train, val, test, full_train, cfg_stan,
                                               n_knots)
    print("\nConversion heads — held-out beta-binomial NLL per row (lower is better):")
    print(conversions.pivot_table(index="head", columns="variant", values="test_nll")
          .reindex(columns=["carry_forward", "linear", "logit_own", "logit_own_spline"])
          .round(4).to_string())

    table = pd.concat([counts, conversions], ignore_index=True)
    chosen = table[table["selected"]]
    print("\nValidation-selected variant per head, against its no-fit floor:")
    for _, row in chosen.iterrows():
        floor = table[(table["head"] == row["head"])
                      & (table["variant"] == "carry_forward")].iloc[0]
        metric, better = ("test_r2", row["test_r2"] - floor["test_r2"]) \
            if row["kind"] == "count" else ("test_nll", floor["test_nll"] - row["test_nll"])
        mark = "" if row["beats_floor"] else "   <-- DOES NOT CLEAR THE FLOOR"
        print(f"  {row['head']:<12} {row['variant']:<18} {metric} "
              f"{row[metric]:8.4f} vs floor {floor[metric]:8.4f} "
              f"({better:+.4f}){mark}")

    failed = sorted(set(table["head"]) - set(table.loc[
        (table["variant"] != "carry_forward") & table["beats_floor"], "head"]))
    if failed:
        print(f"\n/!\\  NO variant clears the no-fit floor for: {', '.join(failed)}")
        print("   `ftm|fta` is the known and expected case — free-throw percentage is pure "
              "player skill\n   with no context to add, so an empirical-Bayes shrink of the "
              "prior is already optimal.\n   Any OTHER head here means check the "
              "specification before reporting a null.")
    else:
        print("\nEvery head has a variant clearing the no-fit floor.")

    substitution, diag_subst = substitution_arm(train, val, test, full_train, cfg_stan,
                                                n_knots)
    print("\n3PA/2PA substitution — two count heads vs `fga` count x `fg3a | fga` share:")
    print(substitution.round(4).to_string(index=False))
    print("  Comparable because the map is a bijection with unit Jacobian on the integers:\n"
          "  (fg2a, fg3a) and (fga, fg3a) are the same point in different coordinates.\n"
          "  Negative `reparam_minus_canonical` favours the reparameterization; the\n"
          "  validation row is the one that decides.")

    diag = diagnostics_frame(diag_counts + diag_conv + diag_subst)
    artifacts = {
        "metrics": (table, out_dir / "stan_component_metrics.csv"),
        "substitution": (substitution, out_dir / "stan_component_substitution.csv"),
        "diagnostics": (diag, out_dir / "stan_component_diagnostics.csv"),
    }
    paths = {}
    for name, (frame, dest) in artifacts.items():
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")

    bad = diag[~diag["converged"]]
    print(f"\nSampler over {len(diag)} fits: max R-hat {diag['max_rhat'].max():.4f}, "
          f"{int(diag['divergences'].sum())} divergences, "
          f"{diag['wall_clock_s'].sum() / 60:.1f} min total")
    if len(bad):
        print(f"/!\\  {len(bad)} fits failed a convergence bar: "
              f"{', '.join(bad['label'].head(10))}")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
