"""Shared plumbing for the Stan heads: compile, sample, and report the diagnostics.

Every head in `stan_availability.py`, `stan_minutes.py` and `stan_components.py` goes
through here, so that "did it converge?" is answered the same way each time and the
answer lands in a CSV rather than in a scrollback buffer.

## What this module standardizes, and why each piece is load-bearing

**Compiled binaries never touch the repo.** `.stan` sources live in `src/stan/` and are
checked in; the executables cmdstanpy builds from them go to `outputs/stan/`, which
`.gitignore` already covers. Staleness is handled here rather than by cmdstanpy's default,
which reuses an `exe_file` that exists without checking whether the source moved
underneath it — a silently wrong fit is the worst failure mode available to this project.

**The prior scale is derived from the L2 penalty it replaces**, not chosen. A penalized
MLE minimizing `-loglik + l2*||beta||^2` has exactly the posterior mode of a model with
`beta ~ normal(0, 1/sqrt(2*l2))`. `prior_sd_for_l2` is that identity, and it is what makes
"the Stan posterior mean reproduces the point MLE" a check with a defined answer instead of
a hopeful comparison.

**Diagnostics are extracted, not eyeballed.** R-hat, bulk/tail ESS, divergences,
treedepth saturation and wall clock come back as a dict per fit and are written out. A
divergence count above zero is reported loudly: on these heads it means the geometry went
wrong, and the containing metric is then not trustworthy no matter how good it looks.
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

STAN_DIR = Path(__file__).resolve().parents[1] / "stan"
BUILD_DIR = Path("outputs/stan")

# Sampler defaults. 1000/1000 x 4 chains is the CmdStan default and is ample for these
# fits — the largest is ~10^4 rows against ~20 parameters, where the posterior is close
# to Gaussian and mixing is easy.
CHAINS = 4
WARMUP = 1000
SAMPLES = 1000
ADAPT_DELTA = 0.9
SEED = 42

# Convergence bars. R-hat 1.01 is the modern (Vehtari et al. 2021) threshold rather than
# the older 1.1, and 400 is the ESS floor that makes a posterior mean's Monte Carlo error
# small against its own posterior sd.
RHAT_MAX = 1.01
ESS_MIN = 400.0
MAX_TREEDEPTH = 10        # NUTS's default; nothing here overrides it

# Half-normal scale on the year random effect's sd, on the LINEAR PREDICTOR scale. The
# league's year-to-year movement is 1-7% depending on the quantity (`make season-effects`,
# `yoy_sd_pct`), which on a log or logit scale is ~0.01-0.07 — so 0.25 is weakly
# informative by roughly an order of magnitude, and the data decides. Deliberately not
# tighter: a prior that pins sigma_year near the measured league movement would make "the
# head recovers the league's spread" a foregone conclusion rather than a check.
YEAR_SD_SCALE = 0.25


def prior_sd_for_l2(l2: float) -> float:
    """Prior sd whose posterior mode equals a penalized MLE with L2 weight `l2`.

    `-loglik + l2*||beta||^2` and `-loglik + ||beta||^2/(2*sigma^2)` are the same
    objective when `sigma = 1/sqrt(2*l2)`. Used so the ported head is a port and not a
    different model that happens to resemble one.
    """
    if l2 <= 0:
        raise ValueError(f"l2 must be positive to correspond to a proper prior; got {l2}")
    return float(1.0 / np.sqrt(2.0 * l2))


def year_block(n_rows: int, seasons: "pd.Series | np.ndarray | None" = None,
               levels: list | None = None,
               scale: float = YEAR_SD_SCALE) -> tuple[dict, list]:
    """The year-random-effect data block for `betabinomial_glm` / `negbinomial_glm`.

    Both `.stan` files declare `S`, `season_idx` and `year_sd_scale` unconditionally
    because Stan has no optional data. Passing `seasons=None` returns the **disabled**
    block — `S = 0`, every index 0 — which makes the parameter vectors zero-length and the
    model bit-for-bit the one that existed before the year effect was added. Every head
    that does not want a year term gets it from here rather than hand-writing three keys,
    so "disabled" has exactly one definition.

    Returns the data keys and the season **levels** in index order. The levels matter:
    they are the training seasons the fitted `year_z` corresponds to, and a held-out
    season must NOT map into them — at prediction time the effect is a fresh draw, not a
    lookup. Any season absent from `levels` is mapped to 0, which the model never reads.
    """
    if seasons is None:
        return {"S": 0, "season_idx": [0] * int(n_rows), "year_sd_scale": float(scale)}, []
    values = np.asarray(seasons)
    order = list(levels) if levels is not None else sorted(set(values.tolist()))
    lookup = {s: i + 1 for i, s in enumerate(order)}
    idx = [int(lookup.get(v, 0)) for v in values.tolist()]
    return ({"S": len(order), "season_idx": idx, "year_sd_scale": float(scale)}, order)


def rho_block(n_rows: int, bins: "np.ndarray | None" = None,
              n_bins: int | None = None) -> dict:
    """The dispersion-bin data block for `betabinomial_glm`.

    That file declares `n_rho` and `rho_bin` unconditionally because Stan has no optional
    data. Passing `bins=None` returns the **disabled** block — `n_rho = 1`, every row in
    bin 1 — which is the shared-dispersion model exactly, in the same way `year_block`'s
    `S = 0` is the no-year-effect model exactly. Every head that wants one scalar
    dispersion gets it from here rather than hand-writing two keys, so "shared" has one
    definition and the nesting is pinned in one place.

    `bins` is **1-based** and must cover `1..n_bins` on the fitting rows; a bin with no
    rows would leave its `rho` at the prior, which is uniform, and any row scored into it
    later would get a dispersion drawn from nothing. That is a build failure rather than a
    metric, so it raises.
    """
    if bins is None:
        return {"n_rho": 1, "rho_bin": [1] * int(n_rows)}
    idx = np.asarray(bins, dtype=int)
    if len(idx) != int(n_rows):
        raise ValueError(f"rho_bin has {len(idx)} entries for {n_rows} rows")
    total = int(n_bins if n_bins is not None else idx.max())
    if idx.min() < 1 or idx.max() > total:
        raise ValueError(f"rho_bin must lie in 1..{total}; got {idx.min()}..{idx.max()}")
    missing = sorted(set(range(1, total + 1)) - set(idx.tolist()))
    if missing:
        raise ValueError(
            f"dispersion bins {missing} have no fitting rows — their rho would be drawn "
            f"from the uniform prior and applied to real rows at prediction time")
    return {"n_rho": total, "rho_bin": idx.tolist()}


# The low component's mean cannot exceed this. A "disrupted season" is one where the
# player misses more than half the schedule, and the bound is what stops the two
# components from label-switching — structural rather than hopeful. Mirrors
# `availability_window.MU_LOW_MAX`, which the point-MLE ladder fitted under.
MU_LOW_MAX = 0.5

# Normal scale on `gamma`, pi's covariate block. Weakly informative on standardized
# columns: at 2.5 a one-sd move in any covariate is free to swing the disruption odds by
# well over the 8.8x spread the point MLE fitted, so this rules out divergent coefficients
# rather than shrinking real ones. Deliberately NOT `prior_sd_for_l2(l2)`: the penalty in
# `availability_window` reaches `beta[1:]` only and leaves gamma unpenalized inside a box,
# which has no Bayesian analogue — so this is the one place the port is a choice rather
# than the identity `prior_sd_for_l2` makes everywhere else.
GAMMA_SCALE = 2.5


def pi_block(n_rows: int, Z: "np.ndarray | None" = None,
             gamma_scale: float = GAMMA_SCALE,
             mu_low_max: float = MU_LOW_MAX) -> dict:
    """The low-availability mixture's data block for `betabinomial_glm`.

    That file declares `P`, `Z`, `gamma_scale` and `mu_low_max` unconditionally because
    Stan has no optional data. Passing `Z=None` returns the **disabled** block — `P = 0`
    and a zero-column design — which makes `theta`, `mu_low`, `rho_low` and `gamma`
    zero-length and the model bit-for-bit the one that existed before the mixture was
    added, in the same way `year_block`'s `S = 0` and `rho_block`'s `n_rho = 1` are. Every
    head that does not want a mixture gets it from here rather than hand-writing four keys,
    so "disabled" has exactly one definition and the nesting is pinned in one place.

    `Z` carries **no intercept column**: `theta` is the scale, and an intercept inside the
    logit would put the nesting point at `gamma_0 -> -inf` instead of at an attainable
    parameter value.
    """
    if Z is None:
        return {"P": 0, "Z": np.zeros((int(n_rows), 0)),
                "gamma_scale": float(gamma_scale), "mu_low_max": float(mu_low_max)}
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or len(Z) != int(n_rows):
        raise ValueError(f"Z must be (rows x P) matching {n_rows} rows; got {Z.shape}")
    if Z.shape[1] == 0:
        raise ValueError("an empty Z is the DISABLED block — pass Z=None for it, so that "
                         "'no mixture' is one state rather than two that look alike")
    return {"P": int(Z.shape[1]), "Z": Z, "gamma_scale": float(gamma_scale),
            "mu_low_max": float(mu_low_max)}


def chain_summary(fit, names: list[str]) -> pd.DataFrame:
    """Per-chain posterior means, for a target that R-hat alone does not police.

    R-hat compares between-chain to within-chain variance and is the right diagnostic for a
    unimodal posterior explored at different rates. It is the **wrong** one for a mixture:
    four chains that each sit in a different mode, none of them mixing, can post a
    respectable R-hat while describing four different models. The point MLE of this
    likelihood needed multi-start for exactly that reason
    (`docs/availability-window-plan.md` §7b), so the chains are reported one at a time and
    `spread_in_sds` — the largest gap between two chain means, in pooled posterior sds — is
    the number to read.
    """
    draws = fit.draws(concat_chains=False)          # (iterations, chains, columns)
    columns = list(fit.column_names)
    rows = []
    for name in names:
        for j, column in enumerate(columns):
            if column != name and not column.startswith(f"{name}["):
                continue
            values = np.asarray(draws[:, :, j], dtype=float)
            pooled_sd = float(values.std(ddof=1))
            means = values.mean(axis=0)
            for chain in range(values.shape[1]):
                rows.append({
                    "parameter": column, "chain": chain + 1,
                    "mean": float(means[chain]),
                    "sd": float(values[:, chain].std(ddof=1)),
                    "q2_5": float(np.percentile(values[:, chain], 2.5)),
                    "q97_5": float(np.percentile(values[:, chain], 97.5)),
                    "pooled_mean": float(values.mean()),
                    "pooled_sd": pooled_sd,
                    "spread_in_sds": float((means.max() - means.min()) / pooled_sd)
                    if pooled_sd > 0 else 0.0,
                })
    return pd.DataFrame(rows)


class YearTerm:
    r"""One implementation of the year random effect, held by all four head classes.

    Disabled by default (`column=None`), in which case every method is a no-op returning
    zeros and the Stan data block is the `S = 0` one — so a head that does not want a
    season term is unchanged rather than "changed but with the coefficient near zero".

    ## What happens at prediction time, and why it is a DRAW rather than a lookup

    The fitted `year_z[s]` exist only for training seasons. The season being forecast has
    no `z`, and inventing one by carrying forward the last fitted value would be a season
    fixed effect smuggled in — the exact thing that is unusable here. So the predictive
    integrates over a **fresh** `z ~ normal(0, 1)`:

        eta_new = alpha + x'beta + sigma_year * z,   z ~ N(0, 1) per posterior draw

    One `z` per posterior draw, **shared across every row in that draw**. That sharing is
    the entire mechanism: a league shift is perfectly correlated across players, so it does
    not diversify away in a portfolio the way independent per-player error does. Drawing an
    independent `z` per player would reproduce the marginal widening while destroying the
    only property that makes it matter.

    ## The trap: mean-zero on the linear predictor is NOT mean-zero on the response

    Under a log link `E_z[exp(sigma*z)] = exp(sigma^2/2) > 1`, so integrating over the year
    effect *raises* every predicted count by that factor rather than leaving the mean
    alone. It is small at the measured league movement (sigma ~ 0.03 gives +0.05%) but it
    is not zero, and it is the difference between "this only widens the predictive" being
    true and merely nearly true. `response_multiplier` reports it so the claim is checked
    rather than assumed.

    ## One stream per head, not one stream for all of them

    `stream` splits the random draw so two heads fitted with the same `seed` do not get
    the *identical* `z` sequence. That would make every head's year effect perfectly
    correlated, which is a strong claim and a measured wrong one: `make season-effects`
    (`season_effects_shock_correlation.csv`) puts the mean pairwise correlation of
    detrended league movements at **-0.009** across 136 pairs — no common factor — with
    large correlations confined to specific pairs, `fg2a`-`fg3a` at **-0.833** being the
    3PA/2PA substitution the heads already reparameterize away. Independent streams are
    therefore the right default, and a shared one has to be argued for per pair.
    """

    def __init__(self, column: str | None = None, scale: float = YEAR_SD_SCALE,
                 seed: int = SEED, stream: str = ""):
        self.column, self.scale, self.seed = column, float(scale), int(seed)
        self.stream = str(stream)
        self.levels: list = []
        self.sigma_draws = np.zeros(0)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng([self.seed, zlib.crc32(self.stream.encode())])

    @property
    def enabled(self) -> bool:
        return self.column is not None

    def data(self, train: "pd.DataFrame") -> dict:
        """Stan data keys, and a record of which seasons the fitted `z` correspond to."""
        if not self.enabled:
            block, self.levels = year_block(len(train))
            return block
        block, self.levels = year_block(len(train), train[self.column].to_numpy(),
                                        scale=self.scale)
        return block

    def absorb(self, fit) -> None:
        """Keep the `sigma_year` draws; `year_z` is deliberately discarded.

        The fitted `z` values describe seasons that are over. Nothing downstream may read
        them — if they were kept, using them would be a season fixed effect — so they are
        not stored at all rather than stored and trusted not to be used.
        """
        if not self.enabled:
            self.sigma_draws = np.zeros(0)
            return
        draws = np.atleast_1d(fit.stan_variable("sigma_year"))
        self.sigma_draws = np.asarray(draws).reshape(len(draws), -1)[:, 0]

    def shift(self, idx: np.ndarray) -> np.ndarray:
        """`sigma_year * z` per posterior draw — one value per draw, shared across rows."""
        idx = np.asarray(idx)
        if not self.enabled or self.sigma_draws.size == 0:
            return np.zeros(len(idx))
        return self.sigma_draws[idx] * self._rng().standard_normal(len(idx))

    def response_multiplier(self, link: str = "log") -> float:
        """`E_z[exp(sigma*z)]` — how much integrating the year effect moves the MEAN.

        Exactly 1.0 when disabled. For a log link this is the multiplicative bias a
        mean-zero linear-predictor term induces on the response scale; for a logit link
        there is no closed form and the Monte Carlo estimate through `inv_logit` is what
        the predictive actually uses, so this reports the log-scale figure as the bound.
        """
        if not self.enabled or self.sigma_draws.size == 0:
            return 1.0
        return float(np.mean(np.exp(0.5 * self.sigma_draws ** 2)))

    def summary(self) -> dict:
        if not self.enabled or self.sigma_draws.size == 0:
            return {"year_effect": False, "sigma_year": 0.0, "sigma_year_sd": 0.0,
                    "n_train_seasons": 0, "response_multiplier": 1.0}
        return {"year_effect": True,
                "sigma_year": float(self.sigma_draws.mean()),
                "sigma_year_sd": float(self.sigma_draws.std(ddof=1)),
                "n_train_seasons": len(self.levels),
                "response_multiplier": self.response_multiplier()}


def compile_model(name: str, stan_dir: Path | str = STAN_DIR,
                  build_dir: Path | str = BUILD_DIR):
    """Compile `<name>.stan`, keeping every build artifact outside the source tree.

    cmdstanpy writes the generated `.hpp` and the executable next to whichever `.stan`
    file it is handed, so the source is **copied** into `outputs/stan/` and compiled
    there. `outputs/` is already gitignored, so the checked-in `src/stan/` stays clean
    without needing a new ignore rule.

    The copy is rewritten only when the text actually differs, which leaves its mtime
    alone on a no-op run — that is what lets cmdstanpy's own exe-vs-source staleness check
    do the right thing instead of recompiling every invocation.
    """
    from cmdstanpy import CmdStanModel

    source = Path(stan_dir) / f"{name}.stan"
    if not source.exists():
        raise FileNotFoundError(f"no Stan source at {source}")
    build = Path(build_dir)
    build.mkdir(parents=True, exist_ok=True)

    staged = build / f"{name}.stan"
    text = source.read_text()
    if not staged.exists() or staged.read_text() != text:
        staged.write_text(text)
    return CmdStanModel(stan_file=staged)


def cmdstan_version() -> str:
    from cmdstanpy import cmdstan_path
    return Path(cmdstan_path()).name


def sample(model, data: dict, chains: int = CHAINS, warmup: int = WARMUP,
           samples: int = SAMPLES, seed: int = SEED,
           adapt_delta: float = ADAPT_DELTA, show_progress: bool = False,
           label: str = "", inits: dict | None = None,
           metric: str | None = None) -> tuple[object, dict]:
    """Run NUTS and return the fit alongside its diagnostics.

    **Pass `inits`.** Stan's default initialization is uniform(-2, 2) on the unconstrained
    scale, which on a K-feature logit model puts the starting linear predictor at roughly
    2*sqrt(K) — about 9 at K = 19. `inv_logit` saturates to exactly 1.0 in double precision
    by an argument of 37, and a beta-binomial shape parameter of exactly 0 is a rejection.
    Those rejections are recoverable and warmup usually survives them, but they are noisy,
    they waste adaptation, and on a harder posterior they are how a chain fails to start at
    all. Every head here inits at the intercept-only solution with zero slopes.

    Wall clock is measured around the whole call rather than read from CmdStan's own
    timing, so it includes startup and the CSV round-trip — the number a caller planning
    forty fits actually needs.
    """
    started = time.perf_counter()
    # `metric=None` keeps CmdStan's default (diag_e). A caller passes "dense_e" when
    # the posterior carries strong linear correlations a diagonal metric cannot
    # absorb — cheap at these dimensions (every head here is < ~30 parameters) and
    # worth an order of magnitude in treedepth on the composition head.
    extra = {} if metric is None else {"metric": metric}
    fit = model.sample(data=data, chains=chains, iter_warmup=warmup,
                       iter_sampling=samples, seed=seed, adapt_delta=adapt_delta,
                       inits=inits, show_progress=show_progress, show_console=False,
                       **extra)
    seconds = time.perf_counter() - started
    return fit, diagnostics(fit, seconds, label=label)


def diagnostics(fit, seconds: float, label: str = "") -> dict:
    """R-hat, ESS, divergences, treedepth and wall clock for one fit.

    Summarized over the *sampled parameters only*: `lp__` and the sampler's own
    `__`-suffixed columns are excluded, since their R-hat is a diagnostic of a different
    thing and would mask a parameter that genuinely failed to mix.
    """
    summary = fit.summary()
    params = summary[~summary.index.astype(str).str.endswith("__")]

    def worst(col: str, how: str) -> float:
        if col not in params.columns or params.empty:
            return float("nan")
        values = pd.to_numeric(params[col], errors="coerce").dropna()
        if values.empty:
            return float("nan")
        return float(values.max() if how == "max" else values.min())

    method = fit.method_variables()
    divergent = int(np.sum(method["divergent__"])) if "divergent__" in method else -1
    # 10 is NUTS's default max_treedepth, which nothing here overrides. Saturation is a
    # statement about efficiency rather than validity — the sampler ran out of doubling
    # budget — and it is what a badly conditioned design (a B-spline basis, say) looks like.
    treedepth = (int(np.sum(method["treedepth__"] >= MAX_TREEDEPTH))
                 if "treedepth__" in method else -1)

    max_rhat = worst("R_hat", "max")
    min_ess = min(worst("ESS_bulk", "min"), worst("ESS_tail", "min"))
    return {
        "label": label,
        "max_rhat": max_rhat,
        "min_ess_bulk": worst("ESS_bulk", "min"),
        "min_ess_tail": worst("ESS_tail", "min"),
        "divergences": divergent,
        "treedepth_saturated": treedepth,
        "n_draws": int(fit.num_draws_sampling * fit.chains),
        "wall_clock_s": float(seconds),
        "cmdstan": cmdstan_version(),
        "converged": bool(np.isfinite(max_rhat) and max_rhat <= RHAT_MAX
                          and np.isfinite(min_ess) and min_ess >= ESS_MIN
                          and divergent == 0),
    }


def warn_if_unconverged(diag: dict) -> None:
    """Print a loud line for any fit that failed a convergence bar.

    Deliberately noisy. A divergence count above zero means the sampler could not
    integrate the posterior geometry, and every metric computed from that fit inherits the
    problem — including, and especially, the ones that look good.
    """
    if diag.get("converged"):
        return
    problems = []
    if not (diag["max_rhat"] <= RHAT_MAX):
        problems.append(f"R-hat {diag['max_rhat']:.4f} > {RHAT_MAX}")
    ess = min(diag["min_ess_bulk"], diag["min_ess_tail"])
    if not (ess >= ESS_MIN):
        problems.append(f"ESS {ess:.0f} < {ESS_MIN:.0f}")
    if diag["divergences"]:
        problems.append(f"{diag['divergences']} divergences")
    if diag["treedepth_saturated"]:
        problems.append(f"{diag['treedepth_saturated']} treedepth-saturated draws")
    print(f"   /!\\  {diag['label'] or 'fit'}: {', '.join(problems)} — "
          f"treat its metrics as untrustworthy, not merely noisy")


def posterior(fit, names: list[str]) -> dict[str, np.ndarray]:
    """Draws for the named variables, each shaped (draws, ...)."""
    return {name: np.atleast_1d(fit.stan_variable(name)) for name in names}


def thin(n_draws: int, keep: int, seed: int = SEED) -> np.ndarray:
    """Evenly spaced draw indices — for pushing a posterior through a costly transform.

    Evenly spaced rather than random: the draws arrive chain-major, so a regular stride
    takes from every chain in the same proportion, and there is nothing left to randomize
    once the chains have mixed.
    """
    if keep >= n_draws:
        return np.arange(n_draws)
    return np.linspace(0, n_draws - 1, keep).round().astype(int)


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per row from predictive samples, in the units of `y`.

    `src/models/availability.py::crps` sums over an explicit pmf grid, which is right for
    games played (0..83) and impossible for minutes (0..~4,000) or a season rebound total
    — the grid alone would be draws x rows x support.

    Uses the energy form, `CRPS = E|X - y| - 0.5*E|X - X'|`, with the second expectation
    evaluated in closed form from the order statistics rather than by a second O(S^2)
    double sum:

        sum_ij |x_i - x_j| = 2 * sum_i (2i - S - 1) * x_(i)    (i = 1..S, x sorted)

    so the whole thing is O(S log S) per row. `tests/test_stan_heads.py` pins it against
    the literal double sum on small inputs, and against the exact pmf CRPS on a discrete
    distribution — the same treatment `season_total.py::crps_from_atoms` gets, because an
    unchecked closed form is exactly the kind of thing that is quietly wrong by a factor
    of two.

    `samples` is (draws x rows); `y` is (rows,).
    """
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(y):
        raise ValueError(f"samples must be (draws x rows) matching y; got "
                         f"{samples.shape} against {len(y)} rows")
    S = samples.shape[0]
    term1 = np.abs(samples - y[None, :]).mean(axis=0)
    ordered = np.sort(samples, axis=0)
    weights = (2 * np.arange(1, S + 1) - S - 1).astype(float)[:, None]
    term2 = 2.0 * (weights * ordered).sum(axis=0) / (S * S)
    return term1 - 0.5 * term2


def pit_from_samples(samples: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Randomized PIT from predictive samples — uniform iff calibrated.

    Randomized across the probability mass at `y` itself for the same reason
    `availability.pit_values` is: the non-randomized PIT of a discrete predictive is not
    uniform even under a perfect model, so its histogram would show discreteness
    artifacts and read as miscalibration.
    """
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)[None, :]
    below = (samples < y).mean(axis=0)
    at = (samples == y).mean(axis=0)
    return below + np.random.default_rng(seed).random(len(below)) * at


def ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of PIT values from uniform — one calibration number."""
    u = np.sort(np.asarray(u, dtype=float))
    grid = np.linspace(0, 1, 101)
    return float(np.max(np.abs(np.searchsorted(u, grid) / max(len(u), 1) - grid)))


def standardized(train: pd.DataFrame, frames: list[pd.DataFrame], features: list[str]
                 ) -> tuple[list[np.ndarray], object]:
    """Z-score `features` using **train** moments only, applied to every frame.

    Fitting the scaler on the pooled frame would leak the held-out distribution into the
    design — the same class of error `assert_point_in_time` guards against on the time
    axis, and just as invisible in the output.
    """
    from sklearn.preprocessing import StandardScaler

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        X = frame[features].to_numpy(dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler().fit(matrix(train))
    return [scaler.transform(matrix(f)) for f in frames], scaler


def diagnostics_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[
        ["label", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
         "treedepth_saturated", "n_draws", "wall_clock_s", "converged", "cmdstan"]]
