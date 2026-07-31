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


def prior_sd_for_l2(l2: float) -> float:
    """Prior sd whose posterior mode equals a penalized MLE with L2 weight `l2`.

    `-loglik + l2*||beta||^2` and `-loglik + ||beta||^2/(2*sigma^2)` are the same
    objective when `sigma = 1/sqrt(2*l2)`. Used so the ported head is a port and not a
    different model that happens to resemble one.
    """
    if l2 <= 0:
        raise ValueError(f"l2 must be positive to correspond to a proper prior; got {l2}")
    return float(1.0 / np.sqrt(2.0 * l2))


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
