"""Does Stan's marginal path compute the integral it claims to? — at full scale.

`tests/test_stan_composition.py` checks the quadrature against a brute-force dense grid on
tiny synthetic units, which is the right shape for a unit test and is what caught the
flat-unit placement bug. It cannot do this: build the **real** pilot frame (a minute of
pandas), hand it to Stan through the production driver, and compare the target against an
independent numpy evaluation of the same integral over 97,587 rows, 2,062 player-season
units and all 1,487 feasibility-bounded rows at once.

Two log-posterior evaluations, no sampling — seconds after the frame is built. That is what
makes it worth having as its own target rather than a scratch script: the thing it verifies
is the one approximation in the whole representation, and re-verifying it after any change to
`composition_glm.stan`'s `Q > 0` block should cost a coffee rather than a fit.

**The comparison is a DIFFERENCE between two parameter vectors, deliberately.** Stan's `~`
statements drop constants and `beta_binomial_lupmf` drops the data-only `lchoose(m, y)`; all
of those are parameter-independent, so they cancel in a difference and what survives is
exactly the quantity the quadrature computes.

⚠️ **`sig_figs` is load-bearing here and its absence is a silent trap.** CmdStan reports
`log_prob` at EIGHT significant figures by default, and this target is ~2.55e6 — so the
default rounds every value to +/-0.05, and the first run of this check "failed" at a gap of
1.4e-2 that was entirely the reporting. At the synthetic scale (~200) eight figures is 1e-6
and the problem never appears, which is precisely why it would be missed.

Usage:
    python -m src.models.composition_quadrature_check
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import betaln, gammaln, logsumexp

from src.models.held_out import selection_split
from src.models.stan_composition import (GH_INFLATE, PILOT_FIRST_SEASON, Q_NODES,
                                         RHO_BINS, TEST_SEASONS, StanComposition,
                                         composition_frame, variants)
from src.models.stan_utils import compile_model

# Eight significant figures on a 2.5e6 target is +/-0.05. Ask for all of them.
SIG_FIGS = 17

# Floating-point identity is the bar, not "close enough": both sides evaluate the same
# formula in double precision over the same 88,389 terms, so anything above rounding noise
# is a real disagreement about what the model block computes.
TOLERANCE = 1e-3


def numpy_target(data: dict, alpha: float, beta: np.ndarray, rho: np.ndarray,
                 sigma: np.ndarray) -> float:
    """The `Q > 0` model block, re-implemented from the DATA DICT alone.

    Deliberately written against `data` rather than against the frame or the term object:
    an error shared with `QuadratureTerm` would cancel and the check would pass on a bug.
    What it re-derives is only the arithmetic Stan does with what it was handed.
    """
    X = np.asarray(data["X"], dtype=float)
    u_row = np.asarray(data["u_row"], dtype=int) - 1
    u_start = np.asarray(data["u_start"], dtype=int) - 1
    u_len = np.asarray(data["u_len"], dtype=int)
    u_bin = np.asarray(data["u_bin"], dtype=int) - 1
    center = np.asarray(data["u_center"], dtype=float)
    curv = np.asarray(data["u_curv"], dtype=float)
    gh_x = np.asarray(data["gh_x"], dtype=float)
    gh_log_w = np.asarray(data["gh_log_w"], dtype=float)
    inflate = float(data["gh_inflate"])

    y_u = np.asarray(data["y"], dtype=float)[u_row]
    m_u = np.asarray(data["m"], dtype=float)[u_row]
    lo_u = np.asarray(data["lo"], dtype=int)[u_row]
    rb_u = np.asarray(data["rho_bin"], dtype=int)[u_row] - 1
    unit_of_row = np.repeat(np.arange(len(u_start)), u_len)
    bound = np.flatnonzero(lo_u > 0)

    eta_u = (np.asarray(data["logit_prior"], dtype=float) + alpha + X @ beta)[u_row]
    s_u = (1 - rho[rb_u]) / rho[rb_u]
    sig = sigma[u_bin]
    prec = curv + 1.0 / sig ** 2
    c = center * curv / prec
    scale = inflate / np.sqrt(prec)

    node_lp = np.empty((len(u_start), len(gh_x)))
    for q, (x, log_w) in enumerate(zip(gh_x, gh_log_w)):
        uq = c + np.sqrt(2.0) * scale * x
        e = eta_u + uq[unit_of_row]
        p = 0.5 * (1 + np.tanh(0.5 * np.clip(e, -400, 400)))
        a, b = s_u * p, s_u * (1 - p)
        term = betaln(y_u + a, m_u - y_u + b) - betaln(a, b)
        if len(bound):
            term[bound] -= _log_tail(m_u[bound], lo_u[bound].astype(float),
                                     a[bound], b[bound])
        node_lp[:, q] = (log_w + np.log(np.sqrt(2.0) * scale)
                         - 0.5 * (uq / sig) ** 2 - np.log(sig) - 0.5 * np.log(2 * np.pi)
                         + np.add.reduceat(term, u_start))
    return float(
        logsumexp(node_lp, axis=1).sum()
        - 0.5 * (alpha / data["intercept_scale"]) ** 2
        - 0.5 * float(np.sum((beta / data["beta_scale"]) ** 2))
        - 0.5 * float(np.sum((sigma / data["u_sd_scale"]) ** 2)))


def _log_tail(n: np.ndarray, k0: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log P(Y >= lo), summed UPWARD by the pmf-ratio recurrence — Stan's own shape.

    Not `1 - cdf`: when the tail is tiny the head mass rounds to exactly 1.0 and `log1m(1)`
    is -inf, which is the failure `beta_binomial_log_tail_mass` was written to avoid.
    """
    term = (gammaln(n + 1) - gammaln(k0 + 1) - gammaln(n - k0 + 1)
            + betaln(k0 + a, n - k0 + b) - betaln(a, b))
    total, k = term.copy(), k0.copy()
    for _ in range(int(n.max())):
        live = k < n - 0.5
        if not live.any():
            break
        ratio = np.where(live, ((n - k) * (a + k)) / ((k + 1.0) * (b + n - k - 1.0)), 1.0)
        term = term + np.log(ratio)
        total = np.where(live, np.logaddexp(total, term), total)
        k = k + 1.0
    return total


def run(cfg: dict) -> pd.DataFrame:
    comp = cfg.get("stan", {}).get("composition", {})
    eff = comp.get("effects", {})
    first_season = str(eff.get("first_season", PILOT_FIRST_SEASON))
    quad = eff.get("quadrature", {})

    # `composition_frame`, NOT `head_frame` — the same builder `composition_effects` uses.
    # The verdict does not depend on which frame this runs over (it is checking arithmetic,
    # and any valid data would do), but exercising the rows the LADDER will fit means the
    # check sees that ladder's own `u_center` / `u_curv`, including whatever edge cases the
    # window happens to contain. A check that ran on a different frame from the thing it is
    # guarding would be the weaker of the two for no reason.
    frame = composition_frame(cfg)
    windowed = frame[frame["season"] >= first_season].reset_index(drop=True)
    train, val = selection_split(windowed, TEST_SEASONS)
    tr, _, feats, dispersed, n_rho = variants(train, val)["betabinom_ot_graded"]

    head = StanComposition(feats, dispersed, n_rho, name="quadrature-check",
                           quadrature=True, n_sigma=RHO_BINS,
                           q_nodes=int(quad.get("nodes", Q_NODES)),
                           gh_inflate=float(quad.get("inflate", GH_INFLATE)))
    data = head.stan_data(tr)
    bounded = int((np.asarray(data["lo"])[np.asarray(data["u_row"]) - 1] > 0).sum())
    print(f"  {data['P']:,} rows, {data['n_unit']:,} units, Q = {data['Q']}, "
          f"n_sigma = {data['n_sigma']}, {bounded:,} bounded rows")

    model = compile_model("composition_glm")
    rng = np.random.default_rng(0)
    K = int(data["K"])
    arms = [
        {"alpha": -0.06, "beta": rng.normal(0, 0.05, K),
         "rho": np.array([0.140, 0.107, 0.091, 0.074]),
         "sigma_u": np.full(RHO_BINS, 0.44)},
        {"alpha": 0.12, "beta": rng.normal(0, 0.05, K),
         "rho": np.array([0.120, 0.100, 0.090, 0.080]),
         "sigma_u": np.array([0.52, 0.40, 0.36, 0.30])},
    ]
    rows = []
    for i, pars in enumerate(arms):
        stan_lp = float(model.log_prob(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in pars.items()},
            data=data, jacobian=False, sig_figs=SIG_FIGS).iloc[0, 0])
        np_lp = numpy_target(data, pars["alpha"], pars["beta"], pars["rho"],
                             pars["sigma_u"])
        rows.append({"arm": i, "stan_log_prob": stan_lp, "numpy_log_prob": np_lp,
                     "abs_gap": abs(stan_lp - np_lp)})
        print(f"  arm {i}: stan {stan_lp:.8f}  numpy {np_lp:.8f}  "
              f"gap {abs(stan_lp - np_lp):.3e}")

    table = pd.DataFrame(rows)
    d_stan = table["stan_log_prob"].iloc[0] - table["stan_log_prob"].iloc[1]
    d_np = table["numpy_log_prob"].iloc[0] - table["numpy_log_prob"].iloc[1]
    gap = abs(d_stan - d_np)
    table["difference_stan"] = d_stan
    table["difference_numpy"] = d_np
    table["difference_gap"] = gap
    table["n_rows"] = data["P"]
    table["n_units"] = data["n_unit"]
    table["q_nodes"] = data["Q"]
    table["first_season"] = first_season

    print(f"\n  difference between arms: stan {d_stan:.8f}, numpy {d_np:.8f}, "
          f"gap {gap:.3e} ({gap / abs(d_np):.2e} relative)")
    if gap > TOLERANCE:
        raise AssertionError(
            f"the marginal path disagrees with an independent evaluation of the same "
            f"integral by {gap:.3e} nats. This is a correctness failure, not a tuning "
            f"one — do not read any metric from an `mq` arm until it is explained. "
            f"Check `sig_figs` first: CmdStan's default of 8 rounds a 2.5e6 target to "
            f"+/-0.05.")
    print("  PASS — the model block computes the integral it claims to.")

    dest = Path(cfg["evaluation"]["predictions_dir"]) / "composition_quadrature_check.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print(f"Saved {len(table):,} rows → {dest}")
    return table


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
