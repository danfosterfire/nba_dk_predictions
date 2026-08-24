r"""The eleven component rate heads in Stan — seven counts, four conversions.

The third link in the chain: `counts | min` and `makes | attempts`. Together with the
availability and minutes heads this is the full output contract, and `dk_pts` is
reassembled deterministically from a joint draw rather than predicted directly.

| head | likelihood | exposure / trials |
|---|---|---|
| `fga` `fta` `reb` `ast` `stl` `blk` `tov` | negative binomial, log link | season minutes |
| `fg3a\|fga` `fg2m\|fg2a` `fg3m\|fg3a` `ftm\|fta` | beta-binomial, logit link | attempts |

`fg2a` is **derived**, not fitted: `fga - fg3a`, exactly as `pts` is derived from the
makes. Total attempts are the count and the three-point mix is a share of them, so the
3PA/2PA substitution is enforced by construction instead of being left to the residual
copula. Adopted 2026-08-03 on `docs/shot-attempt-basis-plan.md`; the head count is
unchanged at eleven.

## Eleven fits, not one joint model — and this is an identity, not a shortcut

The parameter blocks are disjoint and the priors independent, so the joint posterior
factorizes **exactly**: eleven separate fits recover the identical posterior a single
model would. `megamodel.stan` in the prior attempt is the cautionary case — every head had
its own `beta_*` with an independent prior and the player-effect terms were commented out,
so it was thirteen independent GLMs in one file paying the full joint-fit price. It ran on
`sample_frac(0.01)`. Ninety-nine percent of the data given up for a coupling that was not
in the model.

The correlation the simulator needs enters at **draw** time: one `min` draw pushed through
all eleven heads as exposure (minutes is **46.4%** of within-player residual variance, by far
the largest common factor), then a Gaussian copula for the remainder if needed —
conditioning minutes out, the residual off-diagonals average **+0.007** with a max of
**+0.1329** (`fga`-`reb`). The one structural exception was the 3PA/2PA substitution, and the
shot-attempt basis removes it from the copula entirely rather than coupling two count heads.
`substitution_arm` below now measures the RETIRED two-count basis against the shipped one.

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

**A validation split, and only that.** Variants are selected on a split carved out of
train, and since 2026-08-05 the test seasons are neither fitted nor scored here —
`src/models/held_out.py` raises on anything that reaches for them, and
`src/final_evaluation.py` takes the held-out reading once. Removing the test column removed
two things at once: the standing invitation to select on it, and a confound, because the
test side used to **refit on train + validation at double the sampler iterations**, so a
val/test disagreement conflated the evaluation rows with the training data and the chain
length and could not serve as the replication check it looked like. Selection now runs at
full-length chains, since there is no longer a cheap side to trade against.

The measured expectation (`make component-rates`) is that
**scale beats curvature**: `log E[rate] = beta*log(prior rate)` makes the model
`rate ~ prior_rate^beta`, which is the right shape, while linear-in-raw-rate inside `exp()`
is badly misspecified — R^2 0.520 on `fg3a` and 0.638 on `blk` under Poisson, and far worse
under the negative binomial actually used. Splines then matter most on the skewed heads. The spline basis here is
built on `log(own)` rather than raw `own`, so it strictly nests `log_own` and the
comparison isolates curvature *given* the right scale.

Usage:
    python -m src.models.stan_components
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year
from src.models.availability import EPS, RHO_MAX, RHO_MIN, fit_dispersion
from src.models.availability import _neg_loglik as beta_binomial_nll
from src.models.component_rates import (BIO_COLS, CONTEXT_COLS, CONVERSION_HEADS,
                                        COUNT_HEADS, add_log, add_spline,
                                        build_design, carry_forward,
                                        carry_forward_conversion, fit_nb_dispersion,
                                        impute, nb_nll)
from src.models.held_out import selection_split
from src.models.stan_utils import (YearTerm, compile_model, crps_from_samples,
                                   diagnostics_frame, ks_uniform, pi_block,
                                   pit_from_samples, posterior, prior_sd_for_l2,
                                   rho_block, sample, standardized, thin,
                                   warn_if_unconverged)

COUNT_MODEL = "negbinomial_glm"
CONVERSION_MODEL = "betabinomial_glm"

SPLINE_KNOTS = 5
PREDICTIVE_SAMPLES = 1000

# ── The preseason block, adopted 2026-08-15 ───────────────────────────────────────────
#
# `docs/preseason-plan.md` session 6b. Every one of the eleven heads carries it, and the
# shipped column is the **volume-shrunk** delta — `pre_d_<head>` multiplied by
# `min_pre / (min_pre + k)` at that head's own `k`, fitted on the fitting half — plus the four
# age-split missing indicators. Five columns per head.
#
# **Every head clears the rolling-origin half of the bar**, at 11 to 13 of 13 origins with
# intervals clear of zero on 4,300 scored fitting-half rows. **Seven also clear validation**:
# `fga` (−3.0093 [−4.3826, −1.7217]), `fg3a|fga` (−1.4373), `reb` (−1.0929), `ast` (−1.0296),
# `fta` (−0.6300), `tov` (−0.2213) and `fg2m|fg2a` (−0.1541). The remaining four — `stl`,
# `blk`, `ftm|fta`, `fg3m|fg3a` — have favourable point estimates at both readings and
# validation intervals that reach across zero on 706 rows. **No head anywhere in the round
# has an interval clear of zero on the wrong side**, so nothing here ships against evidence
# of harm; the four ship against a half of the bar that could not resolve them, which is the
# same owner decision P2 recorded on the availability head.
#
# **The shrink is the form the fitting half chose, and it is not P1's additive term.**
# `own_delta_shrunk` beats the declared primary on the fitting half on every head, intervals
# clear of zero. `k` runs 20 to 320 pseudo-minutes and is READ from the artifact that fitted
# it rather than pinned here — a per-36 rate over 60 preseason minutes divides by the same
# exposure a minutes total is measured on, so the right `k` is a property of the head.
#
# **Season-centring is NOT used here**, unlike `stan_minutes`. It loses to the uncentred delta
# on this family with intervals clear of zero on three heads: the compression it corrects is a
# property of *levels*, and a per-36 rate has already divided the exposure out.
#
# ⚠️ Three heads ship a block P1's ΔR² screen called actively harmful — `blk` (−0.00847,
# z = −3.18), `fg2m|fg2a` (−0.00845, z = −9.44) and `fta` (−0.00161, z = −2.32). None of the
# three reproduced as harm at a paired interval; `fta` and `fg2m|fg2a` clear both halves
# outright and `blk` clears the rolling one. The screen was a single inner split against a
# permutation null with no row-level uncertainty, and it also misranked the heads it admitted
# in both directions. `stan.components.preseason: false` is the exact rollback.
PRESEASON = True

#: Columns appended per head. The delta is the head's own, on its own link scale; the four
#: indicators are shared and are P1 decision 3's age split.
PRESEASON_MISSING_COLS = ["pre_missing__<24", "pre_missing__24-27", "pre_missing__28-31",
                          "pre_missing__32+"]

# ⚠️ **`fg3m|fg3a` carries no block, and it is the one head in the round measured as WORSE
# with one.** Ten of the eleven improve under the posterior at a median retention of 0.991;
# this one goes the other way, and it loses on three metrics rather than one — CRPS 4.19350
# against its control's 4.16436, NLL 3.16495 against 3.16432, PIT KS 0.02536 against 0.02416.
#
# Three instruments agree, which is what makes it a finding rather than a noisy row:
#
#   1. **P1's attribution.** The head's apparent +0.01488 ΔR² was entirely the shared
#      indicator pair (+0.01987); its own preseason 3P% delta was −0.00237. The attribution
#      split exists to catch exactly this and it caught it first.
#   2. **6b's pooled point MLE**: +0.00688 — the block already scored worse on the full
#      validation frame before any sampler ran.
#   3. **The posterior control**: +0.02914, the same sign and larger.
#
# The mechanism is the one `ftm|fta` shows from the other side: a conversion delta is a logit
# of a percentage taken over a handful of preseason attempts, and shooting percentage is the
# least persistent quantity in the box score. Prior-season 3P% over ~200 attempts is simply a
# better estimate than preseason 3P% over ~15, so the block adds variance and no signal.
# Owner decision, 2026-08-15, taken on the posterior reading.
#
# Named by the `made` column because that is what `head_preseason_cols` is keyed on.
PRESEASON_EXCLUDE: frozenset[str] = frozenset({"fg3m"})

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


def head_label(component: str, attempted: str | None = None) -> str:
    """`"reb"` or `"ftm|fta"` — the key every artifact in this family is indexed by."""
    return component if attempted is None else f"{component}|{attempted}"


def head_preseason_cols(component: str, preseason: bool | None = None) -> list[str]:
    """The five columns `component`'s arm adds, or `[]` when the block is off.

    Per head rather than shared, unlike `stan_minutes.PRESEASON_COLS`, because each head's
    delta is on **its own link** — `log1p` of a per-36 rate for a count, `logit` of a
    percentage for a conversion. A shared block here would put `reb`'s preseason rebounding
    on `blk`'s linear predictor.

    `PRESEASON_EXCLUDE` opts a head out entirely, which is what makes the block a per-head
    decision rather than a family-wide one — `fg3m|fg3a` is measured as *worse* with it, so
    it fits the head that shipped before 2026-08-15 while its ten siblings do not.
    """
    if not (PRESEASON if preseason is None else bool(preseason)):
        return []
    if component in PRESEASON_EXCLUDE:
        return []
    return [f"pre_d_{component}_shrunk"] + list(PRESEASON_MISSING_COLS)


def head_features(features: list[str], component: str,
                  preseason: bool | None = None) -> list[str]:
    """A variant's feature list plus that head's preseason block, appended.

    Appended rather than woven in, so the block is a **suffix** on every variant: the sweep
    still answers "which curvature on the prior-rate term" with the block held common, and a
    persisted recipe's column order stays stable when the flag flips.
    """
    return list(features) + head_preseason_cols(component, preseason)


def shrinkage_constants(cfg: dict) -> dict[str, float]:
    """`head -> the volume shrink `k` session 6b fitted`, off its own artifact.

    Read rather than pinned, for the reason `posteriors.selected_variant` is read: `k` is a
    **fitted quantity**, estimated on an inner carve of the fitting half by
    `components_preseason.fit_shrinkage`, and a constant copied into this module is a
    constant that can silently disagree with the run that chose it. The artifact is also
    what `make docs-audit` re-derives the quoted grid from.
    """
    from src.models.components_preseason import heads as _armed

    path = Path(cfg["evaluation"]["predictions_dir"]) / "components_preseason_shrinkage.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing and the component heads ship a volume-shrunk preseason "
            f"delta — run `make components-preseason`, or set `stan.components.preseason: "
            f"false` to fit the pre-2026-08-15 heads exactly.")
    table = pd.read_csv(path)
    chosen = table[table["selected"]].set_index("head")["k"].astype(float).to_dict()
    missing = [h.name for h in _armed() if h.name not in chosen]
    if missing:
        raise ValueError(f"{path} selects no shrinkage constant for: {', '.join(missing)}")
    return {str(k): float(v) for k, v in chosen.items()}


def head_design(cfg: dict, preseason: bool | None = None,
                design: pd.DataFrame | None = None) -> pd.DataFrame:
    """`build_design` plus every head's preseason block — **this family's path, no other's**.

    Separate from `component_rates.build_design` for the reason `stan_minutes.head_design` is
    separate from its own builder: that function is how `season_terms`, `posteriors`, the
    substitution sweep and `components_preseason` itself reach their rows, and a column that
    is structurally zero before 2004-05 must not enter any of them by accident.

    The block is built by `components_preseason.attach_preseason` — the same function the 6b
    ladder measured it with — so the coefficients these heads fit are coefficients on columns
    that were measured, not on a second implementation of them.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    if design is None:
        targets = pd.read_parquet(features_dir / "component_targets.parquet")
        design = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    # `design` lets the forward path (`features/forward_design.py`) bring its own rows
    # through the SAME preseason attachment; `None` is today's behaviour exactly.
    if not (PRESEASON if preseason is None else bool(preseason)):
        return design

    # Function-level: `components_preseason` imports this module at the top.
    from src.models.components_preseason import (attach_preseason, heads as _armed,
                                                 with_shrunk_delta)

    panel_path = features_dir / "preseason.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} is missing and the component heads ship a preseason block — run "
            f"`make preseason`, or set `stan.components.preseason: false` to fit the "
            f"pre-2026-08-15 heads exactly.")
    out = attach_preseason(design, pd.read_parquet(panel_path))

    # Each head's delta shrunk at ITS OWN fitted `k`. One frame carries all eleven shrunk
    # columns because the column names are per head, so nothing collides.
    constants = shrinkage_constants(cfg)
    for head in _armed():
        out = with_shrunk_delta(out, head.delta, constants[head.name])

    wanted = sorted({c for h in _armed()
                     for c in head_preseason_cols(h.component, True)})
    missing = [c for c in wanted if c not in out.columns]
    if missing:
        raise ValueError(f"`attach_preseason` did not produce {missing}")
    bad = [c for c in wanted if out[c].isna().any()]
    if bad:
        raise ValueError(f"NaN in the shipped preseason block: {bad}")
    return out


def head_fitting_rows(train_full: pd.DataFrame, train_covered: pd.DataFrame,
                      component: str, preseason: bool | None = None) -> pd.DataFrame:
    """The fitting rows for ONE head — cut only if that head carries a block.

    **The window is a per-head property, not a family-wide one**, and treating it as shared
    is a defect this module shipped for one afternoon. The cut exists solely because a
    missing-preseason indicator on a pre-2005 row is an era dummy; a head with no such
    indicator has nothing to protect against, so cutting it throws away 2,248 of 8,630
    training rows (26%) to buy nothing.

    That is exactly `fg3m|fg3a`'s position: `PRESEASON_EXCLUDE` opts it out on its own
    measurement, so `stan.components.preseason: false` must mean the pre-2026-08-15 head
    *including* its 1997-98 window, which is what this module's config comment already
    promises.

    Heads on different windows is normal here rather than a compromise — they are fitted
    separately and the factorization is exact, and `stan_minutes` (2004-05) and
    `stan_composition` (1996-97) already differed before this. The decision that opted the
    head out is unaffected: the control it was measured against was fitted on the covered
    window too, so that comparison held the window fixed.
    """
    return train_covered if head_preseason_cols(component, preseason) else train_full


def covered_fitting_rows(train: pd.DataFrame, cfg: dict,
                         preseason: bool | None = None) -> pd.DataFrame:
    """The fitting rows, cut to the seasons the preseason panel actually covers.

    **Applied to the FITTING rows only** — the design is built over every season regardless
    and the validation rows are never touched, the discipline `stan_minutes` and
    `stan_availability` both document.

    Required by the block rather than chosen for its own sake: this design fits from 1997-98
    and the panel begins at 2004-05, so 2,248 of 8,630 training rows would carry the
    missing-preseason indicator for a reason that is a fact about the NBA's API rather than
    about the player. 6b priced the cut on its own so it cannot be credited to the block —
    the covered-window incumbent is within **0.133** CRPS of the full-window one on every
    head, 4.8% of the block at worst, and on `tov` and `stl` the cut *helps*. That is a very
    different bill from the quarter of the increment `stan_minutes` pays, because prior-season
    rates are the most persistent quantity in the project and the lost seasons buy little.
    """
    if not (PRESEASON if preseason is None else bool(preseason)):
        return train
    from src.models.stan_minutes import first_covered_season

    first = _season_start_year(first_covered_season(cfg))
    return train[train["season"].map(_season_start_year) >= first].copy()


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
                        attempted: str, n_knots: int = SPLINE_KNOTS,
                        own: str | None = None
                        ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """The same scale-then-curvature ladder, on the logit that a logit link wants.

    `own` names the prior-season rate column explicitly. It defaults to the
    `{made}_pct_lag1` convention every conversion head in `CONVERSION_HEADS` follows,
    but the convention does NOT generalize: the `fg3a | fga` share head's own rate is
    the attempt-*mix* share `fg3a_share_lag1`, and `fg3a_pct_lag1` — which the
    convention would ask for — is three-point *shooting* percentage. That column does
    not exist today, so the call would raise; the danger is someone adding it, at which
    point the head would silently fit on shooting accuracy instead of shot mix and
    destroy its entire rationale. Naming the column is the fix.
    """
    own = own or f"{made}_pct_lag1"
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
                 seed: int = 42, predictive_samples: int = PREDICTIVE_SAMPLES,
                 year_column: str | None = None, metric: str | None = None):
        self.features, self.component, self.name = features, component, name
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed
        self.predictive_samples = predictive_samples
        self.metric = metric
        # `year_column=None` is the disabled term, which makes the Stan parameter vectors
        # zero-length and the model identical to the one that existed before it. `stream`
        # is the head's name, so two heads fitted with the same seed draw INDEPENDENT year
        # effects — which is what the measured cross-component shock correlation supports.
        self.year = YearTerm(year_column, seed=seed, stream=name)

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
             "phi_inv_scale": PHI_INV_SCALE, **self.year.data(train)},
            chains=self.chains, warmup=self.warmup, samples=self.samples,
            seed=self.seed, label=self.name, metric=self.metric,
            inits={"alpha": float(np.log(max(y.sum(), 1) / exposure.sum())),
                   "beta": np.zeros(X.shape[1]).tolist(), "phi_inv": 0.05})
        warn_if_unconverged(self.diagnostics)

        draws = posterior(fit, ["alpha", "beta", "phi"])
        self.alpha_draws = draws["alpha"].reshape(-1)
        self.beta_draws = draws["beta"].reshape(len(self.alpha_draws), -1)
        self.phi_draws = draws["phi"].reshape(-1)
        self.phi = float(self.phi_draws.mean())
        self.year.absorb(fit)
        return self

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    def mu_draws(self, df: pd.DataFrame, keep: int) -> tuple[np.ndarray, np.ndarray]:
        idx = thin(len(self.alpha_draws), keep)
        eta = self._design(df) @ self.beta_draws[idx].T + self.alpha_draws[idx][None, :]
        # One year draw per posterior draw, added to every row alike — the shared factor
        # is the point, not the marginal width.
        eta = eta + self.year.shift(idx)[None, :]
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
                 predictive_samples: int = PREDICTIVE_SAMPLES,
                 year_column: str | None = None, metric: str | None = None):
        self.features, self.made, self.attempted = features, made, attempted
        self.name, self.l2 = name, l2
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed
        self.predictive_samples = predictive_samples
        self.metric = metric
        self.year = YearTerm(year_column, seed=seed, stream=name)

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
             "intercept_scale": INTERCEPT_SCALE, **self.year.data(live),
             # One dispersion for every row: `rho_block()` with no bins is the shared-rho
             # model exactly, which is what these four conversion heads have always fitted.
             **rho_block(len(live)),
             # And no low-availability mixture: `pi_block()` with no design is `P = 0`,
             # which makes that block's parameters zero-length and this target the one
             # these four heads have always fitted.
             **pi_block(len(live))},
            chains=self.chains, warmup=self.warmup, samples=self.samples,
            seed=self.seed, label=self.name, metric=self.metric,
            # `rho` is a vector[n_rho] in the Stan source, so its init is a list even
            # when the vector has one entry.
            inits={"alpha": float(np.log(share / (1 - share))),
                   "beta": np.zeros(X.shape[1]).tolist(), "rho": [0.01]})
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

    def p_draws(self, df: pd.DataFrame, keep: int) -> tuple[np.ndarray, np.ndarray]:
        idx = thin(len(self.alpha_draws), keep)
        # (rows x K) @ (K x draws) -> (rows x draws), transposed to draws-major.
        eta = (self._design(df) @ self.beta_draws[idx].T
               + self.alpha_draws[idx][None, :]
               + self.year.shift(idx)[None, :]).T
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

def _iters(cfg_stan: dict) -> dict:
    """Full-length chains for every fit.

    There used to be a `fast` arm at half the iterations for the selection side, paid for
    by the test side running long. With the test side gone there is nothing to trade
    against — and the split budget was a second difference sitting inside a comparison that
    was being read as a replication check.
    """
    return {"warmup": int(cfg_stan.get("warmup", 1000)),
            "samples": int(cfg_stan.get("samples", 1000))}


def _metric_columns(scored: dict) -> dict:
    """Every metric from `score_count` / `score_conversion`, prefixed `val_`.

    Prefixed even though there is only one split now, because these column names are read
    by `season_terms.selected_specs`, `substitution_sweep` and the docs auditor, and a bare
    `r2` would be indistinguishable from the `test_r2` this artifact used to carry.
    """
    return {f"val_{k}": v for k, v in scored.items()}


def merge_heads(existing: pd.DataFrame | None, fresh: pd.DataFrame,
                key: str = "head") -> pd.DataFrame:
    """`fresh` replacing `existing`'s rows for the heads it covers, others kept.

    What makes `--heads` honest rather than a hand-patch. The eleven heads are fitted
    **separately** — that is the factorization identity this whole module rests on — so a
    head's rows depend on nothing outside itself, and refitting one leaves the other ten
    bit-identical at a fixed seed. Re-deriving them anyway costs ~2.7 h to reproduce numbers
    that already exist.

    `posteriors.py --groups` is the standing precedent and states the same rationale: it
    re-reads the manifest at every flush so two partial runs merge instead of the second
    clobbering the first. This is that, one target over.

    ⚠️ The safety property is that **selection is head-local**. `_finalize` picks `selected`
    and `beats_floor` within a head's own block, so a merged file cannot have a stale winner
    from a comparison that spanned heads. If a cross-head selection is ever added, this
    function stops being safe and the full sweep becomes mandatory again.
    """
    if existing is None or existing.empty:
        return fresh
    kept = existing[~existing[key].isin(set(fresh[key]))]
    return pd.concat([kept, fresh], ignore_index=True)


def sweep_counts(train_covered, val, cfg_stan, n_knots,
                 preseason: bool | None = None,
                 only: tuple[str, ...] | None = None,
                 train_full: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, list[dict]]:
    """Every count head x variant, on the VALIDATION split only.

    Halves the fit count and doubles what each fit is worth: the old sweep ran each variant
    twice, once short against `val` and once long against `test`, and only the first of
    those was ever allowed to decide anything.
    """
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics = [], []

    for component in COUNT_HEADS:
        if only is not None and component not in only:
            continue
        # The window is a per-head property — see `head_fitting_rows`.
        train = head_fitting_rows(train_full if train_full is not None else train_covered,
                                  train_covered, component, preseason)
        floor_val = count_floor(train, val, component, seed)
        rows.append({"head": component, "kind": "count", "variant": "carry_forward",
                     "n_features": 0, **_metric_columns(floor_val)})

        arms = count_variants(train, val, component, n_knots)
        for label, (v_tr, v_te, v_base) in arms.items():
            # The block is common to every arm, so the sweep still answers "which curvature
            # on the prior-rate term" rather than crossing two axes on one selection split.
            v_features = head_features(v_base, component, preseason)
            v_model = StanCount(v_features, component, name=f"{component}/{label}/val",
                                chains=chains, seed=seed,
                                **_iters(cfg_stan)).fit(v_tr)
            diagnostics.append(v_model.diagnostics)
            v_y = v_te[component].to_numpy(float)
            v = score_count(v_y, v_model.predict_mean(v_te),
                            v_model.predict_samples(v_te, seed), v_model.phi, seed)
            rows.append({"head": component, "kind": "count", "variant": label,
                         "n_features": len(v_features), **_metric_columns(v)})

        rows, diagnostics = _control_arm(
            rows, diagnostics, arms, component, None, cfg_stan, preseason,
            lambda f, te, comp=component: _score_count_arm(f, te, comp, seed))
    return _finalize(pd.DataFrame(rows), "val_r2", higher_is_better=True), diagnostics


def _score_count_arm(model: "StanCount", frame: pd.DataFrame, component: str,
                     seed: int) -> dict:
    y = frame[component].to_numpy(float)
    return score_count(y, model.predict_mean(frame),
                       model.predict_samples(frame, seed), model.phi, seed)


#: The variant each family's no-preseason control is taken at. `None` means "whichever the
#: sweep selected", resolved after the fact — the control has to sit at the arm that ships or
#: it is comparing two changes at once.
CONTROL_SUFFIX = "__no_preseason"


def _control_arm(rows: list[dict], diagnostics: list[dict], arms: dict,
                 component: str, attempted: str | None, cfg_stan: dict,
                 preseason: bool | None, score_fn) -> tuple[list[dict], list[dict]]:
    """The same variant on the same rows with the preseason block REMOVED.

    `docs/preseason-plan.md` session 6b, "what it does not settle": the point MLE that
    measured the block collapses the posterior over `beta` to its mode, so a Stan fit owes an
    answer to whether the increment survives integrating over coefficient uncertainty. Every
    arm above carries the block, so the sweep alone cannot say — it compares curvatures, not
    the block.

    Isolated the way `stan_minutes.sweep` isolates its own: the fitting window is the covered
    one either way, so the comparison is the five columns and nothing else. It is a control
    and never a candidate — `_finalize` excludes it from `selected` — because "ship the arm
    without the block" is a decision taken on 6b's evidence rather than one this run is
    powered to revisit.
    """
    # No block on this head means nothing for a control to isolate — an excluded head's
    # "control" would be a duplicate fit of the arm that ships, at full sampler cost.
    if not head_preseason_cols(component, preseason):
        return rows, diagnostics
    label = head_label(component, attempted)
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    # Taken at the arm this family's sweep just selected, so the control and the shipped head
    # differ by the block alone.
    kind = "count" if attempted is None else "conversion"
    fitted = [r for r in rows if r["head"] == label and r["variant"] != "carry_forward"]
    key = "val_r2" if kind == "count" else "val_nll"
    best = (max(fitted, key=lambda r: r[key]) if kind == "count"
            else min(fitted, key=lambda r: r[key]))["variant"]
    c_tr, c_te, c_base = arms[best]
    name = f"{best}{CONTROL_SUFFIX}"
    if attempted is None:
        model = StanCount(list(c_base), component, name=f"{label}/{name}/val",
                          chains=chains, seed=seed, **_iters(cfg_stan)).fit(c_tr)
    else:
        model = StanConversion(list(c_base), component, attempted,
                               name=f"{label}/{name}/val", chains=chains, seed=seed,
                               **_iters(cfg_stan)).fit(c_tr)
    diagnostics.append(model.diagnostics)
    rows.append({"head": label, "kind": kind, "variant": name,
                 "n_features": len(c_base), **_metric_columns(score_fn(model, c_te))})
    return rows, diagnostics


def sweep_conversions(train_covered, val, cfg_stan, n_knots,
                      preseason: bool | None = None,
                      only: tuple[str, ...] | None = None,
                      train_full: pd.DataFrame | None = None
                      ) -> tuple[pd.DataFrame, list[dict]]:
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics = [], []

    for made, attempted in CONVERSION_HEADS:
        head = f"{made}|{attempted}"
        if only is not None and head not in only:
            continue
        train = head_fitting_rows(train_full if train_full is not None else train_covered,
                                  train_covered, made, preseason)
        floor_val = conversion_floor(train, val, made, attempted, seed)
        rows.append({"head": head, "kind": "conversion", "variant": "carry_forward",
                     "n_features": 0, **_metric_columns(floor_val)})

        arms = conversion_variants(train, val, made, attempted, n_knots)
        for label, (v_tr, v_te, v_base) in arms.items():
            v_features = head_features(v_base, made, preseason)
            v_model = StanConversion(v_features, made, attempted,
                                     name=f"{head}/{label}/val", chains=chains,
                                     seed=seed, **_iters(cfg_stan)).fit(v_tr)
            diagnostics.append(v_model.diagnostics)
            v = _score_conv(v_model, v_te, made, attempted, seed)
            rows.append({"head": head, "kind": "conversion", "variant": label,
                         "n_features": len(v_features), **_metric_columns(v)})

        rows, diagnostics = _control_arm(
            rows, diagnostics, arms, made, attempted, cfg_stan, preseason,
            lambda f, te, m=made, a=attempted: _score_conv(f, te, m, a, seed))
    return _finalize(pd.DataFrame(rows), "val_nll", higher_is_better=False), diagnostics


def _score_conv(model: StanConversion, frame: pd.DataFrame, made: str, attempted: str,
                seed: int) -> dict:
    ok = frame[attempted].to_numpy(float) > 0
    live = frame[ok]
    n = np.rint(live[attempted].to_numpy(float)).astype(int)
    y = np.minimum(np.rint(live[made].to_numpy(float)).astype(int), n)
    return score_conversion(y.astype(float), n.astype(float), model.predict_p(live),
                            model.predict_samples(live, seed), model.rho, seed)


def _finalize(table: pd.DataFrame, val_col: str,
              higher_is_better: bool) -> pd.DataFrame:
    """Mark the validation-selected variant per head, and whether it clears the floor.

    Two separate flags on purpose. `selected` answers "which variant would I ship";
    `beats_floor` answers "is this a model at all". A head can be selected and still fail
    the floor — `ftm|fta` is the known case, and reporting it as a win would be the error
    this column exists to prevent.

    **Both now read the same column, and that is the change worth noticing.** `beats_floor`
    used to be a *held-out* fact while `selected` was a validation one, so the artifact
    carried a head that had been chosen on one split and certified on another. Since the
    test side is no longer scored here, the floor comparison is the honest one it should
    always have been: does the variant I would ship beat arithmetic on the rows I chose it
    with? The end-of-project certification is `src/final_evaluation.py`'s job.
    """
    # A `--heads` run that names no head of this kind sweeps nothing, and an empty frame has
    # no columns to flag. Returned with the schema the caller expects so `pd.concat` and
    # `merge_heads` downstream see a well-formed zero-row block rather than a shapeless one.
    if table.empty:
        return pd.DataFrame(columns=["head", "kind", "variant", "n_features", val_col,
                                     "selected", "beats_floor", "is_control"])
    out = table.copy()
    out["selected"] = False
    out["beats_floor"] = False
    # The no-preseason control is scored like any other row and can never be selected: it is
    # the thing the shipped arm is measured AGAINST, and letting it win would silently
    # re-decide the block on a run that exists to price it. `stan_minutes.sweep` marks its own
    # the same way.
    out["is_control"] = out["variant"].str.endswith(CONTROL_SUFFIX)
    for head, block in out.groupby("head"):
        fitted = block[(block["variant"] != "carry_forward") & ~block["is_control"]]
        if fitted.empty:
            continue
        best = (fitted[val_col].idxmax() if higher_is_better else fitted[val_col].idxmin())
        out.loc[best, "selected"] = True
        floor = block[block["variant"] == "carry_forward"][val_col].iloc[0]
        better = (block[val_col] > floor) if higher_is_better else (block[val_col] < floor)
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
    # The raw lag-1 counts too, because the shrunk conversion floor for `fg3a | fga`
    # reads `{made}_lag1` / `{attempted}_lag1` rather than per-36 rates.
    out["fga_lag1"] = (out["fg2a_lag1"].to_numpy(float)
                       + out["fg3a_lag1"].to_numpy(float))
    prior = (out["fg2a_p36_lag1"].to_numpy(float) + out["fg3a_p36_lag1"].to_numpy(float))
    out["fga_p36_lag1"] = prior
    with np.errstate(invalid="ignore", divide="ignore"):
        out["fg3a_share_lag1"] = np.where(prior > 0,
                                          out["fg3a_p36_lag1"].to_numpy(float) / prior,
                                          np.nan)
    return out


def substitution_arm(train, val, cfg_stan, n_knots
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
    joint NLL over the scored rows is therefore an apples-to-apples number and not a
    scale-mismatched one.

    Validation only since 2026-08-05. The `for split in {...}` loop is kept with one entry
    rather than unrolled, because `split` is a column of the artifact and `_g0_select`
    keys on it — the shape stays right if a second frame is ever legitimately added.
    """
    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    rows, diagnostics = [], []

    for split, (tr, te) in {"val": (train, val)}.items():
        tr, te = add_substitution_columns(tr), add_substitution_columns(te)

        # Arm A — the canonical pair, two independent count heads.
        nll_a = np.zeros(len(te))
        for component in ("fg2a", "fg3a"):
            variants = count_variants(tr, te, component, n_knots)
            v_tr, v_te, features = variants["log_own"]
            model = StanCount(features, component, name=f"subst/{component}/{split}",
                              chains=chains, seed=seed,
                              **_iters(cfg_stan)).fit(v_tr)
            diagnostics.append(model.diagnostics)
            nll_a += nb_nll(v_te[component].to_numpy(float),
                            np.clip(model.predict_mean(v_te), 1e-6, None), model.phi)

        # Arm B — total attempts as a count, three-point share as a binomial.
        variants = count_variants(tr, te, "fga", n_knots)
        v_tr, v_te, features = variants["log_own"]
        fga_model = StanCount(features, "fga", name=f"subst/fga/{split}", chains=chains,
                              seed=seed, **_iters(cfg_stan)).fit(v_tr)
        diagnostics.append(fga_model.diagnostics)
        nll_b = nb_nll(v_te["fga"].to_numpy(float),
                       np.clip(fga_model.predict_mean(v_te), 1e-6, None), fga_model.phi)

        # Exactly `conversion_variants(...)["logit_own"]` with the own-rate column named
        # explicitly. It has to be named: the `{made}_pct_lag1` convention would ask for
        # `fg3a_pct_lag1`, which is three-point *shooting* percentage and not the
        # attempt-*mix* share this head is about. `substitution_sweep` pins that the
        # refactor reproduces this arm's recorded NLL to full precision.
        share_variants = conversion_variants(tr, te, "fg3a", "fga", n_knots,
                                             own="fg3a_share_lag1")
        s_tr, s_te, share_features = share_variants["logit_own"]
        share_model = StanConversion(share_features, "fg3a", "fga",
                                     name=f"subst/fg3a|fga/{split}", chains=chains,
                                     seed=seed, **_iters(cfg_stan)).fit(s_tr)
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


# ── Gate 0: the substitution comparison, un-handicapped ───────────────────────

SUBSTITUTION_COUNT_VARIANTS = ("linear", "log_own", "log_own_spline")
SUBSTITUTION_SHARE_VARIANTS = ("linear", "logit_own", "logit_own_spline")
SUBSTITUTION_PAIR = ("fg2a", "fg3a")

# Gate 0 measured the RETIRED two-count basis against the one now shipped, so arm A's
# heads are no longer in `stan_component_metrics.csv` and `selected_specs` cannot supply
# their variants. These are what that artifact selected *before* adoption — a fact about
# the retired basis, pinned here so the gate stays re-runnable rather than silently
# falling back to `log_own`, which is the exact handicap the gate exists to remove.
# The best-of-16 grid has no such fallback: it is read from the pre-adoption metrics file
# and, once that is refitted, survives only inside the gate's own artifact.
LEGACY_ARM_A_SPECS = {"fg2a": "log_own", "fg3a": "log_own_spline"}


def _share_nll(model: "StanConversion", frame: pd.DataFrame, made: str,
               attempted: str) -> np.ndarray:
    """Per-row beta-binomial NLL, zero where there were no attempts to convert."""
    n = np.rint(frame[attempted].to_numpy(float)).astype(int)
    y = np.minimum(np.rint(frame[made].to_numpy(float)).astype(int), n)
    live = n > 0
    out = np.zeros(len(frame))
    p = model.predict_p(frame[live])
    out[live] = -_beta_binomial_logpmf(y[live], n[live], p, model.rho)
    return out


def substitution_sweep(train, val, cfg_stan, n_knots,
                       predictions_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Gate 0 — the same comparison as `substitution_arm`, with both arms un-handicapped.

    `substitution_arm` fits **every** head at `log_own`. That is a straw man for arm A:
    `fg3a` selects `log_own_spline`, and at `log_own` it reads held-out R² 0.3719 with
    `beats_floor = False` against 0.9046 for the spline it actually ships. 0.3056 of the
    recorded 0.7927-nat margin is that handicap alone.

    So here arm A is fitted at **each head's own validation-selected variant**, read from
    the artifact that selected it (`season_terms.selected_specs`) rather than re-derived,
    and arm B is **swept for real** instead of being pinned at one variant. Additive
    separability is what makes the sweep cheap: the joint density factors as
    `p(fga) · p(fg3a | fga)`, and the two factors share no parameters, so they select
    **independently** — 3 + 3 fits per split, not 9 combinations.

    **Validation only since 2026-08-05, re-run 2026-08-06.** The gate was already selecting
    on validation and reporting test as confirmation; the test half is now simply not run,
    which halves the fits and removes the column a reader could mistake for a replication.
    It also removes a real hazard specific to this gate, and that one is measured rather
    than hypothetical: `fg3a|fga @ logit_own` **clears** its no-fit floor on test (4.611402
    against 4.652797) and **fails** it on validation (4.635963 against 4.619109). A reader
    taking the test column would have shipped a variant that loses to arithmetic on the
    split that selects. Only `logit_own_spline` clears on both.

    The comparison stays legitimate for the same reason it always was: `(fg2a, fg3a)` and
    `(fga, fg3a)` are the same point in different coordinates, a bijection with unit
    Jacobian on the integers, so the two joint log-densities are directly comparable.
    """
    from src.models.season_terms import selected_specs

    seed = int(cfg_stan.get("seed", 42))
    chains = int(cfg_stan.get("chains", 4))
    specs, _ = selected_specs(Path(predictions_dir))
    specs = {**LEGACY_ARM_A_SPECS, **{k: v for k, v in specs.items()
                                      if k in LEGACY_ARM_A_SPECS}}
    rows, diagnostics = [], []

    print("\nGate 0 — the substitution comparison at each head's SELECTED variant")
    print(f"  arm A specs read from stan_component_metrics.csv: "
          f"{', '.join(f'{h}@{specs[h]}' for h in SUBSTITUTION_PAIR)}")

    for split, (tr, te) in {"val": (train, val)}.items():
        tr, te = add_substitution_columns(tr), add_substitution_columns(te)
        per_head: dict[tuple[str, str], np.ndarray] = {}

        # ── Arm A: the canonical pair, each at its own selected variant ────────
        for component in SUBSTITUTION_PAIR:
            spec = specs[component]
            variants = count_variants(tr, te, component, n_knots)
            v_tr, v_te, features = variants[spec]
            model = StanCount(features, component,
                              name=f"gate0/{component}/{spec}/{split}",
                              chains=chains, seed=seed,
                              **_iters(cfg_stan)).fit(v_tr)
            diagnostics.append(model.diagnostics)
            nll = nb_nll(v_te[component].to_numpy(float),
                         np.clip(model.predict_mean(v_te), 1e-6, None), model.phi)
            per_head[("two_counts", component)] = nll
            rows.append(_g0_row(split, "two_counts", component, spec, len(features),
                                nll, count_floor(tr, te, component, seed)["nll"],
                                selected=True))

        # ── Arm B, factor 1: total attempts as a count ─────────────────────────
        fga_floor = count_floor(tr, te, "fga", seed)["nll"]
        fga_variants = count_variants(tr, te, "fga", n_knots)
        for label in SUBSTITUTION_COUNT_VARIANTS:
            v_tr, v_te, features = fga_variants[label]
            model = StanCount(features, "fga", name=f"gate0/fga/{label}/{split}",
                              chains=chains, seed=seed,
                              **_iters(cfg_stan)).fit(v_tr)
            diagnostics.append(model.diagnostics)
            nll = nb_nll(v_te["fga"].to_numpy(float),
                         np.clip(model.predict_mean(v_te), 1e-6, None), model.phi)
            per_head[("fga_x_fg3a_share", f"fga/{label}")] = nll
            rows.append(_g0_row(split, "fga_x_fg3a_share", "fga", label, len(features),
                                nll, fga_floor))

        # ── Arm B, factor 2: the three-point share on `fga` trials ─────────────
        # `own=` is mandatory here — the `{made}_pct_lag1` convention would silently ask
        # for shooting percentage instead of the attempt mix. See `conversion_variants`.
        share_floor = conversion_floor(tr, te, "fg3a", "fga", seed)["nll"]
        share_variants = conversion_variants(tr, te, "fg3a", "fga", n_knots,
                                             own="fg3a_share_lag1")
        for label in SUBSTITUTION_SHARE_VARIANTS:
            s_tr, s_te, features = share_variants[label]
            model = StanConversion(features, "fg3a", "fga",
                                   name=f"gate0/fg3a|fga/{label}/{split}",
                                   chains=chains, seed=seed,
                                   **_iters(cfg_stan)).fit(s_tr)
            diagnostics.append(model.diagnostics)
            nll = _share_nll(model, s_te, "fg3a", "fga")
            per_head[("fga_x_fg3a_share", f"fg3a|fga/{label}")] = nll
            rows.append(_g0_row(split, "fga_x_fg3a_share", "fg3a|fga", label,
                                len(features), nll, share_floor))

        rows.append({"analysis": "joint", "split": split, "arm": "two_counts",
                     "head": "fg2a+fg3a", "variant": "selected", "n": len(te),
                     "mean_nll": float(sum(per_head[("two_counts", c)]
                                           for c in SUBSTITUTION_PAIR).mean())})
        for fga_label in SUBSTITUTION_COUNT_VARIANTS:
            for share_label in SUBSTITUTION_SHARE_VARIANTS:
                joint = (per_head[("fga_x_fg3a_share", f"fga/{fga_label}")]
                         + per_head[("fga_x_fg3a_share", f"fg3a|fga/{share_label}")])
                rows.append({"analysis": "joint", "split": split,
                             "arm": "fga_x_fg3a_share", "head": "fga+fg3a|fga",
                             "variant": f"{fga_label}+{share_label}", "n": len(te),
                             "mean_nll": float(joint.mean())})

    out = _g0_select(pd.DataFrame(rows))
    rows_grid = _arm_a_grid(Path(predictions_dir))
    return pd.concat([out, rows_grid], ignore_index=True), diagnostics


def _g0_row(split: str, arm: str, head: str, variant: str, n_features: int,
            nll: np.ndarray, floor_nll: float, selected: bool | None = None) -> dict:
    return {"analysis": "head", "split": split, "arm": arm, "head": head,
            "variant": variant, "n_features": n_features, "n": len(nll),
            "mean_nll": float(nll.mean()), "floor_nll": float(floor_nll),
            "beats_floor": bool(nll.mean() < floor_nll),
            "selected": selected}


def _g0_select(table: pd.DataFrame) -> pd.DataFrame:
    """Validation picks each arm-B factor independently, and the joint row follows.

    Independent selection is not a shortcut — the two factors share no parameters, so
    the joint NLL is a *sum* and minimising it is exactly minimising each term.
    """
    out = table.copy()
    val = out[(out["analysis"] == "head") & (out["split"] == "val")
              & (out["arm"] == "fga_x_fg3a_share")]
    chosen = {str(head): str(block.loc[block["mean_nll"].idxmin(), "variant"])
              for head, block in val.groupby("head")}
    for head, variant in chosen.items():
        picked = ((out["analysis"] == "head") & (out["arm"] == "fga_x_fg3a_share")
                  & (out["head"] == head) & (out["variant"] == variant))
        out.loc[picked, "selected"] = True
    out.loc[(out["analysis"] == "head") & out["selected"].isna(), "selected"] = False

    combo = f"{chosen.get('fga', '')}+{chosen.get('fg3a|fga', '')}"
    is_joint = out["analysis"] == "joint"
    out.loc[is_joint, "selected"] = (
        ((out.loc[is_joint, "arm"] == "two_counts")
         | (out.loc[is_joint, "variant"] == combo)))

    sel = out[is_joint & out["selected"]].set_index(["split", "arm"])["mean_nll"]
    for split in out.loc[is_joint, "split"].unique():
        margin = float(sel[(split, "fga_x_fg3a_share")] - sel[(split, "two_counts")])
        out.loc[is_joint & (out["split"] == split), "reparam_minus_canonical"] = margin
    return out


def _arm_a_grid(predictions_dir: Path) -> pd.DataFrame:
    """Arm A's best-of-16, straight from the shipped artifact — zero extra fits.

    Sixteen (fg2a variant x fg3a variant) combinations, summed. The point is the
    *minimum*: it turns "we removed the handicap" into "arm B wins even against arm A's
    most favourable configuration", which is the version that cannot be argued with.
    Missing artifact returns empty, matching the module's skip-don't-fail convention.

    **It returns empty now, by two independent routes, and neither is a defect.** Adoption
    removed `fg2a` and `fg3a` from `stan_component_metrics.csv`, so there are no rows to
    grid over; the held-out conversion removed `test_nll`, so there is no column to grid on
    either. Both guards are below and both fire.

    **The recorded best-of-16 is therefore gone from every artifact**, including this
    gate's own, which was rebuilt validation-only on 2026-08-06. It is preserved as a
    presence-checked figure in `docs/shot-attempt-basis-plan.md` and nothing verifies it.
    That is an acceptable loss and the reason is worth stating: the grid minimum
    (`log_own_spline + log_own_spline`, 10.476413) beat arm A's *own selected* pair
    (10.478052) by **0.001640 nats**, so "arm B wins even against arm A's most favourable
    configuration" and "arm B wins against arm A as it would actually be fitted" were never
    materially different claims, and the gate still measures the second one.
    """
    path = Path(predictions_dir) / "stan_component_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if "test_nll" not in table.columns:
        return pd.DataFrame()
    per = {c: table[table["head"] == c].set_index("variant")["test_nll"].to_dict()
           for c in SUBSTITUTION_PAIR}
    if not all(per.values()):
        return pd.DataFrame()
    rows = [{"analysis": "arm_a_grid", "split": "test", "arm": "two_counts",
             "head": "fg2a+fg3a", "variant": f"{a}+{b}",
             "mean_nll": float(x + y)}
            for a, x in per["fg2a"].items() for b, y in per["fg3a"].items()]
    out = pd.DataFrame(rows)
    out["selected"] = out["mean_nll"] == out["mean_nll"].min()
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict, heads: tuple[str, ...] | None = None) -> dict[str, Path]:
    """The full sweep, or `heads` alone merged into the artifact already on disk.

    `heads=None` is the whole family and is what `make stan-components` runs. A subset is the
    `posteriors --groups` idiom: the eleven heads are fitted separately, so refitting one
    leaves the other ten bit-identical at a fixed seed and re-deriving them costs ~2.7 h of
    sampler time to reproduce numbers already on disk. `merge_heads` states the property that
    makes it sound.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_stan = cfg.get("stan", {})
    n_knots = int(cfg_stan.get("components", {}).get("spline_knots", SPLINE_KNOTS))
    test_seasons = 2

    preseason = bool(cfg_stan.get("components", {}).get("preseason", PRESEASON))

    design = head_design(cfg, preseason)
    train_full, val = selection_split(design, test_seasons)
    # The covered-window cut, on the FITTING rows only. Required by the block rather than
    # chosen for its own sake — see `covered_fitting_rows`.
    train = covered_fitting_rows(train_full, cfg, preseason)

    print(f"Stan component heads: {len(design):,} player-seasons, "
          f"{design['season'].nunique()} target seasons")
    print(f"  The test split is LOCKED — this sweep fits and scores VALIDATION only\n"
          f"  (src/models/held_out.py). The held-out reading is taken once, by "
          f"`make final-evaluation`.")
    print(f"  {len(train):,} fit / {len(val):,} select "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")
    if preseason:
        constants = shrinkage_constants(cfg)
        carried = [h for h in sorted(constants) if head_preseason_cols(h.split("|")[0])]
        print(f"  PRESEASON BLOCK ON (session 6b) for {len(carried)} of "
              f"{len(constants)} heads: five columns each — the head's\n  own delta, "
              f"volume-shrunk at its fitted k, plus the four age-split missing indicators."
              f"\n  k = "
              + ", ".join(f"{h} {constants[h]:.0f}" for h in carried) + ".")
        if PRESEASON_EXCLUDE:
            print(f"  OPTED OUT: {', '.join(sorted(PRESEASON_EXCLUDE))} — measured WORSE "
                  f"with the block on three metrics at the\n  posterior, and P1's own "
                  f"attribution said its gain was the indicator rather than\n  preseason "
                  f"3P%. It fits the pre-2026-08-15 head exactly.")
        print(f"  Fitting rows cut to the covered window: {len(train):,} of "
              f"{len(train_full):,} ({len(train) / len(train_full):.1%}). Each head also "
              f"fits a\n  `{CONTROL_SUFFIX.lstrip('_')}` control at its selected variant, "
              f"which is what says whether the\n  increment survives integrating over "
              f"`beta`. `stan.components.preseason: false` is the\n  exact rollback.")
    else:
        print("  PRESEASON BLOCK OFF — the pre-2026-08-15 heads exactly.")
    print(f"  {len(COUNT_HEADS)} count heads + {len(CONVERSION_HEADS)} conversion heads, "
          f"fitted SEPARATELY — the chain factorizes the joint posterior exactly.\n")

    counts, diag_counts = sweep_counts(train, val, cfg_stan, n_knots, preseason, heads,
                                       train_full)
    print("Count heads — validation R^2 on the season total (higher is better):")
    print(counts.pivot_table(index="head", columns="variant", values="val_r2")
          .reindex(columns=["carry_forward", "linear", "log_own", "log_own_spline"])
          .round(4).to_string())

    conversions, diag_conv = sweep_conversions(train, val, cfg_stan, n_knots, preseason,
                                               heads, train_full)
    print("\nConversion heads — validation beta-binomial NLL per row (lower is better):")
    print(conversions.pivot_table(index="head", columns="variant", values="val_nll")
          .reindex(columns=["carry_forward", "linear", "logit_own", "logit_own_spline"])
          .round(4).to_string())

    table = pd.concat([counts, conversions], ignore_index=True)
    chosen = table[table["selected"]]
    print("\nValidation-selected variant per head, against its no-fit floor:")
    for _, row in chosen.iterrows():
        floor = table[(table["head"] == row["head"])
                      & (table["variant"] == "carry_forward")].iloc[0]
        metric, better = ("val_r2", row["val_r2"] - floor["val_r2"]) \
            if row["kind"] == "count" else ("val_nll", floor["val_nll"] - row["val_nll"])
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

    if heads is not None:
        # The substitution arm is a basis comparison over `fga`/`fg3a`, not a per-head sweep,
        # so a targeted refit neither changes it nor may silently blank it.
        subst_path = out_dir / "stan_component_substitution.csv"
        substitution = (pd.read_csv(subst_path) if subst_path.exists()
                        else pd.DataFrame())
        diag_subst = []
        print(f"\n3PA/2PA substitution: not re-run (targeted refit); "
              f"{len(substitution):,} rows kept from disk.")
    else:
        substitution, diag_subst = substitution_arm(train, val, cfg_stan, n_knots)
    print("\n3PA/2PA substitution — two count heads vs `fga` count x `fg3a | fga` share:")
    print(substitution.round(4).to_string(index=False))
    print("  Comparable because the map is a bijection with unit Jacobian on the integers:\n"
          "  (fg2a, fg3a) and (fga, fg3a) are the same point in different coordinates.\n"
          "  Negative `reparam_minus_canonical` favours the reparameterization; the\n"
          "  validation row is the one that decides.")

    diag = diagnostics_frame(diag_counts + diag_conv + diag_subst)
    if heads is not None:
        # Merge into what is on disk rather than replacing it. Diagnostics are keyed by a
        # `<head>/<variant>/val` label, so the refitted heads are identified by prefix.
        m_path, d_path = (out_dir / "stan_component_metrics.csv",
                          out_dir / "stan_component_diagnostics.csv")
        table = merge_heads(pd.read_csv(m_path) if m_path.exists() else None, table)
        if d_path.exists():
            old = pd.read_csv(d_path)
            stale = old["label"].str.split("/").str[0].isin(set(heads))
            diag = pd.concat([old[~stale], diag], ignore_index=True)
        print(f"\nTargeted refit of {', '.join(heads)} — merged into the artifact on disk; "
              f"the other\nheads' rows are untouched and were never re-derived.")
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


def run_substitution_sweep(cfg: dict) -> dict[str, Path]:
    """Gate 0 on its own, writing its own artifact — `make stan-substitution`.

    Deliberately a separate entry point from `run`. `substitution_arm` is called from
    inside `run`, which writes all three component CSVs together, so refreshing the
    substitution comparison through it would cost that target's full sweep and would also
    rewrite `stan_component_metrics.csv` — the artifact this sweep *reads* arm A's selected
    specs from. Eight fits against thirty-seven.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_stan = cfg.get("stan", {})
    n_knots = int(cfg_stan.get("components", {}).get("spline_knots", SPLINE_KNOTS))

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    train, val = selection_split(design, 2)
    print(f"Gate 0 — {len(design):,} player-seasons, {len(train):,} fit / "
          f"{len(val):,} select ({', '.join(sorted(val['season'].unique()))} as "
          f"validation).\n  The test split is LOCKED (src/models/held_out.py).")

    table, diagnostics = substitution_sweep(train, val, cfg_stan, n_knots, out_dir)
    heads = table[table["analysis"] == "head"]
    print("\nPer-factor NLL (lower is better; selection reads validation only):")
    print(heads[["split", "arm", "head", "variant", "n_features", "mean_nll",
                 "floor_nll", "beats_floor", "selected"]]
          .round(6).to_string(index=False))
    joint = table[table["analysis"] == "joint"]
    print("\nJoint NLL per player-season — the bijection makes these comparable:")
    print(joint[joint["selected"]][["split", "arm", "variant", "mean_nll",
                                    "reparam_minus_canonical"]]
          .round(6).to_string(index=False))
    grid = table[table["analysis"] == "arm_a_grid"]
    if len(grid):
        best = grid.loc[grid["mean_nll"].idxmin()]
        print(f"\nArm A best-of-16 (from stan_component_metrics.csv, no new fits): "
              f"{best['variant']} at {best['mean_nll']:.6f}")

    diag = diagnostics_frame(diagnostics)
    paths = {}
    for name, frame in (("substitution_sweep", table),
                        ("substitution_sweep_diagnostics", diag)):
        dest = out_dir / f"stan_component_{name}.csv"
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")
    print(f"\nSampler over {len(diag)} fits: max R-hat {diag['max_rhat'].max():.4f}, "
          f"{int(diag['divergences'].sum())} divergences, "
          f"{diag['wall_clock_s'].sum() / 60:.1f} min total")
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit the component rate heads.")
    parser.add_argument("--gate0", action="store_true",
                        help="run the 3PA/2PA substitution sweep instead")
    parser.add_argument("--heads", default=None,
                        help="comma-separated subset to refit (e.g. 'fg3m|fg3a'); every "
                             "other head keeps its rows from the artifact on disk. The "
                             "heads are fitted separately, so a subset refit is exact "
                             "rather than approximate — see `merge_heads`. Omit for the "
                             "full sweep.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    if args.gate0:
        run_substitution_sweep(cfg)
    else:
        subset = tuple(h.strip() for h in args.heads.split(",")) if args.heads else None
        if subset:
            known = set(COUNT_HEADS) | {f"{m}|{a}" for m, a in CONVERSION_HEADS}
            unknown = [h for h in subset if h not in known]
            if unknown:
                raise SystemExit(f"unknown head(s): {', '.join(unknown)}; "
                                 f"choose from {sorted(known)}")
        run(cfg, subset)
