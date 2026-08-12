// Beta-binomial GLM: y successes out of n trials, logit-linear mean, and a dispersion
// rho that is either shared or graded over bins supplied as data.
//
// THREE of this project's heads are this same likelihood with different data, and that
// is the factorization argument made concrete rather than asserted:
//
//   availability   y = games played      n = team games            (docs/availability-plan.md)
//   minutes        y = season minutes    n = summed game length over games played
//   conversion     y = makes             n = attempts              (fg2m|fg2a, fg3m|fg3a, ftm|fta)
//
// The generative chain is availability -> min | available -> counts | min -> makes |
// attempts. With distinct parameter blocks and independent priors the joint posterior
// factorizes exactly, so fitting these separately recovers the identical posterior a
// joint model would. One file, fitted once per head.
//
// Parameterization: mean mu and intra-class correlation rho, so that
//   a = mu*(1-rho)/rho,  b = (1-mu)*(1-rho)/rho,  var(y) = n*mu*(1-mu)*[1 + (n-1)*rho].
// rho IS the overdispersion the availability profile measures: at n = 82 a 20x variance
// ratio is rho ~ 0.235.
//
// `rho` carries no prior statement, which in Stan means uniform on its constrained
// (0,1) support. That is deliberate: it makes the posterior mode exactly the penalized
// MLE that `src/models/availability.py::BetaBinomialGLM` finds, so "the posterior mean
// reproduces the MLE" is a check with a defined answer rather than a vague expectation.
// For the same reason `beta_scale` is passed in rather than hard-coded — the caller sets
// it to 1/sqrt(2*l2) to match that head's L2 penalty exactly.
//
// ── The dispersion bins ──────────────────────────────────────────────────────────────
//
// `rho` is a VECTOR indexed by a bin supplied as data, transplanted from
// composition_glm.stan where the same device grades the allocation step's dispersion by
// prior minutes share. **n_rho = 1 with rho_bin all-ones is EXACTLY the shared-rho
// model** — one parameter, one value of `s` per row, the identical target — which is the
// same nesting discipline `S = 0` uses for the year effect above. A test pins it, because
// that nesting is what makes this an addition rather than a silent change to five heads
// that never asked for a graded dispersion.
//
// On availability the bins are prior-season MPG role buckets
// (`src/eda/season_effects.ROLE_EDGES`), where one rho for every player is measurably too
// tight for fringe players and too wide for stars — 0.3084 against 0.2456 fitted at the
// point MLE, a 1.26x spread on the full window and 1.47x on the shipped 2012-13 one. It
// is milder than the composition head's 2.07x and it is a consistent, nearly free win;
// see docs/availability-window-plan.md §4. Bins are a property of the ROW, known before
// the season starts, so this grades on information the head already has rather than on
// the target.
// ── The optional year-level random effect ────────────────────────────────────────────
//
// A season FIXED effect is unusable at prediction time — there is no dummy for a season
// that has not happened. A year-level RANDOM effect is usable, because at prediction time
// it contributes mean zero and its variance: it widens the predictive rather than shifting
// it. It is also the only form that can touch `yoy_sd_pct`, the irreducible year-to-year
// spread a trend term provably cannot reach (`make season-effects`).
//
// **S = 0 disables it EXACTLY.** `year_z` and `sigma_year` are then zero-length, so the
// parameter space, the priors and the likelihood are identical to the model without this
// block — which is what preserves this file's load-bearing property that with a
// normal(0, 1/sqrt(2*l2)) prior the posterior MODE is exactly the penalized MLE
// `src/models/availability.py::BetaBinomialGLM` finds. A test pins the nesting.
//
// ── The optional low-availability mixture ────────────────────────────────────────────
//
//   y ~ (1 - pi_i) * BetaBinom(n, mu_i, rho_i)  +  pi_i * BetaBinom(n, mu_low, rho_low)
//   pi_i = theta * inv_logit(z_i' gamma),  theta in [0, 1]
//
// The AVAILABILITY head only. `docs/availability-window-plan.md` §7c measured the low tail
// as a separate EVENT rather than an extreme draw of a per-game rate: an Achilles rupture
// in October is not a frailty, and one shape knob cannot make it one. Under
// `a = mu(1-rho)/rho` the frailty's shape and its variance are the same parameter, so
// `b < 1` — a density diverging exactly at "played every game" — is forced on 52.7% of
// rows, and moving `rho` is why the single-component head could only halve the miss. A
// second component with its OWN mean and dispersion halves the boundary error again
// (0.0201 -> 0.0109) and ties on CRPS.
//
// `theta` is a BOUNDED scale on [0, 1] rather than an intercept inside the logit, and that
// is the whole reason the nesting below is an equality instead of a limit: `theta = 0` is
// an attainable parameter value where `gamma_0 -> -inf` is a statement about floating
// point. `z` carries no intercept, because `theta` is the scale. `mu_low` is bounded at
// `mu_low_max` so the two components cannot label-switch — with a scalar mean against a
// covariate-driven one they are already distinguishable, and the bound makes it structural.
//
// **P = 0 disables the block EXACTLY, and so does theta = 0.** Two nestings, because two
// different things could go wrong. `P = 0` gives zero-length `theta`, `mu_low`, `rho_low`
// and `gamma`, so the parameter space, the priors and the likelihood are the five other
// heads' unchanged model — the same device `S = 0` uses. With `P > 0` the mixture enters as
// an ADDITIVE correction to the untouched beta-binomial line below, and at `pi = 0` that
// correction is exactly `0.0` per row rather than a small number: the likelihood statement
// those five heads have always fitted is still the one being evaluated, bit for bit. A test
// pins both, on Stan's own `log_prob`.
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;                     // standardized on TRAIN only; intercept is separate
  array[N] int<lower=0> n;            // trials
  array[N] int<lower=0> y;            // successes, y <= n on every row
  real<lower=0> beta_scale;           // 1/sqrt(2*l2) reproduces an L2 penalty of l2
  real<lower=0> intercept_scale;
  int<lower=1> n_rho;                 // 1 is the shared dispersion, exactly
  array[N] int<lower=1, upper=n_rho> rho_bin;   // bin edges from TRAIN rows only
  int<lower=0> S;                     // training seasons; 0 disables the year effect
  array[N] int<lower=0> season_idx;   // 1..S, ignored (and all zero) when S == 0
  real<lower=0> year_sd_scale;        // half-normal scale on sigma_year
  int<lower=0> P;                     // pi's covariates; 0 disables the mixture EXACTLY
  matrix[N, P] Z;                     // standardized on TRAIN only; NO intercept column
  real<lower=0> gamma_scale;          // normal scale on gamma, ignored when P == 0
  real<lower=0, upper=1> mu_low_max;  // upper bound on the low component's mean
}
transformed data {
  int H = S > 0 ? 1 : 0;
  int M = P > 0 ? 1 : 0;              // one mixture scalar each, or none at all
}
parameters {
  // Bounds mirror RHO_MIN / RHO_MAX in `src/models/availability.py` exactly, so this is
  // the same parameter space the MLE searches rather than a wider one. They are guard
  // rails, not a prior: at rho -> 1 the beta collapses to a two-point distribution on 0
  // and n, which is never a useful forecast and is numerically nasty (the shape
  // parameters both go to 0); at rho -> 0 they both diverge. Fitted values land near
  // 0.28 on availability and far lower elsewhere, so neither bound binds.
  // Length n_rho: one dispersion per bin, and n_rho = 1 is the shared-rho model exactly.
  vector<lower=1e-6, upper=0.95>[n_rho] rho;
  real alpha;
  vector[K] beta;
  vector[S] year_z;                   // zero-length when S == 0
  vector<lower=0>[H] sigma_year;      // zero-length when S == 0
  // All four zero-length when P == 0. `theta`'s lower bound of 0 is the nesting point and
  // is attainable, which is the property `docs/availability-window-plan.md` §7b requires;
  // `mu_low` and `rho_low` mirror the main component's own guard rails.
  vector<lower=0, upper=1>[M] theta;
  vector<lower=1e-6, upper=mu_low_max>[M] mu_low;
  vector<lower=1e-6, upper=0.95>[M] rho_low;
  vector[P] gamma;
}
model {
  // `eta` is local rather than a `transformed parameter`: it is one value per row, so
  // saving it would write N x draws numbers to disk (~40M on the availability head) for
  // a quantity every consumer recomputes from alpha/beta anyway.
  vector[N] eta = alpha + X * beta;
  // Per-row dispersion by bin, still one vectorized beta_binomial call below: multiple
  // indexing (`rho[rho_bin]`) gathers the right rho for every row at once. With n_rho = 1
  // every entry is the same number, which is the scalar `s` this line used to be.
  vector[N] s = (1 - rho[rho_bin]) ./ rho[rho_bin];

  alpha ~ normal(0, intercept_scale);
  beta ~ normal(0, beta_scale);
  if (S > 0) {
    // Non-centered: with ~28 seasons and a small sigma the centered form is a funnel.
    year_z ~ std_normal();
    sigma_year ~ normal(0, year_sd_scale);
    eta += sigma_year[1] * year_z[season_idx];
  }
  // `s * inv_logit(-eta)` rather than the algebraically identical `s * (1 - inv_logit(eta))`.
  // In double precision `inv_logit(eta)` saturates to exactly 1.0 by eta ~ 37, so the
  // subtraction yields a beta shape parameter of exactly 0 and the whole target is
  // rejected; `inv_logit(-eta)` stays positive until eta ~ 745. Same model, ~20 orders of
  // magnitude more headroom, and it removes essentially all of the warmup rejections that
  // a wide-open linear predictor otherwise produces.
  y ~ beta_binomial(n, s .* inv_logit(eta), s .* inv_logit(-eta));

  // The mixture, as a CORRECTION to the line above rather than a replacement for it:
  //
  //   log[(1-pi) L_main + pi L_low] = log L_main + log1m(pi) + log1p_exp(logit(pi) + d)
  //
  // with `d = log L_low - log L_main`. Writing it as a correction is what makes the
  // nesting bit for bit: at `theta = 0` the whole block is skipped, so the target IS the
  // untouched beta-binomial statement above rather than a value that rounds to it.
  //
  // `theta > 0` is the exact test, not a proxy — `pi_i = theta * inv_logit(.)` and
  // `inv_logit` is strictly positive, so `pi` is zero on every row or on none. Branching
  // on the scalar also keeps `log(0)` out of the autodiff graph entirely, which a per-row
  // guard could not do vectorized: the limit of the correction as `pi -> 0` is 0, and a
  // -inf whose adjoint is multiplied by a zero derivative is a nan rather than the zero it
  // mathematically is. The sampler works on the unconstrained scale, where `theta = 0`
  // sits at -inf and is never evaluated.
  //
  // Written through `lbeta` rather than through `beta_binomial_lpmf` because a mixture
  // needs the per-row density BEFORE it is summed, and Stan's lpmf vectorizes to a sum —
  // so the scalar form is a loop of 2N autodiff calls, which measured ~10x the cost of
  // the whole rest of the fit. `lbeta` vectorizes, and the binomial coefficient is common
  // to both components and cancels inside `d`, so it is never formed at all.
  if (P > 0 && theta[1] > 0) {
    vector[N] a = s .* inv_logit(eta);
    vector[N] b = s .* inv_logit(-eta);
    real s_low = (1 - rho_low[1]) / rho_low[1];
    real a_low = s_low * mu_low[1];
    real b_low = s_low * (1 - mu_low[1]);
    vector[N] yv = to_vector(y);
    vector[N] nv = to_vector(n);
    vector[N] d = (lbeta(yv + a_low, nv - yv + b_low) - lbeta(a_low, b_low))
                  - (lbeta(yv + a, nv - yv + b) - lbeta(a, b));
    vector[N] pi = theta[1] * inv_logit(Z * gamma);
    target += sum(log1m(pi)) + sum(log1p_exp(log(pi) - log1m(pi) + d));
  }
  if (P > 0) {
    gamma ~ normal(0, gamma_scale);
  }
}
