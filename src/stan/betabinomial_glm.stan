// Beta-binomial GLM: y successes out of n trials, logit-linear mean, one shared
// dispersion rho.
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
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;                     // standardized on TRAIN only; intercept is separate
  array[N] int<lower=0> n;            // trials
  array[N] int<lower=0> y;            // successes, y <= n on every row
  real<lower=0> beta_scale;           // 1/sqrt(2*l2) reproduces an L2 penalty of l2
  real<lower=0> intercept_scale;
  int<lower=0> S;                     // training seasons; 0 disables the year effect
  array[N] int<lower=0> season_idx;   // 1..S, ignored (and all zero) when S == 0
  real<lower=0> year_sd_scale;        // half-normal scale on sigma_year
}
transformed data {
  int H = S > 0 ? 1 : 0;
}
parameters {
  // Bounds mirror RHO_MIN / RHO_MAX in `src/models/availability.py` exactly, so this is
  // the same parameter space the MLE searches rather than a wider one. They are guard
  // rails, not a prior: at rho -> 1 the beta collapses to a two-point distribution on 0
  // and n, which is never a useful forecast and is numerically nasty (the shape
  // parameters both go to 0); at rho -> 0 they both diverge. Fitted values land near
  // 0.28 on availability and far lower elsewhere, so neither bound binds.
  real<lower=1e-6, upper=0.95> rho;
  real alpha;
  vector[K] beta;
  vector[S] year_z;                   // zero-length when S == 0
  vector<lower=0>[H] sigma_year;      // zero-length when S == 0
}
model {
  // `eta` is local rather than a `transformed parameter`: it is one value per row, so
  // saving it would write N x draws numbers to disk (~40M on the availability head) for
  // a quantity every consumer recomputes from alpha/beta anyway.
  vector[N] eta = alpha + X * beta;
  real s = (1 - rho) / rho;

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
  y ~ beta_binomial(n, s * inv_logit(eta), s * inv_logit(-eta));
}
