// Negative-binomial (NB2) rate GLM with a log link and an exposure offset.
//
// The eight component count heads — fg2a, fg3a, fta, reb, ast, stl, blk, tov — are this
// model, fitted season-collapsed: `y` is the season total and `exposure` the season's
// minutes. That collapse is an algebraic identity, not an approximation. For
// y_g ~ Poisson(m_g * e^eta) with eta constant within a player-season,
//
//   prod_g p(y_g) = Poisson(Y; M*e^eta) x Multinomial(y; Y, m_g/M)
//
// and the multinomial factor is free of eta, so 10,194 player-season rows recover the
// same coefficients as 731,906 player-game rows. What the collapse discards is the
// within-season allocation: per-game covariates, per-game dispersion, and serial
// dependence. `phi` here is therefore SEASON-level heterogeneity — a season total cannot
// distinguish a per-game random effect from a per-season one, since iid per-game noise is
// diluted by ~1/G while a shared season multiplier passes through in full.
//
// The specification is SCALE, not curvature (`CLAUDE.md`, `make component-rates`). A log
// link wants a multiplicative predictor: with log(prior rate) in X the model is
// rate ~ prior_rate^beta, which is the right shape. Linear-in-raw-rate inside exp() is
// badly misspecified — held-out R^2 0.520 on fg3a and 0.638 on blk against 0.879/0.820
// once the scale is fixed.
//
// `phi` is parameterized through its inverse: phi_inv = 0 is exactly Poisson, so the
// prior sits on the *departure* from Poisson and the boundary is reachable rather than
// pushed to infinity. A half-normal on 1/sqrt(phi) is the usual recommendation; the
// exponential below is the same idea with a heavier tail, and at ~10^4 rows it is
// dominated by the data on every head.
//
// ── The optional year-level random effect ────────────────────────────────────────────
//
// A season FIXED effect is unusable at prediction time — there is no dummy for a season
// that has not happened. A year-level RANDOM effect is usable, because at prediction time
// it contributes mean zero and its variance: it widens the predictive rather than shifting
// it. That is the whole point, and it is the only form that can touch `yoy_sd_pct`, the
// irreducible year-to-year spread a trend term provably cannot reach (`make season-effects`;
// subtracting a linear trend shifts the mean of the year-over-year changes and leaves their
// variance exactly unchanged, since diff(a + b*x) is the constant b).
//
// **S = 0 disables it EXACTLY, not approximately.** `year_z` and `sigma_year` are then
// zero-length vectors, so the parameter space, the priors and the likelihood are identical
// to the model without this block — the same nesting identity `composition_glm.stan` gets
// from `n_rho = 1`, and it is what lets one file serve both arms of the ablation. A test
// pins it.
//
// NON-CENTERED on purpose: with ~28 seasons and a small sigma, `year ~ normal(0, sigma)`
// is a funnel and NUTS handles it badly. `sigma * z` with `z ~ std_normal()` is the
// standard reparameterization and samples cleanly at this group count.
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;                     // standardized on TRAIN only; intercept is separate
  array[N] int<lower=0> y;            // season total of the component
  vector<lower=0>[N] exposure;        // season minutes — the offset, never a coefficient
  real<lower=0> beta_scale;
  real<lower=0> intercept_scale;
  real<lower=0> phi_inv_scale;
  int<lower=0> S;                     // training seasons; 0 disables the year effect
  array[N] int<lower=0> season_idx;   // 1..S, ignored (and all zero) when S == 0
  real<lower=0> year_sd_scale;        // half-normal scale on sigma_year
}
transformed data {
  // Constant across the fit, so it is computed once instead of at every gradient
  // evaluation.
  vector[N] log_exposure = log(exposure);
  int H = S > 0 ? 1 : 0;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> phi_inv;
  vector[S] year_z;                   // zero-length when S == 0
  vector<lower=0>[H] sigma_year;      // zero-length when S == 0
}
transformed parameters {
  real<lower=0> phi = inv(phi_inv);
}
model {
  vector[N] eta = log_exposure + alpha + X * beta;
  alpha ~ normal(0, intercept_scale);
  beta ~ normal(0, beta_scale);
  phi_inv ~ exponential(inv(phi_inv_scale));
  if (S > 0) {
    year_z ~ std_normal();
    sigma_year ~ normal(0, year_sd_scale);
    eta += sigma_year[1] * year_z[season_idx];
  }
  y ~ neg_binomial_2_log(eta, phi);
}
