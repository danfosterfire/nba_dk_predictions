// Beta-geometric spell duration: a geometric per-game exit hazard with a Beta frailty
// integrated out analytically — the same device betabinomial_glm.stan uses for the season
// count, one level down, so this file introduces no explicit latents either and dense_e
// stays viable.
//
//   e ~ Beta(a, b),  T | e ~ Geometric(e)
//   P(T = t)  = B(a + 1, b + t - 1) / B(a, b)
//   P(T >= t) = B(a,     b + t - 1) / B(a, b)      <- right-censoring, one branch
//
// Parameterized as mu = a/(a+b) — which IS P(T = 1), the first-game exit hazard, so the
// linear predictor reads directly — and kappa = a + b. kappa -> infinity is the plain
// geometric, and the data rejects it by 11,278 log-likelihood points at one extra
// parameter on 68,530 interior spells (`make games-played`).
//
// ── Why the two edge classes are DATA rather than separate models ────────────────────
//
// A spell is one of three things and they differ only in which factor of the same
// likelihood they contribute, so one expression covers all of them:
//
//   interior        starts and ends inside the timeline        P(T = t)
//   right-censored  still running at the last game             P(T >= t)
//   left-truncated  already in progress at the first game      residual duration
//
// The first two differ *only* by whether `a` is incremented, which is why the target below
// has no branch at all. The third is a shift on mu: by memorylessness the forward
// recurrence time of a geometric is geometric with the same hazard, so under a Beta(a, b)
// frailty the residual duration is beta-geometric with the frailty size-biased by 1/e —
// i.e. Beta(a - 1, b). That is a *prediction*, so `open_shift` is fitted freely and
// compared against it rather than being fixed at it: agreement validates the renewal
// assumption and disagreement localizes the failure. Measured on the full window it
// disagrees, and informatively — the left-truncated spells are 58.7% `not_rostered` games,
// so they are roster mechanics rather than an injury in progress.
//
// ── H_open = 0 disables the offset EXACTLY ───────────────────────────────────────────
//
// The shipped head fits *within-tenure* spells, where a tenure opens and closes on a game
// the player appeared in — so every spell is interior, `truncated` is all zeros and
// `open_shift` is unidentified. Rather than let it sample its prior and print as a
// parameter that did not mix, `H_open` makes the vector zero-length when no row carries
// the flag, which is the same device `S = 0` uses for the year effect: the parameter
// space, the priors and the likelihood become identical to the model without it.
data {
  int<lower=0> N;                            // collapsed spell rows
  int<lower=0> K;                            // 0 is legal — the base head has no covariates
  matrix[N, K] X;                            // standardized on TRAIN only
  array[N] int<lower=1> t;                   // observed length in games
  vector<lower=0, upper=1>[N] censored;      // 1 = ran to the end of the schedule
  vector<lower=0, upper=1>[N] truncated;     // 1 = already in progress at game 1
  vector<lower=0>[N] w;                      // multiplicity from the collapse
  real<lower=0> beta_scale;                  // 1/sqrt(2*l2), per prior_sd_for_l2
  real<lower=0> intercept_scale;
  real<lower=0> kappa_scale;
  int<lower=0> S;                            // 0 disables the year effect EXACTLY
  array[N] int<lower=0> season_idx;
  real<lower=0> year_sd_scale;
}
transformed data {
  int H = S > 0 ? 1 : 0;
  int H_open = max(truncated) > 0 ? 1 : 0;
  vector[N] tv = to_vector(t);
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> kappa;
  vector[H_open] open_shift;                 // logit offset for an in-progress spell
  vector[S] year_z;
  vector<lower=0>[H] sigma_year;
}
model {
  vector[N] eta = alpha + X * beta;

  alpha      ~ normal(0, intercept_scale);
  beta       ~ normal(0, beta_scale);
  kappa      ~ normal(0, kappa_scale);       // half-normal on the <lower=0> support
  open_shift ~ normal(0, 1);                 // zero-length is a no-op
  if (H_open) eta += open_shift[1] * truncated;
  if (S > 0) {
    // Non-centered: with ~30 seasons and a small sigma the centered form is a funnel.
    year_z ~ std_normal();
    sigma_year ~ normal(0, year_sd_scale);
    eta += sigma_year[1] * year_z[season_idx];
  }

  // inv_logit(-eta) rather than 1 - inv_logit(eta), for the reason betabinomial_glm.stan
  // documents: the subtraction yields a shape parameter of exactly 0 by eta ~ 37 and
  // rejects the draw; this form holds to eta ~ 745.
  vector[N] a = kappa * inv_logit(eta);
  vector[N] b = kappa * inv_logit(-eta);
  // One expression covers both classes: a censored row contributes P(T >= t), an observed
  // row P(T = t), and they differ only by whether `a` is incremented.
  target += dot_product(w, lbeta(a + (1 - censored), b + tv - 1) - lbeta(a, b));
}
