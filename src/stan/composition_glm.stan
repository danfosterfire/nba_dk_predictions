// Team-game minutes composition: a multinomial decomposed into sequential binomial
// trials, with the per-player cap enforced through the TRIALS rather than checked
// after the fact.
//
// For team-game g, the N = 5 * game_length minutes are allocated among the K players
// who played, ordered by prior-season minutes share (largest first, rookies last).
// Player k draws
//
//   y_k ~ BetaBinomial(m_k, p_k, rho)     m_k = min(U, R_k),  R_k = N - sum_{j<k} y_j
//
// and the last player takes the remainder deterministically. `dispersed = 0` collapses
// the same file to binomial steps — the pure stick-breaking decomposition piloted in
// demo_decomposed_multinomial.stan, which the measured game-level dispersion
// (4.65x binomial, `make stan-minutes`) predicts will be far too tight.
//
// Two deliberate departures from that demo, both fixes:
//   * The demo declares upper bounds U but never enforces them in the likelihood — a
//     star's step at mean ~40 of 240 puts real mass above 48. Here the cap is
//     structural: trials are the remaining CAPACITY min(U, R), so support never
//     exceeds the cap and no O(support) truncation CDF enters the gradient. Where no
//     cap binds (R <= U) this is exactly the multinomial stick-break.
//   * The demo's final check compares an integer count to a real probability. Here
//     `transformed data` rejects on sum(y) != N, y > m, or y < lo — data bugs are
//     build failures, not metrics.
//
// The feasibility LOWER bound lo_k = max(0, R_k - J_k * U) (J_k players remaining
// after k) guarantees the rest of the roster can absorb the remainder — it is what
// makes the deterministic last step valid by induction. It binds only in
// short-rotation games (K = 6 occurs in 10 of 71,092 team-games), so it is an
// explicit truncation over the near-empty set of rows where lo > 0.
//
// The offset is the no-fit floor: logit_prior = logit(carry-forward conditional
// share), so alpha = beta = 0 IS prior shares renormalized over who played. beta
// fits deviations from proportional redistribution.
//
// m, lo and the row order are DATA — functions of the observed allocation — so the
// likelihood is one vectorized beta_binomial over the non-deterministic rows, the
// same shape as betabinomial_glm.stan. At simulation time they are recomputed
// sequentially from each draw; see src/models/stan_composition.py.
//
// ── The optional per-(player, season) random effect ───────────────────────────
//
// Added 2026-08-09. `make minutes-unification` scored this head's summed draws at the
// SEASON unit and found them 4.68x too narrow (predictive sd 64.65 against the marginal
// head's 302.75), and traced that to a missing parameter rather than to the team
// constraint: a SHARED shift is definitionally a re-allocation here and has 0.000000% of
// the residual variance to reach, but a per-(player, season) shift is not shared — one
// player's breakout takes minutes from a teammate, which is exactly what the constraint
// permits. Injecting `sigma * z` per unit per posterior draw into the fitted posterior
// moved the season-total sd to 239.45 and tied the marginal head on CRPS. This block
// fits it instead of injecting it.
//
// The index is (player, season) and NOT player: a minutes role is a property of the
// season a player is in, and a career-long effect would be absorbed by `logit_share_lag1`
// and the offset.
//
// **U_n = 0 disables it EXACTLY.** `u_z` and `sigma_u` are then zero-length, so the
// parameter space, the priors and the likelihood are identical to the model without this
// block. That is the same `S = 0` device `betabinomial_glm.stan` uses for its year effect
// and the same one `n_rho_par` already uses here for the binomial arm — a house pattern
// rather than an import. A test pins the nesting, because it is the only thing separating
// "a parameter was added" from "the shipped head was silently changed".
//
// `u_centered` picks the parameterization, and the reason it is a knob rather than a
// constant is that neither choice is right for this frame. Rows per unit run median 57,
// p10 11, minimum 1: the well-informed units would prefer the centered form and the
// one-game units are a funnel under it. Non-centered is the default because the funnel is
// the failure that produces wrong answers rather than slow ones — but a tiny step size with
// treedepth saturation is the *other* symptom, and the centered arm is the response to it.
// Both are the same model; only the geometry NUTS walks differs.
functions {
  // log P(Y >= lo), summed UPWARD from lo in log space via the pmf ratio
  // recurrence. Two numerical facts force this exact shape, both paid for in runs:
  //   * NOT beta_binomial_lcdf: Stan routes the beta-binomial CDF through the
  //     generalized hypergeometric (grad_F32), whose autodiff is slow enough that
  //     ~400 truncation rows made one gradient cost seconds.
  //   * NOT 1 - head_mass: when the tail is tiny the head mass rounds to exactly
  //     1.0 and log1m(1) is -inf at the fixed init — which killed every chain of
  //     the binomial arm's first full-data fit. Both the pmf at y and the tail
  //     normalization are tiny together; their ratio is O(1), so summing the tail
  //     directly keeps the target finite everywhere.
  //   pmf ratio: pmf(k+1) / pmf(k) = (n-k)(a+k) / ((k+1)(b+n-k-1))
  real beta_binomial_log_tail_mass(int lo, int n, real a, real b) {
    real log_term = beta_binomial_lpmf(lo | n, a, b);
    real log_total = log_term;
    for (k in lo:(n - 1)) {
      log_term += log(((n - k) * (a + k)) / ((k + 1.0) * (b + n - k - 1.0)));
      log_total = log_sum_exp(log_total, log_term);
    }
    return log_total;
  }
  real binomial_log_tail_mass(int lo, int n, real p) {
    real log_term = binomial_lpmf(lo | n, p);
    real log_total = log_term;
    for (k in lo:(n - 1)) {
      log_term += log(((n - k) * p) / ((k + 1.0) * (1 - p)));
      log_total = log_sum_exp(log_total, log_term);
    }
    return log_total;
  }
}
data {
  int<lower=1> G;                       // team-games
  int<lower=1> P;                       // player rows, concatenated team-game-major
  int<lower=0> K;                       // features
  array[G] int<lower=1> start;          // first row of team-game g (1-based)
  array[G] int<lower=2> len;            // players in team-game g (observed 6..15)
  array[P] int<lower=0> y;              // integer minutes, largest-remainder rounded
  array[P] int<lower=0> m;              // trials = min(U, R): remaining capacity
  array[P] int<lower=0> lo;             // feasibility lower bound, almost always 0
  array[P] int<lower=0, upper=1> is_last;  // deterministic remainder rows
  array[G] int<lower=1> N_total;        // 5 * game_length: 240 / 265 / 290 / ...
  array[G] int<lower=1> U;              // per-player cap = game_length: 48 / 53 / ...
  vector[P] logit_prior;                // stick-breaking carry-forward offset
  matrix[P, K] X;                       // standardized on TRAIN only
  int<lower=0, upper=1> dispersed;      // 0 = binomial steps, 1 = beta-binomial
  // Dispersion bins over prior minutes share. n_rho = 1 with rho_bin all-ones is
  // EXACTLY the shared-rho model, so the graded variant nests it and the two are
  // one code path. Measured on the pilot, one shared rho is too tight for fringe
  // players (realized/simulated variance ratio 1.59) and too wide for stars (0.70)
  // — a 34-mpg starter's allocation step is genuinely steadier than a reserve's.
  int<lower=1> n_rho;
  array[P] int<lower=1, upper=n_rho> rho_bin;   // bin edges from TRAIN quantiles only
  real<lower=0> beta_scale;             // 1/sqrt(2*l2) reproduces an L2 penalty of l2
  real<lower=0> intercept_scale;
  int<lower=0> U_n;                     // (player, season) units; 0 disables the effect
  array[P] int<lower=0> unit_idx;       // 1..U_n, ignored (and all zero) when U_n == 0
  real<lower=0> u_sd_scale;             // half-normal scale on sigma_u
  int<lower=0, upper=1> u_centered;     // 0 = non-centred (default), 1 = centred
}
transformed data {
  // One deterministic last row per team-game, so P - G rows carry likelihood.
  int n_lik = P - G;
  int n_rho_par = dispersed ? n_rho : 0;   // no dispersion parameter on the binomial arm
  int H_u = U_n > 0 ? 1 : 0;               // no sigma_u when the effect is disabled
  int n_bound = 0;
  for (r in 1:P) {
    if (!is_last[r] && lo[r] > 0) n_bound += 1;
  }
  array[n_lik] int lik;
  array[n_bound] int bound;
  {
    int i = 1;
    int j = 1;
    for (r in 1:P) {
      if (!is_last[r]) {
        lik[i] = r;
        i += 1;
        if (lo[r] > 0) {
          bound[j] = r;
          j += 1;
        }
      }
    }
    if (i != n_lik + 1) reject("is_last must mark exactly one row per team-game");
  }
  // Integrity rejects — the demo's flaws, fixed as loud data-bug failures. The last
  // row's cap is covered by y <= m: its m is min(U, R), so a remainder above the cap
  // fails here rather than fitting silently.
  for (g in 1:G) {
    int total = 0;
    for (j in 0:(len[g] - 1)) {
      int r = start[g] + j;
      if (y[r] > m[r]) reject("y exceeds trials min(U, R) at row ", r);
      if (y[r] < lo[r]) reject("y below the feasibility bound at row ", r);
      total += y[r];
    }
    if (total != N_total[g]) {
      reject("minutes sum to ", total, " not N = ", N_total[g], " in team-game ", g);
    }
  }
}
parameters {
  // Zero-size when dispersed = 0, so one file serves both arms and the binomial arm
  // is exactly the rho -> 0 limit. Length n_rho when dispersed: one dispersion per
  // prior-minutes bin, and n_rho = 1 is the shared-rho model exactly. Bounds mirror
  // RHO_MIN / RHO_MAX in src/models/availability.py; no prior on rho, so the mode is
  // the penalized MLE, and each bin carries thousands of rows.
  vector<lower=1e-6, upper=0.95>[n_rho_par] rho;
  real alpha;
  vector[K] beta;
  vector[U_n] u_z;                      // zero-length when U_n == 0
  vector<lower=0>[H_u] sigma_u;         // zero-length when U_n == 0
}
model {
  vector[P] eta = logit_prior + alpha + X * beta;

  alpha ~ normal(0, intercept_scale);
  beta ~ normal(0, beta_scale);
  if (U_n > 0) {
    sigma_u ~ normal(0, u_sd_scale);
    if (u_centered) {
      // The SAME model, walked in the other coordinates: u ~ normal(0, sigma_u) entering
      // eta directly. Preferred when every unit is well informed, which the median 57 rows
      // per unit says most of them are.
      u_z ~ normal(0, sigma_u[1]);
      eta += u_z[unit_idx];
    } else {
      u_z ~ std_normal();
      eta += sigma_u[1] * u_z[unit_idx];
    }
  }

  if (dispersed) {
    // Per-row dispersion by bin, still one vectorized beta_binomial call: multiple
    // indexing (`rho[rho_bin[lik]]`) gathers the right rho for every row at once.
    vector[n_lik] s = (1 - rho[rho_bin[lik]]) ./ rho[rho_bin[lik]];
    // `s * inv_logit(-eta)` rather than `s * (1 - inv_logit(eta))`: inv_logit
    // saturates to exactly 1.0 by eta ~ 37 and the subtraction makes a beta shape
    // parameter exactly 0, rejecting the whole target. Same convention as
    // betabinomial_glm.stan, pinned by a test.
    target += beta_binomial_lupmf(y[lik] | m[lik], s .* inv_logit(eta[lik]),
                                  s .* inv_logit(-eta[lik]));
    for (i in 1:n_bound) {
      int r = bound[i];
      real s_r = (1 - rho[rho_bin[r]]) / rho[rho_bin[r]];
      // Normalize over the feasible support [lo, m]: divide by P(Y >= lo).
      target += -beta_binomial_log_tail_mass(lo[r], m[r], s_r * inv_logit(eta[r]),
                                             s_r * inv_logit(-eta[r]));
    }
  } else {
    target += binomial_logit_lupmf(y[lik] | m[lik], eta[lik]);
    for (i in 1:n_bound) {
      int r = bound[i];
      target += -binomial_log_tail_mass(lo[r], m[r], inv_logit(eta[r]));
    }
  }
}
