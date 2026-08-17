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
//
// ── The MARGINAL representation of the same effect (`Q > 0`) ──────────────────
//
// Added 2026-08-16, `docs/composition-quadrature-plan.md`. The block above samples the
// `u_i`; this one INTEGRATES THEM OUT, per unit, by Gauss-Hermite quadrature inside the
// model block. The sampler then never sees a latent: parameters drop from 2,234 at the
// pilot window (12,307 at the full window) to ~35 at ANY window, the funnel cannot exist,
// and `dense_e` — which needs warmup >= 20 x parameters and is worth ~10x the wall clock
// on this head — comes back into reach. The two blocks are two representations of ONE
// posterior, which is why both stay in the file: their agreement is the only correctness
// check that does not rely on the quadrature being right.
//
// It is exact in structure. Given the data the likelihood FACTORIZES BY UNIT — every row's
// trials and bounds are data, each likelihood row belongs to exactly one (player, season),
// and the `u_i` are independent — so the joint marginal is a product of independent 1-D
// integrals and the only approximation anywhere is the numerical rule.
//
// **Node placement affects ACCURACY ONLY, never the estimand**, which is what licenses the
// data-derived centering below: the integral is the integral wherever the nodes sit. That
// is not a hopeful reading of the theory, it is measured — `docs/composition-quadrature-plan.md`
// §4 step 0 profiles sigma under two different placements and gets log-marginals agreeing
// to three decimals at every grid point.
//
// `u_center` and `u_curv` are the mode and curvature of each unit's log-likelihood in `u`,
// fitted ONCE IN THE DRIVER by Newton at the head's own no-fit floor — data, not an inner
// optimization, and deliberately not read off a fitted posterior so that placement can
// never give the head a silent dependence on the artifact it exists to replace. Only their
// closed-form combination with `sigma_u` moves here, which is a differentiable change of
// variables and costs three scalar ops per unit.
//
// **Why a Newton-fitted center rather than the plan's "shrunken deviation".** Measured at
// the pilot window over 2,062 units and eight values of sigma: fixed nodes need Q > 61 and
// are still 2.6 nats out; the plan's moment rule needs Q ~ 31; the Newton rule needs
// **Q = 21** for a worst-unit error of 2.5e-6 nats. Placement quality IS the node count.
//
// **`Q = 0` takes the vectorized path above EXACTLY** — `gh_x`, `gh_log_w` and every unit
// array are zero-length and `sigma_u` is sized by the block that is on. Same device as
// `S = 0` in betabinomial_glm.stan, `U_n = 0` above, and `n_rho = 1` here.
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
  // ── The marginal representation. Q = 0 disables ALL of it, exactly. ──
  int<lower=0> Q;                       // quadrature nodes per unit
  vector[Q] gh_x;                       // Gauss-Hermite abscissae
  vector[Q] gh_log_w;                   // log(w_q) + x_q^2 — the e^{x^2} folded in
  int<lower=0> n_unit;                  // units carrying at least one likelihood row
  array[n_unit] int<lower=1> u_start;   // 1-based start in the UNIT-MAJOR ordering
  array[n_unit] int<lower=1> u_len;
  array[n_unit] int<lower=1> u_bin;     // sigma bin, constant within a unit
  vector[n_unit] u_center;              // Newton mode of the unit log-lik in u — DATA
  vector[n_unit] u_curv;                // Newton curvature there — DATA
  real<lower=0> gh_inflate;             // node-scale inflation over the Laplace sd
  int<lower=1> n_sigma;                 // sigma bins; 1 nests the shared-sigma model
  int<lower=0> n_umap;                  // == the likelihood row count when Q > 0
  array[n_umap] int<lower=1> u_row;     // likelihood rows, in unit-major order
}
transformed data {
  // One deterministic last row per team-game, so P - G rows carry likelihood.
  int n_lik = P - G;
  int n_rho_par = dispersed ? n_rho : 0;   // no dispersion parameter on the binomial arm
  // The two representations are ALTERNATIVES, not layers: one sigma_u vector, sized by
  // whichever block is on. Length n_sigma under the marginal path (graded from day one),
  // length 1 under the sampled-latent path (which carries a scalar), zero under neither —
  // which is the model that shipped before either existed.
  int H_u = Q > 0 ? n_sigma : (U_n > 0 ? 1 : 0);
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
  // ── The marginal path's unit-major view, built once ───────────────────────
  // Zero-length under Q = 0, so none of it costs anything when the block is off.
  int n_uq = Q > 0 ? n_umap : 0;
  int n_bound_u = 0;
  array[n_uq] int<lower=1> u_of_row;      // which unit each unit-major row belongs to
  vector[n_uq] y_u;                       // successes, unit-major
  vector[n_uq] my_u;                      // trials - successes, unit-major
  array[n_uq] int u_rho_bin;
  if (Q > 0) {
    array[P] int seen = rep_array(0, P);
    int pos = 1;
    if (n_umap != n_lik) {
      reject("u_row has ", n_umap, " entries against ", n_lik, " likelihood rows");
    }
    for (i in 1:n_unit) {
      if (u_start[i] != pos) {
        reject("u_start[", i, "] = ", u_start[i], " is not contiguous; expected ", pos);
      }
      if (u_bin[i] > n_sigma) reject("u_bin[", i, "] exceeds n_sigma");
      for (j in pos:(pos + u_len[i] - 1)) {
        int r = u_row[j];
        if (is_last[r]) reject("u_row[", j, "] points at a deterministic remainder row");
        if (seen[r] != 0) reject("row ", r, " appears in more than one unit");
        seen[r] = 1;
        u_of_row[j] = i;
        y_u[j] = y[r];
        my_u[j] = m[r] - y[r];
        u_rho_bin[j] = rho_bin[r];
        if (lo[r] > 0) n_bound_u += 1;
      }
      pos += u_len[i];
    }
    if (pos != n_umap + 1) reject("u_len does not cover u_row exactly");
    // EVERY likelihood row must be in exactly one unit, or the marginal is integrating
    // a different model from the one the vectorized path fits — silently.
    for (i in 1:n_lik) {
      if (seen[lik[i]] == 0) reject("likelihood row ", lik[i], " is in no unit");
    }
    if (!dispersed) {
      reject("the marginal path is beta-binomial only; set dispersed = 1 or Q = 0");
    }
    if (U_n > 0) {
      reject("Q > 0 and U_n > 0 are two representations of the SAME effect; pick one");
    }
  }
  array[n_bound_u] int bound_u;           // bound rows, as positions in the unit-major view
  {
    int j = 1;
    for (t in 1:n_uq) {
      if (lo[u_row[t]] > 0) {
        bound_u[j] = t;
        j += 1;
      }
    }
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
  // Length n_sigma under the marginal path, 1 under the sampled latent, 0 under neither.
  vector<lower=0>[H_u] sigma_u;
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

  if (Q > 0) {
    // ── The marginal path: one 1-D integral per unit, nothing latent in the posterior ──
    //
    // Per unit i the nodes sit at the Laplace approximation of ITS OWN posterior in u,
    // which is `u_center`/`u_curv` (data) combined with the current `sigma_u` in closed
    // form. Rewriting the change of variables u = c + sqrt(2) s x:
    //
    //   int N(u | 0, sigma) f(u) du
    //     = log_sum_exp_q [ log w_q + x_q^2 + log(sqrt(2) s) + log N(u_q | 0, sigma)
    //                       + log f(u_q) ]
    //
    // exact for ANY (c, s) — the placement buys accuracy, never a different answer.
    vector[n_unit] sig = sigma_u[u_bin];
    vector[n_unit] prec = u_curv + inv_square(sig);
    vector[n_unit] u_c = u_center .* u_curv ./ prec;
    vector[n_unit] u_s = gh_inflate * inv_sqrt(prec);
    vector[n_uq] eta_u = eta[u_row];                  // unit-major gather, once per gradient
    vector[n_uq] s_u = (1 - rho[u_rho_bin]) ./ rho[u_rho_bin];
    matrix[n_unit, Q] node_lp;

    sigma_u ~ normal(0, u_sd_scale);
    for (q in 1:Q) {
      vector[n_unit] uq = u_c + sqrt2() * gh_x[q] * u_s;
      vector[n_uq] e = eta_u + uq[u_of_row];
      // Same `inv_logit(-eta)` convention as the vectorized path — never `1 - inv_logit`.
      vector[n_uq] a = s_u .* inv_logit(e);
      vector[n_uq] b = s_u .* inv_logit(-e);
      // The beta-binomial log-pmf less its data-only `lchoose(m, y)`, which is exactly what
      // `beta_binomial_lupmf` drops. It is constant across nodes, so it factors straight out
      // of the log_sum_exp and shifts the target by a data constant — the same convention
      // the vectorized path is already on, which is what keeps the two comparable.
      vector[n_uq] term = lbeta(y_u + a, my_u + b) - lbeta(a, b);
      for (t in 1:n_bound_u) {
        int j = bound_u[t];
        term[j] -= beta_binomial_log_tail_mass(lo[u_row[j]], m[u_row[j]], a[j], b[j]);
      }
      for (i in 1:n_unit) {
        node_lp[i, q] = gh_log_w[q] + log(sqrt2() * u_s[i])
                        + normal_lpdf(uq[i] | 0, sig[i])
                        + sum(segment(term, u_start[i], u_len[i]));
      }
    }
    for (i in 1:n_unit) {
      target += log_sum_exp(node_lp[i]);
    }
  } else if (dispersed) {
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
