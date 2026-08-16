# Composition quadrature plan — fitting `sigma_u` by marginalizing the latents

**Opened 2026-08-16. This is a workplan, not a results doc: nothing here has been built.**
It is deliberately **not** in `make docs-audit`. Figures quoted from audited artifacts say
so; figures from this session's scratch measurements are labelled scratch and earn audited
status only when a step below gives them a `make` target.

## 1. The problem, and why a third representation

The composition head's season-unit underdispersion is a missing **per-(player, season)**
parameter (`docs/minutes-window-plan.md`, `README.md` §2). Two representations of that
parameter exist today:

- **The injection** (ships): `sim.minutes.player_season_sigma`, a constant σ plugged into
  `rehydrate_composition`. Works, but σ is grid-searched per offset change (the preseason
  blend forced 0.450 → 0.375 by hand), carries no posterior uncertainty, and `beta` is
  estimated at σ = 0 — never jointly.
- **The sampled latent** (`U_n` block in `src/stan/composition_glm.stan`): the honest fit,
  and it does not converge in budget. At the pilot window it is a 2,204-latent hierarchical
  posterior (min ESS 35, R̂ 1.0948 at 500+500; four sampler-side remedies all made it worse —
  `docs/potential-to-dos.md` §2). At the **full window it is 12,307 latents**, extrapolated
  at 37.6 h pre-blend, and `dense_e` is unreachable at any warmup (`choose_metric` needs
  warmup ≥ 20 × params). A post-blend re-attempt at the pilot window is in flight as of this
  writing (Gate A priced it at 14.6 h; see §7).

The third representation: **integrate each `u_i` out numerically inside the model block**
(1-D Gauss–Hermite quadrature per unit), so the sampler never sees a latent. Parameters drop
from 2,234 to ~35 at *any* window; the funnel never exists; `dense_e` comes back. The two
key facts that make it exact-in-structure:

- Given the data, the likelihood **factorizes by unit**: every row's trials `m` and bounds
  are data, each row belongs to exactly one (player, season), and the `z_i` are independent
  — so the joint marginal is a product of independent 1-D integrals. The only approximation
  anywhere is the numerical rule itself.
- Node placement affects **accuracy only, never the estimand** — the integral is the
  integral wherever the nodes sit. This is what licenses the data-centered scheme in §4
  step 0.

## 2. The design

### The marginal path in `composition_glm.stan`

Same file, house nesting: new data `Q` (nodes), `gh_z` / `gh_log_w` (node positions and
log-weights, computed in the driver), unit-major ragged indices (`u_start`, `u_len`,
`row_of` over likelihood rows), and per-unit σ-bin index. **`Q = 0` takes the existing
vectorized path exactly** — the same device as `S = 0`, `U_n = 0`, `n_rho = 1`. The `U_n`
sampled-latent block **stays in the file**: the two are representations of the same
posterior, and their probe-scale agreement is the strongest correctness check available
(§4 step 4).

Per log-posterior evaluation, per unit `i`:

```
lq[q] = gh_log_w[q]
        + sum over unit i's rows of beta_binomial_lpmf(y | m, eta0 + sigma[bin_i] * gh_z[q])
        (+ the same tail-mass normalization the row already carries, at the shifted eta)
target += log_sum_exp(lq)
```

All in log space. The truncation rows (`lo > 0`, 1.7% of rows) are evaluated per node like
everything else. The per-unit loop is exactly the shape `reduce_sum` slices over if cost
demands it (§4 step 2b).

### The quadrature rules (so the next session does not re-derive them)

Standard GH, for `∫ φ(z) f(z) dz` with `(x_q, w_q)` from `scipy.special.roots_hermite(Q)`:

```
z_q = sqrt(2) * x_q          log_w_q = log(w_q) - 0.5 * log(pi)
∫ ≈ Σ_q exp(log_w_q) f(z_q)
```

Centered/scaled variant with **data-derived** per-unit center `c_i` and scale `s_i`
(change of variables `z = c_i + sqrt(2) s_i x`, exact for any `c_i`, `s_i`):

```
log-term_q = log(w_q) + log(sqrt(2) * s_i) + x_q^2
             + normal_lpdf(c_i + sqrt(2) s_i x_q | 0, 1)
             + log f(c_i + sqrt(2) s_i x_q)
```

`c_i`, `s_i` are data (computed once in the driver from the unit's own fitting rows — a
shrunken observed logit-share deviation for `c_i`, a rough information-based `s_i` clamped
to [0.2, 1]), so no inner optimization enters the gradient. Well-placed centers typically
cut the required Q to 5–10; that is the whole reason the variant exists.

### Role-graded σ, from day one

`sigma_u` becomes a **vector over the head's own `rho_bin`** (train-quantile bins of
`w_share`, already assigned per row by the recipe, constant within unit): `u_i ~ N(0,
sigma[bin_i])`, half-normal(`u_sd_scale`) prior per bin, `n_sigma = 1` nesting the shared-σ
model exactly — one code path, the `n_rho` pattern one parameter over.

The motivating measurement (**scratch, 2026-08-16**: validation, 1,111 units, persisted
`train` posterior through the `rehydrate → predict_samples → season_totals` path, PIT tails
nominal 0.05/side):

| role (`rho_bin`) | n | σ=0 sd_ratio | shipped σ=0.375 sd_ratio | shipped PIT KS | tails lo/hi |
|---|---|---|---|---|---|
| 1 fringe | 456 | 4.42 | **1.73** | 0.196 | .160 / .066 |
| 2 bench | 275 | 4.30 | **1.17** | 0.197 | .193 / .040 |
| 3 starter | 191 | 3.98 | **0.93** | 0.061 | .089 / .042 |
| 4 star | 189 | 3.83 | **0.85** | 0.082 | .048 / .011 |

Two readings. The raw deficit (σ=0 column) is nearly role-flat, but the **constant
logit-scale σ lands unevenly on the minutes scale**: fringe still 1.73× underdispersed,
stars overdispersed at 0.85× — a ~2× gradient in what σ wants, the same axis and direction
as the fitted per-game ρ (0.140 fringe → 0.074 star). And the fringe/bench miss is **not
pure spread**: their PIT tails are asymmetric (realized seasons collapse below the
predictive far more often than they exceed it — "the role evaporated"), which a symmetric
Gaussian `u` narrows but cannot shape. Graded σ should roughly halve the fringe miss; the
residue is a skewness question, recorded in §5 as a falsifier boundary, not chased here.

(Pooled bias is exactly 0 in both arms by the team-sum constraint — every team-game's
minutes are fully allocated — so the per-bucket biases are pure allocation errors. A useful
wiring check for whoever re-runs this.)

## 3. Why this beats paying the latent cost again

- **The scaling axis changes.** The latent representation scales in *parameters* (2,204
  pilot → 12,307 full window — realistically never fittable); the marginal scales only in
  *rows* (~35 params at any window). The full-window fitted σ — the thing that would
  actually ship in production and retire the marginal head (`README.md` §2) — is only
  credible under the marginal representation.
- **The `dense_e` dividend.** ~35 params at warmup 1000 clears the 20× rule the 2,234-param
  arm never can. The measured cliff on this head is ~10× wall clock (`diag_e` treedepth 8–9
  against `dense_e` 4, per the config's own comment).
- **Cost arithmetic** (to be replaced by step-0/step-5 measurements): post-blend `base` at
  `dense_e`, 1000+1000, is ~880 s. The marginal gradient is ~Q× `base`'s likelihood work, so
  Q = 25 fixed-node ≈ 6 h serial, Q = 10 centered ≈ 2.5 h — before `reduce_sum`, against
  14.6 h (Gate A, post-blend) for the latent arm with convergence unassured. Full window ≈
  6.5× rows.

## 4. Implementation steps, in order

**Step 0 — Q study in numpy, before any Stan.** Evaluate the marginal log-likelihood at
fixed θ (posterior means from the persisted `train` composition artifact; σ ∈ {0.375,
0.45}) under Q and 2Q+1 nodes, both node schemes. Deliverable: the smallest stable Q per
scheme, and the scheme decision. An afternoon; no sampler. **If fixed-node needs Q > ~40
and centered > ~15, stop and re-read §5's falsifier before spending more.**

**Step 1 — the Stan change.** §2's data block and marginal path in `composition_glm.stan`,
`Q = 0` nesting, `sigma_u` as `vector[n_sigma]`, keep the `U_n` block. Integrity rejects
mirror the existing ones (every likelihood row in exactly one unit; bins in range).

**Step 2 — the driver.** In `src/models/stan_composition.py`: unit-major index construction,
nodes/weights (and `c_i`/`s_i` for the centered scheme) as data, config knobs. Proposed:
two new arms in `stan.composition.effects.arms` — `mq` (marginal, shared σ) and `mq_graded`
— plus `stan.composition.effects.quadrature: {q, scheme}`. **2b, only if step-5 cost
demands:** `reduce_sum` over units, `STAN_THREADS`, `threads_per_chain`.

**Step 3 — tests** (house style: plain `assert`, synthetic builders):
- quadrature marginal log-lik matches a brute-force dense-grid integral on tiny synthetic
  units, both schemes;
- `Q = 0` reproduces the shipped path exactly;
- `n_sigma = 1` graded ≡ shared.

**Step 4 — probe-scale agreement.** Fit `mq` on the one-season probe frame and compare
`(beta, rho, sigma_u)` posteriors against the latent `ps` arm at the same scale — and
against the in-flight post-blend `ps` pilot fit if it converged (§7). Disagreement here is
a bug, full stop.

**Step 5 — the pilot ladder.** `mq`, `mq_graded`, and the post-blend `base` control, blended
frame, pilot window, through the `composition_effects` machinery (a separate output
directory, as the post-blend round already does — the pre-blend record is audited). Gates
in §5.

**Step 6 — full window + persistence, only if §5 passes.** The full-window fit the latent
representation cannot do; `make posteriors` persists `sigma_u_draws` (per-bin, with the bin
mapping in `extras`). Named code touches: `rehydrate_composition` and the `PlayerSeasonTerm`
path currently assume a scalar σ — the graded version needs the unit → bin lookup at predict
time (the recipe already assigns `rho_bin`, so the mapping is available). Promote the
per-role PIT table to a `make` target in the same session so §2's table stops being scratch.

## 5. Gates and falsifiers

Gates, stated before anything runs:
- **Convergence** at the standard bars — the thing the latent arm failed. Non-negotiable.
- **Correctness**: steps 3 and 4 agree.
- **P3 guard**: per-team-game CRPS must not trade away the incumbent's win (a fitted σ
  widens per-game marginals slightly; measure it).
- **P2**: season-unit CRPS/PIT against the marginal head *and against the shipped
  injection* — the injection is the real incumbent. Per-role PIT must improve fringe/bench
  tails without wrecking starter/star.
- **σ sanity**: the shared arm's posterior lands in the five-routes band 0.375–0.48; the
  graded arm brackets it with fringe > star.

Falsifiers, and what each would mean:
- **Step 0 needs huge Q even centered** → the integrand is nastier than the theory says;
  the cost case collapses; stay with the injection (graded, via the sweep machinery, which
  needs no Stan at all).
- **Step 4 disagrees** → a bug in the quadrature or the latent fit; do not proceed on
  metrics.
- **`mq_graded` ties `mq` everywhere** → the per-role table above was spread the shared σ
  already buys at the season unit; keep the shared arm, record the null.
- **Graded σ fails to cut the fringe left tail** → the residue is the skew/"role
  evaporated" component, which is a different likelihood question (skewed or mixture `u`)
  and gets its own `potential-to-dos` entry rather than scope creep here.
- **Everything converges but ties the injection on every metric** → plausible, and the ship
  argument is then structural, made explicitly: full-window path, σ that re-estimates when
  the offset moves, posterior uncertainty integrated, marginal-head retirement path. The
  availability mixture shipped on exactly that shape of argument.

## 6. Relationship to the in-flight post-blend latent fit

A post-blend `ps` re-attempt (pilot window, 1000+1000, blended frame via
`stan_composition.head_frame`, outputs under `outputs/predictions/effects_postblend/`)
launched 2026-08-16 ~06:46; Gate A priced the `ps` arm at **14.6 h**. Whatever it returns
feeds this plan rather than competing with it:

- **converged** → it is step 4's pilot-scale reference posterior, the best possible
  correctness check for `mq`; the quadrature case narrows to the full window + graded σ +
  cost;
- **unconverged again** → the latent route is dead at two offsets and this plan is the only
  fitted path left;
- either way its `base` row (per-team-game CRPS 4.2645 post-blend pilot) is the step-5
  control.
