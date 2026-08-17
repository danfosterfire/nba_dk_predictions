# Composition quadrature plan — fitting `sigma_u` by marginalizing the latents

**Opened 2026-08-16.** It is deliberately **not** in `make docs-audit`. Figures quoted from
audited artifacts say so; figures from scratch measurements are labelled scratch and earn
audited status only when a step below gives them a `make` target.

**Session 1 (2026-08-16) built steps 0 through 3 and the code is in the repo.** §8 is the
log. In one line: the falsifier did not fire, the representation is implemented and tested
against a brute-force integral, and the node count turned out to be a property of the node
*placement* rather than of the integrand — which is the session's one real finding and the
reason the plan's own §4 step 0 boundary was nearly tripped by a fixable mistake.

## ⛔ PARKED 2026-08-16, by owner decision, on cost against reach — not on a failed gate

**Steps 5 and 6 will not be run.** Every gate this plan set was passing: `mq` converges where
the latent arm does not, the two agree on σ to 0.35%, the quadrature matches a brute-force
integral to floating-point identity, and the graded σ turns out to be real. The line is parked
anyway, and the reason is the arithmetic in §8's feasibility section rather than any of that.

The full window is **~74 h serially** for one arm and needs `reduce_sum` to come down. Spending
three days of fitting buys a σ that is *estimated* rather than plugged in — but it does **not**
buy the clean "everything comes from one posterior" simulator, because the injected σ is one of
five draw-time inputs the simulator is *given* rather than fits (`docs/sim-inputs-plan.md`), and
retiring one of them leaves the other four. **Partway to that goal is not worth three days of
sampler**, and the honest read of §8 is that the marginal representation makes the fit
*possible*, not *cheap*.

**What replaces it: `docs/draw-time-calibration-plan.md`** — grade the injected σ by role at
draw time. Hours rather than days, no sampler, and it targets the same measured defect (a
constant logit-scale σ leaves fringe 1.73× underdispersed and stars 0.85× over).

✅ **Built, measured and shipped the same day (2026-08-16), in 13.3 minutes of numpy.** It
ships `[0.600, 0.375, 0.375, 0.300]` — a **2.00×** spread against §8's fitted **2.09×**, with
both **ends** inside one grid step of the fitted values (0.60898 fringe, 0.29075 star). Two
instruments sharing no arithmetic, at the same size and direction. The middle two buckets
come out lower than the fit, which is what a CRPS objective does against a variance-matching
one. **That corroborates the parking**: the gradient this line discovered was worth having
and did not need the sampler to deliver it, so what a fitted `sigma_u` still uniquely buys is
narrower than it looked — σ estimated jointly with `beta`, and a predictive that integrates
over σ's posterior instead of plugging one in.

**Two things from this session carry into that work and are the reason it is now cheaper than
when it was written:**

1. **The unit → bin σ lookup is already built and tested.** That plan's one code touch outside
   its sweep modules — "shared with the quadrature plan's step 6, built once" — is done:
   `PlayerSeasonTerm` carries `sigma_draws` as `(draws × n_sigma)` and resolves each unit's bin
   off `rho_bin`, `rehydrate_composition` passes a graded block through unflattened, and a test
   pins that seam.
2. **The fitted graded σ is an independent reference for its grid optima** — 0.60898 fringe /
   0.51013 bench / 0.46440 starter / 0.29075 star, from a *converged* fit. That plan's leading
   falsifier is "per-bucket optima come back flat at ~0.375 → the per-role miss is not
   σ-fixable"; a converged fit that is emphatically not flat is strong prior evidence against
   it, and it gives the grid a shape to be checked against rather than only a scalar.

**What stays in the repo, and why nothing is being ripped out.** `Q = 0` nests the shipped head
exactly, the block is inert when off, `make composition-quadrature-check` guards it, and the
whole thing is pinned by twelve tests. It costs nothing to leave and would cost real work to
rebuild. **To revive it**: set `stan.composition.effects.arms: [mq]` with a `label`, do §4 step
2b first, and read §8's feasibility table before budgeting.

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
— plus `stan.composition.effects.quadrature: {q, scheme}`. ~~**2b, only if step-5 cost
demands:**~~ **2b is REQUIRED for step 6 and optional for step 5** — the session-1 probe
prices the full window at ~74 h serially against the pilot's ~11.5 h; see §8: `reduce_sum`
over units, `STAN_THREADS`, `threads_per_chain`.

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

---

## 7. In-flight latent fit — what it returned

The post-blend `ps` re-attempt of §6 **did not deliver a pilot-window arm.** It wrote its
deviation table at 06:46, its Gate A probe at 14:09 UTC and the `base` control at 14:28, and
then stopped; `outputs/predictions/effects_postblend/composition_effects_metrics.csv` carries
`base` alone and no process survived. So §6's first branch — "converged, and it becomes step
4's reference posterior" — is **not** available, and the quadrature arm has no same-scale
latent posterior to be checked against yet.

What it did leave is worth recording, because it is the third unfavourable reading on the
latent route:

| fit | draws | max R̂ | min ESS bulk | divergences | wall clock |
|---|---|---|---|---|---|
| `probe/ps-one-season` | 800 | **1.25204** | **13.1** | 0 | 1,395 s |
| `effects/base` (control) | 4,000 | 1.00471 | 4,980 | 0 | 1,135 s |

⚠️ **Do not over-read the probe row.** It is 200+200 draws by design — a *timing* probe, not
a convergence attempt — so R̂ 1.25 there is not by itself evidence the pilot arm would fail.
What it does establish is that a 605-unit one-season latent fit costs 1,395 s against the
control's 1,135 s for a **five times longer** fit on **four and a half times** the rows, which
is the cost shape the whole plan is a response to.

The `base` row **is** usable and is step 5's control: post-blend, pilot window, 1000+1000,
`dense_e`, converged, per-team-game CRPS **4.264532**, season-unit CRPS **161.722** against
the marginal head (`crps_delta` **+24.765 [+18.103, +31.888]**, verdict `loses`), predictive
sd **57.51**. *(Scratch — `outputs/predictions/effects_postblend/`, outside `make docs-audit`.)*

---

## 8. Session 1 (2026-08-16) — steps 0–3 built

### Step 0 — the Q study. The falsifier did not fire, and the reason it nearly did is the finding.

Measured in numpy at the pilot window on the **2,062** (of 2,204) training units that carry a
likelihood row — 88,389 rows, median 47 per unit, p10 6, min 1. Worst per-unit error in the
marginal log-likelihood, over σ ∈ [0.1, 1.0], against a Q=141 reference that self-checks to
1e-13 against a different placement:

| placement | Q=11 | Q=15 | Q=21 | Q=25 | Q=41 | Q=61 |
|---|---|---|---|---|---|---|
| **fixed nodes** (§4's first scheme) | — | — | 9.3e+0 | — | 4.1e+0 | 2.6e+0 |
| §2's "shrunken deviation + rough information" | 5.3e+0 | 2.3e+0 | 3.6e-1 | ~1e-3 | 1e-9 | — |
| **Newton mode/curvature, inflate 1.2** | 8.3e-1 | 6.0e-2 | **2.5e-6** | 4.9e-7 | — | — |

*(Scratch. No `make` target yet — step 6 promotes it.)*

**Fixed nodes are dead**, decisively and in the direction §5 anticipated: not "Q > ~40" but
Q > 61 and still **2.6 nats** out on the worst unit. The mechanism is visible in the data —
the median unit's likelihood in `u` has an sd of **0.108** against a prior sd of ~0.42, so
the integrand is a narrow spike sitting where fixed nodes spaced O(1) apart simply are not.

**But §2's own centering rule needed Q ≈ 31, which is inside the "stop and re-read the
falsifier" boundary §4 set for the centered scheme.** It was not the integrand being nasty.
It was the rule: `c_i` from a shrunken observed deviation and `s_i` from a rough information
count is a *linearized* Laplace approximation, and it is poor exactly where `|D_i|` is large.
Replacing it with a real Newton solve for each unit's likelihood mode and curvature — still
computed once, in the driver, at fixed data — takes the same integral from Q≈31 to **Q=21 at
2.5e-6**. **The node count is a property of the placement, not of the integrand.** Had the
plan's rule been taken as the centered scheme's verdict, step 0 would have come within one
factor of two of killing an approach that works.

Three design changes fell out of the study and all three are in the shipped code:

1. **The center and scale are σ-adaptive, in closed form, inside Stan.** §2 has `c_i`, `s_i`
   as fixed data, which forces a reference σ; that reads 2.3e+1 at σ = 0.1 while the sampler
   is free to go there. Keeping the *likelihood's* mode and curvature as data and combining
   them with the current `sigma_u` as `prec = curv + 1/σ²`, `c = mode·curv/prec`,
   `s = inflate/√prec` is a differentiable change of variables costing three scalar ops per
   unit, and it holds accuracy across the whole σ range. It is **not** an inner optimization
   and does not violate §1's rule.
2. **The nodes are placed from the head's own no-fit floor** (η = the stick-breaking offset,
   α = β = 0), not from the persisted posterior. §4 step 0 proposed reading θ off the
   artifact; that works — 2.5e-6 either way — but it would give the head a silent dependence
   on the artifact it exists to replace, for no accuracy. Placement changes accuracy and
   never the estimand, so the clean provenance is free.
3. **`inflate` is an interior optimum, not a safety margin.** At Q=21: **2.5e-6** at 1.2,
   3.9e-5 at 1.4, 4.4e-4 at 1.6, and 7.6e-2 at 1.0. Too-wide nodes step over a narrow spike
   as badly as too-narrow ones miss its tails. Anyone "adding margin" here makes it worse.

**And the study produced a σ estimate, which is the first one in the project that
marginalizes the latents rather than injecting them.** Profiling `Σ_i log m_i(σ)` at fixed θ
with the half-normal(0,1) prior puts the mode at **σ = 0.4375**, with curvature implying a
posterior sd of ~**0.007**. That sits inside the five routes on record (0.375 validation-CRPS
grid, ≈0.41 calibration, 0.450 train-CRPS grid, 0.4776/0.4809 the two Stan latent fits) and
nearest the calibration route. It is a **profile at fixed θ and a shared σ, not a posterior**,
so it is a sixth route rather than a result — but it is the one route that does the integral.
*(Scratch.)*

**Two structural facts the plan did not have**, both now handled in code and pinned by tests:

- **142 of 2,204 units carry no likelihood row at all** — every one of their rows is a
  team-game's deterministic remainder. Their marginal is `∫ φ(z) dz = 1`, so they contribute
  nothing and are simply absent from the unit list rather than being zero-length entries.
- **A unit's likelihood in `u` can be genuinely flat**, and then it has no mode to find.
  Newton walks off — measured at **21.2** on a two-row synthetic unit — and the first
  implementation here made that catastrophic by flooring the curvature *up* to prior strength,
  which kept the wandered mode: at σ = 0.65 the centre landed at **6.29** with a spread of
  0.65, putting every node in the far tail and the integral **0.77 nats** wrong against a
  dense grid. The fix is to let a flat unit be flat — floor the curvature at ~0 instead, so
  `prec` collapses to the prior, the centre shrinks to ~0 and the nodes become the prior's
  own, which is the right placement for a unit the data says nothing about. The mode is also
  clipped to ±10 each iteration. `tests/test_stan_composition.py` pins both branches.
  **This bug was found by the step-3 brute-force test and by nothing else** — it is invisible
  at the pilot window, where 0 units are flat.

### Step 1 — the Stan change

`src/stan/composition_glm.stan` carries the marginal path. New data: `Q`, `gh_x`,
`gh_log_w` (`log w_q + x_q²`, folded in the driver), the unit-major ragged view
(`n_unit`, `u_start`, `u_len`, `u_bin`, `n_umap`, `u_row`), `u_center`, `u_curv`,
`gh_inflate`, `n_sigma`. `sigma_u` is `vector[H_u]` with
`H_u = Q > 0 ? n_sigma : (U_n > 0 ? 1 : 0)`, so **`Q = 0` is the shipped head exactly** and
the sampled-latent block is untouched and still in the file.

The inner sum is fully vectorized per node — `lbeta(y+a, m-y+b) - lbeta(a,b)` over all
88,389 rows at once, then `sum(segment(...))` per unit — rather than a scalar loop, so the
gradient cost is ~Q× the existing likelihood and not Q× a much slower one. It uses the
`lupmf` convention (dropping the data-only `lchoose`), which is constant across nodes and
therefore factors straight out of the `log_sum_exp`.

Integrity rejects, all in `transformed data`: `n_umap == n_lik`, `u_start` contiguous, every
likelihood row in exactly one unit and no row in two, `u_bin ≤ n_sigma`, `dispersed = 1`
(the marginal path is beta-binomial only, loudly), and **`Q > 0` with `U_n > 0` rejects** —
the two are representations of one effect and enabling both would double it silently.

### Step 2 — the driver

`stan_composition.py` gains `QuadratureTerm` (the data block), `unit_laplace` (the Newton
placement), `gauss_hermite`, `_unit_arrays`, `_floor_dispersion_by_bin`, and the constants
`Q_NODES = 21` / `GH_INFLATE = 1.2` / `CURV_MIN` / `MODE_CLIP`. `StanComposition` takes
`quadrature=`, `q_nodes=`, `gh_inflate=`, `n_sigma=`.

`PlayerSeasonTerm` still owns **predict** time under both representations — the fitted
`sigma_u` lands in the same place and the predictive integrates a fresh `z` per (unit, draw)
either way — and it now carries `sigma_draws` as `(draws × n_sigma)`, picking each unit's bin
off `rho_bin`. A scalar or one-column σ broadcasts, so the shipped injection and every
artifact written before today are unchanged arithmetic. Its new `sampled` flag is what lets
the effect be on for prediction while Stan is handed the `U_n = 0` block.

`effect_variants` returns a **`NamedTuple` (`EffectArm`)** rather than a bare tuple: it grew
from seven fields to nine, and `composition_effects` was already reading it by magic index
(`built[a][5]`). Positional unpacking and slicing still work, so no existing call site
changed meaning. Two new arms: **`mq`** (shared σ) and **`mq_graded`** (σ per `rho_bin`).

Config: `stan.composition.effects.quadrature.{nodes, inflate}` and
`stan.composition.effects.label`. The label is new and is a **build gate** — see the runbook
below for what it prevents; `composition_effects.artifact_stem` is the same device
`composition_preseason_fit` already carries.

⚠️ **Gate A's extrapolation now scales the marginal arm by ROWS ALONE.** The existing
formula takes a geometric mean of the row and *unit* scales, which is right for a latent arm
and exactly wrong here: `mq` is ~35 parameters at every window, so charging it a 5.6× unit
scale would price it as if it were `ps` — the comparison the item exists to settle.

### Step 3 — tests

Twelve new tests in `tests/test_stan_composition.py` (57 passing in that file, 1,880 in the
suite).

- **the quadrature marginal against a brute-force dense trapezoid grid**, at `n_sigma` 1 and
  2, comparing a *difference* between two parameter vectors so every dropped constant
  cancels. Deliberately not quadrature on the reference side, so the test cannot pass by
  reproducing the approximation it is checking. This is the test that found the flat-unit
  bug.
- `Q = 0` reproduces the vectorized path, and `Q > 0` **moves** the target (a nesting test
  that only checks the disabled side would pass on a block that never engaged).
- `n_sigma = 1` is the shared model in the **likelihood**, with the only difference being the
  extra half-normal prior term — stated as an exact identity rather than approximate
  equality, because a two-bin model genuinely has two priors.
- the two representations may not be enabled together (Python `ValueError` and Stan reject).
- the node placement finds a maximum for interior modes, and a clipped mode is accompanied by
  a curvature small enough that the shrunken centre is < 0.01.
- a flat unit reports mode 0 and ~zero curvature rather than a wandered mode.
- a unit with no likelihood row is absent rather than zero-length.
- a graded σ shifts each unit by its own bin, `n_sigma = 1` reproduces the scalar injection
  exactly, and a graded σ survives `rehydrate_composition` — the seam every consumer goes
  through, where a flatten would be silent because the head would still draw, and draw narrow.
- **the marginal arm reaches `dense_e` at any window** and the latent arm cannot at either,
  which is the structural claim the whole representation rests on, written as arithmetic.
- the effects ladder cannot overwrite the audited pilot record, and the config's `nodes` /
  `inflate` match what the driver defaults to.

**And the same check runs at full scale as its own target — `make composition-quadrature-check`**
([composition_quadrature_check.py](../src/models/composition_quadrature_check.py)), which the
unit suite cannot afford because the frame takes a minute to build. Production driver →
`StanComposition.stan_data` → Stan `log_prob`, against an independent numpy evaluation of the
same integral written from the **data dict alone** (so an error shared with `QuadratureTerm`
cannot cancel), on the real pilot frame: 97,587 rows, 2,062 units, `n_sigma = 4`, all 1,487
truncation rows. They agree to **5.6e-9 absolute on a target of −2,553,697.03**, and the
between-arm difference to **6.5e-9** — a relative gap of **4.9e-12**. That is floating-point
identity rather than agreement within a tolerance, and it costs two log-posterior
evaluations, so it is the cheapest possible guard on the one approximation in the whole
representation. **Run it after any change to the `Q > 0` block, and before reading a metric
off any `mq` arm.** It raises rather than warns.

It builds its rows with `composition_frame`, the same builder `composition_effects` uses,
rather than the shipped `head_frame`. The verdict does not depend on the choice — it is
checking arithmetic — but running over the rows the *ladder* fits means the check sees that
ladder's own node placement rather than a different one.

⚠️ **One trap worth writing down for whoever repeats it.** CmdStan reports `log_prob` at
**eight significant figures** by default, and this target is ~2.55e6 — so the default rounds
every value to ±0.05 and the first run of this check "failed" at a gap of 1.4e-2 that was
entirely the *reporting*. Pass `sig_figs=17`. At the synthetic scale (~200) eight figures is
1e-6 and the issue never appears, which is exactly why it would be missed.

### What is NOT done

Steps 4, 5 and 6. Specifically:

- ~~**Step 4's agreement check has no reference.**~~ ✅ **Answered, at probe scale rather than
  pilot scale.** §7 killed the planned route (no post-blend pilot latent posterior exists),
  but the 2026-08-09 Gate A probe fitted `ps` on exactly the rows this session's probe used,
  and the two representations agree to 0.4% on `sigma_u` — see the step-4 table above. A
  same-run `ps` refit is in flight to remove the cross-session caveat. What remains genuinely
  open is agreement at the **pilot** window, which needs a pilot-scale latent fit that has now
  failed to materialize twice; the case for `mq` no longer depends on getting one.
- **No pilot ladder, no full window, nothing persisted.** The *persistence* path turns out to
  need no change, which was worth checking rather than assuming: `posteriors._thinned` slices
  `sigma_u_draws` on the draw axis and `rehydrate_composition` assigns it through, so a
  `(draws × n_sigma)` block survives both and `PlayerSeasonTerm.shift` resolves each unit's
  bin off `rho_bin` at predict time. A test now pins that seam, because a flatten there would
  be silent — the head would still draw, and draw narrow. What is genuinely untested is the
  whole round trip through `make posteriors`, which cannot happen until an `mq_graded` arm
  has been fitted.
- ~~**`mq_graded` has never been fitted**, so §2's role-graded motivation is still a
  motivation.~~ ✅ **Fitted at probe scale and the gradient is real** — 0.609 fringe to 0.291
  star, see the step-4 table. What is still open is whether grading *buys* anything: this
  probe fits and does not score, so Gate P2 (season-unit CRPS and the per-role PIT tail) needs
  the pilot ladder.

### Step 4 — the probe-scale fits. `mq` converges, and it reproduces the latent σ.

One training season of the pilot window (2021-22: 26,039 rows, 605 units), 200 + 200 × 4
chains, **metric forced** rather than sized — `choose_metric` would put a 200-warmup run on
`diag_e` and the ladder runs at warmup 1000, so a sized probe would have measured the Q-fold
cost under a metric the ladder will not use. *(Scratch — no `make` target; step 5 supersedes
it.)*

| arm | metric | wall clock | max R̂ | min ESS bulk | divergences | treedepth sat. | `sigma_u` |
|---|---|---|---|---|---|---|---|
| `base` (control) | `dense_e` | **124 s** | 1.0172 | 526 | 0 | 0 | — |
| **`mq`** | `dense_e` | **2,713 s** | **1.0138** | **719** | **0** | **0** | **0.47927** |
| **`mq_graded`** | `dense_e` | **2,615 s** | **1.0069** | **683** | **0** | **0** | 0.46856 (mean) |
| `ps` (the sampled latent) | `diag_e` | 1,468 s | **1.1317** | **25** | 0 | 19 | 0.48095 |

**The gate that matters passed: `mq` converges and `ps`, on the same rows in the same run,
does not.** R̂ 1.0138 against 1.1317, min ESS 719 against 25, 0 treedepth-saturated draws
against 19. That is the bar the latent arm has now failed three times — 1.0948 / ESS 35 in
2026-08-09's Gate A, 1.25204 / ESS 13 in §7's post-blend re-attempt, and 1.1317 / ESS 25
here — and clearing it is the entire reason this representation exists. The `base` control
reproduces its own recorded Gate A figures (124 s against 125 s, R̂ 1.0172 against 1.0172),
so this is the same machine those readings came from.

**And step 4's agreement check is delivered — same run, same rows, same seed.** `mq` puts
`sigma_u` at **0.47927** and `ps` at **0.48095**: a **0.35%** difference between two
representations of one posterior computed by completely different machinery, one integrating
605 latents out and the other sampling them. It also lands inside the pair 2026-08-09
recorded (0.4776 non-centred, 0.4809 centred) and inside the five routes on record.
⚠️ Read it for what it is: `ps` is *under-converged*, so this corroborates the **size** rather
than validating either fit against the other. The correctness claim rests on
`make composition-quadrature-check`, not on this.

**And the graded arm is the session's second real result: σ grades by role, monotonically,
and §5's "`mq_graded` ties `mq` everywhere" falsifier did not fire.**

| bin | role | fitted σ |
|---|---|---|
| 1 | fringe | **0.60898** |
| 2 | bench | 0.51013 |
| 3 | starter | 0.46440 |
| 4 | star | **0.29075** |

A **2.09×** spread, fringe above star, against a shared-arm posterior sd of 0.0168 — so the
end bins are roughly nineteen posterior sds apart and this is not resolution noise. §2
predicted "a ~2× gradient in what σ wants, the same axis and direction as the fitted per-game
ρ", from a completely different instrument (per-role PIT of the *injected* constant). The
fitted per-game ρ spread on this head is **1.91×** (0.14019 fringe → 0.0735171 star). Two
independent routes to the same gradient, at the same size, in the same direction.

`mq_graded` also converges *better* than `mq` (R̂ 1.0069 against 1.0138) at the same cost
(21.1× against 21.9× — the difference is machine contention, not the model), so the grading
is free. ⚠️ **What this does NOT say is that grading helps.** It says σ genuinely differs by
role and the head can estimate that. Whether it improves season-unit CRPS or the fringe PIT
tail is Gate P2, which needs the pilot ladder's scoring pass — this probe fits and does not
score. The shared arm's 0.47927 sits between bins 2 and 3, which is what a mean of a graded
quantity looks like.

⚠️ **`mq` is 1.85× the wall clock of `ps` at this scale, and that is the honest cost line.**
The marginal arm is not cheaper than the latent one on one season — it is cheaper *at the
window that matters*, because its cost scales with rows alone while the latent arm's scales
with rows **and** units, from 605 here to 2,204 at the pilot and 12,307 at the full window.
Quoting the 1.85× without that is as misleading as omitting it.

**The cost is Q, and almost exactly Q.** 2,713 s against 124 s is **21.9×** at `Q = 21` — so
the per-node overhead beyond the likelihood itself is a few percent, and the two levers below
are worth a few percent rather than a factor. That is the trade stated plainly: 21.9× the
shipped arm's gradient, against a parameter block that never converged at any cost.

⚠️ **Both arms logged non-fatal `beta_binomial_lpmf` exceptions during warmup, and this is
NOT new.** `base` threw **28**, from the *shipped vectorized likelihood* itself
(`beta_binomial_lupmf`, lines 369–370); `mq` threw **20**, from `beta_binomial_log_tail_mass`
on the truncation rows. Same condition at two call sites — a beta shape parameter underflowing
to exactly 0 at extreme warmup η — and Stan rejects the proposal and continues, which is why
both fits converged with 0 divergences. The marginal path hit it *less often* than the head
that ships. It would have been easy to read this as the new block being unsafe and to "fix" it
by clamping η, which would have changed a converged audited head's target on the basis of a
warning it was already producing.

The node offset is also **structurally bounded**, which is worth knowing before anyone reaches
for a clamp: `u_s = inflate/√(curv + 1/σ²) ≤ inflate·σ`, so the spread is bounded by the prior
scale itself, and `|u| ≤ MODE_CLIP + √2·inflate·σ·max|x_q|` — about `10 + 9.4σ` at Q = 21. It
cannot run away with σ.

### Is it feasible on wall time? Yes at the pilot, and NOT at the full window without 2b

**Per-fit wall clock is the wrong metric and it flatters `ps`.** The currency is effective
samples, and the two arms are not producing the same thing:

| arm | ESS per draw | ESS per second | seconds per effective sample |
|---|---|---|---|
| `mq` | **0.898** | **0.2648** | **3.78** |
| `ps` | 0.031 | 0.0171 | 58.34 |

`mq` delivers **15.5×** the effective samples per second. Put the other way: `ps` needs
**28.6×** more draws to reach `mq`'s ESS, which is **11.6 h** against `mq`'s **0.75 h** — and
that is the optimistic reading, because ESS does not scale linearly in a chain that is stuck.
The 1.85× per-fit ratio compares 800 nearly-independent draws against 800 draws worth 25.

**A consequence specific to `mq`: the sampling half of the budget is cheap to cut.** At 0.898
ESS per draw, 4 chains × 300 samples clears the 400-ESS bar with room to spare. Warmup is the
part that is floored — `choose_metric` needs ≥ 20 × params for `dense_e`, so 620 for `mq` and
680 for `mq_graded`. A **700 + 300** budget is therefore justified rather than a corner cut,
and `ps` cannot use the same trick: it needs *more* draws, not fewer.

Extrapolating by rows with this head's own 1.63× correction (`mq`'s parameter block does not
grow, so rows are the only axis):

| window | rows × | 1000+1000, serial | 700+300, serial | 700+300 + `reduce_sum` ≈3× |
|---|---|---|---|---|
| pilot (97,587 rows) | 3.75× | 23.0 h | **11.5 h** | 3.8 h |
| full (631,158 rows) | 24.2× | 148.9 h | 74.4 h | **~25 h** |

**The pilot window is comfortably feasible** — one arm overnight at 700+300, and it is the
only representation that yields a usable posterior there at all.

⚠️ **The full window is not, serially, and §3's "the full-window σ is only credible under the
marginal representation" needs that qualification.** 74 h is three days for one arm. So
**§4 step 2b (`reduce_sum` over units, `STAN_THREADS`) is required for step 6, not optional**
— the per-unit loop is exactly the shape it slices over, and this machine has 14 cores
(10 performance), so 4 chains × 3 threads fits. ⚠️ The ≈3× in that column is an *assumption*,
not a measurement; measure it before planning a full-window run on it. Dropping to `Q = 15`
at `inflate = 1.4` (6.4e-4 nats, still 150× inside tolerance) is a further 29% if needed.

Two caveats on the extrapolation, both pointing the same way. It assumes `mq`'s **mixing**
holds at larger windows — plausible, since the parameter count is constant and `dense_e` is
retained, but unmeasured, and if ESS per draw degrades then the samples cut is unsafe. And the
1.63× correction is this head's measured row-superlinearity, which may understate a 24× jump.
**The pilot run should report its own ESS per draw before anyone budgets a full-window fit.**

The comparison at the full window is not 74 h against `ps`'s extrapolated ~38 h, either. It is
74 h for something that converges against an arm carrying **12,307** latents that has not
converged at **605**.

### ⚠️ The pilot ladder is NOT affordable as specified, and Gate A will say so

Extrapolating the probe by rows (3.75×) and iterations (5×) with this head's own 1.63×
correction, `stan.composition.effects.arms: [base, mq, mq_graded]` prices at roughly **52 h**
against a 24 h budget, and Gate A aborts. A single-arm run — `arms: [mq]` — prices at **~23 h**
and passes, narrowly.

**But that extrapolation is probably pessimistic on one axis, and the evidence is this head's
own.** `stan_composition.sweep` records that doubling chain length cost it **33%, not 100%**,
because longer warmup adapts a better step size and buys fewer leapfrog steps per iteration.
Gate A's `iter_scale` is linear and takes no account of that. If the 5× iteration increase
costs ~2.2× rather than 5×, `mq` at the pilot window is ~10 h and the three-arm ladder ~21 h.
**That is a documented property of this head, not a guess — but it is also not a measurement
of *this* arm, so do not bake it into the gate.** Run the arms one at a time and let each one
measure itself; `_flush` merges by arm name, which is exactly what makes that safe.

### Cost levers, if the probe says the ladder is too dear

§4 step 2b names `reduce_sum` as the escape hatch. Two cheaper ones came out of writing the
model block and are recorded here so nobody has to re-derive them. **Neither is implemented**
— the shipped block is the straightforward one, because a tested correct version is worth
more than an untested fast one, and `make composition-quadrature-check` makes either safe to
try in minutes.

1. **Hoist the per-unit terms out of the node loop.** `log(sqrt(2)·s_i)` and
   `normal_lpdf(u_q | 0, σ_i)`'s `−log σ_i` do not depend on `q`, so they factor out of the
   `log_sum_exp` — `log_sum_exp_q(a_q + c) = c + log_sum_exp_q(a_q)`. They must still be
   *added* (both depend on `sigma_u`), but once per unit rather than once per unit per node.
   That removes `n_unit × Q` scalar `normal_lpdf` calls — 43,000 per gradient at the pilot —
   for an exactly identical target.
2. **Replace `segment` with a cumulative sum.** The per-unit reduction is currently
   `sum(segment(term, u_start[i], u_len[i]))`, which copies `u_len[i]` autodiff variables per
   unit per node — ~1.9M copies per gradient. `cumulative_sum(term)` once per node and
   differencing the endpoints is O(1) per unit. ⚠️ Check the numerics before adopting: the
   partial sums reach ~3e5 while a unit's own sum is ~1e2, so the differencing gives up about
   five significant figures. At the terms' observed magnitudes the induced error is ~3e-11,
   far under the 2.5e-6 the node count is sized for — but that margin is an *estimate*, and
   the check target will say for certain.

The honest framing is that the trade was never "the marginal arm is cheap." It is `Q`
evaluations of the likelihood per gradient against a parameter block that never converged —
so the question the probe answers is whether `Q = 21` under `dense_e` beats 2,204 latents
under `diag_e`, not whether it is free.

### The runbook for step 5, and the trap in it

⚠️ **`make composition-effects` writes `outputs/predictions/composition_effects_*.csv`, and
`make docs-audit` re-derives the 2026-08-09 pilot `base` arm's per-team-game CRPS
(**4.45614**) from that file.** `_flush` merges by *arm name*, so a quadrature round writing
its own `base` there would answer a different question under that figure's name and fail the
audit on a bookkeeping change — the same failure `stan_composition_*.csv` has its own target
to avoid, and the reason `stan.composition.preseason.label` exists one module over.

That guard did not exist here and now does. `stan.composition.effects.label` suffixes every
artifact this target writes; the shipped value is `""`, which reproduces the 2026-08-09 paths
exactly, and a test pins both halves. So step 5 is:

```yaml
stan.composition.effects.label: quadrature      # -> composition_effects_quadrature_*.csv
stan.composition.effects.arms:  [base, mq, mq_graded]
```

then `make composition-effects`. `base` first because it is cheap and exercises the whole
pipeline — scoring, the season-unit comparison, the artifacts — before hours are spent, which
is `make stan`'s "cheapest first" rule; each arm checkpoints as it lands.

Gate A will price the run off a one-season `mq` probe. **Its extrapolation now scales the
marginal arm by rows alone**, deliberately: `mq` is ~35 parameters at every window, so
charging it the 5.6× unit scale the latent arm is charged would price it as if it were `ps`
— which is the comparison this whole item exists to settle.

Read §5's gates in this order, and do not skip the first: **`mq` must converge.** That is the
bar `ps` failed twice, and it is the only reason any of this exists. Everything after it is a
metric comparison that means nothing if the chains did not mix.
