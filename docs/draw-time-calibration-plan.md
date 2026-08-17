# Draw-time calibration plan — grading the injected σ by role

**Opened 2026-08-16.** One calibration of a **simulation-time draw input** — no Stan, no
likelihood change, and the knob nests the shipped behaviour exactly: the composition's
injected per-(player, season) σ, graded by role.

**This doc is not itself in `make docs-audit`, but its results now are**, through `README.md`:
the step got a `make` target the same day it was opened, so `_role_sigma_claims` in
`src/docs_audit.py` re-derives nineteen of §9's figures from `minutes_role_sigma.csv` and
fails the build on a mismatch. Adding this file to `_DOCS` is a one-line change whenever
someone wants the tables here audited directly rather than through their README twin. The
**scratch** table under §The evidence predates the target and is superseded by §9 — it is kept
only because the two agree, which is a wiring check on the module that replaced it.

## ✅ STATUS: BUILT, MEASURED AND SHIPPED, 2026-08-16

`make minutes-role-sigma` → `outputs/predictions/minutes_role_sigma.csv`. Module:
[src/models/minutes_role_sigma.py](../src/models/minutes_role_sigma.py), ~14 min, no
sampler, no head refitted. **`sim.minutes.player_season_sigma_by_role = [0.600, 0.375, 0.375,
0.300]`** over the composition's own `rho_bin` (fringe → star), a **2.00×** spread. All three
steps below are done and §9 is the readout.

**The one thing that was NOT part of the plan and was done first**: `minutes_unification`'s
`SHIPPED_PS_SIGMA` still read **0.450** while config read 0.375 — a fallback for a missing
key, so nothing misbehaved, and a trap for exactly the reader about to change the injection.
It is 0.375 now and a test pins it against `configs/default.yaml` so the two cannot drift
again.

### The nesting, stated once

An **absent** `sim.minutes.player_season_sigma_by_role` leaves `player_season_sigma` in
charge and reproduces the pre-2026-08-16 draw **bit-for-bit** — the nesting is the config
schema, not a code branch, and deleting four lines from `configs/default.yaml` is the exact
rollback. That is checked rather than asserted: `make minutes-unification` re-ran through the
refactored draw path and its artifact was byte-identical to the run before it.
`minutes_unification.shipped_injection` is the single resolver every consumer comes through
(`sim/season.py`, `model_cards.py`, the gate itself), so there is no second place that can
hold a different answer.

## ⬆️ PROMOTED 2026-08-16: this is now the line of work, not the sibling

This was written as the injection-side twin of `docs/composition-quadrature-plan.md`'s graded
*fitted* σ, to be "superseded gracefully if the quadrature fit ships". **The quadrature line was
parked the same day** — on cost against reach rather than a failed gate — so the supersession
will not happen and this is the plan that runs. The parking rationale is in that document's
header; the short version is that a fitted σ costs ~74 h at the full window and still leaves
four other draw-time inputs the simulator is *given* rather than fits, so it does not deliver
the all-posterior simulator that would have justified the fitting time.

**Two things the quadrature session leaves behind make this cheaper than when it was written,
and one of them changes a falsifier:**

1. ✅ **The unit → bin σ lookup is BUILT AND TESTED.** "The one code touch outside the sweep
   modules … shared with the quadrature plan's step 6 and built once" — it is built.
   `PlayerSeasonTerm` carries `sigma_draws` as `(draws × n_sigma)` and resolves each unit's bin
   off `rho_bin`; a one-column or scalar σ broadcasts, so the shipped injection is unchanged
   arithmetic; `rehydrate_composition` passes a graded block through unflattened; and
   `tests/test_stan_composition.py` pins both the shift and that rehydration seam. Step 3's
   plumbing is therefore mostly done and what remains is the config key and the sweep.
2. ✅ **A converged fitted graded σ now exists as an independent reference** — **0.60898**
   fringe / **0.51013** bench / **0.46440** starter / **0.29075** star, from an `mq_graded` fit
   at probe scale with R̂ 1.0069 and min ESS 683 (`docs/composition-quadrature-plan.md` §8).
   ⚠️ **This weakens, but does not settle, the leading falsifier below.** "Per-bucket optima
   come back flat at ~0.375" is now unlikely on prior evidence: a *fitted* σ on the same
   `rho_bin` axis spans **2.09×** and is monotone in role. It is not proof — the fit is one
   season, and a fitted likelihood-scale σ and a draw-time calibration σ answer different
   questions (what the data support against what makes the season-unit predictive calibrated).
   **Treat it as a shape to check the grid against, never as the values to ship**: reading the
   fitted numbers into the config without running the grid would import a probe-scale estimate
   into a production draw input, which is the sort of shortcut this repo has been bitten by.

(A second half drafted alongside this — a role × season-stage term in the availability
layout — was moved the same day to `docs/potential-to-dos.md` §15 and **parked**: the 2026
offseason draft-lottery reform targets the tanking incentive that drives the dynamics it
would calibrate, so no data season represents the regime it would ship into. The entry
carries the full caveat and the measurement that would un-park it.)

---

## The plan — grade `sim.minutes.player_season_sigma` by role

### The evidence (scratch, 2026-08-16 — ⚠️ SUPERSEDED by §9's audited table)

Kept because the two agree to rounding, which is the wiring check on the module that
replaced it: §9 reads 1.7276 / 1.1834 / 0.9341 / 0.8601 against the 1.73 / 1.17 / 0.93 / 0.85
below, on the same 456 / 275 / 191 / 189 units.

Season-unit calibration of the shipped injection (σ = 0.375), validation, 1,111 units,
through the `rehydrate → predict_samples → season_totals` path, bucketed by the head's own
`rho_bin`:

| role | n | σ=0 sd_ratio | shipped sd_ratio | shipped PIT tails lo/hi (nominal .05) |
|---|---|---|---|---|
| 1 fringe | 456 | 4.42 | **1.73** | .160 / .066 |
| 2 bench | 275 | 4.30 | **1.17** | .193 / .040 |
| 3 starter | 191 | 3.98 | **0.93** | .089 / .042 |
| 4 star | 189 | 3.83 | **0.85** | .048 / .011 |

The raw deficit is role-flat; the **constant logit-scale σ lands unevenly on the minutes
scale** — fringe still 1.73× underdispersed, stars overdispersed at 0.85×. A ~2× σ gradient
falling with role, the same axis and direction as the fitted per-game ρ (0.140 → 0.074).
Known residue this part does **not** fix: the fringe/bench PIT tails are asymmetric ("the
role evaporated" left-skew), which is a shape question, recorded in the quadrature plan's
falsifiers.

### Design

`u_i ~ N(0, σ_role(i))`, role = the unit's `rho_bin` (assigned by the recipe's transform,
constant within unit). Config: a new optional `sim.minutes.player_season_sigma_by_role`
mapping bin → σ; **absent, the scalar key behaves exactly as today** — the nesting is the
config schema itself. `minutes_unification.rehydrate_composition` and the `PlayerSeasonTerm`
path gain the unit → bin lookup (the same touch the quadrature plan's step 6 names — build
it once, both consumers use it).

### Steps

1. **Extend the grid machinery to a σ vector.** `player_season_effect_sweep` and
   `estimate_sigma_on_train` already re-run drawn effects through the head's own allocation;
   the change is a per-unit σ lookup before the draw. Sweep per-bucket grids
   **coordinate-wise** (the buckets couple weakly through the team constraint — a fringe
   `u` moves starters' minutes at the margin — so run one refinement pass after the first
   sweep and confirm the optima are stable).
2. **Estimate on the fitting half, confirm on validation** — the shipped scalar's own
   precedent: 0.450 was the train-grid optimum pre-blend, and σ moved to 0.375 when the two
   grids agreed post-blend. Ship rule: per-bucket train and validation optima agree (same
   grid step); a bucket where they disagree keeps the shared 0.375.
3. **Ship + instruments.** Wire the config key; re-run the per-role PIT as the readout and
   **promote it to a `make` target** (shared deliverable with the quadrature plan — whichever
   lands first builds it, and the table above stops being scratch). Re-read two recorded
   trade measurements at the graded values: the subset teammate-coupling dilution
   (−0.0509 → −0.0388 at σ = 0.450, recorded in `docs/potential-to-dos.md` §2 — grading
   moves it per-bucket) and the season-unit gap to the marginal head (the shipped −5.91
   CRPS win must not regress).

### Gates

- Fringe and bench sd_ratio move toward 1 and their low tails toward nominal, **without**
  degrading starter/star (whose fix is σ down — the grid finds that for free).
- Pooled season-unit CRPS vs the marginal head: no regression from the shipped win.
- Team-sum error stays exactly 0 (the constraint is untouched by construction; assert it
  anyway).

### Falsifiers

- **Per-bucket optima come back flat at ~0.375** → the per-role miss is not σ-fixable (it
  is the skew residue); record the null and leave the scalar. ⚠️ Now unlikely on prior
  evidence (see the promotion note above: a converged fitted σ spans 2.09× on this axis), so
  a flat result is *more* interesting rather than less — it would mean the likelihood and the
  season-unit predictive want different things, which is worth a paragraph and not a shrug.
- **Coordinate-wise sweeps do not stabilize** → the team-constraint coupling is stronger
  than expected; the honest instrument is then a small joint grid over (fringe, star) with
  the middle interpolated, before concluding anything.
- ~~If the quadrature fit ships a graded `sigma_u_draws`, this part's constants retire~~ —
  **withdrawn 2026-08-16**: the quadrature line is parked, so nothing supersedes these
  constants and the comparison runs the other way. The fitted values are a *reference* for the
  grid optima, not a replacement for them.

---

## Cost and blast radius

Hour-scale compute (grid sweeps through the existing draw-and-score machinery), no sampler,
no shipped head refitted, no audited record touched. The one code touch outside the sweep
modules — the unit → bin σ lookup in `rehydrate_composition` — is shared with the
quadrature plan's step 6 and built once.

⚠️ **The "no audited record touched" line held for the measurement and not for the ship.**
Grading the injection changes what `sim/season.py` draws, so the season tensors, the strategy
sweep and the README figures quoting them all go stale the moment the config key appears. The
chain (`make simulate-season bracket draft-sim strategy-sweep`) was re-run in the same
session, with the pre-graded `strategy_*.csv` and `sim_season_gate_a.csv` retained first —
`docs/preseason-plan.md` P5 lost a whole attribution by overwriting them.

---

## 9. Results — 2026-08-16

`make minutes-role-sigma`, validation 1,111 player-seasons and the last two training seasons'
1,145, both through the `rehydrate → predict_samples → season_totals` path at 1,000 posterior
draws.

### Step 1 — the shared-σ grid, scored by bucket

The cheap first answer, and the one that decides whether the rest is worth running: score
each bucket's own units along the same grid, every bucket moving together. Season CRPS
minutes.

| σ | train pooled | fringe | bench | starter | star | | validation pooled | fringe | bench | starter | star |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.150 | 117.633 | 90.351 | 157.198 | 155.014 | 101.460 | | 125.254 | 88.889 | 172.047 | 146.896 | 123.035 |
| 0.225 | 112.556 | 86.941 | 149.392 | 147.338 | 98.029 | | 119.781 | 85.169 | 163.692 | 141.714 | 117.230 |
| 0.300 | 109.551 | 83.991 | 144.657 | 143.206 | **97.959** | | 116.622 | 82.184 | 157.681 | **140.580** | **115.756** |
| **0.375** | **108.469** | 81.926 | **142.585** | **141.959** | 100.516 | | **115.535** | 80.144 | 154.749 | 142.049 | 117.069 |
| 0.450 | 108.834 | 80.624 | 142.827 | 142.742 | 104.593 | | 116.181 | 79.056 | **154.250** | 145.687 | 120.543 |
| 0.525 | 110.271 | 79.841 | 144.596 | 145.388 | 109.787 | | 117.935 | **78.572** | 155.264 | 150.782 | 125.399 |
| 0.600 | 112.606 | **79.792** | 147.532 | 149.052 | 115.846 | | 120.749 | 78.715 | 157.791 | 157.156 | 131.475 |
| 0.675 | 115.523 | 79.946 | 151.506 | 153.727 | 122.450 | | 124.144 | 79.141 | 161.217 | 164.294 | 138.208 |
| 0.750 | 118.923 | 80.552 | 155.931 | 158.961 | 129.579 | | 128.026 | 79.955 | 165.176 | 172.065 | 145.447 |

**The pooled column reproduces the shipped scalar on both halves** — train 108.469 at 0.375
against `minutes_unification.csv`'s own 108.4687 — which is the wiring check that matters:
the graded machinery run at a flat vector is the scalar estimator.

**The per-bucket columns do not agree with it or with each other.** Fringe wants 0.600 on
train and 0.525 on validation; star wants 0.300 on both. That is the finding, and it is
visible before any coordinate descent.

**The fringe curve is shallow and the star curve is not.** Fringe moves 79.79 → 79.95 between
0.600 and 0.675 on train (0.2%), while star moves 97.96 → 115.85 between 0.300 and 0.600
(18%). So the confident half of the gradient is the *star* end, and the fringe value is the
one a later re-measurement is most likely to move.

### Step 2 — coordinate-wise, and the buckets turn out to be uncoupled

| split | pass-2 optimum | stable? | moved off the shared-grid start? |
|---|---|---|---|
| train | **0.600 / 0.375 / 0.375 / 0.300** | yes | **no** |
| validation | 0.525 / 0.450 / 0.300 / 0.300 | yes | **no** |

Both halves were stable on the second pass, and — the stronger result — **neither coordinate
search moved a single bucket off its marginal-profile starting point.** The plan budgeted a
refinement pass for the team constraint coupling a fringe `u` to a starter's minutes; at this
grid resolution that coupling does not reach the optimum. The second falsifier ("coordinate
sweeps do not stabilize") did not fire, and the joint (fringe, star) grid it would have
called for is not needed.

### The ship rule, applied

| bucket | train | validation | gap | ships |
|---|---|---|---|---|
| 1 fringe | 0.600 | 0.525 | 1 step | **0.600** |
| 2 bench | 0.375 | 0.450 | 1 step | **0.375** *(unchanged)* |
| 3 starter | 0.375 | 0.300 | 1 step | **0.375** *(unchanged)* |
| 4 star | 0.300 | 0.300 | 0 | **0.300** |

All four agree to a grid step, so all four ship their **train** value — selection reads the
fitting half, validation confirms. **Two of the four do not move**, which is worth saying
plainly: the shipped change is fringe 0.375 → 0.600 and star 0.375 → 0.300, and the middle of
the roster keeps the scalar it already had.

### Cross-check against the fitted σ — the ends agree, the middle does not

| bucket | shipped here (draw-time grid) | fitted in Stan (`composition-quadrature-plan.md` §8) |
|---|---|---|
| fringe | **0.600** | 0.60898 |
| bench | 0.375 | 0.51013 |
| starter | 0.375 | 0.46440 |
| star | **0.300** | 0.29075 |
| spread | **2.00×** | 2.09× |

Two instruments sharing no arithmetic — a validation-scored CRPS grid over a plugged-in
constant, and a marginalized hierarchical posterior at probe scale — land the **ends** within
a grid step of each other and the spread within 5%. The middle two buckets are lower here,
which is what a CRPS objective does against a variance-matching one: CRPS rewards sharpness,
so it settles below the value that would put `sd_ratio` at exactly 1. That is visible
directly in the readout below, where the shipped starter and star buckets sit slightly
*under*-dispersed rather than at 1.0.

### Step 3 — the readout, on validation

| arm | role | n | CRPS | predictive sd | sd_ratio | PIT KS | tail lo | tail hi |
|---|---|---|---|---|---|---|---|---|
| shipped scalar | pooled | 1,111 | 115.5349 | 187.81 | 1.1922 | 0.1105 | 0.1341 | 0.0495 |
| **graded** | pooled | 1,111 | **114.8514** | 196.21 | **1.1446** | **0.0870** | **0.1044** | 0.0396 |
| shipped scalar | fringe | 456 | 80.1443 | 102.76 | 1.7276 | 0.1925 | 0.1535 | 0.0724 |
| **graded** | fringe | 456 | **79.0510** | 139.29 | **1.2757** | **0.1358** | **0.0833** | 0.0351 |
| shipped scalar | bench | 275 | 154.7490 | 231.00 | 1.1834 | 0.1964 | 0.1891 | 0.0400 |
| graded | bench | 275 | 154.7369 | 233.86 | 1.1734 | 0.1900 | 0.1745 | 0.0436 |
| shipped scalar | starter | 191 | 142.0493 | 274.29 | 0.9341 | 0.0740 | 0.0890 | 0.0419 |
| graded | starter | 191 | 142.0532 | 274.99 | 0.9395 | 0.0692 | 0.0890 | 0.0471 |
| shipped scalar | star | 189 | 117.0694 | 242.80 | 0.8601 | 0.0719 | 0.0476 | 0.0159 |
| **graded** | star | 189 | **115.7028** | 199.14 | **1.0444** | 0.0833 | 0.0688 | 0.0370 |

**Gate 1 — fringe and bench move toward 1 without degrading starter/star: PASSES, with one
blemish recorded rather than smoothed.** Fringe is the whole result: `sd_ratio` 1.7276 →
**1.2757**, its low PIT tail 0.1535 → **0.0833** against a nominal 0.05, and PIT KS 0.1925 →
0.1358. Bench barely moves because bench ships the value it already had — its 0.01 is draw
noise, not an effect. Starter is *identical* for the same reason. Star's `sd_ratio` improves
markedly, 0.8601 → **1.0444** (over-dispersed to slightly under), and its CRPS falls 1.37
minutes. ⚠️ **Star's PIT KS drifts the wrong way, 0.0719 → 0.0833**, and that is the
CRPS-versus-calibration trade named above: the grid chose 0.300 on CRPS where a
variance-matching target would have chosen something nearer 0.32.

**Gate 2 — no regression against the marginal head: PASSES and improves.** On the 742
player-seasons both heads cover, the composition's season CRPS goes 130.6918 → **130.1526**
against the marginal head's 136.6030, so the recorded win widens from **−5.9112 [−10.3858,
−1.5014]** to **−6.4504 [−10.9991, −2.0419]** — an interval still clear of zero.

**Gate 3 — the team constraint: PASSES exactly.** Team-season total sd across draws is **0**
at the graded vector, asserted in code rather than assumed. A per-unit shock re-allocates
minutes *within* a team-game, so grading it cannot reach the constraint; the assertion is
there because "cannot" is what a wiring bug looks like from the inside.

**And the second recorded trade re-read, which step 3 also owes: grading recovers about a
quarter of the injection's teammate-coupling dilution.** On the 963 single-team validation
player-seasons, mean pairwise teammate correlation reads **−0.0504** un-injected, **−0.0392**
at the shared σ and **−0.0422** graded, against the **−0.0664** a fixed team sum forces. So
the dilution goes 0.0112 → **0.0082**, **27%** of it back, from a change made for a different
reason — the buckets whose σ *fell* dilute less, and stars are what a stack is usually built
around. `docs/potential-to-dos.md` §2 carries the entry; the trade is still unpriced in the
contest unit.

**The first falsifier did not fire.** Per-bucket optima did not come back flat: the spread is
2.00× and monotone in role, on both halves independently.

### What this is worth, and what it is not

Pooled, it is **0.68 CRPS minutes on 1,111 player-seasons** — about 0.6%. The case for it is
not the pooled number: it is that a single σ was missing in **both directions at once**, and
the bucket it was worst for is the one the marginal minutes head cannot even score. Fringe
player-seasons are 41% of the composition's validation units and include the 369 rows the
`≥ 200 prior minutes` filter drops, so they are exactly the population the composition ships
*for*.

The residue this does not touch is unchanged and still recorded: the fringe and bench PIT
tails remain asymmetric (0.0833 low against 0.0351 high after grading), which is the "the
role evaporated" left-skew. A symmetric Gaussian `u` narrows it and cannot shape it. That is
a skewness question, not a σ question, and it stays in the quadrature plan's falsifiers.

### The chain re-run, and what it did downstream

`make simulate-season bracket draft-sim strategy-sweep`, with the pre-graded artifacts kept
so the delta is a paired read rather than a memory. **Gate A improves slightly on every row
it touches and nothing regresses:**

| Gate A row | 2022-23 pre → post | 2023-24 pre → post |
|---|---|---|
| `season_total_dk` MAE | 360.9636 → **360.7979** | 373.6643 → **373.6347** |
| `season_total_dk` bias | −11.0438 → **−9.9523** | −54.1581 → **−52.5111** |
| `season_total_dk` CRPS | 251.2897 → **250.6781** | 257.2838 → **256.9231** |
| `season_minutes_spread` CRPS | 281.7443 → **281.3661** | 270.9916 → **270.7433** |
| `season_minutes_spread` conditional sd | 270.86 → 272.92 | 267.02 → 272.26 |

The dk_pts season total moves by less than 0.2 MAE, which is the right size: σ buys season-
level *spread*, and the bonus is the only place spread reaches the mean. The bias improvement
of ~1.1 and ~1.6 dk_pts is the larger effect and runs in the direction the recorded
season-total bias has been stuck in.

⚠️ **Two diagnostics moved and they did not move together**, which matters for
`docs/sim-inputs-plan.md` item 2: block inflation 1.7285 → **1.8104** (2022-23) toward its
2.4167 target while game-level dispersion 7.4985 → **7.5867** moved *away* from 4.7163.
Raising a bucket's season-constant variance adds serial structure and per-game spread at
once, so a σ change cannot separate them — which is the argument for that item's AR(1), a
mechanism that redistributes variance rather than adding it. That item's premise is
re-measured there.

### ⚠️ The contest layer neither confirms nor contradicts this, and that was expected

`strategy_shipped.csv`'s two headline numbers both moved — simulated lift at the 600k
0.230387 → **0.236915**, realized 0.197293 → **0.199380** — and **neither is evidence for the
grading.**

The **simulated** side is not a comparison across these two runs at all. Gate C's `rho` is
solved by bisection *per arm* against that arm's own realized skill gap, and it moved
0.362309 → **0.358192** (2022-23) and 0.311690 → **0.317198** (2023-24). The two runs
therefore score in different worlds — the caveat `README.md` §2 already carries about
`sim_lift`, firing exactly as documented.

The **realized** side *is* comparable, because both arms are scored against the same box
scores. It is a wash:

| cell | pre → post | Δ |
|---|---|---|
| 2022-23 600k | 0.263546 → 0.254543 | −0.009003 |
| 2023-24 600k | 0.131040 → 0.144217 | **+0.013177** |
| 2022-23 20k | 0.333412 → 0.373263 | **+0.039850** |
| 2023-24 20k | 0.405463 → 0.318890 | −0.086574 |
| 2023-24 88k | −0.121943 → −0.101281 | **+0.020662** |

**4 of 10 cells improve, at a mean of −0.0027**, and the +0.002087 the 600k headline shows is
one season up and one down. The `adp` control moved **exactly 0.000000**, as it must — an ADP
board does not read our tensor — so unlike the preseason block's counterfactual there is no
"the world got easier" story to rule in or out either.

**This is the expected result, not a disappointment, and it does not weaken the ship.** The
realized readout has N = 2 seasons and `README.md` §3 already records that it "selected
nothing"; the graded σ is worth 0.68 CRPS minutes pooled at the season unit, which is far
under what a two-season contest readout can resolve. The decision was taken at the unit it was
measured at, and the contest re-run is here to keep the artifacts consistent rather than to
adjudicate.
