# Draw-time calibration plan — grading the injected σ by role

**Opened 2026-08-16.** Not in `make docs-audit`; scratch figures are labelled and earn audited
status only when their step gets a `make` target. One calibration of a **simulation-time draw
input** — no Stan, no likelihood change, and the knob nests the shipped behaviour exactly: the
composition's injected per-(player, season) σ, graded by role.

## 📋 STATUS: NOTHING HERE IS BUILT. Queued for a later session.

**No injection code or config was touched on 2026-08-16.** `sim.minutes.player_season_sigma`
is still the scalar **0.375**, there is no `player_season_sigma_by_role` key, and
`minutes_unification.py`, `minutes_window.py` and `src/sim/` are unmodified. The one
injection-adjacent change that session made is `PlayerSeasonTerm.shift` learning to take a
`(draws × n_sigma)` σ — **backwards-compatible by construction and pinned by a test**: a scalar
or one-column σ broadcasts to every unit, which is the arithmetic it already did, and the
shipped injection passes a 1-D block. So the plumbing below is ready and the shipped behaviour
is unchanged.

### What the next session does, in order

1. §Steps 1 — extend `player_season_effect_sweep` and `estimate_sigma_on_train` to a per-unit
   σ: `sigma_vec[unit_bin][codes]` in place of the scalar, reusing `PlayerSeasonTerm._unit_bins`
   for the lookup rather than minting a second one. Sweep coordinate-wise, one refinement pass,
   confirm the optima are stable.
2. §Steps 2 — per-bucket optima on the fitting half, confirmed on validation, with the ship
   rule already written there (agree to a grid step, or the bucket keeps the shared 0.375).
3. §Steps 3 — wire `sim.minutes.player_season_sigma_by_role`, promote the per-role PIT table to
   a `make` target so §The evidence stops being scratch, and re-read the two recorded trade
   measurements at the graded values.

Then the gates and falsifiers below, unchanged.

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

### The evidence (scratch, 2026-08-16)

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
