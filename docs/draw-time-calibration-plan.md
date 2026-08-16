# Draw-time calibration plan — grading the injected σ by role

**Opened 2026-08-16. A workplan, not a results doc: nothing here has been built.** Not in
`make docs-audit`; scratch figures are labelled and earn audited status only when their
step gets a `make` target. One calibration of a **simulation-time draw input** — no Stan,
no likelihood change, and the knob nests the shipped behaviour exactly: the composition's
injected per-(player, season) σ, graded by role. It is the injection-side sibling of
`docs/composition-quadrature-plan.md`'s graded fitted σ: days rather than weeks, no
sampler risk, and superseded gracefully if the quadrature fit ships.

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
  is the skew residue); record the null and leave the scalar.
- **Coordinate-wise sweeps do not stabilize** → the team-constraint coupling is stronger
  than expected; the honest instrument is then a small joint grid over (fringe, star) with
  the middle interpolated, before concluding anything.
- If the quadrature fit ships a graded `sigma_u_draws`, this part's constants retire — but
  its per-bucket grid optima remain useful as the per-bucket extension of the "five
  independent routes" check on the fitted values.

---

## Cost and blast radius

Hour-scale compute (grid sweeps through the existing draw-and-score machinery), no sampler,
no shipped head refitted, no audited record touched. The one code touch outside the sweep
modules — the unit → bin σ lookup in `rehydrate_composition` — is shared with the
quadrature plan's step 6 and built once.
