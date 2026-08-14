# Next session — after session 4d

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md`. Everything
below is measurement. **Nothing in this round is committed** — the working tree carries the
whole of 2026-08-14.

## What happened on 2026-08-14

Three items from the previous handoff: **1** (the composition's full-window fit), **4** (both
availability to-dos), and **5** (both small items). Items **2** (the P5 chain) and **3**
(session 6b, the rate heads' arms) are untouched.

### 1. The composition's preseason arm survives the window the head fits — session 4d

`make composition-preseason-fit` at `first_season: "1996-97"` / `label: covered` →
`composition_preseason_fit_covered.csv`. **Three** fits, 7.76 h, max R̂ 1.00436, 0
divergences, 0 treedepth saturation.

- **The gate passes and the increment grows a third time.** −0.20883 (pilot) →
  **−0.23418 [−0.24551, −0.22314]** CRPS minutes per player-game on the draft pool, retention
  **1.040 → 1.115**, `team_sum_abs_error` exactly 0 on all six arms.
- **A third arm is what made the round decidable.** `base_full_window` fits 1996-97 carrying
  no preseason column — it is the shipped head. Against it the arm wins at **both** units:
  `ship_margin` **−0.25409 [−0.26520, −0.24315]** per player-game and **−17.27296
  [−22.32569, −11.92164]** per player-season.
- **The coverage cut costs −0.01991 [−0.02515, −0.01498] per player-game and it helps** —
  P3's direction, but **8.5%** of the increment rather than the quarter P3 paid. The
  decomposition is exactly additive: `window_cost + fitted_increment = ship_margin`.
- **4c's floor lesson reproduces on 4.6× the rows**: the floor's season increment spans zero
  (−6.36419 [−15.31395, +2.68556]), the fitted one does not (−17.13948). Retention 1.11 per
  game against 2.69 per season, one posterior.
- It is the **mean**: season MAE falls 17.70 min while the predictive sd *narrows* 59.58 →
  57.64. `sim.minutes.player_season_sigma` is untouched.

### 4. Both availability to-dos — `docs/availability-window-plan.md` §15

**Item 9 (§15a).** `make availability-absence` gained a `population` round. Every arm is
**fitted once and scored twice**; the population is a mask on the *scored* rows.
**§7 selected `mixture` on a boundary error measured pooled, and on the draft pool the two
likelihoods swap places** — `betabinom` − `mixture` is +0.00891 [+0.00420, +0.00994] pooled
and **−0.00308 [−0.00733, −0.00210]** draftable, and it **replicates rolling** (+0.00786 and
−0.00049, both clear of zero). The head is unchanged because D1's other half selects it there
instead (CRPS +0.04280 validation, +0.02864 rolling). **The sign is the finding and the size
is not** — the draftable margin shrinks 6.3× between readings, which item 9 predicted about
its own 1.84× before the round ran.

**Item 11 (§15b).** `availability_no_prior.run_recency` crosses the estimator's population
with the pool's depth. **The cancellation is confirmed and the fix is falsified.** At depth 5
the biases are +0.0642 (roster) and −10.0796 (all) — the drift is real and separable — but
CRPS degrades monotonically and the roster arm goes from −0.3486 at 13/19 origins to +0.2753
at 7/19, because `graded_share` falls 0.7926 → 0.5524. **The second change it needs is an
estimator class, not a shallower pool** — a shrunk cell estimator, the same instrument P4(a)
named for the *key* axis.

### 5. Both small items

- **The provenance stamp ships.** `stan_utils.diagnostics_frame` — the one function every
  Stan head goes through — now appends `git_commit`, `git_dirty`, `src_digest`, `written_at`.
  `src_digest` is the field that would have caught 2026-08-13: the two fits shared a commit
  and were both dirty, so only a content hash over `src/models`, `src/stan` and
  `src/features` separates them.
- **`stan.composition.effects.warmup` 500 → 1000.** `base` (30 params) and `team` (36) both
  needed 600 for `dense_e`. ⚠️ The nuance the previous handoff did not have: `ps` carries
  2,204 effects, cannot reach `dense_e` at any warmup, and so pays **2×** rather than saving
  10× — worth it anyway, because at 500 it posted **max R̂ 1.13173**. `announce_metric` moved
  into `stan_composition` beside `choose_metric` and now prints in `composition_effects` too.

### Three lessons worth carrying forward

**The previous handoff's "~9.92 h per arm" was wrong by ~4×, and it nearly mis-scoped the
session.** 9.92 h is the whole four-variant `make stan-composition` sweep (2.01 + 2.60 + 2.50
+ 2.67 h plus probe and comparator). One fit of the shipped variant is ~2.5 h. Read
`stan_composition_diagnostics.csv` per row before budgeting.

**Do not run other jobs on the box while a composition arm samples.** Arm 1 took 8,873 s
against a ~6,700 s projection because the §15 rounds and two test-suite passes shared the 14
cores. Arms 2 and 3 on a quiet box came in at 8,006 s and 11,052 s.

**An artifact written across a code edit records the wrong code, and this round produced a
live example.** `composition_preseason_fit_covered_diagnostics.csv` carries **no** provenance
stamp, because `diagnostics_frame` gained one *during* the 7.76 h run and the process had
already imported it. Nothing about the fits is affected. The next run of any head will carry
it.

## Do NOT re-decide these

1. **`k = 80`, `route = offset_only`, `betabinom_ot_graded`** — 4b decisions off the fitting
   half. Validation prefers `k = 160` and reading it would select on the split the arm is
   scored against.
2. **The composition's preseason arm is measured, NOT shipped.** `stan.composition.preseason`
   configures the *measurement* target only; no consumer reads it and `make stan-composition`
   is untouched.
3. **`sim.minutes.player_season_sigma` stays 0.450.** 4d's season-unit gain is mean, not
   spread, and the predictive sd went the other way.
4. **The availability head is unchanged.** §15a is a documentation correction, not a
   selection reversal — D1 selects `mixture` on both populations, by different routes.
5. **`sim.availability.no_design_level` stays `tenure_draft` on the all-rows estimator.**
   §15b ships nothing.
6. Everything the previous handoffs list under this heading still holds.

## Verified green at the end of the pass

```
make docs-audit        # 0 disagreements, 0 stale claims, 3,521 figures checked
make dashboard-audit   # 0 orphaned artifacts, 0 pending constants
pytest tests/          # 1,819 passed
```

`make dashboard-audit` reports 241 `reviewed`-date drifts, unchanged by this round.

## The work

### 1. 🔥 The chain is stale — P5, and it is now clearly the largest item
`make simulate-season`, `weekly-scores`, `bracket`, `draft-sim`, `strategy-sweep`. Hours of
compute. Two pending head changes (availability and marginal minutes) plus a re-read σ that
did not move.

**4d changes the calculus here.** The composition arm now beats the shipped head decisively
at both units, so the question "should the composition's preseason arm ship?" is live rather
than open — and if it does, the chain has to be re-run *after* it lands rather than before.
Adopting it means `first_season: 2004-05` on `stan.composition` plus `make posteriors
--groups composition` at all three fit windows (~6 h: train, train_val, full — the same
season floor, different splits). **Decide that before spending the chain**, or the sweep gets
run twice.

One open question that is genuinely a decision rather than a measurement: `make
stan-composition`'s ~11 audited figures describe the un-blended head at 1996-97. Adopting 4d
makes them describe a head that no longer exists. Re-running a 9.92 h ladder to refresh
records is usually not worth the sampler time — but it should be decided deliberately.

### 2. Session 6b — the five surviving rate heads' arms
`ast`, `fga`, `stl`, `tov`, `reb`, plus `ftm|fta`. A session P1 *added*. Untouched.

### 3. Two follow-ups §15 opened
- **A shrunk cell estimator for the no-design level**, replacing the hard `MIN_CELL`
  fallback. Both axes of P4(a) now point at it: the key axis graded only 54% of its rolling
  rows, and §15b's recency cut fails because grading collapses. It is a different estimator
  class, so it makes a new ladder rather than a new arm.
- **The era question on the composition**, which 4d touched twice without measuring for it:
  `window_cost` is the first paired interval saying the pre-2005 seasons cost this head
  something, and ρ rising 0.11479 → 0.12592 at the longer window is a second reading.
  `potential-to-dos.md` item 1.

### 4. Small, un-scheduled
- The `make stan-*` **metric** artifacts still carry no provenance; the stamp went on the
  **diagnostics** artifacts, which is where every head shares a code path. Metric artifacts
  are written per-module and would need six edits.
- `dashboard-audit`'s 241 `reviewed`-date drifts have been carried for several sessions.
