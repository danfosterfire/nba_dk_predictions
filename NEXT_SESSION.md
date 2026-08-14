# Next session — after the σ re-read and P4

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md`. Everything
below is measurement. **The repo is committed and clean**; the previous session's "nothing is
committed" warning is discharged.

## What happened on 2026-08-14

Three of the four items the last handoff listed are done; P4 ran, and session 4b ran after
it.

- **The minutes-unification stake was re-read** and it reverses a verdict. It was also
  *broken*, not merely stale: the module built its frame through `stan_minutes.build_design`,
  which carries no preseason column, so rehydrating the shipped head raised `KeyError`.
- **`train_val` posteriors were refitted.** Both windows now carry the preseason blocks
  (availability 10 columns, minutes 5).
- **P4 ran, both halves.** (a) fails, (b) clears on five of eight rate targets. Nothing ships.
- **Session 4b ran and passes.** The composition's preseason arm earns a Stan fit; the round
  found the offset carries it and the ordering does not.
- Still open: **the chain re-run** (item 2 below), which is the whole of P5.

### The lesson worth carrying forward

**A consumer that rehydrates a head must build its frame through that head's own design
path.** `stan_minutes.head_design` existed for exactly this and `minutes_unification` was not
using it — the docstring even listed that module as a `build_design` consumer, which was true
when written and wrong the moment the head gained a block. The rule now stated in that
docstring is the general one: *anything that scores the shipped head goes through
`head_design`; anything that fits its own model on those rows keeps `build_design`.*

**And one correction, because it is the kind that reads as true.** Session 4b was first
written up saying the composition's `w_share` "is not a feature" and enters the head twice. It
enters **three** ways — `OWN = logit_share_lag1` is `logit(w_share)` and is in every variant's
feature list. The two a coefficient cannot reach are the **offset** (`logit_prior`) and the
**allocation order**, and those are what the round measures. The floor gate is unaffected and
is in fact cleaner than the original claim: `FloorComposition` sets `eta = 0`, so it switches
the feature route off and isolates exactly the two unreachable ones. Fixed in `e017c7f`.

## Do NOT re-decide these

1. **`sim.minutes.player_season_sigma` stays 0.450.** Its train grid is the composition scored
   against realized minutes and the marginal head is not in it, so nothing about the preseason
   block could move it, and nothing did. What moved is the *verdict* below.
2. **The injected composition now LOSES to the marginal head** (+6.26 [+0.92, +11.49] at the
   shipped σ, from −1.49 [−6.14, +3.22]). Retiring `stan_minutes` is closed, not live.
3. **P4(a) failed and nothing from it ships.** `sim.availability.no_design_level` stays
   `tenure_draft` on the all-rows estimator.
4. **P4(b) ships nothing either** — a no-prior player is not in the component heads at all, so
   there is no consumer to give a rate to. It establishes *which* prior to use if one is added.
5. Everything the previous handoff listed under this heading still holds: the availability
   gate failed and the block ships anyway; the shipped arms are fixed; `volume_centered` is
   `withdrawn`; a complete preseason is a production precondition; P1 decision 4 is reversed.

## Verified green at the end of the pass

```
make docs-audit        # 0 disagreements, 0 stale claims, 3,341 figures checked
make dashboard-audit   # 0 orphaned artifacts, 0 pending constants
pytest tests/          # 1,792 passed
```

`make dashboard-audit` reports 241 `reviewed`-date drifts. That is a report rather than a gate
and it was at that scale before this round.

## The work

**Start with item 1 unless you want the chain running in the background all session.** It is
the only item whose result changes what the chain should be run *on*, and it is ~1 h against
the chain's several.

### 0. ⚙️ Fit the composition's 4b arm — the recommended start
`make composition-preseason` established that blending a preseason minutes share into
`w_share` is worth **−0.19972 [−0.21661, −0.18225]** CRPS minutes per player-game with nothing
fitted. What is missing is the fit: a **pilot-window** `stan-composition` run of the
blended-offset arm at `k = 80`, against a same-window control, on the pattern
`composition_effects` already uses (a `base` arm at the same window, never the full-window
incumbent).

The floor is a screen and not a substitute — `beta` can correct an offset the floor cannot, so
the increment could shrink under the posterior, or grow as P3's own did. That is the open
question and it is one fit.

Three things not to re-derive:
- the route that pays is the **offset**, not the ordering (103% against 3.75%);
- **nothing here needs centring** — `logit_prior` is a within-team ratio `w_k / tail_k`, so the
  common multiplicative compression that forced P3's centring cancels exactly;
- `k = 80` comes from an inner carve of the fitting half. Validation prefers 160; using it
  would be selecting on the split the arm is scored against.

### 1. The chain is stale — P5, and it is the only large item left
`make simulate-season`, `weekly-scores`, `bracket`, `draft-sim`, `strategy-sweep`. Hours of
compute. This is what would price the availability block **in the contest**, which P2
explicitly could not: its gain is calibration, and §7l is the standing precedent that a head
change reaching the draw as shape rather than order can be a measured null there.

Note the chain now has *two* pending head changes rather than one — the availability and
minutes blocks — plus a re-read σ that did not move.

### 2. Session 6b — the five surviving rate heads' arms
`ast`, `fga`, `stl`, `tov`, `reb`, plus `ftm|fta`. A session P1 *added*. P4(b) is weak
corroboration from a disjoint population: the same five-plus-the-mix shape showed up there on
players with no prior season at all.

### 3. Two open items on the availability head, both in `potential-to-dos.md`
- **Item 9** — the head's boundary defect on the draft pool, still unmeasured. It asks whether
  §7's mixture selection survives being read on the population the head serves.
- **Item 11** — new. The no-design availability rate is pooled over a population it is never
  applied to (0.5447 realized against 0.1571). Worth −0.3486 CRPS at 13 of 19 rolling origins
  and it fails validation for a reason the item states as a checkable hypothesis.

### 4. Small, un-scheduled
The `make stan-*` metric artifacts still carry no record of which code version wrote them,
which is what let the wrong availability arm reach the docs on 2026-08-13. `posteriors.py`
persists `n_features` and `preseason_columns` for exactly this.
