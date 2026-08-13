# Next session — after the σ re-read and P4

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md`. Everything
below is measurement. **The repo is committed and clean**; the previous session's "nothing is
committed" warning is discharged.

## What happened on 2026-08-14

Three of the four items the last handoff listed are done, and P4 ran.

- **The minutes-unification stake was re-read** and it reverses a verdict. It was also
  *broken*, not merely stale: the module built its frame through `stan_minutes.build_design`,
  which carries no preseason column, so rehydrating the shipped head raised `KeyError`.
- **`train_val` posteriors were refitted.** Both windows now carry the preseason blocks
  (availability 10 columns, minutes 5).
- **P4 ran, both halves.** (a) fails, (b) clears on five of eight rate targets. Nothing ships.
- Still open: **the chain re-run** (item 2 below), which is the whole of P5.

### The lesson worth carrying forward

**A consumer that rehydrates a head must build its frame through that head's own design
path.** `stan_minutes.head_design` existed for exactly this and `minutes_unification` was not
using it — the docstring even listed that module as a `build_design` consumer, which was true
when written and wrong the moment the head gained a block. The rule now stated in that
docstring is the general one: *anything that scores the shipped head goes through
`head_design`; anything that fits its own model on those rows keeps `build_design`.*

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
make docs-audit        # 0 disagreements, 0 stale claims, 3,309 figures checked
make dashboard-audit   # 0 orphaned artifacts, 0 pending constants
pytest tests/          # 1,781 passed
```

`make dashboard-audit` reports 241 `reviewed`-date drifts. That is a report rather than a gate
and it was at that scale before this round.

## The work

### 1. The chain is stale — P5, and it is the only large item left
`make simulate-season`, `weekly-scores`, `bracket`, `draft-sim`, `strategy-sweep`. Hours of
compute. This is what would price the availability block **in the contest**, which P2
explicitly could not: its gain is calibration, and §7l is the standing precedent that a head
change reaching the draw as shape rather than order can be a measured null there.

Note the chain now has *two* pending head changes rather than one — the availability and
minutes blocks — plus a re-read σ that did not move.

### 2. Session 4b — the composition's preseason arm at the pilot window
**Promoted by this session's re-read.** The marginal head now beats the composition by 33.45
CRPS minutes at the season unit, up from 25.70, entirely because it gained a block the
composition does not have. Giving the composition its own arm is the direct answer, and P3's
gate already opened it. `potential-to-dos.md` item 1 measured the pilot window at ~6× cheaper
than the full one.

### 3. Session 6b — the five surviving rate heads' arms
`ast`, `fga`, `stl`, `tov`, `reb`, plus `ftm|fta`. A session P1 *added*. P4(b) is weak
corroboration from a disjoint population: the same five-plus-the-mix shape showed up there on
players with no prior season at all.

### 4. Two open items on the availability head, both in `potential-to-dos.md`
- **Item 9** — the head's boundary defect on the draft pool, still unmeasured. It asks whether
  §7's mixture selection survives being read on the population the head serves.
- **Item 11** — new. The no-design availability rate is pooled over a population it is never
  applied to (0.5447 realized against 0.1571). Worth −0.3486 CRPS at 13 of 19 rolling origins
  and it fails validation for a reason the item states as a checkable hypothesis.

### 5. Small, un-scheduled
The `make stan-*` metric artifacts still carry no record of which code version wrote them,
which is what let the wrong availability arm reach the docs on 2026-08-13. `posteriors.py`
persists `n_features` and `preseason_columns` for exactly this.
