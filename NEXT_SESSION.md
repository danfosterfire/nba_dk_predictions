# Next session — after session 4c

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md`. Everything
below is measurement. **Nothing in this round is committed** — the working tree carries
session 4c in full.

## What happened on 2026-08-13 (session 4c)

The previous handoff's item 0 ran and **passes**. `make composition-preseason-fit`
(`src/models/composition_preseason_fit.py`) fits 4b's blended-offset arm at the pilot window
against a same-window control, plus both frames' own no-fit floors at the same 200 predictive
draws.

- **The increment survives the posterior and grows.** Draftable, per player-game:
  **−0.20883 [−0.22129, −0.19693]** fitted against the floor's −0.20081 [−0.21697, −0.18411]
  on the same frames — **retention 1.040**. That is P3's direction, not the shrink 4b allowed
  for.
- **The season unit reverses half of 4b decision 4.** The floor's tie (−5.31580 [−14.44243,
  **+3.83015**]) becomes **−14.62649 [−19.48503, −9.54922]** once fitted, at 2.75×. The
  *reason* 4b gave survives — the season predictive sd **narrows** (56.25 → 55.31), so no
  spread was manufactured and σ is still the only parameter for that — but what moves is the
  **mean**, season-total MAE falling **15.05** minutes.
- **4b's suggestive cross-artifact line is now within-artifact and holds**: the *un-fitted*
  blended floor (4.41998) beats the *fitted* incumbent-offset arm (4.44188).
- Sampler: max R̂ **1.0047**, 0 divergences, 0 treedepth saturation, 41 min over two fits.

### Two lessons worth carrying forward

**`FloorComposition` is a conservative screen per game and a misleading one per season.** Its
mean is the offset alone (`eta = 0`), so it cannot see anything a coefficient re-weights —
and that blindness *compounds with aggregation*, which is why the per-game retention is 1.04
and the per-season one is 2.75. Any future round that reaches for this floor as a cheap gate
should read it at the per-game unit only.

**`warmup: 500` on the composition is not a cheaper fit — it is a slower one, and nothing
says so.** `choose_metric` grants `dense_e` only at `warmup ≥ 20 × parameters`; this arm has
30, so the threshold is exactly **600**. At 500 the head silently drops to `diag_e`, which
this head's own probe measures at treedepth 8–9 against 4. A first attempt ran **32 minutes
without completing its 500 warmup draws**, against `composition_effects`' **880 s** for the
same arm at the same window for a whole 500+500 fit. The only early signal is the metric
itself — CmdStan writes no draw until warmup ends and its progress lines are buffered away —
so `composition_preseason_fit.announce_metric` now prints it in the first second and warns on
`diag_e`, with a test pinning the threshold. ⚠️ **The `effects` block sits under the same
cliff and has not been changed.**

## Do NOT re-decide these

1. **`k = 80`, `route = offset_only`, `betabinom_ot_graded`.** All three are 4b decisions off
   the fitting half; validation prefers `k = 160` and reading it would select on the split
   the arm is scored against.
2. **The composition's preseason arm is measured, NOT shipped.** Everything is 2018-19
   onward. `stan.composition.preseason` configures the *measurement* target only — no
   consumer reads it and `make stan-composition` is untouched.
3. **`sim.minutes.player_season_sigma` stays 0.450**, and nothing in 4c touches it. The
   season-unit gain here is mean, not spread, and the predictive sd went the other way.
4. Everything the previous two handoffs list under this heading still holds: the injected
   composition loses to the marginal head (+6.26 [+0.92, +11.49]); P4(a) and P4(b) ship
   nothing; the availability gate failed and the block ships anyway.

## Verified green at the end of the pass

```
make docs-audit        # 0 disagreements, 0 stale claims, 3,402 figures checked
make dashboard-audit   # 0 orphaned artifacts, 0 pending constants
pytest tests/          # 1,803 passed
```

`make dashboard-audit` reports 241 `reviewed`-date drifts, unchanged by this round.

## The work

### 1. ⚙️ The composition's full-window preseason fit — the direct continuation
The pilot passed at both units, so the remaining question is whether it survives the full
window. **It needs P3's coverage cut first**: the preseason panel starts at 2004-05 and this
head fits from 1996-97, so the missing indicator would be an era dummy on the early rows —
and P3 priced its own cut at **1.19 CRPS minutes before any preseason column existed**, a
quarter of that round's increment. Budget ~9.92 h per arm at the full window, so this is a
two-arm run of most of a day and Gate A should be read before starting.

Note the arm is a change to the **allocation mean**, so it reaches the draw as *order* rather
than pure *shape* — §7l's standing precedent for a measured null in the contest applies less
cleanly here than to the availability block.

### 2. The chain is stale — P5, and it is still the largest item
`make simulate-season`, `weekly-scores`, `bracket`, `draft-sim`, `strategy-sweep`. Hours of
compute. Two pending head changes (availability and marginal minutes) plus a re-read σ that
did not move; the composition arm is **not** among them unless item 1 ships first.

### 3. Session 6b — the five surviving rate heads' arms
`ast`, `fga`, `stl`, `tov`, `reb`, plus `ftm|fta`. A session P1 *added*.

### 4. Two open items on the availability head, both in `potential-to-dos.md`
- **Item 9** — the head's boundary defect on the draft pool, still unmeasured.
- **Item 11** — the no-design availability rate is pooled over a population it is never
  applied to (0.5447 realized against 0.1571).

### 5. Small, un-scheduled
- The `make stan-*` metric artifacts still carry no record of which code version wrote them,
  which is what let the wrong availability arm reach the docs on 2026-08-13.
- **`stan.composition.effects.warmup: 500` is under the `dense_e` cliff** (see above). Any
  re-run of `make composition-effects` should raise it to 1000 first, or it pays ~10× for
  nothing.
