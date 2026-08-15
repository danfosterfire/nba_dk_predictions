# Next session — after P5

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md` P5.
**Nothing in this round is committed** — the working tree carries the whole of 2026-08-14's
evening pass. (The previous handoff said the same thing and was wrong: the morning's work
*was* committed at `762c5e2`. Check `git status` rather than trusting this line.)

## What happened on 2026-08-14 (evening)

**P5 ran end to end.** The composition adopted its preseason blend, the ladder and posteriors
were re-fitted behind it, σ moved, and the chain was re-run to the strategy sweep. Two owner
decisions were taken during the session and both are recorded in `dashboard/decisions.py`.

### 1. The composition's preseason arm SHIPPED — `docs/preseason-plan.md` P5

`stan.composition.preseason.adopt: true`, `k = 80` on `offset_only`, window cutting itself to
2004-05. It ships on 4d's `ship_margin`: **−0.25409 [−0.26520, −0.24315]** CRPS minutes per
player-game and **−17.27296** per player-season against the head that was shipping.

**The adoption is a separate door and that is the whole design.** `stan_composition.head_frame`
is `stan_minutes.head_design`'s rule one head over — `run`, `posteriors`, `sim/season`,
`minutes_unification` and `model_cards` go through it; `composition_preseason`,
`composition_effects`, `minutes_window` and `rookie_priors` keep building on the untouched
`composition_frame`. That second list is why the door exists: every gate arm in 4b–4d is
scored against a `base` control built with no hook, and a blend reaching it from config would
have collapsed three sessions of margins silently. A test pins it.

`make stan-composition` re-run: **444.8 min**, max R̂ 1.00804, 0 divergences over 6 fits,
`betabinom_ot_graded` **re-selected** against an offset that moved on 73% of rows. CRPS
4.4945 → **4.2617**; vs the comparator −0.2898 → **−0.4186** (−6.06% → −8.94%); calibration
cut 35% → **40.9%**. ρ fell at every tier (2.07× → **1.91×** spread) — the mechanism 4c/4d
predicted, and it points *opposite* to the window effect, which raises ρ. Both readings are
now recorded side by side in `minutes-composition-plan.md`.

`make posteriors --window train --groups composition`: **122.6 min**, R̂ 1.0026, round-trip
exact. The artifact records `preseason_blend_k` and `preseason_route`, and `model_cards`
**raises** on a mismatch rather than carding a blended posterior against an un-blended frame.

### 2. σ moved 0.450 → **0.375**, because its input changed

`make minutes-unification` re-scored. Both grids now put the optimum at 0.375 — train
108.4687 against 0.450's 108.834, validation 130.692 against 132.437 — where before the blend
they sat one step apart. At 0.375 the injected composition **beats** the marginal head
(**−5.91125 [−10.3858, −1.50137]**) where before it lost at +6.26 [+0.92, +11.49]. PIT KS
0.0665 against the marginal head's 0.0668.

Also: the composition now **clears** the season-unit no-fit floor (155.94 against 161.29),
which reverses one of the sharpest lines in the repo. The spread verdict is unaffected — it is
still 4.58× too narrow, and PIT KS is what separates them.

### 3. The marginal-head retirement question was MIS-FRAMED

`src/sim/` imports neither `StanMinutes` nor `rehydrate_minutes` and never looks up the
`minutes` artifact. **The simulator's minutes have come from the composition plus the injected
σ all along.** `make model-cards` says so independently: *"16 of them in the simulator's draw
path (gp_entry, gp_exit, gp_onset, **minutes** are not)"*.

So retiring it means ceasing to **fit** it, and the case for keeping it does not depend on
which head predicts better: it is the `independent_comparator` the −0.4186 headline is
measured against, and the season-unit reference σ is calibrated against, at **514 s** against
the composition's **7,357 s**. README's framing was corrected; the head stays.

### 4. The chain — 63 minutes, and everything improved

| | before | after |
|---|---|---|
| Gate A season-total MAE (22-23 / 23-24) | 397.36 / 398.45 | **363.234 / 377.510** |
| Gate A season-total bias | −26.50 / −71.15 | **−15.4388 / −66.6411** |
| 600k simulated lift | 0.1890 | **0.2358** |
| 600k realized lift | 0.1713 | **0.204098** |

Gate D **still fails at 0 of 6**. ⚠️ **None of this is attributable to the preseason block** —
the composition, σ, the ADP field and the error injection were all re-fitted in one pass and
the previous `strategy_*.csv` was overwritten. Isolating the block needs the pre-block tensors
kept and a paired re-run.

## Do NOT re-decide these

1. **`k = 80`, `route = offset_only`** — 4b decisions off the fitting half.
2. **`betabinom_ot_graded`** — re-selected on the blended offset, not carried over.
3. **σ = 0.375** — read off TRAIN, and both grids agree exactly. It moved once because its
   input moved; it does not move again without the head moving again.
4. **The marginal head stays fitted.** See §3 — this is settled on a structural argument, not
   on a metric.
5. Everything the previous handoffs list under this heading still holds.

## Verified green at the end of the pass

```
make docs-audit        # 0 disagreements, 0 stale claims, 3,510 figures
make dashboard-audit   # 0 orphaned artifacts, 0 pending constants
pytest tests/          # 1,828 passed
```

`make dashboard-audit` reports 285 findings, all `reviewed`-date drift, unchanged in kind.

## The work

### 1. 🔥 What the preseason block is worth in the contest — still open
The one question P2, 4d and P5 all logged and none answered. Needs the pre-block tensors
retained and a paired re-run against them. This is the last thing standing between the round
and a defensible "the block was worth shipping" claim.

### 2. The season-total bias — `potential-to-dos.md` item 12
Gate A's bias is **−15.44 / −66.64** against a −3.06 bar, and it is pre-existing rather than
introduced here. Two caveats on the bar itself: it is 873 pooled rows against the check's
386/387, and it is a **full-season** figure against the simulator's **91% tournament window**.
Item 12 measures one mechanism — the layout fits a full season and is applied to the front
91%, worth −5.92 / −7.29 dk_pts, i.e. **38% and 11%** of the gap. Two other channels were
falsified in the same sitting (per-game production is flat across the boundary; games played
does not track the bias). ⚠️ The naive version of item 12's measurement gives the **opposite
sign** — read the entry before re-running it.

### 3. Session 6b — the five surviving rate heads' arms
`ast`, `fga`, `stl`, `tov`, `reb`, plus `ftm|fta`. Untouched, still.

### 4. Small, un-scheduled
- **`make strategy-sweep` took 53 minutes against the 4.8 the docs quote**, at unchanged
  scale (`sim.strategy.n_sims` is 500 and applied). A profiler put it in `np.searchsorted` and
  `np.argsort`, nothing pathological. Likely the arms items 6–7 added while the sweep was
  deliberately stale. That cost figure is owed a re-measurement, not a re-quote.
- **Buffered stdout cost real diagnostic time three times today** — the ladder, the posteriors
  and the sweep all ran blind, and the sweep needed `/usr/bin/sample` to confirm it was alive.
  `PYTHONUNBUFFERED=1` on the long `make` targets would fix it.
- The `make stan-*` **metric** artifacts still carry no provenance stamp; it went on the
  **diagnostics** artifacts.
- `dashboard-audit`'s `reviewed`-date drift has been carried for several sessions.

## Two lessons worth carrying forward

**A verdict asserted in prose beside its own numbers will go stale silently.** Two live
examples this round: `minutes_unification` printed "does not clear the no-fit carry-forward
floor" beside figures saying it did, and the dashboard's minutes caption did the same. Both
now derive the verdict from the frame they quote. This is the `make docs-audit` failure mode
one level in — inside the code that *writes* the artifact, where no doc guard reaches.

**A paired table cannot be half-refreshed.** `availability-window-plan.md` §8b compares
`pooled` against `tenure_draft`; the chain moves only the shipped side, so re-pointing it
would leave a documented *gap* comparing two different chains. Both sides are now
`historical=True` with the current values stated in prose beside them.
