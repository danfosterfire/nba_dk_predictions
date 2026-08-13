# Next session — after the preseason ports and their documentation pass

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md`. The
bookkeeping round is **done**; what is left is measurement work, and all of it is P4/P5.

## What happened on 2026-08-13 / 08-14

P2's availability arm **failed its gate as written** and shipped anyway on an explicit owner
decision; P3's minutes arm cleared its gate outright. Both heads carry a preseason block,
both were ported to Stan, both posteriors were re-persisted, and the whole documentation
surface was brought into agreement with them.

- **Availability** ships **P1's full ten columns** (`stan_availability.PRESEASON_COLS`) — the
  arm the *rolling harness* selected, reversing P1 decision 4.
- **Minutes** ships the **season-centred delta** plus four age-split indicators, fitting
  window cut to 2004-05 (8,306 → 6,152 rows).
- `stan.availability.preseason: false` and `stan.minutes.preseason: false` are exact
  rollbacks. Neither block reaches the shared builders.

### The one thing worth carrying forward as a lesson

**The availability head was ported twice and the first port was the wrong arm.** The 14:19
artifacts carried the *withdrawn* `volume_centered` arm (five columns, 40 terms) because
`PRESEASON_COLS` was switched to the ten-column block at 14:25 — six minutes later. The
figures from that run reached `docs/preseason-plan.md` and `dashboard/decisions.py` before
anything noticed; they were caught by arithmetic (45 terms = 20 β + 10 + 4 ρ + 3 + 8 γ) and
the head re-fitted. **An artifact carries no record of which code version wrote it.**
`posteriors.py` persists `n_features` and `preseason_columns` for exactly this; the
`make stan-*` metric artifacts carry no equivalent. Adding one is a cheap, un-scheduled idea.

## Do NOT re-decide these

1. **The availability gate failed and the block ships anyway.** Recorded deliberately. Do not
   re-run the gate, move the bar, or "fix" the failure.
2. **The shipped arms are fixed** — ten columns on availability, the centred delta on minutes.
3. **`volume_centered` is `withdrawn`, not deleted**, and the vs-primary table is why nothing
   was lost by withdrawing it.
4. **A complete preseason is a production precondition.** The runbook's Oct 17–20 window is
   load-bearing.
5. **P1 decision 4 is reversed**, on the rolling harness's `crps_vs_primary` column.

## Repo state — everything below is UNCOMMITTED

Branch `incorporate-preseason-data`. Commit `cf3f92d "Step P2"` holds the P2 ladder module,
its tests and the validation-half docs. Everything after is in the working tree: the ports,
the rewiring, the registry entries, the re-run artifacts, and this round's documentation pass
(`README.md`, `docs/project-spec.md`, `docs/availability-plan.md`, `docs/predictions-plan.md`,
`docs/model-development-notes.md`, `docs/preseason-plan.md`, `docs/potential-to-dos.md`,
`configs/default.yaml`, `dashboard/decisions.py`, `src/docs_audit.py`).

**Nothing is committed. A commit is the first thing this session should do or decline.**

### Verified green at the end of the pass

```
make docs-audit        # 0 disagreements, 0 stale claims, 3,219 figures checked
make model-cards       # 20 heads, worst recipe design error 0.00e+00
make dashboard-audit   # 0 orphaned artifacts, 0 pending constants
pytest tests/          # 1,766 passed
```

`make dashboard-audit` reports 237 `reviewed`-date drifts. That is a report rather than a
gate and it was at that scale before this round; it will grow once these edits are committed,
since "last changed" is read from git.

## The work — all of it measurement, none of it bookkeeping

### 1. `sim.minutes.player_season_sigma = 0.450` is stale — do this first
The minutes head's fitted season-level ρ moved **0.05025 → 0.041894** (~9% narrower
season-total predictive), and this head ships *for* its season-level spread. σ was calibrated
against the pre-block head in `minutes-window-plan.md` §4. Re-read the composition-vs-marginal
stake before the chain is trusted: `make minutes-unification` is cheap (seconds, no CmdStan,
reads the persisted posteriors) and its artifact is now stale against the posteriors on disk.
**Every `minutes-unification` figure in `README.md` §2 and §3 predates the block** and is
flagged as such in the doc rather than silently refreshed — refreshing it is this item.

### 2. The chain is stale — P5
`make simulate-season`, `weekly-scores`, `bracket`, `draft-sim`, `strategy-sweep`. Hours of
compute. This is also what would price the availability block *in the contest*, which P2
explicitly could not: its gain is calibration, and §7l is the standing precedent that a head
change reaching the draw as shape rather than order can be a measured null there.

### 3. Unstarted sessions
- **P4** — no-prior ladder + rookie rate priors, sized at 29.7% of panel rows.
- **Session 4b** — the composition's preseason arm at the pilot window, opened by P3's gate.
- **Session 6b** — the five surviving rate heads' arms (`ast`, `fga`, `stl`, `tov`, `reb`,
  plus `ftm|fta`), a session P1 *added*.
- **`potential-to-dos.md` item 9** — the availability head's boundary defect on the draft
  pool, still open and unmeasured. It is the one that asks whether §7's mixture selection
  survives being read on the population the head serves.

### 4. Posteriors at the other windows
Only the **`train`** window was refitted. `data/features/posteriors/train_val/` still holds
**pre-block** availability (2026-08-12) and minutes (2026-08-08) pickles. `make posteriors
--window train_val` is what fixes it. Not urgent, and note it is *not* what
`src/final_evaluation.py` reads — that module refits from scratch through
`stan_availability.fit_and_score` and was already rewired to `head_design` / `head_features`,
so the final readout scores the head that ships. The risk is a consumer that loads the
`train_val` pickles for a test-side readout and silently gets the pre-block head.
