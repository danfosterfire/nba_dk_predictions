# Shipping the rookie-inclusive tensors — the `train` re-run and the `full` production path

**Opened 2026-08-22.** `docs/rookie-rates-plan.md` §7g and §7i built the wider unit
population and deliberately did not ship it: `sim/season.build_context` scores the union of
two rate families and lays the availability ladder's returnees out on the real schedule,
but every tensor `make simulate-season` has written is still the rung-0 veteran one, and
§7i's rookie-inclusive pair lives beside them under a `_rookieinclusive` label. This round
closes that gap at the **`train`** window, and builds the thing that does not exist at any
window — the **production path to a 2026-27 board** the live draft room can open. The
*board* itself is October-gated on data nobody has yet (§1a); the path is not, and it is
what the October crunch has three days to not be debugging.

Six sessions (§5), each appending its results to §7.

## ✅ STATUS: ROUND CLOSED 2026-08-23. ALL SIX SESSIONS RUN (§7a-§7f); §4'S SIX ACCEPTANCE CONDITIONS TAKEN AND GREEN.

**The round has two halves and they fail differently.** The `train` half is a *re-run*: no
new code, ~4 h of numpy, and **359 audited claims to rewrite** — the compute is the easy
part and `make docs-audit` is a gate. The `full` half is a *build*: no target writes a
production tensor today, `sim/season.run` refuses a season outside the selection split
under any window, and `dashboard/draft_room.py` cannot open a 2026-27 board at all. The
second half is the round's point, not its tail, so it runs **first** (§5b) — it is the only
half that touches the path to the board actually being drafted, and it edits the module the
other half then runs.

**🔴 What §5b ships is the path and a rehearsal, not a board.** The 2026-27 preseason has
not been played, so every head would score that season on its missing-preseason arm — §1a
measures it at 0 of 489 availability rows, 0 of 425 component rows and 0 of 116 rookie
rows. The persisted tensor therefore stamps its own preseason coverage and carries a
self-arming staleness guard that turns into a refusal the moment the October logs land.

---

## 1. What is on disk, and what this round changes

`data/features/sim_tensor_<season>.npz`, read back on 2026-08-22:

| tensor | units | window | `unit_family` | injected σ | vintage |
|---|---:|---|---|---|---|
| `2018-19` | 366 | train | **absent** | **0.450 shared** | 2026-08-12 |
| `2021-22` | 400 | train | **absent** | **0.450 shared** | 2026-08-12 |
| `2022-23` | 386 | train | absent | role-graded | 2026-08-16 |
| `2023-24` | 387 | train | absent | role-graded | 2026-08-16 |
| `2022-23_rookieinclusive` | **471** | train | yes | role-graded | 2026-08-22 |
| `2023-24_rookieinclusive` | **467** | train | yes | role-graded | 2026-08-22 |
| `2024-25` | 386 | train_val | absent | role-graded | 2026-08-21 |
| `2025-26` | 405 | train_val | absent | role-graded | 2026-08-21 |
| `2026-27` | — | — | — | — | **does not exist** |

Three separate facts are in that table and only one of them is the rookie program.

**1. The shipped population is rung-0 veterans.** 386 and 387 units against the labelled
pair's 471 and 467. The added rows split **72 true rookies + 13 `returnee_lag2`** on
2022-23 and **74 + 6** on 2023-24, and the shipped count reproduces the labelled tensor's
rung-0 census exactly (386 and 387), which is the check that the two tensors describe the
same veterans plus a disjoint remainder.

**2. 🔴 The two training-season tensors are two config generations stale, and that has
nothing to do with rookies.** They carry no `player_season_sigma_by_role` key and a shared
`player_season_sigma` of **0.450** — so they predate both the 0.450 → 0.375 move of
2026-08-14 (the composition adopting its preseason-blended offset,
`docs/preseason-plan.md` P5) and the role-graded injection of 2026-08-16
(`docs/draw-time-calibration-plan.md`). `make weekly-scores` reads all four training and
validation tensors into **one pooled panel**, so `weekly_score_index.csv` — 95 audited
claims in `docs/simulations-plan.md`, written 2026-08-15 — currently describes a panel
whose two halves were drawn under different minutes injections. This round is the first
thing that fixes it, and it would have been worth doing with no rookie on the board.

**3. There is no production tensor, and the one that could be built today is not the one
to draft off.** `make forward-board` simulates 2026-27 in memory and persists only the
population census (`forward_board_population_2026_27.csv`: **419** veteran + **6**
lag-recovered + **116** true-rookie units, 197 of them ADP-priced);
`make posteriors-production` has fitted 31 heads at `full`; and nothing joins the two. But
the join is only half the gap — see §1a.

### 1a. 🔴 The production board is October-gated, and the gate is the preseason block

The forward frames build for 2026-27 **with the preseason block entirely absent**, because
the 2026-27 preseason has not been played. Measured 2026-08-22 by building the frames:

| frame | forward rows | carrying a preseason block |
|---|---:|---:|
| availability design | 489 | **0** |
| component design | 425 | **0** |
| rookie design | 116 | **0** |

`preseason.parquet` ends at **2025-26**. Every `min_pre`, `pre_per36_*` and
`pre_d_*_shrunk` column is NaN or zero, `has_preseason` is **0** on every row, and the four
age-split missing indicators fire on **100%** of the availability design
(108 / 214 / 104 / 63 across `<24` / `24-27` / `28-31` / `32+`). That is the designed
fallback working correctly — and it means every head scores 2026-27 on its
missing-preseason arm.

**That is not a small hole.** `docs/preseason-plan.md` measures the preseason block as
worth more than fitting itself on several heads, and `docs/rookie-rates-plan.md` §7i
measures the rookie floor's plug-in at **99.1%** preseason coverage with a median `w` of
**0.3057** — so a rookie built today lands on his draft-slot bucket exactly, with none of
his own October evidence. A tensor built now is the **August** board: everybody priced as
though the preseason never happened.

Two further inputs are October-gated more softly. The **roster snapshot** (577 players)
churns with every signing to the opener, and the Makefile's own `rosters` comment says a
board drafted in October off an August roster is drafting last summer's league. The **ADP
board** last captured 2026-08-20, where `docs/adp-plan.md` A0 wants the timing-matched
mid-October capture. `docs/final-evaluation-plan.md` §6d and `docs/preseason-plan.md`'s
production runbook both already say this; what neither says is what it means for a
*persisted* tensor, which is what §5b now has to answer.

## 2. What already exists — do not rebuild

- **The posteriors.** 31 heads at `train`, `train_val` and `full`. `make production-check`
  is green on 31 of 31 at `full` against a ten-column specification comparison. **Nothing
  in this round refits anything**, at any window, for any reason.
- **The union.** `sim/season.component_units` takes the two rate families' union and
  `save_tensor` writes `unit_family` and `lag_rung` per row (`docs/rookie-rates-plan.md`
  §7g).
- **The availability ladder.** `stan.availability.lag_ladder: [returnee_lag2]` behind
  `availability.rung_zero` at eight fitting paths (`docs/availability-window-plan.md`
  §16j).
- **The forward frames.** `forward_board.forward_frames` builds every frame
  `build_context` needs for a season with no game log, including the true-rookie design,
  and `--frames-only` has already run it on 2026-27 (`docs/rookie-rates-plan.md` §7h).
- **The labelling machinery.** `--tensor-label` suffixes the tensor, the cached field,
  Gate A and every sweep artifact. The *machinery* stays; only the `_rookieinclusive`
  label retires (§5e).
- **`posteriors.assert_production`.** The precedent §5b copies: a typed `--production`
  flag plus a stated reason, distinct from the test-split unlock.

## 3. 🔴 Constraints this round inherits

**C1 — THE HELD-OUT TENSORS ARE OUT OF SCOPE AND STAY THAT WAY.**
`sim_tensor_2024-25.npz` and `sim_tensor_2025-26.npz` were written by
`src/final_evaluation.py` under the one-shot unlock. Redrawing them is a second reading of
a spent split. **Decided 2026-08-22: the record is worth more than the consistency.** Do
not unlock the split, do not redraw them, and do not "fix" the mismatch that results.
After this round the `train` and `full` tensors carry two rate families and the
`train_val` pair carries one, **permanently and by choice**, and
`docs/final-evaluation-plan.md` §7 says so in those words.

**C2 — Nothing is refitted.** Numpy only, no CmdStan. The posteriors already carry all 31
heads at every window; a tensor re-run is a re-*draw*, not a re-fit. Any session that finds
itself wanting a sampler has found a scope error.

**C3 — Supersede rather than delete.** Where a doc's argument rests on a number this round
moves, the old figure stays beside the new one under `Claim(historical=True)`, per
`docs/docs-audit.md`. The failure mode for a live figure is that it drifted; for a
superseded one it is that somebody tidied it away.

**C4 — A selection is re-run, never hand-edited.** Gate C's solved `rho` and scale, the
sweep's shipped arm, the strategy table's selected rows and Gate B's fitted noise are
*selections*. They are re-read from the re-run artifact. Editing a selected value to match
a new artifact is the one edit that leaves every number plausible and the decision wrong.

**C5 — The two frozen counterfactuals do not move, and must not be re-captured.**
`make mixture-value` and `make preseason-contest` read `outputs/predictions/mixture_arms/`
and `preseason_arms/` — frozen snapshots of two complete chains — and never the live
artifacts. Re-running them after this round reproduces their numbers exactly, so their
**135 audited claims** (101 in `docs/preseason-plan.md`, 34 in
`docs/availability-window-plan.md`) do **not** move. Re-capturing either arm would need
the counterfactual half refitted (1 h of CmdStan for the mixture, ~3.5 h for the preseason
block, each ×2 arms), which C2 forbids. What changes is only their *provenance*: both
captures are now snapshots of a superseded chain, and §5e records that in prose rather
than in numbers.

**C6 — The production PATH ships this round; the production BOARD ships in October.**
§1a is why. What can be finished today is everything except the data: the unlock, the
persistence, the cache-key fingerprint, the draft-room wiring, and a **rehearsal tensor**
that proves the path end to end on the frames that exist. What cannot be finished today is
the tensor anyone drafts off. So the round's deliverable at `full` is a working path plus a
tensor that **knows it is provisional and says so**, and the October runbook's step 3
("build the tensors and simulation results") becomes one command against a path that has
already been exercised — which is exactly the rehearsal `docs/preseason-plan.md`'s runbook
asks for when it says the real window is too short to debug a join in.

## 4. Acceptance, stated before any result

1. `make docs-audit` **green** — 0 disagreements, 0 stale claims.
2. `make dashboard-audit` — orphaned artifacts **0**, pending provenance markers **0**
   (both are 0 today and must stay 0).
3. `.venv/bin/python -m pytest tests/` **green** (2,084 passing at the round's open).
4. `make production-check` **green** on the model half.
5. **The draft room opens a 2026-27 board** end to end, with true rookies and
   lag-recovered returnees rankable on it, from `sim_tensor_2026-27.npz` at the `full`
   window — **and says out loud that it was built with 0% preseason coverage** (C6). The
   staleness guard is *self-arming*: it passes in August, when no preseason exists
   anywhere, and refuses in October the moment `game_logs_pre_season_2026_27.csv` lands on
   disk beside a tensor that does not carry it.
6. Every artifact that moves has its `dashboard/decisions.py` entry updated with a fresh
   `reviewed` date.

## 5. The session ladder

### 5a. Session 1 — the frame, and the inventory that sizes the round

The plan doc, its `CLAUDE.md` router line, the round's opening decision-registry entries,
and the strengthening of `docs/final-evaluation-plan.md` §7 that C1 requires. Plus the
half that is measurement rather than prose: **which audited claims actually move**, read
off the claim registry rather than estimated, because the round's cost is the doc rewrite
and a wrong estimate mis-scopes every session after it. No artifact is regenerated.

### 5b. Session 2 — the production path, and the rehearsal that proves it

**The round's point.** Runs before the `train` re-run because it edits `sim/season.py`,
which the re-run then executes, and because it moves no audited figure — every claim it
creates is new.

Four pieces:

1. **A production unlock in the simulator**, on `posteriors.assert_production`'s own
   precedent. `--production` typed, not inferred from `--window full`; its own stated
   reason; **never** `held_out.unlocked`. Simulating 2026-27 is a deployment act, not a
   measurement, and the guard it needs is the deployment one. The existing
   `assert_season_allowed` raises for 2026-27 today with a message about the test split,
   which is the wrong diagnosis for a season that has no data to leak — the fit window is
   what has read the held-out seasons, not the target.
2. **`sim_tensor_2026-27.npz` at `full`**, built through `forward_board.forward_frames`
   (the season has no game log, so the retrospective builders have nothing to hand
   `build_context`) and persisted by the same `save_tensor`. A Makefile target and a Gate
   A treatment that keeps the production run out of the audited pooled table.

   **🔴 It is a REHEARSAL tensor, and the code has to know that (C6).** With no 2026-27
   preseason it is the August board — §1a. So `save_tensor` gains a preseason-coverage
   stamp beside `composition_variant` and `availability_layout`, for the reason those two
   are there: it changes the board materially and a consumer holding two tensors has no
   other way to tell them apart. On top of it, a **self-arming staleness guard** —
   *does the season's preseason log exist on disk, and does this tensor carry it?* In
   August both are no and the tensor is honest; in October the log lands and the same
   check turns the tensor into a hard refusal, with the rebuild command in the message. A
   coverage-is-zero warning would nag now and go quiet exactly when it matters; this
   fires only when it can be acted on, which is why the guard is written against the
   input's existence rather than against the coverage figure. `make production-check`
   gains the tensor as a row, so the readiness list stops ending one step short of the
   artifact the draft actually consumes.
3. **🔴 `load_field`'s cache key does not cover the tensor, and the live recommender is
   the one consumer that is exposed.** The key carries the field *configuration* —
   noise model, rank noise, need weight, seat composition, sim count — and the *filename*
   carries `--tensor-label`; nothing carries the tensor's content. So a rebuild at the
   same filename is invisible to it, and the caches on disk are already stale:
   `draft_room_field_2022-23.npz` is 2026-08-12 and `2023-24` is 2026-08-11, both older
   than the 2026-08-16 tensors they are read against. **The strategy sweep is immune, and
   for a narrower reason than it looks**: every arm it reports rebuilds the field on the
   tensor it is scoring (`gate_c`, `field_cut_line`, the realized replay and the pick-log
   stake all call `draft_room.build_field` themselves), so `Room.field_round` never
   reaches a reported figure. `priceable_room` rebuilds it too — but only on the
   symmetric path; `restrict=False` returns the room with its cached field intact, so the
   asymmetric arm carries a stale cache it happens not to consume. The blast radius today
   is therefore exactly `dashboard/draft_room.py` and `make draft-room-prep` — **draft
   night** — with the asymmetric arm one refactor away from joining it. Delete the stale
   caches, and add a fingerprint of the scored tensor to the key so the next population
   change cannot do this again.
4. **The draft room opens it.** `load_room`'s `assert_season_allowed` has to admit a
   production season on the evidence the tensor itself carries (`fit_window == "full"`),
   and `dashboard/draft_room.py`'s `sim_tensor_*.npz` glob has to stop reading
   `2022-23_rookieinclusive` as a season. The room surfaces the coverage stamp as a
   banner rather than burying it in provenance, because the person who needs it is
   looking at a board on a thirty-second clock. Verified end to end, not asserted.

Tests for the unlock (both directions), the fingerprint, the glob, the production season's
admission, and **both states of the staleness guard** — silent when the preseason log is
absent, refusing when it is present and the tensor predates it. The second half is the one
that matters and is the one a test can take today, since the October state is reachable by
writing the file.

**What §5b does NOT do**: ship a board anyone drafts off. That is the October crunch, and
after this session it is `make simulate-production` against a path that has been run.

### 5c. Session 3 — the `train` tensors and the cheap chain

`make simulate-season` on **2018-19, 2021-22, 2022-23, 2023-24**, then `weekly-scores`,
`bracket`, `draft-sim`, `draft-sim-need`. Minutes of compute, and the reproduction check
that licenses everything after it: the freshly-drawn 2022-23 and 2023-24 tensors against
§7i's labelled pair, reported rather than gated — the labelled pair was drawn from the
same seed on the same code, so a difference is a finding and an agreement is the check
that the shipped and labelled populations are one thing.

Rewrites the claims on `sim_season_gate_a.csv` (**41**) and `weekly_score_*` (**112**).
§1's second fact lands here: the weekly panel stops mixing two minutes injections.

### 5d. Session 4 — the sweeps

`strategy-sweep`, `strategy-sweep-need`, `rookie-floor`, `draft-room-prep`,
`ladder-board`. ~3 h, and the round's compute. C4 governs it: Gate C's rotation is
**solved** on this tensor and the sweep's arm is **selected** from this table, so both are
re-read.

Rewrites the claims on `strategy_*` (README **4**, `docs/simulations-plan.md` **2**,
`docs/rookie-rates-plan.md` ~**60**) and `availability_ladder_board.csv` (**70**).

### 5e. Session 5 — retiring the label, and reconciling §7i

Once the shipped tensor is the rookie-inclusive one, `make strategy-sweep-rookie` *is*
`make strategy-sweep` and `_rookieinclusive` names a population that no longer has a
variant to be a variant of. So:

- `rookie_recovery.TENSOR_LABEL` → `""`; the target reads the shipped tensor. Its **49**
  claims are re-derived, and if §5c's reproduction check held they do not move — which is
  itself the check.
- The two labelled Makefile targets retire; `--tensor-label` stays as machinery.
- The **labelled CSVs stay on disk** and the **labelled tensors go**. The CSVs are §7i's
  record and are what its ~30 claims are presence-checked against; the tensors are 195 MB
  of a population that now ships. (`make docs-audit` *skips* a missing artifact rather
  than failing it, so deleting a claimed CSV would silently retire its guard.)
- §7i's prose that rests on the two arms differing — "the two Gate A tables now differ",
  the ⚠️ closing bullet — is superseded under C3 rather than deleted.
- C5's provenance note: both frozen counterfactuals are now snapshots of a superseded
  chain, recorded in prose, with their numbers untouched.

### 5f. Session 6 — close-out

§4's six acceptance conditions, run rather than asserted; the `reviewed` sweep across
every moved registry entry; README's four figures; and the round's own "what this does not
settle".

## 6. Session runbook — the prompts

One prompt per session, for a fresh session each. Each ends by appending its results to §7
and updating `dashboard/decisions.py`.

1. ✅ **Done 2026-08-22 — §7a.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 1 (§5a)**: open the plan
   > doc, its router line, the opening registry entries, the `final-evaluation-plan.md` §7
   > strengthening, and the audited-claim inventory. Regenerate nothing.
2. ✅ **Done 2026-08-22 — §7b.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 2 (§5b): the production
   > path** — a typed `--production` unlock in `sim/season.py` on
   > `posteriors.assert_production`'s precedent (never the test-split unlock),
   > `sim_tensor_2026-27.npz` at the `full` window through `forward_board.forward_frames`,
   > a tensor fingerprint on `load_field`'s cache key with the stale caches deleted, the
   > draft room's season glob and season guard, and an end-to-end open of a 2026-27 board.
   > Per C6 the tensor is a **rehearsal** — no 2026-27 preseason exists — so stamp its
   > preseason coverage into `save_tensor`, add the self-arming staleness guard and its
   > `production-check` row, and surface the coverage in the room. Tests for each,
   > including both states of the guard. No audited figure moves.
3. ✅ **Done 2026-08-22 — §7c.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 3 (§5c): the `train`
   > tensors and the cheap chain** — `make simulate-season` on 2018-19, 2021-22, 2022-23
   > and 2023-24, then weekly-scores, bracket, draft-sim, draft-sim-need. Report the
   > reproduction check against §7i's labelled pair. Rewrite the 153 audited claims on
   > `sim_season_gate_a.csv` and `weekly_score_*`, superseding where an argument rests on
   > the old number.
4. ✅ **Done 2026-08-22/23 — §7d.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 4 (§5d): the sweeps** —
   > strategy-sweep, strategy-sweep-need, rookie-floor, draft-room-prep, ladder-board.
   > Re-run and re-read every selected figure (Gate C's rho and scale, the sweep's shipped
   > arm, the strategy table) rather than hand-editing it. Rewrite the ~136 audited claims
   > on `strategy_*` and `availability_ladder_board.csv`.
5. ✅ **Done 2026-08-23 — §7e.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 5 (§5e): retire the
   > `_rookieinclusive` label** — point `rookie_recovery` at the shipped tensor, retire the
   > two labelled targets while keeping `--tensor-label`, delete the labelled tensors and
   > keep the labelled CSVs, and supersede the §7i prose that rests on the two arms
   > differing. Record the frozen counterfactuals' provenance note.
6. ✅ **Done 2026-08-23 — §7f.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 6 (§5f): close-out** —
   > §4's six acceptance conditions run rather than asserted, the `reviewed` sweep, the
   > README figures, and the round's "what this does not settle".

## 7. Results

### 7a. Session 1 — the frame, and what the inventory says the round costs (run 2026-08-22)

No artifact was regenerated. What the session produced is this document, its router line,
four registry entries, the `docs/final-evaluation-plan.md` §7 edit C1 requires, and one
measurement.

#### The inventory, read off the claim registry

`src/docs_audit.py` carries **6,066** claims. Grouping them by the artifact each reads and
intersecting with the artifacts a tensor re-run writes:

| doc | claims on tensor-derived artifacts | of those, **move** |
|---|---:|---:|
| `docs/rookie-rates-plan.md` | 138 | **138** |
| `docs/availability-window-plan.md` | 132 | **98** |
| `docs/simulations-plan.md` | 119 | **119** |
| `docs/preseason-plan.md` | 101 | **0** |
| `README.md` | 4 | **4** |
| **total** | **494** | **359** |

**The round was scoped at 393 and the true figure is 359, and both halves of that
correction matter.** The 494 total is 101 larger than the scoping estimate because
`docs/preseason-plan.md`'s preseason-contest block was not counted; the *moving* figure is
135 smaller than 494 because that block and the mixture block do not move at all. Both are
C5: `make mixture-value` and `make preseason-contest` read frozen capture directories and
never the live artifacts, so re-running them after this round reproduces their numbers to
the digit. **135 audited claims that looked like work are not work** — and the two frozen
captures becoming snapshots of a superseded chain is a provenance note (§5e), not a
figure.

**And 93 of the 359 are already `historical=True`** — presence-checked rather than
value-checked, so they survive the round by construction provided nobody tidies them away.
The live rewrite is therefore **266 figures**, which is the number to plan §5c and §5d
against.

#### What the moving set is made of

| artifact | claims | written by |
|---|---:|---|
| `weekly_score_index.csv` | 95 | `make weekly-scores` |
| `availability_ladder_board.csv` | 70 | `make ladder-board` |
| `rookie_recovery.csv` | 49 | `make rookie-recovery` |
| `sim_season_gate_a.csv` | 41 | `make simulate-season` |
| `strategy_rookie_floor.csv` | 39 | `make rookie-floor` |
| `weekly_score_period.csv` | 15 | `make weekly-scores` |
| `strategy_injection.csv` | 10 | `make strategy-sweep` |
| `sim_season_gate_a_rookieinclusive.csv` | 10 | §7i, labelled |
| `strategy_realized_rookieinclusive.csv` | 10 | §7i, labelled |
| `strategy_injection_rookieinclusive.csv` | 7 | §7i, labelled |
| eight others | 13 | — |

#### What this document claims to `make docs-audit`, and what it deliberately does not

Five claims — the 2026-27 production census (**419** / **6** / **116** units, **197**
market-priced) and the **31** heads at `full` — which is what it takes to enter a doc into
the registry at all, on `docs/rookie-rates-plan.md`'s own incremental precedent. Two
populations are outside it on purpose:

- **The inventory figures above are properties of `src/docs_audit.py` itself.** Claiming
  6,066 or 359 there would be the registry checking itself: the value check would compare a
  literal in this doc against a number derived from the literals in this doc. They are
  guarded by being re-derivable in one pass over `CLAIMS`, which is what §7a's tables are.
- **The tensor censuses cannot be claimed yet.** 386/387 against 471/467 and their
  per-family splits are the figures that most deserve a guard, and they live in `.npz`
  files while `docs_audit.table()` reads CSV and parquet only. Recorded as a gap rather
  than worked around.

#### Three things the inventory settles

**1. The compute is not the round.** ~4 h of numpy against **266 live figures** to
re-read, most of them in argued prose that has to survive the re-read. §5c and §5d are each
a few hours of machine time and a day of documentation.

**2. `weekly_score_index.csv` is the single largest block, and it is stale for a reason
that predates the rookie program.** 95 claims, written 2026-08-15 off a pooled panel whose
2018-19 and 2021-22 halves carry a **shared** injected σ of 0.450 while its 2022-23 and
2023-24 halves carry the role-graded vector — §1's second fact. This round is the first
thing that has re-run `make weekly-scores` since the 2026-08-16 tensor rebuild.

**3. The live draft room is the exposed consumer, and it is exposed today.**
`load_field`'s cache key covers the field's configuration and not the tensor its round
totals were scored on, and the two validation caches on disk (2026-08-11 and 2026-08-12)
are already older than the tensors they are read against. The sweep escapes because every
arm it reports calls `draft_room.build_field` on the tensor it is scoring rather than
reading `Room.field_round`; `dashboard/draft_room.py` reads the cache as written, so draft
night prices our entry against a field drafted off a board that no longer exists. §5b fixes
it **before** §5c moves the tensors, which is the only ordering that does not widen the
window.

#### What Session 1 settles, and what it does not

- **Settled**: the round is 359 audited claims, not 393 — **266 of them live** and 93
  historical — and the 135-claim difference against the 494 total is two frozen
  counterfactuals that cannot move without a refit C2 forbids.
- **Settled**: C1 is recorded in both places it has to be — here, and in
  `docs/final-evaluation-plan.md` §7, which now says the held-out tensors are frozen
  rookie-less **by decision** rather than merely stale.
- **Settled**: the ordering. The production build runs first because it edits the module
  the re-run executes and because it closes a live defect in the draft-night path.
- **Settled, and it changed the round's acceptance**: the production board is
  **October-gated on the preseason block**, measured rather than assumed — 0 of 489
  availability rows, 0 of 425 component rows and 0 of 116 rookie rows carry one, and
  `preseason.parquet` ends at 2025-26 (§1a). The round therefore ships the production
  *path* and a rehearsal tensor that knows it is provisional, not a board to draft off
  (C6). **This was a real error in the round as originally scoped**, which read "persist
  `sim_tensor_2026-27.npz` at `full`" as the deliverable and would have produced the
  August board under the shipped name with nothing saying so.
- **Not settled**: whether the freshly-drawn 2022-23 and 2023-24 tensors reproduce §7i's
  labelled pair. `src/sim/season.py` was edited after the labelled tensors were written
  (2026-08-22 20:17 against 19:53), and the edit *looks* additive — `unit_family` and
  `lag_rung` in `save_tensor`, the `tensor_label` plumbing — but "looks additive" is not a
  reproduction check. §5c takes it and reports it either way.
- **Not settled, and deliberately not**: what the wider population is worth in contest
  units on the shipped chain. `docs/rookie-rates-plan.md` §7i priced it at the cut line and
  refused to resolve it in the contest at N = 2 realized seasons; re-running the sweep on
  the shipped tensor does not add a season. §5d re-reads the sweep because the artifacts
  have to describe the tensor on disk, not because two realized seasons have become
  informative.

### 7b. Session 2 — the production path, built and exercised end to end (run 2026-08-22)

All four pieces shipped, the board opened, and **no audited figure moved**: Gate A gained
no rows (the production run skips it by construction), `make docs-audit` reads 0
disagreements / 0 stale, and `make dashboard-audit` still reads 0 orphaned artifacts / 0
pending markers. The figures below are new and, like §7a's, deliberately outside the claim
registry — the tensor census lives in an `.npz` the auditor cannot read (§7a records the
gap), and everything else here is re-derivable by `make simulate-production` plus the
tests.

#### The unlock — `assert_production_season`, three refusals

`sim/season.py --production`, on `posteriors.assert_production`'s precedent and **never**
`held_out.unlocked`. The old refusal for 2026-27 was `assert_season_allowed`'s test-split
message — the wrong diagnosis for a season with no data to leak, since what has read the
held-out seasons is the `full` fit window, guarded at fitting time. The new guard refuses,
each with its own reason: (1) an **unplayed season without the typed flag** — a window is
a value that gets passed around, an unlock is an act somebody performs; (2) a **played
season under `--production`** — C1's second door, closed: the flag can never redraw a
held-out (or any played) tensor; (3) any **window but `full`**. "Unplayed" is read from
`component_targets.parquet` and never from a design frame, because the forward path
CONCATS target-season rows into the design and the split labels SHIFT — the guard must not
believe the frame it guards (`bracket.split_frame`'s own argument, one artifact over).

#### The tensor — `make simulate-production`

`sim_tensor_2026-27.npz` at `full`, through `forward_board.forward_frames` (lazy-imported
by `run`; the target defaults to the live DK board's season via `production_check`'s own
`next_season`, so the checklist and the build cannot disagree about which season is next).
**541 scorable units — 419 rung-0 veterans + 6 lag-recovered returnees + 116 true rookies
— × 20 periods × 2,000 sims, 108 MB**, matching §1a's census exactly. Stamped
`preseason_coverage = 0.0%`, `preseason_log_rows = 0` (C6), beside `composition_variant`
and `availability_layout` for the reason those two are there. Gate A is skipped with its
reason printed: no realized 2026-27 exists to reproduce, so the pooled audited table gains
no rows.

#### 🔴 The rehearsal caught two real October bugs, which is the argument for it

1. **The rookie floors had no forward prior tier.** The expanding draft-bucket prior
   travels inside each persisted floor artifact and covers the seasons the fitted design
   carried, so the first attempt to score a 2026-27 rookie raised
   `rookie_rates.floor_level`'s own refusal — correct, and terminal for the October crunch
   had it surfaced there. The fix is `season.extend_rookie_priors`: the target season's
   row pools played rookie rows STRICTLY before it (the same expanding rule as every other
   row — no refit, no leakage, knowable in September), appended **in memory** to the
   floor steps at score time, production path only. The pickle on disk stays exactly what
   `make posteriors-production` wrote; 10 heads extended on 2026-27 (`reb` is fitted, not
   a floor).
2. **`no_design_availability` could not index the target season.**
   `availability_no_prior.appearance_gap` takes its season order from `data.seasons`,
   which lists seasons with *data* — a production target is one past its end and raised
   `KeyError: '2026-27'`. `build_context` now appends the target label when absent, which
   is exact: the list is ascending and an unplayed season sorts after every played one.

One observation, reported not gated: **7.2%** of simulated team-games (354,631 of
4,920,000) needed the fewer-than-five-available feasibility repair, far above a
retrospective season's rate — the roster snapshot's 577-player membership gives the
allocation thinner team tails than a played season's grid does. Worth re-reading in
October on the real 19-man rosters before treating it as a property of the board.

#### The cache key — `load_field` now covers the tensor

`draft_room.tensor_fingerprint` (blake2b over the tensor file) rides on `save_field` /
`load_field` beside the configuration keys. A cache without the fingerprint is **never**
served to a caller who names one — `load_room` always does now — so the next population
change cannot silently price draft night against a board that no longer exists. Content
rather than mtime, so copying or touching a tensor cannot invalidate a field that still
describes it. The two stale validation caches (`draft_room_field_2022-23.npz` 2026-08-12,
`2023-24` 2026-08-11) are **deleted**; the legacy caches that remain (2025-26, the two
labelled ones) self-invalidate on their next open at ~12 s each. The sweep's arms still
build their own fields and never touch the cache, exactly as §7a measured.

#### The room opens it, and says what it is

- `load_room` admits the season on the evidence the tensor carries —
  `season_admissible`: `fit_window == "full"` is a window only the production unlock can
  have written, and the split frame is not even built to check it (a callable, unpaid).
  Every other tensor still faces `assert_season_allowed`.
- `load_room` passes `raw_dir`, arming `season.assert_tensor_current` — **the self-arming
  staleness guard**: silent today (no 2026-27 preseason exists anywhere), a hard refusal
  naming `make preseason && make simulate-production` the moment
  `game_logs_pre_season_2026_27.csv` lands beside a tensor stamped `log_rows = 0`.
  Unstamped (pre-§5b) tensors are skipped — their retrospective builders cannot
  desynchronize this way.
- The season glob takes `NNNN-NN` only, so `2022-23_rookieinclusive` no longer parses as
  a season the room then fails to open; 2026-27 lists.
- The coverage stamp is a **banner** — "REHEARSAL board … the August board. Practice on
  it; do not draft off it." — plus a sidebar coverage line, because the person who needs
  it is on a thirty-second clock.
- `make production-check` gains the tensor as a row with three states (missing /
  ready-as-REHEARSAL / **STALE**), read through `assert_tensor_current` itself so the
  checklist and the load-time refusal cannot disagree. Today: `ready … 2,000 sims at
  `full`, preseason coverage 0% — REHEARSAL`.

**Verified end to end, not asserted**: `load_room(cfg, "2026-27")` opens in 13.4 s on the
first launch (field build + fingerprint save; cache hit thereafter), the board carries
942 pool players with **457 priceable**, and a full first-pick `evaluate` returns 457
ranked candidates in **0.22 s** against Gate E's 1.0 s — with **34 true rookies and all
6 lag-recovered returnees rankable** (best returnee: Tyrese Haliburton, table rank 45).
The top of the board is Wembanyama / Jokić, i.e. the August board behaving like one.

#### Tests

17 new, all green in the full suite (2,101): the unlock's three refusals and its
admission with the split still locked, `unplayed_season` against an injected-design
shift, `preseason_log_rows`' header-only convention, the coverage share, **both states
of the staleness guard** (August-silent and October-refusing, the October state taken
today by writing the file) at the guard and again at `load_tensor`'s armed read, the
`save_tensor` stamp round-trip, the forward prior tier (value, count, idempotence, and
the pickle's own rows untouched), the fingerprint key (mismatch, legacy-cache refusal,
content-not-name), `season_admissible` both ways, the room's glob, and
`production_check.tensor_row`'s three states.

#### What Session 2 settles, and what it does not

- **Settled**: the October runbook's step 3 is now one command against a path that has
  been run — `make simulate-production`, rebuilt after `make preseason` — and the two
  bugs that would have been debugged inside the three-day window were instead fixed with
  two months of slack.
- **Settled**: the draft-night cache defect (§7a's third finding) is closed *before*
  §5c moves the tensors, the only ordering that does not widen the window.
- **Not settled, by design (C6)**: everything about what the 2026-27 board is *worth*.
  The tensor is the August board and says so on its face; no figure from it is quotable
  as a production number, and the board anyone drafts off does not exist until October.
- **Not settled**: the 7.2% feasibility-repair rate on the forward grid (above), and the
  five October-gated season rows `make production-check` still lists as missing —
  which before October is the schedule, not a defect.


### 7c. Session 3 — the `train` tensors, the cheap chain, and the reproduction check (run 2026-08-22)

`make simulate-season` on 2018-19, 2021-22, 2022-23 and 2023-24 (default seed 0, 2,000
sims, `train` window), then `make weekly-scores`, `make bracket`, `make draft-sim` and
`make draft-sim-need`. ~13 minutes of numpy end to end, and the doc rewrite it licensed: all **153**
audited claims on `sim_season_gate_a.csv` and `weekly_score_*` were re-read — 41 + 112,
exactly §5c's scoping — of which **93** were already historical and survived by presence,
and the **60** live ones were either refreshed from the re-run artifacts or superseded
under C3. Two claims outside the 153 moved with them (below).

#### The reproduction check: bit-for-bit, which retires §7a's open item

The freshly-drawn 2022-23 and 2023-24 tensors against §7i's `_rookieinclusive` labelled
pair, key by key: **every shared array is bit-identical** — `dk_pts`, `games_played`,
`season_minutes`, `player_id`, `prior_minutes`, `unit_family`, `lag_rung`, both sigma keys
and all provenance scalars — and every Gate A cell agrees to the last digit. The only
difference is two keys the fresh tensors carry that the labelled pair predates:
`preseason_coverage` and `preseason_log_rows`, §5b's stamp, which `save_tensor` now writes
unconditionally. So Session 2's edit was additive in exactly the way §7a refused to assume,
the shipped and labelled populations are one thing, and §5e's plan to re-derive
`rookie_recovery`'s 49 claims expecting no movement is licensed. The labelled tensors are
now 195 MB of duplicate bits and §5e deletes them; the labelled CSVs stay as §7i's record.

#### The four tensors, and Gate A re-read over them

Unit censuses by ladder rung (from the `.npz` — the registry cannot claim these, §7a's
recorded gap): **444** = 366 rung-0 + 6 `returnee_lag2` + 72 true rookies on 2018-19,
**487** = 400 + 6 + 81 on 2021-22, **471** = 386 + 13 + 72 on 2022-23 and
**467** = 387 + 6 + 74 on 2023-24 — each season's rung-0 count reproducing its rookie-less
census exactly, the same disjoint-remainder check §1 ran on the labelled pair.

| season | n | MAE | CRPS | R² | bias |
|---|---:|---:|---:|---:|---:|
| 2018-19 | 444 | **378.04** | **261.60** | 0.6957 | −53.90 |
| 2021-22 | 487 | **344.41** | **241.72** | 0.6976 | −36.12 |
| 2022-23 | 471 | 345.43 | 238.67 | 0.7323 | −16.54 |
| 2023-24 | 467 | 362.11 | 250.35 | 0.7209 | −52.18 |

The validation rows are the labelled pair's to the digit (the reproduction check from the
artifact side). The training rows move enormously — 2018-19 MAE 437.51 → 378.04, R²
0.5892 → 0.6957; 2021-22 MAE 390.17 → 344.41, R² 0.6341 → 0.6976 — and almost none of that
is the rookie program: those two tensors were two config generations stale (§1's second
fact), so the re-draw handed them the 0.375 preseason-blended composition offset and the
role-graded injection in the same pass that widened the population. Entangled by design;
this table is not a rookie-value measurement, and §7i's cut-line ladder remains the place
the population change is priced.

Reported, not gated: the feasibility-repair rate reads **0.16–0.28%** of simulated
team-games across the four seasons (8,419 / 7,897 / 13,530 / 8,280 of 4,920,000), against
0.12–0.18% on the rookie-less population — more thin-tail units, more short-rotation
repairs, and nothing like §7b's 7.2% forward-grid rate.

#### The weekly panel — 37,380 player-periods, and the mixed-injection debt paid

`make weekly-scores` re-read the pooled panel at **37,380** player-periods (30,780
rookie-less), and §1's second fact lands: the panel no longer mixes two minutes
injections. The full readout lives beside the facet table in `docs/simulations-plan.md`;
the shape of it is that every observed column falls (the ~6,600 added player-periods are
rookies and returnees who score little — a population change under the mean, never an
improvement), the train-side fit jump is the injection fix (one-week train R² 0.3985 →
0.4720, MAE 29.93 → 26.60), zero weeks rise toward a quarter of the panel on both sides
of the comparison, and the KS span narrows at the wide end (0.0127–0.0582 →
0.0143–0.0491). The per-period bias profile stays flat and shallow (−1.18 to −1.80
over the five claimed weeks), so `tenure_merge`'s shape result survives the population.

#### Two claims outside the 153 moved, and one invariance assumption broke

- **§7i's two "units the union adds" claims (85, 80)** read `GATE_A_ROOKIE minus
  SIM_GATE_A`, whose live difference is now 0 by reproduction. Superseded as statements
  about the rookie-less pair, beside §7i's shipped-column claims (360.80 / 373.63 and
  friends), which went historical the moment the shipped table became the union.
- **`docs/availability-window-plan.md` §8b's realized no-design league share was
  value-checked as chain-invariant, and it is not population-invariant.** §16j's ladder
  gives `returnee_lag2` players availability design rows, so this re-run moved 13 / 6 of
  them out of the no-design population and the *realized* bar fell 0.1057 → **0.0889**
  (2022-23) and 0.0992 → **0.0928** (2023-24). The pre-ladder readings are held
  historical; the ladder-on rows are claimed live from that doc.

#### The rest of the cheap chain — re-run, and nothing audited moved there

`make bracket`: the symmetric-field null covers −rake on all 10 season × tournament
rows, every prize pool pays out to the cent, and the field-size chain agrees with
`economics.advance_table` on all 20 rounds. `make draft-sim` re-fits Gate B on the wider
board and **selects the same field model** — `tiered` noise at `rank_noise_sd = 4.00`,
MAE(fit) 5.922 against the 17.0-pick bar — and `make draft-sim-need` again selects
`need_weight = 0`, nesting the shipped pure-ADP field, so the measured-null verdict on
field lineup reasoning survives the population change. No claim reads these artifacts;
they are re-run so nothing downstream of the tensors describes a board that no longer
exists.

#### What Session 3 settles, and what it does not

- **Settled**: the reproduction check. Same seed, same code, same tensors to the bit —
  the difference §7a could not rule out does not exist, and every §7i comparison built on
  the labelled pair now describes the shipped population too.
- **Settled**: the weekly panel's mixed-injection debt (§1's second fact), which would
  have been worth this round with no rookie on the board.
- **Not settled, deliberately (C4)**: everything the sweep owns. `strategy_*.csv`,
  `availability_ladder_board.csv`, `rookie_recovery.csv` and the draft-room prep still
  describe the rookie-less tensors until §5d re-runs and re-reads them — Gate C's rho and
  scale and the sweep's shipped arm are selections, re-read there rather than here.
- **Not settled**: what the wider population is worth in contest units — §7a's closing
  bullet stands; re-running the chain adds no realized season.

### 7d. Session 4 — the sweeps, re-run and re-read (run 2026-08-22/23)

`make strategy-sweep` (53 min), `make strategy-sweep-need` (52), `make rookie-floor` (53),
`make draft-room-prep` (2) and `make ladder-board` (2) — ~2.7 h of numpy against the
re-drawn tensors, and the re-read C4 demanded. Wall clocks measure the machine; the two
short ones are real (the prep consumed the fingerprinted field caches the night's own runs
had already written, and the ladder board's ~25 min estimate dated from a contended
machine). Like §7b's and §7c's, the figures here are quoted from artifacts other docs'
builders claim — this doc's own registry entry stays at its five claims.

#### The reproduction, which makes the re-read a confirmation rather than a movement

§7c proved the re-drawn 2022-23 / 2023-24 tensors bit-identical to §7i's labelled pair;
this session's sweep inherits that all the way to the artifacts: **`make strategy-sweep`
reproduces `make strategy-sweep-rookie` bit-for-bit** — the two uninjected Gate C worlds
agree to a max gap of **0.0** across all 12 rows, all **240** sweep rows and all **240**
realized-replay rows match to the last digit, and the injection records agree on `rho`,
`scale_g`, `achieved_mae` and `n_board` exactly. Every §7i comparison that read
"labelled minus shipped" therefore reads **0** from the live artifacts now; those claims
went historical under C3 and §7i carries the supersede note. The un-paired caveat §7i led
with retires with the comparison it qualified.

#### The selections, re-read (C4)

- **Gate C's rotation and scale**: `rho` **0.3443** (2022-23) and **0.2825** (2023-24),
  `scale_g` **1.1875** and **1.1811**, `achieved_mae` 400.4586 on target both seasons.
  The rookie-less 0.3582 / 0.3172 and 1.1337 / 1.1405 are held historical.
- **The shipped arm**: `lineup_value_blend30` re-selected on **all four multi-entry
  structures** — sim lift 0.2645 / 0.2862 / 0.2764 / 0.2744 (600k / 20k / 50k / 15k) —
  with the single-entry 88k staying at `lineup_value`, §7i's one flip, unchanged.
  `strategy_shipped.csv`'s 600k headline reads **0.2645 simulated / 0.3112 realized**, and
  README's rounded pair moved 0.24 → **0.26** and 0.20 → **0.31**.
- **Gate B's fitted noise** (consumed, not re-fitted here): §7c's re-fit on the wider
  board — `tiered` at `rank_noise_sd = 4.00`, pooled MAE(fit) 5.9224 against the 17.0
  bar — is what every field in this session drafted under. Unmoved.
- **The strategy table**: 24 arms, 500 worlds per season, the same ordering shape —
  stacking and tight exposure caps still lose, the autodraft twin still equals
  `blend_caps_dk` roster-for-roster and still beats the uncapped click (**+0.0125086**
  at the 600k), and the shipped objective is still what automation gives up
  (**0.0855400**) — the 2026-08-16 execution pair went historical beside the fresh one
  in `docs/simulations-plan.md`.

#### The rookie floor, re-read on the board that prices both families

`make rookie-floor`'s asymmetry now prices the market-silent fringe, not the rookie class:
our seat takes **411 of 448** rows (2022-23) and **417 of 464** (2023-24), the 37 / 47
masked rows carry **0 / 3** ADP prices, and the field takes **0.0000 / 0.0111**
unpriceable players per entry against 1.26 / 1.19 before. The cut line the field sets
moves **+25.4** and **−29.0** dk_pts — and reproduces `make rookie-recovery`'s
`rookie`-rung residual **to the digit** (+25.416667 / −28.958333), §7i's rung-0 licence
check arriving one rung up, out of the separate module. Gate C reproduces the shipped run
at **0.0** in *both* arms. The contest half still refuses to resolve and no longer even
trends: shipped arm **−0.0496** over eight readings spanning −0.2937 to +0.0440, 18 of 24
arms losing at a median of −0.0225. And the simulated arm's uniform positivity — §5a's
predicted artifact — collapses with its cause, **+0.0603 → +0.0057**, now that the field
wastes ~0 picks on zero-scoring rows. §7a of `docs/rookie-rates-plan.md` carries the full
supersede block; its rookie-less table is the historical record.

#### The rest of the chain

- **The need probe** (`strategy-sweep-need`): the direction holds a third time — all 24
  arms read higher lift against the stipulated 8-pick lean pooled over the multi-entry
  structures (shipped arm 0.2645 → 0.3089 at the 600k) — and the fitted pure-ADP field
  stays the conservative opponent. The one flip is the single-entry 88k, the table's
  noisiest cell.
- **Draft-room prep**: fields rebuilt and served through §7b's fingerprinted cache key on
  its first production exercise; the symmetric-field null reproduces to **−2.53e-08**;
  Gate E passes at **114-125 ms** mean (max 209) against the 1,000 ms bar on the union
  boards. No audited claim reads these artifacts; they are re-run so draft night does not
  price against a board that no longer exists.
- **The ladder board**: `make ladder-board` re-run on the re-drawn chain writes a CSV
  **byte-identical** to §16j's 2026-08-22 artifact — all 70 audited claims on
  `availability_ladder_board.csv` confirmed with zero movement, which is what "the
  artifacts describe the tensor on disk" looks like when the reading never depended on
  the persisted tensor in the first place. §16j's stale "not re-run" bullet is closed.

#### The claim ledger

**65 claims went historical** under C3 — 42 on the §7a floor block, 19 on §7i's
labelled-versus-shipped comparisons, 2 on §7b's census cross-check (the drafting layer's
ADP-priced hole now reads 0 / 3), and the 2026-08-16 execution-axis pair — and **45 live
claims were added** over the re-run artifacts, including the floor's union-board readout,
the live rho / scale / board twins, three exact-reproduction claims (Gate C gap, sweep,
realized replay), and the two cross-module cut-delta agreements. The registry stands at
**6,193 claims, 0 disagreements, 0 stale**; `make dashboard-audit` still reads 0 orphaned
artifacts and 0 pending markers. Beyond the scoped set, four unclaimed prose sites were
annotated rather than left to drift: the priceable-board blocks in
`docs/simulations-plan.md`, the validation reference pair in
`docs/final-evaluation-plan.md` §4d, and the shipped-lift endpoints in
`docs/draw-time-calibration-plan.md`.

#### What Session 4 settles, and what it does not

- **Settled**: the sweep artifacts describe the tensors on disk. Every selection survived
  the re-read — same shipped arm on every tier that selects, Gate C re-solved rather than
  hand-edited, Gate B unmoved — and the reproduction against §7i's labelled run is exact,
  so the round's §5e can retire the label knowing the two populations are one thing at
  the artifact level, not only at the tensor level.
- **Settled**: the rookie floor's live artifact carries the closed reading, measured by
  two modules that agree to the digit, and the old +173.1 / +111.9 is a historical record
  rather than a live figure anywhere.
- **Settled, incidentally**: the mechanism of the simulated-arm artifact (§5a's
  prediction) — it was the field's zero-scoring picks, and it is gone with them.
- **Not settled, deliberately (§7a's closing bullet, unchanged)**: what the wider
  population is worth in contest units. The re-run added no realized season; the contest
  half of the floor straddles zero exactly as it did at N = 2, and nothing in this
  session is contest evidence.
- **Not done here**: `rookie_recovery.csv` still reads the labelled tensor and the two
  labelled Makefile targets still exist — §5e's session, which also owes the C5
  provenance note on the two frozen counterfactuals.

### 7e. Session 5 — the label retired, and §7i reconciled (run 2026-08-23)

No compute beyond one minutes-long re-run of `make rookie-recovery`, and no audited figure
moved — which was the session's own acceptance test, licensed in advance by §7c's
bit-for-bit reproduction. The registry stands where §7d left it: **6,193 claims, 0
disagreements, 0 stale**.

#### `rookie_recovery` reads the shipped tensor, and its 49 claims did not move

`rookie_recovery.TENSOR_LABEL` is now `""` and `make rookie-recovery` re-derived
`rookie_recovery.csv` off the shipped 2022-23 / 2023-24 tensors. Compared against a copy
of the labelled run: **every value column is identical** (max absolute numeric difference
0.0 across all 8 rows × 13 columns); the only change in the file is the `tensor_label`
column reading empty instead of `_rookieinclusive`. All 49 audited claims re-derive to the
same figures — the non-movement §5c predicted, observed rather than assumed. Before
anything was deleted, the shipped pair was re-checked against the labelled pair one last
time: all 20 shared arrays bit-identical on both seasons, unit censuses equal
(471 = 386 + 13 + 72 and 467 = 387 + 6 + 74), the shipped side differing only by the two
preseason-stamp keys §5b added.

#### What was deleted, what was retired, and what stays

- **Deleted**: `sim_tensor_2022-23_rookieinclusive.npz` and
  `sim_tensor_2023-24_rookieinclusive.npz` (§7c's 195 MB of duplicate bits), plus their
  two field caches `draft_room_field_<season>_rookieinclusive.npz` — caches over tensors
  that no longer exist, which nothing can ever open again once the label has no default
  reader.
- **Retired**: `make strategy-sweep-rookie` (it *was* `make strategy-sweep` by §7d's
  bit-for-bit reproduction — a target that re-runs another target under a second name is
  a trap, not a convenience), and `rookie-recovery`'s labelled default. Both targets'
  Makefile comments now say where the label went. `--tensor-label` stays end to end —
  tensor, cached field, Gate A, every sweep artifact — as the machinery for the next
  population change.
- **Kept, deliberately**: all nine labelled CSVs — `sim_season_gate_a_rookieinclusive.csv`
  and the eight `strategy_*_rookieinclusive.csv` — as §7i's frozen record. They carry its
  ~30 presence- and value-checked claims, and `make docs-audit` *skips* a missing artifact
  rather than failing it, so deleting a claimed CSV would silently retire its guard.

#### §7i reconciled under C3

Two supersede notes landed in `docs/rookie-rates-plan.md` §7i beside the prose §7c/§7d had
already flagged: the section opener (which names the labelled tensor and the retired
target as its reproduce commands) and the ⚠️ closing bullet ("the shipped tensors on disk
are still the rookie-less ones"), which now carries the full resolution — including the
one clause the round measured wrong before it could matter: a tensor re-run does *not*
move "the mixture arms", because `make mixture-value` reads its frozen capture (C5). The
doc-header paragraphs for §7g and §7i carry matching one-line notes, and `docs/pipeline.md`
and `docs/simulations-plan.md`'s target inventories follow the Makefile. Old figures stay
in place under the audit's historical flag; nothing was tidied away.

#### The C5 provenance note, recorded where the captures are read

Both frozen counterfactuals are now snapshots of a superseded chain, recorded in prose
with their numbers untouched: `docs/availability-window-plan.md` §7l (the mixture pair,
`outputs/predictions/mixture_arms/`, captured 2026-08-12) and `docs/preseason-plan.md`'s
paired-counterfactual chain section (the preseason block, `preseason_arms/`, captured
2026-08-15). Each note says the same three things: the target reads only its capture and
reproduces to the digit; what changed is what the capture *describes* — the rookie-less
chain of its capture date; and re-capturing would need the counterfactual half refitted,
which C2 forbids and no open question needs. The 135 claims C5 protects stay exactly
where they were.

#### What Session 5 settles, and what it does not

- **Settled**: the label is retired with nothing resting on it. The two populations were
  proven one thing at the tensor level (§7c), the artifact level (§7d), and now the last
  consumer (`rookie_recovery`) reads the shipped chain with zero movement.
- **Settled**: the C5 provenance debt. Both capture directories are documented as
  snapshots of the superseded chain, in the docs whose claims read them.
- **Not settled, unchanged (§7a's closing bullet)**: what the wider population is worth
  in contest units — this session added no realized season and re-read nothing the sweep
  owns.
- **Left for Session 6 (§5f)**: the acceptance sweep run rather than asserted, the
  `reviewed` pass across the registry, README's four figures, and the round's "what this
  does not settle".

### 7f. Session 6 — close-out: the acceptance sweep taken, the backlog cleared (run 2026-08-23)

Nothing was regenerated and no audited figure moved. The session's product is §4's six
conditions **taken** on the working tree as it stands, the registry's 296-entry
reviewed-date backlog worked off rather than waved through, and this section.

#### The six acceptance conditions, run

1. **`make docs-audit` green** — 6,193 claims, **0 disagreements, 0 stale**: 4,935
   value-checked against their artifacts, 1,226 superseded values presence-checked, 32
   cost figures presence-checked. Re-run after every edit this session made.
2. **`make dashboard-audit`** — **0** missing artifacts, **0** orphaned artifacts, **0**
   pending markers (all three held), and the stale-entry count went **296 → 0** under the
   `reviewed` sweep below. Total findings: **0**, the first zero since the check gained
   the reviewed-date arm.
3. **`.venv/bin/python -m pytest tests/`** — **2,101 passed** (full suite, re-run after
   the registry edits; the registry-touching subset re-run again and green).
4. **`make production-check`** — model half **DONE**: 31 of 31 heads at `full`, all
   matching the `train` specification on the ten-column comparison. The tensor row reads
   `ready … 2,000 sims at full, preseason coverage 0% — REHEARSAL: built before the
   preseason existed; rebuild after the October fetch`. The five missing season rows are
   the 2026-27 schedule, not defects (§7b).
5. **The draft room opens the 2026-27 board, re-verified end to end** rather than carried
   forward from §7b. Engine: `load_room(cfg, "2026-27")` opens in **2.1 s** on the
   fingerprinted field cache (13.4 s cold in §7b), 942 pool players with **457
   priceable**, and a full first-pick `evaluate` returns **457 ranked candidates in
   0.22 s** against Gate E's 1.0 s — with **34 true rookies and all 6 lag-recovered
   returnees rankable**, best returnee Tyrese Haliburton at rank 45, reproducing §7b's
   census exactly; the top of the board is still Wembanyama / Jokić / Dončić, the August
   board behaving like one. Dashboard: under Streamlit's `AppTest` (the tool
   `docs/docs-audit.md` prescribes), the standalone room's season selectbox defaults to
   `2026-27`, the 🚧 **REHEARSAL board** warning banner renders with coverage 0% and the
   rebuild command, and the sidebar carries the coverage line. The staleness guard stayed
   silent — no `game_logs_pre_season_2026_27.csv` exists — and its October refusal arm is
   covered by §5b's test, which takes that state by writing the file.
6. **Every moved artifact's registry entry re-verified with a fresh `reviewed` date** —
   the sweep below, including the round's own two `open` entries closed
   (`the-rookie-inclusive-tensors-are-not-rebuilt-yet` → `built` over the shipped
   tensors; `shipping-the-wider-population-is-a-documentation-round-not-a-compute-round`
   → `measured`, carrying the close-out line).

#### The `reviewed` sweep — what it covered, how, and what it did not

**In scope: 355 of the registry's 382 entries** — the 296 pre-existing stale findings,
every entry sourced from a doc this round edited (stale the moment the round commits),
and every entry whose `reproduce` names a re-run artifact. Not a blind date bump; three
instruments, then judgment per entry:

- a **figure scan** of every in-scope entry's quoted numbers against its source doc;
- a **historical-only scan** — entries quoting a figure their doc now holds only under
  `Claim(historical=True)` — which flagged 35, each triaged (most are dated measurement
  records whose figures properly went historical in earlier rounds and whose decisions
  stand);
- a **per-doc git-diff review** across all 18 backlog docs, from each doc's oldest stale
  review date to HEAD, reading the removals for reversals. All were doc evolution —
  planning tables replaced by shipped readings — and none reversed an entry's standing.

**14 entries got text corrections** where the words no longer described the artifact or
the doc: dated supersession notes on `gate-a-season-simulator-marginals`,
`composition-game-level-dispersion-too-wide`, `copula-is-not-the-residual-matrix`,
`the-p5-chain-re-run-reads-higher-and-cannot-attribute-it`,
`sim-tensor-is-player-by-period`, `weekly-scores-are-gate-a-at-the-unit-the-lineup-is-set-at`,
`the-held-out-chain-does-not-confirm-the-drafting-edge` (the ~0.24 / ~0.20 reference pair
dated, mirroring §4d's own annotation) and `the-mixtures-contest-value-is-confounded-not-measured`;
the `production-check` entry gained the tensor row; `pick-log-stake-execution-priced`
gained its provenance line (below); `cross-season-join` gained the preseason term the
2026-08-13 problem-statement change added; `dr-figures-still-prose-only` its
fourteen-figure framing; `minutes-window-does-not-move-the-injected-sigma` the 0.375 note;
and `availability-worth-211-dk-pts` its vintage note (the 07-30 test readings, the
validation re-derivation, and the held-out −211.1288 that landed back on the headline).
**331 `reviewed` dates were refreshed to 2026-08-23**; the other 24 in-scope entries were
already at 2026-08-23 from Sessions 4–5.

**What the sweep did not do**: re-read all 382 entries line by line — the 27 out-of-scope
entries (no stale finding, source untouched, artifact unmoved) were left exactly as they
were, and the scans verify figures and reversals, not every clause of prose. That is the
judgment the backlog instruction called for, reported as such.

#### README, verified rather than re-edited

The four tensor-derived figures re-derive from their artifacts: **24** strategies and
**500** worlds (unchanged values), and the 600k pair **0.2645 → ~0.26** simulated /
**0.3112 → ~0.31** replayed, edited in §5d and correct as they stand. No README edit this
session; `make docs-audit` re-derives every README claim and is green.

#### What this round does not settle

- **What the wider population is worth in contest units.** §7a's closing bullet, standing
  at the round's close exactly as at its open: no session added a realized season, the
  floor's contest half straddles zero at N = 2, and §7i's cut-line ladder remains the
  pricing instrument.
- **The board.** The 2026-27 tensor is the **August rehearsal** and says so on its face.
  The board anyone drafts off does not exist until the October crunch: `make preseason &&
  make simulate-production` after the final preseason game, a roster refetch, and the
  timing-matched mid-October ADP capture (`docs/adp-plan.md` A0). The path is exercised;
  the data is not here yet, by construction (C6).
- **The 7.2% forward-grid feasibility-repair rate** (§7b) — to be re-read on the real
  October rosters before it is treated as a property of the board.
- **The pick-log stake.** `strategy_pick_log_stake.csv` / `_paired.csv` (2026-08-11) were
  deliberately not among §5d's five targets and still describe the rookie-less chain; its
  registry entry now says so. The reading's use — execution is objective-dependent, and
  moot for data capture — is not population-sensitive, and re-running it is a decision
  for whoever next needs the number.
- **The standing asymmetry C1 chose**: the `train` and `full` tensors carry two rate
  families and the `train_val` pair carries one, permanently. Settled rather than open —
  restated here because §7f is the last place a reader looks before assuming the mismatch
  is an oversight.
- **The docs-audit gap §7a recorded**: the tensor censuses live in `.npz` files the
  auditor cannot read, so the figures that most deserve a guard still have none.
