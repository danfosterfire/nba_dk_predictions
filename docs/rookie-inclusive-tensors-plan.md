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

## ✅ STATUS: SESSION 1 RUN 2026-08-22 (§7a). SESSIONS 2-6 OPEN.

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
2. > I'm working on the NBA prediction project (CLAUDE.md). Read
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
3. > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 3 (§5c): the `train`
   > tensors and the cheap chain** — `make simulate-season` on 2018-19, 2021-22, 2022-23
   > and 2023-24, then weekly-scores, bracket, draft-sim, draft-sim-need. Report the
   > reproduction check against §7i's labelled pair. Rewrite the 153 audited claims on
   > `sim_season_gate_a.csv` and `weekly_score_*`, superseding where an argument rests on
   > the old number.
4. > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 4 (§5d): the sweeps** —
   > strategy-sweep, strategy-sweep-need, rookie-floor, draft-room-prep, ladder-board.
   > Re-run and re-read every selected figure (Gate C's rho and scale, the sweep's shipped
   > arm, the strategy table) rather than hand-editing it. Rewrite the ~136 audited claims
   > on `strategy_*` and `availability_ladder_board.csv`.
5. > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-inclusive-tensors-plan.md and execute **Session 5 (§5e): retire the
   > `_rookieinclusive` label** — point `rookie_recovery` at the shipped tensor, retire the
   > two labelled targets while keeping `--tensor-label`, delete the labelled tensors and
   > keep the labelled CSVs, and supersede the §7i prose that rests on the two arms
   > differing. Record the frozen counterfactuals' provenance note.
6. > I'm working on the NBA prediction project (CLAUDE.md). Read
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
