# Next session — after P5 closed

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md`'s two P5
sections. **Check `git status` rather than trusting any line here about what is committed** —
the last two handoffs both got that wrong in opposite directions.

## What happened on 2026-08-15

**P5 closed.** `make preseason-contest` took the one measurement the gate specified and P5's
own run could not: what the preseason block is worth in the contest, as a paired
counterfactual rather than a re-read. **The preseason round is now complete — every gate P0
through P5 is closed.**

### 1. The instrument — `src/sim/preseason_contest.py`, `make preseason-contest`

`src/sim/mixture_value.py` one round over: two arms of the same chain, captured under one
code, both written into `outputs/predictions/preseason_block_contest.csv` with their
`captured_at`, so a delta claim reads ONE row rather than differencing two artifacts of
unreconstructable vintage.

`base` flips the three keys that ARE the block and refits all three head groups (19/24/25
features against 29/29/25; windows 2012-13 / 1997-98 / 1996-97). It is **not** an ablation to
zero — each key is a documented exact rollback, so `base` is the chain that was shipping
before the round. 185.7 sampler minutes, R̂ 1.0073 / 1.0052 / 1.0044, **0 divergences**.

⚠️ **σ is frozen at 0.375 in BOTH arms.** It is not part of the block — it is a downstream
constant whose input moved at P5 — so freezing it is what makes the delta the block's. `base`
is "today's chain with the block removed", not "the chain as of 2026-08-13".

⚠️ **Only two of the three heads reach this readout.** `src/sim/` never loads the marginal
minutes head, so P3's block — the largest of the three by its own gate — is structurally
invisible to every contest figure. Its key is flipped and its posterior refitted anyway so
the arm name means what it says, and a test pins the import fact by parsing.

### 2. The attribution is near-total — the headline

| season-total MAE | P5's recorded "before" | `base` | `preseason` | the block |
|---|---|---|---|---|
| 2022-23 | 397.36 | **397.24747** | **363.23449** | **−34.01297** |
| 2023-24 | 398.45 | **397.95546** | **377.50963** | **−20.44583** |

The counterfactual lands within **0.11** and **0.49** dk_pts of P5's own pre-block figures, so
σ + the ADP field + the injection are worth about half a dk_pt between them and **essentially
all of P5's Gate A gain is the block**. "None of this is attributable to the preseason block"
is superseded in both `README.md` and the plan doc.

### 3. The board moves — the reversal against the mixture round

Mean |Δrank| **16.411458** over the 192 drafted picks against the mixture's 3.1979; rank
correlation 0.962988, top-100 overlap 88%, **96 of 192** picks moving a full round. §7l's
precedent was that a head change arriving as SHAPE is a board null; this one arrives as the
allocation MEAN and the board says so. Stars gain **+81.317731** of mean season total while
`mean_gp` *falls* — minutes and production, not availability.

### 4. The contest — coherence, not one row

Every simulated tournament is a null at a bar of **0.074835**. But `adp`, whose board is
identical across arms, moved **−0.006490** against the 24-strategy mean of **+0.034429** —
which **inverts** the mixture round, where `adp` captured essentially the whole shift and made
the null a world effect. Realized lift is positive in **10 of 10** cells, **+0.102767** at the
600k, weakest cell 0.003700. Gate C's `rho` fell in both seasons, independent corroboration.

⚠️ **An instrument correction rode with this.** `RESOLVED` was being stamped on realized rows
using a bar bootstrapped from 500 *simulated* worlds; the realized side has one world per
season. Those rows now read `paired/2sn` and get their own `realized` block reporting sign
agreement and season spread. **`mixture_value.py` has the same latent flaw** — it never
surfaced there only because its realized delta was small. Worth fixing next time that target
is touched.

## Do NOT re-decide these

1. Everything the previous handoffs list under this heading still holds.
2. **σ = 0.375**, and it is now frozen across counterfactual arms by design.
3. **The realized side is priced by pairing, not by the simulated bar.** A test pins it.
4. **The block shipped**, and P5's closing measurement supports it. The ship itself was 4d's
   prediction-unit decision and was never under review here.

## Verified green

```
make docs-audit        # 0 disagreements, 0 stale claims, 3,566 figures
make dashboard-audit   # 0 orphaned artifacts, 0 pending constants, 285 findings
pytest tests/          # 1,840 passed (1,828 + this session's 12)
```

`make docs-audit` caught a real error during this session: a realized-lift figure typed from a
rounded print rather than read from the artifact. That is the gate working.

## The work

### 1. Session 6b — the five surviving rate heads' arms
`ast`, `fga`, `stl`, `tov`, `reb`, plus `ftm|fta`. Untouched, still. **This is now the only
unrun session in the preseason round's map.**

### 2. The season-total bias — `potential-to-dos.md` item 12
Gate A's bias is **−15.44 / −66.64** against a −3.06 bar, pre-existing rather than introduced.
The counterfactual moved it *toward* zero (+11.29 / +4.06), so the block helps but does not
close it. Two caveats on the bar itself: 873 pooled rows against the check's 386/387, and a
**full-season** figure against the simulator's **91% tournament window**. ⚠️ The naive version
of item 12's measurement gives the **opposite sign** — read the entry before re-running it.

### 3. Split the block's two reaching heads
This round moves availability and the composition together. Which one buys the contest gain is
unmeasured, and it is another paired pass on the same instrument — `--groups availability`
alone is ~6 minutes, so the cheap half is cheap.

### 4. Small, un-scheduled
- ✅ **`make strategy-sweep` re-measured at 49.2 min** (2,954 s), confirming P5's 53 and
  settling that the docs' 4.8 min is stale rather than a regression. The other stages:
  simulate-season 2.4 min, bracket 6.4 min, draft-sim 19 s.
- `PYTHONUNBUFFERED=1` on long `make` targets — used throughout this session and it worked;
  worth making the default.
- The `make stan-*` **metric** artifacts still carry no provenance stamp.
- `dashboard-audit`'s `reviewed`-date drift, carried for several sessions.

## Lessons worth carrying forward

**A bar borrowed from the wrong instrument reads as rigour.** The realized `RESOLVED` flag was
inherited from a module where it happened to be harmless. It is the `make docs-audit` failure
mode one level in — inside the code that *computes the verdict*, where no doc guard reaches —
and the same shape as the two prose examples the previous handoff recorded.

**Two traces are needed to prove a refit landed.** The `reach` block first checked
`first_season` alone, which would have reported the availability head as never having
refitted: its block is columns at a fixed window, while the composition's is a blend that adds
no columns at a moving one. Each head is invisible in the other's trace.
