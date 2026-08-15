# Next session — after 6b closed and the round merged

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md`'s session 6b
section. **Check `git status` rather than trusting any line here about what is committed** —
three handoffs in a row got that wrong.

## Branch

**The preseason round is complete and `incorporate-preseason-data` is merged into
`tweaks-post-dashboard`** (fast-forward, 2026-08-15). ⚠️ **Nothing has been pushed.** The
merge is local; `origin/tweaks-post-dashboard` is still at the pre-round commit. Pushing is
the owner's call.

## What happened on 2026-08-15 (second session)

**Session 6b ran — the last unrun session in the round's map.** `make components-preseason`
(`src/models/components_preseason.py`) arms the six rate heads P1 short-listed and holds them
to the P2/P3 bar. **Every gate P0 through P5 plus 6b is now closed.**

### 1. The instrument — point MLE, under five minutes, no CmdStan

Each arm is the head's **shipped variant** (read from `stan_component_metrics.csv`, never
hard-coded — the six do not agree) plus preseason columns, fitted through
`component_rates.fit_count_head` / `fit_conversion_head`. Six heads × seven arms × two
populations, plus a 13-origin rolling harness. Same shape as `minutes_preseason`, one family
over. 18 new tests.

### 2. Three of six clear — and P1's top-ranked head is the null

| head | validation | rolling | gate |
|---|---|---|---|
| `fga` | −2.7502 [−3.9831, −1.5961] | −3.0663, **13/13** | ✅ |
| `ast` | −0.8503 [−1.3175, −0.3491] | −0.9492, 12/13 | ✅ |
| `reb` | −0.8213 [−1.3263, −0.3384] | −0.8138, 12/13 | ✅ |
| `tov` | −0.0480 [−0.2421, **+0.1619**] | −0.2860, 12/13 | ❌ validation |
| `stl` | −0.0850 [−0.1678, **+0.0026**] | −0.0671, 12/13 | ❌ validation, by a whisker |
| `ftm\|fta` | −0.0129 [−0.0639, +0.0391] | −0.0249, 9/13 | ❌ **both** |

**`ftm|fta` was P1's LARGEST rate increment** (+0.0176 R², z = 18.8) and is the round's only
two-sided failure — and no arm gets it over its own floor. The mechanism is that a conversion
delta is a logit of a percentage over ~10–40 preseason free throws (sd **2.2023** on the logit
scale against `ast`'s 0.3617). A ΔR² screen cannot see that; a distributional bar can. **P1's
own "what it does not settle" fired, on the head P1 ranked first.**

### 3. The headline: on two heads the block beats the whole fitted head

Read against the no-fit floor rather than zero: on `reb` fitting buys **0.1507** CRPS and the
block buys **0.8213** more (**5.45×**); on `fga` the ratio is **1.07×**. The README's "the rate
side is nearly saturated" line is now qualified in place — it is saturated against
*prior-season* information, and six preseason games are not that.

### 4. Two method findings that generalize

- **The volume term wants an empirical-Bayes shrink on rates**, not P1's additive
  `pre_log_min`. `own_delta_shrunk` beats the declared primary on the fitting half on all five
  count heads with intervals clear of zero, and read on that promoted arm the gate count goes
  **3 → 4** (`tov` flips). Selected `k` runs 20 to **320** pseudo-minutes. Mechanism: a minutes
  total over 60 preseason minutes is measured *on* those minutes; a per-36 rate *divides* by
  them.
- **Season-centring is a device for LEVELS, not for deltas in general.** It shipped on P2 and
  P3 and **loses here** with intervals clear of zero on `fga`, `reb` and `tov`. A per-36 rate
  has already divided the exposure out, so there is no season-level nuisance left to remove.

### 5. Three risk entries updated, all in the same direction

The rolling-shrinkage pattern inverted a **third** time (validation ÷ rolling: 0.90, 0.90,
1.27, 0.17, 1.01, 0.52 — not one head shows the §12e/§14f shape). The coverage cut costs
**≤0.133** CRPS here against a quarter of the increment on P3. And P1 decision 5's
pooled/draftable restriction is **nearly a no-op** on this family (ratios 0.87–1.29, draftable
*larger* on five of six), because the `≥ 200 prior minutes` filter already removes 91.4% of the
population it corrects for.

## Do NOT re-decide these

1. Everything the previous handoffs list under this heading still holds.
2. **σ = 0.375**, frozen across counterfactual arms by design.
3. **The realized side is priced by pairing, not by the simulated bar.** A test pins it.
4. **Nothing from 6b ships.** A cleared gate earns a Stan port and a port is a separate door —
   the same structure as P3 earning the composition a pricing session rather than an adoption.
5. **`blk`, `fta`, `fg2m|fg2a`, `fg3m|fg3a` get no arm** (P1 decision 2), and `ftm|fta` is now
   a recorded null too. The conversion family is a null on preseason data in all four heads.

## Verified green

```
make docs-audit        # 0 disagreements, 0 stale claims, 3,716 figures
make dashboard-audit   # 0 orphaned artifacts, 0 pending constants, 292 findings
pytest tests/          # 1,858 passed (1,840 + 18)
```

All 292 dashboard-audit findings are the `reviewed`-date drift carried for several sessions —
0 missing artifacts, 0 orphans.

## The work

### 1. 🔥 `potential-to-dos.md` item 13 — port the three cleared arms and price them
The direct continuation, and the largest unexploited increment measured in the project. Unlike
the marginal minutes head these six are **in the simulator's draw path**, so the whole pricing
chain exists: a `head_design` port behind a `stan.components.preseason` flag, scored against a
same-window control, then `make posteriors --groups components` and one `make
preseason-contest` pass. The entry names the bar, the falsifier, and the retention figure to
watch (three rounds found the increment *grew* under the posterior; a retention below 1.0
would be the first).

### 2. The open decision that now stands on five heads
**Whether a rolling-only win should ship.** P2 registered it and did not take it; `stl` and
`tov` now stand in exactly that position, `stl` missing validation by **+0.0026** with 12 of 13
origins. It is a decision about the bar, and three rounds have now failed it in the direction
the bar was not written for. What would settle it without widening anything is more scored
validation seasons — and those are the test split.

### 3. The season-total bias — `potential-to-dos.md` item 12
Unchanged from the last handoff. Gate A's bias is −15.44 / −66.64 against a −3.06 bar,
pre-existing rather than introduced. ⚠️ The naive version of the measurement gives the
**opposite sign** — read the entry before re-running it.

### 4. Split the block's two reaching heads
Also unchanged. Which of availability / composition buys the contest gain is unmeasured;
`--groups availability` alone is ~6 minutes.

### 5. Small, un-scheduled
- `PYTHONUNBUFFERED=1` on long `make` targets — used again this session and it worked; worth
  making the default.
- The `make stan-*` **metric** artifacts still carry no provenance stamp.
- `dashboard-audit`'s `reviewed`-date drift, carried for several sessions and now at 292.
- `mixture_value.py` has the same latent `RESOLVED`-bar flaw `preseason_contest.py` fixed;
  worth fixing next time that target is touched.

## Lessons worth carrying forward

**A screen ranks by signal; a bar ranks by signal net of the noise carrying it.** P1's ΔR²
put `ftm|fta` first across the whole rate family and `reb` last among the clearing heads. At a
distributional unit `ftm|fta` is the only two-sided failure and `reb` has the largest
block-to-fit ratio in the round. Both reversals have the same cause — a noisy regressor with
real signal in it still raises R² on a point estimate, and only a predictive has to pay for
the noise. This is the third instance of the general rule the repo already records as "a head
is only a model at the unit it was scored at".

**A device is not a finding until its scope is measured.** Season-centring won on two heads
and was recorded as "the arm to watch". It loses on the third family, and the reason was
predictable from the mechanism — the compression it corrects is a property of *levels*, and a
per-36 rate has already divided the exposure out. The arm was written to test exactly that and
came out the other way, which is the argument for carrying an attribution arm you expect to
lose.
