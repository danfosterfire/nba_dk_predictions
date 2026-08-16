# Next session — the documentation pass 6b owes

Read `CLAUDE.md` and `docs/project-spec.md` first, then `docs/preseason-plan.md`'s session 6b
section. **Check `git status` rather than trusting any line here about what is committed.**

## ⛔ The repo is RED, deliberately

```
make docs-audit    130 disagreements   ← FAILS
pytest tests/      1,865 pass, 1 FAIL  ← test_docs_audit, reporting the same 130
```

Nothing is broken. The code is committed and green on its own tests; **every quoted component
figure in the docs is stale**, because the covered-window cut moved the fitting rows for ten
of the eleven rate heads and the whole chain re-ran behind them. Clearing that is this
session's job and essentially all of it.

## Branch

`tweaks-post-dashboard`, **20 ahead of origin, nothing pushed.** Two commits from 2026-08-15:
`a0f3b08` (the 6b ladder at six heads) and `ace35d5` (the widening, the port, the chain).

## What happened on 2026-08-15, part two

### 1. The ladder widened to all eleven heads, and the exclusions were wrong

P1's ΔR² screen admitted six heads. All three it called **actively harmful** failed to
reproduce as harm at a paired interval:

| head | P1 screen | validation | rolling | gate |
|---|---|---|---|---|
| `fta` | −0.00161, z −2.32 | **−0.6300 [−0.8518, −0.2181]** | −0.8141, 13/13 | ✅ |
| `fg2m\|fg2a` | −0.00845, **z −9.44** | **−0.1541 [−0.2238, −0.0186]** | −0.1581, 12/13 | ✅ |
| `blk` | −0.00847, z −3.18 | −0.0857 [−0.2074, +0.0573] | −0.1649, 12/13 | rolling only |

And two of the five excluded heads were never negative: `fg3m|fg3a` read **+0.01488 at
z = 7.53** (excluded on attribution — the gain was the shared indicator) and `fg3a|fga`
**+0.00467 delta-carried**, better than two heads that *were* armed. `fg3a|fga` turns out to
be among the strongest results in the round: **−1.4373** validation, **−1.9544** rolling,
13 of 13 origins.

On the shipped `own_delta_shrunk` arm **every one of the eleven clears the rolling half**;
seven also clear validation. **No head anywhere in the round has an interval clear of zero on
the wrong side.**

### 2. Ten of eleven ship; `fg3m|fg3a` is rolled back

Retention under the posterior: **10 of 11 hold, median 0.991**. `fg3m|fg3a` is the sole
reversal — **+0.02914** CRPS, and worse on CRPS, NLL *and* PIT KS at once. Three instruments
agree (P1's attribution, 6b's pooled point MLE at +0.00688, the posterior control), so
`stan_components.PRESEASON_EXCLUDE` opts it out and it now reproduces the pre-block head
exactly: fitted NLL **3.1676** against the 3.1677 the docs have carried since before 6b.

### 3. The chain, run end to end

Gate A improves on both seasons. Because this pass reuses P5's **byte-identical base
capture**, the components share is separable by differencing the two paired passes:

| | base | shipped | 4-key delta | P5's 3-key | **components adds** |
|---|---|---|---|---|---|
| MAE 2022-23 | 397.2475 | 360.9636 | −36.2839 | −34.0130 | **−2.2709** |
| MAE 2023-24 | 397.9555 | 373.6643 | −24.2912 | −20.4458 | **−3.8454** |

Bias moves **+15.68 / +16.55** toward zero. Contest: simulated **null** in all five
tournaments at a 0.0748 bar, realized **positive in all five** (+0.0960 at the 600k), and the
`adp` control moved **−0.0052** — the wrong way for a world effect.

### 4. Four wiring gaps, three of which would have shipped a head that was not the head

Recorded because the *pattern* matters more than any one of them — a head can be selected
under one specification and persisted under another, and every artifact stays internally
consistent:

- **`posteriors.component_artifacts`** built rows through `component_rates.build_design` and
  would have persisted eleven heads with **no preseason columns** while the config and metrics
  artifact both said the block was on.
- **`src/sim/season.py`** did the same in two places. Caught at *run time* by
  `PosteriorRecipe._block`, 40 s into a 60-minute chain — the guard names the builder the
  frame should have come from, which is why the error identified its own fix.
- **`manifest_row`** has a fixed column list and never picked up the new `extras`, so the
  trace reached the pickles and not the CSV.
- **`covered_fitting_rows`** cut the window family-wide, so the rolled-back head was fitted on
  **6,382** rows instead of 8,630 — the pre-block columns on the post-block window — while the
  run printed that it fitted the pre-block head "exactly".

All four fixed, all pinned by tests, including an AST test that stops `season.py` importing
the plain builder under any alias.

## 🔥 The work: the documentation pass

### 1. The 130 docs-audit disagreements

```
docs/model-development-notes.md   39      README.md                  13
docs/preseason-plan.md            38      docs/predictions-plan.md   11
docs/simulations-plan.md          22      docs/shot-attempt-basis-plan.md  6
docs/train-validate-test-split.md  1
```

Almost all are one cause: **the component heads' fitting window moved from 8,630 rows
(1997-98) to 6,382 (2004-05)**, so every quoted R², NLL, CRPS and floor moved with it. Follow
the house convention — the superseded figure stays beside its correction, and
`historical=True` in `src/docs_audit.py` is how it is registered.

⚠️ **Do not "fix" these by re-running anything.** The artifacts are correct; the prose is
stale. `docs/no-refits-for-record-keeping` is the standing rule.

### 2. `docs/preseason-plan.md` session 6b — rewrite, do not patch

The section was written for the **six-head** run and every headline in it is now wrong twice
over: "3 of 6 clear" is now 6 of 11 at the gate and 10 of 11 shipped, and the figures moved
again when the window cut went per-head. It needs:

- the eleven-head gate table and the screen-reversal finding (the strongest result in the
  round: a ΔR² screen's *sign* did not survive on any of the three heads it called harmful)
- the retention table and `fg3m|fg3a`'s rollback as its own subsection
- the session map row and the gates section
- the four risk entries, which currently cite six-head numbers

### 3. `dashboard/decisions.py`

Four entries exist from the six-head run and all need updating; `fg3m|fg3a`'s exclusion needs
a new one — it is the only measured-worse result in the entire preseason round and belongs on
the decision log for that reason alone.

### 4. `README.md`, `docs/pipeline.md`

README §2 and §3 carry six-head prose and the 5.45× / 1.07× ratios, which moved. `pipeline.md`
carries a `components-preseason` block written for six heads.

### 5. Register the new figures

`_preseason_components()` in `src/docs_audit.py` claims the six-head run. It needs the eleven
heads, the retention table, the control arms, and the contest delta.

## Do NOT re-decide these

1. Everything the previous handoffs list under this heading still holds.
2. **σ = 0.375**, frozen across counterfactual arms by design.
3. **Ten of eleven component heads ship the block**, and `fg3m|fg3a` does not. Owner decision
   2026-08-15 on the posterior reading.
4. **`stl`, `blk`, `ftm|fta`, `fg3m|fg3a`'s siblings ship against the validation half** — they
   pass rolling and cannot be resolved on 706 validation rows. Same owner decision P2 recorded.
5. **`tov`'s variant flip to `log_own_spline`** is accepted; the CRPS/R² disagreement is small
   and was explicitly waived.

## Loose ends, all small

- **The `reach` block has no `components` row.** `preseason_contest.reach_rows` reports
  `refit_landed` for availability, minutes and composition only, so the family this round
  shipped has no trace in the artifact that prices it. All four config keys do show 0→1.
- **Gate C's `rho` is fitted on the rows the backtest scores.** It is solved per season *and
  per arm* by bisection against that season's realized model-vs-market skill gap — not a
  static constant. `README.md` names four simulator inputs calibrated this way and `rho` is
  not among them; it probably should be. It also means the two arms' simulated worlds are not
  the same world, which is why `sim_lift` is not arm-comparable (P5 already says this).
- **`potential-to-dos.md` item 13** — port and price — is now **done**, and should be marked
  as such rather than left open.
- **Item 14** is new and unstarted: `make stan` discards draws that `make posteriors` then
  refits, worth **4.9 h per full train-window run**.
- `PYTHONUNBUFFERED=1` on long `make` targets, still worth making the default.

## Lessons worth carrying forward

**A screen's sign is not evidence about the heads it failed.** P1's permutation z said `blk`,
`fta` and `fg2m|fg2a` did *worse than a shuffled block* — which reads as evidence of harm
rather than absence of gain. None of the three reproduced, and two clear the real bar
outright. The same screen had already misranked the heads it *passed*, in both directions.
Excluding on a screen that has been contradicted is the same error as trusting it.

**Wire every consumer, not the ones you happen to be reading.** Four call sites needed the
same change; two were found by reading, one by a run-time guard, one by checking output that
should have contained a column and did not. The guard that caught the third existed because
someone wrote `builder` into the error message — the cheapest thing in this session and the
only reason that failure took 40 seconds to diagnose instead of an afternoon.

**A partial-run path is worth building the first time you want one.** `--heads` on
`stan_components` and `--rebuild-manifest` on `posteriors` each took minutes and each replaced
a multi-hour re-run. `posteriors --groups` was the precedent and had been there all along.
