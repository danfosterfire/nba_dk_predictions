### ✅ The rule is now a capability, not a convention — `src/models/held_out.py`

**A rule that lives only in prose gets followed until it is inconvenient, and this one
already failed.** The games-played head's Gate D was *specified* with the incumbent's test
figures as its bars, run on the test split, and settled which model ships — on a 0.0013
CRPS margin a paired bootstrap could not distinguish from zero, which **reversed** when
re-decided on validation. Nothing in the code objected, because nothing in the code knew.

So reaching the test split now **raises** unless something has explicitly unlocked it:

- **`availability.split_seasons` is the choke point** every head goes through, and the guard
  lives *there* rather than at each call site — so a new head cannot forget to add a guard it
  never had to add. It returns `(kept, _HeldOut(held))`, and `_HeldOut` raises on
  `__getitem__` / `to_numpy` / `merge` but not on `len()` or `.columns`. The guard is on
  **use**, not on carving: `train, _ = split_seasons(...)` and `len(test)` stay legal because
  a module has to be able to separate the rows to know what to exclude.
- **`held_out.selection_split(design)` → `(train, validation)`** is what a sweep calls. The
  test seasons are never materialized, so there is nothing to read by accident.
- **`held_out.final_split(design)` → `(full_train, test)`** is guarded, and
  `src/final_evaluation.py` (`make final-evaluation`) is the only thing that unlocks it *to
  score*. It reads which arm shipped **from the artifact** rather than re-deciding, refits
  on train **plus** validation, and scores test once. Registered heads: `availability`,
  `games_played`, `season_total`, and — since 2026-08-21 — `chain`, which is not a head but
  the workflow: a board built from the deployed posterior, drafted under the shipped
  strategy and scored on the box scores that happened.
- `tests/conftest.py` unlocks for the suite (unit tests exercise the split machinery on
  synthetic four-season frames); `tests/test_held_out.py` re-locks and tests the lock, and
  **AST-walks every converted module** to assert it names `selection_split` and never
  `split_seasons`.

### ✅ The split was spent on 2026-08-21, and a SECOND unlocker exists

**`make final-evaluation` has now been run for the three model heads.** (The `chain`
reading is a separate, hours-long run and is not in yet — `final_evaluation.csv` merges by
head, so taking it later does not retract these.) What it said is
`docs/final-evaluation-plan.md`; the one-line version is that the project's headline
replicated — the availability head is worth **210.2978** dk_pts of season-total MAE on
validation and **211.1288** on the held-out seasons.

⚠️ **With one caveat that belongs in this doc rather than that one.** The season-total half
of that statement re-derives figures the project already had: the whole table was a test
evaluation until 2026-08-05 and its retired readings are still presence-checked as
historical claims. They reproduce exactly, because that table composes the *plain*
beta-binomial and nothing in it changed. The availability head's own held-out figures had
never been computed under any discipline, and those are the new ones.

The same day added the other legitimate reason to read those seasons, and it is a different
act: `posteriors.assert_production` unlocks the split for the **production fit**, which
refits the shipped specification on every season there is so the upcoming season's board is
not throwing two years of data away. `docs/preseason-plan.md`'s October runbook has always
specified that fit; until 2026-08-21 the command it named could not run, because nothing
unlocked what it reached.

**A second unlocker is safe here for one reason, and it is pinned by a test rather than
argued**: the production fit produces coefficients and takes no measurement, so nothing it
writes is a number a decision could read. `tests/test_held_out.py` asserts that
`posteriors.py` names no scoring function — the day it grows one, the production window
becomes an unmeasured way to read the test split, which is the shape of the Gate D failure
that produced this module.

It is gated on **ordering** as well as on intent: `assert_production` refuses to run until
`outputs/predictions/final_evaluation.csv` exists. Deploy before measuring and there is no
honest measurement left to take — from the moment a production fit exists, every candidate
model has seen the test seasons. The gate fires once and never again.

### ⚠️ The deployed model was not the selected model, and nothing could see it

`require_window` stops a consumer reading coefficients fitted on a wider split. It says
nothing about *which model* those coefficients belong to, and on 2026-08-21 that gap was
live: `data/features/posteriors/train_val/` had been on disk since 2026-08-08, carrying the
right window and a **different specification** — it predated the preseason block on ten of
eleven rate heads and the composition's adopted offset. A held-out reading taken off it
would have described a model nobody ships.

`posteriors.assert_same_specification` is the missing half. It compares two windows on the
eight columns that define a specification rather than a fit — `family`, `variant`,
`fit_first_season`, `n_features`, `preseason`, `preseason_columns`, `player_season_effect`,
`sigma_u` — and refuses a wider window that is missing a head, because a partial `--groups`
run is the normal way to produce one. The chain readout calls it before it simulates
anything.

**Why the final evaluation refits rather than predicting test from the training fit.** A test
figure should describe the model that would actually deploy, which has seen train *and*
validation. Predicting test from a train-only fit is a *lower bound* on deployed performance,
not an estimate of it. The cost is that a val-versus-test comparison is confounded — different
rows, different training data — which is exactly why the sweeps no longer emit a test column
at all. **A val/test gap was never a replication check here**, and the project stopped
pretending it was on 2026-08-05.

**Converted heads, and what it cost.** `stan_availability`, `stan_minutes`,
`stan_components`, `stan_composition`, `stan_games_played`, `season_terms`, `season_total`,
`component_rates`, and `eda.season_effects`'s `carry_forward_bias`. Roughly **half the
sampler time** across the Stan heads, since the test side was the majority of the fits at
double the iterations — and selection was raised to full-length chains everywhere, because
the short/long split was a second confound sitting inside a comparison being read as a
replication.

- **`component_rates` was the one head with no guard at all**, because it defined its own
  `split_seasons` rather than importing the shared one. Six identical lines routing around
  the lock. A test now pins that it cannot come back.
- **Five findings reversed on the move, and all five had passed on test.** Gate E of the
  games-played plan (✅ by 0.03 dk_pts on test → ❌ by 6.34 on validation); `fg3a|fga` under
  sklearn (clears its floor on test, fails on validation); the games-played Gate D that
  started this; **`fta`**, added 2026-08-06 when `stan_components` was re-run, which
  failed its no-fit floor by 0.0024 on test and clears it by 0.0144 on validation
  (**+0.0198** since the preseason block, so the reversal widened rather than narrowed),
  retiring "the whole free-throw family fails"; and **the availability model ladder**,
  added 2026-08-08, where the GBM went from third to first. **Three survived unchanged**,
  which is the useful contrast: the availability
  Stan port still reproduces the MLE on every one of its terms (21/21 then, 24/24 after the
  2026-08-11 window and role-graded dispersion, 35/35 since the 2026-08-12 mixture), the
  `reb` alpha-trap pair
  reproduces 0.928 / 0.662 to a thousandth across the split change, and the substitution
  arm's validation margin reproduces at −0.771 across a half-length-to-full-length change
  in every one of its fits. ⚠️ **That last figure is a 2026-08-06 reading and is not the
  current one**: `stan_component_substitution.csv` is written by `make stan-components`,
  which re-ran on 2026-08-15 with the preseason block on ten of eleven heads, so the margin
  now reads **−0.7218**. What reproduced across the chain-length change still reproduced;
  the heads underneath it changed afterwards, which is a different event and does not
  retract the reproduction.
  - **Every one of the five reversals was a sub-1.5% margin on test that a paired bootstrap
    could not have called.** That is the pattern the conversion established and then
    confirmed on its last head: the split move does not overturn findings with real margins,
    it overturns the ones that were never distinguishable from zero and were being reported
    as verdicts. The availability ladder is the case where the interval was actually
    computed rather than inferred after the fact — GBM-minus-GLM at −0.1297 with a 95% CI of
    [−0.3154, +0.0672].
- **✅ `stan_composition` is converted AND its artifact regenerated (2026-08-08)** — the
  last code/artifact disagreement in the repo is closed, and **nothing reversed**: same
  selected arm, same ordering, same gate outcomes, with no arm moving more than 0.0019 CRPS
  across a doubling of chain length.
  - **The deferral argument was wrong, and the way it was wrong is the lesson.** It rested
    on "the run buys no decision" — true, and beside the point. Held-out figures sitting in
    an artifact are quoted from the docs and rendered on the dashboard as the head's
    performance, and they become the bar a successor arm is measured against. That is
    precisely how the games-played Gate D acquired test-set bars. **A stale artifact is not
    inert.**
  - **⚠️ It also rested on a cost estimate that was itself a `CLAUDE.md`-style false
    correction.** On 2026-08-06 the module's "the conversion roughly halves it" was
    "corrected" to "saves ~40%, 12.6–14.7 h", by assuming the validation fits would take 2×
    as long once selection moved to full-length chains. Measured: they took **1.33×**
    (7.35 h → 9.78 h), and `betabinom/val` ran *faster* at double the iterations (9,358 s
    against 9,615 s), because longer warmup adapts a better step size and buys fewer
    leapfrog steps. Total **9.92 h** against 21.13 h — a **53%** saving, so the original
    "halves it" was right and the correction was the error. **Sampler cost is not linear in
    the knob you are turning**, and an estimate from an untested proportionality is worth
    less than the run it replaces.
- **✅ `make availability-model` is converted and re-run (2026-08-08) — the last head, and
  the one where it mattered most. The CONVERSION IS COMPLETE; nothing in `src/` scores the
  held-out split outside `src/final_evaluation.py`.** This module does not merely report on
  the test seasons, it *decides* on them: the four-way ladder picks a mean function,
  `workload_ablation` picks a feature block, `nonlinearity_ablation` picks a basis.
  - **One of the three decisions had its ordering reverse and none of the three changed.**
    The head-vs-GBM-vs-ridge CRPS ladder went from **10.795 / 10.888 / 10.896 / 13.614** on
    test to **9.876 (GBM) / 10.004 (ridge) / 10.006 (GLM) / 13.387** on validation — the GLM
    from first to third. It still ships, because a paired bootstrap over the 883 rows puts
    the GBM's margin at −0.1297 with a 95% CI of **[−0.3154, +0.0672]**, and because both
    challengers are *worse* than the GLM on the lowest realized-games quartile, which is the
    population the head exists for. See the ladder bullet under "Established facts".
  - **The playoff-workload block and the nonlinearity null both survive untouched** — same
    signs, same orderings, and every nonlinearity figure reproduced to five decimals because
    that ablation was already selecting on the frames `selection_split` returns.
  - **This makes it five reversals, and the pattern holds for all five**: every one was a
    margin a paired interval could not distinguish from zero, being reported as a verdict.
    The availability ladder is the cleanest instance, because this time the interval was
    *computed* rather than reconstructed afterwards — `availability.ladder_comparison` ships
    it as an artifact.
  - **`availability.split_seasons` still exists and is still where the guard lives**, so
    `tests/test_held_out.py` checks this module per *function* rather than module-wide: `run`
    must name `selection_split`, and no sweep may name `split_seasons`.
