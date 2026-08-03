# Shot-Attempt Basis Plan: `fga` × `fg3a | fga` instead of two independent counts

A planning doc in the house pattern — the measurement is done and recorded here in full;
the adoption is specified and **not yet taken**. It closes the re-measurement that
`README.md`'s *Measured but not adopted* section names as the blocking step for the first
of its two rows.

## The problem

`component_rates.COUNT_HEADS` fits `fg2a` and `fg3a` as two **independent** negative
binomial counts. They are not independent: a three-point attempt *substitutes* for a two.
That coupling is the single largest structural off-diagonal the residual copula would
otherwise have to carry — `make residual-correlation` measures it at **−0.1248**
minutes-conditioned, against an off-diagonal mean of **+0.0071** across the eleven heads.

Coupling two count heads would break the exact posterior factorization the whole
architecture rests on (`CLAUDE.md`, "fit the Stan heads SEPARATELY"). Reparameterizing does
not. Model the **total** attempts as a count and the three-point **share** as a binomial on
those trials:

```
(fg2a, fg3a)   ->   (fga = fg2a + fg3a,  fg3a | fga)
```

The substitution is then enforced by construction — one more three is one fewer two,
identically — while the chain `counts | min -> makes | attempts` stays intact and every
head keeps its own parameter block.

**The comparison is legitimate because the map is a bijection with unit Jacobian on the
integers.** `(fg2a, fg3a)` and `(fga, fg3a)` are the same point in different coordinates,
so the two joint log-densities are directly comparable and their difference is a real
quantity rather than a scale mismatch. This is the property the composition head's joint
NLL explicitly does *not* have, and the two must not be quoted as the same kind of evidence.

There is a second, independent argument that does not depend on any of the above: shot-mix
**shares persist like counts** (`sco_pct_fga_3pt` at 0.886 in `persistence.csv`) while
*conversion* percentages do not (`fg3_pct` 0.500). The share is a first-class feature; the
percentage would not be.

## Why this had to be re-measured

`stan_components.substitution_arm` reported **−0.771 nats on validation and −0.793 on test**
and those figures were quoted in three places as settled. They were measured against a
**straw man**: the arm fits *every* head at the `log_own` variant, but `fg3a`'s shipped spec
is `log_own_spline`, and at `log_own` that head reads held-out R² **0.3719** with
`beats_floor = False` against **0.9046** for the spline it actually selects. Arm A was
handicapped by roughly the difference between a working model and a broken one on one of its
two terms.

Two smaller defects came with it. Arm B was pinned at one variant and never swept, so the
comparison was "arm A's best-but-one against arm B's first guess". And the share head's
own-rate column was assembled by hand rather than through `conversion_variants`, because
that function derives the column by name convention — `f"{made}_pct_lag1"`, which for this
head resolves to `fg3a_pct_lag1`, three-point **shooting percentage**, not the attempt-mix
share. The column does not exist, so the call would raise; the hazard is that someone adds
it, at which point the head silently fits on shooting accuracy and its entire rationale is
gone. `conversion_variants` now takes an explicit `own=` parameter and a test pins both arms
of the behaviour.

## Gate 0 — the measurement

`make stan-substitution` (`stan_components.substitution_sweep` →
`outputs/predictions/stan_component_substitution_sweep.csv`, 52 rows). Its own target and
its own artifact, deliberately: `substitution_arm` is called from inside
`stan_components.run()`, which writes all three component CSVs together, so refreshing this
comparison through `make stan-components` would cost that target's 209 minutes *and* rewrite
`stan_component_metrics.csv` — the artifact this sweep reads arm A's selected specs and
best-of-16 grid from. Sixteen fits against seventy-four.

10,194 player-seasons; 9,403 train / **791** test (2024-25, 2025-26), validation split
8,630 fit / **773** select (2022-23, 2023-24). Selection reads validation only.

- **Arm A** at each head's own validation-selected variant, read via
  `season_terms.selected_specs()` from the artifact that chose it rather than re-derived —
  `fg2a @ log_own`, `fg3a @ log_own_spline`.
- **Arm B** swept for real: `fga` over (`linear`, `log_own`, `log_own_spline`) and
  `fg3a | fga` over (`linear`, `logit_own`, `logit_own_spline`). **Additive separability
  makes this 3 + 3 fits, not 9 combinations**: the joint density factors as
  `p(fga) · p(fg3a | fga)`, the two factors share no parameters, so minimising the sum is
  exactly minimising each term. `_g0_select` picks per factor and a test pins that the joint
  row it marks is the pair of per-factor winners.

**The regression check passed to full precision, which is what licenses everything below.**
Refitting arm A's test side reproduces `stan_component_metrics.csv` exactly — `fg2a@log_own`
at **5.229492** and `fg3a@log_own_spline` at **5.248561** — and the refactored share arm at
`(log_own, logit_own)` reproduces the recorded **9.991042**, to nine decimal places. Nothing
upstream drifted between July and this measurement, so the corrected margin is a correction
and not a different experiment.

### Per-factor NLL, against each factor's own no-fit floor

Held-out mean negative log-likelihood per player-season; lower is better. Conversion floors
are **shrunk** carry-forwards with `k` fitted on train only, per `CLAUDE.md`.

| arm | factor | variant | val NLL | test NLL | val floor | test floor | selected |
|---|---|---|---|---|---|---|---|
| A | `fg2a` | `log_own` | 5.262971 | 5.229492 | 5.271028 | 5.292842 | ✓ |
| A | `fg3a` | `log_own_spline` | 5.241963 | 5.248561 | 5.903396 | 5.731185 | ✓ |
| B | `fga` | `linear` | 5.407438 | 5.426520 | 5.445210 | 5.432802 | |
| B | `fga` | `log_own` | 5.390057 | 5.379640 | 5.445210 | 5.432802 | |
| B | **`fga`** | **`log_own_spline`** | **5.388533** | **5.376766** | 5.445210 | 5.432802 | **✓** |
| B | `fg3a\|fga` | `linear` | 4.912507 | 4.929134 | 4.619109 | 4.652797 | *fails floor* |
| B | `fg3a\|fga` | `logit_own` | 4.636033 | 4.611402 | 4.619109 | 4.652797 | *fails floor on val* |
| B | **`fg3a\|fga`** | **`logit_own_spline`** | **4.615620** | **4.607737** | 4.619109 | 4.652797 | **✓** |

### The verdict

| split | arm A (selected) | arm B (selected) | margin |
|---|---|---|---|
| validation | 10.504935 | **10.004153** | **−0.500782** |
| test | 10.478052 | **9.984503** | **−0.493549** |

**Gate 0 passes.** Arm B wins on both splits, by more than the un-handicapping alone
predicted: 0.305646 of the recorded 0.792657-nat test margin was the handicap, leaving
−0.487010, and sweeping arm B recovers a further −0.006539 for **−0.493549**.

**Against arm A's most favourable configuration it still wins by −0.491910.** The best of
all sixteen (`fg2a` variant × `fg3a` variant) combinations in `stan_component_metrics.csv`
is `log_own_spline + log_own_spline` at **10.476413** — published from the artifact at zero
extra fits. So the result does not depend on which variant arm A is granted.

### ⭐ The finding worth remembering: the coordinate change beats the fitting

Comparing the two bases **at their no-fit floors** — prior per-36 rate × minutes for the
count, shrunk carry-forward for the share, no features anywhere:

| | arm A | arm B | difference |
|---|---|---|---|
| no-fit floor, test | 11.024027 | **10.085599** | **−0.938427** |
| best fitted, test | 10.476413 | 9.984503 | −0.491910 |

**Arm B's no-fit floor beats arm A's best fitted configuration by −0.390814 nats.** Writing
the identity in the right basis is worth more than everything the canonical basis's fitting
buys, and it is ~79% of the total margin: arm B's own fitted heads add only **−0.101096**
on top of its floor.

That reframes the result. This is not "a better model of shot attempts" — it is the same
information written in coordinates where the dependence is structural instead of residual.
It is the same shape as the repo's standing finding that the component rate side is nearly
saturated by a carry-forward: when the floor is that strong, the parameterization is where
the remaining leverage is.

### Two secondary results

- **The share head needs its spline to clear its floor, and `logit_own` alone does not.**
  On validation `logit_own` reads 4.636033 against a floor of 4.619109 — *below* the floor —
  and only `logit_own_spline` clears it, at 4.615620. On test `logit_own` does clear
  (4.611402 vs 4.652797), so **a test-only reading would have shipped a variant that fails
  its floor on the split that selects.** Same shape as the NB-vs-Poisson result on the count
  heads and the `binomial` arm on the composition: the flexible term is not a refinement,
  it is the difference between a model and a failure. `linear` fails on both splits.
- **For `fga` the spline is nearly free and nearly pointless** — 5.388533 against
  `log_own`'s 5.390057 on validation, a margin of 0.0015. Unlike `fg3a`, where the spline is
  decisive, `fga` is a well-behaved total and the log scale is essentially the whole answer.
  It is selected because validation selects it, not because it matters.

### Sampler

16 fits, **max R̂ 1.0087**, **0 divergences**, 46.5 minutes total. A handful of
treedepth-saturated iterations on the spline arms (≤0.6% of draws), which is the recorded
B-spline conditioning cost and not a convergence problem.

## What adoption requires — specified, not done

**Stop here for review.** Everything below is the design for the change; no fitting code has
been repointed.

### The head count stays eleven

Counts become `fga, fta, reb, ast, stl, blk, tov` (**7**) and conversions become
`fg3a|fga, fg2m|fg2a, fg3m|fg3a, ftm|fta` (**4**). `fg2a` stops being a head and becomes
**derived** — `fg2a = fga − fg3a` — exactly as `pts` already is. Nothing about the output
contract changes: the same twelve quantities are still produced per player-game and `dk_pts`
is still reassembled from the same eight.

This matters because `tests/test_residual_correlation.py` asserts `len(COMPONENTS) == 11`,
and that assertion should keep passing rather than being edited.

### The generic machinery mostly absorbs it

With `fga` in `COUNT_HEADS` and `("fg3a", "fga")` in `CONVERSION_HEADS`,
`component_rates.build_design`'s five lag-column loops produce `fga_p36_lag1`, `fga_lag1`,
`fg3a_pct_lag1` and `fg3a_lag1` for free, and `count_variants` / `conversion_variants` need
no changes beyond the `own=` parameter already added.

**One genuine gap.** `season_totals` builds its aggregation as
`{c: "sum" for c in COUNT_HEADS + made}` — so with `fg2a` out of `COUNT_HEADS` it is no
longer summed, yet it is still needed as the **trials** for `fg2m|fg2a`, and
`out[f"{m}_pct"] = out[m] / out[a]` divides by it. `fg2a` has to be added explicitly, as a
derived column rather than a head.

Note also that `fg3a_pct_lag1` — the column the naming convention would generate — becomes
*real* under the new head list, since `("fg3a", "fga")` is a conversion head. That is
precisely the condition under which the `own=` bug becomes silent instead of loud, which is
why it was fixed before this measurement rather than after.

### Three sites that would fail silently

All three degrade rather than raise, which is the failure mode this repo has been bitten by
before:

1. **`season_terms.py:822-823`** — the eleven-head completeness gate does
   `if any((head, arm) not in models ...): continue`. A missing head drops the whole arm
   from the season-total table with no message, so a half-migrated head list reads as "that
   arm was not run".
2. **`selected_specs` + `LEAGUE_SERIES.get(...) → None → continue`** — a stale
   `stan_component_metrics.csv` (one written before the migration) makes `specs.get("fga")`
   miss and silently fall back to `DEFAULT_COUNT_SPEC = "log_own"`. Given that `fg3a` at
   `log_own` is the exact failure this doc exists to correct, a silent fallback to it is the
   worst available default.
3. **`residual_correlation.to_matrix`** filters with `[c for c in COMPONENTS if c in
   wide.index]`, so an unknown component name is dropped from the matrix rather than
   reported, and `substitution_r` then degrades to NaN.

Each should raise on an unrecognised head before the migration lands.

### `season_terms._draw_components` is the load-bearing change

`_draw_components` (`:738-753`) draws every `COUNT_HEADS` entry independently and then draws
makes on **the drawn attempts** — `rng.binomial(np.rint(counts[attempted]), ...)` — because
the chain is `makes | attempts` and conditioning on realized attempts would leak the target.
`test_compose_season_dk_puts_makes_on_the_DRAWN_attempts` pins that.

Under the new basis `counts["fg2a"]` does not exist, so the draw order must become:

```
fga  ->  fg3a | fga  ->  fg2a = fga − fg3a  ->  fg2m | fg2a,  fg3m | fg3a,  ftm | fta
```

**Nothing in the current code can express that ordering** — a conversion head's output
becoming another head's exposure is a new kind of edge in the draw graph. This is the one
place adoption is a design change rather than a list edit, and it is where the review should
concentrate.

### Open question for the reviewer — flagged, not decided

`residual_correlation.SUBSTITUTION_PAIR = ("fg3a", "fg2a")`. Under the new basis that
coupling is removed *by construction*, so the cell no longer exists among the modelled
eleven — the matrix would be over `fga` and `fg3a|fga` instead, and the −0.1248 that
motivated the whole reparameterization would simply be absent.

**Recommendation, not a decision:** move the matrix to the new eleven and keep the old pair
as an explicit "removed by construction" contrast row, so the artifact still shows what the
change bought rather than showing nothing where the finding used to be. The alternative —
dropping the pair silently — would make the copula's own artifact stop recording the reason
the copula got smaller. This is a judgement about what the artifact is *for*, so it belongs
to the reviewer.

## What this does NOT deliver

- **It is not adopted.** `COUNT_HEADS` still lists `fg2a` and `fg3a`, `sweep_counts` still
  iterates that list, and `substitution_arm` still ships beside it as the original ablation.
  Read the head list, not this doc, when asking what is fitted today.
- **It says nothing about `fg2m | fg2a` or the other conversion heads**, which are unchanged
  in both bases and cancel out of the comparison entirely.
- **It does not remove the copula.** Ten other off-diagonals remain; this removes the
  largest one and the only structurally-forced one.
- **No season term**, same as every other head — see the season-term verdict in
  `docs/predictions-plan.md`.

## Risks

- **The margin is per player-season on 791 test rows**, and no paired bootstrap has been run
  on it. The repo has one recorded false positive that survived a paired interval
  (the minutes-head nonlinearity arm, 95% CI [−0.079, −0.015] and it did not replicate), so
  the defence here is that the result replicates across splits and survives arm A's
  best-of-16 — not the interval.
- **Both arm-B factors selected spline variants**, and spline bases are the repo's known
  HMC cost centre. An orthogonalized (QR-whitened) basis is the standing fix if the shipped
  spec ends up carrying two of them.
- **`fga` is derived, not fetched.** `add_substitution_columns` builds it from `fg2a + fg3a`
  and their lags, so a change to either upstream column silently changes the head's target.
  `fga − (fg2a + fg3a) == 0` on all 731,906 rows today and a test pins the additivity.
