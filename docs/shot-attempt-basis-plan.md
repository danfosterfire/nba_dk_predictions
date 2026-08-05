# Shot-Attempt Basis Plan: `fga` × `fg3a | fga` instead of two independent counts

A planning doc in the house pattern — direction first, results filled in as they land.
**Gate 0 measured it (2026-08-03) and it was adopted the same day**, so this doc carries
both the measurement that justified the change and the record of what changing it actually
did. `component_rates.COUNT_HEADS` is now `fga, fta, reb, ast, stl, blk, tov` and
`CONVERSION_HEADS` is `(fg3a, fga), (fg2m, fg2a), (fg3m, fg3a), (ftm, fta)`.

## The problem

`component_rates.COUNT_HEADS` **used to fit** `fg2a` and `fg3a` as two **independent**
negative binomial counts. They are not independent: a three-point attempt *substitutes*
for a two.
That coupling is the single largest structural off-diagonal the residual copula would
otherwise have to carry — `make residual-correlation` measures it at **−0.1248**
minutes-conditioned, against an off-diagonal mean of **+0.0070** across the eleven heads.

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
head resolves to `fg3a_pct_lag1`. In the two-count basis that column does not exist, so the
call raises. `conversion_variants` now takes an explicit `own=` parameter and a test pins
both arms of the behaviour.

> ⚠️ **This doc originally claimed `fg3a_pct_lag1` would be "three-point shooting
> percentage" and that adopting the basis would make the head silently fit on accuracy
> instead of mix. Adoption falsified that, and the correction is worth keeping.** Under the
> new head lists `season_totals` builds each conversion head's own rate as
> `out[f"{m}_pct"] = out[m] / out[a]`, so with `("fg3a", "fga")` a head the generated column
> is `fg3a_pct = fg3a / fga` — **the attempt mix, which is exactly right**. Shooting
> percentage remains `fg3m_pct = fg3m / fg3a`, a different column. Verified on the rebuilt
> design: `fg3a_pct` means **0.2686** (a 3PA share) against `fg3m_pct` **0.2922** (a 3P%).
> So the naming convention generalizes after all, and `own=` is now belt-and-braces rather
> than load-bearing — worth keeping for explicitness, but the hazard it was written against
> is not real in the shipped basis.

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

## What adoption required — and what it did

Adopted 2026-08-03. Everything in this section was specified before the change and is now
recorded with the outcome; where the specification was wrong, the correction is inline
rather than edited away.

### The head count stays eleven — it did

Counts become `fga, fta, reb, ast, stl, blk, tov` (**7**) and conversions become
`fg3a|fga, fg2m|fg2a, fg3m|fg3a, ftm|fta` (**4**). `fg2a` stops being a head and becomes
**derived** — `fg2a = fga − fg3a` — exactly as `pts` already is. Nothing about the output
contract changes: the same twelve quantities are still produced per player-game and `dk_pts`
is still reassembled from the same eight.

This matters because `tests/test_residual_correlation.py` asserts `len(COMPONENTS) == 11`,
and that assertion should keep passing rather than being edited.

### The generic machinery mostly absorbed it

With `fga` in `COUNT_HEADS` and `("fg3a", "fga")` in `CONVERSION_HEADS`,
`component_rates.build_design`'s five lag-column loops produce `fga_p36_lag1`, `fga_lag1`,
`fg3a_pct_lag1` and `fg3a_lag1` for free, and `count_variants` / `conversion_variants` need
no changes beyond the `own=` parameter already added.

**Two genuine gaps, one predicted and one not.** `season_totals` built its aggregation as
`{c: "sum" for c in COUNT_HEADS + made}`, so with `fg2a` out of `COUNT_HEADS` it stopped
being summed even though it is still the **trials** for `fg2m|fg2a`. That one was foreseen.
The second was not: the per-36 loop ran over `COUNT_HEADS` alone, and `build_design` asks
for `{attempted}_p36_lag1` as every conversion head's volume feature — so `fg2a_p36` and
`fg3a_p36` also stopped existing, and two of the four conversion heads would have lost their
volume column.

Both are fixed by deriving the lists from *both* head lists rather than from `COUNT_HEADS`:
`volume_columns()` is `COUNT_HEADS + made + attempted` and `rate_columns()` is
`COUNT_HEADS + attempted`. **In the two-count basis every attempted column is already a
count head, so both reduce to the old behaviour exactly** — which is what makes the change
safe to verify against the retired basis before flipping the lists.

### Three sites that would fail silently — all three now raise

All three degraded rather than raised, which is the failure mode this repo has been bitten
by before. All three were closed **before** the head lists were flipped:

1. **`season_terms.season_total_arms`** — the eleven-head completeness gate did
   `if any((head, arm) not in models ...): continue`, so a half-migrated head list read as
   "that arm was not run". It now distinguishes the two cases: **no** heads for an arm is
   still a skip (the arm was not fitted), but a *partial* arm raises and names the missing
   heads.
2. **`selected_specs`** — a stale `stan_component_metrics.csv` written before the migration
   makes `specs.get("fga")` miss and fall back to `DEFAULT_COUNT_SPEC = "log_own"`. Given
   that `log_own` on a skewed attempt head is the exact failure this doc exists to correct,
   a silent fallback to it is the worst available default. It now prints the heads it could
   not resolve and says to refit.
3. **`residual_correlation.to_matrix`** filtered with `[c for c in COMPONENTS if c in
   wide.index]`, so an unknown component was dropped from the matrix rather than reported
   and `substitution_r` degraded to NaN. It now raises on **either** a missing or an
   unrecognised name — a matrix missing a head is not a smaller copula, it is a wrong one.

### `season_terms._draw_components` was the load-bearing change

`_draw_components` (`:738-753`) draws every `COUNT_HEADS` entry independently and then draws
makes on **the drawn attempts** — `rng.binomial(np.rint(counts[attempted]), ...)` — because
the chain is `makes | attempts` and conditioning on realized attempts would leak the target.
`test_compose_season_dk_puts_makes_on_the_DRAWN_attempts` pins that.

Under the new basis `counts["fg2a"]` does not exist, so the draw order must become:

```
fga  ->  fg3a | fga  ->  fg2a = fga − fg3a  ->  fg2m | fg2a,  fg3m | fg3a,  ftm | fta
```

**Nothing in the old code could express that ordering** — a conversion head's output
becoming another head's exposure is a new kind of edge in the draw graph. `_draw_components`
now resolves trials through a `trials_for` helper that walks the chain and materializes
`DERIVED_COUNTS` on demand, so the two new edges (`fg3a` becoming trials, `fg2a` being a
difference) are explicit. In the two-count basis every `attempted` is already a fitted
count, so the helper returns on its first branch and behaviour is byte-identical.

One numerical detail the specification missed: `fga` and `fg3a` are drawn from **different
posteriors**, so nothing forces `fg3a <= fga` on a given draw. The derived `fg2a` is
therefore clipped at zero. It bites on a vanishing share of draws — the share head's mean is
nowhere near 1 — and a negative trials count would be an error rather than a small bias.

### The open question, and how it was resolved

`residual_correlation.SUBSTITUTION_PAIR` was `("fg3a", "fg2a")`, and under the new basis
that coupling is removed by construction, so the cell no longer exists among the modelled
eleven. The doc flagged this as a reviewer decision with a recommendation attached. **The
recommendation was taken**: the matrix moved to the new eleven, and the old pair ships as an
explicit contrast under `basis == "legacy_two_count_basis"` — outside `COMPONENTS`, and
therefore outside the matrix, which is what lets `to_matrix` be strict about names.

`SUBSTITUTION_PAIR` is now `("fg3a|fga", "fga")` — the coupling the new basis *does* carry —
and `LEGACY_SUBSTITUTION_PAIR` holds the retired one. `summarize` emits both, side by side,
so the artifact measures what the change bought instead of going quiet where the finding
used to be.

### ⭐ What it bought the copula, measured

| | two-count basis | shot-attempt basis |
|---|---|---|
| the substitution cell | **−0.1248** (`fg3a`–`fg2a`) | **−0.0836** (`fga`–`fg3a\|fga`) |
| largest off-diagonal | +0.1422 (`fg2a`–`reb`) | **+0.1329** (`fga`–`reb`) |
| off-diagonal mean, 11 heads | +0.0071 | **+0.0070** |
| minimum eigenvalue | +0.7559 | **+0.7853** |

The largest negative coupling shrinks by a third and the matrix becomes **better
conditioned**. Note what is *not* claimed: the cell does not go to zero. The substitution
**identity** is gone — one more three is exactly one fewer two, by construction — and what
remains at −0.0836 is a genuine residual relation between shot *volume* and shot *mix*,
which is a different fact and one the copula should carry.

### ⚠️ A new finding the two-count basis could not surface

`make serial-correlation` on the new heads: **`fg3a | fga` has a lag-1 excess of +0.101
(z = 94) and a 10-game block variance inflation of 1.57×** — the largest non-minutes
dependence in the table, above the `fga` count it splits. The three *shooting* conversion
heads remain clean nulls (max |excess| 0.0122).

So "conversion head" stopped being a synonym for "shooting head". Shot **selection** drifts
within a season the way minutes do; shooting **accuracy** does not. The module's summary
line used to take one maximum over all conversion heads, which after adoption would have
reported that drift as a hot hand *and* hidden that the shooting heads are still nulls — it
now reports the two groups separately. `fg3a | fga` is a conversion head by likelihood, not
by subject matter, and it is the one head besides minutes with a serial story worth a second
look.

## What this does NOT deliver

- **It says nothing about `fg2m | fg2a` or the other conversion heads**, which are unchanged
  in both bases and cancel out of the comparison entirely.
- **It does not re-open the season-term verdict.** `make season-terms` is an ablation over
  whichever heads are shipped, so its artifacts have to be regenerated on the new head
  list — but the verdict it reached (no head carries a season term except minutes) is about
  league-level drift, not about the shot basis.
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
- **`fga` is a real fetched column, but `fg2a` is now derived.** `component_targets.parquet`
  carries `fga` directly from the game logs, and `fga − (fg2a + fg3a) == 0` on all 731,906
  rows, so the head's target is not reconstructed. The derived side is `fg2a = fga − fg3a`
  at draw time, which is where the zero-clip lives.
- **Gate 0 is a measurement of the RETIRED basis and is no longer fully re-runnable.** Arm
  A's heads are gone from `stan_component_metrics.csv`, so `substitution_sweep` falls back
  to `LEGACY_ARM_A_SPECS` — pinned constants recording what that artifact selected before
  adoption — and the best-of-16 grid, which is read from the pre-adoption metrics file,
  survives only inside the gate's own artifact. That artifact is the record; re-deriving the
  grid would need the pre-adoption metrics CSV restored.
