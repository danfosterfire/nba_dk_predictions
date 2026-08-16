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
`outputs/predictions/stan_component_substitution_sweep.csv`, 18 rows). Its own target and
its own artifact, deliberately: `substitution_arm` is called from inside
`stan_components.run()`, which writes all three component CSVs together, so refreshing this
comparison through `make stan-components` would cost that target's whole sweep *and* rewrite
`stan_component_metrics.csv` — the artifact this sweep reads arm A's selected specs from.
Eight fits against thirty-seven.

**Validation only, since 2026-08-06.** `src/models/held_out.py` made the test split a
capability rather than a convention, and this gate went through it with every other head:
the test seasons are neither fitted nor scored here, and the end-of-project reading is
`make final-evaluation`'s job. The gate was *already* selecting on validation and reporting
test as confirmation, so nothing about the verdict turns on the change — but it removes a
column a reader could mistake for a replication, and on this gate specifically it removes a
live hazard, because `logit_own` clears its floor on test and fails it on validation. What
the re-run retired is recorded in full below.

10,194 player-seasons; 8,630 fit / **773** select (2022-23, 2023-24 as validation).

- **Arm A** at each head's own selected variant, read via `season_terms.selected_specs()`
  from the artifact that chose it rather than re-derived — `fg2a @ log_own`,
  `fg3a @ log_own_spline`. Adoption removed both heads from that artifact, so the lookup
  now falls back to `LEGACY_ARM_A_SPECS`, which pins exactly those two variants. That
  fallback is load-bearing: without it the specs default to `log_own`, which is precisely
  the handicap this gate exists to remove.
- **Arm B** swept for real: `fga` over (`linear`, `log_own`, `log_own_spline`) and
  `fg3a | fga` over (`linear`, `logit_own`, `logit_own_spline`). **Additive separability
  makes this 3 + 3 fits, not 9 combinations**: the joint density factors as
  `p(fga) · p(fg3a | fga)`, the two factors share no parameters, so minimising the sum is
  exactly minimising each term. `_g0_select` picks per factor and a test pins that the joint
  row it marks is the pair of per-factor winners.

**The regression check is exact, and that is what licenses everything below.** Pinning arm B
at the single `(log_own, logit_own)` configuration reproduces `substitution_arm`'s own arm B
— **10.025950** against **10.025950**, a difference of **1.78e-15**, i.e. floating-point
identity. The two are computed by different code in different modules from different entry
points, so this is the strongest form the check can take, and it is stronger than the
retired one it replaces: that compared this gate's *test* side against a July figure, where
this compares two current measurements on the same rows.

Three weaker corroborations behind it. The retired run fitted the validation side at
500/500 and with the test half gone every fit runs 1000/1000; **no per-factor NLL below
moved by more than 2.0e-04** across that doubling — the fourth decimal, against a margin of
−0.501041, so the quantity being decided is ~2,500× the largest movement the refit produced.
The four no-fit floors are pure arithmetic and reproduce to the digit. And `make
stan-components` fits `fga` and `fg3a|fga` through a *different* code path — the
`{made}_pct_lag1` naming convention rather than this gate's explicit `own=` — agreeing to
within **1.7e-04** on all six of their shared cells, the same order as the gate's own
sampling noise.

⚠️ **That last corroboration expired on 2026-08-15**, for the same reason the decomposition
below did: `make stan-components` re-ran with the preseason block and `make stan-substitution`
did not, so the six shared fitted cells now differ by **0.053 to 0.082** nats. The two floors
still nearly agree — **1.9e-04** on `fga` and **3.9e-04** on `fg3a|fga` — and the residual
there is itself informative: a count floor is unshrunk arithmetic and cannot move, while a
conversion floor's `k` is fitted on the training half, so the window cut reaches it. The first
two corroborations are internal to this artifact and are unaffected.

### Per-factor NLL, against each factor's own no-fit floor

Validation mean negative log-likelihood per player-season; lower is better. Conversion
floors are **shrunk** carry-forwards with `k` fitted on train only, per `CLAUDE.md`.

| arm | factor | variant | val NLL | val floor | selected |
|---|---|---|---|---|---|
| A | `fg2a` | `log_own` | 5.263114 | 5.271028 | ✓ |
| A | `fg3a` | `log_own_spline` | 5.242045 | 5.903396 | ✓ |
| B | `fga` | `linear` | 5.407505 | 5.445210 | |
| B | `fga` | `log_own` | 5.389932 | 5.445210 | |
| B | **`fga`** | **`log_own_spline`** | **5.388496** | 5.445210 | **✓** |
| B | `fg3a\|fga` | `linear` | 4.912709 | 4.619109 | *fails floor* |
| B | `fg3a\|fga` | `logit_own` | 4.636018 | 4.619109 | *fails floor* |
| B | **`fg3a\|fga`** | **`logit_own_spline`** | **4.615622** | 4.619109 | **✓** |

### The verdict

| arm A (selected) | arm B (selected) | margin |
|---|---|---|
| 10.505159 | **10.004118** | **−0.501041** |

**Gate 0 passes**, and the whole handicap decomposition is now measurable on the one split,
where it used to need the test column. `substitution_arm` fits *both* arm-A heads at
`log_own` and scores **10.797078**; fitting `fg3a` at the `log_own_spline` it actually
selects brings arm A to **10.505159**, so **the handicap is 0.291919 nats**. Arm B pinned at
`(log_own, logit_own)` is 10.025950 and swept is 10.004118, worth a further **−0.021832**.
The recorded −0.771128 margin therefore decomposes exactly:

```
−0.771128  (recorded, both arms handicapped)
 +0.291919  un-handicapping arm A — the correction, and it costs arm B most of its lead
 −0.021832  sweeping arm B
 ─────────
 −0.501041  Gate 0
```

The gate survives its own correction with 65% of the recorded margin intact.

⚠️ **That decomposition is the 2026-08-06 state, and its two halves have since come apart. It
is kept because it is the arithmetic the adoption was argued on, not because it still closes.**
The `substitution_arm` figures come from `stan_component_substitution.csv`, which
`make stan-components` rewrites; the swept figures come from
`stan_component_substitution_sweep.csv`, which only `make stan-substitution` rewrites — and
that target is **not** part of `make stan`. The 2026-08-15 components run put the preseason
block on ten of eleven heads, so the first artifact is now a **post-block** reading and the
second is still a **pre-block** one:

| quantity | artifact | 2026-08-06 | today |
|---|---|---|---|
| recorded margin | `…_substitution.csv` | −0.771128 | **−0.7218** |
| handicapped arm A | `…_substitution.csv` | 10.797078 | **10.738** |
| arm B pinned | `…_substitution.csv` | 10.025950 | **10.0162** |
| arm B pinned | `…_substitution_sweep.csv` | 10.025950 | *unchanged* |

**The floating-point identity check at the head of this section is the casualty, and it did
exactly the job it was built for.** Two modules computing the same quantity from different
entry points agreed to **1.78e-15**; they now disagree by **9.8e-03**, and that gap is the
preseason block rather than a bug — the check caught a configuration drift that no single
artifact could have shown. Restoring it means re-running `make stan-substitution` so both
sides sit on the same heads, which is a refit for record-keeping and is deliberately not being
done. **Until then the handicap (0.291919) and the surviving share (65%) are not
re-derivable**: subtracting a pre-block number from a post-block one is not a decomposition,
and quoting the difference as though it were would be the exact error this doc was written to
correct.

None of this reopens the verdict. Gate 0's own margin — **−0.501041**, both arms swept inside
one artifact — is internally consistent and unmoved, and the finding below it, that the
reparameterized *floor* beats the canonical *fitted* configuration, is a comparison within the
sweep alone.

### ⭐ The finding worth remembering: the coordinate change beats the fitting

Comparing the two bases **at their no-fit floors** — prior per-36 rate × minutes for the
count, shrunk carry-forward for the share, no features anywhere:

| | arm A | arm B | difference |
|---|---|---|---|
| no-fit floor | 11.174424 | **10.064318** | **−1.110105** |
| fitted (selected) | 10.505159 | 10.004118 | −0.501041 |

**Arm B's no-fit floor beats arm A's fitted configuration by −0.440841 nats.** Writing the
identity in the right basis is worth more than everything the canonical basis's fitting
buys, and it is ~88% of the total margin: arm B's own fitted heads add only **−0.060200**
on top of its floor.

The floors are the one part of this gate that is arithmetic all the way down, so this table
is the most stable thing in the doc — and the gap it reports is **wider** on validation than
the −0.938427 the retired test column showed. Since the floors themselves reproduce exactly,
the widening is the fitted side moving, not the benchmark.

That reframes the result. This is not "a better model of shot attempts" — it is the same
information written in coordinates where the dependence is structural instead of residual.
It is the same shape as the repo's standing finding that the component rate side is nearly
saturated by a carry-forward: when the floor is that strong, the parameterization is where
the remaining leverage is.

### Two secondary results

- **The share head needs its spline to clear its floor, and `logit_own` alone does not.**
  On validation `logit_own` reads 4.636018 against a floor of 4.619109 — *below* the floor —
  and only `logit_own_spline` clears it, at 4.615622. **This is the one place in this gate
  where the split move changes what would ship**: on test `logit_own` cleared (4.611402
  against a 4.652797 floor), so a reader taking the test column would have selected a
  variant that fails its floor on the split that selects. Same shape as the NB-vs-Poisson
  result on the count heads and the `binomial` arm on the composition: the flexible term is
  not a refinement, it is the difference between a model and a failure. `linear` fails
  outright.
- **For `fga` the spline is nearly free and nearly pointless** — 5.388496 against
  `log_own`'s 5.389932 on validation, a margin of 0.0014. Unlike `fg3a`, where the spline is
  decisive, `fga` is a well-behaved total and the log scale is essentially the whole answer.
  It is selected because validation selects it, not because it matters.

### Sampler

8 fits, **max R̂ 1.0047**, **0 divergences**, **26.8** minutes total, and **0** treedepth
saturations. Dropping the test half halved the fit count while every remaining fit went from
half-length to full-length chains, and the sampler is *better behaved* for it on both
counts: max R̂ falls from 1.0087 to 1.0047, and the 5 saturated iterations the retired run
logged on `fg3a@log_own_spline` are gone. The spline arms remain the cost centre —
`fg3a|fga@logit_own_spline` alone is 9.7 of the 26.8 minutes.

### ⚠️ What the validation-only re-run retired, 2026-08-06

Everything in this block is a **superseded measurement kept as a record**, not a live figure.
`src/models/held_out.py` locked the test split, `substitution_sweep` stopped fitting or
scoring it, and this gate's artifact was rebuilt with the validation half only. No artifact
verifies these numbers any more, so `src/docs_audit.py` carries them as
`Claim(historical=True)` — presence-checked, so they cannot be tidied away, and exempt from
the value check, because there is nothing left to check them against. **The figures above
are the live ones; every figure below is history.**

The retired run scored **791** test rows against the validation half's 773, over **52**
artifact rows and **16** fits (max R̂ **1.0087**, 0 divergences, **46.5** minutes) — the
test side being the majority of that compute, at double the sampler iterations.

| arm | factor | variant | test NLL | test floor |
|---|---|---|---|---|
| A | `fg2a` | `log_own` | 5.229492 | 5.292842 |
| A | `fg3a` | `log_own_spline` | 5.248561 | 5.731185 |
| B | `fga` | `linear` | 5.426520 | 5.432802 |
| B | `fga` | `log_own` | 5.379640 | 5.432802 |
| B | `fga` | `log_own_spline` | 5.376766 | 5.432802 |
| B | `fg3a\|fga` | `linear` | 4.929134 | 4.652797 |
| B | `fg3a\|fga` | `logit_own` | 4.611402 | 4.652797 |
| B | `fg3a\|fga` | `logit_own_spline` | 4.607737 | 4.652797 |

The test verdict was arm A **10.478052** against arm B **9.984503**, a margin of
**−0.493549** — the figure the adoption decision was written against. Its decomposition:
**0.305646** of the recorded 0.792657-nat margin was the handicap, leaving **−0.487010**,
and sweeping arm B added a further **−0.006539**. The `(log_own, logit_own)` cell summed to
**9.991042**, which is what made it the regression check against `substitution_arm`.

Two things retired with the test column that the validation half cannot replace:

- **Arm A's best-of-16.** The grid was read out of `stan_component_metrics.csv`'s `test_nll`
  at zero extra fits; adoption removed `fg2a` and `fg3a` from that file and the split move
  removed the column, so `_arm_a_grid` now returns empty by both routes. Its minimum was
  `log_own_spline + log_own_spline` at **10.476413**, against which arm B won by
  **−0.491910**. **The loss costs the argument 0.001640 nats** — that is the whole distance
  between the grid minimum and arm A's *own selected* configuration (10.478052), which the
  gate still fits. "Arm B wins even against arm A's most favourable configuration" and "arm
  B wins against arm A as it would actually be fitted" were never meaningfully different
  claims here, and only the second one needs the grid.
- **The floor comparison on test**, which read **11.024027** against **10.085599** for a
  **−0.938427** gap, putting arm B's floor **−0.390814** ahead of arm A's best fitted and
  arm B's own fitting at **−0.101096** on top of its floor. The validation table above is
  the live version of this and the gap is *wider* there, so the finding survives the move
  with room to spare.

The handicapped pair that motivated the whole gate — **−0.771** on validation and **−0.793**
on test, the latter unsigned as **0.792657** — is now history on both halves.
`stan_component_substitution.csv` carried the validation figure and was value-checked there
until 2026-08-15, when `make stan-components` re-ran with the preseason block and the same
cell became **−0.7218**; that is the live claim now and −0.771 is presence-checked beside it.
The test figure had been recovered from *this* artifact's test rows after that file went
validation-only on 2026-08-06; the re-run removed that last copy, so it is a presence-checked
record like the rest of this block. **Neither movement touches the gate**, which is decided
inside the sweep alone.

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

- **The margin is per player-season on 773 validation rows**, and no paired bootstrap has
  been run on it. The repo has one recorded false positive that survived a paired interval
  (the minutes-head nonlinearity arm, 95% CI [−0.079, −0.015] and it did not replicate), so
  the interval was never the defence. **What the defence used to be — that the result
  replicates across splits — is gone with the test column**, and that is the honest cost of
  the move. What replaces it is weaker in kind but not nothing: the margin survived a
  doubling of chain length in the fifth decimal, it is ~25× the largest movement any
  refit has produced, and the floor-to-floor comparison that carries most of it involves no
  sampling at all.
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
  adoption. The best-of-16 grid is worse off than that: it was read from the pre-adoption
  metrics file's `test_nll`, and **both** halves of that lookup are now gone — the head rows
  to adoption, the column to the split move — so `_arm_a_grid` returns empty by two
  independent routes and re-deriving it would need the pre-adoption metrics CSV restored
  *and* the test split unlocked. It is worth 0.001640 nats, so this is a bookkeeping loss
  rather than an evidential one.
