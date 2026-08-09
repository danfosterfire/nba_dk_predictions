# Minutes Composition Plan: The Team-Game Allocation as a Decomposed Multinomial

This is a planning doc in the house pattern — direction first, results filled in as the
results land. It executes the "team-game composition alternative" costed in
`docs/predictions-plan.md`. It was built as a **pilot gated on measurement**; the pilot
passed Gates A–D, the full-window refit (Gate E) was taken on **2026-08-04**, and the head
now ships in `make stan`. It is **not** a replacement for the season-collapsed minutes head
(`make stan-minutes`) — the two compose, and what each owns is stated in the results below.

## The problem

The season minutes head predicts each player's minutes **marginally**. Two things it
cannot represent, both per-game facts:

- **Zero-sum.** A team's minutes are exactly `5 × game_length` per team-game. Independent
  per-player draws violate it on every simulated game, and the violation is the mechanism
  behind teammate-absence redistribution — when a starter sits, his minutes go
  *somewhere*, and independent marginals send them nowhere.
- **The individual cap.** No player exceeds `game_length`. The season head's simulator
  imposes it row-wise (`BetaBinom(game_length, μ, ρ_game)`), so each form gets one
  constraint exactly and the other not at all — the trade recorded in
  `docs/predictions-plan.md`'s warning box.

This plan builds the model that gets **both by construction**: allocate each team-game's
`N = 5 × game_length` minutes among the K players who played, by decomposing the
multinomial into a **sequence of binomial trials** — piloted in
`src/stan/demo_decomposed_multinomial.stan` — ordered by prior-season minutes share, with
per-player caps `U = game_length` enforced through the trials, not checked after the fact.

## Settled decisions (2026-07-30, with Danny — do not relitigate)

1. **One pooled Stan fit across regulation and overtime.** `N` (240/265/290/…) and `U`
   (48/53/58/…) enter as per-row data, so one fit handles every game length natively; an
   `n_overtimes` covariate (and an OT × own-share interaction) lets allocation shift in
   OT. Separate per-class fits would fragment 2OT into ~528 team-games and 3OT/4OT into
   82/12 — unfittable — while sharing nothing with the 94% of games at regulation.
2. **K = players who played** (mean 10.36 per team-game, K ∈ [6, 15], full 1996-97+
   window). The availability head already owns the played/not-played margin; the
   composition answers "given who played, how are the minutes split". The dressed-roster
   alternative (played + DNP) would restrict the window to 2006-07+ and duplicate a
   margin another head models.
3. **Deliverable = this doc + a pilot** on a recent-season subset with a timing probe,
   posterior predictive checks, and a comparison against the season head + ρ_game
   comparator. The full-window fit is a later decision, gated on the pilot (Gate E).
   ✅ **Gate E taken 2026-08-04** — the head is fitted on all 30 seasons.
4. **The game-length class predictor is a geometric tail** fit on training seasons:
   P(any OT) and P(one more OT | current), which covers 3OT/4OT for free. A covariate
   model is not worth it at a 5.93% base rate, and projected closeness is not knowable
   preseason.

## What is already measured (cite, don't re-derive)

- **71,092 team-games** (35,546 regular-season games × 2), **736,410 played player-games**,
  mean 10.36 players per team-game — `availability_panel.parquet` ⋈
  `game_length.parquet`, both unfiltered. Verified this session: 0 unmatched lengths,
  0 unreliable, no team-game with `N > K·U` (the allocation is feasible everywhere).
- **Team sums close to 0.0009 minutes mean absolute residual**; the worst game is off by
  3.08 minutes (a scorer error, absorbed by rounding and reported by the builder). The
  0.617 figure in `CLAUDE.md` is the per-team grid-snap residual, a different quantity.
- **Game-level dispersion is 4.65× binomial** (ρ_game = 0.0776, `make stan-minutes`,
  `stan_minutes.game_level_dispersion`) — the fact that says plain-binomial allocation
  steps will be far too tight, and the reason the ladder's expected winner is the
  beta-binomial arm.
- **OT counts 1,942 / 264 / 41 / 6** at 1/2/3/4 overtimes over 37,986 games of both
  season types (`make game-length`; the fitted tail uses regular-season train games
  only, so quote the basis when quoting the parameters). The continuation probability
  is nearly constant in depth — 311/2,253 = 0.138, 47/311 = 0.151, 6/47 = 0.128 — which
  is what makes two parameters enough.
- **~14% of roster minutes lack a usable S-1 row** (`make context-value`). The
  composition **cannot drop those players** — the sum must be complete — so rookie/no-prior
  handling is mandatory here, unlike every existing head, which filters instead.
- **Serial dependence is out of scope.** This model is iid across games given features;
  the 2.43× ten-game block variance inflation stays with the residual-serial-process
  recommendation in `docs/predictions-plan.md`. What is being bought is zero-sum,
  cap-exactness, and fitted redistribution — not serial structure.

## The model

### Sequential decomposition, ordered by prior share

Within a team-season, order players by prior-season minutes share (largest first;
no-prior players last — see below). For team-game g with ordered played players
1..K, allocate sequentially: player k draws

```
y_k ~ BetaBinomial(m_k, p_k, ρ)        m_k = min(U_g, R_k)      R_k = N_g − Σ_{j<k} y_j
logit(p_k) = logit(p̄_k) + α + x_k'β    p̄_k = clip( (w_k / Σ_{j≥k} w_j) · R_k / m_k )
```

with `w` the prior-share composition and the last player deterministic, `y_K = R_K`. The
binomial arm is the ρ → 0 special case of the same file.

- **Trials are the remaining *capacity*, `min(U, R)` — that is how the cap is enforced.**
  Where no cap binds (`R ≤ U`, the later steps) this is exactly the demo's stick-breaking
  decomposition of the multinomial; where it binds, trials-at-capacity replaces an upper
  truncation of `Binomial(R, ·)` at `U`. The two differ only where the pure multinomial is
  already being modified, and trials-as-data keeps the likelihood **fully vectorized** —
  an upper-truncated beta-binomial needs an `O(support)` CDF inside the gradient, which
  is the difference between a `betabinomial_glm.stan`-shaped fit and days of compute.
- **The feasibility lower bound is data too.** `lo_k = max(0, R_k − J_k·U)` (J_k players
  remaining after k) guarantees the players still to come can absorb the remainder, which
  is what makes the deterministic last step valid by induction. Measured on the built
  frame it binds on **1.65% of rows** (10,978 of 665,318 non-last) — more than the
  short-rotation intuition suggested, because ordering is by *prior* share, so a
  heavy-minute no-prior rookie sits late in the sequence and leaves a large remainder
  behind the veterans. It is handled as an explicit truncation for exactly the rows
  where `lo > 0`, a loop over ~1.65% of the data.
- **The offset is the floor.** With α = β = 0 the mean allocation is
  `w_k / Σw × N` — prior shares renormalized over who played, carried forward. That is
  the mandatory no-fit floor, sharing the model's code path, and it is **already a
  redistribution model**: a missing teammate shrinks `Σw`, scaling every remaining
  player's allocation up proportionally. β then fits *deviations* from proportional
  redistribution — the "who absorbs the minutes" question as a fitted quantity.
- **The offset saturates at `OFFSET_CLIP = (0.02, 0.93)`, and that bound was bought
  with a failed run.** On ~1% of rows (6,511 of 665,318) the proportional
  carry-forward demands *more than the cap* — a star on a roster whose played-set
  prior shares sum well below 5 gets `w/Σw × N > U`. A hard clip at `1 − 1e-3`
  asserts "he plays 47.95 of 48" with a near-zero beta shape parameter; the data
  disagree by 5–15 minutes, the curvature at those rows is enormous, and the first
  smoke fit spent 60+ minutes inside warmup at max treedepth without producing one
  draw. Saturating at 0.93 (~44.6 of 48 — a realistic "cap binds" allocation) plus
  an `offset_clipped` indicator feature — so β can learn the correction on exactly
  those rows — restores a `betabinomial_glm`-like geometry.
- **Two demo flaws are fixed, loudly.** `demo_decomposed_multinomial.stan` declares `U`
  but never enforces it in the likelihood (a star's step at mean ~40 of 240 puts real
  mass above 48 — with 4.65× dispersion, ~16%), and its final check compares an integer
  count to a real probability. Here the cap is structural (trials), and `transformed
  data` rejects on `Σy ≠ N`, `y > m`, or `y < lo` — build failures, not metrics.

### Four sampling-cost lessons, each bought with a failed or crippled run

Net effect of the four: the one-season probe went from **60+ minutes without one
draw** to **65 seconds for 300+200 × 2 chains at treedepth 4**.

1. **The offset saturation above** — a hard `1 − 1e-3` clip put near-zero beta shape
   parameters on ~6,500 rows and collapsed the step size.
2. **Never call `beta_binomial_lccdf` inside the gradient — and never compute a tail
   as `1 − head`.** Stan routes the beta-binomial CDF through the generalized
   hypergeometric (`grad_F32`), whose autodiff is slow enough that ~400 truncation
   rows made one gradient cost seconds: `optimize(iter=30)` took **>600s** with it
   and **0.8s** after replacing it with a pmf-ratio-recurrence sum. The first
   replacement summed the *head* and subtracted — and the binomial arm's first
   full-data fit then died at initialization on every chain, because a tiny tail
   rounds the head mass to exactly 1.0 and `log1m(1)` is −inf while the true target
   contribution (pmf over tail, both tiny) is O(1) and finite. The shipped form sums
   the **tail upward in log space** (`beta_binomial_log_tail_mass`, pinned against
   `scipy` including a tail small enough that the head form returns −inf). Same
   story as the repo's standing `s * inv_logit(-eta)` convention, twice over: the
   algebraically identical form is not the computationally identical form.
3. **Per-column imputation flags are a degenerate subspace here.** A rookie loses
   every design column at once, so `impute()`'s 18 flags were exact copies —
   C(18,2) = 153 duplicate standardized columns, which the sampler pays for in
   treedepth. One `design_missing` indicator carries the same information
   (`test_variants_carry_no_duplicate_feature_columns` pins it). The other heads
   never hit this because they *filter* the no-prior rows; the composition keeps
   them, so it inherits the whole block-missingness structure.
4. **Use a dense metric.** With the three fixes above the posterior still held NUTS
   at treedepth 8–9 under the default diagonal metric — residual linear correlations
   a diagonal metric cannot absorb. `metric="dense_e"` drops it to treedepth 4 and
   cuts the same probe fit **645s → 65s**; at ~25 parameters the dense adaptation
   costs nothing. (`stan_utils.sample` now takes `metric`; the other heads keep the
   default.)

### Ordering, and the no-prior players

The order key is computable preseason: prior-season realized minutes share (from the
unfiltered panel, lag 1, falling back to lags 2–3 with a staleness flag), veterans
descending, then no-prior players by an **expanding-window rookie share prior** per draft
bucket (`team_context.DRAFT_BUCKETS` + undrafted; means over seasons strictly before the
row's season, so it is point-in-time by construction), tie-broken by draft number then
`player_id`. The same imputed share feeds `w` in the offset and the `logit_share_lag1`
feature, with a `no_prior` flag. Ordering is fixed within a team-season; each game's
played subset inherits it.

**Order-dependence is real for the dispersed arm** — with beta-binomial steps and caps
the ordering is part of the model, not a matter of indifference as it is for the pure
multinomial identity. Largest-first is the canonical choice (the best-informed sticks
break first); a reversed-order refit of one variant is cheap insurance if the pilot
result looks order-sensitive.

### Features

The season minutes head's block, per player-row: `FEATURE_COLS` minus
`minutes_per_game_lag1`, plus `logit_share_lag1` (`src/models/availability.py`,
`src/models/stan_minutes.py`), plus `no_prior` and the imputation flags, all standardized
on train rows (player-games, so standardization weights players by games played —
deliberate, it matches the likelihood's weighting). The OT variant adds `n_overtimes` and
`n_overtimes × logit_share_lag1` — per-game covariates, which is the point: OT minutes
should tilt toward the players already playing most.

### Integer minutes

Minutes are recorded to the second; the multinomial needs integers summing to exactly
`N`. Largest-remainder rounding within each team-game: floor everyone, hand the deficit
to the largest fractional parts, never lifting a player past `U`. The max team-game
residual it absorbs is 3.08 minutes on one game; the builder reports the distribution.

## Variant ladder and selection

| variant | steps | features | what it isolates |
|---|---|---|---|
| `carry_forward` | beta-binomial, ρ fitted on train residuals | offset only, no Stan | the no-fit floor |
| `binomial` | binomial (`dispersed=0`) | offset + β | is the pure decomposition enough? (expected: too tight) |
| `betabinom` | beta-binomial | offset + β | the expected winner, per the 4.65× fact |
| `betabinom_ot` | beta-binomial | + `n_overtimes`, OT × own | does OT shift allocation? |

Selection on **validation CRPS only** (2022-23/2023-24 carved from train; test
2024-25/2025-26 is confirmation — this repo has already shipped one false positive
selected on test). Every variant is quoted against the floor; `beats_floor` on every row.
An `independent_comparator` row — the season head refit outside the evaluation split, plus
independent `BetaBinom(game_length, μ, ρ_game)` draws per player-game — is reported in
the same table but excluded from selection: it is the incumbent, not a variant.

## Pilot design and gates

Frame built over all seasons (lags and expanding rookie priors need the history), fitted
from `stan.composition.first_season`, which was **2018-19** for the pilot and is
**1996-97** now that Gate E is taken. In order:

- **Gate A — timing.** Fit `betabinom` on the most recent train season at select iters;
  extrapolate linearly in rows × iters to the six-fit sweep. Abort loudly past
  `max_extrapolated_hours` (24). Fallbacks, in order: cut `binomial` from the ladder,
  shorten select chains, subsample **train** team-games (never val/test).
- **Gate B — convergence.** R̂ ≤ 1.01, 0 divergences on the test-side fits (select-side
  slack tolerated per the `fg2m|fg2a` precedent); treedepth saturation reported, not fatal.
- **Gate C — the floor.** The selected variant beats `carry_forward` on validation CRPS,
  confirmed directionally on test. Nothing clears it → per `CLAUDE.md`, not a model; stop
  and record the null.
- **Gate D — the incumbent.** Selected variant's test per-player-game CRPS ≤ the
  comparator's. **If it is a wash, the decision moves to what marginal CRPS cannot see**:
  the comparator's team-sum error against the composition's exact zero, and
  dispersion-by-tier calibration. Zero-sum is the capability being bought; the argument
  must be made on those terms, not as "a better minutes model" — the season head clears
  its own floor by only +0.030 R², so the marginal headroom is known to be small.
- **Gate E — full window** (decided later, recorded here): only if B–D pass and the
  measured per-row cost extrapolates the 736k-row fit inside the 6–16 h envelope
  `docs/predictions-plan.md` already costed. ✅ **Taken.** The extrapolation said 12.8 h and
  the sweep took **20.9 h** — outside that envelope, and the reason (superlinear per-row
  cost) is recorded under Gate A in the results.

### Posterior predictive checks

- **Team sums**: exact by construction for the composition (asserted on every draw);
  the comparator's mean |Σ − N| is reported beside it — the asymmetry is the finding.
- **Starter share in OT**: observed vs simulated share of team minutes taken by the top-5
  ordered players, regulation vs OT games — the check on the OT variant's mechanism.
- **Dispersion by tier**: variance ratio (realized squared residual over mean simulated
  variance) by quartile of prior share — where the binomial arm should fail and the
  beta-binomial arm should not, and where a single shared ρ will show its limits.
- **Joint per-team-game NLL** vs the independent model, in the `substitution_arm` output
  format. ⚠️ Reported with an explicit caveat: unlike the 3PA/2PA case this is **not** a
  unit-Jacobian bijection — the composition concentrates mass on the simplex slice the
  data always satisfies, so it wins joint NLL partly by knowing the constraint. Report,
  don't headline.

## What this does NOT deliver

- **Serial structure.** iid across games given features; the 2.43× block inflation needs
  the residual serial process regardless of how this pilot lands.
- **DNP-CD.** The played/not margin stays with the availability head (decision 2).
- **Per-position dispersion.** One shared ρ; a 34-mpg starter's step is steadier than a
  fringe player's. Rank- or share-graded ρ is a named future variant, on the evidence of
  the tier PPC.
- **A season term.** Same gap as every other head — see the season-effects TODO in
  `docs/predictions-plan.md`.

## Risks

- **Compounding at prediction time**: the played set K comes from availability draws, so
  offsets are recomputed per simulated roster and availability error amplifies into
  minutes error rather than averaging out (`docs/predictions-plan.md`'s caveat, inherited
  in full).
- **The rookie share prior carries ~14% of roster minutes** on draft-bucket means; the
  `team_context.rookie_priors` machinery is the upgrade path if the pilot shows those
  rows dominating the error.
- **Standardization on replicated rows** weights the scaler by games played; documented
  rather than corrected, since the likelihood weights the same way.
- **44 played rows have `min == 0`** (sub-30-second cameos rounding down); y = 0 is in
  the support, kept.

## Results — full window, re-measured on VALIDATION 2026-08-08, `make stan-composition`

**Verdict: Gates A–E all pass. Gate E is taken — the head is fitted on all 30 seasons and
is now part of `make stan`.** The composition clears its floor and beats the incumbent on
the incumbent's own metric, while delivering the team-sum exactness that was the point of
building it. Every figure below is from `outputs/predictions/stan_composition_*.csv` and
audited by `make docs-audit`.

> **Read this before comparing against anything written before 2026-08-08.** Gate E was
> originally taken on the **test** split, on 2026-08-04, before `src/models/held_out.py`
> locked it. The sweep is now validation-only, the artifact has been regenerated, and every
> figure in this section is a **validation** measurement on 52,295 rows / 4,920 team-games.
> The retired test column is preserved in full below rather than deleted, because the
> head's whole verdict was once quoted from it. **Nothing reversed.** The selected arm, the
> ordering of all six rows, and every gate outcome are unchanged.

Full window 1996-97 →: train 631,158 rows / 61,252 team-games (1996-97 → 2021-22),
val 52,295 / 4,920 (2022-23 + 2023-24). The test seasons are not fitted or scored.

| variant | val CRPS | val R² | val PIT KS | selected |
|---|---|---|---|---|
| `carry_forward` (floor) | 4.6776 | 0.4442 | 0.0496 | |
| `binomial` | 4.9388 | 0.4752 | **0.1919** | fails the floor |
| `betabinom` | 4.5417 | 0.4699 | 0.0496 | |
| `betabinom_ot` | 4.5431 | 0.4697 | 0.0494 | |
| **`betabinom_ot_graded`** | **4.4945** | 0.4741 | 0.0428 | **✓** |
| `independent_comparator` | 4.7842 | 0.4024 | 0.0483 | (incumbent) |

**The selection replicated across a doubling of chain length, which is what makes it
readable.** The validation side used to run at `select_warmup`/`select_samples` and now runs
full-length, so every fitted row is a fresh measurement — and **no arm moved by more than
0.0019 CRPS** (`binomial` −0.0006, `betabinom` −0.0004, `betabinom_ot` +0.0000, the graded
arm +0.0019). `carry_forward` and `independent_comparator` reproduced to six decimals, the
first because it is arithmetic and the second because it never trains on the composition
window at all. Contrast the season-term ablation, where 9 of 13 heads flipped their selected
arm under a change that should not have touched them: an ablation whose winner moves when
the sampler is merely run longer is measuring noise, and this one does not move.

**`binomial`'s calibration failure now has a validation twin, which it did not before.** The
old schema wrote `test_pit_ks` and no `val_pit_ks`, so the sharpest statement of the arm's
failure was a held-out number; it now reads **0.1919** on validation against the floor's
0.0496, reproducing the retired 0.1948 almost exactly.

- **Gate A — timing, and it is the gate that behaves worst.** The probe fit 26,039 rows in
  155 s and extrapolated the sweep to **8.3 h**; the sweep took **9.78 h** of sampler time
  across its four arms, **1.17×** the estimate. The cause is that per-row cost is
  **superlinear in rows**: 5.95 ms/row on the 26k-row probe against **14.83 ms/row** on the
  631k-row `betabinom` fit. More data sharpens the posterior, which shrinks the step size,
  which buys more leapfrog steps per iteration on top of a per-gradient cost that is itself
  linear. **Treat Gate A as a lower bound, not an estimate.** The two-pass sweep missed by
  1.63×, so the miss is smaller here but the direction is invariant, and the honest reading
  is that the multiplier is not a constant — see the retired block.
- **Gate B — convergence, and it is now clean.** Max R̂ **1.00935**, **0** divergences over
  6 fits, and **every fit converged**. The one fit that failed the 1.01 bar in the two-pass
  run was `binomial/val` at R̂ 1.0113 with ESS 421 — a *selection* fit running at half
  length. At full length the same fit reads R̂ 1.00649 with ESS 917. Raising selection to
  full-length chains removed the only diagnostic failure this head had.
- **Gate C — the floor.** The selected variant beats `carry_forward` by **−0.1832** minutes
  of validation CRPS (**−3.92%**). And the `binomial` arm is **worse than the floor**
  (4.9388 against 4.6776) with PIT KS 0.1919 against 0.0496: the pure stick-breaking
  decomposition is far too tight, exactly as ρ_game = 4.65× binomial predicted. The
  dispersion is not an optional refinement — it is the difference between a model and a
  failure, the NB-vs-Poisson lesson again.
- **Gate D — the incumbent.** **−0.2898** minutes of validation CRPS against the incumbent
  (4.4945 vs 4.7842, **−6.06%**) — not the expected wash — plus the capability gap: the
  comparator's mean absolute team-sum error is **33.89** minutes per team-game against the
  composition's exact **0** on every draw of every game. The comparator also carries a
  **−0.5521** minute bias where the composition's is 0 by construction.
- **Gate E — the full window itself.** Taken. `stan-composition` joins `make stan`, after
  `stan-minutes`, which it imports from and measures itself against.
- The OT interaction remains real but tiny: `betabinom_ot` **loses** to plain `betabinom`
  by 0.0014 CRPS on validation. Treat it as a refinement, not a driver; it survives in the
  shipped arm because the graded variant is built on it.

### What the full window changed against the pilot

The ordering is unchanged and every gate still passes, but three things moved enough to
restate rather than carry forward.

| | pilot (2018-19 →) | full window (1996-97 →) |
|---|---|---|
| vs the floor | −0.312 | **−0.1832** |
| vs the incumbent | −0.406 | **−0.2898** |
| graded ρ, fringe → star | 0.1480 / 0.1125 / 0.0874 / 0.0613 | **0.1768 / 0.1300 / 0.1115 / 0.0855** |
| shared ρ | 0.0970 | **0.1211** |
| ρ spread | 2.41× | **2.07×** |
| mean \|variance ratio − 1\|, graded | 0.1055 | **0.1782** |

The pilot column is a test-split measurement and the full-window column is validation, so
the two are not a controlled contrast — different rows and different training data. What
they are good for is direction and magnitude, and on both the story is unchanged.

- **Every ρ is larger and the spread is smaller.** Fitting 26 seasons instead of four raises
  the fitted dispersion at every tier (fringe 0.148 → 0.177, star 0.061 → 0.085) and
  compresses the fringe-to-star ratio from 2.41× to **2.07×**. Role grading is still real
  and still monotone; it is less extreme once the model has to cover three decades of
  rotation practice rather than one.
- **The absolute win is smaller and still decisive.** −0.2898 against the incumbent instead
  of −0.406, and −0.1832 against the floor instead of −0.312. Neither is close to a wash.
- **⚠️ The graded arm's calibration gain is real but weaker, and the pilot oversold it.**
  Mean |variance ratio − 1| falls **0.2730 → 0.1782**, a **35%** cut rather than the pilot's
  59%, and the star tier lands at **0.8136** rather than the pilot's near-exact 0.988.

**Posterior predictive checks** (validation split, selected variant):

| tier | shared ρ | graded ρ |
|---|---|---|
| q1 fringe | **1.2224** | **1.0465** |
| q2 | 0.8430 | 0.8132 |
| q3 | 0.6589 | 0.7071 |
| q4 star | **0.6285** | **0.8136** |
| mean \|ratio − 1\| | **0.2730** | **0.1782** |

- Three of the four tiers improve — fringe 1.22 → 1.05, q3 0.66 → 0.71 and star 0.63 → 0.81
  — and **q2 still gets slightly worse** (0.843 → 0.813), which is the same structural point
  the pilot recorded: the fitted ρ is the dispersion of a **sequential step** while the
  ratio is measured on a player's **marginal** minutes, and because the order is prior-share
  descending, a low-share player breaks his stick last and inherits the accumulated
  remainder variation from everyone ahead of him. Grading step dispersion does not map
  one-to-one onto marginal variance by tier.
- **Three of four tiers sit below 1**, where the pilot had two. The model is systematically
  a little over-dispersed in aggregate at full window — worth a look before the simulator
  consumes it, and a better-posed target than chasing q2 specifically.
- **Starter share in OT** — observed: top-5 ordered players take **0.6013** of team minutes
  in regulation and **0.6423** in OT (+4.1 pp). The composition simulates **0.5912** →
  **0.6415**, reproducing the shift as +5.0 pp with a −1.0 pp level undershoot; the
  independent comparator compresses it (**0.5923** → **0.6280**).
- **Joint per-team-game NLL** (plug-in, non-bijection caveat) — composition **32.862**
  against independent **37.984** on validation. Contrast, not a headline.

**OT tail** — ⚠️ **moved out of this head on 2026-08-09.** `fit_ot_tail`,
`sample_game_length` and `ot_tail_check` are deleted from `stan_composition.py`, and
`stan_composition_ot_tail.csv` with them; `src/models/stan_game_length.py`
(`make stan-game-length`) owns the game-length draw now, as a Bayesian head with a full
posterior and a fitted season trend. This head was never a consumer — it reads the
**realized** `game_length` on every row it fits or scores, and only a forward simulation
needs a draw — so the tail was parked here rather than belonging here. See
`docs/simulations-plan.md`, "The second prerequisite".

**The figures below did not move**, because the retired pair survives as the new head's
mandatory no-fit floor: the same function on the same rows. They are re-derived by
`make stan-game-length` and audited from
`outputs/predictions/stan_game_length_{metrics,ppc}.csv`.

Invariant to the split by construction: the floor receives 1996-97 → 2021-22 either way, and
the fitted parameters reproduced to six decimals across the re-run. Fit on 30,626
regular-season training games: p_any = **0.0608**, p_more = **0.1408**. Scored on 2,460
validation team-games: predicted **128.4** / 18.1 / 3.0 games at 1/2/3+ OT against observed
**120** / 18 / 0 — the two-parameter form holds and overpredicts single-OT by **7.0%**, a
milder era decline in OT rate than the 15.7% the retired test column showed.

**That 7.0% is what the replacement fixes**, and it is the reason the two-parameter form was
retired rather than merely relocated: a fitted season slope takes the summed OT-class error
on the same 2,460 games from **22.97** to **9.42**. The caveat this paragraph used to end
with — "`sample_game_length` is the simulator's game-length draw and carries that caveat with
it" — no longer applies to anything the simulator will call.

### The retired TEST column, held for the record

Gate E was taken on these figures on 2026-08-04, before `src/models/held_out.py` locked the
held-out split. The sweep no longer fits or scores test, so **no artifact can back them**:
`src/docs_audit.py` carries them as `Claim(historical=True)`, which presence-checks them and
exempts them from the value check. The failure that guards is **deletion**, not drift. They
are kept because this head's entire published verdict was once quoted from them, and because
"what did the held-out column say before we stopped looking at it" is the question a reader
of the old `CLAUDE.md` will arrive with.

| variant | test CRPS | test R² | test PIT KS |
|---|---|---|---|
| `carry_forward` (floor) | 4.8576 | 0.3678 | 0.0354 |
| `binomial` | 4.9732 | 0.4259 | 0.1948 |
| `betabinom` | 4.5893 | 0.4247 | 0.0405 |
| `betabinom_ot` | 4.5848 | 0.4255 | 0.0414 |
| `betabinom_ot_graded` | 4.5592 | 0.4244 | 0.0393 |
| `independent_comparator` | 4.9140 | 0.3301 | 0.0769 |

Everything else that was measured on test and is now measured on validation:

- **Gates C and D**: the selected arm beat the floor by −0.2985 and the incumbent by
  −0.3548 (−7.2%), against −0.1832 and −0.2898 (−6.06%) on validation. The comparator's
  team-sum error was 36.87 minutes and its bias −1.2856.
- **Starter share in OT**: observed 0.5882 → 0.6314, simulated 0.5998 → 0.6464.
- **Variance ratios**, shared then graded: 1.3466 / 0.8089 / 0.7691 / 0.5973 against
  1.0845 / 0.7687 / 0.8217 / 0.7757, mean |ratio − 1| 0.2928 → 0.1796, a 39% cut.
- **Joint NLL**: composition 33.614 against independent 38.828.
- **OT tail**: predicted 256.9 single-OT games against 222 observed on 4,920 test
  team-games.

And the figures retired not by the split but by the **one-pass sweep**, which refits the
same four arms at full length on train alone:

- **The fitted ρ** read 0.1751 / 0.1285 / 0.1099 / 0.0839 graded and 0.1195 shared, a 2.09×
  spread. Every value rose by ~0.0016 on the re-run and the spread narrowed to 2.07×.
- **Gate A** extrapolated 12.8 h against an actual 20.9 h, a **1.63×** miss, at
  15.23 ms/row; max R̂ was 1.0113 over 11 fits.
- **Per-arm cost**: `binomial` 191.8 min, `betabinom` 506.3 min, of a 1,251.8-minute total.

### The fallback order in Gate A was mis-prioritised, and the diagnostics say so

The gate's abort message lists the fallbacks as "cut `binomial` from the ladder, shorten
select chains, subsample train team-games". **`binomial` is still the cheapest arm** —
**120.5** min of the sweep's **586.9**, or **20.5%** — so cutting it saves the least of any
arm while silently degrading audited claims to `no-such-row` skips and removing the result
that dispersion is load-bearing. The expensive arm is `betabinom_ot_graded` at **160.2 min
(27.3%)**, with `betabinom` at **156.0 min (26.6%)** and `betabinom_ot` at **150.2 min
(25.6%)**.

**⚠️ The one-pass sweep flattened the cost distribution, and that weakens the original
argument rather than overturning it.** In the two-pass run `betabinom` took 40.4% and
`binomial` 15.3% — a 2.6× spread that made "cut the cheapest arm" obviously wrong. At full
length on one split the four arms sit between 20.5% and 27.3%, a 1.33× spread, because the
old figure was dominated by the *test* refit on 631k rows at double the iterations. Cutting
`binomial` now saves a fifth of the run rather than a seventh; it is still the worst arm to
cut, but the margin is no longer dramatic.

The recorded order came from Gate A's own equal-cost-per-variant assumption, which the
diagnostics still refute, if less sharply. **`betabinom_ot` is the one arm that must never
be cut**: `run()` reads `models["betabinom_ot"]["val"].rho` after the whole sweep and before
any CSV is written, so losing it costs the entire run at the last step, and it is the only
shared-ρ twin the graded-vs-shared contrast can be made against. The honest fallback order
is: shorten select chains first, then subsample train team-games, and cut arms last.

### Operational notes from the full-window run

- **Each arm checkpoints as it completes** — its metric row, both diagnostics, and the
  pickled `alpha`/`beta`/`rho` draws plus scaler — to `outputs/checkpoints/stan_composition/`.
  `run()` writes its six CSVs only at the very end, so without this a crash at hour 20 loses
  everything; with it the PPC / joint-NLL / OT-tail tail can be re-driven from the pickles in
  minutes. `FITTED_VARIANTS` is also ordered decision-relevant-first, so the graded arm and
  its shared-ρ twin land in the first third of the run rather than the last.
- **Memory was never the constraint, and the reason is worth recording.** The parent
  process's RSS *falls* to ~70 MB during each fit: it blocks on the cmdstan child processes,
  which read their data from a JSON file, so macOS pages out the ~2.5 GB of frames it is not
  touching. That reads alarmingly like a leak on `ps` and is the opposite — it is the OS
  reclaiming idle pages, and they page back in when the fit returns.

---

## The per-(player, season) random effect — the head-level record, 2026-08-09

Driven by `docs/simulations-plan.md`'s item 3d; that document carries the design, the gates
and the verdict. This section is the head's own record of what changed in it.

### What the head gained

`composition_glm.stan` carries an **optional** `sigma_u * u_z[unit_idx]` term on the linear
predictor, indexed by **(player, season)** and disabled exactly by `U_n = 0`:

```
eta = logit_prior + alpha + X * beta + sigma_u * u_z[unit_idx]
```

`u_z` and `sigma_u` are zero-length when `U_n = 0`, so the disabled model is not "the same
model with a small coefficient" — it is literally the parameter space, priors and likelihood
that produced every figure above. This file's load-bearing property is unaffected. It is the
`S = 0` device from `betabinomial_glm.stan`, and this file already used the same trick for
`n_rho_par` on the binomial arm, so it is a house pattern rather than an import.

**The nesting is pinned as an identity, not as source text.** At `sigma_u = 0` the effect
model's log density exceeds the `U_n = 0` model's by exactly `−½Σz²`, which is the
`std_normal` prior on `z` and nothing else — so the likelihood, the stick-breaking offset,
the truncation term, the priors on `alpha`/`beta` and the dispersion term are all untouched.
`tests/test_stan_composition.py::test_U_n_zero_nests_exactly_inside_the_player_season_model`.

The index is the season and not the career: a player's minutes role is a property of the
season he is in, and a career-long effect would be absorbed by `logit_share_lag1` and the
offset. Non-centred by default, because rows per unit run median 57 but p10 11 and minimum 1
— the well-informed units would prefer centred and the one-game units funnel under it, so
divergences are the diagnostic and a centred arm is the first response.

### Why the metric is now conditional, and what it cost

`fit` chose `dense_e` unconditionally, on a measurement recorded above: at ~25 parameters the
dense adaptation is free, and it took the one-season probe from treedepth 8–9 and 645 s under
`diag_e` to treedepth 4 and 65 s. **That argument does not survive the random effect.** The
full training window carries **12,307** player-season units against 631,158 rows, so a dense
metric is a 12,332-square mass matrix — roughly 1.2 GB and a Cholesky of it per adaptation
window. The metric is therefore chosen by the data (`diag_e` whenever `U_n > 0`) and recorded
in the diagnostics, and the treedepth win is given back.

**Measured, on one training season of the pilot window (26,039 rows, 605 units, 200 + 200 ×
4 chains):** the shipped specification fits in **125 s** under `dense_e` with max R̂ 1.0172,
minimum ESS 400, 0 divergences and no treedepth saturation. The same rows with the effect on,
under `diag_e`, take **1,496 s** — **12.0×** — and do *not* converge at that budget: max R̂
**1.0948** against the 1.01 bar and minimum ESS **35** against 400.

**The diagnosis is mixing, not geometry, and the distinction decides what to do about it.**
Zero divergences with 17 treedepth-saturated draws and a step size of 0.00942 is a sampler
taking very long trajectories through a poorly conditioned diagonal metric — not one falling
into a funnel. A funnel would say "change the parameterization"; this says "the trajectories
are long and there is no cheap metric that shortens them", so 1,496 s is a **lower bound** on
a usable fit rather than an estimate of one.

**`ps_centered` tested the other half of that claim and it is a null.** On identical rows the
centred arm reads 2,510 s, R̂ 1.1067, min ESS 27, **0 divergences** and **212**
treedepth-saturated draws. The wall clock is contended and not a clean comparison; the
saturation count is not, because it is a property of the geometry rather than of the machine,
and it rises **12.5×**. Zero divergences in *both* coordinate systems, with saturation
sharply worse under centring, says the posterior is not funnelling either way — so the
parameterization is not the lever, and the plan's stated first response does not apply. The
premise it rested on does not survive this window either: it argued from "p10 11, minimum 1",
where the pilot's units run median 49 with only **1.8%** carrying a single row.

### Five configurations were tested and the SHIPPED one is the best of them

The hypothesis was that the graded `rho` and `sigma_u` compete for the same within-unit
overdispersion — a ridge between two parameters absorbing one quantity, which is what a
collapsed step size with no divergences looks like. **It is refuted, and so is every other
sampler-side lever.** All five arms are the same one-season frame, features and iterations;
only the dispersion, the parameterization or the metric varies.

| arm | dispersion | parameterization | max R̂ | min ESS | div | treedepth-sat | fitted σ_u |
|---|---|---|---|---|---|---|---|
| `base` | graded ×4 | — | **1.0172** | **400** | 0 | **0** | — |
| **`ps` (shipped spec)** | **graded ×4** | non-centred | **1.0948** | **35** | 0 | 17 | **0.4776** |
| `ps_centered` | graded ×4 | centred | 1.1067 | 27 | 0 | 212 | 0.4809 |
| `ps_shared_rho` | shared ×1 | non-centred | 1.1390 | 20 | 0 | 0 | 0.4986 |
| `ps_no_rho` | none | non-centred | 1.3289 | 11 | 0 | **791** | 0.5414 |

**Read the diagnostics, not the wall clocks.** The probes ran under varying contention — up
to three concurrent fits on 14 cores — so their timings are not comparable to each other.
R̂, ESS and treedepth saturation are properties of the geometry and are.

Three conclusions follow. **Removing `rho` makes it dramatically worse**, not better: R̂
1.3289 and 791 of 800 draws at max treedepth. `rho` is *helping* — strip the dispersion and
the binomial likelihood sharpens, each unit's `u_z` is pinned hard by its own rows, and the
geometry degrades. **Grading `rho` is better than sharing it** (R̂ 1.0948 against 1.1390),
so the four-bin dispersion the head already ships is doing useful work alongside the effect
rather than fighting it. And **the shipped configuration is the best random-effect arm on
every diagnostic that matters** — no parameterization, dispersion structure or metric tried
here improves on it.

### The dense metric is out too, and for the reason nobody checked first

A fourth attempt: `dense_e` is what took this head from treedepth 8-9 to treedepth 4 before
the effect existed, and 791-of-800 treedepth saturation is exactly the symptom it fixes. The
module refused it on every random-effect arm on **memory** grounds, and that reasoning is
wrong — the matrix is 3.1 MB at 605 units and 38.1 MB at 2,204, against 1.2 GB only at the
full window. So it was switched back on and measured.

**It is far worse, and the constraint that actually binds is estimability, not memory.** A
dense metric estimates a `P × P` covariance *from the warmup draws*, so it needs draws on the
order of the parameter count:

| | params | warmup | draws per parameter |
|---|---|---|---|
| effect-free head | 26 | 1,000 | **38.5** |
| one-season probe | 635 | 1,000 | 1.57 |
| pilot window | 2,234 | 1,000 | 0.45 |

Below one draw per parameter the adaptation is rank-deficient, CmdStan's regularization
shrinks it back toward diagonal, and the Cholesky is paid for nothing. Measured: the `ps` arm
under `dense_e` ran past **an hour** on rows `diag_e` finished in 25 minutes, and was killed
rather than finished. More warmup does not rescue it — 605 units would need ~12,700 warmup
draws to reach 20 per parameter.

**The original instinct was right and the reasoning behind it was wrong, and correcting only
the reasoning made things worse.** `choose_metric` now gates on both, with
`DENSE_DRAWS_PER_PARAM` as the binding term; at 1,000 warmup draws it admits ~50 parameters,
which is precisely the regime the effect-free head lives in and why `dense_e` was measured to
help there.

**So the cost is intrinsic to the parameter, not a configuration mistake.** Adding 605 unit
parameters to this likelihood makes a posterior NUTS walks slowly, and **four** attempts to
tune that away have failed — centring, sharing `rho`, dropping `rho`, and the dense metric.
What remains untried is mechanical rather than statistical: `reduce_sum` threading, against
one vectorized call plus **1,487** scalar truncation calls per gradient at the pilot window
with 4 chains on 14 cores.

**`sigma_u` moves in exactly the direction the mechanism predicts**, which is a check on the
whole ablation: 0.4776 with graded `rho`, 0.4986 with one shared `rho`, 0.5414 with none.
The less dispersion the likelihood carries, the more of it the random effect absorbs.

The cheaper lever is not in the sampler at all — see `docs/potential-to-dos.md`, where the
**fitting window** is worth 6.0× with evidence that it costs nothing.

### Two warmup rejection classes the probes surfaced, and only one is new

Both are non-fatal — warmup recovers — but they waste adaptation, and one of them is a live
trap for whoever revisits the centred arm.

**The beta-binomial shape underflow is PRE-EXISTING and not the random effect's doing.**
`beta_binomial_lpmf: First prior sample size parameter[k] is 0` fires ~28 times per fit at
line 212, and it fires **equally in the `base` arm** — 28 for `base` against 28 for `ps` on
the same probe. So it is a property of the shipped head, not of the new block, and it was
worth checking before blaming the parameter. The mechanism is the mirror of the one this
file's header already guards: `s * inv_logit(-eta)` was adopted so that `b` survives large
*positive* `eta`, and `a = s * inv_logit(eta)` is left unguarded against large *negative*
`eta`, where `inv_logit` underflows to exactly 0 at around −745. The rejections are proof
that warmup reaches that far down on some rows.

**The centred arm's prior is undefined at `sigma_u = 0`, and that one IS new.**
`normal_lpdf: Scale parameter is 0` at line 196 fires only in `ps_centered`, because
`vector<lower=0>[H_u] sigma_u` admits exactly 0 and `u_z ~ normal(0, sigma_u[1])` has no
density there. The non-centred branch is immune — `u_z ~ std_normal()` does not reference
`sigma_u` at all. It is recorded rather than fixed because `ps_centered` is a measured null
and is not in the shipped ladder; a lower bound of `1e-9` on `sigma_u` would close it if the
arm is ever revived.

The practical consequence for this head should not be softened: **a full-window fit with the
effect is not a same-day operation** — extrapolated by rows and units with this head's own
measured 1.63× Gate A correction, roughly 6.3 h per random-effect arm at the pilot window and
~38 h at the full one. That is why the ladder runs at the pilot window first and why the
injection retains a shippable fallback with σ estimated on `train`.

**One number came back for free and it is the best kind: a replication.** The under-converged
one-season fit puts `sigma_u` at **0.4776**, against **0.375** from the injection grid scored
on validation and **0.450** from the same grid scored on train. Three routes to the effect
size — a Stan parameter, a validation-scored grid and a train-scored grid — inside a band of
0.10. That is Gate P5 passing, and it is corroboration of the size rather than a value to
ship, because the chains had not mixed.

### What is NOT in this head's own sweep, and why

`make stan-composition` is unchanged — same four arms, same artifacts, same selected variant.
The new ladder is `make composition-effects`, writing
`outputs/predictions/composition_effects_*.csv`. Three reasons, and the first is a build
gate: `stan_composition_metrics.csv` is this head's record and `make docs-audit` re-derives
eleven quoted figures from it, so a partial run at a pilot window would have failed the gate
on bookkeeping rather than on a measurement. The incumbent is deliberately not refitted — its
posterior is on disk at `data/features/posteriors/<window>/composition.pkl` — and the arm
ordering is a separate decision from the full-window commitment, which is the path this head
already took once from Gate A to Gate E.

### The team-context block, and one departure from the specification

The head's 25 features are all properties of the player alone; a head whose entire job is
dividing a fixed team pot among teammates carried nothing about the teammates. The `team`
arms add five columns from `data/features/team_context_tierA.parquet` — `role_crowding`
(minutes-weighted archetype similarity, leave-one-out), `teammate_usage_max/sum/load` and
`n_teammates` — chosen tight rather than complete, because the deviation they are meant to
explain is only ~7.5% predictable and this is the most expensive fit in the project.

The block gets **one** `team_missing` indicator, not one per column, for the reason
`design_missing` exists: five identical flags are a degenerate subspace the sampler pays for
in treedepth. But it is a **second** indicator rather than a reuse of `design_missing`, which
is a deliberate departure from the plan's instruction and is measured rather than assumed:
the two mark different rows. `design_missing` flags a composition row with no
availability-design row at all; the team block additionally misses **4.14%** of pilot
training rows that *do* have a design, because `team_context_tierA.parquet` is built off the
season matrix's qualified frame. It covers **93.55%** of pilot training rows and **94.20%** of
validation rows. Folding the two together would leave that 4.14% imputed to the training mean
with nothing for `beta` to correct on.
