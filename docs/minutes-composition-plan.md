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
  its own floor by only +0.041 R², so the marginal headroom is known to be small.
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

## Results — full window, 2026-08-04, `make stan-composition`

**Verdict: Gates A–E all pass. Gate E is taken — the head is fitted on all 30 seasons and
is now part of `make stan`.** The composition clears its floor and beats the incumbent on
the incumbent's own metric, while delivering the team-sum exactness that was the point of
building it. Every figure below is from `outputs/predictions/stan_composition_*.csv` and
audited by `make docs-audit`.

Full window 1996-97 →: train 631,158 rows / 61,252 team-games (1996-97 → 2021-22),
val 52,295 / 4,920 (2022-23 + 2023-24), test 52,957 / 4,920 (2024-25 + 2025-26).

| variant | val CRPS | test CRPS | test R² | test PIT KS | selected |
|---|---|---|---|---|---|
| `carry_forward` (floor) | 4.6776 | 4.8576 | 0.3678 | 0.0354 | |
| `binomial` | 4.9394 | 4.9732 | 0.4259 | **0.1948** | fails the floor |
| `betabinom` | 4.5422 | 4.5893 | 0.4247 | 0.0405 | |
| `betabinom_ot` | 4.5430 | 4.5848 | 0.4255 | 0.0414 | |
| **`betabinom_ot_graded`** | **4.4926** | **4.5592** | 0.4244 | 0.0393 | **✓** |
| `independent_comparator` | 4.7842 | 4.9140 | 0.3301 | 0.0769 | (incumbent) |

**The `independent_comparator` row is the control, and it came back identical.** It trains
on `minutes_build_design(cfg)` filtered by season and never on the composition window, and
it scores the same 52,957 rows either way — so it was predicted to be invariant to the
window change, and it reproduced **4.7842 / 4.9140 / 0.3301 / 0.0769** exactly. Its
team-sum error (36.87), the two *observed* starter shares and the whole OT-tail block
reproduced exactly too. That is what licenses reading everything else as a window effect
rather than a frame change.

- **Gate A — timing, and it is the one gate that did NOT behave.** The probe fit 26,039
  rows in 150 s and extrapolated the sweep to **12.8 h**; the sweep actually took
  **20.9 h** of sampler time, **1.63×** the estimate. The cause is that per-row cost is
  **superlinear in rows**: 5.75 ms/row on the 26k-row probe against **15.23 ms/row** on the
  631k-row `betabinom` validation fit. More data sharpens the posterior, which shrinks the
  step size, which buys more leapfrog steps per iteration on top of a per-gradient cost that
  is itself linear. The pilot's linear model was accurate to 1.2% over a 16× extrapolation
  and is off by 63% over a 24× one, so **treat Gate A as a lower bound, not an estimate**,
  and keep `max_extrapolated_hours` well above the number you would accept.
- **Gate B — convergence.** Max R̂ **1.0113**, **0** divergences over 11 fits. The one fit
  over the 1.01 bar is `binomial/val` — the arm that fails anyway — and every composition
  fit that matters converged at R̂ ≤ 1.0065.
- **Gate C — the floor.** The selected variant beats `carry_forward` by **−0.2985** minutes
  of test CRPS (−6.1%), replicating validation (−0.1851). And the `binomial` arm is again
  **worse than the floor** (4.9732 against 4.8576) with PIT KS **0.1948** against 0.0354:
  the pure stick-breaking decomposition is far too tight, exactly as ρ_game = 4.65×
  binomial predicted. The dispersion is not an optional refinement — it is the difference
  between a model and a failure, the NB-vs-Poisson lesson again.
- **Gate D — the incumbent.** **−0.3548** minutes of test CRPS against the incumbent
  (4.5592 vs 4.9140, **−7.2%**) — not the expected wash — plus the capability gap: the
  comparator's mean absolute team-sum error is **36.87** minutes per team-game against the
  composition's exact **0** on every draw of every game. The comparator also carries a
  −1.2856 minute bias where the composition's is 0 by construction.
- **Gate E — the full window itself.** Taken. `stan-composition` joins `make stan`, after
  `stan-minutes`, which it imports from and measures itself against.
- The OT interaction remains real but tiny, and at full window it **loses** validation to
  plain `betabinom` by 0.0008 CRPS while winning test by 0.0045. Treat it as a refinement,
  not a driver; it survives in the shipped arm because the graded variant is built on it.

### What the full window changed against the pilot

The ordering is unchanged and every gate still passes, but three things moved enough to
restate rather than carry forward.

| | pilot (2018-19 →) | full window (1996-97 →) |
|---|---|---|
| selected test CRPS | 4.5078 | **4.5592** |
| vs the floor | −0.312 | **−0.2985** |
| vs the incumbent | −0.406 | **−0.3548** |
| graded ρ, fringe → star | 0.1480 / 0.1125 / 0.0874 / 0.0613 | **0.1751 / 0.1285 / 0.1099 / 0.0839** |
| shared ρ | 0.0970 | **0.1195** |
| ρ spread | 2.41× | **2.09×** |
| mean \|variance ratio − 1\|, graded | 0.1055 | **0.1796** |

- **Every ρ is larger and the spread is smaller.** Fitting 26 seasons instead of four raises
  the fitted dispersion at every tier (fringe 0.148 → 0.175, star 0.061 → 0.084) and
  compresses the fringe-to-star ratio from 2.41× to **2.09×**. Role grading is still real
  and still monotone; it is less extreme once the model has to cover three decades of
  rotation practice rather than one.
- **The absolute win is smaller and still decisive.** −0.3548 against the incumbent instead
  of −0.406, and −0.2985 against the floor instead of −0.312. Both shrank by roughly a
  tenth; neither is close to a wash.
- **⚠️ The graded arm's calibration gain is real but weaker, and the pilot oversold it.**
  Mean |variance ratio − 1| falls **0.2928 → 0.1796**, a **39%** cut rather than the pilot's
  59%, and the star tier lands at **0.7757** rather than the pilot's near-exact 0.988.

**Posterior predictive checks** (test split, selected variant):

| tier | shared ρ | graded ρ |
|---|---|---|
| q1 fringe | **1.3466** | **1.0845** |
| q2 | 0.8089 | 0.7687 |
| q3 | 0.7691 | 0.8217 |
| q4 star | **0.5973** | **0.7757** |
| mean \|ratio − 1\| | **0.2928** | **0.1796** |

- The two extremes still improve substantially — fringe 1.35 → 1.08 and star 0.60 → 0.78 —
  and **q2 still gets slightly worse** (0.809 → 0.769), which is the same structural point
  the pilot recorded: the fitted ρ is the dispersion of a **sequential step** while the
  ratio is measured on a player's **marginal** minutes, and because the order is prior-share
  descending, a low-share player breaks his stick last and inherits the accumulated
  remainder variation from everyone ahead of him. Grading step dispersion does not map
  one-to-one onto marginal variance by tier.
- **Now three of four tiers sit below 1**, where the pilot had two. The model is
  systematically a little over-dispersed in aggregate at full window — worth a look before
  the simulator consumes it, and a better-posed target than chasing q2 specifically.
- **Starter share in OT** — observed: top-5 ordered players take **0.5882** of team minutes
  in regulation and **0.6314** in OT (+4.3 pp). The composition simulates **0.5998** →
  **0.6464**, reproducing the shift (+4.7 pp) with a +1.2 pp level overshoot; the
  independent comparator compresses it (0.5973 → 0.6346).
- **Joint per-team-game NLL** (plug-in, non-bijection caveat) — composition **33.614**
  against independent **38.828** on test. Contrast, not a headline.

**OT tail** — unchanged, as predicted: `fit_ot_tail` receives 1996-97 → 2021-22 in both
windows. Fit on 30,626 regular-season training games: p_any = **0.0608**, p_more =
**0.1408**. Held out on 4,920 games: predicted **256.9** / 36.2 / 5.9 games at 1/2/3+ OT
against observed **222** / 30 / 0 — the two-parameter form holds but overpredicts OT by
~16% on recent seasons, a mild era decline in OT rate. `sample_game_length` is the
simulator's game-length draw and carries that caveat with it.

### The fallback order in Gate A was mis-prioritised, and the diagnostics say so

The gate's abort message lists the fallbacks as "cut `binomial` from the ladder, shorten
select chains, subsample train team-games". **`binomial` is the *cheapest* arm** — 191.8 min
of the sweep's 1,251.8, or **15.3%** — so cutting it saves the least of any arm while
silently degrading five audited claims to `no-such-row` skips and removing the result that
dispersion is load-bearing. The expensive arm is `betabinom` at **506.3 min (40.4%)**, with
`betabinom_ot` at 21.4% and `betabinom_ot_graded` at 22.9%.

The recorded order came from Gate A's own equal-cost-per-variant assumption, which the
diagnostics refute. **`betabinom_ot` is the one arm that must never be cut**: `run()` reads
`models["betabinom_ot"]["test"].rho` after the whole sweep and before any CSV is written, so
losing it costs the entire run at the last step, and it is the only shared-ρ twin the
graded-vs-shared contrast can be made against. The honest fallback order is: shorten select
chains first, then subsample train team-games, and cut arms last.

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
