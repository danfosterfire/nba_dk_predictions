# Minutes Composition Plan: The Team-Game Allocation as a Decomposed Multinomial

This is a planning doc in the house pattern — direction first, results filled in as the
pilot lands. It executes the "team-game composition alternative" costed in
`docs/predictions-plan.md`, as a **pilot gated on measurement**, not a committed
replacement for the season-collapsed minutes head (`make stan-minutes`).

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
An `independent_comparator` row — the season head refit on the pilot window, plus
independent `BetaBinom(game_length, μ, ρ_game)` draws per player-game — is reported in
the same table but excluded from selection: it is the incumbent, not a variant.

## Pilot design and gates

Frame built over all seasons (lags and expanding rookie priors need the history), fitted
from `stan.composition.first_season` = **2018-19**. In order:

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
  `docs/predictions-plan.md` already costed.

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

## Results — pilot run 2026-07-31, `make stan-composition`

**Verdict: Gates A–D all pass.** The composition clears its floor and beats the
incumbent on the incumbent's own metric — while delivering the team-sum exactness that
was the point of building it. Every figure below is from
`outputs/predictions/stan_composition_*.csv` and audited by `make docs-audit`.

Pilot window 2018-19 →: train 97,587 rows / 9,198 team-games (2018-19 → 2021-22),
val 52,295 / 4,920 (2022-23 + 2023-24), test 52,957 / 4,920 (2024-25 + 2025-26).

| variant | val CRPS | test CRPS | test R² | test PIT KS | selected |
|---|---|---|---|---|---|
| `carry_forward` (floor) | 4.6331 | 4.8194 | 0.3679 | 0.0178 | |
| `binomial` | 4.9345 | 4.9429 | 0.4307 | **0.1942** | fails the floor |
| `betabinom` | 4.5107 | 4.5360 | 0.4306 | 0.0201 | |
| **`betabinom_ot`** | **4.5101** | **4.5322** | 0.4310 | 0.0199 | **✓** |
| `independent_comparator` | 4.7842 | 4.9140 | 0.3301 | 0.0769 | (incumbent) |

- **Gate A** — probe fit of 26,039 rows in 141 s → sweep extrapolated to **1.8** h;
  the actual sampler total came in at **116.0** min over 9 fits.
- **Gate B** — max R̂ **1.0093** (a comparator spline fit; every composition fit is at
  or below 1.0053), **0** divergences anywhere, 0 treedepth-saturated draws on the
  composition fits.
- **Gate C** — the selected variant beats the floor by **−0.287** minutes of test CRPS
  (−6.0%), replicating validation (−0.123). And the `binomial` arm is **worse than the
  floor** (4.9429 against 4.8194) with PIT KS 0.1942 against the floor's 0.0178: the
  pure stick-breaking decomposition is far too tight, exactly as ρ_game = 4.65×
  binomial predicted. The dispersion is not an optional refinement — it is the
  difference between a model and a failure, the NB-vs-Poisson lesson again.
- **Gate D** — **−0.382** minutes of test CRPS against the incumbent (4.5322 vs
  4.9140, −7.8%) — not the expected wash, an outright win on the incumbent's own
  marginal metric — plus the capability gap: the comparator's mean absolute team-sum
  error is **36.87** minutes per team-game against the composition's exact **0** on
  every draw of every game.
- The OT interaction is real but tiny: `betabinom_ot` won validation by 0.0006 CRPS.
  Treat it as a refinement, not a driver.

**Posterior predictive checks** (test split, selected variant):

- **Starter share in OT** — observed: top-5 ordered players take **0.5882** of team
  minutes in regulation and **0.6314** in OT (+4.3 pp). The composition simulates
  **0.6001** → **0.6411** — it reproduces the shift (+4.1 pp) with a +1.2 pp level
  overshoot; the independent comparator compresses the shift (0.5973 → 0.6346).
- **Dispersion by prior-share tier** — realized-over-simulated variance ratio runs
  **1.59** (q1 fringe) / 0.96 / 0.89 / **0.70** (q4 stars): one shared ρ is too tight
  for fringe players and too wide for stars, precisely the limitation named before
  the run. A share-graded ρ is the first candidate improvement.
- **Joint per-team-game NLL** (plug-in, non-bijection caveat) — composition **33.64**
  against independent **38.83** on test. Contrast, not a headline.

**OT tail** — fit on 30,626 regular-season training games: p_any = **0.0608**,
p_more = **0.1408**. Held out on 4,920 games (2022-23 → 2025-26): predicted
**256.9** / 36.2 / 5.9 games at 1/2/3+ OT against observed **222** / 30 / 0 — the
two-parameter form holds but overpredicts OT by ~16% on recent seasons, a mild era
decline in OT rate. Fine for a game-length draw; one more entry for the season-effects
ledger in `docs/predictions-plan.md`.

**Gate E (full window) is open but not taken.** Measured cost is 5.41 ms/row at
select iters, which extrapolates the 736k-row full-window sweep to ~8–10 h — inside
the 6–16 h envelope `docs/predictions-plan.md` costed. Recommended order: the
share-graded ρ variant first (it addresses the one measured miscalibration), Gate E
second, then wiring `sample_game_length` + the composition simulator into the season
simulator when that exists.
