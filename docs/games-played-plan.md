# Games-Played Plan: the availability head as a Stan spell process

This is a planning doc, not a measurement report. Update it in place as pieces get built, the
way `docs/availability-plan.md` was.

It resolves the design question `docs/availability-plan.md` left open:

> Either **(a)** fit the game-level spell/hazard model hierarchically in Stan directly, which
> gets the per-game process and the posterior in one object but is **a much larger fit**, or
> **(b)** keep the season-level beta-binomial in Stan and calibrate the spell simulator to
> match its posterior predictive GP marginal per player. **(b) is the cheaper first step.**

**The answer is (a), and the premise that it is "a much larger fit" is wrong.** Every figure
below was measured during planning; each is marked with how it was obtained. Nothing here is
an artifact yet — the first implementation task is to make it one.

---

## Status: what exists, and what (b) actually is

`src/models/stan_availability.py` + `src/stan/betabinomial_glm.stan` ship the season-level
beta-binomial posterior, verified against the point MLE (21/21 coefficients inside the 95%
interval, 254 s, 0 divergences, held-out CRPS **10.795** games on 911 rows).

**The spell simulator does not exist — not one line.** An exhaustive search for
`spell|hazard|markov|onset|duration|simulator` across `src/`, `configs/`, `Makefile` and
`outputs/` finds only *descriptive* code: `features/availability.py::absence_spells` (a spell
extractor) and `eda/availability.py::serial_structure` (the 2-state diagnostic that
*falsifies* the geometric spell model). `dashboard/tabs/simulations.py:31` already reserves
the artifact slot `outputs/predictions/spell_process.csv` and renders
`st.warning("The simulator is not built.")`.

So option (b) is exactly half-built, and **the missing half is the same simulator (a) needs.**
The only difference is where its parameters come from — fitted to transition data (a), or
tuned to match a marginal (b). **(a) does not duplicate (b); it replaces a calibration step
with a fit.** That is what makes (a) the better buy even though it costs an overnight run and
(b) costs zero Stan hours.

---

## Why (a) is cheap: the chain collapses

**Every feature in `FEATURE_COLS` is constant within a player-season** (all lag-1/2/3 plus
age) — that is the prediction-time constraint, not a modelling choice. For a Markov chain with
*observed* states and season-constant transition probabilities,

```
L_i  =  h_i^{a_i} (1-h_i)^{b_i}  ·  r_i^{c_i} (1-r_i)^{d_i}
```

with `(a,b,c,d)` the four transition counts. Those counts are **sufficient statistics** — the
same algebraic collapse the count heads already use (`negbinomial_glm.stan:5-12`), not an
approximation.

**Verified numerically.** On a simulated chain (500 player-seasons × 82 games, 3
season-constant features), the game-level Bernoulli likelihood over 29,827 at-risk rows and
the collapsed binomial likelihood over 500 rows differ by a **constant free of β** (0.000e+00
at two different β), and their MLEs agree to **9.3e-08** — the same order as the recorded
7.0e-08 on the Poisson collapse. This check goes in the test suite verbatim.

Measured on `availability_panel.parquet` (1,314,238 rows):

| | full window | appearance window |
|---|---|---|
| transitions | 1,297,766 | 942,597 |
| **collapsed binomial rows** (2 per cell) | **32,944** | **32,944** |
| collapse ratio | **39.4×** | 28.6× |
| onsets / at-risk | 75,838 / 727,246 (h = 0.1043) | 68,530 / 719,938 (h = 0.0952) |
| recoveries / at-risk | 75,496 / 570,520 (r = 0.1323) | 68,530 / 222,659 (r = 0.3078) |
| absence spells | 82,804 | 68,530 |

**Two riders on the collapse.**

- **It is exact only where the hazard is memoryless.** The recovery side is not (below), so it
  does not reduce to two counts; it reduces to one row per **spell**, its own sufficient
  reduction — 82,804 rows, further collapsible to ~51,000 distinct `(cell, length, censoring
  class)` rows. The two pieces together are the whole likelihood.
- **The collapse handles right-censoring for free**, and this is a real advantage of the chain
  representation over a duration representation. The likelihood is a product over *observed*
  transitions; the terminal state contributes no factor. There is no censoring term to get
  wrong on the onset side.

**So the whole of (a) is ~85k likelihood rows against `stan-composition`'s 631k**, on simpler
likelihoods. Anchoring on `stan_availability` (10,361 rows × 21 params = 253.7 s) rather than
on the composition head (whose superlinearity comes from growing roster size under `dense_e`
and does not transfer to a fixed-width GLM), the arm ladder below is **60–90 min**, not the
"much larger fit" the question assumed. `dense_e` is available throughout — every head here is
under ~30 parameters.

### The onset hazard's decline is sorting, not state dependence — checked

The empirical onset hazard falls steeply with games-since-return: **0.3164** at a one-game
streak to **0.0265** past 41, an 11.9× fall. That looks like it should break the collapse. It
does not: a pure-frailty model with *zero* state dependence reproduces almost exactly that
shape, **11.8×**, purely by sorting — players with high onset hazards break their streaks
early, so long streaks are populated by low-hazard players. Bucket by bucket, simulated and
observed agree within 10% across streaks of 4–20. **The frailty the beta-binomial marginalizes
is precisely what generates this curve, so the collapse keeps it for free.**

What survives is small and localized: the observed hazard is **2.0× / 1.5× / 1.2×** the
frailty-only prediction at streaks of 1 / 2 / 3, gone by game 4. A post-return reintegration
effect — real, and concentrated in the high-hazard population the tail forecast depends on.
One arm on the ladder, not a reformulation.

### Rest days are a measured null — do not build the arm

Onset hazard by days since the team's previous game, full window:

| rest | b2b (1d) | 2d | 3d | 4d+ |
|---|---|---|---|---|
| at-risk | 164,831 | 415,602 | 106,939 | 39,657 |
| **onset hazard** | 0.1056 | 0.1025 | 0.1055 | **0.1142** |

Flat, and the *highest* hazard is at the longest rest. `docs/availability-plan.md` predicted
back-to-backs would drive the one-game-absence process; on the onset margin they do not.

**And stratifying to keep per-game covariates would break the frailty anyway.**
`betabinomial_glm.stan` draws an independent Beta *per row*, so splitting one player-season
into four rest-bucket rows gives it four independent frailties instead of one shared one. That
is a different model, not a refinement. Any within-cell stratification needs an explicit
per-cell latent — the thing this design exists to avoid.

---

## The dispersion budget: additive, and already over-supplied

`docs/availability-plan.md` reads clustering's **3.96×** against the GP marginal's **22.7×**
and concludes the remainder must come from between-player heterogeneity. Two corrections:

**1. The composition is additive, not multiplicative.** With a player-season frailty on the
mean and Markov clustering within, `Var(GP) = E_θ[Var(GP|θ)] + Var_θ(E[GP|θ])`, which with
`ρ ≡ Var(μ)/(μ̄(1−μ̄))` reduces exactly to

```
inflation  =  C  +  ρ · (n − C)
```

returning `1 + (n−1)ρ` at `C = 1` as it must. So "22.7 / 3.96 ⇒ frailty must supply 5.7×" is
the wrong reading.

**2. The 3.96× is measured on a different population and window than the 22.7×.**
`serial_structure` runs on the **appearance window over all players**; the 22.7× runs on
**established rotation players over the full window**. Matched to the same frame, the same
population's transition rates are `P(play|played) = 0.9443`, `P(play|missed) = 0.1333`, giving
**C = 9.58** — clustering supplies **42% of the budget, not 17%**. The required frailty is then

```
ρ = (22.70 − 9.581) / (82 − 9.581) = 0.181
```

against the incumbent's fitted **0.2757**. Stacking the incumbent's ρ on the measured
clustering predicts `9.581 + 0.2757 × 72.419 = 29.6`, a **30% overshoot**.

**There is no dispersion hole to fill. There is a surplus to avoid.** Monte Carlo agrees with
the algebra: featureless simulations span 9.4× (pooled hazards) to 36.1× (both frailties) and
bracket the target from both sides. So GP-marginal CRPS will be a wash at best — exactly as it
was for the Stan port, where the marginal was never the argument. **Decide in writing now that
this head ships on tail calibration and joint structure, with GP CRPS as a non-regression bar
rather than the win condition**, or the same argument gets lost twice.

---

## ⚠️ The frame: a plain full-window two-state chain does not work

**This corrects a decision taken during planning.** The intent — headline fit on the full
window, 30 seasons, target identical to the incumbent's `gp_share` — is preserved. The
mechanism is not.

Gate 0 is a numpy Monte Carlo with **per-cell empirical hazards**: the most generous
parameterization possible, since no covariate block can beat a cell's own observed rates. If
the process class fails there, it fails everywhere. On the 5,267 established-rotation
player-seasons:

| | mean | sd | overdispersion | P(GP<41) | P(GP<60) |
|---|---|---|---|---|---|
| **observed** | 0.759 | 0.243 | 26.40× | **0.109** | 0.318 |
| simulated, full window | 0.742 | 0.253 | 27.45× | **0.190** | 0.356 |

**The left tail is ~75% over-predicted** — and under a second reconstruction that keys the
denominator on `team_games` rather than summing panel rows, the miss is worse still (mean 7.4
points low, P(<41) 0.212). Both reconstructions agree on the direction and on the diagnosis.

**Why.** A waived player's cell has `r̂ ≈ 0`, so a two-state recurrent chain makes him
absorbing **from his first onset** rather than from the game he was actually cut. The
departure is relocated earlier in the season, dragging the mean down and the tail out. *A
departure is an absorbing hitting time, not a low recovery rate.* The same gate on the
appearance window — where departures are excluded by construction — recovers the mean to
within 0.004.

**The fix keeps all 30 seasons.** `status` only exists from 2006-07, so the three-state arm
cannot carry the headline; but the tenure split is identifiable **structurally on all 30
seasons** from `in_appearance_window`, which is already on the panel:

```
team_games  =  pre-tenure  +  tenure  +  post-tenure
GP          =  played games inside [first appearance, last appearance]
```

So the head becomes **entry index × exit index × within-tenure two-state chain**, which
reconstructs full-window `gp_share` exactly and confines the chain to where it demonstrably
works. Each factor answers a separate question — when he joined, when he stopped, how often he
missed while there. That is what `in_appearance_window` was built for. Measured on the 12,787
single-team player-seasons: **34.4%** have a delayed first appearance and **37.3%** a trailing
absence — so roughly a third of player-seasons have a tenure factor to fit at each end.

> The *shares* above reproduced across two independent reconstructions during planning; the
> mean lengths did not (19.7 / 17.7 games under one construction against 6.8 / 6.6 under
> another, on identical shares). The gap is a denominator convention and is unresolved. Do not
> quote a mean tenure length until `spell_classes` emits one.

### Multi-team player-seasons must be excluded from the fit

**12.2% of player-seasons are multi-team but carry 36.7% of full-window missed games**,
because the full window counts a traded player as rostered all year for *both* teams. The
panel's process runs per `(season, player, team)`; the incumbent's target runs per
`(player, season)`. Simulating per cell and summing double-counts those absences.

**Fit on single-team player-seasons only (12,787 of 14,569 = 87.8%), predict for everyone** —
every covariate is player-season level, so nothing blocks prediction — and record the exclusion
as a measured line rather than a silent filter. Evaluation stays on all 911 test rows so CRPS
stays comparable to 10.795. This is consistent with the standing scope note in `CLAUDE.md`
that mid-season churn is out of scope.

### What the 2006-07+ `status` arm actually buys

`not_rostered` is **98.5%-per-game persistent** — absorbing is the right idealization — and it
is a *tenure* fact, not an availability one: 58.6% of its rows precede the player's first
appearance, 40.2% follow his last, **1.1% are interior**.

Cross-tabulating "outside the appearance window" against "`not_rostered`" over all missed games
from 2006-07 gives **82.7% agreement**. The interesting cell is the **16.75%** that are missed
*outside* the appearance window while **still rostered** — season-ending and preseason injury.
That is the highest-value population in the whole head, and the structural proxy cannot see it.
**That, not "separating injury from roster mechanics" in the abstract, is what the arm buys.**

---

## The duration head: beta-geometric

Spell length conditional on a spell starting. The geometric null is falsified in both tails
(48.3% single-game observed against 30.8% geometric; 6.35% ≥10 against 3.65%). Five candidates
fitted by MLE to the 68,530 interior spells:

| model | params | nll | AIC | P(T=1) | P(T≥10) | P(T≥26) |
|---|---|---|---|---|---|---|
| **observed** | — | — | — | **0.4829** | **0.0635** | **0.0106** |
| **beta-geometric** | **2** | **126,171.9** | **252,347.7** | **0.4777** | 0.0590 | **0.0124** |
| mixture of 2 geometrics | 3 | 126,312.5 | 252,631.1 | 0.4654 | 0.0723 | 0.0089 |
| discrete Weibull | 2 | 126,631.1 | 253,266.3 | 0.5015 | 0.0660 | 0.0060 |
| negative binomial (shifted) | 2 | 127,223.2 | 254,450.5 | 0.5015 | 0.0719 | 0.0045 |
| geometric | 1 | 137,450.3 | 274,902.6 | 0.3078 | 0.0365 | 0.0001 |

**Beta-geometric wins AIC by 283 with one fewer parameter than the mixture**, and is the only
candidate that reaches the extreme tail. Fitted `a = 1.762, b = 1.926`.

> ⚠️ **A planning-session error worth recording, because it is easy to make again.** The
> mixture was initially argued for on the grounds that "a frailty cannot lift `P(T=1)` above
> the mean hazard, since `P(T=1) = E[q]`". The identity is true; the application was not. It
> held `E[q]` fixed at **0.3078**, the pooled recovery rate — but that is the *length-biased
> average over at-risk games*, not the first-game exit hazard. In a free fit `μ = a/(a+b)` is
> a parameter, and it lands at **0.4777**, right on the observed 0.4829. Fixing a free
> parameter at another quantity's value and concluding the model class is inadequate is the
> shape of the mistake.

Four further reasons it suits this repo specifically:

1. **It is the same device as the incumbent** — a geometric hazard with a Beta frailty
   integrated out analytically, which is `betabinomial_glm.stan`'s move one level down. No
   explicit latents, so `dense_e` stays viable.
2. **Closed-form survival**: `P(T ≥ t) = B(a, b+t−1)/B(a,b)`. Right-censoring is one branch,
   not machinery.
3. **Left truncation has an exact closed form too** (below), costing zero parameters.
4. Two parameters, one of which takes the covariates, so the design matrix is `FEATURE_COLS`
   unchanged.

### Right-censoring is 8.8% of spells and 28.6% of the missed games

Full window, 82,804 spells:

| class | spells | share | missed games | share | mean length | `not_rostered` share |
|---|---|---|---|---|---|---|
| interior (= appearance window) | 68,530 | 82.8% | 222,659 | 38.5% | 3.25 | 0.010 |
| left-truncated (in progress at game 1) | 6,966 | 8.4% | 189,668 | **32.8%** | 27.23 | 0.587 |
| right-censored (runs to season end) | 7,308 | 8.8% | 165,501 | **28.6%** | 22.65 | 0.462 |

**61.4% of full-window missed games sit in edge spells, and roughly half of those games are
`not_rostered`** — which is another statement of why the tenure factors must come out first.

Beta-geometric fitted three ways on the full window:

| treatment | a | b | E[T] | P(T≥26) |
|---|---|---|---|---|
| drop censored | 1.088 | 1.323 | 5.84 | 0.0377 |
| censored treated as complete | 0.917 | 1.252 | 7.68 | 0.0549 |
| **proper right-censoring** | **0.721** | **0.978** | **9.95** | **0.0741** |

Dropping censored spells understates `P(T≥26)` by **2.0×**; treating them as complete still
understates by 1.35×. Both fail silently.

> With proper censoring `a = 0.721 < 1`, so the fitted beta-geometric has **no finite mean**.
> The simulator is unaffected — a spell is truncated by the remaining schedule anyway — but
> `E[T]` must never be quoted from that fit, and `make docs-audit` will happily pin a
> meaningless number if allowed to.

**Left truncation has an exact answer.** By memorylessness the forward recurrence time of a
geometric is geometric with the same hazard, so under a Beta(a,b) frailty the *residual*
duration of an in-progress spell is beta-geometric with the frailty size-biased by `1/e` —
i.e. **Beta(a−1, b)**. Fit the in-progress offset freely and compare against `a − 1`:
agreement validates the renewal assumption, disagreement localizes it.

### `src/stan/betageometric_duration.stan`

```stan
// Beta-geometric spell duration: a geometric per-game exit hazard with a Beta frailty
// integrated out analytically — the same device betabinomial_glm.stan uses for the season
// count, one level down, so this file introduces no explicit latents either.
//
//   e ~ Beta(a, b),  T | e ~ Geometric(e)
//   P(T = t)  = B(a + 1, b + t - 1) / B(a, b)
//   P(T >= t) = B(a,     b + t - 1) / B(a, b)      <- right-censoring, one branch
//
// Parameterized as mu = a/(a+b) — which IS P(T = 1), the first-game exit hazard, so the
// linear predictor reads directly — and kappa = a + b. kappa -> infinity is the plain
// geometric; the data rejects it by 11,278 log-likelihood points at one extra parameter
// (68,530 interior spells).
data {
  int<lower=0> N;                            // collapsed spell rows
  int<lower=0> K;
  matrix[N, K] X;                            // standardized on TRAIN only
  array[N] int<lower=1> t;                   // observed length in games
  vector<lower=0, upper=1>[N] censored;      // 1 = ran to the end of the schedule
  vector<lower=0, upper=1>[N] truncated;     // 1 = already in progress at game 1
  vector<lower=0>[N] w;                      // multiplicity from the collapse
  real<lower=0> beta_scale;                  // 1/sqrt(2*l2), per prior_sd_for_l2
  real<lower=0> intercept_scale;
  real<lower=0> kappa_scale;
  int<lower=0> S;                            // 0 disables the year effect EXACTLY
  array[N] int<lower=0> season_idx;
  real<lower=0> year_sd_scale;
}
transformed data {
  int H = S > 0 ? 1 : 0;
  vector[N] tv = to_vector(t);
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> kappa;
  real open_shift;                           // logit offset for an in-progress spell
  vector[S] year_z;
  vector<lower=0>[H] sigma_year;
}
model {
  vector[N] eta = alpha + X * beta + open_shift * truncated;
  alpha      ~ normal(0, intercept_scale);
  beta       ~ normal(0, beta_scale);
  kappa      ~ normal(0, kappa_scale);       // half-normal on the <lower=0> support
  open_shift ~ normal(0, 1);
  if (S > 0) {
    year_z ~ std_normal();
    sigma_year ~ normal(0, year_sd_scale);
    eta += sigma_year[1] * year_z[season_idx];
  }
  // inv_logit(-eta) rather than 1 - inv_logit(eta), for the reason betabinomial_glm.stan
  // documents: the subtraction yields a shape parameter of exactly 0 by eta ~ 37 and
  // rejects the draw; this form holds to eta ~ 745.
  vector[N] a = kappa * inv_logit(eta);
  vector[N] b = kappa * inv_logit(-eta);
  // One expression covers both classes: a censored row contributes P(T >= t), an observed
  // row P(T = t), and they differ only by whether `a` is incremented.
  target += dot_product(w, lbeta(a + (1 - censored), b + tv - 1) - lbeta(a, b));
}
```

---

## Module structure

```
src/models/games_played.py        NEW — the point-MLE reference: collapse, spell classes,
                                  the closed-form duration fits, the simulator, Gate 0.
                                  NO Stan. This mirrors the repo's own pattern —
                                  availability.py preceded stan_availability.py, and the
                                  port is verified against it.
src/models/stan_games_played.py   NEW — the Stan arms. Imports evaluate / crps /
                                  pit_values / pit_table / build_design / split_seasons
                                  from models.availability and the simulator from
                                  games_played.py, so a metric difference cannot be a
                                  metric-implementation difference.
src/stan/betageometric_duration.stan  NEW — the ONLY new .stan file.
src/stan/betabinomial_glm.stan    UNCHANGED — reused verbatim for the onset head
                                  (y = onsets, n = at-risk transitions), the entry head
                                  and the exit head.
src/features/availability.py      EXTEND — `spell_classes(panel, window)` (absence_spells
                                  plus censored / truncated / tenure_position) and
                                  `collapse_transitions(panel)`. absence_spells unchanged.
src/eda/availability.py           EXTEND — `serial_structure` gains a rotation-subpopulation
                                  key so the matched-population C = 9.58 has an artifact.
                                  Existing appearance-window rows untouched, so no
                                  docs_audit claim moves.
src/models/season_total.py        EXTEND — one new GP treatment consuming the simulated pmf.
```

Gate 0 must not live in `stan_games_played.py`: it is a numpy job that has to run without a
CmdStan toolchain.

**Closest templates:** `src/models/stan_minutes.py` for structure (`variants` → class →
`sweep(train, val, test)` selecting on validation with test for confirmation → `run(cfg)`),
and `src/models/stan_composition.py` for the Gate-A cost probe (`probe_timing`,
`MAX_EXTRAPOLATED_HOURS`) and per-arm checkpointing (`_checkpoint`).

## The arm ladder

Each arm is fit → simulate → score, with a stop rule. **Select on validation; quote test for
confirmation only.** This repo has shipped one test-selected false positive
(`nonlinearity_ablation`, paired bootstrap P(Δ<0) = 99.7%, did not replicate) and caught a
second (`trend_x_role`, best two arms on test and worst two on validation) — **both in this
same head.**

| arm | frame | adds | stop rule |
|---|---|---|---|
| **G0** | rotation subpop, both windows | Gate 0: empirical-hazard Monte Carlo against 22.70 / 0.109 / 0.318. No Stan. | ~1 min. Ships as an artifact either way |
| **A0** | incumbent design | refit `BetaBinomialGLM` in-process so the floor is code-identical to the challenger's scoring path | CRPS must reproduce 10.795 |
| **A1** | single-team, appearance window | onset beta-binomial + beta-geometric duration, **tenure observed** (oracle entry/exit) | if A1 with oracle tenure loses to 10.795, the process class is wrong and A2–A4 are dead |
| **A2** | single-team, full window | + fitted entry and exit heads. The shippable 30-season head | must beat 10.795 to ship |
| **A3** | single-team, 2006-07+ | three-state, `status`-identified | ships if it beats A2 restricted to the same rows — report both denominators |
| **A4** | winner | covariates on the duration head; reintegration as an explicit-latent variant | optional, gated on A2/A3 clearing |
| **G** | winner | `season_total.py` wiring, against 435.1 MAE / 316.9 CRPS | the number that justifies the work |
| **B** | fallback | calibrate rather than fit (below) | always ships |

**The fallback is a closed form, not a fit.** Given the incumbent's per-player predictive mean
`μ*` and inflation `I*`, and a within-player clustering `C` from the measured transition rates,
invert the additive identity:

```
ρ   = (I* − C) / (n − C)      # residual frailty the chain must not double-count
ρ_M = (C − 1) / (C + 1)       # lag-1 autocorrelation reproducing C
r   =      μ*  · (1 − ρ_M)    # recovery hazard
h   = (1 − μ*) · (1 − ρ_M)    # onset hazard
```

Two lines, exact. It reproduces the incumbent's marginal *by construction* while getting the
game-level clustering right — and it is worth pinning as a test regardless of which arm wins,
since it is the invariant every arm should satisfy at its own fitted `C`.

## Gates

| Gate | Pass condition |
|---|---|
| **0** | the process class reproduces the GP marginal at its **ceiling** (per-cell empirical hazards). Full window fails; the tenure decomposition is the response |
| **A** | cost probe under budget — **treat as a lower bound**, it under-predicted `stan-composition` by 1.63× |
| **B** | onset head clears its no-fit floor (prior-season onset rate carried forward) |
| **C** | simulated spell shape matches `availability_profile.csv`, **and the hazard-by-k curve keeps falling past k = 20** |
| **D** | **the head gate** — GP CRPS ≤ **10.795**, PIT KS ≤ **0.096**, and the tail: `P(GP<41)` near **0.109**, `P(GP<60)` near **0.318** on the rotation subpopulation. The incumbent reads 15.0% / 34.8% against 11.8% / 36.9%, so the tail is where the win has to come from |
| **E** | season-total MAE ≤ **435.1** and CRPS ≤ **316.9** dk_pts through `season_total.py` |

**Failing D lands in option (b), not in the bin** — the same simulator, hazards calibrated by
the closed form above. Say so in the module docstring so the fallback is a documented branch
rather than a rescue.

## Artifacts, target, config

```
outputs/predictions/stan_games_played_metrics.csv       one row per (arm, group, metric)
outputs/predictions/stan_games_played_coefficients.csv  onset / duration / entry / exit
outputs/predictions/stan_games_played_diagnostics.csv   diagnostics_frame(), one row per fit
outputs/predictions/stan_games_played_spells.csv        the three spell classes per window,
                                                        with the censoring-bias table
outputs/predictions/stan_games_played_gate.csv          Gate 0 — sim vs obs, per window
outputs/predictions/stan_games_played_pit.csv
outputs/predictions/stan_games_played_predictions.csv
outputs/predictions/spell_process.csv                   the dashboard's registered slot
```

**`spell_process.csv` is not optional.** `dashboard/tabs/simulations.py::CHAIN` hard-codes
that path as its build check, and `make dashboard-audit`'s orphan check will otherwise flag all
six `stan_games_played_*` families as unreachable — trading one finding for six. Emit it *and*
repoint the `CHAIN` row at `stan_games_played_metrics.csv`.

```makefile
stan-games-played:
	$(PYTHON) -m src.models.stan_games_played
```

Add to `.PHONY` beside `stan-availability`, and to the `stan` aggregate **after** it — this
head imports the incumbent as its floor, the same ordering constraint `stan-composition` has
on `stan-minutes`. Hold it out of the aggregate until gate D passes, matching how
`stan-substitution` and `season-terms` are held out.

```yaml
stan:
  games_played:
    # Single-team player-seasons only. 12.2% of player-seasons are multi-team but carry
    # 36.7% of full-window missed games, because a traded player is "rostered all year"
    # for BOTH teams — simulating per cell and summing double-counts his absences.
    # Prediction is unaffected: every covariate is player-season level.
    single_team_only: true
    # Ignoring censoring understates P(T >= 26) by 2.0x. See this doc.
    handle_censoring: true
    # Half-normal scale on the beta-geometric concentration a + b. Fitted values land near
    # 3.7 on interior spells, so this is weakly informative by an order of magnitude.
    kappa_scale: 25.0
    # Simulated seasons per posterior draw; 400 x 50 puts the GP pmf's Monte Carlo error
    # well under 0.01 games.
    sim_seasons: 50
    arms: [gate, floor, within_tenure, full_window, three_state]
```

## Tests — `tests/test_games_played.py`, plain assert, synthetic builders

1. **`test_collapse_reproduces_the_game_level_likelihood`** — explicit product over
   transitions vs `h^a(1−h)^b r^c(1−r)^d`, equal to 1e-12. *This is the test that makes the
   whole design legitimate; write it first.*
2. `test_collapse_is_conditional_on_the_initial_state` — the collapsed likelihood is invariant
   to `s0`, so the initial state needs its own head. Guards the one place the collapse quietly
   loses information.
3. `test_beta_geometric_pmf_sums_to_one_and_survival_is_its_own_tail`.
4. `test_right_censoring_recovers_a_known_shape_where_ignoring_it_does_not` — assert the
   *direction* of the naive fit's bias.
5. `test_left_truncated_residual_is_the_size_biased_beta_geometric` — pins the `Beta(a−1, b)`
   identity that `open_shift` is measured against.
6. `test_spell_classes_partition_the_missed_games` — interior + truncated + censored ==
   `missed_games`, per cell. The one arithmetic error that would silently move every figure in
   the censoring table.
7. `test_departure_is_absorbing_in_the_simulator`.
8. `test_gp_pmf_sums_to_one_and_is_supported_on_0_to_team_games`.
9. `test_collapsed_multiplicity_weights_match_the_uncollapsed_likelihood`.
10. `test_closed_form_calibration_inverts_the_variance_identity` — round-trips
    `inflation = C + ρ(n−C)`, and guards against the multiplicative misreading returning.
11. `test_two_state_chain_reproduces_a_known_transition_matrix`.
12. `test_stan_beta_geometric_recovers_known_parameters` — small synthetic Stan fit, mirroring
    `tests/test_stan_heads.py`. Marked slow.
13. `test_multi_team_player_seasons_are_excluded_from_the_fit_but_not_the_evaluation`.
14. `test_frailty_is_drawn_per_player` — `rng.beta(a, b)` without `size=` returns a **scalar**,
    and that bug silently gave a whole simulated population one shared hazard during planning.
    It does not announce itself; assert the drawn vector's length and variance.

## Verification

```bash
.venv/bin/pytest tests/test_games_played.py tests/test_availability.py \
                 tests/test_availability_model.py tests/test_stan_heads.py
make stan-games-played          # 60-90 min estimated; Gate A's estimate is a lower bound
make season-total               # gate E, in dk_pts
make docs-audit                 # MUST pass — it gates prose against artifacts
make dashboard-audit            # spell_process.csv fills a reserved pending marker
```

## Doc and registry obligations

- **`make docs-audit` is a gate and a pytest test.** `3.96`, `0.905`, `0.308`, `0.597`,
  `0.483`, `0.0635` are already claimed against `availability_profile.csv` at the
  **appearance window**, from *both* `availability-plan.md` and `CLAUDE.md`. Those
  measurements stay correct; only the interpretation around them changes. **Keep the quoted
  strings in place** or the presence check fails — rewrite the surrounding prose, and add the
  matched-population `C = 9.58` as a *new* artifact row with a new claim rather than editing
  the old one. The superseded "clustering explains roughly a sixth" reading is a textbook
  `Claim(historical=True)`: it is a reversal, and the reversals are the most useful thing in
  that doc.
- `docs/availability-plan.md` — resolve the open design question at line 813 in place.
- `CLAUDE.md` — the established-facts entry. `dashboard/decisions.py` — a registry entry under
  topic `simulations`, and repoint `dashboard/tabs/simulations.py::CHAIN`'s spell-process row.

## The three most likely failure modes

1. **The full-window two-state process cannot reproduce `gp_share`.** Not a risk — a measured
   failure, and the reason this doc's frame changed. *Cheap check: Gate 0, ~1 minute of numpy,
   before any Stan.*
2. **The head wins nothing on GP-marginal CRPS, discovered after the sweep instead of before.**
   The dispersion budget is already over-supplied (`C + ρ(n−C)` overshoots by 30% at the
   incumbent's ρ). *Cheap check: evaluate that identity at the fitted `C` before scoring
   anything.* Ship on the tail and the joint; keep GP CRPS as a non-regression bar.
3. **The duration head fits roster mechanics and gets called an injury model.** On the full
   window 61.4% of missed games sit in edge spells and ~half of those games are
   `not_rostered`. *Cheap check: emit the spell-class table as an artifact before fitting, and
   report the `not_rostered` share of the fitted population beside every duration
   coefficient.*
