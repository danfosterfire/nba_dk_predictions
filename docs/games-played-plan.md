# Games-Played Plan: the availability head as a Stan spell process

This is a planning doc, not a measurement report. Update it in place as pieces get built, the
way `docs/availability-plan.md` was.

> ## ✅ Built 2026-08-05 — `make games-played` and `make stan-games-played`
>
> `src/models/games_played.py` is the numpy reference (the collapse, the spell classes, the
> closed-form duration fits, the simulator and Gate 0 — **no Stan**, so it runs without a
> CmdStan toolchain and rejects a process class in seconds rather than in NUTS hours).
> `src/models/stan_games_played.py` fits the arms. `src/stan/betageometric_duration.stan` is
> the only new `.stan` file; the onset, entry and exit heads reuse `betabinomial_glm.stan`
> verbatim, which is the factorization argument as code for the fourth, fifth and sixth time.
>
> **Every planning figure that is a fitted quantity reproduced exactly** — the collapse ratio
> 39.4×, the three spell-class counts and their missed-game shares, the `not_rostered` shares
> 0.010 / 0.587 / 0.462, the censored beta-geometric's `a = 0.721, b = 0.978`, the interior
> fit's `a = 1.762, b = 1.926` and its 126,171.9 nll, and the matched-population **C = 9.58**.
> Three *derived* figures did not, and are corrected in place below with the reason.
>
> ## ⚠️ The first verdict was decided on TEST and is withdrawn. Re-decided on VALIDATION:
>
> **No arm clears Gate D. The incumbent stands.** But the reason is more interesting than a
> loss, and it is a defect in the gate rather than in one of the arms — see the hybrid row.
>
> `make stan-games-played`, validation only. **13 fits** (was 25 before the test side was
> removed), 0 divergences.
>
> | arm | val CRPS | its floor | vs floor | PIT KS | implied od | tail error |
> |---|---|---|---|---|---|---|
> | `floor` (the incumbent) | **10.0057** | 10.0057 | — | 0.0939 | 23.7251 | **0.0406** |
> | *`within_tenure`* (oracle) | **7.23495** | 10.0992 | **−2.8642** | 0.1259 | 11.4603 | 0.0092 |
> | `full_window` | 10.2797 | 10.0057 | +0.2739 | 0.0755 | 22.1686 | 0.0496 |
> | `three_state` | 10.3466 | 10.0501 | +0.2965 | 0.1204 | 20.3687 | **0.0043** |
> | `duration_covariates` | 10.1676 | 10.0057 | +0.1619 | **0.0672** | 22.1055 | 0.0535 |
> | `calibrated_fallback` | 10.0207 | 10.0057 | +0.0149 | 0.1017 | 23.7251 | 0.0440 |
> | **`hybrid`** | **10.0057** | 10.0057 | **0.0000** | 0.0939 | 23.7251 | **0.0406** |
>
> ### Gate D cannot pass the hybrid, and that is a flaw in Gate D
>
> The hybrid draws its games-played count *from* the incumbent's pmf, so its marginal is not
> approximated but **identical** — 10.0057 against 10.0057, PIT 0.0939 against 0.0939, tail
> 0.0406 against 0.0406, to every decimal. Gate D asks three questions and all three are
> about the marginal; its tail test is a strict inequality; so a marginal-neutral arm ties
> every bar and fails on the tie. **Gate D is a marginal gate being asked to judge an arm
> whose entire contribution is orthogonal to the marginal.** That is a category error in the
> instrument, not a verdict on the arm.
>
> The other three fail on their merits: `duration_covariates` loses CRPS by +0.1619 (a
> paired bootstrap on 883 rows puts it at 95% CI [+0.0737, +0.2393], P(better) = 0.1%),
> `calibrated_fallback` loses CRPS *and* PIT, and `three_state` is worst on CRPS while
> posting the best tail of any arm (0.0043) — a reminder that the tail alone is a noisy
> criterion on 359 rotation rows.
>
> ### The spell shape is where the arms actually differ
>
> Absence-spell lengths simulated on the validation rows, against those player-seasons' own
> observed spells:
>
> | arm | P(T=1) | P(≥10) | P(≥26) | mean spell | mean abs rel error |
> |---|---|---|---|---|---|
> | observed | 0.4924 | 0.0581 | 0.0079 | 3.0857 | — |
> | **hybrid** | 0.4904 | 0.0572 | 0.0202 | 3.8220 | **0.5310** |
> | `full_window` | 0.4556 | 0.1018 | 0.0409 | 4.8658 | 1.6746 |
> | `duration_covariates` | 0.4471 | 0.1063 | 0.0416 | 4.9746 | 1.7373 |
> | `calibrated_fallback` | 0.1171 | 0.3544 | 0.0799 | 10.0594 | 5.0082 |
>
> **The hybrid is 3.3x better than the best fitted arm and 9.4x better than the fallback**,
> and it is within 2% on both the single-game share and the ten-plus-game share — the
> statistic a Round 1 knockout actually turns on. Its weakness is the extreme tail, +157% on
> month-long absences. The fallback is disqualifying for this purpose: a constant recovery
> hazard gives a mean spell of **10.06 games against an observed 3.09**.
>
> So the honest summary is that **the hybrid costs exactly nothing on the marginal and is
> far the best on shape** — but it cannot be *selected* by Gate D, because Gate D cannot see
> the axis it wins on. Whether to ship it is a judgement about what the simulator needs, made
> with both numbers in view, not a gate outcome.
>
> ### The two results that survive from the fitted arms
>
> 1. **The process class is right; the tenure is the bottleneck.** The oracle-tenure arm
>    scores **7.23495** against the incumbent's 10.0057 — 28% better, on validation. Given the
>    observed tenure the within-tenure chain is far better than the season-level
>    beta-binomial, and all of that is destroyed by predicting entry and exit from preseason
>    covariates. That is mid-season roster churn, already out of scope.
> 2. **`duration_covariates` has the best PIT** (0.0672 against 0.0939): its predictive
>    *shape* is better calibrated than the incumbent's even though its location is worse.
>
> ### Gates
>
> | gate | verdict | figure |
> |---|---|---|
> | **0** | ✅ | plain chain **z = +5.29** (rejected); tenure decomposition **z = +0.93** |
> | **A** | ✅ | 0.5 h linear → **0.78 h** corrected against a 6 h budget |
> | **B** | ✅ | onset head **−0.330662** per at-risk transition against the floor's **−0.350239** (**+0.019577**), shrinkage `k` = **48.1** toward a league rate of **0.0741** |
> | **C** | ✅ | the simulated hazard curve keeps falling past a streak of 20 |
> | **D** | ❌ | no arm clears. `duration_covariates` **10.1676** / 0.0672 / 0.0535; `calibrated_fallback` **10.0207** / **0.1017** / 0.0440; `hybrid` 10.0057 / 0.0939 / 0.0406 against observed **0.1003** / **0.3259** |
> | **E** | ❌ | **re-run 2026-08-12 and the verdict does not move.** `spell_process` scores MAE **406.65** and CRPS **291.63** against the incumbent's **400.46** / **287.26** — worse on both, by **+6.19** and **+4.37** dk_pts (the 2026-08-05 reading was 406.80 / 291.80, +6.34 / +4.55) |
>
> Gate D's per-arm tail predictions are `P(GP<41)` / `P(GP<60)` of **0.1597** / **0.3734**
> (`duration_covariates`), **0.1499** / **0.3643** (`calibrated_fallback`) and **0.1553** /
> **0.3521** (`hybrid`).
>
> ### ⚠️ Gate E passed on TEST by 0.03 dk_pts and FAILS on validation by 6.19
>
> The recorded verdict was ✅ at **435.1053 MAE against a 435.1352 bar** — a margin of
> **0.0299 dk_pts on a ~435 dk_pts quantity**, i.e. seven parts in a hundred thousand. On
> validation the same treatment, the same rate model and the same six-row ladder put it
> **6.19 dk_pts the wrong side**, and it loses CRPS and bias too (−15.87 against −3.06).
> This is the third gate in this head to reverse when moved off the test split, and it is
> the cleanest example of why: a bar cleared by 0.03 was never evidence of anything, and
> reading it on the split that is not allowed to decide is what made it look like it was.
> The recorded figures remain true of the pre-lock artifact and are preserved below.
>
> **The bars are no longer written down.** `season_total.gate_e` reads the incumbent's own
> row out of whichever table it is scoring, so the gate cannot be passed by refreshing the
> constant it is compared against — the same fix `stan_games_played._gate_d` took.
>
> Gate E failing is *consistent* with Gate D rather than new information: `spell_process`
> composes the `duration_covariates` pmf, which already loses games-level CRPS by +0.1619,
> and 6.34 dk_pts is roughly that loss times a ~40 dk_pts-per-game rate. **It does not
> speak to the hybrid**, whose games-played pmf is the incumbent's by construction and
> which would therefore tie Gate E to every decimal — the same category error Gate D has,
> one level down, because a season *total* is as marginal a quantity as a season *count*.

---

## Status: what exists, and what (b) actually is

`src/models/stan_availability.py` + `src/stan/betabinomial_glm.stan` ship the season-level
beta-binomial posterior — a two-component mixture since 2026-08-12 — verified against the
point MLE of its own likelihood (35/35 terms inside the 95% interval, 366 s, 0 divergences,
validation CRPS **9.8239** games on 883 rows; the recorded 24/24 at 94 s and 9.8155 are the
pre-mixture readings, and 21/21, 254 s and held-out **10.795** on 911 rows the pre-lock,
pre-window ones).

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

The empirical onset hazard falls steeply with games-since-return. ✅ Measured by
`make games-played` (`analysis == "hazard_by_streak"`) over the appearance window:

| played streak | 1 | 2 | 3 | 4 | 5 | 6–10 | 11–20 | 21–41 | 42+ |
|---|---|---|---|---|---|---|---|---|---|
| **onset hazard** | **0.2976** | 0.1912 | 0.1386 | 0.1067 | 0.0926 | 0.0672 | 0.0463 | 0.0338 | **0.0208** |
| at risk | 81,525 | 55,387 | 43,601 | 36,694 | 32,125 | 120,539 | 142,519 | 133,322 | 74,226 |

A **14.3×** fall from a one-game streak to past 42.

> ⚠️ The recorded **0.3164** → **0.0265** and its 11.9× are superseded: the endpoints move
> with where the last bucket is cut, and the shipped buckets put 21–41 and 42+ in separate
> rows rather than pooling them. The *shape* is unchanged and it is the shape that carries
> the argument.

That looks like it should break the collapse. It
does not: a pure-frailty model with *zero* state dependence reproduces almost exactly that
shape, purely by sorting — players with high onset hazards break their streaks
early, so long streaks are populated by low-hazard players. **The frailty the beta-binomial
marginalizes is precisely what generates this curve, so the collapse keeps it for free**, and
Gate C checks that the *simulated* curve keeps falling past a streak of 20 for exactly that
reason: a simulator whose frailty has collapsed to a point mass produces a flat curve, and
that failure is invisible in the GP marginal because a shared hazard and a distribution of
hazards with the same mean give the same mean.

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
the process class fails there, it fails everywhere.

> ### ✅ Measured — `make games-played`, `outputs/predictions/stan_games_played_gate.csv`
>
> **The frame is single-team established-rotation player-seasons, and that is not a
> detail.** The panel's process runs per (season, player, team) while games played is a per
> (player, season) quantity, so on a multi-team row `gp` counts one team's games against
> that team's whole schedule — a traded player reads as having missed half the season twice
> over. Leaving them in moves the **observed** left tail from 0.111 to 0.237, which is large
> enough to hide the very overshoot the gate exists to detect: the gate then *passes* the arm
> this doc rejects, for a reason that has nothing to do with the process class. That is the
> third construction of this frame and it is the one that ships; the two planning-session
> reconstructions disagreed on levels and agreed on direction, and the levels below supersede
> both.
>
> On **4,667** single-team rotation player-seasons, 200 simulated seasons each:
>
> | arm | mean | sd | overdispersion | P(GP<41) | P(GP<60) |
> |---|---|---|---|---|---|
> | **observed** | 0.8070 | 0.2120 | 23.66× | **0.1112** | 0.3107 |
> | `full_window_chain` | 0.7988 | 0.2399 | 29.37× | **0.1356** | 0.3189 |
> | `tenure_decomposition` | 0.8132 | 0.2197 | 26.05× | **0.1155** | 0.3009 |
>
> **The plain full-window chain over-predicts the left tail by 21.9% and the tenure
> decomposition by 3.8%.** The bar is the sampling error of the observed proportion rather
> than a chosen tolerance — Monte Carlo error here is negligible, but the observed tail is
> itself an estimate from a few thousand player-seasons, and `sqrt(p(1-p)/n)` = 0.0046 is
> that uncertainty. At two of them the plain chain fails at **z = +5.29** and the tenure
> decomposition passes at **z = +0.93**.
>
> **Read the overdispersion column too.** At its *empirical ceiling* the tenure
> decomposition already overshoots the observed variance by 10% (26.05 against 23.66) — so
> the dispersion surplus this doc warns about is visible before any head is fitted, and it is
> a property of the process class rather than of the covariate block.
>
> The gate is judged on `P(GP < 41)` and not on the mean, deliberately: every arm reproduces
> the mean to within a point, because a chain running at a cell's own observed rates can
> hardly miss it. A gate on the mean would pass the arm this doc rejects.

**The left tail is over-predicted** — and under a second reconstruction that keys the
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

> ~~The *shares* above reproduced across two independent reconstructions during planning; the
> mean lengths did not (19.7 / 17.7 games under one construction against 6.8 / 6.6 under
> another, on identical shares). The gap is a denominator convention and is unresolved. Do not
> quote a mean tenure length until `spell_classes` emits one.~~
>
> ✅ **Resolved 2026-08-05 — the two constructions are conditional and unconditional means of
> the same measurement, and both are correct.** `tenure_frame` on the 12,787 single-team
> player-seasons gives mean pre/post tenure of **19.67 / 17.67 games conditional on being
> nonzero** and **6.76 / 6.59 unconditional**. The shares are unchanged (34.4% / 37.3%), which
> is exactly why they reproduced while the means did not: the conditional mean divides by the
> 34.4%, and `19.67 × 0.3436 = 6.76`. Neither figure was wrong; naming the denominator is the
> whole fix. Quote the **conditional** pair when describing how long a late arrival or an
> early departure lasts, and the unconditional pair when describing the league.

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
from 2006-07 gives **82.7% agreement** over 422,219 missed games. ✅ Reproduced by
`make games-played` (`analysis == "status_agreement"`), along with the 98.5% persistence
above; the recorded **16.75%** below is **16.74%** measured, a rounding difference and not a
correction. The interesting cell is the **16.74%** that are missed
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
| drop censored | **1.088** | **1.323** | 16.03 | **0.0398** |
| censored treated as complete | **0.917** | **1.252** | — | **0.0598** |
| **proper right-censoring** | **0.721** | **0.978** | — | **0.0861** |

Dropping censored spells understates `P(T≥26)` by **2.16×**; treating them as complete still
understates by 1.44×. Both fail silently.

> ⚠️ **The three fitted `(a, b)` pairs reproduced exactly; the E[T] and P(T≥26) columns did
> not, and are corrected above (was 5.84 / 7.68 / 9.95 and 0.0377 / 0.0549 / 0.0741).** The
> planning session computed the two derived columns by a construction that does not follow
> from its own fitted parameters — `E[T] = (a+b−1)/(a−1)` at `a = 1.088, b = 1.323` is
> **16.03**, not 5.84, and the two E[T] rows below it were quoted from a distribution that has
> no mean at all. Since the *parameters* agree to three decimals across two independent
> implementations, the fits were right and the derivation on top of them was not.
> `tests/test_games_played.py` now pins `sum P(T=t) + P(T>t) == 1` at every truncation point
> and the mean identity numerically, so the derived column cannot drift from the fitted one
> again.

> With proper censoring `a = 0.721 < 1`, so the fitted beta-geometric has **no finite mean**
> — and its tail is *polynomial* rather than geometric, `P(T ≥ t) ~ t^−a`, which is why 40,000
> terms capture only 73% of the mass. The simulator is unaffected — a spell is truncated by
> the remaining schedule anyway — but `E[T]` must never be quoted from that fit, and
> `make docs-audit` will happily pin a meaningless number if allowed to. `fit_beta_geometric`
> returns `mean_defined` beside `mean_spell` so the two cannot be separated.

**Left truncation has an exact answer.** By memorylessness the forward recurrence time of a
geometric is geometric with the same hazard, so under a Beta(a,b) frailty the *residual*
duration of an in-progress spell is beta-geometric with the frailty size-biased by `1/e` —
i.e. **Beta(a−1, b)**. Fit the in-progress offset freely and compare against `a − 1`:
agreement validates the renewal assumption, disagreement localizes it.

> ### ✅ Measured — and it **disagrees**, which is the informative outcome
>
> Fitted jointly with the interior spells (the offset is unidentified on the truncated
> spells alone — the base `mu` and the shift are then the same parameter and the optimizer
> walks it to infinity, measured), the free offset is **−1.7042** on the logit scale, giving
> an in-progress `mu` of **0.1367** against the identity's prediction of **0.3027**. The
> in-progress spells are **2.2× longer** than length-biased renewal allows.
>
> **That is the roster-mechanics signal, arriving through a completely different door.** The
> left-truncated spells are **58.7%** `not_rostered` games: a player absent on opening night
> is mostly not an injury in progress, he is someone who was not on the team yet. A renewal
> process cannot represent that, and the identity says so rather than absorbing it. This is
> the third independent measurement pointing the same way, after the spell-class table and
> the 82.7% status agreement — and it is why the within-tenure duration head is fitted on
> **interior spells only**.
>
> Two things had to be right for this to be a check rather than an artifact, and one of them
> was wrong on the first attempt. The identity requires **length-biased** selection — a fixed
> time point falls inside a spell with probability proportional to its length, which is what
> size-biases the frailty by `E[T|q] = 1/q` and turns Beta(a, b) into Beta(a−1, b). Sampling
> spells *uniformly* instead reproduces the base distribution; `tests/test_games_played.py`
> caught that at 0.29 in the first pmf cell. And it needs `a > 1`, since length-biasing is
> normalizable only then — the full-window fit lands at `a = 0.721`, so the identity is
> reported as **undefined** there rather than computed anyway.

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

**As built, with the three places it differs from the sketch above and why:**

- **`spell_classes` and `collapse_transitions` landed in `src/features/availability.py`
  as planned, alongside a third — `tenure_frame`.** The tenure identity
  (`team_games = pre + tenure + post`, `gp` = played inside the tenure) is used by Gate 0,
  by the design builder and by the simulator, so deriving it in three places was the
  alternative.
- **`season_total.py` consumes an artifact, not an import.** It reads
  `stan_games_played_gp_pmf.csv` — the held-out games-played pmf in long form — because
  `make season-total` has to run on a machine with no CmdStan toolchain. Long form rather
  than one column per game index, for the reason `residual_correlation.csv` is long: a wide
  schema breaks the first time a season is not 82 games.
- **`betageometric_duration.stan`'s censoring branches are live code, and A3 is what
  exercises them.** Within an *appearance* tenure every spell is interior, so the shipped
  arms never reach them — but A3's tenure is the **rostered** window, where a player who
  tore an ACL in March stays on the roster and his absence genuinely runs to the season's
  end — so spells inside the rostered window *can* be censored, and thousands are.
  `spell_rows_for(..., require_interior=)` asserts the interior property where it is
  supposed to hold rather than assuming it, and that assertion is what surfaced the
  distinction: the first three-state run failed on it.

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

> ### ⚠️ RETIRED — the pre-lock ladder, whose verdict was read off the TEST column
>
> **Kept as the record of a reversal, not as a result.** The heading used to read "THE LADDER
> RAN — and the two-line **calibration** beats every fitted arm", and every conclusion in this
> block was taken from the `test CRPS` column. The summary at the top of this doc is what
> replaces it: no arm ships, the incumbent stands, and the fallback loses CRPS *and* PIT on
> the split that is allowed to decide. The `val CRPS` column below is the only one that was
> ever admissible, and reading it alone already gives the current verdict — which is the
> point of preserving the table rather than deleting it.
>
> `make stan-games-played`, 2026-08-05. **25 fits, 0 divergences, max R̂ 1.0071, min ESS 856,
> 0 treedepth-saturated draws, 42.7 min.** 11,272 player-seasons, 10,361 train / 911 test
> (2024-25, 2025-26 held out), validation on 2022-23/23-24. 9,753 of 11,272 rows (86.5%) are
> single-team and therefore fittable; the rest keep their row and lose their process targets.
>
> | arm | val CRPS | test CRPS | its own floor | vs floor | PIT KS | implied od | tail error | selected |
> |---|---|---|---|---|---|---|---|---|
> | `floor` (the incumbent) | 10.0057 | 10.7952 | 10.7952 | — | 0.0963 | 23.33 | 0.0264 | |
> | *`within_tenure`* (oracle) | *7.2265* | *7.0391* | *10.9870* | *−3.9479* | *0.1227* | *10.84* | *0.0123* | |
> | `full_window` | 10.2705 | 10.8981 | 10.7952 | +0.1029 | 0.0679 | 21.48 | 0.0205 | |
> | `three_state` | 10.3484 | 11.2470 | 10.9795 | +0.2676 | 0.0992 | 19.83 | 0.0271 | |
> | **`duration_covariates`** | **10.1625** | 10.8026 | 10.7952 | +0.0074 | 0.0679 | 21.35 | 0.0192 | **✓** |
> | **`calibrated_fallback`** | 10.0207 | **10.7939** | 10.7952 | **−0.0013** | 0.0907 | 23.33 | **0.0174** | *not selectable* |
>
> **⭐ The sharpest result is the oracle arm, and it is not the one that ships.** Hand the
> process the *observed* tenure and it scores **7.0391** against its own floor's 10.9870 — a
> **−3.95 game** improvement, 36%, on the same rows. So the process class is not merely
> adequate, it is dramatically better than the season-level beta-binomial **given the
> tenure** — and every bit of that advantage is destroyed by having to predict the tenure
> from preseason covariates. **The bottleneck is not the absence process. It is knowing when
> a player joins and leaves a roster**, which is mid-season churn, which `CLAUDE.md` already
> scopes out as irreducible. That is a much more specific statement of where the remaining
> value is than "availability is hard".
>
> **Every fitted full-window arm loses to the incumbent**, by +0.007 to +0.268 CRPS. That is
> what the dispersion-budget argument predicted *in writing before the sweep ran*, and it is
> why GP CRPS was fixed as a non-regression bar rather than the win condition. Note the
> mechanism is visible in the `implied od` column: the fitted arms land at 21.3–21.5 against
> the incumbent's 23.33, so they are not over-dispersed as feared — the tenure factors cost
> them accuracy rather than calibration.
>
> **`three_state` is a null and the only arm that fails its own floor** (+0.2676 against a
> floor refit on the same 2006-07+ rows). Identifying the tenure from the box-score `status`
> rather than structurally does **not** help, even though it reaches the 16.74% of missed
> games the structural proxy cannot see. Recorded so it is not rebuilt.
>
> **⚠️ THIS IS THE CLAIM THAT REVERSED.** It read: *what ships is option (b) — the fallback —
> and it beats all four fitted arms.* Two
> lines of algebra, no features, no Stan: invert `inflation = C + ρ(n − C)` at the measured
> `C = 9.5806` to get hazards that reproduce the incumbent's marginal by construction while
> getting the game-level clustering right. It scores **10.7939** against the incumbent's
> 10.7952 — a wash, exactly as predicted — and cuts the tail error from **0.0264 to 0.0174**.
>
> On validation it loses on both counts it was selected for: CRPS **10.0207** against the
> incumbent's **10.0057**, and PIT KS **0.1017** against **0.0963**. **Two independent defects
> produced that ✅ and either alone would have been enough.** The margin was −0.0013 on a
> ~10.8 game quantity, which is nothing; and the single input the closed form takes was itself
> contaminated, because `measured_clustering` computed `C` over all 30 seasons rather than
> over train plus validation — so the fallback's one "measured" constant had seen the rows it
> was later certified on. `dashboard/decisions.py::games-played-ships-the-calibration-not-the-fit`
> carries the corrected `C` and the test CRPS it implies, at which the arm fails that bar too.
> **The "coordinate change beats the fitting" reading does not survive here.** That shape is
> real for the shot-attempt basis, where it is measured on validation and the margin is
> −0.501 nats; it was borrowed for this head on the strength of a number that was neither.

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

> ### ⚠️ RETIRED — "it is what ships". The identities do still hold.
>
> The heading read "It is what ships, and the identities hold", and the second half is the
> half that survives. `tests/test_games_played.py` still pins the simulator against the
> algebra: the mean is exact by construction, and the variance lands within 2% of the target
> at `C = 9.58` and **exactly** at `C = 1`, where there is no clustering term to approximate.
> (The ~2% shortfall is not an error — `(1+ρ)/(1−ρ)` is the *asymptotic* inflation and a
> season is 82 games, not infinitely many.) An invariant worth pinning regardless of which
> arm wins is worth pinning when none does, so that test stays.
>
> Fitted values on the held-out board: `ρ_M` = **0.8110**, residual `ρ` = **0.1899**, mean
> onset hazard **0.0700**, mean recovery hazard **0.1190**. **This block is now the only copy
> of those four numbers.** `spell_process.csv` writes the calibrated parameters only when the
> fallback is the shipping arm, and it no longer is, so `src/docs_audit.py` holds `ρ_M` as a
> presence-checked historical claim instead of a value-checked one — deletion, not drift, is
> what can still go wrong here.
>
> **The explanation of why it won is retired along with the win.** It ran: *it inherits a
> marginal that a validated head already calibrated, and then spends its own structure on the
> shape — which games, clustered the way the transition data says.* The mechanism is sound and
> it is exactly what the **hybrid** arm is built on; what was wrong was attributing it to the
> fallback, whose simulated spells are the worst of any arm — mean length **10.0594** games
> against an observed **3.0857**. A constant recovery hazard cannot produce the observed spell
> shape, so the fallback inherits the marginal and then gets the shape wrong, which is the
> opposite of the claim above. The hybrid inherits the same marginal and gets the shape right.

## Gates

| Gate | Pass condition | as implemented |
|---|---|---|
| **0** | the process class reproduces the GP marginal at its **ceiling** (per-cell empirical hazards). Full window fails; the tenure decomposition is the response | `P(GP<41)` within **2 standard errors of the observed proportion** — the weakest bar that still separates the arms, and derived rather than chosen |
| **A** | cost probe under budget — **treat as a lower bound**, it under-predicted `stan-composition` by 1.63× | the extrapolation is **multiplied by 1.63** and compared against the budget, so the known bias is applied rather than remembered |
| **B** | onset head clears its no-fit floor (prior-season onset rate carried forward) | held-out binomial log-likelihood **per at-risk transition**, against a floor whose shrinkage constant is fitted on train only |
| **C** | simulated spell shape matches `availability_profile.csv`, **and the hazard-by-k curve keeps falling past k = 20** | the curve is read off *simulated* seasons and compared with the observed one bucket for bucket |
| **D** | **the head gate** — GP CRPS ≤ **10.795**, PIT KS ≤ **0.096**, and the tail: `P(GP<41)` near **0.109**, `P(GP<60)` near **0.318** on the rotation subpopulation. The incumbent reads 15.0% / 34.8% against 11.8% / 36.9%, so the tail is where the win has to come from | CRPS and PIT are non-regression bars; the win condition is **mean absolute tail error** across the two thresholds, against the incumbent's on the same rows |
| **E** | season-total MAE ≤ **435.1** and CRPS ≤ **316.9** dk_pts through `season_total.py` | a sixth GP treatment, `spell_process`, holding the rate model and every other row fixed |

**Gate 0 is judged on the left tail and not on the mean, and that is a decision rather than
a convenience.** Every arm reproduces the mean to within a point — a chain running at each
cell's own observed rates can hardly miss it — so a gate on the mean would pass the arm this
doc rejects. `P(GP < 41)` is where a relocated departure shows up, and it is what Gate D is
later judged on.

### ✅ Every gate ran, and every one passes — but not for the arm that was expected to carry it

| gate | verdict | figure |
|---|---|---|
| **0** | ✅ | plain chain **z = +5.29** (rejected); tenure decomposition **z = +0.93** |
| **A** | ✅ | 0.4 h linear → **0.7 h** corrected against a 6 h budget; **actual 42.7 min** |
| **B** | ✅ | onset head **−0.350374** per at-risk transition against the floor's −0.366963 (**+0.016590**) |
| **C** | ✅ | the simulated hazard curve keeps falling past a streak of 20 |
| **D** | ✅ | **not** by the fitted arm — `duration_covariates` fails CRPS at 10.8026. The **calibrated fallback** passes all three: CRPS **10.7939**, PIT KS **0.0907**, tail error **0.0174** against the incumbent's 0.0264 |
| **E** | ✅ | season-total MAE **435.1053** against the 435.1352 bar, CRPS **316.6483** against 316.9385, bias **+3.75** against +6.12 |

**Gate A's correction is worth recording on its own.** `stan-composition` measured a 1.63×
under-prediction and this head applied it rather than remembering it: 0.4 h linear × 1.63 =
**0.7 h** against an actual **42.7 min = 0.71 h**. A bias that is measured once and then
applied is worth more than a bias that is measured once and then written down.

**Gate E is a wash on MAE and a small real gain on CRPS and bias**, which is exactly what a
head calibrated to reproduce the incumbent's marginal should produce. It clears the bar by
0.03 dk_pts of MAE and 0.29 of CRPS, and it halves the bias. Nobody should read those as the
justification for the work — the justification is the game-level process, which the
season-total marginal cannot see at all.

**Gate B's floor is a shrunk carry-forward and the shrinkage is fitted, not chosen**: `k` =
**42.8** pseudo-observations toward a league onset rate of **0.0771**. An unshrunk
carry-forward would be meaningless here for the reason `component_rates` records for the
conversion heads — a player whose prior season carried two at-risk transitions and no onsets
has a prior rate of exactly 0.000, and a binomial likelihood at `p = 0` on real exposure is
non-finite.

**The two Gate D candidates, in full:**

| | CRPS | PIT KS | `P(GP<41)` | `P(GP<60)` | tail error |
|---|---|---|---|---|---|
| observed | — | — | **0.1180** | **0.3687** | — |
| incumbent (`floor`) | 10.7952 | 0.0963 | 0.1499 | 0.3478 | 0.0264 |
| `duration_covariates` | 10.8026 ❌ | 0.0679 ✅ | **0.1551** | **0.3700** | 0.0192 ✅ |
| **`calibrated_fallback`** | **10.7939** ✅ | **0.0907** ✅ | **0.1443** | **0.3601** | **0.0174** ✅ |

Both challengers beat the incumbent on the tail; only the fallback also holds the CRPS bar.
Note that the fitted arm *over*-shoots `P(GP<41)` (0.1551 against 0.1499) while improving the
aggregate tail error — its gain comes entirely from the 60-game threshold, where it lands at
0.3700 against an observed 0.3687. The fallback improves both.

> ⚠️ **The tail comparison is weaker than the point estimates make it look, and the
> denominator is why.** Gate D's target is measured on **339** rotation test rows, so the
> observed `P(GP<41) = 0.1180` carries a standard error of **0.0175** and `P(GP<60) = 0.3687`
> one of **0.0262**. Both candidates and the incumbent sit within ~2 of those of the target.
> The *ordering* is a deterministic property of three predictive distributions and is not in
> doubt; how far any of them is from the truth is. Do not quote "34% closer" as though the
> truth were known to three decimals.

**Failing D lands in option (b), not in the bin** — the same simulator, hazards calibrated by
the closed form above. Say so in the module docstring so the fallback is a documented branch
rather than a rescue.

## Artifacts, target, config

```
# make games-played — numpy only, no CmdStan toolchain required
outputs/predictions/stan_games_played_gate.csv          Gate 0 — sim vs obs, per arm
outputs/predictions/stan_games_played_spells.csv        the three spell classes, the
                                                        censoring-bias table, the duration
                                                        candidates, the hazard-by-streak
                                                        curve and the left-truncation check
outputs/predictions/stan_games_played_collapse.csv      the sufficient-statistic reduction

# make stan-games-played — the fitted arms
outputs/predictions/stan_games_played_metrics.csv       one row per arm
outputs/predictions/stan_games_played_coefficients.csv  onset / duration / entry / exit
outputs/predictions/stan_games_played_diagnostics.csv   diagnostics_frame(), one row per fit
outputs/predictions/stan_games_played_gates.csv         the A-D verdicts, one row per gate
outputs/predictions/stan_games_played_gp_pmf.csv        the held-out GP pmf, long form —
                                                        Gate E's input to season_total.py
outputs/predictions/stan_games_played_pit.csv
outputs/predictions/stan_games_played_predictions.csv
outputs/predictions/spell_process.csv                   the dashboard's registered slot
```

**`spell_process.csv` is not optional.** `dashboard/tabs/simulations.py::CHAIN` hard-codes
that path as its build check, and `make dashboard-audit`'s orphan check will otherwise flag all
six `stan_games_played_*` families as unreachable — trading one finding for six. Emit it *and*
repoint the `CHAIN` row at `stan_games_played_metrics.csv`.

```makefile
games-played:
	$(PYTHON) -m src.models.games_played

stan-games-played: games-played
	$(PYTHON) -m src.models.stan_games_played
```

**Two targets, not one, and the split is the point.** `games-played` is Gate 0 and the
closed forms — pure numpy, no CmdStan, ~6 seconds. Rejecting a process class should not
require a toolchain or an overnight run, and this doc's frame changed *because* that check
was cheap enough to run first.

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
   - ✅ **Confirmed, and the cheap check was cheaper than budgeted**: `make games-played` is
     **6 seconds** end to end, including the closed-form duration fits. The plain chain
     over-predicts the left tail at **z = +5.29**. Two things about the *check* rather than
     the result are worth keeping: the gate had to be judged on the tail rather than the
     mean (every arm reproduces the mean), and it had to be run on **single-team** rows —
     the first version of the frame included traded players, which moved the *observed*
     tail from 0.111 to 0.237 and made the gate pass the arm it exists to reject.
2. **The head wins nothing on GP-marginal CRPS, discovered after the sweep instead of before.**
   The dispersion budget is already over-supplied (`C + ρ(n−C)` overshoots by 30% at the
   incumbent's ρ). *Cheap check: evaluate that identity at the fitted `C` before scoring
   anything.* Ship on the tail and the joint; keep GP CRPS as a non-regression bar.
3. **The duration head fits roster mechanics and gets called an injury model.** On the full
   window 61.4% of missed games sit in edge spells and ~half of those games are
   `not_rostered`. *Cheap check: emit the spell-class table as an artifact before fitting, and
   report the `not_rostered` share of the fitted population beside every duration
   coefficient.*
   - ✅ **Confirmed three independent ways, which is why the shipped duration head fits
     interior spells only.** The spell-class table puts `not_rostered` at **58.7%** of
     left-truncated and **46.2%** of right-censored games against **1.0%** of interior ones.
     The structural proxy agrees with `not_rostered` on 82.7% of missed games from 2006-07.
     And the renewal identity — which knows nothing about rosters — **fails in the same
     direction**: in-progress spells are 2.2× longer than length-biased renewal allows,
     because a player absent on opening night is mostly not an injury in progress, he is
     someone who was not on the team yet. Three doors, one room.
