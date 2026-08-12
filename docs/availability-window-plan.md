# Availability window plan: the fitting window, a season trend, where `rho` lives, and which likelihood

The availability head misses **both ends** of its own distribution, in opposite directions,
and no metric it has ever been gated on can see that. This doc is the measurement of why,
the ladder that separates the candidate causes, and what it settled.

`make availability-window` → `outputs/predictions/availability_window.csv`, one row per arm.
Built 2026-08-11. It is a **point-MLE specification ladder**, not a shipping path: an arm
that wins here earns a Stan port in `stan_availability.py`, it does not ship from here.

---

## 1. The defect

From `model_card_ecdf.csv`, the head's own posterior-predictive band against the observed
ECDF on the 883 validation player-seasons and the 9,478 training ones:

| | train observed | train predicted | validation observed | validation predicted |
|---|---|---|---|---|
| P(GP < 10) | **5.70%** | 4.16% [3.66, 4.58] | **8.15%** | 5.21% [3.85, 6.68] |
| P(GP ≥ 82) | **6.60%** | 7.56% [6.93, 8.22] | **2.72%** | 5.95% [4.75, 7.81] |

All four sit **outside** the 95% band. Two more grid points fix the shape: the head
*over*-predicts P(GP ≤ 1) on train (0.68% against an observed 0.46%) and under-predicts
P(GP ≤ 75) on validation (78.3% against 84.6%). So the fitted Beta frailty is too
**U-shaped** — too much mass on both boundaries, too little in the shoulders at 2–15 and
70–80 games. One defect, two symptoms.

> ⚠️ `docs/potential-to-dos.md` item 4 recorded this as "under-predicting the frequency of
> players playing all 82 games **and** playing very few". The low half is right. The high
> half is **backwards** — the head over-predicts a full schedule, by 2.19× on validation.
> Corrected here rather than in place, because the direction is what tells you the
> mechanism.

**Why it matters more than its CRPS cost.** A Round-1 knockout is decided by the best 7 of
16 in a week. A player who plays four games is a dead roster slot and an iron man is a
ceiling; both errors make a drafted roster look *more reliable than it is*, which biases
every strategy axis that trades ceiling against reliability — stacking, handcuffing,
diversification. `README.md` already calls this head the largest lever on the season total.

### 1b. What the shipped head does to it — measured 2026-08-11, and it does not close

The windowed, role-graded head is now what `make posteriors` and `make model-cards` carry, so
the card above can be re-read against the same statistic. **Both misses narrowed and neither
came inside the band.** Validation, the split the defect was stated on and the only one whose
population did not change:

| | observed | predicted, incumbent | predicted, shipped | outside the band by |
|---|---|---|---|---|
| P(GP < 10) | **8.15%** | 5.21% [3.85, 6.68] | **5.66%** [4.42, 7.36] | 1.47 pp → **0.79 pp** |
| P(GP ≥ 82) | **2.72%** | 5.95% [4.75, 7.81] | **4.30%** [3.17, 6.12] | 2.04 pp → **0.45 pp** |

Read the third column against the second and the fourth on its own. The signed errors go
−2.94 → **−2.49** pp and +3.23 → **+1.59** pp, which is 15% of the low-tail miss and 51% of
the high-tail one; the *excursion past the 95% band* falls by 46% and 78%, because the shorter
window also widens the band. The shoulder the defect's second symptom named moves with them —
P(GP ≤ 75) on validation goes 78.3% → **81.0%** against an observed 84.6%, closing 43% of that
gap. The point-MLE ladder in §4 predicted errors of −0.0230 and +0.0126 for this arm; the
Bayesian head's posterior predictive reads −0.0249 and +0.0159, so the two independent
measurements agree in direction and to within a fifth of a percentage point.

**So the CRPS and PIT wins in §4 stand and the headline claim does not.** "The head misses
both ends of its own distribution" is still true of the head that ships. What §4 bought is a
smaller miss on the same side of the same band, not a calibrated boundary — which is exactly
what result 4 said in advance: *the low tail is a functional-form limit of the beta-binomial*,
and the high tail is a shape the only instrument that closes it (a season trend) closes by
wrecking the body. The remaining instrument is a different frailty, not a different window.

**Train is not a like-for-like row.** The window cuts the *fitting* rows, so the training card
now describes 4,027 player-seasons from 2012-13 rather than 9,478 from 1997-98, and its
observed column moves with the population: P(GP ≥ 82) reads 3.82% observed against 4.57%
predicted, where the incumbent read 6.60% against 7.56%. The head did not get better at the
2000s; it stopped being asked about them.

`make model-cards` → `outputs/predictions/model_card_ecdf.csv`, `head == "availability"`.

---

## 2. The league moved, and the shape of the movement is measured

Measured on the head's **own design rows** — `season_availability(panel, "full")` through
`build_design`, with `selection_split` dropping the held-out seasons, so the population is
the head's and the split is enforced by the guard rather than by a filter. It reconciles
with the ECDF card to the decimal (train P(GP<10) 5.5% against the card's 5.70%; validation
P(GP≥82) 2.7% against 2.72%), which is the check that the frame is right.

Mean `gp_share` sits flat near **0.710 for twenty seasons** and then falls to 0.605–0.638.

**A sup-F scan over every candidate breakpoint, calibrated against a 5,000-replicate
Monte-Carlo null** (the max over breakpoints has no standard F distribution):

| null | sup-F | best break | p |
|---|---|---|---|
| constant | **91.8** (null 95th pct 9.2) | **2017-18** | < 0.0002 |
| linear trend | **33.2** (null 95th pct 10.5) | 2019-20 | < 0.0002 |

Three qualifications, all load-bearing:

- **The location is not identified.** F reads 91.8 / 90.6 / 83.9 at 2017-18 / 2018-19 /
  2019-20. "Somewhere in 2017–2020" is the honest statement.
- **It is a slope change, not a step.** BIC prefers a broken trend (−229.8) and
  step-plus-broken-trend (−235.7) over a level shift (−220.8). Pre-2017 slope
  **−0.00065**/season against post-2017 **−0.0103**, sixteen times steeper, t = −3.01.
- **COVID cannot be fully separated.** Dropping the two shortened seasons leaves sup-F at
  85.2, so it is not a season-length artifact — but pre-COVID data *alone* (through
  2018-19) gives only sup-F 11.9 with a −0.019 shift. The change begins before the pandemic
  and its magnitude is established almost entirely by seasons after it.

The two tails break in **different places**, which is why no single window can serve both:

| statistic | best break | F | shift |
|---|---|---|---|
| mean gp_share | 2017-18 | 91.8 | −0.065 |
| P(played every game) | **2004-05** | 37.6 | −0.078 |
| P(GP < 10) | 2017-18 | 14.0 | +0.022 |

The iron-man share has been eroding since the **mid-2000s** — a twenty-season decline, not a
recent event.

> ⚠️ **Build this series through the head's own design builder, never by filtering the
> feature frame by hand.** `availability_features.parquet` carries **two rows per
> player-season** — an `appearance` window and a `full` one — and `availability.py:929`
> fits the `full` one. Reconstructing the population by hand picked the wrong window in one
> attempt and fanned out 4× on a merge in another, and neither failure raised. Both were
> caught only because the resulting P(GP<10) level disagreed with `model_card_ecdf.csv`,
> which is the argument for keeping a second independent measurement of the same quantity.
> Going through `build_design` + `selection_split` costs one function call and cannot get
> either wrong.

---

## 3. The ladder

Three axes crossed, 18 arms plus a league/age floor, seconds to run:

| axis | levels | what it tests |
|---|---|---|
| **window** | `full` (1996-97), `three_point_era` (2012-13), `post_break` (2017-18) | was the movement a completed level shift? |
| **season term** | `none`, `trend`, `trend_x_role` | is it a continuing slope? |
| **dispersion** | `shared`, `role` (prior-MPG buckets) | is one `rho` for every player the defect? |

Windows restrict the **fitting half only**. `add_trend` centres on train, so validation sits
outside the fitted range in every arm. `RoleGradedBetaBinomial` re-fits `rho` inside each
`season_effects.ROLE_EDGES` bucket holding the mean fixed, and nests the shared head exactly
when the buckets agree (pinned by a test).

Metrics add **tail coverage** — predicted against observed rate at `below_10 / 41 / 60` and
at `GP == team_games` (each row's own schedule, so shortened seasons contribute a reachable
boundary) — plus 50/80/95% interval coverage, on top of CRPS, MAE, R², PIT KS and a paired
bootstrap against the incumbent.

> **`mean_abs_tail_error` is kept but is not the selector, and that is deliberate.** It
> averages all four thresholds, and 41 and 60 sit in the **body** of an 82-game
> distribution. An arm that shifts location can fix both boundaries and wreck those two
> while its four-threshold mean gets *worse* — which is exactly what the trend arms do.
> `boundary_tail_error` is the pair the defect is about.

---

## 4. What it settled

`full__none__shared` reproduces the incumbent's **10.0057** CRPS to four decimals, which is
the check that the ladder's reference is the shipped head.

Observed on validation: P(<10) **0.0815**, P(<41) 0.2911, P(<60) 0.5300, P(full) **0.0272**.

| arm | CRPS | vs incumbent [95%] | PIT KS | err P(<10) | err P(full) | err P(<41) | err P(<60) |
|---|---|---|---|---|---|---|---|
| `full__none__shared` *(incumbent)* | 10.0057 | — | 0.0939 | −0.0290 | +0.0337 | +0.0149 | −0.0017 |
| `full__none__role` | 9.9891 | −0.017 [−0.032, −0.001] | 0.0939 | −0.0251 | +0.0291 | +0.0120 | −0.0050 |
| `three_point_era__none__shared` | 9.8444 | −0.161 [−0.234, −0.089] | 0.0632 | −0.0289 | +0.0187 | +0.0274 | +0.0228 |
| **`three_point_era__none__role`** | **9.8247** | **−0.181 [−0.257, −0.103]** | **0.0588** | −0.0230 | +0.0126 | +0.0218 | +0.0188 |
| `three_point_era__trend__role` | 10.0677 | +0.062 [−0.130, +0.252] | 0.1244 | **−0.0060** | **−0.0019** | **+0.0733** | **+0.0793** |
| `post_break__trend__shared` | 10.7940 | +0.788 [+0.445, +1.118] | 0.1906 | +0.0085 | −0.0104 | **+0.1382** | **+0.1454** |

**Four results.**

**1. The window is real and it is not the one the break test picked.**
`three_point_era` (2012-13, 4,027 rows) beats the incumbent by **−0.161** CRPS with an
interval clear of zero, and posts the best PIT of any shared-`rho` arm. `post_break`
(2017-18) does *not* clear — −0.091 [−0.218, +0.037] — because five target seasons and 2,089
rows is too few. **The break's location and the best fitting window are different
questions**: one is where the regime changed, the other is a bias–variance trade, and here
the trade wants more rows than the regime does.

**2. A season trend buys the boundaries by wrecking the middle.** It is the only instrument
that closes the tails — `three_point_era__trend__role` gets P(full) error to −0.0019 and
P(<10) to −0.0060 — and it does so by **shifting the whole distribution down**, so P(<41)
and P(<60) blow out to +0.073 and +0.079 and both CRPS and PIT degrade. On the short window
it is catastrophic (+0.788 CRPS). **A location instrument cannot fix a shape defect**, and
this is the cleanest demonstration of it in the repo: the arms with the best boundary
coverage are the worst models.

**3. `rho` is role-graded, mildly, and it is not where the era lives.** Fitted per prior-MPG
bucket on the full window: **0.3084** for `<12 mpg` against **0.2456** for `30+ mpg`, a
**1.26×** spread (1.47× on the 2012-13 window). Direction is sensible — fringe players are
more variable — but it is far milder than the composition head's **2.07×**. Grading it is a
small, consistent, nearly free win (−0.017 CRPS on the full window, −0.020 on the era one)
and it never hurts a single metric **in the table as it stood**.

> ⚠️ **Scoped 2026-08-11**, when §7f added a shoulder column the table did not have. Role
> grading is a **wash** there: `shoulder_error` goes 0.0280 → **0.0290** on the full window
> and 0.02092 → 0.02104 on the era one. So "never hurts a single metric" was true of the
> four thresholds then measured and is not true in general. It still wins CRPS, PIT,
> `boundary_tail_error` and `point_mass_error`, so the decision stands — but the
> unqualified version of the claim does not, and a metric added later is exactly how that
> kind of sentence gets found out.

**4. The low tail survives every instrument.** The best non-trend arm moves P(GP<10) error
from −0.0290 to −0.0230 — about a fifth of it. The window does essentially nothing for it
(−0.0289 against −0.0290), role-graded `rho` a little, and only the trend closes it, at the
cost above. **The low tail is a functional-form limit of the beta-binomial**, not an era
effect and not a dispersion-pooling one.

---

## 4b. Confirmation on the full rowset — rolling origin, fitting half only

Nineteen arms on 883 rows is a multiplicity problem and a power problem at once. The
confirmation answers it **without spending validation twice**: an origin walks across the
fitting half, each arm fits on the seasons before it and scores the season itself, and the
rows pool. **13 origins, 5,142 scored rows** — 5.8× the validation set — and not one of them
is a validation or held-out row, so the validation reading stays the arbiter.

It also reparameterizes the window into the thing that will still mean something when the
production fit runs: a **lookback length** rather than an absolute first season.

`three_point_era__none__role`'s two components both replicate. The third does not, and the
way it fails is the most useful result in this doc.

| lookback | mean fit rows | CRPS | vs `all` | origins won | PIT KS | boundary err | **body err** |
|---|---|---|---|---|---|---|---|
| all | 6,630 | 9.9478 | −0.007 | 9/13 | 0.0526 | 0.0246 | 0.0062 |
| 12 | 4,481 | 9.9258 | −0.029 | 9/13 | 0.0416 | 0.0222 | 0.0065 |
| **8** | **3,025** | **9.9180** | **−0.037** | **9/13** | 0.0344 | 0.0200 | 0.0082 |
| 5 | 1,911 | 9.9280 | −0.027 | 7/13 | 0.0299 | 0.0188 | 0.0095 |
| 3 | 1,155 | 9.9804 | +0.026 | 4/13 | **0.0240** | **0.0176** | 0.0122 |

*(role-graded `rho`, no season term. `body err` is the mean absolute error at the 41- and
60-game thresholds, which sit in the body of an 82-game distribution rather than its tails.)*

**1. The window replicates, and it has a genuine interior optimum.** CRPS falls to a minimum
at a **lookback of 8 seasons** and rises again at 5 and 3 — a bias-variance curve, not a
monotone preference for recency. The win is consistent rather than pooled: 9 of 13 origins.

**2. Role-graded `rho` replicates at every single lookback**, in both season-term specs —
5/5 and 5/5. It is the most robust finding here.

**3. Calibration and accuracy want different windows, and that is the finding to build on.**
PIT KS and boundary error improve **monotonically** as the window shortens, all the way to a
3-season lookback, while CRPS turns around at 8. The mean function wants rows; the
dispersion and the shape want *recent* rows. One window is being asked to serve two
estimands with opposite optima.

**4. The season trend is a COVID correction wearing a trend's clothes.** Pooled over all
origins it *looks* like an improvement — CRPS 9.9087 against 9.9180 at lookback 8. Per
origin it is not:

| origin | 2015 | 2019 | 2020 | **2021** | median of 13 |
|---|---|---|---|---|---|
| trend − none CRPS | +0.058 | +0.025 | −0.050 | **−0.194** | **+0.0060** |

**The entire pooled gain is 2020 and 2021** — the COVID and Omicron seasons — and the median
origin is *hurt*. Excluding the two extremes the mean effect is +0.0045. It helps only where
the league moved abruptly, and it helps there by fitting a transient.

That is exactly why it fails on validation: fitted through 2021-22 it extrapolates that
transient forward into 2022-23 and 2023-24, where availability partially **recovered**
(P(full schedule) 0.011 → 0.021 → 0.033). The body blowout is the overshoot. The trend also
pays for its boundaries out of the body on the fitting half — body error 0.0184 against
0.0082 at lookback 8 — so the mechanism is the same at both origins; only the size of the
damage differs. **The verdict stands and now has a cause.**

**5. The defect itself replicates in the fitting half.** Every arm under-predicts P(GP<10)
(−0.012 to −0.019) and over-predicts a full schedule (+0.016 to +0.037). It is not a
validation artifact.

> **The rolling harness is now the selection instrument**, and that is worth more than any
> single row of the table. Anything tuned on 5,142 fitting-half rows across 13 origins
> leaves the 883 validation rows intact as a confirmation, which is the order these two
> readings should have been taken in.

### `rho`, specifically

The hypothesis this ladder was built to test was that **one `rho` shared across 25 seasons**
of a league whose availability shape changed was producing the boundary mass. That is
**largely falsified**: `rho` moves only from **0.2806** (full) to **0.2627** (2012-13) to
**0.2608** (2017-18) — **−6.4%** across windows whose mean `gp_share` differs by 0.09. The
dispersion is not what drifted.

What survives is the *other* pooling: one `rho` for every **player**. That is real, worth
grading, and too small to be the whole defect at 1.26–1.47×.

---

## 5. What this leaves, in order

1. ~~**Port `three_point_era__none__role` to `stan_availability.py`**~~ ✅ **Done
   2026-08-11.** `stan.availability` in `configs/default.yaml` carries `first_season:
   2012-13` and `role_rho: true`; `betabinomial_glm.stan` now takes `rho` as a vector
   indexed by a data-supplied bin, with `n_rho = 1` reproducing the shared-`rho` target bit
   for bit (asserted on Stan's own `log_prob`, so the four other heads on that file are
   untouched) — and setting `first_season: null` / `role_rho: false` refits the incumbent
   posterior to within Monte Carlo error, which is the rollback path exercised rather than
   asserted. The window cuts the head's **own fitting rows** inside `fit`, never
   `availability_design`.

   On validation the Stan posterior reads CRPS **9.8155** (plug-in 9.8136) against the
   incumbent's 10.0071, and the dispersion comes back fitted **jointly** rather than
   profiled: **0.3176** for `<12 mpg` against **0.2064** for `30+ mpg`, a **1.54×** spread
   where the point MLE on the same rows gives 0.3147 / 0.2142 and 1.47×. Both point MLEs,
   refitted on the same 4,027 windowed rows, reproduce this ladder to four decimals —
   **9.8444** and **9.8247** — which is the check that the head fits the arm selected here
   rather than something nearby.

   Two things came out differently from the plan and are worth carrying forward. **The joint
   fit trades a little calibration for a little sharpness**: it beats the profiled point
   estimate on CRPS (9.8136 against 9.8247) and loses to it on PIT KS (0.0679 against
   0.0588), grading `rho` slightly harder at both ends. And **the window has a price that is
   not CRPS** — whole-board shared-β spread rose from 222.8 to **297.2** games and board
   inflation from +6.7% to **+12.3%**, because 4,027 fitting rows leave a wider posterior on
   β than 9,478 do. That is more honest rather than worse, and it is still +0.5% on a 15-man
   roster, but every consumer of `stan_availability_board.csv` is now reading a number twice
   as large. The gates that hold this head as a floor (`season-total`'s Gate E,
   `stan-games-played`) have **not** been re-run yet.
2. **Do not ship a season trend.** Recorded as a null so it is not rebuilt. `season_terms`
   already selected `trend` for `gp` on a 0.011 CRPS margin and did not adopt it; this
   explains *why* that was right and adds the reason — it is the wrong shape of instrument.
3. **The low tail needs a likelihood change, not a covariate.** Candidates, none yet
   measured: a zero-or-few-games inflation component; a beta-binomial with a
   covariate-dependent `rho` rather than bucketed; or accepting that entry/exit is the real
   process and revisiting the tenure decomposition (`docs/games-played-plan.md`), whose
   oracle-tenure arm already scores **7.2265** against the incumbent's 10.0057.

   ✅ **Measured 2026-08-11 — see §7.** All five likelihood arms and the free arm are on the
   ladder. The tenure decomposition is a **null** (§7a), the low tail is a **missing
   component** rather than a wrong frailty shape (§7c results 1–2), and the divergence share
   moved without being the mechanism (§7c result 5). Nothing ships from there.

   **Sharpened 2026-08-11 and moved to `docs/potential-to-dos.md` item 5.** The mechanism is
   now measured rather than suspected: `a` and `b` are both functions of `(μ, ρ)`, so the
   frailty's boundary behaviour and its variance are the *same parameter*. `b < 1` — a
   density that diverges at `p = 1` — holds on **52.7%** of validation rows and on **82.2%**
   of the `24-30 mpg` bucket, because it is implied by `ρ > (1 − μ)/(2 − μ)` and the fitted
   `ρ` sits just above that threshold for every high-`μ` bucket. That is why moving `ρ` could
   only halve the miss, and it reframes the candidate list around likelihoods that free the
   boundary from the dispersion.
4. ~~**Selection multiplicity.**~~ ✅ **Closed** by §4b — 13 origins and 5,142 fitting-half
   rows. The window and the role-graded dispersion both replicate; the trend's apparent
   pooled gain turned out to be two COVID origins.
5. **Optimize the trade rather than picking a window.** §4b shows CRPS optimal at a
   lookback of 8 while PIT and boundary error keep improving to 3, so one window cannot
   serve both estimands. The candidates, in the order their evidence supports them:
   **(a)** fit `beta` on a long window and `rho` on a short one — the head already
   alternates between them, so this costs nothing structurally and is exactly what the
   divergent optima ask for; **(b)** replace truncation with **exponential season
   weighting** `w = lambda^(target - s)`, which is a smooth generalization of a step
   function with one continuous knob, tunable on the rolling harness; **(c)** shrink the
   short-window fit toward the long-window one instead of discarding rows, which in the
   Stan port is just the long-window posterior used as the prior; **(d)** let the drift sit
   where it actually is — probably the intercept and the role terms rather than the lag
   slopes — and window only those.
6. **A confound in this ladder, to fix before (a)-(d) are trusted.** `l2` is pinned at 1.0
   at every lookback, and a 1,155-row fit wants more regularization than a 6,630-row one.
   The short lookbacks are therefore under-regularized for their row count, so some of the
   turnaround at 5 and 3 is a penalty artifact rather than variance. Sweep `l2` jointly
   with the lookback on the rolling harness before reading the optimum as a fact.
7. **Recency and representativeness are different knobs.** 2019-20 through 2021-22 are
   recent *and* unrepresentative, and §4b shows they dominate the trend's apparent value. A
   weighting scheme indexed on age treats them as maximally relevant for the 2026-27 fit.
   An explicit regime indicator, or excluding them, is a separate axis from lookback and
   should be swept as one.

## 5b. Four optimizations, all measured, none confirmed

`make availability-weighting` → `availability_weighting.csv` and
`availability_weighting_confirmation.csv`. Built 2026-08-11 to act on §5's list. Selection
runs on the rolling harness (13 origins, fitting half only); the validation reading is taken
**once, at the end**, after the recipe is fixed.

### What each bought on the rolling harness

| experiment | best arm | CRPS | vs reference | origins won |
|---|---|---|---|---|
| `l2_by_lookback` | lookback 8, `l2` = 16 | 9.9146 | −0.0333 | 7/13 |
| `split_window` | `beta` 8, `rho` 5 | 9.9175 | −0.0304 | 7/13 |
| `decay` | `lambda` = 0.80 | 9.9155 | −0.0323 | 10/13 |
| **`block_window`** | **intercept + workload** | **9.8983** | **−0.0359** | **11/13** |

Three findings from that half, all of which stand as measurements:

**1. The 8-season optimum is real, not a penalty artifact.** With `l2` swept from 0.25 to
256 the interior optimum survives and sharpens — the best cell is (lookback 8, `l2` 16) and
every finite lookback's optimum is now interior rather than on a grid edge. Shorter windows
genuinely want more shrinkage: `all` prefers `l2` ≈ 1, lookback 8 and 5 prefer 16, lookback
3 prefers 64. **The correction I issued mid-session was wrong and the original reading was
right.** The arithmetic in that correction is real — a fixed penalty against a summed
log-likelihood does bite harder as N falls — but it is dominated by the plain fact that
fewer rows want more shrinkage.

**2. `beta` and `rho` each want a window, and the two effects are additive rather than
substitutable.** From (all, all) at 9.9478: windowing `rho` alone to 5 seasons buys −0.0139,
windowing `beta` alone to 8 buys −0.0139, and doing both buys −0.0303. But once `beta` is
windowed the extra gain from decoupling `rho` is **−0.0005** — nothing. Decoupling is only
worth it if you want to keep every row for the mean, where it recovers about half the
benefit for free.

**3. The drift is in the *level*, not the relationships.** This is the sharpest result in
the study. Splicing one coefficient block at a time from an 8-season fit into an
all-seasons fit, with `rho` held on the short window in every arm:

| block | columns | CRPS vs reference | 95% |
|---|---|---|---|
| intercept + workload | 5 | **−0.0359** | [−0.0535, −0.0187] |
| intercept | 1 | −0.0197 | [−0.0296, −0.0098] |
| all 20 columns | 20 | −0.0162 | [−0.0368, +0.0049] |
| workload | 4 | −0.0138 | [−0.0297, +0.0018] |
| `gp_share` | 3 | −0.0001 | [−0.0118, +0.0106] |
| `minutes` | 4 | +0.0050 | [−0.0058, +0.0172] |
| absence history | 4 | **+0.0146** | [+0.0043, +0.0258] |
| age curve | 4 | **+0.0172** | [+0.0078, +0.0267] |

**Five of twenty columns carry the whole effect, and windowing the other fifteen actively
hurts** — the age curve and the absence-history block are significantly *worse* on recent
seasons only. Windowing everything is worse than windowing five columns. The league moved
the level of availability and the workload relationship; how age and absence history predict
availability did not change.

### The validation confirmation — and it is a negative

| arm | val CRPS | PIT KS | err P(<10) | err P(full) |
|---|---|---|---|---|
| **`three_point_era` window + role `rho`** *(§4's arm)* | **9.8247** | **0.0588** | −0.0230 | +0.0126 |
| lookback 8 + role `rho` | 9.8289 | 0.0650 | −0.0242 | +0.0095 |
| lookback 8 + `l2` = 16 | 9.8328 | 0.0623 | −0.0249 | +0.0099 |
| decay `lambda` = 0.85 | 9.8641 | 0.0639 | −0.0237 | +0.0114 |
| decay `lambda` = 0.80 | 9.8662 | 0.0725 | −0.0229 | **+0.0075** |
| decay `lambda` = 0.90 | 9.8808 | 0.0656 | −0.0245 | +0.0163 |
| **block-windowed** *(the rolling winner)* | **9.9265** | 0.0644 | −0.0247 | +0.0099 |
| incumbent | 10.0057 | 0.0939 | −0.0290 | +0.0337 |

**None of the four optimizations beats the plain window that §4 already selected**, and the
arm that won the rolling harness most decisively is the *worst* of the challengers here.
Nothing new ships.

**The discordance has one explanation and it is the same one as the trend's.** Every
instrument here leans harder on recent seasons — a shorter effective window, a geometric
decay, an intercept fitted on eight seasons — and the most recent training seasons are the
COVID trough. Leaning on them over-corrects downward into two validation seasons that
partially recovered. The tail column shows the mechanism directly: decay 0.80 posts the
**best** full-schedule error of any arm (+0.0075) and a worse CRPS, which is over-correction,
not calibration. The 2012-13 window works because ten seasons dilute the trough; a decay with
an effective lookback of 5–7 does not.

**A second reason the block arm specifically fails: a spliced coefficient vector is not a fit
of anything.** The blocks are not orthogonal, so coefficients estimated on eight seasons are
co-adapted to *each other*, and dropping five of them into a vector estimated on twenty-five
breaks that. The principled version is hierarchical shrinkage toward the long-window
estimate — §5's candidate (c), still unbuilt — where the blocks are fitted jointly under
different priors rather than transplanted.

> **This also exposes a limit of the rolling harness.** Every origin in it is exactly **one**
> season ahead and sits inside the training half; validation is one *and two* seasons ahead
> and sits on the far side of a regime transient. So the harness cannot see the failure mode
> that matters most here, and it systematically prefers arms that lean recent. It remains the
> right instrument for multiplicity and power — it caught the trend's two-origin artifact —
> but it is not a substitute for the validation reading, and this section is the evidence.

---

## 6. The same question for the minutes heads — measured, not yet laddered

The marginal minutes head has the **same defect shape and a different break**. On the
`stan_minutes` unit (season minutes ÷ games × game length, rotation players, through
2023-24): cross-player sd **0.1918 → 0.1626** (−15.2%), and P(rate ≥ 0.75) — a ≥36 mpg
workhorse — **0.157 → 0.015**, a **10.4×** fall against a mean that moved only −7.4%. Its
break completes around **2014-15** and is flat afterwards, which leaves **eight** training
seasons after it against availability's five.

Two consequences, both untested:

- **The window ladder is better posed for minutes than for availability** — more post-break
  training rows, and a drift that has stopped rather than continued.
- **The head ships a year *random effect*, not a trend.** Per `season_terms`, a year effect
  is mean-zero at prediction time and contributes *variance*; only a trend moves location.
  So the minutes head currently has no instrument pointed at a location drift — and, given
  result 2 above, that is probably correct, but it has not been measured.

The stake is not marginal. `make minutes-unification` ships the marginal head **only** for
its season-level spread (predictive sd 302.75 against the composition's 64.65). If that
spread is averaged over a window whose cross-player sd contracted 15%, a short-window refit
should narrow the predictive and lower the injection σ the composition needs to tie it —
which would revise a shipped decision. Counterweight: the head's season-unit PIT KS is
0.0735 on validation, so it is not obviously too wide today.

The **composition** head is the era-stable one and is the lowest priority: its unit is a
*share* of a fixed pot, so league drift divides out before the head sees it. Team-game HHI
moves 0.1279 → 0.1161 across 1996-97 → 2023-24, about 9%, and has partially reverted since
2018-19.

> The minutes and composition series above use a rotation filter (`gp ≥ 20`, `mpg ≥ 10`)
> rather than those heads' own row filters, so they are directional. The 10.4× is far too
> large for a population definition to flip; the −15.2% is not, and should be rebuilt
> through each head's own design before anyone acts on it.

---

## 7. The fourth axis: the likelihood — measured 2026-08-11

§4 settled that no instrument on the *mean* closes the boundary, and §5.3 recorded why in
arithmetic rather than in hypothesis: under `a = μ(1−ρ)/ρ` and `b = (1−μ)(1−ρ)/ρ` the
frailty's **shape** and its **variance** are the same parameter, so `b < 1` — a Beta density
that diverges at `p = 1`, sitting exactly on "played every game" — is forced whenever
`ρ > (1−μ)/(2−μ)`. That is why moving `ρ` could only halve the miss.

So the fourth axis varies the **frailty**, holding window, season term and dispersion at the
shipped arm (`three_point_era`, `none`, `role`). It is deliberately **not** crossed with the
full grid: 19 arms was already the multiplicity problem §4b exists to answer, and crossing
would make it 95.

`make availability-window` → `availability_likelihood.csv`,
`availability_likelihood_rolling.csv`.

**Every arm reproduces the head it extends at its own nesting parameter values**, asserted on
the log-likelihood rather than written down. `π = 0`, `g = 0` and `θ = 0` are each a *finite,
attainable* parameter value rather than a limit — which is why `π` is `θ·σ(γ'z)` with `θ`
bounded rather than `σ(γ₀ + γ'z)`, and why the ordering constraint is a bounded increment
rather than `exp(s)`. Mixture weights sit on the boundary of their parameter space, and a
nesting check that has to be taken as `γ₀ → −∞` is a statement about floating point rather
than about the model. `logitnormal` is the one arm that does not nest the incumbent, by
design, and is pinned to the **binomial** instead. That is the `n_rho = 1` / `U_n = 0`
discipline, and the rollback path.

### 7a. The free arm first: the tenure decomposition loses on the boundary too

`stan_games_played` is the structural alternative to a frailty, and §5.3 nominated it because
it had never been scored on this statistic. Its composite pmf was already on disk, so the
comparison cost a file read. Scored on the **same 883 validation rows**, with the pre-window
incumbent reproducing `stan_games_played_gates.csv`'s **10.005738** Gate D bar exactly:

| arm | CRPS | vs `betabinom` [95%] | PIT KS | **boundary err** | body err |
|---|---|---|---|---|---|
| incumbent `full__none__shared` | 10.0057 | *(context)* | 0.0939 | 0.0314 | 0.0083 |
| shipped Stan head (2012-13 + role `ρ`) | 9.8156 | *(context)* | 0.0667 | **0.0202** | 0.0108 |
| **tenure decomposition** `duration_covariates` | **10.1625** | **+0.350 [+0.227, +0.473]** | 0.0672 | **0.0333** | 0.0102 |

*(the bootstrap column is against §7c's `betabinom` reference at CRPS 9.8125; the first two
rows are carried for context and are scored on the same 883 rows by the same code.)*

**A head that loses on the mean also loses on the boundary.** 0.0333 is the worst
`boundary_tail_error` of every arm measured in this doc — worse than the incumbent it was
built to structurally replace, by **+0.0132** with a paired interval of **[+0.0117, +0.0146]**,
clear of zero in the wrong direction. The direction of the errors is what makes it worth the
read: it makes the **same two errors on the same sides**, and the low one is *larger*.

| | observed | incumbent | shipped | tenure |
|---|---|---|---|---|
| err P(GP < 10) | 0.0815 | −0.0290 | −0.0247 | **−0.0328** |
| err P(GP = full) | 0.0272 | +0.0337 | +0.0159 | **+0.0339** |

Its PIT KS is as good as the shipped head's, so it is a well-calibrated distribution that is
still wrong at both ends, and its **body** is the one thing it wins (−0.0019 at 41 games).

> ⚠️ Two corrections to the motivation that nominated it. The **7.2265** §5.3 quotes is the
> `within_tenure` arm, flagged `oracle_tenure: True, selectable: False` in
> `stan_games_played.py:151` — it holds tenure at its **observed** value and covers 751 rows,
> so it is not a forecast and was never a candidate. The forecastable arm is
> `duration_covariates`. And "a head that loses on the mean wins on the boundary" was the
> hypothesis; the measurement says it loses on both.

**The defect surviving a change of generative structure is the useful half.** An entry ×
exit × two-state-chain model with beta-geometric spells is not a beta-binomial in any
respect, and it reproduces the beta-binomial's two boundary errors in direction and
magnitude. That is evidence the missing ingredient is a **component**, not a frailty shape —
which is what the ladder below then separates directly.

### 7b. The five likelihoods

| arm | frees | nests the incumbent at |
|---|---|---|
| `betabinom` | nothing — the reference row | — |
| `mixture` | the low tail, as a separate **event**: `π_i·BetaBinom(μ_low, ρ_low) + (1−π_i)·BetaBinom(μ_i, ρ_i)`, with covariates on `π` | `θ = 0` |
| `finite_mix` | the whole shape — K latent durability classes with **ordered** mean offsets | `g = 0` |
| `logitnormal` | boundary behaviour — a binomial GLMM, whose frailty *cannot* diverge | **does not** (nests the binomial) |
| `beta_rect` | tail mass, symmetrically — `θ·U(0,1) + (1−θ)·Beta`, the one-parameter control | `θ = 0` |

Two implementation notes that changed what the arms could find.

**`finite_mix`'s components are beta-binomials, not binomials, and the nesting rule forced
it.** The pure Heckman–Singer device puts K support points in place of the continuous mixing
distribution, and at K = 1 that is a *binomial* rather than the incumbent. Beta-binomial
components with ordered offsets nest the incumbent exactly and contain the pure version as
the `ρ → 0` corner of the same family — so the fitted `ρ` becomes the readout rather than an
assumption.

**Multi-start is not optional here, and skipping it produced a false null.** Started at its
own nesting point, a three-class mixture sat on the bound, reported success, and reproduced
the incumbent to four decimals — because at `g = 0` every class carries identical
responsibility and the surface is flat in exactly the direction that separates them. With
separated starts the same arm finds a structure worth 21.9 log-likelihood points. The
`start_loglik_spread` column records this per arm: **21.9** and **29.3** for `finite_mix` at
K = 3 and K = 4, against **0.007** for `mixture` and **0.002** for `beta_rect`. The finite
mixtures are genuinely multimodal; the other two are not.

### 7c. The ladder

Observed on validation: P(<10) **0.0815**, P(full) **0.0272**, P(<41) 0.2911, P(<60) 0.5300.
The selector is `boundary_tail_error`; `body_error` is reported beside it and **never
averaged in**, per §3.

| arm | params | CRPS | vs `betabinom` [95%] | PIT KS | **boundary err** | body err | diverges at `p=1` |
|---|---|---|---|---|---|---|---|
| `betabinom` *(reference, joint)* | 24 | 9.8125 | — | 0.0667 | 0.0201 | 0.0107 | **52.1%** |
| `betabinom_two_stage` *(§4's arm)* | — | 9.8247 | +0.012 [−0.017, +0.041] | 0.0588 | 0.0178 | 0.0203 | — |
| **`mixture`** | 35 | 9.8237 | +0.011 [−0.028, +0.051] | 0.0631 | **0.0109** | 0.0047 | 37.8% |
| `beta_rect` | 25 | **9.7561** | **−0.056 [−0.096, −0.015]** | 0.0599 | 0.0165 | 0.0032 | 35.3% |
| `finite_mix` (K=3) | 28 | 9.7667 | −0.046 [−0.076, −0.014] | 0.0620 | 0.0181 | **0.0012** | 32.4% |
| `finite_mix` (K=4) | 30 | **9.7390** | **−0.074 [−0.106, −0.039]** | 0.0577 | 0.0169 | 0.0016 | 34.6% |
| `finite_mix` (K=2) | 26 | 9.8114 | −0.001 [−0.004, +0.002] | 0.0667 | 0.0201 | 0.0094 | 49.4% |
| `logitnormal` | 24 | 9.8616 | +0.049 [−0.057, +0.155] | 0.0801 | 0.0195 | 0.0116 | **0.0%** |
| `tenure_decomposition` | — | 10.1625 | +0.350 [+0.227, +0.473] | 0.0672 | 0.0333 | 0.0102 | — |

*(`diverges at p=1` is the share of predictive **mass** sitting under a frailty with `b < 1`,
which is the same statistic as §5.3's row share on a one-component head and a real one on the
alternatives. K=2 and K=4 are a sensitivity on K, not competitors for selection.)*

**The selector gets an interval too**, because a boundary margin quoted bare is the thing
this project calls a prompt rather than a finding. `boundary_tail_error` is a *non-linear*
statistic — two absolute values of differences of means — so the bootstrap resamples rows and
recomputes it rather than averaging a per-row score, paired inside the row the way the CRPS
one is:

| arm | boundary err | vs `betabinom` [95%] | clears |
|---|---|---|---|
| `betabinom` | 0.0201 [0.0100, 0.0294] | — | — |
| **`mixture`** | **0.0109** [0.0030, 0.0203] | **−0.0089 [−0.0099, −0.0042]** | ✅ |
| `beta_rect` | 0.0165 | −0.0035 [−0.0041, −0.0019] | ✅ |
| `finite_mix` (K=3) | 0.0181 | −0.0020 [−0.0023, −0.0017] | ✅ |
| `logitnormal` | 0.0195 | −0.0004 [−0.0082, **+0.0108**] | ❌ |
| `tenure_decomposition` | 0.0333 | **+0.0132 [+0.0117, +0.0146]** | ❌ *(worse)* |

`logitnormal` is the only fitted arm whose interval spans zero, and the tenure
decomposition's is clear of zero **in the wrong direction** — both of which are the point
rather than an inconvenience.

**The reference row is refitted jointly, and that is a finding in itself.** Every arm here is
a joint MLE of its own likelihood, while `RoleGradedBetaBinomial` fits `β` under a shared `ρ`
and then profiles `ρ` per bucket holding the mean fixed. Scoring a profiled reference against
joint alternatives would confound the likelihood with the estimator, so the reference was
refitted jointly — and **joint estimation of the same model is worth −0.012 CRPS for free**
(9.8125 against 9.8247), landing beside the Stan port's 9.8136 plug-in. §5.1 recorded that
gap as something "the joint fit" bought and attributed it to the Bayesian fit; it is the
estimator, and the point MLE reproduces it.

**Six results.**

**1. A missing component beats a different shape, and one arm was built to prove it either
way.** `logitnormal`'s frailty *cannot* diverge — its density vanishes at both ends — so if
the boundary mass were a shape defect it should fix the high tail by construction and worsen
the low one. **That is exactly what happened.** It is the only arm whose full-schedule error
changes *sign* (−0.0072, where every Beta arm over-predicts), it posts the worst low-tail
error of any fitted arm (−0.0319 against the reference's −0.0253), and its
`boundary_tail_error` barely moves (0.0195 against 0.0201) because the two cancel. It is also
a genuinely **worse fit of the same data** — training log-likelihood **−16,331.5** against
the reference's −16,239.2 at the *same* parameter count (20 coefficients + 4 dispersions in
both). Both are log-likelihoods of the same counts under the same dominating measure, so
that is a like-for-like 92-point loss. Removing the divergence does not close
the boundary; it trades one end for the other.

**2. So the low tail is not a frailty phenomenon, and `mixture` is the arm that says so.**
Giving the disrupted season its own component halves the selector: `boundary_tail_error`
**0.0109** against 0.0201, with the low-tail error going −0.0253 → **−0.0132** and the
full-schedule error +0.0150 → **+0.0085**. It posts the best training log-likelihood of every
arm (**−16,174.5**, a gain of **64.6** over the reference) and it is a **tie** on CRPS
(+0.011, interval spanning zero). What it fits is interpretable: `θ = 0.112` with a mean
`π` of **4.9%**, a low component centred at `μ_low = 0.0999` — about **8 games of 82**, and
not at its bound — and `ρ_low = 0.044`. An Achilles rupture in October is a different event,
not an extreme draw of a per-game rate, and the head can now say so.

**And `π`'s covariates carry real signal**, which is the arm's own distinguishing claim: it
can say *who* is at risk, where a wider frailty can only say that someone is. `π` runs from
**1.2%** at the 10th percentile of players to **10.8%** at the 90th, a **8.8×** spread on
age, prior absence and playoff workload. A flat `π` would have made this a two-component
mixture with a constant weight — i.e. `finite_mix` at K = 2, which buys nothing.

**3. The one-parameter control wins CRPS — but only on the metrics that cannot see the
shoulder.** `beta_rect` adds a single uniform component and beats the reference by
**−0.056 CRPS** [−0.096, −0.015] while improving PIT (0.0599), the boundary (0.0165) and the
body (0.0032). It costs **one** unpenalized parameter against `mixture`'s eleven, and it
beats `mixture` on CRPS while losing to it on the boundary. Read together with result 2:
*tail mass* is what CRPS wants, and *which* tail is what the boundary wants. A symmetric
hedge buys the first and only half of the second.

> ✅ **The extra parameter is not what buys it — swept 2026-08-11, see §7d.** Refitting the
> reference at each of eight penalties from 0 to 256 moves it by **0.00034** CRPS at its best,
> so **99.4%** of this −0.056 survives a reference regularized as favourably as the grid
> allows, and matching every arm at its own optimum makes the margin marginally larger rather
> than smaller. The pinned penalty is 1.5 parts in 100,000 of the objective at these row
> counts. So this row is the likelihood's, and the qualification below is the one that
> stands.

> ⚠️ **Qualified by §7f.** Flat mass everywhere also pushes the 71–81 error from +0.0186 to
> **+0.0273**, so `beta_rect` is the *worst* Beta arm on `shoulder_error` (0.0290 against the
> reference's 0.0253). Its CRPS win is real and its calibration win is confined to the two
> regions §3's metric set happened to measure.

**4. Three durability classes, and the third is the ceiling.** `finite_mix` at K = 2 is the
incumbent to within noise (−0.001, interval spanning zero) — two classes buy nothing. K = 3
finds offsets **0.000 / 1.690 / 5.097** at weights **0.082 / 0.910 / 0.008**: a fragile 8%, a
typical 91%, and a vanishing iron-man class. K = 4 scores best on CRPS (−0.074) and its
fitted structure is K = 3 relabelled — offsets **0.000 / 2.185 / 2.187 / 5.235** at weights
**0.040 / 0.000 / 0.951 / 0.009**, with two classes collapsed onto each other and one carrying
zero weight. **Its extra CRPS is not extra structure**, which is the reason K is reported as
a sensitivity rather than swept for a winner.

**5. The shape diagnostic moved, and it is not the operative variable.** The reference puts
**52.1%** of predictive mass under a divergent frailty — reproducing §5.3's 52.7% scratch
measurement on the Stan posterior, which is the check that the two readings are of the same
thing. Every arm that improved anything moved it down by 15–20 points of mass (`finite_mix`
32.4%, `beta_rect` 35.3%, `mixture` 37.8%), and every one of them also pulled `ρ` down at
every role bucket — the reference's **0.3167 / 0.2690 / 0.2523 / 0.2056** becomes
`finite_mix`'s 0.2855 / 0.2217 / 0.1950 / 0.1484. So the added component *absorbs* dispersion,
which is what "the classes are doing the frailty's job" looks like. But `ρ` does not collapse
toward zero, so the shoulders want a continuous frailty **and** a discrete one — and the arm
that eliminates divergence entirely is the arm that fails. **The divergence is a symptom of
the single shape knob, not the mechanism of the miss.**

**6. No arm repeats the season trend's failure, so this axis is not a null.** §4's warning
was that the arms with the best boundary coverage were the worst models — a location shift
closes both tails and blows out the body. Nothing here does that: every arm that improves the
boundary **also** improves the body, `mixture` from 0.0107 to 0.0047 and `finite_mix` to
0.0012. The trap the ladder was built to detect did not fire.

### 7d. The confound, measured — and it is a null

**`l2 = 1.0` is pinned across arms so the contrast is the likelihood alone, and a fixed
penalty is not neutral between them.** The penalty reaches `β[1:]` only, so `mixture` carries
**eleven** extra unpenalized parameters against the incumbent's zero, `finite_mix` four, and
`beta_rect` one. Every margin in §7c was therefore recorded as an **upper bound** on the
likelihood's own contribution. §5.6 records the same confound one axis over, where `l2` was
pinned across lookbacks of very different row counts and turned out to matter.

✅ **Swept 2026-08-11. It does not matter here, and the margins survive essentially whole.**
`make availability-window` → `availability_l2_sweep.csv`, `availability_l2_verdict.csv`:
every arm refitted at each of **eight** penalties from 0 to 256 at the shipped window, with
the grid **anchored at 0** — the unpenalized MLE — so the reference's optimum cannot sit on a
low edge with "sweep further down" still available as a move.

| `l2` | `betabinom` | `beta_rect` | `finite_mix` | `mixture` |
|---|---|---|---|---|
| **0** | **9.8122** | **9.7554** | **9.7659** | **9.8232** |
| 0.0625 | 9.8122 | 9.7554 | 9.7660 | 9.8232 |
| 0.25 | 9.8123 | 9.7556 | 9.7661 | 9.8235 |
| 1 *(pinned)* | 9.8125 | 9.7561 | 9.7667 | 9.8237 |
| 4 | 9.8134 | 9.7574 | 9.7684 | 9.8258 |
| 16 | 9.8146 | 9.7598 | 9.7722 | 9.8290 |
| 64 | 9.8235 | 9.7713 | 9.7862 | 9.8423 |
| 256 | 9.8663 | 9.8223 | 9.8580 | 9.8918 |

**1. The reference gains 0.00034 CRPS from the most favourable penalty it can be given.**
Its optimum is at `l2 = 0` — 9.812192 against 9.812533 at the pinned 1.0. So `beta_rect`'s
margin against a reference regularized as favourably as the grid allows is **−0.056052**
[−0.096, −0.015] against the pinned **−0.056392**: **99.4% of it survives**. `finite_mix`
keeps 99.3%, and `mixture` remains a tie either way (+0.0112 against +0.0112). Matching every
arm at its own optimum instead — the like-for-like reading — moves `beta_rect` to −0.056822,
i.e. the margin gets marginally *larger*. **Every ordering in §7c stands unchanged.**

**2. The reason is arithmetic, and it is why this was worth ten minutes rather than a
session.** At the fitted solution `β[1:]·β[1:]` is **0.2418** for the reference, against a
training log-likelihood of **−16,239.2**. So the pinned penalty is **1.5 parts in 100,000**
of the objective — the features are standardized and the coefficients are already shrunk by
4,027 rows of data, so `l2 = 1` is numerically almost no penalty at all. It takes `l2 = 64`
before any arm moves by as much as the *smallest* margin in §7c, and by then every arm has
moved together and the ordering is still the same.

**3. Which is not the same answer §5b got, and the difference is the axis.** There, sweeping
`l2` jointly with the *lookback* mattered — 1,155-row fits preferred `l2 = 64` and 8-season
fits preferred 16. Here the row count is held fixed at the shipped window, so the penalty has
nothing to trade against. **A penalty matters when the row count is the axis, and not when
the likelihood is.**

**So §7c's margins are the likelihood's own**, and the caveat that the arms are unequally
advantaged is now a measured 0.6% of one of them rather than an unbounded upper bound. What
it does *not* do is change any verdict: `mixture` was selected on calibration with a CRPS
guard, and both halves of that read identically at every penalty on the grid.

### 7e. Confirmation on the rolling harness — and only one finding survives it

Eight arms on 883 validation rows is the same multiplicity and power problem §4b was built
for, so the axis goes through the same instrument: 13 origins across the fitting half, each
arm fitted on the 8 seasons before the origin and scoring the origin season itself, **5,142
rows and not one of them a validation or held-out row**. Lookback 8 is §4b's interior CRPS
optimum and the closest fitting-half analogue of the 2012-13 window — an absolute first
season means nothing at a 2011 origin.

| arm | CRPS | vs `betabinom` [95%] | origins won | PIT KS | **boundary err** [95% vs ref] | body err |
|---|---|---|---|---|---|---|
| `betabinom` *(reference)* | 9.9040 | — | — | 0.0361 | 0.0209 | **0.0042** |
| `beta_rect` | **9.8985** | −0.0055 [−0.0212, +0.0102] | **9/13** | 0.0344 | 0.0170 [−0.0038: −0.0040, −0.0037] | 0.0052 |
| **`mixture`** | 9.9031 | −0.0009 [−0.0170, +0.0162] | 7/13 | 0.0356 | **0.0126** [**−0.0083: −0.0085, −0.0080**] | 0.0090 |
| `finite_mix` (K=3) | 9.9051 | +0.0011 [−0.0093, +0.0116] | 8/13 | 0.0339 | 0.0201 [−0.0008: −0.0009, −0.0007] | 0.0055 |
| `logitnormal` | 10.0534 | **+0.1494 [+0.0970, +0.2047]** | 1/13 | 0.0479 | 0.0114 [−0.0095: −0.0157, −0.0034] | 0.0156 |

**1. The CRPS wins do not replicate.** On validation `beta_rect`, `finite_mix` and
`finite_mix_k4` all beat the reference with intervals clear of zero (−0.056, −0.046, −0.074).
On 5.8× the rows their margins shrink by roughly an order of magnitude and every interval
spans zero — and `finite_mix`'s **sign flips**. `beta_rect` is the most consistent of them at
9 of 13 origins, but −0.0055 CRPS is not the −0.056 the validation row reads. **Read the
validation CRPS column as one draw, not as a result.**

**2. The boundary result does replicate, and it is `mixture`'s.** 0.0209 → **0.0126** is a
**−40%** cut against validation's −46%, and the two paired intervals overlap squarely:
**−0.0083 [−0.0085, −0.0080]** here against −0.0089 [−0.0099, −0.0042] there. Same arm, same
direction, same size, from disjoint rows — the only figure in this section that reproduces
across the two readings. Its CRPS is a tie in both (−0.0009 here, +0.011 there).
`beta_rect`'s smaller boundary gain also reproduces almost exactly (−19% here against −18%
on validation). `finite_mix`'s effectively does **not**: −0.0008 against validation's −0.0020,
an order of magnitude below `mixture`'s and 4% of the reference's error. So of the three arms
that improved the selector on validation, one is a solid effect, one is a modest one, and one
is real but negligible.

**3. `logitnormal` is the second clean instance of §4's trap, and §7f says exactly what it
does with the mass.** It posts the *best* boundary error of any arm here (0.0114), and
unlike on validation the gain is **significant** — −0.0095 [−0.0157, −0.0034]. It pays for it
with the **worst** CRPS by a factor of thirty (+0.1494, interval nowhere near zero), the worst
PIT, the worst body error (0.0156, 3.7× the reference's), and 1 of 13 origins. It is again the
only arm whose full-schedule error is **negative** (−0.0090). **An arm can buy the boundary
outright and still be the worst model on the table** — which is exactly what the season trend
did in §4, and exactly why `body_error` is reported beside the selector rather than averaged
into it. Had `boundary_tail_error` been the only column, this arm would have won the axis.
§7f identifies the mechanism: it cannot put mass at exactly 82, so it relocates it into
71-81, where its error is **+0.0621** against the reference's +0.0186. The boundary
"improvement" is mass moved a few games down the schedule, and a metric with a point mass on
one side and nothing beside it cannot tell the two apart.

**4. On the harness, the boundary is bought partly out of the body — which it was not on
validation.** The reference has the *best* body error here (0.0042) and every alternative is
worse; `mixture` goes 0.0042 → 0.0090. On validation every alternative *improved* the body.
The trade is real but small, and it is the honest qualification on result 2.

> The limit §5b recorded applies unchanged: every origin here is exactly **one** season ahead
> and sits inside the training half, while validation is one *and two* seasons ahead and on
> the far side of a regime transient. The harness answers multiplicity and power; it does not
> replace the validation reading. What it did here is what it did to the season trend —
> separate a robust effect from a lucky one, in both directions.

### 7f. The metric set was asymmetric, and the upper shoulder was the larger miss

Added 2026-08-11, after the axis had already been scored. The defect in §1 is stated as
"too little in the shoulders at 2–15 and **70–80** games", and **nothing in this ladder ever
measured 70–80**:

| region | what was measured before |
|---|---|
| `GP = 0` | nothing — folded into `below_10` |
| `GP < 10` | ✅ the low half of `boundary_tail_error`, a ten-game-wide shoulder |
| `GP < 41`, `< 60` | ✅ `body_error` |
| **60 → 81** | **nothing at all** |
| `GP = team_games` | ✅ the high half — but a **single point mass** |

So the selector was asymmetric: a wide region on one side against one point on the other.
Three additions fix it, and all three are reported beside the existing summaries rather than
folded into them.

**1. Upper thresholds counted in games missed.** `missed ≤ 11` and `≤ 5` are ">70 of 82" and
">76 of 82" — and unlike `gp > 70` they are the *same event* in a 66-game season, which is
the same schedule-invariance argument the `full_schedule` threshold already made.

**2. Exclusive bands** — `zero` / `1–9` / `missed 1–11` / `missed 0` — because a cumulative
lets errors of **opposite sign inside one tail cancel**, and §1 measured exactly that: the
head over-predicts `P(GP ≤ 1)` while under-predicting `P(GP < 10)`.

**3. A localized shape distance**, `low_shape_ks` / `high_shape_ks`: the largest gap anywhere
on the calibration curve within 15 games of each end. A band is one number per region and
can be right on average while the distribution *inside* it is the wrong shape.

#### What it found on the head that ships

| region | predicted | observed | error |
|---|---|---|---|
| `GP = 0` | 0.69% | **0.00%** | **+0.0069** |
| `GP` 1–9 | 5.00% | 8.15% | **−0.0316** |
| **missed 1–11** (71–81 games) | 24.29% | 22.42% | **+0.0187** |
| `GP` = full | 4.30% | 2.72% | +0.0158 |
| **missed ≤ 5** (77+ games) | **16.92%** | **12.12%** | **+0.0480** |

**The upper shoulder is a bigger miss than the upper boundary** — +0.0187 across 71–81 games
against +0.0158 at exactly 82, and **+0.0480** cumulatively at 77+ games, three times the
boundary error. `high_shape_ks` is **0.0480** against `low_shape_ks`'s 0.0252, so the
worst-calibrated region of the whole distribution is 77–82 games, and no metric in §3's set
pointed at it. And the low tail's two halves do err in opposite directions: **+0.69 pp** at
exactly zero — an event that occurs **zero times** in 883 validation rows — against
**−3.16 pp** at 1–9. `below_10` was netting them to −0.0247.

#### Three things this changes

**1. The window looks better, not worse.** Its largest single effect is on the region nobody
was measuring: `high_shape_ks` **0.0744 → 0.0402** and `missed ≤ 5` error 0.0744 → 0.0402, a
46% cut. The shipped decision is *more* justified than the metrics it was made on showed.

**2. A single threshold can be perfect while the shape is worse — measured, not argued.**
`three_point_era__trend__role`, §4's null, gets `missed ≤ 5` error to **−0.00006** — exact —
while its `high_shape_ks` is **0.0547**, *worse* than the non-trend arm's 0.0402. It nails one
point on the upper curve and undershoots badly at missed 6–15, because a location shift has
to overshoot somewhere. That is the clearest argument in this doc for a shape distance over
one more threshold.

**3. It reorders §7c, and against the arm that won CRPS.**

| arm | boundary | **shoulder** [95% vs ref] | point mass | `low_shape_ks` | `high_shape_ks` |
|---|---|---|---|---|---|
| `betabinom` *(reference)* | 0.0201 | 0.0253 | 0.0109 | 0.0258 | 0.0469 |
| **`mixture`** | **0.0109** | **0.0235** [−0.0021: −0.0086, −0.0010] ✅ | **0.0068** | **0.0132** | 0.0414 |
| `beta_rect` | 0.0165 | **0.0290** [+0.0034: −0.0030, +0.0042] ❌ | 0.0083 | 0.0242 | 0.0421 |
| `finite_mix` (K=3) | 0.0181 | 0.0274 [+0.0019] ❌ | 0.0114 | 0.0250 | 0.0410 |
| `finite_mix` (K=4) | 0.0169 | 0.0269 [+0.0014] ❌ | 0.0121 | 0.0222 | 0.0414 |
| `logitnormal` | 0.0195 | **0.0480** [+0.0223: +0.0158, +0.0246] ❌ | 0.0046 | 0.0319 | 0.0592 |
| `tenure_decomposition` | 0.0333 | 0.0215 [−0.0028: −0.0148, +0.0152] | 0.0170 | 0.0344 | 0.0538 |

**On validation, `mixture` is the only fitted arm that improves the shoulders**, and the only
one that improves *every* regional metric while tying on CRPS. Its `low_shape_ks` of
**0.0132** against 0.0258 is the largest single calibration gain on the table.

**`beta_rect`'s CRPS win comes with a shoulder regression here.** A uniform component adds
flat mass *everywhere*, which buys CRPS and the boundary and pushes the 71–81 error from
+0.0186 up to **+0.0273** — the worst of the Beta arms. The `finite_mix` arms do the same
thing more mildly. So "the one-parameter control wins" is true only on the metrics that
cannot see the shoulder, and §7c result 3 should be read with that beside it.

#### The rolling harness, and one sign that does not replicate

| arm | shoulder err | vs `betabinom` [95%] | err band 71–81 |
|---|---|---|---|
| `betabinom` *(reference)* | 0.0214 | — | **−0.0222** |
| **`mixture`** | **0.0078** | **−0.0128 [−0.0138, −0.0064]** ✅ | −0.0046 |
| `beta_rect` | 0.0152 | −0.0061 [−0.0064, −0.0057] ✅ | −0.0113 |
| `finite_mix` (K=3) | 0.0205 | −0.0009 [−0.0011, −0.0007] ✅ | −0.0195 |
| `logitnormal` | 0.0289 | +0.0075 [−0.0043, +0.0193] ❌ | **+0.0421** |

**1. `mixture`'s shoulder win replicates and is larger here** — a **60%** cut against
validation's 8%, on 5.8× the rows, with an interval nowhere near zero. Of everything measured
on this axis it is the most robust single effect.

**2. `beta_rect`'s shoulder effect does not replicate *in sign*, and the reason is
instructive.** The reference **under**-predicts the 71–81 band on the fitting half (−0.0222)
and **over**-predicts it on validation (+0.0186). Flat mass moves that band one way, so it
helps where the reference is short and hurts where it is long. `mixture` improves it in
**both** readings despite the reference's error pointing in opposite directions — reducing a
magnitude rather than pushing a level. That is the difference between a hedge and a component,
and it is the strongest argument on this axis for `mixture` over `beta_rect`.

**3. The fitting half shows the divergence mechanism more cleanly than validation does.**
There the head over-predicts the exact iron man (+0.0252) *and* under-predicts the 71–81 band
(−0.0222) — mass piled on the boundary and pulled out of the shoulder immediately below it,
which is what a Beta density with `b < 1` does, drawn to scale. On validation both errors are
positive because the level is off as well. §1's U-shape is a fitting-half property first.

**4. `logitnormal`'s mass relocation replicates exactly.** It is the only arm that pushes the
71–81 band *positive* (+0.0421 against a reference of −0.0222), the same direction and the
same mechanism as its +0.0621 on validation.

**And the shape metric supplies `logitnormal`'s mechanism.** Its high-shoulder error is
**+0.0621** against the reference's +0.0186 — 3.3×, the worst on the table — while its
full-schedule error is the only *negative* one. It cannot put mass at exactly 82, so it piles
it into 71–81. **The "best boundary error of any arm" the rolling harness credited it with in
§7e was mass relocation into the shoulder, not calibration** — which is what a metric with a
point mass on one side and nothing beside it will always miss.

### 7g. What this settles, and what it does not

**Settled.**

- **The tenure decomposition is not the answer** (§7a). A null, recorded so it is not rebuilt.
- **The low tail is a missing component, not a wrong frailty shape.** The arm that adds a
  component halves the boundary error in both readings, on overlapping intervals from
  disjoint rows; the arm that removes the divergence entirely fails the other way in both.
  Those two facts together are what no single arm could have established, and they are why
  `logitnormal` was fitted despite being expected to lose.
- **The selector needs the body column beside it, demonstrated a second time.** On the
  rolling harness `logitnormal` posts the largest and a *significant* boundary gain of any
  arm and is simultaneously the worst model on the table. §4 found that pattern in the season
  trend; this is the independent replication of it, in a different instrument.
- **The divergence is a symptom, not the mechanism.** 52.1% of predictive mass → 32–38% on
  the arms that helped, `ρ` falling at every bucket without collapsing, and the arm at 0%
  being the worst model — which §7f resolves into a mass-relocation story rather than a
  calibration one.
- **The metric set was asymmetric and is no longer** (§7f). The upper shoulder turned out to
  be a *larger* miss than the upper boundary it was pooled next to, and it was the region the
  window helped most. A metric added after the fact is also what found the one overclaim in
  §4 result 3.
- **The pinned `l2` is not doing the work** (§7d). Swept over eight penalties from 0 to 256,
  the reference's best is worth **0.00034** CRPS and **99.4%** of `beta_rect`'s margin
  survives it, because at 4,027 rows the penalty is 1.5 parts in 100,000 of the objective.
  §7c's margins are the likelihood's own. The contrast with §5b — where `l2` mattered a great
  deal — is that a penalty matters when the **row count** is the axis and not when the
  likelihood is.
- **Joint estimation of the shipped head is worth −0.012 CRPS for free**, and §5.1's
  attribution of that gap to the Bayesian fit was wrong about the cause.

**Not settled.**

> ✅ **Two of the four below closed on 2026-08-12** — the likelihood question and the cost of
> a port — and §7h is the measurement. `mixture` is selected under D1, ported behind
> `stan.availability.mixture`, and its verdict survives the port with the calibration margins
> attenuated but still clear of zero. The two that remain are the last two.

- ~~**Which likelihood**~~ ✅ **Settled 2026-08-12** — `mixture` ships (`D1`, §7h). Kept
  below because the *argument* is the useful part, and because it records what the decision
  cost: adopting it means the objective for this head is tail calibration with a CRPS guard,
  which is a change of rule and not a metric. **`mixture` is the only arm** that improves
  **every** regional metric (boundary, shoulders, point masses, body) while
  tying on CRPS, the only one whose selector win replicates on the rolling harness, and the
  only one whose **shoulder** win replicates *in sign* — it reduces the magnitude of that
  error on both readings, where the reference's own error points in opposite directions.
  `beta_rect` wins CRPS and moves the shoulder by pushing a level, so it helps on one
  reading and hurts on the other; `logitnormal` is a measured failure. Whether a
  CRPS-neutral calibration win is a shipping criterion for this head was the judgement call
  the ladder could not produce a number for, and `docs/availability-mixture-ship-plan.md` D1
  made it — **stated before the arm it admits was ported**, which is the point of writing a
  rule down rather than inferring it from the arm that happened to win.
- ~~**What a Stan port would cost.**~~ ✅ **Measured 2026-08-12** (§7h). It cost a covariate
  block and an additive correction in `betabinomial_glm.stan` — which six other heads share,
  and which `P = 0` and `θ = 0` both leave bit-for-bit unchanged — plus **3.9×** the sampler
  time. The prediction that this was "a session's work rather than a parameter" was right.
  What the estimate missed is that the port **attenuates** the win it ports: 89% of the
  boundary margin and 62% of the shoulder margin survive, so the shipped head is a slightly
  weaker version of the arm that was selected.
- ~~**Whether the pinned `l2` is doing the work** (§7d).~~ ✅ **Closed 2026-08-11** — a null,
  moved to the settled list above.
- **The exchangeable-trials assumption**, which none of these arms touches. Absences come in
  *spells* — beta-geometric, beating the geometric by 11,278 log-likelihood points at one
  extra parameter — and one 40-game spell and forty single-game absences give identical `gp`
  and very different distributions. A beta-binomial absorbs the variance inflation from that
  clustering but not its shape, and neither does any arm above.

### 7h. The Stan port — measured 2026-08-12, and it reproduces the ladder

`make stan-availability-mixture` → `stan_availability_mixture.csv`,
`stan_availability_mixture_parameters.csv`, `stan_availability_mixture_chains.csv`,
`stan_availability_mixture_diagnostics.csv`.

The mixture is now in `src/stan/betabinomial_glm.stan` behind `P`, the number of covariates
on `π`, and in `stan_availability.StanAvailability` behind `stan.availability.mixture`.
**Both nestings are exact and both are asserted on Stan's own `log_prob`**
(`tests/test_stan_heads.py`): `P = 0` makes `θ`, `μ_low`, `ρ_low` and `γ` zero-length, which
is the parameter space the other five heads on that file have always had; and with `P > 0`,
`θ = 0` leaves the target **bit for bit** identical — equality on doubles, not a tolerance.
That is possible because the mixture enters as an *additive correction* to the untouched
beta-binomial statement rather than replacing it, and the correction is skipped entirely at
`θ = 0`. `θ > 0` is the exact test rather than a proxy, since `π_i = θ·σ(z_i'γ)` and `σ` is
strictly positive, so `π` is zero on every row or on none.

**The point MLE refitted on the same rows reproduces §7c's row to 0.000000** on all six
metrics — the check that the two tables describe the same population before any of the rest
is comparable.

| arm | CRPS | PIT KS | **boundary** | body | **shoulder** | point mass |
|---|---|---|---|---|---|---|
| `betabinom` *(reference, point MLE)* | 9.8125 | 0.0667 | 0.0201 | 0.0107 | 0.0253 | 0.0109 |
| `mixture` *(point MLE — §7c's row, reproduced)* | 9.8237 | 0.0631 | **0.0108** | 0.0047 | **0.0235** | 0.0068 |
| Stan plug-in | 9.8195 | 0.0643 | 0.0123 | 0.0045 | 0.0251 | 0.0073 |
| **Stan posterior** *(what ships)* | 9.8239 | 0.0643 | **0.0120** | **0.0038** | **0.0243** | 0.0075 |

**D1's rule holds for the object that ships, not only for the ladder row it is a port of.**
Every margin is a paired bootstrap against the single-component reference on the same 883
validation rows, because a boundary margin quoted bare is a prompt rather than a finding:

| arm | CRPS vs `betabinom` | boundary vs `betabinom` | shoulder vs `betabinom` |
|---|---|---|---|
| `mixture` (point MLE) | +0.0112 [−0.0280, +0.0511] | −0.0089 [−0.0099, −0.0042] | −0.0021 [−0.0086, −0.0010] |
| Stan plug-in | +0.0069 [−0.0320, +0.0478] | −0.0076 [−0.0085, −0.0046] | −0.0005 [−0.0070, **+0.0005**] ❌ |
| **Stan posterior** | +0.0113 [−0.0257, +0.0503] | **−0.0079 [−0.0088, −0.0043]** ✅ | **−0.0013 [−0.0078, −0.0003]** ✅ |

**Three things in that table are worth reading rather than skipping.**

**1. The port attenuates the calibration win without losing it.** The shipped posterior keeps
**89%** of the point MLE's boundary margin (−0.0079 against −0.0089) and **62%** of its
shoulder margin (−0.0013 against −0.0021). Both still exclude zero and CRPS is still
non-inferior, so D1 passes — but the arm that ships is a slightly weaker version of the arm
that was selected, and quoting §7c's 0.0108 as the shipped head's boundary error would
overstate it by 11%.

**2. The plug-in fails the shoulder half and the posterior passes it**, which is the one
place these two rows disagree qualitatively. The plug-in is one mixture at the posterior
mean; the posterior is a mixture *over* the posterior, and integrating the parameter
uncertainty is what pushes its shoulder error back down to −0.0013. This is the opposite of
the pattern in the single-component port, where plug-in and posterior differ by 0.002 CRPS
and nothing else — a reminder that "the posterior barely moves the marginal metric" was a
statement about *that* likelihood.

**3. The body improves most, and it was never the selector.** 0.0107 → **0.0038**, the best
of any arm on the table including the point MLE's 0.0047. §7c result 6's finding — that no
arm here repeats the season trend's failure of buying boundaries out of the middle — survives
the port and gets stronger.

**`π` is the same object it was at the point MLE.** Mean **4.82%** against 4.88%, running
from **1.23%** at the 10th percentile of validation players to **10.58%** at the 90th, an
**8.6×** spread against 8.8×. So the arm's distinguishing claim — that it can say *who* is at
risk, where a wider frailty can only say that someone is — is a property of the fitted
posterior and not of the optimizer.

**The sampler behaved, and the multimodality the point MLE warned about did not appear.**
R̂ **1.0073**, min ESS **1,399**, **0** divergences. The point MLE needed multi-start (§7b),
so the four chains were started at `θ = 0.02 / 0.08 / 0.25 / 0.50` and reported **one at a
time** — R̂ is the wrong instrument for a mixture, because four chains each stuck in a
different mode can post a respectable R̂ while describing four different models. The largest
between-chain gap is **0.276** pooled posterior sds, on `μ_low`; `θ` reads 0.1112 / 0.1072 /
0.1073 / 0.1057 across chains started 25× apart. This posterior is unimodal where the
optimizer's surface was not.

**The MLE sits inside the 95% credible interval for 11 of 11 mixture terms**, at
`θ` **0.1078** against 0.1116, `μ_low` **0.1133** against 0.1000 and `ρ_low` **0.0578**
against 0.0441. That agreement is *weaker evidence than the coefficient block's* and the
difference matters: the prior on `β` is `normal(0, 1/√(2·l2))`, which makes the posterior
mode exactly the penalized MLE, so agreement there is a check with a defined answer. `θ`,
`μ_low` and `ρ_low` are **bounded** parameters the ladder fits inside a box with no penalty,
and `γ` carries a `normal(0, 2.5)` prior here and none there — the one number in this port
that is a choice rather than an identity. The predictive table above is the check; the
parameter table only localizes a disagreement if one appears. `γ`'s individual coefficients
are the visible cost: `age` reads 0.39 ± 1.53 against the MLE's 1.14, because eight
coefficients identified through 5% of the rows are weakly identified one at a time. What
survives is what `π` does, which is item 3 above.

**Cost: 366 s of sampler time against the single-component head's 94, a 3.9× increase.**
Worth recording because the first implementation was **~10×** worse than that: written as a
per-row loop of `beta_binomial_lpmf` calls it did not finish warmup in 20 minutes. A mixture
needs the per-row density *before* it is summed and Stan's lpmf vectorizes to a sum, so the
obvious form is a scalar loop; writing it through vectorized `lbeta` instead — with the
binomial coefficient cancelling inside the component difference, so it is never formed —
recovers almost all of it. The ship plan predicted that a slow fit would be "a signal about
the geometry"; it was a signal about the autodiff graph, and the geometry is fine.

**Two consumers are deliberately not wired for it and raise rather than mis-describe the
head.** `posteriors.availability_artifact` and `rehydrate_availability` both refuse a mixture
artifact, because `DesignRecipe` carries **one** scaler and `π`'s covariate block is not a
subset of the mean's — a persisted artifact would rehydrate as the single-component head with
nothing raising. That is §4 of `docs/availability-mixture-ship-plan.md`.

**And one debt is opened here rather than closed.** `stan_availability_metrics.csv` and every
port figure quoted against it in `docs/availability-plan.md`, `docs/facts-archive.md` and
`docs/model-development-notes.md` still describe the **single-component** head — `make
stan-availability` was not re-run, so `make docs-audit` stays green over a consistent set of
figures rather than a half-refreshed one. Re-running it will move roughly forty quoted
figures at once, which is the propagation session's job and not a port check's.

---

## 8. If `mixture` is to ship: what has to be decided, and what is already owed

Scoped 2026-08-11. Nothing here is a measurement; it is the list of judgement calls and
debts a Stan port would run into, written down so the port is not the place they get
discovered.

### The decisions

**1. Is this head selected on CRPS, or on calibration?** The load-bearing one. Every
availability decision to date has been taken on **mean CRPS with a paired bootstrap**.
`mixture` *ties* CRPS (+0.011, interval spanning zero) and wins every regional metric.
Shipping it means deciding the objective for this head is tail calibration. `README.md` §4
already argues that position — *"A model that improves marginal CRPS by 1% and gets the
correlation structure wrong is worth less here than one that does the reverse"* — but it has
never been the stated rule for **this** head, and the call should be made before more numbers
arrive rather than reverse-engineered from them.

**2. Does a shape win need a contest-level demonstration first?** Nobody has shown that
moving `P(missed ≤ 5)` from 16.9% toward the observed 12.1% changes a draft. Only
`make strategy-sweep` can, and it is expensive. Gate on it, or accept the calibration
evidence plus the mechanical argument in §1.

**3.** ~~**Where the mixture lives in Stan.**~~ ✅ **Done 2026-08-12** (§7h). The optional
block, not a fork: `betabinomial_glm.stan` serves **six** heads (availability, minutes, four
conversions, overtime onset), and both nestings are exact on Stan's own `log_prob` — `P = 0`
for the parameter space and `θ = 0` for the target, the latter *bit for bit* because the
mixture enters as an additive correction to the untouched beta-binomial statement. The
`n_rho` precedent held.

**4. What `π`'s covariate block actually is.** The eight columns in `PI_COLS` are one reading
of "age, prior absence, playoff workload". That is a shipped choice which lands in the
persisted `DesignRecipe`, not a default. **Kept as-is through the port** and now duplicated
as `stan_availability.PI_FEATURES`, because `availability_window` imports `season_terms`
which imports `stan_availability` and a top-level import would be a cycle — a test pins the
two lists equal, since two copies of a list is exactly how a head comes to ship a different
model from the one that was selected.

**5.** ~~**Whether the `l2` confound is settled first**~~ ✅ **Settled 2026-08-11, and it was
a null** (§7d). Eight penalties from 0 to 256 for every arm at the shipped window: the
reference's best is worth **0.00034** CRPS, **99.4%** of `beta_rect`'s margin survives it, and
`mixture` ties on CRPS at every penalty on the grid. Nothing about the port's premises moved,
and the mixture's eleven unpenalized parameters turn out not to be an advantage worth the
sentence they were given — the penalty is 1.5 parts in 100,000 of the objective at these row
counts. The prediction in this item — "they probably survive" — was right, and it now has a
number.

### Two debts that predate this axis

**1.** ~~**The simulator re-implements this head rather than drawing through it, and it is
already out of sync.**~~ ✅ **Fixed 2026-08-11.** `src/sim/season.py` inlined the
beta-binomial draw with a **scalar** dispersion, `np.full(n_players, rho_draws[draw])`, while
the persisted `train` posterior has carried `rho_draws` of shape **(1000, 4)** with
`n_rho: 4` since the role-graded head shipped. **`make simulate-season` was run and it did
fail there**, not only at the expression — `ValueError: could not broadcast input array from
shape (4,) into shape (539,)` at sim 0 of 2,000 — which was the check this section asked for
and the difference between a broken target and a broken line. The `train_val` artifact is
still `(1000,)` with `role_rho: None`, so it predates the window round: one window raised and
the other was silently stale.

The fix is **not** `predict_samples`, and the reason is worth keeping. That returns **games
played** for the design rows, one row per player-season; the simulator needs the **rate**,
because it applies that rate to each of a player's *cells* — a player traded mid-season has
more than one — before handing the count to `allocate_spells`. A head-level predictive cannot
be split across cells. So the head's dispersion axis is reconstructed instead, from the `cut`
recipe step the artifact already carries (`season.availability_rho_bin`), which is the same
door the composition head is read through and needs no import of `stan_availability`. Both
artifact shapes are handled, `rho_bin`'s 1-based convention is pinned by a test, and a player
with no design row falls into the **lowest** bucket — `role_bins`' own rule, and the widest
dispersion.

**Every Gate A row improved, and it is the first reading taken against the head that ships**
(`docs/simulations-plan.md`, "Gate A, in full"): games played CRPS 9.6754 → **9.5262** and
9.7829 → **9.6140**, season-total MAE 402.14 → 399.03 and 407.89 → 402.48. The two
bonus-on-realized-minutes rows reproduce to four decimals, which is the control — they
condition on realized minutes and played games, so they are the only rows the fix could not
have moved.

The mixture would have made the same line worse, since `π`, `μ_low`, `ρ_low` and `π`'s design
matrix would all have had to be inlined too. Reconstructing the recipe instead means a mixture
port extends the recipe rather than the simulator.

**2. Two gates are owed from the window round.** §5.1 records that `season-total`'s Gate E
and `stan-games-played` — both of which hold this head as a floor — have not been re-run. If
the mixture goes in, re-run them once afterwards rather than twice.

### The order the evidence supports

~~Fix the simulator's draw path → sweep `l2`~~ ✅ **both done 2026-08-11** → ~~port with
`π = 0` nesting asserted~~ ✅ **done 2026-08-12, §7h** → re-run the two owed gates → then
decide whether the strategy sweep is required. The first two were cheap and de-risked
everything after them: one was a live break that the target reproduced, and the other was a
null that removes the last stated qualification from §7c.

**What the port session leaves for the next one**, beyond the two owed gates:

1. **`make posteriors` and `rehydrate_availability` raise on a mixture head**, by design —
   `DesignRecipe` carries one scaler and `π`'s block needs its own. Wire the second design
   block through `availability_artifact`, `_thinned`'s draw list and `rehydrate_availability`
   before anything downstream can read this posterior.
2. **`make stan-availability` has not been re-run**, so `stan_availability_metrics.csv` and
   the ~40 port figures audited against it across `docs/availability-plan.md`,
   `docs/facts-archive.md` and `docs/model-development-notes.md` still describe the
   single-component head. `make docs-audit` is green over a *consistent* set of figures; it
   will go red the moment that target runs, which is the intended alarm and not a surprise.
   Refresh those quotes in the same session that re-runs it, keeping the beta-binomial port's
   figures as `historical=True` rows the way the window round kept the pre-window ones.

**One consequence to carry into the port's session.** The simulator's tensors were rebuilt,
so everything downstream of `sim_tensor_*.npz` — `make bracket`, `make draft`,
`make strategy-sweep` — is now scored against a *previous* tensor. Those artifacts were not
re-run, because §4's propagation session re-runs them anyway once the mixture lands, and
running the sweep twice is the expense this ordering exists to avoid.
