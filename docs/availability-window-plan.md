# Availability window plan: the fitting window, a season trend, and where `rho` lives

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
and it never hurts a single metric in the table.

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
