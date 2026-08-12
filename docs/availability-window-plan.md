# Availability window plan: the fitting window, a season trend, where `rho` lives, and which likelihood

The availability head misses **both ends** of its own distribution, in opposite directions,
and no metric it has ever been gated on can see that. This doc is the measurement of why,
the ladder that separates the candidate causes, and what it settled.

`make availability-window` → `outputs/predictions/availability_window.csv`, one row per arm.
Built 2026-08-11. It is a **point-MLE specification ladder**, not a shipping path: an arm
that wins here earns a Stan port in `stan_availability.py`, it does not ship from here.

> **§9 is the current state of the whole line of work** — what shipped, what is a measured
> null not to be rebuilt, and what is still open, ranked by stake. Start there. §10 closes
> its items 3 and 4 (`make availability-regime`), leaving item 5 as the only remaining one
> that changes the model rather than the fitting rule.

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
   as large.

   ⚙️ **Superseded by the mixture on 2026-08-12 (§7i)**, which is where this head's shipped
   figures now live. The gates that hold it as a floor (`season-total`'s Gate E,
   `stan-games-played`) were owed from this round and were re-run there, once, rather than
   twice.
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
5. ✅ **Closed — (a), (b) and (d) by §5b, (c) by §10c/§10d on 2026-08-12. All four are
   measured NULLS.** Do not rebuild any of them: split `beta`/`rho` windows, exponential
   decay, per-block windowing and shrinkage toward the long-window fit were all fitted and
   **none beats the plain window on validation**, for one shared reason (they lean on the
   COVID trough). §10c also withdraws the reason §5b gave for *why* (c) would be the
   principled version of the block-splice arm: fitted jointly, the same coefficient
   partition does not beat the transplant. The original text:

   **Optimize the trade rather than picking a window.** §4b shows CRPS optimal at a
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
6. ~~**A confound in this ladder, to fix before (a)-(d) are trusted.**~~ ✅ **Closed by
   §5b's `l2_by_lookback` experiment**, which swept `l2` from 0.25 to 256 against the
   lookback and found the 8-season optimum **survives and sharpens** — so the turnaround at
   5 and 3 is variance, not a penalty artifact. (The *other* `l2` confound, on the
   likelihood axis, is §7d and was closed separately on 2026-08-12.) The original text:
   `l2` is pinned at 1.0
   at every lookback, and a 1,155-row fit wants more regularization than a 6,630-row one.
   The short lookbacks are therefore under-regularized for their row count, so some of the
   turnaround at 5 and 3 is a penalty artifact rather than variance. Sweep `l2` jointly
   with the lookback on the rolling harness before reading the optimum as a fact.
7. ✅ **Closed 2026-08-12 as a NULL — §10a, §10b, §10d.** §5b sharpened the motive and §10
   spent it: every optimization in §5b failed by leaning on the COVID trough, which is
   precisely the population this item is about. Swept as its own axis and crossed with
   lookback, it needed a **contamination harness** to be visible at all — the plain
   walk-forward is blind at 11 of 13 origins — and the instrument that works there does not
   survive validation on either likelihood. The original text:

   **Recency and representativeness are different knobs.** 2019-20 through 2021-22 are
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
>
> ⚙️ **§10 is the second half of that evidence and makes it four instances.** It also shows
> the limit is worse than "cannot see": on the regime axis the harness's two live origins
> *are* the trough seasons, so it does not merely miss the effect, it ranks the arms
> backwards (§10a). The fix used there — staging the production situation by injecting the
> unrepresentative block into an ordinary origin, with a same-size ordinary block as the
> control — is the general form, and it is available to any future axis whose population
> the walk-forward cannot reach.

---

## 6. The same question for the minutes heads — ✅ **laddered 2026-08-12, in `docs/minutes-window-plan.md`**

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

### 6a. What the ladder found — and the caveat above was right about both halves

`make minutes-window`; the round is written up in
[docs/minutes-window-plan.md](minutes-window-plan.md). Everything above is kept as written
because the caveat is the part that paid off. Four corrections and one null, in the order
they change what someone would do:

**1. The caveat's own reasoning was correct, and it splits the two claims.** Rebuilt through
`stan_minutes.build_design`'s own rows, the workhorse fold survives the population change at
**8.0×** (0.1075 → 0.0134) — a different number, the same event — while the sd contraction
falls from −15.2% to **−9.0%**. §6's arithmetic about which claim could flip was right on
both counts.

**2. "Flat afterwards" is wrong: the contraction has stopped and partially reverted.** On the
head's own rows cross-player sd bottoms at **0.1563** in 2019-20 and rises in every season
since, to **0.1824** in 2023-24 — back to its 2012-13 level. Pooled, 2014-15 → 2018-19 reads
**0.1663** against 2019-20 → 2023-24's **0.1706**. On the composition's rows there is no
recent contraction left at all. **So a short window no longer buys a narrower population.**

**3. The break is 2010-11, not 2014-15** — for the sd (sup-F 107.15) and the workhorse tail
(203.07) on the head's own rows, and in the same place on all three populations. The *mean*
breaks two seasons later at 2012-13. Same lesson as §4 result 1: where the regime changed and
where the best window starts are different questions.

**4. The window does not replicate; the dispersion does.** Every window arm beats the
incumbent on validation with an interval clear of zero, and matched by fit-row count on the
rolling harness the window collapses to **−0.079 [−0.59, +0.43]**, 6 of 13 origins — while
**role-graded ρ** reads −1.672 [−2.55, −0.80] on validation against −1.376 [−1.73, −1.00] on
the harness, **13 of 13 origins**, at a **2.12–3.03×** spread. That is the largest graded
dispersion in the project, against this head's own 1.26–1.54× on availability and the
composition's 2.07×. §4b's verdict reached from the other direction: on availability ρ barely
moved across windows and era pooling was falsified; here ρ *does* move (−15% across
lookbacks) and capturing it still does not pay. **Pooling across eras is not the defect;
pooling across players is.**

**5. The stake is a NULL, and it could not have been anything else.**
`sim.minutes.player_season_sigma = 0.450` **stands**. A short window does narrow the marginal
predictive — 302.04 → 277.06, and 265.67 once ρ is graded, −12.0% — but the tie boundary
moves the *wrong way*, from σ **0.200** against the incumbent to **0.300** against the best
arm, because a short window improves the head's CRPS (−5.3) more than it narrows its spread
and is therefore a *harder* reference to tie. And the framing above was mistaken in a way
worth recording: σ is selected by the **composition's own** CRPS optimum on training rows, so
the marginal head appears nowhere in that estimator and no property of it can move the
constant. What it moves is the verdict — the tie band narrows from [0.200, 0.525] to
[0.300, 0.450], leaving the shipped 0.450 at its upper edge.

**And the round pushes the retirement question backwards.** The marginal head comes out of it
*better* — CRPS 144.23 → 138.91, PIT KS 0.0737 → 0.0392, the latter better than every
injected composition arm including the shipped σ's 0.0659.

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
  the ladder could not produce a number for, and **D1** in §8
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
nothing raising. ✅ Wired 2026-08-12 — §7i.

**And one debt is opened here rather than closed.** `stan_availability_metrics.csv` and every
port figure quoted against it in `docs/availability-plan.md`, `docs/facts-archive.md` and
`docs/model-development-notes.md` still describe the **single-component** head — `make
stan-availability` was not re-run, so `make docs-audit` stays green over a consistent set of
figures rather than a half-refreshed one. Re-running it will move roughly forty quoted
figures at once, which is the propagation session's job and not a port check's.
✅ **Closed 2026-08-12 by §7i.**

---

### 7i. The shipped head — propagated 2026-08-12

`make stan-availability` re-run, `make posteriors`, `make model-cards`,
`make simulate-season`. The mixture is what the whole chain now reads.

**The shipped metrics table** (`stan_availability_metrics.csv`, 883 validation rows, every
point MLE refitted on the same 4,027 windowed rows):

| arm | CRPS | MAE | R² | PIT KS | ρ (main) |
|---|---|---|---|---|---|
| `stan_plug_in` | 9.8195 | 14.3610 | 0.3851 | 0.0643 | 0.2261 |
| `mixture_mle` | 9.8237 | 14.3785 | 0.3844 | **0.0631** | 0.2245 |
| **`stan_posterior`** *(ships)* | **9.8239** | 14.3770 | 0.3847 | 0.0643 | 0.2261 |
| `beta_binomial_role_rho` | 9.8247 | 14.4211 | 0.3889 | **0.0588** | 0.2627 |
| `beta_binomial` | 9.8444 | 14.4211 | 0.3889 | 0.0632 | 0.2627 |

Every row reproduces §7h to four decimals, which is the control: the shipped target and the
port check fit the same head on the same rows and had better agree before anything else is
read. R̂ **1.0073**, min ESS **1,399**, **0** divergences, **366 s**.

**Three things this readout settles.**

**1. The port check's reference was wrong, and fixing it changed the verdict from a
half-failure to a clean pass.** `fit_and_score` compared the shipped posterior against
`mle` — the *single-component* shared-ρ optimum — and read the MLE inside the 95% credible
interval for **19 of 24** terms with `rho[30+ mpg]` at **z = −6.34**. That is not port
drift. The mixture takes the disrupted seasons out of the main component, so its dispersion
genuinely falls (0.2261 against 0.2627), and comparing the two is comparing two likelihoods.
Referenced against `mixture_mle` — the arm the head is a port *of* — the same posterior reads
**35 of 35** terms inside, max gap **0.74165**, largest **0.597 posterior sd** (`rho_low`),
and every dispersion term agrees to **z ≤ 0.29**. The module already made this argument one
axis over, about fitting both on the same *rows*; it now makes it about the same
*likelihood*, and the coefficient artifact carries all 35 terms rather than 24.

**2. The mixture widens the dispersion grading from 1.54× to 1.97×, and which end moves is
the finding.** Main-component ρ, point MLE against posterior:

| bucket | fit rows | MLE | posterior | sd | z | single-component posterior |
|---|---|---|---|---|---|---|
| `<12 mpg` | 608 | 0.3095 | **0.3124** | 0.0129 | +0.230 | 0.3176 |
| `12-24` | 1,664 | 0.2369 | 0.2394 | 0.0089 | +0.288 | 0.2698 |
| `24-30` | 868 | 0.2080 | 0.2086 | 0.0104 | +0.057 | 0.2532 |
| `30+ mpg` | 887 | 0.1593 | **0.1589** | 0.0087 | −0.046 | 0.2064 |

The fringe bucket barely moves and the star bucket falls by a fifth. A star's blown-up
season used to be carried as dispersion inside one component; now it is the low component,
and what is left is a genuinely tighter healthy-season rate. **That is the arm's substantive
claim arriving in a parameter rather than in a metric** — and it is the reason the head is
worth more to a draft than its CRPS tie suggests, because a star's downside is now a
*separate event with a probability* instead of a fat tail on his ordinary season.

**3. PIT KS is the one metric the shipped head loses on, and D1 says so in advance.**
0.0643 against the role-graded arm's 0.0588. A single KS distance integrates the whole curve
and is dominated by the middle deciles; D1 selects on **regional** tail calibration with a
CRPS non-inferiority guard, and on the regions it names the mixture wins (boundary 0.0120
against 0.0201, shoulder 0.0243 against 0.0253). Recording it plainly because a rule stated
before the measurement is only worth something if the losses are reported under it too.

**The board term grew, which is the mixture's between-component variance arriving in the
joint** (`stan_availability_board.csv`):

| portfolio | independent sd | shared-β sd | inflation | *was* |
|---|---|---|---|---|
| 12 | 69.060 | 6.499 | +0.4% | +0.4% |
| 15 | 76.810 | 7.669 | **+0.5%** | +0.5% |
| 30 | 108.652 | 13.370 | +0.8% | +0.7% |
| 150 | 242.604 | 58.432 | +2.9% | +2.4% |
| 883 | 589.169 | 332.588 | **+14.8%** | +12.3% |

The conclusion is unchanged and so is the warning attached to it: real for board-wide
exposure, near-irrelevant for one roster, and the 883 figure must never be quoted as if it
applied to a 15-man team.

**`π` is the same object it was at the point MLE and at the port**: mean **4.82%**, **1.23%**
at the 10th percentile of scored players to **10.58%** at the 90th, an **8.60×** spread.

**What the propagation itself required, in code.** `DesignRecipe` now carries a **second
design block** — `pi_features` and its own `pi_scaler` — because `PI_FEATURES` enters through
a different link and cannot share the mean's nineteen-column scaler. Three consequences were
each a silent failure waiting to happen, and each is now pinned by a test:

- **The round-trip would have checked the wrong quantity and passed.** `response="mean_mu"`
  serves the reference through `mu_draws`, which under a mixture is the **main component's**
  mean — a number the shipped head never reports. The head's response is now
  `mixture_mean_mu`, routed through `predict_mean` on both sides. A distinct name rather than
  a widened `mean_mu`, so a consumer that switches on `response` raises instead of quietly
  serving a different function of the same draws.
- **A dataclass default does not survive unpickling.** Every artifact written before those
  two fields restores without them, so `recipe.pi_features` would raise `AttributeError`
  rather than falling back. `DesignRecipe.__setstate__` applies the defaults under the
  restored state.
- **The simulator has to draw the component first.** `season.availability_rates` draws
  per player from whichever component won, never a blend of the two rates — the blend is
  precisely the season between healthy and disrupted that the arm exists to say does not
  happen. `π = 0` reproduces the single-component draw **bit for bit**, rng calls included.

The model card also emits the eleven mixture terms, against `π`'s own scaler. A card showing
`alpha`, `beta` and `rho` for this head would be describing the model that *didn't* ship.

**One thing the propagation broke, and the guard is what found it.** `make model-cards` fails
the build when the 95% ECDF ribbon moves more than 0.02 between two halves of its draws, and
the mixture's genuinely wider predictive pushed the availability head from 0.0097 to
**0.0216** at the shipped 200 draws. The budget went to **400** — re-measured at
300/400/600/800/1000, not extrapolated — where the worst gated head is availability itself at
**0.0136** and `make model-cards` costs 17 s rather than 10. That constant was justified by a
measurement and stayed correct only until the model under it moved, which is the argument for
gating a budget rather than asserting one.

---

### 7j. The two gates owed from the window round — re-run, and neither verdict moves

Both were outstanding since §5 item 1 and were re-run once here rather than twice.

| gate | reads | verdict | figures |
|---|---|---|---|
| `season-total` **Gate E** | `spell_process` against the incumbent on the deliverable | ❌ **fails, unchanged** | MAE **406.65** against 400.46 (+6.19), CRPS **291.63** against 287.26 (+4.37), bias −15.87 against −3.06 |
| `make stan-games-played` **Gate D** | the spell process's own arms against the incumbent | ❌ **no arm clears, unchanged** | `duration_covariates` 10.1676 / 0.0672 / 0.0535; `calibrated_fallback` 10.0207 / 0.1017 / 0.0440; `hybrid` 10.0057 / 0.0939 / 0.0406 |

**Neither could have moved, and saying why is worth more than the re-run.** Both gates hold
the **point-MLE incumbent** as their floor — `availability.BetaBinomialGLM`, full-window and
shared-ρ — not the shipped Stan head. `season_total.gp_treatments` fits it in-process and
`stan_games_played._floor_scores` refits it per split. Neither module reads
`stan_availability`'s posterior at all; `stan_games_played` imports only
`availability_design`, the frame builder. So the window round, the role-graded dispersion and
now the mixture were all invisible to them by construction, and the re-run was insurance
rather than a live risk. **§5 item 1's phrase "hold this head as a floor" was imprecise** —
they hold the *incumbent* as a floor, which is a different object from the head that ships.

What did move is the fourth decimal, on every arm, because the spell process is a Monte Carlo
simulation over freshly-sampled posteriors: `within_tenure` 7.2265 → **7.23495**,
`duration_covariates` 10.1625 → **10.1676**, its spell-shape error 1.8105 → **1.7373**, and
Gate E's margin +6.34 → **+6.19**. 47 audited figures across three docs moved and no verdict
did, which is the useful shape for a re-run to have: the gates are not resting on margins
that noise can flip. Both remain failures by margins a bootstrap can resolve —
`duration_covariates` loses CRPS with a 95% interval of [+0.0737, +0.2393].

---

### 7k. D2's value measurement — and it is confounded, which is the finding

`make bracket` → `make draft-sim` → `make strategy-sweep`, all against the rebuilt tensor.
D2 asked what the calibration win is worth in Round-1 advance probability, **as a value
measurement rather than a gate**. Here is what came back, and then why it does not answer
the question that was asked.

**The shipped arm (`lineup_value_blend30`) in the 600k Shootaround**, pooled over the two
validation seasons:

| reading | now | recorded | is the change resolved? |
|---|---|---|---|
| simulated lift over the symmetric-field null | **+0.1890** [+0.1037, +0.2789] | 0.2107 | **no** — 0.2107 sits inside the interval |
| realized lift, the same portfolios on real box scores | **+0.1713** | 0.1268 | **no** — 2 seasons, +0.2890 and +0.0535 |
| autodraft twin over the uncapped click | **+0.0144** | +0.0091 | resolved, same sign |
| lift given up by autodrafting the board | **0.0975** | 0.092 | same sign |

**Every verdict the sweep carries is unchanged.** Gate C passes — the injected world
reproduces the market's measured skill gap (+0.0491 on 2022-23, +0.0163 on 2023-24, against
an uninjected world that has it *backwards* at −0.1230 / −0.1229). Gate D still fails in
**0 of 6** paired comparisons. The shipped arm is still `lineup_value_blend30`, separated
from **23 of 23** rivals in the 600k paired comparison, nearest rival `lineup_value` at
−0.0345 [−0.0449, −0.0237].

**And now the confound, which is the honest headline.** The recorded 0.2107 / 0.1268 were
measured against a tensor built **before the window round** — before the 2012-13 window,
before the role-graded `rho`, and before the simulator's availability draw was fixed to
gather `rho` per player rather than broadcast a scalar. The window round rebuilt the tensors
and *deliberately did not re-run the sweep* (§8, "one consequence to carry into the port's
session"), on the reasoning that the propagation session would re-run it anyway and running
it twice was the expense to avoid. That reasoning saved an hour and cost the measurement:
**this run moves two things at once**, and 0.2107 → 0.1890 cannot be attributed to the
mixture.

So D2 is answered only in the weak form:

- **What is established.** Nothing in the contest readout moved detectably. The simulated
  lift's interval contains the previous point estimate; the realized side has N = 2 seasons
  and intervals that cover most of the table; and no gate flipped. Whatever the window,
  the role-graded dispersion and the mixture did *together*, it is smaller than this
  instrument can resolve at 500 worlds per season.
- **What is not.** The mixture's own contribution. Isolating it needs a single-component
  counterfactual — `stan.availability.mixture: false`, then `posteriors` →
  `simulate-season` → `bracket` → `draft-sim` → `strategy-sweep`, about 2.5 hours — with
  everything else held at today's code.

**The methodological lesson is worth more than the number.** A value measurement whose
baseline is not re-measured under the same code is not a value measurement, and the decision
to skip one sweep to save an hour is exactly what produced that. If the counterfactual is
run, it should be run as a *pair* — both arms on the same day, same code — rather than
against a recorded figure of unknown vintage.

**None of this bears on whether the mixture ships.** D2 made the contest readout explicitly
non-blocking, so a null here was always going to leave the head where D1 put it. What a
clean counterfactual would buy is knowledge about the *next* head: whether tail-calibration
wins in this project reach the contest at all, which is currently unmeasured in either
direction. ✅ **Run as a pair on 2026-08-12 — §7l.**

---

### 7l. The pair, run — and the contest value is a measured null with a mechanism

`make mixture-value`, 2026-08-12. §7k's answer had to be discarded because its baseline
predated the window round; this is the same question asked properly. Two arms of one chain —
`stan.availability.mixture` **false** and **true**, nothing else touched — each through
`posteriors --groups availability` → `simulate-season` → `bracket` → `draft-sim` →
`strategy-sweep`, back to back on one afternoon at **60 min** and **65 min**. Only one of the
twenty heads is refitted, which is what makes an arm an hour rather than a day.

**The pair is valid, and establishing that was worth the second run on its own.** The mixture
arm was re-run rather than remembered, and it reproduces the recorded run **exactly**: all
seven posterior draw arrays at max |diff| **0.000e+00**, both `sim_tensor_*.npz` bit for bit
on `dk_pts` and the games-played twin, and all three bracket tables and all four draft tables
byte-identical. The chain is deterministic under its seeds. So §7k's figures *were* today's
code — which is now a measurement rather than a hope, and it is exactly the thing that could
not be asserted before.

#### The head reaches the draw, and it reaches it as shape rather than as order

Off the tensors, same players, same seed:

| | 2022-23 | 2023-24 |
|---|---|---|
| board rank correlation **between the arms** | **0.9990** | **0.9993** |
| top-100 overlap | 98% | 99% |
| mean \|Δrank\| over the 192 drafted picks | **3.1979** | 2.9219 |
| max \|Δrank\| over the drafted picks | 19 | 12 |

The *order* is the same board. The *shape* is not, and it moves the way §7f said it should.
§7f found the head over-predicts the upper shoulder — `missed ≤ 5` at 16.92% against an
observed 12.12%, the worst-calibrated region of the distribution. Twenty scoring periods span
~76 games rather than 82, so the tensor's thresholds are its own: `gp ≥ 75` is this window's
"missed ≤ 1", where a star's iron-man season lives (2022-23, single → mixture):

| bucket | P(gp ≥ 75) | P(gp ≤ 41) | q10 season total | mean sd |
|---|---|---|---|---|
| `<12 mpg` | 0.0061 → 0.0052 | 0.5017 → 0.4959 | 155.70 → 126.20 (**−29.49**) | ×1.017 |
| `12-24` | 0.0329 → 0.0297 | 0.2744 → 0.2562 | 517.70 → 485.94 (−31.76) | ×1.024 |
| `24-30` | 0.0602 → 0.0522 | 0.1573 → 0.1349 | 989.43 → 1,018.51 (+29.08) | ×1.014 |
| `30+ mpg` | **0.1023 → 0.0834** | 0.0958 → 0.0773 | 1,299.55 → 1,345.96 (**+46.40**) | ×1.021 |

2023-24 carries every sign, with the star bucket at 0.0385 → 0.0316 and q10 **+52.75**. The
iron-man frequency falls at every bucket — the §7f correction arriving — and **the q10 splits
by role**: a star's tenth-percentile season gets *better* by 46 dk_pts while a fringe player's
gets 29 worse. That is §7i's ρ table, the star's main-component dispersion falling by a fifth
once his disrupted seasons live somewhere else, arriving as a per-player quantity in the
deliverable rather than as a coefficient.

#### The contest does not move, and the control is what establishes it

| tournament | simulated lift | realized lift |
|---|---|---|
| 600k_shootaround | 0.1631 → **0.1890** | 0.1881 → **0.1713** |
| 20k_spin_move | 0.1727 → 0.2060 | 0.2076 → 0.2102 |
| 50k_four_pt_play | 0.1771 → 0.2084 | 0.1486 → 0.1556 |
| 15k_and_one | 0.1663 → 0.1923 | 0.1207 → 0.1205 |
| 88k_alley_oop | 0.3541 → 0.4359 | 0.3601 → 0.3973 |

Both columns hold `lineup_value_blend30` in **both** arms rather than each arm's own
selection, which matters at `88k_alley_oop` — the single-component arm selects `blend_a70`
there, and reading `strategy_shipped.csv` per arm would report the gap between two different
strategies as the mixture's value.

The simulated column favours the mixture in all five rows and **that is one result, not
five**: the five tournaments share the same worlds and the same portfolios, differing only in
pod size and payout. The axis that can discriminate is the sweep's own **24 strategies**,
because they consume the board differently — and one of them does not consume it at all.

| across the 24 strategies, 600k, pooled over both seasons | |
|---|---|
| mean lift delta | **+0.0097** |
| sd across strategies | **0.0161** |
| spread | −0.0333 to +0.0310, **20** of 24 positive |
| **`adp` — the control, whose board is identical in both arms** | 0.0397 → 0.0490, **+0.0093** |
| `lineup_value_blend30`'s delta, in sds of that spread | **+1.0091** |
| strategy ordering between arms | Spearman **0.9174**, same top arm |

**A strategy that never reads the model captures the entire mean shift.** `adp` ranks on the
market alone, so its board is byte-identical across the arms and its **+0.0093** contains no
drafting whatsoever — it is a property of the world each arm generated. The shipped arm's
+0.0260 sits **one standard deviation** above that mean, in a spread where four strategies
move the *other* way (`model_q75` at −0.0333) on boards correlated at 0.999. And the
instrument's own resolution, derived from the sweep's bootstrap rather than asserted, is
**0.0876** at 95% for an arm-to-arm gap: every row in the contest table is under it.

So the simulated lift is **self-scored** — each arm is measured against a symmetric-field null
inside a world that arm generated, and a head with a wider predictive posts a higher lift
because its world spreads rosters further apart. The realized column is the only reading whose
truth, actual box scores, is common to both arms; there the sign **flips** at the largest
contest and the deltas run −0.0002 to +0.0373 on two seasons at one realization each.

#### Every verdict is unchanged, in both arms

Gate C passes in both. Gate D fails in **0 of 6** paired comparisons in both. The injection
solves its own ρ per arm (0.4189 / 0.4234 against 0.4268 / 0.4164), which is correct — it is
calibrated per world by construction. `lineup_value_blend30` is separated from **23 of 23**
rivals in both and is the top arm in both; the nearest rival differs (`bracket_ev_blend30` at
−0.0218 against `lineup_value` at −0.0345) and neither is close.

#### What this buys for the next head

**A tail-calibration win in this project reaches the shape of the draw and not the order of
the board — and the drafting layer ranks.** That is the transferable finding and it is
mechanistic rather than statistical: a change leaving the board correlated at 0.999 cannot be
expressed by a strategy that consumes an ordering, however much it improves the distribution.
It does not say tail calibration is worthless. It says the current *consumer* has no channel
for it, which is a fact about the drafting layer and points at the shape-reading objectives
(`bracket_ev`, +0.0233, against `bracket_ev_blend30`'s +0.0054 — the same noise scale as
everything else here) rather than at the head.

**And `make strategy-sweep` is not a value metric for a model change.** Its simulated side
scores each model inside its own world, so a cross-model reading off it measures the world as
much as the board; `adp_only_lift` is the row that proves this rather than asserts it, and it
is in the artifact for exactly that reason. The only cross-model reading the sweep produces is
the realized one, and it has two seasons in it. Any future "what is this head worth in the
contest" has to be asked of the realized readout, of a shape-reading objective, or of a larger
world count — and the first two are cheap while the third is not: resolving a 0.02 gap needs
roughly 80× the worlds.

---

## 8. If `mixture` is to ship: what has to be decided, and what is already owed

Scoped 2026-08-11. Nothing here is a measurement; it is the list of judgement calls and
debts a Stan port would run into, written down so the port is not the place they get
discovered.

### The decisions

**1.** ~~**Is this head selected on CRPS, or on calibration?**~~ ✅ **Settled 2026-08-11 —
this is the head's selection rule, and it was stated before the arm was measured.**

> **D1. The availability head is selected on tail calibration, with a CRPS
> non-inferiority guard.** An arm ships if it improves the tail calibration metrics **and**
> its CRPS is non-inferior — the paired-bootstrap interval must exclude a material loss.
>
> `mixture` passes: CRPS **+0.011** with an interval of [−0.028, +0.051] spanning zero,
> against `boundary_tail_error` **−0.0089** [−0.0099, −0.0042] and `shoulder_error`
> **−0.0021** [−0.0086, −0.0010] on validation, and −0.0128 [−0.0138, −0.0064] on the
> rolling harness.
>
> **This is a change of rule and it is deliberate.** Every prior decision on this head was
> taken on mean CRPS. The reason to move is in `README.md` §4 — *"A model that improves
> marginal CRPS by 1% and gets the correlation structure wrong is worth less here than one
> that does the reverse"* — and in §1 above: a dead roster slot and an iron man are the two
> events a Round-1 knockout turns on, and both errors make a drafted roster look more
> reliable than it is. **The guard is what stops a future arm trading real accuracy for
> tails.**
>
> The rule is stated *before* the next arm is measured, which is the point of writing it
> down rather than inferring it from the arm that happened to win. It is also what makes the
> shipped head's **worse** global PIT KS (0.0643 against the role-graded arm's 0.0588) a
> reported loss under a standing rule rather than a figure to explain away — see §7i.

**2.** ~~**Does a shape win need a contest-level demonstration first?**~~ ✅ **Settled
2026-08-11: no.** Port first, then measure the contest value with `make strategy-sweep` as a
**value measurement rather than a gate**. Nobody had shown that moving `P(missed ≤ 5)` from
16.9% toward the observed 12.1% changes a draft; that remained an open question and simply
was not a blocking one. The measurement itself is in §7k — and it came back **confounded**, because the baseline it is compared against predates the window round.

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

**What the port session left for the next one**, beyond the two owed gates — ✅ **both closed
2026-08-12, §7i**:

1. ~~**`make posteriors` and `rehydrate_availability` raise on a mixture head**~~ ✅ Done.
   `DesignRecipe` carries a second design block (`pi_features` / `pi_scaler`), the four
   mixture draw arrays travel through `_thinned`, and the head's response became
   `mixture_mean_mu` so the round-trip checks the predictive mean rather than the main
   component's. Both traps §4 predicted were real and are pinned by tests.
2. ~~**`make stan-availability` has not been re-run**~~ ✅ Done, and the alarm fired exactly
   as described: 23 audited figures across `docs/predictions-plan.md` and
   `docs/model-development-notes.md` went red at once and were refreshed, with the
   single-component port's figures kept as `historical=True` rows the way the window round
   kept the pre-window ones.

**One consequence carried into that session.** The simulator's tensors were rebuilt, so
everything downstream of `sim_tensor_*.npz` — `make bracket`, `make draft`,
`make strategy-sweep` — was scored against a *previous* tensor. Those artifacts were
deliberately not re-run in the port session, because the propagation session re-runs them
anyway once the mixture lands, and running the sweep twice is the expense this ordering
exists to avoid.

---

## 9. What this whole line of work leaves, after the mixture shipped

Written 2026-08-12, once the mixture round landed. §5's list was about the *window*; this is
the state of everything, ranked by stake. Each item names what would settle it.

**1.** ~~**The minutes heads have the same defect and nobody has laddered them**~~ ✅
**Closed 2026-08-12 — `make minutes-window`, [docs/minutes-window-plan.md](minutes-window-plan.md),
summarized in §6a.** The item read: the largest open stake in this line of work, because it
can revise a *shipped* decision rather than only add one — if the marginal head's
season-level spread is averaged over a contracted window, a short-window refit should narrow
the predictive and lower the injection `σ`, moving
`sim.minutes.player_season_sigma = 0.450`.

**The shipped decision does not move, and the stake was mis-framed.** σ is selected by the
*composition's* own CRPS optimum on training rows, so the marginal head is not in that
estimator and no window on it can move the constant. The tie boundary it *can* move goes the
wrong way — 0.200 → 0.300 — because a short window improves the marginal head's CRPS more
than it narrows its spread, making it a harder reference. **What the round found instead is
on the axis the window was crossed with**: role-graded ρ replicates at 13 of 13 rolling
origins at a 2.12–3.03× spread, the largest in the project, while the window itself does not
replicate at all. Honouring §6's own caveat first was the right call and it paid: the 10.4×
survived the population change at 8.0×, the −15.2% did not (−9.0%), the break is 2010-11
rather than 2014-15, and "flat afterwards" turned out to be a **reversion**.

The item this leaves is smaller and concrete: **port the graded ρ to `stan_minutes`**, which
is one call — `rho_block(len(train))` becomes `role_bins`, and `n_rho = 1` already reproduces
the incumbent bit for bit on the shared `betabinomial_glm.stan`.

**2.** ~~**The contest value of a tail-calibration win is still unmeasured in either
direction**~~ ✅ **Closed 2026-08-12 — `make mixture-value`, §7l. It is a null, and the null
has a mechanism.** The item read: the D2 sweep moved two things at once and cannot attribute
anything to the mixture; the fix is a **paired** counterfactual, both arms on the same day and
the same code, never against a recorded figure of unknown vintage.

The pair was run, and the first thing it returned was that the chain is **deterministic** —
the mixture arm's re-run reproduces the recorded one bit for bit through posteriors, tensors,
bracket and draft. So the vintage was never actually the problem; not being able to *know* it
was. The finding proper is that **the mixture changes the shape of the draw and not the order
of the board** (rank correlation 0.9990/0.9993, 3.20 places of mean movement over the 192
drafted picks) while cutting the iron-man frequency and splitting the season-total q10 by role
(+46.40 dk_pts for stars, −29.49 for fringe). The contest reading is a null against a
resolution of 0.0876, and the `adp` control — a strategy whose board is identical in both arms
— captures **+0.0093** of the +0.0097 mean shift across 24 strategies, which is what makes it
a *measured* null rather than an underpowered one.

What it leaves is a question about the consumer rather than the head: **the drafting layer
ranks, so it has no channel for a distributional improvement.** The shape-reading objectives
(`bracket_ev`) are where that channel would be, and they move no more than noise here.

**3.** ~~**Recency and representativeness are different knobs**~~ ✅ **Closed 2026-08-12 as
a NULL — `make availability-regime`, §10a/§10b/§10d.** The item read: unmeasured, and §5b
handed it a motive — every optimization there failed by leaning on the COVID trough, so an
explicit regime indicator for 2019-20 → 2021-22, or excluding them, is a separate axis from
lookback.

It needed a new instrument before it could be measured at all: the plain walk-forward
harness is **structurally blind** to it, with no arm active at more than 2 of 13 origins, and
what it does see it sees backwards because those two origins are the trough seasons
themselves. On a contamination harness that stages the production situation the mechanism is
unambiguous — the trough biases the predicted availability share down by 1.66pp and an
indicator recovers 1.40pp of it — with a same-size ordinary block as a working control. None
of it survives validation, on the point MLE or on the shipped mixture.

**4.** ~~**Shrinkage toward the long-window fit**~~ ✅ **Closed 2026-08-12 as a NULL, and it
withdrew §5b's diagnosis — `make availability-regime`, §10c/§10d.** The item read: the only
survivor of §5's four candidates, and §5b names it the principled version of the arm that
failed — fit the blocks jointly under different priors rather than transplanting
coefficients between fits.

Built as a per-coefficient Gaussian prior centred on the long-window fit, with `λ = 0` and
`λ = ∞` reproducing the plain short-window fit and the long-window coefficients exactly. It
wins on the rolling harness and loses on validation by **+0.1146 [+0.0444, +0.1821]**. The
sharper result is the matched-pair control: fitted **jointly**, §5b's own five drifting
columns do **not** beat the transplant at the same two windows (+0.0055 [−0.0017, +0.0131]),
so co-adaptation was never what was wrong with that arm — leaning recent was. In the Stan
port this would still be the long-window posterior used as the prior; there is now no reason
to build it.

**5. The exchangeable-trials assumption, which no arm on the likelihood axis touched**
(§7g). Absences come in **spells** — beta-geometric, beating the geometric by 11,278
log-likelihood points at one extra parameter — and one 40-game spell and forty single-game
absences give identical `gp` and very different distributions. A beta-binomial absorbs the
variance inflation from that clustering but not its shape, and neither does the mixture.

**6. A decision that was taken and never implemented.** `docs/availability-ship-plan.md`
decision 2 specifies a three-class imputed role bucket for the **no-prior population** —
rookies from draft position via `stan_composition.rookie_share_priors`, returning veterans
from their bucket at last appearance conditioned on gap length, everyone else the lowest
bucket. Only the third class exists in code: `stan_availability.role_bins` falls back to the
lowest bucket and its own docstring calls that "a guard rather than a live branch". The
population is real — about **14.7%** of season-start roster minutes per `README.md` — but it
reaches the head through the *simulator's* `no_design_availability` path rather than through
the design, so the first question is whether the rule is needed where it was specified.

> ⚠️ **`docs/availability-ship-plan.md` is stale scaffolding that cannot simply be deleted.**
> Its own closing step says to delete it, and the window round it plans has landed — but
> `src/models/stan_availability.py:334` cites its **decision 2** for a live code rule, and
> item 6 above is that decision still outstanding. Relocate decision 2 (and decision 3, the
> fixed role bins) into this doc and repoint the code comment *before* deleting it.

---

## 10. §9's two open axes — measured 2026-08-12, and **both are nulls**

`make availability-regime` → `availability_regime.csv`, `availability_shrinkage.csv`,
`availability_regime_confirmation.csv`. Built to close §9 items 3 and 4:
**(1)** an explicit regime indicator for 2019-20 → 2021-22, or excluding those seasons,
swept against lookback (§5.7); **(2)** shrinkage toward the long-window fit — the blocks
fitted jointly under different priors rather than transplanted (§5.5(c)).

§5's (a), (b) and (d) are not rebuilt. §5b's spliced arm *is*, as a control, because axis
(2)'s whole claim is that it is that arm's principled version.

### 10a. The regime axis had to be re-instrumented before it could be measured at all

**The plain rolling harness is structurally blind to it.** Origins run 2009 → 2021 on the
fitting half and the regime block is the last three seasons in it, so an arm that excludes
or indicates those seasons produces a bit-identical fit at every origin whose fitting rows
end before 2019. Measured rather than asserted, via an `origins_active` column: across the
45 arms of the lookback × regime cross, **no arm is active at more than 2 of 13 origins**,
and 5 are the incumbent by definition. The two live origins are 2020 and 2021 — the same
two §4b caught the season trend's apparent gain hiding in.

**And what it does see, it sees backwards.** Paired within its own lookback (a second
column, because the pooled reference mixes the lookback's effect into every cell), at
lookback 8:

| arm | CRPS vs `lb8__none` | 95% | origins active |
|---|---|---|---|
| `dummy_lag` | **−0.0198** | [−0.0341, −0.0046] | 1 |
| `dummy_both` | **−0.0169** | [−0.0265, −0.0062] | 2 |
| `w0.50` | +0.0018 | [−0.0014, +0.0051] | 2 |
| `exclude` | +0.0072 | [−0.0009, +0.0155] | 2 |
| `dummy_target` | **+0.0081** | [+0.0036, +0.0124] | 2 |

The signs are the harness's population, not the axis. Its two live origins *are* the trough
seasons, so an arm that predicts an ordinary season is penalized for being right about the
wrong year, and an arm that can read a disrupted prior season is rewarded. Production is
the opposite population. **Read no interval in that table as a replication either**: 11 of
13 origins contribute exactly-zero pairs, so a bootstrap over rows is a within-origin
sampling statement about one origin wearing thirteen origins' clothes.

### 10b. The contamination harness — 10 replicates where the walk-forward has none

The situation the production fit is in is *fitting set contains the trough, target season
does not*, and the fitting half contains **zero** instances of it: the first ordinary target
season after the trough is 2022-23, which is validation. So the selector stages it. For each
origin 2009 → 2018 the regime block is **injected** into the fitting rows and the arm is
scored on the origin season, against `clean` — the same fit without it.

This is anachronistic by construction and is an **ablation of a mechanism, not a forecast**;
no row of it is a walk-forward score. `exclude` is absent because excluding the injected
block *is* `clean`, exactly — what the harness can test is whether an instrument that
**keeps** the rows recovers what exclusion gets for free.

| arm | CRPS | vs `clean` | 95% | origins won | share bias |
|---|---|---|---|---|---|
| `contaminated__dummy_target` | **9.9951** | **−0.0171** | [−0.0325, −0.0027] | 8/10 | −0.0026 |
| `contaminated__dummy_both` | 9.9968 | −0.0154 | [−0.0311, −0.0005] | 7/10 | −0.0023 |
| `contaminated__dummy_lag` | 9.9992 | −0.0129 | [−0.0273, +0.0001] | 8/10 | −0.0046 |
| **`clean`** | **10.0121** | — | — | — | −0.0060 |
| `contaminated__w0.25` | 10.0132 | +0.0011 | [−0.0074, +0.0093] | 6/10 | −0.0096 |
| `contaminated__w0.50` | 10.0175 | +0.0054 | [−0.0100, +0.0202] | 6/10 | −0.0124 |
| `contaminated` | 10.0307 | +0.0186 | [−0.0067, +0.0428] | 4/10 | −0.0166 |

Three readings. The **damage is visible in the mechanism before it is visible in CRPS**:
contamination pushes the predicted availability share down by 1.66 percentage points
against `clean`'s 0.60, which is the trough leaking into the intercept, while its CRPS cost
of +0.0186 has an interval covering zero. An **indicator recovers it and then beats it** —
`dummy_target` cuts the bias to 0.26pp and buys −0.0171 over `clean` itself, because the
1,285 injected rows are still information once the level shift is absorbed. And
**downweighting does not work**: partial exclusion partially removes the rows without ever
letting the model *say* what was different about them, so w0.50 lands between the two and
improves neither.

Re-run with the **core block** (2020-21 and 2021-22 only, the two seasons the league series
actually singles out — 2019-20 at 0.6739 sits between 2017-18's 0.6762 and 2018-19's
0.6704), the same arm reads **−0.0125 [−0.0276, +0.0017]**: same sign, interval now covering
zero. The bubble season carries part of the effect.

**The control is what makes it a measurement.** `contaminated` differs from `clean` in two
ways at once — the rows are unrepresentative, and they are ~1,300 extra rows arriving out of
chronological order. `regime_placebo` injects an ordinary block of the same size (2016-18)
instead, over the 7 origins where that block sits strictly after the origin:

| arm | CRPS | vs `inject_placebo` | 95% |
|---|---|---|---|
| `inject_regime` | 9.7426 | **+0.0402** | [+0.0163, +0.0653] |
| `clean` | 9.7085 | +0.0061 | [−0.0107, +0.0232] |
| **`inject_placebo`** | **9.7024** | — | — |

Three ordinary seasons injected out of order cost nothing. The trough costs, and costs
significantly. **The damage is the regime, not the row count and not the anachronism.**

### 10c. Shrinkage toward the long-window fit

`ShrunkBetaBinomialGLM` replaces the transplant with a prior:

```
-loglik(short rows) + l2·||β[1:]||² + Σⱼ λⱼ·(βⱼ − β_longⱼ)²
```

`λ = 0` is the plain short-window fit's objective and `λ = ∞` pins that coefficient to the
long-window estimate exactly, by dropping it from the optimization rather than by a large
finite penalty — so both endpoints are arms that already existed and the family is a
one-knob extension. The `free_drift` variant frees §5b's five drifting columns (intercept +
workload) and shrinks the other fifteen, which is that arm's coefficient partition **fitted
jointly** instead of imported across two fits.

**The harness reproduces §5b before it extends it**, which is what makes the comparison
readable: `long_only` scores **9.9478**, §5b's `(all, all)` to four decimals;
`splice8__intercept_workload` scores **9.8983**, its block winner to four decimals; and the
gap between the splice and `shrink8__laminf` — β entirely on the long fit with ρ on the
short window, which is exactly §5b's `short__none` reference — is **−0.0359**, its headline
figure to four decimals. `shrinkN__lam0` reproduces `shortN__only` (9.918054 against
9.917984).

| arm | CRPS | vs `long_only` | 95% | origins won |
|---|---|---|---|---|
| `shrink5__free_drift__laminf` | **9.8939** | −0.0539 | [−0.0750, −0.0322] | 10/13 |
| `shrink5__free_drift__lam1024` | 9.8973 | −0.0505 | [−0.0742, −0.0268] | 10/13 |
| **`splice8__intercept_workload`** *(§5b's arm)* | **9.8983** | −0.0496 | [−0.0682, −0.0307] | **12/13** |
| `shrink5__lam256` | 9.9025 | −0.0453 | [−0.0674, −0.0235] | 9/13 |
| `shrink8__free_drift__laminf` | 9.9038 | −0.0441 | [−0.0585, −0.0299] | 10/13 |
| `short8__only` | 9.9180 | −0.0299 | [−0.0527, −0.0078] | 7/13 |
| `shrink8__laminf` *(ρ windowed, β not)* | 9.9342 | −0.0136 | [−0.0199, −0.0071] | 11/13 |

The knob behaves: within each family CRPS falls monotonically in λ to an interior optimum
(at short = 8 the global family runs 9.9181 → 9.9175 → 9.9164 → 9.9145 → 9.9117 →
**9.9088** → 9.9102 across λ = 0 … 1024), so shrinkage is doing something rather than
selecting an endpoint.

**But §5b's stated mechanism does not survive its own matched test, and that is the finding
here.** §5b diagnosed its spliced arm as failing because *"a spliced coefficient vector is
not a fit of anything"* — blocks co-adapted to each other, dropped into a vector estimated
elsewhere. Fit those same five columns **jointly**, conditional on the other fifteen pinned
at their long-window values, same two windows: `shrink8__free_drift__laminf` against
`splice8__intercept_workload` reads **+0.0055 [−0.0017, +0.0131]**, winning 4 of 13 origins.
The joint fit is, if anything, *worse*. The best shrinkage arm beats the splice only by also
shortening the window to five seasons, and even then by **−0.0044 [−0.0175, +0.0081]** — a
tie. **Co-adaptation was not what was wrong with the spliced arm.** What was wrong with it
is in §10d, and it is the same thing that is wrong with every arm in this section.

### 10d. The one validation reading — nothing ships, on either likelihood

Taken once, after every recipe above was fixed, and on **two likelihoods**: the point-MLE
`RoleGradedBetaBinomial` that selection ran on, and the two-component mixture that actually
ships (§7i). A recipe that reweights rows or shrinks coefficients has no guarantee of
surviving a likelihood whose second component already exists to absorb disrupted seasons.

| arm | val CRPS | vs shipped | 95% | PIT KS | boundary |
|---|---|---|---|---|---|
| **`shipped__2012_role_rho`** | **9.8247** | — | — | **0.0588** | 0.0178 |
| `regime__w0.50` | 9.8192 | −0.0056 | [−0.0289, +0.0195] | 0.0692 | 0.0198 |
| `regime__w0.25` | 9.8225 | −0.0023 | [−0.0421, +0.0402] | 0.0758 | 0.0212 |
| `shrink__global__2012__lam256` | 9.8363 | +0.0116 | [−0.0053, +0.0278] | 0.0611 | 0.0183 |
| `regime__exclude` | 9.8367 | +0.0119 | [−0.0490, +0.0780] | 0.0830 | 0.0231 |
| `regime__dummy_target` *(the selected arm)* | 9.8423 | +0.0175 | [−0.0326, +0.0693] | 0.0779 | 0.0233 |
| `regime__core_exclude` | 9.8534 | +0.0286 | [−0.0254, +0.0894] | 0.0822 | 0.0223 |
| `shrink__global__lb5__lam256` | 9.9046 | **+0.0798** | [+0.0258, +0.1304] | 0.0759 | **0.0143** |
| `shrink__free_drift__2012__laminf` | 9.9001 | **+0.0753** | [+0.0368, +0.1139] | 0.0605 | 0.0180 |
| `shrink__free_drift__lb5__laminf` *(the rolling winner)* | 9.9393 | **+0.1146** | [+0.0444, +0.1821] | 0.0854 | **0.0122** |
| `regime__dummy_both` | 9.9350 | **+0.1103** | [+0.0122, +0.2095] | 0.0799 | **0.0133** |
| `regime__dummy_lag` | 9.9857 | **+0.1610** | [+0.0472, +0.2807] | 0.0833 | **0.0113** |
| **`mixture__shipped_2012`** | **9.8237** | — | — | 0.0631 | **0.0108** |
| `mixture__regime_exclude` | 9.8709 | +0.0472 | [−0.0221, +0.1261] | 0.0931 | 0.0180 |
| `mixture__regime_dummy_target` | 9.8925 | **+0.0688** | [+0.0145, +0.1258] | 0.0886 | 0.0161 |

**Not one arm beats the shipped head.** The best challenger, `w0.50`, is a tie (−0.0056,
interval spanning zero) and is worse on PIT *and* on the boundary — which is the metric §7
selected the shipped head on, so it is not close. Everything the rolling harnesses actually
**selected** loses, and loses with an interval excluding zero: the shrinkage winner by
+0.1146, and the contamination winner by +0.0688 once it is applied to the head that ships.

**The discordance is §5b's, for the fourth time, and the tail column shows the mechanism
directly.** The arms that lose most on CRPS post the *best* boundary errors of any
beta-binomial arm here — 0.0113, 0.0122, 0.0133, 0.0143 against the shipped head's 0.0178 —
because every one of them leans harder on the trough and shifts the whole predictive
downward. That buys the low tail and pays for it in location. It is exactly what decay 0.80
did in §5b, and it is the answer to §10c: what was wrong with the spliced arm was never its
estimator, it was **which seasons it leans on**, and fixing the estimator cannot fix that.

**The one arm that is genuinely mis-specified rather than merely over-corrected is
`dummy_lag`, and the confirmation table carries the reason as data.** The disrupted seasons
are *consecutive*, so inside the shipped window every fitting row carrying a regime lag also
carries a regime target: `lag_implies_target_train` = **1.000** over **885** flagged rows of
1,285. On validation the two separate completely — **433** of 883 rows carry the lag flag
and **0** carry the target flag. The coefficient is estimated only where the two coincide
and applied only where they do not, so it extrapolates a trough level onto a season that
recovered. That is an identification failure, not a feature that failed, and no amount of
data inside this window fixes it.

### 10e. What this settles

1. **§9 item 3 (recency ≠ representativeness) closes as a null.** An explicit regime
   indicator and exclusion are both measurable improvements *on the staged mechanism* —
   the contamination harness is unambiguous, with a working control — and neither survives
   contact with validation on either likelihood. The mechanism was real and the correction
   is not worth making.
2. **§9 item 4 (shrinkage toward the long-window fit) closes as a null, and takes §5b's
   diagnosis with it.** The principled version of the spliced arm does not beat the spliced
   arm at matched configuration, so co-adaptation was the wrong explanation for that
   failure. The right one is the one §5b's own closing paragraph gave for everything else
   in it: leaning recent over-corrects into two seasons that partly recovered.
3. **The shipped head is unchanged.** `three_point_era` window, no season term, role-graded
   ρ, two-component mixture — §7i, untouched.
4. **The rolling harness has now selected an arm that validation rejected four times**
   (the trend in §4b, the four optimizations in §5b, the regime instrument and the
   shrinkage recipe here). Its limit is no longer a caveat to be restated; it is a
   measured property with four instances and one shared mechanism, and any future arm on
   this head that wins on the harness by leaning recent should be treated as failing until
   validation says otherwise.
5. **What is left is the population, not the estimator.** Every instrument in §5b and §10
   is a way of re-weighting or re-pooling the same 4,027 rows, and the ceiling on that is
   now well mapped. §9 item 5 — the exchangeable-trials assumption, which no arm on any
   axis has touched — is the only remaining item on this head that changes the *model*
   rather than the fitting rule.
