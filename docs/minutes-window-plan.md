# Minutes window plan: the marginal head's fitting window, its dispersion, and what that does to the injection

`make minutes-window` → `minutes_window_era.csv`, `minutes_window_break.csv`,
`minutes_window.csv`, `minutes_window_rolling.csv`, `minutes_window_stake.csv`.
Module: [src/models/minutes_window.py](../src/models/minutes_window.py).

This is the minutes half of the question `docs/availability-window-plan.md` asked of the
availability head, run the same way and with the same instruments: a point-MLE ladder scored
on validation with a paired bootstrap, confirmed on a rolling origin over the fitting half,
with selection reading validation and nothing else.

It exists because `docs/availability-window-plan.md` §9 item 1 called it **the largest open
stake in that line of work** — the only item on the list that could revise a *shipped*
decision rather than add one. `make minutes-unification` ships the marginal minutes head
**solely** for its season-level spread (season-total predictive sd **302.75** minutes against
the composition's 64.65 when this round ran; **277.23** against **60.5757** since both heads
took preseason blocks, §4), so if that spread turned out to be an average over a contracted
window, `sim.minutes.player_season_sigma` would move. It has since moved for a different
reason — **0.450 → 0.375 on 2026-08-14**, because the composition's own blend shifted the
grid the constant is read off.

**It does not move.** The stake is a null, and the mechanism is the opposite of the one that
was hypothesized. What the round *does* find is a different and larger win on the axis it was
crossed with, and a replication failure on the axis it was about.

---

## 1. §6's own caveat, honoured — the two series rebuilt

§6 measured the marginal head's cross-player sd contracting **15.2%** and P(rate ≥ 0.75)
falling **10.4×**, with a break "completing around 2014-15 and flat afterwards". It also
flagged that both series came from a **rotation filter** (`gp ≥ 20`, `mpg ≥ 10`) rather than
either head's own row filter, and reasoned that a 10.4× fall is too large for a population
definition to flip while a 15.2% contraction is not.

**That reasoning was exactly right, and the rebuild splits the two claims apart.** Each
population is built by its own head's design builder — `stan_minutes.build_design` and
`stan_composition.composition_frame` — rather than by filtering a feature frame by hand,
which is the failure mode §2 documents one head over. The composition's rate is rescaled to
the marginal head's unit (a share of *game* length, not of the `5 × game_length` team pot),
so 0.75 means a 36-mpg workhorse in every row of the table.

| population | sd first → last | change | P(≥ 0.75) first → last | fold | mean |
|---|---|---|---|---|---|
| `rotation_filter` (§6's) | 0.1918 → 0.1626 | **−15.2%** | 0.1574 → 0.0151 | **10.4×** | −7.4% |
| `minutes_head` (its own rows) | 0.2004 → 0.1824 | **−9.0%** | 0.1075 → 0.0134 | **8.0×** | −6.2% |
| `composition_head` (its own rows) | 0.2277 → 0.2053 | −9.8% | 0.1134 → 0.0105 | 10.8× | −9.5% |

**1. The workhorse collapse survives the population change, as §6 predicted.** It is 8.0× on
the head's own rows against 10.4× on the rotation filter — a different number, the same
event. The 36-mpg season has essentially stopped existing: 0.1075 of the head's rows in
1997-98 against 0.0134 in 2023-24, and the *pooled* figures are starker still, **0.1027**
before 2014-15 against **0.0151** after.

**2. The sd contraction does not survive as stated.** −15.2% becomes **−9.0%**, which is a
different claim about how much room a window has to work with.

**3. And the contraction has stopped and partially reverted, which "flat afterwards" hides.**
This is the finding the rebuild adds rather than corrects. Pooled over the head's own rows:

| block | sd_rate | sd_logit | mean_rate | P(≥ 0.75) |
|---|---|---|---|---|
| pre-2014-15 | 0.1942 | 0.8851 | 0.4902 | 0.1027 |
| 2014-15 → 2018-19 | **0.1663** | 0.7410 | 0.4714 | 0.0154 |
| 2019-20 → 2023-24 | **0.1706** | 0.7635 | 0.4739 | 0.0148 |

The per-season series bottoms at **0.1563** in 2019-20 and has risen in every season since —
0.1658, 0.1699, 0.1773, **0.1824** — back to where it sat in 2012-13. **A short window
therefore buys a cross-player spread that is no longer contracted**, which is the first
reason the stake below reads as it does. On the *composition's* rows the reversion is
sharper still: its 2023-24 `sd_logit` of **1.0789** is the highest of the last twelve
seasons, so on that head there is no recent contraction left to exploit at all.

**4. The break is 2010-11, not 2014-15**, and it is in the same place on all three
populations. sup-F over every candidate breakpoint against a 5,000-replicate Monte-Carlo
null (the max over breakpoints has no standard F distribution, so the critical value is
simulated — §2's device applied to a season-level series):

| population | statistic | sup-F | break | null 95th | p | shift |
|---|---|---|---|---|---|---|
| `minutes_head` | mean_rate | 79.30 | 2012-13 | 9.28 | < 0.0002 | −0.0187 |
| `minutes_head` | sd_rate | 107.15 | **2010-11** | 8.84 | < 0.0002 | −0.0280 |
| `minutes_head` | p_workhorse | 203.07 | **2010-11** | 8.82 | < 0.0002 | −0.0957 |
| `composition_head` | sd_rate | 119.90 | 2010-11 | 9.41 | < 0.0002 | −0.0238 |
| `rotation_filter` | p_workhorse | 203.62 | 2010-11 | 9.04 | < 0.0002 | −0.0952 |

So §6's 2014-15 was four seasons late, and — as on availability — **the location of the break
and the best fitting window are different questions.** The dispersion and the tail break
together in 2010-11; the *mean* breaks two seasons later, at 2012-13.

> The composition's own concentration measure moves as §6 said and stays the low-priority
> one: mean team-game HHI **0.1275** (1996-97) → **0.1157** (2023-24), trough **0.1131** in
> 2018-19. A share of a fixed pot divides league drift out before the head sees it.

---

## 2. The ladder

Four windows × two dispersion modes, plus the mandatory no-fit floor. The feature variant is
held at `logit_own_spline`, the one `make stan-minutes` selected — this is a window ladder,
not a second feature selection, and `assert_shipped_variant` raises rather than laddering an
arm the head does not ship. Spline knots are re-fitted on **each window's own rows**, since a
window is a restriction of the fitting half and anything estimated from data is estimated
there.

`post_break` is 2010-11 (§1's scan), `three_point_era` is 2012-13 (the window the
availability head ships, carried for comparability), and `post_2014` is §6's own claimed
break, on the ladder so its premise is tested rather than inherited.

| arm | rows | CRPS | vs incumbent [95%] | PIT KS | predictive sd | ρ | ρ spread |
|---|---|---|---|---|---|---|---|
| `carry_forward` *(floor)* | 8,306 | 161.289 | +17.062 [+11.17, +23.16] | 0.1242 | 330.84 | 0.0623 | — |
| `full__shared` *(incumbent)* | 8,306 | **144.228** | — | 0.0737 | **302.04** | 0.0501 | 1.00× |
| `full__role` | 8,306 | 142.555 | −1.672 [−2.55, −0.80] | 0.0652 | 297.46 | 0.0501 | 2.12× |
| `post_break__shared` | 4,187 | 142.407 | −1.821 [−2.86, −0.77] | 0.0554 | 284.24 | 0.0442 | 1.00× |
| `post_break__role` | 4,187 | 140.622 | −3.606 [−4.88, −2.24] | 0.0360 | 275.06 | 0.0442 | 2.72× |
| `three_point_era__shared` | 3,517 | 141.836 | −2.392 [−3.63, −1.08] | 0.0587 | 284.70 | 0.0445 | 1.00× |
| `three_point_era__role` | 3,517 | 139.554 | −4.674 [−6.16, −3.11] | 0.0420 | 273.92 | 0.0445 | 2.92× |
| `post_2014__shared` | 2,832 | 141.540 | −2.687 [−4.19, −1.14] | 0.0537 | 277.06 | 0.0421 | 1.00× |
| **`post_2014__role`** | 2,832 | **138.911** | **−5.317 [−7.02, −3.61]** | **0.0392** | 265.67 | 0.0421 | **3.03×** |

**The reference is the shipped head, and that is checked rather than asserted.**
`full__shared` reads CRPS **144.228** and predictive sd **302.04** against the Stan head's
own season-unit figures in `minutes_unification.csv` — **144.352** and **302.75** when this
was written. The two differ by the posterior over `β`, which the point MLE collapses to its
mode, and by nothing else, so an arm that moves either column here moves it there.

⚠️ **That control lapsed on 2026-08-13 and the artifact now reads 136.603 and 277.23.** The
Stan head gained the preseason block (`docs/preseason-plan.md` P3) and this ladder did not, so
the two sides are no longer the same head and the agreement above is a historical reading
rather than a live check. Nothing on the ladder moves — every arm here is the point MLE on
`build_design`, which carries no preseason column — but **the ladder's incumbent is no longer
the head that ships**, and re-running the window axis against a preseason-armed reference is
the honest version of this check. It is not done.

**Every arm on the ladder beats the incumbent on validation with an interval clear of zero.**
Read alone, that says ship the shortest window with the graded dispersion. §3 says otherwise
about half of it.

### The dispersion is strongly role-graded, and that is the headline of this table

| arm | `<12 mpg` | `12-24` | `24-30` | `30+ mpg` | spread |
|---|---|---|---|---|---|
| `full__role` | 0.06130 | 0.06055 | 0.05306 | 0.02894 | 2.12× |
| `post_break__role` | 0.06203 | 0.05288 | 0.04481 | 0.02283 | 2.72× |
| `three_point_era__role` | 0.06249 | 0.05392 | 0.04439 | 0.02140 | 2.92× |
| `post_2014__role` | 0.05997 | 0.05157 | 0.04096 | 0.01982 | **3.03×** |

Monotone in role in every window, and **larger than any other graded dispersion in the
project**: availability's is 1.26–1.54× and the composition's is 2.07×. The direction is the
same everywhere — a fringe player's season minutes are far less predictable than a star's —
but here it is a factor of three, and a single ρ for every player is pricing a `30+ mpg`
starter's season at three times its own dispersion.

### Two costs the CRPS column does not show

**Bias worsens on every short window.** `full__shared` reads **−14.78** minutes of
season-total bias against `post_2014__role`'s **−22.56** — the short windows predict ~8
minutes further low per player-season while scoring better on CRPS and PIT. The mean is not
what a window buys.

**Central coverage degrades while tail coverage holds.** Realized 50% interval coverage runs
0.5768 (`full__shared`) → 0.5283 (`post_2014__role`), moving *away* from nominal in the arm
with the better PIT KS, while 95% coverage holds at 0.9609 → 0.9528. The graded, windowed arm
is better calibrated overall and slightly too sharp through the middle.

---

## 3. The rolling-origin confirmation — and it splits the two axes

Eight arms on 742 validation player-seasons is a multiplicity problem and a power problem at
once. The confirmation answers it **without spending validation twice**: an origin walks
across the fitting half, each arm fits on the seasons before it and scores the season itself,
and the rows pool. **13 origins, 4,517 scored rows**, none of them a validation or held-out
row. It also reparameterizes the window as a **lookback length**, which is the thing that
still means something when the production fit runs for 2026-27.

| arm | mean fit rows | CRPS | vs `all__shared` [95%] | origins won | PIT KS | predictive sd | mean ρ |
|---|---|---|---|---|---|---|---|
| **`8__role`** | 2,690 | **155.668** | **−1.778 [−2.36, −1.20]** | 12/13 | 0.0414 | 294.84 | 0.0482 |
| `12__role` | 3,972 | 155.708 | −1.738 [−2.20, −1.28] | 12/13 | 0.0406 | 302.44 | 0.0502 |
| `all__role` | 5,839 | 156.070 | −1.376 [−1.73, −1.00] | **13/13** | 0.0440 | 312.55 | 0.0530 |
| `5__role` | 1,699 | 156.303 | −1.143 [−1.87, −0.40] | 9/13 | 0.0426 | 288.60 | 0.0463 |
| `12__shared` | 3,972 | 157.030 | −0.416 [−0.78, −0.07] | 9/13 | 0.0379 | 306.05 | 0.0502 |
| `3__role` | 1,024 | 157.109 | −0.337 [−1.29, +0.57] | 8/13 | 0.0418 | 281.98 | 0.0452 |
| `8__shared` | 2,690 | 157.367 | −0.079 [−0.59, +0.43] | 6/13 | 0.0356 | 299.85 | 0.0482 |
| `all__shared` *(reference)* | 5,839 | 157.446 | — | — | 0.0456 | 314.78 | 0.0530 |
| `5__shared` | 1,699 | 157.555 | +0.109 [−0.56, +0.76] | 7/13 | 0.0419 | 294.32 | 0.0463 |
| `3__shared` | 1,024 | 158.439 | +0.993 [+0.14, +1.83] | 5/13 | 0.0419 | 290.66 | 0.0452 |

**1. The role-graded dispersion replicates and the window does not.** Matched by fit-row
count — rows rather than nominal seasons are what a window actually trades:

| axis | validation | rolling origin | origins won |
|---|---|---|---|
| window `three_point_era` ↔ lookback 12 | −2.392 [−3.63, −1.08] | −0.416 [−0.78, −0.07] | 9/13 |
| window `post_2014` ↔ lookback 8 | −2.687 [−4.19, −1.14] | **−0.079 [−0.59, +0.43]** | 6/13 |
| role-graded ρ, window held | −1.672 [−2.55, −0.80] | **−1.376 [−1.73, −1.00]** | **13/13** |

Role grading agrees between the two readings to within 0.3 CRPS minutes and wins **every
origin**. The window's validation gain is five to thirty times what the fitting half
supports, and at the ladder's own best window it is an interval spanning zero and a
coin-flip win rate. **The ladder's headline arm is mostly the axis it was crossed with.**

**2. The window's own curve does have an interior optimum, and it is shallow.** Among the
graded arms CRPS falls from `all` (−1.376) to 12 (−1.738) to 8 (−1.778) and back up at 5
(−1.143) and 3 (−0.337) — the same bias–variance shape the availability harness found at the
same lookback of 8, but worth about 0.4 CRPS minutes on a 156-minute base, against role
grading's 1.4.

**3. ρ falls monotonically with the window, and that is not where the win is.** Mean fitted ρ
runs 0.0530 (`all`) → 0.0502 (12) → 0.0482 (8) → 0.0463 (5) → 0.0452 (3), a **−15%** move
across the range — the dispersion *does* drift, exactly as §1's series says the population
does. But the shared-ρ arms that capture that drift and nothing else are a wash on the
fitting half. Pooling across **eras** is not the defect; pooling across **players** is. This
is the same verdict `docs/availability-window-plan.md` §4b reached about its own ρ, arrived
at from the opposite direction — there the era pooling was falsified because ρ *did not*
move, here because it moves and does not pay.

---

## 4. The stake — measured, and it is a null

`make minutes-unification` ships the marginal head **solely** for its season-level spread,
and injects a per-(player, season) effect into the composition to close the gap. The
hypothesis §6 and §9 recorded: a short-window refit narrows the marginal predictive, so the
composition needs **less** injected σ to draw level, and `sim.minutes.player_season_sigma =
0.450` moves.

`injection_restake` measures it directly. The composition's persisted posterior is
rehydrated and never refitted; the σ grid is simulated once and each marginal arm is a
different reference in the same paired bootstrap, since the composition's draws do not depend
on what they are compared to.

**Each window's marginal head at the season unit**, on the 742 player-seasons both heads
cover:

| arm | CRPS | PIT KS | predictive sd | vs `full` |
|---|---|---|---|---|
| `full__shared` | 144.228 | 0.0737 | **302.04** | — |
| `post_break__shared` | 142.407 | 0.0554 | 284.24 | −5.9% |
| `three_point_era__shared` | 141.836 | 0.0587 | 284.70 | −5.7% |
| `post_2014__shared` | 141.540 | 0.0537 | 277.06 | −8.3% |
| `post_2014__role` | **138.911** | **0.0392** | 265.67 | **−12.0%** |

**The first half of the hypothesis holds: a short window does narrow the predictive**, by
5.7% to 8.3%, and by 12.0% once the dispersion is graded as well. But the tie boundary moves
the **wrong way**:

| reference arm | σ band that ties | shipped σ = 0.450 |
|---|---|---|
| `full__shared` | **[0.200, 0.525]** | ties |
| `post_break__shared` | [0.250, 0.525] | ties |
| `three_point_era__shared` | [0.250, 0.525] | ties |
| `post_2014__shared` | [0.250, 0.525] | ties |
| `post_2014__role` | **[0.300, 0.450]** | ties |

*(grid step 0.050; the injection loses at both ends — under-dispersed below the band,
over-dispersed above it.)*

**The mechanism is that the verdict reads CRPS, not spread.** A short window improves the
marginal head's *accuracy* (−5.3 CRPS) more than it narrows its *predictive* (−12%), so it is
a **harder** reference to tie, not an easier one. The composition needs **more** injected σ
to reach it: the tie boundary rises from 0.200 against the incumbent to 0.300 against the
best arm.

**Three consequences.**

**1. `sim.minutes.player_season_sigma = 0.450` does not move, and could not have.** The
shipped σ is selected by the **composition's own** CRPS optimum on training rows
(`minutes_unification.estimate_sigma_on_train`) — the marginal head appears nowhere in that
estimator. So no property of the marginal head can move it under the rule that chose it. §6
and §9 stated the stake as though σ were calibrated *against* the marginal head; it is not,
and noticing that is most of the answer. What the marginal head can move is the tie boundary,
which is a statement about the *verdict*, not about the constant.

**2. The shipped σ still ties every arm, but the margin has thinned.** Against
`post_2014__role` the tie band collapses to **[0.300, 0.450]** and 0.450 is its **upper
edge**: one grid step higher (0.525) loses. So the constant survives a stronger marginal head
without change, and would not survive one much stronger than that.

⚠️ **That last clause was a prediction and it came true within the week.** The preseason block
shipped on `stan_minutes` on 2026-08-13 and took the *actual* head to season-unit CRPS
**136.603** — stronger than `post_2014__role`, the strongest arm on this ladder. Re-read on
2026-08-14, σ = 0.450 no longer ties it: **+6.26 [+0.92, +11.49]**, an interval clear of zero,
and σ = 0.375 is the nearest tie left at +5.57 [−0.07, +11.06]. The table above is unchanged
and still correct about what it measures — every arm on it is the point MLE on `build_design`,
which carries no preseason column — but the reference it is quoted against is no longer the
head in the chain. **The constant still does not move**, for the reason finding 1 gives; the
*verdict* did.

**3. The round moves the retirement question in the opposite direction from the one it was
opened for.** §9 item 1 was written in the hope of weakening the marginal head's one
remaining claim. Instead the head gets materially better — CRPS 144.23 → 138.91 and PIT KS
0.0737 → 0.0392 — and its PIT was then better than **every** injected composition arm's,
including the shipped σ's and the grid optimum's. Retiring `stan_minutes` looked a *less*
live prospect after this round than before it. ✅ **Confirmed twice over by the preseason
block**, which was a second thing the marginal head carried and the composition did not,
worth 7.75 CRPS minutes at the unit the two are compared at.

⚠️ **Both halves of that expired on 2026-08-14, and the question turned out to be
mis-framed.** The composition took its own preseason block (`docs/preseason-plan.md` P5), so
the second claim is gone; and at the re-estimated σ = 0.375 the injected composition now
*beats* the marginal head (CRPS 130.692 against 136.60) at a PIT KS of **0.0665499** against
**0.0668** — indistinguishable, where this finding's whole point was that the marginal head
owned calibration. At the previously shipped σ = 0.450 the injected arm's PIT is
**0.0952291** and the un-injected composition's predictive sd is **60.5757**.

**But none of that decides the retirement, because the premise underneath it is false.**
`src/sim/` imports neither `StanMinutes` nor `rehydrate_minutes` and never looks up
`artifacts["minutes"]` — the simulator's minutes have come from the composition plus the
injected σ all along, and what it takes from this module is `beta_shapes` (arithmetic),
`game_level_dispersion` (a data measurement in which the fitted object never appears) and two
Gate bars read from artifacts. Retiring the head therefore means ceasing to **fit** it, and
the reason to keep doing so is that it is the `independent_comparator` in
`stan_composition`'s ladder and the season-unit reference σ is calibrated against — at 514 s
against the composition's 7,357 s. **A head that loses is still the instrument the winner is
measured with.** The development that would genuinely retire it is a *fitted* `sigma_u`.

---

## 5. What ships, and what this leaves

**Nothing ships from here**, in the same sense `availability_window` ships nothing: this is a
point-MLE ladder, and an arm that wins here earns a Stan port rather than a place in the
chain. The recommendation it produces is nonetheless specific, and it is not the one the
round was opened to test.

**1. Grade the dispersion; do not move the window.** Role-graded ρ is the finding: −1.672
[−2.55, −0.80] on validation, −1.376 [−1.73, −1.00] on the rolling harness, **13 of 13
origins**, and a 2.12–3.03× spread that is the largest in the project. The window is
−0.079 [−0.59, +0.43] on the fitting half at the ladder's own best window, and its validation
margin does not replicate. In Stan this is one change: `betabinomial_glm.stan` already takes
`rho` as a vector indexed by a data-supplied bin — `stan_availability` ships that arm and
`stan_minutes` currently calls `rho_block(len(train))` with no bins, which is `n_rho = 1` and
the shared model exactly. The port is switching that call to `role_bins`, and `n_rho = 1`
already reproduces the incumbent bit for bit.

**2. The stake is closed as a null.** `sim.minutes.player_season_sigma = 0.450` stands.
Recorded so it is not re-opened: the constant is not calibrated against the marginal head, so
no window on the marginal head can move it. ✅ **Re-tested 2026-08-14 by a change that came
from outside this axis** — the preseason block — and the rule held exactly: σ's train grid
reproduced its 0.450 optimum unchanged, because the marginal head is not in that estimator.

**3. What a graded-ρ port would owe.** Two things this ladder cannot answer. The point MLE
profiles ρ per bucket holding the mean fixed; the Stan port fits them **jointly**, and on
availability that trade cost a little calibration for a little sharpness (PIT KS 0.0588 →
0.0679) and graded ρ slightly harder at both ends. And the whole-board consequence is
unmeasured here: a graded ρ narrows a star's season predictive by a factor of three relative
to a fringe player's, which lands directly on how a simulated draft board's top end spreads.

**4. Two things the ladder deliberately did not cross.** A **season term**, because
`season_terms` already measured a year effect for this head and it ships one — and §1's
finding that the drift has reverted is an argument for re-measuring that, not for adding a
trend. And the **composition's** window, which stays the low-priority one: §1 finds no recent
contraction on its rows at all, its unit is a share of a fixed pot, and its 9.92 h fit is the
most expensive thing in the project.

**5. The reversion in §1 is a live question of its own.** Cross-player sd on the head's own
rows has risen in each of the last four measured seasons, from 0.1563 (2019-20) to 0.1824
(2023-24), and the composition's is at a twelve-season high. Every window arm in §2 fits a
period that includes the trough and predicts two seasons that are past it. If the reversion
continues, the window's *sign* flips — and the two held-out seasons are exactly where that
would first be visible, which is a reason to look at it once, at the end, rather than now.
