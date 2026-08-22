# Potential to-dos

A parking lot for ideas that are **measurable and not yet scheduled**. Each entry says what
to compare, what evidence already points at it, and what would settle it — enough to start
from without re-deriving the motivation. Nothing here is a commitment, and nothing here is a
result.

This file is deliberately **not** in `make docs-audit`. The figures it quotes are either
already audited from their own artifacts elsewhere, or are scratch measurements labelled as
such; an entry only earns audited claims once it has been built and has a `make` target
behind it. See `docs/docs-audit.md`.

---

## 1. Fit on the three-point era rather than on 1996-97 onward

**Compare the current `train` fits against fits trained on a recent suffix of the available
seasons.** Every head currently fits from 1996-97; the question is whether the early seasons
are contributing anything, or whether they are averaging a different sport into the
coefficients.

### Why this is worth measuring

Three things point the same way, and the first is already on disk.

**The extra seasons appear to buy nothing.** `make composition-effects` fitted the shipped
composition specification on the pilot window (2018-19 onward, 4 training seasons, 97,587
rows) and scored it on the **same 742 validation player-seasons** the full-window incumbent
is scored on:

| | pilot: 4 train seasons | full: 26 train seasons |
|---|---|---|
| per-team-game CRPS | **4.4561** | 4.4945 |
| per-team-game R² | **0.4758** | 0.4741 |
| per-team-game PIT KS | **0.0311** | 0.0428 |
| season CRPS | 171.57 | **170.06** |
| season MAE | **199.05** | 200.28 |
| season R² | **0.8855** | 0.8848 |
| season predictive sd | 58.81 | **64.65** |

The shorter window wins five of seven, including the per-team-game metrics the head's own
sweep selects on. **Read this as a prompt, not an answer**: it is one arm, the two fits ran
at different iteration counts (500+500 against 1000+1000), and there is no bootstrap on a
0.04 CRPS gap.

**The target season is 2026-27, and the early seasons are a different game.** League
three-point attempt share, computed from `component_targets.parquet`:

| era | seasons | 3PA share | movement |
|---|---|---|---|
| shortened line | 1996-97 | 0.2117 | the NBA's 22-foot experiment, its final season |
| line moved back | 1997-98 | 0.1593 | **−0.0524**, the largest single-season move in the series |
| slow drift | 1998-99 → 2011-12 | 0.1682 → 0.2255 | +0.057 over 14 seasons |
| **the modern era** | **2012-13 → 2025-26** | **0.2434 → 0.4144** | **+0.171 over 14 seasons** |

**2012-13 is the natural breakpoint**: it is the first season with a jump above +0.017 after
a decade of drift, and the acceleration is sustained rather than a blip. A secondary
observation worth its own line — **1996-97 is from a different rule regime entirely.** The
league shortened the three-point line for 1994-95 through 1996-97 and moved it back for
1997-98, and the −0.0524 drop is the largest movement in thirty seasons of data. The first
season in this project's dataset is not comparable to the second.

**And it is the largest available cost lever on the fitted `sigma_u` work.** Extrapolated
from the measured Gate A probe, one random-effect composition arm costs **37.6 h** at the
full window against **6.3 h** at 2018-19 onward — a **6.0×** reduction from the window alone,
before any sampler change. It also brings the dense mass matrix from 1.2 GB down to 38 MB,
which makes `dense_e` available again (`stan_composition.choose_metric`). Those compound.

### What to compare, and how

A window ladder on the cheap heads first, since the point is whether the coefficients move
at all and that does not need the expensive one:

| window | first season | rationale |
|---|---|---|
| `full` | 1996-97 | the incumbent |
| `post_line_change` | 1997-98 | drops only the anomalous rule regime — the cheapest possible test |
| **`three_point_era`** | **2012-13** | the measured breakpoint |
| `recent` | 2018-19 | the composition pilot window, already measured for one arm |

Score every arm on the **same** validation seasons, at **matched iteration counts**, with a
paired bootstrap on the differences — the discipline the availability ladder already uses,
and the absence of which is why the table above is a prompt rather than a finding. Run it on
`stan-availability` and `stan-components` before the composition: they are minutes rather
than hours, and if the coefficients are stable there the case for the expensive head is much
stronger.

### One implementation wrinkle to handle first

`stan.composition.first_season` is read by **both** `stan_composition.run` — whose artifact
`make docs-audit` re-derives eleven quoted figures from — and
`posteriors.composition_artifact`. Changing it to re-scope the production fit would silently
re-scope the incumbent's sweep too, and the audit would fail on a bookkeeping change rather
than a measured one. Split the key before using it, or accept re-running the sweep and
refreshing every claim behind it.

### What would falsify it

The early seasons earning their keep would show up as the long-window fit winning on the
paired bootstrap at the **season** unit, which is where the composition's shorter window was
marginally worse (CRPS 171.57 against 170.06, predictive sd 58.81 against 64.65). That gap is
inside the noise on one arm; a proper ladder would say whether it is real. The other way this
fails is coverage rather than accuracy — fewer seasons means fewer player-seasons for the
expanding rookie prior and for any head with a sparse tail, and that is worth checking
directly rather than assuming.

**Note that only the *fitting* window would shorten.** `composition_frame` is built over all
30 seasons regardless, because the lags reach back three seasons and the rookie prior is an
expanding point-in-time average — and it costs 1.9 s, so there is no reason to trim it.

---

## 2. Finish the fitted `sigma_u` — where the 2026-08-09 session stopped, and why

**→ 2026-08-16: this entry now has a workplan, and the primary route changed.**
`docs/composition-quadrature-plan.md` supersedes the "what to try next" ladder below:
rather than making the 2,204-latent posterior affordable, it removes the latents by
per-unit quadrature (parameters ~35 at any window, `dense_e` back in reach) and grades
σ by `rho_bin` — motivated by a per-role PIT scratch measurement showing the shared
σ = 0.375 leaves fringe units 1.73× underdispersed while overdispersing stars at 0.85×.
A post-blend re-attempt of the latent `ps` arm is also in flight (Gate A: 14.6 h);
either result feeds that plan's step 4. The ladder below is kept as the record of the
latent route's cost and failures.

**Everything is built and tested; what is missing is a converged fit.** The capability
shipped that day: the optional `sigma_u` block in `composition_glm.stan` with `U_n = 0`
nesting the shipped head exactly, `PlayerSeasonTerm`, the team-context join,
`make composition-effects`, `make posteriors` persisting the scale, and
`rehydrate_composition` applying it. The **injection** ships in its place
(`sim.minutes.player_season_sigma = 0.450`, estimated on train), so nothing downstream is
blocked.

### Where it stopped

The `ps` arm was fitting at the **pilot window — 4 training seasons, 2018-19 → 2021-22**,
97,587 rows, 9,198 team-games, **2,204 player-season units**, at 500 + 500 × 4 chains. It was
killed after **2 h 20 m of wall clock with its chains 98 minutes into warmup and zero of
1,000 sampling draws produced.** Extrapolating the one-season probe's warmup by rows, units
and iteration count puts warmup alone near 4.5 h and the arm near **9 h** — the pessimistic
end of the 4–9 h Gate A range, at a window already 6× cheaper than the full one.

`base` completed and its row is on disk. `_flush` merges by arm, so **re-running
`arms: [ps]` resumes without re-spending `base` or Gate A.**

### Four sampler-side levers were tried and all four failed

Same one-season frame, features and iterations; only the named thing varies. Read the
diagnostics, not the wall clocks — up to three fits shared 14 cores at various points.

| arm | varies | max R̂ | min ESS | treedepth-sat | verdict |
|---|---|---|---|---|---|
| `base` | no effect | 1.0172 | 400 | 0/800 | the head without the parameter |
| **`ps`** | **shipped spec** | **1.0948** | **35** | 17/800 | **best of the effect arms** |
| `ps_centered` | centred | 1.1067 | 27 | 212/800 | worse |
| `ps_shared_rho` | shared `rho` | 1.1390 | 20 | 0/800 | worse |
| `ps_no_rho` | no `rho` | 1.3289 | 11 | **791/800** | much worse |
| `ps` + `dense_e` | dense metric | — | — | — | killed past 1 h against `diag_e`'s 25 min |

**`rho` is helping, not competing.** The hypothesis was a ridge between `rho` and `sigma_u`
absorbing the same overdispersion; removing `rho` makes it dramatically worse, because
stripping the dispersion sharpens the binomial likelihood, pins each unit's `u_z` by its own
rows, and degrades the geometry. Grading `rho` also beats sharing it.

**The dense metric fails on estimability, not memory.** It adapts a `P × P` covariance from
the warmup draws, so it needs draws on the order of the parameter count — 1.57 per parameter
at 605 units and 0.45 at 2,204, against 38.5 for the effect-free head. Below one, the
adaptation is rank-deficient and CmdStan shrinks it back toward diagonal.
`stan_composition.choose_metric` now gates on both constraints.

**So the cost is intrinsic to the parameter.** A 2,204-parameter hierarchical posterior is
one NUTS walks slowly, and no configuration tried changes that.

### What to try next, in order

1. **`reduce_sum` threading.** The only untried lever and the one that needs no modelling
   decision: the likelihood is one vectorized call plus **1,487** scalar
   `beta_binomial_log_tail_mass` calls per gradient at the pilot window, each an
   O(m − lo) autodiffed `log_sum_exp` recurrence, against 4 chains on 14 cores.
2. **Cheapen the truncation term.** Those 1,487 rows are 1.7% of the frame and exist only
   because `lo > 0` in short-rotation games. Measure what dropping the normalization costs
   before optimizing it.
3. **Longer chains, not different ones.** Every arm failed on ESS rather than divergences,
   which is under-mixing rather than broken geometry — the one honest response to which is
   more iterations, and the reason the 9 h estimate is a floor.
4. **Accept the injection permanently** if none of the above lands. It ties the marginal
   head, and what a fit uniquely buys — β estimated jointly, and the predictive integrating
   over σ's posterior instead of plugging one in — should be weighed against ~9 h per arm
   per window rather than assumed worth it.

### One measurement worth taking regardless

**The effect trades season-level spread against subset teammate coupling, and nobody has
priced the trade.** Measured at σ = 0.450 against σ = 0: the mean pairwise teammate
correlation weakens from **−0.0509 to −0.0388** (the constraint forces −0.0664), while the
team's season total stays **exactly fixed** — 0.0000 predictive sd on complete team-game
blocks at both.

✅ **Re-read 2026-08-16 at the shipped values, which is `docs/draw-time-calibration-plan.md`
step 3's own deliverable, and grading recovers about a quarter of the dilution.** On the same
963 single-team validation player-seasons, `make minutes-role-sigma`:

| arm | mean pairwise teammate r |
|---|---|
| un-injected (σ = 0) | **−0.0504** |
| shared σ = 0.375 | −0.0392 |
| **graded [0.600, 0.375, 0.375, 0.300]** | **−0.0422** |
| what a fixed team sum forces | −0.0664 |

The dilution is 0.0112 at the shared value and **0.0082** graded, so **27%** of it comes back
— for free, from a change made for a different reason. The mechanism is the obvious one: the
buckets whose σ *fell* (star) dilute less, and a star's minutes are what a stack is usually
built around. The trade is still unpriced in the unit that matters — only the strategy sweep
can say whether stacking and handcuffing care — and the σ = 0.450 row above is kept because it
is the reading the original entry was written against. The dilution is real and confined to the single-team *subset*, which is
what a 2–3 player stack is, and it is a property of **having** a per-player-season effect
rather than of injecting one — a fitted `sigma_u` draws independent `u_z` too and would
dilute identically. Stacking and handcuffing sit on one side of that trade and the
season-unit calibration on the other, and only the strategy sweep (build item 8) can say
which the tournament objective cares about more.

---

## 3. Do the three games-played binomial heads over-predict their own observable?

**Compare each tenure head's own reported mean against the realized mean of what it models,
then ask whether the gap survives composition into `SpellProcess`.** Surfaced 2026-08-10 by
`make model-cards`, which drew these heads' predictives for the first time — nothing in the
project had, because the tenure decomposition scores its *composite* through `predict_pmf`
and never the parts.

### The measurement, on the training frame at the `train` posterior window

| head | models | head's own mean | realized | gap |
|---|---|---|---|---|
| `gp_onset` | absence spells started, out of at-risk games | 5.300 | 4.081 | **+29.9%** |
| `gp_exit` | games after tenure ended, out of exit trials | 6.802 | 5.828 | **+16.7%** |
| `gp_entry` | games before tenure began, out of entry trials | 4.506 | 4.269 | +5.6% |
| `availability` | games played, out of team games | 54.518 | 55.247 | −1.3% |

Aggregated over trials rather than rows the picture is the same — `gp_onset` puts the onset
rate at 0.0962 against a realized 0.0741 — so it is not row-weighting.

### It is the head, not the emitter, and that is already checked

`model_cards.predictive_bias` compares the *drawn* predictive against the head's own
`predict` and reads **−0.06%** for `gp_onset`; the posterior mean of `mu` agrees with the
plug-in one to within 0.1%. So the card is faithfully reporting what the head says.

### The suspect

A beta-binomial with heavy dispersion on a target with a large zero mass — 64% of `gp_exit`
rows are zero, 15% of `gp_onset` rows — does not pin its fitted mean to the empirical mean the
way a binomial GLM with an intercept does: its score equation for the intercept is a
digamma expression, not a residual sum. Under a `Beta(a, b)` frailty with small shapes the
likelihood is happiest putting mass at both ends, and the *mean* of that fit can sit well
above the mean of the data it fits.

### What would settle it

1. **Does it survive composition?** `SpellProcess` combines entry, exit, onset and duration
   into a games-played pmf, and that composite is what ships and what `make games-played`
   scores against its floor. Score the composite's mean against realized games played on the
   same rows; if the composite is centred, the parts being individually off-centre is a
   property of the decomposition rather than a defect in it.
2. **Is the ECDF calibrated where the mean is not?** A U-shaped beta-binomial can miss the
   mean and still get the distribution roughly right, which is exactly the case where CRPS
   and PIT pass. `model_card_ecdf.csv` already carries the curve; read the deviation from
   `q50` rather than the in-or-out verdict, per `docs/model-cards-plan.md`.
3. **Only if both fail**: refit the onset head with the mean pinned — an offset, or a
   quasi-likelihood — and compare CRPS on validation against the incumbent. That is a real
   model change and needs the ladder treatment, not a patch.

This is a measurement, not a defect report: the heads clear their own gates as scored today,
and nothing above has been through validation.

## 4. ✅ Go back over the games played models — MEASURED 2026-08-11, moved to its own doc

Was: *"The beta binomial model predictions are under-predicting the frequency of players
playing all 82 games and also under predicting the frequency of players playing very few
(< 10) games."*

**Half of that is right and half is backwards, which is what told us the mechanism.** The
low half holds — the head puts **5.21%** of validation player-seasons below ten games
against an observed **8.15%**. The high half runs the other way: it puts **5.95%** at a full
schedule against an observed **2.72%**, so it *over*-predicts an iron man by 2.19×. Both are
outside the posterior-predictive band, and both are wrong on the fitting half too. One
defect, two symptoms — the fitted Beta frailty is too **U-shaped**, with too much mass on
both boundaries and too little in the shoulders.

`make availability-window` is the ladder that separated the candidate causes, and
**`docs/availability-window-plan.md` is now the live document** — the era break test, the
window × trend × dispersion results, what ships, and the same question carried over to the
minutes heads. Three things it settled, so they are not re-opened here:

- a shorter **fitting window** (2012-13 on) plus a **role-graded `rho`** wins CRPS by
  −0.181 with a clear paired interval, wins PIT, and cuts boundary error 43%;
- a **season trend** is a null — it closes both boundaries by shifting location and wrecks
  the body of the distribution doing it;
- **`rho` is not where the era lives** (it moves −6.4% across windows); the low tail
  survives every instrument and is a functional-form limit.

What remains open is item 3's question and the low tail's likelihood — both carried in that
doc's "what this leaves" section rather than here.

---

## 5. ✅ A different frailty for the availability head — MEASURED 2026-08-11

**All five arms and the free arm are measured. `docs/availability-window-plan.md` §7 is the
live document**; what follows is the motivation that opened the item, kept because the
mechanism it derived is what the ladder then tested. Four things it settled, so they are not
re-opened here:

- **The free arm is a null.** The tenure decomposition loses on the boundary too —
  `boundary_tail_error` **0.0333** against the shipped head's 0.0202, the worst of any arm —
  and makes the *same two errors on the same sides*. The **7.2265** cited below is the
  `within_tenure` **oracle** arm (`selectable: False`, 751 rows), not a forecast. §7a.
- **The low tail is not a frailty phenomenon.** `mixture` halves the selector to **0.0109**
  and ties on CRPS; `logitnormal`, which removes the divergence entirely, fails the *other*
  way and is a worse fit of the same data at the same parameter count. A missing component,
  not a wrong shape. §7c results 1–2.
- **The one-parameter control wins CRPS** (`beta_rect`, −0.056 [−0.096, −0.015]) and loses
  the boundary to `mixture`. Tail mass is what CRPS wants; *which* tail is what the boundary
  wants. §7c result 3.
- **The divergence share moved but is not the mechanism** — 52.1% of predictive mass →
  32–38%, while `ρ` falls at every bucket without collapsing. §7c result 5.

Nothing ships from there: it is a point-MLE ladder, and an arm that wins earns a Stan port in
a later session.

---

**The motivation, as written when the item opened.** Compare the beta-binomial's Beta frailty
against likelihoods whose boundary behaviour is a *separate* parameter from its dispersion.
Opened 2026-08-11, after the windowed role-graded
head shipped and `model_card_ecdf.csv` showed the defect it was refitted for had **narrowed
without closing**: on validation P(GP < 10) went 5.21% → **5.66%** against an observed 8.15%,
and P(GP ≥ 82) went 5.95% → **4.30%** against an observed 2.72%, both still outside the 95%
band (`docs/availability-window-plan.md` §1b).

This is the item `availability-window-plan.md` §5.3 opens with three candidates. It is here
rather than there because the mechanism is now measured, which changes what the candidates
should be.

### Why: the head has one shape knob and the data wants two

Under the shipped parameterization `a = μ(1−ρ)/ρ` and `b = (1−μ)(1−ρ)/ρ`, so the frailty's
*shape* and its *variance* are the same parameter. **`b < 1` makes the Beta density diverge at
`p = 1`** — an integrable spike sitting exactly on "played every game" — and that happens
whenever

```
ρ > (1 − μ) / (2 − μ)
```

A **scratch measurement** over the persisted `train` posterior (posterior-mean `μ` and `ρ`,
883 validation rows; five lines of numpy over `posteriors/train/availability.pkl`, no `make`
target behind it yet):

| bucket | n | μ | ρ | a | b | share with b<1 | realized P(GP = team_games) |
|---|---|---|---|---|---|---|---|
| `<12 mpg` | 153 | 0.416 | 0.3176 | 0.89 | 1.25 | 11.8% | 0.7% |
| `12-24` | 363 | 0.619 | 0.2701 | 1.67 | 1.03 | 48.8% | 2.2% |
| **`24-30`** | 169 | 0.732 | 0.2535 | 2.15 | **0.79** | **82.2%** | 3.6% |
| **`30+ mpg`** | 198 | 0.757 | 0.2067 | 2.91 | **0.93** | **66.2%** | 4.5% |

**52.7% of validation rows carry that spike, and it concentrates where the statistic lives.**
For a 30+ mpg player the threshold is ρ > 0.196 and the fitted value is **0.2067** — the head
needs that dispersion to match the observed variance, and any value above 0.196 puts a
divergence at a full schedule. One parameter is being asked to set the variance and the
boundary behaviour at once, and the data wants them in opposite directions. That is why
`three_point_era__none__role` could only halve the miss: it moved ρ, and ρ is the wrong knob.

It also explains the *asymmetry* the ECDF shows. Only 13.5% of rows have `a < 1`, so the head
has a spike at the top for half the league and one at the bottom for a seventh of it — while
the observed data wants the reverse.

### What to compare, in cost order

| arm | what it frees | nests the incumbent at | cost |
|---|---|---|---|
| **re-score the tenure decomposition** on `boundary_tail_error` | nothing — it is already fitted | n/a | **zero**, the pmf is on disk |
| **disrupted-season mixture** `π_i·(low) + (1−π_i)·BetaBinom` | the low tail, as a separate event | `π = 0` | one Stan block |
| **finite-mixture (Heckman–Singer) frailty**, K latent classes | the whole shape | `K = 1` | `log_sum_exp`, ordered means |
| **logit-normal frailty** (binomial GLMM) | boundary behaviour — it *cannot* diverge | — | one non-centred block |
| beta-rectangular, `θ·U(0,1) + (1−θ)·Beta` | tail mass, symmetrically | `θ = 0` | one parameter, a control |

**1. Re-scoring the existing tenure decomposition is free and might already be the answer.**
`stan_games_played` *is* the structural alternative — entry index × exit index × within-tenure
chain, built because "a departure is an absorbing hitting time, not a low recovery rate" — and
it has never been compared on this statistic. Gate D was CRPS-shaped, `stan_games_played_gp_pmf.csv`
is already an artifact, and §5.3 records that its oracle-tenure arm scores **7.2265** against
the incumbent's 10.0057. A head that loses on the mean and wins on the boundary is the same
"one posterior, two units, opposite verdicts" pattern `make minutes-unification` already found,
and it would be settled by reading a file rather than by fitting anything.

**2. The low tail is plausibly not a frailty at all.** An Achilles rupture in October is a
different event, not an extreme draw of a per-game rate. Covariates for `π` are already in
`FEATURE_COLS` (age, prior-season absence, the playoff-workload block).

**3. The finite mixture is the version that commits to no mechanism.** K support points reach
the shoulders at 2–15 and 70–80 games directly, no divergence is possible, and the
beta-binomial is its continuous limit. Note the role-graded `ρ` is already a crude version of
it — graded on *observed* prior MPG — and its measured spread is only **1.54×**, which is what
a weak proxy for a latent durability class would look like.

**4. Fit the logit-normal because it should fail the other way.** Its tails are lighter than
the Beta's at both ends, so it ought to fix the high tail by construction and worsen the low
one. That separates "the frailty's shape is wrong" from "a component is missing", which no
single arm can do on its own.

Skip Kumaraswamy, the simplex distribution and the generalized beta: different algebra, same
single-shape-knob problem.

### The deeper caveat, which points back at arm 1

**The exchangeable-trials assumption is itself suspect.** Absences come in *spells* — measured
in this repo as beta-geometric, beating the geometric by **11,278** log-likelihood points at
one extra parameter. A beta-binomial absorbs the variance inflation from that clustering but
not its shape: one 40-game spell and forty single-game absences give identical `gp` and very
different distributions. So the low tail is plausibly a *duration* phenomenon rather than a
rate one, which is the same conclusion arm 1 reaches from the other direction.

### What would settle it

`make availability-window` runs all of these as-is. **The selector must stay
`boundary_tail_error` with `body err` reported beside it** — §4 of
`availability-window-plan.md` already showed an arm can buy both boundaries by wrecking the
middle, and the arms with the best boundary coverage there were the worst models. Confirm on
the rolling-origin harness (§4b) before believing any margin, and hold every arm to the
house nesting discipline: `π = 0`, `K = 1`, `θ = 0` must each reproduce the shipped posterior,
the way `n_rho = 1` and `U_n = 0` already do.

### What would falsify it

The mixture and the finite-mixture arms buying the boundaries at the same body cost the season
trend paid, in which case the honest conclusion is that 82 games out of a fixed schedule is
simply not a beta-binomial and the tenure decomposition is the only structural answer. The
other way it fails is CRPS: an arm that fixes the shape and loses the mean is not shippable
against a head whose mean function is the largest measured lever in the project.
---

## 6. ✅ Lay the tenure edge blocks separately from the interior spells — MEASURED AND SHIPPED 2026-08-12

**`docs/availability-window-plan.md` §13 is the live document**; what follows is the
motivation that opened the item, kept because the mechanism it named is what the ladder then
tested. `sim.availability.layout = tenure_merge` ships. Four things it settled, so they are
not re-opened here:

- **The entry was right that a tenure factor was missing, and wrong about which key it
  needs.** The edge fraction is conditional on *how much the player missed*, not on his role
  — it runs 0.1321 → 0.5679 across missed-share bins while the four role buckets inside a bin
  span about three points. §11b's role gradient on the pooled edge share is mostly a
  composition effect. Role still keys the *end*: the leading block's share falls 4.2× from
  fringe to star while the trailing block's rises. §13a.
- **The falsifier this entry named fired, and it inverts the entry's remedy.** `allocate_spells`
  collapses to one block when the draw wants more spells than the schedule has gaps, and that
  fires on **41.28%** of fringe rows against **2.32%** of star rows. Removing it alone takes
  the pooled longest-dead-run recovery from 1.1162 to **0.2707** — so "stop truncating" on
  its own makes the simulator much worse, because the accident was doing most of the
  clumping §11d credited to the beta-geometric. §13b.
- **Neither half ships alone and the compound closes the sign flip.** `tenure_merge` lands
  `longest_dead_run` recovery in **[0.852, 1.039]** across role against the shipped layout's
  [0.587, 1.680], and independently reproduces the realized spell-length distribution it was
  never scored on. §13c, §13d.
- **The entry's proposed instrument — `stan_games_played`'s fitted entry and exit heads — is
  the wrong one.** They do not condition on `gp`, and the ladder holds `gp` fixed, so they
  would return blocks longer than the missed total. An empirical resample of the fitting
  rows' own `(pre/missed, post/missed)` pairs conditions on exactly what is fixed, and
  transfers to the simulator's draw order unchanged. §13a.

What remains open is the contest reading. `make simulate-season`, `make weekly-scores`,
`make bracket` and `make draft-sim` are re-run on the new tensor — Gate A unmoved to four
decimals on every games-played row, which is the arrangement being orthogonal to the season
unit, and the scoring-period gate moving the two rows it should. **`make strategy-sweep` is
not re-run, on purpose**, and `strategy_*.csv` is therefore the one stale stage in the repo.
§7l says to expect a null there and to read `bracket_ev` if it is not — but the reason for
deferring is **item 7 below**, not cost: a level change to the no-design availability rate
supersedes any sweep run before it, for the reason that entry now carries.

---

**The motivation, as written when the item opened.**
**Give `sim/season.py`'s availability layout a tenure factor.** Today
`games_played.allocate_spells` fits one pooled beta-geometric on **interior** spells — the
appearance window is its frame — and then places every drawn spell at a uniform random start
over the whole schedule. Both halves of that are wrong for the games that are not interior.

### Why this is worth measuring

`make availability-exchangeability` (`docs/availability-window-plan.md` §11) priced the
whole exchangeability axis and found the layout step already pays 63–82% of it. The residual
is the part this entry is about, and it is *named* rather than inferred:

- **44.17%** of the availability head's 85,341 missed fitting-row games are **tenure edge
  blocks** — a delayed first appearance or a trailing absence — not interior spells.
  `docs/games-played-plan.md` establishes those are an absorbing hitting time rather than a
  low recovery rate, so both their shape and their position differ from what is being drawn.
- **That 44.17% is two processes with opposite role signatures, and the target has to be
  chosen accordingly.** Split by the panel's `status`, **20.68%** of missed games are edge
  blocks the player was **not rostered** for — falling **13.3×** from fringe (36.46%) to star
  (2.75%) — and **23.50%** are edge blocks he was rostered through, i.e. preseason and
  season-ending injury, *rising* **2.5×** the other way (14.86% to 36.77%). Only the second
  is an availability event. The first is a question about the head's denominator.
- **The residual's sign flips by role, and the two causes are different.** The shipped layout
  **overshoots** the `<12 mpg` bucket's longest dead run (10.7233 against 8.0224) because
  36.46% of that bucket's missed games are one not-rostered block that the layout shatters
  into scattered spells; it **undershoots** stars (1.7370 against 2.3218) because 36.77% of
  theirs is a season-ending injury and a beta-geometric fitted on interior spells with a mean
  of 3.2261 games has no draw that long.
- **Trades are not the mechanism.** Multi-team player-seasons are excluded from the
  measurement frame (593 of 4,027 fitting rows, 14.7%), so an ending tenure here is a player
  leaving the league, not one continuing elsewhere.

### What to compare, and how

The pieces already exist. `stan_games_played` fits an **entry index** and an **exit index**
as beta-binomial heads on `betabinomial_glm.stan`, and `games_played.tenure_frame` is the
frame. So the arm is: draw pre-tenure and post-tenure lengths from those heads, place them at
the ends, and hand `allocate_spells` only the interior remainder and the interior games as
its schedule. `allocate_spells`' existing behaviour must be recoverable exactly at zero
tenure, the same nesting discipline `n_rho = 1` and `U_n = 0` already carry.

**Run the cheaper half first.** The not-rostered block is 20.68% of missed games and needs no
new head at all — it is a contiguous run at an end, and placing it there rather than
scattering it is a change to `allocate_spells`' placement step alone. It is also the half
that produces the *largest* single residual (the fringe bucket's 168% overshoot). The
still-rostered injury block is the half that needs the entry and exit heads. If the placement
change alone closes the fringe overshoot, the second half can be priced on its own merits
rather than bundled.

Score it on `make availability-exchangeability`'s own ladder, which is built for this: the
three arms become four, `gp` stays fixed at its realized value on every row, and the readout
is `recovered_share` per role on `p_dead_period`, `longest_dead_run` and `p_dead_run`. The
target is a `recovered_share` near 1.0 in **every** bucket rather than 1.68 in one and 0.59
in another.

### What would settle it

Closing the sign flip. If a tenure-aware layout brings the fringe bucket down from 168% and
the star bucket up from 69% without moving the pooled number much, the mechanism named in §11
was right and the arm is worth its cost. Then, and only then, price it downstream: it is a
change to the simulator's draw path, so it costs `make simulate-season` plus `make bracket`,
`make draft` and `make strategy-sweep`, and §7l's finding — that the drafting layer *ranks*
and so has no channel for a distributional improvement — says to expect the contest reading
to be a null and to check `bracket_ev` specifically if it is not.

### What would falsify it

The layout already being right for the wrong reason. `allocate_spells` truncates its last
spell to make the missed total exact and collapses to a single block when the drawn spells
cannot fit, so it manufactures long blocks on heavily-absent rows by accident — which is
plausibly why the fringe bucket overshoots. If a tenure-aware arm moves the fringe bucket by
less than that truncation does, the defect is in the fitting loop rather than in the missing
factor, and the cheaper fix is to stop truncating.

---

## 7. ✅ Grade the no-design availability *level* — MEASURED AND SHIPPED 2026-08-12

**`docs/availability-window-plan.md` §8b is the live document**; what follows is the
motivation that opened the item, kept because the population it named is the one the ladder
then measured. `sim.availability.no_design_level = tenure_draft` ships. Four things it
settled, so they are not re-opened here:

- **The entry was right that the level is graded and wrong about the key.** The draft bucket
  is a **3.17×** gradient for a *first* appearance (0.2613 undrafted to 0.8294 lottery top-5)
  and a non-monotone **1.74×** near-flat for a *return* (0.2200 to 0.3837), because a
  returning veteran's draft night is a decade old. Keying both classes on the bucket hands a
  returning ex-top-5 pick **0.7491** where his class realizes **0.3837** — the same "a key
  applied where its signal is not" error §8a withdrew decision 2 for, one axis over. So the
  shipped arm is the **cross**, and it beats the bucket alone on both splits
  (−0.8923 [−1.6575, −0.1332] on validation, −0.3713 [−0.5562, −0.1928] rolling). §8b.
- **The defect was larger than "one rate for a 3.3× spread" implies.** The pooled scalar
  scores validation R² **−0.0865** on its own population: worse than predicting their mean.
  The graded arm scores 0.4316 and cuts CRPS from **14.4551** to **9.8689** games. §8b.
- **The bar this entry named does not exist, and that changed what was built.** "The bar is
  Gate A … since these players are in the tensor" — they are not. **0 of 106** no-design
  players in 2022-23 are scorable units; they are in the *grid*. Every row Gate A scores has
  a bit-identical `mu` under both arms, so the only channel is the zero-sum one, and Gate A
  gained a **`no_design_team_minutes_share`** row to read it at the team.
- **It half-closes §8a's other finding for free.** Part of the population's 0.4337
  unconditional dispersion was between-class variation in the *level*; grading the mean drops
  the residual `ρ` to **0.3480** rolling, so the fringe bucket's 0.3176 fallback now brackets
  it rather than sitting 27% below it.

What remains open is the contest reading, which this shares with item 6: the tensors moved
and `make bracket`, `make draft` and `make strategy-sweep` were not re-run. §7l says to
expect a null, because the drafting layer ranks and a distributional change has no channel
through a ranking.

---

**The motivation, as written when the item opened.**
**`sim/season.no_design_availability` returns a single scalar** — the pooled `gp / team_games`
of no-design player-seasons strictly before the target — and every rostered player the
availability head has no row for gets it. An undrafted free agent and a first overall pick
are handed the same availability.

### Why this is worth measuring

`make availability-no-prior` (`docs/availability-window-plan.md` §8a) measured the population
this scalar covers, on the 2,616 no-design player-seasons selection may read:

| draft bucket | rows | realized `μ` | P(GP < 10) |
|---|---|---|---|
| undrafted | 781 | **0.2500** | **0.4264** |
| second round | 626 | 0.4036 | 0.2077 |
| late first | 435 | 0.5669 | 0.0920 |
| lottery | 247 | 0.7326 | 0.0202 |
| lottery top-5 | 138 | **0.8316** | **0.0000** |

**3.3260×** on the level, and the left tail runs from 42.6% to exactly zero. Against that,
realized *dispersion* spans 1.1557× — which is why §8a withdrew the decision that graded the
dispersion and left this one standing. The population is not small: about **14.7%** of
season-start roster minutes per `README.md`, and 106 of 539 rostered players in 2022-23.

It also lands on the contest directly rather than only on a metric. Minutes allocation is
zero-sum, so an availability rate given to a rookie comes straight out of his teammates'
minutes; scoring a top-5 pick at the pooled 0.4223 moves minutes *toward* the veterans on his
team, and scoring an undrafted call-up at the same 0.4223 moves them away. Both are wrong and
they are wrong in opposite directions, so the errors do not cancel at the team level.

### What to compare, and how

The estimator is already in the repo twice over. `rookie_share_priors` is an expanding-window,
point-in-time mean per draft bucket for exactly this population, with `DRAFT_BUCKETS` and
`UNDRAFTED_BUCKET` from `features/team_context.py`; `no_design_availability` is the same
construction one column over. So the arm is `no_design_availability` returning a **per-bucket
Series** rather than a scalar, pooling only seasons strictly before the target and only
seasons selection may read — the same rule it already follows, keyed on one more column.

Returning veterans are the second axis and the same table has them: gap-2 seasons realize
0.3141 and gap-3+ realize 0.2662 against the pooled 0.4223, so gap length is worth a second
key. That one is the genuinely new measurement, and it is small — 254 and 135 rows.

The bar is Gate A in `outputs/predictions/sim_season_gate_a.csv`, since these players are in
the tensor. Note what the pooled scalar already fixed once: scoring them at the head's
*intercept* put them at 58.4 simulated games against a realized 30.1, and the scalar is the
correction. This entry is the next term of the same series, not a new idea.

### It is the item the contest layer is waiting on — noted 2026-08-12

**`strategy_*.csv` is deliberately stale, and this entry is why.** Item 6 shipped a change to
the simulator's draw path and re-ran everything behind it except the sweep. The reason to
stop there rather than spend the hour is that this item would supersede the result, and the
two changes are not the same kind:

| | item 6's layout | this item |
|---|---|---|
| what moves | *where* a player's absences fall | *how many* games he plays |
| `gp` at the season unit | **exactly preserved** | changes for ~14.7% of roster minutes |
| Gate A | unmoved to four decimals | the entry's own bar |
| channel to the draft board | none — §7l, the layer *ranks* | **rankings**, directly |

A lottery top-5 pick moving from the pooled 0.4223 to a realized 0.8316 walks up the board,
and because minutes are zero-sum it pushes teammates who are not in this population down it.
That is the one channel §7l says the drafting layer has, so unlike the layout this can move
the sweep's *conclusions* rather than only its digits. **Run the sweep after this, not
before.** Item 8 carries the same caveat only if the block ports, which its own falsification
note calls the less likely outcome.

### What would settle it

Gate A's games-played CRPS and season-total MAE moving in the right direction, plus the
minutes displacement being checked at the team level rather than only at the player level —
the zero-sum argument above says a per-player improvement could still be a team-level wash,
and that is the thing worth knowing.

### What would falsify it

The buckets not being knowable at draft time in the production frame. `bio_draft_number` is
~15% NaN by construction, which is fine — that is the `undrafted` bucket and it is the
largest and best-estimated of the five — but the *veteran* gap length needs the panel, and a
player signed after the draft board is built has no row anywhere. If the graded rate can only
be applied to a minority of the population, the pooled scalar is doing more work than the
table suggests.

---

## 8. ✅ Cross the absence-composition block against the arm that actually ships — MEASURED 2026-08-12

**`docs/availability-window-plan.md` §14 is the live document**; what follows is the
motivation that opened the item, kept because the redundancy mechanism it named is what the
round then tested. **Nothing ships**, and the reason is §10e rather than a judgement. Five
things it settled, so they are not re-opened here:

- **The entry's central prediction was wrong, and the round is worth more for it.** The block's
  margins do **not** collapse under the two-component head: CRPS **−0.0575** [−0.1060, −0.0071]
  against `mixture`, which is **94%** of what it bought off `betabinom`. The **interaction** is
  what establishes that rather than the margin's size — **+0.0035** [−0.0079, +0.0149], 16.7×
  below the main effect — since `mixture` is 0.0112 CRPS *worse* than `betabinom` and so the
  easier of the two references. The two attacks are **complementary**, not two routes to one
  correction.
  `π` says *who* is at risk from age, absence volume and playoff workload; the four shares say
  *what kind* of absence he had; the second is not recoverable from the first. §14c, §14e.
- **On validation it is the first arm on this head to improve every regional metric at once** —
  CRPS, boundary (0.0109 → **0.0097**), body (0.0047 → **0.0021**), shoulder and both point
  masses, with no trade anywhere. It clears the round's bar, which is D1 with its halves
  swapped: a CRPS interval clear of zero with the boundary *held*, since `mixture` already
  spent the boundary gain. §14a, §14c.
- **And it still does not ship, exactly as §12 did not.** The CRPS margin shrinks **4.2×** on
  the rolling harness to −0.0136 [−0.0407, **+0.0132**] at 4 of 7 origins — the second failure
  to replicate, against the second likelihood, at almost the same factor as §12e's 5.8×. §14f.
- **The shrinkage is a population fact rather than a power fact, and that is what would have to
  be attacked next.** The rolling interval is **1.8× narrower** than the validation one on 3.3×
  the rows, so the second reading is the more precise of the two and the *effect* is what got
  smaller. The instrument is more scored seasons — and 2024-25 / 2025-26 are the test split.
  **No further arm on either axis should be built** until then. §14f, §14g.
- **The block belongs on `β` and not on `π`, which is this entry's own second falsifier.**
  Adding it to `π` costs **+0.0098** CRPS, makes the shoulder established worse (**+0.00158**
  [+0.00099, +0.00177]) and buys 5.8 training log-likelihood points and the best PIT KS on the
  table — four parameters that fit and do not predict. So the composition is a **mean-function
  fact, not a disruption-risk fact**, and `PI_COLS` stays at eight columns. §14d.

What remains open is only the port path, and it is conditional: `FrailtyGLM.pi_features`,
`StanAvailability`'s existing `pi_features` argument and `posteriors.py`'s persistence mean the
capability is built. If the block is ever confirmed, the port is `attach_absence_mix` on the
availability head's **own** design path — never `build_design`, which six other heads import.

---

**The motivation, as written when the item opened.**
**`docs/availability-window-plan.md` §12 measured the block against `betabinom`, and the head
that ships is `mixture`.** The block is worth **−0.0610** CRPS [−0.1148, −0.0047] and
**−0.00174** of `boundary_tail_error` [−0.00240, −0.00106] on the single-component reference.
Whether either survives on the two-component head is unmeasured, and it is the one thing
between the block and a port.

### Why this is worth measuring

The two are plausibly redundant and plausibly complementary, and the mechanism says which
half is which.

`mixture` closes the boundary by giving the disrupted season its own component with a weight
`π` that carries covariates — age, prior absence, playoff workload — so it already says *who*
is at risk. The absence-composition block says *why he missed last year*, which is a
different question about the same players and is not in `PI_COLS`. If the block's signal is
mostly "this player's absences were injuries rather than scratches, so he is fragile", the
mixture's `π` may already have it through `trailing_missed_lag1` and `n_spells_lag1`; if it
is mostly "a fifth of his missed games were games he was not rostered for", nothing in the
mixture can see it, and §11b says that fifth is real and fringe-driven.

The CRPS half is the less redundant one. The block improves the *mean* function
(train log-likelihood +18.9, validation R² +0.006, PIT KS 0.0667 → 0.0588) and `mixture` is a
tie on CRPS by construction — it was selected on calibration with a CRPS guard. So an arm
carrying both would be the first on this head to hold `mixture`'s boundary **and** a real
CRPS gain, if the two do not cancel.

### What to compare, and how

Three arms, on the same 883 validation rows, with window, season term and dispersion held at
the shipped arm as every round on this axis has done:

| arm | what it answers |
|---|---|
| `mixture` | the incumbent, already on the table at CRPS 9.8237 / boundary 0.0109 |
| `mixture + absence_mix` on `β` only | does the block help the mean under a two-component head |
| `mixture + absence_mix` on `β` **and** `PI_COLS` | does knowing *why* he missed say *who* gets a disrupted season |

The second and third are the real question and they are not the same arm. `π`'s covariate
list is deliberately short — §7's note is that nineteen more unpenalized parameters on 4,027
rows would be measuring the `l2` confound rather than the mechanism — so adding four columns
to `PI_COLS` is a decision to be made against that, not a free extension.

`availability_absence.arm_spec` already builds the feature list; the missing piece is a
`pi_features` argument on `MixtureFrailty` so the two covariate blocks can move
independently. Everything else — the ladder, the interaction bootstrap, the rolling harness —
runs unchanged.

### What would settle it

D1 on the third arm against `mixture`: `boundary_tail_error` no worse, and a CRPS interval
clear of zero on the good side. That would be the first arm on this head to buy both, and it
would make the block a port rather than a measurement.

### What would falsify it

The block's margins collapsing under the mixture, which is the likelier outcome and is why
this is an entry rather than a plan. `mixture` already spends eleven unpenalized parameters;
four more that duplicate `π`'s information would show up as a CRPS margin whose interval
reopens across zero. It would also be falsified in a more useful way if the block helped `β`
and did nothing on `π` — that would say the composition is a mean-function fact rather than a
disruption-risk fact, and would settle where it belongs if it is ever ported.

---

## 9. ✅ Re-read the availability head's boundary selection on the population it serves — MEASURED 2026-08-13

**Built as `make availability-absence`'s `population` round →
`availability_absence_population.csv`; the write-up is
`docs/availability-window-plan.md` §15a.** The answer is the one this entry's
"what would settle it" called the *unlikelier* outcome: on the draft pool the
`betabinom` − `mixture` boundary margin is **−0.00308 [−0.00733, −0.00210]**, so
the two likelihoods **swap places** on §7's own selection metric with an interval
clear of zero. What keeps `mixture` shipped is the other half of D1 — it wins CRPS
there at **+0.04280 [+0.00406, +0.08235]**, where pooled it only tied. So the head
is unchanged and its recorded *reason* is a pooled statement.

**The rolling harness agrees on every ordering and on none of the sizes**, which is
this entry's own caveat firing on the round it licensed. The draftable boundary
margin reads **−0.00049 [−0.00086, −0.00012]** against validation's −0.00308 — the
same sign, clear of zero, **6.3× smaller** — and the between-population level ratio
goes 1.84× to **1.09×**. The text above says "the sign flip is the robust half and
the 1.84× is not"; it was right, and a future round quoting −0.003 as the effect
size would be quoting 772 rows.

## 9x. The original entry, kept for the record

**`docs/availability-window-plan.md` §7 selected the two-component `mixture` on a
`boundary_tail_error` of 0.0201 → 0.0109, measured on all 883 validation rows. On the 772 of
them that are on a season-start roster the shipped head reads 0.01998**, and the
single-component reference was never scored there. If the mixture's calibration gain does not
transfer to the draft pool, the head that ships was chosen on a frame it is never applied to.

### Why this is worth measuring

`docs/preseason-plan.md` P2 measured the shipped head on both populations because P1 decision
5 requires it, and the reference row is the part that has nothing to do with the preseason:

| `mixture` | pooled (n = 883) | draftable (n = 772) |
|---|---|---|
| `boundary_tail_error` | 0.01085 | **0.01998** (1.84×) |
| the same, on P2's 10-origin rolling harness | 0.01615 | 0.01846 (**1.14×**) |
| P(GP < 10) predicted / observed | 0.0683 / 0.0815 — **under** by 0.0132 | 0.0582 / 0.0259 — **over** by 0.0323 |
| the same error, rolling | **under** by 0.0123 | **over** by 0.0179 |
| PIT KS | 0.0631 | 0.0928 |
| realized 50% coverage (nominal 0.5) | 0.598 | 0.639 |

**The sign flip is the robust half and the 1.84× is not.** The low-tail error reverses
direction between the populations at *both* readings; the boundary *level* ratio is 1.84× on
validation and 1.14× on the fitting half, because the pooled upper-boundary error is much
larger there. So the thing to chase is the sign, and any round that quotes 1.84× as a
constant is quoting a validation reading.

**The low-tail error changes sign between the two.** Pooled, the head under-predicts the dead
season; on the draft pool it over-predicts it by 2.25×, assigning 5.8% of draftable players a
sub-ten-game season where 2.6% realize one. The pooled figure is the average of two opposite
errors, which is §1's cumulative-threshold warning one level up — the cancellation happens
across populations rather than across bands.

That does not by itself impugn the selection: `betabinom` may be worse still on the draft
pool, in which case the mixture's margin survives and only its *level* was flattered. Nobody
has looked, because every round on this axis (§7, §12, §14) scored the pooled frame.

### What to compare, and how

Two fits, and the machinery exists. `availability_absence.crossed_ladder` already fits
`betabinom` and `mixture` at the shipped window; what it lacks is the population split that
`availability_preseason.score_population` implements — score the same fitted arms on
`on_season_start_roster > 0` as well as on everything, with `_bootstrap_arms` paired inside
the row as usual. Roughly two minutes of point MLE.

| row | what it answers |
|---|---|
| `betabinom`, draftable | is the single-component head *also* worse on the draft pool |
| `mixture`, draftable | already measured: 0.01998 |
| the margin between them, draftable | whether §7's 0.0201 → 0.0109 survives the restriction |

Worth taking the same reading on §14's `mixture + absence_mix` arm while the frames are in
memory, since that block's own margins were quoted pooled as well.

### What would settle it

The paired boundary margin `mixture − betabinom` on the draftable rows. If it is negative and
clear of zero, §7's selection stands and the only correction is that every boundary figure in
that document is quoted on a frame 12.6% larger than the draft pool. If it straddles zero or
flips sign, the mixture's calibration case was a pooled artifact and the ladder is worth
re-reading — which would be a shipped-head reversal, so the rolling harness (§4b, restricted
the same way) is part of the bar rather than a follow-up.

### What would falsify it

The two populations giving the same ordering, which is the likelier outcome: 87.4% of the
validation rows are draftable, so a statistic that moves 1.84× between them is being driven by
the 12.6% that are not, and a *difference* of two arms can be stable where each arm's level is
not. That would still be worth having on the record — it is the difference between "the
selection is fine" and "nobody checked".


---

## 10. ✅ Give the availability rolling harness a paired interval against the *primary* arm — BUILT AND MEASURED 2026-08-13

**`availability_preseason.rolling_confirmation` paired every arm against the reference and
never against the declared primary, so P2 could not say whether the arm that led its rolling
table was distinguishable from the one it declared.** `minutes_preseason.rolling_confirmation`
had that column (`crps_vs_primary`, plus `origins_won_vs_primary`) and P3 used it to promote
the centred delta on a fitting-half decision rather than a validation-driven swap. This was a
gap in an instrument, not an open question about the world, and it was cheap.

**It was added and the harness re-run the same day, and it settled the decision it was asked
to settle — in the direction the entry called a reversal.** `crps_vs_primary`,
`crps_vs_primary_lo/hi`, `origins_won_vs_primary` and the same four for the boundary now ride
on every row of `availability_preseason_rolling.csv` and on the validation ladder.

### What it found, on the 10-origin harness and the draftable population

| arm, against the declared primary `volume` | CRPS [95%] | origins | `boundary_tail_error` [95%] |
|---|---|---|---|
| **`p1_block`** | **−0.0977 [−0.1569, −0.0408]** | **8 / 10** | **−0.0014 [−0.0019, −0.0010]** |
| `volume_centered` | −0.0445 [−0.0919, **+0.0008**] | 7 / 10 | **+0.0016 [+0.0012, +0.0020]** |
| `participation` | −0.0093 [−0.0509, **+0.0358**] | 6 / 10 | −0.0007 [−0.0010, −0.0003] |
| `volume_pi` | +0.0025 [−0.0313, +0.0349] | 6 / 10 | −0.0011 [−0.0014, −0.0008] |

**P1 decision 4 is reversed, and the availability head ships P1's full ten-column block.**
Both of `p1_block`'s margins clear zero on the fitting half, which is P3's promotion rule —
an arm preferred after seeing validation is a validation-driven swap; preferred on the
rolling harness it is a decision the selection split never paid for. The entry said an
interval clear of zero "reverses P1 decision 4 on this head"; it is clear of zero.

**The same column also closed the round's other promotion question, negatively.**
`volume_centered` leads the validation ladder on CRPS and PIT KS, and against the primary on
the fitting half it is a **tie on CRPS** and **worse on the boundary with an interval clear
of zero** — giving up the half of the bar the block actually cleared. It was adopted for a
different reason (robustness to drafting before the preseason ends) and withdrawn the same
day when the premise was contradicted; see `dashboard/decisions.py`,
`preseason-availability-arm-survives-a-truncated-draft`.

### What it cost

One extra `paired_bootstrap` call per row over arrays the function already had in memory,
plus the re-run (~70 minutes of point-MLE mixture fits). The instrument is now symmetric with
the minutes harness, which is what a later preseason round on any head will want.

### What it did not settle

Nothing about the world was ever at stake here — but note what the reversal did *not* do to
the gate. `p1_block` beating `volume` on the fitting half says which arm to carry; it says
nothing about whether either clears P2's conjunction, which still fails on validation. See
item 9, which is the open question on this head.

---

## 11. ⚙️ Pool the no-design availability rate over rostered players only — MEASURED 2026-08-13, and **the recency half is falsified**

**Built as `availability_no_prior.run_recency` →
`availability_no_prior_recency.csv`; the write-up is
`docs/availability-window-plan.md` §15b.** The cross this entry specified was run
— estimator population × pool depth (all / 10 / 5 seasons) on the shipped
`tenure_draft` key — and it split:

- **The cancellation story is confirmed, by direct measurement.** At depth 5 the
  two estimators' validation biases are **+0.0642** (roster) and **−10.0796**
  (all): cutting the pool's depth removes the era drift from both and leaves the
  population error, which only the all-rows estimator has. Its full-depth
  **−0.3646** is those two opposite errors meeting, exactly as hypothesized.
- **The fix is falsified, by this entry's own falsifier.** CRPS degrades
  monotonically under the cut on both estimators and both splits; the roster arm's
  rolling win goes from **−0.3486 at 13/19 origins** to **+0.2753 at 7/19**.
  `graded_share` says why — **0.7926 → 0.5524** rolling — because a five-season
  pool holds 379 roster rows against 1,263 and `MIN_CELL` starves.

**So this is one change, not two, and the missing second change is an estimator
class rather than a shallower pool** — a shrunk cell estimator instead of a hard
`MIN_CELL` fallback, which is the same instrument P4(a) named for the *key* axis.
That is what this entry now points at; nothing ships and
`sim.availability.no_design_level` stays `tenure_draft` on the all-rows estimator.

## 11x. The original entry, kept for the record

**Opened 2026-08-14 by `docs/preseason-plan.md` P4(a), which found it while measuring
something else.** Not a preseason item — the defect predates the preseason work entirely and
would exist if none of it had happened.

### What is wrong

`sim/season.no_design_availability` is applied to players on an **October roster** and to
nobody else: the simulator's grid is the draft pool. The rate it hands them is pooled over
**every** no-design player-season before the target, which includes everyone who signed in
January. Over the covered window those two populations realize:

| | rows | realized `gp / team_games` | share with no preseason row |
|---|---|---|---|
| on a season-start roster | **1,336** | **0.5447** | 3.4% |
| not on one | **695** | **0.1571** | 38.6% |

So a third of the estimator's rows describe a population it is never applied to, and they
realize under a third of the rate. `no-design-availability-is-graded-by-tenure-and-draft-slot`
inherited this without noticing, because §8b varied the *key* and held the pool fixed.

### What P4 already measured

On the shipped `tenure_draft` arm, scored on the draftable rows:

| | roster-pooled vs all-pooled |
|---|---|
| rolling CRPS | **−0.3486 [−0.6860, −0.0183]**, **13 of 19** origins |
| rolling bias | **−5.3904 → +1.7935** games |
| validation CRPS | **+0.7354 [−0.0336, +1.5067]**, 0 of 2 origins |
| validation bias | −0.3646 → **+6.4056** games |

It wins more rolling origins than any preseason arm in that round and repairs a seven-game
bias swing, and **it fails the validation half of the same gate**, so it was recorded rather
than shipped.

### Why the validation half fails, which is the interesting part

The three rolling origins it loses are the **last three**, and two of those three *are* the
validation seasons. On 2022-23 and 2023-24 the all-rows estimator has a bias of −0.3646
games — nearly perfect — which is a cancellation between a population error (pooling in
January signings pulls the rate down) and an era drift (this population's realized rate has
risen), not accuracy. The roster estimator removes one of the two and the other is left
exposed at +6.41.

That is a hypothesis with a check attached: if it is right, the roster-pooled arm should
*also* want an era correction, and the two should be separable on the fitting half where 19
origins are available.

### What would settle it

- Cross the estimator population with a **recency window** on the pool (all seasons before
  the target against the last 5 or 10), on the rolling harness. If the roster estimator's
  recent-origin losses close under a recency cut, the cancellation story is right and the
  fix is two changes rather than one.
- Then re-read validation once, on whichever arm the fitting half selects. This is the P3
  promotion rule and it is what keeps the decision off the split it is scored against.

### What would falsify it

The roster estimator losing the recent origins *under* a recency window too. That would mean
the pooled estimator is better on recent seasons for a reason that is not the era drift, and
the honest reading would be that 1,336 draftable rows over 19 origins cannot resolve a
0.35-CRPS effect — the same power problem P2 hit one head up.

### What it is worth if it works

Bounded by §8b's own simulator readout, which is the template: that change moved season-total
dk_pts MAE by **3.18** and the per-team no-design minutes share error by **17.2%**. This one
is a smaller move on the same channel — no scored unit's `μ` changes, and the whole effect
arrives through the team's fixed minutes pot.

---

## 12. The availability layout fits a full season and is applied to a 91% sub-window

**Measured 2026-08-14**, in the course of reading Gate A after the P5 chain re-run. Scratch
figures from a transcript probe, not a `make` target — which is why this is an entry rather
than a result, and why nothing here is in `make docs-audit`.

### The defect, stated once

The availability head fits `gp_share` over the **whole regular season**, and
`allocate_spells` (`sim.availability.layout = tenure_merge`) then places a player's missed
games across that whole season. But `make simulate-season` only scores the **tournament
window** — DK's Round 4 closes before the NBA regular season does, so `realized_frame` sums
`dk_total` over games carrying a scoring slot and nothing else.

That window is **91.1%** (2022-23) and **90.7%** (2023-24) of the schedule, and it is not a
random 91%: it is the *front* of the season. Availability is measurably higher there, so a
layout that spreads a full-season absence budget evenly hands the scored window too many
absences.

### The evidence

Realized played-rate, **on the draft pool with every team game as the denominator** — P1
decision 5's population and the availability head's own denominator:

| season | inside the window | outside | difference |
|---|---|---|---|
| 2022-23 | **0.5865** | 0.5509 | **+3.6 pp** |
| 2023-24 | **0.5722** | 0.5309 | **+4.1 pp** |

⚠️ **The population and the denominator are both load-bearing, and the naive version of this
measurement gives the OPPOSITE sign.** Read over `in_appearance_window` rows instead, the
same comparison returns −9.2 pp and −9.0 pp — players look *more* available late. That
reading is an artifact: the appearance window is bounded by a player's first and last game,
so a season-ending injury in March removes him from the April denominator entirely. It is the
tenure-edge effect `availability-window-plan.md` §13 already sizes at **44.17%** of missed
games, firing on a new measurement. Anyone re-running this must use team games as the
denominator or they will conclude the reverse.

### What it is worth — real, and an order of magnitude short

The head fits the full-season rate (0.5834 / 0.5683) and the window realizes a higher one, so
a uniform layout under-plays the scored window by **0.27** (2022-23) and **0.33** (2023-24)
games per player. At the realized **22.03** / **22.06** dk_pts per played game that is
**−5.92** and **−7.29** dk_pts of season total.

Against Gate A's measured season-total bias of **−15.44** and **−66.64**, this explains
**38.4%** and **10.9%**. So it is a genuine contributor and it is not the explanation.

### Why this is worth measuring anyway

It is the only channel tested so far that points the right way. Two others were falsified in
the same sitting:

- **Per-game production does not change across the boundary.** Minutes per game read 23.027
  inside against 22.995 outside (2022-23) and 22.602 against 22.600 (2023-24); dk per game is
  flat in one season and slightly *higher outside* in the other. Whatever the late-season
  rest dynamic does to individual stars, it does not move the league-level per-game rate, so
  the rate heads are not where this bias lives.
- **Games played in aggregate does not track the bias.** 2021-22 draws **+1.938** games per
  player *more* than realized and still under-predicts season totals by **48.96** dk_pts. The
  two columns do not move together across the four Gate A seasons at all.

### What to compare, and how

1. **The cheap arm first, because it needs no refit.** Give `allocate_spells` a
   window-aware placement — weight the hazard of an absence landing in period `t` by the
   realized late-season gradient, estimated on the **fitting half** only. `gp` is unchanged
   by construction (a permutation of the played/missed vector leaves the season total
   exactly where it was, which is `availability-window-plan.md` §11's whole point), so this
   cannot move any head's likelihood and is a pure layout change.
2. **Score it at the unit that exposed it**: Gate A's `season_total_dk` bias row, per season,
   against the same four seasons. The bar to beat is the current −15.44 / −66.64, and the
   prediction is a **+5.9 / +7.3** dk_pts improvement — which is falsifiable, and small
   enough that it needs the paired form rather than two point estimates.
3. **Then the contest unit**, `make strategy-sweep`, where §7l is the standing precedent that
   a change arriving as *shape* rather than as *order* can be a measured null.

### What would settle it

The season-total bias moving by the predicted amount and no more. If it moves by
substantially more, the layout was carrying something else too and the decomposition above is
wrong. If it does not move at all, the arithmetic linking played games to dk_pts through a
league-average 22 dk/game multiplier is too crude — that conversion is applied to a per-player
mean and is the weakest step in the estimate.

### What would falsify it

The late-season availability gradient failing to replicate on the training seasons. Both
readings here are validation seasons, measured while investigating something else, and two
seasons agreeing is the weakest form of replication this project accepts.

### The larger question it does not answer

**What the other 60–90% of the bias is.** It varies 4× between two adjacent seasons (−15.44
against −66.64) on identical knobs, which a structural window effect cannot do. Gate A's own
framing — "a miss here is a **wiring fault** rather than a modelling one" — argues for
decomposing the level miss across the eleven component heads before assuming it is a
modelling problem. That decomposition is an artifact read rather than a refit, and it should
come before anything here is built.

---

## 13. ✅ DONE — port the cleared rate heads' preseason arms to Stan, and price them

**Opened and closed 2026-08-15 by session 6b** (`docs/preseason-plan.md`). Kept as the record
of what was asked and what the answer turned out to be, because on two counts the entry was
wrong about its own scope.

**What shipped.** Not three heads and not six: **all eleven were armed and ten ship**. The
entry was written against P1's six-head short list, and the round's largest finding is that the
short list was wrong — the screen's *sign* did not survive on any of the three heads it called
actively harmful, two of those clear the real bar outright, and a fourth excluded head,
`fg3a|fga`, is the third-largest result in the round. The eleventh, `fg3m|fg3a`, is the only
head anywhere in the preseason round measured as *worse* with a block, and it is rolled back by
`stan_components.PRESEASON_EXCLUDE`.

**What settled it, against the bar this entry set.** The ported arms beat their same-window
`__no_preseason` controls, and the increment **survives the posterior on ten of eleven at a
median retention of 0.985**, four of them growing — which was the falsification condition
named below and it did not fire. The contest pass ran as specified: the realized lift moves the
block's way in **9** of 10 season × tournament cells with the tenth at exactly **0.000000**, so
no cell moves against it.

**Where the entry mispredicted.** It expected a **board null** on §7l's precedent, since these
heads arrive as component rates rather than as the allocation mean. The board moved anyway —
rank correlation **0.960627 / 0.968677**, **93 / 98** of 192 drafted picks moving a full round —
and the season-total readout moved too, which was the outcome it *did* predict: the components
add **−2.27088** and **−3.84536** of season-total MAE on top of P5's three keys, recoverable by
differencing because the `base` capture is byte-identical across the two passes.

**The ratio this entry quoted has grown.** It recorded `reb` at **5.45×** and `fga` at
**1.07×**, both read on the declared primary arm. Read on the arm that actually ships they are
**7.25×** and **1.17×**, and `fg3a|fga` joins them at **3.77×** — three heads where the
preseason block is worth more than the entire fitted head is worth over arithmetic.

**The decision it did not take, still not taken.** *Whether heads should ship on the rolling
half alone.* Four now do — `stl`, `blk`, `fg3m|fg3a`'s stablemates and `ftm|fta` — passing the
rolling harness at 9–13 of 13 origins with intervals clear of zero while validation cannot
resolve them on 706 rows. That is the open decision P2 registered, now standing on four more
heads, and it is a decision about the *bar*. It remains the bar owner's.

---

## 14. `make stan` throws away the draws it just paid for, and `make posteriors` refits them

**Opened 2026-08-15**, from an owner question during session 6b's port: *don't all the shipped
models include the train window?* They do. `make stan` fits every head at its shipped variant
on the training rows, evaluates it, writes metrics and per-row predictions, and **discards the
coefficient draws**. `make posteriors --window train` then refits precisely that model to
persist them. The Makefile states the first half as a fact — "`make stan` ... THROWS THE
COEFFICIENT DRAWS AWAY, so simulating from the joint posterior has meant refitting" — without
treating it as something to fix.

### The two fits are the same fit, and that is checkable rather than assumed

`stan_utils.sample`'s `label` argument reaches `diagnostics()` only; it never touches
`model.sample()`. The draws are determined by data, chains, warmup, samples, seed,
`adapt_delta`, inits and metric, and both call sites read every one of those from the same
config keys. The design steps that could differ do not: `impute` takes its means and
`add_spline` its knots from **train** alone, so the transformed fitting frame is identical.

The manifest corroborates it. For every component head whose variant matches,
`posteriors`' `n_features` is the sweep's minus exactly the five preseason columns, on the
same **8,630** rows; `fg3m|fg3a`, which carries no block on either side, reads **14** on both.

### What it costs today

| fit window | `make posteriors` | already fitted by `make stan`? |
|---|---|---|
| `train` | **4.89 h** — components 2.09 h, composition 2.04 h | **yes, fully duplicated** |
| `train_val` | 8.31 h | no — different rows, never fitted |
| `full` | not built | no |

So ~**4.9 h per full re-run** at the train window, or **2.1 h** for a components-only pass —
which is the pass this project makes most often, since a head change is usually one family.
Only the **selected** variant is duplicated: the sweep fits three variants per head plus a
control (43 fits in 6b's run) and `posteriors` refits the 11 winners.

### What to build

1. **Hold each head's fitted model until selection resolves, then persist the winner**
   through the existing `_finish` / `save` path rather than rebuilding it. Memory is not the
   obstacle — 33 component models at 4,000 draws and K ≤ 19 is ~20 MB.
2. **Keep `make posteriors` as the entry point for the other two windows.** This retires the
   duplicated `train` fits, not the target: `train_val` and `full` are genuinely separate
   fits, and the design-recipe construction and round-trip verification have to live
   somewhere regardless.
3. **Fix the `impute` fragility in the same pass**, because merging the paths either fixes it
   or exposes it. `impute` creates a `<col>__miss` flag when **either** train or test has a
   NaN. The sweep passes the full validation frame (773 rows); `posteriors` passes
   `probe_rows(val)`, a thinned subset. A column with a NaN in `val` but not in the probe
   subset gives the two paths **different feature lists** for what is supposed to be the same
   head. It has not bitten — today's counts line up — but a feature list must not depend on
   the evaluation frame.

### Why this is worth doing beyond the wall clock

The two paths can **silently disagree**, and that is the same shape as the incident
`docs/preseason-plan.md` P2 records: an artifact carries no record of the code that wrote it,
so a head selected under one specification and persisted under another produces two internally
consistent artifacts that describe different models. Session 6b hit the live version — until
it was wired, `posteriors` built its rows through `component_rates.build_design` and would
have persisted eleven heads with **no preseason columns** while `stan_component_metrics.csv`
and the config both said the block was on. Nothing in the repo would have objected. One path
removes the class of error rather than the instance.

### What would settle it

The persisted draws from a `make stan` run reproducing a `make posteriors --window train`
artifact **exactly** — same coefficients, same `max_design_error` and `max_prediction_error`
under the existing round-trip check. That is a bit-for-bit assertion rather than a tolerance,
because the claim is that the two are the same fit.

### What would falsify it

Any head where the two paths do not reproduce. The `impute` case above is the known candidate
and would show up as a feature-count mismatch rather than a numerical one. A second candidate
is any head whose sweep applies a `metric` (`dense_e`) that `posteriors` does not, or vice
versa — the composition head is the one that passes a metric, so check it first.

### What it does not change

**Nothing about any measured result.** This is plumbing: the same model, fitted once instead
of twice. No gate, no ladder and no shipped figure moves, which is also what makes it safe to
do between rounds rather than during one.

---

## 15. A role × season-stage term in the availability layout — parked for nonstationarity

**Parked on arrival, 2026-08-16, and the reason is the regime, not the mechanism.** Drafted
as Part B of `docs/draw-time-calibration-plan.md` and moved here the same day: **the 2026
offseason draft-lottery reform penalizes the worst-performing teams, which targets the
tanking incentive directly** — and tanking is upstream of most of what this term would
calibrate (veteran shutdowns on eliminated teams, tank-driven rest, late-season rotation
churn on bad teams). That stacks on a second, smaller break already inside the data: the
65-game award rule arrived in 2023-24, one of the two validation seasons. So the dynamics
this term would fit shift substantially in 2026-27 **in a way no training or validation
season represents**, and a stage gradient calibrated to the old regime is plausibly worse
than the uniform placement that ships. Even a clean train → validation replication would
not license shipping it for the 2026-27 draft; that is why this is an entry and not a plan.

### What it would be

`sim.availability.layout = tenure_merge` lays tenure edge blocks by an empirical resample
and interior spells at **uniform random starts** (`docs/availability-window-plan.md`
§11/§13, the live doc). Uniform placement cannot carry within-season structure — star
maintenance rest clustering late, shutdowns, rotations tightening in a playoff race (healthy
scratches count as missed games in this target). The edges already carry the coarsest
version: a trailing block *is* a late-season shutdown, and the trailing share rises ~2.5×
fringe → star while the leading share falls ~4.2× (§13a). The open question is the
**interior** spells and whatever stage gradient the edges do not absorb.

The contest stake is concentrated, which is what made this worth drafting at all: the
elimination rounds sit at the end of the season (concluding 4/4, the final scoring periods),
Round 1 is a zero-consolation knockout, and the layout work already found a 9.1×
scoring-period-unit understatement once before it shipped. A star whose dead games cluster
late, laid uniformly, has his P(dead during the elimination periods) understated.

### What to measure, if it is ever un-parked

The measurement gate drafted as step B0, unchanged: extend
`make availability-exchangeability` with **stage-resolved** recovered_share cells —
dead-game rate by role × season-stage tercile, dead-period rate by role × scoring-period
index with the last-4-periods × star cell called out, interior spells separated from edges —
under the shipped layout, `gp` held at realized. **Read the two validation seasons
separately**: 2022-23 is peak load management and 2023-24 is the first 65-game-rule season,
so a pooled gap conflates the regimes. If uniform placement already recovers ≈1 in every
cell even on old-regime data, the question closes for free and the reform never matters.

If a gap exists, the build is placement-only first (empirical start-position weights by
role × stage, uniform nesting the shipped layout exactly, `gp` untouched by construction);
the spell-length-by-stage covariate is a fitted-head change and gets its own priced entry.

### What would un-park it

A season of data under the new lottery regime to check the old-regime gradient against —
realistically after 2026-27, i.e. after this project's target draft. Or B0 run as pure
knowledge showing the *shipped* layout already misses stage cells badly on **both**
validation seasons *in the same direction* despite their regime difference — stability
across that break is the one old-regime signal that would deserve any weight.

## 16. Rookies are not on the board at all — a rate head has no lag to score them from

**Every board this project has produced, forward or retrospective, contains zero true
rookies.** Surfaced 2026-08-21 while profiling the forward board rehearsal's unshared
units: the 2023-24 retrospective component design carries **0** players whose first
played season is the target — Wembanyama is not in the 387 scorable units the shipped
path ranked that season. The mechanism is structural, not a filter to relax:
`component_rates.MIN_PRIOR_MINUTES` demands 200 prior-season minutes because **every
rate feature is a lag**, and below it the own-rate features do not exist rather than
being noisy.

The consequence is asymmetric with availability: rookies DO absorb minutes (the
composition's expanding draft-bucket priors, `rookie_share_priors`) and DO get an
availability rate (the graded no-design level), so their teammates' minutes are drained
correctly — but their own dk_pts are never scored, so the recommender can never draft
one while the ADP field will. Round-1 rookies routinely carry top-100 ADP.

### What to compare

A rookie rate arm needs a feature set that exists before the opener with no NBA lag:
draft slot (already bucketed in `features/team_context.py`), and — the genuinely
promising one — **current-season preseason box scores**, which the preseason capture
already lands and which rookies play heavily in. The floor to clear is the current
behaviour priced in contest units: how much Round-1 advance probability does a strategy
lose by being structurally unable to roster any rookie, measured by giving the simulated
field (which drafts by ADP) its rookies while our seat cannot. If the floor is within
noise, the head is not worth building.

### What would settle it

The same gate every component head cleared: a validation-scored arm against the no-fit
floor, at the season-total unit (`make season-total`'s frame), on the rookie population
only. Preseason-rate → regular-rate transfer for rookies is the load-bearing unknown;
`docs/preseason-plan.md`'s P-series measured that transfer for veterans, never for
players with no prior season.
