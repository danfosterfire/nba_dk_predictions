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
blocks at both. The dilution is real and confined to the single-team *subset*, which is
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

## 5. A different frailty for the availability head — the boundary is welded to the variance

**Compare the beta-binomial's Beta frailty against likelihoods whose boundary behaviour is a
*separate* parameter from its dispersion.** Opened 2026-08-11, after the windowed role-graded
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