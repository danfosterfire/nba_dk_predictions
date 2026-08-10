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
