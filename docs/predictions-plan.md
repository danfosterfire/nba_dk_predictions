# Predictions Plan: From Marginal dk_pts to Draftable, Simulatable Rankings

This is a planning doc, not a measurement report — it records direction, not results. Update
it with real numbers as pieces get built, the way `availability-plan.md` was updated in place.

## Purpose and scope

`README.md` and `CLAUDE.md` already settle the per-player, per-game dk_pts problem: decompose
into shot classes, compose `season_total = gp × rate`, use a beta-binomial availability head.
This plan covers the layer **above** that, needed specifically to support draft strategy
(`docs/simulations-plan.md`): calibrated **joint** predictive distributions across players, not
just marginals, plus a market-aware (ADP-blended) ranking. Nothing here re-derives an
established fact from `CLAUDE.md`; it builds on top of it.

## Scope: regular season only — settled 2026-07-29

**Every fitting frame in this plan is regular season only**, and `preprocess.load_raw`
now enforces it by default (`season_type="regular"`). This was decided against measurement
rather than by assumption, and the decisive reason is the product, not the statistics:
the DK best-ball contest's final round ends **4/4** (`docs/dk_best_ball_rules.md`), before
the playoffs begin. There is no playoff game to predict.

The statistics agree and say something sharper. Playoff minutes are not a level shift that
an indicator variable could absorb — they are a **role interaction whose sign flips**.
Median playoff-to-regular MPG ratio over 5,762 player-seasons with ≥20 regular-season games
and a playoff appearance: **bench (<12 mpg) 0.505, rotation (12–24) 0.761, starter (24+)
1.054**, with 66% of starters playing *more*. Availability moves too — appearance rates of
0.711 / 0.905 / 0.958 across the same buckets, because rotations shorten (corrected
2026-07-30 from 0.664 / 0.838 / 0.922, a per-(player, team) denominator that double-counts a
traded player and records him as absent from the team he left). Pooling would fit
one coefficient to a −50% and a +5% effect simultaneously, and the bench population is
large enough to pull it toward compression, which is precisely the wrong direction for the
players a draft cares about.

**Playoff logs remain valuable as prior-season workload features** —
`prior_playoff_games` / `prior_playoff_minutes` on the availability plan's list. A deep run
is ~20 extra high-intensity games that regular-season logs cannot see. That is a feature of
season S-1, not a row to fit, and the distinction is the whole point:
`src/features/game_length.py` reads both season types deliberately because game length is a
property of a game rather than a target.

## The component targets — the output contract

This is the spec every head below is written against. **`dk_pts` is deterministic given these
twelve quantities and is never predicted directly.** Attempts and minutes enter only as
**exposure** and **trials**; they contribute nothing to DK scoring themselves.

| Component | Distribution | Exposure / trials |
|---|---|---|
| `min` (given availability) | successes / trials | trials = **game length**: 48, or 53/58/… in OT. *Not* a count — it is bounded, and the bound is a random variable. |
| `fga` | count — negative binomial | `min` |
| `fg3a` \| `fga` | successes / trials — the three-point **share of attempts** | `fga` |
| *`fg2a`* | **derived**, `fga − fg3a` — not a head | — |
| `fta` | count — negative binomial | `min` |
| `fg2m` | successes / trials | `fg2a` |
| `fg3m` | successes / trials | `fg3a` |
| `ftm` | successes / trials | `fta` |
| `reb` | count — negative binomial | `min` |
| `ast` | count — negative binomial | `min` |
| `stl` | count — negative binomial | `min` |
| `blk` | count — negative binomial | `min` |
| `tov` | count — negative binomial | `min` |

**The shot-attempt basis is `fga` × `fg3a | fga`** — seven negative-binomial counts and four
beta-binomial conversions, eleven heads. A three-point attempt *substitutes* for a two, so
total attempts are the count and the three-point mix is a share of them; `fg2a` becomes
derived, exactly as `pts` already is, and is still the trials for `fg2m | fg2a`. **The draw
order is therefore `fga → fg3a | fga → fg2a = fga − fg3a → makes`** — a conversion head's own
draw becomes a later head's trials. `season_terms._draw_components` materializes it; nothing
else may reorder it. The measurement that settled this is below; the full argument is in
`docs/shot-attempt-basis-plan.md`.

Availability sits **upstream** of all twelve — it gates whether the player-game exists at all
(`docs/availability-plan.md`), and `min` is drawn conditional on it.

Reassembly is exact: `pts = 2·fg2m + 3·fg3m + ftm`, then `preprocess.compute_dk_pts` with
`fg3m`, `reb`, `ast`, `stl`, `blk`, `tov`. Only **eight** components reach the scoring
function — `fg2m`, `fg3m`, `ftm`, `reb`, `ast`, `stl`, `blk`, `tov`. `min` and the attempt
counts matter solely through the exposure and trials they supply to those eight. When the
targets are built from the raw logs, `fg2a`/`fg2m` must be computed by subtraction
(`fga - fg3a`, `fgm - fg3m`) because the stored `FGA`/`FGM` *include* threes.

The double-double / triple-double bonus is a simultaneous threshold on
`pts`/`reb`/`ast`/`stl`/`blk`, so **the deliverable is a joint draw, not twelve marginals** —
`E[bonus] ≠ bonus(E[x])`. This is the same requirement the correlation work below imposes for
a different reason, and the two should be satisfied by one mechanism.

## What's already decided and built — not re-litigated here

- dk_pts decomposed into shot classes and counting components (`CLAUDE.md`, `make target-profile`).
- Availability is a beta-binomial GLM head; it beats ridge/GBM/league-age baseline on CRPS and
  is worth **210.3** dk_pts of season-total MAE (`src/models/availability.py`, `docs/availability-plan.md`).
- `season_total = gp × rate` composition (`src/models/season_total.py`).
- **The LSTM/Transformer trunk is deprioritized — decision, not just a finding.** See
  `CLAUDE.md`: prior-game-order features add ~+0.6pp R² over four season aggregates, which the
  sequence trunk mostly re-derives. This has come up multiple times in planning; treat it as
  settled and do not revisit without new evidence that per-game sequence structure matters more
  than measured.

## Lessons from the prior attempt (`~/Documents/nba_stats`)

Corrected summary, for the record:

- **The component-decomposition strategy was already tried and is worth reviving.**
  `01-scripts` fit Stan GLMs, PCA+GLMMTMB, brms, tidymodels, and XGBoost per shot/stat
  component (ast, blk, fg2a/fg2m, fg3a/fg3m, fta/ftm, min, played, reb, stl, tov) on player PCA
  × team PCA (teammates' prior season, aggregated) × opponent PCA (opposing players' prior
  season, aggregated), plus a "megamodel" joining them. This repo's `CLAUDE.md` independently
  re-derived the same decomposition as a hard fact — good confirmation the strategy is sound.
  Revive it on this repo's stronger feature set: persistence-weighted, reliability-shrunk,
  VIF-pruned columns, plus `teammate_usage_load` and the availability head, none of which
  existed last time.

  **`01-stan_glm_models/megamodel.stan` already implements the output contract above**, and
  read closely it is largely a template to carry forward rather than rewrite:
  `fg2a/fg3a/fta/reb/ast/blk/stl/tov ~ poisson(minutes .* exp(Xβ))` with minutes as a linear
  exposure, `fg2m/fg3m/ftm ~ binomial_logit(attempts, Xβ)`, and `played ~ bernoulli_logit`
  in the same block. Four specific things to change, each of which the current spec or this
  repo's measurements already answer:
  - **Minutes was `normal(μ, σ) T[0,48]`**, which forced `mutate(min = ifelse(min == 48, 47.9, min))`
    and `filter(min < 48)` — a hack against the boundary that also **discards every overtime
    game**. The successes/trials form with trials = actual game length is the fix, and it is
    why the contract above specifies minutes that way rather than as a count.
  - **No hierarchy was ever actually fitted.** `nplayers` and `playerid` are commented out in
    both the data block and the parameters. So the prior attempt's "GLMM" was a pooled GLM at
    the megamodel stage; partial pooling is genuinely new work, not a revival. (Which is the
    right expectation to hold anyway — see the dispersion note below.)
  - **Poisson only, no NB and no dispersion term.** `nbinom_from_playergame.stan` exists
    separately but was not wired into the megamodel.
  - **It was fitted on `sample_frac(0.01)`** — 1% of rows. That is the compute wall the
    collapse above removes, and it is the single biggest reason to expect a better fit this
    time rather than merely a tidier one.
- **The prior model solved a different problem than this repo does, and the `lag_5` features
  are where that shows.** `05-prepare_training_data.r` builds a trailing 5-game mean of every
  component (`min_lag_5`, `fg2a_lag_5`, …) and feeds them to every head. Those are
  *within-season* observations, so that model is an in-season nowcaster, not a one-shot
  preseason forecast — and `filter(!is.na(min_lag_5))` silently drops each player's first five
  games. Worth knowing before porting any of it: under this repo's prediction-time constraint
  those columns are unavailable at prediction time. The instinct behind them is sound and is
  now measured (see "Is there a hot streak?" below) — but it is worth ~1.03× on conversion and
  ~2.4× on minutes, so it should be one sequential minutes/availability layer rather than a
  lag block on all twelve heads.
- **The cross-season join was already correct there**, and independently matches this repo's
  construction: `prev_season_id = season_id - 1`, teammate aggregates built from the
  *current*-season roster described by *prior*-season stats with the player himself excluded
  (leave-one-out), opponent aggregates the same way, both minutes-weighted (`gp*min`). Splits
  were 2024 production / 2023 test / 2022 validation, with 2019 dropped as COVID. That is a
  point in favour of the construction in `CLAUDE.md`, arrived at twice independently.
- **Availability was not entirely unmodeled.** `training_played_from_boxscores_with_prevseason_*.rds`
  is a binomial games-played model, fit alongside the other components — so games-played
  variance had *some* representation.
- **The actual gap was the currently-known-injury overlay.** `11-formulate_injuries.R` joined a
  single hand-scraped return date and applied it as a deterministic step function — 0 before the
  date, full predicted value after, never updated, no distribution over the estimate. That is
  the Jaylen Green failure mode exactly: the return-date point estimate was wrong (predicted
  ~1 week, actual ~half season) and nothing in the model hedged against that.
- **This repo's beta-binomial head + report-calibration transfer function together supersede
  both pieces at once**: one calibrated distribution incorporating generic season-length
  overdispersion *and* (via the dated injury-report snapshot, `docs/availability-plan.md`)
  point-in-time, reason-conditioned signal — with the point-in-time discipline that plan
  establishes, which the old deterministic overlay had none of.
- **What was never built, either time:** any joint/correlation structure across players. And,
  separately, in the draft-simulation layer — see `docs/simulations-plan.md` — any use of ADP
  or portfolio-level exposure control.

## Chosen direction: Bayesian hierarchical GLMM, revived

Partial pooling suits this project's small-n problems (892 team-seasons; sparse-prior rookies).
But the deciding reason is **output currency**: the simulation layer needs many correlated
draws of a season to evaluate portfolio- and payout-shape-dependent objectives (P(cash),
P(advance)), not a single expected value. A point-estimate model, however accurate, cannot
supply that.

- **Component heads**: keep the established decomposition — Poisson/NB attempts × beta-shrunk
  conversion per shot class, the paired free-throw-trip structure, count heads for
  ast/reb/stl/blk/tov — hierarchical over player and team, season fixed effects per the
  "always absorb season" rule.
- **Availability**: consume the existing beta-binomial head's posterior as an upstream
  gate/multiplier. Do not re-model it here — it's built and validated.
- **GBM stays a secondary, comparison-only model** for components with strong nonlinear
  opponent × archetype interactions, evaluated under the same decision rule that already killed
  it once for availability: it has to beat the GLMM on a proper distributional metric (CRPS),
  not point accuracy, and it needs a distributional wrapper (quantile/NGBoost/conformal) to even
  compete on that axis.

## Fitting strategy: collapse the likelihood, simulate at the game level

The information set is a cross-season join, so **every feature except opponent / home / rest is
constant within a player-season.** That makes the game-level likelihood collapsible, which is
what turns 731,906 player-games into ~10,900 player-season rows per head.

**Both head types collapse exactly, including the successes/trials ones.** Write `η` for a
linear predictor constant within player-season `i`:

- *Counts.* `y_g ~ Poisson(m_g·e^η)` ⟹ `∏_g p(y_g) = Poisson(Y; M·e^η) × Multinomial(y; Y, m_g/M)`
  with `Y = Σy_g`, `M = Σm_g`. The multinomial factor is free of `η`.
- *Conversions.* `y_g ~ Binomial(n_g, θ)` ⟹
  `∏_g p(y_g) = Binomial(Y; N, θ) × [∏C(n_g,y_g)/C(N,Y)]` with `N = Σn_g`. The bracketed
  multivariate-hypergeometric factor is free of `θ`.

Both are algebraic identities, not approximations — the same expression collected differently,
exactly as the `pts` identity is. The season-collapsed posterior over `β` is **identical** to
the game-level one (verified numerically: max coefficient difference 7.0e-08 on the Poisson
side). The conversion heads are in fact the *cleanest* case, because they carry no exposure
model at all — just summed makes over summed attempts.

**What the collapse discards is precisely the second factor**: the within-season allocation of
the season total across games. Three things live only there, and only these three:

1. **Per-game covariates** — opponent, home/away, rest. Worth ~1–2% of variance (`CLAUDE.md`),
   and recoverable in a cheap second pass with `x_i'β̂` as an offset.
2. **Per-game dispersion.** A season total cannot distinguish a per-*game* random effect from a
   per-*season* one: a shared season multiplier passes its full relative overdispersion into the
   total, while iid per-game noise is diluted by ~1/G. So an NB fitted on season totals
   estimates **season-level heterogeneity**, and the game-level dispersion is a separate
   quantity requiring a game-level pass.
3. **Serial dependence.** Measured below.

None of this constrains the *simulator*. The collapse is a device for estimating `β`; the
generative model stays per-game — draw `min`, then counts with `min` as exposure, then makes
as binomial on those attempts. Fit collapsed, simulate expanded.

### Is there a hot streak? — measured

> Measured 2026-07-28 on `component_targets.parquet`, 592,796 player-games / 9,052
> player-seasons (played, `min ≥ 5`, ≥40 games). Pearson residuals against each player-season's
> own rate with `min` as exposure, so **minutes are already conditioned out** of the count rows.
> Null = game order shuffled *within* player-season, which preserves every marginal and the
> within-season demeaning bias. Reproduce with **`make serial-correlation`**
> (`src/eda/serial_correlation.py` → `outputs/eda/serial_correlation.csv`).

| component | lag-1 ρ | shuffled null | excess | 10-game block variance inflation |
|---|---|---|---|---|
| **`min`** | 0.278 | −0.016 | **+0.294** | **2.43×** |
| **`fg3a\|fga`** — shot **mix** | 0.085 | −0.016 | **+0.101** | **1.57×** |
| `fga` | 0.045 | −0.016 | +0.061 | 1.38× |
| `ast` | 0.024 | −0.016 | +0.039 | 1.22× |
| `fta` | 0.015 | −0.016 | +0.031 | 1.18× |
| `reb` | 0.014 | −0.015 | +0.030 | 1.17× |
| `blk` | 0.007 | −0.016 | +0.023 | 1.13× |
| `stl` | −0.003 | −0.016 | +0.013 | 1.08× |
| `tov` | −0.006 | −0.014 | +0.008 | 1.07× |
| **`ftm\|fta`** | −0.014 | −0.026 | +0.012 | 1.10× |
| **`fg2m\|fg2a`** | −0.015 | −0.018 | **+0.002** | **1.03×** |
| **`fg3m\|fg3a`** | −0.019 | −0.017 | **−0.002** | **1.01×** |

> ⚠️ **Refreshed 2026-08-03 for the shot-attempt basis, and the change is not cosmetic.**
> The `fg3a` / `fg2a` count rows (0.065 / +0.080 / 1.48× and 0.062 / +0.077 / 1.46×) are
> retired with the two-count basis. In their place `fga` reads +0.061 / 1.38× and the **shot
> mix** `fg3a|fga` reads **+0.101 / 1.57×** — the largest non-minutes dependence in the
> table, above the count it splits. So a "conversion head" is no longer necessarily a
> shooting head: shot *selection* drifts within a season, shooting *accuracy* does not, and
> the three shooting heads remain clean nulls at max |excess| 0.0122. Quoting one maximum
> over all four conversion heads would report the drift as a hot hand and hide that.

> ⚠️ **The whole table was refreshed 2026-07-31 against `serial_correlation.csv`, having
> been partially stale.** It previously read `min` excess **+0.297** against a null of
> **−0.017**, `fg3a` **0.061 / +0.077 / 1.46×** and `ftm|fta` **−0.008 / −0.023 / +0.015 /
> 1.11×**. `CLAUDE.md`'s copy of the same table had already been refreshed and this one had
> not, which is the exact failure mode `make docs-audit` exists to catch — and it caught it.
> Every cell moved by ≤ 0.006 and no sign, ordering or conclusion changes. The z column moved
> more, because it is a ratio of two small numbers.

**There is no shooting hot hand, and the answer splits exactly along the attempts/conversion
line the season-level persistence work already found.** Conversion percentages are serially
independent — `fg3m|fg3a` excess is −0.002 (z = −1.3) and `fg2m|fg2a` is +0.002 (z = 1.9),
both dead nulls on ~600k pairs. So **the specific thing worried about — the successes/trials
heads — is the one place collapsing costs nothing**, because constant-`θ`-within-season is
what the data actually looks like.

What *is* autocorrelated is the **exposure** side: minutes at 2.43× block-variance inflation,
and shot volume at ~1.46× *on top of* minutes (the count residuals already condition on actual
minutes, so this is volume persistence beyond playing time). Decay is slower than AR(1) —
minutes reads 0.278 / 0.212 / 0.170 / 0.113 at lags 1/2/3/5, where AR(1) would give 0.278 /
0.078 / 0.022. Removing a within-season linear trend drops lag-1 to 0.196 and speeds the decay,
so roughly **a third of it is slow role drift and two-thirds a shock with a 3–5 game
e-folding** — rotation changes, injury ramps, blowout clusters. This is role dynamics, not
shooting form.

**Consequences for the build:**

- Collapse the conversion heads with no reservations.
- Collapse the count heads for `β`, and treat the ~1.1–1.5× aggregate inflation as a **variance
  correction estimated once from game-level residuals**, not a reason to fit 731,906 rows.
- **Minutes is where the sequential model has to go.** It is the exposure driving all eleven
  other components, its inflation is 2.4×, and it is the one head where an iid-across-games
  simulation will visibly misstate the spread of a simulated season. It also already needs a
  sequential treatment for a separate reason — the availability spell process
  (`docs/availability-plan.md`) is a game-indexed Markov/semi-Markov chain, and minutes is its
  natural companion: availability decides *whether*, minutes decides *how much*. Build them as
  one sequential layer under the collapsed rate heads.

### Should the minutes head be fitted per-game? — ⏸️ deferred 2026-07-30, case recorded

The season-collapsed head is built (`make stan-minutes`) and the *cheap comparator* — its
posterior plus the separately measured game-level dispersion and block inflation, simulated
per game — is the current plan. This subsection records the case for replacing it with a
genuine per-game fit so the decision can be made on evidence once the full pipeline exists,
rather than re-argued from scratch. **Revisit after the simulator is standing**, against
held-out season-total CRPS and bonus-threshold calibration.

**First, a premise to retire: the 48-minute / overtime bound is NOT a reason to go per-game.**
`n` is already the summed *real* length of the games he played, so the estimand is a share of
actual floor time and the data satisfy `y_g ≤ n_g` by construction — that is what
`game_length.parquet` exists for. What is true is narrower: the season model contains no
per-game quantity, so a *simulator* must impose the bound itself. That is a two-line fix —
draw `y_g ~ BetaBinom(n_g, μ_i, ρ_game)` with the measured ρ_game = 0.0776. A simulator
concern, already solved, not an estimation gap.

**The real argument is zero-sum.** A team's minutes are exactly `5 × game_length` per
team-game — a hard constraint that binds *per game*, not per season — and it is the mechanism
behind "Explicit minutes/usage redistribution on teammate absence" below, which neither this
repo nor `nba_stats` has ever built. It cannot be expressed at season level because *who is
out* varies game to game and the redistribution is conditional on exactly that.

**What a per-game fit buys, ranked by measured size:**

| gain | size | comment |
|---|---|---|
| **serial dependence** | **2.43×** 10-game block variance inflation | the largest, and invisible at season level |
| **heterogeneous per-game dispersion** | one ρ for everyone today | a 34-mpg starter's minutes are far steadier than a fringe player's |
| **teammate-absence redistribution** | unmeasured | a new capability, not a variance correction |
| per-game covariates | smallest | home/away is 0.03% of dk_pts variance |

Two caveats on that table, both of which shrink it:

- **Serial dependence is variance, not forecastability.** `make serial-correlation` attributes
  roughly a third to slow role drift and two-thirds to shocks with a 3–5 game e-folding —
  rotation churn and injury ramps. A season-ahead forecast cannot predict those; a per-game
  model only gets their *spread* right.
- **The per-game covariates that actually move minutes are unknowable at prediction time.**
  Only opponent, home/away and rest are schedule-derived. Blowout status, foul trouble and
  in-game injury are the real drivers and none are forecastable.

So the honest framing: **per-game buys calibration of the season-total distribution, not
accuracy of its mean.** For a threshold-and-order-statistic product that is the thing that
pays — the double-double bonus is a per-game threshold and minutes drive it — but the case
must be made on those terms, not as "a better minutes model". The season head already scores
R² 0.8835 against a no-fit floor of 0.8536, so the headroom being competed for is small.

**Costs, with real numbers.** 731,863 played regular-season player-games against the season
head's 8,306 rows — **88× the data**. The season spline fit took 955 s (linear: 175 s) —
measured while another head sampled alongside it, so an upper bound — and so a
plain per-game beta-binomial GLM with no latent state is order **6–16 h**: tolerable, and it
delivers per-game covariates and heterogeneous dispersion. Adding a **latent AR state per
player-season** introduces ~500,000 latent variables, which HMC handles badly without
marginalization — days per fit and divergences that cannot be tuned away. If serial dependence
is wanted, the tractable routes are a marginalized state-space (latent integrated out) or an
explicit lagged-observation term, **not** a free latent per game.

**Two things that will bite:**

- **Endogeneity.** Minutes and availability are jointly determined — a 12-minute night may be
  a player nursing something that later becomes an absence. Conditioning minutes on
  availability inherits selection that is arguably *worse* at game level, where it does not
  average out.
- **Compounding at prediction time.** Teammate availability is *simulated*, not observed. A
  redistribution model conditioned on who is out amplifies availability error into minutes
  error instead of averaging over it.

### The team-game composition alternative — ✅ SHIPPED, full window 2026-08-04

> **See `docs/minutes-composition-plan.md` for the full result.** `make stan-composition`,
> now part of `make stan`.
> The costing below stands; the warning box that follows it is **resolved**, and by a form
> neither option in it anticipated.
>
> The team-game minutes are allocated by decomposing the multinomial into **sequential
> binomial trials** with the per-player cap carried in the **trials** (`m_k = min(U, R_k)`,
> the remaining capacity) rather than as a truncation. That gets **both** constraints:
> the individual cap by construction, the team total by the deterministic last step.
> Fitted on all 30 seasons and scored on validation (2022-23/2023-24), the selected variant
> scores **4.4945** minutes of CRPS against the no-fit floor's 4.6776 and the independent
> per-player draw's **4.7842** —
> so it beats the incumbent on the incumbent's own marginal metric, which this plan
> expected to be a wash, *and* the independent draw's mean team-sum error is **33.89**
> minutes per team-game against the composition's exact zero.
>
> Two results worth carrying back here. **The pure decomposition fails**: the plain
> binomial arm scores 4.9388 with PIT KS 0.1919, below the floor — the measured 4.65×
> game-level dispersion is not optional, exactly as the NB-vs-Poisson result on the count
> heads. And **the offset is the floor**, so proportional redistribution comes for free
> and `β` fits deviations from it — which makes "who absorbs the minutes when a starter
> sits" a fitted quantity, the thing the redistribution section below wants.
>
> The dispersion is **graded by prior-share quartile** (fitted 0.1768 fringe to 0.0855
> star, a 2.07× spread against one shared 0.1211), which cuts mean |variance ratio − 1|
> by 35% and lands the star tier at **0.81**. Still open: the fringe tier reads **1.05**
> and q2 0.81 — grading a *step* dispersion does not map one-to-one onto *marginal*
> variance, because a low-share player breaks his stick last and inherits the remainder
> variation ahead of him. **Gate E was taken at the full window on 2026-08-04 and the head
> ships in `make stan`**; the figures here are from the 2026-08-08 validation refit, and
> the retired test column is preserved in `docs/minutes-composition-plan.md`. This is
> **iid across games** and therefore does *not* address the 2.43× block inflation — the
> residual serial process below is unaffected by it.

Rather than 731,863 player-game rows, model the **team-game composition directly**: a
Dirichlet-multinomial over the `5 × game_length` minutes among the players who dressed. That
is **71,092 team-game rows** (35,546 regular-season games × 2) at ~10.3 players each, so the
same information in a tenth as many likelihood terms, and — the point — **zero-sum holds by
construction rather than as a penalty**. It is the natural formulation for the thing the
redistribution section actually wants, and it makes "who absorbs the minutes when a starter
sits" a fitted parameter instead of a hand-set rule.

> ~~⚠️ **It trades one exact constraint for the other, and that is the catch.**~~ —
> **resolved 2026-07-31, and the resolution is the last sentence taken literally.** The
> concern was real as stated: a Dirichlet-multinomial enforces the team total exactly but
> does **not** bound any individual at `game_length`, while the per-player beta-binomial is
> the mirror image, so "**neither form gets both**" was true of *those two* forms. What was
> also true and easy to skip past is the escape hatch already written here — "getting both
> needs a constrained allocation step". The sequential binomial decomposition **is** that
> step, and the constraint costs nothing: setting each step's trials to the remaining
> capacity `min(U, R)` bounds the player, and the deterministic last step closes the total.
> No truncation CDF in the gradient, no rejection sampling. Both constraints are asserted on
> every simulated draw rather than checked in posterior predictive.

### Why not just an explicit lagged term? — measured 2026-07-30, it buys **46%** of the excess

The obvious cheap route to serial dependence is an observation-driven AR: put the previous
game's minutes in as a covariate,

```
logit(theta_g) = x'beta + phi * f(y_{g-1})
```

**The cost instinct is right** — this adds one column and *zero* latent variables, so it stays
an ordinary GLM and never approaches the ~500,000-latent state-space wall. But a single lagged
term is measurably the wrong process, and the gap is large enough to decide the question.

**The decay is nowhere near geometric.** `outputs/eda/serial_correlation.csv`, 583,744 pairs:

| lag | observed | AR(1) at φ = 0.2785 | ratio |
|---|---|---|---|
| 1 | 0.2785 | 0.2785 | 1.0× |
| 2 | 0.2116 | 0.0775 | **2.7×** |
| 3 | 0.1698 | 0.0216 | **7.9×** |
| 5 | 0.1131 | 0.0017 | 68× |
| 10 | 0.0339 | ~0 | — |

An AR(1) matched to the observed lag-1 implies a 10-game block variance inflation of
**1.665×** against the measured **2.432×** — it reproduces **46% of the excess**, and the
excess is the entire quantity of interest. Full per-game compute for under half the
calibration gain.

**The missing half is structural, not a tuning failure.** The `min_detrended` row shows that
removing a within-season linear trend drops block inflation to **1.706×**, so slow role drift
alone accounts for **~51% of the excess**. Note that is far larger than the lag-1 statistic
implies — detrending moves lag-1 only 0.278 → 0.196, about 30% — because **drift is
long-memory and aggregates much more strongly than it correlates one step ahead.** That gap is
exactly why `block_inflation` is the decision-relevant column and lag-1 is not.

And drift is what an observation-driven term **cannot** represent: it is a moving *level*, not
a dependence on the last observation. Capturing it that way needs many lags or a moving
average of the recent past — at which point a state-space model has been rebuilt with worse
conditioning.

**Three further objections, in descending order of how much they matter:**

- **At prediction time the lag carries no information at all.** Every player starts at game 1
  with no observed lag, so all ~82 are self-generated. Unlike `nba_stats`'s `lag_5` nowcaster
  — where the lag was *observed* — this is purely a variance-generating device, and it
  therefore competes against imposing the measured 2.43× directly, which costs nothing.
- **Dynamic-panel attenuation, pointing the wrong way.** `y_{g-1}` is a noisy proxy for the
  player's own level, so `phi` absorbs between-player variation belonging to `beta`. At
  simulation time game 1 is seeded from the now-attenuated `beta` and every later game
  inherits it — which would surface as the minutes head's **−33 to −41 minute held-out bias
  getting worse**, and that is already its weakest number.
- **Absences break the lag's definition.** If a player misses games 10–25, is `y_{g-1}` for
  game 26 his last appearance sixteen games earlier, or a zero? Different models, neither
  obviously right, and the choice couples minutes to availability *structurally* rather than
  only statistically.

### ⭐ The recommended third option: fit the serial structure on residuals, level held fixed

Keep the season-collapsed `beta` and fit **only** the serial process on game-level residuals,
with the level pinned at the season posterior. This is the "variance correction estimated once
from game-level residuals" the count heads already get, applied to minutes with a real process
instead of a single scalar. It is the best-value option on the table:

- **Cheap** — residuals, not a 731,863-row joint fit.
- **Immune to the attenuation problem**, because `beta` is never re-estimated.
- **Free to use two components**, which is what the data actually show: a slow within-season
  level drift *plus* a short shock with a 3–5 game e-folding. One geometric decay fits neither,
  and the 46% figure above is the price of pretending otherwise.

It does **not** deliver zero-sum or teammate redistribution — only the composition model does
that. So the realistic sequencing is: residual serial process first (cheap, fixes the
calibration gap), redistribution heuristic second, composition model only if the posterior
predictive says the heuristic is not enough.

> **A confound that sits underneath all three options.** Neither the availability nor the
> minutes head carries any season term, and a first look (`docs/availability-plan.md`, "Load
> management") finds a large role-graded era effect: heavy-minute players lost **−0.101 of
> games-played share** from 2004–2010 to 2023–2025, roughly 8 games of an 82-game season,
> against −0.037 for fringe players — while minutes *given role* stayed flat. Fixing serial
> dependence in a model whose level is drifting solves the smaller problem first. Settle the
> era question before spending days on per-game compute.

**Decision: none of the three is started, deliberately.** The cheap comparator is already
fitting, and this project's record on ceilings says it may capture most of the gain. The
reassessment should be a measurement — per-game, composition, or residual-serial *versus* the
season posterior driven by the measured ρ_game (4.65× binomial) and 2.43× block inflation —
scored on held-out season-total CRPS and on whether simulated bonus rates match realized ones.
The comparator costs an afternoon; the per-game fit costs days.

### Season effects: the league moves, and no head should carry a term for it — settled 2026-07-31

**Reproduce with `make season-effects`** (`src/eda/season_effects.py`) →
`outputs/eda/season_effects_{league_rates,summary,carry_forward_bias}.csv`. Every figure
below comes from those three artifacts; do not re-derive them by hand.

Every head in this project — availability, minutes, all eleven components — is fitted on 30
pooled seasons with **no season term**. That sits against the repo's own standing rule
("ALWAYS absorb season when regressing on 30 pooled seasons"), and the reason it was never
applied is the prediction-time constraint: **a season fixed effect for season S does not
exist when forecasting S.** So the real question is which of two *usable* forms each
quantity needs.

**They are complementary, not alternatives, and that is an identity rather than a finding.**
Subtracting a linear trend shifts the **mean** of the year-over-year changes and leaves
their **variance** exactly unchanged, because `diff(a + b·x)` is the constant `b`. Therefore:

- a **year-on-year trend fixed effect** removes `yoy_mean_pct` — the drift that biases a
  carry-forward predictor in the *same direction every single year*;
- a **year-level random effect** is the only thing that touches `yoy_sd_pct`, the
  irreducible year-to-year spread.

Neither substitutes for the other. Pinned by
`tests/test_season_effects.py::test_a_trend_removes_the_yoy_MEAN_and_leaves_the_yoy_SD_untouched`.

| quantity | max/min | trend %/season | trend R² | yoy sd % | verdict |
|---|---|---|---|---|---|
| **`fg3a_pct`** — the shot **mix** | **2.64×** | **+3.58** | **0.93** | 6.19 | **trend + year effect** |
| `fta` | 1.21× | −0.55 | 0.63 | 4.36 | year effect only |
| `blk` | 1.14× | −0.14 | 0.12 | 3.70 | year effect only |
| `ast` | 1.30× | +0.73 | 0.64 | 3.08 | year effect only |
| `stl` | 1.18× | −0.08 | **0.03** | 3.03 | year effect only |
| `tov` | 1.16× | −0.33 | 0.59 | 2.89 | year effect only |
| `fga` | 1.14× | +0.47 | 0.84 | 1.52 | year effect only |
| `fg3m_pct` | 1.08× | +0.10 | 0.27 | 2.06 | year effect only |
| `reb` | 1.11× | +0.25 | 0.59 | 1.48 | year effect only |
| `fg2m_pct` | 1.20× | +0.61 | 0.84 | 1.46 | year effect only |
| `ftm_pct` | 1.08× | +0.19 | 0.77 | 1.06 | year effect only |
| **`minutes_share`** | 1.09× | −0.25 | 0.81 | **0.92** | year effect only |
| `gp_share` [<12 mpg] | 1.48× | −0.71 | 0.48 | **9.04** | year effect only |
| `gp_share` [12–24] | 1.30× | −0.54 | 0.64 | 4.21 | year effect only |
| `gp_share` [24–30] | 1.25× | −0.39 | 0.45 | 3.85 | year effect only |
| `gp_share` [all] | 1.24× | −0.51 | 0.71 | 2.86 | year effect only |
| `gp_share` [30+ mpg] | 1.21× | −0.49 | 0.74 | 2.84 | year effect only |

**The three-point MIX is the only quantity where a trend is worth extrapolating** —
`fg3a_pct`, the three-point share of attempts, at R² 0.93 and **+3.58%/season** — and note
its worst single year is **−24.7%**, the 1997-98 three-point line being moved back
after three shortened seasons.

> ⚠️ **Refreshed 2026-08-04 for the shot-attempt basis, and the split is informative.** The
> retired `fg3a` *count* row read 2.97× / **+4.07** / 0.93 / 6.53 and the retired `fg2a` row
> 1.32× / −0.87 / 0.86 / 2.42. Modelling total attempts and the mix separates them: `fga`
> drifts at only **+0.47%/season** over a 1.14× band, so essentially all of the three-point
> climb is *which* shots are taken rather than how many. The trend R² is unchanged at 0.93,
> so the "only extrapolable trend" verdict survives the basis change intact. **Everything else is shock**, `fta` and `stl` most starkly:
`stl` has a trend R² of **0.03**, i.e. essentially no drift at all and 3% of pure
year-to-year noise.

**`fta` is the case that prompted this**, and it behaves exactly as refereeing would
predict — a 1.21× band with 4.4% yoy sd and swings past ±5% in 9 of 29 transitions:
**+7.6% in 2004-05** (hand-checking crackdown), −7.8% 2011-12, −6.0% 2017-18, +7.3%
2022-23, −7.5% 2023-24, **+8.6% in 2025-26**. There is no trend to extrapolate.

**What ignoring it costs — the no-fit floor's bias on the validation seasons.**
Carry-forward lags any league move by exactly one season, so its bias *is* the season effect
measured on the scale the heads are scored on. No fitted head corrects it, because none has
a season term. Each cell is shown beside that season's own league move, because the lag
predicts they should carry **opposite** signs:

| component | league move 2022-23 | bias 2022-23 | league move 2023-24 | bias 2023-24 | both |
|---|---|---|---|---|---|
| **`fta`** | +7.3% | **−4.4%** | −7.5% | **+10.7%** | +2.9% |
| `ast` | +2.5% | −2.2% | +5.6% | **−6.2%** | **−4.2%** |
| `blk` | −1.5% | +4.3% | **+10.6%** | **−5.4%** | −0.8% |
| `stl` | −4.6% | +5.1% | +2.7% | −3.5% | +0.8% |
| `tov` | +2.7% | −0.6% | −3.8% | +5.6% | +2.5% |
| `reb` | −2.5% | +3.8% | +0.4% | −0.0% | +1.9% |
| `fga` | −0.0% | +1.4% | +0.9% | +0.9% | +1.1% |

**The lag is a measured relationship, not an assertion about two components.** The bias
opposes its season's league move in **13 of 14** cells and the two correlate at **−0.944**
(`opposes_league_move` / `league_yoy_pct` in the artifact; `fga` 2023-24 is the single
exception, on a league move of under 1%). `fta` is the clearest case in both directions:
**−4.4%** against the league's +7.3% rise into 2022-23, then **+10.7%** against its −7.5%
fall into 2023-24.

**Read a per-season cell, never the `both` column alone.** Averaging two seasons whose
league moves point in opposite directions reports a *lag* as a *level* — `fta` pools to a
mild +2.9% and `blk` to −0.8%, both hiding swings of 15 points. The `both` column is
retained because a head that carries no season term inherits whatever the pooled bias
happens to be, but it describes the seasons scored, not the component.

> ⚠️ **This table was a TEST evaluation until 2026-08-08** and read, on 2024-25/2025-26
> over 791 rows: **`fta`** −3.1% / **−10.7%** / **−7.0%**, `stl` −10.0% / +0.7% / −4.6%,
> `fg3a` −7.0% / −1.3% / −4.2%, `tov` −4.6% / −1.8% / −3.2%, `ast` −1.1% / −4.8% / −2.9%,
> `blk` +6.5% / +6.0% / **+6.2%**. The prose read: "`fta`'s −10.7% in 2025-26 is the +8.6%
> league jump arriving one year late. `blk` is biased +6% in *both* seasons, which is drift,
> not noise."
>
> **The mechanism survives and both headline readings do not.** `fta` is not carrying a
> −7.0% systematic bias — that was one season's −10.7% averaged with a −3.1%, and on
> validation the same component reads +2.9%. And **`blk` reverses sign**: +4.3% then −5.4%,
> against "+6% in both". The "drift, not noise" reading was the weaker claim all along, and
> the artifact said so — `blk`'s trend R² is **0.118** on a −0.14%/season slope, which is
> the profile of a component with no drift to speak of. Two adjacent seasons moving the same
> way is what noise looks like half the time.
>
> **What the move bought is the check that replaces them.** The retired table's cells oppose
> their league move in only **10 of 14** (r = −0.865) against validation's 13 of 14, so the
> lag is *cleaner* on the split that is allowed to decide. A component-level headline was
> never the finding; the relationship is.

> **Why this outranks the shared-β correlation the Stan work was built for.** A league shift
> is **perfectly correlated across every player**, so it does not diversify away: a −7% error
> on free throws is −7% on a whole roster's free-throw points. The shared-β parameter
> uncertainty measured in `stan_availability.board_correlation` is worth **+0.5%** on a
> 15-man roster. Season effects are the larger non-diversifiable risk by an order of
> magnitude, and they are currently modelled as exactly zero.
>
> ✅ **Now measured, and the "order of magnitude" was an under-statement by a further
> order** — `make season-terms` (`season_term_roster_spread.csv`): a year effect widens a
> 15-man roster's season-total dk_pts spread by **+10.4%** against shared-β's **+0.2%**,
> i.e. ~**50×**, and by **+254%** across the whole 773-player validation board against
> **+6.4%**. See the verdict below, which also finds this is the *only* thing a season term
> is worth: on point accuracy the ceiling is **≤5% of MAE**.
>
> ⚠️ **Superseded 2026-08-07**: the recorded **+11.6%** / **+278%** and the ~**95×** ratio
> were measured on the 791-player *test* board. The comparison is unchanged in kind — a
> league shift still does not diversify and shared-β still does — but quote the validation
> figures, and note the shared-β side (+0.2% / +6.4%) is itself measured on a different
> board again, so the ratio is indicative rather than exact. It moved again on 2026-08-11,
> when the availability head took a 2012-13 window and its board figures roughly doubled to
> +0.5% / +12.3%; the conclusion is unchanged, since a year effect is still worth ~20× the
> shared-β term on a roster, but do not read the ratio to a significant figure.

**One piece is genuinely knowable at prediction time and should not be lumped in with the
rest.** Rule changes and points of emphasis are announced in the summer, before opening
night — the 2021-22 non-basketball-moves emphasis was public in advance. A manual
league-level override is therefore legitimate under the point-in-time discipline, unlike
anything drawn from within the season.

### ✅ VERDICT — the ablation ran 2026-07-31, and **no head ships a season term**

**Reproduce with `make season-terms`** (`src/models/season_terms.py`) →
`outputs/predictions/season_term_{metrics,season_total,roster_spread,bonus,sigma_vs_league,diagnostics}.csv`.
**54 fits, 0 divergences, max R̂ 1.0105, 79.3 min** (2026-08-07, validation only). Two fits
sit marginally over the 1.01 R̂ bar — `tov/base` at 1.0103 and `min/base` at 1.0105 — and
**both are `base` arms**, so the season terms are not what strains the sampler. Four arms
per head — `base`, `trend`, `year`, `trend_year` — on top of each head's already-selected
specification, plus `trend_x_role` and `trend_x_role_year` for availability. Selected on the
validation split (2022-23/2023-24), every arm quoted against its no-fit floor. Every fit uses
`metric="dense_e"`: on the `blk` spline base that is **13.4 s against 236.6 s** with
treedepth saturation **0 against 35**, which is what made a full Bayesian ablation
affordable at all.

> ⚠️ **This was a 108-fit, both-splits run until 2026-08-07** (max R̂ 1.0142, 155.9 min, five
> fits over the R̂ bar, four of them `base` arms). `src/models/held_out.py` now raises on the
> test seasons, so the test side is not fitted at all — which is where **half the sampler
> time** went. The end-of-project reading is `src/final_evaluation.py`'s, taken once.
>
> ⭐ **Every per-head validation number reproduced to five decimal places**, and all 13
> selected arms are unchanged. That is a **determinism check, not a replication**: the
> validation fits were always validation fits, at the same seed and the same 500/500 budget,
> so removing the test fits could not have moved them and it did not. It confirms the
> conversion changed nothing it should not have. The *replication* evidence for this
> ablation is still the July-versus-August comparison below, where the arms did shuffle.

> ⚠️ The ablation runs at the **selection** sampler budget (500/500), so the `base` arm is
> not directly comparable to `stan_component_metrics.csv`, which now runs full-length chains
> everywhere. Differences between the two tables are sampler noise. Within this table every
> arm shares one budget, which is what the contrast needs.

#### 0. Read this before quoting the selected arms

**11 of 13 heads select a season term** — `trend` ×5, `year` ×3, `trend_year` ×3, `base`
only for `reb` and `stl`. On the July head list almost every head selected `base`. That
looks like a reversal and is not, because **10 of the 13 margins are under 1% of the
selection metric**:

| head | selected | margin over `base` | | head | selected | margin over `base` |
|---|---|---|---|---|---|---|
| `reb` | `base` | 0.0000% | | `min` | `year` | 0.198% |
| `stl` | `base` | 0.0000% | | `fga` | `year` | 0.264% |
| `fg3m\|fg3a` | `trend_year` | 0.008% | | `blk` | `trend` | 0.352% |
| `fg3a\|fga` | `year` | 0.046% | | `fta` | `trend` | 0.785% |
| `ftm\|fta` | `trend_year` | 0.087% | | `tov` | `trend` | 1.302% |
| `gp` | `trend` | 0.107% | | `fg2m\|fg2a` | `trend` | 1.883% |
| | | | | `ast` | `trend_year` | 2.209% |

An ablation whose winner flips across a head-list change that does not touch most of its
heads is **measuring noise**, and that is a stronger statement of "the terms are worth
nothing" than the original run could make — it now has a replication test behind it, and it
failed. Only `ast` (2.21%) and `fg2m|fg2a` (1.88%) move on a margin worth a second look, and
neither is a shot-attempt head, so neither is explained by the basis change.

**The margin is read on each head's own selection metric** — `val_nll` for the conversion
heads, `val_crps` for everything else — because that is what `_finalize` selects on. Worth
stating because `src/docs_audit.py` got it wrong once: it inferred the metric from whether
`val_nll` was populated, and the 2026-08-07 re-run began writing `val_nll` for the count
heads, which silently switched them onto a column they are not selected on. The symptom was
`stl` reporting a 0.008% margin while its selected arm was `base` — an arm that is the
minimum by construction can only have a margin of zero.

#### 1. The trend is refuted, and most sharply on the one quantity that predicted it

> ⚠️ **This whole section describes the RETIRED `fg3a` count head and is preserved as the
> record of that basis.** The shot-attempt basis replaced it with `fga` × `fg3a | fga` on
> 2026-08-03, so `fg3a` is no longer fitted and none of the figures below have a row in the
> current artifact — they are presence-checked, not value-checked. The section stays because
> **this is the sharpest evidence against a trend the project ever produced**, and its
> successor cannot restate it: `fg3a | fga` selects `year` on a **0.046%** margin, which is
> noise. The general claim it supports — a trend adds a second correction on top of a
> carry-forward that already moved — is basis-independent and still stands.

`fg3a` is the only quantity `make season-effects` marks **"trend + year effect"** — trend R²
0.93 at +4.07%/season. On the head it was the **worst** case for a trend:

| `fg3a` arm | val CRPS | test CRPS | held-out bias |
|---|---|---|---|
| `carry_forward` | — | 38.747 | −4.17% |
| **`base`** (selected) | **33.247** | **33.723** | **−3.74%** |
| `trend` | 35.443 | 33.900 | **+8.44%** |
| `year` | 34.309 | 35.408 | −9.01% |
| `trend_year` | 36.108 | 34.706 | **+11.48%** |

The trend more than doubles the absolute bias and flips its sign. **Two mechanisms, both
measured:**

- **The three-point trend decelerated.** +4.07%/season over 30 seasons, but **+1.61%/season
  over the last six** and a mean year-over-year change of **+1.39%** across the last five
  transitions (`season_effects_league_rates.csv`). A 30-season slope extrapolated one and
  two seasons into a flattening series over-shoots, and the held-out seasons are exactly
  where it flattened.
- **The head's own feature already carries the league level.** The dominant term is
  `log(fg3a_p36_lag1)` — the player's prior-season rate — which moves with the league. The
  carry-forward lags a league move by exactly one season, so what is needed is a *one-season*
  increment; a fitted trend adds a second, independently-sized correction on top of a base
  that has already moved. That is why the correction over-shoots rather than merely being
  mis-sized.

**Across the eight count heads the trend made held-out bias worse on six of eight**
(`fg3a` −3.74→+8.44, `fta` −7.40→−9.35, `blk` +4.75→+7.18, `fg2a` −0.27→−3.07,
`stl` −6.68→−5.98 and `tov` −3.36→−4.82). And where the trend *did* win on test it won on
heads with no era story at all — `stl`, whose trend R² is **0.03**. That is the overfitting
signature, not a finding.

⚠️ **On validation the same count is 3 of 7, and the heads are different ones.** Trend
against base, `val_bias_pct`: `fga` −0.06→**+1.25**, `reb` +2.07→**+2.79**, `stl`
+0.41→**+1.17** get worse; `fta` +1.55→+0.23, `ast` −3.73→−0.27, `blk` −1.79→−0.42 and `tov`
+2.39→+1.19 get better. **Do not read this as the trend being rehabilitated.** Two things
changed at once — the retired basis and the split — and the direction of a sub-1% bias
move on a head whose whole season-term margin is under 1% is exactly the quantity this
ablation has shown to be noise-dominated. What survives is that the trend moves bias in
**both directions across heads**, which is the premise §2 rests on, and that is unchanged.

#### 2. The trend's win on season-total dk_pts is cross-component cancellation

Composed through the chain over the eight scoring components, on the **773 validation**
player-seasons (bonus excluded, so this is exact and linear in the components):

| arm | MAE | bias | CRPS | coverage 50 / 80 / 95 |
|---|---|---|---|---|
| `base` | **105.71** | **−16.44** | 76.05 | 0.454 / 0.765 / 0.908 |
| `trend` | 105.76 | **+10.21** | **75.67** | 0.459 / 0.762 / 0.903 |
| `year` | 106.52 | −23.40 | 76.68 | 0.462 / 0.768 / 0.909 |
| `trend_year` | **105.00** | **+9.22** | **75.46** | 0.463 / 0.766 / 0.915 |

**The trend does not win MAE at all here — it is 0.05 dk_pts *worse* than base — and it
flips the bias from −16.44 to +10.21.** The component biases behind it move in **both
directions** (§1), so what the aggregate does is those errors **cancelling in the DK sum**:
the same cross-component cancellation this repo documents at 8.30× on
`teammate_assist_supply`, appearing as a false positive for a season term. A model that is
wrong in both directions and right on average is not a model of the league.

> ⭐ **This resolves the one open question the verdict carried, and it resolves it back to
> the original answer.** The table was **test-only until 2026-08-07** and read `base`
> 108.56 / −36.89 / 78.27, `trend` **106.06** / **−13.57** / 75.91, `year` 110.56 / −45.87 /
> 79.92, `trend_year` 105.92 / −12.89 / 76.06 — on which `trend` improved **both** MAE and
> bias, so the cancellation story no longer explained the aggregate win and `CLAUDE.md`
> flagged it as needing a human decision rather than a doc edit. On the split that is
> allowed to decide, the anomaly is gone: `trend` improves neither MAE nor bias, and the
> July reading (a trend winning MAE while flipping bias to +7.60) is what the validation
> frame reproduces in shape. **No decision is owed — the mechanism was never in doubt, only
> the arithmetic on one held-out frame.**

⚠️ **This table is uniform-arm, so it is confirmation and not selection.** The per-head table
above is what selects, and both now read the same split.

⚠️ **Coverage is well below nominal on every arm** (0.45 / 0.76 / 0.91 against 0.50 / 0.80 /
0.95) and no season term fixes it. That is the independent-draw composition missing the
residual copula, the same defect §8 attributes the bonus shortfall to — not an argument
about season terms in either direction.

#### 3. The oracle bounds the entire question at ~3% of MAE

`oracle_league` rescales each scored season by its own realized total — a perfect per-season
league multiplier, applied to every player. It is the most general form *any* league-level
term can take, so it is the ceiling on a fitted trend, a year effect and a manual override
alike, and it cannot be a model because it reads the season it forecasts.

As a share of the base arm's MAE it is worth: `fta` **4.97%**, `reb` 2.58%, `tov` **2.36%**,
`ast` 1.29%, `blk` **1.17%**, `stl` 0.36%, `fga` **−0.00%** — **median 1.29%, maximum
4.97%.** So even oracular knowledge of the league shift buys almost nothing at the player
level, because player-level error dominates it. **That is the single most important number
here**: it bounds every form of season term, and it is small.

`fga` is negative, which is not a defect: a perfect per-season rescale can make MAE
fractionally worse on a head whose league level barely moves, because it shifts every
player to correct an aggregate that was already right. It is the cleanest possible statement
that there is nothing there to win.

> ⚠️ **Superseded 2026-08-07 — measured on `test_mae` until then**, where it read `stl`
> **3.11%**, `blk` 2.32%, `fta` **2.16%**, `reb` 1.71%, `fga` **0.81%**, `ast` 0.58%, `tov`
> **0.01%**, median **1.71%**, max 3.11%. **The ordering reshuffled almost completely** —
> `stl` fell from first (3.11%) to sixth (0.36%) and `fta` rose from third to first — while
> the *size* barely moved: median 1.71% → 1.29%, max 3.11% → 4.97%. A ranking that
> reorders itself across a change of two scored seasons is measuring season-specific
> accident, not a stable property of the heads, which is the same conclusion §0 reaches
> from the selection margins. **Quote the magnitude, never the order.**
>
> The one substantive change is the ceiling's height: it is now **~5% of MAE** on the worst
> head rather than ~3%, so "the oracle bounds this at ~3%" should be quoted as **≤5%**. The
> conclusion is unaffected — a perfect league oracle is still worth a twentieth of the error
> at best, and no fitted form reaches an oracle.

#### 4. The year effect does exactly what it should, and recovers the league independently

Median validation ΔR² against `base` is **−0.00027** across the thirteen heads, which is the
mean-zero property surviving contact with the data. The largest is `fg2m|fg2a` at **0.022**,
and every other head is inside ±0.001.

It is also a **defined check rather than a hopeful comparison**: `sigma_year` is fitted by
NUTS on player-season rows and knows nothing about `make season-effects`, which measures the
league rate directly as totals over totals. On a log link the two are the same number in the
same units, and **3 of the 7 count heads land within 20%** —

| head | fitted `sigma_year` | measured league yoy sd | ratio |
|---|---|---|---|
| `reb` | 1.58% | 1.48% | **1.07** |
| `tov` | 2.56% | 2.89% | 0.89 |
| `blk` | 3.15% | 3.70% | **0.85** |
| `stl` | 2.30% | 3.03% | 0.76 |
| `fta` | 3.23% | 4.36% | 0.74 |
| `fga` | 2.10% | 1.52% | 1.38 |
| `ast` | 4.32% | 3.08% | 1.40 |
| **`fg3a\|fga`** (logit) | **11.42%** | 6.19% | **1.84** |

> ⚠️ **Every σ in this table moved on 2026-08-07, and NOT because the sampler is noisy.**
> The old `year_*` columns were written from the **test arm's** fit — trained on 27 seasons,
> train plus validation — into a row whose CRPS came from the **validation** arm, trained on
> 25. One row, two different models' numbers, and `year_n_train_seasons` (27 → 25) is the
> column that shows it. Nothing was leaking, but the σ being compared against the league
> series was never the σ of the model the rest of the row describes. Both now come from the
> validation fit.
>
> The recorded table read, as `sigma_year` / league yoy sd / ratio: `blk` 3.66% / 3.70% /
> 0.99, `tov` 2.71% / 2.89% / 0.94, `fta` 3.78% / 4.36% / **0.87**, `stl` 2.49% / 3.03% /
> **0.82**, `reb` 1.75% / 1.48% / 1.18, `fg2a` 3.03% / **2.42%** / 1.25, `ast` 4.46% /
> 3.08% / 1.45 and `fg3a` **15.43%** / **6.53%** / **2.36** — i.e. **5 of 8 within 20%**,
> on a head list that still carried `fg2a` and `fg3a`. Fitting on two fewer seasons
> shrinks σ on six of seven heads, which pushes `fta` (0.87 → 0.74) and `stl` (0.82 → 0.76)
> out of the 20% band. **The check still passes in substance** — a quantity fitted by NUTS
> on player-season rows lands within a factor of ~1.4 of a league aggregate it has no
> knowledge of, on every count head — but "5 of 8 within 20%" is now **3 of 7**, and the
> band was always an arbitrary cut through a continuum.

The retired `fg3a` at 2.36× was the tell: with no trend term to carry the secular climb, the
year effect absorbs **drift as a sequence of shocks** and then zeroes it at prediction time —
which is why its bias went to −9.01%. Its successor `fg3a | fga` shows the same signature
in the same place, at **1.84×** the league movement of the three-point *mix* — the highest
ratio of any head, and the mix is exactly the quantity with the strong secular trend. The
logit-link rows (`gp` 4.20×, `ftm|fta` 3.72×, `fg2m|fg2a` 3.16×, `min` 2.52×) differ by a
1/(1−p) factor and are a sanity check only.

**The Jensen inflation is real and negligible**: mean-zero on the linear predictor is *not*
mean-zero on the response, since `E[exp(σz)] = exp(σ²/2)`, and the measured multiplier runs
**1.000 to 1.008** across the heads (1.000 to 1.012 on the retired test-arm fits). Worth
reporting rather than assuming away; not worth correcting.

#### 5. What the year effect IS worth — and the plan understated it by another order of magnitude

Roster season-total dk_pts spread, sd of the summed total across posterior draws, averaged
over 200 random rosters:

| roster | `base` sd | `year` sd | inflation | shared-β, comparable board |
|---|---|---|---|---|
| 12 | 537 | 579 | **+7.95%** | +0.2% |
| **15** | **594** | **655** | **+10.4%** | **+0.2%** |
| 30 | 847 | 1,025 | +21.0% | +0.3% |
| 150 | 1,894 | 3,475 | +83.5% | +1.1% |
| 773 (whole validation board) | 4,465 | 15,812 | **+254%** | +6.4% |

**On a 15-man roster the year effect is worth ~50× the shared-β term** the Stan work was
built for. The plan's "larger by an order of magnitude" is still understated by a further
order. The mechanism is the one the plan named: a league shift is perfectly correlated across
players, so it grows as **N** while independent error grows as **sqrt(N)** — and unlike
shared-β, which is negligible on one roster and only matters board-wide, this one is already
+8% at twelve players.

⚠️ **The shared-β column is a different board and a different head.** It comes from
`stan_availability.board_correlation`, which measures the availability head alone on its own
validation board; this column composes eleven heads on 773 component-design rows. The
comparison is the right one in kind — non-diversifiable against diversifiable — but the
ratio is indicative, not a like-for-like division.

⚠️ **Treat these as an upper bound.** The fitted `sigma_year` exceeds the independently
measured league movement on `fga`, `ast` and `fg3a|fga`, so part of this spread is
player-level heterogeneity the year effect has absorbed rather than league movement.

> ⚠️ **Superseded 2026-08-07 — measured on the 791-player TEST board until then**, where it
> read 753 / 866 (**+8.9%**) at 12, **557** / **621** (**+11.6%**) at 15, 1,213 / 1,671
> (+22.1%) at 30, 2,796 / 6,652 (+91.1%) at 150 and 7,098 / 32,967 (**+278%**) across all
> **791**. The whole-board row is keyed on board size, so it is a different row rather than
> a moved value. Every inflation figure came down slightly and the shape — near-flat to 30
> players, then growing hard — is unchanged, which is the part the argument uses.

#### 6. Minutes is the one head that adopts a season term — and it falsifies a recorded hypothesis

| `min` arm | val CRPS | val R² | val bias (minutes) |
|---|---|---|---|
| `carry_forward` | 161.45 | 0.854 | **+23.9** |
| `base` | 144.09 | 0.883 | −14.2 |
| `trend` | 144.44 | 0.882 | **−25.0** |
| **`year`** (selected) | **143.81** | **0.884** | **−13.0** |
| `trend_year` | 144.28 | 0.882 | −26.9 |

The year effect wins, and it is the one selection in this ablation with a margin that has
survived a change that should not touch it: `logit_own_spline` + `year` reproduced across
the shot-attempt basis change **and** across the split conversion, to four decimal places
both times. Contrast §0, where the arms shuffle on sub-1% margins.

**And the trend settles an open question in `docs/availability-plan.md`.** That doc proposed
the minutes head's bias as "the signature an era effect would leave". Adding a trend makes
the bias **worse by 10.8 minutes** (−14.2 → −25.0), and by 12.7 on `trend_year`. An era
effect the head was failing to track would have been *corrected* by a trend, not amplified
by it. So the minutes bias is **shrinkage toward a 30-season mean, not an era effect** — the
hypothesis is falsified rather than left open, and the falsification is what survives here
even though the bias level itself did not.

> ⚠️ **The bias column changed meaning on 2026-08-07 and the sign structure changed with
> it.** The old bare `bias` column was the **held-out** one, reading `carry_forward`
> **−5.69**, `base` **−41.05**, `trend` **−56.59**, `year` **−38.19**, `trend_year` −56.63,
> beside test CRPS of 168.24 / 147.02 / 148.78 / **146.54** / 148.85. On validation the
> **floor over-predicts by +23.9 minutes** where it under-predicted by −5.7 on test, and the
> fitted arms sit −13 to −27 rather than −38 to −57. This matches `stan_minutes` exactly,
> whose own validation floor bias is +23.91, and it is why `CLAUDE.md` withdrew "the fitted
> heads carry a −33 to −41 minute bias against the floor's −5.7". **What reproduces is the
> GAP, not the level**: the fitted arms sit 27–51 minutes below the floor on both splits.
> Do not quote a signed bias level for this head; quote its distance from the carry-forward.

#### 7. Availability: the season × role interaction is a validation null

`docs/availability-plan.md` finds the availability era effect is role-graded (−0.101 of
games-played share for heavy-minute players against −0.037 for fringe), and
`season_effects_regimes.csv` now confirms it as a dated policy step — the 2023-24
Participation Policy is a **−4.63%** level break on `gp_share [30+ mpg]` at **p = 0.008**
while the fringe bucket is +5.37% and not significant. So the era effect is real. It does not
transfer:

| arm | features | **val CRPS** | val R² | val bias |
|---|---|---|---|---|
| `carry_forward` (league/age) | 0 | 13.387 | −0.057 | +5.52 |
| `base` | 19 | 10.007 | 0.374 | +0.98 |
| **`trend`** (selected) | 20 | **9.997** | **0.375** | −1.87 |
| `year` | 19 | 10.015 | 0.373 | +1.39 |
| `trend_year` | 20 | 10.005 | 0.374 | −1.82 |
| `trend_x_role` | 26 | **10.078** | 0.366 | −1.87 |
| `trend_x_role_year` | 26 | **10.087** | 0.364 | −1.79 |

**The two role arms are the worst two of the seven on validation** — the most expensive arms
in the ablation, at 26 features, buying a *loss* of 0.08 games against `base`. And the arm
that *is* selected, `trend`, is worth **0.010 games of CRPS**, which is nothing.

> ⭐ **The finding here WAS the val/test disagreement, so it is worth stating plainly that
> the test half is now retired.** Those same two role arms were the **best two on test**
> (**10.742** and **10.736** against `base` **10.797**, with `carry_forward` 13.614,
> `trend` 10.765, `year` 10.813 and `trend_year` 10.757) while being the worst two on
> validation. That is the exact false-positive shape this repo has already shipped once —
> the nonlinearity arm whose paired bootstrap on test read [−0.079, −0.015] with
> P(Δ<0) = 99.7% and did not replicate — and it was caught only because selection never read
> the test column. `src/models/held_out.py` now stops the test column from being computed at
> all, so **this particular contrast cannot be re-run**; it is preserved here as the record
> of why the lock exists. The verdict does not depend on it: the role arms lose on
> validation, which is the only split that decides.

#### 8. Bonus-threshold calibration

| arm | predicted / game | realized / game | bias |
|---|---|---|---|
| `base` | 0.1207 | 0.1370 | **−11.7%** |
| `trend` | 0.1264 | 0.1370 | −7.6% |
| `year` | 0.1189 | 0.1370 | −12.8% |
| `trend_year` | 0.1272 | 0.1370 | **−6.9%** |

Every arm under-predicts the bonus by 7–13%, and **that level is mostly not the season
term** — this composition draws the eleven heads independently given realized minutes, so it
omits the positive cross-component dependence the bonus threshold needs (measured
off-diagonals average +0.0225 across the seven counts). It is a standing argument for the
residual copula, not against a season term. The *ordering* tracks the bias story exactly:
the two trend-carrying arms over-predict the components least and so under-predict the bonus
least, which is the same cancellation §2 describes arriving at a convex functional.

⚠️ **Superseded 2026-08-07** — on the 791-player test board the same table read **−14.8%**,
**−10.2%**, **−16.5%** and **−9.9%** against a realized 0.1391/game, i.e. a 12–19% shortfall.
The gap narrowed by about three points on validation and the ordering is identical.

### What to do, and what not to

- **Do not add a trend to any head.** ⚠️ **This withdraws the head-level reading of
  "`fg3a` is worth a trend"**, which `make season-effects` supports as a statement about the
  *league series* and which is false of the *head*: the head's carry-forward feature already
  tracks the level, and the 30-season slope is no longer the current slope.
- **Do not add a year effect to the component heads for accuracy.** It is mean-zero by
  construction, and where it absorbs drift it costs bias.
- **Do give the simulator a year effect as a variance component**, because it is the only
  mechanism in the project that produces non-diversifiable board-level risk, and it is worth
  ~95× the shared-β term on a 15-man roster. Take σ from the **measured** league movement
  (`season_effects_summary.csv`, `yoy_sd_pct`) rather than the fitted per-head value, which
  over-shoots on three of eight count heads. Draw it **per head, not shared** — the measured
  cross-component shock correlation is **−0.009** on average over 136 pairs, so there is no
  common factor, though the `fg2a`–`fg3a` pair at **−0.833** is the substitution the heads
  already reparameterize away.
- **Adopt the year effect on the minutes head.** It wins on both splits and improves the
  standing bias.
- **Keep the manual override path** (`stan.season_terms.league_override` in
  `configs/default.yaml`). It is legitimate point-in-time information because rule changes
  are announced in the summer — but the oracle says its ceiling is ~3% of MAE, so wire it and
  do not expect much.
- **Do not extrapolate any trend across 2023-24.** `season_effects_regimes.csv` finds a
  significant break, and with only **3** post-break seasons the level+slope arm's
  one-season-ahead extrapolation moves by up to **+22.1%**, which is noise. The break test
  disqualifies extrapolation rather than supplying a better slope.

### The rate side is nearly saturated — measured, with a mandatory floor

`make component-rates` (`src/models/component_rates.py`), season-collapsed heads on 10,194
player-seasons, scored on validation (2022-23/23-24). Three results bind on everything below:

- **`carry_forward` — prior per-36 rate x actual minutes / 36, no fitting at all — scores
  validation R² 0.81–0.95**, and the best of seven fitted variants beats it by only **+0.0013 to
  +0.0334**. Every component head must be quoted against it; `beats_floor` is on every output
  row. Combined with `oracle_gp` 214.4 vs `oracle_rate` 261.9 on the season total, this says
  remaining value is in **availability and the joint structure**, not in richer rate features.
- **The specification is scale, not curvature**: `log E[rate] = β·log(prior rate)`. Linear-in-
  raw-rate inside `exp()` is misspecified, catastrophically for the zero-heavy heads (`blk`
  **0.651** against **0.775** on the log scale, and the retired `fg3a` 0.520 against 0.879).
  Splines add **+0.054** (`blk`) and ≤ +0.005 elsewhere; `age × own` and `mpg × own`
  interactions are a null. (Held-out, before 2026-08-05: `blk` 0.637 / 0.820, spline +0.040.)
- **Walk-forward PCA of the 156-column season matrix is a null.** Refitted per target season on
  S-1 and earlier only (a pooled basis leaks the future invisibly), 10 components replacing the
  raw context columns: within ±0.003 of the raw spec on every head. The style and tracking
  families add nothing once you have the player's own prior rate and his minutes. Use the cheap
  raw spec.
- **`ftm|fta` is the one head nothing beats** — free-throw percentage is pure player skill, so
  an empirical-Bayes shrink of the prior is already optimal. The conversion side is worth
  ~1–2% of its NLL at most (`fg2m|fg2a` **+0.079**, `fg3m|fg3a` **+0.028**; held-out +0.072
  and +0.032). ⚠️ On validation the shot-mix head `fg3a|fga` joins it in failing under
  sklearn — 4.6317 against a floor of 4.6191, where on test it cleared. The Stan run splines
  on `logit(own)` rather than raw `own` and clears comfortably, so read this as the sklearn
  probe being the coarser instrument.

> ⚠️ **The first version of this measurement was an artifact**, and the failure mode is worth
> carrying: `sklearn`'s `PoissonRegressor` averages the deviance by the weight sum, so with
> minutes as `sample_weight` an `alpha=1.0` crushes every coefficient *silently* — a flexible
> basis partially compensates, so splines and interactions looked like real signal. The no-fit
> floor is what exposed it, which is why it is now mandatory rather than advisory.

### Architecture: separate Stan models, one per head — settled *and built* 2026-07-29

> ### ✅ Built — `make stan`, cmdstanpy 1.3.0 / CmdStan 2.39.0
>
> `src/models/stan_{availability,minutes,components}.py` over **two** `.stan` files in
> `src/stan/`, which is the factorization argument as code rather than as prose:
> `betabinomial_glm.stan` serves availability *and* minutes *and* the three conversion heads
> — the same likelihood with different `y`/`n` — and `negbinomial_glm.stan` serves the eight
> counts. Sources are checked in; cmdstanpy compiles a copy into `outputs/stan/`, which
> `.gitignore` already covers, so no binary is committed and no generated `.hpp` lands in
> the source tree.
>
> **Availability** ports the validated point MLE and reproduces it: validation CRPS 9.8136
> against 9.8444, ρ 0.2595 against 0.2627, and the MLE inside the 95% credible interval for
> **24/24** terms. The prior is set to `normal(0, 1/sqrt(2·l2))` precisely so the
> posterior *mode* is the penalized MLE, making that a defined check. R̂ 1.0050, 0
> divergences, 94 s. What the posterior adds is `Var_θ(Σ_i E[Y_i|θ])` — exactly 0 for any
> point estimate — but **its size depends on the portfolio**, and this plan's framing
> oversold it. The independent term grows as sqrt(N) and the shared-β term as N, so measured
> on the validation board the spread inflation is **+0.5% on a 15-player roster** and **+12.3%
> across all 883**. Real for board-wide exposure across many lineups; near-irrelevant for one
> drafted team. This matters for the "joint / correlation modeling across teammates" section
> below: shared *parameter* uncertainty is not the correlation source a single roster needs —
> shared **team state** and the shared `min` draw still are.
>
> ⚙️ **Both figures are the 2026-08-11 head**, which fits a 2012-13 window with a role-graded
> ρ (`docs/availability-window-plan.md` §4). The full-window, shared-ρ head it replaced read
> CRPS 10.0063 against 10.0057, ρ 0.2808 against 0.2806, 21/21 terms, R̂ 1.0019, 196 s, and
> +0.2% / **+6.7%** board inflation. The board term roughly doubled because 4,027 fitting
> rows leave a wider posterior on β than 9,478 do — the one place the window costs something.
>
> ⚠️ **The port check and the board table were held-out measurements until 2026-08-05**,
> reading CRPS 10.7947 against 10.7952, ρ 0.2759 against 0.2757, R̂ 1.0025, 254 s, and +0.2%
> / **+6.4%** across a **911**-player board. Both moved to validation with every other head;
> the board additionally *belongs* there, being a simulator input.
>
> **Minutes** is new and is the first head to use the real trials denominator — successes
> out of actual game length, never 48. See `docs/availability-plan.md` for the sweep; it
> clears its no-fit floor by +0.030 R² and −17.5 minutes of CRPS, and it reports the
> **game-level** dispersion (4.65× binomial) separately from the season-level ρ the collapse
> estimates, because the simulator needs the former and the fit only sees the latter.
>
> ⚠️ **The minutes sweep was a held-out measurement until 2026-08-06** and read a gain of
> **+0.041** R² and **−21.4** minutes over the floor, on a test R² of **0.8572** against the
> floor's **0.8166**, with the spline arm costing **899** s against the linear arm's
> **189** s. Those figures are the retired test refits at double the iterations; the
> per-game costing above is scaled from the validation fits the artifact now holds.
>
> **The eleven component heads** are built too — 37 fits, **0 divergences**, max R̂ 1.0076,
> 137.4 min of compute. 10,194 player-seasons, 8,630 fit / 773 validation on 2022-23 and
> 2023-24. Three findings, two of which **overturn what the sklearn run above measured**:
>
> - **`log(own)` alone is not sufficient under a negative binomial.** The Poisson fit has
>   `log_own` at 0.7748 (`blk`; 0.8204 before the split moved) and, on the retired basis,
>   0.8791 (`fg3a`); under NB the identical spec collapses to
>   **0.6730** and **0.3719**, both far below their floors, and only a spline recovers them
>   (0.8309, 0.9046). NB2's `var = μ + μ²/φ` down-weights large counts, so the fit is driven
>   by the low-count mass — exactly where the log-scale relation is most curved. **The
>   "splines are worth ≤ +0.003 outside `fg3a`/`blk`" guidance above is Poisson-specific**;
>   under NB the validation split picks a spline for **four** heads (`fga`, `ast`, `blk`,
>   `stl`), and on `blk` it is the difference between a model and a failure.
> - **`linear` is worse than the sklearn run suggested** — validation R² **−0.2744** on `blk`
>   against 0.638 under sklearn, and **−19.00** on the retired `fg3a`. Linear-in-raw-rate
>   inside `exp()` is not merely misspecified, it is unusable, and it fails the no-fit floor
>   on five of the seven count heads.
> - **⭐ Adopting the shot-attempt basis (2026-08-04) retired the −19.00 case entirely.**
>   `fg3a` is no longer a count head; `fga` replaces it and is the best-behaved count in the
>   project — floor **0.9514**, the highest of the seven, selected at **0.9584**, and
>   `log_own` already at **0.9581**. Where the wrong scale cost `fg3a` a catastrophic
>   −19.00, it costs `fga` **0.9489**, i.e. 0.0025 R². A total is far less skewed than its
>   three-point part. `blk` at −0.2744 is now the only negative linear arm left, and
>   the `fg3a` figures above are retained as the record of the basis that was retired.
> - **⚠️ `fta` cleared its floor when the sweep moved to validation, and "the whole
>   free-throw family fails" is withdrawn.** It reads **0.8909** against a floor of
>   **0.8765** at its selected `log_own`, where the test column had it at 0.8649 against
>   0.8673 — a failure by 0.0024, which was never a margin worth a finding. **`ftm|fta`
>   still fails at every variant and is now the only head in the project that does.** That
>   was always the better-founded half: free-throw *percentage* has a pure-player-skill
>   argument that trips to the line never had.
>
> ⚠️ **Every figure in this block was a TEST measurement until 2026-08-06.** The superseded
> readings, kept because the `fta` reversal is only legible beside them: `blk` `log_own`
> **0.6794** and spline **0.8579**; `blk` linear **−1.393**; `fga` floor **0.9464**, linear
> **0.9396**, `log_own` **0.9501**, spline **0.9505**; and the sampler at **74** fits,
> **305.0** min, max R̂ **1.0118**. The sweep ran 9,403 train / **791** test. Nothing here
> reversed except free throws, and the two arms that look most changed — `blk` linear from
> −1.393 to −0.2744, `fga` linear from 0.0068 below its floor to 0.0025 — tell the same
> story at a different magnitude.
>
> **✅ The 3PA/2PA reparameterization is settled, and it wins decisively.** `fga` as a count ×
> `fg3a | fga` as a binomial share beats two independent count heads by **−0.771 nats on
> validation and −0.793 on test**, per player-season, on the joint density of `(fg2a, fg3a)`
> (10.797 → 10.026; 10.784 → 9.991), replicating on both splits. The comparison is legitimate
> because `(fg2a, fg3a) ↔ (fga, fg3a)` is a **bijection with unit Jacobian on the integers**,
> so the two joint log-densities are directly comparable. The recommendation below is now a
> measurement.
>
> ⚠️ **Both artifacts went validation-only on 2026-08-06, so the test half of that pair
> (10.784 → 9.991, −0.793) is now a record rather than a measurement.** It had survived the
> conversion of `stan_component_substitution.csv` by being read out of
> `stan_component_substitution_sweep.csv` instead, but `make stan-substitution` was
> re-run validation-only the same day and that copy is gone too. Those three figures are
> presence-checked in `src/docs_audit.py` and value-checked nowhere. **The validation half
> reproduced unchanged at full-length chains** and is still audited: 10.797 → 10.026 at
> −0.771.
>
> Every head is quoted against `carry_forward`, and every variant is selected on the
> validation split. **There is no longer a test column to report** — the held-out reading is
> taken once, by `make final-evaluation`.

**Fit the heads separately.** The chain of conditionals factorizes the joint posterior exactly
when parameter blocks are distinct, so separate fits are not an approximation — they recover the
identical posterior. Full argument in `CLAUDE.md`; the short version is that `megamodel.stan`
shared no parameter between any two heads, so its joint fit bought nothing and cost 99% of the
data. Correlation for the simulator comes from (1) a shared `min` draw, then (2) a residual
copula if needed — measured off-diagonals average **+0.0225** across the seven counts, max **+0.1329**. Go joint only for a
correlated multivariate player effect, and only after measuring it is worth it (the player random
effect on rates was largely in-sample leakage). Reparameterize the 3PA/2PA substitution as
`fga` count × `fg3a | fga` share rather than coupling two Poissons — ✅ **measured 2026-07-30
and worth −0.771 nats per player-season on validation**, see the built-block above.

## New model surface area

### Joint / correlation modeling across teammates

Everything built so far — this repo and `nba_stats` both — predicts players marginally. A
portfolio needs the joint distribution. Sources of correlation to capture: shared team state
(pace, injury shocks, minutes redistribution), zero-sum usage within a team, same-game
stacking.

Two candidate approaches:

- A copula over the per-player marginals, driven by shared team-level factors.
- A full team-state Monte Carlo: sample one shared team state per simulated game/week (who's
  out, how usage and pace redistribute), then draw players conditionally on it.

Prefer the second — it's the natural extension of the spell-based season simulator
`availability-plan.md` already recommends, and it reuses the team-context machinery that
already exists rather than inventing a new correlation model from scratch.

### Explicit minutes/usage redistribution on teammate absence

`teammate_usage_load` (already the strongest own-team feature) and the availability posterior
together imply a concrete, buildable feature that neither prior attempt built: when a
teammate's availability draw says "out," redistribute a share of his usage/minutes to
teammates weighted by their `teammate_usage_load` role, instead of treating every player's
minutes as independent of teammates' health. `nba_stats`'s injury adjustment zeroed the hurt
player and left every other prediction untouched — this is the direct fix.

**The team-game composition model above is the principled version of this — and it is now
built, so the fork is closed in its favour.** The argument was that a redistribution *rule*
weighted by `teammate_usage_load` is a hand-set heuristic on top of independent marginals,
whereas a composition model makes the same redistribution a **fitted** quantity with
zero-sum exact by construction; the advice was to measure the cheap heuristic first because
the composition was "days".

It was not days, and the heuristic is no longer the cheaper path. The composition's offset
*is* proportional redistribution — a missing teammate shrinks the renormalizer and scales
every remaining player up — so the heuristic arrives as the `β = 0` special case, and the
fitted `β` measures deviations from it. There is nothing left for a hand-set rule to add
that the fit does not already estimate. See `docs/minutes-composition-plan.md`.

## ADP as a prediction input / validation signal

**See `docs/adp-plan.md`** — sourcing, storage, point-in-time rules and the integration decision
all live there, measured. This section records only what that plan concluded *for this layer*, so
the two do not drift.

**ADP does not feed the GLMM.** It stays a strategy-layer input, downstream of predictions
(`docs/simulations-plan.md`). Four reasons, in full there; the load-bearing one is that under a
knockout payout the edge *is* model-minus-market, so a model fit on ADP reproduces the benchmark
it is meant to be measured against. Coverage seals it: ADP exists for ~250 players over 12
seasons against the component heads' 10,900 player-seasons over 30.

**The one exception, and it is worth measuring.** 14.7% of roster minutes have no usable
prior-season row (8.7% true rookies, 5.1% sub-threshold, 0.9% returnees), where the model imputes
from `bio_draft_number` while the market has seen summer league and camp. For those rows only, use
ADP to set the **prior mean on the rate head**, with prior weight tied to the existing
`0.924 · m/(m+66)` reliability curve so it decays to zero as prior minutes accumulate. Established
players stay market-free. Test it against the `bio_draft_number` imputation on held-out CRPS
before shipping; given this repo's record on ceilings, a settled null is the likely outcome.

**What the sourcing research settled** (details and reproduction in `docs/adp-plan.md`):

- The DK snapshot is real ADP — fractional, live-computed, **249 of 698 pool rows** carry a value.
  DK's best-ball pages have **zero Wayback snapshots** and no unauthenticated API, so the one file
  is genuinely irreplaceable and the 2026-27 board (opening ~October 2026) is a **dated capture
  commitment**, not a cron job.
- FantasyPros via Wayback parses across **all 12 seasons 2014-15 → 2025-26** with no missing
  season — and it **does** carry CBS in several years, contrary to the "Yahoo + ESPN only"
  assumption recorded here previously.
- The two-option fork above is **resolved, and to neither option**: a one-dimensional monotone
  recalibration of consensus onto DK's scale is worth **24.4 → 17.3 picks** of cross-validated
  error, while a position offset — the obvious next term — adds a further −2.0. One anchor
  identifies a shape, not a model.
  - ⚠️ **The offset figure was −0.3 until 2026-07-31, and the correction is not a rounding.**
    Both numbers are right for their own population: −0.29 is the planning-session measurement
    on 218 pairs, preserved in `docs/adp-plan.md`, and −2.01 is `adp_profile.csv` on the 226
    pairs the improved matching cascade recovered. The eight extra pairs are names the surname
    guards let through, and they land disproportionately in the late rounds where the position
    effect is largest. **This weakens the "one anchor identifies a shape" argument without
    overturning it** — the offset is now 11.6% of the recalibrated error rather than 1.7%,
    against the monotone step's own −7.1 picks, so the ordering of the two terms is unchanged
    and the decision to keep the offset "for interpretability rather than accuracy" is worth
    revisiting when the second anchor lands in October 2026.

## Validation

- Same temporal walk-forward split as the rest of the repo.
- CRPS/PIT discipline matching the availability head — not point accuracy alone.
- New: posterior-predictive checks for the joint/correlation claims specifically — e.g., does
  simulated team-total variance match realized team-total variance held out; does simulated
  same-team pairwise dk_pts covariance match realized covariance.

## Open questions / risks

- Whether the joint/correlation structure earns its complexity is unmeasured. Follow the
  project's standing discipline: start with a cheap copula baseline and compare it against an
  independent-marginals baseline on held-out team-total variance before building the full
  team-state simulator.
- ADP sourcing is unresolved (above).
- ~~A hierarchical Stan/PyMC fit at this scale may carry real compute/engineering cost — worth a
  small-scale timing check~~ — ✅ **answered by building it.** Season-collapsed, the whole
  surface is cheap: the availability head is 254 s and the eight count heads are ~1–3 min each
  on 4 chains. The cost wall is **not** the hierarchy, it is abandoning the collapse — see the
  per-game minutes subsection, where the same head goes from 8,306 rows to 731,863.
- ~~**Whether minutes should be fitted per-game at all is open and deliberately deferred**~~
  — **partly answered 2026-07-31.** Of the three options costed under "Fitting strategy",
  the **team-game composition model is built and wins** (`make stan-composition`,
  `docs/minutes-composition-plan.md`): **−0.2898** minutes of validation CRPS against the
  independent per-player draw, and both the individual cap and the team total exact by
  construction. It was an afternoon rather than days, because the season collapse was never
  what made it expensive — the numerics were.
  - Still open, and **unaffected by this**: the composition is iid across games, so it does
    nothing about the 2.43× ten-game block inflation. **The residual serial process remains
    the recommended next move** on that axis, and it now sits *on top of* the composition
    rather than beside a per-player marginal.
  - ~~Also open: one shared ρ across the roster~~ — **✅ graded 2026-07-31** by
    prior-share quartile, cutting mean |variance ratio − 1| by 59% and fixing the star
    tier outright. What remains is the fringe tier at 1.21 and the full-window fit at
    ~10–12 h.
  - The two retired premises stand: the 48-minute bound is not a reason to go per-game, and
    a single lagged-observation term reproduces only **46%** of the block-variance excess.
