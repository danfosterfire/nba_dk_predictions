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
| `fg2a` | count — Poisson or NB | `min` |
| `fg3a` | count — Poisson or NB | `min` |
| `fta` | count, but **arrives in pairs** — model *trips* and double | `min` |
| `fg2m` | successes / trials | `fg2a` |
| `fg3m` | successes / trials | `fg3a` |
| `ftm` | successes / trials | `fta` |
| `reb` | count | `min` |
| `ast` | count | `min` |
| `stl` | count | `min` |
| `blk` | count | `min` |
| `tov` | count | `min` |

Availability sits **upstream** of all twelve — it gates whether the player-game exists at all
(`docs/availability-plan.md`), and `min` is drawn conditional on it.

Reassembly is exact: `pts = 2·fg2m + 3·fg3m + ftm`, then `preprocess.compute_dk_pts` with
`fg3m`, `reb`, `ast`, `stl`, `blk`, `tov`. Only **eight** components reach the scoring
function — `fg2m`, `fg3m`, `ftm`, `reb`, `ast`, `stl`, `blk`, `tov`. `min` and the three
attempt counts matter solely through the exposure and trials they supply to those eight, and
`fg2a`/`fg2m` must be derived (`fga - fg3a`, `fgm - fg3m`) because the stored `FGA`/`FGM`
*include* threes.

The double-double / triple-double bonus is a simultaneous threshold on
`pts`/`reb`/`ast`/`stl`/`blk`, so **the deliverable is a joint draw, not twelve marginals** —
`E[bonus] ≠ bonus(E[x])`. This is the same requirement the correlation work below imposes for
a different reason, and the two should be satisfied by one mechanism.

## What's already decided and built — not re-litigated here

- dk_pts decomposed into shot classes and counting components (`CLAUDE.md`, `make target-profile`).
- Availability is a beta-binomial GLM head; it beats ridge/GBM/league-age baseline on CRPS and
  is worth **211.1** dk_pts of season-total MAE (`src/models/availability.py`, `docs/availability-plan.md`).
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
R² 0.8572 against a no-fit floor of 0.8166, so the headroom being competed for is small.

**Costs, with real numbers.** 731,863 played regular-season player-games against the season
head's 9,048 rows — **81× the data**. The season spline fit took 899 s (linear: 189 s), so a
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
> Fitted on all 30 seasons and held out on 2024-25/2025-26, the selected variant scores
> **4.5592** minutes of CRPS against the no-fit floor's 4.8576 and the independent
> per-player draw's **4.9140** —
> so it beats the incumbent on the incumbent's own marginal metric, which this plan
> expected to be a wash, *and* the independent draw's mean team-sum error is **36.87**
> minutes per team-game against the composition's exact zero.
>
> Two results worth carrying back here. **The pure decomposition fails**: the plain
> binomial arm scores 4.9732 with PIT KS 0.1948, below the floor — the measured 4.65×
> game-level dispersion is not optional, exactly as the NB-vs-Poisson result on the count
> heads. And **the offset is the floor**, so proportional redistribution comes for free
> and `β` fits deviations from it — which makes "who absorbs the minutes when a starter
> sits" a fitted quantity, the thing the redistribution section below wants.
>
> The dispersion is now **graded by prior-share quartile** (fitted 0.1480 fringe to
> 0.0613 star, a 2.41× spread against one shared 0.0970), which cuts mean |variance
> ratio − 1| by 59% and lands the star tier at 0.99. Still open: the fringe tier
> remains **1.21** and q2 is now 0.84 — grading a *step* dispersion does not map
> one-to-one onto *marginal* variance, because a low-share player breaks his stick
> last and inherits the remainder variation ahead of him. The full-window fit
> (~10–12 h) has not been run. This is
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

**What ignoring it costs — the no-fit floor's bias on the held-out seasons.** Carry-forward
lags any league move by exactly one season, so its bias *is* the season effect measured on
the scale the heads are scored on. No fitted head corrects it, because none has a season
term:

| component | 2024-25 | 2025-26 | both |
|---|---|---|---|
| **`fta`** | −3.1% | **−10.7%** | **−7.0%** |
| `stl` | −10.0% | +0.7% | −4.6% |
| `fg3a` | −7.0% | −1.3% | −4.2% |
| `tov` | −4.6% | −1.8% | −3.2% |
| `ast` | −1.1% | −4.8% | −2.9% |
| `blk` | +6.5% | +6.0% | **+6.2%** |

`fta`'s −10.7% in 2025-26 is the +8.6% league jump arriving one year late. `blk` is biased
+6% in *both* seasons, which is drift, not noise.

> **Why this outranks the shared-β correlation the Stan work was built for.** A league shift
> is **perfectly correlated across every player**, so it does not diversify away: a −7% error
> on free throws is −7% on a whole roster's free-throw points. The shared-β parameter
> uncertainty measured in `stan_availability.board_correlation` is worth **+0.2%** on a
> 15-man roster. Season effects are the larger non-diversifiable risk by an order of
> magnitude, and they are currently modelled as exactly zero.
>
> ✅ **Now measured, and the "order of magnitude" was an under-statement by a further
> order** — `make season-terms` (`season_term_roster_spread.csv`): a year effect widens a
> 15-man roster's season-total dk_pts spread by **+11.6%** against shared-β's **+0.2%**,
> i.e. ~**95×**, and by **+278%** across the whole 791-player board against **+6.4%**. See
> the verdict below, which also finds this is the *only* thing a season term is worth: on
> point accuracy the ceiling is ~3% of MAE.

**One piece is genuinely knowable at prediction time and should not be lumped in with the
rest.** Rule changes and points of emphasis are announced in the summer, before opening
night — the 2021-22 non-basketball-moves emphasis was public in advance. A manual
league-level override is therefore legitimate under the point-in-time discipline, unlike
anything drawn from within the season.

### ✅ VERDICT — the ablation ran 2026-07-31, and **no head ships a season term**

**Reproduce with `make season-terms`** (`src/models/season_terms.py`) →
`outputs/predictions/season_term_{metrics,season_total,roster_spread,bonus,sigma_vs_league,diagnostics}.csv`.
108 fits, **0 divergences**, **0 treedepth-saturated draws**, max R̂ 1.0142, 155.9 min. Five
fits sit marginally over the 1.01 R̂ bar — worst 1.0142, all with ESS ≥ 371 and no
divergences — and **four of the five are `base` arms**, so the season terms are not what
strains the sampler. Four arms per head — `base`,
`trend`, `year`, `trend_year` — on top of each head's already-selected specification, plus
`trend_x_role` and `trend_x_role_year` for availability. Selected on the validation split
(2022-23/2023-24), confirmed on test (2024-25/2025-26), every arm quoted against its no-fit
floor. Every fit uses `metric="dense_e"`: on the `blk` spline base that is **13.4 s against
236.6 s** with treedepth saturation **0 against 35**, which is what made 108 full Bayesian
fits affordable at all.

> ⚠️ The ablation runs at the **selection** sampler budget (500/500) on *both* splits, so
> the `base` arm's test column is not directly comparable to `stan_component_metrics.csv`,
> whose test fits run at 1000/1000. Differences between the two tables are sampler noise.
> Within this table every arm shares one budget, which is what the contrast needs.

#### 1. The trend is refuted, and most sharply on the one quantity that predicted it

`fg3a` is the only quantity `make season-effects` marks **"trend + year effect"** — trend R²
0.93 at +4.07%/season. On the head it is the **worst** case for a trend:

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

**Across the eight count heads the trend makes held-out bias worse on six of eight**
(`fg3a` −3.74→+8.44, `fta` −7.40→−9.35, `blk` +4.75→+7.18, `fg2a` −0.27→−3.07,
`stl` −6.68→−5.98 and `tov` −3.36→−4.82; only `stl` and `reb` are unharmed). And where the
trend *does* win on test it wins on heads with no era story at all — `stl`, whose trend R²
is **0.03**. That is the overfitting signature, not a finding.

#### 2. The trend's win on season-total dk_pts is cross-component cancellation

Composed through the chain over the eight scoring components, held out on 791 player-seasons
(bonus excluded, so this is exact and linear in the components):

| arm | MAE | bias | CRPS | coverage 50 / 80 / 95 |
|---|---|---|---|---|
| `base` | 108.56 | −36.89 | 78.27 | 0.612 / 0.858 / 0.980 |
| **`trend`** | **106.06** | **−13.57** | **75.91** | 0.623 / 0.885 / 0.984 |
| `year` | 110.56 | −45.87 | 79.92 | **0.584 / 0.847 / 0.979** |
| `trend_year` | 105.92 | −12.89 | 76.06 | 0.618 / 0.885 / 0.984 |

Read alone, the `trend` row says ship a trend everywhere. It should not be read alone. The
component biases behind it move in **both directions** — `fg3a` +8.44%, `blk` +7.18% against
`fta` −9.35%, `fg2a` −3.07% — so the aggregate improvement is those errors **cancelling in
the DK sum**, which is the same cross-component cancellation this repo already documents at
8.30× on `teammate_assist_supply`, now appearing as a false positive for a season term. A
model that is wrong in both directions and right on average is not a model of the league.

⚠️ **This table is uniform-arm and test-only, so it is confirmation and not selection.** The
per-head validation table above is what selects.

#### 3. The oracle bounds the entire question at ~3% of MAE

`oracle_league` rescales each held-out season by its own realized total — a perfect
per-season league multiplier, applied to every player. It is the most general form *any*
league-level term can take, so it is the ceiling on a fitted trend, a year effect and a
manual override alike, and it cannot be a model because it reads the season it forecasts.

As a share of the base arm's MAE it is worth: `stl` **3.11%**, `blk` 2.32%, `fta` **2.16%**,
`reb` 1.71%, `fga` **0.81%**, `ast` 0.58%, `tov` 0.01% — **median 1.71%,
maximum 3.11%.** So even oracular knowledge of the league shift buys almost nothing at the
player level, because player-level error dominates it. `fta` carries a **−7.0%** systematic
bias and removing it entirely recovers **2.2%** of MAE. **That is the single most important
number here**: it bounds every form of season term, and it is small.

#### 4. The year effect does exactly what it should, and recovers the league independently

Median held-out ΔR² against `base` is **+0.00004** across the thirteen heads, which is the
mean-zero property surviving contact with the data. The largest is `fg3a` at **0.017**, the
head where it absorbs drift rather than shock.

It is also a **defined check rather than a hopeful comparison**: `sigma_year` is fitted by
NUTS on player-season rows and knows nothing about `make season-effects`, which measures the
league rate directly as totals over totals. On a log link the two are the same number in the
same units, and **5 of the 8 count heads land within 20%** —

| head | fitted `sigma_year` | measured league yoy sd | ratio |
|---|---|---|---|
| `blk` | 3.66% | 3.70% | **0.99** |
| `tov` | 2.71% | 2.89% | 0.94 |
| `fta` | 3.78% | 4.36% | 0.87 |
| `stl` | 2.49% | 3.03% | 0.82 |
| `reb` | 1.75% | 1.48% | 1.18 |
| `fg2a` | 3.03% | 2.42% | 1.25 |
| `ast` | 4.46% | 3.08% | 1.45 |
| **`fg3a`** | **15.43%** | 6.53% | **2.36** |

`fg3a` at 2.36× is the tell: with no trend term to carry the secular climb, the year effect
absorbs **drift as a sequence of shocks** and then zeroes it at prediction time — which is
why its bias goes to −9.01%. The logit-link rows (`gp` 4.47×, the conversions ~3.6×) differ
by a 1/(1−p) factor and are a sanity check only.

**The Jensen inflation is real and negligible**: mean-zero on the linear predictor is *not*
mean-zero on the response, since `E[exp(σz)] = exp(σ²/2)`, and the measured multiplier runs
**1.000 to 1.012** across the heads. Worth reporting rather than assuming away; not worth
correcting.

#### 5. What the year effect IS worth — and the plan understated it by another order of magnitude

Roster season-total dk_pts spread, sd of the summed total across posterior draws, averaged
over 200 random rosters:

| roster | `base` sd | `year` sd | inflation | shared-β, same board |
|---|---|---|---|---|
| 12 | 753 | 866 | **+8.9%** | +0.2% |
| **15** | **557** | **621** | **+11.6%** | **+0.2%** |
| 30 | 1,213 | 1,671 | +22.1% | +0.3% |
| 150 | 2,796 | 6,652 | +91.1% | +1.1% |
| 791 (whole board) | 7,098 | 32,967 | **+278%** | +6.4% |

**On a 15-man roster the year effect is worth ~95× the shared-β term** the Stan work was
built for. The plan's "larger by an order of magnitude" is understated by a further order.
The mechanism is the one the plan named: a league shift is perfectly correlated across
players, so it grows as **N** while independent error grows as **sqrt(N)** — and unlike
shared-β, which is negligible on one roster and only matters board-wide, this one is already
+15% at twelve players.

⚠️ **Treat these as an upper bound.** The fitted `sigma_year` exceeds the independently
measured league movement on `fg3a`, `ast` and `fg2a`, so part of this spread is player-level
heterogeneity the year effect has absorbed rather than league movement.

#### 6. Minutes is the one head that adopts a season term — and it falsifies a recorded hypothesis

| `min` arm | val CRPS | test CRPS | bias (minutes) |
|---|---|---|---|
| `carry_forward` | 161.45 | 168.24 | −5.69 |
| `base` | 144.09 | 147.02 | −41.05 |
| `trend` | 144.44 | 148.78 | **−56.59** |
| **`year`** (selected) | **143.81** | **146.54** | **−38.19** |
| `trend_year` | 144.28 | 148.85 | −56.63 |

The year effect wins on **both** splits — the replication bar this repo insists on — and
nudges the standing −33 to −41 minute bias to −38.2.

**And the trend settles an open question in `docs/availability-plan.md`.** That doc proposed
the minutes head's held-out bias as "the signature an era effect would leave". Adding a trend
makes the bias **worse by 15.5 minutes** (−41.0 → −56.6) on both arms that carry one. So the
minutes bias is **shrinkage toward a 30-season mean, not an era effect** — the hypothesis is
falsified rather than left open.

#### 7. Availability: the season × role interaction is a validation null

`docs/availability-plan.md` finds the availability era effect is role-graded (−0.101 of
games-played share for heavy-minute players against −0.037 for fringe), and
`season_effects_regimes.csv` now confirms it as a dated policy step — the 2023-24
Participation Policy is a **−4.63%** level break on `gp_share [30+ mpg]` at **p = 0.008**
while the fringe bucket is +5.37% and not significant. So the era effect is real. It does not
transfer:

| arm | features | **val CRPS** | test CRPS | bias |
|---|---|---|---|---|
| `carry_forward` (league/age) | 0 | 13.387 | 13.614 | +6.80 |
| `base` | 19 | 10.007 | 10.797 | +1.68 |
| **`trend`** (selected) | 20 | **9.997** | 10.765 | −1.29 |
| `year` | 19 | 10.015 | 10.813 | +2.20 |
| `trend_year` | 20 | 10.005 | 10.757 | −1.24 |
| `trend_x_role` | 26 | **10.078** | **10.742** | −1.23 |
| `trend_x_role_year` | 26 | **10.087** | **10.736** | −1.27 |

**The two role arms are the best two on test and the worst two on validation.** That is the
exact false-positive shape this repo has already shipped once — the nonlinearity arm whose
paired bootstrap on test read [−0.079, −0.015] with P(Δ<0) = 99.7% and did not replicate. It
is caught here because selection never reads the test column. And the arm that *is* selected,
`trend`, is worth **0.010 games of CRPS** on validation, which is nothing.

#### 8. Bonus-threshold calibration

| arm | predicted / game | realized / game | bias |
|---|---|---|---|
| `base` | 0.1154 | 0.1391 | **−14.8%** |
| `trend` | 0.1216 | 0.1391 | −10.2% |
| `year` | 0.1127 | 0.1391 | −16.5% |
| `trend_year` | 0.1222 | 0.1391 | −9.9% |

Every arm under-predicts the bonus by 12–19%, and **that level is mostly not the season
term** — this composition draws the eleven heads independently given realized minutes, so it
omits the positive cross-component dependence the bonus threshold needs (measured
off-diagonals average +0.0225 across the seven counts). It is a standing argument for the
residual copula, not against a season term. The *ordering* tracks the bias story exactly.

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
player-seasons, held out on 2024-25/2025-26. Three results bind on everything below:

- **`carry_forward` — prior per-36 rate x actual minutes / 36, no fitting at all — scores
  held-out R² 0.82–0.94**, and the best of seven fitted variants beats it by only **+0.0019 to
  +0.0203**. Every component head must be quoted against it; `beats_floor` is on every output
  row. Combined with `oracle_gp` 221.3 vs `oracle_rate` 302.7 on the season total, this says
  remaining value is in **availability and the joint structure**, not in richer rate features.
- **The specification is scale, not curvature**: `log E[rate] = β·log(prior rate)`. Linear-in-
  raw-rate inside `exp()` is misspecified, catastrophically for the zero-heavy heads (`fg3a`
  0.520, `blk` 0.637 against 0.879 / 0.820 on the log scale). Splines add +0.030 (`fg3a`) and
  +0.040 (`blk`) and ≤ +0.003 elsewhere; `age × own` and `mpg × own` interactions are a null.
- **Walk-forward PCA of the 156-column season matrix is a null.** Refitted per target season on
  S-1 and earlier only (a pooled basis leaks the future invisibly), 10 components replacing the
  raw context columns: within ±0.003 of the raw spec on every head. The style and tracking
  families add nothing once you have the player's own prior rate and his minutes. Use the cheap
  raw spec.
- **`ftm|fta` is the one head nothing beats** — free-throw percentage is pure player skill, so
  an empirical-Bayes shrink of the prior is already optimal. The conversion side is worth
  ~1–2% of its NLL at most (`fg2m|fg2a` +0.072, `fg3m|fg3a` +0.032).

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
> **Availability** ports the validated point MLE and reproduces it: held-out CRPS 10.7947
> against 10.7952, ρ 0.2759 against 0.2757, and the MLE inside the 95% credible interval for
> **21/21** coefficients. The prior is set to `normal(0, 1/sqrt(2·l2))` precisely so the
> posterior *mode* is the penalized MLE, making that a defined check. R̂ 1.0025, 0
> divergences, 254 s. What the posterior adds is `Var_θ(Σ_i E[Y_i|θ])` — exactly 0 for any
> point estimate — but **its size depends on the portfolio**, and this plan's framing
> oversold it. The independent term grows as sqrt(N) and the shared-β term as N, so measured
> on the held-out board the spread inflation is **+0.2% on a 15-player roster** and **+6.4%
> across all 911**. Real for board-wide exposure across many lineups; near-irrelevant for one
> drafted team. This matters for the "joint / correlation modeling across teammates" section
> below: shared *parameter* uncertainty is not the correlation source a single roster needs —
> shared **team state** and the shared `min` draw still are.
>
> **Minutes** is new and is the first head to use the real trials denominator — successes
> out of actual game length, never 48. See `docs/availability-plan.md` for the sweep; it
> clears its no-fit floor by +0.041 R² and −21.4 minutes of CRPS, and it reports the
> **game-level** dispersion (4.65× binomial) separately from the season-level ρ the collapse
> estimates, because the simulator needs the former and the fit only sees the latter.
>
> **The eleven component heads** are built too — 74 fits, **0 divergences**, max R̂ 1.0118,
> 305.0 min of compute. 10,194 player-seasons, 9,403 train / 791 test, validation on 2022-23
> and 2023-24. Three findings, two of which **overturn what the sklearn run above measured**:
>
> - **`log(own)` alone is not sufficient under a negative binomial.** The Poisson fit has
>   `log_own` at 0.8204 (`blk`) and 0.8791 (`fg3a`); under NB the identical spec collapses to
>   **0.6794** and **0.3719**, both far below their floors, and only a spline recovers them
>   (0.8579, 0.9046). NB2's `var = μ + μ²/φ` down-weights large counts, so the fit is driven
>   by the low-count mass — exactly where the log-scale relation is most curved. **The
>   "splines are worth ≤ +0.003 outside `fg3a`/`blk`" guidance above is Poisson-specific**;
>   under NB the validation split picks a spline for **four** heads (`fg3a`, `blk`, `ast`,
>   `stl`), and on two of them it is the difference between a model and a failure.
> - **`linear` is worse than the sklearn run suggested** — held-out R² **−19.00** on `fg3a`
>   and **−1.393** on `blk`, against 0.520/0.638. Linear-in-raw-rate inside `exp()` is not
>   merely misspecified, it is unusable.
> - **⭐ Adopting the shot-attempt basis (2026-08-04) retired the −19.00 case entirely.**
>   `fg3a` is no longer a count head; `fga` replaces it and is the best-behaved count in the
>   project — floor **0.9464**, the highest of the seven, selected at **0.9505**, and
>   `log_own` already at **0.9501**. Where the wrong scale cost `fg3a` a catastrophic
>   −19.00, it costs `fga` **0.9396**, i.e. 0.0068 R². A total is far less skewed than its
>   three-point part. `blk` at −1.393 is now the only spectacular linear failure left, and
>   the `fg3a` figures above are retained as the record of the basis that was retired.
> - **`fta` now joins `ftm|fta` below its floor** (best fitted 0.8649 against 0.8673), so the
>   whole free-throw family fails. Unlike `ftm|fta` there is no "pure player skill" argument
>   for trips to the line, so this deserves a second look rather than acceptance.
>
> **✅ The 3PA/2PA reparameterization is settled, and it wins decisively.** `fga` as a count ×
> `fg3a | fga` as a binomial share beats two independent count heads by **−0.771 nats on
> validation and −0.793 on test**, per player-season, on the joint density of `(fg2a, fg3a)`
> (10.797 → 10.026; 10.784 → 9.991), replicating on both splits. The comparison is legitimate
> because `(fg2a, fg3a) ↔ (fga, fg3a)` is a **bijection with unit Jacobian on the integers**,
> so the two joint log-densities are directly comparable. The recommendation below is now a
> measurement.
>
> Every head is quoted against `carry_forward`, and every variant is selected on a
> validation split with the test column reported for confirmation only.

**Fit the heads separately.** The chain of conditionals factorizes the joint posterior exactly
when parameter blocks are distinct, so separate fits are not an approximation — they recover the
identical posterior. Full argument in `CLAUDE.md`; the short version is that `megamodel.stan`
shared no parameter between any two heads, so its joint fit bought nothing and cost 99% of the
data. Correlation for the simulator comes from (1) a shared `min` draw, then (2) a residual
copula if needed — measured off-diagonals average **+0.0225** across the seven counts, max **+0.1329**. Go joint only for a
correlated multivariate player effect, and only after measuring it is worth it (the player random
effect on rates was largely in-sample leakage). Reparameterize the 3PA/2PA substitution as
`fga` count × `fg3a | fga` share rather than coupling two Poissons — ✅ **measured 2026-07-30
and worth −0.79 nats per player-season**, see the built-block above.

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
  per-game minutes subsection, where the same head goes from 9,048 rows to 731,863.
- ~~**Whether minutes should be fitted per-game at all is open and deliberately deferred**~~
  — **partly answered 2026-07-31.** Of the three options costed under "Fitting strategy",
  the **team-game composition model is built and wins** (`make stan-composition`,
  `docs/minutes-composition-plan.md`): −0.3548 minutes of held-out CRPS against the
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
