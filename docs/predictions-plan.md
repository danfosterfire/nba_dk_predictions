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
  is worth ~205 dk_pts of season-total MAE (`src/models/availability.py`, `docs/availability-plan.md`).
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
| **`min`** | 0.279 | −0.017 | **+0.297** | **2.43×** |
| `fg3a` | 0.061 | −0.016 | +0.077 | 1.46× |
| `fg2a` | 0.058 | −0.016 | +0.074 | 1.45× |
| `ast` | 0.023 | −0.017 | +0.040 | 1.22× |
| `fta` | 0.013 | −0.017 | +0.029 | 1.18× |
| `reb` | 0.012 | −0.017 | +0.028 | 1.16× |
| `blk` | 0.007 | −0.017 | +0.024 | 1.13× |
| `stl` | −0.003 | −0.017 | +0.014 | 1.08× |
| `tov` | −0.007 | −0.016 | +0.009 | 1.07× |
| **`ftm\|fta`** | −0.008 | −0.023 | +0.015 | 1.11× |
| **`fg2m\|fg2a`** | −0.016 | −0.017 | **+0.001** | **1.03×** |
| **`fg3m\|fg3a`** | −0.019 | −0.019 | **−0.001** | **1.01×** |

**There is no shooting hot hand, and the answer splits exactly along the attempts/conversion
line the season-level persistence work already found.** Conversion percentages are serially
independent — `fg3m|fg3a` excess is −0.001 (z = −0.5) and `fg2m|fg2a` is +0.001 (z = 0.8), both
dead nulls on ~600k pairs. So **the specific thing worried about — the successes/trials heads —
is the one place collapsing costs nothing**, because constant-`θ`-within-season is what the data
actually looks like.

What *is* autocorrelated is the **exposure** side: minutes at 2.43× block-variance inflation,
and shot volume at ~1.45× *on top of* minutes (the count residuals already condition on actual
minutes, so this is volume persistence beyond playing time). Decay is slower than AR(1) —
minutes reads 0.279 / 0.212 / 0.170 / 0.113 at lags 1/2/3/5, where AR(1) would give 0.279 /
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
head's 9,048 rows — **81× the data**. The season spline fit took 752 s (linear: 168 s), so a
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

### The team-game composition alternative — worth costing before a per-game player model

Rather than 731,863 player-game rows, model the **team-game composition directly**: a
Dirichlet-multinomial over the `5 × game_length` minutes among the players who dressed. That
is **71,092 team-game rows** (35,546 regular-season games × 2) at ~10.3 players each, so the
same information in a tenth as many likelihood terms, and — the point — **zero-sum holds by
construction rather than as a penalty**. It is the natural formulation for the thing the
redistribution section actually wants, and it makes "who absorbs the minutes when a starter
sits" a fitted parameter instead of a hand-set rule.

> ⚠️ **It trades one exact constraint for the other, and that is the catch.** A
> Dirichlet-multinomial over 240 minutes among ~10 players enforces the team total exactly but
> does **not** bound any individual at `game_length` — nothing stops a draw allocating one
> player 100 minutes, only the fitted concentration making it rare. The current per-player
> beta-binomial is the mirror image: individual cap exact, team total only approximate.
> **Neither form gets both**, and getting both needs a constrained/truncated allocation step.
> Whichever is chosen, the other constraint has to be checked in posterior predictive rather
> than assumed.

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

### ⏰ TODO — season effects: no head carries one, and the league moves. Measured 2026-07-30

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
| **`fg3a`** | **2.97×** | **+4.07** | **0.93** | 6.96 | **trend + year effect** |
| `fta` | 1.21× | −0.55 | 0.63 | 4.34 | year effect only |
| `blk` | 1.14× | −0.14 | 0.12 | 3.69 | year effect only |
| `ast` | 1.30× | +0.73 | 0.64 | 3.05 | year effect only |
| `stl` | 1.18× | −0.08 | **0.03** | 3.03 | year effect only |
| `tov` | 1.16× | −0.33 | 0.59 | 2.89 | year effect only |
| `fg2a` | 1.32× | −0.87 | 0.86 | 2.42 | year effect only |
| `fg3m_pct` | 1.08× | +0.11 | 0.27 | 2.06 | year effect only |
| `reb` | 1.10× | +0.25 | 0.59 | 1.48 | year effect only |
| `fg2m_pct` | 1.20× | +0.61 | 0.84 | 1.46 | year effect only |
| `ftm_pct` | 1.08× | +0.19 | 0.77 | 1.06 | year effect only |
| **`minutes_share`** | 1.09× | −0.25 | 0.81 | **0.92** | year effect only |
| `gp_share` [<12 mpg] | 1.48× | −0.71 | 0.48 | **9.04** | year effect only |
| `gp_share` [12–24] | 1.30× | −0.54 | 0.64 | 4.21 | year effect only |
| `gp_share` [24–30] | 1.25× | −0.39 | 0.45 | 3.85 | year effect only |
| `gp_share` [all] | 1.24× | −0.51 | 0.71 | 2.86 | year effect only |
| `gp_share` [30+ mpg] | 1.21× | −0.49 | 0.74 | 2.84 | year effect only |

**`fg3a` is the only quantity where a trend is worth extrapolating** (R² 0.93 at +4.07%/season)
— and note its worst single year is −24.4%, the 1997-98 three-point line being moved back
after three shortened seasons. **Everything else is shock**, `fta` and `stl` most starkly:
`stl` has a trend R² of **0.03**, i.e. essentially no drift at all and 3% of pure
year-to-year noise.

**`fta` is the case that prompted this**, and it behaves exactly as refereeing would
predict — a 1.21× band with 4.3% yoy sd and swings past ±5% in 9 of 29 transitions:
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

**One piece is genuinely knowable at prediction time and should not be lumped in with the
rest.** Rule changes and points of emphasis are announced in the summer, before opening
night — the 2021-22 non-basketball-moves emphasis was public in advance. A manual
league-level override is therefore legitimate under the point-in-time discipline, unlike
anything drawn from within the season.

**⏰ TODO — assess a year-level random effect and a year-on-year trend fixed effect across
every head**: the eleven dk components, availability, and minutes. This is scoped as its own
session; the prompt is below. Delete this TODO once the ablation has run and its verdict is
recorded here.

<details>
<summary><b>Prompt for a new session</b></summary>

```
Assess whether the heads in this repo need a season term, and which kind. Read CLAUDE.md,
docs/predictions-plan.md (the "TODO — season effects" section) and docs/availability-plan.md
(the "Load management" section) first — the measurement already exists, reproducible with
`make season-effects` → outputs/eda/season_effects_*.csv. Do not re-derive it; build on it.

The established facts you are starting from:
- No head carries a season term today: not availability, not minutes, not any of the eleven
  components.
- A season FIXED effect is unusable at prediction time — there is no dummy for a season that
  has not happened. The two usable forms are a year-on-year TREND extrapolated one season
  forward, and a YEAR-LEVEL RANDOM EFFECT.
- These are complementary, not alternatives: detrending shifts the mean of the year-over-year
  changes and leaves their variance exactly unchanged. A trend fixes bias; only a year effect
  addresses spread.
- `fg3a` is the ONLY quantity where a trend is worth extrapolating (R² 0.93, +4.07%/season).
  Everything else is shock — `stl` has a trend R² of 0.03.
- The cost is measured: the no-fit floor carries −7.0% bias on `fta` (−10.7% in 2025-26) and
  +6.2% on `blk` across the held-out seasons.
- `docs/availability-plan.md` separately finds the availability era effect is ROLE-GRADED:
  heavy-minute players lost −0.101 of games-played share from 2004-2010 to 2023-2025 against
  −0.037 for fringe players. So for availability the candidate is a season × role
  interaction, not a level shift.

What to build:
1. A trend variant and a year-random-effect variant for each head, on top of the existing
   Stan specs in src/models/stan_{availability,minutes,components}.py. The year effect is a
   hierarchical term over season with a fitted sd; at prediction time it contributes mean 0
   and its variance, which is the entire point — it widens the predictive rather than
   shifting it.
2. For availability, test the season × role interaction specifically, not just a level term.
3. Quote every variant against its no-fit floor (mandatory — see CLAUDE.md) and select on a
   VALIDATION split, never on test. This repo has already shipped one false positive whose
   paired bootstrap on test read [-0.079, -0.015] with P(Δ<0) = 99.7% and did not replicate.

How to judge it, and this is the part that matters:
- A year random effect should NOT improve held-out point accuracy — it is mean-zero by
  construction. If R²/MAE moves much, something is wrong. Judge it on CALIBRATION: CRPS, PIT
  uniformity, and whether simulated season-total intervals achieve nominal coverage.
- A trend term SHOULD improve point accuracy and bias, and only on `fg3a` per the measurement
  above. If it helps everywhere, suspect overfitting to the last two seasons.
- The decisive downstream metric is season-total dk_pts and bonus-threshold calibration, not
  per-component R².

Two traps specific to this question:
- Two confounds sit inside the window: 2019-20 and 2020-21 are a COVID health-protocol
  regime, and the NBA's Player Participation Policy arrived in 2023-24. A smooth trend
  extrapolated across a policy discontinuity is actively wrong. Test for a break.
- Rule changes are ANNOUNCED before the season, so a manual league-level override is
  legitimate point-in-time information. Keep that path open rather than forcing everything
  through a fitted trend.

Conventions: python -m src.<module>, a matching Makefile target in .PHONY, cfg =
yaml.safe_load(open("configs/default.yaml")) in __main__, plain-assert tests with synthetic
builders. Stan sources in src/stan/, compiled binaries stay out of git.
```

</details>

### The rate side is nearly saturated — measured, with a mandatory floor

`make component-rates` (`src/models/component_rates.py`), season-collapsed heads on 10,194
player-seasons, held out on 2024-25/2025-26. Three results bind on everything below:

- **`carry_forward` — prior per-36 rate x actual minutes / 36, no fitting at all — scores
  held-out R² 0.82–0.94**, and the best of seven fitted variants beats it by only **+0.0019 to
  +0.0228**. Every component head must be quoted against it; `beats_floor` is on every output
  row. Combined with `oracle_gp` 221.3 vs `oracle_rate` 302.7 on the season total, this says
  remaining value is in **availability and the joint structure**, not in richer rate features.
- **The specification is scale, not curvature**: `log E[rate] = β·log(prior rate)`. Linear-in-
  raw-rate inside `exp()` is misspecified, catastrophically for the zero-heavy heads (`fg3a`
  0.520, `blk` 0.638 against 0.879 / 0.820 on the log scale). Splines add +0.030 (`fg3a`) and
  +0.041 (`blk`) and ≤ +0.003 elsewhere; `age × own` and `mpg × own` interactions are a null.
- **Walk-forward PCA of the 156-column season matrix is a null.** Refitted per target season on
  S-1 and earlier only (a pooled basis leaks the future invisibly), 10 components replacing the
  raw context columns: within ±0.003 of the raw spec on every head. The style and tracking
  families add nothing once you have the player's own prior rate and his minutes. Use the cheap
  raw spec.
- **`ftm|fta` is the one head nothing beats** — free-throw percentage is pure player skill, so
  an empirical-Bayes shrink of the prior is already optimal. The conversion side is worth
  ~1–2% of its NLL at most (`fg2m|fg2a` +0.061, `fg3m|fg3a` +0.037).

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
> 208.6 min of compute. 10,194 player-seasons, 9,403 train / 791 test, validation on 2022-23
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
copula if needed — measured off-diagonals average **+0.013**, max 0.157. Go joint only for a
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

**The team-game composition model above is the principled version of this**, and the choice
between them is a real fork: a redistribution *rule* weighted by `teammate_usage_load` is a
hand-set heuristic applied on top of independent marginals, whereas a Dirichlet-multinomial
over the team's `5 × game_length` minutes makes the same redistribution a **fitted** quantity
with zero-sum exact by construction. The heuristic is an afternoon and the composition model
is days; measure the heuristic first, since it may be most of the gain.

## ADP as a prediction input / validation signal

**See `docs/adp-plan.md`** — sourcing, storage, point-in-time rules and the integration decision
all live there, measured. This section records only what that plan concluded *for this layer*, so
the two do not drift.

**ADP does not feed the GLMM.** It stays a strategy-layer input, downstream of predictions
(`docs/simulations-plan.md`). Four reasons, in full there; the load-bearing one is that under a
knockout payout the edge *is* model-minus-market, so a model fit on ADP reproduces the benchmark
it is meant to be measured against. Coverage seals it: ADP exists for ~250 players over 12
seasons against the component heads' 10,900 player-seasons over 30.

**The one exception, and it is worth measuring.** 15.9% of roster minutes have no usable
prior-season row (9.4% true rookies, 5.4% sub-threshold, 1.1% returnees), where the model imputes
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
  recalibration of consensus onto DK's scale is worth **24.0 → 17.0 picks** of cross-validated
  error, while a position offset — the obvious next term — adds only −0.3. One anchor identifies a
  shape, not a model.

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
- **Whether minutes should be fitted per-game at all is open and deliberately deferred**
  (2026-07-30). Three options are costed under "Fitting strategy" — a per-game player model,
  a team-game composition model, and a residual-only serial process (**the recommended
  first move**). Two premises are already retired there: the 48-minute bound is *not* a
  reason to go per-game, and a single lagged-observation term reproduces only **46%** of the
  measured block-variance excess. Decide by measurement against the cheap comparator, once
  the simulator is standing.
