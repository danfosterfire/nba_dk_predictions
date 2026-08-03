# NBA Best Ball Points Prediction

Predict a player's performance in a popular NBA best-ball fantasy contest (`dk_pts`) for
**each game** of an upcoming season, using only information available before the
season starts.

A chain of Bayesian component models — availability, minutes-as-team-composition, and the
box-score components of `dk_pts` — is fitted in Stan on 30 seasons of NBA box scores. Those
fits are built; the layers they exist to serve are not yet. Drawing whole seasons from their
joint posterior, ranking players from those draws, and backtesting drafting strategies under
real contest rules are the next three stages, and are labelled as planned below.

This file is the overview. `CLAUDE.md` is the working reference for conventions and measured
facts; `docs/*-plan.md` hold the detailed designs.

---

## 1. Introduction

### The target

Scoring is a linear function of the box score, plus a nonlinear bonus:

```
dk_pts = 1.0·PTS + 0.5·FG3M + 1.25·REB + 1.5·AST + 2.0·STL + 2.0·BLK − 0.5·TOV + bonus
bonus  = 1.5 for a double-double, 4.5 for a triple-double
```

Implemented once, in [src/data/preprocess.py](src/data/preprocess.py) (`compute_dk_pts`).

Full contest rules are in [docs/dk_best_ball_rules.md](docs/dk_best_ball_rules.md): 16
players drafted in a snake draft, frozen for the season, with the best 7 by slot scoring
each week and four elimination rounds ending 4/4.

### The prediction-time constraint

This is the defining constraint of the project. Before the season starts we know the
**schedule**, the **season-start rosters** (definitively which team each player is on), and
every team's and player's **previous-season** statistics. We do *not* know within-season
trades, current-season minutes, injuries, or form.

The information set is therefore a cross-season join: **current-season roster membership ×
prior-season statistics**. Two consequences shape everything downstream. Minutes weights in
any roster aggregate must come from season S−1, since season-S minutes are unknown. And
every feature except the schedule-derived ones is constant within a player-season — the only
per-game variation available is opponent, home/away and rest.

### Where the variance actually is

Measured on 254,167 player-games (2014-15 → 2023-24) by `make variance-budget`
([src/eda/variance_budget.py](src/eda/variance_budget.py)). Within-player-season residual sd
is **9.445** dk_pts against a total sd of 14.566.

| Source | Share of variance | basis |
|---|---|---|
| Player-season identity | **57.96%** | of total per-game variance |
| **Own minutes played** | **46.40%** — *unknowable in advance* | of within-player residual |
| Opponent × season | 0.691% | of within-player residual |
| Home / away | 0.034% | of within-player residual |

Never quote a row without its basis: identity is 58% *of the total*, everything else is a
share of the remaining 42%. The opponent rows are in-sample ANOVAs on contemporaneous
opponent identity — ceilings, not achievable gains.

**So roughly 90% of attainable skill is getting each player's season-level rate right, and
1–2% is per-game modulation.** That budget is why this project spends its effort on
availability, minutes and component rates rather than on matchup features, and why the
deliverable is a *distribution* rather than a sharper point estimate.

### This project

This is the AI-assisted version 2 of a
[hand-coded effort using the R stack](https://github.com/danfosterfire/nba_stats). It

1. replaces an R/tidyverse stack with a Python one, for fun and learning;
2. refines the per-box-score-component Bayesian models (Stan);
3. adds simulated drafts and seasons, to use the full posteriors and test drafting
   strategies rather than learning them live;
4. communicates process and findings through a Streamlit dashboard.

---

## 2. Methods

### Data

30 seasons (1996-97 → 2025-26), ~380 MB of raw CSVs pulled from `nba_api` by
[src/data/fetch.py](src/data/fetch.py) and cleaned by
[src/data/preprocess.py](src/data/preprocess.py) into 731,906 regular-season player-games.
Four capture programs feed the availability side: the 20-season box-score inactive/DNP
backfill ([src/data/boxscore_status.py](src/data/boxscore_status.py), 2006-07+), the NBA's
daily injury-report PDFs ([src/data/injury_reports.py](src/data/injury_reports.py)), the
ESPN status feed ([src/data/injuries.py](src/data/injuries.py)), and DraftKings /
FantasyPros ADP boards ([src/data/adp_draftkings.py](src/data/adp_draftkings.py),
[src/data/adp_fantasypros.py](src/data/adp_fantasypros.py)).

Three of those sources are **not backfillable** and are on a deadline: the injury-report
PDFs age out of the CDN after ~7 months, the ESPN feed has no history, and the DraftKings
draft board is login-gated with zero Wayback presence and open only in October. `make
daily-capture` must stay on a cron; see [docs/adp-plan.md](docs/adp-plan.md).

Season-level features are assembled by [src/eda/season_matrix.py](src/eda/season_matrix.py)
into **Tier A** (30 seasons, 10,900 qualified player-seasons × 150 box-score columns) and
**Tier B** (13 seasons, adding tracking and hustle families, a strict column superset).
Each has an unfiltered twin (`season_matrix_roster_*`) for roster aggregation, because the
PCA's `GP≥20 & MIN≥10` filter drops real teammates who consume real minutes.

The full season-level EDA pipeline runs as `make eda`; each stage is one module in
[src/eda/](src/eda/) writing one artifact.

### The output contract — twelve components, and `dk_pts` falls out

**`dk_pts` is deterministic given the components and is never predicted directly.** Twelve
quantities are modelled per player-game, built by
[src/features/targets.py](src/features/targets.py):

| Component | Likelihood | Exposure / trials |
|---|---|---|
| `min` (given availability) | successes / trials | **game length** — 48 in regulation; 53/58/… in OT |
| `fg2a` `fg3a` `fta` `reb` `ast` `stl` `blk` `tov` | count (negative binomial) | `min` |
| `fg2m` `fg3m` `ftm` | successes / trials | `fg2a` / `fg3a` / `fta` |

Only eight of the twelve reach the scoring function; `min` and the three attempt counts
matter solely through the exposure and trials they supply. Reassembly is exact and linear:
`pts = 2·fg2m + 3·fg3m + ftm`, then `compute_dk_pts`.

> **The two shot-attempt counts are the one row of this table known to be beatable.** A
> three-point attempt *substitutes* for a two, so fitting `fg2a` and `fg3a` as independent
> counts leaves that dependence for the residual copula to carry. A reparameterization that
> removes it by construction has been measured and wins — see *Measured but not adopted*
> below. **The table describes what is fitted today, not what scored best.**

Two things force this decomposition rather than a single `dk_pts` head. First, the
double-double bonus is a **simultaneous threshold** on five components, so
`E[bonus] ≠ bonus(E[x])` and the deliverable is a *joint draw*, not twelve marginals.
Second, context effects **cancel across components** in the DK sum — `teammate_assist_supply`
moves 2.106 dk_pts of gross component movement into −0.254 net, an 8.30× cancellation
(`make context-value`), and the opponent main effect cancels 1.98× (`make opponent`). A
single `dk_pts` head fits the residue.

The trials denominator for minutes is derived rather than assumed:
[src/features/game_length.py](src/features/game_length.py) recovers every game's length from
summed team minutes ÷ 5 — validated by the two teams being independent estimates that
disagree on **0 of 37,986 games**. 5.93% of games go to overtime, so truncating at 48 would
censor the top of the minutes distribution in exactly the games where stars play most.

### The chain, and why the heads are fitted separately

The generative structure is a chain of conditionals:

```
availability → min | available → counts | min → makes | attempts
```

With disjoint parameter blocks and independent priors the joint posterior **factorizes
exactly**, so separate fits recover the identical posterior a single joint model would.
This is an identity, not an approximation, and it is what makes full Bayesian inference
affordable here — the prior attempt's `megamodel.stan` paid the joint-fit price for a
coupling that was not in the model and had to run on 1% of the data.

A second identity does the rest of the work: for `y_g ~ Poisson(m_g·e^η)` with η constant
within a player-season, the product over games factors into a Poisson on the season total
times a multinomial free of η. So ~10,000 player-season rows recover the coefficients of
731,906 player-game rows.

The correlation the simulator needs therefore enters at **draw time**, not fit time: one
shared `min` draw per player-game pushed through all eleven component heads as exposure
(minutes being 46.4% of within-player residual variance), plus a Gaussian copula on the
remainder. [src/eda/residual_correlation.py](src/eda/residual_correlation.py) writes that
matrix, conditioned on minutes, as an explicit simulator input.

### The fitted heads — what ships today

Three `.stan` sources in [src/stan/](src/stan/) serve every head, which is the
factorization argument as code:

| Source | Serves | Driver | In `make stan` |
|---|---|---|---|
| `betabinomial_glm.stan` | availability, minutes, the 3 conversion heads | [stan_availability.py](src/models/stan_availability.py), [stan_minutes.py](src/models/stan_minutes.py), [stan_components.py](src/models/stan_components.py) | **yes** |
| `negbinomial_glm.stan` | the 8 count heads | [stan_components.py](src/models/stan_components.py) | **yes** |
| `composition_glm.stan` | the team-game minutes allocation | [stan_composition.py](src/models/stan_composition.py) | **no — pilot** |

Shared plumbing — compilation, sampling, and diagnostic extraction into a CSV rather than a
scrollback buffer — is in [src/models/stan_utils.py](src/models/stan_utils.py).
`make stan` runs exactly the three heads that ship: `stan-availability`, `stan-minutes`,
`stan-components`. `make stan-composition` is a separate target, and that separation is
deliberate rather than an oversight — see the next section.

**Availability** ([src/models/availability.py](src/models/availability.py) is the point-MLE
reference, [stan_availability.py](src/models/stan_availability.py) the Bayesian port) is a
beta-binomial GLM over games played out of team games. Games played is the largest lever on
the season total and the least persistent quantity in the project, so the head shrinks hard
toward a league/age baseline and emits a distribution.

**Minutes** in the shipped chain is [stan_minutes.py](src/models/stan_minutes.py): the
marginal `min | available`, fitted season-collapsed as successes out of real game length,
selecting `logit(own) + spline`. It supplies two of the three numbers the simulator needs —
the season-level mean, and separately the **game-level** dispersion, which a season total
cannot identify on its own. A second, richer minutes model exists and is not in the chain;
it is the first entry in the next section.

**Components** ([stan_components.py](src/models/stan_components.py)) fits the eight negative
binomial counts and three beta-binomial conversions, each against a mandatory no-fit floor.
Its head lists come from [component_rates.py](src/models/component_rates.py) — `COUNT_HEADS`
still fits `fg2a` and `fg3a` independently, while `substitution_arm` fits the reparameterized
alternative beside them as an ablation and beats it, as noted under the output contract.

**Season terms** ([src/models/season_terms.py](src/models/season_terms.py), `make
season-terms`) is a 108-fit ablation asking whether any head needs a year trend or a year
random effect to track league-wide era movement. [src/eda/season_effects.py](src/eda/season_effects.py)
measures the league series it would be correcting for.

### Measured but not adopted

**Two results have beaten the shipped specification and are still not in it.** Both are easy
to misread as settled — the measurement is done, the code exists, the numbers are quoted in
Results — and both have been misread that way in this repo's own documentation. The table in
the previous section and the contents of `make stan` are the authority on what runs; a
result appearing in Results is not.

| Result | Margin | What exists today | Blocking step |
|---|---|---|---|
| `fga` count × `fg3a \| fga` share, replacing two independent attempt counts | −0.500782 val / −0.493549 test nats per player-season | an ablation arm beside the shipped heads, and a re-measurement | swap `COUNT_HEADS`, per `docs/shot-attempt-basis-plan.md` |
| Team-game minutes composition, replacing independent per-game draws | −0.406 min test CRPS, and exact team totals | a pilot fitted on 2018-19 onward | Gate E — the full-window refit |

**The shot-attempt reparameterization.** `component_rates.COUNT_HEADS` still lists `fg2a`
and `fg3a` as two independent negative binomial counts, and that list is what `sweep_counts`
iterates. The re-measurement this row used to be blocked on is **done** — `make
stan-substitution`, 16 fits, [docs/shot-attempt-basis-plan.md](docs/shot-attempt-basis-plan.md)
— and the result survives it. ⚠️ **The previously recorded −0.771 / −0.793 was measured
against a handicapped comparison**: `substitution_arm` fits every head at the `log_own`
variant, but `fg3a`'s shipped spec is `log_own_spline`, and at `log_own` that head reads test
R² **0.3719** with `beats_floor = False` against **0.9046** for the spline it actually
selects. Fitting arm A at each head's own selected variant and sweeping arm B for real still
gives **−0.493549** on test, and **−0.491910** against arm A's *best-of-16* configuration.

⭐ **The re-measurement produced a sharper result than the one it was checking.** Compared at
their **no-fit floors** — no features anywhere — the two bases score **11.024027** against
**10.085599**. Arm B's floor beats arm A's *best fitted* configuration by **−0.390814**, ~79%
of the total margin, while arm B's own fitting adds only **−0.101096** on top of its floor.
This is not a better model of shot attempts; it is the same information written in
coordinates where the dependence is structural instead of residual.

**The minutes composition.** [stan_composition.py](src/models/stan_composition.py) allocates
each team-game's `5 × game_length` minutes among the players who played by decomposing the
multinomial into sequential binomial trials, ordered by prior-season minutes share, with the
per-player cap enforced through the trials (`m_k = min(U, R_k)`) rather than checked
afterwards. It gets both per-game constraints — the exact team total and the per-player cap
— where the shipped marginal head gets only the cap, and it is the only head that represents
teammate-absence redistribution. Its own `independent_comparator` names `stan_minutes` as
"the plan of record being compared against", and it imports that head rather than replacing
it. Gates A–D passed on the pilot; **Gate E, the full-window refit, is open and not taken** —
costed at ~10–12 h for the 736k-row sweep. Note that promotion would not retire
`stan_minutes`: the composition head produces neither the season-level mean nor the
game-level dispersion the simulator also needs.

Neither is recorded as a decision reversal, because neither has been reversed — they are
measurements the build has not yet absorbed. `docs/minutes-composition-plan.md` holds the
gate definitions; `CLAUDE.md` holds both results in full.

### Simulation — planned

Not built. [docs/simulations-plan.md](docs/simulations-plan.md) holds the specification,
which is pinned by measurements rather than guesses:

- **Draw, never plug in.** `E[min]` and `E[gp]` are wrong inputs to a threshold bonus.
- **Three minutes numbers, which compose rather than substitute**: the season-level mean
  from the minutes head, the **game-level** dispersion (4.65× binomial, measured separately
  because a season total cannot separate per-game from per-season noise), and the **2.43×**
  ten-game block variance inflation from `make serial-correlation` for serial dependence
  between games.
- **Sequential structure goes on minutes and nowhere else.** There is no shooting hot hand:
  both field-goal conversion heads are measured nulls, so those eleven heads stay collapsed.
- **Bonus overdispersion is unit-specific** — 0.10 at the season unit, **0.025** at the
  player-game unit the simulator will actually draw at
  ([src/features/targets.py](src/features/targets.py), `make component-targets`).
- **Availability absences are a mixture, not a Markov chain.** A constant hazard matches the
  mean spell length and misses both tails, so a 2-component or semi-Markov process is needed
  for the season-total joint distribution.

### Ranking, drafting and tournaments — planned

Also not built, and the reason the simulation layer exists. Simulated seasons produce a
posterior over each player's season trajectory; that becomes a draft ranking, which is
executed in a simulated snake draft against an ADP-based opponent model, and the resulting
portfolios are replayed against realized box scores under real tournament structures.

The pieces already in place are the market side and the contest economics.
[src/features/adp.py](src/features/adp.py) builds a point-in-time-safe ADP panel — ADP is a
forecast of the same target, so it is the most leakage-prone input in the repo and gets
three separate dates per row. [dashboard/economics.py](dashboard/economics.py) derives the
tournament structures from `data/raw/dk_best_ball_tournament_*.csv`: five real tournaments,
Round 1 a zero-consolation knockout in every one, and rake expressed as a break-even edge
hurdle (+10.45% to +17.60%) because that is the unit a measured edge can be compared in.

ADP belongs in *this* layer, not in the prediction layer — that is a settled decision, so the
blend weight sweeps a real axis rather than a weight the model already absorbed.

### What was tried and deprioritized

An LSTM/Transformer trunk over prior-season game logs ([src/models/lstm.py](src/models/lstm.py),
[transformer.py](src/models/transformer.py), [src/train.py](src/train.py)) is retained but
deprioritized. Held out, the prior-season game *sequence* is worth **+0.0059 R²** over four
season aggregates, and +0.0074 above its own shuffled-order null — the trunk is mostly
re-deriving a season mean the season matrix already holds.

---

## 3. Results

Headlines only. Every figure is reproduced by the `make` target named beside it and lands in
`outputs/`. **This file is audited**: `make docs-audit` re-derives each quoted figure from
its artifact and **exits non-zero** on disagreement, so a headline copied here and never
refreshed fails the build rather than quietly misleading.

**The availability head is the largest measured win.** `make availability-model` /
`make season-total`. Held out on 2024-25 and 2025-26, scored by CRPS in games, the
beta-binomial GLM reads **10.795** against a GBM's 10.888, ridge's 10.896 and a league/age
baseline's 13.614 — gradient boosting does not beat a 19-feature GLM. On the actual
deliverable it is worth **−211 dk_pts of season-total MAE** against assuming a full season
(646.3 → 435.1), with bias falling from +541.9 to +6.1. The oracles settle which half of the
error dominates: perfect games played gives MAE 221.3 against perfect rate's 302.7.

**The component rate side is nearly saturated from prior-season information alone.**
`make component-rates` / `make stan-components`. A no-fit floor — prior per-36 rate × actual
minutes, no fitting — scores held-out R² **0.82–0.94**, and the best fitted head beats it by
+0.0019 to +0.0228. Every head is quoted against that floor; `fta` and `ftm|fta` do not clear
it at all, so the whole free-throw family currently fails.

**The specification that matters is scale, not curvature — except where the likelihood
changes the answer.** Putting the player's own prior rate in on the log scale is worth
almost everything; linear-in-raw-rate inside `exp()` is unusable (`fg3a` held-out R²
**−19.00**). Under the negative binomial, though, splines are not a refinement but the
difference between a model and a failure on the skewed heads (`blk` 0.679 → 0.858), which
reverses what the Poisson fits implied.

**The minutes composition beats the independent draw on the independent draw's own metric —
on a pilot window, and it is not yet shipped.** `make stan-composition`. Held out, test CRPS
**4.5078** against the independent comparator's 4.9140 (−8.3%), while also hitting the team
total exactly where the independent draw misses by 36.87 minutes per team-game. The plan
predicted a wash and budgeted for arguing on capability instead. Two sub-results: the pure
binomial decomposition is *worse* than the no-fit floor (PIT KS 0.194) — dispersion is the
difference between a model and a failure again — and dispersion is genuinely role-graded,
fitted at 0.148 for fringe players against 0.061 for stars, a 2.41× spread that cuts
calibration error by 59%. See *Measured but not adopted*.

**The 3PA/2PA substitution is best handled by reparameterization — re-measured
un-handicapped, and still not shipped.** `make stan-substitution`. Modelling `fga` as a count
and `fg3a | fga` as a beta-binomial share on `fga` trials beats two independent count heads
by **−0.493549 nats** per player-season on test and −0.500782 on validation, with each head
fitted at its own selected variant and both arms swept — a legitimate comparison because the
coordinate change is a bijection with unit Jacobian. The originally recorded −0.793 / −0.771
had both arms pinned at `log_own`, where `fg3a` scores test R² **0.3719** against **0.9046**
for the spline it actually selects, so the canonical arm was handicapped; removing the
handicap costs 0.306 nats of the margin and the result survives anyway. **The strongest
version is that the basis beats the model**: the reparameterized *no-fit floor* beats the
canonical basis's *best fitted* configuration by **−0.390814**. See *Measured but not
adopted*.

**No head ships a season term, and the ceiling on ever needing one is ~3% of MAE.** `make
season-terms`. An oracle that rescales each held-out season by its own realized league total
— the ceiling on any trend, year effect or manual override — is worth a median **1.32%** of
base MAE across heads. A trend worsens held-out bias on 6 of 8 count heads, most sharply on
`fg3a`, the one quantity whose league series most looked like it wanted one. The minutes
head is the single exception and adopts a year effect. What a year effect *is* worth is
joint spread: **+19.0%** on a 15-man roster's season-total sd, against +0.2% from shared
coefficient uncertainty.

**The sampler behaved.** 74 component fits with 0 divergences, 108 season-term fits with 0
divergences and 0 treedepth saturation, and the availability port reproduces the point MLE
with the MLE inside the 95% credible interval for 21 of 21 terms. Cost is concentrated
entirely in the spline variants. 725 tests pass (`.venv/bin/pytest tests/`).

---

## 4. Discussion

The variance budget in §1 says the season-level rate is ~90% of attainable skill, and the
results bear that out from an uncomfortable direction: the rate side is close to saturated
by a no-fit carry-forward, while availability — the largest lever on the season total — is
the *least* persistent quantity measured here (r = 0.317 year over year). The oracle
comparison makes this concrete. Perfect knowledge of games played is worth more than perfect
knowledge of the rate. Effort spent on richer rate features is spent against a floor that is
already 0.82–0.94 R²; effort spent on the availability distribution and on the joint
structure between components is not.

That is why the deliverable is a joint draw rather than a set of marginals, and it is where
the modelling and the contest meet. The bonus is a simultaneous threshold on five
components; a best-ball lineup is a max over 16 players each week; and a tournament payout is
convex to the point of being a zero-consolation knockout in Round 1. All three are functions
of the *shape* of the distribution, not of its mean. A model that improves marginal CRPS by
1% and gets the correlation structure wrong is worth less here than one that does the
reverse — which is the argument for the Bayesian posterior (every player shares
coefficients, so one draw moves the whole board together) and for the composition head (a
team's minutes are zero-sum, so a teammate's absence has to land somewhere).

The honest caveat is unchanged and worth restating. **Five games of the current season
settle 86% of the season total** (R² 0.859, `make target-profile`), against +0.0086 R² for
the entire team-context block. The prior-season-only constraint costs roughly two orders of
magnitude more than any feature in this repo can buy back. It is a product requirement — the
draft happens before the season — and everything here is built for that frame, but it sets
the ceiling.

Known limits, logged so they are not rediscovered: mid-season trades are not modelled (13.6%
of players appeared for 2+ teams in 2023-24); 14.7% of season-start roster minutes have no
usable prior-season row and are imputed rather than dropped; and the free-throw family does
not currently clear its floor.

---

## 5. Repository map

```
src/data/       fetch, preprocess, availability capture (box-score status, injury
                reports, ESPN feed), ADP capture
src/features/   component targets, game length, team context, opponent, availability, ADP
src/eda/        the season-level analysis pipeline — one module per artifact
src/models/     the Stan heads (availability, minutes, composition, components,
                season terms) plus the sklearn references they are checked against
src/stan/       three .stan sources for eleven-plus heads
dashboard/      nine-tab project walkthrough; reads artifacts only, never refits
docs/           plan docs — predictions, availability, minutes composition, ADP,
                simulations, EDA, provenance, dashboard, contest rules
```

## 6. Quick start

```bash
make install          # create .venv and install requirements
make fetch            # pull raw data from nba_api (long)
make eda              # the full season-level EDA sweep
make stan             # fit the availability, minutes and component heads
make dashboard        # Streamlit walkthrough at http://localhost:8501
make test             # pytest
```

The Stan heads need a CmdStan toolchain, which pip does not manage:

```bash
.venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Always use `.venv`, never the system Python. Two guards run over the documentation itself:
`make docs-audit` re-derives every quoted figure in the plan docs from its artifact and
fails on a mismatch, and `make dashboard-audit` reports drift between the decision registry
and the docs it distills. See `CLAUDE.md` for conventions.
