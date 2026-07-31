# Provenance Plan: Every Figure Gets a `make` Target

This is a planning doc, not a measurement report. It records direction; update it in place as
pieces land, the way `availability-plan.md` was.

> **Companion plan:** `docs/dashboard-plan.md`, which this one unblocks. Three of that plan's
> nine tabs are gated here.

> ## ✅ **All ten items landed 2026-07-29.** Two new modules, eight extensions, **93 new tests**
> (572 total, green), and `make eda` clean from an empty `data/features` + `outputs/eda` in
> ~9 minutes. Five new artifacts, seven extended.
>
> **Six figures did not reproduce, and every one of them turned out to be the recorded value
> that was wrong.** The plan's risk register predicted this ("a figure that cannot be
> reproduced was never checkable") and the corrections all move in the safe direction — the
> conclusions they supported hold, and two are now stronger than they were. Full list in
> [Corrections](#corrections-six-figures-that-did-not-reproduce); each is also recorded in
> `CLAUDE.md` beside the claim it changes.
>
> The single most consequential one: **own minutes is 46.4% of the within-player-season
> residual, not 18.6%.** The recorded figure conditioned on the raw minutes *level* instead of
> the within-player minutes *deviation*, which attenuates it by more than half. Minutes are a
> far larger common factor than the project believed, which strengthens the shared-`min` draw
> the simulator is built on.

## Purpose

This repo's discipline is that a claim ships with a runnable source. Most of it does — 30
`make` targets write 40-odd artifacts, and `CLAUDE.md` names the target beside almost every
number. But **roughly a dozen headline figures exist only as prose**, because they were measured
in planning-session scratch scripts that were never promoted to a module. They are not less
true; they are less checkable, and nothing recomputes them when the data underneath moves.

The dashboard forced the issue, because putting a typed constant on a screen beside a live
artifact read makes both look equally verifiable. Rather than mark the difference and live with
it, the decision taken on 2026-07-29 was to close it:

> **Every figure this project asserts is produced by a `make` target and written to an
> artifact.** Where that is not true today, the measurement gets promoted to a module before it
> goes on a dashboard.

The payoff is not presentational. Four of the ten items below are **load-bearing for modeling
regardless of the dashboard** — the residual correlation matrix is a direct simulator input,
the alpha-sensitivity rows and the match-audit ablation are permanent regression guards against
two traps that have already cost real time, and the bonus calibration is the one number
`expected_bonus` must not be changed without. Those would be worth building if the dashboard
did not exist.

---

## Scope: figures versus incident records

"Back every figure with a make target" needs a boundary, or it becomes "re-probe
`stats.nba.com` for a regression that happened on 2025-04-10." The line:

**A figure** quantifies a property of the data or a model — a variance share, a correlation, a
CRPS, a coverage rate. It is a function of files on disk plus code. It **must** be reproducible,
because if it is not, nobody can tell whether it is still true.

**An incident record** documents a one-time diagnosis of an external system at a past date —
that `BoxScoreSummaryV2` silently dropped `InactivePlayers` for 223 of 228 games in 2025-26,
that `player_stats_defense` came back empty from a poisoned server-side cache, that
prosportstransactions sits behind a Cloudflare challenge, that a macOS TCC grant cost 28 days
of ESPN snapshots. Re-deriving these is either impossible (the state has changed), pointless
(the fix is already in the code), or antisocial (it means hammering a third party).

Incident records stay as prose, carry their date and doc reference, and are rendered in the
dashboard as `incident` entries **without a live-number claim**. What makes that honest rather
than a loophole is that in every case the *defence* is reproducible even when the incident is
not: the header-only-file bug is pinned by a test, the V3 routing is pinned by a test, the three
unfetchable games are counted in a manifest on disk. The claim "this failed once" is history;
the claim "it cannot fail that way again" is tested.

Recorded here so the boundary is not re-argued each time: **incidents are the exception, and a
figure does not get to call itself an incident because measuring it would be inconvenient.**

---

## Inventory

### Already backed — no work needed

Confirmed by reading the artifacts, not by assuming. Listed because it bounds the work: the gap
is ~10 items, not the whole project.

| figures | artifact |
|---|---|
| Availability ceiling ladder, carryover gradient, GP overdispersion + histogram bins, spell distribution, the 3-year null, absence-reason decomposition, the two-window bracket, Markov/spell-shape falsification | `outputs/eda/availability_profile.csv` (234 rows, 9 measurement families) |
| Baseline CRPS ladder, PIT, workload ablation, nonlinearity ablation, minutes-nonlinearity attribution | `outputs/predictions/availability_*.csv` |
| Season-total contrast including both oracles | `season_total_metrics.csv` |
| Stan availability port verification, coefficients, diagnostics, board correlation decomposition | `stan_availability_*.csv` |
| Minutes variant ladder, the two dispersions | `stan_minutes_{metrics,dispersion}.csv` |
| Component no-fit floor and all seven fitted variants per head | `component_rate_metrics.csv` (82 rows) |
| Per-component serial correlation and block inflation | `serial_correlation.csv` |
| Game-length validation: team disagreement count, worst residual, OT rate by season | `game_length_coverage.csv` |
| Report-designation transfer function and its coverage | `report_calibration.csv`, `report_transfer.parquet` |
| ADP agreement, recalibration ladder, position bias, tier gaps | `adp_profile.csv`, `adp_transfer.parquet` |
| Persistence per column, aging curves, target dispersion, feature diagnostics, first-*k* season-total predictiveness | `persistence.csv`, `aging_curves.csv`, `target_profile.csv`, `feature_diagnostics.csv` |
| Tournament economics — rake, advance rates, payout convexity | derived live from `data/raw/dk_best_ball_tournament_*.csv` |

### The gap — ten items, all closed

| # | figures | target | module | artifact | status |
|---|---|---|---|---|---|
| 1 | The variance budget | `make variance-budget` | `src/eda/variance_budget.py` **new** | `variance_budget.csv` (25 rows) | ✅ **1 correction** |
| 2 | Cross-component cancellation ratios | `make context-value`, `make opponent` | `context_value.py`, `opponent.py` | 3 + 5 new columns | ✅ **1 correction** |
| 3 | The season total's two log factors | `make target-profile` | `src/eda/target.py` | 5 new rows | ✅ exact |
| 4 | Playoff-vs-regular scope evidence | `make availability-profile` | `src/eda/availability.py` | 26 new rows | ✅ **1 correction** |
| 5 | Roster description coverage | `make context-value` | `src/eda/context_value.py` | `roster_coverage_profile_tier{A,B}.csv` | ✅ **1 correction** |
| 6 | **The residual correlation matrix** | `make residual-correlation` | `src/eda/residual_correlation.py` **new** | `residual_correlation.csv` (242 rows) | ✅ population named |
| 7 | Bonus calibration | `make component-targets` | `src/features/targets.py` | `bonus_calibration.csv` | ✅ **2 corrections** |
| 8 | Minutes feasibility | `make game-length` | `src/features/game_length.py` | 31 new rows + an assert | ✅ exact |
| 9 | Name-match audit | `make adp-panel` | `src/features/adp.py` | `adp_match_audit.csv` (57 rows) | ✅ exact |
| 10 | Regularization sensitivity | `make component-rates` | `src/models/component_rates.py` | 112 new rows | ✅ exact |

Two new modules, eight extensions. The extensions were the bulk of the value and the smaller
half of the work, because each one added a section to a module that already loaded the right
data. **Every extension added rows or columns; none rewrote an existing schema**, and the
existing tests confirm it — `game_length_coverage.csv` and `component_rate_metrics.csv` gained
an `analysis` column so their two sections stay separable, and `target_profile.csv` gained
`r2` / `sd` / `p10`.

---

## The work items

### 1 · `make variance-budget` — the frame

The variance budget is the first thing the dashboard shows and the most-quoted table in the
project, and it has no artifact. `README.md` says to reproduce it with
`opponent.py::variance_ceiling`, which is true for the opponent rows only — the player-season
identity, own-minutes and home/away rows have no runnable source at all.

New `src/eda/variance_budget.py`, reusing `opponent.variance_ceiling` and
`feature_diagnostics.cell_importance` rather than reimplementing either. One row per source:

```
source, n_games, share_of_variance, basis, in_sample, null_construction, seasons
```

`basis` distinguishes "share of total per-game variance" (player-season identity) from "share of
within-player residual" (everything else) — conflating those two denominators is the easiest way
to misread the table, so the artifact should make it impossible.

Two things it must carry beyond the five headline rows: the **within-player-season residual sd**
(the units everything else is quoted against), and **both null constructions** for the
interaction row, since the same statistic reads +0.97% permuting opponent and +0.31% permuting
archetype and quoting one without naming the marginal is a recorded past error.

*Verification:* the opponent rows must reproduce `opponent_matchup_tierA.csv`'s in-sample
figures, and the null rows must match `feature_diagnostics.csv`'s `null_reproduction` section to
within the 0.005 pp already recorded between those two implementations.

**✅ Measured**, 254,167 player-games in 2014-15..2023-24 (198,509 with a prior-season
archetype), `n_shuffles = 20`:

| row | basis | recorded | measured |
|---|---|---|---|
| player-season identity | total | 58.0% | **57.957%** |
| **own minutes** | residual | **18.6%** | **46.399%** ⚠️ |
| own minutes, nonparametric | residual | — | 46.189% |
| own minutes, saturated per player-season | residual | — | 59.398% |
| own minutes, *superseded* raw-level construction | residual | 18.6% | 18.650% |
| opponent × season | residual | 0.69% | **0.691%** |
| home / away | residual | 0.03% | **0.034%** |
| opp × arch × season, above shuffle-**opponent** | residual | +0.97% | **+0.962%** |
| opp × arch × season, above shuffle-**archetype** | residual | +0.31% | **+0.308%** |
| within-player-season residual sd | dk_pts | 9.4 | **9.445** |
| opponent effect sd (season-averaged) | dk_pts | — | 0.867 |

Two notes on the verification as written. **`opponent_matchup_tierA.csv` never carried the
in-sample figures** — it holds the *held-out* contrast, and the ceiling was print-only inside
`opponent.run`. That is itself an instance of the gap this plan exists to close, and
`variance_budget.csv` now holds it. The checkable reference is `feature_diagnostics.csv`'s
`null_reproduction` section, and the two implementations agree to **0.0050 pp and 0.0020 pp** —
exactly the recorded `reproduction_gap`. Both helpers are run on identical rows and the gap
ships as its own row, so they cannot drift apart silently.

The main effects are measured on the **full** window (254,167) and the interaction on the
archetype panel (198,509), because requiring a prior-season archetype drops 22% of the window
and moves opponent × season from 0.691% to 0.793%. Both frames are emitted, suffixed, each
with its own `n_games`.

### 2 · Cross-component cancellation

**The single strongest measured argument for the component-head architecture**, and it is prose.
`team_context_value_tier*.csv` carries partial *r* per component, which is not the same
quantity: the cancellation ratio needs each feature's per-sd effect in **component units**,
weighted by DK coefficients, summed in absolute value (gross) against summed signed (net).

Extend `context_value.py` to add `gross_dk_movement`, `net_dk_movement` and
`cancellation_ratio` columns, and `opponent.py` to write the same pair for the opponent main
effect (the 1.105-gross-against-0.785-net figure). Same rows, three more columns — no new
artifact, no new target.

*Verification:* `net_dk_movement` must agree with the existing `r_dk_pts_per_game` column in
sign for every feature; a disagreement means the DK weighting is wrong.

**✅ Measured.** `context_value.py` now emits `effect_<outcome>` per feature — the per-sd effect
in the component's **own units**, `r_c × sd(residual y_c)` off the same residuals as the
existing correlations — plus the DK-weighted triple. Tier A, 8,038 transitions, season
absorbed. All four recorded cancellation ratios reproduce to the quoted precision:

| feature | recorded ratio | gross | net | measured ratio |
|---|---|---|---|---|
| `teammate_assist_supply` | 8.3× (2.11 into 0.25) | 2.106 | −0.254 | **8.298×** |
| `role_crowding` | 8.2× | 0.588 | −0.072 | **8.215×** |
| `teammate_spacing` | 5.0× | 0.869 | +0.173 | **5.012×** |
| `team_pace` | 4.7× | 0.669 | +0.141 | **4.739×** |

**The units in the prose were mislabelled.** `CLAUDE.md` said `teammate_assist_supply` "moves
ast/36 −0.361, reb/36 +0.218, blk/36 +0.191 **per sd**". Those are the *partial correlations*
already in `team_context_value_tierA.csv`, not per-sd movements — a correlation is
dimensionless and cannot be DK-weighted, which is why the gross/net pair could not be
reconstructed from them. The ratios were right; the sentence describing how they were built
was not.

**The sign check needed a magnitude floor.** As specified it fires on tier B for
`role_crowding` (r = +0.010) and `team_pace` (r = −0.016) — both recorded *nulls*, where the
sign of a near-zero net and a near-zero correlation are each noise. Features with
`|r_dk_pts_per_game| < 0.02` are now reported as unchecked rather than failing; above it, all
five (tier A) and six (tier B) agree.

`opponent.py` gained `opponent_sd`, `opponent_sd_corrected`, `dk_abs_weight` and
`dk_weighted_opponent_sd` per outcome, with the gross/net/ratio triple on the `dk_pts` row.
**The recorded 1.105-against-0.785 pair did not reproduce** — see
[Corrections](#corrections-six-figures-that-did-not-reproduce).

### 3 · The season total's two factors

`target_profile.csv` already carries the first-*k* rows (buckets 5/10/20/41/60) — verified —
but not the log decomposition. Extend `target.py` with an `analysis == "season_total_decomposition"`
section: R² of `log(total)` on `log(rate)` alone and on `log(games)` alone, plus games-played
mean, sd and p10/p90. Small, and it completes a table the dashboard's opening tab leans on.

**✅ Measured, and every figure is exact.** The population is player-seasons with **≥ 10
games** — `SEASON_TOTAL_MIN_GAMES`, now a named constant, and the reason the figures are
quotable at all:

| figure | recorded | measured |
|---|---|---|
| R² of `log(total)` on `log(rate)` | 84.5% | **0.8454** |
| R² of `log(total)` on `log(games)` | 73.4% | **0.7343** |
| both together (the identity) | — | **1.0000** |
| games: mean / sd / p10 / p90 | 56 / 20.8 / 23 / 80 | **56.0 / 20.8 / 23 / 80** |
| n | — | 12,996 |

Two things the section adds. The `log_rate_and_games` row must return **exactly** 1.0 because
`log(total) = log(rate) + log(games)` is arithmetic — a free internal check that the
decomposition is of the right identity. And the two factors correlate at **+0.585**, which is
why the shares sum past 1: they are overlapping shares, not a partition, and quoting them as
though they partitioned would double-count.

The games floor is load-bearing rather than cosmetic. `preprocess.clean` filters on a player's
*career* games, not his season, so the unfiltered frame carries one- and two-game seasons whose
per-game rate is noise: it reads **78.5% / 81.7%** and **reverses which factor dominates**. A
test pins that the floor changes the answer.

### 4 · Playoff scope evidence

The regular-season-only decision is justified first by the product (`docs/dk_best_ball_rules.md`:
Round 4 ends 4/4) and second by a sign flip across role buckets. The second half is prose.

`src/eda/availability.py` already loads playoff logs for the workload block, so this is a
section rather than a module: `measurement == "playoff_scope"`, one row per prior-season role
bucket (`<12` / `12–24` / `24+` mpg) with median playoff-to-regular MPG ratio, appearance rate,
and n.

*Verification:* the buckets must partition the qualifying population exactly, and the starter
bucket's ratio must exceed 1.0 — the sign flip is the finding, so a version that does not
reproduce it has failed.

**✅ Measured, unweighted per the recorded exception.** Buckets are on the *same* season's
regular-season mpg, which is what makes "playoff-to-regular ratio by role" coherent; the
qualifying population is 5,762 player-seasons with ≥ 20 regular-season games and a playoff
appearance, and **0 rows fall in no bucket**:

| role | n | median reg mpg | median playoff mpg | ratio (recorded) | share playing more | appearance rate (recorded) |
|---|---|---|---|---|---|---|
| bench < 12 | 831 | 9.2 | 4.2 | **0.505** (0.505) | 0.146 | **0.711** (0.664) ⚠️ |
| rotation 12–24 | 2,242 | 18.1 | 13.6 | **0.761** (0.761) | 0.260 | **0.905** (0.838) ⚠️ |
| starter 24+ | 2,689 | 31.4 | 33.3 | **1.054** (1.054) | **0.658** | **0.958** (0.922) ⚠️ |

The MPG ratios, the n and the 66%-of-starters figure reproduce exactly. **The appearance rates
do not** — see [Corrections](#corrections-six-figures-that-did-not-reproduce). The sign flip,
which is the finding, is intact either way and the ordering is if anything sharper.

The appearance rate's denominator is scoped to players on a **playoff team**, because otherwise
"his team missed the playoffs" and "he did not dress" become the same fact — the same
distinction this module keeps everywhere else.

### 5 · Roster description coverage

`team_context_tier*.parquet` carries `stats_source`, `reliability` and `roster_coverage` per
row, so the head-count split is derivable today. What is not is the **minutes-weighted** split,
which is the honest denominator and the reason the figure is quoted at all: weighting by the S-1
minutes the aggregate uses would show a rookie contributing zero and every roster looking fully
covered.

Extend `context_value.py` → `outputs/eda/roster_coverage_profile.csv`: share of realized
season-S roster minutes by `stats_source`, overall and per team-season, with the p50/p90/max
distribution and the worst offenders named.

*Verification:* the three shares plus `prior` must sum to 1.0 per team-season, and the
head-count column must reproduce the `stats_source` mix already derivable from the parquet.

**✅ Measured** → `outputs/eda/roster_coverage_profile_tier{A,B}.csv`. Both verifications pass:
the four shares partition every team-season to **2.22e-16**, and the head-count mix reproduces
`stats_source` to **0.000%** on all three of its categories.

Two findings the item did not anticipate, and both mattered.

**`team_context.coverage_report` already existed and was never called** — and it fell back to
`prior_minutes` because nothing ever supplied `season_minutes`, i.e. it would have reported the
circular weighting its own docstring warns against. It is now called, generalized to take the
category list, and fed realized season-S minutes.

**The four-way split cannot come from `stats_source`.** That column knows only prior / stale /
rookie against the *inclusive* frame; the sub-threshold line is a fact about the **qualified**
matrix. `description_source` derives all four from the two matrices directly, which is why it
is independently testable — and the cross-check against `stats_source` is exact.

| source | tier A minutes | tier A rows | recorded |
|---|---|---|---|
| `prior` | 85.29% | 73.42% | — |
| `sub_threshold` | 5.07% | 10.65% | 5.4% |
| `returnee` | 0.91% | 1.83% | 1.1% |
| `rookie` | 8.73% | 14.10% | 9.4% |
| **undescribed** | **14.71%** | 26.58% | **15.9%** ⚠️ |

Per team-season: mean 14.8%, p50 13.0%, p90 29.0%, max 50.4% (CLE 1997-98). **The recorded
15.9% is the whole-season roster, not the season-start one** — see
[Corrections](#corrections-six-figures-that-did-not-reproduce). Both windows are emitted, and
the head-count/minutes gap is the reason both weightings ship: a rookie is 14.1% of roster rows
and 8.7% of roster minutes, and neither substitutes for the other.

### 6 · `make residual-correlation` — the simulator's input

**The highest-value item here, and it would be worth building with no dashboard at all.**
`docs/predictions-plan.md` and `docs/simulations-plan.md` both specify that cross-component
correlation enters at draw time via a shared `min` draw plus, if needed, a Gaussian copula on
residuals. The copula needs **the matrix**. What exists is three prose summary statistics of
it — mean +0.013, max 0.157, and the 3PA/2PA substitution at −0.110 — measured once in a
planning session over 101,588 games.

New `src/eda/residual_correlation.py` → `outputs/eda/residual_correlation.csv`: the full
pairwise conditional correlation matrix across the eleven non-minutes components, residualized
on minutes as exposure, in long form (`component_a, component_b, r, n_games, basis`) so it can
be pivoted or joined without a header convention. Long form specifically because the simulator
will read it programmatically, and a wide matrix with component names as columns is a schema
that breaks the moment a head is added.

It should also carry the **minutes-conditioned** flag explicitly, because the unconditional
correlations are large and the conditional ones are small — mixing them up would produce a
simulator with roughly double the intended cross-component coupling.

*Verification:* the diagonal is exactly 1.0; the matrix is symmetric; the `fg3a`–`fg2a` cell is
negative (the substitution the reparameterization exists to handle); and the three prose
summaries fall out of the matrix within sampling error of the recorded values.

**✅ Measured** → `outputs/eda/residual_correlation.csv`, 242 rows in 1.2 s (so the "could be
expensive" risk did not materialize and no sampling was needed). The frame is
`serial_correlation.py`'s — 592,796 player-games, 9,052 player-seasons — and the residual
helpers are imported from it, so the two artifacts sit on identical rows with one definition of
"the residual against the player-season's own rate with minutes as exposure".

Diagonal exactly 1.0, matrix symmetric, `fg3a`–`fg2a` negative, minimum eigenvalue **+0.756**
so it is comfortably PSD and usable as a copula without a nearest-PSD correction. Both ordered
pairs are emitted so a plain `pivot` returns a square matrix.

| summary | recorded | full frame (592,796) | 2021-22+ (101,482) |
|---|---|---|---|
| off-diagonal mean, 8 counts | +0.013 | **+0.0121** | +0.0187 |
| off-diagonal mean, all 11 heads | — | +0.0071 | +0.0091 |
| max off-diagonal | 0.157, `fg2a`–`reb` | +0.1422, **`fg2a`–`reb`** | **+0.1558**, `fg2a`–`reb` |
| `fg3a`–`fg2a` substitution | −0.110 | −0.1248 | −0.0823 |

**The recorded population is identified: 2021-22 onward.** The recorded n of 101,588 matches
that subset's 101,482 to within 106 rows, and it is where the 0.157 maximum comes from. The
full frame is what ships — more data, and the same rows as `serial_correlation.csv` — and it
reproduces the mean (+0.0121 against +0.013) and the sign and rough size of the substitution.
All three prose summaries are confirmed in substance; the exact values were population-specific
and the population was never stated, which is the whole failure mode this plan exists to fix.

The `raw` basis is carried beside the conditioned one and the contrast is larger than expected:
off-diagonals average **+0.112 against +0.007**, a **16×** inflation, with the largest raw cell
`fg2a`–`fta` at **+0.492** — two shot-volume counts both scaling with the minutes they were
accumulated over, not a cross-component dependence. Building the copula on the raw matrix would
impose 16× the intended coupling *on top of* the shared minutes draw that produced it, which is
why `minutes_conditioned` is an explicit column rather than an implicit filename convention.

### 7 · Bonus calibration

`targets.py::expected_bonus` carries `overdispersion=0.10`, calibrated against 11,627
player-seasons, with independent sampling measured 23% too low. `CLAUDE.md` says not to change
it without re-running the calibration — and there is no target that re-runs the calibration.

Extend `targets.py` → `outputs/eda/bonus_calibration.csv`: realized mean bonus per game, the
Monte Carlo estimate at the shipped overdispersion, the independent-sampling estimate, the bias
of each, and the same broken out by minutes bucket (the calibration's claim is that it holds
across buckets, which is checkable and currently unchecked).

*Verification:* the shipped overdispersion's bias must be within ~0.001 dk_pts/game of zero and
the independent estimate must read low by ~23%. This turns "do not change it without
re-calibrating" from an instruction into a test.

**✅ Measured** → `outputs/eda/bonus_calibration.csv`. Both halves of the verification pass, on
11,938 player-seasons with ≥ 200 season minutes:

| overdispersion | E[bonus] | realized | bias | relative |
|---|---|---|---|---|
| 0.00 (independent) | 0.0951 | 0.1231 | −0.0280 | **−22.7%** |
| **0.10 (shipped)** | 0.1239 | 0.1231 | **+0.0009** | +0.7% |
| fitted optimum | — | — | 0.0 | **0.0968** |

So the shipped 0.10 is the season-unit optimum to two figures, and independent sampling reads
22.7% low against the recorded 23%. **What `BONUS_OVERDISPERSION` actually is** was also
unstated and is now in the code: the variance of the shared per-game Gamma frailty in
`expected_bonus` — which does two jobs at once, making each category's marginal negative
binomial *and* inducing the positive dependence the bonus needs. It is not a dispersion of
`dk_pts` and it is not fitted by any head.

Two corrections came out of the bucket break, which the item correctly called "checkable and
currently unchecked" — see
[Corrections](#corrections-six-figures-that-did-not-reproduce). The short version: the
**aggregate** fit is good and the **per-bucket** fit is not, and the value the simulator needs
is **0.025, not 0.10**, because the simulator draws per game.

Cost note: the player-season arm sweeps the whole grid at ~1.5 s a point; the player-game arm is
60× the rows so it gets three points at `n_samples = 256`, which is four orders below the
±0.001 tolerance in Monte-Carlo error on a mean over 708k rows. `make component-targets` grows
by ~2.5 minutes.

### 8 · Minutes feasibility

`game_length_coverage.csv` covers the derivation (disagreements, residuals, OT rate) but not the
**feasibility join** that makes `min ~ Binomial(game_length, ·)` well posed: 100% coverage of
731,906 player-games, zero rows with `min > game_length`, max ratio exactly 1.0000, 1,650
player-games above 48 minutes, observed max 63.0.

Extend `game_length.py` with a feasibility section. Note the recorded quirk while doing it:
`game_id` is `int64` in `component_targets.parquet` and zero-padded in the box-score files, so
the join goes through `boxscore_status.pad_game_id`.

*Verification:* the zero-violations row is the whole point — it must be computed and asserted,
not reported. A nonzero count is a build failure, not a metric.

**✅ Measured, and every figure is exact.** `assert_feasible` raises rather than reporting, and
`make game-length` calls it:

| figure | recorded | measured |
|---|---|---|
| join coverage | 100.0% of 731,906 | **100.0% of 731,906** |
| rows with `min > game_length` | 0 | **0** (asserted) |
| max `min / game_length` | 1.0000 | **1.0000** |
| player-games above 48 minutes | 1,650 | **1,650** |
| observed maximum minutes | 63.0 | **63.0** |

The assertion has two arms, because they fail differently and both are silent otherwise: a
violation means the binomial support is wrong, and an *unmatched* player-game means the join is
broken — which is the `pad_game_id` trap, and which would otherwise show up only as a quietly
smaller fitting frame. Both sides go through `pad_game_id` so the join is type-agnostic, and a
test feeds it an int on one side and a zero-padded string on the other.

`game_length_coverage.csv` gained an `analysis` column (`derivation` / `feasibility`) so the
per-season derivation rows and the 31 feasibility rows stay separable. The derivation figures
are unchanged: 37,986 games, 0 disagreements, worst residual 0.617 min, 5.93% OT.

### 9 · Name-match audit

`CLAUDE.md` records this as the most dangerous measurement in the repo: an unmatched rate is
monotonically increasing in the error it is supposed to detect, and the rejected surname-initial
rule scored 0.0% unmatched while fabricating 11 of its 12 non-exact matches
(`Cameron Boozer` → Carlos Boozer, `RJ Davis` → Ricky Davis).

The rule for handling it — "list every non-exact match and read them" — has no artifact. Extend
`src/features/adp.py` → `outputs/eda/adp_match_audit.csv`: every surviving non-exact match with
both names, the rule that matched it, and the seasons-apart gap, **plus the rejected rule re-run
as an ablation** with its false-match count and examples.

Keeping the rejected rule running permanently is the point. It is a live demonstration that the
metric is untrustworthy, sitting next to the metric — and if someone later "simplifies" the
cascade back toward it, the audit says so immediately.

*Verification:* `no_nba_history` stays separate from `unmatched` in the output, per the standing
rule that collapsing them turns "the 2026 draft class exists" into a fake defect. The ablation
must reproduce a false-match count well above the cascade's.

**✅ Measured** → `outputs/eda/adp_match_audit.csv`, 57 rows over 3,591 unique
(source, name, season) rows. `no_nba_history` (174 rows) is counted apart from `unmatched` (17)
and excluded from the rate's denominator.

**All 16 surviving non-exact matches are listed and every one reads correctly** — the tier is
small enough to eyeball, which is the feature: `Enes Kanter` → Enes Freedom, `Alexandre Sarr` →
Alex Sarr, `Hansen Yang` → Yang Hansen, `Gregory Jackson` → GG Jackson, `Nene Hilario` → Nene,
`Louis Williams` → Lou Williams, each with its rule and its seasons-apart gap.

**The ablation reproduces the warning exactly.** The rejected surname-initial rule makes 31
matches, **23 of which the cascade refuses**, and it scores **0.00% unmatched against the
cascade's 0.50%**. The recorded examples come straight back out — `Cameron Boozer` → **Carlos
Boozer** and `Darryn Peterson` → **Drew Peterson** — alongside new ones the record did not name
(`Mikel Brown Jr.` → Moses Brown, `Baba Miller` → Brandon Miller, `Dillon Mitchell` → Davion
Mitchell). The recorded count was 11-of-12 on the two DK boards alone; 23-of-31 is the same rule
over both sources and nine seasons, so the scope differs and the character does not.

A *better* unmatched rate on a rule that fabricates matches, sitting in the same file as the
rate, is the artifact this item wanted: the metric is monotonically increasing in the error it
is supposed to detect, and now it says so on every run.

### 10 · Regularization sensitivity

The `sklearn` alpha trap cost a full set of published figures: `PoissonRegressor` averages
deviance by the weight sum, so with `sample_weight = minutes` (Σw ≈ 1e7) `alpha=1.0` silently
crushes every coefficient, and `reb` read R² 0.662 against 0.928. It failed quietly because a
flexible basis partially compensates.

Extend `component_rates.py` with an `alpha_sensitivity` section: held-out R² per head across an
alpha grid spanning the broken and correct regimes, plus the no-fit floor on the same axis. Two
uses — it documents the trap with a curve instead of an anecdote, and it is a regression guard,
because a future refactor that reintroduces a default alpha shows up as the curve's left end
crossing the floor.

*Verification:* the curve must reproduce the recorded pair (≈0.66 at `alpha=1.0`, ≈0.93 at
`alpha ≤ 0.01`) for `reb`, and the floor line must sit above the fitted line at high alpha —
which is the visual statement of why the floor is mandatory.

**✅ Measured, and the recorded pair is exact.** 112 rows appended to
`component_rate_metrics.csv` under `analysis == "alpha_sensitivity"`: eight count heads × two
variants × a seven-point grid from 1e-8 to 10, with `floor_r2` on every row so the crossing
needs no join.

`reb` on the `linear` spec reads **0.9278 at alpha = 1e-8**, **0.9322 at alpha = 0.01** and
**0.6620 at alpha = 1.0** — against the recorded 0.928 and 0.662. At alpha = 10, **16 of 16
fits fall below the no-fit floor**, which is the visual statement the item asked for.

The sweep runs on both the raw-rate spec (where the recorded figure was measured) and the
shipped `log_own` spec, because "does over-regularization hurt what we actually ship" is the
question a guard needs to answer. It does: median R² loss from the shipped alpha to 1.0 is
**0.228** on `linear` and **0.289** on `log_own`, worst `fg3a` at **0.595**.

One refinement on the guard as specified. "The fitted line crossing the floor" cannot be the
trigger on its own — `blk` and `fg3a` lose to the floor at *every* alpha under `log_own`,
because those two heads need splines, which is a documented modelling finding rather than the
penalty misbehaving. The guard therefore compares each head against **its own optimum on the
grid** (all of which sit at ≤ 0.1) and reports the loss to alpha = 1.0.

---

## Corrections: six figures that did not reproduce

Every one is recorded in `CLAUDE.md` beside the claim it changes, per the risk register. In all
six the *recorded* value turned out to be the wrong one, and in all six the conclusion the
figure supported survives — twice more strongly than before. That is the pattern the plan
predicted: these were not less true, they were less checkable, and four of the six were
unreproducible because a **population or a construction was never stated**.

### 1 · Own minutes is 46.4% of the within-player residual, not 18.6%

The largest and most consequential. The recorded construction conditions the within-player-season
residual on the **raw minutes level**, pooled across players, which is attenuated by
construction: a 30-minute game is *below* average for a 34-mpg starter and far *above* it for an
18-mpg reserve, so their residuals cancel inside the cell. `own_minutes_raw_level` reproduces
the recorded 18.6% to three digits (18.650%), which is how the construction was identified.

Conditioning instead on the minutes **deviation** — "he played eight more minutes than he
usually does", the contrast the residual is *defined* by — gives **46.399%**, stable at 46.189%
under a nonparametric fit, against a saturated per-player-season upper bound of 59.398%. The
mechanism is directly measurable and also now on disk: dk_pts per extra minute runs **0.852** at
the bottom mpg tier to **1.230** at the top, so no single function of the raw level can
represent it.

**Direction: safe.** Minutes matter *more* than recorded, which strengthens every argument built
on the shared-`min` draw — `docs/predictions-plan.md`'s "draw `min` once and push it through all
eleven heads", and `residual_correlation.py`'s finding that what is left over is small. All four
constructions ship, `own_minutes_raw_level` included, so the superseded figure stays legible
beside the corrected one instead of being quietly overwritten.

### 2 · The opponent cancellation is ~2×, and the recorded pair mixed two bases

Recorded: 1.105 gross against 0.785 net, implying 1.41×. Measured on the same panel, four
internally consistent constructions all land between **1.84× and 2.11×**:

| construction | gross | net | ratio |
|---|---|---|---|
| per-season sd of cell means, minutes-weighted (**shipped**) | 1.981 | 0.987 | 2.01× |
| the same, sampling variance removed (**shipped as `_corrected`**) | 1.803 | 0.911 | 1.98× |
| pooled over seasons | 0.856 | 0.406 | 2.11× |
| ridge fit on the opponent's prior-season profile | 0.935 | 0.509 | 1.84× |

The ratio is robust to the choice precisely *because* each is internally consistent, so a 1.41×
cannot come from any one of them — gross and net were measured on different bases in the
original. `cross_component_cancellation` now takes gross and net from the same `sd_col` and says
so in its docstring, and a test pins that the ratio is invariant to the column.

**Direction: safe.** The components move ~2× as much as their DK sum does, not 1.4× — a stronger
argument for component heads than the recorded figure made.

### 3 · The bonus calibration does *not* hold across minutes buckets

The aggregate claim reproduces (+0.0009 bias at 0.10, 22.7% low under independent sampling), but
"with good fit across minutes buckets" does not. At the shipped value the per-mpg-bucket bias
runs **−0.0142 at 12–18 mpg against +0.0291 at 30–48** — a 0.0433 spread that cancels to
+0.0009. The buckets do not fit; their errors offset.

The reason is structural rather than a tuning miss: one scalar frailty variance is standing in
for minutes variation whose *relative* size differs by bucket, so no single value can fit them
all at the player-season unit.

### 4 · The simulator needs overdispersion 0.025, not 0.10

The other half of the same finding, and the one with teeth. At the **player-game** unit —
expected counts = per-36 rate × that game's actual minutes, which is how `expected_dk_pts` is
called and how the simulator will draw — minutes are no longer hidden inside the frailty, so the
residual overdispersion is ~4× smaller. Fitted: **0.0248**. And at that value the fit holds
across *every* minutes bucket (bias −0.0004 to +0.0007) rather than only in aggregate:

| minutes bucket | bias at 0.00 | bias at **0.025** | bias at 0.10 |
|---|---|---|---|
| 0–12 | −0.0000 | **+0.0000** | +0.0002 |
| 12–18 | −0.0011 | **+0.0001** | +0.0052 |
| 18–24 | −0.0063 | **+0.0004** | +0.0196 |
| 24–30 | −0.0129 | **+0.0007** | +0.0338 |
| 30–48 | −0.0140 | **−0.0004** | +0.0362 |

Using 0.10 per game over-predicts the bonus by **+0.036 dk_pts/game for 30+ minute players** —
exactly the players the bonus is worth most for. `BONUS_OVERDISPERSION` is left at 0.10 because
it is correct for its documented unit and no production code reads it yet;
`BONUS_GAME_OVERDISPERSION = 0.025` is added beside it with the unit stated in both docstrings.

### 5 · Roster coverage is 14.7% on the roster the model actually uses

The recorded 15.9% / 9.4% / 5.4% / 1.1% is the **whole-season** roster. Measured across seven
window sizes, window 82 gives 15.79% / 9.16% / 5.52% / 1.11% — a match — while the shipped
`roster_window_games = 10` gives **14.71% / 8.73% / 5.07% / 0.91%**.

The gap is mechanical and it is the user's own observation: rookies and players absent the
previous season **arrive late**, so widening the window pulls in disproportionately many of
them. The shipped figure is the one that describes a real bias, because the season-start roster
is the only one knowable before the season and therefore the only one `team_context` aggregates.
Both windows are emitted so the difference stays visible.

**Direction: the recorded figure was pessimistic by 1.2 pp.** "Never drop them" is unaffected —
it is still ~15% of realized minutes with p90 29% and a worst roster at 50%.

### 6 · Playoff appearance rates are 0.711 / 0.905 / 0.958

Recorded: 0.664 / 0.838 / 0.922. Measured with a per-(player, team) denominator — one row for
every team a player appeared for — the numbers are 0.647 / 0.821 / 0.927, which brackets the
record and identifies the construction.

That construction is wrong for this project's own stated reason: it records a player traded away
from a playoff team in February as having "not appeared" for it, which is roster churn
misread as unavailability — the conflation `features/availability.py` refuses to make everywhere
else — and it double-counts him. Keying on the player's **last** team gives 0.711 / 0.905 /
0.958.

**Direction: the recorded figures overstated how much rotations shorten.** The ordering, which
is the finding, is unchanged and slightly sharper.

### Not corrections, but unstated populations now named

Two figures reproduce once their population is stated, and could not have been checked before:

- **The residual correlation matrix** was measured on 2021-22 onward (101,482 rows against the
  recorded 101,588), which is where the 0.157 maximum comes from. The full 592,796-row frame
  ships and gives +0.0121 / +0.1422 / −0.1248.
- **The bonus calibration's level pair** (0.098 vs 0.127) came from a filter of roughly ≥ 250
  season minutes (11,607 player-seasons against the recorded 11,627). The artifact uses the
  project's standing ≥ 200-minute threshold and reports 0.0951 vs 0.1231; the *ratio* and the
  *bias*, which are what the claim rests on, are unaffected. The docstring's "(2014-15 →
  2025-26)" was also wrong: only 6,067 player-seasons exist in that window, so a calibration on
  11,627 of them cannot have come from it.

---

## What is still prose-only

The inventory's ten items are closed. Auditing `CLAUDE.md` while doing them turned up **six
further figures with no runnable source**, all missed by the original inventory and all in the
PCA / archetype / dimension-reduction line that `docs/dashboard-plan.md` defers:

| figure | where it is asserted |
|---|---|
| kNN over player-seasons leaks identity — 18.8% nearest, 37.7% within 5 | `CLAUDE.md`, DR section |
| The mean centroid collapses — 336 team-season pairs in the closest 1% | `CLAUDE.md`, DR section |
| Minutes-weighted mean of PC scores equals the projection of the mean, to 3.6e-15 | `CLAUDE.md`, DR section |
| Mid-season churn — 78/572 players (13.6%) appeared for 2+ teams in 2023-24 | `CLAUDE.md`, scope section |
| 98.4% of minutes rows are non-integer | `game_length.py`, `stan_minutes.py` docstrings |
| The global reliability curve `0.924 · m/(m+66)`, fitted on 11,272 pairs | `team_context.py` constants |

None is load-bearing for a decision that is still open — the DR line is not feeding the
pipeline, mid-season churn is explicitly out of scope, and the per-column reliability curves in
`persistence.csv` already supersede the global one. They are listed so the count is a number
rather than an impression, and so a future sweep starts from a list instead of a re-read.

### Eight more, found by extending `make docs-audit` to the other three docs (2026-07-31)

Registering a claim per quoted figure across `docs/predictions-plan.md`, `docs/adp-plan.md` and
`CLAUDE.md` forces the question "which artifact is this from?" on every number, and eight came
back with no answer. Same character as the six above — none is load-bearing, and every one is
either a costing estimate or an alternative construction quoted beside a shipped one.

| figure | where it is asserted | why it has no target |
|---|---|---|
| The three unshipped opponent-cancellation ratios — **2.01×** raw per-season sd, **2.11×** pooled, **1.84×** ridge on the prior profile | `CLAUDE.md`, cross-component cancellation | `opponent.py` emits only the shipped `_corrected` construction (1.98×). The other three were measured once to show the ratio is robust to the choice. |
| The **1.665×** AR(1)-implied block inflation, and the **46% / 51% / 30%** shares derived from it | `docs/predictions-plan.md`, "Why not just an explicit lagged term?" | A comparator computed from `serial_correlation.csv`'s `lag1`, not stored. Cheap to promote if the residual serial process is built. |
| Per-game minutes costing — **731,863** rows, **81×** the season head, **6–16 h**, **~500,000** latent variables | `docs/predictions-plan.md`, the deferred per-game question | Estimates, not measurements. |
| Composition costing — **71,092** team-game rows, **35,546** games, **~10.3** players each | `docs/predictions-plan.md`, the composition alternative | Superseded in substance by `stan_composition_metrics.csv`, which reports the realized 52,957 / 4,920 held-out split. |
| The DK carry-forward ladder — **30.29** raw / **26.71** isotonic CV MAE, ρ **0.743**, on 175 players | `docs/adp-plan.md`, "What the second DK board does and does not buy" | `adp_profile.csv`'s ladder predicts DK from *consensus*; the carry-forward arm predicts DK from *prior-year DK* and is a separate fit. Worth promoting — it is the measurement that justifies the FantasyPros pipeline's existence. |
| The freeze-rule identical-`AVG` shares — **33.8% / 100.0% / 0.9%**, and the flip thresholds **0.0055–0.0871** against **0.0667–0.2273** | `docs/adp-plan.md`, the freeze rule | The rule is implemented in `adp_fantasypros.py`; the shares that established it are not written out. |
| Load management — heavy-minute players lost **−0.101** of games-played share against **−0.037** for fringe, MPG given role flat | `docs/availability-plan.md`, "Load management" | Explicitly labelled a *first look*. `season_effects_summary.csv` carries the `gp_share` role buckets, so this is a small extension rather than a new module. |
| The sequence-feature ablation — **0.7599** aggregates / **0.7657** plus order features / **0.7584** shuffled | `CLAUDE.md`, the LSTM/Transformer decision | Measured once. It is the evidence for a *settled* decision not to build the sequence trunk, so re-deriving it has low value — but it is the one prose-only figure a decision actually rests on. |

The first two are the ones to promote if any are: the opponent ratios because `CLAUDE.md`
quotes four numbers where the artifact holds one, and the DK carry-forward ladder because it is
the load-bearing argument for a whole data pipeline.

`make dashboard-audit` does not exist yet (it is stage 1 of `docs/dashboard-plan.md`), so the
"zero provenance-marked typed constants" completion criterion cannot be evaluated from this side
— but there is now an artifact behind every figure the three gated tabs need.

---

## Conventions and wiring

Per `CLAUDE.md`, with nothing novel:

- `python -m src.<module>` entry points; `cfg = yaml.safe_load(open("configs/default.yaml"))` in
  `__main__`; `Path(...).mkdir(parents=True, exist_ok=True)` before writes;
  `f"... {n:,} ... → {dest}"` progress lines.
- Two new `Makefile` targets — `variance-budget`, `residual-correlation` — in `.PHONY` and added
  to `make eda` in dependency order. Both read `component_targets.parquet` and the season
  matrices, so they sit after `component-targets` and after `opponent`.
- Extensions add sections to existing outputs and need no new targets, which is deliberate:
  eight of ten items cost nothing at the Makefile level.
- Tests per item, plain `assert` with synthetic builders, in the existing per-module test files
  (`tests/test_target.py`, `tests/test_availability.py`, …) rather than one new file — the
  extensions belong to their modules.
- **Absorb season and minutes-weight where the standing rules say to.** Items 1, 2, 3 and 5 all
  regress across 30 pooled seasons, which is exactly where this repo has produced a false
  finding before. Item 4 is availability-adjacent and therefore **unweighted**, per the recorded
  exception.

---

## Staging

Ordered by what unblocks the dashboard soonest, with the two standalone-value items early
because they are worth doing regardless.

| stage | items | unblocks |
|---|---|---|
| **1** | 6 (residual correlation), 7 (bonus calibration) | Dashboard tab 7; and the simulator gets its copula input |
| **2** | 1 (variance budget), 3 (season-total decomposition) | Dashboard tab 1 |
| **3** | 5 (roster coverage), 9 (match audit) | Dashboard tab 2 |
| **4** | 2 (cancellation), 4 (playoff scope) | Completes tabs 1 and 3's decision cards |
| **5** | 8 (minutes feasibility), 10 (alpha sensitivity) | Retires the pending markers in dashboard tabs 5 and 6 |

Stage 1 can start immediately. Stage 5's item 10 touches `component_rates.py`, which the
in-flight `make stan-components` run does not, so there is no conflict.

## Verification, overall

- `make eda` from clean rebuilds every artifact including the two new ones, and every figure the
  dashboard renders resolves to a row in one of them.
- `make dashboard-audit` reports **zero** provenance-marked typed constants. That is the
  completion criterion for this plan, and it is a number rather than a judgement.
- Every recorded value that a new target recomputes is compared against the prose figure it
  replaces, and **a disagreement is investigated rather than adopted**. These figures have been
  quoted in decisions; if a promoted measurement disagrees with its scratch-script original, one
  of the two is wrong and it matters which.

## Risks

- **A promoted measurement may not reproduce its prose original.** This is the real risk and
  also the reason to do the work: a figure that cannot be reproduced was never checkable, and
  finding that out now is strictly better than finding it out after another decision rests on
  it. Any disagreement gets recorded in `CLAUDE.md` as a correction, not quietly overwritten.
- **Scope creep from "figure" to "everything."** The figures-versus-incidents line above is the
  control, and the closing sentence of that section is the part that matters: inconvenience does
  not make a figure an incident.
- **Item 6 could be expensive.** The conditional correlation is over ~700k player-games and 55
  pairs. If a full pass is slow, the honest response is to sample and record the sample size in
  the artifact, not to reduce the pair set.
- **Eight extensions touch eight tested modules.** Each one risks changing an existing output's
  schema. The mitigation is that all eight *add* rows or columns and none rewrite one, and the
  existing tests will say otherwise if that slips.
