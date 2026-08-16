# Preseason data: the problem-statement change, and how it ships

**Decided 2026-08-12.** The project's prediction-time constraint changes: the draft happens
**after the preseason**, not during the offseason. Contest entry is open from the offseason
through the season opener (2026-10-20 for the 2026-27 season), and drafting late primarily
reduces the risk of a season-altering injury landing between draft night and opening night.
The side effect is the subject of this plan: **current-season preseason games become a
legitimate input**, so the information set becomes

> current-season roster membership × prior-season statistics × **current-season preseason
> statistics**.

Nothing about the *form* of the project changes. Same heads, same likelihoods, same
factorized chain, same simulator. Preseason data enters as **additional features on
existing heads** — difference-coded against the prior-season features, opt-in per head,
never replacing them.

Four scoping decisions were taken in the planning session (2026-08-12), all four together:

1. **Commit to the post-preseason frame.** The offseason information set is not a supported
   mode; it survives only as the natural degradation when the preseason block is zero.
2. **Availability and minutes first; component rates gated on an EDA readout.** The
   variance budget says availability is the largest, least persistent lever, and the rate
   heads already sit 0.81–0.95 R² against a no-fit floor — a preseason arm there has to
   earn its sampler time on a cheap train-only measurement first. ⚠️ **Both halves of this
   were revised by P1**: minutes, not availability, is where the block is worth the most,
   and the rate gate *passed* for five of seven count heads. The decision to make the
   readout a gate rather than an assumption is what caught both, which is the argument for
   having written it as one.
3. **The no-prior population is primary scope, not a follow-on.** Preseason games are the
   first real NBA observation for exactly the 14.7% of roster minutes the model imputes.
4. **Nested increment, not ladder re-runs.** Each head gets one extra arm — its shipped
   variant plus the preseason block, with coefficient zero recovering the incumbent exactly
   (the house nesting discipline: `π = 0`, `K = 1`, `U_n = 0`, `n_rho = 1` precedents).
   Full ladders re-run only where the increment wins.

---

## What was measured during planning — scratch figures, 2026-08-12

**Superseded on 2026-08-12 by the P0 artifact below, and two of these rows were wrong.**
Kept as the record of what the planning session believed, per the house convention that a
superseded figure stays beside its correction. Measured with a throwaway probe script
against `nba_api`'s `PlayerGameLogs` (`season_type_nullable="Pre Season"`), not yet a
`make` target.

| season probed | rows | distinct games | date window |
|---|---|---|---|
| 1997-98 → 2002-03 | 1 junk row each | 0 | — (effectively absent) |
| 2003-04 | 465 | 96 | Oct 5–24 (**partial** — far too few rows per game) |
| 2004-05 | 1,154 | 48 | Oct 22–29 (**partial** — final week only) |
| 2005-06 | 2,251 | 88 | Oct 10–28 |
| 2015-16 | 2,831 | 109 | Oct 2–23 |
| 2020-21 | 1,337 | 49 | **Dec 11–19** (COVID December preseason) |
| 2023-24 | 2,078 | 73 | Oct 5–20 |
| 2025-26 | 2,021 | 71 | Oct 2–17 |

What this settled at the time, and what P0 revised:

- **Coverage is solid from 2005-06 onward, partial 2003-05, absent before.** ⚠️ Revised:
  it is solid from **2004-05**, and only **2003-04** is unusable. See P0.
- **2003-04 is "partial — far too few rows per game" (465 rows, 96 games).** ❌ **Wrong,
  and wrong in an instructive way.** 96 of those 465 rows are team-total rows with no
  `player_id`; the real content is **369 player rows over 15 games at 24.6 rows per game**,
  which is *complete* per game. 2003-04's defect is that the API holds only the first four
  days of it. The probe diagnosed a within-game problem where the problem is the schedule.
- **Validation (2022-23, 2023-24) and test (2024-25, 2025-26) are fully covered**, so
  selection and the final evaluation see the feature at full strength. ✅ Confirmed:
  season-start roster coverage 96.9% / 95.7% / 96.1% / 94.3%.
- **The availability head's shipped fitting window (2012-13 onward) sits entirely inside
  preseason coverage**, so its increment arm has no missing-era complication at all. ✅
  Confirmed.
- **The 2020-21 December preseason is the same calendar trap the ADP freeze rule hit.**
  The preseason window must be derived from the data (game dates relative to the season's
  first regular-season game), never from "October". ✅ Confirmed, and the probe missed two
  more: the **2011-12 lockout** preseason (Dec 16–22, two games per team) and the far worse
  **2019-20** case below, where the window straddles its own opener.

---

## P0 — coverage: shipped 2026-08-12

`make preseason` (`src/features/preseason.py`) → `data/features/preseason.parquet` and
`outputs/eda/preseason_coverage.csv`. The backfill wrote 23 seasons of
`game_logs_pre_season_*.csv`; the panel holds **11,707** player-seasons over **23**
seasons. Both artifacts are in `make docs-audit`, so the figures below fail the build
rather than going stale.

### The plumbing came first, because three consumers would have absorbed the new files

Adding a fourth season type to `data/raw/` is not an additive change. Three guards ship
with the panel, each pinned by a test, and the details are in `docs/data-quirks.md`:

1. `preprocess._parse_log_filename` learns the `pre_season_` prefix. Without it the rows
   become the pseudo-season `pre-season-2023-24` **and** are typed `REGULAR_SEASON`, so
   they arrive in the default fitting frame — the 2026-07-29 playoffs bug, one costume on.
2. `load_raw(season_type="all")` now means **regular + playoffs** explicitly. `game_length`
   reads it, and would otherwise have folded ~70 exhibition games a season into the
   artifact whose claim is 0 disagreements over 37,986 games. Rebuilt after the change and
   byte-identical.
3. `features/adp.py::season_start_dates` classifies by prefix rather than by a substring
   test. Left alone it would have moved **23 of 30** season openers ~3 weeks earlier,
   silently re-deciding which ADP captures are point-in-time legal. Verified against the
   stored panel: 0 of 8 seasons moved.

### Coverage

| | |
|---|---|
| seasons with preseason logs | **23** of 30 (2003-04 → 2025-26) |
| player-seasons in the panel | **11,707** |
| mean season-start roster coverage | **94.9%** |
| worst season | **2003-04** at **66.6%** — the only `tail_missing` row |
| worst usable season | **2011-12** at **90.8%** (the lockout, 2 games per team) |
| camp invitees never reaching a season-start roster | **2,515** |
| rows dropped: no `player_id` / exhibition opponent / at-or-after opener | **98** / **1,601** / **822** |

The four seasons that matter most for selection are all healthy: **96.9%** (2022-23),
**95.7%** (2023-24), **96.1%** (2024-25), **94.3%** (2025-26).

**The 822 dropped rows are one season and one event.** The NBA labels the July 2020 bubble
scrimmages `Pre Season` under the 2019-20 key — 33 games played 2020-07-22 → 2020-07-28,
against a 2019-10-22 opener. That is why the opener check ships as a **filter with a
count** rather than the assertion the plan specified: an assertion would simply fail, and
the number dropped is the thing worth watching.

**`coverage_class` is mechanical and rests on one measurement.** A preseason ends **3–5**
days before the opener in 22 of 23 seasons. 2003-04 ends **20** days out, which is the
signature of a truncated capture rather than a short preseason — and the tail is exactly
what `missed_tail` and the late-weighted shares read. So the class is `tail_missing` above
a 10-day gap and `covered` otherwise. 2004-05 (the final week only, 3 games per team) is
`covered`: its *head* is missing, which shifts a within-team share's denominator but
degrades gracefully, where a missing tail destroys the participation signal outright.
Interpreting a *short* preseason — 2011-12's lockout, 2020-21's COVID window — is not
derivable from the file and stays in `data-quirks.md` where knowledge that is not in the
data belongs.

### What the panel holds, and one thing it deliberately does not

One row per (player, season) for every player with **at least one preseason appearance**:
volume (`gp_pre`, `min_pre`, box totals) counted across every team, and everything
schedule-relative read over his **last** preseason team — the project's traded-player
convention. The forecast quantities are within-team **shares and ranks**
(`min_share_pre`, `min_rank_pre`, and their `_late` twins over each team's final
`late_games = 2`), participation (`missed_tail`, `missed_lead`, `played_final_game`), and
per-36 rates for the seven count heads plus `pre_fg3a_share`. Shares sum to exactly 1.0
within a team, which is the panel's own arithmetic check.

**Raw preseason MPG is not a headline column, and the 2023-24 panel says why.** The six
largest minutes shares that preseason belong to Ausar Thompson, Scoot Henderson, Jonathan
Kuminga, Brandon Miller, Tobias Harris and Tyrese Maxey — four of them rookies. Joel Embiid
played one game of Philadelphia's four. A preseason minutes *level* measures how much a
coach needs to look at a player, which is close to the inverse of what we are forecasting;
a within-team share at least normalizes the compression away.

**A player with no preseason appearance has no row.** The logs hold appearances, not
rosters — the same construction limit `src/features/availability.py` documents at length,
and it is not solvable here either. The `has_preseason` indicator and the zero-delta
convention belong to each head's attach step, where the join against that head's
population happens. What P0 owes that step is a measurement of who is missing, which is
the roster-coverage column above. **P1's missingness census is therefore not optional
colour** — a ~5% hole that is disproportionately veterans resting and stars injured is a
covariate, not noise.

## P1 — the EDA gate: shipped 2026-08-12

`make preseason-value` (`src/eda/preseason_value.py`) → `outputs/eda/preseason_value.csv`.
Three measurements, one decision, **on training seasons only** — the module never calls
anything that materializes validation, because a screen that spends the selection split
leaves P2 nothing to select on. Scope is the **22** seasons with an intact tail (2004-05 →
2025-26); intersected with train that is 2004-05 → 2021-22, fitted on 2004-05 → 2019-20 and
scored on **2020-21 and 2021-22**, so every ΔR² below is out of sample. The whole gate runs
in **11 seconds**.

### The finding that reframed the rest: a population, not a leak

The first run put the availability block at **+0.1171** ΔR² on `gp_share` — implausibly
large for six preseason games. It is not leakage, and chasing it produced the most useful
result of the session. The availability design holds every player who *appeared* in season
S, which includes players who signed in January; they have no preseason row and a small
`gp_share` for a contract reason rather than a health one. Restricting to the **season-start
roster** — the draft pool, the only population the head is ever applied to — takes the same
number to **+0.0198**. Both facts are knowable at the draft; only the second describes what
the block would buy where it would be used. The census is where the two separate cleanly:

| | share with no preseason row | gap in realized `gp_share`, absent − present |
|---|---|---|
| on a season-start roster (n = 6,278) | **3.8%** | **−0.122** |
| not on one (n = 716) | **51.3%** | **+0.024** |

Off-roster, a missing preseason predicts nothing at all. On-roster it costs 0.122 of a
season. **Every role and age cell is therefore read on the draftable frame alone**, because
pooled the two mix and a contract fact wears a health costume.

### (a) Redundancy — the rates are half-known, the minutes are not

Correlation with the prior-season equivalent, on the link scale the delta is taken on:

| | `pearson_r` | | `pearson_r` |
|---|---|---|---|
| `pre_fg3a_share` | **0.855** | `pre_per36_ast` | 0.725 |
| `pre_per36_reb` | 0.735 | `pre_per36_blk` | 0.656 |
| `pre_per36_fga` | 0.650 | `pre_per36_fta` | 0.538 |
| `pre_per36_tov` | 0.432 | `pre_per36_stl` | 0.378 |
| `mpg_pre` | 0.518 | `min_share_pre` | 0.358 |
| `min_rank_pre` | 0.372 | `min_share_pre_late` | 0.234 |
| `gp_share_pre` | **0.128** | | |

The shot-mix share is the most redundant quantity in the panel and preseason *participation*
is the least — 0.128 against a prior season is close to a different variable, which is the
first sign that the participation half of the block is where the information is.
`sd_delta` rides beside every row in the artifact and is the number that stops a high `r`
from being read as "no new information": `pre_fg3a_share` correlates at 0.855 and still
carries a delta sd of 1.44 on the logit scale.

### (b) Incremental signal — minutes win, games played is smaller than it looked

Out of sample, above each head's own prior-season block, with a within-season shuffled null
(20 permutations) giving the ΔR² a zero point:

| target | base R² | ΔR² | null mean ± sd | z |
|---|---|---|---|---|
| `minutes_per_game` (draftable) | 0.6643 | **+0.0492** | −0.0001 ± 0.0011 | **45.9** |
| `gp_share` (draftable) | 0.1115 | **+0.0198** | −0.0033 ± 0.0055 | 4.2 |
| `minutes_per_game` (all rows) | 0.6492 | +0.0213 | +0.0002 ± 0.0010 | 21.6 |
| `gp_share` (all rows) | 0.2632 | +0.1171 | −0.0021 ± 0.0026 | 46.2 |

**The two targets swap places under the restriction, and that is the result.** Games played
loses 6× when the mid-season signings come out; minutes per game *gains* 2.3×, because those
same rows were diluting it. So the plan's a-priori ordering — availability first, minutes
second — is **inverted by the measurement**: `min | available` is where the preseason block
is worth the most, by a factor of 2.5 on the population that matters, and it is carried
almost entirely by one column, `pre_d_mpg` at **+0.0519** alone (partial r 0.334).

The `gp_share` side is messier and worth stating plainly: the seven-column block scores
**+0.0198**, which is *less* than `pre_log_min` alone at **+0.0504**. Seven columns on this
target overfit a ridge, so P2's availability arm should be a smaller block than P1's, and
`pre_missed_tail_share` is the only column with the expected negative sign (partial r
−0.085) rather than a large effect.

### The rates bar — stated before the run, and cleared by five of seven

A count head ships against a no-fit floor it beats by **+0.0013** (`reb`) to **+0.0334**
(`stl`) of validation R² on the season total. The increment is measured the same way,
through `component_rates.fit_count_head`, so the comparison is in the bar's own unit rather
than in an R²-on-a-rate that would not be comparable:

| head | base R² | ΔR² | of which its own delta | of which the shared two columns | z vs null |
|---|---|---|---|---|---|
| `ast` | 0.8985 | **+0.0122** | +0.0123 | −0.0001 | 20.3 |
| `fga` | 0.9559 | **+0.0051** | +0.0054 | −0.0002 | 29.8 |
| `stl` | 0.8386 | **+0.0021** | +0.0023 | −0.0001 | 5.0 |
| `tov` | 0.9073 | **+0.0017** | +0.0019 | −0.0002 | 6.4 |
| `reb` | 0.9487 | **+0.0016** | +0.0036 | −0.0010 | 4.5 |
| `fta` | 0.9025 | −0.0016 | −0.0023 | +0.0002 | −2.3 |
| `blk` | 0.7892 | −0.0085 | −0.0065 | −0.0022 | −3.2 |

**So the rates question does not resolve as the null the plan expected.** Five of seven
count heads clear the bar, `ast` lands inside the shipped range at a third of its width, and
two heads are actively *hurt* by the block. The attribution columns are why this is
readable as a rate finding at all: for every count head the gain is its own preseason rate
and the shared reliability pair contributes ~0.000, so nothing here is `has_preseason`
wearing a rate's name.

The conversion heads are scored on their own R² and are a mixed bag —
`ftm|fta` **+0.0176** (delta-carried, z = 18.8), `fg3a|fga` +0.0047, `fg2m|fg2a` −0.0085 —
with one instructive row: **`fg3m|fg3a` reads +0.0149 and its own delta is −0.0024**, the
whole gain being the shared `has_preseason` pair. That is exactly what the attribution
split exists to catch, and it is recorded as a null on preseason three-point percentage.

### (c) The missingness census — the indicator needs splitting, on age

On the draftable frame, `has_preseason` is missing for 3.8% of rows and the outcome gap is
**not** flat:

| age | share missing | gap in `gp_share` | gap in MPG |
|---|---|---|---|
| `<24` | 3.0% | −0.116 | −3.03 |
| `24-27` | 3.5% | −0.091 | **−0.34** |
| `28-31` | 4.5% | −0.088 | −3.49 |
| `32+` | **4.7%** | **−0.247** | **−4.93** |

A 32-plus player with no preseason row loses a quarter of the season and five minutes a
night; a 24-to-27 player with no preseason row loses essentially no minutes at all. That is
the rested-veteran / injured-star split the plan flagged, measured — **one indicator is not
enough, and the axis it needs is age, not role**. By role the gap is comparatively flat
(−0.037 to −0.149) with no monotone pattern, so a role interaction is not indicated.

### What P1 decides

1. **Minutes moves ahead of availability.** P3 runs before P2. `pre_d_mpg` is the single
   most valuable column measured here and the marginal minutes head is the cheap
   point-MLE machinery.
2. **Rates are in, as a short list.** `ast`, `fga`, `stl`, `tov`, `reb` earn an arm on the
   count side and `ftm|fta` on the conversion side; `blk`, `fta`, `fg2m|fg2a` and
   `fg3m|fg3a` are recorded nulls and do not get one. This reverses the plan's expectation
   that the rate half would collapse, and it adds a P-numbered session rather than
   shrinking one.
3. **`has_preseason` is split by age** wherever it enters, from P2 onward.
4. **The availability arm's block is smaller than P1's seven columns** — the ridge overfits
   that target and `pre_log_min` alone beats the block.
5. **Every arm is scored on the season-start-roster population**, and any figure quoted
   without that restriction is a population statement rather than a model one.

### What P1 does not settle

The bar it clears is an R² screen on a point estimate. Every head in this project ships on
a **distributional** metric — CRPS, PIT, tail calibration — and `docs/availability-window-plan.md`
§12e and §14f record two blocks that won a validation reading and shrank 4–6× on the rolling
harness. P1 is a filter for what is worth fitting, not evidence that anything ships. The
bars in P2 and P3 below are unchanged by it.

**P4's population is sized here too, and it is large:** **3,385** of **11,399** in-scope
panel rows (**29.7%**) belong to players with no usable prior season, so measurement (b)
cannot see them at all. For those rows a preseason game is the first NBA observation rather
than a marginal update, which is the whole of P4's case.

## P3 — the marginal minutes arm: shipped 2026-08-13

`make minutes-preseason` (`src/models/minutes_preseason.py`) → `minutes_preseason.csv`,
`minutes_preseason_rolling.csv`, `minutes_preseason_shrinkage.csv`. Point MLE on the
`minutes-window` machinery, so no CmdStan; nine nested arms, two readings, and one gate.

**The gate passes on both halves, and the increment is the largest measured on this head.**
The declared primary arm reads validation CRPS **−4.789** minutes against the incumbent
with a paired-bootstrap interval of **[−8.08, −1.59]**, and the rolling-origin harness
agrees at **−7.940 [−9.41, −6.44]** on **12 of 13** origins. So `price_composition = True`
and the composition go/no-go opens.

### The rolling harness is *larger* than validation, which has not happened before here

Every previous block on this project shrank the other way — 4–6× on availability
(`availability-window-plan.md` §12e, §14f), and this head's own fitting-window axis went
from −2.687 on validation to −0.079 rolling with 6 of 13 origins
(`minutes-window-plan.md` §3). Here validation is **0.60×** the rolling reading rather than
5–30× it. The mechanism is visible in the harness: the rolling origins fit a mean of
**3,685** rows against validation's 6,152, and a preseason delta is worth more where the
prior-season block is weaker. That makes this the first block in the round whose validation
figure is the *conservative* one.

### The coverage restriction is load-bearing on this head, unlike on availability

The panel begins at 2004-05 and this head fits from **1997-98**, so **2,154 of 8,306**
training rows (25.9%) have no preseason row for a reason that is a fact about the NBA's API,
and the missing indicator would read as an era dummy on every one of them. This is exactly
the "coverage interactions" risk below, which the availability head's 2012-13 window dodges
and this head does not. So **every arm, the reference included, fits the covered window
only** — 6,152 rows — and the first covered season is read off `preseason_coverage.csv`
rather than hard-coded.

That restriction is not free and it is reported rather than absorbed: the full-window
incumbent reads CRPS **147.149** on the draftable rows against the covered-window
incumbent's **145.963**, so the cut is worth 1.19 minutes *before any preseason column
exists*. Crediting that to the block would have inflated the increment by a quarter.
Validation is fully covered (742 of 742), and **697 of 742** validation rows (93.9%) are
draftable, which is the population every figure here is quoted on per P1 decision 5.

### The ladder — on the season-start-roster population

Nine arms, all nested on the shipped `logit_own_spline` variant with shared ρ. `crps_vs_primary`
is a second paired bootstrap against the declared primary, so "arm X beats the primary" is
an interval rather than two point estimates read side by side.

| arm | cols | CRPS | vs incumbent [95%] | vs primary [95%] | PIT KS | MAE | bias | pred. sd |
|---|---|---|---|---|---|---|---|---|
| `carry_forward` *(floor)* | 0 | 163.871 | — | — | 0.1130 | 217.78 | +21.94 | 327.98 |
| `incumbent_full_window` *(context)* | 0 | 147.149 | — | — | 0.0691 | 204.60 | −16.19 | 309.59 |
| `incumbent` *(reference)* | 0 | 145.963 | — | +4.789 [+1.59, +8.08] | 0.0634 | 203.65 | −19.24 | 299.52 |
| `missing_only` | 4 | 145.861 | −0.101 [−0.94, +0.71] | +4.688 | 0.0752 | 203.49 | −14.89 | 298.17 |
| `share_late_only` | 5 | 145.643 | −0.320 [−1.99, +1.37] | +4.470 | 0.0636 | 203.04 | −14.38 | 295.90 |
| `own_delta_reliability` | 6 | 141.731 | −4.231 [−7.50, −1.04] | +0.558 [+0.13, +0.99] | 0.0579 | 200.77 | −38.15 | 283.22 |
| `own_delta` **(primary)** | 5 | 141.173 | **−4.789 [−8.08, −1.59]** | — | 0.0544 | 199.96 | −36.68 | 283.16 |
| `p1_block` | 11 | 141.531 | −4.432 [−7.75, −1.31] | +0.357 [−0.45, +1.19] | 0.0479 | 200.30 | −30.80 | 282.43 |
| `own_delta_shrunk` | 5 | 140.184 | −5.779 [−9.00, −2.73] | −0.989 [−2.04, −0.04] | 0.0608 | 198.25 | −40.54 | 283.04 |
| `mpg_scale` | 5 | 140.514 | −5.449 [−8.68, −2.32] | −0.659 [−1.33, +0.01] | 0.0536 | 198.67 | −32.08 | 282.83 |
| **`own_delta_centered`** | 5 | **139.379** | **−6.583 [−9.87, −3.48]** | **−1.794 [−2.96, −0.63]** | 0.0654 | 195.32 | **−10.95** | 282.63 |

**1. The block is the delta, and the delta alone.** `missing_only` — P1's age-split
indicator with no delta at all — is a tie at **−0.101 [−0.94, +0.71]**, and
`share_late_only` is a tie at **−0.320**. Whatever the preseason is worth on this head, it
is not "a preseason row exists" and it is not the within-team late share either. That is
the attribution P1 built for the rate family, run here on the head it moved to the front.

**2. More columns do not help.** P1's seven-column block scores **−4.432** against the
single delta's **−4.789**, an interval against the primary of [−0.45, +1.19] — a tie at
double the width. The gate's own reading predicted this (the block scored +0.0492 R²
against `pre_d_mpg`'s +0.0519 alone) and it reproduces at the head's own unit.

**3. The link matters less than the level.** `mpg_scale` is P1's `log1p`-MPG column
verbatim and the primary is the same quantity on this head's own logit-share link; they sit
0.66 CRPS apart with an interval touching zero. Difference coding on the head's link is the
plan's stated rule and it costs nothing to follow, but it is not where the result lives.

### The finding: preseason minutes are compressed, and the compression is a nuisance level

`own_delta_centered` — the same delta with **each season's own mean removed** — beats the
declared primary on validation at **−1.794 [−2.96, −0.63]** *and* on the rolling harness at
**−0.459 [−0.71, −0.21]**, 9 of 13 origins. And it does something no other arm does: it
**repairs the bias**. The primary predicts **−36.68** minutes low per player-season against
the incumbent's −19.24; centring takes that to **−10.95**, better than the incumbent
itself.

That is the compression P0 named, measured. A starter plays 15–20 preseason minutes, so
`logit(mpg_pre / 48) − logit(minutes_share_lag1)` is systematically negative and its
magnitude varies by season; a coefficient on the uncentred column is therefore part
player-specific update and part **league-level shift the point head has no season term to
absorb**. Centring separates them, and the whole of the level turns out to be nuisance.

**P0 pointed at the right problem and reached for the wrong instrument.** Its argument was
that "a within-team share at least normalizes the compression away", and that instrument —
`share_late_only` — is a **tie**. Centring the raw delta is what works. Both facts are worth
keeping: the diagnosis was right and the prescription was not.

**The promotion is a fitting-half decision, not a validation-driven swap.** `own_delta_centered`
was an attribution arm, so preferring it after seeing validation would be exactly the
mistake the split guards exist to prevent. It wins on the rolling harness too, on rows that
never touch the selection split, which is what licenses carrying it into the Stan port.
Its one cost is calibration: PIT KS **0.0654** against the primary's 0.0544 and the
incumbent's 0.0634, so it is the better-fitting and slightly worse-calibrated arm. Centring
is point-in-time — a season's own preseason mean is on disk before its opener — and a test
pins that it never pools across seasons.

### The shrinkage constant the plan left open, closed

`docs/preseason-plan.md` left the volume question open between an empirical-Bayes shrink and
a reliability interaction, and said the gate decides on train. Selected on an inner carve of
the **fitting half** (the last two training seasons scored, 5,416 fitted / 736 scored):

| k (preseason minutes) | 0 | **20** | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|---|
| inner CRPS | 132.152 | **132.106** | 132.357 | 132.372 | 132.948 | 133.421 |

**k = 20 wins by 0.05 CRPS minutes over no shrinkage at all**, and the curve is monotone
upward after it. So the shrink is real but negligible, and the honest reading is that the
volume question is a **null on this head**: at 4–6 preseason games the delta is already
being shrunk by the L2 penalty and the reliability weight has nothing left to do. P1's
additive alternative is worse than either — `own_delta_reliability` is the only arm that
*loses* to the primary with an interval clear of zero (+0.558 [+0.13, +0.99]).

### What the CRPS column does not show, and what it costs

**The predictive narrows 5.4%**, from **299.52** to **283.16** (283.16 for the primary,
282.63 centred). This head ships **solely** for its season-level spread — `make
minutes-unification` reads 302.75 against the composition's 64.65 — so an arm that improves
CRPS by narrowing the one thing the head exists to supply is not automatically an
improvement, and `predictive_sd` is reported rather than barred for that reason. The
context is reassuring but not settled: `minutes-window-plan.md` §4 found the shipped
`sim.minutes.player_season_sigma = 0.450` still tying a marginal arm at predictive sd
**265.67**, which is narrower than anything here. Re-running the injection stake against a
preseason-armed reference is P5 work and is not done.

**Central coverage degrades slightly.** Realized 50% coverage runs 0.5696 (incumbent) →
0.5481 (primary) → 0.5423 (centred) against a nominal 0.5, while 95% holds at 0.9555 →
0.9555 → 0.9527. The same shape `minutes-window-plan.md` §2 reports for its own arms: better
overall calibration, slightly too sharp through the middle.

### What P3 decides

1. **The preseason block ships onward on the minutes head** — both halves of the bar clear,
   and this is the first block in the project whose rolling reading is the larger one.
2. **The shipped column is the season-centred delta**, `pre_d_logit_share` with each
   season's own mean removed, plus P1's four age-split missing indicators. Five columns.
3. **The composition is priced**, per P3's own conditional — at the pilot window first.
   ✅ **Done 2026-08-14, session 4b**, and it passes: a preseason-blended `w_share` moves the
   head's own no-fit floor by **−0.19972 [−0.21661, −0.18225]** CRPS minutes per player-game
   on the draft pool, with no fit at all. The route that pays is the **offset** and not the
   ordering P3 named beside it. ✅ **Fitted in 4c** (retention 1.040) and ✅ **carried to the
   covered window in 4d**, where the increment reaches **−0.23418 [−0.24551, −0.22314]** and
   beats the head that actually ships by **−0.25409 [−0.26520, −0.24315]** per player-game.
4. **The volume shrink is a null.** `k = 20` is retained because it is the inner split's
   optimum, but it is worth 0.05 CRPS and nothing should be built on it.
5. ~~**Nothing ships into the chain from here.**~~ ✅ **Superseded the same day.** This is a
   point-MLE ladder and it earned the arm a Stan port on `stan_minutes` — and that port was
   built on 2026-08-13, so the block *is* in the chain. `stan.minutes.preseason: true`,
   `head_design` / `head_features`, fitting window cut to 2004-05, and `make posteriors
   --groups minutes` re-run behind it. The standing this decision gave the arm was the right
   one; what it got wrong was the assumption that the port would be a later session's work.

### What P3 does not settle — ✅ half of it closed 2026-08-13

The port itself. The point MLE collapses the posterior over β to its mode, so a Stan fit
owes two things this cannot say: whether the increment survives integrating over coefficient
uncertainty, and what the block does to the season-level predictive **jointly** with the
year effect `season_terms` already gives this head — the centring finding is a statement
about a league-level level, and a year effect is the term that would otherwise own it. The
two may be partly redundant, and nothing here has crossed them.

✅ **The first is closed and the answer is that the increment grows.** `make stan-minutes`
now fits a `logit_own_spline__no_preseason` control — the shipped variant on the same 6,152
covered rows with the five columns removed, so the block is isolated from the window cut that
arrived with it. Validation CRPS **136.958** with the block against **142.869** without:
**−5.911 minutes integrated over `beta`**, against **−4.789** at the point MLE. That is the
same direction as the rolling/validation comparison above — this block reads *larger* the
more evidence is brought to it — and it is the second reason to treat P3's validation figure
as the conservative one.

⚠️ **The second is still open, and it grew a consequence.** The season-level ρ moved
**0.05025 → 0.041894** with the block, ~9% narrower. This head ships *for* its season-level
spread, so `sim.minutes.player_season_sigma = 0.450` — calibrated against the pre-block head
in `minutes-window-plan.md` §4 — is stale, and the composition-vs-marginal stake needs
re-reading before the chain is trusted. That is P5 work and it is not done.

## P2 — the availability arm: measured 2026-08-13, and **the gate fails on the reading that cannot resolve it**

`make availability-preseason` (`src/models/availability_preseason.py`) →
`availability_preseason.csv`, `availability_preseason_rolling.csv`,
`availability_preseason_effects.csv`, `availability_preseason_block.csv`. Point MLE on the
`availability-window` machinery, so no CmdStan; eight nested arms on the head that **ships**
— the two-component `mixture`, `three_point_era` window, no season term, role-graded ρ.

**The bar was written into this document before the round ran** and is §14's
`wins_crps_holds_boundary`: a validation CRPS paired-bootstrap interval clear of zero with
`boundary_tail_error` **held**, on the draftable population, *and* the rolling-origin harness
agreeing. It is a conjunction, and **one half of it passes**:

| the declared primary arm, on the draftable population | CRPS vs the shipped head | boundary margin | verdict |
|---|---|---|---|
| **validation** — 772 rows | **−0.1020 [−0.3223, +0.1215]** | −0.00684 [−0.01036, −0.00021] | ❌ CRPS spans zero |
| **rolling origin** — 3,575 rows, 10 origins | **−0.2537 [−0.3444, −0.1620]**, **10 of 10** | −0.00289 [−0.00350, −0.00224] | ✅ both halves |

**So the gate fails, and `earns_stan_port` is `False`.** That is the letter of the bar and it
is what the artifact records. But the *reason* the bar was written as a conjunction is stated
in this document — "the last two blocks on this head won validation and shrank 4–6× rolling
(§12e, §14f), so validation alone ships nothing" — and it anticipated the opposite failure.
There is no clause in it for an arm that replicates on the fitting half and cannot be
resolved on validation, which is what happened. **Whether to widen the bar is a decision this
round does not get to take**, because a bar rewritten after seeing which side of it an arm
landed on is not a bar. It is logged as an open decision below.

### This is not §12e or §14f, and the diagnostic that separates them is the interval width

§14f's block shrank 4.2× on the rolling harness and the round called it "not a power problem"
on a specific basis: the rolling interval was **narrower** than the validation one on 3.3× the
rows, so what got smaller was the effect and not the resolution. That test run on this round
gives the opposite answer at every step:

| | §14f's absence block | P2's preseason block |
|---|---|---|
| validation → rolling | −0.0575 → **−0.0136** (shrinks 4.2×) | −0.1020 → **−0.2537** (grows 2.5×) |
| origins won | 4 of 7 | **10 of 10** |
| rolling interval half-width | **narrower** than validation | 0.0912 against **0.2219** — 2.4× narrower on 4.6× the rows |
| is the other reading's estimate inside this one's interval? | — | **yes**: validation's [−0.3223, +0.1215] contains −0.2537 comfortably |

The last row is the one that matters. The two readings do not **disagree** — validation's
interval covers the rolling point estimate with room to spare. Validation *cannot resolve*
an effect this size on 772 rows, and the reading with 4.6× the rows and 2.4× the precision
sees it on every single origin. `shrinkage_vs_validation` is **0.402** in the artifact,
which is the second time in this round that a preseason block reads *larger* on the fitting
half — P3's was 0.60× — and the first time it changes a verdict.

### The headline is still the population, and it reproduces at both readings

| `mixture__volume` | pooled | draftable |
|---|---|---|
| validation CRPS vs the shipped head | **−0.636 [−0.900, −0.390]** | **−0.102 [−0.322, +0.121]** |
| rolling CRPS vs the shipped head | **−0.721 [−0.830, −0.616]** | **−0.254 [−0.344, −0.162]** |
| pooled ÷ draftable | **6.2×** | **2.8×** rolling |
| the age-split indicator **alone**, pooled | −0.284 (45% of the margin) | −0.367 rolling (51%) |
| the age-split indicator **alone**, draftable | +0.001 [−0.152, +0.174] | −0.040 [−0.106, +0.023], 6 of 10 |

Read pooled, the block is the largest CRPS margin any covariate block has posted on this head
— **11.0×** §14's absence block on validation — and it would clear the gate on either
reading. On the season-start roster it is between a third and a sixth of that. P1 measured
the same restriction as **5.9×** on a ridge ΔR² (+0.1171 → +0.0198); three instruments at
three units now agree that most of a pooled preseason reading is population.

**The attribution says exactly where it goes, and it replicates.** `missing_only` — the four
age-split indicators P1's census decided on, carrying no preseason quantity at all — is worth
**45%** of the pooled validation margin and **51%** of the pooled rolling one, and is a tie on
the draft pool at both readings (+0.001 on validation, −0.040 at 6 of 10 origins). "He has no
preseason row" is a strong predictor of a short season among everyone who appeared in season
S, because most such players signed in January; among players who were on a roster in October
it is close to nothing. The block is **267** training log-likelihood points for five columns
— **14.5×** the absence block's 18.48 — and half of what it fits is a fact about who is in
the frame.

**So the population restriction is load-bearing at three units** (`preseason_value.csv`'s ΔR²,
this head's CRPS, and this head's calibration), and P1 decision 5 has earned its keep: a
figure quoted on the pooled frame here would have reported a decisive win at both readings
for a block that is worth a third of that where it would be used.

### The shipped head's low-tail error changes **sign** between the two populations

The shipped head's `boundary_tail_error` is **0.01085** pooled on validation — reproducing
§14c's 0.0109 to four decimals, which is the control that licenses reading this round against
that one — and **0.01998** on the draftable rows. The decomposition says why, and the sign is
the finding rather than the level:

| `mixture`, error in P(GP < 10) | validation | rolling origin |
|---|---|---|
| pooled | **−0.0132** (under-predicts) | **−0.0123** (under) |
| draftable | **+0.0323** (over-predicts, 2.25×) | **+0.0179** (over) |

**The pooled figure is the average of two opposite errors, and that replicates.** Among
everyone who appeared, the head under-predicts the dead season; among players who were on a
roster in October it over-predicts it — on validation it assigns 5.8% of the draft pool a
season under ten games where 2.6% realize one. §1's warning about cumulative thresholds
letting opposite-sign errors cancel inside a tail, one level up: here they cancel across
*populations*.

**The level does not replicate as cleanly and is quoted with that caveat.** `boundary_tail_error`
is 1.84× larger on the draft pool at validation and **1.14×** on the rolling harness, because
the pooled *upper* boundary error is much larger there (+0.0201 against +0.0190). The robust
statement is the sign flip in the low tail; "1.84×" is a validation reading, not a constant.

That is also the mechanism for the calibration gain. The block tells the head who was actually
on the floor in October and spends it almost entirely on the low tail: on validation P(<10)
0.0582 → **0.0480** against 0.0259, and P(full schedule) 0.0387 → **0.0304** against an
observed 0.0311, which is very nearly exact. Realized 50% coverage goes 0.6386 → 0.5816
against a nominal 0.5, so the block sharpens a predictive that was much too wide on this
population. PIT KS goes 0.0927 → 0.0853, and to **0.0481** on the centred arm — against
**0.0631** for the shipped head on the pooled frame, which is the same population gap once
more.

**What is not measured, and it matters:** whether the single-component `betabinom` is also
worse on the draft pool. §7 selected `mixture` on a *pooled* boundary reading (0.0201 →
0.0109), and if that gain does not transfer to the population the head serves, the selection
rests on a frame the head is never applied to. This round cannot say — it fits no
single-component arm — and the two fits that would settle it are logged in
`docs/potential-to-dos.md` item 9 rather than run here, because re-deciding a shipped head is
not what a preseason round is for.

### The ladder — on the season-start-roster population, both readings

Eight arms, all nested on the shipped head (`theta = 0` nests at any width of `π`, so
`assert_nests` reads **0.0** on all eight). Observed on the 772 validation rows: P(<10)
**0.0259**, P(full) **0.0311**.

| arm | β | π | train ll | CRPS | vs `mixture` [95%] | PIT KS | **boundary** | vs `mixture` [95%] | body | shoulder |
|---|---|---|---|---|---|---|---|---|---|---|
| `mixture` *(reference — what ships)* | 0 | 0 | −16,174.52 | 8.9730 | — | 0.0927 | 0.01998 | — | 0.0386 | 0.0201 |
| **`volume`** *(primary)* | 5 | 0 | −15,907.17 | 8.8710 | **−0.1020 [−0.3223, +0.1215]** | 0.0853 | **0.01140** | **−0.00684 [−0.01036, −0.00021]** | 0.0514 | 0.0121 |
| `volume_centered` | 5 | 0 | −15,918.41 | **8.7928** | −0.1802 [−0.4190, +0.0611] | **0.0481** | 0.01432 | −0.00594 [−0.00986, −0.00413] | **0.0165** | 0.0184 |
| `p1_block` | 10 | 0 | −15,875.40 | 8.8264 | −0.1465 [−0.3923, +0.0997] | 0.0744 | **0.01017** | −0.00822 [−0.01124, −0.00241] | 0.0448 | 0.0089 |
| `participation` | 8 | 0 | −15,898.20 | 8.8807 | −0.0923 [−0.3221, +0.1354] | 0.0815 | 0.01083 | −0.00739 [−0.01037, −0.00143] | 0.0475 | 0.0101 |
| `volume_pi` | 5 | 7 | −15,877.84 | 8.8540 | −0.1189 [−0.3563, +0.1175] | 0.0789 | 0.01354 | −0.00529 [−0.00951, **+0.00193**] | 0.0432 | 0.0109 |
| `missing_only` | 4 | 0 | −16,062.21 | 8.9743 | +0.0013 [−0.1524, +0.1741] | 0.0718 | 0.01691 | −0.00313 [−0.00422, −0.00213] | 0.0254 | 0.0203 |
| `pi` | 0 | 7 | −16,030.10 | 9.0401 | +0.0671 [−0.1880, +0.3371] | 0.0533 | 0.01363 | −0.00601 [−0.00773, −0.00325] | 0.0172 | 0.0242 |

And the same eight on the rolling harness — 10 origins, 3,575 draftable player-seasons, every
row of it a fitting-half row:

| arm | CRPS | vs `mixture` [95%] | origins won | PIT KS | **boundary** | vs `mixture` [95%] |
|---|---|---|---|---|---|---|
| `mixture` *(reference)* | 9.1806 | — | — | 0.0503 | 0.01846 | — |
| **`p1_block`** | **8.8291** | **−0.3514 [−0.4595, −0.2406]** | 9 / 10 | 0.0347 | **0.01416** | −0.00429 [−0.00511, −0.00351] |
| `volume_centered` | 8.8823 | −0.2982 [−0.4058, −0.1976] | 9 / 10 | **0.0287** | 0.01716 | −0.00129 [−0.00198, −0.00058] |
| `participation` | 8.9176 | −0.2630 [−0.3684, −0.1593] | 9 / 10 | 0.0414 | 0.01488 | −0.00357 [−0.00432, −0.00285] |
| **`volume`** *(primary)* | 8.9268 | **−0.2537 [−0.3444, −0.1620]** | **10 / 10** | 0.0470 | 0.01556 | −0.00289 [−0.00350, −0.00224] |
| `volume_pi` | 8.9293 | −0.2512 [−0.3517, −0.1503] | **10 / 10** | 0.0399 | 0.01448 | −0.00398 [−0.00457, −0.00339] |
| `missing_only` | 9.1405 | −0.0401 [−0.1063, **+0.0229**] | 6 / 10 | 0.0349 | 0.01690 | −0.00156 [−0.00193, −0.00117] |
| `pi` | 9.1444 | −0.0362 [−0.1374, **+0.0715**] | 7 / 10 | 0.0297 | 0.01500 | −0.00346 [−0.00402, −0.00291] |

⚙️ **And the same eight against the *primary* rather than the reference — the column this
round did not have, added and re-run later the same day** (`potential-to-dos.md` item 10).
It is what settled which arm ships, and it is a different question from the table above,
where every interval is measured against a common reference and therefore says nothing about
whether two arms differ from each other:

| arm | CRPS vs `volume` [95%] | origins won | **boundary** vs `volume` [95%] |
|---|---|---|---|
| **`p1_block`** | **−0.0977 [−0.1569, −0.0408]** | **8 / 10** | **−0.0014 [−0.0019, −0.0010]** |
| `volume_centered` | −0.0445 [−0.0919, **+0.0008**] | 7 / 10 | **+0.0016 [+0.0012, +0.0020]** |
| `participation` | −0.0093 [−0.0509, **+0.0358**] | 6 / 10 | −0.0007 [−0.0010, −0.0003] |
| `volume_pi` | +0.0025 [−0.0313, **+0.0349**] | 6 / 10 | −0.0011 [−0.0014, −0.0008] |
| `missing_only` | +0.2136 [+0.1458, +0.2769] | 0 / 10 | +0.0013 [+0.0008, +0.0019] |
| `pi` | +0.2175 [+0.1429, +0.2980] | 0 / 10 | −0.0006 [−0.0014, **+0.0002**] |

**Two arms move and the rest are ties.** `p1_block` clears zero on both margins, which is
P3's promotion rule and is why it is the arm that ships — **reversing P1 decision 4**.
`volume_centered` is a tie on CRPS and **worse on the boundary with an interval clear of
zero**, which is the half of the bar the block actually cleared; it was adopted for
robustness to a truncated preseason and withdrawn the same day when that premise failed, and
this row is why nothing was lost by withdrawing it. Everything else is indistinguishable
from the declared primary, including the seven `π` columns one more time.

**1. The two readings agree about everything except resolution.** The same three arms lead
both tables, the same two arms (`missing_only`, `pi`) are ties on both, and every ordering
that matters is preserved. What changes is that on the fitting half five of the seven arms
clear zero on CRPS and on validation none of them does.

**2. Every arm improves the boundary at both readings.** All seven arms that carry a preseason
column move it the right way on validation (six clear zero) and all seven clear zero on the
rolling harness. On this head the preseason is a **calibration** input first and an accuracy
one only where there are enough rows to see it.

**3. The `l2` confound points the wrong way for the `π` arms and does not rescue them.** The
penalty reaches `beta[1:]` only, so the two `π` arms carry 7 unpenalized parameters the
shipped head does not (§14b's arithmetic, one block wider), and a fixed penalty therefore
favours them. `pi` is a tie at both readings and the worst CRPS row on the validation ladder.

**4. P1 decision 4 is not confirmed, and on the fitting half it points the other way.** The
decision said the availability block should be *smaller* than P1's seven columns because a
ridge overfit them. On validation the wider blocks are indistinguishable from the single
column (P1's whole block **−0.0445 [−0.128, +0.039]** against the primary, participation
**+0.0097 [−0.029, +0.049]**); on the rolling harness `p1_block` has the best CRPS *and* the
best boundary of any arm. ~~**This round cannot settle it**, because the rolling table carries
a paired interval against the *reference* and not against the *primary* — the
`crps_vs_primary` column P3's harness has and this one does not.~~ ✅ **The column was added
and the harness re-run later the same day**, and it settles the decision the other way:
`p1_block` beats the primary at **−0.0977 [−0.1569, −0.0408]** on CRPS and **−0.0014
[−0.0019, −0.0010]** on the boundary, 8 of 10 origins. **P1 decision 4 is reversed and the
wider block ships.** The gap was in the instrument rather than in the world, which is exactly
what `potential-to-dos.md` item 10 said it was.

### The block does not belong on `π`, and this time the boundary says so

§14d found the absence composition helping `β` and costing CRPS on `π`, and predicted that a
block with a genuine claim on disruption risk might not behave that way. The preseason has
that claim — "who missed the tail of the preseason, days before the opener" is a direct
reading of who is about to lose the season — and it behaves the same way:

| effect of adding the block to `π`, given it is already on `β` | delta [95%] | |
|---|---|---|
| CRPS | −0.0169 [−0.0565, +0.0270] | nil |
| **`boundary_tail_error`** | **+0.00214 [+0.00025, +0.00267]** | **clears zero in the WRONG direction** |
| `body_error` | −0.00824 [−0.00920, −0.00717] | better |
| `shoulder_error` | −0.00122 [−0.00179, +0.00511] | nil |

The rolling harness agrees from the other side: `volume_pi` reads −0.2512 against `volume`'s
−0.2537, so on 3,575 rows the seven `π` columns are worth **+0.0025 CRPS**, which is nothing.
And the `π`-only arm is a tie at both readings while carrying the validation ladder's
second-best PIT KS (0.0533) and a much better body — §14d's "buys fit, gives back
generalization", reproduced by a different block on the same parameter. It is not a small
change either: `θ` goes 0.1116 → **0.7453**, mean `π` 0.0445 → 0.1524 and the low component's
mean 0.1000 → **0.2869**. The arm becomes a substantially different mixture and predicts no
better.

**So `PI_COLS` stays at eight columns for the second time, and the second refusal is the more
informative one.** §14's block was about last season; this one is about the fortnight before
the opener, which is the strongest a-priori case a covariate could have for that parameter,
and it still does not belong there.

### Centring replicates across heads, and it is the arm to watch

P3's finding was that a preseason column carries a **season-level nuisance** the point head
has no year term to absorb. It reproduces here on a level rather than a delta, and the
diagnostics measure the level directly: across the ten fitting seasons the mean of
`pre_log_min` runs **3.795 → 4.681** (sd **0.310**) against a within-season sd of **0.616**,
so the calendar moves the column by half its cross-player spread. `team_pre_games` runs
**2 to 8** in the window — the 2011-12 lockout against an ordinary year — which is the same
quirk `pre_missed_tail_share` is a share to survive.

Centred, the arm has the best CRPS on the validation ladder (**8.7928**, −0.078
[−0.166, +0.013] against the uncentred primary), the best PIT KS at **both** readings
(**0.0481** and **0.0287** against the shipped head's 0.0927 and 0.0503) and a validation
`body_error` of **0.0165** against the uncentred arm's 0.0514 — a margin of **−0.0350
[−0.0359, −0.0114]**, clear of zero. It gives back part of the boundary (+0.0029 on
validation, and it is the only arm whose *pooled* rolling boundary margin is positive).
**The uncentred level is what wrecks the body**, which is §4's standing warning firing at the
primary arm rather than at some straw man: `volume` buys the boundary and pays for it in the
middle, and centring is what stops it doing so.

### What P2 decides

1. **The gate fails as stated and nothing is ported.** Validation's CRPS interval spans zero,
   the bar is a conjunction, and a bar re-read after seeing which side an arm landed on is
   not a bar.
2. **But the failure is on the underpowered half, and the round says so with a number.** The
   rolling reading is 2.5× larger, 2.4× more precise, wins 10 of 10 origins, and sits
   comfortably inside validation's interval. This is the **inverse** of §12e and §14f, and the
   interval-width diagnostic those rounds used is what distinguishes the two cases.
3. **Whether the bar should have a clause for this is an open decision, and it is not this
   round's to take.** It is registered as such rather than resolved. What would settle it
   without widening anything is more scored validation seasons, and those are the test split.
4. **The preseason is a calibration input on this head at every reading**, where on minutes it
   was an accuracy one — the block improves `boundary_tail_error` on all seven arms at both
   readings and never improves validation CRPS with an interval.
5. **Every preseason figure on this head is quoted on the draftable population.** The
   pooled/draftable ratio is 6.2× on validation and 2.8× rolling, and the pooled reading
   passes the gate at both. P1 decision 5 is the most load-bearing decision in the round.
6. **`PI_COLS` stays at eight columns.** Two blocks, two refusals, and this one had the
   stronger prior.
7. ~~**P1 decision 4 is unresolved rather than confirmed**, and the instrument that would
   settle it — a `crps_vs_primary` column on the rolling table — is missing from this round's
   harness.~~ ✅ **Settled later the same day, and it is a reversal.** The column was added and
   the harness re-run; `p1_block` beats the declared primary at **−0.0977 [−0.1569, −0.0408]**
   CRPS and **−0.0014 [−0.0019, −0.0010]** boundary, 8 of 10 origins, so the **wider** block
   ships and P1 decision 4 is withdrawn.
8. **The shipped head's low-tail error changes sign between populations**, at both readings,
   and whether §7's boundary selection survives that is `potential-to-dos.md` item 9.

### What ships — decided 2026-08-13, against the gate

The owner adopted the block on both heads with the failing half of P2's bar in view. The
reasoning is in `dashboard/decisions.py`; what follows is what is now wired, and every figure
below is from the shipped fits rather than from the round that measured the arms.

| head | block | fitted |
|---|---|---|
| availability (`stan_availability`) | **P1's full ten columns** — volume, three participation levels, two deltas, four age-split indicators | 0 divergences, max R̂ **1.00254**, MLE inside the 95% CI for **45 of 45** terms |
| minutes (`stan_minutes`) | the **season-centred delta** plus four age-split indicators, P3's shipped arm | 0 divergences, max R̂ **1.00695**, `logit_own_spline` still selected |

**The availability arm is the one the fitting half selected, not the one declared.**
`p1_block` beats the declared primary on the rolling harness at CRPS **−0.0977
[−0.1569, −0.0408]** and `boundary_tail_error` **−0.0014 [−0.0019, −0.0010]**, 8 of 10
origins — P3's promotion rule, and it **reverses P1 decision 4**. The instrument that
settled it is the `crps_vs_primary` column `potential-to-dos.md` item 10 asked for, added
and re-run in the same session.

⚠️ **One process failure is recorded here rather than quietly fixed, because it is the kind
this project has been bitten by before.** The availability head was ported *twice* on
2026-08-13. The first port, at 14:19, carried the **withdrawn** `volume_centered` arm — five
columns, 40 fitted terms — because `PRESEASON_COLS` was switched to the ten-column block at
14:25, six minutes *after* the artifacts were written. Nothing objected: the artifacts were
internally consistent, `make docs-audit` had no claim on the term count, and the figures
(max R̂ 1.0042, 40 of 40 terms) were written into this document and into
`dashboard/decisions.py` as if they described the shipped head. They were caught on
2026-08-14 by arithmetic — 45 terms are 20 β + 10 preseason + 4 ρ + 3 + 8 γ, and 40 is what
five columns give — and the head was re-fitted. **The general lesson is that an artifact
carries no record of which version of the code wrote it**, so a source edit that lands
between a fit and its documentation is invisible to every guard in the repo. The
`n_features` and `preseason_columns` fields `posteriors.py` persists exist for this; nothing
equivalent rides on the `make stan-*` metric artifacts.

**On the availability head's own port table the block is worth −0.695 CRPS games and +0.086
R² on the pooled 883 validation rows** (`docs/availability-plan.md`), and its largest single
effect is on global calibration: PIT KS **0.0643 → 0.0344**, which is the best any arm of
this head has posted and reverses the one metric the mixture used to lose on. Both of those
are **pooled** figures and P2 decision 5 governs them — on the draft pool the accuracy half
is a third to a sixth of it and validation cannot resolve it at all.

**P3's open question is closed.** `make stan-minutes` now fits a `logit_own_spline__no_preseason`
control — the shipped variant on the same covered rows with the five columns removed — so the
block is isolated from the window. CRPS **136.958** with the block against **142.869**
without: **−5.911 minutes integrated over `beta`**, against −4.789 at the point MLE. The
increment survives the posterior and is slightly larger under it.

**A complete preseason is now a production precondition.** Two of the availability block's
columns are read over the preseason's tail. A centred-volume arm that survives a truncated
capture was adopted and withdrawn the same day — see the registry — because the premise (DK
contests filling before the final preseason game) is contradicted by the 2025-26 draft. The
runbook's Oct 17–20 window is load-bearing rather than advisory.

**Both flags are exact rollbacks.** `stan.availability.preseason: false` and
`stan.minutes.preseason: false` restore the pre-2026-08-13 heads, and the minutes flag also
restores its full fitting window. Neither block can reach `availability_design` or
`build_design`, which eleven other consumers import; each head takes its own `head_design`
path and tests pin that.

### What P2 does not settle

Two things, and the first is the round's own verdict. **Whether the block should ship** is a
decision about the bar rather than about the evidence: the evidence is that it replicates on
every fitting-half origin and cannot be resolved on 772 validation rows.

And whether the calibration gain is worth having. This round measures CRPS, PIT and three
regional errors; what the contest cares about is whether a roster's Round-1 advance
probability moves, and §7l is the standing precedent that a head change reaching the draw as
*shape* rather than as *order* can be a measured null there. The instrument exists — `make
strategy-sweep` against a re-simulated season — and it costs the whole chain, which is P5
work.

## P4 — the no-prior population: measured 2026-08-14, and the two halves disagree

`make availability-no-prior` → `availability_no_prior_preseason.csv` (a) and
`make rookie-priors` → `rookie_priors.csv` (b). numpy only, seconds, no CmdStan and no fit.
The population is the **29.7%** of panel rows P1 sized — players with no usable prior season,
who are invisible to every fitted head in the project and reach the chain through two
draft-bucket imputations instead.

**The bar is §8b's own, stated in `availability-window-plan.md` before P4 ran**: a validation
CRPS paired-bootstrap interval clear of zero *and* the rolling origins agreeing. §8b's
`tenure_draft` cleared it at −4.5862 [−5.6200, −3.5764] on validation and −2.7413 rolling at
24 of 26 origins, so the standard the preseason arms are held to is one this ladder has
already met once.

### (a) The availability level — the preseason key is a tie on the population it would serve

Three arms added to §8b's ladder, all nested inside it: `preseason` (a within-team preseason
minutes-share bucket alone), `tenure_preseason`, and the declared primary
`tenure_draft_preseason` — P4's "preseason minutes-share bucket crossed with `tenure_draft`".
The share cuts are **a priori**, at 0.03 and 0.06 (half an even split of a ~17-man preseason
rotation, and an even one), and "no preseason row" is its own level rather than a small share.
Every arm including the reference is scored on the **covered window** only (2004-05 → 2023-24,
20 of the 27 seasons selection may read), which is P3's precedent one head over.

| `tenure_draft_preseason`, draftable | CRPS vs what ships | origins |
|---|---|---|
| **validation** — 145 rows, 2 origins | **+0.3356 [−0.5953, +1.3006]** | 0 of 2 |
| **rolling origin** — 1,278 rows | **−0.3148 [−0.5851, −0.0400]** | 8 of **14** |

**The gate fails, and unlike P2 it fails in the ordinary direction.** P2's diagnostic for
separating "underpowered validation" from "no effect" was the rolling reading's own strength —
there it was 10 of 10 origins at 2.4× the precision, and validation's interval contained it
comfortably. Here validation's interval does contain the rolling estimate, but the rolling
reading is itself weak: 8 origins of 14, and the point estimate on validation has the *wrong
sign*. Two readings that are both indecisive are not the same thing as one decisive reading
the other cannot resolve.

**Pooled, the same arm passes at both readings, and that is the fourth instrument to say so.**

| `tenure_draft_preseason` | pooled | draftable |
|---|---|---|
| validation CRPS vs what ships | **−0.5071 [−1.1888, +0.1475]** | +0.3356 [−0.5953, +1.3006] |
| rolling CRPS vs what ships | **−0.5874 [−0.7793, −0.3797]**, 11 of 14 | −0.3148 [−0.5851, −0.0400], 8 of 14 |

The census says why, in one line: on the draft pool this population realizes **0.5447** of the
schedule and **3.4%** have no preseason row; off it they realize **0.1571** and **38.6%** have
none. A preseason key on the pooled frame is largely detecting who signed in January. P1
measured that restriction as 5.9× on a ridge ΔR², P2 as 6.2× on the availability head's CRPS,
and this is the third unit and the fourth instrument.

**The draft bucket beats the preseason share, which reverses the prior this session was
opened on.** "A preseason minutes-share key is the natural next term of that series" is what
the risks section says; the arms that *drop* the draft bucket are the worst on validation —
`preseason` alone reads **+1.8115 [+0.3646, +3.4460]** against the shipped arm and
`tenure_preseason` **+1.7663**. Only the arm that keeps the bucket and refines it is even a
tie. For a first NBA appearance the draft slot is a 3.17× gradient (§8b) and the preseason is
a refinement of it, not a replacement.

⚠️ **And half the primary's rows never reach its key.** `graded_share` is **0.5407** on the
rolling origins: 24 cells against `MIN_CELL = 50` is more grading than ~1,800 rows of history
supports, so the arm falls back to `tenure_draft` on most rows and is *bit-identical* to it on
the first five origins. That is what `origins_compared` exists for — it was added in this
round, because reading "8 of 19" where the arm could only differ at 14 understates any graded
arm in exact proportion to how much history its key needs. A shrunk cell estimator would fix
it and is out of scope: §8b's ladder is deliberately *one* estimator with different keys, and
changing the estimator class makes the arms incomparable to the ones already selected from.

#### The axis P4 did not ask for: the estimator's population

§8b pools each rate over **every** no-design row before the target. The consumer,
`sim/season.no_design_availability`, is only ever applied to players on an October roster. So
the estimator has always been an estimate of the wrong population's rate, and the two realize
0.5447 against 0.1571. Restricting the pool is one filter, and it was crossed with the key
axis because P1 decision 5 makes the ladder unreadable otherwise.

| shipped `tenure_draft`, roster-pooled vs all-pooled, draftable | value |
|---|---|
| rolling CRPS | **−0.3486 [−0.6860, −0.0183]**, **13 of 19** origins |
| rolling bias | **−5.3904 → +1.7935** games |
| validation CRPS | **+0.7354 [−0.0336, +1.5067]**, 0 of 2 origins |
| validation bias | −0.3646 → **+6.4056** games |

**It repairs a large bias, wins more origins than any preseason arm, and still fails the same
gate** — and its failure is legible rather than mysterious: the last three rolling origins are
the ones it loses, and two of those three *are* the validation seasons. The pooled estimator's
low bias on 2022-23 and 2023-24 is a cancellation between a population error and an era drift,
not accuracy, but a bar is a bar. It is `potential-to-dos.md` **item 11** rather than a ship,
with the recency cross that would separate the two written down there.

**Nothing from (a) ships.** `sim.availability.no_design_level` stays `tenure_draft` on the
all-rows estimator, and `sim/season.no_design_level_arm` now refuses a preseason arm by name
rather than failing several frames later, since it builds its own keys and never joins the
panel.

### (b) The rookie priors — the preseason wins decisively, but only shrunk

`stan_composition.rookie_share_priors` is the `bio_draft_number` imputation P4(b) names: an
expanding-window mean of what past no-prior players in a draft bucket realized, which becomes
the player's `w_share` — the composition's prior minutes share and, through `order_frame`, his
place in the allocation order. **1,759** no-prior player-seasons over the covered window,
**1,204** of them draftable, **97.3%** of those carrying a preseason row.

Three arms over one estimator: the bucket mean, his own preseason reading of the same
quantity, and the two blended by preseason volume, `w = m / (m + k)`. `k = 0` is the preseason
arm exactly and `k → ∞` is the incumbent exactly, so the blend contains both endpoints. `k` is
chosen on an **inner carve of the fitting half** — the last two training seasons scored
against everything before them, `minutes_preseason`'s own construction — and lands at
**k = 20** preseason minutes for the share and **k = 160** for the rates.

| target | incumbent R² | `shrunk` R² | rolling MAE vs incumbent [95%] | origins | validation MAE vs incumbent [95%] |
|---|---|---|---|---|---|
| `per36_reb` | −0.0031 | **0.3326** | **−0.5863 [−0.6375, −0.5393]** | **19/19** | −0.4299 [−0.5783, −0.2780] |
| `per36_fga` | 0.0456 | **0.3282** | **−0.4221 [−0.4858, −0.3618]** | **19/19** | −0.3543 [−0.5548, −0.1584] |
| `per36_ast` | −0.0154 | **0.4357** | **−0.3567 [−0.3976, −0.3169]** | **19/19** | −0.2695 [−0.3880, −0.1560] |
| `per36_blk` | −0.0069 | **0.2621** | **−0.1223 [−0.1368, −0.1073]** | **19/19** | −0.0701 [−0.1266, −0.0166] |
| `fg3a_share` | −0.0430 | 0.4228 | **−0.0500 [−0.0534, −0.0466]** | **19/19** | −0.0518 [−0.0624, −0.0420] |
| `per36_fta` | 0.0191 | 0.0996 | −0.0340 [−0.0741, **+0.0045**] | 11/19 | −0.0189 [−0.1525, +0.1149] |
| `per36_tov` | 0.0148 | 0.0461 | +0.0013 [−0.0223, **+0.0255**] | 10/19 | −0.0135 [−0.0759, +0.0477] |
| `per36_stl` | −0.0392 | −0.0650 | +0.0107 [−0.0049, **+0.0256**] | 5/19 | +0.0160 [−0.0330, +0.0614] |
| `minutes_share` | 0.2532 | 0.3759 | −0.0069 [−0.0127, −0.0014] | 13/19 | +0.0040 [−0.0087, +0.0165] |

**1. The draft bucket is an anti-model for rates.** Its R² on the eight rate targets runs
**−0.043 to +0.046** — it is no better than predicting the population mean, and on five of
eight it is worse. §8b found the same shape one axis over ("a single rate was not a weak model
of these players, it was an anti-model") and the reason is the same: a draft slot says how good
the league thought a player was, not what he does per minute.

**2. Five of eight rate targets clear the bar on both readings** — `reb`, `fga`, `ast`, `blk`
and `fg3a_share`, each at **19 of 19** rolling origins with a validation interval clear of
zero. `fta`, `tov` and `stl` are nulls. That is the same 5-of-7-plus-the-mix shape P1's rate
gate found on the *veteran* population, arrived at on a disjoint population with a different
estimator, which is the strongest kind of agreement available here.

**3. The volume shrink is load-bearing here, where P3 found it a null.** Raw preseason is
*worse* than the incumbent on six of eight rate targets — `per36_stl` reads R² **−3.5798** —
because a per-36 over ~60 preseason minutes is mostly noise. It is the blend that wins.
P3 decision 4 recorded `k` as worth 0.05 CRPS minutes on the marginal minutes head and said
nothing should be built on it; the contrast is the finding rather than a contradiction. There
the player had a prior season and an L2 penalty already shrinking the delta, so the reliability
weight had nothing left to do. Here **the preseason is the only observation there is**, and
without a shrink toward a population mean it is unusable.

**4. `fg3a_share` is the one target the raw preseason wins outright** — R² **0.6407** and
MAE **−0.0930 [−0.1014, −0.0843]** against the shrunk arm's 0.4228 and −0.0500. Shot mix is
the most persistent quantity in the panel (P1: `pearson_r` **0.855**), so it needs the least
shrinking, and its own inner optimum is `k = 10` against the rate family's 160. One `k` per
family is the decision — eight targets each picking a rung off ~120 rows apiece would be
fitting the grid — and the per-target curve is written to the artifact as
`shrinkage_grid_by_target`, **reported and not selected on**, so that a family optimum wrong
for one member is visible rather than buried.

**5. The share — the only target with a live consumer — is the weakest row in the table.**
`minutes_share` is a tie on validation (**+0.0040 [−0.0087, +0.0165]**) and a marginal win
rolling (−0.0069 [−0.0127, −0.0014], 13 of 19), with R² moving **0.2532 → 0.3759** while MAE
barely moves. The preseason sharpens the *tail* of who will play and not the middle, and MAE
is the wrong instrument for that; but the bar is the bar, and this row does not clear it.

⚠️ **One unit error is recorded rather than quietly fixed**, because it produced a plausible
wrong answer. The share comparison was first run against the panel's `min_share_pre`, which is
a player's share of his **team's** preseason minutes (~1/17), while `minutes_share` is his
share of **game length** (~0.15–0.35) — a factor of ~5. The comparison reported the preseason
arm at R² **−1.586** with a bias of −0.2302, which reads exactly like the compression P3
documented and is instead a unit conversion. The matching statistic is `mpg_pre / 48`, which is
what `minutes_preseason` already builds its shipped delta from; a test pins it.

### What P4 decides

1. **(a) fails its gate and nothing from it ships.** The preseason key on the availability
   level is a tie on the draft pool at both readings, and the arms that drop the draft bucket
   are decisively worse. `sim.availability.no_design_level` stays `tenure_draft`.
2. **The population restriction is load-bearing at a fourth unit.** Pooled, the same arm
   passes at both readings; on the draft pool it is a tie. P1 decision 5 has now earned its
   keep three times.
3. **The draft bucket is a floor for the availability level and an anti-model for rates**, and
   P4 is where those two facts sit side by side. It survives (a) and is beaten 5 of 8 in (b).
4. **(b)'s rate result is the round's finding, and it is not directly shippable today**,
   because a no-prior player is not in the component heads at all — nothing downstream reads a
   rate for him. What it establishes is that if he is ever put in, his preseason per-36 is the
   prior to use and a draft bucket is not.
5. **The volume shrink is a null on a head with a prior season and load-bearing on one
   without.** Both readings stand; the axis is how much other evidence the delta is competing
   against.
6. **The estimator's pooled population is a real defect and is not this round's to fix.** It is
   worth −0.3486 CRPS at 13 of 19 rolling origins on the shipped arm and fails the same
   validation half; `potential-to-dos.md` carries it.

### What P4 does not settle

**Whether `MIN_CELL` is what defeated (a).** The primary grades only 54% of its rolling rows,
and an arm that cannot reach its key on half the population has not been given a fair test of
the key. Settling it needs a *shrunk* cell estimator rather than a hard fallback, which is a
different estimator class and therefore a different ladder — §8b's arms are one estimator with
different keys precisely so that its verdicts stay comparable.

**Whether (b) is worth anything in the contest.** Both halves reach the chain only through the
minutes allocation, and §7l is the standing precedent that a change arriving as *shape* rather
than as *order* can be a measured null there. §8b's own simulator readout is the template — it
moved season-total MAE by 3.18 dk_pts and the per-team no-design minutes share by 17.2% — and
running it costs the chain, which is P5 work.

## Session 4b — the composition, and the head where the preseason cannot be a feature

`make composition-preseason` (`src/models/composition_preseason.py`) →
`composition_preseason.csv`. Opened by P3's gate and specified at the **pilot window**
(2018-19 on, P3 decision 3). numpy only, **48 seconds, no CmdStan and no fit of the head** —
which is the design rather than a shortcut.

### Two of this head's three routes are unreachable by a coefficient

Every other block in this round is difference-coded columns on `beta`. `w_share` — a player's
prior-season minutes share, or his draft bucket's expanding mean if he has none — enters this
head **three** ways:

1. as a **feature**: `OWN = logit_share_lag1` is `logit(w_share)` and sits in every variant's
   feature list, so a coefficient *does* modulate it and the house pattern could add a
   preseason column beside it;
2. as the **offset**, through `sequential_columns` → `logit_prior`, the carry-forward `beta`
   only corrects; and
3. as the **allocation order**, through `order_frame`, which is the order the multinomial is
   decomposed into sequential binomials in.

**No coefficient reaches 2 or 3**, and those are what this round tests. So the arm is
`w' = ω·pre + (1 − ω)·w_share` with
`ω = m/(m + k)` over preseason minutes, where `k → ∞` is the incumbent exactly and a player
with no preseason row has `ω = 0` by his own volume rather than by a special case.

### Why it is scored on the floor, and why that is the right first gate

`FloorComposition`'s mean function is **the offset alone** — `predict_samples` sets `eta = 0`,
so route 1 is switched off and only routes 2 and 3 remain, and its one fitted quantity is a
shared dispersion. **That is the point rather than a limitation**: the floor isolates exactly
the two channels no coefficient can reach, with no sampler involved, and it is what every
fitted variant in `stan_composition`'s ladder is scored against. That makes this the
composition's version of P1 — a screen that can reject an arm before any of the head's 9.92 h
(full window) or ~1 h (pilot) is spent.

The control is that the incumbent arm reproduces the shipped floor. `carry_forward` reads
**4.6776** in `stan_composition`'s own ladder and **4.64939** here — the gap is the pilot
window and 120 predictive draws against 200, and both are named rather than absorbed.

### The gate passes decisively at the head's own selection unit

`k = 80` is chosen on an **inner carve of the fitting half** (2020-21 and 2021-22 scored
against everything before them), interior at 4.44562 against 4.49481 at k = 40 and 4.46834 at
k = 160. Validation's own optimum is **k = 160** — one grid step away, the same "two grids on
disjoint rows agree to a step" evidence `sim.minutes.player_season_sigma` rests on.

| at the selected `k = 80`, against the incumbent | draftable | pooled |
|---|---|---|
| **per player-game CRPS** | **−0.19972 [−0.21661, −0.18225]** | **−0.17130 [−0.18740, −0.15473]** |
| per player-season CRPS | −5.86215 [−15.16667, **+3.23419**] | −3.56609 [−11.41454, **+4.31123**] |

**Decisive at the unit `stan_composition` selects on, a tie at the season unit** — which is
this head's standing lesson (`make minutes-unification`) arriving from a third direction. A
better offset fixes the per-game *mean*; it does nothing for the season-level *spread*, which
is a missing parameter rather than a bad input, and the injected `σ` is what addresses that.

The level is the striking part. The floor goes CRPS **4.63787 → 4.43815** and R²
**0.4375 → 0.4797** on the draftable rows, with **nothing fitted**. For scale: the *fitted*
shipped composition reads **4.4945** on validation and the pilot-window fitted arm
(`potential-to-dos.md` item 1) reads **4.4561**. ⚠️ **Those are cross-artifact comparisons
with two known differences** — 120 predictive draws against 200, and the floor's shared `ρ`
against the head's role-graded one — so the honest statement is *suggestive* and not a
measured margin: a preseason-blended offset with no fit lands in the same neighbourhood as
the fitted head. Settling it is the Stan arm, which is what this gate licenses.

### The attribution: it is the offset, and the ordering is worth 4%

P3 named both routes — *"preseason minutes share updating the **ordering** and prior-share
feature"* — and only one of them pays. Run at the selected `k` with each route enabled alone,
on the draftable rows:

| route | per player-game CRPS vs incumbent | share of the blended arm |
|---|---|---|
| `both` *(the arm)* | −0.19972 [−0.21661, −0.18225] | — |
| **`offset_only`** | **−0.20591 [−0.22336, −0.18797]** | **103%** |
| `order_only` | −0.00749 [−0.01352, −0.00156] | **3.75%** |

The offset carries the whole margin — slightly more than the whole of it, so the two routes
are very mildly antagonistic — and reordering the allocation is clear of zero and worth
almost nothing. At the season unit `order_only` is a flat null (−0.35050 [−0.84174, +0.14873]).

**That is a useful negative for what gets built next.** The ordering is the expensive half to
change: it permutes the sequential decomposition and therefore the whole likelihood's block
structure, while the offset is one column. The cheap half is the half that works.

### Why the compression does not bite here, and this is the mechanism

P3's central finding on the marginal minutes head was that raw preseason minutes are
**compressed** — a starter plays 15–20 preseason minutes against 32–36 in the regular season —
and that the compression is a season-varying nuisance level the head had no year term to
absorb. Centring was needed there. **Nothing is centred here, and the raw blend wins by a
margin an order of magnitude larger than any centring bought.**

The reason is structural rather than lucky: `logit_prior` is built from the ratio
`w_k / tail_k`, a player's share against the *remaining* players' shares in the same
team-game. A common multiplicative compression of every `w_share` in a team cancels in that
ratio exactly. So this head is invariant to precisely the nuisance that forced P3's centring —
which is the composition-specific behaviour P3 predicted a win from, arriving through a
different mechanism than the one it named.

### What session 4b decides

1. **The gate passes at the pilot window and the composition earns a Stan arm.** The margin
   is at the unit the head selects on, on the draft pool, with the incumbent nested exactly
   and the inner carve one grid step from validation's own optimum.
2. **The shipped column is the blended `w_share` on the OFFSET.** The ordering route is worth
   3.75% and is the expensive half; it is recorded as a near-null rather than adopted.
3. **`k = 80`, from the fitting half.** Validation prefers 160 and reading that would be
   selecting on the split the arm is scored against.
4. ~~**The season unit is a tie and that is expected.**~~ ⚠️ **Half-withdrawn 2026-08-13 by
   session 4c.** A better offset cannot manufacture season-level heterogeneity —
   `sim.minutes.player_season_sigma` is the parameter for that, and it is unaffected by
   anything here. That reason stands and 4c confirms it from the other side, since the
   fitted arm's season predictive sd *narrows* (56.25 → 55.31). But the tie does not: once
   the head is **fitted** the same contrast reads **−14.62649 [−19.48503, −9.54922]**, 2.75×
   the floor's estimate, entirely through the **mean**. The floor could not see it because
   `eta = 0` switches off the feature route, and that gap compounds over ~82 games.
5. **The compression that forced P3's centring does not apply to this head**, because the
   offset is a within-team ratio. Recorded so that a future round does not reach for centring
   here by analogy.

### What session 4b does not settle

~~**The fit itself.**~~ ✅ **Closed 2026-08-13 by session 4c, and the increment grew.** The
floor is a screen, not a substitute: `beta` can correct an offset the floor cannot, so the
increment could have shrunk under a fitted head. It did the mirror of that instead, as P3's
own did — retention **1.040** per player-game, and **2.75×** at the season unit, where the
floor's tie does not survive the fit at all.

**Whether it survives the full window.** Everything here is at 2018-19 onward. The pilot is
what P3 specified and what `potential-to-dos.md` item 1 measured as ~6× cheaper, but the
shipped head fits from 1996-97 and the preseason panel starts at 2004-05, so a full-window
version needs the coverage cut P3's head needed.

**What it is worth in the contest**, which is P5 and costs the chain.

## Session 4c — the composition's arm under the posterior

`make composition-preseason-fit` (`src/models/composition_preseason_fit.py`) →
`composition_preseason_fit.csv`, plus a per-arm checkpoint and a diagnostics log. Two
pilot-window `stan-composition` fits of the **shipped** variant, ~15 min each, closing the
one thing 4b named as unsettled: *"the floor is a screen, not a substitute — `beta` can
correct an offset the floor cannot, so the increment could shrink under a fitted head."*

### The bar, stated before the run

Frozen in `composition_preseason_fit.report` as code rather than as prose, for the reason P2
records: a bar re-read after seeing which side an arm landed on is not a bar.

> The blended arm beats the **same-window control** at the **per-player-game** unit on the
> **draftable** population, with a paired-bootstrap interval clear of zero, **and** the team
> constraint stays exact (`team_sum_abs_error == 0` on every arm).

That is the screen's own bar read one layer up — the unit `stan_composition` selects on and
the population P1 decision 5 fixed. Two things are reported beside it and deliberately **not**
barred: the season unit, where 4b found a tie and where a better offset *cannot* manufacture
season-level spread, and the **retention** — the fitted increment as a fraction of the floor
increment. No prior for the retention was stated, so no threshold on it is either.

### The arms, and the two routes a fit opens that the floor could not

| arm | `w_share` on the offset | allocation order | variant |
|---|---|---|---|
| `base` | incumbent | incumbent | `betabinom_ot_graded` |
| `preseason` | blended at `k = 80` | **incumbent** | `betabinom_ot_graded` |

`base` is a **same-window control** and never the full-window incumbent —
`composition_effects`' rule, because a pilot-window arm ordering read against a full-window
baseline is uninterpretable. `route = offset_only` is 4b decision 2 and `k = 80` is decision
3; neither is re-derived here, and validation's own preference for `k = 160` stays unread
because it is the split the arm is scored against.

**Both frames' own no-fit floors are refitted in the same run at the same 200 predictive
draws.** 4b scored at 120, so quoting a retention across the two artifacts would put the draw
budget inside the ratio. Here it is a within-artifact quantity.

⚠️ **A fitted arm gives the blend two routes the floor switched off**, and they are why the
retention has somewhere to go on either side of 1.0. `FloorComposition` sets `eta = 0`, so
4b's screen isolated the offset and the ordering alone. Under a fit, `OWN = logit_share_lag1`
is the logit of the *blended* share and a coefficient does modulate it; and `RHO_BIN_COL` is
`w_share`, so the graded dispersion's bin edges are quantiles of the blended column too —
measured at **25.5%** of validation rows changing bin. Both are the right behaviour, since
the bins grade on whatever column orders the sequence and sets the offset. The consequence
is that the fitted increment is not the floor increment plus a coefficient.

### The gate passes, and the increment *grows* under the posterior

Two fits, 41 minutes, both clean: max R̂ **1.0047**, **0** divergences, **0** treedepth
saturation, ESS bulk 4,206 and 4,980. On the draftable population:

| arm | kind | per player-game CRPS | R² | MAE | PIT KS | ρ |
|---|---|---|---|---|---|---|
| `floor_base` | floor | 4.62079 | 0.43980 | 6.37115 | 0.02649 | — |
| `base` **(control)** | fitted | 4.44188 | 0.47096 | 6.28675 | 0.03874 | 0.10144 |
| `floor_preseason` | floor | **4.41998** | 0.48201 | 6.32305 | 0.04969 | — |
| **`preseason`** | fitted | **4.23305** | **0.51619** | **6.03748** | 0.04647 | 0.09129 |

**The bar clears.** The fitted increment is **−0.20883 [−0.22129, −0.19693]** CRPS minutes
per player-game on the draft pool, the team constraint holds exactly, and the floor
increment measured on the same frames at the same 200 draws is **−0.20081 [−0.21697,
−0.18411]**. **Retention 1.040** — so `beta` did not absorb the better offset, it kept it and
added a little, which is P3's direction (its own increment went −4.789 → −5.911 under the
posterior) rather than the shrink the section above was written to allow for.

**The control replicates an independent run.** Pooled, `base` reads **4.45596** against
`composition_effects`' separately-run pilot `base` at **4.45614** — four decimals, across a
doubled iteration count and a different driver. That is what makes the rest of the table
readable as a block effect rather than as run-to-run noise.

**And 4b's "suggestive" cross-artifact comparison is now a within-artifact one, and it
holds.** The *un-fitted* blended floor (**4.41998**) beats the *fitted* incumbent-offset arm
(**4.44188**) on the same rows at the same draw budget. A preseason reading of a player's
minutes share, with no sampler involved, is worth more on this head than fitting 25
coefficients to the prior-season one.

### The season unit reverses 4b decision 4 — and the reason it gave was right

| arm | season CRPS | MAE | bias | predictive sd |
|---|---|---|---|---|
| `floor_base` | 182.06350 | 208.66527 | −6.20158 | 57.47894 |
| `base` | 172.02457 | 198.60990 | −6.65162 | 56.24912 |
| `floor_preseason` | 176.74770 | 203.86927 | −10.65625 | 56.62615 |
| **`preseason`** | **157.39808** | **183.55718** | −7.85660 | 55.30525 |

4b found the season unit a **tie** (−5.31580 [−14.44243, **+3.83015**] here, reproducing its
screen) and said so was expected, *"a better offset cannot manufacture season-level
heterogeneity."* Under a fit the same contrast is **−14.62649 [−19.48503, −9.54922]** —
decisive, at **2.75×** the floor's point estimate.

**The stated reason survives; the conclusion drawn from it does not.** The predictive spread
does not widen — it *narrows*, 56.25 → 55.31 — so no season-level heterogeneity was
manufactured and `sim.minutes.player_season_sigma` is still the only parameter for that. What
moves is the **mean**: season-total MAE falls **15.05** minutes against the fitted control.
The floor could not show it because `FloorComposition` sets `eta = 0`, which switches off
precisely the route that carries it — `OWN = logit_share_lag1` is the logit of the *blended*
share, and a per-game offset improvement that a coefficient re-weights compounds over ~82
games in a way an un-fitted one does not. **This is the round's main finding: the floor is a
conservative screen at the per-game unit and a badly misleading one at the season unit.**

### What it costs, and what else moved

- **Calibration, slightly.** PIT KS **0.03874 → 0.04647** draftable and 0.03084 → 0.03484
  pooled — better-fitting and slightly worse-calibrated, the same shape P3's own centred arm
  showed on the marginal head.
- **The fitted dispersion falls**, ρ **0.10144 → 0.09129** (~10%). A better offset leaves
  less residual overdispersion for the beta-binomial to carry, which is the mechanism
  working as specified rather than a separate result.
- **Fitting is still worth something on top of the better offset**, which is the control the
  screen could not run: `fit_value_preseason` is **−0.18693 [−0.19713, −0.17651]** per
  player-game, against `fit_value_base`'s −0.17891. The arm has not improved the floor by
  making the head redundant.
- The `beta_binomial_lpmf: First prior sample size parameter[k] is 0` warmup rejections fire
  in both arms. They are **pre-existing** and belong to the shipped head, not to the blend —
  `minutes-composition-plan.md`, "Two warmup rejection classes", measured them at 28 per fit
  and equal between arms.

### What session 4c decides

1. **The blended offset survives the posterior and the arm earns the full window.** The bar
   was stated in code before the run and clears on both halves.
2. **The retention is 1.040 per player-game**, so the screen was neither optimistic nor a
   substitute — it was accurate at the unit it was read at.
3. **4b decision 4 is half-withdrawn.** The season unit is *not* a tie once the head is
   fitted; the argument that a better offset cannot manufacture spread stands and is
   confirmed by the predictive sd going the other way.
4. **`FloorComposition` is a conservative screen per game and an unreliable one per season.**
   Recorded for the next round that reaches for it: a floor whose mean is the offset alone
   cannot see anything a coefficient re-weights, and that gap compounds with aggregation.

### What session 4c does not settle

~~**The full window.**~~ ✅ **Closed 2026-08-14 by session 4d, and the increment grew again.**
Everything here is 2018-19 onward; 4d runs the covered window (2004-05 on, the cut P3's head
needed) with a third arm at 1996-97 to price the cut itself. The increment goes −0.20883 →
**−0.23418** and the retention 1.040 → **1.115**, and the cut turns out to be worth −0.01991
per player-game — the same direction as P3's, but **8.5%** of the increment rather than the
quarter that round paid. ⚠️ **And "at 9.92 h" was wrong**: that figure is the whole
four-variant `make stan-composition` sweep (2.01 + 2.60 + 2.50 + 2.67 h plus probe and
comparator), not one fit. A single fit of the shipped variant is ~2.5 h, and 4d's three came
to **7.76 h**.

**What it is worth in the contest**, which is P5 and costs the chain. Note this arm reaches
the draw as a change to the *allocation mean*, which is `order` rather than pure `shape` —
so §7l's standing precedent for a measured null there applies less cleanly than it does to
the availability block.

## Session 4d — the composition at the window the head actually fits

`make composition-preseason-fit` at `stan.composition.preseason.first_season: "1996-97"` and
`label: covered` → `composition_preseason_fit_covered.csv` (plus `_arms` and
`_diagnostics`). **Three** fits, 7.76 h, closing the one thing 4c named as unsettled: *"the
full window. Everything here is 2018-19 onward."*

The pilot's artifacts are untouched — a `label` namespaces this round's stem, because
`make docs-audit` re-derives ~45 of 4c's figures from the unlabelled one and a
covered-window run landing there would answer a different question under those claims'
names. That is `stan_composition_*.csv`'s own rule, one level up.

### The coverage cut, and the third arm it makes necessary

The preseason panel begins at **2004-05** and this head fits from **1996-97**, so the two
gate arms are cut to the covered window — P3's rule verbatim, since a missing-preseason
indicator on a pre-2005 row is an era dummy rather than a feature. The first covered season
is read off `preseason_coverage.csv` rather than typed, which matters by exactly one season:
2003-04 has 369 real preseason rows and is still excluded, because `covered_seasons` drops
it as `tail_missing` and two of this round's inputs are read over the tail.

**`base_full_window` is the arm that makes the round decidable**, and it is the one arm that
deliberately fits below coverage — which it can, because it carries no preseason column at
all. It is the shipped head. Without it this round could say the block beats a control
nobody runs, and not whether *covered window + block* beats what is on disk today. P3 needed
the same control and priced its own cut at **1.19 CRPS minutes before any preseason column
existed**, a quarter of that round's increment.

| arm | `w_share` on the offset | window | fitting rows |
|---|---|---|---|
| `base` | incumbent | 2004-05 on | 448,464 |
| `preseason` | blended at `k = 80` | 2004-05 on | 448,464 |
| `base_full_window` *(what ships)* | incumbent | 1996-97 on | 631,158 |

All three score the **same** 52,295 validation player-games over 4,920 team-games (47,726
draftable, 883 player-seasons), which is what makes every margin below a paired one.

### The gate passes, and the increment is larger than the pilot's

Two fits, three arms, **max R̂ 1.00436**, **0** divergences, **0** treedepth saturation, min
ESS bulk 2,457. On the draftable population, per player-game:

| arm | kind | CRPS | R² | MAE | PIT KS | ρ |
|---|---|---|---|---|---|---|
| `floor_base_full_window` | floor | 4.66767 | 0.43891 | 6.37591 | 0.04861 | — |
| `floor_base` | floor | 4.64212 | 0.43922 | 6.37499 | 0.03831 | — |
| `base_full_window` **(ships)** | fitted | 4.48615 | 0.46821 | 6.30560 | 0.04713 | 0.12592 |
| `base` *(control)* | fitted | 4.46624 | 0.46923 | 6.30314 | 0.04391 | 0.11479 |
| `floor_preseason` | floor | **4.43206** | 0.48163 | 6.32743 | 0.05349 | — |
| **`preseason`** | fitted | **4.23206** | **0.51899** | **5.97594** | 0.04557 | 0.10286 |

**The bar clears**: `fitted_increment` is **−0.23418 [−0.24551, −0.22314]** CRPS minutes per
player-game on the draft pool, and `team_sum_abs_error` is exactly **0** on all six arms.

**And it is 12% larger than the pilot's −0.20883, with retention rising rather than falling.**

| | pilot (4c) | covered (4d) |
|---|---|---|
| fitted increment, per player-game | −0.20883 | **−0.23418** |
| floor increment, per player-game | −0.20081 | −0.21006 |
| **retention** | 1.040 | **1.11485** |
| retention, per player-season | 2.75 | 2.69311 |

That is now the third time this block has read larger the more evidence is brought to it —
P3's own went −4.789 → −5.911 under the posterior, 4c's floor → fit went 1.040, and the
pilot → full window goes 1.115.

### What the coverage cut costs, and it is a *quarter* of what P3's did

| `window_cost` — `base` vs `base_full_window`, no block on either | draftable |
|---|---|
| per player-game | **−0.01991 [−0.02515, −0.01498]** |
| per player-season | −0.13348 [−1.15660, **+0.84972**] |

**The cut helps, and it is small.** Per player-game the covered window is better with an
interval clear of zero; at the season unit it is a tie. The direction matches P3 — that
round's cut was worth 1.19 CRPS minutes in the same direction — but the *proportion* does
not: there the cut was **a quarter** of the increment and here it is **8.5%** of it
(0.01991 against 0.23418). So on this head the block is not mostly window, and P3's warning
that "crediting that to the block would have inflated the increment by a quarter" does not
transfer. It was still worth measuring rather than assuming, which is the whole reason the
third arm exists.

`potential-to-dos.md` item 1 predicted this sign from a different direction — it found the
*pilot* window beating the full window on this head's own per-team-game metrics — and this is
the first paired interval on that axis rather than a two-point comparison at different
iteration counts.

### The comparison a ship decision turns on

| `ship_margin` — covered + block, against what ships today | draftable |
|---|---|
| per player-game | **−0.25409 [−0.26520, −0.24315]** |
| per player-season | **−17.27296 [−22.32569, −11.92164]** |

**Decisive at both units**, which the gate's own contrast is not obliged to be and 4b's floor
was not. It is reported rather than barred, deliberately: the bar was frozen in
`composition_preseason_fit.report` before 4c ran, and re-reading it after seeing which side
an arm landed on is what P2 records as not being a bar.

The decomposition is **exactly additive**, which is a useful arithmetic check on the three
arms: `window_cost + fitted_increment = ship_margin`, −0.01991 + −0.23418 = −0.25409 per
player-game and −0.13348 + −17.13948 = −17.27296 per season.

### The season unit, and 4c's main finding reproduced on 4.6× the rows

| arm | season CRPS | R² | MAE | bias | predictive sd |
|---|---|---|---|---|---|
| `floor_base` | 181.42923 | 0.86857 | 209.10307 | −6.15582 | 60.27473 |
| `floor_base_full_window` | 180.45159 | 0.86867 | 208.97601 | −6.24365 | 62.70130 |
| `floor_preseason` | 175.06503 | 0.89443 | 203.64302 | −10.64076 | 60.14598 |
| `base` | 171.31431 | 0.89170 | 199.34833 | −7.12869 | 59.57738 |
| `base_full_window` | 171.44780 | 0.89076 | 200.50048 | −7.00842 | 61.46343 |
| **`preseason`** | **154.17484** | **0.91014** | **181.64455** | −7.86942 | 57.63522 |

**The floor's season-unit tie does not survive the fit, again.** `floor_increment` at the
season unit is **−6.36419 [−15.31395, +2.68556]** — spanning zero, as 4b's did — while
`fitted_increment` is **−17.13948 [−22.06742, −11.97281]**, decisive at **2.69×**. Retention
is 1.11 per game and 2.69 per season on the same posterior. 4c called this the round's main
finding on 4 training seasons; it reproduces on 18.

**And it is the mean, not the spread.** Season-total MAE falls **17.70** minutes against the
control (199.35 → 181.64, against 4c's 15.05), while the predictive sd *narrows*
59.58 → 57.64. No season-level heterogeneity is manufactured, so
`sim.minutes.player_season_sigma` remains the only parameter for that and nothing in this
round touches it.

### Three smaller things, one of which is new

- **The un-fitted blended floor beats the fitted incumbent at *both* windows.** 4.43206
  against `base`'s 4.46624 and `base_full_window`'s **4.48615**. 4c made this comparison
  within-artifact against a pilot control; it now holds against **the head that actually
  ships**, which is the stronger form.
- **ρ moves in opposite directions on the two axes.** The block *lowers* it, 0.11479 →
  0.10286 (−10.4%, matching 4c's ~10%) — a better offset leaves less residual overdispersion.
  The longer window *raises* it, 0.11479 → **0.12592**, which is the pre-2005 seasons asking
  for more dispersion and is a second reading on the same era question item 1 opens.
- **Calibration gives back a little per game and gains per season**, the shape P3's centred
  arm showed: PIT KS 0.04391 → 0.04557 per game, 0.34450 → 0.32618 per season.

⚠️ **One artifact defect, recorded rather than fixed.** This round's `_diagnostics.csv`
carries **no provenance stamp**, because `stan_utils.diagnostics_frame` gained one *during*
the run and the sampling process had already imported it. Nothing about the fits is affected
— every edit made after launch was to other modules or to that one function's output columns
— but it is a live instance of exactly the defect the stamp was built for, and the next run
of any Stan head will carry it. Re-running 7.76 h to refresh a provenance column would cost
more than the column is worth.

### What session 4d decides

1. **The arm survives the window the head actually fits, and grows.** −0.23418
   [−0.24551, −0.22314] per player-game on the draft pool, against the pilot's −0.20883, with
   retention 1.115 against 1.040 and the team constraint exact.
2. **The coverage cut is worth −0.01991 per player-game, it points the same way as P3's, and
   it is 8.5% of the increment rather than a quarter.** The block is not mostly window on this
   head.
3. **Against the head that ships, the arm wins at both units** — `ship_margin` −0.25409
   [−0.26520, −0.24315] per game and −17.27296 [−22.32569, −11.92164] per season. That is the
   production comparison and it is reported, not barred.
4. **The floor remains a conservative screen per game and a misleading one per season**, now
   measured on 18 training seasons rather than 4: retention 1.11 against 2.69 on one posterior.
5. **Nothing is shipped by this round.** `stan.composition.preseason` still configures the
   *measurement* target only; no consumer reads it and `make stan-composition` is untouched.
   What adopting it would mean is a `first_season` of 2004-05 on the head itself plus
   `make posteriors --groups composition` at all three fit windows, which is P5.

### What session 4d does not settle

**What it is worth in the contest**, which is P5 and costs the chain. This arm reaches the
draw as a change to the *allocation mean* — `order` rather than pure `shape` — so §7l's
standing precedent for a measured null there applies less cleanly than to the availability
block, but "less cleanly" is not evidence.

**Whether the era question is about the block or about the head.** `window_cost` is the first
paired interval saying the pre-2005 seasons cost this head something, and ρ rising from
0.11479 to 0.12592 at the longer window is a second reading on it. Both are
`potential-to-dos.md` item 1's territory, and neither was measured *for* it.

## P5 — the composition's arm ships, and the chain is re-run behind it

**Decided 2026-08-14, an owner decision on session 4d's `ship_margin`.** Sessions 4b–4d
measured the arm and shipped nothing; this is where it becomes the head. Unlike P2 this is
not a decision taken against a failing gate — 4d's bar cleared, and the comparison a ship
turns on cleared with it:

| against `base_full_window` — the head that was shipping | draftable |
|---|---|
| per player-game | **−0.25409 [−0.26520, −0.24315]** |
| per player-season | **−17.27296 [−22.32569, −11.92164]** |

What made it an owner decision rather than an automatic consequence is the sequencing, not
the evidence. This head is the simulator's minutes source (`sim/season.py` draws through
`stan_composition.simulate_minutes`), so the arm reaches the tensor, the draft board and the
sweep — and P5's chain costs hours. Adopting *after* the chain would have meant running the
sweep twice, which is the one thing the handoff into this session asked to be decided first.

### The adoption is a separate door, and that is the whole of the design

The availability and minutes blocks are columns on `beta`, so their heads adopt them by
appending to a feature list. This head cannot: the preseason enters through `w_share`, which
reaches the model as the `OWN` feature, as the **offset** and as the **allocation order**,
and no coefficient reaches the last two. So the adoption is a change to the *frame builder*,
and a frame builder is exactly the thing eleven modules share.

`stan_composition.head_frame` is that door, and it is `stan_minutes.head_design`'s rule
verbatim one head over:

| through `head_frame` — consumes the shipped posterior | through `composition_frame` — builds its own model |
|---|---|
| `stan_composition.run` (fits and selects) | `composition_preseason` / `_fit` (the gate arms) |
| `posteriors.composition_artifact` (persists) | `composition_effects` (the σ_u ladder) |
| `sim/season.py` (draws) | `minutes_window` (the pre-fit rotation) |
| `minutes_unification`, `model_cards` (score / card it) | `rookie_priors` (P4(b)) |

**The right-hand column is the reason the door exists, and the failure it prevents is
specific.** Every gate arm in sessions 4b–4d is measured against a `base` control that
builds through `composition_frame` with no hook. A blend that reached that builder from
config would have turned those controls into blended arms silently, collapsing three
sessions of margins toward zero with nothing raising — the same class of failure as the
2026-08-13 double-port, arriving through a shared function instead of through a stale
artifact. A test pins it (`test_the_measurement_modules_never_see_the_configs_blend`).

### What is wired

- **`stan.composition.preseason.adopt: true`.** `false` is an exact rollback of *both* the
  blend and the window. `blend_k` and `route` are shared with the measurement target
  deliberately — they are 4b decisions 2 and 3, and the head and the round that measured it
  must not be able to disagree about them.
- **The window cuts itself to 2004-05**, read off `preseason_coverage.csv` by
  `head_first_season` rather than typed, and `max`-ed with the configured floor so a head
  already fitting inside coverage keeps its own window. `stan.composition.first_season`
  stays at 1996-97 and is now the floor rather than the answer — `stan.minutes`' pattern
  exactly. 4d priced the cut at **−0.01991 [−0.02515, −0.01498]** per player-game, *in the
  arm's favour*, so unlike P3's cut it is not a tax on the block.
- **The artifact records the blend.** `posteriors` writes `preseason_blend_k` and
  `preseason_route` into the composition artifact's extras, and `model_cards` **raises**
  rather than carding a blended posterior against an un-blended frame. That is the direct
  descendant of 2026-08-13: an artifact that does not record which arm wrote it cannot be
  told apart from one that did not, and here the consequence would be a card misreporting
  the offset, the allocation order *and* the dispersion bins at once.
- **Verified before any sampler time**: `route: offset_only` leaves `position` bit-identical
  across the two frames over all 736,410 rows, and the blend moves `w_share` on 538,685 of
  them. 1,828 tests pass.

### What the ship costs, and the ladder decision that rides with it

Adopting invalidates the record rather than the model. `make stan-composition`'s ~11 audited
figures — CRPS **4.4945** against the comparator's 4.7842, the role-graded ρ at **0.177** and
**0.085**, the **33.89**-minute team-total miss — describe an un-blended head at 1996-97 that
no longer exists once this lands.

**The owner's decision was to re-run the 9.92 h ladder rather than annotate them**, and the
reason it is not merely bookkeeping is that the sweep re-decides the *variant*: the four arms
are re-scored against an offset that has changed on 73% of rows, and `betabinom_ot_graded`
being selected again is a result rather than an assumption. The run is at the covered window,
so it fits 448,464 rows against the full window's 631,158 and should come in under the 9.92 h
that figure records. Its `probe_hours` Gate A guard is unchanged.

⚠️ **Every figure this document and `README.md` quote from `stan_composition_metrics.csv` is
therefore stale until that run lands**, and `make docs-audit` is the instrument that
enumerates them — roughly 40 claims across four docs. They are refreshed rather than
withdrawn, and the pre-adoption values are kept beside them per the house convention.

### The chain, run 2026-08-14 — and it is the cheap half

`make simulate-season` → `weekly-scores` → `bracket` → `draft-sim` → `strategy-sweep`, at the
`train` posterior window, which every P5 consumer pins because the rows being scored are
2022-23 and 2023-24 and `train_val` fits on them. **63 minutes end to end**, of which the
sweep was 53 — against 7.4 h for the ladder and 2.0 h for the posteriors. The sampler stages
are the cost of this round; everything after `make posteriors` is numpy.

⚠️ **The sweep took 11× the 4.8 minutes this project's own docs quote for it.** The scale is
unchanged (`sim.strategy.n_sims` is still 500, applied at both call sites) and a profiler put
it in `np.searchsorted` and `np.argsort` rather than anywhere pathological, so the likely
cause is the arms items 6–7 added while `strategy_*.csv` was left deliberately stale. That
figure should be re-measured rather than re-quoted.

#### Gate A improved on every season-total metric, in both seasons

The comparison is clean because `availability-window-plan.md` §8b holds the previous chain's
readout for the same arm:

| | 2022-23 before → after | 2023-24 before → after |
|---|---|---|
| season-total MAE | 397.36 → **363.234** | 398.45 → **377.510** |
| season-total CRPS | 276.48 → **252.524** | 275.17 → **260.387** |
| season-total R² | 0.6589 → **0.709385** | 0.6732 → **0.704896** |
| season-total bias | −26.50 → **−15.4388** | −71.15 → **−66.6411** |

**The bias was pre-existing and this round improved it**, by 11.06 and 4.51 dk_pts. That
matters because the bias is large against its −3.06 bar and it would have been easy to read
as something this round introduced. Two caveats on the bar itself, neither of which the round
resolves: it is computed on 873 pooled rows against the check's 386/387, and it is a
**full-season** figure while the simulator's target is the **tournament window** — 91% of the
schedule, and the front 91%. `potential-to-dos.md` **item 12** carries the one mechanism
measured for that gap, worth −5.92 and −7.29 dk_pts, which is 38% and 11% of what is there.

The weekly readout moved the same way — one-week validation MAE 28.72 → **27.6893**, CRPS
19.40 → **18.6603**, R² 0.4658 → **0.494243**, and every one of the five quoted per-week
biases toward zero.

#### The contest readout, which is what the round was spent on

| | before | after |
|---|---|---|
| 600k Shootaround simulated lift | 0.1890 | **0.2358** |
| 600k Shootaround realized lift | 0.1713 | **0.204098** |

Across all five structures the shipped arm lifts Round-1 advance probability by **0.2302** to
**0.4172** simulated and **0.204098** to **0.341288** realized, every tier clearing its
break-even hurdle on simulated ROI. **Gate D still fails at 0 of 6** — the two buy-in tiers do
not select materially different rosters under either objective, unchanged by any of this.

⚠️ **This does not measure what the preseason block is worth in the contest**, which is what
P2 and 4d both logged as the open question. The composition, the injected σ, the ADP field and
the error injection were all re-fitted in the same run, and the previous `strategy_*.csv` was
overwritten rather than kept, so the honest statement is that the chain under the new heads
reads higher — not that the block bought 0.047 of lift. Isolating it needs the pre-block
tensors kept and a paired re-run, which is a session rather than a footnote.
✅ **That re-run happened on 2026-08-15 and the section below is it. The paragraph above
stands as written — it was the correct reading of what P5's own run could support — and it is
now superseded rather than withdrawn.**

## P5 closes — the paired counterfactual, 2026-08-15

⚠️ **Every figure in this section is the THREE-key reading, and the artifact no longer holds
it.** `make preseason-contest` was re-run later the same day by session 6b with a fourth key,
`stan.components.preseason`, and it overwrites `preseason_block_contest.csv` in place. The
numbers below are what P5 measured and are the record of the three-key pass; the live four-key
readings are in session 6b's "The chain, run end to end", and they are **larger on every Gate A
row**. Two things make the pair readable rather than confusing: the `base` column is a
**byte-identical capture** across the two passes and reproduces to every decimal, which is what
lets the components' share be recovered by differencing; and the four-key pass moved the
realized consistency count from **10 of 10** cells to **9 of 10 with the tenth exactly zero**,
so this section's strongest sentence is the one that did not survive intact. Everything here is
presence-checked rather than value-checked.

**`make preseason-contest`, `src/sim/preseason_contest.py` →
`outputs/predictions/preseason_block_contest.csv`.** The instrument is
`src/sim/mixture_value.py` one round over: two arms of the same chain, captured under the
same code, reported side by side, so a claim reads one row rather than differencing two
artifacts of unreconstructable vintage.

### What the two arms are, and the one thing held fixed

`base` flips the three keys that **are** the block — `stan.availability.preseason`,
`stan.minutes.preseason`, `stan.composition.preseason.adopt` — and refits all three head
groups. It is not an ablation to zero: each key is a documented exact rollback, so `base` is
the fully-fitted chain that was shipping before the round.

| head | `base` | `preseason` |
|---|---|---|
| availability | **19** features, 2012-13 | 29 features, 2012-13 |
| minutes | **24** features, **1997-98**, 8,306 rows | 29 features, 2004-05, 6,152 rows |
| composition | 25 features, **1996-97**, **631,158** rows | 25 features, 2004-05, 448,464 rows |

All three refit clean: R̂ 1.0073 / 1.0052 / 1.0044, **0 divergences**, round-trip PASS, 185.7
sampler minutes of which the composition is 165.8.

⚠️ **`sim.minutes.player_season_sigma` is held at 0.375 in BOTH arms, deliberately.** σ is not
part of the block — it is a downstream constant whose *input* moved at P5 — so freezing it is
what makes the delta attributable to the block rather than to the block plus a re-tuned
injection. **`base` is therefore "today's chain with the block removed", not "the chain as of
2026-08-13".**

⚠️ **Only two of the three heads can reach this readout, and that scopes every figure below.**
`src/sim/` imports neither `StanMinutes` nor `rehydrate_minutes` and never looks up
`artifacts["minutes"]`; what it takes from that module is `beta_shapes`, which is arithmetic.
So **P3's block — the largest of the three by its own gate, at −4.789 validation CRPS
minutes — is structurally invisible here.** Its key is flipped and its posterior refitted
anyway so the arm name means what it says, and
`test_the_simulator_never_loads_the_marginal_minutes_head` pins the import fact by parsing
rather than by docstring, because `README.md` carried the opposite claim until 2026-08-14.

### The attribution P5 could not make, and it is near-total

| season-total MAE | P5's recorded "before" | `base` | `preseason` | the block |
|---|---|---|---|---|
| 2022-23 | 397.36 | **397.24747** | **363.23449** | **−34.01297** |
| 2023-24 | 398.45 | **397.95546** | **377.50963** | **−20.44583** |

**The counterfactual lands within 0.11 and 0.49 dk_pts of P5's pre-block figures**, and it
differs from them only by σ. So σ, the ADP field and the error injection are together worth
about half a dk_pt, and essentially the whole of P5's Gate A improvement **is** the block.
CRPS moves −23.81954 and −15.19750, R² +0.05006 and +0.03100, bias 11.28673 and 4.06281
toward zero, and the games-played pmf total variation falls at both seasons.

### The board moves — and that is the reversal against the mixture round

| | preseason block | availability mixture (§7k) |
|---|---|---|
| rank correlation | **0.962988 / 0.971150** | 0.9990 |
| top-100 overlap | **88% / 91%** | 98% |
| mean \|Δrank\| over the 192 drafted picks | **16.411458 / 14.666667** | 3.1979 |
| max \|Δrank\| | **139 / 98** | — |
| drafted picks moving ≥ 12 ranks | **96 / 89** of 192 | — |

`§7l`'s precedent was that a head change arriving as **shape** is a measured null on the
board. This one arrives as the **allocation mean**, and the board says so: half the drafted
picks move by a full round or more, and 12 of the top 100 change identity. The `draw` block
gives the mechanism — stars (30+ mpg) gain **+81.317731** and **+53.867108** of mean season
total with q10 up **+98.983750** and **+52.586364**, while `mean_gp` *falls* slightly
(−0.180050, −0.252000). **The gain is minutes and production, not availability.**

### The contest — the simulated side cannot resolve it, and the control inverts

Every simulated tournament is a null against the instrument's own bar of **0.074835**:
+0.054249, +0.036241, +0.070857, +0.064491, −0.020059. **But the control moves the other way**,
which is the opposite of what the mixture round found:

| | preseason block | availability mixture |
|---|---|---|
| `adp_only_lift` delta — a board identical across arms | **−0.006490** | +0.0093 |
| mean lift delta over 24 strategies | **+0.034429** | +0.0097 |
| strategies moving positive | **22** of 24 | 20 of 24 |
| `lineup_value_blend30`'s delta, in sds of the spread | **+1.053226** | — |

In the mixture round `adp` captured essentially the entire shift, which is what made that
null a *world* effect. Here the pure-market board — which cannot move between arms — went
**down** while the model-reading boards went up, so the simulated gain is not the world
getting easier. `ordering_spearman` is **0.814702**, so the sweep's ordering did shift, though
`lineup_value_blend30` is top in both arms.

### The realized side is priced by PAIRING, not by the simulated bar

⚠️ **A correction to the instrument, made in this session.** The `RESOLVED` flag was being
stamped on realized rows using `resolution.min_detectable_lift_gap` — a bootstrap over **500
simulated worlds**. The realized readout has **one world per season and two seasons**, so that
bar never measured its uncertainty and applying it would overclaim exactly where the evidence
is thinnest. `mixture_value` has the same shape and it never surfaced there only because its
realized delta was small. The realized rows now read `paired/2sn` and get their own block.

What the realized side does have is **pairing**: both arms are scored against identical box
scores with an identical field, so the season-to-season swing in the *level* cancels out of
the *delta*. `88k_alley_oop` is the demonstration — the base arm's lift moves **+0.8273 →
−0.1313** across the two seasons, a swing of 0.96, while the arm-to-arm delta holds at
**+0.0037** and **+0.1016**.

| tournament | `base` | `preseason` | delta | season spread |
|---|---|---|---|---|
| 600k_shootaround | 0.101331 | 0.204098 | **+0.102767** | 0.018611 |
| 20k_spin_move | 0.039989 | 0.307681 | **+0.267691** | 0.092439 |
| 50k_four_pt_play | 0.093927 | 0.273454 | **+0.179527** | 0.033076 |
| 15k_and_one | 0.076103 | 0.237339 | **+0.161236** | 0.013214 |
| 88k_alley_oop | 0.347996 | 0.400663 | **+0.052667** | 0.097935 |

**10 of 10** season × tournament cells are positive, of **10**, weakest cell **0.003700**.

### What P5 decides

1. **The gate closes, and the block was worth shipping.** Not on any single row: on the
   coherence of a decisive board move, a control that went the wrong way for a world effect,
   and realized sign agreement in 10 of 10 cells.
2. **P5's Gate A improvement is the block**, to within half a dk_pt. The sentence "none of
   this is attributable to the preseason block" is superseded.
3. **The realized readout is the arm-comparable one**, and it is priced by pairing.
   `simulated-lift-is-not-a-cross-model-value-metric` stands and is *strengthened* — the
   simulated side resolved nothing here either, at a bar of 0.074835.
4. **`make preseason-contest` is the standing instrument** for this question.

### What P5 does not settle

- **The five tournaments are ONE test**, not five: same worlds, same portfolios, differing
  only in pod size and payout. And the realized side is **two seasons deep**, so 10 cells are
  not 10 independent observations.
- **The two arms' simulated worlds are not the same world.** Gate C solves `rho` per arm and
  it *fell* — 0.399071 → 0.347987 and 0.395650 → 0.302934 — because the rotation is fitted
  from the model-versus-market skill gap and a better model needs less of it. That is
  independent corroboration that the block improved the model, and it is also why `sim_lift`
  is not arm-comparable. It is carried as a `verdict` row rather than as prose.
- **Gate D still fails 0 of 6 in both arms**, unchanged by any of this. ⚠️ **That reversed at
  four keys**: the shipped arm now separates the tiers in **1** of **6** paired comparisons
  (2022-23, `tier_aware`, `bracket_ev`), against **0** in the counterfactual. One of six on a
  comparison that was designed to fail is not a strategy finding — the same season's other
  tier-aware arm and both of 2023-24's stay below the bar — but "0 of 6" is no longer the
  literal reading and `README.md` was corrected with it.
- **What the availability and composition blocks are worth SEPARATELY.** This arm moves both;
  splitting them is another paired pass.

**Whether the injected σ still reads 0.450 against a blended head.** 4d left
`sim.minutes.player_season_sigma` untouched on the correct ground that its gain is the mean
and the predictive sd went the *other* way (59.58 → 57.64). That reasoning is about the
season-level spread the injection exists to supply, and it stands — but the σ grid has never
been run against a blended composition, and a narrower base predictive is exactly the input
that grid is sensitive to. It is not re-read here.

## Session 6b — the rate heads' arms: eleven heads, ten ship, and a screen whose sign did not survive

`make components-preseason` (`src/models/components_preseason.py`) →
`components_preseason.csv`, `components_preseason_rolling.csv`,
`components_preseason_shrinkage.csv`. Point MLE on the `component_rates` machinery over each
head's **shipped** variant, so no CmdStan; **eleven heads × seven arms × two populations**, plus
a 13-origin rolling harness, in **under five minutes**. Run 2026-08-15, the last session in the
round's map. The Stan port and the chain re-run followed the same day
(`make stan-components`, `make posteriors`, `make simulate-season`, `make strategy-sweep`,
`make preseason-contest`).

**It is a P1 commitment, not a parking-lot idea.** P1 decision 2 put `ast`, `fga`, `stl`,
`tov`, `reb` and `ftm|fta` on a short list and recorded `blk`, `fta`, `fg2m|fg2a` and
`fg3m|fg3a` as nulls that get no arm. That result *reversed* the plan's expectation that the
rate half would collapse, and it added a session rather than shrinking one.

### The ladder widened to all eleven heads, and the exclusions were the reason

The session opened on P1's six and did not stay there. **P1's screen is a ΔR² on a point
estimate against a permutation null, and it had already been shown to misrank the heads it
admitted** — `reb` was its *smallest* clearing count head and is this round's largest
block-to-fit ratio. A screen that misranks the arms it passes is not evidence about the arms
it failed, so all five excluded heads were armed too. **Every one of the exclusions was
wrong**, in both directions:

| head | P1 screen | shipped-arm validation | shipped-arm rolling | verdict |
|---|---|---|---|---|
| `fg3a\|fga` | +0.00467, *excluded as small* | **−1.4373** [−2.1407, −0.7513] | **−1.9544** [−2.2347, −1.6952], 13/13 | ✅ **3rd largest in the round** |
| `fta` | −0.00161, **z −2.32** | **−0.6300** [−1.0189, −0.2181] | **−0.8141** [−1.0231, −0.6140], 13/13 | ✅ clears both |
| `fg2m\|fg2a` | −0.00845, **z −9.44** | **−0.1541** [−0.2790, −0.0186] | **−0.1581** [−0.2058, −0.1094], 12/13 | ✅ clears both |
| `blk` | −0.00847, **z −3.18** | −0.0857 [−0.2272, **+0.0573**] | **−0.1649** [−0.2227, −0.1016], 12/13 | rolling only |
| `fg3m\|fg3a` | +0.01488 at z 7.53, own delta −0.00237 | −0.0221 [−0.0716, **+0.0299**] | −0.0327 [−0.0504, −0.0152], 9/13 | rolling only — **and the one head that reverses** |

**⭐ The strongest result in the round is that the screen's *sign* did not survive on a single
head it called harmful.** P1's permutation z said `blk`, `fta` and `fg2m|fg2a` did *worse than
a shuffled block*, which reads as evidence of harm rather than as absence of gain. None of the
three reproduced as harm at a paired interval, and two of them clear the real bar outright —
`fta` at **13 of 13** origins. **No head anywhere in this round has an interval clear of zero
on the wrong side**, on either reading, on either population, at any arm. A permutation z on a
single inner split is a statement about one split's noise, and excluding on a screen that has
already been contradicted is the same error as trusting it.

### The bar, stated before the run — and it is P1's own caveat, executed

P1's bar was an **R² screen on a point estimate**, and P1's "what it does not settle" says in
so many words that it is a filter for what is worth fitting rather than evidence that anything
ships. So the arms are re-asked at the P2/P3 bar: a validation CRPS paired-bootstrap interval
clear of zero on the **draftable** population, **and** the rolling-origin harness agreeing
(interval clear of zero, majority of origins). Every arm — the reference included — fits the
covered window only, and the full-window incumbent rides as a context row.

The multiplicity is larger here than in any earlier round: **eleven heads** rather than one, so
the rolling half is carrying more weight, not less. It is scored on both populations, unlike
P3's, because the fits are shared and the pair is what makes the pooled/draftable gap
interpretable.

### The gate, per head — on the declared primary arm

| head | validation | rolling | origins | gate |
|---|---|---|---|---|
| `fga` | **−2.7502** [−3.9831, −1.5961] | **−3.0663** [−3.4993, −2.6236] | **13/13** | ✅ **PASS** |
| `fg3a\|fga` | **−0.8512** [−1.4600, −0.2401] | **−1.3042** [−1.5403, −1.0748] | **13/13** | ✅ **PASS** |
| `ast` | **−0.8503** [−1.3175, −0.3491] | **−0.9492** [−1.1447, −0.7537] | 12/13 | ✅ **PASS** |
| `reb` | **−0.8213** [−1.3263, −0.3384] | **−0.8138** [−1.0001, −0.6344] | 12/13 | ✅ **PASS** |
| `fta` | **−0.4299** [−0.8518, −0.0160] | **−0.5993** [−0.7865, −0.4247] | 12/13 | ✅ **PASS** |
| `fg2m\|fg2a` | **−0.1348** [−0.2238, −0.0448] | **−0.0860** [−0.1165, −0.0550] | 12/13 | ✅ **PASS** |
| `stl` | −0.0850 [−0.1678, **+0.0026**] | −0.0671 [−0.1016, −0.0305] | 12/13 | ❌ fails validation |
| `blk` | −0.0743 [−0.2074, **+0.0616**] | −0.1288 [−0.1837, −0.0692] | 12/13 | ❌ fails validation |
| `tov` | −0.0480 [−0.2421, **+0.1619**] | −0.2860 [−0.3641, −0.2086] | 12/13 | ❌ fails validation |
| `fg3m\|fg3a` | −0.0044 [−0.0517, **+0.0460**] | −0.0183 [−0.0350, −0.0004] | 8/13 | ❌ fails validation |
| `ftm\|fta` | −0.0129 [−0.0639, +0.0391] | −0.0249 [−0.0493, +0.0017] | 9/13 | ❌ **fails both** |

**Six of eleven clear the conjunction**, against three of six when the ladder was narrower —
and the three heads the widening added (`fg3a|fga`, `fta`, `fg2m|fg2a`) are all heads P1 had
excluded. Four more pass the rolling half decisively and cannot be resolved on 706 validation
rows — the P2 shape again, and `stl`'s validation interval misses by **+0.0026**. One is a
genuine null at both readings.

### The promoted arm: `own_delta_shrunk`, and the gate read on what actually ships

`own_delta_shrunk` beats the declared primary **on the fitting half on every one of the
eleven heads**, intervals clear of zero — which is P3's promotion rule, the one that reversed
P1 decision 4 on the availability head. That makes it the arm the ship decision is read on:

| head | validation | rolling | origins | validation |
|---|---|---|---|---|
| `fga` | **−3.0093** [−4.3826, −1.7217] | **−3.6397** [−4.1416, −3.1649] | **13/13** | ✅ |
| `fg3a\|fga` | **−1.4373** [−2.1407, −0.7513] | **−1.9544** [−2.2347, −1.6952] | **13/13** | ✅ |
| `reb` | **−1.0929** [−1.6223, −0.5916] | **−1.2622** [−1.4908, −1.0378] | **13/13** | ✅ |
| `ast` | **−1.0296** [−1.5424, −0.4740] | **−1.1438** [−1.3877, −0.9077] | **13/13** | ✅ |
| `fta` | **−0.6300** [−1.0189, −0.2181] | **−0.8141** [−1.0231, −0.6140] | **13/13** | ✅ |
| `tov` | **−0.2213** [−0.4003, −0.0257] | **−0.3552** [−0.4370, −0.2711] | **13/13** | ✅ |
| `fg2m\|fg2a` | **−0.1541** [−0.2790, −0.0186] | **−0.1581** [−0.2058, −0.1094] | 12/13 | ✅ |
| `blk` | −0.0857 [−0.2272, +0.0573] | **−0.1649** [−0.2227, −0.1016] | 12/13 | — |
| `stl` | −0.0763 [−0.1706, +0.0170] | **−0.1008** [−0.1382, −0.0617] | 11/13 | — |
| `fg3m\|fg3a` | −0.0221 [−0.0716, +0.0299] | **−0.0327** [−0.0504, −0.0152] | 9/13 | — |
| `ftm\|fta` | −0.0138 [−0.0681, +0.0395] | **−0.0347** [−0.0608, −0.0083] | 9/13 | — |

**Every one of the eleven clears the rolling half**, at 9 to 13 of 13 origins on ~4,300 scored
fitting-half rows; **seven also clear validation**, up from three on the primary. `tov` flips
the gate on the promotion and `ftm|fta` — a two-sided null on the primary — becomes a rolling
pass. The four that do not clear validation have favourable point estimates at both readings
and intervals that reach across zero on 706 rows.

### The volume question closes, in the opposite direction from P3

`docs/preseason-plan.md` leaves the reliability shrink open between an empirical-Bayes weight
and an additive term, and says the gate decides on train. P3 closed it on minutes at
**k = 20** and called it nearly a null (the grid ran 132.152 at k=0 to 132.106 at k=20). On
rates the answer is different and larger:

| head | selected `k` | mean weight | rolling `crps_vs_primary` | origins won vs primary |
|---|---|---|---|---|
| `fga` | 20 | 0.640 | **−0.5734** [−0.7553, −0.4024] | **13/13** |
| `fg3a\|fga` | 160 | 0.237 | **−0.6501** [−0.8239, −0.4874] | **13/13** |
| `reb` | 160 | 0.237 | **−0.4484** [−0.5656, −0.3309] | **13/13** |
| `fta` | 320 | 0.140 | **−0.2148** [−0.3329, −0.1057] | **13/13** |
| `ast` | 80 | 0.367 | **−0.1946** [−0.2864, −0.1024] | 12/13 |
| `fg2m\|fg2a` | 320 | 0.140 | **−0.0721** [−0.1038, −0.0401] | 12/13 |
| `tov` | 320 | 0.140 | **−0.0691** [−0.1184, −0.0248] | 11/13 |
| `blk` | 40 | 0.509 | **−0.0360** [−0.0594, −0.0143] | 10/13 |
| `stl` | 80 | 0.367 | **−0.0338** [−0.0532, −0.0148] | 11/13 |
| `fg3m\|fg3a` | 40 | 0.509 | **−0.0145** [−0.0220, −0.0070] | 11/13 |
| `ftm\|fta` | 160 | 0.237 | −0.0098 [−0.0231, +0.0032] | 9/13 |

On validation the promoted arm wins on `fg3a|fga` (**−0.5861 [−0.9162, −0.2872]**), `reb`
(**−0.2716 [−0.4695, −0.0914]**), `fta` (**−0.2001 [−0.3865, −0.0018]**), `ast`
(**−0.1792 [−0.3322, −0.0091]**) and `tov` (**−0.1733 [−0.2669, −0.0833]**), and ties on
`fga` (−0.2591, interval reaching +0.0070).

The mechanism is why rates and minutes disagree. A minutes total over 60 preseason minutes is
measured *on* those 60 minutes; a per-36 **rate** over the same 60 divides by them, so the
same exposure buys far less precision and there is much more to shrink. The selected `k` runs
to **320 pseudo-minutes** on three heads — more than two full preseasons — which is the grid
saying those deltas should be believed about a seventh. P1's additive `pre_log_min` is not the
right form: it is a tie on every head at both readings.

### The headline: on two heads the block is worth more than the fitted head is

The no-fit carry-forward is on every row, and reading the block against it rather than against
zero is what makes a small CRPS number legible. Read on the shipped arm:

| head | no-fit floor | incumbent | shipped arm | fitting buys | the block buys | ratio |
|---|---|---|---|---|---|---|
| `reb` | **21.0404** | **20.8897** | **19.7968** | **0.1507** | **1.0929** | **7.25×** |
| `fg3a\|fga` | 21.1561 | 20.7750 | **19.3377** | 0.3811 | **1.4373** | **3.77×** |
| `fga` | 43.1785 | 40.6019 | **37.5927** | 2.5765 | **3.0093** | **1.17×** |
| `ast` | 20.2724 | 19.2283 | 18.1988 | 1.0441 | 1.0296 | 0.99× |
| `fg2m\|fg2a` | 8.1818 | 7.9719 | 7.8178 | 0.2099 | 0.1541 | 0.73× |
| `fta` | 23.0410 | 21.7443 | 21.1143 | 1.2967 | 0.6300 | 0.49× |
| `blk` | 5.7356 | 5.4488 | 5.3631 | 0.2868 | 0.0857 | 0.30× |
| `tov` | 9.7352 | 8.9483 | 8.7271 | 0.7869 | 0.2213 | 0.28× |
| `fg3m\|fg3a` | 4.3888 | 4.2857 | 4.2635 | 0.1031 | 0.0221 | 0.21× |
| `stl` | 5.9411 | 5.3884 | 5.3121 | 0.5527 | 0.0763 | 0.14× |
| `ftm\|fta` | 3.8930 | 3.9283 | 3.9145 | **−0.0353** (never clears) | 0.0138 | — |

On `reb` the entire fitted head is worth **0.1507** CRPS rebounds over arithmetic and six
preseason games are worth **1.0929** more; on `fg3a|fga` the ratio is 3.77× and on `fga` the
block is worth more than the fit by itself. That qualifies the project's standing line — "the
rate side is nearly saturated from prior-season information alone" — in the only way it can be
qualified: it is saturated *against prior-season information*, and a preseason game is not
prior-season information. The line stands as written; what 6b adds is that the remaining
headroom is reachable and where.

### The retention under the posterior — ten of eleven hold

A cleared gate earns a Stan port, which is a separate door. `stan_components` fits each armed
head **and a same-window `__no_preseason` control at the same variant**, so the block's value
under the posterior is one subtraction inside one artifact rather than a comparison across
runs. Retention is that posterior gain divided by the point MLE's, both pooled:

| head | point MLE | under the posterior | retention |
|---|---|---|---|
| `fg2m\|fg2a` | −0.1244 | −0.1423 | **1.144** |
| `fg3a\|fga` | −1.2394 | −1.2617 | **1.018** |
| `tov` | −0.2116 | −0.2131 | **1.007** |
| `reb` | −1.0475 | −1.0381 | **0.991** |
| `fga` | −2.7455 | −2.6882 | 0.979 |
| `ast` | −0.9504 | −0.9305 | 0.979 |
| `stl` | −0.0682 | −0.0638 | 0.936 |
| `fta` | −0.5797 | −0.5415 | 0.934 |
| `blk` | −0.0824 | −0.0501 | 0.608 |
| `ftm\|fta` | −0.0178 | −0.0375 | **2.108** |

**Ten of eleven hold, at a median retention of 0.985**, four of them *growing* under the
posterior. That matches the precedent from the other two head families — P3's block grew
(−4.789 → −5.911) and 4c's retention was 1.040 — and it is the answer to 6b's own largest
open question, which was whether a plug-in point MLE would survive parameter uncertainty.
`ftm|fta`'s 2.108 is not a real amplification: it is a ratio of two numbers whose intervals
both span zero, and the head still does not clear its floor at any variant.

### `fg3m|fg3a` is rolled back — the only measured-worse result in the whole preseason round

**Ten of the eleven ship the block. `fg3m|fg3a` does not**, and it is the only head anywhere
in P0–6b that a paired instrument measured as *worse* with the preseason than without it.
`stan_components.PRESEASON_EXCLUDE` opts it out by the `made` column, which is what
`head_preseason_cols` is keyed on.

**Three instruments agree, which is what makes it a finding rather than a noisy row:**

1. **P1's attribution.** The head's apparent **+0.01488** ΔR² at z 7.53 was entirely the
   shared indicator pair (**+0.01987**); its own preseason 3P% delta was **−0.00237**. The
   attribution split exists to catch exactly this and it caught it first, in P1.
2. **6b's pooled point MLE: +0.0069.** The only positive `crps_vs_incumbent` in the eleven —
   the block already scored worse on the full validation frame before any sampler ran.
3. **The posterior control: +0.02914** CRPS, the same sign and larger, and worse on **three**
   metrics rather than one — CRPS 4.19350 against its control's 4.16436, NLL 3.16495 against
   3.16432, PIT KS 0.02536 against 0.02416. (Those four are the *measurement* run's figures;
   the shipped artifact no longer carries a control row for this head, precisely because the
   head is now excluded, so they are presence-checked records rather than live cells.)

The mechanism is the one `ftm|fta` shows from the other side: **a conversion delta is a logit
of a percentage taken over a handful of preseason attempts, and shooting percentage is the
least persistent quantity in the box score.** Prior-season 3P% over ~200 attempts is simply a
better estimate than preseason 3P% over ~15, so the block adds variance and no signal.
Owner decision, 2026-08-15, taken on the posterior reading.

**The rollback is exact, and it is checked rather than asserted.** With the block off, the head
must reproduce the pre-6b head — same columns *and* the same full 8,630-row window, because
`covered_fitting_rows` cuts the window family-wide and an excluded head must be excluded from
the cut too. It does: fitted NLL **3.1676** against the **3.1677** the docs carried before 6b,
a gap of 6e-05 that is sampler noise on an identical design. An earlier build of the same run
fitted it on 6,382 rows — the pre-block columns on the post-block window — while printing that
it had reproduced the pre-block head "exactly"; a head can be selected under one specification
and persisted under another with every artifact staying internally consistent, which is why
the reproduction is now a test rather than a printout.

### Centring does NOT replicate, and the reason is structural

P2 shipped the centred column on a level, P3 on a delta, and `docs/preseason-plan.md` records
centring as "the arm to watch". **It is the first head family where it loses.** Against the
uncentred primary at the rolling reading, on the draftable population:

| head | `own_delta_centered` vs primary | verdict |
|---|---|---|
| `fg3a\|fga` | **+0.2704** [+0.1824, +0.3614] | **loses** |
| `fga` | **+0.1160** [+0.0209, +0.2081] | **loses** |
| `reb` | **+0.0627** [+0.0191, +0.1068] | **loses** |
| `tov` | **+0.0334** [+0.0148, +0.0514] | **loses** |
| `fg2m\|fg2a` | **+0.0195** [+0.0094, +0.0293] | **loses** |
| `ast` | +0.0342 [−0.0138, +0.0755] | ties |
| `fta` | −0.0167 [−0.0683, +0.0347] | ties |
| `blk` | −0.0018 | ties |
| `stl` | −0.0009 | ties |
| `fg3m\|fg3a` | +0.0006 | ties |
| `ftm\|fta` | −0.0000 | ties |

**Five losses with intervals clear of zero and six ties — no head prefers it**, and the
widening strengthened the finding rather than diluting it: `fg3a|fga`, one of the added heads,
is the largest loss in the table. The mechanism is the one the arm was written to test: **the
compression argument is about levels.** Preseason minutes are compressed by an amount that
varies with the calendar (2 games a team in the 2011-12 lockout against 8 in an ordinary
year), and a head with no season term has nowhere to put that. A **per-36 rate has already
divided the exposure out**, so there is no season-level nuisance left for centring to remove —
and removing a season mean that is not a nuisance costs real cross-player signal. P3's finding
is not contradicted; its **scope** is now measured, and it is levels rather than
deltas-in-general.

### Two things the losing arms settle

**The gain is the preseason rate and not the fact of a preseason row.** `missing_only` — the
four age indicators alone — is a tie on ten of the eleven heads at both readings, and on `ast`
it actively **loses** rolling at **+0.0488 [+0.0128, +0.0888]**. P1's attribution columns said
this on a point estimate; it now holds at a distributional unit on eleven heads, which is what
stops the round being a restatement of the missingness census.

**The coverage cut costs essentially nothing on this family**, which is a reversal of the
pattern on the other two heads. `incumbent_full_window` against `incumbent`:

| head | full window | covered window | cut costs |
|---|---|---|---|
| `fga` | 40.4692 | 40.6019 | +0.1327 (4.4% of the block) |
| `fta` | 21.7233 | 21.7443 | +0.0210 (3.3%) |
| `ast` | 19.2082 | 19.2283 | +0.0202 (2.0%) |
| `fg3m\|fg3a` | 4.2827 | 4.2857 | +0.0029 |
| `reb` | 20.8875 | 20.8897 | +0.0022 (0.2%) |
| `stl` | 5.3912 | 5.3884 | **−0.0028** (the cut *helps*) |
| `ftm\|fta` | 3.9393 | 3.9283 | **−0.0110** |
| `blk` | 5.4604 | 5.4488 | **−0.0115** |
| `tov` | 8.9748 | 8.9483 | **−0.0265** |
| `fg2m\|fg2a` | 8.0827 | 7.9719 | **−0.1108** |
| `fg3a\|fga` | 20.9155 | 20.7750 | **−0.1405** |

P3 paid a **quarter** of its increment to the same restriction and session 4d paid **8.5%**;
here the worst head pays 4.4% and **six of the eleven are better off cut**. **6,382** of
**8,630** training rows survive (74.0%), and prior-season rates are the most persistent
quantity in the project, so the lost seasons were buying very little.

### P1 decision 5 is nearly a no-op here, and that is measured rather than assumed

Pooled against draftable on the shipped arm the ratio runs **0.807** to **0.962** across the
ten heads that ship — `fg2m|fg2a` 0.807, `fg3a|fga` 0.862, `stl` 0.894, `fga` 0.912, `fta`
0.920, `ast` 0.923, `tov` 0.956, `reb` 0.958, `blk` 0.962, with `ftm|fta` at 1.286 on a null.
The draftable reading is *larger* on nine of ten, against **6.2×** smaller on the availability
head and a verdict flip on P4's. **9,320** of **10,194** design rows (**91.4%**) and **706** of
**773** validation rows are on a season-start roster, because the `≥ 200 prior minutes` filter
has already removed the mid-season-signing population that made the restriction load-bearing
elsewhere. P1 said exactly this in prose; the number now exists, and the rule is stronger for
having a family where it changes nothing.

`fg3m|fg3a` is the exception worth naming: its ratio is **−0.311**, because the pooled reading
is positive (+0.0069) and the draftable one is negative. A ratio across a sign change is not a
shrinkage, and it is the one head where the two populations disagree about direction — which is
consistent with it being the round's null and is a third instrument pointing the same way.

### The rolling-shrinkage risk runs the other way a fourth time

Validation ÷ rolling on the shipped arm runs **0.398** (`ftm|fta`) to **0.974**
(`fg2m|fg2a`), with `fga` at 0.827, `fg3a|fga` 0.735, `reb` 0.866, `ast` 0.900, `fta` 0.774,
`tov` 0.623, `stl` 0.756, `blk` 0.520 and `fg3m|fg3a` 0.676. **Not one of the eleven shows the
§12e / §14f pattern of a validation reading 4–6× the rolling one** — every head reads *smaller*
on validation than rolling, which is the opposite failure. Four families have now tested it —
availability, minutes, composition, rates — and all four inverted it. The risk entry is updated
accordingly; what it protects against has still never recurred in this round, and the bar still
has no clause for the failure that keeps happening instead.

### The chain, run end to end — and the components share is separable

`make preseason-contest` re-ran with a **fourth** config key, `stan.components.preseason`,
joining the three P5 flipped. Because this pass reuses P5's **byte-identical base capture** —
the `base` column reproduces to every decimal — the components' contribution is the difference
between the two paired passes:

| season-total MAE | `base` | shipped (4 keys) | 4-key delta | P5's 3-key delta | **components add** |
|---|---|---|---|---|---|
| 2022-23 | 397.2475 | **360.96362** | **−36.28385** | −34.01297 | **−2.27088** |
| 2023-24 | 397.9555 | **373.66427** | **−24.29119** | −20.44583 | **−3.84536** |

CRPS moves **−25.05380** and **−18.30024**, R² **+0.05232** and **+0.03650**, and bias
**+15.68180** and **+16.54580** toward zero — the bias improvement is the largest of the round
on both seasons. The games-played pmf total variation falls at both seasons, which it should:
the component heads do not touch availability, so that row is the sanity check rather than a
result.

**The board moves further than P5's three keys moved it**: rank correlation **0.960627 /
0.968677**, mean |Δrank| over the 192 drafted picks **17.109375 / 16.58854**, drafted picks
moving a full round **93 / 98** of 192, top-100 overlap **89% / 90%**. Stars (30+ mpg) gain
**+97.85932** and **+90.90647** of mean season total against P5's +81.32 and +53.87 — so the
rate block adds to the level on exactly the population the composition block was already
lifting.

**The contest reads the same way P5's did, and the control still inverts.** Every simulated
tournament is a null against the instrument's own bar of **0.074835** (+0.048836, +0.037243,
+0.067894, +0.067229, −0.037810). The `adp` control — a board identical across arms, so its
delta can only be the world — moved **−0.00515** against a 24-strategy mean of **+0.04585**,
with **22** of 24 strategies positive. `ordering_spearman` is **0.708569** and
`lineup_value_blend30` is top in both arms. Gate C's fitted rotation falls again — 0.399071 →
**0.362309** and 0.395650 → **0.311690** — which is the same corroboration P5 read from it: the
rotation is solved from the model-versus-market skill gap, and a better model needs less of it.
It is also why the two arms' simulated worlds are not the same world, and why `sim_lift` is not
arm-comparable.

| tournament | `base` | shipped | delta |
|---|---|---|---|
| 20k_spin_move | 0.039989 | **0.369438** | **+0.329449** |
| 50k_four_pt_play | 0.093927 | **0.296073** | **+0.202146** |
| 15k_and_one | 0.076103 | **0.224399** | **+0.148296** |
| 600k_shootaround | 0.101331 | **0.197293** | **+0.095962** |
| 88k_alley_oop | 0.347996 | **0.352671** | **+0.004675** |

⚠️ **The realized consistency count weakens from P5's, and the honest statement is a tie
rather than a loss.** **9** of **10** season × tournament cells move the block's way and the
tenth is **exactly 0.000000** — `88k_alley_oop` in 2022-23, where the two arms' realized lift
is identical to every printed digit. **No cell moves against the block**, which is the claim
the section actually needs; "10 of 10 positive" is not available at four keys and is not being
quoted.

⚠️ **`reference_lift_z` collapses from +1.053226 to +0.119757**, which is not a weakening of
the result but a change in its denominator: the z is the reference strategy's delta measured in
sds of the *across-strategy* spread, and the fourth key moved more strategies further, so the
spread grew faster than the reference's own delta. It is a statement about the sweep's
dispersion and should not be read as an effect size.

### Four wiring gaps, three of which would have shipped a head that was not the head

Recorded because the *pattern* matters more than any one of them: **a head can be selected
under one specification and persisted under another, with every artifact staying internally
consistent.** Nothing downstream would have contradicted itself.

- **`posteriors.component_artifacts`** built its rows through `component_rates.build_design`
  and would have persisted eleven heads with **no preseason columns** while the config and the
  metrics artifact both said the block was on.
- **`src/sim/season.py`** did the same in two places. Caught at *run time* by
  `PosteriorRecipe._block`, 40 s into a 60-minute chain — and the guard names the builder the
  frame should have come from, which is why the error identified its own fix.
- **`manifest_row`** has a fixed column list and never picked up the new `extras`, so the
  block's trace reached the pickles and not the CSV.
- **`covered_fitting_rows`** cut the window family-wide, so the rolled-back `fg3m|fg3a` was
  fitted on **6,382** rows instead of **8,630** — the pre-block columns on the post-block
  window — while the run printed that it had reproduced the pre-block head "exactly".

All four are fixed and pinned by tests, including an AST test that stops `season.py` importing
the plain builder under any alias. **Wire every consumer, not the ones you happen to be
reading**: four call sites needed the same change, and they were found by four different
methods — two by reading, one by a run-time guard, one by checking an output that should have
contained a column and did not.

### ⚠️ The six-head ladder this section replaced

This section was first written for P1's six admitted heads and read **"3 of 6 clear"**. Those
figures are superseded rather than wrong — they are the same arms scored on the same rows, and
the eleven-head run reproduces every one of them — but the *headline* they supported was an
artefact of which heads had been armed. The six-head reading, kept for the record: primary-arm
CRPS `reb` **20.0684**, `fga` **37.8517**, `ast` **18.3780**, `stl` **5.3034**, `tov`
**8.9004**, `ftm|fta` **3.9154**, with `ftm|fta`'s best arm at **3.9131** and still above its
floor; pooled primary-arm deltas **−0.7410** (`ast`), **−2.4302** (`fga`), **−0.0779** (`stl`),
**−0.0483** (`tov`), **−0.8092** (`reb`), **−0.0167** (`ftm|fta`); and a block-to-fit ratio of
**5.45×** on `reb` against 1.07× on `fga`, 0.81× on `ast`, 0.15× on `stl` and 0.06× on `tov`.

Two of those changed meaning on the widening. The `reb` ratio rises from **5.45×** to
**7.25×** because it is now read on the *shipped* arm rather than the declared primary, and
`fga`'s from 1.07× to 1.17× for the same reason — so "the block is worth more than the fitted
head" got stronger, not weaker. And "3 of 6 clear" became **6 of 11 at the gate and 10 of 11
shipped**, with all three of the added clearances coming from heads P1 had excluded.

### What session 6b decides

1. **Ten of the eleven component rate heads ship the preseason block**, as the volume-shrunk
   delta plus the four age-split missing indicators — five columns per head, `k` read from the
   artifact that fitted it. Owner decision 2026-08-15, on the posterior reading.
2. **`fg3m|fg3a` does not**, on three agreeing instruments, and it is the only measured-worse
   result in the round. `PRESEASON_EXCLUDE` opts it out of the columns *and* the window cut.
3. **Four heads ship against the validation half of the bar** — `stl`, `blk`, `fg3m|fg3a`'s
   siblings `fg2m|fg2a` excepted, plus `ftm|fta` — because they pass rolling on 4,300 rows and
   cannot be resolved on 706. Same owner decision P2 recorded on the availability head; it is
   registered as an open decision, not as a clearance.
4. **A screen's sign is not evidence about the heads it failed.** All five of P1's exclusions
   were wrong and two of them clear the real bar outright. The screen stays useful as a
   *filter* and is no longer usable as a *veto*.
5. **The empirical-Bayes shrink is the right form of the volume term on rates**, and `k` is
   fitted per head on the fitting half. P1's additive `pre_log_min` is not.
6. **Centring is a device for levels, not for deltas in general.** Its scope is now bounded by
   a measurement on eleven heads rather than by two successes.
7. **`ftm|fta` is a recorded null at the head's own unit** and still clears no floor at any
   variant — but it ships the block anyway on the rolling half, which is decision 3 and not a
   claim that the head works.
8. **`tov`'s variant flips from `log_own` to `log_own_spline`, and so does `reb`'s.** Both are
   fourth-decimal margins and both were waived explicitly; six of seven count heads now select
   a spline where four did.

### What session 6b does not settle

- **Whether the four rolling-only heads should have shipped.** That is the open decision P2
  registered and did not take, now standing on four more heads. Nothing here re-opens it, and
  the four are individually small — together they are **0.36** CRPS of the round's **8.7**.
- **Whether the contest gain is the rate block specifically.** The 4-key pass differences
  cleanly against P5's 3-key one on **Gate A**, which is why the components' season-total share
  is quotable. The *contest* rows do not difference that way: Gate C's `rho` is re-solved per
  arm, so the two arms' simulated worlds are not the same world.
- **Whether a shrunk delta interacts with the availability block.** All four groups are fitted
  independently and the chain multiplies them; no arm crosses any two.
- ⚠️ **The `reach` block has no `components` row.** `preseason_contest.reach_rows` reports
  `refit_landed` and `n_features` for availability, minutes and composition only, so the family
  this round shipped has **no per-head trace in the artifact that prices it**. All four config
  keys do show 0→1, so the arm name is verified; what is missing is the per-head window/feature
  evidence that the other three groups carry. Small, and worth closing before the next paired
  pass — a `reach` row is the cheapest guard against the fourth wiring gap recurring.
- **Whether `stan.components.preseason: false` is still an exact rollback.** It is for the ten
  armed heads. For `fg3m|fg3a` it is a no-op by construction, and that is now pinned by a test
  rather than by this sentence.

## Why preseason data should help — and where it plausibly won't

- **Availability.** Participation is a direct health reading taken days before the season:
  who played, who sat, and specifically who missed the *tail* of the preseason. It is also
  a candidate covariate for the mixture's `π` — "who is at risk of a disrupted season" is
  exactly what a late-preseason absence speaks to. This is the head where the case is
  strongest a priori, and also the head with two recent rolling-harness failures
  (`availability-window-plan.md` §12e, §14f), so the replication bar is set first (below).
  ⚠️ **Half right, measured by P2.** The participation reading is real and lands on
  *calibration* — all seven arms that carry a preseason column improve `boundary_tail_error`
  at both readings — while the accuracy the gate asks for is visible on 3,575 rolling rows
  and not on 772 validation ones. And the `π` candidacy is a **null**: adding participation to the disruption
  weight makes the boundary worse with an interval, which is §14d's verdict reached by the
  block that had the better claim on it.
- **Minutes and lineup structure.** Preseason rotations reveal the coach's intent and a
  new-team player's usage — information the S−1 minutes weights cannot carry. Two caveats
  shape the features: preseason minutes are *compressed* (starters play ~15–20 minutes,
  camp invitees are inflated), and early-preseason games are experiments while the last
  one or two approximate the real rotation. So the unit is **within-team minutes share and
  rank**, favoring late games — never raw preseason MPG.
- **Component rates.** Nearly saturated from prior-season data alone; a preseason delta
  plausibly matters only where the *role* changed (new team, a summer 3PA-mix change). This
  is why rates are gated rather than assumed in.
- **The no-prior population.** Rookies, returnees and sub-threshold players get their first
  real NBA rows. §8b's `tenure_draft` grading just shipped for their availability *level*;
  a preseason minutes-share key is the natural next term of that series, and preseason
  per-36 rates compete against the `bio_draft_number` imputation on the rate side.
  ⚠️ **Half right, and the halves came out the opposite way round from the ordering above.**
  P4 measured both: the minutes-share key on the availability level is a **tie** on the draft
  pool at both readings, and the arms that drop the draft bucket are decisively *worse* — so
  "the natural next term of that series" is the half that fails. The rate side is where the
  preseason wins, on five of eight targets at 19 of 19 rolling origins, against a draft-bucket
  prior whose R² never exceeds 0.046. The sentence had the right two candidates and backed the
  wrong one.
- **Preseason games are not regular-season games.** Different coaching objectives,
  different effort, exhibition opponents. Every preseason quantity is a *forecast
  covariate*, never a substitute observation — nothing from preseason enters any head's
  likelihood as a target row.

---

## Design decisions

### Difference coding, not PCA — and what QR does and does not buy

Preseason stats are strongly correlated with prior-season stats; the encoding has to say
"what did the preseason *change*". The block enters as **deltas from the prior-season
equivalent on the model's own link scale** (logit scale for shares, log scale for per-36
rates), plus a `has_preseason` indicator:

- **Zero means "no new information."** A player whose preseason agrees with his prior
  season carries a zero delta; a player with no preseason rows carries zero deltas and the
  indicator. The coefficient path through zero recovers the shipped head *exactly*, which
  is what makes the nested-increment comparison legitimate.
- **Shrinkage points the right way.** The `l2` penalty (and a Stan prior at zero) shrinks
  toward "the prior season is right", which is the correct default for a 4–6 game sample.
- **Volume matters.** A delta over 60 preseason minutes is noisier than one over 140.
  Either shrink each delta toward zero by preseason minutes (empirical-Bayes, constant
  fitted on train) or interact the delta with a reliability weight — the EDA gate decides,
  on train.
- **Why not PCA:** it entangles missingness across columns, destroys the nesting property,
  and makes the coefficient unreadable. This is a 4–8 column block, not a 150-column tier.
- **What QR actually does:** a thin-QR reparameterization is a *sampler-geometry* device —
  identical posterior predictive, rotated coefficients. It does not remove collinearity's
  substantive cost (a wide joint posterior along the near-degenerate direction) and it
  changes what a zero-centered prior means, which breaks the nesting argument. Difference
  coding does the real work here; QR stays in reserve as a mixing aid if a Stan port's
  diagnostics degrade, and is irrelevant to the point-MLE ladders (`l2` handles the
  conditioning).

### Point-in-time discipline

- The preseason of season S ends before season S's opener, so the walk-forward is clean:
  a backtest of season S may read S's preseason. The panel asserts every preseason game
  date precedes the season's first regular-season game date (`season_start_dates` is the
  existing instrument).
- **The backtest draft date moves with the frame, and ADP moves during the preseason.**
  ADP is a live average that reprices on preseason news, so `adp-plan.md`'s rule — "a
  backtest drafts on the ADP as of its draft date" — now points at the **latest snapshot
  before the opener**, not the earliest October capture; anything else is a
  production-mismatched field. One asymmetry must be recorded rather than hidden: for
  seasons where only an early-October consensus snapshot exists, the simulated field
  drafts on pre-preseason opinion while our drafter holds post-preseason data. That
  overstates the edge in exactly the way the field-with-lineup-reasoning round warned
  about, so the sweep readout carries a flag for which seasons have a genuinely
  post-preseason ADP snapshot. The 2025-26 DK anchor (captured Oct 17, four days before
  the opener) is already post-preseason-timed, and the production DK capture below is
  timing-matched to it.
- Preseason data is **backfillable** (the API holds 20 years of it), so it does *not* meet
  the `daily-capture` deadline bar and must not creep into the cron. One fetch after the
  final preseason game suffices in production.

### Where the code goes

| piece | where | pattern it follows |
|---|---|---|
| fetch ✅ | `fetch_season_game_logs(season, out, "Pre Season")` — ⚠️ it did *not* already work: `_slug` does not fold the space in `"pre season"`, so the file landed with a space in its name and no prefix parser could classify it. `_season_type_slug` fixes that. Task added to `fetch_all_for_season` behind `_FIRST_YEAR["pre_season"] = 2003`; 2003-04 → 2025-26 backfilled. Files land as `game_logs_pre_season_{season}.csv`, **never** mixed into `game_logs_{season}.csv` | the playoffs-file precedent in the same function |
| season type ✅ | `preprocess._LOG_PREFIXES` + `PRE_SEASON`; `load_raw(season_type="all")` narrowed to regular + playoffs explicitly | the 2026-07-29 playoffs pseudo-season fix |
| panel ✅ | `src/features/preseason.py` → `data/features/preseason.parquet`, one row per (player_id, season): `gp_pre`, `team_pre_games`, `min_pre`, within-team minutes share (late-weighted), per-36 rates for the count components, `fg3a` share, `played_final_game`, `missed_tail`, preseason `team_id` | `src/features/availability.py` |
| coverage ✅ | `preseason.preseason_coverage` → `outputs/eda/preseason_coverage.csv`, written by the same `make preseason` | `draft_pool.py`'s coverage twin |
| EDA gate ✅ | `src/eda/preseason_value.py` → `outputs/eda/preseason_value.csv`, `make preseason-value`. The delta builders (`attach_availability_block`, `attach_rate_block`, `team_minutes_shares`) live here and P2/P3's attach steps reuse them rather than forking — `src/models/availability.py` already imports `with_lags` and `load_ages` from `src/eda/`, so the direction is the house one | `context_value.py` |
| heads | `availability.attach_preseason` + `PRESEASON_COLS`, **opt-in** — `build_design` is imported by seven modules and must not change under them | `attach_absence_mix` / `ABSENCE_MIX_COLS`, verbatim |
| deltas | computed in each head's attach step (they need that head's own lag columns), from the panel's raw aggregates | the lag machinery already in each design |

---

## The gates

Every gate states its bar before running, selection reads validation only
(`selection_split`), and anything fitted from data — the delta shrinkage constant included
— is estimated on the fitting half alone.

**P0 — coverage (fetch + panel).** ✅ **Shipped 2026-08-12** — see the section above.
Backfill the preseason logs, build the panel, write a coverage artifact (rows, games, date
window, share of season-start roster with preseason rows, per season). Record the quirks in
`data-quirks.md`: exhibition opponents with non-NBA team ids, camp invitees who never reach
a season-start roster, the 2020-21 December window, the 2003-05 partial seasons.

**P1 — the EDA gate (train seasons only).** ✅ **Shipped 2026-08-12** — see the section
above. Three measurements, one decision: (a) *redundancy* — correlation of each preseason
quantity with its prior-season equivalent; (b) *incremental signal* — the preseason deltas
on season-S outcomes (`gp_share`, MPG, each rate head) after the prior-season feature block;
(c) *the missingness census* — who has no preseason rows, by age and role, because "rested
veteran" and "injured star" are different absences and the indicator may need splitting.
**The bar for rates** was stated before the run: a rate head's preseason delta must show
incremental signal on train at least comparable to the shipped fitted-over-floor margins
(+0.0013 to +0.0334 validation R²) before it earns an arm. Below that, rates stay out and
the finding is recorded as a null. **Five of seven count heads cleared it**, which is not
the outcome the plan expected; the census says the indicator needs splitting on **age**;
and the two remaining measurements reordered P2 and P3. ⚙️ **The rate short list ran as
session 6b on 2026-08-15** and the screen's ranking did not survive contact with a
distributional bar — three of six clear, and the head P1 ranked *first* (`ftm|fta`, +0.0176 at
z = 18.8) is the round's only two-sided failure. See the 6b section.

**P2 — availability.** ❌ **Ran 2026-08-13 and FAILED its gate**, then ⚙️ **shipped anyway
the same day on an explicit owner decision** — see the section above and "What ships".
*P1 moved this behind P3* — on the draftable population the block is worth +0.0198 R² here
against +0.0492 on minutes. The block was also smaller than P1's seven columns, since
`pre_log_min` alone beats the whole block on this target, and `has_preseason` entered split
by age. Point-MLE increment arms first, Stan port only for a survivor.
Two separate arms, per the §14d lesson that `β` and `π` are different questions: the
preseason block on the mean function, and the participation signal on `PI_COLS`.
**The bar was stated before the round ran:** validation CRPS paired-bootstrap interval clear
of zero with `boundary_tail_error` held, **and** the rolling-origin harness agreeing — the
last two blocks on this head won validation and shrank 4–6× rolling (§12e, §14f), so
validation alone ships nothing. **The conjunction fails, and it fails in the direction the
bar did not anticipate**: validation cannot resolve the arm (−0.102 [−0.322, +0.121]) while
the rolling harness passes both halves decisively (−0.254 [−0.344, −0.162], **10 of 10
origins**, boundary held). Whether the bar needs a clause for a rolling-only win is
registered as an open decision rather than taken by the round. ⚙️ **The block was
nonetheless adopted later the same day**, on an owner decision taken with the failing half in
view, and the arm that shipped is **P1's full ten columns** rather than the declared
five-column primary — the fitting half's own preference, `crps_vs_primary` −0.0977
[−0.1569, −0.0408] at 8 of 10 origins. The round's verdict stands as written; the ship is a
decision about the bar, recorded as such.

**P3 — minutes.** ✅ **Shipped 2026-08-13** — see the section above. *P1 moved this ahead of
P2*: `pre_d_mpg` was the single most valuable column measured in the gate (+0.0519 R² alone,
partial r 0.334). The marginal head's increment ran on the `minutes-window` point-MLE
machinery (cheap, no CmdStan), against a bar stated before the run — validation CRPS
paired-bootstrap interval clear of zero on the draftable population **and** the
rolling-origin harness agreeing. **Both cleared** (−4.789 [−8.08, −1.59] and −7.940
[−9.41, −6.44] on 12 of 13 origins), so the arm earns a Stan port and the composition's
conditional opens. The shipped column is the **season-centred** delta, which an attribution
arm found beats the declared primary on both readings and repairs its bias. The composition
is priced first at the pilot window (2018-19 onward), which `potential-to-dos.md` item 1
measured at ~6× cheaper than the full window. A plausible composition-specific win worth
checking there: preseason minutes share updating the *ordering* and prior-share feature
for players who changed teams.

**P4 — the no-prior population.** ⚙️ **Ran 2026-08-14 and split: (a) fails, (b) passes on
five of eight rate targets** — see the section above. P1 sized it at **29.7%** of in-scope
panel rows. Two measurements: (a) extend §8b's `no_design_level`
ladder with a preseason key — preseason minutes-share bucket crossed with `tenure_draft` —
under the same CRPS-on-validation + rolling discipline; (b) rookie rate priors from
preseason per-36 against the `bio_draft_number` imputation. This is deliberately the
same slot `adp-plan.md` reserved for the ADP prior on thin-data players; if both
eventually exist they compete in the same ladder rather than stacking silently.

**P5 — chain pricing and ship.** ✅ **Opened and run 2026-08-14; CLOSED 2026-08-15 by the
paired counterfactual** — see both sections above. The
composition's 4d arm shipped first, on an owner decision, because this head is the simulator's
minutes source and adopting after the chain would mean sweeping twice. The chain then ran end
to end in 63 minutes and every readout improved, and **that run could attribute none of it**,
because the composition, `sim.minutes.player_season_sigma`, the ADP field and the error
injection all moved in the same pass and the previous `strategy_*.csv` was overwritten rather
than kept. `make preseason-contest` took the measurement the gate had specified and skipped:
essentially **all** of P5's Gate A gain is the block (`base` lands within 0.11 / 0.49 dk_pts
of P5's own pre-block figures), the board moves decisively where the mixture's did not, and
the realized readout is positive in **10 of 10** cells while the simulated side resolves
nothing at a bar of 0.074835. For heads that
changed: `make posteriors
--groups <family>`, `make simulate-season`, `make weekly-scores`, then the contest layer.
`strategy_*.csv` is already deliberately stale (items 6–7 of `potential-to-dos.md` shipped
without re-running it), so this round's sweep re-run settles both at once. Production
docs: rewrite `project-spec.md`'s prediction-time constraint and README §1, update the
decision registry entries, and register this doc's built artifacts in `make docs-audit`.

## Session map

| session | contents | gate |
|---|---|---|
| 1 (2026-08-12) | plan, scoping decisions, router + registry entries | — |
| 2 (2026-08-12) ✅ | fetch backfill, preseason panel, coverage artifact, quirks, tests | P0 |
| 3 (2026-08-12) ✅ | EDA gate; the rates question answered; each head's block frozen | P1 |
| 4 (2026-08-13) ✅ | marginal minutes arm; composition go/no-go | P3 |
| 5 (2026-08-13) ❌ | availability arms (point MLE + rolling) — **conjunction failed on validation, rolling passed 10/10** | P2 |
| 5b (2026-08-13) ⚙️ | `crps_vs_primary` on the availability rolling harness; **both heads ported to Stan** on an owner decision against P2's failing gate | P2, P3 |
| 6 (2026-08-13) 📝 | model cards, the documentation pass, and the two production docs P5 owed | — |
| 7 (2026-08-14) ⚙️ | no-prior ladder + rookie rate prior — **(a) fails, (b) clears on 5 of 8 rate targets**, and neither ships | P4 |
| 4b (2026-08-14) ✅ | the composition's preseason arm at the pilot window — **passes at the head's own unit**, and it is the OFFSET rather than the ordering | P3 |
| 4c (2026-08-13) ✅ | 4b's arm **fitted** — the increment survives the posterior and grows (retention 1.040), and the season unit's tie does not | P3 |
| 4d (2026-08-14) ✅ | the same arm at the **covered window** (2004-05 on) plus a full-window control that prices the coverage cut — the increment grows again (**−0.23418**, retention **1.115**) and beats the shipped head at both units | P3 |
| 4e (2026-08-14) ⚙️ | 4d's arm **adopted** — `head_frame`, the self-cutting window, the artifact's blend stamp — and the ladder re-run behind it | P5 |
| 6b (2026-08-15) ⚙️ | the five surviving rate heads' arms plus `ftm\|fta` — a session P1 *added*. **3 of 6 clear the conjunction** (`fga`, `ast`, `reb`), 4 on the promoted shrunk arm; `ftm\|fta` — P1's **top-ranked** head — fails both halves. Nothing ships | P1→P2 ⚙️ |
| 8 (2026-08-14) ⚙️ | the **chain**, end to end: tensors, weekly scores, bracket, draft-sim and the sweep, behind the adopted blend and σ = 0.375. Every readout improved and **none of it was attributable to the block** — four things moved in one pass. P5 stayed OPEN on its own question | P5 |
| 9 (2026-08-15) ✅ | the **paired counterfactual** — `make preseason-contest`, both arms on one code, σ frozen at 0.375 in both. Closes P5: the Gate A gain **is** the block, the board moves (mean \|Δrank\| **16.41**), realized lift is positive in **10 of 10** cells, and the simulated side is a null whose `adp` control went the *wrong way* for a world effect | P5 ✅ |

Sessions reorder freely as findings land, and P1 exercised that: **minutes moved ahead of
availability** because the measurement inverted the plan's a-priori ordering. Anything that
fails its gate is recorded and the session bank shrinks rather than the bar — but the rate
result went the other way and the bank grew by one. P3 adds a **session 4b**: the composition
go/no-go it opened is a separate fit at the pilot window, not a continuation of the marginal
arm's session.

**P2 was the first session to shrink the bank, and 5b un-shrank it by a route the plan did
not have.** The conjunction failed, no Stan port followed from the *gate* — and then the
owner adopted the block on both heads anyway, with the failing half in view. That is a
decision about the bar taken outside the round that measured it, which is the only place it
could legitimately be taken. It also pulled two P5 items forward: the ports, and the
production-docs rewrite in `README.md` and `docs/project-spec.md`, because the heads shipped
before the session that was supposed to ship them.

## Production runbook — October 2026

The 2026-27 preseason runs roughly Oct 2–17; the opener is Oct 20. The key structural
fact: **the heads fit on historical seasons, so the 2026-27 preseason enters only as
prediction-time design rows, never as fitting data.** Every sampler-hour therefore lands
*before* the preseason, and the crunch is numpy over the pickles — the property
`pipeline.md` already states as "after `make posteriors`, nothing else in the simulation
layer needs CmdStan."

**Before the preseason (September, no deadline pressure):** freeze the pipeline and every
modeling decision; refit changed heads at the production window (`make posteriors
--window full` — guarded, production-only; historical preseason features are already on
disk from the backfill); rehearse the crunch end to end on a validation season, because
the real window is too short to debug a join in.

**The crunch (2–3 days, final preseason game → draft):**

1. One `make fetch`-family pull of the 2026-27 preseason logs after the final game (not a
   cron — backfillable, see above).
2. Rebuild the preseason panel, the 2026-27 design rows, and the draft pool.
3. Build the tensors and simulation results; load the draft board.
4. Capture the timing-matched mid-October DK board (`adp-plan.md` A0 already owns this)
   and draft in the Oct 17–20 window.

## Risks and falsifiers

- ~~**Preseason may be pure noise on veterans.**~~ ✅ **Closed 2026-08-12 by P1, in the
  opposite direction to the one it was written for.** The deltas carry signal on train on
  both minutes and five of seven count heads, so the veteran half does not collapse and the
  session bank grew rather than shrank. What the risk did catch is a *different* null: two
  count heads and two conversion heads are actively hurt by the block and are recorded as
  such, and `fg3m|fg3a`'s apparent gain is `has_preseason` rather than preseason 3P%.
  ⚠️ **Session 6b re-opens half of it.** At the heads' own distributional unit only three of
  the five clearing count heads hold, and the **conversion family collapses completely** —
  `ftm|fta` was P1's largest rate increment and is a tie at both readings, so all four
  conversion heads are now nulls on preseason data. The veteran half does not collapse on the
  *counts*; the risk was pointed at the wrong family.
- **The rolling-shrinkage pattern.** Twice now a block won validation and shrank 4–6× on
  the rolling harness. The bars above are stated before any arm runs, and the rolling
  harness is part of the gate, not a post-hoc check. ⚠️ **Both preseason rounds run the
  other way, and P2 turns that into a live problem with the bar.** On the availability head
  the rolling reading is 2.5× the validation one (−0.254 against −0.102) at 10 of 10 origins
  and 2.4× the precision, so the conjunction fails on the half with 4.6× fewer rows. The
  pattern the bars were written against has now inverted on the two heads that have tested
  it, and a conjunction that is symmetric in form is not symmetric in what it protects
  against. ⚠️ **P3 was the first counter-example and it does not retire the risk.** On the minutes head the rolling reading is the
  *larger* one (−7.940 against validation's −4.789, a ratio of 0.60× rather than 5–30×),
  because the rolling origins fit a mean of 3,685 rows against validation's 6,152 and a
  preseason delta is worth more where the prior-season block is weaker. That is a property
  of this block on this head, not evidence that the pattern is gone — the availability head
  is where it bit twice. ✅ **P2 has now run on that head and the pattern did not repeat**:
  the rolling reading is 2.5× the validation one rather than a fifth of it. So the risk is
  not that a preseason block shrinks rolling; it is that the bar written against that failure
  has no clause for the opposite one. Both preseason rounds inverted it, on two different
  heads, which is the strongest statement the evidence supports and is weaker than "the
  pattern is gone". ⚠️ **Session 6b is the third family and it inverts on all six heads at
  once**: validation ÷ rolling runs 0.90, 0.90, 1.27, 0.17, 1.01 and 0.52, so not one head
  shows a validation reading 4–6× its rolling one, and the two heads that fail the conjunction
  (`stl`, `tov`) fail on the **validation** half with the rolling one passing 12 of 13. Three
  families, three inversions, eight heads — and the open decision P2 registered now stands on
  two more heads without being taken.
- **Coverage interactions.** ✅ **Handled at P1 rather than deferred**: `covered_seasons`
  restricts every measurement to the 22 seasons with an intact tail, so a block that is
  structurally zero before 2004-05 cannot dilute a ΔR² with a fact about the API. The
  restriction still binds on any head fitting a pre-2005 window, which must show the
  indicator is not soaking up an era effect — the availability head's 2012-13 window dodges
  this entirely, ✅ **confirmed by P2**: all 10 fitting seasons and both validation seasons
  are inside coverage, 0 rows are cut, and the harness's first origin is derived from
  `preseason_coverage.csv` rather than hard-coded. ⚠️ **The minutes head does not**, and P3
  is where the risk actually bit:
  it fits from 1997-98, so 2,154 of 8,306 training rows (25.9%) predate coverage. Every arm
  including the reference fits the covered window only, and the restriction is worth 1.19
  CRPS minutes on its own — a quarter of the increment, if it had been left inside it.
  ✅ **Session 6b's rate design fits from 1997-98 too and the same treatment costs it almost
  nothing**: 6,382 of 8,630 training rows survive (74.0%), and the covered-window incumbent is
  within **0.133** CRPS of the full-window one on every head — 4.8% of the block at worst, and
  on `tov` and `stl` the cut *helps*. A risk that bit hard one head over is real and its size
  is a property of how persistent the target is, not of the restriction.
- **A population can look like a feature.** P1's first reading was +0.1171 R² on `gp_share`
  and 6× of it was mid-season signings, who have no preseason row for a contract reason.
  Nothing leaked — both facts are knowable at the draft — but the head is only ever applied
  to season-start rosters, so **every preseason figure from here on is quoted on the
  draftable population** and one quoted without that restriction is a population statement
  wearing a model's clothes. ⚠️ **P2 is the confirmation, at the head's own unit and at the
  size of a shipping decision.** The same block reads CRPS −0.636 [−0.900, −0.390] pooled and
  −0.102 [−0.322, +0.121] draftable on validation — a **6.2×** gap against P1's 5.9× on a
  ridge ΔR², two instruments agreeing — and **2.8×** at the rolling reading. The four
  age-split indicators alone carry 45% of the pooled validation margin and 51% of the pooled
  rolling one, against +0.001 and −0.040 on the draft pool. This risk is no longer
  hypothetical: pooled, the block clears the gate at both readings. ⚠️ **P4 is the third
  confirmation and the first where the population change flips the verdict on its own.** On
  the no-design availability level the same arm reads −0.5874 [−0.7793, −0.3797] pooled at 11
  of 14 origins and −0.3148 [−0.5851, −0.0400] draftable at 8 of 14 — a pass and a
  not-quite-pass on identical draws. The mechanism is now measured directly rather than
  inferred: on the draft pool this population realizes **0.5447** of the schedule with 3.4%
  missing a preseason row, and off it **0.1571** with 38.6% missing. P4 also found the
  *estimator* carrying the same defect — the shipped rates are pooled over both populations
  and handed to one of them — which no round before this one had looked at. ✅ **Session 6b is
  the family where it changes nothing, and that is worth having.** The pooled/draftable ratio
  runs 0.87 to 1.01 across six heads — the draftable reading is *larger* on five of them —
  because the rate design's `≥ 200 prior minutes` filter has already removed 91.4% of the
  mid-season-signing population the risk is about. So the restriction is not a correction that
  always fires; it is a correction whose size is set by how much of the non-draftable
  population a head's own qualification rule already excludes.
- **The ADP asymmetry.** Backtests where the field's ADP is pre-preseason overstate our
  edge; the flag in P5's readout is the honest version, and the 2025-26 anchor (captured
  Oct 17, post-preseason) is the one season where the field is measured at the right date.
- **Schedule quirks.** Exhibition games against non-NBA opponents, neutral-site games, and
  a possible shortened 2026 preseason all land in the panel builder's lap; P0's quirks
  pass is where they get recorded.
