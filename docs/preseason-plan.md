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

## Why preseason data should help — and where it plausibly won't

- **Availability.** Participation is a direct health reading taken days before the season:
  who played, who sat, and specifically who missed the *tail* of the preseason. It is also
  a candidate covariate for the mixture's `π` — "who is at risk of a disrupted season" is
  exactly what a late-preseason absence speaks to. This is the head where the case is
  strongest a priori, and also the head with two recent rolling-harness failures
  (`availability-window-plan.md` §12e, §14f), so the replication bar is set first (below).
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
and the two remaining measurements reordered P2 and P3.

**P2 — availability.** *P1 moved this behind P3* — on the draftable population the block
is worth +0.0198 R² here against +0.0492 on minutes. The block is also smaller than P1's
seven columns, since `pre_log_min` alone beats the whole block on this target, and
`has_preseason` enters split by age. Point-MLE increment arms first, Stan port only for a
survivor.
Two separate arms, per the §14d lesson that `β` and `π` are different questions: the
preseason block on the mean function, and the participation signal on `PI_COLS`.
**The bar is stated now:** validation CRPS paired-bootstrap interval clear of zero with
`boundary_tail_error` held, **and** the rolling-origin harness agreeing — the last two
blocks on this head won validation and shrank 4–6× rolling (§12e, §14f), so validation
alone ships nothing.

**P3 — minutes.** *P1 moved this ahead of P2*: `pre_d_mpg` is the single most valuable
column measured in the gate (+0.0519 R² alone, partial r 0.334). The marginal head's
increment runs on the `minutes-window` point-MLE machinery (cheap, no CmdStan). The composition is priced only if the marginal arm wins,
and first at the pilot window (2018-19 onward), which `potential-to-dos.md` item 1
measured at ~6× cheaper than the full window. A plausible composition-specific win worth
checking there: preseason minutes share updating the *ordering* and prior-share feature
for players who changed teams.

**P4 — the no-prior population.** P1 sized it at **29.7%** of in-scope panel rows.
Two measurements: (a) extend §8b's `no_design_level`
ladder with a preseason key — preseason minutes-share bucket crossed with `tenure_draft` —
under the same CRPS-on-validation + rolling discipline; (b) rookie rate priors from
preseason per-36 against the `bio_draft_number` imputation. This is deliberately the
same slot `adp-plan.md` reserved for the ADP prior on thin-data players; if both
eventually exist they compete in the same ladder rather than stacking silently.

**P5 — chain pricing and ship.** For heads that changed: `make posteriors
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
| 4 | marginal minutes arm; composition go/no-go | P3 |
| 5 | availability arms (point MLE + rolling), Stan port if survived | P2 |
| 6 | no-prior ladder + rookie rate prior | P4 |
| 6b | the five surviving rate heads' arms — a session P1 *added* | P1→P2 |
| 7 | posteriors, simulator gates, strategy sweep, spec/README rewrite | P5 |

Sessions 4–6 reorder freely as findings land, and P1 exercised that: **minutes moved ahead
of availability** because the measurement inverted the plan's a-priori ordering. Anything
that fails its gate is recorded and the session bank shrinks rather than the bar — but the
rate result went the other way and the bank grew by one.

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
- **The rolling-shrinkage pattern.** Twice now a block won validation and shrank 4–6× on
  the rolling harness. The bars above are stated before any arm runs, and the rolling
  harness is part of the gate, not a post-hoc check.
- **Coverage interactions.** ✅ **Handled at P1 rather than deferred**: `covered_seasons`
  restricts every measurement to the 22 seasons with an intact tail, so a block that is
  structurally zero before 2004-05 cannot dilute a ΔR² with a fact about the API. The
  restriction still binds on any head fitting a pre-2005 window, which must show the
  indicator is not soaking up an era effect — the availability head's 2012-13 window dodges
  this entirely.
- **A population can look like a feature.** P1's first reading was +0.1171 R² on `gp_share`
  and 6× of it was mid-season signings, who have no preseason row for a contract reason.
  Nothing leaked — both facts are knowable at the draft — but the head is only ever applied
  to season-start rosters, so **every preseason figure from here on is quoted on the
  draftable population** and one quoted without that restriction is a population statement
  wearing a model's clothes.
- **The ADP asymmetry.** Backtests where the field's ADP is pre-preseason overstate our
  edge; the flag in P5's readout is the honest version, and the 2025-26 anchor (captured
  Oct 17, post-preseason) is the one season where the field is measured at the right date.
- **Schedule quirks.** Exhibition games against non-NBA opponents, neutral-site games, and
  a possible shortened 2026 preseason all land in the panel builder's lap; P0's quirks
  pass is where they get recorded.
