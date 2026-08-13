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
   earn its sampler time on a cheap train-only measurement first.
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
| EDA gate | `src/eda/preseason_value.py` → `outputs/eda/preseason_value.csv`, `make preseason-value` | `context_value.py` |
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

**P1 — the EDA gate (train seasons only).** Three measurements, one decision:
(a) *redundancy* — correlation of each preseason quantity with its prior-season
equivalent; (b) *incremental signal* — partial R² of the preseason deltas on season-S
outcomes (`gp_share`, MPG, each per-36 rate) after the prior-season feature block;
(c) *the missingness census* — who has no preseason rows, by age and role, because "rested
veteran" and "injured star" are different absences and the indicator may need splitting.
**The bar for rates:** a rate head's preseason delta must show incremental signal on train
at least comparable to the shipped fitted-over-floor margins (+0.0013 to +0.0334 validation
R²) before it earns an arm. Below that, rates stay out and the finding is recorded as a
null.

**P2 — availability.** Point-MLE increment arms first, Stan port only for a survivor.
Two separate arms, per the §14d lesson that `β` and `π` are different questions: the
preseason block on the mean function, and the participation signal on `PI_COLS`.
**The bar is stated now:** validation CRPS paired-bootstrap interval clear of zero with
`boundary_tail_error` held, **and** the rolling-origin harness agreeing — the last two
blocks on this head won validation and shrank 4–6× rolling (§12e, §14f), so validation
alone ships nothing.

**P3 — minutes.** The marginal head's increment runs on the `minutes-window` point-MLE
machinery (cheap, no CmdStan). The composition is priced only if the marginal arm wins,
and first at the pilot window (2018-19 onward), which `potential-to-dos.md` item 1
measured at ~6× cheaper than the full window. A plausible composition-specific win worth
checking there: preseason minutes share updating the *ordering* and prior-share feature
for players who changed teams.

**P4 — the no-prior population.** Two measurements: (a) extend §8b's `no_design_level`
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
| 3 | EDA gate; decide the rates question; freeze each head's exact block | P1 |
| 4 | availability arms (point MLE + rolling), Stan port if survived | P2 |
| 5 | marginal minutes arm; composition go/no-go | P3 |
| 6 | no-prior ladder + rookie rate prior | P4 |
| 7 | posteriors, simulator gates, strategy sweep, spec/README rewrite | P5 |

Sessions 4–6 reorder freely as findings land; anything that fails its gate is recorded and
the session bank shrinks rather than the bar.

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

- **Preseason may be pure noise on veterans.** P1 answers this for ~zero sampler cost;
  if the deltas carry nothing on train, the whole veteran half collapses to a recorded
  null and the no-prior half (P4) proceeds alone — preseason data cannot be *less*
  informative than `bio_draft_number` for a player with no NBA rows.
- **The rolling-shrinkage pattern.** Twice now a block won validation and shrank 4–6× on
  the rolling harness. The bars above are stated before any arm runs, and the rolling
  harness is part of the gate, not a post-hoc check.
- **Coverage interactions.** Eight early training seasons carry indicator-only rows; any
  head fitting a pre-2005 window must show the indicator is not soaking up an era effect
  (the availability head's 2012-13 window dodges this entirely).
- **The ADP asymmetry.** Backtests where the field's ADP is pre-preseason overstate our
  edge; the flag in P5's readout is the honest version, and the 2025-26 anchor (captured
  Oct 17, post-preseason) is the one season where the field is measured at the right date.
- **Schedule quirks.** Exhibition games against non-NBA opponents, neutral-site games, and
  a possible shortened 2026 preseason all land in the panel builder's lap; P0's quirks
  pass is where they get recorded.
