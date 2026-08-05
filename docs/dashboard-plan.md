# Dashboard Plan: From EDA Explorer to Project Walkthrough

This is a planning doc, not a measurement report. It records direction; update it in place as
pieces land, the way `availability-plan.md` was.

> **Companion plan:** `docs/provenance-plan.md`. This dashboard follows a firm rule that every
> figure on screen is read from an artifact a `make` target produced. Roughly a dozen of the
> project's headline figures exist only as prose today, so that plan promotes them to real
> targets first. **Three of this plan's nine tabs are gated on it**, deliberately — see
> [Staging](#staging).

> ## ✅ **All six stages landed 2026-07-30.** `dashboard/app.py` is a 1,137-line nine-tab EDA
> explorer no longer; it is a 12-module package rendering a nine-tab walkthrough over
> **117 hero numbers, 50 charts and 57 tables**, every one of them read from an artifact.
>
> **The completion criterion is met and it is a number, not a judgement:**
> `make dashboard-audit` reports **0 typed constants pending, 0 orphaned artifacts, 0 total
> findings**. The provenance precondition held — all ten items in `docs/provenance-plan.md`
> were verified populated on disk before stage 1 began, so **no tab needed a pending marker**
> and `layout.pending_marker` ships unused, waiting for the next unbacked figure.
>
> Four things went differently from the plan, all recorded below where they apply:
> - **The registry is 97 entries, not the estimated 50–60.** Covering every artifact family
>   honestly took more; see [The registry as shipped](#the-registry-as-shipped).
> - **`make stan-components` finished mid-build**, at 07:16 on 2026-07-30. Tab 6 was written
>   for the in-progress state as specified, then wired to the real artifacts — which are named
>   `stan_component_*`, singular, not `stan_components_*`.
> - **The BBRef entry is an `incident`, not a `null`** — see
>   [Status vocabulary](#the-status-vocabulary-as-shipped).
> - **The orphan check needed globs**, and it immediately earned its keep: dropping tab 3's
>   renderers took the orphan count from 40 to 59, because those renderers' string literals had
>   been the only thing accounting for nine artifact families.

## Purpose and audience

`dashboard/app.py` is a **nine-tab explorer over the EDA artifacts** — one tab per module in
`src/eda/`. That was the right shape when the EDA sweep *was* the project. It no longer is:
the availability head, the minutes head, the component-rate floor, the Stan ports, the
box-score backfill, the ADP pipeline and the tournament economics all landed after it was
written, and **none of them have a tab**. `serial_correlation.csv`, `availability_profile.csv`,
`report_calibration.csv`, `component_rate_metrics.csv`, `season_total_metrics.csv`,
`adp_profile.csv`, every `stan_*` output and both `dk_best_ball_tournament_*.csv` files are
unreachable from the dashboard today.

The revamp reorganizes it **by topic rather than by module**, into nine tabs that follow the
project end to end:

1. Problem & constraints
2. Data collection
3. Exploratory data analysis
4. The availability head
5. The minutes head
6. The DK component-rate heads
7. Season simulations
8. Drafting strategy
9. Decision log

**Audience: the project architect, wanting a birds-eye view of the decisions.** That is a
different reader from the one the current dashboard serves, and it sets two design rules that
override the instinct to show everything:

- **Altitude is the feature.** Each tab opens with its decisions and a handful of hero numbers.
  Every table, matrix and per-column ranking goes *behind an expander*. The default screen for
  a tab should be readable in under a minute; the existing per-chart `table_view` expanders
  already establish the idiom, and this extends it from "the accessible twin of a chart" to
  "the detail behind a claim."
- **Reversals are content, not embarrassment.** This project has a strong record of catching
  its own false findings — the opponent interaction, `teammate_spacing` as era drift, the
  `sklearn` alpha artifact, the nonlinearity false positive, `total_minutes_incl_playoffs`.
  For this reader those are among the most valuable things on the page, so `withdrawn` is a
  first-class status rather than a deletion.

A published version is a stated eventual goal; see [Publishing](#publishing) for what that
constrains now.

---

## Decisions taken

Settled 2026-07-29, in answer to the questions this plan's first draft left open. Recorded here
so they are not relitigated during implementation.

| decision | rationale |
|---|---|
| **Split `dashboard/` into a package**, in this repo, on the same `.venv` | Nine narrative tabs will not fit one file; `app.py` is already 1,137 lines with nine thin ones. No new environment, no separate service. |
| **Registry as code** (`dashboard/decisions.py`), not YAML | Typed, near its consumer, testable without a parser. `CLAUDE.md` gains a standing instruction to update it — see [Keeping the registry honest](#keeping-the-registry-honest). |
| **A ninth tab for the decision log** | It is the single most on-brief artifact for this reader; burying it inside tab 1 undersells it. |
| **The provenance rule is firm, and the prose figures get promoted to `make` targets first** | See `docs/provenance-plan.md`. Accepting a delay on three tabs in exchange for every future decision resting on a reproducible number. |
| **Tab 3 ships narrative-only for now** | The PCA/archetype line is not currently feeding the pipeline, so its figures are deferred rather than ported. See [Tab 3](#3--exploratory-data-analysis). |
| **Tabs 7 and 8 ship as specification for now** | Model-selection and diagnostic visuals come later, once the simulator and draft layers exist to diagnose. |

---

## The organizing device: a decision registry

The decisions currently live as prose spread across `CLAUDE.md`, `README.md` and six plan
docs. Prose cannot be filtered, counted, or cross-referenced, and re-typing it into nine tab
functions would create nine more copies of it.

So the revamp introduces **one structured record of every load-bearing decision**, in
`dashboard/decisions.py`, and every tab renders its own slice of it:

```python
Decision(
    id="components-not-dk-pts",
    topic="problem",
    claim="Predict the twelve components; never predict dk_pts directly.",
    because="Team-context effects move components hard and in opposite directions, then "
            "cancel in the DK sum — 2.11 dk_pts gross into 0.25 net. And the bonus is a "
            "threshold on five components, so E[bonus] != bonus(E[x]).",
    status="settled",
    reproduce="make context-value → outputs/eda/team_context_value_tierA.csv",
    source="docs/predictions-plan.md",     # the doc this was distilled from
    reviewed="2026-07-29",                 # when a human last checked it against that doc
    date="2026-07-29",                     # when the decision was taken
)
```

Status vocabulary, fixed and closed:

| status | meaning |
|---|---|
| `built` | code exists and its artifact is on disk |
| `settled` | decided against evidence, not to be relitigated |
| `measured` | a number we have, not yet a decision |
| `null` | tested and found to be worth nothing — recorded so it is not rebuilt |
| `withdrawn` | previously believed, then falsified by a better measurement |
| `open` | known gap with no answer yet |
| `blocked` | cannot proceed, with the unblocking condition named |
| `deadline` | will be permanently lost if not done by a date |
| `incident` | a one-time diagnosis of an external system, not reproducible by design |

`incident` exists because of a scoping question the provenance rule raises and must answer:
`BoxScoreSummaryV2` silently dropping `InactivePlayers` after 2025-04-10 is a real, load-bearing
finding, but "223 of 228 games came back empty" describes a third party's behaviour on a past
date. Re-deriving it would mean re-probing `stats.nba.com` to no purpose. Those entries carry
their date and doc reference and are rendered **without** a live-number claim. See
`docs/provenance-plan.md` for the full figures-vs-incidents split.

Expected size ~50–60 entries.

### The registry as shipped

**97 entries**, not 50–60. The estimate was low for a reason worth recording: the orphaned-
artifact check makes the registry an *artifact map* as well as a decision log, so every family
the pipeline writes needs either a tab that renders it or an entry that names its make target.
Covering the deferred PCA / archetype line alone — which no tab reads by design — took several
entries that a pure decision log would not have had.

| topic | entries | | status | entries |
|---|---|---|---|---|
| problem | 11 | | `settled` | 37 |
| data | 17 | | `measured` | 28 |
| eda | 18 | | `built` | 8 |
| availability | 21 | | `withdrawn` | 7 |
| minutes | 7 | | `incident` | 6 |
| components | 10 | | `null` | 5 |
| simulations | 8 | | `open` | 5 |
| drafting | 8 | | `deadline` | 3 |
| **total** | **100** | | `blocked` | 1 |

**The registry caught its first drift within the hour, which is the mechanism working.**
`make stan-components` finished mid-build and updated `docs/predictions-plan.md` with results
that **overturn two entries written earlier the same day**: the "splines are worth ≤ +0.003
outside `fg3a`/`blk`" guidance turns out to be *Poisson-specific* — under the negative binomial
that actually ships, `log_own` collapses to 0.679 on `blk` and 0.372 on `fg3a` and only a
spline recovers them — and `fta` joins `ftm|fta` below its floor, which has no "pure player
skill" explanation and is now an `open` defect. Both are recorded as new entries rather than
edits, per the standing rule that a reversal keeps its entry.

**And putting a live artifact read beside a prose figure caught a third thing, which is the
provenance rule paying for itself.** The season-total table's **R² column was wrong in both
`CLAUDE.md` and `docs/availability-plan.md`** — 0.10 / 0.47 / 0.55 / 0.59 / 0.78 / 0.88 against
an actual 0.141 / 0.493 / 0.559 / 0.595 / 0.773 / 0.885 — while every other figure in the block
reproduced exactly. It was hand-typed and never recomputed when the MAE side was refreshed for
the playoff-workload change. No decision moves, since the head's value rests on MAE. Corrected
in both docs with the analysis recorded beside the table, and entered as `withdrawn` rather
than silently overwritten, per `docs/provenance-plan.md`'s rule that a disagreement is
investigated rather than adopted.

`reproduce` accepts a **glob** where one target owns a family — `make pca → data/features/pca_*`
is one line for twenty files. `audit.py` resolves globs for both the existence check and the
orphan check, so a family that stops being written still shows up either way. Without globs the
orphan check would have forced twenty near-duplicate entries or a second hand-maintained list.

### The status vocabulary as shipped

Unchanged and closed, with one classification the plan got wrong. This plan's tab-2 list called
"BBRef cannot supply historical injuries" a `null`; it ships as an **`incident`**. The boundary
that decides it is `docs/provenance-plan.md`'s own figures-versus-incidents line, drawn after
this plan was written: `null` is for nulls that are *measurements* and therefore carry an
artifact — `role_crowding`, the `age × own` interactions, `total_minutes_incl_playoffs` — while
a dated diagnosis of a third party's website is an incident, rendered without a live-number
claim. Classifying it `null` would have obliged it to name an artifact that could only be
produced by re-scraping.

### Tab 9 · Decision log

The whole registry, as one filterable table: by topic, by status, by date. Plus three summary
reads that only exist once the decisions are structured —

- **Status mix**, so "how much of this project is settled vs open" is a number.
- **The reversal thread** — every `withdrawn` entry with what replaced it and what caught it.
  This is the tab's most valuable single view for this reader.
- **The deadline board** — every `deadline` entry with its date, sorted by urgency. Today that
  is the October 2026 DK board and the Wayback backfill; `daily-capture` is the standing one.

### Keeping the registry honest

Drift against `CLAUDE.md` is the single largest risk in this plan, and it gets four
mitigations rather than one.

**1. An explicit precedence rule, written where a reader will hit it.** A new
`dashboard/README.md` states plainly:

> `CLAUDE.md` and the `docs/*-plan.md` files are the source of truth for every claim on this
> dashboard. The registry in `decisions.py` is a **distillation** of them for browsing, not an
> authority. Where the two disagree, the docs are right and the registry is stale — fix the
> registry.

**2. A standing instruction in `CLAUDE.md`**, added in stage 1 alongside the registry itself
(not before, or it would point at a file that does not exist). Proposed text, to sit at the end
of the "Conventions" section:

> - **When a load-bearing decision is taken, reversed, or measured, add or update its entry in
>   `dashboard/decisions.py`** alongside the `CLAUDE.md` / plan-doc edit. The registry is what
>   the dashboard's decision log renders, and it carries `source` and `reviewed` so
>   `make dashboard-audit` can flag entries whose source doc has moved since. Statuses come
>   from a closed vocabulary — a reversal becomes `withdrawn` and keeps its entry rather than
>   being deleted, because the reversals are the most useful thing on that page.

**3. `make dashboard-audit`** (`dashboard/audit.py`, run as `python -m dashboard.audit`) — a
report, not a gate. Four checks:

| check | why it matters |
|---|---|
| Every `reproduce` artifact exists, unless the status is `open`/`blocked`/`deadline`/`incident` | Catches a registry entry citing a target that was renamed or never built |
| Every entry's `source` doc has no git commit newer than its `reviewed` date | The docs-folder sweep the drift risk actually needs: "`docs/availability-plan.md` changed since these 9 entries were last checked" |
| **Artifacts on disk that no tab and no entry references** | The inverse check, and the one that would have caught this whole problem automatically — nine orphaned artifact families accumulated silently because nothing was looking |
| Count of provenance-marked typed constants remaining | Should trend to zero as `provenance-plan.md` lands; a rising count is a regression |

Git commit dates (`git log -1 --format=%cI -- <doc>`) rather than filesystem mtime, which a
checkout resets.

**4. Cadence.** The first check is also a `pytest` test, so it cannot rot. The other three are
report-only — failing the suite because someone edited a doc would train people to ignore it.
For the periodic sweep, a **weekly launchd job** appending to a log, matching the
`daily-capture` precedent already established in the `Makefile` (and inheriting the same Full
Disk Access grant, which is already in place). Reviewing that log is a two-minute job; the
audit's whole purpose is to make it two minutes instead of a re-read of six plan docs.

**As shipped:** `com.nba-deep-learning.dashboard-audit`, Mondays at 09:00, appending a
timestamped report to `outputs/dashboard_audit.log`. `RunAtLoad` is false — launchd's own
catch-up covers a missed week, and nothing downstream waits on it. Verified by kickstarting it:
`launchctl list` reports exit **0** and the log carries the run.

**The orphan check earned its keep immediately, and in the way the risk register predicted.**
Dropping tab 3's eight renderers took the orphan count from 40 to **59** — because those
renderers' string literals had been the only thing accounting for nine artifact families
(`pca_*`, `archetypes_*`, `aging_curves.csv`, `team_composition_*`, `season_matrix_*` and the
rest). Nothing else in the repo knew they existed. Bringing it to zero meant either rendering
each family or naming it in a registry entry with its make target, which is what makes "the
deferred DR line is *recorded* rather than rendered" an auditable statement instead of a
promise. It also means **the orphan check must be brought back to zero deliberately whenever a
tab is removed**, rather than being allowed to drift up.

### The provenance rule

> **Every figure on the dashboard is read from an artifact that a `make` target produced.**
> Where no such artifact exists yet, the figure does not go on the page until one does — see
> `docs/provenance-plan.md`. The only exception is an `incident` entry, which carries a date
> and a doc reference instead of a number.

Until the provenance work lands, a tab that needs an unbacked figure renders a visible
**pending marker** naming the target that will supply it, rather than the typed number. That
makes the gap legible instead of papering over it, and `make dashboard-audit` counts the
markers so the count trends to zero.

---

## Tab specifications

Each tab lists its panels and the artifact each panel reads. **⏳ marks a panel gated on
`docs/provenance-plan.md`.**

### 1 · Problem & constraints

The frame. What is being predicted, what is knowable when, and where the signal is.

| panel | source |
|---|---|
| DK scoring formula and the bonus | `docs/dk_best_ball_rules.md`; formula mirrors `preprocess.compute_dk_pts` |
| What is known at prediction time vs not; the cross-season join and its three consequences | narrative |
| **The variance budget** — player-season identity, own minutes, opponent, home/away — labelled as **in-sample ceilings** | ⏳ `make variance-budget` → `outputs/eda/variance_budget.csv` |
| **The ceiling-vs-achievable contrast** — those ceilings beside the held-out 0.368% main effect and +0.022% interaction | `outputs/eda/opponent_matchup_tierA.csv` (live) |
| The output contract — twelve components, which eight reach scoring, why the deliverable is a joint draw | narrative table |
| Regular season only, and why the reason is the product (Round 4 ends **4/4**) before it is the statistics (the MPG ratio's sign flip across role buckets) | `docs/dk_best_ball_rules.md` + ⏳ `availability_profile.csv` (`playoff_scope`) |
| **The honest caveat** — five current-season games settle 86% of the season total, against +0.0086 R² for the entire own-team block | `outputs/eda/target_profile.csv` (`analysis == season_total`, live) |
| The season total's two factors — how much is explained by log rate alone vs log games alone | ⏳ `target_profile.csv` (`season_total_decomposition`) |

Decisions surfaced here: components not dk_pts · joint draw not marginals · regular season
only · LSTM/Transformer trunk deprioritized (`settled`, do not relitigate) · opponent
interaction guidance `withdrawn`.

### 2 · Data collection

What was gathered, what it cost, what cannot be gathered, and what the joins can and cannot be
trusted to do.

| panel | source |
|---|---|
| Scale tiles — 30 seasons, ~284 MB raw, ~30 families, 731,906 player-games | live row counts via parquet metadata (`num_rows`, no full read) |
| Artifact inventory — every `data/features/` and `outputs/` artifact with rows and mtime, and the make target for anything missing | filesystem walk (live) |
| Coverage heatmap, family × season | `data/features/coverage_report.csv` (live) — moved from the current Coverage tab |
| The **four** coverage boundaries: core 30, tracking 13, estimated 12, hustle 11, and the inactive list at 2006-07 | coverage report (live) |
| **Three capture programs with deadlines**, each with its live archive count: injury-report PDFs (178 files, ~7-month rolling retention, launchd), ESPN feed (5 snapshots, **28 days permanently lost** to a macOS TCC outage), DK ADP boards (2 files, login-gated, October 2026 load-bearing) | `data/raw/injury_reports/`, `data/raw/injuries/`, `data/raw/dk_draft_rankings/` (live) |
| The box-score backfill — 25,706 of 25,709 games — and the append-only-manifest trap (77 error rows against 3 real failures) | `data/raw/_boxscore_status_manifest.csv` (live) |
| **Roster description coverage** — the share of roster minutes with no usable S-1 row, split into rookies / sub-threshold / returnees | ⏳ `outputs/eda/roster_coverage_profile.csv` |
| **The name-join discipline card** — the rejected "same surname + same first initial" rule scored 0.0% unmatched *and* fabricated most of its non-exact matches. An unmatched rate is monotonically increasing in the error it is supposed to detect. | ⏳ `outputs/eda/adp_match_audit.csv` — the rejected rule re-run as a permanent ablation, plus the readable list of surviving non-exact matches |
| **Failure modes that cost a run**, as `incident` cards: poisoned server-side cache · header-only files counted as fetched · `BoxScoreSummaryV2` **silently** dropping `InactivePlayers` · `nba_api`'s own V3 parser raising on 3 stub payloads | registry (`incident` — dated, no live number) |
| Point-in-time discipline — dated-at-publication sources only; `return_date` is a forecast, never the realized return | narrative |

Decisions: capture before backfill · archive raw, re-parse offline · two independent guards
beat one clever matching rule · `no_nba_history` kept apart from `unmatched` · BBRef cannot
supply historical injuries (`null`, checked not assumed).

### 3 · Exploratory data analysis

**Narrative-only in this pass.** The eight EDA figure sections in today's app are *not* ported:
the PCA / archetype line is not currently feeding the pipeline, and porting charts nothing
consumes would be the largest block of work in the revamp for the least return. Later, the
subset of visuals that turn out to matter to the pipeline comes back — chosen then, on
evidence, rather than inherited wholesale now.

What the tab does carry: the six EDA findings that changed the build, as decision cards.

- **Absorb season, always.** `teammate_spacing` reads +0.315 pooled and +0.035 with season
  absorbed. This produced one false finding that reached the README.
- **Minutes-weight per-36 rates** — and it changes the *ranking*, not just coefficients:
  `stl` 0.743 weighted against 0.401 unweighted.
- **Share vs conversion, not count vs percentage** — `sco_pct_fga_3pt` 0.886 beside
  `fg3_pct` 0.500.
- **Archetypes are a partition of a continuum** — silhouette peaks at 0.181; use the soft
  membership vector, never a hard label.
- **The feature matrix is singular, not merely collinear** — rank 144/149, sixteen columns at
  exactly infinite VIF because families ship literal duplicates.
- **Aging lives in availability, not in rates** — a ±15% rate arc against a −54% minutes arc.

Each card links to the artifact and make target behind it, so the numbers stay checkable even
with no chart on screen.

**What happens to the existing renderers.** `charts.py` keeps every figure builder
(`fig_heatmap` / `fig_bars` / `fig_lines` / `fig_scatter`) — they are tested, reusable, and
needed the moment figures return. The eight `tab_*` renderers are removed from the app;
commit `e1a24c4` holds them if a section is wanted back verbatim. One side effect worth
noting: `season_pairs()` was the dashboard's only import from `src/`, so dropping it makes
**"the dashboard reads artifacts and nothing else"** an invariant rather than a convention.

### 4 · The availability head

The richest artifact-backed tab in the project, and the one whose story is most complete:
measurement → ceiling → baselines → downstream value → Stan port. **Fully live-backed — no
gated panels**, confirmed against `availability_profile.csv`'s 234 rows.

| panel | source |
|---|---|
| Why it matters — GP persists at **0.316**, the least persistent quantity in the project | `outputs/eda/persistence.csv`, `availability_profile.csv` |
| **~20× overdispersed** against a binomial, with the GP histogram showing a mode at 72–82 and a long left tail; 26.7% of established rotation players below 60 games | `availability_profile.csv` (`overdispersion`) |
| The two roster windows **bracket** the truth and neither resolves it — 0.768 vs 0.560 played rate | `availability_profile.csv` (`window_bracket`) |
| **Two nulls**: a 3-year average does not beat 1 year (0.392 vs 0.398); longest spell persists at 0.090. There is no durability latent. | `availability_profile.csv` (`multiyear`, `persistence`) |
| The in-sample ceiling ladder, R² ≈ 0.24, and prior MPG matching prior GP exactly | `availability_profile.csv` (`predictor_r2`) |
| Baseline ladder in CRPS — GLM **10.795**, GBM 10.888, ridge 10.896, league/age 13.614 — and the stopping rule it triggered | `outputs/predictions/availability_metrics.csv` |
| PIT histogram, GLM against the baseline's KS 0.175 | `availability_pit.csv` |
| **What it is worth on the deliverable** — the season-total table including both oracles, −211.1 dk_pts MAE, and `oracle_gp` 221.3 vs `oracle_rate` 302.7 settling which half dominates | `season_total_metrics.csv` |
| Playoff-workload ablation, **and the sign inversion**: every playoff column predicts *better* availability. Selection, not fatigue. | `availability_workload_ablation.csv` |
| Nonlinearity ablation, **as a methodology card**: the test split preferred every curved variant, a paired bootstrap said P(Δ<0) = 99.7%, and it was still a false positive. Select on validation. | `availability_nonlinearity.csv` |
| Absence **reasons** — +0.0285 above a shuffled null, ≈69 sd; `missed_scratch` persists at 0.469, higher than GP itself, while `missed_injury` is **+0.034** | `availability_profile.csv` (`decomposition`) |
| Stan port verification — MLE inside the 95% interval for **21/21** coefficients, R̂ 1.0025, 0 divergences; a coefficient forest plot | `stan_availability_{coefficients,diagnostics,metrics}.csv` |
| **Board correlation, with the correction**: shared-β inflation is 1.0016 at 12 players and ~1.064 across all 911. The full-board figure must not be quoted for a 15-man roster. | `stan_availability_board.csv` |
| Report-calibration transfer function — the designation scale is **not monotone** (`Available` 0.852 plays less than `Probable` 0.914); condition on reason, merge Probable and Available | `outputs/eda/report_calibration.csv`, `report_transfer.parquet` |
| Status: spell simulator deliberately not built · preseason-snapshot ablation `blocked` until the archive crosses a season boundary · `FEATURE_COLS` consumes no reason column yet (`open`) | registry |

### 5 · The minutes head

Small, self-contained, and the most instructive contrast in the project — its specification
answer is the **opposite** of the component heads'.

| panel | source |
|---|---|
| Why separate: `min` is the exposure for eleven heads and the largest single common factor | narrative |
| **The trials denominator is derivable exactly** — 37,986 games, **0** disagreements between two independent team-side estimates, worst residual 0.617 min against a 2.5 min boundary. The two teams agreeing *is* the validation. | `outputs/eda/game_length_coverage.csv` (live) |
| OT rate by season — **5.93%** overall, so capping at 48 discards ~6% of games and censors the top of the distribution exactly where stars play most | `game_length_coverage.csv` (live) |
| The feasibility check — 100% join coverage, zero rows with `min > game_length`, max ratio exactly 1.0000 | ⏳ `make game-length` (feasibility rows) |
| Variant ladder against the no-fit floor — `carry_forward` 168.24 → `logit(own)` + spline **146.85**, +0.041 R², −21.4 min CRPS | `stan_minutes_metrics.csv` (live, with `selected` / `beats_floor`) |
| **The specification finding**: the logit scale is a dead wash (0.8565 vs 0.8565) while curvature pays and replicates on both splits — the mirror image of the count heads, and the two answers must not be pooled into one rule | `stan_minutes_metrics.csv` + cross-link to tab 6 |
| Where the curvature is: a **floor at the bottom** of the prior-MPG range (4.3 → 10.5 mpg), not a ceiling at the top; `age` splines are actively worse | `availability_minutes_nonlinearity.csv` (live) |
| **Two dispersions, and the simulator needs the one the fit does not estimate** — season-level ρ 0.0495 against game-level 0.0776 (**4.65×** binomial), plus 2.43× block inflation on top | `stan_minutes_dispersion.csv`, `serial_correlation.csv` (live) |
| `open` defect: a −33 to −41 minute held-out bias against the floor's −5.7, which would compound through eleven heads that take these minutes as exposure | registry |

### 6 · The DK component-rate heads

| panel | source |
|---|---|
| The output contract from the head side: eight scoring components, three attempt exposures, `min` as trials | narrative |
| **The no-fit floor is nearly the whole model** — `carry_forward` scores held-out R² 0.82–0.94 and the best of seven fitted variants beats it by +0.0019 to +0.0203. The centrepiece chart: floor vs best fitted per head, with `beats_floor`. | `outputs/predictions/component_rate_metrics.csv` (live, 82 rows) |
| **Scale, not curvature** — `log(own)` recovers nearly everything in one term; linear-in-raw-rate inside `exp()` is catastrophic for the skewed heads (`fg3a` 0.520, `blk` 0.638). Splines help only `fg3a` and `blk`. | same |
| Three `null`s: `age × own` and `mpg × own` interactions · walk-forward PCA of all 156 columns (±0.003) · `ftm\|fta`, where nothing beats the floor because FT% is pure player skill | same |
| **The `sklearn` alpha card** — `PoissonRegressor` averages deviance by the weight sum, so with `sample_weight = minutes` (Σw ≈ 1e7) `alpha=1.0` crushes every coefficient *silently*. The floor is what caught it, which is why it is now mandatory. | ⏳ `component_rate_metrics.csv` (`alpha_sensitivity` rows — the trap re-run as a permanent ablation) |
| Dispersion by minutes bucket — `pts` at 2.16–2.31 in every bucket against the shot classes at ~1.0, so decomposing *removes* the misspecification rather than patching it with an NB | `outputs/eda/target_profile.csv` (live) |
| Zero-inflation is a minutes artifact — 0.0% zeros above 18 minutes — but `blk` is still 0 in 58.6% of 30–48 minute games | same |
| **No hot hand**: both field-goal conversion heads are dead nulls (1.01×, 1.03× block inflation), so the binomial collapse costs nothing. What is autocorrelated is exposure. | `serial_correlation.csv` (live) |
| The season-collapse identity — exact, not approximate — cited to the test that pins it rather than to a figure | `tests/` (named in the card) |
| **Separate fits, not a megamodel** — the posterior factorizes exactly when parameter blocks are distinct; `megamodel.stan` shared no parameter between any two heads and paid for it with `sample_frac(0.01)` | narrative |
| Per-head build tracker | `stan_components_*` when written — **currently mid-fit**, so this panel must render an "in progress" state, not a missing-artifact warning |

### 7 · Season simulations

Nothing here is built, and the tab says so at the top rather than reading as though it were.
Its job is to show that the *specification* is already pinned by measurements — which is the
honest status and, for this reader, the interesting part. Model diagnostics arrive once there
is a simulator to diagnose.

| panel | source |
|---|---|
| Status banner: not built; next in the implementation plan | registry |
| **The generative chain** — availability → `min \| available` → counts `\| min` → makes `\| attempts` — as a per-head table carrying likelihood, exposure/trials, build status and artifact. Doubles as the build tracker. | registry + artifact presence |
| Where correlation comes from: a **shared `min` draw** first, then a residual copula only if needed | ⏳ `make residual-correlation` → `outputs/eda/residual_correlation.csv` — the full conditional matrix, which the simulator needs as an *input*, not just as a summary |
| **Block variance inflation per component** — the factor by which an independent-draws simulator understates the variance of an aggregate. `min` at 2.43× against 1.01–1.11× for conversions. | `serial_correlation.csv` (live) |
| The bonus needs the joint: `E[bonus] ≠ bonus(E[x])`, and independent sampling is measurably too low | ⏳ `outputs/eda/bonus_calibration.csv` |
| **Never plug in `E[min]` or `E[gp]` — draw them**, and never cap minutes at 48 | narrative + tab 5 cross-link |
| The spell process, with the simple version already falsified: a 2-state chain gives ρ 0.597 and 3.96× inflation against **22.7×** measured, so clustering is ~a sixth of it. Constant hazard implies geometric spells and misses **both** tails (0.483 singles observed vs 0.308; 0.0635 at 10+ vs 0.0365) → a mixture or semi-Markov process. | `availability_profile.csv` (`serial_structure`, `spell_distribution` — live) |
| What the contest requires the simulator to implement: 16-man frozen roster, best 7 of 16 by slot each week (2 G / 2 F / 1 C / 2 UTIL), scoring period = first game through last game of the week's game set, a suspended game scoring in the period it is **played**, four rounds, cascading tie-breaks | `docs/dk_best_ball_rules.md` |
| Validation plan: posterior-predictive checks on held-out team-total variance and same-team pairwise covariance — not point accuracy | narrative |

### 8 · Drafting strategy

The tab with the most genuinely unexploited data on disk: both tournament CSVs parse cleanly
and every economics figure below derives live.

| panel | source |
|---|---|
| Contest mechanics — roster, weekly lineup, scoring (identical to `compute_dk_pts`, verified), snake draft in pods of 12, auto-draft logic (queue → pre-draft ranking → 8G/8F/3C caps), four rounds with no redraft | `docs/dk_best_ball_rules.md` |
| **The five real tournaments** — entries 216 → 35,280, fee \$1 → \$450, with rake, advance rates per round, and 1st prize as a multiple of entry fee, all derived | `data/raw/dk_best_ball_tournament_{metadata,prize_structure}.csv` (live; the metadata file carries 18 trailing unnamed columns and needs deliberate parsing) |
| **Rake as the break-even edge hurdle** `1/(1−rake) − 1` — +10.45% on `88k_alley_oop` against +17.60% on `600k_shootaround`, i.e. the cheap tournaments demand **68% more edge just to return the fee**. The right units, because a raw rake percentage is not denominated like a measured edge. | derived (live) |
| **Round 1 is a zero-consolation knockout in all five** — top 2 of 12 advance, \$0 for ranks 3–12 — so the objective is P(advance), not E[score], and the tie-break mechanics stop being a footnote | derived (live) |
| Payout convexity scales with **entry-fee tier**, not with "is this a tournament": 10,000× the fee at the top of the cheap one, 44× and a near-flat final table on the expensive one | derived (live) |
| The DK board facts — `ID` is a **persistent** player key (667 shared, 100% name agreement), 249 of 698 rows carry ADP, right-censored near pick 186, and a July board is not the same measurement as an October one | `adp_draftkings.parquet`, `adp_dk_id_map.parquet` (live) |
| **The freeze rule** — the table is frozen ~11 months, so *a snapshot's calendar date is not its season*; the 2025-09-06 snapshot is 2024-25 ADP, and any `month >= 10` rule misassigns six 2020 snapshots because 2020-21 tipped off in December | `adp_panel.parquet`, `outputs/eda/adp_profile.csv` (live) |
| The recalibration ladder — raw 24.38 → monotone **17.32** picks, the position offset adding only −0.3 — plus centers at **+13.88** and the rounds-9+ gap at **32.31** where 9 of 16 picks are made | `adp_profile.csv`, `adp_transfer.parquet` (live), with the `n_anchors = 1` warning rendered loudly |
| **Decision: ADP stays in the strategy layer, not the GLMM.** Under a knockout payout the edge *is* model-minus-market, so a model fit on ADP reproduces its own benchmark. Plus the one narrow exception — an ADP prior for thin-data players only — with its pre-registered test. | registry |
| Deadlines: an **early-to-mid October 2026 DK board**, timing-matched to the Oct-2025 anchor · the Wayback backfill at 23 of 259 snapshots | registry (`deadline`) |

---

## Structure

Split into a package, moving the palette and figure builders **verbatim** so no chart
behaviour changes:

```
dashboard/
  README.md       # the precedence rule: CLAUDE.md and docs/ outrank this dashboard
  __init__.py
  app.py          # page config, sidebar, tab dispatch — thin
  theme.py        # SERIES, THEMES, ALL_PAIRS_CAP, theme(), apply_theme(), ordinal_colors()
  charts.py       # fig_heatmap / fig_bars / fig_lines / fig_scatter — kept for tab 3's return
  layout.py       # table_view, note, stat_tiles, decision_card, status_badge, pending_marker
  artifacts.py    # load_cfg, features_dir, eda_dir, read_table, read_pickle, optional, inventory
  decisions.py    # the registry — pure data, no streamlit import
  economics.py    # tournament rake / hurdle / advance-rate derivations — pure, no streamlit
  audit.py        # make dashboard-audit — pure logic + a __main__ block
  tabs/
    problem.py  data.py  eda.py  availability.py  minutes.py
    components.py  simulations.py  drafting.py  decision_log.py
```

Shipped as specified. Three details settled during the build:

- **`app.py` still inserts the repo root on `sys.path`**, but for `dashboard.*` rather than
  `src.*`: `streamlit run` puts the *script's* directory on the path, not the project root, so
  the package would not otherwise import. The "no `src/` imports" invariant is unaffected and
  is pinned by `test_the_dashboard_imports_nothing_from_src`, which walks the package with `ast`.
- **`artifacts.pipeline_health` derives its expectations from the registry's `reproduce`
  fields** rather than from a second hand-maintained list, so "what should exist" has exactly
  one definition and the sidebar reports the same set `make dashboard-audit` checks.
- **`layout.py` uses Streamlit's named badge colours, not the eight-slot series palette.** A
  status chip is interface, not data; borrowing a data slot for it would imply an encoding that
  is not there.

Three constraints on the split:

- **`decisions.py`, `economics.py` and `audit.py` must not import streamlit.** They hold the
  only new logic worth testing, and keeping them pure means the tests exercise them directly
  rather than through the `importlib` file-loading trick `tests/test_dashboard.py` needs today.
- Each tab module exposes one `render(ctx)`, where `ctx` carries the theme dict, tier, era mode
  and resolved artifact paths. The current signatures pass `(th, tier, mode)` positionally and
  inconsistently; one context object stops that from growing nine ways.
- `python -m dashboard.audit` is a deliberate, minor departure from the repo's
  `python -m src.<module>` convention: the audit is about the dashboard, not the data pipeline,
  and putting it under `src/eda/` would misfile it. Its `Makefile` target and `.PHONY` entry
  follow the convention normally.

**Sidebar.** With tab 3's figures deferred, `Tier` and `Era mode` scope only the coverage
heatmap, so they move under a small "Data scope" heading with a caption saying so — leaving
them looking global would imply the availability head has a tier, which it does not.
`Appearance` stays global. Added: a **pipeline health** panel — how many expected artifacts are
present, with the missing ones' make targets — which is the "what is actually built" question
this reader opens the app to ask.

---

## Tests

`tests/test_dashboard.py` (230 lines) already pins the palette rules and figure builders.
Three things need attention:

- **Two tests fail by design** and must be updated, not deleted:
  `test_the_module_declares_nine_tabs` (asserts `len(app.TABS) == 9`, first `"Coverage"`, last
  `"Feature diagnostics"`) and `test_every_tab_name_has_a_renderer` (the nine `tab_*`
  callables). They become a nine-*topic* assertion over `dashboard/tabs/*.render`. The count
  coincidence is worth noticing and not relying on.
- `_app()` loads `app.py` by file path via `importlib`. After the split, the palette tests
  import `dashboard.theme` and `dashboard.charts` directly — simpler, and it stops the test
  from executing the whole app module to check a colour constant.
- Everything else moves unchanged, which is the check that the split changed no behaviour.

New tests, in the same plain-`assert` synthetic-builder style:

- Every registry entry has a status from the closed vocabulary, a non-empty `claim` and
  `because`, and a `source` naming a file that exists.
- Every `reproduce` artifact path exists, unless the status is
  `open`/`blocked`/`deadline`/`incident`. **The anti-drift guard**, and the one audit check
  that is a hard test rather than a report.
- `economics.py`: rake and break-even hurdle on a synthetic prize table match hand-computed
  values, and the derived advance counts reproduce the five known tournaments (1-of-12 through
  2-of-6) from the real CSV.
- `audit.py`: the staleness check flags a synthetic entry whose `reviewed` predates its source
  doc's last commit, and the orphan check flags an artifact no entry references.
- `optional()` returns `None` for a missing path without raising.
- Every tab module exposes a callable `render`.

---

## Publishing

Not a goal for this pass, but deferring tab 3's figures changes the picture favourably: the
dashboard no longer needs `season_matrix_roster_tierA.parquet` (5 MB), the PCA scores (4 MB) or
any pickle. What remains is `outputs/eda/*` and `outputs/predictions/*`, both small, plus a few
hundred KB of ADP parquet. **`component_targets.parquet` (37 MB) is not read today and must not
start being read.** Two notes for later:

- Record which artifacts each tab needs, so a trimmed deploy bundle is a filtering job rather
  than a rewrite.
- Nothing on the page is secret, but the tournament and ADP data are third-party captures.
  Worth a deliberate check before anything is published, not an assumption.

---

## Staging

Each stage leaves the app runnable. Stages 1–2 and 4 do not depend on the provenance work;
stage 5 does, and is explicitly gated.

| stage | scope | done when | |
|---|---|---|---|
| **1** | The split: `theme`/`charts`/`layout`/`artifacts`, nine tab shells, `ctx`, sidebar rework, `dashboard/README.md`, the `CLAUDE.md` instruction, registry scaffold (~15 entries), `audit.py` + `make dashboard-audit`, tests updated | `make dashboard` serves 9 tabs; `make dashboard-audit` runs clean; `pytest tests/` green | ✅ |
| **2** | Tabs 4, 5, 6 — availability, minutes, components. Fully artifact-backed today, so these are the substance of the walkthrough and come first. | every panel reads its artifact; the two ⏳ panels here show pending markers; tab 6 renders correctly both before and after the in-flight `stan-components` fit lands | ✅ |
| **3** | Tab 9 — decision log — and the registry completed to ~50–60 entries, including the `incident` and `withdrawn` sets | status mix, reversal thread and deadline board all render; audit's typed-constant count is accurate | ✅ **97 entries** |
| **4** | Tabs 3, 8 — EDA narrative, drafting strategy — plus `economics.py` and its tests | five-tournament economics derived live, not typed | ✅ |
| **5** | Tabs 1, 2, 7 — **gated on `docs/provenance-plan.md`** — plus retiring the pending markers in tabs 5 and 6 | zero pending markers remain; `make dashboard-audit` reports zero typed constants | ✅ **no markers were ever needed** |
| **6** | `README.md` and `CLAUDE.md` dashboard sections updated to nine tabs; weekly audit launchd job installed | both docs describe the shipped app; `launchctl list` reports exit 0 | ✅ |
| **later** | The subset of EDA figures that turn out to matter to the pipeline, chosen on evidence | — | |

**Stage 2's ⏳ panels never appeared.** The plan expected pending markers on the minutes
feasibility rows and the component alpha-sensitivity rows; both had already landed as items 8
and 10 of `docs/provenance-plan.md`, so they render live like everything else. Verifying the
precondition against disk before starting — rather than trusting the companion plan's
checkmarks — is what established that, and it is worth doing that way next time too.

**Stage 2 also caught the `stan-components` fit landing mid-build.** Tab 6 was written with the
in-progress branch the plan asked for, the fit finished at 07:16, and the panel was then wired
to the real artifacts. Both branches are in the code, because the in-progress state is the one
a fresh checkout hits. The artifacts are `stan_component_{metrics,diagnostics,substitution}.csv`
— **singular `component`**, which the plan guessed wrong — and they carry a finding the plan did
not anticipate: the `fg3a | fga` share reparameterization beats two independent counts by
~0.78 joint NLL per player-season **on both splits**, confirming a prediction `CLAUDE.md` had
recorded as reasoning rather than as measurement.

### Verification, as run

Every stage was verified with a Streamlit `AppTest` harness that executes `app.py` for real in
both appearance modes — `st.tabs` renders all its children, so an exception in any tab surfaces —
rather than with a `curl` against port 8501, which only proves Streamlit served its HTML shell
and would have passed against a completely broken app. `make dashboard` was checked at the end
of every stage as well. Final state: 9 tabs, 117 `st.metric` tiles, 50 charts, 57 tables,
0 exceptions, 0 missing-artifact warnings, in light **and** dark.

---

## Risks

- **Registry drift** is the main one. Four mitigations are specified above, and the inverse
  orphan check is the one most likely to earn its keep — nine artifact families went unnoticed
  precisely because nothing was looking for them. It is reduced, not eliminated: a number
  corrected in `CLAUDE.md` still has to be corrected here.
- **The provenance gate delays three tabs**, accepted deliberately. The mitigation is
  sequencing: the three fully-backed head tabs ship first, so the dashboard is useful well
  before the gate clears.
- **Tabs 7 and 8 read as specification**, and a walkthrough whose late tabs are plans can seem
  thinner than the project is. Leading both with what is *measured and binding* — block
  inflation, the falsified spell chain, the real tournament economics — is the mitigation.
- **Deferring tab 3 removes working code.** `charts.py` and its ~20 tests are kept, so what is
  lost is eight renderers recoverable from commit `e1a24c4`. The risk is that "add back what
  matters" never happens; the audit's orphaned-artifact check will keep saying so, which is the
  intended nag.
