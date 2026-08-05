# Predicting Availability: Games Played and Minutes per Game

Games played is simultaneously **the largest lever on the season DK total and the least
predictable input in the project**. `log(season total)` is 73.4% explained by `log(games)`
alone, and games played persists year over year at r = 0.316 — against 0.779 for minutes
per game and 0.90+ for the stickiest rate columns.

This plan covers what data can inform an availability forecast, what has to be gathered,
and how to model it. Every number below was measured during planning; the reproduction
command is given with each block.

> ## ✅ Audited against the artifacts 2026-07-30 — and now audited **automatically**
>
> Every artifact-backed number in this document was recomputed from its artifact. **42
> disagreed** and are corrected in place, each with a note beside the figure. They fall into
> four groups:
>
> - **The season-total R² column** — six values, none reproducible by any construction.
>   Hand-typed and never recomputed when the MAE side was refreshed. Analysis at the table.
> - **The report-calibration block** — the archive is unchanged (12,406 rows, 142 dates, 532
>   players all still match) but the box-score backfill closed 331 previously `uncovered`
>   rows, moving every cell slightly. A clean, explainable refresh.
> - **Playoff appearance rates** (0.664 / 0.838 / 0.922 → 0.711 / 0.905 / 0.958) — the
>   superseded per-(player, team) denominator, already corrected in `CLAUDE.md` and
>   `README.md` and missed here and in `docs/predictions-plan.md`.
> - **Scratch-panel leftovers and last-digit rounding** — the simulator calibration targets
>   (20.1× / 32.6% / 10.6%) still carried 7-season values and **contradicted this document's
>   own overdispersion section**; the "47.6% single-game" figure likewise contradicted its
>   own spell table at 48.3%.
>
> **No conclusion in this plan changes.** Every ordering, stopping rule and decision
> survives — the corrections move figures, not findings.
>
> This is now enforced rather than remembered: **`make docs-audit`** re-derives every quoted
> figure below from its artifact and exits non-zero on a disagreement. See
> `src/docs_audit.py`.

---

## The constraint (as amended)

The prediction-time constraint still holds: **the season is forecast in one shot before
game 1**, from prior-season stats and season-start rosters. There is no in-season updating.

What is amended is the **injury state at the prediction date**. Between seasons we do know,
as of whatever date we forecast on:

- who had offseason surgery and what the stated timeline is,
- who got hurt in training camp or the preseason,
- who ended season S-1 on an unresolved injury and is still rehabbing.

So the availability head takes one extra input the rest of the model does not: a
**dated injury snapshot**, valid as of the prediction date D, where D sits between the end
of season S-1 and the start of season S. That is legitimately knowable and is not leakage,
provided the snapshot is reconstructed from sources dated *at or before D* — see
"Point-in-time discipline" below, which is the single largest correctness risk in this plan.

---

## What I measured before planning

Everything in this section is a runnable report: **`make availability` then
`make availability-profile`** → `outputs/eda/availability_profile.csv` (260 rows), over 30
seasons and 1,314,238 player-games. Figures that differ from this plan's first draft were
recomputed from a 7-season scratch panel onto the full 30; the values here supersede it.

### The internal ceiling is R² ≈ 0.24

7,276 player-seasons that have three prior seasons, season-absorbed, predicting next-season
games-played share. **These are in-sample fits** — read them as an ordering and a ceiling, not
as achievable gains. This repo has already been burned once by comparing an in-sample ceiling
to a held-out gain (see the opponent section of `README.md`), so the distinction is stated up
front.

| Predictor set | R² unweighted | R² minutes-weighted |
|---|---|---|
| prior GP alone | 0.159 | 0.079 |
| **prior MPG alone** | **0.159** | 0.057 |
| GP lags 1–3 | 0.184 | 0.092 |
| GP + MPG (lag 1) | 0.214 | 0.104 |
| GP + MPG lags 1–3 | 0.222 | 0.111 |
| + age, age², career year | **0.236** | 0.116 |

**Minutes weighting is not the settled question here that it is for per-36 rates**, and both
columns are reported because the choice moves the answer by ~2×. Weighting exists to stop a
rate measured over three garbage-time minutes from dominating a regression — real measurement
error. Games played has no such error: "he played 12 games" is exact. Weighting by minutes
therefore down-weights precisely the injured seasons this head exists to predict, so **the
unweighted column is the appropriate one for availability**, even though `persistence.csv`
reports the weighted figure everywhere else. The weighted column is kept for continuity.

Two results matter more than the ceiling itself:

- **Minutes per game predicts next season's games played as well as games played does** —
  0.159 against 0.159 unweighted. MPG is a *rotation-status* measurement, not a health
  measurement, and it persists at 0.779 against GP's 0.317. Much of what looks like
  "availability" is really "is this player in the rotation at all" — a question the repo's
  existing role and team-context features already speak to.
- **Age earns its place here and nowhere else**, confirming `make aging`: the rate arc is
  ±15% peak-to-34 while the minutes arc is −54% peak-to-37.

The persistence figures reproduce `persistence.py` on the same 11,272 consecutive-season
pairs — `gp_share` 0.317 against the recorded 0.316, `minutes_per_game` 0.779 against 0.779,
`total_minutes` 0.641 against 0.640. That agreement is the cross-check that the panel is
built correctly.

### Two nulls that reshape the problem

- **There is no durability latent recoverable from availability history.** A three-year
  average of games-played share predicts next season *no better than one year*: r = 0.392 vs
  0.398 unweighted, 0.26 vs 0.28 weighted. The null holds under both weightings. Averaging
  more history does not denoise it, because there is little signal to denoise.
- **Injury severity does not persist.** Longest absence spell in season S-1 against season S:
  **r = 0.090**, and `long_spells` 0.088. Number of distinct spells reaches 0.407, but that is
  mostly rotation churn, not health.

Together these say the popular framing — "identify injury-prone players from their injury
history" — is not supported. Whatever improves this head will come from *state* (who is hurt
right now, and how badly) and from *role*, not from a durability score.

### Season GP is event-driven, not binomial — ~20× overdispersed

Established rotation players only (prior season ≥ 20 mpg **and** ≥ 70% of games played),
n = 5,267 on the full window:

```
mean games-played share  0.803
observed sd              0.209
binomial sd (82 flips)   0.044
OVERDISPERSION           22.7x        (19.8x on the appearance window, n = 5,639)
```

| Games played | count |
|---|---|
| 0–16 | 115 |
| 16–32 | 206 |
| 32–48 | 404 |
| 48–64 | 985 |
| 64–80 | 2,400 |
| 80–88 | 1,157 |

**26.7% of established, previously-healthy rotation players fall below 60 games; 9.6% below
41.** The distribution is a mode near 72–82 with a long left tail, not a bell around the
mean. This is the most important structural fact in the plan: GP is generated by a small
number of **large discrete events**, not by 82 near-independent coin flips.

Consequence: a point estimate of GP is close to useless for the season total, and a model
trained on squared error against GP will regress everyone toward ~65 games and never produce
the tail. The head must emit a **distribution**.

> The 82-game denominator is applied to `gp_share`, so shortened seasons (1998-99, 2011-12,
> 2019-20, 2020-21) are rescaled rather than dropped. They compress the spread slightly; the
> figure is a floor.

### Absences are heavily clustered

68,530 absence spells on the appearance window, 30 seasons:

| Spell length | value |
|---|---|
| exactly 1 game | 48.3% of spells |
| p50 | 2 games |
| p75 | 3 games |
| p90 | 7 games |
| p99 | 26 games |

**36.2% of all missed games sit in spells of 10+.** So there are really two absence
processes: frequent one-game absences (rest, minor knocks, back-to-backs) and rare long
spells that dominate the games-missed total.

On the *full* window the same statistics read 42.6% single-game and **72.4%** of missed games
in spells of 10+, with p90 = 20. That gap is not a finding about injuries — it is the window
admitting waived and traded players, whose "absence" is the rest of the season. It is a good
illustration of why the bracket is reported rather than resolved by picking one construction.

### The carryover signal is real but mostly already in prior GP

Ending season S-1 on an unresolved absence, against season S outcomes (full window, 30
seasons — the gradient is monotone here where the 7-season draft was noisy):

| Trailing missed games at end of S-1 | n | first-20 play rate in S | season GP share in S |
|---|---|---|---|
| 0 | 7,697 | 0.647 | 0.714 |
| 1–2 | 1,454 | 0.629 | 0.681 |
| 3–5 | 507 | 0.573 | 0.634 |
| 6–10 | 458 | 0.573 | 0.625 |
| 11–20 | 440 | 0.530 | 0.592 |
| **21+** | **716** | **0.457** | **0.516** |

A real and large effect on the tail group (~6.4% of player-seasons). But its **incremental**
R² over prior games-played share is only **+0.0011** (0.1007 → 0.1018), because a player who
missed the last 21 games already has a low prior GP. The carryover feature earns its place by
being *specific* — it identifies which low-GP players are still hurt — not by adding bulk R².

### The measurement that cannot be made from data on disk

Building this panel exposed the core gap. Game logs contain **only games actually played**:
10.7 rows per team-game against a ~15-man roster, and `AVAILABLE_FLAG` is uniformly 1 in
every season, so it carries no information. Absences must be reconstructed as
*team schedule minus player appearances*, and that requires knowing when the player was on
the roster — which the game logs only reveal through appearances.

Both available constructions are biased, in opposite directions (30 seasons, 1,314,238
player-games, 14,569 player-seasons — a row count that matches
`season_matrix_roster_tierA.parquet` exactly):

| Roster window | rows | played rate | fails on |
|---|---|---|---|
| First-to-last **appearance** | 959,069 | 0.768 | Blind to season-ending injuries by construction — the window always ends on a game he played. `trailing_missed` is **identically 0** across all 14,569 player-seasons. |
| **Full season** if he played ≥1 game for the team | 1,314,238 | 0.560 | Counts waived, traded and 10-day players as rostered all year. Mean `trailing_missed` 6.27. |

They bracket the truth, and **neither can separate "injured" from "not on the roster" from
"healthy scratch."** That single distinction is what the whole head depends on, and it is
not obtainable from anything currently in `data/raw/`.

`src/features/availability.py` therefore ships **both**, as one artifact: the panel is built
on the full-season window with an `in_appearance_window` flag, since the appearance window is
a strict subset. Filtering the flag reproduces the narrower construction exactly.

> **A bug worth recording, because it is easy to reintroduce.** `played` must be keyed on
> `(player_id, team_id, game_id)`, not `(player_id, game_id)`. A `game_id` belongs to *both*
> teams in the game, so the two-key merge marks a traded player as having played for his
> **old** team in games he actually played for his new one — which also stretches his
> appearance window to the end of the season. The error inflated the appearance panel from
> 251,611 rows to 270,039 and moved its played rate from 0.727 to 0.681 on a 7-season sample.
> `tests/test_availability.py::test_played_is_keyed_on_team_not_just_game` pins it.

---

## Data to gather

Ranked by value per unit of cost. Tier 2 is listed second but is **time-critical and should
be started first** — see the warning.

### Tier 1 — nba_api, backfillable, high confidence

**1. Per-game inactive lists and DNP reasons — the core asset.**

Verified live during planning. Two endpoints per game:

- `BoxScoreSummaryV2` → `InactivePlayers` result set: who was on the roster but not
  available. 7 rows on 2023-24 opening night.
- `BoxScoreTraditionalV2` (`BoxScoreTraditionalV3` for 2025-26+) → `COMMENT` per player:
  `DNP - Coach's Decision`, `DND - Injury/Illness`, `NWT - League Suspension`.

Together these give, for every player on every game's roster, a three-way status:
**played / dressed-but-scratched / inactive**, with a stated reason. That is precisely the
separation the panel cannot make.

| | |
|---|---|
| Coverage | **2006-07 → 2025-26** (20 seasons). Probed directly: the inactive list returns 0 rows for 2001-02 through 2005-06 and is populated from 2006-07 on. |
| Volume | ~24,600 games × 2 calls |
| Cost | ~8–14 h wall clock at the existing 0.6 s delay |
| Quirk | V2 traditional returns **0 rows** for 2025-26 (deprecated); V3 works and exposes the same `comment` field. Route by season. |

> ⚠️ **Correction to this section, found while implementing it.** `BoxScoreSummaryV2` does
> not merely have "known issues" after 2025-04-10 — it **fails silently**. It keeps
> returning 200 with a populated `GameSummary` and simply drops the `InactivePlayers`
> result set, so a backfill built on it records "nobody was inactive" instead of erroring.
> Measured on the first pass: **223 of 228** games in 2025-26 came back with zero
> inactives, which would have looked like a finding about modern rosters. Use
> **`BoxScoreSummaryV3`** — `boxScoreSummary.{home,away}Team.inactives`, with the team id
> on the parent block rather than the player. It covers the whole 2006-07 → 2025-26 range
> and agrees with V2 exactly where V2 works (6 on 2006-07 opening night, 8 on 2023-24's).
> Per-game inactive counts are recorded in the manifest so a regression of this kind shows
> up as a column of zeros.

This is a new coverage boundary for the project — 2006-07 — alongside the existing Tier A
(30 seasons) and Tier B (13 seasons) lines. Document it in `CLAUDE.md` under "Coverage".

**2. Playoff game logs.** `fetch_season_game_logs` hardcodes
`season_type_nullable="Regular Season"`. A deep playoff run is ~20 extra high-intensity games
of prior-season workload, entirely invisible today. One extra call per season — the cheapest
item in this plan.

**3. `CommonTeamRoster`.** Official roster per team-season, with experience. Gives roster
membership *without* inferring it from appearances, which is the bias documented above. One
call per team-season (~600 total).

### Tier 2 — time-critical, and **cannot be backfilled**

**4. NBA official injury report PDFs.** Verified live. Fully predictable URL:

```
https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MMAM.pdf
```

Published **every 30 minutes**, around the clock on game days. The PDF's internal
`CreationDate` matches the filename exactly (`D:20260303170004-05'00'` ↔ `2026-03-03_05_00PM`),
so files are individually addressable with no index scraping. Each report carries per-player
participation status (Out / Doubtful / Questionable / Probable / Available) **and a stated
reason**, which is richer than anything in the box-score data.

> ⚠️ **Retention is ~7 months, rolling. This archive cannot be backfilled.**
> Probed on 2026-07-27: `2025-12-20` returns 403, `2025-12-27` returns 206, and everything
> from late December 2025 forward is present. Gaps within that window are game-schedule
> artifacts, not retention — `2026-02-15` (All-Star break) is absent while `2026-01-15`,
> `2026-03-03` and `2026-04-15` are present.
>
> **Every day without an archiver is a day of history permanently lost.** Standing up the
> daily capture is the one task in this plan with a deadline attached, and it should land
> before the backfill, which can wait indefinitely.
>
> ✅ **Captured on 2026-07-27**: 175 reports over the 210-day sweep, **2025-12-29 →
> 2026-07-19**, 13,759 player-report rows, 532 distinct players. The 36 misses are 403s on
> days with no games (the offseason tail, and the All-Star break), not retention gaps. Note
> the archive extends into July: Summer League files injury reports too. **The retained
> window is now secured; keeping it growing is a cron entry, not a code change.**
>
> Parsing notes that cost time and are pinned by tests: the pages are landscape with a
> flipped text matrix (y increases *downward*); cells are left-aligned and wrap, so a chunk
> belongs to the last column starting at or before it and never to the nearer midpoint; a
> wrapped reason is typeset *centred* on its row and can continue **across a page break**
> ("Deep Vein Thrombosis", "Stress Reaction" both split that way); and a team that missed
> the filing deadline gets a `NOT YET SUBMITTED` row in the *reason* column with no player,
> which needs its own anchor or it corrupts the nearest player's row. That row is kept — it
> is the difference between "nobody on this team is hurt" and "this team did not say".

One report per day (the 5:00 PM ET one, the mandated deadline report) is sufficient and
costs ~170 requests per season. Capturing the full 30-minute grid is ~8,000/season and is not
worth it for this use case.

**5. ESPN injury API daily snapshot.** `src/data/injuries.py` already exists and works, but
`data/raw/injuries/` holds exactly **one** snapshot, from 2026-06-28. It is a current-status
feed, so it has no history and cannot acquire any retroactively. Put it on the same daily
schedule. Its `return_date` field is a *forecast*, which makes it more valuable than the box
scores for the preseason snapshot and more dangerous — see below.

### Tier 3 — third-party, with access caveats stated honestly

**6. prosportstransactions.com** would be the ideal historical source — a dated,
event-level missed-game log with body part and reason going back to ~1999, and the standard
source in academic NBA injury work. **It is behind a Cloudflare managed challenge**: both
`WebFetch` and a plain browser-UA `curl` to `/robots.txt` are served the JS interstitial, not
the file. Getting past that means solving a challenge deliberately erected to stop automated
access, and I do not recommend building that. Realistic options, in order:

- check whether a maintained mirror of the dataset exists on Kaggle/GitHub (several have
  historically been published), and take it with provenance noted;
- manual export of the search results for the seasons that matter, done by hand in a browser;
- drop it, and rely on Tier 1 for history.

**7. Basketball-Reference.** `robots.txt` sets `Crawl-delay: 3` and **disallows `*/gamelog/`**
— so per-player game logs are off limits. Transaction pages
(`/leagues/NBA_YYYY_transactions.html`) and `/friv/injuries.fcgi` are permitted. Usable as a
cross-check on the Tier 1 backfill and for transaction dates, at 3 s/request. Not a primary
source.

> ### ⛔ Checked, and it cannot supply historical injury data
>
> Fetched on 2026-07-27 at the required 3 s delay, permitted paths only:
>
> - **`/leagues/NBA_2024_transactions.html`** — 323 KB, and **zero occurrences of
>   "injur"** in the entire page. It is trades, signings, waivers and draft picks. The old
>   formal *Injured List* was replaced by the *Inactive List* in 2005, and IL placements
>   stopped being recorded as transactions, so there is nothing here for the modern era.
> - **`/friv/injuries.fcgi`** — 38 rows, i.e. **who is hurt today**. It is a current-status
>   page with no history, so it is subject to exactly the same point-in-time prohibition as
>   the ESPN feed: it can build history going forward and can never fill a past row.
> - **`*/gamelog/`**, which *would* carry per-game DNP reasons, is `Disallow`ed.
>
> **So the answer is no.** The box-score backfill already running is the best historical
> source that exists for this, and the NBA report PDFs are the richest — forward-only.
>
> **What Basketball-Reference *is* still good for**, and it is not nothing: the transaction
> log is a **dated event stream of roster movement**, which is exactly what disambiguates
> the `inactive` status. `InactivePlayers` conflates injury with roster mechanics; joining
> a player's inactive stretch against "was he waived/traded/assigned around that date"
> separates "unavailable because hurt" from "unavailable because he was not on the team".
> That directly attacks the limitation that makes `missed_injury` weak, and it works
> pre-2006-07 where no inactive list exists at all. It is a roster-membership source, not
> an injury source — which is a smaller claim than the original bullet implied.
>
> The one remaining avenue for historical *reasons* is a published mirror of
> prosportstransactions on Kaggle/GitHub, taken with provenance noted. The site itself
> stays off limits.

### Point-in-time discipline

The failure mode that would quietly invalidate the whole head:

**Never build a historical feature from a "current status" source.** The ESPN feed and the
BBRef injury page describe today. If a 2019 training row is filled in from a source scraped
in 2026, the feature encodes the *resolved outcome* — it knows how long the injury actually
lasted. That is leakage of exactly the target.

Rules:

- Only **dated-at-publication** sources may fill historical rows: the NBA report PDFs (dated
  in the filename and the PDF metadata), transaction logs (dated events), and the box-score
  status backfill (dated by game).
- The daily ESPN and PDF captures build history **going forward only**. Rows before the
  archiver's start date stay null and carry an explicit `snapshot_source` = `none`.
- `return_date` must be stored **as forecast on the snapshot date**, never overwritten with
  the realized return. Store the realized return separately as the target.
- Every derived feature carries `as_of_date`, and the builder asserts
  `as_of_date <= season_start_date` for every training row.

---

## Feature design

Keyed on `(player_id, season)` with an explicit `as_of_date`, written to
`data/features/availability_panel.parquet` (per player-game) and
`availability_features.parquet` (per player-season), following the existing artifact split.

**Already built** (`make availability`, no new data required). The panel carries
`played`, `in_appearance_window`, `team_game_index` and `min`; the per-season frame carries
`gp`, `gp_share`, `team_games`, `total_minutes`, `minutes_per_game`, `window_games`,
`window_share`, `available_rate`, `trailing_missed`, `end_play_rate`, `start_play_rate`,
`n_spells`, `longest_spell`, `single_game_spells`, `long_spells` and `missed_games` — one row
per (player, season, **window**), so both bracketing constructions are available downstream.

**Preseason injury state** — the new information, and the reason for Tier 2:
`is_injured_at_D`, `days_since_injury`, `forecast_return_date`, `forecast_games_missed`
(intersect the forecast return with the team's actual schedule), `injury_body_part`,
`is_major_surgery`, `snapshot_source`, `snapshot_staleness_days`.

**Carryover from S-1** — `trailing_missed` and `end_play_rate` are built and measured above.
`ended_season_inactive` — the clean version that distinguishes "hurt" from "waived" — waits on
`InactivePlayers`.

**Decomposed prior-season availability** — the point of the backfill. Split games missed into
`missed_injury`, `missed_scratch` (DNP-CD), `missed_suspension`, `missed_not_rostered`, plus
spell counts and the long/short split at the 10-game boundary established above. The spell
counts exist now; the *reasons* are what the backfill adds. Measured on the current labels
the aggregate decomposition buys nothing — `available_rate` persists at 0.217 and
`missed_games` at 0.137, against `gp_share`'s 0.317 — but that is the version that cannot
tell injury from scratch. Retest on the real labels; if it still buys nothing, record it as a
settled null rather than leaving it open.

> ### ✅ Settled on the full backfill, and it is **not** the expected null
>
> `make boxscore-status` completed **2026-07-28** — 25,706 games of 25,709 attempted,
> 2006-07 → 2025-26, mean `status_coverage` 0.696 and ≥99.6% for every season from
> 2006-07 on. Re-measured over
> **7,673 usable season pairs** (full window, in-sample, season-absorbed, unweighted),
> against 464 in the first read:
>
> | model | R² (7,673 pairs) | first read (464 pairs) |
> |---|---|---|
> | prior `gp_share` alone | 0.2353 | 0.2121 |
> | + `missed_games` (the aggregate) | 0.2365 | 0.2184 |
> | + the 8-way reason split | **0.2648** | 0.2417 |
> | *the same 8 columns on shuffled rows* | *0.2363 (sd 0.00041)* | *0.2225 (sd 0.0047)* |
>
> The split is **+0.0285 above its own chance level, ≈ 69 sd**. Two things changed with
> the sample and both strengthen the finding: the chance level collapsed from +0.0104 to
> **+0.0010** (eight free columns buy almost nothing at 7,673 rows), and the effect itself
> grew slightly rather than regressing toward zero. **The aggregate `missed_games` is worth
> +0.0012** — the decomposition is essentially the entire effect, and the plan's original
> expectation that the aggregate would be a null was correct for the wrong reason.
>
> **Which reasons carry it is the interesting part**, and it inverts the intuition:
>
> | reason (season S-1) | r vs season-S `gp_share` | persistence of the column |
> |---|---|---|
> | `missed_scratch` (DNP - Coach's Decision) | **−0.353** | **0.469** |
> | `missed_inactive` | −0.233 | 0.208 |
> | `missed_not_rostered` | −0.164 | 0.136 |
> | `missed_personal` / `missed_suspension` / `missed_other` / `missed_gleague` | ≈ 0 | ≤ 0.05 |
> | **`missed_injury`** | **+0.034** | 0.168 |
> | *(`missed_games`, for comparison)* | *−0.339* | *0.233* |
>
> The *rotation* reasons predict next-season availability and the *injury* reason does
> not — its coefficient is slightly **positive**, and at about the magnitude of the
> longest-spell null already recorded above (r = 0.090). Both point the same way: this is
> the plan's own "much of what looks like availability is really rotation status", now
> measured on labels that can separate the two. The mechanism for the positive sign is
> selection — a player carrying a `DND - Injury/Illness` comment **dressed**, so the column
> partly marks "was a rotation player" rather than "was hurt".
>
> **The headline number of this whole plan may be the persistence column.**
> `missed_scratch` persists at **0.469** — higher than games played itself (0.317),
> higher than `missed_games` (0.233), and higher than every other availability measure
> except minutes per game. Being benched is a far more stable property of a player than
> being hurt. That is the durability null stated positively.
>
> One caveat survives the larger sample: **`missed_injury` is still the wrong column for
> the health question.** It counts only injury-flagged players who dressed. The genuinely
> unavailable ones are `inactive` and the endpoint states no reason for them, so the split
> stays finer on the healthy side than on the sick side. And this is a nulled in-sample
> ceiling, not a held-out gain.
>
> **What follows, and is now actionable**: feed `missed_scratch`, `missed_inactive` and
> `missed_not_rostered` to the availability head as separate features rather than
> `missed_games`, and do *not* bother with `missed_injury`.
>
> ⚠️ **Caveat added 2026-07-30 — these are pooled over 2006–2025 and the labels have drifted.**
> Among heavy-minute players `missed_scratch` has nearly disappeared as a label (≈1.0–1.6
> games in 2011–2016 to 0.03–0.07 by 2022-25) while `missed_inactive` roughly doubled: modern
> teams deactivate a rested star rather than dressing him DNP-Coach's Decision. So the column
> carrying the 0.469 persistence is not the column carrying the signal in the seasons being
> forecast. See "Load management" under Modeling approach before wiring these in.
> **`src/models/availability.py::FEATURE_COLS` consumes none of them today** — it is 15
> lag/age columns with no reason split at all. That is the one outstanding feature change
> on this head with measured evidence behind it.

> **The machinery.** `season_availability` now emits the full split
> (`missed_injury` / `missed_scratch` / `missed_inactive` / `missed_gleague` /
> `missed_personal` / `missed_suspension` / `missed_not_rostered` / `missed_other` /
> `missed_unknown`), which partitions `missed_games` exactly, alongside `status_coverage`.
> `src/eda/availability.py::decomposition` runs the retest: persistence of each reason
> column, its correlation with next-season `gp_share`, and the R² ladder from prior
> `gp_share` → + `missed_games` → + the reason split. Both sides of a season pair must
> clear 90% coverage.
>
> Below 200 usable pairs it reports **`insufficient_coverage`** rather than a number — a
> thin result here would be a statement about how far the backfill has run, and the
> aggregate version of this null has already been mistaken for a finding once.

### Report calibration — the transfer function, measured

A snapshot says "Questionable — Left Ankle; Sprain"; the head needs `P(play)`. That mapping
is a *transfer function*, and it does **not** need a season boundary — only an overlap
between the PDF archive and the box-score backfill, which already exists inside 2025-26.
`make report-calibration` (`src/eda/report_calibration.py`) measures it, so the October
unblock is a measurement rather than an invention.

**12,338 of 12,406 report rows joined to a realized outcome — 99.5%, and 0.0% unmatched
names.** 142 game dates, 532 players. The 0.55% dropped are `uncovered`: Summer League (the
archive runs to July, those games have no NBA box score).

| designation | n | P(play) | P(dnp) | P(inactive) | P(absent) | min if played |
|---|---|---|---|---|---|---|
| Out | 8,539 | **0.002** | 0.094 | 0.898 | 0.006 | 11.0 |
| Doubtful | 499 | 0.030 | 0.222 | 0.747 | 0.000 | 16.7 |
| Questionable | 1,788 | **0.498** | 0.199 | 0.303 | 0.001 | 23.5 |
| Probable | 602 | **0.914** | 0.066 | 0.020 | 0.000 | 24.7 |
| Available | 910 | **0.855** | 0.118 | 0.024 | 0.003 | 23.8 |

> ⚠️ **This block was refreshed 2026-07-30.** It previously read 12,007 of 12,406 (96.8%)
> with n = 8,400 / 480 / 1,681 / 579 / 867. The **archive is unchanged** — 12,406 report
> rows, 142 dates, 532 players all still match — so this is not the PDF capture moving.
> What moved is the **box-score side**: 331 rows went from `uncovered` to scored as the
> backfill closed the missing 2025-26 games the earlier text called out. Every conclusion
> below survives; the `Doubtful` minutes figure moved most (14.0 → 16.7) on the smallest
> cell, 15 players who dressed.

Four findings, three of which change how the snapshot should be consumed:

- **The five-level scale is not monotone.** `Available` plays *less often* than `Probable`.
  It is reason mix rather than label noise: excluding G-League rows the scale reads 0.001 /
  0.016 / 0.559 / 0.920 / **0.903**, and the surviving 1.7 pp gap sits inside a ~1.6 pp
  standard error. **Condition on `reason_category`, and treat Probable and Available as one
  designation.** `report_transfer.parquet` is therefore keyed on (designation, reason) with
  a marginal row as the fallback for reasons too rare to have a cell.
- **`Out` is near-deterministic and sticky** — 0.002 play rate, and **98.5% unchanged** in
  the next day's report. This is the designation a preseason snapshot mostly carries, and it
  is the one that behaves.
- **A stale `Questionable` is worth about what a fresh one is** — p_play 0.475 at lead 1
  against 0.498 pooled, though it is revised 61.1% of the time. Encouraging for a
  snapshot read weeks ahead, and the only handle the archive has on horizon decay, since
  reports only ever lead a game by 0 or 1 days.
- **There is no minutes haircut to model.** A Questionable who plays gets 23.5 minutes
  against a Probable's 24.7. The designation acts on the play/not-play margin, not on
  workload — so it belongs in the availability head and not in the MPG head.

> **This also retires a Tier-3 dependency.** The plan wanted Basketball-Reference transaction
> logs to separate "inactive because hurt" from "inactive because not on the team". The PDFs
> state it outright: `Out|Not With Team` is **12.5% `absent`** from the box score and
> `Out|Trade Pending` **14.3%**, against **0.2%** for `Out|Injury/Illness` — roster mechanics
> are the only reasons that generate `absent` at all. Forward-only, so BBRef remains the only
> route for *history*, but the preseason snapshot needs no scrape.
>
> That split is also why `absent` and `unmatched` are separate outcomes in this module: one
> is a fact about the roster, the other a fact about the name matching. Collapsing them would
> turn a join bug into a finding about roster mechanics.

> **Playoff rows are a *feature* of season S-1, never a row to fit.** Settled 2026-07-29
> and recorded in `CLAUDE.md`: every fitting frame is regular season only, because the DK
> contest ends 4/4 and because playoff minutes are a role interaction whose sign flips
> (median MPG ratio 0.505 bench / 0.761 rotation / **1.054** starter) with availability
> shifting alongside it (appearance rates 0.711 / 0.905 / 0.958 — corrected 2026-07-30
> from 0.664 / 0.838 / 0.922, a per-(player, team) denominator that double-counts a traded
> player and records him as absent from the team he left). Pooling them would
> corrupt both heads this plan builds. The workload features below are exactly the right
> use of the playoff logs — and the only one. `preprocess.load_raw` defaults to
> `season_type="regular"`; ask for `"playoffs"` explicitly when building these.

> ### ✅ Built 2026-07-29 — and the mechanism is the opposite of the one it was built for
>
> `features.availability.playoff_workload` / `attach_workload` add `playoff_games`,
> `playoff_minutes`, `playoff_mpg`, `made_playoffs`, `playoff_minutes_share`,
> `total_minutes_incl_playoffs`, `career_minutes`, `career_seasons`. 41.8% of
> player-seasons made the playoffs, mean 8.3 games / 194 minutes (max 26), and
> `total_minutes` undercounts those players' mileage by **10.8%**.
>
> `models.availability.workload_ablation` measures it on the plan's own decision rule —
> held-out CRPS, everything fixed but the feature list:
>
> | variant | features | CRPS | vs baseline | R² |
> |---|---|---|---|---|
> | baseline | 15 | 10.914 | — | 0.268 |
> | **+ playoff workload** | **19** | **10.795** | **−0.119** | **0.283** |
> | + playoff only | 18 | 10.817 | −0.097 | 0.281 |
> | + `career_minutes` only | 16 | 10.883 | −0.031 | 0.271 |
>
> It clears the bar, and downstream it is worth **+6.2 dk_pts of season-total MAE**. In
> sample the block is **+0.0178 above its own shuffled null** (0.2998 vs 0.2820, sd 0.0002).
>
> **But the sign is wrong for fatigue, and that is the finding.** Every single-season
> playoff column predicts *better* next-season availability: `playoff_mpg` r = +0.282 raw
> and +0.055 controlling for `gp_share` and MPG, `playoff_minutes_share` +0.108 controlled.
> Playoff participation marks a good player on a good team, and selection beats fatigue
> outright — the same inversion this plan already recorded for `missed_injury`. The one
> column pointing the way fatigue would is **`career_minutes` at −0.067** controlled, which
> is cumulative mileage and matches what `make aging` implied by putting the availability
> arc at −54% peak-to-37 against a ±15% rate arc.
>
> **A null worth recording**: `total_minutes_incl_playoffs` is *worse* than regular-season
> `total_minutes` (in-sample R² 0.2816 vs 0.2843), even though the observation motivating
> it — that `total_minutes` undercounts real mileage — is true. Folding playoff minutes into
> the total mixes team quality into a clean workload measure. It is excluded from
> `FEATURE_COLS` and a test pins that so it does not get "fixed" back in.
>
> Still not built from this bullet: `back_to_backs_played` and the age × mileage
> interaction the paragraph below asks for.

**Workload** — `prior_total_minutes`, `prior_playoff_games`, `prior_playoff_minutes`,
`back_to_backs_played`, `minutes_per_game`, and a career-cumulative minutes total. Age
interacts here: high mileage at 34 is not high mileage at 24.

**Role and rotation** — prior MPG (the strongest single predictor found), usage,
`teammate_usage_load` from `team_context.py`, depth-chart position among the season-S roster.
This is the persistent half of availability and it reuses machinery that already exists.

**Bio and age** — age, age², height, weight, BMI, career year, `draft_bucket`. Per
`make aging`, age belongs in this head and essentially nowhere else.

**Schedule** — for the per-game head: rest days, back-to-back flag
(`matchup.py::add_back_to_back_flag` exists), games in the last 7 days, road-trip position.
Back-to-backs drive the one-game-absence process that is 48.3% of all spells.

---

## Modeling approach

### Emit a distribution, not a point estimate

Given 20× overdispersion and a nonlinear season-total objective, the head's output must be a
predictive distribution over games played. The season total is then

```
E[season_total] = Σ_g P(play in game g) · E[dk_pts | play, g]
```

for the mean, but the *distribution* of the total requires simulating the season, because
absences are clustered and the DK bonus is a threshold.

### Is an autoregressive / Markov availability process the answer? — measured

Absences are obviously clustered, so an AR model is the natural instinct. `make
availability-profile` now measures it directly (`serial_structure`), because the simplest
version — a 2-state chain, play/miss, with constant transition probabilities — makes two
predictions that can be checked rather than assumed. Appearance window, 942,597
transitions:

| | |
|---|---|
| `P(play \| played last game)` | **0.905** |
| `P(play \| missed last game)` | **0.308** |
| lag-1 autocorrelation ρ = p − q | **0.597** |
| implied variance inflation `(1+ρ)/(1-ρ)` | **3.96×** |
| stationary play rate vs observed | 0.764 vs 0.768 ✓ |

**The serial correlation is real and large** — ρ = 0.60 — and the chain reproduces the
stationary play rate almost exactly. Two things follow, and they point in opposite
directions:

**1. Clustering is not where the overdispersion comes from.** A chain with ρ = 0.60
inflates variance **3.96×**, against the **22.7×** measured in `overdispersion`. So serial
correlation explains roughly a *sixth* of it and the rest is **between-player
heterogeneity** — which an AR process cannot generate and which the beta-binomial head
already models. An autoregressive binomial is therefore a **complement** to the current
head, not a replacement for it, and on the GP marginal alone it should not be expected to
win: that marginal is already calibrated (PIT KS 0.096, tail 15.0%/34.8% predicted against
11.8%/36.9% observed).

**2. The simple chain is wrong in a specific, falsifiable way.** A constant hazard implies
**geometric** spells. Fitting q from the transition rate matches the mean spell exactly —
and misses both tails:

| | observed | geometric(q) |
|---|---|---|
| spells of exactly 1 game | **0.483** | 0.308 |
| spells of 10+ games | **0.0635** | 0.0365 |
| mean spell | 3.25 | 3.25 (by construction) |

Too many one-game absences **and** too many long ones, with the mean right. That is the
signature of a **mixture**, and it quantifies what the spell-distribution section above
only asserted. A single-hazard AR model cannot produce it at any ρ.

**Recommendation.** Build the AR structure, but as a **2-component (or semi-Markov)
process** — a short-absence hazard and a separate long-absence one — layered *on top of*
the beta-binomial's between-player heterogeneity, not instead of it. And build it for the
two things it uniquely enables rather than for GP CRPS:

- **Correlated game-level absences** for the season-total distribution. The DK bonus is a
  threshold, so the total needs the joint process across games, not just the GP marginal.
- **A natural home for the preseason snapshot**: "starts the season Out, expected back in
  ~20 games" is an *initial state*, which only a sequential model can consume.

That is the spell simulator below, now with measured parameters and a measured reason the
one-state version fails.

### A spell-based season simulator (recommended)

Mirrors the structure the repo already uses for `expected_bonus`, and for the same reason:
independent sampling gets the mean roughly right and the shape badly wrong.

1. **Initial state** at game 1, drawn from the preseason injury snapshot — healthy, or
   injured with a forecast return distribution centred on the stated timeline.
2. **Availability process** — alternating spells. A per-game injury *onset* hazard, and a
   *duration* draw when a spell starts. Fit onset and duration separately, since the measured
   spell distribution is clearly a mixture (48.3% single-game, a tail to 26+).
3. **Rotation process** — conditional on being available, `P(dressed and played)` from the
   role features. This is the persistent half and should carry most of the between-player
   spread.
4. **Minutes** — `E[min | played]`, with the age curve and a ramp for players returning from
   a major injury.

Aggregate across simulated seasons to get the GP distribution and, composed with the rate
heads, the season-total distribution.

**Calibrate the clustering parameter against realized GP dispersion**, exactly as
`targets.py::expected_bonus` calibrates `overdispersion=0.10` against 11,938 player-seasons.
The target to hit is the **22.7×** figure and the left-tail shares (**26.7%** below 60
games, **9.6%** below 41) among established rotation players. A simulator that reproduces the mean but not
the tail has failed at the thing that matters.

### ✅ Decided 2026-07-29, and ✅ **built the same day** — `make stan-availability`

> **Done.** `src/models/stan_availability.py` + `src/stan/betabinomial_glm.stan`, on
> cmdstanpy 1.3.0 / CmdStan 2.39.0. Option **(b)**: the season-level beta-binomial in Stan,
> with the spell simulator still to be calibrated against its posterior predictive.
>
> **The port is verified, not assumed.** The prior is set to `normal(0, 1/sqrt(2·l2))`,
> which makes the posterior *mode* exactly the penalized MLE the existing head finds — so
> agreement is a check with a defined answer. On 10,361 train / 911 test:
>
> | | MLE | Stan plug-in | Stan posterior |
> |---|---|---|---|
> | CRPS (games) | 10.7952 | **10.7947** | 10.7953 |
> | R² on `gp_share` | 0.2831 | 0.2832 | 0.2832 |
> | PIT KS | 0.0963 | 0.0952 | 0.0963 |
> | ρ | 0.2757 | 0.2759 | 0.2759 |
>
> Max coefficient gap **0.0127**, largest gap **0.095 posterior sd**, and the MLE sits inside
> the 95% credible interval for **21/21** terms. R̂ **1.0025**, min ESS 2,402, **0
> divergences**, 254 s over 4 chains. Both `evaluate` and `crps` are imported from the MLE
> module rather than reimplemented, so a metric difference could not have been a
> metric-implementation difference.
>
> **The marginal metric was a wash, exactly as this plan predicted — and that was never the
> argument.** What the posterior buys is `Var_θ(Σ_i E[Y_i|θ])`: all players share β, so one
> draw moves the whole board together, and that term is **0 by construction** under any
> point estimate.
>
> ⚠️ **This plan oversold it, and the correction is measurable.** The independent term grows
> as sqrt(N) and the shared-β term as N, so their ratio scales as sqrt(N) and the *size of
> the portfolio* decides whether it matters at all. Measured on the 911-player held-out
> board with random subsets: **+0.2%** spread inflation on a 12- or 15-player roster,
> +0.3% at 30, +1.1% at 150, **+6.4%** across the whole board. So "how wrong could my whole board be at
> once" is a real question for **board-wide exposure across many lineups**, and very nearly
> a non-question for one drafted team. The 219-game full-board figure must not be quoted as
> if it applied to a 15-man roster.
>
> ⚠️ **One expectation was wrong and is worth recording.** The integrated predictive is *not*
> necessarily wider per player: the mixture adds `Var_θ(E[Y|θ])` but replaces `Var(Y|θ̄)`
> with `E_θ[Var(Y|θ)]`, and `n·μ(1−μ)·[1+(n−1)ρ]` is concave in μ, so Jensen pushes back.
> Measured at **+0.046 against −0.082 games²** — marginally *narrower*. The joint is the
> claim; the marginal width is not.

**Do it, and fit it separately from the minutes and component heads.** The generative structure
is a chain of conditionals (availability → `min | available` → counts `| min` → makes
`| attempts`), and with distinct parameter blocks the joint posterior **factorizes exactly** —
separate fits recover the identical posterior. `megamodel.stan` in the prior repo is the
cautionary case: no parameter was shared between any two heads, so it was thirteen independent
GLMs paying a joint-fit price, and it ran on `sample_frac(0.01)`. See `CLAUDE.md` for the full
argument, including where the simulator's correlation actually comes from (a shared `min` draw,
then a residual copula whose measured off-diagonals average +0.013).

Option **(b)** below is therefore the plan: season-level beta-binomial in Stan, spell simulator
calibrated to its posterior predictive. The reasoning that motivated the question in the first
place is kept below because it is still what justifies going Bayesian here at all.

The head as built (`BetaBinomialGLM`) is a point-MLE
fit: L-BFGS-B with an analytic gradient, coefficients and `ρ` fixed at their optima. It emits a
predictive distribution over games played, but **no parameter uncertainty** — every simulated
season is drawn from one fitted model. The proposal is to fit the same likelihood in Stan and
draw `(β, ρ)` from the posterior when simulating each player-game.

Why this is worth discussing rather than assuming:

- **The marginal metric will barely move.** At 10,361 training rows against ~15 features, the
  posterior mean and the MLE will be close and GP-marginal CRPS should improve very little. If
  the case is argued on CRPS it will probably fail, and that is the wrong reason to do it.
- **The real argument is the joint distribution**, which is what
  `docs/predictions-plan.md` needs and what nothing in this repo currently supplies. All players
  share `β`, so a posterior draw shifts *every* player's availability together — a genuine,
  correctly-calibrated source of cross-player correlation that comes free with the fit, rather
  than an invented copula. For a draft portfolio, "how wrong could my whole board be at once" is
  a different and more important question than "how wrong is this one player", and only the
  posterior answers it.
- **Compute is not an objection.** ~10,900 rows and ~20 parameters is a small Stan model.
- **The two recorded traps do not go away.** `n = max(team_games, gp)` is still required — a
  non-finite log-likelihood is *worse* under HMC than under L-BFGS-B, since it poisons the
  trajectory rather than just stopping the optimizer. The numeric-gradient convergence failure
  does disappear, since Stan supplies exact gradients by autodiff.

The open design question is how the posterior composes with the spell process above. Either
(a) fit the game-level spell/hazard model hierarchically in Stan directly, which gets the
per-game process and the posterior in one object but is a much larger fit, or (b) keep the
season-level beta-binomial in Stan and calibrate the spell simulator to match its posterior
predictive GP marginal per player. **(b) is the cheaper first step** and preserves the existing
validated head; (a) is where it ends up if the spell process needs its own partial pooling.

### Baselines it has to beat

Do not skip these; the ceiling is low enough that a simple model may well win.

1. **League/age baseline** — the README's standing instruction to "shrink hard toward a
   league/age baseline rather than the player's own prior GP." Given r = 0.316, this is a
   serious contender.
2. **Ridge** on the feature block above → the R² ≈ 0.17 ceiling.
3. **Beta-binomial GLM** on games played out of team games — the natural likelihood for a
   bounded overdispersed count, and it emits a distribution for free.
4. **GBM** on the same features, with a shuffled null from
   `feature_diagnostics.cell_importance` so any reported gain ships with its chance level.

### Minutes per game

Much easier, and worth keeping separate: MPG persists at 0.79. A per-36-style regression with
the age curve, role features and team context should get most of the way. The one
availability-specific wrinkle is the **post-injury minutes ramp** for players who begin the
season rehabbing — which is exactly the population the preseason snapshot identifies.

> ### 🔬 Measured 2026-07-29, before the head is built: **make the prior-MPG response
> nonlinear, and leave age alone**
>
> `models.availability.minutes_nonlinearity_probe` (a ridge stand-in, so the R² is not a
> claim about the eventual head — only about curved vs straight). Predicting next-season
> MPG, selection on a validation split and test for confirmation:
>
> | variant | val R² | test R² |
> |---|---|---|
> | linear (incl. `age + age_sq`) | 0.6768 | 0.6670 |
> | **+ quadratics** | **0.6914** | **0.6746** |
> | splines k=4 | 0.6930 | 0.6736 |
>
> Both splits move together, unlike the games-played arm where the same test is a null.
> Splining one column at a time says **`minutes_per_game_lag1` carries essentially all of
> it** (Δ val +0.0124, Δ test +0.0077), `total_minutes_lag1` adds a whisper, and everything
> else including `age` (−0.0008 / −0.0007) is noise or worse.
>
> **The load-management arc is real but already absorbed.** `make aging` has MPG peaking at
> 27 and reaching 0.515 by 37, so the shape exists — but `age + age_sq` already fits it, and
> a spline on age is *worse* than the quadratic. What actually pays is a **floor at the
> bottom of the prior-MPG range**: mean next-season MPG runs 4.3 → 10.5 (**+6.1**),
> 9.3 → 12.4 (+3.1), 15.1 → 15.8 (+0.7), then a roughly parallel −2.0 above 26. The local
> slope *rises* from 0.39 to ~1.0. Some of that is survivorship, since a 4-mpg player only
> appears if he has a next-season row at all.
>
> Two implications for the head when it is built: give prior MPG a spline or at least a
> quadratic, and do **not** spend flexibility on the age term.

> ### ✅ Built 2026-07-29 — `make stan-minutes`, and the probe's advice held
>
> `src/models/stan_minutes.py`, sharing `src/stan/betabinomial_glm.stan` with the
> availability head — the same likelihood with different data, which is the factorization
> argument as code. **Successes out of actual game length, never 48**: `y` = season minutes,
> `n` = summed game length over the games he played, so it is the conditional
> `min | available` and composes with the availability head rather than double-counting
> absences. 9,804 player-seasons, 9,048 train / 756 test, **0 rows clamped by rounding**.
>
> | variant | val CRPS | test CRPS | test R² | selected |
> |---|---|---|---|---|
> | `carry_forward` (no-fit floor) | 161.45 | 168.24 | 0.8166 | |
> | linear | 144.62 | 147.18 | 0.8565 | |
> | `logit(own)` | 145.44 | 147.35 | 0.8565 | |
> | `logit(own)` + quadratic | 144.83 | 147.21 | 0.8574 | |
> | **`logit(own)` + spline** | **144.13** | **146.85** | 0.8572 | **✓** |
>
> Selected on validation; the test column confirms rather than decides. It clears the no-fit
> floor by **+0.0407 R² and −21.4 minutes of CRPS**. R̂ ≤ 1.0093, **0 divergences** over 8
> fits, 2,183 s total.
>
> **The probe was right about where to spend flexibility, and the reason is now sharper.**
> Curvature on prior minutes pays and it replicates across both splits — but the *scale*
> change that is decisive for the component count heads is a **dead wash** here (0.8565
> against linear's 0.8565). So "put the predictor on the link's scale" does not generalize
> from the counts to this head, and the two answers should not be pooled into one rule.
>
> **Two dispersions, and the simulator needs the one this fit does not estimate.**
> Season-level ρ = **0.0495**; game-level ρ measured against each player-season's own mean
> over 713,947 player-games = **0.0776**, i.e. **4.65× binomial** at a 48-minute game. A
> season total cannot separate a per-game random effect from a per-season one. Drawing
> per-game minutes from the season-level ρ would make every simulated game far too close to
> the player's average — and the 2.43× block-variance inflation this repo already measured
> for minutes is a further, sequential effect on top.
>
> **Open defect**: the fitted heads carry a **−33 to −41 minute** held-out bias against the
> floor's −5.7 (about −2.7%), the price of shrinkage on a held-out season. It costs nothing
> on R²/MAE/CRPS but would compound through the eleven component heads that take these
> minutes as exposure. Worth a bias correction before the simulator consumes it.
> The post-injury minutes ramp is still not built.

---

### 🔬 Load management: is there a season × role interaction? — first look 2026-07-30, and yes

**Neither head carries any season term.** Both pool 30 seasons flat, which sits awkwardly
against this repo's own standing rule — *"ALWAYS absorb season when regressing on 30 pooled
seasons"*, a rule written after pooled `teammate_spacing` and `team_pace` correlations
collapsed from +0.315/+0.336 to +0.035/+0.091 under season fixed effects. The media story
about teams resting stars to bank mileage is the obvious mechanism for an era effect here, so
it is worth checking rather than assuming either way.

**A first look says the effect is real, large, and role-graded.** Mean `gp_share` on the full
window, players with ≥10 games, bucketed by that season's minutes per game:

| era | <12 mpg | 12–24 | 24–30 | **30+ mpg** |
|---|---|---|---|---|
| 2004–2010 | 0.408 | 0.728 | 0.842 | **0.877** |
| 2011–2016 | 0.403 | 0.707 | 0.849 | 0.856 |
| 2017–2022 | 0.355 | 0.656 | 0.790 | 0.817 |
| **2023–2025** | 0.371 | 0.635 | 0.759 | **0.776** |

Heavy-minute players lost **−0.101 of games-played share** across the window — about **8 games
of an 82-game season** — against −0.037 for the fringe bucket. The gradient runs the way load
management predicts: the more a player plays, the more availability he has lost.

**But it is availability, not workload — minutes given role are nearly unchanged:**

| era | <12 mpg | 12–24 | 24–30 | 30+ mpg |
|---|---|---|---|---|
| 2004–2010 | 8.47 | 17.89 | 26.89 | 34.83 |
| **2023–2025** | 8.06 | 17.87 | 26.95 | 33.13 |

Within a role bucket MPG is flat to three significant figures in the middle two buckets. Only
the 30+ bucket moved, −1.7 mpg. **So teams are playing stars fewer *games*, not shorter
ones** — rest days, not minutes limits. That is a clean split, and it says the era term
belongs in the **availability** head first and the minutes head second, if at all. The raw
34.2 → 31.5 MPG decline visible among ≥28-mpg players is mostly *composition* — the very top
of the minutes distribution being shaved — not players being pulled earlier.

**The mechanism label has drifted, which matters for the reason-split feature.** Among
≥28-mpg players, `missed_inactive` roughly doubled (≈5–9 games in 2006–2016 to **15–16** in
2021-22 and 2024-25/2025-26) while `missed_scratch` **collapsed** (≈1.0–1.6 games in 2011–2016
to **0.03–0.07** by 2022-25). Modern teams *deactivate* a rested star rather than dressing him
as DNP-Coach's Decision. This complicates the plan's headline finding above: `missed_scratch`
persists at 0.469 **pooled over 2006–2025**, but it has nearly vanished as a label in the
seasons the model is actually forecasting. Check whether that is a behaviour change or a
change in how the `comment` field is populated before building on it.

**Why this is not a simple "add season fixed effects" fix.** A season dummy for season S does
not exist at prediction time — that is the whole prediction-time constraint. So the options
are a smooth trend extrapolated one season forward, a recency weighting of training rows (the
repo already has a staleness prior, `×0.96`/season, which `persistence.csv` shows is
column-specific), or simply restricting the training window. All three are testable on
held-out CRPS against the current flat-pooled fit, and that comparison is the actual
experiment.

> ### ❌ The experiment ran, and the season × role interaction is a validation NULL
>
> **`make season-terms`** (`src/models/season_terms.py` →
> `outputs/predictions/season_term_metrics.csv`). Six arms on this head — the 2×2 over trend
> and year effect, plus the two role-interaction arms this section's finding calls for. Role
> is bucketed on **prior-season** MPG, so it is knowable before the season starts; bucketing
> on the target season's MPG would put the outcome in the design, which is the ⚠️ above.
>
> | arm | features | **val CRPS** | test CRPS | bias | 80% coverage |
> |---|---|---|---|---|---|
> | `carry_forward` (league/age) | 0 | 13.387 | 13.614 | +6.80 | 0.827 |
> | `base` | 19 | 10.007 | 10.797 | +1.68 | 0.857 |
> | **`trend`** (selected) | 20 | **9.997** | 10.765 | −1.29 | 0.866 |
> | `year` | 19 | 10.015 | 10.813 | +2.20 | 0.856 |
> | `trend_year` | 20 | 10.005 | 10.757 | −1.24 | 0.866 |
> | `trend_x_role` | 26 | **10.078** | **10.742** | −1.23 | 0.864 |
> | `trend_x_role_year` | 26 | **10.087** | **10.736** | −1.27 | 0.864 |
>
> **The two role arms are the best two on test and the worst two on validation.** That is
> the exact shape of the false positive this project has already shipped once — the
> nonlinearity arm whose paired bootstrap on test read [−0.079, −0.015] with P(Δ<0) = 99.7%
> and did not replicate — and it is caught here only because selection never reads the test
> column. Seven extra parameters buying −0.055 on one split and +0.071 on the other is
> noise.
>
> And the arm that *is* selected is worth **0.010 games of validation CRPS**, which is
> nothing. So: **the era effect is real in the league series and does not transfer into a
> better availability forecast.** The 2023-24 policy break is significant at p = 0.008 on
> `gp_share [30+ mpg]`, the role gradient is in the right direction and confirmed by two
> independent estimators — and none of it survives as a feature. The reason is visible in
> the ceiling: a *perfect* per-season league multiplier is worth at most 3.1% of MAE on any
> component, because player-level error dominates a league-level one.
>
> **What a season term IS worth on this head is spread, not accuracy** — see
> `docs/predictions-plan.md`, where a year effect widens a 15-man roster's season-total
> spread by +19.0% against shared-β's +0.2%. That is a simulator input, not a feature.

Two confounds sit inside the window and must be handled before any trend is fitted:

- **2019-20 and 2020-21 are a health-protocol regime**, not a load-management one — bubble,
  quarantines, and contact tracing. They are pooled in flat today.
- **The NBA's Player Participation Policy arrived in 2023-24** and penalises resting stars, so
  a policy *discontinuity* is plausible and a smooth trend extrapolated across it would be
  actively wrong. The first look shows no reversal (≥28-mpg `gp_share` reads 0.821 in 2023-24,
  0.759 in 2024-25, 0.742 in 2025-26), but that slice is composition-sensitive and is not a
  test.

> ### ✅ Both confounds are now tested rather than flagged — `make season-effects`
>
> `regime_tests` (`src/eda/season_effects.py` → `outputs/eda/season_effects_regimes.csv`).
> They get **different tests because they are different shapes**, which is the part worth
> carrying: COVID is a *transient regime* — two seasons unlike their neighbours, after which
> the league returns — so it gets an indicator that is zero for the forecast season; the
> Participation Policy is a *permanent* rule change, so it gets a break from 2023-24 on.
> Modelling COVID as a break would put the whole post-2021 series on the wrong intercept.
>
> **The policy break is real, significant, and role-graded exactly as the table above
> predicts.** A level-only break at 2023-24:
>
> | series | level break | p | one-season-ahead shift |
> |---|---|---|---|
> | **`gp_share` [30+ mpg]** | **−4.63%** | **0.008** | −2.90% |
> | `gp_share` [all] | −4.43% | 0.024 | −2.78% |
> | `gp_share` [24-30] | −4.92% | 0.066 | −3.08% |
> | `gp_share` [12-24] | −3.46% | 0.180 | −2.16% |
> | `gp_share` [<12 mpg] | **+5.37%** | 0.280 | +3.30% |
>
> The heavy-minute bucket loses and the fringe bucket does not — the sign even flips. That
> is the load-management gradient arriving as a dated policy step rather than as drift, and
> it is a second, independent confirmation from a different estimator than the era table.
>
> **COVID, by contrast, is undetectable at the league-rate level: 0 of 17 series are
> significant**, `gp_share [30+ mpg]` reading +0.22% at p = 0.91. The two seasons are strange
> in ways that a league *rate* does not see — the schedule was shortened, and `gp_share`
> normalizes by team games.
>
> ⚠️ **The break test disqualifies a trend rather than supplying a better one, and the
> column that says so is `regime_seasons` = 3.** A level+**slope** break fits its slope on
> the three post-break seasons alone; extrapolated one year further that moves the forecast
> by up to **+22.1%** (`gp_share [<12 mpg]`) and **+18.8%** (`stl`). Those are noise, not
> corrections. Read `regime_seasons` before `next_season_shift_pct`, and prefer the
> level-only arm — which is why both are emitted.

**One loose end this may or may not explain.** The Stan minutes head carries a **−33 to −41
minute** held-out bias against the no-fit floor's −5.7, and the floor is bias-free precisely
because it carries only the *immediately prior* season forward and is therefore era-current by
construction, while the fitted head shrinks toward a 30-season average. That is the signature
an era effect would leave. **But the naive direction does not match** — MPG within role is
flat or slightly falling, which would produce over-prediction, and the head under-predicts. So
treat this as a hypothesis with an obvious test, not as a diagnosis.

> ### ❌ FALSIFIED 2026-07-31 — the minutes bias is shrinkage, not era. `make season-terms`
>
> The obvious test ran. Adding a year-on-year trend to the minutes head makes the bias
> **worse by 15.5 minutes** — −41.0 on `base` against **−56.6** on `trend` and −56.6 on
> `trend_year` — on both arms that carry one. An era effect the head was failing to track
> would have been *corrected* by a trend, not amplified by it. So the bias is the fitted
> head shrinking toward a 30-season mean, and the era hypothesis is closed.
>
> **What does help is the year random effect, and it is the only head where one does.** It
> wins on **both** splits — validation CRPS **143.81** against base's 144.09, test
> **146.54** against 147.02 — which is the replication bar this repo insists on, and it
> nudges the bias to **−38.2**. That is a partial improvement rather than a fix; a genuine
> bias correction before the simulator consumes these minutes is still owed.

> ⚠️ **The buckets condition on the *same* season's MPG, which is endogenous to
> availability, and the full window counts waived and traded players as rostered all year.**
> Re-derive with the role bucket taken from **S-1** before treating the role gradient as
> settled.

**✅ Now reproducible — `make season-effects`** (`src/eda/season_effects.py` →
`outputs/eda/season_effects_{league_rates,summary,carry_forward_bias}.csv`). The
role-bucketed `gp_share` series above is `availability_rates`, and it sits alongside the
same decomposition for all eleven components and for minutes. What it adds:

- **`gp_share` is shock, not drift, at every role level.** Trend R² runs 0.45–0.74 with
  year-over-year sd of **2.8% (30+ mpg)** to **9.0% (<12 mpg)** — so even the star bucket,
  where the era story is strongest, is not a clean trend a fixed effect could extrapolate.
- **`minutes_share` is the steadiest quantity in the whole project** — 1.09× range, 0.92%
  yoy sd. That is a second, independent confirmation of the finding above that the era
  effect is in *availability* rather than in minutes-given-role.
- A **trend fixed effect and a year-level random effect do different jobs** and neither
  substitutes for the other: detrending shifts the mean of the year-over-year changes and
  leaves their variance exactly unchanged. For availability, that means a trend term could
  correct the systematic drift but only a year effect can represent the ±3–9% a season can
  move.

**⏰ This is an open TODO with a scoped session prompt** — see "TODO — season effects" in
`docs/predictions-plan.md`. For availability specifically the candidate is a **season × role
interaction**, not a level shift, because the decline is graded by minutes played.

## Validation

- **Temporal walk-forward by season**, matching the existing split. Hold out the last two
  seasons.
- **Distributional metrics, not just R².** CRPS and a PIT histogram over the predicted GP
  distribution, because the tail is the point. Report MAE and R² alongside for continuity
  with the rest of the repo.
- **The downstream metric that actually matters**: season DK total MAE with the availability
  head wired in, against the current implicit treatment. That is the number that justifies
  the work. ✅ **Answered — `make season-total`, and it justifies it.** See below.
- **Tail-specific reporting**: accuracy on the 9.6% of established players who fall below 41
  games. Aggregate metrics will hide this group entirely.
- **Ablation of the preseason snapshot**, which is the only genuinely new information: run
  the head with and without it. This isolates the value of Tier 2 and tells you whether the
  daily archiving is worth maintaining.
- **A leakage assertion in the test suite**: every training row's `as_of_date` precedes its
  season start.

---

## Staging

Conventions per `CLAUDE.md`: `python -m src.<module>` entry points, matching `Makefile`
targets in `.PHONY`, `cfg = yaml.safe_load(open("configs/default.yaml"))` in `__main__`,
`Path(...).mkdir(parents=True, exist_ok=True)` before writes, `f"... {n:,} ... → {dest}"`
progress lines, plain-`assert` tests with synthetic builders.

| Stage | Module | Make target | Notes |
|---|---|---|---|
| **A** | `src/data/injury_reports.py` | `injury-reports` | ✅ **done** — 175 reports archived, 2025-12-29 → 2026-07-19, 13,759 player-report rows. The whole retention window. |
| **A** | extend `src/data/injuries.py` | `injuries` | ✅ **done** — `snapshot_date`, `return_date_forecast`, `snapshot_as_of`, and a `--daily` no-op guard. |
| **B** | `src/data/boxscore_status.py` | `boxscore-status` | ✅ **done** — resumable per game, V2/V3 routing, manifest. The ~24,600-game run itself is long and is still executing. |
| **B** | extend `src/data/fetch.py` | `fetch` | ✅ **done** — 30 seasons of playoff logs (50,355 player-games) and `CommonTeamRoster` (13,268 player-team-seasons). |
| **C** | `src/features/availability.py` | `availability` | ✅ **done** — plus the three-way `status` column and the missed-reason split, added without restructuring. |
| **D** | `src/eda/availability.py` | `availability-profile` | ✅ **done** — plus `decomposition`, the reason-split retest, now run on the finished backfill (7,673 pairs, 234 measurement rows). |
| **E** | `src/models/availability.py` | `availability-model` | ✅ **done** — four baselines, CRPS/PIT. **The simulator was not built: see below.** |
| **D2** | `src/eda/report_calibration.py` | `report-calibration` | ✅ **done** — the designation → `P(play)` transfer function, measured on the 2025-26 archive/backfill overlap. Unblocks nothing else; de-risks stage A's payoff. |
| **G** | `src/models/season_total.py` | `season-total` | ✅ **done** — the downstream metric. Composes `gp × rate` with the rate model held fixed; the head is worth **−205 dk_pts MAE**. |
| **H** | `src/models/stan_availability.py` | `stan-availability` | ✅ **done 2026-07-29** — the point MLE ported to Stan and verified against it (21/21 coefficients inside the 95% interval). Supplies the posterior the simulator needs. |
| **H** | `src/models/stan_minutes.py` | `stan-minutes` | ✅ **done 2026-07-29** — `min \| available` as successes out of real game length. Clears its no-fit floor by +0.041 R². |
| **F** | dashboard tab | `dashboard` | Not started. A tenth tab over the availability artifacts, matching the existing read-only pattern. |

`make daily-capture` runs both stage-A captures and **must stay scheduled**. It is installed
as a launchd agent (`~/Library/LaunchAgents/com.nba-deep-learning.daily-capture.plist`)
rather than a crontab line, because macOS cron silently skips any run whose time passed while
the machine was asleep or off, while launchd re-runs a missed `StartCalendarInterval` job on
wake. `make capture-status` reports which days are archived, which had no report to archive,
and which were **missed**. 51 new tests; 308 pass overall.

---

## ⛔ Blocked — not skipped, and what unblocks each

One item in this plan cannot be completed yet; the second was resolved on 2026-07-28 and
its entry is kept for the record. They are recorded here rather than left
implicit, because both would otherwise look like omissions.

### 1. The preseason-snapshot ablation — blocked on the archive reaching a season boundary

This is the plan's **only genuinely new input** and the entire justification for stage A: a
dated injury snapshot at prediction date D, which the rest of the model does not have.
`src/models/availability.py` therefore measures the internal ceiling *without* it.

It cannot run yet. The archives that would supply it begin **2025-12-29** (NBA PDFs) and
**2026-06-28** (ESPN) and build forward only. Filling a historical training row from them
would mean reading a current-status source for a past date — the exact leak
"Point-in-time discipline" above forbids, and the one that would invalidate the head while
making it score *better*.

**Unblocked when** the daily capture has covered an offseason→season boundary, i.e. the
first prediction date D that has a dated snapshot at or before it. Practically: the
2026-27 season opener. Until then, `snapshot_source` stays `none` for every row.
The launchd dependency below is now resolved, so this is on track.

**What can be done before then, and has been:** the *transfer function* from a report
designation to a realized outcome does not need a season boundary — only an overlap between
the PDF archive and the box-score backfill, which already exists inside 2025-26. See
"Report calibration" below. That turns the October unblock into a measurement with a
pre-validated mapping rather than a mapping invented on the spot against one season.

### ~~2. The full-window decomposition verdict~~ — ✅ resolved 2026-07-28

`make boxscore-status` **finished**: 25,706 of 25,709 games across 2006-07 → 2025-26, ~30 h
wall clock plus a short second pass. `make availability && make availability-profile` were
re-run on it and the provisional block above is replaced with the full-window verdict —
7,673 pairs, +0.0285 above the shuffled null, ≈69 sd. Mean `status_coverage` went 0.079 →
**0.696** (the remaining 0.304 is the ten pre-2006-07 seasons, where no inactive list
exists), and the panel's status mix went from 523,022 `unknown` rows to 155,609.

**A second pass cleared both defects the main run left.** The backfill resumes from the
*data* file rather than the manifest, so anything that produced no rows is retried
automatically, and one re-run was enough:

- **All 85 missing 2025-26 playoff games recovered.** The startup race described below had
  never been cleared, so the manifest held only `225…` regular-season ids for that season.
  This mattered more than the count suggests: 2025-26 is the **test season** for every
  held-out evaluation in the repo. The cause was a race rather than a bug —
  `season_game_ids` reads `game_logs_playoffs_<season>.csv` if it exists; the first backfill
  started at 14:41 and enumerated 2025-26 first, but that file was not written until 14:53.
  Every later season picked its playoffs up. Still worth a guard: enumerate all seasons up
  front, or fail loudly when a season's playoff log is absent while neighbouring seasons
  have one.
- **All 68 transient network faults cleared on one retry** (connection resets and read
  timeouts against `stats.nba.com`, clustered in 2012-13 and 2013-14).
- **3 games remain unfetchable, and are not a defect in this repo.** `0022500259`–`0022500261`
  (2025-11-19) hit an unguarded `arena.get("arenaId")` inside `nba_api`'s *own* V3 parser,
  which raises in the constructor before this module's guard can run. Their raw payload is a
  stub — `arena`, `teamId` and `inactives` all null — and V2 returns 0 inactives because the
  date sits inside its silent-failure window, so there is nothing to recover. Deliberately
  left unfetched rather than recording the traditional side alone, which would mislabel
  their inactives as `not_rostered`. See `CLAUDE.md` under "Data quirks".

> **Count games, not manifest rows.** The manifest is append-only: a retried game keeps its
> failure row *and* gains a success row. After the second pass it holds **77 error rows**
> against **3** games that actually lack data. Dedupe to games with no successful attempt
> before reporting, or a fully-recovered backfill reads as broken.

> **A non-defect, recorded because it looks exactly like the defect this plan warns about.**
> 34 games report zero inactives, which is the `BoxScoreSummaryV2` silent-failure signature.
> It is not. All 34 are playoff games (`42…` ids) with exactly **30 dressed** — playoff teams
> activate all 15, so an empty inactive list is correct. The V2 trap would have shown up as
> whole *date ranges* of zeros including regular-season games, not as 30-dressed playoff
> games. Check the dressed count before concluding the endpoint failed.

### ~~Operational blocker on both: macOS TCC~~ — ✅ resolved 2026-07-27

The launchd agent could not read the repo, because it lives under `~/Documents`, which macOS
protects: the job could enter the directory but not read files in it, and exited 2 with
`Operation not permitted`. Full Disk Access for `/bin/zsh` fixed it.

**Verified, not assumed.** `launchctl kickstart -p gui/$(id -u)/com.nba-deep-learning.daily-capture`
now runs to completion and `launchctl list` reports last exit status **0**; the
`getcwd: Operation not permitted` lines at the top of `data/raw/daily_capture.log` are the
stale pre-fix run. Check it the same way after any OS update — TCC grants are revocable, and
this failure is silent from the archive's point of view.

**Cost of the outage: 28 ESPN snapshot days, permanently lost** (2026-06-29 → 2026-07-26 —
`make capture-status` lists them). They are offseason days, so the loss is small, but it is
the exact failure mode this plan warns about and it is worth recording that it happened
rather than quietly re-basing the archive. The NBA PDF side lost nothing: 175 reports
archived, **0 missed**, the whole retention window.

### Not blocked, just not started

Stage F (the dashboard tab). ~~the **minutes-per-game head**~~ — ✅ **built 2026-07-29**, see
"Minutes per game" above; `make stan-minutes`. The **spell simulator** also remains unbuilt,
deliberately: stage E's decision rule says build it only if the baselines are beaten, and
they were not. What has changed since that rule was written is that the Stan head now
supplies a *posterior* to calibrate a simulator against, which is option (b) of the design
question above.

### What stage E answered

**The baselines were not beaten by anything more sophisticated, so the spell simulator is
not built.** Held out on 2024-25 and 2025-26 (10,361 train / 911 test), CRPS in games:

| Model | CRPS | MAE | R² | PIT KS | implied overdispersion |
|---|---|---|---|---|---|
| **beta-binomial GLM** | **10.795** | 15.39 | 0.283 | 0.096 | 23.3× |
| GBM | 10.888 | 15.41 | 0.262 | 0.079 | 19.9× |
| ridge | 10.896 | 15.39 | 0.275 | 0.103 | 22.7× |
| league/age baseline | 13.614 | 18.94 | −0.084 | 0.174 | 29.5× |

> Updated 2026-07-29 with the playoff/mileage workload block (19 features, was 15). The
> pre-block figures were GLM 10.914 / ridge 10.98 / GBM 11.04. The GBM gained slightly more
> than the GLM did (−0.152 against −0.119), narrowing the margin from 0.126 to 0.093 without
> changing the ordering. See "Workload" below for the ablation.

The GBM does not beat a 19-feature GLM, which is the plan's own stopping condition. Two
independent checks say the *distribution* is the part doing the work, exactly as the
overdispersion finding predicted:

- The dispersion fitted purely by maximum likelihood lands at **20–30× implied
  overdispersion**, recovering the ~20× measured separately in `availability_profile.csv`.
- On the left tail, the GLM predicts **15.0% / 34.8%** of established rotation players below
  41 / 60 games against **11.8% / 36.9%** observed. The league/age baseline reads 24.8% /
  45.0% — it produces the tail by being vague about everyone.

Held-out R² of 0.268 is **not** the 0.236 ceiling beaten: that figure is in-sample and
season-absorbed, a different quantity.

Two traps cost real time and are recorded in `CLAUDE.md`: 13 traded players have
`gp > team_games`, which makes the summed log-likelihood non-finite at *every* ρ; and a
joint numeric-gradient fit over 17 parameters silently fails to converge because the
objective is ~1e5 while a finite-difference step moves it by ~1e-3.

### What stage G answered — the number that justifies the work

Stage E measured the head in **games**. Games are not the deliverable, and the plan flagged
that gap itself. `src/models/season_total.py` closes it: it composes
`season_total = gp × dk_per_game_played`, holding a **fixed** rate model (a plain ridge,
held-out R² 0.770 on the rate) and varying only the games-played treatment, so every
difference below is the availability head and nothing else. Held out on 2024-25 and
2025-26, 10,294 train / 896 test.

| GP treatment | MAE | RMSE | R² | bias | CRPS |
|---|---|---|---|---|---|
| full season (the naive case) | 646.3 | 831.2 | 0.141 | **+541.9** | — |
| prior GP carried forward | 475.0 | 638.5 | 0.493 | +41.9 | — |
| league/age baseline | 476.0 | 595.8 | 0.559 | +23.2 | 340.8 |
| **beta-binomial head** | **435.1** | **570.8** | **0.595** | **+6.1** | **316.9** |
| *oracle rate* (predicted GP, true rate) | *302.7* | *427.7* | *0.773* | *+5.3* | — |
| *oracle GP* (true GP, predicted rate) | *221.3* | *304.2* | *0.885* | *−33.2* | — |

> ⚠️ **The R² column was corrected 2026-07-30**, from 0.10 / 0.47 / 0.55 / 0.59 / 0.78 /
> 0.88. The partial-refresh note further down this list — "figures updated 2026-07-29 from
> 441.3 / −205.0 / 508.9" — is the direct evidence: the MAE side of this table was refreshed
> for the playoff-workload change and the R² column was not. But **it is not simply a
> pre-workload snapshot either**, which was the obvious reading and is falsified:
> `full_season`, `prior_gp` and `oracle_gp` never touch the availability head, so the
> workload change cannot move their R², and `full_season` was the most wrong of the six (by
> 0.041). Since equal RMSE pins SSE, only `ss_tot` could differ, and solving per row gives
> six mutually inconsistent denominators. The column was hand-typed and never recomputed.
> Reproduce by recomputing R² from `outputs/predictions/season_total_predictions.csv`,
> which matches `season_total_metrics.csv` exactly on all six rows.

- **The head is worth −211.1 dk_pts of season-total MAE (−32.7%)** against assuming a full
  season, and −39.9 against carrying prior games forward. On established rotation players
  it is −151.9 (651.3 → 499.4, −23.3%) — smaller, because regulars miss less time, so the
  aggregate figure should not be quoted as if it applied to them.
- **The oracles settle which half dominates, which was previously an assertion.** Perfect
  games played gives 221.3 MAE against perfect rate's 302.7: availability carries
  **213.8** dk_pts of the remaining error and the rate **132.5**. Games played really is
  the larger lever on the season total, and it is now measured against its own alternative
  rather than argued from persistence coefficients.
- **Bias is where the naive treatment dies.** +541.9 of its 646.3 MAE is bias — it assumes
  82 games for everyone. The head's +6.1 is essentially zero, which is the calibrated
  distribution doing exactly the job stage E's PIT said it would.
- `oracle_gp` is **invariant to the availability head by construction** (true GP × predicted
  rate), so it holding at 221.3 across the workload change is the check that only the GP
  treatment moved. Figures updated 2026-07-29 from 441.3 / −205.0 / 508.9.
- CRPS is computed on the **pushforward** of the GP pmf through `k → k·rate`, so the
  season-total distribution is scored in dk_pts rather than only its mean. The O(K) kernel
  reduction in `crps_from_atoms` is pinned against the literal double sum in
  `tests/test_season_total.py`, because it is the one piece of non-obvious math here.

> **What this is not.** The rate model is deliberately plain and is *not* the project's
> eventual component-head model; its R² should not be read as one. It is identical across
> every row, which is all the availability contrast requires. And `E[GP × rate] =
> E[GP] × E[rate]` assumes conditional independence given the features, which is not exactly
> true — `oracle_rate` carries that residual.

**The preseason-snapshot ablation cannot be run yet.** It is the one genuinely new input
this plan identifies, but the archives that would supply it start on 2025-12-29 and build
forward only. Filling a historical row from them would mean using a current-status source —
the exact leak "Point-in-time discipline" forbids. The ablation waits until the daily
capture has covered a season boundary.

---

## Risks and open questions

- **The ceiling may simply be low.** R² ≈ 0.24 in-sample from internal data, no durability
  latent, and injury severity at r = 0.090. It is entirely possible that the preseason
  snapshot plus a well-calibrated distribution is the whole win, and that the sophisticated
  modeling adds little. Stage D was built to find that out cheaply, and its answer so far is
  "the ceiling is low" — so **stage E should start with the baselines and stop if they are not
  beaten**, rather than building the simulator first.
- **The snapshot's value is concentrated in a small population.** Only ~6.4% of
  player-seasons end S-1 on an absence of 21+ games. High leverage on those, near zero
  elsewhere — which is fine, since those are exactly the players a season-total forecast gets
  most wrong.
- **The prediction date D is a free parameter.** Forecasting on 1 August and on 15 October
  are different problems, and the later date is strictly more informative. The panel should
  carry `as_of_date` so a model can be trained and evaluated at several D values; the choice
  is a product decision, not a modeling one.
- **Historical depth of injury *reasons* stops at 2006-07**, and the richer reason vocabulary
  in the official PDFs only exists going forward from now. Training rows before 2006-07 can
  carry availability but not its decomposition.
- **`InactivePlayers` conflates causes.** A player can be inactive for injury, personal
  reasons, G-League assignment or roster mechanics. The `comment` field disambiguates some
  but not all. Expect to hand-label a reason vocabulary; budget for it.
- **Mid-season trades remain out of scope**, consistent with the rest of the project, and
  they interact with this head: a traded player's roster window changes mid-season.
