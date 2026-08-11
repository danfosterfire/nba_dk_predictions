# Average Draft Position: Sourcing, Storage, and Where It Belongs

ADP is the only input in this project that describes **the field rather than the player**. Every
other feature tries to predict what a player will score; ADP predicts what the other 11 seats in
the draft pod will do. That makes it useless for the thing the rest of the repo optimizes and
indispensable for the thing `docs/simulations-plan.md` cannot build without it — an opponent
model.

This plan covers what ADP data actually exists, what has to be captured and by when, how to store
it under this repo's point-in-time rules, and — the fork the earlier planning left open — whether
it feeds the prediction layer or stays downstream in the strategy layer.

Every number below was measured during planning on 2026-07-28. Where a measurement came from a
scratch script rather than a `make` target, that is stated; Stage A turns them into targets.

> **Two populations, both correct — do not reconcile them.** "What I measured before planning"
> and "The short version" quote the scratch run on the **218** pairs a looser join produced;
> "[What implementing it changed](#what-implementing-it-changed--measured-not-planned)" quotes
> `outputs/eda/adp_profile.csv` on the **226** pairs the guarded cascade recovers. Only the
> second set is reproducible, and `make docs-audit` audits only that set — the planning figures
> are registered as `historical`, so they are checked for *deletion* rather than for agreement.
> Correcting them to the artifact would erase the record of what implementing the plan changed,
> which is the point of keeping both.

## Current status

> ⏰ **TODO — 2026-08-01: run `make adp-fantasypros -- --backfill`** (i.e.
> `.venv/bin/python -m src.data.adp_fantasypros --backfill`). Only **23 of 259** archived
> snapshots are on disk. The sweep was blocked on 2026-07-28 by Wayback throttling
> (**HTTP 498** — its rate-limit code, returned after the planning-session sweep), which
> clears on its own. The run is resumable and skips anything already archived, so
> re-running costs nothing and a partial run is safe to repeat. **Delete this note once it
> has completed** and update the season-coverage table in `docs/adp-plan.md`.

**DraftKings ADP is the second thing in this repo with a deadline, and it is a manual
one.** The board is login-gated, has **zero** Wayback snapshots, exposes no API, and is
live only while contests are open (~Oct). It cannot be scraped or backfilled: a board not
downloaded from the draft lobby while it is open is gone permanently. Two are captured
(2025-10-17, 2026-07-28); the load-bearing one still to get is an **early-to-mid October
2026** board, timing-matched to the 2025 anchor. **See `docs/adp-plan.md`.**

**Tier A** = 30 seasons, box-score families, 10,900 player-seasons × 150 features.
**Tier B** = 13 seasons (2013-14+), adds tracking/hustle, 5,077 × 287, strict column
superset of Tier A. Artifacts that later modeling consumes go to `data/features/`;
human-readable analysis reports go to `outputs/eda/`.

---

## The short version

- **DraftKings NBA Best Ball ADP has no history and no feed.** DK's best-ball pages have **zero
  Wayback snapshots**, there is no unauthenticated API, and every third-party DK ADP tracker found
  is NFL-only. Nothing can recover history that was not captured live.
- **Two DK boards are now held** — 2025-26 (2026-07-28 shows the 2026-27 board is already open).
  `DK ID` is a **persistent player key** (667 shared ids, 100% name agreement), so the DK side of
  the join is solved permanently.
- **A 12-season market-proxy series is available and parses**: FantasyPros' NBA ADP page, via the
  Wayback Machine, covers **2014-15 → 2025-26 with no missing season**.
- **The consensus proxy is good but materially biased**, and the bias is structural, not noise:
  Spearman ρ = 0.870 against the DK anchor, mean absolute rank gap **21.9 picks**, and DK drafts
  centers **11.9 picks earlier** than a Yahoo/ESPN consensus — the category-league artifact,
  measured rather than assumed.
- **The fork resolves to neither stated option.** A one-dimensional **monotone recalibration** of
  consensus onto DK's scale cuts cross-validated error from 24.0 to **17.0 picks** (−29%), while
  the obvious structural feature (position offset) adds only −0.3 further. One anchor can identify
  a shape; it cannot identify a player-level model, and it does not need to.
- **Carrying DK ADP forward is not a substitute for the consensus pipeline** — 26.7 picks CV MAE
  against the recalibration's 17.0, and structurally blind to the players who moved most
  (2026 rookies, plus Haliburton and Lillard returning from lost seasons).
- **Urgency is real but is not the injury-PDF pattern.** FantasyPros freezes for ~11 months, so
  its capture has slack. DK's board is login-gated, unarchived, and live only while contests run.
  **The 2026-27 board is open now and is being captured manually**; the load-bearing one is an
  **early-to-mid October 2026** board, timing-matched to the Oct-2025 anchor.
- **The second anchor is blocked on FantasyPros flipping to 2026-27, not on DK.** The DK side is
  already in hand.
- **ADP belongs in the strategy layer, not the GLMM** — with one narrow, principled exception for
  thin-data players.

---

## What ADP is for, and why that decides everything downstream

`docs/simulations-plan.md` establishes that Round 1 is a **zero-consolation knockout in all five
real tournaments** — top 2 of 12 advance, nothing for ranks 3–12. Under that payout, being right
is not the objective; **being right where the field is wrong** is. Two distinct uses follow, and
they want different things from the data:

1. **Opponent modeling in the draft simulator.** Opponents draft off DK's default board plus
   noise (the rules doc's autodraft logic: queue → pre-draft ranking → 8G/8F/3C caps). This wants
   DK's *actual* ADP, on DK's scale, because it determines *who is still on the board at pick k* —
   the single most important quantity in a snake draft.
2. **Differentiation measurement.** Knowing how *owned* a player will be, so a strategy can pay
   for leverage instead of paying for consensus. This wants relative ownership, and tolerates a
   proxy.

Use 1 is the one that cannot be faked, and it is the reason the DK anchor matters more than its
sample size of one suggests.

---

## What I measured before planning

### The DK anchors — what is actually in the files

**Two boards are now in the repo**, both manually downloaded from the DK draft lobby:

```
data/raw/dk_draft_rankings/DkPreDraftRankings_Oct17_2025.csv    # 2025-26 board
data/raw/dk_draft_rankings/DkPreDraftRankings_July28_2026.csv   # 2026-27 board
```

| | Oct 17 2025 | Jul 28 2026 |
|---|---|---|
| Season | 2025-26 | **2026-27** |
| Player pool rows | 698 | **942** |
| Rows carrying an ADP value | **249** (35.7%) | **202** (21.4%) |
| ADP range | 1.053 → 186.3 | 1.139 → 184.1 |
| ADP p25 / p50 / p75 | 63.0 / 126.1 / 178.6 | 53.2 / 101.1 / 153.7 |
| Distinct team codes | 30 | **31** |
| Days before season start | ~4 | ~84 |

Columns are `ID, Name, Position, ADP, Team` plus a stray blank column and an `Instructions`
column. The July file is unambiguously a **2026-27** board: the 2026 draft class carries ADP
(AJ Dybantsa 41.8, Cameron Boozer 47.2, Darryn Peterson 53.2) and **178 of 667 returning players
changed teams** (Giannis → MIA, LeBron → PHI, Harden → CLE). The 31st team code is **`NO`**, a stray variant of `NOP` rather than the `FA` pseudo-team the
plan guessed; stage A's alias map collapses it, so `adp_draftkings.parquet` carries 30.

Facts that shape everything else:

- **The ADP is a real, live-computed average, not a rank.** Seven significant figures
  (`1.0526223`) means DK is dividing a sum of pick numbers by a draft count that changes
  continuously. It is a *current-status* quantity in the precise sense this repo uses the term.
- ✅ **`ID` is a stable persistent DK player key** — 667 ids appear in both files and the name
  agrees on **100.0%** of them (Jokic = 830650 in both). So the DK side of the join is solved
  permanently: **map DK `ID` → `nba_api` `player_id` once, by hand if necessary, and reuse it
  forever.** Only the consensus sources need fuzzy name matching. This retires most of the
  name-matching risk this plan originally carried.
- **The ADP column is right-censored** — ~200–250 players over a 192-pick draft, maxing near 185.
  Players drafted rarely pile up against the boundary rather than extending past it.
- **Board thickness depends on how long drafts have been running.** The July board has 202 ADP'd
  players against October's 249, and is less compressed (median 101 vs 126) because fewer drafts
  have resolved the deep tail. **A July board and an October board are not the same measurement.**

### FantasyPros via Wayback — it parses, with three schema breaks

CDX index (`web.archive.org/cdx/search/cdx?url=fantasypros.com/nba/adp/overall.php`, requires a
browser UA and tolerates ~1 request per 10 s before 503-ing): **259 snapshots with status 200,
2014-10-15 → 2026-07-02.**

13 snapshots were downloaded across 12 years and parsed. **11 parsed cleanly; 2 failed**, and
both failures are FantasyPros' own `"Sorry, this report is not available at the moment"`
placeholder on early-September dates — a page state, not a parser problem, and both years have
dozens of other snapshots.

`<table id="data">` is present in **every** snapshot from 2014 to 2026. That single stability is
what makes a uniform parser possible. What changes:

| break | detail |
|---|---|
| **Column set** | `Rank, Player (Team), POS, Yahoo, ESPN, CBS, AVG` (2014–15) → `Rank, Player, Yahoo, ESPN, CBS, AVG` (2017+, POS folded into the player cell) → **CBS dropped** in 2020, 2022, 2025 → CBS present again 2023 |
| **Player cell** | `Kevin Durant ( OKC )` → `LeBron James CLE` → `Russell Westbrook (OKC - PG)` — three formats, plus a trailing live-status token (`DTD`, `OUT`) |
| **Free agents** | `Patric Young (FA - PF,C)` — `FA` appears as a team code |

> **The CBS column answers a research question directly.** The premise going in was that
> FantasyPros aggregates "Yahoo + ESPN only — not CBS". That is true of the *current* page and
> false of the archive: CBS is a first-class column in 2014, 2015, 2017, 2019 and 2023.
> **A CBS-specific series is recoverable from the same page, so no separate Yahoo/ESPN/CBS scrape
> is warranted.** Which sources are present must be read per snapshot, never assumed.

Row counts run 150–273. Two gotchas:

- **Wayback's `id_` (raw) endpoint returns the originally-stored gzip.** Snapshots from 2022 on
  are gzip-compressed on disk; naively decoding them as UTF-8 yields "no table" and looks exactly
  like a JS-rendered page. It is not — sniff for the `1f 8b` magic and inflate. This cost real
  time and would have led to a wrong conclusion that modern snapshots are unusable.
- **`AVG` is not always the mean of the displayed source columns.** It matches exactly in 12 of
  19 snapshots and diverges in the rest (2022-10-04: Yahoo 4, ESPN 5, `AVG` 3.5). Either
  undisplayed sources contribute or the displayed integers are rounded. **Take the site's `AVG`
  as given; do not recompute it.** Row counts also wobble on adjacent days (148 vs 203 in October
  2022), which is the archive catching the page mid-recompute — flag those snapshots rather than
  trusting them.

### The freeze rule — the finding that closes every coverage gap

Counted by calendar year, preseason (Sep–Oct) snapshots look fatally uneven: 2016, 2017 and 2024
have **zero**. That reading is wrong, because the page does not update continuously.

Measured on identical-`AVG` share across consecutive 2025 snapshots:

| pair | overlap | identical `AVG` |
|---|---|---|
| 2025-01-19 → 2025-07-10 | 260 | 33.8% |
| 2025-07-10 → 2025-07-20 | 265 | **100.0%** |
| 2025-07-20 → 2025-09-06 | 265 | **100.0%** |
| 2025-09-06 → 2025-10-05 | 217 | **0.9%** |

**The table is frozen at the completed season's ADP for most of the year and flips only when the
next season's drafts begin.** The flip between 2025-09-06 and 2025-10-05 is total. The content
confirms it beyond doubt: Haliburton reads `7.0` and Tatum `8.5` before the flip and **both read
`141.5` after** — the market repricing two Achilles tears from the 2025 playoffs. Deep players
keep drifting mid-season (the 33.8% row), so the top of the board freezes earlier than the tail.

Two consequences, and the first is a trap:

> ⚠️ **A snapshot's calendar date is not its season.** The 2025-09-06 snapshot looks like a
> 2025-26 preseason capture and is in fact **2024-25 ADP**, unchanged since the previous
> October. Labeling by date would mislabel it by a full season — silently, with a plausible-looking
> table. Season must be assigned by **detecting the flip against the neighbouring snapshot**,
> never by a month cutoff.

> ⚠️ **And the calendar rule fails outright on 2020-21.** That season tipped off 2020-12-22, so
> drafts ran in December. Verified: the 2020-10-23 snapshot is **identical** to 2020-09-16 —
> Anthony Davis / Giannis / Curry / Harden / Towns — i.e. still 2019-20. Any `month >= 10` rule
> assigns six 2020 snapshots to the wrong season.

Applying flip detection rather than a calendar rule, **every season from 2014-15 to 2025-26 is
covered** — the apparent 2016, 2017 and 2024 holes close because an off-season snapshot preserves
that season's frozen board:

| season | usable snapshots | earliest |
|---|---|---|
| 2014-15 | 10 | 2014-10-15 |
| 2015-16 | 14 | 2015-10-07 |
| **2016-17** | **2** | 2017-02-02 |
| 2017-18 | 50 | 2017-11-18 |
| 2018-19 | 79 | 2018-10-03 |
| 2019-20 | 54 | 2019-10-07 |
| 2020-21 | 26 | 2020-10-23 *(needs flip detection — see above)* |
| 2021-22 | 5 | 2021-10-26 |
| 2022-23 | 7 | 2022-10-02 |
| 2023-24 | 3 | 2023-10-17 |
| **2024-25** | **4** | 2025-01-19 |
| 2025-26 | 5 | 2025-10-05 |

2016-17 and 2024-25 are recoverable **only** from mid-season snapshots, so their deep-player ADP
carries the drift the 33.8% row measures. Record a `snapshot_lag_days` per row and treat those two
seasons as lower-confidence rather than dropping them.

### The crux: DK versus consensus, measured on the one paired observation

FantasyPros 2025-10-05 (255 rows) against the DK anchor 2025-10-17 (249 with ADP), joined on a
normalized name (NFKD → ASCII, suffixes and punctuation stripped):

| | |
|---|---|
| Matched | **218 of 249 (87.6%)** |
| Spearman ρ | **0.8704** |
| Pearson r | 0.8911 |
| Mean absolute rank gap | **21.9 picks** |
| Median / p90 | 14.2 / 50.0 |
| Gaps > 20 picks | **38.1%** |
| Gaps > 40 picks | **16.5%** |

The 12.4% unmatched are a mix of genuine pool differences and name variants that a stricter
normalizer must handle — `Kristaps Porzingis` (diacritics), `Alexandre Sarr` vs `Alex Sarr`,
`Fred VanVleet`. This is the same class of problem
`report_calibration.py` solved to **0.0% unmatched names**; hold this join to that standard.

**A 21.9-pick mean gap is ~1.8 rounds in a 12-team draft.** Consensus is not a drop-in for DK's
board.

**The bias is structural and the mechanism is the predicted one:**

| DK position | mean signed rank gap | n |
|---|---|---|
| **C** | **+11.86** (DK drafts earlier) | 38 |
| F | +0.16 | 83 |
| G | −4.78 | 97 |

Yahoo/ESPN ADP is **category-league** ADP, where centers are discounted for wrecking FT% and
supplying no 3PM or assists. DK Best Ball is a **points** league that pays rebounds 1.25 and
blocks 2.0 flat. The top of the "DK drafts earlier" list is five centers — Wendell Carter Jr.
(+99), Yves Missi (+94), Ryan Kalkbrenner (+92), Mitchell Robinson (+62), Neemias Queta (+58).
**This is a scoring-system difference, not sampling noise, and it will recur every season.**

**Disagreement is concentrated in exactly the rounds that fill most of the roster:**

| consensus tier | mean abs gap | n |
|---|---|---|
| Rounds 1–2 | **5.1** | 23 |
| Rounds 3–4 | 9.7 | 23 |
| Rounds 5–8 | 10.5 | 43 |
| **Rounds 9+** | **30.8** | 129 |

The elite tier is nearly interchangeable between sources; the late rounds are close to
uncorrelated. Since 16 rounds means **9 of 16 picks land in "rounds 9+"**, a proxy that is only
accurate at the top is accurate where it matters least.

### Resolving the fork: how much transfer function can one anchor support?

Five-fold cross-validated, predicting DK ADP from the consensus `AVG`, n = 218:

| approach | CV MAE (picks) |
|---|---|
| consensus `AVG` used raw as DK ADP | 24.02 |
| consensus *ordering* vs DK ordering | 21.89 |
| + global linear rescale | 21.24 |
| **+ monotone (isotonic) rescale** | **16.99** |
| + linear + C/F/G offset | 20.78 |
| + isotonic + C/F/G offset | 16.70 |

Read this carefully, because it does not say what either candidate option assumed:

- **Option (b) — "use consensus directly" — leaves 7.0 picks per player on the table**, 29% of the
  error. The gap is not model-able opinion, it is **scale**: DK's 249 players compressed into
  1–186 against FantasyPros' 250+ ranks over a wider spread. A monotone map fixes curvature and
  range, which is a shape with roughly one degree of freedom.
- **Option (a) — "predict DK's ADP from other sources" — overreaches.** The most obvious
  structural feature, the position offset whose effect is plainly real (C = −5.3 picks after
  rescaling), buys only **−0.29 picks** on top of the isotonic map, because within-position sd is
  18–26 picks. If the strongest structural covariate is worth 1.7% of the error, a richer model
  fit to a single anchor is fitting noise.
- **Censoring is where the residual lives.** 54 of 218 players sit above ADP 170. Restricted to
  the uncensored region, isotonic CV MAE falls from 17.0 to **14.4 picks**. Model the censoring
  explicitly (a Tobit-style boundary, or simply flag `adp_censored`) rather than letting the
  pile-up drive the fit.

> **Decision: fit a monotone (isotonic or rank-quantile) recalibration from the consensus series
> onto DK's ADP scale, anchored on the single paired observation, plus a fixed C/F/G offset kept
> for interpretability rather than accuracy.** Re-fit — do not merely re-validate — the moment a
> second anchor exists. One pair identifies a shape; two pairs begin to test whether the shape is
> stable. Until then the recalibration ships with `n_anchors = 1` recorded beside it, and any
> backtest that depends on it reports results with and without.

### What the second DK board does and does not buy

The 2026-27 board arrived on 2026-07-28. It does **not** yet give a second anchor, for a reason
the freeze rule predicts: **the consensus side has not flipped.** The live FantasyPros page on
2026-07-28 is titled *"Fantasy Basketball 2025-26"*, contains no 2026 draft class
(Dybantsa/Boozer/Peterson all absent), and is still serving the frozen 2025-26 board. So we hold
DK's side of a pair whose counterpart arrives around **October 2026**.

> ⚠️ **And it is decorated with live data, which is the trap this plan already warned about, now
> confirmed on the live page.** That same frozen-2025-26 table lists Giannis on **MIA** and Anthony
> Davis on **WAS** — 2026 offseason teams — with `DTD`/`OUT` tokens describing July 2026. The ADP
> column is 2025-26; the team and status columns are today. **Three columns of one table have three
> different as-of dates.** Store `adp_as_of` (flip-detected) separately from `captured_at`, and
> never read team or status as contemporaneous with the ADP.

What it *does* buy is a measurement that was not previously possible, and it settles a question
the sourcing strategy depends on: **can DK ADP simply be carried forward, skipping the consensus
pipeline entirely?** 175 players carry ADP on both boards:

| predictor of a DK board | CV MAE (picks) | ρ |
|---|---|---|
| prior-year DK ADP, raw | 30.29 | 0.743 |
| prior-year DK ADP + isotonic carry-forward | 26.71 | — |
| **contemporaneous consensus + isotonic** (from above) | **16.99** | 0.870 |

**No — carry-forward is ~10 picks worse, so the FantasyPros pipeline earns its place.** The
comparison is deliberately unfair in carry-forward's favour to state it, and it still loses: the
consensus task is *contemporaneous* while carry-forward is a year ahead. They are complements, not
substitutes. Two structural reasons carry-forward fails, both visible in the data:

- **It is blind to exactly the players whose value moved most.** 13.4% of 2026-27 ADP'd players
  have no prior-year DK ADP — the 2026 rookies, plus **Tyrese Haliburton (ADP 20.5)** and
  **Damian Lillard (85.4)**, who were undrafted in 2025-26 while injured and are now back.
- **A year of roster churn intervenes** — 178 team changes between the two boards.

Three things the second board *does* settle:

- **The compression the isotonic map corrects is a stable property of DK's format**, not a
  one-season accident — both boards show the same squashed shape against a 192-pick draft. That
  partially de-risks the "re-fit annually, the map may not be constant" concern below.
- **The DK-side join is solved permanently** (the persistent `ID`, above).
- **Capture timing is now a known variable, not an unknown one.** A July board is thinner and
  less compressed than an October one, so the October 2026 capture should be **timing-matched** to
  the Oct-2025 anchor rather than treated as interchangeable with July's.

### Sources checked and rejected — recorded so they are not re-checked

Per the standing rule in `docs/availability-plan.md` (check for a mirror before scraping):

- **No mirror of historical fantasy-basketball ADP exists on GitHub or Kaggle.** Searching
  returns *NBA draft* datasets (real-draft prospects, 1947→present) — a different quantity that
  shares a word. Nothing carries preseason fantasy ADP by season. **A scraper is therefore
  warranted**, which was not a foregone conclusion.
- **DraftKings has no NBA ADP tracker, first- or third-party.** DK Network, Establish The Run,
  Occupy Fantasy, Footballguys and Draft Sharks all publish DK Best Ball ADP trackers — **all
  NFL**. NBA Best Ball is too small a market. Nothing to piggyback on.
- **`api.draftkings.com` is live but exposes nothing relevant unauthenticated.**
  `/draftgroups/v1/draftgroups` returns `400 SPO117` with or without a sport parameter;
  `/bestball/v1/predraftrankings` and `/sites/US-DK/bestball/v1/rankings` both 404. The
  `draftkings.com/2025-nba-best-ball` page serves 200 (692 KB) but references pre-draft rankings
  only in marketing copy. **The CSV is a logged-in download from the draft lobby**, which is how
  the existing one was obtained.
- **`draftkings.com/*nba-best-ball*` has ZERO Wayback snapshots.** There is no archived DK ADP,
  for any season, anywhere. This is the finding that makes the single CSV irreplaceable.
- **RotoWire** (`/basketball/adp.php`) — 31 archived snapshots, 2021-10-18 → 2026-04-20, 7
  preseason. **Hashtag Basketball** (`/fantasy-basketball-adp`) — 23 snapshots, 2021-10-06 →
  2026-02-20, 4 preseason, and it carries **Fantrax** which FantasyPros does not. Neither extends
  before 2021, so neither adds historical depth; both are useful **cross-checks** on 2021+ and a
  hedge against FantasyPros changing its page. Deprioritized, not dismissed.

---

## Urgency — honestly graded, because it differs by source

`CLAUDE.md` reserves the deadline language for feeds that cannot be backfilled. ADP splits into
two very different cases and conflating them would either cause a real loss or manufacture a false
alarm.

### 🟥 DraftKings — a genuine deadline, and the board is **already open**

Login-gated, zero Wayback presence, no API, and **live only while a game set is open**. The
2026-27 board was open and carrying ADP on **2026-07-28**, roughly 12 weeks before the season —
earlier than the 2025-26 evidence suggested (DK Network published its 2025-26 rankings on
2025-10-08/13; the anchor was downloaded 2025-10-17).

What is lost by missing it is not just one file. DK's ADP *moves* through the draft season, and
that trajectory is precisely what an opponent model wants. The July and October boards differ
materially in thickness and compression, so the sequence matters as much as any single capture.

**This is a calendar commitment, not a cron job**, because the download needs an authenticated
session. Concretely:

- ✅ 2026-07-28 board captured (`DkPreDraftRankings_July28_2026.csv`).
- **Capture roughly weekly from now through the 2026-27 opener**, to trace the preseason ADP
  curve that no other source can reconstruct.
- 🎯 **One capture is load-bearing: an early-to-mid October 2026 board**, timing-matched to the
  Oct-17-2025 anchor. Paired with the FantasyPros board once it flips, that is the **second
  anchor** the recalibration needs — and matching the calendar offset keeps the two pairs
  comparable instead of confounding season change with capture timing.

### 🟨 FantasyPros — real but slack, because of the freeze

The freeze rule means the 2026-27 board, once set in October 2026, stays on the page until
roughly September 2027. Missing a day costs nothing; missing a **year** costs a season.

The genuine risk is not retention, it is **archive sparsity**: 2024 got a single Wayback snapshot
all year, and 2023-24 only three. Relying on someone else's crawler is what made 2016-17 and
2024-25 recoverable only from drifted mid-season captures. Capturing directly removes that
dependency. `robots.txt` permits it (below), so this is cheap insurance — daily during the draft
window, weekly otherwise.

### Scraping etiquette — checked, not assumed

`fantasypros.com/robots.txt`, fetched 2026-07-28:

```
User-agent: *
Disallow: /ajax/     /api/     /json/     /xml/
Disallow: /nfl/ranker/   /mlb/ranker/   /nba/ranker/
Crawl-delay: 5
```

**`/nba/adp/overall.php` is permitted**; the `/api/` and `/json/` paths that would be more
convenient are not — so parse the rendered table, do not go looking for the JSON behind it.
Honor `Crawl-delay: 5`. The Wayback CDX endpoint needs a browser UA and rate-limits at roughly one
request per 10 s; back off on 503 and 429 rather than hammering.

---

## Point-in-time discipline

ADP is a market forecast of the same target the model predicts, which makes it the most
leakage-prone input in the project. The rules from `docs/availability-plan.md` apply unchanged,
plus two specific to this source.

- **`as_of_date` is the capture/archive date, never the season start.** Every row carries it, and
  the panel builder asserts `as_of_date <= season_start_date` for any training or backtest row —
  reusing `models.availability.assert_point_in_time` rather than writing a second checker.
- **`season` is derived from flip detection, not from the calendar.** See the two traps above.
  A mislabeled season here is a full year of lookahead — the model would be handed the market's
  opinion *after* it had seen the season it is being asked to predict.
- 🔴 **The `DTD` / `OUT` token in the FantasyPros player cell is current-status at capture time,
  not draft-day status.** On the 2025-01-19 snapshot it reports Luka Doncic `OUT` for a January
  calf injury, sitting in a row whose ADP was frozen the previous October. Using that token as
  preseason injury state would inject a mid-season realized outcome into a preseason feature —
  exactly the leak that plan forbids, in a source that otherwise looks safe. **Store it as
  `status_at_capture` and never join it to the availability head's preseason snapshot.**
- **A backtest drafts on the ADP as of its draft date.** For a 2018-19 backtest, that is the
  2018-10-03 snapshot, not the season's final frozen board — deep-player ADP drifts through the
  season (33.8% identical Jan→Jul), so the end-of-window board is contaminated by information the
  drafter did not have.
- Rows for seasons with no snapshot at or before the draft date carry `snapshot_source = none`
  and stay null, matching the availability convention.

---

## Storage

Raw captures are the artifact worth keeping — parsers get rewritten, and re-parsing is offline,
the same reasoning that governs the injury-report PDFs.

```
data/raw/adp/
  fantasypros/   fp_adp_<YYYY-MM-DD>.html.gz     + _fp_manifest.csv
  draftkings/    dk_predraft_<YYYY-MM-DD>.csv    + _dk_manifest.csv
data/features/
  adp_panel.parquet          # one row per (source, source_detail, player, season, as_of_date)
  adp_transfer.parquet       # the fitted consensus → DK recalibration, mirroring report_transfer
```

`adp_panel.parquet` columns:

| column | notes |
|---|---|
| `player_name`, `player_key` | raw, and the normalized join key |
| `player_id` | `nba_api` id where matched; null otherwise, with `match_method` recorded |
| `team`, `position` | as printed; `FA` is a legal team value |
| `source` | `fantasypros` \| `draftkings` \| `rotowire` \| `hashtag` |
| `source_detail` | `yahoo` \| `espn` \| `cbs` \| `fantrax` \| `avg` \| `dk` — **per snapshot**, never assumed |
| `adp`, `adp_rank` | the site's value verbatim; `AVG` is **not** recomputed |
| `adp_censored` | DK only — at/near the 192-pick boundary |
| `season` | flip-detected, not calendar-derived |
| `as_of_date`, `captured_at` | point-in-time keys |
| `snapshot_lag_days` | days from season start; large ⇒ drifted mid-season capture |
| `status_at_capture` | `DTD`/`OUT` — quarantined, see above |
| `snapshot_source` | `wayback` \| `live` \| `manual` \| `none` |
| `pool_size`, `n_sources` | for the mid-recompute consistency flag |

Manifests record per-capture row counts so a regression shows up as a column of anomalies, exactly
as the injury-report manifest records per-day inactive counts.

---

## Where ADP belongs: prediction layer or strategy layer?

**Recommendation: strategy layer, downstream of predictions — with one narrow exception.**

Four reasons, in decreasing order of force:

1. **Feeding ADP to the GLMM destroys the quantity the draft actually monetizes.** Under a
   knockout payout, edge *is* model-minus-market. A model trained with ADP as a feature partially
   reproduces the market by construction, and the disagreement it then reports is no longer a
   measurement of edge — it is shrinkage toward the thing it was fit on. The signal and the
   benchmark must stay separable.
2. **Coverage is fatally thin for a training feature.** ADP exists for ~250 players per season
   across 12 seasons; the component heads train on **10,900 player-seasons across 30**. That is
   under 10% of rows, concentrated in exactly the high-minutes players the model already predicts
   best. Adding it forces either aggressive imputation or a drastic sample cut, and
   `CLAUDE.md`'s singularity finding says the feature matrix does not need more collinear columns.
3. **The measured relationship says ADP carries little the model lacks.** Consensus and DK differ
   mostly by a monotone rescale, and both are ordinal summaries of the same public information the
   season matrix already holds. What ADP adds is *what the field believes*, which is not a
   property of the player.
4. **It keeps the ADP-blend `α` in `docs/simulations-plan.md` tunable.** If ADP is baked into the
   predictions, `α` no longer sweeps a real axis — the market's weight is already inside
   `model_z`, and the backtest cannot recover it.

So the three mechanics already listed in `docs/simulations-plan.md` — continuous z-score blend,
confidence-weighted blend, hard disagreement audit — stay where they are, operating on model
output. This plan adds only that they should blend against a **DK-recalibrated** consensus rather
than a raw one, since the recalibration is worth 7 picks and the C/F/G bias would otherwise be
read as model edge when it is a scoring-system artifact.

### The exception: ADP as a prior where the model is provably blind

The prediction-time constraint says **14.7% of roster minutes have no usable prior-season row** —
8.7% true rookies, 5.1% sub-threshold, 0.9% returnees. For those players the model is imputing
from `bio_draft_number`, while the market has seen summer league, training camp and coach
quotes. That is a real information advantage, and it is the one place ADP is not redundant.

Admit it the Bayesian way rather than as a feature: use ADP to set the **prior mean on the rate
head** for players whose `reliability` is below threshold, with prior weight tied to the existing
`0.924 · m/(m+66)` shrinkage machinery — high ADP weight where `m` is small, decaying to zero as
prior minutes accumulate. This keeps every established player's prediction market-free (preserving
point 1), touches only the rows where the model has nothing, and reuses the reliability curve
rather than inventing a second one.

**Measure it before shipping it**: hold out the last two seasons, compare rate-head CRPS on
thin-data players with and without the ADP prior. If it does not beat the `bio_draft_number`
imputation, record it as a settled null and keep ADP purely in the strategy layer. Given how
often this repo's ceilings have come in low, that is the likely outcome and the cheap test is
worth running first.

---

## Staging

Conventions per `CLAUDE.md`: `python -m src.<module>` entry points, matching `Makefile` targets in
`.PHONY`, `cfg = yaml.safe_load(open("configs/default.yaml"))` in `__main__`,
`Path(...).mkdir(parents=True, exist_ok=True)` before writes, `f"... {n:,} ... → {dest}"` progress
lines, plain-`assert` tests with synthetic builders.

| Stage | Module | Make target | Notes |
|---|---|---|---|
| **A0** | — (manual) | — | ✅ **Standing routine, owned by the user**: pull the DK pre-draft rankings CSV from the lobby through the 2026-27 preseason. Drop files in `data/raw/dk_draft_rankings/` as `DkPreDraftRankings_<Mon><D>_<YYYY>.csv`. Two boards captured so far. |
| **A** | `src/data/adp_draftkings.py` | `adp-draftkings` | ✅ **done** — ingest, team aliases, censoring flag, `id_stability` check, and the one-time DK `ID` → `player_id` map (**0 unmatched of 811 matchable**). |
| **A** | `src/data/adp_fantasypros.py` | `adp-fantasypros` | ✅ **done** — Wayback backfill + live capture + offline `--reparse`, gzip sniffing, per-snapshot source detection, the freeze rule, and `--status`. |
| **B** | `src/features/adp.py` | `adp-panel` | ✅ **done** — `adp_panel.parquet`, the match cascade (**0.50% unmatched**), `attach_dating` / `training_rows` / `assert_point_in_time`. |
| **C** | `src/eda/adp_profile.py` | `adp-profile` | ✅ **done** — `adp_transfer.parquet` + `outputs/eda/adp_profile.csv`, carrying `n_anchors` and a loud warning while it is 1. |
| **D** | — | — | Wire the recalibrated ADP into the draft simulator's opponent model (`docs/simulations-plan.md`). Blocked on the simulator, not on data. |
| **E** | — | — | The thin-data ADP-prior ablation (above). Blocked on the Bayesian rate head. |

`make adp` runs A→C in order; `make adp-status` reports coverage for both sources without
making a request. 40 tests in `tests/test_adp.py`.

**Since 2026-08-10 that coverage is also an artifact.** `make capture-calendar`
(`src/data/capture_calendar.py`) re-reads the same manifests both ADP sources' `--status`
paths read and writes `outputs/eda/capture_{calendar,programs}.csv`, so the dashboard's
"Inputs beyond the heads" page can draw the two boards beside the two daily injury feeds. It
records the distinction that matters here: FantasyPros is `recovery = archive` because
Wayback holds its history, and DraftKings is `recovery = never`. See
[dashboard-plan.md](dashboard-plan.md#step-6-as-built--inputs-beyond-the-heads).

> ⏳ **The Wayback backfill has not been run yet, and that is the one outstanding task.**
> 23 snapshots are archived (8 seasons) against the 259 the CDX index lists. Live capture
> and parsing are verified end-to-end — a live pull on 2026-07-28 returned 310 KB / 260
> rows and was correctly labelled **2025-26**, the frozen board — but the archive host
> began returning **HTTP 498** (its rate-limit code) after the planning sweep, so
> `--backfill` needs to run once the throttle clears. It is resumable and skips anything
> already on disk, so re-running it is free. Nothing else depends on it: `--reparse` works
> offline, and the transfer function only needs seasons that already have a DK anchor.

### What implementing it changed — measured, not planned

Reproduce with `make adp`. Four things the plan got wrong or under-specified, each found by
running the code against the real archive:

- 🔴 **A name-matching cascade can score a perfect unmatched rate by fabricating rows.**
  The obvious rule — same surname, same first initial — produced **11 false matches out of
  12** on the two real DK boards: `Cameron Boozer` → **Carlos Boozer**, `Darryn Peterson` →
  **Drew Peterson**, `RJ Davis` → **Ricky Davis**, `Javante McCoy` → **Jelani McCoy**. It
  also reported **0.0% unmatched**, because every fabricated match counts as a success.
  **An unmatched rate is therefore not a sufficient check on a join** — the surviving
  non-exact matches have to be listed and read. Two guards replaced it: a first name must
  be a genuine *prefix* of the other (≥3 chars), and the candidate must have played within
  `RECENT_SEASONS` of the board. Both are load-bearing; prefix alone still matched
  `Mikel Brown Jr.` → **Mike Brown** (last seen 1996-97).
- **Rookies and join bugs must be counted separately.** 162 DK pool entries have no
  `player_id` because they have never played an NBA game — the entire 2026 draft class among
  them. Reported as `no_nba_history`, not `unmatched`, for the same reason
  `report_calibration.py` keeps `absent` apart from `unmatched`. The residual genuine
  unmatched is **0.50%**: four players (Wang Zhelin, Rade Zagorac, Marcus Zegarowski,
  Patric Young) who appear on fantasy boards and never played in the NBA.
- **The injury-status token is an open vocabulary and cannot be enumerated.** The observed
  set across 12 years is `DTD`, `OUT`, `FA`, `O`, `G-League`, `NWT`, `RET`, `TWO-WAY`,
  `ACT`, `IR`, `D-LEAGUE`, `SUS` — hyphenated, mixed-case, and single-letter. An
  enumeration silently folded the token into the player's name instead
  (`Kevin Durant OKC O`), which then failed to match and looked like a *matching* problem
  rather than a *parsing* one. Strip whatever follows the team; do not whitelist.
- **Season flips are not detectable from the identical-AVG share**, which the plan assumed
  they would be. Real flips score 0.0055–0.0871 and ordinary within-draft-window drift
  scores 0.0667–0.2273 — overlapping ranges, no threshold. Identity is now used only for
  the question it answers cleanly ("did the board move at all": frozen pairs score exactly
  1.0000 against a non-frozen maximum of 0.3360), and the season comes from the calendar
  with frozen snapshots inheriting. That combination reproduces every observed season
  label, including the COVID case, which `tests/test_adp.py` pins.

The profile reproduces the planning-session numbers on a larger matched set (226 pairs
against 218, the difference being names the improved cascade recovered): Spearman **0.8675**
against 0.8704, mean absolute rank gap **23.1** picks, centers **+13.88**, and the ladder
**24.38 → 17.32** for the monotone step with the tier pattern unchanged (5.09 picks in
rounds 1–2 against **32.31** in rounds 9+).

Add `adp-fantasypros` to `daily-capture` during the draft window. **Do not** add
`adp-draftkings` — it cannot run unattended.

---

## Risks and open questions

- **The recalibration rests on one anchor and cannot be validated, only fit.** This is the
  irreducible limitation. It is mitigated by keeping the fit to ~one degree of freedom, and it
  resolves in **October 2026**, when FantasyPros flips to 2026-27 and completes the pair whose DK
  side is already captured. Until then, report backtests both with and without it.
- **The second anchor is contingent on the consensus flip, not on the DK capture.** The DK side is
  in hand; the pair completes only when FantasyPros updates. If it flips late, or the page changes,
  the anchor count stays at 1 for another year — which is the argument for capturing RotoWire and
  Hashtag Basketball as live cross-checks from now rather than treating them as optional.
- **The 2025-26 anchor may not transfer to 2026-27.** The C/F/G bias is mechanistic — points
  versus category scoring — so it should be stable. The *scale* compression depends on DK's pool
  size and draft volume, which can change. Re-fit annually; do not assume the map is a constant.
- **DK's pool is 698 players; consensus sources carry ~250.** Recalibration says nothing about the
  ~450 players DK lists with no ADP at all. For a 192-pick draft this is mostly irrelevant, but a
  simulator that runs the board dry needs a fallback ordering for them.
- **Name matching is unsolved at the required standard — but the problem is now half its
  original size.** DK's `ID` is persistent (verified on 667 players), so the DK side needs a
  one-time id map rather than fuzzy matching forever. The consensus sources still need it: the
  measured join is 87.6% against the 0.0%-unmatched standard `report_calibration.py` set.
  Diacritics (`Kristaps Porzingis`), short forms (`Alexandre`/`Alex Sarr`), suffixes and `FA` team
  codes are the known failure classes.
- **FantasyPros could change or remove the page**, which is the argument for capturing raw HTML
  now and for keeping RotoWire/Hashtag as live cross-checks rather than dismissing them.
- **The whole ADP thread may matter less than the tournament-selection findings already in
  `docs/simulations-plan.md`** — rake spans 9.5–15.0% and advance rates 1-of-12 to 2-of-6 across
  the five real tournaments, both known exactly and both plausibly larger levers than a 7-pick
  ADP recalibration. Sequence the work accordingly.
