# Data quirks

- `MIN` in season files is **per-game**, not a total (except where `per_mode` says
  otherwise). Total minutes = `MIN * GP`.
- **Per-36 rule:** `MIN` always carries the same basis as the stats beside it, so
  `stat / MIN * 36` is correct for both `PerGame` *and* `Totals`. `PerMinute` is `stat * 36`.
  `Per100Possessions` has no minutes basis and raises.
- Season CSVs are already **one row per player**, deduped across trades; a traded player is
  attributed wholly to his **last** team.
- ~~**`preprocess.load_raw` turns playoff logs into pseudo-seasons**~~ — ✅ **fixed
  2026-07-29.** It globbed `game_logs_*.csv`, which also matches the 30
  `game_logs_playoffs_*.csv` files added by the availability work, and derived `season`
  from the filename slug — so a playoff file became `playoffs-1996-97`, doubling the
  season count for every `groupby(["player_id", "season"])`. `load_raw` now takes
  `season_type` (`"regular"` default / `"playoffs"` / `"all"`), parses the prefix off the
  slug so both types share one `season` label, and tags rows with `season_type`, which
  `clean` carries through. An unknown value raises rather than returning an empty frame.
  Rebuilding gives back exactly the pre-existing 731,906 rows, so the fix restored intent
  rather than changing data.
- **`component_targets.parquet` is a filtered frame, not the raw log.** `preprocess.clean`
  drops players below `data.min_games` (786,765 raw rows → 731,906). Fine for per-player
  analysis; wrong for anything that aggregates a whole team-game, which is why
  `game_length.py` reads raw.
- **The nba_api dump lives in `data/raw/nbastats/`, not in `data/raw/` itself** (moved
  2026-08-17). The parent holds the things `make fetch` does *not* write and cannot
  re-create: the four capture archives that are not backfillable (`injury_reports/`,
  `injuries/`, `adp/`, `dk_draft_rankings/`), the hand-placed tournament CSVs, and the two
  manifests. ~790 season files were drowning that out.
  - **`cfg["data"]["raw_dir"]` still means `data/raw` — the parent — everywhere.** Nothing
    in the config or in any caller changed. What changed is that every path to a fetched
    season CSV now goes through `fetch.nbastats_dir(raw_dir)`, which is the **only** place
    the subdirectory is named. Functions join it themselves rather than taking a second
    argument, so the three modules that read both sides (`boxscore_status`,
    `season_matrix`, `report_calibration`) still take one `raw_dir`.
  - **The two manifests are deliberately outside it.** `_fetch_manifest.csv` and
    `_boxscore_status_manifest.csv` are metadata *about* the fetch, they sit beside the
    other archives, and `fetch._record_fetch` / `boxscore_status._append_manifest` write
    them to the parent. `season_matrix.load_manifest` reads from the parent to match.
  - A new reader that forgets the join fails loudly with `FileNotFoundError` naming the
    directory it looked in, not silently with an empty frame — every raw reader in the
    repo raises on no-files-found.
- `player_shot_locations` has a **two-row MultiIndex header**; every other family is flat.
- Empty `LeagueDashPlayerStats` responses are **poisoned server-side cache entries** keyed
  on the full query, not rate limiting. Backoff does not help; cycling `per_mode_detailed`
  does. `fetch.py` handles this and records the winning mode in `data/raw/_fetch_manifest.csv`.
- Player `"Four Factors"` is genuinely unsupported by the endpoint — its stats live in
  Advanced. Team four-factors works fine.
- Coverage: core families 30 seasons; tracking/`pt_shot` 13 (2013-14+); estimated 12
  (2014-15+); hustle 11 (2015-16+, and **2015-16 covers only 147 players** — a partial
  first-season rollout, not a fetch failure); **box-score inactive lists 20 (2006-07+)**.
- **The inactive list is a third coverage boundary: 2006-07.** Probed on both sides —
  0 rows for 2005-06, 6 on 2006-07 opening night. Earlier seasons still separate played
  from dressed-but-scratched (the traditional `COMMENT` field goes back further) but
  cannot see who was inactive.
- **Two box-score endpoints have version traps, and they fail in opposite ways.**
  `BoxScoreTraditionalV2` returns **0 rows** from 2025-26 on — loud, obvious, route to V3
  (the field is lower-case `comment`). `BoxScoreSummaryV2` is the dangerous one: after
  **2025-04-10** it still returns 200 with a populated `GameSummary` and silently drops
  the `InactivePlayers` result set, so a backfill built on it records "nobody was
  inactive" rather than failing. Measured: 223 of 228 games in 2025-26 came back empty.
  **Use `BoxScoreSummaryV3`** (`boxScoreSummary.{home,away}Team.inactives`, team id on the
  parent block) — it covers 2006-07 → 2025-26 and agrees with V2 exactly where V2 works.
- **A third V3 failure mode: `nba_api`'s own parser raises on a stub payload, and the guard
  in `_inactive_rows` cannot catch it.** Three games (`0022500259`–`0022500261`, all
  2025-11-19) return a `boxScoreSummary` whose `arena`, `teamId` and `inactives` are all
  `null`. `nba_api/stats/endpoints/_parsers/boxscoresummaryv3.py::get_arena_info_data`
  calls `arena.get("arenaId")` unguarded, so it throws `AttributeError: 'NoneType' object
  has no attribute 'get'` **inside the constructor** — before `get_dict()` is ever reached.
  The `payload.get("boxScoreSummary") or {}` guard therefore never runs, and the error text
  reads like a bug in this repo when it is upstream. Confirmed there is nothing to recover:
  the raw JSON (via `NBAStatsHTTP().send_api_request`, which bypasses the parser) is a stub,
  and V2 returns 0 `InactivePlayers` for the same games because 2025-11-19 is inside its
  silent-failure window. Only the *traditional* side survives (28 dressed, 19 played).
  **Left unfetched deliberately.** Recording the dressed rows without the inactive list
  would mislabel those games' inactives as `not_rostered` — the exact collapse the panel's
  per-game coverage tracking exists to prevent. With no status rows the game is
  `status_covered = 0` and every row is `unknown`, which is correct. 3 of 25,709 games.
- **The backfill manifest is append-only, so counting `error` rows overstates the damage.**
  A retried game keeps its old failure row and gains a new success row. After the second
  pass the manifest holds 77 error rows against **3** games that actually lack data — dedupe
  to games with no successful attempt (`set(all) - set(error.isna())`) before reporting.
  Every transient network fault (connection resets, read timeouts — 68 of them, clustered in
  2012-13 and 2013-14) cleared on one retry.
- **NBA injury-report PDFs**: `CreationDate` matches the filename key exactly, a 403 means
  the key does not exist (aged out, or no games — the All-Star break is absent from the
  middle of a live window), and neither is retryable. The PDF is landscape with a flipped
  text matrix, so **y increases downward** and rows sort ascending. Cells are left-aligned
  and wrap: assign a chunk to the *last* column starting at or before it, never by
  midpoints. A wrapped reason is typeset **centred** on its row (first line above the
  player name, continuation below) and can **wrap across a page break**, landing at the
  top of the next page as a tail of the previous page's last row. A team that missed the
  filing deadline gets a row with `NOT YET SUBMITTED` in the *reason* column and no
  player — it needs its own anchor or it corrupts the nearest player's row.
- The 7 `team_stats_*` families and `team_estimated_metrics` feed
  `src/features/opponent.py`. `team_stats_opponent`'s `OPP_*` columns are *what opponents
  recorded against this team* — already the per-component opponent profile. Never rebuild
  this by aggregating player rows.
- **Team files break the per-36 rule.** `fetch_team_stats` uses `PerMode="PerGame"` and the
  counting stats are per game, but `MIN` and `POSS` are **season totals** (`MIN` ≈ 3966 =
  48.4 × 82). Normalize per-100 possessions with `POSS / GP`, never `stat / MIN * 36`.
- **⚠️ `sklearn`'s two regularization conventions are opposite, and with exposure weights
  the difference is ~7 orders of magnitude.** `PoissonRegressor` (and every
  `_GeneralizedLinearRegressor`) minimizes `deviance / (2·Σw) + alpha·‖coef‖²` — the data
  term is **averaged by the weight sum**. Fitting a rate with `sample_weight = minutes`
  makes `Σw ≈ 10⁷`, so the default-looking `alpha=1.0` is an enormous penalty and shrinks
  every coefficient to nearly zero. Measured on the `reb` season-total head: R² **0.662 at
  alpha=1.0 against 0.928 at alpha ≤ 0.01**, and it fails *quietly* — the fit converges,
  coefficients are finite, and a flexible basis partially compensates, so a spline or
  interaction looks like it is buying real signal. `LogisticRegression` is the reverse: its
  objective is `Σ w·logloss + ‖coef‖²/(2C)`, **not** averaged, so with the same weights
  `C=1.0` is *weak*. **Always sanity-check a weighted GLM against a no-fit baseline** — for
  the component heads, `prior rate × actual minutes`, which scores R² 0.81–0.95 and would
  have exposed this immediately.
  - ✅ **The trap is now a curve and a permanent regression guard**: `make component-rates`
    (`alpha_sensitivity`, `analysis == "alpha_sensitivity"` — 98 rows over 7 count heads × 2
    variants × a 1e-8…10 grid, with `floor_r2` on every row). `reb` on the raw-rate spec reads
    **0.9181 at alpha=1e-8**, **0.9267 at 0.01** and **0.6619 at alpha=1.0** — and the middle
    pair reproduces the recorded 0.928 / 0.662 to a thousandth **across the move from the
    held-out split to validation**, which is the strongest available statement that this is a
    property of the penalty and not of any particular rows. At alpha=10, **14 of 14 fits fall
    below the no-fit floor**, the visual statement of why the floor is mandatory. (Recorded
    held-out: 0.9278 / **0.9322** / **0.6620**, and 16 of 16 — the 16 dating from the retired
    eight-head basis.)
  - **It hurts the shipped spec too, not just the misspecified one.** Median R² loss from
    `POISSON_ALPHA` to alpha=1.0 is **0.200** on `linear` and **0.305** on `log_own`, worst
    `blk` at **0.497**. Every head's best alpha on the grid is ≤ 0.01. (Recorded: 0.228 /
    0.289, worst `fg3a` at 0.595 — a retired head.)
  - **The guard compares each head against its own optimum, not against the floor.** `blk`
    loses to the floor at *every* alpha under `log_own` because it needs a spline — a
    documented modelling finding, not the penalty misbehaving — so a floor crossing alone
    cannot be the trigger.
- **Preseason logs are a fourth season type, and every one of their quirks is a way to
  contaminate a frame silently.** `make preseason` (`src/features/preseason.py` →
  `data/features/preseason.parquet`, `outputs/eda/preseason_coverage.csv`) — see
  `docs/preseason-plan.md`.
  - **A new file prefix in `data/raw/nbastats/` is a pseudo-season waiting to happen.** This is the
    playoffs bug of 2026-07-29 in a second costume, and worse: `_parse_log_filename` does
    not merely invent the label `pre-season-2023-24`, it also returns `REGULAR_SEASON`, so
    unrecognized rows arrive in the **default** frame rather than an opt-in one. Prefixes
    belong in `preprocess._LOG_PREFIXES`, never in a caller's glob.
  - **`load_raw(season_type="all")` means regular + playoffs, and excludes the preseason
    deliberately.** A preseason game is a forecast covariate, never a target row.
    `src/features/game_length.py` is the reason the default runs that way: it reads `"all"`
    to derive every game's length from summed team minutes, and would have absorbed ~70
    exhibition games a season into an artifact whose whole claim is that its two
    independent estimates disagree on **0** of 37,986 games.
  - **The API's season-type label carries a space.** `_slug("pre season")` leaves it, so
    the file lands as `game_logs_pre season_2023_24.csv` — unclassifiable by prefix.
    `fetch._season_type_slug` folds spaces as well as dashes.
  - **The 2019-20 preseason file straddles its own opener.** The July 2020 bubble
    scrimmages carry the `Pre Season` label under the 2019-20 season key: **822** rows over
    33 games played 2020-07-22 → 2020-07-28, against a **2019-10-22** opener. Reading them
    as a 2019-20 feature is not a subtle leak, it is the season itself. Every row is
    filtered against the season's first regular-season game date and the drop is counted
    per season, because an assertion here would simply fail.
  - **The same instrument is load-bearing for ADP.** `features/adp.py::season_start_dates`
    used to exclude playoff files by a substring test on the filename. With preseason files
    on disk that test passes them through, and since they carry the same `SEASON_YEAR` it
    would move **23 of 30** season openers ~3 weeks earlier — silently re-deciding which
    ADP captures are point-in-time legal. It now classifies by `_parse_log_filename`.
  - **Exhibition opponents are not NBA teams.** Real Madrid (12315), Flamengo (12325),
    Maccabi Ra'anana (50009), the New Zealand Breakers (15020) and the Cairns Taipans
    (15022) appear with ids outside the `1610612737–1610612766` block, and so do their
    players — up to **10** such teams in a season, **1,601** rows across the backfill. The
    NBA team's own rows in those games are real preseason games and are kept.
  - **Some rows have no `player_id`.** 2003-04 carries **96** of them — team-total rows with
    `MIN` of exactly 480 (= 2 × 5 × 48) or 530 (one overtime) — plus one stray each in
    2010-11 and 2017-18, **98** in all. Every season before 2003-04 is a *single* such row
    and nothing else, which is why `fetch._FIRST_YEAR["pre_season"] = 2003` stops the
    orchestrator writing those files: a header plus one junk row reads as populated to
    `_has_data_rows` and would never re-fetch.
  - **The preseason window must be derived, never assumed to be October.** 2020-21 ran
    **Dec 11–19** (COVID) and 2011-12 ran **Dec 16–22** (the lockout, exactly 2 games per
    team). Both are real, complete preseasons at an unusual date — the same calendar trap
    the ADP freeze rule hit.
  - **Coverage runs 2003-04 → 2025-26 and only 2003-04 is unusable.** A preseason ends
    **3–5** days before the opener in 22 of the 23 seasons; 2003-04 ends **20** days out
    with **15** games across **24** teams, which is the signature of a truncated capture
    rather than a short preseason. That is what `coverage_class == "tail_missing"` marks,
    and the tail is exactly what the participation features are read over. 2004-05 is the
    milder version — the final week only, Oct 22–29, **3** games per team — and it is
    classified `covered` because its tail is present.
- **The season matrix serves the PCA, not roster aggregation.** Its `GP≥20 & MIN≥10` filter
  keeps garbage-time per-36 outliers out of the PCA, but it also drops real teammates who
  consume real minutes. Build roster aggregates on `season_matrix_roster_tier*.parquet`
  (the unfiltered twin) with reliability shrinkage, never on the qualified matrix.
- **`scipy.stats.betabinom.pmf` loses ~5e-9 of relative accuracy when `rho` reaches its
  guard rail, and it is the *reference* that goes wrong, not the caller.** At `rho = 1e-6`
  the shape parameters reach ~1e6, where the function evaluates two log-betas of magnitude
  1e6 that cancel to a number of order one; measured against 50-digit arithmetic it reads
  **+5.5e-9** relative at `k = 70, n = 82` where a successive-ratio recursion reads
  **+8.5e-15**. This surfaced as `assert_nests` failing at 1e-8 for
  `availability_absence.CompoundCountingFrailty` on a synthetic frame that drew *binomial*
  counts — leaving `rho` nothing to fit, so every arm sat on `RHO_MIN`. The fix was the
  fixture (draw overdispersed counts, which is also what the head actually fits), not the
  tolerance. Two consequences worth carrying: a nesting assertion written as an equality
  against `betabinom.logpmf` is only as exact as that function is, and any beta-binomial
  pmf grid evaluated in an inner loop is both faster and more accurate built from
  `P(k+1)/P(k) = ((n−k)/(k+1))·((a+k)/(b+n−k−1))` with a `logsumexp` normalization than
  from `scipy` — ~10× on a (4,027 × 83) grid.
