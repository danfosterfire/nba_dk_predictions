"""Tab 2 · Data collection.

What was gathered, what it cost, what cannot be gathered, and what the joins can and
cannot be trusted to do.

Three of this project's capture programs have **deadlines** — sources that cannot be
backfilled — so the archive counts here are live filesystem reads rather than
recorded numbers. A stale count on this panel would be the one place a number going
quietly wrong has an irreversible cost.
"""

import pandas as pd
import streamlit as st

from dashboard import decisions as D
from dashboard.artifacts import ROOT, Ctx, inventory, optional, read_table
from dashboard.charts import fig_bars, fig_heatmap
from dashboard.layout import (decision_cards, detail, note, provenance, stat_tiles,
                              tab_header, table_view)

TOPIC = "data"

# Capture programs whose sources cannot be backfilled. Counted live off disk.
CAPTURES = [
    ("NBA injury-report PDFs", "data/raw/injury_reports", "*.pdf",
     "~7-month rolling CDN retention, then a 403 forever. On launchd, daily.",
     "make injury-reports"),
    ("ESPN injury feed", "data/raw/espn_fantasy", "*",
     "No history at all — a snapshot not taken is gone. 28 days were lost "
     "permanently to a macOS Full Disk Access lapse.", "make injuries"),
    ("DraftKings ADP boards", "data/raw/dk_draft_rankings", "*.csv",
     "Login-gated, no API, zero Wayback snapshots, live only while contests are "
     "open (~Oct). **Manual capture.**", "manual download"),
    ("FantasyPros ADP snapshots", "data/raw/adp/fantasypros", "*.gz",
     "Wayback-backfillable, unlike the others — 23 of 259 archived so far.",
     "make adp-fantasypros -- --backfill"),
]


def render(ctx: Ctx) -> None:
    tab_header(
        "Data collection",
        "Thirty seasons of game logs, ~30 stat families, a 20-season box-score "
        "backfill and four capture programs. What matters here is less the volume "
        "than which parts cannot be re-obtained and which joins can be trusted.")

    _scale(ctx)
    _captures(ctx)
    _backfill(ctx)
    _coverage(ctx)
    _roster_coverage(ctx)
    _name_joins(ctx)
    _inventory(ctx)

    st.markdown("---")
    st.markdown("### Decisions about the data")
    decision_cards(D.by_topic(TOPIC))


# ── Scale ─────────────────────────────────────────────────────────────────────

def _scale(ctx: Ctx) -> None:
    raw = ROOT / "data" / "raw"
    size_mb = sum(p.stat().st_size for p in raw.rglob("*") if p.is_file()) / 1e6 \
        if raw.exists() else 0.0
    logs = len(list(raw.glob("game_logs_*.csv"))) if raw.exists() else 0

    targets = ctx.features("component_targets.parquet")
    player_games = None
    if targets.exists():
        import pyarrow.parquet as pq
        player_games = pq.ParquetFile(targets).metadata.num_rows

    stat_tiles([
        ("Raw data on disk", f"{size_mb:,.0f} MB",
         "Everything under `data/raw/`, including the box-score backfill and the "
         "PDF archive."),
        ("Game-log files", f"{logs}",
         "One per season per type. Regular season and playoffs are separate files "
         "sharing one `season` label."),
        ("Player-games in the fitting frame",
         f"{player_games:,}" if player_games else "—",
         "Row count read from the parquet footer, not by loading the 37 MB file."),
        ("Capture programs with deadlines", "3",
         "Injury-report PDFs, the ESPN feed and the DK boards. See below."),
    ])
    note("`component_targets.parquet` is a **filtered** frame — `preprocess.clean` "
         "drops players below `data.min_games`. Fine for per-player analysis, wrong "
         "for anything that aggregates a whole team-game, which is why "
         "`game_length.py` reads the raw logs instead.")


# ── Capture programs ──────────────────────────────────────────────────────────

def _captures(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Capture programs — the parts that cannot be backfilled")

    rows = []
    for name, rel, pattern, why, target in CAPTURES:
        d = ROOT / rel
        count = len([p for p in d.glob(pattern) if p.is_file()]) if d.exists() else 0
        rows.append({"program": name, "files on disk": count, "path": rel,
                     "why it has a deadline": why, "target": target})
    frame = pd.DataFrame(rows)
    st.dataframe(frame[["program", "files on disk", "why it has a deadline",
                        "target"]], width="stretch", hide_index=True)

    st.error(
        "**`daily-capture` is the only thing in this repo with a daily deadline.** "
        "Both its sources are current-status feeds: the NBA report PDFs age out of "
        "the CDN after ~7 months and the ESPN feed has no history at all. A day the "
        "job does not run is a day permanently lost — which is not hypothetical, "
        "since a Full Disk Access lapse already cost 28 days of ESPN snapshots. It "
        "runs under **launchd rather than cron** because macOS cron silently skips a "
        "run whose time passed while the machine was asleep, and launchd catches "
        "missed runs up on wake.")
    note("`make injury-reports` is idempotent and re-parsing is offline "
         "(`--reparse`), because **the archived PDFs — not the parsed CSVs — are the "
         "artifact worth keeping**. A parser bug costs nothing as long as the source "
         "is on disk.")
    st.warning(
        "**The DraftKings board is the second deadline, and it is a manual one.** It "
        "is login-gated, has zero Wayback snapshots, exposes no API and is live only "
        "while contests are open. A board not downloaded from the draft lobby while "
        "it is open is gone permanently. The load-bearing one still to get is an "
        "**early-to-mid October 2026** board, timing-matched to the October 2025 "
        "anchor — it is what would take the ADP recalibration from one anchor to two.")
    provenance("live filesystem counts under `data/raw/`; see the Decision log tab's "
               "deadline board")


# ── The box-score backfill ────────────────────────────────────────────────────

def _backfill(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The box-score status backfill")

    path = ROOT / "data" / "raw" / "_boxscore_status_manifest.csv"
    if not path.exists():
        st.warning("`data/raw/_boxscore_status_manifest.csv` not found — run "
                   "`make boxscore-status`.")
        return
    man = read_table(str(path))

    attempted = set(man["game_id"])
    succeeded = set(man[man["error"].isna()]["game_id"])
    failed = attempted - succeeded
    error_rows = int(man["error"].notna().sum())

    stat_tiles([
        ("Games with data", f"{len(succeeded):,}",
         f"Of {len(attempted):,} attempted, 2006-07 → 2025-26."),
        ("Games that genuinely lack data", f"{len(failed)}",
         "Left unfetched deliberately — see below."),
        ("Error rows in the manifest", f"{error_rows}",
         "Which is **not** the failure count."),
        ("Seasons covered", f"{man['season'].nunique()}",
         "The inactive list starts at 2006-07, probed on both sides rather than "
         "assumed."),
    ])
    st.info(
        f"**The manifest is append-only, so counting `error` rows overstates the "
        f"damage — {error_rows} error rows against {len(failed)} games that actually "
        f"lack data.** A retried game keeps its old failure row and gains a new "
        f"success row, so the count has to dedupe to games with no successful "
        f"attempt. Every transient network fault cleared on one retry.")
    if failed:
        st.warning(
            f"The {len(failed)} remaining games "
            f"({', '.join(sorted(str(g) for g in failed))}) return a stub payload "
            f"that `nba_api`'s own V3 parser raises on, **inside the constructor**, "
            f"so the guard in `_inactive_rows` cannot catch it. There is nothing to "
            f"recover: the raw JSON is a stub and V2 is inside its own silent-failure "
            f"window. Recording the dressed rows without the inactive list would "
            f"mislabel those games' inactives as `not_rostered`, so with no status "
            f"rows the game is `status_covered = 0` and every row is `unknown` — "
            f"which is correct.")

    by_season = (man[man["error"].isna()].groupby("season")
                 .agg(games=("game_id", "nunique"),
                      inactive=("inactive_rows", "sum")).reset_index())
    st.plotly_chart(
        fig_bars(by_season, "season", ["games"], ctx.th,
                 "Games with box-score status, by season", axis_title="games",
                 height=460),
        width="stretch")
    table_view(by_season, "Backfill by season — table view")
    provenance("`make boxscore-status` → `data/raw/boxscore_status_*.csv`, "
               "`data/raw/_boxscore_status_manifest.csv`")


# ── Coverage ──────────────────────────────────────────────────────────────────

def _coverage(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Coverage — four boundaries, not one")

    cov = optional(ctx.features("coverage_report.csv"), target="make season-matrix")
    if cov is None:
        return

    scoped = cov[cov["tier"].isin(["A", "B"] if ctx.tier == "B" else ["A"])]
    grid = scoped.pivot_table(index="family", columns="season", values="matched",
                              aggfunc="max", fill_value=0)
    st.plotly_chart(
        fig_heatmap(grid, ctx.th, f"Players matched per family × season "
                                  f"(tier {ctx.tier})", "players", height=560,
                    hover="%{y}<br>%{x} · %{z:.0f} players"),
        width="stretch")
    note("**Core box-score families run all 30 seasons; tracking and `pt_shot` start "
         "in 2013-14, estimated in 2014-15, hustle in 2015-16 — and the box-score "
         "inactive list at 2006-07.** 2015-16 hustle covers only 147 players, which "
         "is a partial first-season rollout rather than a fetch failure. The blank "
         "cells are the boundaries, and they are why Tier A and Tier B exist as "
         "separate frames.")
    table_view(scoped.sort_values(["season", "family"]), "Coverage — table view")
    provenance("`make season-matrix` → `data/features/coverage_report.csv`")


# ── Roster description coverage ───────────────────────────────────────────────

def _roster_coverage(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### How much of a roster the prior season actually describes")

    path = ctx.eda(f"roster_coverage_profile_tier{ctx.tier}.csv")
    prof = optional(path, target="make context-value")
    if prof is None:
        return

    league = prof[(prof["scope"] == "league") & (prof["window"] == "season_start")]
    if league.empty:
        return

    def share(key: str, metric: str) -> float:
        hit = league[(league["key"] == key) & (league["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    stat_tiles([
        ("Roster minutes with no usable S-1 row",
         f"{share('undescribed', 'share_of_minutes'):.1%}",
         "On the season-start roster — the only window knowable before the season, "
         "and therefore the only one `team_context` aggregates."),
        ("… true rookies", f"{share('rookie', 'share_of_minutes'):.1%}",
         "No prior NBA season at all."),
        ("… sub-threshold", f"{share('sub_threshold', 'share_of_minutes'):.1%}",
         "Played, but below the qualified matrix's GP/MIN filter."),
        ("… returnees", f"{share('returnee', 'share_of_minutes'):.1%}",
         "Absent the previous season entirely."),
    ])

    categories = ["prior", "sub_threshold", "returnee", "rookie"]
    plot = pd.DataFrame({
        "source": categories,
        "share of minutes": [share(c, "share_of_minutes") for c in categories],
        "share of headcount": [share(c, "share_of_headcount") for c in categories],
    })
    st.plotly_chart(
        fig_bars(plot, "source", ["share of minutes", "share of headcount"], ctx.th,
                 "Roster description source — weighted two ways",
                 axis_title="share", height=320),
        width="stretch")
    st.info(
        "**Both weightings ship because neither substitutes for the other.** A rookie "
        "is a much larger share of roster *rows* than of roster *minutes*. The "
        "minutes-weighted split is the honest denominator for an aggregate — but "
        "weighting by the *prior-season* minutes the aggregate uses would show a "
        "rookie contributing zero and every roster looking fully covered, which is "
        "the circularity the module's own docstring warns against.")
    st.warning(
        "**Never drop them.** The undescribed share reaches ~29% at p90 and ~50% for "
        "the worst team-season; dropping those players silently biases every "
        "aggregate toward veterans. The unfiltered `season_matrix_roster_tier*` twin "
        "exists for exactly this, carrying `stats_source` and `reliability` per row.")
    note("The long-recorded 15.9% figure was the **whole-season** roster; the "
         "season-start window reads lower because rookies and returnees *arrive "
         "late*, so a wider window pulls in disproportionately many of them. Both "
         "windows are emitted so the difference stays visible.")

    with detail("Roster coverage — league, distribution and worst team-seasons"):
        st.dataframe(prof.round(6), width="stretch", hide_index=True)
    provenance(f"`make context-value` → "
               f"`outputs/eda/roster_coverage_profile_tier{ctx.tier}.csv`")


# ── The name-join discipline card ─────────────────────────────────────────────

def _name_joins(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The most dangerous measurement in this repo")

    audit = optional(ctx.eda("adp_match_audit.csv"), target="make adp-panel")
    if audit is None:
        return

    metrics = audit[audit["metric"].notna()]

    def metric(name: str) -> float:
        hit = metrics[metrics["metric"] == name]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    st.error(
        "**An unmatched rate is monotonically increasing in the error it is supposed "
        "to detect.** Every fabricated match *improves* the score. That makes it the "
        "one metric in this project that rewards its own worst failure mode — so it "
        "is never reported alone, and the rejected rule is kept running permanently "
        "beside it as a live demonstration.")

    fuzzy = audit[audit["section"] == "fuzzy_match"]
    ablation = audit[audit["section"].str.contains("ablation", na=False)]

    left, right = st.columns(2)
    with left:
        st.markdown("**The shipped cascade**")
        st.caption(f"{len(fuzzy)} surviving non-exact matches — small enough to "
                   f"eyeball, which is the feature rather than a limitation.")
        if not fuzzy.empty:
            st.dataframe(
                fuzzy[["board_name", "matched_name", "rule", "board_season",
                       "seasons_apart"]],
                width="stretch", hide_index=True)
    with right:
        st.markdown("**The rejected rule, re-run as a permanent ablation**")
        if not ablation.empty:
            st.dataframe(ablation.dropna(axis=1, how="all"), width="stretch",
                         hide_index=True)
        st.caption("'Same surname + same first initial' scores a *better* unmatched "
                   "rate than the cascade while fabricating most of its non-exact "
                   "matches — Cameron Boozer → Carlos Boozer, Darryn Peterson → Drew "
                   "Peterson. If someone later simplifies the cascade back toward it, "
                   "this panel says so immediately.")

    st.info(
        "**Two independent guards beat one clever rule.** A first name must be a "
        "genuine ≥3-character *prefix* **and** the candidate must have played within "
        "~3 seasons of the row. Either alone still invents people: prefix-only "
        "matched a 2026 rookie to a player last seen in 1996-97. And **'no such "
        "entity' is kept apart from 'join failed'** — DK pool entries who have never "
        "played an NBA game are reported as `no_nba_history`, not as `unmatched`, "
        "because collapsing them turns 'the 2026 draft class exists' into a fake "
        "defect that hides real ones.")
    note("Name-based joins exist in exactly two places — the report calibration "
         "(`game_id` + `name_key`, exact match, no fuzzy tier at all) and the ADP "
         "modules. Everything else keys on `player_id`.")

    with detail("Match audit — every row"):
        st.dataframe(audit, width="stretch", hide_index=True)
    provenance("`make adp-panel` → `outputs/eda/adp_match_audit.csv`")


# ── Artifact inventory ────────────────────────────────────────────────────────

def _inventory(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Artifact inventory")
    note("Everything the pipeline has written, with row counts read from parquet "
         "footers rather than by loading the files. `make dashboard-audit` checks the "
         "inverse: artifacts here that no tab reads and no decision names.")

    inv = inventory()
    if inv.empty:
        st.warning("No artifacts found — run `make eda`.")
        return
    by_dir = inv.groupby("directory").agg(files=("artifact", "count"),
                                          mb=("mb", "sum")).reset_index()
    stat_tiles([(row.directory, f"{row.files} files", f"{row.mb:,.1f} MB")
                for row in by_dir.itertuples()])
    with detail(f"All {len(inv)} artifacts"):
        st.dataframe(inv[["directory", "artifact", "rows", "mb", "modified"]],
                     width="stretch", hide_index=True)
