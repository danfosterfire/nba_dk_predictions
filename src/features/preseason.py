"""Current-season preseason games, reduced to one row per (player, season).

**This is the only current-season observation the project is allowed to read.** The draft
happens after the preseason and before the opener (`docs/preseason-plan.md`), so the
information set gains a term: current-season roster membership × prior-season statistics ×
**current-season preseason statistics**. Everything here is a *forecast covariate* for the
season about to start — nothing in this file is ever a target row in any head's likelihood.

## What a preseason game is not

Preseason minutes are not regular-season minutes and the panel never pretends otherwise.
Starters play ~15–20 minutes, camp invitees are inflated, the early games are experiments
and the last one or two approximate the real rotation. So the quantities built here are
**within-team shares and ranks** plus participation, and the raw totals they are derived
from; a raw preseason MPG is deliberately not a headline column.

## Four quirks that shape the code, all measured rather than assumed

- **The window must be derived, never assumed to be October.** 2020-21's preseason ran
  Dec 11–19 (COVID) and 2011-12's ran Dec 16–22 (the lockout, two games per team). The
  same calendar trap the ADP freeze rule hit.
- **2019-20's preseason file straddles its own opener.** The NBA labels the July 2020
  bubble scrimmages `Pre Season` under the 2019-20 season key: 822 rows over 33 games
  played 2020-07-22 → 2020-07-28, nine months *after* the 2019-10-22 opener. Reading them
  as a feature for 2019-20 is not a subtle leak, it is the season itself. Every row is
  therefore filtered against the season's first regular-season game date, and the count
  dropped is reported per season rather than assumed to be zero.
- **Exhibition opponents are not NBA teams.** Real Madrid, Flamengo, Maccabi Ra'anana, the
  New Zealand Breakers and the Cairns Taipans all appear with their own `team_id` outside
  the `1610612737–1610612766` block, and so do their players. The NBA team's own rows in
  those games are real preseason games and are kept; the opponent's rows are dropped, which
  also removes ~50–70 non-NBA players a season who would otherwise enter the panel.
- **Some rows have no `player_id`.** 2003-04 carries 96 of them — team-total rows with
  `MIN` of exactly 480 (= 2 × 5 × 48) or 530 (one overtime) — plus one stray each in
  2010-11 and 2017-18. Every season before 2003-04 is a *single* such row and nothing else,
  which is why `fetch._FIRST_YEAR["pre_season"]` stops the orchestrator from writing those
  files at all: a header plus one junk row reads as a populated file and would never
  re-fetch.

## Who is in the panel, and who is deliberately not

Only players with **at least one preseason appearance**. A player with none has no row,
because the game logs hold appearances rather than rosters — the same construction problem
`src/features/availability.py` documents at length, and it is not solvable here either.
The `has_preseason` indicator and the zero-delta convention belong to each head's own
attach step, where the join against that head's population happens. What this module owes
that step is a *measurement* of who is missing, which is what
`outputs/eda/preseason_coverage.csv` carries.

Volume totals count every team the player appeared for; schedule-relative measures
(`missed_tail`, the shares and the ranks) are read over his **last** preseason team's
schedule, matching the project convention that a traded player is attributed wholly to his
last team.

Usage:
    python -m src.features.preseason
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year, _slug
from src.data.preprocess import compute_dk_pts
from src.features.adp import season_start_dates
from src.features.team_context import season_start_rosters
from src.models.component_rates import COUNT_HEADS

# The 30 NBA franchise ids form one contiguous block. Everything outside it is an
# exhibition opponent — a foreign club or an all-star selection — carrying its own players.
NBA_TEAM_ID_MIN = 1610612737
NBA_TEAM_ID_MAX = 1610612766

# Columns read off the raw preseason log. The box-score set is exactly what the component
# heads model, so the panel can produce a per-36 delta for every one of them.
BOX_COLS = ["min", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",
            "reb", "ast", "stl", "blk", "tov", "pts"]
LOG_COLS = ({"SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
             "GAME_ID", "GAME_DATE"} | {c.upper() for c in BOX_COLS})

# How many of a team's final preseason games count as "late". The early games are
# experiments; the last one or two approximate the rotation the season opens with, which is
# the only part of a preseason rotation that forecasts anything. Two rather than one because
# a single game is one rest decision away from being uninformative.
LATE_GAMES = 2

# A preseason ends within a few days of the opener — measured at 3–5 days for 22 of the 23
# seasons on file. 2003-04 is the sole exception at 20 days, which is the signature of a
# *truncated capture* rather than a short preseason: the API holds only the first four days
# of it. A gap wider than this means the tail of the preseason is missing, and the tail is
# exactly what `missed_tail` and the late-weighted shares are read over.
MAX_OPENER_GAP_DAYS = 10

COVERAGE_COLS = [
    "season", "season_start_year", "opener", "coverage_class",
    "rows", "rows_null_player", "rows_non_nba_team", "rows_after_opener", "rows_kept",
    "games", "players", "nba_teams", "non_nba_teams", "non_nba_team_ids",
    "first_date", "last_date", "window_days", "opener_gap_days", "rows_per_game",
    "team_games_min", "team_games_median", "team_games_max",
    "roster_players", "roster_with_preseason", "roster_coverage", "camp_invitees",
]


# ── Reading ───────────────────────────────────────────────────────────────────

def read_preseason_log(season: str, raw_dir: str | Path) -> pd.DataFrame:
    """Raw preseason player-game rows for one season, lower-cased and typed.

    Returns everything the file holds, junk included, because
    `preseason_coverage` counts what gets dropped and a reader that drops silently
    cannot be audited. `clean_rows` is the filter.
    """
    path = Path(raw_dir) / f"game_logs_pre_season_{_slug(season)}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=lambda c: c in LOG_COLS, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for c in ["player_id", "team_id", "game_id"] + BOX_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "season", season)
    return df


def is_nba_team(team_id: pd.Series) -> pd.Series:
    """True for the 30 NBA franchises, False for exhibition opponents."""
    return pd.to_numeric(team_id, errors="coerce").between(
        NBA_TEAM_ID_MIN, NBA_TEAM_ID_MAX)


def clean_rows(df: pd.DataFrame, opener: pd.Timestamp | None) -> pd.DataFrame:
    """Drop junk rows, exhibition-opponent rows, and anything at or after the opener.

    The opener filter is the point-in-time guarantee and it is a *filter*, not an
    assertion, because 2019-20 would fail an assertion: its file carries the July 2020
    bubble scrimmages under the same season key. Passing `opener=None` (no regular-season
    log on disk yet — the production case in October) keeps every row, since there is no
    date to be later than.
    """
    if df.empty:
        return df
    out = df.dropna(subset=["player_id", "team_id", "game_id", "game_date"])
    out = out[is_nba_team(out["team_id"])]
    if opener is not None:
        out = out[out["game_date"] < pd.Timestamp(opener)]
    out = out.copy()
    for c in ("player_id", "team_id", "game_id"):
        out[c] = out[c].astype("int64")
    out[BOX_COLS] = out[BOX_COLS].fillna(0.0).astype(float)
    return out.reset_index(drop=True)


def team_schedule(rows: pd.DataFrame, late_games: int = LATE_GAMES) -> pd.DataFrame:
    """Each NBA team's preseason games, indexed chronologically.

    A game against a foreign club is still a preseason game for the NBA team, so it is in
    the schedule — only the opponent's own rows were dropped upstream.
    """
    games = (rows[["team_id", "game_id", "game_date"]].drop_duplicates()
             .sort_values(["team_id", "game_date", "game_id"]))
    games["team_game_index"] = games.groupby("team_id").cumcount()
    games["team_pre_games"] = games.groupby("team_id")["game_id"].transform("size")
    games["is_late"] = (games["team_game_index"]
                        >= games["team_pre_games"] - late_games).astype(int)
    return games.reset_index(drop=True)


# ── The panel ─────────────────────────────────────────────────────────────────

def _shares(rows: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Within-team minutes share and rank, per (team, player), over the rows given."""
    by_pt = (rows.groupby(["team_id", "player_id"], as_index=False)
             .agg(**{f"min{suffix}": ("min", "sum"), f"gp{suffix}": ("game_id", "nunique")}))
    team_min = by_pt.groupby("team_id")[f"min{suffix}"].transform("sum")
    by_pt[f"min_share{suffix}"] = by_pt[f"min{suffix}"] / team_min.replace(0, np.nan)
    by_pt[f"min_rank{suffix}"] = (by_pt.groupby("team_id")[f"min_share{suffix}"]
                                  .rank(ascending=False, method="min").astype("Int64"))
    return by_pt


def _tail_position(rows: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Where a player's appearances sit in his team's preseason schedule.

    `missed_tail` is the count of his team's final preseason games he did not appear in —
    the health reading the plan is after, taken days before the opener. `missed_lead` is
    its mirror at the front, which separates "arrived late" from "shut down early".
    """
    idx = rows.merge(sched[["team_id", "game_id", "team_game_index", "team_pre_games"]],
                     on=["team_id", "game_id"], how="left")
    out = (idx.groupby(["team_id", "player_id"], as_index=False)
           .agg(first_game_index=("team_game_index", "min"),
                last_game_index=("team_game_index", "max"),
                team_pre_games=("team_pre_games", "max")))
    out["missed_lead"] = out["first_game_index"].astype(int)
    out["missed_tail"] = (out["team_pre_games"] - 1 - out["last_game_index"]).astype(int)
    out["played_final_game"] = (out["missed_tail"] == 0).astype(int)
    return out


def build_season_panel(season: str, raw_dir: str | Path,
                       opener: pd.Timestamp | None = None,
                       late_games: int = LATE_GAMES) -> pd.DataFrame:
    """One row per (player_id, season) for every player with a preseason appearance."""
    rows = clean_rows(read_preseason_log(season, raw_dir), opener)
    if rows.empty:
        return pd.DataFrame()

    sched = team_schedule(rows, late_games)

    # Volume across every team he appeared for — the totals the per-36 rates come off.
    rows = rows.copy()
    rows["dk_pts"] = compute_dk_pts(rows)
    totals = (rows.groupby(["season", "player_id"], as_index=False)
              .agg(player_name=("player_name", "last"),
                   gp_pre=("game_id", "nunique"),
                   n_teams_pre=("team_id", "nunique"),
                   first_date=("game_date", "min"),
                   last_date=("game_date", "max"),
                   dk_pts_pre=("dk_pts", "sum"),
                   **{f"{c}_pre": (c, "sum") for c in BOX_COLS}))

    # Schedule-relative measures over his LAST preseason team.
    last_team = (rows.sort_values(["game_date", "game_id"])
                 .groupby(["season", "player_id"], as_index=False)
                 .agg(team_id=("team_id", "last")))

    late_rows = rows.merge(sched.loc[sched["is_late"] == 1, ["team_id", "game_id"]],
                           on=["team_id", "game_id"], how="inner")
    # The full-window share frame is keyed per (team, player), so its `min`/`gp` are that
    # team's alone — `totals` already carries the across-team volume and owns those names.
    shares = _shares(rows, "_pre").drop(columns=["min_pre", "gp_pre"])
    frame = (shares
             .merge(_shares(late_rows, "_pre_late"), on=["team_id", "player_id"], how="left")
             .merge(_tail_position(rows, sched), on=["team_id", "player_id"], how="left"))
    team_late = (sched.groupby("team_id", as_index=False)
                 .agg(team_late_games=("is_late", "sum")))
    frame = frame.merge(team_late, on="team_id", how="left")

    out = totals.merge(last_team, on=["season", "player_id"], how="left")
    out = out.merge(frame, on=["team_id", "player_id"], how="left")

    # A player who appeared for his last team but in none of its final games has a share
    # of zero, not a missing one — that is the signal, so it must not read as absent.
    # `min_rank_pre_late` is the exception and stays null: he has no rank among players who
    # played, and inventing a last place would make "did not play" look like "played least".
    for c in ("min_pre_late", "gp_pre_late", "min_share_pre_late"):
        out[c] = out[c].fillna(0.0)
    out["min_rank_pre_late"] = out["min_rank_pre_late"].astype("Float64")

    out["gp_share_pre"] = out["gp_pre"] / out["team_pre_games"].replace(0, np.nan)
    out["mpg_pre"] = out["min_pre"] / out["gp_pre"].replace(0, np.nan)

    # Per-36 rates for the seven count heads, plus the three-point share of attempts —
    # the two bases `src/models/component_rates.py` actually models.
    denom = out["min_pre"].replace(0, np.nan)
    for c in COUNT_HEADS:
        out[f"pre_per36_{c}"] = out[f"{c}_pre"] / denom * 36.0
    out["pre_per36_dk_pts"] = out["dk_pts_pre"] / denom * 36.0
    out["pre_fg3a_share"] = out["fg3a_pre"] / out["fga_pre"].replace(0, np.nan)

    out.insert(1, "season_start_year", _season_start_year(season))
    return out.sort_values(["player_id"]).reset_index(drop=True)


def build_panel(seasons: list[str], raw_dir: str | Path,
                late_games: int = LATE_GAMES) -> pd.DataFrame:
    """The panel across seasons. Seasons with no preseason file contribute nothing."""
    starts = season_start_dates(raw_dir)
    frames = []
    for s in seasons:
        opener = pd.Timestamp(starts[s]) if s in starts else None
        frame = build_season_panel(s, raw_dir, opener, late_games)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No preseason game log CSVs in {raw_dir}. Run `make fetch` — the backfill "
            "writes game_logs_pre_season_*.csv from 2003-04 onwards.")
    return pd.concat(frames, ignore_index=True)


# ── Coverage ──────────────────────────────────────────────────────────────────

def season_coverage(season: str, raw_dir: str | Path, opener: pd.Timestamp | None,
                    roster: pd.DataFrame) -> dict:
    """One coverage row: what the file holds, what was dropped, and who is missing.

    The three drop counts are reported separately because they mean different things —
    a junk row is an API artifact, an exhibition-opponent row is a real game that is not
    ours to model, and a post-opener row is a *leak that was caught*.
    """
    raw = read_preseason_log(season, raw_dir)
    rec = {c: 0 for c in COVERAGE_COLS}
    rec.update(season=season, season_start_year=_season_start_year(season),
               opener=str(pd.Timestamp(opener).date()) if opener is not None else "",
               coverage_class="absent", non_nba_team_ids="",
               first_date="", last_date="", roster_coverage=np.nan,
               rows_per_game=np.nan, opener_gap_days=np.nan)
    rec["roster_players"] = int(roster["player_id"].nunique())
    if raw.empty:
        return rec

    rec["rows"] = len(raw)
    null_player = raw["player_id"].isna()
    rec["rows_null_player"] = int(null_player.sum())
    kept = raw[~null_player]
    non_nba = ~is_nba_team(kept["team_id"])
    rec["rows_non_nba_team"] = int(non_nba.sum())
    rec["non_nba_teams"] = int(kept.loc[non_nba, "team_id"].nunique())
    rec["non_nba_team_ids"] = " ".join(
        str(int(t)) for t in sorted(kept.loc[non_nba, "team_id"].dropna().unique()))

    rows = clean_rows(raw, opener)
    if opener is not None:
        pre_opener = clean_rows(raw, None)
        rec["rows_after_opener"] = int(len(pre_opener) - len(rows))
    if rows.empty:
        return rec

    rec["rows_kept"] = len(rows)
    rec["games"] = int(rows["game_id"].nunique())
    rec["players"] = int(rows["player_id"].nunique())
    rec["nba_teams"] = int(rows["team_id"].nunique())
    rec["rows_per_game"] = round(len(rows) / rec["games"], 2)
    first, last = rows["game_date"].min(), rows["game_date"].max()
    rec["first_date"], rec["last_date"] = str(first.date()), str(last.date())
    rec["window_days"] = int((last - first).days) + 1
    if opener is not None:
        rec["opener_gap_days"] = int((pd.Timestamp(opener) - last).days)

    team_games = rows.groupby("team_id")["game_id"].nunique()
    rec["team_games_min"] = int(team_games.min())
    rec["team_games_median"] = float(team_games.median())
    rec["team_games_max"] = int(team_games.max())

    seen = set(rows["player_id"].unique())
    on_roster = set(roster["player_id"].unique())
    rec["roster_with_preseason"] = len(on_roster & seen)
    rec["roster_coverage"] = (len(on_roster & seen) / len(on_roster)) if on_roster else np.nan
    rec["camp_invitees"] = len(seen - on_roster)

    gap = rec["opener_gap_days"]
    rec["coverage_class"] = ("tail_missing"
                             if pd.notna(gap) and gap > MAX_OPENER_GAP_DAYS
                             else "covered")
    return rec


def preseason_coverage(seasons: list[str], raw_dir: str | Path,
                       roster_window_games: int = 10) -> pd.DataFrame:
    """The coverage artifact — one row per season, including seasons with no file."""
    starts = season_start_dates(raw_dir)
    rosters = season_start_rosters(seasons, raw_dir, roster_window_games)
    recs = []
    for s in seasons:
        opener = pd.Timestamp(starts[s]) if s in starts else None
        recs.append(season_coverage(s, raw_dir, opener,
                                    rosters[rosters["season"] == s]))
    return pd.DataFrame(recs)[COVERAGE_COLS]


# ── Artifacts ─────────────────────────────────────────────────────────────────

def save_artifacts(panel: pd.DataFrame, coverage: pd.DataFrame,
                   features_dir: str | Path, eda_dir: str | Path) -> dict[str, Path]:
    features_dir, eda_dir = Path(features_dir), Path(eda_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)
    paths = {"panel": features_dir / "preseason.parquet",
             "coverage": eda_dir / "preseason_coverage.csv"}
    panel.to_parquet(paths["panel"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    return paths


def load_panel(features_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(features_dir) / "preseason.parquet")


def run(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    eda_dir = Path(cfg["eda"]["output_dir"])
    seasons = cfg["data"]["seasons"]
    p_cfg = cfg.get("features", {}).get("preseason", {})
    late = p_cfg.get("late_games", LATE_GAMES)
    window = cfg.get("features", {}).get("team_context", {}).get("roster_window_games", 10)

    panel = build_panel(seasons, raw_dir, late)
    coverage = preseason_coverage(seasons, raw_dir, window)

    print(f"Built preseason panel: {len(panel):,} player-seasons over "
          f"{panel['season'].nunique()} seasons")
    have = coverage[coverage["rows_kept"] > 0]
    print(f"  coverage: {len(have)} of {len(coverage)} seasons carry preseason logs "
          f"({have['season'].min()} → {have['season'].max()}); "
          f"{int((coverage['coverage_class'] == 'tail_missing').sum())} tail-missing")
    print(f"  season-start roster coverage: {have['roster_coverage'].mean():.1%} mean, "
          f"{have['roster_coverage'].min():.1%} worst ({have.loc[have['roster_coverage'].idxmin(), 'season']}), "
          f"{int(have['camp_invitees'].sum()):,} camp invitees never reach one")

    dropped = have[["rows_null_player", "rows_non_nba_team", "rows_after_opener"]].sum()
    print(f"  dropped: {int(dropped['rows_null_player']):,} rows with no player_id, "
          f"{int(dropped['rows_non_nba_team']):,} exhibition-opponent rows, "
          f"{int(dropped['rows_after_opener']):,} rows at or after the opener")
    leaky = have[have["rows_after_opener"] > 0]
    for r in leaky.itertuples():
        print(f"    🔴 {r.season}: {r.rows_after_opener:,} rows postdate the "
              f"{r.opener} opener and were filtered — the July 2020 bubble scrimmages "
              "carry the `Pre Season` label under the 2019-20 key")
    short = have.nsmallest(3, "team_games_median")
    print("  shortest preseasons (median team games): "
          + ", ".join(f"{r.season} {r.team_games_median:.0f} ({r.first_date}→{r.last_date})"
                      for r in short.itertuples()))

    paths = save_artifacts(panel, coverage, features_dir, eda_dir)
    print(f"Saved {len(panel):,} panel rows → {paths['panel']}")
    print(f"Saved {len(coverage):,} coverage rows → {paths['coverage']}")
    return paths


if __name__ == "__main__":
    from src.features.preseason import run as _run

    cfg = yaml.safe_load(open("configs/default.yaml"))
    _run(cfg)
