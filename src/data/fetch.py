"""Pull all available player and team data from nba_api and persist to data/raw/."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguedashplayerbiostats,
    leaguedashplayerclutch,
    leaguedashplayerptshot,
    leaguedashplayershotlocations,
    leaguedashplayerstats,
    leaguedashptstats,
    leaguedashteamstats,
    leaguehustlestatsplayer,
    playergamelogs,
    playerestimatedmetrics,
    teamestimatedmetrics,
)
from nba_api.stats.static import players as players_static

DELAY = 0.6  # seconds between calls to respect rate limits

# Player tracking measure types from the SportVU/Second Spectrum system
PT_MEASURE_TYPES = [
    "SpeedDistance",  # avg speed and distance traveled
    "Possessions",    # touches, front-court touches, time of possession
    "CatchShoot",     # catch-and-shoot attempts and makes
    "PullUpShot",     # pull-up shot attempts and makes
    "Drives",         # drives to the basket
    "Passing",        # passes made, potential assists, secondary assists
    "ElbowTouch",     # elbow touches
    "PostTouch",      # post touches
    "PaintTouch",     # paint touches
]

# Measure types for LeagueDashPlayerStats and LeagueDashTeamStats.
# "Four Factors" is omitted for players: the endpoint returns a response with the
# LeagueDashPlayerStats result set missing entirely under every per_mode. Its stats
# (EFG_PCT, TM_TOV_PCT, OREB_PCT, FTA_RATE) are all carried by Advanced anyway.
# Team four-factors works fine and stays.
PLAYER_STAT_MEASURES = ["Base", "Advanced", "Defense", "Misc", "Scoring", "Usage"]
TEAM_STAT_MEASURES = ["Base", "Advanced", "Defense", "Four Factors", "Misc", "Scoring", "Opponent"]

# per_mode values cycled when a LeagueDashPlayerStats query comes back empty.
# The API caches negative results keyed on the full query string, so an empty
# response is not a rate limit and backoff does not clear it — changing any
# parameter (per_mode is the only one that is safe to vary) misses the poisoned
# cache entry and reaches the backend. Empty responses return in ~0.1s; real
# ones take ~2s.
PLAYER_STAT_PER_MODES = ["PerGame", "Totals", "Per100Possessions", "PerMinute"]

# Records which per_mode each player_stats file was actually fetched with, so
# downstream loaders can normalize counting stats to a common basis.
MANIFEST_NAME = "_fetch_manifest.csv"

# Earliest start-year for endpoints that weren't always available.
# Seasons before these years are skipped rather than attempted and errored.
#   player_tracking / pt_shot: SportVU cameras in all arenas from 2013-14
#   hustle:                    LeagueHustleStatsPlayer added for 2015-16
#   estimated:                 RAPM-based estimated metrics from 2014-15
_FIRST_YEAR: dict[str, int] = {
    "player_tracking": 2013,
    "pt_shot":         2013,
    "hustle":          2015,
    "estimated":       2014,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slug(season: str) -> str:
    return season.replace("-", "_")


def _season_start_year(season: str) -> int:
    """Return the calendar year a season starts in ('2013-14' → 2013)."""
    return int(season.split("-")[0])


def _save(df: pd.DataFrame, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    if len(df) == 0:
        print(f"  WARNING: saved 0 rows → {dest} (empty API response; file will be re-fetched next run)")
    else:
        print(f"  Saved {len(df):,} rows → {dest}")
    return dest


def _has_data_rows(dest: Path, header_rows: int = 1) -> bool:
    """True if `dest` holds at least one data row beyond its header.

    Reads only the first few lines rather than parsing the file — the raw
    directory holds hundreds of MB of CSVs and this runs once per file per run.
    """
    if dest.stat().st_size == 0:
        return False
    with open(dest) as f:
        for _ in range(header_rows):
            if not f.readline():
                return False
        return bool(f.readline().strip())


def _skip_or_fetch(dest: Path, label: str, header_rows: int = 1) -> bool:
    """True if `dest` is already populated and the fetch can be skipped.

    A file with a header but no data rows counts as *not* fetched, so empty
    responses that reached disk self-heal on the next run instead of being
    skipped forever.
    """
    if dest.exists():
        if _has_data_rows(dest, header_rows):
            print(f"Already fetched {label}, skipping.")
            return True
        print(f"Re-fetching {label} (existing file has no data rows)...")
        return False
    print(f"Fetching {label}...")
    return False


def _record_fetch(output_dir: Path, family: str, season: str, per_mode: str, rows: int) -> None:
    """Upsert one (family, season) row into data/raw/_fetch_manifest.csv."""
    dest = output_dir / MANIFEST_NAME
    row = {"family": family, "season": season, "per_mode": per_mode, "rows": rows}
    if dest.exists():
        manifest = pd.read_csv(dest)
        manifest = manifest[~((manifest["family"] == family) & (manifest["season"] == season))]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([row])
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest.sort_values(["family", "season"]).to_csv(dest, index=False)


# ── Players static ────────────────────────────────────────────────────────────

def fetch_all_players(active_only: bool = True) -> pd.DataFrame:
    """Return a DataFrame of all (active) NBA players with their IDs."""
    all_players = players_static.get_active_players() if active_only else players_static.get_players()
    return pd.DataFrame(all_players)


# ── Game logs ─────────────────────────────────────────────────────────────────

def fetch_season_game_logs(season: str, output_dir: str | Path = "data/raw",
                           season_type: str = "Regular Season") -> Path:
    """Download all player game logs for a season.

    Playoff logs land in their own file rather than being mixed into the regular-season
    one: every downstream consumer treats `game_logs_{season}.csv` as the regular season,
    and a deep run would otherwise silently inflate games played. As prior-season
    *workload* they matter — ~20 extra high-intensity games that are invisible today.
    """
    output_dir = Path(output_dir)
    suffix = "" if season_type == "Regular Season" else f"_{_slug(season_type.lower())}"
    dest = output_dir / f"game_logs{suffix}_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"game_logs/{season_type} {season}"):
        return dest
    df = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
    ).get_data_frames()[0]
    return _save(df, dest)


def fetch_team_rosters(season: str, output_dir: str | Path = "data/raw",
                       delay: float = DELAY) -> Path:
    """CommonTeamRoster for every team that played in `season`, as one file.

    Official roster membership *with experience*, which the availability panel otherwise
    has to infer from appearances — the bias documented in
    `src/features/availability.py`. Teams come off the season's own game log rather than
    the static team list, so relocated and defunct franchises resolve to the ids that
    actually played that year, and everything stays keyed on `team_id`.
    """
    output_dir = Path(output_dir)
    dest = output_dir / f"team_rosters_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"team_rosters {season}"):
        return dest

    log_path = output_dir / f"game_logs_{_slug(season)}.csv"
    if not log_path.exists():
        print(f"  Skipping team_rosters for {season} (no game log to read teams from)")
        return dest
    team_ids = sorted(pd.read_csv(log_path, usecols=["TEAM_ID"],
                                  low_memory=False)["TEAM_ID"].dropna().unique())

    frames = []
    for i, team_id in enumerate(team_ids):
        try:
            df = commonteamroster.CommonTeamRoster(
                team_id=int(team_id), season=season).get_data_frames()[0]
        except Exception as exc:
            print(f"  ERROR fetching roster for team {team_id} in {season}: {exc}")
            continue
        # The endpoint's own SEASON is the start year ("2023"); overwrite it with the
        # project's season key so this file joins to everything else in data/raw.
        df["SEASON"] = season
        frames.append(df)
        if i < len(team_ids) - 1:
            time.sleep(delay)

    rosters = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _save(rosters, dest)


# ── Season-level player stats ─────────────────────────────────────────────────

def fetch_player_stats(season: str, output_dir: Path, measure_type: str = "Base") -> Path:
    """LeagueDashPlayerStats — season averages across multiple measure types.

    measure_type options: Base, Advanced, Defense, Misc, Scoring, Usage

    On an empty response, cycles per_mode (see PLAYER_STAT_PER_MODES) to sidestep
    the cached negative result, and records the winning mode in the fetch manifest
    so downstream loaders can normalize the file to a common basis.
    """
    slug = measure_type.lower().replace(" ", "_")
    family = f"player_stats_{slug}"
    dest = output_dir / f"{family}_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"player_stats/{measure_type} {season}"):
        return dest

    df = pd.DataFrame()
    per_mode = PLAYER_STAT_PER_MODES[0]
    for i, per_mode in enumerate(PLAYER_STAT_PER_MODES):
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense=measure_type,
            per_mode_detailed=per_mode,
        ).get_data_frames()[0]
        if len(df) > 0:
            break
        remaining = PLAYER_STAT_PER_MODES[i + 1:]
        if remaining:
            print(f"  Empty response at per_mode={per_mode}; retrying as {remaining[0]}")
            time.sleep(DELAY)

    _record_fetch(output_dir, family, season, per_mode, len(df))
    return _save(df, dest)


def fetch_player_bio_stats(season: str, output_dir: Path) -> Path:
    """LeagueDashPlayerBioStats — age, height, weight, experience, draft info."""
    dest = output_dir / f"player_bio_stats_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"player_bio_stats {season}"):
        return dest
    df = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_simple="PerGame",
    ).get_data_frames()[0]
    return _save(df, dest)


def fetch_player_shot_locations(season: str, output_dir: Path) -> Path:
    """LeagueDashPlayerShotLocations — FGA/FGM by zone (RA, paint, mid-range, corners, above break)."""
    dest = output_dir / f"player_shot_locations_{_slug(season)}.csv"
    # This family alone writes a two-row (MultiIndex) header.
    if _skip_or_fetch(dest, f"player_shot_locations {season}", header_rows=2):
        return dest
    df = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    return _save(df, dest)


def fetch_player_pt_shot(season: str, output_dir: Path) -> Path:
    """LeagueDashPlayerPtShot — shot breakdown by type (dribble-off, catch-and-shoot, etc.)."""
    dest = output_dir / f"player_pt_shot_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"player_pt_shot {season}"):
        return dest
    df = leaguedashplayerptshot.LeagueDashPlayerPtShot(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_simple="PerGame",
    ).get_data_frames()[0]
    return _save(df, dest)


def fetch_player_clutch(season: str, output_dir: Path) -> Path:
    """LeagueDashPlayerClutch — stats in clutch situations (last 5 min, ≤5 pts)."""
    dest = output_dir / f"player_clutch_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"player_clutch {season}"):
        return dest
    df = leaguedashplayerclutch.LeagueDashPlayerClutch(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    return _save(df, dest)


def fetch_hustle_stats(season: str, output_dir: Path) -> Path:
    """LeagueHustleStatsPlayer — contested shots, deflections, charges drawn, loose balls."""
    dest = output_dir / f"player_hustle_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"player_hustle {season}"):
        return dest
    df = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_time="PerGame",
    ).get_data_frames()[0]
    return _save(df, dest)


def fetch_player_estimated_metrics(season: str, output_dir: Path) -> Path:
    """PlayerEstimatedMetrics — estimated plus/minus, off/def ratings (EPM-style)."""
    dest = output_dir / f"player_estimated_metrics_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"player_estimated_metrics {season}"):
        return dest
    df = playerestimatedmetrics.PlayerEstimatedMetrics(
        season=season,
        season_type="Regular Season",
    ).get_data_frames()[0]
    return _save(df, dest)


def fetch_pt_stats(season: str, output_dir: Path, pt_measure_type: str) -> Path:
    """LeagueDashPtStats — player tracking stats for one measure type.

    pt_measure_type options: SpeedDistance, Possessions, CatchShoot, PullUpShot,
                             Drives, Passing, ElbowTouch, PostTouch, PaintTouch
    """
    slug = pt_measure_type.lower()
    dest = output_dir / f"player_tracking_{slug}_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"player_tracking/{pt_measure_type} {season}"):
        return dest
    df = leaguedashptstats.LeagueDashPtStats(
        season=season,
        season_type_all_star="Regular Season",
        pt_measure_type=pt_measure_type,
        player_or_team="Player",
        per_mode_simple="PerGame",
    ).get_data_frames()[0]
    return _save(df, dest)


# ── Season-level team stats ───────────────────────────────────────────────────

def fetch_team_stats(season: str, output_dir: Path, measure_type: str = "Base") -> Path:
    """LeagueDashTeamStats — team season averages across multiple measure types.

    measure_type options: Base, Advanced, Defense, Four Factors, Misc, Scoring, Opponent
    """
    slug = measure_type.lower().replace(" ", "_")
    dest = output_dir / f"team_stats_{slug}_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"team_stats/{measure_type} {season}"):
        return dest
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense=measure_type,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    return _save(df, dest)


def fetch_team_estimated_metrics(season: str, output_dir: Path) -> Path:
    """TeamEstimatedMetrics — estimated pace, off/def ratings for each team."""
    dest = output_dir / f"team_estimated_metrics_{_slug(season)}.csv"
    if _skip_or_fetch(dest, f"team_estimated_metrics {season}"):
        return dest
    df = teamestimatedmetrics.TeamEstimatedMetrics(
        season=season,
        season_type="Regular Season",
    ).get_data_frames()[0]
    return _save(df, dest)


# ── Orchestration ─────────────────────────────────────────────────────────────

def fetch_all_for_season(season: str, output_dir: str | Path = "data/raw", delay: float = DELAY) -> None:
    """Fetch every supported data type for one season, respecting rate limits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    year = _season_start_year(season)
    tasks: list[tuple[str, callable]] = []

    # Game logs (per-game box score), regular season and playoffs
    tasks.append(("game_logs", lambda: fetch_season_game_logs(season, output_dir)))
    tasks.append(("game_logs/Playoffs",
                  lambda: fetch_season_game_logs(season, output_dir, "Playoffs")))

    # Official rosters — 30 calls, so it goes last in the per-season block below
    tasks.append(("team_rosters", lambda: fetch_team_rosters(season, output_dir, delay)))

    # Player stats by measure type
    for m in PLAYER_STAT_MEASURES:
        tasks.append((f"player_stats/{m}", lambda m=m: fetch_player_stats(season, output_dir, m)))

    # Player supplementary season-level stats (always available)
    tasks += [
        ("player_bio_stats",      lambda: fetch_player_bio_stats(season, output_dir)),
        ("player_shot_locations", lambda: fetch_player_shot_locations(season, output_dir)),
        ("player_clutch",         lambda: fetch_player_clutch(season, output_dir)),
    ]

    # Player tracking-derived shot breakdown (2013-14+)
    if year >= _FIRST_YEAR["pt_shot"]:
        tasks.append(("player_pt_shot", lambda: fetch_player_pt_shot(season, output_dir)))
    else:
        print(f"  Skipping player_pt_shot for {season} (available from {_FIRST_YEAR['pt_shot']}-XX onwards)")

    # Hustle stats (2015-16+)
    if year >= _FIRST_YEAR["hustle"]:
        tasks.append(("player_hustle", lambda: fetch_hustle_stats(season, output_dir)))
    else:
        print(f"  Skipping player_hustle for {season} (available from {_FIRST_YEAR['hustle']}-XX onwards)")

    # Estimated metrics (2014-15+)
    if year >= _FIRST_YEAR["estimated"]:
        tasks.append(("player_estimated_metrics", lambda: fetch_player_estimated_metrics(season, output_dir)))
    else:
        print(f"  Skipping player_estimated_metrics for {season} (available from {_FIRST_YEAR['estimated']}-XX onwards)")

    # Player tracking stats — SportVU/Second Spectrum (2013-14+)
    if year >= _FIRST_YEAR["player_tracking"]:
        for pt in PT_MEASURE_TYPES:
            tasks.append((f"player_tracking/{pt}", lambda pt=pt: fetch_pt_stats(season, output_dir, pt)))
    else:
        print(f"  Skipping player tracking for {season} (available from {_FIRST_YEAR['player_tracking']}-XX onwards)")

    # Team stats by measure type
    for m in TEAM_STAT_MEASURES:
        tasks.append((f"team_stats/{m}", lambda m=m: fetch_team_stats(season, output_dir, m)))

    if year >= _FIRST_YEAR["estimated"]:
        tasks.append(("team_estimated_metrics", lambda: fetch_team_estimated_metrics(season, output_dir)))
    else:
        print(f"  Skipping team_estimated_metrics for {season} (available from {_FIRST_YEAR['estimated']}-XX onwards)")

    for i, (label, task) in enumerate(tasks):
        try:
            task()
        except Exception as exc:
            print(f"  ERROR fetching {label} for {season}: {exc}")
        if i < len(tasks) - 1:
            time.sleep(delay)


def fetch_all_seasons(seasons: list[str], output_dir: str | Path = "data/raw", delay: float = DELAY) -> None:
    """Fetch all data types for every season in the list."""
    for season in seasons:
        print(f"\n{'=' * 55}")
        print(f"  Season: {season}")
        print("=" * 55)
        fetch_all_for_season(season, output_dir, delay)


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("configs/default.yaml"))
    fetch_all_seasons(cfg["data"]["seasons"], cfg["data"]["raw_dir"])
