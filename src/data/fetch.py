"""Pull player game logs from nba_api and persist to data/raw."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelogs, commonallplayers
from nba_api.stats.static import players as players_static


def fetch_all_players(active_only: bool = True) -> pd.DataFrame:
    """Return a DataFrame of all (active) NBA players with their IDs."""
    all_players = players_static.get_active_players() if active_only else players_static.get_players()
    return pd.DataFrame(all_players)


def fetch_season_game_logs(season: str, output_dir: str | Path = "data/raw") -> Path:
    """Download all player game logs for a season and save to a CSV.

    Args:
        season: NBA season string, e.g. "2023-24".
        output_dir: Directory to write the CSV.

    Returns:
        Path to the written CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"game_logs_{season.replace('-', '_')}.csv"

    if dest.exists():
        print(f"Already fetched {season}, skipping.")
        return dest

    print(f"Fetching game logs for {season}...")
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    df = logs.get_data_frames()[0]
    df.to_csv(dest, index=False)
    print(f"  Saved {len(df):,} rows to {dest}")
    return dest


def fetch_all_seasons(seasons: list[str], output_dir: str | Path = "data/raw", delay: float = 0.6) -> list[Path]:
    """Fetch multiple seasons, respecting nba_api rate limits with a small delay."""
    paths = []
    for season in seasons:
        path = fetch_season_game_logs(season, output_dir)
        paths.append(path)
        time.sleep(delay)
    return paths


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("configs/default.yaml"))
    fetch_all_seasons(cfg["data"]["seasons"], cfg["data"]["raw_dir"])
