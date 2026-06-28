"""Clean and merge raw game log CSVs into a single processed DataFrame."""

from pathlib import Path

import pandas as pd
import yaml


KEEP_COLS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FGA",
    "FGM",
    "FG3A",
    "FG3M",
    "FTA",
    "FTM",
    "PLUS_MINUS",
]

RENAME = {c: c.lower() for c in KEEP_COLS}


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    frames = []
    for f in sorted(raw_dir.glob("game_logs_*.csv")):
        df = pd.read_csv(f)
        # filename: game_logs_2021_22.csv -> season "2021-22"
        slug = f.stem.replace("game_logs_", "")
        df["season"] = slug.replace("_", "-")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No game log CSVs found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def compute_dk_pts(df: pd.DataFrame) -> pd.Series:
    """DraftKings NBA fantasy points scoring."""
    dd_cats = (df[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    bonus = dd_cats.map({0: 0, 1: 0, 2: 1.5, 3: 4.5, 4: 4.5, 5: 4.5})
    return (
        df["pts"] * 1.0
        + df["fg3m"] * 0.5
        + df["reb"] * 1.25
        + df["ast"] * 1.5
        + df["stl"] * 2.0
        + df["blk"] * 2.0
        + df["tov"] * -0.5
        + bonus
    )


def clean(df: pd.DataFrame, min_games: int = 20) -> pd.DataFrame:
    # Keep only the columns we care about (drop silently if absent); preserve season
    cols = [c for c in KEEP_COLS if c in df.columns]
    extra = ["season"] if "season" in df.columns else []
    df = df[cols + extra].rename(columns=RENAME).copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home"] = df["matchup"].str.contains("vs\\.").astype(int)
    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    df = df.dropna(subset=["min", "pts", "reb", "ast"])

    df["dk_pts"] = compute_dk_pts(df)

    # Drop players with too few games
    counts = df.groupby("player_id")["game_id"].transform("count")
    df = df[counts >= min_games]

    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def run(cfg: dict) -> Path:
    df_raw = load_raw(cfg["data"]["raw_dir"])
    df = clean(df_raw, min_games=cfg["data"]["min_games"])

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "game_logs.parquet"
    df.to_parquet(dest, index=False)
    print(f"Processed {len(df):,} rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
