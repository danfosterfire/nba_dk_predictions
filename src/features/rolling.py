"""Rolling aggregate features over recent game logs per player."""

import pandas as pd


STAT_COLS = ["dk_pts", "pts", "reb", "ast", "min", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "tov", "stl", "blk"]


def add_rolling_features(df: pd.DataFrame, windows: list[int] = [5, 10, 15]) -> pd.DataFrame:
    """Add per-player rolling mean and std for each stat over each window size.

    Operates in-place on a copy; assumes df is sorted by (player_id, game_date).
    """
    df = df.copy()
    for col in STAT_COLS:
        if col not in df.columns:
            continue
        for w in windows:
            # shift(1) so we only use past games, not the current one
            rolled = df.groupby("player_id")[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"{col}_roll{w}_mean"] = rolled

            rolled_std = df.groupby("player_id")[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).std().fillna(0)
            )
            df[f"{col}_roll{w}_std"] = rolled_std

    return df


def add_shooting_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Derived efficiency columns — avoids division by zero."""
    df = df.copy()
    df["fg_pct"] = df["fgm"] / df["fga"].replace(0, pd.NA)
    df["fg3_pct"] = df["fg3m"] / df["fg3a"].replace(0, pd.NA)
    df["ft_pct"] = df["ftm"] / df["fta"].replace(0, pd.NA)
    df[["fg_pct", "fg3_pct", "ft_pct"]] = df[["fg_pct", "fg3_pct", "ft_pct"]].fillna(0)
    return df
