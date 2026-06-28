"""Opponent and matchup context features."""

import pandas as pd


def add_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the MATCHUP column to extract the opponent team abbreviation."""
    df = df.copy()
    # MATCHUP format: "LAL vs. GSW" (home) or "LAL @ GSW" (away)
    df["opponent"] = df["matchup"].str.extract(r"(?:vs\.|@)\s+(\w+)")
    return df


def add_opponent_defensive_rating(df: pd.DataFrame, def_ratings: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-computed opponent defensive ratings.

    Args:
        df: Game log DataFrame with an 'opponent' column.
        def_ratings: DataFrame with columns ['team', 'season', 'def_rtg'].
    """
    if "opponent" not in df.columns:
        df = add_opponent(df)
    df = df.merge(
        def_ratings.rename(columns={"team": "opponent", "def_rtg": "opp_def_rtg"}),
        on=["opponent"],
        how="left",
    )
    return df


def add_back_to_back_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag games where the player played the previous calendar day."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"])
    prev_date = df.groupby("player_id")["game_date"].shift(1)
    df["is_b2b"] = ((df["game_date"] - prev_date).dt.days == 1).astype(int)
    return df
