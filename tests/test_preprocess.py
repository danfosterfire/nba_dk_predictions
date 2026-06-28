import pandas as pd
import pytest
from src.data.preprocess import clean


def _make_df(n_games: int = 30, player_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "PLAYER_ID": [player_id] * n_games,
        "PLAYER_NAME": ["Test Player"] * n_games,
        "TEAM_ABBREVIATION": ["LAL"] * n_games,
        "GAME_ID": range(n_games),
        "GAME_DATE": pd.date_range("2024-01-01", periods=n_games, freq="2D").strftime("%Y-%m-%d"),
        "MATCHUP": ["LAL vs. GSW"] * n_games,
        "WL": ["W"] * n_games,
        "MIN": [30.0] * n_games,
        "PTS": [20.0] * n_games,
        "REB": [5.0] * n_games,
        "AST": [4.0] * n_games,
        "STL": [1.0] * n_games,
        "BLK": [0.5] * n_games,
        "TOV": [2.0] * n_games,
        "FGA": [15.0] * n_games,
        "FGM": [8.0] * n_games,
        "FG3A": [5.0] * n_games,
        "FG3M": [2.0] * n_games,
        "FTA": [4.0] * n_games,
        "FTM": [3.0] * n_games,
        "PLUS_MINUS": [5.0] * n_games,
    })


def test_clean_keeps_expected_columns():
    df = clean(_make_df())
    assert "pts" in df.columns
    assert "game_date" in df.columns
    assert "home" in df.columns


def test_clean_filters_by_min_games():
    df_few = _make_df(n_games=5, player_id=1)
    df_many = _make_df(n_games=30, player_id=2)
    combined = pd.concat([df_few, df_many], ignore_index=True)
    result = clean(combined, min_games=20)
    assert set(result["player_id"].unique()) == {2}


def test_clean_parses_home_flag():
    df = _make_df()
    df["MATCHUP"] = ["LAL vs. GSW"] * 15 + ["LAL @ GSW"] * 15
    result = clean(df, min_games=1)
    assert result["home"].isin([0, 1]).all()
