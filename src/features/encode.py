"""Categorical encoding and feature matrix construction."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

from src.features.rolling import add_rolling_features, add_shooting_efficiency
from src.features.matchup import add_opponent, add_back_to_back_flag


TARGET_COLS = ["dk_pts"]


def build_feature_matrix(df: pd.DataFrame, windows: list[int] = [5, 10, 15]) -> pd.DataFrame:
    """Apply all feature engineering steps and return the enriched DataFrame."""
    df = add_opponent(df)
    df = add_back_to_back_flag(df)
    df = add_shooting_efficiency(df)
    df = add_rolling_features(df, windows=windows)

    # Encode team and opponent as integer IDs
    for col in ["team_abbreviation", "opponent"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].fillna("UNK"))

    df["home"] = df["home"].fillna(0).astype(int)
    df = df.fillna(0)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of numeric feature columns (excluding targets and metadata)."""
    exclude = set(TARGET_COLS + [
        "player_id", "player_name", "team_abbreviation", "game_id",
        "game_date", "matchup", "opponent", "wl",
        "dk_pts", "pts", "reb", "ast", "stl", "blk", "tov",
        "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "min", "plus_minus",
    ])
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler


def save_artifacts(scaler: StandardScaler, feature_cols: list[str], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)


def load_artifacts(out_dir: str | Path) -> tuple[StandardScaler, list[str]]:
    out_dir = Path(out_dir)
    with open(out_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(out_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return scaler, feature_cols


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "game_logs.parquet")
    df = build_feature_matrix(df, windows=cfg["features"]["rolling_windows"])

    feature_cols = get_feature_cols(df)
    scaler = fit_scaler(df, feature_cols)

    df[feature_cols] = scaler.transform(df[feature_cols])

    out_dir = Path(cfg["data"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "features.parquet", index=False)
    save_artifacts(scaler, feature_cols, out_dir)
    print(f"Feature matrix: {df.shape} → {out_dir / 'features.parquet'}")
    print(f"Features ({len(feature_cols)}): {feature_cols[:10]} ...")
