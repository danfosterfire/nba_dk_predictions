"""Run inference for upcoming games given recent player game logs."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import PlayerSequenceDataset
from src.features.encode import build_feature_matrix, load_artifacts
from src.train import build_model, set_seed


def predict_from_logs(
    recent_logs: pd.DataFrame,
    cfg_path: str = "configs/default.yaml",
) -> pd.DataFrame:
    """Predict the next game stats for each player in recent_logs.

    Args:
        recent_logs: DataFrame of recent game logs (same schema as processed data).
                     Must contain at least seq_len games per player.
        cfg_path:    Path to config YAML.

    Returns:
        DataFrame with columns [player_id, player_name] + target_cols.
    """
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dir = Path(cfg["data"]["features_dir"])
    scaler, feature_cols = load_artifacts(features_dir)
    target_cols = cfg["features"]["target_stats"]
    seq_len = cfg["features"]["sequence_length"]

    df = build_feature_matrix(recent_logs, windows=cfg["features"]["rolling_windows"])
    df[feature_cols] = scaler.transform(df[feature_cols].fillna(0))

    model = build_model(cfg, input_size=len(feature_cols), num_targets=len(target_cols)).to(device)
    model.load_state_dict(torch.load(Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt", map_location=device))
    model.eval()

    records = []
    for player_id, player_df in df.groupby("player_id", sort=False):
        player_df = player_df.sort_values("game_date")
        if len(player_df) < seq_len:
            continue
        seq = player_df[feature_cols].to_numpy(dtype=np.float32)[-seq_len:]
        x = torch.from_numpy(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]
        row = {"player_id": player_id, "player_name": player_df["player_name"].iloc[-1]}
        row.update(dict(zip(target_cols, pred)))
        records.append(row)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick smoke test using the processed data
    cfg = yaml.safe_load(open("configs/default.yaml"))
    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "game_logs.parquet")
    preds = predict_from_logs(df)
    print(preds.head(10).to_string(index=False))

    out = Path(cfg["evaluation"]["predictions_dir"]) / "next_game_predictions.csv"
    preds.to_csv(out, index=False)
    print(f"Saved predictions → {out}")
