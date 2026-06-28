"""Inference: given a player's prior-season game logs, predict dk_pts per game."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import SeasonBoundaryDataset
from src.train import build_model, set_seed


def predict_from_prior_season(
    prior_season_logs: pd.DataFrame,
    cfg_path: str = "configs/default.yaml",
) -> pd.DataFrame:
    """Predict dk_pts for each player based on their prior-season game logs.

    Args:
        prior_season_logs: Processed game logs for the PRIOR season (same schema
                           as game_logs.parquet — must include all columns in
                           SeasonBoundaryDataset.FEATURE_COLS).
        cfg_path:          Path to config YAML.

    Returns:
        DataFrame with columns [player_id, player_name, predicted_dk_pts].
    """
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dir = Path(cfg["data"]["features_dir"])
    with open(features_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(features_dir / "feature_cols.pkl", "rb") as f:
        feat_cols = pickle.load(f)

    seq_len = cfg["features"]["sequence_length"]
    num_targets = len(cfg["features"]["target_stats"])

    model = build_model(cfg, input_size=len(feat_cols), num_targets=num_targets).to(device)
    model.load_state_dict(
        torch.load(Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt", map_location=device)
    )
    model.eval()

    df = prior_season_logs.copy()
    df[feat_cols] = scaler.transform(df[feat_cols].fillna(0))

    records = []
    for player_id, pdf in df.groupby("player_id", sort=False):
        pdf = pdf.sort_values("game_date")
        if len(pdf) == 0:
            continue
        prior_feats = pdf[feat_cols].to_numpy(dtype=np.float32)
        seq = prior_feats[-seq_len:]
        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), len(feat_cols)), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        x = torch.from_numpy(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_dk = model(x).cpu().numpy()[0, 0]
        records.append({
            "player_id": player_id,
            "player_name": pdf["player_name"].iloc[-1],
            "predicted_dk_pts": float(pred_dk),
        })

    return pd.DataFrame(records).sort_values("predicted_dk_pts", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    # Use the most recent season in processed data as the "prior season" for inference
    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "game_logs.parquet")
    latest_season = sorted(df["season"].unique())[-1]
    prior_logs = df[df["season"] == latest_season]

    preds = predict_from_prior_season(prior_logs)
    print(preds.head(20).to_string(index=False))

    out = Path(cfg["evaluation"]["predictions_dir"])
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "next_season_predictions.csv", index=False)
    print(f"Saved predictions → {out / 'next_season_predictions.csv'}")
