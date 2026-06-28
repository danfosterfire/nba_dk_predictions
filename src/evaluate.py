"""Evaluate a trained model on the test split and report per-stat MAE / RMSE."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import yaml

from src.data.dataset import PlayerSequenceDataset
from src.features.encode import load_artifacts
from src.train import build_model, set_seed


def evaluate(cfg_path: str = "configs/default.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dir = Path(cfg["data"]["features_dir"])
    df = pd.read_parquet(features_dir / "features.parquet")
    scaler, feature_cols = load_artifacts(features_dir)
    target_cols = cfg["features"]["target_stats"]

    dataset = PlayerSequenceDataset(df, feature_cols=feature_cols, target_cols=target_cols,
                                    seq_len=cfg["features"]["sequence_length"])

    val_size = int(len(dataset) * cfg["training"]["val_split"])
    test_size = int(len(dataset) * cfg["training"]["test_split"])
    train_size = len(dataset) - val_size - test_size
    _, _, test_ds = random_split(dataset, [train_size, val_size, test_size])

    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = build_model(cfg, input_size=len(feature_cols), num_targets=len(target_cols)).to(device)
    model.load_state_dict(torch.load(Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt", map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            preds = model(X.to(device)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    rows = []
    for i, stat in enumerate(target_cols):
        mae = np.mean(np.abs(preds[:, i] - targets[:, i]))
        rmse = np.sqrt(np.mean((preds[:, i] - targets[:, i]) ** 2))
        rows.append({"stat": stat, "mae": round(mae, 3), "rmse": round(rmse, 3)})

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "test_metrics.csv", index=False)
    return results


if __name__ == "__main__":
    evaluate()
