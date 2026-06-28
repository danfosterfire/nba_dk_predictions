"""Training loop for LSTM / Transformer models."""

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
import yaml

from src.data.dataset import SeasonBoundaryDataset
from src.models.lstm import PlayerLSTM
from src.models.transformer import PlayerTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, input_size: int, num_targets: int) -> nn.Module:
    model_cfg = cfg["model"]
    if model_cfg["type"] == "lstm":
        return PlayerLSTM(
            input_size=input_size,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            num_targets=num_targets,
            dropout=model_cfg["dropout"],
        )
    elif model_cfg["type"] == "transformer":
        return PlayerTransformer(
            input_size=input_size,
            d_model=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            num_targets=num_targets,
            dropout=model_cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")


def _season_split(all_seasons: list[str]) -> tuple[set, set, set]:
    """Return (train_predict, val_predict, test_predict) season sets.

    Prediction season S requires data from season S-1 as input, so we need
    at least 2 seasons total to produce any samples.

    With N total seasons we get N-1 prediction seasons; the last two are
    held out for val and test, the rest form the training set.
    """
    predict = all_seasons[1:]
    if len(predict) < 2:
        raise ValueError(
            f"Need ≥3 seasons to produce a train/val/test split; got {all_seasons}. "
            "Add more seasons to data.seasons in configs/default.yaml."
        )
    return set(predict[:-2]), {predict[-2]}, {predict[-1]}


def train(cfg_path: str = "configs/default.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "game_logs.parquet")

    all_seasons = sorted(df["season"].unique())
    train_seasons, val_seasons, test_seasons = _season_split(all_seasons)

    print(f"All seasons:  {all_seasons}")
    print(f"Train target seasons: {sorted(train_seasons)}")
    print(f"Val   target season:  {sorted(val_seasons)}")
    print(f"Test  target season:  {sorted(test_seasons)}")

    # Fit scaler only on the prior seasons used as input for training
    train_input_seasons = {
        all_seasons[all_seasons.index(s) - 1]
        for s in train_seasons
    }
    feat_cols = SeasonBoundaryDataset.FEATURE_COLS
    feat_cols = [c for c in feat_cols if c in df.columns]

    scaler = StandardScaler()
    scaler.fit(df[df["season"].isin(train_input_seasons)][feat_cols].fillna(0))

    df = df.copy()
    df[feat_cols] = scaler.transform(df[feat_cols].fillna(0))

    dataset = SeasonBoundaryDataset(df, seq_len=cfg["features"]["sequence_length"])
    print(f"Dataset: {len(dataset):,} total samples")

    seasons = dataset.sample_seasons
    train_idx = [i for i, s in enumerate(seasons) if s in train_seasons]
    val_idx   = [i for i, s in enumerate(seasons) if s in val_seasons]
    test_idx  = [i for i, s in enumerate(seasons) if s in test_seasons]

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2)

    num_targets = len(cfg["features"]["target_stats"])
    model = build_model(cfg, input_size=len(feat_cols), num_targets=num_targets).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * len(X)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Persist scaler and feature column list for inference
    features_dir = Path(cfg["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)
    with open(features_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(features_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(feat_cols, f)

    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
