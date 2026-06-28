"""Training loop for LSTM / Transformer models."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml
import pandas as pd

from src.data.dataset import PlayerSequenceDataset
from src.features.encode import load_artifacts, TARGET_COLS
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


def train(cfg_path: str = "configs/default.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    features_dir = Path(cfg["data"]["features_dir"])
    df = pd.read_parquet(features_dir / "features.parquet")
    scaler, feature_cols = load_artifacts(features_dir)
    target_cols = cfg["features"]["target_stats"]

    dataset = PlayerSequenceDataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seq_len=cfg["features"]["sequence_length"],
    )
    print(f"Dataset: {len(dataset):,} samples, {len(feature_cols)} features, {len(target_cols)} targets")

    val_size = int(len(dataset) * cfg["training"]["val_split"])
    test_size = int(len(dataset) * cfg["training"]["test_split"])
    train_size = len(dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2)

    model = build_model(cfg, input_size=len(feature_cols), num_targets=len(target_cols)).to(device)
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

    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
