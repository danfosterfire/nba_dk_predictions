"""XGBoost baseline — one model per target stat, trained on rolling features."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


TARGET_COLS = ["pts", "reb", "ast", "min"]


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_cols: list[str] = TARGET_COLS,
) -> dict[str, xgb.XGBRegressor]:
    models = {}
    for i, stat in enumerate(target_cols):
        model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        val_mae = mean_absolute_error(y_val[:, i], model.predict(X_val))
        print(f"  {stat:>4s}  val MAE = {val_mae:.3f}")
        models[stat] = model
    return models


def save_baselines(models: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stat, model in models.items():
        with open(out_dir / f"baseline_{stat}.pkl", "wb") as f:
            pickle.dump(model, f)


def load_baselines(out_dir: str | Path, target_cols: list[str] = TARGET_COLS) -> dict:
    out_dir = Path(out_dir)
    models = {}
    for stat in target_cols:
        with open(out_dir / f"baseline_{stat}.pkl", "rb") as f:
            models[stat] = pickle.load(f)
    return models
