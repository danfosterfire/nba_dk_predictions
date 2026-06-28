import numpy as np
import pandas as pd
import torch
from src.data.dataset import PlayerSequenceDataset


def _make_feature_df(n_games: int = 20, n_players: int = 3) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_cols = ["f1", "f2", "f3"]
    target_cols = ["pts", "reb", "ast", "min"]
    rows = []
    for pid in range(n_players):
        for i in range(n_games):
            row = {"player_id": pid, "game_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)}
            row.update({f: float(i + pid) for f in feature_cols})
            row.update({t: float(i) for t in target_cols})
            rows.append(row)
    return pd.DataFrame(rows), feature_cols, target_cols


def test_dataset_length():
    df, feat, tgt = _make_feature_df(n_games=20, n_players=3)
    seq_len = 5
    ds = PlayerSequenceDataset(df, feat, tgt, seq_len=seq_len)
    # Each player contributes (n_games - seq_len) samples
    assert len(ds) == 3 * (20 - seq_len)


def test_dataset_shapes():
    df, feat, tgt = _make_feature_df()
    seq_len = 10
    ds = PlayerSequenceDataset(df, feat, tgt, seq_len=seq_len)
    x, y = ds[0]
    assert x.shape == (seq_len, len(feat))
    assert y.shape == (len(tgt),)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
