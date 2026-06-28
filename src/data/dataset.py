"""PyTorch Dataset that returns fixed-length game-log sequences per player."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PlayerSequenceDataset(Dataset):
    """Each sample is (sequence, target) for one player game.

    sequence: float32 tensor of shape (seq_len, num_features)
    target:   float32 tensor of shape (num_targets,)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        seq_len: int = 10,
    ):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        for _, player_df in df.groupby("player_id", sort=False):
            player_df = player_df.sort_values("game_date")
            features = player_df[self.feature_cols].to_numpy(dtype=np.float32)
            targets = player_df[self.target_cols].to_numpy(dtype=np.float32)

            # Slide a window: seq_len past games → next game target
            for i in range(self.seq_len, len(player_df)):
                seq = features[i - self.seq_len : i]
                tgt = targets[i]
                self.samples.append((seq, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, tgt = self.samples[idx]
        return torch.from_numpy(seq), torch.from_numpy(tgt)
