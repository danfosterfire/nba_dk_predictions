"""PyTorch Datasets for NBA player game-log sequences."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SeasonBoundaryDataset(Dataset):
    """Predicts each current-season game from the prior season's last N games.

    For player P in season S, every game in S shares the same input sequence:
    the last seq_len games P played in season S-1.  No current-season data
    leaks into the input features.

    Requires a 'season' column in df (e.g. "2023-24").
    """

    FEATURE_COLS = [
        "pts", "reb", "ast", "stl", "blk", "tov",
        "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",
        "min", "home", "dk_pts",
    ]

    def __init__(self, df: pd.DataFrame, seq_len: int = 10):
        self.seq_len = seq_len
        self._seqs: list[np.ndarray] = []
        self._targets: list[float] = []
        self.sample_seasons: list[str] = []
        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        feat_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        for _, pdf in df.groupby("player_id", sort=False):
            pdf = pdf.sort_values("game_date")
            seasons = sorted(pdf["season"].unique())
            if len(seasons) < 2:
                continue
            for i in range(1, len(seasons)):
                prior = pdf[pdf["season"] == seasons[i - 1]].sort_values("game_date")
                current = pdf[pdf["season"] == seasons[i]].sort_values("game_date")
                if len(prior) == 0 or len(current) == 0:
                    continue
                prior_feats = prior[feat_cols].to_numpy(dtype=np.float32)
                seq = prior_feats[-self.seq_len:]
                if len(seq) < self.seq_len:
                    pad = np.zeros((self.seq_len - len(seq), len(feat_cols)), dtype=np.float32)
                    seq = np.concatenate([pad, seq], axis=0)
                for _, row in current.iterrows():
                    self._seqs.append(seq.copy())
                    self._targets.append(float(row["dk_pts"]))
                    self.sample_seasons.append(seasons[i])

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._seqs[idx]),
            torch.tensor([self._targets[idx]], dtype=torch.float32),
        )


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
