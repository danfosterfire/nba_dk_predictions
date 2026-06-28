"""LSTM-based sequence model for player stat prediction."""

import torch
import torch.nn as nn


class PlayerLSTM(nn.Module):
    """Encodes a sequence of game logs with an LSTM, then projects to target stats.

    Args:
        input_size:  Number of features per time step.
        hidden_size: LSTM hidden dimension.
        num_layers:  Number of stacked LSTM layers.
        num_targets: Number of output stats to predict.
        dropout:     Dropout applied between LSTM layers (and after final hidden state).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_targets: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # Use the final layer's hidden state
        out = self.head(h_n[-1])
        return out  # (batch, num_targets)
