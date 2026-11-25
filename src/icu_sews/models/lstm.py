"""Inference architecture matching data/models/train_ensemble.py (ICU-7)."""

import torch
from torch import nn


class SepsisLSTM(nn.Module):
    """Consume the full left-padded 24-hour window, including missing-hour channels."""

    def __init__(self, n_in, n_static, hidden, layers, dropout, head):
        super().__init__()
        self.lstm = nn.LSTM(
            n_in,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, 1)
        )

    def forward(self, x, st):
        out, _ = self.lstm(x)
        return self.head(torch.cat([out[:, -1], st], dim=1)).squeeze(1)
