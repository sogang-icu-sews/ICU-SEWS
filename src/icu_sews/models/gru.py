"""GRU sequence model for 24-hour patient observations.

Owner: 이택훈
Next work: masking, Dataset/DataLoader, focal loss, early stopping, and calibration data export.
"""

import torch
from torch import nn


class GruSepsisModel(nn.Module):
    """Return one sepsis logit for each patient window."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch shaped ``[batch, time, features]``."""
        _, hidden = self.gru(sequence)
        return self.classifier(hidden[-1]).squeeze(-1)

