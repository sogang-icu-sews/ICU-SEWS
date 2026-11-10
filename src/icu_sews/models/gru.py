"""GRU sequence model for 24-hour patient observations.

Owner: 이택훈. Supports right-padded variable-length windows and static covariates.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GruSepsisModel(nn.Module):
    """Return one sepsis logit for each patient window."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        static_size: int = 0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_size = static_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + static_size, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        static: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ignore right padding; real missing observations remain valid time steps."""
        if lengths is not None:
            sequence = pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        _, hidden = self.gru(sequence)
        state = hidden[-1]
        if self.static_size:
            if static is None or static.shape != (state.shape[0], self.static_size):
                raise ValueError("Static input shape does not match the model")
            state = torch.cat([state, static], dim=1)
        return self.classifier(self.dropout(state)).squeeze(-1)
