"""Shared prediction contracts used by models and the API.

Owners: 이택훈, 김진호
Changing this contract requires both owners to review the pull request.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """Model-agnostic prediction returned to the serving layer."""

    probability: float
    top_features: list[dict[str, float | str]] = field(default_factory=list)
    model_version: str = "untrained-scaffold"


class ProbabilityModel(Protocol):
    """Minimal interface required for an ensemble component."""

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        """Return a one-dimensional probability array."""
        ...

