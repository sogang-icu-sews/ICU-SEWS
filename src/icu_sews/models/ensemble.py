"""Soft-voting probability ensemble.

Owner: 이택훈
Weights are configured in ``configs/model.yaml`` and tuned on validation data only.
"""

import numpy as np


def soft_vote(
    tree_probability: np.ndarray,
    sequence_probability: np.ndarray,
    tree_weight: float = 0.6,
) -> np.ndarray:
    """Combine aligned probability arrays with validated weights."""
    if not 0.0 <= tree_weight <= 1.0:
        raise ValueError("tree_weight must be between 0 and 1")
    if tree_probability.shape != sequence_probability.shape:
        raise ValueError("Ensemble inputs must have identical shapes")
    return tree_weight * tree_probability + (1.0 - tree_weight) * sequence_probability
