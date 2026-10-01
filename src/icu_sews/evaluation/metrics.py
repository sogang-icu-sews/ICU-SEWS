"""Core discrimination and calibration metrics.

Owner: 이택훈
The official PhysioNet utility implementation must be added from the cited source
and verified against the official examples before reporting results.
"""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class MetricSummary:
    """Minimum model comparison metrics."""

    auroc: float
    auprc: float
    brier_score: float

    def to_dict(self) -> dict[str, float]:
        """Return a serializable representation."""
        return asdict(self)


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> MetricSummary:
    """Calculate metrics from aligned binary labels and probabilities."""
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have identical shapes")
    return MetricSummary(
        auroc=float(roc_auc_score(labels, probabilities)),
        auprc=float(average_precision_score(labels, probabilities)),
        brier_score=float(brier_score_loss(labels, probabilities)),
    )


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute equal-width Expected Calibration Error."""
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    if total == 0:
        raise ValueError("labels must not be empty")

    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        included = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if included.any():
            accuracy = labels[included].mean()
            confidence = probabilities[included].mean()
            error += included.mean() * abs(accuracy - confidence)
    return float(error)
