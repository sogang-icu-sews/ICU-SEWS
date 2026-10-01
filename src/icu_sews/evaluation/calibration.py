"""Probability calibration helpers.

Owner: 이택훈
Fit on validation or out-of-fold predictions; never fit on the final test set.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt scaling for one-dimensional model scores."""

    def __init__(self) -> None:
        self.model = LogisticRegression()

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibration mapping on held-out predictions."""
        self.model.fit(probabilities.reshape(-1, 1), labels)

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probabilities."""
        return self.model.predict_proba(probabilities.reshape(-1, 1))[:, 1]
