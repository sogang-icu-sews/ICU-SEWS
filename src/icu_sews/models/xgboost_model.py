"""XGBoost wrapper with a stable inference interface.

Owner: 이택훈
Next work: training pipeline, Group K-Fold tuning, class weights, persistence, and SHAP.
"""

from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier


class XGBoostSepsisModel:
    """Thin wrapper that keeps training code separate from API code."""

    def __init__(self, **parameters: object) -> None:
        self.model = XGBClassifier(**parameters)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the classifier on leakage-safe training features."""
        self.model.fit(features, labels)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        """Return the positive-class probability."""
        return self.model.predict_proba(features)[:, 1]

    def save(self, path: Path) -> None:
        """Persist the fitted wrapper."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

