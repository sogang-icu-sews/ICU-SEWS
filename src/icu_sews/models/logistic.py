"""Current-hour logistic baseline. Owner: 이택훈."""

from collections.abc import Mapping

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_pipeline(parameters: Mapping[str, object] | None = None) -> Pipeline:
    """Fit imputation/scaling only when the training pipeline is fitted.

    No forward fill, rolling features, missingness indicators, or label shifts
    are used in this deliberately small baseline. Empty training columns remain
    present and are filled with zero by SimpleImputer.
    """
    options = {"max_iter": 1000, "solver": "lbfgs", "random_state": 42}
    options.update(parameters or {})
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**options)),
        ]
    )
