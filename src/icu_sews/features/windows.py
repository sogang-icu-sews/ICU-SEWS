"""Causal 24-hour rolling-window feature generation.

Owner: 정상민
Scaffold policy: each window stays within one patient and uses current/past rows only.
"""

from collections.abc import Sequence

import pandas as pd


DEFAULT_STATISTICS = ("mean", "std", "min", "max", "median")


def create_rolling_statistics(
    patient: pd.DataFrame,
    value_columns: Sequence[str],
    observation_hours: int = 24,
) -> pd.DataFrame:
    """Create basic causal rolling statistics for one time-sorted patient."""
    if patient["patient_id"].nunique() != 1:
        raise ValueError("create_rolling_statistics accepts one patient at a time")

    ordered = patient.sort_values("ICULOS", kind="stable").copy()
    rolling = ordered[list(value_columns)].rolling(window=observation_hours, min_periods=1)
    result = ordered[["patient_id", "ICULOS", "SepsisLabel"]].copy()

    for statistic in DEFAULT_STATISTICS:
        values = getattr(rolling, statistic)()
        values.columns = [f"{column}__{statistic}__{observation_hours}h" for column in value_columns]
        result = pd.concat([result, values], axis=1)

    return result

