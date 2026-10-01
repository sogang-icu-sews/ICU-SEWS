"""Leakage-safe patient-level dataset splitting.

Owner: 정상민
The generated patient IDs must be reused by every model and evaluation run.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PatientSplit:
    """Immutable patient identifiers for the three holdout groups."""

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def split_patient_ids(
    patient_summary: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> PatientSplit:
    """Stratify one-row-per-patient data without splitting a patient across sets."""
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    if patient_summary["patient_id"].duplicated().any():
        raise ValueError("patient_summary must contain exactly one row per patient")

    train, remainder = train_test_split(
        patient_summary,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=patient_summary["is_sepsis"],
    )
    relative_validation = validation_ratio / (validation_ratio + test_ratio)
    validation, test = train_test_split(
        remainder,
        train_size=relative_validation,
        random_state=random_seed,
        stratify=remainder["is_sepsis"],
    )

    result = PatientSplit(
        train_ids=tuple(sorted(train["patient_id"])),
        validation_ids=tuple(sorted(validation["patient_id"])),
        test_ids=tuple(sorted(test["patient_id"])),
    )
    _assert_disjoint(result)
    return result


def _assert_disjoint(split: PatientSplit) -> None:
    train_ids = set(split.train_ids)
    validation_ids = set(split.validation_ids)
    test_ids = set(split.test_ids)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("A patient appears in more than one split")

