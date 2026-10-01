"""Tests for clinical score missingness and threshold behavior. Owner: 정상민."""

from icu_sews.features.clinical_scores import calculate_qsofa


def test_qsofa_thresholds() -> None:
    result = calculate_qsofa(
        respiratory_rate=22,
        systolic_blood_pressure=100,
        altered_mental_status=True,
    )
    assert result.total == 3


def test_qsofa_does_not_treat_missing_as_normal() -> None:
    result = calculate_qsofa(None, 120, None)
    assert result.total == 0
    assert result.respiratory is None
    assert result.altered_mental_status is None

