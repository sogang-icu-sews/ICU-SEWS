"""Clinical screening scores that can be computed from available inputs.

Owner: 정상민
The full SOFA score is intentionally not fabricated because Challenge 2019
does not expose every required component. Missing inputs remain explicit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QsofaResult:
    """qSOFA total and component availability."""

    total: int
    respiratory: int | None
    systolic_blood_pressure: int | None
    altered_mental_status: int | None


def calculate_qsofa(
    respiratory_rate: float | None,
    systolic_blood_pressure: float | None,
    altered_mental_status: bool | None,
) -> QsofaResult:
    """Calculate qSOFA without silently treating missing values as normal."""
    respiratory = None if respiratory_rate is None else int(respiratory_rate >= 22)
    pressure = (
        None if systolic_blood_pressure is None else int(systolic_blood_pressure <= 100)
    )
    mental_status = None if altered_mental_status is None else int(altered_mental_status)
    observed = [value for value in (respiratory, pressure, mental_status) if value is not None]
    return QsofaResult(
        total=sum(observed),
        respiratory=respiratory,
        systolic_blood_pressure=pressure,
        altered_mental_status=mental_status,
    )

