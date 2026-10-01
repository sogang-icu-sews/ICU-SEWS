"""Pydantic request and response contracts shared with the dashboard.

Owners: 김진호, 임경수
Unknown measurement fields are accepted so the schema can carry all Challenge columns.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HourlyMeasurement(BaseModel):
    """One hour of patient data."""

    model_config = ConfigDict(extra="allow")

    iculos: int = Field(alias="ICULOS", ge=1)


class PredictionRequest(BaseModel):
    """A single patient's current and past observations."""

    patient_id: str = Field(min_length=1, max_length=100)
    observations: list[HourlyMeasurement] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def validate_chronology(self) -> "PredictionRequest":
        """Reject duplicate or decreasing ICU hours."""
        hours = [row.iculos for row in self.observations]
        if hours != sorted(hours) or len(hours) != len(set(hours)):
            raise ValueError("ICULOS values must be unique and sorted")
        return self


class FeatureContribution(BaseModel):
    """One human-readable model contribution."""

    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    """Stable response consumed by the Streamlit dashboard."""

    patient_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str
    model_version: str
    is_mock: bool
    top_features: list[FeatureContribution] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service readiness metadata."""

    status: str
    model_loaded: bool
    mock_mode: bool

