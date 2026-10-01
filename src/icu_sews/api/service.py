"""Model lifecycle and prediction orchestration.

Owner: 김진호
The deterministic mock exists only so API/UI integration can begin before training.
"""

import os

from icu_sews.api.schemas import FeatureContribution, PredictionRequest, PredictionResponse


class PredictorService:
    """Load production artifacts once and expose a stable prediction method."""

    def __init__(self) -> None:
        self.mock_mode = os.getenv("SEWS_USE_MOCK_MODEL", "false").lower() == "true"
        self.model_loaded = self.mock_mode
        # TODO(김진호/이택훈): load preprocessor, model, calibrator, thresholds here.

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Return a mock integration result until trained artifacts are connected."""
        if not self.mock_mode:
            raise RuntimeError("Trained model artifacts are not connected")

        last = request.observations[-1].model_dump(by_alias=True)
        respiratory_rate = float(last.get("Resp", 18.0) or 18.0)
        systolic_pressure = float(last.get("SBP", 120.0) or 120.0)
        risk = min(0.99, max(0.01, 0.15 + max(0.0, respiratory_rate - 20) * 0.03))
        risk += max(0.0, 105 - systolic_pressure) * 0.005
        risk = min(risk, 0.99)

        if risk >= 0.75:
            level = "alert"
        elif risk >= 0.50:
            level = "warning"
        else:
            level = "normal"

        return PredictionResponse(
            patient_id=request.patient_id,
            risk_score=risk,
            risk_level=level,
            model_version="mock-scaffold-v1",
            is_mock=True,
            top_features=[
                FeatureContribution(feature="Resp_demo", shap_value=0.0),
                FeatureContribution(feature="SBP_demo", shap_value=0.0),
            ],
        )
