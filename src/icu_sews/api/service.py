"""Model lifecycle and prediction orchestration.

Owner: 김진호
The deterministic mock exists only so API/UI integration can begin before training.
"""

import logging
import math
import os

import pandas as pd

from icu_sews.api.model_loader import LoadedLogisticModel, load_logistic_model
from icu_sews.api.schemas import FeatureContribution, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class InvalidModelInputError(ValueError):
    """모델에 전달할 측정값이 올바르지 않을 때 발생하는 예외."""

    def __init__(self, field: str, observation_index: int) -> None:
        super().__init__(f"{field} must be a finite number or null")
        self.field = field
        self.observation_index = observation_index


class PredictorService:
    """Load production artifacts once and expose a stable prediction method."""

    def __init__(self) -> None:
        self.mock_mode = os.getenv("SEWS_USE_MOCK_MODEL", "false").lower() == "true"
        self.loaded_model: LoadedLogisticModel | None = None
        self.model_loaded = False
        self.prediction_ready = False

        # Mock 모드는 실제 모델 파일 없이 사용할 수 있다.
        if self.mock_mode:
            self.model_loaded = True
            self.prediction_ready = True
            return

        # 실제 모델 파일이 들어 있는 폴더 경로를 설정에서 읽는다.
        model_dir = os.getenv("SEWS_MODEL_DIR", "").strip()

        if not model_dir:
            logger.warning("Model loading skipped: SEWS_MODEL_DIR is not set")
            return

        # 파일이 없거나 손상되어도 상태 확인용 API는 실행되도록 한다.
        try:
            loaded_model = load_logistic_model(model_dir)
        except Exception:
            logger.exception("Failed to load the logistic model")
            return

        self.loaded_model = loaded_model
        self.model_loaded = True
        self.prediction_ready = True

        logger.info("Logistic model loaded and prediction is ready")

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """설정에 따라 실제 모델 또는 Mock으로 예측 결과를 반환한다."""
        if not self.mock_mode:
            if not self.prediction_ready or self.loaded_model is None:
                raise RuntimeError("Prediction unavailable: model is not loaded")

            return self._predict_real(request)

        last = request.observations[-1].model_dump(by_alias=True)
        resp_value = last.get("Resp")
        sbp_value = last.get("SBP")

        respiratory_rate = 18.0 if resp_value is None else float(resp_value)
        systolic_pressure = 120.0 if sbp_value is None else float(sbp_value)
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

    def _predict_real(self, request: PredictionRequest) -> PredictionResponse:
        """마지막 관측값을 학습된 모델에 전달하고 예측 결과를 반환한다."""
        model = self.loaded_model
        if model is None:
            raise RuntimeError("Prediction unavailable: model is not loaded")

        observation_index = len(request.observations) - 1
        latest = request.observations[-1].model_dump(by_alias=True)

        values: list[float] = []

        # 모델을 학습할 때 사용한 입력 항목 순서대로 값을 정리한다.
        for feature in model.features:
            value = latest.get(feature)

            # 생략하거나 null로 보낸 값은 모델의 결측값 처리기에 맡긴다.
            if value is None:
                values.append(float("nan"))
                continue

            # 이 시점에 불리언인 값은 숫자 1이나 0으로 변환하지 않는다.
            if isinstance(value, bool):
                raise InvalidModelInputError(feature, observation_index)

            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise InvalidModelInputError(feature, observation_index) from error

            # 사용자가 직접 보낸 NaN이나 무한대는 거부한다.
            if not math.isfinite(numeric_value):
                raise InvalidModelInputError(feature, observation_index)

            values.append(numeric_value)

        # 한 행은 마지막 관측 시점을 나타낸다.
        # 열 이름과 순서는 모델을 학습할 때 사용한 것과 일치시킨다.
        frame = pd.DataFrame(
            [values],
            columns=model.features,
            dtype="float64",
        )

        try:
            probabilities = model.pipeline.predict_proba(frame)

            # 한 행에 대해 클래스 0과 1의 확률이 반환돼야 한다.
            if probabilities.shape != (1, 2):
                raise ValueError("Unexpected probability output shape")

            negative_score = float(probabilities[0, 0])
            risk_score = float(probabilities[0, 1])

            # 두 확률 모두 유한한 값이고 0~1 범위인지 확인한다.
            if any(
                not math.isfinite(score) or not 0.0 <= score <= 1.0
                for score in (negative_score, risk_score)
            ):
                raise ValueError("Model returned invalid probabilities")

            # 두 클래스의 확률 합이 약 1인지 확인한다.
            if not math.isclose(
                negative_score + risk_score,
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError("Model probabilities do not sum to one")

        except Exception as error:
            # 상세 오류는 서버 로그에 남긴다.
            logger.exception("Logistic model prediction failed")

            # API에는 예측이 실패했다는 메시지를 전달한다.
            raise RuntimeError("Prediction unavailable: model prediction failed") from error

        return PredictionResponse(
            patient_id=request.patient_id,
            risk_score=risk_score,
            risk_level="unclassified",
            model_version="logistic-baseline",
            is_mock=False,
            top_features=[],
        )
