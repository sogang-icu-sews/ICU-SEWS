"""실제 모델 파일을 사용하는 API 연결 경로를 검증한다."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from icu_sews.api.main import app


@pytest.fixture
def model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Pipeline:
    """가상 데이터로 작은 모델을 만들고 임시 폴더에 저장한다."""
    features = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp", "Age"]

    # 연결 검사용 가상 데이터이며 임상적 의미는 없다.
    training_data = pd.DataFrame(
        [
            [70, 98, 36.5, 120, 80, 93, 16, 30],
            [80, 97, 36.8, 115, 75, 88, 18, 40],
            [90, 96, 37.0, 110, 70, 83, 20, 50],
            [110, 94, 38.0, 95, 60, 72, 25, 60],
            [120, 92, 38.5, 85, 55, 65, 30, 70],
            [130, 90, 39.0, 80, 50, 60, 35, 80],
        ],
        columns=features,
        dtype="float64",
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(training_data, [0, 0, 0, 1, 1, 1])

    joblib.dump(pipeline, tmp_path / "logistic_pipeline.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model_config": {"features": features}}),
        encoding="utf-8",
    )

    # 이 테스트 동안만 실제 모델 모드와 임시 경로를 적용한다.
    monkeypatch.setenv("SEWS_USE_MOCK_MODEL", "false")
    monkeypatch.setenv("SEWS_MODEL_DIR", str(tmp_path))

    return pipeline


def test_real_prediction_matches_pipeline(model_artifact: Pipeline) -> None:
    """API 결과가 마지막 관측값으로 직접 계산한 모델 결과와 일치한다."""
    # 마지막 관측값의 입력을 모델의 열 순서대로 직접 구성한다.
    # Temp는 null, DBP와 MAP은 생략한 상황이다.
    expected_input = pd.DataFrame(
        [[105, 95, float("nan"), 95, float("nan"), float("nan"), 23, 65]],
        columns=["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp", "Age"],
        dtype="float64",
    )
    expected_score = float(model_artifact.predict_proba(expected_input)[0, 1])

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model_loaded": True,
            "mock_mode": False,
        }

        response = client.post(
            "/predict",
            json={
                "patient_id": "INTEGRATION-001",
                "observations": [
                    {
                        "ICULOS": 1,
                        "HR": 70,
                        "O2Sat": 98,
                        "Temp": 36.5,
                        "SBP": 120,
                        "DBP": 80,
                        "MAP": 93,
                        "Resp": 16,
                        "Age": 65,
                    },
                    {
                        # 입력 순서를 모델의 열 순서와 다르게 보낸다.
                        "ICULOS": 2,
                        "Age": 65,
                        "Resp": "23",
                        "SBP": 95,
                        "HR": "105",
                        "O2Sat": 95,
                        "Temp": None,
                    },
                ],
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["patient_id"] == "INTEGRATION-001"
    assert result["risk_score"] == pytest.approx(expected_score)
    assert result["risk_level"] == "unclassified"
    assert result["model_version"] == "logistic-baseline"
    assert result["is_mock"] is False
    assert result["top_features"] == []


def test_real_prediction_rejects_invalid_hr(model_artifact: Pipeline) -> None:
    """모델 입력에 사용할 HR이 숫자가 아니면 위치와 함께 422를 반환한다."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patient_id": "INTEGRATION-002",
                "observations": [{"ICULOS": 1, "HR": "hello"}],
            },
        )

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(
        error["loc"] == ["body", "observations", 0, "HR"]
        and error["type"] == "invalid_model_measurement"
        for error in errors
    )


def test_missing_model_returns_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모델 파일이 없으면 상태 확인은 가능하고 예측에는 503을 반환한다."""
    monkeypatch.setenv("SEWS_USE_MOCK_MODEL", "false")
    monkeypatch.setenv("SEWS_MODEL_DIR", str(tmp_path / "missing"))

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "degraded",
            "model_loaded": False,
            "mock_mode": False,
        }

        response = client.post(
            "/predict",
            json={
                "patient_id": "INTEGRATION-003",
                "observations": [{"ICULOS": 1, "Resp": 23, "SBP": 95}],
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == ("Prediction unavailable: model is not loaded")


def test_prediction_failure_returns_503(
    model_artifact: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모델 실행 중 오류가 발생하면 내부 오류 대신 정해진 메시지를 반환한다."""

    def fail_prediction(frame: pd.DataFrame) -> None:
        raise ValueError("internal-test-error")

    with TestClient(app) as client:
        loaded_model = app.state.predictor.loaded_model
        assert loaded_model is not None

        monkeypatch.setattr(
            loaded_model.pipeline,
            "predict_proba",
            fail_prediction,
        )

        response = client.post(
            "/predict",
            json={
                "patient_id": "INTEGRATION-004",
                "observations": [{"ICULOS": 1, "Resp": 23, "SBP": 95}],
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == ("Prediction unavailable: model prediction failed")
    assert "internal-test-error" not in response.text
