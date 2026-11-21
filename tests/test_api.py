"""API contract smoke tests. Owner: 김진호."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ["SEWS_USE_MOCK_MODEL"] = "true"

from icu_sews.api.main import app  # noqa: E402


def test_health_and_predict() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["mock_mode"] is True

        response = client.post(
            "/predict",
            json={
                "patient_id": "A_p000001",
                "observations": [{"ICULOS": 1, "Resp": 23, "SBP": 95}],
            },
        )
        assert response.status_code == 200
        assert response.json()["is_mock"] is True


# Resp에 잘못된 값이 들어갈 경우
def test_predict_rejects_invalid_resp() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-001",
                "observations": [{"ICULOS": 1, "Resp": "hello", "SBP": 95}],
            },
        )

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(error["loc"][-1] == "Resp" for error in errors)


# SBP에 잘못된 값이 들어갈 경우
def test_predict_rejects_invalid_sbp() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-001",
                "observations": [{"ICULOS": 1, "Resp": 23, "SBP": "hello"}],
            },
        )

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(error["loc"][-1] == "SBP" for error in errors)


# Resp와 SBP를 생략하거나 null로 보내도 요청을 허용하는지 확인
@pytest.mark.parametrize(
    "observation",
    [
        {"ICULOS": 1},
        {"ICULOS": 1, "Resp": None, "SBP": None},
    ],
    ids=["omitted", "null"],
)
def test_predict_accepts_missing_measurements(observation: dict) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-001",
                "observations": [observation],
            },
        )

        assert response.status_code == 200
        assert response.json()["patient_id"] == "TEST-001"
        assert response.json()["is_mock"] is True


# 숫자 문자열을 보냈을 때 숫자를 보낸 경우와 같은 결과인지 확인
def test_predict_accepts_numeric_strings() -> None:
    with TestClient(app) as client:
        numeric_response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-001",
                "observations": [{"ICULOS": 1, "Resp": 23, "SBP": 95}],
            },
        )
        string_response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-001",
                "observations": [{"ICULOS": 1, "Resp": "23", "SBP": "95"}],
            },
        )

        assert numeric_response.status_code == 200
        assert string_response.status_code == 200
        assert string_response.json()["is_mock"] is True
        assert string_response.json()["risk_score"] == numeric_response.json()["risk_score"]
        assert string_response.json()["risk_level"] == numeric_response.json()["risk_level"]


# NaN과 양수·음수 무한대를 나타내는 문자열을 거부하는지 확인
@pytest.mark.parametrize("field", ["Resp", "SBP"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite_strings(field: str, value: str) -> None:
    observation = {"ICULOS": 1, "Resp": 23, "SBP": 95}
    observation[field] = value

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-001",
                "observations": [observation],
            },
        )

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(
            error["loc"] == ["body", "observations", 0, field] and error["type"] == "finite_number"
            for error in errors
        )


# 입력한 0을 결측값으로 취급해 기본값으로 바꾸지 않는지 확인
def test_predict_preserves_zero_sbp() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-ZERO-SBP",
                "observations": [{"ICULOS": 1, "Resp": 23, "SBP": 0}],
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["is_mock"] is True
        assert result["risk_score"] == pytest.approx(0.765)
        assert result["risk_level"] == "alert"


# ICULOS가 없거나 1보다 작으면 요청을 거부하는지 확인
@pytest.mark.parametrize(
    "observation",
    [
        {"Resp": 23, "SBP": 95},
        {"ICULOS": 0, "Resp": 23, "SBP": 95},
        {"ICULOS": -1, "Resp": 23, "SBP": 95},
    ],
    ids=["missing", "zero", "negative"],
)
def test_predict_rejects_invalid_iculos(observation: dict) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-001",
                "observations": [observation],
            },
        )

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(error["loc"] == ["body", "observations", 0, "ICULOS"] for error in errors)


# 기록의 시간이 중복되거나 역순이면 요청을 거부하는지 확인
@pytest.mark.parametrize(
    "hours",
    [
        [1, 1],
        [2, 1],
    ],
    ids=["duplicate", "decreasing"],
)
def test_predict_rejects_invalid_chronology(hours: list[int]) -> None:
    observations = [{"ICULOS": hour, "Resp": 23, "SBP": 95} for hour in hours]

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "patient_id": "TEST-001",
                "observations": observations,
            },
        )

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(
            error["loc"] == ["body"]
            and error["type"] == "value_error"
            and "ICULOS values must be unique and sorted" in error["msg"]
            for error in errors
        )
