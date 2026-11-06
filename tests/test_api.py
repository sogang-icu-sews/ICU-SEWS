"""API contract smoke tests. Owner: 김진호."""

import os

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
