import math
import os
from pathlib import Path

from fastapi.testclient import TestClient

from icu_sews.api.gru_loader import load_gru_bundle
from icu_sews.api.gru_predictor import GruPredictor
from icu_sews.api.main import app

model_dir = Path.home() / "Projects" / "SEWS-models" / "gru_v2_platt_final"

os.environ["SEWS_USE_MOCK_MODEL"] = "false"
os.environ["SEWS_MODEL_TYPE"] = "gru"
os.environ["SEWS_MODEL_DIR"] = str(model_dir)

observations = [
    {
        "ICULOS": hour,
        "HR": 85.0,
        "O2Sat": 97.0,
        "Temp": 37.0,
        "SBP": 120.0,
        "MAP": 80.0,
        "Resp": 18.0,
        "Age": 65.0,
        "Gender": 1,
        "Unit1": 1,
        "Unit2": 0,
    }
    for hour in range(1, 25)
]

direct = GruPredictor(load_gru_bundle(str(model_dir))).predict(observations)

with TestClient(app) as client:
    health = client.get("/health")
    assert health.status_code == 200, health.text
    assert health.json() == {
        "status": "ok",
        "model_loaded": True,
        "mock_mode": False,
    }
    print("Health:", health.json())

    response = client.post(
        "/predict",
        json={"patient_id": "TEST-GRU", "observations": observations},
    )
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["patient_id"] == "TEST-GRU"
    assert result["is_mock"] is False
    assert math.isclose(
        result["risk_score"], direct["risk_score"], abs_tol=1e-7
    )
    assert result["alarm"] == direct["alarm"]
    assert result["threshold"] == direct["threshold"]
    assert result["model_version"] == direct["model_version"]
    assert result["risk_level"] == (
        "alert" if direct["alarm"] else "below_threshold"
    )
    print("Prediction:", result)
    print("API/direct prediction match: OK")

    invalid = client.post(
        "/predict",
        json={
            "patient_id": "TEST-INVALID",
            "observations": [{"ICULOS": 1, "HR": "hello"}],
        },
    )
    assert invalid.status_code == 422, invalid.text
    assert invalid.json()["detail"][0]["loc"] == [
        "body", "observations", 0, "HR"
    ]
    print("Invalid HR rejected: OK")

    gap = client.post(
        "/predict",
        json={
            "patient_id": "TEST-GAP",
            "observations": [{"ICULOS": 1}, {"ICULOS": 3}],
        },
    )
    assert gap.status_code == 422, gap.text
    print("Time gap rejected: OK")

# 모델 폴더가 없으면 준비 상태가 degraded이고 예측은 503이어야 한다.
os.environ["SEWS_MODEL_DIR"] = str(model_dir / "__missing_test_directory__")

with TestClient(app) as client:
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["model_loaded"] is False

    response = client.post(
        "/predict",
        json={"patient_id": "TEST-MISSING", "observations": observations},
    )
    assert response.status_code == 503, response.text
    print("Missing model returns 503: OK")

print("GRU API integration check: OK")