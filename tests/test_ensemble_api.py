"""Release integration checks; set SEWS_TEST_ENSEMBLE_DIR for real weights."""

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.special import expit, logit

from icu_sews.api.ensemble_loader import load_ensemble_bundle
from icu_sews.api.ensemble_predictor import EnsemblePredictor
from icu_sews.api.gru_predictor import GruInputError
from icu_sews.api.main import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_dir():
    path = os.getenv("SEWS_TEST_ENSEMBLE_DIR")
    if not path:
        pytest.skip("Set SEWS_TEST_ENSEMBLE_DIR to run frozen model integration checks")
    return path


@pytest.fixture(scope="module")
def predictor(model_dir):
    return EnsemblePredictor(load_ensemble_bundle(model_dir, ROOT / "artifacts"))


def records(n):
    return [
        {"ICULOS": i + 1, "HR": 75 + i, "Resp": 18, "SBP": 120, "Age": 60, "Gender": 1}
        for i in range(n)
    ]


@pytest.mark.parametrize("length", [1, 6, 24])
def test_real_prediction_and_calibration(predictor, length):
    result = predictor.predict(records(length))
    components = result["component_probabilities"]
    raw = expit(
        sum(
            predictor.bundle["weights"][k] * logit(np.clip(components[k], 1e-7, 1 - 1e-7))
            for k in ("lgbm", "lstm", "gru")
        )
    )
    cal = predictor.bundle["platt"]
    expected = expit(cal["slope"] * logit(np.clip(raw, 1e-7, 1 - 1e-7)) + cal["intercept"])
    assert result["risk_score"] == pytest.approx(expected, abs=1e-12)
    assert result["alarm"] == (expected >= result["threshold"])
    left, packed, _, lengths, _ = predictor.prepare(records(length))
    np.testing.assert_array_equal(packed[:, :length], left[:, -length:])
    assert np.count_nonzero(packed[:, length:]) == 0
    assert lengths.tolist() == [length]


@pytest.mark.parametrize("mode", ["lstm", "ensemble_equal"])
def test_other_modes(model_dir, mode):
    predictor = EnsemblePredictor(load_ensemble_bundle(model_dir, ROOT / "artifacts", mode))
    result = predictor.predict([{"ICULOS": 1}])
    assert 0 <= result["risk_score"] <= 1
    assert result["is_mock"] is False


@pytest.mark.parametrize("value", [True, "bad", float("inf"), float("nan")])
def test_reject_invalid_extra_measurement(predictor, value):
    with pytest.raises(GruInputError):
        predictor.predict([{"ICULOS": 1, "HR": value}])


def test_reject_nonconsecutive_hours(predictor):
    with pytest.raises(GruInputError):
        predictor.predict([{"ICULOS": 1}, {"ICULOS": 3}])


def test_reject_modified_frozen_file(model_dir, tmp_path):
    (tmp_path / "frozen.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_ensemble_bundle(tmp_path, ROOT / "artifacts")


def test_real_api(model_dir, monkeypatch):
    monkeypatch.setenv("SEWS_USE_MOCK_MODEL", "false")
    monkeypatch.setenv("SEWS_MODEL_TYPE", "ensemble")
    monkeypatch.setenv("SEWS_MODEL_DIR", model_dir)
    monkeypatch.setenv("SEWS_PREPROCESS_DIR", str(ROOT / "artifacts"))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        response = client.post(
            "/predict", json={"patient_id": "synthetic", "observations": records(6)}
        )
        assert response.status_code == 200
        assert response.json()["is_mock"] is False
        bad = client.post(
            "/predict",
            json={"patient_id": "synthetic", "observations": [{"ICULOS": 1, "HR": "bad"}]},
        )
        assert bad.status_code == 422


def test_missing_ensemble_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("SEWS_USE_MOCK_MODEL", "false")
    monkeypatch.setenv("SEWS_MODEL_TYPE", "ensemble")
    monkeypatch.setenv("SEWS_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SEWS_PREPROCESS_DIR", str(ROOT / "artifacts"))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        response = client.post(
            "/predict", json={"patient_id": "synthetic", "observations": records(1)}
        )
        assert response.status_code == 503
