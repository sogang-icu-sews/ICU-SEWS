"""Final pipeline: patient roles, preprocessing isolation, resume and test locking."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from icu_sews.models import final_gru as final
from icu_sews.models import train_gru as training


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "inputs"
    data, art = root / "data", root / "artifacts"
    data.mkdir(parents=True)
    art.mkdir()
    cfg = training.read_json(training.REPO / "artifacts/preprocess_config.json")
    registry = training.read_json(training.REPO / "artifacts/feature_registry.json")
    records, patients = [], []
    for fold in [0, 1, 2, 3, 4, -1]:
        for site in ["A", "B"]:
            for positive in [0, 1]:
                for _ in range(4):
                    pid = len(patients) + 1
                    split = "test" if fold == -1 else "dev"
                    patients.append([pid, site, positive, split, fold])
                    for hour in range(16):
                        records.append(
                            [
                                len(records),
                                pid,
                                hour + 1,
                                site,
                                split,
                                fold,
                                int(positive and hour >= 8),
                            ]
                        )
    em = pd.DataFrame(records, columns=final.KEYS)
    manifest = pd.DataFrame(patients, columns=["Patient_ID", "set", "sepsis", "split", "fold"])
    manifest.to_csv(art / "patient_split.csv", index=False)
    em.to_parquet(data / "eval_meta.parquet", index=False)
    rng = np.random.default_rng(7)
    signal = em.SepsisLabel.to_numpy()[:, None] * 4
    values = (rng.normal(size=(len(em), 24)) + signal).astype(np.float32)
    np.save(data / "seq_values.npy", values)
    np.save(data / "seq_row_start.npy", np.arange(len(em)) // 16 * 16)
    static = pd.DataFrame(np.zeros((len(em), 5)), columns=training.STATIC)
    static["Age"] = 50.0
    static.insert(0, "row_id", np.arange(len(em)))
    static.to_parquet(data / "gru_static.parquet", index=False)
    for name, value in [
        ("preprocess_config", cfg),
        ("feature_registry", registry),
        ("seq_meta", {"vars": cfg["model_prep"]["sequence"]["vars"], "row_order": "row_id"}),
    ]:
        (art / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    torch.set_num_threads(2)
    return root


def test_roles_and_preprocessing_exclude_all_holdouts(prepared):
    d = training.load_data(prepared)
    patients, rows = final.roles_for(d["em"])
    assert patients.Patient_ID.is_unique
    assert set(patients.loc[patients.role == "train", "fold"]) == {0, 1, 2, 3}
    assert set(patients.loc[patients.role.isin(["calibration", "threshold"]), "fold"]) == {4}
    assert set(patients.loc[patients.role == "test", "split"]) == {"test"}
    original = final.fit_preprocessing(d, rows["train"])
    changed = dict(d, values=np.array(d["values"]), static=d["static"].copy())
    heldout = np.concatenate([rows[r] for r in ["calibration", "threshold", "test"]])
    changed["values"][heldout] = 999999
    changed["static"].loc[heldout, "Age"] = 999999
    assert final.fit_preprocessing(changed, rows["train"]) == original


def test_final_fit_resume_test_isolation_and_reuse(prepared, tmp_path, monkeypatch):
    d = training.load_data(prepared)
    _, rows = final.roles_for(d["em"])
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    device = torch.device("cpu")
    predicted_rows = []
    real_predict = training.predict

    def record_predict(model, d, ids, *args):
        predicted_rows.append(set(ids))
        return real_predict(model, d, ids, *args)

    monkeypatch.setattr(training, "predict", record_predict)
    final.fit(d, continuous, device)
    assert predicted_rows == [set(rows["calibration"]), set(rows["threshold"])]
    assert not (continuous / "test_predictions.parquet").exists()
    policy = training.read_json(continuous / "threshold.json")
    tune = pd.read_parquet(continuous / "threshold_predictions.parquet")
    metrics = final.alarm_metrics(tune, tune.p_calibrated.to_numpy() >= policy["threshold"])
    assert metrics["nonsepsis_patient_false_alarm_rate"] <= 0.10

    real_save = training.atomic_checkpoint

    def interrupt(path, value):
        real_save(path, value)
        if Path(path).name == "latest.pt" and value["epoch"] == 1:
            raise RuntimeError("simulated disconnect")

    monkeypatch.setattr(training, "atomic_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        final.fit(d, resumed, device)
    monkeypatch.setattr(training, "atomic_checkpoint", real_save)
    final.fit(d, resumed, device, resume=True)
    a = torch.load(continuous / "final.pt", weights_only=False)["state_dict"]
    b = torch.load(resumed / "final.pt", weights_only=False)["state_dict"]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert training.read_json(continuous / "platt.json") == training.read_json(
        resumed / "platt.json"
    )
    assert training.read_json(continuous / "threshold.json") == training.read_json(
        resumed / "threshold.json"
    )

    monkeypatch.setenv("MPLBACKEND", "module://missing_colab_backend.backend_inline")
    summary = final.evaluate(d, continuous, device)
    assert summary["complete"] and summary["test_evaluated"]
    assert predicted_rows[-1] == set(rows["test"])
    predictions = pd.read_parquet(continuous / "test_predictions.parquet")
    assert set(predictions.row_id) == set(rows["test"])
    assert predictions.p_calibrated.between(0, 1).all()
    assert (continuous / "test_calibration.png").stat().st_size > 0
    calls = len(predicted_rows)
    assert final.evaluate(d, continuous, device) == summary
    final.fit(d, continuous, device, resume=True)
    assert len(predicted_rows) == calls
    (continuous / "threshold.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        final.evaluate(d, continuous, device)


def test_evaluate_requires_freeze_and_resume_rejects_changed_settings(prepared, tmp_path):
    d = training.load_data(prepared)
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(FileNotFoundError):
        final.evaluate(d, out, torch.device("cpu"))
    training.atomic_json(out / "run_identity.json", {"wrong": "settings"})
    with pytest.raises(ValueError, match="identical data"):
        final.fit(d, out, torch.device("cpu"), resume=True)
