"""Exact agreement with physical whole-patient resampling, including duplicates."""

import json

import numpy as np
import pandas as pd
import pytest

from icu_sews.evaluation.bootstrap_gru import PatientMetrics, bootstrap, load_predictions, main
from icu_sews.evaluation.compare_gru import alarm_metrics, probability_metrics, sha256


@pytest.fixture
def predictions():
    records = []
    rng = np.random.default_rng(17)
    for pid in range(1, 9):
        positive = pid % 2 == 0
        for h in range(16 + pid):
            y = int(positive and (h >= 12 or pid == 8))
            # Ties and boundary probabilities exercise rank formulas and log clipping.
            p = rng.choice([0.0, 0.02, 0.1, 0.3, 0.6, 1.0])
            records.append(
                [len(records), pid, h + 1, "A" if pid <= 4 else "B", "test", -1, y, p, p >= 0.3]
            )
    return pd.DataFrame(
        records,
        columns=[
            "row_id",
            "Patient_ID",
            "ICULOS",
            "set",
            "split",
            "fold",
            "SepsisLabel",
            "p_calibrated",
            "alarm",
        ],
    )


def duplicate_patients(frame, patient_ids):
    chunks = []
    for draw, pid in enumerate(patient_ids):
        group = frame[frame.Patient_ID == pid].copy()
        group["Patient_ID"] = draw + 1  # Repeated copies must not be merged by groupby.
        chunks.append(group)
    return pd.concat(chunks, ignore_index=True)


def test_weighted_metrics_match_explicit_patient_copies(predictions):
    cache = PatientMetrics(predictions, 0.12)
    draws = [1, 1, 1, 2, 3, 4, 6, 8]
    counts = np.array([draws.count(pid) for pid in cache.patients])
    observed = cache.evaluate(counts)
    repeated = duplicate_patients(predictions, draws)
    expected = probability_metrics(
        repeated.SepsisLabel.to_numpy(),
        repeated.p_calibrated.to_numpy(),
        np.full(len(repeated), 0.12),
    )
    expected.update(alarm_metrics(repeated, repeated.alarm.to_numpy()))
    for name, value in observed.items():
        assert value == pytest.approx(expected[name], nan_ok=True), name


def test_one_class_and_empty_draw_metrics_are_undefined(predictions):
    cache = PatientMetrics(predictions, 0.12)
    result = cache.evaluate(np.array([8, 0, 0, 0, 0, 0, 0, 0]))
    assert np.isnan(result["AUROC"]) and np.isnan(result["AUPRC"])
    assert np.isnan(result["early_recall_6to12h"])
    assert all(np.isnan(v) for v in cache.evaluate(np.zeros(cache.n, dtype=int)).values())


def test_bootstrap_is_deterministic_and_percentiles_match(predictions):
    ci, samples = bootstrap(predictions, 0.12, repetitions=40, seed=5)
    ci2, samples2 = bootstrap(predictions, 0.12, repetitions=40, seed=5)
    pd.testing.assert_frame_equal(ci, ci2)
    pd.testing.assert_frame_equal(samples, samples2)
    for row in ci.itertuples(index=False):
        values = samples.loc[samples.scope == row.scope, row.metric].dropna()
        np.testing.assert_allclose(
            [row.ci_lower, row.ci_upper], np.quantile(values, [0.025, 0.975])
        )
        assert row.valid_replicates == len(values)
    # Confirm the first replicate is exactly a patient draw, not an hourly-row draw.
    ids = sorted(predictions.Patient_ID.unique())
    rng = np.random.default_rng(5)
    draws = np.array(ids)[rng.integers(len(ids), size=len(ids))]
    repeated = duplicate_patients(predictions, draws)
    expected = alarm_metrics(repeated, repeated.alarm.to_numpy())
    first = samples[(samples.replicate == 0) & (samples.scope == "all")].iloc[0]
    assert first.Utility == pytest.approx(expected["Utility"])
    assert first.nonsepsis_patient_false_alarm_rate == pytest.approx(
        expected["nonsepsis_patient_false_alarm_rate"]
    )


def test_prediction_integrity_and_frozen_threshold(predictions, tmp_path):
    path = tmp_path / "test_predictions.parquet"
    predictions.to_parquet(path, index=False)
    (tmp_path / "platt.json").write_text(json.dumps({"baseline_probability": 0.12}))
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.3}))
    frozen = {
        "artifacts": {name: sha256(tmp_path / name) for name in ["platt.json", "threshold.json"]}
    }
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    summary = {
        "complete": True,
        "test_evaluated": True,
        "rows": len(predictions),
        "patients": 8,
        "frozen_sha256": sha256(tmp_path / "frozen.json"),
        "artifacts": {"test_predictions.parquet": sha256(path)},
    }
    scopes = {
        "all": predictions,
        "set_A": predictions[predictions["set"] == "A"],
        "set_B": predictions[predictions["set"] == "B"],
        "ICULOS_le60": predictions[predictions.ICULOS <= 60],
    }
    summary["probability"] = [
        {
            "scope": scope,
            "method": "platt",
            **probability_metrics(
                f.SepsisLabel.to_numpy(), f.p_calibrated.to_numpy(), np.full(len(f), 0.12)
            ),
        }
        for scope, f in scopes.items()
    ]
    summary["alarms"] = [
        {"scope": scope, **alarm_metrics(f, f.alarm.to_numpy())}
        for scope, f in scopes.items()
        if scope != "ICULOS_le60"
    ]
    (tmp_path / "test_summary.json").write_text(json.dumps(summary))
    manifest = predictions.groupby("Patient_ID").agg(
        set=("set", "first"), split=("split", "first"), sepsis=("SepsisLabel", "max")
    )
    manifest.to_csv(tmp_path / "manifest.csv")
    loaded, baseline, threshold = load_predictions(tmp_path, tmp_path / "manifest.csv")
    assert len(loaded) == len(predictions) and baseline == 0.12 and threshold == 0.3
    out = tmp_path / "bootstrap"
    main(
        [
            "--run",
            str(tmp_path),
            "--out",
            str(out),
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--repetitions",
            "20",
        ]
    )
    assert json.loads((out / "metadata.json").read_text())["complete"]
    ci = pd.read_csv(out / "bootstrap_95ci.csv")
    assert (ci["requested_replicates"] == 20).all()
    assert (ci["valid_replicates"] <= 20).all()
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.4}))
    with pytest.raises(ValueError, match="Frozen file changed"):
        load_predictions(tmp_path, tmp_path / "manifest.csv")
