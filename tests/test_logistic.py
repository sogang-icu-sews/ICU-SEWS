"""Leakage, manifest, raw-data, and end-to-end checks for the baseline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from icu_sews.models.logistic import build_logistic_pipeline
from icu_sews.models.train_logistic import get_patient_split, load_records, run

SPLIT = {"train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15, "random_seed": 42}


def test_preprocessing_is_fitted_only_on_train() -> None:
    train = pd.DataFrame({"HR": [70.0, 80.0, np.nan, 90.0], "Temp": [np.nan] * 4})
    pipeline = build_logistic_pipeline()
    pipeline.fit(train, [0, 0, 1, 1])
    np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, [80, 0])
    mean_before = pipeline.named_steps["scaler"].mean_.copy()
    probabilities = pipeline.predict_proba(pd.DataFrame({"HR": [10000.0], "Temp": [45.0]}))
    np.testing.assert_array_equal(pipeline.named_steps["scaler"].mean_, mean_before)
    assert np.isfinite(probabilities).all()


def test_shared_split_reused_and_mismatch_rejected(tmp_path: Path) -> None:
    records = pd.DataFrame(
        {"patient_id": [f"A_p{i:03}" for i in range(40)], "SepsisLabel": [0, 1] * 20}
    )
    path = tmp_path / "split.parquet"
    first = get_patient_split(records, path, SPLIT)
    reused = get_patient_split(records, path, {**SPLIT, "random_seed": 99})
    pd.testing.assert_frame_equal(first, reused)
    assert first["patient_id"].is_unique
    assert set(first["split"]) == {"train", "validation", "test"}
    with pytest.raises(ValueError, match="differ"):
        get_patient_split(records.iloc[:-1], path, SPLIT)


@pytest.mark.parametrize("rows", ["1|80|2\n", "1|80|0\n1|90|1\n"])
def test_invalid_raw_labels_or_duplicate_time_rejected(tmp_path: Path, rows: str) -> None:
    folder = tmp_path / "A"
    folder.mkdir()
    (folder / "p001.psv").write_text("ICULOS|HR|SepsisLabel\n" + rows)
    with pytest.raises(ValueError):
        load_records(tmp_path, {"A": "A"}, ["HR"])


def test_end_to_end_persists_model_and_validation_only(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    folder = raw / "A"
    folder.mkdir(parents=True)
    for i in range(40):
        (folder / f"p{i:03}.psv").write_text(
            f"ICULOS|HR|SepsisLabel\n2|{80 + i}|{i % 2}\n1|NaN|0\n"
        )
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_dir": str(raw),
                    "sources": {"A": "A"},
                    "processed_dir": str(tmp_path / "processed"),
                },
                "split": SPLIT,
                "label": {"column": "SepsisLabel", "apply_additional_shift": False},
            }
        )
    )
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump({"features": ["HR"], "model": {}, "threshold": 0.5}))
    output = tmp_path / "run"
    metrics = run(data_config, model_config, output)
    assert 0 <= metrics["auroc"] <= 1
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["test_evaluated"] is False
    split = pd.read_parquet(output / "patient_split.parquet")
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    assert set(predictions["patient_id"]) == set(
        split.loc[split.split == "validation", "patient_id"]
    )
    assert not predictions.duplicated(["patient_id", "ICULOS"]).any()
    assert predictions["probability"].between(0, 1).all()
    pipeline = joblib.load(output / "logistic_pipeline.joblib")
    assert pipeline.feature_names_in_.tolist() == ["HR"]
    with pytest.raises(FileExistsError):
        run(data_config, model_config, output)
