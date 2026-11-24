"""Compare GRU API results with the team's saved predictions."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from icu_sews.api.main import app
from icu_sews.models.train_gru import digest, load_data

ROOT = Path.home() / "Downloads" / "SEWS_GRU-20261121T092957Z-1-001" / "SEWS_GRU"
REFERENCE_DIR = ROOT / "final" / "gru_v2_platt_final"
MODEL_DIR = Path.home() / "Projects" / "SEWS-models" / "gru_v2_platt_final"
TOLERANCE = 1e-6


def main():
    print("Loading reference files...")

    frozen = json.loads((REFERENCE_DIR / "frozen.json").read_text(encoding="utf-8"))

    # 현재 서버가 사용하는 모델이 기준 결과를 만든 모델과 같은지 검사.
    for name in ["final.pt", "preprocessing.json", "platt.json", "threshold.json"]:
        expected_hash = frozen["artifacts"][name]
        assert digest(MODEL_DIR / name) == expected_hash, f"Model artifact mismatch: {name}"

    # 팀 코드로 데이터 정렬, 환자 경계, 전처리 코드 일치 여부 검사.
    data = load_data(ROOT / "prepared")

    # Windows 경로 구분자를 정규화한 뒤 데이터 해시 비교.
    actual_hashes = {name.replace("\\", "/"): value for name, value in data["hashes"].items()}
    for name, expected_hash in frozen["identity"]["data"].items():
        assert actual_hashes.get(name) == expected_hash, f"Prepared data mismatch: {name}"

    print("Model and prepared data identity: OK")

    reference = pd.read_parquet(REFERENCE_DIR / "calibration_predictions.parquet")
    meta = data["em"]
    values = data["values"]
    starts = data["starts"]
    static = data["static"]
    variables = data["cfg"]["model_prep"]["sequence"]["vars"]

    assert not reference.empty
    assert reference["row_id"].is_unique
    assert np.isfinite(reference["p_calibrated"]).all()
    assert reference["p_calibrated"].between(0, 1).all()

    ids = reference["row_id"].to_numpy(dtype=np.int64)
    assert ((ids >= 0) & (ids < len(meta))).all()

    # 基準の行が同じ患者・同じ時点を指していることを確認。
    for column in ["Patient_ID", "ICULOS"]:
        assert np.array_equal(
            reference[column].to_numpy(),
            meta.iloc[ids][column].to_numpy(),
        ), f"Reference alignment mismatch: {column}"

    threshold = float(
        json.loads((MODEL_DIR / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    )

    reference = reference.copy()
    reference["window_length"] = np.minimum(ids - starts[ids] + 1, 24)
    reference["distance"] = abs(reference["p_calibrated"] - threshold)

    # 短い記録・24時間記録・閾値付近・ランダム標本を含める。
    selected = pd.concat(
        [
            reference.groupby("window_length", sort=True).head(1),
            reference.nsmallest(16, "distance"),
            reference[reference["p_calibrated"] >= threshold].head(8),
            reference[reference["p_calibrated"] < threshold].head(8),
            reference.sample(n=min(32, len(reference)), random_state=42),
        ]
    ).drop_duplicates("row_id")

    assert (selected["p_calibrated"] >= threshold).any()
    assert (selected["p_calibrated"] < threshold).any()

    print("Selected cases:", len(selected))
    print(
        "Window lengths:",
        sorted(selected["window_length"].unique().tolist()),
    )

    os.environ["SEWS_USE_MOCK_MODEL"] = "false"
    os.environ["SEWS_MODEL_TYPE"] = "gru"
    os.environ["SEWS_MODEL_DIR"] = str(MODEL_DIR)

    results = []

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["model_loaded"] is True
        assert health.json()["mock_mode"] is False

        for number, row in enumerate(selected.itertuples(index=False), start=1):
            row_id = int(row.row_id)
            start = max(int(starts[row_id]), row_id - 23)
            observations = []

            for index in range(start, row_id + 1):
                observation = {
                    name: None if pd.isna(value) else float(value)
                    for name, value in zip(variables, values[index], strict=True)
                }

                patient_static = static.iloc[index]
                for name in ["Age", "Gender"]:
                    value = patient_static[name]
                    observation[name] = None if pd.isna(value) else float(value)

                # 保存済みの病棟one-hotから、APIの元の入力形式に戻す。
                observation["Unit1"] = int(patient_static["Unit_MICU"] == 1)
                observation["Unit2"] = int(patient_static["Unit_SICU"] == 1)
                observation["ICULOS"] = int(meta.iloc[index]["ICULOS"])
                observations.append(observation)

            response = client.post(
                "/predict",
                json={
                    "patient_id": str(row.Patient_ID),
                    "observations": observations,
                },
            )
            assert response.status_code == 200, f"row_id={row_id}: {response.text}"

            actual = response.json()
            expected_score = float(row.p_calibrated)
            expected_alarm = expected_score >= threshold
            difference = abs(actual["risk_score"] - expected_score)

            passed = (
                difference <= TOLERANCE
                and actual["alarm"] == expected_alarm
                and actual["threshold"] == threshold
                and actual["is_mock"] is False
            )

            results.append(
                {
                    "row_id": row_id,
                    "window_length": len(observations),
                    "expected_score": expected_score,
                    "api_score": actual["risk_score"],
                    "absolute_difference": difference,
                    "expected_alarm": expected_alarm,
                    "api_alarm": actual["alarm"],
                    "passed": passed,
                }
            )

            print(
                f"[{number}/{len(selected)}] "
                f"row={row_id} hours={len(observations)} "
                f"diff={difference:.3e} "
                f"{'PASS' if passed else 'FAIL'}"
            )

    report = pd.DataFrame(results)
    output = Path(__file__).resolve().parent / "gru_reference_results.csv"
    report.to_csv(output, index=False, encoding="utf-8-sig")

    print()
    print("Checked cases:", len(report))
    print("Maximum probability difference:", report["absolute_difference"].max())
    print(
        "Alarm mismatches:",
        int((report["expected_alarm"] != report["api_alarm"]).sum()),
    )
    print("Report:", output)

    failed = report[~report["passed"]]
    if not failed.empty:
        print(failed.to_string(index=False))
        raise AssertionError("Reference comparison failed. See the CSV report.")

    print("GRU reference comparison: PASS")


if __name__ == "__main__":
    main()
