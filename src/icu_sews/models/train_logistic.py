"""Train a reproducible baseline from raw PSV; leave test performance sealed."""

import argparse
import hashlib
import io
import json
import platform
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from icu_sews.data.loaders import iter_patient_files
from icu_sews.data.split import split_patient_ids
from icu_sews.evaluation.metrics import calculate_metrics, expected_calibration_error
from icu_sews.models.logistic import build_logistic_pipeline


def load_records(raw_dir: Path, sources: dict, features: list[str]) -> tuple[pd.DataFrame, str]:
    """Read selected numeric columns, validate keys/labels, and hash raw contents."""
    if not features or len(features) != len(set(features)):
        raise ValueError("Features must be nonempty and unique")
    if set(features) & {"patient_id", "source", "record_file", "SepsisLabel", "ICULOS"}:
        raise ValueError("Identifiers, time keys, and labels cannot be baseline features")
    files = list(iter_patient_files(raw_dir, sources))
    for source in sources:
        if not any(item[0] == source for item in files):
            raise ValueError(f"No PSV files for source {source}; check data.sources paths")
    if not files:
        raise ValueError("No PSV files found")
    columns = [*features, "ICULOS", "SepsisLabel"]

    def read_one(item: tuple[str, Path]) -> tuple[str, np.ndarray, str]:
        source, path = item
        raw = path.read_bytes()
        try:
            frame = pd.read_csv(io.BytesIO(raw), sep="|", usecols=columns)
            values = frame[columns].to_numpy(dtype=np.float64)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid numeric data or columns: {path}") from exc
        if not len(values):
            raise ValueError(f"Empty patient file: {path}")
        hours, labels = values[:, -2], values[:, -1]
        if not np.isfinite(hours).all() or (hours < 1).any() or (hours % 1 != 0).any():
            raise ValueError(f"Invalid ICULOS: {path}")
        if len(np.unique(hours)) != len(hours):
            raise ValueError(f"Duplicate patient time: {path}")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Invalid SepsisLabel: {path}")
        if np.isinf(values[:, :-2]).any():
            raise ValueError(f"Infinite feature value: {path}")
        return f"{source}_{path.stem}", values[np.argsort(hours)], hashlib.sha256(raw).hexdigest()

    arrays, ids, counts = [], [], []
    digest = hashlib.sha256()
    # Independent file reads only; input ordering stays deterministic.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (patient_id, values, content_hash) in enumerate(pool.map(read_one, files), 1):
            ids.append(patient_id)
            arrays.append(values)
            counts.append(len(values))
            digest.update(f"{patient_id}:{content_hash}\n".encode())
            if index % 2000 == 0 or index == len(files):
                print(f"Loaded {index}/{len(files)} patients", flush=True)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source-prefixed patient IDs")
    records = pd.DataFrame(np.concatenate(arrays), columns=columns)
    records.insert(
        0, "patient_id", pd.Categorical.from_codes(np.repeat(np.arange(len(ids)), counts), ids)
    )
    records["ICULOS"] = records["ICULOS"].astype("int32")
    records["SepsisLabel"] = records["SepsisLabel"].astype("int8")
    return records, digest.hexdigest()


def get_patient_split(records: pd.DataFrame, path: Path, settings: dict) -> pd.DataFrame:
    """Reuse an existing shared split; reject partial or conflicting manifests."""
    summary = records.groupby("patient_id", observed=True)["SepsisLabel"].max().reset_index()
    summary = summary.rename(columns={"SepsisLabel": "is_sepsis"}).sort_values("patient_id")
    if path.exists():
        split = pd.read_parquet(path)
        if not {"patient_id", "split"}.issubset(split.columns):
            raise ValueError("Split file must contain patient_id and split")
        split = split[["patient_id", "split"]].copy()
    else:
        result = split_patient_ids(
            summary,
            train_ratio=settings["train_ratio"],
            validation_ratio=settings["validation_ratio"],
            test_ratio=settings["test_ratio"],
            random_seed=settings["random_seed"],
        )
        split = pd.DataFrame(
            [
                (pid, name)
                for name, ids in [
                    ("train", result.train_ids),
                    ("validation", result.validation_ids),
                    ("test", result.test_ids),
                ]
                for pid in ids
            ],
            columns=["patient_id", "split"],
        )
    if split["patient_id"].isna().any() or split["patient_id"].duplicated().any():
        raise ValueError("Split file has missing or duplicate patient IDs")
    if set(split["patient_id"]) != set(summary["patient_id"]):
        raise ValueError("Split patients differ from raw data; do not silently resplit")
    if set(split["split"]) != {"train", "validation", "test"}:
        raise ValueError("Split file must use train, validation, and test")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_parquet(path, index=False)
    return split


def run(data_config: Path, model_config: Path, output: Path) -> dict:
    """Fit on train, evaluate validation, and persist an inference-ready pipeline."""
    if output.exists():
        raise FileExistsError(f"Choose a fresh run directory: {output}")
    data = yaml.safe_load(data_config.read_text(encoding="utf-8"))
    model = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    if data["label"]["column"] != "SepsisLabel" or data["label"]["apply_additional_shift"]:
        raise ValueError("This baseline requires the official unshifted SepsisLabel")
    threshold = float(model["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one")
    features = model["features"]
    records, fingerprint = load_records(
        Path(data["data"]["raw_dir"]), data["data"]["sources"], features
    )
    split_path = Path(data["data"]["processed_dir"]) / "patient_split.parquet"
    split = get_patient_split(records, split_path, data["split"])
    membership = records["patient_id"].map(split.set_index("patient_id")["split"])
    train = records.loc[membership == "train"]
    validation = records.loc[membership == "validation"]
    for name, frame in [("train", train), ("validation", validation)]:
        if frame["SepsisLabel"].nunique() != 2:
            raise ValueError(f"Both classes are required in {name}")
    print(f"Training on {len(train):,} rows; validating on {len(validation):,} rows", flush=True)
    pipeline = build_logistic_pipeline(model["model"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(train[features], train["SepsisLabel"])
    probabilities = pipeline.predict_proba(validation[features])[:, 1]
    labels = validation["SepsisLabel"].to_numpy()
    metrics = calculate_metrics(labels, probabilities).to_dict()
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics.update(
        ece=expected_calibration_error(labels, probabilities),
        positive_prevalence=float(labels.mean()),
        threshold=threshold,
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        specificity=float(tn / (tn + fp)),
        confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": fingerprint,
        "data_config": data,
        "model_config": model,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "pyyaml"]
        },
        "split_sha256": hashlib.sha256(
            split.sort_values("patient_id").to_csv(index=False).encode()
        ).hexdigest(),
        "split_counts": {
            name: {
                "patients": int((split["split"] == name).sum()),
                "rows": int((membership == name).sum()),
            }
            for name in ["train", "validation", "test"]
        },
        "train_missing_fraction": train[features].isna().mean().to_dict(),
        "validation_missing_fraction": validation[features].isna().mean().to_dict(),
        "all_missing_train_features": train[features]
        .columns[train[features].isna().all()]
        .tolist(),
        "evaluation_unit": "patient-hour",
        "auprc_definition": "sklearn average_precision_score (AP)",
        "test_evaluated": False,
        "label_policy": "official SepsisLabel at the current row; no additional shift",
    }
    output.mkdir(parents=True)
    joblib.dump(pipeline, output / "logistic_pipeline.joblib")
    restored = joblib.load(output / "logistic_pipeline.joblib")
    np.testing.assert_allclose(
        restored.predict_proba(validation[features].iloc[:100])[:, 1], probabilities[:100]
    )
    for name, payload in [("metrics.json", metrics), ("metadata.json", metadata)]:
        (output / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    exported = validation[["patient_id", "ICULOS", "SepsisLabel"]].copy()
    exported["probability"] = probabilities
    exported.to_parquet(output / "validation_predictions.parquet", index=False)
    split.to_parquet(output / "patient_split.parquet", index=False)
    pd.DataFrame(
        {
            "feature": features,
            "standardized_coefficient": pipeline.named_steps["classifier"].coef_[0],
        }
    ).to_csv(output / "coefficients.csv", index=False)
    print(json.dumps({"output": str(output), "validation": metrics}, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/logistic.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / f"logistic_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}",
    )
    args = parser.parse_args()
    run(args.data_config, args.model_config, args.output)


if __name__ == "__main__":
    main()
