"""Load the team's frozen ens_v1 fold-4 models without importing training scripts."""

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch

from icu_sews.models.gru import GruSepsisModel
from icu_sews.models.lstm import SepsisLSTM
from icu_sews.models.train_gru import REPO, STATIC, module

CANDIDATES = {
    "lstm": "LSTM",
    "ensemble": "앙상블 (Optuna 가중치)",
    "ensemble_equal": "앙상블 (같은 가중치)",
}


def checked_bytes(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Artifact hash mismatch: {Path(path).name}")
    return data


def load_ensemble_bundle(model_dir, preprocess_dir, mode="ensemble"):
    """Use only the provided ens_v1 and its original fold-4 preprocessing statistics.

    preprocess_dir points to SEWS_GRU/prepared/artifacts. No fitting occurs here.
    The manifest binds this release to the exact supplied artifacts, not arbitrary defaults.
    """
    if mode not in CANDIDATES:
        raise ValueError(f"Unsupported ensemble mode: {mode}")
    if not str(preprocess_dir).strip():
        raise ValueError("SEWS_PREPROCESS_DIR must point to prepared/artifacts")
    directory, prep = Path(model_dir), Path(preprocess_dir)
    manifest = json.loads(Path(__file__).with_name("ens_v1_manifest.json").read_text("utf-8"))
    frozen = json.loads(checked_bytes(directory / "frozen.json", manifest["frozen_sha256"]))
    if frozen["run"] != manifest["run"] or frozen["smoke"]:
        raise ValueError("Expected the full ens_v1 run")
    configs = {
        name: json.loads(checked_bytes(prep / name, expected))
        for name, expected in manifest["preprocessing_files"].items()
    }
    cfg = configs["preprocess_config.json"]
    registry = configs["feature_registry.json"]
    params = configs["preprocess_params/fold4.json"]
    if cfg["version"] != 4 or cfg["window"]["hours"] != 24 or params["fold"] != 4:
        raise ValueError("Expected config v4, 24-hour windows and fold-4 parameters")
    for name, expected in [
        ("sews_features", registry["module_sha256"]),
        ("sews_preprocess", registry["preprocess"]["module_sha256"]),
    ]:
        data = (REPO / "data/src" / f"{name}.py").read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Published preprocessing code changed: {name}")
    sequence = cfg["model_prep"]["sequence"]
    if sequence["vars"] != params["sequence"]["vars"]:
        raise ValueError("Sequence order mismatch")
    for part in ("sequence", "tabular"):
        values = params[part]
        for key in ("center", "scale"):
            vector = values[key]
            vector = list(vector.values()) if isinstance(vector, dict) else vector
            arr = np.asarray(vector, dtype=float)
            if not np.isfinite(arr).all() or (key == "scale" and (arr <= 0).any()):
                raise ValueError(f"Invalid {part} {key}")
        if not math.isfinite(values["clip"]) or values["clip"] <= 0:
            raise ValueError("Invalid preprocessing clip")
    feature_names = [f["name"] for f in registry["features"] if not f["optional"]]
    if feature_names != module("sews_features").feature_names(cfg, include_optional=False):
        raise ValueError("Feature registry order mismatch")
    n_in = len(sequence["vars"]) + 2 * len(sequence["mask_delta_vars"])
    blobs = {}

    def artifact(name):
        if name not in blobs:
            blobs[name] = checked_bytes(directory / name, frozen["files"][name])
        return blobs[name]

    weights = json.loads(artifact("ensemble_weights.json"))
    platt = json.loads(artifact("platt.json"))
    thresholds = json.loads(artifact("thresholds.json"))["thresholds"]
    if weights != frozen["weights"] or platt != frozen["platt"]:
        raise ValueError("Calibration/weights differ from frozen run")
    if thresholds != frozen["threshold"]:
        raise ValueError("Thresholds differ from frozen run")
    candidate = CANDIDATES[mode]
    for value in platt[candidate].values():
        if not math.isfinite(float(value)):
            raise ValueError("Non-finite calibration coefficient")
    threshold = float(thresholds[candidate])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid alarm threshold")
    selected_weights = weights["equal" if mode == "ensemble_equal" else "optuna"]
    if (
        set(selected_weights) != {"lgbm", "lstm", "gru"}
        or any(not math.isfinite(v) or v < 0 for v in selected_weights.values())
        or not math.isclose(sum(selected_weights.values()), 1.0, abs_tol=1e-8)
    ):
        raise ValueError("Invalid ensemble weights")
    models = {}
    kinds = ["lstm"] if mode == "lstm" else ["lstm", "gru", "lgbm"]
    for kind in kinds:
        metadata = json.loads(artifact(f"oof/{kind}_fold4.json"))
        name = f"oof/{kind}_fold4." + ("txt" if kind == "lgbm" else "pt")
        data = artifact(name)
        if metadata["fold"] != 4 or metadata["train_rows"] != params["sequence"]["n_rows"]:
            raise ValueError("Model/preprocessing training fold mismatch")
        if hashlib.sha256(data).hexdigest() != metadata["model_sha256"]:
            raise ValueError("Model metadata hash mismatch")
        if kind == "lgbm":
            import lightgbm as lgb

            model = lgb.Booster(model_str=data.decode("utf-8"))
            if model.num_feature() != len(feature_names):
                raise ValueError("LightGBM feature count mismatch")
            # Training used a NumPy matrix; LightGBM records Column_0 ... Column_124.
            if model.feature_name() not in [
                feature_names,
                [f"Column_{i}" for i in range(len(feature_names))],
            ]:
                raise ValueError("LightGBM feature order mismatch")
        else:
            saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            hp = saved["hp"]
            if hp != metadata["params"]:
                raise ValueError("Checkpoint hyperparameters differ from metadata")
            if kind == "lstm":
                model = SepsisLSTM(
                    n_in, len(STATIC), hp["hidden"], hp["layers"], hp["dropout"], hp["head"]
                )
            else:
                model = GruSepsisModel(n_in, hp["hidden"], hp["layers"], hp["dropout"], len(STATIC))
            model.load_state_dict(saved["state_dict"], strict=True)
            if not all(torch.isfinite(t).all().item() for t in model.state_dict().values()):
                raise ValueError("Non-finite model weights")
            model.eval()
        models[kind] = model
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "models": models,
        "config": cfg,
        "preprocessing": params,
        "static_columns": STATIC,
        "feature_names": feature_names,
        "mode": mode,
        "weights": selected_weights,
        "platt": platt[candidate],
        "threshold": threshold,
        "model_version": f"ens-v1-{mode}-{identity}",
    }
