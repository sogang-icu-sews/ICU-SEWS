"""Load and validate the final GRU bundle. Owner: 김진호."""

import hashlib
import io
import json
import math
from pathlib import Path

import torch

from icu_sews.models.gru import GruSepsisModel


def load_gru_bundle(model_dir: str) -> dict:
    """Load trusted team artifacts on CPU; do not train or modify them."""
    directory = Path(model_dir)
    frozen = json.loads((directory / "frozen.json").read_text(encoding="utf-8"))
    blobs = {}
    for name in ["final.pt", "preprocessing.json", "platt.json", "threshold.json"]:
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != frozen["artifacts"][name]:
            raise ValueError(f"Model artifact hash mismatch: {name}")
        blobs[name] = data

    saved = torch.load(io.BytesIO(blobs["final.pt"]), map_location="cpu", weights_only=True)
    preprocessing = json.loads(blobs["preprocessing.json"])
    platt = json.loads(blobs["platt.json"])
    threshold = json.loads(blobs["threshold.json"])
    if saved["preprocessing"] != preprocessing:
        raise ValueError("Checkpoint preprocessing does not match preprocessing.json")

    config = saved["config"]
    sequence = config["model_prep"]["sequence"]
    static_columns = saved["static_columns"]
    if config["window"]["hours"] != 24:
        raise ValueError("Expected a 24-hour window")
    if sequence["vars"] != preprocessing["sequence"]["vars"]:
        raise ValueError("Sequence feature order mismatch")
    if static_columns != preprocessing["tabular"]["columns"]:
        raise ValueError("Static feature order mismatch")
    if saved["model_config"]["input_size"] != len(sequence["vars"]) + 2 * len(
        sequence["mask_delta_vars"]
    ):
        raise ValueError("GRU input size mismatch")
    if saved["model_config"]["static_size"] != len(static_columns):
        raise ValueError("GRU static size mismatch")

    for key in ["slope", "intercept", "clip"]:
        if not math.isfinite(float(platt[key])):
            raise ValueError(f"Invalid Platt coefficient: {key}")
    if not 0 < float(platt["clip"]) < 0.5:
        raise ValueError("Invalid Platt clip")
    if not 0 <= float(threshold["threshold"]) <= 1:
        raise ValueError("Invalid alarm threshold")
    if threshold["comparison"] != ">=" or threshold["cooldown_hours"] != 0:
        raise ValueError("Unsupported alarm policy")

    model = GruSepsisModel(**saved["model_config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError("Non-finite model weights")
    model.eval()
    return {
        "model": model,
        "config": config,
        "preprocessing": preprocessing,
        "static_columns": static_columns,
        "platt": platt,
        "threshold": threshold,
        "model_version": "gru-v2-platt-final-" + hashlib.sha256(blobs["final.pt"]).hexdigest()[:12],
    }
