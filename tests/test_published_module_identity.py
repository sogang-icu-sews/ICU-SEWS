"""Protect published data and frozen-model source identities from formatting edits."""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["sews_features", "sews_preprocess", "sews_eval"])
def test_published_source_identity(name):
    registry = json.loads((REPO / "artifacts/feature_registry.json").read_text("utf-8"))
    expected = {
        "sews_features": registry["module_sha256"],
        "sews_preprocess": registry["preprocess"]["module_sha256"],
        # Evaluation source in the v2 + Platt final run (commit 59c7379).
        "sews_eval": "c039f6e011088d8102823a230ec0536dd24c86bc23bf783cd17e0e3c3709df5f",
    }
    source = (REPO / f"data/src/{name}.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source).hexdigest() == expected[name], (
        f"{name} differs from the published release; do not reformat frozen sources. "
        "See docs/PUBLISHED_MODULE_COMPATIBILITY.md before changing the data/model contract."
    )
