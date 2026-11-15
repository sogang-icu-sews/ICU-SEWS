"""Patient boundaries, padding, fold isolation, and restart equivalence."""

import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from icu_sews.models.gru import GruSepsisModel
from icu_sews.models.train_gru import (
    STATIC,
    load_data,
    main,
    module,
    pack_inputs,
    validate_alignment,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "inputs"
    data, art = root / "data", root / "artifacts"
    data.mkdir(parents=True)
    (art / "preprocess_params").mkdir(parents=True)
    cfg = json.loads((REPO / "artifacts/preprocess_config.json").read_text("utf-8"))
    registry = json.loads((REPO / "artifacts/feature_registry.json").read_text("utf-8"))
    em, manifest = [], []
    for patient in range(12):
        fold = patient // 2 if patient < 10 else -1
        split = "dev" if patient < 10 else "test"
        site = "A" if patient % 2 == 0 else "B"
        positive = patient % 2
        manifest.append([patient + 1, site, positive, split, fold])
        for hour in range(8):
            em.append(
                [len(em), patient + 1, hour + 1, site, split, fold, int(positive and hour >= 5)]
            )
    em = pd.DataFrame(
        em, columns=["row_id", "Patient_ID", "ICULOS", "set", "split", "fold", "SepsisLabel"]
    )
    manifest = pd.DataFrame(manifest, columns=["Patient_ID", "set", "sepsis", "split", "fold"])
    manifest.to_csv(art / "patient_split.csv", index=False)
    em.to_parquet(data / "eval_meta.parquet", index=False)
    rng = np.random.default_rng(7)
    values = rng.normal(size=(len(em), 24)).astype(np.float32)
    values[::3, 0] = np.nan
    np.save(data / "seq_values.npy", values)
    np.save(data / "seq_row_start.npy", np.arange(len(em)) // 8 * 8)
    static = pd.DataFrame(rng.normal(size=(len(em), 5)), columns=STATIC)
    static.insert(0, "row_id", np.arange(len(em)))
    static.to_parquet(data / "gru_static.parquet", index=False)
    pp = module("sews_preprocess")
    for k in range(5):
        mask = (em.split == "dev") & (em.fold != k)
        params = {
            "sequence": pp.fit_seq(values[mask], cfg),
            "tabular": pp.fit_tabular(static.loc[mask, STATIC], cfg),
        }
        (art / f"preprocess_params/fold{k}.json").write_text(json.dumps(params), "utf-8")
    for name, value in [
        ("preprocess_config", cfg),
        ("feature_registry", registry),
        ("seq_meta", {"vars": cfg["model_prep"]["sequence"]["vars"], "row_order": "row_id"}),
    ]:
        (art / f"{name}.json").write_text(json.dumps(value), "utf-8")
    return root


def test_padding_does_not_change_logits():
    torch.manual_seed(3)
    model = GruSepsisModel(3, hidden_size=4, num_layers=1, dropout=0, static_size=2).eval()
    real = torch.randn(1, 3, 3)
    static = torch.randn(1, 2)
    padded = torch.cat([real, torch.randn(1, 21, 3) * 100], dim=1)
    with torch.no_grad():
        a = model(real, static, torch.tensor([3]))
        b = model(padded, static, torch.tensor([3]))
    torch.testing.assert_close(a, b)


def test_left_padding_is_relocated():
    x = np.arange(24, dtype=np.float32)[None, :, None]
    windows = {"x": x, "mask": np.ones_like(x), "delta": np.zeros_like(x)}
    result = pack_inputs(windows, np.array([3]))
    np.testing.assert_array_equal(result[0, :3, 0], [21, 22, 23])
    assert not result[0, 3:].any()


def test_future_and_previous_patient_cannot_change_window(prepared):
    d = load_data(prepared)
    pp = d["pp"]
    params = json.loads((prepared / "artifacts/preprocess_params/fold0.json").read_text())
    rows = np.array([9])  # second patient's second hour
    original = pp.seq_windows(d["values"], d["starts"], rows, params["sequence"], d["cfg"])
    changed = np.array(d["values"])
    changed[:8] = 9999
    changed[10:] = -9999
    modified = pp.seq_windows(changed, d["starts"], rows, params["sequence"], d["cfg"])
    for key in original:
        np.testing.assert_array_equal(original[key], modified[key])


def test_patient_fold_leakage_rejected(prepared):
    d = load_data(prepared)
    em = d["em"].copy()
    em.loc[0, "fold"] = 4
    manifest = pd.read_csv(prepared / "artifacts/patient_split.csv")
    with pytest.raises(ValueError, match="cross splits"):
        validate_alignment(em, d["starts"], manifest)


def test_training_resume_matches_uninterrupted_and_excludes_test(prepared, tmp_path):
    base = [
        "--root",
        str(prepared),
        "--device",
        "cpu",
        "--hidden-size",
        "4",
        "--batch-size",
        "16",
        "--pos-weight",
    ]
    resumed, continuous = tmp_path / "resumed", tmp_path / "continuous"
    main([*base, "--out", str(resumed), "--epochs", "1"])
    main([*base, "--out", str(resumed), "--epochs", "2", "--resume"])
    main([*base, "--out", str(continuous), "--epochs", "2"])
    a = torch.load(resumed / "fold0/latest.pt", weights_only=False)
    b = torch.load(continuous / "fold0/latest.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    oof = pd.read_parquet(resumed / "oof_predictions.parquet")
    assert set(oof.Patient_ID) == {1, 2}
    assert oof.row_id.is_unique and oof.p_gru.between(0, 1).all()
    with pytest.raises(ValueError, match="settings changed"):
        main([*base, "--out", str(resumed), "--epochs", "3", "--resume", "--seed", "8"])


def test_all_folds_export_exactly_development_rows(prepared, tmp_path):
    out = tmp_path / "all_folds"
    main(
        [
            "--root",
            str(prepared),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--hidden-size",
            "4",
            "--epochs",
            "1",
            "--folds",
            "0",
            "1",
            "2",
            "3",
            "4",
        ]
    )
    oof = pd.read_parquet(out / "oof_predictions.parquet")
    assert set(oof.row_id) == set(range(80))
    assert oof.row_id.is_unique
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete_5fold"] and not summary["test_evaluated"]


def test_zip_preparation_matches_shared_cleaning(prepared, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "prepare_gru", REPO / "scripts/prepare_gru_colab.py"
    )
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    fake_repo = tmp_path / "repo"
    shutil.copytree(prepared / "artifacts", fake_repo / "artifacts")
    (fake_repo / "data/src").mkdir(parents=True)
    shutil.copy2(REPO / "data/src/sews_features.py", fake_repo / "data/src/sews_features.py")
    manifest = pd.read_csv(fake_repo / "artifacts/patient_split.csv")
    manifest["set"] = "A"
    manifest.to_csv(fake_repo / "artifacts/patient_split.csv", index=False)
    seq_meta_path = fake_repo / "artifacts/seq_meta.json"
    seq_meta = json.loads(seq_meta_path.read_text())
    seq_meta["n_rows"] = 96
    seq_meta_path.write_text(json.dumps(seq_meta), "utf-8")
    cfg = json.loads((fake_repo / "artifacts/preprocess_config.json").read_text("utf-8"))
    archive_path = tmp_path / "training.zip"
    expected_values, expected_static = [], []
    with zipfile.ZipFile(archive_path, "w") as archive:
        for patient in manifest.itertuples(index=False):
            raw = pd.DataFrame({v: np.full(8, 10.0) for v in seq_meta["vars"]})
            raw["Age"], raw["Gender"], raw["Unit1"], raw["Unit2"] = 100.0, 1.0, 1.0, 0.0
            raw["HR"], raw["Temp"], raw["Calcium"] = 80.0, 37.0, 2.0
            raw["ICULOS"] = np.arange(1, 9)
            raw["SepsisLabel"] = (np.arange(8) >= 5).astype(int) * patient.sepsis
            name = f"training_setA/training_setA/p{patient.Patient_ID:06d}.psv"
            archive.writestr(name, raw.to_csv(sep="|", index=False))
            clean, _ = prep.sf.clean(raw, cfg)
            expected_values.append(clean[seq_meta["vars"]].to_numpy(np.float32))
            expected_static.extend(clean.Age.tolist())
    out = tmp_path / "generated"
    monkeypatch.setattr(prep, "REPO", fake_repo)
    monkeypatch.setattr(sys, "argv", ["prepare", "--archive", str(archive_path), "--out", str(out)])
    prep.main()
    actual = load_data(out)
    np.testing.assert_allclose(actual["values"], np.concatenate(expected_values), equal_nan=True)
    np.testing.assert_array_equal(actual["static"].Age, expected_static)
    assert (out / "artifacts/gru_preparation_complete.json").is_file()
    assert len(pd.read_csv(out / "artifacts/gru_source_files.csv")) == 12
