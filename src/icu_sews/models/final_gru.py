"""Fixed v2 GRU -> disjoint Platt calibration -> threshold -> untouched test.

The design is fixed before opening test performance. Previously used development
labels informed model selection; calibration/tuning scores are not new validation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split
from torch import nn

from icu_sews.evaluation.compare_gru import (
    alarm_metrics,
    fit_calibrator,
    probability_metrics,
    rate_threshold,
    reliability,
)
from icu_sews.models import train_gru as training
from icu_sews.models.gru import GruSepsisModel

PLAN = {
    "seed": 42,
    "epochs": 3,
    "epoch_reason": "Median of v2 development best epochs [2,2,3,3,3]; fixed before test",
    "batch_size": 128,
    "learning_rate": 0.0003,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.2,
    "pos_weight": True,
    "held_out_dev_fold": 4,
    "calibration": "unweighted Platt on clipped logit(p), clip=1e-7",
    "policy": "patient_false_alarm10",
    "target_false_alarm_rate": 0.10,
    "alarm_rule": "p_calibrated >= threshold at every hour; no cooldown",
}
KEYS = ["row_id", "Patient_ID", "ICULOS", "set", "split", "fold", "SepsisLabel"]


def roles_for(em):
    patients = em.groupby("Patient_ID", sort=True).agg(
        set=("set", "first"),
        split=("split", "first"),
        fold=("fold", "first"),
        sepsis=("SepsisLabel", "max"),
    )
    patients["role"] = "test"
    dev = patients.split == "dev"
    patients.loc[dev, "role"] = "train"
    selected = patients[dev & (patients.fold == PLAN["held_out_dev_fold"])]
    strata = selected["set"].astype(str) + "_" + selected.sepsis.astype(str)
    if len(selected) == 0 or strata.value_counts().min() < 2:
        raise ValueError("Fold 4 needs at least two patients per site/outcome stratum")
    cal, tune = train_test_split(
        selected.index,
        test_size=0.5,
        random_state=PLAN["seed"],
        stratify=strata,
    )
    patients.loc[cal, "role"] = "calibration"
    patients.loc[tune, "role"] = "threshold"
    rows = {
        role: np.flatnonzero(em.Patient_ID.map(patients.role).to_numpy() == role)
        for role in ["train", "calibration", "threshold", "test"]
    }
    if sum(map(len, rows.values())) != len(em):
        raise ValueError("Every observation must have exactly one role")
    for role, ids in rows.items():
        if not len(ids) or len(np.unique(em.SepsisLabel.iloc[ids])) != 2:
            raise ValueError(f"{role} needs both outcome classes")
    same = np.diff(em.Patient_ID.to_numpy()) == 0
    if (np.diff(em.SepsisLabel.to_numpy(dtype=int))[same] < 0).any():
        raise ValueError("SepsisLabel must not revert from one to zero")
    return patients.reset_index(), rows


def fit_preprocessing(d, train_rows):
    """Refit only on final GRU training patients, including static scaling."""
    return {
        "sequence": d["pp"].fit_seq(d["values"][train_rows], d["cfg"]),
        "tabular": d["pp"].fit_tabular(d["static"].iloc[train_rows][training.STATIC], d["cfg"]),
    }


def apply_platt(p, coefficients):
    p = np.asarray(p, dtype=np.float64)
    return expit(
        coefficients["slope"] * logit(np.clip(p, 1e-7, 1 - 1e-7)) + coefficients["intercept"]
    )


def identity_for(d):
    files = [
        Path(__file__),
        Path(training.__file__),
        Path(training.__file__).with_name("gru.py"),
        training.REPO / "src/icu_sews/evaluation/compare_gru.py",
        training.REPO / "data/src/sews_eval.py",
    ]
    from importlib.metadata import version

    return {
        "plan": PLAN,
        "data": d["hashes"],
        "code": {p.name: training.digest(p) for p in files},
        "versions": {
            name: version(name)
            for name in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]
        },
    }


def verify_freeze(out, identity):
    frozen = training.read_json(out / "frozen.json")
    if frozen["identity"] != identity:
        raise ValueError(
            "Frozen run differs from current inputs/code/settings; do not retune on test"
        )
    for name, expected in frozen["artifacts"].items():
        if training.digest(out / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    return frozen


def fit(d, out, device, resume=False):
    out = Path(out)
    identity = identity_for(d)
    if (out / "frozen.json").exists():
        verify_freeze(out, identity)
        print("Final model, Platt and threshold already frozen; training skipped.", flush=True)
        return
    if (out / "test_started.json").exists() or (out / "test_summary.json").exists():
        raise ValueError("Test has been opened: refitting this run is forbidden")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_identity.json").exists():
        if not resume or training.read_json(out / "run_identity.json") != identity:
            raise ValueError("Resume requires identical data, code, versions and settings")
    else:
        if any(out.iterdir()):
            raise ValueError("Unrecognized nonempty output directory")
        training.atomic_json(out / "run_identity.json", identity)

    patients, rows = roles_for(d["em"])
    patients.to_csv(out / "patient_roles.csv", index=False)
    print("Patients by role:", patients.role.value_counts().to_dict(), flush=True)
    params = fit_preprocessing(d, rows["train"])
    training.atomic_json(out / "preprocessing.json", params)
    static = d["pp"].apply_tabular(d["static"], params["tabular"], training.STATIC)
    cfg = d["cfg"]["model_prep"]["sequence"]
    model_cfg = {k: PLAN[k] for k in ["hidden_size", "num_layers", "dropout"]}
    model_cfg.update(
        input_size=len(cfg["vars"]) + 2 * len(cfg["mask_delta_vars"]),
        static_size=len(training.STATIC),
    )
    seed = PLAN["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    model = GruSepsisModel(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PLAN["learning_rate"])
    y = d["labels"][rows["train"]]
    weight = float((y == 0).sum() / (y == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    checkpoint = out / "latest.pt"
    history, start = [], 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, history = saved["epoch"], saved["history"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(start, PLAN["epochs"]):
        before = time.perf_counter()
        model.train()
        total_loss = 0.0
        for _, x, s, lengths, labels in training.batches(
            d,
            rows["train"],
            params,
            static,
            PLAN["batch_size"],
            device,
            seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, s, lengths), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / len(rows["train"]),
                "seconds": time.perf_counter() - before,
            }
        )
        training.atomic_checkpoint(
            checkpoint,
            {
                "identity": identity,
                "epoch": epoch + 1,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
        )
        training.atomic_json(out / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
    # No best-epoch selection on calibration, threshold or test labels.
    training.atomic_json(out / "history.json", history)
    training.atomic_checkpoint(
        out / "final.pt",
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": model_cfg,
            "config": d["cfg"],
            "preprocessing": params,
            "static_columns": training.STATIC,
            "epoch": PLAN["epochs"],
            "identity": identity,
            "pos_weight": weight,
        },
    )
    predictions = {}
    for role in ["calibration", "threshold"]:
        predictions[role] = training.predict(
            model,
            d,
            rows[role],
            params,
            static,
            PLAN["batch_size"],
            device,
        ).astype(np.float64)
    cal_y = d["labels"][rows["calibration"]]
    fitted = fit_calibrator(predictions["calibration"], cal_y, "platt")
    coefficients = {
        "slope": float(fitted.coef_[0, 0]),
        "intercept": float(fitted.intercept_[0]),
        "clip": 1e-7,
        "formula": "sigmoid(slope * logit(clip(p_raw, clip, 1-clip)) + intercept)",
        "baseline_probability": float(cal_y.mean()),
    }
    if coefficients["slope"] <= 0:
        raise ValueError("Platt slope is not positive; inspect development fit before test")
    training.atomic_json(out / "platt.json", coefficients)
    tune = d["em"].iloc[rows["threshold"]][KEYS].copy()
    p = apply_platt(predictions["threshold"], coefficients)
    patients_tune = (
        tune.assign(p=p)
        .groupby("Patient_ID")
        .agg(
            y=("SepsisLabel", "max"),
            p=("p", "max"),
        )
    )
    threshold = rate_threshold(patients_tune.loc[patients_tune.y == 0, "p"], 0.10)
    policy = {
        "policy": PLAN["policy"],
        "threshold": threshold,
        "target_false_alarm_rate": 0.10,
        "comparison": ">=",
        "cooldown_hours": 0,
    }
    training.atomic_json(out / "threshold.json", policy)
    for role in ["calibration", "threshold"]:
        saved = d["em"].iloc[rows[role]][KEYS].copy()
        saved["p_raw"] = predictions[role]
        saved["p_calibrated"] = apply_platt(predictions[role], coefficients)
        saved.to_parquet(out / f"{role}_predictions.parquet", index=False)
    tuning_metrics = alarm_metrics(tune, p >= threshold)
    training.atomic_json(
        out / "development_summary.json",
        json_safe(
            {
                "patients": patients.role.value_counts().to_dict(),
                "rows": {k: len(v) for k, v in rows.items()},
                "threshold": policy,
                "threshold_fitting_scores": tuning_metrics,
                "test_evaluated": False,
                "note": "Threshold fitting scores only; not independent validation performance",
            }
        ),
    )
    artifacts = [
        "final.pt",
        "preprocessing.json",
        "platt.json",
        "threshold.json",
        "patient_roles.csv",
        "history.json",
        "development_summary.json",
    ]
    training.atomic_json(
        out / "frozen.json",
        {
            "identity": identity,
            "artifacts": {name: training.digest(out / name) for name in artifacts},
            "test_evaluated": False,
        },
    )
    print(f"Frozen model + Platt + threshold saved: {out}", flush=True)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def plot_test(curves, out):
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for method, group in curves.groupby("method"):
        ax.plot(group.mean_prediction, group.observed_rate, "o-", label=method)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Final held-out test calibration",
    )
    ax.legend()
    fig.savefig(out / "test_calibration.png", dpi=150)
    ax.set(xlim=(0, 0.1), ylim=(0, 0.1))
    fig.savefig(out / "test_calibration_zoom.png", dpi=150)
    plt.close(fig)


def evaluate(d, out, device):
    out = Path(out)
    verify_freeze(out, identity_for(d))
    lock = {"frozen_sha256": training.digest(out / "frozen.json")}
    started = out / "test_started.json"
    if started.exists():
        if training.read_json(started) != lock:
            raise ValueError("Test already opened for a different frozen model")
    else:
        training.atomic_json(started, lock)
    summary_path = out / "test_summary.json"
    if summary_path.exists():
        summary = training.read_json(summary_path)
        if summary["frozen_sha256"] != lock["frozen_sha256"]:
            raise ValueError("Test summary does not belong to this frozen model")
        for name, expected in summary["artifacts"].items():
            if training.digest(out / name) != expected:
                raise ValueError(f"Saved test artifact changed: {name}")
        print(f"Test already completed; reusing saved results: {summary_path}", flush=True)
        return summary
    _, rows = roles_for(d["em"])
    saved = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    model = GruSepsisModel(**saved["model_config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    params = saved["preprocessing"]
    static = d["pp"].apply_tabular(d["static"], params["tabular"], training.STATIC)
    test = d["em"].iloc[rows["test"]][KEYS].reset_index(drop=True)
    print(f"Evaluating fixed model: {test.Patient_ID.nunique()} test patients", flush=True)
    test["p_raw"] = training.predict(
        model, d, rows["test"], params, static, PLAN["batch_size"], device
    ).astype(np.float64)
    coefficients = training.read_json(out / "platt.json")
    policy = training.read_json(out / "threshold.json")
    test["p_calibrated"] = apply_platt(test.p_raw.to_numpy(), coefficients)
    test["alarm"] = test.p_calibrated >= policy["threshold"]
    test.to_parquet(out / "test_predictions.parquet", index=False)
    probability_rows, alarm_rows, curves = [], [], []
    scopes = {
        "all": np.ones(len(test), dtype=bool),
        "set_A": test["set"] == "A",
        "set_B": test["set"] == "B",
        "ICULOS_le60": test.ICULOS <= 60,
    }
    for method, column in [("raw", "p_raw"), ("platt", "p_calibrated")]:
        curve = reliability(test.SepsisLabel.to_numpy(), test[column].to_numpy()).reset_index()
        curve["method"] = method
        curves.append(curve)
        for scope, mask in scopes.items():
            group = test[mask]
            if group.empty:
                continue
            probability_rows.append(
                {
                    "method": method,
                    "scope": scope,
                    **probability_metrics(
                        group.SepsisLabel.to_numpy(),
                        group[column].to_numpy(),
                        np.full(len(group), coefficients["baseline_probability"]),
                    ),
                }
            )
    for scope in ["all", "set_A", "set_B"]:
        group = test[scopes[scope]]
        if not group.empty:
            alarm_rows.append(
                {
                    "scope": scope,
                    "policy": policy["policy"],
                    **alarm_metrics(group, group.alarm.to_numpy()),
                }
            )
    pd.DataFrame(probability_rows).to_csv(out / "test_probability_metrics.csv", index=False)
    pd.DataFrame(alarm_rows).to_csv(out / "test_alarm_metrics.csv", index=False)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out / "test_reliability_bins.csv", index=False)
    plot_test(curves, out)
    files = [
        "test_predictions.parquet",
        "test_probability_metrics.csv",
        "test_alarm_metrics.csv",
        "test_reliability_bins.csv",
        "test_calibration.png",
        "test_calibration_zoom.png",
    ]
    summary = json_safe(
        {
            "complete": True,
            "test_evaluated": True,
            **lock,
            "patients": int(test.Patient_ID.nunique()),
            "rows": len(test),
            "threshold": policy,
            "probability": probability_rows,
            "alarms": alarm_rows,
            "artifacts": {name: training.digest(out / name) for name in files},
            "limitations": [
                "Single fixed split/model; no confidence intervals in this report",
                "Development data informed model/epoch/policy selection",
                "Onset inferred as first positive + 6h; starts-positive excluded from timing only",
                "Hourly flags without cooldown; not notification event counts",
                "Do not adjust model, calibration or threshold based on these test scores",
            ],
        }
    )
    training.atomic_json(summary_path, summary)
    print(f"Final test evaluation complete: {summary_path}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("GPU unavailable: choose a Colab GPU runtime")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    d = training.load_data(args.root)
    device = torch.device(args.device)
    if args.stage == "fit":
        fit(d, args.out, device, args.resume)
    else:
        evaluate(d, args.out, device)


if __name__ == "__main__":
    main()
