"""Within-original-fold, patient-separated calibration and alarm comparison.

Each original GRU validation fold is split into three patient groups. Roles rotate:
calibrator fitting, threshold selection, evaluation. All three groups were excluded
from that fold's GRU gradient training. Epoch/hyperparameter selection already used
development labels: this is development analysis, NOT an unbiased nested-CV estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[3]
KEYS = ["row_id", "Patient_ID", "ICULOS", "set", "fold", "SepsisLabel"]
METHODS = ["raw", "platt", "isotonic"]
POLICIES = ["row_specificity95", "patient_false_alarm10", "max_utility"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utility_module():
    spec = importlib.util.spec_from_file_location("sews_eval", REPO / "data/src/sews_eval.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_oof(frame, manifest):
    required = [*KEYS, "p_gru"]
    if not set(required).issubset(frame):
        raise ValueError(f"Required OOF columns: {required}")
    if frame[KEYS].isna().any().any() or frame.empty:
        raise ValueError("Missing metadata or empty OOF")
    if frame.row_id.duplicated().any() or frame.duplicated(["Patient_ID", "ICULOS"]).any():
        raise ValueError("Duplicate OOF observations")
    p = frame.p_gru.to_numpy()
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    if not frame.SepsisLabel.isin([0, 1]).all() or set(frame.fold) != set(range(5)):
        raise ValueError("Expected binary labels and all five original folds")
    if not manifest.Patient_ID.is_unique:
        raise ValueError("Duplicate patients in split manifest")
    ordered = frame.sort_values(["Patient_ID", "ICULOS"]).reset_index(drop=True)
    grouped = ordered.groupby("Patient_ID", sort=False)
    if (grouped[["fold", "set"]].nunique() != 1).any().any():
        raise ValueError("Patient crosses folds or sites")
    actual = grouped.agg(
        fold=("fold", "first"), set=("set", "first"), sepsis=("SepsisLabel", "max")
    ).sort_index()
    expected = (
        manifest[manifest.split == "dev"]
        .set_index("Patient_ID")[["fold", "set", "sepsis"]]
        .sort_index()
    )
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("OOF patients/folds/sites/outcomes do not match the dev manifest") from exc
    same = np.diff(ordered.Patient_ID.to_numpy()) == 0
    if not (np.diff(ordered.ICULOS.to_numpy())[same] == 1).all():
        raise ValueError("Missing/reordered hours within a patient's OOF record")
    if (np.diff(ordered.SepsisLabel.to_numpy(dtype=int))[same] < 0).any():
        raise ValueError("SepsisLabel must not return from one to zero")
    return ordered[required]


def paired_oof(v2, v3, manifest):
    a, b = validate_oof(v2, manifest), validate_oof(v3, manifest)
    try:
        pd.testing.assert_frame_equal(a[KEYS], b[KEYS], check_dtype=False)
    except AssertionError as exc:
        raise ValueError("v2 and v3 must contain identical observation IDs and labels") from exc
    result = a.rename(columns={"p_gru": "v2"})
    result["v3"] = b.p_gru.to_numpy()
    return result


def assign_groups(frame, seed=42):
    pts = frame.groupby("Patient_ID", sort=True).agg(
        fold=("fold", "first"), set=("set", "first"), sepsis=("SepsisLabel", "max")
    )
    pts["group"] = -1
    for fold in range(5):
        selected = pts[pts.fold == fold]
        strata = selected["set"].astype(str) + "_" + selected.sepsis.astype(str)
        if strata.value_counts().min() < 3:
            raise ValueError("Each original fold needs >=3 patients in each site/outcome stratum")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold)
        for group, (_, ids) in enumerate(cv.split(selected, strata)):
            pts.loc[selected.index[ids], "group"] = group
    return pts.reset_index()


def role_masks(frame, groups, fold, rotation):
    in_fold = frame.fold.to_numpy() == fold
    assignment = frame.Patient_ID.map(groups.set_index("Patient_ID")["group"]).to_numpy()
    return (
        in_fold & (assignment == (rotation + 1) % 3),
        in_fold & (assignment == (rotation + 2) % 3),
        in_fold & (assignment == rotation),
    )


def fit_calibrator(p, y, method):
    if method == "raw":
        return None
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration requires both outcomes")
    if method == "platt":
        return LogisticRegression(C=1e6, max_iter=2000).fit(
            logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1), y
        )
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p, y)
    raise ValueError(f"Unknown method: {method}")


def calibrated(p, fitted, method):
    if method == "raw":
        return np.asarray(p, dtype=float).copy()
    if method == "platt":
        return expit(fitted.coef_[0, 0] * logit(np.clip(p, 1e-7, 1 - 1e-7)) + fitted.intercept_[0])
    # Keep one float64 ULP above tied scores meaningful when applying rate thresholds.
    # NumPy 2 weak scalar promotion can otherwise round the threshold back to float32.
    return np.asarray(fitted.predict(p), dtype=np.float64)


def rate_threshold(negative_scores, max_rate):
    """Conservative threshold respecting ties and the empirical false-alarm budget."""
    scores = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    if not len(scores) or not 0 <= max_rate < 1:
        raise ValueError("Need negative observations and a rate in [0,1)")
    allowed = int(np.floor(max_rate * len(scores)))
    return float(np.nextafter(scores[allowed], np.inf))


def utility_parts(frame):
    ev = utility_module()
    idx = ev._time_index(frame)
    u0 = ev.row_utility(frame, np.zeros(len(frame), dtype=bool), idx)
    u1 = ev.row_utility(frame, np.ones(len(frame), dtype=bool), idx)
    _, rel, septic = idx
    best = septic & (rel >= -12) & (rel <= 3)
    best_total = ev.row_utility(frame, best, idx).sum()
    return u0, u1 - u0, float(best_total - u0.sum())


def utility_threshold(frame, p):
    """Exact optimum over distinct scores, plus the no-alarm option; ties stay together."""
    _, delta, _ = utility_parts(frame)
    order = np.argsort(-p, kind="stable")
    scores = p[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    improvement = np.r_[0.0, np.cumsum(delta[order])[ends]]
    thresholds = np.r_[np.nextafter(scores[0], np.inf), scores[ends]]
    return float(thresholds[np.argmax(improvement)])


def select_thresholds(frame, p):
    patients = (
        frame.assign(score=p)
        .groupby("Patient_ID")
        .agg(y=("SepsisLabel", "max"), score=("score", "max"))
    )
    return {
        "row_specificity95": rate_threshold(p[frame.SepsisLabel.to_numpy() == 0], 0.05),
        "patient_false_alarm10": rate_threshold(patients.loc[patients.y == 0, "score"], 0.10),
        "max_utility": utility_threshold(frame, p),
    }


def probability_metrics(y, p, baseline):
    brier = float(np.mean((p - y) ** 2))
    reference = float(np.mean((baseline - y) ** 2))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(
        float((bins == k).mean()) * abs(float(p[bins == k].mean() - y[bins == k].mean()))
        for k in range(10)
        if (bins == k).any()
    )
    both = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "prevalence": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "AUROC": float(roc_auc_score(y, p)) if both else np.nan,
        "AUPRC": float(average_precision_score(y, p)) if both else np.nan,
        "Brier": brier,
        "baseline_Brier": reference,
        "Brier_skill": 1 - brier / reference if reference > 0 else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ECE_10_equal_width": ece,
    }


def safe_ratio(num, den):
    return float(num / den) if den else np.nan


def alarm_metrics(frame, alarms):
    """Hourly alarm flags, not episodes; onset-derived timing excludes left-censored cases."""
    y = frame.SepsisLabel.to_numpy(dtype=int)
    a = np.asarray(alarms, dtype=bool)
    tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    tn, fn = int((~a & (y == 0)).sum()), int((~a & (y == 1)).sum())
    ordered = frame.reset_index(drop=True).copy()
    grouped = ordered.groupby("Patient_ID", sort=False)
    onset = (
        ordered.ICULOS.where(ordered.SepsisLabel == 1).groupby(ordered.Patient_ID).transform("min")
        + 6
    )
    rel = ordered.ICULOS.to_numpy() - onset.to_numpy()
    early = a & (rel >= -12) & (rel <= -6)
    before = a & (rel >= -12) & (rel < 0)
    pt = grouped.agg(y=("SepsisLabel", "max"), starts_positive=("SepsisLabel", "first"))
    pt["alarm"] = pd.Series(a).groupby(ordered.Patient_ID).max()
    pt["early"] = pd.Series(early).groupby(ordered.Patient_ID).max()
    pt["before"] = pd.Series(before).groupby(ordered.Patient_ID).max()
    pt["lead"] = pd.Series(np.where(before, -rel, np.nan)).groupby(ordered.Patient_ID).max()
    eligible = (pt.y == 1) & (pt.starts_positive == 0)
    leads = pt.loc[eligible & pt.before, "lead"]
    nonseptic_rows = ordered.Patient_ID.map(pt.y).to_numpy() == 0
    _, delta, denom = utility_parts(ordered)
    return {
        "row_sensitivity": safe_ratio(tp, tp + fn),
        "row_PPV": safe_ratio(tp, tp + fp),
        "row_specificity": safe_ratio(tn, tn + fp),
        "alarm_hours_per100h": float(100 * a.mean()),
        "false_alarm_hours_per100_nonsepsis_h": 100
        * safe_ratio(int((a & nonseptic_rows).sum()), int(nonseptic_rows.sum())),
        "nonsepsis_patient_false_alarm_rate": float(pt.loc[pt.y == 0, "alarm"].mean()),
        "sepsis_patient_ever_alarm_rate": float(pt.loc[pt.y == 1, "alarm"].mean()),
        "early_recall_6to12h": float(pt.loc[eligible, "early"].mean()),
        "pre_onset_recall_0to12h": float(pt.loc[eligible, "before"].mean()),
        "median_lead_hours_detected": float(leads.median()) if len(leads) else np.nan,
        "timing_eligible_patients": int(eligible.sum()),
        "timing_excluded_starts_positive": int(((pt.y == 1) & (pt.starts_positive == 1)).sum()),
        "Utility": safe_ratio(float(delta[a].sum()), denom),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def reliability(y, p):
    q = pd.qcut(p, q=10, labels=False, duplicates="drop")
    if np.isnan(q).all():
        q = np.zeros(len(p), dtype=int)
    return (
        pd.DataFrame({"bin": q, "p": p, "y": y})
        .groupby("bin")
        .agg(mean_prediction=("p", "mean"), observed_rate=("y", "mean"), rows=("y", "size"))
    )


def compare(frame, out, seed=42, make_plots=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups = assign_groups(frame, seed)
    groups.to_csv(out / "patient_groups.csv", index=False)
    y = frame.SepsisLabel.to_numpy(dtype=int)
    baseline = np.full(len(frame), np.nan)
    predictions = {(m, c): np.full(len(frame), np.nan) for m in ["v2", "v3"] for c in METHODS}
    flags = {(m, c, p): np.zeros(len(frame), dtype=bool) for m, c in predictions for p in POLICIES}
    thresholds, roles = [], []
    seen = np.zeros(len(frame), dtype=int)
    for fold in range(5):
        for rotation in range(3):
            cal, tune, test = role_masks(frame, groups, fold, rotation)
            seen[test] += 1
            baseline[test] = y[cal].mean()
            roles.append(
                {
                    "fold": fold,
                    "rotation": rotation,
                    "calibration_group": (rotation + 1) % 3,
                    "threshold_group": (rotation + 2) % 3,
                    "evaluation_group": rotation,
                    "calibration_rows": int(cal.sum()),
                    "threshold_rows": int(tune.sum()),
                    "evaluation_rows": int(test.sum()),
                }
            )
            for model in ["v2", "v3"]:
                raw = frame[model].to_numpy()
                for method in METHODS:
                    fitted = fit_calibrator(raw[cal], y[cal], method)
                    p_tune = calibrated(raw[tune], fitted, method)
                    p_test = calibrated(raw[test], fitted, method)
                    predictions[model, method][test] = p_test
                    if fitted is not None:
                        directory = out / "calibrators" / model
                        directory.mkdir(parents=True, exist_ok=True)
                        joblib.dump(
                            fitted, directory / f"fold{fold}_rotation{rotation}_{method}.joblib"
                        )
                    chosen = select_thresholds(frame[tune], p_tune)
                    for policy, threshold in chosen.items():
                        flags[model, method, policy][test] = p_test >= threshold
                        check = alarm_metrics(frame[tune], p_tune >= threshold)
                        thresholds.append(
                            {
                                "model": model,
                                "method": method,
                                "fold": fold,
                                "rotation": rotation,
                                "policy": policy,
                                "threshold": threshold,
                                "tuning_specificity": check["row_specificity"],
                                "tuning_patient_false_alarm_rate": check[
                                    "nonsepsis_patient_false_alarm_rate"
                                ],
                            }
                        )
            print(
                f"Completed calibration/evaluation: original fold {fold}, rotation {rotation}",
                flush=True,
            )
    if not (seen == 1).all() or not np.isfinite(baseline).all():
        raise RuntimeError("Every development row must be evaluated exactly once")
    probability_rows, alarm_rows, curves = [], [], []
    scopes = {
        "all": np.ones(len(frame), dtype=bool),
        "ICULOS_le60": frame.ICULOS.to_numpy() <= 60,
        "set_A": frame["set"].to_numpy() == "A",
        "set_B": frame["set"].to_numpy() == "B",
    }
    for (model, method), p in predictions.items():
        if not np.isfinite(p).all():
            raise RuntimeError("Incomplete probability predictions")
        saved = frame[KEYS].copy()
        saved["p_raw"], saved["p_calibrated"], saved["baseline_probability"] = (
            frame[model],
            p,
            baseline,
        )
        for policy in POLICIES:
            saved[f"alarm_{policy}"] = flags[model, method, policy]
        saved.to_parquet(out / f"predictions_{model}_{method}.parquet", index=False)
        curve = reliability(y, p).reset_index()
        curve["model"], curve["method"] = model, method
        curves.append(curve)
        for fold in [-1, 0, 1, 2, 3, 4]:
            fold_mask = (
                np.ones(len(frame), dtype=bool) if fold == -1 else frame.fold.to_numpy() == fold
            )
            for scope, mask in scopes.items():
                use = fold_mask & mask
                if not use.any():
                    continue
                probability_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "fold": fold,
                        "scope": scope,
                        **probability_metrics(y[use], p[use], baseline[use]),
                    }
                )
            for policy in POLICIES:
                use = fold_mask
                alarm_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "fold": fold,
                        "policy": policy,
                        **alarm_metrics(frame[use], flags[model, method, policy][use]),
                    }
                )
    prob = pd.DataFrame(probability_rows)
    alarm = pd.DataFrame(alarm_rows)
    prob.to_csv(out / "probability_metrics.csv", index=False)
    alarm.to_csv(out / "alarm_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "thresholds.csv", index=False)
    pd.DataFrame(roles).to_csv(out / "roles.csv", index=False)
    curve_table = pd.concat(curves, ignore_index=True)
    curve_table.to_csv(out / "reliability_bins.csv", index=False)
    prob[prob.fold >= 0].groupby(["model", "method", "scope"])[
        ["AUROC", "AUPRC", "Brier", "Brier_skill", "log_loss", "ECE_10_equal_width"]
    ].agg(["mean", "std"]).to_csv(out / "probability_fold_summary.csv")
    alarm[alarm.fold >= 0].groupby(["model", "method", "policy"])[
        [
            "row_sensitivity",
            "row_PPV",
            "row_specificity",
            "nonsepsis_patient_false_alarm_rate",
            "early_recall_6to12h",
            "pre_onset_recall_0to12h",
            "Utility",
        ]
    ].agg(["mean", "std"]).to_csv(out / "alarm_fold_summary.csv")
    if make_plots:
        # Colab's inline backend belongs to its kernel, not our isolated Python.
        # Override it before import: matplotlib validates MPLBACKEND during import.
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for model, ax in zip(["v2", "v3"], axes, strict=True):
            for method in METHODS:
                c = curve_table[(curve_table.model == model) & (curve_table.method == method)]
                ax.plot(c.mean_prediction, c.observed_rate, marker="o", label=method)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set(
                xlabel="Mean predicted probability",
                ylabel="Observed positive fraction",
                title=f"{model}: patient-held-out calibration",
            )
            ax.legend()
        fig.savefig(out / "calibration.png", dpi=150)
        for ax in axes:
            ax.set(xlim=(0, 0.1), ylim=(0, 0.1))
        fig.savefig(out / "calibration_zoom.png", dpi=150)
        plt.close(fig)
    return prob, alarm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO / "artifacts/patient_split.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.v2.resolve() == args.v3.resolve():
        parser.error("v2 and v3 must be different experiment directories")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new output directory; existing comparison results are preserved")
    frames, hashes = [], {}
    for model, directory in [("v2", args.v2), ("v3", args.v3)]:
        summary = json.loads((directory / "summary.json").read_text("utf-8"))
        if summary.get("complete_5fold") is not True or summary.get("test_evaluated") is not False:
            raise ValueError(f"{model}: expected completed five-fold development predictions only")
        path = directory / "oof_predictions.parquet"
        loaded = pd.read_parquet(path)
        counts = {int(f["fold"]): int(f["all"]["rows"]) for f in summary["folds"]}
        if set(counts) != set(range(5)) or loaded.groupby("fold").size().to_dict() != counts:
            raise ValueError(f"{model}: OOF row counts do not match the five-fold summary")
        frames.append(loaded)
        hashes[model] = {
            "run": directory.name,
            "oof_sha256": sha256(path),
            "summary_sha256": sha256(directory / "summary.json"),
        }
    manifest = pd.read_csv(args.manifest)
    frame = paired_oof(*frames, manifest)
    compare(frame, args.out, args.seed)
    metadata = {
        "complete": True,
        "seed": args.seed,
        "rows": len(frame),
        "patients": int(frame.Patient_ID.nunique()),
        "inputs": hashes,
        "manifest_sha256": sha256(args.manifest),
        "script_sha256": sha256(__file__),
        "utility_module_sha256": sha256(REPO / "data/src/sews_eval.py"),
        "test_evaluated": False,
        "gru_retrained": False,
        "python": sys.version,
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "design": "Rotate calibrate/tune/evaluate patient groups inside each original fold.",
        "limitation": (
            "GRU epoch/hyperparameter selection already used dev labels. Development analysis only."
        ),
        "alarms": "Hourly flags, no cooldown or first-alarm suppression; not alarm episodes.",
        "timing": "Onset = first positive label + 6h; starts-positive cases excluded from timing.",
        "deployment": "Split-specific calibrators/thresholds; not a final deployment bundle.",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    print(f"Comparison complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
