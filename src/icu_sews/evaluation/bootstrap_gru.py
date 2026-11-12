"""Patient-cluster percentile bootstrap of an already frozen test prediction set.

Patient multiplicities weight every hour and every patient-level contribution.
No refitting, threshold optimization or resampling of individual hours occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from icu_sews.evaluation.compare_gru import sha256, utility_module


def ratio(a, b):
    return float(a / b) if b else np.nan


class PatientMetrics:
    """Cache score order/ties and additive patient statistics for exact weighted metrics."""

    def __init__(self, frame, baseline, include_alarms=True):
        f = frame.sort_values(["Patient_ID", "ICULOS"]).reset_index(drop=True)
        self.patients = f.Patient_ID.unique()
        ids = pd.Index(self.patients).get_indexer(f.Patient_ID)
        self.n = len(self.patients)

        def sums(x):
            return np.bincount(ids, weights=np.asarray(x, dtype=float), minlength=self.n)

        y = f.SepsisLabel.to_numpy(dtype=int)
        p = f.p_calibrated.to_numpy(dtype=float)
        self.additive = {
            "rows": sums(np.ones(len(f))),
            "positives": sums(y),
            "predictions": sums(p),
            "Brier": sums((p - y) ** 2),
            "baseline_Brier": sums((baseline - y) ** 2),
        }
        clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        self.additive["log_loss"] = sums(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))
        bins = np.minimum((p * 10).astype(int), 9)
        self.ece_errors = np.stack([sums((p - y) * (bins == b)) for b in range(10)])
        order = np.argsort(-p, kind="stable")
        scores = p[order]
        self.sorted_ids, self.sorted_y = ids[order], y[order]
        self.starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
        self.include_alarms = include_alarms
        if not include_alarms:
            return
        a = f.alarm.to_numpy(dtype=bool)
        pt = f.groupby("Patient_ID", sort=False).agg(
            y=("SepsisLabel", "max"),
            first=("SepsisLabel", "first"),
            alarm=("alarm", "max"),
        )
        onset = f.ICULOS.where(f.SepsisLabel == 1).groupby(f.Patient_ID).transform("min") + 6
        rel = f.ICULOS.to_numpy() - onset.to_numpy()
        before = a & (rel >= -12) & (rel < 0)
        early = a & (rel >= -12) & (rel <= -6)
        eligible = ((pt.y == 1) & (pt["first"] == 0)).to_numpy()
        lead = pd.Series(np.where(before, -rel, np.nan)).groupby(f.Patient_ID, sort=False).max()
        self.leads = np.where(eligible, lead.to_numpy(), np.nan)
        self.patient_stats = {
            "nonseptic": (pt.y.to_numpy() == 0).astype(float),
            "septic": pt.y.to_numpy(dtype=float),
            "eligible": eligible.astype(float),
            "excluded": ((pt.y == 1) & (pt["first"] == 1)).to_numpy(dtype=float),
            "false_patient": ((pt.y == 0) & pt.alarm).to_numpy(dtype=float),
            "septic_alarm": ((pt.y == 1) & pt.alarm).to_numpy(dtype=float),
            "early": (eligible & (sums(early) > 0)).astype(float),
            "before": (eligible & (sums(before) > 0)).astype(float),
        }
        nonseptic = pt.y.to_numpy()[ids] == 0
        self.additive.update(
            {
                "TP": sums(a & (y == 1)),
                "FP": sums(a & (y == 0)),
                "TN": sums(~a & (y == 0)),
                "FN": sums(~a & (y == 1)),
                "alarm_rows": sums(a),
                "nonseptic_rows": sums(nonseptic),
                "false_nonseptic_rows": sums(a & nonseptic),
            }
        )
        ev = utility_module()
        idx = ev._time_index(f)
        _, utility_rel, septic = idx
        u0 = ev.row_utility(f, np.zeros(len(f), dtype=bool), idx)
        observed = ev.row_utility(f, a, idx)
        best = ev.row_utility(f, septic & (utility_rel >= -12) & (utility_rel <= 3), idx)
        self.additive["utility_gain"] = sums(observed - u0)
        self.additive["utility_denominator"] = sums(best - u0)

    def evaluate(self, counts):
        counts = np.asarray(counts, dtype=int)
        if counts.shape != (self.n,) or (counts < 0).any():
            raise ValueError("Expected one nonnegative multiplicity per patient")
        totals = {k: float(np.sum(v * counts)) for k, v in self.additive.items()}
        n = totals["rows"]
        result = {k: ratio(totals[k], n) for k in ["Brier", "baseline_Brier", "log_loss"]}
        result.update(
            prevalence=ratio(totals["positives"], n),
            mean_prediction=ratio(totals["predictions"], n),
            Brier_skill=1 - ratio(totals["Brier"], totals["baseline_Brier"]),
            ECE_10_equal_width=ratio(np.abs((self.ece_errors * counts).sum(axis=1)).sum(), n),
        )
        weights = counts[self.sorted_ids]
        pos = np.add.reduceat(weights * self.sorted_y, self.starts)
        neg = np.add.reduceat(weights * (1 - self.sorted_y), self.starts)
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        result["AUROC"] = ratio(np.sum((tp - pos / 2) * neg), tp[-1] * fp[-1])
        precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=(tp + fp) > 0)
        result["AUPRC"] = ratio(np.sum(pos * precision), tp[-1]) if fp[-1] else np.nan
        if not self.include_alarms:
            return result
        pt = {k: float(np.sum(v * counts)) for k, v in self.patient_stats.items()}
        result.update(
            {
                "row_sensitivity": ratio(totals["TP"], totals["TP"] + totals["FN"]),
                "row_PPV": ratio(totals["TP"], totals["TP"] + totals["FP"]),
                "row_specificity": ratio(totals["TN"], totals["TN"] + totals["FP"]),
                "nonsepsis_patient_false_alarm_rate": ratio(pt["false_patient"], pt["nonseptic"]),
                "sepsis_patient_ever_alarm_rate": ratio(pt["septic_alarm"], pt["septic"]),
                "early_recall_6to12h": ratio(pt["early"], pt["eligible"]),
                "pre_onset_recall_0to12h": ratio(pt["before"], pt["eligible"]),
                "Utility": ratio(totals["utility_gain"], totals["utility_denominator"]),
                "alarm_hours_per100h": 100 * ratio(totals["alarm_rows"], n),
                "false_alarm_hours_per100_nonsepsis_h": 100
                * ratio(totals["false_nonseptic_rows"], totals["nonseptic_rows"]),
            }
        )
        valid = np.isfinite(self.leads)
        leads = np.repeat(self.leads[valid], counts[valid])
        result["median_lead_hours_detected"] = float(np.median(leads)) if len(leads) else np.nan
        return result


def bootstrap(frame, baseline, repetitions=2000, seed=42):
    if repetitions < 2:
        raise ValueError("At least two replicates required")
    patients = pd.Index(sorted(frame.Patient_ID.unique()))
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "set_A": frame["set"] == "A",
        "set_B": frame["set"] == "B",
        "ICULOS_le60": frame.ICULOS <= 60,
    }
    evaluators = {
        scope: PatientMetrics(frame[mask], baseline, scope != "ICULOS_le60")
        for scope, mask in masks.items()
        if mask.any()
    }
    indexes = {scope: patients.get_indexer(ev.patients) for scope, ev in evaluators.items()}
    points = {scope: ev.evaluate(np.ones(ev.n, dtype=int)) for scope, ev in evaluators.items()}
    rng = np.random.default_rng(seed)
    records = []
    for rep in range(repetitions):
        # Ordinary cluster bootstrap: prevalence and hospital mix can vary per replicate.
        counts = np.bincount(
            rng.integers(len(patients), size=len(patients)), minlength=len(patients)
        )
        for scope, ev in evaluators.items():
            records.append(
                {"replicate": rep, "scope": scope, **ev.evaluate(counts[indexes[scope]])}
            )
        if (rep + 1) % 100 == 0 or rep + 1 == repetitions:
            print(f"Patient bootstrap {rep + 1}/{repetitions}", flush=True)
    samples = pd.DataFrame(records)
    rows = []
    for scope, point in points.items():
        selected = samples[samples.scope == scope]
        for metric, estimate in point.items():
            valid = selected[metric].dropna()
            lower, upper = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_replicates": len(valid),
                    "requested_replicates": repetitions,
                }
            )
    return pd.DataFrame(rows), samples


def load_predictions(directory, manifest_path):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    summary, frozen = read("test_summary.json"), read("frozen.json")
    if summary.get("complete") is not True or summary.get("test_evaluated") is not True:
        raise ValueError("A completed frozen test evaluation is required")
    if sha256(directory / "frozen.json") != summary["frozen_sha256"]:
        raise ValueError("Frozen run hash mismatch")
    for name in ["platt.json", "threshold.json"]:
        if sha256(directory / name) != frozen["artifacts"][name]:
            raise ValueError(f"Frozen file changed: {name}")
    path = directory / "test_predictions.parquet"
    if sha256(path) != summary["artifacts"][path.name]:
        raise ValueError("Test predictions hash mismatch")
    f = pd.read_parquet(path)
    required = [
        "row_id",
        "Patient_ID",
        "ICULOS",
        "set",
        "split",
        "fold",
        "SepsisLabel",
        "p_calibrated",
        "alarm",
    ]
    if not set(required).issubset(f) or f[required].isna().any().any() or f.empty:
        raise ValueError("Missing required prediction columns/values")
    if len(f) != summary["rows"] or f.Patient_ID.nunique() != summary["patients"]:
        raise ValueError("Test dimensions differ from completed evaluation")
    if f.row_id.duplicated().any() or f.duplicated(["Patient_ID", "ICULOS"]).any():
        raise ValueError("Duplicate test observations")
    if not (f.split == "test").all() or not (f.fold == -1).all():
        raise ValueError("Only original test predictions are allowed")
    if not f.SepsisLabel.isin([0, 1]).all() or not f.p_calibrated.between(0, 1).all():
        raise ValueError("Invalid labels or probabilities")
    threshold = read("threshold.json")["threshold"]
    if not f.alarm.isin([False, True]).all() or not np.array_equal(
        f.alarm, f.p_calibrated >= threshold
    ):
        raise ValueError("Alarm flags do not match the frozen threshold")
    f = f.sort_values(["Patient_ID", "ICULOS"]).reset_index(drop=True)
    group = f.groupby("Patient_ID")
    if (group["set"].nunique() != 1).any():
        raise ValueError("Patient crosses hospitals")
    actual = group.agg(set=("set", "first"), sepsis=("SepsisLabel", "max")).sort_index()
    manifest = pd.read_csv(manifest_path)
    expected = (
        manifest[manifest.split == "test"].set_index("Patient_ID")[["set", "sepsis"]].sort_index()
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    same = np.diff(f.Patient_ID.to_numpy()) == 0
    if (np.diff(f.ICULOS.to_numpy())[same] != 1).any() or (
        np.diff(f.SepsisLabel.to_numpy(dtype=int))[same] < 0
    ).any():
        raise ValueError("Incomplete or non-monotone patient records")
    return f, read("platt.json")["baseline_probability"], threshold


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "artifacts/patient_split.csv",
    )
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output must be new; original evaluation is never overwritten")
    f, baseline, threshold = load_predictions(args.run, args.manifest)
    print(
        f"Loaded {f.Patient_ID.nunique()} patients / {len(f)} hours; fixed threshold {threshold}",
        flush=True,
    )
    ci, samples = bootstrap(f, baseline, args.repetitions, args.seed)
    # Require the point estimates to reproduce the previously reported frozen test scores.
    summary = json.loads((args.run / "test_summary.json").read_text(encoding="utf-8"))
    expected = {r["scope"]: dict(r) for r in summary["probability"] if r["method"] == "platt"}
    for row in summary["alarms"]:
        expected[row["scope"]].update(row)
    for row in ci.itertuples(index=False):
        reference = expected[row.scope].get(row.metric)
        if reference is not None and not np.isclose(row.estimate, reference, rtol=1e-9, atol=1e-10):
            raise ValueError(f"Point estimate mismatch: {row.scope}/{row.metric}")
    args.out.mkdir(parents=True, exist_ok=True)
    ci.to_csv(args.out / "bootstrap_95ci.csv", index=False)
    samples.to_csv(args.out / "bootstrap_replicates.csv", index=False)
    metadata = {
        "complete": True,
        "method": "ordinary patient-cluster bootstrap, percentile 2.5/97.5",
        "repetitions": args.repetitions,
        "seed": args.seed,
        "threshold": threshold,
        "patients": int(f.Patient_ID.nunique()),
        "rows": len(f),
        "predictions_sha256": sha256(args.run / "test_predictions.parquet"),
        "summary_sha256": sha256(args.run / "test_summary.json"),
        "script_sha256": sha256(__file__),
        "limitations": [
            "Conditional on the trained GRU, Platt and threshold; no retraining uncertainty",
            "Patients sampled as independent clusters; does not estimate new-hospital uncertainty",
            "Subgroups use the same full-cohort draws; intervals are not simultaneous",
            "Undefined replicates are omitted per metric and their valid counts are reported",
        ],
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(ci[ci.scope == "all"].to_string(index=False), flush=True)
    print(f"Saved confidence intervals: {args.out}", flush=True)


if __name__ == "__main__":
    main()
