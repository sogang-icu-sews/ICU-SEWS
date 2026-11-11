"""Calibrator/threshold isolation, exact threshold ties and timing definitions."""

import numpy as np
import pandas as pd
import pytest

from icu_sews.evaluation.compare_gru import (
    alarm_metrics,
    assign_groups,
    calibrated,
    compare,
    fit_calibrator,
    paired_oof,
    probability_metrics,
    rate_threshold,
    role_masks,
    select_thresholds,
    utility_module,
    utility_threshold,
)


@pytest.fixture
def data():
    rows, patients = [], []
    rng = np.random.default_rng(12)
    patient = 0
    for fold in range(5):
        for site in ["A", "B"]:
            for positive in [0, 1]:
                for _ in range(3):
                    patient += 1
                    patients.append([patient, site, positive, "dev", fold])
                    for hour in range(20):
                        y = int(positive and hour >= 10)
                        p = np.clip(0.12 + 0.3 * y + rng.normal(0, 0.2), 0.001, 0.999)
                        rows.append([len(rows), patient, hour + 1, site, fold, y, p])
    frame = pd.DataFrame(
        rows, columns=["row_id", "Patient_ID", "ICULOS", "set", "fold", "SepsisLabel", "p_gru"]
    )
    manifest = pd.DataFrame(patients, columns=["Patient_ID", "set", "sepsis", "split", "fold"])
    return frame, manifest


def test_pairing_aligns_rows_and_rejects_test_patients(data):
    frame, manifest = data
    b = frame.sample(frac=1, random_state=4).copy()
    paired = paired_oof(frame, b, manifest)
    np.testing.assert_array_equal(paired.v2, paired.v3)
    bad = frame.copy()
    bad.loc[0, "SepsisLabel"] = 1
    with pytest.raises(ValueError):
        paired_oof(frame, bad, manifest)
    altered = manifest.copy()
    altered.loc[0, "split"] = "test"
    with pytest.raises(ValueError):
        paired_oof(frame, b, altered)


def test_evaluation_labels_cannot_change_calibrator_or_threshold(data):
    frame, manifest = data
    f = paired_oof(frame, frame, manifest)
    groups = assign_groups(f)
    seen = np.zeros(len(f), dtype=int)
    for fold in range(5):
        for rotation in range(3):
            cal, tune, test = role_masks(f, groups, fold, rotation)
            ids = [set(f.loc[m, "Patient_ID"]) for m in [cal, tune, test]]
            assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            assert set(f.loc[cal | tune | test, "fold"]) == {fold}
            seen[test] += 1
    assert (seen == 1).all()
    cal, tune, test = role_masks(f, groups, 0, 0)
    polluted = f.copy()
    polluted.loc[test, "SepsisLabel"] = 1 - polluted.loc[test, "SepsisLabel"]
    a = fit_calibrator(f.v2.to_numpy()[cal], f.SepsisLabel.to_numpy()[cal], "platt")
    b = fit_calibrator(polluted.v2.to_numpy()[cal], polluted.SepsisLabel.to_numpy()[cal], "platt")
    np.testing.assert_array_equal(a.coef_, b.coef_)
    p = calibrated(f.v2.to_numpy()[tune], a, "platt")
    assert select_thresholds(f[tune], p) == select_thresholds(polluted[tune], p)


def test_rate_threshold_handles_ties_and_zero_budget():
    scores = np.array([0.9] * 4 + [0.2] * 16)
    threshold = rate_threshold(scores, 0.1)
    assert (scores >= threshold).mean() <= 0.1
    assert not (scores >= rate_threshold(scores, 0)).any()


def test_float32_isotonic_ties_respect_false_alarm_budget():
    raw = np.array([0.1] * 10 + [0.8] * 10, dtype=np.float32)
    labels = np.array([0] * 9 + [1] + [0] * 5 + [1] * 5)
    fitted = fit_calibrator(raw, labels, "isotonic")
    p = calibrated(raw, fitted, "isotonic")
    assert p.dtype == np.float64
    negatives = p[labels == 0]
    threshold = rate_threshold(negatives, 0.05)
    assert (negatives >= threshold).mean() <= 0.05


def test_utility_optimum_matches_exhaustive_search(data):
    frame, _ = data
    frame = frame[frame.fold == 0]
    p = frame.p_gru.to_numpy()
    threshold = utility_threshold(frame, p)
    ev = utility_module()
    candidates = np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]
    scores = [ev.normalized_utility(frame, p >= t) for t in candidates]
    assert ev.normalized_utility(frame, p >= threshold) == pytest.approx(max(scores))


def test_alarm_timing_excludes_starts_positive_and_counts_patients():
    rows = []
    for pid in [1, 2, 3]:
        for hour in range(20):
            label = int(pid == 1 and hour >= 10 or pid == 3)
            rows.append([pid, hour + 1, label])
    frame = pd.DataFrame(rows, columns=["Patient_ID", "ICULOS", "SepsisLabel"])
    alarm = np.zeros(60, dtype=bool)
    alarm[8] = True  # patient 1: onset at hour 17; alarm at hour 9 => eight hours early
    alarm[20:22] = True  # two hourly alarms, one non-septic patient
    alarm[40] = True  # starts-positive patient: exclude from timing only
    result = alarm_metrics(frame, alarm)
    assert result["early_recall_6to12h"] == 1
    assert result["median_lead_hours_detected"] == 8
    assert result["timing_eligible_patients"] == 1
    assert result["timing_excluded_starts_positive"] == 1
    assert result["nonsepsis_patient_false_alarm_rate"] == 1
    assert result["false_alarm_hours_per100_nonsepsis_h"] == 10
    ev = utility_module()
    assert result["Utility"] == pytest.approx(ev.normalized_utility(frame, alarm))
    manual = sum(
        ev.official_utility(g.SepsisLabel.to_numpy(), alarm[g.index])
        for _, g in frame.groupby("Patient_ID")
    )
    assert ev.row_utility(frame, alarm).sum() == pytest.approx(manual)


def test_comparison_outputs_same_rows_and_valid_probabilities(data, tmp_path):
    frame, manifest = data
    second = frame.copy()
    second["p_gru"] = second.p_gru / 2
    f = paired_oof(frame, second, manifest)
    prob, alarms = compare(f, tmp_path / "comparison", make_plots=True)
    assert len(prob[(prob.fold == -1) & (prob.scope == "all")]) == 6
    assert len(alarms[alarms.fold == -1]) == 18
    for model in ["v2", "v3"]:
        for method in ["raw", "platt", "isotonic"]:
            saved = pd.read_parquet(tmp_path / f"comparison/predictions_{model}_{method}.parquet")
            assert len(saved) == len(f)
            assert saved.row_id.is_unique
            assert saved.p_calibrated.between(0, 1).all()
    thresholds = pd.read_csv(tmp_path / "comparison/thresholds.csv")
    row = thresholds[thresholds.policy == "row_specificity95"]
    patient = thresholds[thresholds.policy == "patient_false_alarm10"]
    assert (row.tuning_specificity >= 0.95 - 1e-12).all()
    assert (patient.tuning_patient_false_alarm_rate <= 0.1 + 1e-12).all()
    assert (tmp_path / "comparison/calibration.png").is_file()


def test_brier_skill_reference_is_explicit():
    y = np.array([0, 0, 1, 0])
    base = np.full(4, 0.2)
    result = probability_metrics(y, base, base)
    assert result["Brier_skill"] == 0
