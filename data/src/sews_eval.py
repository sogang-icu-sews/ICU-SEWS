"""SEWS 공용 평가 모듈: 모든 모델(로지스틱 회귀, LightGBM, LSTM, 앙상블)을 같은 기준으로 평가한다.

- evaluate(em, p)                 : AUROC, AUPRC, ICULOS<=60 AUROC, 조기 구간 AUROC, 환자 단위 지표, Brier, ECE
- normalized_utility(em, pred)    : PhysioNet 2019 공식 Utility (정규화)
- threshold_report(em, p, thr)    : 임계값 하나에서의 Utility, 재현율, PPV, 특이도, 환자 단위 경보 비율
- sens_at_spec(y, p, 0.95)        : 특이도 95%에서의 민감도
- fit_platt / apply_platt, fit_isotonic, crossfit_calibrate : 보정 (검증 fold는 보정기 학습에 쓰지 않음)
- crossfit_threshold(em, p)       : fold k의 임계값은 나머지 fold로 고름

Utility 함수는 PhysioNet/CinC Challenge 2019 공식 평가 코드
(https://github.com/physionetchallenges/evaluation-2019, BSD 2-Clause, Copyright (c) 2019 PhysioNet)의
compute_prediction_utility와 같은 정의를 벡터화한 것이다. official_utility()는 원본 반복문을 그대로 옮긴 참조 구현이며
검사(같은 값인지 확인)에만 쓴다.

em(eval_meta)은 환자·시간 순으로 정렬돼 있어야 하고, 넘기는 행은 환자 기록 전체여야 한다(fold 단위 선택은 만족).
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

U = {'dt_early': -12, 'dt_optimal': -6, 'dt_late': 3, 'max_u_tp': 1.0, 'min_u_fn': -2.0, 'u_fp': -0.05, 'u_tn': 0.0}


# ----------------------------------------------------------------------------- Utility (공식 정의)
def official_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Literal port of the official compute_prediction_utility (reference for tests only)."""
    dt_early, dt_optimal, dt_late = U['dt_early'], U['dt_optimal'], U['dt_late']
    max_u_tp, min_u_fn, u_fp, u_tn = U['max_u_tp'], U['min_u_fn'], U['u_fp'], U['u_tn']
    if np.any(labels):
        is_septic, t_sepsis = True, np.argmax(labels) - dt_optimal
    else:
        is_septic, t_sepsis = False, float('inf')
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal
    u = np.zeros(len(labels))
    for t in range(len(labels)):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
    return float(np.sum(u))


def _time_index(em: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: t (0-based row index in the record), d = t - t_sepsis (NaN if non-septic), septic flag."""
    pid = em['Patient_ID'].to_numpy()
    t = em.groupby('Patient_ID', sort=False).cumcount().to_numpy()
    lab = em['SepsisLabel'].to_numpy()
    first = pd.Series(np.where(lab == 1, t, np.inf)).groupby(pid).transform('min').to_numpy()
    septic = np.isfinite(first)
    t_sepsis = np.where(septic, first - U['dt_optimal'], np.nan)
    return t, t - t_sepsis, septic


def row_utility(em: pd.DataFrame, pred: np.ndarray, _idx=None) -> np.ndarray:
    """Vectorized utility per row (same definition as the official function)."""
    _, d, septic = _idx if _idx is not None else _time_index(em)
    pred = np.asarray(pred).astype(bool)
    m1 = U['max_u_tp'] / (U['dt_optimal'] - U['dt_early'])
    b1 = -m1 * U['dt_early']
    m2 = -U['max_u_tp'] / (U['dt_late'] - U['dt_optimal'])
    b2 = -m2 * U['dt_late']
    m3 = U['min_u_fn'] / (U['dt_late'] - U['dt_optimal'])
    b3 = -m3 * U['dt_optimal']
    with np.errstate(invalid='ignore'):
        early = septic & (d <= U['dt_optimal'])
        late = septic & (d > U['dt_optimal']) & (d <= U['dt_late'])
        u = np.zeros(len(pred))
        u = np.where(septic & pred & early, np.maximum(m1 * d + b1, U['u_fp']), u)
        u = np.where(septic & pred & late, m2 * d + b2, u)
        u = np.where(septic & ~pred & late, m3 * d + b3, u)
        u = np.where(~septic & pred, U['u_fp'], u)
        u = np.where(~septic & ~pred, U['u_tn'], u)
    return u


def normalized_utility(em: pd.DataFrame, pred: np.ndarray, _idx=None) -> float:
    """(observed - inaction) / (best - inaction), summed over all patients in em."""
    idx = _idx if _idx is not None else _time_index(em)
    _, d, septic = idx
    with np.errstate(invalid='ignore'):
        best = septic & (d >= U['dt_early']) & (d <= U['dt_late'])
    obs = row_utility(em, pred, idx).sum()
    bst = row_utility(em, best, idx).sum()
    ina = row_utility(em, np.zeros(len(em), bool), idx).sum()
    return float((obs - ina) / (bst - ina))


# ----------------------------------------------------------------------------- 지표
def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def sens_at_spec(y: np.ndarray, p: np.ndarray, spec: float = 0.95) -> float:
    """Highest sensitivity among ROC points with specificity >= spec."""
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate(em: pd.DataFrame, p: np.ndarray) -> dict:
    """Threshold-free metrics on the given rows (complete patient records)."""
    y = em['SepsisLabel'].to_numpy()
    rel = em['rel'].to_numpy()
    le60 = em['iculos_le60'].to_numpy() == 1
    early_pos = (y == 1) & (rel >= -6) & (rel <= -1)                 # 라벨 시작 후 0~5h (발병 전)
    m_early = early_pos | (y == 0)
    pt = pd.DataFrame({'pid': em['Patient_ID'].to_numpy(), 'y': y, 'p': p}).groupby('pid').max()
    return {
        'AUROC': roc_auc_score(y, p),
        'AUPRC': average_precision_score(y, p),
        'AUROC_ICULOS<=60': roc_auc_score(y[le60], p[le60]),
        'AUROC_early(rel -6~-1 vs neg)': roc_auc_score(y[m_early], p[m_early]),
        'Sens@95%Spec': sens_at_spec(y, p, 0.95),
        'patient_AUROC(max)': roc_auc_score(pt['y'], pt['p']),
        'patient_AUPRC(max)': average_precision_score(pt['y'], pt['p']),
        'Brier': float(np.mean((p - y) ** 2)),
        'ECE': ece(y, p),
        'base_rate': float(y.mean()),
    }


def threshold_report(em: pd.DataFrame, p: np.ndarray, thr: float, _idx=None) -> dict:
    """Binary metrics at one threshold (rows and patients)."""
    y = em['SepsisLabel'].to_numpy()
    a = p >= thr
    tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    fn, tn = int((~a & (y == 1)).sum()), int((~a & (y == 0)).sum())
    pt = pd.DataFrame({'pid': em['Patient_ID'].to_numpy(), 'y': y, 'a': a}).groupby('pid').max()
    return {
        'threshold': float(thr), 'utility': normalized_utility(em, a, _idx),
        'recall': tp / max(tp + fn, 1), 'PPV': tp / max(tp + fp, 1), 'specificity': tn / max(tn + fp, 1),
        'alert_rows_%': 100 * a.mean(), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'sepsis_pts_alerted_%': 100 * pt.loc[pt['y'] == 1, 'a'].mean(),
        'nonsepsis_pts_alerted_%': 100 * pt.loc[pt['y'] == 0, 'a'].mean(),
    }


def threshold_grid(p: np.ndarray, n: int = 60) -> np.ndarray:
    """Candidate thresholds: alert rates from 0.1% to 50% of rows."""
    return np.unique(np.quantile(p, 1 - np.geomspace(0.001, 0.5, n)))


def best_utility_threshold(em: pd.DataFrame, p: np.ndarray, grid=None) -> tuple[float, float]:
    idx = _time_index(em)
    grid = threshold_grid(p) if grid is None else grid
    us = [normalized_utility(em, p >= t, idx) for t in grid]
    k = int(np.argmax(us))
    return float(grid[k]), float(us[k])


def crossfit_threshold(em: pd.DataFrame, p: np.ndarray) -> tuple[np.ndarray, list]:
    """Fold k alerts use the utility-optimal threshold chosen on the other folds."""
    folds = em['fold'].to_numpy()
    pred = np.zeros(len(p), bool)
    chosen = []
    for k in np.unique(folds):
        o = folds != k
        thr, _ = best_utility_threshold(em[o], p[o])
        pred[folds == k] = p[folds == k] >= thr
        chosen.append(thr)
    return pred, chosen


# ----------------------------------------------------------------------------- 보정
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling on the logit of p: returns (a, b) for sigmoid(a * logit(p) + b)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    return 1 / (1 + np.exp(-(ab[0] * _logit(p) + ab[1])))


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p, y)


def crossfit_calibrate(em: pd.DataFrame, p: np.ndarray, method: str) -> np.ndarray:
    """Calibrated OOF probabilities: fold k uses a calibrator fit on the other folds only."""
    y = em['SepsisLabel'].to_numpy()
    folds = em['fold'].to_numpy()
    out = np.empty(len(p))
    for k in np.unique(folds):
        o, m = folds != k, folds == k
        if method == 'platt':
            out[m] = apply_platt(p[m], fit_platt(p[o], y[o]))
        else:
            out[m] = fit_isotonic(p[o], y[o]).predict(p[m])
    return out
