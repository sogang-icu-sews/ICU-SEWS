"""평가(sews_eval) 검사: 벡터화 Utility = PhysioNet 공식 함수, 정규화, 보정, 지표."""

import numpy as np
import pandas as pd
import pytest
import sews_eval as ev


@pytest.fixture(scope="module")
def em(raw) -> pd.DataFrame:
    return raw[["Patient_ID", "ICULOS", "SepsisLabel"]].reset_index(drop=True)


@pytest.mark.parametrize("rate", [0.0, 0.02, 0.3, 0.9, 1.0])
def test_vectorized_utility_equals_official(em, rate):
    rng = np.random.default_rng(int(rate * 100))
    pred = rng.random(len(em)) < rate
    u = ev.row_utility(em, pred)
    for _, g in em.groupby("Patient_ID", sort=False):
        ix = g.index.to_numpy()
        ref = ev.official_utility(g["SepsisLabel"].to_numpy(), pred[ix])
        assert np.isclose(u[ix].sum(), ref, rtol=0, atol=1e-12)


def test_normalized_utility_bounds(em):
    assert em["SepsisLabel"].any()  # 패혈증 환자가 있어야 의미가 있다
    _, d, septic = ev._time_index(em)
    with np.errstate(invalid="ignore"):
        best = septic & (d >= -12) & (d <= 3)
    assert ev.normalized_utility(em, best) == 1.0  # 최적 예측
    assert ev.normalized_utility(em, np.zeros(len(em), bool)) == 0.0  # 아무것도 안 함


def test_label_is_not_shifted_again(em):
    """t_sepsis = 첫 라벨 1 시각 + 6 (공식 라벨이 이미 6시간 앞당겨져 있음)."""
    t, d, septic = ev._time_index(em)
    for _, g in em.groupby("Patient_ID", sort=False):
        lab = g["SepsisLabel"].to_numpy()
        ix = g.index.to_numpy()
        if lab.any():
            first = int(np.argmax(lab))
            assert np.allclose(d[ix], t[ix] - (first + 6))
        else:
            assert not septic[ix].any()


def test_calibration_and_metrics():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)  # 이미 보정된 확률
    assert ev.ece(y, p) < 0.03
    assert ev.ece(y, np.clip(p * 0.2, 0, 1)) > 0.2
    ab = ev.fit_platt(p, y)
    q = ev.apply_platt(p, ab)
    assert np.all((q > 0) & (q < 1))
    assert np.corrcoef(p, q)[0, 1] > 0.99  # 단조 변환
    iso = ev.fit_isotonic(p, y)
    assert np.all(np.diff(iso.predict(np.sort(p))) >= 0)
    assert 0.0 <= ev.sens_at_spec(y, p, 0.95) <= 1.0
