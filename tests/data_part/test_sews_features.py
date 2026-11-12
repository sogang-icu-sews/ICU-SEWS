"""특징 추출(sews_features) 검사: 손계산, 학습용 = API용, 미래 미사용, 정제."""

import numpy as np
import pandas as pd
import pytest
import sews_features as sf
from sews_testdata import RAW_COLS


@pytest.fixture(scope="module")
def clean_df(cfg, raw):
    out, _ = sf.clean(raw, cfg)
    return out


@pytest.fixture(scope="module")
def F(cfg, clean_df):
    return sf.build_features(clean_df, cfg)


def test_feature_count_and_no_time_shortcut(cfg, F):
    names = sf.feature_names(cfg)
    assert list(F.columns) == names
    assert len(sf.feature_names(cfg, include_optional=False)) == 125
    assert "ICULOS" not in names
    assert sf.banned_name_hits(names, cfg) == []  # 시간이 쌓이는 특징 금지


def test_clean_rules(cfg, raw, clean_df):
    ranges = cfg["plausible_ranges"]
    for v, (lo, hi) in ranges.items():
        if v in clean_df:
            x = clean_df[v].dropna()
            assert ((x >= lo) & (x <= hi)).all(), v  # 범위 밖 → NaN
    thr = cfg["transforms"]["calcium_ionized_threshold"]
    ion = raw["Calcium"] < thr
    assert clean_df.loc[ion, "Calcium"].isna().all()
    assert (clean_df.loc[ion, "Calcium_ionized"] == raw.loc[ion, "Calcium"]).all()
    ar = cfg["transforms"]["age_recode"]
    assert not (clean_df["Age"] == ar["from"]).any()
    unknown = raw["Unit1"].isna() & raw["Unit2"].isna()
    assert (clean_df.loc[unknown, "Unit"] == "Unknown").all()
    assert len(clean_df) == len(raw)  # 행을 지우지 않는다


def test_hand_calculated_toy(cfg):
    """ICU-3 [2-3]의 손계산 예시: 8시간 기록의 마지막 행."""
    toy = pd.DataFrame(np.nan, index=range(8), columns=RAW_COLS)
    toy["Patient_ID"], toy["ICULOS"], toy["SepsisLabel"] = 999_999, np.arange(1, 9), 0
    toy["Age"], toy["Gender"], toy["Unit1"], toy["Unit2"] = 60.0, 1.0, 1.0, 0.0
    toy.loc[[0, 6, 7], "HR"] = [90, 100, 110]
    toy.loc[[4, 7], "Temp"] = 37.0
    toy.loc[[2, 6], "Platelets"] = [120, 90]
    toy.loc[[0, 7], "MAP"] = [75, 65]
    toy.loc[5, "Creatinine"], toy.loc[5, "BUN"] = 1.0, 20
    toy.loc[3, "Lactate"] = 2.0
    toy.loc[7, "Resp"], toy.loc[7, "SBP"] = 24, 110
    toy, _ = sf.clean(toy, cfg)
    expected = {
        "HR_last": 110, "HR_mean": 100, "HR_median": 100, "HR_min": 90,
        "HR_max": 110, "HR_std": 10, "HR_iqr": 10, "HR_skew": 0, "HR_cv": 0.1,
        "HR_slope": 630 / 258, "HR_delta6h": 20,
        "Temp_std": 0, "Temp_iqr": 0, "Temp_slope": 0, "Temp_dev": 0,
        "Temp_skew": np.nan,  # skew는 측정 3회 이상
        "Platelets_last": 90, "Platelets_delta_prev": -30,
        "sofa_coagulation": 2, "sofa_cardiovascular": 1, "sofa_renal": 0,
        "sofa_liver": np.nan, "sofa_delta": 2,
        "meas_lactate_since_h": 4, "meas_lactate_n24h": 1,
        "meas_fio2_since_h": 24, "meas_fio2_n24h": 0,
        "qsofa_resp": 1, "qsofa_sbp": 0, "qsofa_partial": 1,
        "BUN_Creatinine_ratio": 20, "Unit_MICU": 1, "Unit_Unknown": 0,
    }  # fmt: skip
    got_ref = sf.build_features_window(toy, cfg)  # API용
    got_vec = sf.build_features(toy, cfg).iloc[-1]  # 학습용
    for k, v in expected.items():
        assert np.isclose(got_ref[k], v, equal_nan=True), (k, got_ref[k], v)
        assert np.isclose(got_vec[k], v, equal_nan=True), (k, got_vec[k], v)


def test_training_features_equal_api_features(cfg, clean_df, F):
    """모든 행에서 벡터화 계산(학습용) = 창 단위 참조 계산(API용)."""
    pos = clean_df.groupby("Patient_ID").cumcount().to_numpy()
    for i in range(len(clean_df)):
        win = clean_df.iloc[i - min(pos[i], 23) : i + 1]
        ref = sf.build_features_window(win, cfg).to_numpy(float)
        got = F.iloc[i].to_numpy(float)
        ok = np.isclose(got, ref, rtol=1e-7, atol=1e-9, equal_nan=True)
        assert ok.all(), (i, list(F.columns[~ok]))


def test_window_uses_only_last_24_rows(cfg, clean_df):
    """24행보다 긴 창을 줘도, 앞에 빈 행을 채워도 결과가 같다."""
    pid = clean_df["Patient_ID"].value_counts().idxmax()
    g = clean_df[clean_df["Patient_ID"] == pid]
    assert len(g) > 24
    a = sf.build_features_window(g, cfg).to_numpy(float)
    b = sf.build_features_window(g.tail(24), cfg).to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)
    short = g.head(5)
    padded = short.set_axis(range(19, 24)).reindex(range(24))  # 앞 19행은 빈 행
    assert np.allclose(
        sf.build_features_window(short, cfg).to_numpy(float),
        sf.build_features_window(padded, cfg).to_numpy(float),
        equal_nan=True,
    )


def test_no_future_data(cfg, clean_df, F):
    """시각 t 이후 행을 무작위 값으로 바꾸거나 지워도 t의 특징은 그대로다."""
    rng = np.random.default_rng(1)
    clin = [c for c in cfg["plausible_ranges"] if c in clean_df]
    pos = clean_df.groupby("Patient_ID").cumcount().to_numpy()
    n = clean_df.groupby("Patient_ID")["ICULOS"].transform("size").to_numpy()
    targets = np.flatnonzero(pos < n - 1)[::7]  # 뒤에 행이 남은 시각
    for i in targets:
        pid = clean_df.at[i, "Patient_ID"]
        g = clean_df[clean_df["Patient_ID"] == pid].copy()
        later = g.index > i
        for v in clin:
            lo, hi = cfg["plausible_ranges"][v]
            g.loc[later, v] = rng.uniform(lo, hi, later.sum())
        F_mod = sf.build_features(g.reset_index(drop=True), cfg)
        k = int(pos[i])
        assert np.allclose(
            F_mod.iloc[k].to_numpy(float), F.iloc[i].to_numpy(float), equal_nan=True
        ), i
