"""전처리(sews_preprocess) 검사: 학습 행만으로 학습, LSTM 창 = API 창, 미래 미사용."""

import numpy as np
import pandas as pd
import pytest
import sews_features as sf
import sews_preprocess as pp


@pytest.fixture(scope="module")
def clean_df(cfg, raw):
    out, _ = sf.clean(raw, cfg)
    return out


def test_train_mask_excludes_fold_and_test():
    em = pd.DataFrame(
        {
            "split": ["dev"] * 6 + ["test"] * 2,
            "fold": [0, 0, 1, 1, 2, 2, -1, -1],
        }
    )
    assert pp.train_mask(em, 0).tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert pp.train_mask(em, "final").tolist() == [1] * 6 + [0, 0]


def test_tabular_params_come_from_train_rows_only(cfg, clean_df):
    F = sf.build_features(clean_df, cfg)
    train = F.iloc[: len(F) // 2]
    params = pp.fit_tabular(train, cfg)
    F_shifted = F.copy()
    F_shifted.iloc[len(F) // 2 :] += 1000.0  # 검증 행이 바뀌어도
    assert pp.fit_tabular(F_shifted.iloc[: len(F) // 2], cfg) == params  # 파라미터 동일
    X = pp.apply_tabular(F, params)
    assert X.dtype == np.float32 and not np.isnan(X).any()  # NaN → 0
    assert np.abs(X).max() <= params["clip"]
    for c in params["passthrough"]:
        assert params["center"][c] == 0.0 and params["scale"][c] == 1.0


def _seq_inputs(cfg, clean_df):
    vars_ = pp.seq_spec(cfg)["vars"]
    values = clean_df[vars_].to_numpy(np.float32)
    pid = clean_df["Patient_ID"].to_numpy()
    first = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1]
    row_start = np.repeat(first, np.diff(np.r_[first, len(pid)]))
    return values, row_start


def test_seq_windows_equal_api_windows(cfg, clean_df):
    values, row_start = _seq_inputs(cfg, clean_df)
    sp = pp.fit_seq(values, cfg)
    rows = np.arange(len(clean_df))
    W = pp.seq_windows(values, row_start, rows, sp, cfg)
    assert W["x"].shape == (len(rows), 24, values.shape[1])
    for r in rows[::3]:
        win = clean_df.iloc[row_start[r] : r + 1]
        one = pp.seq_window_from_frame(win, sp, cfg)
        for k in ("x", "mask", "delta"):
            assert np.array_equal(one[k][0], W[k][r]), (r, k)


def test_seq_windows_stay_inside_patient_and_past(cfg, clean_df):
    values, row_start = _seq_inputs(cfg, clean_df)
    sp = pp.fit_seq(values, cfg)
    rows = np.arange(len(clean_df))
    base = pp.seq_windows(values, row_start, rows, sp, cfg)
    for r in rows[::5]:
        v2 = values.copy()
        v2[r + 1 :] = 50.0  # 미래 행(다른 환자 포함)을 바꿔도
        v2[: row_start[r]] = -50.0  # 이전 환자 행을 바꿔도
        W = pp.seq_windows(v2, row_start, np.array([r]), sp, cfg)
        for k in ("x", "mask", "delta"):
            assert np.array_equal(W[k][0], base[k][r]), (r, k)
    n_steps = rows - row_start + 1  # 기록 시작 전 칸은 측정 없음
    first_rows = rows[n_steps < 24]
    for r in first_rows[::5]:
        pad = 24 - n_steps[r]
        assert base["mask"][r, :pad].sum() == 0
        assert np.all(base["delta"][r, :pad] == 1.0)
