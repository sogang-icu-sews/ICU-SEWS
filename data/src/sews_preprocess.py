"""SEWS fold별 전처리 모듈: 스케일링·대치 파라미터(학습 행에서만 학습) + LSTM 입력 창.

- train_mask(em, k)                 : fold k의 학습 행 (dev & fold != k).
                                      k='final'이면 dev 전체. test는 항상 제외
- fit_tabular(F_train, cfg)         : 특징별 중앙값·IQR (학습 행만)
- apply_tabular(F, params, cols)    : (x - 중앙값) / IQR → ±clip → NaN은 0
                                      (= 학습 중앙값으로 대치)
- fit_seq(values_train, cfg)        : LSTM 입력 변수별 중앙값·IQR
                                      (학습 행의 실제 측정값만)
- seq_windows(values, row_start, rows, seq_params, cfg)
                                    : 행 r마다 최근 24행 창 (B, 24, ·).
                                      학습(배치)과 API(1개)가 같은 함수를 씀
- seq_window_from_frame(win, seq_params, cfg)
                                    : API용. 정제된 최근 ≤24행 DataFrame → 창 1개

LightGBM은 결측을 직접 처리하므로 이 파라미터가 필요 없다 (features.parquet 원본 사용).
"""

import numpy as np
import pandas as pd

WIN = 24


# ------------------------------------------------------------ 학습 행 선택
def train_mask(em: pd.DataFrame, k) -> np.ndarray:
    """Rows used to fit fold-k parameters: dev outside fold k ('final' = all dev)."""
    dev = (em["split"] == "dev").to_numpy()
    if k == "final":
        return dev
    return dev & (em["fold"].to_numpy() != int(k))


# ------------------------------------------------------------ 표 형태 (LR, MLP 등)
def _robust(x: np.ndarray) -> tuple[float, float]:
    """Median and IQR of observed values (IQR 0 → std, std 0 → 1)."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    q75, q25 = np.percentile(x, [75, 25])
    s = float(q75 - q25)
    if not s > 0:
        s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if not s > 0:
        s = 1.0
    return med, s


def fit_tabular(F_train: pd.DataFrame, cfg: dict) -> dict:
    mp = cfg["model_prep"]["tabular"]
    center, scale = {}, {}
    for c in F_train.columns:
        if c in mp["passthrough"]:
            center[c], scale[c] = 0.0, 1.0
        else:
            center[c], scale[c] = _robust(F_train[c].to_numpy(np.float64))
    return {
        "columns": list(F_train.columns),
        "center": center,
        "scale": scale,
        "clip": mp["clip"],
        "passthrough": list(mp["passthrough"]),
        "n_rows": int(len(F_train)),
    }


def apply_tabular(F: pd.DataFrame, params: dict, cols=None) -> np.ndarray:
    cols = list(cols) if cols is not None else params["columns"]
    c = np.array([params["center"][k] for k in cols])
    s = np.array([params["scale"][k] for k in cols])
    X = (F[cols].to_numpy(np.float64) - c) / s
    X = np.clip(X, -params["clip"], params["clip"])
    return np.nan_to_num(X, nan=0.0).astype(np.float32)


# ------------------------------------------------------------ 시퀀스 (LSTM, GRU 등)
def seq_spec(cfg: dict) -> dict:
    sq = cfg["model_prep"]["sequence"]
    vars_ = sq["vars"]
    mvars = sq["mask_delta_vars"]
    return {
        "vars": vars_,
        "mask_vars": mvars,
        "mask_idx": [vars_.index(v) for v in mvars],
    }


def fit_seq(values_train: np.ndarray, cfg: dict) -> dict:
    spec = seq_spec(cfg)
    center, scale = [], []
    for j in range(values_train.shape[1]):
        m, s = _robust(values_train[:, j].astype(np.float64))
        center.append(m)
        scale.append(s)
    return {
        "vars": spec["vars"],
        "center": center,
        "scale": scale,
        "clip": cfg["model_prep"]["sequence"]["clip"],
        "n_rows": int(len(values_train)),
    }


def seq_windows(
    values,
    row_start: np.ndarray,
    rows: np.ndarray,
    sp: dict,
    cfg: dict,
    return_raw: bool = False,
) -> dict:
    """Windows for rows r: steps t-23..t (left-padded with empty rows before start).

    x     : 창 안 forward fill 값을 스케일 (아직 측정 전이면 0)   (B, 24, V)
    mask  : 그 시간에 실제로 측정했는가 (mask 대상 변수만)          (B, 24, M)
    delta : 창 안 마지막 측정 이후 시간 / 24 (창 안 측정 전이면 1)   (B, 24, M)
    """
    spec = seq_spec(cfg)
    rows = np.asarray(rows, dtype=np.int64)
    steps = np.arange(WIN)
    idx = rows[:, None] - (WIN - 1 - steps)[None, :]  # (B, 24)
    valid = idx >= np.asarray(row_start)[rows][:, None]  # 같은 환자의 기록만
    raw = np.asarray(values[np.clip(idx, 0, None).ravel()], dtype=np.float32).reshape(
        len(rows), WIN, -1
    )
    raw[~valid] = np.nan
    obs = ~np.isnan(raw)
    li = np.maximum.accumulate(np.where(obs, steps[None, :, None], -1), axis=1)  # 마지막 측정 위치
    ff = np.take_along_axis(raw, np.clip(li, 0, None), axis=1)
    ff[li < 0] = np.nan
    delta = np.where(li >= 0, steps[None, :, None] - li, WIN).astype(np.float32)
    c = np.asarray(sp["center"], dtype=np.float32)
    s = np.asarray(sp["scale"], dtype=np.float32)
    x = np.nan_to_num(np.clip((ff - c) / s, -sp["clip"], sp["clip"]), nan=0.0).astype(np.float32)
    mi = spec["mask_idx"]
    out = {
        "x": x,
        "mask": obs[..., mi].astype(np.float32),
        "delta": delta[..., mi] / WIN,
    }
    if return_raw:
        out["raw"], out["ff"], out["delta_h"] = raw, ff, delta
    return out


def seq_window_from_frame(win: pd.DataFrame, sp: dict, cfg: dict, return_raw: bool = False) -> dict:
    """API path: one cleaned window (<=24 rows, oldest first) -> seq_windows (B=1)."""
    v = win.tail(WIN)[seq_spec(cfg)["vars"]].to_numpy(np.float32)
    return seq_windows(
        v, np.zeros(len(v), dtype=np.int64), np.array([len(v) - 1]), sp, cfg, return_raw
    )


def iter_batches(
    rows: np.ndarray, batch_size: int, values, row_start, sp, cfg, shuffle=True, seed=0
):
    """Yield (row_ids, windows) batches for training loops (plain numpy)."""
    rows = np.asarray(rows)
    order = np.random.default_rng(seed).permutation(len(rows)) if shuffle else np.arange(len(rows))
    for i in range(0, len(rows), batch_size):
        b = np.sort(rows[order[i : i + batch_size]])
        yield b, seq_windows(values, row_start, b, sp, cfg)
