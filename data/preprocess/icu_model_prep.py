# %% [markdown]
# SEWS 4단계 (ICU-4): fold별 스케일링·대치 파라미터, LSTM 입력 배열
#
# 원본은 Colab 노트북 ICU-4_model_prep.ipynb이다. 모듈 셀(%%writefile)만 저장소의 src/sews_preprocess.py를 쓰도록 바꿨다.
# 실행: `python preprocess/icu_model_prep.py` (ICU-3 산출물이 있어야 한다)


# %%
import os
import sys
from pathlib import Path

# 실행 환경 (Colab·로컬 공통). 환경 변수로 바꿀 수 있다.
DRIVE_DIR = os.environ.get("SEWS_ROOT", "/content/drive/MyDrive/capston")  # 데이터·산출물 루트
try:
    REPO_DIR = Path(__file__).resolve().parents[1]  # python 으로 실행할 때
except NameError:
    REPO_DIR = Path(os.environ.get("SEWS_REPO", os.getcwd()))  # Jupyter·Colab 셀로 실행할 때
SRC_DIR = str(REPO_DIR / "src")  # 공용 모듈 (src/sews_*.py)


# %%
try:  # Colab에서만 Drive 연결
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    pass


# %% [markdown]
# # ICU-4 모델링 준비 (4단계): fold별 스케일링·대치 + LSTM 입력
#
# **입력**: 3단계 산출물 (`data/features.parquet`, `data/eval_meta.parquet`, `artifacts/feature_registry.json`,
# `artifacts/preprocess_config.json` v3, `src/sews_features.py`) + `data/master.parquet`
#
# **출력** (Drive `capston/` 아래)
# | 파일 | 내용 |
# |---|---|
# | `src/sews_preprocess.py` | fold별 전처리 모듈. 학습과 API가 같은 파일을 import |
# | `artifacts/preprocess_params/fold0~4.json`, `final_dev.json` | 스케일링·대치 파라미터 (fold k는 fold k 밖의 dev 행에서만, final은 dev 전체에서 학습) |
# | `data/seq_values.npy`, `data/seq_row_start.npy`, `artifacts/seq_meta.json` | LSTM 입력용 원시 값 배열 (창은 학습할 때 배치마다 만듦) |
# | `artifacts/preprocess_config.json` | v4 (v3는 `preprocess_config_v3.json`으로 백업) |
# | `artifacts/feature_registry.json` | 4단계 정보(`preprocess`) 추가 |
#
# **모델별로 쓰는 것**
# | 모델 | 입력 |
# |---|---|
# | LightGBM | `features.parquet` 원본 그대로 (NaN 포함, 파라미터 불필요) |
# | 로지스틱 회귀·MLP | `pp.apply_tabular(F, 파라미터['tabular'], model_cols)` |
# | LSTM·GRU | `pp.iter_batches(...)` → `x`(값) · `mask`(측정 여부) · `delta`(마지막 측정 이후 시간) + 정적 입력 |
#
# **실행**: 위에서부터 순서대로 (기본 Colab 런타임). 셀마다 `[OK]` 확인.


import copy
import hashlib
import importlib
import json

# %%
# ===== [4-0] 준비: 3단계 산출물 불러오기 =====
import os
import sys
import time

import numpy as np
import pandas as pd

ART_DIR = f"{DRIVE_DIR}/artifacts"
DATA_DIR = f"{DRIVE_DIR}/data"
PARAM_DIR = f"{ART_DIR}/preprocess_params"
os.makedirs(PARAM_DIR, exist_ok=True)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import sews_features as sf

importlib.reload(sf)

CFG_PATH = f"{ART_DIR}/preprocess_config.json"
with open(CFG_PATH, encoding="utf-8") as f:
    cur_cfg = json.load(f)
if cur_cfg["version"] == 3:  # 처음 실행
    cfg3 = cur_cfg
else:  # 다시 실행: v4를 이미 저장했으면 v3 백업에서 시작
    with open(f"{ART_DIR}/preprocess_config_v3.json", encoding="utf-8") as f:
        cfg3 = json.load(f)
assert cfg3["version"] == 3
with open(f"{ART_DIR}/feature_registry.json", encoding="utf-8") as f:
    registry = json.load(f)
# (저장소 버전) 바이트가 달라도 Drive 원본 모듈과 실행 코드가 같으면 통과 (src/sews_check.py)
from sews_check import module_matches  # noqa: E402

print(
    "[OK] 특징 모듈:",
    module_matches(
        f"{SRC_DIR}/sews_features.py",
        registry["module_sha256"],
        ref_dir=f"{DRIVE_DIR}/src",
    ),
)

F = pd.read_parquet(f"{DATA_DIR}/features.parquet")
em = pd.read_parquet(f"{DATA_DIR}/eval_meta.parquet")
names = [r["name"] for r in registry["features"]]
model_cols = [r["name"] for r in registry["features"] if not r["optional"]]
assert (F["row_id"].to_numpy() == em["row_id"].to_numpy()).all() and list(F.columns[1:]) == names
F = F.drop(columns="row_id")

# LSTM 입력용 원시 값: 3단계와 같은 정제 규칙, 같은 행 순서
master = pd.read_parquet(f"{DATA_DIR}/master.parquet").sort_values(
    ["Patient_ID", "ICULOS"], kind="stable"
)
clean_df, _ = sf.clean(master.reset_index(drop=True), cfg3)
del master
assert (clean_df["Patient_ID"].to_numpy() == em["Patient_ID"].to_numpy()).all()
assert (clean_df["ICULOS"].to_numpy() == em["ICULOS"].to_numpy()).all()
DEV = (em["split"] == "dev").to_numpy()
print(f"[OK] features {F.shape[0]:,}행 x {F.shape[1]}열, eval_meta·정제 데이터와 행 순서 일치")
print(
    f"[OK] 특징 모듈 sha256 = registry 기록 ({registry['module_sha256'][:12]}…), 시작 config v{cfg3['version']}"
)


# %% [markdown]
# ### [4-1] 전처리 모듈 (`src/sews_preprocess.py`)
# 이 셀은 모듈 파일을 Drive에 씁니다. LSTM 창은 학습(배치)과 API(1개)가 **같은 함수**(`seq_windows`)로 만듭니다.


# %%
# 모듈은 저장소의 src/sews_preprocess.py를 그대로 쓴다 (노트북에서는 이 셀이 %%writefile로 Drive에 썼다).
assert os.path.exists(f"{SRC_DIR}/sews_preprocess.py"), f"{SRC_DIR}/sews_preprocess.py가 없다"


# %%
# ===== [4-2] 모듈 불러오기 + config v4 (4단계 규칙) =====
import sews_preprocess as pp

importlib.reload(pp)

cfg = copy.deepcopy(cfg3)
cfg["version"] = 4
cfg["changelog"].append("v4: fold별 스케일링·대치 규칙과 LSTM 입력 규칙 추가 (model_prep)")
fv = cfg["feature_vars"]
seq_vars = fv["vitals_window"] + fv["labs"]
no_ind = [v for v in cfg["no_indicator_features"] if v in seq_vars]
cfg["model_prep"] = {
    "fit_rows": "fold k 파라미터 = dev 중 fold != k 행만, final = dev 전체 (최종 모델·API용). test는 사용하지 않음",
    "lightgbm": "features.parquet 원본(NaN 포함) 그대로 사용, 파라미터 불필요",
    "tabular": {
        "scaler": "(x - 중앙값) / IQR, IQR=0이면 표준편차, 그것도 0이면 1",
        "impute": "스케일 후 NaN → 0 (= 학습 행 중앙값으로 대치)",
        "clip": 10,
        "passthrough": [
            "Gender",
            "Unit_MICU",
            "Unit_SICU",
            "Unit_Unknown",
            "qsofa_resp",
            "qsofa_sbp",
            "sofa_cardiovascular",
        ],  # 0/1 값은 그대로
    },
    "sequence": {
        "vars": seq_vars,
        "mask_delta_vars": [v for v in seq_vars if v not in no_ind],
        "no_mask_vars": no_ind,
        "no_mask_reason": "Part 1: 측정 여부가 세트(병원)마다 달라 측정 지표를 만들지 않기로 한 변수",
        "value": "창 안 forward fill 후 (x - 중앙값) / IQR, ±clip, 창 안 측정 전이면 0",
        "delta": "창 안 마지막 측정 이후 시간 / 24 (창 안 측정 전이면 1)",
        "padding": "기록 시작 전 시간은 빈 행 (값 0, mask 0, delta 1)",
        "static": ["Age", "Gender", "Unit_MICU", "Unit_SICU", "Unit_Unknown"],
        "clip": 10,
        "storage": "data/seq_values.npy (float32, row_id 순, 정제 원시 값) + data/seq_row_start.npy",
    },
}
spec = pp.seq_spec(cfg)
assert set(cfg["model_prep"]["tabular"]["passthrough"]) <= set(names)
for c in cfg["model_prep"]["tabular"]["passthrough"]:
    assert F[c].dropna().isin([0, 1]).all(), c
assert len(spec["vars"]) == 24 and len(spec["mask_vars"]) == 24 - len(no_ind)

V3_BACKUP = f"{ART_DIR}/preprocess_config_v3.json"
if not os.path.exists(V3_BACKUP):
    with open(V3_BACKUP, "w", encoding="utf-8") as f:
        json.dump(cfg3, f, ensure_ascii=False, indent=2)
with open(CFG_PATH, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print("[OK] config v4 저장 (v3 백업: preprocess_config_v3.json)")
print(
    f"  LSTM 입력: 값 {len(spec['vars'])}개 + mask·delta {len(spec['mask_vars'])}개씩 "
    f"(mask 없음: {no_ind}) + 정적 {len(cfg['model_prep']['sequence']['static'])}개"
)


# %%
# ===== [4-3] fold별 파라미터 학습·저장 (fold 0~4 + final) =====
# 표 형태: 특징 126개의 중앙값·IQR / 시퀀스: 원시 변수 24개의 중앙값·IQR (실제 측정값만)
seq_values = clean_df[spec["vars"]].to_numpy(np.float32)  # (행, 24), row_id 순
FOLDS = [0, 1, 2, 3, 4, "final"]
params = {}
t0 = time.time()
for k in FOLDS:
    m = pp.train_mask(em, k)
    params[k] = {
        "fold": k,
        "tabular": pp.fit_tabular(F[m], cfg),
        "sequence": pp.fit_seq(seq_values[m], cfg),
    }
    with open(
        f"{PARAM_DIR}/fold{k}.json" if k != "final" else f"{PARAM_DIR}/final_dev.json",
        "w",
    ) as f:
        json.dump(params[k], f, indent=1)
print(f"[OK] 파라미터 6세트 저장 ({time.time() - t0:.0f}초): {PARAM_DIR}")

# ---- 검사 ----
test_rows = (em["split"] == "test").to_numpy()
fold_arr = em["fold"].to_numpy()
checks = {}
checks["학습 행에 test·검증 fold 행이 없음"] = all(
    not (pp.train_mask(em, k) & (test_rows | (fold_arr == k))).any() for k in range(5)
)
checks["fold k 학습 행 수 = dev - fold k (final = dev 전체)"] = (
    all(
        params[k]["tabular"]["n_rows"] == DEV.sum() - (DEV & (fold_arr == k)).sum()
        for k in range(5)
    )
    and params["final"]["tabular"]["n_rows"] == DEV.sum()
)
# 오염 검사: fold 0의 검증 행과 test 행 값을 망가뜨려도 fold 0 파라미터가 그대로여야 함
bad = (fold_arr == 0) | test_rows
F_bad = F.copy()
F_bad.loc[bad, model_cols] = F_bad.loc[bad, model_cols] * 1000 + 7
v_bad = seq_values.copy()
v_bad[bad] = v_bad[bad] * 1000 + 7
m0 = pp.train_mask(em, 0)
checks["검증 fold·test 값을 망가뜨려도 fold 0 파라미터 동일"] = (
    pp.fit_tabular(F_bad[m0], cfg) == params[0]["tabular"]
    and pp.fit_seq(v_bad[m0], cfg) == params[0]["sequence"]
)
del F_bad, v_bad
spread = pd.DataFrame({k: params[k]["tabular"]["center"] for k in range(5)}).T
n_diff = int((spread.nunique() > 1).sum())
checks[f"fold마다 다른 학습 행으로 학습됨 (중앙값이 fold별로 다른 특징 {n_diff}개)"] = n_diff > 0
for k, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {k}")
assert all(checks.values())
print("\n[fold별 중앙값 예시 (표 형태)]")
print(spread[["HR_mean", "Temp_last", "WBC_last", "Lactate_last", "Creatinine_last"]].round(3))


# %%
# ===== [4-4] 표 형태 변환 확인 + 로지스틱 회귀 점검 (fold 0) =====
# 스케일·대치를 쓰는 모델이 실제로 학습되는지 확인 (하이퍼파라미터 탐색 없음, test 미사용)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

p0 = params[0]["tabular"]
tr = pp.train_mask(em, 0)
va = DEV & (em["fold"].to_numpy() == 0)
Xtr = pp.apply_tabular(F[tr], p0, model_cols)
Xva = pp.apply_tabular(F[va], p0, model_cols)
y = em["SepsisLabel"].to_numpy()

nonpass = [i for i, c in enumerate(model_cols) if c not in p0["passthrough"]]
obs_tr = F.loc[tr, model_cols].notna().to_numpy()
med_after = np.array([np.median(Xtr[obs_tr[:, i], i]) for i in nonpass])
checks = {
    "변환 후 NaN·inf 없음": np.isfinite(Xtr).all() and np.isfinite(Xva).all(),
    f"값 범위 ±{p0['clip']} 안": np.abs(Xtr).max() <= p0["clip"]
    and np.abs(Xva).max() <= p0["clip"],
    "학습 행에서 측정값의 중앙값 = 0 (스케일 기준이 학습 행)": np.abs(med_after).max() < 1e-4,
    "0/1 특징은 값 그대로": all(
        np.array_equal(Xtr[:, model_cols.index(c)], F.loc[tr, c].fillna(0).to_numpy(np.float32))
        for c in p0["passthrough"]
    ),
}
for k, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {k}")
assert all(checks.values())

t0 = time.time()
lr = LogisticRegression(max_iter=300, C=1.0).fit(Xtr, y[tr])
p = lr.predict_proba(Xva)[:, 1]
le60 = em["iculos_le60"].to_numpy()[va] == 1
print(
    f"\n[fold 0 로지스틱 회귀] AUROC {roc_auc_score(y[va], p):.3f} | AUPRC {average_precision_score(y[va], p):.3f} | "
    f"AUROC (ICULOS<=60) {roc_auc_score(y[va][le60], p[le60]):.3f} | {time.time() - t0:.0f}초"
)
del Xtr, Xva, obs_tr


# %%
# ===== [4-5] LSTM 입력 저장: 원시 값 배열 + 환자 시작 위치 (창은 학습할 때 잘라 씀) =====
# 모든 행의 24시간 창을 미리 만들면 약 13GB → 원시 값(행 x 24변수)만 저장하고 배치마다 창을 만듦
row_start = (em.groupby("Patient_ID")["row_id"].transform("min")).to_numpy(np.int64)
SEQ_PATH = f"{DATA_DIR}/seq_values.npy"
START_PATH = f"{DATA_DIR}/seq_row_start.npy"
np.save(SEQ_PATH, seq_values)
np.save(START_PATH, row_start)
seq_meta = {
    "vars": spec["vars"],
    "mask_vars": spec["mask_vars"],
    "n_rows": int(len(seq_values)),
    "dtype": "float32",
    "row_order": "row_id",
    "window": pp.WIN,
    "files": {"values": "data/seq_values.npy", "row_start": "data/seq_row_start.npy"},
}
with open(f"{ART_DIR}/seq_meta.json", "w", encoding="utf-8") as f:
    json.dump(seq_meta, f, ensure_ascii=False, indent=2)

vals_mm = np.load(SEQ_PATH, mmap_mode="r")  # 학습 때처럼 메모리 매핑으로 읽기
assert vals_mm.shape == seq_values.shape and np.array_equal(
    vals_mm[:1000], seq_values[:1000], equal_nan=True
)
assert (np.load(START_PATH) == row_start).all()
print(
    f"[OK] seq_values.npy {os.path.getsize(SEQ_PATH) / 1e6:.0f} MB ({vals_mm.shape[0]:,}행 x {vals_mm.shape[1]}변수), "
    f"seq_row_start.npy, seq_meta.json"
)


# %%
# ===== [4-6] LSTM 창 검사: 원본과 일치, 3단계 특징과 일치, API 경로와 일치, 24시간 원칙 =====
sp0 = params[0]["sequence"]
rng = np.random.default_rng(42)
pos = em["row_id"].to_numpy() - row_start
dev_idx = np.flatnonzero(DEV)
pick = np.sort(
    np.unique(
        np.concatenate(
            [
                rng.choice(dev_idx, 2000, replace=False),
                rng.choice(dev_idx[pos[dev_idx] < 23], 500, replace=False),
            ]
        )
    )
)
W = pp.seq_windows(vals_mm, row_start, pick, sp0, cfg, return_raw=True)
B = len(pick)

# (1) 창의 원시 값 = 정제 데이터의 최근 24행 (기록 시작 전은 빈 행)
ok1 = True
for b, r in enumerate(pick):
    L = min(pos[r], 23) + 1
    ref = clean_df[spec["vars"]].iloc[r - L + 1 : r + 1].to_numpy(np.float32)
    ok1 &= np.isnan(W["raw"][b, : 24 - L]).all() and np.array_equal(
        W["raw"][b, 24 - L :], ref, equal_nan=True
    )

# (2) 마지막 시점이 3단계 특징과 같은지: ffill 값 = {변수}_last, mask 합 = n24h, delta = since_h
Fp = F.iloc[pick]
last_ok = all(
    np.allclose(W["ff"][:, -1, j], Fp[f"{v}_last"].to_numpy(), rtol=1e-6, equal_nan=True)
    for j, v in enumerate(spec["vars"])
)
meas_ok = True
for g, members in cfg["measurement_indicator_groups"].items():
    js = [spec["vars"].index(m_) for m_ in members]
    any_obs = ~np.isnan(W["raw"][:, :, js]).all(axis=2)  # 묶음 중 하나라도 측정
    li = np.where(any_obs.any(axis=1), 23 - np.argmax(any_obs[:, ::-1], axis=1), -1)
    since = np.where(li >= 0, 23 - li, 24)
    meas_ok &= np.array_equal(any_obs.sum(axis=1), Fp[f"meas_{g}_n24h"].to_numpy())
    meas_ok &= np.array_equal(since, Fp[f"meas_{g}_since_h"].to_numpy())
    if len(members) == 1:  # 단일 변수면 delta와도 비교
        mj = spec["mask_vars"].index(members[0])
        meas_ok &= np.allclose(W["delta"][:, -1, mj] * 24, Fp[f"meas_{g}_since_h"].to_numpy())

# (3) API 경로(DataFrame 1개 → 창 1개)와 배치 경로가 완전히 같은지 + 빈 행 채움 무관
api_ok, pad_ok = True, True
for b, r in enumerate(pick[:300]):
    L = min(pos[r], 23) + 1
    win = clean_df.iloc[r - L + 1 : r + 1]
    a = pp.seq_window_from_frame(win, sp0, cfg)
    api_ok &= all(np.array_equal(a[k][0], W[k][b]) for k in ("x", "mask", "delta"))
    if L < 24:
        padded = pd.concat(
            [pd.DataFrame(np.nan, index=range(24 - L), columns=win.columns), win],
            ignore_index=True,
        )
        pa = pp.seq_window_from_frame(padded, sp0, cfg)
        pad_ok &= all(np.array_equal(pa[k][0], a[k][0]) for k in ("x", "mask", "delta"))

# (4) 24시간 원칙: 창보다 오래된 행을 망가뜨려도 창이 그대로
long_rows = pick[pos[pick] >= 30]
_, first = np.unique(em["Patient_ID"].to_numpy()[long_rows], return_index=True)
long_rows = np.sort(long_rows[first])  # 환자당 1개 (같은 환자의 다른 창을 함께 망가뜨리지 않게)
vb = seq_values.copy()
for r in long_rows:
    vb[row_start[r] : r - 23] = rng.uniform(0, 500, (r - 23 - row_start[r], vb.shape[1]))
Wb = pp.seq_windows(vb, row_start, long_rows, sp0, cfg)
Wa = pp.seq_windows(vals_mm, row_start, long_rows, sp0, cfg)
ok4 = all(np.array_equal(Wa[k], Wb[k]) for k in ("x", "mask", "delta"))
del vb

checks = {
    f"창 원시 값 = 정제 데이터 최근 24행 ({B:,}개 창)": ok1,
    "마지막 시점 ffill 값 = 3단계 {변수}_last (24개 변수)": last_ok,
    "측정 횟수·마지막 측정 이후 시간 = 3단계 meas_* 특징": meas_ok,
    "API 경로 = 배치 경로 (300개 창, 완전히 동일)": api_ok,
    "빈 행 채움(padding) 무관": pad_ok,
    f"24시간 밖을 망가뜨려도 창 동일 ({len(long_rows):,}개 창)": ok4,
    "x에 NaN 없음, mask는 0/1, delta는 0~1": np.isfinite(W["x"]).all()
    and np.isin(W["mask"], [0, 1]).all()
    and W["delta"].min() >= 0
    and W["delta"].max() <= 1,
}
for k, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {k}")
assert all(checks.values())

# 배치 속도·크기 (학습 루프 예상)
t0 = time.time()
nb_ = 0
# wb: 마지막 배치를 아래에서 크기 확인에 쓴다
for _rows_b, wb in pp.iter_batches(dev_idx[: 512 * 20], 512, vals_mm, row_start, sp0, cfg, seed=0):  # noqa: B007
    nb_ += 1
dt = (time.time() - t0) / nb_
mb = sum(v.nbytes for v in wb.values()) / 1e6
print(
    f"\n[배치] 512개 창: x {wb['x'].shape}, mask {wb['mask'].shape}, delta {wb['delta'].shape} | "
    f"{mb:.1f} MB | 배치당 {1000 * dt:.0f} ms"
)
del W, Wa, Wb


# %%
# ===== [4-7] registry 갱신 + 모델링 인계 요약 =====
with open(f"{SRC_DIR}/sews_preprocess.py", "rb") as f:
    pp_sha = hashlib.sha256(f.read()).hexdigest()
registry["config_version"] = cfg["version"]
registry["preprocess"] = {
    "module": "src/sews_preprocess.py",
    "module_sha256": pp_sha,
    "params": {
        str(k): f"artifacts/preprocess_params/{'final_dev' if k == 'final' else f'fold{k}'}.json"
        for k in FOLDS
    },
    "tabular_passthrough": cfg["model_prep"]["tabular"]["passthrough"],
    "clip": cfg["model_prep"]["tabular"]["clip"],
    "sequence": {**seq_meta, "static": cfg["model_prep"]["sequence"]["static"]},
    "usage": {
        "lightgbm": "features.parquet[model_cols] 그대로 (NaN 포함)",
        "tabular_nn_lr": 'pp.apply_tabular(F, fold_k_params["tabular"], model_cols)',
        "lstm": 'pp.iter_batches(train_rows, bs, np.load(seq_values, mmap_mode="r"), row_start, fold_k_params["sequence"], cfg)'
        ' + 정적 입력 pp.apply_tabular(F.iloc[rows], fold_k_params["tabular"], static)',
        "validation_rows": "eval_meta: split == 'dev' & fold == k",
        "test_rows": "eval_meta: split == 'test' → 최종 평가 1회만, final_dev 파라미터 사용",
    },
}
with open(f"{ART_DIR}/feature_registry.json", "w", encoding="utf-8") as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)

print("[Drive에 저장된 4단계 파일]")
for p in [
    CFG_PATH,
    V3_BACKUP,
    f"{SRC_DIR}/sews_preprocess.py",
    SEQ_PATH,
    START_PATH,
    f"{ART_DIR}/seq_meta.json",
    f"{ART_DIR}/feature_registry.json",
] + [f"{PARAM_DIR}/{fn}" for fn in sorted(os.listdir(PARAM_DIR))]:
    print(f"  {os.path.relpath(p, DRIVE_DIR):45s} {os.path.getsize(p) / 1e6:8.2f} MB")


# %% [markdown]
# ## 정상일 때 결과 (전체 데이터로 미리 실행해 확인한 값)
#
# | 셀 | 확인할 값 |
# |---|---|
# | [4-0] | features 1,552,210행 x 126열, 특징 모듈 sha256 = registry 기록, 시작 config v3 |
# | [4-2] | LSTM 입력: 값 24개 + mask·delta 21개씩 (mask 없음: pH, PaCO2, PTT) + 정적 5개 |
# | [4-3] | 파라미터 6세트 저장, `[OK]` 4개 (fold별 중앙값이 다른 특징 47개) |
# | [4-4] | `[OK]` 4개, fold 0 로지스틱 회귀 AUROC 0.773 / AUPRC 0.070 / ICULOS≤60 AUROC 0.760 |
# | [4-5] | seq_values.npy 149 MB (1,552,210행 x 24변수) |
# | [4-6] | `[OK]` 7개 (창 2,500개 원본 일치, API 경로 300개 완전 일치, 24시간 밖 조작 610개 창 동일) |
# | [4-7] | 4단계 파일 13개 목록 |
#
# - 로지스틱 회귀 결과는 sklearn 버전에 따라 소수 셋째 자리가 조금 다를 수 있습니다.
# - 최대 메모리 사용량 약 5 GB. 전체 실행 약 5분.
