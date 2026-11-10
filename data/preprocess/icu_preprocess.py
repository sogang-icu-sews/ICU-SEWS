# %% [markdown]
# SEWS 1~3단계 (ICU-3): 정제 확정, 특징 125+1개 추출, 누수 검사, fold 0 기준선
#
# 원본은 Colab 노트북 ICU-3_preprocess.ipynb이다. 셀 순서와 코드는 그대로이고, 모듈 셀(%%writefile)만
# 저장소의 src/sews_features.py를 쓰도록 바꿨다.
# 실행: `python preprocess/icu3_preprocess.py` (SEWS_ROOT에 0단계 산출물이 있어야 한다)


# %%
import os
import sys
from pathlib import Path

# 실행 환경 (Colab·로컬 공통). 환경 변수로 바꿀 수 있다.
DRIVE_DIR = os.environ.get('SEWS_ROOT', '/content/drive/MyDrive/capston')   # 데이터·산출물 루트
try:
    REPO_DIR = Path(__file__).resolve().parents[1]                            # python 으로 실행할 때
except NameError:
    REPO_DIR = Path(os.environ.get('SEWS_REPO', os.getcwd()))                 # Jupyter·Colab 셀로 실행할 때
SRC_DIR = str(REPO_DIR / 'src')                                               # 공용 모듈 (src/sews_*.py)


# %%
try:  # Colab에서만 Drive 연결
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass


# %% [markdown]
# # ICU-3 전처리·특징 추출 (1~3단계)
#
# **입력**: 0단계에서 Drive에 저장한 3개 파일 (`data/master.parquet`, `artifacts/patient_split.csv`, `artifacts/preprocess_config.json` v2)
#
# **출력** (모두 Drive `capston/` 아래)
# | 파일 | 내용 |
# |---|---|
# | `src/sews_features.py` | 정제 + 특징 계산 모듈. 학습과 API가 **같은 파일**을 import |
# | `artifacts/preprocess_config.json` | v3 (v2는 `preprocess_config_v2.json`으로 백업) |
# | `data/features.parquet` | `row_id` + 특징 126열 (모델 입력 125 + 선택 1), float32 |
# | `data/eval_meta.parquet` | 평가 전용 정보 (라벨, 분할, fold, t_sepsis, rel, early_window, obs_hours …) |
# | `artifacts/feature_registry.json` | 특징 목록·묶음·원천 변수·dev 결측률·모듈 sha256 |
# | `artifacts/feature_missing_report.csv` | 특징별 결측률 (dev, set A/B) |
#
# **실행**: 위에서부터 순서대로. 기본 Colab 런타임(RAM 12.7GB)으로 충분. 셀마다 `[OK]`가 나오는지 확인.
# 지켜야 하는 원칙: test 환자는 계산만 하고 요약·학습에 쓰지 않음, 공식 라벨은 이동하지 않음, 특징은 최근 24행만 사용.


# %% [markdown]
# ## 1단계: 정제 확정


# %%
# ===== [1-0] 경로와 입력: 0단계에서 Drive에 저장한 파일 3개만 불러온다 =====
import os
import sys
import json
import time
import copy
import hashlib
import importlib
import numpy as np
import pandas as pd

ART_DIR = f'{DRIVE_DIR}/artifacts'
DATA_DIR = f'{DRIVE_DIR}/data'

master = pd.read_parquet(f'{DATA_DIR}/master.parquet')
split = pd.read_csv(f'{ART_DIR}/patient_split.csv')
CFG_PATH = f'{ART_DIR}/preprocess_config.json'
with open(CFG_PATH, encoding='utf-8') as f:
    cur_cfg = json.load(f)
if cur_cfg['version'] == 2:                       # 처음 실행
    cfg2 = cur_cfg
else:                                             # 다시 실행: v3를 이미 저장했으면 v2 백업에서 시작
    with open(f'{ART_DIR}/preprocess_config_v2.json', encoding='utf-8') as f:
        cfg2 = json.load(f)
assert cfg2['version'] == 2

assert len(master) == 1_552_210 and master['Patient_ID'].nunique() == 40_336
assert set(split['Patient_ID']) == set(master['Patient_ID'])
master = master.sort_values(['Patient_ID', 'ICULOS'], kind='stable').reset_index(drop=True)
same_pt = master['Patient_ID'].to_numpy()[1:] == master['Patient_ID'].to_numpy()[:-1]
assert (np.diff(master['ICULOS'].to_numpy())[same_pt] == 1).all(), '환자 안에서 1시간 간격이 아닌 곳이 있음'

sp_idx = split.set_index('Patient_ID')
SPLIT = master['Patient_ID'].map(sp_idx['split']).to_numpy()
FOLD = master['Patient_ID'].map(sp_idx['fold']).to_numpy()
DEV = SPLIT == 'dev'
print(f'[OK] master {len(master):,}행 / 환자 {master.Patient_ID.nunique():,}명 '
      f'(dev 행 {DEV.sum():,} / test 행 {(~DEV).sum():,})')
print('[OK] 환자 안 행 간격: 모두 1시간 → 24행 = 24시간')
print(f'[OK] 시작 config: v{cfg2["version"]}')


# %% [markdown]
# ### [1-1] 특징 모듈 (`src/sews_features.py`)
# 이 셀은 모듈 파일을 Drive에 **씁니다**(실행해도 계산은 하지 않음).
# 학습용 계산(`build_features`)과 API용 참조 계산(`build_features_window`)을 일부러 다른 코드로 작성했고, 3단계에서 두 결과가 같은지 확인합니다.


# %%
# 모듈은 저장소의 src/sews_features.py를 그대로 쓴다 (노트북에서는 이 셀이 %%writefile로 Drive에 썼다).
assert os.path.exists(f'{SRC_DIR}/sews_features.py'), f'{SRC_DIR}/sews_features.py가 없다'


# %%
# ===== [1-2] 모듈 불러오기 + config v3 (결정표 13개 반영) =====
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import sews_features as sf
importlib.reload(sf)                              # 모듈 셀을 고쳐 다시 실행했을 때 반영

cfg = copy.deepcopy(cfg2)
cfg['version'] = 3
cfg['changelog'].append('v3: SOFA 상태 STALE 삭제(24h 창에서는 생길 수 없음), 특징 계산 규칙 명시, '
                        '모델 입력 제외 항목(관찰 시간, SOFA 상태, Calcium_ionized), eval_meta 정의')
cfg['sofa']['states'] = ['OK', 'NOT_MEASURED', 'NOT_COMPUTABLE']
cfg['sofa']['display_only'] = {
    'state': 'OK(창 안 측정) / NOT_MEASURED(창 안 측정 없음) / NOT_COMPUTABLE(데이터셋에 변수 없음)',
    'value_age_h': '마지막 측정 이후 시간(0~23). 대시보드 표시용, 모델 입력 아님',
}
cfg['feature_rules'] = {
    'values': '창 안에서 실제로 측정된 값만 사용 (forward fill 값은 쓰지 않음)',
    'window_rows': '현재 행 포함 최근 24행 (t-23 ~ t)',
    'std': 'ddof=1, 측정 2회 이상',
    'iqr': 'np.percentile(linear) 75% - 25%, 측정 2회 이상',
    'skew': '모집단 g1 = m3 / m2^1.5, 측정 3회 이상',
    'cv': 'std / |mean|, 측정 2회 이상',
    'slope': '측정점 최소제곱 직선의 기울기 (단위/시간), 측정 2회 이상',
    'delta6h': '현재 값 - (t-6 이전 창 안 마지막 측정값), 측정 2회 이상',
    'constant_window': '창 안 값이 모두 같으면 std = iqr = skew = slope = 0',
    'lab_delta_prev': '창 안 마지막 측정값 - 창 안 직전 측정값',
    'measurement': 'meas_<묶음>_n24h = 창 안 측정한 시간 수, meas_<묶음>_since_h = 마지막 측정 이후 시간 (없으면 24)',
    'sofa_delta': '항목별 (현재 점수 - 창 안 측정값들의 최저 점수)의 합, 계산 가능한 항목만',
    'padding': '창 앞을 값이 전부 빈 행으로 채워도 결과 동일',
}
cfg['model_input_excluded'] = {
    'obs_hours': '관찰 시간(24h 상한) 단독 AUROC 0.567 → 시간 지름길. eval_meta와 API data_sufficiency에만 사용',
    'sofa_states': 'STALE 삭제 후에는 점수 결측 여부와 같은 정보. API 표시용',
    'Calcium_ionized': 'setB에만 있는 값 → 세트(병원) 표지 위험',
}
cfg['optional_features'] = list(cfg['derived_features']['optional'])
cfg['eval_only_columns'] = ['row_id', 'Patient_ID', 'ICULOS', 'set', 'split', 'fold', 'SepsisLabel',
                            't_sepsis', 'rel', 'label_at_start', 'iculos_le60', 'early_window', 'obs_hours']
cfg['eval_meta'] = {
    't_sepsis': '첫 SepsisLabel=1 시각 + 6h (공식 라벨이 t_sepsis-6부터 1이므로 역산. 라벨 자체는 이동하지 않음)',
    'rel': 'ICULOS - t_sepsis (비패혈증은 NaN)',
    'label_at_start': '기록 첫 행부터 라벨 1인 환자 (실제 t_sepsis는 첫 행+6h보다 이를 수 있음)',
    'iculos_le60': 'ICULOS <= 60 (시간 지름길이 없는 평가 구간)',
    'early_window': '-12 <= rel < -6 (라벨은 0이지만 경보가 유용한 구간, PhysioNet utility 기준)',
    'obs_hours': '창 안 행 수 (1~24)',
}

# ---- 일관성 검사 ----
reg = sf.feature_registry(cfg)
names = [r['name'] for r in reg]
model_cols = [r['name'] for r in reg if not r['optional']]
blocked_src = (set(cfg['excluded_features']) | set(cfg['redundant_features'])
               | set(cfg['banned_time_features']['columns']) | {'Calcium_ionized'})
assert len(names) == len(set(names)), '특징 이름 중복'
assert len(model_cols) == 125 and len(names) == 126, (len(model_cols), len(names))
assert sf.banned_name_hits(names, cfg) == [], sf.banned_name_hits(names, cfg)
assert not any(set(r['source']) & blocked_src for r in reg), '제외 변수로 만든 특징이 있음'
assert not (set(cfg['model_input_excluded']) & set(names))
assert 'STALE' not in cfg['sofa']['states']
assert cfg['labels']['sepsis']['extra_shift_h'] == 0            # 기준 3

V2_BACKUP = f'{ART_DIR}/preprocess_config_v2.json'
if not os.path.exists(V2_BACKUP):
    with open(V2_BACKUP, 'w', encoding='utf-8') as f:
        json.dump(cfg2, f, ensure_ascii=False, indent=2)
with open(CFG_PATH, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
with open(CFG_PATH, encoding='utf-8') as f:
    assert json.load(f) == cfg
groups = pd.Series([r['group'] for r in reg if not r['optional']]).value_counts()
print('[OK] config v3 저장 (v2 백업: preprocess_config_v2.json)')
print(f'[OK] 모델 입력 특징 {len(model_cols)}개 + 선택 {len(names) - len(model_cols)}개 {cfg["optional_features"]}')
print('  묶음별:', groups.to_dict())


# %%
# ===== [1-3] 정제: 고정 규칙이라 전체 행에 적용, 요약은 dev 행으로만 (test 봉인) =====
clean_df, _ = sf.clean(master, cfg)
_, log_dev = sf.clean(master[DEV], cfg)
label_ref = master[['Patient_ID', 'ICULOS', 'SepsisLabel']].copy()     # 3단계 라벨 검사용 원본
assert len(clean_df) == len(master)

thr = cfg['transforms']['calcium_ionized_threshold']
for v, (lo, hi) in cfg['plausible_ranges'].items():
    x = clean_df[v]
    assert not ((x < lo) | (x > hi)).any(), v
assert not (clean_df['Calcium'] < thr).any()
assert (clean_df['Age'] != cfg['transforms']['age_recode']['from']).all()
assert clean_df['Unit'].notna().all()
assert (clean_df['SepsisLabel'].to_numpy() == master['SepsisLabel'].to_numpy()).all()   # 라벨은 건드리지 않음

print('[정제 규칙별 변경 행 수 (dev)]')
print(log_dev[log_dev['rows_changed'] > 0].to_string(index=False))
u = clean_df.loc[DEV].groupby('set', observed=True)['Unit'].value_counts(normalize=True).mul(100).round(1).unstack()
print('\n[Unit 분포 (dev, set별 %)]')
print(u)
print(f'\n[OK] 정제 완료: {len(clean_df):,}행, Age 최댓값 {clean_df["Age"].max():.0f}')
del master


# %% [markdown]
# ## 2단계: 특징 추출


# %%
# ===== [2-0] 정의 확인: 손으로 계산한 답과 비교 (8시간짜리 가상 환자 1명) =====
# 3단계의 일치 검사는 '두 계산이 서로 같은지'만 본다. 여기서는 '정의 자체가 맞는지'를 손계산 답으로 확인.
#   HR: -7h 90, -1h 100, 0h 110 → 평균 100, 표준편차 10, IQR 10, 왜도 0, CV 0.1,
#       기울기 = 630/258 ≈ 2.4419 /h (손계산 최소제곱), delta6h = 110 - 90 (t-6 이전 마지막 값)
#   Temp: 37.0이 두 번 → 값이 모두 같음 → std = iqr = slope = 0, 37도 편차 0
#   Platelets: -5h 120(1점), -1h 90(2점) → 응고 2점, 창 안 최저 1점 → 변화 +1, delta_prev -30
#   MAP: 75 → 0h 65 → 심혈관 1점, 변화 +1 | Creatinine 1.0 → 신장 0점 | Bilirubin 없음 → 간 NaN
#   → sofa_delta = 1 + 1 + 0 = 2 | Lactate -4h 측정 → since 4h, 1회 | FiO2 없음 → since 24h, 0회
#   Resp 24 → qSOFA 호흡 1 | SBP 110 → qSOFA 혈압 0 | BUN/Cr = 20 / 1.0 = 20
toy_cols = [c for c in clean_df.columns if c not in ('Unit',)]
toy = pd.DataFrame(np.nan, index=range(8), columns=toy_cols)
toy['Patient_ID'], toy['ICULOS'], toy['SepsisLabel'] = 999_999, np.arange(1, 9), 0
toy['Age'], toy['Gender'], toy['Unit1'], toy['Unit2'] = 60.0, 1.0, 1.0, 0.0
toy.loc[[0, 6, 7], 'HR'] = [90, 100, 110]
toy.loc[[4, 7], 'Temp'] = 37.0
toy.loc[[2, 6], 'Platelets'] = [120, 90]
toy.loc[[0, 7], 'MAP'] = [75, 65]
toy.loc[5, 'Creatinine'], toy.loc[5, 'BUN'] = 1.0, 20
toy.loc[3, 'Lactate'] = 2.0
toy.loc[7, 'Resp'], toy.loc[7, 'SBP'] = 24, 110
toy, _ = sf.clean(toy, cfg)

expected = {
    'HR_last': 110, 'HR_mean': 100, 'HR_median': 100, 'HR_min': 90, 'HR_max': 110, 'HR_std': 10, 'HR_iqr': 10,
    'HR_skew': 0, 'HR_cv': 0.1, 'HR_slope': 630 / 258, 'HR_delta6h': 20,
    'Temp_std': 0, 'Temp_iqr': 0, 'Temp_slope': 0, 'Temp_dev': 0, 'Temp_skew': np.nan,   # skew는 3회 이상
    'Platelets_last': 90, 'Platelets_delta_prev': -30,
    'sofa_coagulation': 2, 'sofa_cardiovascular': 1, 'sofa_renal': 0, 'sofa_liver': np.nan, 'sofa_delta': 2,
    'meas_lactate_since_h': 4, 'meas_lactate_n24h': 1, 'meas_fio2_since_h': 24, 'meas_fio2_n24h': 0,
    'qsofa_resp': 1, 'qsofa_sbp': 0, 'qsofa_partial': 1, 'BUN_Creatinine_ratio': 20,
    'Unit_MICU': 1, 'Unit_Unknown': 0,
}
got_ref = sf.build_features_window(toy, cfg)              # 참조용(API)
got_vec = sf.build_features(toy, cfg).iloc[-1]            # 학습용
wrong = [(k, v, got_ref[k], got_vec[k]) for k, v in expected.items()
         if not (np.isclose(got_ref[k], v, equal_nan=True) and np.isclose(got_vec[k], v, equal_nan=True))]
print(f"  [{'OK' if not wrong else 'FAIL'}] 손계산 답 {len(expected)}개 = 참조용 = 학습용")
for w in wrong:
    print('   틀림:', w)
assert not wrong
del toy


# %%
# ===== [2-1] 특징 계산 (학습용, 전체 행 한 번에) =====
# 순수 함수: fold·라벨과 무관하고 각 행의 최근 24행만 사용 → 한 번만 계산 (대치·스케일링은 4단계에서 fold별)
t0 = time.time()
F = sf.build_features(clean_df, cfg)                 # float64 (3단계 검사가 끝날 때까지 메모리에 유지)
print(f'[OK] 특징 계산 {time.time() - t0:.0f}초: {F.shape[0]:,}행 x {F.shape[1]}열')

assert list(F.columns) == names and len(F) == len(clean_df)
vals = F.to_numpy()
assert not np.isinf(vals).any(), 'inf 값이 있음'
for c in ['Age', 'Gender', 'Unit_MICU', 'Unit_SICU', 'Unit_Unknown']:
    assert F[c].notna().all(), c
assert (F[['Unit_MICU', 'Unit_SICU', 'Unit_Unknown']].sum(axis=1) == 1).all()
for g in cfg['measurement_indicator_groups']:
    assert F[f'meas_{g}_since_h'].between(0, 24).all() and F[f'meas_{g}_n24h'].between(0, 24).all()
for c in sf.SOFA_COMPONENTS:
    assert F[f'sofa_{c}'].dropna().isin([0, 1, 2, 3, 4]).all()
assert (F['sofa_delta'].dropna() >= 0).all()                      # 현재 - 창 안 최저 ≥ 0
print('[OK] 값 범위 검사 통과 (inf 없음, 정적 특징 결측 없음, SOFA 점수 0~4, 변화량 ≥ 0)')
del vals


# %%
# ===== [2-2] eval_meta: 평가에만 쓰는 정보 (모델 입력과 분리된 별도 표) =====
em = clean_df[['Patient_ID', 'ICULOS', 'set', 'SepsisLabel']].copy()
em.insert(0, 'row_id', np.arange(len(em), dtype=np.int64))
em['split'] = SPLIT
em['fold'] = FOLD.astype(int)
onset = em.loc[em['SepsisLabel'] == 1].groupby('Patient_ID')['ICULOS'].min()
em['t_sepsis'] = em['Patient_ID'].map(onset + 6)
em['rel'] = em['ICULOS'] - em['t_sepsis']
first_label = em.groupby('Patient_ID')['SepsisLabel'].transform('first')
em['label_at_start'] = (first_label == 1).astype(np.int8)
em['iculos_le60'] = (em['ICULOS'] <= 60).astype(np.int8)
em['early_window'] = em['rel'].between(-12, -6, inclusive='left').astype(np.int8)
em['obs_hours'] = np.minimum(em.groupby('Patient_ID').cumcount() + 1, 24).astype(np.int8)
assert set(em.columns) == set(cfg['eval_only_columns'])
em = em[cfg['eval_only_columns']]                      # config에 적힌 순서로 정렬

d = em[DEV]
pt_d = d.groupby('Patient_ID').agg(sepsis=('SepsisLabel', 'max'), at_start=('label_at_start', 'first'))
print(f"[dev] 행 {len(d):,} | 양성 행 {int(d.SepsisLabel.sum()):,} ({100 * d.SepsisLabel.mean():.2f}%) | "
      f"패혈증 환자 {int(pt_d.sepsis.sum()):,}명 (첫 행부터 양성 {int(pt_d.at_start.sum()):,}명)")
print(f"[dev] early_window 행 {int(d.early_window.sum()):,} | ICULOS<=60 행 {100 * d.iculos_le60.mean():.1f}% | "
      f"obs_hours<24 행 {100 * (d.obs_hours < 24).mean():.1f}%")
assert int(pt_d.sepsis.sum()) == 2_492


# %%
# ===== [2-3] 저장: features.parquet / eval_meta.parquet / feature_registry.json / 결측률 보고서 =====
FEAT_PATH = f'{DATA_DIR}/features.parquet'
META_PATH = f'{DATA_DIR}/eval_meta.parquet'
REG_PATH = f'{ART_DIR}/feature_registry.json'
MISS_PATH = f'{ART_DIR}/feature_missing_report.csv'

feat_out = F.astype(np.float32)
feat_out.insert(0, 'row_id', em['row_id'].to_numpy())
feat_out.to_parquet(FEAT_PATH, index=False)
em.to_parquet(META_PATH, index=False)

# 결측률 (dev만, set별) — 메모리를 아끼려고 결측 여부(bool)만 한 번 만들어 행을 고름
na = F.isna().to_numpy()
is_a = (em['set'] == 'A').to_numpy()
miss = pd.DataFrame({
    'missing_%_dev': na[DEV].mean(axis=0) * 100,
    'missing_%_A': na[DEV & is_a].mean(axis=0) * 100,
    'missing_%_B': na[DEV & ~is_a].mean(axis=0) * 100,
}, index=names).round(2)
del na
miss['group'] = [r['group'] for r in reg]
miss.index.name = 'feature'
miss.to_csv(MISS_PATH)

with open(f'{SRC_DIR}/sews_features.py', 'rb') as f:
    module_sha = hashlib.sha256(f.read()).hexdigest()
registry = {
    'config_version': cfg['version'],
    'module': 'src/sews_features.py',
    'module_sha256': module_sha,                    # API가 같은 버전의 모듈을 쓰는지 확인용
    'n_model_input': len(model_cols),
    'n_optional': len(names) - len(model_cols),
    'row_key': 'row_id',
    'features': [dict(r, missing_pct_dev=float(miss.loc[r['name'], 'missing_%_dev'])) for r in reg],
}
with open(REG_PATH, 'w', encoding='utf-8') as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)

# 다시 읽어서 확인 (열 단위로 비교해 메모리 절약)
del feat_out
f2 = pd.read_parquet(FEAT_PATH)
m2 = pd.read_parquet(META_PATH)
assert f2.shape == (len(F), len(names) + 1) and list(f2.columns[1:]) == names
assert (f2['row_id'].to_numpy() == m2['row_id'].to_numpy()).all()
for c in names:
    assert np.allclose(f2[c].to_numpy(np.float64), F[c].to_numpy(), rtol=1e-6, atol=1e-6, equal_nan=True), c
pd.testing.assert_frame_equal(m2, em.reset_index(drop=True), check_dtype=False)
del f2, m2

print(f'[OK] features.parquet  {os.path.getsize(FEAT_PATH) / 1e6:.1f} MB ({len(F):,}행, row_id + {len(names)}열)')
print(f'[OK] eval_meta.parquet {os.path.getsize(META_PATH) / 1e6:.1f} MB ({em.shape[1]}열)')
print(f'[OK] feature_registry.json (module sha256 {module_sha[:12]}…), feature_missing_report.csv')
print('\n[묶음별 결측률 (dev, %)] 중앙값 / 최댓값')
print(miss.groupby('group')['missing_%_dev'].agg(['count', 'median', 'max']).round(1))
print('\n[결측률 높은 특징 상위 8개 (dev, %)]')
print(miss.sort_values('missing_%_dev', ascending=False).head(8)[['missing_%_dev', 'missing_%_A', 'missing_%_B']])


# %% [markdown]
# ## 3단계: 누수 검사와 기준선
# | 검사 | 확인하는 것 |
# |---|---|
# | ① 금지 컬럼 | ICULOS·평가 전용 컬럼·제외 변수가 특징 파일에 없음 |
# | ② 라벨 무결성 | 라벨이 원본과 행 단위로 같고 추가 이동이 없음 |
# | ③ 학습-API 일치 | 학습용 계산 = 24행만 받는 참조 계산 (24시간 원칙 + API 일치) |
# | ④ 미래 차단 | 미래를 지우고 24시간 밖을 망가뜨려도 특징이 그대로 (음성 대조 포함) |
# | 기준선 | fold 0 LightGBM으로 특징이 작동하는지, 시간 지름길이 없는지 |


# %%
# ===== [3-1] 누수 검사 ①: 금지 컬럼·이름 (저장된 features.parquet의 실제 컬럼으로 확인) =====
import pyarrow.parquet as pq
schema = pq.read_schema(FEAT_PATH)
saved = schema.names
saved_feat = saved[1:]
reg_by_name = {r['name']: r for r in reg}
checks = {
    '첫 컬럼이 row_id, 나머지가 registry 특징과 같은 순서': saved[0] == 'row_id' and saved_feat == names,
    '특징 이름이 금지 컬럼·패턴과 겹치지 않음': sf.banned_name_hits(saved_feat, cfg) == [],
    'ICULOS·평가 전용 컬럼·라벨이 특징에 없음': not (set(cfg['eval_only_columns']) & set(saved_feat)),
    '제외·중복 변수로 만든 특징 없음': not any(set(reg_by_name[n]['source']) & blocked_src for n in saved_feat),
    '관찰 시간·SOFA 상태가 특징에 없음': not (({'obs_hours'} | {f'{c}_state' for c in sf.SOFA_COMPONENTS}) & set(saved_feat)),
    '모든 특징이 float32': all(str(schema.field(n).type) == 'float' for n in saved_feat),
}
for k, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {k}")
assert all(checks.values())


# %%
# ===== [3-2] 누수 검사 ②: 라벨 무결성 (기준 3: 공식 라벨을 다시 이동하지 않음) =====
lab = em[['Patient_ID', 'ICULOS', 'SepsisLabel']].merge(
    label_ref, on=['Patient_ID', 'ICULOS'], how='outer', suffixes=('', '_orig'), indicator=True)
g = em.groupby('Patient_ID')['SepsisLabel']
onset_orig = label_ref[label_ref.SepsisLabel == 1].groupby('Patient_ID')['ICULOS'].min()
checks = {
    '행 수와 (환자, 시간) 키가 원본과 같음': len(lab) == len(label_ref) == len(em) and (lab['_merge'] == 'both').all(),
    '모든 행의 라벨이 원본과 같음': (lab['SepsisLabel'] == lab['SepsisLabel_orig']).all(),
    '환자별 라벨이 한 번 1이 되면 계속 1 (단조)': (g.cummax() == em['SepsisLabel']).all(),
    '라벨 시작 시각이 원본과 같음 (추가 이동 0h)': onset.sort_index().equals(onset_orig.sort_index()),
    't_sepsis = 라벨 시작 + 6h': em.groupby('Patient_ID')['t_sepsis'].first().dropna().astype(float)
                                     .equals((onset + 6).astype(float)),
    'features와 eval_meta의 row_id 순서가 같음': (em['row_id'].to_numpy() == np.arange(len(F))).all(),
    '패혈증 환자 수 dev 2,492 / test 440': em.groupby(['split', 'Patient_ID'])['SepsisLabel'].max()
                                          .groupby('split').sum().to_dict() == {'dev': 2492, 'test': 440},
}
for k, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {k}")
assert all(checks.values())
del lab


# %%
# ===== [3-3] 누수 검사 ③: 24시간 원칙 + 학습-API 일치 =====
# 학습용 계산(전체 행)과 참조용 계산(환자 1명의 최근 24행만 받는 API용)을 dev 행 표본에서 비교.
# 참조용은 24행 밖을 볼 수 없으므로, 두 값이 같으면 학습 특징도 24행 밖을 쓰지 않았다는 뜻.
rng = np.random.default_rng(42)
pos = em.groupby('Patient_ID').cumcount().to_numpy()
dev_idx = np.flatnonzero(DEV)
pick = np.unique(np.concatenate([
    rng.choice(dev_idx, 2000, replace=False),                                          # 무작위
    rng.choice(dev_idx[em['SepsisLabel'].to_numpy()[dev_idx] == 1], 500, replace=False),  # 양성 행
    rng.choice(dev_idx[pos[dev_idx] < 23], 500, replace=False),                        # 기록 24h 미만
]))
t0 = time.time()
bad_rows, bad_feat = 0, pd.Series(0, index=names)
pad_bad = 0
for i in pick:
    win = clean_df.iloc[i - min(pos[i], 23): i + 1]
    ref = sf.build_features_window(win, cfg).to_numpy(float)
    ok = np.isclose(F.iloc[i].to_numpy(), ref, rtol=1e-7, atol=1e-9, equal_nan=True)
    if not ok.all():
        bad_rows += 1
        bad_feat[~ok] += 1
    if pos[i] < 23:                                  # 앞을 빈 행으로 채워 24행으로 만들어도 같은지
        pad = pd.DataFrame(np.nan, index=range(23 - pos[i]), columns=win.columns)
        padded = pd.concat([pad, win], ignore_index=True)
        pad_bad += not np.allclose(sf.build_features_window(padded, cfg).to_numpy(float), ref, equal_nan=True)
n_early = int((pos[pick] < 23).sum())
print(f'비교 행 {len(pick):,}개 (양성 {int(em.SepsisLabel.to_numpy()[pick].sum()):,}, 기록 24h 미만 {n_early:,}) '
      f'| {time.time() - t0:.0f}초')
print(f"  [{'OK' if bad_rows == 0 else 'FAIL'}] 학습용 = 참조용(API): 불일치 행 {bad_rows} / {len(pick):,}")
print(f"  [{'OK' if pad_bad == 0 else 'FAIL'}] 빈 행 채움(padding) 무관: 불일치 {pad_bad} / {n_early:,}")
if bad_rows:
    print(bad_feat[bad_feat > 0].sort_values(ascending=False).head(10))
assert bad_rows == 0 and pad_bad == 0


# %%
# ===== [3-4] 누수 검사 ④: 미래 차단 + 24시간 밖 무시 (학습용 계산 자체를 직접 흔들어 보기) =====
# 환자 300명마다 시각 t를 고르고 (1) t 이후 기록을 지우고 (2) t-23보다 오래된 기록을 무작위 값으로 바꾼 뒤
# 학습용 계산을 다시 돌림 → t 시점 특징이 원래 값과 '완전히' 같아야 함
# 음성 대조: 환자 전체 평균 HR(일부러 만든 누수 특징)은 같은 조작에서 값이 바뀌어야 함 → 검사가 누수를 잡을 수 있다는 증거
CLIN = cfg['feature_vars']['vitals_window'] + cfg['feature_vars']['labs']
rng = np.random.default_rng(7)
n_rows = em.groupby('Patient_ID').size()
cand = n_rows[(n_rows >= 48) & n_rows.index.isin(em.loc[DEV, 'Patient_ID'].unique())].index
pts = np.sort(rng.choice(cand, 300, replace=False))          # 학습용 계산은 환자 순 정렬 필요
start_of = em.groupby('Patient_ID')['row_id'].min()

pieces, targets = [], []
for p in pts:
    s, n = start_of[p], n_rows[p]
    t = int(rng.integers(30, n - 1))                        # 앞에 24행 넘는 과거, 뒤에 미래가 남도록
    piece = clean_df.iloc[s: s + t + 1].copy()              # (1) 미래 삭제
    old = piece.index[: t + 1 - 24]                          # (2) 24행 밖 과거를 무작위 값으로
    for v in CLIN:
        lo, hi = cfg['plausible_ranges'][v]
        piece.loc[old, v] = rng.uniform(lo, hi, len(old))
    pieces.append(piece)
    targets.append(s + t)
mod = pd.concat(pieces)
F_mod = sf.build_features(mod, cfg)
last_rows = F_mod.groupby(mod['Patient_ID'].to_numpy()).tail(1)
same_feat = np.array_equal(last_rows.to_numpy(), F.iloc[targets].to_numpy(), equal_nan=True)

leaky = clean_df.groupby('Patient_ID')['HR'].transform('mean')            # 일부러 만든 누수 특징
leaky_mod = mod.groupby('Patient_ID')['HR'].transform('mean')
changed = (~np.isclose(leaky_mod.groupby(mod['Patient_ID']).tail(1).to_numpy(),
                       leaky.iloc[targets].to_numpy(), equal_nan=True)).sum()
print(f"  [{'OK' if same_feat else 'FAIL'}] 미래 삭제 + 24시간 밖 조작 후에도 125+1개 특징 완전히 동일 (환자 {len(pts)}명)")
print(f"  [{'OK' if changed > 0.9 * len(pts) else 'FAIL'}] 음성 대조: 누수 특징은 {changed}/{len(pts)}명에서 값이 바뀜 "
      f"→ 이 검사는 누수를 잡아낼 수 있음")
assert same_feat and changed > 0.9 * len(pts)
del mod, F_mod, leaky, leaky_mod


# %%
# ===== [3-5] 기준선 점검: fold 0 LightGBM (모델링 인계 전 '특징이 제대로 작동하는가' 확인용) =====
# dev만 사용 (fold 1~4 학습, fold 0 검증). test는 열지 않음. 하이퍼파라미터 탐색 없음.
# 비교: 같은 모델에 ICULOS를 넣으면 → 시간 지름길이 얼마나 성능을 부풀리는지 확인 (본 모델에는 넣지 않음)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

y = em['SepsisLabel'].to_numpy()
tr = DEV & (FOLD != 0)
va = DEV & (FOLD == 0)
le60 = em['iculos_le60'].to_numpy() == 1
X = F[model_cols].astype(np.float32)
params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)


def evaluate(Xtr, Xva, label):
    t0 = time.time()
    clf = lgb.LGBMClassifier(**params).fit(Xtr, y[tr])
    p = clf.predict_proba(Xva)[:, 1]
    yv, lv = y[va], le60[va]
    row = {'model': label, 'AUROC': roc_auc_score(yv, p), 'AUPRC': average_precision_score(yv, p),
           'AUROC (ICULOS<=60)': roc_auc_score(yv[lv], p[lv]), 'sec': round(time.time() - t0)}
    return clf, row


clf, r1 = evaluate(X[tr], X[va], f'우리 특징 {len(model_cols)}개')
X['ICULOS'] = em['ICULOS'].to_numpy(np.float32)             # 비교용으로만 추가 (저장 파일에는 없음)
_, r2 = evaluate(X[tr], X[va], '+ ICULOS (비교용)')
res = pd.DataFrame([r1, r2]).set_index('model').round(3)
print(f'[fold 0 검증] 학습 행 {tr.sum():,} / 검증 행 {va.sum():,} (양성 비율 {100 * y[va].mean():.2f}%)')
print(res)
imp = pd.Series(clf.booster_.feature_importance('gain'), index=model_cols)
imp = (imp / imp.sum() * 100).sort_values(ascending=False)
print('\n[중요도 상위 15개 (gain %)]')
print(imp.head(15).round(1).to_string())
assert 0.70 <= r1['AUROC'] < 0.97, '기준선 AUROC가 예상 범위를 벗어남 (특징 오류 또는 누수 의심)'
del X


# %%
# ===== [3-6] 1~3단계 결과 요약: 4단계(모델링 준비)로 넘길 파일 =====
print('[Drive에 저장된 파일]')
for p in [CFG_PATH, V2_BACKUP, f'{SRC_DIR}/sews_features.py', FEAT_PATH, META_PATH, REG_PATH, MISS_PATH]:
    print(f'  {os.path.relpath(p, DRIVE_DIR):40s} {os.path.getsize(p) / 1e6:8.2f} MB')
print(f'\n특징: 모델 입력 {len(model_cols)}개 + 선택 {len(names) - len(model_cols)}개 | 행 {len(F):,} | '
      f'누수 검사 4종 통과 | fold 0 기준선 AUROC {r1["AUROC"]:.3f} (ICULOS<=60 {r1["AUROC (ICULOS<=60)"]:.3f})')


# %% [markdown]
# ## 정상일 때 결과 (전체 데이터로 미리 실행해 확인한 값)
#
# | 셀 | 확인할 값 |
# |---|---|
# | [1-0] | master 1,552,210행 / 환자 40,336명 (dev 행 1,318,889 / test 행 233,321), 시작 config v2 |
# | [1-2] | 모델 입력 특징 125개 + 선택 1개 `['shock_index']`. 묶음별 vital_window 66, lab 36, measurement 6, sofa 5, static 5, derived 4, qsofa 3 |
# | [1-3] | Calcium<3 → Calcium_ionized 8,439행, Resp 범위 밖 1,473행, Age 100→90 12,439행 (모두 dev 기준). Age 최댓값 90 |
# | [2-0] | 손계산 답 33개 = 참조용 = 학습용 |
# | [2-1] | 1,552,210행 x 126열, 값 범위 검사 통과 (계산 시간 약 1분 반) |
# | [2-2] | dev 양성 행 23,723 (1.80%), 패혈증 환자 2,492명 (첫 행부터 양성 366명), early_window 행 11,937 |
# | [2-3] | features.parquet 약 225 MB, eval_meta.parquet 약 8 MB. 결측률 최고 Bilirubin_total_delta_prev 96.56% |
# | [3-1]~[3-2] | `[OK]` 6개 + 7개 |
# | [3-3] | 비교 행 2,999개 중 불일치 0, 빈 행 채움 불일치 0 / 1,800 |
# | [3-4] | 특징 완전히 동일 (환자 300명), 음성 대조 300/300 |
# | [3-5] | fold 0: 우리 특징 AUROC 0.799 / AUPRC 0.073 / ICULOS≤60 AUROC 0.792, + ICULOS(비교용) 0.833 / 0.113 / 0.804 |
#
# - LightGBM 결과는 Colab의 lightgbm 버전에 따라 소수 셋째 자리가 조금 다를 수 있습니다. 나머지 숫자는 같아야 합니다.
# - 최대 메모리 사용량 약 5.7 GB (기본 Colab 런타임 12.7 GB 안).
