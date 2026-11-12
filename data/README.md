# SEWS 데이터 파트: EDA · 전처리 · 특징 추출 · 로지스틱 회귀

ICU 패혈증 조기 경보 시스템(SEWS) 프로젝트(캡스톤디자인, 팀 포텐)에서 데이터 파트 코드를 모은 저장소다.
담당 범위는 EDA, 환자 단위 분리, 결측 처리, 특징 추출, qSOFA·가용 SOFA 계산, 로지스틱 회귀 기준선 모델이다.

원본은 Google Colab 노트북(ICU-2 ~ ICU-5)이다. `src/`와 `models/`는 Black(88자)·타입 힌트·docstring 기준을 맞췄고,
`eda/`와 `preprocess/`는 노트북 셀을 그대로 옮긴 것이다. 이 저장소의 `.py` 파일은 노트북의 셀 순서와 코드를 그대로 옮긴
percent 형식(`# %%` 셀)이라서, VS Code·Jupyter에서 셀 단위로 실행할 수도 있고 `python`으로 한 번에 실행할 수도 있다.

## 폴더 구조

```
eda/eda_icu.py                 # ICU-2: 데이터 병합, 환자 분리, Part 1 (A/B 비교), Part 2 (라벨·시간·신호), 0단계 저장
preprocess/icu_preprocess.py   # ICU-3: 1단계 정제, 2단계 특징 125+1개, 3단계 누수 검사·fold 0 기준선
preprocess/icu_model_prep.py   # ICU-4: fold별 스케일링·대치 파라미터, LSTM 입력 배열
models/train_logreg.py          # ICU-5: 로지스틱 회귀 5-fold CV, 보정, 경보 임계값, 결과 저장
models/train_lstm.py            # ICU-6 : LSTM 적용 5-fold CV, 보정, 경보 임계값 (AUROC 0.776, Utility 0.282), 결과 저장
src/sews_features.py            # 정제 + 24시간 창 특징 (학습용 계산 / API용 참조 계산)
src/sews_preprocess.py          # fold별 스케일링·대치, LSTM 창 생성
src/sews_eval.py                # 공용 평가: AUROC·AUPRC·조기 구간·환자 단위·Brier·ECE, PhysioNet 공식 Utility
```

## 데이터

PhysioNet/CinC Challenge 2019 training_setA/B (`.psv`, 40,336명, 1,552,210행).
데이터와 산출물(parquet, npy, 결과 폴더)은 용량 때문에 저장소에 올리지 않는다(`.gitignore`).
`training.zip`을 데이터 루트 폴더(`SEWS_ROOT`)에 두고 아래 순서로 실행하면 모든 산출물이 다시 만들어진다.

## 실행 순서

```bash
pip install -r requirements.txt
export SEWS_ROOT=/content/drive/MyDrive/capston   # training.zip 위치이자 산출물 저장 위치 (기본값)
export SEWS_RAW=/content/physionet                # training.zip 압축 해제 위치 (기본값)

python eda/eda_icu2.py                  # 약 5분. figures/eda, artifacts/ (실행 위치) + SEWS_ROOT/data, artifacts
python preprocess/icu3_preprocess.py    # 약 5분.  features.parquet, eval_meta.parquet, feature_registry.json
python preprocess/icu4_model_prep.py    # 약 5분.  preprocess_params/, seq_values.npy
python models/train_logreg.py --root $SEWS_ROOT   # 약 10~13분. results/logreg/
```

Colab에서는 각 파일을 셀 단위로 실행하면 된다(첫 셀이 Drive를 연결한다). 메모리 7GB 환경에서 전체 순서대로 실행해
노트북과 같은 결과가 나오는 것을 확인했다(EDA 출력 숫자 일치, 기준선 AUROC 0.799, 로지스틱 회귀 AUROC 0.769).
특징·전처리 모듈은 항상 저장소의 `src/`를 쓴다. 3·4단계는 모듈의 sha256을 `feature_registry.json`에 기록하고,
로지스틱 회귀는 실행 전에 같은 모듈인지 확인한다.

## 지킨 원칙

1. 환자 ID 기준으로 먼저 분리한다 (test 15% 봉인, dev 85%를 5-fold, set × 패혈증 여부 층화, SEED 42).
2. 대치값·스케일러·Calibration·임계값은 학습 데이터 또는 지정된 검증 데이터에서만 학습한다.
3. 예측 시각 이후 데이터를 특징에 쓰지 않는다 (모든 특징은 최근 24행만 사용).
4. 공식 `SepsisLabel`을 다시 6시간 이동시키지 않는다.
5. 완전한 SOFA 계산에 필요한 변수가 없으면 `계산 불가`로 표시한다 (호흡·신경·총점).

## 주요 결정

- **시간 지름길 제거**: 비패혈증 기록이 약 60시간에서 잘려 있어 ICULOS 단독 AUROC 0.635가 ICULOS ≤ 60 행에서는 0.491이다.
  ICULOS와 시간이 쌓이는 특징을 모두 금지했다. 넣으면 LightGBM AUROC가 0.799 → 0.833으로 부풀려진다.
- **병원 간 결측 차이**: 한 병원이라도 90% 이상 미측정인 변수 7개를 제외하고, 병원 간 미측정 차이가 큰
  DBP·pH·PaCO2·PTT는 측정 여부 지표를 만들지 않았다.
- **측정 패턴 특징**: 의료진이 의심할 때 내는 검사(Lactate, FiO2, 간 기능)에만 측정 횟수·마지막 측정 이후 시간을 만들었다.
- **SOFA +2 라벨 미사용**: 부분 SOFA +2 이벤트 비율이 비패혈증 12.2% / 패혈증 15.3%로 구분되지 않아 공식 라벨을 쓴다.

## 결과 (dev 5-fold OOF, 행 단위, test 미사용)

| 모델 | AUROC | AUPRC | AUROC (ICULOS ≤ 60) | 비고 |
|---|---|---|---|---|
| 로지스틱 회귀 (C=0.01, balanced) | 0.769 ± 0.008 | 0.064 ± 0.003 | 0.757 | Utility 0.273, Brier 0.0172 (Isotonic 보정) |
| LightGBM 기준선 (fold 0, 튜닝 없음) | 0.799 | 0.073 | 0.792 | ICU-3 [3-5] 점검용 |
| LSTM 5-fold CV, 보정, 경보 임계값 (AUROC 0.776, Utility 0.282) |

## 출처

- 데이터: Reyna et al., *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, Critical Care Medicine, 2020.
- Utility 함수: PhysioNet/CinC Challenge 2019 공식 평가 코드
  (https://github.com/physionetchallenges/evaluation-2019, BSD 2-Clause, Copyright (c) 2019 PhysioNet)의
  `compute_prediction_utility`와 같은 정의를 `src/sews_eval.py`에서 벡터화했다. 원본 반복문은 검사용 참조 구현(`official_utility`)으로 포함했다.
