# v2 + Platt 최종 학습·경보 기준·test 평가

[최종 Colab 노트북](https://colab.research.google.com/github/sogang-icu-sews/ICU-SEWS/blob/codex/gru-colab/notebooks/ICU_SEWS_GRU_Final.ipynb)

GPU 런타임에서 위에서부터 실행한다. Drive의 기존 `SEWS_GRU/prepared`를 재사용하며
없을 때만 `capston/training.zip`을 처리한다. 학습은 별도 Python 3.10 / PyTorch 2.8.0 CUDA 환경이다.
노트북 커널의 Python 버전은 바꾸지 않는다. 이전 실험의 가중치를 이어 학습하지 않고,
선택한 v2 설정으로 최종 단일 GRU를 처음부터 학습한다.

## test 확인 전 고정한 설계

| 항목 | 설정 |
|---|---|
| GRU | hidden 64, 1 layer, dropout 0.2, static 5개 |
| 입력 | 기존 v4 정제·24시간 창·측정 mask·delta·가변 길이 처리 유지 |
| 학습 | Adam, lr 0.0003, batch 128, clip grad 1.0, seed 42 |
| 양성 가중치 | 최종 GRU 학습 행의 음성/양성 비율 |
| epoch | 3회 고정: v2 개발 best epoch [2,2,3,3,3]의 중앙값 |
| 보정 | logit(clip(p,1e-7,1-1e-7))를 입력으로 하는 비가중 Platt |
| 경보 | 비패혈증 환자 오경보율 10% 목표, 매시간 p_calibrated >= threshold |

학습 종료 시점을 보정/임계값 설정 그룹으로 다시 고르지 않는다. 3 epochs는 기존 개발 결과에서
정한 선택이며 새 데이터에서 최적이라는 보장은 없다. 이번 결과를 보고 epoch를 다시 고르면 안 된다.

## 환자 분할과 전처리

기존 `artifacts/patient_split.csv`의 dev 85% / test 15%를 그대로 보존한다.

- dev fold 0~3: GRU 학습. 27,428명, 전체 환자의 약 68%.
- dev fold 4: 6,857명을 병원×환자 패혈증 여부로 층화해 seed 42로 반분한다.
  Platt 학습 3,428명 / 경보 임계값 설정 3,429명. 각각 전체의 약 8.5%.
- test: 기존 6,051명, 약 15%. 재분할하거나 개발 환자를 추가하지 않는다.

`sews_preprocess.fit_seq`와 `fit_tabular`로 **GRU 학습 행만** 사용해 중앙값/IQR을 다시 산출한다.
그 파라미터를 보정·임계값·test에 동일하게 적용한다. 정적 변수 Age도 같은 규칙이다.
기존 `final_dev` 통계는 쓰지 않는다. 공통 정제 모듈과 해시 계약은 변경하지 않는다.
모든 창은 같은 환자의 현재/과거 관측만 포함한다.

Platt 그룹은 최종 GRU의 학습·학습 종료 결정에 참여하지 않는다. 임계값 그룹도 GRU나 Platt의
학습에 쓰지 않는다. 다만 이 개발 환자들은 앞선 v2/v3 선택에 쓰였으므로 이 그룹의 점수를
독립적인 최종 검증으로 제시하지 않는다. 최종 성능은 test 결과로 보고한다.

## 경보 임계값

임계값 설정 그룹에서 비패혈증 환자별 최대 보정 확률을 계산한다. 경보가 한 번 이상 발생하는
비패혈증 환자 비율이 10% 이하가 되도록 동점을 보수적으로 처리해 단일 임계값을 고른다.
확률 임계값을 0.10이나 0.5로 고정한다는 뜻이 아니다. 같은 값을 전체 test와 병원 A/B에 적용한다.
test에서 실제 오경보율이 10%를 넘을 수도 있으며 test를 보고 임계값을 재설정하지 않는다.
이 목표는 연구 비교 기준이며 임상 운영 기준으로 검증된 것은 아니다.

매시간 경보 플래그를 평가한다. cooldown·연속 경보 통합·첫 경보 이후 중단은 적용하지 않는다.
Platt는 확률 해석을 보정하는 절차이고 조기 감지 개선을 보장하지 않는다.
저장되는 threshold 그룹 점수는 임계값을 맞춘 데이터의 점수로 명시한다.

## 실행과 재개

노트북의 기본 `RUN_NAME`은 `gru_v2_platt_final`이다.

1. Drive 연결.
2. 별도 저장소 `/content/ICU-SEWS-GRU-final`과 GPU 환경 준비.
3. 입력을 `/content/sews_gru_final_inputs`로 복사.
4. 최종 GRU 3 epochs → Platt → 임계값 → `frozen.json` 생성.
5. 고정된 모델로 test 평가 → 표·그래프·완료 표시 저장.
6. 결과 확인 셀.

최초 코드 commit을 `SEWS_GRU/final_code_pins/<RUN_NAME>.txt`에 기록하고 다음 런타임도 같은
버전을 사용한다. 진행 로그는 `SEWS_GRU/final_logs/<RUN_NAME>/`에 저장한다.
코랩 inline 그래프 설정이 별도 환경에 전달되는 문제를 피하려고 matplotlib import 전에 Agg를 지정한다.

학습 도중 연결이 끊기면 같은 RUN_NAME으로 준비 셀부터 다시 실행한다. 매 epoch의 모델,
optimizer, 난수 상태를 보존하고 마지막 완료 epoch부터 재개한다. 중간 epoch는 다시 계산한다.
완료된 fit은 파일 해시를 확인한 후 건너뛴다. test 도중 실패해도 고정된 동일 모델로 다시 계산할 수 있다.
`test_summary.json`이 있으면 결과 파일의 해시를 확인하고 저장된 결과를 재사용한다.

코드·데이터·버전·설정이 달라지면 재개를 거부한다. test를 본 뒤 RUN_NAME을 바꾸어 새 모델을 고르고
같은 test를 반복 검증용으로 쓰지 않는다. 파일 보호는 같은 실행 폴더 안의 실수 방지이며,
사용자가 다른 폴더에서 test를 재사용하는 것까지 차단하지는 않는다.
test 결과만 볼 때는 Drive 연결과 결과 확인 셀만 실행하면 된다.

## 최종 파일

저장 위치: `MyDrive/SEWS_GRU/final/gru_v2_platt_final/`

- `final.pt`: 최종 가중치·구조·전처리 설정·통계·입력 순서·epoch.
- `platt.json`: slope/intercept/clip/적용 공식, 보정 그룹의 상수 예측 기준 확률.
- `threshold.json`: 보정 확률 기준의 임계값, 비교 연산자, 경보 정책.
- `preprocessing.json`, `patient_roles.csv`, `history.json`: 재현에 필요한 기록.
- `run_identity.json`, `frozen.json`: 입력·코드·라이브러리 버전 및 고정 파일의 해시.
- `latest.pt`: 재개용 체크포인트. 직접 생성한 신뢰할 수 있는 파일만 읽는다.
- `calibration_predictions.parquet`, `threshold_predictions.parquet`: 보정·임계값 설정 예측.
- `test_predictions.parquet`: test 시간 행별 raw/Platt 확률·경보.
- `test_probability_metrics.csv`: 전체·A/B·ICULOS≤60의 raw/Platt 확률 지표.
- `test_alarm_metrics.csv`: 전체·A/B의 **동일 고정 임계값** 경보 성능.
- `test_reliability_bins.csv`, `test_calibration*.png`: 보정 곡선과 구간별 행 수.
- `test_summary.json`: 최종 완료 표시·지표·한계·test 결과 파일 해시.

확률 지표는 AUROC/AP(AUPRC), Brier, 보정 그룹 양성 비율을 사용한 baseline Brier,
Brier skill, log loss, 동일 너비 10구간 ECE이다. 보정 곡선은 분위수 구간이라 ECE의 구간과 다르다.
Utility·조기 감지는 비교 노트북과 같은 정의다. 첫 양성 라벨 +6시간으로 발병 시각을 추정하고
첫 행부터 양성인 환자는 조기 감지·선행 시간에서만 제외한다. Utility에는 포함한다.
60시간 이내 경보 지표는 환자 기록을 잘라 결과가 달라지는 것을 피하려고 계산하지 않는다.

기존 비교의 fold별 보정기/임계값을 섞지 않는다. API 연결에는 이 최종 GRU·전처리·Platt·임계값
묶음을 사용한다. 현재 API 연결은 별도 작업이다. 점수는 단일 모델의 점 추정치로 신뢰구간은 포함하지 않는다.

## 검증 범위

최종 예측을 고정한 채 환자 단위 95% 신뢰구간을 추가하려면
[GRU_BOOTSTRAP.md](GRU_BOOTSTRAP.md)의 별도 CPU 노트북을 사용한다.
기존 최종 코드·모델·동결 파일을 변경하거나 test에 맞춰 재학습하지 않는다.

합성 환자 데이터로 학습/보정/임계값/test 역할 분리, holdout 변화에 대한 전처리 불변,
중단 후 재개와 연속 학습의 가중치 일치, fit 단계의 test 예측 제외, 임계값 예산,
test 결과·그래프 생성, 완료 결과 재사용, 고정 파일 변경 거부를 검사한다.
실제 Colab GPU 학습이나 사용자 Drive의 최종 test 성능을 로컬에서 실행한 것은 아니다.

2026-11-11 검증: 로컬 Python 3.10 CPU에서 전체 pytest **32개 통과**.
신규 Python 코드의 Ruff 검사·포맷 검사 및 노트북 6개 코드 셀의 문법 검사도 통과했다.
