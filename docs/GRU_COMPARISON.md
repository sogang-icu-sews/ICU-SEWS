# GRU v2·v3: 보정 및 경보 비교

[Colab에서 비교 노트북 열기](https://colab.research.google.com/github/sogang-icu-sews/ICU-SEWS/blob/codex/gru-colab/notebooks/ICU_SEWS_GRU_Compare.ipynb)

Drive 연결 후 위에서부터 실행한다. GPU·GRU 재학습·원본 ZIP은 필요 없다.
학습 노트북과 독립적인 Python 3.10 평가 환경을 설치한다.
기본 실험 폴더는 `MyDrive/SEWS_GRU/results/gru_v2_lr0003`와
`gru_v3_lr0003_no_weight`이다. 각 폴더의 `summary.json`, `oof_predictions.parquet`를 사용한다.
결과는 `MyDrive/SEWS_GRU/comparisons/v2_v3_<실행시각>`에 저장한다.

v2 + Platt를 선택한 이후의 최종 단일 GRU 학습·경보 기준 확정·test 평가는
[GRU_FINAL.md](GRU_FINAL.md)와 별도 최종 노트북을 사용한다.

## 평가 설계

각 원래 GRU fold 안에서 환자를 병원×패혈증 여부로 층화하여 3개 그룹으로 나눈다.
3번 회전하며 보정 학습 / 임계값 선택 / 평가를 서로 다른 환자 그룹이 맡는다.
각 행은 평가에 한 번만 참여하고 두 모델은 같은 그룹을 사용한다.
GRU의 gradient 학습에서는 원래 fold 전체가 제외되었으므로, 세 역할 모두 해당 GRU의
학습 환자에 포함되지 않는다. 다른 원래 fold의 OOF로 보정기를 학습하면 그 OOF를 만든 GRU가
현재 평가 환자로 학습했을 수 있으므로, 여기서는 같은 원래 fold의 예측만 사용한다.

하지만 GRU 최적 epoch·v2/v3 설정 선택에는 이미 개발 데이터 라벨이 사용되었다.
이 과정은 그 선택을 되돌리지 못하므로 **개발 분석이며 불편 추정이나 완전한 nested CV가 아니다.**
최종 확정 후 별도 Test에서 검증해야 한다. 프로그램은 Test 파일을 열거나 평가하지 않는다.
보정 방법이나 모델을 이 결과로 선택했다면 같은 결과를 독립적인 최종 성능으로 보고하지 않는다.

## 입력 검사

- 양 모델의 row_id·환자·시간·라벨·병원·fold를 정렬 후 일치 확인.
- 환자별 fold/병원 단일성, 중복 행, 시간 간격, 단조 라벨, 유한한 0~1 예측 확인.
- 저장소 환자 manifest의 dev 전체 환자·fold·병원·환자 결과와 대조, test 포함 거부.
- summary의 complete_5fold/test_evaluated 상태와 실제 fold별 행 수 대조.
- 원본 OOF·summary·manifest·실행 코드 해시 및 라이브러리 버전 기록.

## 보정 및 확률 지표

raw, Platt(logit 입력의 비가중 로지스틱 회귀), Isotonic을 모두 평가한다.
보정 방법의 자동 선택은 하지 않는다. 모든 후보를 같은 평가 환자에서 보고한다.
AUROC, Average Precision(AUPRC), Brier, log loss, 동일 너비 10-bin ECE,
양성 비율과 평균 예측을 저장한다. 전체·60시간 이내·병원 A/B 결과를 제공한다.

baseline_Brier는 해당 보정 학습 그룹의 양성 비율을 평가 그룹에 상수로 예측한 결과다.
평가 라벨로 상수 기준을 학습하지 않는다. Brier_skill = 1 - Brier/baseline_Brier이다.
ECE와 Brier만으로 보정 상태를 확정하지 말고 구간별 표본 수와 보정 곡선을 함께 확인한다.
분위수 10-bin 보정 곡선과 동일 너비 ECE는 구간 정의가 서로 다르다.

단일 Platt 보정기가 양의 기울기를 가지면 순위를 보존하지만, 여러 보정기의 평가 예측을
합친 값에서는 순위가 달라질 수 있다. Isotonic은 동점을 만들어 AUPRC/AUROC를 바꿀 수 있다.

## 경보 정책

모든 임계값은 평가 환자와 별개인 threshold 그룹에서 정한다.
아래 목표는 개발 비교용이며 임상 운영 기준이 아니다.

| 정책 | threshold 그룹의 선택 기준 |
|---|---|
| row_specificity95 | 공식 라벨 음성 시간 행의 오경보 비율 ≤5% |
| patient_false_alarm10 | 기록 전체가 비패혈증인 환자 중 한 번 이상 경보 받은 비율 ≤10% |
| max_utility | PhysioNet Utility 최대화, 동점은 더 적은 경보 선택 |

점수 동점을 함께 처리하므로 보수적인 임계값이 선택될 수 있다. 평가 그룹에서 목표가
그대로 달성된다고 보장하지 않으며 실제 특이도와 환자 오경보율을 함께 보고한다.
확률 보정이 순위를 유지하면 동일한 순위/오경보 목표의 경보 결과도 거의 같을 수 있다.
보정만으로 조기 감지나 경보 성능이 향상된다고 가정하지 않는다.

경보는 매 시간 독립적으로 `p >= threshold`로 정의하며 cooldown·재알림 억제를 적용하지 않는다.
`alarm_hours_per100h`는 100 관찰 시간당 경보 상태인 시간 수다. 별도 경보 에피소드 수가 아니다.
행 단위 FP에는 나중에 패혈증이 생긴 환자의 아직 라벨이 0인 시간이 포함된다.
비패혈증 환자 지표는 기록 전체의 라벨이 0인 환자만 대상으로 한다.

## 조기 감지와 Utility

첫 양성 라벨의 ICULOS +6으로 발병 시각을 역산한다. 직접 관측한 임상 발병 시각이 아니다.
첫 행부터 양성인 환자는 발병 시각을 복원하기 어려우므로 조기 감지/선행 시간 분모에서 제외한다.
제외 환자 수와 평가 가능한 양성 환자 수를 출력한다.

- early_recall_6to12h: 추정 발병 [-12,-6]시간에 한 번 이상 경보가 나온 환자 비율.
- pre_onset_recall_0to12h: 추정 발병 [-12,0)시간에 한 번 이상 경보가 나온 환자 비율.
- median_lead_hours_detected: [-12,0)에서 감지된 환자만의 최초 유효 경보 선행 시간 중앙값.
  감지된 환자만의 조건부 값이므로 감지율과 함께 읽는다.
- 첫 발병 12시간보다 더 이른 경보는 위 조기 지표의 성공으로 세지 않는다.
- Utility는 저장소 `data/src/sews_eval.py`의 공식 정의를 재사용한다. 처음부터 양성인 환자도
  공식 평가와 동일하게 포함한다. 조기 지표의 제외 규칙과 다르다.

시간 구간을 잘라 환자 결과가 달라지는 것을 피하려고 Utility·환자 경보 지표는 전체 기록으로
계산한다. 60시간 이내 서브셋은 확률 지표만 보고한다.

## 산출물

`probability_metrics.csv`, `alarm_metrics.csv`의 fold=-1은 합친 평가 결과,
fold=0~4는 기존 GRU fold별 결과다. `*_fold_summary.csv`는 원래 5개 fold의 평균·표준편차다.
15개 회전 그룹을 독립적인 15개 실험처럼 취급하지 않는다.

`patient_groups.csv`, `roles.csv`, `thresholds.csv`로 환자 역할과 선택한 임계값을 추적한다.
`predictions_<model>_<method>.parquet`에 보정 확률·상수 기준·정책별 경보 플래그를 보관한다.
`calibrators/`의 보정기와 임계값은 분할별 평가용이다. 최종 단일 모델의 배포 산출물이 아니며,
이 값들을 평균내서 최종 경보 기준으로 사용하지 않는다.
`metadata.json`이 생성된 실행만 완료된 것으로 간주한다.

## 참고

비교 실행의 표준 출력과 오류는 셀에 직접 표시하고
`comparisons/logs/v2_v3_<timestamp>.log`에도 저장한다. 실패 시 마지막 80줄이
예외 메시지에 포함된다. `CalledProcessError`만 보이는 이전 노트북을 사용 중이면
최신 노트북을 다시 열고 환경 준비 및 비교 실행 셀을 실행한다. GRU 재학습은 필요 없다.
실패한 결과 폴더를 덮어쓰지 않도록 재실행은 새 이름으로 시작한다.

로컬 Python 3.10에서 환자 역할 분리, 평가 라벨 오염 시 보정기·임계값 불변,
동점/float32 임계값, Utility 최적화와 공식 참조 계산, 첫 행 양성 환자 제외,
동일 OOF 행 및 출력 파일 검사를 수행했다. 실제 v2/v3 예측 파일은 Drive에 있으므로
실제 성능 비교 수치는 이 노트북을 실행한 후 확인해야 한다.

- [scikit-learn 확률 보정](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn 임계값 튜닝](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [PhysioNet 2019 데이터·평가 정의](https://physionet.org/content/challenge-2019/1.0.0/)
- [PhysioNet 공식 Utility 코드](https://github.com/physionetchallenges/evaluation-2019)
