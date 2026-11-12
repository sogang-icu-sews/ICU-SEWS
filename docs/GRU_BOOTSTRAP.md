# 최종 test 환자 단위 bootstrap 95% 신뢰구간

[CPU Colab 노트북](https://colab.research.google.com/github/sogang-icu-sews/ICU-SEWS/blob/codex/gru-colab/notebooks/ICU_SEWS_GRU_Bootstrap.ipynb)

Drive의 `SEWS_GRU/final/gru_v2_platt_final/`에서 `test_predictions.parquet`,
`test_summary.json`, `frozen.json`, `platt.json`, `threshold.json`을 읽는다.
요약 표만으로는 시간 행의 환자 내 상관이나 확률 순위를 복원할 수 없으므로
AUROC·AP·Brier·Utility의 환자 단위 bootstrap을 계산할 수 없다.

## 방법

- 환자 N명 중 N명을 복원추출한다. 모든 관측 시간을 한 묶음으로 유지한다.
- 일반(non-stratified) cluster bootstrap, 2,000회, seed 42.
- 한 환자가 k번 뽑히면 그 환자의 모든 시간 행과 환자 수준 지표 기여를 k번 반영한다.
- 시간별 지표는 원래 평가처럼 시간 행 가중 방식이다. 환자별 AUROC를 평균내는 방식이 아니다.
- 확률 순서/동점을 사전 정렬하고 환자별 가산 통계를 재사용한다. 실제로 환자를 k번 복제하고
  복제마다 새 ID를 부여한 뒤 재평가한 것과 동일한 결과를 테스트로 검증한다.
- 환자별 Utility의 평균 대신 복원추출 표본 전체의 Utility 분자·분모를 합산한 비율을 계산한다.
- 검출 환자의 선행 시간 중앙값도 환자 추출 횟수를 반영한다.
- 각 지표의 재추출 분포에서 2.5/97.5 percentile을 구한다. 원래 test 점수를 estimate로 보고한다.
- 모델·Platt·임계값은 고정한다. 재학습이나 임계값 재설정은 하지 않는다.

전체, 병원 A/B, ICULOS≤60에 동일한 전체 환자 복원추출을 적용한다. 병원별 환자 수와
양성 비율은 표본마다 변할 수 있다. ≤60시간 그룹은 확률 지표만 계산한다.
정의할 수 없는 지표(예: 양성 환자가 없는 표본의 조기 감지율)는 제외하고 유효 반복 수를 보고한다.
각 지표의 개별 95% 구간이며 다중 비교를 보정한 동시 신뢰구간은 아니다.

## 검증 및 한계

원본 파일 해시, 공식 test 환자 목록·라벨, 시간 연속성, 동결 임계값과 경보 플래그 일치를 확인한다.
bootstrap 원점수가 최종 test 보고서와 일치해야 결과를 저장한다.
원본 `frozen.json`과 모델·최종 평가 파일을 변경하지 않는다. 별도 저장소와 CPU 환경에서 실행한다.

현재 **고정된 모델의 test 표본 변동성**에 대한 신뢰구간이다. 학습 seed·개발 데이터·모델 선택·
Platt 학습·임계값 선택의 불확실성은 포함하지 않는다. 환자를 독립적인 추출 단위로 가정하며,
새로운 병원으로 일반화할 때의 병원 간 변동을 추정하는 방법도 아니다.

합성 데이터에서 중복 환자의 명시적 복제와 가중 방식의 지표 일치, 경계 확률·동점,
단일 클래스·빈 하위 표본, 재현성, percentile 계산, 고정 파일 변경 거부를 검증한다.
실제 최종 예측은 사용자 Drive에 있으므로 Colab 실행 전 실제 신뢰구간 수치를 산출했다고 주장하지 않는다.

## 결과

`SEWS_GRU/confidence_intervals/gru_v2_platt_final_<timestamp>/`:

- `bootstrap_95ci.csv`: scope, metric, estimate, ci_lower, ci_upper, 유효/요청 반복 수.
- `bootstrap_replicates.csv`: 재추출별 지표.
- `metadata.json`: 입력·코드 해시, 방법, seed, 반복 수, 한계.

로그는 인접 `logs/` 폴더에 저장한다. 중단 후 같은 seed로 재실행하면 처음부터 계산하며,
최종 GRU 학습이나 test 추론을 다시 할 필요는 없다.

환자 내 반복 측정을 묶어 추출하는 근거:
[Hmisc repeated-measurements bootstrap](https://search.r-project.org/CRAN/refmans/Hmisc/html/rm.boot.html),
[ClusterBootstrap 방법 논문](https://link.springer.com/article/10.3758/s13428-019-01252-y).
