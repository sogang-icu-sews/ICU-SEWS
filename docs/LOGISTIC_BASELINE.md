# 로지스틱 회귀 기준선

담당: 이택훈

원본 PSV의 현재 시간 측정값 8개로 공식 `SepsisLabel`을 예측한다.
24시간 특징 생성이나 다른 담당자의 최종 전처리를 기다릴 필요가 없다.

## 실행

저장소 루트 `C:\ICU_SEWS`에서 PowerShell로 실행한다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
.venv\Scripts\python.exe scripts/train_logistic.py
```

최초 실행 환경은 기존 `.venv`의 Python 3.12다. 팀의 `pyproject.toml`과
`.python-version`은 Python 3.10 기준으로 유지한다. 이 실행기는 전체 앱을 설치하지
않고 `src`를 직접 읽는다. Python 3.10 팀 환경에서는 같은 최소 의존성 파일로
호환 버전을 설치하고 다시 검증해야 한다. 실행별 실제 버전은 metadata.json에 남긴다.

경로나 변수·하이퍼파라미터를 바꾸려면 다음 설정을 사용한다.

- `configs/data.yaml`: 실제 원본 경로, 분할 설정, 공통 분할 저장 폴더.
- `configs/logistic.yaml`: 특징 순서, 로지스틱 회귀 설정, 진단용 임계값.
- `--data-config`, `--model-config`, `--output`: 대체 설정/새 실행 폴더 지정.

기본 데이터 경로는 다음과 같다.

```text
data/raw/training_setA/training/*.psv
data/raw/training_setB/training_setB/*.psv
```

## 처리와 평가 규칙

1. 출처+파일명으로 환자 ID를 만들고, 환자별 `ICULOS`로 정렬한다.
2. 필요한 열, 수치 변환, 빈 파일, 무한값, 잘못된 시간·라벨, 중복 시간을 검증한다.
   생리학적 이상치 판정이나 환자 제외 규칙은 아직 적용하지 않는다.
3. 양성 환자 여부로 층화한 환자 단위 70/15/15 분할(seed 42)을 생성한다.
   `data/processed/patient_split.parquet`가 있으면 그대로 재사용한다.
   환자 집합·중복·분할 이름이 잘못된 파일은 오류로 중단하며 임의로 덮어쓰지 않는다.
4. Train에서만 변수별 중앙값 대치와 표준화를 학습한다. Train 전체에서 결측인
   변수는 0으로 채우고 메타데이터에 표시한다. 결측 표시 변수나 forward fill은 없다.
5. HR, O2Sat, Temp, SBP, DBP, MAP, Resp, Age를 입력한다.
   환자 ID, 시간, 출처, 라벨은 입력 특징에서 제외한다. 라벨 추가 시프트는 없다.
6. Train 전체로 로지스틱 회귀를 학습한다. 수렴 실패는 오류로 중단한다.
7. Validation에서 시간 행 단위 AUROC, AP, Brier, ECE 및 고정 0.5 임계값의
   Precision/Recall/Specificity를 계산한다. AP를 `auprc` 필드에 저장하며
   사다리꼴 PR 면적과 구분한다. 양성 비율도 함께 보고한다.
8. Test는 분할·구조 확인에만 포함하며 예측·성능 평가에는 사용하지 않는다.
   0.5는 진단용 기본값이며 최종 경보 임계값이 아니다. 보정, 튜닝, CV,
   공식 Utility Score, MLflow 및 API 연결은 이번 기준선 범위에 포함하지 않는다.

## 결과

실행마다 `artifacts/logistic_<UTC시각>/`에 저장한다. 기존 실행 폴더는 덮어쓰지 않는다.

| 파일 | 내용 |
|---|---|
| `logistic_pipeline.joblib` | 결측 처리기, 스케일러, 모델을 포함한 Pipeline |
| `metrics.json` | Validation 평가 결과 |
| `metadata.json` | 설정, 패키지·Python 버전, 원본 내용 해시, 분할 해시, 결측률 |
| `patient_split.parquet` | 해당 실행에서 사용한 공통 분할 사본 |
| `validation_predictions.parquet` | 환자 ID, 시간, 정답, 예측 확률 |
| `coefficients.csv` | 표준화된 변수에 대한 모델 계수; 인과적 효과가 아님 |

원본 데이터 및 생성 결과는 Git에서 제외된다. 공통 분할을 다른 담당자에게 전달할 때는
동일한 원본 데이터와 함께 사용한다. 기존 공통 분할이 있으면 설정의 seed보다 우선한다.

저장된 Pipeline에는 특징 이름과 순서도 포함된다. 추론 시 같은 열 순서의 DataFrame을
`predict_proba`에 전달하고 양성 클래스 열 `[:, 1]`을 사용한다. 저장 직후 재로딩해
동일 입력의 확률이 일치하는지 자동 검증한다. 로컬에서 만든 신뢰 가능한 모델만 로드한다.

## 검증

```powershell
.venv\Scripts\python.exe -m pytest tests/test_logistic.py tests/test_data_loading.py
.venv\Scripts\python.exe -m ruff check src/icu_sews/models/logistic.py src/icu_sews/models/train_logistic.py scripts/train_logistic.py tests/test_logistic.py
```

테스트는 전처리 통계의 Train 한정 학습, 공통 분할 재사용과 불일치 거부,
잘못된 라벨·중복 시간 거부, 모델 저장·검증 예측의 환자 범위를 확인한다.
