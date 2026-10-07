# ICU Sepsis Early Warning System (SEWS)

PhysioNet Computing in Cardiology Challenge 2019의 ICU 시계열 데이터를 이용해 패혈증 위험을 조기에 예측하는 교육·연구용 End-to-End AI 프로젝트입니다.

> 이 저장소는 실제 진단이나 진료에 사용하는 의료기기가 아닙니다. 외부 검증과 의료진 검토 없이 임상 의사결정에 사용해서는 안 됩니다.

## 기본 설계

- 데이터: PhysioNet Challenge 2019 `training_setA`, `training_setB`
- 주 라벨: 공식 `SepsisLabel` — 이미 조기 예측 시점이 반영되어 있으므로 이중 시프트 금지
- 환자 ID: 데이터 출처와 파일명을 결합한 값(예: `A_p000001`)
- 데이터 분할: 환자 기준 Train/Validation/Test = 70/15/15, seed 42
- 관찰 창: 최근 24시간
- 모델: Logistic Regression 기준선 + XGBoost + GRU + Soft Voting
- 딥러닝: PyTorch
- 평가: AUROC, AUPRC, Sensitivity@95% Specificity, Brier Score, ECE, Utility Score
- 실험 관리: 로컬 MLflow
- 서비스: FastAPI + Streamlit
- 환경: Python 3.10, uv, Docker Compose, Ruff, Pytest

## 팀 역할

| 이름 | 역할 | 책임 |
|---|---|---|
| 정상민 | 데이터·임상 로직 | EDA, 환자 분리, 결측 처리, 특징 추출, qSOFA·가용 SOFA |
| 이택훈 | 머신러닝·딥러닝 | XGBoost, GRU, 앙상블, 평가, Calibration, SHAP |
| 김진호 | 백엔드 | FastAPI, 스키마, 모델 서빙, 테스트, 추론 지연 측정 |
| 임경수 | 프론트엔드 | Streamlit, 임상 시각화, 경보 UI, API 연동 |

세부 담당 파일과 해야 할 작업은 [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md), 협업 규칙은 [`docs/COLLABORATION.md`](docs/COLLABORATION.md)를 참고합니다.

## 빠른 시작

### 1. 로컬 개발 환경

```bash
uv sync --group dev
cp .env.example .env
```

Windows PowerShell에서는 다음 명령을 사용할 수 있습니다.

```powershell
uv sync --group dev
Copy-Item .env.example .env
```

### 2. 테스트와 코드 검사

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

### 3. 개발용 서비스 실행

```bash
docker compose up --build
```

- API 문서: <http://localhost:8000/docs>
- 대시보드: <http://localhost:8501>

초기 뼈대는 `SEWS_USE_MOCK_MODEL=true`로 실행됩니다. 이는 프론트엔드와 백엔드 통합용 가짜 예측이며 실제 모델 결과가 아닙니다. 학습된 모델을 연결한 뒤 반드시 `false`로 변경합니다.

## 데이터 배치

원본 데이터는 Git에 올리지 않습니다. 다운로드한 파일은 다음 경로에 둡니다.

```text
data/raw/
├── training_setA/
│   ├── p000001.psv
│   └── ...
└── training_setB/
    ├── p100001.psv
    └── ...
```

PSV 파일 하나가 환자 한 명이며 각 행은 1시간의 기록입니다. 같은 환자의 데이터가 서로 다른 데이터 분할에 포함되지 않도록 합니다.

## 저장소 구조

```text
configs/                 데이터·모델·서비스 설정
data/                    원본·중간·전처리 데이터(내용은 Git 제외)
artifacts/               모델·보정기·임계값(내용은 Git 제외)
src/icu_sews/data/       데이터 로딩과 환자 분할
src/icu_sews/features/   임상 점수와 Sliding Window 특징
src/icu_sews/models/     XGBoost, GRU, 앙상블
src/icu_sews/evaluation/ 평가·Calibration
src/icu_sews/api/        FastAPI 서버
src/icu_sews/dashboard/  Streamlit 대시보드
tests/                   자동 테스트
docs/                    설계·API·파일별 작업 문서
```

## 핵심 원칙

1. 환자 ID 기준으로 데이터를 먼저 분리한다.
2. 대치값·스케일러·Calibration은 학습 데이터 또는 지정된 검증 데이터에서만 학습한다.
3. 예측 시각 이후의 데이터를 특징에 사용하지 않는다.
4. 공식 `SepsisLabel`을 다시 6시간 이동시키지 않는다.
5. 대시보드는 모델을 직접 로드하지 않고 API만 호출한다.
6. 완전한 SOFA 계산에 필요한 변수가 없으면 `계산 불가`로 표시한다.

## 현재 상태

원본 PSV로 로지스틱 회귀 기준선을 학습하는 실행기를 제공합니다.
설치·실행·평가 규칙은 [`docs/LOGISTIC_BASELINE.md`](docs/LOGISTIC_BASELINE.md)를 참고합니다.

```powershell
.venv\Scripts\python.exe --version  # Python 3.10.13
.venv\Scripts\python.exe -m pip install -r requirements-baseline-lock.txt
.venv\Scripts\python.exe scripts/train_logistic.py
```

API는 아직 Mock 예측 경로를 사용합니다. 기준선은 별도 학습 산출물이며 API 연결,
XGBoost/GRU 학습과 임상 검증은 완료되지 않았습니다.
