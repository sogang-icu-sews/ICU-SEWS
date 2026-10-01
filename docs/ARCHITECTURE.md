# 시스템 아키텍처

## 데이터 및 서비스 흐름

```text
PhysioNet PSV
    ↓ 정상민
환자 식별 → 환자 단위 분할 → 인과적 전처리 → 24시간 특징/시퀀스
    ↓ 이택훈
Logistic Regression / XGBoost / GRU → Soft Voting → Calibration → SHAP
    ↓ 김진호
FastAPI: health / predict / predict_batch / explain
    ↓ 임경수
Streamlit: 환자 목록 / 상세 추이 / Warning·Alert / 설명
```

## 경계 규칙

1. 데이터 모듈은 환자별 분할 ID와 모델 입력 파일을 생성한다.
2. 모델 모듈은 API나 Streamlit을 import하지 않는다.
3. API는 저장된 모델과 전처리기를 로드하며 모델 학습을 수행하지 않는다.
4. 대시보드는 모델 또는 전처리 파일을 직접 읽지 않고 HTTP API만 호출한다.
5. 모든 설정은 `configs/` 또는 환경 변수에 두며 코드에 경로와 임계값을 하드코딩하지 않는다.

## 데이터 누수 방지

- 원본 파일명에서 환자 ID를 만든 다음 환자 단위로 먼저 분할한다.
- 결측 대치값, 스케일러, 특징 선택기는 Train에서만 학습한다.
- Calibration은 Validation 또는 Out-of-Fold 예측으로 학습한다.
- Test는 최종 잠금 평가에만 사용한다.
- 한 시점의 특징은 해당 시점과 과거 23시간만 사용한다.
- 공식 `SepsisLabel`에는 조기 예측 시점이 반영되어 있으므로 추가로 6시간 이동시키지 않는다.

## 산출물 계약

```text
data/processed/patient_split.parquet
data/processed/tabular_features.parquet
data/processed/sequences.npz
artifacts/preprocessor.pkl
artifacts/tree_model.pkl
artifacts/gru_model.pt
artifacts/calibrator.pkl
artifacts/thresholds.json
artifacts/model_metadata.json
```

파일명은 초기 계약이며 구현 과정에서 변경할 수 있다. 변경 시 `docs/FILE_GUIDE.md`, API 계약, 관련 설정을 같은 Pull Request에서 갱신한다.

