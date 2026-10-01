# 파일별 작업 및 담당자

이 문서는 “누가 어느 파일에서 무엇을 했고 다음에 무엇을 해야 하는지”를 추적하기 위한 기준 문서다. 새 파일을 추가하거나 책임이 바뀌면 같은 Pull Request에서 이 문서를 수정한다.

## 공통·환경 파일

| 파일 | 초기 작성 내용 | 다음 작업 | 담당 |
|---|---|---|---|
| `README.md` | 프로젝트 개요, 기본값, 실행법, 팀 역할 | 실제 성능·최종 실행법 갱신 | 전원 |
| `pyproject.toml` | Python 3.10, 런타임·개발 의존성, Ruff/Pytest 설정 | 설치 검증 후 버전 잠금 | 전원 |
| `.python-version` | 기본 Python 3.10.13 지정 | 팀 환경 합의 시 수정 | 전원 |
| `.env.example` | API·모델·MLflow 환경 변수 예시 | 비밀값 없이 새 변수 문서화 | 김진호 |
| `.gitignore` | 환자 데이터, 모델, 로그, 비밀정보 제외 | 신규 생성물 반영 | 전원 |
| `docker-compose.yml` | API·대시보드 연결과 Health Check | MLflow 추가, 운영 리소스 제한 | 김진호·임경수 |
| `Dockerfile.api` | Python 3.10 API 이미지 | 의존성 계층·이미지 크기 최적화 | 김진호 |
| `Dockerfile.dashboard` | Streamlit 이미지 | 캐시·헬스체크 보완 | 임경수 |
| `.github/workflows/ci.yml` | Ruff와 Pytest CI | 커버리지·Docker 빌드 검사 | 전원 |
| `data/README.md` | 데이터 단계별 저장 원칙 | 출처·버전·체크섬·생성 명령 | 정상민 |
| `artifacts/README.md` | 모델 산출물 계약 | 최종 버전 배포 방식 | 이택훈·김진호 |

## 설정 파일

| 파일 | 초기 작성 내용 | 다음 작업 | 담당 |
|---|---|---|---|
| `configs/data.yaml` | 데이터 경로, 70/15/15, seed 42, 24시간 창, 공식 라벨 | 실제 처리 결과와 데이터 버전 추가 | 정상민 |
| `configs/model.yaml` | XGBoost+GRU+Soft Voting, Platt, 임시 경보값 | 검증 결과로 파라미터·임계값 갱신 | 이택훈 |
| `configs/serving.yaml` | API·지연 측정·UI 정책 | 실제 배포 조건 반영 | 김진호·임경수 |

## 데이터·특징 파일

| 파일 | 초기 작성 내용 | 다음 작업 | 담당 |
|---|---|---|---|
| `src/icu_sews/config.py` | YAML 설정 로더 | 설정 스키마 검증 | 공동 |
| `data/loaders.py` | 파일명 기반 환자 ID, PSV 로드·통합·정렬 | 40개 열·범위 검증, 대용량 최적화 | 정상민 |
| `data/split.py` | 계층화된 환자 단위 70/15/15 분할 | 분할 파일 저장·재사용 CLI | 정상민 |
| `features/clinical_scores.py` | 결측을 숨기지 않는 qSOFA | 가용 SOFA 구성요소와 검사 | 정상민 |
| `features/windows.py` | 환자별 인과적 rolling 통계 뼈대 | IQR, slope, ROC, CV, 결측 특징 | 정상민 |

## 모델·평가 파일

| 파일 | 초기 작성 내용 | 다음 작업 | 담당 |
|---|---|---|---|
| `models/base.py` | 모델과 API 사이 예측 계약 | 모델 메타데이터·설명 계약 확장 | 이택훈·김진호 |
| `models/xgboost_model.py` | XGBoost fit/predict/save wrapper | CV, 튜닝, class weight, SHAP | 이택훈 |
| `models/gru.py` | 기본 GRU 네트워크 | Masking, 학습 루프, Focal Loss | 이택훈 |
| `models/ensemble.py` | Soft Voting 함수 | OOF 기반 가중치 선택 | 이택훈 |
| `evaluation/metrics.py` | AUROC, AUPRC, Brier, ECE | 공식 Utility·임계값·신뢰구간 | 이택훈 |
| `evaluation/calibration.py` | Platt Scaling wrapper | Isotonic 비교와 저장 | 이택훈 |

## 백엔드·프론트엔드 파일

| 파일 | 초기 작성 내용 | 다음 작업 | 담당 |
|---|---|---|---|
| `api/schemas.py` | 24시간 요청과 예측 응답 스키마 | 모든 변수 범위, 배치·설명 스키마 | 김진호 |
| `api/service.py` | 모델 생명주기와 명시적 Mock 예측 | 실제 모델·전처리·Calibration 연결 | 김진호·이택훈 |
| `api/main.py` | `/health`, `/predict` | `/predict_batch`, `/explain`, 로깅 | 김진호 |
| `dashboard/api_client.py` | API Health·Predict 호출 | 재시도·타임아웃·오류 분류 | 임경수 |
| `dashboard/app.py` | Mock 통합 화면과 위험도 요청 | 환자 목록, 상세, SHAP, Calibration | 임경수 |

## 테스트·문서 파일

| 파일 | 초기 작성 내용 | 다음 작업 | 담당 |
|---|---|---|---|
| `tests/test_data_loading.py` | 환자 ID와 시간 정렬 테스트 | setA/B·오류 파일·중복 검사 | 정상민 |
| `tests/test_clinical_scores.py` | qSOFA 경계값·결측 테스트 | SOFA 구성요소 테스트 | 정상민 |
| `tests/test_ensemble.py` | Soft Voting 계산 테스트 | 가중치·배열 오류 사례 | 이택훈 |
| `tests/test_api.py` | Health와 Predict 계약 테스트 | 배치·오류·실제 모델 테스트 | 김진호 |
| `docs/ARCHITECTURE.md` | 모듈 경계와 누수 방지 원칙 | 최종 Mermaid 다이어그램 | 전원 |
| `docs/API_CONTRACT.md` | 요청·응답 초안 | 백엔드/프론트 확정 계약 | 김진호·임경수 |
| `docs/COLLABORATION.md` | 브랜치·리뷰·커밋 규칙 | 팀 합의 사항 반영 | 전원 |

