# ICU-SEWS Mock 버전 외부 배포 완료 및 변경사항 보고

작성일: 2026-11-13

## 1. 배포 결과

ICU-SEWS의 Mock 버전을 Render에 배포했습니다. 현재 웹 브라우저에서 아래 주소로 대시보드에 접속할 수 있습니다.

- **외부 접속 URL:** https://icu-sews-mock.onrender.com
- **배포 플랫폼:** Render
- **Render 서비스명:** `icu-sews-mock`
- **배포 브랜치:** `main`
- **현재 실행 모델:** Mock 모델

외부 URL에서 대시보드가 열리는 것을 확인했습니다. 이번 작업은 실제 임상 예측 모델의 배포가 아니라, 실제 모델 통합 전 대시보드/API를 외부에서 실행하고 시연할 수 있도록 한 1차 배포입니다.

## 2. 배포를 위해 추가한 파일

PR [#7](https://github.com/sogang-icu-sews/ICU-SEWS/pull/7)에서 다음 5개 파일을 추가했습니다.

| 파일 | 목적 |
|---|---|
| `Dockerfile` | Render에서 실행할 Docker 이미지 구성 |
| `requirements-deploy.txt` | FastAPI 및 Streamlit Mock 실행에 필요한 최소 패키지 정의 |
| `scripts/start.sh` | FastAPI를 시작하고 헬스 체크 후 Streamlit 대시보드 실행 |
| `.dockerignore` | 가상환경, 원본 데이터, 아티팩트 등 불필요한 파일의 이미지 포함 방지 |
| `render.yaml` | Render 웹 서비스 및 환경 변수 설정 |

## 3. 기존 코드에 대한 영향

- 배포 설정 PR에서는 A/B/C 담당 모델·데이터 관련 기존 소스 파일을 수정하지 않았습니다.
- 기존 FastAPI와 Streamlit Dashboard 구조를 유지했습니다.
- Dashboard는 기존 API 클라이언트를 통해 FastAPI와 통신합니다.
- 배포 환경에서는 `SEWS_USE_MOCK_MODEL=true`로 설정해 Mock 모델을 사용합니다.
- 기존 Dashboard의 PSV/CSV 업로드 흐름을 외부 배포 환경에서도 사용할 수 있도록 구성했습니다.

이번 PR은 애플리케이션의 모델 구현을 변경한 것이 아니라, 기존 애플리케이션을 컨테이너 환경에서 실행하고 Render에 배포하기 위한 설정을 추가한 것입니다.

## 4. 확인한 사항

- 로컬 테스트: `pytest` 33개 통과
- Dashboard와 FastAPI 간 통신 및 Mock 예측 요청 확인
- Render 배포 로그에서 서비스 실행 완료 확인
- 외부 URL에서 대시보드 접속 확인

위 테스트 및 접속 확인은 Mock 구성의 동작 확인이며, 실제 모델의 예측 성능이나 임상적 유효성을 검증한 것은 아닙니다.

## 5. 실제 모델 통합 시 후속 작업

B/C 담당 실제 모델이 준비되면 다음 작업이 필요합니다.

1. 실제 모델의 추론 코드와 필요한 전처리·후처리 로직을 PredictorService에 연결
2. 모델 파일 및 전처리기·보정기·임계값 등 필요한 아티팩트의 배포 방식 결정
3. 실제 모델 실행에 필요한 패키지를 배포 의존성에 추가
4. Mock 설정을 실제 모델 실행 설정으로 변경
5. 테스트 및 외부 환경 검증 후 Render 재배포

실제 모델 통합 시 기존 API 입출력 계약과 Dashboard 동작을 가능한 한 유지하는 방향으로 협의가 필요합니다.

## 6. GitHub 변경 이력

- PR: [#7 — build(deploy): add Render mock deployment configuration](https://github.com/sogang-icu-sews/ICU-SEWS/pull/7)
- 병합 커밋: `434fefe`
- 저장소: https://github.com/sogang-icu-sews/ICU-SEWS

---

**요약:** ICU-SEWS Mock 버전의 외부 배포 및 브라우저 접속 확인을 완료했습니다. 현재 서비스는 Mock 예측을 사용하므로 실제 환자 위험도 판단이나 임상적 의사결정에 사용해서는 안 됩니다. 실제 모델 통합과 검증은 별도 후속 작업입니다.
