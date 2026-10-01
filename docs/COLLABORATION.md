# 협업 규칙

## 브랜치

- 기본 브랜치: `main`
- 정상민: `feature/data-pipeline`
- 이택훈: `feature/modeling`
- 김진호: `feature/backend`
- 임경수: `feature/frontend`

한 브랜치에 여러 기능을 오래 쌓지 말고 작업 단위별 짧은 브랜치를 권장한다.

## Pull Request

1. 본인이 담당한 파일도 최소 한 명의 리뷰를 받은 뒤 병합한다.
2. 공통 계약을 바꾸면 영향을 받는 담당자의 리뷰가 필요하다.
3. 데이터·모델 파일과 `.env`는 커밋하지 않는다.
4. PR 설명에 작업 내용, 테스트 방법, 영향받는 파일을 적는다.

## 커밋 메시지

Conventional Commits를 사용한다.

```text
feat(data): add patient-level split
feat(model): add GRU training loop
feat(api): add batch prediction endpoint
feat(ui): add patient risk list
test(api): validate malformed observations
docs: update model handoff contract
```

## 계약 변경 시 리뷰 담당

| 변경 항목 | 필수 협의자 |
|---|---|
| 라벨·환자 분할·특징 이름 | 정상민, 이택훈 |
| 모델 입력·저장 형식 | 이택훈, 김진호 |
| API 요청·응답 JSON | 김진호, 임경수 |
| 경보 임계값·표시 문구 | 이택훈, 임경수 |
| Docker Compose | 김진호, 임경수 및 영향받는 담당자 |

## 완료 기준

- 코드와 설정이 커밋되어 있다.
- 테스트 또는 수동 검증 절차가 있다.
- 담당 문서와 파일 가이드가 갱신되어 있다.
- 다른 팀원의 환경에서 실행할 수 있다.

