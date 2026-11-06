# API 계약 초안

담당: 김진호(백엔드), 임경수(프론트엔드), 이택훈(모델 출력)

## `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "mock_mode": true
}
```

## `POST /predict`

요청은 한 환자의 관측값 1~24개를 포함한다.
각 관측값의 ICULOS는 1 이상의 정수이며, 중복 없이 오름차순이어야 한다.
현재 API는 관측값 사이의 시간 간격이나 전체 시간 범위가 24시간 이내인지는 검사하지 않는다.
현재 로지스틱 모델은 마지막 관측값 하나를 사용한다.

```json
{
  "patient_id": "A_p000001",
  "observations": [
    {
      "ICULOS": 1,
      "HR": 88,
      "Resp": 23,
      "SBP": 95,
      "MAP": 68,
      "Lactate": 2.3
    }
  ]
}
```

현재 실제 로지스틱 모델의 응답 형식은 다음과 같다.
아래 risk_score는 형식을 보여주기 위한 예시이며 실제 계산 결과가 아니다.

```json
{
  "patient_id": "A_p000001",
  "timestamp": "2026-11-03T13:00:00Z",
  "risk_score": 0.42,
  "risk_level": "unclassified",
  "model_version": "logistic-baseline",
  "is_mock": false,
  "top_features": []
}
```

- `risk_score`: 모델의 클래스 1에 대한 예측 확률.
- `risk_level`: 실제 모델의 위험 등급 기준을 아직 연결하지 않아 `unclassified`를 반환한다. 정상이라는 의미가 아니다.
- `model_version`: 현재는 모델 종류를 나타내는 고정 문자열이며, 개별 학습 산출물을 식별하지는 못한다.
- `is_mock`: 실제 모델 실행 시 `false`, Mock 실행 시 `true`.
- `top_features`: 실제 SHAP 계산을 연결하지 않아 빈 목록을 반환한다.
- Mock은 기존 `normal`, `warning`, `alert` 등급을 사용한다.
- 프론트엔드는 `unclassified`를 별도로 표시해야 하며, 정상 등급으로 해석하면 안 된다.

## 남은 구현 작업

- `/predict_batch` 요청 크기 제한과 부분 실패 정책
- `/explain`의 SHAP 데이터 구조
- qSOFA 및 가용 SOFA 구성요소 응답
- 단위와 생리적 허용 범위 검증
- 24시간 미만 입력의 Padding/Masking 정책
- 인증·권한·감사 로그는 실제 배포 범위를 정한 뒤 별도 설계



## Resp 및 SBP 입력 검증

작성자: 김진호

`observations`의 각 기록에 적용되는 규칙이다.

- `ICULOS`는 필수이며, 1 이상의 정수여야 한다.
- `Resp`와 `SBP`는 생략하거나 `null`로 전달할 수 있다.
- `Resp`와 `SBP`를 생략하면 내부적으로 `None`으로 처리한다.
- 숫자로 변환 가능한 문자열은 허용한다. 예: `"Resp": "23"`
- 숫자로 변환할 수 없는 문자열은 거부한다. 예: `"Resp": "hello"`
- 무한대와 NaN은 허용하지 않는다.
- 입력 검증에 실패하면 HTTP 422와 오류 위치를 반환한다.
- 현재 Resp 및 SBP의 생리적 허용 범위 검사는 구현하지 않았다.

### 검증한 테스트

아래 테스트는 Mock 환경에서 수행했다.

- 상태 확인 및 정상 예측 요청.
- Resp 또는 SBP에 숫자로 변환할 수 없는 문자열을 보내면 HTTP 422 반환.
- Resp와 SBP를 모두 생략하거나 모두 null로 보내면 요청 허용.
- 숫자 입력과 숫자 문자열 입력의 risk_score 및 risk_level이 동일함.
- Resp 또는 SBP에 "NaN", "Infinity", "-Infinity" 문자열을 보내면
  HTTP 422와 해당 필드의 finite_number 오류 반환.
- SBP에 0을 보내면 기본값으로 대체하지 않고 계산에 사용.
- ICULOS 누락, 0 또는 음수 입력 시 HTTP 422 반환.
- ICULOS가 중복되거나 시간순으로 정렬되지 않은 경우 HTTP 422 반환.

실제 학습 모델의 예측 성능과 생리적 허용 범위는 검증 대상에 포함하지 않았다.
NaN과 무한대는 문자열 입력으로 검증했다.

### Mock 모드의 결측값 처리

- Resp가 생략되거나 null이면 Mock 계산에서 18.0을 사용한다.
- SBP가 생략되거나 null이면 Mock 계산에서 120.0을 사용한다.
- 입력값 0은 결측값과 구분하여 그대로 사용한다.
- 위 입력 검증 테스트는 Mock 환경에서 수행했다.
- 아래 모델 연결 검증은 별도의 임시 학습 모델을 사용하며, 팀의 실제 학습 모델 성능을 검증한 것은 아니다.
- 생리적 허용 범위 검사는 아직 구현하지 않았다.
---

## 로지스틱 모델 연결

### 실행 설정

서버 시작 시 환경변수를 읽고, 프로세스당 모델을 한 번 불러온다.
환경변수나 모델 파일을 변경했다면 서버를 재시작해야 한다.

Mock 실행:

```powershell
$env:SEWS_USE_MOCK_MODEL = "true"
uv run uvicorn icu_sews.api.main:app --host 127.0.0.1 --port 8000
```

실제 모델 실행:

```powershell
$env:SEWS_USE_MOCK_MODEL = "false"
$env:SEWS_MODEL_DIR = "artifacts/전달받은_모델_폴더"
uv run uvicorn icu_sews.api.main:app --host 127.0.0.1 --port 8000
```

실제 모델 경로는 전달받은 파일이 들어 있는 폴더로 변경해야 한다.
해당 폴더에는 다음 두 파일이 필요하다.

- `logistic_pipeline.joblib`
- `metadata.json`

프로젝트에서 생성하거나 팀원이 전달한 신뢰할 수 있는 모델 파일만 사용한다.

### 모델 로딩 검사

- 메타데이터의 `model_config.features`가 비어 있지 않은 문자열 목록인지 검사한다.
- 입력 항목 이름의 중복을 검사한다.
- 모델이 scikit-learn Pipeline인지 검사한다.
- 단계 이름과 순서가 `imputer`, `scaler`, `classifier`인지 검사한다.
- 각 단계의 학습 상태와 `predict_proba` 지원 여부를 검사한다.
- 모델의 학습 입력 항목과 메타데이터의 이름·순서가 일치하는지 검사한다.
- 출력 클래스가 `[0, 1]`인지 검사한다.

### 입력 및 예측 처리

- 마지막 관측값을 모델 입력으로 사용한다.
- 메타데이터의 입력 항목 순서에 맞춰 한 행의 DataFrame을 만든다.
- 현재 저장소의 로지스틱 설정 항목은 HR, O2Sat, Temp, SBP, DBP, MAP, Resp, Age이다.
- 실제 실행 시 입력 항목과 순서는 불러온 모델의 메타데이터를 따른다.
- 모델 입력 항목이 생략되거나 null이면 NaN으로 전달하고, 학습된 imputer가 처리한다.
- 실제 모델에는 Mock 기본값인 Resp 18.0, SBP 120.0을 적용하지 않는다.
- 마지막 관측값의 모델 입력 항목은 숫자로 변환 가능한지와 유한한 값인지 검사한다.
- Resp와 SBP에는 모든 관측값에 대해 기존 스키마 검증도 적용된다.
- 클래스 1의 확률을 risk_score로 반환한다.
- 출력 확률의 형태, 유한성, 0~1 범위 및 합이 약 1인지 검사한다.
- 현재 로지스틱 모델은 여러 시점의 추세나 24시간 시계열 전체를 사용하지 않는다.

### 상태 확인

GET /health는 다음 상태를 반환한다.

| 실행 상태 | status | model_loaded | mock_mode |
| --- | --- | --- | --- |
| Mock 준비 완료 | ok | true | true |
| 실제 모델 로딩 성공 | ok | true | false |
| 실제 모델 미설정 또는 로딩 실패 | degraded | false | false |

Mock의 model_loaded=true는 기존 호환 동작이며, 실제 모델 파일을 읽었다는 의미가 아니다.
현재 health 상태는 초기화 결과를 나타낸다.
개별 예측 요청이 실패했다고 health 상태가 자동으로 degraded로 바뀌지는 않는다.

### 오류 응답

- HTTP 422: 요청 스키마 검증 실패 또는 실제 모델에 전달할 측정값 변환·유한성 검사 실패.
- HTTP 503: 모델이 로딩되지 않았거나 모델 예측 실행·출력 검사 실패.

모델이 없을 때:

```json
{
  "detail": "Prediction unavailable: model is not loaded"
}
```

모델 예측 실행 또는 출력 검사 실패 시:

```json
{
  "detail": "Prediction unavailable: model prediction failed"
}
```

### 요청 및 오류 로그

- HTTP 메서드, 경로, 상태 코드, 응답 생성까지의 처리 시간(ms)을 기록한다.
- 400 미만 응답은 INFO, 400~499는 WARNING, 500 이상은 ERROR로 기록한다.
- 처리되지 않은 예외는 request_failed 로그와 예외 정보를 남긴다.
- 모델 로딩 및 예측 실패 시 서버에 예외 로그를 남긴다.
- 요청 요약 로그에는 환자 ID와 측정값을 넣지 않는다.
- 현재 로그는 콘솔에 출력하며, 파일 저장·보관·회전 정책은 별도로 구현하지 않았다.

### 검증 결과와 범위

- 기존 테스트 27개와 모델 연결 테스트 4개를 합쳐 전체 31개 통과.
- 임시 모델 파일을 저장하고 API에서 불러오는 경로 검증.
- 마지막 관측값 선택, 입력 순서, 숫자 문자열, 생략·null 처리 및 모델 직접 계산 결과와의 일치 검증.
- 잘못된 HR 입력의 422 반환 검증.
- 모델 부재 및 모델 실행 실패 시 503 반환 검증.
- Mock 모드에서 정상 요청 200과 잘못된 요청 422의 콘솔 로그 및 처리 시간 수동 확인.
- 기존 TestClient 관련 사용 중단 예정 경고 1개가 남아 있다.

아직 확인하거나 구현하지 않은 범위:

- 팀에서 전달할 실제 학습 모델 파일을 이용한 실행 확인.
- 실제 데이터에서의 모델 성능 및 확률 보정 검증.
- 실제 모델의 위험 등급 기준 연결.
- SHAP 결과 연결.
- 프론트엔드와의 실제 연동 확인.