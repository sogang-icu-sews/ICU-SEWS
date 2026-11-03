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

요청은 한 환자의 현재 및 과거 최대 24시간 기록이다.

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

초기 응답 계약은 다음과 같다.

```json
{
  "patient_id": "A_p000001",
  "timestamp": "2026-01-01T00:00:00Z",
  "risk_score": 0.71,
  "risk_level": "warning",
  "model_version": "ensemble-v1",
  "is_mock": false,
  "top_features": [
    {"feature": "lactate_trend", "shap_value": 0.31}
  ]
}
```

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
- 위 기본값은 Mock 계산용이며, 실제 모델의 결측값 처리 정책을 뜻하지 않는다.
- 생리적 허용 범위 검사는 아직 구현하지 않았다.
---
