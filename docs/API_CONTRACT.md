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

- 기존 Mock 상태 확인 및 정상 예측 요청 테스트 통과.
- `Resp`에 `"hello"`를 보내면 HTTP 422와 Resp 오류 위치 반환.
- `SBP`에 `"hello"`를 보내면 HTTP 422와 SBP 오류 위치 반환.

위 테스트는 Mock 환경에서 수행했으며, 실제 모델 예측을 검증한 것은 아니다.
---