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
