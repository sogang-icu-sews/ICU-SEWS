"""HTTP client used by the Streamlit dashboard.

The dashboard communicates through FastAPI and never loads model files directly.
"""

import os
from typing import Any

import httpx

API_URL = os.getenv("SEWS_API_URL", "http://localhost:8000").rstrip("/")
HEALTH_TIMEOUT = 5.0
PREDICT_TIMEOUT = 10.0


def _request_json(method: str, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Request JSON with bounded retries for transient connection/server failures."""
    timeout = HEALTH_TIMEOUT if path == "/health" else PREDICT_TIMEOUT
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            response = httpx.request(
                method,
                f"{API_URL}{path}",
                json=payload,
                timeout=timeout,
            )
            # Retry transient server errors once; surface client errors immediately.
            if response.status_code >= 500 and attempt == 0:
                continue
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"{path} 응답은 JSON 객체여야 합니다.")
            return data
        except (httpx.TimeoutException, httpx.TransportError) as error:
            last_error = error
            if attempt == 1:
                raise
        except httpx.HTTPStatusError:
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{path} 요청이 완료되지 않았습니다.")


def get_health() -> dict[str, Any]:
    """Read API readiness information."""
    return _request_json("GET", "/health")


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Request one patient prediction."""
    return _request_json("POST", "/predict", payload=payload)
