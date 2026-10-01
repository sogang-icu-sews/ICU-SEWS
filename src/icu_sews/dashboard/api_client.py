"""HTTP client used by the dashboard.

Owner: 임경수
The dashboard must communicate through FastAPI and must never load model files directly.
"""

import os
from typing import Any

import httpx


API_URL = os.getenv("SEWS_API_URL", "http://localhost:8000")


def get_health() -> dict[str, Any]:
    """Read API readiness information."""
    response = httpx.get(f"{API_URL}/health", timeout=5.0)
    response.raise_for_status()
    return response.json()


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Request one patient prediction."""
    response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
    response.raise_for_status()
    return response.json()

