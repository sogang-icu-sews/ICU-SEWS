#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/app/src}"
export SEWS_USE_MOCK_MODEL="${SEWS_USE_MOCK_MODEL:-true}"
export SEWS_API_URL="${SEWS_API_URL:-http://127.0.0.1:8000}"

uvicorn icu_sews.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python - <<'PY'
import time
import urllib.error
import urllib.request

url = "http://127.0.0.1:8000/health"
for _ in range(30):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(1)
else:
    raise SystemExit("FastAPI did not become healthy within 30 seconds")
PY

streamlit run src/icu_sews/dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-10000}" \
  --server.headless true
