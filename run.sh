#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

exec "$PY" -m uvicorn ml_server.app.main:app --host "$HOST" --port "$PORT" --reload
