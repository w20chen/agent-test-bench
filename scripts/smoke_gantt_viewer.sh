#!/usr/bin/env bash
# Gantt viewer backend smoke — starts the FastAPI server against
# demo/gantt_viewer/configs/example.yaml and exercises the 3 API endpoints
# that back AC1 + AC2. Intentionally does NOT run a browser e2e — the
# frontend Solid build is verified separately via vitest + tsc. The pytest
# suite (`make gantt-viewer-test`) owns behavioral coverage; this script
# only guards the "server boots + serves traces" smoke from a cold shell.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${GANTT_VIEWER_PORT:-8765}"
# Project uses conda env "ML" — activate it before running, or override PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN='$PYTHON_BIN' not found; activate your conda ML env or set PYTHON_BIN" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the smoke test" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$PYTHON_BIN" -m uvicorn demo.gantt_viewer.backend.app:create_app \
  --factory --host 127.0.0.1 --port "$PORT" \
  >/tmp/gantt-viewer-smoke.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
    break
  fi
  sleep 0.2
done

if ! curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
  echo "server did not become ready — see /tmp/gantt-viewer-smoke.log" >&2
  exit 1
fi

health_json="$(curl -sf "http://127.0.0.1:${PORT}/api/health")"
if ! echo "$health_json" | grep -q '"status":"ok"'; then
  echo "unexpected /api/health response: $health_json" >&2
  exit 1
fi

traces_json="$(curl -sf "http://127.0.0.1:${PORT}/api/traces")"
if ! echo "$traces_json" | grep -q '"traces"'; then
  echo "unexpected /api/traces response: $traces_json" >&2
  exit 1
fi

# Extract the first descriptor id; require python because grep/sed on JSON is fragile.
first_id="$(
  echo "$traces_json" \
    | "$PYTHON_BIN" -c 'import json,sys; traces=json.load(sys.stdin)["traces"]; print(traces[0]["id"] if traces else "")'
)"
if [[ -z "$first_id" ]]; then
  echo "no traces discovered in example.yaml config" >&2
  exit 1
fi

payload_status="$(
  curl -s -o /tmp/gantt-viewer-payload.json -w '%{http_code}' \
    -X POST "http://127.0.0.1:${PORT}/api/payload" \
    -H 'Content-Type: application/json' \
    -d "{\"ids\":[\"$first_id\"]}"
)"

if [[ "$payload_status" != "200" ]]; then
  echo "POST /api/payload failed with $payload_status" >&2
  cat /tmp/gantt-viewer-payload.json >&2
  exit 1
fi

n_traces="$(
  "$PYTHON_BIN" -c 'import json; print(len(json.load(open("/tmp/gantt-viewer-payload.json"))["traces"]))'
)"
if [[ "$n_traces" != "1" ]]; then
  echo "expected 1 trace in payload response, got $n_traces" >&2
  exit 1
fi

echo "gantt viewer smoke passed (descriptor: $first_id)"
