#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-.venv-server}"
SERVER_PYTHON="${REPO_ROOT}/${VENV_DIR}/bin/python"

MODEL_PATH="${MODEL_PATH:-/workspace/agent-sched-bench/models/Qwen3-32B-AWQ}"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.10.2}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_PREEMPTION_MODE="${VLLM_PREEMPTION_MODE:-recompute}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
VLLM_SCHEDULER_HOOK="${VLLM_SCHEDULER_HOOK:-0}"
VLLM_HEALTH_TIMEOUT_S="${VLLM_HEALTH_TIMEOUT_S:-180}"
VLLM_POLL_INTERVAL_S="${VLLM_POLL_INTERVAL_S:-2.0}"
VLLM_SMOKE_MODEL="${VLLM_SMOKE_MODEL:-auto}"
VLLM_SMOKE_PROMPT="${VLLM_SMOKE_PROMPT:-Reply with the word READY.}"
VLLM_LOG_PATH="${VLLM_LOG_PATH:-${REPO_ROOT}/results/processed/vllm_server.log}"
VLLM_REPORT_PATH="${VLLM_REPORT_PATH:-${REPO_ROOT}/results/processed/vllm_server_report.json}"
VLLM_PREEMPTION_REPORT_PATH="${VLLM_PREEMPTION_REPORT_PATH:-${REPO_ROOT}/results/processed/vllm_preemption_report.json}"
VLLM_SCHEDULER_HOOK_REPORT_PATH="${VLLM_SCHEDULER_HOOK_REPORT_PATH:-${REPO_ROOT}/results/processed/vllm_scheduler_hook_report.json}"

log() {
  printf '[ENV-3a] %s\n' "$*"
}

require_server_python() {
  if [[ ! -x "${SERVER_PYTHON}" ]]; then
    printf 'Missing repo-local server Python: %s\n' "${SERVER_PYTHON}" >&2
    printf 'Run ENV-1 first on the target server.\n' >&2
    exit 1
  fi
}

require_model_path() {
  # Accept both local directories and HuggingFace repo IDs (e.g. meta-llama/Llama-3.1-8B-Instruct)
  if [[ -d "${MODEL_PATH}" ]]; then
    return 0
  fi
  if [[ "${MODEL_PATH}" == */* && "${MODEL_PATH}" != /* ]]; then
    log "MODEL_PATH looks like an HF repo ID: ${MODEL_PATH}"
    return 0
  fi
  printf 'Model path does not exist: %s\n' "${MODEL_PATH}" >&2
  printf 'Run ENV-2 successfully before ENV-3a.\n' >&2
  exit 1
}

install_vllm() {
  log "Installing ${VLLM_SPEC} into ${SERVER_PYTHON}"
  uv pip install --python "${SERVER_PYTHON}" "${VLLM_SPEC}"
}

start_server() {
  mkdir -p "$(dirname "${VLLM_LOG_PATH}")"
  : > "${VLLM_LOG_PATH}"
  log "Starting raw vLLM server; logs -> ${VLLM_LOG_PATH}"
  local hook_args=()
  if [[ "${VLLM_SCHEDULER_HOOK}" == "1" ]]; then
    hook_args=(--enable-scheduler-hook --scheduler-hook-report-path "${VLLM_SCHEDULER_HOOK_REPORT_PATH}")
  fi
  local tool_args=()
  if [[ "${VLLM_ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
    tool_args=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
  fi
  PYTHONPATH="${REPO_ROOT}/src" "${SERVER_PYTHON}" -m serving.engine_launcher \
    --model-path "${MODEL_PATH}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --dtype "${VLLM_DTYPE}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --preemption-mode "${VLLM_PREEMPTION_MODE}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    "${hook_args[@]}" \
    "${tool_args[@]}" \
    $( [[ "${VLLM_ENABLE_CHUNKED_PREFILL}" == "1" ]] && printf '%s ' '--enable-chunked-prefill' ) \
    >>"${VLLM_LOG_PATH}" 2>&1 &
  SERVER_PID=$!
  log "vLLM server pid=${SERVER_PID}"
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    log "Stopping vLLM server pid=${SERVER_PID}"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

verify_server() {
  log "Running raw vLLM readiness checks"
  PYTHONPATH="${REPO_ROOT}/src" "${SERVER_PYTHON}" -m serving.health_check \
    --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
    --metrics-url "http://127.0.0.1:${VLLM_PORT}/metrics" \
    --model "${VLLM_SMOKE_MODEL}" \
    --timeout-s "${VLLM_HEALTH_TIMEOUT_S}" \
    --poll-interval-s "${VLLM_POLL_INTERVAL_S}" \
    --prompt "${VLLM_SMOKE_PROMPT}" \
    --output "${VLLM_REPORT_PATH}" \
    --vllm-spec "${VLLM_SPEC}" \
    --model-path "${MODEL_PATH}" \
    --fail-on-mismatch
}

write_preemption_report() {
  log "Writing vLLM preemption report to ${VLLM_PREEMPTION_REPORT_PATH}"
  PYTHONPATH="${REPO_ROOT}/src" "${SERVER_PYTHON}" -m harness.scheduler_hooks \
    --metrics-url "http://127.0.0.1:${VLLM_PORT}/metrics" \
    --log-file "${VLLM_LOG_PATH}" \
    --output "${VLLM_PREEMPTION_REPORT_PATH}"
}

main() {
  require_server_python
  require_model_path
  install_vllm
  trap cleanup EXIT INT TERM
  start_server
  verify_server
  write_preemption_report
  log "ENV-3a completed successfully; server is running. Press Ctrl-C to stop."
  wait "${SERVER_PID}"
}

main "$@"
