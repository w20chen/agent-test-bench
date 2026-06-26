#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# SWE-rebench N:M Simulation Sweep on ARM (320 vCPUs)
#
# Replays 40 SWE-rebench traces with N ∈ {40, 80, 160, 320} agent instances,
# cyclic allocation, 1 CPU per agent (Docker --cpus=1), cloud_model replay.
#
# Prerequisites:
#   - $SOURCE_TRACES_DIR env var pointing to the 40 trace directories
#   - Docker installed and running
#   - ARM-compatible Docker images available (see tasks.json)
#   - Python 3.12+ with project deps installed
#
# Usage:
#   export SOURCE_TRACES_DIR=/path/to/traces/swe-rebench/model/timestamp
#   bash scripts/run_simulate_sweep.sh
#
# Output:
#   traces/simulate/swe-rebench/sweep_40a_1cpu/
#   traces/simulate/swe-rebench/sweep_80a_1cpu/
#   traces/simulate/swe-rebench/sweep_160a_1cpu/
#   traces/simulate/swe-rebench/sweep_320a_1cpu/
#
#   Each contains:
#     system_resources.jsonl    — system-wide resource timeline
#     agent_timeline.jsonl      — per-agent start/end wall-clock times
#     simulate_cloud_model_*.jsonl  — combined trace
#     <agent_id>--aN/attempt_1/ — per-agent trace + resources + HTML viz
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Configuration ────────────────────────────────────────────────────────────

SOURCE_TRACES_DIR="${SOURCE_TRACES_DIR:-}"
if [[ -z "${SOURCE_TRACES_DIR}" ]]; then
  echo "ERROR: SOURCE_TRACES_DIR is not set." >&2
  echo "  export SOURCE_TRACES_DIR=/path/to/your/40/traces" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_TRACES_DIR}" ]]; then
  echo "ERROR: SOURCE_TRACES_DIR does not exist: ${SOURCE_TRACES_DIR}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONTAINER_EXE="${CONTAINER_EXE:-docker}"
REPLAY_SPEED="${REPLAY_SPEED:-1}"
SWEEP_VALUES="${SWEEP_VALUES:-40 80 160 320}"
CPU_LIMIT="${CPU_LIMIT:-1}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/traces/simulate/swe-rebench}"

MONITOR_SCRIPT="${REPO_ROOT}/scripts/system_resource_monitor.py"
TIMELINE_SCRIPT="${REPO_ROOT}/scripts/extract_agent_timeline.py"
PLOT_SCRIPT="${REPO_ROOT}/scripts/plot_system_resources.py"
SIMULATE_MODULE="trace_collect.cli"

# ── Helpers ──────────────────────────────────────────────────────────────────

_now_ts() { date +%Y%m%dT%H%M%S; }

_hr() {
  echo
  echo "══════════════════════════════════════════════════════════════════════════"
  echo "  $*"
  echo "══════════════════════════════════════════════════════════════════════════"
  echo
}

MONITOR_PID=""
STOP_FILE=""

# Cleanup on interrupt: kill any lingering monitor process and remove stop-file.
_cleanup() {
  set +e
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    echo "[cleanup] Stopping monitor (pid ${MONITOR_PID})..."
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${STOP_FILE:-}" ]] && [[ -f "${STOP_FILE}" ]]; then
    rm -f "${STOP_FILE}" || true
  fi
  echo "[cleanup] Done."
}
trap _cleanup EXIT INT TERM

# ── Preflight ────────────────────────────────────────────────────────────────

_hr "Preflight checks"

TRACE_COUNT=$(find "${SOURCE_TRACES_DIR}" -name "trace.jsonl" -type f | wc -l)
echo "Source traces found:  ${TRACE_COUNT}"
if [[ "${TRACE_COUNT}" -eq 0 ]]; then
  echo "ERROR: No trace.jsonl files found under ${SOURCE_TRACES_DIR}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" &>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} not found" >&2
  exit 1
fi
echo "Python:               $(${PYTHON_BIN} --version)"

if ! command -v "${CONTAINER_EXE}" &>/dev/null; then
  echo "ERROR: ${CONTAINER_EXE} not found" >&2
  exit 1
fi
echo "Container runtime:    $(${CONTAINER_EXE} --version | head -1)"

if ! ${CONTAINER_EXE} info &>/dev/null; then
  echo "ERROR: ${CONTAINER_EXE} daemon is not running or not accessible" >&2
  exit 1
fi

HOST_CORES=$(${PYTHON_BIN} -c "import os; print(os.cpu_count() or 1)")
HOST_MEM=$(${PYTHON_BIN} -c "import psutil; m=psutil.virtual_memory(); print(f'{m.total/(1024**3):.0f} GB')")
echo "Host cores:           ${HOST_CORES}"
echo "Host memory:          ${HOST_MEM}"

# Warn if total allocation exceeds host cores at the high end
MAX_N=$(echo "${SWEEP_VALUES}" | tr ' ' '\n' | sort -n | tail -1)
TOTAL_ALLOC=$(echo "${MAX_N} * ${CPU_LIMIT}" | bc -l 2>/dev/null || echo "${MAX_N}")
echo "Max total CPU alloc:  ${TOTAL_ALLOC} (${MAX_N} agents × ${CPU_LIMIT} cpu)"
echo

# ── Main sweep loop ──────────────────────────────────────────────────────────

STARTED_AT=$(_now_ts)
SUMMARY_FILE="${BASE_OUTPUT_DIR}/sweep_summary_${STARTED_AT}.txt"
mkdir -p "${BASE_OUTPUT_DIR}"

echo "Sweep started at ${STARTED_AT}" | tee "${SUMMARY_FILE}"
echo "Source traces: ${SOURCE_TRACES_DIR} (${TRACE_COUNT} traces)" | tee -a "${SUMMARY_FILE}"
echo "Host: ${HOST_CORES} cores, ${HOST_MEM} RAM" | tee -a "${SUMMARY_FILE}"
echo | tee -a "${SUMMARY_FILE}"

for N in ${SWEEP_VALUES}; do
  _hr "Sweep: N=${N} agents × ${CPU_LIMIT} cpu"

  OUTPUT_DIR="${BASE_OUTPUT_DIR}/sweep_${N}a_${CPU_LIMIT}cpu"
  mkdir -p "${OUTPUT_DIR}"

  RUN_LOG="${OUTPUT_DIR}/simulate.log"
  STOP_FILE="${OUTPUT_DIR}/.stop_monitor"
  MONITOR_OUTPUT="${OUTPUT_DIR}/system_resources.jsonl"
  TIMELINE_OUTPUT="${OUTPUT_DIR}/agent_timeline.jsonl"

  # Remove stale stop-file from previous run (if any)
  rm -f "${STOP_FILE}"

  # ── Start system resource monitor ──────────────────────────────────────
  echo "[$(date +%H:%M:%S)] Starting system resource monitor..."
  "${PYTHON_BIN}" "${MONITOR_SCRIPT}" \
    --output "${MONITOR_OUTPUT}" \
    --interval 1.0 \
    --stop-file "${STOP_FILE}" \
    --verbose \
    >> "${OUTPUT_DIR}/monitor.log" 2>&1 &
  MONITOR_PID=$!
  echo "  Monitor pid: ${MONITOR_PID}"

  # Give the monitor a moment to write its first sample
  sleep 1
  if ! kill -0 "${MONITOR_PID}" 2>/dev/null; then
    echo "  ⚠️  System monitor failed to start (pid ${MONITOR_PID} already dead)."
    echo "  Check ${OUTPUT_DIR}/monitor.log for errors."
    echo "  Continuing without system monitoring..."
    MONITOR_PID=""
  fi

  # ── Run simulate ────────────────────────────────────────────────────────
  echo "[$(date +%H:%M:%S)] Running simulate with ${N} agents (stderr shown below, full log → ${RUN_LOG})..."
  RUN_START=$(date +%s)

  set +e  # capture exit code without aborting the sweep
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m "${SIMULATE_MODULE}" simulate \
    --source-dir "${SOURCE_TRACES_DIR}" \
    --mode cloud_model \
    --container "${CONTAINER_EXE}" \
    --num-agents "${N}" \
    --trace-assignment manifest \
    --cpu-limit "${CPU_LIMIT}" \
    --resource-monitoring on \
    --pmu-monitoring off \
    --ksys-monitoring off \
    --replay-speed "${REPLAY_SPEED}" \
    --output-dir "${OUTPUT_DIR}" \
    > "${RUN_LOG}" 2> >(tee -a "${RUN_LOG}" >&2)
  SIMULATE_EXIT=$?
  set -e

  RUN_END=$(date +%s)
  RUN_ELAPSED=$((RUN_END - RUN_START))
  RUN_ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((RUN_ELAPSED/3600)) $(((RUN_ELAPSED%3600)/60)) $((RUN_ELAPSED%60)))

  # ── Stop system resource monitor ────────────────────────────────────────
  if [[ -n "${MONITOR_PID}" ]]; then
    echo "[$(date +%H:%M:%S)] Stopping system resource monitor (pid ${MONITOR_PID})..."
    touch "${STOP_FILE}"
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi

  # ── Check result ────────────────────────────────────────────────────────
  if [[ ${SIMULATE_EXIT} -ne 0 ]]; then
    echo "  ⚠️  simulate exited with code ${SIMULATE_EXIT} (elapsed: ${RUN_ELAPSED_FMT})"
    echo "  Last 20 lines of log:"
    tail -20 "${RUN_LOG}" 2>/dev/null | sed 's/^/    /' || true
    echo "  See full log: ${RUN_LOG}"
    {
      echo "N=${N}: FAILED (exit=${SIMULATE_EXIT}, elapsed=${RUN_ELAPSED_FMT})"
      echo "  log: ${RUN_LOG}"
    } | tee -a "${SUMMARY_FILE}"
    continue
  fi

  echo "  ✓ simulate completed (elapsed: ${RUN_ELAPSED_FMT})"

  # ── Extract agent timeline ──────────────────────────────────────────────
  echo "[$(date +%H:%M:%S)] Extracting agent timeline..."
  "${PYTHON_BIN}" "${TIMELINE_SCRIPT}" \
    --input-dir "${OUTPUT_DIR}" \
    --output "${TIMELINE_OUTPUT}" \
    >> "${RUN_LOG}" 2>&1

  # ── Generate system resource visualization ────────────────────────────
  if [[ -f "${MONITOR_OUTPUT}" ]] && [[ -s "${MONITOR_OUTPUT}" ]]; then
    echo "[$(date +%H:%M:%S)] Generating system resource visualization..."
    "${PYTHON_BIN}" "${PLOT_SCRIPT}" \
      --input "${MONITOR_OUTPUT}" \
      --output "${OUTPUT_DIR}/system_viz.html" \
      >> "${RUN_LOG}" 2>&1
    echo "  → ${OUTPUT_DIR}/system_viz.html"
  fi

  # ── Print summary ───────────────────────────────────────────────────────
  echo
  echo "  --- Agent Timeline Summary (N=${N}) ---"
  "${PYTHON_BIN}" "${TIMELINE_SCRIPT}" \
    --input-dir "${OUTPUT_DIR}" \
    --output /dev/null \
    2>/dev/null || true

  MONITOR_SAMPLES=$(wc -l < "${MONITOR_OUTPUT}" 2>/dev/null || echo "0")
  echo
  echo "  System monitor samples: ${MONITOR_SAMPLES}"
  echo "  Output dir:             ${OUTPUT_DIR}"

  {
    echo "N=${N}: OK (elapsed=${RUN_ELAPSED_FMT}, monitor_samples=${MONITOR_SAMPLES})"
    echo "  output: ${OUTPUT_DIR}"
  } | tee -a "${SUMMARY_FILE}"

done

# ── Final summary ────────────────────────────────────────────────────────────

FINISHED_AT=$(_now_ts)
_hr "Sweep complete"
echo "Started:  ${STARTED_AT}"
echo "Finished: ${FINISHED_AT}"
echo
echo "Summary log: ${SUMMARY_FILE}"
echo
cat "${SUMMARY_FILE}"

# ── Throughput summary ───────────────────────────────────────────────────────

echo
echo "═══════════════════════════════════════════════"
echo "  Throughput Summary"
echo "═══════════════════════════════════════════════"
printf "  %-6s  %-12s  %-14s  %-14s\n" "N" "Wall Time" "Agents/s" "Agents/min"
echo "  ──────  ────────────  ──────────────  ──────────────"
while IFS= read -r line; do
  n_val=$(echo "${line}" | sed -n 's/^N=\([0-9]*\): OK.*/\1/p')
  elapsed_str=$(echo "${line}" | sed -n 's/.*elapsed=\([0-9:]*\).*/\1/p')
  if [[ -n "${n_val}" && -n "${elapsed_str}" ]]; then
    h=$(echo "${elapsed_str}" | cut -d: -f1 | sed 's/^0*//')
    m=$(echo "${elapsed_str}" | cut -d: -f2 | sed 's/^0*//')
    s=$(echo "${elapsed_str}" | cut -d: -f3 | sed 's/^0*//')
    h=${h:-0}; m=${m:-0}; s=${s:-0}
    total_s=$((h * 3600 + m * 60 + s))
    if [[ ${total_s} -gt 0 ]]; then
      aps=$(echo "scale=3; ${n_val} / ${total_s}" | bc -l 2>/dev/null || echo "N/A")
      apm=$(echo "scale=1; ${n_val} / ${total_s} * 60" | bc -l 2>/dev/null || echo "N/A")
    else
      aps="N/A"; apm="N/A"
    fi
    printf "  %-6s  %-12s  %-14s  %-14s\n" "${n_val}" "${elapsed_str}" "${aps}" "${apm}"
  fi
done < "${SUMMARY_FILE}"
echo
