#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Rebench-vs-Rebench Scheduling Sweep
#
# Compares two scheduling strategies for two SWE-rebench container workloads.
# Use one source directory for a CPU/memory-heavy case and one source directory
# for an LLM/context-heavy case with roughly comparable duration:
#
#   Strategy A ("sequential"): Process N/2 workload-A traces first (all cores),
#                              then N/2 workload-B traces (all cores).
#   Strategy B ("interleaved"): Process N/2 workload-A + N/2 workload-B
#                               traces simultaneously, splitting cores evenly.
#
# Both strategies process exactly N total traces per round (N/2 of each type)
# and are constrained to the same total CPU cores, making wall-clock time the
# primary comparison metric.
#
# Prerequisites:
#   - SOURCE_TRACES_DIR_A: directory of pre-collected SWE-rebench traces
#   - SOURCE_TRACES_DIR_B: directory of pre-collected SWE-rebench traces
#   - Docker installed and running
#   - Python 3.12+ with project deps installed
#
# Usage:
#   export SOURCE_TRACES_DIR_A=/path/to/swe-rebench/cpu-memory-heavy/attempt_1
#   export SOURCE_TRACES_DIR_B=/path/to/swe-rebench/llm-heavy/attempt_1
#   bash scripts/run_mixed_scheduling_sweep.sh
#
# Recommended inspected cases from C:\Users\29068\Desktop\agent_datasets:
#   SOURCE_TRACES_DIR_A=.../swe-rebench/AI4S2S__lilio-49/attempt_1
#   SOURCE_TRACES_DIR_B=.../swe-rebench/Azure__azure-cli-2955/attempt_1
#
# Output:
#   traces/simulate/mixed_sweep/
#     sweep_summary_<timestamp>.txt
#     sequential_N<agents>/
#       workload_a/    workload-A simulate output
#       workload_b/    workload-B simulate output
#     interleaved_N<agents>/
#       workload_a/    workload-A simulate output (N/2 agents)
#       workload_b/    workload-B simulate output (N/2 agents)
#
#   Each subdirectory contains:
#     system_resources.jsonl    system-wide resource timeline
#     agent_timeline.jsonl      per-agent start/end wall-clock times
#     simulate.log              full simulate stdout+stderr
#     monitor.log               resource monitor stderr
#     system_viz.html           system-wide resource visualization
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -- Configuration -------------------------------------------------------------

SOURCE_TRACES_DIR_A="${SOURCE_TRACES_DIR_A:-${SOURCE_TRACES_DIR_REBENCH:-}}"
SOURCE_TRACES_DIR_B="${SOURCE_TRACES_DIR_B:-}"
WORKLOAD_A_LABEL="${WORKLOAD_A_LABEL:-cpu-memory-heavy}"
WORKLOAD_B_LABEL="${WORKLOAD_B_LABEL:-llm-heavy}"

if [[ -z "${SOURCE_TRACES_DIR_A}" ]]; then
  echo "ERROR: SOURCE_TRACES_DIR_A is not set." >&2
  echo "  export SOURCE_TRACES_DIR_A=/path/to/swe-rebench/cpu-memory-heavy/attempt_1" >&2
  exit 1
fi
if [[ -z "${SOURCE_TRACES_DIR_B}" ]]; then
  echo "ERROR: SOURCE_TRACES_DIR_B is not set." >&2
  echo "  export SOURCE_TRACES_DIR_B=/path/to/swe-rebench/llm-heavy/attempt_1" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_TRACES_DIR_A}" ]]; then
  echo "ERROR: SOURCE_TRACES_DIR_A does not exist: ${SOURCE_TRACES_DIR_A}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_TRACES_DIR_B}" ]]; then
  echo "ERROR: SOURCE_TRACES_DIR_B does not exist: ${SOURCE_TRACES_DIR_B}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONTAINER_EXE="${CONTAINER_EXE:-docker}"
REPLAY_SPEED="${REPLAY_SPEED:-1}"
STRICT_SYSTEM_MONITOR="${STRICT_SYSTEM_MONITOR:-1}"

# SWEEP_VALUES: total agent concurrency levels to sweep.
# Each round processes N total traces: N/2 workload A + N/2 workload B.
# Strategy A: sequential (A then B, full cores each phase).
# Strategy B: interleaved (A + B in parallel, split cores).
SWEEP_VALUES="${SWEEP_VALUES:-40 80 160 320}"

# TOTAL_CORES: system-wide CPU core budget.  The entire script is pinned to
# cores 0..(TOTAL_CORES-1) via taskset, and the two parallel simulate
# processes in the interleaved strategy are further partitioned.
TOTAL_CORES="${TOTAL_CORES:-160}"

# CPU_LIMIT: per-container Docker --cpus quota.
CPU_LIMIT="${CPU_LIMIT:-1}"

# TASK_SOURCE_A / TASK_SOURCE_B: optional tasks JSON overrides.
# Both workloads default to the canonical SWE-rebench local task cache.
TASK_SOURCE_A="${TASK_SOURCE_A:-${TASK_SOURCE_REBENCH:-${REPO_ROOT}/data/swe-rebench/tasks.json}}"
TASK_SOURCE_B="${TASK_SOURCE_B:-${TASK_SOURCE_A}}"

# WORKERS: multiprocessing workers for cloud_model replay.
# Auto-detected from host core count; capped at TOTAL_CORES.
WORKERS="${WORKERS:-$(${PYTHON_BIN} -c "import os; print(min(os.cpu_count() or 1, ${TOTAL_CORES}))")}"

# PREP_CONCURRENCY: system-wide maximum concurrent container preparations.
# 0 = auto (20, preserving the historical limit).
PREP_CONCURRENCY="${PREP_CONCURRENCY:-0}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/traces/simulate/mixed_sweep}"

MONITOR_SCRIPT="${REPO_ROOT}/scripts/system_resource_monitor.py"
TIMELINE_SCRIPT="${REPO_ROOT}/scripts/extract_agent_timeline.py"
PLOT_SCRIPT="${REPO_ROOT}/scripts/plot_system_resources.py"
SIMULATE_MODULE="trace_collect.cli"

# -- Helpers -------------------------------------------------------------------

_now_ts() { date +%Y%m%dT%H%M%S; }

_hr() {
  echo
  echo "========================================================================"
  echo "  $*"
  echo "========================================================================"
  echo
}

_validate_task_source() {
  local task_source="$1"
  local label="$2"
  if [[ -z "${task_source}" ]]; then
    echo "ERROR: TASK_SOURCE_${label} is empty; SWE-rebench replays require a task source." >&2
    exit 1
  fi
  if [[ ! -f "${task_source}" ]]; then
    echo "ERROR: TASK_SOURCE_${label} does not exist: ${task_source}" >&2
    exit 1
  fi
}

_validate_swe_rebench_source() {
  local src_dir="$1"
  local label="$2"
  local first_trace
  first_trace=$(find "${src_dir}" -name "trace.jsonl" -type f | sort | head -1)
  if [[ -z "${first_trace}" ]]; then
    echo "ERROR: Workload ${label} has no trace.jsonl under ${src_dir}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - "${first_trace}" "${label}" <<'PY'
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
label = sys.argv[2]
with trace_path.open("r", encoding="utf-8") as handle:
    metadata = json.loads(handle.readline())

if metadata.get("type") != "trace_metadata":
    raise SystemExit(f"ERROR: Workload {label} first record is not trace_metadata: {trace_path}")
if metadata.get("benchmark") != "swe-rebench":
    raise SystemExit(
        f"ERROR: Workload {label} is not a SWE-rebench trace "
        f"(benchmark={metadata.get('benchmark')!r}): {trace_path}"
    )
runtime_mode = metadata.get("agent_runtime_mode")
if runtime_mode == "host_controller":
    raise SystemExit(f"ERROR: Workload {label} is host-mode, expected container-mode: {trace_path}")
if not metadata.get("instance_id"):
    raise SystemExit(f"ERROR: Workload {label} trace metadata is missing instance_id: {trace_path}")
PY
}

# Track background simulate PIDs for cleanup on interrupt.
_BG_PIDS=()

_cleanup() {
  set +e
  for pid in "${_BG_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[cleanup] Killing background simulate process (pid ${pid})..."
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  # Wait briefly for graceful shutdown.
  for pid in "${_BG_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  echo "[cleanup] Done."
}
trap _cleanup EXIT INT TERM

# Run a single simulate invocation with resource monitoring.
# Arguments:
#   $1: source traces directory
#   $2: output directory
#   $3: number of agents (--num-agents)
#   $4: label for logging
#   $5: (optional) taskset CPU range override
#   $6: optional path to tasks JSON (--task-source); empty = auto
#   $7: resource monitoring mode (on|off), default on.
_run_simulate() {
  local src_dir="$1"
  local output_dir="$2"
  local num_agents="$3"
  local label="$4"
  local cpu_range="${5:-}"
  local task_source="${6:-}"
  local resource_mon="${7:-on}"

  mkdir -p "${output_dir}"

  local run_log="${output_dir}/simulate.log"
  local stop_file="${output_dir}/.stop_monitor"
  local monitor_output="${output_dir}/system_resources.jsonl"
  local timeline_output="${output_dir}/agent_timeline.jsonl"
  local monitor_pid=""

  rm -f "${stop_file}"

  # Start system resource monitor.
  echo "[$(date +%H:%M:%S)] [${label}] Starting system resource monitor..."
  "${PYTHON_BIN}" "${MONITOR_SCRIPT}" \
    --output "${monitor_output}" \
    --interval 1.0 \
    --stop-file "${stop_file}" \
    --verbose \
    >> "${output_dir}/monitor.log" 2>&1 &
  monitor_pid=$!
  echo "  Monitor pid: ${monitor_pid}"

  sleep 1
  if ! kill -0 "${monitor_pid}" 2>/dev/null; then
    echo "  WARNING: System monitor failed to start (pid ${monitor_pid} already dead)."
    echo "  Check ${output_dir}/monitor.log for errors."
    monitor_pid=""
    if [[ "${STRICT_SYSTEM_MONITOR}" == "1" ]]; then
      echo "  ERROR: STRICT_SYSTEM_MONITOR=1, refusing to continue without system monitoring."
      return 1
    fi
    echo "  Continuing without system monitoring because STRICT_SYSTEM_MONITOR=${STRICT_SYSTEM_MONITOR}."
  fi

  # Build simulate command.
  local sim_cmd=(
    "${PYTHON_BIN}" -m "${SIMULATE_MODULE}" simulate
    --source-dir "${src_dir}"
    --mode cloud_model
    --container "${CONTAINER_EXE}"
    --num-agents "${num_agents}"
    --trace-assignment manifest
    --cpu-limit "${CPU_LIMIT}"
    --workers "${WORKERS}"
    --prep-concurrency "${PREP_CONCURRENCY}"
    --resource-monitoring "${resource_mon}"
    --pmu-monitoring off
    --ksys-monitoring off
    --replay-speed "${REPLAY_SPEED}"
    --output-dir "${output_dir}"
  )
  if [[ -n "${task_source}" ]]; then
    sim_cmd+=(--task-source "${task_source}")
  fi

  # Run simulate.
  echo "[$(date +%H:%M:%S)] [${label}] Running simulate with ${num_agents} agents..."
  local run_start
  run_start=$(date +%s)

  set +e
  if [[ -n "${cpu_range}" ]]; then
    PYTHONPATH="${REPO_ROOT}/src" PYTHONUNBUFFERED=1 \
      taskset -c "${cpu_range}" \
      "${sim_cmd[@]}" \
      2>&1 | tee "${run_log}"
  else
    PYTHONPATH="${REPO_ROOT}/src" PYTHONUNBUFFERED=1 \
      "${sim_cmd[@]}" \
      2>&1 | tee "${run_log}"
  fi
  local simulate_exit=${PIPESTATUS[0]}
  set -e

  local run_end
  run_end=$(date +%s)
  local run_elapsed=$((run_end - run_start))
  local run_elapsed_fmt
  run_elapsed_fmt=$(printf '%02d:%02d:%02d' $((run_elapsed/3600)) $(((run_elapsed%3600)/60)) $((run_elapsed%60)))

  # Stop monitor.
  if [[ -n "${monitor_pid}" ]]; then
    echo "[$(date +%H:%M:%S)] [${label}] Stopping monitor (pid ${monitor_pid})..."
    touch "${stop_file}"
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
  fi

  # Post-process outputs.
  if [[ ${simulate_exit} -ne 0 ]]; then
    echo "  WARNING: [${label}] simulate exited with code ${simulate_exit} (elapsed: ${run_elapsed_fmt})"
    echo "  Last 20 lines of log:"
    tail -20 "${run_log}" 2>/dev/null | sed 's/^/    /' || true
    return 1
  fi

  echo "  OK: [${label}] simulate completed (elapsed: ${run_elapsed_fmt})"

  # Extract agent timeline
  echo "[$(date +%H:%M:%S)] [${label}] Extracting agent timeline..."
  "${PYTHON_BIN}" "${TIMELINE_SCRIPT}" \
    --input-dir "${output_dir}" \
    --output "${timeline_output}" \
    >> "${run_log}" 2>&1 || true

  # Generate system resource visualization
  if [[ -f "${monitor_output}" ]] && [[ -s "${monitor_output}" ]]; then
    echo "[$(date +%H:%M:%S)] [${label}] Generating system visualization..."
    local plot_extra=()
    if [[ -f "${timeline_output}" ]]; then
      plot_extra=(--timeline "${timeline_output}")
    fi
    "${PYTHON_BIN}" "${PLOT_SCRIPT}" \
      --input "${monitor_output}" \
      --output "${output_dir}/system_viz.html" \
      "${plot_extra[@]}" \
      >> "${run_log}" 2>&1 || true
    echo "  -> ${output_dir}/system_viz.html"
  fi

  echo "${run_elapsed}" > "${output_dir}/.elapsed_s"
  return 0
}

# -- Preflight -----------------------------------------------------------------

_hr "Preflight checks"

WORKLOAD_A_TRACE_COUNT=$(find "${SOURCE_TRACES_DIR_A}" -name "trace.jsonl" -type f | wc -l)
WORKLOAD_B_TRACE_COUNT=$(find "${SOURCE_TRACES_DIR_B}" -name "trace.jsonl" -type f | wc -l)
echo "Workload A (${WORKLOAD_A_LABEL}) traces found: ${WORKLOAD_A_TRACE_COUNT}"
echo "Workload B (${WORKLOAD_B_LABEL}) traces found: ${WORKLOAD_B_TRACE_COUNT}"

if [[ "${WORKLOAD_A_TRACE_COUNT}" -eq 0 ]]; then
  echo "ERROR: No trace.jsonl files found under ${SOURCE_TRACES_DIR_A}" >&2
  exit 1
fi
if [[ "${WORKLOAD_B_TRACE_COUNT}" -eq 0 ]]; then
  echo "ERROR: No trace.jsonl files found under ${SOURCE_TRACES_DIR_B}" >&2
  exit 1
fi

_validate_task_source "${TASK_SOURCE_A}" "A"
_validate_task_source "${TASK_SOURCE_B}" "B"
_validate_swe_rebench_source "${SOURCE_TRACES_DIR_A}" "A"
_validate_swe_rebench_source "${SOURCE_TRACES_DIR_B}" "B"
echo "Workload trace metadata: SWE-rebench container-mode validation passed."

for N in ${SWEEP_VALUES}; do
  if [[ ! "${N}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SWEEP_VALUES must contain positive even integers (got ${N})." >&2
    exit 1
  fi
  if [[ "${N}" -le 0 ]]; then
    echo "ERROR: SWEEP_VALUES must contain positive even integers (got ${N})." >&2
    exit 1
  fi
  if [[ $((N % 2)) -ne 0 ]]; then
    echo "ERROR: SWEEP_VALUES must contain even integers so each round runs exactly N agents (got ${N})." >&2
    exit 1
  fi
done

# Warn if trace counts are less than what we need for the max N.
MAX_N=$(echo "${SWEEP_VALUES}" | tr ' ' '\n' | sort -n | tail -1)
HALF_MAX_N=$((MAX_N / 2))
if [[ "${WORKLOAD_A_TRACE_COUNT}" -lt "${HALF_MAX_N}" ]]; then
  echo "  WARNING: Workload A trace count (${WORKLOAD_A_TRACE_COUNT}) < HALF_MAX_N (${HALF_MAX_N})."
  echo "     Manifest assignment will cycle traces for N=${MAX_N}."
fi
if [[ "${WORKLOAD_B_TRACE_COUNT}" -lt "${HALF_MAX_N}" ]]; then
  echo "  WARNING: Workload B trace count (${WORKLOAD_B_TRACE_COUNT}) < HALF_MAX_N (${HALF_MAX_N})."
  echo "     Manifest assignment will cycle traces for N=${MAX_N}."
fi

if ! command -v "${PYTHON_BIN}" &>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} not found" >&2
  exit 1
fi
echo "Python:                 $(${PYTHON_BIN} --version)"

if ! command -v "${CONTAINER_EXE}" &>/dev/null; then
  echo "ERROR: ${CONTAINER_EXE} not found" >&2
  exit 1
fi
echo "Container runtime:      $(${CONTAINER_EXE} --version | head -1)"

if ! command -v taskset &>/dev/null; then
  echo "ERROR: taskset not found (required for CPU core pinning)" >&2
  exit 1
fi
echo "taskset:                $(command -v taskset)"

if ! ${CONTAINER_EXE} info &>/dev/null; then
  echo "ERROR: ${CONTAINER_EXE} daemon is not running or not accessible" >&2
  exit 1
fi

HOST_CORES=$(${PYTHON_BIN} -c "import os; print(os.cpu_count() or 1)")
HOST_MEM=$(${PYTHON_BIN} -c "import psutil; m=psutil.virtual_memory(); print(f'{m.total/(1024**3):.0f} GB')")
echo "Host cores:             ${HOST_CORES}"
echo "Host memory:            ${HOST_MEM}"
echo "CPU budget (total):     ${TOTAL_CORES} cores"
echo "CPU limit (per container): ${CPU_LIMIT}"
echo "Workers:                ${WORKERS}"

# Warn if TOTAL_CORES exceeds host cores.
if [[ "${TOTAL_CORES}" -gt "${HOST_CORES}" ]]; then
  echo "  WARNING: TOTAL_CORES (${TOTAL_CORES}) > host cores (${HOST_CORES})."
  echo "     CPU affinity will be capped to available cores."
  EFFECTIVE_TOTAL=${HOST_CORES}
else
  EFFECTIVE_TOTAL=${TOTAL_CORES}
fi

# Validate we have enough cores to split for interleaved mode at the lowest N.
MIN_N=$(echo "${SWEEP_VALUES}" | tr ' ' '\n' | sort -n | head -1)
if [[ $((MIN_N / 2)) -lt 1 ]]; then
  echo "ERROR: Smallest N (${MIN_N}) / 2 < 1 agent for interleaved mode." >&2
  echo "  Add a larger N to SWEEP_VALUES or reduce to a single comparison." >&2
  exit 1
fi

# File descriptor limit check.
CURRENT_ULIMIT=$(ulimit -n 2>/dev/null || echo "unknown")
MIN_ULIMIT=$((MAX_N * 5))
echo "fd limit (ulimit -n):   ${CURRENT_ULIMIT}  (min recommended: ${MIN_ULIMIT})"
if [[ "${CURRENT_ULIMIT}" != "unknown" ]] && [[ "${CURRENT_ULIMIT}" -lt "${MIN_ULIMIT}" ]]; then
  NEW_ULIMIT=$((MIN_ULIMIT > 65536 ? MIN_ULIMIT : 65536))
  echo "  WARNING: fd limit too low for ${MAX_N} agents; raising to ${NEW_ULIMIT}"
  ulimit -n "${NEW_ULIMIT}" 2>/dev/null || {
    echo "  ERROR: Failed to raise ulimit (check /etc/security/limits.conf)"
    exit 1
  }
  echo "  OK: fd limit raised to $(ulimit -n)"
fi

# Verify no existing Docker containers will conflict.
EXISTING_CONTAINERS=$(${CONTAINER_EXE} ps -q 2>/dev/null | wc -l)
if [[ "${EXISTING_CONTAINERS}" -gt 0 ]]; then
  echo "  WARNING: ${EXISTING_CONTAINERS} existing container(s) running."
  echo "     Consider stopping them to avoid resource contention."
fi
echo

# -- Apply global CPU affinity -------------------------------------------------

# Pin the entire script process tree to EFFECTIVE_TOTAL cores.
# Individual simulate invocations in interleaved mode further partition this.
CORE_RANGE_FULL="0-$((EFFECTIVE_TOTAL - 1))"
HALF_CORES=$((EFFECTIVE_TOTAL / 2))
CORE_RANGE_LEFT="0-$((HALF_CORES - 1))"
CORE_RANGE_RIGHT="${HALF_CORES}-$((EFFECTIVE_TOTAL - 1))"

echo "Core partitioning:"
echo "  Full range:   ${CORE_RANGE_FULL}"
echo "  Left half:    ${CORE_RANGE_LEFT}"
echo "  Right half:   ${CORE_RANGE_RIGHT}"
echo

# Pin this script to the full core range.
taskset -pc "${CORE_RANGE_FULL}" $$

# -- Main sweep loop -----------------------------------------------------------

STARTED_AT=$(_now_ts)
SUMMARY_FILE="${BASE_OUTPUT_DIR}/sweep_summary_${STARTED_AT}.txt"
mkdir -p "${BASE_OUTPUT_DIR}"

{
  echo "Rebench-vs-Rebench Scheduling Sweep - started at ${STARTED_AT}"
  echo ""
  echo "Configuration:"
  echo "  Workload A label:     ${WORKLOAD_A_LABEL}"
  echo "  Workload A traces:    ${SOURCE_TRACES_DIR_A} (${WORKLOAD_A_TRACE_COUNT} traces)"
  echo "  Workload B label:     ${WORKLOAD_B_LABEL}"
  echo "  Workload B traces:    ${SOURCE_TRACES_DIR_B} (${WORKLOAD_B_TRACE_COUNT} traces)"
  echo "  Host:                 ${HOST_CORES} cores, ${HOST_MEM} RAM"
  echo "  CPU budget:           ${TOTAL_CORES} cores (effective: ${EFFECTIVE_TOTAL})"
  echo "  CPU limit/container:  ${CPU_LIMIT}"
  echo "  Workers:              ${WORKERS}"
  echo "  Task source (A):      ${TASK_SOURCE_A}"
  echo "  Task source (B):      ${TASK_SOURCE_B}"
  echo "  Sweep values (N):     ${SWEEP_VALUES}"
  echo ""
  echo "Strategies:"
  echo "  A) sequential:  N/2 workload-A agents then N/2 workload-B agents (full cores)"
  echo "  B) interleaved: N/2 workload-A + N/2 workload-B agents simultaneously (split cores)"
  echo ""
} | tee "${SUMMARY_FILE}"

for N in ${SWEEP_VALUES}; do
  HALF_N=$((N / 2))

  # Strategy A: Sequential - all workload A first, then all workload B.
  _hr "Strategy A (sequential): N=${N} (${HALF_N} A -> ${HALF_N} B)"

  SEQ_OUTPUT_DIR="${BASE_OUTPUT_DIR}/sequential_N${N}"
  SEQ_A_DIR="${SEQ_OUTPUT_DIR}/workload_a"
  SEQ_B_DIR="${SEQ_OUTPUT_DIR}/workload_b"
  mkdir -p "${SEQ_A_DIR}" "${SEQ_B_DIR}"

  SEQ_START=$(date +%s)

  # Phase 1: N/2 workload A (all cores)
  echo "[$(date +%H:%M:%S)] Phase 1/2: Running ${HALF_N} workload-A agents (${WORKLOAD_A_LABEL}, all cores)..."
  A_OK=0
  if _run_simulate \
    "${SOURCE_TRACES_DIR_A}" \
    "${SEQ_A_DIR}" \
    "${HALF_N}" \
    "seq-a" \
    "${CORE_RANGE_FULL}" \
    "${TASK_SOURCE_A}" \
    "on"; then
    A_OK=1
  fi

  # Phase 2: N/2 workload B (all cores)
  echo "[$(date +%H:%M:%S)] Phase 2/2: Running ${HALF_N} workload-B agents (${WORKLOAD_B_LABEL}, all cores)..."
  B_OK=0
  if _run_simulate \
    "${SOURCE_TRACES_DIR_B}" \
    "${SEQ_B_DIR}" \
    "${HALF_N}" \
    "seq-b" \
    "${CORE_RANGE_FULL}" \
    "${TASK_SOURCE_B}" \
    "on"; then
    B_OK=1
  fi

  SEQ_END=$(date +%s)
  SEQ_ELAPSED=$((SEQ_END - SEQ_START))
  SEQ_ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((SEQ_ELAPSED/3600)) $(((SEQ_ELAPSED%3600)/60)) $((SEQ_ELAPSED%60)))

  A_ELAPSED=$(cat "${SEQ_A_DIR}/.elapsed_s" 2>/dev/null || echo "N/A")
  B_ELAPSED=$(cat "${SEQ_B_DIR}/.elapsed_s" 2>/dev/null || echo "N/A")

  echo "  Strategy A (sequential) N=${N}: done (elapsed: ${SEQ_ELAPSED_FMT})"
  echo "    Workload A phase: ${A_ELAPSED}s, workload B phase: ${B_ELAPSED}s"

  {
    echo ""
    echo "N=${N}  strategy=sequential  elapsed=${SEQ_ELAPSED_FMT}  elapsed_s=${SEQ_ELAPSED}"
    echo "  workload_a_s=${A_ELAPSED}  workload_b_s=${B_ELAPSED}"
    echo "  workload_a_ok=${A_OK}  workload_b_ok=${B_OK}"
    echo "  output: ${SEQ_OUTPUT_DIR}"
  } | tee -a "${SUMMARY_FILE}"

  # Strategy B: Interleaved - workload A and workload B simultaneously.
  _hr "Strategy B (interleaved): N=${N} (${HALF_N} A + ${HALF_N} B parallel)"

  INTERLEAVED_OUTPUT_DIR="${BASE_OUTPUT_DIR}/interleaved_N${N}"
  INTERLEAVED_A_DIR="${INTERLEAVED_OUTPUT_DIR}/workload_a"
  INTERLEAVED_B_DIR="${INTERLEAVED_OUTPUT_DIR}/workload_b"
  mkdir -p "${INTERLEAVED_A_DIR}" "${INTERLEAVED_B_DIR}"

  INTERLEAVED_START=$(date +%s)

  # Launch both simultaneously in background.
  # Workload A gets left half of cores, workload B gets right half.

  A_BG_OK=0
  B_BG_OK=0

  echo "[$(date +%H:%M:%S)] Launching workload A (${HALF_N} agents, cores ${CORE_RANGE_LEFT})..."
  _run_simulate \
    "${SOURCE_TRACES_DIR_A}" \
    "${INTERLEAVED_A_DIR}" \
    "${HALF_N}" \
    "int-a" \
    "${CORE_RANGE_LEFT}" \
    "${TASK_SOURCE_A}" \
    "on" &
  A_PID=$!
  _BG_PIDS+=("${A_PID}")

  echo "[$(date +%H:%M:%S)] Launching workload B (${HALF_N} agents, cores ${CORE_RANGE_RIGHT})..."
  _run_simulate \
    "${SOURCE_TRACES_DIR_B}" \
    "${INTERLEAVED_B_DIR}" \
    "${HALF_N}" \
    "int-b" \
    "${CORE_RANGE_RIGHT}" \
    "${TASK_SOURCE_B}" \
    "on" &
  B_PID=$!
  _BG_PIDS+=("${B_PID}")

  echo "  Workload A pid: ${A_PID}, workload B pid: ${B_PID}"

  # Wait for both to complete.
  if wait "${A_PID}"; then
    A_BG_OK=1
    echo "[$(date +%H:%M:%S)] Workload A (${HALF_N} agents) completed."
  else
    echo "[$(date +%H:%M:%S)] Workload A (${HALF_N} agents) FAILED."
  fi
  # Remove from tracked PIDs now that it's done.
  for i in "${!_BG_PIDS[@]}"; do
    if [[ "${_BG_PIDS[$i]}" == "${A_PID}" ]]; then
      unset '_BG_PIDS[$i]'
      break
    fi
  done

  if wait "${B_PID}"; then
    B_BG_OK=1
    echo "[$(date +%H:%M:%S)] Workload B (${HALF_N} agents) completed."
  else
    echo "[$(date +%H:%M:%S)] Workload B (${HALF_N} agents) FAILED."
  fi
  for i in "${!_BG_PIDS[@]}"; do
    if [[ "${_BG_PIDS[$i]}" == "${B_PID}" ]]; then
      unset '_BG_PIDS[$i]'
      break
    fi
  done

  INTERLEAVED_END=$(date +%s)
  INTERLEAVED_ELAPSED=$((INTERLEAVED_END - INTERLEAVED_START))
  INTERLEAVED_ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((INTERLEAVED_ELAPSED/3600)) $(((INTERLEAVED_ELAPSED%3600)/60)) $((INTERLEAVED_ELAPSED%60)))

  A_BG_ELAPSED=$(cat "${INTERLEAVED_A_DIR}/.elapsed_s" 2>/dev/null || echo "N/A")
  B_BG_ELAPSED=$(cat "${INTERLEAVED_B_DIR}/.elapsed_s" 2>/dev/null || echo "N/A")

  echo "  Strategy B (interleaved) N=${N}: done (elapsed: ${INTERLEAVED_ELAPSED_FMT})"
  echo "    Workload A: ${A_BG_ELAPSED}s, workload B: ${B_BG_ELAPSED}s"

  {
    echo ""
    echo "N=${N}  strategy=interleaved  elapsed=${INTERLEAVED_ELAPSED_FMT}  elapsed_s=${INTERLEAVED_ELAPSED}"
    echo "  workload_a_s=${A_BG_ELAPSED}  workload_b_s=${B_BG_ELAPSED}"
    echo "  workload_a_ok=${A_BG_OK}  workload_b_ok=${B_BG_OK}"
    echo "  output: ${INTERLEAVED_OUTPUT_DIR}"
  } | tee -a "${SUMMARY_FILE}"

done

# -- Final summary -------------------------------------------------------------

FINISHED_AT=$(_now_ts)
_hr "Sweep complete"
echo "Started:  ${STARTED_AT}"
echo "Finished: ${FINISHED_AT}"
echo
echo "Summary log: ${SUMMARY_FILE}"
echo
cat "${SUMMARY_FILE}"

# -- Comparison table ----------------------------------------------------------

echo
echo "========================================================================"
echo "  Scheduling Strategy Comparison"
echo "========================================================================"
printf "  %-6s  %-14s  %-14s  %-14s  %-10s\n" \
  "N" "Sequential" "Interleaved" "Diff (s)" "Winner"
echo "  ------  --------------  --------------  --------------  ----------"

while IFS= read -r line; do
  n_val=$(echo "${line}" | sed -n 's/^N=\([0-9]*\)  strategy=sequential.*/\1/p')
  seq_elapsed=$(echo "${line}" | sed -n 's/^N=.*strategy=sequential.*elapsed_s=\([0-9]*\).*/\1/p')
  if [[ -z "${n_val}" ]] || [[ -z "${seq_elapsed}" ]]; then
    continue
  fi
  # Find the matching interleaved line.
  int_line=$(grep "^N=${n_val}  strategy=interleaved" "${SUMMARY_FILE}" 2>/dev/null || true)
  int_elapsed=$(echo "${int_line}" | sed -n 's/.*elapsed_s=\([0-9]*\).*/\1/p')
  if [[ -z "${int_elapsed}" ]]; then
    printf "  %-6s  %-14s  %-14s  %-14s  %-10s\n" \
      "${n_val}" "${seq_elapsed}s" "N/A" "N/A" "N/A"
    continue
  fi
  diff_s=$((seq_elapsed - int_elapsed))
  if [[ ${diff_s} -gt 0 ]]; then
    winner="interleaved"
    diff_str="+${diff_s}"
  elif [[ ${diff_s} -lt 0 ]]; then
    winner="sequential"
    diff_str="${diff_s}"
  else
    winner="tie"
    diff_str="0"
  fi
  seq_fmt=$(printf '%02d:%02d:%02d' $((seq_elapsed/3600)) $(((seq_elapsed%3600)/60)) $((seq_elapsed%60)))
  int_fmt=$(printf '%02d:%02d:%02d' $((int_elapsed/3600)) $(((int_elapsed%3600)/60)) $((int_elapsed%60)))
  printf "  %-6s  %-14s  %-14s  %-14s  %-10s\n" \
    "${n_val}" "${seq_fmt}" "${int_fmt}" "${diff_str}s" "${winner}"
done < "${SUMMARY_FILE}"
echo
