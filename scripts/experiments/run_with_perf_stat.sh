#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/experiments/run_with_perf_stat.sh OUTPUT_DIR -- COMMAND [ARGS...]

Wrap a command with system-wide perf stat for Kunpeng/ARM placement runs.
The wrapper fails if perf itself fails, so partial counter output is not
mistaken for a valid measurement.
EOF
}

if [[ $# -lt 3 || "${2:-}" != "--" ]]; then
  usage >&2
  exit 2
fi

out_dir="$1"
shift 2
mkdir -p "${out_dir}"

events=(
  cycles
  instructions
  cache-references
  cache-misses
  r04
  r03
  r14
  r01
  r12
  r10
  r19
)

event_csv="$(IFS=,; echo "${events[*]}")"
{
  printf 'events=%s\n' "${event_csv}"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
} > "${out_dir}/perf_command.txt"

perf stat -x, --no-big-num -a -e "${event_csv}" -o "${out_dir}/perf_stat.csv" -- "$@"
