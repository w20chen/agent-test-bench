#!/usr/bin/env bash
# run_demo_scheduler.sh - Run demo workloads through the tool scheduler.
#
# Each workload is scheduled individually and results are saved to
# demo_scheduler_profiles.jsonl.
# Results can then be summarized with:
#   python summarize_scheduler_profiles.py demo_scheduler_profiles.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCHEDULER_PKG="prototype.tool_scheduler"
OUTPUT="$SCRIPT_DIR/demo_scheduler_profiles.jsonl"
PYTHON="${PYTHON:-python3}"

# Clear previous output
rm -f "$OUTPUT"

echo "=== Tool Scheduler Demo ==="
echo "Output: $OUTPUT"
echo ""

run_schedule() {
    local desc="$1"
    shift
    echo "--- $desc ---"
    $PYTHON -m "$SCHEDULER_PKG" \
        --output "$OUTPUT" \
        --dry-run \
        --save-samples \
        --verbose \
        --cooldown 5.0 \
        --alpha 0.3 \
        -- "$@"
    echo ""
}

# 1. Short tool (sleep 0.5s)
run_schedule "Short tool" sleep 0.5

# 2. CPU serial workload
run_schedule "CPU serial" "$PYTHON" -c "
import time
t0=time.monotonic()
while time.monotonic()-t0<8:
    sum(i**0.5 for i in range(20000))
"

# 3. CPU parallel workload (4 workers)
run_schedule "CPU parallel (4 workers)" "$PYTHON" "$SCRIPT_DIR/workloads/cpu_parallel.py" --workers 4 --seconds 8

# 4. CPU parallel workload (8 workers)
run_schedule "CPU parallel (8 workers)" "$PYTHON" "$SCRIPT_DIR/workloads/cpu_parallel.py" --workers 8 --seconds 8

# 5. Phased CPU workload
run_schedule "Phased CPU" "$PYTHON" "$SCRIPT_DIR/prototype/tool_scheduler/workloads/phased_cpu.py"

# 6. Memory scan workload
run_schedule "Memory scan" "$PYTHON" "$SCRIPT_DIR/prototype/tool_scheduler/workloads/memory_scan.py" --seconds 8

# 7. Process tree workload
run_schedule "Process tree" "$PYTHON" "$SCRIPT_DIR/workloads/process_tree.py" --children 4 --seconds 5

echo ""
echo "=== Demo complete ==="
echo "Results saved to: $OUTPUT"
echo "Summarize with: python summarize_scheduler_profiles.py $OUTPUT"
