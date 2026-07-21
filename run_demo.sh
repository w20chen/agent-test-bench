#!/usr/bin/env bash
# run_demo.sh - Run demo workloads through the tool profiler.
#
# Each workload is profiled individually and results are saved to demo_profiles.jsonl.
# Results can then be summarized with: python summarize_profiles.py demo_profiles.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILER_PKG="prototype.tool_profiler"
OUTPUT="$SCRIPT_DIR/demo_profiles.jsonl"
PYTHON="${PYTHON:-python3}"

# Clear previous output
rm -f "$OUTPUT"

echo "=== Tool Profiler Demo ==="
echo "Output: $OUTPUT"
echo ""

run_profile() {
    local desc="$1"
    shift
    echo "--- $desc ---"
    $PYTHON -m "$PROFILER_PKG" \
        --warmup-seconds 2.0 \
        --sample-interval 0.2 \
        --output "$OUTPUT" \
        --verbose \
        -- "$@"
    echo ""
}

# 1. Short tool (sleep 0.5s)
run_profile "Short tool" sleep 0.5

# 2. Sleep 5s (idle/waiting)
run_profile "Sleep 5s" sleep 5

# 3. CPU serial workload
run_profile "CPU serial" "$PYTHON" "$SCRIPT_DIR/workloads/cpu_serial.py" --seconds 5

# 4. CPU parallel workload (4 workers)
run_profile "CPU parallel (4 workers)" "$PYTHON" "$SCRIPT_DIR/workloads/cpu_parallel.py" --workers 4 --seconds 5

# 5. I/O workload
run_profile "I/O worker" "$PYTHON" "$SCRIPT_DIR/workloads/io_worker.py" --seconds 5

# 6. Process tree workload
run_profile "Process tree" "$PYTHON" "$SCRIPT_DIR/workloads/process_tree.py" --children 4 --seconds 5

# 7. Build scenario (make -j) if available
if command -v make &>/dev/null; then
    # Create a temporary directory with a trivial C file for make
    TMPDIR=$(mktemp -d)
    cat > "$TMPDIR/Makefile" <<'MAKEFILE'
.PHONY: all
all:
	@echo "build: nothing to do (demo target)"
MAKEFILE
    run_profile "Make (trivial)" make -C "$TMPDIR" -j "$(nproc 2>/dev/null || echo 2)"
    rm -rf "$TMPDIR"
else
    echo "--- Make (skipped: make not found) ---"
    echo ""
fi

echo "=== Demo complete ==="
echo ""
echo "Summary:"
$PYTHON "$SCRIPT_DIR/summarize_profiles.py" "$OUTPUT"
