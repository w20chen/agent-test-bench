#!/usr/bin/env bash
# Pre-pull Docker images referenced by SWE-rebench tasks.
#
# Usage:
#   ./scripts/setup/pull_swe_rebench_images.sh [tasks.json] [--sample N] [--parallel N]
#
# Options:
#   --sample N       Pull images for the first N tasks only.
#   --instance-ids   Comma-separated list of instance_ids to pull.
#   --parallel N     Concurrent pulls (default: 1 = sequential).
#
# Arguments:
#   tasks.json  — path to a tasks JSON file. Default: data/swe-rebench/tasks.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASKS_FILE=""
SAMPLE=""
INSTANCE_IDS=""
PARALLEL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)
            SAMPLE="${2:-}"
            shift 2
            ;;
        --instance-ids)
            INSTANCE_IDS="${2:-}"
            shift 2
            ;;
        --parallel)
            PARALLEL="${2:-1}"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            TASKS_FILE="$1"
            shift
            ;;
    esac
done

TASKS_FILE="${TASKS_FILE:-${PROJECT_ROOT}/data/swe-rebench/tasks.json}"

if [[ ! -f "$TASKS_FILE" ]]; then
    echo "[pull-images] ERROR: tasks.json not found at $TASKS_FILE" >&2
    echo "[pull-images] Run: make download-swe-rebench" >&2
    exit 1
fi

# Build the python filter expression.
PY_FILTER="tasks = json.load(open(sys.argv[1]))"
if [[ -n "$INSTANCE_IDS" ]]; then
    # Match specific instance_ids (order preserved).
    PY_FILTER="$PY_FILTER
ids = set(sys.argv[2].split(','))
tasks = [t for t in tasks if t.get('instance_id') in ids]"
    PY_ARGS=("$TASKS_FILE" "$INSTANCE_IDS")
elif [[ -n "$SAMPLE" ]]; then
    PY_FILTER="$PY_FILTER
tasks = tasks[:int(sys.argv[2])]"
    PY_ARGS=("$TASKS_FILE" "$SAMPLE")
else
    PY_ARGS=("$TASKS_FILE")
fi

echo "[pull-images] Filtering tasks from $TASKS_FILE ..."

IMAGES=$(python3 -c "
import json, sys
$PY_FILTER
images = sorted(set(
    t.get('docker_image') or t.get('image_name')
    for t in tasks
    if t.get('docker_image') or t.get('image_name')
))
for img in images:
    print(img)
" "${PY_ARGS[@]}")

if [[ -z "$IMAGES" ]]; then
    echo "[pull-images] No images found to pull."
    exit 0
fi

TOTAL=$(echo "$IMAGES" | grep -c . || true)
echo "[pull-images] Pulling $TOTAL images (parallel=$PARALLEL)"
echo ""

if [[ "$PARALLEL" -le 1 ]]; then
    COUNT=0
    FAILED=0
    while IFS= read -r image; do
        [[ -z "$image" ]] && continue
        COUNT=$((COUNT + 1))
        echo "[pull-images] [$COUNT/$TOTAL] $image"
        if docker pull "$image" 2>&1 | tail -1; then
            echo "  → OK"
        else
            echo "  → FAILED (continuing...)"
            FAILED=$((FAILED + 1))
        fi
    done <<< "$IMAGES"
else
    # Parallel pulls via xargs.
    echo "$IMAGES" | xargs -P "$PARALLEL" -I {} bash -c '
        echo "[pull] {}"
        if docker pull {} >/dev/null 2>&1; then
            echo "[pull] {} → OK"
        else
            echo "[pull] {} → FAILED"
        fi
    '
    FAILED=0  # xargs doesn't easily track failures; assume best-effort
fi

echo ""
echo "[pull-images] Done: $TOTAL images attempted"

