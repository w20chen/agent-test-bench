#!/usr/bin/env bash
# Download SWE-rebench (filtered split) and write to data/swe-rebench/tasks.json.
#
# Usage:
#   conda activate ML
#   ./scripts/setup/swe_rebench_data.sh
#
# Env vars:
#   SWE_REBENCH_SPLIT  — HF split to load (default: filtered; alternatives: test)
#   SWE_REBENCH_N      — Limit to top-N tasks via the plugin's select_subset
#                        (default: 0 = all rows, ~6542 for filtered split).
#                        Use this for quick iteration; the full dump is ~80 MB.
#
# Idempotent: if tasks.json already exists, prints the row count and exits 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

TASKS_FILE="data/swe-rebench/tasks.json"
SPLIT="${SWE_REBENCH_SPLIT:-filtered}"
N="${SWE_REBENCH_N:-0}"

if [[ -f "$TASKS_FILE" ]]; then
    count=$(python -c "import json; print(len(json.load(open('${TASKS_FILE}'))))")
    echo "[setup] SKIP swe_rebench_data: ${TASKS_FILE} already exists (${count} tasks)"
    exit 0
fi

mkdir -p "$(dirname "$TASKS_FILE")"
echo "[setup] Downloading nebius/SWE-rebench (split=${SPLIT}, n=${N})..."

PYTHONPATH="${REPO_ROOT}/src" python - <<PYEOF
import json
import os
from pathlib import Path

from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig

n = int(os.environ.get("SWE_REBENCH_N", "0"))
split = os.environ.get("SWE_REBENCH_SPLIT", "filtered")

config_path = Path("configs/benchmarks/swe-rebench.yaml")
config = BenchmarkConfig.from_yaml(config_path)
# Honor the SWE_REBENCH_SPLIT override without having to edit the YAML.
config.harness_split = split

plugin = get_benchmark_class(config.slug)(config)
tasks = plugin.load_tasks()
print(f"[setup] Loaded {len(tasks)} tasks from nebius/SWE-rebench ({split})")

if n > 0:
    tasks = plugin.select_subset(tasks, n=n)
    print(f"[setup] Selected first {len(tasks)} tasks via plugin.select_subset")

out = Path("${TASKS_FILE}")
out.write_text(
    json.dumps(tasks, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)
print(f"[setup] Wrote {len(tasks)} tasks to {out}")
PYEOF

echo "[setup] swe_rebench_data done"
