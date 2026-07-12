#!/usr/bin/env bash
# Download SWE-bench Verified and write a task cache.
#
# Usage:
#   conda activate ML
#   ./scripts/setup/swebench_data.sh
#
# Env vars:
#   SWE_BENCH_VERIFIED_N  Limit to N tool-intensive tasks using the plugin's
#                         select_subset. Default: config selection_n (32).
#                         Set to 0 for the full SWE-bench Verified split.
#
# Idempotent: if tasks.json already exists, prints the row count and exits 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

TASKS_FILE="data/swebench_verified/tasks.json"
N="${SWE_BENCH_VERIFIED_N:-}"

if [[ -f "$TASKS_FILE" ]]; then
    count=$(python -c "import json; print(len(json.load(open('${TASKS_FILE}'))))")
    echo "[setup] SKIP swebench_data: ${TASKS_FILE} already exists (${count} tasks)"
    exit 0
fi

echo "[setup] Downloading SWE-bench Verified dataset (n=${N:-config})..."
PYTHONPATH="${REPO_ROOT}/src" python - <<'PYEOF'
import json
import os
from pathlib import Path

from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig

config = BenchmarkConfig.from_yaml(Path("configs/benchmarks/swe-bench-verified.yaml"))
plugin = get_benchmark_class(config.slug)(config)
tasks = plugin.load_tasks()

n_value = os.environ.get("SWE_BENCH_VERIFIED_N")
if n_value is None or n_value == "":
    tasks = plugin.select_subset(tasks)
    selection_label = f"config selection_n={config.selection_n}"
else:
    n = int(n_value)
    if n < 0:
        raise ValueError("SWE_BENCH_VERIFIED_N must be >= 0")
    if n > 0:
        tasks = plugin.select_subset(tasks, n=n)
        selection_label = f"subset n={n}"
    else:
        selection_label = "full split"

out = Path("data/swebench_verified/tasks.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(tasks, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)
print(f"[setup] Wrote {len(tasks)} tasks to {out} ({selection_label})")
PYEOF
echo "[setup] swebench_data done"
