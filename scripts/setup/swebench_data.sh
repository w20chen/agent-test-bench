#!/usr/bin/env bash
# Download SWE-bench Verified and select tool-intensive tasks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

TASKS_FILE="data/swebench_verified/tasks.json"

if [[ -f "$TASKS_FILE" ]]; then
    count=$(python -c "import json; print(len(json.load(open('${TASKS_FILE}'))))")
    echo "[setup] SKIP swebench_data: ${TASKS_FILE} already exists (${count} tasks)"
    exit 0
fi

echo "[setup] Downloading SWE-bench Verified dataset..."
PYTHONPATH="${REPO_ROOT}/src" python - <<'PYEOF'
import json
from pathlib import Path

from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig

config = BenchmarkConfig.from_yaml(Path("configs/benchmarks/swe-bench-verified.yaml"))
plugin = get_benchmark_class(config.slug)(config)
tasks = plugin.select_subset(plugin.load_tasks())

out = Path("data/swebench_verified/tasks.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(tasks, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)
print(f"[setup] Wrote {len(tasks)} tasks to {out}")
PYEOF
echo "[setup] swebench_data done"
