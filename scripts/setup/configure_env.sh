#!/usr/bin/env bash
# Generate .env from .env.example and write HF_TOKEN + MODEL_PATH.
# Expects HF_TOKEN to be set in the environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[setup] ERROR: HF_TOKEN is not set" >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/Qwen3-32B-AWQ}"

if [[ ! -f ".env" ]]; then
    echo "[setup] Creating .env from .env.example"
    cp .env.example .env
fi

# Update or append a key=value in .env (in-place, no temp file needed via Python)
python - <<PY
import os
from pathlib import Path

env_file = Path("${REPO_ROOT}/.env")
updates = {
    "HF_TOKEN": os.environ["HF_TOKEN"],
    "MODEL_PATH": "${MODEL_PATH}",
}

lines = env_file.read_text().splitlines()
found = {k: False for k in updates}

for i, line in enumerate(lines):
    for key, val in updates.items():
        if line.startswith(f"{key}=") or line.startswith(f"# {key}="):
            lines[i] = f"{key}={val}"
            found[key] = True

for key, val in updates.items():
    if not found[key]:
        lines.append(f"{key}={val}")

env_file.write_text("\n".join(lines) + "\n")
print("[setup] .env updated")
PY
