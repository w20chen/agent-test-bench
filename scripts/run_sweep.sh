#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/src/harness/sweep.py" ]]; then
  echo "Sweep orchestrator is not implemented yet." >&2
  exit 1
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/src" python -m harness.sweep "$@"
