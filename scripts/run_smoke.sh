#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_ROOT}"

if [[ "$#" -gt 0 ]]; then
  "${PYTHON_BIN}" -m pytest "$@"
  exit 0
fi

"${PYTHON_BIN}" -m pytest \
  tests/test_bootstrap.py \
  tests/test_env1.py \
  tests/test_env2.py \
  tests/test_env3a.py \
  tests/test_env3b.py \
  tests/test_env3c.py
