#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-${REPO_ROOT}/results}"
RESULTS_SOURCE="${RESULTS_SOURCE:-}"

if [[ -z "${RESULTS_SOURCE}" ]]; then
  echo "Set RESULTS_SOURCE to a remote rsync path such as user@host:/path/to/results/." >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for result collection." >&2
  exit 1
fi

mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -avz "${RESULTS_SOURCE}" "${LOCAL_RESULTS_DIR}/"
