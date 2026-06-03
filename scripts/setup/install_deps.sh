#!/usr/bin/env bash
# Install repo (editable) into the active conda env. Caller must activate ML.
#
# Uses uv as the package installer (faster than pip; reads pyproject.toml the
# same way). Env management still belongs to conda — uv only writes packages
# into ${CONDA_PREFIX}.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "ML" ]]; then
    echo "[install_deps] ERROR: activate conda env ML first" >&2
    exit 1
fi

if python -c "import trace_collect" >/dev/null 2>&1; then
    echo "[install_deps] SKIP: trace_collect already importable"
    exit 0
fi

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    echo "[install_deps] installing uv to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    command -v uv >/dev/null 2>&1 || {
        echo "[install_deps] ERROR: uv install did not put uv on PATH" >&2
        exit 1
    }
}
ensure_uv

echo "[install_deps] Installing repo (editable) into ${CONDA_PREFIX} via uv"
uv pip install --python "$(command -v python)" -e ".[dev]"
echo "[install_deps] done"
