#!/usr/bin/env bash
# Single-entry bootstrap for new Linux x86_64 servers.
#
# Steps run in order; each marks ~/.bootstrap-progress on success so re-runs
# resume from the failing step. Force-rerun a step by deleting its line from
# the progress file.
#
# Required env vars (only for the final smoke step):
#   BOOTSTRAP_SMOKE_API_BASE  — OpenAI-compatible endpoint, e.g.
#                                http://127.0.0.1:44345/v1 (vllm/sglang)
# Optional:
#   BOOTSTRAP_SMOKE_API_KEY   — default: dummy
#   BOOTSTRAP_SMOKE_MODEL     — default: Qwen/Qwen3-Coder-30B-A3B-Instruct
#   HF_TOKEN                  — required for download_model step
#   BOOTSTRAP_VASTAI=1        — force the podman step on non-vastai hosts
#   BOOTSTRAP_SKIP            — comma-separated step keys to skip (e.g.
#                                "swe_rebench_data,download_model,terminal_bench_smoke"
#                                for terminal-bench-only hosts that don't need
#                                the swe-rebench dataset or the pinned AWQ model)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROGRESS="${HOME}/.bootstrap-progress"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
touch "${PROGRESS}"
cd "${REPO_ROOT}"

is_done() { grep -q "^${1}=ok$" "${PROGRESS}"; }
mark_done() { echo "${1}=ok" >> "${PROGRESS}"; }

is_skipped() {
    local key="$1"
    [[ -z "${BOOTSTRAP_SKIP:-}" ]] && return 1
    # match key as a comma-separated token in BOOTSTRAP_SKIP
    [[ ",${BOOTSTRAP_SKIP}," == *",${key},"* ]]
}

step() {
    local key="$1"; shift
    if is_skipped "${key}"; then
        echo "[bootstrap] SKIP ${key} (in BOOTSTRAP_SKIP)"
        return 0
    fi
    if is_done "${key}"; then
        echo "[bootstrap] SKIP ${key}"
        return 0
    fi
    echo "[bootstrap] >>> ${key}"
    "$@"
    mark_done "${key}"
}

do_arm_check() {
    local arch
    arch="$(uname -m)"
    if [[ "${arch}" == "aarch64" || "${arch}" == "arm64" ]]; then
        if [[ "$(id -u)" -eq 0 ]]; then
            bash "${SETUP_DIR}/arm_setup.sh" install
        else
            sudo bash "${SETUP_DIR}/arm_setup.sh" install
        fi
    else
        echo "[bootstrap] arm_check: x86_64, no-op"
    fi
}

do_podman() {
    if [[ ! -e /etc/vastai-host ]] && [[ "${BOOTSTRAP_VASTAI:-}" != "1" ]]; then
        echo "[bootstrap] podman: not on vast.ai, skipping podman setup"
        return 0
    fi
    bash "${SETUP_DIR}/install_podman_vastai.sh"
    bash "${SETUP_DIR}/start_podman_socket.sh"
}

do_miniconda() {
    if command -v conda >/dev/null 2>&1; then
        echo "[bootstrap] miniconda: conda already on PATH"
        return 0
    fi
    if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
        echo "[bootstrap] miniconda: ${CONDA_ROOT} present, sourcing"
    else
        local installer="/tmp/miniconda3-installer.sh"
        echo "[bootstrap] miniconda: downloading installer"
        curl -fsSL -o "${installer}" \
            https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash "${installer}" -b -p "${CONDA_ROOT}"
        rm -f "${installer}"
    fi
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda --version
}

# Source conda for steps that follow (idempotent).
_source_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    fi
}

do_conda_env_ml() {
    _source_conda
    if conda env list | awk '{print $1}' | grep -qx "ML"; then
        echo "[bootstrap] conda_env_ml: ML already exists"
        return 0
    fi
    conda create -n ML -y python=3.12
}

do_install_uv() {
    if command -v uv >/dev/null 2>&1; then
        echo "[bootstrap] install_uv: uv already on PATH ($(uv --version))"
        return 0
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    command -v uv >/dev/null 2>&1 || {
        echo "[bootstrap] install_uv: uv install did not put uv on PATH" >&2
        exit 1
    }
    uv --version
}

do_clone_repos() {
    _source_conda
    conda run -n ML bash "${SETUP_DIR}/clone_repos.sh" \
        data/swe-rebench/tasks.json data/swe-rebench/repos
}

do_install_deps() {
    _source_conda
    conda run -n ML bash "${SETUP_DIR}/install_deps.sh"
}

do_swe_rebench_data() {
    _source_conda
    conda run -n ML bash "${SETUP_DIR}/swe_rebench_data.sh"
}

do_download_model() {
    _source_conda
    : "${HF_TOKEN:?set HF_TOKEN to download the model}"
    conda run -n ML bash "${SETUP_DIR}/download_model.sh"
}

do_terminal_bench_smoke() {
    _source_conda
    : "${BOOTSTRAP_SMOKE_API_BASE:?set BOOTSTRAP_SMOKE_API_BASE (e.g. http://127.0.0.1:44345/v1) — required for end-to-end smoke}"
    local api_key="${BOOTSTRAP_SMOKE_API_KEY:-dummy}"
    local model="${BOOTSTRAP_SMOKE_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
    PYTHONPATH=src OPENAI_API_KEY="${api_key}" \
        conda run -n ML python -m trace_collect.cli \
            --provider openai \
            --api-base "${BOOTSTRAP_SMOKE_API_BASE}" \
            --api-key "${api_key}" \
            --model "${model}" \
            --benchmark terminal-bench \
            --scaffold openclaw \
            --container docker \
            --mcp-config none \
            --max-iterations 50 \
            --instance-ids fix-git
}

step arm_check            do_arm_check
step podman               do_podman
step miniconda            do_miniconda
step conda_env_ml         do_conda_env_ml
step install_uv           do_install_uv
step clone_repos          do_clone_repos
step install_deps         do_install_deps
step swe_rebench_data     do_swe_rebench_data
step download_model       do_download_model
step terminal_bench_smoke do_terminal_bench_smoke

echo "[bootstrap] DONE — conda env ML ready, terminal-bench smoke passed"
