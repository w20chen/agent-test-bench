#!/usr/bin/env bash
# Ephemeral cloud-GPU setup for Terminal-Bench + OpenClaw runs.
#
# Uses uv directly (no conda) — see CLAUDE.md "Runtime entry points".
# Verified on Ubuntu 22.04 + RTX 6000 Ada (fresh instance, 64.247.196.218),
# total ~2 min: project deps 5s + torch 30s + 30 GB FP8 snapshot 60s.
#
# Two backend modes (set SETUP_BACKEND):
#   transformers  — HF transformers + KV recording backend (capstone runs)
#   vllm          — OpenAI-compatible vLLM server (no recording)
#   both          — install both into .venv (let uv resolver pick torch)
#
# Tunables (env vars):
#   REPO_ROOT         — repo root              (default: script_dir/../..)
#   VENV_PATH         — .venv location         (default: $REPO_ROOT/.venv)
#   PYTHON_VERSION    — Python pin             (default: 3.12)
#   MODEL_ID          — HF model id            (default: Qwen3-Coder-30B-A3B-Instruct-FP8)
#                         FP8 default — bf16 30B-A3B needs a 96 GB card.
#   HF_HOME           — HF cache root          (default: $HOME/hf_cache)
#   HF_TOKEN          — required for private repos; REQUIRE_HF_TOKEN=1 default
#   SETUP_BACKEND     — transformers|vllm|both (default: transformers)
#   INSTALL_TORCH     — install torch into venv (default: 1; vllm pulls its own)
#   PREWARM_MODEL     — snapshot_download model (default: 1)
#   TORCH_SPEC        — pip spec               (default: torch==2.6.0+cu124)
#   TORCH_INDEX_URL                           (default: pytorch.org cu124 index)
#   VLLM_SPEC         — pip spec               (default: vllm)
#   HF_HUB_ENABLE_HF_TRANSFER                 (default: 1; Rust parallel downloader)
#   REQUIRE_HF_TOKEN  — fail if HF_TOKEN unset (default: 1; private-repo users)
#   SETUP_DOCKER_FIREWALL — iptables INPUT changes (default: 0; opt-in)
#
# Two-layer startup design (do NOT confuse them):
#
#   Layer 1 — cloud-provider "startup command" box: runs once at instance boot,
#             typically as root. Responsible for switching to the SSH-in user
#             via sudo -Hu, cloning the repo into ${HOME}/agent-sched-bench,
#             and invoking this script. Provider logs may capture this block.
#
#   Layer 2 — this script: self-contained install + venv + model prewarm.
#             Hard-asserts non-root invocation (run via Layer 1's sudo -Hu).
#             Uses sudo internally for the apt/docker steps that need root.
#             Does NOT persist anything to host shell config (no /etc/profile.d,
#             no .bashrc edits). After it returns, env is gone — print_next_steps
#             shows the one-liner that re-establishes env for the actual run.
#
# Layer-1 (paste into cloud startup-command box):
#   /usr/bin/bash <<'STARTUP'
#   set -euo pipefail
#   TARGET_USER="$(getent passwd | awk -F: '$3>=1000 && $3<60000 && $6 ~ /^\/home\// {print $1; exit}')"
#   export HF_TOKEN="..."   # WARNING: provider logs this line; rotate after run
#   sudo --preserve-env=HF_TOKEN -Hu "${TARGET_USER}" bash -c '
#     set -euo pipefail
#     REPO_DIR="${HOME}/agent-sched-bench"
#     [ -d "${REPO_DIR}/.git" ] || /usr/bin/git clone --branch main \
#       https://github.com/HCAHOI/agent-sched-bench.git "${REPO_DIR}"
#     cd "${REPO_DIR}"
#     /usr/bin/git pull --ff-only origin main
#     /usr/bin/bash scripts/setup/terminal_bench_server.sh
#   '
#   STARTUP
#
# Layer-2 (manual SSH-in form, equivalent — already non-root):
#   cd ~/agent-sched-bench
#   HF_TOKEN=hf_xxx bash scripts/setup/terminal_bench_server.sh
set -euo pipefail

# This script assumes it runs as a non-root /home user. The Layer-1
# startup-command box must do the user switch (sudo -Hu <user>) before
# invoking this script — see docstring above. apt/docker calls below use
# sudo internally for the few steps that need root.
if [ "$(id -u)" -eq 0 ]; then
  echo "[terminal-bench-setup] ERROR: this script must run as a non-root /home user." >&2
  echo "[terminal-bench-setup]        Wrap your invocation in sudo -Hu <user>." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

SETUP_LOG_DIR="${SETUP_LOG_DIR:-${HOME}}"
SETUP_LOG_PATH="${SETUP_LOG_PATH:-${SETUP_LOG_DIR}/terminal-bench-setup-$(date -u +%Y%m%dT%H%M%SZ).log}"
SETUP_STATUS_PATH="${SETUP_STATUS_PATH:-${HOME}/terminal-bench-setup.status}"
mkdir -p "$(dirname "${SETUP_LOG_PATH}")"
exec > >(tee -a "${SETUP_LOG_PATH}") 2>&1

VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
SETUP_BACKEND="${SETUP_BACKEND:-transformers}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
PREWARM_MODEL="${PREWARM_MODEL:-1}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0+cu124}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
REQUIRE_HF_TOKEN="${REQUIRE_HF_TOKEN:-1}"
SETUP_DOCKER_FIREWALL="${SETUP_DOCKER_FIREWALL:-0}"

log()   { printf '[terminal-bench-setup] %s\n' "$*"; }
fatal() { printf '[terminal-bench-setup] ERROR: %s\n' "$*" >&2; exit 1; }

write_status() {
  local state="$1"
  local detail="${2:-}"
  {
    printf 'state=%s\n'          "${state}"
    printf 'detail=%s\n'         "${detail}"
    printf 'backend=%s\n'        "${SETUP_BACKEND}"
    printf 'model=%s\n'          "${MODEL_ID}"
    printf 'updated_utc=%s\n'    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo=%s\n'           "${REPO_ROOT}"
    printf 'venv=%s\n'           "${VENV_PATH}"
    printf 'log=%s\n'            "${SETUP_LOG_PATH}"
  } > "${SETUP_STATUS_PATH}"
}

on_exit() {
  local status=$?
  if [ "${status}" -eq 0 ]; then
    write_status "ok" "setup completed"
  else
    write_status "failed" "exit_status=${status}"
  fi
  # tee runs in a background subshell; ensure its stdout is flushed before exit
  exec >&- 2>&- || true
  wait 2>/dev/null || true
}
trap on_exit EXIT
write_status "running" "setup started backend=${SETUP_BACKEND}"
log "writing log to ${SETUP_LOG_PATH}"
log "writing status to ${SETUP_STATUS_PATH}"
log "backend=${SETUP_BACKEND} model=${MODEL_ID} venv=${VENV_PATH}"

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    fatal "need root privileges for: $*"
  fi
}

target_user() {
  if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
    printf '%s\n' "${SUDO_USER}"
  else
    id -un
  fi
}

install_system_packages() {
  command -v apt-get >/dev/null 2>&1 || fatal "apt-get not found; this setup targets Ubuntu/Debian"
  log "installing system packages"
  as_root apt-get update -qq
  # Intentionally no python — uv installs its own. Docker handled separately.
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    acl \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    iproute2 \
    iptables \
    jq \
    pkg-config \
    rsync \
    tmux \
    unzip \
    zstd
}

ensure_docker_usable() {
  if ! command -v docker >/dev/null 2>&1; then
    log "docker not present; installing docker.io"
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io docker-compose-plugin
    as_root systemctl enable --now docker >/dev/null 2>&1 \
      || as_root service docker start >/dev/null 2>&1 \
      || true
  fi
  local user
  user="$(target_user)"
  if getent group docker >/dev/null 2>&1; then
    as_root usermod -aG docker "${user}" || true
  fi
  # setfacl gives the current user docker.sock access without needing a new
  # login (group changes don't take effect mid-session).
  if [ -S /var/run/docker.sock ] && command -v setfacl >/dev/null 2>&1; then
    as_root setfacl -m "u:${user}:rw" /var/run/docker.sock || true
  fi
  if docker info >/dev/null 2>&1; then
    log "docker usable by ${user}"
  elif as_root docker info >/dev/null 2>&1; then
    log "docker works via root; setfacl should make user access work in this session"
  else
    fatal "docker daemon is not usable"
  fi
}

configure_docker_bridge_firewall() {
  [ "${SETUP_DOCKER_FIREWALL}" = "1" ] || {
    log "skipping iptables changes (SETUP_DOCKER_FIREWALL=0; opt-in only)"
    return 0
  }
  command -v iptables >/dev/null 2>&1 || return 0
  # docker0 = default bridge; br-<hash> = user-defined networks
  log "allowing Docker bridge traffic to host services (docker0 + br-*)"
  as_root bash -c '
    set -e
    iptables -C INPUT -i docker0 -p tcp -j ACCEPT 2>/dev/null || \
      iptables -I INPUT 1 -i docker0 -p tcp -j ACCEPT
    iptables -C INPUT -i br-+ -p tcp -j ACCEPT 2>/dev/null || \
      iptables -I INPUT 1 -i br-+ -p tcp -j ACCEPT
  ' || log "warning: iptables update failed; set HF_RECORDING_PUBLIC_HOST manually if containers cannot reach the host"
}

install_uv() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    log "uv already installed ($(uv --version))"
    return 0
  fi
  # Some cloud images ship /home/$USER/.config as root-owned, which makes the
  # uv installer fail when it writes its self-update receipt. Force-fix
  # ownership before the curl|sh step.
  local user
  user="$(target_user)"
  as_root install -d -m 0755 -o "${user}" -g "${user}" "${HOME}/.config"
  log "installing uv to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || fatal "uv install did not put uv on PATH"
  log "uv $(uv --version) installed"
}

setup_python_env() {
  export PATH="${HOME}/.local/bin:${PATH}"
  log "ensuring Python ${PYTHON_VERSION} via uv-managed binaries"
  uv python find "${PYTHON_VERSION}" >/dev/null 2>&1 || uv python install "${PYTHON_VERSION}"

  log "creating venv at ${VENV_PATH}"
  uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}" --seed

  local py="${VENV_PATH}/bin/python"
  [ -x "${py}" ] || fatal "venv python missing at ${py}"

  log "installing project (editable) + dev extras"
  uv pip install --python "${py}" -e ".[dev]"
  log "installing huggingface_hub[hf_transfer] for parallel HF downloads"
  uv pip install --python "${py}" "huggingface_hub[hf_transfer]>=0.30,<1.0"
}

install_backend_torch() {
  [ "${INSTALL_TORCH}" = "1" ] || return 0
  local py="${VENV_PATH}/bin/python"
  if "${py}" - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
  then
    log "torch+CUDA already available in venv"
  else
    log "installing ${TORCH_SPEC} from ${TORCH_INDEX_URL}"
    uv pip install --python "${py}" --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
  fi
  "${py}" - <<'PY'
import torch
print(f"[terminal-bench-setup] torch={torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[terminal-bench-setup] gpu={torch.cuda.get_device_name(0)}")
PY
}

install_backend_vllm() {
  local py="${VENV_PATH}/bin/python"
  log "installing ${VLLM_SPEC}"
  uv pip install --python "${py}" "${VLLM_SPEC}"
  "${py}" -c "import vllm; print(f'[terminal-bench-setup] vllm={vllm.__version__}')"
}

install_backend() {
  case "${SETUP_BACKEND}" in
    transformers) install_backend_torch ;;
    vllm)         install_backend_vllm ;;
    both)         install_backend_torch; install_backend_vllm ;;
    *) fatal "SETUP_BACKEND must be one of: transformers, vllm, both (got '${SETUP_BACKEND}')" ;;
  esac
}

configure_huggingface() {
  export HF_HOME
  export HF_HUB_ENABLE_HF_TRANSFER
  mkdir -p "${HF_HOME}"

  if command -v git-lfs >/dev/null 2>&1; then
    git lfs install --skip-repo >/dev/null 2>&1 || true
  fi

  "${VENV_PATH}/bin/python" - <<'PY'
import importlib.metadata as md
print(f"[terminal-bench-setup] huggingface_hub={md.version('huggingface-hub')}")
PY

  if [ -z "${HF_TOKEN:-}" ]; then
    if [ "${REQUIRE_HF_TOKEN}" = "1" ]; then
      fatal "HF_TOKEN is not set; export it before invoking this script (needed for HF private repos — see REQUIRE_HF_TOKEN=0 to bypass for public-only setups)"
    fi
    log "HF_TOKEN not set; public-repo access only"
    return 0
  fi
  # Persist token to the standard HF storage so subsequent HF API / git-LFS
  # calls work without HF_TOKEN re-export. `hf auth login` exists in both
  # huggingface_hub 0.x and 1.x with identical semantics.
  log "writing HF token via hf auth login"
  "${VENV_PATH}/bin/hf" auth login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null
  local who
  who=$("${VENV_PATH}/bin/python" -c \
    "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>&1) \
    || fatal "HF auth login wrote token but whoami failed: ${who}"
  log "HF authenticated as ${who}"
}

prefetch_terminal_bench() {
  log "loading Terminal-Bench registry and pinned dataset"
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "${VENV_PATH}/bin/python" - <<'PY'
from pathlib import Path
from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig

config = BenchmarkConfig.from_yaml(Path("configs/benchmarks/terminal-bench.yaml"))
benchmark = get_benchmark_class(config.slug)(config)
tasks = benchmark.load_tasks()
print(f"[terminal-bench-setup] loaded_tasks={len(tasks)}")
if not tasks:
    raise SystemExit("Terminal-Bench load_tasks() returned empty list")
PY
}

prewarm_model() {
  [ "${PREWARM_MODEL}" = "1" ] || return 0
  export HF_HOME
  mkdir -p "${HF_HOME}"

  log "prefetching ${MODEL_ID} into HF_HOME=${HF_HOME}"
  HF_HOME="${HF_HOME}" \
  HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  MODEL_ID="${MODEL_ID}" \
    "${VENV_PATH}/bin/python" - <<'PY'
import os, time
from huggingface_hub import snapshot_download

t0 = time.time()
path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"[terminal-bench-setup] model_cached={path} elapsed={time.time()-t0:.1f}s")
PY
}

verify_setup() {
  log "verifying setup"
  "${VENV_PATH}/bin/python" - <<'PY'
import importlib.metadata as md
import sys

print(f"[terminal-bench-setup] python={sys.version.split()[0]}")
for package in ("agent-sched-bench", "terminal-bench", "transformers", "torch", "huggingface-hub"):
    try:
        print(f"[terminal-bench-setup] {package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"[terminal-bench-setup] WARNING {package} not installed")
PY
  docker version --format '[terminal-bench-setup] docker client={{.Client.Version}} server={{.Server.Version}}' || true
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || log "warning: nvidia-smi not available"
}

print_next_steps() {
  cat <<EOF

[terminal-bench-setup] DONE (backend=${SETUP_BACKEND})

Launch fix-git smoke (transformers + KV recording, single copy-paste):

  cd ${REPO_ROOT} && source ${VENV_PATH}/bin/activate && \\
    INSTANCE_ID=fix-git \\
    MODEL_ID=${MODEL_ID} \\
    ./scripts/launch_kv_capstone.sh none baseline-fix-git

(launch_kv_capstone.sh already defaults HF_HOME, HF_HUB_OFFLINE=1,
 TRANSFORMERS_OFFLINE=1, OPENAI_API_KEY=dummy — no extra exports needed.)

vLLM serve (after running this script with SETUP_BACKEND=vllm or both):

  source ${VENV_PATH}/bin/activate && \\
    vllm serve ${MODEL_ID} --port 44345 --host 127.0.0.1
  # then point openclaw at http://127.0.0.1:44345/v1 via OPENAI_BASE_URL
EOF
}

install_system_packages
ensure_docker_usable
configure_docker_bridge_firewall
install_uv
setup_python_env
install_backend
configure_huggingface
prefetch_terminal_bench
prewarm_model
verify_setup
print_next_steps
