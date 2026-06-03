#!/usr/bin/env bash
# Download Qwen3-32B-AWQ from HuggingFace into models/.
# Requires HF_TOKEN in env and conda env ML active.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-32B-AWQ}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/Qwen3-32B-AWQ}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"

TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.51,<5.0}"
HF_HUB_SPEC="${HF_HUB_SPEC:-huggingface_hub>=0.30,<1.0}"

log() {
  printf '[setup/download_model] %s\n' "$*"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  log "installing uv to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || {
    echo "[setup/download_model] ERROR: uv install did not put uv on PATH" >&2
    exit 1
  }
}

install_download_dependencies() {
  ensure_uv
  log "Installing download dependencies into ${CONDA_PREFIX} via uv"
  uv pip install --python "$(command -v python)" "${TRANSFORMERS_SPEC}" "${HF_HUB_SPEC}"
}

download_from_huggingface() {
  log "Downloading ${MODEL_REPO} → ${MODEL_DIR}"
  HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  MODEL_REPO="${MODEL_REPO}" \
  MODEL_DIR="${MODEL_DIR}" \
  python - <<'PY'
from __future__ import annotations
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    repo_type="model",
    local_dir=os.environ["MODEL_DIR"],
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
}

write_model_path_env() {
  log "Recording MODEL_PATH in ${ENV_FILE}"
  MODEL_DIR="${MODEL_DIR}" ENV_FILE="${ENV_FILE}" python - <<'PY'
from __future__ import annotations
import os
from pathlib import Path

env_file = Path(os.environ["ENV_FILE"])
model_dir = os.environ["MODEL_DIR"]
lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
updated = False
for index, line in enumerate(lines):
    if line.startswith("MODEL_PATH="):
        lines[index] = f"MODEL_PATH={model_dir}"
        updated = True
        break
if not updated:
    lines.append(f"MODEL_PATH={model_dir}")
env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

main() {
  if [[ "${CONDA_DEFAULT_ENV:-}" != "ML" ]]; then
    echo "[setup/download_model] ERROR: activate conda env ML first" >&2
    exit 1
  fi

  if [[ -f "${MODEL_DIR}/config.json" ]]; then
    log "SKIP: ${MODEL_DIR}/config.json already exists"
    exit 0
  fi

  install_download_dependencies
  mkdir -p "${MODEL_DIR}"
  download_from_huggingface
  write_model_path_env
  log "download_model done"
}

main "$@"
