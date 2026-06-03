#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo bash scripts/setup/arm_setup.sh [install|status|check] [--dry-run] [--smoke-image IMAGE]

Commands:
  install   Enable amd64 cross-arch execution support on ARM Docker hosts.
  status    Print current ARM/cross-arch runtime status.
  check     Run an amd64 smoke container and verify execution works.

Options:
  --dry-run           Print actions without executing them.
  --smoke-image IMG   Override the amd64 smoke image.
                      Default: docker.io/library/busybox:1.36.1

Environment:
  ARM_SETUP_NON_INTERACTIVE=1  Suppress interactive behavior (currently informational only).
EOF
}

log() {
  printf '[arm-setup] %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "This command must run as root." >&2
    exit 1
  fi
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run]'
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

host_arch() {
  uname -m
}

is_arm_host() {
  case "$(host_arch)" in
    aarch64|arm64) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_binfmt_misc() {
  if [[ -e /proc/sys/fs/binfmt_misc/status ]]; then
    return 0
  fi
  if command -v modprobe >/dev/null 2>&1; then
    run_cmd modprobe binfmt_misc
  fi
  if [[ ! -e /proc/sys/fs/binfmt_misc/status ]]; then
    run_cmd mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc
  fi
}

show_status() {
  log "host arch: $(host_arch)"
  log "docker server arch: $(docker info --format '{{.Architecture}}')"
  if [[ -e /proc/sys/fs/binfmt_misc/status ]]; then
    log "binfmt_misc: $(cat /proc/sys/fs/binfmt_misc/status)"
  else
    log "binfmt_misc: unavailable"
  fi
  if [[ -e /proc/sys/fs/binfmt_misc/qemu-x86_64 ]]; then
    log "qemu-x86_64 handler: present"
    sed -n '1,20p' /proc/sys/fs/binfmt_misc/qemu-x86_64
  else
    log "qemu-x86_64 handler: missing"
  fi
}

install_amd64_emulation() {
  ensure_binfmt_misc
  log "installing amd64 binfmt handler via tonistiigi/binfmt"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    run_cmd docker run --privileged --rm tonistiigi/binfmt --install amd64
    return 0
  fi
  if docker run --privileged --rm tonistiigi/binfmt --install amd64; then
    return 0
  fi
  log "docker-based binfmt install failed; falling back to apt packages"
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get is unavailable, and tonistiigi/binfmt install failed." >&2
    return 1
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y qemu-user-static binfmt-support
  if command -v update-binfmts >/dev/null 2>&1; then
    update-binfmts --enable qemu-x86_64 || true
  fi
  if command -v systemctl >/dev/null 2>&1; then
    systemctl restart systemd-binfmt || true
  fi
}

run_smoke_check() {
  local output
  log "running amd64 smoke check with ${SMOKE_IMAGE}"
  if output="$(docker run --rm --platform linux/amd64 "${SMOKE_IMAGE}" /bin/sh -lc 'uname -m && echo arm-setup-ok' 2>&1)"; then
    :
  else
    printf '%s\n' "${output}" >&2
    return 1
  fi
  printf '%s\n' "${output}"
  if ! grep -Eq 'x86_64|amd64' <<<"${output}"; then
    echo "Smoke check did not report an amd64 userspace." >&2
    return 1
  fi
  if ! grep -q 'arm-setup-ok' <<<"${output}"; then
    echo "Smoke check marker missing from amd64 container output." >&2
    return 1
  fi
}

main() {
  local mode="install"
  DRY_RUN=0
  SMOKE_IMAGE="docker.io/library/busybox:1.36.1"

  if [[ $# -gt 0 ]]; then
    case "$1" in
      install|status|check)
        mode="$1"
        shift
        ;;
    esac
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --smoke-image)
        SMOKE_IMAGE="${2:-}"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  need_cmd docker
  need_cmd uname
  need_cmd grep
  need_cmd sed

  if [[ "${mode}" == "status" ]]; then
    show_status
    exit 0
  fi

  require_root

  if ! is_arm_host; then
    log "host is not ARM; no amd64-on-arm bootstrap needed"
    if [[ "${mode}" == "check" ]]; then
      run_smoke_check
    else
      show_status
    fi
    exit 0
  fi

  if [[ "${mode}" == "check" ]]; then
    show_status
    run_smoke_check
    exit 0
  fi

  log "pre-install status"
  show_status
  install_amd64_emulation
  log "post-install status"
  show_status
  run_smoke_check
}

main "$@"
