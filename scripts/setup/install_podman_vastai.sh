#!/usr/bin/env bash
# Phase 3 of trace-sim-vastai-pipeline plan: install podman + dependencies
# on a fresh vast.ai container so the upstream SWE-Bench harness (which
# hard-imports `docker` SDK) can talk to a podman socket via DOCKER_HOST.
#
# Pre-mortem A items 1+2 are encoded here as fail-loud preflights:
#   1. fuse-overlayfs missing → fail with descriptive message (do NOT
#      silently fall back to vfs storage driver — vfs makes a SWE-Bench
#      image pull take ~45 minutes and consume 3x disk).
#   2. /etc/subuid missing for current user → print exact remediation
#      command and exit nonzero (do NOT auto-run usermod, which requires
#      root and can affect other tenants on shared hosts).
#
# Usage:
#   bash scripts/setup/install_podman_vastai.sh
#
# Idempotent: re-running on a properly configured host is a no-op.

set -euo pipefail

SCRIPT_NAME="install_podman_vastai"
log() { echo "[${SCRIPT_NAME}] $*"; }
fatal() { echo "[${SCRIPT_NAME}] FATAL: $*" >&2; exit 1; }

log "Phase 3 of trace-sim-vastai-pipeline plan"
log "Target: rootless podman + DOCKER_HOST shim for vast.ai (no DinD)"

# ─── Step 1: Install packages (apt-get + idempotent dpkg-query check) ──
PACKAGES=(podman fuse-overlayfs slirp4netns uidmap)

if command -v apt-get &>/dev/null; then
    log "Detected apt-get; checking package state for: ${PACKAGES[*]}"
    missing=()
    for pkg in "${PACKAGES[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
            missing+=("$pkg")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        log "Installing missing packages: ${missing[*]}"
        # Use sudo only if not running as root (vast.ai sometimes runs as root)
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update && apt-get install -y "${missing[@]}"
        else
            sudo apt-get update && sudo apt-get install -y "${missing[@]}"
        fi
    else
        log "All packages already installed"
    fi
else
    log "apt-get not available — assuming packages are pre-installed"
    # Verify podman is at least findable; otherwise this will explode in the
    # preflight steps below.
    command -v podman &>/dev/null || fatal "podman not found and apt-get unavailable; install manually"
fi

# ─── Step 2: Pre-mortem A item 1 — fuse-overlayfs availability ──
# Check that the fuse kernel module is loadable OR /dev/fuse exists.
# Without FUSE, podman falls back to vfs storage which is catastrophically
# slow for SWE-Bench images.
log "Preflight: fuse-overlayfs availability"

if [ -e /dev/fuse ]; then
    log "  /dev/fuse exists ✓"
elif command -v modprobe &>/dev/null && modprobe fuse 2>/dev/null; then
    log "  modprobe fuse succeeded ✓"
else
    fatal "FUSE not available. Without FUSE, podman falls back to the vfs storage driver, which makes SWE-Bench image pulls take ~45min and consume 3x disk. Resolve by: (a) picking a vast.ai instance template with FUSE enabled, (b) running the container with --device /dev/fuse, or (c) asking your provider to enable fuse-overlayfs in the kernel. DO NOT proceed with vfs."
fi

# ─── Step 3: Pre-mortem A item 2 — /etc/subuid configuration ──
log "Preflight: /etc/subuid configuration for current user"

CURRENT_USER="$(id -un)"

# Skip the subuid check when running as root — root doesn't need user
# namespace remapping for rootful container ops.
if [ "$(id -u)" -eq 0 ]; then
    log "  running as root; subuid check skipped (rootful path)"
else
    if [ ! -f /etc/subuid ]; then
        fatal "/etc/subuid does not exist. Rootless podman cannot allocate user namespaces without subuid mappings. Remediation (run as root): touch /etc/subuid /etc/subgid && usermod --add-subuids 100000-165535 --add-subgids 100000-165535 ${CURRENT_USER}"
    fi

    if ! grep -q "^${CURRENT_USER}:" /etc/subuid; then
        fatal "User '${CURRENT_USER}' has no entry in /etc/subuid. Rootless podman will fail with EPERM on every container op. Remediation (run as root): usermod --add-subuids 100000-165535 --add-subgids 100000-165535 ${CURRENT_USER}"
    fi

    if ! grep -q "^${CURRENT_USER}:" /etc/subgid; then
        fatal "User '${CURRENT_USER}' has no entry in /etc/subgid. Same EPERM failure as missing subuid. Remediation (run as root): usermod --add-subgids 100000-165535 ${CURRENT_USER}"
    fi

    log "  ${CURRENT_USER} has subuid + subgid entries ✓"
fi

# ─── Step 4: Smoke pull `hello-world` to validate the storage driver ──
log "Smoke: pull hello-world via podman to validate storage driver"

if ! podman pull docker.io/library/hello-world &>/tmp/podman_smoke.log; then
    cat /tmp/podman_smoke.log >&2
    fatal "podman pull hello-world failed; see output above. This usually indicates either a registry connectivity issue OR a storage driver misconfiguration."
fi

# Verify storage driver is overlay (not vfs)
storage_driver="$(podman info --format '{{.Store.GraphDriverName}}' 2>/dev/null || echo unknown)"
if [ "$storage_driver" = "vfs" ]; then
    fatal "podman is using the 'vfs' storage driver instead of 'overlay'. This will make SWE-Bench evaluation catastrophically slow. Check that fuse-overlayfs is installed and that the container has access to /dev/fuse."
elif [ "$storage_driver" = "overlay" ]; then
    log "  storage driver: overlay ✓"
else
    log "  WARNING: unexpected storage driver '${storage_driver}' (expected 'overlay'). Proceeding cautiously."
fi

# ─── Step 5: Save podman info to log for debugging ──
mkdir -p .omc/logs
podman info > .omc/logs/phase3-podman-info.log 2>&1 || true
log "podman info saved to .omc/logs/phase3-podman-info.log"

# ─── Step 6: Done ──
log "Podman is installed and configured for rootless operation"
log "Next: bash scripts/setup/start_podman_socket.sh"
log "      then: export DOCKER_HOST=unix:///run/user/\$UID/podman/podman.sock"
