#!/usr/bin/env bash
# Phase 3 of trace-sim-vastai-pipeline plan: start a long-lived podman
# system service on a Unix socket and export DOCKER_HOST so the upstream
# SWE-Bench harness's `docker.from_env()` calls transparently route to
# podman.
#
# Pre-mortem A item 3: --time=0 disables idle shutdown so the socket
# does not disappear mid-run. The PID is persisted to .omc/state/podman.pid
# so the smoke script can liveness-check + restart if needed.
#
# Usage:
#   bash scripts/setup/start_podman_socket.sh
#   eval "$(bash scripts/setup/start_podman_socket.sh --print-export)"
#
# The second form prints the DOCKER_HOST export to stdout so a parent
# shell can `eval` it. The first form just starts the service and prints
# the export instructions to stderr.

set -euo pipefail

SCRIPT_NAME="start_podman_socket"
log() { echo "[${SCRIPT_NAME}] $*" >&2; }
fatal() { echo "[${SCRIPT_NAME}] FATAL: $*" >&2; exit 1; }

PRINT_EXPORT=0
if [ "${1:-}" = "--print-export" ]; then
    PRINT_EXPORT=1
fi

# ─── Step 1: Locate the user runtime dir + socket path ──
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
mkdir -p "${RUNTIME_DIR}/podman" 2>/dev/null || true
SOCKET_PATH="${RUNTIME_DIR}/podman/podman.sock"
DOCKER_HOST_VALUE="unix://${SOCKET_PATH}"

log "Runtime dir: ${RUNTIME_DIR}"
log "Socket path: ${SOCKET_PATH}"

# ─── Step 2: Ensure .omc/state exists for PID file ──
mkdir -p .omc/state

PID_FILE=".omc/state/podman.pid"

# ─── Step 3: Check if a podman service is already running on this socket ──
if [ -f "$PID_FILE" ]; then
    existing_pid="$(cat "$PID_FILE")"
    if kill -0 "$existing_pid" 2>/dev/null; then
        log "podman system service already running (pid ${existing_pid})"
        if [ -S "$SOCKET_PATH" ]; then
            log "  socket present ✓"
            if [ "$PRINT_EXPORT" -eq 1 ]; then
                echo "export DOCKER_HOST=${DOCKER_HOST_VALUE}"
            else
                log ""
                log "To use this socket from the harness, run:"
                log "  export DOCKER_HOST=${DOCKER_HOST_VALUE}"
            fi
            exit 0
        else
            log "  WARNING: pid file references a live process but socket is missing"
            log "  killing stale process and restarting"
            kill "$existing_pid" 2>/dev/null || true
        fi
    else
        log "stale pid file (process ${existing_pid} dead); removing"
        rm -f "$PID_FILE"
    fi
fi

# ─── Step 4: Start a fresh podman system service with --time=0 ──
log "Starting podman system service (--time=0, no idle shutdown)"

# Run in background with nohup so the parent shell can exit cleanly
nohup podman system service \
    --time=0 \
    "${DOCKER_HOST_VALUE}" \
    > .omc/logs/podman-service.log 2>&1 &

new_pid=$!

# Detach from the parent shell's process group so the service survives
# the parent's exit. This requires `disown` if available; the bash
# `disown` builtin is the standard way.
disown "$new_pid" 2>/dev/null || true

echo "$new_pid" > "$PID_FILE"
log "podman system service started (pid ${new_pid})"
log "logs: .omc/logs/podman-service.log"

# ─── Step 5: Wait for the socket to become available ──
log "Waiting for socket to become available"

for i in 1 2 3 4 5 6 7 8 9 10; do
    if [ -S "$SOCKET_PATH" ]; then
        log "  socket ready after ${i} attempts ✓"
        break
    fi
    sleep 0.5
done

if [ ! -S "$SOCKET_PATH" ]; then
    fatal "socket ${SOCKET_PATH} did not appear after 5 seconds. Check .omc/logs/podman-service.log for errors."
fi

# ─── Step 6: Verify the docker SDK can talk to it ──
log "Verifying docker SDK compatibility (DOCKER_HOST=${DOCKER_HOST_VALUE})"

if command -v python3 &>/dev/null; then
    if DOCKER_HOST="${DOCKER_HOST_VALUE}" python3 -c "
import sys
try:
    import docker
except ImportError:
    print('  WARNING: python docker SDK not installed; skipping smoke check', file=sys.stderr)
    sys.exit(0)
client = docker.from_env()
assert client.ping(), 'docker.from_env().ping() returned False'
print('  docker.from_env().ping() == True ✓', file=sys.stderr)
" ; then
        :
    else
        fatal "docker SDK could not talk to the podman socket. Check .omc/logs/podman-service.log."
    fi
fi

# ─── Step 7: Print the DOCKER_HOST export ──
if [ "$PRINT_EXPORT" -eq 1 ]; then
    # Print to stdout so the caller can `eval` it
    echo "export DOCKER_HOST=${DOCKER_HOST_VALUE}"
else
    log ""
    log "Podman socket is up. To use it from the harness, run:"
    log "  export DOCKER_HOST=${DOCKER_HOST_VALUE}"
    log ""
    log "Or eval the export form:"
    log "  eval \"\$(bash scripts/setup/start_podman_socket.sh --print-export)\""
fi
