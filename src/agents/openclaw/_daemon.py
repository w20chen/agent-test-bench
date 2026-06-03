"""Background process management for OpenClaw async mode."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def write_pid_file(
    pid_file: Path,
    pid: int,
    session_id: str,
    *,
    trace_file: Path | None = None,
) -> None:
    """Write a JSON PID file for a daemon process.

    ``trace_file`` is persisted so ``get_session_status()`` can locate the
    canonical trace JSONL directly.
    """
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "pid": pid,
        "session_id": session_id,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if trace_file is not None:
        data["trace_file"] = str(trace_file)
    pid_file.write_text(json.dumps(data) + "\n", encoding="utf-8")

def read_pid_file(pid_file: Path) -> dict[str, Any] | None:
    if not pid_file.exists():
        return None
    try:
        return json.loads(pid_file.read_text(encoding="utf-8").strip())
    except (json.JSONDecodeError, OSError):
        return None

def cleanup_pid_file(pid_file: Path) -> None:
    try:
        pid_file.unlink(missing_ok=True)
    except OSError:
        pass

def pid_file_for_session(workspace: Path, session_id: str) -> Path:
    return workspace / ".openclaw" / "pids" / f"{session_id}.pid"

def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def spawn_daemon(
    cmd: list[str],
    pid_file: Path,
    session_id: str,
    *,
    extra_env: dict[str, str] | None = None,
    trace_file: Path | None = None,
) -> int:
    """Spawn a detached daemon process and write its PID file."""
    log_dir = pid_file.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{session_id}.log"

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    with open(log_file, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )

    write_pid_file(pid_file, proc.pid, session_id, trace_file=trace_file)
    return proc.pid

def get_session_status(session_id: str, workspace: Path) -> dict[str, Any]:
    """Get the current status of a session."""
    pid_file = pid_file_for_session(workspace, session_id)
    pid_data = read_pid_file(pid_file)

    status: str
    pid: int | None = None
    trace_file: Path | None = None

    if pid_data:
        pid = pid_data.get("pid")
        if pid_data.get("trace_file"):
            trace_file = Path(pid_data["trace_file"])
        if pid and _is_pid_alive(pid):
            status = "running"
        else:
            # Keep the PID file after exit so repeated status checks can still
            # resolve the persisted trace path for completed sessions.
            status = "completed"
    else:
        status = "unknown"

    n_actions = 0
    n_iterations = 0
    elapsed_s = 0.0
    if trace_file is not None and trace_file.exists():
        try:
            distinct_iters: set[int] = set()
            for line in trace_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_type = rec.get("type")
                if rec_type == "action":
                    n_actions += 1
                    distinct_iters.add(rec.get("iteration", 0))
                elif rec_type == "summary":
                    elapsed_s = rec.get("elapsed_s", elapsed_s)
                    n_actions = rec.get("n_actions", n_actions)
                    n_iterations = rec.get("n_iterations", n_iterations)
            if not n_iterations:
                n_iterations = len(distinct_iters)
        except OSError:
            pass

    return {
        "session_id": session_id,
        "status": status,
        "pid": pid,
        "workspace": str(workspace),
        "trace_file": str(trace_file) if trace_file is not None else None,
        "n_actions": n_actions,
        "n_iterations": n_iterations,
        "elapsed_s": round(elapsed_s, 2),
    }
