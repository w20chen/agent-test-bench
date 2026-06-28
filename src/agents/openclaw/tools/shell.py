import asyncio
import json
import os
import re
import shlex
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from agents.openclaw.tools.base import Tool
from trace_collect.exec_classifier import classify_exec_tool_name


MAX_EXEC_TOOL_TIMEOUT_SEC = 600


# ---------------------------------------------------------------------------
# In-container per-process-tree resource sampler (VTune coarse companion)
# ---------------------------------------------------------------------------

def _read_proc_stat(pid: int) -> tuple[int, int] | None:
    """Return ``(utime, stime)`` in jiffies from ``/proc/<pid>/stat``."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return None
    # comm field ``(name)`` can contain spaces; find last ``)``.
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 1 :].split()
    # fields[0]=state, [1]=ppid, …, [11]=utime (field 14), [12]=stime (field 15)
    if len(fields) > 12:
        try:
            return int(fields[11]), int(fields[12])
        except (ValueError, IndexError):
            pass
    return None


def _read_proc_statm_rss(pid: int) -> int | None:
    """Return RSS in pages from ``/proc/<pid>/statm``, or None."""
    try:
        text = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return None
    parts = text.split()
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def _read_proc_io(pid: int) -> tuple[int, int] | None:
    """Return ``(read_bytes, write_bytes)`` from ``/proc/<pid>/io``."""
    try:
        text = Path(f"/proc/{pid}/io").read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return None
    rb = wb = 0
    found_any = False
    for line in text.splitlines():
        if line.startswith("read_bytes:"):
            try:
                rb = int(line.split(":", 1)[1].strip())
                found_any = True
            except (ValueError, IndexError):
                pass
        elif line.startswith("write_bytes:"):
            try:
                wb = int(line.split(":", 1)[1].strip())
                found_any = True
            except (ValueError, IndexError):
                pass
    return (rb, wb) if found_any else None


def _read_proc_ctxt(pid: int) -> int | None:
    """Return total context switches from ``/proc/<pid>/status``."""
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return None
    total = 0
    found = False
    for line in text.splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            if key in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
                try:
                    total += int(parts[1].strip())
                    found = True
                except ValueError:
                    pass
    return total if found else None


def _get_descendant_pids(root_pid: int) -> list[int]:
    """Return all descendant PIDs of *root_pid* (including *root_pid*).

    Walks ``/proc/<pid>/task/*/children`` recursively.  Silently skips
    processes that have already exited.
    """
    seen: set[int] = set()
    queue = [root_pid]
    while queue:
        pid = queue.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children_text = Path(
                f"/proc/{pid}/task/{pid}/children"
            ).read_text(encoding="utf-8")
        except (OSError, PermissionError):
            continue
        for child_str in children_text.split():
            try:
                queue.append(int(child_str))
            except ValueError:
                pass
    return sorted(seen)


def _sample_proc_tree(
    root_pid: int,
    interval_s: float,
    stop_event: threading.Event,
    out_path: str,
) -> None:
    """Background thread: periodically sample the process tree rooted at *root_pid*.

    Writes one JSON object per line to *out_path* (JSONL).  Each line has:
    ``epoch``, ``cpu_jiffies`` (cumulative), ``rss_kb`` (instantaneous),
    ``disk_read_bytes`` / ``disk_write_bytes`` (cumulative),
    ``context_switches`` (cumulative), ``n_pids`` (descendant count).

    The first sample may have zero children if the process hasn't spawned
    them yet — consumers should discard it or handle jiffies==0 gracefully.
    """
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fh:
        while not stop_event.is_set():
            tick = time.time()
            pids = _get_descendant_pids(root_pid)
            total_jiffies = 0
            total_rss_pages = 0
            total_disk_r = 0
            total_disk_w = 0
            total_ctxt = 0

            for pid in pids:
                stat = _read_proc_stat(pid)
                if stat is not None:
                    total_jiffies += stat[0] + stat[1]
                rss = _read_proc_statm_rss(pid)
                if rss is not None:
                    total_rss_pages += rss
                io_vals = _read_proc_io(pid)
                if io_vals is not None:
                    total_disk_r += io_vals[0]
                    total_disk_w += io_vals[1]
                ctxt = _read_proc_ctxt(pid)
                if ctxt is not None:
                    total_ctxt += ctxt

            sample: dict[str, Any] = {
                "epoch": tick,
                "cpu_jiffies": total_jiffies,
                "rss_kb": int(total_rss_pages * 4),  # page size = 4 KiB on Linux
                "disk_read_bytes": total_disk_r,
                "disk_write_bytes": total_disk_w,
                "context_switches": total_ctxt,
                "n_pids": len(pids),
            }
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fh.flush()

            elapsed = time.time() - tick
            remainder = max(0.0, interval_s - elapsed)
            if stop_event.wait(remainder):
                break


class ExecTool(Tool):
    _DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",  # del /f, del /q
            r"\brmdir\s+/s\b",  # rmdir /s
            r"(?:^|[;&|]\s*)format\b",  # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",  # disk operations
            r"\bdd\s+if=",  # dd
            r">\s*/dev/sd",  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = MAX_EXEC_TOOL_TIMEOUT_SEC
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 300, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        env = os.environ.copy()
        if self.path_append:
            # Prepend bootstrap/tool paths so they take priority over host
            # ~/.local/bin (critical on ARM+QEMU where the host home directory
            # is bind-mounted and contains ARM64 binaries that cannot execute
            # inside the x86_64 container).
            env["PATH"] = self.path_append + os.pathsep + env.get("PATH", "")

        # When --vtune is active (signalled via env from the host), wrap a
        # matching tool invocation with VTune so it is profiled for exactly
        # its lifetime.  Detection uses the project's exec classifier (same
        # logic that produces ``exec-pytest``, ``exec-make``, etc. tool names
        # in traces) and matches against the ``VTUNE_TOOLS`` env var
        # (comma-separated full tool names, default ``exec-pytest``).
        #
        # Examples recognised as exec-pytest:
        #   ``pytest test_foo.py``, ``python -m pytest tests/``,
        #   ``timeout 120 pytest -v``, ``cd /x && pytest``
        # Examples NOT recognised:
        #   ``pip install pytest`` (classifies as ``exec-pip``),
        #   ``echo pytest`` (classifies as ``exec-echo``),
        #   ``grep pytest *.py`` (classifies as ``exec-grep``)
        run_command = command
        vtune_window: dict[str, Any] | None = None
        vtune_tools_raw = os.environ.get("VTUNE_TOOLS", "exec-pytest")
        vtune_tools = {t.strip() for t in vtune_tools_raw.split(",") if t.strip()}
        if os.environ.get("VTUNE_PROFILE") == "1" and classify_exec_tool_name(
            "exec", {"command": command}
        ) in vtune_tools:
            vtune_bin = os.environ.get("VTUNE_BIN", "vtune")
            vtune_out = os.environ.get("VTUNE_OUT", cwd)
            # Microsecond-resolution timestamp to avoid directory collisions
            # when concurrent pytest invocations start in the same second.
            now_ns = time.time_ns()
            run_dir = os.path.join(
                vtune_out,
                f"pytest_{time.strftime('%Y%m%dT%H%M%S')}"
                f"_{(now_ns // 1_000) % 1_000_000:06d}_{os.getpid()}",
            )
            os.makedirs(run_dir, exist_ok=True)
            run_command = (
                f"{shlex.quote(vtune_bin)} -collect uarch-exploration -data-limit=0 "
                f"-r {shlex.quote(os.path.join(run_dir, 'result'))} "
                f"-- bash -lc {shlex.quote(command)}"
            )
            vtune_window = {"dir": run_dir, "cmd": command, "ts_start": time.time()}

        # Per-invocation proc-tree sampler (only for VTune-wrapped commands).
        # Runs in-container alongside the subprocess to collect per-pytest
        # CPU / memory / disk-I/O / context-switch samples that are scoped
        # to the exact process tree, avoiding container-level cgroup
        # contamination from concurrent tool invocations.
        _sampler_thread: threading.Thread | None = None
        _sampler_stop: threading.Event | None = None

        try:
            process = await asyncio.create_subprocess_shell(
                run_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=(sys.platform != "win32"),
            )

            if vtune_window is not None and sys.platform == "linux":
                _sampler_stop = threading.Event()
                _sampler_thread = threading.Thread(
                    target=_sample_proc_tree,
                    args=(
                        process.pid,
                        0.5,
                        _sampler_stop,
                        os.path.join(vtune_window["dir"], "per_tool_samples.jsonl"),
                    ),
                    daemon=True,
                )
                _sampler_thread.start()
                # Give the subprocess a moment to spawn its children before
                # the sampler's first tick, so the first retained sample has
                # meaningful process-tree coverage.
                time.sleep(0.05)

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # Kill the entire process group so descendants (e.g. python
                # train.py spawned by `sh -c`) don't survive as orphans.
                if sys.platform != "win32":
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError) as e:
                        logger.debug("killpg failed: {}", e)
                        process.kill()
                else:
                    process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    if sys.platform != "win32":
                        try:
                            os.waitpid(process.pid, os.WNOHANG)
                        except (ProcessLookupError, ChildProcessError) as e:
                            logger.debug("Process already reaped or not found: {}", e)
                return f"Error: Command timed out after {effective_timeout} seconds"
            finally:
                # Always stop the per-tool proc-tree sampler, even on timeout.
                if _sampler_stop is not None:
                    _sampler_stop.set()
                if _sampler_thread is not None:
                    _sampler_thread.join(timeout=2.0)

            if vtune_window is not None:
                vtune_window["ts_end"] = time.time()
                vtune_window["returncode"] = process.returncode
                try:
                    with open(
                        os.path.join(vtune_window["dir"], "window.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(vtune_window, f)
                except OSError:
                    pass

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Head + tail truncation to preserve both start and end of output
            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        from agents.openclaw.security.network import contains_internal_url

        if contains_internal_url(cmd):
            return (
                "Error: Command blocked by safety guard (internal/private URL detected)"
            )

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return (
                    "Error: Command blocked by safety guard (path traversal detected)"
                )

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    p = Path(expanded).expanduser().resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        # Windows: match drive-root paths like `C:\` as well as `C:\path\to\file`
        # NOTE: `*` is required so `C:\` (nothing after the slash) is still extracted.
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]*", command)
        posix_paths = re.findall(
            r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command
        )  # POSIX: /absolute only
        home_paths = re.findall(
            r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command
        )  # POSIX/Windows home shortcut: ~
        return win_paths + posix_paths + home_paths
