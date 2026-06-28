"""Execute tool calls inside Docker/Podman containers via a persistent agent."""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_OPENCLAW_EXEC_DEFAULT_TIMEOUT_S = 300.0
_OPENCLAW_EXEC_MAX_TIMEOUT_S = 600.0


def _unwrap_tool_args(
    *,
    tool_name: str | None,
    tool_args_json: str,
) -> tuple[str | None, dict[str, Any], bool]:
    """Return (resolved_tool_name, params, is_nested_openclaw_style)."""
    try:
        parsed = json.loads(tool_args_json or "{}")
    except json.JSONDecodeError:
        raise
    if not isinstance(parsed, dict):
        return tool_name, {}, False

    if tool_name and isinstance(parsed.get(tool_name), dict):
        return tool_name, parsed[tool_name], True

    if len(parsed) == 1:
        only_name, only_value = next(iter(parsed.items()))
        if isinstance(only_value, dict):
            return (tool_name or only_name), only_value, True

    return tool_name, parsed, False


def _resolve_exec_timeout_s(params: dict[str, Any]) -> float:
    """Mirror OpenClaw ExecTool timeout semantics during replay.

    Source traces may omit ``timeout`` when the tool relied on its default.
    To preserve source behavior, replay must use the original tool default
    instead of the simulator's global fallback. Explicit values are capped to
    the same 600s ceiling enforced by ``ExecTool``.
    """

    raw_timeout = params.get("timeout", _OPENCLAW_EXEC_DEFAULT_TIMEOUT_S)
    try:
        timeout_s = float(raw_timeout)
    except (TypeError, ValueError):
        timeout_s = _OPENCLAW_EXEC_DEFAULT_TIMEOUT_S
    if timeout_s <= 0:
        timeout_s = _OPENCLAW_EXEC_DEFAULT_TIMEOUT_S
    return min(timeout_s, _OPENCLAW_EXEC_MAX_TIMEOUT_S)


# ---------------------------------------------------------------------------
# In-container replay agent script.
#
# Runs as a single persistent python3 process inside the Docker container.
# Reads JSON-line requests from stdin, dispatches to tool handlers,
# writes JSON-line responses to stdout.  All subprocess.run calls use
# capture_output=True to prevent stdout pollution of the protocol.
# ---------------------------------------------------------------------------
_REPLAY_AGENT_SCRIPT = textwrap.dedent(r'''
import json, os, sys, subprocess, difflib, signal, time

def _find_match(content, old_text):
    if old_text in content:
        return old_text, content.count(old_text)
    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()
    candidates = []
    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i : i + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))
    if candidates:
        return candidates[0], len(candidates)
    return None, 0

def _not_found_msg(old_text, content, path):
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = len(old_lines)
    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(None, old_lines, lines[i:i+window]).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i
    if best_ratio > 0.5:
        diff = "\n".join(difflib.unified_diff(
            old_lines, lines[best_start:best_start+window],
            fromfile="old_text (provided)",
            tofile=f"{path} (actual, line {best_start+1})", lineterm=""))
        return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%} similar) at line {best_start+1}:\n{diff}"
    return f"Error: old_text not found in {path}. No similar text found. Verify the file content."

_MAX_OUTPUT = 10_000

def _truncate_output(text, limit=_MAX_OUTPUT):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n\n... ({len(text) - limit} chars truncated) ...\n\n" + text[-half:]

def _exec_command(cmd, timeout, cwd="/testbed"):
    """Run one shell command; returns (text, returncode, timed_out).

    Output format matches :class:`ExecTool`: stdout, then ``STDERR:\\n``
    prefix for stderr.  The ``Exit code:`` line is appended by the
    orchestration layer (``execute_trace_tool``) so both the
    original collect path and the replay path produce identical output.
    """
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"}
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd,
                           capture_output=True, text=True, timeout=timeout, env=env)
        output_parts = []
        if r.stdout:
            output_parts.append(r.stdout.rstrip())
        if r.stderr and r.stderr.strip():
            output_parts.append(f"STDERR:\n{r.stderr.rstrip()}")
        result = "\n".join(output_parts) if output_parts else "(no output)"
        return result, r.returncode, False
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds", 124, True

def handle_exec(args):
    cmd = args.get("command", "")
    timeout = args.get("timeout", 600)
    output, rc, _timed_out = _exec_command(cmd, timeout)
    return {"ok": rc == 0, "result": _truncate_output(output), "returncode": rc}

def handle_commands(args):
    cmds = args.get("commands", [])
    timeout = args.get("timeout", 600)
    all_output = []
    last_rc = 0
    for i, cmd in enumerate(cmds):
        output, rc, _timed_out = _exec_command(cmd, timeout)
        if len(cmds) > 1:
            all_output.append(f"[call {i}]\n{output}")
        else:
            all_output.append(output)
        last_rc = rc
    combined = "\n".join(all_output) if all_output else ""
    return {"ok": last_rc == 0, "result": combined, "returncode": last_rc}

_READ_MAX_CHARS = 128_000
_READ_DEFAULT_LIMIT = 2000

def handle_read_file(args):
    path = args.get("path", "")
    # 1-indexed offset (matches ReadFileTool default: offset=1 = first line).
    offset = int(args.get("offset", 1))
    limit = int(args.get("limit", _READ_DEFAULT_LIMIT))
    try:
        content = open(path).read()
        if not content:
            return {"ok": True, "result": f"(Empty file: {path})"}
        lines = content.splitlines()
        total = len(lines)
        if offset < 1:
            offset = 1
        if offset > total:
            return {
                "ok": False,
                "result": f"Error: offset {offset} is beyond end of file ({total} lines)",
            }
        start = offset - 1
        end = min(start + limit, total)
        selected = lines[start:end]
        numbered = "\n".join(
            f"{start + i + 1}| {ln}" for i, ln in enumerate(selected)
        )
        if end < total:
            numbered += (
                f"\n\n(Showing lines {offset}-{end} of {total}."
                f" Use offset={end + 1} to continue.)"
            )
        else:
            numbered += f"\n\n(End of file — {total} lines total)"
        if len(numbered) > _READ_MAX_CHARS:
            trimmed: list[str] = []
            chars = 0
            for line in numbered.splitlines(keepends=True):
                chars += len(line)
                if chars > _READ_MAX_CHARS:
                    break
                trimmed.append(line.rstrip("\n"))
            numbered = "\n".join(trimmed)
        return {"ok": True, "result": numbered}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}

def handle_write_file(args):
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return {"ok": True, "result": f"Successfully wrote {len(content)} bytes to {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}

def handle_edit_file(args):
    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    replace_all = args.get("replace_all", False)
    try:
        raw = open(path, "rb").read()
        uses_crlf = b"\r\n" in raw
        content = raw.decode("utf-8").replace("\r\n", "\n")
        match, count = _find_match(content, old_text.replace("\r\n", "\n"))
        if match is None:
            return {"ok": False, "result": _not_found_msg(old_text, content, path)}
        if count > 1 and not replace_all:
            return {
                "ok": False,
                "result": (
                    f"Warning: old_text appears {count} times. "
                    "Provide more context to make it unique, or set replace_all=true."
                ),
            }
        norm_new = new_text.replace("\r\n", "\n")
        new_content = (
            content.replace(match, norm_new)
            if replace_all
            else content.replace(match, norm_new, 1)
        )
        if uses_crlf:
            new_content = new_content.replace("\n", "\r\n")
        open(path, "wb").write(new_content.encode("utf-8"))
        return {"ok": True, "result": f"Successfully edited {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error editing file: {e}"}

_LIST_IGNORE = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".coverage", "htmlcov",
}
_LIST_MAX = 200

def handle_list_dir(args):
    path = args.get("path", ".")
    recursive = args.get("recursive", False)
    max_entries = args.get("max_entries")
    try:
        if not os.path.exists(path):
            return {"ok": False, "result": f"Error: Directory not found: {path}"}
        if not os.path.isdir(path):
            return {"ok": False, "result": f"Error: Not a directory: {path}"}

        cap = max_entries or _LIST_MAX
        items: list[str] = []
        total = 0

        if recursive:
            for dirpath, dirnames, filenames in os.walk(path):
                # Filter ignored dirs in-place so os.walk does not
                # descend into them (mirrors Path.rglob semantics).
                dirnames[:] = sorted(
                    d for d in dirnames if d not in _LIST_IGNORE
                )
                rel_dir = os.path.relpath(dirpath, path)
                for d in dirnames:
                    total += 1
                    if len(items) < cap:
                        rel = (
                            os.path.join(rel_dir, d)
                            if rel_dir != "."
                            else d
                        )
                        items.append(f"{rel}/")
                for f in sorted(filenames):
                    if f in _LIST_IGNORE:
                        continue
                    total += 1
                    if len(items) < cap:
                        rel = (
                            os.path.join(rel_dir, f)
                            if rel_dir != "."
                            else f
                        )
                        items.append(rel)
        else:
            for name in sorted(os.listdir(path)):
                if name in _LIST_IGNORE:
                    continue
                total += 1
                if len(items) < cap:
                    pfx = (
                        "📁 "
                        if os.path.isdir(os.path.join(path, name))
                        else "📄 "
                    )
                    items.append(f"{pfx}{name}")

        if not items and total == 0:
            return {"ok": True, "result": f"Directory {path} is empty"}

        result = "\n".join(items)
        if total > cap:
            result += (
                f"\n\n(truncated, showing first {cap} of {total} entries)"
            )
        return {"ok": True, "result": result}
    except PermissionError as e:
        return {"ok": False, "result": f"Error: {e}"}
    except Exception as e:
        return {"ok": False, "result": f"Error listing directory: {e}"}

HANDLERS = {
    "exec": handle_exec,
    "commands": handle_commands,
    "read_file": handle_read_file,
    "write_file": handle_write_file,
    "edit_file": handle_edit_file,
    "list_dir": handle_list_dir,
}

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        tool = req.get("tool", "")
        args = req.get("args", {})
        handler = HANDLERS.get(tool)
        if handler:
            t0 = time.monotonic()
            resp = handler(args)
            resp["inner_duration_ms"] = (time.monotonic() - t0) * 1000
        else:
            resp = {"ok": False, "result": f"Error: Unsupported tool {tool!r}"}
    except Exception as e:
        resp = {"ok": False, "result": f"Error: agent dispatch failed: {e}"}
    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
    sys.stdout.flush()
''').strip()


# Idempotent tools safe to retry after agent restart.
_IDEMPOTENT_TOOLS = frozenset({"read_file", "list_dir"})


class ContainerAgent:

    # Container Python interpreter candidates, searched in order.
    # Mirrors the probing logic in task_container.resolve_running_container_exec_config.
    _PYTHON_CANDIDATES: tuple[str, ...] = (
        "/usr/bin/python3",
        "/usr/bin/python",
        "/usr/local/bin/python3",
        "/usr/local/bin/python",
        "python3",
        "python",
    )

    def __init__(
        self,
        container_id: str,
        container_executable: str,
        pythonpath: str | None = None,
    ) -> None:
        self._container_id = container_id
        self._executable = container_executable
        self._process: asyncio.subprocess.Process | None = None
        self._python_runtime: str = "python3"  # fallback, overwritten in start()
        self._pythonpath: str | None = pythonpath

    async def _probe_python(self) -> str:
        """Find a working Python >=3.11 interpreter inside the container."""
        probe_script = (
            "import sys; "
            "raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
        )
        for cand in self._PYTHON_CANDIDATES:
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._executable, "exec", "-i", "-w", "/testbed",
                    self._container_id, cand, "-c", probe_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.wait(), timeout=30)
                if proc.returncode == 0:
                    logger.info(
                        "ContainerAgent python probe: %s (cid=%s)",
                        cand, self._container_id[:12],
                    )
                    return cand
            except (asyncio.TimeoutError, OSError):
                continue
        raise RuntimeError(
            "ContainerAgent: no Python >=3.11 found in container "
            f"{self._container_id[:12]}.  Tried: "
            + ", ".join(self._PYTHON_CANDIDATES)
        )

    async def start(self) -> None:
        self._python_runtime = await self._probe_python()
        cmd: list[str] = [
            self._executable, "exec", "-i", "-w", "/testbed",
        ]
        # Propagate PYTHONPATH so replayed subprocesses (e.g. pytest)
        # can find packages installed by bootstrap_task_container_python.
        if self._pythonpath:
            cmd.extend(["-e", f"PYTHONPATH={self._pythonpath}"])
        cmd.extend([
            self._container_id, self._python_runtime, "-u", "-c", _REPLAY_AGENT_SCRIPT,
        ])
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,  # 1MB — agent responses can exceed default 64KB
        )
        logger.info(
            "ContainerAgent started: cid=%s pid=%s runtime=%s",
            self._container_id[:12], self._process.pid, self._python_runtime,
        )

    async def stop(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdin and not self._process.stdin.is_closing():
                self._process.stdin.close()
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            self._process.kill()
            await self._process.wait()
        self._process = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _restart(self) -> None:
        logger.warning("ContainerAgent restarting: cid=%s", self._container_id[:12])
        await self.stop()
        await self.start()

    async def execute(
        self,
        request: dict[str, Any],
        *,
        timeout_s: float = 600.0,
    ) -> dict[str, Any]:
        """Send a request and return the response. Restarts on crash."""
        tool_name = request.get("tool", "")
        for attempt in range(2):
            if not self.alive:
                if attempt == 0:
                    await self._restart()
                else:
                    return {"ok": False, "result": "Error: agent process dead"}

            proc = self._process
            assert proc is not None and proc.stdin is not None and proc.stdout is not None

            line = json.dumps(request, ensure_ascii=False) + "\n"
            try:
                proc.stdin.write(line.encode())
                await proc.stdin.drain()
                raw = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=timeout_s + 5.0,
                )
            except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
                await self._restart()
                if tool_name in _IDEMPOTENT_TOOLS:
                    continue
                return {"ok": False, "result": "[timeout]", "returncode": 124}

            if not raw:
                # EOF — agent crashed
                await self._restart()
                if tool_name in _IDEMPOTENT_TOOLS:
                    continue
                return {"ok": False, "result": "Error: agent process crashed"}

            # Skip stray non-JSON lines (e.g. Python warnings, sitecustomize output)
            decoded = raw.decode(errors="replace").strip()
            for _skip in range(50):
                if decoded.startswith("{"):
                    break
                logger.debug("Skipping non-JSON agent output: %s", decoded[:120])
                try:
                    raw = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=timeout_s + 5.0,
                    )
                    decoded = raw.decode(errors="replace").strip()
                except (asyncio.TimeoutError, BrokenPipeError):
                    return {"ok": False, "result": "[timeout]", "returncode": 124}
            else:
                return {"ok": False, "result": "Error: agent emitted no JSON response"}

            try:
                return json.loads(decoded)
            except json.JSONDecodeError:
                return {"ok": False, "result": f"Error: invalid agent response: {decoded[:200]}"}

        return {"ok": False, "result": "Error: agent restart failed"}


def _resolve_tool_request(
    tool_name: str | None,
    params: dict[str, Any],
    command_timeout_s: float,
) -> tuple[dict[str, Any] | None, float]:
    """Build a JSON-line request plus the outer response timeout."""

    # Shell commands
    if "command" in params:
        timeout_s = _resolve_exec_timeout_s(params)
        return (
            {"tool": "exec", "args": {"command": params["command"], "timeout": timeout_s}},
            timeout_s,
        )
    if "commands" in params:
        timeout_s = _resolve_exec_timeout_s(params)
        return (
            {
                "tool": "commands",
                "args": {"commands": list(params["commands"]), "timeout": timeout_s},
            },
            timeout_s,
        )

    if tool_name == "exec":
        command = params.get("command")
        commands = params.get("commands")
        if command:
            timeout_s = _resolve_exec_timeout_s(params)
            return (
                {"tool": "exec", "args": {"command": command, "timeout": timeout_s}},
                timeout_s,
            )
        if commands:
            timeout_s = _resolve_exec_timeout_s(params)
            return (
                {
                    "tool": "commands",
                    "args": {"commands": list(commands), "timeout": timeout_s},
                },
                timeout_s,
            )
        return None, command_timeout_s  # missing command/commands

    if tool_name == "read_file":
        return (
            {
                "tool": "read_file",
                "args": {
                    "path": params.get("path", ""),
                    "offset": int(params.get("offset", 1)),
                    "limit": int(params.get("limit", 2000)),
                },
            },
            command_timeout_s,
        )

    if tool_name == "write_file":
        return (
            {
                "tool": "write_file",
                "args": {"path": params.get("path", ""), "content": params.get("content", "")},
            },
            command_timeout_s,
        )

    if tool_name == "edit_file":
        return (
            {"tool": "edit_file", "args": {
                "path": params.get("path", ""),
                "old_text": params.get("old_text", ""),
                "new_text": params.get("new_text", ""),
                "replace_all": bool(params.get("replace_all", False)),
            }},
            command_timeout_s,
        )

    if tool_name == "list_dir":
        return (
            {
                "tool": "list_dir",
                "args": {
                    "path": params.get("path", "."),
                    "recursive": bool(params.get("recursive", False)),
                    "max_entries": int(params.get("max_entries", 0)) or None,
                },
            },
            command_timeout_s,
        )

    # ── Host-mode tools (web, spawn, message) ──────────────────────
    if tool_name == "web_search":
        return {
            "tool": "web_search",
            "args": {
                "query": params.get("query", ""),
                "count": int(params.get("count", 5)),
            },
        }, command_timeout_s

    if tool_name == "web_fetch":
        return {
            "tool": "web_fetch",
            "args": {
                "url": params.get("url", ""),
                "max_length": int(params.get("max_length", 5000)),
            },
        }, command_timeout_s

    if tool_name == "spawn":
        return {
            "tool": "spawn",
            "args": {
                "task": params.get("task", ""),
                "label": params.get("label", "subagent"),
            },
        }, command_timeout_s

    if tool_name == "sessions_yield":
        return {
            "tool": "sessions_yield",
            "args": {
                "message": params.get("message", ""),
            },
        }, command_timeout_s

    if tool_name == "message":
        return {
            "tool": "message",
            "args": {
                "channel": params.get("channel", "cli"),
                "text": params.get("text", ""),
            },
        }, command_timeout_s

    return None, command_timeout_s  # unsupported tool


class HostAgent:
    """Execute tools directly on the host (no container) for cloud_model replay.

    Mirrors :class:`ContainerAgent` execute/start/stop interface so that
    ``_replay_cloud_model_session`` works for host-mode benchmarks.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        web_search_config: Any = None,
        web_proxy: str | None = None,
        exec_timeout: float = 600.0,
    ) -> None:
        from agents.openclaw.tools.filesystem import (
            EditFileTool,
            ListDirTool,
            ReadFileTool,
            WriteFileTool,
        )
        from agents.openclaw.tools.shell import ExecTool
        from agents.openclaw.tools.web import WebFetchTool, WebSearchTool

        self._tools: dict[str, Any] = {
            "exec": ExecTool(
                working_dir=str(workspace),
                timeout=exec_timeout,
            ),
            "commands": ExecTool(
                working_dir=str(workspace),
                timeout=exec_timeout,
            ),
            "read_file": ReadFileTool(workspace=workspace, allowed_dir=workspace),
            "write_file": WriteFileTool(workspace=workspace, allowed_dir=workspace),
            "edit_file": EditFileTool(workspace=workspace, allowed_dir=workspace),
            "list_dir": ListDirTool(workspace=workspace, allowed_dir=workspace),
            "web_search": WebSearchTool(config=web_search_config, proxy=web_proxy),
            "web_fetch": WebFetchTool(proxy=web_proxy),
        }
        self._workspace = workspace

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    @property
    def alive(self) -> bool:
        return True

    async def execute(
        self,
        request: dict[str, Any],
        *,
        timeout_s: float = 600.0,
    ) -> dict[str, Any]:
        """Execute *request* on the host and return ``{ok, result, inner_duration_ms}``."""
        tool_name = request.get("tool", "")
        args: dict[str, Any] = request.get("args", {}) or {}

        tool = self._tools.get(tool_name)
        if tool is None:
            if tool_name in ("spawn", "message", "sessions_yield"):
                return {
                    "ok": True,
                    "result": json.dumps(args, ensure_ascii=False),
                    "inner_duration_ms": 0.0,
                }
            return {"ok": False, "result": f"Error: Unsupported tool {tool_name!r}"}

        try:
            t0 = time.monotonic()
            if tool_name in ("exec", "commands"):
                # ExecTool.execute expects command as kwarg or commands as kwarg
                result: Any
                if tool_name == "commands" and "commands" in args:
                    result = await tool.execute(commands=args["commands"])
                else:
                    result = await tool.execute(
                        command=args.get("command", ""),
                        timeout=args.get("timeout"),
                    )
            elif tool_name == "read_file":
                result = await tool.execute(
                    path=args.get("path", ""),
                    offset=int(args.get("offset", 1)),
                    limit=int(args.get("limit", 2000)),
                )
            elif tool_name == "write_file":
                result = await tool.execute(
                    path=args.get("path", ""),
                    content=args.get("content", ""),
                )
            elif tool_name == "edit_file":
                result = await tool.execute(
                    path=args.get("path", ""),
                    old_text=args.get("old_text", ""),
                    new_text=args.get("new_text", ""),
                    replace_all=bool(args.get("replace_all", False)),
                )
            elif tool_name == "list_dir":
                result = await tool.execute(path=args.get("path", "."))
            elif tool_name == "web_search":
                result = await tool.execute(
                    query=args.get("query", ""),
                    count=int(args.get("count", 5)),
                )
            elif tool_name == "web_fetch":
                result = await tool.execute(
                    url=args.get("url", ""),
                    max_length=int(args.get("max_length", 5000)),
                )
            else:
                result = await tool.execute(**args)
            inner_duration_ms = (time.monotonic() - t0) * 1000.0
            return {
                "ok": True,
                "result": str(result) if result is not None else "",
                "inner_duration_ms": inner_duration_ms,
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {"ok": False, "result": f"Error: {exc}", "returncode": 1}

async def execute_trace_tool(
    *,
    agent: ContainerAgent,
    tool_name: str | None,
    tool_args_json: str,
    command_timeout_s: float,
) -> tuple[str, bool, float | None]:
    """Execute one trace tool call via the persistent in-container agent."""

    resolved_name, params, _nested = _unwrap_tool_args(
        tool_name=tool_name,
        tool_args_json=tool_args_json,
    )

    request, request_timeout_s = _resolve_tool_request(
        resolved_name,
        params,
        command_timeout_s,
    )

    if request is None:
        return f"Error: Unsupported replay tool {resolved_name!r}", False, None

    resp = await agent.execute(request, timeout_s=request_timeout_s)
    result = resp.get("result", "")
    ok = resp.get("ok", False)
    inner_duration_ms = resp.get("inner_duration_ms")

    # Append exit code for exec-style commands
    if request["tool"] in ("exec", "commands"):
        rc = resp.get("returncode", -1)
        result = f"{result}\n\nExit code: {rc}".strip()

    return result, ok, inner_duration_ms
