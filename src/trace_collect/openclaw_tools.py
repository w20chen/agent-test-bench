"""Execute tool calls inside Docker/Podman containers via a persistent agent."""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
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
        return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%}) at line {best_start+1}:\n{diff}"
    return f"Error: old_text not found in {path}. No similar text found."

_MAX_OUTPUT = 10_000

def _truncate_output(text, limit=_MAX_OUTPUT):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n\n... ({len(text) - limit} chars truncated) ...\n\n" + text[-half:]

def handle_exec(args):
    cmd = args.get("command", "")
    timeout = args.get("timeout", 600)
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"}
    try:
        r = subprocess.run(cmd, shell=True, cwd="/testbed",
                           capture_output=True, text=True, timeout=timeout, env=env)
        output = (r.stdout or "") + (r.stderr or "")
        return {"ok": r.returncode == 0, "result": _truncate_output(output), "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "result": "[timeout]", "returncode": 124}

def handle_commands(args):
    cmds = args.get("commands", [])
    timeout = args.get("timeout", 600)
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"}
    all_output = []
    last_rc = 0
    for i, cmd in enumerate(cmds):
        try:
            r = subprocess.run(cmd, shell=True, cwd="/testbed",
                               capture_output=True, text=True, timeout=timeout, env=env)
            all_output.append((r.stdout or "") + (r.stderr or ""))
            last_rc = r.returncode
        except subprocess.TimeoutExpired:
            all_output.append("[timeout]")
            last_rc = 124
    if len(cmds) > 1:
        combined = "\n".join(f"[call {k}]\n{out}" for k, out in enumerate(all_output))
    else:
        combined = all_output[0] if all_output else ""
    return {"ok": last_rc == 0, "result": combined, "returncode": last_rc}

_READ_MAX_CHARS = 128_000
_READ_DEFAULT_LIMIT = 2000

def handle_read_file(args):
    path = args.get("path", "")
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", _READ_DEFAULT_LIMIT))
    try:
        content = open(path).read()
        if not content:
            return {"ok": True, "result": f"(Empty file: {path})"}
        lines = content.splitlines()
        selected = lines[offset:offset + limit]
        numbered = "\n".join(f"{offset + i + 1}| {ln}" for i, ln in enumerate(selected))
        if len(numbered) > _READ_MAX_CHARS:
            numbered = numbered[:_READ_MAX_CHARS] + f"\n\n... (truncated at {_READ_MAX_CHARS} chars)"
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
        return {"ok": True, "result": f"Successfully wrote {path}"}
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
            return {"ok": False, "result": f"Warning: old_text appears {count} times. Provide more context or set replace_all=true."}
        norm_new = new_text.replace("\r\n", "\n")
        new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)
        if uses_crlf:
            new_content = new_content.replace("\n", "\r\n")
        open(path, "wb").write(new_content.encode("utf-8"))
        return {"ok": True, "result": f"Successfully edited {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error editing file: {e}"}

_LIST_IGNORE = {".git", "node_modules", "__pycache__", ".venv", ".tox", ".mypy_cache", ".pytest_cache"}
_LIST_MAX = 200

def handle_list_dir(args):
    path = args.get("path", ".")
    try:
        entries = sorted(e for e in os.listdir(path) if e not in _LIST_IGNORE)
        if len(entries) > _LIST_MAX:
            entries = entries[:_LIST_MAX]
            entries.append(f"... ({len(os.listdir(path)) - _LIST_MAX} more entries)")
        return {"ok": True, "result": "\n".join(entries)}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}

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

    def __init__(self, container_id: str, container_executable: str) -> None:
        self._container_id = container_id
        self._executable = container_executable
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            self._executable, "exec", "-i", "-w", "/testbed",
            self._container_id, "python3", "-u", "-c", _REPLAY_AGENT_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,  # 1MB — agent responses can exceed default 64KB
        )
        logger.info(
            "ContainerAgent started: cid=%s pid=%s",
            self._container_id[:12], self._process.pid,
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
        return {"tool": "read_file", "args": {"path": params.get("path", "")}}, command_timeout_s

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
        return {"tool": "list_dir", "args": {"path": params.get("path", ".")}}, command_timeout_s

    return None, command_timeout_s  # unsupported tool


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
