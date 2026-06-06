"""Classify ``exec`` tool calls by the base command being executed.

When an agent runs ``exec`` with a shell command like ``grep -r foo .``,
this module extracts the base command (``grep``) and produces a
categorised tool name such as ``exec-grep``.  Unmatched commands stay as
plain ``exec`` so the trace retains the original label rather than
losing information.

Strategy
--------
1. Parse the first whitespace-delimited token from the command string.
2. Strip common prefixes (env-var assignments, ``sudo``, path prefixes).
3. Look up the token in a static mapping of known commands.
4. Return ``exec-<category>`` on match, ``"exec"`` otherwise.

Adding a new command
--------------------
Edit ``_COMMAND_CATEGORY_MAP`` — no other file changes needed.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Map base-command → category slug.
# When adding entries prefer the most common invocation (e.g. ``python3``
# is mapped, not ``python``, because ``python3`` is the typical venv
# entry point on modern Linux; the normalisation step handles ``python``
# -> ``python3`` automatically).
_COMMAND_CATEGORY_MAP: dict[str, str] = {
    # Search / inspection
    "grep": "grep",
    "egrep": "grep",
    "fgrep": "grep",
    "rg": "grep",
    "find": "find",
    "fd": "find",
    "locate": "find",
    "which": "which",
    "whereis": "which",
    "type": "which",
    # File content
    "cat": "cat",
    "head": "head",
    "tail": "tail",
    "less": "less",
    "more": "less",
    # File listing / navigation
    "ls": "ls",
    "dir": "ls",
    "cd": "cd",
    "pushd": "cd",
    "popd": "cd",
    "pwd": "pwd",
    # File ops
    "mkdir": "mkdir",
    "cp": "cp",
    "mv": "mv",
    "rm": "rm",
    "rmdir": "rm",
    "chmod": "chmod",
    "chown": "chmod",
    "touch": "touch",
    "ln": "ln",
    # Text processing
    "sed": "sed",
    "awk": "awk",
    "sort": "sort",
    "uniq": "uniq",
    "wc": "wc",
    "tr": "tr",
    "cut": "cut",
    "tee": "tee",
    "diff": "diff",
    "patch": "diff",
    "xargs": "xargs",
    # Shell builtins / scripting
    "echo": "echo",
    "printf": "echo",
    "source": "source",
    "export": "export",
    "env": "env",
    "unset": "env",
    "set": "env",
    # Python ecosystem
    "python": "python",
    "python3": "python",
    "pip": "pip",
    "pip3": "pip",
    "pytest": "pytest",
    "python3.12": "python",
    "python3.11": "python",
    "python3.10": "python",
    "python3.9": "python",
    # Node / JS
    "node": "node",
    "npm": "npm",
    "npx": "npm",
    "yarn": "npm",
    "pnpm": "npm",
    # Version control
    "git": "git",
    # Network
    "curl": "curl",
    "wget": "curl",
    # Package managers
    "apt": "apt",
    "apt-get": "apt",
    "apt-cache": "apt",
    "yum": "apt",
    "dnf": "apt",
    "apk": "apt",
    "brew": "apt",
    "conda": "conda",
    "mamba": "conda",
    # Container
    "docker": "docker",
    "podman": "docker",
    # System
    "systemctl": "systemctl",
    "service": "systemctl",
    "ps": "ps",
    "kill": "kill",
    "killall": "kill",
    "top": "top",
    "htop": "top",
    "df": "df",
    "du": "df",
    "free": "free",
    "mount": "mount",
    "umount": "mount",
    # Make / build
    "make": "make",
    "cmake": "make",
    "ninja": "make",
    "gcc": "gcc",
    "g++": "gcc",
    "clang": "gcc",
    "clang++": "gcc",
    # Archive
    "tar": "tar",
    "gzip": "tar",
    "gunzip": "tar",
    "zip": "tar",
    "unzip": "tar",
    # Misc
    "true": "true",
    "false": "true",
    "test": "test",
    "sleep": "sleep",
    "date": "date",
    "time": "time",
    "watch": "watch",
    "man": "man",
    "info": "man",
    "su": "su",
    "sudo": "su",
    "bash": "bash",
    "sh": "bash",
    "zsh": "bash",
}

# Priority tiers for disambiguating compound commands (e.g.
# ``cd /x && python3 setup.py build`` → ``python``, not ``cd``).
# Higher number = more likely to be the "main action".  Commands not
# listed here default to priority 1 (neutral).
_COMMAND_PRIORITY: dict[str, int] = {
    # Tier 3+ — primary actions (build, test, install, deploy)
    "pip": 4,
    "pip3": 4,
    "pytest": 4,
    "python": 3,
    "python3": 3,
    "python3.12": 3,
    "python3.11": 3,
    "python3.10": 3,
    "python3.9": 3,
    "git": 3,
    "docker": 3,
    "podman": 3,
    "make": 3,
    "cmake": 3,
    "ninja": 3,
    "gcc": 3,
    "g++": 3,
    "clang": 3,
    "clang++": 3,
    "apt": 3,
    "apt-get": 3,
    "yum": 3,
    "dnf": 3,
    "apk": 3,
    "brew": 3,
    "conda": 3,
    "mamba": 3,
    "npm": 3,
    "npx": 3,
    "yarn": 3,
    "pnpm": 3,
    "node": 3,
    "systemctl": 3,
    "service": 3,
    "curl": 3,
    "wget": 3,
    "su": 3,
    "sudo": 3,
    # Tier 2 — inspection / search / processing
    "grep": 2,
    "egrep": 2,
    "fgrep": 2,
    "rg": 2,
    "find": 2,
    "fd": 2,
    "sed": 2,
    "awk": 2,
    "diff": 2,
    "patch": 2,
    "cat": 2,
    "head": 2,
    "tail": 2,
    "less": 2,
    "more": 2,
    "sort": 2,
    "uniq": 2,
    "wc": 2,
    "tar": 2,
    "gzip": 2,
    "gunzip": 2,
    "zip": 2,
    "unzip": 2,
    "chmod": 2,
    "chown": 2,
    "cp": 2,
    "mv": 2,
    "rm": 2,
    "rmdir": 2,
    "mkdir": 2,
    "touch": 2,
    "ln": 2,
    "kill": 2,
    "killall": 2,
    "mount": 2,
    "umount": 2,
    "ps": 2,
    "top": 2,
    "htop": 2,
    "df": 2,
    "du": 2,
    "free": 2,
    "which": 2,
    "whereis": 2,
    "man": 2,
    "watch": 2,
    "xargs": 1,  # plumbing — the command it runs is the real action
    # Tier 1 — navigation / trivial / setup
    "cd": 1,
    "pushd": 1,
    "popd": 1,
    "ls": 1,
    "dir": 1,
    "pwd": 1,
    "echo": 1,
    "printf": 1,
    "true": 1,
    "false": 1,
    "test": 1,
    "sleep": 1,
    "date": 1,
    "time": 1,
    "source": 1,
    "export": 1,
    "env": 1,
    "unset": 1,
    "set": 1,
    "bash": 1,
    "sh": 1,
    "zsh": 1,
    "tee": 1,
    "cut": 1,
    "tr": 1,
    "locate": 1,
    "type": 1,
    "info": 1,
}

# Regex to strip leading env-var assignments (VAR=val VAR2=val2 ...).
_ENV_ASSIGN_RE = re.compile(r"^(?:\s*[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S+)\s*)+")

# Regex to strip leading `sudo` / `nice` / `nohup` / etc.
_PREAMBLE_RE = re.compile(r"^\s*(?:sudo|nice|nohup|ionice|chroot|flock|stdbuf|timeout)\s+")

# Common path prefix to strip, e.g. /usr/bin/grep → grep
_PATH_PREFIX_RE = re.compile(r"^/(?:usr/)?(?:local/)?(?:s?)bin/")


def _tokenize_segment(segment: str) -> str:
    """Extract the base command token from a single shell segment.

    A segment is one piece of a command chain (between ``&&``, ``;``, ``|``).
    Returns the normalised command token, or ``""`` if none found.
    """
    seg = segment.strip()
    if not seg:
        return ""

    # Strip env-var assignments
    seg = _ENV_ASSIGN_RE.sub("", seg).strip()

    # Strip preamble (sudo, nice, ...)
    seg = _PREAMBLE_RE.sub("", seg).strip()

    if not seg:
        return ""

    # Extract first token
    parts = seg.split()
    token = parts[0]

    # Strip path prefix: /usr/bin/grep → grep
    token = _PATH_PREFIX_RE.sub("", token)

    # Handle `command` builtin: `command grep` → `grep`
    if token == "command" and len(parts) > 1:
        token = parts[1]

    # ``xargs`` is plumbing — look through to what it runs
    if token == "xargs" and len(parts) > 1:
        # Skip xargs flags (those starting with -) to find the real command
        for p in parts[1:]:
            if not p.startswith("-"):
                token = p
                break

    return token


def _extract_base_command(command: str) -> str:
    """Extract the most meaningful command token from a shell command.

    Splits the command into segments on ``&&``, ``;``, and ``|``, then
    picks the segment with the highest priority base command.
    For example ``cd /x && python3 setup.py build`` → ``python3``
    (priority 3) rather than ``cd`` (priority 1).
    """
    if not command or not isinstance(command, str):
        return "exec"

    # Split into segments
    segments: list[str] = []
    current = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "|" and (i == 0 or command[i - 1] != "\\"):
            segments.append(current)
            current = ""
            i += 1
            continue
        if ch == "&" and i + 1 < len(command) and command[i + 1] == "&":
            segments.append(current)
            current = ""
            i += 2
            continue
        if ch == ";":
            segments.append(current)
            current = ""
            i += 1
            continue
        current += ch
        i += 1
    segments.append(current)

    # Extract token + priority for each segment
    best_token = "exec"
    best_priority = -1
    for seg in segments:
        token = _tokenize_segment(seg)
        if not token:
            continue
        prio = _COMMAND_PRIORITY.get(token, 1)  # default neutral
        # >= so that when priorities tie, the rightmost segment wins
        # (e.g. ``find | xargs grep`` → ``grep``, the final consumer)
        if prio >= best_priority:
            best_priority = prio
            best_token = token

    return best_token


def classify_exec_tool_name(
    tool_name: str,
    tool_args: str | dict[str, Any] | None,
) -> str:
    """Classify an exec tool call by its base command.

    Args:
        tool_name: The original tool name (e.g. ``"exec"``).
        tool_args: The tool arguments — either a JSON string or a dict
                   containing a ``"command"`` key.

    Returns:
        A classified tool name like ``"exec-grep"``, or ``tool_name``
        unchanged if the call is not an exec or the command is unrecognised.
    """
    if tool_name != "exec":
        return tool_name

    # Parse tool_args to extract the command string
    command: str | None = None

    if isinstance(tool_args, str):
        try:
            parsed = json.loads(tool_args)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            # OpenClaw-style: {"command": "grep ..."}
            # Claude Code style: {"exec": {"command": "..."}}
            command = parsed.get("command") or ""
            if not command and "exec" in parsed and isinstance(parsed["exec"], dict):
                command = parsed["exec"].get("command", "")
    elif isinstance(tool_args, dict):
        command = tool_args.get("command", "")
        if not command and "exec" in tool_args and isinstance(tool_args["exec"], dict):
            command = tool_args["exec"].get("command", "")

    if not command:
        return tool_name  # Can't classify without a command

    base = _extract_base_command(command)
    category = _COMMAND_CATEGORY_MAP.get(base)
    if category is None:
        return tool_name  # Unrecognised → keep as plain "exec"

    return f"exec-{category}"


# Convenience: apply classification to a tool data dict in-place.
def classify_tool_data(data: dict[str, Any]) -> dict[str, Any]:
    """Classify the tool_name in a tool-exec data dict and return it."""
    tool_name = data.get("tool_name", "")
    if tool_name != "exec":
        return data
    tool_args = data.get("tool_args", "")
    classified = classify_exec_tool_name(tool_name, tool_args)
    if classified != tool_name:
        data = dict(data)  # shallow copy to avoid mutating caller's dict
        data["tool_name"] = classified
    return data


# ---------------------------------------------------------------------------
# Trace-level post-processing
# ---------------------------------------------------------------------------

def rewrite_trace_with_exec_classification(trace_path: Path) -> int:
    """Read a canonical trace JSONL, classify ``exec`` tool names, write back.

    This is the primary integration point for the collection pipeline:
    called after the scaffold has finished writing the trace file, before
    downstream consumers (visualisers, simulators, inspectors) read it.

    Handles:
    * ``tool_exec`` action records — classifies ``tool_name`` and updates
      ``action_id`` to match.
    * ``summary`` records — rebuilds ``tool_ms_by_name`` and
      ``tool_timeouts`` from the classified actions so pie charts and
      stats reflect the split categories.

    Returns the number of records whose tool_name was changed.
    """
    import json as _json
    from pathlib import Path as _Path

    trace_path = _Path(trace_path)
    if not trace_path.exists():
        return 0

    raw = trace_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    changed = 0
    rewritten: list[str] = []
    # Accumulate classified tool stats so we can rebuild the summary.
    tool_ms_by_name: dict[str, float] = {}
    tool_timeouts: dict[str, int] = {}
    has_summary = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            rewritten.append(line)
            continue

        try:
            rec = _json.loads(stripped)
        except _json.JSONDecodeError:
            rewritten.append(line)
            continue

        if rec.get("type") == "action" and rec.get("action_type") == "tool_exec":
            data = rec.get("data")
            if isinstance(data, dict):
                tool_name = data.get("tool_name", "")
                tool_args = data.get("tool_args", "")
                classified = classify_exec_tool_name(tool_name, tool_args)
                duration = data.get("duration_ms", 0.0) or 0.0
                success = data.get("success", True)
                tool_ms_by_name[classified] = (
                    tool_ms_by_name.get(classified, 0.0) + duration
                )
                if not success:
                    tool_timeouts[classified] = (
                        tool_timeouts.get(classified, 0) + 1
                    )
                if classified != tool_name:
                    data["tool_name"] = classified
                    old_action_id = rec.get("action_id", "")
                    if old_action_id and tool_name in old_action_id:
                        rec["action_id"] = old_action_id.replace(
                            tool_name, classified, 1
                        )
                    changed += 1
            rewritten.append(_json.dumps(rec, ensure_ascii=False))
        elif rec.get("type") == "summary":
            has_summary = True
            # Rebuild summary keys from classified actions.
            if tool_ms_by_name:
                rec["tool_ms_by_name"] = tool_ms_by_name
            if tool_timeouts:
                rec["tool_timeouts"] = tool_timeouts
            rewritten.append(_json.dumps(rec, ensure_ascii=False))
        else:
            rewritten.append(line)

    if changed or (has_summary and tool_ms_by_name):
        tmp = trace_path.with_suffix(trace_path.suffix + ".tmp")
        tmp.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        tmp.replace(trace_path)

    return changed


# Re-export Path for the convenience of callers that import just this module.
from pathlib import Path  # noqa: E402
