"""Classify ``exec`` tool calls by the base command being executed.

When an agent runs ``exec`` with a shell command like ``grep -r foo .``,
this module extracts the base command (``grep``) and produces a
categorised tool name such as ``exec-grep``.  Safely normalised unknown
executables keep their identity (for example ``custom_check`` becomes
``exec-custom_check``); ambiguous inputs stay as plain ``exec``.

Strategy
--------
1. Split shell segments while respecting quotes.
2. Parse the executable position and strip wrappers and path prefixes.
3. Look up known commands and select one winner with stable priorities.
4. Return a known category or a strictly validated executable basename.

Adding a new command
--------------------
Edit ``_COMMAND_CATEGORY_MAP`` — no other file changes needed.
"""

from __future__ import annotations

import json
import re
import shlex
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
    # Encoding / decoding
    "base64": "base64",
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
    # Django test runner: ``python -m django test`` is the SWE-bench
    # Django equivalent of pytest (same role, same profiling target).
    "django": "pytest",
    "python3.12": "python",
    "python3.11": "python",
    "python3.10": "python",
    "python3.9": "python",
    # R ecosystem
    "R": "r",
    "Rscript": "r",
    # JVM / distributed data tools
    "scala": "scala",
    "scalac": "scala",
    "sbt": "sbt",
    "spark-submit": "spark",
    "spark-shell": "spark",
    "pyspark": "spark",
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
    # Notebook / interactive
    "jupyter": "jupyter",
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
    # Database
    "sqlite3": "sqlite3",
    "duckdb": "duckdb",
    "psql": "psql",
    "mysql": "mysql",
    "mariadb": "mariadb",
    "mongosh": "mongosh",
    "redis-cli": "redis-cli",
    # Archive
    "tar": "tar",
    "gzip": "tar",
    "gunzip": "tar",
    "zip": "tar",
    "unzip": "tar",
    # Inspection / checksums
    "xxd": "xxd",
    "md5sum": "checksum",
    "sha1sum": "checksum",
    "sha256sum": "checksum",
    "sha512sum": "checksum",
    "file": "file",
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
    "django": 4,  # python -m django test (SWE-bench Django test runner)
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
    "R": 3,
    "Rscript": 3,
    "scala": 3,
    "scalac": 3,
    "sbt": 3,
    "spark-submit": 4,
    "spark-shell": 4,
    "pyspark": 4,
    "jupyter": 3,
    "sqlite3": 3,
    "duckdb": 3,
    "psql": 3,
    "mysql": 3,
    "mariadb": 3,
    "mongosh": 3,
    "redis-cli": 3,
    # Tier 2 — real work (search, transform, file ops, system)
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
    "xxd": 2,
    "md5sum": 2,
    "sha1sum": 2,
    "sha256sum": 2,
    "sha512sum": 2,
    "file": 2,
    "base64": 2,
    # Tier 1 — output post-processing / navigation / trivial / setup
    "xargs": 1,  # plumbing — the command it runs is the real action
    "head": 1,
    "tail": 1,
    "less": 1,
    "more": 1,
    "sort": 1,
    "uniq": 1,
    "wc": 1,
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

# Unknown executables must become bounded identifiers, not raw shell tokens.
_SAFE_EXECUTABLE_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
_MAX_EXECUTABLE_SLUG_LENGTH = 64
_ENV_ASSIGN_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)
_SHELL_RESERVED_WORDS = frozenset({
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do",
    "done", "case", "esac", "in", "function", "select", "coproc",
})

# Wrapper syntax is parsed conservatively. Unknown options leave the wrapper as
# the classified executable rather than treating an option operand as a tool.
_WRAPPER_VALUE_OPTIONS: dict[str, frozenset[str]] = {
    "sudo": frozenset({
        "-C", "--close-from", "-D", "--chdir", "-g", "--group", "-h",
        "--host", "-p", "--prompt", "-R", "--chroot", "-r", "--role",
        "-t", "--type", "-u", "--user",
    }),
    "nice": frozenset({"-n", "--adjustment"}),
    "nohup": frozenset(),
    "ionice": frozenset({"-c", "--class", "-n", "--classdata"}),
    "chroot": frozenset({"--userspec", "--groups"}),
    "flock": frozenset({"-w", "--wait", "-E", "--conflict-exit-code"}),
    "stdbuf": frozenset({
        "-i", "--input", "-o", "--output", "-e", "--error",
    }),
    "timeout": frozenset({"-k", "--kill-after", "-s", "--signal"}),
}
_WRAPPER_FLAG_OPTIONS: dict[str, frozenset[str]] = {
    "sudo": frozenset({
        "-A", "-b", "-E", "-H", "-i", "--login", "-K", "-k", "-n",
        "-P", "-S", "-s", "--shell",
    }),
    "nice": frozenset(),
    "nohup": frozenset(),
    "ionice": frozenset({"-t", "--ignore"}),
    "chroot": frozenset({"--skip-chdir"}),
    "flock": frozenset({
        "-s", "--shared", "-x", "--exclusive", "-u", "--unlock", "-n",
        "--nonblock", "-o", "--close", "-F", "--no-fork", "--verbose",
    }),
    "stdbuf": frozenset(),
    "timeout": frozenset({"--preserve-status", "--foreground", "-v", "--verbose"}),
}
_WRAPPER_NON_EXECUTING_OPTIONS: dict[str, frozenset[str]] = {
    "sudo": frozenset({"-e", "--edit", "-l", "--list", "-V", "--version", "-v", "--validate"}),
    "nice": frozenset({"--help", "--version"}),
    "nohup": frozenset({"--help", "--version"}),
    "ionice": frozenset({"-p", "--pid", "-P", "--pgid", "-u", "--uid", "--help", "--version"}),
    "chroot": frozenset({"--help", "--version"}),
    "flock": frozenset({"-c", "--command", "--help", "-V", "--version"}),
    "stdbuf": frozenset({"--help", "--version"}),
    "timeout": frozenset({"--help", "--version"}),
}
_WRAPPER_POSITIONAL_OPERANDS: dict[str, int] = {
    "chroot": 1,
    "flock": 1,
    "timeout": 1,
}

_XARGS_VALUE_OPTIONS = frozenset({
    "-a", "--arg-file", "-E", "-I", "-L", "-n", "--max-args", "-P",
    "--max-procs", "-s",
    "--max-chars", "-d", "--delimiter",
})
_XARGS_FLAG_OPTIONS = frozenset({
    "-0", "--null", "-o", "--open-tty", "-p", "--interactive", "-r",
    "--no-run-if-empty", "-t", "--verbose", "-x", "--exit",
})
_XARGS_OPTIONAL_ATTACHED_OPTIONS = frozenset({"-e", "-i", "-l"})
_XARGS_OPTIONAL_EQUALS_OPTIONS = frozenset({
    "--eof", "--replace", "--max-lines",
})
_XARGS_NON_EXECUTING_OPTIONS = frozenset({"--help", "--version"})


def _executable_basename(token: str) -> str:
    """Return a POSIX executable basename without accessing the filesystem."""
    return token.rsplit("/", 1)[-1]


def _safe_unknown_category(token: str) -> str | None:
    """Return a bounded category slug for an unknown executable token."""
    basename = _executable_basename(token).lower()
    if basename in _SHELL_RESERVED_WORDS:
        return None
    if not basename or len(basename) > _MAX_EXECUTABLE_SLUG_LENGTH:
        return None
    if _SAFE_EXECUTABLE_RE.fullmatch(basename) is None:
        return None
    return basename


def _consume_options(
    parts: list[str],
    start: int,
    *,
    value_options: frozenset[str],
    flag_options: frozenset[str],
    non_executing_options: frozenset[str],
    optional_attached_options: frozenset[str] = frozenset(),
    optional_equals_options: frozenset[str] = frozenset(),
) -> tuple[int, bool] | None:
    """Consume a supported option prefix and report non-executing modes."""
    idx = start
    non_executing = False
    short_value_options = {option for option in value_options if len(option) == 2}
    short_flag_options = {option for option in flag_options if len(option) == 2}

    while idx < len(parts):
        option = parts[idx]
        if option == "--":
            return idx + 1, non_executing
        if option == "-" or not option.startswith("-"):
            return idx, non_executing

        option_name = option.split("=", 1)[0]
        if option_name in non_executing_options:
            non_executing = True
            idx += 1
            continue
        if option in optional_attached_options or any(
            option.startswith(prefix) and len(option) > len(prefix)
            for prefix in optional_attached_options
        ):
            idx += 1
            continue
        if option in optional_equals_options or any(
            option.startswith(f"{prefix}=") for prefix in optional_equals_options
        ):
            idx += 1
            continue
        if option_name in value_options:
            if "=" in option:
                idx += 1
                continue
            if len(option_name) == 2 and len(option) > 2:
                idx += 1
                continue
            if idx + 1 >= len(parts):
                return None
            idx += 2
            continue
        if option in flag_options:
            idx += 1
            continue
        if len(option) > 2 and option.startswith("-") and not option.startswith("--"):
            if any(option.startswith(prefix) for prefix in short_value_options):
                idx += 1
                continue
            if all(f"-{char}" in short_flag_options for char in option[1:]):
                idx += 1
                continue
        return None

    return idx, non_executing


def _skip_assignments(parts: list[str], start: int) -> int:
    """Skip shell assignment words that precede an executable."""
    idx = start
    while idx < len(parts) and _ENV_ASSIGN_TOKEN_RE.fullmatch(parts[idx]):
        idx += 1
    return idx


def _unwrap_command_wrappers(parts: list[str], start: int) -> int:
    """Return the command index after supported wrappers, conservatively."""
    idx = _skip_assignments(parts, start)
    while idx < len(parts):
        wrapper_idx = idx
        wrapper = _executable_basename(parts[idx])
        if wrapper not in _WRAPPER_VALUE_OPTIONS:
            return idx

        option_start = idx + 1
        if (
            wrapper == "nice"
            and option_start < len(parts)
            and re.fullmatch(r"-\d+", parts[option_start]) is not None
        ):
            option_start += 1

        consumed = _consume_options(
            parts,
            option_start,
            value_options=_WRAPPER_VALUE_OPTIONS[wrapper],
            flag_options=_WRAPPER_FLAG_OPTIONS[wrapper],
            non_executing_options=_WRAPPER_NON_EXECUTING_OPTIONS[wrapper],
        )
        if consumed is None:
            return wrapper_idx
        idx, non_executing = consumed
        if non_executing:
            return wrapper_idx

        positional_count = _WRAPPER_POSITIONAL_OPERANDS.get(wrapper, 0)
        if idx + positional_count >= len(parts):
            return wrapper_idx
        idx += positional_count
        idx = _skip_assignments(parts, idx)

    return idx


def _xargs_command_index(parts: list[str], start: int) -> int | None:
    """Return the xargs command index, or ``None`` when it is ambiguous."""
    consumed = _consume_options(
        parts,
        start,
        value_options=_XARGS_VALUE_OPTIONS,
        flag_options=_XARGS_FLAG_OPTIONS,
        non_executing_options=_XARGS_NON_EXECUTING_OPTIONS,
        optional_attached_options=_XARGS_OPTIONAL_ATTACHED_OPTIONS,
        optional_equals_options=_XARGS_OPTIONAL_EQUALS_OPTIONS,
    )
    if consumed is None:
        return None
    idx, non_executing = consumed
    if non_executing or idx >= len(parts):
        return None
    return idx


def _tokenize_segment(segment: str) -> str:
    """Extract the base command token from a single shell segment.

    A segment is one piece of a command chain (between ``&&``, ``;``, ``|``).
    Returns the normalised command token, or ``""`` if none found.

    The executable is taken from command position rather than by scanning
    arbitrary arguments for familiar command names. A narrow recovery for
    malformed navigation prefixes preserves legacy trace behaviour.
    """
    seg = segment.strip()
    if not seg:
        return ""

    try:
        parts = shlex.split(seg, posix=True)
    except ValueError:
        return ""
    if not parts:
        return ""

    _NAVIGATION_TOKENS = frozenset({
        "cd", "pushd", "popd", "pwd", "ls", "dir",
    })
    token_idx = _unwrap_command_wrappers(parts, 0)
    if token_idx >= len(parts):
        return ""
    token = _executable_basename(parts[token_idx])

    # ``command -v/-V`` queries a name without executing it. Only execution
    # forms (``command NAME``, ``command -p NAME``, ``command -- NAME``) unwrap.
    if token == "command" and len(parts) > 1:
        token_idx += 1
        while token_idx < len(parts) and parts[token_idx].startswith("-"):
            option = parts[token_idx]
            if "v" in option[1:] or "V" in option[1:]:
                return "exec"
            if option not in {"-p", "--"}:
                return "command"
            token_idx += 1
        if token_idx == len(parts):
            return "command"
        token = _executable_basename(parts[token_idx])

    # Preserve the existing recovery for malformed navigation-prefixed traces,
    # while avoiding argument scans for ordinary and unknown executables.
    if token in _NAVIGATION_TOKENS:
        best_action_token = ""
        best_action_idx = -1
        best_action_priority = -1
        for idx, raw_token in enumerate(parts[token_idx + 1:], token_idx + 1):
            candidate = _executable_basename(raw_token)
            priority = _COMMAND_PRIORITY.get(candidate, -1)
            if priority > best_action_priority:
                best_action_token = candidate
                best_action_idx = idx
                best_action_priority = priority
        if best_action_priority >= 3:
            token = best_action_token
            token_idx = best_action_idx

    # ``xargs`` is plumbing; look through to what it runs.
    if token == "xargs" and token_idx >= 0:
        command_idx = _xargs_command_index(parts, token_idx + 1)
        if command_idx is not None:
            token_idx = command_idx
            token = _executable_basename(parts[token_idx])

    # ``python -m <module>`` redirects to a known module category so that
    # ``python -m pytest`` is classified as exec-pytest, not exec-python.
    _PYTHON_INTERPS = frozenset({
        "python", "python3", "python3.9", "python3.10", "python3.11", "python3.12",
    })
    if token in _PYTHON_INTERPS and token_idx >= 0:
        if len(parts) > token_idx + 2 and parts[token_idx + 1] == "-m":
            module_token = parts[token_idx + 2]
            # Only redirect if the module is known — unknown modules stay as python.
            if module_token in _COMMAND_CATEGORY_MAP:
                token = module_token

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

    # Split into segments, respecting shell quoting so that ``|``, ``&&``,
    # and ``;`` inside single/double quotes don't trigger a split.
    segments: list[str] = []
    current = ""
    i = 0
    in_single = False
    in_double = False
    while i < len(command):
        ch = command[i]

        # Track quote state (shell-style, with backslash escape for ")
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            # Only toggle if not escaped
            if i == 0 or command[i - 1] != "\\":
                in_double = not in_double

        # Only split when outside quotes
        if not in_single and not in_double:
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
        A classified tool name like ``"exec-grep"``. Safely normalised
        unknown executables retain their basename; ambiguous commands remain
        unchanged.
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
    if base == "exec":
        return tool_name
    category = _COMMAND_CATEGORY_MAP.get(base)
    if category is None:
        category = _safe_unknown_category(base)
    if category is None:
        return tool_name

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
