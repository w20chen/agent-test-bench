"""Analyze Terminal-Bench tool time shares over a remote SSH dataset.

The script executes a read-only Python analyzer on the remote host and writes a
local JSON summary. The SSH password is read from TB_SSH_PASS.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

import paramiko


REMOTE_ANALYZER = r"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(os.environ["TB_TRACE_ROOT"])
MAX_FILES = int(os.environ.get("TB_MAX_FILES", "0") or "0")

COMMAND_CATEGORY = {
    "python": "python", "python3": "python", "python3.12": "python",
    "python3.11": "python", "python3.10": "python", "python3.9": "python",
    "pip": "pip", "pip3": "pip", "pytest": "pytest", "django": "pytest",
    "R": "r", "Rscript": "r", "git": "git", "curl": "curl", "wget": "curl",
    "apt": "apt", "apt-get": "apt", "apt-cache": "apt", "yum": "apt",
    "dnf": "apt", "apk": "apt", "conda": "conda", "mamba": "conda",
    "npm": "npm", "npx": "npm", "yarn": "npm", "pnpm": "npm",
    "node": "node", "make": "make", "cmake": "make", "ninja": "make",
    "gcc": "gcc", "g++": "gcc", "clang": "gcc", "clang++": "gcc",
    "docker": "docker", "podman": "docker", "bash": "bash", "sh": "bash",
    "zsh": "bash", "tail": "tail", "head": "head", "cat": "cat",
    "grep": "grep", "egrep": "grep", "fgrep": "grep", "rg": "grep",
    "find": "find", "fd": "find", "sed": "sed", "awk": "awk",
    "tar": "tar", "gzip": "tar", "gunzip": "tar", "zip": "tar",
    "unzip": "tar", "7z": "tar", "sqlite3": "sqlite3", "duckdb": "duckdb",
    "psql": "psql", "mysql": "mysql", "mariadb": "mariadb",
    "systemctl": "systemctl", "service": "systemctl", "aria2c": "aria2c",
    "yt-dlp": "yt-dlp", "mlflow": "mlflow", "lake": "lake",
    "setfacl": "acl", "getfacl": "acl", "stat": "stat",
}
COMMAND_PRIORITY = {
    "pip": 4, "pip3": 4, "pytest": 4, "django": 4, "spark-submit": 4,
    "python": 3, "python3": 3, "python3.12": 3, "python3.11": 3,
    "python3.10": 3, "python3.9": 3, "git": 3, "docker": 3, "podman": 3,
    "make": 3, "cmake": 3, "ninja": 3, "gcc": 3, "g++": 3, "clang": 3,
    "clang++": 3, "apt": 3, "apt-get": 3, "yum": 3, "dnf": 3, "apk": 3,
    "conda": 3, "mamba": 3, "npm": 3, "npx": 3, "node": 3, "curl": 3,
    "wget": 3, "R": 3, "Rscript": 3, "systemctl": 3, "service": 3,
}
SAFE_EXEC_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}$")
ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)


def pct(values, q):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def iter_jsonl(path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def is_trace_file(path):
    name = path.name.lower()
    if path.suffix.lower() != ".jsonl":
        return False
    return "trace" in name or name in {"actions.jsonl", "events.jsonl"}


def attempt_key(path):
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return str(path)
    parts = rel.parts
    for idx, part in enumerate(parts):
        if re.fullmatch(r"attempt_\d+", part):
            return "/".join(parts[: idx + 1])
    return str(rel)


def trace_priority(path):
    name = path.name.lower()
    parts = [part.lower() for part in path.parts]
    if name == "trace.jsonl":
        return 100
    if "agent-logs" in parts and name == "openclaw-trace.jsonl":
        return 90
    if name.endswith("-terminal-bench-trace.jsonl"):
        return 80
    return 10


def action_kind(row):
    return row.get("action_type") or row.get("type") or ""


def action_data(row):
    data = row.get("data")
    return data if isinstance(data, dict) else {}


def tool_name(row):
    data = action_data(row)
    name = data.get("tool_name") or data.get("tool") or row.get("tool_name")
    if name:
        name = str(name)
        if name == "exec":
            classified = classify_exec(data.get("tool_args"))
            if classified:
                return classified
        return name
    return "unknown"


def extract_command(tool_args):
    parsed = tool_args
    if isinstance(tool_args, str):
        try:
            parsed = json.loads(tool_args)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    command = parsed.get("command")
    if isinstance(command, str) and command:
        return command
    nested = parsed.get("exec")
    if isinstance(nested, dict) and isinstance(nested.get("command"), str):
        return nested["command"]
    return None


def split_segments(command):
    segments = []
    current = []
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single and (i == 0 or command[i - 1] != "\\"):
            in_double = not in_double
        if not in_single and not in_double:
            if ch == "|" or ch == ";":
                segments.append("".join(current))
                current = []
                i += 1
                continue
            if ch == "&" and i + 1 < len(command) and command[i + 1] == "&":
                segments.append("".join(current))
                current = []
                i += 2
                continue
        current.append(ch)
        i += 1
    segments.append("".join(current))
    return segments


def strip_leading_comments(segment):
    lines = []
    for line in segment.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def segment_base(segment):
    segment = strip_leading_comments(segment)
    if not segment.strip():
        return None
    try:
        parts = shlex.split(segment, posix=True)
    except ValueError:
        return None
    idx = 0
    while idx < len(parts) and ENV_ASSIGN_RE.fullmatch(parts[idx]):
        idx += 1
    while idx < len(parts) and parts[idx] in {"sudo", "time", "timeout", "nohup"}:
        wrapper = parts[idx]
        idx += 1
        while idx < len(parts) and parts[idx].startswith("-"):
            option = parts[idx]
            idx += 1
            if idx < len(parts) and option in {"-u", "-g", "-k", "-s"}:
                idx += 1
        if wrapper == "timeout" and idx < len(parts):
            idx += 1
    if idx >= len(parts):
        return None
    base = parts[idx].rsplit("/", 1)[-1]
    if base in {"python", "python3", "python3.12", "python3.11", "python3.10", "python3.9"}:
        if idx + 2 < len(parts) and parts[idx + 1] == "-m":
            module = parts[idx + 2]
            if module in COMMAND_CATEGORY:
                return module
    return base


def classify_exec(tool_args):
    command = extract_command(tool_args)
    if not command:
        return None
    best = None
    best_priority = -1
    for segment in split_segments(command):
        base = segment_base(segment)
        if not base:
            continue
        priority = COMMAND_PRIORITY.get(base, 1)
        if priority >= best_priority:
            best = base
            best_priority = priority
    if not best:
        return None
    category = COMMAND_CATEGORY.get(best)
    if category is None:
        lowered = best.lower()
        if SAFE_EXEC_RE.fullmatch(lowered):
            category = lowered
    if not category:
        return None
    return "exec-" + category


def duration_s(row):
    data = action_data(row)
    for obj in (data, row):
        value = obj.get("duration_ms")
        if value is not None:
            try:
                return max(0.0, float(value) / 1000.0)
            except (TypeError, ValueError):
                pass
    start = row.get("ts_start")
    end = row.get("ts_end")
    try:
        if start is not None and end is not None:
            return max(0.0, float(end) - float(start))
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def timestamp_bounds(rows):
    starts = []
    ends = []
    for row in rows:
        try:
            if row.get("ts_start") is not None:
                starts.append(float(row["ts_start"]))
            if row.get("ts_end") is not None:
                ends.append(float(row["ts_end"]))
        except (TypeError, ValueError):
            continue
    if starts and ends and max(ends) >= min(starts):
        return min(starts), max(ends)
    return None, None


def looks_like_tool(row):
    kind = action_kind(row)
    if kind == "tool_exec":
        return True
    return row.get("type") == "action" and bool(action_data(row).get("tool_name"))


trace_files = []
for path in ROOT.rglob("*"):
    if path.is_file() and is_trace_file(path):
        trace_files.append(path)
trace_files.sort()
if MAX_FILES > 0:
    trace_files = trace_files[:MAX_FILES]
candidate_trace_file_count = len(trace_files)
selected_by_attempt = {}
for path in trace_files:
    key = attempt_key(path)
    current = selected_by_attempt.get(key)
    if current is None or (trace_priority(path), -len(str(path))) > (
        trace_priority(current),
        -len(str(current)),
    ):
        selected_by_attempt[key] = path
trace_files = sorted(selected_by_attempt.values())

tool_totals = Counter()
tool_counts = Counter()
tool_attempts = Counter()
tool_call_durations = defaultdict(list)
tool_attempt_shares = defaultdict(list)
attempts = []
skipped = []
unclassified_exec = {
    "calls": 0,
    "total_s": 0.0,
    "missing_command_calls": 0,
    "examples": [],
}

for path in trace_files:
    rows = list(iter_jsonl(path))
    tool_rows = [row for row in rows if looks_like_tool(row)]
    if not tool_rows:
        skipped.append(str(path))
        continue
    start, end = timestamp_bounds(rows)
    span_s = (end - start) if start is not None and end is not None else None
    per_tool = Counter()
    for row in tool_rows:
        data = action_data(row)
        name = tool_name(row)
        dur = duration_s(row)
        if dur <= 0:
            continue
        raw_name = data.get("tool_name") or data.get("tool") or row.get("tool_name")
        if str(raw_name) == "exec" and name == "exec":
            command = extract_command(data.get("tool_args"))
            unclassified_exec["calls"] += 1
            unclassified_exec["total_s"] += dur
            if not command:
                unclassified_exec["missing_command_calls"] += 1
            if len(unclassified_exec["examples"]) < 80:
                unclassified_exec["examples"].append(
                    {
                        "duration_s": dur,
                        "path": str(path),
                        "command": command[:500] if command else None,
                        "tool_args_type": type(data.get("tool_args")).__name__,
                        "row_keys": sorted(row.keys()),
                        "data_keys": sorted(data.keys()),
                        "data_preview": {
                            key: str(value)[:200]
                            for key, value in data.items()
                            if key in {"tool_name", "tool_args", "command", "args", "input"}
                        },
                    }
                )
        per_tool[name] += dur
        tool_totals[name] += dur
        tool_counts[name] += 1
        tool_call_durations[name].append(dur)
    if not per_tool:
        continue
    tool_sum = sum(per_tool.values())
    e2e_s = span_s if span_s and span_s > 0 else tool_sum
    for name, dur in per_tool.items():
        tool_attempts[name] += 1
        tool_attempt_shares[name].append(dur / e2e_s if e2e_s > 0 else 0.0)
    attempts.append(
        {
            "path": str(path),
            "e2e_s": e2e_s,
            "trace_span_s": span_s,
            "tool_time_s": tool_sum,
            "tool_share": tool_sum / e2e_s if e2e_s > 0 else 0.0,
            "tool_count": sum(1 for row in tool_rows if duration_s(row) > 0),
        }
    )

total_e2e = sum(item["e2e_s"] for item in attempts)
total_tool = sum(item["tool_time_s"] for item in attempts)

tools = []
for name, total_s in tool_totals.most_common():
    durs = tool_call_durations[name]
    shares = tool_attempt_shares[name]
    tools.append(
        {
            "tool": name,
            "total_s": total_s,
            "share_of_total_e2e": total_s / total_e2e if total_e2e > 0 else 0.0,
            "share_of_total_tool": total_s / total_tool if total_tool > 0 else 0.0,
            "calls": tool_counts[name],
            "attempts": tool_attempts[name],
            "median_call_s": median(durs) if durs else None,
            "p90_call_s": pct(durs, 0.90),
            "p95_call_s": pct(durs, 0.95),
            "median_attempt_share": median(shares) if shares else None,
            "p90_attempt_share": pct(shares, 0.90),
        }
    )

print(
    json.dumps(
        {
            "root": str(ROOT),
            "candidate_trace_files": candidate_trace_file_count,
            "trace_files_considered": len(trace_files),
            "attempts_with_tools": len(attempts),
            "skipped_trace_files_without_tools": len(skipped),
            "total_e2e_s": total_e2e,
            "total_tool_s": total_tool,
            "overall_tool_share": total_tool / total_e2e if total_e2e > 0 else 0.0,
            "tools": tools,
            "attempts": attempts,
            "unclassified_exec": unclassified_exec,
        },
        ensure_ascii=False,
    )
)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="202.120.39.13")
    parser.add_argument("--port", type=int, default=17722)
    parser.add_argument("--user", default="weitian")
    parser.add_argument(
        "--root",
        default="/data/share/datasets/agent_datasets/terminal-bench-p2",
    )
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/terminal_bench_p2_tool_times.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.environ.get("TB_SSH_PASS")
    if not password:
        raise SystemExit("TB_SSH_PASS is required")

    encoded = base64.b64encode(REMOTE_ANALYZER.encode("utf-8")).decode("ascii")
    remote_cmd = (
        "TB_TRACE_ROOT="
        + json.dumps(args.root)
        + " TB_MAX_FILES="
        + json.dumps(str(args.max_files))
        + " python3 -c "
        + json.dumps(
            "import base64; exec(base64.b64decode("
            + repr(encoded)
            + ").decode('utf-8'))"
        )
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    try:
        _, stdout, stderr = client.exec_command(remote_cmd, timeout=600)
        stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
    finally:
        client.close()

    if exit_code != 0:
        raise SystemExit(f"remote analyzer failed ({exit_code}):\n{stderr_text}")

    data: dict[str, Any] = json.loads(stdout_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {args.output}")
    print(
        "attempts={attempts_with_tools} trace_files={trace_files_considered} "
        "total_e2e_h={e2e:.2f} tool_share={share:.1%}".format(
            attempts_with_tools=data["attempts_with_tools"],
            trace_files_considered=data["trace_files_considered"],
            e2e=data["total_e2e_s"] / 3600.0,
            share=data["overall_tool_share"],
        )
    )
    print("top tools by total end-to-end share:")
    for row in data["tools"][:20]:
        print(
            "{tool:24s} total_h={total_h:8.2f} e2e_share={e2e_share:6.2%} "
            "calls={calls:7d} attempts={attempts:5d} med={med:7.2f}s p95={p95:7.2f}s "
            "med_attempt_share={med_share:6.2%}".format(
                tool=row["tool"][:24],
                total_h=row["total_s"] / 3600.0,
                e2e_share=row["share_of_total_e2e"],
                calls=row["calls"],
                attempts=row["attempts"],
                med=row["median_call_s"] or 0.0,
                p95=row["p95_call_s"] or 0.0,
                med_share=row["median_attempt_share"] or 0.0,
            )
        )


if __name__ == "__main__":
    main()
