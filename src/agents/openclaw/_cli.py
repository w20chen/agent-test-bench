"""Standalone OpenClaw CLI.

This module keeps argument parsing and daemon entrypoints in one place while
reusing the same SessionRunner path as evaluation runs.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_call import UnifiedProvider, add_llm_config_arguments, resolve_llm_config
from llm_call.config import (
    nonnegative_float_arg,
    positive_float_arg,
    positive_int_arg,
    top_p_arg,
)


def _ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


class CLIStreamHook:
    """Print real-time agent events to stderr."""

    def __init__(self, session_id: str, *, quiet: bool = False) -> None:
        self.sid_short = session_id[:10]
        self.quiet = quiet
        self._iter_start = 0.0
        self._n_iterations = 0
        self._total_tokens = 0
        self._wall_start = time.monotonic()

    def _log(self, event_type: str, detail: str) -> None:
        if self.quiet:
            return
        print(
            f"[{self.sid_short}] {_ts()} {event_type:<10} {detail}",
            file=sys.stderr,
            flush=True,
        )

    def wants_streaming(self) -> bool:
        return not self.quiet

    async def before_iteration(self, context: Any) -> None:
        self._iter_start = time.monotonic()

    async def before_execute_tools(self, context: Any) -> None:
        if self.quiet or not context.tool_calls:
            return
        for tc in context.tool_calls:
            args_preview = json.dumps(tc.arguments, ensure_ascii=False)[:80]
            self._log("TOOL", f"{tc.name}({args_preview})")

    async def after_iteration(self, context: Any) -> None:
        self._n_iterations += 1
        usage = context.usage or {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        self._total_tokens += pt + ct
        lat_ms = (time.monotonic() - self._iter_start) * 1000 if self._iter_start else 0
        self._log(
            "ITER",
            f"step {self._n_iterations}, tokens: {pt}+{ct}, llm: {lat_ms:.0f}ms",
        )

    async def on_stream(self, context: Any, delta: str) -> None:
        pass

    async def on_stream_end(self, context: Any, *, resuming: bool) -> None:
        pass

    def finalize_content(self, context: Any, content: str | None) -> str | None:
        return content

    def print_summary(self) -> None:
        elapsed = time.monotonic() - self._wall_start
        self._log(
            "DONE",
            f"completed in {elapsed:.1f}s, {self._n_iterations} steps, "
            f"{self._total_tokens} total tokens",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agents.openclaw",
        description="Run the OpenClaw agent on an arbitrary task.",
    )

    parser.add_argument("--prompt", help="Task prompt for the agent.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Query status of an async session (requires --session-id).",
    )

    add_llm_config_arguments(parser)

    parser.add_argument(
        "--workspace",
        default=".",
        help="Working directory for the agent (default: cwd).",
    )
    parser.add_argument(
        "--trace-output",
        default=None,
        help=(
            "Output path for the trace JSONL file. Default: "
            "<repo>/traces/openclaw_cli/<model_slug>/<UTC_TS>/<session_id>.jsonl "
            "(relative to the agent-sched-bench repo root, NOT the workspace). "
            "Set this to override (e.g., for one-off experiments)."
        ),
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help="Optional MCP config YAML passed through to the OpenClaw session runner.",
    )

    parser.add_argument(
        "--async",
        dest="run_async",
        action="store_true",
        help="Run in background, print session ID and exit.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Resume/append to an existing session.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max agent iterations (default: 100).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens per LLM call (default: 4096).",
    )
    parser.add_argument(
        "--temperature",
        type=nonnegative_float_arg,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
    )
    parser.add_argument(
        "--top-p",
        type=top_p_arg,
        default=None,
        help="Optional nucleus sampling top_p value.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int_arg,
        default=None,
        help="Optional top_k sampling value for compatible providers.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=positive_float_arg,
        default=None,
        help="Optional repetition penalty for compatible providers.",
    )
    parser.add_argument(
        "--malformed-retry-budget",
        type=int,
        default=3,
        help="Max malformed tool-call retries per logical step before giving up (default: 3).",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress event stream, only print final output.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output final result as JSON to stdout.",
    )

    parser.add_argument("--_daemon", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_pid-file", default=None, help=argparse.SUPPRESS)

    return parser


def _resolve_llm_config(args: argparse.Namespace):
    try:
        config = resolve_llm_config(
            provider=args.provider,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            environ=os.environ,
        )
        if not config.api_key:
            print(
                f"ERROR: Set {config.env_key} or pass --api-key.",
                file=sys.stderr,
            )
            sys.exit(1)
        return config
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


def _resolve_repo_root() -> Path | None:
    """Walk up from this file to find the agent-sched-bench repo root.

    Returns ``None`` if no ``pyproject.toml`` is found above this file —
    callers should fall back to ``Path.cwd()``.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _slug_model(name: str) -> str:
    """Convert a model identifier to a filesystem-safe slug.

    e.g., ``z-ai/glm-5.1`` -> ``z-ai_glm-5.1``.
    """
    return name.replace("/", "_").replace(":", "_")


def _resolve_trace_output(
    args: argparse.Namespace, session_id: str, model: str
) -> Path:
    """Resolve the trace output path for the current CLI invocation.

    ``--trace-output`` wins; otherwise prefer the repo-root trace tree and
    fall back to the current working directory when the repo root is unknown.
    """
    if args.trace_output:
        return Path(args.trace_output).expanduser().resolve()

    base = _resolve_repo_root() or Path.cwd().resolve()
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        base
        / "traces"
        / "openclaw_cli"
        / _slug_model(model)
        / ts
        / f"{session_id}.jsonl"
    )


def _run_sync(args: argparse.Namespace) -> int:
    llm_config = _resolve_llm_config(args)

    from agents.openclaw._session_runner import SessionRunner
    from trace_collect.collector import load_mcp_servers

    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    session_id = args.session_id or f"oc-{uuid.uuid4().hex[:8]}"
    session_key = (
        f"cli:{session_id}" if not session_id.startswith("cli:") else session_id
    )

    is_daemon = getattr(args, "_daemon", False)
    cli_hook = CLIStreamHook(session_id, quiet=is_daemon or args.quiet)

    if not is_daemon:
        print(
            f"Session: {session_key} | Provider: {llm_config.name} | "
            f"Model: {llm_config.model} | Workspace: {workspace}",
            file=sys.stderr,
        )

    provider = UnifiedProvider(
        api_key=llm_config.api_key,
        api_base=llm_config.api_base,
        default_model=llm_config.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    trace_file = _resolve_trace_output(args, session_id, llm_config.model)
    # mkdir deferred to SessionRunner — avoids leaving empty dirs on early exit
    if not is_daemon:
        print(f"Trace: {trace_file}", file=sys.stderr)

    runner = SessionRunner(
        provider,
        model=llm_config.model,
        max_iterations=args.max_iterations,
        extra_hooks=[cli_hook],
        mcp_servers=load_mcp_servers(args.mcp_config),
        malformed_retry_budget=args.malformed_retry_budget,
    )

    if is_daemon:
        _install_daemon_signal_handlers(args)

    cli_hook._log("START", f'prompt="{args.prompt[:80]}"')

    try:
        result = asyncio.run(
            runner.run(
                prompt=args.prompt,
                workspace=workspace,
                session_key=session_key,
                trace_file=trace_file,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    cli_hook.print_summary()

    content = result.content
    if content is None:
        cli_hook._log(
            "WARN",
            "Agent completed without direct response — check workspace for output.",
        )

    if args.output_json:
        output = {
            "session_key": session_key,
            "content": content,
            "steps": cli_hook._n_iterations,
            "tokens": cli_hook._total_tokens,
            "elapsed_s": round(result.elapsed_s, 2),
            "trace_file": str(trace_file),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    elif content:
        print(content)

    return 0


def _install_daemon_signal_handlers(args: argparse.Namespace) -> None:
    pid_file = getattr(args, "_pid_file", None)

    def _cleanup(*_args: Any) -> None:
        if pid_file:
            try:
                Path(pid_file).unlink(missing_ok=True)
            except OSError:
                pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, _cleanup)
    if pid_file:
        atexit.register(lambda: Path(pid_file).unlink(missing_ok=True))


def _run_async(args: argparse.Namespace) -> int:
    from agents.openclaw._daemon import spawn_daemon

    llm_config = _resolve_llm_config(args)
    session_id = args.session_id or f"oc-{uuid.uuid4().hex[:8]}"
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    pid_dir = workspace / ".openclaw" / "pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / f"{session_id}.pid"

    trace_file = _resolve_trace_output(args, session_id, llm_config.model)
    # mkdir deferred to daemon's SessionRunner — avoids empty dirs if spawn fails

    cmd = [
        sys.executable,
        "-m",
        "agents.openclaw",
        "--_daemon",
        "--_pid-file",
        str(pid_file),
        "--prompt",
        args.prompt,
        "--workspace",
        str(workspace),
        "--provider",
        llm_config.name,
        "--model",
        llm_config.model,
        "--api-base",
        llm_config.api_base,
        "--max-iterations",
        str(args.max_iterations),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--malformed-retry-budget",
        str(args.malformed_retry_budget),
        "--session-id",
        session_id,
        "--trace-output",
        str(trace_file),
    ]
    if args.top_p is not None:
        cmd.extend(["--top-p", str(args.top_p)])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.repetition_penalty is not None:
        cmd.extend(["--repetition-penalty", str(args.repetition_penalty)])
    if args.mcp_config is not None:
        cmd.extend(["--mcp-config", str(args.mcp_config)])

    pid = spawn_daemon(
        cmd,
        pid_file,
        session_id,
        extra_env={llm_config.env_key: llm_config.api_key},
        trace_file=trace_file,
    )

    result = {
        "session_id": session_id,
        "pid": pid,
        "pid_file": str(pid_file),
        "workspace": str(workspace),
        "trace_file": str(trace_file),
    }
    print(json.dumps(result, indent=2))
    return 0


def _run_status(args: argparse.Namespace) -> int:
    from agents.openclaw._daemon import get_session_status

    if not args.session_id:
        print("ERROR: --status requires --session-id.", file=sys.stderr)
        return 1

    workspace = Path(args.workspace).expanduser().resolve()
    status = get_session_status(args.session_id, workspace)
    print(json.dumps(status, indent=2))
    return 0


def main() -> None:
    args = build_parser().parse_args()

    if args.status:
        sys.exit(_run_status(args))

    if not args.prompt:
        print(
            "ERROR: --prompt is required (unless using --status).",
            file=sys.stderr,
        )
        build_parser().print_usage(sys.stderr)
        sys.exit(1)

    if args._daemon:
        sys.exit(_run_sync(args))
    elif args.run_async:
        sys.exit(_run_async(args))
    else:
        sys.exit(_run_sync(args))
