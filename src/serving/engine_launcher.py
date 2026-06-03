from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from serving._common import print_or_exec_command


@dataclass(slots=True)
class VLLMServerConfig:
    """Configuration for launching the raw vLLM OpenAI-compatible server."""

    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: str = "float16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    enable_chunked_prefill: bool = True
    # `--preemption-mode` was removed in vLLM 0.10+; pass an explicit value only
    # when targeting older versions that still accept the flag.
    preemption_mode: str | None = None
    max_num_seqs: int = 256
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    enable_scheduler_hook: bool = False
    scheduler_hook_report_path: str | None = None
    capture_startup_log_path: str | None = None


def build_vllm_command(config: VLLMServerConfig) -> list[str]:
    """Build the raw vLLM api_server command from structured config."""
    server_args = [
        "--model",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        f"{config.gpu_memory_utilization:.2f}",
        "--max-num-seqs",
        str(config.max_num_seqs),
    ]
    if config.preemption_mode is not None:
        server_args.extend(["--preemption-mode", config.preemption_mode])
    if config.enable_chunked_prefill:
        server_args.append("--enable-chunked-prefill")
    if config.enable_auto_tool_choice:
        server_args.append("--enable-auto-tool-choice")
    if config.tool_call_parser:
        server_args.extend(["--tool-call-parser", config.tool_call_parser])
    if config.enable_scheduler_hook:
        if not config.scheduler_hook_report_path:
            raise ValueError(
                "scheduler_hook_report_path is required when scheduler hook is enabled"
            )
        return [
            sys.executable,
            "-m",
            "harness.vllm_entrypoint_with_hooks",
            "--hook-report-path",
            config.scheduler_hook_report_path,
            "--",
            *server_args,
        ]
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *server_args]


def _exec_with_stderr_tee(command: list[str], log_path: str) -> int:
    """Spawn `command` as a subprocess, teeing combined stdout+stderr to
    both `log_path` and the parent's stderr.

    vLLM 0.10+ writes INFO logs to stdout (not stderr); merging the two
    streams ensures both engine startup lines AND tqdm progress bars
    land in the captured log so `vllm_startup_parser` can extract a
    baseline regardless of vLLM version.
    """
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    def _tee() -> None:
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:  # type: ignore[arg-type]
                sys.stderr.write(line)
                sys.stderr.flush()
                f.write(line)
                f.flush()

    t = threading.Thread(target=_tee, daemon=True)
    t.start()
    proc.wait()
    t.join(timeout=5.0)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally launch a raw vLLM API server command."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--preemption-mode", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--enable-auto-tool-choice", action="store_true")
    parser.add_argument("--tool-call-parser", default=None)
    parser.add_argument("--enable-scheduler-hook", action="store_true")
    parser.add_argument("--scheduler-hook-report-path")
    parser.add_argument(
        "--capture-startup-log",
        default=None,
        metavar="PATH",
        help="Tee vLLM stderr to this file while also forwarding to parent stderr.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the resolved command as JSON instead of replacing the process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VLLMServerConfig(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=args.enable_chunked_prefill,
        preemption_mode=args.preemption_mode,
        max_num_seqs=args.max_num_seqs,
        enable_auto_tool_choice=args.enable_auto_tool_choice,
        tool_call_parser=args.tool_call_parser,
        enable_scheduler_hook=args.enable_scheduler_hook,
        scheduler_hook_report_path=args.scheduler_hook_report_path,
        capture_startup_log_path=args.capture_startup_log,
    )
    command = build_vllm_command(config)
    if args.capture_startup_log and not args.print_only:
        returncode = _exec_with_stderr_tee(command, args.capture_startup_log)
        sys.exit(returncode)
    print_or_exec_command(config=config, command=command, print_only=args.print_only)


if __name__ == "__main__":
    main()
