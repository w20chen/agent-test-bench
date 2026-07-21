"""Tool invocation runner: launch, monitor, collect samples, produce profiles."""

from __future__ import annotations

import io
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import psutil

from .sampler import sample_process_tree, SamplePoint
from .metrics import compute_windows, aggregate_samples, AggregatedMetrics

logger = logging.getLogger(__name__)


# Use a small epsilon to avoid division by zero
_EPSILON = 1e-12


def _format_bytes(n: int) -> str:
    """Format byte count in human-readable form."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KiB"
    return f"{n} B"


def _print_early_profile(agg: AggregatedMetrics, command_str: str, elapsed: float) -> None:
    """Print the EARLY PROFILE block to stderr."""
    lines = [
        "",
        "EARLY PROFILE",
        f"  command:              {command_str}",
        f"  elapsed:              {elapsed:.2f} s",
        f"  behavior:             {agg.preliminary_behavior}",
        f"  stability:            {agg.profile_stability}",
        f"  effective cores avg:  {agg.avg_effective_cores:.1f}",
        f"  effective cores peak: {agg.peak_effective_cores:.1f}",
        f"  processes peak:       {agg.peak_process_count}",
        f"  threads peak:         {agg.peak_thread_count}",
        f"  RSS peak:             {_format_bytes(agg.rss_peak_bytes)}",
        f"  read:                 {_format_bytes(agg.total_read_bytes)}",
        f"  write:                {_format_bytes(agg.total_write_bytes)}",
        "",
    ]
    for line in lines:
        print(f"[tool-profiler] {line}" if line else "[tool-profiler]", file=sys.stderr)


def _print_final_profile(agg: AggregatedMetrics, command_str: str, wall_time: float, exit_code: int) -> None:
    """Print the FINAL PROFILE block to stderr."""
    lines = [
        "",
        "FINAL PROFILE",
        f"  wall time:            {wall_time:.2f} s",
        f"  exit code:            {exit_code}",
        f"  behavior:             {agg.preliminary_behavior}",
        f"  stability:            {agg.profile_stability}",
        f"  effective cores avg:  {agg.avg_effective_cores:.1f}",
        f"  effective cores p50:  {agg.p50_effective_cores:.1f}",
        f"  effective cores p90:  {agg.p90_effective_cores:.1f}",
        f"  effective cores peak: {agg.peak_effective_cores:.1f}",
        f"  processes peak:       {agg.peak_process_count}",
        f"  threads peak:         {agg.peak_thread_count}",
        f"  RSS peak:             {_format_bytes(agg.rss_peak_bytes)}",
        f"  read:                 {_format_bytes(agg.total_read_bytes)}",
        f"  write:                {_format_bytes(agg.total_write_bytes)}",
        "",
    ]
    for line in lines:
        print(f"[tool-profiler] {line}" if line else "[tool-profiler]", file=sys.stderr)


def _shorten_uuid(u: str, n: int = 4) -> str:
    return u.split("-")[0][:n]


def run_tool(
    command: list[str],
    *,
    warmup_seconds: float = 2.0,
    sample_interval: float = 0.2,
    output_path: str = "tool_profiles.jsonl",
    verbose: bool = False,
    save_samples: bool = False,
    shell_command: bool = False,
) -> int:
    """Run an external command with resource profiling.

    Args:
        command: The command and arguments to execute.
        warmup_seconds: Duration of the early profile observation window.
        sample_interval: Time between consecutive samples.
        output_path: Path for JSONL output file.
        verbose: If True, print per-sample summaries to stderr.
        save_samples: If True, include all raw samples in JSONL output.

    Returns:
        The exit code of the tools process.
    """
    invocation_id = str(uuid.uuid4())
    if shell_command:
        if len(command) != 1:
            raise ValueError("shell_command mode requires exactly one command string")
        command_str = command[0]
    else:
        command_str = " ".join(command)
    cwd = os.getcwd()
    start_time = datetime.now(timezone.utc).isoformat()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(
        f"[tool-profiler] started invocation={_shorten_uuid(invocation_id)} "
        f"command={command_str}",
        file=sys.stderr,
    )

    # Keep the profiled workload in this process group/session so an outer
    # harness timeout can terminate the wrapper and workload together.  Existing
    # shell commands can opt into shell parsing with shell_command=True.
    if shell_command:
        popen_kwargs: dict = {"shell": True}
        popen_cmd: str | list[str] = command_str
    elif sys.platform == "win32":
        popen_kwargs = {"shell": False}
        popen_cmd = command
    else:
        popen_kwargs = {"shell": False}
        popen_cmd = command

    # Handle redirected/pseudo stdin/stdout/stderr (e.g., under pytest)
    # Some environments replace sys.std* with objects lacking fileno().
    def _safe_stream(stream, default):
        """Return stream if usable, otherwise default (e.g., DEVNULL)."""
        try:
            stream.fileno()
            return stream
        except (AttributeError, OSError, io.UnsupportedOperation):
            return default

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            popen_cmd,
            stdin=_safe_stream(sys.stdin, subprocess.DEVNULL),
            stdout=_safe_stream(sys.stdout, subprocess.DEVNULL),
            stderr=_safe_stream(sys.stderr, subprocess.DEVNULL),
            cwd=cwd,
            env=os.environ.copy(),
            **popen_kwargs,
        )
    except FileNotFoundError as e:
        print(f"[tool-profiler] ERROR: command not found: {e}", file=sys.stderr)
        return 127
    except Exception as e:
        print(f"[tool-profiler] ERROR: failed to start command: {e}", file=sys.stderr)
        return 1

    root_pid = proc.pid
    print(f"[tool-profiler] pid={root_pid}", file=sys.stderr)

    def _terminate_process_tree(pid: int, sig: int | None = None) -> None:
        """Terminate the profiled process tree rooted at *pid*."""
        try:
            root = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

        try:
            children = root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            children = []

        for child in reversed(children):
            try:
                if sig is not None and sys.platform != "win32":
                    os.kill(child.pid, sig)
                else:
                    child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass

        try:
            if sig is not None and sys.platform != "win32":
                os.kill(root.pid, sig)
            else:
                root.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

        _gone, alive = psutil.wait_procs([*children, root], timeout=5.0)
        for proc_alive in alive:
            try:
                proc_alive.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    received_signal: int | None = None
    previous_handlers: dict[int, object] = {}

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        print(
            f"\n[tool-profiler] signal {signum} received, terminating workload tree",
            file=sys.stderr,
        )
        _terminate_process_tree(root_pid, signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError, AttributeError):
            pass

    samples: list[SamplePoint] = []
    sample_lock = threading.Lock()
    stop_sampling = threading.Event()
    early_profile_emitted = False
    early_profile_available = False
    early_samples: list[SamplePoint] = []
    early_windows: list = []

    start_mono = time.monotonic()

    def sampler_loop() -> None:
        nonlocal early_profile_emitted, early_profile_available, early_samples, early_windows

        while not stop_sampling.is_set():
            elapsed = time.monotonic() - start_mono
            sp = sample_process_tree(root_pid, elapsed)

            if sp is not None:
                with sample_lock:
                    samples.append(sp)

                if verbose:
                    print(
                        f"[tool-profiler] t={elapsed:.1f}s "
                        f"procs={sp.process_count} threads={sp.thread_count} "
                        f"cpu={sp.cpu_total_time_s:.2f}s "
                        f"rss={_format_bytes(sp.rss_bytes)}",
                        file=sys.stderr,
                    )

            # Check if warmup window reached
            if (
                not early_profile_emitted
                and elapsed >= warmup_seconds
                and sp is not None
            ):
                with sample_lock:
                    early_samples = list(samples)
                early_windows_list = compute_windows(early_samples)
                early_agg = aggregate_samples(early_samples, early_windows_list)
                early_windows = early_windows_list
                early_profile_available = True

                _print_early_profile(early_agg, command_str, elapsed)
                print(
                    f"[tool-profiler] warmup window reached at {elapsed:.2f}s",
                    file=sys.stderr,
                )
                early_profile_emitted = True

            stop_sampling.wait(sample_interval)

    sampler_thread = threading.Thread(target=sampler_loop, daemon=True)
    sampler_thread.start()

    # Wait for the tool to finish; signal handlers above clean descendants.
    exit_code: int = -1
    try:
        exit_code = proc.wait()
    except KeyboardInterrupt:
        _handle_signal(signal.SIGINT, None)
        exit_code = proc.returncode if proc.returncode is not None else 130
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, AttributeError):
                pass
        stop_sampling.set()
        sampler_thread.join(timeout=3.0)

    if received_signal is not None:
        exit_code = 128 + received_signal

    total_wall = time.monotonic() - start_mono

    # Collect final sample
    final_sp = sample_process_tree(root_pid, total_wall)
    if final_sp is not None:
        with sample_lock:
            samples.append(final_sp)

    # Compute final aggregate
    windows = compute_windows(samples)
    final_agg = aggregate_samples(samples, windows)
    short_tool = total_wall < warmup_seconds

    # Compute early-final comparison
    effective_cores_relative_error: Optional[float] = None
    behavior_changed: Optional[bool] = None
    stability_changed: Optional[bool] = None

    if early_profile_available and final_agg.num_samples > 0:
        early_agg = aggregate_samples(early_samples, early_windows if early_windows else compute_windows(early_samples))
        effective_cores_relative_error = (
            abs(early_agg.avg_effective_cores - final_agg.avg_effective_cores)
            / max(final_agg.avg_effective_cores, _EPSILON)
        )
        behavior_changed = (
            early_agg.preliminary_behavior != final_agg.preliminary_behavior
        )
        stability_changed = (
            early_agg.profile_stability != final_agg.profile_stability
        )

    if received_signal is not None:
        print(
            f"[tool-profiler] command terminated by signal",
            file=sys.stderr,
        )
    else:
        signame = ""
        if exit_code < 0:
            try:
                signame = f" (signal {signal.Signals(-exit_code).name})"
            except (ValueError, AttributeError):
                pass
        print(
            f"[tool-profiler] command exited with code {exit_code}{signame}",
            file=sys.stderr,
        )

    _print_final_profile(final_agg, command_str, total_wall, exit_code)

    # Build JSONL record
    record: dict = {
        "schema_version": 1,
        "invocation_id": invocation_id,
        "command": command,
        "command_string": command_str,
        "cwd": cwd,
        "root_pid": root_pid,
        "start_time": start_time,
        "warmup_seconds": warmup_seconds,
        "sample_interval": sample_interval,
        "exit_code": exit_code,
        "early_profile": {
            "available": early_profile_available,
        },
        "final_profile": {
            "total_wall_time_s": total_wall,
            "short_tool": short_tool,
            "avg_effective_cores": final_agg.avg_effective_cores,
            "p50_effective_cores": final_agg.p50_effective_cores,
            "p90_effective_cores": final_agg.p90_effective_cores,
            "peak_effective_cores": final_agg.peak_effective_cores,
            "peak_process_count": final_agg.peak_process_count,
            "peak_thread_count": final_agg.peak_thread_count,
            "rss_peak_bytes": final_agg.rss_peak_bytes,
            "total_read_bytes": final_agg.total_read_bytes,
            "total_write_bytes": final_agg.total_write_bytes,
            "parallelism_cv": final_agg.parallelism_cv,
            "profile_stability": final_agg.profile_stability,
            "preliminary_behavior": final_agg.preliminary_behavior,
        },
    }

    if early_profile_available:
        early_agg = aggregate_samples(early_samples, early_windows if early_windows else compute_windows(early_samples))
        record["early_profile"] = {
            "available": True,
            "elapsed_s": early_agg.elapsed_end_s,
            "avg_effective_cores": early_agg.avg_effective_cores,
            "peak_effective_cores": early_agg.peak_effective_cores,
            "parallelism_cv": early_agg.parallelism_cv,
            "profile_stability": early_agg.profile_stability,
            "preliminary_behavior": early_agg.preliminary_behavior,
        }
        record["early_final_comparison"] = {
            "effective_cores_relative_error": effective_cores_relative_error,
            "behavior_changed": behavior_changed,
            "stability_changed": stability_changed,
        }

    if save_samples:
        record["samples"] = [
            {
                "timestamp_s": s.timestamp_s,
                "elapsed_s": s.elapsed_s,
                "process_count": s.process_count,
                "thread_count": s.thread_count,
                "cpu_user_time_s": s.cpu_user_time_s,
                "cpu_system_time_s": s.cpu_system_time_s,
                "cpu_total_time_s": s.cpu_total_time_s,
                "rss_bytes": s.rss_bytes,
                "vms_bytes": s.vms_bytes,
                "read_bytes": s.read_bytes,
                "write_bytes": s.write_bytes,
                "read_count": s.read_count,
                "write_count": s.write_count,
                "voluntary_context_switches": s.voluntary_context_switches,
                "involuntary_context_switches": s.involuntary_context_switches,
                "minor_page_faults": s.minor_page_faults,
                "major_page_faults": s.major_page_faults,
            }
            for s in samples
        ]

    # Append JSONL
    try:
        if output_path == "-":
            print(json.dumps(record, ensure_ascii=False), file=sys.stdout)
        else:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[tool-profiler] ERROR: cannot write output: {e}", file=sys.stderr)

    return exit_code
