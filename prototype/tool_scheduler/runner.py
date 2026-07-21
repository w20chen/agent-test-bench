"""Main runner: orchestrates monitoring, prediction, and scheduling.

Launches the tool command, runs the monitoring + decision loops,
and produces the final JSONL output.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from .monitor import Monitor, MonitorSample
from .predictor import Predictor
from .topology import Topology, discover as discover_topology, hardcoded_topology
from .cost_model import CostModelConfig
from .scheduler import Scheduler, Decision, DEFAULT_COOLDOWN_SECONDS
from .output import build_record
from .bandwidth import (
    BandwidthCollector,
    MemoryDomainConfig,
    load_memory_domain_configs,
    _auto_detect_memory_domains,
)

logger = logging.getLogger(__name__)

# Decision interval: how often to evaluate scheduling
DECISION_INTERVAL = 1.0  # 1 second

# Minimum samples before predictor starts
MIN_SAMPLES_FOR_PREDICTION = 2


def _format_decision(decision: Decision) -> str:
    """Format a decision as a human-readable block for stderr output."""
    lines = [
        f"\n[decision @ {decision.elapsed_s:.1f}s]",
        f"predicted cores: {decision.predicted_cores:.1f}",
        f"memory sensitivity: {decision.memory_sensitivity}",
        "",
    ]

    if decision.current_cost_breakdown:
        cb = decision.current_cost_breakdown
        lines.extend([
            "current:",
            f"  placement: {cb['placement']}",
            f"  core cost:   {cb['core_cost']:.2f}",
            f"  memory cost: {cb['memory_cost']:.2f}",
            f"  move cost:   {cb['move_cost']:.2f}",
            f"  total cost:  {cb['total_cost']:.2f}",
            "",
        ])

    if decision.best_cost_breakdown and decision.action == "recommend_move":
        cb = decision.best_cost_breakdown
        lines.extend([
            "best candidate:",
            f"  placement: {cb['placement']}",
            f"  core cost:   {cb['core_cost']:.2f}",
            f"  memory cost: {cb['memory_cost']:.2f}",
            f"  move cost:   {cb['move_cost']:.2f}",
            f"  total cost:  {cb['total_cost']:.2f}",
            "",
        ])

    if decision.gain is not None:
        lines.append(f"gain: {decision.gain:.2f}")
    lines.append(f"action: {decision.action.upper()}")

    return "\n".join(lines)


def run_tool(
    command: list[str],
    *,
    output_path: str = "profiles.jsonl",
    dry_run: bool = True,
    save_samples: bool = False,
    verbose: bool = False,
    cost_config: Optional[CostModelConfig] = None,
    history_db: Optional[dict[str, dict]] = None,
    memory_sensitivity_overrides: Optional[dict[str, str]] = None,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    alpha: float = 0.3,
    bandwidth_config_path: Optional[str] = None,
    hardcode_topology: bool = False,
) -> int:
    """Run an external command with online load prediction and scheduling.

    Args:
        command: The command and arguments to execute.
        output_path: Path for JSONL output file.
        dry_run: If True, only generate recommendations without applying them.
        save_samples: If True, include all raw samples in JSONL output.
        verbose: If True, print per-sample summaries to stderr.
        cost_config: Cost model configuration.
        history_db: Existing history profiles keyed by command signature.
        memory_sensitivity_overrides: Manual memory sensitivity overrides.
        cooldown_seconds: Minimum seconds between consecutive recommendations.
        alpha: EMA alpha for core prediction.

    Returns:
        The exit code of the tool process.
    """
    invocation_id = str(uuid.uuid4())
    command_str = " ".join(command)
    cwd = os.getcwd()
    start_time = datetime.now(timezone.utc).isoformat()

    # Build command signature for history lookup
    cmd_sig = _build_command_signature(command)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path) or "."
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if history_db is None:
        history_db = {}

    # Apply memory sensitivity overrides
    if memory_sensitivity_overrides:
        for sig, sensitivity in memory_sensitivity_overrides.items():
            history_db.setdefault(sig, {})["memory_sensitivity"] = sensitivity

    if cost_config is None:
        cost_config = CostModelConfig()

    print(
        f"[tool-scheduler] invocation={invocation_id[:8]} "
        f"command={command_str}",
        file=sys.stderr,
    )

    # Discover topology (or use hardcoded)
    if hardcode_topology:
        topology = hardcoded_topology()
        print(
            "[tool-scheduler] using hardcoded 320-core topology "
            f"(2 NUMA, {topology.total_physical_cores} phys, SMT=True)",
            file=sys.stderr,
        )
    else:
        topology = discover_topology()
    if not topology.available:
        print(
            "[tool-scheduler] WARNING: CPU topology unavailable, "
            "scheduling disabled",
            file=sys.stderr,
        )

    # Start bandwidth collectors if on Linux with PMU
    _bw_started = False
    if topology.available and sys.platform == "linux":
        try:
            if bandwidth_config_path:
                bw_configs = load_memory_domain_configs(bandwidth_config_path)
            else:
                bw_configs = _auto_detect_memory_domains()
            if bw_configs and any(
                c.pmu_read_event or c.pmu_write_event or c.pmu_combined_event
                for c in bw_configs
            ):
                BandwidthCollector.start_all(bw_configs)
                _bw_started = True
                print(
                    "[tool-scheduler] bandwidth collectors started for "
                    f"{len(bw_configs)} memory domain(s)",
                    file=sys.stderr,
                )
        except Exception as exc:
            logger.debug("Bandwidth collector init skipped: %s", exc)

    # Launch the tool as a new process group
    if sys.platform == "win32":
        popen_kwargs: dict = {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            "shell": True,
        }
        popen_cmd: str | list[str] = subprocess.list2cmdline(command)
    else:
        popen_kwargs = {
            "preexec_fn": os.setsid,
            "shell": True,
        }
        popen_cmd = " ".join(shlex.quote(arg) for arg in command)

    # Safe stream handling
    def _safe_stream(stream, default):
        try:
            stream.fileno()
            return stream
        except (AttributeError, OSError):
            return default

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
        print(f"[tool-scheduler] ERROR: command not found: {e}", file=sys.stderr)
        return 127
    except Exception as e:
        print(f"[tool-scheduler] ERROR: failed to start command: {e}", file=sys.stderr)
        return 1

    root_pid = proc.pid
    print(f"[tool-scheduler] pid={root_pid}", file=sys.stderr)

    # Start monitoring
    monitor = Monitor(root_pid, sample_interval=0.5)
    monitor.start()

    # Start predictor
    predictor = Predictor(alpha=alpha)

    # Start scheduler
    scheduler = Scheduler(
        predictor=predictor,
        topology=topology,
        history=history_db,
        cost_config=cost_config,
        cooldown_seconds=cooldown_seconds,
        command_sig=cmd_sig,
    )

    # Decision loop: runs in the main thread, checks every 1s
    start_mono = time.monotonic()
    decisions: list[Decision] = []
    running = True
    exit_code: int = -1

    while running:
        # Check if process exited
        poll_result = proc.poll()
        if poll_result is not None:
            exit_code = poll_result
            running = False

        elapsed = time.monotonic() - start_mono
        latest = monitor.latest

        # Update predictor with latest sample
        if latest is not None and latest.effective_cores > 0:
            state = predictor.update(latest.effective_cores)

            # Check for phase change
            if predictor.check_divergence() and predictor.stable:
                print(
                    f"[tool-scheduler] phase change detected at {elapsed:.1f}s, "
                    f"resetting stability",
                    file=sys.stderr,
                )
                scheduler.reset_after_phase_change()

            if verbose and latest.effective_cores > 0:
                print(
                    f"[tool-scheduler] t={elapsed:.1f}s "
                    f"eff_cores={latest.effective_cores:.1f} "
                    f"pred_cores={predictor.predicted_cores:.1f} "
                    f"stable={predictor.stable}",
                    file=sys.stderr,
                )

        # Evaluate scheduling decision
        if topology.available:
            decision = scheduler.evaluate(elapsed, root_pid)
            if decision is not None:
                decisions.append(decision)
                print(_format_decision(decision), file=sys.stderr)

        # Wait for next decision interval (or until process exits)
        if running:
            try:
                proc.wait(timeout=DECISION_INTERVAL)
                exit_code = proc.returncode if proc.returncode is not None else -1
                running = False
            except subprocess.TimeoutExpired:
                pass

    # Stop monitoring
    monitor.stop()

    total_wall = time.monotonic() - start_mono
    short_tool = total_wall < 2.0

    # Collect final samples
    samples = monitor.samples

    # Compute final profile
    effective_cores_list = [
        s.effective_cores for s in samples if s.effective_cores > 0
    ]
    sorted_cores = sorted(effective_cores_list)

    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        k = (p / 100.0) * (len(values) - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return values[int(k)]
        return values[int(f)] * (c - k) + values[int(c)] * (k - f)

    median_cores = _percentile(sorted_cores, 50) if sorted_cores else 0.0
    p90_cores = _percentile(sorted_cores, 90) if sorted_cores else 0.0
    peak_cores = max(effective_cores_list) if effective_cores_list else 0.0
    rss_peak = max((s.rss_bytes for s in samples), default=0)

    # Update history
    if effective_cores_list and not short_tool:
        hist_entry = history_db.setdefault(cmd_sig, {})
        hist_entry["median_effective_cores"] = median_cores
        hist_entry["p90_effective_cores"] = p90_cores
        hist_entry["median_runtime"] = total_wall
        hist_entry["observation_count"] = hist_entry.get("observation_count", 0) + 1
        hist_entry.setdefault("memory_sensitivity", "unknown")

    # Build and write JSONL record
    record = build_record(
        invocation_id=invocation_id,
        command=command,
        root_pid=root_pid,
        exit_code=exit_code,
        runtime_s=total_wall,
        samples=samples,
        decisions=decisions,
        median_effective_cores=median_cores,
        p90_effective_cores=p90_cores,
        peak_effective_cores=peak_cores,
        rss_peak_bytes=rss_peak,
        short_tool=short_tool,
        save_samples=save_samples,
        topology=topology,
    )

    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[tool-scheduler] ERROR: cannot write output: {e}", file=sys.stderr)

    # Stop bandwidth collectors
    if _bw_started:
        try:
            BandwidthCollector.stop_all()
        except Exception:
            pass

    # Print final summary
    print(
        f"\n[tool-scheduler] done: runtime={total_wall:.1f}s exit={exit_code} "
        f"median_cores={median_cores:.1f} p90_cores={p90_cores:.1f} "
        f"decisions={len(decisions)}",
        file=sys.stderr,
    )

    return exit_code


def _build_command_signature(command: list[str]) -> str:
    """Build a command signature for history lookup.

    Uses executable name + first few significant arguments.
    """
    if not command:
        return "unknown"

    # Get executable basename
    exe = os.path.basename(command[0])

    # Take first few non-flag-like arguments (up to 3)
    significant_args: list[str] = []
    for arg in command[1:]:
        if arg.startswith("-"):
            if arg in ("-j", "--jobs", "-p", "--processes", "-n"):
                significant_args.append(arg)
            continue
        significant_args.append(arg)
        if len(significant_args) >= 2:
            break

    if significant_args:
        return f"{exe} {' '.join(significant_args)}"
    return exe
