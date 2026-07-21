"""JSONL output formatting for tool_scheduler records."""

from __future__ import annotations

from typing import Optional

from .monitor import MonitorSample
from .scheduler import Decision
from .topology import Topology


def build_record(
    *,
    invocation_id: str,
    command: list[str],
    root_pid: int,
    exit_code: int,
    runtime_s: float,
    samples: list[MonitorSample],
    decisions: list[Decision],
    median_effective_cores: float,
    p90_effective_cores: float,
    peak_effective_cores: float,
    rss_peak_bytes: int,
    short_tool: bool,
    save_samples: bool = False,
    topology: Optional[Topology] = None,
) -> dict:
    """Build the JSONL record for a single tool invocation.

    Args:
        invocation_id: UUID for this invocation.
        command: The full command list.
        root_pid: Root process PID.
        exit_code: Tool exit code.
        runtime_s: Total wall-clock runtime.
        samples: All collected monitor samples.
        decisions: All scheduling decisions.
        median_effective_cores: Median effective cores across all windows.
        p90_effective_cores: P90 effective cores.
        peak_effective_cores: Peak effective cores.
        rss_peak_bytes: Peak RSS in bytes.
        short_tool: True if runtime < 2 seconds.
        save_samples: If True, include all sample data.
        topology: Hardware topology info.

    Returns:
        A dict suitable for JSON serialization.
    """
    record: dict = {
        "schema_version": 1,
        "invocation_id": invocation_id,
        "command": command,
        "root_pid": root_pid,
        "exit_code": exit_code,
        "runtime_s": round(runtime_s, 3),
        "decisions": [d.to_dict() for d in decisions],
        "final_profile": {
            "median_effective_cores": round(median_effective_cores, 2),
            "p90_effective_cores": round(p90_effective_cores, 2),
            "peak_effective_cores": round(peak_effective_cores, 2),
            "rss_peak_bytes": rss_peak_bytes,
            "short_tool": short_tool,
            "n_samples": len(samples),
            "n_decisions": len(decisions),
        },
    }

    if save_samples and samples:
        record["samples"] = [
            {
                "elapsed_s": round(s.elapsed_s, 3),
                "effective_cores": round(s.effective_cores, 2),
                "rss_bytes": s.rss_bytes,
                "process_count": s.process_count,
                "thread_count": s.thread_count,
                "cpu_user_time_s": round(s.cpu_user_time_s, 4),
                "cpu_system_time_s": round(s.cpu_system_time_s, 4),
                "read_bytes": s.read_bytes,
                "write_bytes": s.write_bytes,
            }
            for s in samples
        ]

    if topology is not None and topology.available:
        record["topology"] = {
            "total_logical_cpus": topology.total_logical_cpus,
            "total_physical_cores": topology.total_physical_cores,
            "numa_nodes": topology.numa_nodes,
            "smt_enabled": topology.smt_enabled,
        }

    return record
