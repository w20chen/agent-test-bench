"""Per-CPU idle detection from /proc/stat.

Reads /proc/stat to compute per-CPU utilization and determine which
cores are "idle" (below a configurable threshold).  Used by the
scheduler to estimate available physical cores in each LLC group.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default threshold: CPU with utilization below this fraction is "idle"
DEFAULT_IDLE_THRESHOLD = 0.10  # 10%

# /proc/stat path
_PROC_STAT = Path("/proc/stat")


class CpuUtilization:
    """Per-CPU utilization snapshot computed from /proc/stat."""

    def __init__(self) -> None:
        self._prev: dict[int, tuple[float, float]] = {}  # cpu_id -> (idle, total)
        self._prev_time: float = 0.0

    def sample(self) -> dict[int, float]:
        """Read /proc/stat and return per-CPU utilization [0, 1].

        Returns empty dict on non-Linux or parse failure.
        """
        if not _PROC_STAT.exists():
            return {}

        try:
            text = _PROC_STAT.read_text(encoding="ascii", errors="replace")
        except OSError:
            return {}

        now = time.monotonic()
        current: dict[int, tuple[float, float]] = {}

        for line in text.splitlines():
            if not line.startswith("cpu"):
                continue
            # Skip aggregate "cpu" line (no digit after "cpu")
            parts = line.split()
            if len(parts) < 5:
                continue
            cpu_label = parts[0]
            if cpu_label == "cpu":
                continue
            try:
                cpu_id = int(cpu_label[3:])
            except ValueError:
                continue

            # Fields: user nice system idle iowait irq softirq steal guest guest_nice
            # Some kernels may have fewer fields
            values = [int(v) for v in parts[1:]]
            idle = values[3] if len(values) > 3 else 0
            # iowait is included in idle for utilization calc
            if len(values) > 4:
                idle += values[4]  # iowait is "idle" time
            total = sum(values)
            current[cpu_id] = (float(idle), float(total))

        if not self._prev or self._prev_time == 0.0:
            self._prev = current
            self._prev_time = now
            return {}  # Need at least two samples for delta

        result: dict[int, float] = {}
        for cpu_id, (idle_cur, total_cur) in current.items():
            prev = self._prev.get(cpu_id)
            if prev is None:
                continue
            idle_prev, total_prev = prev
            idle_delta = idle_cur - idle_prev
            total_delta = total_cur - total_prev
            if total_delta <= 0:
                result[cpu_id] = 0.0
            else:
                # Utilization = 1 - (idle_fraction)
                util = 1.0 - (idle_delta / total_delta)
                # Clamp to [0, 1]
                result[cpu_id] = max(0.0, min(1.0, util))

        self._prev = current
        self._prev_time = now
        return result


# Module-level singleton for reuse across scheduler evaluations
_cpu_util = CpuUtilization()


def get_cpu_utilization() -> dict[int, float]:
    """Return per-CPU utilization [0, 1], keyed by logical CPU ID.

    Returns empty dict on first call (needs two samples for delta) or
    on non-Linux platforms.
    """
    return _cpu_util.sample()


def count_idle_cores(
    cpu_list: list[int],
    threshold: float = DEFAULT_IDLE_THRESHOLD,
) -> int:
    """Count how many CPUs in *cpu_list* are idle.

    A CPU is "idle" if its utilization is below *threshold*.

    Args:
        cpu_list: List of logical CPU IDs to check.
        threshold: Utilization below this is considered idle.

    Returns:
        Number of idle logical CPUs.
    """
    util = get_cpu_utilization()
    if not util:
        # No data yet -- assume all are idle (first sample)
        return len(cpu_list)
    idle_count = 0
    for cpu_id in cpu_list:
        if util.get(cpu_id, 1.0) <= threshold:
            idle_count += 1
    return idle_count


def idle_breakdown(
    cpu_list: list[int],
    threshold: float = DEFAULT_IDLE_THRESHOLD,
    physical_cores_per_cpu: dict[int, int] | None = None,
    utilization: dict[int, float] | None = None,
) -> tuple[int, int]:
    """Break down *cpu_list* into (available_physical_cores, available_smt_threads).

    Only counts cores whose utilization is below *threshold*.

    When *physical_cores_per_cpu* is provided (mapping cpu_id → physical_core_id),
    SMT siblings of the same physical core are counted as threads rather than
    independent physical cores.  A physical core is "available" if at least one
    of its SMT siblings is idle.

    Args:
        cpu_list: List of logical CPUs in the candidate group.
        threshold: Utilization below this is considered idle.
        physical_cores_per_cpu: Optional mapping for SMT awareness.

    Returns:
        (available_physical_cores, available_smt_threads)
    """
    util = utilization if utilization is not None else get_cpu_utilization()
    return idle_breakdown_from_utilization(
        cpu_list,
        util,
        threshold=threshold,
        physical_cores_per_cpu=physical_cores_per_cpu,
    )


def idle_breakdown_from_utilization(
    cpu_list: list[int],
    utilization: dict[int, float],
    threshold: float = DEFAULT_IDLE_THRESHOLD,
    physical_cores_per_cpu: dict[int, int] | None = None,
) -> tuple[int, int]:
    """Break down idle CPUs using a caller-provided utilization snapshot."""

    util = utilization
    if not util:
        # No data -- assume all available
        if physical_cores_per_cpu:
            phys = set(physical_cores_per_cpu.get(c, c) for c in cpu_list)
            return len(phys), len(cpu_list) - len(phys)
        return len(cpu_list), 0

    idle_cpus = {c for c in cpu_list if util.get(c, 1.0) <= threshold}

    if physical_cores_per_cpu:
        # Group by physical core
        phys_to_cpus: dict[int, list[int]] = {}
        for c in cpu_list:
            pc = physical_cores_per_cpu.get(c, c)
            phys_to_cpus.setdefault(pc, []).append(c)

        phys_available = 0
        smt_available = 0
        for pc, siblings in phys_to_cpus.items():
            idle_siblings = [s for s in siblings if s in idle_cpus]
            if idle_siblings:
                phys_available += 1
                # First idle sibling counts as physical core,
                # remaining count as SMT threads
                smt_available += len(idle_siblings) - 1
        return phys_available, smt_available

    return len(idle_cpus), 0
