"""Process tree sampling using psutil.

Acquires root + descendant processes and collects aggregated metrics
for the entire invocation process set at each sample point.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SamplePoint:
    """A single aggregated sample of the invocation process tree."""

    timestamp_s: float
    elapsed_s: float
    root_pid: int

    process_count: int
    thread_count: int

    cpu_user_time_s: float
    cpu_system_time_s: float
    cpu_total_time_s: float

    rss_bytes: int
    vms_bytes: int

    read_bytes: Optional[int]
    write_bytes: Optional[int]
    read_count: Optional[int]
    write_count: Optional[int]

    voluntary_context_switches: Optional[int]
    involuntary_context_switches: Optional[int]

    minor_page_faults: Optional[int]
    major_page_faults: Optional[int]


def _get_process_and_children(
    root: psutil.Process,
) -> list[psutil.Process]:
    """Get root process and all live descendants.

    Handles NoSuchProcess, AccessDenied, and ZombieProcess gracefully.
    """
    procs: list[psutil.Process] = []
    try:
        if root.is_running():
            procs.append(root)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

    try:
        children = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return procs

    for child in children:
        try:
            if child.is_running():
                procs.append(child)
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            # Short-lived child that exited between children() call and now.
            # This is expected; skip it silently.
            continue

    return procs


def _safe_get_io(p: psutil.Process) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Safely retrieve I/O counters. Returns None for each field on failure."""
    try:
        io = p.io_counters()
        return io.read_bytes, io.write_bytes, io.read_count, io.write_count
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
        return None, None, None, None


def _safe_get_ctx_switches(p: psutil.Process) -> tuple[Optional[int], Optional[int]]:
    """Safely retrieve context switch counts."""
    try:
        ctx = p.num_ctx_switches()
        return ctx.voluntary, ctx.involuntary
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
        return None, None


def _safe_get_page_faults(p: psutil.Process) -> tuple[Optional[int], Optional[int]]:
    """Safely retrieve page fault counts.

    NOTE: On some platforms (e.g., macOS), psutil may not expose
    memory_full_info() or page fault details. In that case fields are null.
    """
    try:
        # Prefer memory_full_info() for page fault details
        mem = p.memory_full_info()
        return getattr(mem, "minflt", None), getattr(mem, "majflt", None)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
        return None, None


def _opt_sum(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """Sum two Optional[int] values. If both None, return None."""
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _opt_sum_float(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Sum two Optional[float] values. If both None, return None."""
    if a is None and b is None:
        return None
    return (a or 0.0) + (b or 0.0)


def sample_process_tree(root_pid: int, elapsed_s: float) -> Optional[SamplePoint]:
    """Take one aggregated sample of the process tree rooted at `root_pid`.

    Args:
        root_pid: PID of the root process.
        elapsed_s: Seconds elapsed since invocation start.

    Returns:
        SamplePoint with aggregated metrics, or None if the root process
        is no longer accessible.
    """
    try:
        root = psutil.Process(root_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

    procs = _get_process_and_children(root)
    if not procs:
        return None

    now = time.monotonic()

    # Aggregate CPU times
    cpu_user = 0.0
    cpu_system = 0.0
    procs_seen = 0
    threads_seen = 0
    for p in procs:
        try:
            times = p.cpu_times()
            cpu_user += times.user
            cpu_system += times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        try:
            threads_seen += p.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        procs_seen += 1

    # Aggregate memory (RSS, VMS)
    rss_total = 0
    vms_total = 0
    for p in procs:
        try:
            mem = p.memory_info()
            rss_total += mem.rss
            vms_total += mem.vms
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Aggregate I/O
    read_bytes: Optional[int] = None
    write_bytes: Optional[int] = None
    read_count: Optional[int] = None
    write_count: Optional[int] = None
    for p in procs:
        rb, wb, rc, wc = _safe_get_io(p)
        read_bytes = _opt_sum(read_bytes, rb)
        write_bytes = _opt_sum(write_bytes, wb)
        read_count = _opt_sum(read_count, rc)
        write_count = _opt_sum(write_count, wc)

    # Aggregate context switches
    vol_ctx: Optional[int] = None
    invol_ctx: Optional[int] = None
    for p in procs:
        v, iv = _safe_get_ctx_switches(p)
        vol_ctx = _opt_sum(vol_ctx, v)
        invol_ctx = _opt_sum(invol_ctx, iv)

    # Aggregate page faults
    minflt: Optional[int] = None
    majflt: Optional[int] = None
    for p in procs:
        mn, mj = _safe_get_page_faults(p)
        minflt = _opt_sum(minflt, mn)
        majflt = _opt_sum(majflt, mj)

    return SamplePoint(
        timestamp_s=now,
        elapsed_s=elapsed_s,
        root_pid=root_pid,
        process_count=procs_seen,
        thread_count=threads_seen,
        cpu_user_time_s=cpu_user,
        cpu_system_time_s=cpu_system,
        cpu_total_time_s=cpu_user + cpu_system,
        rss_bytes=rss_total,
        vms_bytes=vms_total,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        read_count=read_count,
        write_count=write_count,
        voluntary_context_switches=vol_ctx,
        involuntary_context_switches=invol_ctx,
        minor_page_faults=minflt,
        major_page_faults=majflt,
    )
