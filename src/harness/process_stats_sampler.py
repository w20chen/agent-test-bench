"""Background sampler for host-process CPU, memory, and optional I/O stats."""

from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.memory_bandwidth import attach_host_memory_bandwidth
from harness.micro_arch import attach_micro_arch


def _now_sample() -> dict[str, Any]:
    now = datetime.now(tz=timezone.utc)
    return {
        "timestamp": now.isoformat().replace("+00:00", ""),
        "epoch": now.timestamp(),
    }


def _read_proc_io(pid: int) -> dict[str, int] | None:
    """Read block-layer I/O bytes from ``/proc/<pid>/io``.

    Uses ``read_bytes`` / ``write_bytes`` (bytes actually fetched from /
    written to the storage layer), NOT ``rchar`` / ``wchar`` (syscall-
    level bytes that include TTY, pipe, socket, and page-cache hits).

    Only reports the root PID — does not recurse into children.
    Prefer ``_sample_with_psutil`` when psutil is available because it
    sums across the full process tree.
    """
    try:
        text = Path(f"/proc/{pid}/io").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        try:
            values[key.strip()] = int(value.strip())
        except ValueError:
            continue
    read_bytes = values.get("read_bytes")
    write_bytes = values.get("write_bytes")
    if read_bytes is None and write_bytes is None:
        return None
    return {
        "read_bytes": read_bytes or 0,
        "write_bytes": write_bytes or 0,
    }


def _read_proc_net_dev(pid: int) -> dict[str, int] | None:
    """Read cumulative network RX/TX bytes from ``/proc/<pid>/net/dev``.

    Skips the loopback interface (lo). Returns total bytes across all
    real (non-loopback) interfaces, matching the ``net_rx_bytes`` /
    ``net_tx_bytes`` field names used by ``ContainerStatsSampler``.
    """
    try:
        text = Path(f"/proc/{pid}/net/dev").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    rx_total = 0
    tx_total = 0
    found = False
    for line in text.splitlines():
        # Skip header lines (contain "|" or start with "Inter-")
        if "|" in line or line.startswith("Inter-"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        iface = parts[0].strip()
        if iface == "lo":
            continue
        fields = parts[1].split()
        if len(fields) >= 9:
            try:
                rx_total += int(fields[0])
                tx_total += int(fields[8])
                found = True
            except ValueError:
                continue
    return {"net_rx_bytes": rx_total, "net_tx_bytes": tx_total} if found else None


def _read_proc_context_switches(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    total = 0
    found = False
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep or key not in {
            "voluntary_ctxt_switches",
            "nonvoluntary_ctxt_switches",
        }:
            continue
        try:
            total += int(value.strip())
            found = True
        except ValueError:
            continue
    return total if found else None


def _sample_with_ps(pid: int) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "%cpu=", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    if len(parts) < 2:
        return None
    try:
        cpu_percent = float(parts[0])
        rss_kb = float(parts[1])
    except ValueError:
        return None
    sample = _now_sample()
    sample.update(
        {
            "mem_usage": f"{rss_kb / 1024:.3f}MiB",
            "mem_percent": "0%",
            "cpu_percent": f"{cpu_percent:.3f}%",
        }
    )
    return sample


def _cache_process(
    pid: int,
    *,
    psutil_module: Any,
    process_cache: dict[int, Any] | None,
) -> Any:
    if process_cache is None:
        return psutil_module.Process(pid)
    process = process_cache.get(pid)
    if process is None:
        process = psutil_module.Process(pid)
        try:
            process.cpu_percent(interval=None)
        except Exception:
            pass
        process_cache[pid] = process
    return process


def _expand_exclude_set(root_pids: set[int]) -> set[int]:
    """Return *root_pids* plus all current descendants of each root PID.

    This handles the case where an instrumented subprocess (e.g. ksys)
    spawns helper child processes — excluding only the root PID would
    miss those helpers.
    """
    if not root_pids:
        return set()
    try:
        import psutil
    except ImportError:
        return set(root_pids)
    result = set(root_pids)
    for pid in root_pids:
        try:
            for child in psutil.Process(pid).children(recursive=True):
                result.add(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
            result.add(pid)  # keep the root even if it's gone
    return result


def _sample_with_psutil(
    pid: int,
    process_cache: dict[int, Any] | None = None,
    ctx_high_water: dict[tuple[int, float], int] | None = None,
    io_high_water: dict[tuple[int, float], tuple[int, int]] | None = None,
    exclude_pids: set[int] | None = None,
    include_children: bool = True,
) -> dict[str, Any] | None:
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return None
    try:
        process = _cache_process(
            pid,
            psutil_module=psutil,
            process_cache=process_cache,
        )
    except Exception:
        return None
    children: list[Any] = []
    if include_children:
        try:
            children = list(process.children(recursive=True))
        except Exception:
            children = []
    # Exclude instrumented subprocesses (e.g. ksys) and their entire
    # subtrees from the metrics so they don't pollute the agent's
    # resource accounting.
    _exclude = exclude_pids or set()
    if _exclude:
        children = [c for c in children if c.pid not in _exclude]
    if process_cache is not None:
        for child in children:
            child_pid = getattr(child, "pid", None)
            if isinstance(child_pid, int) and child_pid not in process_cache:
                try:
                    child.cpu_percent(interval=None)
                except Exception:
                    pass
                process_cache[child_pid] = child
        children = [
            process_cache.get(getattr(child, "pid", None), child)
            for child in children
        ]
    processes = [process, *children]
    cpu = 0.0
    rss = 0
    disk_read_bytes = 0
    disk_write_bytes = 0
    context_switches = 0
    found_io = False
    found_context = False
    for proc in processes:
        try:
            cpu += float(proc.cpu_percent(interval=None))
        except Exception:
            pass
        try:
            rss += int(proc.memory_info().rss)
        except Exception:
            pass
        try:
            io_counters = proc.io_counters()
            proc_read = int(getattr(io_counters, "read_bytes", 0))
            proc_write = int(getattr(io_counters, "write_bytes", 0))
            if io_high_water is not None:
                # Keep each process's last-known I/O counters keyed by
                # (pid, create_time) so a child exiting between samples
                # does not make the summed counter drop (which would
                # otherwise yield negative rates).
                try:
                    key = (int(proc.pid), float(proc.create_time()))
                except Exception:
                    key = (int(getattr(proc, "pid", 0)), 0.0)
                prev = io_high_water.get(key, (0, 0))
                io_high_water[key] = (
                    max(prev[0], proc_read),
                    max(prev[1], proc_write),
                )
            else:
                disk_read_bytes += proc_read
                disk_write_bytes += proc_write
            found_io = True
        except Exception:
            pass
        try:
            ctxt = proc.num_ctx_switches()
            proc_ctx = int(ctxt.voluntary + ctxt.involuntary)
            if ctx_high_water is not None:
                # Keep each process's last count keyed by (pid, create_time) so
                # a child exiting between samples does not make the summed
                # counter drop (which would otherwise yield negative rates).
                try:
                    key = (int(proc.pid), float(proc.create_time()))
                except Exception:
                    key = (int(getattr(proc, "pid", 0)), 0.0)
                ctx_high_water[key] = max(ctx_high_water.get(key, 0), proc_ctx)
            else:
                context_switches += proc_ctx
            found_context = True
        except Exception:
            pass
    sample = _now_sample()
    sample.update(
        {
            "mem_usage": f"{rss / (1024 * 1024):.3f}MiB",
            "mem_percent": "0%",
            "cpu_percent": f"{cpu:.3f}%",
        }
    )
    if found_io:
        if io_high_water is not None:
            disk_read_bytes = sum(v[0] for v in io_high_water.values())
            disk_write_bytes = sum(v[1] for v in io_high_water.values())
        sample["disk_read_bytes"] = disk_read_bytes
        sample["disk_write_bytes"] = disk_write_bytes
    if found_context:
        if ctx_high_water is not None:
            context_switches = sum(ctx_high_water.values())
        sample["context_switches"] = context_switches
    if len(processes) > 1:
        sample["process_count"] = len(processes)
    return sample


def _fallback_sample() -> dict[str, Any]:
    sample = _now_sample()
    sample.update(
        {
            "mem_usage": "0MiB",
            "mem_percent": "0%",
            "cpu_percent": "0%",
        }
    )
    return sample


class ProcessStatsSampler(threading.Thread):
    """Sample host-process stats with the ContainerStatsSampler-like interface."""

    # Consolidate per-process high-water marks every N samples to bound
    # dict growth on long-running agents that spawn many short-lived
    # children (prevents slow memory creep).
    _HW_CONSOLIDATE_EVERY = 200

    def __init__(
        self,
        pid: int | None = None,
        *,
        interval_s: float = 1.0,
        exclude_pids: set[int] | None = None,
        include_children: bool = True,
    ) -> None:
        target_pid = os.getpid() if pid is None else pid
        super().__init__(daemon=True, name=f"proc-stats-{target_pid}")
        self.pid = target_pid
        self.interval_s = interval_s
        self._exclude_pids: set[int] = exclude_pids or set()
        self._include_children = include_children
        self._stop_event = threading.Event()
        self._samples: list[dict[str, Any]] = []
        self._sample_count: int = 0
        self._psutil_process_cache: dict[int, Any] = {}
        # Per-(pid, create_time) high-water marks of context switches so the
        # summed counter stays monotonic even as child processes come and go.
        self._ctx_high_water: dict[tuple[int, float], int] = {}
        # Per-(pid, create_time) high-water marks of disk I/O (read_bytes,
        # write_bytes) for the same reason.
        self._io_high_water: dict[tuple[int, float], tuple[int, int]] = {}
        # Consolidated totals absorbed from previous high-water windows.
        self._consolidated_ctx: int = 0
        self._consolidated_io_read: int = 0
        self._consolidated_io_write: int = 0

    def _maybe_consolidate_hw(self) -> None:
        """Periodically fold per-process high-water marks into running totals.

        Without consolidation the ``_ctx_high_water`` and ``_io_high_water``
        dicts grow without bound on agents that spawn many short-lived
        children.  Folding preserves the cumulative counter while keeping
        the working set small.
        """
        self._sample_count += 1
        if self._sample_count % self._HW_CONSOLIDATE_EVERY != 0:
            return
        if self._ctx_high_water:
            self._consolidated_ctx += sum(self._ctx_high_water.values())
            self._ctx_high_water.clear()
        if self._io_high_water:
            for _rb, _wb in self._io_high_water.values():
                self._consolidated_io_read += _rb
                self._consolidated_io_write += _wb
            self._io_high_water.clear()

    def _collect_sample(self) -> dict[str, Any] | None:
        self._maybe_consolidate_hw()
        # Dynamically expand exclude_pids to cover the full subtree of each
        # excluded root (e.g. ksys may spawn helper processes).
        expanded_exclude = _expand_exclude_set(self._exclude_pids)
        sample = _sample_with_psutil(
            self.pid,
            process_cache=self._psutil_process_cache,
            ctx_high_water=self._ctx_high_water,
            io_high_water=self._io_high_water,
            exclude_pids=expanded_exclude,
            include_children=self._include_children,
        ) or _sample_with_ps(self.pid)
        if sample is None:
            sample = _fallback_sample()
        # Add consolidated high-water totals so exited-process
        # contributions are preserved across consolidation windows.
        if self._consolidated_ctx:
            prev = sample.get("context_switches", 0)
            sample["context_switches"] = prev + self._consolidated_ctx
        if self._consolidated_io_read or self._consolidated_io_write:
            prev_r = sample.get("disk_read_bytes", 0)
            prev_w = sample.get("disk_write_bytes", 0)
            sample["disk_read_bytes"] = prev_r + self._consolidated_io_read
            sample["disk_write_bytes"] = prev_w + self._consolidated_io_write
        # Only fall back to /proc/<pid>/io (root PID only, no children)
        # when psutil did not already provide process-tree I/O counters.
        if "disk_read_bytes" not in sample:
            proc_io = _read_proc_io(self.pid)
            if proc_io is not None:
                sample["disk_read_bytes"] = proc_io["read_bytes"]
                sample["disk_write_bytes"] = proc_io["write_bytes"]
        ctxt = _read_proc_context_switches(self.pid)
        if ctxt is not None and "context_switches" not in sample:
            sample["context_switches"] = ctxt
        net = _read_proc_net_dev(self.pid)
        if net is not None:
            sample["net_rx_bytes"] = net["net_rx_bytes"]
            sample["net_tx_bytes"] = net["net_tx_bytes"]
        attach_host_memory_bandwidth(sample, interval_s=self.interval_s)
        # Micro-arch PMU sampling runs at ≥ 2× the process-stats rate
        # to reduce stair-step artefacts from alternating group rotation.
        _micro_arch_interval = max(0.5, self.interval_s / 2)
        attach_micro_arch(sample, interval_s=_micro_arch_interval)
        return sample

    def run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._collect_sample()
            if sample is not None:
                self._samples.append(sample)
            if self._stop_event.wait(self.interval_s):
                break

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=self.interval_s + 6.0)
        if not self._samples:
            sample = self._collect_sample()
            if sample is not None:
                self._samples.append(sample)
        return list(self._samples)
