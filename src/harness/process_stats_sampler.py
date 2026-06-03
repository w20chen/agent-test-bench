"""Background sampler for host-process CPU, memory, and optional I/O stats."""

from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.memory_bandwidth import attach_host_memory_bandwidth


def _now_sample() -> dict[str, Any]:
    now = datetime.now(tz=timezone.utc)
    return {
        "timestamp": now.isoformat().replace("+00:00", ""),
        "epoch": now.timestamp(),
    }


def _read_proc_io(pid: int) -> dict[str, int] | None:
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


def _sample_with_psutil(
    pid: int,
    process_cache: dict[int, Any] | None = None,
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
    try:
        children = list(process.children(recursive=True))
    except Exception:
        children = []
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
            disk_read_bytes += int(getattr(io_counters, "read_bytes", 0))
            disk_write_bytes += int(getattr(io_counters, "write_bytes", 0))
            found_io = True
        except Exception:
            pass
        try:
            ctxt = proc.num_ctx_switches()
            context_switches += int(ctxt.voluntary + ctxt.involuntary)
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
        sample["disk_read_bytes"] = disk_read_bytes
        sample["disk_write_bytes"] = disk_write_bytes
    if found_context:
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

    def __init__(self, pid: int | None = None, *, interval_s: float = 1.0) -> None:
        target_pid = os.getpid() if pid is None else pid
        super().__init__(daemon=True, name=f"proc-stats-{target_pid}")
        self.pid = target_pid
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._samples: list[dict[str, Any]] = []
        self._psutil_process_cache: dict[int, Any] = {}

    def _collect_sample(self) -> dict[str, Any] | None:
        sample = _sample_with_psutil(
            self.pid,
            process_cache=self._psutil_process_cache,
        ) or _sample_with_ps(self.pid)
        if sample is None:
            sample = _fallback_sample()
        proc_io = _read_proc_io(self.pid)
        if (
            proc_io is not None
            and "disk_read_bytes" not in sample
            and "disk_write_bytes" not in sample
        ):
            sample["disk_read_bytes"] = proc_io["read_bytes"]
            sample["disk_write_bytes"] = proc_io["write_bytes"]
        ctxt = _read_proc_context_switches(self.pid)
        if ctxt is not None and "context_switches" not in sample:
            sample["context_switches"] = ctxt
        attach_host_memory_bandwidth(sample, interval_s=self.interval_s)
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
