"""Background sampler for container CPU, memory, I/O, and network statistics.

The summary format extends the harness resources.json schema with disk I/O,
network I/O, and context switch metrics alongside the original CPU/memory fields.

Disk I/O is read from the container's cgroup io.stat (cgroup v2), which
correctly aggregates across ALL processes in the container — including
those spawned via ``docker exec``.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.memory_bandwidth import attach_host_memory_bandwidth

logger = logging.getLogger(__name__)

# 4-field pipe-delimited format: mem_usage | mem% | cpu% | net_io
_STATS_FORMAT = "{{.MemUsage}}|{{.MemPerc}}|{{.CPUPerc}}|{{.NetIO}}"


def _resolve_container_pid(
    container_id: str, *, executable: str,
) -> int | None:
    """Get the host-visible init PID of a running container."""
    try:
        result = subprocess.run(
            [executable, "inspect", "--format", "{{.State.Pid}}", container_id],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            pid = int(result.stdout.strip())
            return pid if pid > 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def _resolve_cgroup_path(pid: int) -> Path | None:
    """Resolve the cgroup v2 filesystem path for a container's init PID.

    Reads /proc/<pid>/cgroup which on cgroup v2 has a single line::

        0::<relative-path>

    Returns the absolute cgroup filesystem path, or None if unavailable.
    """
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except (PermissionError, FileNotFoundError, OSError):
        return None
    for line in text.strip().splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            relative = parts[2].strip()
            if relative:
                cgroup_path = Path(f"/sys/fs/cgroup{relative}")
                if cgroup_path.exists():
                    return cgroup_path
    return None


def _read_cgroup_io_stat(cgroup_path: Path) -> dict[str, int] | None:
    """Read io.stat from a cgroup v2 path, aggregating across all devices.

    Format per line: ``<major>:<minor> rbytes=N wbytes=N rios=N wios=N ...``
    Returns total read_bytes and write_bytes summed across all devices.
    """
    io_stat = cgroup_path / "io.stat"
    try:
        text = io_stat.read_text(encoding="utf-8")
    except (PermissionError, FileNotFoundError, OSError):
        return None
    total_read = 0
    total_write = 0
    found = False
    for line in text.strip().splitlines():
        for field in line.split():
            if field.startswith("rbytes="):
                try:
                    total_read += int(field.split("=", 1)[1])
                    found = True
                except ValueError:
                    pass
            elif field.startswith("wbytes="):
                try:
                    total_write += int(field.split("=", 1)[1])
                    found = True
                except ValueError:
                    pass
    return {"read_bytes": total_read, "write_bytes": total_write} if found else None


def _read_cgroup_pids(cgroup_path: Path) -> list[int]:
    """Read all PIDs from a cgroup's cgroup.procs file."""
    procs = cgroup_path / "cgroup.procs"
    try:
        text = procs.read_text(encoding="utf-8")
    except (PermissionError, FileNotFoundError, OSError):
        return []
    pids: list[int] = []
    for line in text.strip().splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


def _read_pid_starttime(pid: int) -> int | None:
    """Read process start time (clock ticks since boot) from /proc/<pid>/stat."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (PermissionError, FileNotFoundError, OSError):
        return None
    # Field 22 (1-indexed) is starttime. comm (field 2) can contain ')'.
    close_paren = text.rfind(")")
    if close_paren < 0:
        return None
    fields = text[close_paren + 1:].split()
    # After comm: fields[0]=state, fields[1]=ppid, ..., fields[19]=starttime
    if len(fields) > 19:
        try:
            return int(fields[19])
        except ValueError:
            pass
    return None


def _read_pid_context_switches(pid: int) -> int | None:
    """Read total (voluntary + nonvoluntary) context switches for one PID."""
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (PermissionError, FileNotFoundError, OSError):
        return None
    total = 0
    found = False
    for line in text.splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            if key in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
                try:
                    total += int(parts[1].strip())
                    found = True
                except ValueError:
                    pass
    return total if found else None


def _aggregate_context_switches(pids: list[int]) -> int | None:
    """Sum voluntary + nonvoluntary context switches across all PIDs."""
    total = 0
    found = False
    for pid in pids:
        ctxt = _read_pid_context_switches(pid)
        if ctxt is not None:
            total += ctxt
            found = True
    return total if found else None


def _read_io_via_exec(
    container_id: str, *, executable: str, timeout_s: float = 5.0,
) -> dict[str, int] | None:
    """Fallback: read container I/O from inside the container.

    Tries cgroup io.stat inside the container namespace first (monotonic,
    includes exited processes). Falls back to summing /proc/*/io (non-
    monotonic but better than nothing).
    """
    # Prefer cgroup io.stat inside container (monotonic)
    script = (
        "import os\n"
        "r=w=0;found=False\n"
        "try:\n"
        "  for l in open('/sys/fs/cgroup/io.stat'):\n"
        "    for f in l.split():\n"
        "      if f.startswith('rbytes='): r+=int(f.split('=')[1]);found=True\n"
        "      elif f.startswith('wbytes='): w+=int(f.split('=')[1]);found=True\n"
        "except: pass\n"
        "if found: print(r,w)\n"
        "else:\n"
        "  import glob\n"
        "  for p in glob.glob('/proc/[0-9]*/io'):\n"
        "    try:\n"
        "      d={}\n"
        "      for l in open(p):\n"
        "        k,v=l.split(':',1)\n"
        "        d[k.strip()]=int(v.strip())\n"
        "      r+=d.get('read_bytes',0);w+=d.get('write_bytes',0)\n"
        "    except: pass\n"
        "  print(r,w)\n"
    )
    try:
        result = subprocess.run(
            [executable, "exec", container_id, "python3", "-c", script],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        if result.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    parts = result.stdout.strip().split()
    if len(parts) == 2:
        try:
            return {"read_bytes": int(parts[0]), "write_bytes": int(parts[1])}
        except ValueError:
            pass
    return None


def _read_ctxt_via_exec(
    container_id: str, *, executable: str, timeout_s: float = 5.0,
) -> int | None:
    """Fallback: sum context switches from all processes inside the container."""
    script = (
        "import glob\n"
        "t=0\n"
        "for p in glob.glob('/proc/[0-9]*/status'):\n"
        "  try:\n"
        "    for l in open(p):\n"
        "      if 'ctxt_switches' in l:\n"
        "        t+=int(l.split(':')[1].strip())\n"
        "  except: pass\n"
        "print(t)\n"
    )
    try:
        result = subprocess.run(
            [executable, "exec", container_id, "python3", "-c", script],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        if result.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _parse_pipe_stats(raw: str) -> dict[str, Any] | None:
    """Parse pipe-delimited stats output (3-field legacy or 4-field with NetIO)."""
    parts = (raw or "").strip().split("|")
    if len(parts) < 3:
        return None
    now = datetime.now(tz=timezone.utc)
    sample: dict[str, Any] = {
        "timestamp": now.isoformat().replace("+00:00", ""),
        "epoch": now.timestamp(),
        "mem_usage": parts[0],
        "mem_percent": parts[1],
        "cpu_percent": parts[2],
    }
    if len(parts) >= 4:
        sample["net_io"] = parts[3]
    return sample


def _parse_memory_mb(mem_usage: str) -> float | None:
    if not mem_usage:
        return None
    left = mem_usage.split("/")[0].strip()
    try:
        if left.endswith("GiB"):
            return float(left[:-3]) * 1024
        if left.endswith("MiB"):
            return float(left[:-3])
        if left.endswith("KiB"):
            return float(left[:-3]) / 1024
        if left.endswith("GB"):
            return float(left[:-2]) * 1000
        if left.endswith("MB"):
            return float(left[:-2])
        if left.endswith("KB"):
            return float(left[:-2]) / 1000
        if left.endswith("kB"):
            return float(left[:-2]) / 1000
        if left.endswith("B"):
            return float(left[:-1]) / (1024 * 1024)
    except ValueError:
        return None
    return None


def _parse_percent(value: str) -> float | None:
    try:
        return float(value.replace("%", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_net_io_bytes(net_io: str) -> tuple[float | None, float | None]:
    """Parse NetIO string like '1.5kB / 2.3MB' into (rx_bytes, tx_bytes)."""
    if not net_io or "/" not in net_io:
        return None, None
    parts = net_io.split("/", 1)
    return _parse_size_bytes(parts[0].strip()), _parse_size_bytes(parts[1].strip())


def _parse_size_bytes(s: str) -> float | None:
    """Parse a human-readable size string to bytes (SI and binary units)."""
    s = s.strip()
    if not s:
        return None
    _units = {
        "TiB": 1024**4, "GiB": 1024**3, "MiB": 1024**2, "KiB": 1024,
        "TB": 1e12, "GB": 1e9, "MB": 1e6, "KB": 1e3, "kB": 1e3, "B": 1,
    }
    for unit, multiplier in _units.items():
        if s.endswith(unit):
            try:
                return float(s[:-len(unit)].strip()) * multiplier
            except ValueError:
                return None
    return None


def _minmaxavg(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "avg": 0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute resources.json summary from a sample list.

    Extends the AgentCGroup-compatible format with disk I/O, network I/O,
    and context switch fields.
    """
    if not samples:
        return {
            "sample_count": 0,
            "duration_seconds": 0,
            "memory_mb": {"min": 0, "max": 0, "avg": 0},
            "cpu_percent": {"min": 0, "max": 0, "avg": 0},
            "memory_total_mb_s": {"min": 0, "max": 0, "avg": 0},
            "memory_read_mb_s": {"min": 0, "max": 0, "avg": 0},
            "memory_write_mb_s": {"min": 0, "max": 0, "avg": 0},
            "memory_bandwidth_available": False,
            "memory_bandwidth_source": None,
            "memory_bandwidth_reason": None,
            "disk_read_mb": {"min": 0, "max": 0, "avg": 0, "delta": 0},
            "disk_write_mb": {"min": 0, "max": 0, "avg": 0, "delta": 0},
            "net_rx_mb": {"min": 0, "max": 0, "avg": 0, "delta": 0},
            "net_tx_mb": {"min": 0, "max": 0, "avg": 0, "delta": 0},
            "context_switches": {"min": 0, "max": 0, "avg": 0, "delta": 0},
        }

    mem_values: list[float] = []
    cpu_values: list[float] = []
    mem_total_values: list[float] = []
    mem_read_values: list[float] = []
    mem_write_values: list[float] = []
    disk_read_values: list[float] = []
    disk_write_values: list[float] = []
    net_rx_values: list[float] = []
    net_tx_values: list[float] = []
    ctxt_values: list[float] = []
    mem_bw_available = False
    mem_bw_source: str | None = None
    mem_bw_reason: str | None = None

    for sample in samples:
        mem_mb = _parse_memory_mb(sample.get("mem_usage", ""))
        if mem_mb is not None:
            mem_values.append(mem_mb)
        cpu_val = _parse_percent(sample.get("cpu_percent", ""))
        if cpu_val is not None:
            cpu_values.append(cpu_val)
        mem_total = sample.get("memory_total_mb_s")
        if mem_total is not None:
            mem_total_values.append(float(mem_total))
        mem_read = sample.get("memory_read_mb_s")
        if mem_read is not None:
            mem_read_values.append(float(mem_read))
        mem_write = sample.get("memory_write_mb_s")
        if mem_write is not None:
            mem_write_values.append(float(mem_write))
        if "memory_bandwidth_available" in sample:
            mem_bw_available = bool(sample.get("memory_bandwidth_available"))
        if sample.get("memory_bandwidth_source") is not None:
            mem_bw_source = str(sample.get("memory_bandwidth_source"))
        if sample.get("memory_bandwidth_reason") is not None:
            mem_bw_reason = str(sample.get("memory_bandwidth_reason"))

        # Disk I/O (from cgroup io.stat, stored as bytes in sample)
        rb = sample.get("disk_read_bytes")
        if rb is not None:
            disk_read_values.append(rb / (1024 * 1024))
        wb = sample.get("disk_write_bytes")
        if wb is not None:
            disk_write_values.append(wb / (1024 * 1024))

        # Network I/O (from stats — decimal units, so divide by 1e6)
        net_rx = sample.get("net_rx_bytes")
        if net_rx is not None:
            net_rx_values.append(net_rx / 1_000_000)
        net_tx = sample.get("net_tx_bytes")
        if net_tx is not None:
            net_tx_values.append(net_tx / 1_000_000)

        # Context switches (from cgroup pids aggregate)
        ctxt = sample.get("context_switches")
        if ctxt is not None:
            ctxt_values.append(float(ctxt))

    duration = 0.0
    if len(samples) > 1:
        duration = float(samples[-1]["epoch"]) - float(samples[0]["epoch"])

    def _delta(values: list[float]) -> float:
        if len(values) < 2:
            return 0
        return values[-1] - values[0]

    return {
        "sample_count": len(samples),
        "duration_seconds": duration,
        "memory_mb": _minmaxavg(mem_values),
        "cpu_percent": _minmaxavg(cpu_values),
        "memory_total_mb_s": _minmaxavg(mem_total_values),
        "memory_read_mb_s": _minmaxavg(mem_read_values),
        "memory_write_mb_s": _minmaxavg(mem_write_values),
        "memory_bandwidth_available": mem_bw_available,
        "memory_bandwidth_source": mem_bw_source,
        "memory_bandwidth_reason": mem_bw_reason,
        "disk_read_mb": {**_minmaxavg(disk_read_values), "delta": _delta(disk_read_values)},
        "disk_write_mb": {**_minmaxavg(disk_write_values), "delta": _delta(disk_write_values)},
        "net_rx_mb": {**_minmaxavg(net_rx_values), "delta": _delta(net_rx_values)},
        "net_tx_mb": {**_minmaxavg(net_tx_values), "delta": _delta(net_tx_values)},
        "context_switches": {**_minmaxavg(ctxt_values), "delta": _delta(ctxt_values)},
    }


class ContainerStatsSampler(threading.Thread):
    """Background thread that samples container stats + cgroup I/O metrics.

    Disk I/O is read from cgroup v2 ``io.stat`` which aggregates ALL
    processes in the container (including ``docker exec`` workers).
    Context switches are summed across all PIDs in the cgroup.

    Usage::

        sampler = ContainerStatsSampler(container_id="abc123", interval_s=1.0)
        sampler.start()
        # ... agent runs ...
        samples = sampler.stop()
        summary = summarize_samples(samples)
    """

    def __init__(
        self,
        container_id: str,
        *,
        interval_s: float = 1.0,
        executable: str = "podman",
        subprocess_timeout_s: float = 5.0,
    ) -> None:
        super().__init__(daemon=True, name=f"stats-{container_id[:12]}")
        self.container_id = container_id
        self.interval_s = interval_s
        self.executable = executable
        self.subprocess_timeout_s = subprocess_timeout_s
        self._stop_event = threading.Event()
        self._samples: list[dict[str, Any]] = []
        self._cgroup_path: Path | None = None
        self._io_mode: str | None = None  # "cgroup", "exec", or None
        # Context switches keyed by (pid, starttime) for stable identity across PID reuse
        self._pid_ctxt: dict[tuple[int, int], int] = {}
        # High-water marks for exec-mode disk I/O (non-monotonic source)
        self._io_hwm_read: int = 0
        self._io_hwm_write: int = 0

    def _ensure_io_source(self) -> None:
        """Resolve and cache the I/O data source on first call.

        Tries cgroup v2 io.stat first (host-side, zero overhead).
        Falls back to exec-based aggregation if cgroup unavailable.
        """
        if self._io_mode is not None:
            return
        pid = _resolve_container_pid(self.container_id, executable=self.executable)
        if pid is None:
            logger.debug("Could not resolve host PID for %s", self.container_id[:12])
            self._io_mode = "exec"
            return
        cgroup_path = _resolve_cgroup_path(pid)
        if cgroup_path is not None:
            try:
                (cgroup_path / "io.stat").read_text(encoding="utf-8")
                self._cgroup_path = cgroup_path
                self._io_mode = "cgroup"
                return
            except (PermissionError, FileNotFoundError, OSError):
                pass
        logger.info(
            "Cgroup I/O unavailable for %s; falling back to exec-based aggregation",
            self.container_id[:12],
        )
        self._io_mode = "exec"

    def _sample_io(self, sample: dict[str, Any]) -> None:
        """Attach disk I/O and context switch data to a sample dict."""
        self._ensure_io_source()

        io_data: dict[str, int] | None = None
        if self._io_mode == "cgroup" and self._cgroup_path is not None:
            io_data = _read_cgroup_io_stat(self._cgroup_path)
            # Context switches: update per-PID high-water marks so exited
            # processes keep their last-known counts in the total.
            pids = _read_cgroup_pids(self._cgroup_path)
            if pids:
                for pid in pids:
                    starttime = _read_pid_starttime(pid)
                    ctxt = _read_pid_context_switches(pid)
                    if ctxt is not None and starttime is not None:
                        self._pid_ctxt[(pid, starttime)] = ctxt
                if self._pid_ctxt:
                    sample["context_switches"] = sum(self._pid_ctxt.values())
        elif self._io_mode == "exec":
            # Halve timeout per call so total stays within stop() budget
            half_timeout = self.subprocess_timeout_s / 2
            io_data = _read_io_via_exec(
                self.container_id,
                executable=self.executable,
                timeout_s=half_timeout,
            )
            ctxt = _read_ctxt_via_exec(
                self.container_id,
                executable=self.executable,
                timeout_s=half_timeout,
            )
            if ctxt is not None:
                # Exec-mode: aggregate count, use high-water mark (non-monotonic source)
                self._pid_ctxt[(0, 0)] = max(self._pid_ctxt.get((0, 0), 0), ctxt)
                sample["context_switches"] = sum(self._pid_ctxt.values())

        if io_data:
            rb = io_data.get("read_bytes", 0)
            wb = io_data.get("write_bytes", 0)
            if self._io_mode == "exec":
                # High-water mark: exec-based I/O is non-monotonic
                self._io_hwm_read = max(self._io_hwm_read, rb)
                self._io_hwm_write = max(self._io_hwm_write, wb)
                rb = self._io_hwm_read
                wb = self._io_hwm_write
            sample["disk_read_bytes"] = rb
            sample["disk_write_bytes"] = wb

        # Network I/O from stats output
        net_io = sample.get("net_io")
        if net_io:
            rx, tx = _parse_net_io_bytes(net_io)
            if rx is not None:
                sample["net_rx_bytes"] = rx
            if tx is not None:
                sample["net_tx_bytes"] = tx
        attach_host_memory_bandwidth(sample, interval_s=self.interval_s)

    def run(self) -> None:
        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            try:
                result = subprocess.run(
                    [
                        self.executable,
                        "stats",
                        "--no-stream",
                        "--format",
                        _STATS_FORMAT,
                        self.container_id,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.subprocess_timeout_s,
                    check=False,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                break
            if result.returncode != 0:
                break
            sample = _parse_pipe_stats(result.stdout)
            if sample is not None:
                self._sample_io(sample)
                self._samples.append(sample)
            elapsed = time.monotonic() - tick_start
            remainder = max(0.0, self.interval_s - elapsed)
            if self._stop_event.wait(remainder):
                break

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=self.subprocess_timeout_s + self.interval_s + 1.0)
        return list(self._samples)

    def get_summary(self) -> dict[str, Any]:
        return summarize_samples(self._samples)
