"""Memory bandwidth monitoring via PMU (perf-based).

Detects HiSilicon DDRC PMU (and other PMU devices), and provides a
shared per-memory-domain Collector that reads bandwidth counters via
``perf stat``.

When PMU is unavailable, all bandwidth_utilization values remain None
and the cost model degrades gracefully (memory_cost = 0.0).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# sysfs path for PMU devices
_PMU_DEVICES = Path("/sys/bus/event_source/devices")

# Default sustainable bandwidth per NUMA node in GiB/s.
# These are conservative defaults; real values should come from config.
DEFAULT_SUSTAINABLE_BANDWIDTH_GIB_S = 50.0  # Per NUMA domain

# Default sampling interval for bandwidth collector
DEFAULT_BW_SAMPLE_INTERVAL = 1.0  # 1 second


@dataclass
class MemoryDomainConfig:
    """Configuration for a single memory domain (NUMA node / SCCL)."""

    numa_node: int
    sustainable_bandwidth_gib_s: float = 50.0
    # PMU event names for read/write bandwidth (perf syntax)
    pmu_read_event: str | None = None
    pmu_write_event: str | None = None
    # Or a combined bandwidth event
    pmu_combined_event: str | None = None


@dataclass
class BandwidthSnapshot:
    """A single bandwidth measurement for a memory domain."""

    numa_node: int
    timestamp_s: float
    read_gib_s: float = 0.0
    write_gib_s: float = 0.0
    total_gib_s: float = 0.0
    utilization: Optional[float] = None  # current / sustainable
    available: bool = False  # True if PMU data is available


class BandwidthCollector:
    """Shared per-memory-domain bandwidth collector using perf stat.

    Each NUMA node gets at most one collector instance.  The collector
    runs a background thread that periodically samples the DDRC PMU
    counters and computes bandwidth utilization.

    When PMU is unavailable, this silently produces BandwidthSnapshot
    with available=False.
    """

    # Class-level registry: numa_node -> BandwidthCollector
    _collectors: dict[int, "BandwidthCollector"] = {}

    def __init__(
        self,
        numa_node: int,
        config: MemoryDomainConfig | None = None,
        sample_interval: float = DEFAULT_BW_SAMPLE_INTERVAL,
    ) -> None:
        self._numa_node = numa_node
        self._config = config or MemoryDomainConfig(numa_node=numa_node)
        self._sample_interval = sample_interval
        self._latest: BandwidthSnapshot = BandwidthSnapshot(
            numa_node=numa_node, timestamp_s=0.0, available=False,
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @classmethod
    def get(cls, numa_node: int, config: MemoryDomainConfig | None = None) -> "BandwidthCollector":
        """Get or create the shared collector for a NUMA node."""
        if numa_node not in cls._collectors:
            cls._collectors[numa_node] = cls(numa_node, config)
        return cls._collectors[numa_node]

    @classmethod
    def start_all(cls, configs: list[MemoryDomainConfig] | None = None) -> None:
        """Start collectors for all configured memory domains."""
        if configs is None:
            # Try to auto-detect from sysfs
            configs = _auto_detect_memory_domains()
        for cfg in configs:
            collector = cls.get(cfg.numa_node, cfg)
            collector.start()

    @classmethod
    def stop_all(cls) -> None:
        """Stop all running collectors."""
        for collector in list(cls._collectors.values()):
            collector.stop()

    @classmethod
    def snapshot_all(cls) -> dict[int, BandwidthSnapshot]:
        """Get the latest snapshot from all collectors."""
        return {
            node: collector.latest
            for node, collector in cls._collectors.items()
        }

    @property
    def latest(self) -> BandwidthSnapshot:
        with self._lock:
            return self._latest

    def start(self) -> None:
        """Start the background sampling thread."""
        if self._running:
            return
        # Check PMU availability first
        if not _pmu_available():
            logger.info(
                "PMU not available for NUMA %d, bandwidth monitoring disabled",
                self._numa_node,
            )
            # Mark as unavailable but still allow start (graceful degradation)
            with self._lock:
                self._latest = BandwidthSnapshot(
                    numa_node=self._numa_node,
                    timestamp_s=time.monotonic(),
                    available=False,
                )
            return

        if not self._config.pmu_read_event and not self._config.pmu_write_event \
                and not self._config.pmu_combined_event:
            logger.info(
                "No PMU events configured for NUMA %d, bandwidth monitoring disabled",
                self._numa_node,
            )
            return

        self._running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            "Bandwidth collector started for NUMA %d (interval=%.1fs)",
            self._numa_node,
            self._sample_interval,
        )

    def stop(self) -> None:
        """Stop the background sampling thread."""
        self._running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        """Main sampling loop using perf stat."""
        while not self._stop.is_set():
            snapshot = self._sample_perf()
            with self._lock:
                self._latest = snapshot
            self._stop.wait(self._sample_interval)

    def _sample_perf(self) -> BandwidthSnapshot:
        """Run perf stat for one interval and parse bandwidth.

        Uses ``perf stat`` with the configured PMU events.
        Falls back gracefully on any error.
        """
        now = time.monotonic()

        # Build perf events
        events: list[str] = []
        if self._config.pmu_combined_event:
            events.append(self._config.pmu_combined_event)
        if self._config.pmu_read_event:
            events.append(self._config.pmu_read_event)
        if self._config.pmu_write_event:
            events.append(self._config.pmu_write_event)

        if not events:
            return BandwidthSnapshot(
                numa_node=self._numa_node,
                timestamp_s=now,
                available=False,
            )

        # Build perf stat command (consistent with memory_bandwidth.py).
        # All PMU events are passed as a comma-separated list to a single -e,
        # avoiding the fragile index-based insert that produced duplicate -e
        # flags when more than one event was configured.
        cmd = [
            "perf", "stat",
            "-a",  # system-wide
            "-x", ",",  # CSV output
            "-e", ",".join(events),
            "-I", str(int(self._sample_interval * 1000)),  # interval in ms
            "--", "sleep", str(self._sample_interval),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(self._sample_interval * 2 + 10),
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("perf stat failed for NUMA %d: %s", self._numa_node, e)
            return BandwidthSnapshot(
                numa_node=self._numa_node,
                timestamp_s=now,
                available=False,
            )

        if result.returncode != 0:
            logger.debug(
                "perf stat returned %d for NUMA %d: %s",
                result.returncode,
                self._numa_node,
                result.stderr[:200] if result.stderr else "",
            )
            return BandwidthSnapshot(
                numa_node=self._numa_node,
                timestamp_s=now,
                available=False,
            )

        # Parse CSV output
        # Format: timestamp,count,unit,event
        read_val = 0.0
        write_val = 0.0
        combined_val = 0.0

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                count = float(parts[1])
            except ValueError:
                continue
            event_str = parts[3] if len(parts) > 3 else ""

            # Heuristic: match event to read/write/combined
            if self._config.pmu_read_event and self._config.pmu_read_event in event_str:
                read_val = count
            elif self._config.pmu_write_event and self._config.pmu_write_event in event_str:
                write_val = count
            elif self._config.pmu_combined_event and self._config.pmu_combined_event in event_str:
                combined_val = count

        # Convert to GiB/s
        # perf stat counts are per-interval; divide by interval for rate
        interval_s = self._sample_interval
        if combined_val > 0:
            total_gib_s = combined_val / (1024**3) / interval_s
        else:
            read_gib_s = read_val / (1024**3) / interval_s if read_val > 0 else 0.0
            write_gib_s = write_val / (1024**3) / interval_s if write_val > 0 else 0.0
            total_gib_s = read_gib_s + write_gib_s

        sustainable = self._config.sustainable_bandwidth_gib_s
        utilization = min(total_gib_s / sustainable, 1.0) if sustainable > 0 else None

        return BandwidthSnapshot(
            numa_node=self._numa_node,
            timestamp_s=now,
            read_gib_s=read_val / (1024**3) / interval_s if combined_val == 0 and read_val > 0 else 0.0,
            write_gib_s=write_val / (1024**3) / interval_s if combined_val == 0 and write_val > 0 else 0.0,
            total_gib_s=total_gib_s,
            utilization=utilization,
            available=True,
        )


def _pmu_available() -> bool:
    """Check if any PMU devices are available via sysfs."""
    if not _PMU_DEVICES.exists():
        return False
    try:
        entries = list(_PMU_DEVICES.iterdir())
        return len(entries) > 0
    except OSError:
        return False


def _detect_ddrc_pmu() -> list[str]:
    """Detect HiSilicon DDRC PMU device names.

    Returns list of PMU device names (e.g., ['hisi_sccl1_ddrc0', ...]).
    """
    if not _PMU_DEVICES.exists():
        return []
    ddrc_devices: list[str] = []
    for entry in sorted(_PMU_DEVICES.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        # HiSilicon DDRC PMU: hisi_sccl<N>_ddrc<M>
        if "ddrc" in name.lower():
            ddrc_devices.append(name)
        # Also check for uncore PMU with DDR
        if "uncore" in name.lower() and "ddr" in name.lower():
            ddrc_devices.append(name)
    return ddrc_devices


def _detect_perf_ddr_events() -> list[str]:
    """Use perf list to find DDR bandwidth events.

    Returns list of event strings suitable for perf stat -e.
    """
    try:
        result = subprocess.run(
            ["perf", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    events: list[str] = []
    for line in result.stdout.splitlines():
        line_lower = line.lower()
        if "ddr" in line_lower or "ddrc" in line_lower:
            # Try to extract event name
            # Format: "  hisi_sccl1_ddrc0/flux_rd/           [Kernel PMU event]"
            parts = line.strip().split()
            if parts:
                event = parts[0].rstrip("/")
                if "/" in event:
                    events.append(event)

    return events


def _auto_detect_memory_domains() -> list[MemoryDomainConfig]:
    """Auto-detect memory domain configurations from sysfs and perf.

    Returns a list of MemoryDomainConfig, one per NUMA node with
    detected PMU events.
    """
    configs: list[MemoryDomainConfig] = []

    # Discover NUMA nodes
    node_dir = Path("/sys/devices/system/node")
    numa_nodes: list[int] = []
    if node_dir.exists():
        for entry in sorted(node_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("node"):
                try:
                    numa_nodes.append(int(entry.name[4:]))
                except ValueError:
                    continue

    if not numa_nodes:
        numa_nodes = [0]  # Assume single node

    # Try to detect DDR PMU events
    ddrc_devices = _detect_ddrc_pmu()
    perf_events = _detect_perf_ddr_events()

    for node in numa_nodes:
        cfg = MemoryDomainConfig(numa_node=node)

        # Try to map DDRC PMU to NUMA node
        # On Kunpeng, SCCL maps to NUMA; DDRC PMU name often encodes SCCL
        for dev in ddrc_devices:
            # e.g., hisi_sccl1_ddrc0 -> SCCL 1
            if f"sccl{node}" in dev or f"sccl{node}" in dev.lower():
                # Use flux_rd and flux_wr events
                cfg.pmu_read_event = f"{dev}/flux_rd/"
                cfg.pmu_write_event = f"{dev}/flux_wr/"
                break

        # Fallback: use perf list events
        if not cfg.pmu_read_event and perf_events:
            # Assign first available read/write events
            for evt in perf_events:
                if "rd" in evt.lower() and not cfg.pmu_read_event:
                    cfg.pmu_read_event = evt
                elif "wr" in evt.lower() and not cfg.pmu_write_event:
                    cfg.pmu_write_event = evt

        # Use default sustainable bandwidth
        cfg.sustainable_bandwidth_gib_s = DEFAULT_SUSTAINABLE_BANDWIDTH_GIB_S

        configs.append(cfg)

    return configs


def load_memory_domain_configs(path: str | Path) -> list[MemoryDomainConfig]:
    """Load memory domain configurations from a JSON file.

    Expected format:
    [
      {
        "numa_node": 0,
        "sustainable_bandwidth_gib_s": 80.0,
        "pmu_read_event": "hisi_sccl0_ddrc0/flux_rd/",
        "pmu_write_event": "hisi_sccl0_ddrc0/flux_wr/"
      }
    ]
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Memory domain config must be a JSON array, got {type(raw).__name__}")

    configs: list[MemoryDomainConfig] = []
    for entry in raw:
        configs.append(MemoryDomainConfig(
            numa_node=int(entry["numa_node"]),
            sustainable_bandwidth_gib_s=float(
                entry.get("sustainable_bandwidth_gib_s", DEFAULT_SUSTAINABLE_BANDWIDTH_GIB_S)
            ),
            pmu_read_event=entry.get("pmu_read_event"),
            pmu_write_event=entry.get("pmu_write_event"),
            pmu_combined_event=entry.get("pmu_combined_event"),
        ))
    return configs


def get_bandwidth_utilization(numa_node: int) -> Optional[float]:
    """Get the current bandwidth utilization for a NUMA node.

    Returns None if PMU is unavailable or collector not started.
    """
    collector = BandwidthCollector._collectors.get(numa_node)
    if collector is None:
        return None
    return collector.latest.utilization
