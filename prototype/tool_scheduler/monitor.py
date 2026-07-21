"""Continuous monitoring of process tree at 500ms intervals.

Reuses sample_process_tree from prototype.tool_profiler.sampler for
per-sample data acquisition.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from prototype.tool_profiler.sampler import sample_process_tree, SamplePoint

logger = logging.getLogger(__name__)

# Default sampling interval
DEFAULT_SAMPLE_INTERVAL = 0.5  # 500ms


@dataclass
class MonitorSample:
    """A single monitoring sample with derived effective_cores."""

    timestamp_s: float
    elapsed_s: float
    root_pid: int

    process_count: int
    thread_count: int

    cpu_user_time_s: float
    cpu_system_time_s: float
    cpu_total_time_s: float

    rss_bytes: int

    read_bytes: Optional[int]
    write_bytes: Optional[int]

    # Derived: effective cores computed as delta from previous sample
    effective_cores: float = 0.0

    @classmethod
    def from_sample_point(
        cls,
        sp: SamplePoint,
        prev_cpu_total: Optional[float] = None,
        prev_elapsed: Optional[float] = None,
    ) -> "MonitorSample":
        """Create a MonitorSample from a SamplePoint, computing effective_cores.

        Args:
            sp: The raw sample point from the process tree.
            prev_cpu_total: Previous sample's cumulative CPU time.
            prev_elapsed: Previous sample's elapsed time.

        Returns:
            MonitorSample with effective_cores computed via delta method.
        """
        effective_cores = 0.0
        if prev_cpu_total is not None and prev_elapsed is not None:
            cpu_delta = max(sp.cpu_total_time_s - prev_cpu_total, 0.0)
            wall_delta = sp.elapsed_s - prev_elapsed
            if wall_delta > 0:
                effective_cores = cpu_delta / wall_delta

        return cls(
            timestamp_s=sp.timestamp_s,
            elapsed_s=sp.elapsed_s,
            root_pid=sp.root_pid,
            process_count=sp.process_count,
            thread_count=sp.thread_count,
            cpu_user_time_s=sp.cpu_user_time_s,
            cpu_system_time_s=sp.cpu_system_time_s,
            cpu_total_time_s=sp.cpu_total_time_s,
            rss_bytes=sp.rss_bytes,
            read_bytes=sp.read_bytes,
            write_bytes=sp.write_bytes,
            effective_cores=effective_cores,
        )


class Monitor:
    """Continuously samples the process tree at a fixed interval.

    Runs a daemon thread that calls sample_process_tree() every
    sample_interval seconds and stores derived MonitorSamples.
    """

    def __init__(
        self,
        root_pid: int,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    ) -> None:
        self._root_pid = root_pid
        self._sample_interval = sample_interval
        self._samples: list[MonitorSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_mono: float = 0.0

    @property
    def samples(self) -> list[MonitorSample]:
        """Return a copy of all collected samples (thread-safe)."""
        with self._lock:
            return list(self._samples)

    @property
    def latest(self) -> Optional[MonitorSample]:
        """Return the most recent sample, or None."""
        with self._lock:
            return self._samples[-1] if self._samples else None

    def start(self) -> None:
        """Start the monitoring thread."""
        self._start_mono = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the monitoring thread to stop and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        """Main monitoring loop."""
        prev_cpu_total: Optional[float] = None
        prev_elapsed: Optional[float] = None

        while not self._stop.is_set():
            elapsed = time.monotonic() - self._start_mono
            sp = sample_process_tree(self._root_pid, elapsed)

            if sp is not None:
                ms = MonitorSample.from_sample_point(
                    sp,
                    prev_cpu_total=prev_cpu_total,
                    prev_elapsed=prev_elapsed,
                )
                with self._lock:
                    self._samples.append(ms)
                prev_cpu_total = sp.cpu_total_time_s
                prev_elapsed = sp.elapsed_s

            self._stop.wait(self._sample_interval)
