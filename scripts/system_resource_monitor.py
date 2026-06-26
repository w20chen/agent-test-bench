#!/usr/bin/env python3
"""System-wide resource monitor for simulation experiments.

Samples system CPU%, memory%, disk I/O, network I/O, and Docker container
count at a configurable interval and writes a JSONL log.  Runs as a
background process; stops on SIGTERM/SIGINT or when a stop-file appears.

Usage::

    python scripts/system_resource_monitor.py --output /path/to/system_resources.jsonl &
    MONITOR_PID=$!
    # ... run experiment ...
    kill $MONITOR_PID
    # or: touch /path/to/stop

Output format (one JSON record per sample)::

    {
      "ts": 1719000000.123,
      "cpu_percent": 45.2,
      "cpu_count": 320,
      "mem_percent": 62.3,
      "mem_used_gb": 800.4,
      "mem_total_gb": 1008.0,
      "disk_read_mb": 123456.7,
      "disk_write_mb": 78901.2,
      "net_rx_mb": 5000.1,
      "net_tx_mb": 3000.4,
      "container_count": 320,
      "load_1m": 180.5,
      "load_5m": 150.2,
      "load_15m": 120.8
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_container_count() -> int | None:
    """Return the number of running Docker/Podman containers, or None."""
    for exe in ("docker", "podman"):
        try:
            result = subprocess.run(
                [exe, "ps", "-q"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = [ln for ln in result.stdout.strip().split("\n") if ln]
                return len(lines)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def run_monitor(
    output_path: Path,
    *,
    interval_s: float = 1.0,
    stop_file: Path | None = None,
) -> None:
    """Sample system resources and write JSONL until stopped.

    Args:
        output_path: Path to write the JSONL log.
        interval_s: Sampling interval in seconds.
        stop_file: If provided, monitor exits when this file exists.
    """
    import psutil

    # Accumulate disk / network counters from the first sample so we
    # record cumulative totals, not deltas.  psutil.disk_io_counters() and
    # net_io_counters() return counters since boot.
    first_disk = psutil.disk_io_counters()
    first_net = psutil.net_io_counters()
    _first_disk_ok = first_disk is not None
    _first_net_ok = first_net is not None
    if first_disk is None:
        first_disk_read = 0.0
        first_disk_write = 0.0
    else:
        first_disk_read = first_disk.read_bytes
        first_disk_write = first_disk.write_bytes
    if first_net is None:
        first_net_rx = 0.0
        first_net_tx = 0.0
    else:
        first_net_rx = first_net.bytes_recv
        first_net_tx = first_net.bytes_sent

    _stop_flag = False

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal _stop_flag
        _stop_flag = True
        logger.info("Received signal %d, stopping monitor...", signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "System resource monitor started: output=%s interval=%.1fs",
        output_path, interval_s,
    )

    sample_count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        while not _stop_flag:
            if stop_file is not None and stop_file.exists():
                logger.info("Stop file %s detected, stopping monitor.", stop_file)
                break

            try:
                ts = time.time()
                cpu = psutil.cpu_percent(interval=None)
                cpu_count = psutil.cpu_count() or 0
                mem = psutil.virtual_memory()
                disk = psutil.disk_io_counters()
                net = psutil.net_io_counters()
                load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
                container_count = _get_container_count()

                # Disk I/O — cumulative bytes since boot, offset by first sample
                disk_read = (
                    (disk.read_bytes - first_disk_read) / (1024 * 1024)
                    if disk and _first_disk_ok
                    else 0.0
                )
                disk_write = (
                    (disk.write_bytes - first_disk_write) / (1024 * 1024)
                    if disk and _first_disk_ok
                    else 0.0
                )

                # Network I/O — cumulative bytes since boot, offset by first sample
                net_rx = (
                    (net.bytes_recv - first_net_rx) / (1024 * 1024)
                    if net and _first_net_ok
                    else 0.0
                )
                net_tx = (
                    (net.bytes_sent - first_net_tx) / (1024 * 1024)
                    if net and _first_net_ok
                    else 0.0
                )

                record = {
                    "ts": ts,
                    "cpu_percent": cpu,
                    "cpu_count": cpu_count,
                    "mem_percent": mem.percent,
                    "mem_used_gb": mem.used / (1024**3),
                    "mem_total_gb": mem.total / (1024**3),
                    "disk_read_mb": disk_read,
                    "disk_write_mb": disk_write,
                    "net_rx_mb": net_rx,
                    "net_tx_mb": net_tx,
                    "container_count": container_count,
                    "load_1m": load[0],
                    "load_5m": load[1],
                    "load_15m": load[2],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                sample_count += 1
            except Exception:
                logger.exception("Error during sampling (sample %d)", sample_count)

            time.sleep(interval_s)

    logger.info(
        "Monitor stopped. %d samples written to %s",
        sample_count, output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="System-wide resource monitor for simulation experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSONL system resource log.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="Optional stop-file path. Monitor exits when this file exists.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    run_monitor(
        output_path=args.output,
        interval_s=args.interval,
        stop_file=args.stop_file,
    )


if __name__ == "__main__":
    main()
