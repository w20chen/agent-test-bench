#!/usr/bin/env python3
"""CPU serial workload: single-threaded compute for a configurable duration.

Usage:
    python workloads/cpu_serial.py [--seconds 5]
"""

from __future__ import annotations

import argparse
import time


def cpu_burn(duration: float) -> None:
    """Perform single-threaded arithmetic for `duration` seconds."""
    end = time.monotonic() + duration
    x = 0.0
    while time.monotonic() < end:
        x = (x + 1.0) * 1.00001 % 1e9
    # Prevent optimizer from removing the loop
    _ = x


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU serial workload")
    parser.add_argument("--seconds", type=float, default=5.0, help="Duration in seconds")
    args = parser.parse_args()

    print(f"[cpu_serial] burning CPU for {args.seconds:.1f}s (single thread)...")
    cpu_burn(args.seconds)
    print("[cpu_serial] done")


if __name__ == "__main__":
    main()
