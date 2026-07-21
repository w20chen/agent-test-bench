#!/usr/bin/env python3
"""CPU parallel workload: multiprocessing workers for a configurable duration.

Usage:
    python workloads/cpu_parallel.py --workers 4 --seconds 5
"""

from __future__ import annotations

import argparse
import multiprocessing
import time


def cpu_burn(duration: float) -> None:
    """Perform single-threaded arithmetic for `duration` seconds."""
    end = time.monotonic() + duration
    x = 0.0
    while time.monotonic() < end:
        x = (x + 1.0) * 1.00001 % 1e9
    _ = x


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU parallel workload")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--seconds", type=float, default=5.0, help="Duration in seconds")
    args = parser.parse_args()

    print(f"[cpu_parallel] launching {args.workers} workers for {args.seconds:.1f}s...")
    procs = []
    for i in range(args.workers):
        p = multiprocessing.Process(target=cpu_burn, args=(args.seconds,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("[cpu_parallel] done")


if __name__ == "__main__":
    main()
