#!/usr/bin/env python3
"""Process tree workload: parent spawns multiple child processes.

Usage:
    python workloads/process_tree.py --children 4 --seconds 5
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time


def child_worker(child_id: int, duration: float) -> None:
    """Child process: print its PID and do light work for `duration` seconds."""
    print(f"  [child {child_id}] pid={os.getpid()}", flush=True)
    end = time.monotonic() + duration
    while time.monotonic() < end:
        time.sleep(0.1)
    print(f"  [child {child_id}] exiting", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process tree workload")
    parser.add_argument("--children", type=int, default=4, help="Number of child processes")
    parser.add_argument("--seconds", type=float, default=5.0, help="Duration in seconds")
    args = parser.parse_args()

    print(f"[process_tree] parent pid={os.getpid()}, spawning {args.children} children for {args.seconds:.1f}s...")

    procs = []
    for i in range(args.children):
        p = multiprocessing.Process(target=child_worker, args=(i, args.seconds))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("[process_tree] done")


if __name__ == "__main__":
    main()
