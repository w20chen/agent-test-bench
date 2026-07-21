"""Parallel CPU workload: multiprocessing with configurable workers.

Usage:
    python cpu_parallel.py --workers 4 --seconds 8

Expected: effective_cores > 1, predicted_cores ≈ worker count
"""

import argparse
import multiprocessing
import time


def worker(seconds: float) -> None:
    """CPU-bound worker function."""
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        result = 0.0
        for i in range(200_000):
            result += (i ** 0.5) * (i % 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel CPU workload")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()

    procs = []
    for _ in range(args.workers):
        p = multiprocessing.Process(target=worker, args=(args.seconds,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
