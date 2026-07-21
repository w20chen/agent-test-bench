"""Phased CPU workload: transitions between serial and parallel phases.

Phases:
  - 0-4s: single-threaded
  - 4-12s: 8 workers (parallel)
  - 12-16s: single-threaded

Validates:
  - Prediction recalibrates across phase changes
  - No unstable recommendations during transitions
"""

import multiprocessing
import sys
import time


def worker(seconds: float) -> None:
    """CPU-bound worker function."""
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        result = 0.0
        for i in range(200_000):
            result += (i ** 0.5) * (i % 100)


def serial_phase(duration: float) -> None:
    """Single-threaded CPU work."""
    worker(duration)


def parallel_phase(workers: int, duration: float) -> None:
    """Multi-process CPU work."""
    procs = []
    for _ in range(workers):
        p = multiprocessing.Process(target=worker, args=(duration,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def main() -> None:
    print("[phased] Phase 1: serial (4s)", file=sys.stderr)
    serial_phase(4.0)

    print("[phased] Phase 2: parallel 8 workers (8s)", file=sys.stderr)
    parallel_phase(8, 8.0)

    print("[phased] Phase 3: serial (4s)", file=sys.stderr)
    serial_phase(4.0)

    print("[phased] Done", file=sys.stderr)


if __name__ == "__main__":
    main()
