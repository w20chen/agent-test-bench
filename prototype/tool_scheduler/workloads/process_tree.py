"""Process tree workload: parent creates multiple child processes.

Validates:
  - process_count > 1
  - effective_cores aggregates entire process tree
"""

import argparse
import multiprocessing
import os
import sys
import time


def child_worker(worker_id: int, seconds: float) -> None:
    """Child process doing CPU work."""
    print(f"[child {worker_id}] pid={os.getpid()} started", file=sys.stderr)
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        result = 0.0
        for i in range(100_000):
            result += (i ** 0.5) * (i % 50)
    print(f"[child {worker_id}] pid={os.getpid()} done", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process tree workload")
    parser.add_argument("--children", type=int, default=4,
                        help="Number of child processes (default: 4)")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="Duration in seconds (default: 5)")
    parser.add_argument("--spawn-threads", type=int, default=0,
                        help="Threads per child (default: 0)")
    args = parser.parse_args()

    print(f"[parent] pid={os.getpid()} creating {args.children} children",
          file=sys.stderr)

    procs = []
    for i in range(args.children):
        p = multiprocessing.Process(target=child_worker, args=(i, args.seconds))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print(f"[parent] all children done", file=sys.stderr)


if __name__ == "__main__":
    main()
