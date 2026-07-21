"""Memory bandwidth workload: sequential large array scan.

If PMU is available, validates memory bandwidth utilization rises.
If PMU is unavailable, validates graceful degradation.
"""

import argparse
import sys
import time


def memory_scan(size_mb: int, duration_s: float, num_threads: int = 1) -> None:
    """Perform sequential scan over a large array."""
    # Allocate array (~size_mb MB of floats)
    num_elements = (size_mb * 1024 * 1024) // 8
    import array
    try:
        arr = array.array("d", [0.0]) * num_elements
    except MemoryError:
        # Try smaller allocation
        num_elements = num_elements // 2
        arr = array.array("d", [0.0]) * num_elements

    print(
        f"[memscan] allocated {len(arr) * 8 / (1024*1024):.1f} MB, "
        f"scanning for {duration_s}s",
        file=sys.stderr,
    )

    start = time.monotonic()
    iterations = 0
    while time.monotonic() - start < duration_s:
        # Sequential scan with compute
        s = 0.0
        for i in range(len(arr)):
            s += arr[i] * 0.5
        iterations += 1
        # Prevent optimization
        if iterations % 100 == 0:
            arr[0] = s

    print(f"[memscan] completed {iterations} scan iterations", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory bandwidth workload")
    parser.add_argument("--size-mb", type=int, default=512,
                        help="Array size in MB (default: 512)")
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="Duration in seconds (default: 8)")
    args = parser.parse_args()
    memory_scan(args.size_mb, args.seconds)


if __name__ == "__main__":
    main()
