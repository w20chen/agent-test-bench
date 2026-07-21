"""Serial CPU workload: single-threaded computation for ~8 seconds.

Expected: effective_cores ≈ 1, predicted_cores ≈ 1
"""

import time


def compute(n: int) -> float:
    """Do some CPU-bound work."""
    result = 0.0
    for i in range(n):
        result += (i ** 0.5) * (i % 100)
    return result


def main() -> None:
    start = time.monotonic()
    target = 8.0
    while time.monotonic() - start < target:
        compute(200_000)


if __name__ == "__main__":
    main()
