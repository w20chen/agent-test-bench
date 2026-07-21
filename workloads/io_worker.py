#!/usr/bin/env python3
"""I/O workload: sustained read/write to a temporary file.

Usage:
    python workloads/io_worker.py --seconds 5 --block-size 1048576
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time


def io_work(duration: float, block_size: int) -> None:
    """Write and read a temporary file for `duration` seconds."""
    end = time.monotonic() + duration
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmpfile.name
    tmpfile.close()

    data = os.urandom(block_size)

    try:
        iteration = 0
        while time.monotonic() < end:
            iteration += 1
            # Write phase
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # Read phase
            with open(tmp_path, "rb") as f:
                _ = f.read()

            if iteration % 100 == 0:
                print(f"[io_worker] iteration {iteration}", flush=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="I/O workload")
    parser.add_argument("--seconds", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument(
        "--block-size",
        type=int,
        default=1_048_576,
        help="Block size in bytes for each write/read cycle",
    )
    args = parser.parse_args()

    print(f"[io_worker] running I/O for {args.seconds:.1f}s (block={args.block_size}B)...")
    io_work(args.seconds, args.block_size)
    print("[io_worker] done")


if __name__ == "__main__":
    main()
