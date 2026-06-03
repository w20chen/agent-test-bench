"""Disk space preflight check for the attempt pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path


class DiskSpaceError(RuntimeError):
    """Raised when the preflight finds less free space than the caller requires."""


def preflight_disk(path: Path, min_free_gb: float) -> float:
    """Abort when *path*'s filesystem has < *min_free_gb* gigabytes free.

    Returns the free space in GB on success so callers can log it.

    Walks up to the first existing parent so the check works even when the
    caller passes a run directory that has not been created yet.
    """
    probe = Path(path).resolve()
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise DiskSpaceError(
            f"Insufficient free disk at {probe}: {free_gb:.2f} GB available, "
            f"{min_free_gb:.2f} GB required"
        )
    return free_gb
