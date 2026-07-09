"""Small helpers for full-sequence attention plotting scripts."""

from __future__ import annotations

from typing import Any


def _require_samples(payload: dict[str, Any]) -> list[Any]:
    """Return non-empty ``samples`` from a plotting payload.

    Raises:
        ValueError: If the payload does not contain any samples.
    """
    samples = payload.get("samples")
    if not samples:
        raise ValueError("no samples available for attention plot")
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    return samples
