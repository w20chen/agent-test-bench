from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import TraceAction


def build_run_id(
    system: str, workload: str, concurrency: int, started_at: datetime | None = None
) -> str:
    """Construct the canonical run id: {system}_{workload}_{N}_{timestamp}."""
    ts = started_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    utc_ts = ts.astimezone(timezone.utc)
    timestamp = (
        utc_ts.strftime("%Y%m%dT%H%M%S") + f"{int(utc_ts.microsecond / 1000):03d}Z"
    )
    return f"{system}_{workload}_{concurrency}_{timestamp}"


class TraceLogger:
    """Append-only JSONL trace logger for step and summary events."""

    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        self.path = Path(output_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode supports resume: existing records are preserved and
        # collector.py deduplicates via load_completed_ids().
        self._handle = self.path.open("a", encoding="utf-8")

    def log_metadata(
        self,
        scaffold: str,
        execution_environment: str = "container",
        **kwargs: Any,
    ) -> None:
        """Write a trace_metadata header record declaring the scaffold and run context.

        This must be the first record in the trace so downstream tools
        (timeline, analysis) can dispatch to the correct parser.
        """
        entry = {
            "type": "trace_metadata",
            "scaffold": scaffold,
            "trace_format_version": 5,
            "execution_environment": execution_environment,
            **kwargs,
        }
        self._handle.write(json.dumps(entry) + "\n")
        self._handle.flush()

    def log_trace_action(self, agent_id: str, action: TraceAction) -> None:
        """Write a v4 TraceAction record."""
        entry = action.to_dict()
        self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._handle.flush()


    def log_summary(self, agent_id: str, summary: dict[str, Any]) -> None:
        entry = {"type": "summary", "agent_id": agent_id, **summary}
        self._handle.write(json.dumps(entry) + "\n")
        self._handle.flush()

    def log_event(
        self,
        agent_id: str,
        category: str,
        event: str,
        data: dict[str, Any],
        *,
        iteration: int = 0,
        ts: float | None = None,
    ) -> None:
        """Write a v4 envelope event record."""
        import time
        entry = {
            "type": "event",
            "agent_id": agent_id,
            "category": category,
            "event": event,
            "iteration": iteration,
            "ts": ts if ts is not None else time.time(),
            "data": data,
        }
        self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
