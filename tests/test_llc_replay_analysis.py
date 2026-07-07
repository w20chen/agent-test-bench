from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from scripts.experiments.analyze_llc_replay_results import (
    discover_placements,
    load_rows,
    summarize,
    write_notes,
)


@contextmanager
def _temp_dir() -> Iterator[Path]:
    base = Path(".tmp-tests")
    base.mkdir(exist_ok=True)
    path = base / f"agent-test-bench-llc-analysis-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_trace(path: Path, *, agent_id: str, tool_ms: float, llm_ms: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "simulate_mode": "cloud_model",
                        "instance_id": agent_id,
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "agent_id": agent_id,
                        "ts_start": 100.0,
                        "ts_end": 100.0 + llm_ms / 1000.0,
                        "data": {"llm_latency_ms": llm_ms},
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "agent_id": agent_id,
                        "ts_start": 101.0,
                        "ts_end": 101.0 + tool_ms / 1000.0,
                        "data": {
                            "tool_name": "exec-pytest",
                            "duration_ms": tool_ms,
                            "success": True,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_analysis_discovers_manifest_placements_without_legacy_names() -> None:
    with _temp_dir() as tmp_path:
        manifest = {
            "source_trace": "trace.jsonl",
            "num_agents": 2,
            "replay_speed": 1.0,
            "cluster_size": 4,
            "runs": [
                {"placement": "os_default"},
                {
                    "placement": "compact_llc",
                    "agent_assignments": [{"agent_index": 0, "cpuset_cpus": "0"}],
                },
                {
                    "placement": "spread_clusters_same_llc",
                    "agent_assignments": [{"agent_index": 0, "cpuset_cpus": "0"}],
                },
            ],
        }
        (tmp_path / "experiment_manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        _write_trace(
            tmp_path / "compact_llc" / "agent-a" / "attempt_1" / "trace.jsonl",
            agent_id="agent-a",
            tool_ms=100.0,
            llm_ms=20.0,
        )
        _write_trace(
            tmp_path
            / "spread_clusters_same_llc"
            / "agent-a"
            / "attempt_1"
            / "trace.jsonl",
            agent_id="agent-a",
            tool_ms=80.0,
            llm_ms=20.0,
        )

        placements = discover_placements(tmp_path)
        rows = load_rows(tmp_path, placements)
        summary = summarize(rows, placements)
        write_notes(tmp_path, summary)

        assert placements == ["os_default", "compact_llc", "spread_clusters_same_llc"]
        assert [row["placement"] for row in summary] == [
            "compact_llc",
            "spread_clusters_same_llc",
        ]
        notes = (tmp_path / "analysis_notes.md").read_text(encoding="utf-8")
        assert "This analysis is descriptive" in notes
        assert "Recommended wording" not in notes
        assert "Per-agent CPU assignments are recorded" in notes
