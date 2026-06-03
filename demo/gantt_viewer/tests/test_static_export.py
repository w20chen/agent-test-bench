"""Tests for standalone Gantt HTML export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.gantt_viewer.backend import static_export
from demo.gantt_viewer.tests.helpers import write_trace


def _llm_action(agent_id: str = "agent-1") -> dict:
    return {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": agent_id,
        "iteration": 0,
        "ts_start": 1000.0,
        "ts_end": 1001.0,
        "data": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _write_resources(attempt_dir: Path) -> None:
    (attempt_dir / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "epoch": 1000.2,
                        "cpu_percent": "10%",
                        "mem_usage": "64MiB",
                    },
                    {
                        "epoch": 1000.8,
                        "cpu_percent": "20%",
                        "mem_usage": "80MiB",
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_frontend_dist(dist: Path) -> None:
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (assets / "index-test.js").write_text("window.__STATIC_TEST__ = true;", encoding="utf-8")
    (assets / "index-test.css").write_text("body{background:#000}", encoding="utf-8")
    (dist / "index.html").write_text(
        """
<!doctype html>
<html>
  <head>
    <script type="module" crossorigin src="/assets/index-test.js"></script>
    <link rel="stylesheet" crossorigin href="/assets/index-test.css">
  </head>
  <body><div id="root"></div></body>
</html>
""".strip(),
        encoding="utf-8",
    )


def _write_cohort(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    raw_root = tmp_path / "traces" / "swe-rebench" / "z-ai-glm-5.1" / "20-tasks-combined"
    sim_root = tmp_path / "simulate"
    task_source = tmp_path / "data" / "swe-rebench" / "tasks.json"
    task_source.parent.mkdir(parents=True)
    task_source.write_text("[]\n", encoding="utf-8")
    manifest = []
    task_ids = ("task-a", "task-b")
    source_refs = [
        f"/root/agent-sched-bench/traces/swe-rebench/z-ai-glm-5.1/run/{task_id}/attempt_1/trace.jsonl"
        for task_id in task_ids
    ]
    for task_id, source_ref in zip(task_ids, source_refs, strict=True):
        raw_trace = write_trace(
            raw_root / task_id / "attempt_1" / "trace.jsonl",
            [_llm_action(task_id)],
            scaffold="openclaw",
            model="z-ai/glm-5.1",
            max_iterations=100,
            metadata_overrides={
                "benchmark": "swe-rebench",
                "benchmark_split": "filtered",
                "instance_id": task_id,
                "mode": "collect",
                "runtime_proof": {"sys_path": [source_ref]},
            },
        )
        manifest.append(
            {
                "source_trace": str(raw_trace.relative_to(manifest_path.parent)),
                "task_source": str(task_source.relative_to(manifest_path.parent)),
            }
        )
        sim_trace = write_trace(
            sim_root / "closed_loop" / task_id / "attempt_1" / "trace.jsonl",
            [_llm_action(task_id)],
            scaffold="openclaw",
            model="z-ai/glm-5.1",
            max_iterations=100,
            metadata_overrides={
                "instance_id": task_id,
                "mode": "simulate",
                "replay_target": "cloud_replay",
                "simulate_mode": "cloud_model",
                "source_trace": source_ref,
                "source_trace_count": len(task_ids),
                "source_traces": source_refs,
                "trace_manifest": "configs/simulate/openclaw-glm-19-manifest.json",
            },
        )
        _write_resources(sim_trace.parent)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, sim_root, raw_root, task_source


def _patch_expected_cohort(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_root: Path,
    task_source: Path,
    trace_count: int = 2,
) -> None:
    monkeypatch.setattr(static_export, "EXPECTED_TRACE_COUNT", trace_count)
    monkeypatch.setattr(static_export, "EXPECTED_SOURCE_TRACE_ROOT", raw_root.resolve())
    monkeypatch.setattr(static_export, "EXPECTED_TASK_SOURCE", task_source.resolve())


def test_export_closed_loop_static_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, sim_root, raw_root, task_source = _write_cohort(tmp_path)
    dist = tmp_path / "dist"
    _write_frontend_dist(dist)
    monkeypatch.setattr(static_export, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(static_export, "SIM_SWEEP_ROOT", sim_root)
    monkeypatch.setattr(static_export, "FRONTEND_DIST_PATH", dist)
    _patch_expected_cohort(monkeypatch, raw_root=raw_root, task_source=task_source)

    result = static_export.export_swe_rebench_glm_openclaw_100(
        output_dir=tmp_path / "out",
        group="closed_loop",
    )

    [exported] = result["exports"]
    assert exported["group"] == "closed_loop"
    assert exported["n_traces"] == 2
    assert exported["resource_samples"] == 4
    assert exported["empty_resource_timelines"] == 0

    html = Path(exported["path"]).read_text(encoding="utf-8")
    assert 'id="gantt-viewer-snapshot-bootstrap"' in html
    assert "/assets/index-test.js" not in html
    assert "/assets/index-test.css" not in html
    assert "window.__STATIC_TEST__ = true" in html
    assert "body{background:#000}" in html
    assert '"clockMode": "real"' in html
    assert '"themeMode": "light"' in html
    assert '"timeMode": "sync"' in html
    assert '"viewMode": "layered"' in html

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["preset"] == static_export.PRESET_SWE_REBENCH_GLM_OPENCLAW_100


def test_export_rejects_wrong_source_metadata(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "task-a" / "attempt_1" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
        model="other-model",
        max_iterations=100,
        metadata_overrides={
            "benchmark": "swe-rebench",
            "benchmark_split": "filtered",
            "instance_id": "task-a",
            "mode": "collect",
        },
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([{"source_trace": str(trace.relative_to(tmp_path))}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model='other-model' expected 'z-ai/glm-5.1'"):
        static_export._load_and_validate_manifest_traces(manifest_path)


def test_export_rejects_manifest_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _sim_root, raw_root, task_source = _write_cohort(tmp_path)
    _patch_expected_cohort(
        monkeypatch,
        raw_root=raw_root,
        task_source=task_source,
        trace_count=3,
    )

    with pytest.raises(ValueError, match="expected exactly 3"):
        static_export._load_and_validate_manifest_traces(manifest_path)


def test_export_rejects_mismatched_sim_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, sim_root, raw_root, task_source = _write_cohort(tmp_path)
    _patch_expected_cohort(monkeypatch, raw_root=raw_root, task_source=task_source)
    monkeypatch.setattr(static_export, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(static_export, "SIM_SWEEP_ROOT", sim_root)

    write_trace(
        sim_root / "closed_loop" / "task-a" / "attempt_1" / "trace.jsonl",
        [_llm_action("task-a")],
        scaffold="openclaw",
        model="z-ai/glm-5.1",
        max_iterations=100,
        metadata_overrides={
            "instance_id": "task-a",
            "mode": "simulate",
            "replay_target": "cloud_replay",
            "simulate_mode": "local_model",
            "source_trace": (
                "/root/agent-sched-bench/traces/swe-rebench/z-ai-glm-5.1/"
                "run/task-a/attempt_1/trace.jsonl"
            ),
            "source_trace_count": 2,
            "source_traces": [
                "/root/agent-sched-bench/traces/swe-rebench/z-ai-glm-5.1/"
                "run/task-a/attempt_1/trace.jsonl",
                "/root/agent-sched-bench/traces/swe-rebench/z-ai-glm-5.1/"
                "run/task-b/attempt_1/trace.jsonl",
            ],
            "trace_manifest": "configs/simulate/openclaw-glm-19-manifest.json",
        },
    )

    with pytest.raises(ValueError, match="simulate_mode='local_model' expected 'cloud_model'"):
        static_export.export_swe_rebench_glm_openclaw_100(
            output_dir=tmp_path / "out",
            group="closed_loop",
        )


def test_export_rejects_sim_source_trace_set_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, sim_root, raw_root, task_source = _write_cohort(tmp_path)
    _patch_expected_cohort(monkeypatch, raw_root=raw_root, task_source=task_source)
    monkeypatch.setattr(static_export, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(static_export, "SIM_SWEEP_ROOT", sim_root)

    write_trace(
        sim_root / "closed_loop" / "task-a" / "attempt_1" / "trace.jsonl",
        [_llm_action("task-a")],
        scaffold="openclaw",
        model="z-ai/glm-5.1",
        max_iterations=100,
        metadata_overrides={
            "instance_id": "task-a",
            "mode": "simulate",
            "replay_target": "cloud_replay",
            "simulate_mode": "cloud_model",
            "source_trace": (
                "/root/agent-sched-bench/traces/swe-rebench/z-ai-glm-5.1/"
                "run/task-a/attempt_1/trace.jsonl"
            ),
            "source_trace_count": 2,
            "source_traces": [
                "/root/agent-sched-bench/traces/swe-rebench/z-ai-glm-5.1/"
                "run/task-a/attempt_1/trace.jsonl",
                "/root/agent-sched-bench/traces/swe-rebench/z-ai-glm-5.1/"
                "other/task-b/attempt_1/trace.jsonl",
            ],
            "trace_manifest": "configs/simulate/openclaw-glm-19-manifest.json",
        },
    )

    with pytest.raises(ValueError, match="source_traces do not match curated manifest"):
        static_export.export_swe_rebench_glm_openclaw_100(
            output_dir=tmp_path / "out",
            group="closed_loop",
        )


def test_downsample_resource_timelines_preserves_endpoints() -> None:
    payload = {
        "traces": [
            {
                "resource_timeline": [
                    {"t": idx, "t_abs": idx, "cpu_percent": idx, "memory_mb": idx}
                    for idx in range(10)
                ]
            }
        ]
    }

    static_export._downsample_resource_timelines(payload, max_samples=4)

    assert [sample["t"] for sample in payload["traces"][0]["resource_timeline"]] == [
        0,
        3,
        6,
        9,
    ]


def test_render_html_escapes_inlined_raw_text_breakout() -> None:
    html = static_export._render_html(
        title="test",
        snapshot={
            "mode": "snapshot",
            "payload": {"registries": {"markers": {}, "spans": {}}, "traces": []},
            "trace_ids": [],
            "visible_trace_ids": [],
        },
        frontend={
            "script": 'console.log("</script><script>bad()</script>")',
            "style": "body::before{content:'</style><script>bad()</script>'}",
        },
    )

    assert "</script><script>bad" not in html
    assert "</style><script>bad" not in html
    assert "<\\/script><script>bad()<\\/script>" in html
    assert "<\\/style><script>bad()</script>" in html
