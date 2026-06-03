from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks._research import HostResearchOpenClawRunner
from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.deep_research_bench import DeepResearchBenchBenchmark
from trace_collect.attempt_pipeline import AttemptContext

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_config(**extras: object) -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="deep-research-bench",
        display_name="DeepResearchBench",
        harness_dataset="example/deepresearchbench",
        harness_split="test",
        trace_root=Path("traces/deep-research-bench"),
        default_max_iterations=100,
        selection_n=32,
        selection_seed=42,
        default_prompt_template="default",
        extras={
            "id_field": "id",
            "question_field": "prompt",
            "answer_field": "article",
            "topic_field": "topic",
            "difficulty_field": "difficulty",
            "domain_field": "domain",
            "reference_kind": "generated_report",
            **extras,
        },
    )


def test_deep_research_bench_registered() -> None:
    assert REGISTRY["deep-research-bench"] is DeepResearchBenchBenchmark
    assert get_benchmark_class("deep-research-bench") is DeepResearchBenchBenchmark


def test_deep_research_bench_normalize_task_preserves_research_fields() -> None:
    plugin = DeepResearchBenchBenchmark(_make_config())

    normalized = plugin.normalize_task(
        {
            "id": "drb-1",
            "prompt": "What happened?",
            "article": "A documented event.",
            "topic": "history",
            "difficulty": "hard",
            "domain": "humanities",
        }
    )

    assert normalized == {
        "instance_id": "drb-1",
        "problem_statement": "What happened?",
        "reference_answer": "A documented event.",
        "topic": "history",
        "difficulty": "hard",
        "domain": "humanities",
        "reference_kind": "generated_report",
        "repo": None,
        "image_name": None,
        "docker_image": None,
    }


def test_deep_research_bench_load_tasks_uses_configured_hf_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    datasets_mod = types.ModuleType("datasets")

    def load_dataset(dataset: str, *, split: str):
        seen["dataset"] = dataset
        seen["split"] = split
        return [
            {
                "id": "drb-1",
                "prompt": "Question",
                "article": "Answer",
            }
        ]

    datasets_mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    plugin = DeepResearchBenchBenchmark(_make_config())
    tasks = plugin.load_tasks()

    assert seen == {"dataset": "example/deepresearchbench", "split": "test"}
    assert tasks[0]["instance_id"] == "drb-1"


def test_deep_research_bench_load_tasks_maps_single_data_file_to_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    datasets_mod = types.ModuleType("datasets")

    def load_dataset(dataset: str, *, split: str, data_files):
        seen["dataset"] = dataset
        seen["split"] = split
        seen["data_files"] = data_files
        return [
            {
                "id": "drb-1",
                "prompt": "Question",
                "article": "Answer",
            }
        ]

    datasets_mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    plugin = DeepResearchBenchBenchmark(
        _make_config(data_files="generated_reports/openai-deepresearch.jsonl")
    )
    tasks = plugin.load_tasks()

    assert seen == {
        "dataset": "example/deepresearchbench",
        "split": "test",
        "data_files": {"test": "generated_reports/openai-deepresearch.jsonl"},
    }
    assert tasks[0]["instance_id"] == "drb-1"


def test_deep_research_bench_runtime_and_runner_gating() -> None:
    plugin = DeepResearchBenchBenchmark(_make_config())

    assert plugin.execution_environment == "host"
    assert plugin.runtime_mode_for("openclaw") == "host_controller"
    assert plugin.runtime_mode_for("tongyi-deepresearch") == "host_controller"


def test_deep_research_bench_builds_tongyi_deepresearch_runner() -> None:
    plugin = DeepResearchBenchBenchmark(_make_config())

    runner = plugin.build_runner(
        scaffold="tongyi-deepresearch",
        model="qwen-plus-latest",
        api_base="https://example.com/v1",
        api_key="test-key",
        max_iterations=100,
        client=SimpleNamespace(),
    )

    assert runner.benchmark_slug == "deep-research-bench"
    # TongyiDeepResearchRunner is the concrete class
    from agents.tongyi_deepresearch import TongyiDeepResearchRunner
    assert isinstance(runner, TongyiDeepResearchRunner)


def test_deep_research_bench_committed_config_matches_generated_report_schema() -> None:
    config = BenchmarkConfig.from_yaml(
        REPO_ROOT / "configs/benchmarks/deep-research-bench.yaml"
    )
    assert config.harness_split == "test"
    plugin = DeepResearchBenchBenchmark(config)

    normalized = plugin.normalize_task(
        {
            "id": 51,
            "prompt": "Research question",
            "article": "Reference report",
        }
    )

    assert normalized["instance_id"] == "51"
    assert normalized["problem_statement"] == "Research question"
    assert normalized["reference_answer"] == "Reference report"
    assert normalized["reference_kind"] == "generated_report"
    assert normalized["topic"] is None


def test_host_research_runner_stamps_host_trace_metadata(tmp_path: Path) -> None:
    runner = object.__new__(HostResearchOpenClawRunner)
    runner.benchmark_slug = "deep-research-bench"
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "type": "trace_metadata",
                "scaffold": "openclaw",
                "trace_format_version": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runner._stamp_trace_metadata(
        trace_path,
        instance_id="drb-1",
        prompt_template="default",
    )

    metadata = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["benchmark"] == "deep-research-bench"
    assert metadata["execution_environment"] == "host"
    assert metadata["instance_id"] == "drb-1"
    assert metadata["prompt_template"] == "default"


def test_host_research_runner_stamps_mcp_config_in_metadata(tmp_path: Path) -> None:
    runner = object.__new__(HostResearchOpenClawRunner)
    runner.benchmark_slug = "deep-research-bench"
    runner.mcp_config = "configs/mcp/context7.yaml"
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "type": "trace_metadata",
                "scaffold": "openclaw",
                "trace_format_version": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runner._stamp_trace_metadata(
        trace_path,
        instance_id="drb-1",
        prompt_template="default",
    )

    metadata = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["run_config"]["mcp_config"] == "configs/mcp/context7.yaml"


def test_host_research_runner_uses_attempt_scoped_workspace_and_session(
    tmp_path: Path,
) -> None:
    runner = object.__new__(HostResearchOpenClawRunner)
    runner.workspace_base = tmp_path / "workspace"
    ctx = AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="drb-1",
        attempt=2,
        task={"instance_id": "drb-1", "problem_statement": "Question"},
        model="qwen-plus-latest",
        scaffold="openclaw",
        source_image=None,
        execution_environment="host",
    )

    assert runner._workspace_for(ctx) == tmp_path / "workspace" / "drb-1" / "attempt_2"
    assert runner._session_key_for(ctx) == "research:drb-1:attempt_2"
