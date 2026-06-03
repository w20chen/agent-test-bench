from __future__ import annotations

import base64
import hashlib
import sys
import types
from pathlib import Path

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks._research import HostResearchOpenClawRunner
from agents.benchmarks.browsecomp import BrowseCompBenchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_config(**extras: object) -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="browsecomp",
        display_name="BrowseComp",
        harness_dataset="example/browsecomp",
        harness_split="test",
        trace_root=Path("traces/browsecomp"),
        default_max_iterations=100,
        selection_n=32,
        selection_seed=42,
        default_prompt_template="default",
        extras={
            "id_field": "id",
            "question_field": "question",
            "answer_field": "answer",
            "source_urls_field": "source_urls",
            **extras,
        },
    )


def _encrypt_xor_sha256(plaintext: str, password: str) -> str:
    raw = plaintext.encode()
    digest = hashlib.sha256(password.encode()).digest()
    key = digest * (len(raw) // len(digest)) + digest[: len(raw) % len(digest)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode()


def test_browsecomp_registered() -> None:
    assert REGISTRY["browsecomp"] is BrowseCompBenchmark
    assert get_benchmark_class("browsecomp") is BrowseCompBenchmark


def test_browsecomp_normalize_task_preserves_source_urls() -> None:
    plugin = BrowseCompBenchmark(_make_config())

    normalized = plugin.normalize_task(
        {
            "id": "bc-1",
            "question": "Which source states the fact?",
            "answer": "The cited source.",
            "source_urls": ["https://example.com/a", "https://example.com/b"],
        }
    )

    assert normalized == {
        "instance_id": "bc-1",
        "problem_statement": "Which source states the fact?",
        "reference_answer": "The cited source.",
        "source_urls": ["https://example.com/a", "https://example.com/b"],
        "repo": None,
        "image_name": None,
        "docker_image": None,
    }


def test_browsecomp_normalize_task_decodes_json_url_list() -> None:
    plugin = BrowseCompBenchmark(_make_config())

    normalized = plugin.normalize_task(
        {
            "id": "bc-2",
            "question": "Question",
            "answer": "Answer",
            "source_urls": '["https://example.com/a"]',
        }
    )

    assert normalized["source_urls"] == ["https://example.com/a"]


def test_browsecomp_load_tasks_uses_configured_hf_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    datasets_mod = types.ModuleType("datasets")

    def load_dataset(dataset: str, *, split: str):
        seen["dataset"] = dataset
        seen["split"] = split
        return [
            {
                "id": "bc-1",
                "question": "Question",
                "answer": "Answer",
                "source_urls": [],
            }
        ]

    datasets_mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    plugin = BrowseCompBenchmark(_make_config())
    tasks = plugin.load_tasks()

    assert seen == {"dataset": "example/browsecomp", "split": "test"}
    assert tasks[0]["instance_id"] == "bc-1"


def test_browsecomp_runtime_and_runner_gating() -> None:
    plugin = BrowseCompBenchmark(_make_config())

    assert plugin.execution_environment == "host"
    assert plugin.runtime_mode_for("openclaw") == "host_controller"
    assert plugin.runtime_mode_for("tongyi-deepresearch") == "host_controller"


def test_browsecomp_committed_config_matches_long_context_schema() -> None:
    config = BenchmarkConfig.from_yaml(REPO_ROOT / "configs/benchmarks/browsecomp.yaml")
    assert config.harness_split == "test"
    plugin = BrowseCompBenchmark(config)
    canary = "canary"

    normalized = plugin.normalize_task(
        {
            "_row_index": "0",
            "problem": _encrypt_xor_sha256("Question", canary),
            "answer": _encrypt_xor_sha256("Answer", canary),
            "urls": _encrypt_xor_sha256(
                '[["https://example.com/a", "required"]]',
                canary,
            ),
            "canary": canary,
        }
    )

    assert normalized["instance_id"] == "0"
    assert normalized["problem_statement"] == "Question"
    assert normalized["reference_answer"] == "Answer"
    assert normalized["source_urls"] == [
        {"url": "https://example.com/a", "role": "required"}
    ]


def test_browsecomp_decrypts_configured_field_names() -> None:
    plugin = BrowseCompBenchmark(
        _make_config(
            encrypted=True,
            question_field="query",
            answer_field="gold",
            source_urls_field="url_list",
        )
    )
    canary = "canary"

    normalized = plugin.normalize_task(
        {
            "id": "bc-custom",
            "query": _encrypt_xor_sha256("Custom question", canary),
            "gold": _encrypt_xor_sha256("Custom answer", canary),
            "url_list": _encrypt_xor_sha256(
                '["https://example.com/custom"]',
                canary,
            ),
            "canary": canary,
        }
    )

    assert normalized["problem_statement"] == "Custom question"
    assert normalized["reference_answer"] == "Custom answer"
    assert normalized["source_urls"] == ["https://example.com/custom"]


def test_host_research_runner_prompt_includes_source_urls_not_reference() -> None:
    runner = object.__new__(HostResearchOpenClawRunner)
    runner.benchmark_slug = "browsecomp"

    prompt = runner._render_prompt(
        {
            "problem_statement": "Question",
            "reference_answer": "Answer",
            "source_urls": [{"url": "https://example.com/a", "role": "required"}],
        },
        prompt_template="default",
    )

    assert "Question" in prompt
    assert "https://example.com/a" in prompt
    assert "Answer" not in prompt
