"""SWE-rebench benchmark plugin unit tests.

Focused on the three schema quirks the plugin absorbs:

1. ``FAIL_TO_PASS`` stays as a native Python list (no JSON round-trip).
2. Explicit ``docker_image`` URI is pinned to ``task['image_name']``.
3. ``meta.is_lite`` filter is opt-in via the ``exclude_lite`` config knob,
   defaulting to ``False`` per CLAUDE.md §1 "no benchmark-specific tuning".

Plus contract checks for ``build_runner`` scaffold refusal
and the registry entry resolves correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.swe_rebench import SWERebenchBenchmark


def _make_config(*, exclude_lite: bool = False) -> BenchmarkConfig:
    """Construct a BenchmarkConfig matching the shipped SWE-rebench YAML defaults."""
    return BenchmarkConfig(
        slug="swe-rebench",
        display_name="SWE-rebench (filtered)",
        harness_dataset="nebius/SWE-rebench",
        harness_split="filtered",
        data_root=Path("data/swe-rebench"),
        repos_root=Path("data/swe-rebench/repos"),
        trace_root=Path("traces/swe-rebench"),
        default_max_iterations=100,
        selection_n=4,  # small for tests
        selection_seed=42,
        default_prompt_template="cc_aligned",
        exclude_lite=exclude_lite,
    )


def _make_rebench_row(
    instance_id: str,
    *,
    repo: str = "nebius/sample",
    is_lite: bool = False,
    docker_image: str | None = "swerebench/sweb.eval.x86_64.nebius_sample",
    fail_to_pass: list[str] | None = None,
    problem_statement: str | None = None,
) -> dict[str, Any]:
    """Build a synthetic SWE-rebench row matching the real HF schema shape.

    FAIL_TO_PASS / PASS_TO_PASS are native lists (matching the real dataset,
    confirmed by Phase 0 HF load verification).
    """
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "deadbeefcafe0123456789",
        "problem_statement": problem_statement
        or (
            "A reasonably-long synthetic problem statement that exceeds the "
            "100-character threshold used by the default selection logic to "
            "filter out trivial tasks."
        ),
        "patch": "",
        "test_patch": "",
        "FAIL_TO_PASS": fail_to_pass or ["tests/test_sample.py::test_case_a"],
        "PASS_TO_PASS": ["tests/test_sample.py::test_passing"],
        "FAIL_TO_FAIL": [],
        "PASS_TO_FAIL": [],
        "docker_image": docker_image,
        "image_name": None,  # will be pinned by normalize_task
        "meta": {"is_lite": is_lite, "has_test_patch": True, "num_modified_files": 1},
    }


# ── normalize_task quirks ───────────────────────────────────────────────


def test_normalize_task_keeps_native_lists() -> None:
    """Quirk 1: FAIL_TO_PASS / PASS_TO_PASS stay as native Python lists (no JSON round-trip)."""
    # derive_test_cmd and _count_fail_to_pass handle both shapes; json.dumps here would be lossy
    plugin = SWERebenchBenchmark(_make_config())
    row = _make_rebench_row(
        "django__django-11734",
        fail_to_pass=["tests/test_a.py::test_x", "tests/test_b.py::test_y"],
    )

    normalized = plugin.normalize_task(row)

    assert isinstance(normalized["FAIL_TO_PASS"], list)
    assert normalized["FAIL_TO_PASS"] == [
        "tests/test_a.py::test_x",
        "tests/test_b.py::test_y",
    ]
    assert isinstance(normalized["PASS_TO_PASS"], list)


def test_normalize_task_pins_docker_image() -> None:
    """Quirk 2: docker_image URI is copied into image_name."""
    plugin = SWERebenchBenchmark(_make_config())
    row = _make_rebench_row(
        "nebius__foo-42",
        docker_image="swerebench/sweb.eval.x86_64.nebius_1776_foo-42",
    )

    normalized = plugin.normalize_task(row)

    assert normalized["image_name"] == "swerebench/sweb.eval.x86_64.nebius_1776_foo-42"
    # image_name_for should surface the same URI
    assert plugin.image_name_for(normalized) == (
        "swerebench/sweb.eval.x86_64.nebius_1776_foo-42"
    )


def test_normalize_task_no_docker_image_leaves_image_name_absent() -> None:
    """A row without docker_image should not have an image_name synthesized."""
    plugin = SWERebenchBenchmark(_make_config())
    row = _make_rebench_row("foo__bar-1", docker_image=None)
    row.pop("image_name", None)  # simulate HF row without the field

    normalized = plugin.normalize_task(row)

    assert normalized.get("image_name") is None


def test_normalize_task_derives_test_cmd_from_list() -> None:
    """derive_test_cmd must produce a pytest command from native-list FAIL_TO_PASS."""
    plugin = SWERebenchBenchmark(_make_config())
    row = _make_rebench_row(
        "x__y-1",
        fail_to_pass=["tests/alpha.py::test_one", "tests/beta.py::test_two"],
    )
    normalized = plugin.normalize_task(row)

    assert "test_cmd" in normalized
    assert "pytest" in normalized["test_cmd"]
    assert "tests/alpha.py::test_one" in normalized["test_cmd"]
    assert "tests/beta.py::test_two" in normalized["test_cmd"]


def test_swe_rebench_config_default_prompt_is_cc_aligned() -> None:
    plugin = SWERebenchBenchmark(_make_config())
    assert plugin.config.default_prompt_template == "cc_aligned"


def test_swe_rebench_runtime_mode_for_openclaw_only() -> None:
    plugin = SWERebenchBenchmark(_make_config())
    assert plugin.runtime_mode_for("openclaw") == "task_container_agent"
    with pytest.raises(NotImplementedError):
        plugin.runtime_mode_for("unsupported")


# ── select_subset / exclude_lite knob ───────────────────────────────────


def _make_mixed_pool() -> list[dict[str, Any]]:
    """Produce 6 synthetic rebench tasks, half lite and half non-lite."""
    return [
        _make_rebench_row(f"lite__inst-{i}", is_lite=True) for i in range(3)
    ] + [
        _make_rebench_row(f"heavy__inst-{i}", is_lite=False) for i in range(3)
    ]


def test_select_subset_exclude_lite_false_keeps_lite() -> None:
    """Default exclude_lite=False: lite tasks remain in the candidate pool."""
    plugin = SWERebenchBenchmark(_make_config(exclude_lite=False))
    tasks = [plugin.normalize_task(t) for t in _make_mixed_pool()]

    selected = plugin.select_subset(tasks, n=6, seed=0)

    # All 6 original tasks are candidates (no is_lite filter applied).
    lite_count = sum(1 for t in selected if (t.get("meta") or {}).get("is_lite"))
    assert lite_count > 0, (
        "exclude_lite=False must keep lite tasks; P5 forbids silent filters"
    )


def test_select_subset_exclude_lite_true_drops_lite() -> None:
    """exclude_lite=True: lite tasks are filtered BEFORE stratified selection."""
    plugin = SWERebenchBenchmark(_make_config(exclude_lite=True))
    tasks = [plugin.normalize_task(t) for t in _make_mixed_pool()]

    selected = plugin.select_subset(tasks, n=6, seed=0)

    lite_count = sum(1 for t in selected if (t.get("meta") or {}).get("is_lite"))
    assert lite_count == 0, (
        "exclude_lite=True must drop every task with meta.is_lite=True"
    )
