"""Per-benchmark prompt template resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_collect.prompt_loader import (
    _PROMPTS_ROOT,
    load_prompt_template,
    render_prompt,
)


def test_load_default_for_terminal_bench_returns_minimal_template() -> None:
    text = load_prompt_template("default", "terminal-bench")
    # The whole point of the per-benchmark refactor: tb must NOT inherit
    # the SWE-style wrapper (which named a non-existent `bash` tool and a
    # patch-submission flow that has no meaning in tb).
    assert "<pr_description>" not in text
    assert "bash tool calls" not in text
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in text
    assert "{{task}}" in text


def test_load_default_for_swe_bench_verified_returns_swe_template() -> None:
    text = load_prompt_template("default", "swe-bench-verified")
    assert "<pr_description>" in text
    assert "{{task}}" in text


def test_hyphen_to_underscore_normalization(tmp_path, monkeypatch) -> None:
    fake_root = tmp_path / "prompts"
    (fake_root / "foo_bar").mkdir(parents=True)
    (fake_root / "foo_bar" / "default.md").write_text("hello {{task}}", encoding="utf-8")
    monkeypatch.setattr("trace_collect.prompt_loader._PROMPTS_ROOT", fake_root)

    assert load_prompt_template("default", "foo-bar") == "hello {{task}}"


def test_missing_benchmark_dir_raises_with_empty_available_list(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("trace_collect.prompt_loader._PROMPTS_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        load_prompt_template("default", "no-such-bench")

    assert "Available templates: []" in str(excinfo.value)


def test_template_without_placeholder_raises_value_error(
    tmp_path, monkeypatch
) -> None:
    fake_root = tmp_path / "prompts"
    (fake_root / "demo").mkdir(parents=True)
    (fake_root / "demo" / "default.md").write_text("no placeholder here", encoding="utf-8")
    monkeypatch.setattr("trace_collect.prompt_loader._PROMPTS_ROOT", fake_root)

    with pytest.raises(ValueError, match="missing the required"):
        load_prompt_template("default", "demo")


def test_render_substitutes_placeholder() -> None:
    rendered = render_prompt("before {{task}} after", "the task")
    assert rendered == "before the task after"


def test_real_prompt_dirs_match_committed_layout() -> None:
    # Regression guard: if any per-benchmark dir disappears or is renamed,
    # the corresponding benchmark plugin's runtime load will break. Pin the
    # required dirs explicitly so the failure surfaces here, not in a run.
    for slug in ("swe_rebench", "swe_bench_verified", "terminal_bench"):
        assert (_PROMPTS_ROOT / slug).is_dir(), f"missing prompt dir: {slug}"
