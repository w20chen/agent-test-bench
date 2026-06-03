"""Focused tests for OpenClaw prompt workspace rendering."""

from __future__ import annotations

from pathlib import Path

from agents.openclaw._context import ContextBuilder


def test_context_builder_uses_project_workspace_for_prompt_identity(
    tmp_path: Path,
) -> None:
    state_workspace = tmp_path / "state"
    state_workspace.mkdir()
    project_workspace = tmp_path / "project"
    project_workspace.mkdir()
    (project_workspace / "AGENTS.md").write_text(
        "project instructions\n",
        encoding="utf-8",
    )

    prompt = ContextBuilder(
        state_workspace,
        project_workspace=project_workspace,
    ).build_system_prompt()

    assert f"Your workspace is at: {project_workspace.resolve()}" in prompt
    assert f"Agent state directory: {state_workspace.resolve()}" in prompt
    assert (
        f"Long-term memory: {state_workspace.resolve()}/memory/MEMORY.md"
        in prompt
    )
    assert "project instructions" in prompt
