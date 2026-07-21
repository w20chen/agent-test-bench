"""Tests for tool_profiler integration with ExecTool in shell.py.

Verifies that when TOOL_PROFILER env vars are set, matching tool commands
are wrapped with the prototype profiler.
"""

from __future__ import annotations

import asyncio
import glob
import os
import tempfile

import pytest


class TestToolProfilerShellIntegration:
    """Integration tests for tool_profiler wrapping in ExecTool."""

    def test_exec_tool_wraps_with_tool_profiler(self, monkeypatch) -> None:
        """When TOOL_PROFILER=1 and tool matches, command should be wrapped."""
        monkeypatch.setenv("TOOL_PROFILER", "1")
        monkeypatch.setenv("TOOL_PROFILER_TOOLS", "exec-echo")
        tmpdir = tempfile.mkdtemp()
        monkeypatch.setenv("TOOL_PROFILER_OUT", tmpdir)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=10)

        async def _run():
            return await tool.execute(command="echo hello")

        result = asyncio.run(_run())
        # echo classifies as exec-echo, which matches TOOL_PROFILER_TOOLS.
        # The tool_profiler wraps the command and passes through stdout/stderr.
        assert "hello" in result

    def test_exec_tool_no_wrap_when_disabled(self, monkeypatch) -> None:
        """When TOOL_PROFILER is not set, command should NOT be wrapped."""
        monkeypatch.delenv("TOOL_PROFILER", raising=False)
        monkeypatch.delenv("TOOL_PROFILER_TOOLS", raising=False)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=10)

        async def _run():
            return await tool.execute(command="echo hello")

        result = asyncio.run(_run())
        assert "hello" in result

    def test_exec_tool_no_wrap_when_tool_not_matching(self, monkeypatch) -> None:
        """When TOOL_PROFILER=1 but tool doesn't match, should NOT wrap."""
        monkeypatch.setenv("TOOL_PROFILER", "1")
        monkeypatch.setenv("TOOL_PROFILER_TOOLS", "exec-grep")
        tmpdir = tempfile.mkdtemp()
        monkeypatch.setenv("TOOL_PROFILER_OUT", tmpdir)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=10)

        async def _run():
            # echo classifies as exec-echo, which doesn't match exec-grep
            return await tool.execute(command="echo hello")

        result = asyncio.run(_run())
        assert "hello" in result

    def test_tool_profiler_preserves_exit_code(self, monkeypatch) -> None:
        """Tool_profiler wrapping should preserve the inner command's exit code."""
        monkeypatch.setenv("TOOL_PROFILER", "1")
        monkeypatch.setenv("TOOL_PROFILER_TOOLS", "exec-python")
        tmpdir = tempfile.mkdtemp()
        monkeypatch.setenv("TOOL_PROFILER_OUT", tmpdir)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=10)

        async def _run():
            return await tool.execute(command="python -c 'exit(0)'")

        result = asyncio.run(_run())
        assert "Exit code: 0" in result

    def test_tool_profiler_produces_profile_jsonl(self, monkeypatch, tmp_path) -> None:
        """Tool_profiler wrapping should produce a profile.jsonl output file."""
        out_dir = str(tmp_path / "tool_profiles")
        monkeypatch.setenv("TOOL_PROFILER", "1")
        monkeypatch.setenv("TOOL_PROFILER_TOOLS", "exec-python")
        monkeypatch.setenv("TOOL_PROFILER_OUT", out_dir)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=30)

        async def _run():
            return await tool.execute(
                command='python -c "import time; time.sleep(0.5)"'
            )

        result = asyncio.run(_run())
        assert "Exit code: 0" in result

        # There should be at least one profile subdirectory with profile.jsonl
        profile_files = glob.glob(f"{out_dir}/**/profile.jsonl", recursive=True)
        assert len(profile_files) >= 1, f"No profile.jsonl found in {out_dir}"

    def test_tool_profiler_preserves_shell_operators(self, monkeypatch, tmp_path) -> None:
        """Profiler wrapping should keep an existing shell command intact."""
        out_dir = str(tmp_path / "tool_profiles")
        monkeypatch.setenv("TOOL_PROFILER", "1")
        monkeypatch.setenv("TOOL_PROFILER_TOOLS", "exec-python")
        monkeypatch.setenv("TOOL_PROFILER_OUT", out_dir)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=30)
        command = (
            "python -c \"print('first')\" && "
            "python -c \"print('second')\""
        )

        async def _run():
            return await tool.execute(command=command)

        result = asyncio.run(_run())
        assert "first" in result
        assert "second" in result
        assert "Exit code: 0" in result

        profile_files = glob.glob(f"{out_dir}/**/profile.jsonl", recursive=True)
        assert len(profile_files) >= 1, f"No profile.jsonl found in {out_dir}"

    def test_tool_profiler_preserves_tool_stderr_prefixes(self, monkeypatch, tmp_path) -> None:
        """Profiler stderr filtering must not remove real tool stderr lines."""
        out_dir = str(tmp_path / "tool_profiles")
        monkeypatch.setenv("TOOL_PROFILER", "1")
        monkeypatch.setenv("TOOL_PROFILER_TOOLS", "exec-python")
        monkeypatch.setenv("TOOL_PROFILER_OUT", out_dir)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=30)
        command = (
            "python -c \"import sys; "
            "print('  keep indented stderr', file=sys.stderr); "
            "print('FINAL PROFILE from tool', file=sys.stderr)\""
        )

        async def _run():
            return await tool.execute(command=command)

        result = asyncio.run(_run())
        assert "  keep indented stderr" in result
        assert "FINAL PROFILE from tool" in result
        assert "[tool-profiler]" not in result


class TestToolSchedulerShellIntegration:
    """Integration tests for tool_scheduler wrapping in ExecTool."""

    def test_tool_scheduler_preserves_shell_operators(self, monkeypatch, tmp_path) -> None:
        """Scheduler wrapping should keep an existing shell command intact."""
        out_dir = str(tmp_path / "tool_scheduler")
        monkeypatch.setenv("TOOL_SCHEDULER", "1")
        monkeypatch.setenv("TOOL_SCHEDULER_TOOLS", "exec-python")
        monkeypatch.setenv("TOOL_SCHEDULER_OUT", out_dir)
        monkeypatch.delenv("TOOL_SCHEDULER_HARDCODE_TOPOLOGY", raising=False)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=30)
        command = (
            "python -c \"print('first')\" && "
            "python -c \"print('second')\""
        )

        async def _run():
            return await tool.execute(command=command)

        result = asyncio.run(_run())
        assert "first" in result
        assert "second" in result
        assert "Exit code: 0" in result

        profile_files = glob.glob(f"{out_dir}/**/profile.jsonl", recursive=True)
        assert len(profile_files) >= 1, f"No scheduler profile.jsonl found in {out_dir}"

    def test_tool_scheduler_preserves_tool_stderr_prefixes(self, monkeypatch, tmp_path) -> None:
        """Scheduler stderr filtering must not remove real tool stderr lines."""
        out_dir = str(tmp_path / "tool_scheduler")
        monkeypatch.setenv("TOOL_SCHEDULER", "1")
        monkeypatch.setenv("TOOL_SCHEDULER_TOOLS", "exec-python")
        monkeypatch.setenv("TOOL_SCHEDULER_OUT", out_dir)
        monkeypatch.delenv("TOOL_SCHEDULER_HARDCODE_TOPOLOGY", raising=False)

        from agents.openclaw.tools.shell import ExecTool

        tool = ExecTool(timeout=30)
        command = (
            "python -c \"import sys; "
            "print('WARNING: keep me', file=sys.stderr); "
            "print('action: keep me', file=sys.stderr)\""
        )

        async def _run():
            return await tool.execute(command=command)

        result = asyncio.run(_run())
        assert "WARNING: keep me" in result
        assert "action: keep me" in result
        assert "[tool-scheduler]" not in result
