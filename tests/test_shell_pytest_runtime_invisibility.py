from __future__ import annotations

import asyncio
from pathlib import Path

from agents.openclaw.tools.shell import ExecTool
from trace_collect.pytest_runtime_prediction import HIDDEN_RUNTIME_DIR_ARG


def test_pytest_runtime_mechanism_is_not_in_subprocess_env(tmp_path: Path) -> None:
    invocation_dir = tmp_path / "pytest_runtime"
    tool = ExecTool(timeout=20)

    async def _run() -> str:
        return await tool.execute(
            command=(
                "python -c \"import os; print('visible=' + "
                "str(any(k.startswith('OPENCLAW_PYTEST_RUNTIME') "
                "or k == 'PYTEST_PLUGINS' for k in os.environ)))\""
            ),
            **{HIDDEN_RUNTIME_DIR_ARG: str(invocation_dir)},
        )

    result = asyncio.run(_run())

    assert "visible=False" in result
    assert "OPENCLAW_PYTEST_RUNTIME" not in result
    assert not (invocation_dir / "pytest_runtime.json").exists()


def test_pytest_plugin_disable_args_do_not_disable_outer_command_timing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    tests_dir = project / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n",
        encoding="utf-8",
    )
    invocation_dir = tmp_path / "pytest_runtime"
    tool = ExecTool(working_dir=str(project), timeout=20)

    async def _run() -> str:
        return await tool.execute(
            command=(
                "python -m pytest -p no:openclaw_pytest_runtime_plugin tests -q"
            ),
            **{HIDDEN_RUNTIME_DIR_ARG: str(invocation_dir)},
        )

    result = asyncio.run(_run())

    assert "Exit code: 0" in result
    assert "openclaw_pytest_runtime_plugin" not in result
    assert not (invocation_dir / "pytest_runtime.json").exists()
