from __future__ import annotations

import pytest
from pathlib import Path
from types import SimpleNamespace

from agents.openclaw.runtime_deps import (
    OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS,
    OPENCLAW_MCP_RUNTIME_REQUIREMENTS,
)
from agents.terminal_bench.openclaw_agent import TerminalBenchOpenClawAgent
from terminal_bench.agents.failure_mode import FailureMode


class StubAgent(TerminalBenchOpenClawAgent):
    @classmethod
    def _build_wheel(cls) -> Path:
        return Path("/tmp/agent_sched_bench-0.1.0-py3-none-any.whl")


def test_install_script_uses_virtualenv() -> None:
    agent = StubAgent(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    script_path = agent._install_agent_script_path
    content = script_path.read_text(encoding="utf-8")
    assert "python3 -m venv /installed-agent/venv" in content
    for requirement in OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS:
        assert requirement in content
    assert (
        "/installed-agent/venv/bin/python -m pip install --no-deps /installed-agent/agent_sched_bench-0.1.0-py3-none-any.whl"
        in content
    )
    for heavy_dep in (
        "datasets",
        "terminal-bench",
        "transformers",
        "accelerate",
        "trafilatura",
    ):
        assert heavy_dep not in content


def test_install_script_adds_mcp_only_when_configured() -> None:
    plain_agent = StubAgent(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    mcp_agent = StubAgent(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        mcp_config_path="/tmp/context7.yaml",
    )

    plain_content = plain_agent._install_agent_script_path.read_text(encoding="utf-8")
    mcp_content = mcp_agent._install_agent_script_path.read_text(encoding="utf-8")

    for requirement in OPENCLAW_MCP_RUNTIME_REQUIREMENTS:
        assert requirement not in plain_content
        assert requirement in mcp_content


def test_run_command_uses_venv_openclaw_and_iteration_limit() -> None:
    agent = StubAgent(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    commands = agent._run_agent_commands("hello task")
    assert len(commands) == 1
    command = commands[0].command
    assert command.startswith("/installed-agent/venv/bin/openclaw ")
    assert "--max-iterations 25" in command
    assert commands[0].max_timeout_sec == float("inf")


def test_run_command_uses_configured_agent_timeout() -> None:
    agent = StubAgent(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        agent_timeout_sec=120,
    )
    commands = agent._run_agent_commands("hello task")
    assert commands[0].max_timeout_sec == 120.0


def test_run_command_resolves_host_local_api_base_inside_container() -> None:
    agent = StubAgent(
        model_name="local-model",
        provider_name="openai",
        api_base="http://172.17.0.1:33895/v1",
        api_key="test-key",
        env_key="OPENAI_API_KEY",
        max_iterations=25,
    )

    command = agent._run_agent_commands("hello task")[0].command

    assert command.startswith("/installed-agent/venv/bin/openclaw ")
    assert '--api-base "${OPENCLAW_API_BASE}"' in command
    assert "/installed-agent/venv/bin/openclaw " in command

    env_setup = agent._create_env_setup_file()
    assert "OPENCLAW_API_BASE='http://172.17.0.1:33895/v1'" in env_setup
    assert "/proc/net/route" in env_setup


def test_run_command_forwards_mcp_config_to_container() -> None:
    agent = StubAgent(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
        mcp_config_path="/tmp/context7.yaml",
    )
    command = agent._run_agent_commands("hello task")[0].command
    assert "--mcp-config /installed-agent/context7.yaml --workspace ." in command


def test_bootstrap_checks_real_venv_creation() -> None:
    command = StubAgent._bootstrap_dependencies_command()
    assert "python3 -m venv --help" not in command
    assert 'python3 -m venv "$probe_root/venv"' in command
    assert '"$probe_root/venv/bin/python" -m pip --version' in command
    assert "python3 python3-pip python3-venv" in command


def test_agent_reads_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in TerminalBenchOpenClawAgent._ENV_PASSTHROUGH:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    agent = StubAgent(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        provider_name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key=None,
        env_key="OPENROUTER_API_KEY",
        max_iterations=25,
    )
    assert agent._env == {
        "OPENROUTER_API_KEY": "env-key",
        "OPENCLAW_API_BASE": "https://openrouter.ai/api/v1",
    }


def test_perform_task_cleans_tmux_session_on_agent_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = StubAgent(
        model_name="local-model",
        provider_name="openai",
        api_base="http://127.0.0.1:8000/v1",
        api_key="dummy",
        env_key="OPENAI_API_KEY",
        max_iterations=25,
        agent_timeout_sec=120,
    )
    calls: list[object] = []

    class FakeContainer:
        id = "container-id"

        def exec_run(self, cmd, user=None):
            calls.append(("exec_run", cmd, user))
            return SimpleNamespace(exit_code=0, output=b"")

    class FakeSession:
        def __init__(self) -> None:
            self.container = FakeContainer()
            self._session_name = "agent"

        def copy_to_container(self, paths, container_dir=None):
            calls.append(("copy_to_container", paths, container_dir))

        def send_keys(self, keys, **kwargs):
            calls.append(("send_keys", keys, kwargs))

        def send_command(self, command):
            calls.append(("send_command", command))
            raise TimeoutError("agent command timed out")

        def capture_pane(self, capture_entire=False):
            calls.append(("capture_pane", capture_entire))
            return "pane output"

    def fake_run(cmd, **kwargs):
        calls.append(("subprocess.run", cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "agents.terminal_bench.openclaw_agent.subprocess.run",
        fake_run,
    )

    result = agent.perform_task(
        "solve it",
        FakeSession(),
        logging_dir=tmp_path,
    )

    assert result.failure_mode == FailureMode.AGENT_TIMEOUT
    assert (tmp_path / "openclaw-timeout.marker").read_text(encoding="utf-8") == (
        "timeout\n"
    )
    assert (tmp_path / "openclaw-timeout-pane.txt").read_text(
        encoding="utf-8"
    ) == "pane output"
    assert any(call[0] == "send_keys" and call[1] == ["C-c"] for call in calls)
    cleanup_commands = [
        call[1]
        for call in calls
        if call[0] == "exec_run" and call[1][:2] == ["sh", "-lc"]
    ]
    assert cleanup_commands
    assert "tmux kill-session -t agent" in cleanup_commands[-1][2]
    assert (
        "pkill -TERM -f '/installed-agent/venv/bin/openclaw'"
        in (cleanup_commands[-1][2])
    )


def test_perform_task_cleans_tmux_session_on_bootstrap_timeout(
    tmp_path: Path,
) -> None:
    agent = StubAgent(
        model_name="local-model",
        provider_name="openai",
        api_base="http://127.0.0.1:8000/v1",
        api_key="dummy",
        env_key="OPENAI_API_KEY",
        max_iterations=25,
        agent_timeout_sec=120,
    )
    calls: list[object] = []

    class FakeContainer:
        def exec_run(self, cmd, user=None):
            calls.append(("exec_run", cmd, user))
            if cmd[:1] == ["timeout"] and cmd[2:4] == ["bash", "-lc"]:
                return SimpleNamespace(exit_code=124, output=b"")
            return SimpleNamespace(exit_code=0, output=b"")

    class FakeSession:
        def __init__(self) -> None:
            self.container = FakeContainer()
            self._session_name = "agent"

        def copy_to_container(self, paths, container_dir=None):
            calls.append(("copy_to_container", paths, container_dir))

        def send_keys(self, keys, **kwargs):
            calls.append(("send_keys", keys, kwargs))

        def send_command(self, command):
            calls.append(("send_command", command))

        def capture_pane(self, capture_entire=False):
            calls.append(("capture_pane", capture_entire))
            return "bootstrap pane"

    result = agent.perform_task(
        "solve it",
        FakeSession(),
        logging_dir=tmp_path,
    )

    assert result.failure_mode == FailureMode.AGENT_TIMEOUT
    assert (tmp_path / "openclaw-timeout.marker").read_text(encoding="utf-8") == (
        "timeout\n"
    )
    assert (tmp_path / "openclaw-timeout-pane.txt").read_text(
        encoding="utf-8"
    ) == "bootstrap pane"
    assert not any(call[0] == "copy_to_container" for call in calls)
    assert not any(
        call[0] == "send_keys"
        and call[1] == ["source /installed-agent/setup-env.sh", "Enter"]
        for call in calls
    )
    cleanup_commands = [
        call[1]
        for call in calls
        if call[0] == "exec_run" and call[1][:2] == ["sh", "-lc"]
    ]
    assert cleanup_commands
    assert "tmux kill-session -t agent" in cleanup_commands[-1][2]
