from agents.openclaw.config.schema import ExecToolConfig
from agents.openclaw.tools.shell import ExecTool, _prepare_exec_env


def test_exec_tool_default_timeout_is_300_seconds() -> None:
    tool = ExecTool()

    assert tool.timeout == 300


def test_exec_tool_config_default_timeout_is_300_seconds() -> None:
    cfg = ExecToolConfig()

    assert cfg.timeout == 300


def test_exec_tool_env_isolates_task_commands_from_runtime(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/runtime/pydeps:/repo/src")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv(
        "PATH",
        "/home/user/.cache/task-container-bootstrap/.pyuserbase/bin:/usr/local/bin:/usr/bin",
    )
    monkeypatch.setattr(
        "agents.openclaw.tools.shell.Path.home",
        lambda: __import__("pathlib").Path("/home/user"),
    )

    env = _prepare_exec_env(
        "/tmp/openclaw-task-userbase/bin",
        isolate_runtime_env=True,
    )

    assert "PYTHONPATH" not in env
    assert "PYTHONNOUSERSITE" not in env
    assert env["PYTHONUSERBASE"] == "/tmp/openclaw-task-userbase"
    assert env["PATH"].startswith("/tmp/openclaw-task-userbase/bin")
    assert ".cache/task-container-bootstrap/.pyuserbase/bin" not in env["PATH"]


def test_exec_tool_env_preserves_host_python_env_by_default(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/host/project/src")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("PYTHONUSERBASE", "/host/userbase")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")

    env = _prepare_exec_env("/custom/bin")

    assert env["PYTHONPATH"] == "/host/project/src"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONUSERBASE"] == "/host/userbase"
    assert env["PATH"].startswith("/custom/bin")
