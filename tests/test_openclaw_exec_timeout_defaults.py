from agents.openclaw.config.schema import ExecToolConfig
from agents.openclaw.tools.shell import ExecTool


def test_exec_tool_default_timeout_is_300_seconds() -> None:
    tool = ExecTool()

    assert tool.timeout == 300


def test_exec_tool_config_default_timeout_is_300_seconds() -> None:
    cfg = ExecToolConfig()

    assert cfg.timeout == 300
