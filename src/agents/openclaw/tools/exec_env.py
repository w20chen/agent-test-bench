from __future__ import annotations

import os
from pathlib import Path


TASK_TOOL_USERBASE_DEFAULT = "/tmp/openclaw-task-userbase"


def drop_path_entry(path_value: str, entry_to_drop: str) -> str:
    if not path_value:
        return path_value
    sep = ":" if os.pathsep not in path_value and ":" in path_value else os.pathsep
    drop = os.path.normcase(os.path.normpath(entry_to_drop))
    kept = [
        entry
        for entry in path_value.split(sep)
        if os.path.normcase(os.path.normpath(entry or ".")) != drop
    ]
    return sep.join(kept)


def prepare_exec_env(
    path_append: str = "",
    *,
    isolate_runtime_env: bool = False,
) -> dict[str, str]:
    """Build the environment used by OpenClaw shell tool subprocesses."""
    env = os.environ.copy()

    if isolate_runtime_env:
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONNOUSERSITE", None)
        env["PYTHONUSERBASE"] = env.get(
            "OPENCLAW_TASK_USERBASE",
            TASK_TOOL_USERBASE_DEFAULT,
        )

        bootstrap_bin = (
            Path.home() / ".cache" / "task-container-bootstrap" / ".pyuserbase" / "bin"
        )
        env["PATH"] = drop_path_entry(env.get("PATH", ""), str(bootstrap_bin))
    if path_append:
        env["PATH"] = path_append + os.pathsep + env.get("PATH", "")
    return env
