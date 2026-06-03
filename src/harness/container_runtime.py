"""Container-runtime-specific command helpers."""

from __future__ import annotations

import os


def container_run_user_args(container_executable: str) -> list[str]:
    """Return runtime-specific user mapping args for ``run`` commands."""
    if container_executable == "podman":
        return ["--userns=keep-id"]
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def image_exists_command(
    image: str,
    *,
    container_executable: str,
) -> list[str]:
    """Return a CLI-specific image existence probe command."""
    if container_executable == "podman":
        return [container_executable, "image", "exists", image]
    return [container_executable, "image", "inspect", image]
