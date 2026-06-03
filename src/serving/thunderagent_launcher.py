from __future__ import annotations

import argparse
from dataclasses import dataclass

from serving._common import print_or_exec_command, resolve_python_sibling_executable


@dataclass(slots=True)
class ThunderAgentConfig:
    """Configuration for launching the ThunderAgent proxy."""

    backends: str
    host: str = "0.0.0.0"
    port: int = 9000
    backend_type: str = "vllm"
    profile: bool = True
    metrics: bool = True
    profile_dir: str = "/tmp/thunderagent_profiles"


def build_thunderagent_command(config: ThunderAgentConfig) -> list[str]:
    """Build the ThunderAgent CLI command from structured config."""
    command = [
        resolve_python_sibling_executable("thunderagent"),
        "--backend-type",
        config.backend_type,
        "--backends",
        config.backends,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--profile-dir",
        config.profile_dir,
    ]
    if config.profile:
        command.append("--profile")
    if config.metrics:
        command.append("--metrics")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally launch a ThunderAgent proxy command."
    )
    parser.add_argument("--backends", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--backend-type", default="vllm")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--profile-dir", default="/tmp/thunderagent_profiles")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ThunderAgentConfig(
        backends=args.backends,
        host=args.host,
        port=args.port,
        backend_type=args.backend_type,
        profile=args.profile,
        metrics=args.metrics,
        profile_dir=args.profile_dir,
    )
    command = build_thunderagent_command(config)
    print_or_exec_command(config=config, command=command, print_only=args.print_only)


if __name__ == "__main__":
    main()
