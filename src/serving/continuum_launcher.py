from __future__ import annotations

import argparse
from dataclasses import dataclass

from serving._common import print_or_exec_command, resolve_python_sibling_executable


@dataclass(slots=True)
class ContinuumServerConfig:
    """Configuration for launching the Continuum-modified vLLM server."""

    model_path: str
    port: int = 8001
    tensor_parallel_size: int = 1
    enable_cpu_offload: bool = False
    cpu_offload_gib: int = 200


def build_continuum_command(config: ContinuumServerConfig) -> list[str]:
    """Build the `vllm serve` command for Continuum scheduling mode."""
    command = [
        resolve_python_sibling_executable("vllm"),
        "serve",
        config.model_path,
        "--scheduling-policy",
        "continuum",
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--port",
        str(config.port),
    ]
    if config.enable_cpu_offload:
        command.extend(
            [
                "--kv-transfer-config",
                '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
            ]
        )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally launch a Continuum scheduling server command."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--cpu-offload-gib", type=int, default=200)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ContinuumServerConfig(
        model_path=args.model_path,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_cpu_offload=args.enable_cpu_offload,
        cpu_offload_gib=args.cpu_offload_gib,
    )
    command = build_continuum_command(config)
    print_or_exec_command(config=config, command=command, print_only=args.print_only)


if __name__ == "__main__":
    main()
