from __future__ import annotations

import argparse
import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

from harness.scheduler_hooks import apply_scheduler_hook


def normalize_forwarded_args(args: list[str]) -> list[str]:
    """Drop a leading `--` separator before forwarding argv to vLLM."""
    if args and args[0] == "--":
        return args[1:]
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch vLLM with automatic scheduler hooks."
    )
    parser.add_argument("--hook-report-path", required=True)
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = apply_scheduler_hook()
    report_path = Path(args.hook_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(status), indent=2) + "\n", encoding="utf-8"
    )
    sys.argv = [
        "vllm.entrypoints.openai.api_server",
        *normalize_forwarded_args(args.remainder),
    ]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
