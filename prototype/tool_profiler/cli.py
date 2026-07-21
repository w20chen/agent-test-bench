"""CLI entry point for tool_profiler.

Usage:
    python -m prototype.tool_profiler -- <command...>
    python tool_profiler.py -- <command...>
"""

from __future__ import annotations

import argparse
import logging
import sys

from .runner import run_tool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. If None, uses sys.argv[1:].

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Profile an external tool invocation by sampling its process tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m prototype.tool_profiler -- python run.py
  python -m prototype.tool_profiler --warmup-seconds 3 -- make -j8
  python -m prototype.tool_profiler --save-samples -- pytest -q
  python -m prototype.tool_profiler --shell-command -- "pytest -q && echo ok"
        """,
    )

    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Early profile observation window in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.2,
        help="Time between consecutive samples in seconds (default: 0.2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tool_profiles.jsonl",
        help="JSONL output file path (default: tool_profiles.jsonl).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample summaries to stderr.",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Include all raw sample data in JSONL output.",
    )
    parser.add_argument(
        "--shell-command",
        action="store_true",
        help=(
            "Treat the command after -- as one shell command string. This is "
            "intended for wrapping an existing shell command without changing "
            "operators such as &&, pipes, redirects, or environment assignment."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to profile (after --).",
    )

    args = parser.parse_args(argv)

    # Clean up the command: argparse.REMAINDER includes the first
    # positional after '--' but may also include a literal '--'.
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    if not args.command:
        parser.error("No command specified. Use: tool_profiler.py -- <command>")
    if args.shell_command:
        if len(args.command) != 1:
            parser.error(
                "--shell-command requires exactly one shell command string after --"
            )

    return args


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Exits with the same exit code as the profiled tool.
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    exit_code = run_tool(
        command=args.command,
        warmup_seconds=args.warmup_seconds,
        sample_interval=args.sample_interval,
        output_path=args.output,
        verbose=args.verbose,
        save_samples=args.save_samples,
        shell_command=args.shell_command,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
