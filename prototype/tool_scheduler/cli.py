"""CLI entry point for tool_scheduler.

Usage:
    python -m prototype.tool_scheduler --output profiles.jsonl --dry-run -- <command...>
    python tool_scheduler.py --dry-run -- <command...>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .runner import run_tool
from .cost_model import CostModelConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Online load prediction & hardware-aware scheduling for tool invocations. "
            "Monitors process tree, predicts CPU demand, and generates dry-run "
            "placement recommendations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m prototype.tool_scheduler --dry-run -- python3 run.py
  python -m prototype.tool_scheduler --dry-run -- make -j "$(nproc)"
  python -m prototype.tool_scheduler --save-samples --dry-run -- pytest -q
  python -m prototype.tool_scheduler --output profiles.jsonl --dry-run -- python3 run.py
        """,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="profiles.jsonl",
        help="JSONL output file path (default: profiles.jsonl).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only output recommendations, do not apply CPU affinity (default).",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Include all raw sample data in JSONL output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample summaries to stderr.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="EMA alpha for core prediction (default: 0.3).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Minimum seconds between consecutive recommendations (default: 5.0).",
    )
    parser.add_argument(
        "--cost-config",
        type=str,
        default=None,
        help="Path to JSON cost model configuration file.",
    )
    parser.add_argument(
        "--history-db",
        type=str,
        default=None,
        help="Path to JSON history database file (read/write).",
    )
    parser.add_argument(
        "--memory-sensitivity",
        type=str,
        default=None,
        help=(
            "Memory sensitivity overrides as JSON: "
            '{"python3 run.py": "high", "make -j": "low"}'
        ),
    )
    parser.add_argument(
        "--bandwidth-config",
        type=str,
        default=None,
        help=(
            "Path to JSON memory domain bandwidth configuration. "
            "Auto-detect PMU if not provided."
        ),
    )
    parser.add_argument(
        "--hardcode-topology",
        action="store_true",
        help=(
            "Use hardcoded 320-core Kunpeng-style topology instead of sysfs "
            "discovery. Useful for testing on non-Linux or when sysfs is "
            "unavailable."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to profile and schedule (after --).",
    )

    args = parser.parse_args(argv)

    # Clean up command
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    if not args.command:
        parser.error("No command specified. Use: tool_scheduler.py -- <command>")

    return args


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Load cost config
    cost_config = None
    if args.cost_config:
        try:
            with open(args.cost_config, "r", encoding="utf-8") as f:
                cost_config = CostModelConfig.from_dict(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[tool-scheduler] WARNING: cannot load cost config: {e}", file=sys.stderr)

    # Load history DB
    history_db: dict[str, dict] = {}
    if args.history_db:
        try:
            with open(args.history_db, "r", encoding="utf-8") as f:
                history_db = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass  # Start fresh if unreadable

    # Parse memory sensitivity overrides
    memory_sensitivity_overrides = None
    if args.memory_sensitivity:
        try:
            memory_sensitivity_overrides = json.loads(args.memory_sensitivity)
        except json.JSONDecodeError as e:
            print(
                f"[tool-scheduler] WARNING: cannot parse --memory-sensitivity: {e}",
                file=sys.stderr,
            )

    exit_code = run_tool(
        command=args.command,
        output_path=args.output,
        dry_run=args.dry_run,
        save_samples=args.save_samples,
        verbose=args.verbose,
        cost_config=cost_config,
        history_db=history_db,
        memory_sensitivity_overrides=memory_sensitivity_overrides,
        cooldown_seconds=args.cooldown,
        alpha=args.alpha,
        bandwidth_config_path=args.bandwidth_config,
        hardcode_topology=args.hardcode_topology,
    )

    # Save history DB if path provided
    if args.history_db and history_db:
        try:
            with open(args.history_db, "w", encoding="utf-8") as f:
                json.dump(history_db, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"[tool-scheduler] WARNING: cannot save history: {e}", file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
