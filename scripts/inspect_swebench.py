#!/usr/bin/env python3
"""
SWE-bench Inspection Tool
==========================

A **standalone script** for inspecting SWE-bench Verified and SWE-rebench
test cases. No agent system integration needed. Ideal for reviewers who want
to quickly understand benchmark content.

Commands:
  list              List all available tasks (instance_id + summary)
  info <id>         View full task details (problem statement, test command, image name, etc.)
  pull <id>         Pull the Docker image (may take a while)
  ls <id> [path]    List files under /testbed in the container
  cat <id> <path>   View a file inside the container
  export <id> <src> <dst>  Export files/directories from container to local
  diff <id>         Show git diff of /testbed (relative to base_commit)
  shell <id>        Enter an interactive bash shell in the container

Usage:
  conda activate ML
  python scripts/inspect_swebench.py --benchmark swe-bench-verified list
  python scripts/inspect_swebench.py --benchmark swe-bench-verified info sympy__sympy-12345
  python scripts/inspect_swebench.py --benchmark swe-bench-verified pull sympy__sympy-12345
  python scripts/inspect_swebench.py --benchmark swe-bench-verified ls sympy__sympy-12345
  python scripts/inspect_swebench.py --benchmark swe-rebench list
  python scripts/inspect_swebench.py --benchmark swe-rebench pull <instance_id>

Dependencies:
  pip install datasets docker
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Docker / Podman helpers
# ---------------------------------------------------------------------------

CONTAINER_EXEC = os.environ.get("CONTAINER_EXEC", "docker")


def _run(*args: str, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    cmd = [str(a) for a in args]
    return subprocess.run(cmd, check=check, text=True, timeout=timeout)


def _run_capture(*args: str, timeout: int | None = None) -> str:
    """Run a command, return trimmed stdout.  Raise on failure."""
    result = subprocess.run(
        [str(a) for a in args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def docker_pull(image: str) -> None:
    """Pull a Docker image."""
    print(f"[pull] Pulling {image} ...")
    _run(CONTAINER_EXEC, "pull", image, timeout=3600)
    print(f"[pull] Done: {image}")


def docker_image_exists(image: str) -> bool:
    result = subprocess.run(
        [CONTAINER_EXEC, "image", "inspect", image],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


def docker_run_detached(image: str) -> str:
    """Start a container from image in detached mode, return container ID."""
    cid = _run_capture(CONTAINER_EXEC, "run", "-d", image, "sleep", "3600")
    return cid


def docker_stop_rm(container_id: str) -> None:
    subprocess.run([CONTAINER_EXEC, "stop", container_id], check=False, capture_output=True)
    subprocess.run([CONTAINER_EXEC, "rm", "-f", container_id], check=False, capture_output=True)


def docker_exec(container_id: str, *args: str) -> str:
    """Execute a command inside container, return stdout."""
    return _run_capture(CONTAINER_EXEC, "exec", container_id, *args, timeout=120)


def docker_cp(container_id: str, src: str, dst: str) -> None:
    """Copy file/dir from container to local host."""
    _run(CONTAINER_EXEC, "cp", f"{container_id}:{src}", dst, timeout=120)


def with_testbed_container(image: str, action: callable, *args: Any) -> Any:
    """Start a temp container from `image`, run `action(container_id, *args)`, clean up."""
    if not docker_image_exists(image):
        print(f"[!] Image {image} not found locally. Please pull first.")
        sys.exit(1)
    cid = docker_run_detached(image)
    try:
        return action(cid, *args)
    finally:
        docker_stop_rm(cid)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# Backward compatibility for datasets>=3.0
#
# Two issues:
# (1) datasets>=3.0 renamed `List` to `LargeList`, but cached dataset_info.json
#     still references the `List` type name → register the alias.
# (2) Arrow files encode list-typed columns as ``[Value(...)]`` (→ Sequence),
#     while cached dataset_info.json may encode them as ``LargeList(...)``.
#     Patch ``Features.reorder_fields_as`` to normalise before comparing.
try:
    import datasets.features.features as _ff
    from datasets.features.features import LargeList, _FEATURE_TYPES

    if "List" not in _FEATURE_TYPES:
        _FEATURE_TYPES["List"] = LargeList

    def _normalise_largelist(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _normalise_largelist(v) for k, v in obj.items()}
        if isinstance(obj, LargeList):
            return [obj.feature]
        return obj

    _orig_reorder_fields_as = _ff.Features.reorder_fields_as

    def _patched_reorder_fields_as(
        self: _ff.Features, other: _ff.Features
    ) -> _ff.Features:
        return _orig_reorder_fields_as(_normalise_largelist(self), other)

    _ff.Features.reorder_fields_as = _patched_reorder_fields_as  # type: ignore[assignment]
except ImportError:
    pass


BENCHMARK_CONFIGS = {
    "swe-bench-verified": {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "desc": "SWE-Bench Verified (princeton-nlp)",
    },
    "swe-rebench": {
        "dataset": "nebius/SWE-rebench",
        "split": "filtered",
        "desc": "SWE-rebench / filtered split (nebius)",
    },
}


def load_tasks(benchmark: str) -> list[dict[str, Any]]:
    """Load dataset from HuggingFace and return the task list."""
    from datasets import load_dataset

    cfg = BENCHMARK_CONFIGS[benchmark]
    ds = load_dataset(cfg["dataset"], split=cfg["split"])
    tasks: list[dict[str, Any]] = []
    for row in ds:
        task = dict(row)
        task["_benchmark"] = benchmark
        tasks.append(task)
    return tasks


def find_task(tasks: list[dict[str, Any]], instance_id: str) -> dict[str, Any] | None:
    for t in tasks:
        if t.get("instance_id") == instance_id:
            return t
    # Fuzzy match
    matches = [t for t in tasks if instance_id in t.get("instance_id", "")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[!] Multiple matches: {[m['instance_id'] for m in matches]}")
        return None
    return None


def get_image_name(task: dict[str, Any]) -> str:
    """Get the Docker image name for a task."""
    benchmark = task.get("_benchmark", "")
    # SWE-rebench has an explicit docker_image field
    if benchmark == "swe-rebench":
        img = task.get("docker_image") or task.get("image_name")
        if img:
            return str(img)
    # SWE-bench Verified: derive from instance_id
    img = task.get("image_name") or task.get("docker_image")
    if img:
        return str(img)
    instance_id = task.get("instance_id", "")
    docker_compatible_id = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_list(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """List all tasks."""
    n = args.n or len(tasks)
    keyword = args.keyword or ""

    shown = 0
    for t in tasks:
        if keyword and keyword.lower() not in json.dumps(t, default=str).lower():
            continue
        iid = t.get("instance_id", "?")
        repo = t.get("repo", "?")
        image = get_image_name(t)
        ftp = t.get("FAIL_TO_PASS", [])
        if isinstance(ftp, str):
            try:
                ftp = json.loads(ftp)
            except (json.JSONDecodeError, TypeError):
                ftp = []
        n_ftp = len(ftp) if isinstance(ftp, list) else 0
        print(f"  {iid}")
        print(f"    repo: {repo}")
        print(f"    image: {image}")
        print(f"    FAIL_TO_PASS tests: {n_ftp}")
        print()
        shown += 1
        if shown >= n:
            break

    total = len(tasks)
    if keyword:
        print(f"--- Matched {shown} / {total} tasks (keyword: '{keyword}') ---")
    else:
        print(f"--- Showing first {shown} / {total} tasks ---")


def cmd_info(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Show detailed information for a single task."""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] Task not found: {args.instance_id}")
        sys.exit(1)

    iid = task.get("instance_id", "?")
    print(f"{'='*60}")
    print(f"  instance_id:     {iid}")
    print(f"  repo:            {task.get('repo', '?')}")
    print(f"  base_commit:     {task.get('base_commit', '?')}")
    print(f"  image_name:      {get_image_name(task)}")
    print(f"  {('-'*56)}")

    # Problem statement
    problem = task.get("problem_statement", task.get("problem", ""))
    print(f"  [Problem Statement]")
    print(textwrap.indent(problem if problem else "(none)", "    "))

    # Tests
    ftp = task.get("FAIL_TO_PASS", [])
    if isinstance(ftp, str):
        try:
            ftp = json.loads(ftp)
        except (json.JSONDecodeError, TypeError):
            ftp = []
    ptp = task.get("PASS_TO_PASS", [])
    if isinstance(ptp, str):
        try:
            ptp = json.loads(ptp)
        except (json.JSONDecodeError, TypeError):
            ptp = []

    print(f"  {('-'*56)}")
    print(f"  FAIL_TO_PASS ({len(ftp)} tests):")
    for t in ftp:
        print(f"    - {t}")

    print(f"  PASS_TO_PASS ({len(ptp)} tests):")
    for t in ptp:
        print(f"    - {t}")

    # Test command
    test_cmd = task.get("test_cmd", "")
    if test_cmd:
        print(f"  [Generated Test Command]")
        print(f"    {test_cmd}")

    # SWE-rebench specific fields
    if task.get("_benchmark") == "swe-rebench":
        install_cfg = task.get("install_config")
        if install_cfg:
            print(f"  install_config: {json.dumps(install_cfg, indent=4, default=str)}")

    # All other fields
    print(f"  {('-'*56)}")
    print(f"  [All Fields]")
    skip_keys = {"problem_statement", "problem", "FAIL_TO_PASS", "PASS_TO_PASS",
                 "test_cmd", "image_name", "docker_image", "install_config",
                 "instance_id", "repo", "base_commit", "_benchmark"}
    for k, v in task.items():
        if k in skip_keys or k.startswith("_"):
            continue
        print(f"    {k}: {v}")


def _parse_test_node_id(test_name: str) -> tuple[str | None, str]:
    """Parse a pytest node ID into (file_path, test_spec).

    Examples:
        "test_func (module.path.ClassName)" -> ("tests/module/path.py", "ClassName.test_func")
        "test_func"                           -> (None, "test_func")
        "Plain description text"              -> (None, "Plain description text")
    """
    # Match pytest verbose format: "test_name (module.path.ClassName)"
    m = re.match(r'^(.+?)\s+\((.+?)\)\s*$', test_name)
    if m:
        test_spec = m.group(1).strip()
        module_path = m.group(2).strip()
        # module_path like "auth_tests.test_validators.UsernameValidatorsTests"
        # The file is in tests/<module_path with . -> />.py, strip last segment if it's a class
        parts = module_path.split(".")
        # Remove the class name (last segment) to get the module file path
        file_module = ".".join(parts[:-1]) if parts[-1][0].isupper() else module_path
        file_path = "tests/" + file_module.replace(".", "/") + ".py"
        # If the class part was the module, re-check
        return (file_path, f"{parts[-1]}.{test_spec}" if parts[-1][0].isupper() else test_spec)
    # Try simpler format: "module.path.test_func" (no class in parens)
    m2 = re.match(r'^([\w.]+)\.(\w+)$', test_name)
    if m2:
        module_path = m2.group(1)
        test_spec = m2.group(2)
        file_path = "tests/" + module_path.replace(".", "/") + ".py"
        return (file_path, test_spec)
    return (None, test_name)


def _collect_test_files(task: dict[str, Any]) -> dict[str, list[str]]:
    """Parse FAIL_TO_PASS and return {file_path: [test_spec, ...]}."""
    ftp = task.get("FAIL_TO_PASS", [])
    if isinstance(ftp, str):
        try:
            ftp = json.loads(ftp)
        except (json.JSONDecodeError, TypeError):
            ftp = []

    files: dict[str, list[str]] = {}
    unknown: list[str] = []
    for test_name in ftp:
        file_path, test_spec = _parse_test_node_id(str(test_name))
        if file_path:
            files.setdefault(file_path, []).append(test_spec)
        else:
            unknown.append(test_spec)
    if unknown:
        files["(unmapped — plain descriptions)"] = unknown
    return files


def cmd_tests(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Show test files involved in FAIL_TO_PASS."""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] Task not found: {args.instance_id}")
        sys.exit(1)

    test_files = _collect_test_files(task)

    if args.file:
        # Show a specific test file from the container
        image = get_image_name(task)
        if not docker_image_exists(image):
            print(f"[!] Image {image} not found locally. Please pull first.")
            sys.exit(1)
        target = args.file if args.file.startswith("/") else f"/testbed/{args.file}"
        print(f"[tests] {target}:")
        print("-" * 60)

        def do_cat(cid: str, fp: str) -> None:
            try:
                out = docker_exec(cid, "cat", fp)
                print(out)
            except subprocess.CalledProcessError as e:
                print(f"[!] Error: {e.stderr}")

        with_testbed_container(image, do_cat, target)
        print("-" * 60)
        return

    # Summary mode
    total_tests = sum(len(v) for v in test_files.values())
    print(f"FAIL_TO_PASS: {total_tests} tests across {len(test_files)} files")
    print()
    for file_path, tests in sorted(test_files.items()):
        print(f"  {file_path}  ({len(tests)} tests)")
        for t in tests[:8]:
            print(f"      - {t}")
        if len(tests) > 8:
            print(f"      ... and {len(tests) - 8} more")
        print()

    print("---")
    print("To view a specific test file:")
    print(f"  python scripts/inspect_swebench.py -b {task.get('_benchmark', 'swe-bench-verified')} tests {task.get('instance_id', 'ID')} -f tests/auth_tests/test_validators.py")


def cmd_pull(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Pull Docker image."""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] Not found: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)
    docker_pull(image)


def cmd_ls(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """List files in container /testbed."""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] Not found: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)
    path = args.path or "/testbed"

    def do_ls(cid: str, p: str) -> None:
        print(f"[ls] Files under /testbed (container: {cid[:12]}):")
        try:
            out = docker_exec(cid, "ls", "-lah", p)
            print(out)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error: {e.stderr}")

    with_testbed_container(image, do_ls, path)


def cmd_cat(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """View a file inside the container."""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] Not found: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)

    def do_cat(cid: str, filepath: str) -> None:
        print(f"[cat] {filepath}:")
        print("-" * 60)
        try:
            out = docker_exec(cid, "cat", filepath)
            print(out)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error: {e.stderr}")
        print("-" * 60)

    with_testbed_container(image, do_cat, args.path)


def cmd_export(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Export files/directories from container."""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] Not found: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def do_export(cid: str, src: str, dst_path: str) -> None:
        print(f"[export] {src} -> {dst_path}")
        docker_cp(cid, src, dst_path)
        print(f"[export] Done")

    with_testbed_container(image, do_export, args.src, str(dst.absolute()))


def cmd_diff(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Show the gold fix patch (diff) that would resolve the task.

    By default, displays the ``patch`` and ``test_patch`` fields from the
    dataset — these are the ground-truth diffs that an agent is expected to
    reproduce.  Use ``--container`` to instead compute a live git diff inside
    the container (useful after you've made manual edits in a shell session).
    """
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] Not found: {args.instance_id}")
        sys.exit(1)

    if args.container:
        # Live git diff inside the container
        image = get_image_name(task)
        base = task.get("base_commit", "HEAD~1")

        def do_diff(cid: str, base_commit: str) -> None:
            print(f"[diff] base_commit = {base_commit}")
            print("-" * 60)
            try:
                out = docker_exec(cid, "sh", "-c", f"cd /testbed && git diff --stat {base_commit}")
                print(out if out else "(no uncommitted changes)")
            except subprocess.CalledProcessError as e:
                print(f"[!] git diff failed: {e.stderr}")
            print("-" * 60)

        with_testbed_container(image, do_diff, base)
        return

    # Default: show the gold patch from the dataset
    patch = task.get("patch", "")
    test_patch = task.get("test_patch", "")

    if patch:
        print(f"[Gold Fix Patch]  ({len(patch)} chars)")
        print("=" * 60)
        print(patch)
    else:
        print("[Gold Fix Patch]  (none — this task may not have a code patch)")

    if test_patch:
        print()
        print(f"[Test Patch]  ({len(test_patch)} chars)")
        print("=" * 60)
        print(test_patch)

    if not patch and not test_patch:
        print()
        print("Tip: use --container to run a live git diff inside the container instead.")


def cmd_shell(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Enter an interactive shell in the container."""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] Not found: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)

    if not docker_image_exists(image):
        print(f"[!] Image {image} not found locally. Please pull first.")
        sys.exit(1)

    print(f"[shell] Starting container and entering /testbed ...")
    print(f"  Exit: type exit or press Ctrl+D")
    os.execvp(CONTAINER_EXEC, [
        CONTAINER_EXEC, "run", "--rm", "-it",
        "-w", "/testbed",
        image,
        "/bin/bash",
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-bench Inspection Tool — browse SWE-bench tasks without an agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          %(prog)s --benchmark swe-bench-verified list
          %(prog)s --benchmark swe-bench-verified list -k django
          %(prog)s --benchmark swe-bench-verified info sympy__sympy-12345
          %(prog)s --benchmark swe-bench-verified pull sympy__sympy-12345
          %(prog)s --benchmark swe-bench-verified ls sympy__sympy-12345
          %(prog)s --benchmark swe-bench-verified cat sympy__sympy-12345 /testbed/setup.py
          %(prog)s --benchmark swe-bench-verified diff sympy__sympy-12345
          %(prog)s --benchmark swe-bench-verified export sympy__sympy-12345 /testbed ./export/
          %(prog)s --benchmark swe-bench-verified shell sympy__sympy-12345

          %(prog)s --benchmark swe-rebench list
          %(prog)s --benchmark swe-rebench info <id>
          %(prog)s --benchmark swe-rebench pull <id>
          %(prog)s --benchmark swe-rebench shell <id>
        """),
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=["swe-bench-verified", "swe-rebench"],
        default="swe-bench-verified",
        help="Select benchmark (default: swe-bench-verified)",
    )
    parser.add_argument(
        "--cache-file", "-c",
        type=Path,
        default=None,
        help="Use local JSON cache instead of downloading from HF each time (e.g. data/swebench_verified/tasks.json)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List all tasks")
    p_list.add_argument("-n", type=int, default=20, help="Show first N (default 20)")
    p_list.add_argument("-k", "--keyword", type=str, default="", help="Filter by keyword")

    # info
    p_info = sub.add_parser("info", help="Show task details")
    p_info.add_argument("instance_id", help="Task instance_id")

    # pull
    p_pull = sub.add_parser("pull", help="Pull Docker image")
    p_pull.add_argument("instance_id", help="Task instance_id")

    # ls
    p_ls = sub.add_parser("ls", help="List files in container")
    p_ls.add_argument("instance_id", help="Task instance_id")
    p_ls.add_argument("path", nargs="?", default="/testbed", help="Path (default /testbed)")

    # cat
    p_cat = sub.add_parser("cat", help="View file in container")
    p_cat.add_argument("instance_id", help="Task instance_id")
    p_cat.add_argument("path", help="File path (e.g. /testbed/setup.py)")

    # export
    p_export = sub.add_parser("export", help="Export files/dirs from container")
    p_export.add_argument("instance_id", help="Task instance_id")
    p_export.add_argument("src", help="Container path (e.g. /testbed)")
    p_export.add_argument("dst", help="Local destination path")

    # diff
    p_diff = sub.add_parser("diff", help="Show gold fix patch (use --container for live git diff)")
    p_diff.add_argument("instance_id", help="Task instance_id")
    p_diff.add_argument("--container", action="store_true", help="Run live git diff inside the container instead")

    # tests
    p_tests = sub.add_parser("tests", help="Show FAIL_TO_PASS test files grouped by source file")
    p_tests.add_argument("instance_id", help="Task instance_id")
    p_tests.add_argument("-f", "--file", type=str, default=None, help="View a specific test file (e.g. tests/auth_tests/test_validators.py)")

    # shell
    p_shell = sub.add_parser("shell", help="Enter interactive bash shell")
    p_shell.add_argument("instance_id", help="Task instance_id")

    args = parser.parse_args()

    # Load tasks
    if args.cache_file and args.cache_file.exists():
        print(f"[load] Loading from cache: {args.cache_file}")
        tasks = json.loads(args.cache_file.read_text(encoding="utf-8"))
        for t in tasks:
            t["_benchmark"] = args.benchmark
    else:
        print(f"[load] Loading {BENCHMARK_CONFIGS[args.benchmark]['desc']} from HuggingFace ...")
        tasks = load_tasks(args.benchmark)
        print(f"[load] Loaded {len(tasks)} tasks")

    # Dispatch command
    command_map = {
        "list": cmd_list,
        "info": cmd_info,
        "tests": cmd_tests,
        "pull": cmd_pull,
        "ls": cmd_ls,
        "cat": cmd_cat,
        "export": cmd_export,
        "diff": cmd_diff,
        "shell": cmd_shell,
    }
    handler = command_map.get(args.command)
    if handler:
        handler(tasks, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
