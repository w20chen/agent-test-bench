#!/usr/bin/env python3
"""
SWE-bench 实例检查工具 (Inspection Tool)
=========================================

一个**独立脚本**，用于查看 SWE-bench Verified 和 SWE-rebench 的具体 case，
无需接入 Agent 系统。适合给审阅者快速了解 benchmark 内容。

功能:
  list              列出所有可用任务 (instance_id + 简述)
  info <id>         查看某个任务的完整信息 (问题描述、测试命令、镜像名等)
  pull <id>         拉取 Docker 镜像 (可能需要较长时间)
  ls <id> [path]    列出容器 /testbed 目录下的文件
  cat <id> <path>   查看容器内的某个文件
  export <id> <src> <dst>  从容器中导出文件/目录到本地
  diff <id>         查看容器 /testbed 的 git diff (相对于 base_commit)
  shell <id>        进入容器的交互式 bash shell

用法:
  # 激活环境后直接运行
  conda activate ML
  python scripts/inspect_swebench.py --benchmark swe-bench-verified list
  python scripts/inspect_swebench.py --benchmark swe-bench-verified info sympy__sympy-12345
  python scripts/inspect_swebench.py --benchmark swe-bench-verified pull sympy__sympy-12345
  python scripts/inspect_swebench.py --benchmark swe-bench-verified ls sympy__sympy-12345
  python scripts/inspect_swebench.py --benchmark swe-rebench list
  python scripts/inspect_swebench.py --benchmark swe-rebench pull <instance_id>

依赖:
  pip install datasets docker
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Docker / Podman 辅助函数
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
    """Pull a Docker image, with retries."""
    print(f"[pull] 正在拉取 {image} ...")
    _run(CONTAINER_EXEC, "pull", image, timeout=3600)
    print(f"[pull] 完成: {image}")


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
        print(f"[!] 镜像 {image} 不存在，请先 pull。")
        sys.exit(1)
    cid = docker_run_detached(image)
    try:
        return action(cid, *args)
    finally:
        docker_stop_rm(cid)


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------

# 对于 datasets>=3.0 的向后兼容
try:
    import datasets.features.features as _ff
    from datasets.features.features import LargeList, _FEATURE_TYPES
    if "List" not in _FEATURE_TYPES:
        _FEATURE_TYPES["List"] = LargeList
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
    """从 HuggingFace 加载数据集并返回任务列表。"""
    from datasets import load_dataset

    cfg = BENCHMARK_CONFIGS[benchmark]
    ds = load_dataset(cfg["dataset"], split=cfg["split"])
    tasks: list[dict[str, Any]] = []
    for row in ds:
        task = dict(row)
        # 添加派生字段
        task["_benchmark"] = benchmark
        tasks.append(task)
    return tasks


def find_task(tasks: list[dict[str, Any]], instance_id: str) -> dict[str, Any] | None:
    for t in tasks:
        if t.get("instance_id") == instance_id:
            return t
    # 模糊匹配
    matches = [t for t in tasks if instance_id in t.get("instance_id", "")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[!] 多个匹配: {[m['instance_id'] for m in matches]}")
        return None
    return None


def get_image_name(task: dict[str, Any]) -> str:
    """获取任务的 Docker 镜像名称。"""
    benchmark = task.get("_benchmark", "")
    # SWE-rebench 有明确的 docker_image 字段
    if benchmark == "swe-rebench":
        img = task.get("docker_image") or task.get("image_name")
        if img:
            return str(img)
    # SWE-bench Verified: 从 instance_id 推导
    img = task.get("image_name") or task.get("docker_image")
    if img:
        return str(img)
    instance_id = task.get("instance_id", "")
    docker_compatible_id = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------

def cmd_list(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """列出所有任务。"""
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
        print(f"--- 匹配 {shown} / {total} 个任务 (关键词: '{keyword}') ---")
    else:
        print(f"--- 显示前 {shown} / {total} 个任务 ---")


def cmd_info(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """显示单个任务的详细信息。"""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] 未找到任务: {args.instance_id}")
        sys.exit(1)

    iid = task.get("instance_id", "?")
    print(f"{'='*60}")
    print(f"  instance_id:     {iid}")
    print(f"  repo:            {task.get('repo', '?')}")
    print(f"  base_commit:     {task.get('base_commit', '?')}")
    print(f"  image_name:      {get_image_name(task)}")
    print(f"  {('-'*56)}")

    # 问题描述
    problem = task.get("problem_statement", task.get("problem", ""))
    print(f"  [问题描述 / Problem Statement]")
    print(textwrap.indent(problem[:2000] if problem else "(无)", "    "))
    if len(problem) > 2000:
        print(f"    ... (截断，共 {len(problem)} 字符)")

    # 测试
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
    for t in ftp[:20]:
        print(f"    - {t}")
    if len(ftp) > 20:
        print(f"    ... 共 {len(ftp)} 个")

    print(f"  PASS_TO_PASS ({len(ptp)} tests):")
    for t in ptp[:10]:
        print(f"    - {t}")
    if len(ptp) > 10:
        print(f"    ... 共 {len(ptp)} 个")

    # 测试命令
    test_cmd = task.get("test_cmd", "")
    if test_cmd:
        print(f"  [Generated Test Command]")
        print(f"    {test_cmd}")

    # SWE-rebench 特有字段
    if task.get("_benchmark") == "swe-rebench":
        install_cfg = task.get("install_config")
        if install_cfg:
            print(f"  install_config: {json.dumps(install_cfg, indent=4, default=str)[:500]}")

    # 其他所有字段
    print(f"  {('-'*56)}")
    print(f"  [所有字段]")
    skip_keys = {"problem_statement", "problem", "FAIL_TO_PASS", "PASS_TO_PASS",
                 "test_cmd", "image_name", "docker_image", "install_config",
                 "instance_id", "repo", "base_commit", "_benchmark"}
    for k, v in task.items():
        if k in skip_keys or k.startswith("_"):
            continue
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        print(f"    {k}: {v_str}")


def cmd_pull(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """拉取 Docker 镜像。"""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] 未找到: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)
    docker_pull(image)


def cmd_ls(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """列出容器 /testbed 的文件。"""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] 未找到: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)
    path = args.path or "/testbed"

    def do_ls(cid: str, p: str) -> None:
        print(f"[ls] /testbed 下的文件 (容器: {cid[:12]}):")
        try:
            out = docker_exec(cid, "ls", "-lah", p)
            print(out)
        except subprocess.CalledProcessError as e:
            print(f"[!] 错误: {e.stderr}")

    with_testbed_container(image, do_ls, path)


def cmd_cat(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """查看容器内文件。"""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] 未找到: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)

    def do_cat(cid: str, filepath: str) -> None:
        print(f"[cat] {filepath}:")
        print("-" * 60)
        try:
            out = docker_exec(cid, "cat", filepath)
            print(out)
        except subprocess.CalledProcessError as e:
            print(f"[!] 错误: {e.stderr}")
        print("-" * 60)

    with_testbed_container(image, do_cat, args.path)


def cmd_export(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """从容器导出文件/目录。"""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] 未找到: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def do_export(cid: str, src: str, dst_path: str) -> None:
        print(f"[export] {src} -> {dst_path}")
        docker_cp(cid, src, dst_path)
        print(f"[export] 完成")

    with_testbed_container(image, do_export, args.src, str(dst.absolute()))


def cmd_diff(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """显示容器内 /testbed 的 git diff。"""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] 未找到: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)
    base = task.get("base_commit", "HEAD~1")

    def do_diff(cid: str, base_commit: str) -> None:
        print(f"[diff] base_commit = {base_commit}")
        print("-" * 60)
        try:
            out = docker_exec(cid, "sh", "-c", f"cd /testbed && git diff --stat {base_commit}")
            print(out)
        except subprocess.CalledProcessError as e:
            print(f"[!] git diff 失败: {e.stderr}")
        print("-" * 60)

    with_testbed_container(image, do_diff, base)


def cmd_shell(tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """进入容器交互式 shell。"""
    task = find_task(tasks, args.instance_id)
    if not task:
        print(f"[!] 未找到: {args.instance_id}")
        sys.exit(1)
    image = get_image_name(task)

    if not docker_image_exists(image):
        print(f"[!] 镜像 {image} 不存在，请先 pull。")
        sys.exit(1)

    print(f"[shell] 正在启动容器并进入 /testbed ...")
    print(f"  退出: 输入 exit 或 Ctrl+D")
    # 直接 exec，继承终端
    os.execvp(CONTAINER_EXEC, [
        CONTAINER_EXEC, "run", "--rm", "-it",
        "-w", "/testbed",
        image,
        "/bin/bash",
    ])


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-bench 实例检查工具 — 查看/浏览 SWE-bench 任务，无需 Agent 系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        示例:
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
        help="选择 benchmark (默认: swe-bench-verified)",
    )
    parser.add_argument(
        "--cache-file", "-c",
        type=Path,
        default=None,
        help="使用本地 JSON 缓存而非每次从 HF 下载 (如 data/swebench_verified/tasks.json)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="列出所有任务")
    p_list.add_argument("-n", type=int, default=20, help="显示前 N 个 (默认 20)")
    p_list.add_argument("-k", "--keyword", type=str, default="", help="按关键词过滤")

    # info
    p_info = sub.add_parser("info", help="查看任务详细信息")
    p_info.add_argument("instance_id", help="任务 instance_id")

    # pull
    p_pull = sub.add_parser("pull", help="拉取 Docker 镜像")
    p_pull.add_argument("instance_id", help="任务 instance_id")

    # ls
    p_ls = sub.add_parser("ls", help="列出容器内文件")
    p_ls.add_argument("instance_id", help="任务 instance_id")
    p_ls.add_argument("path", nargs="?", default="/testbed", help="路径 (默认 /testbed)")

    # cat
    p_cat = sub.add_parser("cat", help="查看容器内文件")
    p_cat.add_argument("instance_id", help="任务 instance_id")
    p_cat.add_argument("path", help="文件路径 (如 /testbed/setup.py)")

    # export
    p_export = sub.add_parser("export", help="从容器导出文件/目录")
    p_export.add_argument("instance_id", help="任务 instance_id")
    p_export.add_argument("src", help="容器内路径 (如 /testbed)")
    p_export.add_argument("dst", help="本地目标路径")

    # diff
    p_diff = sub.add_parser("diff", help="查看 git diff")
    p_diff.add_argument("instance_id", help="任务 instance_id")

    # shell
    p_shell = sub.add_parser("shell", help="进入容器交互式 bash")
    p_shell.add_argument("instance_id", help="任务 instance_id")

    args = parser.parse_args()

    # 加载任务
    if args.cache_file and args.cache_file.exists():
        print(f"[load] 从缓存加载: {args.cache_file}")
        tasks = json.loads(args.cache_file.read_text(encoding="utf-8"))
        for t in tasks:
            t["_benchmark"] = args.benchmark
    else:
        print(f"[load] 从 HuggingFace 加载 {BENCHMARK_CONFIGS[args.benchmark]['desc']} ...")
        tasks = load_tasks(args.benchmark)
        print(f"[load] 加载了 {len(tasks)} 个任务")

    # 执行命令
    command_map = {
        "list": cmd_list,
        "info": cmd_info,
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
