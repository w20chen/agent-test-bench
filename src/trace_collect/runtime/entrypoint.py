"""In-container entrypoint for scaffold execution."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

from llm_call import UnifiedProvider, resolve_llm_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trace_collect.runtime.entrypoint",
    )
    parser.add_argument("--mode", choices=["preflight", "run"], required=True)
    return parser.parse_args()


def _load_request() -> dict[str, Any]:
    raw = sys.stdin.read()
    return json.loads(raw) if raw.strip() else {}


def _runtime_proof(container_id: str | None = None) -> dict[str, Any]:
    return {
        "container_id": container_id,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "project_root": str(Path(__file__).resolve().parents[3]),
        "sys_path": sys.path[:8],
    }


def _write_result(result_path: str, payload: dict[str, Any]) -> None:
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _trace_summary_totals(
    trace_file: Path,
) -> tuple[float | None, float | None, int | None]:
    total_llm_ms: float | None = None
    total_tool_ms: float | None = None
    total_tokens: int | None = None
    if not trace_file.exists():
        return total_llm_ms, total_tool_ms, total_tokens
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "summary":
            continue
        total_llm_ms = record.get("total_llm_ms")
        total_tool_ms = record.get("total_tool_ms")
        total_tokens = record.get("total_tokens")
    return total_llm_ms, total_tool_ms, total_tokens


def _run_preflight(request: dict[str, Any]) -> dict[str, Any]:
    probe = Path(request["writable_probe"])
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)

    for module_name in request.get("imports", []):
        __import__(module_name)

    payload = {
        "ok": True,
        "runtime_proof": _runtime_proof(request.get("container_id")),
    }
    _write_result(request["result_path"], payload)
    return payload


async def _run_openclaw(request: dict[str, Any]) -> dict[str, Any]:
    from agents.benchmarks import get_benchmark_class
    from agents.benchmarks.base import BenchmarkConfig
    from agents.openclaw.eval.types import EvalTask
    from trace_collect.collector import load_mcp_servers

    repo_root = Path(__file__).resolve().parents[3]
    benchmark_slug = request["benchmark"]
    config = BenchmarkConfig.from_yaml(
        repo_root / "configs" / "benchmarks" / f"{benchmark_slug}.yaml"
    )
    benchmark = get_benchmark_class(benchmark_slug)(config)
    exec_path_append = request.get("exec_path_append", "")
    llm_config = resolve_llm_config(
        provider=request.get("provider_name"),
        api_base=request.get("api_base"),
        api_key=request.get("api_key"),
        model=request.get("model"),
        environ={},
    )
    provider = UnifiedProvider(
        api_key=llm_config.api_key,
        api_base=llm_config.api_base,
        default_model=llm_config.model,
        **dict(request.get("generation_config") or {}),
    )
    runner = benchmark.build_runner(
        scaffold="openclaw",
        provider=provider,
        workspace_base=Path(request["workspace_base"]),
        max_iterations=int(request["max_iterations"]),
        context_window_tokens=int(request["max_context_tokens"]),
        model=llm_config.model,
        mcp_servers=load_mcp_servers(request.get("mcp_config")),
        exec_path_append=exec_path_append,
        generation_config=dict(request.get("generation_config") or {}),
    )
    task_raw = dict(request["task"])
    eval_task = EvalTask(
        instance_id=task_raw["instance_id"],
        problem_statement=task_raw.get("problem_statement", ""),
        workspace_dir=Path(request["workspace_dir"]),
        repo=task_raw.get("repo"),
        base_commit=task_raw.get("base_commit"),
        image_name=task_raw.get("image_name"),
    )
    result = await runner.run_task(
        eval_task,
        prompt_template=request["prompt_template"],
        tool_workspace=Path(request["tool_workspace"]),
        exec_working_dir=request.get("exec_working_dir"),
        trace_file=Path(request["trace_file"]),
    )
    total_llm_ms, total_tool_ms, total_tokens = _trace_summary_totals(
        Path(request["trace_file"])
    )
    payload = {
        "trace_path": str(request["trace_file"]),
        "model_patch": result.model_patch,
        "success": bool(result.model_patch),
        "exit_status": result.stop_reason,
        "error": result.error,
        "n_iterations": result.n_iterations,
        "total_llm_ms": total_llm_ms,
        "total_tool_ms": total_tool_ms,
        "total_tokens": total_tokens,
        "runtime_proof": {
            **_runtime_proof(request.get("container_id")),
            "agent_runtime_mode": request.get(
                "agent_runtime_mode", "task_container_agent"
            ),
        },
    }
    _write_result(request["result_path"], payload)
    return payload


async def _run_request(request: dict[str, Any]) -> dict[str, Any]:
    kind = request["kind"]
    if kind == "run_openclaw":
        return await _run_openclaw(request)
    raise ValueError(f"unsupported runtime request kind: {kind!r}")


def main() -> None:
    args = _parse_args()
    request = _load_request()

    try:
        if args.mode == "preflight":
            _run_preflight(request)
            return
        stdout_path = Path(request["raw_stdout_path"])
        stderr_path = Path(request["raw_stderr_path"])
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
            contextlib.redirect_stdout(stdout_handle),
            contextlib.redirect_stderr(stderr_handle),
        ):
            asyncio.run(_run_request(request))
    except Exception:
        if "result_path" in request:
            _write_result(
                request["result_path"],
                {
                    "error": traceback.format_exc(),
                    "trace_path": request.get("trace_file", ""),
                    "model_patch": "",
                    "exit_status": "error",
                    "n_iterations": None,
                    "total_llm_ms": None,
                    "total_tool_ms": None,
                    "total_tokens": None,
                    "runtime_proof": {
                        **_runtime_proof(request.get("container_id")),
                        "agent_runtime_mode": request.get(
                            "agent_runtime_mode", "task_container_agent"
                        ),
                    },
                },
            )
        raise


if __name__ == "__main__":
    main()
