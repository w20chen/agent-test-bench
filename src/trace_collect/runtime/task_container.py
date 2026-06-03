"""Host-side helpers for task-container agent parity."""

from __future__ import annotations

import json
import os
import platform
import ssl
import subprocess
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.openclaw.runtime_deps import OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOTNAME = "_task_container_runtime"
_REDACTED_SECRET = "***REDACTED***"
_DEFAULT_RUNTIME_PYTHONPATH = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
_CONTAINER_SYSTEM_PYTHON = "/usr/bin/python3"
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple"
_GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
_GET_PIP_FETCH_ATTEMPTS = 3
_GET_PIP_FETCH_BACKOFF_SECONDS = 1.0
_ARCH_ALIASES = {
    "amd64": "amd64",
    "x86_64": "amd64",
    "arm64": "arm64",
    "aarch64": "arm64",
}
_CONTAINER_PYTHON_CANDIDATES = (
    "/usr/bin/python3",
    "/usr/bin/python",
    "/opt/conda/bin/python",
    "/opt/conda/envs/ML/bin/python",
    "python3",
    "python",
)


def _format_probe_failure_details(result: subprocess.CompletedProcess[str]) -> str:
    parts: list[str] = []
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stdout:
        parts.append(f"stdout: {stdout}")
    if stderr:
        parts.append(f"stderr: {stderr}")
    return "; ".join(parts)


def _bootstrap_marker_matches(
    marker: Path,
    *,
    requirements: tuple[str, ...],
    runtime: str,
) -> bool:
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload == {"requirements": list(requirements), "python": runtime}


def _is_retryable_get_pip_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        return isinstance(
            reason,
            (
                ssl.SSLError,
                TimeoutError,
                ConnectionResetError,
                OSError,
            ),
        )
    return isinstance(
        exc,
        (
            ssl.SSLError,
            TimeoutError,
            ConnectionResetError,
            OSError,
        ),
    )


def _download_get_pip(get_pip: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, _GET_PIP_FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(_GET_PIP_URL, timeout=120) as response:
                payload = response.read()
            tmp_path = get_pip.with_suffix(".tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(get_pip)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= _GET_PIP_FETCH_ATTEMPTS or not _is_retryable_get_pip_error(
                exc
            ):
                raise
            time.sleep(_GET_PIP_FETCH_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    if last_error is not None:
        raise last_error


@dataclass(slots=True)
class TaskContainerPreflightProof:
    hostname: str
    cwd: str
    python_executable: str
    python_prefix: str
    project_root: str
    sys_path: list[str]
    container_id: str | None = None


@dataclass(slots=True)
class TaskContainerRunResult:
    success: bool
    exit_status: str | None
    model_patch: str
    error: str | None
    n_iterations: int | None
    total_llm_ms: float | None
    total_tool_ms: float | None
    total_tokens: int | None
    runtime_proof: dict[str, Any]
    trace_path: Path
    raw_stdout_path: Path
    raw_stderr_path: Path


@dataclass(slots=True, frozen=True)
class TaskContainerExecConfig:
    runtime: str
    pythonpath: str
    start_extra_args: tuple[str, ...]
    bootstrap: bool = False
    bootstrap_site_dir: Path | None = None
    image_platform: str | None = None


def _normalize_arch(raw: str | None) -> str | None:
    if raw is None:
        return None
    return _ARCH_ALIASES.get(raw.lower(), raw.lower())


def _host_linux_platform() -> str | None:
    if platform.system() != "Linux":
        return None
    arch = _normalize_arch(platform.machine())
    if arch is None:
        return None
    return f"linux/{arch}"


def _inspect_image_platform(
    image: str,
    *,
    container_executable: str,
) -> str | None:
    if not image:
        return None
    result = subprocess.run(
        [
            container_executable,
            "image",
            "inspect",
            image,
            "--format",
            "{{.Architecture}} {{.Os}}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    if len(parts) != 2:
        return None
    arch, os_name = parts
    norm_arch = _normalize_arch(arch)
    if not norm_arch:
        return None
    return f"{os_name.lower()}/{norm_arch}"


def current_container_python_runtime() -> str:
    """Host-side Python; must run with conda env ML active."""
    if os.environ.get("CONDA_DEFAULT_ENV") != "ML":
        raise RuntimeError(
            "must run inside conda env 'ML' (got "
            f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}); "
            "run `conda activate ML` or use scripts/setup/bootstrap.sh"
        )
    return sys.executable


def task_container_runtime_dir(attempt_dir: Path, scaffold: str) -> Path:
    return attempt_dir.resolve() / RUNTIME_ROOTNAME / scaffold


def project_mount_args(
    attempt_dir: Path,
    *,
    include_host_system_mounts: bool | None = None,
) -> list[str]:
    """Return extra `podman run` args for parity mode."""

    task_container_runtime_dir(attempt_dir, "bootstrap").mkdir(
        parents=True, exist_ok=True
    )
    repo_root = REPO_ROOT.resolve()
    attempt_dir = attempt_dir.resolve()
    args: list[str] = []
    mounts: list[tuple[Path, bool]] = [
        (attempt_dir, False),
        (repo_root, False),
    ]
    if include_host_system_mounts is None:
        include_host_system_mounts = platform.system() == "Linux"
    if include_host_system_mounts:
        for raw in ("/usr", "/lib", "/lib64", "/etc", "/bin", "/sbin", "/tmp", "/var"):
            path = Path(raw)
            if path.exists():
                mounts.append((path, raw not in {"/tmp", "/var"}))

    seen: set[Path] = set()
    for path, read_only in mounts:
        if path in seen:
            continue
        seen.add(path)
        suffix = ":ro" if read_only else ""
        args.extend(["-v", f"{path}:{path}{suffix}"])
    return args


def resolve_task_container_exec_config(
    *,
    attempt_dir: Path,
    image: str,
    container_executable: str,
) -> TaskContainerExecConfig:
    image_platform = _inspect_image_platform(
        image,
        container_executable=container_executable,
    )
    host_platform = _host_linux_platform()
    use_host_runtime = host_platform is not None and (
        image_platform is None or image_platform == host_platform
    )
    start_args = list(
        project_mount_args(
            attempt_dir,
            include_host_system_mounts=use_host_runtime,
        )
    )
    if image_platform is not None:
        start_args = ["--platform", image_platform, *start_args]

    if use_host_runtime:
        return TaskContainerExecConfig(
            runtime=current_container_python_runtime(),
            pythonpath=_DEFAULT_RUNTIME_PYTHONPATH,
            start_extra_args=tuple(start_args),
            bootstrap=False,
            bootstrap_site_dir=None,
            image_platform=image_platform,
        )

    site_dir = task_container_runtime_dir(attempt_dir, "bootstrap") / "pydeps"
    return TaskContainerExecConfig(
        runtime=_CONTAINER_SYSTEM_PYTHON,
        pythonpath=f"{site_dir}:{_DEFAULT_RUNTIME_PYTHONPATH}",
        start_extra_args=tuple(start_args),
        bootstrap=True,
        bootstrap_site_dir=site_dir,
        image_platform=image_platform,
    )


def resolve_running_container_exec_config(
    *,
    container_id: str,
    exec_config: TaskContainerExecConfig,
    container_executable: str,
    cwd: str = "/testbed",
) -> TaskContainerExecConfig:
    if not exec_config.bootstrap:
        return exec_config

    probe_script = """
set -eu
for cand in "$@"; do
  if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
      "$cand" -c 'import sys; print(sys.executable)'
      exit 0
    fi
  fi
done
exit 1
"""
    result = subprocess.run(
        [
            container_executable,
            "exec",
            "-i",
            "-w",
            cwd,
            container_id,
            "/bin/sh",
            "-s",
            "--",
            *_CONTAINER_PYTHON_CANDIDATES,
        ],
        input=probe_script,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        details = _format_probe_failure_details(result)
        raise RuntimeError(
            "task-container python probe failed: "
            "no Python >=3.11 interpreter found in container"
            + (f" ({details})" if details else "")
        )
    runtime = result.stdout.strip()
    if not runtime:
        raise RuntimeError("task-container python probe failed: empty interpreter path")
    return TaskContainerExecConfig(
        runtime=runtime,
        pythonpath=exec_config.pythonpath,
        start_extra_args=exec_config.start_extra_args,
        bootstrap=exec_config.bootstrap,
        bootstrap_site_dir=exec_config.bootstrap_site_dir,
        image_platform=exec_config.image_platform,
    )


def write_task_container_request(
    *,
    attempt_dir: Path,
    scaffold: str,
    payload: dict[str, Any],
) -> Path:
    path = task_container_runtime_dir(attempt_dir, scaffold) / "request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_redact_request_payload(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _redact_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def _redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (_REDACTED_SECRET if key == "api_key" else _redact(child))
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [_redact(item) for item in value]
        return value

    return _redact(payload)


def exec_task_container_entrypoint(
    *,
    container_id: str,
    request_path: Path,
    request_payload: dict[str, Any] | None = None,
    runtime: str,
    pythonpath: str | None,
    timeout: float,
    container_executable: str,
    cwd: str = "/testbed",
) -> subprocess.CompletedProcess[str]:
    request = request_payload or json.loads(request_path.read_text(encoding="utf-8"))
    kind = str(request.get("kind") or "")
    if not kind:
        raise ValueError(f"missing request kind in {request_path}")
    mode = "preflight" if kind == "preflight" else "run"
    return subprocess.run(
        [
            container_executable,
            "exec",
            "-i",
            "-w",
            cwd,
            "-e",
            f"PYTHONPATH={pythonpath or _DEFAULT_RUNTIME_PYTHONPATH}",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            container_id,
            runtime,
            "-m",
            "trace_collect.runtime.entrypoint",
            "--mode",
            mode,
        ],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def read_task_container_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight_task_container_runtime(
    *,
    container_id: str,
    attempt_dir: Path,
    imports: list[str] | None = None,
    runtime: str | None = None,
    pythonpath: str | None = None,
    container_executable: str,
) -> TaskContainerPreflightProof:
    effective_runtime = runtime or current_container_python_runtime()
    runtime_dir = task_container_runtime_dir(attempt_dir, "preflight")
    import_list = imports or [
        "trace_collect.runtime.entrypoint",
        "agents.openclaw.eval.runner",
        "harness.trace_logger",
    ]
    request_payload = {
        "kind": "preflight",
        "result_path": str(runtime_dir / "result.json"),
        "imports": import_list,
        "writable_probe": str(runtime_dir / "writable.probe"),
        "container_id": container_id,
    }
    request_path = write_task_container_request(
        attempt_dir=attempt_dir,
        scaffold="preflight",
        payload=request_payload,
    )
    result = exec_task_container_entrypoint(
        container_id=container_id,
        request_path=request_path,
        request_payload=request_payload,
        runtime=effective_runtime,
        pythonpath=pythonpath,
        timeout=120,
        container_executable=container_executable,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "task-container preflight failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    payload = read_task_container_result(runtime_dir / "result.json")
    proof = payload.get("runtime_proof") or {}
    return TaskContainerPreflightProof(**proof)


def run_task_container_agent(
    *,
    container_id: str,
    request: dict[str, Any],
    timeout: float,
    runtime: str | None = None,
    pythonpath: str | None = None,
    container_executable: str,
) -> TaskContainerRunResult:
    effective_runtime = runtime or current_container_python_runtime()
    raw_stdout_path = Path(request["raw_stdout_path"])
    raw_stderr_path = Path(request["raw_stderr_path"])
    raw_stdout_path.parent.mkdir(parents=True, exist_ok=True)
    raw_stderr_path.parent.mkdir(parents=True, exist_ok=True)
    request_path = write_task_container_request(
        attempt_dir=Path(request["result_path"]).parents[2],
        scaffold=request["scaffold"],
        payload=request,
    )
    try:
        result = exec_task_container_entrypoint(
            container_id=container_id,
            request_path=request_path,
            request_payload=request,
            runtime=effective_runtime,
            pythonpath=pythonpath,
            timeout=timeout,
            container_executable=container_executable,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout_path.write_text(
            (exc.stdout or exc.output or "")
            if isinstance((exc.stdout or exc.output or ""), str)
            else ((exc.stdout or exc.output or b"").decode("utf-8", errors="replace")),
            encoding="utf-8",
        )
        raw_stderr_path.write_text(
            exc.stderr
            if isinstance(exc.stderr or "", str)
            else (exc.stderr or b"").decode("utf-8", errors="replace"),
            encoding="utf-8",
        )
        raise RuntimeError(f"task-container run timed out after {timeout}s") from exc
    except Exception as exc:
        if not raw_stdout_path.exists():
            raw_stdout_path.write_text("", encoding="utf-8")
        raw_stderr_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        raise
    if not raw_stdout_path.exists():
        raw_stdout_path.write_text(result.stdout, encoding="utf-8")
    if not raw_stderr_path.exists():
        raw_stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "task-container run failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    payload = read_task_container_result(Path(request["result_path"]))
    success = payload.get("success")
    if success is None:
        success = bool(payload.get("model_patch"))
    return TaskContainerRunResult(
        success=bool(success),
        exit_status=payload.get("exit_status"),
        model_patch=payload.get("model_patch", "") or "",
        error=payload.get("error"),
        n_iterations=payload.get("n_iterations"),
        total_llm_ms=payload.get("total_llm_ms"),
        total_tool_ms=payload.get("total_tool_ms"),
        total_tokens=payload.get("total_tokens"),
        runtime_proof=payload.get("runtime_proof") or {},
        trace_path=Path(payload.get("trace_path") or request.get("trace_file") or ""),
        raw_stdout_path=raw_stdout_path,
        raw_stderr_path=raw_stderr_path,
    )


def bootstrap_task_container_python(
    *,
    container_id: str,
    exec_config: TaskContainerExecConfig,
    extra_requirements: tuple[str, ...] = (),
    container_executable: str,
    cwd: str = "/testbed",
) -> None:
    if not exec_config.bootstrap or exec_config.bootstrap_site_dir is None:
        return

    marker = exec_config.bootstrap_site_dir / ".bootstrap-ready.json"
    requirements = tuple(
        dict.fromkeys(OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS + extra_requirements)
    )
    if _bootstrap_marker_matches(
        marker,
        requirements=requirements,
        runtime=exec_config.runtime,
    ):
        return
    if marker.exists():
        marker.unlink(missing_ok=True)
        shutil.rmtree(exec_config.bootstrap_site_dir, ignore_errors=True)

    userbase = exec_config.bootstrap_site_dir.parent / ".pyuserbase"
    userbase.mkdir(parents=True, exist_ok=True)
    get_pip = userbase / "get-pip.py"
    if not get_pip.exists():
        _download_get_pip(get_pip)

    pip_index_url = (
        os.environ.get("TASK_CONTAINER_PIP_INDEX_URL")
        or os.environ.get("PIP_INDEX_URL")
        or _DEFAULT_PIP_INDEX_URL
    )
    script = f"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

site_dir = pathlib.Path({str(exec_config.bootstrap_site_dir)!r})
marker = pathlib.Path({str(marker)!r})
userbase = pathlib.Path({str(userbase)!r})
requirements = {list(requirements)!r}
site_dir.mkdir(parents=True, exist_ok=True)
userbase.mkdir(parents=True, exist_ok=True)

if marker.exists():
    print("bootstrap runtime: reuse existing site-packages")
    raise SystemExit(0)

env = dict(os.environ)
env["PYTHONUSERBASE"] = str(userbase)
get_pip = userbase / "get-pip.py"
subprocess.check_call(
    [sys.executable, str(get_pip), "--user", "--break-system-packages"],
    env=env,
)
pip_bin = userbase / "bin" / "pip"
if not pip_bin.exists():
    pip_bin = userbase / "bin" / "pip3"
if not pip_bin.exists():
    raise RuntimeError("pip bootstrap succeeded but pip executable is missing")
subprocess.check_call(
    [
        str(pip_bin),
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--break-system-packages",
        "--target",
        str(site_dir),
        "-i",
        {pip_index_url!r},
        *requirements,
    ],
    env=env,
)
marker.write_text(
    json.dumps({{"requirements": requirements, "python": sys.executable}}),
    encoding="utf-8",
)
shutil.rmtree(userbase, ignore_errors=True)
"""
    result = subprocess.run(
        [
            container_executable,
            "exec",
            "-i",
            "-w",
            cwd,
            container_id,
            exec_config.runtime,
            "-",
        ],
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "task-container python bootstrap failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
