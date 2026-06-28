"""Host-side helpers for task-container agent parity."""

from __future__ import annotations

import fcntl
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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from agents.openclaw.runtime_deps import OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOTNAME = "_task_container_runtime"
_REDACTED_SECRET = "***REDACTED***"
_DEFAULT_RUNTIME_PYTHONPATH = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
_CONTAINER_SYSTEM_PYTHON = "/usr/bin/python3"
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple"
_SHARED_BOOTSTRAP_CACHE = Path.home() / ".cache" / "task-container-bootstrap"
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
    # Absolute paths for common Python installations in container images.
    # Ordered by likelihood: python:3.11-slim places python3 in /usr/local/bin,
    # Debian/Ubuntu in /usr/bin.  SWE-bench official images use conda at
    # /opt/miniconda3 (py311).  PATH-based fallbacks (python3, python) are
    # tried last so an explicit path is preferred when available.
    "/usr/local/bin/python3",   # python:3.11-slim
    "/usr/local/bin/python",    # symlink created by ensure_fixed_image()
    "/opt/miniconda3/bin/python3",  # SWE-bench official images (conda py311)
    "/opt/conda/bin/python3",       # alternative conda prefix
    "/usr/bin/python3",         # Debian/Ubuntu default
    "/usr/bin/python",
    "python3",                  # PATH-based fallback
    "python",                   # PATH-based fallback (last resort)
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
    arch: str | None = None,
    site_dir: Path | None = None,
) -> bool:
    """True when *marker* was written for the exact same *requirements* and *arch*.

    Architecture MUST be included in the marker because ``--only-binary=:all:``
    enforces pre-built wheels which are architecture-specific.  An ARM64 cache
    containing ``_tiktoken.cpython-311-aarch64-linux-gnu.so`` will fail with
    a circular-import error when reused for an x86_64 container (Python looks
    for ``_tiktoken.cpython-311-x86_64-linux-gnu.so``, doesn't find it, and
    falls back to a Python submodule import that hits the still-initialising
    ``tiktoken/__init__.py``).

    Python version is intentionally NOT included so that the same cache works
    across minor Python updates and symlink variations (e.g. ``/usr/local/bin/python``
    vs ``/usr/local/bin/python3`` on the same image).

    When *site_dir* is provided, also verifies that the actual installed
    packages match the manifest recorded in the marker.  This prevents stale
    cache contamination where extra packages from a previous run leak into
    a new container via the shared ``pydeps`` directory (mounted from the
    host home directory by ``start_task_container``).
    """
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected: dict[str, Any] = {"requirements": list(requirements)}
    if arch:
        expected["arch"] = arch
    # Also accept markers that have a "packages" manifest (new format).
    # The payload is considered matching as long as requirements + arch
    # coincide — the package-list check is done separately below.
    req_arch_match = all(
        payload.get(k) == v for k, v in expected.items()
    )
    if not req_arch_match:
        return False

    # Verify the installed package set hasn't been contaminated.
    # The marker may include a "packages" manifest (new format) or not
    # (legacy format).  When site_dir is provided, we cross-check the
    # actual .dist-info directories against the manifest.
    #
    # Legacy markers (without a "packages" key) are considered stale
    # because we cannot verify the cache hasn't been contaminated.
    if site_dir is not None and site_dir.exists():
        recorded = payload.get("packages")
        if recorded is None:
            print(
                f"[bootstrap] legacy marker (no package manifest), "
                f"forcing rebuild: {marker}",
                file=sys.stderr,
                flush=True,
            )
            return False
        recorded_set = set(recorded)
        if recorded_set:
            actual = _list_bootstrap_packages(site_dir)
            if actual != recorded_set:
                extra = actual - recorded_set
                missing = recorded_set - actual
                print(
                    f"[bootstrap] cache contamination detected in {site_dir}: "
                    + (f"extra={sorted(extra)} " if extra else "")
                    + (f"missing={sorted(missing)}" if missing else ""),
                    file=sys.stderr,
                    flush=True,
                )
                return False

    return True


def _list_bootstrap_packages(site_dir: Path) -> set[str]:
    """Return the set of installed package names in *site_dir*.

    Inspects ``.dist-info`` directories (PEP 376) to determine which
    packages are currently installed.  This is a lightweight check that
    does not require importing any packages or running pip.

    Keep in sync with the inline bootstrap script that writes the
    ``"packages"`` manifest to the marker JSON.
    """
    packages: set[str] = set()
    try:
        for entry in site_dir.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if name.endswith(".dist-info"):
                # PEP 376: <package_name>-<version>.dist-info
                # Strip the .dist-info suffix first, then rsplit at the
                # last dash to separate the version from the package name.
                # The .dist-info directory name already uses the PEP 503
                # normalised form (lowercase, underscores for non-alnum).
                stem = name[: -len(".dist-info")]
                pkg_name = stem.rsplit("-", 1)[0] if "-" in stem else stem
                packages.add(pkg_name)
    except OSError:
        return set()
    return packages


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
    """Return extra `podman run` args for container execution.

    By default, host system directories are NEVER mounted into the container.
    This guarantees identical tool resolution across host architectures
    (x86_64 vs ARM64).  Host system mounts were historically enabled in
    "parity mode" (same-arch) but produced host-dependent behaviour that
    broke cross-architecture trace replay.

    Set *include_host_system_mounts* to ``True`` only for interactive
    debugging when you need host tools inside the container.
    """

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
        include_host_system_mounts = False
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
    """Resolve the execution config for running code inside a task container.

    Always uses bootstrap mode (container's own Python) regardless of host
    architecture.  This guarantees that the same SWE-bench task image produces
    identical tool resolution on x86_64 and ARM64 hosts.

    The previous "parity mode" (host Python + host system mounts when
    image platform matched host platform) was removed because it introduced
    host-dependent behaviour that broke cross-architecture trace replay.
    """
    image_platform = _inspect_image_platform(
        image,
        container_executable=container_executable,
    )
    start_args = list(
        project_mount_args(
            attempt_dir,
            include_host_system_mounts=False,
        )
    )
    if image_platform is not None:
        start_args = ["--platform", image_platform, *start_args]

    site_dir = _SHARED_BOOTSTRAP_CACHE / "pydeps"
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

    print(
        f"[bootstrap] probing container {container_id[:12]} for Python >=3.11 ...",
        file=sys.stderr,
        flush=True,
    )
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
    print(
        f"[bootstrap] probing done: {runtime}",
        file=sys.stderr,
        flush=True,
    )
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
    cmd = [
        container_executable,
        "exec",
        "-i",
        "-w",
        cwd,
        "-e",
        f"PYTHONPATH={pythonpath or _DEFAULT_RUNTIME_PYTHONPATH}",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        container_id,
        runtime,
        "-m",
        "trace_collect.runtime.entrypoint",
        "--mode",
        mode,
    ]
    if mode == "preflight":
        return subprocess.run(
            cmd,
            input=json.dumps(request, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    # run mode: stream stdout to host terminal in real-time
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_lines: list[str] = []
    try:
        assert proc.stdout is not None
        assert proc.stdin is not None
        # Write stdin in a way that avoids deadlock
        stdin_data = json.dumps(request, ensure_ascii=False)
        proc.stdin.write(stdin_data)
        proc.stdin.close()
        for line in proc.stdout:
            print(line, end="", flush=True)
            stdout_lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    stderr_data = proc.stderr.read() if proc.stderr else ""
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="".join(stdout_lines),
        stderr=stderr_data,
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


@contextmanager
def _bootstrap_lock() -> Iterator[None]:
    """Acquire an exclusive cross-process lock on the shared bootstrap cache.

    Prevents concurrent :func:`bootstrap_task_container_python` calls (e.g.
    from simulate ``--workers``) from racing on ``get-pip.py`` and corrupting
    the shared ``.pyuserbase`` directory.
    """
    lock_dir = _SHARED_BOOTSTRAP_CACHE
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".bootstrap.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _bootstrap_arch(exec_config: TaskContainerExecConfig) -> str:
    """Extract the container architecture slug from *exec_config*.

    Returns e.g. ``"amd64"`` or ``"arm64"``.  When the image platform
    could not be inspected, falls back to the host architecture (which
    is correct for native-mode runs where the image arch always matches
    the host).
    """
    if exec_config.image_platform:
        # "linux/amd64" → "amd64"  (also normalises x86_64→amd64, aarch64→arm64)
        parts = exec_config.image_platform.split("/")
        if len(parts) == 2:
            return _normalize_arch(parts[1]) or parts[1]
    # platform.machine() always returns a non-None str; _normalize_arch
    # always returns a str for non-None input, so this never returns None.
    return _normalize_arch(platform.machine())  # type: ignore[return-value]


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

    arch = _bootstrap_arch(exec_config)
    marker = exec_config.bootstrap_site_dir / ".bootstrap-ready.json"
    requirements = tuple(
        dict.fromkeys(OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS + extra_requirements)
    )

    # Fast-path: check marker before acquiring the lock so workers that
    # arrive after the first one has finished skip the lock entirely.
    if _bootstrap_marker_matches(
        marker,
        requirements=requirements,
        arch=arch,
        site_dir=exec_config.bootstrap_site_dir,
    ):
        print(
            f"[bootstrap] shared cache hit ({arch}): {marker}",
            file=sys.stderr,
            flush=True,
        )
        return

    # Serialise access to the shared bootstrap cache so concurrent
    # workers don't race on get-pip.py / pip install.
    with _bootstrap_lock():
        # Double-check: another process may have finished while we waited.
        if _bootstrap_marker_matches(
            marker,
            requirements=requirements,
            arch=arch,
            site_dir=exec_config.bootstrap_site_dir,
        ):
            print(
                f"[bootstrap] shared cache hit ({arch}, after lock): {marker}",
                file=sys.stderr,
                flush=True,
            )
            return

        if marker.exists():
            print(
                f"[bootstrap] shared cache stale, rebuilding: {marker}",
                file=sys.stderr,
                flush=True,
            )
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
        print(
            f"[bootstrap] installing pip + {len(requirements)} runtime deps "
            f"from {pip_index_url} into container {container_id[:12]} ({arch})...",
            file=sys.stderr,
            flush=True,
        )
        script = f"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time as _time

def _log(msg):
    print(f"[bootstrap] {{msg}}", flush=True)

site_dir = pathlib.Path({str(exec_config.bootstrap_site_dir)!r})
marker = pathlib.Path({str(marker)!r})
userbase = pathlib.Path({str(userbase)!r})
requirements = {list(requirements)!r}
arch = {arch!r}
site_dir.mkdir(parents=True, exist_ok=True)
userbase.mkdir(parents=True, exist_ok=True)

if marker.exists():
    print("bootstrap runtime: reuse existing site-packages")
    raise SystemExit(0)

env = dict(os.environ)
env["PYTHONUSERBASE"] = str(userbase)
get_pip = userbase / "get-pip.py"
_log("step 1/3: bootstrapping pip via get-pip.py ...")
subprocess.check_call(
    [
        sys.executable,
        str(get_pip),
        "--user",
        "--break-system-packages",
        "--index-url",
        {pip_index_url!r},
    ],
    env=env,
)
pip_bin = userbase / "bin" / "pip"
if not pip_bin.exists():
    pip_bin = userbase / "bin" / "pip3"
if not pip_bin.exists():
    raise RuntimeError("pip bootstrap succeeded but pip executable is missing")
_log(f"step 2/3: pip install {{len(requirements)}} packages from {pip_index_url!r} ...")
_start = _time.time()
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
_elapsed = _time.time() - _start
_log(f"step 2/3: pip install done in {{_elapsed:.1f}}s")
_log("step 3/3: writing marker (with package manifest) ...")
# Collect the set of installed packages for cache-contamination detection.
# Keep in sync with _list_bootstrap_packages (host-side).
_installed: list[str] = []
for _entry in site_dir.iterdir():
    if _entry.is_dir() and _entry.name.endswith(".dist-info"):
        _stem = _entry.name[: -len(".dist-info")]
        _pkg = _stem.rsplit("-", 1)[0] if "-" in _stem else _stem
        _installed.append(_pkg)
marker.write_text(
    json.dumps({{
        "requirements": requirements,
        "arch": arch,
        "packages": sorted(_installed),
    }}),
    encoding="utf-8",
)
# Keep userbase intact so the agent can use pip at runtime.
# get-pip.py is no longer needed; pip binary + lib are kept.
_get_pip = userbase / "get-pip.py"
if _get_pip.exists():
    _get_pip.unlink()
_log("bootstrap complete")
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
            stdout=sys.stderr,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "task-container python bootstrap failed: "
                f"{result.stderr.strip()}"
            )
