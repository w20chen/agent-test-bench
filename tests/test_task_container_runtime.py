"""Tests for task-container runtime helpers."""

from __future__ import annotations

import json
import ssl
import urllib.error
from pathlib import Path

import pytest

from agents.openclaw.runtime_deps import OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS
from trace_collect.runtime.task_container import (
    TaskContainerExecConfig,
    TaskContainerRunResult,
    bootstrap_task_container_python,
    current_container_python_runtime,
    preflight_task_container_runtime,
    project_mount_args,
    resolve_task_container_exec_config,
    resolve_running_container_exec_config,
    run_task_container_agent,
)


def test_project_mount_args_include_attempt_dir_and_repo(
    tmp_path: Path,
) -> None:
    args = project_mount_args(tmp_path / "attempt")
    joined = " ".join(args)

    assert str((tmp_path / "attempt").resolve()) in joined
    assert str((Path(__file__).resolve().parents[1]).resolve()) in joined


def test_project_mount_args_skip_host_system_mounts_off_linux(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "trace_collect.runtime.task_container.platform.system",
        lambda: "Darwin",
    )

    joined = " ".join(project_mount_args(tmp_path / "attempt"))
    assert "/etc:/etc:ro" not in joined
    assert "/usr:/usr:ro" not in joined


def test_current_container_python_runtime_returns_sys_executable_under_conda_ml(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "ML")
    import sys

    assert current_container_python_runtime() == sys.executable


def test_current_container_python_runtime_requires_conda_ml(monkeypatch):
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    with pytest.raises(RuntimeError, match="conda env 'ML'"):
        current_container_python_runtime()


def test_resolve_task_container_exec_config_bootstraps_on_cross_platform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "trace_collect.runtime.task_container.platform.system",
        lambda: "Darwin",
    )
    monkeypatch.setattr(
        "trace_collect.runtime.task_container._inspect_image_platform",
        lambda image, *, container_executable: "linux/amd64",
    )

    config = resolve_task_container_exec_config(
        attempt_dir=tmp_path / "attempt",
        image="localhost/example:latest",
        container_executable="docker",
    )

    assert isinstance(config, TaskContainerExecConfig)
    assert config.bootstrap is True
    assert config.runtime == "/usr/bin/python3"
    assert config.image_platform == "linux/amd64"
    assert config.start_extra_args[:2] == ("--platform", "linux/amd64")
    assert all("/etc:/etc:ro" not in arg for arg in config.start_extra_args)
    assert config.bootstrap_site_dir is not None
    assert str(config.bootstrap_site_dir) in config.pythonpath


def test_resolve_running_container_exec_config_probes_python(monkeypatch) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath="/deps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=Path("/tmp/pydeps"),
        image_platform="linux/amd64",
    )

    def fake_run(*args, **kwargs):
        class Result:
            returncode = 0
            stdout = "/opt/conda/envs/ML/bin/python\n"
            stderr = ""

        return Result()

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    resolved = resolve_running_container_exec_config(
        container_id="cid-1",
        exec_config=exec_config,
        container_executable="docker",
    )

    assert resolved.runtime == "/opt/conda/envs/ML/bin/python"
    assert resolved.pythonpath == exec_config.pythonpath


def test_resolve_running_container_exec_config_raises_without_python(
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath="/deps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=Path("/tmp/pydeps"),
        image_platform="linux/amd64",
    )

    def fake_run(*args, **kwargs):
        class Result:
            returncode = 1
            stdout = "probe stdout\n"
            stderr = "probe stderr\n"

        return Result()

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    try:
        resolve_running_container_exec_config(
            container_id="cid-1",
            exec_config=exec_config,
            container_executable="docker",
        )
    except RuntimeError as exc:
        assert "no Python >=3.11 interpreter found" in str(exc)
        assert "stdout: probe stdout" in str(exc)
        assert "stderr: probe stderr" in str(exc)
    else:
        raise AssertionError("expected probe failure")


def test_bootstrap_task_container_python_uses_resolved_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/opt/conda/envs/ML/bin/python",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse()

    def fake_run(*args, **kwargs):
        seen["cmd"] = args[0]
        seen["input"] = kwargs["input"]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=("mcp>=1.0",),
        container_executable="docker",
    )

    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"
    assert "/opt/conda/envs/ML/bin/python" in seen["cmd"]
    input_script = str(seen["input"])
    for requirement in OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS:
        assert requirement in input_script
    assert "mcp>=1.0" in str(seen["input"])
    for heavy_dep in (
        "datasets",
        "terminal-bench",
        "transformers",
        "accelerate",
        "trafilatura",
    ):
        assert heavy_dep not in input_script


def test_bootstrap_task_container_python_retries_transient_get_pip_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    seen: dict[str, object] = {"attempts": 0, "sleeps": []}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["attempts"] = int(seen["attempts"]) + 1
        if int(seen["attempts"]) < 3:
            raise urllib.error.URLError(ssl.SSLEOFError("eof"))
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse()

    def fake_sleep(delay: float) -> None:
        sleeps = seen["sleeps"]
        assert isinstance(sleeps, list)
        sleeps.append(delay)

    def fake_run(*args, **kwargs):
        del args, kwargs

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.time.sleep", fake_sleep)
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=(),
        container_executable="docker",
    )

    assert seen["attempts"] == 3
    assert seen["sleeps"] == [1.0, 2.0]
    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"


def test_bootstrap_task_container_python_does_not_retry_http_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    seen = {"attempts": 0}

    def fake_urlopen(url: str, timeout: int):
        del url, timeout
        seen["attempts"] += 1
        raise urllib.error.HTTPError(
            "https://bootstrap.pypa.io/get-pip.py",
            404,
            "not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        "trace_collect.runtime.task_container.time.sleep", lambda *_: None
    )

    with pytest.raises(urllib.error.HTTPError):
        bootstrap_task_container_python(
            container_id="cid-1",
            exec_config=exec_config,
            extra_requirements=(),
            container_executable="docker",
        )

    assert seen["attempts"] == 1


def test_bootstrap_task_container_python_rebuilds_when_marker_requirements_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    exec_config.bootstrap_site_dir.mkdir(parents=True, exist_ok=True)
    marker = exec_config.bootstrap_site_dir / ".bootstrap-ready.json"
    marker.write_text(
        '{"requirements": ["openai>=2.0,<3.0"], "python": "/usr/bin/python3"}',
        encoding="utf-8",
    )
    stale_file = exec_config.bootstrap_site_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["url"] = url
        return FakeResponse()

    def fake_run(*args, **kwargs):
        seen["cmd"] = args[0]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=(),
        container_executable="docker",
    )

    assert not stale_file.exists()
    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"
    assert marker.exists() is False


def test_preflight_task_container_runtime_reads_runtime_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    attempt_dir = Path("relative-attempt")
    result_path = (
        tmp_path
        / "relative-attempt"
        / "_task_container_runtime"
        / "preflight"
        / "result.json"
    )

    def fake_exec(**kwargs):
        request = json.loads(Path(kwargs["request_path"]).read_text(encoding="utf-8"))
        seen.update(request)
        seen["runtime"] = kwargs["runtime"]
        seen["pythonpath"] = kwargs["pythonpath"]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "runtime_proof": {
                        "container_id": "cid-1",
                        "hostname": "host-a",
                        "cwd": "/testbed",
                        "python_executable": "/opt/conda/envs/ML/bin/python",
                        "python_prefix": "/opt/conda/envs/ML",
                        "project_root": "/repo",
                        "sys_path": ["/repo/src"],
                    },
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    proof = preflight_task_container_runtime(
        container_id="cid-1",
        attempt_dir=attempt_dir,
        imports=["trace_collect.runtime.entrypoint", "agents.openclaw.eval.runner"],
        runtime="/usr/bin/python3",
        pythonpath="/tmp/site:/repo/src:/repo",
        container_executable="docker",
    )

    assert proof.container_id == "cid-1"
    assert proof.python_executable == "/opt/conda/envs/ML/bin/python"
    assert Path(str(seen["result_path"])).is_absolute()
    assert Path(str(seen["writable_probe"])).is_absolute()
    assert Path(str(seen["result_path"])) == result_path
    assert seen["imports"] == [
        "trace_collect.runtime.entrypoint",
        "agents.openclaw.eval.runner",
    ]
    assert seen["runtime"] == "/usr/bin/python3"
    assert seen["pythonpath"] == "/tmp/site:/repo/src:/repo"


def test_run_task_container_agent_reads_result_and_writes_raw_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"

    def fake_exec(**kwargs):
        assert kwargs["runtime"] == "/usr/bin/python3"
        assert kwargs["pythonpath"] == "/tmp/site:/repo/src:/repo"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "diff --git a/x b/x",
                    "exit_status": "Submitted",
                    "error": None,
                    "n_iterations": 3,
                    "total_llm_ms": 1.0,
                    "total_tool_ms": 2.0,
                    "total_tokens": 4,
                    "runtime_proof": {"hostname": "container-a"},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = "stdout text"
            stderr = "stderr text"

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    result = run_task_container_agent(
        container_id="cid-2",
        timeout=10,
        runtime="/usr/bin/python3",
        pythonpath="/tmp/site:/repo/src:/repo",
        container_executable="docker",
        request={
            "scaffold": "openclaw",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert isinstance(result, TaskContainerRunResult)
    assert result.success is True
    assert stdout_path.read_text(encoding="utf-8") == "stdout text"
    assert stderr_path.read_text(encoding="utf-8") == "stderr text"


def test_run_task_container_agent_preserves_existing_raw_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("container stdout", encoding="utf-8")
    stderr_path.write_text("container stderr", encoding="utf-8")

    def fake_exec(**kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "diff --git a/x b/x",
                    "exit_status": "Submitted",
                    "error": None,
                    "n_iterations": 3,
                    "total_llm_ms": 1.0,
                    "total_tool_ms": 2.0,
                    "total_tokens": 4,
                    "runtime_proof": {"hostname": "container-a"},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    run_task_container_agent(
        container_id="cid-2",
        timeout=10,
        runtime="/usr/bin/python3",
        pythonpath="/tmp/site:/repo/src:/repo",
        container_executable="docker",
        request={
            "kind": "run_openclaw",
            "scaffold": "openclaw",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert stdout_path.read_text(encoding="utf-8") == "container stdout"
    assert stderr_path.read_text(encoding="utf-8") == "container stderr"


def test_run_task_container_agent_prefers_explicit_success_over_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "ML")
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"

    def fake_exec(**kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "",
                    "exit_status": "completed",
                    "error": None,
                    "runtime_proof": {},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    result = run_task_container_agent(
        container_id="cid-3",
        timeout=10,
        container_executable="docker",
        request={
            "kind": "run_openclaw",
            "scaffold": "openclaw",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert result.success is True


def test_run_task_container_agent_timeout_writes_partial_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "ML")
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"

    def fake_exec(**kwargs):
        raise __import__("subprocess").TimeoutExpired(
            cmd="podman exec ...",
            timeout=10,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    try:
        run_task_container_agent(
            container_id="cid-2",
            timeout=10,
            container_executable="docker",
            request={
                "kind": "run_openclaw",
                "scaffold": "openclaw",
                "result_path": str(result_path),
                "trace_file": str(trace_path),
                "raw_stdout_path": str(stdout_path),
                "raw_stderr_path": str(stderr_path),
            },
        )
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected timeout failure")

    assert stdout_path.read_text(encoding="utf-8") == "partial stdout"
    assert stderr_path.read_text(encoding="utf-8") == "partial stderr"


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_preflight_task_container_runtime_passes_container_executable_to_exec(
    tmp_path: Path,
    monkeypatch,
    container_executable: str,
) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "ML")
    seen: dict[str, object] = {}
    result_path = (
        tmp_path / "attempt" / "_task_container_runtime" / "preflight" / "result.json"
    )

    def fake_exec(**kwargs):
        seen["container_executable"] = kwargs["container_executable"]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "runtime_proof": {
                        "container_id": "cid-1",
                        "hostname": "host-a",
                        "cwd": "/testbed",
                        "python_executable": "/usr/bin/python3",
                        "python_prefix": "/usr",
                        "project_root": "/repo",
                        "sys_path": ["/repo/src"],
                    }
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    preflight_task_container_runtime(
        container_id="cid-1",
        attempt_dir=tmp_path / "attempt",
        container_executable=container_executable,
    )

    assert seen["container_executable"] == container_executable
