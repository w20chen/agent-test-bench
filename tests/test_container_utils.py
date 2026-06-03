"""Tests for harness.disk_preflight, container_image_prep, container_stats_sampler."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.container_image_prep import (  # noqa: E402
    clear_image_cache,
    drop_cached_fixed_image,
    ensure_fixed_image,
    ensure_source_image,
    normalize_image_reference,
    prune_dangling_images,
    remove_image,
)
from harness.container_stats_sampler import (  # noqa: E402
    ContainerStatsSampler,
    _parse_pipe_stats,
    summarize_samples,
)
from harness.disk_preflight import (  # noqa: E402
    DiskSpaceError,
    preflight_disk,
)


# ---------------------------------------------------------------------------
# disk_preflight
# ---------------------------------------------------------------------------


def test_preflight_disk_raises_on_shortfall(tmp_path: Path) -> None:
    with pytest.raises(DiskSpaceError) as exc:
        preflight_disk(tmp_path, min_free_gb=10**12)
    assert "GB required" in str(exc.value)


# ---------------------------------------------------------------------------
# container_image_prep
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_image_cache() -> None:
    clear_image_cache()
    yield
    clear_image_cache()


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_fixed_image_builds_when_derivative_missing(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == [container_executable, "image", "inspect"] and "--format" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="amd64 linux\n",
                stderr="",
            )
        # existence probe → missing (returncode=1)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        # run -d → container id
        if cmd[1:3] == ["run", "-d"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid_xyz\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        fixed, elapsed = ensure_fixed_image(
            "swerebench/foo:latest",
            container_executable=container_executable,
            host_uid=1000,
            host_gid=1000,
        )
    assert fixed == "swebench-fixed-docker.io_swerebench_foo_latest"
    assert elapsed >= 0.0
    # Expect: fixed exists (miss), source exists (miss), pull, run -d, exec chown, commit, stop, rm
    verbs = [" ".join(c[1:3]) for c in calls]
    if container_executable == "podman":
        assert verbs.count("image exists") == 2
        assert verbs.count("image inspect") == 1
    else:
        assert verbs.count("image inspect") == 3
    assert "pull docker.io/swerebench/foo:latest" in [" ".join(c[1:4]) for c in calls]
    assert any("run -d" in v for v in verbs)
    run_cmd = next(cmd for cmd in calls if cmd[1:3] == ["run", "-d"])
    assert run_cmd[:5] == [
        container_executable,
        "run",
        "-d",
        "--platform",
        "linux/amd64",
    ]
    assert any("exec cid_xyz" == " ".join(c[1:3]) for c in calls)
    assert any("commit cid_xyz" == " ".join(c[1:3]) for c in calls)


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_fixed_image_raises_on_build_failure(
    container_executable: str,
) -> None:
    def boom(cmd, **kwargs):
        if cmd[:3] == [container_executable, "image", "inspect"] and "--format" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="amd64 linux\n",
                stderr="",
            )
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd[1:3] == ["run", "-d"]:
            raise subprocess.CalledProcessError(1, cmd, stderr="kaboom")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=boom,
    ):
        with pytest.raises(RuntimeError, match="Failed to build"):
            ensure_fixed_image(
                "swerebench/img",
                container_executable=container_executable,
            )


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_source_image_pulls_when_missing(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        ensure_source_image(
            "swerebench/source:latest",
            container_executable=container_executable,
        )

    assert calls[0] == (
        [container_executable, "image", "exists", "docker.io/swerebench/source:latest"]
        if container_executable == "podman"
        else [
            container_executable,
            "image",
            "inspect",
            "docker.io/swerebench/source:latest",
        ]
    )
    assert calls[1] == [
        container_executable,
        "pull",
        "docker.io/swerebench/source:latest",
    ]


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_source_image_retries_transient_pull_failure(
    container_executable: str,
) -> None:
    calls = []
    pull_attempts = 0

    def fake_run(cmd, **kwargs):
        nonlocal pull_attempts
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd[:2] == [container_executable, "pull"]:
            pull_attempts += 1
            if pull_attempts < 3:
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    stdout="",
                    stderr="failed to do request: EOF",
                )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch(
            "harness.container_image_prep.subprocess.run",
            side_effect=fake_run,
        ),
        patch("harness.container_image_prep.time.sleep", lambda *_: None),
    ):
        ensure_source_image(
            "swerebench/source:latest",
            container_executable=container_executable,
        )

    assert pull_attempts == 3


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_ensure_source_image_does_not_retry_nonretryable_pull_failure(
    container_executable: str,
) -> None:
    pull_attempts = 0

    def fake_run(cmd, **kwargs):
        nonlocal pull_attempts
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd[:2] == [container_executable, "pull"]:
            pull_attempts += 1
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="manifest unknown",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch(
            "harness.container_image_prep.subprocess.run",
            side_effect=fake_run,
        ),
        patch("harness.container_image_prep.time.sleep", lambda *_: None),
    ):
        with pytest.raises(RuntimeError, match="manifest unknown"):
            ensure_source_image(
                "swerebench/source:latest",
                container_executable=container_executable,
            )

    assert pull_attempts == 1


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_drop_cached_fixed_image_forces_reprobe(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        ensure_fixed_image(
            "swerebench/cached:latest",
            container_executable=container_executable,
        )
        drop_cached_fixed_image("swerebench/cached:latest")
        ensure_fixed_image(
            "swerebench/cached:latest",
            container_executable=container_executable,
        )

    exists_calls = [
        cmd
        for cmd in calls
        if (
            container_executable == "podman"
            and cmd[:3] == ["podman", "image", "exists"]
        )
        or (
            container_executable == "docker"
            and cmd[:3] == ["docker", "image", "inspect"]
        )
    ]
    assert len(exists_calls) == 2


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_remove_image_and_prune_dangling_images(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        assert (
            remove_image(
                "docker.io/swerebench/source:latest",
                container_executable=container_executable,
            )
            is True
        )
        prune_dangling_images(container_executable=container_executable)

    assert [
        container_executable,
        "image",
        "rm",
        "-f",
        "docker.io/swerebench/source:latest",
    ] in calls
    assert [container_executable, "image", "prune", "-f"] in calls


def test_normalize_image_reference_qualifies_short_names() -> None:
    assert normalize_image_reference("hello-world") == "docker.io/library/hello-world"
    assert normalize_image_reference("alpine:3.20") == "docker.io/library/alpine:3.20"
    assert (
        normalize_image_reference("alpine@sha256:deadbeef")
        == "docker.io/library/alpine@sha256:deadbeef"
    )
    assert (
        normalize_image_reference("swerebench/foo:latest")
        == "docker.io/swerebench/foo:latest"
    )
    assert (
        normalize_image_reference("docker.io/swerebench/foo:latest")
        == "docker.io/swerebench/foo:latest"
    )


def test_normalize_image_reference_respects_registry_prefix_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TASK_CONTAINER_IMAGE_REGISTRY_PREFIX", "docker.1ms.run")

    assert (
        normalize_image_reference("hello-world") == "docker.1ms.run/library/hello-world"
    )
    assert (
        normalize_image_reference("swerebench/foo:latest")
        == "docker.1ms.run/swerebench/foo:latest"
    )
    assert (
        normalize_image_reference("docker.io/swerebench/foo:latest")
        == "docker.io/swerebench/foo:latest"
    )


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_remove_image_keeps_local_fixed_tags_unqualified(
    container_executable: str,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if container_executable == "podman" and cmd[:3] == [
            "podman",
            "image",
            "exists",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if container_executable == "docker" and cmd[:3] == [
            "docker",
            "image",
            "inspect",
        ]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        assert (
            remove_image(
                "swebench-fixed-local_tag",
                container_executable=container_executable,
            )
            is True
        )

    assert calls[0] == (
        [container_executable, "image", "exists", "swebench-fixed-local_tag"]
        if container_executable == "podman"
        else [container_executable, "image", "inspect", "swebench-fixed-local_tag"]
    )
    assert calls[1] == [
        container_executable,
        "image",
        "rm",
        "-f",
        "swebench-fixed-local_tag",
    ]


# ---------------------------------------------------------------------------
# container_stats_sampler
# ---------------------------------------------------------------------------


def test_parse_pipe_stats_parses_pipe_format() -> None:
    pipe = _parse_pipe_stats("1MB / 1GB|0.1%|0.5%")
    assert pipe is not None and pipe["mem_usage"] == "1MB / 1GB"


def test_stats_sampler_collects_samples_and_stops_cleanly() -> None:
    # Emit the new pipe format.
    pipe_output = "1MB / 1GB|0.1%|0.5%"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=pipe_output, stderr="")

    sampler = ContainerStatsSampler(
        container_id="abc",
        interval_s=0.05,
    )
    with patch(
        "harness.container_stats_sampler.subprocess.run",
        side_effect=fake_run,
    ):
        sampler.start()
        time.sleep(0.2)
        samples = sampler.stop()
    assert len(samples) >= 2
    assert samples[0]["mem_usage"] == "1MB / 1GB"


def test_summarize_samples_computes_min_max_avg() -> None:
    samples = [
        {
            "epoch": 1000.0,
            "mem_usage": "100MB / 1GB",
            "mem_percent": "10%",
            "cpu_percent": "5%",
        },
        {
            "epoch": 1002.0,
            "mem_usage": "200MB / 1GB",
            "mem_percent": "20%",
            "cpu_percent": "15%",
        },
        {
            "epoch": 1004.0,
            "mem_usage": "300MB / 1GB",
            "mem_percent": "30%",
            "cpu_percent": "25%",
        },
    ]
    summary = summarize_samples(samples)
    assert summary["sample_count"] == 3
    assert summary["duration_seconds"] == 4.0
    assert summary["memory_mb"]["min"] == 100.0
    assert summary["memory_mb"]["max"] == 300.0
    assert summary["memory_mb"]["avg"] == 200.0
    assert summary["cpu_percent"]["min"] == 5.0
    assert summary["cpu_percent"]["max"] == 25.0
    assert summary["cpu_percent"]["avg"] == 15.0
