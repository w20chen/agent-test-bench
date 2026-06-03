"""Helpers for preparing writable derivative container images.

The derivative keeps the upstream image content while making ``/testbed``
writeable for agent runs.
"""

from __future__ import annotations

import os
import subprocess
import time

from harness.container_runtime import image_exists_command

_IMAGE_CACHE: dict[str, tuple[str, float]] = {}
_PULL_ATTEMPTS = 3
_PULL_BACKOFF_SECONDS = 1.0
_ARCH_ALIASES = {
    "amd64": "amd64",
    "x86_64": "amd64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


def _image_slug(source_image: str) -> str:
    return source_image.replace("/", "_").replace(":", "_").replace("@", "_")


def normalize_image_reference(image: str) -> str:
    """Return a fully qualified image reference when possible."""
    if not image:
        return ""
    registry_prefix = os.environ.get("TASK_CONTAINER_IMAGE_REGISTRY_PREFIX", "").strip()
    if registry_prefix:
        registry_prefix = registry_prefix.rstrip("/")
    if "/" not in image:
        if registry_prefix:
            return f"{registry_prefix}/library/{image}"
        return f"docker.io/library/{image}"
    head = image.split("/", 1)[0]
    if "." in head or ":" in head or head == "localhost":
        return image
    if registry_prefix:
        return f"{registry_prefix}/{image}"
    return f"docker.io/{image}"


def fixed_image_name_for(source_image: str) -> str:
    return f"swebench-fixed-{_image_slug(source_image)}"


def _image_exists(image: str, executable: str) -> bool:
    result = subprocess.run(
        image_exists_command(image, container_executable=executable),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _run(
    cmd: list[str], *, check: bool = True, timeout: int = 180
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _normalize_arch(raw: str | None) -> str | None:
    if raw is None:
        return None
    return _ARCH_ALIASES.get(raw.lower(), raw.lower())


def _inspect_image_platform(image: str, executable: str) -> str | None:
    result = _run(
        [
            executable,
            "image",
            "inspect",
            image,
            "--format",
            "{{.Architecture}} {{.Os}}",
        ],
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
    if norm_arch is None:
        return None
    return f"{os_name.lower()}/{norm_arch}"


def _is_retryable_pull_failure(text: str) -> bool:
    lowered = text.lower()
    retryable_markers = (
        "eof",
        "unexpected eof",
        "connection reset",
        "i/o timeout",
        "tls handshake timeout",
        "context deadline exceeded",
        "temporarily unavailable",
    )
    return any(marker in lowered for marker in retryable_markers)


def _pull_source_image(image: str, executable: str) -> None:
    last_error: str | None = None
    for attempt in range(1, _PULL_ATTEMPTS + 1):
        result = _run(
            [executable, "pull", image],
            check=False,
            timeout=3600,
        )
        if result.returncode == 0:
            return
        output = (result.stderr or result.stdout or "").strip()
        last_error = output or f"exit code {result.returncode}"
        if attempt >= _PULL_ATTEMPTS or not _is_retryable_pull_failure(last_error):
            raise RuntimeError(f"Failed to pull source image {image}: {last_error}")
        time.sleep(_PULL_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    if last_error is not None:
        raise RuntimeError(f"Failed to pull source image {image}: {last_error}")


def ensure_source_image(
    source_image: str,
    *,
    container_executable: str,
) -> None:
    """Ensure ``source_image`` exists locally, pulling when missing."""
    source_image = normalize_image_reference(source_image)
    if not source_image:
        return
    if _image_exists(source_image, container_executable):
        return
    _pull_source_image(source_image, container_executable)


def remove_image(
    image: str,
    *,
    container_executable: str,
    normalize: bool = False,
) -> bool:
    """Best-effort local image removal.

    Returns ``True`` when an image existed and was removed, ``False`` when the
    image was already absent.
    """
    if normalize:
        image = normalize_image_reference(image)
    if not image or not _image_exists(image, container_executable):
        return False
    result = _run(
        [container_executable, "image", "rm", "-f", image],
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to remove image {image}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return True


def prune_dangling_images(*, container_executable: str) -> None:
    """Best-effort prune of dangling image layers."""
    result = _run(
        [container_executable, "image", "prune", "-f"],
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to prune dangling images: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _build_fixed_image(
    source_image: str,
    fixed_name: str,
    executable: str,
    uid: int,
    gid: int,
    image_platform: str | None,
) -> None:
    """Commit a writable derivative with /testbed chowned to ``uid:gid``.

    Implementation mirrors agentcgroup/scripts/run_swebench.py::_fix_permissions.
    """
    run_cmd = [executable, "run", "-d"]
    if image_platform:
        run_cmd.extend(["--platform", image_platform])
    run_cmd.extend([source_image, "sleep", "120"])
    start = _run(run_cmd, check=True, timeout=180)
    container_id = start.stdout.strip()
    try:
        _run(
            [
                executable,
                "exec",
                container_id,
                "chown",
                "-R",
                f"{uid}:{gid}",
                "/testbed",
            ],
            check=True,
            timeout=120,
        )
        _run(
            [executable, "commit", container_id, fixed_name],
            check=True,
            timeout=240,
        )
    finally:
        _run([executable, "stop", container_id], check=False, timeout=30)
        _run([executable, "rm", "-f", container_id], check=False, timeout=30)


def ensure_fixed_image(
    source_image: str,
    *,
    container_executable: str,
    host_uid: int | None = None,
    host_gid: int | None = None,
) -> tuple[str, float]:
    """Return ``(fixed_image_name, elapsed_seconds)``.

    Reuses a cached derivative image for repeat calls within the same process
    and also skips the build step when the derivative already exists on the
    host (checked via the runtime-specific image existence probe). The CC
    reference always
    builds the fixed image once per source — there is no probe-first path,
    because the container always needs the ``/testbed`` ownership fix.
    """
    source_image = normalize_image_reference(source_image)
    if source_image in _IMAGE_CACHE:
        return _IMAGE_CACHE[source_image]

    fixed_name = fixed_image_name_for(source_image)

    if _image_exists(fixed_name, container_executable):
        _IMAGE_CACHE[source_image] = (fixed_name, 0.0)
        return _IMAGE_CACHE[source_image]

    ensure_source_image(
        source_image,
        container_executable=container_executable,
    )
    image_platform = _inspect_image_platform(source_image, container_executable)

    uid = host_uid if host_uid is not None else os.getuid()
    gid = host_gid if host_gid is not None else os.getgid()

    t0 = time.time()
    try:
        _build_fixed_image(
            source_image,
            fixed_name,
            container_executable,
            uid,
            gid,
            image_platform,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # Fall back to the source image rather than leave the caller hanging.
        _IMAGE_CACHE[source_image] = (source_image, 0.0)
        raise RuntimeError(
            f"Failed to build fixed derivative image for {source_image}: {exc}"
        ) from exc
    elapsed = time.time() - t0
    _IMAGE_CACHE[source_image] = (fixed_name, elapsed)
    return _IMAGE_CACHE[source_image]


def clear_image_cache() -> None:
    _IMAGE_CACHE.clear()


def drop_cached_fixed_image(source_image: str) -> None:
    """Forget any cached fixed-image lookup for ``source_image``."""
    _IMAGE_CACHE.pop(normalize_image_reference(source_image), None)
