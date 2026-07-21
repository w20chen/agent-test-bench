"""Trace collection entrypoints for SWE-style benchmarks."""

from __future__ import annotations

from contextlib import ExitStack
from concurrent.futures import Future, ThreadPoolExecutor
import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_call import UnifiedProvider
from agents.openclaw.runtime_deps import OPENCLAW_MCP_RUNTIME_REQUIREMENTS

# `serving.recording` (and its KV-eviction transitive deps) pulls in `torch`
# via `transformers.cache_utils`. Container venvs that only need MCP/scaffold
# helpers from this module must not pay that import cost — keep this lazy
# inside the `record_internals` branch in `collect_traces`.
from harness.container_image_prep import (
    drop_cached_fixed_image,
    ensure_source_image,
    is_arm_host,
    normalize_image_reference,
    prune_dangling_images,
    remove_image,
    resolve_arm_base_image,
    use_arm_qemu,
)
from harness.ksys import KsysSession
from trace_collect.attempt_pipeline import (
    AttemptContext,
    AttemptResult,
    run_attempt,
    start_task_container,
    stop_task_container,
)
from trace_collect.monitoring import (
    MonitoringPolicy,
    resolve_collect_monitoring,
)
from trace_collect.package_runtime_prediction import (
    merge_pip_predictions_into_shared_history,
    seed_pip_history_from_shared,
    seed_pip_family_history_from_shared,
)
from trace_collect.pytest_runtime_prediction import (
    merge_pytest_predictions_into_shared_history,
    seed_pytest_history_from_shared,
    seed_pytest_family_history_from_shared,
)
from trace_collect.python_script_runtime_prediction import (
    merge_python_script_predictions_into_shared_history,
    seed_python_script_history_from_shared,
    seed_python_script_family_history_from_shared,
)
from trace_collect.runtime.task_container import (
    bootstrap_task_container_python,
    preflight_task_container_runtime,
    resolve_task_container_exec_config,
    resolve_running_container_exec_config,
    run_task_container_agent,
)

if TYPE_CHECKING:
    from agents.benchmarks.base import Benchmark
    from serving.kv_policies.base import EvictionPolicyConfig
    from serving.sparse_attention.base import SparseAttentionConfig

logger = logging.getLogger(__name__)
_DOCKER_HOST_GATEWAY = "172.17.0.1"


def _safe_scope_dir_name(value: str, *, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", value).strip("._-")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80] or fallback}-{digest}"


def _history_bucket_dir(run_dir: Path, tool: str, scope: str, value: str) -> Path:
    return (
        run_dir.resolve()
        / f"{tool}_runtime_db"
        / scope
        / _safe_scope_dir_name(value, fallback=scope)
    )


def _pip_history_scope_dir(run_dir: Path, instance_id: str) -> Path:
    return run_dir.resolve() / "pip_runtime_db" / _safe_scope_dir_name(
        instance_id,
        fallback="instance",
    )


def _python_script_history_scope_dir(run_dir: Path, instance_id: str) -> Path:
    return run_dir.resolve() / "python_script_runtime_db" / _safe_scope_dir_name(
        instance_id,
        fallback="instance",
    )


def _pytest_history_scope_dir(run_dir: Path, instance_id: str) -> Path:
    return run_dir.resolve() / "pytest_runtime_db" / _safe_scope_dir_name(
        instance_id,
        fallback="instance",
    )


def _task_family_scope(task: dict[str, Any], instance_id: str) -> str:
    """Build a coarse family key for cross-instance history sharing.

    Family scope is intentionally minimal: **only repo**.  Image family,
    Python version, and install config are deliberately excluded so that
    all instances of the same repository share a single history bucket.
    This maximises the chance of a cold-start hit.
    """
    repo = task.get("repo")
    if isinstance(repo, str) and repo:
        return f"repo={repo}"
    return f"instance={instance_id}"


def _family_history_scope_dir(
    run_dir: Path,
    tool: str,
    task: dict[str, Any],
    instance_id: str,
) -> Path:
    return _history_bucket_dir(
        run_dir,
        tool,
        "family",
        _task_family_scope(task, instance_id),
    )


def _recording_server_public_host(
    *,
    execution_environment: str,
    runtime_mode: str,
    container_executable: str | None,
) -> str | None:
    explicit = os.environ.get("HF_RECORDING_PUBLIC_HOST")
    if explicit:
        return explicit
    if container_executable == "docker" and (
        execution_environment == "container" or runtime_mode == "task_container_agent"
    ):
        return _DOCKER_HOST_GATEWAY
    return None


_ATTEMPT_DIR_NAME_RE = re.compile(r"^attempt_(\d+)$")


def _mcp_config_label(mcp_config: str | None) -> str | None:
    """Map ``--mcp-config`` to the value stored in trace metadata."""
    if mcp_config is None:
        return None
    if mcp_config == "none":
        return "none"
    return Path(mcp_config).name


def load_mcp_servers(mcp_config: str | None) -> dict:
    """Parse a ``--mcp-config`` argument into ``dict[str, MCPServerConfig]``."""
    if mcp_config is None or mcp_config == "none":
        return {}

    import yaml

    path = Path(mcp_config)
    if not path.exists():
        raise FileNotFoundError(f"--mcp-config path does not exist: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"--mcp-config root must be a mapping, got {type(raw).__name__}"
        )

    from agents.openclaw.config.schema import MCPServerConfig

    servers: dict[str, Any] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"--mcp-config entry {name!r} must be a mapping, "
                f"got {type(entry).__name__}"
            )
        servers[name] = MCPServerConfig.model_validate(entry)
    return servers


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _generation_config(
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }.items()
        if value is not None
    }


@dataclass(slots=True)
class CollectedTaskResult:
    """Per-task summary emitted alongside attempt artifacts."""

    instance_id: str
    attempt_dir: Path
    success: bool
    model_patch: str = ""
    exit_status: str | None = None
    error: str | None = None
    elapsed_s: float | None = None
    n_iterations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "attempt_dir": str(self.attempt_dir),
            "trace_file": str(self.attempt_dir / "trace.jsonl"),
            "success": self.success,
            "model_patch": self.model_patch,
            "exit_status": self.exit_status,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
            "n_iterations": self.n_iterations,
        }


def build_run_dir(benchmark: "Benchmark", model: str) -> Path:
    """Build run directory from benchmark plugin config: ``trace_root/model/ts/``."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_model = model.replace("/", "-").replace(":", "-")
    return benchmark.config.trace_root / safe_model / ts


def load_completed_ids(run_dir: Path) -> set[str]:
    """Return instance_ids whose ``attempt_*/run_manifest.json`` is ``completed`` or ``exhausted``.

    Only the nested attempt layout is supported — no legacy flat scan.

    ``exhausted`` (max_iterations reached) is treated as done because
    re-running a task that already consumed its full iteration budget
    would waste compute without changing the outcome.
    """
    completed: set[str] = set()
    if not run_dir.exists():
        return completed
    for instance_dir in run_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        for attempt_dir in sorted(instance_dir.glob("attempt_*")):
            manifest_path = attempt_dir / "run_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("status") in ("completed", "exhausted"):
                completed.add(instance_dir.name)
                break
    return completed


def next_attempt_number(run_dir: Path, instance_id: str) -> int:
    """Return the next attempt index for ``run_dir/instance_id``."""
    instance_dir = run_dir / instance_id
    if not instance_dir.exists():
        return 1

    max_attempt = 0
    for child in instance_dir.iterdir():
        if not child.is_dir():
            continue
        match = _ATTEMPT_DIR_NAME_RE.fullmatch(child.name)
        if match is None:
            continue
        max_attempt = max(max_attempt, int(match.group(1)))
    return max_attempt + 1


def write_results_jsonl(results: list[CollectedTaskResult], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def _select_tasks(
    tasks: list[dict[str, Any]],
    *,
    instance_ids: list[str] | None,
    sample: int | None,
    skip: int | None = None,
) -> list[dict[str, Any]]:
    """Filter tasks while preserving the explicit ``instance_ids`` order.

    *instance_ids* selects specific tasks by ID.
    *skip* drops the first N tasks (applied before *sample*).
    *sample* keeps only the first N remaining tasks.
    """
    selected = list(tasks)
    if instance_ids is not None:
        by_id = {task["instance_id"]: task for task in tasks}
        missing = [
            instance_id for instance_id in instance_ids if instance_id not in by_id
        ]
        if missing:
            raise ValueError(f"No tasks matched instance_ids: {missing}")
        selected = [by_id[instance_id] for instance_id in instance_ids]
    if skip is not None:
        selected = selected[skip:]
    if sample is not None:
        selected = selected[:sample]
    return selected


def _task_source_image(benchmark: "Benchmark", task: dict[str, Any]) -> str | None:
    """Return the source image for *task*.

    On ARM hosts this returns the shared ARM base image (the per-task
    x86_64 images do not exist for arm64), but only for benchmarks that
    actually need container images (``image_name_for`` returns non-None).
    The actual repo checkout and install steps are handled by
    ``ensure_arm_fixed_image``.
    """
    image_name = benchmark.image_name_for(task)
    if not image_name:
        return None
    # ARM-native mode: use the shared ARM base image (x86_64 task images
    # don't exist for arm64).  QEMU mode: use the original x86_64 image
    # and let Docker binfmt handle the emulation.
    if is_arm_host() and not use_arm_qemu():
        return resolve_arm_base_image(benchmark.config)
    return normalize_image_reference(str(image_name))


def _log_image_status(
    tasks: list[dict[str, Any]],
    benchmark: "Benchmark",
    container_executable: str,
) -> None:
    """Log which task images are already pulled (✅) vs missing (⬇)."""
    import subprocess as _sp

    unique_images: dict[str, str] = {}
    for t in tasks:
        img = _task_source_image(benchmark, t)
        if img:
            unique_images.setdefault(img, t["instance_id"])

    ready: list[str] = []
    missing: list[str] = []
    for img, instance_id in unique_images.items():
        if not img:
            continue
        result = _sp.run(
            [container_executable, "image", "inspect", img],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            ready.append(instance_id)
        else:
            missing.append(instance_id)

    if ready:
        logger.info("Images ready (%d): %s", len(ready), ", ".join(ready))
    if missing:
        logger.info("Images need pull (%d): %s", len(missing), ", ".join(missing))
    if not ready and not missing:
        logger.info("Images: no container images needed")


def _next_pending_source_image(
    benchmark: "Benchmark",
    tasks: list[dict[str, Any]],
    *,
    current_index: int,
    completed: set[str],
    rerun_completed: bool = False,
) -> str | None:
    """Return the next incomplete task's source image, if any."""
    for next_task in tasks[current_index + 1 :]:
        if next_task["instance_id"] in completed and not rerun_completed:
            continue
        source_image = _task_source_image(benchmark, next_task)
        if source_image:
            return source_image
    return None


def _ensure_task_source_ready(
    *,
    instance_id: str,
    source_image: str | None,
    prefetched_source_image: str | None,
    prefetch_future: Future[None] | None,
    container_executable: str | None,
) -> None:
    """Ensure the current task's source image is available locally."""
    if not source_image:
        return
    if container_executable is None:
        raise ValueError("container_executable is required when source_image is set")
    if prefetch_future is not None and prefetched_source_image == source_image:
        try:
            prefetch_future.result()
            logger.info("prefetch ready for %s image=%s", instance_id, source_image)
            return
        except Exception as exc:
            logger.warning(
                "prefetch failed for %s image=%s; retrying foreground: %s",
                instance_id,
                source_image,
                exc,
            )
    ensure_source_image(
        source_image,
        container_executable=container_executable,
    )


def _cleanup_task_images(
    *,
    instance_id: str,
    source_image: str | None,
    fixed_image: str | None,
    keep_source_image: str | None,
    container_executable: str | None,
    run_dir: Path | None = None,
    remove_containers: bool = False,
    force_cleanup: bool = False,
    container_labels: dict[str, str] | None = None,
) -> None:
    """Best-effort cleanup that keeps only the current/next-image budget.

    On ARM hosts the base image is shared across all tasks and is never
    removed.  Only per-task fixed derivative images are cleaned up.

    Set env ``KEEP_IMAGES_ABOVE_GB`` (e.g. ``100``) to skip image removal
    when free disk exceeds the threshold.  Default: always clean up.
    """
    if not source_image and not fixed_image:
        return
    if container_executable is None:
        raise ValueError("container_executable is required for image cleanup")

    # Never remove the shared ARM base image (native mode only).
    # In QEMU mode per-task x86_64 images are cleaned up normally.
    if is_arm_host() and not use_arm_qemu():
        source_image = None

    keep_gb_str = os.environ.get("KEEP_IMAGES_ABOVE_GB", "")
    if keep_gb_str and run_dir is not None and not force_cleanup:
        try:
            keep_gb = float(keep_gb_str)
            free_gb = shutil.disk_usage(run_dir).free / (1024**3)
            if free_gb > keep_gb:
                logger.info(
                    "cleanup %s skipped: %.1f GB free > %.1f GB threshold",
                    instance_id,
                    free_gb,
                    keep_gb,
                )
                return
        except (ValueError, OSError):
            pass

    removed_any = False
    try:
        if fixed_image and fixed_image != source_image:
            if remove_containers:
                _remove_containers_for_attempt(
                    instance_id=instance_id,
                    image=fixed_image,
                    container_executable=container_executable,
                    labels=container_labels,
                )
            removed_fixed = remove_image(
                fixed_image,
                container_executable=container_executable,
            )
            removed_any = removed_fixed or removed_any
            if removed_fixed:
                logger.info(
                    "cleanup %s removed fixed image %s",
                    instance_id,
                    fixed_image,
                )
    except Exception as exc:
        logger.warning(
            "cleanup %s failed removing fixed image %s: %s",
            instance_id,
            fixed_image,
            exc,
        )

    try:
        if source_image and source_image != keep_source_image:
            removed_source = remove_image(
                source_image,
                container_executable=container_executable,
            )
            removed_any = removed_source or removed_any
            if removed_source:
                logger.info(
                    "cleanup %s removed source image %s",
                    instance_id,
                    source_image,
                )
    except Exception as exc:
        logger.warning(
            "cleanup %s failed removing source image %s: %s",
            instance_id,
            source_image,
            exc,
        )

    if source_image:
        drop_cached_fixed_image(source_image)

    if not removed_any:
        return
    try:
        prune_dangling_images(
            container_executable=container_executable,
        )
    except Exception as exc:
        logger.warning("cleanup %s prune failed: %s", instance_id, exc)


def _container_ids_for_attempt(
    *,
    image: str,
    container_executable: str,
    labels: dict[str, str],
) -> list[str]:
    filters = ["ancestor=" + image]
    filters.extend(f"label={key}={value}" for key, value in sorted(labels.items()))
    cmd = [container_executable, "ps", "-a"]
    for filter_expr in filters:
        cmd.extend(["--filter", filter_expr])
    cmd.extend(["--format", "{{.ID}}"])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to list containers for image {image}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _remove_containers_for_attempt(
    *,
    instance_id: str,
    image: str,
    container_executable: str,
    labels: dict[str, str] | None,
) -> bool:
    if not labels:
        logger.warning(
            "cleanup %s skipped residual container removal for image %s: "
            "attempt labels unavailable",
            instance_id,
            image,
        )
        return False
    container_ids = _container_ids_for_attempt(
        image=image,
        container_executable=container_executable,
        labels=labels,
    )
    if not container_ids:
        return False
    result = subprocess.run(
        [container_executable, "rm", "-f", *container_ids],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to remove containers for image {image}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    logger.warning(
        "cleanup %s removed %d residual container(s) for image %s",
        instance_id,
        len(container_ids),
        image,
    )
    return True


def _free_disk_gb(path: Path) -> float | None:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except OSError:
        return None


def _container_storage_root(container_executable: str | None) -> Path | None:
    if container_executable is None:
        return None
    try:
        result = subprocess.run(
            [container_executable, "info", "--format", "{{.DockerRootDir}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    if not root or root == "<no value>":
        return None
    return Path(root)


def _low_disk_locations(
    *,
    run_dir: Path,
    container_executable: str | None,
    min_free_disk_gb: float,
) -> list[tuple[str, Path, float]]:
    locations: list[tuple[str, Path]] = [("run_dir", run_dir)]
    container_root = _container_storage_root(container_executable)
    if container_root is not None:
        locations.append(("container_storage", container_root))

    low: list[tuple[str, Path, float]] = []
    seen: set[str] = set()
    for label, path in locations:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        free_gb = _free_disk_gb(path)
        if free_gb is not None and free_gb < min_free_disk_gb:
            low.append((label, path, free_gb))
    return low


async def _run_scaffold_tasks(
    *,
    benchmark: "Benchmark",
    tasks: list[dict[str, Any]],
    run_dir: Path,
    model: str,
    scaffold: str,
    container_executable: str | None,
    prompt_template: str | None,
    min_free_disk_gb: float,
    monitoring_policy: MonitoringPolicy | None = None,
    concurrency: int = 1,
    rerun_completed: bool = False,
    inner_factory,
    recording_provider: Any | None = None,
) -> Path:
    """Iterate over tasks, wrapping each in ``run_attempt``.

    When ``concurrency > 1``, each task spawns *concurrency* agent instances
    simultaneously. The resolved policy may enable isolated container
    CPU/memory/I/O sampling, but PMU and host memory bandwidth are always off.
    Ksys, when enabled, runs once for the entire concurrent batch.
    """
    if monitoring_policy is None:
        monitoring_policy = MonitoringPolicy(
            resource_requested="off",
            pmu_requested="off",
            ksys_requested="off",
            resource_enabled=False,
            pmu_enabled=False,
            memory_bandwidth_enabled=False,
            ksys_enabled=False,
            concurrent=concurrency > 1,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(run_dir)
    if completed:
        if rerun_completed:
            logger.info(
                "Rerun requested: %d completed/exhausted task(s) remain eligible",
                len(completed),
            )
        else:
            logger.info("Resuming: %d tasks already completed", len(completed))

    results: list[CollectedTaskResult] = []
    total = len(tasks)
    prefetched_source_image: str | None = None
    prefetch_future: Future[None] | None = None
    resolved_prompt_template = _resolve_prompt_template(
        benchmark=benchmark,
        prompt_template=prompt_template,
    )

    # ── ksys system metrics (concurrent mode) ──────────────────────
    # In concurrent mode ksys runs once for the entire batch (not per-task),
    # capturing system-wide metrics independently from per-container samplers.
    ksys_session: KsysSession | None = None
    if concurrency > 1 and monitoring_policy.ksys_enabled:
        ksys_session = KsysSession.start(
            output_dir=run_dir.resolve(),
            log_dir=run_dir,
        )
        if ksys_session is not None:
            logger.info(
                "ksys collect started for concurrent batch (×%d)",
                concurrency,
            )

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="image-prefetch"
    ) as executor:
        for i, task in enumerate(tasks):
            instance_id = task["instance_id"]
            if instance_id in completed and not rerun_completed:
                logger.info(
                    "[%d/%d] SKIP %s (already completed)", i + 1, total, instance_id
                )
                continue

            logger.info("[%d/%d] START %s (%s)", i + 1, total, instance_id, scaffold)
            t0 = time.monotonic()

            # ARM-native mode: validate repo mirrors exist before
            # attempting container tasks.  QEMU mode skips this — the
            # x86_64 task images already contain the repo code.
            if (
                is_arm_host()
                and not use_arm_qemu()
                and benchmark.execution_environment == "container"
            ):
                repos_root = benchmark.config.repos_root
                if repos_root is None:
                    raise ValueError(
                        f"ARM host requires repos_root in benchmark config. "
                        f"Add repos_root to {benchmark.config.slug}.yaml"
                    )
                if not repos_root.exists():
                    raise ValueError(
                        f"ARM repos_root does not exist: {repos_root}. "
                        f"Run: make setup-{benchmark.config.slug.replace('-', '_')}-repos"
                    )

            source_image = _task_source_image(benchmark, task)
            next_source_image = _next_pending_source_image(
                benchmark,
                tasks,
                current_index=i,
                completed=completed,
                rerun_completed=rerun_completed,
            )

            attempt_ctx = AttemptContext(
                run_dir=run_dir,
                instance_id=instance_id,
                attempt=next_attempt_number(run_dir, instance_id),
                task=task,
                model=model,
                scaffold=scaffold,
                source_image=source_image,
                prompt_template=resolved_prompt_template,
                agent_runtime_mode=benchmark.runtime_mode_for(scaffold),
                execution_environment=getattr(
                    benchmark,
                    "execution_environment",
                    "container",
                ),
                # ARM-native fields (ignored on x86 hosts).
                arm_repo=task.get("repo"),
                arm_base_commit=task.get("base_commit"),
                arm_install_config=task.get("install_config"),
                arm_repos_root=getattr(benchmark.config, "repos_root", None),
            )

            _inner = inner_factory(task)

            # _cleanup_ctx tracks which AttemptContext to use for image
            # cleanup in the finally block.  In sequential mode it is
            # simply the single attempt_ctx; in concurrent mode it is
            # overwritten with the first concurrent context once that
            # branch is entered.
            _cleanup_ctx = attempt_ctx

            try:
                _ensure_task_source_ready(
                    instance_id=instance_id,
                    source_image=source_image,
                    prefetched_source_image=prefetched_source_image,
                    prefetch_future=prefetch_future,
                    container_executable=container_executable,
                )
                prefetched_source_image = None
                prefetch_future = None

                if (
                    next_source_image
                    and next_source_image != source_image
                    and container_executable is not None
                ):
                    logger.info(
                        "prefetch start for next task after %s image=%s",
                        instance_id,
                        next_source_image,
                    )
                    prefetched_source_image = next_source_image
                    prefetch_future = executor.submit(
                        ensure_source_image,
                        next_source_image,
                        container_executable=container_executable,
                    )

                if concurrency > 1:
                    # ── concurrent mode ──────────────────────────────────
                    # The resolved policy controls per-container monitoring.
                    # Ksys, if enabled, was started once before the task loop.
                    base_attempt = next_attempt_number(run_dir, instance_id)
                    concurrent_coros: list[
                        tuple[int, AttemptContext, dict[str, Any]]
                    ] = []

                    for n in range(concurrency):
                        concurrent_ctx = AttemptContext(
                            run_dir=run_dir,
                            instance_id=instance_id,
                            attempt=base_attempt + n,
                            task=task,
                            model=model,
                            scaffold=scaffold,
                            source_image=source_image,
                            prompt_template=resolved_prompt_template,
                            agent_runtime_mode=benchmark.runtime_mode_for(
                                scaffold
                            ),
                            execution_environment=getattr(
                                benchmark,
                                "execution_environment",
                                "container",
                            ),
                            arm_repo=task.get("repo"),
                            arm_base_commit=task.get("base_commit"),
                            arm_install_config=task.get("install_config"),
                            arm_repos_root=getattr(
                                benchmark.config,
                                "repos_root",
                                None,
                            ),
                        )
                        concurrent_inner = inner_factory(task)
                        concurrent_kwargs: dict[str, Any] = {
                            "inner": concurrent_inner,
                            "min_free_disk_gb": min_free_disk_gb,
                            "container_executable": container_executable,
                            "enable_ksys": False,
                            "disable_resource_monitoring": (
                                not monitoring_policy.resource_enabled
                            ),
                            "enable_pmu_monitoring": (
                                monitoring_policy.pmu_enabled
                            ),
                            "enable_memory_bandwidth_monitoring": (
                                monitoring_policy.memory_bandwidth_enabled
                            ),
                            "monitoring_policy": monitoring_policy.to_dict(),
                        }
                        if recording_provider is not None:
                            concurrent_kwargs["recording_provider"] = (
                                recording_provider
                            )
                        concurrent_coros.append(
                            (n, concurrent_ctx, concurrent_kwargs)
                        )

                    # Run each attempt in its own OS thread via
                    # asyncio.to_thread().  This is necessary because
                    # the task_container_agent path (and potentially
                    # other scaffolds) uses synchronous subprocess.run()
                    # calls that would otherwise block the asyncio event
                    # loop, preventing other coroutines from starting.
                    def _run_attempt_sync(
                        ctx: AttemptContext, **kw: Any,
                    ) -> AttemptResult:
                        return asyncio.run(run_attempt(ctx, **kw))

                    raw_results = await asyncio.gather(
                        *[
                            asyncio.to_thread(
                                _run_attempt_sync, ctx, **kw,
                            )
                            for _, ctx, kw in concurrent_coros
                        ],
                        return_exceptions=True,
                    )

                    # Collect per-attempt results.
                    for idx, (n, ctx, _) in enumerate(concurrent_coros):
                        raw = raw_results[idx]
                        if isinstance(raw, BaseException):
                            logger.exception(
                                "FAILED %s attempt_%d",
                                instance_id,
                                base_attempt + n,
                            )
                            results.append(
                                CollectedTaskResult(
                                    instance_id=instance_id,
                                    attempt_dir=ctx.attempt_dir,
                                    success=False,
                                    model_patch="",
                                    exit_status="error",
                                    error=f"{type(raw).__name__}: {raw}",
                                    elapsed_s=time.monotonic() - t0,
                                )
                            )
                        elif isinstance(raw, AttemptResult):
                            results.append(
                                CollectedTaskResult(
                                    instance_id=instance_id,
                                    attempt_dir=ctx.attempt_dir,
                                    success=raw.success,
                                    model_patch=getattr(raw, "model_patch", "")
                                    or "",
                                    exit_status=raw.exit_status,
                                    error=raw.error,
                                    elapsed_s=time.monotonic() - t0,
                                    n_iterations=raw.n_iterations,
                                )
                            )
                            # Per-attempt HTML visualization.
                            try:
                                from trace_collect.html_viz import generate_html

                                viz_path = (
                                    ctx.attempt_dir
                                    / benchmark.viz_filename(instance_id)
                                )
                                viz_path.write_text(
                                    generate_html(ctx.attempt_dir),
                                    encoding="utf-8",
                                )
                            except Exception:
                                logger.warning(
                                    "viz %s attempt_%d failed (non-fatal)",
                                    instance_id,
                                    base_attempt + n,
                                    exc_info=True,
                                )
                        else:
                            logger.error(
                                "Unexpected result type for %s attempt_%d: %s",
                                instance_id,
                                base_attempt + n,
                                type(raw).__name__,
                            )
                            results.append(
                                CollectedTaskResult(
                                    instance_id=instance_id,
                                    attempt_dir=ctx.attempt_dir,
                                    success=False,
                                    model_patch="",
                                    exit_status="error",
                                    error=(
                                        f"Unexpected result type: "
                                        f"{type(raw).__name__}"
                                    ),
                                    elapsed_s=time.monotonic() - t0,
                                )
                            )

                    # Use the first attempt context for image cleanup
                    # (all concurrent attempts share the same source image).
                    _cleanup_ctx = concurrent_coros[0][1]

                else:
                    # ── sequential mode (original behaviour) ─────────────
                    run_attempt_kwargs: dict[str, Any] = {
                        "inner": _inner,
                        "min_free_disk_gb": min_free_disk_gb,
                        "container_executable": container_executable,
                        "enable_ksys": monitoring_policy.ksys_enabled,
                        "disable_resource_monitoring": (
                            not monitoring_policy.resource_enabled
                        ),
                        "enable_pmu_monitoring": monitoring_policy.pmu_enabled,
                        "enable_memory_bandwidth_monitoring": (
                            monitoring_policy.memory_bandwidth_enabled
                        ),
                        "monitoring_policy": monitoring_policy.to_dict(),
                    }
                    if recording_provider is not None:
                        run_attempt_kwargs["recording_provider"] = (
                            recording_provider
                        )
                    result = await run_attempt(
                        attempt_ctx, **run_attempt_kwargs
                    )
            except Exception as exc:
                logger.exception("FAILED %s", instance_id)
                results.append(
                    CollectedTaskResult(
                        instance_id=instance_id,
                        attempt_dir=attempt_ctx.attempt_dir,
                        success=False,
                        model_patch="",
                        exit_status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        elapsed_s=time.monotonic() - t0,
                    )
                )
            else:
                if concurrency <= 1:
                    # Sequential mode: single result.
                    results.append(
                        CollectedTaskResult(
                            instance_id=instance_id,
                            attempt_dir=attempt_ctx.attempt_dir,
                            success=result.success,
                            model_patch=getattr(result, "model_patch", "")
                            or "",
                            exit_status=result.exit_status,
                            error=result.error,
                            elapsed_s=time.monotonic() - t0,
                            n_iterations=result.n_iterations,
                        )
                    )
                    logger.info(
                        "[%d/%d] DONE %s success=%s elapsed=%.1fs",
                        i + 1,
                        total,
                        instance_id,
                        results[-1].success,
                        results[-1].elapsed_s,
                    )

                    # Generate per-task HTML visualization.
                    try:
                        from trace_collect.html_viz import generate_html

                        viz_path = attempt_ctx.attempt_dir / benchmark.viz_filename(
                            instance_id
                        )
                        viz_path.write_text(
                            generate_html(attempt_ctx.attempt_dir),
                            encoding="utf-8",
                        )
                        logger.info(
                            "viz %s → %s (%.1f KB)",
                            instance_id,
                            viz_path,
                            viz_path.stat().st_size / 1024,
                        )
                    except Exception:
                        logger.warning(
                            "viz %s failed (non-fatal)",
                            instance_id,
                            exc_info=True,
                        )
                else:
                    # Concurrent mode: results already collected above.
                    n_success = sum(
                        1 for r in results[-concurrency:]
                        if r.success
                    )
                    logger.info(
                        "[%d/%d] DONE %s ×%d success=%d/%d elapsed=%.1fs",
                        i + 1,
                        total,
                        instance_id,
                        concurrency,
                        n_success,
                        concurrency,
                        time.monotonic() - t0,
                    )
            finally:
                low_disk_locations = _low_disk_locations(
                    run_dir=run_dir,
                    container_executable=container_executable,
                    min_free_disk_gb=min_free_disk_gb,
                )
                cleanup_low_disk = bool(low_disk_locations)
                if cleanup_low_disk:
                    details = "; ".join(
                        f"{label}={path} free={free_gb:.2f}GB"
                        for label, path, free_gb in low_disk_locations
                    )
                    logger.warning(
                        "cleanup %s running in low-disk mode: %s "
                        "< %.2f GB threshold; not keeping next source image",
                        instance_id,
                        details,
                        min_free_disk_gb,
                    )
                _cleanup_task_images(
                    instance_id=instance_id,
                    source_image=source_image,
                    fixed_image=_cleanup_ctx.fixed_image,
                    keep_source_image=None if cleanup_low_disk else next_source_image,
                    container_executable=container_executable,
                    run_dir=run_dir,
                    remove_containers=cleanup_low_disk,
                    force_cleanup=cleanup_low_disk,
                    container_labels=_cleanup_ctx.container_labels(),
                )

    # ── stop ksys after all tasks complete ─────────────────────────
    if ksys_session is not None:
        ksys_session.stop()

    write_results_jsonl(results, run_dir / "results.jsonl")
    logger.info("Results written to %s", run_dir / "results.jsonl")
    return run_dir


async def collect_traces(
    *,
    scaffold: str,
    api_base: str,
    api_key: str,
    model: str,
    benchmark: "Benchmark",
    provider_name: str | None = None,
    env_key: str | None = None,
    max_iterations: int = 100,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    sample: int | None = None,
    skip: int | None = None,
    instance_ids: list[str] | None = None,
    run_id: str | None = None,
    rerun_completed: bool = False,
    max_context_tokens: int = 256_000,
    container_executable: str | None = None,
    mcp_config: str | None = None,
    prompt_template: str | None = None,
    min_free_disk_gb: float = 30.0,
    capture_pytest_scripts: bool = True,
    capture_pytest_runtime: bool = True,
    capture_pip_runtime: bool = True,
    capture_python_script_runtime: bool = True,
    record_internals: bool = False,
    resource_monitoring: str = "auto",
    pmu_monitoring: str = "auto",
    ksys_monitoring: str = "auto",
    tool_profiling: str = "off",
    tool_profiling_tools: list[str] | None = None,
    concurrency: int = 1,
    eviction_config: "EvictionPolicyConfig | None" = None,
    sparse_attention_config: "SparseAttentionConfig | None" = None,
) -> Path:
    """Collect traces for any scaffold supported by the benchmark plugin."""
    if record_internals and scaffold != "openclaw":
        raise ValueError(
            "--record-internals currently supports scaffold='openclaw' only"
        )
    if concurrency > 1 and instance_ids is None and sample is None and skip is None:
        raise ValueError(
            "--concurrency > 1 requires --instance-ids, --sample, or --skip. "
            "Running all benchmark instances with high concurrency is "
            "almost certainly unintentional. "
            "Specify --instance-ids <id> or --sample N to limit the scope."
        )
    benchmark.validate_scaffold_support(scaffold)
    execution_environment = benchmark.execution_environment
    if execution_environment not in {"container", "host"}:
        raise ValueError(
            f"Unsupported benchmark.execution_environment: {execution_environment!r}"
        )

    runtime_mode = benchmark.runtime_mode_for(scaffold)
    if runtime_mode not in {"host_controller", "task_container_agent"}:
        raise NotImplementedError(
            f"Unsupported benchmark.runtime_mode_for({scaffold!r}): {runtime_mode!r}"
        )
    if tool_profiling not in {"off", "vtune", "ksys", "tool_profiler", "tool_scheduler"}:
        raise ValueError(
            f"--tool-profiling must be 'off', 'vtune', 'ksys', 'tool_profiler', "
            f"or 'tool_scheduler', got {tool_profiling!r}"
        )
    vtune = tool_profiling == "vtune"
    tool_profiler_mode = tool_profiling == "tool_profiler"
    tool_scheduler_mode = tool_profiling == "tool_scheduler"
    if vtune and runtime_mode != "task_container_agent":
        raise ValueError(
            "--tool-profiling vtune wraps in-container commands and only applies to "
            f"task-container benchmarks, not runtime_mode={runtime_mode!r}"
        )
    if tool_profiler_mode and runtime_mode != "task_container_agent":
        raise ValueError(
            "--tool-profiling tool_profiler wraps in-container commands and only "
            f"applies to task-container benchmarks, not runtime_mode={runtime_mode!r}"
        )
    if tool_scheduler_mode and runtime_mode != "task_container_agent":
        raise ValueError(
            "--tool-profiling tool_scheduler wraps in-container commands and only "
            f"applies to task-container benchmarks, not runtime_mode={runtime_mode!r}"
        )
    # Future: add ksys per-tool validation here when implemented.
    if tool_profiling_tools is None:
        tool_profiling_tools = ["exec-pytest"]
    if (
        execution_environment == "container" or runtime_mode == "task_container_agent"
    ) and container_executable is None:
        raise ValueError("--container required for container-mode benchmarks")

    effective_ksys = ksys_monitoring
    monitoring_policy = resolve_collect_monitoring(
        resource=resource_monitoring,
        pmu=pmu_monitoring,
        ksys=effective_ksys,
        concurrency=concurrency,
        execution_environment=execution_environment,
        tool_profiling=tool_profiling,
        tool_profiling_tools=tool_profiling_tools,
    )

    run_dir = Path(run_id) if run_id else build_run_dir(benchmark, model)
    generation_config = _generation_config(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    with ExitStack() as cleanup_stack:
        recording_provider = None
        recording_server = None
        if record_internals:
            # Lazy: `serving.recording` triggers `transformers.cache_utils → torch`.
            from serving.recording import HFRecordingProvider, HFRecordingServer

            recording_provider = cleanup_stack.enter_context(
                HFRecordingProvider(
                    default_model=model,
                    eviction_config=eviction_config,
                    sparse_attention_config=sparse_attention_config,
                    temperature=temperature if temperature is not None else 0.1,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
            )
            recording_server = cleanup_stack.enter_context(
                HFRecordingServer(
                    recording_provider,
                    public_host=_recording_server_public_host(
                        execution_environment=execution_environment,
                        runtime_mode=runtime_mode,
                        container_executable=container_executable,
                    ),
                )
            )
        effective_api_base = recording_server.api_base if recording_server else api_base
        effective_api_key = "hf-recording" if recording_server else api_key
        effective_provider_name = "openai" if recording_server else provider_name
        provider = recording_provider or UnifiedProvider(
            api_key=api_key,
            api_base=api_base,
            default_model=model,
            **generation_config,
        )
        runner = None
        if runtime_mode == "host_controller":
            runner = benchmark.build_runner(
                scaffold=scaffold,
                provider=provider,
                workspace_base=run_dir / "_workspace_base",
                max_iterations=max_iterations,
                context_window_tokens=max_context_tokens,
                model=model,
                provider_name=effective_provider_name,
                env_key=env_key,
                api_base=effective_api_base,
                api_key=effective_api_key,
                mcp_config=mcp_config,
                mcp_servers=load_mcp_servers(mcp_config),
                generation_config=generation_config,
            )

        tasks = _select_tasks(
            benchmark.load_tasks(),
            instance_ids=instance_ids,
            skip=skip,
            sample=sample,
        )

        logger.info(
            "Selected %d task(s): %s",
            len(tasks),
            ", ".join(t["instance_id"] for t in tasks),
        )

        # Report which images are already pulled.
        if container_executable:
            _log_image_status(tasks, benchmark, container_executable)

        def make_inner(task: dict[str, Any]):
            async def inner(ctx: AttemptContext) -> AttemptResult:
                if ctx.agent_runtime_mode == "task_container_agent":
                    if scaffold != "openclaw":
                        raise NotImplementedError(
                            "task-container collection currently supports "
                            f"scaffold='openclaw', got {scaffold!r}"
                        )
                    if container_executable is None:
                        raise ValueError(
                            "container_executable is required for task-container runs"
                        )
                    return await _run_openclaw_in_task_container(
                        ctx=ctx,
                        task=task,
                        benchmark=benchmark,
                        provider_name=effective_provider_name,
                        api_base=effective_api_base,
                        api_key=effective_api_key,
                        model=model,
                        max_iterations=max_iterations,
                        generation_config=generation_config,
                        max_context_tokens=max_context_tokens,
                        mcp_config=mcp_config,
                        container_executable=container_executable,
                        record_internals=record_internals,
                        tool_profiling=tool_profiling,
                        tool_profiling_tools=tool_profiling_tools,
                        capture_pytest_scripts=capture_pytest_scripts,
                        capture_pytest_runtime=capture_pytest_runtime,
                        capture_pip_runtime=capture_pip_runtime,
                        capture_python_script_runtime=capture_python_script_runtime,
                    )

                assert runner is not None
                run_task_kwargs: dict[str, Any] = {
                    "attempt_ctx": ctx,
                    "prompt_template": ctx.prompt_template,
                }
                run_task_params = inspect.signature(runner.run_task).parameters
                if "capture_pytest_scripts" in run_task_params:
                    run_task_kwargs["capture_pytest_scripts"] = capture_pytest_scripts
                if "capture_pytest_runtime" in run_task_params:
                    run_task_kwargs["capture_pytest_runtime"] = capture_pytest_runtime
                if "pytest_history_dir" in run_task_params and capture_pytest_runtime:
                    run_task_kwargs["pytest_history_dir"] = (
                        _pytest_history_scope_dir(run_dir, instance_id)
                    )
                if "pytest_family_history_dir" in run_task_params and capture_pytest_runtime:
                    run_task_kwargs["pytest_family_history_dir"] = (
                        _family_history_scope_dir(run_dir, "pytest", task, instance_id)
                    )
                if "capture_pip_runtime" in run_task_params:
                    run_task_kwargs["capture_pip_runtime"] = capture_pip_runtime
                if "pip_history_dir" in run_task_params and capture_pip_runtime:
                    run_task_kwargs["pip_history_dir"] = (
                        _pip_history_scope_dir(run_dir, instance_id)
                    )
                if "pip_family_history_dir" in run_task_params and capture_pip_runtime:
                    run_task_kwargs["pip_family_history_dir"] = (
                        _family_history_scope_dir(run_dir, "pip", task, instance_id)
                    )
                if "capture_python_script_runtime" in run_task_params:
                    run_task_kwargs["capture_python_script_runtime"] = (
                        capture_python_script_runtime
                    )
                if (
                    "python_script_history_dir" in run_task_params
                    and capture_python_script_runtime
                ):
                    run_task_kwargs["python_script_history_dir"] = (
                        _python_script_history_scope_dir(run_dir, instance_id)
                    )
                if (
                    "python_script_family_history_dir" in run_task_params
                    and capture_python_script_runtime
                ):
                    run_task_kwargs["python_script_family_history_dir"] = (
                        _family_history_scope_dir(run_dir, "python_script", task, instance_id)
                    )
                result = await runner.run_task(
                    task,
                    **run_task_kwargs,
                )
                if not isinstance(result, AttemptResult):
                    raise TypeError(
                        "benchmark runner returned "
                        f"{type(result).__name__}, expected AttemptResult"
                    )
                if result.trace_path is not None:
                    run_config: dict[str, Any] = {
                        "capture_pytest_scripts": capture_pytest_scripts,
                        "capture_pytest_runtime": capture_pytest_runtime,
                        "capture_pip_runtime": capture_pip_runtime,
                        "capture_python_script_runtime": (
                            capture_python_script_runtime
                        ),
                        "rerun_completed": rerun_completed,
                    }
                    if record_internals:
                        run_config["record_internals"] = True
                    _stamp_trace_run_config(result.trace_path, run_config)
                return result

            return inner

        return await _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=tasks,
            run_dir=run_dir,
            model=model,
            scaffold=scaffold,
            container_executable=container_executable,
            prompt_template=prompt_template,
            min_free_disk_gb=min_free_disk_gb,
            monitoring_policy=monitoring_policy,
            concurrency=concurrency,
            rerun_completed=rerun_completed,
            inner_factory=make_inner,
            recording_provider=recording_provider,
        )


def _finalize_tool_profiler(
    out_dir: Path,
    attempt_dir: Path,
) -> None:
    """Collect per-tool profile JSONL files and write a summary.

    Each matched tool invocation writes ``profile.jsonl`` in a timestamped
    subdirectory of *out_dir*.  This function discovers them, aggregates
    key metrics, and writes ``tool_profiles_summary.json`` to *attempt_dir*.
    """
    profile_files = sorted(out_dir.glob("*/profile.jsonl"))
    if not profile_files:
        return

    records: list[dict] = []
    for pf in profile_files:
        try:
            text = pf.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue

    if not records:
        return

    # Aggregate summary
    n_total = len(records)
    n_early = sum(1 for r in records if r.get("early_profile", {}).get("available"))
    n_short = sum(1 for r in records if r.get("final_profile", {}).get("short_tool"))

    behaviors: dict[str, int] = {}
    for r in records:
        beh = r.get("final_profile", {}).get("preliminary_behavior", "unknown")
        behaviors[beh] = behaviors.get(beh, 0) + 1

    cores = [
        r.get("final_profile", {}).get("avg_effective_cores", 0.0)
        for r in records
    ]
    avg_cores = sum(cores) / len(cores) if cores else 0.0

    # Early-final consistency
    records_with_early = [
        r for r in records if r.get("early_profile", {}).get("available")
    ]
    behavior_changed = sum(
        1 for r in records_with_early
        if r.get("early_final_comparison", {}).get("behavior_changed")
    )
    errors = [
        r.get("early_final_comparison", {}).get("effective_cores_relative_error", 0.0) or 0.0
        for r in records_with_early
    ]
    avg_rel_error = sum(errors) / len(errors) if errors else 0.0

    summary: dict = {
        "schema_version": 1,
        "n_tool_invocations": n_total,
        "n_short_tools": n_short,
        "n_with_early_profile": n_early,
        "behavior_distribution": behaviors,
        "avg_effective_cores": avg_cores,
        "early_final_consistency": {
            "behavior_changed": behavior_changed,
            "n_with_early": len(records_with_early),
            "avg_effective_cores_relative_error": avg_rel_error,
        },
    }

    summary_path = attempt_dir / "tool_profiles_summary.json"
    try:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass

    # Also copy all individual profile files into the attempt dir for
    # easy access alongside the trace.
    profiles_dest = attempt_dir / "tool_profiles"
    profiles_dest.mkdir(parents=True, exist_ok=True)
    for pf in profile_files:
        dest = profiles_dest / pf.parent.name / pf.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(pf.read_bytes())
        except OSError:
            pass


def _finalize_tool_scheduler(
    out_dir: Path,
    attempt_dir: Path,
) -> None:
    """Collect per-tool scheduler JSONL files and write a summary.

    Each matched tool invocation writes ``profile.jsonl`` in a timestamped
    subdirectory of *out_dir*.  This function discovers them, aggregates
    key metrics, and writes ``tool_scheduler_summary.json`` to *attempt_dir*.
    """
    profile_files = sorted(out_dir.glob("*/profile.jsonl"))
    if not profile_files:
        return

    records: list[dict] = []
    for pf in profile_files:
        try:
            text = pf.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue

    if not records:
        return

    n_total = len(records)
    n_short = sum(
        1 for r in records
        if r.get("final_profile", {}).get("short_tool")
    )
    n_decisions = sum(len(r.get("decisions", [])) for r in records)
    n_moves = sum(
        1 for r in records
        for d in r.get("decisions", [])
        if d.get("action") == "recommend_move"
    )

    cores = [
        r.get("final_profile", {}).get("median_effective_cores", 0.0)
        for r in records
    ]
    avg_cores = sum(cores) / len(cores) if cores else 0.0

    summary: dict = {
        "schema_version": 1,
        "n_tool_invocations": n_total,
        "n_short_tools": n_short,
        "n_decisions": n_decisions,
        "n_recommend_moves": n_moves,
        "avg_median_effective_cores": avg_cores,
    }

    summary_path = attempt_dir / "tool_scheduler_summary.json"
    try:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass

    # Copy all individual profile files into the attempt dir
    profiles_dest = attempt_dir / "tool_scheduler"
    profiles_dest.mkdir(parents=True, exist_ok=True)
    for pf in profile_files:
        dest = profiles_dest / pf.parent.name / pf.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(pf.read_bytes())
        except OSError:
            pass


def _normalize_openclaw_trace(
    src: Path,
    dst: Path,
    *,
    benchmark: "Benchmark",
    model: str,
    api_base: str,
    max_iterations: int,
    instance_id: str,
    mcp_config_label: str | None = None,
    prompt_template: str | None = None,
    agent_runtime_mode: str | None = None,
    runtime_proof: dict[str, Any] | None = None,
    record_internals: bool = False,
    generation_config: dict[str, Any] | None = None,
    capture_pytest_scripts: bool = True,
    capture_pytest_runtime: bool = True,
    capture_pip_runtime: bool = True,
    capture_python_script_runtime: bool = True,
) -> None:
    """Copy an OpenClaw trace into the attempt dir, merging trace metadata."""
    lines = src.read_text(encoding="utf-8").splitlines()
    source_metadata: dict[str, Any] | None = None
    body_start = 0
    for idx, line in enumerate(lines):
        if not line.strip():
            body_start = idx + 1
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            body_start = idx + 1
            continue
        if rec.get("type") == "trace_metadata":
            source_metadata = rec
            body_start = idx + 1
        else:
            body_start = idx
        break

    execution_environment = getattr(benchmark, "execution_environment", "container")
    merged: dict[str, Any] = {
        "type": "trace_metadata",
        "scaffold": "openclaw",
        "trace_format_version": 5,
        "mode": "collect",
        "scaffold_capabilities": {"unknown": True},
    }
    if source_metadata is not None:
        merged.update(source_metadata)
    merged.setdefault("benchmark", benchmark.config.slug)
    if benchmark.config.harness_split is not None:
        merged.setdefault("benchmark_split", benchmark.config.harness_split)
    merged.setdefault("model", model)
    merged.setdefault("api_base", api_base)
    merged.setdefault("max_iterations", max_iterations)
    merged.setdefault("instance_id", instance_id)
    if prompt_template is not None:
        merged["prompt_template"] = prompt_template
    if agent_runtime_mode is not None:
        merged["agent_runtime_mode"] = agent_runtime_mode
    if runtime_proof:
        merged["runtime_proof"] = runtime_proof
    merged["type"] = "trace_metadata"
    merged["trace_format_version"] = 5
    merged["execution_environment"] = execution_environment
    if mcp_config_label is not None:
        run_config = merged.get("run_config") or {}
        run_config["mcp_config"] = mcp_config_label
        merged["run_config"] = run_config
    if record_internals:
        run_config = merged.get("run_config") or {}
        run_config["record_internals"] = True
        merged["run_config"] = run_config
    if generation_config:
        run_config = merged.get("run_config") or {}
        run_config["generation"] = dict(generation_config)
        merged["run_config"] = run_config
    run_config = merged.get("run_config") or {}
    run_config["capture_pytest_scripts"] = capture_pytest_scripts
    run_config["capture_pytest_runtime"] = capture_pytest_runtime
    run_config["capture_pip_runtime"] = capture_pip_runtime
    run_config["capture_python_script_runtime"] = capture_python_script_runtime
    merged["run_config"] = run_config

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(json.dumps(merged, ensure_ascii=False) + "\n")
        for line in lines[body_start:]:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "trace_metadata":
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _resolve_prompt_template(
    *,
    benchmark: "Benchmark",
    prompt_template: str | None,
) -> str:
    return prompt_template or benchmark.config.default_prompt_template


def _stamp_trace_run_config(trace_path: Path, values: dict[str, Any]) -> None:
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        if not replaced and rec.get("type") == "trace_metadata":
            run_config = dict(rec.get("run_config") or {})
            run_config.update(values)
            rec["run_config"] = run_config
            replaced = True
        out.append(json.dumps(rec, ensure_ascii=False))
    trace_path.write_text("\n".join(out) + "\n", encoding="utf-8")


async def _run_openclaw_in_task_container(
    *,
    ctx: AttemptContext,
    task: dict[str, Any],
    benchmark: "Benchmark",
    provider_name: str | None,
    api_base: str,
    api_key: str,
    model: str,
    container_executable: str,
    max_iterations: int,
    generation_config: dict[str, Any] | None,
    max_context_tokens: int,
    mcp_config: str | None,
    record_internals: bool = False,
    tool_profiling: str = "off",
    tool_profiling_tools: list[str] | None = None,
    capture_pytest_scripts: bool = True,
    capture_pytest_runtime: bool = True,
    capture_pip_runtime: bool = True,
    capture_python_script_runtime: bool = True,
) -> AttemptResult:
    fixed_image = ctx.fixed_image or task.get("image_name") or ""
    if not fixed_image:
        raise RuntimeError(f"Task {ctx.instance_id!r} has no image_name")

    runtime_dir = ctx.attempt_dir.resolve() / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    proof = None
    runtime = None
    runtime_proof = None
    exec_config = resolve_task_container_exec_config(
        attempt_dir=ctx.attempt_dir,
        image=fixed_image,
        container_executable=container_executable,
    )
    start_extra_args = list(exec_config.start_extra_args)
    vtune_out_dir = ctx.attempt_dir.resolve() / "vtune"
    tool_profiler_out_dir = ctx.attempt_dir.resolve() / "tool_profiles"
    tool_scheduler_out_dir = ctx.attempt_dir.resolve() / "tool_scheduler"
    vtune = tool_profiling == "vtune"
    ksys_tool = tool_profiling == "ksys"
    tool_profiler_mode = tool_profiling == "tool_profiler"
    tool_scheduler_mode = tool_profiling == "tool_scheduler"
    if vtune:
        from trace_collect.vtune_report import vtune_container_run_args

        start_extra_args.extend(
            vtune_container_run_args(vtune_out_dir, tools=tool_profiling_tools or ["exec-pytest"])
        )
    elif ksys_tool:
        # ksys per-tool: coarse-only for now (no ksys binary wrapping).
        # Activate the in-container /proc sampler via env vars.
        vtune_out_dir.mkdir(parents=True, exist_ok=True)
        tools = ",".join(tool_profiling_tools or ["exec-pytest"])
        start_extra_args.extend([
            "-e", "VTUNE_PROFILE=1",
            "-e", f"VTUNE_OUT={vtune_out_dir.resolve()}",
            "-e", f"VTUNE_TOOLS={tools}",
        ])
    elif tool_profiler_mode:
        # tool_profiler per-tool: uses the prototype online process-tree profiler.
        # The profiler wraps matching tool commands with:
        #   python -m prototype.tool_profiler --output <profile.jsonl> -- <command>
        # Activation is via env vars read by ExecTool.execute() in shell.py.
        tool_profiler_out_dir.mkdir(parents=True, exist_ok=True)
        tools = ",".join(tool_profiling_tools or ["exec-pytest"])
        start_extra_args.extend([
            "-e", "TOOL_PROFILER=1",
            "-e", f"TOOL_PROFILER_OUT={tool_profiler_out_dir.resolve()}",
            "-e", f"TOOL_PROFILER_TOOLS={tools}",
        ])
    elif tool_scheduler_mode:
        # tool_scheduler per-tool: uses the prototype online load-prediction
        # scheduler.  Wraps matching tool commands with:
        #   python -m prototype.tool_scheduler --hardcode-topology
        #     --output <profile.jsonl> --dry-run -- <command>
        # Activation is via env vars read by ExecTool.execute() in shell.py.
        tool_scheduler_out_dir.mkdir(parents=True, exist_ok=True)
        tools = ",".join(tool_profiling_tools or ["exec-pytest"])
        start_extra_args.extend([
            "-e", "TOOL_SCHEDULER=1",
            "-e", f"TOOL_SCHEDULER_OUT={tool_scheduler_out_dir.resolve()}",
            "-e", f"TOOL_SCHEDULER_TOOLS={tools}",
            # Use hardcoded topology inside container (host topology
            # may not be visible from within container).
            "-e", "TOOL_SCHEDULER_HARDCODE_TOPOLOGY=1",
        ])
    container_id = start_task_container(
        fixed_image,
        executable=container_executable,
        extra_args=start_extra_args,
        labels=ctx.container_labels(),
    )
    ctx.mark_container_ready(container_id)

    # Start resource sampling directly in a thread — the asyncio watcher in
    # run_attempt cannot make progress because the calls below block the event
    # loop with synchronous subprocess.run().  Skip when the caller has
    # opted out (e.g. --concurrency > 1).
    _stats_sampler: ContainerStatsSampler | None = None
    if not ctx.disable_resource_monitoring:
        from harness.container_stats_sampler import ContainerStatsSampler
        _stats_sampler = ContainerStatsSampler(
            container_id=container_id,
            interval_s=0.5,
            executable=container_executable,
            # When VTune is active, the host-side perf-stat PMU sampling
            # competes with VTune's own PMU collection for the same set of
            # hardware counters, causing kernel multiplexing and degraded
            # accuracy on both sides.  Disable the host PMU to give VTune
            # exclusive counter access during profiled pytest windows.
            # ksys / coarse-only mode does not use PMU, so host PMU stays on.
            enable_pmu=ctx.enable_pmu_monitoring and not vtune,
            enable_memory_bandwidth=ctx.enable_memory_bandwidth_monitoring,
        )
        _stats_sampler.start()

    try:
        exec_config = resolve_running_container_exec_config(
            container_id=container_id,
            exec_config=exec_config,
            container_executable=container_executable,
        )
        preflight_imports = [
            "trace_collect.runtime.entrypoint",
            "agents.openclaw.eval.runner",
            "harness.trace_logger",
        ]
        bootstrap_requirements: tuple[str, ...] = ()
        if mcp_config not in {None, "none"}:
            preflight_imports.append("agents.openclaw.tools.mcp")
            bootstrap_requirements = OPENCLAW_MCP_RUNTIME_REQUIREMENTS
        if tool_profiler_mode:
            # tool_profiler needs psutil inside the container for process-tree
            # sampling.  Add it to the bootstrap requirements so it's
            # guaranteed to be importable when ExecTool wraps commands with
            # ``python -m prototype.tool_profiler``.
            bootstrap_requirements = bootstrap_requirements + ("psutil",)
        if tool_scheduler_mode:
            # tool_scheduler also needs psutil inside the container for
            # process-tree monitoring.
            bootstrap_requirements = bootstrap_requirements + ("psutil",)
        bootstrapped_config = bootstrap_task_container_python(
            container_id=container_id,
            exec_config=exec_config,
            extra_requirements=bootstrap_requirements,
            container_executable=container_executable,
        )
        if bootstrapped_config is not None:
            exec_config = bootstrapped_config
        proof = preflight_task_container_runtime(
            container_id=container_id,
            attempt_dir=ctx.attempt_dir,
            imports=preflight_imports,
            runtime=exec_config.runtime,
            pythonpath=exec_config.pythonpath,
            container_executable=container_executable,
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        pip_runtime_dir = (
            ctx.attempt_dir.resolve() / "pip_runtime"
            if capture_pip_runtime
            else None
        )
        pip_shared_history_dir = (
            _pip_history_scope_dir(ctx.run_dir, ctx.instance_id)
            if capture_pip_runtime
            else None
        )
        pip_family_history_dir = (
            _family_history_scope_dir(ctx.run_dir, "pip", task, ctx.instance_id)
            if capture_pip_runtime
            else None
        )
        if pip_runtime_dir is not None and pip_shared_history_dir is not None:
            seed_pip_history_from_shared(
                shared_history_root=pip_shared_history_dir,
                attempt_prediction_root=pip_runtime_dir,
            )
            if pip_family_history_dir is not None:
                seed_pip_family_history_from_shared(
                    shared_history_root=pip_family_history_dir,
                    attempt_prediction_root=pip_runtime_dir,
                )
        python_script_runtime_dir = (
            ctx.attempt_dir.resolve() / "python_script_runtime"
            if capture_python_script_runtime
            else None
        )
        python_script_shared_history_dir = (
            _python_script_history_scope_dir(ctx.run_dir, ctx.instance_id)
            if capture_python_script_runtime
            else None
        )
        python_script_family_history_dir = (
            _family_history_scope_dir(
                ctx.run_dir,
                "python_script",
                task,
                ctx.instance_id,
            )
            if capture_python_script_runtime
            else None
        )
        if (
            python_script_runtime_dir is not None
            and python_script_shared_history_dir is not None
        ):
            seed_python_script_history_from_shared(
                shared_history_root=python_script_shared_history_dir,
                attempt_prediction_root=python_script_runtime_dir,
            )
            if python_script_family_history_dir is not None:
                seed_python_script_family_history_from_shared(
                    shared_history_root=python_script_family_history_dir,
                    attempt_prediction_root=python_script_runtime_dir,
                )
        pytest_runtime_dir = (
            ctx.attempt_dir.resolve() / "pytest_runtime"
            if capture_pytest_runtime
            else None
        )
        pytest_shared_history_dir = (
            _pytest_history_scope_dir(ctx.run_dir, ctx.instance_id)
            if capture_pytest_runtime
            else None
        )
        pytest_family_history_dir = (
            _family_history_scope_dir(ctx.run_dir, "pytest", task, ctx.instance_id)
            if capture_pytest_runtime
            else None
        )
        if pytest_runtime_dir is not None and pytest_shared_history_dir is not None:
            seed_pytest_history_from_shared(
                shared_history_root=pytest_shared_history_dir,
                attempt_prediction_root=pytest_runtime_dir,
            )
            if pytest_family_history_dir is not None:
                seed_pytest_family_history_from_shared(
                    shared_history_root=pytest_family_history_dir,
                    attempt_prediction_root=pytest_runtime_dir,
                )
        runtime = run_task_container_agent(
            container_id=container_id,
            timeout=(max_iterations * 120) + 300,
            runtime=exec_config.runtime,
            pythonpath=exec_config.pythonpath,
            request={
                "kind": "run_openclaw",
                "scaffold": "openclaw",
                "result_path": str(runtime_dir / "run.result.json"),
                "container_id": container_id,
                "benchmark": benchmark.config.slug,
                "provider_name": provider_name,
                "api_base": api_base,
                "api_key": api_key,
                "model": model,
                "max_iterations": max_iterations,
                "generation_config": generation_config or {},
                "max_context_tokens": max_context_tokens,
                "prompt_template": ctx.prompt_template,
                "agent_runtime_mode": ctx.agent_runtime_mode,
                "mcp_config": (
                    str(Path(mcp_config).resolve())
                    if mcp_config not in {None, "none"}
                    else mcp_config
                ),
                "task": task,
                "workspace_base": str(runtime_dir / "workspace_base"),
                "workspace_dir": str(runtime_dir / "workspace_base" / ctx.instance_id),
                "tool_workspace": "/testbed",
                "exec_path_append": "/tmp/openclaw-task-userbase/bin",
                "exec_working_dir": "/testbed",
                "trace_file": str(runtime_dir / "trace.raw.jsonl"),
                "pytest_capture_dir": (
                    str(ctx.attempt_dir.resolve() / "pytest_scripts")
                    if capture_pytest_scripts
                    else None
                ),
                "pytest_runtime_dir": (
                    str(pytest_runtime_dir)
                    if pytest_runtime_dir is not None
                    else None
                ),
                "pip_runtime_dir": (
                    str(pip_runtime_dir)
                    if pip_runtime_dir is not None
                    else None
                ),
                "python_script_runtime_dir": (
                    str(python_script_runtime_dir)
                    if python_script_runtime_dir is not None
                    else None
                ),
                "raw_stdout_path": str(stdout_path),
                "raw_stderr_path": str(stderr_path),
                "container_executable": container_executable,
            },
            container_executable=container_executable,
        )
        if pip_runtime_dir is not None and pip_shared_history_dir is not None:
            merge_pip_predictions_into_shared_history(
                shared_history_root=pip_shared_history_dir,
                attempt_prediction_root=pip_runtime_dir,
            )
            if pip_family_history_dir is not None:
                merge_pip_predictions_into_shared_history(
                    shared_history_root=pip_family_history_dir,
                    attempt_prediction_root=pip_runtime_dir,
                )
        if (
            python_script_runtime_dir is not None
            and python_script_shared_history_dir is not None
        ):
            merge_python_script_predictions_into_shared_history(
                shared_history_root=python_script_shared_history_dir,
                attempt_prediction_root=python_script_runtime_dir,
            )
            if python_script_family_history_dir is not None:
                merge_python_script_predictions_into_shared_history(
                    shared_history_root=python_script_family_history_dir,
                    attempt_prediction_root=python_script_runtime_dir,
                )
        if pytest_runtime_dir is not None and pytest_shared_history_dir is not None:
            merge_pytest_predictions_into_shared_history(
                shared_history_root=pytest_shared_history_dir,
                attempt_prediction_root=pytest_runtime_dir,
            )
            if pytest_family_history_dir is not None:
                merge_pytest_predictions_into_shared_history(
                    shared_history_root=pytest_family_history_dir,
                    attempt_prediction_root=pytest_runtime_dir,
                )
        runtime_proof = {
            **asdict(proof),
            **runtime.runtime_proof,
        }
        _normalize_openclaw_trace(
            src=runtime.trace_path,
            dst=ctx.attempt_dir / "trace.jsonl",
            benchmark=benchmark,
            model=model,
            api_base=api_base,
            max_iterations=max_iterations,
            instance_id=ctx.instance_id,
            mcp_config_label=_mcp_config_label(mcp_config),
            prompt_template=ctx.prompt_template,
            agent_runtime_mode=ctx.agent_runtime_mode,
            runtime_proof=runtime_proof,
            record_internals=record_internals,
            generation_config=generation_config,
            capture_pytest_scripts=capture_pytest_scripts,
            capture_pytest_runtime=capture_pytest_runtime,
            capture_pip_runtime=capture_pip_runtime,
            capture_python_script_runtime=capture_python_script_runtime,
        )
    finally:
        if _stats_sampler is not None:
            ctx.samples = _stats_sampler.stop()
        if vtune or ksys_tool:
            from trace_collect.vtune_report import finalize_vtune

            finalize_vtune(
                vtune_out_dir,
                ctx.samples or [],
            )
        if tool_profiler_mode:
            _finalize_tool_profiler(
                tool_profiler_out_dir,
                ctx.attempt_dir,
            )
        if tool_scheduler_mode:
            _finalize_tool_scheduler(
                tool_scheduler_out_dir,
                ctx.attempt_dir,
            )
        container_logs = stop_task_container(
            container_id,
            executable=container_executable,
        )
        ctx.container_stdout = "\n".join(
            part
            for part in [
                stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "",
                stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "",
                container_logs,
            ]
            if part
        )
    assert runtime is not None
    assert runtime_proof is not None
    return AttemptResult(
        success=runtime.success,
        exit_status=runtime.exit_status,
        trace_path=ctx.attempt_dir / "trace.jsonl",
        model_patch=runtime.model_patch,
        n_iterations=runtime.n_iterations,
        total_llm_ms=runtime.total_llm_ms,
        total_tool_ms=runtime.total_tool_ms,
        total_tokens=runtime.total_tokens,
        error=runtime.error,
        runtime_proof=runtime_proof,
    )
