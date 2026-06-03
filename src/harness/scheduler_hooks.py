from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from harness.prometheus import parse_prometheus_metric_values

@dataclass(slots=True)
class GpuMemoryBreakdown:

    gpu_index: int
    pid: int
    total_pid_mib: float
    weights_mib: float | None
    kv_cache_used_mib: float | None
    kv_cache_total_mib: float | None
    activations_mib: float | None
    ts: float


@dataclass(slots=True)
class GpuBaseline:

    weights_mib: float
    kv_cache_total_mib: float
    model: str
    dtype: str
    tensor_parallel_size: int


@dataclass(slots=True)
class GpuComponentBreakdown:

    step_index: int
    attn_mib: float
    mlp_mib: float
    other_activations_mib: float
    per_module: list[dict]
    measurement_kind: str


@dataclass(slots=True)
class PreemptionSnapshot:

    num_preemptions_total: float | None
    gpu_cache_usage_perc: float | None
    cpu_cache_usage_perc: float | None
    gpu_prefix_cache_hit_rate: float | None
    cpu_prefix_cache_hit_rate: float | None
    gpu_memory_breakdown: GpuMemoryBreakdown | None = None

@dataclass(slots=True)
class EvictionEvent:

    seq_id: str
    tokens: int
    reason: str
    gpu_usage: float

@dataclass(slots=True)
class SchedulerHookStatus:

    runtime_hook_enabled: bool
    vllm_version: str | None
    target: str
    reason: str | None = None

EVICTION_PATTERN = re.compile(
    r"EVICT seq_id=(?P<seq_id>\S+) tokens=(?P<tokens>\d+) reason=(?P<reason>\S+) gpu_usage=(?P<gpu_usage>[0-9.]+)"
)

def parse_prometheus_metrics(metrics_payload: str) -> PreemptionSnapshot:
    # vLLM 0.10+ renamed `gpu_cache_usage_perc` → `kv_cache_usage_perc` and
    # dropped `gpu_prefix_cache_hit_rate`. Both old and new gauge names map
    # to the same alias so either vLLM version populates it.
    wanted = {
        "vllm:num_preemptions_total": "num_preemptions_total",
        "vllm:gpu_cache_usage_perc": "gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc": "gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc": "cpu_cache_usage_perc",
        "vllm:gpu_prefix_cache_hit_rate": "gpu_prefix_cache_hit_rate",
        "vllm:cpu_prefix_cache_hit_rate": "cpu_prefix_cache_hit_rate",
    }
    values = parse_prometheus_metric_values(
        metrics_payload,
        wanted,
        include_missing=True,
    )
    return PreemptionSnapshot(**values, gpu_memory_breakdown=None)

def empty_snapshot() -> PreemptionSnapshot:
    """Return a PreemptionSnapshot with all fields set to None.

    Used by `get_snapshot` and the simulator when no `metrics_url` is
    provided (explicit opt-out: simulate runs without vLLM metrics
    integration). Each call returns a fresh instance so callers cannot
    accidentally share state through a singleton.
    """
    return PreemptionSnapshot(
        num_preemptions_total=None,
        gpu_cache_usage_perc=None,
        cpu_cache_usage_perc=None,
        gpu_prefix_cache_hit_rate=None,
        cpu_prefix_cache_hit_rate=None,
        gpu_memory_breakdown=None,
    )

def get_snapshot(
    metrics_url: str | None = None,
    *,
    timeout_s: float = 5.0,
) -> PreemptionSnapshot:
    """Fetch a single PreemptionSnapshot from a vLLM /metrics endpoint.

    Args:
        metrics_url: Prometheus endpoint URL. If None or empty, returns
            an `empty_snapshot()` (explicit opt-out — sim runs without
            metrics integration). This is the only "fallback" path; HTTP
            failure when a URL IS set raises explicitly per CLAUDE.md
            "no silent fallbacks for real operations".
        timeout_s: HTTP timeout in seconds. Defaults to 5s.

    Returns:
        PreemptionSnapshot with all five fields populated from the
        Prometheus payload (or all-None if `metrics_url` is unset).

    Raises:
        httpx.HTTPError: if `metrics_url` is set but the fetch fails.
        ValueError: if the payload is malformed.
    """
    if not metrics_url:
        return empty_snapshot()
    response = httpx.get(metrics_url, timeout=timeout_s)
    response.raise_for_status()
    return parse_prometheus_metrics(response.text)

def parse_eviction_events(log_text: str) -> list[EvictionEvent]:
    events: list[EvictionEvent] = []
    for line in log_text.splitlines():
        match = EVICTION_PATTERN.search(line)
        if not match:
            continue
        events.append(
            EvictionEvent(
                seq_id=match.group("seq_id"),
                tokens=int(match.group("tokens")),
                reason=match.group("reason"),
                gpu_usage=float(match.group("gpu_usage")),
            )
        )
    return events

def scheduler_log_snippet() -> str:
    return (
        'logger.info(f"EVICT seq_id={seq.seq_id} tokens={seq.get_len()} '
        'reason={reason} gpu_usage={self.block_manager.gpu_utilization}")'
    )

def _safe_vllm_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None

def apply_scheduler_hook() -> SchedulerHookStatus:
    version = _safe_vllm_version()
    target = "vllm.v1.core.sched.scheduler.Scheduler.schedule"
    if version is None:
        raise RuntimeError("vllm is not installed")
    if not version.startswith("0.8."):
        raise RuntimeError(f"Unsupported vllm version for scheduler hook: {version}")

    scheduler_module = importlib.import_module("vllm.v1.core.sched.scheduler")
    scheduler_cls = getattr(scheduler_module, "Scheduler", None)
    if scheduler_cls is None or not hasattr(scheduler_cls, "schedule"):
        raise RuntimeError(f"Scheduler target not found: {target}")
    if getattr(scheduler_cls.schedule, "_agent_benchmark_hooked", False):
        return SchedulerHookStatus(
            runtime_hook_enabled=True,
            vllm_version=version,
            target=target,
        )

    original_schedule = scheduler_cls.schedule
    logger = getattr(scheduler_module, "logger")

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        before_preempted = {
            req_id
            for req_id, req in getattr(self, "requests", {}).items()
            if getattr(getattr(req, "status", None), "name", "") == "PREEMPTED"
        }
        result = original_schedule(self, *args, **kwargs)
        for req_id, req in getattr(self, "requests", {}).items():
            status_name = getattr(getattr(req, "status", None), "name", "")
            if status_name == "PREEMPTED" and req_id not in before_preempted:
                logger.info(
                    "EVICT seq_id=%s tokens=%s reason=preempted gpu_usage=%s",
                    req_id,
                    int(getattr(req, "num_tokens_with_spec", 0) or 0),
                    float(
                        getattr(getattr(self, "kv_cache_manager", None), "usage", 0.0)
                        or 0.0
                    ),
                )
        return result

    wrapped._agent_benchmark_hooked = True  # type: ignore[attr-defined]
    scheduler_cls.schedule = wrapped  # type: ignore[assignment]
    return SchedulerHookStatus(
        runtime_hook_enabled=True,
        vllm_version=version,
        target=target,
    )

def build_report(metrics_payload: str, log_text: str) -> dict[str, Any]:
    snapshot = parse_prometheus_metrics(metrics_payload)
    events = parse_eviction_events(log_text)
    return {
        "metrics_fetch_ok": True,
        "baseline_provided": False,
        "preemption_counter_delta": None,
        "scheduler_log_provided": bool(log_text),
        "scheduler_hook_runtime_confirmed": bool(events),
        "evidence_scope": "current_run_log" if log_text else "cumulative_metrics_only",
        "scheduler_hook_status": None,
        "preemption_snapshot": asdict(snapshot),
        "eviction_events": [asdict(event) for event in events],
        "instrumentation_snippet": scheduler_log_snippet(),
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and parse vLLM preemption metrics plus scheduler-hook logs."
    )
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--baseline-preemptions-total", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--write-hook-status")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.write_hook_status:
        status = apply_scheduler_hook()
        Path(args.write_hook_status).write_text(
            json.dumps(asdict(status), indent=2) + "\n",
            encoding="utf-8",
        )
        return
    response = httpx.get(args.metrics_url, timeout=10.0)
    response.raise_for_status()
    metrics_payload = response.text
    log_text = Path(args.log_file).read_text(encoding="utf-8") if args.log_file else ""
    report = build_report(metrics_payload=metrics_payload, log_text=log_text)
    report["baseline_provided"] = args.baseline_preemptions_total is not None
    current_total = report["preemption_snapshot"]["num_preemptions_total"]
    if args.baseline_preemptions_total is not None and current_total is not None:
        report["preemption_counter_delta"] = (
            current_total - args.baseline_preemptions_total
        )
        report["evidence_scope"] = "baseline_delta"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
