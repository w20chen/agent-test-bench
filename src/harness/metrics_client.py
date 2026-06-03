"""Stable client wrapper for vLLM scheduler metrics access (Phase 2).

Wraps `harness.scheduler_hooks.get_snapshot()` so the simulator has a
single object to instantiate per run and call `get_snapshot()` on per
iteration. When `metrics_url` is None the client returns a fresh
all-None snapshot (the explicit opt-out path); HTTP errors propagate
as exceptions per CLAUDE.md "no silent fallbacks for real operations".
"""

from __future__ import annotations

import logging
import time

from harness.metrics import sample_nvidia_smi_per_pid
from harness.scheduler_hooks import (
    GpuBaseline,
    GpuMemoryBreakdown,
    PreemptionSnapshot,
    empty_snapshot,
    get_snapshot,
)


class GpuPidNotFoundError(RuntimeError):
    """Raised when GPU tracking is enabled but the vLLM PID is not on any GPU."""


class VLLMMetricsClient:
    """Stateful client for repeated vLLM scheduler metric fetches.

    When `gpu_baseline` and `vllm_pid` are both provided, each
    `get_snapshot()` call also samples GPU memory via nvidia-smi and
    derives a `GpuMemoryBreakdown` attached to the returned snapshot.

    Activations are computed as a residual:
        activations_mib = total_pid_mib - weights_mib - kv_cache_used_mib
    Clamped to 0 when negative (with a warning) — protects against
    nvidia-smi quantization noise + KV cache fragmentation but never
    fabricates values.
    """

    def __init__(
        self,
        metrics_url: str | None,
        *,
        timeout_s: float = 5.0,
        gpu_baseline: GpuBaseline | None = None,
        vllm_pid: int | None = None,
    ) -> None:
        self.metrics_url = metrics_url
        self.timeout_s = timeout_s
        self.gpu_baseline = gpu_baseline
        self.vllm_pid = vllm_pid

    @property
    def is_enabled(self) -> bool:
        return bool(self.metrics_url)

    @property
    def is_gpu_tracking_enabled(self) -> bool:
        return bool(self.metrics_url and self.gpu_baseline and self.vllm_pid)

    def get_snapshot(self) -> PreemptionSnapshot:
        snapshot = get_snapshot(self.metrics_url, timeout_s=self.timeout_s)
        if not self.is_gpu_tracking_enabled:
            return snapshot
        breakdown = self._sample_gpu_memory(snapshot.gpu_cache_usage_perc)
        # PreemptionSnapshot uses slots so we reconstruct rather than mutate.
        return PreemptionSnapshot(
            num_preemptions_total=snapshot.num_preemptions_total,
            gpu_cache_usage_perc=snapshot.gpu_cache_usage_perc,
            cpu_cache_usage_perc=snapshot.cpu_cache_usage_perc,
            gpu_prefix_cache_hit_rate=snapshot.gpu_prefix_cache_hit_rate,
            cpu_prefix_cache_hit_rate=snapshot.cpu_prefix_cache_hit_rate,
            gpu_memory_breakdown=breakdown,
        )

    def _sample_gpu_memory(
        self, kv_cache_usage_perc: float | None
    ) -> GpuMemoryBreakdown:
        assert self.gpu_baseline is not None and self.vllm_pid is not None
        row = sample_nvidia_smi_per_pid(self.vllm_pid)
        if row is None:
            raise GpuPidNotFoundError(
                f"vLLM PID {self.vllm_pid} not found in nvidia-smi compute apps; "
                "GPU tracking enabled but PID is not visible (vLLM crashed? wrong PID?)"
            )
        weights_mib = self.gpu_baseline.weights_mib
        kv_total_mib = self.gpu_baseline.kv_cache_total_mib
        kv_used_mib: float | None = None
        if kv_cache_usage_perc is not None:
            # vLLM names this metric "perc" but some versions expose a fraction
            # in [0, 1] while others expose a percentage in [0, 100]. Bridge
            # both conventions: if value > 1 treat as percent, else as fraction.
            frac = kv_cache_usage_perc / 100.0 if kv_cache_usage_perc > 1.0 else kv_cache_usage_perc
            kv_used_mib = max(0.0, frac * kv_total_mib)
        total_mib = float(row["memory_used_mib"])
        if kv_used_mib is not None:
            activations = total_mib - weights_mib - kv_used_mib
            if activations < 0:
                logging.warning(
                    "GpuMemoryBreakdown residual activations negative (%.1f MiB); clamping to 0. "
                    "total=%.1f weights=%.1f kv_used=%.1f",
                    activations, total_mib, weights_mib, kv_used_mib,
                )
                activations = 0.0
        else:
            activations = None
        return GpuMemoryBreakdown(
            gpu_index=int(row["gpu_index"]),
            pid=int(row["pid"]),
            total_pid_mib=total_mib,
            weights_mib=weights_mib,
            kv_cache_used_mib=kv_used_mib,
            kv_cache_total_mib=kv_total_mib,
            activations_mib=activations,
            ts=time.time(),
        )


__all__ = [
    "GpuPidNotFoundError",
    "VLLMMetricsClient",
    "empty_snapshot",
    "get_snapshot",
]
