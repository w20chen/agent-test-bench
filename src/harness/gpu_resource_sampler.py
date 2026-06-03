"""Background GPU memory sampler for local-simulate vLLM runs.

Polls (a) `vllm:gpu_cache_usage_perc` from the vLLM Prometheus endpoint
and (b) per-PID GPU memory via nvidia-smi at a configurable rate, then
combines them with the one-shot `GpuBaseline` to emit
`GpuMemoryBreakdown` time series.

Lifecycle modeled on `ContainerStatsSampler`: explicit start() returns,
the run-loop runs in an asyncio task, stop() flushes to disk.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

from harness.metrics_client import GpuPidNotFoundError, VLLMMetricsClient
from harness.scheduler_hooks import GpuBaseline, GpuMemoryBreakdown

logger = logging.getLogger(__name__)


class GpuResourceSampler:
    """Background sampler producing a GPU-memory time series."""

    def __init__(
        self,
        *,
        metrics_url: str,
        gpu_baseline: GpuBaseline,
        vllm_pid: int,
        output_path: Path,
        sample_hz: float = 10.0,
    ) -> None:
        if sample_hz <= 0:
            raise ValueError(f"sample_hz must be positive, got {sample_hz!r}")
        self._client = VLLMMetricsClient(
            metrics_url=metrics_url,
            gpu_baseline=gpu_baseline,
            vllm_pid=vllm_pid,
        )
        self._gpu_baseline = gpu_baseline
        self._output_path = Path(output_path)
        self._period_s = 1.0 / float(sample_hz)
        self._samples: list[GpuMemoryBreakdown] = []
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._started_at: float | None = None
        self._stopped: bool = False

    async def start(self) -> None:
        """Start the sampling task. Fails fast if PID is not visible."""
        # Take one sample synchronously to fail fast on PID-not-found
        # (matches CLAUDE.md "no silent fallback" rule and gives the user
        # an immediate error rather than a silent zero-sample run).
        first = await asyncio.to_thread(self._sample_once)
        self._samples.append(first)
        self._started_at = time.time()
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the loop and flush to disk. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
        self._flush()

    @property
    def samples(self) -> list[GpuMemoryBreakdown]:
        return list(self._samples)

    def _sample_once(self) -> GpuMemoryBreakdown:
        snapshot = self._client.get_snapshot()
        if snapshot.gpu_memory_breakdown is None:
            # is_gpu_tracking_enabled was true (we constructed with all three),
            # so this should never happen. If it does, the contract is broken.
            raise RuntimeError(
                "GpuResourceSampler: client returned no gpu_memory_breakdown despite gpu tracking being enabled"
            )
        return snapshot.gpu_memory_breakdown

    async def _run_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                sample = await asyncio.to_thread(self._sample_once)
                self._samples.append(sample)
            except GpuPidNotFoundError:
                # vLLM crashed mid-run. Stop sampling and let stop() flush
                # what we have. Do NOT silently swallow — log it loudly.
                logger.error(
                    "GpuResourceSampler: vLLM PID disappeared mid-run; halting sampler"
                )
                break
            except Exception as exc:
                logger.warning("GpuResourceSampler sample failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._period_s)
            except asyncio.TimeoutError:
                continue

    def _flush(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        duration_s = (time.time() - self._started_at) if self._started_at else 0.0
        peak_total = max((s.total_pid_mib for s in self._samples), default=0.0)
        peak_act = max(
            (s.activations_mib for s in self._samples if s.activations_mib is not None),
            default=0.0,
        )
        payload: dict[str, Any] = {
            "gpu_baseline": dataclasses.asdict(self._gpu_baseline),
            "gpu_samples": [dataclasses.asdict(s) for s in self._samples],
            "summary": {
                "n_samples": len(self._samples),
                "duration_s": duration_s,
                "peak_total_pid_mib": peak_total,
                "peak_activations_mib": peak_act,
            },
        }
        self._output_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
