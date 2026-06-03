"""Module-level memory profiler for deep-profile mode.

Identifies attention and MLP submodules by class-name pattern matching,
attaches forward hooks, and emits per-step `GpuComponentBreakdown`
records.

The actual hook callback (5-8 lines) is filled in by the researcher
because the choice of measurement (allocated vs reserved, pre-vs-post
delta, peak vs steady) is a research-taste decision; the scaffold
around it (module identification, TP guard, lifecycle, output format)
is provided.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable

import torch
from torch import nn

from harness.scheduler_hooks import GpuComponentBreakdown


_ATTN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"attention", r"attn"]]
_MLP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"mlp", r"feedforward", r"ffn"]]

ATTN_KIND = "attn"
MLP_KIND = "mlp"


def _classify(module: nn.Module) -> str | None:
    name = type(module).__name__
    if any(p.search(name) for p in _ATTN_PATTERNS):
        return ATTN_KIND
    if any(p.search(name) for p in _MLP_PATTERNS):
        return MLP_KIND
    return None


# ---------- USER CONTRIBUTION POINT ----------
# The researcher fills this in. Trade-offs documented in plan §User
# Contribution Point. Default placeholder uses memory_allocated() pre/post
# delta as a starting point; it is INTENTIONALLY simple — replace with
# the policy your experiment requires.
def default_memory_measurement(
    module: nn.Module,
    inputs: tuple[Any, ...],
    output: Any,
    pre_alloc_bytes: int,
) -> dict[str, float | str]:
    """Return a dict with 'value_mib' and 'measurement_kind'.

    # TODO(user): replace with your preferred measurement strategy:
    #   - torch.cuda.memory_allocated() delta (default below) — actual activations
    #   - torch.cuda.memory_reserved() delta — caching-allocator view (more like nvidia-smi)
    #   - peak-tracking via torch.cuda.reset_peak_memory_stats() + torch.cuda.max_memory_allocated()
    # Keep returned dict shape stable so the rest of the profiler keeps working.
    """
    if not torch.cuda.is_available():
        return {"value_mib": 0.0, "measurement_kind": "cpu_dummy"}
    post = torch.cuda.memory_allocated()
    delta_mib = max(0.0, (post - pre_alloc_bytes) / (1024.0 * 1024.0))
    return {"value_mib": delta_mib, "measurement_kind": "memory_allocated_delta"}
# ---------------------------------------------


class ComponentMemoryProfiler:
    """Holds forward hooks on attn/mlp modules and accumulates per-step records."""

    def __init__(
        self,
        module_records: list[tuple[str, str, nn.Module]],
        measurement_fn: Callable[..., dict[str, float | str]] = default_memory_measurement,
    ) -> None:
        # module_records: list of (full_path, kind, module)
        self._modules = module_records
        self._measurement_fn = measurement_fn
        self._handles: list[Any] = []
        self._current_step: list[dict[str, Any]] = []
        self._steps: list[GpuComponentBreakdown] = []
        self._pre_alloc: dict[int, int] = {}
        self._step_index = 0

    @property
    def steps(self) -> list[GpuComponentBreakdown]:
        return list(self._steps)

    def attach(self) -> None:
        for path, kind, module in self._modules:
            self._handles.append(
                module.register_forward_pre_hook(self._make_pre_hook(id(module)))
            )
            self._handles.append(
                module.register_forward_hook(self._make_post_hook(path, kind, module))
            )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def record_step(self) -> None:
        """Aggregate the per-module records into a step-level breakdown.
        Call after each generate() step."""
        attn_total = sum(r["value_mib"] for r in self._current_step if r["kind"] == ATTN_KIND)
        mlp_total = sum(r["value_mib"] for r in self._current_step if r["kind"] == MLP_KIND)
        kind_set = {r["measurement_kind"] for r in self._current_step}
        kind = next(iter(kind_set)) if len(kind_set) == 1 else "mixed"
        self._steps.append(
            GpuComponentBreakdown(
                step_index=self._step_index,
                attn_mib=float(attn_total),
                mlp_mib=float(mlp_total),
                other_activations_mib=0.0,  # default profiler doesn't measure non-attn/mlp
                per_module=[
                    {
                        "module_path": r["path"],
                        "module_class": r["module_class"],
                        "kind": r["kind"],
                        "value_mib": r["value_mib"],
                    }
                    for r in self._current_step
                ],
                measurement_kind=str(kind),
            )
        )
        self._current_step = []
        self._step_index += 1

    def _make_pre_hook(self, mod_id: int) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            self._pre_alloc[mod_id] = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
        return hook

    def _make_post_hook(self, path: str, kind: str, module: nn.Module) -> Callable[..., Any]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            pre = self._pre_alloc.get(id(module), 0)
            measurement = self._measurement_fn(module, inputs, output, pre)
            self._current_step.append({
                "path": path,
                "module_class": type(module).__name__,
                "kind": kind,
                "value_mib": float(measurement["value_mib"]),
                "measurement_kind": str(measurement["measurement_kind"]),
            })
            return output
        return hook


def attach_component_hooks(
    model: nn.Module,
    *,
    measurement_fn: Callable[..., dict[str, float | str]] | None = None,
) -> ComponentMemoryProfiler:
    """Walk the module tree, find attn/mlp submodules, attach forward hooks."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() > 1:
            raise RuntimeError(
                f"deep-profile mode supports tensor_parallel_size=1 only "
                f"(got world_size={torch.distributed.get_world_size()}); see plan §Out of Scope"
            )
    records: list[tuple[str, str, nn.Module]] = []
    for path, module in model.named_modules():
        kind = _classify(module)
        if kind:
            records.append((path, kind, module))
    if not records:
        logging.warning(
            "attach_component_hooks: no attn/mlp submodules detected in model %s; "
            "profile output will be empty. Check the class-name patterns.",
            type(model).__name__,
        )
    profiler = ComponentMemoryProfiler(
        records,
        measurement_fn=measurement_fn or default_memory_measurement,
    )
    profiler.attach()
    return profiler
