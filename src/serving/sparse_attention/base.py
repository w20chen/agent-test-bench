"""Scaffolding for sparse attention methods.

`BaseSparseAttention` is a Protocol: implementations build a 4D additive
attention mask consumed by an HF pre-forward hook on each `self_attn` module.
Unlike `kv_policies/`, sparsity does NOT alter K/V cache contents — it
constrains which key positions each query may attend to within an otherwise
unmodified cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SparseAttentionConfig:
    """User-facing config for a sparse attention method.

    Method-specific knobs sit alongside the shared ones (`name`, `record`).
    `sliding` / `streaming` consume only `sink_size` / `recent_window`.
    Dynamic methods use `budget` plus optional block/page knobs. Frozen to match
    `EvictionPolicyConfig`'s contract — providers stash a single config and
    pass it into the factory.
    """

    name: Literal["none", "sliding", "streaming", "heavy_hitter", "block_topk", "quest"]
    record: bool = False
    # Default sink_size=4 follows StreamingLLM (Xiao et al. 2024,
    # arXiv:2309.17453): the first few tokens carry the "attention sink"
    # that prevents softmax-distribution collapse when the middle is masked.
    sink_size: int = 4  # sliding
    # 256 is a reasonable default for short-context smoke tests; tune per
    # model and workload. Not anchored to any specific paper.
    recent_window: int = 256  # sliding
    observe_only: bool = False  # run full attention; record what sparse WOULD select
    budget: int | None = None  # dynamic methods
    block_size: int = 16  # block_topk / quest page size
    score_reduction: Literal["max", "mean"] = "max"  # block/page score reduction
    phase_scope: Literal["decode_only"] = "decode_only"

    def __post_init__(self) -> None:
        if self.observe_only and not self.record:
            raise ValueError(
                "observe_only=True requires record=True; otherwise the run "
                "is a no-op with no recording (every layer's pre-hook would "
                "build the would-be mask and immediately discard it)."
            )


@dataclass(frozen=True)
class SparseAttentionContext:
    """Forward state available to query-aware sparse attention methods."""

    module: Any
    hidden_states: Any
    position_embeddings: tuple[Any, Any] | None
    past_key_values: Any
    attention_mask: Any


@runtime_checkable
class BaseSparseAttention(Protocol):
    """Protocol every sparse attention method must satisfy.

    The HF pre-forward hook calls `build_additive_mask` per layer per forward,
    adds the returned tensor to any existing `attention_mask` kwarg, and then
    (when recording) appends one row of `record_metadata` to the recorder.

    `build_additive_mask` returns a tensor broadcastable to `[B, H, Q, K]`
    where 0 means "attend" and `-inf` means "mask". The recording pre-hook
    always passes a non-None attention mask to HF attention; methods must
    therefore include the causal upper-triangular cut themselves for Q > 1
    prefill masks.

    `requires_full_prefill` is True when the method's correctness depends on
    every prefill token's attention landing in the AttentionBus (e.g.,
    heavy_hitter accumulates per-key historical scores). When True, the HF
    backend MUST NOT enable session KV cache delta-prefill — every chat()
    call must re-prefill the full prompt so the bus sees every key position.
    Stateless / decode-recomputed methods (sliding, block_topk, quest) set
    this False and benefit from cross-call KV reuse.
    """

    name: str
    observe_only: bool
    requires_full_prefill: bool

    def build_additive_mask(
        self,
        *,
        layer_idx: int,
        query_len: int,
        key_len: int,
        phase: str,
        decode_step: int = -1,
        device: "torch.device",
        dtype: "torch.dtype",
        context: SparseAttentionContext | None = None,
    ) -> "torch.Tensor | None":
        """Returns the additive mask, or `None` when `self.observe_only=True` to skip [1,1,Q,K] tensor allocation. The pre-hook must not write back when None is returned."""

    def record_metadata(
        self,
        *,
        layer_idx: int,
        phase: str,
        decode_step: int,
    ) -> dict[str, Any]:
        """Per-call metadata recorded alongside the (layer, phase, step) row."""

    def reset_state(self) -> None:
        """Optional lifecycle hook called at attempt/call boundaries."""
