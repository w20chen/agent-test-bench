"""Random KV eviction policy.

Uniform-random eviction over the *entire* current key range. Used as the
control baseline against which StreamingLLM (step 4) and H2O (step 6) are
compared in the eviction-policy ablation.

Design choices (locked here so reviewers can audit later policies against the
same baseline contract):

(a) **No "always-keep newest token" exception.** Pure random — every position
    in `[0, key_len)` is equally evictable. The H2O paper's random ablation
    follows this convention; carving out the most recent slot would silently
    bias the baseline toward streaming-style locality and inflate its quality.

(b) **Per-layer independent sampling.** Each `_decide_evict(layer_idx, ...)`
    call draws fresh indices from the same per-instance RNG. Layers do *not*
    share an evict mask. This matches H2O's per-layer eviction semantics in
    step 6 (every layer sees its own attention scores and evicts independently)
    so policies remain comparable along the same axis.

Determinism contract: two `RandomEvictCache` instances with the same
`config.seed` will, when fed the same sequence of `_decide_evict` calls,
produce byte-identical evict sets. Achieved with a per-instance `torch.Generator`
on CPU; we deliberately avoid the global `torch.manual_seed()` because the
recording session also draws from the global RNG (sampled-attention rows) and
seeding it would either collide or perturb upstream sampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)

if TYPE_CHECKING:
    from serving.kv_policies.recorder import KVEvictionRecorder


class RandomEvictCache(BaseEvictionCache):
    """`BaseEvictionCache` with uniform-random over-budget eviction."""

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
    ) -> None:
        super().__init__(config, num_layers, recorder=recorder)
        if config.budget is None or config.budget <= 0:
            raise ValueError(
                f"RandomEvictCache requires positive config.budget; got {config.budget!r}"
            )
        # Per-instance RNG so two instances with the same seed are bit-identical
        # (test G16#3) and the global torch RNG used by attention sampling is
        # not perturbed.
        self._generator = torch.Generator(device="cpu").manual_seed(int(config.seed))

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        budget = int(self.config.budget)  # type: ignore[arg-type]
        if key_len <= budget:
            return EvictionDecision(
                keep_indices=list(range(key_len)),
                evict_indices=[],
                reason="none",
            )
        n_evict = key_len - budget
        # randperm on CPU keeps the device-agnostic determinism contract; the
        # selected indices are then applied via index_select on the K/V device
        # in BaseEvictionCache._physically_drop.
        perm = torch.randperm(key_len, generator=self._generator).tolist()
        evict = sorted(perm[:n_evict])
        keep = sorted(perm[n_evict:])
        return EvictionDecision(
            keep_indices=keep,
            evict_indices=evict,
            reason="over_budget",
        )
