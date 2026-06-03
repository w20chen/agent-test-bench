"""Pub/sub bus for post-softmax attention tensors.

Step 5 introduces this thin dispatcher so `LayerCapturer._sampled_attention_rows`
can publish the post-softmax attention tensor exactly once per layer, before
any reduce/reshape, and let downstream consumers (H2O in step 6) observe the
same values without re-running `torch.softmax`.

Shape contract
--------------

Publishers MUST emit ``attn`` shaped ``(B, num_q_heads, n_query_rows, key_len)``
where ``num_q_heads`` is the number of *query* heads (not KV heads — GQA is
already expanded back to the per-query-head layout at this point) and
``n_query_rows`` is the number of sampled query positions for this forward
(``len(row_indices)`` in the capturer). The dtype matches the underlying
attention module (typically bf16 / fp16) — consumers must cast as needed.

``query_positions`` is a ``LongTensor`` of shape ``(n_query_rows,)`` holding
the absolute key-axis position of each sampled query in the *full* sequence
(prefill + decode so far). ``key_len`` is the size of the key axis at this
forward (post-eviction if a KV-policy cache shrunk it).

``phase`` is one of ``"prefill"`` or ``"decode"``, matching
``attention.npz.record_phase``.

When ``full_prefill=True``, the payload is a bounded chunk of an otherwise
full prefill scan and is delivered only to consumers that opt in with
``prefill_observe_mode="full"``. This lets H2O accumulate paper-faithful
prefill scores without forcing ``attention.npz`` to store every query row.

Suspend semantics
-----------------

`LayerCapturer.suspend_attention()` increments a suspend counter that the
capturer uses to skip its own reduce. The bus respects that gate by default
(``suspended=True`` -> consumers with ``always_active=False`` are skipped),
which keeps user-driven probe pauses honest. Cache-eviction subscribers that
must observe every step (H2O) opt out via ``always_active=True``.

The bus does no tensor work itself: each ``publish`` is a tight `for` loop
calling ``observe`` on each subscriber.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class AttentionConsumer(Protocol):
    """Protocol for downstream consumers of post-softmax attention.

    Implementations expose a class- or instance-level ``always_active`` flag.
    When ``False`` (default), `AttentionBus.publish` skips this consumer while
    `LayerCapturer.suspend_attention()` is active. H2O sets this to ``True``
    so its score accumulator never silently desyncs from the cache state.
    """

    always_active: bool
    prefill_observe_mode: str

    def observe(
        self,
        *,
        layer: int,
        attn: "torch.Tensor",
        query_positions: "torch.Tensor",
        key_len: int,
        phase: str,
    ) -> None:
        ...


class AttentionBus:
    """Synchronous pub/sub for post-softmax attention tensors.

    Pure dispatcher: holds a list of consumers and forwards `publish(...)`
    calls to each. No tensor reshaping, no clones, no async — keeping the
    softmax-once invariant cheap and easy to reason about.
    """

    def __init__(self) -> None:
        self._consumers: list[AttentionConsumer] = []

    def subscribe(self, consumer: AttentionConsumer) -> None:
        self._consumers.append(consumer)

    def unsubscribe(self, consumer: AttentionConsumer) -> None:
        self._consumers.remove(consumer)

    def n_consumers(self) -> int:
        return len(self._consumers)

    def has_full_prefill_consumers(self) -> bool:
        return any(
            str(getattr(consumer, "prefill_observe_mode", "sampled")) == "full"
            for consumer in self._consumers
        )

    def publish(
        self,
        *,
        layer: int,
        attn: "torch.Tensor",
        query_positions: "torch.Tensor",
        key_len: int,
        phase: str,
        suspended: bool,
        full_prefill: bool = False,
    ) -> None:
        if not self._consumers:
            return
        for consumer in self._consumers:
            if phase == "prefill":
                mode = str(getattr(consumer, "prefill_observe_mode", "sampled"))
                if full_prefill and mode != "full":
                    continue
                if not full_prefill and mode == "full":
                    continue
            if suspended and not getattr(consumer, "always_active", False):
                continue
            consumer.observe(
                layer=layer,
                attn=attn,
                query_positions=query_positions,
                key_len=key_len,
                phase=phase,
            )


__all__ = ["AttentionBus", "AttentionConsumer"]
