"""Pub/sub semantics for `AttentionBus` (plan step 5).

Bus is a pure dispatcher; these tests pin the two contract bits the H2O
subscriber (step 6) is going to rely on:

  1. publish() forwards the *exact same* keyword args to every subscriber.
  2. suspend gating: when LayerCapturer is suspended, only `always_active`
     consumers see the publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from serving.recording.attention_bus import AttentionBus, AttentionConsumer


@dataclass
class _RecordingConsumer:
    """Test stub that captures every observe() call as kwargs dict."""

    always_active: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    def observe(
        self,
        *,
        layer: int,
        attn: torch.Tensor,
        query_positions: torch.Tensor,
        key_len: int,
        phase: str,
    ) -> None:
        self.calls.append(
            {
                "layer": layer,
                "attn": attn,
                "query_positions": query_positions,
                "key_len": key_len,
                "phase": phase,
            }
        )


def test_recording_consumer_satisfies_protocol() -> None:
    # runtime_checkable Protocol: catches accidental drift between the
    # consumer contract and the H2O implementation in step 6.
    consumer = _RecordingConsumer()
    assert isinstance(consumer, AttentionConsumer)


def test_publish_dispatches_to_subscribers() -> None:
    bus = AttentionBus()
    a = _RecordingConsumer(always_active=False)
    b = _RecordingConsumer(always_active=True)
    bus.subscribe(a)
    bus.subscribe(b)
    assert bus.n_consumers() == 2

    attn = torch.zeros(1, 4, 2, 8)
    query_positions = torch.tensor([0, 1], dtype=torch.long)

    bus.publish(
        layer=3,
        attn=attn,
        query_positions=query_positions,
        key_len=8,
        phase="prefill",
        suspended=False,
    )

    for consumer in (a, b):
        assert len(consumer.calls) == 1
        call = consumer.calls[0]
        assert call["layer"] == 3
        # Same tensor identity — bus must not clone or detach.
        assert call["attn"] is attn
        assert call["query_positions"] is query_positions
        assert call["key_len"] == 8
        assert call["phase"] == "prefill"


def test_publish_skips_suspended_unless_always_active() -> None:
    bus = AttentionBus()
    passive = _RecordingConsumer(always_active=False)
    sticky = _RecordingConsumer(always_active=True)
    bus.subscribe(passive)
    bus.subscribe(sticky)

    bus.publish(
        layer=0,
        attn=torch.zeros(1, 1, 1, 4),
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=4,
        phase="decode",
        suspended=True,
    )

    assert passive.calls == []  # gated out
    assert len(sticky.calls) == 1  # opted in via always_active

    # Sanity: when not suspended, both fire again.
    bus.publish(
        layer=1,
        attn=torch.zeros(1, 1, 1, 4),
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=4,
        phase="decode",
        suspended=False,
    )
    assert len(passive.calls) == 1
    assert len(sticky.calls) == 2


def test_publish_with_no_subscribers_is_noop() -> None:
    # The `--kv-policy none` regression gate: zero subscribers must mean
    # zero overhead and zero observable side effects.
    bus = AttentionBus()
    assert bus.n_consumers() == 0
    bus.publish(
        layer=0,
        attn=torch.zeros(1, 1, 1, 1),
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=1,
        phase="prefill",
        suspended=False,
    )


def test_unsubscribe_removes_consumer() -> None:
    bus = AttentionBus()
    a = _RecordingConsumer()
    b = _RecordingConsumer()
    bus.subscribe(a)
    bus.subscribe(b)
    bus.unsubscribe(a)
    assert bus.n_consumers() == 1
    bus.publish(
        layer=0,
        attn=torch.zeros(1, 1, 1, 1),
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=1,
        phase="prefill",
        suspended=False,
    )
    assert a.calls == []
    assert len(b.calls) == 1
