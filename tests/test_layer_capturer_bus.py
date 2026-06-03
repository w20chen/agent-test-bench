"""LayerCapturer ↔ AttentionBus integration (plan G16#5 + #6).

Two regression gates that step 5 must pass before step 6 (H2O) lands:

* **G16#5 — softmax-once invariant**: when the bus has subscribers, the
  capturer must NOT add a second `torch.softmax` call. H2O consumes the
  tensor we already computed; if a future change quietly re-softmaxes inside
  a consumer, this test fails.

* **G16#6 — `--kv-policy none` byte-equality**: with no bus, or a bus that
  has no subscribers, the capturer's reduce outputs (segment_mass / topk /
  heavy hitter) must be element-wise identical to the pre-step-5 path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from serving.recording import RecordingConfig
from serving.recording.attention_bus import AttentionBus
from serving.recording.hooks import LayerCapturer


# ---------------------------------------------------------------------------
# Toy model + cache fixtures (same shape as test_recording_e2e._Toy*).
# ---------------------------------------------------------------------------


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 4
        self.num_key_value_groups = 1
        self.scaling = 0.5
        self.q_proj = nn.Linear(8, 4, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
    ):
        del position_embeddings, attention_mask, past_key_values
        return hidden_states, None


class _FakeCache:
    def __init__(self, key_states: torch.Tensor) -> None:
        self.key_states = key_states

    def __getitem__(self, layer_idx: int):
        if layer_idx != 0:
            raise KeyError(layer_idx)
        return self.key_states, None


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = _ToyAttention()
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate = nn.Linear(8, 3, bias=False)


@dataclass
class _RecordingSubscriber:
    """Captures publish payloads but does NOT recompute softmax.

    Critical for G16#5: any consumer that re-softmaxed the input would
    silently invalidate the "softmax once" claim. The H2O subscriber in
    step 6 must follow this same no-recompute pattern.
    """

    always_active: bool = True
    prefill_observe_mode: str = "sampled"
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
                "attn_shape": tuple(attn.shape),
                "attn_dtype": attn.dtype,
                "query_positions": query_positions.tolist(),
                "key_len": key_len,
                "phase": phase,
            }
        )


def _segments() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]


def _make_capturer(*, attention_bus: AttentionBus | None) -> tuple[_ToyModel, LayerCapturer]:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2, max_prefill_queries=4),
        model_summary={"name": "toy", "num_experts_per_tok": 2},
        attention_bus=attention_bus,
    )
    return model, capturer


def _drive_one_attention_call(model: _ToyModel) -> None:
    """One prefill-shape attention forward pass; same kwargs as test_recording_e2e."""
    attn = model.model.layers[0].self_attn
    cos = torch.ones(1, 4, 4, dtype=torch.float32)
    sin = torch.zeros(1, 4, 4, dtype=torch.float32)
    attn(
        torch.zeros(1, 4, 8),
        position_embeddings=(cos, sin),
        past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
    )


# ---------------------------------------------------------------------------
# G16#5: softmax-once invariant.
# ---------------------------------------------------------------------------


class _SoftmaxCounter:
    """Drop-in replacement for torch.softmax that counts calls."""

    def __init__(self) -> None:
        self.count = 0
        self._real = torch.softmax

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._real(*args, **kwargs)


def test_softmax_called_once_per_forward_with_active_subscriber(monkeypatch) -> None:
    """G16#5: one forward -> one softmax call, even with a bus subscriber.

    The bus subscriber must consume the tensor we already softmaxed; this
    test fails if a future H2O implementation tries to re-softmax raw scores.
    """
    bus = AttentionBus()
    subscriber = _RecordingSubscriber(always_active=True)
    bus.subscribe(subscriber)

    model, capturer = _make_capturer(attention_bus=bus)

    counter = _SoftmaxCounter()
    monkeypatch.setattr(torch, "softmax", counter)

    capturer.start_attempt(_TmpDir())  # type: ignore[arg-type]
    with capturer.recording_session(
        call_idx=0,
        segments=_segments(),
        input_token_count=4,
    ):
        _drive_one_attention_call(model)

    assert counter.count == 1, (
        f"expected exactly one torch.softmax call, got {counter.count}; "
        "did a consumer re-softmax raw scores?"
    )
    # Subscriber actually received the tensor.
    assert len(subscriber.calls) == 1
    call = subscriber.calls[0]
    assert call["layer"] == 0
    assert call["phase"] == "prefill"
    # Shape contract: (B, num_q_heads, n_query_rows, key_len)
    assert call["attn_shape"][0] == 1  # batch
    assert call["attn_shape"][-1] == 4  # key_len
    assert call["query_positions"] == [0, 1, 2, 3]


def test_softmax_called_once_per_forward_without_subscribers(monkeypatch) -> None:
    """Bus exists but no subscribers: still exactly one softmax call."""
    bus = AttentionBus()
    model, capturer = _make_capturer(attention_bus=bus)

    counter = _SoftmaxCounter()
    monkeypatch.setattr(torch, "softmax", counter)

    capturer.start_attempt(_TmpDir())  # type: ignore[arg-type]
    with capturer.recording_session(
        call_idx=0,
        segments=_segments(),
        input_token_count=4,
    ):
        _drive_one_attention_call(model)

    assert counter.count == 1


# ---------------------------------------------------------------------------
# G16#6: bus-vs-no-bus produces byte-identical reduce outputs.
# ---------------------------------------------------------------------------


def _capture_records(*, attention_bus: AttentionBus | None) -> list[dict[str, Any]]:
    """Run one forward pass through the capturer and return its prefill records."""
    torch.manual_seed(1234)  # deterministic linear-projection weights
    model, capturer = _make_capturer(attention_bus=attention_bus)
    capturer.start_attempt(_TmpDir())  # type: ignore[arg-type]
    with capturer.recording_session(
        call_idx=0,
        segments=_segments(),
        input_token_count=4,
    ):
        _drive_one_attention_call(model)
    # Pull a snapshot of the records the capturer accumulated. We deepcopy
    # the numpy arrays so two runs can be compared without aliasing.
    return [
        {key: (val.copy() if isinstance(val, np.ndarray) else val) for key, val in rec.items()}
        for rec in capturer._prefill_records
    ]


def test_reduce_output_unchanged_when_bus_is_none() -> None:
    """G16#6: bus=None vs bus=empty AttentionBus -> identical reduce arrays."""
    no_bus = _capture_records(attention_bus=None)
    empty_bus = _capture_records(attention_bus=AttentionBus())

    assert len(no_bus) == len(empty_bus) == 1
    a, b = no_bus[0], empty_bus[0]
    assert a.keys() == b.keys()
    for key in (
        "segment_mass",
        "topk_indices",
        "topk_weights",
        "heavy_indices",
        "heavy_weights",
        "query_heads",
        "query_positions",
    ):
        np.testing.assert_array_equal(
            a[key], b[key], err_msg=f"field {key} diverged between None bus and empty bus"
        )
    assert a["layer"] == b["layer"]
    assert a["phase"] == b["phase"]
    assert a["decode_step"] == b["decode_step"]


def test_reduce_output_unchanged_with_passive_subscriber() -> None:
    """A subscriber that only observes (no recompute) must not perturb the
    capturer's own reduce — the H2O subscriber in step 6 follows this rule.
    """
    no_bus = _capture_records(attention_bus=None)

    bus = AttentionBus()
    bus.subscribe(_RecordingSubscriber(always_active=True))
    with_subscriber = _capture_records(attention_bus=bus)

    assert len(no_bus) == len(with_subscriber) == 1
    a, b = no_bus[0], with_subscriber[0]
    for key in (
        "segment_mass",
        "topk_indices",
        "topk_weights",
        "heavy_indices",
        "heavy_weights",
    ):
        np.testing.assert_array_equal(
            a[key], b[key], err_msg=f"field {key} diverged when adding a passive subscriber"
        )


def test_full_prefill_consumer_gets_chunked_rows_without_uncapping_recording() -> None:
    """Full-prefill H2O consumers get all rows in chunks; attention.npz stays sampled."""
    bus = AttentionBus()
    subscriber = _RecordingSubscriber(always_active=True, prefill_observe_mode="full")
    bus.subscribe(subscriber)
    model, capturer = _make_capturer(attention_bus=bus)

    capturer.start_attempt(_TmpDir())  # type: ignore[arg-type]
    with capturer.recording_session(
        call_idx=0,
        segments=[
            {
                "role": "user",
                "message_index": 0,
                "token_start": 0,
                "token_end": 8,
                "has_content": True,
                "has_tool_calls": False,
            }
        ],
        input_token_count=8,
    ):
        attn = model.model.layers[0].self_attn
        cos = torch.ones(1, 8, 4, dtype=torch.float32)
        sin = torch.zeros(1, 8, 4, dtype=torch.float32)
        attn(
            torch.zeros(1, 8, 8),
            position_embeddings=(cos, sin),
            past_key_values=_FakeCache(torch.zeros(1, 1, 8, 4)),
        )

    assert [call["query_positions"] for call in subscriber.calls] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert capturer._prefill_records[0]["query_positions"].tolist() == [0, 3, 4, 7]


# ---------------------------------------------------------------------------
# Helpers — minimal tmp dir stand-in for `start_attempt`. We never write to it
# in these tests (we don't call flush()), but `start_attempt` requires a path.
# ---------------------------------------------------------------------------


class _TmpDir:
    """Path-like that's just barely good enough for `start_attempt(...).mkdir`."""

    def __init__(self) -> None:
        import tempfile
        from pathlib import Path

        self._path = Path(tempfile.mkdtemp(prefix="layer_capturer_bus_"))

    def __fspath__(self) -> str:
        return str(self._path)

    def __str__(self) -> str:
        return str(self._path)
