"""Observe-only sparse attention: pre-hook MUST NOT modify attention_mask.

When `SparseAttentionConfig.observe_only=True`, the pre-hook still computes
the would-be sparse mask and still records `kept_count` / `density` /
`extras_json` rows, but the attention computation itself runs as full
attention. This lets us collect "what would sparse have selected" traces
without degrading generation quality (the enforce mode breaks long-context
tasks badly enough that traces become unrepresentative).

Three contracts pinned here:
1. `kwargs["attention_mask"]` survives the pre-hook unchanged (identity-or-
   equal) when observe_only=True.
2. The recorder still receives one row per hook invocation.
3. `validate_attention_method_exclusivity` early-returns when observe_only
   is True — so KV eviction + sparse-observe can coexist in one run.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from serving.kv_policies.base import EvictionPolicyConfig
from serving.recording import RecordingConfig
from serving.recording.hooks import LayerCapturer
from serving.sparse_attention import build_sparse_attention
from serving.sparse_attention.base import SparseAttentionConfig
from serving.sparse_attention.config import validate_attention_method_exclusivity
from serving.sparse_attention.recorder import SparseAttentionRecorder


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

    def forward(self, hidden_states, **_kwargs):  # pragma: no cover
        return hidden_states, None


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = _ToyAttention()


def _make_capturer(*, observe_only: bool) -> LayerCapturer:
    config = SparseAttentionConfig(
        name="sliding",
        sink_size=2,
        recent_window=4,
        record=True,
        observe_only=observe_only,
    )
    method = build_sparse_attention(config, num_layers=1)
    return LayerCapturer(
        _ToyModel(),
        config=RecordingConfig(attention_top_k=2, decode_window=2, max_prefill_queries=4),
        model_summary={"name": "toy"},
        sparse_attention=method,
    )


def test_observe_only_leaves_attention_mask_unchanged() -> None:
    """The pre-hook must not touch kwargs["attention_mask"] in observe mode."""
    capturer = _make_capturer(observe_only=True)
    pre_hook = capturer._sparse_pre_hook(layer=0)

    Q = K = 8
    dtype = torch.float32
    neg_inf = torch.finfo(dtype).min
    causal = torch.zeros((1, 1, Q, K), dtype=dtype)
    for q in range(Q):
        for k in range(K):
            if k > q:
                causal[0, 0, q, k] = neg_inf
    causal_input = causal.clone()
    causal_ptr_before = causal_input.data_ptr()

    hidden_states = torch.zeros((1, Q, 8), dtype=dtype)
    _args, new_kwargs = pre_hook(
        module=capturer,
        args=(hidden_states,),
        kwargs={"attention_mask": causal_input, "past_key_values": None},
    )

    out = new_kwargs["attention_mask"]
    assert out is causal_input, (
        "observe_only must not replace the attention_mask tensor; pre-hook is "
        "supposed to leave kwargs untouched."
    )
    assert out.data_ptr() == causal_ptr_before, "tensor identity lost"
    assert torch.equal(out, causal), "tensor contents mutated in-place"


def test_observe_only_with_no_upstream_mask_stays_none() -> None:
    """When caller didn't pass attention_mask, observe_only must not synthesize one.

    In enforce mode the pre-hook materializes a [1,1,Q,K] sparse mask when
    `existing is None`. That writeback is the whole point of enforce. In
    observe mode we must keep it None so SDPA's implicit causal path fires.
    """
    capturer = _make_capturer(observe_only=True)
    pre_hook = capturer._sparse_pre_hook(layer=0)

    hidden_states = torch.zeros((1, 8, 8), dtype=torch.float32)
    _args, new_kwargs = pre_hook(
        module=capturer,
        args=(hidden_states,),
        kwargs={"past_key_values": None},
    )

    assert "attention_mask" not in new_kwargs or new_kwargs.get("attention_mask") is None, (
        "observe_only must not synthesize an attention_mask when caller "
        f"didn't pass one; got {new_kwargs.get('attention_mask')!r}"
    )


def test_observe_only_still_records_rows() -> None:
    """Recording is the whole point of observe_only — it must still fire."""
    capturer = _make_capturer(observe_only=True)
    recorder = SparseAttentionRecorder(call_idx=0, method_name="sliding")
    capturer.set_sparse_recorder(recorder)
    capturer._session = {"call_idx": 0, "input_token_count": 8}

    pre_hook = capturer._sparse_pre_hook(layer=0)
    hidden_states = torch.zeros((1, 8, 8), dtype=torch.float32)
    pre_hook(
        module=capturer,
        args=(hidden_states,),
        kwargs={"past_key_values": None},
    )

    assert recorder.n_records() == 1, (
        f"observe_only must still record selection rows; got {recorder.n_records()}"
    )

    integrity = capturer._sparse_attention_integrity(sparse_records=1)
    assert integrity["sparse_attention_observe_only"] is True
    assert integrity["sparse_attention_records"] == 1
    assert integrity["sparse_attention_records_match_expected"] is True


def test_enforce_mode_still_writes_back_mask() -> None:
    """Regression guard: observe_only=False keeps the original enforce path live."""
    capturer = _make_capturer(observe_only=False)
    pre_hook = capturer._sparse_pre_hook(layer=0)

    hidden_states = torch.zeros((1, 8, 8), dtype=torch.float32)
    _args, new_kwargs = pre_hook(
        module=capturer,
        args=(hidden_states,),
        kwargs={"past_key_values": None},
    )

    mask = new_kwargs["attention_mask"]
    assert mask is not None, "enforce mode must materialize a mask when none was passed"
    assert mask.shape == (1, 1, 8, 8), f"expected [1,1,Q,K] enforce mask, got {tuple(mask.shape)}"
    neg_inf = torch.finfo(mask.dtype).min
    # sink=2, recent=4, K=8 -> {0,1,4,5,6,7} kept; {2,3} sparse-masked at every q.
    for q in range(8):
        for k in (2, 3):
            assert float(mask[0, 0, q, k]) == neg_inf, (
                f"enforce mask broken at (q={q}, k={k}): {float(mask[0, 0, q, k])}"
            )


# ---------------------------------------------------------------------------
# Validator: KV eviction + sparse observe-only must coexist
# ---------------------------------------------------------------------------


def test_validator_allows_kv_plus_observe_only() -> None:
    kv_cfg = EvictionPolicyConfig(name="h2o", budget=256)
    sp_cfg = SparseAttentionConfig(
        name="sliding", sink_size=4, recent_window=8, observe_only=True, record=True
    )
    validate_attention_method_exclusivity(kv_cfg, sp_cfg)  # must not raise


def test_validator_rejects_kv_plus_enforce_sparse() -> None:
    kv_cfg = EvictionPolicyConfig(name="h2o", budget=256)
    sp_cfg_enforce = SparseAttentionConfig(
        name="sliding", sink_size=4, recent_window=8, observe_only=False
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_attention_method_exclusivity(kv_cfg, sp_cfg_enforce)


def test_validator_accepts_either_alone() -> None:
    kv_cfg = EvictionPolicyConfig(name="h2o", budget=256)
    sp_cfg = SparseAttentionConfig(name="sliding", sink_size=4, recent_window=8)
    validate_attention_method_exclusivity(kv_cfg, None)
    validate_attention_method_exclusivity(None, sp_cfg)
    validate_attention_method_exclusivity(None, None)
