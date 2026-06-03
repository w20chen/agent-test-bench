"""Exclusivity tests: kv_policy + sparse_attention must not coexist.

Two paths exercise the gate:
1. CLI-level `validate_attention_method_exclusivity` (raises immediately).
2. `HFRecordingProvider.__init__` belt-and-suspenders assert that fires
   even when callers bypass the CLI.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from serving.kv_policies.base import EvictionPolicyConfig
from serving.sparse_attention.base import SparseAttentionConfig
from serving.sparse_attention.config import validate_attention_method_exclusivity


def test_validator_rejects_both_non_none() -> None:
    kv = EvictionPolicyConfig(name="h2o", budget=256, sink_size=4, recent_window=64)
    sparse = SparseAttentionConfig(name="sliding", sink_size=4, recent_window=64)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_attention_method_exclusivity(kv, sparse)


def test_validator_allows_either_alone() -> None:
    kv = EvictionPolicyConfig(name="streaming", budget=64, sink_size=4, recent_window=60)
    sparse = SparseAttentionConfig(name="sliding", sink_size=4, recent_window=64)
    # Both single-side calls should be no-ops.
    validate_attention_method_exclusivity(kv, None)
    validate_attention_method_exclusivity(None, sparse)
    validate_attention_method_exclusivity(None, None)


def test_provider_init_rejects_both_non_none() -> None:
    """Provider must short-circuit before any model load."""
    kv = EvictionPolicyConfig(name="h2o", budget=256, sink_size=4, recent_window=64)
    sparse = SparseAttentionConfig(name="sliding", sink_size=4, recent_window=64)

    # Patch `AutoTokenizer.from_pretrained` / `AutoModelForCausalLM.from_pretrained`
    # so the test never touches the network or GPU. The validator should raise
    # BEFORE either patched call fires.
    tokenizer_call_count = 0
    model_call_count = 0

    def _fake_tokenizer(*_args, **_kwargs):
        nonlocal tokenizer_call_count
        tokenizer_call_count += 1
        raise AssertionError("tokenizer load reached despite validator")

    def _fake_model(*_args, **_kwargs):
        nonlocal model_call_count
        model_call_count += 1
        raise AssertionError("model load reached despite validator")

    # `HFRecordingProvider.__init__` imports AutoTokenizer/AutoModelForCausalLM
    # lazily inside the constructor; patch the canonical module path so the
    # validator's pre-load `ValueError` is what surfaces.
    import transformers

    from serving.recording import backend_hf

    with (
        patch.object(
            transformers.AutoTokenizer, "from_pretrained", side_effect=_fake_tokenizer
        ),
        patch.object(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            side_effect=_fake_model,
        ),
        pytest.raises(ValueError, match="mutually exclusive"),
    ):
        backend_hf.HFRecordingProvider(
            default_model="dummy",
            eviction_config=kv,
            sparse_attention_config=sparse,
        )
    assert tokenizer_call_count == 0
    assert model_call_count == 0
