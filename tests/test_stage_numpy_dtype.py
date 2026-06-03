from __future__ import annotations

import numpy as np
import torch

from serving.recording.hooks import _stage_numpy


def _legacy_numpy_cast(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def test_stage_numpy_fp16_source_matches_legacy_path() -> None:
    tensor = torch.tensor(
        [[0.125, -1.5, 3.25], [7.5, 0.0, -0.03125]],
        dtype=torch.float16,
    ).t()

    old_out = _legacy_numpy_cast(tensor)
    new_out = _stage_numpy(tensor, np.float32).materialize()

    assert old_out.dtype == np.float32
    assert new_out.dtype == np.float32
    np.testing.assert_array_equal(new_out, old_out)


def test_stage_numpy_bf16_source_matches_legacy_or_torch_upcast() -> None:
    tensor = torch.tensor(
        [[0.125, -1.5, 3.25], [7.5, 0.0, -0.03125]],
        dtype=torch.bfloat16,
    ).t()

    new_out = _stage_numpy(tensor, np.float32).materialize()
    reference = tensor.detach().to(dtype=torch.float32).cpu().numpy()

    assert new_out.dtype == np.float32
    np.testing.assert_array_equal(new_out, reference)
    try:
        old_out = _legacy_numpy_cast(tensor)
    except TypeError:
        return
    np.testing.assert_array_equal(new_out, old_out)


def test_fp16_round_trip_for_attention_probabilities() -> None:
    """Lock fp16 quantization budget for probability fields in [0,1].

    Attention top-k weights, segment_mass rows, and expert softmax weights
    all live in [0,1] and now serialize to fp16. Validate the loss budget
    matches the assumption (atol=1e-3) for typical values, and assert no
    NaN/inf are produced even for tiny values that may underflow.
    """
    rng = np.random.default_rng(0)
    orig = rng.uniform(0.0, 1.0, size=1000).astype(np.float32)
    roundtrip = orig.astype(np.float16).astype(np.float32)
    assert np.allclose(orig, roundtrip, atol=1e-3)

    tiny = np.asarray([1e-3, 1e-4, 1e-5, 1e-6, 0.0], dtype=np.float32)
    tiny_round = tiny.astype(np.float16).astype(np.float32)
    assert not np.any(np.isnan(tiny_round))
    assert not np.any(np.isinf(tiny_round))
    # 1e-3 must survive cleanly; smaller values may underflow to 0 but never
    # produce non-finite output.
    assert abs(float(tiny_round[0]) - 1e-3) < 1e-4
