"""Tests for the YAML + CLI overlay resolver in `serving.sparse_attention.config`.

Four scenarios (mirrors `tests/test_kv_eviction_config.py`):
1. CLI only (no yaml)
2. YAML only (`--sparse-attn-config`, all `--sparse-attn-*` flags at default)
3. YAML + explicit CLI override
4. Neither (`--sparse-attn none`, no yaml) -> None

Plus exclusivity validator: kv+sparse both non-None -> ValueError.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from serving.sparse_attention.config import (
    load_sparse_attention_config,
    validate_attention_method_exclusivity,
)


def _ns(**kwargs) -> argparse.Namespace:
    base = {
        "sparse_attn": "none",
        "sparse_attn_sink_size": 4,
        "sparse_attn_recent_window": 256,
        "sparse_attn_record": "on",
        "sparse_attn_config": None,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


# --- Scenario 1: CLI-only ---------------------------------------------------


def test_none_returns_none() -> None:
    assert load_sparse_attention_config(_ns()) is None


def test_cli_only_sliding() -> None:
    cfg = load_sparse_attention_config(_ns(sparse_attn="sliding"))
    assert cfg is not None
    assert cfg.name == "sliding"
    assert cfg.sink_size == 4
    assert cfg.recent_window == 256
    assert cfg.record is True


def test_cli_only_sliding_with_overrides() -> None:
    cfg = load_sparse_attention_config(
        _ns(
            sparse_attn="sliding",
            sparse_attn_sink_size=8,
            sparse_attn_recent_window=128,
            sparse_attn_record="off",
        )
    )
    assert cfg is not None
    assert cfg.sink_size == 8
    assert cfg.recent_window == 128
    assert cfg.record is False


def test_cli_only_block_topk_requires_budget() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="requires `budget`"):
        load_sparse_attention_config(_ns(sparse_attn="block_topk"))


def test_cli_only_dynamic_method_with_budget() -> None:
    cfg = load_sparse_attention_config(
        _ns(
            sparse_attn="quest",
            sparse_attn_budget=1024,
            sparse_attn_block_size=32,
            sparse_attn_score_reduction="mean",
        )
    )
    assert cfg is not None
    assert cfg.name == "quest"
    assert cfg.budget == 1024
    assert cfg.block_size == 32
    assert cfg.score_reduction == "mean"
    assert cfg.phase_scope == "decode_only"


def test_dynamic_budget_must_cover_sink_recent() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="budget >= sink_size"):
        load_sparse_attention_config(
            _ns(
                sparse_attn="heavy_hitter",
                sparse_attn_budget=4,
                sparse_attn_sink_size=2,
                sparse_attn_recent_window=8,
            )
        )


def test_streaming_alias_uses_sliding_knobs() -> None:
    cfg = load_sparse_attention_config(_ns(sparse_attn="streaming"))
    assert cfg is not None
    assert cfg.name == "streaming"
    assert cfg.sink_size == 4
    assert cfg.recent_window == 256


def test_cli_negative_sink_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        load_sparse_attention_config(
            _ns(sparse_attn="sliding", sparse_attn_sink_size=-1)
        )


def test_cli_zero_sink_and_window_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="sink_size \\+ recent_window > 0"):
        load_sparse_attention_config(
            _ns(
                sparse_attn="sliding",
                sparse_attn_sink_size=0,
                sparse_attn_recent_window=0,
            )
        )


# --- Scenario 2: YAML only --------------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_yaml_only_drives_config(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "name: sliding\nsink_size: 16\nrecent_window: 512\nrecord: false\n",
    )
    cfg = load_sparse_attention_config(_ns(sparse_attn_config=str(yaml_path)))
    assert cfg is not None
    assert cfg.name == "sliding"
    assert cfg.sink_size == 16
    assert cfg.recent_window == 512
    assert cfg.record is False


def test_yaml_unknown_key_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path, "name: sliding\nsink_size: 4\nrecent_window: 256\nbogus: 1\n"
    )
    with pytest.raises(argparse.ArgumentTypeError, match="unknown keys"):
        load_sparse_attention_config(_ns(sparse_attn_config=str(yaml_path)))


def test_yaml_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
        load_sparse_attention_config(_ns(sparse_attn_config=str(tmp_path / "nope.yaml")))


def test_yaml_non_mapping_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "list.yaml"
    yaml_path.write_text("- name: sliding\n", encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="mapping"):
        load_sparse_attention_config(_ns(sparse_attn_config=str(yaml_path)))


# --- Scenario 3: YAML + CLI overlay ----------------------------------------


def test_cli_overrides_yaml_sink_size(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path, "name: sliding\nsink_size: 4\nrecent_window: 64\n"
    )
    cfg = load_sparse_attention_config(
        _ns(sparse_attn_config=str(yaml_path), sparse_attn_sink_size=32)
    )
    assert cfg is not None
    assert cfg.sink_size == 32
    # Untouched yaml value passes through.
    assert cfg.recent_window == 64


def test_cli_default_does_not_overwrite_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "name: sliding\nsink_size: 16\nrecent_window: 512\n",
    )
    cfg = load_sparse_attention_config(_ns(sparse_attn_config=str(yaml_path)))
    assert cfg is not None
    assert cfg.sink_size == 16
    assert cfg.recent_window == 512


def test_explicit_sparse_attn_overrides_yaml_name(tmp_path: Path) -> None:
    # Sliding is the only registered method right now; we round-trip the same
    # name via CLI to confirm the override branch fires without relying on a
    # second method existing.
    yaml_path = _write_yaml(
        tmp_path, "name: sliding\nsink_size: 4\nrecent_window: 16\n"
    )
    cfg = load_sparse_attention_config(
        _ns(sparse_attn_config=str(yaml_path), sparse_attn="sliding")
    )
    assert cfg is not None
    assert cfg.name == "sliding"


def test_sparse_attn_record_cli_off_overrides_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path, "name: sliding\nsink_size: 4\nrecent_window: 16\nrecord: true\n"
    )
    cfg = load_sparse_attention_config(
        _ns(sparse_attn_config=str(yaml_path), sparse_attn_record="off")
    )
    assert cfg is not None
    assert cfg.record is False


# --- Exclusivity validator -------------------------------------------------


class _KVStub:
    def __init__(self, name: str = "h2o") -> None:
        self.name = name


def test_validator_both_none_ok() -> None:
    validate_attention_method_exclusivity(None, None)


def test_validator_kv_only_ok() -> None:
    validate_attention_method_exclusivity(_KVStub(), None)


def test_validator_sparse_only_ok() -> None:
    sparse = load_sparse_attention_config(_ns(sparse_attn="sliding"))
    validate_attention_method_exclusivity(None, sparse)


def test_validator_both_set_raises() -> None:
    sparse = load_sparse_attention_config(_ns(sparse_attn="sliding"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_attention_method_exclusivity(_KVStub("h2o"), sparse)
