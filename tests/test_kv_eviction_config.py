"""Tests for the YAML + CLI overlay resolver in `serving.kv_policies.config`.

Step 8 introduces `--kv-config PATH` alongside the bare CLI flags. The
adapter has to handle four scenarios:

1. CLI only (no yaml).  -- back-compat with step 3.
2. YAML only (`--kv-config`, all `--kv-*` flags at default).
3. YAML + explicit CLI override.
4. Neither (`--kv-policy none`, no yaml) -> None.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from serving.kv_policies.config import load_eviction_config


def _ns(**kwargs) -> argparse.Namespace:
    """Build an argparse-like namespace with the same defaults as cli.py."""
    base = {
        "kv_policy": "none",
        "kv_budget": None,
        "kv_sink_size": 4,
        "kv_recent_window": 256,
        "kv_aggregate": "sum",
        "kv_record": "on",
        "kv_config": None,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


# --- Scenario 1: CLI-only paths (back-compat with step 3) ----------------


def test_none_returns_none() -> None:
    assert load_eviction_config(_ns()) is None


def test_cli_only_h2o() -> None:
    cfg = load_eviction_config(_ns(kv_policy="h2o", kv_budget=512))
    assert cfg is not None
    assert cfg.name == "h2o"
    assert cfg.budget == 512
    # Defaults preserved.
    assert cfg.sink_size == 4
    assert cfg.recent_window == 256
    assert cfg.aggregate == "sum"
    assert cfg.prefill_mode == "full"


def test_cli_only_streaming_with_overrides() -> None:
    cfg = load_eviction_config(
        _ns(
            kv_policy="streaming",
            kv_budget=1024,
            kv_sink_size=8,
            kv_recent_window=1016,
        )
    )
    assert cfg is not None
    assert cfg.name == "streaming"
    assert cfg.budget == 1024
    assert cfg.sink_size == 8
    assert cfg.recent_window == 1016


def test_cli_only_streaming_defaults_recent_to_capacity() -> None:
    cfg = load_eviction_config(_ns(kv_policy="streaming", kv_budget=1024))
    assert cfg is not None
    assert cfg.name == "streaming"
    assert cfg.sink_size == 4
    assert cfg.recent_window == 1020


def test_cli_missing_budget_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="budget"):
        load_eviction_config(_ns(kv_policy="random"))


def test_cli_negative_budget_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="budget > 0"):
        load_eviction_config(_ns(kv_policy="random", kv_budget=-5))


# --- Scenario 2: YAML only ------------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_yaml_only_drives_config(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "name: h2o\nbudget: 2048\nsink_size: 8\nrecent_window: 512\naggregate: ema\nema_decay: 0.7\nprefill_mode: full\n",
    )
    cfg = load_eviction_config(_ns(kv_config=str(yaml_path)))
    assert cfg is not None
    assert cfg.name == "h2o"
    assert cfg.budget == 2048
    assert cfg.sink_size == 8
    assert cfg.recent_window == 512
    assert cfg.aggregate == "ema"
    assert cfg.ema_decay == pytest.approx(0.7)
    assert cfg.prefill_mode == "full"


def test_yaml_record_off(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "name: streaming\nbudget: 1024\nrecord: false\n",
    )
    cfg = load_eviction_config(_ns(kv_config=str(yaml_path)))
    assert cfg is not None
    assert cfg.record is False


def test_yaml_unknown_key_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "name: random\nbudget: 16\nbogus_key: 42\n",
    )
    with pytest.raises(argparse.ArgumentTypeError, match="unknown keys"):
        load_eviction_config(_ns(kv_config=str(yaml_path)))


def test_yaml_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
        load_eviction_config(_ns(kv_config=str(tmp_path / "nope.yaml")))


def test_yaml_non_mapping_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "list.yaml"
    yaml_path.write_text("- name: random\n  budget: 16\n", encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="mapping"):
        load_eviction_config(_ns(kv_config=str(yaml_path)))


def test_yaml_missing_budget_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, "name: h2o\n")
    with pytest.raises(argparse.ArgumentTypeError, match="requires `budget`"):
        load_eviction_config(_ns(kv_config=str(yaml_path)))


# --- Scenario 3: YAML + CLI overlay ---------------------------------------


def test_cli_overrides_yaml_budget(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path, "name: h2o\nbudget: 1024\nsink_size: 8\n"
    )
    # Explicit --kv-budget=2048 must win over yaml.
    cfg = load_eviction_config(
        _ns(kv_config=str(yaml_path), kv_budget=2048)
    )
    assert cfg is not None
    assert cfg.budget == 2048
    # Untouched yaml values pass through.
    assert cfg.sink_size == 8


def test_cli_default_does_not_overwrite_yaml(tmp_path: Path) -> None:
    """Argparse defaults must not silently overwrite yaml values; only an
    explicit (non-default) CLI flag should override.
    """
    yaml_path = _write_yaml(
        tmp_path,
        "name: streaming\nbudget: 1024\nsink_size: 16\nrecent_window: 512\n",
    )
    # All CLI fields at their argparse defaults.
    cfg = load_eviction_config(_ns(kv_config=str(yaml_path)))
    assert cfg is not None
    # yaml's 16 / 512 must survive, not the cli defaults 4 / 256.
    assert cfg.sink_size == 16
    assert cfg.recent_window == 512


def test_explicit_kv_policy_overrides_yaml_name(tmp_path: Path) -> None:
    """Explicit --kv-policy beats the yaml `name`."""
    yaml_path = _write_yaml(tmp_path, "name: random\nbudget: 64\n")
    cfg = load_eviction_config(
        _ns(kv_config=str(yaml_path), kv_policy="streaming", kv_budget=64)
    )
    assert cfg is not None
    assert cfg.name == "streaming"


def test_kv_record_cli_off_overrides_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path, "name: random\nbudget: 64\nrecord: true\n"
    )
    cfg = load_eviction_config(
        _ns(kv_config=str(yaml_path), kv_record="off")
    )
    assert cfg is not None
    assert cfg.record is False
