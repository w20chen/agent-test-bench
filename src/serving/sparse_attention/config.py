"""CLI / YAML -> SparseAttentionConfig adapter.

Mirrors `serving.kv_policies.config` field-for-field so the two subsystems
share the same resolution semantics: YAML supplies the base map, explicitly
passed `--sparse-attn-*` flags overlay, and `--sparse-attn none` (the
argparse default) does not clobber a yaml-supplied `name`.

`validate_attention_method_exclusivity(kv_config, sparse_config)` is
exported from this module rather than the provider so CLI-time errors
surface before model load — failing in `HFRecordingProvider.__init__` would
already have paid the load cost.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from serving.sparse_attention.base import SparseAttentionConfig

# CLI flag attr -> SparseAttentionConfig field. `sparse_attn` becomes `name`
# because the CLI uses the `sparse_attn_*` prefix.
_CLI_TO_FIELD = {
    "sparse_attn": "name",
    "sparse_attn_sink_size": "sink_size",
    "sparse_attn_recent_window": "recent_window",
    "sparse_attn_record": "record",
    "sparse_attn_observe_only": "observe_only",
    "sparse_attn_budget": "budget",
    "sparse_attn_block_size": "block_size",
    "sparse_attn_score_reduction": "score_reduction",
    "sparse_attn_phase_scope": "phase_scope",
}

_FIELD_COERCERS = {
    "name": str,
    "sink_size": int,
    "recent_window": int,
    "record": lambda v: v if isinstance(v, bool) else str(v).lower() in {"on", "true", "1", "yes"},
    "observe_only": lambda v: v if isinstance(v, bool) else str(v).lower() in {"on", "true", "1", "yes"},
    "budget": int,
    "block_size": int,
    "score_reduction": str,
    "phase_scope": str,
}

_DYNAMIC_METHODS = {"heavy_hitter", "block_topk", "quest"}
_VALID_NAMES = {"none", "sliding", "streaming", *_DYNAMIC_METHODS}


def _allowed_fields() -> set[str]:
    return {f.name for f in fields(SparseAttentionConfig)}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise argparse.ArgumentTypeError(
            f"--sparse-attn-config path does not exist: {path}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError(
            f"--sparse-attn-config {path}: expected a YAML mapping, "
            f"got {type(raw).__name__}"
        )
    allowed = _allowed_fields()
    unknown = set(raw.keys()) - allowed
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--sparse-attn-config {path}: unknown keys {sorted(unknown)}; "
            f"allowed = {sorted(allowed)}"
        )
    return dict(raw)


def _coerce(field_name: str, value: Any) -> Any:
    coercer = _FIELD_COERCERS.get(field_name)
    if coercer is None or value is None:
        return value
    return coercer(value)


def load_sparse_attention_config(
    args: Any, *, config_path: str | None = None
) -> SparseAttentionConfig | None:
    """Build a `SparseAttentionConfig` from yaml + CLI overlay, or None.

    Resolution order matches `load_eviction_config` (same overlay rules):

    1. If `--sparse-attn-config PATH` (or the `config_path` kwarg) is set,
       load that yaml as the base map (flat keys mirroring
       `SparseAttentionConfig`). Empty file or no yaml starts from empty.
    2. For each CLI flag in `_CLI_TO_FIELD`, an explicit (non-default) value
       overrides yaml. `--sparse-attn none` is treated as the implicit
       default and only kicks in when no yaml file is supplied.
    3. Validate: `name` != `"none"` to return a config; `sliding` requires
       both `sink_size` and `recent_window` to be present (non-negative).

    Returns None when the resolved policy is `"none"` so callers can keep
    the `if sparse_attention_config is not None` gate idiom.
    """
    yaml_path = getattr(args, "sparse_attn_config", None) or config_path
    base: dict[str, Any] = {}
    if yaml_path is not None:
        base = _load_yaml(Path(yaml_path))

    cli_defaults = {
        "sparse_attn": "none",
        "sparse_attn_sink_size": 4,
        "sparse_attn_recent_window": 256,
        "sparse_attn_record": "on",
        "sparse_attn_observe_only": False,
        "sparse_attn_budget": None,
        "sparse_attn_block_size": 16,
        "sparse_attn_score_reduction": "max",
        "sparse_attn_phase_scope": "decode_only",
    }

    merged: dict[str, Any] = dict(base)
    for cli_attr, field_name in _CLI_TO_FIELD.items():
        cli_value = getattr(args, cli_attr, cli_defaults.get(cli_attr))
        default_value = cli_defaults.get(cli_attr)
        cli_explicit = cli_value != default_value
        if cli_attr == "sparse_attn":
            if cli_explicit:
                merged[field_name] = cli_value
            elif yaml_path is None and field_name not in merged:
                merged[field_name] = cli_value
        else:
            if cli_explicit:
                merged[field_name] = cli_value
            elif field_name not in merged and default_value is not None:
                merged[field_name] = default_value

    name = merged.get("name", "none")
    if name == "none":
        return None
    if name not in _VALID_NAMES:
        raise argparse.ArgumentTypeError(
            f"sparse attention method {name!r} is unsupported; "
            f"choose one of {sorted(_VALID_NAMES)}"
        )

    if name in {"sliding", "streaming"}:
        if "sink_size" not in merged:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires `sink_size` "
                "(set via --sparse-attn-sink-size or YAML `sink_size:`)"
            )
        if "recent_window" not in merged:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires `recent_window` "
                "(set via --sparse-attn-recent-window or YAML `recent_window:`)"
            )
        sink = int(merged["sink_size"])
        recent = int(merged["recent_window"])
        if sink < 0 or recent < 0:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires non-negative "
                f"sink_size and recent_window; got sink_size={sink!r}, "
                f"recent_window={recent!r}"
            )
        if sink + recent <= 0:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires sink_size + recent_window > 0"
            )
    elif name in _DYNAMIC_METHODS:
        budget = merged.get("budget")
        if budget is None:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires `budget` "
                "(set via --sparse-attn-budget or YAML `budget:`)"
            )
        budget_int = int(budget)
        sink = int(merged.get("sink_size", 4))
        recent = int(merged.get("recent_window", 256))
        block = int(merged.get("block_size", 16))
        score_reduction = str(merged.get("score_reduction", "max"))
        phase_scope = str(merged.get("phase_scope", "decode_only"))
        if budget_int <= 0:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires budget > 0 (got {budget!r})"
            )
        if sink < 0 or recent < 0:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires non-negative "
                f"sink_size and recent_window; got sink_size={sink!r}, "
                f"recent_window={recent!r}"
            )
        if budget_int < sink + recent:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires budget >= sink_size + "
                f"recent_window ({budget_int} < {sink} + {recent})"
            )
        if block <= 0:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires block_size > 0 (got {block!r})"
            )
        if score_reduction not in {"max", "mean"}:
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} requires score_reduction in "
                f"{{'max', 'mean'}}; got {score_reduction!r}"
            )
        if phase_scope != "decode_only":
            raise argparse.ArgumentTypeError(
                f"sparse attention {name!r} currently supports only "
                f"phase_scope='decode_only'; got {phase_scope!r}"
            )
        merged["budget"] = budget_int
        merged["sink_size"] = sink
        merged["recent_window"] = recent
        merged["block_size"] = block
        merged["score_reduction"] = score_reduction
        merged["phase_scope"] = phase_scope

    allowed = _allowed_fields()
    kwargs: dict[str, Any] = {}
    for field_name in allowed:
        if field_name not in merged:
            continue
        kwargs[field_name] = _coerce(field_name, merged[field_name])

    return SparseAttentionConfig(**kwargs)


def validate_attention_method_exclusivity(
    kv_config: Any,
    sparse_config: SparseAttentionConfig | None,
) -> None:
    """Raise if KV eviction and sparse attention are simultaneously active.

    The two subsystems address mutually exclusive points in the design space:
    eviction shrinks the K/V cache (physical drop), sparse attention keeps
    the full cache but masks key positions per query. Combining them would
    layer two competing keep-sets and silently corrupt the heavy-hitter
    accounting on the eviction side.

    Either or both arguments may be None; only the kv+sparse double-on
    case raises.
    """
    if sparse_config is not None and sparse_config.observe_only:
        return
    if kv_config is not None and sparse_config is not None:
        raise ValueError(
            "kv_policy and sparse_attention are mutually exclusive when sparse is in enforce mode. "
            "Use --sparse-attn-observe-only to enable side-channel recording alongside kv eviction. "
            f"Got kv_policy={getattr(kv_config, 'name', kv_config)!r} and "
            f"sparse_attention={sparse_config.name!r}."
        )
