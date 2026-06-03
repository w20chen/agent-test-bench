"""CLI / YAML -> EvictionPolicyConfig adapter.

Step 8 extends the step-3 adapter to accept a YAML file (`--kv-config PATH`)
in addition to bare CLI flags. Resolution order is documented in the body of
`load_eviction_config`: yaml provides the base, CLI flags overlay it, and the
merge result must satisfy the same `EvictionPolicyConfig` invariants the
caches enforce at construction time.

YAML schema is a flat map mirroring `EvictionPolicyConfig`'s fields one-for-one.
We deliberately skip Hydra here because the schema is small (~10 keys) and
keeping a single yaml.safe_load call keeps the loading semantics auditable.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from serving.kv_policies.base import EvictionPolicyConfig

# CLI flag attr name -> EvictionPolicyConfig field name. Only flags that map
# 1:1 into the dataclass appear here; `kv_policy` becomes `name` and
# `kv_budget` becomes `budget` because CLI uses the `kv_*` prefix.
_CLI_TO_FIELD = {
    "kv_policy": "name",
    "kv_budget": "budget",
    "kv_sink_size": "sink_size",
    "kv_recent_window": "recent_window",
    "kv_aggregate": "aggregate",
    "kv_record": "record",
}

# Coercion table for YAML/CLI string-y values into the dataclass field type.
# Kept narrow: only the fields users actually set today have entries.
_FIELD_COERCERS = {
    "budget": int,
    "sink_size": int,
    "recent_window": int,
    "heavy_ratio": float,
    "ema_decay": float,
    "seed": int,
    "record": lambda v: v if isinstance(v, bool) else str(v).lower() in {"on", "true", "1", "yes"},
    "prefill_mode": str,
    "aggregate": str,
    "name": str,
}


def _allowed_fields() -> set[str]:
    return {f.name for f in fields(EvictionPolicyConfig)}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise argparse.ArgumentTypeError(f"--kv-config path does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError(
            f"--kv-config {path}: expected a YAML mapping, got {type(raw).__name__}"
        )
    allowed = _allowed_fields()
    unknown = set(raw.keys()) - allowed
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--kv-config {path}: unknown keys {sorted(unknown)}; "
            f"allowed = {sorted(allowed)}"
        )
    return dict(raw)


def _coerce(field_name: str, value: Any) -> Any:
    coercer = _FIELD_COERCERS.get(field_name)
    if coercer is None or value is None:
        return value
    return coercer(value)


def load_eviction_config(args: Any) -> EvictionPolicyConfig | None:
    """Build an `EvictionPolicyConfig` from yaml + CLI overlay, or None.

    Resolution order
    ----------------
    1. If `--kv-config PATH` is set, load that yaml as the base map (flat
       keys mirroring `EvictionPolicyConfig`). Empty file or no `--kv-config`
       starts from an empty base.
    2. For each CLI flag listed in `_CLI_TO_FIELD`, if the user *explicitly*
       set it (i.e. the parsed value differs from the argparse default), the
       CLI value overrides the yaml value. The argparse default for
       `--kv-policy` is `"none"`; treating that as a real override would
       silently disable yaml-supplied policies, so we special-case it: a
       CLI `"none"` only kicks in when there's no yaml file.
    3. Validate the merged map: `name` must be present and != `"none"`,
       `budget` must be a positive int. For streaming, an omitted
       `recent_window` is resolved to `budget - sink_size` so `--kv-budget`
       alone means "fixed cache capacity"; explicit YAML/CLI windows remain
       explicit and are validated by the cache subclass.

    Returns None when the resolved policy is `"none"` (or absent and no
    yaml supplied), so callers can keep the simple `if eviction_config is
    not None` pattern.
    """
    yaml_path = getattr(args, "kv_config", None)
    base: dict[str, Any] = {}
    if yaml_path is not None:
        base = _load_yaml(Path(yaml_path))

    # Argparse defaults for the kv_* flags. Anything matching the default and
    # not set in yaml is left to the dataclass default; anything matching the
    # default but with a yaml value present yields the yaml value. Anything
    # different from the default overrides yaml.
    cli_defaults = {
        "kv_policy": "none",
        "kv_budget": None,
        "kv_sink_size": 4,
        "kv_recent_window": 256,
        "kv_aggregate": "sum",
        "kv_record": "on",
    }

    merged: dict[str, Any] = dict(base)
    explicit_cli_fields: set[str] = set()
    for cli_attr, field_name in _CLI_TO_FIELD.items():
        cli_value = getattr(args, cli_attr, cli_defaults.get(cli_attr))
        default_value = cli_defaults.get(cli_attr)
        cli_explicit = cli_value != default_value
        if cli_attr == "kv_policy":
            # `--kv-policy none` is the implicit default. Only let it override
            # a yaml-supplied name when the user did not pass --kv-config.
            if cli_explicit:
                merged[field_name] = cli_value
                explicit_cli_fields.add(field_name)
            elif yaml_path is None and field_name not in merged:
                merged[field_name] = cli_value
        else:
            if cli_explicit:
                merged[field_name] = cli_value
                explicit_cli_fields.add(field_name)
            elif field_name not in merged and default_value is not None:
                # Carry the argparse default into the merged map only when
                # yaml didn't speak — keeps the dataclass default visible
                # in CLI-only flows just like step 3.
                merged[field_name] = default_value

    name = merged.get("name", "none")
    if name == "none":
        return None

    budget = merged.get("budget")
    if budget is None:
        raise argparse.ArgumentTypeError(
            f"kv policy {name!r} requires `budget` "
            "(set via --kv-budget or YAML `budget:`)"
        )
    try:
        budget_int = int(budget)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"kv policy {name!r}: `budget` must be an int, got {budget!r}"
        ) from exc
    if budget_int <= 0:
        raise argparse.ArgumentTypeError(
            f"kv policy {name!r} requires budget > 0 (got {budget!r})"
        )
    merged["budget"] = budget_int

    if name == "streaming":
        recent_explicit = (
            "recent_window" in base or "recent_window" in explicit_cli_fields
        )
        if not recent_explicit:
            sink = int(merged.get("sink_size", EvictionPolicyConfig.sink_size))
            recent = budget_int - sink
            if recent <= 0:
                raise argparse.ArgumentTypeError(
                    f"kv policy 'streaming' requires budget > sink_size "
                    f"when recent_window is omitted (got budget={budget_int}, sink={sink})"
                )
            merged["recent_window"] = recent

    # Drop unknown keys (already validated by _load_yaml for yaml; CLI keys
    # are all in _CLI_TO_FIELD which targets known fields). Coerce remaining
    # values into the dataclass-expected types.
    allowed = _allowed_fields()
    kwargs: dict[str, Any] = {}
    for field_name in allowed:
        if field_name not in merged:
            continue
        kwargs[field_name] = _coerce(field_name, merged[field_name])

    return EvictionPolicyConfig(**kwargs)
