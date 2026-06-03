"""Shared attention-state helpers for recording and sparse selectors."""

from __future__ import annotations

from typing import Any


def project_query_states(module: Any, hidden_states: Any) -> Any:
    """Project hidden states into query heads shaped ``[B, Hq, Q, D]``."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    return module.q_norm(module.q_proj(hidden_states).reshape(hidden_shape)).transpose(1, 2)


def project_key_states(module: Any, hidden_states: Any) -> Any:
    """Project hidden states into key heads shaped ``[B, Hkv, Q, D]``."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    return module.k_norm(module.k_proj(hidden_states).reshape(hidden_shape)).transpose(1, 2)


def apply_rotary_to_states(states: Any, position_embeddings: tuple[Any, Any]) -> Any:
    """Apply HF-style rotary embeddings to projected Q/K states."""
    import torch

    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    left = states[..., : states.shape[-1] // 2]
    right = states[..., states.shape[-1] // 2 :]
    rotated = torch.cat((-right, left), dim=-1)
    return (states * cos) + (rotated * sin)


def select_rotary_positions(
    position_embeddings: tuple[Any, Any],
    row_indices: list[int],
) -> tuple[Any, Any]:
    """Select rotary embedding rows for sampled query indices."""
    import torch

    index: Any | None = None
    selected = []
    for tensor in position_embeddings:
        if index is None:
            index = torch.as_tensor(row_indices, dtype=torch.long, device=tensor.device)
        selected.append(tensor.index_select(-2, index))
    return selected[0], selected[1]


def nonempty_tensor(value: Any) -> Any | None:
    """Return ``value`` if it is a non-empty tensor-like object."""
    if value is None or not hasattr(value, "numel"):
        return None
    if int(value.numel()) == 0:
        return None
    return value


def cached_key_states(past_key_values: Any, layer_idx: int) -> Any | None:
    """Read cached key states from common HF cache layouts."""
    if past_key_values is None:
        return None
    try:
        return nonempty_tensor(past_key_values[layer_idx][0])
    except (AttributeError, KeyError, IndexError, TypeError):
        pass

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        try:
            return nonempty_tensor(getattr(layers[layer_idx], "keys", None))
        except (IndexError, TypeError):
            pass

    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache is not None:
        try:
            return nonempty_tensor(key_cache[layer_idx])
        except (IndexError, TypeError):
            pass
    return None


def current_query_states(
    *,
    module: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
    row_indices: list[int] | None = None,
) -> Any:
    """Return rotary-applied query states for all or selected query rows."""
    if row_indices is None:
        q_states = project_query_states(module, hidden_states)
        return apply_rotary_to_states(q_states, position_embeddings)
    q_states = project_query_states(module, hidden_states[:, row_indices, :])
    return apply_rotary_to_states(
        q_states,
        select_rotary_positions(position_embeddings, row_indices),
    )


def current_key_states(
    *,
    module: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
) -> Any:
    """Return rotary-applied key states for the current forward input."""
    key_states = project_key_states(module, hidden_states)
    return apply_rotary_to_states(key_states, position_embeddings)


def full_key_states_for_pre_hook(
    *,
    module: Any,
    layer_idx: int,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
    past_key_values: Any,
) -> Any:
    """Return cached keys plus current keys for a pre-forward hook.

    HF updates the cache inside the attention forward, after pre-hooks run.
    Query-aware sparse selectors therefore need to concatenate the cached K
    states with the current token/prompt K states themselves.
    """
    import torch

    current = current_key_states(
        module=module,
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
    )
    cached = cached_key_states(past_key_values, layer_idx)
    if cached is None:
        return current
    return torch.cat([cached.to(device=current.device, dtype=current.dtype), current], dim=-2)


__all__ = [
    "apply_rotary_to_states",
    "cached_key_states",
    "current_key_states",
    "current_query_states",
    "full_key_states_for_pre_hook",
    "nonempty_tensor",
    "project_key_states",
    "project_query_states",
    "select_rotary_positions",
]
