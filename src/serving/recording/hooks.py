"""Forward hooks and artifact writer for HF internal recordings."""

from __future__ import annotations

import json
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from serving.recording.recording import (
    DecodeRingBuffer,
    RecordingConfig,
    expert_load_per_segment,
    heavy_hitter,
    padded_top_k,
    query_sampling_seed,
    segment_bucket,
    select_query_positions,
    token_segment_ids,
)
from serving.sparse_attention.base import SparseAttentionContext
from serving.sparse_attention.state import (
    apply_rotary_to_states as _apply_rotary_to_states,
    cached_key_states as _cached_key_states,
    current_query_states as _current_query_states,
    project_key_states as _project_key_states,
)

if TYPE_CHECKING:
    # Lazy: avoid forcing transformers import via kv_policies at module load.
    from serving.kv_policies.recorder import KVEvictionRecorder
    from serving.recording.attention_bus import AttentionBus
    from serving.sparse_attention.base import BaseSparseAttention
    from serving.sparse_attention.recorder import SparseAttentionRecorder


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn$")
_GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
_LLM_ACTION_ID_RE = re.compile(r"^llm_(\d+)$")


def _layer_index(module_name: str) -> int | None:
    match = _LAYER_RE.search(module_name)
    if match:
        return int(match.group(1))
    if module_name.endswith(".self_attn") or module_name == "self_attn":
        return -1
    return None


def _gate_layer_index(module_name: str) -> int | None:
    match = _GATE_RE.search(module_name)
    return int(match.group(1)) if match else None


def _as_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@dataclass(frozen=True)
class _PendingNumpyArray:
    """CPU-staged tensor whose NumPy view is materialized at flush time."""

    tensor: Any
    dtype: np.dtype
    source_device: Any | None
    _synchronized: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.tensor.shape)

    def mark_synchronized(self) -> None:
        object.__setattr__(self, "_synchronized", True)

    def synchronize(self) -> None:
        if self.source_device is None or self._synchronized:
            return
        import torch

        torch.cuda.synchronize(self.source_device)
        self.mark_synchronized()

    def materialize(self) -> np.ndarray:
        self.synchronize()
        array = self.tensor.numpy()
        if array.dtype != self.dtype:
            return array.astype(self.dtype, copy=False)
        return array

    def __array__(self, dtype: Any = None) -> np.ndarray:
        array = self.materialize()
        if dtype is None:
            return array
        return array.astype(dtype, copy=False)

    def astype(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.materialize().astype(*args, **kwargs)

    def copy(self) -> np.ndarray:
        return self.materialize().copy()


def _stage_numpy(tensor: Any, dtype: Any) -> _PendingNumpyArray:
    import torch

    np_dtype = np.dtype(dtype)
    torch_dtype = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.int32): torch.int32,
    }.get(np_dtype)
    staged = tensor.detach()
    if torch_dtype is not None and staged.dtype != torch_dtype:
        staged = staged.to(dtype=torch_dtype)
    source_device = staged.device if staged.device.type == "cuda" else None
    if source_device is not None:
        cpu_tensor = torch.empty_like(
            staged,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        cpu_tensor.copy_(staged, non_blocking=True)
    else:
        cpu_tensor = staged.to(device=torch.device("cpu"))
    return _PendingNumpyArray(
        cpu_tensor,
        np_dtype,
        source_device,
    )


def _materialize_array(value: Any) -> np.ndarray:
    if isinstance(value, _PendingNumpyArray):
        return value.materialize()
    return value


def _encode_topk_csr(
    indices: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode padded dense top-k rows as row-wise CSR arrays."""
    if indices.ndim != 2 or weights.ndim != 2:
        raise ValueError(
            f"top-k arrays must be rank 2, got {indices.shape} and {weights.shape}"
        )
    if indices.shape != weights.shape:
        raise ValueError(
            f"top-k index/weight shape mismatch: {indices.shape} vs {weights.shape}"
        )

    row_offsets = np.zeros(indices.shape[0] + 1, dtype=np.int64)
    valid = indices >= 0
    counts = valid.sum(axis=1, dtype=np.int64)
    row_offsets[1:] = np.cumsum(counts, dtype=np.int64)
    if int(row_offsets[-1]) == 0:
        return (
            row_offsets,
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float16),
        )
    return (
        row_offsets,
        indices[valid].astype(np.int32, copy=False),
        weights[valid].astype(np.float16, copy=False),
    )


def _span_span_matrix(
    *,
    segment_mass: np.ndarray,
    query_positions: np.ndarray,
    query_row_offsets: np.ndarray,
    segments: list[dict[str, Any]],
    n_segments: int,
    generated_segment_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate query-segment by key-segment attention mass per record."""
    if segment_mass.ndim != 2:
        raise ValueError(f"segment_mass must be rank 2, got {segment_mass.shape}")
    if segment_mass.shape[1] != n_segments:
        raise ValueError(
            f"segment_mass width {segment_mass.shape[1]} != n_segments {n_segments}"
        )
    if query_positions.ndim != 1:
        raise ValueError(f"query_positions must be rank 1, got {query_positions.shape}")
    if query_positions.shape[0] != segment_mass.shape[0]:
        raise ValueError(
            f"query_positions length {query_positions.shape[0]} "
            f"!= segment_mass rows {segment_mass.shape[0]}"
        )
    if query_row_offsets.ndim != 1:
        raise ValueError(
            f"query_row_offsets must be rank 1, got {query_row_offsets.shape}"
        )
    n_records = max(0, int(query_row_offsets.shape[0]) - 1)
    raw = np.zeros((n_records, n_segments, n_segments), dtype=np.float32)
    counts = np.zeros((n_records, n_segments), dtype=np.int32)
    row_sums = np.zeros((n_records, n_segments), dtype=np.float32)
    if n_records == 0 or query_positions.size == 0:
        return raw, counts, raw.copy(), row_sums

    total_tokens = int(query_positions.max()) + 1
    token_ids = token_segment_ids(
        total_tokens,
        segments,
        generated_segment_id=generated_segment_id,
    ).numpy()
    query_segments = np.full(query_positions.shape, -1, dtype=np.int64)
    valid_positions = (query_positions >= 0) & (query_positions < token_ids.shape[0])
    query_segments[valid_positions] = token_ids[query_positions[valid_positions]]

    for record_idx in range(n_records):
        start = int(query_row_offsets[record_idx])
        end = int(query_row_offsets[record_idx + 1])
        if end <= start:
            continue
        record_segments = query_segments[start:end]
        record_rows = segment_mass[start:end]
        valid_segments = record_segments[
            (record_segments >= 0) & (record_segments < n_segments)
        ]
        for query_segment in np.unique(valid_segments):
            mask = record_segments == int(query_segment)
            row_count = int(mask.sum())
            if row_count <= 0:
                continue
            counts[record_idx, int(query_segment)] = row_count
            raw[record_idx, int(query_segment)] = record_rows[mask].sum(axis=0)

    row_sums = raw.sum(axis=2).astype(np.float32, copy=False)
    row_sums_expanded = row_sums[:, :, None]
    active = (counts[:, :, None] > 0) & (row_sums_expanded > 0)
    normalized = np.where(
        active,
        raw / np.maximum(row_sums_expanded, np.finfo(np.float32).tiny),
        raw,
    )
    return normalized.astype(np.float32, copy=False), counts, raw, row_sums


def _routing_count_summary(
    choices: Any,
    token_ids: Any,
    *,
    n_segments: int,
    n_experts: int,
) -> dict[str, np.ndarray | str]:
    """Discrete top-k expert assignment counts plus a labeled overflow proxy."""
    import torch

    if choices.ndim != 2:
        raise ValueError(f"expert choices must be rank 2, got {choices.shape}")
    n_tokens = int(choices.shape[0])
    counts = torch.zeros(
        (n_segments, n_experts),
        dtype=torch.int32,
        device=choices.device,
    )
    top_k = int(choices.shape[1]) if choices.ndim == 2 else 0
    if n_tokens > 0 and n_experts > 0 and top_k > 0:
        segment_ids = token_ids[:n_tokens].to(device=choices.device, dtype=torch.long)
        valid_segments = (segment_ids >= 0) & (segment_ids < n_segments)
        # Hoist scalar-one outside the rank loop; index_put_ broadcasts a 0-d
        # value across the index set, so we don't need to materialize per-rank
        # `ones` and can skip the `.any()` / `.sum().item()` GPU->CPU syncs.
        scalar_one = torch.ones((), dtype=torch.int32, device=choices.device)
        for rank in range(top_k):
            experts = choices[:, rank].to(dtype=torch.long)
            valid = valid_segments & (experts >= 0) & (experts < n_experts)
            # `index_put_` on empty index tensors is a safe no-op; scalar 1
            # broadcasts across the index set, so the count is unnecessary.
            counts.index_put_(
                (segment_ids[valid], experts[valid]),
                scalar_one,
                accumulate=True,
            )

    count_np = _as_numpy(counts).astype(np.int32)
    total_per_expert = count_np.sum(axis=0, dtype=np.int64)
    n_assignments = int(count_np.sum())
    capacity = (
        int(np.ceil(float(n_assignments) / float(n_experts)))
        if n_experts > 0
        else 0
    )
    expected_overflow = np.maximum(total_per_expert - capacity, 0).astype(np.int32)
    return {
        "expert_token_count": count_np,
        "expert_token_count_unit": "topk_assignments",
        "expert_capacity": np.asarray(capacity, dtype=np.int32),
        "expert_expected_overflow_count": expected_overflow,
        "expected_dropped_token_count": np.asarray(
            int(expected_overflow.sum()),
            dtype=np.int32,
        ),
        "drop_signal_mode": "expected_uniform_capacity",
    }


def _synchronize_pending_arrays(records: list[dict[str, Any]]) -> None:
    pending: list[_PendingNumpyArray] = []
    devices: dict[str, Any] = {}
    for record in records:
        for value in record.values():
            if isinstance(value, _PendingNumpyArray) and value.source_device is not None:
                pending.append(value)
                devices[str(value.source_device)] = value.source_device
    if not devices:
        return

    import torch

    for device in devices.values():
        torch.cuda.synchronize(device)
    for value in pending:
        value.mark_synchronized()


def _arg_or_kw(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
    *,
    default: Any = None,
) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _load_trace_llm_calls(trace_path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "action" and record.get("action_type") == "llm_call":
                calls.append(record)
    return calls


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_trace_indices(values: list[int | None]) -> list[int] | None:
    if not values or any(value is None for value in values):
        return None
    concrete = [int(value) for value in values if value is not None]
    if concrete != sorted(concrete) or len(set(concrete)) != len(concrete):
        return None
    offset = concrete[0]
    if offset not in {0, 1}:
        return None
    return [value - offset for value in concrete]


def _trace_call_indices(trace_calls: list[dict[str, Any]]) -> tuple[list[int], str] | None:
    iteration_indices = _normalize_trace_indices(
        [_int_or_none(call.get("iteration")) for call in trace_calls]
    )
    if iteration_indices is not None:
        return iteration_indices, "iteration"

    action_id_values: list[int | None] = []
    for call in trace_calls:
        action_id = call.get("action_id")
        match = _LLM_ACTION_ID_RE.match(action_id) if isinstance(action_id, str) else None
        action_id_values.append(_int_or_none(match.group(1)) if match else None)
    action_id_indices = _normalize_trace_indices(action_id_values)
    if action_id_indices is not None:
        return action_id_indices, "action_id"
    return None


class LayerCapturer:
    """Capture reduced attention and routing tensors for one HF model."""

    def __init__(
        self,
        model: Any,
        *,
        config: RecordingConfig,
        model_summary: dict[str, Any],
        kv_recorder: "KVEvictionRecorder | None" = None,
        attention_bus: "AttentionBus | None" = None,
        sparse_attention: "BaseSparseAttention | None" = None,
    ) -> None:
        self.config = config
        self.model_summary = dict(model_summary)
        self._handles = []
        self._attempt_dir: Path | None = None
        self._session: dict[str, Any] | None = None
        self._prefill_records: list[dict[str, Any]] = []
        self._decode_records = DecodeRingBuffer(config.decode_window)
        self._routing_records: list[dict[str, Any]] = []
        self._routing_seen_prefill: set[int] = set()
        self._routing_decode_steps: dict[int, int] = {}
        self._meta: dict[str, Any] = {}
        self._attention_suspended = 0
        # Per-head span stats accumulators. Keyed by (layer_idx, phase, decode_step).
        # Values: {"mean_sum": [H, S], "var_sum": [H, S], "n_queries": int}
        self._head_stats: dict[tuple[int, str, int], dict[str, Any]] = {}
        self._head_stats_n_segments: int = 0
        # KV eviction recorder is per-call; lifecycle is owned by the caller
        # (HFRecordingProvider in step 3+) which swaps it before each
        # `recording_session()`. LayerCapturer only flushes whatever the
        # current recorder buffered when the session ends.
        self._kv_recorder: "KVEvictionRecorder | None" = kv_recorder
        # Bus is per-provider and lives across calls. None preserves the
        # pre-step-5 behaviour byte-for-byte (no publish at all).
        self._attention_bus: "AttentionBus | None" = attention_bus
        # Attempt-level KV policy summary written to meta.json on
        # `finish_attempt`. Provider sets via `set_kv_policy_meta(...)`
        # before `start_attempt`. Default None = `--kv-policy none`.
        self._kv_policy_meta: dict[str, Any] | None = None
        # Symmetric slot for sparse_attention (see set_sparse_attention_meta).
        self._sparse_attention_meta: dict[str, Any] | None = None
        self._attempt_extra_meta: dict[str, Any] = {}
        # Optional manual override for `config.max_prefill_queries`. None = use
        # the frozen RecordingConfig value. H2O full-prefill scoring no longer
        # uses this path; it streams full rows in bounded chunks below so the
        # recording sample cap stays intact.
        self._max_prefill_queries_override: int | None = None

        # Sparse attention method instance (one per provider) + per-call
        # recorder. Pre-hooks read both via closure-bound getters so the
        # provider can swap the recorder between calls without re-registering
        # hooks. When `sparse_attention is None`, no pre-hooks are installed.
        self._sparse_attention: "BaseSparseAttention | None" = sparse_attention
        self._sparse_recorder: "SparseAttentionRecorder | None" = None
        # Append counter for `record_step`; recorder receives a globally-
        # monotone step id within the call. Cleared on each recorder swap.
        self._sparse_step_counter: int = 0
        self._sparse_layer_indices: tuple[int, ...] = ()
        self._sparse_hook_counts_by_layer: dict[int, int] = {}

        n_attention_modules = 0
        n_gate_modules = 0
        layer_indices: list[int] = []
        for name, module in model.named_modules():
            layer = _layer_index(name)
            if layer is not None:
                n_attention_modules += 1
                if layer >= 0:
                    layer_indices.append(layer)
                self._handles.append(
                    module.register_forward_hook(self._hook(layer), with_kwargs=True)
                )
            gate_layer = _gate_layer_index(name)
            if gate_layer is not None:
                n_gate_modules += 1
                self._handles.append(module.register_forward_hook(self._gate_hook(gate_layer)))
        if n_attention_modules == 0:
            raise ValueError("no attention modules matched '.layers.<n>.self_attn'")
        self._sparse_layer_indices = tuple(sorted(set(layer_indices)))
        # Sparse-attention pre-hooks are installed only when an active method
        # is configured. Each closure binds (layer_idx, method-getter,
        # recorder-getter, session-getter) so swapping the recorder per call
        # does not require unregistering anything.
        if self._sparse_attention is not None:
            for name, module in model.named_modules():
                layer = _layer_index(name)
                if layer is None or layer < 0:
                    continue
                self._handles.append(
                    module.register_forward_pre_hook(
                        self._sparse_pre_hook(layer),
                        with_kwargs=True,
                    )
                )
        if config.per_head_stats_layers and layer_indices:
            num_hidden_layers = max(layer_indices) + 1
            invalid = [i for i in config.per_head_stats_layers if i >= num_hidden_layers]
            assert not invalid, (
                f"per_head_stats_layers contains indices >= num_hidden_layers "
                f"({num_hidden_layers}): {invalid}"
            )
        self.model_summary["router_capture_mode"] = (
            "gate_forward_hook" if n_gate_modules else "none"
        )
        self._decode_records = DecodeRingBuffer(
            config.decode_window * max(1, n_attention_modules)
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def kv_recorder(self) -> "KVEvictionRecorder | None":
        """Currently-attached KV eviction recorder, or None."""
        return self._kv_recorder

    def set_kv_recorder(self, recorder: "KVEvictionRecorder | None") -> None:
        """Swap the KV recorder. Caller (provider) drives the per-call lifecycle.

        Not thread-safe; assumes HF inference runs single-threaded per provider
        (the project's actual usage).
        """
        self._kv_recorder = recorder

    def sparse_recorder(self) -> "SparseAttentionRecorder | None":
        """Currently-attached sparse-attention recorder, or None."""
        return self._sparse_recorder

    def set_sparse_recorder(
        self, recorder: "SparseAttentionRecorder | None"
    ) -> None:
        """Swap the sparse-attention recorder. Caller drives per-call lifecycle.

        Resetting the step counter here keeps `record_step` monotone within
        this recorder's lifetime; it is NOT joinable to
        `kv_eviction.npz.record_step` because the two recorders fire at
        different attention-pipeline stages and are mutually exclusive at
        runtime (one always has zero rows).

        Not thread-safe; assumes HF inference runs single-threaded per
        provider (the project's actual usage).
        """
        self._sparse_recorder = recorder
        self._sparse_step_counter = 0
        self._sparse_hook_counts_by_layer = {
            int(layer): 0 for layer in self._sparse_layer_indices
        }

    def set_kv_policy_meta(self, meta: dict[str, Any] | None) -> None:
        """Stash the attempt-level KV policy summary for `meta.json`.

        Provider populates this in `start_attempt` (or just before) so that
        `kv_policy.prefill_score_bias` is recorded once per attempt rather
        than per call. Pass None for `--kv-policy none`.
        """
        self._kv_policy_meta = dict(meta) if meta is not None else None

    def set_sparse_attention_meta(self, meta: dict[str, Any] | None) -> None:
        """Stash the attempt-level sparse-attention summary for `meta.json`.

        Mirrors `set_kv_policy_meta` so the `sparse_attention` block sits
        next to `kv_policy` in the rendered meta. Pass None for
        `--sparse-attn none`.
        """
        self._sparse_attention_meta = dict(meta) if meta is not None else None

    def set_attempt_extra_meta(self, meta: dict[str, Any]) -> None:
        """Merge provider-owned attempt metadata into the next meta.json write."""
        self._attempt_extra_meta.update(dict(meta))

    def start_attempt(self, recordings_dir: Path) -> None:
        self._attempt_dir = Path(recordings_dir)
        self._attempt_dir.mkdir(parents=True, exist_ok=True)
        self._attempt_extra_meta = {}
        self._meta = {
            "model": self.model_summary,
            "recording_config": asdict(self.config),
            "iters": [],
        }
        if getattr(self, "_kv_policy_meta", None) is not None:
            self._meta["kv_policy"] = dict(self._kv_policy_meta)
        if getattr(self, "_sparse_attention_meta", None) is not None:
            self._meta["sparse_attention"] = dict(self._sparse_attention_meta)

    def finish_attempt(self, trace_path: Path | None = None) -> None:
        if self._attempt_dir is None:
            return
        if self._attempt_extra_meta:
            self._meta.update(dict(self._attempt_extra_meta))
        if trace_path is not None and trace_path.exists():
            self._align_meta_to_trace(trace_path)
        (self._attempt_dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._attempt_dir = None

    def _align_meta_to_trace(self, trace_path: Path) -> None:
        trace_calls = _load_trace_llm_calls(trace_path)
        trace_indices = _trace_call_indices(trace_calls)
        original_iters = [dict(item) for item in self._meta.get("iters", [])]
        aligned: list[dict[str, Any]] = []
        orphaned: list[dict[str, Any]] = []
        missing_recording_iters: list[dict[str, Any]] = []
        trace_call_by_idx: dict[int, dict[str, Any]] = {}
        alignment_source = "unavailable"
        if trace_indices is not None:
            indices, alignment_source = trace_indices
            trace_call_by_idx = dict(zip(indices, trace_calls))
        for item in original_iters:
            call_idx = int(item.get("call_idx", -1))
            trace_call = trace_call_by_idx.get(call_idx)
            if trace_call is not None:
                item["trace_action_id"] = trace_call.get("action_id")
                item["trace_iteration"] = trace_call.get("iteration")
                aligned.append(item)
            else:
                item["trace_action_id"] = None
                item["trace_iteration"] = None
                item["trace_alignment_status"] = "orphan_no_llm_action"
                orphaned.append(item)
        recorded_call_idxs = {
            int(item.get("call_idx", -1))
            for item in original_iters
            if item.get("call_idx") is not None
        }
        for trace_idx, trace_call in sorted(trace_call_by_idx.items()):
            if int(trace_idx) in recorded_call_idxs:
                continue
            missing_recording_iters.append(
                {
                    "call_idx": int(trace_idx),
                    "trace_action_id": trace_call.get("action_id"),
                    "trace_iteration": trace_call.get("iteration"),
                    "trace_alignment_status": "missing_recording_iter",
                }
            )
        self._meta["iters"] = aligned
        if orphaned:
            self._meta["orphan_iters"] = orphaned
        if missing_recording_iters:
            self._meta["missing_recording_iters"] = missing_recording_iters
        self._meta["alignment"] = {
            "trace_path": str(trace_path),
            "trace_llm_actions": len(trace_calls),
            "recording_iters": len(original_iters),
            "aligned_iters": len(aligned),
            "orphan_iters": len(orphaned),
            "missing_recording_iters": len(missing_recording_iters),
            "alignment_source": alignment_source,
        }

    @contextmanager
    def recording_session(
        self,
        *,
        call_idx: int,
        segments: list[dict[str, Any]],
        input_token_count: int,
        generation: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if self._attempt_dir is None:
            raise RuntimeError("start_attempt() must be called before recording")
        if self._session is not None:
            raise RuntimeError("nested recording sessions are not supported")

        iter_dir = self._attempt_dir / f"iter_{call_idx:04d}"
        session_segments = []
        for segment in segments:
            payload = dict(segment)
            if "first_seen_call" not in payload:
                payload["first_seen_call"] = int(call_idx)
                payload["first_seen_call_inferred"] = True
            else:
                payload.setdefault("first_seen_call_inferred", False)
            session_segments.append(payload)
        self._session = {
            "call_idx": call_idx,
            "iter_dir": iter_dir,
            "segments": session_segments,
            "input_token_count": input_token_count,
            "attention_sampling": self._attention_sampling_metadata(
                call_idx=call_idx
            ),
            "started_at": time.time(),
            "generated_segment_id": len(segments),
            "generation": generation if generation is not None else None,
            "flushed": False,
        }
        self._prefill_records = []
        self._decode_records.clear()
        self._routing_records = []
        self._routing_seen_prefill.clear()
        self._routing_decode_steps.clear()
        self._head_stats = {}
        self._head_stats_n_segments = len(segments) + 1
        try:
            yield
        except BaseException:
            self._prefill_records = []
            self._decode_records.clear()
            self._routing_records = []
            self._routing_seen_prefill.clear()
            self._routing_decode_steps.clear()
            self._session = None
            raise
        finally:
            if self._session is not None and self._session.get("flushed"):
                self._session = None

    @contextmanager
    def suspend_attention(self) -> Iterator[None]:
        self._attention_suspended += 1
        try:
            yield
        finally:
            self._attention_suspended -= 1

    @contextmanager
    def unbounded_prefill_queries(self) -> Iterator[None]:
        """Temporarily disable the prefill query-row sample cap.

        This is retained for debugging and direct recording experiments. The
        paper-faithful H2O path does not use it because uncapping the recording
        rows can materialize a full QxK tensor on long prompts. Restores the
        previous override on exit; nesting is supported by stashing the prior
        value in a local.
        """
        previous = self._max_prefill_queries_override
        # `2**31 - 1` is what `select_query_positions` treats as "no cap"
        # since `query_len <= max_queries` short-circuits to the identity
        # range. Practical sequence lengths sit comfortably below this.
        self._max_prefill_queries_override = 2_147_483_647
        try:
            yield
        finally:
            self._max_prefill_queries_override = previous

    def _hook(self, layer: int):
        def capture(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            if self._session is None or self._attention_suspended:
                return
            self._capture_sampled_attention(layer, module, args, kwargs)

        return capture

    def _sparse_pre_hook(self, layer: int):
        """Pre-forward hook that ORs the sparse mask onto `attention_mask`.

        Bound once per layer at construction; closure-captures `layer`. The
        method instance and recorder live on `self` so the provider can swap
        the recorder per-call without re-registering hooks. Returns
        `(args, kwargs)` so the modified attention_mask reaches the wrapped
        forward. A None session (no `recording_session()` active) still
        applies the mask — the sparsity is part of the model semantics, not
        a recording concern.
        """

        def pre_hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            method = self._sparse_attention
            if method is None:
                return args, kwargs
            hidden_states = _arg_or_kw(args, kwargs, 0, "hidden_states")
            if hidden_states is None or hidden_states.ndim != 3:
                # Not a path we know how to mask; leave the forward untouched.
                return args, kwargs
            position_embeddings = _arg_or_kw(
                args, kwargs, 1, "position_embeddings", default=None
            )
            upstream_attention_mask = _arg_or_kw(
                args, kwargs, 2, "attention_mask", default=kwargs.get("attention_mask")
            )
            query_len = int(hidden_states.shape[-2])
            past_key_values = _arg_or_kw(
                args,
                kwargs,
                3,
                "past_key_values",
                default=kwargs.get("past_key_value"),
            )
            cached_len = 0
            if past_key_values is not None:
                try:
                    cached_len = int(past_key_values.get_seq_length(layer))
                except (AttributeError, TypeError):
                    cached = _cached_key_states(past_key_values, layer)
                    cached_len = 0 if cached is None else int(cached.shape[-2])
            key_len = cached_len + query_len
            # Phase rule mirrors LayerCapturer's: prefill if multi-token or
            # no session yet; decode if single-token AFTER input was consumed.
            input_tokens = (
                int(self._session["input_token_count"]) if self._session else 0
            )
            if query_len == 1 and key_len > input_tokens:
                phase = "decode"
                decode_step = max(0, key_len - input_tokens - 1)
            else:
                phase = "prefill"
                decode_step = -1
            context = SparseAttentionContext(
                module=module,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                attention_mask=upstream_attention_mask,
            )
            sparse_mask = method.build_additive_mask(
                layer_idx=layer,
                query_len=query_len,
                key_len=key_len,
                phase=phase,
                decode_step=decode_step,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                context=context,
            )
            observe_only = bool(method.observe_only)
            assert sparse_mask is not None or observe_only, (
                f"build_additive_mask returned None in enforce mode "
                f"(method={method.name!r}, layer={layer}, phase={phase!r}, "
                f"query_len={query_len}, key_len={key_len}). The None return "
                "is reserved for observe-only; enforce mode must materialize "
                "the sparse mask tensor."
            )
            if sparse_mask is not None and not observe_only:
                # Qwen3 decoder layers call self_attn with pure kwargs beyond
                # `hidden_states`; if a future HF version starts passing
                # attention_mask positionally, `existing is None` here will let
                # the materialize-fresh branch fire and overwrite the wrong
                # arg, surfacing fast in tests rather than silently corrupting.
                existing = kwargs.get("attention_mask")
                if existing is not None:
                    import torch

                    # Cast/broadcast onto the existing mask's device/dtype so the
                    # downstream `scores + mask` stays in a single dtype. We
                    # `expand` rather than `broadcast_to` so the resulting tensor
                    # carries the same query-dim as the existing mask (the
                    # downstream LayerCapturer relies on `mask.shape[-2] ==
                    # query_len` to slice sampled rows).
                    if existing.ndim == 4 and existing.shape[-2] >= 1:
                        sparse_for_add = sparse_mask.to(
                            device=existing.device, dtype=existing.dtype
                        ).expand(-1, -1, existing.shape[-2], -1)
                    else:
                        sparse_for_add = sparse_mask.to(
                            device=existing.device, dtype=existing.dtype
                        )
                    # Sparse mask is a "force-mask" not a weighted decrement:
                    # positions already masked by the upstream causal mask plus
                    # sparse should stay at finfo.min, not double-saturate to
                    # -inf which can overflow fp16 and produce NaN in some SDPA
                    # backends.
                    neg_inf = torch.finfo(existing.dtype).min
                    sparse_masked = sparse_for_add < 0
                    kwargs["attention_mask"] = torch.where(
                        sparse_masked,
                        torch.full_like(existing, neg_inf),
                        existing,
                    )
                else:
                    # No upstream attention_mask (HF's SDPA path may rely on its
                    # implicit causal mask). Materialise a [1,1,Q,K] sparse-only
                    # mask so the LayerCapturer's `mask.shape[-2] == query_len`
                    # contract is satisfied without depending on broadcast.
                    kwargs["attention_mask"] = sparse_mask.expand(
                        1, 1, query_len, key_len
                    ).contiguous()
            # else: observe_only — sparse_mask is None by contract;
            # kwargs["attention_mask"] left untouched, SDPA uses implicit causal.
            if self._session is not None:
                self._sparse_hook_counts_by_layer[layer] = (
                    self._sparse_hook_counts_by_layer.get(layer, 0) + 1
                )
            recorder = self._sparse_recorder
            # Apply mask semantics always, but only RECORD rows while a real
            # chat session is active. Warmup forwards (session is None) have
            # no `input_token_count`, so the phase derived above is unreliable
            # (Q==1 warmup forwards would be misfiled as "decode"); dropping
            # the row avoids corrupting `sparse_attention.npz` semantics.
            if recorder is not None and self._session is not None:
                kept = (
                    method.kept_count(key_len)
                    if hasattr(method, "kept_count")
                    else key_len
                )
                extras = method.record_metadata(
                    layer_idx=layer,
                    phase=phase,
                    decode_step=decode_step,
                )
                effective_counter = getattr(
                    method, "effective_kept_count_sum", None
                )
                if callable(effective_counter):
                    effective_kept = int(
                        effective_counter(query_len=query_len, key_len=key_len)
                    )
                    extras = dict(extras)
                    extras["effective_kept_count_sum"] = effective_kept
                    denom = int(query_len) * int(key_len)
                    extras["effective_density"] = (
                        float(effective_kept) / float(denom) if denom > 0 else 0.0
                    )
                recorder.append(
                    step=self._sparse_step_counter,
                    layer=layer,
                    phase=phase,
                    decode_step=decode_step,
                    query_len=query_len,
                    key_len=key_len,
                    kept_count=int(kept),
                    extras=extras,
                )
                self._sparse_step_counter += 1
            return args, kwargs

        return pre_hook

    def _sparse_attention_integrity(self, *, sparse_records: int) -> dict[str, Any]:
        """Build sparse-attention hook/recording integrity metadata."""
        counts = [
            int(self._sparse_hook_counts_by_layer.get(layer, 0))
            for layer in self._sparse_layer_indices
        ]
        hook_invocations = int(sum(counts))
        recording_enabled = self._sparse_recorder is not None
        expected_records = hook_invocations if recording_enabled else 0
        observed_layers = sum(1 for count in counts if count > 0)
        min_hooks = min(counts) if counts else 0
        max_hooks = max(counts) if counts else 0
        return {
            "sparse_attention_recording_enabled": bool(recording_enabled),
            "sparse_attention_observe_only": bool(
                self._sparse_attention.observe_only
                if self._sparse_attention is not None
                else False
            ),
            "sparse_attention_records": int(sparse_records),
            "sparse_attention_expected_records": int(expected_records),
            "sparse_attention_records_match_expected": bool(
                int(sparse_records) == int(expected_records)
            ),
            "sparse_attention_expected_layers": int(len(counts)),
            "sparse_attention_observed_layers": int(observed_layers),
            "sparse_attention_hook_invocations": hook_invocations,
            "sparse_attention_hooks_per_layer_min": int(min_hooks),
            "sparse_attention_hooks_per_layer_max": int(max_hooks),
            "sparse_attention_hooks_balanced": bool(min_hooks == max_hooks),
        }

    def _attention_sampling_metadata(
        self,
        *,
        call_idx: int,
        query_len: int | None = None,
        sampled_query_count: int | None = None,
        unbounded: bool = False,
    ) -> dict[str, Any]:
        seed = query_sampling_seed(self.config.generation_seed, call_idx)
        configured_max = int(self.config.max_prefill_queries)
        if query_len is None or sampled_query_count is None:
            return {
                "prefill_query_sampler": "stratified_seeded_jitter",
                "prefill_query_seed": seed,
                "configured_max_prefill_queries": configured_max,
                "effective_max_prefill_queries": configured_max,
                "unbounded_prefill_queries": False,
                "prefill_query_count": None,
                "sampled_prefill_queries": None,
            }

        all_rows = int(sampled_query_count) == int(query_len)
        return {
            "prefill_query_sampler": (
                "all_rows" if all_rows else "stratified_seeded_jitter"
            ),
            "prefill_query_seed": None if all_rows else seed,
            "configured_max_prefill_queries": configured_max,
            "effective_max_prefill_queries": None
            if unbounded
            else configured_max,
            "unbounded_prefill_queries": bool(unbounded),
            "prefill_query_count": int(query_len),
            "sampled_prefill_queries": int(sampled_query_count),
        }

    def _gate_hook(self, layer: int):
        def capture(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            if self._session is None:
                return
            self._record_router_tensor(layer=layer, path=("gate",), tensor=output)

        return capture

    def _token_ids_for_key_len(self, key_len: int):
        assert self._session is not None
        return token_segment_ids(
            key_len,
            self._session["segments"],
            generated_segment_id=self._session["generated_segment_id"],
        )

    def _capture_sampled_attention(
        self,
        layer: int,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden_states = _arg_or_kw(args, kwargs, 0, "hidden_states")
        position_embeddings = _arg_or_kw(args, kwargs, 1, "position_embeddings")
        attention_mask = _arg_or_kw(args, kwargs, 2, "attention_mask", default=None)
        past_key_values = _arg_or_kw(
            args,
            kwargs,
            3,
            "past_key_values",
            default=kwargs.get("past_key_value"),
        )
        if hidden_states is None or position_embeddings is None:
            raise ValueError("attention hook requires hidden_states and position_embeddings")
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be rank 3, got {hidden_states.shape}")

        assert self._session is not None
        query_len = int(hidden_states.shape[-2])
        # Manual override path: when `unbounded_prefill_queries()` is active,
        # treat every query row as sampled. Normal H2O full-prefill scoring
        # leaves this cap bounded and streams full rows only to H2O consumers.
        max_queries = (
            self._max_prefill_queries_override
            if self._max_prefill_queries_override is not None
            else self.config.max_prefill_queries
        )
        row_indices = (
            [0]
            if query_len == 1
            else select_query_positions(
                query_len,
                max_queries,
                seed=query_sampling_seed(
                    self.config.generation_seed,
                    int(self._session["call_idx"]),
                ),
            )
        )
        key_states = self._key_states(
            module=module,
            layer=layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )
        key_len = int(key_states.shape[-2])
        phase = (
            "decode"
            if query_len == 1 and key_len > self._session["input_token_count"]
            else "prefill"
        )
        if phase == "prefill":
            self._session["attention_sampling"] = self._attention_sampling_metadata(
                call_idx=int(self._session["call_idx"]),
                query_len=query_len,
                sampled_query_count=len(row_indices),
                unbounded=self._max_prefill_queries_override is not None,
            )
        query_positions = [key_len - query_len + idx for idx in row_indices]
        capture_head_stats = (
            layer >= 0
            and bool(self.config.per_head_stats_layers)
            and layer in self.config.per_head_stats_layers
        )
        if capture_head_stats and phase == "prefill":
            existing = self._head_stats.get((layer, "prefill", 0))
            if existing is not None and existing["n_queries"] > 0:
                raise RuntimeError(
                    f"layer {layer}: second prefill capture detected — "
                    "multi-prefill per session is unsupported for per-head stats"
                )
        use_full_prefill_bus = (
            phase == "prefill"
            and self._attention_bus is not None
            and self._attention_bus.has_full_prefill_consumers()
        )
        if use_full_prefill_bus:
            attn_rows = self._full_prefill_attention_rows(
                module=module,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                key_states=key_states,
                row_indices=row_indices,
                layer=layer,
                capture_head_stats=capture_head_stats,
                token_ids_full=self._token_ids_for_key_len(key_len),
                n_segments=self._session["generated_segment_id"] + 1,
            )
            attn_full_sampled = None
        else:
            attn_full_sampled, attn_rows = self._sampled_attention_rows(
                module=module,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                key_states=key_states,
                row_indices=row_indices,
                query_positions=query_positions,
                layer=layer,
            )
        token_ids = self._token_ids_for_key_len(key_len).to(device=attn_rows.device)
        n_segments = self._session["generated_segment_id"] + 1

        if capture_head_stats and not use_full_prefill_bus:
            if phase == "decode":
                if key_len <= self._session["input_token_count"]:
                    raise RuntimeError(
                        f"layer {layer}: decode attention at key_len={key_len} <= "
                        f"input_token_count={self._session['input_token_count']}; "
                        "unexpected KV boundary for per-head stats"
                    )
                decode_step = key_len - self._session["input_token_count"] - 1
            else:
                decode_step = 0
            # attn_full_sampled: [1, H, Q, K] — already computed by _sampled_attention_rows
            self._accumulate_head_stats(
                layer_idx=layer,
                phase=phase,
                decode_step=decode_step,
                attn=attn_full_sampled[0],
                token_ids=token_ids,
                n_segments=n_segments,
            )

        segment_mass = segment_bucket(attn_rows, token_ids, n_segments)
        top_indices, top_weights = padded_top_k(attn_rows, self.config.attention_top_k)
        hitter_indices, hitter_weights = heavy_hitter(attn_rows, self.config.attention_top_k)
        n_heads = int(attn_rows.shape[0] // len(row_indices))
        head_indices = np.repeat(
            np.arange(n_heads, dtype=np.int32), len(row_indices)
        )
        record = {
            "layer": layer,
            "phase": phase,
            "decode_step": max(0, key_len - self._session["input_token_count"] - 1)
            if phase == "decode"
            else -1,
            "query_heads": head_indices,
            "query_positions": np.tile(
                np.asarray(query_positions, dtype=np.int32), n_heads
            ),
            "segment_mass": _stage_numpy(segment_mass, np.float32),
            "topk_indices": _stage_numpy(top_indices, np.int32),
            "topk_weights": _stage_numpy(top_weights, np.float32),
            "heavy_indices": _stage_numpy(hitter_indices[0], np.int32),
            "heavy_weights": _stage_numpy(hitter_weights[0], np.float32),
        }
        if phase == "decode":
            self._decode_records.append(record)
        else:
            self._prefill_records.append(record)

    def _accumulate_head_stats(
        self,
        *,
        layer_idx: int,
        phase: str,
        decode_step: int,
        attn: Any,
        token_ids: Any,
        n_segments: int,
    ) -> None:
        import torch

        # Fix 7: guard against segment count changing mid-session.
        if self._head_stats and n_segments != self._head_stats_n_segments:
            raise RuntimeError(
                f"segment count changed mid-session: expected "
                f"{self._head_stats_n_segments}, got {n_segments}"
            )

        # attn: [H, Q, K]
        H = int(attn.shape[0])
        Q = int(attn.shape[1])
        K = int(attn.shape[2])
        S = n_segments
        key_ids = token_ids[:K].to(device=attn.device, dtype=torch.long)
        key_float = attn.detach().to(dtype=torch.float32)

        if phase == "prefill":
            key = (layer_idx, "prefill", 0)
            entry = self._head_stats.get(key)
            if entry is None:
                entry = {
                    "mean_sum": torch.zeros((H, S), dtype=torch.float32, device=attn.device),
                    "var_sum": torch.zeros((H, S), dtype=torch.float32, device=attn.device),
                    # kept_count_sum[s] = sum of mask.sum() across all query rows for segment s
                    "kept_count_sum": torch.zeros(S, dtype=torch.int32, device=attn.device),
                    "n_queries": 0,
                }
                self._head_stats[key] = entry
            for s in range(S):
                mask = key_ids == s
                if not bool(mask.any()):
                    continue
                vals = key_float[:, :, mask]  # [H, Q, |S|]
                entry["mean_sum"][:, s] += vals.mean(dim=-1).sum(dim=-1)
                entry["var_sum"][:, s] += vals.var(dim=-1, unbiased=False).sum(dim=-1)
                # mask.sum() counts key positions in this segment; same for all Q rows
                entry["kept_count_sum"][s] += int(mask.sum().item()) * Q
            entry["n_queries"] = entry["n_queries"] + Q
        else:
            key = (layer_idx, "decode", decode_step)
            entry_new: dict[str, Any] = {
                "mean": torch.zeros((H, S), dtype=torch.float32, device=attn.device),
                "var": torch.zeros((H, S), dtype=torch.float32, device=attn.device),
                # kept_count[s] = number of key positions in segment s at this decode step
                "kept_count": torch.zeros(S, dtype=torch.int32, device=attn.device),
            }
            for s in range(S):
                mask = key_ids == s
                if not bool(mask.any()):
                    continue
                vals = key_float[:, :, mask]  # [H, 1, |S|]
                entry_new["mean"][:, s] = vals.mean(dim=-1).squeeze(-1)
                entry_new["var"][:, s] = vals.var(dim=-1, unbiased=False).squeeze(-1)
                entry_new["kept_count"][s] = int(mask.sum().item())
            self._head_stats[key] = entry_new

    def _key_states(
        self,
        *,
        module: Any,
        layer: int,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        past_key_values: Any,
    ) -> Any:
        layer_idx = int(getattr(module, "layer_idx", layer))
        cached = _cached_key_states(past_key_values, layer_idx)
        if cached is not None:
            return cached
        key_states = _project_key_states(module, hidden_states)
        return _apply_rotary_to_states(key_states, position_embeddings)

    def _sampled_attention_rows(
        self,
        *,
        module: Any,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
        key_states: Any,
        row_indices: list[int],
        query_positions: list[int],
        layer: int,
        full_prefill: bool = False,
    ) -> tuple[Any, Any]:
        """Return (attn_full, attn_rows) where attn_full is [1, H, Q, K] pre-head-mean."""
        attn = self._attention_tensor(
            module=module,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            key_states=key_states,
            row_indices=row_indices,
            query_positions=query_positions,
        )
        self._publish_attention(
            layer=layer,
            attn=attn,
            query_positions=query_positions,
            key_len=int(key_states.shape[-2]),
            full_prefill=full_prefill,
        )
        return attn, attn[0].reshape(attn.shape[1] * len(row_indices), int(key_states.shape[-2]))

    def _full_prefill_attention_rows(
        self,
        *,
        module: Any,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
        key_states: Any,
        row_indices: list[int],
        layer: int,
        capture_head_stats: bool = False,
        token_ids_full: Any = None,
        n_segments: int = 0,
    ) -> Any:
        """Publish all prefill rows to full-prefill consumers in bounded chunks.

        This gives H2O paper-faithful prefill scores without constructing a
        full `(heads, query_len, key_len)` attention tensor at once. We retain
        only the normal sampled rows for `attention.npz`, preserving the bounded
        recording footprint.

        When capture_head_stats=True, accumulates per-head stats inline from
        each chunk's already-computed attention tensor (no extra GPU work).
        """
        import torch

        query_len = int(hidden_states.shape[-2])
        key_len = int(key_states.shape[-2])
        if query_len <= 0:
            raise ValueError(f"query_len must be positive, got {query_len}")
        chunk_size = max(1, int(self.config.max_prefill_queries))
        selected: dict[int, Any] = {}
        selected_set = set(int(idx) for idx in row_indices)
        base_position = key_len - query_len
        token_ids_dev: Any = None

        for start in range(0, query_len, chunk_size):
            stop = min(query_len, start + chunk_size)
            chunk_rows = list(range(start, stop))
            chunk_positions = [base_position + idx for idx in chunk_rows]
            attn = self._attention_tensor(
                module=module,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                key_states=key_states,
                row_indices=chunk_rows,
                query_positions=chunk_positions,
            )
            self._publish_attention(
                layer=layer,
                attn=attn,
                query_positions=chunk_positions,
                key_len=key_len,
                full_prefill=True,
            )
            if capture_head_stats:
                if token_ids_dev is None:
                    token_ids_dev = token_ids_full.to(device=attn.device)
                self._accumulate_head_stats(
                    layer_idx=layer,
                    phase="prefill",
                    decode_step=0,
                    attn=attn[0],
                    token_ids=token_ids_dev,
                    n_segments=n_segments,
                )
            local_selected = [
                (row, row - start) for row in chunk_rows if row in selected_set
            ]
            for row, local_idx in local_selected:
                selected[int(row)] = attn[:, :, local_idx : local_idx + 1, :].detach()

        if set(selected) != selected_set:
            missing = sorted(selected_set.difference(selected))
            raise RuntimeError(f"failed to retain sampled prefill rows: {missing}")
        sampled = torch.cat([selected[int(idx)] for idx in row_indices], dim=2)
        return sampled[0].reshape(sampled.shape[1] * len(row_indices), key_len)

    def _attention_tensor(
        self,
        *,
        module: Any,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
        key_states: Any,
        row_indices: list[int],
        query_positions: list[int],
    ) -> Any:
        import torch

        q_states = _current_query_states(
            module=module,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            row_indices=row_indices,
        )
        key_len = int(key_states.shape[-2])
        n_kv_heads = int(key_states.shape[1])
        n_query_heads = int(q_states.shape[1])
        if n_query_heads % n_kv_heads != 0:
            raise ValueError(
                f"query heads must be divisible by kv heads: {n_query_heads} vs {n_kv_heads}"
            )
        groups = n_query_heads // n_kv_heads
        grouped_q = q_states.reshape(
            q_states.shape[0],
            n_kv_heads,
            groups,
            q_states.shape[-2],
            q_states.shape[-1],
        )
        scores = torch.matmul(grouped_q, key_states.unsqueeze(2).transpose(-1, -2))
        scores = scores.reshape(
            q_states.shape[0],
            n_query_heads,
            q_states.shape[-2],
            key_len,
        )
        scores = scores * float(getattr(module, "scaling", q_states.shape[-1] ** -0.5))
        scores = self._apply_attention_mask(
            scores=scores,
            attention_mask=attention_mask,
            row_indices=row_indices,
            query_positions=query_positions,
            query_len=int(hidden_states.shape[-2]),
            key_len=key_len,
        )
        attn = torch.softmax(scores, dim=-1)
        return attn

    def _publish_attention(
        self,
        *,
        layer: int,
        attn: Any,
        query_positions: list[int],
        key_len: int,
        full_prefill: bool,
    ) -> None:
        import torch

        # Publish post-softmax, pre-reshape: shape is the documented
        # (B, num_q_heads, n_query_rows, key_len) the bus contract requires.
        # H2O (step 6) consumes this exact tensor; the capturer's own reduce
        # below stays unchanged so attention.npz bytes don't move when no
        # subscriber is attached.
        if self._attention_bus is not None:
            phase = (
                "decode"
                if (
                    len(query_positions) == 1
                    and self._session is not None
                    and key_len > int(self._session["input_token_count"])
                )
                else "prefill"
            )
            query_pos_tensor = torch.as_tensor(
                query_positions, dtype=torch.long, device=attn.device
            )
            self._attention_bus.publish(
                layer=layer,
                attn=attn,
                query_positions=query_pos_tensor,
                key_len=key_len,
                phase=phase,
                suspended=self._attention_suspended > 0,
                full_prefill=full_prefill,
            )

    def _apply_attention_mask(
        self,
        *,
        scores: Any,
        attention_mask: Any,
        row_indices: list[int],
        query_positions: list[int],
        query_len: int,
        key_len: int,
    ) -> Any:
        import torch

        if attention_mask is not None:
            mask = attention_mask[:, :, :, :key_len].to(
                device=scores.device, dtype=scores.dtype
            )
            if int(mask.shape[-2]) == query_len:
                index = torch.as_tensor(row_indices, dtype=torch.long, device=mask.device)
                mask = mask.index_select(-2, index)
            elif int(mask.shape[-2]) != len(row_indices):
                raise ValueError(
                    f"unsupported attention mask query dimension: {mask.shape}"
                )
            return scores + mask

        key_positions = torch.arange(key_len, device=scores.device)
        query_pos = torch.as_tensor(
            query_positions, dtype=torch.long, device=scores.device
        )
        blocked = key_positions.view(1, 1, 1, key_len) > query_pos.view(
            1, 1, len(query_positions), 1
        )
        return scores.masked_fill(blocked, torch.finfo(scores.dtype).min)

    def _advance_routing_step(self, layer: int, *, n_tokens: int) -> tuple[str, int]:
        if layer < 0:
            return "mixed", -1
        if n_tokens > 1 or layer not in self._routing_seen_prefill:
            self._routing_seen_prefill.add(layer)
            self._routing_decode_steps[layer] = 0
            return "prefill", -1
        step = self._routing_decode_steps.get(layer, 0)
        self._routing_decode_steps[layer] = step + 1
        return "decode", step

    def _advance_routing_decode_step(self, layer: int) -> tuple[str, int]:
        if layer < 0:
            return "mixed", -1
        step = self._routing_decode_steps.get(layer, 0)
        self._routing_seen_prefill.add(layer)
        self._routing_decode_steps[layer] = step + 1
        return "decode", step

    def _routing_phase_for_logits(
        self,
        *,
        layer: int,
        n_tokens: int,
        total_tokens: int,
        nested_path: tuple[int, ...],
    ) -> tuple[str, int]:
        assert self._session is not None
        input_tokens = int(self._session["input_token_count"])
        if n_tokens == int(total_tokens):
            return "mixed", -1
        if n_tokens <= input_tokens:
            return self._advance_routing_step(layer, n_tokens=n_tokens)
        return "mixed", -1

    def record_router_logits(self, outputs: Any, *, total_tokens: int) -> None:
        router_logits = getattr(outputs, "router_logits", None)
        if router_logits is None and isinstance(outputs, dict):
            router_logits = outputs.get("router_logits")
        if router_logits is None or self._session is None:
            return

        n_segments = self._session["generated_segment_id"] + 1
        top_k_experts = int(
            self.model_summary.get("num_experts_per_tok")
            or self.model_summary.get("num_experts_per_token")
            or 1
        )
        for path, tensor in self._iter_tensors(router_logits):
            if tensor.ndim < 2:
                continue
            logits = tensor.reshape(-1, tensor.shape[-1])
            layer = int(path[-1]) if path else -1
            phase, decode_step = self._routing_phase_for_logits(
                layer=layer,
                n_tokens=int(logits.shape[0]),
                total_tokens=total_tokens,
                nested_path=path,
            )
            token_ids = self._routing_token_ids(
                n_tokens=int(logits.shape[0]),
                total_tokens=total_tokens,
                nested_path=path,
                phase=phase,
            ).to(device=logits.device)
            choices, weights, load = expert_load_per_segment(
                logits,
                token_ids,
                n_segments=n_segments,
                top_k_experts=top_k_experts,
            )
            routing_counts = _routing_count_summary(
                choices,
                token_ids,
                n_segments=n_segments,
                n_experts=int(logits.shape[-1]),
            )
            self._routing_records.append(
                {
                    "path": ".".join(str(part) for part in path),
                    "layer": layer,
                    "phase": phase,
                    "decode_step": decode_step,
                    "expert_choice": _stage_numpy(choices, np.int32),
                    "expert_weight": _stage_numpy(weights, np.float32),
                    "expert_load": _stage_numpy(load, np.float32),
                    **routing_counts,
                }
            )

    def _record_router_tensor(
        self,
        *,
        layer: int,
        path: tuple[str, ...],
        tensor: Any,
    ) -> None:
        if self._session is None or tensor is None or tensor.ndim < 2:
            return
        n_segments = self._session["generated_segment_id"] + 1
        top_k_experts = int(
            self.model_summary.get("num_experts_per_tok")
            or self.model_summary.get("num_experts_per_token")
            or 1
        )
        logits = tensor.reshape(-1, tensor.shape[-1])
        phase, decode_step = self._advance_routing_step(
            layer=layer,
            n_tokens=int(logits.shape[0]),
        )
        token_ids = self._gate_token_ids(
            n_tokens=int(logits.shape[0]),
            phase=phase,
        ).to(
            device=logits.device
        )
        choices, weights, load = expert_load_per_segment(
            logits,
            token_ids,
            n_segments=n_segments,
            top_k_experts=top_k_experts,
        )
        routing_counts = _routing_count_summary(
            choices,
            token_ids,
            n_segments=n_segments,
            n_experts=int(logits.shape[-1]),
        )
        self._routing_records.append(
            {
                "path": ".".join(path),
                "layer": layer,
                "phase": phase,
                "decode_step": decode_step,
                "expert_choice": _as_numpy(choices).astype(np.int32),
                "expert_weight": _as_numpy(weights).astype(np.float32),
                "expert_load": _as_numpy(load).astype(np.float32),
                **routing_counts,
            }
        )

    def _current_input_token_ids(self, *, n_tokens: int):
        assert self._session is not None
        input_tokens = int(self._session["input_token_count"])
        if n_tokens < 0 or n_tokens > input_tokens:
            raise ValueError(
                f"cannot map {n_tokens} routing rows onto {input_tokens} input tokens"
            )
        token_ids = token_segment_ids(
            input_tokens,
            self._session["segments"],
            generated_segment_id=None,
        )
        return token_ids[input_tokens - n_tokens : input_tokens]

    def _generated_token_ids(self, *, n_tokens: int):
        assert self._session is not None

        import torch

        return torch.full(
            (n_tokens,),
            self._session["generated_segment_id"],
            dtype=torch.long,
        )

    def _gate_token_ids(self, *, n_tokens: int, phase: str):
        if phase == "prefill":
            return self._current_input_token_ids(n_tokens=n_tokens)
        if phase == "decode":
            return self._generated_token_ids(n_tokens=n_tokens)
        return self._generated_token_ids(n_tokens=n_tokens)

    def _routing_token_ids(
        self,
        *,
        n_tokens: int,
        total_tokens: int,
        nested_path: tuple[int, ...],
        phase: str,
    ):
        assert self._session is not None
        if phase == "prefill":
            return self._current_input_token_ids(n_tokens=n_tokens)
        if phase == "decode":
            return self._generated_token_ids(n_tokens=n_tokens)
        if n_tokens == total_tokens:
            return token_segment_ids(
                total_tokens,
                self._session["segments"],
                generated_segment_id=self._session["generated_segment_id"],
            )
        if len(nested_path) >= 2 and n_tokens <= 1:
            return self._generated_token_ids(n_tokens=n_tokens)
        return self._token_ids_for_key_len(n_tokens)

    def _iter_tensors(self, value: Any, path: tuple[int, ...] = ()):
        import torch

        if torch.is_tensor(value):
            yield path, value
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                yield from self._iter_tensors(child, (*path, idx))

    def flush(self, *, output_token_ids: list[int]) -> None:
        if self._session is None:
            raise RuntimeError("no active recording session")
        if not self._prefill_records and len(self._decode_records) == 0:
            raise RuntimeError("no attention records captured")
        input_tokens = int(self._session["input_token_count"])
        total_tokens = input_tokens + len(output_token_ids)
        segments = [dict(segment) for segment in self._session["segments"]]
        segments.append(
            {
                "role": "generation",
                "message_index": None,
                "token_start": input_tokens,
                "token_end": total_tokens,
                "has_content": bool(output_token_ids),
                "has_tool_calls": False,
                "first_seen_call": int(self._session["call_idx"]),
                "first_seen_call_inferred": False,
            }
        )
        token_ids = token_segment_ids(
            total_tokens,
            segments,
            generated_segment_id=self._session["generated_segment_id"],
        ).tolist()
        decode_records_dropped = self._decode_records.dropped_count()

        iter_dir: Path = self._session["iter_dir"]
        tmp_dir = iter_dir.with_name(f".{iter_dir.name}.tmp-{time.time_ns()}")
        tmp_dir.mkdir(parents=True)
        try:
            (tmp_dir / "segments.json").write_text(
                json.dumps(
                    {
                        "call_idx": self._session["call_idx"],
                        "input_tokens": input_tokens,
                        "output_tokens": len(output_token_ids),
                        "total_tokens": total_tokens,
                        "complete": True,
                        "segments": segments,
                        "token_segment_id": token_ids,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            self._write_attention_npz(tmp_dir / "attention.npz", segments)
            self._write_routing_npz(tmp_dir / "routing.npz", len(segments))
            if self._kv_recorder is not None and self._kv_recorder.n_records() > 0:
                # KV eviction npz lives alongside attention.npz / routing.npz so
                # downstream loaders can join on (call_idx, layer, decode_step).
                self._kv_recorder.write(tmp_dir / "kv_eviction.npz")
            sparse_records = (
                self._sparse_recorder.n_records()
                if self._sparse_recorder is not None
                else 0
            )
            if self._sparse_recorder is not None and sparse_records > 0:
                # Sparse attention npz mirrors kv_eviction.npz placement so the
                # two artifacts share the same per-iter directory contract.
                self._sparse_recorder.write(tmp_dir / "sparse_attention.npz")
            (tmp_dir / ".done").write_text("complete\n", encoding="utf-8")

            if iter_dir.exists():
                if (iter_dir / ".done").exists():
                    raise FileExistsError(
                        f"complete iter directory already exists: {iter_dir}"
                    )
                shutil.rmtree(iter_dir)
            tmp_dir.replace(iter_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        iter_meta = {
            "call_idx": self._session["call_idx"],
            "dir": iter_dir.name,
            "input_tokens": input_tokens,
            "output_tokens": len(output_token_ids),
            "total_tokens": total_tokens,
            "n_segments": len(segments),
            "attention_records": len(self._prefill_records)
            + len(self._decode_records),
            "routing_records": len(self._routing_records),
            "decode_records_dropped_count": decode_records_dropped,
            "recording_integrity": {
                "complete": True,
                "done_sentinel": True,
                "decode_records_dropped_count": decode_records_dropped,
            },
            "attention_sampling": dict(self._session["attention_sampling"]),
            "elapsed_s": time.time() - float(self._session["started_at"]),
        }
        if self._sparse_attention is not None:
            # Only annotate the integrity block when sparse attention is wired
            # so existing recording-only test fixtures stay byte-stable.
            iter_meta["recording_integrity"].update(
                self._sparse_attention_integrity(sparse_records=sparse_records)
            )
        generation = self._session.get("generation")
        if generation is not None:
            iter_meta["generation"] = dict(generation)
        iter_meta["per_head_stats_layers"] = list(self.config.per_head_stats_layers)
        self._meta["iters"].append(iter_meta)
        self._session["flushed"] = True

    def _build_head_span_arrays(
        self, n_segments: int
    ) -> dict[str, np.ndarray]:
        import torch

        layers = sorted(self.config.per_head_stats_layers)
        L_s = len(layers)
        # Determine H from any stored entry, fall back to 0 if nothing recorded
        H = 0
        for entry in self._head_stats.values():
            tensor = entry.get("mean_sum") if "mean_sum" in entry else entry.get("mean")
            if tensor is not None:
                H = int(tensor.shape[0])
                break
        S = n_segments

        # Prefill arrays — NaN where no key positions contributed (Fix 1)
        prefill_mean = np.full((L_s, H, S), np.nan, dtype=np.float32)
        prefill_var = np.full((L_s, H, S), np.nan, dtype=np.float32)
        prefill_query_count = np.int32(0)
        # kept_count_prefill[L_s, S]: total key positions summed over all Q rows (Fix 2)
        prefill_kept_count = np.zeros((L_s, S), dtype=np.int32)

        for li, layer in enumerate(layers):
            key = (layer, "prefill", 0)
            entry = self._head_stats.get(key)
            if entry is None or entry["n_queries"] == 0:
                continue
            n = int(entry["n_queries"])
            mean_sum = entry["mean_sum"]
            var_sum = entry["var_sum"]
            kept_count_sum = entry["kept_count_sum"]
            if isinstance(mean_sum, torch.Tensor):
                mean_sum = mean_sum.detach().cpu().numpy()
            if isinstance(var_sum, torch.Tensor):
                var_sum = var_sum.detach().cpu().numpy()
            if isinstance(kept_count_sum, torch.Tensor):
                kept_count_sum = kept_count_sum.detach().cpu().numpy()
            # Only fill cells that had at least one key position; rest stay NaN
            has_keys = kept_count_sum > 0  # [S]
            mean_divided = mean_sum / n  # [H, S]
            var_divided = var_sum / n    # [H, S]
            for s in range(S):
                if has_keys[s]:
                    prefill_mean[li, :, s] = mean_divided[:, s].astype(np.float16).astype(np.float32)
                    prefill_var[li, :, s] = var_divided[:, s]
            prefill_kept_count[li] = kept_count_sum
            prefill_query_count = np.int32(n)

        # Decode arrays: find T_max per layer
        decode_keys_by_layer: dict[int, list[int]] = {layer: [] for layer in layers}
        for key in self._head_stats:
            l_idx, phase, step = key
            if phase == "decode" and l_idx in decode_keys_by_layer:
                decode_keys_by_layer[l_idx].append(step)

        decode_counts = [len(decode_keys_by_layer[layer_id]) for layer_id in layers]
        T_max = max(decode_counts) if decode_counts else 0

        # NaN-initialize decode mean/var — cells with no key positions stay NaN (Fix 1)
        if T_max > 0:
            decode_mean = np.full((L_s, T_max, H, S), np.nan, dtype=np.float32)
            decode_var = np.full((L_s, T_max, H, S), np.nan, dtype=np.float32)
            decode_step_arr = np.full((L_s, T_max), -1, dtype=np.int32)
            decode_kept_count = np.zeros((L_s, T_max, S), dtype=np.int32)
        else:
            decode_mean = np.full((L_s, 0, H, S), np.nan, dtype=np.float32)
            decode_var = np.full((L_s, 0, H, S), np.nan, dtype=np.float32)
            decode_step_arr = np.zeros((L_s, 0), dtype=np.int32)
            decode_kept_count = np.zeros((L_s, 0, S), dtype=np.int32)
        decode_n = np.asarray(decode_counts, dtype=np.int32)

        for li, layer in enumerate(layers):
            steps = sorted(decode_keys_by_layer[layer])
            for ti, step in enumerate(steps):
                key = (layer, "decode", step)
                entry = self._head_stats.get(key)
                if entry is None:
                    continue
                mean_arr = entry["mean"]
                var_arr = entry["var"]
                kept_count_arr = entry["kept_count"]
                if isinstance(mean_arr, torch.Tensor):
                    mean_arr = mean_arr.detach().cpu().numpy()
                if isinstance(var_arr, torch.Tensor):
                    var_arr = var_arr.detach().cpu().numpy()
                if isinstance(kept_count_arr, torch.Tensor):
                    kept_count_arr = kept_count_arr.detach().cpu().numpy()
                has_keys = kept_count_arr > 0  # [S]
                for s in range(S):
                    if has_keys[s]:
                        decode_mean[li, ti, :, s] = mean_arr[:, s].astype(np.float16).astype(np.float32)
                        decode_var[li, ti, :, s] = var_arr[:, s]
                decode_kept_count[li, ti] = kept_count_arr
                decode_step_arr[li, ti] = step

        return {
            "head_stats_layers": np.asarray(layers, dtype=np.int32),
            "head_span_mean_prefill": prefill_mean.astype(np.float16),
            "head_span_var_prefill": prefill_var.astype(np.float32),
            "head_span_query_count": prefill_query_count,
            "head_span_mean_decode": decode_mean.astype(np.float16),
            "head_span_var_decode": decode_var.astype(np.float32),
            "head_span_decode_step": decode_step_arr,
            "head_span_decode_n": decode_n,
            # Fix 2: denominator sidecars
            "head_span_kept_token_count_prefill": prefill_kept_count,
            "head_span_kept_token_count_decode": decode_kept_count,
        }

    def _write_attention_npz(self, path: Path, segments: list[dict[str, Any]]) -> None:
        n_segments = len(segments)
        records = [*self._prefill_records, *self._decode_records.to_list()]
        _synchronize_pending_arrays(records)
        offsets = [0]
        for record in records:
            offsets.append(offsets[-1] + int(record["segment_mass"].shape[0]))
        n_rows = offsets[-1]
        k = self.config.attention_top_k
        query_row_offsets = np.asarray(offsets, dtype=np.int64)
        query_heads = (
            np.concatenate([r["query_heads"] for r in records])
            if records
            else np.zeros((0,), dtype=np.int32)
        )
        query_positions = (
            np.concatenate([r["query_positions"] for r in records])
            if records
            else np.zeros((0,), dtype=np.int32)
        )
        segment_mass = (
            np.concatenate(
                [_materialize_array(r["segment_mass"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, n_segments), dtype=np.float32)
        )
        topk_indices = (
            np.concatenate(
                [_materialize_array(r["topk_indices"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, k), dtype=np.int32)
        )
        topk_weights = (
            np.concatenate(
                [_materialize_array(r["topk_weights"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, k), dtype=np.float32)
        )
        topk_csr_offsets, topk_csr_indices, topk_csr_weights = _encode_topk_csr(
            topk_indices,
            topk_weights,
        )
        (
            span_span,
            span_span_counts,
            span_span_raw,
            span_span_row_sums,
        ) = _span_span_matrix(
            segment_mass=segment_mass,
            query_positions=query_positions.astype(np.int64, copy=False),
            query_row_offsets=query_row_offsets,
            segments=segments,
            n_segments=n_segments,
            generated_segment_id=int(self._session["generated_segment_id"]),
        )
        head_span_arrays = self._build_head_span_arrays(n_segments)
        np.savez_compressed(
            path,
            call_idx=np.asarray(self._session["call_idx"], dtype=np.int32),
            n_segments=np.asarray(n_segments, dtype=np.int32),
            top_k=np.asarray(k, dtype=np.int32),
            record_layer=np.asarray([r["layer"] for r in records], dtype=np.int32),
            record_phase=np.asarray([r["phase"] for r in records]),
            record_decode_step=np.asarray(
                [r["decode_step"] for r in records], dtype=np.int32
            ),
            query_row_offsets=query_row_offsets,
            query_heads=query_heads,
            query_positions=query_positions,
            segment_mass=segment_mass.astype(np.float16, copy=False),
            span_span_matrix=span_span,
            span_span_matrix_raw=span_span_raw,
            span_span_row_sums=span_span_row_sums,
            span_span_query_counts=span_span_counts,
            topk_storage=np.asarray("csr"),
            topk_csr_offsets=topk_csr_offsets,
            topk_csr_indices=topk_csr_indices,
            topk_csr_weights=topk_csr_weights,
            topk_csr_width=np.asarray(k, dtype=np.int32),
            heavy_indices=np.stack(
                [_materialize_array(r["heavy_indices"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, k), dtype=np.int32),
            heavy_weights=np.stack(
                [_materialize_array(r["heavy_weights"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, k), dtype=np.float32),
            n_query_rows=np.asarray(n_rows, dtype=np.int64),
            **head_span_arrays,
        )

    def _write_routing_npz(self, path: Path, n_segments: int) -> None:
        records = self._routing_records
        # Bulk-flush pending GPU stages once per device; subsequent
        # `_materialize_array` calls below become sync-free numpy reads.
        _synchronize_pending_arrays(records)
        token_offsets = [0]
        n_experts = 0
        top_k = 0
        for record in records:
            choices = record["expert_choice"]
            token_offsets.append(token_offsets[-1] + int(choices.shape[0]))
            n_experts = int(record["expert_load"].shape[1])
            top_k = int(choices.shape[1])
        np.savez_compressed(
            path,
            call_idx=np.asarray(self._session["call_idx"], dtype=np.int32),
            n_segments=np.asarray(n_segments, dtype=np.int32),
            n_experts=np.asarray(n_experts, dtype=np.int32),
            top_k_experts=np.asarray(top_k, dtype=np.int32),
            record_layer=np.asarray([r["layer"] for r in records], dtype=np.int32),
            record_phase=np.asarray([r["phase"] for r in records], dtype="U7"),
            record_decode_step=np.asarray(
                [r["decode_step"] for r in records],
                dtype=np.int32,
            ),
            record_path=np.asarray([r["path"] for r in records]),
            token_row_offsets=np.asarray(token_offsets, dtype=np.int64),
            expert_choice=np.concatenate(
                [_materialize_array(r["expert_choice"]) for r in records], axis=0
            )
            if records
            else np.zeros((0, 0), dtype=np.int32),
            expert_weight=(
                np.concatenate(
                    [_materialize_array(r["expert_weight"]) for r in records], axis=0
                ).astype(np.float16, copy=False)
                if records
                else np.zeros((0, 0), dtype=np.float16)
            ),
            expert_load=np.stack(
                [_materialize_array(r["expert_load"]) for r in records], axis=0
            )
            if records
            else np.zeros((0, n_segments, 0), dtype=np.float32),
            expert_token_count=np.stack(
                [r["expert_token_count"] for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, n_segments, 0), dtype=np.int32),
            expert_token_count_unit=np.asarray(
                [r["expert_token_count_unit"] for r in records],
                dtype="<U32",
            ),
            expert_capacity=np.asarray(
                [int(r["expert_capacity"]) for r in records],
                dtype=np.int32,
            ),
            expert_expected_overflow_count=np.stack(
                [r["expert_expected_overflow_count"] for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, 0), dtype=np.int32),
            expected_dropped_token_count=np.asarray(
                [int(r["expected_dropped_token_count"]) for r in records],
                dtype=np.int32,
            ),
            drop_signal_mode=np.asarray(
                [r["drop_signal_mode"] for r in records],
                dtype="<U32",
            ),
        )
