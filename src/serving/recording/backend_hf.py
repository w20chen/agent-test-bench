"""HuggingFace backend for optional internal recording."""

from __future__ import annotations

import asyncio
import gc
import importlib.metadata
import json
import logging
import os
import secrets
import string
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import json_repair
from transformers import DynamicCache

from agents.openclaw.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from serving.kv_policies import build_eviction_cache
from serving.kv_policies.base import BaseEvictionCache, EvictionPolicyConfig
from serving.kv_policies.recorder import KVEvictionRecorder
from serving.recording.attention_bus import AttentionBus
from serving.recording.hooks import LayerCapturer
from serving.recording.recording import RecordingConfig, segment_role
from serving.sparse_attention import (
    BaseSparseAttention,
    SparseAttentionConfig,
    SparseAttentionRecorder,
    build_sparse_attention,
)
from serving.sparse_attention.config import validate_attention_method_exclusivity


_LOG = logging.getLogger(__name__)


_ALLOWED_MSG_KEYS = frozenset(
    {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}
)
_OPENCLAW_MESSAGE_ID_KEY = "_openclaw_message_id"
_ALNUM = string.ascii_letters + string.digits
_MAX_TORCH_SEED = (2**63) - 1
# Debug escape hatch for byte-equality validation: when set to a truthy value
# the provider falls back to the pre-default-on path (fresh prompt every call,
# no session-shared KV cache, no LCP delta). Use only to A/B verify that
# enabling the session cache produces identical greedy decodes — default is
# enabled. Sparse/eviction policies that *require* a session cache will still
# build one when their config is provided.
_SESSION_CACHE_DISABLED_ENV = "OMC_DISABLE_SESSION_CACHE"


def _session_cache_disabled() -> bool:
    raw = os.environ.get(_SESSION_CACHE_DISABLED_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _short_tool_id() -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _message_has_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content)
    if isinstance(content, list):
        return bool(content)
    return True


def _message_segment_metadata(message: dict[str, Any]) -> dict[str, Any]:
    """Metadata retained on token segments for downstream provenance joins."""
    payload: dict[str, Any] = {
        "has_content": _message_has_content(message),
        "has_tool_calls": bool(message.get("tool_calls")),
    }
    for key in ("tool_call_id", "name"):
        value = message.get(key)
        if value is not None:
            payload[key] = str(value)
    return payload


def _token_count(encoded: Any) -> int:
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(input_ids, "ndim"):
        return (
            int(input_ids.numel()) if input_ids.ndim == 1 else int(input_ids.shape[-1])
        )
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _synchronize_cuda_devices(torch_module: Any) -> None:
    """Synchronize each visible CUDA device before reading cross-device metrics."""
    for device_idx in range(int(torch_module.cuda.device_count())):
        torch_module.cuda.synchronize(device_idx)


def _token_boundary_for_char(
    tokenizer: Any, text: str, offsets: Any, char_pos: int
) -> int:
    if char_pos <= 0:
        return 0
    if offsets is None:
        return _token_count(tokenizer(text[:char_pos], add_special_tokens=False))
    for idx, pair in enumerate(offsets):
        start = int(pair[0])
        if start >= char_pos:
            return idx
    return len(offsets)


def _tokenize_with_offsets(
    tokenizer: Any, text: str
) -> tuple[Any, list[list[int]] | None]:
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded, None
    offsets = encoded.pop("offset_mapping")[0].tolist()
    return encoded, offsets


def _normalize_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json_repair.loads(value)
    if not isinstance(value, dict):
        return {}
    return value


def _normalize_tool_name(value: Any) -> str:
    name = str(value or "").strip()
    while name.startswith("<function="):
        name = name[len("<function=") :].strip()
    while name.endswith(">"):
        name = name[:-1].strip()
    return name


def _parse_tool_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] not in "{[\"'-0123456789" and value not in {"true", "false", "null"}:
        return value
    try:
        return json_repair.loads(value)
    except Exception:
        return value


def _normalize_messages(
    messages: list[dict[str, Any]],
    *,
    preserve_openclaw_message_id: bool = False,
) -> list[dict[str, Any]]:
    allowed_keys = (
        _ALLOWED_MSG_KEYS | {_OPENCLAW_MESSAGE_ID_KEY}
        if preserve_openclaw_message_id
        else _ALLOWED_MSG_KEYS
    )
    sanitized = LLMProvider._sanitize_request_messages(
        LLMProvider._sanitize_empty_content(messages),
        allowed_keys,
    )
    normalized: list[dict[str, Any]] = []
    for message in sanitized:
        copied = dict(message)
        tool_calls = []
        for tool_call in copied.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            tc = dict(tool_call)
            function = dict(tc.get("function") or {})
            function["arguments"] = _normalize_tool_arguments(
                function.get("arguments", tc.get("arguments", {}))
            )
            tc["function"] = function
            tool_calls.append(tc)
        if tool_calls:
            copied["tool_calls"] = tool_calls
        normalized.append(copied)
    return normalized


def _message_signature(message: dict[str, Any]) -> str:
    """Stable signature for detecting when a message index was reused."""
    payload = {
        key: value
        for key, value in message.items()
        if key != _OPENCLAW_MESSAGE_ID_KEY
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _message_provenance_key(message: dict[str, Any]) -> str:
    message_id = message.get(_OPENCLAW_MESSAGE_ID_KEY)
    if message_id is not None:
        return f"openclaw:{message_id}"
    return f"signature:{_message_signature(message)}"


def _strip_openclaw_message_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in message.items() if key != _OPENCLAW_MESSAGE_ID_KEY}
        for message in messages
    ]


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_chat_with_segments(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    first_seen_call_by_message_index: dict[int, int] | None = None,
    default_first_seen_call: int | None = None,
) -> tuple[Any, list[dict[str, Any]], str]:
    messages = _normalize_messages(messages)
    full_text = _apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
    )

    char_segments: list[dict[str, Any]] = []
    previous_end = 0
    for idx, message in enumerate(messages):
        prefix = _apply_chat_template(
            tokenizer,
            messages[: idx + 1],
            tools=tools,
            add_generation_prompt=False,
        )
        if not full_text.startswith(prefix):
            # Some chat templates (e.g. Qwen3) render assistant turns
            # differently when add_generation_prompt=False vs. True (adding
            # <think> markers absent from the full-conversation rendering).
            # We cannot determine the exact char boundary for this turn, so
            # emit an "unmatched" sentinel that covers from the last known
            # boundary to the start of the NEXT turn's prefix (determined on
            # the following iteration). This preserves full token coverage and
            # lets downstream consumers filter "unmatched" out of per-role
            # analyses rather than silently mislabeling those tokens as
            # belonging to the surrounding turn.
            _LOG.warning(
                "chat template prefix misalignment at message_index=%d role=%s; "
                "emitting 'unmatched' segment — downstream per-role stats will "
                "exclude this region",
                idx,
                message.get("role"),
            )
            char_segments.append(
                {
                    "role": "unmatched",
                    "message_index": idx,
                    "char_start": previous_end,
                    "char_end": previous_end,  # end filled by next aligned prefix
                    **_message_segment_metadata(message),
                    "_pending": True,
                }
            )
            continue
        end = len(prefix)
        # Close any pending "unmatched" segments opened on prior misaligned
        # turns. Their true end is the start of THIS aligned turn (i.e.
        # `previous_end`), not `end` — closing at `end` would swallow the
        # current aligned turn's content and make the aligned segment vanish
        # (its `previous_end < end` guard would fail after we advance
        # `previous_end`). Two back-to-back misaligned turns each get their
        # own sentinel with non-overlapping ranges.
        for seg in char_segments:
            if seg.get("_pending"):
                seg["char_end"] = previous_end
                del seg["_pending"]
        if previous_end < end:
            char_segments.append(
                {
                    "role": segment_role(message),
                    "message_index": idx,
                    "char_start": previous_end,
                    "char_end": end,
                    **_message_segment_metadata(message),
                }
            )
        previous_end = end
    # Close any "unmatched" segment that was still pending at loop end
    # (last message misaligned, no subsequent aligned prefix to close it).
    for seg in char_segments:
        if seg.get("_pending"):
            seg["char_end"] = len(full_text)
            del seg["_pending"]
    if previous_end < len(full_text):
        char_segments.append(
            {
                "role": "gen_prompt",
                "message_index": None,
                "char_start": previous_end,
                "char_end": len(full_text),
                "has_content": False,
                "has_tool_calls": False,
            }
        )

    encoded, offsets = _tokenize_with_offsets(tokenizer, full_text)
    segments: list[dict[str, Any]] = []
    for segment in char_segments:
        start = _token_boundary_for_char(
            tokenizer, full_text, offsets, int(segment["char_start"])
        )
        end = _token_boundary_for_char(
            tokenizer, full_text, offsets, int(segment["char_end"])
        )
        if start >= end:
            continue
        first_seen_call = default_first_seen_call
        message_index = segment.get("message_index")
        if (
            first_seen_call_by_message_index is not None
            and message_index is not None
        ):
            first_seen_call = first_seen_call_by_message_index.get(
                int(message_index),
                default_first_seen_call,
            )
        payload = {
            **segment,
            "token_start": start,
            "token_end": end,
        }
        if first_seen_call is not None:
            payload["first_seen_call"] = int(first_seen_call)
            payload["first_seen_call_inferred"] = False
        segments.append(payload)
    return encoded, segments, full_text


def _strip_tool_blocks(text: str, blocks: list[tuple[int, int]]) -> str:
    if not blocks:
        return text
    parts: list[str] = []
    last = 0
    for start, end in blocks:
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    return "".join(parts).strip() or None


def _parse_qwen_xml_tool_calls(text: str) -> tuple[str | None, list[ToolCallRequest]]:
    import re

    calls: list[ToolCallRequest] = []
    blocks: list[tuple[int, int]] = []
    outer_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    function_pattern = re.compile(
        r"<function=([^>\n]+)>\s*(.*?)\s*</function>",
        re.DOTALL,
    )
    parameter_pattern = re.compile(
        r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )
    outer_matches = list(outer_pattern.finditer(text))
    if outer_matches:
        for match in outer_matches:
            function_match = function_pattern.search(match.group(1))
            if function_match is None:
                continue
            args = {
                param_match.group(1).strip(): _parse_tool_value(param_match.group(2))
                for param_match in parameter_pattern.finditer(function_match.group(2))
            }
            calls.append(
                ToolCallRequest(
                    id=_short_tool_id(),
                    name=_normalize_tool_name(function_match.group(1)),
                    arguments=args,
                )
            )
            blocks.append((match.start(), match.end()))
    else:
        # Lenient fallback: model may have dropped the outer <tool_call> wrap
        # under KV-eviction context loss. Empirically observed on
        # Qwen3-Coder-30B (h2o b4096 capstone, iter ≥13) where eviction
        # trimmed the chat template's example. Accept bare
        # <function=...></function> blocks.
        for fm in function_pattern.finditer(text):
            args = {
                pm.group(1).strip(): _parse_tool_value(pm.group(2))
                for pm in parameter_pattern.finditer(fm.group(2))
            }
            calls.append(
                ToolCallRequest(
                    id=_short_tool_id(),
                    name=_normalize_tool_name(fm.group(1)),
                    arguments=args,
                )
            )
            blocks.append((fm.start(), fm.end()))
    return _strip_tool_blocks(text, blocks), calls


def _parse_qwen_json_tool_calls(text: str) -> tuple[str | None, list[ToolCallRequest]]:
    import re

    calls: list[ToolCallRequest] = []
    blocks: list[tuple[int, int]] = []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        try:
            payload = json_repair.loads(match.group(1))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        calls.append(
            ToolCallRequest(
                id=_short_tool_id(),
                name=_normalize_tool_name(payload.get("name")),
                arguments=_normalize_tool_arguments(payload.get("arguments", {})),
            )
        )
        blocks.append((match.start(), match.end()))
    return _strip_tool_blocks(text, blocks), calls


def _parse_glm_tool_calls(text: str) -> tuple[str | None, list[ToolCallRequest]]:
    import re

    calls: list[ToolCallRequest] = []
    blocks: list[tuple[int, int]] = []
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        body = match.group(1).strip()
        if not body:
            continue
        name, _, rest = body.partition("\n")
        args: dict[str, Any] = {}
        for arg_match in re.finditer(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            rest,
            re.DOTALL,
        ):
            key = arg_match.group(1).strip()
            raw_value = arg_match.group(2).strip()
            try:
                args[key] = _parse_tool_value(raw_value)
            except Exception:
                args[key] = raw_value
        calls.append(
            ToolCallRequest(
                id=_short_tool_id(),
                name=_normalize_tool_name(name),
                arguments=args,
            )
        )
        blocks.append((match.start(), match.end()))
    return _strip_tool_blocks(text, blocks), calls


def parse_text_tool_calls(text: str) -> tuple[str | None, list[ToolCallRequest]]:
    content, calls = _parse_qwen_xml_tool_calls(text)
    if calls:
        return content, calls
    content, calls = _parse_qwen_json_tool_calls(text)
    if calls:
        return content, calls
    return _parse_glm_tool_calls(text)


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        value = line.strip()
        if value:
            return value
    return None


def _generation_seed(base_seed: int, call_idx: int) -> int:
    seed = int(base_seed) + int(call_idx)
    return seed % _MAX_TORCH_SEED


def _generation_metadata(
    *,
    seed: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "do_sample": bool(float(temperature) > 0),
        "temperature": float(temperature),
        "top_p": None if top_p is None else float(top_p),
        "top_k": None if top_k is None else int(top_k),
        "repetition_penalty": (
            None if repetition_penalty is None else float(repetition_penalty)
        ),
    }


def _longest_common_prefix(a: Any, b: Any) -> int:
    """Length of the longest matching prefix between two 1-D token id tensors."""
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return 0
    eq = (a[:n] == b[:n]).to(dtype=bool)
    # `argmax` on a bool tensor returns the index of the first True; we want
    # the first False. Flip and use cumprod so the prefix of all-True stays 1
    # and the first mismatch zeros everything after — sum gives the LCP.
    return int(eq.to(dtype=a.dtype).cumprod(dim=0).sum().item())


def _hf_max_memory(torch_module: Any) -> dict[int | str, str] | None:
    gpu_gib = _positive_int_env("HF_RECORDING_MAX_GPU_MEMORY_GIB")
    if gpu_gib is None:
        return None
    device_count = int(torch_module.cuda.device_count())
    if device_count <= 0:
        raise ValueError("HF_RECORDING_MAX_GPU_MEMORY_GIB requires CUDA devices")
    cpu_gib = _positive_int_env("HF_RECORDING_MAX_CPU_MEMORY_GIB") or 128
    max_memory: dict[int | str, str] = {
        device_idx: f"{gpu_gib}GiB" for device_idx in range(device_count)
    }
    max_memory["cpu"] = f"{cpu_gib}GiB"
    return max_memory


class HFRecordingProvider(LLMProvider):
    """OpenClaw provider that records HF model internals while generating."""

    preserve_openclaw_message_ids = True

    def __init__(
        self,
        *,
        default_model: str,
        config: RecordingConfig | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        eviction_config: EvictionPolicyConfig | None = None,
        sparse_attention_config: SparseAttentionConfig | None = None,
        temperature: float = 0.1,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        # Belt-and-suspenders: the CLI layer also validates exclusivity, but
        # any caller bypassing CLI (notebook, test, direct construction)
        # must hit the same gate. Cheap to evaluate before any model load.
        validate_attention_method_exclusivity(eviction_config, sparse_attention_config)
        super().__init__(api_key=api_key, api_base=api_base)
        self.default_model = default_model
        self.config = config or RecordingConfig()
        self.generation = GenerationSettings(
            temperature=temperature,
            max_tokens=4096,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        self._call_idx = 0
        self._chat_lock = threading.Lock()
        # Session-shared cache lives across chat() calls so H2O score buffers,
        # streaming-LLM windows, and eviction state accumulate over the
        # provider's lifetime. Built lazily on first call so the constructor
        # stays free of transformers `Cache` instantiation. The cache type is
        # `BaseEvictionCache` when an eviction policy is configured and a plain
        # `DynamicCache` otherwise — the latter still benefits any multi-call
        # workflow with shared prefix (sparse_attention runs, bare baseline)
        # because HF can resume `past_key_values` from the previous call.
        self._eviction_config = eviction_config
        self._sparse_attention_config = sparse_attention_config
        # Method instance is built once per provider (it carries no per-call
        # state for sliding). The recorder is swapped per call via
        # LayerCapturer.set_sparse_recorder().
        self._sparse_attention: BaseSparseAttention | None = None
        self._session_cache: DynamicCache | None = None
        # Token IDs currently materialised in the session cache, including any
        # decoded tokens from prior calls. (1, T) growing tensor; LCP is
        # computed against this to derive the delta passed to generate().
        self._session_token_ids: Any | None = None
        # Attempt-level audit log persisted to meta.json. It distinguishes full
        # cold prompts, strict-prefix resume prompts, and divergence rebuilds.
        self._session_history: list[dict[str, Any]] = []
        self._message_first_seen: list[tuple[str, int]] = []

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.model_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            default_model,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            default_model,
            torch_dtype=dtype,
            device_map=self.config.device_map,
            max_memory=_hf_max_memory(torch),
            attn_implementation="sdpa",
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()
        self._torch = torch
        self._captures_router_logits = self._model_has_router_logits()
        # Per-provider AttentionBus: lives across calls so future strategy
        # code (H2O in step 6) can subscribe at construction time. Step 5
        # leaves it with zero subscribers, which makes it a no-op dispatch
        # and preserves attention.npz byte-equality vs the pre-step-5 path.
        self._attention_bus = AttentionBus()
        if self._sparse_attention_config is not None:
            self._sparse_attention = build_sparse_attention(
                self._sparse_attention_config,
                num_layers=int(self.model.config.num_hidden_layers),
                recorder=None,
                attention_bus=self._attention_bus,
            )
        self.capturer = LayerCapturer(
            self.model,
            config=self.config,
            model_summary=self._model_summary(),
            attention_bus=self._attention_bus,
            sparse_attention=self._sparse_attention,
        )
        # Attempt-level KV policy summary lands in meta.json. The
        # `prefill_score_bias` flag is the explicit warning that H2O's score
        # accumulator only saw the LayerCapturer-sampled prefill rows
        # (plan E12 — recording bias). False for non-h2o policies; analysis
        # code can branch on it.
        self.capturer.set_kv_policy_meta(self._kv_policy_meta_payload())
        self.capturer.set_sparse_attention_meta(
            self._sparse_attention_meta_payload()
        )

    def _sparse_attention_meta_payload(self) -> dict[str, Any] | None:
        """Build the attempt-level sparse_attention block for meta.json.

        Returns None when no sparse method is configured. Otherwise mirrors
        the serialisable subset of `SparseAttentionConfig`. Per-method
        knobs that don't apply to the active method are emitted as-is from
        the dataclass (a `random_evict`-style approach) so the meta stays
        debuggable without growing a per-method projection.
        """
        cfg = self._sparse_attention_config
        if cfg is None:
            return None
        return {
            "method": cfg.name,
            "sink_size": int(cfg.sink_size),
            "recent_window": int(cfg.recent_window),
            "record": bool(cfg.record),
            "observe_only": bool(cfg.observe_only),
            "budget": int(cfg.budget) if cfg.budget is not None else None,
            "block_size": int(cfg.block_size),
            "score_reduction": str(cfg.score_reduction),
            "phase_scope": str(cfg.phase_scope),
        }

    def _kv_policy_meta_payload(self) -> dict[str, Any] | None:
        """Build the attempt-level kv_policy block for meta.json.

        Returns None when no eviction policy is configured. Otherwise mirrors
        the serialisable subset of EvictionPolicyConfig plus the
        `prefill_score_bias` flag — True when H2O is paired with the sampled
        prefill mode (the bus only sees the LayerCapturer-sampled query rows
        during prefill), False otherwise.
        """
        cfg = self._eviction_config
        if cfg is None:
            return None
        prefill_score_bias = (
            cfg.name == "h2o" and getattr(cfg, "prefill_mode", "full") == "sampled"
        )
        return {
            "name": cfg.name,
            "budget": int(cfg.budget) if cfg.budget is not None else None,
            "sink_size": int(cfg.sink_size),
            "recent_window": int(cfg.recent_window),
            "heavy_ratio": float(cfg.heavy_ratio),
            "aggregate": str(cfg.aggregate),
            "ema_decay": float(cfg.ema_decay),
            "seed": int(cfg.seed),
            "record": bool(cfg.record),
            "prefill_mode": str(cfg.prefill_mode),
            "prefill_score_bias": bool(prefill_score_bias),
        }

    def get_default_model(self) -> str:
        return self.default_model

    def start_attempt(self, recordings_dir: Path) -> None:
        self._drop_session_cache()
        if self._sparse_attention is not None and hasattr(
            self._sparse_attention, "reset_state"
        ):
            self._sparse_attention.reset_state()
        self._call_idx = 0
        self._session_history = []
        self._message_first_seen = []
        self.capturer.start_attempt(recordings_dir)

    def wait_until_idle(self, timeout_s: float | None = None) -> None:
        """Wait until no chat generation is holding the provider lock."""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError(f"timeout_s must be non-negative, got {timeout_s}")
        if timeout_s is None:
            acquired = self._chat_lock.acquire()
        else:
            acquired = self._chat_lock.acquire(timeout=timeout_s)
        if not acquired:
            raise TimeoutError(
                "timed out waiting for HF recording provider to become idle"
            )
        self._chat_lock.release()

    def finish_attempt(self, trace_path: Path | None = None) -> None:
        self.capturer.set_attempt_extra_meta(
            {"session_history": [dict(item) for item in self._session_history]}
        )
        self.capturer.finish_attempt(trace_path=trace_path)

    def _model_summary(self) -> dict[str, Any]:
        cfg = self.model.config
        torch = self._torch
        return {
            "name": self.default_model,
            "architectures": list(getattr(cfg, "architectures", []) or []),
            "model_type": getattr(cfg, "model_type", None),
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "num_experts": getattr(cfg, "num_experts", None)
            or getattr(cfg, "num_local_experts", None),
            "num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None)
            or getattr(cfg, "num_experts_per_token", None),
            "record_router_logits": self._captures_router_logits,
            "recording_config": asdict(self.config),
            "hf_model_commit_hash": getattr(cfg, "_commit_hash", None),
            "torch_version": getattr(torch, "__version__", None),
            "transformers_version": _package_version("transformers"),
            "accelerate_version": _package_version("accelerate"),
            "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "nvidia_driver_version": _nvidia_driver_version()
            if torch.cuda.is_available()
            else None,
        }

    def _model_has_router_logits(self) -> bool:
        cfg = self.model.config
        return any(
            getattr(cfg, name, None) is not None
            for name in (
                "num_experts",
                "num_local_experts",
                "num_experts_per_tok",
                "num_experts_per_token",
            )
        )

    def _input_device(self) -> Any:
        for parameter in self.model.parameters():
            if str(parameter.device) != "meta":
                return parameter.device
        return self._torch.device("cpu")

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _first_seen_calls_for_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        call_idx: int,
    ) -> dict[int, int]:
        signatures = [_message_provenance_key(message) for message in messages]
        previous = list(self._message_first_seen)
        previous_signatures = [signature for signature, _first_seen in previous]

        if previous_signatures == signatures[: len(previous_signatures)]:
            first_seen_values = [
                *[first_seen for _signature, first_seen in previous],
                *[call_idx] * (len(signatures) - len(previous)),
            ]
        else:
            first_seen_values = [call_idx] * len(signatures)
            previous_idx = len(previous) - 1
            for current_idx in range(len(signatures) - 1, -1, -1):
                signature = signatures[current_idx]
                while previous_idx >= 0 and previous[previous_idx][0] != signature:
                    previous_idx -= 1
                if previous_idx < 0:
                    continue
                first_seen_values[current_idx] = int(previous[previous_idx][1])
                previous_idx -= 1

        self._message_first_seen = list(zip(signatures, first_seen_values, strict=True))
        return dict(enumerate(first_seen_values))

    def _build_session_cache(self) -> DynamicCache | None:
        """Build a fresh KV cache for the current session, or None to skip it.

        Returns None when the active sparse method declares
        `requires_full_prefill=True` (heavy_hitter needs every prefill token's
        attention to land in the AttentionBus — session cache delta-prefill
        would skip the cached prefix and silently degrade selection to
        streaming-LLM). Callers must treat None like the env-var-disabled
        path: pass the full prompt each call with no past_key_values.

        Eviction config → policy-specific `BaseEvictionCache` subclass
        (h2o / streaming / random). No eviction → plain `DynamicCache` for
        sparse-only or bare baseline runs, which still benefit from LCP-based
        delta prefill across consecutive chat() calls.
        """
        if (
            self._sparse_attention is not None
            and getattr(self._sparse_attention, "requires_full_prefill", False)
        ):
            return None
        if self._eviction_config is None:
            return DynamicCache()
        return build_eviction_cache(
            self._eviction_config,
            num_layers=int(self.model.config.num_hidden_layers),
            recorder=None,
            attention_bus=self._attention_bus,
            max_position_embeddings=int(self.model.config.max_position_embeddings),
        )

    def _drop_session_cache(self) -> None:
        cache = self._session_cache
        if cache is None:
            return
        # Plain DynamicCache (sparse / baseline runs) has no bus subscription;
        # only BaseEvictionCache subclasses need an unsubscribe attempt.
        if isinstance(cache, BaseEvictionCache) and cache.requires_attention():
            try:
                self._attention_bus.unsubscribe(cache)
            except ValueError:
                pass
        self._session_cache = None
        self._session_token_ids = None

    def _record_disabled_session_history(
        self, *, call_idx: int, new_len: int
    ) -> None:
        """Append the audit-log entry for a 'no session cache this call' path.

        Used by both the env-var escape hatch and the per-method opt-out
        (`requires_full_prefill=True`). Mirrors the legacy pre-default-on
        accounting: `used_session_cache=False`, `lcp=0`, `delta_len=new_len`.
        """
        self._session_history.append(
            {
                "call_idx": call_idx,
                "used_session_cache": False,
                "lcp": 0,
                "cached_len_before": 0,
                "new_len": new_len,
                "delta_len": new_len,
                "diverged": False,
            }
        )

    def _prepare_session_cache(
        self, *, prompt_ids: Any, call_idx: int
    ) -> tuple[Any, bool]:
        """Resolve the cache state for one chat() call.

        Returns `(input_ids, used_session_cache)` — `input_ids` is the tensor
        to pass to `generate()` (delta when the cache is a strict prefix of
        the prompt, full prompt otherwise), and `used_session_cache` records
        whether `past_key_values` will be supplied.

        Thread-safety: must only be called while `_chat_lock` is held.
        `_session_cache.recorder` and `_session_token_ids` are mutated here and
        in `run_generate`; the lock serialises concurrent `chat()` callers.
        """
        new_len = int(prompt_ids.shape[-1])
        # Debug escape hatch: env-var bypass for byte-equality A/B validation.
        # When set, behaves identically to the pre-default-on code path (full
        # prompt every call, no past_key_values supplied). Document usage with
        # OMC_DISABLE_SESSION_CACHE=1; default behavior is enabled.
        if _session_cache_disabled():
            self._record_disabled_session_history(
                call_idx=call_idx, new_len=new_len
            )
            return prompt_ids, False
        # Session cache benefits any multi-call workflow with shared prefix
        # (sparse_attention runs, bare baseline, eviction policies). Not gated
        # on eviction policy any more — see commit message for the per-turn
        # latency motivation. Methods that declare requires_full_prefill=True
        # (heavy_hitter) opt out via `_build_session_cache()` returning None.
        if self._session_cache is None:
            candidate = self._build_session_cache()
            if candidate is None:
                # Per-method opt-out: behaves like the env-var-disabled path.
                # `_session_cache` stays None so `_extend_session_tokens` and
                # the eviction-only hooks in `run_generate` all short-circuit.
                self._record_disabled_session_history(
                    call_idx=call_idx, new_len=new_len
                )
                return prompt_ids, False
            self._session_cache = candidate
            self._session_token_ids = prompt_ids.clone()
            _LOG.debug(
                "session cache built (call_idx=%d, prompt_len=%d)",
                call_idx,
                new_len,
            )
            self._session_history.append(
                {
                    "call_idx": call_idx,
                    "used_session_cache": True,
                    "lcp": 0,
                    "cached_len_before": 0,
                    "new_len": new_len,
                    "delta_len": new_len,
                    "diverged": False,
                }
            )
            return prompt_ids, True
        assert self._session_token_ids is not None
        cached_ids = self._session_token_ids[0]
        # Desync gate for plain DynamicCache: logical token IDs we track must
        # equal the cache's physical seq_len. Eviction caches deliberately
        # diverge (post-evict phys_kv_len < logical len), so we skip the assert
        # for that branch — `_chat_locked` already plumbs phys vs logical
        # lengths explicitly for mask / cache_position.
        if not isinstance(self._session_cache, BaseEvictionCache):
            expected = int(self._session_token_ids.shape[-1])
            actual = int(self._session_cache.get_seq_length(0))
            assert expected == actual, (
                f"session_token_ids ({expected}) != DynamicCache seq_len "
                f"({actual}); _extend_session_tokens / generate desync -- "
                "refusing to proceed with potentially mis-positioned KV"
            )
        new_ids = prompt_ids[0]
        lcp = _longest_common_prefix(cached_ids, new_ids)
        cached_len = int(cached_ids.shape[0])
        _LOG.debug(
            "session cache lcp check (call_idx=%d, lcp=%d, cached_len=%d, "
            "new_len=%d, is_prefix=%s)",
            call_idx,
            lcp,
            cached_len,
            new_len,
            lcp == cached_len,
        )
        if lcp == cached_len and new_len > cached_len:
            # Strict prefix: pass only the delta.
            delta_len = new_len - lcp
            self._session_history.append(
                {
                    "call_idx": call_idx,
                    "used_session_cache": True,
                    "lcp": lcp,
                    "cached_len_before": cached_len,
                    "new_len": new_len,
                    "delta_len": delta_len,
                    "diverged": False,
                }
            )
            return prompt_ids[:, lcp:], True
        # Divergence: monotonic-reprompt assumption violated. Drop and rebuild.
        _LOG.warning(
            "session KV cache diverged from new prompt (lcp=%d, cached_len=%d, "
            "new_len=%d, call_idx=%d); rebuilding fresh cache",
            lcp,
            cached_len,
            new_len,
            call_idx,
        )
        self._drop_session_cache()
        self._session_cache = self._build_session_cache()
        self._session_token_ids = prompt_ids.clone()
        self._session_history.append(
            {
                "call_idx": call_idx,
                "used_session_cache": True,
                "lcp": lcp,
                "cached_len_before": cached_len,
                "new_len": new_len,
                "delta_len": new_len,
                "diverged": True,
            }
        )
        return prompt_ids, True

    def _extend_session_tokens(self, *, prompt_ids: Any, output_ids: list[int]) -> None:
        # No session cache means nothing to extend (escape hatch or already
        # dropped). The cache is the source of truth for whether we should
        # bother tracking token state.
        if self._session_cache is None:
            return
        # Append raw generated token ids to prompt_ids. Note: for models with
        # chain-of-thought generation (e.g. Qwen3 <think>…</think>), the raw
        # output_ids include thinking tokens that differ from how the next
        # call's apply_chat_template re-renders the completed turn. When that
        # happens, _prepare_session_cache will detect a divergence (LCP <
        # cached_len) and rebuild the cache from scratch — a correct fallback
        # at the cost of one extra full-prefill. The divergence path is safe;
        # this simpler storage avoids a re-render whose template behavior is
        # model-specific and hard to predict in the general case.
        generated = self._torch.as_tensor(
            output_ids, dtype=prompt_ids.dtype, device=prompt_ids.device
        ).unsqueeze(0)
        # O(N²) cat per call; acceptable for agent-loop scales (≤100 turns ×
        # ≤32K tokens). Future: track only the suffix delta.
        combined = self._torch.cat([prompt_ids, generated], dim=-1)
        if isinstance(self._session_cache, BaseEvictionCache):
            # Eviction policies intentionally let physical KV length lag
            # behind logical token history (evicted slots reduce physical
            # without erasing token ids). Keep the full logical sequence.
            self._session_token_ids = combined
        else:
            # Plain DynamicCache: logical == physical except for the off-by-
            # one inherent to HF generate(). The LAST generated token
            # (typically EOS, or the max_new_tokens cutoff token) is the
            # OUTPUT of the last forward, not an INPUT, so its KV is not in
            # the cache. Truncate session_token_ids to the cache's actual
            # physical length so the desync invariant holds on the next call.
            cache_len = int(self._session_cache.get_seq_length(0))
            self._session_token_ids = combined[..., :cache_len]

    def close(self) -> None:
        """Release hooks, session cache, and bus subscriptions."""
        self._drop_session_cache()
        capturer = getattr(self, "capturer", None)
        if capturer is not None:
            capturer.close()

    def __enter__(self) -> "HFRecordingProvider":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        try:
            self.finish_attempt()
        except Exception:
            _LOG.exception("defensive finish_attempt() failed during provider exit")
        self.close()
        self._session_token_ids = None
        self._session_cache = None
        if hasattr(self, "model"):
            self.model = None
        if hasattr(self, "tokenizer"):
            self.tokenizer = None
        if hasattr(self, "capturer"):
            self.capturer = None
        gc.collect()
        torch = getattr(self, "_torch", None)
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                _synchronize_cuda_devices(torch)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self._drop_session_cache()
        except Exception:
            pass

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model
        if reasoning_effort is not None:
            return LLMResponse(
                content=(
                    "Error: HF recording provider does not support reasoning_effort."
                ),
                finish_reason="error",
                extra={"error_type": "unsupported_reasoning_effort"},
            )
        if isinstance(tool_choice, dict) or tool_choice not in {None, "auto", "none"}:
            return LLMResponse(
                content=(
                    "Error: HF recording provider does not support forced tool_choice; "
                    'supported values are ["none", "auto"].'
                ),
                finish_reason="error",
                extra={"error_type": "unsupported_tool_choice"},
            )
        if not self._chat_lock.acquire(blocking=False):
            return LLMResponse(
                content=(
                    "Error: HF recording provider supports one in-flight chat call; "
                    "the server is already generating."
                ),
                finish_reason="error",
                extra={"error_type": "concurrent_request"},
            )
        try:
            return await self._chat_locked(
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.generation.top_p if top_p is None else top_p,
                top_k=self.generation.top_k if top_k is None else top_k,
                repetition_penalty=(
                    self.generation.repetition_penalty
                    if repetition_penalty is None
                    else repetition_penalty
                ),
                tool_choice=tool_choice,
            )
        finally:
            self._chat_lock.release()

    async def _chat_locked(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        repetition_penalty: float | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> LLMResponse:
        template_tools = None if tool_choice == "none" else tools
        call_idx = self._call_idx
        self._call_idx += 1
        started_at = time.time()

        normalized_messages_with_ids = _normalize_messages(
            messages,
            preserve_openclaw_message_id=True,
        )
        normalized_messages = _strip_openclaw_message_ids(normalized_messages_with_ids)
        first_seen_calls = self._first_seen_calls_for_messages(
            normalized_messages_with_ids,
            call_idx=call_idx,
        )
        encoded, segments, _prompt_text = tokenize_chat_with_segments(
            self.tokenizer,
            normalized_messages,
            tools=template_tools,
            first_seen_call_by_message_index=first_seen_calls,
            default_first_seen_call=call_idx,
        )
        prompt_ids = (
            encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        )
        input_token_count = int(prompt_ids.shape[-1])
        delta_ids, used_session_cache = self._prepare_session_cache(
            prompt_ids=prompt_ids, call_idx=call_idx
        )

        def run_generate() -> tuple[str, list[int]]:
            input_tensor = delta_ids.to(self._input_device())
            seed = _generation_seed(self.config.generation_seed, call_idx)
            self._torch.manual_seed(seed)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(seed)
            generation_meta = _generation_metadata(
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_tensor,
                "max_new_tokens": max(1, int(max_tokens)),
                "do_sample": bool(generation_meta["do_sample"]),
                "return_dict_in_generate": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                generation_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                generation_kwargs["top_k"] = int(top_k)
            if repetition_penalty is not None:
                generation_kwargs["repetition_penalty"] = float(repetition_penalty)
            # KV recorder is only built for the eviction-policy branch; sparse
            # and bare-baseline runs leave it None. The session cache itself is
            # shared across all branches so consecutive chat() calls can resume
            # past_key_values via LCP-based delta prefill.
            kv_recorder: KVEvictionRecorder | None = None
            if self._eviction_config is not None and self._eviction_config.record:
                # `record=False` runs the policy but skips both the
                # KVEvictionRecorder allocation AND the capturer.flush() npz
                # write; the cache mechanics still run.
                kv_recorder = KVEvictionRecorder(
                    call_idx=call_idx,
                    policy_name=self._eviction_config.name,
                )
            self.capturer.set_kv_recorder(kv_recorder)
            if used_session_cache:
                assert self._session_cache is not None
                # `recorder` / `notify_new_call` are policy-only hooks; plain
                # DynamicCache (sparse / baseline session cache) doesn't expose
                # them. Guard via isinstance so we don't AttributeError when
                # session cache runs without an eviction policy.
                if isinstance(self._session_cache, BaseEvictionCache):
                    self._session_cache.recorder = kv_recorder
                    # Step counters reset per call; physical KV slots and score
                    # buffers persist.
                    self._session_cache.notify_new_call(call_idx)
                generation_kwargs["past_key_values"] = self._session_cache
                # phys_kv_len: physical slots after eviction (attention_mask width).
                # logical_kv_len: absolute conversation position of delta[0] (RoPE).
                # These diverge after eviction; using phys for RoPE silently
                # corrupts embeddings. Explicit cache_position also sidesteps
                # HF's arange(delta_len)[past_length:] empty-tensor crash. For
                # plain DynamicCache the two lengths coincide (no eviction).
                phys_kv_len = self._session_cache.get_seq_length(0)
                delta_len = int(delta_ids.shape[-1])
                logical_kv_len = int(prompt_ids.shape[-1]) - delta_len
                mask_len = phys_kv_len + delta_len
                generation_kwargs["attention_mask"] = self._torch.ones(
                    (1, mask_len), dtype=self._torch.long
                ).to(self._input_device())
                generation_kwargs["cache_position"] = self._torch.arange(
                    logical_kv_len, logical_kv_len + delta_len, dtype=self._torch.long
                ).to(self._input_device())
            else:
                # Escape hatch (OMC_DISABLE_SESSION_CACHE=1): stock DynamicCache,
                # full prompt each call. Matches the pre-default-on path exactly.
                generation_kwargs["attention_mask"] = self._torch.ones_like(
                    input_tensor
                )
            # Per-call sparse-attention recorder swap mirrors the KV recorder
            # pattern. Both subsystems are mutually exclusive at construction
            # time, so at most one of (kv_recorder, sparse_recorder) is ever
            # non-None within a single call.
            sparse_recorder: SparseAttentionRecorder | None = None
            if (
                self._sparse_attention_config is not None
                and self._sparse_attention_config.record
            ):
                sparse_recorder = SparseAttentionRecorder(
                    call_idx=call_idx,
                    method_name=self._sparse_attention_config.name,
                )
            self.capturer.set_sparse_recorder(sparse_recorder)
            if self._sparse_attention is not None and hasattr(
                self._sparse_attention, "reset_state"
            ):
                self._sparse_attention.reset_state()
            # H2O full-prefill scoring is handled inside LayerCapturer in
            # bounded chunks and delivered only to full-prefill consumers. Do
            # not lift the recording sample cap here; doing so would materialize
            # a full QxK attention tensor and can OOM on long prompts.
            prefill_ctx = nullcontext()
            cuda_event_elapsed_ms: float | None = None
            cuda_peak_allocated: dict[str, int] | None = None
            cuda_peak_reserved: dict[str, int] | None = None
            start_event = None
            end_event = None
            event_device = (
                input_tensor.device
                if getattr(input_tensor, "device", None) is not None
                and input_tensor.device.type == "cuda"
                else None
            )
            cuda_available = bool(self._torch.cuda.is_available())
            if cuda_available:
                for device_idx in range(int(self._torch.cuda.device_count())):
                    self._torch.cuda.reset_peak_memory_stats(device_idx)
                if event_device is not None:
                    with self._torch.cuda.device(event_device):
                        start_event = self._torch.cuda.Event(enable_timing=True)
                        end_event = self._torch.cuda.Event(enable_timing=True)
                        start_event.record()
                _synchronize_cuda_devices(self._torch)
            generate_started = time.perf_counter()
            with (
                self._torch.no_grad(),
                self.capturer.recording_session(
                    call_idx=call_idx,
                    segments=segments,
                    input_token_count=input_token_count,
                    generation=generation_meta,
                ),
                prefill_ctx,
            ):
                sequences = self.model.generate(**generation_kwargs)
                if cuda_available:
                    if end_event is not None and event_device is not None:
                        with self._torch.cuda.device(event_device):
                            end_event.record()
                    _synchronize_cuda_devices(self._torch)
                generation_meta["generate_wall_ms"] = (
                    time.perf_counter() - generate_started
                ) * 1000.0
                if (
                    start_event is not None
                    and end_event is not None
                    and event_device is not None
                ):
                    cuda_event_elapsed_ms = float(start_event.elapsed_time(end_event))
                    generation_meta["cuda_event_elapsed_ms"] = cuda_event_elapsed_ms
                    generation_meta["cuda_event_device"] = str(event_device)
                else:
                    generation_meta["cuda_event_elapsed_ms"] = None
                    generation_meta["cuda_event_device"] = None
                if cuda_available:
                    cuda_peak_allocated = {
                        str(device_idx): int(
                            self._torch.cuda.max_memory_allocated(device_idx)
                        )
                        for device_idx in range(int(self._torch.cuda.device_count()))
                    }
                    cuda_peak_reserved = {
                        str(device_idx): int(
                            self._torch.cuda.max_memory_reserved(device_idx)
                        )
                        for device_idx in range(int(self._torch.cuda.device_count()))
                    }
                generation_meta["cuda_peak_memory_allocated_bytes_by_device"] = (
                    cuda_peak_allocated
                )
                generation_meta["cuda_peak_memory_reserved_bytes_by_device"] = (
                    cuda_peak_reserved
                )
                # `sequences` shape depends on whether session cache was
                # active: with `past_key_values`, HF returns just the new
                # tokens (delta + generated); without, full prompt + generated.
                if used_session_cache:
                    output_ids = (
                        sequences[0, delta_ids.shape[-1] :].detach().cpu().tolist()
                    )
                else:
                    output_ids = (
                        sequences[0, input_token_count:].detach().cpu().tolist()
                    )
                self.capturer.flush(output_token_ids=output_ids)
            if isinstance(self._session_cache, BaseEvictionCache):
                # Detach recorder so the next call's recorder swap is clean.
                # Only eviction caches carry a `recorder` slot.
                self._session_cache.recorder = None
            # Symmetric detach for sparse recorder; pre-hook reads via
            # capturer attribute, so clearing here keeps a stale recorder
            # from being touched between calls.
            self.capturer.set_sparse_recorder(None)
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            self._extend_session_tokens(prompt_ids=prompt_ids, output_ids=output_ids)
            return text, output_ids

        # Pre-clear is defensive against fragmentation accumulated by the
        # previous chat; the next chat's pre-clear will sweep what THIS chat
        # leaves behind, so a post-clear here is pure overhead (each call
        # forces a `cuda.empty_cache()` + `gc.collect()` + sync, hidden
        # ~50-200ms / call on 30B-A3B FP8).
        self._clear_cuda_cache()
        text, output_ids = await asyncio.to_thread(run_generate)
        content, tool_calls = parse_text_tool_calls(text)
        elapsed_ms = (time.time() - started_at) * 1000.0
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={
                "prompt_tokens": input_token_count,
                "completion_tokens": len(output_ids),
                "total_tokens": input_token_count + len(output_ids),
            },
            extra={
                "llm_wall_ts_end": time.time(),
                "llm_call_time_ms": elapsed_ms,
                "llm_timing_source": "hf_recording_generate_wall_ms",
            },
        )


class HFRecordingServer:
    """Minimal OpenAI-compatible chat endpoint for task-container agents."""

    def __init__(
        self,
        provider: HFRecordingProvider,
        *,
        bind_host: str = "0.0.0.0",
        public_host: str | None = None,
    ) -> None:
        self.provider = provider
        self._bind_host = bind_host
        self._public_host = public_host or os.environ.get("HF_RECORDING_PUBLIC_HOST")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def api_base(self) -> str:
        if self._server is None:
            raise RuntimeError("server is not running")
        host, port = self._server.server_address
        if host in {"0.0.0.0", "::"}:
            host = self._public_host or "127.0.0.1"
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "HFRecordingServer":
        provider = self.provider

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path not in {"/v1/chat/completions", "/chat/completions"}:
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("content-length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    extra_body = payload.get("extra_body")
                    if not isinstance(extra_body, dict):
                        extra_body = {}
                    response = asyncio.run(
                        provider.chat(
                            messages=payload.get("messages") or [],
                            tools=payload.get("tools"),
                            model=payload.get("model"),
                            max_tokens=int(payload.get("max_tokens") or 4096),
                            temperature=float(payload.get("temperature") or 0.0),
                            top_p=_optional_float(payload.get("top_p")),
                            top_k=_optional_int(
                                payload.get("top_k", extra_body.get("top_k"))
                            ),
                            repetition_penalty=_optional_float(
                                payload.get(
                                    "repetition_penalty",
                                    extra_body.get("repetition_penalty"),
                                )
                            ),
                            reasoning_effort=payload.get("reasoning_effort"),
                            tool_choice=payload.get("tool_choice"),
                        )
                    )
                    body = self._response_body(response)
                    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except Exception as exc:
                    import traceback

                    sys.stderr.write(
                        f"[HFRecordingServer] chat handler raised: {exc!r}\n"
                        f"{traceback.format_exc()}\n"
                    )
                    sys.stderr.flush()
                    self.send_error(500, str(exc)[:200])

            def log_message(self, _format: str, *args: Any) -> None:
                return

            @staticmethod
            def _response_body(response: LLMResponse) -> dict[str, Any]:
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                }
                if response.tool_calls:
                    message["tool_calls"] = [
                        tool_call.to_openai_tool_call()
                        for tool_call in response.tool_calls
                    ]
                return {
                    "id": "hf-recording",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": provider.default_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": response.finish_reason,
                        }
                    ],
                    "usage": response.usage,
                }

        self._server = ThreadingHTTPServer((self._bind_host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
