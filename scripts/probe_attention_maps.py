"""Chat rendering helpers used by attention-map probing tests.

The functions here create deterministic Qwen/GLM-style prompts while keeping
character spans for semantic regions. They are analysis utilities only: they do
not run models or produce benchmark results.
"""

from __future__ import annotations

import copy
import json
from typing import Any


Segment = dict[str, int | str]


def _append(parts: list[str], segments: list[Segment], text: str, role: str) -> None:
    start = sum(len(part) for part in parts)
    parts.append(text)
    end = start + len(text)
    if end > start:
        segments.append({"role": role, "start": start, "end": end})


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values: list[str] = []
        for item in content:
            if isinstance(item, dict):
                values.append(str(item.get("output", item.get("text", ""))))
            else:
                values.append(str(item))
        return "\n".join(value for value in values if value)
    if content is None:
        return ""
    return str(content)


def _tool_outputs(content: Any) -> list[str]:
    if isinstance(content, list):
        outputs: list[str] = []
        for item in content:
            if isinstance(item, dict):
                outputs.append(str(item.get("output", item.get("text", ""))))
            else:
                outputs.append(str(item))
        return outputs
    return [_content_text(content)]


def _normalize_tool_arguments(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    function["arguments"] = arguments


def _tool_call_payload(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function", {})
    if not isinstance(function, dict):
        function = {}
    return {
        "name": function.get("name", ""),
        "arguments": function.get("arguments") or {},
    }


def _render_qwen_chat_with_segments(
    messages: list[dict[str, Any]],
) -> tuple[str, list[Segment]]:
    parts: list[str] = []
    segments: list[Segment] = []

    for message in messages:
        role = message.get("role")
        content = _content_text(message.get("content"))
        if role in {"system", "user"}:
            _append(parts, segments, f"<|im_start|>{role}\n", "meta")
            _append(parts, segments, content, str(role))
            _append(parts, segments, "<|im_end|>\n", "meta")
        elif role == "assistant":
            _append(parts, segments, "<|im_start|>assistant", "meta")
            if content:
                _append(parts, segments, f"\n{content}", "assistant_message")
            for tool_call in message.get("tool_calls") or []:
                payload = json.dumps(_tool_call_payload(tool_call), ensure_ascii=False)
                _append(
                    parts,
                    segments,
                    f"\n<tool_call>\n{payload}\n</tool_call>",
                    "assistant_call",
                )
            _append(parts, segments, "<|im_end|>\n", "meta")
        elif role == "tool":
            _append(parts, segments, "<|im_start|>user", "meta")
            for output in _tool_outputs(message.get("content")):
                _append(
                    parts,
                    segments,
                    f"\n<tool_response>\n{output}\n</tool_response>",
                    "tool_result",
                )
            _append(parts, segments, "<|im_end|>\n", "meta")
    _append(parts, segments, "<|im_start|>assistant\n", "gen_prompt")
    return "".join(parts), segments


def _render_glm_chat_with_segments(
    messages: list[dict[str, Any]],
) -> tuple[str, list[Segment]]:
    parts: list[str] = []
    segments: list[Segment] = []
    _append(parts, segments, "[gMASK]<sop>", "meta")

    for message in messages:
        role = message.get("role")
        content = _content_text(message.get("content"))
        if role in {"system", "user"}:
            _append(parts, segments, f"<|{role}|>\n", "meta")
            _append(parts, segments, content, str(role))
        elif role == "assistant":
            _append(parts, segments, "<|assistant|>", "meta")
            _append(parts, segments, "\n<think></think>", "meta")
            if content:
                _append(parts, segments, f"\n{content}", "assistant_message")
            for tool_call in message.get("tool_calls") or []:
                payload = _tool_call_payload(tool_call)
                call = f"\n<tool_call>{payload['name']}"
                arguments = payload.get("arguments") or {}
                if isinstance(arguments, dict):
                    for key, value in arguments.items():
                        call += (
                            f"\n<arg_key>{key}</arg_key>"
                            f"\n<arg_value>{value}</arg_value>"
                        )
                call += "\n</tool_call>"
                _append(parts, segments, call, "assistant_call")
        elif role == "tool":
            _append(parts, segments, "<|observation|>", "meta")
            for output in _tool_outputs(message.get("content")):
                _append(
                    parts,
                    segments,
                    f"\n<tool_response>\n{output}\n</tool_response>",
                    "tool_result",
                )
    _append(parts, segments, "<|assistant|>", "gen_prompt")
    return "".join(parts), segments


def build_chat_with_segments(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> tuple[Any | None, list[Segment] | None]:
    """Tokenize a rendered chat and return semantic token spans."""
    normalized = copy.deepcopy(messages)
    _normalize_tool_arguments(normalized)

    template = getattr(tokenizer, "chat_template", "")
    if "[gMASK]" in template or "<|assistant|>" in template:
        rendered, char_segments = _render_glm_chat_with_segments(normalized)
    else:
        rendered, char_segments = _render_qwen_chat_with_segments(normalized)

    template_rendered = tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
    )
    if template_rendered != rendered:
        rendered = template_rendered

    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[-1] > max_tokens:
        return None, None
    return input_ids, char_segments
