from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_attention_fullseq_downsample import _require_samples  # noqa: E402
from probe_attention_maps import (  # noqa: E402
    _render_glm_chat_with_segments,
    _render_qwen_chat_with_segments,
    build_chat_with_segments,
)


class _FakeEncoding(dict):
    @property
    def input_ids(self) -> torch.Tensor:
        return self["input_ids"]


class _GoldenTokenizer:
    def __init__(self, family: str, expected_text: str) -> None:
        self.family = family
        self.expected_text = expected_text
        self.seen_messages: list[dict[str, Any]] | None = None
        if family == "qwen":
            self.chat_template = "<|im_start|><tool_call><tool_response>"
        elif family == "glm":
            self.chat_template = "[gMASK]<sop><|assistant|><tool_call><arg_key>"
        else:
            raise ValueError(f"unknown fake tokenizer family: {family}")

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if tokenize:
            raise ValueError("tests only exercise tokenize=False")
        if not add_generation_prompt:
            raise ValueError("tests require add_generation_prompt=True")
        self.seen_messages = messages
        return self.expected_text

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> _FakeEncoding:
        if add_special_tokens:
            raise ValueError("tests expect add_special_tokens=False")
        input_ids = torch.arange(len(text), dtype=torch.long).reshape(1, -1)
        encoded = _FakeEncoding({"input_ids": input_ids})
        if return_offsets_mapping:
            encoded["offset_mapping"] = torch.tensor(
                [[(index, index + 1) for index in range(len(text))]],
                dtype=torch.long,
            )
        return encoded


def _sample_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are precise."},
        {"role": "user", "content": "Find x."},
        {
            "role": "assistant",
            "content": "I will search.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "x", "k": 3},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
        {"role": "assistant", "content": "Final answer."},
    ]


def _expected_qwen_prompt() -> str:
    return (
        "<|im_start|>system\n"
        "You are precise.<|im_end|>\n"
        "<|im_start|>user\n"
        "Find x.<|im_end|>\n"
        "<|im_start|>assistant"
        "\nI will search."
        "\n<tool_call>\n"
        '{"name": "search", "arguments": {"query": "x", "k": 3}}'
        "\n</tool_call><|im_end|>\n"
        "<|im_start|>user"
        "\n<tool_response>\n"
        "result text"
        "\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
        "Final answer.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _expected_glm_prompt() -> str:
    return (
        "[gMASK]<sop>"
        "<|system|>\n"
        "You are precise."
        "<|user|>\n"
        "Find x."
        "<|assistant|>"
        "\n<think></think>"
        "\nI will search."
        "\n<tool_call>search"
        "\n<arg_key>query</arg_key>"
        "\n<arg_value>x</arg_value>"
        "\n<arg_key>k</arg_key>"
        "\n<arg_value>3</arg_value>"
        "\n</tool_call>"
        "<|observation|>"
        "\n<tool_response>\n"
        "result text"
        "\n</tool_response>"
        "<|assistant|>"
        "\n<think></think>"
        "\nFinal answer."
        "<|assistant|>"
    )


def test_qwen_renderer_splits_assistant_call_and_tool_result() -> None:
    rendered, char_segments = _render_qwen_chat_with_segments(_sample_messages())

    assert rendered == _expected_qwen_prompt()
    roles = [segment["role"] for segment in char_segments]
    assert "assistant_message" in roles
    assert "assistant_call" in roles
    assert "tool_result" in roles
    assert "meta" in roles
    assert roles[-1] == "gen_prompt"


def test_glm_renderer_splits_assistant_call_and_tool_result() -> None:
    rendered, char_segments = _render_glm_chat_with_segments(_sample_messages())

    assert rendered == _expected_glm_prompt()
    roles = [segment["role"] for segment in char_segments]
    assert "assistant_message" in roles
    assert "assistant_call" in roles
    assert "tool_result" in roles
    assert "meta" in roles
    assert roles[-1] == "gen_prompt"


def test_build_chat_with_segments_uses_fine_roles_for_qwen_and_glm() -> None:
    cases = {
        "qwen": _expected_qwen_prompt(),
        "glm": _expected_glm_prompt(),
    }
    for family, expected_text in cases.items():
        input_ids, segments = build_chat_with_segments(
            _GoldenTokenizer(family, expected_text), _sample_messages(), max_tokens=4096
        )

        assert input_ids is not None
        assert segments is not None
        roles = {segment["role"] for segment in segments}
        assert {
            "assistant_message",
            "assistant_call",
            "tool_result",
            "meta",
            "gen_prompt",
        } <= roles
        assert all(segment["start"] < segment["end"] for segment in segments)


def test_qwen_build_normalizes_string_tool_arguments() -> None:
    messages = _sample_messages()
    messages[2]["tool_calls"][0]["function"]["arguments"] = '{"query": "x", "k": 3}'
    tokenizer = _GoldenTokenizer("qwen", _expected_qwen_prompt())

    _input_ids, segments = build_chat_with_segments(
        tokenizer, messages, max_tokens=4096
    )

    assert segments is not None
    assert tokenizer.seen_messages is not None
    assert (
        tokenizer.seen_messages[2]["tool_calls"][0]["function"]["arguments"]
        == {"query": "x", "k": 3}
    )


def test_glm_renderer_matches_list_tool_result_template() -> None:
    messages = [{"role": "tool", "content": [{"output": "one"}, {"output": "two"}]}]
    expected = (
        "[gMASK]<sop>"
        "<|observation|>"
        "\n<tool_response>\n"
        "one"
        "\n</tool_response>"
        "\n<tool_response>\n"
        "two"
        "\n</tool_response>"
        "<|assistant|>"
    )

    rendered, char_segments = _render_glm_chat_with_segments(messages)
    input_ids, segments = build_chat_with_segments(
        _GoldenTokenizer("glm", expected), messages, max_tokens=4096
    )

    assert rendered == expected
    assert input_ids is not None
    assert segments is not None
    assert [segment["role"] for segment in char_segments] == [
        "meta",
        "meta",
        "tool_result",
        "tool_result",
        "gen_prompt",
    ]


def test_plotter_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="no samples"):
        _require_samples({"samples": []})
