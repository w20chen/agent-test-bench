"""Tests for _parse_qwen_xml_tool_calls lenient fallback (bare <function=...> blocks)."""

import pytest
from serving.recording.backend_hf import _parse_qwen_xml_tool_calls


def test_wrap_present_valid():
    """Regression: wrapped format still parses correctly."""
    text = "<tool_call><function=ls><parameter=path>/app</parameter></function></tool_call>"
    content, calls = _parse_qwen_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "ls"
    assert calls[0].arguments == {"path": "/app"}
    assert content is None


def test_wrap_missing_only_closer():
    """Observed failure: model dropped opening <tool_call>, only closing </tool_call> present."""
    text = (
        "<function=exec>\n"
        "<parameter=command>\n"
        "python3 /app/aggregate_data.py\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    content, calls = _parse_qwen_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "exec"
    assert calls[0].arguments == {"command": "python3 /app/aggregate_data.py"}


def test_wrap_missing_entirely():
    """Bare <function=...></function> with no <tool_call> tags at all."""
    text = "<function=read_file><parameter=path>/etc/hosts</parameter></function>"
    content, calls = _parse_qwen_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/etc/hosts"}


def test_plain_text_no_calls():
    """No function blocks returns empty call list."""
    text = "Just a thought, no calls here."
    content, calls = _parse_qwen_xml_tool_calls(text)
    assert calls == []
    assert content == text
