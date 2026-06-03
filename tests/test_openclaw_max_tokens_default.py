from __future__ import annotations

import inspect
from pathlib import Path


def test_unified_provider_default_max_tokens_is_4096() -> None:
    from llm_call.openclaw import UnifiedProvider

    sig = inspect.signature(UnifiedProvider.__init__)
    assert sig.parameters["max_tokens"].default == 4096


def test_agent_defaults_max_tokens_is_4096() -> None:
    from agents.openclaw.config.schema import AgentDefaults

    assert AgentDefaults().max_tokens == 4096


def test_cli_max_tokens_default_is_4096() -> None:
    from agents.openclaw._cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if "--max-tokens" in action.option_strings:
            assert action.default == 4096
            return
    raise AssertionError("--max-tokens action not found in parser")


def test_hf_recording_default_max_tokens_is_4096() -> None:
    # Static source check — avoids loading torch/transformers in CI.
    src = Path(__file__).resolve().parents[1] / "src/serving/recording/backend_hf.py"
    text = src.read_text(encoding="utf-8")
    assert "max_tokens=4096" in text, "HFRecordingProvider max_tokens default drifted"
