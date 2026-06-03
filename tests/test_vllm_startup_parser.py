from __future__ import annotations
import logging
from pathlib import Path

import pytest

from harness.vllm_startup_parser import parse_startup_log, parse_startup_log_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_startup_log_v05_extracts_baseline():
    result = parse_startup_log_file(FIXTURES / "vllm_startup_0_5.log")
    assert result is not None
    assert result.weights_mib == pytest.approx(3.32 * 1024, rel=0.001)
    assert result.kv_cache_total_mib == pytest.approx(0.5 * 1024, rel=0.001)
    assert result.model == "Qwen/Qwen3-1.7B"
    assert result.dtype == "float16"
    assert result.tensor_parallel_size == 1


def test_parse_startup_log_v06_extracts_baseline():
    result = parse_startup_log_file(FIXTURES / "vllm_startup_0_6.log")
    assert result is not None
    assert result.weights_mib == pytest.approx(14.22 * 1024, rel=0.001)
    assert result.kv_cache_total_mib == pytest.approx(6.0 * 1024, rel=0.001)
    assert result.dtype == "bfloat16"
    assert result.tensor_parallel_size == 1


def test_parse_startup_log_v07_extracts_baseline():
    result = parse_startup_log_file(FIXTURES / "vllm_startup_0_7.log")
    assert result is not None
    assert result.weights_mib == pytest.approx(0.96 * 1024, rel=0.001)
    assert result.kv_cache_total_mib == pytest.approx(0.25 * 1024, rel=0.001)
    assert result.dtype == "float16"
    assert result.model == "facebook/opt-125m"
    assert result.tensor_parallel_size == 2


def test_parse_startup_log_v020_extracts_baseline():
    result = parse_startup_log_file(FIXTURES / "vllm_startup_0_20.log")
    assert result is not None
    assert result.weights_mib == pytest.approx(2.89 * 1024, rel=0.001)
    assert result.kv_cache_total_mib == pytest.approx(22.63 * 1024, rel=0.001)
    assert result.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert result.dtype == "float16"


def test_parse_startup_log_returns_none_when_weights_missing(caplog):
    text = "GPU KV cache size: 32768 tokens, 0.25 GiB\ndtype=float16\n"
    with caplog.at_level(logging.WARNING):
        result = parse_startup_log(text)
    assert result is None
    assert any("model-weights" in r.message for r in caplog.records)


def test_parse_startup_log_returns_none_when_kv_cache_missing(caplog):
    text = "Loading model weights took 3.32 GiB\ndtype=float16\n"
    with caplog.at_level(logging.WARNING):
        result = parse_startup_log(text)
    assert result is None
    assert any("KV cache" in r.message for r in caplog.records)


def test_parse_startup_log_returns_none_for_empty_input(caplog):
    with caplog.at_level(logging.WARNING):
        result = parse_startup_log("")
    assert result is None
