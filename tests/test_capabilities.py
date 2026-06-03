from __future__ import annotations

import pytest

from agents.capabilities import (
    scaffold_benchmark_matrix,
    validate_scaffold_benchmark,
)


def test_scaffold_benchmark_matrix_uses_plugin_supported_scaffolds() -> None:
    matrix = scaffold_benchmark_matrix()

    assert matrix["openclaw"] >= {
        "swe-bench-verified",
        "swe-rebench",
        "terminal-bench",
        "deep-research-bench",
        "browsecomp",
    }
    assert matrix["tongyi-deepresearch"] == {
        "deep-research-bench",
        "browsecomp",
    }


def test_validate_scaffold_benchmark_accepts_supported_pair() -> None:
    validate_scaffold_benchmark("openclaw", "deep-research-bench")


def test_validate_scaffold_benchmark_rejects_unsupported_pair() -> None:
    with pytest.raises(ValueError, match="does not support"):
        validate_scaffold_benchmark("tongyi-deepresearch", "swe-rebench")


def test_validate_scaffold_benchmark_accepts_tongyi_deepresearch_pair() -> None:
    validate_scaffold_benchmark("tongyi-deepresearch", "deep-research-bench")
    validate_scaffold_benchmark("tongyi-deepresearch", "browsecomp")
