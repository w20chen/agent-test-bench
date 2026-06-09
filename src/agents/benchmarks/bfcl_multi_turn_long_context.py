"""BFCL multi-turn long-context benchmark plugin.

Same stateful multi-turn tasks as ``multi_turn_base`` but BFCL injects large
synthetic backend state (hundreds of files / thousands of records), inflating
tool-return payloads. The shared runner auto-enables BFCL's ``long_context``
flag from the category name, so this file only declares slug + category.
"""

from __future__ import annotations

from typing import ClassVar

from agents.benchmarks._bfcl import BFCLBenchmark


class BFCLMultiTurnLongContext(BFCLBenchmark):
    slug: ClassVar[str] = "bfcl-multi-turn-long-context"
    bfcl_categories: ClassVar[list[str]] = ["multi_turn_long_context"]
