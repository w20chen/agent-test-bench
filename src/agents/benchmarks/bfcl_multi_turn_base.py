"""BFCL multi-turn base benchmark plugin.

Runs BFCL's ``multi_turn_base`` category (stateful multi-turn tool use over
backends such as GorillaFileSystem / TravelAPI / TradingBot) through OpenClaw.
All logic lives in :mod:`agents.benchmarks._bfcl`; this file only declares the
slug and the BFCL category it loads.
"""

from __future__ import annotations

from typing import ClassVar

from agents.benchmarks._bfcl import BFCLBenchmark


class BFCLMultiTurnBase(BFCLBenchmark):
    slug: ClassVar[str] = "bfcl-multi-turn-base"
    bfcl_categories: ClassVar[list[str]] = ["multi_turn_base"]
