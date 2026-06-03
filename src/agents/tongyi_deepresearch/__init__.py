"""Tongyi-DeepResearch scaffold (vendored from Alibaba-NLP/DeepResearch).

Ralplan R3: exports ``TongyiDeepResearchRunner`` as the sole public entrypoint.
Vendor source lives under ``vendor/`` and is pinned at SHA
``f72f75d8c3eb842f2bbbab096a12206ff66e270f``. See ``VENDOR_NOTES.md`` for the
patch audit trail.
"""

from agents.tongyi_deepresearch.runner import TongyiDeepResearchRunner

__all__ = ["TongyiDeepResearchRunner"]
