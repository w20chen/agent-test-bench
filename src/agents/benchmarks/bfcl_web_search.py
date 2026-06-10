"""BFCL web-search benchmark plugin.

Agentic single-user-turn, multi-step web research over BFCL's WebSearchAPI
(``search_engine_query`` via SerpAPI + ``fetch_url_content`` via requests).
Loads both BFCL sub-categories from the same source file:

- ``web_search_base``: search results include snippets (show_snippet=True).
- ``web_search_no_snippet``: snippets withheld, forcing full-page fetches.

The shared runner sets WebSearchAPI's ``show_snippet`` from the entry id.
Requires a web-search API key at run time (e.g. SERPAPI_API_KEY).
"""

from __future__ import annotations

from typing import ClassVar

from agents.benchmarks._bfcl import BFCLBenchmark


class BFCLWebSearch(BFCLBenchmark):
    slug: ClassVar[str] = "bfcl-web-search"
    bfcl_categories: ClassVar[list[str]] = ["web_search_base", "web_search_no_snippet"]
