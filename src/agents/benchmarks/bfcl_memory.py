"""BFCL memory benchmark plugin (vector backend).

Agentic persistent-memory tasks over BFCL's MemoryAPI_vector backend
(SentenceTransformer embeddings + FAISS). Each scenario is a chain of
pre-requisite "memory write" conversations followed by the actual question
entries; the prereq entries must run first so memory state (persisted to
on-disk snapshots by the shared runner) is populated before the questions.

``load_dataset_entry`` returns prereq entries ahead of their questions, so we
override :meth:`select_subset` to preserve that load order instead of sorting
by id. Live runs require ``faiss-cpu`` + ``sentence-transformers``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agents.benchmarks._bfcl import BFCLBenchmark


class BFCLMemory(BFCLBenchmark):
    slug: ClassVar[str] = "bfcl-memory"
    bfcl_categories: ClassVar[list[str]] = ["memory_vector"]

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Preserve load order so prereq entries precede their question entries.

        Sorting by instance_id (the base default) would place question entries
        before their prerequisites and break memory population.
        """
        effective_n = n if n is not None else self.config.selection_n
        return tasks[:effective_n]
