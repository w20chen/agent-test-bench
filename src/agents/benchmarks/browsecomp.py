"""BrowseComp benchmark plugin."""

from __future__ import annotations

from typing import Any, ClassVar

from agents.benchmarks._research import (
    ResearchBenchmark,
    _decrypt_xor_sha256,
    _optional_urls,
    _require_text,
)


class BrowseCompBenchmark(ResearchBenchmark):
    """Host-mode plugin for browsing-comprehension QA tasks."""

    slug: ClassVar[str] = "browsecomp"

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        extras = self.config.extras
        id_field = str(extras.get("id_field", "_row_index"))
        question_field = str(extras.get("question_field", "problem"))
        answer_field = str(extras.get("answer_field", "answer"))
        urls_field = extras.get("source_urls_field", "urls")
        decrypt_fields = {question_field, answer_field}
        if urls_field:
            decrypt_fields.add(str(urls_field))
        row = self._maybe_decrypt_row(raw, encrypted_fields=decrypt_fields)

        instance_id = _require_text(row, id_field, label="instance_id")
        problem_statement = _require_text(
            row,
            question_field,
            label="problem_statement",
        )
        reference_answer = _require_text(row, answer_field, label="reference_answer")
        return {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "reference_answer": reference_answer,
            "source_urls": _optional_urls(
                row,
                str(urls_field) if urls_field else None,
            ),
            "repo": None,
            "image_name": None,
            "docker_image": None,
        }

    def _maybe_decrypt_row(
        self,
        raw: dict[str, Any],
        *,
        encrypted_fields: set[str],
    ) -> dict[str, Any]:
        if not self.config.extras.get("encrypted", False):
            return raw
        canary = _require_text(raw, "canary", label="canary")
        row = dict(raw)
        for field in encrypted_fields:
            if field in row and row[field] is not None:
                row[field] = _decrypt_xor_sha256(str(row[field]), canary)
        return row
