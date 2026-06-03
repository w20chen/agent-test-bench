"""DeepResearchBench benchmark plugin."""

from __future__ import annotations

from typing import Any, ClassVar

from agents.benchmarks._research import (
    ResearchBenchmark,
    _optional_text,
    _require_text,
)


class DeepResearchBenchBenchmark(ResearchBenchmark):
    """Host-mode plugin for long-form deep research QA tasks."""

    slug: ClassVar[str] = "deep-research-bench"

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        extras = self.config.extras
        id_field = str(extras.get("id_field", "id"))
        question_field = str(extras.get("question_field", "prompt"))
        answer_field = str(extras.get("answer_field", "article"))
        topic_field = extras.get("topic_field", "topic")
        difficulty_field = extras.get("difficulty_field", "difficulty")
        domain_field = extras.get("domain_field", "domain")

        instance_id = _require_text(raw, id_field, label="instance_id")
        problem_statement = _require_text(
            raw,
            question_field,
            label="problem_statement",
        )
        reference_answer = _require_text(raw, answer_field, label="reference_answer")
        return {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "reference_answer": reference_answer,
            "topic": _optional_text(raw, str(topic_field) if topic_field else None),
            "difficulty": _optional_text(
                raw,
                str(difficulty_field) if difficulty_field else None,
            ),
            "domain": _optional_text(raw, str(domain_field) if domain_field else None),
            "reference_kind": str(extras.get("reference_kind", "generated_report")),
            "repo": None,
            "image_name": None,
            "docker_image": None,
        }
