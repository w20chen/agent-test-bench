"""Shared helpers for host-mode research-style benchmark plugins."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, ClassVar

from agents.benchmarks.base import Benchmark
from trace_collect.attempt_pipeline import AttemptContext, AttemptResult

INFERENCE_METADATA_FIELDS = ("topic", "difficulty", "domain", "source_urls")


def _require_text(raw: dict[str, Any], field: str, *, label: str) -> str:
    value = raw.get(field)
    if value is None:
        raise ValueError(f"Missing required {label} field {field!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty required {label} field {field!r}")
    return text


def _optional_text(raw: dict[str, Any], field: str | None) -> str | None:
    if not field:
        return None
    value = raw.get(field)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_urls(raw: dict[str, Any], field: str | None) -> list[Any]:
    if not field:
        return []
    value = raw.get(field)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        value = decoded
    if isinstance(value, list):
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(dict(item))
            elif isinstance(item, (list, tuple)) and item:
                entry: dict[str, Any] = {"url": str(item[0]).strip()}
                if len(item) > 1:
                    entry["role"] = item[1]
                normalized.append(entry)
            elif str(item).strip():
                normalized.append(str(item).strip())
        return normalized
    return [str(value).strip()] if str(value).strip() else []


def _decrypt_xor_sha256(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    digest = hashlib.sha256(password.encode()).digest()
    key = digest * (len(encrypted) // len(digest))
    key += digest[: len(encrypted) % len(digest)]
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def _load_hf_rows(
    dataset: str,
    split: str | None,
    *,
    data_files: str | dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore[import]

    if split is None:
        raise ValueError("Host-mode research benchmarks require harness_split")
    kwargs: dict[str, Any] = {"split": split}
    if data_files:
        if isinstance(data_files, str):
            kwargs["data_files"] = {split: data_files}
        else:
            kwargs["data_files"] = data_files
    ds = load_dataset(dataset, **kwargs)
    return [dict(row) for row in ds]


def research_prompt_template_path(benchmark_slug: str, name: str) -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "prompts"
        / benchmark_slug.replace("-", "_")
        / f"{name}.md"
    )


def load_research_prompt_template(benchmark_slug: str, name: str) -> str:
    path = research_prompt_template_path(benchmark_slug, name)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template {name!r} not found at {path}")
    text = path.read_text(encoding="utf-8")
    if "{{task}}" not in text:
        raise ValueError(f"Prompt template {path} is missing '{{{{task}}}}'")
    return text


def research_inference_metadata(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: task[key]
        for key in INFERENCE_METADATA_FIELDS
        if task.get(key)
    }


def render_research_prompt(
    benchmark_slug: str,
    task: dict[str, Any],
    *,
    prompt_template: str,
) -> str:
    template = load_research_prompt_template(benchmark_slug, prompt_template)
    prompt = template.replace("{{task}}", str(task["problem_statement"]))
    metadata = research_inference_metadata(task)
    if metadata:
        prompt += (
            "\n\nInference-time metadata:\n"
            + json.dumps(metadata, ensure_ascii=False, indent=2)
        )
    return prompt


class HostResearchOpenClawRunner:
    """Run one research-style task through OpenClaw in host-controller mode."""

    def __init__(
        self,
        *,
        provider: Any,
        workspace_base: Path,
        max_iterations: int,
        context_window_tokens: int,
        model: str,
        benchmark_slug: str,
        mcp_servers: dict[str, Any] | None = None,
        mcp_config: str | None = None,
        **_: Any,
    ) -> None:
        from agents.openclaw._session_runner import SessionRunner

        self.workspace_base = Path(workspace_base)
        self.model = model
        self.benchmark_slug = benchmark_slug
        self.mcp_config = mcp_config
        self._session_runner = SessionRunner(
            provider,
            model=model,
            max_iterations=max_iterations,
            context_window_tokens=context_window_tokens,
            mcp_servers=mcp_servers or {},
        )

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        workspace = self._workspace_for(attempt_ctx)
        trace_path = attempt_ctx.attempt_dir / "trace.jsonl"
        prompt = self._render_prompt(task, prompt_template=prompt_template)
        result = await self._session_runner.run(
            prompt=prompt,
            workspace=workspace,
            tool_workspace=workspace,
            project_workspace=workspace,
            session_key=self._session_key_for(attempt_ctx),
            trace_file=trace_path,
            instance_id=attempt_ctx.instance_id,
            channel="collect",
        )
        self._stamp_trace_metadata(
            trace_path,
            instance_id=attempt_ctx.instance_id,
            prompt_template=prompt_template,
        )
        n_iterations, total_llm_ms, total_tool_ms, total_tokens = (
            self._trace_summary_totals(trace_path)
        )
        success = result.stop_reason == "completed" and result.error is None
        return AttemptResult(
            success=success,
            exit_status=result.stop_reason,
            trace_path=trace_path,
            model_patch="",
            error=result.error,
            n_iterations=n_iterations,
            total_llm_ms=total_llm_ms,
            total_tool_ms=total_tool_ms,
            total_tokens=total_tokens,
            runtime_proof={
                "agent_runtime_mode": "host_controller",
                "benchmark": self.benchmark_slug,
            },
        )

    def _workspace_for(self, attempt_ctx: AttemptContext) -> Path:
        return (
            self.workspace_base
            / attempt_ctx.instance_id
            / attempt_ctx.attempt_label
        )

    @staticmethod
    def _session_key_for(attempt_ctx: AttemptContext) -> str:
        return f"research:{attempt_ctx.instance_id}:{attempt_ctx.attempt_label}"

    def _stamp_trace_metadata(
        self,
        trace_path: Path,
        *,
        instance_id: str,
        prompt_template: str,
    ) -> None:
        replaced = False
        tmp_path: Path | None = None
        final_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=trace_path.parent,
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                with trace_path.open("r", encoding="utf-8") as src:
                    for line in src:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if not replaced and record.get("type") == "trace_metadata":
                            record.update(
                                {
                                    "benchmark": self.benchmark_slug,
                                    "execution_environment": "host",
                                    "instance_id": instance_id,
                                    "prompt_template": prompt_template,
                                }
                            )
                            if getattr(self, "mcp_config", None) is not None:
                                run_config = record.get("run_config") or {}
                                run_config["mcp_config"] = self.mcp_config
                                record["run_config"] = run_config
                            replaced = True
                        tmp.write(json.dumps(record, ensure_ascii=False))
                        tmp.write("\n")
            if not replaced:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=trace_path.parent,
                    delete=False,
                ) as tmp:
                    final_path = Path(tmp.name)
                    metadata = self._trace_metadata(
                        instance_id=instance_id,
                        prompt_template=prompt_template,
                    )
                    tmp.write(json.dumps(metadata, ensure_ascii=False))
                    tmp.write("\n")
                    assert tmp_path is not None
                    with tmp_path.open("r", encoding="utf-8") as stamped_src:
                        for line in stamped_src:
                            tmp.write(line)
                os.replace(final_path, trace_path)
                return
            assert tmp_path is not None
            os.replace(tmp_path, trace_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
            if final_path is not None and final_path.exists():
                final_path.unlink()

    def _trace_metadata(
        self,
        *,
        instance_id: str,
        prompt_template: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 5,
            "benchmark": self.benchmark_slug,
            "execution_environment": "host",
            "instance_id": instance_id,
            "prompt_template": prompt_template,
        }
        if getattr(self, "mcp_config", None) is not None:
            metadata["run_config"] = {"mcp_config": self.mcp_config}
        return metadata

    def _render_prompt(self, task: dict[str, Any], *, prompt_template: str) -> str:
        return render_research_prompt(
            self.benchmark_slug,
            task,
            prompt_template=prompt_template,
        )

    @staticmethod
    def _trace_summary_totals(
        trace_path: Path,
    ) -> tuple[int | None, float | None, float | None, int | None]:
        n_iterations: int | None = None
        total_llm_ms: float | None = None
        total_tool_ms: float | None = None
        total_tokens: int | None = None
        if not trace_path.exists():
            return n_iterations, total_llm_ms, total_tool_ms, total_tokens
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "summary":
                continue
            n_iterations = record.get("n_iterations")
            total_llm_ms = record.get("total_llm_ms")
            total_tool_ms = record.get("total_tool_ms")
            total_tokens = record.get("total_tokens")
        return n_iterations, total_llm_ms, total_tool_ms, total_tokens


class ResearchBenchmark(Benchmark):
    """Base for host-mode QA/research benchmark plugins."""

    SUPPORTED_SCAFFOLDS: ClassVar[set[str]] = {"openclaw", "tongyi-deepresearch"}

    @property
    def execution_environment(self) -> str:
        return "host"

    def validate_config(self) -> None:
        if not self.config.harness_dataset:
            raise ValueError(f"{self.slug} requires harness_dataset")
        if not self.config.harness_split:
            raise ValueError(f"{self.slug} requires harness_split")

    def validate_scaffold_support(self, scaffold: str) -> None:
        if scaffold not in self.SUPPORTED_SCAFFOLDS:
            raise NotImplementedError(
                f"{self.config.display_name} does not support scaffold={scaffold!r}"
            )

    def runtime_mode_for(self, scaffold: str) -> str:
        self.validate_scaffold_support(scaffold)
        return "host_controller"

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        return None

    def load_tasks(self) -> list[dict[str, Any]]:
        assert self.config.harness_dataset is not None
        data_files = self.config.extras.get("data_files")
        rows = _load_hf_rows(
            self.config.harness_dataset,
            self.config.harness_split,
            data_files=data_files if data_files else None,
        )
        tasks: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            row.setdefault("_row_index", str(index))
            tasks.append(self.normalize_task(row))
        return tasks

    def build_runner(self, *, scaffold: str, **kwargs: Any) -> Any:
        self.validate_scaffold_support(scaffold)
        if scaffold == "openclaw":
            return HostResearchOpenClawRunner(
                benchmark_slug=self.config.slug,
                **kwargs,
            )
        if scaffold == "tongyi-deepresearch":
            from agents.tongyi_deepresearch import TongyiDeepResearchRunner

            return TongyiDeepResearchRunner(
                benchmark_slug=self.config.slug,
                **kwargs,
            )
        raise AssertionError("unreachable")
