from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalTask:
    instance_id: str
    problem_statement: str
    workspace_dir: Path

    repo: str | None = None
    base_commit: str | None = None
    image_name: str | None = None


@dataclass
class EvalResult:
    instance_id: str
    content: str | None
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    trace_file: Path | None = None
    prepare_ms: float | None = None
    run_ms: float | None = None
    workspace_dir: Path | None = None
    base_commit: str | None = None
    official_resolved: bool | None = None
    evaluation_report: dict[str, Any] | None = None
    n_iterations: int | None = None
    container_model_patch: str | None = None

    @property
    def model_patch(self) -> str:
        # Prefer the runner-captured diff when available. Otherwise fall back
        # to the workspace diff, then to content scraping.
        if self.container_model_patch:
            return self.container_model_patch
        patch = self._extract_patch_from_workspace()
        if patch:
            return patch
        return self._extract_patch_from_content()

    # Exclude runtime-owned files so the extracted patch stays source-only.
    _EXCLUDE_PATTERNS = [
        ".nanobot",
        "memory",
        "sessions",
        "trace.jsonl",
        "MEMORY.md",
        "HISTORY.md",
        ".omc",
        "patch.txt",
        "local.bin",
    ]

    def _extract_patch_from_workspace(self) -> str:
        import subprocess

        if not self.workspace_dir or not (self.workspace_dir / ".git").exists():
            return ""
        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.workspace_dir,
                capture_output=True,
                timeout=30,
            )
            diff_target = self.base_commit or "HEAD"
            cmd = ["git", "diff", diff_target, "--", "."]
            for pat in self._EXCLUDE_PATTERNS:
                cmd.append(f":(exclude){pat}")
            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _extract_patch_from_content(self) -> str:
        if not self.content:
            return ""
        import re

        text = self.content.strip()
        if text.startswith("diff --git"):
            return text
        match = re.search(r"```diff\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(diff --git.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("diff --git"):
                return "\n".join(lines[i:]).strip()
        return ""


@dataclass
class EvalTraceSummary:
    type: str = "summary"
    agent_id: str = ""
    program_id: str = ""
    task_id: str = ""
    instance_id: str = ""
    n_iterations: int = 0
    total_llm_ms: float = 0.0
    total_llm_wall_ms: float = 0.0
    total_llm_call_time_ms: float = 0.0
    llm_call_time_count: int = 0
    llm_timing_source: str = "wall_clock_ms"
    total_tool_ms: float = 0.0
    total_tokens: int = 0
    tool_ms_by_name: dict[str, float] = field(default_factory=dict)
    tool_timeouts: dict[str, int] = field(default_factory=dict)
    success: bool | None = None
    elapsed_s: float = 0.0
    prepare_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "agent_id": self.agent_id,
            "program_id": self.program_id,
            "task_id": self.task_id,
            "instance_id": self.instance_id,
            "n_iterations": self.n_iterations,
            "total_llm_ms": self.total_llm_ms,
            "total_llm_wall_ms": self.total_llm_wall_ms,
            "total_llm_call_time_ms": self.total_llm_call_time_ms,
            "llm_call_time_count": self.llm_call_time_count,
            "llm_timing_source": self.llm_timing_source,
            "total_tool_ms": self.total_tool_ms,
            "total_tokens": self.total_tokens,
            "tool_ms_by_name": self.tool_ms_by_name,
            "tool_timeouts": self.tool_timeouts,
            "success": self.success,
            "elapsed_s": self.elapsed_s,
            "prepare_ms": self.prepare_ms,
        }


SCHEDULING = "SCHEDULING"
SESSION = "SESSION"
CONTEXT = "CONTEXT"
LLM = "LLM"
TOOL = "TOOL"
MCP = "MCP"
MEMORY = "MEMORY"
SUBAGENT = "SUBAGENT"

ALL_CATEGORIES = frozenset(
    [
        SCHEDULING,
        SESSION,
        CONTEXT,
        LLM,
        TOOL,
        MCP,
        MEMORY,
        SUBAGENT,
    ]
)


@dataclass
class EvalTraceEvent:
    agent_id: str
    program_id: str
    instance_id: str
    event: str
    category: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0
    iteration: int = 0
    type: str = "event"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "agent_id": self.agent_id,
            "program_id": self.program_id,
            "instance_id": self.instance_id,
            "event": self.event,
            "category": self.category,
            "data": self.data,
            "ts": self.ts,
            "iteration": self.iteration,
        }
