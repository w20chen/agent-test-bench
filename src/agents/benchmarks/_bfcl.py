"""Shared helpers for BFCL host-mode benchmark plugins.

BFCL (Berkeley Function Calling Leaderboard) is an external, read-only repo
located via the ``BFCL_REPO_PATH`` env var (or ``extras.bfcl_repo_path`` in the
benchmark YAML); its Python package lives under
``<root>/berkeley-function-call-leaderboard``.  We never modify BFCL code.  We
import its dataset loader, its function-doc -> tool-schema converter, its
stateful backend instantiation helper, and the backend implementations
themselves (simulated filesystem, booking, web search, vector memory), then
drive each BFCL task through OpenClaw's event loop so the existing host-mode
trace + resource-monitoring + HTML pipeline captures the run.

A BFCL "task" is one BFCL test entry.  Each entry is a multi-turn conversation
against one or more stateful backend classes.  We:

1. instantiate the backend classes (reusing BFCL's ``execute_multi_turn_func_call``),
2. wrap each public backend method as an OpenClaw :class:`Tool` whose schema is
   produced by BFCL's own ``convert_to_tool``,
3. swap those tools into a fresh :class:`AgentLoop` (replacing OpenClaw's default
   tools) and write any BFCL system prompt to ``<workspace>/AGENTS.md`` so
   OpenClaw's ``ContextBuilder`` wraps it,
4. drive each conversation turn as one bus message on a single persistent
   session, letting OpenClaw issue the LLM calls and run the loop.

The trace is recorded by the same ``TraceCollectorHook`` used by every OpenClaw
collection run, so ``llm_call``/``tool_exec`` actions and timing are captured
identically to other benchmarks.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks.base import Benchmark
from agents.openclaw.tools.base import Tool
from trace_collect.attempt_pipeline import AttemptContext, AttemptResult

DEFAULT_BFCL_PACKAGE_SUBDIR = "berkeley-function-call-leaderboard"

# Channel constant mirroring OpenClaw's SessionRunner: a "system"-channel inbound
# whose chat_id encodes "<channel>:<chat_id>" is unwrapped by AgentLoop into the
# real channel/chat_id (see _loop.py:_process_message). We use "collect" so the
# trace is tagged like other collection runs.
_AGENT_CHANNEL = "collect"


# ---------------------------------------------------------------------------
# BFCL import bootstrap
# ---------------------------------------------------------------------------


def _resolve_bfcl_package_root(extras: dict[str, Any]) -> Path:
    # No machine-specific default: the BFCL checkout location must be supplied
    # explicitly (env var preferred, YAML extras as fallback).
    root = os.environ.get("BFCL_REPO_PATH") or extras.get("bfcl_repo_path")
    if not root:
        raise FileNotFoundError(
            "BFCL repo path is not configured. Set the BFCL_REPO_PATH environment "
            "variable to your gorilla checkout (the directory containing "
            f"{DEFAULT_BFCL_PACKAGE_SUBDIR}/), or set extras.bfcl_repo_path in the "
            "benchmark YAML."
        )
    subdir = extras.get("bfcl_package_subdir", DEFAULT_BFCL_PACKAGE_SUBDIR)
    return Path(root).expanduser() / subdir


def ensure_bfcl_importable(extras: dict[str, Any]) -> Path:
    """Add the BFCL package root to ``sys.path`` so ``import bfcl_eval`` works.

    Returns the resolved package root.  Raises ``FileNotFoundError`` with an
    actionable message when the path does not point at a BFCL checkout.
    """
    pkg_root = _resolve_bfcl_package_root(extras)
    if not (pkg_root / "bfcl_eval").is_dir():
        raise FileNotFoundError(
            f"BFCL package not found at {pkg_root}. Set BFCL_REPO_PATH to the "
            f"gorilla repo root that contains {DEFAULT_BFCL_PACKAGE_SUBDIR}/bfcl_eval, "
            "or set extras.bfcl_repo_path in the benchmark YAML."
        )
    path_str = str(pkg_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return pkg_root


def _sanitize(text: str) -> str:
    return re.sub(r"[-./:]", "_", text)


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------


def _serialize_tool_result(result: Any) -> str:
    """Serialize a backend return value to a string tool result.

    Mirrors BFCL's own serialization in
    ``multi_turn_utils.execute_multi_turn_func_call`` (str passthrough, dict ->
    JSON, everything else -> ``str``).
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
    return str(result)


class BFCLTool(Tool):
    """An OpenClaw tool that dispatches to one BFCL backend bound method.

    The schema (name / description / parameters) comes from BFCL's own
    ``convert_to_tool`` output, so the LLM sees exactly BFCL's function docs.
    Execution calls the backend instance method directly (no ``eval``).
    """

    def __init__(self, schema: dict[str, Any], instance: Any, method_name: str) -> None:
        fn = schema.get("function", schema)
        self._name: str = fn["name"]
        self._description: str = fn.get("description", "") or ""
        self._parameters: dict[str, Any] = fn.get(
            "parameters", {"type": "object", "properties": {}}
        )
        self._instance = instance
        self._method_name = method_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        method = getattr(self._instance, self._method_name)
        try:
            # Backend methods are synchronous; calling them inside the coroutine
            # without awaiting keeps each call atomic under asyncio's
            # single-threaded scheduler (BFCL backends are not thread-safe).
            result = method(**kwargs)
        except Exception as exc:  # surfaced to the model as a tool error
            return f"Error during execution: {exc}"
        return _serialize_tool_result(result)


def build_bfcl_tools(
    test_entry: dict[str, Any], involved_instances: dict[str, Any]
) -> list[BFCLTool]:
    """Convert a BFCL entry's function docs into OpenClaw tools.

    Uses BFCL's ``convert_to_tool`` for the OpenAI-format schema and maps each
    function-doc name to the backend instance that exposes it (mirroring
    ``execute_multi_turn_func_call``'s method->instance resolution).
    """
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool

    functions: list[dict[str, Any]] = test_entry["function"]
    schemas = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)

    method_to_instance: dict[str, Any] = {}
    for instance in involved_instances.values():
        for method_name, _ in inspect.getmembers(instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            method_to_instance[method_name] = instance

    tools: list[BFCLTool] = []
    # convert_to_tool preserves order and deep-copies, so functions[i] is the
    # original doc for schemas[i]; we dispatch via the ORIGINAL method name.
    for func, schema in zip(functions, schemas):
        original_name = func["name"]
        instance = method_to_instance.get(original_name)
        if instance is None:
            raise ValueError(
                f"BFCL function {original_name!r} is not exposed by any involved "
                f"backend instance ({list(involved_instances)})"
            )
        tools.append(BFCLTool(schema, instance, original_name))
    return tools


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------


def _split_question(
    question: list[list[dict[str, Any]]],
) -> tuple[str, list[str]]:
    """Split a BFCL question into (system_prompt_text, per_turn_user_texts).

    System messages (injected by BFCL's agentic/memory processors) are pulled
    out and concatenated; each turn contributes the concatenation of its
    ``user`` messages.  Assistant-prefilled turns are not used by the four
    targeted categories and raise to avoid silent mishandling.
    """
    system_parts: list[str] = []
    turn_texts: list[str] = []
    for turn in question:
        user_parts: list[str] = []
        for message in turn:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                if content:
                    system_parts.append(str(content))
            elif role == "user":
                user_parts.append(str(content))
            else:
                raise ValueError(
                    f"Unsupported message role {role!r} in BFCL turn; only "
                    "'system' and 'user' are handled by these categories."
                )
        turn_texts.append("\n\n".join(user_parts))
    return "\n\n".join(system_parts), turn_texts


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BFCLOpenClawRunner:
    """Run one BFCL task end-to-end through OpenClaw in host-controller mode.

    The constructor signature accepts the same kwargs the collector passes to
    every host runner (see ``collector.collect_traces``); unused ones are
    swallowed via ``**_``.
    """

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
        self.provider = provider
        self.workspace_base = Path(workspace_base)
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.model = model
        self.benchmark_slug = benchmark_slug
        self.mcp_servers = mcp_servers or {}
        self.mcp_config = mcp_config

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )
        from bfcl_eval.utils import is_memory, is_web_search

        entry: dict[str, Any] = task["_bfcl_entry"]
        category: str = task["_bfcl_category"]
        test_id: str = entry["id"]
        involved_classes: list[str] = entry["involved_classes"]
        long_context = "long_context" in category or "composite" in category

        # Per-attempt model name so cached backend instances (kept in BFCL module
        # globals keyed by model+test_id) are never reused across attempts.
        bfcl_model_name = _sanitize(f"{self.model}__{attempt_ctx.attempt_label}")

        workspace = self.workspace_base / test_id / attempt_ctx.attempt_label
        workspace.mkdir(parents=True, exist_ok=True)
        trace_path = attempt_ctx.attempt_dir / "trace.jsonl"

        # Memory categories load their initial state from on-disk snapshots; the
        # snapshot dir must be stable across the (sequential) prereq + question
        # tasks of one run, so we derive it from the shared run_dir.
        if is_memory(category):
            self._populate_memory_initial_config(entry, attempt_ctx)
        elif is_web_search(category):
            # WebSearchAPI reads show_snippet from initial_config; base -> True,
            # no_snippet -> False (derived from the entry id by BFCL's helper).
            from bfcl_eval.utils import (
                populate_initial_settings_for_web_search_test_cases,
            )

            populate_initial_settings_for_web_search_test_cases([entry])

        initial_config: dict[str, Any] = entry.get("initial_config", {})

        # Instantiate the stateful backends (reusing BFCL); empty call list just
        # returns the instances with their initial state loaded.
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            bfcl_model_name,
            test_id,
            long_context=long_context,
            is_evaL_run=False,
        )

        # Memory: inject BFCL's memory-system-prompt (with core-memory dump) the
        # same way base_handler.inference_multi_turn_FC does.
        if is_memory(category):
            from bfcl_eval.model_handler.utils import (
                add_memory_instruction_system_prompt,
            )

            assert len(involved_instances) == 1, "Memory tasks use one backend."
            memory_instance = next(iter(involved_instances.values()))
            entry["question"] = add_memory_instruction_system_prompt(
                entry["question"], category, entry["scenario"], memory_instance
            )

        system_prompt, turn_texts = _split_question(entry["question"])
        if system_prompt:
            (workspace / "AGENTS.md").write_text(system_prompt, encoding="utf-8")

        tools = build_bfcl_tools(entry, involved_instances)

        result = await self._drive_conversation(
            test_id=test_id,
            trace_path=trace_path,
            workspace=workspace,
            turn_texts=turn_texts,
            tools=tools,
            prompt_template=prompt_template,
        )

        # Memory: persist final state so a later question entry can load it.
        if is_memory(category):
            next(iter(involved_instances.values()))._flush_memory_to_local_file()

        return result

    def _populate_memory_initial_config(
        self, entry: dict[str, Any], attempt_ctx: AttemptContext
    ) -> None:
        from bfcl_eval.utils import populate_initial_settings_for_memory_test_cases

        memory_state_dir = attempt_ctx.run_dir / "_memory_state"
        memory_state_dir.mkdir(parents=True, exist_ok=True)
        populate_initial_settings_for_memory_test_cases([entry], memory_state_dir)

    async def _drive_conversation(
        self,
        *,
        test_id: str,
        trace_path: Path,
        workspace: Path,
        turn_texts: list[str],
        tools: list[BFCLTool],
        prompt_template: str,
    ) -> AttemptResult:
        from agents.openclaw._loop import AgentLoop
        from agents.openclaw._session_runner import (
            TraceCollectorHook,
            inject_event_callbacks,
        )
        from agents.openclaw.bus.events import InboundMessage
        from agents.openclaw.bus.queue import MessageBus
        from agents.openclaw.eval.collector import ResultCollector
        from agents.openclaw.session.manager import SessionManager

        trace_hook = TraceCollectorHook(
            trace_path, test_id, agent_id=test_id, task_id=test_id
        )
        trace_hook.add_record(self._trace_metadata(test_id, tools, prompt_template))

        bus = MessageBus()
        collector = ResultCollector(bus)
        session_manager = SessionManager(workspace)
        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            workspace=workspace,
            tool_workspace=workspace,
            project_workspace=workspace,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            session_manager=session_manager,
            mcp_servers=self.mcp_servers,
            hooks=[trace_hook],
        )
        # Replace OpenClaw's default tools with BFCL's so the agent sees exactly
        # the benchmark's tool surface.
        for name in list(agent.tools.tool_names):
            agent.tools.unregister(name)
        for tool in tools:
            agent.tools.register(tool)
        inject_event_callbacks(agent, trace_hook)

        session_key = f"bfcl:{test_id}"
        chat_id = test_id
        result_key = f"{_AGENT_CHANNEL}:{chat_id}"
        wall_start = time.monotonic()
        stop_reason = "completed"
        error: str | None = None

        async with AsyncExitStack() as stack:
            await collector.start()
            stack.callback(collector.stop)
            agent_task = asyncio.create_task(agent.run())
            stack.callback(agent.stop)

            for turn_text in turn_texts:
                # Fresh completion signal + result buffer for this turn.
                collector._results.pop(result_key, None)
                collector._done_events.pop(result_key, None)
                await bus.publish_inbound(
                    InboundMessage(
                        channel="system",
                        sender_id="user",
                        chat_id=f"{_AGENT_CHANNEL}:{chat_id}",
                        content=turn_text,
                        session_key_override=session_key,
                    )
                )
                await collector.wait_for_result(result_key)

        elapsed_s = time.monotonic() - wall_start
        try:
            await asyncio.wait_for(agent_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        # The session is keyed internally by the unwrapped channel:chat_id.
        outcome = getattr(agent, "_last_run_outcomes", {}).get(result_key, {})
        stop_reason = str(outcome.get("stop_reason") or "completed")
        error = outcome.get("error")

        await trace_hook.write_summary(
            success=stop_reason == "completed",
            elapsed_s=elapsed_s,
        )

        n_iterations, total_llm_ms, total_tool_ms, total_tokens = _read_summary_totals(
            trace_path
        )
        return AttemptResult(
            success=stop_reason == "completed" and error is None,
            exit_status=stop_reason,
            trace_path=trace_path,
            model_patch="",
            error=error,
            n_iterations=n_iterations,
            total_llm_ms=total_llm_ms,
            total_tool_ms=total_tool_ms,
            total_tokens=total_tokens,
            runtime_proof={
                "agent_runtime_mode": "host_controller",
                "benchmark": self.benchmark_slug,
            },
        )

    def _trace_metadata(
        self, test_id: str, tools: list[BFCLTool], prompt_template: str
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 5,
            "mode": "collect",
            "model": self.model,
            "instance_id": test_id,
            "benchmark": self.benchmark_slug,
            "execution_environment": "host",
            "prompt_template": prompt_template,
            "max_iterations": self.max_iterations,
            "scaffold_capabilities": {
                "tools": [tool.name for tool in tools],
                "memory": False,
                "skills": False,
                "file_ops": "bfcl_backend",
            },
        }
        if self.mcp_config is not None:
            metadata["run_config"] = {"mcp_config": self.mcp_config}
        return metadata


def _read_summary_totals(
    trace_path: Path,
) -> tuple[int | None, float | None, float | None, int | None]:
    if not trace_path.exists():
        return None, None, None, None
    n_iterations = total_llm_ms = total_tool_ms = total_tokens = None
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "summary":
            n_iterations = record.get("n_iterations")
            total_llm_ms = record.get("total_llm_ms")
            total_tool_ms = record.get("total_tool_ms")
            total_tokens = record.get("total_tokens")
    return n_iterations, total_llm_ms, total_tool_ms, total_tokens


# ---------------------------------------------------------------------------
# Benchmark base
# ---------------------------------------------------------------------------


class BFCLBenchmark(Benchmark):
    """Base for host-mode BFCL benchmark plugins.

    Subclasses set :attr:`bfcl_categories` (the BFCL test categories whose
    entries make up this dataset).  Everything else — loading, normalization,
    runner construction — is shared.
    """

    #: BFCL test categories loaded for this benchmark (set by subclasses).
    bfcl_categories: ClassVar[list[str]] = []

    SUPPORTED_SCAFFOLDS: ClassVar[set[str]] = {"openclaw"}

    @property
    def execution_environment(self) -> str:
        return "host"

    def validate_config(self) -> None:
        if not self.bfcl_categories:
            raise ValueError(f"{type(self).__name__} must set bfcl_categories")
        # Fail fast if BFCL is not reachable so misconfiguration surfaces early.
        ensure_bfcl_importable(self.config.extras)

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

    def viz_filename(self, instance_id: str) -> str:
        # "<slug>__<instance_id>.html" so a full run's HTMLs are uniquely
        # identifiable and collectible into one folder.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", instance_id)
        return f"{self.config.slug}__{safe}.html"

    def load_tasks(self) -> list[dict[str, Any]]:
        ensure_bfcl_importable(self.config.extras)
        from bfcl_eval.utils import load_dataset_entry

        tasks: list[dict[str, Any]] = []
        for category in self.bfcl_categories:
            for entry in load_dataset_entry(category):
                task = self.normalize_task(entry)
                task["_bfcl_category"] = category
                tasks.append(task)
        return tasks

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "instance_id": raw["id"],
            "_bfcl_entry": raw,
            "repo": None,
            "image_name": None,
            "docker_image": None,
        }

    def build_runner(self, *, scaffold: str, **kwargs: Any) -> Any:
        self.validate_scaffold_support(scaffold)
        return BFCLOpenClawRunner(benchmark_slug=self.config.slug, **kwargs)
