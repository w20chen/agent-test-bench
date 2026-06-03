"""TongyiDeepResearchRunner: async Runner adapter for the vendored ReAct scaffold.

Wires the adapter-layer trace infrastructure (``trace.py``) into the vendored
``MultiTurnReactAgent`` from ``Alibaba-NLP/DeepResearch``. Uses a context
manager to monkey-patch ``vendor.OpenAI`` and ``vendor.TOOL_CLASS`` for the
duration of one task, then restores module state. Vendor code is synchronous;
we dispatch it via ``asyncio.to_thread``.

Exit status enum:
- ``completed``: vendor returned ``termination=='answer'`` and non-empty prediction
- ``empty_final_response``: vendor returned empty/whitespace prediction
- ``rate_limit_exhausted``: transport-layer backoff retries exhausted
- ``retry_exhausted``: vendor's own call_server max_tries hit (final content is "vllm server error!!!")
- ``error``: uncaught exception

Backend-agnostic: the runner accepts any OpenAI-compatible endpoint via
``api_base`` + ``api_key``; vendor's hardcoded ``http://127.0.0.1:{port}/v1``
is discarded because the monkey-patched ``TracedStreamingOpenAI`` ignores
vendor-provided credentials and uses the runner-injected ones.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from typing import Any

from agents.base import TraceAction
from agents.benchmarks._research import render_research_prompt
from agents.tongyi_deepresearch.trace import (
    RateLimitExhausted,
    TracedStreamingOpenAI,
    _approx_token_count,
    make_traced_tool_class,
)
from harness.trace_logger import TraceLogger
from trace_collect.attempt_pipeline import AttemptContext, AttemptResult

logger = logging.getLogger(__name__)

_VENDOR_RETRY_EXHAUSTED_SENTINEL = "vllm server error!!!"
_VENDOR_NO_ANSWER_SENTINEL = "No answer found."

# Serializes concurrent run_task invocations because vendor module attributes
# (OpenAI, TOOL_CLASS, TOOL_MAP, count_tokens) are patched globally within
# the contextmanager. Concurrent patching would race.
_VENDOR_PATCH_LOCK = threading.Lock()


# Map our repo's tool env-var names onto vendor's expected names. The upstream
# reads SERPER_KEY_ID (Serper's example doc name) and JINA_API_KEYS (plural),
# while our conventions are SERPER_API_KEY and JINA_API_KEY. Bridge both so
# vendor works without requiring the operator to set duplicate exports.
_ENV_ALIAS_MAP = {
    "SERPER_KEY_ID": ("SERPER_API_KEY",),
    "JINA_API_KEYS": ("JINA_API_KEY",),
}
_VISIT_SUMMARIZER_ENV_KEYS = ("API_KEY", "API_BASE", "SUMMARY_MODEL_NAME")
_VENDOR_FILE_ROOT_ENV_KEY = "TONGYI_FILE_ROOT_PATH"
_VENDOR_TOOL_MODULE_GLOBALS = {
    "SERPER_KEY_ID": ("agents.tongyi_deepresearch.vendor.tool_search", "SERPER_KEY"),
    "JINA_API_KEYS": ("agents.tongyi_deepresearch.vendor.tool_visit", "JINA_API_KEYS"),
}


@contextlib.contextmanager
def _override_vendor_env_aliases():
    """Refresh vendor tool credentials per task run and restore afterward."""
    modules: dict[str, Any] = {}
    previous_env = {key: os.environ.get(key) for key in _ENV_ALIAS_MAP}
    previous_globals: dict[str, Any] = {}

    for vendor_name, (module_path, attr_name) in _VENDOR_TOOL_MODULE_GLOBALS.items():
        module = __import__(module_path, fromlist=[attr_name])
        modules[vendor_name] = module
        previous_globals[vendor_name] = getattr(module, attr_name)

    try:
        for vendor_name, aliases in _ENV_ALIAS_MAP.items():
            value = None
            for alias in aliases:
                alias_value = os.environ.get(alias)
                if alias_value:
                    value = alias_value
                    break
            if value is None:
                if previous_env[vendor_name] is None:
                    os.environ.pop(vendor_name, None)
                else:
                    os.environ[vendor_name] = previous_env[vendor_name]
            else:
                os.environ[vendor_name] = value

            module = modules[vendor_name]
            _, attr_name = _VENDOR_TOOL_MODULE_GLOBALS[vendor_name]
            setattr(module, attr_name, os.environ.get(vendor_name, ""))
        yield
    finally:
        for vendor_name, value in previous_env.items():
            if value is None:
                os.environ.pop(vendor_name, None)
            else:
                os.environ[vendor_name] = value
        for vendor_name, previous in previous_globals.items():
            module = modules[vendor_name]
            _, attr_name = _VENDOR_TOOL_MODULE_GLOBALS[vendor_name]
            setattr(module, attr_name, previous)


@contextlib.contextmanager
def _override_vendor_file_root_path(file_root_path: str):
    previous = os.environ.get(_VENDOR_FILE_ROOT_ENV_KEY)
    os.environ[_VENDOR_FILE_ROOT_ENV_KEY] = file_root_path
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_VENDOR_FILE_ROOT_ENV_KEY, None)
        else:
            os.environ[_VENDOR_FILE_ROOT_ENV_KEY] = previous


@contextlib.contextmanager
def _override_visit_summarizer_env(api_key: str, api_base: str, model: str):
    """Override Visit tool summarization env for the duration of one task run.

    Vendor's ``Visit.call_server()`` distills fetched pages into evidence via
    a separate OpenAI-compatible client built from ``API_KEY`` / ``API_BASE``
    / ``SUMMARY_MODEL_NAME`` env vars. The runner already has these values;
    expose them to the vendor tool per-run, then restore the previous process
    environment so later tasks can use different backends safely.
    """
    previous = {key: os.environ.get(key) for key in _VISIT_SUMMARIZER_ENV_KEYS}
    os.environ["API_KEY"] = api_key
    os.environ["API_BASE"] = api_base
    os.environ["SUMMARY_MODEL_NAME"] = model
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _advance_iteration(iteration_state: dict[str, int]) -> int:
    iteration_state["current"] += 1
    return iteration_state["current"]


def _approx_tokens_of_messages(messages: list[dict[str, Any]]) -> int:
    """Approximate total token count across message contents via tiktoken."""
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += _approx_token_count(c)
    return total


@contextlib.contextmanager
def _patched_vendor(
    *,
    api_key: str,
    api_base: str,
    summary_model: str,
    file_root_path: str,
    emit_fn,
    agent_id: str,
    instance_id: str,
    iteration_provider,
    llm_iteration_start_fn,
    max_llm_calls: int,
):
    """Patch vendor module globals for the duration of one run_task.

    Replaces:
    - ``vendor.OpenAI`` with a bound factory returning ``TracedStreamingOpenAI``
      that ignores vendor-passed credentials and uses runner-injected ones.
    - Each instance in ``vendor.TOOL_CLASS`` with a traced subclass that emits
      tool_exec TraceActions.
    - ``vendor.TOOL_MAP`` rebuilt against the traced class set.
    - ``MultiTurnReactAgent.count_tokens`` with a tiktoken approximation so
      vendor doesn't require a real HF tokenizer download at runtime.

    Restores all patched attributes on exit (even on exception) so sequential
    runs stay isolated. Thread-safety is enforced by ``_VENDOR_PATCH_LOCK``.
    """
    from agents.tongyi_deepresearch.vendor import react_agent as vendor

    # Action counter is list-wrapped so make_traced_tool_class closures share
    # a single mutable counter across multiple tool wrappers.
    action_counter = [0]

    # Shared state for the TracedStreamingOpenAI factory: vendor constructs a
    # fresh OpenAI client per call_server invocation, so per-instance counters
    # would reset between rounds and every llm_call would get action_id=llm_1.
    # These mutable containers are captured by the factory closure and persist
    # across all instances for the duration of one run_task call.
    llm_call_counter = [0]
    llm_retry_state: dict[str, Any] = {"last_action_id": None, "last_was_empty": False}

    # Bound OpenAI factory: discards vendor-passed credentials, uses runner's.
    # Injects shared call_counter and retry_state so action_ids stay monotonic
    # and retry_of linkage survives across vendor's per-round client rebuilds.
    class _BoundTracedOpenAI(TracedStreamingOpenAI):
        def __init__(self, *_a, **_k):
            super().__init__(
                api_key=api_key,
                base_url=api_base,
                emit_fn=emit_fn,
                agent_id=agent_id,
                instance_id=instance_id,
                iteration_provider=iteration_provider,
                llm_iteration_start_fn=llm_iteration_start_fn,
                call_counter=llm_call_counter,
                retry_state=llm_retry_state,
            )

    with _VENDOR_PATCH_LOCK:
        orig_openai = vendor.OpenAI
        orig_tool_class = list(vendor.TOOL_CLASS)
        orig_tool_map = dict(vendor.TOOL_MAP)
        orig_count_tokens = vendor.MultiTurnReactAgent.count_tokens
        orig_max_calls = vendor.MAX_LLM_CALL_PER_RUN

        traced_instances = []
        traced_map: dict[str, Any] = {}
        for inst in orig_tool_class:
            TracedCls = make_traced_tool_class(
                type(inst),
                emit_fn=emit_fn,
                agent_id=agent_id,
                instance_id=instance_id,
                iteration_provider=iteration_provider,
                action_counter=action_counter,
            )
            traced_inst = TracedCls()
            traced_instances.append(traced_inst)
            traced_map[traced_inst.name] = traced_inst

        with _override_vendor_env_aliases():
            with _override_vendor_file_root_path(file_root_path):
                with _override_visit_summarizer_env(api_key, api_base, summary_model):
                    try:
                        vendor.OpenAI = _BoundTracedOpenAI
                        vendor.TOOL_CLASS.clear()
                        vendor.TOOL_CLASS.extend(traced_instances)
                        vendor.TOOL_MAP.clear()
                        vendor.TOOL_MAP.update(traced_map)
                        vendor.MultiTurnReactAgent.count_tokens = (
                            lambda self, messages: _approx_tokens_of_messages(messages)
                        )
                        # MAX_LLM_CALL_PER_RUN is frozen at module-load from os.environ;
                        # patch the module attribute directly so runner's max_iterations takes effect.
                        vendor.MAX_LLM_CALL_PER_RUN = max_llm_calls
                        yield vendor
                    finally:
                        vendor.OpenAI = orig_openai
                        vendor.TOOL_CLASS.clear()
                        vendor.TOOL_CLASS.extend(orig_tool_class)
                        vendor.TOOL_MAP.clear()
                        vendor.TOOL_MAP.update(orig_tool_map)
                        vendor.MultiTurnReactAgent.count_tokens = orig_count_tokens
                        vendor.MAX_LLM_CALL_PER_RUN = orig_max_calls


class TongyiDeepResearchRunner:
    """Host-mode Runner wrapping vendored Tongyi-DeepResearch ReAct scaffold."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        max_iterations: int,
        benchmark_slug: str,
        client: Any | None = None,  # accepted for parity; unused (shim builds its own)
        **_: Any,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.benchmark_slug = benchmark_slug
        # client kwarg is accepted for Runner-protocol parity with other scaffolds
        # but unused: the TracedStreamingOpenAI shim constructs its own openai client
        # from api_key/api_base at monkey-patch time.

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        trace_logger = TraceLogger(attempt_ctx.attempt_dir, "trace")
        agent_id = attempt_ctx.instance_id
        instance_id = attempt_ctx.instance_id

        trace_logger.log_metadata(
            scaffold="tongyi-deepresearch",
            execution_environment="host",
            benchmark=self.benchmark_slug,
            model=self.model,
            api_base=self.api_base,
            max_iterations=self.max_iterations,
            instance_id=instance_id,
            prompt_template=prompt_template,
            agent_runtime_mode=attempt_ctx.agent_runtime_mode,
            scaffold_capabilities={
                "tools": ["search", "visit"],
                "memory": False,
                "skills": False,
                "file_ops": "none",
            },
        )

        actions: list[TraceAction] = []
        iteration_state = {"current": -1}

        def emit(action: TraceAction) -> None:
            actions.append(action)
            trace_logger.log_trace_action(agent_id, action)

        def iteration_provider() -> int:
            return max(iteration_state["current"], 0)

        task_prompt = render_research_prompt(
            self.benchmark_slug,
            task,
            prompt_template=prompt_template,
        )

        exit_status = "completed"
        error_msg: str | None = None
        final_answer = ""
        vendor_termination: str | None = None

        try:
            # Vendor is synchronous; wrap entire monkey-patch + invocation in to_thread
            def _sync_run() -> dict[str, Any]:
                from agents.tongyi_deepresearch.vendor import react_agent as vendor

                with _patched_vendor(
                    api_key=self.api_key,
                    api_base=self.api_base,
                    summary_model=self.model,
                    file_root_path=str(attempt_ctx.attempt_dir),
                    emit_fn=emit,
                    agent_id=agent_id,
                    instance_id=instance_id,
                    iteration_provider=iteration_provider,
                    llm_iteration_start_fn=lambda: _advance_iteration(iteration_state),
                    max_llm_calls=self.max_iterations,
                ):
                    agent = vendor.MultiTurnReactAgent(
                        llm={
                            "model": self.model,
                            "generate_cfg": {
                                "temperature": 0.6,
                                "top_p": 0.95,
                                "presence_penalty": 1.1,
                            },
                        },
                    )
                    data = {
                        "item": {"question": task_prompt, "answer": ""},
                        "planning_port": 0,  # vendor reads this for base_url, shim ignores
                    }
                    return agent._run(data, self.model)

            result = await asyncio.to_thread(_sync_run)
            vendor_termination = result.get("termination")
            final_answer = (result.get("prediction") or "").strip()

            if final_answer == _VENDOR_RETRY_EXHAUSTED_SENTINEL:
                exit_status = "retry_exhausted"
                error_msg = "vendor call_server exhausted its own max_tries"
                final_answer = ""
            elif (
                vendor_termination != "answer"
                or not final_answer
                or final_answer == _VENDOR_NO_ANSWER_SENTINEL
            ):
                # Vendor produces sentinel "No answer found." when it never hits
                # <answer>…</answer>; treat that as empty_final_response regardless
                # of whether the sentinel string is technically non-empty.
                exit_status = "empty_final_response"
                error_msg = f"vendor did not reach answer (termination={vendor_termination!r})"
                final_answer = ""

        except RateLimitExhausted as exc:
            logger.warning("TongyiDeepResearchRunner rate-limit exhausted: %s", exc)
            exit_status = "rate_limit_exhausted"
            error_msg = str(exc)
            final_answer = ""
        except Exception as exc:  # noqa: BLE001
            logger.exception("TongyiDeepResearchRunner failed with uncaught error")
            exit_status = "error"
            error_msg = f"{type(exc).__name__}: {exc}"
            final_answer = ""

        success = exit_status == "completed"
        summary = self._build_summary(
            agent_id=agent_id,
            instance_id=instance_id,
            model=self.model,
            actions=actions,
            final_answer=final_answer,
            success=success,
            vendor_termination=vendor_termination,
        )
        trace_logger.log_summary(agent_id, summary)
        trace_logger.close()

        return AttemptResult(
            success=success,
            exit_status=exit_status,
            trace_path=trace_logger.path,
            model_patch="",
            summary=summary,
            error=error_msg,
            n_iterations=summary.get("n_iterations"),
            total_llm_ms=summary.get("total_llm_ms"),
            total_tool_ms=summary.get("total_tool_ms"),
            total_tokens=summary.get("total_tokens"),
            runtime_proof={
                "agent_runtime_mode": "host_controller",
                "benchmark": self.benchmark_slug,
                "vendor_termination": vendor_termination,
            },
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        *,
        agent_id: str,
        instance_id: str,
        model: str,
        actions: list[TraceAction],
        final_answer: str,
        success: bool,
        vendor_termination: str | None,
    ) -> dict[str, Any]:
        # n_turns = distinct logical_turn_id values among llm_call actions
        # (excludes transport_retry spans; they share a turn_id with their parent).
        turn_ids = {
            a.data.get("logical_turn_id")
            for a in actions
            if a.action_type == "llm_call"
            and a.data.get("logical_turn_id")
            and not a.data.get("transport_retry")
        }
        n_turns = len(turn_ids)

        total_llm_ms = sum(
            float(a.data.get("llm_wall_latency_ms") or 0.0)
            for a in actions
            if a.action_type == "llm_call" and not a.data.get("transport_retry")
        )
        total_tool_ms = sum(
            float(a.data.get("duration_ms") or 0.0)
            for a in actions
            if a.action_type == "tool_exec"
        )
        total_tokens = sum(
            int(a.data.get("prompt_tokens") or 0) + int(a.data.get("completion_tokens") or 0)
            for a in actions
            if a.action_type == "llm_call" and not a.data.get("transport_retry")
        )
        tool_ms_by_name: dict[str, float] = {}
        for a in actions:
            if a.action_type != "tool_exec":
                continue
            name = a.data.get("tool_name")
            if name:
                tool_ms_by_name[name] = tool_ms_by_name.get(name, 0.0) + float(
                    a.data.get("duration_ms") or 0.0
                )

        # Count transport retries (429/503) separately — useful for scheduling analysis
        transport_retry_count = sum(
            1 for a in actions if a.action_type == "llm_call" and a.data.get("transport_retry")
        )

        return {
            "agent_id": agent_id,
            "instance_id": instance_id,
            "model": model,
            "n_turns": n_turns,
            "n_iterations": n_turns,
            "total_llm_ms": total_llm_ms,
            "total_tool_ms": total_tool_ms,
            "total_tokens": total_tokens,
            "tool_ms_by_name": tool_ms_by_name,
            "transport_retry_count": transport_retry_count,
            "vendor_termination": vendor_termination,
            "final_answer": final_answer,
            "success": success,
        }
