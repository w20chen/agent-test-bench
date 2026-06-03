from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agents.base import AgentBase, LLMCallResult
from harness.runner import (
    BenchmarkRunner,
    RunnerTaskResult,
    build_agent_factory,
    build_arrival_offsets,
)


class SlowAgent(AgentBase):
    async def run(self, task: dict[str, Any]) -> bool:
        self.task_id = task["task_id"]
        await asyncio.sleep(task.get("sleep_s", 0.01))
        self.task_success = True
        self.trace = [
            self.build_step_record(
                step_idx=0,
                phase="reasoning",
                llm_result=LLMCallResult(
                    content="done",
                    prompt_tokens=1,
                    completion_tokens=1,
                    llm_latency_ms=1.0,
                    raw_response={"id": task["task_id"]},
                ),
                ts_start=1.0,
                ts_end=2.0,
            )
        ]
        return True


class FailingAgent(AgentBase):
    async def run(self, task: dict[str, Any]) -> bool:
        self.task_id = task["task_id"]
        raise RuntimeError("boom")


def make_agent(agent_id: str, api_base: str, model: str) -> SlowAgent:
    return SlowAgent(agent_id=agent_id, api_base=api_base, model=model)


def test_build_agent_factory_rejects_removed_legacy_code_agent() -> None:
    with pytest.raises(ValueError, match="legacy code agent was removed"):
        build_agent_factory("code")


def test_benchmark_runner_n2_processes_all_tasks() -> None:
    tasks = [{"task_id": f"task-{i}", "sleep_s": 0.01} for i in range(4)]
    runner = BenchmarkRunner(
        agent_factory=make_agent,
        api_base="http://localhost:8000/v1",
        model="mock",
        concurrency=2,
        tasks=tasks,
    )
    results = asyncio.run(runner.run())
    assert len(results) == 4
    assert all(isinstance(result, RunnerTaskResult) for result in results)
    assert {result.summary["task_id"] for result in results} == {
        task["task_id"] for task in tasks
    }


def test_benchmark_runner_timeout_marks_task_as_failed() -> None:
    tasks = [{"task_id": "slow", "sleep_s": 0.2}]
    runner = BenchmarkRunner(
        agent_factory=make_agent,
        api_base="http://localhost:8000/v1",
        model="mock",
        concurrency=1,
        tasks=tasks,
        task_timeout_s=0.01,
    )
    results = asyncio.run(runner.run())
    assert len(results) == 1
    assert results[0].summary["success"] is False
    assert results[0].summary["timed_out"] is True


def test_benchmark_runner_converts_task_exception_into_failed_result() -> None:
    runner = BenchmarkRunner(
        agent_factory=lambda agent_id, api_base, model: FailingAgent(
            agent_id=agent_id,
            api_base=api_base,
            model=model,
        ),
        api_base="http://localhost:8000/v1",
        model="mock",
        concurrency=1,
        tasks=[{"task_id": "boom"}],
    )
    results = asyncio.run(runner.run())
    assert len(results) == 1
    assert results[0].summary["success"] is False
    assert results[0].summary["exception_type"] == "RuntimeError"


def test_poisson_arrival_offsets_are_deterministic() -> None:
    offsets = build_arrival_offsets(
        3,
        arrival_mode="poisson",
        arrival_rate_per_s=2.0,
        arrival_seed=7,
    )
    assert offsets == build_arrival_offsets(
        3,
        arrival_mode="poisson",
        arrival_rate_per_s=2.0,
        arrival_seed=7,
    )
    assert offsets[1] >= offsets[0]
