from __future__ import annotations

import argparse
import asyncio
import json
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agents.base import AgentBase
from harness.trace_logger import TraceLogger

AgentFactory = Callable[[str, str, str], AgentBase]

def build_arrival_offsets(
    num_tasks: int,
    *,
    arrival_mode: str,
    arrival_rate_per_s: float | None = None,
    arrival_seed: int | None = None,
) -> list[float]:
    if arrival_mode == "closed_loop":
        return [0.0] * num_tasks
    if arrival_mode != "poisson":
        raise ValueError(f"Unsupported arrival_mode: {arrival_mode}")
    if arrival_rate_per_s is None or arrival_rate_per_s <= 0:
        raise ValueError("arrival_rate_per_s must be positive for poisson mode")

    rng = random.Random(arrival_seed)
    offsets: list[float] = []
    elapsed = 0.0
    for _ in range(num_tasks):
        offsets.append(elapsed)
        elapsed += rng.expovariate(arrival_rate_per_s)
    return offsets

@dataclass(slots=True)
class RunnerTaskResult:

    summary: dict[str, Any]
    trace: list[dict[str, Any]]

def build_agent_factory(
    agent_name: str, agent_kwargs: dict[str, Any] | None = None
) -> AgentFactory:
    raise ValueError(
        f"Unsupported agent name: {agent_name}. "
        "The legacy code agent was removed with its scaffold; "
        "use trace_collect.cli with benchmark plugins instead."
    )

class BenchmarkRunner:

    def __init__(
        self,
        agent_factory: AgentFactory,
        api_base: str,
        model: str,
        concurrency: int,
        tasks: list[dict[str, Any]],
        *,
        arrival_mode: str = "closed_loop",
        task_timeout_s: float | None = None,
        arrival_rate_per_s: float | None = None,
        arrival_seed: int | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self.agent_factory = agent_factory
        self.api_base = api_base
        self.model = model
        self.concurrency = concurrency
        self.tasks = tasks
        self.arrival_mode = arrival_mode
        self.task_timeout_s = task_timeout_s
        self.arrival_rate_per_s = arrival_rate_per_s
        self.arrival_seed = arrival_seed
        self.trace_logger = trace_logger
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    async def _run_single_task(
        self, task: dict[str, Any], idx: int
    ) -> RunnerTaskResult:
        agent = self.agent_factory(f"agent-{idx:04d}", self.api_base, self.model)
        if self.trace_logger is not None:
            agent._trace_logger = self.trace_logger
        try:
            run_coro = agent.run(task)
            if self.task_timeout_s is not None:
                success = await asyncio.wait_for(run_coro, timeout=self.task_timeout_s)
            else:
                success = await run_coro
            agent.task_success = success
            summary = agent.summary()
            summary["timed_out"] = False
        except (TimeoutError, asyncio.TimeoutError):
            agent.task_success = False
            summary = agent.summary()
            summary["timed_out"] = True
            summary["error"] = "timeout"
        except Exception as exc:  # pragma: no cover - defensive harness path
            agent.task_success = False
            summary = agent.summary()
            summary["timed_out"] = False
            summary["error"] = str(exc)
            summary["exception_type"] = type(exc).__name__
        if self.trace_logger is not None:
            self.trace_logger.log_summary(agent.agent_id, summary)
        return RunnerTaskResult(summary=summary, trace=agent.get_trace())

    async def run(self) -> list[RunnerTaskResult]:
        """Execute all queued tasks with concurrency control.

        In ``closed_loop`` mode a two-phase strategy is used:

        1. **Prepare phase** – all agents set up their environments
           (containers, workspaces) in parallel.  This is CPU/IO work
           that does not touch the GPU.
        2. **Execute phase** – all agents start their LLM loops
           simultaneously so that they compete for the GPU from t=0.

        In ``poisson`` mode the original staggered-arrival queue is used.
        """
        if self.arrival_mode == "closed_loop":
            return await self._run_closed_loop()
        return await self._run_poisson()

    async def _run_closed_loop(self) -> list[RunnerTaskResult]:
        agents: list[tuple[AgentBase, dict[str, Any], int]] = []
        for idx, task in enumerate(self.tasks):
            agent = self.agent_factory(f"agent-{idx:04d}", self.api_base, self.model)
            if self.trace_logger is not None:
                agent._trace_logger = self.trace_logger
            agents.append((agent, task, idx))

        await asyncio.gather(*[agent.prepare(task) for agent, task, _ in agents])

        async def _execute(
            agent: AgentBase, task: dict[str, Any], idx: int
        ) -> RunnerTaskResult:
            try:
                run_coro = agent.run(task)
                if self.task_timeout_s is not None:
                    success = await asyncio.wait_for(
                        run_coro, timeout=self.task_timeout_s
                    )
                else:
                    success = await run_coro
                agent.task_success = success
                summary = agent.summary()
                summary["timed_out"] = False
            except (TimeoutError, asyncio.TimeoutError):
                agent.task_success = False
                summary = agent.summary()
                summary["timed_out"] = True
                summary["error"] = "timeout"
            except Exception as exc:
                agent.task_success = False
                summary = agent.summary()
                summary["timed_out"] = False
                summary["error"] = str(exc)
                summary["exception_type"] = type(exc).__name__
            if self.trace_logger is not None:
                self.trace_logger.log_summary(agent.agent_id, summary)
            return RunnerTaskResult(summary=summary, trace=agent.get_trace())

        result_list = await asyncio.gather(
            *[_execute(agent, task, idx) for agent, task, idx in agents]
        )
        return list(result_list)

    async def _run_poisson(self) -> list[RunnerTaskResult]:
        offsets = build_arrival_offsets(
            len(self.tasks),
            arrival_mode=self.arrival_mode,
            arrival_rate_per_s=self.arrival_rate_per_s,
            arrival_seed=self.arrival_seed,
        )

        queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()

        results: list[RunnerTaskResult | None] = [None] * len(self.tasks)

        async def producer() -> None:
            replay_zero = time.monotonic()
            for (index, task), offset in zip(enumerate(self.tasks), offsets):
                if self._stop_requested:
                    break
                delay_s = offset - (time.monotonic() - replay_zero)
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                await queue.put((index, task))
            for _ in range(self.concurrency):
                await queue.put(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        break
                    idx, task = item
                    results[idx] = await self._run_single_task(task, idx)
                finally:
                    queue.task_done()

        producer_task = asyncio.create_task(producer())
        workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
        await producer_task
        await queue.join()
        self._stop_requested = True
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        return [result for result in results if result is not None]

def install_signal_handlers(runner: BenchmarkRunner) -> None:
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        runner.request_stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, _request_stop)
        except NotImplementedError:
            signal.signal(signum, lambda *_args: runner.request_stop())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent benchmark runner.")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument(
        "--arrival-mode", default="closed_loop", choices=["closed_loop", "poisson"]
    )
    parser.add_argument("--arrival-rate-per-s", type=float)
    parser.add_argument("--arrival-seed", type=int)
    parser.add_argument("--task-timeout-s", type=float)
    parser.add_argument("--output")
    return parser.parse_args()

async def _run_cli(args: argparse.Namespace) -> list[RunnerTaskResult]:
    tasks = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
    runner = BenchmarkRunner(
        agent_factory=build_agent_factory(args.agent),
        api_base=args.api_base,
        model=args.model,
        concurrency=args.concurrency,
        tasks=tasks,
        arrival_mode=args.arrival_mode,
        task_timeout_s=args.task_timeout_s,
        arrival_rate_per_s=args.arrival_rate_per_s,
        arrival_seed=args.arrival_seed,
    )
    install_signal_handlers(runner)
    return await runner.run()

def main() -> None:
    args = parse_args()
    started = time.time()
    results = asyncio.run(_run_cli(args))
    payload = {
        "started_at": started,
        "finished_at": time.time(),
        "results": [
            {"summary": result.summary, "trace": result.trace} for result in results
        ],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
