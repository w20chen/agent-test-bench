from __future__ import annotations

import json
import random
from typing import Any


def count_fail_to_pass(task: dict[str, Any]) -> int:
    """Count the number of tests listed in ``FAIL_TO_PASS``."""
    raw = task.get("FAIL_TO_PASS", "[]")
    if isinstance(raw, str):
        try:
            return len(json.loads(raw))
        except json.JSONDecodeError:
            return 1 if raw.strip() else 0
    return len(raw)


def select_tool_intensive(
    tasks: list[dict[str, Any]],
    *,
    repo_quotas: dict[str, int],
    n: int = 32,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a deterministic tool-intensive SWE-bench subset."""
    rng = random.Random(seed)

    candidates = [
        task
        for task in tasks
        if count_fail_to_pass(task) > 0 and len(task.get("problem_statement", "")) > 100
    ]

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for task in candidates:
        repo = task["repo"]
        by_repo.setdefault(repo, []).append(task)

    for repo_tasks in by_repo.values():
        repo_tasks.sort(key=count_fail_to_pass, reverse=True)

    selected: list[dict[str, Any]] = []
    for repo, quota in repo_quotas.items():
        pool = by_repo.get(repo, [])
        selected.extend(pool[: min(quota, len(pool))])

    remaining = n - len(selected)
    if remaining > 0:
        other_repos = [repo for repo in by_repo if repo not in repo_quotas]
        rng.shuffle(other_repos)
        other_pool: list[dict[str, Any]] = []
        for repo in other_repos:
            other_pool.extend(by_repo[repo])
        other_pool.sort(key=count_fail_to_pass, reverse=True)
        selected.extend(other_pool[:remaining])

    selected = selected[:n]
    selected.sort(key=lambda task: task["instance_id"])
    return selected
