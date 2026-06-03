"""Workspace preparation for SWE-bench instances.

Handles git clone, checkout at base_commit, and pip install -e .
to produce a ready-to-edit codebase in the task workspace.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path

from loguru import logger

async def prepare_workspace(
    workspace_dir: Path,
    repo: str,
    base_commit: str,
    *,
    repos_root: Path | None = None,
    clone_timeout: float = 300.0,
    checkout_timeout: float = 60.0,
    install_timeout: float = 600.0,
) -> float:
    """Clone a repo at base_commit into workspace_dir and install it.

    Args:
        workspace_dir: Where to place the cloned repo.
        repo: GitHub repo path, e.g. "django/django".
        base_commit: Git commit hash to check out.
        repos_root: Optional local mirror root. If set and the local
            repo exists there, clones from the mirror instead of GitHub.
        clone_timeout: Timeout for git clone (seconds).
        checkout_timeout: Timeout for git checkout (seconds).
        install_timeout: Timeout for pip install (seconds).

    Returns:
        Elapsed time in milliseconds.

    Raises:
        RuntimeError: If clone or checkout fails.
    """
    wall_start = time.monotonic()

    if workspace_dir.exists():
        logger.info("Cleaning existing workspace {ws}", ws=workspace_dir)
        shutil.rmtree(workspace_dir, ignore_errors=True)

    repo_dir = workspace_dir
    if repos_root and (repos_root / f"{repo.replace('/', '__')}.git").exists():
        local_path = repos_root / f"{repo.replace('/', '__')}.git"
        clone_cmd = f"git clone {local_path} {repo_dir}"
    else:
        repo_url = f"https://github.com/{repo}.git"
        clone_cmd = f"git clone --depth=1 {repo_url} {repo_dir}"

    logger.info("Cloning {repo} → {ws}", repo=repo, ws=workspace_dir)
    result = await _run_bash(clone_cmd, timeout=clone_timeout)
    if result.returncode != 0:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise RuntimeError(
            f"Clone failed for {repo}: {(result.stdout + result.stderr)[:300]}"
        )

    # If we used --depth=1, we need to fetch the specific commit.
    fetch_cmd = f"git -C {repo_dir} fetch --depth=1 origin {base_commit}"
    checkout_cmd = f"git -C {repo_dir} checkout {base_commit}"

    logger.info("Checking out {commit} in {repo}", commit=base_commit[:8], repo=repo)

    # Try fetch + checkout; if fetch fails, try full unshallow
    fetch_result = await _run_bash(fetch_cmd, timeout=checkout_timeout)
    if fetch_result.returncode != 0:
        # Fallback: unshallow and checkout
        logger.warning("Shallow fetch failed, unshallowing {repo}", repo=repo)
        unshallow_cmd = f"git -C {repo_dir} fetch --unshallow origin"
        await _run_bash(unshallow_cmd, timeout=checkout_timeout * 2)

    result = await _run_bash(checkout_cmd, timeout=checkout_timeout)
    if result.returncode != 0:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise RuntimeError(
            f"Checkout failed for {repo}@{base_commit[:8]}: "
            f"{(result.stdout + result.stderr)[:300]}"
        )

    install_cmd = (
        f"cd {repo_dir}"
        " && if [ -f setup.py ] || [ -f pyproject.toml ]; then"
        "   pip install -e . 2>&1 | tail -5;"
        " fi"
    )
    logger.info("Installing {repo} (best effort)", repo=repo)
    await _run_bash(install_cmd, timeout=install_timeout)

    elapsed_ms = (time.monotonic() - wall_start) * 1000
    logger.info(
        "Workspace prepared for {repo}@{commit} in {ms:.0f}ms",
        repo=repo,
        commit=base_commit[:8],
        ms=elapsed_ms,
    )
    return elapsed_ms

async def _run_bash(cmd: str, *, timeout: float) -> subprocess.CompletedProcess:
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
