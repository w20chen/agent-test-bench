"""Runtime trace ingestion helpers for the Gantt viewer."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from demo.gantt_viewer.backend.discovery import sniff_format
from trace_collect.claude_code_import import (
    import_claude_code_session,
    looks_like_claude_code_session,
)


def _default_import_root() -> Path:
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "agent-sched-bench"
        / "gantt-cc-import"
    )


@dataclass(frozen=True, slots=True)
class CanonicalizedTrace:
    canonical_path: Path
    source_format: Literal["trace"]


def ensure_canonical_trace_path(
    path: Path,
    *,
    import_root: Path | None = None,
) -> CanonicalizedTrace:
    """Return a canonical trace path, auto-importing raw Claude Code when needed."""

    resolved_path = Path(path).expanduser().resolve()
    try:
        source_format = sniff_format(resolved_path)
        return CanonicalizedTrace(
            canonical_path=resolved_path,
            source_format=source_format,
        )
    except ValueError:
        if not looks_like_claude_code_session(resolved_path):
            raise

    imported_path = import_claude_code_session(
        session_path=resolved_path,
        output_dir=import_root or _default_import_root(),
        include_sidechains=True,
        run_id=_build_import_run_id(resolved_path),
    ).resolve()
    source_format = sniff_format(imported_path)
    return CanonicalizedTrace(
        canonical_path=imported_path,
        source_format=source_format,
    )


def _build_import_run_id(source_path: Path) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", source_path.stem.lower()).strip("-") or "trace"
    digest = hashlib.sha256(str(source_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"
