"""Runtime trace registry overlay for the dynamic Gantt viewer."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from demo.gantt_viewer.backend.discovery import DiscoveryState, REPO_ROOT
from demo.gantt_viewer.backend.ingest import ensure_canonical_trace_path
from demo.gantt_viewer.backend.schema import TraceDescriptor


logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_STATE_PATH = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "agent-sched-bench"
    / "gantt-runtime-state.json"
)


class RuntimeRegistryConflictError(ValueError):
    """Raised when a runtime registry operation would create a conflict."""


@dataclass(slots=True)
class RuntimeRegisteredTrace:
    id: str
    label: str
    path: str
    source_format: Literal["trace"]
    origin: Literal["path_register", "upload"]


class RuntimeTraceRegistry:
    """Merged trace registry for config + runtime additions/removals."""

    def __init__(
        self,
        discovery_state: DiscoveryState,
        *,
        state_path: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.discovery_state = discovery_state
        self.repo_root = (repo_root or REPO_ROOT).resolve()
        self.state_path = (state_path or DEFAULT_RUNTIME_STATE_PATH).expanduser().resolve()
        self.registered_by_id: dict[str, RuntimeRegisteredTrace] = {}
        self.suppressed_ids: set[str] = set()
        self.descriptors: list[TraceDescriptor] = []
        self.descriptors_by_id: dict[str, TraceDescriptor] = {}
        self.reload()

    def reload(self) -> None:
        """Reload config discovery and re-merge runtime state."""
        self.discovery_state.reload()
        self._load_state_file()
        self._rebuild_effective_descriptors()

    def register_paths(
        self,
        paths: list[str],
        *,
        labels_by_path: dict[str, str] | None = None,
    ) -> list[TraceDescriptor]:
        """Register one or more existing trace files for runtime tracking."""
        if not paths:
            return []

        labels_by_path = labels_by_path or {}
        operations: list[tuple[str, TraceDescriptor | RuntimeRegisteredTrace]] = []
        seen_input_paths: set[Path] = set()
        seen_canonical_paths: set[Path] = set()

        for raw_path in paths:
            input_path = self._resolve_input_path(raw_path)
            if input_path in seen_input_paths:
                raise RuntimeRegistryConflictError(
                    f"duplicate registration request for path: {input_path}"
                )
            seen_input_paths.add(input_path)

            canonicalized = ensure_canonical_trace_path(input_path)
            resolved_path = canonicalized.canonical_path
            if resolved_path in seen_canonical_paths:
                raise RuntimeRegistryConflictError(
                    f"duplicate registration request for canonical path: {resolved_path}"
                )
            seen_canonical_paths.add(resolved_path)

            config_descriptor = self._config_descriptor_for_path(resolved_path)
            if config_descriptor is not None:
                if config_descriptor.id in self.suppressed_ids:
                    operations.append(("unsuppress", config_descriptor))
                    continue
                raise RuntimeRegistryConflictError(
                    f"trace already tracked from config: {resolved_path}"
                )

            existing_runtime = self._runtime_descriptor_for_path(resolved_path)
            if existing_runtime is not None:
                raise RuntimeRegistryConflictError(
                    f"trace already registered at runtime: {resolved_path}"
                )

            source_format = canonicalized.source_format
            label = self._label_for_path(
                raw_path=raw_path,
                resolved_path=input_path,
                labels_by_path=labels_by_path,
            )
            trace_id = _build_runtime_trace_id(resolved_path, label)
            if trace_id in self.descriptors_by_id or trace_id in self.registered_by_id:
                raise RuntimeRegistryConflictError(
                    f"trace id already exists: {trace_id}"
                )
            operations.append(
                (
                    "register",
                    RuntimeRegisteredTrace(
                        id=trace_id,
                        label=label,
                        path=str(resolved_path),
                        source_format=source_format,
                        origin="path_register",
                    ),
                )
            )

        registered_ids: list[str] = []
        for action, item in operations:
            if action == "unsuppress":
                descriptor = item
                assert isinstance(descriptor, TraceDescriptor)
                self.suppressed_ids.discard(descriptor.id)
                registered_ids.append(descriptor.id)
            else:
                registered = item
                assert isinstance(registered, RuntimeRegisteredTrace)
                self.registered_by_id[registered.id] = registered
                self.suppressed_ids.discard(registered.id)
                registered_ids.append(registered.id)

        self._save_state_file()
        self._rebuild_effective_descriptors()
        return [self.descriptors_by_id[trace_id] for trace_id in registered_ids]

    def register_uploaded_descriptor(self, descriptor: TraceDescriptor) -> None:
        """Persist an uploaded descriptor into the runtime registry."""
        existing = self.registered_by_id.get(descriptor.id)
        if existing is not None and existing.path != descriptor.path:
            raise RuntimeRegistryConflictError(
                f"runtime descriptor id already mapped to another path: {descriptor.id}"
            )

        self.registered_by_id[descriptor.id] = RuntimeRegisteredTrace(
            id=descriptor.id,
            label=descriptor.label,
            path=descriptor.path,
            source_format=descriptor.source_format,
            origin="upload",
        )
        self.suppressed_ids.discard(descriptor.id)
        self._save_state_file()
        self._rebuild_effective_descriptors()

    def unregister_ids(self, ids: list[str]) -> tuple[list[str], list[str]]:
        """Untrack descriptors by id without deleting underlying files."""
        removed_ids: list[str] = []
        missing_ids: list[str] = []
        config_ids = {descriptor.id for descriptor in self.discovery_state.descriptors}

        for trace_id in ids:
            if trace_id in self.registered_by_id:
                del self.registered_by_id[trace_id]
                self.suppressed_ids.discard(trace_id)
                removed_ids.append(trace_id)
            elif trace_id in config_ids or trace_id in self.suppressed_ids:
                self.suppressed_ids.add(trace_id)
                removed_ids.append(trace_id)
            else:
                missing_ids.append(trace_id)

        self._save_state_file()
        self._rebuild_effective_descriptors()
        return removed_ids, missing_ids

    def get_descriptor(self, trace_id: str) -> TraceDescriptor | None:
        return self.descriptors_by_id.get(trace_id)

    def list_descriptors(self) -> list[TraceDescriptor]:
        return self.descriptors

    def _config_descriptor_for_path(self, resolved_path: Path) -> TraceDescriptor | None:
        resolved = str(resolved_path.resolve())
        for descriptor in self.discovery_state.descriptors:
            if descriptor.path == resolved:
                return descriptor
        return None

    def _label_for_path(
        self,
        *,
        raw_path: str,
        resolved_path: Path,
        labels_by_path: dict[str, str],
    ) -> str:
        return (
            labels_by_path.get(raw_path)
            or labels_by_path.get(str(resolved_path))
            or _default_label_for_path(resolved_path)
        )

    def _load_state_file(self) -> None:
        self.registered_by_id = {}
        self.suppressed_ids = set()
        if not self.state_path.exists():
            return

        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Ignoring invalid runtime state file %s: %s", self.state_path, exc)
            return

        registered_raw = raw.get("registered", [])
        suppressed_raw = raw.get("suppressed_ids", [])
        if not isinstance(registered_raw, list) or not isinstance(suppressed_raw, list):
            logger.warning("Ignoring malformed runtime state file %s", self.state_path)
            return

        for item in registered_raw:
            if not isinstance(item, dict):
                continue
            try:
                registered = RuntimeRegisteredTrace(**item)
            except TypeError:
                logger.warning("Skipping malformed runtime registration entry in %s", self.state_path)
                continue
            if registered.source_format != "trace":
                logger.warning(
                    "Skipping obsolete runtime registration entry in %s for %s",
                    self.state_path,
                    registered.path,
                )
                continue
            self.registered_by_id[registered.id] = registered
        self.suppressed_ids = {str(item) for item in suppressed_raw}

    def _rebuild_effective_descriptors(self) -> None:
        descriptors_by_id = {
            descriptor.id: descriptor
            for descriptor in self.discovery_state.descriptors
            if descriptor.id not in self.suppressed_ids
        }

        for registered in self.registered_by_id.values():
            path = Path(registered.path)
            if not path.exists():
                logger.warning("Skipping missing runtime trace path %s", path)
                continue
            if registered.id in descriptors_by_id and registered.path != descriptors_by_id[registered.id].path:
                logger.warning(
                    "Skipping runtime trace id conflict for %s at %s",
                    registered.id,
                    registered.path,
                )
                continue
            descriptors_by_id[registered.id] = TraceDescriptor(
                id=registered.id,
                label=registered.label,
                source_format=registered.source_format,
                path=str(path.resolve()),
                size_bytes=path.stat().st_size,
                mtime=path.stat().st_mtime,
            )

        self.descriptors = sorted(descriptors_by_id.values(), key=lambda descriptor: descriptor.id)
        self.descriptors_by_id = {descriptor.id: descriptor for descriptor in self.descriptors}

    def _resolve_input_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        resolved = candidate if candidate.is_absolute() else self.repo_root / candidate
        resolved = resolved.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"trace file not found: {resolved}")
        return resolved

    def _runtime_descriptor_for_path(self, resolved_path: Path) -> RuntimeRegisteredTrace | None:
        resolved = str(resolved_path.resolve())
        for registered in self.registered_by_id.values():
            if registered.path == resolved:
                return registered
        return None

    def _save_state_file(self) -> None:
        if not self.registered_by_id and not self.suppressed_ids:
            if self.state_path.exists():
                self.state_path.unlink()
            return

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "registered": [
                asdict(registered)
                for registered in sorted(self.registered_by_id.values(), key=lambda item: item.id)
            ],
            "suppressed_ids": sorted(self.suppressed_ids),
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _build_runtime_trace_id(path: Path, label: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", label.lower()).strip("-") or "trace"
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"runtime-{slug}-{digest}"


def _default_label_for_path(path: Path) -> str:
    if path.parent.name.startswith("attempt_") and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem
