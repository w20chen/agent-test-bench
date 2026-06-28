"""Resolve resource-monitoring policy for collection and simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

MonitoringMode = Literal["auto", "on", "off"]
MONITORING_CHOICES = ("auto", "on", "off")


@dataclass(frozen=True, slots=True)
class MonitoringPolicy:
    """Requested and resolved monitoring settings for one run."""

    resource_requested: MonitoringMode
    pmu_requested: MonitoringMode
    ksys_requested: MonitoringMode
    resource_enabled: bool
    pmu_enabled: bool
    memory_bandwidth_enabled: bool
    ksys_enabled: bool
    concurrent: bool
    #: Per-tool profiler selection: ``"off"``, ``"vtune"``, or ``"ksys"``.
    tool_profiling: str = "off"
    #: Tool names to profile when *tool_profiling* is not ``"off"``.
    tool_profiling_tools: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable policy record."""
        return asdict(self)


def resolve_ksys_request(
    requested: MonitoringMode,
    *,
    legacy_ksys: bool,
) -> MonitoringMode:
    """Merge legacy ``--ksys`` with ``--ksys-monitoring``."""
    if legacy_ksys and requested == "off":
        raise ValueError(
            "--ksys (or legacy enable_ksys=True) conflicts with "
            f"--ksys-monitoring {requested}"
        )
    return "on" if legacy_ksys else requested


def resolve_collect_monitoring(
    *,
    resource: MonitoringMode,
    pmu: MonitoringMode,
    ksys: MonitoringMode,
    concurrency: int,
    execution_environment: str,
) -> MonitoringPolicy:
    """Resolve collection defaults and reject invalid combinations."""
    concurrent = concurrency > 1
    resource_enabled = resource == "on" or (resource == "auto" and not concurrent)
    if concurrent and execution_environment == "host" and resource == "on":
        raise ValueError(
            "--resource-monitoring on is unsupported for concurrent host "
            "collection because attempts cannot be isolated by PID"
        )
    if concurrent and pmu == "on":
        raise ValueError("--pmu-monitoring on is forbidden with --concurrency > 1")
    if pmu == "on" and not resource_enabled:
        raise ValueError(
            "--pmu-monitoring on requires built-in resource monitoring; "
            "use --resource-monitoring on"
        )
    pmu_enabled = resource_enabled and not concurrent and pmu != "off"
    return MonitoringPolicy(
        resource_requested=resource,
        pmu_requested=pmu,
        ksys_requested=ksys,
        resource_enabled=resource_enabled,
        pmu_enabled=pmu_enabled,
        memory_bandwidth_enabled=resource_enabled and not concurrent,
        ksys_enabled=ksys == "on",
        concurrent=concurrent,
    )


def resolve_simulate_monitoring(
    *,
    resource: MonitoringMode,
    pmu: MonitoringMode,
    ksys: MonitoringMode,
    concurrent: bool,
    has_host_session: bool,
    has_container_session: bool,
) -> MonitoringPolicy:
    """Resolve simulation defaults after source sessions are loaded."""
    if concurrent and pmu == "on":
        raise ValueError(
            "--pmu-monitoring on is forbidden for concurrent simulation"
        )
    if has_host_session and resource == "on":
        raise ValueError(
            "--resource-monitoring on is unsupported for host simulation "
            "because the simulated agent has no isolated process PID"
        )
    resource_enabled = resource == "on" or (
        resource == "auto" and has_container_session
    )
    if pmu == "on" and not resource_enabled:
        raise ValueError(
            "--pmu-monitoring on requires built-in resource monitoring; "
            "use --resource-monitoring on"
        )
    pmu_enabled = resource_enabled and not concurrent and pmu != "off"
    return MonitoringPolicy(
        resource_requested=resource,
        pmu_requested=pmu,
        ksys_requested=ksys,
        resource_enabled=resource_enabled,
        pmu_enabled=pmu_enabled,
        memory_bandwidth_enabled=resource_enabled and not concurrent,
        ksys_enabled=ksys == "on",
        concurrent=concurrent,
    )
