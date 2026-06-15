"""Micro-architecture PMU event sampler -- cross-platform (ARMv8 / x86 / generic).

Provides per-container and per-process sampling of core PMU events:
IPC, L1 data/instruction cache hit rates, and branch misprediction rate.

Architecture
------------

Core PMUs typically expose a limited number of programmable counters
(4-6 on x86, 6 on ARMv8).  To avoid PMU counter multiplexing (which
introduces error on bursty LLM workloads), events are split into two
groups that are sampled alternately:

  Group A (cache):  cycles, instructions, L1D access, L1D refill,
                    L1I access, L1I refill
  Group B (branch): cycles, instructions, branch pred, branch mispred,
                    bus_access  (memory traffic proxy)

Platforms
---------

============= ======= =====================================================
Platform      Source  L1 events
============= ======= =====================================================
``armv8-raw`` ARMv8   ``r03``-``r14`` raw event codes (all ARMv8-A cores)
``x86-intel`` Intel   ``L1-dcache-loads`` etc. (Sandy Bridge+, Linux 3.x+)
``generic``   perf    ``cache-references`` / ``cache-misses`` (LLC-level,
                      less precise but works on all CPUs with a PMU driver)
============= ======= =====================================================

Auto-detection probes ``/sys/bus/event_source/devices/`` and
``/proc/cpuinfo``, falling back to ``generic`` when neither ARMv8 nor
Intel PMU signatures are found.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVENT_SOURCE_ROOT = Path("/sys/bus/event_source/devices")
PROC_CPUINFO = Path("/proc/cpuinfo")
DEFAULT_PERF_EXECUTABLE = "perf"

# ---------------------------------------------------------------------------
# Platform event specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PlatformEventSpecs:
    """Perf event strings for a specific CPU platform.

    Each field maps a logical metric name to a ``perf stat -e`` event
    string -- either a generic named event (``"cycles"``), a hardware
    event (``"L1-dcache-loads"``), or a raw PMU code (``"r04"``).

    Fields that are ``None`` signal "metric not available on this
    platform"; the corresponding event group will omit that event.
    """
    name: str
    label: str  # human-readable, e.g. "ARMv8 raw"
    cycles: str
    instructions: str
    # L1 data cache
    l1d_access: str
    l1d_miss: str
    # L1 instruction cache (may be None on platforms that do not expose it)
    l1i_access: str | None = None
    l1i_miss: str | None = None
    # Branch prediction
    branch_inst: str | None = None
    branch_miss: str | None = None
    # Memory bus traffic proxy (may be None)
    bus_access: str | None = None

    @property
    def has_l1i(self) -> bool:
        return self.l1i_access is not None and self.l1i_miss is not None

    @property
    def has_branch(self) -> bool:
        return self.branch_inst is not None and self.branch_miss is not None


#: ARMv8-A architectural raw event codes.
#: These are mandated by the ARM ARM and work on all ARMv8-A compliant
#: cores (Cortex-A53/57/72/76, Neoverse N1, HiSilicon TSV110 in Kunpeng
#: 920, etc.) regardless of kernel driver version.
ARMV8_RAW_SPECS = PlatformEventSpecs(
    name="armv8-raw",
    label="ARMv8 raw",
    cycles="r11",           # CPU_CYCLES (0x11)
    instructions="r08",     # INST_RETIRED (0x08)
    l1d_access="r04",       # L1D_CACHE (0x04)
    l1d_miss="r03",         # L1D_CACHE_REFILL (0x03)
    l1i_access="r14",       # L1I_CACHE (0x14)
    l1i_miss="r01",         # L1I_CACHE_REFILL (0x01)
    branch_inst="r12",      # BR_PRED (0x12)
    branch_miss="r10",      # BR_MIS_PRED (0x10)
    bus_access="r19",       # BUS_ACCESS (0x19)
)

#: Intel x86 named PMU events (Sandy Bridge / Ivy Bridge / Haswell / ...).
#: ``L1-dcache-loads`` and ``L1-dcache-load-misses`` are architectural
#: events on Intel since SNB.
#: ``L1-icache-loads`` is NOT always available -- we include it but the
#: cache group will silently degrade to L1D-only if it is missing.
X86_INTEL_SPECS = PlatformEventSpecs(
    name="x86-intel",
    label="x86 Intel named",
    cycles="cycles",
    instructions="instructions",
    l1d_access="L1-dcache-loads",
    l1d_miss="L1-dcache-load-misses",
    l1i_access="L1-icache-loads",       # may be absent on older kernels/CPUs
    l1i_miss="L1-icache-load-misses",   # may be absent
    branch_inst="branch-instructions",
    branch_miss="branch-misses",
    bus_access=None,  # not directly available; use branch group IPC instead
)

#: AMD x86 raw PMU events for L1I (Family 15h / 17h / 19h Zen 1-4).
#: Uses ``cpu/event=.../`` raw syntax to bypass the kernel's generic
#: PERF_TYPE_HW_CACHE event mapping, which is unreliable for L1I on many
#: AMD systems (the generic ``L1-icache-loads`` / ``L1-icache-load-misses``
#: may return ``<not supported>`` even though the underlying PMU counters
#: exist).  Raw event codes are stable across all AMD Zen generations
#: (verified against ``amd_hw_cache_event_ids`` and
#: ``amd_hw_cache_event_ids_f17h`` in Linux ``arch/x86/events/amd/core.c``).
#:
#: - L1I access: ``0x0080`` → ``cpu/event=0x80/`` (Instruction Cache Fetches)
#: - L1I miss:   ``0x0081`` → ``cpu/event=0x81/`` (Instruction Cache Misses)
#:
#: L1D uses the standard named events (``L1-dcache-loads`` /
#: ``L1-dcache-load-misses``) which work correctly on AMD; raw events are
#: only used for L1I where the kernel's generic mapping is unreliable.
#: Raw events containing commas (e.g. ``cpu/event=0x41,umask=0x01/``) would
#: break the ``perf stat -x,`` CSV parser, so they are avoided.
X86_AMD_SPECS = PlatformEventSpecs(
    name="x86-amd",
    label="x86 AMD L1I-raw",
    cycles="cycles",
    instructions="instructions",
    l1d_access="L1-dcache-loads",
    l1d_miss="L1-dcache-load-misses",
    l1i_access="cpu/event=0x80/",
    l1i_miss="cpu/event=0x81/",
    branch_inst="branch-instructions",
    branch_miss="branch-misses",
    bus_access=None,
)

#: Architecture-agnostic fallback using only perf generic named events.
#: ``cache-references`` / ``cache-misses`` count at the last-level cache,
#: NOT L1 -- so L1 hit rate derived from these is an approximation.
#: Works on any CPU with a Linux PMU driver (x86, ARM, POWER, RISC-V).
GENERIC_SPECS = PlatformEventSpecs(
    name="generic",
    label="perf generic",
    cycles="cycles",
    instructions="instructions",
    l1d_access="cache-references",   # LLC references (NOT L1D -- approximation)
    l1d_miss="cache-misses",         # LLC misses
    l1i_access=None,
    l1i_miss=None,
    branch_inst="branch-instructions",
    branch_miss="branch-misses",
    bus_access=None,
)

#: All platform specs in priority order (first detected wins).
_ALL_SPECS = (ARMV8_RAW_SPECS, X86_INTEL_SPECS, X86_AMD_SPECS, GENERIC_SPECS)


# ---------------------------------------------------------------------------
# Event group construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CoreEventGroup:
    """A set of PMU events that fit within the hardware counter limit."""
    name: str
    event_specs: tuple[str, ...]  # perf event strings, e.g. "r04", "cycles"
    metrics: tuple[str, ...]      # logical metric names, e.g. "l1d_access"


def _build_groups(specs: PlatformEventSpecs) -> tuple[CoreEventGroup, ...]:
    """Build event groups from platform specs.

    On x86 platforms (4 generic counters), L1D and L1I are split into
    separate groups so each ``perf stat`` call uses ≤ 4 events, avoiding
    PMU counter multiplexing that would consistently ``<not count>`` the
    last events in the list (typically L1I).  On ARMv8 (6 counters) the
    groups stay merged.

    Events whose spec string is ``None`` are omitted.
    """
    # x86 has fewer generic counters → split L1D/L1I to avoid multiplexing
    split_l1 = specs.name.startswith("x86")
    groups: list[CoreEventGroup] = []

    # -- Cache (L1D) group --------------------------------------------------
    cache_specs: list[str] = [specs.cycles, specs.instructions]
    cache_metrics: list[str] = ["cycles", "instructions"]
    cache_specs.append(specs.l1d_access)
    cache_metrics.append("l1d_access")
    cache_specs.append(specs.l1d_miss)
    cache_metrics.append("l1d_miss")

    if specs.has_l1i and not split_l1:
        # ARMv8: enough counters to keep L1I in the cache group
        cache_specs.append(specs.l1i_access)  # type: ignore[arg-type]
        cache_metrics.append("l1i_access")
        cache_specs.append(specs.l1i_miss)    # type: ignore[arg-type]
        cache_metrics.append("l1i_miss")

    groups.append(CoreEventGroup(
        name="cache",
        event_specs=tuple(cache_specs),
        metrics=tuple(cache_metrics),
    ))

    # -- ICache (L1I) group (x86 only, split from cache) --------------------
    if specs.has_l1i and split_l1:
        icache_specs: list[str] = [specs.cycles, specs.instructions]
        icache_metrics: list[str] = ["cycles", "instructions"]
        icache_specs.append(specs.l1i_access)  # type: ignore[arg-type]
        icache_metrics.append("l1i_access")
        icache_specs.append(specs.l1i_miss)    # type: ignore[arg-type]
        icache_metrics.append("l1i_miss")

        groups.append(CoreEventGroup(
            name="icache",
            event_specs=tuple(icache_specs),
            metrics=tuple(icache_metrics),
        ))

    # -- Branch group -------------------------------------------------------
    branch_specs: list[str] = [specs.cycles, specs.instructions]
    branch_metrics: list[str] = ["cycles", "instructions"]

    if specs.has_branch:
        branch_specs.append(specs.branch_inst)   # type: ignore[arg-type]
        branch_metrics.append("branch_inst")
        branch_specs.append(specs.branch_miss)   # type: ignore[arg-type]
        branch_metrics.append("branch_miss")
    if specs.bus_access is not None:
        branch_specs.append(specs.bus_access)
        branch_metrics.append("bus_access")

    groups.append(CoreEventGroup(
        name="branch",
        event_specs=tuple(branch_specs),
        metrics=tuple(branch_metrics),
    ))

    return tuple(groups)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MicroArchReading:
    """One sample from a single event group."""
    available: bool
    group: str | None = None
    scope: str | None = None  # "cgroup", "process", or "system_wide"
    platform: str | None = None  # e.g. "armv8-raw", "x86-intel"
    # Cache metrics (derived)
    l1d_hit_rate: float | None = None
    l1i_hit_rate: float | None = None
    # Branch metrics (derived)
    branch_miss_rate: float | None = None
    # Common across both groups
    ipc: float | None = None           # instructions per cycle
    instructions_per_s: float | None = None
    bus_access_per_s: float | None = None  # memory traffic proxy
    # Raw counts (for downstream aggregation)
    raw_counts: dict[str, float] = field(default_factory=dict)
    # Diagnostic
    source: str = "perf:core-pmu"
    reason: str | None = None


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def _detect_armv8_core_pmu(root: Path = EVENT_SOURCE_ROOT) -> str | None:
    """Return the first ARMv8 core PMU device name, or None."""
    if not root.exists():
        return None
    for child in sorted(root.iterdir()):
        name = child.name.lower()
        if name.startswith("armv8_pmu") or name.startswith("armv8_cortex"):
            return child.name
    return None


def _detect_x86_pmu(root: Path = EVENT_SOURCE_ROOT) -> bool:
    """Return True if an Intel or AMD core PMU is visible in sysfs."""
    if not root.exists():
        return False
    for child in root.iterdir():
        name = child.name.lower()
        if name.startswith(("cpu", "intel", "amd")) and "uncore" not in name:
            return True
    return False


def _read_cpuinfo_vendor() -> str | None:
    """Return the CPU vendor string from /proc/cpuinfo, or None."""
    try:
        text = PROC_CPUINFO.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            if key.strip().lower() in ("vendor_id", "cpu implementer"):
                return value.strip().lower()
    return None


def detect_platform_event_specs(
    event_source_root: Path = EVENT_SOURCE_ROOT,
) -> PlatformEventSpecs:
    """Auto-detect the best ``PlatformEventSpecs`` for the current host.

    Priority: ARMv8 raw -> x86 Intel -> generic (always succeeds).
    """
    # 1. ARMv8 core PMU?
    if _detect_armv8_core_pmu(event_source_root) is not None:
        logger.debug("micro-arch: detected ARMv8 core PMU -> armv8-raw")
        return ARMV8_RAW_SPECS

    # 2. x86 PMU?
    if _detect_x86_pmu(event_source_root):
        vendor = _read_cpuinfo_vendor()
        if vendor == "authenticamd":
            logger.debug("micro-arch: detected x86 PMU + AMD vendor -> x86-amd")
            return X86_AMD_SPECS
        logger.debug("micro-arch: detected x86 PMU -> x86-intel")
        return X86_INTEL_SPECS

    # 3. Check /proc/cpuinfo as fallback
    vendor = _read_cpuinfo_vendor()
    if vendor is not None:
        if vendor == "authenticamd":
            logger.debug("micro-arch: cpuinfo vendor=%s -> x86-amd", vendor)
            return X86_AMD_SPECS
        if vendor == "genuineintel":
            logger.debug("micro-arch: cpuinfo vendor=%s -> x86-intel", vendor)
            return X86_INTEL_SPECS
        if vendor == "0x48":  # HiSilicon (Kunpeng) implementer ID
            logger.debug("micro-arch: cpuinfo implementer=0x48 -> armv8-raw")
            return ARMV8_RAW_SPECS
        # ARM implementer IDs: 0x41 (ARM Ltd), 0x42 (Broadcom),
        # 0x43 (Cavium), 0x48 (HiSilicon), 0x4e (NVIDIA),
        # 0x50 (Applied Micro), 0x51 (Qualcomm)
        if vendor.startswith("0x"):
            logger.debug("micro-arch: cpuinfo implementer=%s -> armv8-raw", vendor)
            return ARMV8_RAW_SPECS

    # 4. Final fallback
    logger.debug("micro-arch: no platform detected -> generic")
    return GENERIC_SPECS


def _has_perf_cgroup_support(perf_executable: str) -> bool:
    """Check whether the installed perf tool supports ``--cgroup``."""
    try:
        result = subprocess.run(
            [perf_executable, "stat", "--help"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return "--cgroup" in (result.stdout or "") + (result.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def resolve_perf_scoping(
    *,
    perf_executable: str,
    cgroup_path: Path | None = None,
    container_pid: int | None = None,
) -> tuple[str | None, list[str]]:
    """Return the best available scoping mode and matching perf args."""
    if cgroup_path is not None and _has_perf_cgroup_support(perf_executable):
        return ("cgroup", ["--cgroup", str(cgroup_path)])
    if container_pid is not None:
        return ("process", ["-p", str(container_pid)])
    return ("system_wide", ["-a"])


# ---------------------------------------------------------------------------
# Perf output parsing
# ---------------------------------------------------------------------------

def _parse_perf_count(raw: str) -> float | None:
    value = raw.strip()
    if not value or value in {"<not counted>", "<not supported>"}:
        return None
    value = value.replace(" ", "")
    try:
        return float(value)
    except ValueError:
        return None


def _parse_perf_stat_output(
    text: str,
    event_specs: tuple[str, ...],
) -> dict[str, float] | None:
    """Parse CSV-formatted perf stat output, matching on event spec strings.

    Returns ``None`` when *no* events could be parsed at all.  Individual
    events that report ``<not counted>`` or ``<not supported>`` are
    silently skipped so the remaining metrics (e.g. L1D hit rate when
    L1I is unavailable on the host CPU) are still usable.

    Event matching uses the third CSV field (the canonical event name
    emitted by ``perf stat -x,``) rather than a substring scan of the
    whole line, avoiding false matches on description fields such as
    ``insn per cycle`` or ``of all branches``.
    """
    counts: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        event_name = parts[2].strip()
        if event_name not in event_specs:
            continue
        value = _parse_perf_count(parts[0])
        if value is None:
            # Event not available on this CPU — skip gracefully
            # so the rest of the group still produces metrics.
            continue
        counts[event_name] = value
    if not counts:
        return None
    return counts


def _classify_perf_failure(stderr: str) -> str:
    message = stderr.lower()
    if "permission" in message or "access to performance monitoring" in message:
        return "permission_denied"
    if "not supported" in message:
        return "pmu_unsupported"
    if "not found" in message:
        return "perf_missing"
    if "cgroup" in message and "not" in message:
        return "cgroup_unsupported"
    return "perf_error"


# ---------------------------------------------------------------------------
# Metric derivation
# ---------------------------------------------------------------------------

def _derive_metrics(
    group: CoreEventGroup,
    counts: dict[str, float],
    interval_s: float,
    platform_name: str,
) -> MicroArchReading:
    """Compute derived metrics from raw perf counts for a single group."""
    raw = {
        metric: counts[spec]
        for spec, metric in zip(group.event_specs, group.metrics)
        if spec in counts  # skip events unavailable on this CPU
    }

    divisor = max(interval_s, 1e-9)

    cycles = raw.get("cycles", 0.0)
    instructions = raw.get("instructions", 0.0)
    ipc = instructions / cycles if cycles > 0 else None
    instr_per_s = instructions / divisor

    l1d_hit_rate: float | None = None
    l1i_hit_rate: float | None = None
    branch_miss_rate: float | None = None
    bus_access_per_s: float | None = None

    if group.name == "cache":
        l1d_access = raw.get("l1d_access", 0.0)
        l1d_miss = raw.get("l1d_miss", 0.0)
        if l1d_access > 0:
            l1d_hit_rate = 1.0 - (l1d_miss / l1d_access)

        l1i_access = raw.get("l1i_access")
        l1i_miss = raw.get("l1i_miss")
        if l1i_access is not None and l1i_miss is not None and l1i_access > 0:
            l1i_hit_rate = 1.0 - (l1i_miss / l1i_access)
        elif l1i_access is None and l1i_miss is not None and instructions > 0:
            # Fallback: when the L1I access PMU event is unsupported on
            # this CPU, estimate L1I accesses from the instruction count.
            # Each retired instruction implies at least one L1I lookup,
            # so instructions is a conservative proxy for L1I accesses.
            l1i_hit_rate = 1.0 - (l1i_miss / instructions)

    elif group.name == "icache":
        # x86-only: L1I events are in a dedicated group (split from cache)
        l1i_access = raw.get("l1i_access")
        l1i_miss = raw.get("l1i_miss")
        if l1i_access is not None and l1i_miss is not None and l1i_access > 0:
            l1i_hit_rate = 1.0 - (l1i_miss / l1i_access)
        elif l1i_access is None and l1i_miss is not None and instructions > 0:
            # Fallback: use instructions as proxy for L1I accesses when
            # the L1I access PMU event is unsupported on this CPU.
            l1i_hit_rate = 1.0 - (l1i_miss / instructions)

    elif group.name == "branch":
        branch_inst = raw.get("branch_inst", 0.0)
        branch_miss = raw.get("branch_miss", 0.0)
        if branch_inst > 0:
            branch_miss_rate = branch_miss / branch_inst

        bus = raw.get("bus_access", 0.0)
        if bus > 0:
            bus_access_per_s = bus / divisor

    return MicroArchReading(
        available=True,
        group=group.name,
        platform=platform_name,
        l1d_hit_rate=l1d_hit_rate,
        l1i_hit_rate=l1i_hit_rate,
        branch_miss_rate=branch_miss_rate,
        ipc=ipc,
        instructions_per_s=instr_per_s,
        bus_access_per_s=bus_access_per_s,
        raw_counts=raw,
    )


# ---------------------------------------------------------------------------
# Single sample
# ---------------------------------------------------------------------------

def sample_core_events_once(
    group: CoreEventGroup,
    *,
    interval_s: float,
    perf_executable: str = DEFAULT_PERF_EXECUTABLE,
    scope_args: list[str] | None = None,
    scope_kind: str | None = None,
    platform_name: str = "generic",
    cgroup_path: Path | None = None,
    container_pid: int | None = None,
) -> MicroArchReading:
    """Run ``perf stat`` for *interval_s* seconds and return derived metrics."""
    if scope_args is None:
        scope_kind, scope_args = resolve_perf_scoping(
            perf_executable=perf_executable,
            cgroup_path=cgroup_path,
            container_pid=container_pid,
        )

    cmd = [
        perf_executable,
        "stat",
        "-x,",
        "--no-big-num",
        *scope_args,
        "-e", ",".join(group.event_specs),
        "--",
        "sleep", f"{interval_s:.6f}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(5.0, interval_s + 5.0),
            check=False,
            env={"LC_ALL": "C"},
        )
    except FileNotFoundError:
        return MicroArchReading(
            available=False, reason="perf_missing", platform=platform_name,
        )
    except subprocess.TimeoutExpired:
        return MicroArchReading(
            available=False, reason="perf_timeout", platform=platform_name,
        )

    perf_output = "\n".join(
        part for part in (result.stdout, result.stderr) if part
    )
    if result.returncode != 0:
        return MicroArchReading(
            available=False,
            reason=_classify_perf_failure(perf_output),
            platform=platform_name,
        )

    counts = _parse_perf_stat_output(perf_output, group.event_specs)
    if counts is None:
        return MicroArchReading(
            available=False, reason="parse_error", platform=platform_name,
        )

    reading = _derive_metrics(group, counts, interval_s, platform_name)
    # Attach scope info (reading is frozen, so we reconstruct)
    return MicroArchReading(
        available=reading.available,
        group=reading.group,
        scope=scope_kind,
        platform=reading.platform,
        l1d_hit_rate=reading.l1d_hit_rate,
        l1i_hit_rate=reading.l1i_hit_rate,
        branch_miss_rate=reading.branch_miss_rate,
        ipc=reading.ipc,
        instructions_per_s=reading.instructions_per_s,
        bus_access_per_s=reading.bus_access_per_s,
        raw_counts=reading.raw_counts,
        source=reading.source,
        reason=reading.reason,
    )


# ---------------------------------------------------------------------------
# Background collector
# ---------------------------------------------------------------------------

class MicroArchCollector(threading.Thread):
    """Background thread that alternates between event groups.

    Each iteration samples one group, sleeps, then samples the next group.
    This avoids PMU counter multiplexing while still providing all metrics
    at alternating time points.
    """

    def __init__(
        self,
        *,
        interval_s: float = 1.0,
        perf_executable: str = DEFAULT_PERF_EXECUTABLE,
        event_source_root: Path = EVENT_SOURCE_ROOT,
        cgroup_path: Path | None = None,
        container_pid: int | None = None,
        _platform: str | None = None,  # test injection point (sys.platform)
        _specs: PlatformEventSpecs | None = None,  # test injection point
    ) -> None:
        super().__init__(daemon=True, name="micro-arch-collector")
        self.interval_s = interval_s
        self.perf_executable = perf_executable
        self.event_source_root = event_source_root
        self.cgroup_path = cgroup_path
        self.container_pid = container_pid
        self._platform_override = _platform
        self._specs_override = _specs
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        # Built lazily after platform detection in run()
        self._groups: tuple[CoreEventGroup, ...] = ()
        self._platform_specs: PlatformEventSpecs | None = None
        # Per-group latest readings (built in run())
        self._latest: dict[str, MicroArchReading] = {}
        self._scope_kind: str | None = None
        self._scope_args: list[str] | None = None

    # -- Public API --------------------------------------------------------

    def latest(
        self, group_name: str | None = None,
    ) -> MicroArchReading | dict[str, MicroArchReading]:
        """Return the latest reading(s)."""
        with self._lock:
            if not self._latest:
                return MicroArchReading(
                    available=False, reason="not_started",
                )
            if group_name is not None:
                return self._latest.get(
                    group_name,
                    MicroArchReading(available=False, reason="unknown_group"),
                )
            return dict(self._latest)

    def _set_latest(self, reading: MicroArchReading) -> None:
        group = reading.group
        if group is None:
            return
        with self._lock:
            self._latest[group] = reading

    # -- Thread body -------------------------------------------------------

    def run(self) -> None:
        plat = (
            self._platform_override
            if self._platform_override is not None
            else sys.platform
        )
        if plat != "linux":
            self._latest = {
                "cache": MicroArchReading(
                    available=False, reason="unsupported_platform", group="cache",
                ),
                "branch": MicroArchReading(
                    available=False, reason="unsupported_platform", group="branch",
                ),
            }
            return

        resolved_perf = shutil.which(self.perf_executable)
        if resolved_perf is None:
            self._latest = {
                "cache": MicroArchReading(
                    available=False, reason="perf_missing", group="cache",
                ),
                "branch": MicroArchReading(
                    available=False, reason="perf_missing", group="branch",
                ),
            }
            return

        # Auto-detect platform event specs
        if self._specs_override is not None:
            self._platform_specs = self._specs_override
        else:
            self._platform_specs = detect_platform_event_specs(
                self.event_source_root,
            )
        specs = self._platform_specs

        # Build event groups
        self._groups = _build_groups(specs)
        platform_name = specs.name
        self._latest = {
            g.name: MicroArchReading(
                available=False, reason="initializing",
                group=g.name, platform=platform_name,
            )
            for g in self._groups
        }

        # Resolve scoping once
        self._scope_kind, self._scope_args = resolve_perf_scoping(
            perf_executable=resolved_perf,
            cgroup_path=self.cgroup_path,
            container_pid=self.container_pid,
        )
        logger.info(
            "micro-arch collector: platform=%s scope=%s interval=%.1fs groups=%d",
            platform_name, self._scope_kind, self.interval_s, len(self._groups),
        )

        group_idx = 0
        terminal_reasons = {
            "permission_denied", "pmu_unsupported", "perf_missing",
        }

        while not self._stop_event.is_set():
            group = self._groups[group_idx % len(self._groups)]
            reading = sample_core_events_once(
                group,
                interval_s=self.interval_s,
                perf_executable=resolved_perf,
                scope_args=(
                    list(self._scope_args) if self._scope_args else None
                ),
                scope_kind=self._scope_kind,
                platform_name=platform_name,
            )
            self._set_latest(reading)

            if not reading.available and reading.reason in terminal_reasons:
                logger.warning(
                    "micro-arch collector terminal: %s", reading.reason,
                )
                return

            group_idx += 1

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=max(2.0, self.interval_s + 1.0))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector_lock = threading.Lock()
_collector: MicroArchCollector | None = None
_TERMINAL_REASONS = {
    "unsupported_platform",
    "perf_missing",
    "pmu_unsupported",
    "permission_denied",
}


def _make_collector(
    interval_s: float,
    cgroup_path: Path | None = None,
    container_pid: int | None = None,
    _platform: str | None = None,
    _specs: PlatformEventSpecs | None = None,
    event_source_root: Path = EVENT_SOURCE_ROOT,
) -> MicroArchCollector:
    return MicroArchCollector(
        interval_s=interval_s,
        cgroup_path=cgroup_path,
        container_pid=container_pid,
        _platform=_platform,
        _specs=_specs,
        event_source_root=event_source_root,
    )


def get_micro_arch_collector(
    *,
    interval_s: float = 1.0,
    cgroup_path: Path | None = None,
    container_pid: int | None = None,
    _platform: str | None = None,
    _specs: PlatformEventSpecs | None = None,
    event_source_root: Path = EVENT_SOURCE_ROOT,
) -> MicroArchCollector:
    """Get or create the module-level ``MicroArchCollector`` singleton."""
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = _make_collector(
                interval_s, cgroup_path=cgroup_path,
                container_pid=container_pid, _platform=_platform,
                _specs=_specs, event_source_root=event_source_root,
            )
            _collector.start()
        elif not _collector.is_alive():
            latest = _collector.latest()
            all_terminal = True
            for reading in (
                latest.values() if isinstance(latest, dict) else [latest]
            ):
                if isinstance(reading, MicroArchReading):
                    if (
                        reading.available
                        or reading.reason not in _TERMINAL_REASONS
                    ):
                        all_terminal = False
                        break
            if all_terminal:
                return _collector
            _collector = _make_collector(
                interval_s, cgroup_path=cgroup_path,
                container_pid=container_pid, _platform=_platform,
                _specs=_specs, event_source_root=event_source_root,
            )
            _collector.start()
        return _collector


def attach_micro_arch(
    sample: dict[str, Any],
    *,
    interval_s: float = 1.0,
) -> None:
    """Attach micro-architecture metrics to a sample dict in-place."""
    collector = get_micro_arch_collector(interval_s=interval_s)
    latest_all = collector.latest()
    if not isinstance(latest_all, dict):
        return

    cache_reading = latest_all.get("cache")
    icache_reading = latest_all.get("icache")  # x86-only: L1I split from cache
    branch_reading = latest_all.get("branch")

    available = bool(
        (cache_reading is not None and cache_reading.available)
        or (icache_reading is not None and icache_reading.available)
        or (branch_reading is not None and branch_reading.available)
    )
    sample["micro_arch_available"] = available

    # Platform info
    platform = None
    if cache_reading is not None and cache_reading.platform is not None:
        platform = cache_reading.platform
    elif icache_reading is not None and icache_reading.platform is not None:
        platform = icache_reading.platform
    elif branch_reading is not None and branch_reading.platform is not None:
        platform = branch_reading.platform
    if platform is not None:
        sample["micro_arch_source"] = f"perf:core-pmu:{platform}"

    # Scope
    scope = None
    if cache_reading is not None and cache_reading.scope is not None:
        scope = cache_reading.scope
    elif icache_reading is not None and icache_reading.scope is not None:
        scope = icache_reading.scope
    elif branch_reading is not None and branch_reading.scope is not None:
        scope = branch_reading.scope
    if scope is not None:
        sample["micro_arch_scope"] = scope

    # Derived metrics from cache group (L1D + IPC)
    if cache_reading is not None:
        if not cache_reading.available and cache_reading.reason is not None:
            sample["micro_arch_reason"] = cache_reading.reason
        if cache_reading.l1d_hit_rate is not None:
            sample["l1d_hit_rate"] = cache_reading.l1d_hit_rate
        # L1I may be in cache group (ARM) or icache group (x86)
        if cache_reading.l1i_hit_rate is not None:
            sample["l1i_hit_rate"] = cache_reading.l1i_hit_rate
        if cache_reading.ipc is not None:
            sample["ipc"] = cache_reading.ipc
        if cache_reading.instructions_per_s is not None:
            sample["instructions_per_s"] = cache_reading.instructions_per_s

    # Derived metrics from icache group (x86-only: L1I + IPC)
    if icache_reading is not None:
        if icache_reading.l1i_hit_rate is not None:
            sample["l1i_hit_rate"] = icache_reading.l1i_hit_rate
        if "ipc" not in sample and icache_reading.ipc is not None:
            sample["ipc"] = icache_reading.ipc
        if (
            "instructions_per_s" not in sample
            and icache_reading.instructions_per_s is not None
        ):
            sample["instructions_per_s"] = icache_reading.instructions_per_s

    # Derived metrics from branch group
    if branch_reading is not None:
        if branch_reading.branch_miss_rate is not None:
            sample["branch_miss_rate"] = branch_reading.branch_miss_rate
        if branch_reading.bus_access_per_s is not None:
            sample["bus_access_per_s"] = branch_reading.bus_access_per_s
        if "ipc" not in sample and branch_reading.ipc is not None:
            sample["ipc"] = branch_reading.ipc
        if (
            "instructions_per_s" not in sample
            and branch_reading.instructions_per_s is not None
        ):
            sample["instructions_per_s"] = branch_reading.instructions_per_s

    # Raw metric series for aggregated analysis
    for reading in (cache_reading, icache_reading, branch_reading):
        if reading is None or not reading.available:
            continue
        for metric, value in reading.raw_counts.items():
            sample.setdefault("micro_arch_raw", {})[metric] = value


def reset_micro_arch_collector_for_tests() -> None:
    """Reset the module-level collector singleton (test helper)."""
    global _collector
    with _collector_lock:
        if _collector is not None:
            _collector.stop()
        _collector = None
