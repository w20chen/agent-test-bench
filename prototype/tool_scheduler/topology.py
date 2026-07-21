"""Hardware CPU topology discovery from Linux sysfs.

Reads /sys/devices/system/cpu/ to build a mapping of:
  CPU -> physical core -> SMT sibling -> LLC group -> NUMA node

Gracefully degrades on non-Linux or when sysfs is unavailable.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SYS_CPU = Path("/sys/devices/system/cpu")


@dataclass
class CpuInfo:
    """Information about a single logical CPU."""

    cpu_id: int
    physical_package_id: int
    core_id: int
    thread_siblings: list[int]  # SMT siblings (self included)
    numa_node: int
    llc_group: int  # LLC group index (may equal numa_node if no finer granularity)


@dataclass
class Topology:
    """Complete CPU topology snapshot."""

    cpus: dict[int, CpuInfo] = field(default_factory=dict)
    numa_nodes: list[int] = field(default_factory=list)
    llc_groups: dict[int, dict[int, list[int]]] = field(
        default_factory=dict
    )  # numa_node -> llc_group -> [cpus]
    physical_cores_per_cpu: dict[int, int] = field(
        default_factory=dict
    )  # cpu_id -> physical_core_id
    smt_siblings: dict[int, list[int]] = field(
        default_factory=dict
    )  # physical_core -> [smt threads]

    total_logical_cpus: int = 0
    total_physical_cores: int = 0
    smt_enabled: bool = False
    available: bool = False  # True if topology was successfully read


def _read_int(path: Path) -> Optional[int]:
    """Read a single integer from a sysfs file."""
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _read_int_list(path: Path) -> list[int]:
    """Read a cpu list like '0-3,8-11' and expand to list of ints."""
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return []
    return _parse_cpu_list(text)


def _parse_cpu_list(text: str) -> list[int]:
    """Parse a cpu list like '0-3,8,10-12' into a flat list."""
    cpus: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _discover_llc_groups(cpu_id: int) -> int:
    """Discover which LLC group a CPU belongs to.

    Reads /sys/devices/system/cpu/cpuN/cache/index*/shared_cpu_list
    to find the largest last-level cache that is shared among CPUs.
    Returns the indexN number of the LLC, or the NUMA node as fallback.
    """
    cpu_cache_dir = _SYS_CPU / f"cpu{cpu_id}" / "cache"
    if not cpu_cache_dir.exists():
        return -1

    best_index: Optional[int] = None
    best_level: int = -1
    best_shared_count: int = 1

    for index_dir in sorted(cpu_cache_dir.iterdir()):
        if not index_dir.is_dir() or not index_dir.name.startswith("index"):
            continue
        try:
            level = _read_int(index_dir / "level")
            shared_list = _read_int_list(index_dir / "shared_cpu_list")
            if level is None or not shared_list:
                continue
            # Prefer the highest level cache (LLC)
            if level > best_level or (
                level == best_level and len(shared_list) > best_shared_count
            ):
                best_level = level
                best_shared_count = len(shared_list)
                # Extract index number
                idx_str = index_dir.name.replace("index", "")
                best_index = int(idx_str)
        except (OSError, ValueError):
            continue

    if best_index is not None:
        return best_index
    return -1


def discover() -> Topology:
    """Discover CPU topology from sysfs.

    Returns a Topology object. If sysfs is unavailable (non-Linux),
    returns a Topology with available=False.
    """
    if sys.platform != "linux":
        logger.warning("CPU topology discovery only supported on Linux")
        return Topology(available=False)

    if not _SYS_CPU.exists():
        logger.warning("sysfs cpu topology not available at %s", _SYS_CPU)
        return Topology(available=False)

    # Discover all online CPUs
    online_path = _SYS_CPU / "online"
    all_cpu_ids: list[int] = []
    if online_path.exists():
        all_cpu_ids = _read_int_list(online_path)
    else:
        # Fallback: enumerate cpuN directories
        for entry in sorted(_SYS_CPU.iterdir()):
            if entry.is_dir() and entry.name.startswith("cpu"):
                try:
                    cpu_id = int(entry.name[3:])
                    all_cpu_ids.append(cpu_id)
                except ValueError:
                    continue

    if not all_cpu_ids:
        logger.warning("No CPUs discovered via sysfs")
        return Topology(available=False)

    # Collect per-CPU info
    cpus: dict[int, CpuInfo] = {}
    numa_nodes_set: set[int] = set()
    llc_groups_raw: dict[int, dict[int, list[int]]] = {}  # numa -> llc -> [cpus]

    for cpu_id in all_cpu_ids:
        cpu_dir = _SYS_CPU / f"cpu{cpu_id}"
        if not cpu_dir.exists():
            continue

        topo_dir = cpu_dir / "topology"

        pkg_id = _read_int(topo_dir / "physical_package_id")
        core_id = _read_int(topo_dir / "core_id")
        thread_siblings = _read_int_list(topo_dir / "thread_siblings_list")

        if pkg_id is None or core_id is None:
            continue

        # NUMA node
        numa_node = -1
        node_path = cpu_dir / "node"
        if node_path.exists():
            # /sys/devices/system/cpu/cpuN/node -> symlink
            try:
                node_target = os.readlink(str(node_path))
                # e.g., ../../node/node0
                node_name = node_target.rstrip("/").split("/")[-1]
                if node_name.startswith("node"):
                    numa_node = int(node_name[4:])
            except (OSError, ValueError):
                pass

        # Fallback: check /sys/devices/system/node/node*/cpulist
        if numa_node < 0:
            for node_dir in sorted(Path("/sys/devices/system/node").glob("node*")):
                try:
                    node_cpus = _read_int_list(node_dir / "cpulist")
                    if cpu_id in node_cpus:
                        node_name = node_dir.name
                        if node_name.startswith("node"):
                            numa_node = int(node_name[4:])
                        break
                except (OSError, ValueError):
                    continue

        # LLC group
        llc_group = _discover_llc_groups(cpu_id)
        if llc_group < 0:
            llc_group = numa_node  # Degrade to NUMA node granularity

        info = CpuInfo(
            cpu_id=cpu_id,
            physical_package_id=pkg_id,
            core_id=core_id,
            thread_siblings=thread_siblings or [cpu_id],
            numa_node=numa_node,
            llc_group=llc_group,
        )
        cpus[cpu_id] = info
        numa_nodes_set.add(numa_node)

        # Build LLC group mapping
        llc_groups_raw.setdefault(numa_node, {}).setdefault(llc_group, []).append(cpu_id)

    # Compute derived fields
    numa_nodes = sorted(numa_nodes_set)

    # Map: physical_core_id -> list of logical CPU ids (SMT siblings)
    # Physical core is identified by (pkg_id, core_id) tuple
    physical_cores: dict[tuple[int, int], list[int]] = {}
    for cpu_id, info in cpus.items():
        key = (info.physical_package_id, info.core_id)
        physical_cores.setdefault(key, []).append(cpu_id)

    # Build physical_cores_per_cpu and smt_siblings
    physical_cores_per_cpu: dict[int, int] = {}
    smt_siblings: dict[int, list[int]] = {}
    for key, cpu_list in physical_cores.items():
        # Use the first CPU in the group as the physical core identifier
        phys_core_id = min(cpu_list)
        for cpu_id in cpu_list:
            physical_cores_per_cpu[cpu_id] = phys_core_id
        smt_siblings[phys_core_id] = sorted(cpu_list)

    total_physical = len(physical_cores)
    has_smt = any(len(v) > 1 for v in physical_cores.values())

    topology = Topology(
        cpus=cpus,
        numa_nodes=numa_nodes,
        llc_groups=llc_groups_raw,
        physical_cores_per_cpu=physical_cores_per_cpu,
        smt_siblings=smt_siblings,
        total_logical_cpus=len(all_cpu_ids),
        total_physical_cores=total_physical,
        smt_enabled=has_smt,
        available=True,
    )

    logger.info(
        "Topology: %d logical CPUs, %d physical cores, %d NUMA nodes, SMT=%s",
        topology.total_logical_cpus,
        topology.total_physical_cores,
        len(numa_nodes),
        has_smt,
    )
    return topology


def hardcoded_topology() -> Topology:
    """Build a hardcoded 320-core Kunpeng-style topology.

    Layout:
      - 320 logical CPUs (0..319), SMT enabled (2 threads per physical core)
      - SMT siblings: (0,1), (2,3), ..., (318,319)  — 160 physical cores
      - 4 NUMA nodes, 80 logical CPUs each:
        NUMA 0: logical 0-79   (40 phys, 10 LLC groups)
        NUMA 1: logical 80-159  (40 phys, 10 LLC groups)
        NUMA 2: logical 160-239 (40 phys, 10 LLC groups)
        NUMA 3: logical 240-319 (40 phys, 10 LLC groups)
      - LLC clusters: 4 physical cores (8 logical) each
        Within each NUMA: LLC 0..9
    """
    TOTAL_LOGICAL = 320
    NUM_NUMA = 4
    LOGICAL_PER_NUMA = 80   # 40 physical × 2 SMT
    PHYS_PER_NUMA = 40
    LOGICAL_PER_LLC = 8     # 4 physical × 2 SMT

    cpus: dict[int, CpuInfo] = {}
    llc_groups_raw: dict[int, dict[int, list[int]]] = {
        n: {} for n in range(NUM_NUMA)
    }

    for cpu_id in range(TOTAL_LOGICAL):
        phys_core = cpu_id // 2                     # 0..159
        sibling = cpu_id + 1 if cpu_id % 2 == 0 else cpu_id - 1
        numa = cpu_id // LOGICAL_PER_NUMA           # 0..3
        phys_in_numa = phys_core - (numa * PHYS_PER_NUMA)  # 0..39 within NUMA
        llc = phys_in_numa // 4                     # 0..9 within NUMA

        info = CpuInfo(
            cpu_id=cpu_id,
            physical_package_id=numa,
            core_id=phys_core,
            thread_siblings=[cpu_id, sibling] if cpu_id % 2 == 0 else [sibling, cpu_id],
            numa_node=numa,
            llc_group=llc,
        )
        cpus[cpu_id] = info

        llc_groups_raw[numa].setdefault(llc, []).append(cpu_id)

    # Build physical_cores_per_cpu and smt_siblings
    physical_cores_per_cpu: dict[int, int] = {}
    smt_siblings: dict[int, list[int]] = {}
    for phys in range(160):
        lo = phys * 2
        hi = phys * 2 + 1
        physical_cores_per_cpu[lo] = lo
        physical_cores_per_cpu[hi] = lo
        smt_siblings[lo] = [lo, hi]

    topology = Topology(
        cpus=cpus,
        numa_nodes=list(range(NUM_NUMA)),
        llc_groups=llc_groups_raw,
        physical_cores_per_cpu=physical_cores_per_cpu,
        smt_siblings=smt_siblings,
        total_logical_cpus=TOTAL_LOGICAL,
        total_physical_cores=160,
        smt_enabled=True,
        available=True,
    )

    logger.info(
        "Hardcoded topology: %d logical CPUs, %d physical cores, "
        "%d NUMA nodes, SMT=True, %d LLC groups",
        topology.total_logical_cpus,
        topology.total_physical_cores,
        len(topology.numa_nodes),
        sum(len(g) for g in topology.llc_groups.values()),
    )
    return topology


def get_current_numa_node(pid: int) -> Optional[int]:
    """Get the NUMA node where a process is currently executing.

    Reads from /proc/pid/numa_maps or falls back to sched_getcpu.
    Returns None on failure.
    """
    if sys.platform != "linux":
        return None

    # Try /proc/pid/numa_maps
    try:
        maps = Path(f"/proc/{pid}/numa_maps").read_text(encoding="ascii", errors="replace")
        # First line might look like: "N0=12345 N1=0 ..."
        for token in maps.split():
            if token.startswith("N") and "=" in token:
                node_str, count_str = token.split("=", 1)
                try:
                    count = int(count_str)
                    if count > 0:
                        return int(node_str[1:])
                except ValueError:
                    continue
    except OSError:
        pass

    return None
