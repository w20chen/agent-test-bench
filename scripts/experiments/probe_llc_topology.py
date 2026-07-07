#!/usr/bin/env python3
"""Probe CPU/LLC topology and generate locality-domain placements."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SYS_CPU = Path("/sys/devices/system/cpu")


@dataclass(frozen=True, slots=True)
class CpuTopology:
    cpu: int
    core_id: int | None
    socket_id: int | None
    numa_node: int | None
    llc_id: str
    llc_level: int
    llc_shared_cpu_list: str


@dataclass(frozen=True, slots=True)
class PlacementGroup:
    name: str
    cpus: list[int]
    numa_node: int | None
    llc_id: str
    description: str
    cluster_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentAssignment:
    agent_index: int
    cpuset_cpus: str | None
    cpu: int | None
    numa_node: int | None
    llc_id: str | None
    cluster_id: str | None


@dataclass(frozen=True, slots=True)
class Placement:
    name: str
    cpus: list[int] | None
    llc_ids: list[str]
    numa_nodes: list[int | None]
    groups: list[PlacementGroup]
    agent_assignments: list[AgentAssignment]
    description: str


def parse_cpu_list(value: str) -> list[int]:
    """Parse Linux CPU-list syntax such as ``0-3,8,10-11``."""
    cpus: list[int] = []
    text = value.strip()
    if not text:
        return cpus
    for part in text.split(","):
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid CPU range: {part!r}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_list(cpus: list[int]) -> str:
    """Format a CPU list as comma-separated numbers for ``taskset -c``."""
    return ",".join(str(cpu) for cpu in cpus)


def format_cpu_range_list(cpus: list[int]) -> str:
    """Format CPUs using Linux range syntax, e.g. ``0-3,80-83``."""
    if not cpus:
        return ""
    ordered = sorted(set(cpus))
    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None


def _online_cpus(root: Path) -> list[int]:
    online = _read_text(root / "online")
    if online:
        return parse_cpu_list(online)
    cpus: list[int] = []
    for child in root.glob("cpu[0-9]*"):
        try:
            cpus.append(int(child.name[3:]))
        except ValueError:
            continue
    return sorted(cpus)


def _numa_node(cpu_dir: Path) -> int | None:
    for node in cpu_dir.glob("node[0-9]*"):
        try:
            return int(node.name[4:])
        except ValueError:
            continue
    return None


def _llc_for_cpu(cpu_dir: Path) -> tuple[str, int, str]:
    caches: list[tuple[int, str, str]] = []
    for index in (cpu_dir / "cache").glob("index*"):
        level = _read_int(index / "level")
        shared = _read_text(index / "shared_cpu_list")
        cache_type = (_read_text(index / "type") or "").lower()
        if level is None or shared is None:
            continue
        if cache_type not in {"unified", "data"}:
            continue
        # `id` may repeat across packages on some kernels; shared_cpu_list is
        # the stable grouping key for the placement decision we need here.
        caches.append((level, shared, shared))
    if not caches:
        raise RuntimeError(f"no cache topology found for {cpu_dir.name}")
    level, llc_id, shared = max(caches, key=lambda item: item[0])
    return llc_id, level, shared


def probe_topology(root: Path = SYS_CPU) -> list[CpuTopology]:
    """Read CPU topology from a Linux sysfs CPU tree."""
    records: list[CpuTopology] = []
    for cpu in _online_cpus(root):
        cpu_dir = root / f"cpu{cpu}"
        topology = cpu_dir / "topology"
        llc_id, llc_level, shared = _llc_for_cpu(cpu_dir)
        records.append(
            CpuTopology(
                cpu=cpu,
                core_id=_read_int(topology / "core_id"),
                socket_id=_read_int(topology / "physical_package_id"),
                numa_node=_numa_node(cpu_dir),
                llc_id=llc_id,
                llc_level=llc_level,
                llc_shared_cpu_list=shared,
            )
        )
    return records


def _read_numa_distances(sys_cpu_root: Path) -> dict[int, dict[int, int]]:
    node_root = sys_cpu_root.parent / "node"
    distances: dict[int, dict[int, int]] = {}
    for node_dir in sorted(node_root.glob("node[0-9]*")):
        try:
            node = int(node_dir.name[4:])
        except ValueError:
            continue
        text = _read_text(node_dir / "distance")
        if not text:
            continue
        values = [int(part) for part in text.split()]
        distances[node] = {idx: value for idx, value in enumerate(values)}
    return distances


def _split_evenly(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _choose_domain_pair(
    domains: list[tuple[tuple[int | None, str], list[int]]],
    *,
    sys_cpu_root: Path,
    far: bool,
) -> list[tuple[tuple[int | None, str], list[int]]]:
    anchor = domains[0]
    anchor_node = anchor[0][0]
    candidates = domains[1:]
    if anchor_node is None:
        return [anchor, candidates[-1 if far else 0]]

    distances = _read_numa_distances(sys_cpu_root)
    anchor_distances = distances.get(anchor_node, {})

    def distance(item: tuple[tuple[int | None, str], list[int]]) -> tuple[int, int]:
        node = item[0][0]
        value = anchor_distances.get(node, 1_000_000 if far else 1_000)
        return value, min(item[1])

    chosen = max(candidates, key=distance) if far else min(candidates, key=distance)
    return [anchor, chosen]


def _make_group(
    *,
    name: str,
    domain_key: tuple[int | None, str],
    domain_cpus: list[int],
    count: int,
    description: str,
    cluster_id: str | None = None,
) -> PlacementGroup:
    if count <= 0:
        raise ValueError(f"{name} requested non-positive CPU count: {count}")
    if len(domain_cpus) < count:
        raise RuntimeError(
            f"{name} needs {count} CPUs from NUMA/LLC domain {domain_key}, "
            f"but only {len(domain_cpus)} are online"
        )
    numa_node, llc_id = domain_key
    return PlacementGroup(
        name=name,
        cpus=domain_cpus[:count],
        numa_node=numa_node,
        llc_id=llc_id,
        description=description,
        cluster_id=cluster_id,
    )


def _make_placement(name: str, groups: list[PlacementGroup], description: str) -> Placement:
    cpus = [cpu for group in groups for cpu in group.cpus]
    return Placement(
        name=name,
        cpus=cpus,
        llc_ids=[group.llc_id for group in groups],
        numa_nodes=[group.numa_node for group in groups],
        groups=groups,
        agent_assignments=[
            AgentAssignment(
                agent_index=idx,
                cpuset_cpus=str(cpu),
                cpu=cpu,
                numa_node=group.numa_node,
                llc_id=group.llc_id,
                cluster_id=group.cluster_id,
            )
            for idx, (group, cpu) in enumerate(
                (group, cpu) for group in groups for cpu in group.cpus
            )
        ],
        description=description,
    )


def _make_group_from_cpus(
    *,
    name: str,
    domain_key: tuple[int | None, str],
    cpus: list[int],
    description: str,
    cluster_id: str | None = None,
) -> PlacementGroup:
    if not cpus:
        raise ValueError(f"{name} must contain at least one CPU")
    numa_node, llc_id = domain_key
    return PlacementGroup(
        name=name,
        cpus=cpus,
        numa_node=numa_node,
        llc_id=llc_id,
        description=description,
        cluster_id=cluster_id,
    )


def _round_robin_select(
    groups: list[tuple[tuple[int | None, str], str | None, list[int]]],
    count: int,
) -> list[tuple[tuple[int | None, str], str | None, int]]:
    selected: list[tuple[tuple[int | None, str], str | None, int]] = []
    offset = 0
    while len(selected) < count:
        made_progress = False
        for domain_key, cluster_id, cpus in groups:
            if offset >= len(cpus):
                continue
            selected.append((domain_key, cluster_id, cpus[offset]))
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise RuntimeError(
                f"only {len(selected)} CPUs are available across placement groups; "
                f"need {count}"
            )
        offset += 1
    return selected


def _placement_from_assignments(
    name: str,
    assignments: list[tuple[tuple[int | None, str], str | None, int]],
    description: str,
) -> Placement:
    grouped: dict[tuple[tuple[int | None, str], str | None], list[int]] = {}
    for domain_key, cluster_id, cpu in assignments:
        grouped.setdefault((domain_key, cluster_id), []).append(cpu)
    groups = [
        _make_group_from_cpus(
            name=f"group{idx}",
            domain_key=domain_key,
            cpus=cpus,
            cluster_id=cluster_id,
            description=description,
        )
        for idx, ((domain_key, cluster_id), cpus) in enumerate(grouped.items())
    ]
    return Placement(
        name=name,
        cpus=[cpu for _domain_key, _cluster_id, cpu in assignments],
        llc_ids=[domain_key[1] for domain_key, _cluster_id, _cpu in assignments],
        numa_nodes=[domain_key[0] for domain_key, _cluster_id, _cpu in assignments],
        groups=groups,
        agent_assignments=[
            AgentAssignment(
                agent_index=idx,
                cpuset_cpus=str(cpu),
                cpu=cpu,
                numa_node=domain_key[0],
                llc_id=domain_key[1],
                cluster_id=cluster_id,
            )
            for idx, (domain_key, cluster_id, cpu) in enumerate(assignments)
        ],
        description=description,
    )


def _inferred_clusters(
    domains: list[tuple[tuple[int | None, str], list[int]]],
    *,
    cluster_size: int,
) -> list[tuple[tuple[int | None, str], str, list[int]]]:
    clusters: list[tuple[tuple[int | None, str], str, list[int]]] = []
    for domain_key, cpus in domains:
        llc_id = domain_key[1]
        ordered = sorted(cpus)
        for idx, start in enumerate(range(0, len(ordered), cluster_size)):
            cluster_cpus = ordered[start : start + cluster_size]
            if cluster_cpus:
                clusters.append((domain_key, f"{llc_id}:cluster{idx}", cluster_cpus))
    return clusters


def build_placements(
    topology: list[CpuTopology],
    *,
    agent_count: int = 8,
    sys_cpu_root: Path = SYS_CPU,
    cluster_size: int = 4,
) -> dict[str, Placement]:
    """Build LLC-domain and inferred sub-LLC cluster placements.

    ``cluster_size`` defaults to 4 because public TaiShan v110/Kunpeng 920
    analysis describes quad-core CPU clusters.  The cluster IDs here are an
    explicit inference from CPU numbering within each Linux LLC shared CPU
    list, not firmware-provided physical CCL identifiers.
    """
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")
    if cluster_size <= 0:
        raise ValueError("cluster_size must be positive")

    by_domain: dict[tuple[int | None, str], list[int]] = {}
    for rec in sorted(topology, key=lambda item: item.cpu):
        by_domain.setdefault((rec.numa_node, rec.llc_id), []).append(rec.cpu)

    domains = sorted(by_domain.items(), key=lambda item: min(item[1]))
    if not domains:
        raise RuntimeError("no CPU locality domains found")

    placements: dict[str, Placement] = {
        "os_default": Placement(
            name="os_default",
            cpus=None,
            llc_ids=[],
            numa_nodes=[],
            groups=[],
            agent_assignments=[
                AgentAssignment(
                    agent_index=idx,
                    cpuset_cpus=None,
                    cpu=None,
                    numa_node=None,
                    llc_id=None,
                    cluster_id=None,
                )
                for idx in range(agent_count)
            ],
            description="No affinity; Linux scheduler default placement.",
        )
    }

    compact_candidates = [
        (domain_key, cpus) for domain_key, cpus in domains if len(cpus) >= agent_count
    ]
    if compact_candidates:
        domain_key, cpus = compact_candidates[0]
        placements["compact_llc"] = _make_placement(
            "compact_llc",
            [
                _make_group(
                    name="llc0",
                    domain_key=domain_key,
                    domain_cpus=cpus,
                    count=agent_count,
                    description="All agents placed in one Linux LLC locality domain.",
                )
            ],
            "All agents placed in one Linux LLC locality domain.",
        )

    llc_rr_groups = [(domain_key, None, cpus) for domain_key, cpus in domains]
    if agent_count >= 2 and len(llc_rr_groups) >= 2:
        placements["spread_llc"] = _placement_from_assignments(
            "spread_llc",
            _round_robin_select(llc_rr_groups, agent_count),
            "Agents distributed round-robin across Linux LLC locality domains.",
        )

    clusters = _inferred_clusters(domains, cluster_size=cluster_size)
    cluster_compact_candidates = [
        (domain_key, cluster_id, cpus)
        for domain_key, cluster_id, cpus in clusters
        if len(cpus) >= agent_count
    ]
    if cluster_compact_candidates:
        domain_key, cluster_id, cpus = cluster_compact_candidates[0]
        placements["compact_cluster"] = _make_placement(
            "compact_cluster",
            [
                _make_group(
                    name="cluster0",
                    domain_key=domain_key,
                    domain_cpus=cpus,
                    count=agent_count,
                    cluster_id=cluster_id,
                    description=(
                        "All agents placed in one inferred sub-LLC CPU cluster."
                    ),
                )
            ],
            (
                "All agents placed in one inferred sub-LLC CPU cluster. "
                "Cluster identity is inferred from CPU numbering."
            ),
        )

    same_llc_cluster_groups: list[tuple[tuple[int | None, str], str | None, list[int]]] = []
    for domain_key, domain_cpus in domains:
        domain_clusters = [
            (key, cluster_id, cpus)
            for key, cluster_id, cpus in clusters
            if key == domain_key
        ]
        if len(domain_clusters) >= 2 and len(domain_cpus) >= agent_count:
            same_llc_cluster_groups = domain_clusters
            break
    if agent_count >= 2 and same_llc_cluster_groups:
        placements["spread_clusters_same_llc"] = _placement_from_assignments(
            "spread_clusters_same_llc",
            _round_robin_select(same_llc_cluster_groups, agent_count),
            (
                "Agents distributed across inferred sub-LLC CPU clusters while "
                "remaining inside one Linux LLC locality domain."
            ),
        )

    if agent_count >= 2 and len(clusters) >= 2:
        placements["spread_clusters_all"] = _placement_from_assignments(
            "spread_clusters_all",
            _round_robin_select(clusters, agent_count),
            "Agents distributed round-robin across all inferred CPU clusters.",
        )

    if agent_count >= 2 and len(domains) >= 2:
        near_domains = _choose_domain_pair(domains, sys_cpu_root=sys_cpu_root, far=False)
        near_counts = _split_evenly(agent_count, 2)
        if all(len(cpus) >= count for (_domain_key, cpus), count in zip(near_domains, near_counts)):
            placements["near_numa_spread"] = _make_placement(
                "near_numa_spread",
                [
                    _make_group(
                        name=f"node{idx}",
                        domain_key=domain_key,
                        domain_cpus=cpus,
                        count=count,
                        description="Agents placed in one of two nearest NUMA/LLC domains.",
                    )
                    for idx, ((domain_key, cpus), count) in enumerate(zip(near_domains, near_counts))
                ],
                "Agents split across two nearby NUMA/LLC domains.",
            )

        far_domains = _choose_domain_pair(domains, sys_cpu_root=sys_cpu_root, far=True)
        far_counts = _split_evenly(agent_count, 2)
        if all(len(cpus) >= count for (_domain_key, cpus), count in zip(far_domains, far_counts)):
            placements["far_numa_spread"] = _make_placement(
                "far_numa_spread",
                [
                    _make_group(
                        name=f"node{idx}",
                        domain_key=domain_key,
                        domain_cpus=cpus,
                        count=count,
                        description="Agents placed in one of two distant NUMA/LLC domains.",
                    )
                    for idx, ((domain_key, cpus), count) in enumerate(zip(far_domains, far_counts))
                ],
                "Agents split across two far NUMA/LLC domains.",
            )

    if len(placements) == 1:
        raise RuntimeError(
            f"no valid placement could allocate {agent_count} CPUs; "
            "check online CPUs, LLC topology, and cluster_size"
        )

    return placements


def write_outputs(
    *,
    output_dir: Path,
    topology: list[CpuTopology],
    placements: dict[str, Placement],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "topology.json").write_text(
        json.dumps([asdict(record) for record in topology], indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "placements.json").write_text(
        json.dumps(
            {name: asdict(placement) for name, placement in placements.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "CPU topology",
        "============",
        "",
        "cpu core socket numa llc_id llc_level shared_cpus",
    ]
    for rec in topology:
        lines.append(
            f"{rec.cpu} {rec.core_id} {rec.socket_id} {rec.numa_node} "
            f"{rec.llc_id} {rec.llc_level} {rec.llc_shared_cpu_list}"
        )
    lines.extend(["", "Placements", "==========", ""])
    for placement in placements.values():
        cpu_text = "default" if placement.cpus is None else format_cpu_range_list(placement.cpus)
        lines.append(
            f"{placement.name}: cpus={cpu_text} numa={placement.numa_nodes} "
            f"llcs={placement.llc_ids}"
        )
        for group in placement.groups:
            cluster_text = "" if group.cluster_id is None else f" cluster={group.cluster_id}"
            lines.append(
                f"  - {group.name}: cpus={format_cpu_range_list(group.cpus)} "
                f"numa={group.numa_node} llc={group.llc_id}{cluster_text}"
            )
        for assignment in placement.agent_assignments:
            if assignment.cpuset_cpus is None:
                lines.append(f"    agent{assignment.agent_index}: default")
                continue
            cluster_text = (
                "" if assignment.cluster_id is None else f" cluster={assignment.cluster_id}"
            )
            lines.append(
                f"    agent{assignment.agent_index}: cpuset={assignment.cpuset_cpus} "
                f"numa={assignment.numa_node} llc={assignment.llc_id}{cluster_text}"
            )
    (output_dir / "topology.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe CPU/LLC topology and emit placement candidates.",
    )
    parser.add_argument(
        "--sys-cpu-root",
        type=Path,
        default=SYS_CPU,
        help="Linux sysfs CPU root (default: /sys/devices/system/cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("traces/experiments/kunpeng_llc/topology"),
        help="Directory for topology.json, placements.json, topology.txt.",
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=8,
        help="Number of agents/cores required for each placement.",
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=4,
        help=(
            "Inferred sub-LLC CPU cluster size. Default 4 matches public "
            "TaiShan v110 CCL descriptions; override if host probing shows "
            "a different mapping."
        ),
    )
    args = parser.parse_args()

    topology = probe_topology(args.sys_cpu_root)
    placements = build_placements(
        topology,
        agent_count=args.agent_count,
        sys_cpu_root=args.sys_cpu_root,
        cluster_size=args.cluster_size,
    )
    write_outputs(output_dir=args.output_dir, topology=topology, placements=placements)
    print(f"Wrote topology and placements to {args.output_dir}")


if __name__ == "__main__":
    main()
