#!/usr/bin/env python3
"""Probe CPU/LLC topology and generate 8-core placement candidates."""

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
class Placement:
    name: str
    cpus: list[int] | None
    llc_ids: list[str]
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


def build_placements(
    topology: list[CpuTopology],
    *,
    agent_count: int = 8,
) -> dict[str, Placement]:
    """Build OS-default, same-LLC, and spread-LLC placements."""
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")

    by_llc: dict[str, list[int]] = {}
    for rec in sorted(topology, key=lambda item: item.cpu):
        by_llc.setdefault(rec.llc_id, []).append(rec.cpu)

    same_candidates = [
        (llc_id, cpus) for llc_id, cpus in by_llc.items() if len(cpus) >= agent_count
    ]
    if not same_candidates:
        raise RuntimeError(
            f"no LLC group contains {agent_count} online CPUs; cannot build same_llc"
        )
    same_llc_id, same_cpus = sorted(
        same_candidates,
        key=lambda item: (min(item[1]), item[0]),
    )[0]

    if len(by_llc) < 2:
        raise RuntimeError(
            "only one LLC group found; cannot build a cross-LLC spread placement"
        )

    llc_groups = [
        (llc_id, cpus)
        for llc_id, cpus in sorted(by_llc.items(), key=lambda item: min(item[1]))
    ]
    spread_pairs: list[tuple[str, int]] = []
    offset = 0
    while len(spread_pairs) < agent_count:
        made_progress = False
        for llc_id, cpus in llc_groups:
            if offset >= len(cpus):
                continue
            spread_pairs.append((llc_id, cpus[offset]))
            made_progress = True
            if len(spread_pairs) == agent_count:
                break
        if not made_progress:
            raise RuntimeError(
                f"only {len(spread_pairs)} CPUs are available across LLC groups; "
                f"need {agent_count} for spread_llc"
            )
        offset += 1
    spread_cpus = [cpu for _, cpu in spread_pairs]
    spread_llcs = [llc_id for llc_id, _ in spread_pairs]

    return {
        "os_default": Placement(
            name="os_default",
            cpus=None,
            llc_ids=[],
            description="No taskset affinity; Linux scheduler default placement.",
        ),
        "same_llc": Placement(
            name="same_llc",
            cpus=same_cpus[:agent_count],
            llc_ids=[same_llc_id],
            description=f"{agent_count} allowed CPUs from one LLC group.",
        ),
        "spread_llc": Placement(
            name="spread_llc",
            cpus=spread_cpus,
            llc_ids=spread_llcs,
            description=(
                f"{agent_count} allowed CPUs distributed round-robin across "
                f"{len(llc_groups)} LLC groups."
            ),
        ),
    }


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
        cpu_text = "default" if placement.cpus is None else format_cpu_list(placement.cpus)
        lines.append(f"{placement.name}: cpus={cpu_text} llcs={placement.llc_ids}")
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
    args = parser.parse_args()

    topology = probe_topology(args.sys_cpu_root)
    placements = build_placements(topology, agent_count=args.agent_count)
    write_outputs(output_dir=args.output_dir, topology=topology, placements=placements)
    print(f"Wrote topology and placements to {args.output_dir}")


if __name__ == "__main__":
    main()
