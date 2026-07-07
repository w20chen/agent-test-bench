from __future__ import annotations

import argparse
import json
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.experiments.probe_llc_topology import (
    build_placements,
    format_cpu_list,
    parse_cpu_list,
    probe_topology,
    write_outputs,
)
from scripts.experiments.run_kunpeng_llc_replay import _build_command
from scripts.experiments.run_kunpeng_llc_replay import run_replay


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cpu(
    root: Path,
    cpu: int,
    *,
    shared: str,
    core_id: int | None = None,
    socket_id: int = 0,
    node_id: int = 0,
) -> None:
    base = root / f"cpu{cpu}"
    resolved_core_id = core_id if core_id is not None else cpu
    _write(base / "topology" / "core_id", f"{resolved_core_id}\n")
    _write(base / "topology" / "physical_package_id", f"{socket_id}\n")
    (base / f"node{node_id}").mkdir(parents=True)
    _write(base / "cache" / "index0" / "level", "1\n")
    _write(base / "cache" / "index0" / "type", "Data\n")
    _write(base / "cache" / "index0" / "shared_cpu_list", f"{cpu}\n")
    _write(base / "cache" / "index3" / "level", "3\n")
    _write(base / "cache" / "index3" / "type", "Unified\n")
    _write(base / "cache" / "index3" / "shared_cpu_list", f"{shared}\n")


@contextmanager
def _temp_dir() -> Iterator[Path]:
    base = Path(".tmp-tests")
    base.mkdir(exist_ok=True)
    path = base / f"agent-test-bench-llc-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_parse_cpu_list() -> None:
    assert parse_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert format_cpu_list([0, 2, 4]) == "0,2,4"


def test_build_llc_and_inferred_cluster_placements() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-15\n")
        for cpu in range(8):
            _cpu(root, cpu, shared="0-7", socket_id=0, node_id=0)
        for cpu in range(8, 16):
            _cpu(root, cpu, shared=f"{cpu}", socket_id=1, node_id=1)

        topology = probe_topology(root)
        placements = build_placements(topology, agent_count=8)

    assert placements["compact_llc"].cpus == list(range(8))
    assert [a.cpu for a in placements["compact_llc"].agent_assignments] == list(range(8))
    assert placements["compact_llc"].llc_ids == ["0-7"]
    assert placements["spread_llc"].cpus == [0, 8, 9, 10, 11, 12, 13, 14]
    assert [a.cpuset_cpus for a in placements["spread_llc"].agent_assignments] == [
        "0",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
    ]
    assert placements["spread_clusters_same_llc"].cpus == [0, 4, 1, 5, 2, 6, 3, 7]
    assert "compact_cluster" not in placements
    assert placements["os_default"].cpus is None


def test_cluster_compact_is_valid_for_four_agents() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-7\n")
        for cpu in range(8):
            _cpu(root, cpu, shared="0-7")

        topology = probe_topology(root)
        placements = build_placements(topology, agent_count=4)

    assert placements["compact_cluster"].cpus == [0, 1, 2, 3]
    assert placements["compact_cluster"].groups[0].cluster_id == "0-7:cluster0"
    assert placements["spread_clusters_same_llc"].cpus == [0, 4, 1, 5]


def test_spread_across_four_llcs_round_robins_domains() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-31\n")
        for group in range(4):
            start = group * 8
            shared = f"{start}-{start + 7}"
            for cpu in range(start, start + 8):
                _cpu(root, cpu, shared=shared, node_id=group)

        topology = probe_topology(root)
        placements = build_placements(topology, agent_count=8)

    assert placements["compact_llc"].cpus == list(range(8))
    assert placements["spread_llc"].cpus == [0, 8, 16, 24, 1, 9, 17, 25]
    assert placements["spread_llc"].llc_ids == [
        "0-7",
        "8-15",
        "16-23",
        "24-31",
        "0-7",
        "8-15",
        "16-23",
        "24-31",
    ]


@pytest.mark.parametrize("agent_count", [1, 2, 4, 8])
def test_placements_have_exact_per_agent_assignments_for_supported_counts(
    agent_count: int,
) -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-31\n")
        for group in range(4):
            start = group * 8
            shared = f"{start}-{start + 7}"
            for cpu in range(start, start + 8):
                _cpu(root, cpu, shared=shared, node_id=group)

        placements = build_placements(probe_topology(root), agent_count=agent_count)

    assert placements["os_default"].cpus is None
    for placement in placements.values():
        assert len(placement.agent_assignments) == agent_count
        assert [item.agent_index for item in placement.agent_assignments] == list(
            range(agent_count)
        )
        if placement.name == "os_default":
            assert all(item.cpuset_cpus is None for item in placement.agent_assignments)
            continue
        assert placement.cpus is not None
        assert len(placement.cpus) == agent_count
        assert [item.cpu for item in placement.agent_assignments] == placement.cpus
        assert all(item.cpuset_cpus == str(item.cpu) for item in placement.agent_assignments)


def test_invalid_when_no_domain_can_allocate_requested_agents() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-3\n")
        for cpu in range(4):
            _cpu(root, cpu, shared="0-3")

        topology = probe_topology(root)

    with pytest.raises(RuntimeError, match="no valid placement"):
        build_placements(topology, agent_count=8)


def test_write_outputs() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-15\n")
        for cpu in range(8):
            _cpu(root, cpu, shared="0-7")
        for cpu in range(8, 16):
            _cpu(root, cpu, shared=f"{cpu}")

        topology = probe_topology(root)
        placements = build_placements(topology, agent_count=8)

        out = tmp_path / "out"
        write_outputs(output_dir=out, topology=topology, placements=placements)

        assert (out / "topology.txt").exists()
        placement_payload = json.loads((out / "placements.json").read_text())
        assert placement_payload["compact_llc"]["cpus"] == list(range(8))
        assert "cluster_id" in placement_payload["compact_llc"]["groups"][0]


def test_replay_command_passes_docker_cpuset_for_placement() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-15\n")
        for cpu in range(8):
            _cpu(root, cpu, shared="0-7")
        for cpu in range(8, 16):
            _cpu(root, cpu, shared=f"{cpu}")
        placements = build_placements(probe_topology(root), agent_count=8)

        command = _build_command(
            source_trace=Path("trace.jsonl"),
            task_source=Path("tasks.json"),
            output_dir=Path("out"),
            placement=placements["compact_llc"],
            container="docker",
            num_agents=8,
            replay_speed=1.0,
            network_mode="none",
            command_timeout_s=600.0,
            workers=1,
            prep_concurrency=8,
            resource_monitoring="on",
            ksys_monitoring="off",
            extra_args=[],
        )

    assert "--cpuset-cpus" not in command
    assert command.count("--agent-cpuset") == 8
    idx = command.index("--agent-cpuset")
    assert command[idx + 1] == "0"
    assert "--cpu-limit" in command
    assert command[command.index("--cpu-limit") + 1] == "1"


def test_replay_dry_run_supports_single_agent_without_shared_cpuset() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-7\n")
        for cpu in range(8):
            _cpu(root, cpu, shared="0-7")

        records = run_replay(
            argparse.Namespace(
                source_trace=Path("trace.jsonl"),
                task_source=Path("tasks.json"),
                output_root=tmp_path / "runs",
                sys_cpu_root=root,
                container="docker",
                placements="os_default,compact_llc",
                num_agents=1,
                cluster_size=4,
                replay_speed=1.0,
                network_mode="none",
                command_timeout=600.0,
                workers=1,
                prep_concurrency=1,
                resource_monitoring="on",
                ksys_monitoring="off",
                dry_run=True,
                simulate_args=[],
            )
        )

    by_placement = {record.placement: record for record in records}
    assert set(by_placement) == {"os_default", "compact_llc"}
    assert "--cpuset-cpus" not in by_placement["compact_llc"].command
    assert by_placement["compact_llc"].command.count("--agent-cpuset") == 1
    assert "--agent-cpuset" not in by_placement["os_default"].command
    assert by_placement["compact_llc"].agent_assignments[0]["cpuset_cpus"] == "0"
