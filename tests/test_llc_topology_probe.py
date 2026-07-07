from __future__ import annotations

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


def test_build_same_and_round_robin_spread_placements() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-15\n")
        for cpu in range(8):
            _cpu(root, cpu, shared="0-7", socket_id=0, node_id=0)
        for cpu in range(8, 16):
            _cpu(root, cpu, shared=f"{cpu}", socket_id=1, node_id=1)

        topology = probe_topology(root)
        placements = build_placements(topology, agent_count=8)

    assert placements["same_llc"].cpus == list(range(8))
    assert placements["same_llc"].llc_ids == ["0-7"]
    assert placements["spread_llc"].cpus == [0, 8, 9, 10, 11, 12, 13, 14]
    assert len(set(placements["spread_llc"].llc_ids)) == 8
    assert placements["os_default"].cpus is None


def test_spread_balances_eight_agents_over_four_llcs() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-31\n")
        for group in range(4):
            start = group * 8
            shared = f"{start}-{start + 7}"
            for cpu in range(start, start + 8):
                _cpu(root, cpu, shared=shared)

        topology = probe_topology(root)
        placements = build_placements(topology, agent_count=8)

    assert placements["same_llc"].cpus == list(range(8))
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


def test_spread_requires_at_least_two_llcs() -> None:
    with _temp_dir() as tmp_path:
        root = tmp_path / "cpu"
        _write(root / "online", "0-7\n")
        for cpu in range(8):
            _cpu(root, cpu, shared="0-7")

        topology = probe_topology(root)

    with pytest.raises(RuntimeError, match="only one LLC group"):
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
        assert placement_payload["same_llc"]["cpus"] == list(range(8))


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
            placement=placements["same_llc"],
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

    assert "--cpuset-cpus" in command
    assert command[command.index("--cpuset-cpus") + 1] == "0,1,2,3,4,5,6,7"
    assert "--cpu-limit" in command
    assert command[command.index("--cpu-limit") + 1] == "1"
