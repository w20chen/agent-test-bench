from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.experiments.probe_kunpeng_llc_slices import (
    ChaseMeasurement,
    ChaseTrial,
    build_chase_command,
    build_trials,
    infer_candidate_clusters,
    parse_chase_output,
    parse_positive_int_list,
    run_trial,
    summarize_measurements,
)
from scripts.experiments.probe_llc_topology import CpuTopology


def _topology_with_smt() -> list[CpuTopology]:
    return [
        CpuTopology(
            cpu=cpu,
            core_id=cpu // 2,
            socket_id=0,
            numa_node=0,
            llc_id="0-15",
            llc_level=3,
            llc_shared_cpu_list="0-15",
        )
        for cpu in range(16)
    ]


def test_infer_candidate_clusters_uses_physical_core_representatives() -> None:
    clusters = infer_candidate_clusters(
        _topology_with_smt(),
        candidate_cluster_size=4,
    )

    assert [cluster.cpus for cluster in clusters] == [
        [0, 2, 4, 6],
        [8, 10, 12, 14],
    ]
    assert clusters[0].candidate_cluster_size == 4
    assert clusters[0].numa_node == 0
    assert clusters[0].llc_id == "0-15"


def test_build_trials_compares_same_and_other_candidate_clusters() -> None:
    trials = build_trials(
        _topology_with_smt(),
        candidate_cluster_sizes=[4],
        runs=1,
        aggressors_per_run=3,
        victim_cpus=[0],
    )
    by_mode = {trial.mode: trial for trial in trials}

    assert by_mode["baseline"].aggressor_cpus == []
    assert by_mode["same_candidate_cluster"].aggressor_cpus == [2, 4, 6]
    assert by_mode["same_candidate_cluster"].skip_reason is None
    assert by_mode["other_candidate_cluster_same_llc"].aggressor_cpus == [8, 10, 12]
    assert by_mode["other_candidate_cluster_same_llc"].skip_reason is None
    assert by_mode["other_llc_domain"].skip_reason == (
        "need 3 aggressors for other_llc_domain, found 0"
    )


def test_build_trials_marks_too_small_candidate_clusters_explicitly() -> None:
    trials = build_trials(
        _topology_with_smt(),
        candidate_cluster_sizes=[2],
        runs=1,
        aggressors_per_run=3,
        victim_cpus=[0],
    )
    same = next(trial for trial in trials if trial.mode == "same_candidate_cluster")
    other_same_llc = next(
        trial for trial in trials if trial.mode == "other_candidate_cluster_same_llc"
    )

    assert same.aggressor_cpus == [2]
    assert same.skip_reason is None
    assert other_same_llc.aggressor_cpus == [4]
    assert other_same_llc.skip_reason is None


def test_parse_chase_output_and_positive_int_list() -> None:
    assert parse_chase_output("ns_per_access 25.248 accesses 316861440") == (
        25.248,
        316861440,
    )
    assert parse_positive_int_list("4,8,4") == [4, 8]
    with pytest.raises(ValueError, match="could not parse"):
        parse_chase_output("no measurement")
    with pytest.raises(ValueError, match="positive"):
        parse_positive_int_list("4,0")


def test_build_chase_command_supports_numactl_and_plain_taskset() -> None:
    with_numactl = build_chase_command(
        cpu=8,
        chase_bin=Path("./chase"),
        membind="0",
        memory_mb=3,
        seconds=6,
    )
    without_numactl = build_chase_command(
        cpu=8,
        chase_bin=Path("./chase"),
        membind=None,
        memory_mb=3,
        seconds=6,
    )

    assert with_numactl == [
        "numactl",
        "--membind=0",
        "taskset",
        "-c",
        "8",
        f".{os.sep}chase",
        "3",
        "6",
    ]
    assert without_numactl == ["taskset", "-c", "8", f".{os.sep}chase", "3", "6"]


def test_chase_c_source_preserves_probe_contract() -> None:
    source = Path("scripts/experiments/chase.c").read_text(encoding="utf-8")

    assert "ns_per_access %.3f accesses %" in source
    assert "CACHE_LINE_BYTES = 64" in source
    assert "clock_gettime" in source
    assert "taskset" not in source
    assert "numactl" not in source


def test_chase_c_compiles_when_gcc_is_available(tmp_path: Path) -> None:
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is not available")

    output = tmp_path / ("chase.exe" if os.name == "nt" else "chase")
    completed = subprocess.run(
        [
            gcc,
            "-O3",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-o",
            str(output),
            "scripts/experiments/chase.c",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()


def test_summary_verdict_uses_configurable_margin() -> None:
    trial_topology = _topology_with_smt()
    trials = build_trials(
        trial_topology,
        candidate_cluster_sizes=[4],
        runs=1,
        aggressors_per_run=3,
        victim_cpus=[0],
    )
    measurements = []
    for trial in trials:
        if trial.skip_reason is not None:
            continue
        ns = {
            "baseline": 20.0,
            "same_candidate_cluster": 25.0,
            "other_candidate_cluster_same_llc": 21.0,
        }[trial.mode]
        measurements.append(
            trial_measurement(
                trial.candidate_cluster_size,
                trial.mode,
                ns,
            )
        )

    summary = summarize_measurements(measurements, support_margin_ratio=0.05)

    assert summary[0]["candidate_cluster_size"] == 4
    assert summary[0]["verdict"] == "supported_by_interference"
    assert summary[0]["same_candidate_cluster_delta_vs_baseline"] == pytest.approx(0.25)


def test_summary_requires_same_cluster_slowdown_over_baseline() -> None:
    measurements = [
        trial_measurement(4, "baseline", 20.0),
        trial_measurement(4, "same_candidate_cluster", 19.0),
        trial_measurement(4, "other_candidate_cluster_same_llc", 18.0),
    ]

    summary = summarize_measurements(measurements, support_margin_ratio=0.05)

    assert summary[0]["verdict"] == "not_supported_by_interference"


def test_summary_normalizes_by_victim_before_candidate_verdict() -> None:
    measurements = [
        trial_measurement(4, "baseline", 10.0, victim_cpu=0),
        trial_measurement(4, "same_candidate_cluster", 13.0, victim_cpu=0),
        trial_measurement(
            4,
            "other_candidate_cluster_same_llc",
            12.0,
            victim_cpu=0,
        ),
        trial_measurement(4, "baseline", 100.0, victim_cpu=8),
        trial_measurement(4, "same_candidate_cluster", 105.0, victim_cpu=8),
        trial_measurement(
            4,
            "other_candidate_cluster_same_llc",
            101.0,
            victim_cpu=8,
        ),
    ]

    summary = summarize_measurements(measurements, support_margin_ratio=0.05)

    assert len(summary[0]["per_victim_effects"]) == 2
    assert summary[0]["same_candidate_cluster_delta_vs_baseline"] == pytest.approx(
        0.175
    )


def test_run_trial_marks_aggressor_failure_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePipe:
        def read(self) -> str:
            return "aggressor failed"

    class FakeProcess:
        returncode = 7
        stderr = FakePipe()

        def wait(self) -> int:
            return self.returncode

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        return FakeProcess()

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="ns_per_access 20.0 accesses 10\n",
            stderr="",
        )

    monkeypatch.setattr(
        "scripts.experiments.probe_kunpeng_llc_slices.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "scripts.experiments.probe_kunpeng_llc_slices.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "scripts.experiments.probe_kunpeng_llc_slices.time.sleep",
        lambda _seconds: None,
    )

    measurement = run_trial(
        ChaseTrial(
            candidate_cluster_size=4,
            mode="same_candidate_cluster",
            run=1,
            victim_cpu=0,
            aggressor_cpus=[2],
            victim_cluster_id="cluster0",
            aggressor_cluster_ids=["cluster0"],
        ),
        chase_bin=Path("./chase"),
        membind="0",
        victim_mb=3,
        victim_sec=6,
        aggressor_mb=16,
        aggressor_sec=9,
        warmup_sec=1.0,
        dry_run=False,
    )

    assert measurement.skipped is True
    assert measurement.returncode == 7
    assert measurement.aggressor_returncodes == [7]
    assert measurement.ns_per_access is None
    assert measurement.skip_reason == "aggressor process failed with return codes [7]"


def test_run_trial_marks_malformed_chase_output_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="not a chase measurement\n",
            stderr="",
        )

    monkeypatch.setattr(
        "scripts.experiments.probe_kunpeng_llc_slices.subprocess.run",
        fake_run,
    )

    measurement = run_trial(
        ChaseTrial(
            candidate_cluster_size=4,
            mode="baseline",
            run=1,
            victim_cpu=0,
            aggressor_cpus=[],
            victim_cluster_id="cluster0",
            aggressor_cluster_ids=[],
        ),
        chase_bin=Path("./chase"),
        membind="0",
        victim_mb=3,
        victim_sec=6,
        aggressor_mb=16,
        aggressor_sec=9,
        warmup_sec=1.0,
        dry_run=False,
    )

    assert measurement.skipped is True
    assert measurement.returncode == 1
    assert measurement.ns_per_access is None
    assert measurement.skip_reason is not None
    assert "could not parse chase output" in measurement.skip_reason


def trial_measurement(
    candidate_cluster_size: int,
    mode: str,
    ns_per_access: float,
    *,
    victim_cpu: int = 0,
) -> ChaseMeasurement:
    return ChaseMeasurement(
        candidate_cluster_size=candidate_cluster_size,
        mode=mode,
        run=1,
        victim_cpu=victim_cpu,
        aggressor_cpus=[],
        victim_cluster_id=f"cluster{victim_cpu}",
        aggressor_cluster_ids=[],
        victim_command=[],
        aggressor_commands=[],
        ns_per_access=ns_per_access,
        accesses=1,
        stdout="",
        stderr="",
        returncode=0,
        aggressor_returncodes=[],
        skipped=False,
        skip_reason=None,
    )
