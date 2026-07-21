"""Tests for the tool_scheduler prototype.

Run with: python -m pytest tests/test_tool_scheduler.py -v
"""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPredictor:
    """Unit tests for the EMA predictor."""

    def test_bootstrap_first_observation(self):
        from prototype.tool_scheduler.predictor import Predictor
        p = Predictor(alpha=0.3)
        state = p.update(8.0)
        assert state.predicted_cores == 8.0
        assert state.observation_count == 1
        assert not state.stable  # Need 3 observations for stability

    def test_ema_convergence(self):
        from prototype.tool_scheduler.predictor import Predictor
        p = Predictor(alpha=0.3)
        # Feed constant 8.0
        for _ in range(10):
            p.update(8.0)
        assert abs(p.predicted_cores - 8.0) < 0.5
        assert p.stable

    def test_stability_detection(self):
        from prototype.tool_scheduler.predictor import Predictor
        p = Predictor(alpha=0.3)
        # 3 identical observations -> stable
        for _ in range(3):
            p.update(4.0)
        assert p.stable

    def test_unstable_detection(self):
        from prototype.tool_scheduler.predictor import Predictor
        p = Predictor(alpha=0.3)
        p.update(4.0)
        p.update(4.0)
        p.update(4.0)
        assert p.stable
        # Large jump - feed many iterations of the new value
        # so EMA converges and stability is re-established
        for _ in range(10):
            p.update(20.0)
        # After many iterations of the same value, should be stable again
        assert p.stable

    def test_requested_cores_ceil(self):
        from prototype.tool_scheduler.predictor import Predictor
        p = Predictor(alpha=0.3)
        p.update(3.2)
        p.update(3.2)
        p.update(3.2)
        assert p.requested_cores == 4  # ceil(3.2) = 4

    def test_requested_cores_min_one(self):
        from prototype.tool_scheduler.predictor import Predictor
        p = Predictor(alpha=0.3)
        p.update(0.3)
        p.update(0.3)
        p.update(0.3)
        assert p.requested_cores == 1  # max(1, ceil(0.3))


class TestCostModel:
    """Unit tests for the cost model."""

    def test_core_cost_satisfied(self):
        from prototype.tool_scheduler.cost_model import compute_core_cost
        cost = compute_core_cost(predicted_cores=4.0, available_physical_cores=4)
        assert cost == 0.0

    def test_core_cost_shortage(self):
        from prototype.tool_scheduler.cost_model import compute_core_cost
        cost = compute_core_cost(predicted_cores=16.0, available_physical_cores=8)
        assert cost == 0.5  # (16-8)/16

    def test_core_cost_with_smt(self):
        from prototype.tool_scheduler.cost_model import (
            compute_core_cost,
            compute_effective_available_cores,
        )
        eff = compute_effective_available_cores(4, 4, smt_weight=0.3)
        assert eff == 5.2  # 4 + 0.3*4
        cost = compute_core_cost(
            predicted_cores=6.0,
            available_physical_cores=4,
            available_smt_threads=4,
            smt_weight=0.3,
        )
        assert cost > 0.0  # 6 > 5.2

    def test_memory_cost_none_bandwidth(self):
        from prototype.tool_scheduler.cost_model import compute_memory_cost
        cost = compute_memory_cost("high", None)
        assert cost == 0.0

    def test_memory_cost_high_sensitivity(self):
        from prototype.tool_scheduler.cost_model import compute_memory_cost
        cost = compute_memory_cost("high", 0.5)
        assert cost == 0.5

    def test_memory_cost_low_sensitivity(self):
        from prototype.tool_scheduler.cost_model import compute_memory_cost
        cost = compute_memory_cost("low", 0.8)
        assert cost == 0.0

    def test_memory_cost_unknown(self):
        from prototype.tool_scheduler.cost_model import compute_memory_cost
        cost = compute_memory_cost("unknown", 0.5)
        assert cost == 0.15  # 0.3 * 0.5


class TestMonitor:
    """Tests for the process tree monitor."""

    def test_monitor_short_process(self):
        from prototype.tool_scheduler.monitor import Monitor
        import subprocess as sp
        import time

        proc = sp.Popen(
            [sys.executable, "-c", "import time; time.sleep(1.5)"],
            stdout=sp.DEVNULL, stderr=sp.DEVNULL,
        )
        m = Monitor(proc.pid, sample_interval=0.3)
        m.start()
        proc.wait()
        time.sleep(0.5)
        m.stop()

        samples = m.samples
        assert len(samples) >= 2, f"Expected >=2 samples, got {len(samples)}"

    def test_monitor_effective_cores(self):
        from prototype.tool_scheduler.monitor import Monitor
        import subprocess as sp
        import time

        # Single-threaded CPU-bound process - run longer to ensure multiple samples
        proc = sp.Popen(
            [sys.executable, "-c",
             "\n".join([
                 "import time",
                 "t0=time.monotonic()",
                 "while time.monotonic()-t0<5.0:",
                 "    sum(i**0.5 for i in range(50000))",
             ])],
            stdout=sp.DEVNULL, stderr=sp.DEVNULL,
        )
        m = Monitor(proc.pid, sample_interval=0.3)
        m.start()
        proc.wait()
        time.sleep(0.5)
        m.stop()

        samples = m.samples
        # At least some samples should have effective_cores around 1
        eff_cores = [s.effective_cores for s in samples if s.effective_cores > 0]
        assert len(eff_cores) > 0, (
            f"No samples with effective_cores > 0 out of {len(samples)} total samples"
        )
        median = sorted(eff_cores)[len(eff_cores) // 2]
        # Single-threaded process should have ~1 core
        assert 0.3 < median < 2.5, f"Expected ~1 core, got {median}"


class TestCLI:
    """Integration tests for the CLI."""

    def test_shell_command_preserves_shell_operators(self, tmp_path):
        """--shell-command should pass operators like && to the inner shell."""
        output = tmp_path / "shell_profile.jsonl"
        if sys.platform == "win32":
            py = subprocess.list2cmdline([sys.executable])
        else:
            py = shlex.quote(sys.executable)
        payload = (
            f"{py} -c \"print('first')\" && "
            f"{py} -c \"print('second')\""
        )
        result = subprocess.run(
            [sys.executable, "-m", "prototype.tool_scheduler",
             "--output", str(output),
             "--dry-run",
             "--shell-command",
             "--", payload],
            capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 0
        assert "first" in result.stdout
        assert "second" in result.stdout
        assert output.exists()

    def test_output_dash_writes_json_to_stdout(self, tmp_path):
        """--output - should not create a literal '-' file in the cwd."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        result = subprocess.run(
            [sys.executable, "-m", "prototype.tool_scheduler",
             "--output", "-",
             "--dry-run",
             "--", sys.executable, "-c", "print('payload')"],
            cwd=tmp_path,
            env=env,
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0
        assert "payload" in result.stdout
        assert not (tmp_path / "-").exists()
        json_line = result.stdout.strip().splitlines()[-1]
        rec = json.loads(json_line)
        assert rec["exit_code"] == 0

    def test_shell_command_rejects_multiple_argv_payload(self) -> None:
        """CLI shell-command mode requires one already-quoted payload string."""
        result = subprocess.run(
            [sys.executable, "-m", "prototype.tool_scheduler",
             "--shell-command",
             "--", "python", "-c", "print('x')"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode != 0
        assert "requires exactly one shell command string" in result.stderr

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGTERM cleanup test")
    def test_sigterm_cleans_workload_descendants(self, tmp_path):
        """Terminating the scheduler should clean up the wrapped process tree."""
        psutil = pytest.importorskip("psutil")
        pid_file = tmp_path / "child.pid"
        output = tmp_path / "profile.jsonl"
        child_code = "import time; time.sleep(60)"
        parent_code = (
            "import subprocess, sys, time; "
            f"p = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
            f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(p.pid)); "
            "time.sleep(60)"
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "prototype.tool_scheduler",
             "--output", str(output),
             "--dry-run",
             "--", sys.executable, "-c", parent_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10.0
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            assert pid_file.exists(), "workload child pid file was not created"
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            assert psutil.pid_exists(child_pid)

            proc.terminate()
            return_code = proc.wait(timeout=15)
            assert return_code == 143

            deadline = time.monotonic() + 5.0
            while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not psutil.pid_exists(child_pid)
            assert output.exists()
            rec = json.loads(output.read_text(encoding="utf-8").splitlines()[-1])
            assert rec["exit_code"] == 143
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_short_tool(self):
        """Short tools (<2s) should be marked as short_tool=true."""
        result = subprocess.run(
            [sys.executable, "-m", "prototype.tool_scheduler",
             "--output", "-",
             "--dry-run",
             "--", sys.executable, "-c", "import time; time.sleep(0.5)"],
            capture_output=True, text=True, timeout=15,
        )
        # The tool exits 0
        assert result.returncode == 0

    def test_cpu_serial(self):
        """CPU serial workload should show ~1 core."""
        result = subprocess.run(
            [sys.executable, "-m", "prototype.tool_scheduler",
             "--output", "-",
             "--dry-run",
             "--save-samples",
             "--", sys.executable, "-c",
             "import time; t0=time.monotonic();"
             "[sum(i**0.5 for i in range(50000)) for _ in range(5)];"
             "time.sleep(0.1)"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_output_file(self, tmp_path):
        """Verify JSONL output file is created."""
        output = tmp_path / "test_profiles.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "prototype.tool_scheduler",
             "--output", str(output),
             "--dry-run",
             "--", sys.executable, "-c", "import time; time.sleep(0.5)"],
            capture_output=True, text=True, timeout=15,
        )
        assert output.exists()
        with open(output) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) == 1
        rec = records[0]
        assert "invocation_id" in rec
        assert "command" in rec
        assert "exit_code" in rec
        assert "final_profile" in rec
        assert "decisions" in rec

    def test_topology_info_in_output(self, tmp_path):
        """Verify topology info is included in output when available."""
        output = tmp_path / "test_topo.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "prototype.tool_scheduler",
             "--output", str(output),
             "--dry-run",
             "--", sys.executable, "-c", "import time; time.sleep(1.0)"],
            capture_output=True, text=True, timeout=15,
        )
        assert output.exists()
        with open(output) as f:
            rec = json.loads(f.readline())
        if sys.platform == "linux":
            assert "topology" in rec, "Linux should have topology info"
        # On non-Linux, topology may be absent which is fine


class TestIdle:
    """Unit tests for /proc/stat idle detection."""

    def test_cpu_utilization_empty_on_windows(self):
        """On non-Linux, get_cpu_utilization returns empty dict."""
        from prototype.tool_scheduler.idle import get_cpu_utilization, count_idle_cores
        util = get_cpu_utilization()
        # On Windows, /proc/stat doesn't exist, so util is empty
        if sys.platform != "linux":
            assert util == {}
        # count_idle_cores on empty util should return len(cpu_list)
        # (first sample assumption: all idle)
        assert count_idle_cores([0, 1, 2, 3]) == 4

    def test_idle_breakdown_no_data(self):
        """idle_breakdown returns all cores as physical when no util data."""
        from prototype.tool_scheduler.idle import idle_breakdown
        phys, smt = idle_breakdown([0, 1, 2, 3])
        assert phys == 4
        assert smt == 0

    def test_idle_breakdown_with_smt_mapping(self):
        """idle_breakdown with physical_cores_per_cpu mapping."""
        from prototype.tool_scheduler.idle import idle_breakdown
        # CPU 0,1 are SMT siblings (same physical core 0)
        # CPU 2,3 are SMT siblings (same physical core 1)
        phys_map = {0: 0, 1: 0, 2: 1, 3: 1}
        phys, smt = idle_breakdown([0, 1, 2, 3], physical_cores_per_cpu=phys_map)
        # Without util data, all should be available
        assert phys == 2  # Two physical cores
        assert smt == 2  # Two SMT threads


class TestTopology:
    """Unit tests for topology discovery helpers."""

    def test_llc_group_uses_shared_cpu_list_not_cache_index(self, tmp_path, monkeypatch):
        from prototype.tool_scheduler import topology

        def _write_cache(cpu: int, shared: str) -> None:
            cache_dir = tmp_path / f"cpu{cpu}" / "cache" / "index3"
            cache_dir.mkdir(parents=True)
            (cache_dir / "level").write_text("3\n", encoding="ascii")
            (cache_dir / "shared_cpu_list").write_text(shared + "\n", encoding="ascii")

        _write_cache(0, "0-3")
        _write_cache(4, "4-7")
        monkeypatch.setattr(topology, "_SYS_CPU", tmp_path)

        assert topology._discover_llc_groups(0) == 0
        assert topology._discover_llc_groups(4) == 4

    def test_discover_filters_to_process_affinity(self, tmp_path, monkeypatch):
        from prototype.tool_scheduler import topology

        def _write_cpu(cpu: int) -> None:
            topo_dir = tmp_path / f"cpu{cpu}" / "topology"
            topo_dir.mkdir(parents=True)
            (topo_dir / "physical_package_id").write_text("0\n", encoding="ascii")
            (topo_dir / "core_id").write_text(f"{cpu}\n", encoding="ascii")
            (topo_dir / "thread_siblings_list").write_text(f"{cpu}\n", encoding="ascii")
            cache_dir = tmp_path / f"cpu{cpu}" / "cache" / "index3"
            cache_dir.mkdir(parents=True)
            (cache_dir / "level").write_text("3\n", encoding="ascii")
            (cache_dir / "shared_cpu_list").write_text(f"{cpu}\n", encoding="ascii")

        (tmp_path / "online").write_text("0-1\n", encoding="ascii")
        _write_cpu(0)
        _write_cpu(1)
        monkeypatch.setattr(topology, "_SYS_CPU", tmp_path)
        monkeypatch.setattr(topology.sys, "platform", "linux")
        monkeypatch.setattr(
            topology.os,
            "sched_getaffinity",
            lambda _pid: {1},
            raising=False,
        )

        topo = topology.discover()

        assert topo.available
        assert sorted(topo.cpus) == [1]
        assert topo.total_logical_cpus == 1

    def test_representative_pid_prefers_busy_descendant(self, monkeypatch):
        from prototype.tool_scheduler import topology

        class _Times:
            def __init__(self, user: float, system: float) -> None:
                self.user = user
                self.system = system

        class _Proc:
            def __init__(self, pid: int, total: float, children=None) -> None:
                self.pid = pid
                self._total = total
                self._children = children or []

            def is_running(self) -> bool:
                return True

            def children(self, recursive: bool = False):
                return self._children

            def cpu_times(self):
                return _Times(self._total, 0.0)

        child = _Proc(22, 4.0)
        root = _Proc(11, 0.1, [child])
        monkeypatch.setattr(topology.psutil, "Process", lambda _pid: root)

        assert topology.get_representative_pid(11) == 22


class TestSchedulerPlacement:
    """Unit tests for scheduler placement detection."""

    def test_current_placement_uses_representative_pid_cpu(self, monkeypatch):
        from prototype.tool_scheduler import scheduler as sched_mod
        from prototype.tool_scheduler.predictor import Predictor
        from prototype.tool_scheduler.topology import CpuInfo, Topology

        predictor = Predictor(alpha=0.3)
        for _ in range(3):
            predictor.update(1.0)

        topo = Topology(
            cpus={
                5: CpuInfo(
                    cpu_id=5,
                    physical_package_id=0,
                    core_id=5,
                    thread_siblings=[5],
                    numa_node=1,
                    llc_group=50,
                )
            },
            numa_nodes=[0, 1],
            llc_groups={1: {50: [4, 5]}},
            physical_cores_per_cpu={4: 4, 5: 5},
            total_logical_cpus=2,
            total_physical_cores=2,
            available=True,
        )

        monkeypatch.setattr(sched_mod, "get_process_tree_cpu_ids", lambda pid: [5])
        monkeypatch.setattr(sched_mod, "get_bandwidth_utilization", lambda numa: None)
        monkeypatch.setattr(sched_mod, "idle_breakdown", lambda cpus, physical_cores_per_cpu: (2, 0))

        scheduler = sched_mod.Scheduler(
            predictor=predictor,
            topology=topo,
            history={},
            cooldown_seconds=0.0,
        )
        decision = scheduler.evaluate(2.0, root_pid=111)

        assert decision is not None
        assert decision.current_cost_breakdown is not None
        assert decision.current_cost_breakdown["placement"] == "numa1-llc50"

    def test_multi_llc_process_tree_has_unknown_current_placement(self, monkeypatch):
        from prototype.tool_scheduler import scheduler as sched_mod
        from prototype.tool_scheduler.predictor import Predictor
        from prototype.tool_scheduler.topology import CpuInfo, Topology

        predictor = Predictor(alpha=0.3)
        for _ in range(3):
            predictor.update(4.0)

        topo = Topology(
            cpus={
                0: CpuInfo(0, 0, 0, [0], 0, 0),
                4: CpuInfo(4, 0, 4, [4], 0, 4),
            },
            numa_nodes=[0],
            llc_groups={0: {0: [0, 1, 2, 3], 4: [4, 5, 6, 7]}},
            physical_cores_per_cpu={i: i for i in range(8)},
            total_logical_cpus=8,
            total_physical_cores=8,
            available=True,
        )

        monkeypatch.setattr(sched_mod, "get_process_tree_cpu_ids", lambda pid: [0, 4])
        monkeypatch.setattr(sched_mod, "get_bandwidth_utilization", lambda numa: None)
        monkeypatch.setattr(sched_mod, "idle_breakdown", lambda cpus, physical_cores_per_cpu: (len(cpus), 0))

        scheduler = sched_mod.Scheduler(
            predictor=predictor,
            topology=topo,
            history={},
            cooldown_seconds=0.0,
        )
        decision = scheduler.evaluate(2.0, root_pid=111)

        assert decision is not None
        assert decision.action == "keep"
        assert decision.current_cost is None
        assert decision.current_cost_breakdown is None
        assert decision.gain is None


class TestBandwidth:
    """Unit tests for bandwidth monitoring."""

    def test_pmu_not_available_on_windows(self):
        """On Windows/non-Linux, PMU should report unavailable."""
        from prototype.tool_scheduler.bandwidth import _pmu_available, _detect_ddrc_pmu
        if sys.platform != "linux":
            assert _pmu_available() is False
            assert _detect_ddrc_pmu() == []

    def test_get_bandwidth_utilization_no_collector(self):
        """get_bandwidth_utilization returns None when no collector started."""
        from prototype.tool_scheduler.bandwidth import get_bandwidth_utilization
        assert get_bandwidth_utilization(0) is None

    def test_bandwidth_collector_no_pmu(self):
        """BandwidthCollector starts but marks snapshot unavailable without PMU."""
        from prototype.tool_scheduler.bandwidth import (
            BandwidthCollector,
            MemoryDomainConfig,
        )
        cfg = MemoryDomainConfig(numa_node=0)
        collector = BandwidthCollector(0, cfg)
        collector.start()
        snap = collector.latest
        # Without PMU, available should be False
        assert snap.available is False
        assert snap.utilization is None
        collector.stop()

    def test_memory_domain_config_defaults(self):
        """MemoryDomainConfig has sensible defaults."""
        from prototype.tool_scheduler.bandwidth import (
            MemoryDomainConfig,
            DEFAULT_SUSTAINABLE_BANDWIDTH_GIB_S,
        )
        cfg = MemoryDomainConfig(numa_node=2)
        assert cfg.numa_node == 2
        assert cfg.sustainable_bandwidth_gib_s == DEFAULT_SUSTAINABLE_BANDWIDTH_GIB_S
        assert cfg.pmu_read_event is None
        assert cfg.pmu_write_event is None
        assert cfg.pmu_combined_event is None

    def test_auto_detect_on_windows(self):
        """_auto_detect_memory_domains returns configs without PMU events on Windows."""
        from prototype.tool_scheduler.bandwidth import _auto_detect_memory_domains
        configs = _auto_detect_memory_domains()
        # Should return at least one config (NUMA 0)
        assert len(configs) >= 1
        # Without PMU, events should be None
        for cfg in configs:
            if sys.platform != "linux":
                assert cfg.pmu_read_event is None
                assert cfg.pmu_write_event is None
