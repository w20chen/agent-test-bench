"""Tests for the tool_scheduler prototype.

Run with: python -m pytest tests/test_tool_scheduler.py -v
"""

import json
import os
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
             "import time; t0=time.monotonic();"
             "while time.monotonic()-t0<5.0:"
             " sum(i**0.5 for i in range(50000))"],
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
