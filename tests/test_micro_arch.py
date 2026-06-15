"""Tests for harness.micro_arch -- cross-platform PMU event sampling."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from harness.micro_arch import (
    ARMV8_RAW_SPECS,
    GENERIC_SPECS,
    X86_AMD_SPECS,
    X86_INTEL_SPECS,
    CoreEventGroup,
    MicroArchCollector,
    MicroArchReading,
    PlatformEventSpecs,
    _build_groups,
    _derive_metrics,
    _detect_armv8_core_pmu,
    _detect_x86_pmu,
    _parse_perf_count,
    _parse_perf_stat_output,
    _read_cpuinfo_vendor,
    attach_micro_arch,
    detect_platform_event_specs,
    get_micro_arch_collector,
    reset_micro_arch_collector_for_tests,
    resolve_perf_scoping,
    sample_core_events_once,
)
from harness.memory_bandwidth import (
    _detect_arm_ddrc_backend,
    detect_perf_backend,
)


@pytest.fixture(autouse=True)
def _reset_collector() -> None:
    reset_micro_arch_collector_for_tests()
    yield
    reset_micro_arch_collector_for_tests()


# ==========================================================================
# Platform event specs
# ==========================================================================

def test_armv8_specs_has_l1i() -> None:
    assert ARMV8_RAW_SPECS.has_l1i is True


def test_armv8_specs_has_branch() -> None:
    assert ARMV8_RAW_SPECS.has_branch is True


def test_x86_intel_specs_has_l1i() -> None:
    assert X86_INTEL_SPECS.has_l1i is True


def test_x86_intel_specs_has_branch() -> None:
    assert X86_INTEL_SPECS.has_branch is True


def test_x86_amd_specs_has_l1i() -> None:
    assert X86_AMD_SPECS.has_l1i is True


def test_x86_amd_specs_has_branch() -> None:
    assert X86_AMD_SPECS.has_branch is True


def test_x86_amd_specs_uses_raw_events() -> None:
    """AMD specs must use cpu/ raw event syntax for L1I to bypass generic mapping."""
    assert X86_AMD_SPECS.l1i_access == "cpu/event=0x80/"
    assert X86_AMD_SPECS.l1i_miss == "cpu/event=0x81/"
    # L1D uses standard named events (work correctly on AMD)
    assert X86_AMD_SPECS.l1d_access == "L1-dcache-loads"
    assert X86_AMD_SPECS.l1d_miss == "L1-dcache-load-misses"


def test_generic_specs_no_l1i() -> None:
    assert GENERIC_SPECS.has_l1i is False


def test_generic_specs_has_branch() -> None:
    assert GENERIC_SPECS.has_branch is True


# ==========================================================================
# Event group construction
# ==========================================================================

def test_build_groups_armv8() -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)
    assert len(groups) == 2
    cache, branch = groups

    assert cache.name == "cache"
    assert len(cache.event_specs) == 6  # cycles, instr, l1d_acc, l1d_miss, l1i_acc, l1i_miss
    assert "r04" in cache.event_specs  # l1d_access
    assert "r03" in cache.event_specs  # l1d_miss
    assert "r14" in cache.event_specs  # l1i_access
    assert "r01" in cache.event_specs  # l1i_miss

    assert branch.name == "branch"
    assert "r12" in branch.event_specs  # branch_inst
    assert "r10" in branch.event_specs  # branch_miss
    assert "r19" in branch.event_specs  # bus_access


def test_build_groups_x86() -> None:
    """x86 splits L1D and L1I into separate groups to avoid multiplexing."""
    groups = _build_groups(X86_INTEL_SPECS)
    assert len(groups) == 3
    cache, icache, branch = groups

    assert cache.name == "cache"
    assert len(cache.event_specs) == 4  # cycles, instr, L1D access, L1D miss
    assert "L1-dcache-loads" in cache.event_specs
    assert "L1-dcache-load-misses" in cache.event_specs
    assert "L1-icache-loads" not in cache.event_specs  # moved to icache group

    assert icache.name == "icache"
    assert len(icache.event_specs) == 4  # cycles, instr, L1I access, L1I miss
    assert "L1-icache-loads" in icache.event_specs
    assert "L1-icache-load-misses" in icache.event_specs

    assert branch.name == "branch"
    assert "branch-instructions" in branch.event_specs
    assert "branch-misses" in branch.event_specs


def test_build_groups_amd() -> None:
    """AMD also splits L1D and L1I into separate groups."""
    groups = _build_groups(X86_AMD_SPECS)
    assert len(groups) == 3
    cache, icache, branch = groups

    # L1D in cache group (named events)
    assert "L1-dcache-loads" in cache.event_specs
    assert "L1-dcache-load-misses" in cache.event_specs

    # L1I in icache group (AMD raw events)
    assert icache.name == "icache"
    assert "cpu/event=0x80/" in icache.event_specs
    assert "cpu/event=0x81/" in icache.event_specs

    assert "branch-instructions" in branch.event_specs
    assert "branch-misses" in branch.event_specs


def test_build_groups_generic() -> None:
    groups = _build_groups(GENERIC_SPECS)
    cache, branch = groups

    # Generic: 4 events in cache group (cycles, instr, cache-refs, cache-misses)
    assert len(cache.event_specs) == 4
    assert "cache-references" in cache.event_specs
    assert "cache-misses" in cache.event_specs

    # Generic: 4 events in branch group (cycles, instr, branch-inst, branch-miss)
    assert len(branch.event_specs) == 4


def test_all_groups_within_counter_limit() -> None:
    """Each event group must fit within the PMU counter limit.

    ARMv8: ≤ 6 counters.  x86: ≤ 4 generic counters (fixed counters
    handle cycles/instructions, so each group is 4 events).
    """
    for specs in (ARMV8_RAW_SPECS, X86_INTEL_SPECS, X86_AMD_SPECS, GENERIC_SPECS):
        for group in _build_groups(specs):
            # x86 groups are split to ≤ 4 events each; ARMv8 ≤ 6
            max_events = 4 if specs.name.startswith("x86") else 6
            assert len(group.event_specs) <= max_events, (
                f"Group '{group.name}' ({specs.name}) has "
                f"{len(group.event_specs)} events (max {max_events})"
            )


def test_all_groups_have_matching_metrics() -> None:
    for specs in (ARMV8_RAW_SPECS, X86_INTEL_SPECS, X86_AMD_SPECS, GENERIC_SPECS):
        for group in _build_groups(specs):
            assert len(group.event_specs) == len(group.metrics), (
                f"Group '{group.name}' ({specs.name}): "
                f"{len(group.event_specs)} events vs {len(group.metrics)} metrics"
            )


# ==========================================================================
# PMU detection
# ==========================================================================

def test_detect_armv8_core_pmu_finds_device(tmp_path: Path) -> None:
    (tmp_path / "armv8_pmuv3_0").mkdir()
    (tmp_path / "breakpoint").mkdir()
    assert _detect_armv8_core_pmu(tmp_path) == "armv8_pmuv3_0"


def test_detect_armv8_core_pmu_finds_cortex(tmp_path: Path) -> None:
    (tmp_path / "armv8_cortex_a76").mkdir()
    assert _detect_armv8_core_pmu(tmp_path) == "armv8_cortex_a76"


def test_detect_armv8_core_pmu_none(tmp_path: Path) -> None:
    (tmp_path / "uncore_imc_0").mkdir()
    assert _detect_armv8_core_pmu(tmp_path) is None


def test_detect_x86_pmu_true(tmp_path: Path) -> None:
    (tmp_path / "cpu").mkdir()
    assert _detect_x86_pmu(tmp_path) is True


def test_detect_x86_pmu_ignores_uncore(tmp_path: Path) -> None:
    (tmp_path / "uncore_imc_0").mkdir()
    assert _detect_x86_pmu(tmp_path) is False


def test_detect_x86_pmu_empty(tmp_path: Path) -> None:
    assert _detect_x86_pmu(tmp_path) is False


def test_detect_platform_prefers_armv8(tmp_path: Path) -> None:
    (tmp_path / "armv8_pmuv3_0").mkdir()
    (tmp_path / "cpu").mkdir()
    result = detect_platform_event_specs(tmp_path)
    assert result.name == "armv8-raw"


def test_detect_platform_falls_back_to_x86(tmp_path: Path) -> None:
    (tmp_path / "cpu").mkdir()
    result = detect_platform_event_specs(tmp_path)
    # Without /proc/cpuinfo AMD vendor, x86 PMU → Intel specs
    assert result.name == "x86-intel"


def test_detect_platform_amd_vendor_via_cpuinfo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When /proc/cpuinfo reports AMD vendor, use AMD raw specs."""
    (tmp_path / "cpu").mkdir()
    monkeypatch.setattr(
        "harness.micro_arch._read_cpuinfo_vendor", lambda: "authenticamd",
    )
    result = detect_platform_event_specs(tmp_path)
    assert result.name == "x86-amd"


def test_detect_platform_intel_vendor_via_cpuinfo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When /proc/cpuinfo reports Intel vendor, use Intel specs."""
    (tmp_path / "cpu").mkdir()
    monkeypatch.setattr(
        "harness.micro_arch._read_cpuinfo_vendor", lambda: "genuineintel",
    )
    result = detect_platform_event_specs(tmp_path)
    assert result.name == "x86-intel"


def test_detect_platform_falls_back_to_generic(tmp_path: Path) -> None:
    result = detect_platform_event_specs(tmp_path)
    assert result.name == "generic"


# ==========================================================================
# Perf count parsing
# ==========================================================================

def test_parse_perf_count_normal() -> None:
    assert _parse_perf_count("12345") == 12345.0
    assert _parse_perf_count("  67890  ") == 67890.0
    assert _parse_perf_count("1 234 567") == 1234567.0


def test_parse_perf_count_not_counted() -> None:
    assert _parse_perf_count("<not counted>") is None
    assert _parse_perf_count("<not supported>") is None


def test_parse_perf_count_empty() -> None:
    assert _parse_perf_count("") is None


# ==========================================================================
# Perf stat output parsing
# ==========================================================================

def test_parse_perf_stat_output_armv8_cache() -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)
    cache_group = groups[0]

    stderr = "\n".join([
        "1000000,,r11,1.00,100.00",
        "800000,,r08,1.00,100.00",
        "300000,,r04,1.00,100.00",
        "15000,,r03,1.00,100.00",
        "200000,,r14,1.00,100.00",
        "2000,,r01,1.00,100.00",
    ])
    counts = _parse_perf_stat_output(stderr, cache_group.event_specs)
    assert counts is not None
    assert counts["r11"] == 1000000.0
    assert counts["r04"] == 300000.0
    assert counts["r03"] == 15000.0


def test_parse_perf_stat_output_x86_cache() -> None:
    groups = _build_groups(X86_INTEL_SPECS)
    cache_group = groups[0]  # L1D only on x86

    stderr = "\n".join([
        "1000000,,cycles,1.00,100.00",
        "800000,,instructions,1.00,100.00",
        "300000,,L1-dcache-loads,1.00,100.00",
        "15000,,L1-dcache-load-misses,1.00,100.00",
    ])
    counts = _parse_perf_stat_output(stderr, cache_group.event_specs)
    assert counts is not None
    assert counts["L1-dcache-loads"] == 300000.0


def test_parse_perf_stat_output_partial() -> None:
    """Partial data (fewer events than requested) now returns available events."""
    groups = _build_groups(ARMV8_RAW_SPECS)
    cache_group = groups[0]
    stderr = "1000000,,r11,1.00,100.00\n800000,,r08,1.00,100.00\n"
    counts = _parse_perf_stat_output(stderr, cache_group.event_specs)
    # Tolerant parser returns the two events that are present.
    assert counts is not None
    assert counts["r11"] == 1000000.0
    assert counts["r08"] == 800000.0
    assert "r04" not in counts


def test_parse_perf_stat_output_skips_not_supported() -> None:
    """Events with <not supported> are silently skipped, not fatal.

    On x86 the cache and icache groups are separate, so L1I unsupported
    events only affect the icache group, not the cache (L1D) group.
    """
    groups = _build_groups(X86_INTEL_SPECS)
    cache_group = groups[0]   # L1D only
    icache_group = groups[1]  # L1I only
    assert icache_group.name == "icache"

    # Cache group: L1I events are NOT in its spec → they are silently ignored
    perf_output_cache = "\n".join([
        "406089353,,cycles,67487307277,100.00,,",
        "352432799,,instructions,67486672980,100.00,0.87,insn per cycle",
        "91900281,,L1-dcache-loads,67486274573,100.00,,",
        "3593604,,L1-dcache-load-misses,67485604558,100.00,3.91,of all L1-dcache accesses",
    ])
    counts_cache = _parse_perf_stat_output(perf_output_cache, cache_group.event_specs)
    assert counts_cache is not None
    assert counts_cache["cycles"] == 406089353.0
    assert counts_cache["instructions"] == 352432799.0
    assert counts_cache["L1-dcache-loads"] == 91900281.0
    assert counts_cache["L1-dcache-load-misses"] == 3593604.0

    # ICache group: L1I access <not supported>, but miss works
    perf_output_icache = "\n".join([
        "406089353,,cycles,67487307277,100.00,,",
        "352432799,,instructions,67486672980,100.00,0.87,insn per cycle",
        "<not supported>,,L1-icache-loads,0,100.00,,",
        "12054829,,L1-icache-load-misses,67484973460,100.00,,",
    ])
    counts_icache = _parse_perf_stat_output(perf_output_icache, icache_group.event_specs)
    assert counts_icache is not None
    assert "L1-icache-loads" not in counts_icache  # <not supported> → skipped
    assert counts_icache["L1-icache-load-misses"] == 12054829.0


def test_parse_perf_stat_output_amd_cache() -> None:
    """AMD cache group: L1D only (L1I is in icache group)."""
    groups = _build_groups(X86_AMD_SPECS)
    cache_group = groups[0]

    stderr = "\n".join([
        "1000000,,cycles,1.00,100.00",
        "800000,,instructions,1.00,100.00",
        "300000,,L1-dcache-loads,1.00,100.00",
        "15000,,L1-dcache-load-misses,1.00,100.00",
    ])
    counts = _parse_perf_stat_output(stderr, cache_group.event_specs)
    assert counts is not None
    assert counts["L1-dcache-loads"] == 300000.0


def test_parse_perf_stat_output_amd_icache() -> None:
    """AMD icache group: L1I raw events in dedicated perf stat call."""
    groups = _build_groups(X86_AMD_SPECS)
    icache_group = groups[1]
    assert icache_group.name == "icache"

    stderr = "\n".join([
        "1000000,,cycles,1.00,100.00",
        "800000,,instructions,1.00,100.00",
        "200000,,cpu/event=0x80/,1.00,100.00",
        "2000,,cpu/event=0x81/,1.00,100.00",
    ])
    counts = _parse_perf_stat_output(stderr, icache_group.event_specs)
    assert counts is not None
    assert counts["cpu/event=0x80/"] == 200000.0
    assert counts["cpu/event=0x81/"] == 2000.0



def test_parse_perf_stat_output_no_false_match_on_description() -> None:
    """Event spec matching must NOT trigger on description substrings.

    ``cycles`` appearing in ``insn per cycle`` must not overwrite a
    previously parsed cycles count.
    """
    groups = _build_groups(X86_INTEL_SPECS)
    branch_group = groups[2]  # x86: cache=0, icache=1, branch=2

    perf_output = "\n".join([
        "181736527,,cycles,67391124157,100.00,,",
        "154318703,,instructions,67390555837,100.00,0.85,insn per cycle",
        "28809073,,branch-instructions,67390186097,100.00,,",
        "1109123,,branch-misses,67389588591,100.00,3.85,of all branches",
    ])
    counts = _parse_perf_stat_output(perf_output, branch_group.event_specs)
    assert counts is not None
    # ``cycles`` must be from the cycles line, NOT overwritten by the
    # instructions line (where ``insn per cycle`` contains ``cycles``).
    assert counts["cycles"] == 181736527.0
    assert counts["instructions"] == 154318703.0
    assert counts["branch-instructions"] == 28809073.0
    assert counts["branch-misses"] == 1109123.0


def test_parse_perf_stat_output_branch() -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)
    branch_group = groups[1]

    stderr = "\n".join([
        "1000000,,r11,1.00,100.00",
        "800000,,r08,1.00,100.00",
        "50000,,r12,1.00,100.00",
        "1500,,r10,1.00,100.00",
        "12000,,r19,1.00,100.00",
    ])
    counts = _parse_perf_stat_output(stderr, branch_group.event_specs)
    assert counts is not None
    assert counts["r12"] == 50000.0
    assert counts["r10"] == 1500.0


# ==========================================================================
# Derived metrics
# ==========================================================================

def test_derive_metrics_cache_armv8() -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)
    cache_group = groups[0]

    counts = {
        "r11": 1_000_000.0, "r08": 800_000.0,
        "r04": 300_000.0, "r03": 15_000.0,
        "r14": 200_000.0, "r01": 2_000.0,
    }
    reading = _derive_metrics(cache_group, counts, 1.0, "armv8-raw")

    assert reading.available is True
    assert reading.group == "cache"
    assert reading.platform == "armv8-raw"
    assert reading.ipc == pytest.approx(0.8)
    assert reading.l1d_hit_rate == pytest.approx(0.95)
    assert reading.l1i_hit_rate == pytest.approx(0.99)


def test_derive_metrics_cache_x86() -> None:
    """x86 cache group only has L1D (L1I is in separate icache group)."""
    groups = _build_groups(X86_INTEL_SPECS)
    cache_group = groups[0]

    counts = {
        "cycles": 1_000_000.0, "instructions": 800_000.0,
        "L1-dcache-loads": 300_000.0, "L1-dcache-load-misses": 15_000.0,
    }
    reading = _derive_metrics(cache_group, counts, 1.0, "x86-intel")

    assert reading.available is True
    assert reading.platform == "x86-intel"
    assert reading.l1d_hit_rate == pytest.approx(0.95)
    assert reading.l1i_hit_rate is None  # L1I is in icache group on x86


def test_derive_metrics_icache_x86() -> None:
    """x86 icache group computes L1I hit rate from dedicated perf stat call."""
    groups = _build_groups(X86_INTEL_SPECS)
    icache_group = groups[1]  # second group is icache on x86
    assert icache_group.name == "icache"

    counts = {
        "cycles": 1_000_000.0, "instructions": 800_000.0,
        "L1-icache-loads": 200_000.0, "L1-icache-load-misses": 2_000.0,
    }
    reading = _derive_metrics(icache_group, counts, 1.0, "x86-intel")

    assert reading.available is True
    assert reading.platform == "x86-intel"
    assert reading.ipc == pytest.approx(0.8)
    assert reading.l1i_hit_rate == pytest.approx(0.99)
    assert reading.l1d_hit_rate is None  # L1D is in cache group


def test_derive_metrics_cache_amd() -> None:
    """AMD cache group: L1D only (L1I is in icache group)."""
    groups = _build_groups(X86_AMD_SPECS)
    cache_group = groups[0]

    counts = {
        "cycles": 1_000_000.0, "instructions": 800_000.0,
        "L1-dcache-loads": 300_000.0, "L1-dcache-load-misses": 15_000.0,
    }
    reading = _derive_metrics(cache_group, counts, 1.0, "x86-amd")

    assert reading.available is True
    assert reading.platform == "x86-amd"
    assert reading.l1d_hit_rate == pytest.approx(0.95)
    assert reading.l1i_hit_rate is None  # L1I is in icache group


def test_derive_metrics_icache_amd() -> None:
    """AMD icache group: L1I via raw events."""
    groups = _build_groups(X86_AMD_SPECS)
    icache_group = groups[1]
    assert icache_group.name == "icache"

    counts = {
        "cycles": 1_000_000.0, "instructions": 800_000.0,
        "cpu/event=0x80/": 200_000.0, "cpu/event=0x81/": 2_000.0,
    }
    reading = _derive_metrics(icache_group, counts, 1.0, "x86-amd")

    assert reading.available is True
    assert reading.platform == "x86-amd"
    assert reading.l1i_hit_rate == pytest.approx(0.99)
    assert reading.l1d_hit_rate is None


def test_derive_metrics_cache_amd_l1i_unavailable() -> None:
    """When L1I access event is <not supported> but miss is available,
    fall back to using instructions as a proxy for L1I accesses."""
    groups = _build_groups(X86_AMD_SPECS)
    icache_group = groups[1]
    assert icache_group.name == "icache"

    # Simulate: L1I access event missing, miss event present (real AMD scenario)
    counts = {
        "cycles": 1_000_000.0, "instructions": 800_000.0,
        # cpu/event=0x80/ missing → l1i_access = None
        "cpu/event=0x81/": 2_000.0,
    }
    reading = _derive_metrics(icache_group, counts, 1.0, "x86-amd")

    assert reading.available is True
    # Fallback: instructions used as proxy for L1I accesses
    # l1i_hit_rate = 1.0 - (2000 / 800000) ≈ 0.9975
    assert reading.l1i_hit_rate == pytest.approx(0.9975)


def test_derive_metrics_cache_generic() -> None:
    """Generic: cache-references/cache-misses are LLC, not L1.
    Hit rate is still computed but is an LLC approximation."""
    groups = _build_groups(GENERIC_SPECS)
    cache_group = groups[0]

    counts = {
        "cycles": 1_000_000.0, "instructions": 800_000.0,
        "cache-references": 300_000.0, "cache-misses": 60_000.0,
    }
    reading = _derive_metrics(cache_group, counts, 1.0, "generic")

    assert reading.available is True
    # L1D is derived from LLC counters on generic
    assert reading.l1d_hit_rate == pytest.approx(0.8)
    # L1I is None on generic (no event)
    assert reading.l1i_hit_rate is None


def test_derive_metrics_branch_armv8() -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)
    branch_group = groups[1]

    counts = {
        "r11": 1_000_000.0, "r08": 800_000.0,
        "r12": 50_000.0, "r10": 1_500.0,
        "r19": 12_000.0,
    }
    reading = _derive_metrics(branch_group, counts, 1.0, "armv8-raw")

    assert reading.branch_miss_rate == pytest.approx(0.03)
    assert reading.bus_access_per_s == pytest.approx(12_000.0)


def test_derive_metrics_branch_x86() -> None:
    groups = _build_groups(X86_INTEL_SPECS)
    branch_group = groups[2]  # x86: cache=0, icache=1, branch=2

    counts = {
        "cycles": 1_000_000.0, "instructions": 800_000.0,
        "branch-instructions": 50_000.0, "branch-misses": 1_500.0,
    }
    reading = _derive_metrics(branch_group, counts, 1.0, "x86-intel")

    assert reading.branch_miss_rate == pytest.approx(0.03)
    assert reading.bus_access_per_s is None  # x86 has no bus_access


def test_derive_metrics_zero_cycles() -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)
    counts = {s: 0.0 for s in groups[0].event_specs}
    reading = _derive_metrics(groups[0], counts, 1.0, "armv8-raw")
    assert reading.ipc is None
    assert reading.l1d_hit_rate is None


# ==========================================================================
# sample_core_events_once
# ==========================================================================

def test_sample_once_success_armv8(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)
    cache_group = groups[0]

    def fake_run(cmd, **kwargs):
        stderr = "\n".join([
            "1000000,,r11,1.00,100.00",
            "800000,,r08,1.00,100.00",
            "300000,,r04,1.00,100.00",
            "15000,,r03,1.00,100.00",
            "200000,,r14,1.00,100.00",
            "2000,,r01,1.00,100.00",
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.micro_arch.subprocess.run", fake_run)
    reading = sample_core_events_once(
        cache_group, interval_s=1.0, scope_args=["-a"],
        scope_kind="system_wide", platform_name="armv8-raw",
    )
    assert reading.available is True
    assert reading.group == "cache"
    assert reading.l1d_hit_rate == pytest.approx(0.95)


def test_sample_once_success_x86(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = _build_groups(X86_INTEL_SPECS)
    cache_group = groups[0]  # L1D only

    def fake_run(cmd, **kwargs):
        stderr = "\n".join([
            "1000000,,cycles,1.00,100.00",
            "800000,,instructions,1.00,100.00",
            "300000,,L1-dcache-loads,1.00,100.00",
            "15000,,L1-dcache-load-misses,1.00,100.00",
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.micro_arch.subprocess.run", fake_run)
    reading = sample_core_events_once(
        cache_group, interval_s=1.0, scope_args=["-a"],
        scope_kind="system_wide", platform_name="x86-intel",
    )
    assert reading.available is True
    assert reading.platform == "x86-intel"
    assert reading.l1d_hit_rate == pytest.approx(0.95)


def test_sample_once_success_amd(monkeypatch: pytest.MonkeyPatch) -> None:
    """AMD cache group: L1D only."""
    groups = _build_groups(X86_AMD_SPECS)
    cache_group = groups[0]

    def fake_run(cmd, **kwargs):
        stderr = "\n".join([
            "1000000,,cycles,1.00,100.00",
            "800000,,instructions,1.00,100.00",
            "300000,,L1-dcache-loads,1.00,100.00",
            "15000,,L1-dcache-load-misses,1.00,100.00",
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.micro_arch.subprocess.run", fake_run)
    reading = sample_core_events_once(
        cache_group, interval_s=1.0, scope_args=["-a"],
        scope_kind="system_wide", platform_name="x86-amd",
    )
    assert reading.available is True
    assert reading.platform == "x86-amd"
    assert reading.l1d_hit_rate == pytest.approx(0.95)


def test_sample_once_permission_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 255, stdout="",
            stderr="Error: No permission to enable event.\n",
        )
    monkeypatch.setattr("harness.micro_arch.subprocess.run", fake_run)

    reading = sample_core_events_once(
        groups[0], interval_s=1.0, scope_args=["-a"],
        scope_kind="system_wide", platform_name="armv8-raw",
    )
    assert reading.available is False
    assert reading.reason == "permission_denied"


def test_sample_once_perf_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = _build_groups(ARMV8_RAW_SPECS)

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("perf")
    monkeypatch.setattr("harness.micro_arch.subprocess.run", fake_run)

    reading = sample_core_events_once(
        groups[0], interval_s=1.0, scope_args=["-a"],
        scope_kind="system_wide", platform_name="armv8-raw",
    )
    assert reading.available is False
    assert reading.reason == "perf_missing"


# ==========================================================================
# Scoping
# ==========================================================================

def test_resolve_perf_scoping_prefers_cgroup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "harness.micro_arch._has_perf_cgroup_support", lambda _: True,
    )
    cgroup = tmp_path / "docker-abc.scope"
    cgroup.mkdir()
    scope, args = resolve_perf_scoping(
        perf_executable="perf", cgroup_path=cgroup, container_pid=1234,
    )
    assert scope == "cgroup"
    assert "--cgroup" in args


def test_resolve_perf_scoping_falls_back_to_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "harness.micro_arch._has_perf_cgroup_support", lambda _: False,
    )
    scope, args = resolve_perf_scoping(
        perf_executable="perf", container_pid=1234,
    )
    assert scope == "process"
    assert "1234" in args


def test_resolve_perf_scoping_falls_back_to_system_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "harness.micro_arch._has_perf_cgroup_support", lambda _: False,
    )
    scope, args = resolve_perf_scoping(perf_executable="perf")
    assert scope == "system_wide"
    assert "-a" in args


# ==========================================================================
# MicroArchCollector
# ==========================================================================

def test_collector_initial_state() -> None:
    collector = MicroArchCollector(interval_s=0.1)
    latest = collector.latest()
    assert isinstance(latest, MicroArchReading)
    assert latest.available is False
    assert latest.reason == "not_started"


def test_collector_latest_single_group_after_init() -> None:
    """After construction but before start, latest() returns not_started."""
    collector = MicroArchCollector(interval_s=0.1)
    reading = collector.latest("cache")
    assert reading.available is False
    assert reading.reason == "not_started"


def test_collector_with_explicit_specs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Collector with explicit _specs uses those specs."""
    (tmp_path / "armv8_pmuv3_0").mkdir()

    def fake_run(cmd, **kwargs):
        stderr = "\n".join([
            "1000000,,r11,1.00,100.00",
            "800000,,r08,1.00,100.00",
            "300000,,r04,1.00,100.00",
            "15000,,r03,1.00,100.00",
            "200000,,r14,1.00,100.00",
            "2000,,r01,1.00,100.00",
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.micro_arch.subprocess.run", fake_run)
    monkeypatch.setattr("harness.micro_arch.shutil.which", lambda _: "/usr/bin/perf")

    collector = MicroArchCollector(
        interval_s=0.01, event_source_root=tmp_path,
        _platform="linux", _specs=ARMV8_RAW_SPECS,
    )
    collector.start()
    threading.Event().wait(0.3)
    collector.stop()

    latest = collector.latest()
    assert isinstance(latest, dict)
    cache = latest.get("cache")
    assert cache is not None
    assert cache.available
    assert cache.platform == "armv8-raw"


def test_collector_non_linux() -> None:
    collector = MicroArchCollector(interval_s=0.01, _platform="darwin")
    collector.start()
    collector.stop()
    latest = collector.latest()
    assert isinstance(latest, dict)
    for reading in latest.values():
        assert reading.available is False
        assert reading.reason == "unsupported_platform"


def test_collector_alternates_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Collector cycles through cache -> branch -> cache."""
    (tmp_path / "armv8_pmuv3_0").mkdir()
    groups = _build_groups(ARMV8_RAW_SPECS)

    call_count = [0]

    def fake_run(cmd, **kwargs):
        call_count[0] += 1
        group_idx = (call_count[0] - 1) % 2
        g = groups[group_idx]
        lines = []
        for spec in g.event_specs:
            lines.append(f"1000000,,{spec},1.00,100.00")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="\n".join(lines))

    monkeypatch.setattr("harness.micro_arch.subprocess.run", fake_run)
    monkeypatch.setattr("harness.micro_arch.shutil.which", lambda _: "/usr/bin/perf")

    collector = MicroArchCollector(
        interval_s=0.01, event_source_root=tmp_path, _platform="linux",
    )
    collector.start()
    threading.Event().wait(0.5)
    collector.stop()

    latest = collector.latest()
    assert isinstance(latest, dict)
    cache = latest.get("cache")
    branch = latest.get("branch")
    assert cache is not None and branch is not None
    assert cache.available or branch.available


# ==========================================================================
# attach_micro_arch
# ==========================================================================

def test_attach_micro_arch_no_collector() -> None:
    reset_micro_arch_collector_for_tests()
    sample: dict[str, object] = {}
    attach_micro_arch(sample, interval_s=1.0)
    assert "micro_arch_available" in sample
    assert sample["micro_arch_available"] is False


# ==========================================================================
# get_micro_arch_collector singleton
# ==========================================================================

def test_get_collector_singleton() -> None:
    reset_micro_arch_collector_for_tests()
    c1 = get_micro_arch_collector(interval_s=1.0)
    c2 = get_micro_arch_collector(interval_s=1.0)
    assert c1 is c2


# ==========================================================================
# ARM DDRC memory bandwidth backend (unchanged from previous)
# ==========================================================================

def test_detect_arm_ddrc_backend_hisilicon(tmp_path: Path) -> None:
    (tmp_path / "hisi_ddrc0").mkdir()
    backend = _detect_arm_ddrc_backend(tmp_path)
    assert backend is not None
    assert backend.kind == "arm_ddrc"
    assert "hisi_ddrc0" in backend.read_specs[0]


def test_detect_arm_ddrc_backend_generic_arm(tmp_path: Path) -> None:
    (tmp_path / "arm_ddrc0").mkdir()
    backend = _detect_arm_ddrc_backend(tmp_path)
    assert backend is not None
    assert backend.kind == "arm_ddrc"


def test_detect_arm_ddrc_backend_with_named_events(tmp_path: Path) -> None:
    """When DDRC PMU has named events, _detect_arm_ddrc_backend skips it
    (it will be caught by _detect_explicit_byte_backend instead)."""
    device = tmp_path / "arm_ddrc0" / "events"
    device.mkdir(parents=True)
    (device / "read_bytes").write_text("event=0x00\n", encoding="utf-8")
    (device / "write_bytes").write_text("event=0x01\n", encoding="utf-8")

    # _detect_arm_ddrc_backend skips devices with named events
    backend = _detect_arm_ddrc_backend(tmp_path)
    assert backend is None

    # detect_perf_backend finds it via explicit_byte_events instead
    backend2 = detect_perf_backend(tmp_path)
    assert backend2 is not None
    assert backend2.kind == "explicit_byte_events"
    assert backend2.read_specs == ("arm_ddrc0/read_bytes/",)


def test_detect_arm_ddrc_backend_none(tmp_path: Path) -> None:
    assert _detect_arm_ddrc_backend(tmp_path) is None


def test_detect_perf_backend_prefers_intel_over_arm(tmp_path: Path) -> None:
    imc_events = tmp_path / "uncore_imc_0" / "events"
    imc_events.mkdir(parents=True)
    (imc_events / "cas_count_read").write_text("event=0x01\n", encoding="utf-8")
    (imc_events / "cas_count_write").write_text("event=0x02\n", encoding="utf-8")
    (tmp_path / "arm_ddrc0").mkdir()
    backend = detect_perf_backend(tmp_path)
    assert backend is not None
    assert backend.kind == "intel_imc_cas"


def test_detect_perf_backend_falls_back_to_arm_ddrc(tmp_path: Path) -> None:
    """When only arm_ddrc WITHOUT events/ exists, raw codes are used."""
    (tmp_path / "arm_ddrc0").mkdir()
    backend = detect_perf_backend(tmp_path)
    assert backend is not None
    assert backend.kind == "arm_ddrc"
    assert backend.source == "perf:arm-ddrc-raw"
