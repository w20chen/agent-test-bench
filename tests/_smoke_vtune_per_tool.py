"""Smoke test: backward compatibility of per-tool VTune coarse sampling."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    print(f"Test dir: {tmp}")

    # Test 1: Old trace — no per_tool_samples.jsonl, container-level coarse
    run_dir1 = tmp / "old_trace" / "attempt_1" / "vtune" / "pytest_20260629T120000_42"
    run_dir1.mkdir(parents=True)
    (run_dir1 / "window.json").write_text(json.dumps({
        "cmd": "pytest old.py -v", "ts_start": 1.0, "ts_end": 6.0,
        "returncode": 0,
    }))
    (run_dir1 / "summary.json").write_text(json.dumps({
        "cmd": "pytest old.py -v", "ts_start": 1.0, "ts_end": 6.0,
        "duration_s": 5.0, "returncode": 0, "n_samples": 10,
        "coarse_source": "container_cgroup",
    }))
    (run_dir1 / "coarse.json").write_text(json.dumps({
        "cpu_percent": {"min": 10, "max": 80, "avg": 40},
        "memory_mb": {"min": 200, "max": 400, "avg": 300},
        "disk_read_mb": {"delta": 5.0},
        "disk_write_mb": {"delta": 1.0},
        "net_rx_mb": {"delta": 0.1},
        "net_tx_mb": {"delta": 0.05},
        "context_switches": {"delta": 5000},
    }))

    # Test 2: New trace — per_tool_samples.jsonl present
    run_dir2 = tmp / "new_trace" / "attempt_1" / "vtune" / "pytest_20260629T120001_43"
    run_dir2.mkdir(parents=True)
    (run_dir2 / "window.json").write_text(json.dumps({
        "cmd": "pytest new.py -v", "ts_start": 1.0, "ts_end": 5.0,
        "returncode": 0,
    }))
    (run_dir2 / "summary.json").write_text(json.dumps({
        "cmd": "pytest new.py -v", "ts_start": 1.0, "ts_end": 5.0,
        "duration_s": 4.0, "returncode": 0, "n_samples": 8,
        "coarse_source": "per_tool_proc",
    }))
    (run_dir2 / "coarse.json").write_text(json.dumps({
        "cpu_percent": {"min": 20, "max": 60, "avg": 35},
        "memory_mb": {"min": 300, "max": 350, "avg": 325},
        "disk_read_mb": {"delta": 2.0},
        "disk_write_mb": {"delta": 0.5},
        "net_rx_mb": {"delta": 0.1},
        "net_tx_mb": {"delta": 0.05},
        "context_switches": {"delta": 3000},
    }))

    script = Path(__file__).parents[1] / "scripts" / "analyze_vtune_aggregate.py"
    result = subprocess.run(
        [sys.executable, str(script), "--input", str(tmp)],
        capture_output=True, text=True, timeout=30,
    )
    print("STDERR:", result.stderr.strip())
    assert result.returncode == 0, f"Script failed: {result.returncode}"

    csv_path = tmp / "vtune_aggregate.csv"
    assert csv_path.exists(), "CSV not generated"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert len(lines) == 3, f"Expected 2 data rows, got {len(lines) - 1}"
    print(f"CSV: {len(lines) - 1} rows x {len(header)} cols")

    # Check coarse_source column exists and has correct values
    assert "coarse_source" in header, "coarse_source column missing"
    src_idx = header.index("coarse_source")
    sources = {line.split(",")[src_idx] for line in lines[1:]}
    assert sources == {"container_cgroup", "per_tool_proc"}, f"Unexpected sources: {sources}"
    print(f"  coarse_source values: {sources}")

    # Check coarse metrics are present
    for key in ["coarse_cpu_percent_avg", "coarse_memory_mb_avg",
                "coarse_disk_read_mb_delta", "coarse_context_switches_delta"]:
        assert key in header, f"{key} column missing"

    # Check the summary.txt is generated
    summary_path = tmp / "vtune_aggregate_summary.txt"
    assert summary_path.exists(), "summary.txt not generated"

    # Check topn.txt is generated
    topn_path = tmp / "vtune_aggregate_topn.txt"
    assert topn_path.exists(), "topn.txt not generated"

    print("PASS: All checks passed — backward compatibility verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
