"""Host-side helpers for the ``--vtune`` in-container pytest profiling feature.

Isolated so the collection path only needs two thin call sites:

- ``vtune_container_run_args`` -> extra ``docker run`` args that mount VTune
  into the task container, grant PMU capabilities, and signal the in-container
  ``ExecTool`` (via env vars) to wrap pytest with VTune.
- ``finalize_vtune`` -> after the agent finishes, slice the already-collected
  ``ContainerStatsSampler`` samples to each pytest window and emit JSON.

VTune has no first-class JSON report, so ``fine.json`` is derived by parsing
``vtune -report summary -format csv``.

When ``--vtune`` is active, each pytest invocation additionally produces a
``per_tool_samples.jsonl`` file (written by the in-container proc-tree sampler
in ``shell.py``).  These samples are scoped to the exact pytest process tree
via ``/proc/<pid>``, so concurrent pytest invocations do not pollute each
other's coarse metrics.  ``finalize_vtune`` prefers per-tool samples and
falls back to container-level ``ContainerStatsSampler`` samples when the
file is absent (backward-compatible with traces collected before this
feature was added).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from harness.container_stats_sampler import summarize_samples

#: Coarse system-metric keys copied out of ``summarize_samples``.
_COARSE_KEYS = (
    "cpu_percent",
    "memory_mb",
    "disk_read_mb",
    "disk_write_mb",
    "net_rx_mb",
    "net_tx_mb",
    "context_switches",
)
#: Perf microarch keys merged into ``fine.json`` alongside the VTune TMA.
_FINE_PERF_KEYS = ("ipc", "l1i_hit_rate", "branch_miss_rate")


def _resolve_vtune() -> tuple[str, str]:
    """Return ``(vtune_bin_abspath, vtune_root_dir)`` or raise.

    No silent fallback: ``--vtune`` is meaningless without VTune installed.
    """
    vtune_bin = os.environ.get("VTUNE_BIN") or shutil.which("vtune")
    if not vtune_bin:
        raise RuntimeError(
            "--vtune requires Intel VTune: set VTUNE_BIN to the vtune binary "
            "or put `vtune` on PATH (source the oneAPI setvars.sh)."
        )
    vtune_bin = str(Path(vtune_bin).resolve())
    root = os.environ.get("VTUNE_ROOT", "/opt/intel/oneapi")
    return vtune_bin, root


def vtune_container_run_args(
    out_dir: Path, *, tools: list[str] | None = None
) -> list[str]:
    """Extra ``docker run`` args enabling in-container VTune profiling.

    The host VTune install is bind-mounted read-only at its native path so the
    same absolute ``VTUNE_BIN`` resolves inside the container. ``out_dir`` lives
    under the bind-mounted attempt directory, so results land on the host too.

    Args:
        out_dir: Directory for per-pytest-window VTune results.
        tools: Exec classifier category slugs to profile (e.g. ``["pytest",
               "make"]``).  Defaults to ``["pytest"]`` when omitted.
    """
    vtune_bin, root = _resolve_vtune()
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_tools = tools or ["exec-pytest"]
    return [
        "--cap-add", "PERFMON",
        "--cap-add", "SYS_ADMIN",
        "-v", f"{root}:{root}:ro",
        "-e", "VTUNE_PROFILE=1",
        "-e", f"VTUNE_BIN={vtune_bin}",
        "-e", f"VTUNE_OUT={out_dir.resolve()}",
        "-e", f"VTUNE_TOOLS={','.join(resolved_tools)}",
    ]


def _vtune_tma(result_dir: Path) -> dict[str, Any]:
    """Parse ``vtune -report summary -format csv`` into a flat metric dict.

    VTune ``-report summary`` CSV has three columns::

        Hierarchy Level, Metric Name, Metric Value
        0, Collection and Platform Info,
        1, CPU,
        ...
        <empty>, Elapsed Time, 5.123
        <empty>, CPI Rate, 0.756
        ...

    We use the second column as key and the third as value, skipping the
    header line and any rows without a non-empty metric name.
    """
    vtune_bin, _ = _resolve_vtune()
    try:
        proc = subprocess.run(
            [vtune_bin, "-report", "summary", "-r", str(result_dir),
             "-format", "csv", "-csv-delimiter", ","],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"vtune report failed: {exc}"}
    if proc.returncode != 0:
        return {"error": (proc.stderr.strip()[:500] or "vtune report nonzero exit")}
    metrics: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split into at most 3 parts: level, name, value
        parts = line.split(",", 2)
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        if not name or name == "Metric Name":
            continue  # skip header and empty-name rows
        value = parts[2].strip() if len(parts) > 2 else ""
        # Convert percentages and numbers where possible
        if value.endswith("%"):
            try:
                metrics[name] = float(value[:-1])
            except ValueError:
                metrics[name] = value
        else:
            try:
                metrics[name] = float(value.replace(",", ""))
            except ValueError:
                metrics[name] = value
    return metrics


def _window_samples(
    samples: list[dict[str, Any]], ts_start: float, ts_end: float
) -> list[dict[str, Any]]:
    return [s for s in samples if ts_start <= float(s.get("epoch", 0.0)) <= ts_end]


# ---------------------------------------------------------------------------
# Per-tool proc-tree sample support (in-container /proc sampler)
# ---------------------------------------------------------------------------

def _read_per_tool_samples(path: Path) -> list[dict[str, Any]] | None:
    """Read a ``per_tool_samples.jsonl`` file, or return None."""
    if not path.exists():
        return None
    samples: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return None
    return samples if samples else None


def _convert_per_tool_samples(
    raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert raw per-tool /proc samples to ContainerStatsSampler format.

    Raw fields (cumulative / instantaneous from ``/proc``):
      ``cpu_jiffies``, ``rss_kb``, ``disk_read_bytes``,
      ``disk_write_bytes``, ``context_switches``

    Converts to:
      ``cpu_percent`` (string like ``"45.2%"``), ``mem_usage`` (string like
      ``"500.0MB"``), ``disk_read_bytes``, ``disk_write_bytes``,
      ``context_switches`` — all compatible with ``summarize_samples()``.

    CPU% is computed from jiffies deltas between consecutive samples.
    The first sample is dropped (no baseline for delta).
    """
    # Resolve jiffies-per-second from the host kernel.
    try:
        import os as _os
        _clk_tck = _os.sysconf(_os.sysconf_names["SC_CLK_TCK"])
    except (KeyError, ValueError, AttributeError):
        _clk_tck = 100  # standard on x86 Linux

    ncpus = os.cpu_count() or 1
    out: list[dict[str, Any]] = []

    prev_jiffies: int | None = None
    prev_time: float | None = None

    for s in raw:
        sample: dict[str, Any] = {"epoch": s.get("epoch", 0.0)}

        # CPU% from jiffies delta
        curr_jiffies = s.get("cpu_jiffies", 0)
        curr_time = s.get("epoch", 0.0)
        if (
            prev_jiffies is not None
            and prev_time is not None
            and isinstance(curr_jiffies, (int, float))
        ):
            delta_j = float(curr_jiffies) - float(prev_jiffies)
            delta_t = curr_time - prev_time
            if delta_t > 0 and delta_j >= 0:
                cpu_pct = (delta_j / _clk_tck) / delta_t / ncpus * 100
                sample["cpu_percent"] = f"{cpu_pct:.2f}%"

        # Memory: RSS in KB → MB string
        rss_kb = s.get("rss_kb", 0)
        if isinstance(rss_kb, (int, float)) and rss_kb > 0:
            sample["mem_usage"] = f"{rss_kb / 1024:.1f}MB"

        # Disk I/O (cumulative counters — summarize_samples uses delta)
        for key in ("disk_read_bytes", "disk_write_bytes"):
            val = s.get(key)
            if isinstance(val, (int, float)):
                sample[key] = int(val)

        # Context switches (cumulative)
        ctxt = s.get("context_switches")
        if isinstance(ctxt, (int, float)):
            sample["context_switches"] = int(ctxt)

        out.append(sample)
        # Always advance the baseline, even when the first sample(s)
        # have no delta yet — otherwise prev_jiffies stays None forever
        # and cpu_percent is never computed.
        if isinstance(curr_jiffies, (int, float)):
            prev_jiffies = int(curr_jiffies)
            prev_time = curr_time

    return out


def finalize_vtune(
    out_dir: Path,
    samples: list[dict[str, Any]],
) -> None:
    """Emit ``summary.json`` + ``coarse.json`` + ``fine.json`` per run.

    Each ``pytest_*`` directory was created in-container with a ``window.json``
    (cmd/ts_start/ts_end), a ``per_tool_samples.jsonl`` (in-container proc-tree
    sampler), and a raw ``result/`` VTune result dir.  When per-tool samples are
    present they are preferred for coarse metrics; otherwise the container-level
    ``ContainerStatsSampler`` samples are time-sliced as a fallback.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return
    for run_dir in sorted(out_dir.glob("pytest_*")):
        window_path = run_dir / "window.json"
        if not window_path.exists():
            continue
        try:
            window = json.loads(window_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ts_start = float(window.get("ts_start", 0.0))
        ts_end = float(window.get("ts_end", ts_start))

        # Prefer per-tool proc-tree samples (accurate, no concurrency
        # pollution) over container-level cgroup samples.
        coarse_source: str
        per_tool_raw = _read_per_tool_samples(run_dir / "per_tool_samples.jsonl")
        if per_tool_raw:
            win = _convert_per_tool_samples(per_tool_raw)
            coarse_source = "per_tool_proc"
        else:
            win = _window_samples(samples, ts_start, ts_end)
            coarse_source = "container_cgroup"

        summary = summarize_samples(win) if win else {}
        summary["_coarse_source"] = coarse_source

        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "cmd": window.get("cmd"),
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "duration_s": ts_end - ts_start,
                    "returncode": window.get("returncode"),
                    "n_samples": len(win),
                    "coarse_source": coarse_source,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "coarse.json").write_text(
            json.dumps({k: summary.get(k) for k in _COARSE_KEYS}, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "fine.json").write_text(
            json.dumps(
                {
                    "vtune_tma": _vtune_tma(run_dir / "result"),
                    "perf": {k: summary.get(k) for k in _FINE_PERF_KEYS},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
