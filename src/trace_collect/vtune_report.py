"""Host-side helpers for the ``--vtune`` in-container pytest profiling feature.

Isolated so the collection path only needs two thin call sites:

- ``vtune_container_run_args`` -> extra ``docker run`` args that mount VTune
  into the task container, grant PMU capabilities, and signal the in-container
  ``ExecTool`` (via env vars) to wrap pytest with VTune.
- ``finalize_vtune`` -> after the agent finishes, slice the already-collected
  ``ContainerStatsSampler`` samples to each pytest window and emit JSON.

VTune has no first-class JSON report, so ``fine.json`` is derived by parsing
``vtune -report summary -format csv``.
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
    """Parse ``vtune -report summary -format csv`` into a flat metric dict."""
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
        if "," not in line:
            continue
        key, _, val = line.partition(",")
        key = key.strip()
        if key:
            metrics[key] = val.strip()
    return metrics


def _window_samples(
    samples: list[dict[str, Any]], ts_start: float, ts_end: float
) -> list[dict[str, Any]]:
    return [s for s in samples if ts_start <= float(s.get("epoch", 0.0)) <= ts_end]


def finalize_vtune(
    out_dir: Path,
    samples: list[dict[str, Any]],
    *,
    coarse: bool,
    fine: bool,
) -> None:
    """Emit ``summary.json`` (+ optional ``coarse.json``/``fine.json``) per run.

    Each ``pytest_*`` directory was created in-container with a ``window.json``
    (cmd/ts_start/ts_end) and a raw ``result/`` VTune result dir.
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
        win = _window_samples(samples, ts_start, ts_end)
        summary = summarize_samples(win) if win else {}

        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "cmd": window.get("cmd"),
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "duration_s": ts_end - ts_start,
                    "returncode": window.get("returncode"),
                    "n_samples": len(win),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if coarse:
            (run_dir / "coarse.json").write_text(
                json.dumps({k: summary.get(k) for k in _COARSE_KEYS}, indent=2) + "\n",
                encoding="utf-8",
            )
        if fine:
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
