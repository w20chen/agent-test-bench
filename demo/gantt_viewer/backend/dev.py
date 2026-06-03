"""CLI entrypoint for the dynamic Gantt viewer backend."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from demo.gantt_viewer.backend.app import FRONTEND_DIST_PATH

VITE_DEV_PORT = 5173


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 CLI parser for the future demo server."""
    parser = argparse.ArgumentParser(
        prog="python -m trace_collect.cli gantt-serve",
        description="Launch the dynamic Gantt viewer server.",
    )
    parser.add_argument(
        "--config",
        default="demo/gantt_viewer/configs/example.yaml",
        help="Path to the viewer discovery config.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run the backend in frontend-dev mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Backend listen port.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Backend listen host.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser on startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Launch the backend server."""
    args = build_parser().parse_args(argv)
    os.environ["GANTT_VIEWER_CONFIG"] = str(Path(args.config).resolve())
    os.environ["GANTT_VIEWER_DEV"] = "1" if args.dev else "0"

    if args.dev:
        vite_process = _spawn_vite_dev_server()
        try:
            try:
                _wait_for_vite_startup("127.0.0.1", VITE_DEV_PORT)
            except RuntimeError as exc:
                _print_vite_stderr_tail(vite_process)
                print(f"ERROR: {exc}", file=sys.stderr)
                raise
            schedule_browser_open(args, host=args.host, port=VITE_DEV_PORT)
            uvicorn.run(
                "demo.gantt_viewer.backend.app:create_app",
                factory=True,
                host=args.host,
                port=args.port,
                reload=False,
            )
        finally:
            _terminate_process(vite_process)
        return

    if not FRONTEND_DIST_PATH.exists():
        print(
            "ERROR: frontend/dist is missing — run 'make gantt-viewer-build' "
            f"(expected: {FRONTEND_DIST_PATH / 'index.html'})",
            file=sys.stderr,
        )
        sys.exit(1)

    schedule_browser_open(args, host=args.host, port=args.port)
    uvicorn.run(
        "demo.gantt_viewer.backend.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=False,
    )


def schedule_browser_open(args: argparse.Namespace, *, host: str, port: int) -> None:
    """Schedule a browser open 1.5s after launch unless --no-browser was passed."""
    if args.no_browser:
        return
    url = f"http://{host}:{port}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()


def _spawn_vite_dev_server() -> subprocess.Popen[str]:
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    return subprocess.Popen(
        [
            "npm",
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(VITE_DEV_PORT),
        ],
        cwd=frontend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _print_vite_stderr_tail(process: subprocess.Popen[str]) -> None:
    """Drain vite's stderr (best-effort) so users see startup errors."""
    if process.stderr is None:
        return
    try:
        tail = process.stderr.read() or ""
    except Exception:
        return
    if tail:
        print("vite stderr tail:", file=sys.stderr)
        print(tail, file=sys.stderr)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _wait_for_vite_startup(
    host: str = "127.0.0.1",
    port: int = VITE_DEV_PORT,
    timeout_s: float = 20.0,
) -> None:
    """Probe the Vite dev server port until it accepts connections or times out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(
        f"vite dev server at {host}:{port} did not become ready within {timeout_s:.1f}s"
    )
