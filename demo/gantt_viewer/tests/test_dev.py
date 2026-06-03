"""Tests for the backend CLI launcher helpers."""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from demo.gantt_viewer.backend import dev


def _noop(*args: object, **kwargs: object) -> None:
    return None


def test_dev_main_exports_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("groups: []\n", encoding="utf-8")

    dist_path = tmp_path / "dist"
    dist_path.mkdir()
    (dist_path / "index.html").write_text("<html></html>", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(app: str, **kwargs: object) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(dev, "FRONTEND_DIST_PATH", dist_path)
    monkeypatch.setattr(dev, "schedule_browser_open", _noop)
    monkeypatch.setattr(dev.uvicorn, "run", fake_run)
    monkeypatch.delenv("GANTT_VIEWER_CONFIG", raising=False)
    monkeypatch.delenv("GANTT_VIEWER_DEV", raising=False)

    dev.main(["--config", str(config_path), "--port", "9999", "--no-browser"])

    assert os.environ["GANTT_VIEWER_CONFIG"] == str(config_path.resolve())
    assert os.environ["GANTT_VIEWER_DEV"] == "0"
    assert captured["app"] == "demo.gantt_viewer.backend.app:create_app"
    assert captured["factory"] is True
    assert captured["port"] == 9999


def test_dev_mode_spawns_vite(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("groups: []\n", encoding="utf-8")

    captured: dict[str, object] = {}
    fake_process = SimpleNamespace(poll=lambda: None)

    def _fake_wait(*args: object, **kwargs: object) -> None:
        captured["waited"] = True

    monkeypatch.setattr(dev, "_spawn_vite_dev_server", lambda: fake_process)
    monkeypatch.setattr(dev, "_wait_for_vite_startup", _fake_wait)
    monkeypatch.setattr(dev, "_terminate_process", lambda process: captured.setdefault("terminated", process))
    monkeypatch.setattr(dev, "schedule_browser_open", _noop)
    monkeypatch.setattr(
        dev.uvicorn,
        "run",
        lambda app, **kwargs: captured.update({"app": app, **kwargs}),
    )

    dev.main(["--dev", "--config", str(config_path), "--no-browser"])

    assert os.environ["GANTT_VIEWER_DEV"] == "1"
    assert captured["waited"] is True
    assert captured["terminated"] is fake_process
    assert captured["reload"] is False


def test_prod_mode_without_dist_exits_nonzero(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("groups: []\n", encoding="utf-8")

    monkeypatch.setattr(dev, "FRONTEND_DIST_PATH", tmp_path / "does-not-exist")
    monkeypatch.setattr(dev, "schedule_browser_open", _noop)
    monkeypatch.setattr(dev.uvicorn, "run", _noop)

    with pytest.raises(SystemExit) as exc_info:
        dev.main(["--config", str(config_path), "--no-browser"])

    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "gantt-viewer-build" in err


def test_wait_for_vite_startup_succeeds_when_socket_open(monkeypatch) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    _, port = listener.getsockname()
    try:
        dev._wait_for_vite_startup("127.0.0.1", port, timeout_s=2.0)
    finally:
        listener.close()


def test_wait_for_vite_startup_raises_on_timeout() -> None:
    closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    closed.bind(("127.0.0.1", 0))
    _, port = closed.getsockname()
    closed.close()

    with pytest.raises(RuntimeError) as exc_info:
        dev._wait_for_vite_startup("127.0.0.1", port, timeout_s=0.5)
    assert "did not become ready" in str(exc_info.value)


def test_schedule_browser_open_respects_no_browser(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(dev.webbrowser, "open", lambda url: calls.append(url))

    args = argparse.Namespace(no_browser=True)
    dev.schedule_browser_open(args, host="127.0.0.1", port=8765)
    assert calls == []

    args_open = argparse.Namespace(no_browser=False)
    timer_calls: list[tuple[float, object]] = []

    class _FakeTimer:
        def __init__(self, interval: float, fn: object) -> None:
            timer_calls.append((interval, fn))
            self._fn = fn

        def start(self) -> None:
            assert callable(self._fn)
            self._fn()

    monkeypatch.setattr(dev.threading, "Timer", _FakeTimer)
    dev.schedule_browser_open(args_open, host="127.0.0.1", port=8765)
    assert calls == ["http://127.0.0.1:8765"]
    assert len(timer_calls) == 1
    assert timer_calls[0][0] == 1.5
