from pathlib import Path

from agents.openclaw.eval.runner import SWEBenchRunner


class _DummyProvider:
    def get_default_model(self) -> str:
        return "dummy-model"


def test_swebench_runner_passes_exec_path_append_into_session_runner(
    tmp_path: Path,
) -> None:
    runner = SWEBenchRunner(
        provider=_DummyProvider(),
        workspace_base=tmp_path / "ws",
        benchmark_slug="swe-rebench",
        exec_path_append="/tmp/a:/tmp/b",
    )

    assert runner._session_runner.exec_config.path_append == "/tmp/a:/tmp/b"
