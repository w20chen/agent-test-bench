"""Lifecycle wrapper for Huawei Kunpeng ``ksys collect``."""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KsysSession:
    """One optional system-wide ksys collection process."""

    process: subprocess.Popen[bytes]

    @classmethod
    def start(
        cls,
        *,
        output_dir: Path,
        log_dir: Path,
    ) -> "KsysSession | None":
        """Start ksys, returning ``None`` when the executable is unavailable."""
        ksys_bin = shutil.which("ksys")
        if ksys_bin is None:
            logger.warning(
                "ksys not found on $PATH; no ksys metrics will be collected"
            )
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (log_dir / "ksys_stdout.txt").open("wb")
        stderr = (log_dir / "ksys_stderr.txt").open("wb")
        try:
            process = subprocess.Popen(
                [ksys_bin, "collect", "-o", str(output_dir.resolve())],
                stdout=stdout,
                stderr=stderr,
                cwd=str(output_dir.resolve()),
            )
        except Exception:
            logger.warning("ksys collect failed to start", exc_info=True)
            return None
        finally:
            stdout.close()
            stderr.close()
        logger.info("ksys collect started (pid=%d)", process.pid)
        return cls(process=process)

    def stop(self) -> None:
        """Stop ksys gracefully, then force termination if necessary."""
        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("ksys did not exit after SIGINT; killing it")
            self.process.kill()
            self.process.wait(timeout=10)
        except Exception:
            logger.warning("ksys stop failed", exc_info=True)
