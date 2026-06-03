"""Audit MoE phase denominators for recorded routing artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "asb-agent-attention-moe-phase-denominator-audit"
VOLUME_NAME = "asb-terminal-recordings"
VOLUME_ROOT = Path("/data")
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "curated14"
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "agent_attention_a1_a5_revised_20260510"

LOCAL_FILE = Path(__file__).resolve()
LOCAL_RECODING_FIGURES = (
    LOCAL_FILE.parents[2] / "scripts" / "recoding_figures"
    if len(LOCAL_FILE.parents) > 2
    else Path("/opt/recoding_figures")
)
RECODING_FIGURES = (
    LOCAL_RECODING_FIGURES
    if LOCAL_RECODING_FIGURES.exists()
    else Path("/opt/recoding_figures")
)

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy").add_local_dir(
    RECODING_FIGURES,
    remote_path="/opt/recoding_figures",
    copy=True,
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=4,
    memory=16384,
    timeout=60 * 30,
)
def run_audit() -> dict[str, Any]:
    """Compute routing-record, token-row, and load denominators by phase."""
    sys.path.insert(0, "/opt/recoding_figures")
    from moe_phase_audit import compute_moe_phase_denominator_audit
    from recording_loader import load_iteration_records

    attempts = sorted(EXTRACT_DIR.glob("*/attempt_1"))
    records = load_iteration_records(attempts)
    payload = compute_moe_phase_denominator_audit(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "moe_phase_denominator_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    volume.commit()
    return payload


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_audit.remote(), indent=2))


if __name__ == "__main__":
    main()
