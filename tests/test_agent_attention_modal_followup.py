from __future__ import annotations

import pytest

from scripts.modal_workspace.agent_attention_modal_followup import (
    OUTPUT_PARENT,
    OUTPUT_PREFIX,
    _new_output_dir,
    _tar_path_for_output_dir,
)


def test_new_output_dir_uses_explicit_run_id() -> None:
    output_dir = _new_output_dir("unit_test_unique_id")

    assert output_dir == OUTPUT_PARENT / f"{OUTPUT_PREFIX}_unit_test_unique_id"


def test_new_output_dir_rejects_existing_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.modal_workspace.agent_attention_modal_followup.OUTPUT_PARENT",
        tmp_path,
    )
    existing = tmp_path / f"{OUTPUT_PREFIX}_collision"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        _new_output_dir("collision")


def test_new_output_dir_rejects_path_like_run_id() -> None:
    with pytest.raises(ValueError):
        _new_output_dir("bad/name")


def test_tar_path_preserves_dotted_run_id() -> None:
    left = OUTPUT_PARENT / f"{OUTPUT_PREFIX}_abc.v1"
    right = OUTPUT_PARENT / f"{OUTPUT_PREFIX}_abc.v2"

    assert _tar_path_for_output_dir(left).name == f"{OUTPUT_PREFIX}_abc.v1.tar.zst"
    assert _tar_path_for_output_dir(right).name == f"{OUTPUT_PREFIX}_abc.v2.tar.zst"
    assert _tar_path_for_output_dir(left) != _tar_path_for_output_dir(right)
