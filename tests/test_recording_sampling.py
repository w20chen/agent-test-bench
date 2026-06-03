from __future__ import annotations

import pytest

from serving.recording.recording import query_sampling_seed, select_query_positions


def test_query_sampling_seed_is_stable_per_call() -> None:
    assert query_sampling_seed(17, 3) == "17:3"


def test_select_query_positions_uses_seeded_stratified_jitter() -> None:
    first = select_query_positions(8, 4, seed="0:0")
    second = select_query_positions(8, 4, seed="0:1")

    assert first == [0, 3, 4, 7]
    assert second == [0, 2, 4, 7]
    assert first != second


def test_select_query_positions_keeps_one_row_per_window() -> None:
    positions = select_query_positions(81, 8, seed="recording-test")

    assert len(positions) == 8
    assert len(set(positions)) == 8
    for idx, position in enumerate(positions):
        start = (idx * 81) // 8
        stop = ((idx + 1) * 81) // 8
        assert start <= position < stop


def test_select_query_positions_preserves_short_queries() -> None:
    assert select_query_positions(4, 80, seed="ignored") == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("query_len", "max_queries"),
    [(0, 4), (4, 0), (-1, 4), (4, -1)],
)
def test_select_query_positions_rejects_invalid_bounds(
    query_len: int,
    max_queries: int,
) -> None:
    with pytest.raises(ValueError):
        select_query_positions(query_len, max_queries)
