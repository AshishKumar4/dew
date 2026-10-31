"""A site's work sharded across map tasks and pooled by a judge (tests/map_judge.py), on payloads of its own:
check 1's draws are tested beside the gate (tests/test_reference_error.py)."""

import pytest
from map_judge import DRAWS, POOL, RECORDS, PoolRefused, pooled_units, record_units


def recorded(monkeypatch, tmp_path, ranges, label="units"):
    """`label`'s units recorded by one map task a range, each unit's payload its own number squared."""
    for start, stop in ranges:
        monkeypatch.setenv(DRAWS, f"{start}:{stop}")
        monkeypatch.setenv(RECORDS, str(tmp_path / f"task {start}"))
        assert record_units(label, lambda first, last: [unit * unit for unit in range(first, last)])
    monkeypatch.delenv(DRAWS)
    monkeypatch.delenv(RECORDS)
    monkeypatch.setenv(POOL, str(tmp_path))


def test_a_judge_pools_every_map_tasks_units_in_order_once(monkeypatch, tmp_path):
    recorded(monkeypatch, tmp_path, [(4, 9), (0, 4), (4, 9)])
    pieces = pooled_units("units", 9)
    assert [(start, stop) for start, stop, _ in pieces] == [(0, 4), (4, 9)]
    assert [square for _, _, payload in pieces for square in payload] == [unit * unit for unit in range(9)]


def test_a_pool_short_of_its_units_or_with_a_gap_is_refused(monkeypatch, tmp_path):
    recorded(monkeypatch, tmp_path, [(0, 3), (5, 9)])
    with pytest.raises(PoolRefused) as gap:
        pooled_units("units")
    recorded(monkeypatch, tmp_path / "short", [(0, 3)])
    with pytest.raises(PoolRefused) as short:
        pooled_units("units", 9)
    assert (gap.value.held, short.value.held) == (3, 3)


def test_outside_a_map_task_or_a_judge_nothing_is_recorded_or_pooled(monkeypatch, tmp_path):
    for name in (DRAWS, RECORDS, POOL):
        monkeypatch.delenv(name, raising=False)
    assert not record_units("units", lambda first, last: [])
    assert pooled_units("units", 9) is None
    assert not list(tmp_path.iterdir())
