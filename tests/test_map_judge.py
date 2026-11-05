"""A site's work sharded across map tasks and pooled by a judge (tests/map_judge.py), on payloads of its own:
check 1's draws are tested beside the gate (tests/test_reference_error.py)."""

import pytest
from check1_wire import DRAWS, POOL, RECORDS
from map_judge import PoolRefused, drawing, pooled_units, record_units


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
    assert not drawing()
    assert not list(tmp_path.iterdir())


def test_a_map_task_draws_and_a_judge_does_not(monkeypatch, tmp_path):
    """Only a map task (DRAWS) is drawing, so a site leaves its judge-only work to the judge (POOL)."""
    monkeypatch.setenv(DRAWS, "0:2")
    assert drawing()
    monkeypatch.delenv(DRAWS)
    monkeypatch.setenv(POOL, str(tmp_path))
    assert not drawing()


def test_a_map_task_records_how_long_it_took_to_reach_its_units_and_to_draw_them(monkeypatch, tmp_path):
    """Beside its record, a map task writes the seconds from its process's start to its units' first (its
    fixed cost: imports, collection, the site's setup and compiles) and those its units took, so a run's map
    tasks say what of each was paid before any draw (Main, 2026-11-02: measure before cutting it)."""
    import json
    import time

    monkeypatch.setenv(DRAWS, "0:2")
    monkeypatch.setenv(RECORDS, str(tmp_path))

    def slow(first, last):
        time.sleep(0.3)
        return list(range(first, last))

    assert record_units("timed", slow)
    [seconds] = [json.loads(path.read_text()) for path in tmp_path.glob("*.seconds.json")]
    assert seconds["start"] == 0 and seconds["stop"] == 2
    assert seconds["draws"] >= 0.3 and 0 < seconds["before"] < time.time()
    monkeypatch.delenv(DRAWS)
    monkeypatch.setenv(POOL, str(tmp_path))
    assert [payload for _, _, payload in pooled_units("timed", 2)] == [[0, 1]]


def test_a_map_tasks_draw_seconds_are_the_time_its_units_took_whatever_the_wall_clock_did(monkeypatch,
                                                                                          tmp_path):
    """The seconds a map task's units took are measured on a clock that only moves forward: a wall clock set
    back an hour while it draws (an NTP step) leaves them the time the draws took."""
    import json
    import time

    monkeypatch.setenv(DRAWS, "0:2")
    monkeypatch.setenv(RECORDS, str(tmp_path))
    wall = time.time

    def stepped(first, last):
        time.sleep(0.2)
        monkeypatch.setattr(time, "time", lambda: wall() - 3600)
        return list(range(first, last))

    assert record_units("stepped", stepped)
    [seconds] = [json.loads(path.read_text()) for path in tmp_path.glob("*.seconds.json")]
    assert seconds["draws"] >= 0.2
