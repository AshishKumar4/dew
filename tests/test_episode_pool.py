"""Real pool agreement, variable episode lengths and global-update parity."""

import sys
from pathlib import Path

import numpy as np
import pytest
from process_support import run_pool, start_process

WORKER = Path(__file__).with_name("tool_episode_worker.py")
pytestmark = pytest.mark.mesh


def run(directory, processes, mode="ok"):
    def start(mode, out, processes, process_id, coordinator):
        return start_process(
            [sys.executable, str(WORKER), str(process_id), str(processes), coordinator, str(out), mode],
            devices=2 // processes)
    return run_pool(mode, directory, processes, start=start, timeout=120)


def test_two_process_variable_turns_match_single_process_actions_and_update(tmp_path):
    single = run(tmp_path / "single", 1)[0]
    pool = run(tmp_path / "pool", 2)
    pooled = [episode for report in pool for episode in report["episodes"]]
    assert pooled == single["episodes"]
    # Rank 0 names the cohort once for the pool: every episode and action on
    # both ranks carries the one binding.
    assert pool[0]["bindings"] == pool[1]["bindings"] and len(pool[0]["bindings"]) == 1
    assert {len(episode["transitions"]) for episode in pooled} == {1, 2}
    assert all(report["opened"] == report["closed"] == 4 for report in pool)
    assert all(report["updates"] == 1 and report["error"] is None for report in pool)
    for suffix in (".npz", ".batch.npz"):
        with np.load(tmp_path / "single" / f"process0{suffix}") as reference:
            for rank in range(2):
                with np.load(tmp_path / "pool" / f"process{rank}{suffix}") as actual:
                    assert set(actual.files) == set(reference.files)
                    for name in reference.files:
                        # Both layouts use two global devices and the same row
                        # partition; this checks exact params and actual draws.
                        np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)


def test_rank_local_environment_error_aborts_peers_and_releases_all_slots(tmp_path):
    reports = run(tmp_path / "pool", 2, "error")
    assert all(report["opened"] == report["closed"] == 4 for report in reports)
    assert all(report["updates"] is None and report["error"] for report in reports)
    assert all(episode["reward"] is None for report in reports for episode in report["episodes"])
    assert all("rank-local tool failed" in report["error"] for report in reports)


def test_inconsistent_cohort_settings_fail_before_creating_environments(tmp_path):
    reports = run(tmp_path / "pool", 2, "mismatch")
    assert all(report["opened"] == report["closed"] == 0 for report in reports)
    assert all(report["updates"] is None and "must agree" in report["error"] for report in reports)
