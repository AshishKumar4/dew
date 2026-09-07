"""Completed tools survive SIGKILL; pending actions and policy provenance survive too."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
import numpy as np
import pytest

from dew.objectives.rl import EpisodeFailure, EpisodeJournal, RecoverableEnvironment
from test_sandbox import IDENTITY, action, environment
from test_tool_episodes import CALL_THREE, EOS, NINE, START, build, collect

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("episode_recovery_worker.py")


def test_subprocess_snapshot_restores_tool_state_without_reset():
    with environment()(IDENTITY) as first:
        assert isinstance(first, RecoverableEnvironment)
        first.reset()
        observation = first.step(action((START,), CALL_THREE, EOS))
        snapshot = first.get_state()
    with environment()(IDENTITY) as second:
        assert isinstance(second, RecoverableEnvironment)
        second.set_state(snapshot)
        final = second.step(action(observation.context, 5, EOS))
        assert observation.context[-1] == NINE
        assert json.loads(final.detail) == {"answer": 9, "expected": 9}


def test_journal_refuses_a_different_policy_at_identical_clocks(tmp_path):
    trainer, rollout = build(environment(), journal=EpisodeJournal(str(tmp_path)))
    state = trainer.initial_state()
    complete = collect(rollout, state)
    assert collect(rollout, state) == complete
    changed = replace(state, params=jax.tree.map(lambda value: value + .001, state.params))
    with pytest.raises(ValueError, match="same policy"):
        collect(rollout, changed)


def test_journal_requires_a_recoverable_environment(tmp_path):
    trainer, rollout = build(journal=EpisodeJournal(str(tmp_path)))
    with pytest.raises(EpisodeFailure, match="get_state/set_state"):
        collect(rollout, trainer.initial_state())


def launch(directory, mode):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=2", "OMP_NUM_THREADS": "1"}
    return subprocess.Popen([sys.executable, str(WORKER), str(directory), mode],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def finish(process):
    try:
        output, _ = process.communicate(timeout=90)
    except BaseException:
        process.kill()
        process.wait()
        raise
    assert process.returncode == 0, output


def test_sigkill_resumes_completed_turns_and_produces_the_same_update(tmp_path):
    """Two real CPU-device runs: committed tools once, actions and update exact."""
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    finish(launch(baseline, "ok"))
    process = launch(resumed, "pause_final")
    try:
        deadline = time.monotonic() + 60
        while not (resumed / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert (resumed / "ready").exists(), process.communicate(timeout=1)[0] if process.poll() is not None else "no pending tool"
    finally:
        process.kill()
        process.wait()
    first_calls = (resumed / "calls.jsonl").read_text().splitlines()
    assert len(first_calls) == 8
    finish(launch(resumed, "ok"))
    assert (resumed / "calls.jsonl").read_text() == (baseline / "calls.jsonl").read_text()
    np.testing.assert_array_equal(np.load(resumed / "parameters.npy"), np.load(baseline / "parameters.npy"))
    original = json.loads((baseline / "episodes.json").read_text())
    restored = json.loads((resumed / "episodes.json").read_text())
    for before, after in zip(original, restored, strict=True):
        assert before["identity"] == after["identity"]
        assert before["reward"] == after["reward"]
        for old, new in zip(before["transitions"], after["transitions"], strict=True):
            old["action"].pop("_binding_id")
            new["action"].pop("_binding_id")
            assert old == new
