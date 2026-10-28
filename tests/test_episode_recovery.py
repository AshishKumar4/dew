"""Completed tools survive SIGKILL; pending actions and policy provenance survive too."""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import jax
import numpy as np
import pytest
from episode_support import (
    CALL_THREE,
    EOS,
    GROUPS,
    IDENTITY,
    NINE,
    START,
    Harness,
    SquareSession,
    action,
    build,
    cohort_snapshot,
    collect,
    environment,
)

from dew.objectives.rl import EpisodeJournal, RecoverableEnvironment, Transcript, session_of

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("episode_recovery_worker.py")


def test_journal_refuses_a_different_policy_at_identical_clocks(tmp_path):
    trainer, collector = build(environment(), journal=EpisodeJournal(str(tmp_path)))
    state = trainer.initial_state()
    complete = collect(collector, state)
    assert collect(collector, state) == complete
    changed = replace(state, variables=jax.tree.map(lambda value: value + .001, state.variables))
    with pytest.raises(ValueError):
        collect(collector, changed)


def test_a_cooldown_branchs_cohorts_are_its_own_beside_the_runs_at_the_same_clocks(tmp_path):
    """A cooldown branch draws at the run's steps and keys under its own weights: its cohorts are its
    own (`Snapshot.branch`), so the run, drawing at those clocks after it, is not refused as another
    policy, and a resumed branch reads its own turns back."""
    from dew.objectives.rl.sessions import Task

    trainer, collector = build(environment(), journal=EpisodeJournal(str(tmp_path)))
    state = trainer.initial_state()
    decayed = replace(state, variables=jax.tree.map(lambda value: value + .001, state.variables))
    snapshot = cohort_snapshot(decayed, 23)
    branch = collector.episodes(replace(snapshot, branch="cooldown-4"), [Task("31")], GROUPS)
    assert collector.episodes(replace(snapshot, branch="cooldown-4"), [Task("31")], GROUPS) == branch
    run = collect(collector, state)
    assert collect(collector, state) == run
    with pytest.raises(ValueError):
        collect(collector, decayed)


def test_journal_tells_apart_policies_whose_weights_are_equal_bytes_of_another_dtype():
    """A policy's weights in float8_e4m3fn and the same bytes read as float8_e4m3fnuz are two
    policies, which NumPy names by one '<V1' code: the journal's policy fingerprint keys each by its
    named dtype, so a recovery under the other refuses, while the same weights match again."""
    import ml_dtypes

    from dew.objectives.rl.journal import policy_digest

    bits = np.arange(64, dtype=np.uint8).reshape(8, 8)
    weights = {"params": {"kernel": bits.view(ml_dtypes.float8_e4m3fn)}}
    other = {"params": {"kernel": bits.view(ml_dtypes.float8_e4m3fnuz)}}
    again = {"params": {"kernel": bits.copy().view(ml_dtypes.float8_e4m3fn)}}
    assert policy_digest(weights) == policy_digest(again)
    assert policy_digest(weights) != policy_digest(other)


class Saying(SquareSession):
    """The square tool, whose observations also say the chat messages they added, as read-only mappings."""

    def reset(self):
        return replace(super().reset(), messages=(MappingProxyType({"role": "user", "content": "square"}),))

    def step(self, action):
        seen = super().step(action)
        return replace(seen, messages=(MappingProxyType({"role": "tool", "content": seen.detail}),))

    def get_state(self):
        return json.dumps({"answer": self.answer}).encode()

    def set_state(self, state):
        self.answer = json.loads(state)["answer"]


class Sayings(Harness):
    @contextmanager
    def __call__(self, identity):
        yield Saying(identity, self)


def test_a_journaled_episode_keeps_the_messages_its_environment_said(tmp_path):
    """Messages an environment gives in any mapping are kept as JSON: a journaled collection writes
    them and reads the same episodes back, and each call's transcript lists those before it."""
    trainer, collector = build(Sayings(), journal=EpisodeJournal(str(tmp_path)))
    state = trainer.initial_state()
    episodes = collect(collector, state)
    assert collect(collector, state) == episodes
    for episode in episodes:
        said = [{"role": "user", "content": "square"},
                *({"role": "tool", "content": turn.observation.detail} for turn in episode.transitions)]
        calls = session_of(episode, group="0").calls
        assert [call.transcript for call in calls] == [Transcript(tuple(said[:turn + 1]))
                                                       for turn in range(len(calls))]


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
        assert (resumed / "ready").exists(), (
            process.communicate(timeout=1)[0] if process.poll() is not None else "no pending tool"
        )
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
        assert before["transitions"] == after["transitions"]

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
