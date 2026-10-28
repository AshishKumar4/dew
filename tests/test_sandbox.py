"""SubprocessEnvironment: real workers, real limits, real cleanup."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from episode_support import (
    CALL_THREE,
    EOS,
    GROUPS,
    IDENTITY,
    NINE,
    PROMPT,
    START,
    action,
    build,
    environment,
    trained_batch,
    verify,
)
from process_support import running

from dew.objectives.rl import EpisodeStatus


def wait_gone(pids, seconds=5.):
    deadline = time.monotonic() + seconds
    while any(running(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.02)
    return [pid for pid in pids if running(pid)]


def test_worker_round_trip_and_process_group_cleanup():
    """The worker and a forked descendant both end with the session."""
    with environment("fork")(IDENTITY) as session:
        initial = session.reset()
        assert initial.context == (START,) and initial.status == EpisodeStatus.RUNNING
        pids = json.loads(initial.detail)
        assert len(pids) == 2 and all(running(pid) for pid in pids)
        after_call = session.step(action(initial.context, CALL_THREE, EOS))
        assert after_call.context == (START, CALL_THREE, EOS, NINE)
        final = session.step(action(after_call.context, 5, EOS))
        assert final.status == EpisodeStatus.COMPLETED
        assert json.loads(final.detail) == {"answer": 9, "expected": 9}
    assert wait_gone(pids) == []


def test_parent_death_kills_the_worker(tmp_path):
    """A SIGKILLed parent takes its worker with it through the death signal."""
    script = tmp_path / "parent.py"
    script.write_text(
        "import json, os, sys, time\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from episode_support import environment, IDENTITY\n"
        "with environment()(IDENTITY) as session:\n"
        "    print(json.loads(session.reset().detail)[0], flush=True)\n"
        "    time.sleep(3600)\n")
    root = Path(__file__).resolve().parents[1]
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")})
    assert parent.stdout is not None
    worker = int(parent.stdout.readline())
    assert running(worker)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    assert wait_gone([worker]) == []


def test_episodes_collect_through_subprocess_workers():
    """The rollout collects a cohort from real workers and scores it."""
    harness = environment()
    episodes = []
    trainer, collector = build(harness, record=episodes.append)
    state = trainer.initial_state()
    batch = trained_batch(collector, state)
    assert len(episodes) == GROUPS
    assert {episode.status for episode in episodes} <= {EpisodeStatus.COMPLETED, EpisodeStatus.TRUNCATED}
    assert {verify(episode) for episode in episodes} <= {-1., 0., 1.}
    assert batch["input_ids"].shape[1] == PROMPT + collector.max_new_tokens
    assert np.isfinite(batch["advantages"]).all()


def test_cancellation_releases_the_real_worker():
    from concurrent.futures import CancelledError

    pids: list[int] = []
    cancellation = CancelledError("owner cancelled the tool episode")
    with pytest.raises(CancelledError) as caught, environment()(IDENTITY) as session:
        pids = json.loads(session.reset().detail)
        raise cancellation
    assert caught.value is cancellation
    assert wait_gone(pids) == []
