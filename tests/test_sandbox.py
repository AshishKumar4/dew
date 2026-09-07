"""SubprocessEnvironment: real workers, real limits, real cleanup."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import jax
import numpy as np
import pytest

from dew.objectives.rl import EpisodeStatus, SandboxLimits, SubprocessEnvironment
from dew.objectives.rl.episodes import Action, EpisodeId
from dew.objectives.rl.sandbox import _ProcessEnvironment
from test_tool_episodes import (
    CALL_THREE, EOS, GROUPS, NINE, PROMPT, SAMPLING, START, build, collect, verify,
)

WORKER = Path(__file__).with_name("sandbox_square_worker.py")
IDENTITY = EpisodeId(task=3, attempt=0, sample=0, seed=(1, 2))


def environment(mode="ok", **limits):
    return SubprocessEnvironment((sys.executable, str(WORKER), mode), SandboxLimits(**limits))


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_gone(pids, seconds=5.):
    deadline = time.monotonic() + seconds
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.02)
    return [pid for pid in pids if alive(pid)]


def action(context, *tokens):
    return Action(tuple(context), tokens, (-.1,) * len(tokens), (0.,) * len(tokens),
                  True, 0, SAMPLING)


def test_worker_round_trip_and_process_group_cleanup():
    """The worker and a forked descendant both end with the session."""
    with environment("fork")(IDENTITY) as session:
        initial = session.reset()
        assert initial.context == (START,) and initial.status == EpisodeStatus.RUNNING
        pids = json.loads(initial.detail)
        assert len(pids) == 2 and all(alive(pid) for pid in pids)
        after_call = session.step(action(initial.context, CALL_THREE, EOS))
        assert after_call.context == (START, CALL_THREE, EOS, NINE)
        final = session.step(action(after_call.context, 5, EOS))
        assert final.status == EpisodeStatus.COMPLETED
        assert json.loads(final.detail) == {"answer": 9, "expected": 9}
    assert wait_gone(pids) == []


def test_wall_time_limit_kills_a_hung_worker():
    started = time.monotonic()
    factory = environment("hang", wall_seconds=.5)
    with factory(IDENTITY) as session:
        with pytest.raises(TimeoutError, match="wall_seconds"):
            session.reset()
        assert isinstance(session, _ProcessEnvironment)
        pid = session.process.pid
    assert time.monotonic() - started < 5
    assert wait_gone([pid]) == []


def test_memory_limit_ends_the_worker_with_its_diagnostic():
    with environment("allocate")(IDENTITY) as session:
        with pytest.raises(ChildProcessError, match="MemoryError"):
            session.reset()


def test_worker_exit_reports_its_code_and_stderr():
    with environment("crash")(IDENTITY) as session:
        with pytest.raises(ChildProcessError, match="code 3.*worker boom"):
            session.reset()


def test_malformed_reply_is_refused():
    with environment("garbage")(IDENTITY) as session:
        with pytest.raises(ValueError, match="malformed JSON"):
            session.reset()


def test_parent_death_kills_the_worker(tmp_path):
    """A SIGKILLed parent takes its worker with it through the death signal."""
    script = tmp_path / "parent.py"
    script.write_text(
        "import json, os, sys, time\n"
        "sys.path.insert(0, %r)\n"
        "from test_sandbox import environment, IDENTITY\n"
        "with environment()(IDENTITY) as session:\n"
        "    print(json.loads(session.reset().detail)[0], flush=True)\n"
        "    time.sleep(3600)\n" % str(Path(__file__).parent))
    root = Path(__file__).resolve().parents[1]
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")})
    assert parent.stdout is not None
    worker = int(parent.stdout.readline())
    assert alive(worker)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    assert wait_gone([worker]) == []


@pytest.mark.parametrize("limits", [{"wall_seconds": 0.}, {"cpu_seconds": 0}, {"memory_bytes": 1.5}])
def test_limits_must_be_positive(limits):
    with pytest.raises(ValueError):
        SandboxLimits(**limits)


def test_command_must_be_argv():
    with pytest.raises(ValueError, match="argv"):
        SubprocessEnvironment(())


def test_episodes_collect_through_subprocess_workers():
    """The rollout collects a cohort from real workers and scores it."""
    harness = environment()
    trainer, rollout = build(harness)
    state = trainer.initial_state()
    episodes = collect(rollout, state)
    assert len(episodes) == GROUPS
    assert {episode.status for episode in episodes} <= {EpisodeStatus.COMPLETED, EpisodeStatus.TRUNCATED}
    assert {verify(episode) for episode in episodes} <= {-1., 0., 1.}
    batch = rollout.project(episodes)
    assert batch["input_ids"].shape[1] == PROMPT + rollout.max_new_tokens
    assert np.isfinite(batch["advantages"]).all()


def test_cpu_limit_kills_a_busy_worker():
    with environment("cpu", cpu_seconds=1, wall_seconds=10.)(IDENTITY) as session:
        with pytest.raises(ChildProcessError, match="code -9"):
            session.reset()


def test_cancellation_releases_the_real_worker():
    from concurrent.futures import CancelledError

    pids: list[int] = []
    cancellation = CancelledError("owner cancelled the tool episode")
    with pytest.raises(CancelledError) as caught:
        with environment()(IDENTITY) as session:
            pids = json.loads(session.reset().detail)
            raise cancellation
    assert caught.value is cancellation
    assert wait_gone(pids) == []
