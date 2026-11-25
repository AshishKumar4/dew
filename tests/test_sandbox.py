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
from episode_support import IDENTITY, action, build, environment, trained_batch, verify
from process_support import running
from tool_policy import (
    ANSWER_NINE,
    ANSWER_SIXTEEN,
    CALL_FOUR,
    CALL_THREE,
    EOS,
    GROUPS,
    NINE,
    PROMPT,
    START,
    WRONG,
)

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


def test_a_measured_parent_measures_the_launcher_it_starts(tmp_path, monkeypatch):
    """A parent measured by coverage (its startup in `COVERAGE_PROCESS_START`) launches a command whose
    launcher is measured too, its lines up to the exec saved, although `-I` and the launcher's minimal
    environment would start no coverage of their own."""
    import coverage

    from dew.rl import _sandbox_exec
    from dew.rl.sandbox import SandboxLimits, launched

    config = tmp_path / "coverage.ini"
    config.write_text(f"[run]\nparallel = true\ndata_file = {tmp_path / 'data'}\n"
                      f"include = {_sandbox_exec.__file__}\n")
    monkeypatch.setenv("COVERAGE_PROCESS_START", str(config))
    with launched(("true",), SandboxLimits(), str(tmp_path)) as process:
        assert process.wait(timeout=60) == 0
    combined = coverage.Coverage(config_file=str(config))
    combined.combine()
    measured = set(combined.get_data().lines(_sandbox_exec.__file__) or ())
    # Coverage starts in main's first statement and is saved in its last before the exec: the statements
    # between, the first after its start and the save itself, are measured.
    statements = sorted({line for _, _, line in _sandbox_exec.main.__code__.co_lines() if line is not None})
    assert {statements[2], statements[-2]} <= measured


def test_a_launched_worker_holds_no_signal_its_starting_threads_end_sends(tmp_path):
    """A sandbox worker (`launched`) holds no parent-death signal (`PR_GET_PDEATHSIG` 0, which an exec
    keeps): its lifetime is its group's lifeline's (`owned_group`), where the signal its launcher installed
    fires when the thread that forked it ends, `_cleanup.start`'s short-lived starter, and killed a healthy
    worker at its start whenever that thread outlived the install (the models review)."""
    from dew.rl.sandbox import SandboxLimits, launched

    reported = ("import ctypes; held = ctypes.c_int(); "
                "ctypes.CDLL(None).prctl(2, ctypes.byref(held), 0, 0, 0); print(held.value, flush=True)")
    with launched((sys.executable, "-c", reported), SandboxLimits(), str(tmp_path)) as process:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == b"0"

def killing_its_parent_ends(script: Path) -> list[int]:
    """Run `script` as a parent and SIGKILL it once it prints a running process's id: that process, where it
    still runs after, none where its group outlives no owner."""
    root = Path(__file__).resolve().parents[1]
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")})
    assert parent.stdout is not None
    process = int(parent.stdout.readline())
    assert running(process)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    return wait_gone([process])

def test_parent_death_kills_the_worker(tmp_path):
    """A SIGKILLed parent takes its worker with it through its group's lifeline (`owned_group`)."""
    script = tmp_path / "parent.py"
    script.write_text(
        "import json, os, sys, time\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from episode_support import environment, IDENTITY\n"
        "with environment()(IDENTITY) as session:\n"
        "    print(json.loads(session.reset().detail)[0], flush=True)\n"
        "    time.sleep(3600)\n")
    assert killing_its_parent_ends(script) == []


def test_parent_death_kills_what_the_worker_forked(tmp_path):
    """A SIGKILLed parent takes its worker's own children with it: the worker's group outlives no owner
    (`dew._cleanup.owned_group`'s lifeline), where the death signal, which Linux clears in a fork, took the
    worker alone and left what it forked running (the audit's row 12172)."""
    script = tmp_path / "parent.py"
    forking = ("import subprocess, sys, time; child = subprocess.Popen(['sleep', '3600']); "
               "print(child.pid, flush=True); time.sleep(3600)")
    script.write_text(
        "import sys, time\n"
        "from dew.rl.sandbox import SandboxLimits, launched\n"
        f"with launched((sys.executable, '-c', {forking!r}), SandboxLimits(), {str(tmp_path)!r}) as worker:\n"
        "    print(worker.stdout.readline().decode().strip(), flush=True)\n"
        "    time.sleep(3600)\n")
    assert killing_its_parent_ends(script) == []


def test_parent_death_kills_the_container_client(tmp_path):
    """A SIGKILLed parent takes the container runtime's client it ran a program through with it
    (`ContainerRunner`): the client's group outlives no owner (`dew._cleanup.start`'s lifeline), where a
    session of its own alone left it running, and the container it would make with it (row 12772). The
    client here a runtime that records its id and waits."""
    pidfile, runtime, script = tmp_path / "client", tmp_path / "runtime", tmp_path / "parent.py"
    runtime.write_text(f"#!/bin/sh\necho $$ > {pidfile}\nexec sleep 3600\n")
    runtime.chmod(0o755)
    script.write_text(
        "import pathlib, threading, time\n"
        "from dew.rl.sandbox import ContainerRunner, Program, SandboxLimits\n"
        f"runner = ContainerRunner('image', runtime={str(runtime)!r})\n"
        "program = Program({'main.py': 'print(5)'}, ('python', 'main.py'))\n"
        "limits = SandboxLimits(wall_seconds=3600)\n"
        "threading.Thread(target=runner, args=(program, limits), daemon=True).start()\n"
        f"path = pathlib.Path({str(pidfile)!r})\n"
        "while not (path.exists() and path.read_text().strip()):\n"
        "    time.sleep(0.05)\n"
        "print(path.read_text().strip(), flush=True)\n"
        "time.sleep(3600)\n")
    assert killing_its_parent_ends(script) == []


def test_episodes_collect_through_subprocess_workers():
    """The rollout collects a cohort from real workers and scores it."""
    harness = environment()
    episodes = []
    trainer, collector = build(harness, record=episodes.append)
    state = trainer.initial_state()
    batch = trained_batch(collector, state, trainer.objective)
    assert len(episodes) == GROUPS
    assert {episode.status for episode in episodes} <= {EpisodeStatus.COMPLETED, EpisodeStatus.TRUNCATED}
    assert {verify(episode) for episode in episodes} <= {-1., 0., 1.}
    assert batch["input_ids"].shape[1] == PROMPT + collector.max_new_tokens
    assert np.isfinite(batch["advantages"]).all()


@pytest.mark.parametrize("mode", ["chat", "compact"])
def test_a_harness_s_requests_and_compaction_reach_each_call_s_transcript(mode):
    """A chat harness's worker parses each action into a request and answers with messages, compact's
    tool result compacting the context. Every collected episode reads back from its record whole, and
    each call's transcript is the conversation so far: the task and the tool call's request with the
    tool's result before the answer, or the compacted summary alone, its response its own request."""
    from sandbox_square_worker import chat_turn

    from dew.objectives.rl import session_of
    from dew.objectives.rl.records import episode_from_record, record_json

    episodes = []
    trainer, collector = build(environment(mode), record=episodes.append)
    trained_batch(collector, trainer.initial_state(), trainer.objective)
    assert len(episodes) == GROUPS
    for episode in episodes:
        assert episode_from_record(json.loads(record_json(episode))) == episode
        value = {CALL_THREE: 3, CALL_FOUR: 4}[episode.transitions[0].action.tokens[0]]
        given = {ANSWER_NINE: 9, ANSWER_SIXTEEN: 16, WRONG: 8}[episode.transitions[1].action.tokens[0]]
        (task,), tool = chat_turn(mode)["messages"], chat_turn(mode, value)
        answer = chat_turn(mode, given=given)
        history = tuple(tool["messages"]) if mode == "compact" else (task, tool["request"], *tool["messages"])
        transcripts = [(call.transcript.messages, call.transcript.response)
                       for call in session_of(episode, group="g").calls]
        assert transcripts == [((task,), tool["request"]), (history, answer["request"])]


def test_cancellation_releases_the_real_worker():
    from concurrent.futures import CancelledError

    pids: list[int] = []
    cancellation = CancelledError("owner cancelled the tool episode")
    with pytest.raises(CancelledError) as caught, environment()(IDENTITY) as session:
        pids = json.loads(session.reset().detail)
        raise cancellation
    assert caught.value is cancellation
    assert wait_gone(pids) == []
