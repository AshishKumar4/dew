"""An OpenEnv environment served by OpenEnv's own server, played through `EnvironmentSource`.

The environment is a guessing game (`openenv_support`) served by `openenv`'s
`create_fastapi_app` over uvicorn, in-process or as a server process the test
kills, and each episode opens its own session through `GenericEnvClient`, as
a trainer reaching a remote environment does. The policy is a bigram model on
Dew's own server whose draws are its argmax: it guesses 3 after the prompt and
7 after the environment's hint.
"""

import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import jax
import jax.numpy as jnp
import numpy as np
import openenv_support
import pytest
import uvicorn
from openenv.core import GenericEnvClient
from tool_policy import ToolPolicy

from dew.inference import NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.interop.openenv import OpenEnv, Said
from dew.objectives.rl import EnvironmentSource, Status, Task, Transcript, pack
from dew.sampling import Sampling

VOCAB, EOS, PROMPT, HINT = 16, 15, 1, 2
"""Ids: the prompt's, the environment's hint's, and the model's EOS; a digit d is id d + 3."""


@pytest.fixture(scope="module")
def environment():
    """The guessing game's server, on a free local port, for the module's tests."""
    port = free_port()
    config = uvicorn.Config(openenv_support.app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    running = threading.Thread(target=server.run, daemon=True)
    running.start()
    while not server.started and running.is_alive():
        running.join(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    running.join()


def guesser():
    """Draws 3 (id 6) after the prompt and 7 (id 10) after a hint, each then EOS."""
    table = np.full((VOCAB, VOCAB), -1e4, np.float32)
    table[PROMPT, 6] = table[HINT, 10] = 0.0
    table[6, EOS] = table[10, EOS] = 0.0
    return table


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def games(url):
    """The guessing game at `url`, one session an episode, its target the task's id."""
    @contextmanager
    def session(task, identity):
        with GenericEnvClient(base_url=url).sync() as opened:
            yield opened

    def said(observation):
        return {"role": "user", "content": observation["hint"]}

    return OpenEnv(session, prompt=lambda observation: Said((PROMPT,), (said(observation),)),
                   reply=lambda observation: Said((HINT,), (said(observation),)),
                   action=lambda ids: {"number": ids[0] - 3},
                   options=lambda task, identity: {"target": int(task.id)})


def played(environment, task, samples=2):
    """`samples` episodes of `task` in `environment`, drawn on the guessing policy."""
    model = ToolPolicy(vocab_size=VOCAB, max_seq_len=64, logits=guesser)
    params = model.init(jax.random.key(0), jnp.ones((1, 1), jnp.int32))
    sampling = Sampling(temperature=1.0, eos_token_ids=EOS)
    server = NativeRolloutServer(Server.from_task(TextGeneration(model, params, None, sampling=sampling),
                                                  slots=4, capacity=64))
    episodes = EnvironmentSource(server, environment, lambda task, episode: 0.0, max_prompt_tokens=32,
                                 max_new_tokens=4, max_turns=4, workers=2)
    try:
        submitted = episodes.submit(Task(task), samples, version=0, key=jax.random.key(1))
        return [future.result() for future in submitted]
    finally:
        episodes.close()
        server.close()


def test_an_openenv_episode_draws_each_turn_on_the_exact_context_and_keeps_the_steps_rewards(environment):
    """Each episode opens its own OpenEnv session, resets it with its task's target, and steps it on
    the guess each turn drew. The second call's context is the first's, its drawn ids and the hint's
    id; each call keeps the reward the environment paid its step and the messages before it; the
    right guess completes the episode, and its calls pack into one chain."""
    rollouts = played(games(environment), "7")
    prompt = {"role": "user", "content": "guess a number from 0 to 9"}
    higher = {"role": "user", "content": "higher"}
    for rollout in rollouts:
        assert (rollout.status, rollout.reward) == (Status.COMPLETED, 0.0)
        first, second = rollout.calls
        assert (first.prompt, first.tokens) == ((PROMPT,), (6, EOS))
        assert (second.prompt, second.tokens) == ((PROMPT, 6, EOS, HINT), (10, EOS))
        assert (first.reward, second.reward) == (-0.1, 1.0)
        assert (first.transcript, second.transcript) == (Transcript((prompt,)), Transcript((prompt, higher)))
    batch = pack(rollouts, 8, credit="turn")
    assert batch["input_ids"].shape == (2, 8)


def test_an_openenv_reset_that_is_already_done_completes_its_episode_without_a_turn(environment):
    """A reset that answers done: the episode is complete before any draw or step."""
    rollouts = played(games(environment), "-1")
    assert [(rollout.status, rollout.calls) for rollout in rollouts] == [(Status.COMPLETED, ())] * 2


def test_a_session_whose_connection_drops_mid_step_is_an_infrastructure_failure(tmp_path):
    """A server killed while it holds a step: the step's call raises the dropped connection as an
    infrastructure failure, so the episode is infra and the scheduler retries it rather than the
    run failing."""
    port, stalled = free_port(), tmp_path / "stalled"
    server = subprocess.Popen([sys.executable, str(Path(openenv_support.__file__)), str(port), str(stalled)])
    try:
        while True:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                assert server.poll() is None, "the OpenEnv server exited"
            time.sleep(0.1)

        def drop():
            while not stalled.exists():
                time.sleep(0.05)
            server.kill()

        threading.Thread(target=drop, daemon=True).start()
        [rollout] = played(games(f"http://127.0.0.1:{port}"), "7", samples=1)
    finally:
        server.kill()
        server.wait()
    assert rollout.status is Status.INFRA_ERROR
