"""An OpenEnv environment served by OpenEnv's own server, played through `EnvironmentSource`.

The environment is a guessing game served in-process by `openenv`'s
`create_fastapi_app` over uvicorn, and each episode opens its own session
through `GenericEnvClient`, as a trainer reaching a remote environment does.
The policy is a bigram model on Dew's own server whose draws are its argmax:
it guesses 3 after the prompt and 7 after the environment's hint.
"""

import socket
import threading
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import uvicorn
from episode_support import ToolPolicy
from openenv.core import GenericEnvClient
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from dew.inference import NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.interop.openenv import OpenEnv, Said
from dew.objectives.rl import EnvironmentSource, Status, Task, Transcript, pack
from dew.sampling import Sampling

VOCAB, EOS, PROMPT, HINT = 16, 15, 1, 2
"""Ids: the prompt's, the environment's hint's, and the model's EOS; a digit d is id d + 3."""


class Guess(Action):
    number: int


class Feedback(Observation):
    hint: str = ""


class Guessing(Environment):
    """Guess the target; a wrong guess costs 0.1 and says higher or lower, the right one pays 1."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.target, self.steps = 0, 0

    def reset(self, seed=None, episode_id=None, **options):
        self.target, self.steps = options["target"], 0
        return Feedback(hint="guess a number from 0 to 9")

    def step(self, action, timeout_s=None, **kwargs):
        self.steps += 1
        if action.number == self.target:
            return Feedback(hint="right", done=True, reward=1.0)
        return Feedback(hint="higher" if action.number < self.target else "lower", reward=-0.1)

    @property
    def state(self):
        return State(step_count=self.steps)


@pytest.fixture(scope="module")
def environment():
    """The guessing game's server, on a free local port, for the module's tests."""
    app = create_fastapi_app(Guessing, Guess, Feedback, max_concurrent_envs=8)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
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


def test_an_openenv_episode_draws_each_turn_on_the_exact_context_and_keeps_the_steps_rewards(environment):
    """Each episode opens its own OpenEnv session, resets it with its task's target, and steps it on
    the guess each turn drew. The second call's context is the first's, its drawn ids and the hint's
    id; each call keeps the reward the environment paid its step and the messages before it; the
    right guess completes the episode, and its calls pack into one chain."""
    @contextmanager
    def session(task, identity):
        with GenericEnvClient(base_url=environment).sync() as opened:
            yield opened

    def said(observation):
        return {"role": "user", "content": observation["hint"]}

    games = OpenEnv(session, prompt=lambda observation: Said((PROMPT,), (said(observation),)),
                    reply=lambda observation: Said((HINT,), (said(observation),)),
                    action=lambda ids: {"number": ids[0] - 3},
                    options=lambda task, identity: {"target": int(task.id)})
    model = ToolPolicy(vocab_size=VOCAB, max_seq_len=64, logits=guesser)
    params = model.init(jax.random.key(0), jnp.ones((1, 1), jnp.int32))
    sampling = Sampling(temperature=1.0, eos_token_ids=EOS)
    server = NativeRolloutServer(Server.from_task(TextGeneration(model, params, None, sampling=sampling),
                                                  slots=4, capacity=64))
    episodes = EnvironmentSource(server, games, lambda task, episode: 0.0, max_prompt_tokens=32,
                                 max_new_tokens=4, max_turns=4, workers=2)
    try:
        submitted = episodes.submit(Task("7"), 2, version=0, key=jax.random.key(1))
        rollouts = [future.result() for future in submitted]
    finally:
        episodes.close()
        server.close()
    prompt = {"role": "user", "content": "guess a number from 0 to 9"}
    higher = {"role": "user", "content": "higher"}
    for rollout in rollouts:
        assert (rollout.status, rollout.reward) == (Status.COMPLETED, 0.0)
        first, second = rollout.calls
        assert (first.prompt_ids, first.sampled_ids) == ((PROMPT,), (6, EOS))
        assert (second.prompt_ids, second.sampled_ids) == ((PROMPT, 6, EOS, HINT), (10, EOS))
        assert (first.reward, second.reward) == (-0.1, 1.0)
        assert (first.transcript, second.transcript) == (Transcript((prompt,)), Transcript((prompt, higher)))
    batch = pack(rollouts, 8, credit="turn")
    assert batch["input_ids"].shape == (2, 8)
