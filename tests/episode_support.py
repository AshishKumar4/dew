"""Trainable tool policy, square environments and sampled episode scaffolding."""

import json
import sys
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.inference import TextGeneration
from dew.nn.protocols import OutputTable
from dew.objectives.rl import GRPOObjective
from dew.objectives.rl.episodes import Action, EpisodeId, EpisodeRollout, EpisodeStatus, Observation
from dew.rl.sandbox import SandboxLimits, SubprocessEnvironment
from dew.sampling import Sampling
from dew.training import Trainer

# Compact tool vocabulary: a call, a tool result, a final answer, and EOS.
PAD, START, CALL_THREE, EOS, NINE, ANSWER_NINE, WRONG, CALL_FOUR, SIXTEEN, ANSWER_SIXTEEN = range(10)
VOCAB = 12
PROMPT, RESPONSE, TURNS, GROUPS = 8, 3, 3, 4
SAMPLING = Sampling(temperature=.7, top_k=2, eos_id=EOS, pad_id=PAD)


IDENTITY = EpisodeId(task=3, attempt=0, sample=0, seed=(1, 2))


def environment(mode: str = "ok", **limits) -> SubprocessEnvironment:
    worker = Path(__file__).with_name("sandbox_square_worker.py")
    return SubprocessEnvironment((sys.executable, str(worker), mode), SandboxLimits(**limits))


def action(context, *tokens: int) -> Action:
    return Action(tuple(context), tokens, (-.1,) * len(tokens), (0.,) * len(tokens),
                  terminated=True, policy_step=0, sampling=SAMPLING)


def transition_logits():
    logits = np.full((VOCAB, VOCAB), -1000., np.float32)
    logits[:, EOS] = 1000.
    logits[START, :] = -1000.
    logits[START, CALL_THREE] = .2
    logits[START, CALL_FOUR] = 0.
    for observation, answer in ((NINE, ANSWER_NINE), (SIXTEEN, ANSWER_SIXTEEN)):
        logits[observation, :] = -1000.
        logits[observation, answer] = .25
        logits[observation, WRONG] = 0.
    return logits


class ToolPolicy(nn.Module):
    """A trainable bigram policy with a finite tool-call vocabulary.

    The shared sampler draws calls and final answers from logits; the test
    environment executes square(3) or square(4). No generated code executes.
    """

    vocab_size: int = VOCAB
    max_seq_len: int = PROMPT + RESPONSE

    def setup(self):
        self.table = self.param("table", lambda key: jnp.asarray(transition_logits()))

    def hidden_states(self, tokens, train=False, attention_mask=None, positions=None, segment_ids=None):
        return jax.nn.one_hot(tokens, self.vocab_size)

    def output_table(self):
        return OutputTable(self.table, vocab_major=False)

    @nn.compact
    def init_cache(self, batch_size):
        self.variable("cache", "seen", lambda: jnp.zeros(batch_size, jnp.int32))

    def __call__(
        self, tokens, train=False, decode=False, attention_mask=None, positions=None, segment_ids=None
    ):
        if decode:
            seen = self.get_variable("cache", "seen")
            valid = jnp.ones_like(tokens, bool) if attention_mask is None else attention_mask
            self.put_variable("cache", "seen", seen + valid.sum(-1))
        return self.hidden_states(tokens) @ self.table


class SquareSession:
    def __init__(self, identity, harness):
        self.identity, self.harness = identity, harness
        self.answer = None

    def reset(self):
        self.harness.opened.append(self.identity)
        if self.harness.failure == "reset":
            raise OSError("environment unavailable")
        return Observation((START,))

    def step(self, action):
        assert action.tokens[-1] == EOS
        command = action.tokens[:-1]
        if self.harness.failure == "tool":
            raise OSError("tool transport failed")
        if self.harness.failure == "cancel":
            raise CancelledError("owner cancelled")
        if self.harness.failure == "cancel_status":
            return Observation((), EpisodeStatus.CANCELLED, "owner cancelled")
        if self.answer is None:
            assert len(command) == 1
            value = {CALL_THREE: 3, CALL_FOUR: 4}[command[0]]
            self.answer = value * value
            self.harness.calls.append((self.identity, value, self.answer))
            observation = {9: NINE, 16: SIXTEEN}[self.answer]
            return Observation((*action.context, *action.tokens, observation),
                               detail=f"square({value}) = {self.answer}")
        answer = {ANSWER_NINE: 9, ANSWER_SIXTEEN: 16, WRONG: 8}[command[0]]
        return Observation(
            (), EpisodeStatus.COMPLETED, json.dumps({"answer": answer, "expected": self.answer})
        )


class Harness:
    def __init__(self, failure=None):
        self.failure = failure
        self.opened, self.closed, self.calls = [], [], []

    @contextmanager
    def __call__(self, identity):
        try:
            yield SquareSession(identity, self)
        finally:
            self.closed.append(identity)
            if self.failure == "close":
                raise OSError("environment release failed")


def verify(episode):
    if episode.status == EpisodeStatus.TRUNCATED:
        return -1.
    result = json.loads(episode.detail)
    return float(result["answer"] == result["expected"])


def build(harness=None, *, record=None, accumulation=1, **changes):
    model = ToolPolicy()
    objective = GRPOObjective(model, PROMPT + RESPONSE - 1, beta=.05)
    trainer = Trainer(objective, optax.sgd(.05), key=jax.random.key(19), accumulation=accumulation)
    rollout = EpisodeRollout(
        TextGeneration(model, objective.init(jax.random.key(0))),
        harness or Harness(),
        verify,
        PROMPT,
        RESPONSE,
        TURNS,
        groups=GROUPS,
        sampling=SAMPLING,
        record=record,
    )
    return trainer, replace(rollout, **changes)


def collect(rollout, state, key=23):
    return rollout.collect(state, {"task_id": np.array([31], np.int32)}, jax.random.key(key))


def per_call(batch, name, episodes):
    """A packed column read back one `[TURNS * RESPONSE]`-wide row per call, in draw order."""
    out = np.zeros((len(episodes) * TURNS, RESPONSE), np.asarray(batch[name]).dtype)
    for index, episode in enumerate(episodes):
        for turn, transition in enumerate(episode.transitions):
            where = (batch["session_index"] == index) & (batch["call_index"] == turn)
            out[index * TURNS + turn, :len(transition.action.tokens)] = np.asarray(batch[name])[where]
    return out

