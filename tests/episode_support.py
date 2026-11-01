"""Square environments and sampled episode scaffolding over the tool policy of tests/tool_policy.py."""

import json
import sys
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import optax
from tool_policy import (
    ANSWER_NINE,
    ANSWER_SIXTEEN,
    CALL_FOUR,
    CALL_THREE,
    EOS,
    GROUPS,
    NINE,
    PAD,
    PROMPT,
    RESPONSE,
    SIXTEEN,
    START,
    TURNS,
    WRONG,
    ToolPolicy,
)

from dew.inference import Draw, TextGeneration
from dew.objectives.rl import GRPOObjective, RolloutScheduler, Snapshot, Synchronous, Task
from dew.objectives.rl.episodes import EpisodeCollector, EpisodeId, EpisodeStatus, Observation
from dew.rl.sandbox import SandboxLimits, SubprocessEnvironment
from dew.sampling import Sampling
from dew.training import Trainer

SAMPLING = Sampling(temperature=.7, top_k=2, eos_token_ids=EOS, pad_token_id=PAD)


IDENTITY = EpisodeId(task=3, attempt=0, sample=0, seed=(1, 2))


def environment(mode: str = "ok", *arguments: str, **limits) -> SubprocessEnvironment:
    worker = Path(__file__).with_name("sandbox_square_worker.py")
    return SubprocessEnvironment((sys.executable, str(worker), mode, *arguments), SandboxLimits(**limits))


def action(context, *tokens: int) -> Draw:
    draw = Draw(tuple(context), tokens, (-.1,) * len(tokens), "stop", 0, (0.,) * len(tokens))
    return draw.checked(SAMPLING)


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
        paid = self.harness.paid
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
            return Observation((*action.prompt, *action.tokens, observation),
                               detail=f"square({value}) = {self.answer}",
                               reward=None if paid is None else paid[0])
        answer = {ANSWER_NINE: 9, ANSWER_SIXTEEN: 16, WRONG: 8}[command[0]]
        return Observation(
            (), EpisodeStatus.COMPLETED, json.dumps({"answer": answer, "expected": self.answer}),
            reward=None if paid is None else paid[1],
        )


class Harness:
    """Square tool sessions; `paid`, where given, is the reward the tool step and the answer step pay."""

    def __init__(self, failure=None, paid=None):
        self.failure, self.paid = failure, paid
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
    """A trainer and the episode collector its scheduler (`episode_scheduler`) draws through."""
    model = ToolPolicy()
    objective = GRPOObjective(model, PROMPT + RESPONSE - 1, beta=.05)
    trainer = Trainer(objective, optax.sgd(.05), key=jax.random.key(19), accumulation=accumulation)
    collector = EpisodeCollector(
        TextGeneration(model, objective.init(jax.random.key(0))),
        harness or Harness(),
        verify,
        PROMPT,
        RESPONSE,
        TURNS,
        sampling=SAMPLING,
        record=record,
    )
    return trainer, replace(collector, **changes)


def episode_scheduler(collector, **changes) -> RolloutScheduler:
    """The trainer's rollout over `collector`: `GROUPS` episodes of each task, collected at each call."""
    return RolloutScheduler(Synchronous(collector), groups=GROUPS, **changes)


def cohort_snapshot(state, key=23) -> Snapshot:
    return Snapshot(state.variables, int(state.step), int(state.updates), jax.random.key(key))


def trained_batch(collector, state, key=23, tasks=(31,)):
    """The batch the trainer's scheduler (`episode_scheduler`) packs from one cohort of `tasks`' episodes."""
    return episode_scheduler(collector)(state, {"task_id": np.asarray(tasks, np.int32)}, jax.random.key(key))


def collect(collector, state, key=23, tasks=(31,)):
    """One cohort of `GROUPS` episodes of each of `tasks` under `state`, as the scheduler collects it."""
    return collector.episodes(cohort_snapshot(state, key), [Task(str(task)) for task in tasks], GROUPS)


def per_call(batch, name, episodes):
    """A packed column read back one `[TURNS * RESPONSE]`-wide row per call, in draw order."""
    out = np.zeros((len(episodes) * TURNS, RESPONSE), np.asarray(batch[name]).dtype)
    for index, episode in enumerate(episodes):
        for turn, transition in enumerate(episode.transitions):
            where = (batch["session_index"] == index) & (batch["call_index"] == turn)
            out[index * TURNS + turn, :len(transition.action.tokens)] = np.asarray(batch[name])[where]
    return out
