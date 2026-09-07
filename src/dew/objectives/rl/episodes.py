"""Tool episodes collected outside jit and projected into ordinary GRPO rows.

Execution belongs to a user-supplied, context-managed environment. Dew has
no default executor. Each model call retains its exact context and sampled
likelihoods; observations are inputs to later calls and never loss targets.
"""

from __future__ import annotations

from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from enum import IntEnum
import math
from typing import Protocol
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np

from dew.data.prompts import LENGTH_KEY
from dew.rl import group_advantage
from dew.objectives.base import Variables
from dew.sampling.text import Generation, Sampling
from dew.nn.inputs import local_rows
from dew.training.state import TrainState

from .rollout import (
    ADVANTAGES_KEY, BEHAVIOR_LOG_PROBS_KEY, IDS_KEY, OLD_LOG_PROBS_KEY,
    RESPONSE_LENGTH_KEY, RESPONSE_MASK_KEY, REWARDS_KEY, TERMINATED_KEY,
)


class EpisodeStatus(IntEnum):
    RUNNING = 0
    COMPLETED = 1
    TRUNCATED = 2
    CANCELLED = 3
    ERROR = 4


@dataclass(frozen=True)
class EpisodeId:
    """An attempt-local sample identity, reproducible from checkpointed work.

    The harness can use this identity for its own idempotency records. Dew
    does not guarantee exactly-once external effects across process failure.
    """

    task: int
    attempt: int
    sample: int
    seed: tuple[int, ...]


@dataclass(frozen=True)
class Observation:
    """The exact next model context, or a terminal environment result.

    The harness owns chat formatting, tool-call parsing, and context
    compaction. Terminal contexts may be empty. Detail can hold a verifier
    result or an artifact reference without encoding it into device arrays.
    """

    context: tuple[int, ...]
    status: EpisodeStatus = EpisodeStatus.RUNNING
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, EpisodeStatus):
            raise TypeError("observation status must be an EpisodeStatus")
        if any(type(token) is not int or token < 0 for token in self.context):
            raise ValueError("observation context must contain nonnegative token ids")
        if self.status == EpisodeStatus.RUNNING and not self.context:
            raise ValueError("a running environment must supply a model context")


@dataclass(frozen=True)
class Action:
    """One actual model call, including EOS and both likelihood distributions.

    Raw probabilities belong to the unmodified model; behavior probabilities
    include Sampling controls. EOS ends the model turn, not the episode.
    """

    context: tuple[int, ...]
    tokens: tuple[int, ...]
    raw_log_probs: tuple[float, ...]
    behavior_log_probs: tuple[float, ...]
    terminated: bool
    policy_step: int
    sampling: Sampling
    _binding_id: str = field(default="", repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        if not self.tokens or len(self.tokens) != len(self.raw_log_probs) or len(self.tokens) != len(self.behavior_log_probs):
            raise ValueError("every sampled action token needs raw and behavior likelihoods")
        if not all(math.isfinite(value) for value in (*self.raw_log_probs, *self.behavior_log_probs)):
            raise ValueError("sampled action likelihoods must be finite")


@dataclass(frozen=True)
class Transition:
    action: Action
    observation: Observation


@dataclass(frozen=True)
class Episode:
    identity: EpisodeId
    policy_step: int
    initial: Observation | None
    transitions: tuple[Transition, ...]
    status: EpisodeStatus
    detail: str = ""
    reward: float | None = None
    # A collection binds exactly once. Its private origin is independent of
    # clocks, which can coincide across different runs or parameter trees.
    # Exclude the origin from value equality so replayed numerical records
    # remain comparable; project checks it explicitly before admitting data.
    _binding_id: str = field(default="", repr=False, compare=False, kw_only=True)


class Environment(Protocol):
    """The caller owns execution and releases resources on context exit."""

    def reset(self) -> Observation: ...

    def step(self, action: Action) -> Observation: ...


EnvironmentFactory = Callable[[EpisodeId], AbstractContextManager[Environment]]
Verifier = Callable[[Episode], float]
EpisodeRecorder = Callable[[Episode], None]


class EpisodeInference(Protocol):
    """A bindable autoregressive policy returning actual sampling likelihoods."""

    def bind(self, variables: Variables, /) -> EpisodeInference: ...

    def __call__(self, inputs: Sequence[Sequence[int]], max_new_tokens: int, /,
                 *, key: jax.Array, sampling: Sampling) -> Generation: ...


class EpisodeFailure(RuntimeError):
    """Collection failed; the partial episode remains available to the caller."""

    def __init__(self, episode: Episode):
        self.episode = episode
        super().__init__(f"episode {episode.identity}: {episode.status.name.lower()}: {episode.detail}")


class EpisodeCancelled(CancelledError):
    def __init__(self, episode: Episode):
        self.episode = episode
        super().__init__(f"episode {episode.identity} cancelled: {episode.detail}")


@dataclass(frozen=True)
class EpisodeRollout:
    """Collect complete groups under one policy snapshot, then train actions.

    Input batches contain integer task_id rows. The environment factory
    resolves each task and owns its tools, timeouts and isolation. A finite
    verifier reward is required for completed and truncated episodes. Errors
    and cancellation abort the whole group before a Trainer update; record
    receives the partial episode before the exception propagates.

    Each model call becomes one fixed-width GRPO row. Set the objective's
    seq_len to max_prompt_tokens + max_new_tokens - 1. The terminal group
    advantage is shared by the episode's actions; no per-turn credit rule
    is inferred. Host records and numeric rows carry the trainer's committed
    update clock. Raw and behavior likelihoods come from actual draws.

    The policy binds one immutable variables snapshot for the whole
    collection. It must use that binding, not a mutable serving default.
    The Trainer cannot update or donate the tree until this call returns.
    Checkpoints resume at Trainer boundaries, not halfway through an
    external tool call.
    """

    policy: EpisodeInference
    environment: EnvironmentFactory
    verifier: Verifier
    max_prompt_tokens: int
    max_new_tokens: int
    max_turns: int
    sampling: Sampling
    groups: int = 2
    record: EpisodeRecorder | None = None

    def __post_init__(self) -> None:
        for name in ("max_prompt_tokens", "max_new_tokens", "max_turns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.groups) is not int or self.groups < 2:
            raise ValueError("an episode group needs at least two samples")
        if self.sampling.eos_id is None:
            raise ValueError("tool episodes need an EOS token to distinguish complete and truncated actions")

    def _episode(self, identity: EpisodeId, policy: EpisodeInference,
                 policy_step: int, key: jax.Array, binding_id: str) -> Episode:
        initial = None
        transitions: list[Transition] = []
        status = EpisodeStatus.RUNNING
        detail = ""
        pending: Action | None = None
        episode: Episode | None = None
        try:
            with self.environment(identity) as environment:
                observation = environment.reset()
                initial = observation
                status, detail = observation.status, observation.detail
                while status == EpisodeStatus.RUNNING:
                    if len(transitions) == self.max_turns:
                        status, detail = EpisodeStatus.TRUNCATED, "episode turn limit reached"
                        break
                    if len(observation.context) > self.max_prompt_tokens:
                        status, detail = EpisodeStatus.TRUNCATED, "next context exceeds max_prompt_tokens"
                        break
                    result = policy(
                        [observation.context], self.max_new_tokens,
                        key=jax.random.fold_in(key, len(transitions)), sampling=self.sampling)
                    if not isinstance(result, Generation):
                        raise TypeError("tool episodes require autoregressive Generation likelihoods")
                    pending = self._action(result, observation.context, policy_step, binding_id)
                    if not pending.terminated:
                        observation = Observation((), EpisodeStatus.TRUNCATED, "model turn reached its token limit")
                    else:
                        observation = environment.step(pending)
                    transitions.append(Transition(pending, observation))
                    pending = None
                    status, detail = observation.status, observation.detail
                episode = Episode(identity, policy_step, initial, tuple(transitions), status, detail,
                                  _binding_id=binding_id)
                if status in (EpisodeStatus.COMPLETED, EpisodeStatus.TRUNCATED):
                    # A verifier may inspect files or services owned by the
                    # environment. Score before its context releases them.
                    reward = float(self.verifier(episode))
                    if not math.isfinite(reward):
                        raise ValueError("episode verifier returned a non-finite reward")
                    episode = replace(episode, reward=reward)
            if episode is None or (status in (EpisodeStatus.COMPLETED, EpisodeStatus.TRUNCATED)
                                   and episode.reward is None):
                raise RuntimeError("environment exited before episode verification completed")
        except BaseException as error:
            status = (EpisodeStatus.CANCELLED if isinstance(
                error, (AsyncCancelledError, CancelledError, KeyboardInterrupt)) else EpisodeStatus.ERROR)
            detail = f"{type(error).__name__}: {error}"
            if pending is not None:
                transitions.append(Transition(pending, Observation((), status, detail)))
            episode = Episode(identity, policy_step, initial, tuple(transitions), status, detail,
                              _binding_id=binding_id)
            if self.record is not None:
                self.record(episode)
            if status == EpisodeStatus.CANCELLED:
                raise EpisodeCancelled(episode) from error
            raise EpisodeFailure(episode) from error
        if self.record is not None:
            self.record(episode)
        if episode.status == EpisodeStatus.CANCELLED:
            raise EpisodeCancelled(episode)
        if episode.status == EpisodeStatus.ERROR:
            raise EpisodeFailure(episode)
        return episode

    def _action(self, result: Generation, context: tuple[int, ...], policy_step: int,
                binding_id: str) -> Action:
        """Validate recorded tokens and likelihoods before an environment acts."""
        rows = result.host()
        tokens = np.asarray(rows.tokens)
        lengths, ended = np.asarray(rows.lengths), np.asarray(rows.terminated)
        raw, behavior = np.asarray(rows.raw_log_probs), np.asarray(rows.behavior_log_probs)
        width = len(context)
        if (tokens.shape != (1, width + self.max_new_tokens)
                or not np.issubdtype(tokens.dtype, np.integer) or np.any(tokens < 0)
                or not np.array_equal(tokens[0, :width], context)):
            raise ValueError("inference must retain the exact requested context in integer tokens")
        if (lengths.shape != (1,) or not np.issubdtype(lengths.dtype, np.integer)
                or not 0 < lengths[0] <= self.max_new_tokens):
            raise ValueError("inference must return a valid sampled action length")
        if raw.shape != (1, self.max_new_tokens) or behavior.shape != raw.shape:
            raise ValueError("inference must record raw and behavior likelihoods for every output slot")
        if ended.shape != (1,) or ended.dtype != np.bool_:
            raise ValueError("inference must record a boolean EOS termination flag")
        count = int(lengths[0])
        actions = tuple(int(value) for value in tokens[0, width:width + count])
        stops = self.sampling.eos_id
        assert stops is not None
        if bool(ended[0]) != bool(np.isin(actions[-1], stops)):
            raise ValueError("inference termination disagrees with the sampled EOS token")
        return Action(context, actions, tuple(float(value) for value in raw[0, :count]),
                      tuple(float(value) for value in behavior[0, :count]),
                      bool(ended[0]), policy_step, self.sampling, _binding_id=binding_id)

    def collect(self, state: TrainState, batch: Mapping[str, object], key: jax.Array) -> tuple[Episode, ...]:
        """Collect task-major groups without optimizer effects."""
        if jax.process_count() != 1:
            raise NotImplementedError(
                "variable-turn episode collection needs a distributed environment coordinator")
        tasks = local_rows(batch["task_id"])
        if tasks.ndim != 1 or not tasks.size or not np.issubdtype(tasks.dtype, np.integer):
            raise ValueError("task_id must be a nonempty vector of integer task identities")
        policy_step = int(state.updates)
        # Copy only the containers; JAX buffers are immutable and remain live
        # until collection returns. Replacing a caller's mapping cannot move
        # the policy halfway through an episode.
        variables = jax.tree.map(lambda leaf: leaf, state.params)
        policy = self.policy.bind(variables)
        binding_id = uuid4().hex
        episodes = []
        for index, task in enumerate(tasks):
            for group in range(self.groups):
                sample = index * self.groups + group
                draw = jax.random.fold_in(key, sample)
                identity = EpisodeId(int(task), int(state.step), sample,
                                     tuple(int(value) for value in np.asarray(jax.random.key_data(draw))))
                episodes.append(self._episode(identity, policy, policy_step, draw, binding_id))
        return tuple(episodes)

    def __call__(self, state: TrainState, batch: Mapping[str, object], key: jax.Array) -> dict[str, np.ndarray]:
        episodes = self.collect(state, batch, key)
        return self.project(episodes)

    def project(self, episodes: Sequence[Episode]) -> dict[str, np.ndarray]:
        """Project action tokens from one collection; padded turns have zero support.

        Every episode and action must retain that collection's private binding
        origin. Equal training clocks do not establish equal weight snapshots.
        """
        if not episodes or len(episodes) % self.groups:
            raise ValueError("projection requires complete episode groups")
        if len({episode.identity for episode in episodes}) != len(episodes):
            raise ValueError("each episode sample must have a distinct identity")
        policy_step = episodes[0].policy_step
        binding_id = episodes[0]._binding_id
        if not binding_id:
            raise ValueError("episode records must retain their collection binding")
        rewards = []
        for index, episode in enumerate(episodes):
            if (episode._binding_id != binding_id
                    or any(turn.action._binding_id != binding_id for turn in episode.transitions)):
                raise ValueError("episode records must come from the same collection binding")
            if (episode.policy_step != policy_step
                    or episode.identity.attempt != episodes[0].identity.attempt
                    or any(turn.action.policy_step != policy_step for turn in episode.transitions)):
                raise ValueError("an episode batch must share one policy snapshot and attempt")
            if episode.status not in (EpisodeStatus.COMPLETED, EpisodeStatus.TRUNCATED) or episode.reward is None:
                raise ValueError("only verified completed or truncated episodes may train")
            if not math.isfinite(episode.reward):
                raise ValueError("episode rewards must be finite")
            if len(episode.transitions) > self.max_turns:
                raise ValueError("episode exceeds max_turns")
            group_start = episodes[index - index % self.groups]
            if episode.identity.task != group_start.identity.task:
                raise ValueError("an advantage group must contain the same task")
            rewards.append(episode.reward)
        advantages = np.asarray(group_advantage(jnp.asarray(rewards, jnp.float32), self.groups))
        rows = len(episodes) * self.max_turns
        prompt, response = self.max_prompt_tokens, self.max_new_tokens
        ids = np.full((rows, prompt + response), self.sampling.pad_id, np.int32)
        mask = np.zeros((rows, response), np.float32)
        raw = np.zeros_like(mask)
        behavior = np.zeros_like(mask)
        lengths = np.zeros(rows, np.int32)
        prompt_lengths = np.ones(rows, np.int32)
        terminated = np.zeros(rows, bool)
        for index, episode in enumerate(episodes):
            for turn_index, transition in enumerate(episode.transitions):
                action = transition.action
                size, count = len(action.context), len(action.tokens)
                if not 0 < size <= prompt or count > response:
                    raise ValueError("recorded action does not fit the projection's token budgets")
                row = index * self.max_turns + turn_index
                ids[row, prompt - size:prompt] = action.context
                ids[row, prompt:prompt + count] = action.tokens
                mask[row, :count] = 1
                raw[row, :count] = action.raw_log_probs
                behavior[row, :count] = action.behavior_log_probs
                lengths[row], prompt_lengths[row] = count, size
                terminated[row] = action.terminated
        return {
            IDS_KEY: ids, RESPONSE_MASK_KEY: mask, OLD_LOG_PROBS_KEY: raw,
            BEHAVIOR_LOG_PROBS_KEY: behavior, RESPONSE_LENGTH_KEY: lengths,
            LENGTH_KEY: prompt_lengths, TERMINATED_KEY: terminated,
            REWARDS_KEY: np.repeat(np.asarray(rewards, np.float32), self.max_turns),
            ADVANTAGES_KEY: np.broadcast_to(np.repeat(advantages, self.max_turns)[:, None], mask.shape),
            "episode_status": np.repeat(np.asarray([episode.status for episode in episodes], np.int32), self.max_turns),
            "task_id": np.repeat(np.asarray([episode.identity.task for episode in episodes], np.int32), self.max_turns),
            "policy_step": np.full(rows, policy_step, np.int32),
        }
