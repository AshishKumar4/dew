"""Tool episodes collected outside jit and projected into ordinary GRPO rows.

Execution belongs to a user-supplied, context-managed environment. Dew has
no default executor. Each model call retains its exact context and sampled
likelihoods; observations are inputs to later calls and never loss targets.
"""

from __future__ import annotations

from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field, replace
from enum import IntEnum
import hashlib
import math
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import numpy as np

from dew.artifacts import agree_process_phase
from dew.nn.inputs import ModelInputs
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

if TYPE_CHECKING:
    from .journal import EpisodeJournal, JournalRun

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
    Context and action ids are nonnegative integers. Actions stop at the
    first configured EOS, and terminated agrees with that final token.
    Vocabulary upper bounds belong to the model-aware caller.
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
        for name, ids in (("context", self.context), ("tokens", self.tokens)):
            if not ids:
                raise ValueError(f"sampled action {name} must be nonempty")
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError(f"sampled action {name} must contain nonnegative integer token ids")
        if len(self.tokens) != len(self.raw_log_probs) or len(self.tokens) != len(self.behavior_log_probs):
            raise ValueError("every sampled action token needs raw and behavior likelihoods")
        for probabilities in (self.raw_log_probs, self.behavior_log_probs):
            if not all(math.isfinite(value) for value in probabilities):
                raise ValueError("sampled action likelihoods must be finite")
        if type(self.terminated) is not bool:
            raise ValueError("sampled action terminated must be a boolean")
        eos = self.sampling.eos_id
        stops = () if eos is None else (eos,) if isinstance(eos, int) else eos
        if self.terminated != (self.tokens[-1] in stops):
            raise ValueError("inference termination disagrees with the sampled EOS token")
        for index, token in enumerate(self.tokens):
            if token in stops and index != len(self.tokens) - 1:
                raise ValueError("sampled action cannot contain tokens after EOS")


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


@runtime_checkable
class RecoverableEnvironment(Environment, Protocol):
    """Opaque snapshots restore tool state without replaying completed calls."""

    def get_state(self) -> bytes: ...

    def set_state(self, state: bytes) -> None: ...

EnvironmentFactory = Callable[[EpisodeId], AbstractContextManager[Environment]]
Verifier = Callable[[Episode], float]
EpisodeRecorder = Callable[[Episode], None]


class EpisodeInference(Protocol):
    """A bindable autoregressive policy returning actual sampling likelihoods."""

    def bind(self, variables: Variables, /) -> EpisodeInference: ...

    def __call__(self, inputs: ModelInputs | Sequence[Sequence[int]], max_new_tokens: int, /,
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


_T = TypeVar("_T")
_P = ParamSpec("_P")


class _PeerFailure(RuntimeError):
    """Another rank failed before the next collective generation call."""


class _StatusStop(RuntimeError):
    """An environment returned an explicit error or cancellation outcome."""


def _phase(operation: Callable[[], _T], name: str) -> _T:
    result: tuple[_T] | None = None
    error = None
    try:
        result = (operation(),)
    except BaseException as failure:
        error = failure
    try:
        agree_process_phase(error, phase=name)
    except BaseException as failure:
        if error is None:
            raise _PeerFailure(str(failure)) from failure
        raise
    assert result is not None
    return result[0]


@dataclass
class _Session:
    identity: EpisodeId
    environment: Environment | None = None
    initial: Observation | None = None
    observation: Observation | None = None
    transitions: list[Transition] = field(default_factory=list)
    pending: Action | None = None
    status: EpisodeStatus = EpisodeStatus.RUNNING
    detail: str = ""
    reward: float | None = None
    error: BaseException | None = None

    def invoke(self, operation: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return operation(*args, **kwargs)
        except BaseException as error:
            self.error = error
            raise

    def observe(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise TypeError("environment methods must return an Observation")
        self.observation = observation
        self.status, self.detail = observation.status, observation.detail
        if self.status in (EpisodeStatus.ERROR, EpisodeStatus.CANCELLED):
            raise _StatusStop(self.detail)

    def episode(self, policy_step: int, binding_id: str) -> Episode:
        return Episode(self.identity, policy_step, self.initial, tuple(self.transitions),
                       self.status, self.detail, self.reward, _binding_id=binding_id)

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
    EpisodeJournal adds durable turn boundaries for environments exposing
    get_state/set_state. Pending external effects require environment-owned
    idempotency; Trainer checkpoints remain the optimizer's recovery boundary.
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
    journal: EpisodeJournal | None = None

    def __post_init__(self) -> None:
        for name in ("max_prompt_tokens", "max_new_tokens", "max_turns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.groups) is not int or self.groups < 2:
            raise ValueError("an episode group needs at least two samples")
        if self.sampling.eos_id is None:
            raise ValueError("tool episodes need an EOS token to distinguish complete and truncated actions")

    def _persist(self, slot: _Session, run: JournalRun | None, policy_step: int, binding: str) -> None:
        if run is not None:
            environment = slot.environment
            if not isinstance(environment, RecoverableEnvironment):
                raise TypeError("EpisodeJournal requires get_state/set_state on the environment")
            snapshot = slot.invoke(environment.get_state)
            if not isinstance(snapshot, bytes):
                raise TypeError("environment get_state must return bytes")
            slot.invoke(run.save, slot.episode(policy_step, binding), slot.pending, snapshot)

    def _open(self, slots: list[_Session], stack: ExitStack, run: JournalRun | None,
              policy_step: int, binding: str) -> None:
        for slot in slots:
            slot.environment = slot.invoke(lambda: stack.enter_context(self.environment(slot.identity)))
            saved = None if run is None else slot.invoke(run.load, slot.identity)
            if saved is None:
                observation = slot.invoke(slot.environment.reset)
                slot.initial = observation
                slot.invoke(slot.observe, observation)
                self._persist(slot, run, policy_step, binding)
            else:
                if not isinstance(slot.environment, RecoverableEnvironment):
                    raise TypeError("EpisodeJournal requires get_state/set_state on the environment")
                slot.invoke(slot.environment.set_state, saved.snapshot)
                episode = saved.episode
                slot.initial, slot.transitions = episode.initial, list(episode.transitions)
                slot.pending, slot.reward = saved.pending, episode.reward
                slot.observation = episode.transitions[-1].observation if episode.transitions else episode.initial
                slot.status, slot.detail = episode.status, episode.detail

    def _inputs(self, slots: list[_Session], turn: int) -> ModelInputs:
        tokens = np.full((len(slots), self.max_prompt_tokens), self.sampling.pad_id, np.int32)
        valid = np.zeros_like(tokens, bool)
        for row, slot in enumerate(slots):
            if slot.status == EpisodeStatus.RUNNING and len(slot.transitions) == turn and slot.pending is None:
                observation = slot.observation
                assert observation is not None
                length = len(observation.context)
                if length > self.max_prompt_tokens:
                    slot.status, slot.detail = EpisodeStatus.TRUNCATED, "next context exceeds max_prompt_tokens"
                else:
                    tokens[row, -length:] = observation.context
                    valid[row, -length:] = True
            if not valid[row].any():
                # Keep global row indices and collective shapes stable. The
                # draw for a finished slot is discarded, never executed or trained.
                valid[row, -1] = True
        return ModelInputs(jnp.asarray(tokens), {"attention_mask": jnp.asarray(valid)})

    def _advance(self, slots: list[_Session], result: Generation, policy_step: int,
                 binding_id: str, turn: int, run: JournalRun | None) -> None:
        if not isinstance(result, Generation):
            raise TypeError("tool episodes require an autoregressive Generation")
        rows = result.host()
        if rows.tokens.shape[0] != len(slots):
            raise TypeError("tool episodes require one autoregressive Generation row per slot")
        # Record every actual draw before invoking any tool. A tool failure
        # must not erase the other requests already sampled in this cohort.
        for row, slot in enumerate(slots):
            if slot.status == EpisodeStatus.RUNNING and len(slot.transitions) == turn and slot.pending is None:
                observation = slot.observation
                assert observation is not None
                slot.pending = slot.invoke(self._action, rows, row, observation.context, policy_step, binding_id)
                self._persist(slot, run, policy_step, binding_id)
        for slot in slots:
            action = slot.pending
            if action is None or len(slot.transitions) != turn:
                continue
            environment = slot.environment
            assert environment is not None
            if action.terminated:
                observation = slot.invoke(environment.step, action)
            else:
                observation = Observation((), EpisodeStatus.TRUNCATED, "model turn reached its token limit")
            if not isinstance(observation, Observation):
                slot.invoke(slot.observe, observation)
            slot.transitions.append(Transition(action, observation))
            slot.pending = None
            slot.invoke(slot.observe, observation)
            self._persist(slot, run, policy_step, binding_id)

    def _verify(self, slots: list[_Session], policy_step: int, binding_id: str, run: JournalRun | None) -> None:
        for slot in slots:
            if slot.reward is not None:
                continue
            if slot.status == EpisodeStatus.RUNNING:
                slot.status, slot.detail = EpisodeStatus.TRUNCATED, "episode turn limit reached"
            def score() -> float:
                reward = float(self.verifier(slot.episode(policy_step, binding_id)))
                if not math.isfinite(reward):
                    raise ValueError("episode verifier returned a non-finite reward")
                return reward
            slot.reward = slot.invoke(score)
            self._persist(slot, run, policy_step, binding_id)

    def _failed(self, slots: list[_Session], error: BaseException,
                policy_step: int, binding_id: str) -> None:
        started = [slot for slot in slots if slot.environment is not None or slot.error is not None]
        if not started:
            raise error
        primary = next((slot for slot in started if slot.error is not None), started[0])
        cancelled = isinstance(error, (AsyncCancelledError, CancelledError, KeyboardInterrupt, _PeerFailure))
        explicit = isinstance(error, _StatusStop)
        if explicit:
            cancelled = primary.status == EpisodeStatus.CANCELLED
        for slot in started:
            slot.status = (EpisodeStatus.ERROR if slot is primary and not cancelled else EpisodeStatus.CANCELLED)
            slot.detail = str(error) if explicit else f"{type(error).__name__}: {error}"
            slot.reward = None
            if slot.pending is not None:
                slot.transitions.append(Transition(slot.pending, Observation((), slot.status, slot.detail)))
                slot.pending = None
        records = tuple(slot.episode(policy_step, binding_id) for slot in started)
        def record() -> None:
            if self.record is not None:
                for episode in records:
                    self.record(episode)
        _phase(record, "episode abort recording")
        episode = records[started.index(primary)]
        failure = EpisodeCancelled(episode) if cancelled else EpisodeFailure(episode)
        if explicit:
            raise failure from None
        raise failure from error

    def _action(self, result: Generation, row: int, context: tuple[int, ...], policy_step: int,
                binding_id: str) -> Action:
        """Validate a cohort row's provenance before an environment acts."""
        tokens = np.asarray(result.tokens)[row]
        lengths, ended = np.asarray(result.lengths), np.asarray(result.terminated)
        raw, behavior = np.asarray(result.raw_log_probs)[row], np.asarray(result.behavior_log_probs)[row]
        width = self.max_prompt_tokens
        expected = np.full(width, self.sampling.pad_id, np.int32)
        expected[-len(context):] = context
        if (tokens.shape != (width + self.max_new_tokens,)
                or not np.issubdtype(tokens.dtype, np.integer) or np.any(tokens < 0)
                or not np.array_equal(tokens[:width], expected)):
            raise ValueError("inference must retain the exact requested context in integer tokens")
        if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
                or not 0 < lengths[row] <= self.max_new_tokens):
            raise ValueError("inference must return a valid sampled action length")
        if raw.shape != (self.max_new_tokens,) or behavior.shape != raw.shape:
            raise ValueError("inference must record raw and behavior likelihoods for every output slot")
        if ended.shape != lengths.shape or ended.dtype != np.bool_:
            raise ValueError("inference must record a boolean EOS termination flag")
        count = int(lengths[row])
        actions = tuple(int(value) for value in tokens[width:width + count])
        return Action(context, actions, tuple(float(value) for value in raw[:count]),
                      tuple(float(value) for value in behavior[:count]),
                      bool(ended[row]), policy_step, self.sampling, _binding_id=binding_id)

    def collect(self, state: TrainState, batch: Mapping[str, object], key: jax.Array) -> tuple[Episode, ...]:
        """Collect fixed cohorts, agreeing host phases before every generation."""
        def prepare():
            tasks = local_rows(batch["task_id"])
            if tasks.ndim != 1 or not tasks.size or not np.issubdtype(tasks.dtype, np.integer):
                raise ValueError("task_id must be a nonempty vector of integer task identities")
            if key.shape != ():
                raise ValueError("episode key must be one PRNG key")
            policy = self.policy.bind(state.params)
            signature = (tasks.size, self.groups, self.max_turns, self.max_prompt_tokens,
                         self.max_new_tokens, self.sampling, int(state.step), int(state.updates),
                         tuple(np.asarray(jax.random.key_data(key)).tolist()), self.journal is not None)
            return tasks, policy, np.frombuffer(hashlib.sha256(repr(signature).encode()).digest(), np.uint8)
        tasks, policy, signature = _phase(prepare, "episode preparation")
        processes, rank = jax.process_count(), jax.process_index()
        if processes > 1:
            multihost_utils.assert_equal(signature, "episode task counts, budgets, sampling and clocks must agree")
        origin = np.frombuffer(uuid4().bytes, np.uint8) if rank == 0 else np.zeros(16, np.uint8)
        if processes > 1:
            origin = multihost_utils.broadcast_one_to_all(origin)
        binding_id = origin.tobytes().hex()
        policy_step = int(state.updates)
        slots = []
        for index, task in enumerate(tasks):
            for group in range(self.groups):
                sample = (rank * tasks.size + index) * self.groups + group
                draw = jax.random.fold_in(key, sample)
                slots.append(_Session(EpisodeId(int(task), int(state.step), int(sample),
                    tuple(int(value) for value in np.asarray(jax.random.key_data(draw))))))
        error = None
        try:
            with ExitStack() as journal_stack:
                run = None
                if self.journal is not None:
                    journal = self.journal
                    def open_journal():
                        from .journal import policy_digest

                        cohort = repr((int(state.step), tuple(np.asarray(jax.random.key_data(key)).tolist())))
                        fingerprint = repr((signature.tobytes().hex(), tasks.tolist(), processes, rank,
                                            policy_digest(state.params)))
                        return journal_stack.enter_context(journal.open(cohort, fingerprint, binding_id))
                    run = _phase(open_journal, "episode journal open")
                    origin = np.frombuffer(bytes.fromhex(run.binding), np.uint8)
                    if processes > 1:
                        origin = multihost_utils.broadcast_one_to_all(origin)
                    binding_id = origin.tobytes().hex()
                    _phase(lambda: run.align(binding_id), "episode journal binding")
                with ExitStack() as stack:
                    _phase(lambda: self._open(slots, stack, run, policy_step, binding_id), "episode reset")
                    for turn in range(self.max_turns):
                        inputs = _phase(lambda: self._inputs(slots, turn), "episode context preparation")
                        active = any(slot.status == EpisodeStatus.RUNNING for slot in slots)
                        if not agree_process_phase(None, phase="episode availability", available=active):
                            break
                        result = _phase(lambda: policy(inputs, self.max_new_tokens,
                            key=jax.random.fold_in(key, turn), sampling=self.sampling), "episode generation")
                        _phase(lambda: self._advance(slots, result, policy_step, binding_id, turn, run), "episode tool step")
                    _phase(lambda: self._verify(slots, policy_step, binding_id, run), "episode verification")
        except BaseException as failure:
            error = failure
        try:
            agree_process_phase(error, phase="episode resource release")
        except BaseException as failure:
            self._failed(slots, failure, policy_step, binding_id)
        records = tuple(slot.episode(policy_step, binding_id) for slot in slots)
        def record() -> None:
            if self.record is not None:
                for episode in records:
                    self.record(episode)
        _phase(record, "episode recording")
        return records


    def __call__(self, state: TrainState, batch: Mapping[str, object], key: jax.Array) -> dict[str, np.ndarray]:
        episodes = self.collect(state, batch, key)
        return _phase(lambda: self.project(episodes), "episode projection")

    def project(self, episodes: Sequence[Episode]) -> dict[str, np.ndarray]:
        """GRPO action rows with one group-relative advantage per episode."""
        batch = self.tensors(episodes)
        rewards = jnp.asarray([episode.reward for episode in episodes], jnp.float32)
        advantages = np.asarray(group_advantage(rewards, self.groups))
        batch[ADVANTAGES_KEY] = np.broadcast_to(np.repeat(advantages, self.max_turns)[:, None],
                                               batch[RESPONSE_MASK_KEY].shape)
        return batch

    def tensors(self, episodes: Sequence[Episode]) -> dict[str, np.ndarray]:
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

            "episode_status": np.repeat(np.asarray([episode.status for episode in episodes], np.int32), self.max_turns),
            "task_id": np.repeat(np.asarray([episode.identity.task for episode in episodes], np.int32), self.max_turns),
            "policy_step": np.full(rows, policy_step, np.int32),
        }
