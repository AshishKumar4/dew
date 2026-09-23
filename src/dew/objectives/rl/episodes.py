"""Tool episodes collected outside jit and projected into ordinary GRPO rows.

Execution belongs to a user-supplied, context-managed environment. Dew has
no default executor. Each model call retains its exact context and sampled
likelihoods; observations are inputs to later calls and never loss targets.
"""

from __future__ import annotations

import hashlib
import math
from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from dew.artifacts import PeerFailure, agree_process_phase, agreed
from dew.nn.inputs import ModelInputs, local_rows
from dew.objectives.base import Batch, Variables
from dew.sampling.text import Generation, Sampling
from dew.training.state import TrainState

from .sessions import OLD_LOG_PROBS_KEY, Call, Session, Status, pack, sampled_values

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
    """Identify one sample within an attempt, reproducibly from checkpointed work.

    The harness can use this identity for its own idempotency records. Dew
    does not guarantee exactly-once external effects across process failure.
    """

    task: int
    attempt: int
    sample: int
    seed: tuple[int, ...]


@dataclass(frozen=True)
class Observation:
    """Carry the exact next model context, or a terminal environment result.

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
    """Record one actual model call, including EOS and both likelihood distributions.

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
        stops = self.sampling.stops
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
    """Restore tool state from an opaque snapshot, without replaying completed calls."""

    def get_state(self) -> bytes: ...

    def set_state(self, state: bytes) -> None: ...

EnvironmentFactory = Callable[[EpisodeId], AbstractContextManager[Environment]]
Verifier = Callable[[Episode], float]
EpisodeRecorder = Callable[[Episode], None]


class EpisodeInference(Protocol):
    """Draw actions from a bound policy, returning the actual sampling likelihoods."""

    def bind(self, variables: Variables, /) -> EpisodeInference: ...

    def __call__(self, inputs: ModelInputs | Sequence[Sequence[int]], max_new_tokens: int, /,
                 *, key: jax.Array, sampling: Sampling) -> Generation: ...


class EpisodeFailure(RuntimeError):
    """Report a failed collection, keeping the partial episode available to the caller."""

    def __init__(self, episode: Episode):
        self.episode = episode
        super().__init__(f"episode {episode.identity}: {episode.status.name.lower()}: {episode.detail}")


class EpisodeCancelled(CancelledError):
    def __init__(self, episode: Episode):
        self.episode = episode
        super().__init__(f"episode {episode.identity} cancelled: {episode.detail}")


_T = TypeVar("_T")
_P = ParamSpec("_P")


class _StatusStop(RuntimeError):
    """An environment returned an explicit error or cancellation outcome."""


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
        """Call `operation`, remembering the error it raised on this slot.

        The abort path names the slot that failed first, which is what
        separates the one true error from the cancellations beside it.
        """
        try:
            return operation(*args, **kwargs)
        except BaseException as error:
            self.error = error
            raise

    def observe(self, observation: Observation) -> None:
        """Adopt an environment's observation as this slot's current state.

        An explicit error or cancellation stops the slot here, so the
        cohort aborts instead of sampling another turn from it.
        """
        if not isinstance(observation, Observation):
            raise TypeError("environment methods must return an Observation")
        self.observation = observation
        self.status, self.detail = observation.status, observation.detail
        if self.status in (EpisodeStatus.ERROR, EpisodeStatus.CANCELLED):
            raise _StatusStop(self.detail)

    def episode(self, policy_step: int, binding_id: str) -> Episode:
        """Freeze this slot's progress into an `Episode` record."""
        return Episode(self.identity, policy_step, self.initial, tuple(self.transitions),
                       self.status, self.detail, self.reward, _binding_id=binding_id)


def turn_limit(observation: Observation, turn: int, *, max_turns: int, max_prompt_tokens: int) -> Observation | None:
    """The truncation that ends a running episode before its call `turn`, or None to draw it.

    Every episode driver applies these limits: `turn` counts calls already
    made, and a context the model cannot read is not drawn from.
    """
    if turn >= max_turns:
        return Observation((), EpisodeStatus.TRUNCATED, "episode turn limit reached")
    if len(observation.context) > max_prompt_tokens:
        return Observation((), EpisodeStatus.TRUNCATED, "next context exceeds max_prompt_tokens")
    return None


def step_action(environment: Environment, action: Action) -> Observation:
    """Step the environment with an action that ended on EOS; one that hit its token limit truncates instead."""
    if action.terminated:
        return environment.step(action)
    return Observation((), EpisodeStatus.TRUNCATED, "model turn reached its token limit")


@dataclass(frozen=True)
class EpisodeRollout:
    """Collect complete episode groups under one policy snapshot, then train on their actions.

    Input batches contain integer task_id rows. The environment factory
    resolves each task and owns its tools, timeouts and isolation. A finite
    verifier reward is required for completed and truncated episodes. Errors
    and cancellation abort the whole group before a Trainer update; record
    receives the partial episode before the exception propagates.

    Episodes train through `sessions.pack`: a call whose context extends the
    previous call's context and actions merges into its chain, and chains
    share rows of max_prompt_tokens + max_new_tokens ids, so set the
    objective's seq_len one below that. The batch keeps one row per possible
    call, which always fits and keeps shapes fixed. The terminal group
    advantage is shared by the episode's actions; no per-turn credit rule is
    inferred, and truncated episodes are masked. Host records and numeric
    rows carry the trainer's committed update clock. Raw and behavior likelihoods come from actual draws.

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
        """Commit one slot's turn and environment snapshot, when a journal is open."""
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
        """Enter every slot's environment and put it at its starting observation.

        A journalled slot with a saved turn restores that snapshot instead
        of resetting, so a resumed cohort never repeats a completed call.
        """
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
        """Pack this turn's contexts into one right-aligned cohort of rows.

        A slot that is finished, or whose context does not fit, still
        occupies its row: the shapes have to match across ranks, and its
        draw is discarded rather than skipped.
        """
        tokens = np.full((len(slots), self.max_prompt_tokens), self.sampling.pad_id, np.int32)
        valid = np.zeros_like(tokens, bool)
        for row, slot in enumerate(slots):
            if slot.status == EpisodeStatus.RUNNING and len(slot.transitions) == turn and slot.pending is None:
                observation = slot.observation
                assert observation is not None
                length = len(observation.context)
                limit = turn_limit(observation, turn, max_turns=self.max_turns,
                                   max_prompt_tokens=self.max_prompt_tokens)
                if limit is not None:
                    slot.status, slot.detail = limit.status, limit.detail
                else:
                    tokens[row, -length:] = observation.context
                    valid[row, -length:] = True
            if not valid[row].any():
                # Keep global row indices and collective shapes stable. The
                # draw for a finished slot is discarded, never executed or trained.
                valid[row, -1] = True
        # Contexts that fill the width leave no padded slot, and a cohort of
        # those carries no validity field at all.
        fields = {} if valid.all() else {"attention_mask": jnp.asarray(valid)}
        return ModelInputs(jnp.asarray(tokens), fields)

    def _advance(self, slots: list[_Session], generation: Generation, policy_step: int,
                 binding_id: str, turn: int, run: JournalRun | None) -> None:
        """Record this turn's draws, then step each slot's environment with them.

        A model turn that stopped on its token limit instead of EOS
        truncates the episode rather than calling the environment.
        """
        if not isinstance(generation, Generation):
            raise TypeError("tool episodes require an autoregressive Generation")
        rows = generation.host()
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
            observation = slot.invoke(step_action, environment, action)
            if not isinstance(observation, Observation):
                slot.invoke(slot.observe, observation)
            slot.transitions.append(Transition(action, observation))
            slot.pending = None
            slot.invoke(slot.observe, observation)
            self._persist(slot, run, policy_step, binding_id)

    def _verify(self, slots: list[_Session], policy_step: int, binding_id: str, run: JournalRun | None) -> None:
        """Score every unscored slot with the verifier and persist the reward.

        A slot still running when the turn budget ran out is truncated
        first, so it is scored as the episode it actually produced.
        """
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
        """Abort the whole cohort and raise for the slot that failed first.

        One slot carries the error and the rest are cancelled, so a
        partial group never reaches an update. Every started slot is
        recorded before the exception propagates.
        """
        started = [slot for slot in slots if slot.environment is not None or slot.error is not None]
        if not started:
            raise error
        primary = next((slot for slot in started if slot.error is not None), started[0])
        cancelled = isinstance(error, (AsyncCancelledError, CancelledError, KeyboardInterrupt, PeerFailure))
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
        agreed("episode abort recording", record)
        episode = records[started.index(primary)]
        failure = EpisodeCancelled(episode) if cancelled else EpisodeFailure(episode)
        if explicit:
            raise failure from None
        raise failure from error

    def _action(self, generation: Generation[np.ndarray], row: int, context: tuple[int, ...], policy_step: int,
                binding_id: str) -> Action:
        """Read one cohort row as an `Action`, validating its provenance first.

        The environment acts on this record, so the row has to carry back
        the exact context that was requested.
        """
        tokens = np.asarray(generation.tokens)[row]
        lengths, ended = np.asarray(generation.lengths), np.asarray(generation.terminated)
        raw, behavior = np.asarray(generation.raw_log_probs)[row], np.asarray(generation.behavior_log_probs)[row]
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

    def _slots(self, tasks, state: TrainState, key: jax.Array, rank: int) -> list[_Session]:
        """Build one session per task and group, each with its own identity.

        The identity is drawn from the cohort key and the sample's global
        index, so a rank's slots are reproducible from checkpointed work.
        """
        slots = []
        for index, task in enumerate(tasks):
            for group in range(self.groups):
                sample = (rank * tasks.size + index) * self.groups + group
                draw = jax.random.fold_in(key, sample)
                slots.append(_Session(EpisodeId(int(task), int(state.step), int(sample),
                    tuple(int(value) for value in np.asarray(jax.random.key_data(draw))))))
        return slots

    def _bound_journal(self, journal_stack: ExitStack, state: TrainState, key: jax.Array, tasks,
                       signature, processes: int, rank: int, binding: str) -> tuple[JournalRun, str]:
        """Open this cohort's journal and agree on the binding it records under.

        A journal that already holds this cohort's turns keeps its own
        binding, so the resumed run stays the collection the stored actions
        were drawn from.
        """
        journal = self.journal
        assert journal is not None

        def open_journal():
            from .journal import policy_digest

            cohort = repr((int(state.step), tuple(np.asarray(jax.random.key_data(key)).tolist())))
            fingerprint = repr((signature.tobytes().hex(), tasks.tolist(), processes, rank,
                                policy_digest(state.params)))
            return journal_stack.enter_context(journal.open(cohort, fingerprint, binding))

        run = agreed("episode journal open", open_journal)
        origin = np.frombuffer(bytes.fromhex(run.binding), np.uint8)
        if processes > 1:
            origin = multihost_utils.broadcast_one_to_all(origin)
        bound = origin.tobytes().hex()
        agreed("episode journal binding", lambda: run.align(bound))
        return run, bound

    def _turns(self, slots: list[_Session], policy: EpisodeInference, key: jax.Array,
               run: JournalRun | None, policy_step: int, binding_id: str) -> None:
        """Run the cohort turn by turn until every rank's slots are finished.

        A turn packs the contexts, agrees on whether any slot is still
        running, generates for the whole cohort and steps the environments.
        Every environment entered here is released before returning, and
        the slots are verified once the loop ends.
        """
        with ExitStack() as stack:
            agreed("episode reset", lambda: self._open(slots, stack, run, policy_step, binding_id))
            for turn in range(self.max_turns):
                inputs = agreed("episode context preparation", lambda: self._inputs(slots, turn))
                active = any(slot.status == EpisodeStatus.RUNNING for slot in slots)
                if not agree_process_phase(None, phase="episode availability", available=active):
                    break
                generation = agreed("episode generation", lambda: policy(inputs, self.max_new_tokens,
                    key=jax.random.fold_in(key, turn), sampling=self.sampling))
                agreed("episode tool step", lambda: self._advance(slots, generation, policy_step, binding_id, turn, run))
            agreed("episode verification", lambda: self._verify(slots, policy_step, binding_id, run))

    def collect(self, state: TrainState, batch: Batch, key: jax.Array) -> tuple[Episode, ...]:
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
        tasks, policy, signature = agreed("episode preparation", prepare)
        processes, rank = jax.process_count(), jax.process_index()
        if processes > 1:
            multihost_utils.assert_equal(signature, "episode task counts, budgets, sampling and clocks must agree")
        origin = np.frombuffer(uuid4().bytes, np.uint8) if rank == 0 else np.zeros(16, np.uint8)
        if processes > 1:
            origin = multihost_utils.broadcast_one_to_all(origin)
        binding_id = origin.tobytes().hex()
        policy_step = int(state.updates)
        slots = self._slots(tasks, state, key, rank)
        error = None
        try:
            with ExitStack() as journal_stack:
                run = None
                if self.journal is not None:
                    run, binding_id = self._bound_journal(
                        journal_stack, state, key, tasks, signature, processes, rank, binding_id)
                self._turns(slots, policy, key, run, policy_step, binding_id)
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
        agreed("episode recording", record)
        return records


    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> dict[str, np.ndarray]:
        episodes = self.collect(state, batch, key)
        return agreed("episode projection", lambda: self.project(episodes))

    def project(self, episodes: Sequence[Episode]) -> dict[str, np.ndarray]:
        """Pack one collection's episodes into GRPO rows through `sessions.pack`.

        Each episode becomes a `Session` (`session_of`), so its calls merge
        into one chain wherever the environment's next context extends the
        previous one, and the chains share `[rows, width]` rows with segment
        ids. `old_log_probs` carries the sampler's raw likelihoods, recorded
        under the same snapshot. A truncated episode is masked, not scored.

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
        sessions = [session_of(episode, group=str(index // self.groups))
                    for index, episode in enumerate(episodes)]
        batch = pack(sessions, self.max_prompt_tokens + self.max_new_tokens,
                     rows=len(episodes) * self.max_turns)
        batch[OLD_LOG_PROBS_KEY] = sampled_values(
            batch, lambda index, number: episodes[index].transitions[number].action.raw_log_probs)
        return batch


def session_of(episode: Episode, *, group: str) -> Session:
    """Read an episode as an engine-style `Session` of the advantage group `group`.

    Each transition's action is one call: its context is the prompt, its
    tokens the sampled ids with their behavior likelihoods, `stop` when it
    ended on EOS and `length` otherwise. An environment-reported error is an
    infrastructure failure; how well the agent did is the verifier's reward.
    """
    if episode.status == EpisodeStatus.RUNNING:
        raise ValueError("a running episode has not ended, so it is no session yet")
    status = {EpisodeStatus.COMPLETED: Status.COMPLETED, EpisodeStatus.TRUNCATED: Status.TRUNCATED,
              EpisodeStatus.ERROR: Status.INFRA_ERROR, EpisodeStatus.CANCELLED: Status.CANCELLED}[episode.status]
    calls = tuple(Call(turn.action.context, turn.action.tokens, turn.action.behavior_log_probs,
                       "stop" if turn.action.terminated else "length", turn.action.policy_step)
                  for turn in episode.transitions)
    identity = episode.identity
    return Session(str(identity.task), group, identity.sample, identity.attempt, calls, status,
                   episode.reward, detail=episode.detail)
