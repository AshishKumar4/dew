"""Tool episodes collected outside jit, each a session the rollout scheduler packs.

Execution belongs to a user-supplied, context-managed environment. Dew has
no default executor. Each model call retains its exact context and sampled
likelihoods; observations are inputs to later calls and never loss targets.
"""

from __future__ import annotations

import hashlib
from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np

from dew._coordination import (
    PeerFailure,
    agree_process_phase,
    agreed,
    agreed_same,
    broadcast_from_process_zero,
)
from dew.batch import Variables, token_id
from dew.data.dataset import DataPartition
from dew.inference.rollouts import Draw
from dew.nn.inputs import ModelInputs, mesh_of, pad_token_rows
from dew.sampling.text import Generation, Sampling

from .sessions import Call, Session, Snapshot, Status, Task, Transcript, finite_reward, json_objects

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
    """The identity of one sample within an attempt, reproducible from checkpointed work.

    `task` is the task id, `attempt` the trainer step that collected the
    sample, `sample` its global index, `seed` the key data drawn for it, and
    `branch` the cooldown branch that drew it (`Snapshot.branch`), None for
    the run's own: a branch draws at the run's steps and keys under other
    weights, and its episodes are its own.
    The harness can use this identity for its own idempotency records. Dew
    does not guarantee exactly-once external effects across process failure.
    """

    task: int
    attempt: int
    sample: int
    seed: tuple[int, ...]
    branch: str | None = None


@dataclass(frozen=True)
class Observation:
    """The exact next model context, or a terminal environment result.

    The harness handles chat formatting, tool-call parsing and context
    compaction. A terminal observation's context may be empty, but a running
    one needs a context. `detail` can hold a verifier result or an artifact
    reference without encoding it into device arrays. `reward` is what the
    environment paid for the action that led here, a step reward, or None
    where it pays only through the verifier at the end. `messages` are the
    chat messages, as JSON, the environment's context holds that it added
    here (a tool's result, a user's turn), which each later call's
    `Transcript` lists; the context's ids stay what the model reads.
    """

    context: tuple[int, ...]
    status: EpisodeStatus = EpisodeStatus.RUNNING
    detail: str = ""
    reward: float | None = None
    messages: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, EpisodeStatus):
            raise TypeError("observation status must be an EpisodeStatus")
        if self.reward is not None:
            object.__setattr__(self, "reward", finite_reward(self.reward))
        object.__setattr__(self, "messages", json_objects(self.messages, "an observation's messages"))
        object.__setattr__(self, "context", tuple(token_id(token, where="an observation's context")
                                                  for token in self.context))
        if self.status == EpisodeStatus.RUNNING and not self.context:
            raise ValueError("a running environment must supply a model context")


@dataclass(frozen=True)
class Transition:
    """One model call of an episode, the `Draw` its environment acted on, and the observation that
    followed it."""

    action: Draw
    observation: Observation


@dataclass(frozen=True)
class Episode:
    """One episode as its driver ran it: its identity, the policy version every call was drawn
    under, the `sampling` policy that drew them, the environment's first observation, every call
    with the observation after it, and how it ended.

    An action ends as `sampling` ends a draw (`dew.inference.rollouts.check_ending`). EOS or a
    stop string ends the model turn, not the episode."""

    identity: EpisodeId
    policy_step: int
    sampling: Sampling
    initial: Observation | None
    transitions: tuple[Transition, ...]
    status: EpisodeStatus
    detail: str = ""
    reward: float | None = None
    # A collection binds exactly once. Its private origin is independent of
    # clocks, which can coincide across different runs or parameter trees.
    # Exclude the origin from value equality so replayed numerical records
    # remain comparable; the journal checks it before resuming a cohort.
    _binding_id: str = field(default="", repr=False, compare=False, kw_only=True)


class Environment(Protocol):
    """Executes one episode's tool calls for the caller.

    `reset` returns the first observation, and `step` returns the
    observation that follows an action. The caller provides each
    environment through a context manager and releases its resources when
    the context exits.
    """

    def reset(self) -> Observation: ...

    def step(self, action: Draw) -> Observation: ...


@runtime_checkable
class RecoverableEnvironment(Environment, Protocol):
    """Saves and restores its tool state as an opaque snapshot, so a resume replays no completed call.

    `get_state` returns the snapshot as bytes, and `set_state` restores it.
    `EpisodeJournal` needs an environment of this kind.
    """

    def get_state(self) -> bytes: ...

    def set_state(self, state: bytes) -> None: ...

EnvironmentFactory = Callable[[EpisodeId], AbstractContextManager[Environment]]
Verifier = Callable[[Episode], float]
EpisodeRecorder = Callable[[Episode], None]


class EpisodeInference(Protocol):
    """Draws actions from a bound policy and returns the actual sampling likelihoods.

    `bind(variables)` returns an inference that generates from that one
    weight snapshot.
    """

    def bind(self, variables: Variables, /) -> EpisodeInference: ...

    def __call__(self, inputs: ModelInputs | Sequence[Sequence[int]], max_new_tokens: int, /,
                 *, key: jax.Array, sampling: Sampling) -> Generation: ...


class EpisodeFailure(RuntimeError):
    """Reports a failed collection and keeps the partial episode in `episode` for the caller."""

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
class EpisodeProgress:
    """One episode while a driver runs it: what its environment showed, its calls and how it ended.

    `EpisodeCollector` advances a cohort of these in lock step, and
    `EnvironmentSource` one per session. Both adopt observations, apply the
    turn and context limits, record each call with the observation that
    followed it and score finished episodes through it. Each keeps its own
    failure policy.
    """

    identity: EpisodeId
    sampling: Sampling
    environment: Environment | None = None
    initial: Observation | None = None
    observation: Observation | None = None
    transitions: list[Transition] = field(default_factory=list)
    pending: Draw | None = None
    """A call already drawn whose observation is not in yet."""
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
        """Adopt an environment's observation as this episode's current state.

        Anything but an `Observation` raises TypeError.
        """
        if not isinstance(observation, Observation):
            raise TypeError("environment methods must return an Observation")
        self.observation = observation
        self.status, self.detail = observation.status, observation.detail

    def begin(self, observation: Observation) -> None:
        """Adopt the environment's first observation."""
        self.observe(observation)
        self.initial = observation

    def next_context(self, *, max_prompt_tokens: int) -> tuple[int, ...] | None:
        """Return the context of the episode's next call, or None when it makes no more calls.

        A context longer than `max_prompt_tokens` truncates the episode here
        instead of being drawn from. The driver bounds the number of calls,
        and `score` truncates an episode that is still running after its
        last one.
        """
        if self.status != EpisodeStatus.RUNNING:
            return None
        observation = self.observation
        assert observation is not None
        if len(observation.context) > max_prompt_tokens:
            self.stop(EpisodeStatus.TRUNCATED, "next context exceeds max_prompt_tokens")
            return None
        return observation.context

    def commit(self, action: Draw, observation: Observation) -> None:
        """Record `action`, the pending call, with the observation that followed it, and adopt that."""
        self.observe(observation)
        self.transitions.append(Transition(action, observation))
        self.pending = None

    def stop(self, status: EpisodeStatus, detail: str) -> None:
        """End the episode at `status` without an observation, as a limit or a driver does."""
        self.status, self.detail = status, detail

    def abort(self, status: EpisodeStatus, detail: str) -> None:
        """End the episode at a failure `status`, unscored. A pending call stays
        on the record with the failure it met."""
        self.stop(status, detail)
        self.reward = None
        if self.pending is not None:
            self.transitions.append(Transition(self.pending, Observation((), status, detail)))
            self.pending = None

    def score(self, verifier: Verifier, policy_step: int, binding_id: str) -> None:
        """Score the finished episode with `verifier`.

        An episode still running made its last allowed call, so it is
        truncated first and scored as the episode it produced. A
        non-finite reward raises ValueError.
        """
        if self.status == EpisodeStatus.RUNNING:
            self.stop(EpisodeStatus.TRUNCATED, "episode turn limit reached")
        self.reward = finite_reward(verifier(self.episode(policy_step, binding_id)))

    def episode(self, policy_step: int, binding_id: str) -> Episode:
        """Freeze this slot's progress into an `Episode` record."""
        return Episode(self.identity, policy_step, self.sampling, self.initial, tuple(self.transitions),
                       self.status, self.detail, self.reward, _binding_id=binding_id)


def _halted(slot: EpisodeProgress) -> None:
    """Abort the cohort at an explicit error or cancellation, instead of sampling another turn."""
    if slot.status in (EpisodeStatus.ERROR, EpisodeStatus.CANCELLED):
        raise _StatusStop(slot.detail)


def step_action(environment: Environment, action: Draw) -> Observation:
    """Step the environment with an action that ended its turn, on EOS or a
    stop string; one that hit its token limit truncates instead."""
    if action.terminated:
        return environment.step(action)
    return Observation((), EpisodeStatus.TRUNCATED, "model turn reached its token limit")


@dataclass(frozen=True)
class EpisodeCollector:
    """Collects tool episodes in lock step under the trainer's own snapshot (`sessions.Collector`).

    Its tasks are integer task ids (`scheduler.task_ids`). The `environment`
    factory resolves each task and is responsible for its tools, timeouts
    and isolation. Completed and truncated episodes need a finite reward
    from `verifier`. An error or a cancellation aborts the whole cohort
    before a `Trainer` update, and `record` receives the partial episodes
    before the exception propagates. `sampling` must set an EOS id or a stop
    string.

    A call whose context extends the previous call's context and actions
    merges into that call's chain (`sessions.pack`), and chains share rows
    of `max_prompt_tokens + max_new_tokens` ids, so set the objective's
    `seq_len` one below that. A collection packs into one row per possible
    call, so it always fits and its shapes stay fixed. An episode's return
    is the verifier's reward plus the step rewards its observations carry
    (`Observation.reward`). Each episode becomes a `Session` (`session_of`).
    Host records and numeric rows record the trainer's committed update
    count. Raw and behavior likelihoods come from the actual draws.

    `policy.bind` fixes the snapshot's variables for the whole collection,
    and the policy must generate from them, not from a mutable serving
    default. The `Trainer` cannot update or donate the tree until the
    collection returns. An `EpisodeJournal` (`journal`) saves each turn
    durably for environments that provide `get_state` and `set_state`. The
    environment itself must make pending external effects idempotent; the
    optimizer still recovers only from `Trainer` checkpoints.
    """

    policy: EpisodeInference
    environment: EnvironmentFactory
    verifier: Verifier
    max_prompt_tokens: int
    max_new_tokens: int
    max_turns: int
    sampling: Sampling
    record: EpisodeRecorder | None = None
    journal: EpisodeJournal | None = None

    def __post_init__(self) -> None:
        for name in ("max_prompt_tokens", "max_new_tokens", "max_turns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sampling.eos_token_ids is None and not self.sampling.stop:
            raise ValueError("tool episodes need an EOS id or a stop string to tell a turn that ended "
                             "from one cut at its token limit")

    @property
    def width(self) -> int:
        """A packed row's ids: the longest context and the most one turn draws."""
        return self.max_prompt_tokens + self.max_new_tokens

    def rows(self, tasks: int, samples: int) -> int:
        """One row per possible call of every episode, so a collection always fits."""
        return tasks * samples * self.max_turns

    def _persist(self, slot: EpisodeProgress, run: JournalRun | None, policy_step: int, binding: str) -> None:
        """Commit one slot's turn and environment snapshot, when a journal is open."""
        if run is not None:
            environment = slot.environment
            if not isinstance(environment, RecoverableEnvironment):
                raise TypeError("EpisodeJournal requires get_state/set_state on the environment")
            snapshot = slot.invoke(environment.get_state)
            if not isinstance(snapshot, bytes):
                raise TypeError("environment get_state must return bytes")
            slot.invoke(run.save, slot.episode(policy_step, binding), slot.pending, snapshot)

    def _open(self, slots: list[EpisodeProgress], stack: ExitStack, run: JournalRun | None,
              policy_step: int, binding: str) -> None:
        """Enter every slot's environment and put it at its starting observation.

        A journalled slot with a saved turn restores that snapshot instead
        of resetting, so a resumed cohort never repeats a completed call.
        """
        for slot in slots:
            slot.environment = slot.invoke(
                lambda slot=slot: stack.enter_context(self.environment(slot.identity))
            )
            saved = None if run is None else slot.invoke(run.load, slot.identity)
            if saved is None:
                observation = slot.invoke(slot.environment.reset)
                slot.invoke(slot.begin, observation)
                slot.invoke(_halted, slot)
                self._persist(slot, run, policy_step, binding)
            else:
                if not isinstance(slot.environment, RecoverableEnvironment):
                    raise TypeError("EpisodeJournal requires get_state/set_state on the environment")
                slot.invoke(slot.environment.set_state, saved.snapshot)
                episode = saved.episode
                slot.initial, slot.transitions = episode.initial, list(episode.transitions)
                slot.pending, slot.reward = saved.pending, episode.reward
                slot.observation = (
                    episode.transitions[-1].observation if episode.transitions else episode.initial
                )
                slot.status, slot.detail = episode.status, episode.detail

    def _inputs(self, slots: list[EpisodeProgress], turn: int) -> ModelInputs:
        """Pack this turn's contexts into one right-aligned cohort of rows.

        A slot that is finished, or whose context does not fit, still
        occupies its row: the shapes have to match across ranks, and its
        draw is discarded rather than skipped.
        """
        contexts = []
        for slot in slots:
            context = None
            if len(slot.transitions) == turn and slot.pending is None:
                context = slot.next_context(max_prompt_tokens=self.max_prompt_tokens)
            # A finished slot keeps its row, one padding id, so global row indices and collective
            # shapes stay stable. Its draw is discarded, never executed or trained.
            contexts.append([self.sampling.pad] if context is None or not len(context) else context)
        tokens, fields = pad_token_rows(contexts, pad_id=self.sampling.pad, width=self.max_prompt_tokens)
        return ModelInputs(jnp.asarray(tokens), {name: jnp.asarray(value) for name, value in fields.items()})

    def _advance(self, slots: list[EpisodeProgress], generation: Generation, policy_step: int,
                 binding_id: str, turn: int, run: JournalRun | None) -> None:
        """Record this turn's draws, then step each slot's environment with them.

        A model turn that stopped on its token limit instead of EOS or a stop
        string truncates the episode rather than calling the environment.
        """
        if not isinstance(generation, Generation):
            raise TypeError("tool episodes require an autoregressive Generation")
        rows = generation.host()
        if rows.tokens.shape[0] != len(slots):
            raise TypeError("tool episodes require one autoregressive Generation row per slot")
        # Record every actual draw before invoking any tool. A tool failure
        # must not erase the other requests already sampled in this cohort.
        for row, slot in enumerate(slots):
            if (
                slot.status == EpisodeStatus.RUNNING
                and len(slot.transitions) == turn
                and slot.pending is None
            ):
                observation = slot.observation
                assert observation is not None
                slot.pending = slot.invoke(self._action, rows, row, observation.context, policy_step)
                self._persist(slot, run, policy_step, binding_id)
        for slot in slots:
            action = slot.pending
            if action is None or len(slot.transitions) != turn:
                continue
            environment = slot.environment
            assert environment is not None
            observation = slot.invoke(step_action, environment, action)
            slot.invoke(slot.commit, action, observation)
            slot.invoke(_halted, slot)
            self._persist(slot, run, policy_step, binding_id)

    def _verify(
        self, slots: list[EpisodeProgress], policy_step: int, binding_id: str, run: JournalRun | None
    ) -> None:
        """Score every unscored slot with the verifier and persist the reward.

        A slot still running when the turn budget ran out is truncated
        first, so it is scored as the episode it actually produced.
        """
        for slot in slots:
            if slot.reward is None:
                slot.invoke(slot.score, self.verifier, policy_step, binding_id)
                self._persist(slot, run, policy_step, binding_id)

    def _failed(self, slots: list[EpisodeProgress], error: BaseException,
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
            slot.abort(EpisodeStatus.ERROR if slot is primary and not cancelled else EpisodeStatus.CANCELLED,
                       str(error) if explicit else f"{type(error).__name__}: {error}")
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

    def _action(
        self,
        generation: Generation[np.ndarray],
        row: int,
        context: tuple[int, ...],
        policy_step: int,
    ) -> Draw:
        """Read one cohort row as a `Draw`, validating its provenance first.

        The environment acts on this record, so the row has to carry back
        the exact context that was requested.
        """
        tokens = np.asarray(generation.tokens)[row]
        lengths, ended = np.asarray(generation.lengths), np.asarray(generation.terminated)
        raw, behavior = (
            np.asarray(generation.raw_log_probs)[row],
            np.asarray(generation.behavior_log_probs)[row],
        )
        width = self.max_prompt_tokens
        expected = np.full(width, self.sampling.pad, np.int32)
        expected[-len(context):] = context
        if (tokens.shape != (width + self.max_new_tokens,)
                or not np.issubdtype(tokens.dtype, np.integer)
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
        return Draw(context, actions, tuple(float(value) for value in behavior[:count]),
                    "stop" if bool(ended[row]) else "length", policy_step,
                    tuple(float(value) for value in raw[:count])).checked(self.sampling)

    def _slots(self, tasks: np.ndarray, samples: int, snapshot: Snapshot,
               rank: int) -> list[EpisodeProgress]:
        """Build one session per task and sample, each with its own identity.

        The identity is drawn from the cohort key and the sample's global
        index, so a rank's slots are reproducible from checkpointed work.
        """
        slots = []
        for index, task in enumerate(tasks):
            for group in range(samples):
                sample = (rank * tasks.size + index) * samples + group
                draw = jax.random.fold_in(snapshot.key, sample)
                slots.append(EpisodeProgress(EpisodeId(int(task), snapshot.step, int(sample),
                    tuple(int(value) for value in np.asarray(jax.random.key_data(draw))), snapshot.branch),
                    self.sampling))
        return slots

    def _bound_journal(self, journal_stack: ExitStack, snapshot: Snapshot, tasks: np.ndarray,
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

            clock = (snapshot.step, tuple(np.asarray(jax.random.key_data(snapshot.key)).tolist()))
            # A cooldown branch's cohorts are its own, apart from the run's at the same step and key.
            cohort = repr(clock if snapshot.branch is None else (snapshot.branch, *clock))
            fingerprint = repr((signature.tobytes().hex(), tasks.tolist(), processes, rank,
                                policy_digest(snapshot.variables)))
            return journal_stack.enter_context(journal.open(cohort, fingerprint, binding))

        run = agreed("episode journal open", open_journal)
        bound = broadcast_from_process_zero(run.binding)
        agreed("episode journal binding", lambda: run.align(bound))
        return run, bound

    def _turns(self, slots: list[EpisodeProgress], policy: EpisodeInference, key: jax.Array,
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
                inputs = agreed("episode context preparation", lambda turn=turn: self._inputs(slots, turn))
                active = any(slot.status == EpisodeStatus.RUNNING for slot in slots)
                if not agree_process_phase(None, phase="episode availability", available=active):
                    break
                generation = agreed("episode generation", lambda inputs=inputs, turn=turn: policy(
                    inputs, self.max_new_tokens,
                    key=jax.random.fold_in(key, turn), sampling=self.sampling))
                agreed("episode tool step", lambda generation=generation, turn=turn: self._advance(
                    slots, generation, policy_step, binding_id, turn, run))
            agreed("episode verification", lambda: self._verify(slots, policy_step, binding_id, run))

    def collect(self, snapshot: Snapshot, tasks: Sequence[Task], samples: int) -> list[list[Session]]:
        """`samples` episodes of each task under `snapshot` (`episodes`), as sessions by task
        (`session_of`)."""
        sessions = [session_of(episode, group="") for episode in self.episodes(snapshot, tasks, samples)]
        return [sessions[index * samples:(index + 1) * samples] for index in range(len(tasks))]

    def episodes(self, snapshot: Snapshot, tasks: Sequence[Task], samples: int) -> tuple[Episode, ...]:
        """Collect `samples` episodes of each task under `snapshot`, task by task.

        The cohorts have fixed shapes, and every process agrees on each host
        phase before each generation. `record`, when set, receives every
        episode. Replicated readers of one data share are refused before any
        environment is opened.
        """
        def prepare():
            mesh = mesh_of(snapshot.variables)
            if mesh is not None and DataPartition.of(mesh).readers > 1:
                raise ValueError("native episodes do not support replicated data-share readers")
            if not all(task.id.removeprefix("-").isdigit() for task in tasks):
                raise ValueError("an episode collection's tasks are integer task ids (`task_ids`)")
            ids = np.asarray([int(task.id) for task in tasks], np.int64)
            policy = self.policy.bind(snapshot.variables)
            key_words = tuple(np.asarray(jax.random.key_data(snapshot.key)).tolist())
            signature = (ids.size, samples, self.max_turns, self.max_prompt_tokens, self.max_new_tokens,
                         self.sampling, snapshot.step, snapshot.updates, key_words, self.journal is not None)
            return ids, policy, repr(signature)
        ids, policy, described = agreed("episode preparation", prepare)
        described = agreed_same("episode cohort (task counts, budgets, sampling and clocks)",
                                lambda: described)
        signature = np.frombuffer(hashlib.sha256(described.encode()).digest(), np.uint8)
        processes, rank = jax.process_count(), jax.process_index()
        binding_id = broadcast_from_process_zero(uuid4().hex if rank == 0 else None)
        policy_step = snapshot.updates
        slots = self._slots(ids, samples, snapshot, rank)
        error = None
        try:
            with ExitStack() as journal_stack:
                run = None
                if self.journal is not None:
                    run, binding_id = self._bound_journal(
                        journal_stack, snapshot, ids, signature, processes, rank, binding_id)
                self._turns(slots, policy, snapshot.key, run, policy_step, binding_id)
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


def session_of(episode: Episode, *, group: str) -> Session:
    """Return `episode` as an engine-style `Session` in the advantage group `group`.

    Each transition's draw becomes one call (`Call.of`), its reward the step
    reward the observation after it carries and its transcript the messages
    the observations before it carried (none where they carried none). An
    error the environment reports becomes `INFRA_ERROR`, because the
    verifier's reward, not the status, says how well the agent did.
    """
    status = {
        EpisodeStatus.COMPLETED: Status.COMPLETED,
        EpisodeStatus.TRUNCATED: Status.TRUNCATED,
        EpisodeStatus.ERROR: Status.INFRA_ERROR,
        EpisodeStatus.CANCELLED: Status.CANCELLED,
    }[episode.status]
    said = list(() if episode.initial is None else episode.initial.messages)
    calls = []
    for turn in episode.transitions:
        calls.append(Call.of(turn.action, reward=turn.observation.reward,
                             transcript=Transcript(tuple(said)) if said else None))
        said.extend(turn.observation.messages)
    identity = episode.identity
    return Session(str(identity.task), group, identity.sample, identity.attempt, tuple(calls), status,
                   episode.reward, detail=episode.detail)
