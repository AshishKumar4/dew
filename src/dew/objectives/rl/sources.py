"""Session sources over a `RolloutServer`: in-process environments and single-turn prompts.

Both implement `SessionSource` for `RolloutScheduler`, drawing every model
call from a versioned `RolloutServer` (Dew's native server, vLLM or SGLang)
one request at a time, so turns of different sessions interleave in the
engine's continuous batch instead of waiting on a lock-step cohort.

`EnvironmentSource` runs the in-process `Environment` protocol that
`EpisodeRollout` runs, one session per sample on a worker thread, with
`EpisodeRollout`'s own per-turn limits (`turn_limit`, `step_action`). Each
turn submits the observation's context and steps the environment with the
drawn action; the episode becomes a `Session` through `session_of`, the
converter `EpisodeRollout` packs with, so both paths feed one packer. Each call keeps
the version its request was submitted under, so a session that spans a
weight push carries both versions. The statuses follow the scheduler's
failure policy: an environment that raises or reports ERROR is an
infrastructure failure, retried and never scored; a context, token or turn
limit truncates; only COMPLETED and TRUNCATED sessions reach the verifier.

`PromptSource` is the single-turn case: one call per sample, scored on
decoded text by a reward on scorer threads as each draw finishes, so
verification overlaps generation.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field, replace
from uuid import uuid4

import numpy as np

from dew.inference.rollouts import Draw, RolloutServer
from dew.objectives.base import Batch

from .episodes import (
    Action,
    Environment,
    Episode,
    EpisodeId,
    EpisodeStatus,
    Observation,
    Transition,
    session_of,
    step_action,
    turn_limit,
)
from .rollout import Reward, prompt_rows
from .sessions import Call, Session, Status, Task


def _scored(value: float) -> float:
    reward = float(value)
    if not math.isfinite(reward):
        raise ValueError("the verifier returned a non-finite score")
    return reward


def _seed(*parts: int) -> int:
    return int(np.random.default_rng(list(parts)).integers(0, 2 ** 31 - 1))


def _failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


class _Refused(Exception):
    """A configuration error found inside a session; it fails the rollout's future."""


@dataclass
class _Handle:
    wake: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False


class EnvironmentSource:
    """Run `Environment` sessions against a `RolloutServer`, one worker thread each.

    `environment(task, identity)` enters one environment per session and
    `verifier(task, episode)` scores completed and truncated episodes; both
    read the task's own payload, and `identity.task` is the submission's
    serial number. `workers` bounds concurrent sessions; the server
    batches their calls. `cancel` stops a session at its next turn or
    mid-draw, never inside `environment.step`: an environment must bound its
    own step time, or a hung step holds its worker until it returns.
    Environments see the draw's raw likelihoods when the server reports
    them, and its behavior likelihoods otherwise, which are the same
    distribution only for a sampling policy without transforms.
    """

    def __init__(self, server: RolloutServer,
                 environment: Callable[[Task, EpisodeId], AbstractContextManager[Environment]],
                 verifier: Callable[[Task, Episode], float], *, max_prompt_tokens: int, max_new_tokens: int,
                 max_turns: int, workers: int = 64, seed: int = 0):
        for name, value in (("max_prompt_tokens", max_prompt_tokens), ("max_new_tokens", max_new_tokens),
                            ("max_turns", max_turns), ("workers", workers)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if server.sampling.eos_id is None:
            raise ValueError("tool episodes need an EOS token to distinguish complete and truncated actions")
        self.server, self.environment, self.verifier = server, environment, verifier
        self.max_prompt_tokens, self.max_new_tokens, self.max_turns = max_prompt_tokens, max_new_tokens, max_turns
        self.seed = seed
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dew-episode")
        self._lock = threading.Lock()
        self._handles: dict[Future[Session], _Handle] = {}
        self._serial = 0

    def submit(self, task: Task, samples: int, *, version: int) -> list[Future[Session]]:
        with self._lock:
            serial = self._serial
            self._serial += 1
        futures = []
        for sample in range(samples):
            handle = _Handle()
            identity = EpisodeId(serial, 0, sample, (self.seed, serial, sample))
            future = self._pool.submit(self._run, task, identity, version, handle)
            with self._lock:
                self._handles[future] = handle
            future.add_done_callback(self._forget)
            futures.append(future)
        return futures

    def cancel(self, futures: Sequence[Future[Session]]) -> None:
        """Stop sessions at their next turn; queued ones never start."""
        for future in futures:
            if future.cancel():
                continue
            with self._lock:
                handle = self._handles.get(future)
            if handle is not None:
                handle.cancelled = True
                handle.wake.set()

    def close(self) -> None:
        """Cancel every session and wait for the running ones to release their environments."""
        with self._lock:
            running = list(self._handles)
        self.cancel(running)
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _forget(self, future: Future[Session]) -> None:
        with self._lock:
            self._handles.pop(future, None)

    def _draw(self, context: tuple[int, ...], seed: int, handle: _Handle) -> Draw | None:
        """One call on the server, or None when the session is cancelled while it runs."""
        handle.wake.clear()
        if handle.cancelled:
            return None
        pending = self.server.submit(context, self.max_new_tokens, seed=seed)
        pending.add_done_callback(lambda _: handle.wake.set())
        handle.wake.wait()
        if handle.cancelled and not pending.done():
            return None
        return pending.result()

    def _action(self, draw: Draw, context: tuple[int, ...]) -> Action:
        sampling = self.server.sampling
        raw = draw.raw_log_probs
        if raw is None:
            if sampling.transforms():
                raise _Refused("the server reports no raw likelihoods and its sampling transforms them")
            raw = draw.behavior_log_probs
        if draw.prompt != context:
            raise _Refused("the server returned a draw for another context")
        return Action(context, draw.tokens, raw, draw.behavior_log_probs, draw.terminated, draw.version, sampling)

    def _run(self, task: Task, identity: EpisodeId, version: int, handle: _Handle) -> Session:
        initial: Observation | None = None
        transitions: list[Transition] = []
        pending: Action | None = None
        status, detail = EpisodeStatus.RUNNING, ""
        try:
            with self.environment(task, identity) as environment:
                observation = environment.reset()
                initial = observation
                for turn in range(self.max_turns + 1):
                    if not isinstance(observation, Observation):
                        raise TypeError("environment methods must return an Observation")
                    status, detail = observation.status, observation.detail
                    if status != EpisodeStatus.RUNNING:
                        break
                    limit = turn_limit(observation, turn, max_turns=self.max_turns,
                                       max_prompt_tokens=self.max_prompt_tokens)
                    if limit is not None:
                        status, detail = limit.status, limit.detail
                        break
                    draw = self._draw(observation.context, _seed(self.seed, identity.task, identity.sample, turn),
                                  handle)
                    if draw is None:
                        status, detail = EpisodeStatus.CANCELLED, "cancelled by the scheduler"
                        break
                    pending = self._action(draw, observation.context)
                    observation = step_action(environment, pending)
                    transitions.append(Transition(pending, observation))
                    pending = None
        except _Refused:
            raise
        except Exception as error:
            status, detail = EpisodeStatus.ERROR, _failure(error)
            if pending is not None:
                # The call was made; it stays on the record with the failure it met.
                transitions.append(Transition(pending, Observation((), status, detail)))
        episode = Episode(identity, version, initial, tuple(transitions), status, detail, None,
                          _binding_id=uuid4().hex)
        if status in (EpisodeStatus.COMPLETED, EpisodeStatus.TRUNCATED):
            try:
                episode = replace(episode, reward=_scored(self.verifier(task, episode)))
            except Exception as error:
                episode = replace(episode, status=EpisodeStatus.ERROR, detail=f"verifier: {_failure(error)}")
        return replace(session_of(episode, group=""), task=task.id)


def prompt_tasks(batch: Batch) -> list[Task]:
    """One single-turn task per prompt row: its ids and the three reward strings.

    The id is a digest of the prompt ids; groups, not ids, keep repeats apart.
    """
    prompts, lengths, sources, truths, infos = prompt_rows(batch)
    rows, width = prompts.shape
    tasks = []
    for row in range(rows):
        ids = tuple(int(token) for token in prompts[row, width - int(lengths[row]):])
        name = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:16]
        tasks.append(Task(name, {"prompt": ids, "source": sources[row], "truth": truths[row], "info": infos[row]}))
    return tasks


@dataclass(frozen=True)
class _Prompt:
    """What `prompt_tasks` puts in a task: the prompt ids and the reward's three strings."""

    ids: tuple[int, ...]
    source: str
    truth: str
    info: str

    @classmethod
    def of(cls, task: Task) -> _Prompt:
        ids, source, truth, extra_info = (task.data.get(name) for name in ("prompt", "source", "truth", "info"))
        if not (isinstance(ids, tuple) and isinstance(source, str) and isinstance(truth, str)
                and isinstance(extra_info, str)):
            raise TypeError(f"task {task.id!r} is not a prompt_tasks task: it needs prompt ids "
                            "and source, truth and info strings")
        return cls(ids, source, truth, extra_info)


class PromptSource:
    """Draw one completion per sample of a `prompt_tasks` task and score its decoded text.

    `reward` scores the completion with EOS excluded, on `scorers` threads.
    A draw that ends on its token budget is
    TRUNCATED and still scored; a failed draw or reward is an
    infrastructure failure. Anything else that fails resolves the rollout's
    future with the exception, so no future is left pending.
    """

    def __init__(self, server: RolloutServer, reward: Reward, *,
                 decode: Callable[[Sequence[int]], str], max_new_tokens: int, scorers: int = 16, seed: int = 0):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("a rollout generates at least one token")
        self.server, self.reward, self.decode, self.max_new_tokens = server, reward, decode, max_new_tokens
        self.seed = seed
        self._scorers = ThreadPoolExecutor(max_workers=scorers, thread_name_prefix="dew-reward")
        self._serial = 0
        self._lock = threading.Lock()

    def submit(self, task: Task, samples: int, *, version: int) -> list[Future[Session]]:
        del version  # each draw reports the version it was submitted under
        with self._lock:
            serial = self._serial
            self._serial += 1
        prompt = _Prompt.of(task)
        return [self._scored(task, prompt, self.server.submit(prompt.ids, self.max_new_tokens,
                                                              seed=_seed(self.seed, serial, k)))
                for k in range(samples)]

    def cancel(self, futures: Sequence[Future[Session]]) -> None:
        """Forget the rollouts; their draws finish on the server and are not scored."""
        for future in futures:
            future.cancel()

    def close(self) -> None:
        """Stop the reward threads; the server belongs to the caller."""
        self._scorers.shutdown(wait=True, cancel_futures=True)

    def _rollout(self, task: Task, prompt: _Prompt, drawn: Future[Draw]) -> Session:
        try:
            draw = drawn.result()
        except Exception as error:
            return Session(task.id, "", 0, 0, (), Status.INFRA_ERROR, None, {}, f"draw: {_failure(error)}")
        call = Call(draw.prompt, draw.tokens, draw.behavior_log_probs, "stop" if draw.terminated else "length",
                    draw.version)
        status = Status.COMPLETED if draw.terminated else Status.TRUNCATED
        try:
            text = self.decode(draw.tokens[:len(draw.tokens) - int(draw.terminated)])
            score = _scored(self.reward(prompt.source, text, prompt.truth, prompt.info))
        except Exception as error:
            return Session(task.id, "", 0, 0, (call,), Status.INFRA_ERROR, None, {}, f"reward: {_failure(error)}")
        return Session(task.id, "", 0, 0, (call,), status, score)

    def _scored(self, task: Task, prompt: _Prompt, drawn: Future[Draw]) -> Future[Session]:
        """The draw's future, chained into its reward on a scorer thread."""
        scored: Future[Session] = Future()

        def score() -> None:
            if scored.cancelled():
                return
            try:
                rollout = self._rollout(task, prompt, drawn)
            except BaseException as failure:
                # Not a failure of the draw or the reward: a broken source, which the scheduler raises.
                with suppress(InvalidStateError):  # cancelled while scoring
                    scored.set_exception(failure)
                return
            with suppress(InvalidStateError):  # cancelled while scoring
                scored.set_result(rollout)

        def chained(_: Future[Draw]) -> None:
            try:
                self._scorers.submit(score)
            except RuntimeError as closed:
                with suppress(InvalidStateError):
                    scored.set_exception(closed)

        drawn.add_done_callback(chained)
        return scored
