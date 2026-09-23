"""Rollout sources over a `RolloutServer`: in-process environments and single-turn prompts.

Both implement `RolloutSource` for `RolloutScheduler`, drawing every model
call from a versioned `RolloutServer` (Dew's native server, vLLM or SGLang)
one request at a time, so turns of different sessions interleave in the
engine's continuous batch instead of waiting on a lock-step cohort.

`EnvironmentSource` runs the in-process `Environment` protocol that
`EpisodeRollout` runs, one session per sample on a worker thread. Each turn
submits the observation's context and steps the environment with the drawn
action; the episode becomes a `Rollout` through `rollout_of`, the converter
`EpisodeRollout` packs with, so both paths feed one packer. Each call keeps
the version its request was submitted under, so a session that spans a
weight push carries both versions. The statuses follow the scheduler's
failure policy: an environment that raises or reports ERROR is an
infrastructure failure, retried and never scored; a context, token or turn
limit truncates; only COMPLETED and TRUNCATED sessions reach the verifier.

`PromptSource` is the single-turn case: one call per sample, scored on
decoded text by a `Reward` on scorer threads as each draw finishes, so
verification overlaps generation.

Verifiers return a float or a `Score` whose `components` travel with the
rollout for logging.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field, replace
from uuid import uuid4

import numpy as np

from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY
from dew.inference.rollouts import Draw, RolloutServer
from dew.nn.inputs import local_rows
from dew.objectives.base import Batch

from .episodes import (
    Action,
    EnvironmentFactory,
    Episode,
    EpisodeId,
    EpisodeStatus,
    Observation,
    Transition,
    rollout_of,
)
from .rollout import Reward, _texts
from .rollouts import Call, Rollout, Status, Task


@dataclass(frozen=True)
class Score:
    """A verifier's reward with its named sub-scores and provenance."""

    reward: float
    components: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""


def _scored(value: float | Score) -> Score:
    score = value if isinstance(value, Score) else Score(float(value))
    if not math.isfinite(score.reward) or not all(math.isfinite(part) for part in score.components.values()):
        raise ValueError("the verifier returned a non-finite score")
    return score


def _seed(*parts: int) -> int:
    return int(np.random.default_rng(list(parts)).integers(0, 2 ** 31 - 1))


def _failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


class _Refused(Exception):
    """A configuration error found inside a session; it fails the rollout's future."""


@dataclass
class _Session:
    wake: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False


class EnvironmentSource:
    """Run `Environment` sessions against a `RolloutServer`, one worker thread each.

    `environment` enters one environment per `EpisodeId`; task ids must be
    decimal integers, as `EpisodeRollout`'s are. `verifier` scores completed
    and truncated episodes. `workers` bounds concurrent sessions; the server
    batches their calls. Environments see the draw's raw likelihoods when
    the server reports them, and its behavior likelihoods otherwise, which
    are the same distribution only for a sampling policy without transforms.
    """

    def __init__(self, server: RolloutServer, environment: EnvironmentFactory,
                 verifier: Callable[[Episode], float | Score], *, max_prompt_tokens: int, max_new_tokens: int,
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
        self._sessions: dict[Future[Rollout], _Session] = {}
        self._serial = 0

    def submit(self, task: Task, samples: int, *, version: int) -> list[Future[Rollout]]:
        if not task.id.isdecimal():
            raise ValueError(f"an environment task id is a decimal integer, got {task.id!r}")
        with self._lock:
            serial = self._serial
            self._serial += 1
        futures = []
        for sample in range(samples):
            session = _Session()
            identity = EpisodeId(int(task.id), serial, sample, (self.seed, serial, sample))
            future = self._pool.submit(self._run, identity, version, session)
            with self._lock:
                self._sessions[future] = session
            future.add_done_callback(self._forget)
            futures.append(future)
        return futures

    def cancel(self, futures: Sequence[Future[Rollout]]) -> None:
        """Stop sessions at their next turn; queued ones never start."""
        for future in futures:
            if future.cancel():
                continue
            with self._lock:
                session = self._sessions.get(future)
            if session is not None:
                session.cancelled = True
                session.wake.set()

    def close(self) -> None:
        """Cancel every session and wait for the running ones to release their environments."""
        with self._lock:
            running = list(self._sessions)
        self.cancel(running)
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _forget(self, future: Future[Rollout]) -> None:
        with self._lock:
            self._sessions.pop(future, None)

    def _draw(self, context: tuple[int, ...], seed: int, session: _Session) -> Draw | None:
        """One call on the server, or None when the session is cancelled while it runs."""
        session.wake.clear()
        if session.cancelled:
            return None
        pending = self.server.submit(context, self.max_new_tokens, seed=seed)
        pending.add_done_callback(lambda _: session.wake.set())
        session.wake.wait()
        if session.cancelled and not pending.done():
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

    def _run(self, identity: EpisodeId, version: int, session: _Session) -> Rollout:
        initial: Observation | None = None
        transitions: list[Transition] = []
        pending: Action | None = None
        status, detail = EpisodeStatus.RUNNING, ""
        try:
            with self.environment(identity) as environment:
                observation = environment.reset()
                initial = observation
                for turn in range(self.max_turns + 1):
                    if not isinstance(observation, Observation):
                        raise TypeError("environment methods must return an Observation")
                    status, detail = observation.status, observation.detail
                    if status != EpisodeStatus.RUNNING:
                        break
                    if turn == self.max_turns:
                        status, detail = EpisodeStatus.TRUNCATED, "episode turn limit reached"
                        break
                    if len(observation.context) > self.max_prompt_tokens:
                        status, detail = EpisodeStatus.TRUNCATED, "next context exceeds max_prompt_tokens"
                        break
                    draw = self._draw(observation.context, _seed(self.seed, identity.attempt,
                                                                 identity.sample, turn), session)
                    if draw is None:
                        status, detail = EpisodeStatus.CANCELLED, "cancelled by the scheduler"
                        break
                    pending = self._action(draw, observation.context)
                    if pending.terminated:
                        observation = environment.step(pending)
                    else:
                        observation = Observation((), EpisodeStatus.TRUNCATED, "model turn reached its token limit")
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
        score = None
        if status in (EpisodeStatus.COMPLETED, EpisodeStatus.TRUNCATED):
            try:
                score = _scored(self.verifier(episode))
            except Exception as error:
                episode = replace(episode, status=EpisodeStatus.ERROR, detail=f"verifier: {_failure(error)}")
        if score is None:
            return rollout_of(episode, group="")
        rollout = rollout_of(replace(episode, reward=score.reward), group="")
        return replace(rollout, components=dict(score.components),
                       detail="; ".join(part for part in (rollout.detail, score.detail) if part))


def prompt_tasks(batch: Batch) -> list[Task]:
    """One single-turn task per prompt row: its ids and the three reward strings.

    The id is a digest of the prompt ids; groups, not ids, keep repeats apart.
    """
    prompts, lengths = local_rows(batch[PROMPT_KEY]), local_rows(batch[LENGTH_KEY])
    sources, truths, infos = (_texts(local_rows(batch[name])) for name in (SOURCE_KEY, TRUTH_KEY, INFO_KEY))
    rows, width = prompts.shape
    if (lengths.shape != (rows,) or not np.issubdtype(lengths.dtype, np.integer)
            or np.any(lengths < 1) or np.any(lengths > width)):
        raise ValueError("prompt_length must contain one valid integer length per row")
    tasks = []
    for row in range(rows):
        ids = tuple(int(token) for token in prompts[row, width - int(lengths[row]):])
        name = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:16]
        tasks.append(Task(name, {"prompt": ids, "source": sources[row], "truth": truths[row], "info": infos[row]}))
    return tasks


class PromptSource:
    """Draw one completion per sample of a `prompt_tasks` task and score its decoded text.

    `reward` scores the completion with EOS excluded, on `scorers` threads.
    A draw that ends on its token budget is TRUNCATED and scored for the
    record only; a failed draw or reward is an infrastructure failure.
    """

    def __init__(self, server: RolloutServer, reward: Reward, *, decode: Callable[[Sequence[int]], str],
                 max_new_tokens: int, scorers: int = 16, seed: int = 0):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("a rollout generates at least one token")
        self.server, self.reward, self.decode, self.max_new_tokens = server, reward, decode, max_new_tokens
        self.seed = seed
        self._scorers = ThreadPoolExecutor(max_workers=scorers, thread_name_prefix="dew-reward")
        self._serial = 0
        self._lock = threading.Lock()

    def submit(self, task: Task, samples: int, *, version: int) -> list[Future[Rollout]]:
        del version  # each draw reports the version it was submitted under
        with self._lock:
            serial = self._serial
            self._serial += 1
        prompt = task.data["prompt"]
        return [self._scored(task, self.server.submit(prompt, self.max_new_tokens, seed=_seed(self.seed, serial, k)))
                for k in range(samples)]

    def cancel(self, futures: Sequence[Future[Rollout]]) -> None:
        """Forget the rollouts; their draws finish on the server and are not scored."""
        for future in futures:
            future.cancel()

    def close(self) -> None:
        """Stop the reward threads; the server belongs to the caller."""
        self._scorers.shutdown(wait=True, cancel_futures=True)

    def _rollout(self, task: Task, drawn: Future[Draw]) -> Rollout:
        try:
            draw = drawn.result()
        except Exception as error:
            return Rollout(task.id, "", 0, 0, (), Status.INFRA_ERROR, None, {}, f"draw: {_failure(error)}")
        call = Call(draw.prompt, draw.tokens, draw.behavior_log_probs, "stop" if draw.terminated else "length",
                    draw.version)
        status = Status.COMPLETED if draw.terminated else Status.TRUNCATED
        try:
            text = self.decode(draw.tokens[:len(draw.tokens) - int(draw.terminated)])
            prompt = task.data
            score = _scored(self.reward(prompt["source"], text, prompt["truth"], prompt["info"]))
        except Exception as error:
            return Rollout(task.id, "", 0, 0, (call,), Status.INFRA_ERROR, None, {}, f"reward: {_failure(error)}")
        return Rollout(task.id, "", 0, 0, (call,), status, score.reward, dict(score.components), score.detail)

    def _scored(self, task: Task, drawn: Future[Draw]) -> Future[Rollout]:
        """The draw's future, chained into its reward on a scorer thread."""
        scored: Future[Rollout] = Future()

        def score() -> None:
            if scored.cancelled():
                return
            with suppress(InvalidStateError):  # cancelled while scoring
                scored.set_result(self._rollout(task, drawn))

        def chained(_: Future[Draw]) -> None:
            try:
                self._scorers.submit(score)
            except RuntimeError as closed:
                with suppress(InvalidStateError):
                    scored.set_exception(closed)

        drawn.add_done_callback(chained)
        return scored
