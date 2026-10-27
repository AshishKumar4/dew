"""Session sources over a `RolloutServer`: in-process environments and single-turn prompts.

Both implement `SessionSource` for `RolloutScheduler`, drawing every model
call from a versioned `RolloutServer` (Dew's native server, vLLM or SGLang)
one request at a time, so turns of different sessions interleave in the
engine's continuous batch instead of waiting on a lock-step cohort.

`EnvironmentSource` runs the in-process `Environment` protocol that
`EpisodeCollector` runs, one session per sample on a worker thread, through
the same per-episode state, `EpisodeProgress`: its context limit, its
record of each call with the observation that followed it, and its finite
verifier reward. Each turn submits the observation's context and steps the
environment with the drawn action; the episode becomes a `Session` through
`session_of`, as `EpisodeCollector`'s do, so both feed one packer. Each
call keeps the version its request was submitted under, so a session that
spans a weight push carries both versions.

The statuses follow the scheduler's failure policy. An infrastructure
failure is an INFRA_ERROR session, retried and never scored: a draw the
server cancelled, an environment that reports ERROR, and an
`InfrastructureError` or `OSError` that the draw, the environment, the
verifier or the reward fails with. Those are an engine that cannot be
reached, times out or answers 5xx or 429 (the remote rollout servers), a
lost connection, a sandbox past its wall clock or with a dead worker
(`SubprocessEnvironment`), a container runtime that starts no container or
stalls removing one (`ContainerRunner`), and any boundary failure a
callback reports as an `InfrastructureError`. A context, token or turn
limit truncates, and only COMPLETED and TRUNCATED sessions reach the
verifier. Any other error is broken code or a broken contract, such as a
non-finite reward or likelihood, or an environment that returns no
`Observation`: it fails the rollout's future, and the scheduler raises it.

`PromptSource` is the single-turn case: one call per sample, scored on
decoded text by a reward on scorer threads as each draw finishes, so
verification overlaps generation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from uuid import uuid4

import jax

from dew.inference.engine import settled
from dew.inference.rollouts import INFRASTRUCTURE, Draw, RolloutServer

from .episodes import Environment, Episode, EpisodeId, EpisodeProgress, EpisodeStatus, session_of, step_action
from .rollout import PromptTask, Reward, scored_draw
from .sessions import Call, Session, Status, Task


def _failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _draw_failure(drawn: Future[Draw]) -> str | None:
    """Why the server returned no draw on the finished `drawn`, or None when it returned one.

    A draw the server cancelled, or one its infrastructure failed, is
    missing; any other failure is a broken server or policy, and is raised.
    """
    if drawn.cancelled():
        return "draw: cancelled by the server"
    failure = drawn.exception()
    if failure is None:
        return None
    if isinstance(failure, INFRASTRUCTURE):
        return f"draw: {_failure(failure)}"
    raise failure


@dataclass
class _Handle:
    wake: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False


class EnvironmentSource:
    """Runs `Environment` sessions against a `RolloutServer`, each on its own worker thread.

    `environment(task, identity)` enters one environment per session, and
    `verifier(task, episode)` scores completed and truncated episodes. Both
    read the task's own payload, and `identity.task` is the submission's
    serial number. `workers` is the most sessions that run at once, and the
    server batches their calls. The server's sampling policy must set an EOS
    id or a stop string. `cancel` stops a session at its next turn or during a draw, but
    never inside `environment.step`. So an environment must bound its own
    step time, as `SubprocessEnvironment` does with its wall clock, or a hung
    step holds its worker until it returns. An action carries the draw's raw
    likelihoods when the server reports them, and None when it does not.
    """

    def __init__(self, server: RolloutServer,
                 environment: Callable[[Task, EpisodeId], AbstractContextManager[Environment]],
                 verifier: Callable[[Task, Episode], float], *, max_prompt_tokens: int, max_new_tokens: int,
                 max_turns: int, workers: int = 64):
        for name, value in (("max_prompt_tokens", max_prompt_tokens), ("max_new_tokens", max_new_tokens),
                            ("max_turns", max_turns), ("workers", workers)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if server.sampling.eos_token_ids is None and not server.sampling.stop:
            raise ValueError("tool episodes need an EOS id or a stop string to tell a turn that ended "
                             "from one cut at its token limit")
        self.server, self.environment, self.verifier = server, environment, verifier
        self.max_prompt_tokens, self.max_new_tokens, self.max_turns = (
            max_prompt_tokens,
            max_new_tokens,
            max_turns,
        )
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dew-episode")
        self._lock = threading.Lock()
        self._handles: dict[Future[Session], _Handle] = {}
        self._serial = 0

    def submit(self, task: Task, samples: int, *, version: int, key: jax.Array) -> list[Future[Session]]:
        with self._lock:
            serial = self._serial
            self._serial += 1
        futures = []
        for sample in range(samples):
            handle = _Handle()
            drawn = jax.random.fold_in(key, sample)
            identity = EpisodeId(serial, 0, sample, tuple(int(word) for word in jax.random.key_data(drawn)))
            future = self._pool.submit(self._run, task, identity, version, drawn, handle)
            with self._lock:
                self._handles[future] = handle
            future.add_done_callback(self._forget)
            futures.append(future)
        return futures

    def cancel(self, futures: Sequence[Future[Session]]) -> None:
        """Stop the given sessions at their next turn, or during a draw. Queued sessions never start."""
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

    def _draw(self, context: tuple[int, ...], key: jax.Array, handle: _Handle) -> Future[Draw] | None:
        """One call on the server, finished, or None when the session is cancelled while it runs."""
        handle.wake.clear()
        if handle.cancelled:
            return None
        pending = self.server.submit(context, self.max_new_tokens, key=key)
        pending.add_done_callback(lambda _: handle.wake.set())
        handle.wake.wait()
        if handle.cancelled and not pending.done():
            return None
        return pending

    def _play(self, episode: EpisodeProgress, environment: Environment, key: jax.Array,
              handle: _Handle) -> None:
        """Run the episode's turns, each a draw on the server and an environment step."""
        episode.begin(environment.reset())
        for turn in range(self.max_turns):
            context = episode.next_context(max_prompt_tokens=self.max_prompt_tokens)
            if context is None:
                return
            drawn = self._draw(context, jax.random.fold_in(key, turn), handle)
            if drawn is None:
                episode.stop(EpisodeStatus.CANCELLED, "cancelled by the scheduler")
                return
            failed = _draw_failure(drawn)
            if failed is not None:
                episode.stop(EpisodeStatus.ERROR, failed)
                return
            draw = drawn.result().checked(self.server.sampling)
            if draw.prompt != context:
                raise ValueError("the server returned a draw for another context")
            episode.pending = draw
            episode.commit(draw, step_action(environment, draw))

    def _run(self, task: Task, identity: EpisodeId, version: int, key: jax.Array, handle: _Handle) -> Session:
        episode = EpisodeProgress(identity, self.server.sampling)
        binding = uuid4().hex
        broken = False
        try:
            with self.environment(task, identity) as environment:
                try:
                    self._play(episode, environment, key, handle)
                except INFRASTRUCTURE:
                    raise
                except BaseException:
                    broken = True
                    raise
            if episode.status not in (EpisodeStatus.ERROR, EpisodeStatus.CANCELLED):
                episode.score(lambda finished: self.verifier(task, finished), version, binding)
        except INFRASTRUCTURE as error:
            if broken:
                # The environment's exit failed while broken code was raising: the failure carries
                # that error as its context and is raised, not retried.
                raise
            # The draw's, the environment's or the verifier's boundary failed: infrastructure, retried.
            episode.abort(EpisodeStatus.ERROR, _failure(error))
        # Each call keeps its draw whole, the engine's routing and sampling support with it.
        return replace(session_of(episode.episode(version, binding), group=""), task=task.id)


def _cancelled_with(job: Future[None], rollout: Future[Session]) -> None:
    """Cancel `rollout` where its reward `job` was cancelled (`PromptSource.close`), and notify its
    waiters: `concurrent.futures.wait` counts a cancelled future done only once it is notified."""
    if job.cancelled() and rollout.cancel():
        rollout.set_running_or_notify_cancel()


class PromptSource:
    """Draws one completion per sample of a `prompt_tasks` task and scores its decoded text.

    `decode` turns the sampled ids into text, and `reward` scores that text,
    without the EOS id a draw ended on but with a stop string it ended on
    (`Draw.response`), on `scorers` threads as soon as each draw finishes. A
    draw that ends on its token budget is TRUNCATED and still scored. A
    draw the server cancelled, and an `InfrastructureError` or `OSError`
    the draw or the reward fails with, such as an unreachable engine or a
    container runtime that started no container, give an INFRA_ERROR
    session. A non-finite reward and any other error of the draw, the
    decode or the reward resolve the rollout's future with the exception, so
    no future is left pending.
    """

    def __init__(
        self,
        server: RolloutServer,
        reward: Reward,
        *,
        decode: Callable[[Sequence[int]], str],
        max_new_tokens: int,
        scorers: int = 16,
    ):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("a rollout generates at least one token")
        self.server, self.reward, self.decode, self.max_new_tokens = server, reward, decode, max_new_tokens
        self._scorers = ThreadPoolExecutor(max_workers=scorers, thread_name_prefix="dew-reward")

    def submit(self, task: Task, samples: int, *, version: int, key: jax.Array) -> list[Future[Session]]:
        prompt = PromptTask.of(task)
        return [self._scored(task, prompt, version, self.server.submit(prompt.ids, self.max_new_tokens,
                                                                       key=jax.random.fold_in(key, k)))
                for k in range(samples)]

    def cancel(self, futures: Sequence[Future[Session]]) -> None:
        """Cancel the rollouts' futures. Their draws still finish on the server but are not scored."""
        for future in futures:
            future.cancel()

    def close(self) -> None:
        """Stop the reward threads: a reward running finishes, and a rollout whose reward has not
        started is cancelled. The caller closes the server; this method does not."""
        self._scorers.shutdown(wait=True, cancel_futures=True)

    def _rollout(self, task: Task, prompt: PromptTask, version: int, drawn: Future[Draw]) -> Session:
        failed = _draw_failure(drawn)
        if failed is not None:
            return Session(task.id, "", 0, 0, (), Status.INFRA_ERROR, None, {}, failed)
        draw = drawn.result().checked(self.server.sampling)
        if draw.version != version:
            return Session(task.id, "", 0, 0, (), Status.INFRA_ERROR, None, {},
                           f"draw: served by version {draw.version}, submitted under {version}")
        return scored_draw(task.id, 0, Call.of(draw), self.decode(draw.response(self.server.sampling)),
                           prompt.record, self.reward)

    def _scored(self, task: Task, prompt: PromptTask, version: int, drawn: Future[Draw]) -> Future[Session]:
        """The draw's future, chained into its reward on a scorer thread."""
        scored: Future[Session] = Future()

        def score() -> None:
            if scored.cancelled():
                return
            try:
                rollout = self._rollout(task, prompt, version, drawn)
            except BaseException as failure:
                # Broken code or a broken contract, not infrastructure: the scheduler raises it.
                settled(scored, failure)
                return
            settled(scored, rollout)

        def chained(_: Future[Draw]) -> None:
            try:
                job = self._scorers.submit(score)
            except RuntimeError as closed:
                settled(scored, closed)
                return
            job.add_done_callback(lambda job: _cancelled_with(job, scored))

        drawn.add_done_callback(chained)
        return scored
