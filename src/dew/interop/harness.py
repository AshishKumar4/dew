"""Agent harnesses as a session source: one harness command per sample, its model calls recorded.

`HarnessSource` runs each sample as one trial of a `Harness`: a command
whose agent reaches the policy only through its own gateway session
(`dew.interop.gateway`), at the OpenAI base URL the trial is given. Dew runs
the command as a subprocess in a process group of its own, reads the
session's calls back from the gateway, and asks the harness how the trial
ended. The harness decides what its command is and what its artifacts say;
the source owns the gateway session, the trial's lifecycle and its
cancellation, the same for every harness.

- `Harbor` (`dew.interop.harbor`) runs a Harbor trial: Harbor starts the
  task's sandbox, runs its agent harness (mini-swe-agent by default) and
  then its verifier.
- `Command` runs any agent CLI that reads its model's base URL from
  `OPENAI_BASE_URL`, in a working directory of its own, and scores the
  directory it leaves with a verifier.

How a trial ends:

- `COMPLETED` when the agent exited cleanly and the verifier scored it.
- `AGENT_ERROR`, with the verifier's score, when the agent failed and the
  verifier still scored it.
- `TRUNCATED` when the agent ran out of time, context or output budget,
  hit its own step limit, or its last call stopped at its length limit.
  It has the verifier's reward when the verifier ran, and the scheduler's
  `truncation` policy decides whether it trains.
- `INFRA_ERROR` for anything else: a sandbox, gateway, engine or verifier
  failure, a trace without ids or likelihoods, an aborted call, or a
  session with no call. The scheduler retries these and never trains on
  them.

Cancelling a trial stops its harness; it does not stop a model call the
engine is still decoding for it (rllm-model-gateway 3b40c37 holds a
non-streaming call's upstream connection after its client goes,
rllm-org/rllm#938), so a cancelled trial's last call may run on to its token
budget.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import itertools
import logging
import os
import queue
import secrets
import selectors
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from dew.cleanup import end_group, exit_code, signal_group
from dew.inference.rollouts import INFRASTRUCTURE
from dew.objectives.rl.sessions import Call, Session, Status, Task

from .gateway import SESSION, Gateway, Recorded, calls, engine_failure, refused_prompt

if TYPE_CHECKING:
    from types import TracebackType

    import jax

_logger = logging.getLogger(__name__)

# How often a trial whose output has closed asks whether its command has exited, as `Popen.wait` asks.
_EXIT_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class Trial:
    """One sample's run of a harness: its command, the environment it adds, the directory it runs
    in (the source's own when None) and the directory its artifacts are under."""

    argv: tuple[str, ...]
    artifacts: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | None = None


@dataclass(frozen=True)
class Judged:
    """How a finished trial ended: its status, the verifier's reward and reward components, and a
    detail naming what happened."""

    status: Status
    reward: float | None = None
    components: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""


class Harness(Protocol):
    """One kind of agent harness, run as one command per sample against a gateway session."""

    def check(self, task: Task) -> None:
        """Raise ValueError when `task` is not one this harness runs, before any trial starts."""
        ...

    def trial(self, task: Task, url: str, name: str) -> Trial:
        """The command that runs `task` once with its model at the OpenAI base URL `url`, writing its
        artifacts under a directory named `name`."""
        ...

    def judge(self, task: Task, trial: Trial, log: str, returncode: int, recorded: Recorded) -> Judged:
        """How the trial ended: from what it wrote, its output `log`, its exit code and the calls
        the gateway recorded."""
        ...


@dataclass(frozen=True)
class Command:
    """Any agent CLI that reaches its model through `OPENAI_BASE_URL`, scored on what it leaves.

    `argv(task, directory)` is the command that runs `task` in `directory`,
    a fresh directory under `workspace` that is the trial's working
    directory. It runs with this process's environment plus `environment`
    and the session's `OPENAI_BASE_URL`, so a command that must not see the
    trainer's credentials or files runs its agent in a sandbox of its own
    (a container, a remote sandbox). `verify(task, directory, returncode)`
    scores the finished directory, or returns None when it cannot (the
    verifier itself failed), which makes the trial infra.

    A trial whose command exited 0 and scored is `COMPLETED`, one that
    exited otherwise and scored is `AGENT_ERROR`. The engine's own events
    come first: an engine failure makes it infra, and a refused prompt or a
    last call stopped at its length limit truncates it.
    """

    argv: Callable[[Task, Path], Sequence[str]]
    verify: Callable[[Task, Path, int], float | None]
    workspace: Path
    environment: Mapping[str, str] = field(default_factory=dict)

    def check(self, task: Task) -> None:
        return None

    def trial(self, task: Task, url: str, name: str) -> Trial:
        # Absolute, as the command runs in it: a path relative to the trainer would not resolve there.
        directory = Path(self.workspace).resolve() / name
        directory.mkdir(parents=True)
        return Trial(tuple(os.fspath(part) for part in self.argv(task, directory)), directory,
                     {**self.environment, "OPENAI_BASE_URL": url}, cwd=directory)

    def judge(self, task: Task, trial: Trial, log: str, returncode: int, recorded: Recorded) -> Judged:
        failed = engine_failure(recorded)
        if failed:
            return Judged(Status.INFRA_ERROR, detail=failed)
        reward = self.verify(task, trial.artifacts, returncode)
        if reward is None:
            return Judged(Status.INFRA_ERROR, detail=f"the verifier scored nothing; the command exited "
                                                     f"{returncode}: {log[-2000:]}")
        truncated = refused_prompt(recorded) or (
            "the last call stopped at its length limit"
            if recorded.calls and recorded.calls[-1].finish_reason == "length" else None)
        if truncated:
            return Judged(Status.TRUNCATED, reward, detail=truncated)
        if not recorded.calls:
            return Judged(Status.INFRA_ERROR, reward, detail="no model call reached the gateway session")
        if returncode == 0:
            return Judged(Status.COMPLETED, reward)
        return Judged(Status.AGENT_ERROR, reward, detail=f"the command exited {returncode}: {log[-2000:]}")


class HarnessSource:
    """Runs each sample as one trial of `harness` behind a recording gateway.

    - `workers` is how many trials run at once.
    - `trial_timeout`, when set, bounds each trial in seconds from its
      start. A trial past this deadline is interrupted as `cancel`
      interrupts it, and resolves `INFRA_ERROR`, which the scheduler
      retries.
    - `grace` is how long a trial has to wind down after each signal of a
      cancel or a deadline (see `cancel`).

    Gateway sessions are named `{task}:{group}:{sample}:{token}`. `group` is
    unique to one `submit` across runs, and `token` is 128 secret bits, so a
    sandbox can address only the session it was given. As a
    `SessionSource`, every future it returns resolves to a `Session`. A
    failure of the harness, its sandbox, the gateway or the engine gives an
    `INFRA_ERROR` session, not an exception, and a cancelled trial gives a
    `CANCELLED` one. Each session's `attempt` is 0, because a retry is a
    fresh `submit` and the scheduler, which assigns group identity, relabels
    it.

    The first `submit` waits up to `ready_timeout` seconds for the gateway
    to report a healthy worker (`Gateway.ready`). So a source started beside
    engines that are still loading does not launch trials that the gateway
    would answer with a 500 and no trace.

    Use it as a context manager, or call `close`, to cancel every trial when
    you are done. A trainer that exits without either, for example on an
    uncaught KeyboardInterrupt, still stops its trials. Trials run on daemon
    threads, and an atexit hook cancels the queued trials and interrupts the
    running ones, then kills their process groups after `grace` seconds. The
    threads start, and the hook registers, at the first `submit`:
    constructing the source starts nothing.
    """

    def __init__(
        self,
        harness: Harness,
        gateway: Gateway,
        *,
        workers: int = 8,
        grace: float = 60.0,
        trial_timeout: float | None = None,
        ready_timeout: float = 900.0,
        ready_poll: float = 2.0,
    ):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive number of concurrent trials")
        if not grace > 0 or (trial_timeout is not None and not trial_timeout > 0):
            raise ValueError("grace and trial_timeout are positive numbers of seconds")
        self._harness, self._gateway = harness, gateway
        self._grace, self._trial_timeout = grace, trial_timeout
        self._ready_timeout, self._ready_poll = ready_timeout, ready_poll
        self._ready = False
        # Serializes the readiness wait apart from `_lock`, so cancel and close never wait on the gateway.
        self._ready_lock = threading.Lock()
        self._closed = False
        self._run = uuid.uuid4().hex[:8]
        self._serial = itertools.count()
        # Daemon workers, not a ThreadPoolExecutor: concurrent.futures joins its pool before atexit
        # hooks run, which would wait out every running trial and start every queued one on exit.
        self._queue: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()
        self._workers = [threading.Thread(target=self._work, name=f"dew-harness-trial-{index}", daemon=True)
                         for index in range(workers)]
        self._started = False
        self._lock = threading.Lock()
        # One record per submitted, unresolved future, queued or running; a done-callback drops it.
        self._records: dict[Future[Session], _Running] = {}
        # Trials run in their own process groups, so a Ctrl-C of the trainer does not reach them: at
        # interpreter exit this hook cancels every record and stops every live trial.
        self._exit_hook = functools.partial(_stop_all, self._records, self._lock, grace)

    def submit(self, task: Task, samples: int, *, version: int, key: jax.Array) -> list[Future[Session]]:
        del key  # the agent's own requests draw through the gateway, which takes no key
        if not SESSION.fullmatch(task.id):
            raise ValueError(f"a harness task id is URL-path safe (letters, digits and ._-); got {task.id!r}")
        self._harness.check(task)
        with self._ready_lock:
            if not self._ready:
                self._gateway.ready(self._ready_timeout, poll=self._ready_poll)
                self._ready = True
        group = f"{self._run}-{next(self._serial)}"
        futures: list[Future[Session]] = []
        for sample in range(samples):
            future: Future[Session] = Future()
            with self._lock:
                if self._closed:
                    raise RuntimeError("the harness source is closed")
                if not self._started:
                    for worker in self._workers:
                        worker.start()
                    atexit.register(self._exit_hook)
                    self._started = True
                self._records[future] = _Running()
                self._queue.put(functools.partial(self._trial, future, task, group, sample, version))
            future.add_done_callback(self._forget_record)
            futures.append(future)
        return futures

    def _forget_record(self, future: Future[Session]) -> None:
        with self._lock:
            self._records.pop(future, None)

    def cancel(self, futures: Sequence[Future[Session]]) -> None:
        """Interrupt the given trials, each of which resolves `CANCELLED`; a queued one never starts.

        A trial whose output is still open `grace` seconds after the
        interrupt gets SIGTERM, and SIGKILL after another `grace` seconds,
        each sent to its whole process group. Output still open a `grace`
        after the SIGKILL is held by a process that left the group, which
        keeps running; the trial stops reading it and resolves.
        """
        with self._lock:
            for future in futures:
                record = self._records.get(future)
                if record is None or record.cancelled:
                    continue
                record.cancelled = True
                self._interrupt(record)

    def _interrupt(self, record: _Running) -> None:
        """Send a trial's process group SIGINT, once, and arm its escalation. The caller holds the lock."""
        process = record.process
        if record.timer is None and process is not None:
            signal_group(process, signal.SIGINT)
            record.timer = _timer(self._grace, self._escalate, record, signal.SIGTERM)

    def _escalate(self, record: _Running, sent: signal.Signals) -> None:
        """The trial's output stayed open for a whole grace period: signal the group again, SIGTERM then
        SIGKILL."""
        with self._lock:
            process = record.process
            if process is None:
                return
            _logger.warning("harness trial %d outlived its grace; sending %s", process.pid, sent.name)
            signal_group(process, sent)
            record.timer = (_timer(self._grace, self._escalate, record, signal.SIGKILL)
                            if sent is signal.SIGTERM else _timer(self._grace, self._release, record))

    def _release(self, record: _Running) -> None:
        """The output outlived the SIGKILL by a grace: wake the trial's read so it stops."""
        with self._lock:
            if record.process is not None:
                os.write(record.wake, b"\0")

    def close(self) -> None:
        """Cancel every unresolved trial, queued or running, and wait for the running ones to tear down."""
        with self._lock:
            # A submission checks this under the same lock, so none queues a trial behind the stop marks.
            self._closed = True
            started = self._started
            pending = list(self._records)
        self.cancel(pending)
        if not started:
            return
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join()
        atexit.unregister(self._exit_hook)

    def _work(self) -> None:
        while (trial := self._queue.get()) is not None:
            trial()

    def __enter__(self) -> HarnessSource:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: TracebackType | None) -> None:
        self.close()

    def _trial(self, future: Future[Session], task: Task, group: str, sample: int, version: int) -> None:
        # The token keeps a sandbox from addressing its group siblings' sessions through the gateway.
        session = f"{task.id}:{group}:{sample}:{secrets.token_hex(16)}"

        def ended(status: Status, records: tuple[Call, ...] = (), reward: float | None = None,
                  components: Mapping[str, float] | None = None, detail: str = "") -> Session:
            return Session(task.id, group, sample, 0, records, status, reward, components or {}, detail)

        try:
            with contextlib.ExitStack() as held:
                # The read's wake pipe exists before the command does, so a cancel never finds a running
                # trial it cannot wake. Failing to make it settles the future as any other failure does.
                woken, wake = os.pipe()
                held.callback(os.close, wake)
                held.callback(os.close, woken)
                with self._lock:
                    record = self._records[future]
                    record.wake = wake
                    queued_cancel = record.cancelled
                # The harness builds its command outside the lock, which cancel and close take.
                trial = None if queued_cancel else self._harness.trial(
                    task, self._gateway.session(session), f"dew-{group}-{sample}")
                with self._lock:
                    process = None if trial is None or record.cancelled else subprocess.Popen(
                        trial.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                        cwd=trial.cwd, env={**os.environ, **trial.environment} if trial.environment else None)
                    record.process = process
                if trial is None or process is None:
                    # Resolved outside the lock: the future's done-callback takes it.
                    future.set_result(ended(Status.CANCELLED))
                    return
                try:
                    log = self._read(process, record, woken)
                except BaseException:
                    with self._lock:
                        # A read that failed takes the trial's group with it, while the group is the trial's.
                        if record.process is not None:
                            signal_group(record.process, signal.SIGKILL)
                    raise
                finally:
                    with self._lock:
                        # Retired before the command is reaped: no signal reaches a pid given to another
                        # process.
                        record.process = None
                        cancelled, expired = record.cancelled, record.expired
                        if record.timer is not None:
                            record.timer.cancel()
                    assert process.stdout is not None
                    process.stdout.close()
                    # What the command left running in its group ends with the trial, a success's too.
                    end_group(process)
                    record.closed.set()
            import httpx

            try:
                verdict = self._verdict(ended, task, trial, session, log, process.returncode, cancelled,
                                        expired, version)
            finally:
                # The session's traces leave the gateway on every exit path. A gateway that cannot
                # delete them leaves a stale session behind, not a wrong verdict.
                try:
                    self._gateway.forget(session)
                except httpx.HTTPError as error:
                    _logger.warning("could not delete gateway session %s: %s", session, error)
            future.set_result(verdict)
        except INFRASTRUCTURE as error:
            # The trial's own boundary failed (a missing executable or working directory, a pipe the
            # host could not make): the session is an infrastructure failure the scheduler retries.
            if not future.done():
                future.set_result(ended(Status.INFRA_ERROR, detail=f"trial: {type(error).__name__}: {error}"))
        except BaseException as error:
            if not future.done():
                future.set_exception(error)

    def _read(self, process: subprocess.Popen[bytes], record: _Running, woken: int) -> str:
        """The command's output, read until it closes and the command exits, or the escalation gives
        up on it.

        Once the output closes, the read asks every `_EXIT_POLL_SECONDS`
        whether the command has exited, under the lock, and retires it in the
        step that reaps it, so no signal reaches its group's id once another
        process could hold it. Past `trial_timeout`, whether the output is
        still open or the command alone runs on, the trial is interrupted as
        `cancel` interrupts it. The escalation of either wakes this read
        through `woken` a `grace` after its SIGKILL, when the output still
        open is held outside the group.
        """
        assert process.stdout is not None
        output = bytearray()
        deadline = None if self._trial_timeout is None else time.monotonic() + self._trial_timeout
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            ready.register(woken, selectors.EVENT_READ)
            while True:
                reading = len(ready.get_map()) > 1
                if not reading:
                    with self._lock:
                        # Read unreaped, so the trial's group keeps its id until `end_group` ends it.
                        if exit_code(process) is not None:
                            record.process = None
                            break
                if deadline is not None and time.monotonic() >= deadline:
                    with self._lock:
                        record.expired = True
                        self._interrupt(record)
                    deadline = None
                waits = ([] if reading else [_EXIT_POLL_SECONDS]) + (
                    [] if deadline is None else [deadline - time.monotonic()])
                events = ready.select(min(waits, default=None))
                if any(key.fd == woken for key, _ in events):
                    break
                for key, _ in events:
                    if not (chunk := os.read(key.fd, 65536)):
                        ready.unregister(key.fd)
                    else:
                        output.extend(chunk)
        return output.decode("utf-8", errors="replace")

    def _verdict(self, ended: Callable[..., Session], task: Task, trial: Trial, session: str, log: str,
                 returncode: int, cancelled: bool, expired: bool, version: int) -> Session:
        import httpx

        try:
            recorded = calls(self._gateway.traces(session), unstamped=version)
        except (httpx.HTTPError, ValueError) as error:
            # The gateway could not be read, or answered traces no call can be read from.
            return ended(Status.CANCELLED if cancelled else Status.INFRA_ERROR,
                         detail=f"{trial.artifacts}: unusable gateway traces: {error}")
        records = recorded.calls
        if cancelled:
            return ended(Status.CANCELLED, records, detail=str(trial.artifacts))
        if expired:
            return ended(Status.INFRA_ERROR, records, detail=(
                f"{trial.artifacts}: interrupted past trial_timeout={self._trial_timeout:g} s"))
        judged = self._harness.judge(task, trial, log, returncode, recorded)
        return ended(judged.status, records, judged.reward, judged.components,
                     f"{trial.artifacts}: {judged.detail}".rstrip(": "))


@dataclass
class _Running:
    """What `cancel` and the deadline need of one submitted sample: whether it was cancelled or ran
    past `trial_timeout`, its running command, the timer of its next escalation, and whether its
    output has closed."""

    cancelled: bool = False
    expired: bool = False
    process: subprocess.Popen[bytes] | None = None
    """The running command, until the trial retires it under the lock: in the step that reaps it
    once its output has closed, or before reaping it when the escalation gives up on the output.
    Until then its pid, the group's id, is not given to another process, so a signal sent to the
    group under the lock while this is set reaches the trial's processes and no others."""
    timer: threading.Timer | None = None
    wake: int = -1
    """The write end of the pipe that stops the trial's read, open while `process` is set."""
    closed: threading.Event = field(default_factory=threading.Event)


def _timer[*Arguments](seconds: float, action: Callable[[*Arguments], None],
                       *arguments: *Arguments) -> threading.Timer:
    """A daemon timer: an escalation still pending must not hold the interpreter open at exit."""
    timer = threading.Timer(seconds, action, arguments)
    timer.daemon = True
    timer.start()
    return timer


def _stop_all(records: dict[Future[Session], _Running], lock: threading.Lock, grace: float) -> None:
    """At interpreter exit: cancel every trial, interrupt the live ones, and
    kill the process group of each whose output is still open `grace`
    seconds later, so a child left behind dies too."""
    with lock:
        live = []
        for record in records.values():
            record.cancelled = True
            if record.process is not None:
                signal_group(record.process, signal.SIGINT)
                live.append(record)
    deadline = time.monotonic() + grace
    for record in live:
        # The trial's own thread reads its output, retires and reaps the command, then marks it closed.
        if not record.closed.wait(max(0.0, deadline - time.monotonic())):
            with lock:
                if record.process is not None:
                    signal_group(record.process, signal.SIGKILL)


__all__ = ["Command", "Harness", "HarnessSource", "Judged", "Trial"]
