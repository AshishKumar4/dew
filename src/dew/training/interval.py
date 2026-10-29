"""What a fit sums between its logs and between its checkpoints.

Every step adds its loss, its finiteness, its analytical work and its
optimizer commit to counters that stay on the devices, in one dispatch, so
the step loop never waits on a number. At its logging cadence the loop takes
a log (`Taken`) and reads it once the next step is dispatched, and restarts
the counters each cadence owns.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.nn.precision import at_least_fp32
from dew.state import Accumulation
from dew.telemetry.instrumentation import (
    NO_FLOPS,
    StepFlops,
    compilation_cache_counts,
    model_flops_utilization,
)
from dew.training.display import TrainingDisplay
from dew.training.inspection import logged
from dew.training.model_flops import TOKENS, TRAINING_FLOPS, _TokenCount
from dew.training.transaction import Committed

# Consecutive non-finite losses that stop a run.
BAD_LOSS_STEPS = 5


COMMITTED_TOKENS = "committed_tokens"
"""The checkpoint control field that holds a run's tokens, those its committed updates trained on."""


def committed_tokens(control: dict) -> int:
    """The tokens a checkpoint's `control` says its run's committed updates trained on, taken out of
    it, 0 where it counted none. A checkpoint whose control counts every step's tokens under
    `tokens`, a field runs wrote before they counted commits, is refused by name: its count holds
    rejected attempts and unfinished windows, and no count of this Dew's reads it."""
    if "tokens" in control:
        raise ValueError("this checkpoint counts every step's tokens under 'tokens', a field runs wrote "
                         f"before they counted their committed updates' ({COMMITTED_TOKENS!r}); resume it "
                         "under the Dew release that wrote it, or train afresh from its weights")
    return int(control.pop(COMMITTED_TOKENS, 0))


class Book(NamedTuple):
    """The loop's device-side counters.

    `loss` sums the loss since the last checkpoint. `bad_run` is the current
    streak of non-finite losses and `worst_bad_run` the longest since the last
    check. `training` and `tokens` sum the objective's analytical training FLOPs
    and real tokens, and `norms` and `commits` the committed gradient norms
    and their count, since the last log.
    """

    loss: jax.Array
    bad_run: jax.Array
    worst_bad_run: jax.Array
    training: jax.Array
    tokens: jax.Array
    norms: jax.Array
    commits: jax.Array


def fresh_book(loss: jax.Array) -> Book:
    """Return the counters a fresh logging interval starts from, on the device the step returns `loss` on.

    Counters on any other device would move to the step's devices at every count,
    and `bookkeep` would compile again.
    """
    dtype = at_least_fp32(loss.dtype)
    zeros = (np.zeros((), dtype), np.zeros((), np.int32), np.zeros((), np.int32),
             np.zeros((), dtype), np.zeros((2,), np.uint32), np.zeros((), dtype), np.zeros((), np.int32))
    return Book(*jax.device_put(zeros, loss.sharding))


@jax.jit
def bookkeep(book: Book, loss: jax.Array, finite: jax.Array, committed: Committed | None, training=0.0,
             tokens=(0, 0)) -> Book:
    """Advance the loop's counters for one step in a single dispatch.

    Loss counters and commit telemetry advance together, without reading
    either on the host between logging ticks.
    """
    bad_run = jnp.where(finite, 0, book.bad_run + 1)
    dtype = at_least_fp32(loss.dtype)
    norms, commits = book.norms, book.commits
    if committed is not None:
        norms, commits = norms + committed.norm.astype(dtype), commits + committed.count
    return Book(book.loss + loss.astype(dtype), bad_run, jnp.maximum(book.worst_bad_run, bad_run),
                book.training + training, _TokenCount.add(book.tokens, tokens), norms, commits)


@dataclasses.dataclass
class Interval:
    """What `fit` sums between logs and between checkpoints.

    `book` holds the device-side counters (`Book`). They move together in one
    dispatch; the host reads them at the logging cadence, once the step after
    the log is queued (`Taken`), never blocking the step loop on a result.
    The first step places them where its loss is, and `fresh` keeps them at
    zero there, for a counter's restart to reuse rather than build again.
    `tokens_before_log` is the real tokens of the runs before this fit and
    `logged_tokens` those of its logged intervals, on the devices.
    `steps` counts since the last checkpoint, the rest since the last log.
    The records and FLOPs are summed per step rather than taken off the
    dataset's batch, so a ramped interval reports the records it read and
    their FLOPs. `model_counted` and `tokens_counted` say whether a step
    since the last log reported its analytical work and real tokens.
    `rollout_seconds` is the time spent sampling, logged under
    train/rollout_seconds when a rollout is set. `waited` is the time the
    loop spent waiting on its next batch and `compiling` the time it spent
    compiling the step for a shape it had not met, both in seconds, since
    the last log. `compiles` counts StepCompiled records over the whole fit,
    so logging and checkpointing never reset it."""

    last_log_time: float
    last_saved: int | None
    tokens_before_log: int = 0
    resumable: int | None = None
    """The step of the last checkpoint a resume can read, its whole state (`Checkpoints.latest`), where
    `last_saved` may be a save of weights alone."""
    logged_tokens: jax.Array | None = None
    book: Book | None = None
    fresh: Book | None = None
    steps: int = 0
    since_log: int = 0
    samples: int = 0
    flops: StepFlops | None = NO_FLOPS
    model_counted: bool = False
    tokens_counted: bool = False
    rollout_seconds: float = 0.0
    waited: float = 0.0
    compiling: float = 0.0
    compiles: int = 0
    inspected: tuple[jax.Array, Mapping[str, jax.Array]] | None = None
    """The latest sampled commit's count and summaries since the last log (`Committed.inspected`)."""

    def count(self, batch: Batch, flops: StepFlops | None, loss: jax.Array, finite: jax.Array,
              committed: Committed | None, aux: Mapping[str, jax.Array] | None = None) -> None:
        """Add one step: its records, its FLOPs, its loss and its optimizer commit."""
        self.since_log += 1
        self.steps += 1
        self.samples += rows_of(batch)
        self.flops = None if self.flops is None or flops is None else self.flops + flops
        if self.book is None:
            self.book = self.fresh = fresh_book(loss)
        aux = {} if aux is None else aux
        self.model_counted |= TRAINING_FLOPS in aux
        self.tokens_counted |= TOKENS in aux
        assert self.fresh is not None
        # A run's tokens are its committed updates' (`Committed.tokens`); a step of one's own
        # (`Trainer(step=)`), which reports no commit, is its own update.
        tokens = aux.get(TOKENS, self.fresh.tokens) if committed is None else committed.tokens
        if committed is not None and committed.inspected:
            self.inspected = (committed.count, committed.inspected)
        self.book = bookkeep(self.book, loss, finite, committed,
                             aux.get(TRAINING_FLOPS, self.fresh.training), tokens)

    def streak(self) -> jax.Array | None:
        """The longest streak of non-finite losses since the last call, on
        the devices, and start it over; None before the first step."""
        if self.book is None or self.fresh is None:
            return None
        worst, self.book = self.book.worst_bad_run, self.book._replace(worst_bad_run=self.fresh.worst_bad_run)
        return worst

    def check_finite(self, step: int, display: TrainingDisplay) -> None:
        """Check the longest streak since the last check now (`checked_streak`):
        the loop checks at its logging cadence as it reads a log (`Taken`),
        and this is the check at the end of a run."""
        streak = self.streak()
        if streak is not None:
            checked_streak(int(on_host(streak, step)), step, display)

    @classmethod
    def resumed(cls, control: dict, *, last_saved: int | None) -> Interval:
        """An interval from now on, its token clock read off a checkpoint's `control`
        (`committed_tokens`), which it takes the clock's field out of."""
        return cls(time.time(), last_saved=last_saved, tokens_before_log=committed_tokens(control),
                   resumable=last_saved)

    @property
    def token_control(self) -> dict[str, int]:
        """The token clock a checkpoint keeps (`Checkpoints.control`): the committed updates' real
        tokens."""
        return {COMMITTED_TOKENS: self.real_tokens}

    @property
    def real_tokens(self) -> int:
        """Lifetime real tokens: the runs before this fit's, its logged
        intervals' and the current interval's, read from the devices."""
        return self.clocked(None)[0]

    def clocked(self, window: Accumulation | None) -> tuple[int, int]:
        """The lifetime real tokens (`real_tokens`) and the tokens an update's open `window` holds
        (`Accumulation.tokens`; 0 for none), in one read off the devices."""
        counted = [tokens for tokens in (self.logged_tokens, self.book.tokens if self.tokens_counted and
                                         self.book is not None else None) if tokens is not None]
        read, held = jax.device_get((counted, None if window is None else window.tokens))
        return (self.tokens_before_log + sum(_TokenCount.host(tokens) for tokens in read),
                0 if held is None else _TokenCount.host(held))

    def take(self, step: int, loss: jax.Array, aux: Mapping[str, jax.Array], accepted: jax.Array,
             copies: tuple[jax.Array | None, jax.Array | None],
             rollout: Mapping[str, Number | float], statistics: Mapping[str, float]) -> Taken:
        """Take the log of the interval ending at `step` without waiting on
        the devices, and start the next interval's counts; `copies` are the
        learning rate and loss scale, copied off the state the next step
        donates (`held`), and `statistics` the data's, read on the host as
        they stand at the step."""
        streak, book = self.streak(), self.book
        rate, scale = copies
        reading = Reading(loss, aux, accepted, streak, rate, scale, rollout,
                          None if book is None else (book.norms, book.commits),
                          book.training if self.model_counted and book is not None else None,
                          book.tokens if self.tokens_counted and book is not None else None, statistics,
                          self.inspected)
        # The copies to the host begin now, so the log's read waits on nothing but the step itself.
        for leaf in jax.tree.leaves(reading):
            if isinstance(leaf, jax.Array):
                leaf.copy_to_host_async()
        taken = Taken(step, self, dataclasses.replace(self), reading,
                      time.time() if isinstance(loss, _Settled) and loss.is_ready() else None)
        if self.tokens_counted and book is not None:
            self.logged_tokens = book.tokens if self.logged_tokens is None else _added_tokens(
                self.logged_tokens, book.tokens)
        self.since_log, self.samples = 0, 0
        self.flops, self.rollout_seconds, self.waited, self.compiling = NO_FLOPS, 0.0, 0.0, 0.0
        self.model_counted = self.tokens_counted = False
        self.inspected = None
        if book is not None and self.fresh is not None:
            self.book = book._replace(training=self.fresh.training, tokens=self.fresh.tokens,
                                      norms=self.fresh.norms, commits=self.fresh.commits)
        return taken

    def saved(self, step: int, *, resumable: bool = True) -> None:
        """Start the next checkpoint interval after the one saved at `step`, a step a resume can read
        where `resumable`."""
        self.last_saved, self.steps = step, 0
        if resumable:
            self.resumable = step
        if self.book is not None and self.fresh is not None:
            self.book = self.book._replace(loss=self.fresh.loss)


@dataclasses.dataclass
class TokenClock:
    """The committed-token clock as a fit's loop reads it (`dew.training._fit.FitPlan.committed`): `read`,
    the count at its last read off the devices (`Interval.clocked`), and `bound`, above any count the
    steps since can have brought it to, so the loop reads it again only where that reaches what a
    budget, a cadence or a schedule reads next, and otherwise dispatches each step without waiting on
    the last.

    Committed tokens grow only by what an update's window held: `held` is the tokens the window open at
    the last read held then, read with the count (`Accumulation.tokens`), and the most each step since
    can add, its rows' tokens (`Objective.row_tokens`). So the bound holds however the windows fall, a
    resumed run's restored window, a ramp's or a schedule's dropped one included. A step of no declared
    tokens leaves it unknown (None) until the next read."""

    read: int = 0
    held: int | None = 0
    """The most the window open at the last read and the steps since can commit; None where unknown."""

    @property
    def bound(self) -> int | None:
        """The most the committed tokens can be after the latest step, or None where unknown."""
        return None if self.held is None else self.read + self.held

    def stepped(self, per_row: int | None, rows: int) -> None:
        """Count a step of `rows` rows, each committing at most `per_row` tokens, None where they declare
        none."""
        self.held = None if self.held is None or per_row is None else self.held + per_row * rows

    def reread(self, read: int, pending: int) -> None:
        """Take `read`, the committed tokens off the devices after the latest step, and `pending`, those
        of the window then open."""
        self.read, self.held = read, pending


@jax.jit
def _added_tokens(left: jax.Array, right: jax.Array) -> jax.Array:
    return _TokenCount.add(left, right)


@jax.jit
def held(rate: jax.typing.ArrayLike | None,
         scale: jax.Array | None) -> tuple[jax.Array | None, jax.Array | None]:
    """Copies, in device memory, of the learning rate and the loss scale a
    log reads from the state. The next step donates that state, and a
    donated buffer is gone once the step is dispatched, however long the
    log waits to be read."""
    def copied(leaf: jax.typing.ArrayLike) -> jax.Array:
        return jnp.copy(jax.device_put(jnp.asarray(leaf), jax.memory.Space.Device))
    return None if rate is None else copied(rate), None if scale is None else copied(scale)


@runtime_checkable
class _Settled(Protocol):
    """An array that says, without waiting, whether its value is computed.
    JAX's arrays do (`is_ready`), though `jax.Array` does not declare it."""

    def is_ready(self) -> bool: ...


Number = jax.Array | np.ndarray
"""One number a log reads: on the devices until the read, on the host after it."""


class Reading(NamedTuple):
    """What a log reads from the devices, as they stood at its step: the
    step's loss, aux and acceptance, the longest non-finite streak, the rate
    and loss scale, the rollout's report, the interval's committed gradient
    norms and commits, analytical work and real tokens, and the data's
    statistics, already on the host."""

    loss: Number
    aux: Mapping[str, Number]
    accepted: Number
    streak: Number | None
    rate: Number | None
    scale: Number | None
    rollout: Mapping[str, Number | float]
    committed: tuple[Number, Number] | None
    training: Number | None
    tokens: Number | None
    statistics: Mapping[str, float]
    inspected: tuple[Number, Mapping[str, Number]] | None = None
    """The latest sampled commit since the last log, its commit count and summaries, or None."""


class Taken(NamedTuple):
    """One logging interval, taken at its last step, `step`, and read on the
    host by the run's log writer (`LogWriter`), off the step loop.

    Reading a step's numbers waits for that step, and read before the next
    step was dispatched, the wait left the devices idle while the host
    logged, read its next batch and dispatched: 22 ms of each 652 ms step of
    Qwen3.5-0.8B on an A100 at log_every=1 (docs/performance.md). Read one
    step late on the loop, as MaxText's metric logger writes its metrics, the
    host still waited on the step before it. So the copies to the host begin
    at the take and a writer thread reads and reports the log while the loop
    steps on, never waiting (row 2313); the loop waits for the writer only
    where it waits on the devices anyway or ends: before a compile, an
    evaluation or a checkpoint, and at the run's end.

    `counts` is the live `interval` as it stood at the take. `done` is when
    the host first saw the step finished: at the take if it already was,
    else at the read, which waits for it. The read ends the interval's clock
    there, and the next interval's clock starts there (`LogWriter.since`).
    """

    step: int
    interval: Interval
    counts: Interval
    reading: Reading
    done: float | None

    def ready(self) -> bool:
        """Whether the read would not wait on the devices."""
        return all(leaf.is_ready() for leaf in jax.tree.leaves(self.reading) if isinstance(leaf, _Settled))

    def read(self, display: TrainingDisplay, since: float | None) -> tuple[Reading, float, float]:
        """The reading on the host, its losses' streak checked (`checked_streak`) at the step it read, the
        interval's seconds since `since`, the previous log's `done` (the interval's own start for the first),
        and its `done`."""
        read = on_host(self.reading, self.step)
        done = time.time() if self.done is None else self.done
        if read.streak is not None:
            checked_streak(int(read.streak), self.step, display)
        return read, done - (self.counts.last_log_time if since is None else since), done


def logged_scalars(read: Reading, counts: Interval, elapsed: float, devices: int, *,
                   rollout: bool) -> dict[str, float]:
    """One logging interval's scalars, as a run logs them, from its reading on the host.

    Loss and objective aux metrics are the logged step's values. train/grad_norm is instead the mean
    over admitted optimizer commits since the last log: bookkeep sums each window's native gradient
    norm and its commit count on device, independently of the checkpoint loss sum. A tick with no
    commit omits it. train/compiles counts this fit's StepCompiled records; persistent-cache counts
    span all jitted programs in this process, as compilation_cache_counts describes. The optimizer
    state supplies the last applied update's rate (`learning_rate`): a rejected attempt leaves it, and
    an accumulation window moves it only when its update commits. The latest sampled commit's
    summaries go under `inspect/` (`dew.training.inspection`), where one committed."""
    from dew.training.memory import peak_memory_gib

    # A work count a process refused (`_counted_batch`) is NaN here, and train/mfu absent.
    work = None if read.training is None else float(read.training)
    scalars = {"train/loss": float(read.loss), **{f"train/{k}": float(v) for k, v in read.aux.items()},
               **throughput(elapsed, counts.since_log, counts.samples, counts.flops, devices,
                            waited=counts.waited, compiling=counts.compiling, training_flops=work,
                            tokens=None if read.tokens is None else _TokenCount.host(read.tokens))}
    scalars["train/accepted"] = float(read.accepted)
    if read.committed is not None:
        norms, commits = read.committed
        if commits:
            scalars['train/grad_norm'] = float(norms / commits)
    scalars['train/compiles'] = float(counts.compiles)
    scalars.update(compilation_cache_counts())
    peak = peak_memory_gib()
    if peak is not None:
        scalars['train/peak_memory_gib'] = peak
    if read.rate is not None:
        scalars["train/learning_rate"] = float(read.rate)
    if read.scale is not None:
        scalars["train/loss_scale"] = float(read.scale)
    if rollout:
        scalars["train/rollout_seconds"] = counts.rollout_seconds
        scalars.update({f"rollout/{name}": float(value) for name, value in read.rollout.items()})
    if read.inspected is not None and int(read.inspected[0]):
        scalars.update(logged(read.inspected[1]))
    scalars.update(read.statistics)
    return scalars


class LogWriter:
    """A fit's logs written off its step loop (row 2313): each one the loop takes (`Taken`) is read on the
    host and reported by one daemon thread, in the order taken, while the loop steps on; the copies to
    the host began at the take (`Interval.take`). A write's failure, a tracker's or a losses' streak, is
    kept for the loop to raise at a point every process reaches alike (`raise_failure`), where it agrees
    on it, never on the writer's thread. `since` is the last write's `done`, where the next interval's
    clock starts."""

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[tuple[Callable[[], None], Future] | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._writes: collections.deque[tuple[Future, Callable[[], bool]]] = collections.deque()
        self.unchecked = 0
        """Writes handed over since the loop last waited on every one: alike on every process."""
        self.since: float | None = None

    def submit(self, write: Callable[[], None], ready: Callable[[], bool]) -> None:
        """Hand `write` to the thread; `ready` says whether it can run without waiting on the devices."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._serve, name="dew-log-writer", daemon=True)
            self._thread.start()
        written: Future = Future()
        self._queue.put((write, written))
        self._writes.append((written, ready))
        self.unchecked += 1

    def _serve(self) -> None:
        while (handed := self._queue.get()) is not None:
            write, written = handed
            try:
                written.set_result(write())
            except BaseException as error:
                written.set_exception(error)

    def raise_failure(self, *, wait: bool, ready_only: bool = False) -> None:
        """Raise the first failed write's error among those done, or, where `wait`, among every write,
        waiting for each; `ready_only` waits only for writes that wait on no device (a failed run's
        cleanup, whose devices or peers may never finish a step)."""
        while self._writes and (self._writes[0][0].done() or (wait and (
                not ready_only or self._writes[0][1]()))):
            self._writes.popleft()[0].result()
        if wait:
            self.unchecked = 0

    def close(self) -> None:
        """Stop the thread once it has written what it holds; a write that waits on a step that never
        ends holds no process open, the thread being a daemon."""
        if self._thread is not None:
            self._queue.put(None)


def on_host[TreeT](counters: TreeT, step: int) -> TreeT:
    """`counters` on the host, in one read.

    The loss counters read every step's loss, so a step that failed on the
    devices surfaces at their read, under the counters' program.
    """
    try:
        return jax.device_get(counters)
    except jax.errors.JaxRuntimeError as error:
        raise RuntimeError(f"a training step before step {step} failed on the devices; JAX reports it "
                           "at the loss counters that read its loss (jit_bookkeep)") from error


def checked_streak(streak: int, step: int, display: TrainingDisplay) -> None:
    """Raise RuntimeError once the loss has been non-finite for
    BAD_LOSS_STEPS steps ending at `step`, and note a shorter streak.

    Checked at the logging cadence so the step loop never synchronises;
    detection is late by at most that many steps, never missed.
    """
    if streak >= BAD_LOSS_STEPS:
        raise RuntimeError(
            f"Loss has been non-finite for {streak} consecutive steps "
            f"ending near step {step}, stopping")
    if streak:
        display.note(f"Non-finite loss for {streak} step(s) before {step}", style="red")


def throughput(elapsed: float, steps: int, samples: int, flops: StepFlops | None, devices: int, *,
               waited: float = 0.0, compiling: float = 0.0,
               training_flops: float | None = None, tokens: float | None = None) -> dict[str, float]:
    """Compute the interval's rates from the records it read and its FLOPs, on a mesh of `devices`.

    The FLOPs are the ones the compiler measured for each step's own
    shape; `flops` is None when a shape's measurement was unavailable.
    train/flops is the objective's analytical training work on real rows a
    step, over the whole mesh, on any device: 3*forward for full FT,
    explicit frozen/trainable matrix work for PEFT. train/mfu is that work
    over the whole mesh's peak, where the device's peak is known. train/hfu keeps hardware HLO
    work including remat; train/hlo_flops_util is the former HLO model
    count with recomputation removed (`StepFlops.utilizations`). No HLO
    count substitutes for MFU.

    `compiling` is the seconds the interval spent compiling the step for
    shapes it met for the first time (a bucket length, a ramp's stage).
    The rates are over the rest of `elapsed`, so a step that compiled
    does not read as a slow step, and the compile is reported on its own
    as train/compile_s. `waited` is the seconds the loop waited on its
    next batch, reported per step as train/data_wait_ms. The loop
    dispatches steps ahead of the device, so a wait the device's queued
    work covers costs no device time; a wait near the step time is a
    stalled input.
    """
    stepping = elapsed - compiling
    if stepping <= 0 or steps <= 0:
        return {}
    step_time = stepping / steps
    scalars = {"train/step_time_ms": step_time * 1000,
               "train/samples_per_sec": samples / stepping,
               "train/data_wait_ms": waited / steps * 1000}
    if compiling > 0:
        scalars["train/compile_s"] = compiling
    if tokens is not None:
        scalars["train/tokens"] = float(tokens) / steps
        scalars["train/tokens_per_s"] = float(tokens) / stepping
    # A batch whose work could not be counted carries NaN (`_counted_batch`),
    # so an interval holding one, on any process, reports no train/flops or train/mfu.
    if training_flops is not None and math.isfinite(training_flops):
        scalars["train/flops"] = float(training_flops) / steps
        per_device = float(training_flops) / devices
        utilization = model_flops_utilization(per_device / steps, step_time)
        if utilization is not None:
            scalars["train/mfu"] = utilization
    return scalars if flops is None else {**scalars, **flops.utilizations(steps, step_time)}
