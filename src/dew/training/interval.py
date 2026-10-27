"""What a fit sums between its logs and between its checkpoints.

Every step adds its loss, its finiteness, its analytical work and its
optimizer commit to counters that stay on the devices, in one dispatch, so
the step loop never waits on a number. At its logging cadence the loop takes
a log (`Taken`) and reads it once the next step is dispatched, and restarts
the counters each cadence owns.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.nn.precision import at_least_fp32
from dew.telemetry.instrumentation import NO_FLOPS, StepFlops
from dew.training.display import TrainingDisplay
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
        return cls(time.time(), last_saved=last_saved, tokens_before_log=committed_tokens(control))

    @property
    def token_control(self) -> dict[str, int]:
        """The token clock a checkpoint keeps (`Checkpoints.control`): the committed updates' real
        tokens."""
        return {COMMITTED_TOKENS: self.real_tokens}

    @property
    def real_tokens(self) -> int:
        """Lifetime real tokens: the runs before this fit's, its logged
        intervals' and the current interval's, read from the devices."""
        counted = [tokens for tokens in (self.logged_tokens, self.book.tokens if self.tokens_counted and
                                         self.book is not None else None) if tokens is not None]
        return self.tokens_before_log + sum(_TokenCount.host(tokens) for tokens in jax.device_get(counted))

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
                          book.tokens if self.tokens_counted and book is not None else None, statistics)
        taken = Taken(step, self, dataclasses.replace(self), reading,
                      time.time() if isinstance(loss, _Settled) and loss.is_ready() else None)
        if self.tokens_counted and book is not None:
            self.logged_tokens = book.tokens if self.logged_tokens is None else _added_tokens(
                self.logged_tokens, book.tokens)
        self.since_log, self.samples = 0, 0
        self.flops, self.rollout_seconds, self.waited, self.compiling = NO_FLOPS, 0.0, 0.0, 0.0
        self.model_counted = self.tokens_counted = False
        if book is not None and self.fresh is not None:
            self.book = book._replace(training=self.fresh.training, tokens=self.fresh.tokens,
                                      norms=self.fresh.norms, commits=self.fresh.commits)
        return taken

    def saved(self, step: int) -> None:
        """Start the next checkpoint interval after the one saved at `step`."""
        self.last_saved, self.steps = step, 0
        if self.book is not None and self.fresh is not None:
            self.book = self.book._replace(loss=self.fresh.loss)


@dataclasses.dataclass
class TokenClock:
    """The committed-token clock as a fit's loop reads it (`dew.training._fit.FitPlan.committed`): `read`,
    the count at its last read off the devices (`Interval.real_tokens`), and `bound`, above any count
    the steps since can have brought it to, so the loop reads it again only where that reaches what a
    budget, a cadence or a schedule reads next, and otherwise dispatches each step without waiting on
    the last.

    A step commits at most its rows' tokens (`Objective.row_tokens`), and a window of `accumulation`
    steps commits after its last: the window a read finds open holds at most `accumulation - 1` steps,
    each of at most the `largest` step's tokens, which a resumed run's steps before it held as well, a
    window never spanning a ramp's or a schedule's stages. A step of no declared tokens leaves the
    bound unknown (None) until the next read, as does the start, whose window is the checkpoint's."""

    read: int = 0
    accumulation: int = 1
    outstanding: int | None = None
    """The most the window open at the last read and the steps since can commit; None where unknown."""
    largest: int | None = 0
    """The most one step of this fit can commit; None once a step declares no tokens."""

    @property
    def bound(self) -> int | None:
        """The most the committed tokens can be after the latest step, or None where unknown."""
        return None if self.outstanding is None else self.read + self.outstanding

    def stepped(self, most: int | None) -> None:
        """Count a step that commits at most `most` tokens, None for a step that declares none."""
        self.largest = None if self.largest is None or most is None else max(self.largest, most)
        self.outstanding = None if self.outstanding is None or most is None else self.outstanding + most

    def reread(self, read: int) -> None:
        """Take `read`, the committed tokens read off the devices after the latest step."""
        self.read = read
        self.outstanding = None if self.largest is None else (self.accumulation - 1) * self.largest


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


class Taken(NamedTuple):
    """One logging interval, taken at its last step, `step`, and read on the
    host once the step after it is dispatched.

    Reading a step's numbers waits for that step, and read before the next
    step was dispatched, the wait left the devices idle while the host
    logged, read its next batch and dispatched: 22 ms of each 652 ms step of
    Qwen3.5-0.8B on an A100 at log_every=1 (docs/performance.md). So the
    loop reads a log one step late, as MaxText's metric logger writes its
    metrics, and at once where the run waits on the devices anyway or ends:
    before a compile, an evaluation or a checkpoint, and at the run's end.

    `counts` is the live `interval` as it stood at the take. `done` is when
    the host first saw the step finished: at the take if it already was,
    else at the read, which waits for it. The read ends the interval's clock
    there and starts the next one's.
    """

    step: int
    interval: Interval
    counts: Interval
    reading: Reading
    done: float | None

    def ready(self) -> bool:
        """Whether the read would not wait on the devices."""
        return all(leaf.is_ready() for leaf in jax.tree.leaves(self.reading) if isinstance(leaf, _Settled))

    def read(self, display: TrainingDisplay) -> tuple[Reading, float]:
        """The reading on the host, its losses' streak checked
        (`checked_streak`) at the step it read, and the interval's seconds."""
        read = on_host(self.reading, self.step)
        done = time.time() if self.done is None else self.done
        self.interval.last_log_time = done
        if read.streak is not None:
            checked_streak(int(read.streak), self.step, display)
        return read, done - self.counts.last_log_time


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
