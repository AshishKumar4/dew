"""Serving for scoring: requests answered by one forward pass apiece, batched as they arrive.

A scoring request reads no cache and draws nothing: a classifier's answer, a decision's, a reward or an
embedding is one pass over its rows. `ScoringServer` queues such requests and answers them in batches,
the oldest first up to a token budget and a row bound, through the task's own compiled pass, whose
shapes the task buckets so a compile is reused across batches. The task (`Scoring`) lays out each
request as it is submitted, so a request it refuses is refused there, alone, and the server only batches
what it admitted.

It is a serving kind as `dew.inference.serving.Server`, autoregressive generation's, is
(`dew.inference.engine.Kind`): it admits waiting requests within its bounds, answers them with one
`step`, and settles their futures, so an `Engine` serves it beside other kinds from one thread. The
weights are placed on their mesh at the storage the task hands them in, and reloaded as generation's
are. A request's future records
when it was submitted, admitted and finished. `step` answers one batch; `run` steps until no request
waits.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax

from dew.inference.engine import Timed, normalized, reloaded, settled
from dew.inference.tasks import held
from dew.nn.inputs import mesh_of
from dew.objectives.base import Variables


@dataclass(frozen=True)
class Scored[Payload]:
    """A request as its task admitted it: the tokens its rows hold, which a batch's budget counts, and
    what the task's pass reads of it."""

    tokens: int
    payload: Payload


@runtime_checkable
class Scoring[Request, Payload, Answer](Protocol):
    """A task served by one forward pass a request (`ScoringServer`). It holds no weights of its
    own: the server holds them resident, and hands them to every pass, so a task and the weights it
    scores with are one pair the server swaps whole (`ScoringServer.reload`)."""

    def admit(self, request: Request) -> Scored[Payload]:
        """`request` laid out for the task's pass; raises ValueError for a request it refuses."""
        ...

    def score(self, variables: Variables, payloads: Sequence[Payload]) -> list[Answer]:
        """Every admitted request's answer, in order, from one call over `variables`: the task's pass
        over their rows, batched into the shapes it compiles."""
        ...


class _Request[Payload, Answer]:
    def __init__(self, scored: Scored[Payload]) -> None:
        self.scored = scored
        self.future: Timed[Answer] = Timed()


class ScoringServer[Request, Payload, Answer]:
    """Answers scoring requests in batches through one task's pass.

    A batch takes the waiting requests oldest first, at most `rows` of them and, past its first, as
    many as `tokens` holds. Build one with `from_task`.
    """

    def __init__(self, task: Scoring[Request, Payload, Answer], variables: Variables, *, tokens: int,
                 rows: int) -> None:
        if tokens < 1 or rows < 1:
            raise ValueError(f"a batch holds a token or more of a request or more, not {tokens} and {rows}")
        self.tokens = tokens
        self.rows = rows
        # Placed as given: the task's pass reads the weights at the storage it was handed them in.
        placed = held(variables)
        self._served = (task, placed)
        """The task and the weights it scores with, one pair: a step reads both at once, and a
        reload replaces both at once, so an answer never pairs one task with another's weights."""
        self.mesh = mesh_of(placed)
        self._source_shapes = jax.tree_util.tree_flatten_with_path(jax.eval_shape(lambda: placed))[0]
        self._queue: deque[_Request[Payload, Answer]] = deque()
        self._ingress = threading.Lock()
        """Orders a submission's append with a failure's sweep."""
        self._failed: BaseException | None = None
        self.arrived: Callable[[], None] = _nothing
        """Called after each submission: an `Engine` serving this kind wakes to it."""

    @classmethod
    def from_task(cls, task: Scoring[Request, Payload, Answer], variables: Variables, *,
                  tokens: int = 8192, rows: int = 64) -> ScoringServer[Request, Payload, Answer]:
        """A server over `task`'s pass with `variables`, held resident; a batch holds at most `rows`
        requests and `tokens` of their rows' tokens."""
        return cls(task, variables, tokens=tokens, rows=rows)

    @property
    def task(self) -> Scoring[Request, Payload, Answer]:
        return self._served[0]

    @property
    def variables(self) -> Variables:
        return self._served[1]

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def busy(self) -> bool:
        """Whether a request waits for a step."""
        return bool(self._queue)

    def submit(self, request: Request) -> Timed[Answer]:
        """Queue one request and return its future answer. The task lays it out here, so a request it
        refuses raises its ValueError here, and the others go on."""
        # Laid out by the task served now; a reload's task lays out what arrives after it.
        waiting: _Request[Payload, Answer] = _Request(self.task.admit(request))
        with self._ingress:
            if self._failed is not None:
                raise RuntimeError("the server stopped after a step failed") from self._failed
            self._queue.append(waiting)
        self.arrived()
        return waiting.future

    def step(self) -> None:
        """Answer one batch of the waiting requests."""
        if self._failed is not None:
            raise RuntimeError("the server stopped after a step failed") from self._failed
        batch: list[_Request[Payload, Answer]] = []
        held = 0
        while self._queue and len(batch) < self.rows:
            waiting = self._queue[0]
            if batch and held + waiting.scored.tokens > self.tokens:
                break
            self._queue.popleft()
            if not waiting.future.set_running_or_notify_cancel():
                continue
            batch.append(waiting)
            held += waiting.scored.tokens
        if not batch:
            return
        now = time.perf_counter()
        for waiting in batch:
            waiting.future.admitted = now
        task, variables = self._served
        with self._context():
            try:
                answers = task.score(variables, [waiting.scored.payload for waiting in batch])
            except BaseException as failure:
                # The batch's requests and every waiting one learn why; the step's caller does too.
                self.fail(failure, batch)
                raise
        now = time.perf_counter()
        for waiting, answer in zip(batch, answers, strict=True):
            waiting.future.finished = now
            settled(waiting.future, answer)

    def run(self) -> None:
        """Step until no request waits."""
        while self._queue:
            self.step()

    def reload(self, task: Scoring[Request, Payload, Answer], variables: Variables) -> None:
        """Serve `task` with `variables` from the next batch on, in place of the current pair, both
        at once: a batch reads the pair it starts with, so its answers pair one task with its own
        weights, as a task's calibration is fitted to the weights it came with. The tree must match
        the served one (`dew.inference.engine.normalized`), and its leaves are copied onto the served
        placement before the swap. Safe against a `step` on another thread."""
        self._served = (task, reloaded(normalized(variables, self._source_shapes), self.variables))

    def _context(self) -> contextlib.AbstractContextManager[None]:
        """The mesh the task's pass traces under."""
        return contextlib.nullcontext() if self.mesh is None else jax.set_mesh(self.mesh)

    def fail(self, failure: BaseException, batch: Sequence[_Request[Payload, Answer]] = ()) -> None:
        """Stop serving after `failure`: every waiting request, and `batch`'s, settles with it, and a
        later submission or step is refused."""
        with self._ingress:
            self._failed = failure
            waiting = [*batch, *self._queue]
            self._queue.clear()
        for request in waiting:
            settled(request.future, failure)


def _nothing() -> None:
    """No engine to wake."""
