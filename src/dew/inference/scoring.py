"""Serving for scoring: requests answered by forward passes over their rows, batched as they arrive.

A scoring request reads no cache and draws nothing: a classifier's answer, a decision's, a reward or an
embedding is a forward over its rows. `ScoringServer` queues such requests and runs one pass of the
task's (`Scoring`) a `step`: the waiting rows oldest first, as many as the task says one pass holds
(`Scoring.fits`), its compiled shapes bucketed so a compile is reused across passes. A request whose
rows outrun one pass carries its rows not yet scored to the next step, and is answered once its last
row is. The task lays out each request as it is submitted, so a request it refuses is refused there,
alone, and the server only batches what it admitted.

It is a serving kind as `dew.inference.serving.Server`, autoregressive generation's, is
(`dew.inference.engine.Kind`): one bounded device call a `step`, so an `Engine` serves it beside other
kinds from one thread without starving them. The weights are placed on their mesh at the storage the
task hands them in, and reloaded as generation's are. A request's future records when it was
submitted, its first row was taken into a pass, and it was answered. `run` steps until no request
waits.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax

from dew.batch import Variables
from dew.inference.engine import Timed, normalized, reloaded, settled
from dew.inference.tasks import held
from dew.nn.inputs import mesh_of


@dataclass(frozen=True)
class Scored[Row, Payload]:
    """A request as its task admitted it: its rows, each scored in one of the task's passes, and what
    the task's answer reads of it besides their outputs."""

    rows: tuple[Row, ...]
    payload: Payload


@runtime_checkable
class Scoring[Request, Row, Payload, Output, Answer](Protocol):
    """A task served by forward passes over its requests' rows (`ScoringServer`). It holds no weights
    of its own: the server holds them resident, and hands them to every pass, so a task and the weights
    it scores with are one pair the server swaps whole (`ScoringServer.reload`)."""

    def admit(self, request: Request) -> Scored[Row, Payload]:
        """`request` laid out as the rows the task's passes read; raises ValueError for a request it
        refuses."""
        ...

    def fits(self, rows: Sequence[Row]) -> bool:
        """Whether `rows` are one pass of the task's: within its budget, and alike enough for one
        compiled forward."""
        ...

    def forward(self, variables: Variables, rows: Sequence[Row]) -> Sequence[Output]:
        """One pass over `rows` with `variables`: each row's output, in order."""
        ...

    def compile(self, variables: Variables) -> Sequence[object]:
        """Compile every program the task's passes run with `variables`, up front, so a server
        compiles none while it serves; returns the shapes it serves at, one program each."""
        ...

    def answer(self, payload: Payload, outputs: Sequence[Output]) -> Answer:
        """A request's answer from what it admitted and its rows' outputs, in its rows' order."""
        ...


class _Request[Row, Payload, Output, Answer]:
    def __init__(self, scored: Scored[Row, Payload], served: tuple[Scoring, Variables]) -> None:
        self.scored = scored
        self.served = served
        """The task that laid the request out, and its weights: the pair that scores it."""
        self.outputs: list[Output | None] = [None] * len(scored.rows)
        self.taken = 0
        """The rows taken into a pass so far."""
        self.future: Timed[Answer] = Timed()


class ScoringServer[Request, Row, Payload, Output, Answer]:
    """Answers scoring requests through one task's passes, one pass a step.

    A pass takes the waiting rows oldest first, as many as the task's `fits` allows, all laid out by
    one task: a request admitted before a reload is scored by the task and weights that admitted it.
    A step looks at most `rows` requests, cancelled ones among them. Build one with `from_task`.
    """

    def __init__(self, task: Scoring[Request, Row, Payload, Output, Answer], variables: Variables, *,
                 rows: int) -> None:
        if rows < 1:
            raise ValueError(f"a step looks at a request or more, not {rows}")
        self.rows = rows
        # Placed as given: the task's pass reads the weights at the storage it was handed them in.
        placed = held(variables)
        self._served = (task, placed)
        """The task and the weights it scores with, one pair: a request keeps the pair that admitted
        it, and a reload replaces both at once, so an answer never pairs one task with another's
        weights."""
        self.mesh = mesh_of(placed)
        self._source_shapes = jax.tree_util.tree_flatten_with_path(jax.eval_shape(lambda: placed))[0]
        self._queue: deque[_Request[Row, Payload, Output, Answer]] = deque()
        self._ingress = threading.Lock()
        """Orders a submission's append with a failure's sweep."""
        self._failed: BaseException | None = None
        self.arrived: Callable[[], None] = _nothing
        """Called after each submission: an `Engine` serving this kind wakes to it."""
        self.programs: tuple[object, ...] = ()
        """The shapes of the programs compiled up front (`from_task`), one each."""

    @classmethod
    def from_task(cls, task: Scoring[Request, Row, Payload, Output, Answer], variables: Variables, *,
                  rows: int = 64) -> ScoringServer[Request, Row, Payload, Output, Answer]:
        """A server over `task`'s passes with `variables`, held resident, every program its passes run
        compiled up front (`Scoring.compile`, the shapes in `programs`); a step looks at most `rows`
        requests. A server built directly compiles each program on its first pass."""
        server = cls(task, variables, rows=rows)
        with server._context():
            server.programs = tuple(task.compile(server.variables))
        return server

    @property
    def task(self) -> Scoring[Request, Row, Payload, Output, Answer]:
        return self._served[0]

    @property
    def variables(self) -> Variables:
        return self._served[1]

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def busy(self) -> bool:
        """Whether a request waits for a pass, or for the rest of its rows."""
        return bool(self._queue)

    def submit(self, request: Request) -> Timed[Answer]:
        """Queue one request and return its future answer. The task lays it out here, so a request it
        refuses raises its ValueError here, and the others go on."""
        # Laid out by the task served now, which scores it with its weights, whatever reloads meanwhile.
        served = self._served
        waiting: _Request[Row, Payload, Output, Answer] = _Request(served[0].admit(request), served)
        with self._ingress:
            if self._failed is not None:
                raise RuntimeError("the server stopped after a step failed") from self._failed
            if waiting.scored.rows:
                self._queue.append(waiting)
        if not waiting.scored.rows:
            # A request of no rows needs no pass: it is answered as it is admitted.
            waiting.future.set_running_or_notify_cancel()
            waiting.future.admitted = waiting.future.finished = time.perf_counter()
            settled(waiting.future, served[0].answer(waiting.scored.payload, []))
            return waiting.future
        self.arrived()
        return waiting.future

    def step(self) -> None:
        """Run one pass over the waiting rows, and answer every request whose last row it scores."""
        if self._failed is not None:
            raise RuntimeError("the server stopped after a step failed") from self._failed
        chosen = self._chosen()
        if not chosen:
            return
        task, variables = chosen[0][0].served
        rows = [request.scored.rows[index] for request, index in chosen]
        requests = list(dict.fromkeys(request for request, _ in chosen))
        with self._context():
            try:
                for (request, index), output in zip(chosen, task.forward(variables, rows), strict=True):
                    request.outputs[index] = output
                done = [request for request in requests if request.taken == len(request.outputs)]
                answers = [task.answer(request.scored.payload, request.outputs) for request in done]
            except BaseException as failure:
                # The pass's requests and every waiting one learn why, as for an output too few or too
                # many; the step's caller does too.
                self.fail(failure, requests)
                raise
        now = time.perf_counter()
        for request, answer in zip(done, answers, strict=True):
            self._queue.remove(request)
            request.future.finished = now
            settled(request.future, answer)

    def _chosen(self) -> list[tuple[_Request[Row, Payload, Output, Answer], int]]:
        """The rows of one pass, each with its request, oldest first: while the task says they fit,
        from requests of one pair, a request's rows in order; at most `rows` requests looked at, a
        cancelled one dropped as it is looked at."""
        chosen: list[tuple[_Request[Row, Payload, Output, Answer], int]] = []
        now = time.perf_counter()
        # The bounded prefix alone: a turn looks at no more than `rows` requests of the queue.
        for request in list(itertools.islice(self._queue, self.rows)):
            if chosen and request.served is not chosen[0][0].served:
                break
            task = request.served[0]
            while request.taken < len(request.scored.rows):
                rows = [kept.scored.rows[index] for kept, index in chosen]
                if chosen and not task.fits([*rows, request.scored.rows[request.taken]]):
                    return chosen
                # A request runs from its first row's pass on; one cancelled before it leaves the queue.
                if request.taken == 0:
                    if not request.future.set_running_or_notify_cancel():
                        self._queue.remove(request)
                        break
                    request.future.admitted = now
                chosen.append((request, request.taken))
                request.taken += 1
        return chosen

    def run(self) -> None:
        """Step until no request waits."""
        while self._queue:
            self.step()

    def reload(self, task: Scoring[Request, Row, Payload, Output, Answer], variables: Variables) -> None:
        """Serve `task` with `variables` in place of the current pair, both at once, for every request
        submitted after: a request is scored by the pair that laid it out, so its answer pairs one
        task with its own weights, as a task's calibration is fitted to the weights it came with.
        The tree must match the served one (`dew.inference.engine.normalized`), and its leaves are
        copied onto the served placement before the swap. Safe against a `step` on another thread."""
        self._served = (task, reloaded(normalized(variables, self._source_shapes), self.variables))

    def _context(self) -> contextlib.AbstractContextManager[None]:
        """The mesh the task's pass traces under."""
        return contextlib.nullcontext() if self.mesh is None else jax.set_mesh(self.mesh)

    def fail(self, failure: BaseException, taken: Sequence[_Request[Row, Payload, Output, Answer]] = ()
             ) -> None:
        """Stop serving after `failure`: every waiting request, and `taken`'s, settles with it, and a
        later submission or step is refused."""
        with self._ingress:
            self._failed = failure
            waiting = list(dict.fromkeys([*taken, *self._queue]))
            self._queue.clear()
        for request in waiting:
            settled(request.future, failure)


def _nothing() -> None:
    """No engine to wake."""
