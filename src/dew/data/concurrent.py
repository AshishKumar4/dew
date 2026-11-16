"""A stage's records run on a pool of its own, several at once, delivered in the order they were
read, with a position a restore reads them again from: a fetch's threads
(`dew.data.sources.urls.Fetch.threads`) or the pool a record stage declares it runs on (`Pool`, data.md
§10's offline execution contract)."""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import multiprocessing
import multiprocessing.pool
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Literal, Protocol

import grain.experimental as gx
import grain.python as grain

type PoolKind = Literal["threads", "processes"]
"""What a record stage declares its step runs on (`Pipeline.map(..., pool=)`): threads for a step that waits
on IO, processes for one that holds the CPU, which threads would share under the interpreter's lock."""


@dataclasses.dataclass(frozen=True)
class Pool:
    """The pool a stage's records run on, `size` of them at once, delivered in their order: threads or
    processes (`PoolKind`)."""

    kind: PoolKind
    size: int

    @classmethod
    def declared(cls, kind: PoolKind) -> Pool:
        """The pool a stage declaring `kind` runs on, sized by this process's host (`Host.current`, its CPUs
        bounded by its affinity and cgroup): a process a CPU, as many threads as the standard library's
        default pool holds over them (`concurrent.futures.ThreadPoolExecutor`: the CPUs and 4, at most 32)."""
        from dew.data.dataset import Host

        cpus = Host.current().cpus
        return cls(kind, cpus if kind == "processes" else min(32, cpus + 4))

    def running(self) -> Running:
        """The pool, started: threads, or processes started afresh (`spawn`: a process forked from one JAX
        runs in is unsafe), each ending once its reading process has (`dies_with_parent`). In a daemonic
        process (a loading worker, which may start none), a process stage runs where the worker reads, its
        read's workers spreading it already."""
        if self.kind == "threads":
            return _Threads(self.size)
        if multiprocessing.current_process().daemon:
            return _InPlace()
        return _Processes(self.size)


class Pending[Result](Protocol):
    """A record's step submitted to a pool (`Running.submit`): its result once it is made, its error
    raised."""

    def result(self) -> Result: ...


class Running(Protocol):
    """A started pool (`Pool.running`): it runs a step over one record (`submit`), and stops (`stop`),
    dropping what has not started and ending what has, bounded: a process stops whatever its step does."""

    def submit[Element, Result](self, step: Callable[[Element], Result],
                                element: Element) -> Pending[Result]: ...

    def stop(self) -> None: ...


class _Threads:
    """A thread pool of the stage's own (`Running`). A thread cannot be ended: one whose step still runs is
    waited for, as a fetch's own threads always were."""

    def __init__(self, size: int):
        self._pool = concurrent.futures.ThreadPoolExecutor(size, thread_name_prefix="dew-step")

    def submit[Element, Result](self, step: Callable[[Element], Result], element: Element) -> Pending[Result]:
        return self._pool.submit(step, element)

    def stop(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


class _Processes:
    """A process pool of the stage's own (`Running`), each process ending with its reading process
    (`dies_with_parent`) and ended when it stops (`multiprocessing.pool.Pool.terminate`), whatever its step
    is doing."""

    def __init__(self, size: int):
        self._pool = multiprocessing.get_context("spawn").Pool(size, initializer=dies_with_parent)

    def submit[Element, Result](self, step: Callable[[Element], Result], element: Element) -> Pending[Result]:
        return _Waited(self._pool.apply_async(step, (element,)))

    def stop(self) -> None:
        self._pool.terminate()
        self._pool.join()


@dataclasses.dataclass(frozen=True)
class _Waited[Result]:
    """A process's pending result (`Pending`)."""

    held: multiprocessing.pool.AsyncResult

    def result(self) -> Result:
        return self.held.get()


class _InPlace:
    """A pool of none (`Running`): each step runs as it is submitted, in the submitting thread."""

    def submit[Element, Result](self, step: Callable[[Element], Result], element: Element) -> Pending[Result]:
        done: concurrent.futures.Future[Result] = concurrent.futures.Future()
        try:
            done.set_result(step(element))
        except Exception as error:
            done.set_exception(error)
        return done

    def stop(self) -> None:
        return None


def dies_with_parent(*_: int) -> None:
    """Ends this process, a Grain worker or a stage's (`_Processes`), once the process that started it has
    ended, however it ended.

    A worker reads until its parent sets a stop event, so a parent killed, timed out or crashed
    left its workers reading into a queue no one empties: 81 of them on one host, 47 GB between
    them, a day later. A spawned worker holds its parent's sentinel, a pipe only the parent
    writes, which reads as ended once the parent has exited; a thread waits on it and ends the
    worker. Grain hands a worker its index and their count, which it reads none of."""
    import multiprocessing.connection
    import os
    import threading

    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        multiprocessing.connection.wait([parent.sentinel])
        os._exit(1)

    threading.Thread(target=watch, name="dew process watching its parent", daemon=True).start()


def ordered[Element, Result](pool: Pool, step: Callable[[Element], Result], elements: Iterable[Element]
                             ) -> Iterator[Result]:
    """`step` over each of `elements` on `pool`, at most its size in flight, each result handed on in the
    elements' order as it is made; the pool stopped once they are all handed on, or the reading stops or
    fails (`Running.stop`). A memory read's preparation runs a stage's records through it
    (`dew.data.materialize`)."""
    running = pool.running()
    pending: collections.deque[Pending[Result]] = collections.deque()
    try:
        for element in elements:
            pending.append(running.submit(step, element))
            if len(pending) >= pool.size:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        running.stop()


class Made[Record](gx.FlatMapTransform):
    """The records a pooled expansion made of one (`dew.data.lowering._through`), handed on one by one."""

    def flat_map(self, element: tuple[Record, ...]) -> tuple[Record, ...]:
        return element


class Concurrently[Element, Result](grain.IterDataset):
    """`step` over each of a stream's records, `pool.size` of them at once on a pool of the stage's
    own (`Pool`), delivered in the order they were read."""

    def __init__(self, parent: grain.IterDataset, step: Callable[[Element], Result], pool: Pool):
        super().__init__(parent)
        self._rows, self._step, self._pool = parent, step, pool

    def __iter__(self) -> grain.DatasetIterator:
        return _ConcurrentIterator(self._rows.__iter__(), self._step, self._pool)


class _ConcurrentIterator[Element, Result](grain.DatasetIterator):
    """Reads up to the pool's size of records ahead, each running on the pool: a bounded queue, which
    holds the read back while the pool is busy. Its position is its parent's before the oldest record it
    has read and not delivered, so a restore reads those again."""

    def __init__(self, parent: grain.DatasetIterator, step: Callable[[Element], Result], pool: Pool):
        super().__init__(parent)
        self._rows, self._step, self._threads = parent, step, pool.size
        self._pool = pool.running()
        self._pending: collections.deque[tuple[Mapping[str, object], Pending[Result]]] = collections.deque()
        self._ended = False

    def _fill(self) -> None:
        while not self._ended and len(self._pending) < self._threads:
            before = self._rows.get_state()
            try:
                element = next(self._rows)
            except StopIteration:
                self._ended = True
                return
            self._pending.append((before, self._pool.submit(self._step, element)))

    def __next__(self) -> Result:
        self._fill()
        if not self._pending:
            raise StopIteration
        # The oldest record leaves only once its step returns, so a step that raises leaves the
        # position before it and a restore runs it again.
        made = self._pending[0][1].result()
        self._pending.popleft()
        return made

    def get_state(self) -> Mapping[str, object]:
        return self._pending[0][0] if self._pending else self._rows.get_state()

    def set_state(self, state: Mapping[str, object]) -> None:
        # The records in flight are read again from the restored position; their results are dropped.
        self._pending.clear()
        self._ended = False
        self._rows.set_state(state)

    def close(self) -> None:
        # Records not yet started are dropped and a process running one ended (`Running.stop`), so a closed
        # reader leaves no process of its own; a thread running one finishes first.
        self._pool.stop()
        self._pending.clear()
        self._rows.close()
