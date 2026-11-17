"""A stage's records run on a pool of its own, several at once, delivered in the order they were
read, with a position a restore reads them again from: a fetch's threads
(`dew.data.sources.urls.Fetch.threads`) or the pool a record stage declares it runs on (`Pool`, data.md
§10's offline execution contract)."""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import multiprocessing
from collections.abc import Callable, Mapping
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

    def running[Element, Result](self, step: Callable[[Element], Result]) -> Running[Element, Result]:
        """The pool, started, running `step`: threads, or processes started afresh (`spawn`: a process forked
        from one JAX runs in is unsafe), each handed `step` once as it starts (`_installed`) and ending once
        its reading process has (`dies_with_parent`). In a daemonic process (a loading worker, which may
        start none), a process stage runs where the worker reads, its read's workers spreading it already."""
        if self.kind == "threads":
            return _Threads(self.size, step)
        if multiprocessing.current_process().daemon:
            return _InPlace(step)
        return _Processes(self.size, step)


class Running[Element, Result](Protocol):
    """A started pool running one step (`Pool.running`): over one record (`submit`), its result a future
    whose `result` raises the step's error, or the pool's where a process of it died (a lost record raises,
    never waits); and it stops (`stop`), dropping what has not started and waiting for what has."""

    def submit(self, element: Element) -> concurrent.futures.Future[Result]: ...

    def stop(self) -> None: ...


class _Threads[Element, Result]:
    """A thread pool of the stage's own (`Running`). A thread cannot be ended: one whose step still runs is
    waited for, as a fetch's own threads always were."""

    def __init__(self, size: int, step: Callable[[Element], Result]):
        self._pool = concurrent.futures.ThreadPoolExecutor(size, thread_name_prefix="dew-step")
        self._step = step

    def submit(self, element: Element) -> concurrent.futures.Future[Result]:
        return self._pool.submit(self._step, element)

    def stop(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


class _Processes[Element, Result]:
    """A process pool of the stage's own (`Running`): each process is handed the step once, as it starts
    (`_installed`), and each record alone after (`_installed_step`); a process that dies, as one whose step
    no process can rebuild does, fails what it held and the pool (`BrokenProcessPool`). Stopped, it drops
    what has not started, and a process running a step finishes that record, as a thread does, before it
    ends and is waited for. Each ends with its reading process too (`dies_with_parent`)."""

    def __init__(self, size: int, step: Callable[[Element], Result]):
        self._pool = concurrent.futures.ProcessPoolExecutor(
            size, mp_context=multiprocessing.get_context("spawn"), initializer=_installed, initargs=(step,))

    def submit(self, element: Element) -> concurrent.futures.Future[Result]:
        return self._pool.submit(_installed_step, element)

    def stop(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


class _InPlace[Element, Result]:
    """A pool of none (`Running`): each step runs as it is submitted, in the submitting thread."""

    def __init__(self, step: Callable[[Element], Result]):
        self._step = step

    def submit(self, element: Element) -> concurrent.futures.Future[Result]:
        done: concurrent.futures.Future[Result] = concurrent.futures.Future()
        try:
            done.set_result(self._step(element))
        except Exception as error:
            done.set_exception(error)
        return done

    def stop(self) -> None:
        return None


_INSTALLED: dict[str, Callable] = {}
"""The step a stage's process runs (`_installed`), the one it was handed as it started."""


def _installed(step: Callable) -> None:
    """Start a stage's process (`_Processes`): ending once its reading process has (`dies_with_parent`), and
    holding `step`, unpickled once, which each record it is handed runs (`_installed_step`)."""
    dies_with_parent()
    _INSTALLED["step"] = step


def _installed_step[Element](element: Element):
    """The step this process holds (`_installed`) over `element`."""
    return _INSTALLED["step"](element)


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

    @staticmethod
    def over[Over, Made](rows: grain.DatasetIterator, step: Callable[[Over], Made],
                         pool: Pool) -> grain.DatasetIterator:
        """The window over an iterator already open (`rows`): a cache's fill runs its steps through it as
        it reads its shard (`dew.data.cache._CachedIterator`)."""
        return _ConcurrentIterator(rows, step, pool)


class _ConcurrentIterator[Element, Result](grain.DatasetIterator):
    """Reads up to the pool's size of records ahead, each running on the pool: a bounded queue, which
    holds the read back while the pool is busy. Its position is its parent's before the oldest record it
    has read and not delivered, so a restore reads those again."""

    def __init__(self, parent: grain.DatasetIterator, step: Callable[[Element], Result], pool: Pool):
        super().__init__(parent)
        self._rows, self._step, self._declared = parent, step, pool
        self._pool: Running[Element, Result] | None = None
        self._pending: collections.deque[tuple[Mapping[str, object], concurrent.futures.Future[Result]]] = (
            collections.deque())
        self._ended = False

    def _fill(self) -> None:
        while not self._ended and len(self._pending) < self._declared.size:
            before = self._rows.get_state()
            try:
                element = next(self._rows)
            except StopIteration:
                self._ended = True
                return
            if self._pool is None:
                self._pool = self._declared.running(self._step)
            self._pending.append((before, self._pool.submit(element)))

    def __next__(self) -> Result:
        self._fill()
        if not self._pending:
            # Its records all handed on: the pool stops, and starts again where a restore reads on.
            self._stopped()
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

    def _stopped(self) -> None:
        if self._pool is not None:
            self._pool.stop()
            self._pool = None

    def close(self) -> None:
        # Records not yet started are dropped and one running finishes and is waited for (`Running.stop`),
        # so a closed reader leaves no process or thread of its own.
        self._stopped()
        self._pending.clear()
        self._rows.close()
