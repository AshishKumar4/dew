"""A stage's records run on a pool of its own, several at once, delivered in the order they were
read, with a position a restore reads them again from: a fetch's threads
(`dew.data.sources.urls.Fetch.threads`) or the pool a record stage declares it runs on (`Pool`, data.md
§10's offline execution contract)."""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import multiprocessing
import pickle
from collections.abc import Callable, Mapping
from typing import Literal

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

    def checked(self, step: object, named: str) -> None:
        """Refuse a process pool for a `step` (the stage `named`) that does not pickle, which no process
        could be handed: a configuration that cannot run, so the read never starts."""
        if self.kind != "processes":
            return
        try:
            pickle.dumps(step)
        except (pickle.PicklingError, TypeError, AttributeError) as error:
            raise ValueError(f"{named} runs on processes (pool='processes'), and its step does not pickle "
                             f"({error}); run it on threads (pool='threads'), or define it at a module's "
                             "top level") from error

    def executor(self) -> concurrent.futures.Executor:
        """The pool itself: threads, or processes started afresh (`spawn`: a process forked from one
        JAX runs in is unsafe). In a daemonic process (a loading worker, which may start none), a
        process stage runs where the worker reads, its read's workers spreading it already."""
        if self.kind == "threads":
            return concurrent.futures.ThreadPoolExecutor(self.size, thread_name_prefix="dew-step")
        if multiprocessing.current_process().daemon:
            return _InPlace()
        spawned = multiprocessing.get_context("spawn")
        return concurrent.futures.ProcessPoolExecutor(self.size, mp_context=spawned)


class _InPlace(concurrent.futures.Executor):
    """An executor that runs each call as it is submitted, in the submitting thread."""

    def submit(self, fn, /, *args, **kwargs):
        done: concurrent.futures.Future = concurrent.futures.Future()
        try:
            done.set_result(fn(*args, **kwargs))
        except Exception as error:
            done.set_exception(error)
        return done


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
        self._pool = pool.executor()
        self._pending: collections.deque[tuple[Mapping[str, object], concurrent.futures.Future[Result]]] = (
            collections.deque())
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
        for _, running in self._pending:
            running.cancel()
        self._pending.clear()
        self._ended = False
        self._rows.set_state(state)

    def close(self) -> None:
        # Records not yet started are dropped; those running finish first, so a closed reader
        # leaves no thread of its own running.
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._rows.close()
