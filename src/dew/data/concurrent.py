"""A stage's records run on a pool of its own, several at once, delivered in the order they were
read, with a position a restore reads them again from (`dew.data.sources.urls.Fetch.threads`)."""

from __future__ import annotations

import collections
import concurrent.futures
from collections.abc import Callable, Mapping

import grain.python as grain


class Concurrently[Element, Result](grain.IterDataset):
    """`step` over each of a stream's records, `threads` of them at once on a pool of the stage's
    own, delivered in the order they were read (`Fetch.threads`)."""

    def __init__(self, parent: grain.IterDataset, step: Callable[[Element], Result], threads: int):
        super().__init__(parent)
        self._rows, self._step, self._threads = parent, step, threads

    def __iter__(self) -> grain.DatasetIterator:
        return _ConcurrentIterator(self._rows.__iter__(), self._step, self._threads)


class _ConcurrentIterator[Element, Result](grain.DatasetIterator):
    """Reads up to `threads` records ahead, each running on the pool. Its position is its parent's
    before the oldest record it has read and not delivered, so a restore reads those again."""

    def __init__(self, parent: grain.DatasetIterator, step: Callable[[Element], Result], threads: int):
        super().__init__(parent)
        self._rows, self._step, self._threads = parent, step, threads
        self._pool = concurrent.futures.ThreadPoolExecutor(threads, thread_name_prefix="dew-step")
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
