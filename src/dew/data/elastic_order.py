"""The orders an elastic read's loading workers read its inputs in (`dew.data.lowering._Build.handed`),
the same whatever their count: by index (`_InputOrder`) or as a stream (`_StreamOrder`), each worker taking
the groups of a batch's count of inputs its slice names, which Grain's interleave of the workers hands on in
turn."""

from __future__ import annotations

import collections
import itertools
from collections.abc import Callable, Mapping, Sequence

import grain.python as pygrain

from dew._records import integer, record
from dew.batch import Batch
from dew.data.storage import MIRRORS, carried_mirrors

type Grouping = Callable[[], Callable[[Sequence], Batch]]
"""What makes a worker's group of inputs one element (`lowering._Grouped`): called once a read opens."""


class _InputOrder(pygrain.IterDataset):
    """An elastic read's indexed rows read in its loading workers (`_Build.handed`), in the inputs'
    order whatever their count: groups of `size` consecutive inputs (`lowering._Grouped`), group k read by
    worker k mod W, which Grain's interleave of the workers hands on in turn, so group after group
    in order. Its position is the input its next group begins at, so a resume opens its workers
    there (`rows[next:]`) on any count; `size` is the chain's batch, the same on every topology, so
    a flat map over it (`_Ungrouped`) finds the group it was inside where it left it."""

    def __init__(self, rows: pygrain.MapDataset, size: int,
                 read: Callable[[pygrain.MapDataset], pygrain.IterDataset], grouping: Grouping):
        super().__init__([])
        self.rows, self.size, self.read, self.grouping = rows, size, read, grouping

    def __iter__(self) -> pygrain.DatasetIterator:
        return _InputOrderIterator(self)


class _InputOrderIterator(pygrain.DatasetIterator):
    def __init__(self, ordered: _InputOrder):
        super().__init__()
        self._ordered, self._next, self._reads = ordered, 0, None

    def _opened(self) -> pygrain.DatasetIterator:
        if self._reads is None:
            ordered = self._ordered
            # Each group a dataset element of its own, so the workers' slices of them, every W-th,
            # interleave back into the inputs' order.
            groups = ordered.rows[self._next:].batch(ordered.size, drop_remainder=False,
                                                     batch_fn=ordered.grouping())
            self._reads = iter(ordered.read(groups))
            self._reads.start_prefetch()
        return self._reads

    def __next__(self) -> Batch:
        group = next(self._opened())
        self._next += self._ordered.size
        return group

    def get_state(self) -> Mapping[str, object]:
        # Beside the next input, what its workers' mirrors did, which their positions carry (row 4339).
        return {"next_input": self._next, MIRRORS: _workers_mirrors(self._reads)}

    def set_state(self, state: Mapping[str, object]) -> None:
        self.close()
        self._next = integer(state["next_input"], "elastic read's next input")

    def close(self) -> None:
        if self._reads is not None:
            self._reads.close()
            self._reads = None


def _workers_mirrors(reads: pygrain.DatasetIterator | None) -> dict[str, dict[str, float]]:
    """The mirror counters an elastic read's open loading workers' positions carry (`carried_mirrors`),
    which its own position, its input's alone, keeps beside it for the run's metrics."""
    return {} if reads is None else carried_mirrors(reads.get_state())


class _StreamOrder(pygrain.IterDataset):
    """An elastic read's streamed rows read in its loading workers (`_Build.handed`), in the stream's order
    whatever their count, as `_InputOrder` reads indexed ones: every worker reads the whole stream and keeps
    the groups of `size` consecutive rows its slice names (`_Kept`), group k in worker k mod W, which
    Grain's interleave of the workers hands on in turn, and the reading process takes each group's records
    and markers apart (`ungrouped`). Its position is the stream's own before the group it is reading, which
    each group carries after it (`_Kept`), and the rows it has taken past it: a resume opens every worker's
    stream there, on any count, and passes them, so none rereads what it read. A group taken whole moves
    the position past it, so the stream's counters in it (a shard's chunks), which a read in workers
    states through its workers' positions, count every row handed on at each group's end; within a group
    they hold the rows before it, as a worker's groups' counters do in any read in workers
    (`_Build.handed`). Without workers (`alone`) each row is a group, so they count each row as it is
    handed on, but for the group a resume stopped inside, read whole as workers group it, so the rows it
    had taken mean the same on every count (row 1896)."""

    def __init__(self, rows: pygrain.IterDataset, size: int,
                 read: Callable[[pygrain.IterDataset], pygrain.IterDataset], grouping: Grouping,
                 ungrouped: Callable[[Batch], Sequence], *, alone: bool = False):
        super().__init__([])
        self.rows, self.size, self.read, self.grouping, self.ungrouped = rows, size, read, grouping, ungrouped
        self.alone = alone

    def __iter__(self) -> pygrain.DatasetIterator:
        return _StreamOrderIterator(self)


class _StreamOrderIterator(pygrain.DatasetIterator):
    def __init__(self, ordered: _StreamOrder):
        super().__init__()
        self._ordered, self._reads = ordered, None
        self._position: Mapping[str, object] | None = None
        self._after: Mapping[str, object] | None = None
        # The rows handed on past `_position` (`_taken`), those of them a resume has still to pass (`_skip`),
        # and the length of the group being read.
        self._taken, self._skip, self._length = 0, 0, 0
        self._pending: collections.deque = collections.deque()

    def _opened(self) -> pygrain.DatasetIterator:
        if self._reads is None:
            ordered = self._ordered
            # Read alone, each row is a group, but for the group a resume stopped inside, read whole as a
            # batch's count of rows, as every count of workers groups it: its rows taken are counted in
            # its order of markers and records (`ungrouped`).
            opening = ordered.size if self._skip else 1
            self._reads = iter(ordered.read(_Kept(ordered.rows, 1 if ordered.alone else ordered.size,
                                                  self._position, ordered.grouping,
                                                  opening if ordered.alone else None)))
            self._reads.start_prefetch()
        return self._reads

    def _read(self) -> None:
        """The next group's rows pending, and those of them a resume has passed already taken off."""
        group = dict(next(self._opened()))
        self._after = record(group.pop("position"), "an elastic stream's position")
        self._pending.extend(self._ordered.ungrouped(group))
        self._length = len(self._pending)
        while self._skip and self._pending:
            self._pending.popleft()
            self._skip -= 1

    def _passed(self) -> None:
        """A group taken whole: the position past it, its counters with it (`_Kept`)."""
        self._position, self._taken = self._after, self._taken - self._length

    def __next__(self):
        while not self._pending:
            self._read()
            if not self._pending:
                self._passed()
        element = self._pending.popleft()
        self._taken += 1
        if not self._pending:
            self._passed()
        return element

    def get_state(self) -> Mapping[str, object]:
        return {"stream": self._position, "taken": self._taken, MIRRORS: _workers_mirrors(self._reads)}

    def set_state(self, state: Mapping[str, object]) -> None:
        self.close()
        position = state["stream"]
        self._position = None if position is None else record(position, "elastic stream's position")
        self._taken = self._skip = integer(state["taken"], "elastic stream's taken rows")
        self._pending.clear()

    def close(self) -> None:
        if self._reads is not None:
            self._reads.close()
            self._reads = None


class _Kept(pygrain.IterDataset):
    """The groups of `size` consecutive rows of the stream `rows` that one loading worker keeps
    (`_StreamOrder`): every group whose number is its slice's (`set_slice`, which stops here, so Grain never
    deals the stream's shards between the workers), from the stream's `position` where given, the first
    `opening` rows long, each grouped (`lowering._Grouped`) with the stream's position after it."""

    def __init__(self, rows: pygrain.IterDataset, size: int, position: Mapping[str, object] | None,
                 grouping: Grouping, opening: int | None = None):
        super().__init__([])
        self.rows, self.size, self.position, self.grouping = rows, size, position, grouping
        self.opening = size if opening is None else opening
        self.first, self.step = 0, 1

    def set_slice(self, selection: slice, sequential_slice: bool = False) -> None:
        del sequential_slice
        self.first, self.step = selection.start or 0, selection.step or 1

    def __iter__(self) -> pygrain.DatasetIterator:
        return _KeptIterator(self)


class _KeptIterator(pygrain.DatasetIterator):
    def __init__(self, kept: _Kept):
        super().__init__()
        self._kept, self._number, self._grouped = kept, 0, kept.grouping()
        self._rows = kept.rows.__iter__()
        if kept.position is not None:
            self._rows.set_state(kept.position)

    def __next__(self) -> Batch:
        kept = self._kept
        while True:
            group = list(itertools.islice(self._rows, kept.opening if self._number == 0 else kept.size))
            if not group:
                raise StopIteration
            number, self._number = self._number, self._number + 1
            if number % kept.step == kept.first:
                return {**self._grouped(group), "position": self._rows.get_state()}

    def get_state(self) -> Mapping[str, object]:
        return {"group": self._number, "rows": self._rows.get_state()}

    def set_state(self, state: Mapping[str, object]) -> None:
        self._number = integer(state["group"], "a kept stream's group")
        self._rows.set_state(record(state["rows"], "a kept stream's rows"))

    def close(self) -> None:
        self._rows.close()
