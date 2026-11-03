"""The orders an elastic read's loading workers read its inputs in (`dew.data.lowering._Build.handed`),
the same whatever their count: by index (`_InputOrder`) or as a stream (`_StreamOrder`), each worker taking
the groups of a batch's count of inputs its slice names, which Grain's interleave of the workers hands on in
turn."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence

import grain.python as pygrain

from dew._records import integer, record
from dew.batch import Batch

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
        return {"next_input": self._next}

    def set_state(self, state: Mapping[str, object]) -> None:
        self.close()
        self._next = integer(state["next_input"], "elastic read's next input")

    def close(self) -> None:
        if self._reads is not None:
            self._reads.close()
            self._reads = None


class _StreamOrder(pygrain.IterDataset):
    """An elastic read's streamed rows read in its loading workers (`_Build.handed`), in the stream's order
    whatever their count, as `_InputOrder` reads indexed ones: every worker reads the whole stream and keeps
    the groups of `size` consecutive rows its slice names (`_Kept`), group k in worker k mod W, which Grain's
    interleave of the workers hands on in turn. Its position is the stream's own after the last group handed
    on, which each group carries (`_Kept`), so a resume opens every worker's stream there, on any count,
    and none rereads what it read (row 1896)."""

    def __init__(self, rows: pygrain.IterDataset, size: int,
                 read: Callable[[pygrain.IterDataset], pygrain.IterDataset], grouping: Grouping):
        super().__init__([])
        self.rows, self.size, self.read, self.grouping = rows, size, read, grouping

    def __iter__(self) -> pygrain.DatasetIterator:
        return _StreamOrderIterator(self)


class _StreamOrderIterator(pygrain.DatasetIterator):
    def __init__(self, ordered: _StreamOrder):
        super().__init__()
        self._ordered, self._position, self._reads = ordered, None, None

    def _opened(self) -> pygrain.DatasetIterator:
        if self._reads is None:
            ordered = self._ordered
            self._reads = iter(ordered.read(_Kept(ordered.rows, ordered.size, self._position,
                                                  ordered.grouping)))
            self._reads.start_prefetch()
        return self._reads

    def __next__(self) -> Batch:
        group = next(self._opened())
        self._position = group.pop("position")
        return group

    def get_state(self) -> Mapping[str, object]:
        return {"stream": self._position}

    def set_state(self, state: Mapping[str, object]) -> None:
        self.close()
        position = state["stream"]
        self._position = None if position is None else record(position, "elastic stream's position")

    def close(self) -> None:
        if self._reads is not None:
            self._reads.close()
            self._reads = None


class _Kept(pygrain.IterDataset):
    """The groups of `size` consecutive rows of the stream `rows` that one loading worker keeps
    (`_StreamOrder`): every group whose number is its slice's (`set_slice`, which stops here, so Grain never
    deals the stream's shards between the workers), from the stream's `position` where given, each grouped
    (`lowering._Grouped`) with the stream's position after it."""

    def __init__(self, rows: pygrain.IterDataset, size: int, position: Mapping[str, object] | None,
                 grouping: Grouping):
        super().__init__([])
        self.rows, self.size, self.position, self.grouping = rows, size, position, grouping
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
            group = list(itertools.islice(self._rows, kept.size))
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
