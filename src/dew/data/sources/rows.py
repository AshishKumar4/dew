"""A shard's indexed rows streamed under the read options its reading gives (`ShardReader`)."""

from collections.abc import Mapping

import grain.python as grain

from dew._records import integer


def streamed_rows(rows: grain.MapDataset, options: grain.ReadOptions | None) -> grain.IterDataset:
    """A shard's indexed `rows` streamed under `options` (`ShardReader`): on Grain's prefetch where they
    read on threads of their own or Grain's defaults (None), else read by index on the thread that draws
    them, a mix member's share of no thread (`_IndexedRows`), which Grain's prefetch of no threads read
    too but registered for its statistics each time a shard opened and never let go (its iterators'
    registry), a few blocks a shard a run kept (row 4079)."""
    if options is not None and options.num_threads == 0:
        return _IndexedRows(rows)
    return rows.to_iter_dataset(options)


class _IndexedRows(grain.IterDataset):
    """A shard's indexed rows read in order on the thread that draws them (`streamed_rows`), its position
    the next index, as Grain's prefetch writes its own, so a position either saved reads in the other."""

    def __init__(self, rows: grain.MapDataset):
        super().__init__(rows)
        self.rows = rows

    def __iter__(self) -> grain.DatasetIterator:
        return _IndexedRowsIterator(self.rows)


class _IndexedRowsIterator(grain.DatasetIterator):
    """`_IndexedRows`' reads."""

    def __init__(self, rows: grain.MapDataset):
        super().__init__()
        self.rows, self.next_index = rows, 0

    def __next__(self):
        if self.next_index >= len(self.rows):
            raise StopIteration
        row = self.rows[self.next_index]
        self.next_index += 1
        return row

    def get_state(self) -> Mapping[str, object]:
        return {"next_index": self.next_index}

    def set_state(self, state: Mapping[str, object]) -> None:
        self.next_index = integer(state["next_index"], "shard rows' next index")
