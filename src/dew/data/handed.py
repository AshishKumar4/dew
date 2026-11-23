"""Data handed to `fit` as it is (`Handed`): a Grain dataset's elements or arrays whole, each a batch the
run steps on (`native`), and the loader a run reads it through (`native_loader`)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import grain.python as pygrain
import numpy as np

from dew.batch import Batch
from dew.data.dataset import (
    Closeable,
    DataPartition,
    Ends,
    FilledBatch,
    Forwarding,
    Loader,
    Reader,
    rows_of,
    take_batch,
)
from dew.data.readers import Replayed

type Handed = pygrain.MapDataset | pygrain.IterDataset | Mapping[str, np.ndarray]
"""Data handed to `fit` as it is, which a run steps on as given (`native`): a Grain dataset's
elements, or arrays whole."""


class _Whole:
    """The same batch every step, the arrays a run was handed whole; its position the steps read. It
    never runs out (`Identified.ends`), and names what it reads as a stream Dew names no further does:
    by its type (`stream_identity`)."""

    ends: Ends = "never"

    def __init__(self, batch: Batch):
        self.batch, self.read = batch, 0

    def identity(self) -> str:
        return type(self).__name__

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        self.read += 1
        return self.batch

    def get_state(self) -> bytes:
        return str(self.read).encode()

    def set_state(self, state: bytes) -> None:
        self.read = int(state)


def _share(batch: Batch, partition: DataPartition, *, held: bool, nominal: int = 0) -> Batch:
    """`partition`'s share of `batch` (`DataPartition`: rows `index :: count`); `held`, a validation
    share, filled out to every share's rows of the pass's `nominal` batch (its largest, its first) or of
    this one where larger, a whole number of its row shards, every row marked (`DataPartition.held`,
    `FilledBatch`), so every share reads a batch its devices split where any does, a pass's short last
    batch in its whole ones' shape and tree, and every real row scores once (row 12474)."""
    rows = rows_of(batch)
    if not held:
        return batch if partition.count == 1 else take_batch(batch, np.arange(rows)[partition.cut()])
    mine, real = partition.held(rows) if partition.count > 1 else (slice(None), True)
    shards = partition.count * partition.row_multiple
    filled = -(-max(rows, nominal) // shards) * partition.row_multiple
    return FilledBatch(filled, real=real, marked=True).map(take_batch(batch, np.arange(rows)[mine]))


class _Shared(Forwarding):
    """The `partition`'s share of each batch a stream yields (`_share`), its position, stop and close
    the stream's; it never ends where it reads a map dataset repeated without end (`native`, `endless`),
    which says so of nothing it reads."""

    def __init__(self, batches: pygrain.DatasetIterator | _Whole | Replayed, partition: DataPartition, *,
                 held: bool, endless: bool = False):
        self._source, self.partition, self.held, self._endless = batches, partition, held, endless
        # A held pass's batch, its first, which its short last one is filled out to (`_share`).
        self._nominal = 0

    @property
    def ends(self) -> Ends:
        return "never" if self._endless else super().ends

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        batch = next(self._source)
        self._nominal = max(self._nominal, rows_of(batch))
        return _share(batch, self.partition, held=self.held, nominal=self._nominal)

    def get_state(self):
        return self._source.get_state()

    def set_state(self, state) -> None:
        self._source.set_state(state)


def native(handed: Handed, *, training: bool) -> Reader:
    """A reader of the data handed to `fit` as it is, each element a batch the run steps on, never
    batched again: a Grain dataset's elements (batched by Grain, `MapDataset.batch`), a map dataset's
    read endlessly where it trains, or a mapping of arrays read whole, every step the same batch (a
    full-batch run; a run of minibatches states them, `Memory(arrays).batch(size)`)."""
    if isinstance(handed, Mapping):
        batch = {name: np.asarray(column) for name, column in handed.items()}
        if training:
            return lambda partition: _Shared(_Whole(batch), partition, held=False)
        return lambda partition: iter([_share(batch, partition, held=True)])
    repeated = training and isinstance(handed, pygrain.MapDataset)
    stream: pygrain.IterDataset = (
        (handed.repeat() if training else handed).to_iter_dataset() if isinstance(handed, pygrain.MapDataset)
        else handed)
    return lambda partition: _Shared(iter(stream), partition, held=not training, endless=repeated)


def native_loader(handed: Handed) -> Loader:
    """The data handed to `fit` as it is (`native`) as the loader the run reads, its global batch
    its first element's rows. A Grain iterable is read once for that element: its run's first stream
    gives it again and reads on (`Replayed`), as the batch contract asks (training.md; row 6663)."""
    if isinstance(handed, pygrain.IterDataset):
        probe = iter(handed)
        try:
            before = probe.get_state()
            first = next(probe)
        except BaseException:
            if isinstance(probe, Closeable):
                probe.close()
            raise
        replayed = [Replayed(probe, first, before)]

        def train(partition: DataPartition) -> Iterator[Batch]:
            return _Shared(replayed.pop() if replayed else iter(handed), partition, held=False)

        return Loader(train=train, val=None, records=None, batch=rows_of(_present(first)))
    first = handed[0] if isinstance(handed, pygrain.MapDataset) else {
        name: np.asarray(column) for name, column in handed.items()}
    batch = rows_of(_present(first))
    records = len(handed) * batch if isinstance(handed, pygrain.MapDataset) else batch
    return Loader(train=native(handed, training=True), val=None, records=records, batch=batch)


def _present(first):
    """`first`, a dataset's first element, refused where it is missing."""
    if first is None:
        raise ValueError("a Grain dataset handed to fit gives its batches, and its first is missing (None)")
    return first
