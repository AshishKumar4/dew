"""A Hugging Face `IterableDataset` as a grain `IterDataset`.

A streamed split is not random access: no length, no index, no second look at
a row that has gone by. So it cannot be a grain source, and nothing here may
list one. What it can be is the head of a grain iterator pipeline, which is
what `HFRows` is: the rows of one process's share, in order, once per pass.
Everything behind it -- the per-record transform, the batch, the bounded
buffer ahead of the step -- is grain's own, the same machinery the
random-access paths use.

Two honest limits are written into the types here.

`split_dataset_by_node` does the sharding, not `IterableDataset.shard`. A
stream's physical shard count is a property of how the data was written; on
a split with two shards, `shard(num_shards=4, index=3)` raises IndexError,
so a pool larger than the shard count would lose ranks. The public node
split assigns whole shards when the counts divide and keeps one row in
`world_size` otherwise, which every rank can do.

`HFRows` reports no position, so a run over one is not checkpointable. What
a streamed split would restore is not the record sequence a run consumed:
`datasets`' state covers its own iterables, while the shuffle buffer, the
shard-to-node assignment and the pass number are part of the order dew
reads, and a restore that put some of that back would resume onto a
sequence nobody verified. `Trainer.fit` already refuses `checkpoint_every`
over a stream that reports no position, and that refusal is the contract
this path takes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Optional

import grain.python as pygrain

Row = Mapping[str, object]
Options = Mapping[str, object]

STREAMING_HINT = ("streaming a Hugging Face dataset needs the streaming extra: "
                  "pip install 'dew-ml[streaming]'")

NO_POSITION = (
    "a streamed Hugging Face split reports no position: its shuffle buffer, "
    "its shard-to-node assignment and its pass number are part of the order "
    "dew reads and are not in what `datasets` restores, so no saved state "
    "names the record sequence a run consumed. Train it with "
    "checkpoint_every=None, or read the split without streaming=True, which "
    "is random access and resumes on any process count.")


def _datasets():
    """The HF `datasets` module, imported on use so `import dew.data` is cheap."""
    try:
        import datasets
    except ImportError as missing:
        raise ImportError(STREAMING_HINT) from missing
    return datasets


def open_rows(name: str, split: str, *, options: Options, seed: int, epoch: int,
              rank: int, world_size: int, buffer: int):
    """One process's rows of `split`, shuffled through a bounded buffer.

    The shuffle comes before the node split and takes a seed every rank
    computes the same way, because the split is over the shuffled shard list:
    two ranks that shuffled differently would disagree about which shards
    each of them owns. `epoch` goes into that seed, so a second pass over the
    same shard is a different order and no pass is held in memory.
    """
    datasets = _datasets()
    rows = datasets.load_dataset(name, split=split, streaming=True, **options)
    if not isinstance(rows, datasets.IterableDataset):
        raise TypeError(
            f"streaming {name!r} split {split!r} gave "
            f"{type(rows).__name__}; a streamed split is an IterableDataset, so "
            f"name one split rather than a whole dataset")
    if buffer > 0:
        rows = rows.shuffle(seed=seed + epoch, buffer_size=buffer)
    if world_size > 1:
        from datasets.distributed import split_dataset_by_node

        rows = split_dataset_by_node(rows, rank=rank, world_size=world_size)
    return rows


class _Rows(pygrain.DatasetIterator):
    """The rows of one process's share, pass after pass.

    `epochs` of None reopens the share when it runs out, which is what an
    endless training stream is; `epochs=1` stops after one pass, which is
    what a validation pass is.
    """

    def __init__(self, open_pass: Callable[[int], object], epochs: Optional[int]):
        super().__init__()
        self._open_pass = open_pass
        self._epochs = epochs
        self._epoch = 0
        self._rows: Optional[Iterator[Row]] = None

    def __next__(self) -> Row:
        while True:
            if self._epochs is not None and self._epoch >= self._epochs:
                raise StopIteration
            if self._rows is None:
                self._rows = iter(self._open_pass(self._epoch))  # type: ignore[call-overload]
            row = next(self._rows, None)
            if row is not None:
                return row
            self._rows = None
            self._epoch += 1

    def get_state(self) -> dict[str, object]:
        """The pass this iterator is on, which is all it knows.

        Grain's iterator protocol declares the pair, and grain's own thread
        prefetch reads the parent's state when it starts, so this cannot
        raise. It is not a resume point: `Unresumable` keeps it away from the
        trainer, and `set_state` says why.
        """
        return {"epoch": self._epoch}

    def set_state(self, state: Mapping[str, object]) -> None:
        raise NotImplementedError(NO_POSITION)

    def close(self) -> None:
        self._rows = None


class HFRows(pygrain.IterDataset):
    """A streamed split's rows for this process, as a grain `IterDataset`.

    Constructing one opens nothing; the first iterator opens the split. Every
    grain iterator transformation applies to it, which is how the streamed
    path shares the transform and buffering machinery with the random-access
    ones rather than growing a reader of its own.
    """

    def __init__(self, name: str, split: str, *, options: Options, seed: int,
                 rank: int, world_size: int, buffer: int, epochs: Optional[int]):
        super().__init__()
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not one of {world_size} processes")
        if epochs is not None and epochs < 1:
            raise ValueError("a streamed pass count is positive or None for endless")
        self.name = name
        self.split = split
        self.options = dict(options)
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.buffer = buffer
        self.epochs = epochs

    def __repr__(self) -> str:
        return (f"HFRows(name={self.name!r}, split={self.split!r}, "
                f"options={sorted(self.options.items())!r}, seed={self.seed}, "
                f"shard={self.rank}/{self.world_size}, buffer={self.buffer}, "
                f"epochs={self.epochs})")

    def _pass(self, epoch: int):
        return open_rows(self.name, self.split, options=self.options, seed=self.seed,
                         epoch=epoch, rank=self.rank, world_size=self.world_size,
                         buffer=self.buffer)

    def __iter__(self) -> pygrain.DatasetIterator:
        return _Rows(self._pass, self.epochs)


class Unresumable:
    """`iterator` with its position withheld, because it does not have one.

    The grain iterators behind a streamed split carry `get_state` and
    `set_state` because grain's iterator protocol declares them, and the
    rows underneath cannot honour either. Forwarding them would make a run
    look checkpointable and resume it onto a sequence nobody verified, so
    this hands on the batches and the shutdown and nothing else. `Trainer.fit`
    reads the pair statically and refuses `checkpoint_every` over this.
    """

    def __init__(self, iterator: pygrain.DatasetIterator):
        self._iterator = iterator

    def __iter__(self) -> Iterator[Row]:
        return self

    def __next__(self) -> Row:
        return next(self._iterator)

    def close(self) -> None:
        self._iterator.close()
