"""A Hugging Face `IterableDataset` as a grain `IterDataset`.

A streamed split is not random access: no length, no index, no second look at
a row that has gone by. So it cannot be a grain source, and nothing here
lists one. It is the head of a grain iterator pipeline instead, which is
what `HFRows` is: the rows of one process's share, in order, once per pass.
Everything behind it -- the per-record transform, the batch, the buffer
ahead of the step -- is grain's own.

Sharding is `datasets.distributed.split_dataset_by_node`, not
`IterableDataset.shard`. A stream's physical shard count is a property of
how the data was written; on a split with two shards,
`shard(num_shards=4, index=3)` raises IndexError, so a pool larger than the
shard count would lose ranks. The node split assigns whole shards when the
counts divide and keeps one row in `world_size` otherwise, which every rank
can do. A rank that still gets no rows at all is a misconfiguration, not a
stream to wait on, and `_Rows` says so rather than reopening an empty share
for ever.

Position: `datasets` restores an unshuffled stream exactly, and grain
composes that with the state of the transform and the batch behind it, so a
stream read in file order resumes on the record it stopped at with the same
per-record draws. Two kinds of stream cannot promise that. A stream dew
shuffles loses the contents of the buffer the shuffle was drawing from,
which are not in the state. A stream a caller hands over has come through
transformations dew did not apply and cannot see, and one of them may lose
as much; the library's own `.shuffle` is the example the reviewer found, and
there is no public way to ask an `IterableDataset` what it has been through.
`refusal` is that distinction, and `providers` withholds the state pair
wherever it is set, which is the non-checkpointable contract `Dataset`
already documents.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Optional

import grain.python as pygrain

if TYPE_CHECKING:  # `datasets` is imported on the first row, not at import
    from datasets import IterableDataset

Row = dict[str, object]

SHUFFLED = (
    "a shuffled streamed split cannot be put back where it stopped: what "
    "`datasets` restores is its place in the rows, and the contents of the "
    "shuffle buffer it was drawing from are not in it. Read the split with "
    "shuffle_buffer=0, which resumes exactly, or train it with "
    "checkpoint_every=None.")

UNSUPPORTED = (
    "`datasets` implements no state for this streamed split's source, so it "
    "cannot be put back where it stopped; train it with "
    "checkpoint_every=None.")

GIVEN = (
    "a streamed dataset handed to load() cannot be put back where it stopped: "
    "it arrives through transformations dew did not apply, an upstream "
    "`.shuffle` among the possibilities, and an `IterableDataset` cannot be "
    "asked what it has been through, so no position over it can be promised "
    "to restore the rows exactly. Name the split with data_files= or the "
    "dataset id, which dew builds and can resume, or train this one with "
    "checkpoint_every=None.")

NO_ROWS = (
    "process {rank} of {world_size} was given none of the rows of {what}. A "
    "streamed split is shared out row by row when its shards do not divide "
    "over the processes, so a split with fewer rows than the pool has "
    "processes leaves a rank with nothing to read and no batch to contribute. "
    "Read it on at most as many processes as it has rows, or read it without "
    "streaming=True.")


def shared(rows: "IterableDataset", *, rank: int, world_size: int) -> "IterableDataset":
    """`rows` reduced to the share process `rank` of `world_size` reads."""
    if world_size <= 1:
        return rows
    from datasets.distributed import split_dataset_by_node

    return split_dataset_by_node(rows, rank=rank, world_size=world_size)


def refusal(rows: "IterableDataset", *, shuffled: bool, given: bool) -> Optional[str]:
    """Why this stream cannot be put back where it stopped, or None if it can.

    Three questions, in the order they can be answered without reading a row:
    whether dew shuffled the stream, whether a caller supplied it and dew
    therefore does not know what it has been through, and whether `datasets`
    implements the state pair for its source at all, which is what calling
    `state_dict` once answers.
    """
    if shuffled:
        return SHUFFLED
    if given:
        return GIVEN
    state = getattr(rows, "state_dict", None)
    load = getattr(rows, "load_state_dict", None)
    if state is None or load is None:
        return UNSUPPORTED
    try:
        state()
    except (NotImplementedError, AttributeError, TypeError, KeyError):
        return UNSUPPORTED
    return None


class _Rows(pygrain.DatasetIterator):
    """The rows of one process's share, pass after pass.

    `epochs` of None reopens the share when it runs out, which is what an
    endless training stream is; `epochs=1` stops after one pass, which is
    what a validation pass is. A pass that yields nothing is the empty-share
    misconfiguration and raises, so the reopening cannot spin and a caller
    closing the pipeline is not waiting on a `next` that never returns.
    """

    def __init__(self, open_pass: Callable[[int], "IterableDataset"], *,
                 epochs: Optional[int], refused: Optional[str], what: str, rank: int,
                 world_size: int):
        super().__init__()
        self._open_pass = open_pass
        self._epochs = epochs
        self._refused = refused
        self._what = what
        self._rank = rank
        self._world_size = world_size
        self._epoch = 0
        self._read = 0
        self._restore: Optional[Mapping[str, object]] = None
        self._split: Optional["IterableDataset"] = None
        self._rows: Optional[Iterator[Row]] = None

    def _open(self) -> Iterator[Row]:
        split = self._split = self._open_pass(self._epoch)
        restore, self._restore = self._restore, None
        if restore is not None:
            split.load_state_dict(dict(restore))
        return iter(split)

    def __next__(self) -> Row:
        while True:
            if self._epochs is not None and self._epoch >= self._epochs:
                raise StopIteration
            rows = self._rows
            if rows is None:
                rows = self._rows = self._open()
            row = next(rows, None)
            if row is not None:
                self._read += 1
                return dict(row)
            if self._read == 0:
                raise ValueError(NO_ROWS.format(rank=self._rank,
                                                world_size=self._world_size,
                                                what=self._what))
            self._rows = self._split = None
            self._read = 0
            self._epoch += 1

    def get_state(self) -> dict[str, object]:
        """The pass, the rows read in it, and the library's own place in them.

        grain's iterator protocol declares the pair and grain's thread
        prefetch reads the parent's state when it starts, so this reports
        what it has whether or not the stream can be restored from it.
        `set_state` is where an unrestorable stream refuses.
        """
        split = self._split
        return {"epoch": self._epoch, "read": self._read,
                "rows": None if split is None else split.state_dict()}

    def set_state(self, state: Mapping[str, object]) -> None:
        if self._refused is not None:
            raise NotImplementedError(self._refused)
        epoch, read, rows = state["epoch"], state["read"], state["rows"]
        if not isinstance(epoch, int) or not isinstance(read, int):
            raise ValueError(f"a streamed position counts passes and rows, not {state}")
        if rows is not None and not isinstance(rows, Mapping):
            raise ValueError(f"a streamed position holds the library's state, not {rows}")
        self._epoch, self._read = epoch, read
        self._restore = rows
        self._split = None
        self._rows = None

    def close(self) -> None:
        self._rows = None
        self._split = None


class HFRows(pygrain.IterDataset):
    """A streamed split's rows for this process, as a grain `IterDataset`.

    Constructing one opens nothing; the first iterator opens the split.
    `open_split()` returns the whole split, freshly -- a pass may restore the
    library's state into it, so two passes sharing one object would leave the
    second starting where the first stopped. This shares it out, and
    shuffles it before the share when `shuffle_buffer` is set, because the
    share is over the shuffled shard list and two ranks that shuffled
    differently would disagree about which shards each of them owns. The
    pass number goes into the shuffle seed, so a second pass over the same
    share is a different order and no pass is held in memory.
    """

    def __init__(self, open_split: Callable[[], "IterableDataset"], *, what: str,
                 seed: int, rank: int, world_size: int, shuffle_buffer: int,
                 epochs: Optional[int], given: bool):
        super().__init__()
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not one of {world_size} processes")
        if epochs is not None and epochs < 1:
            raise ValueError("a streamed pass count is positive or None for endless")
        if shuffle_buffer < 0:
            raise ValueError("a shuffle buffer holds no rows or more")
        self._open_split = open_split
        self.what = what
        self.given = given
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer = shuffle_buffer
        self.epochs = epochs
        self._asked = threading.Lock()
        self._refused: Optional[Optional[str]] = ...  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (f"HFRows({self.what}, seed={self.seed}, "
                f"share={self.rank}/{self.world_size}, "
                f"shuffle_buffer={self.shuffle_buffer}, epochs={self.epochs}, "
                f"given={self.given})")

    def _pass(self, epoch: int) -> "IterableDataset":
        rows = self._open_split()
        if self.shuffle_buffer:
            rows = rows.shuffle(seed=self.seed + epoch, buffer_size=self.shuffle_buffer)
        return shared(rows, rank=self.rank, world_size=self.world_size)

    @property
    def refused(self) -> Optional[str]:
        """Why a position over these rows would not restore them, or None.

        Answered once: the question opens the split, which for a hub dataset
        is a request, and every iterator of these rows has the same answer.
        Under a lock because grain's prefetch thread reads it while the
        caller that made the iterator is still holding one.
        """
        with self._asked:
            if self._refused is ...:
                self._refused = refusal(self._pass(0),
                                        shuffled=bool(self.shuffle_buffer),
                                        given=self.given)
            return self._refused

    @property
    def resumable(self) -> bool:
        """Whether a position saved over these rows puts them back exactly."""
        return self.refused is None

    def __iter__(self) -> pygrain.DatasetIterator[Row]:
        return _Rows(self._pass, epochs=self.epochs, refused=self.refused,
                     what=self.what, rank=self.rank, world_size=self.world_size)


class Unresumable:
    """`iterator` with its position withheld, because it has none to give.

    The grain iterators behind a streamed split carry `get_state` and
    `set_state` because grain's iterator protocol declares them. Forwarding
    them for a stream that cannot be restored exactly would make a run look
    checkpointable and resume it onto a sequence nobody verified, so this
    hands on the batches and the shutdown and nothing else. `Trainer.fit`
    reads the pair statically and refuses `checkpoint_every` over it.
    """

    def __init__(self, iterator: pygrain.DatasetIterator[Row]):
        self._iterator = iterator

    def __iter__(self) -> Iterator[Row]:
        return self

    def __next__(self) -> Row:
        return next(self._iterator)

    def close(self) -> None:
        self._iterator.close()
