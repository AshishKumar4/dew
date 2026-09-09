"""Token datasets over a directory that `tools/tokenize_text.py` wrote:
`train.bin`, `val.bin` and `meta.json`.

`TokenWindows` reads fixed `seq_len + 1` windows off the token stream;
`PackedTokens` packs whole documents into windows of that size and carries
the segment ids and positions the backbone's mask needs. Train shuffles from
`seed`, reshuffled per epoch, and runs forever; val reads `val.bin` once, in
file order, in whole batches, so every validation pass scores the same
windows. Both shard by JAX process.

Both resume from a global record count, because both put the sharding last.
`TokenWindows` reads windows off the stream at a fixed stride; `PackedTokens`
plans its packing over the whole corpus in file order and reads windows off
that plan. A step is then the same windows at any process count, and the
position a checkpoint holds is a place in one order rather than one
process's offset into its shard.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Callable, Iterator, overload

import grain.python as pygrain
import numpy as np

from dew.registry import datasets

from .dataset import (Batch, Dataset, DatasetSpec, Loading, describe, local_batch,
                      train_stream, validation_pass)


def token_files(path: str | None, name: str) -> tuple[str, str]:
    """`(train.bin, val.bin)` of a tokenized directory, both required.

    Reading train.bin in val.bin's place would score the validation pass on
    the windows the model trains on.
    """
    if not path:
        raise ValueError(f"{name} needs path= set to the directory tools/tokenize_text.py wrote")
    root = Path(path)
    train_bin, val_bin = root / "train.bin", root / "val.bin"
    if not train_bin.is_file():
        raise ValueError(f"{path} has no train.bin; tools/tokenize_text.py writes one")
    if not val_bin.is_file():
        raise ValueError(
            f"{path} has a train.bin but no val.bin; tools/tokenize_text.py "
            "--val-fraction writes the held-out split")
    return str(train_bin), str(val_bin)


class _BoundedIterator:
    def __init__(self, source: Iterator[Batch], batches: int):
        self._source: Iterator[Batch] | None = source
        self._iterator: Iterator[Batch] = itertools.islice(source, batches)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def request_stop(self):
        stop = getattr(self._source, "request_stop", None)
        if stop is not None:
            stop()

    def close(self):
        try:
            close = getattr(self._source, "close", None)
            if close is not None:
                close()
        finally:
            self._iterator = iter(())
            self._source = None


def bounded(stream: Callable[[], Iterator[Batch]], batches: int | None) -> Callable[[], Iterator[Batch]]:
    """Limit validation batches while preserving owned source shutdown."""
    if batches is None:
        return stream
    if batches < 0:
        raise ValueError("validation batch limit must be nonnegative")
    if batches == 0:
        return lambda: iter(())
    return lambda: _BoundedIterator(stream(), batches)


@datasets("token_windows")
@dataclasses.dataclass(frozen=True)
class TokenWindows(DatasetSpec):
    """Fixed windows of `seq_len + 1` ids, each starting `seq_len` after the
    last, so record i's last token is record i + 1's first and the model sees
    every transition once. A batch is `{"text": int32 [batch, seq_len + 1]}`.
    `val_batches` bounds a validation pass; None scores all of val.bin. The
    training stream's saved position is a global window count, so a run
    resumes on any process count the global batch divides over."""

    path: str | None = None
    seq_len: int = 256
    val_batches: int | None = 4
    seed: int = 0
    loading: Loading = Loading()

    def load(self, *, batch: int) -> Dataset:
        from .sources.text import TokenFileSource

        train_bin, val_bin = token_files(self.path, "TokenWindows")
        train = TokenFileSource(train_bin, self.seq_len)
        val = TokenFileSource(val_bin, self.seq_len)
        return Dataset(
            train=train_stream(train, [], batch=local_batch(batch),
                               seed=self.seed, loading=self.loading),
            val=bounded(validation_pass(val, [], batch=local_batch(batch), seed=self.seed, loading=self.loading), self.val_batches),
            records=len(train),
            batch=batch,
        )


def chunk_counts(lengths, chunk_len: int):
    """How many chunks of at most `chunk_len` tokens each entry of `lengths` is cut into."""
    return -(-np.asarray(lengths, np.int64) // chunk_len)


def chunk_lengths(lengths, chunk_len: int) -> np.ndarray:
    """The token count of every chunk `DocumentChunks` cuts `lengths` into.

    A document of `chunk_len` tokens or fewer is one chunk of its own length;
    a longer one is whole chunks and a remainder. A document of no tokens is
    no chunks, as `chunk_counts` says.
    """
    lengths = np.asarray(lengths, np.int64)
    counts = chunk_counts(lengths, chunk_len)
    sizes = np.full(int(counts.sum()), chunk_len, np.int64)
    kept = counts > 0
    last = np.cumsum(counts)[kept] - 1
    sizes[last] = lengths[kept] - (counts[kept] - 1) * chunk_len
    return sizes


def first_fit(sizes: np.ndarray, window: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """The plan grain's first-fit packer would follow over chunks of `sizes`.

    Grain's packer holds `bins` open windows, adds each record to the first
    with room, and emits all of them when a record fits in none
    (`grain/_src/python/dataset/transformations/packing.py:341-355`). Over
    the lengths alone that is a plan rather than a pass: which window every
    chunk belongs to, without reading a token. The plan is what lets the
    packing run ahead of the shard, over the whole corpus in file order, so
    a window holds the same chunks at any process count and the loader can
    shard windows the way it shards any other record.

    Returns the chunk indices grouped by window, in the order the packer
    added them, and where each window starts in that grouping. A window the
    packer never added anything to is dropped: it would be a batch row of
    pure padding, which trains on nothing. Only the last set of bins can
    hold one, because a record goes to the first window with room and an
    empty window has room for anything.
    """
    if bins < 1:
        raise ValueError(f"a packer fills at least one window at a time, got {bins}")
    longest = int(sizes.max()) if len(sizes) else 0
    if longest > window:
        raise ValueError(
            f"a chunk of {longest} tokens does not fit a window of {window}; "
            f"documents are cut to the window before they are packed")
    room = [window] * bins
    plan = np.empty(len(sizes), np.int64)
    closed = 0
    for index, size in enumerate(sizes.tolist()):
        chosen = -1
        for candidate, free in enumerate(room):
            if free >= size:
                chosen = candidate
                break
        if chosen < 0:
            # Every open window is short of room, so the packer emits all of
            # them and starts this chunk in the first of a fresh set.
            closed += bins
            room = [window] * bins
            chosen = 0
        plan[index] = closed + chosen
        room[chosen] -= size
    windows = closed + bins - room.count(window)
    starts = np.concatenate([np.zeros(1, np.int64),
                             np.cumsum(np.bincount(plan, minlength=windows))])
    return np.argsort(plan, kind="stable"), starts


class DocumentChunks(pygrain.MapDataset[Batch]):
    """Documents cut into consecutive chunks of at most `chunk_len` tokens.

    A window holds nothing longer than itself, so a document that outgrows
    the window is cut first; each chunk becomes its own segment in the packed
    window, which keeps attention inside the chunk and RoPE running from the
    chunk's own 0.

    The chunk table is built once from the document lengths. A record then
    costs one read of its document and a slice.
    """

    def __init__(self, parent: pygrain.MapDataset, lengths, chunk_len: int):
        super().__init__(parent)
        self._chunk_len = chunk_len
        lengths = np.asarray(lengths, np.int64)
        counts = chunk_counts(lengths, chunk_len)
        self._document = np.repeat(np.arange(len(lengths), dtype=np.int64), counts)
        first_chunk = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self._offset = (np.arange(len(self._document), dtype=np.int64)
                        - first_chunk[self._document]) * chunk_len

    def __len__(self) -> int:
        return len(self._document)

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset[Batch]: ...

    @overload
    def __getitem__(self, index: int) -> Batch: ...

    def __getitem__(self, index):
        # grain's slice is the sharding and windowing API (ds[shard::count]),
        # and an index past the end wraps, so `repeat` is a length change.
        if isinstance(index, slice):
            return self.slice(index)
        index = index % len(self)
        document = self._parent[int(self._document[index])]
        if document is None:
            raise ValueError(f"document {index} of the packed corpus is missing")
        start = int(self._offset[index])
        # Every per-token field is cut the same way, so ids and roles stay
        # aligned inside the chunk.
        return {key: value[start:start + self._chunk_len] for key, value in document.items()}


class PackedWindows(pygrain.MapDataset[Batch]):
    """Documents packed into windows of `window` tokens, by one plan over the
    whole corpus.

    Every window carries, beside each per-token field, `<field>_segment_ids`
    (which chunk of the window each token came from, counted from 1, and 0
    for the padding at the end) and `<field>_positions` (the token's place
    inside its chunk), the two arrays a block-diagonal mask and per-document
    RoPE read; grain's packer writes the same pair per packed feature
    (`grain/_src/python/dataset/transformations/packing_packed_batch.py:116-117`).
    Chunks are cut the same way in every field, so one pair describes them
    all and the arrays are shared rather than copied per field.

    A window is random access, which is the point: `first_fit` plans the
    packing from the document lengths, in file order, ahead of any sharding,
    so window w holds the same chunks in every process of every process
    count and the training stream can shuffle, shard and count windows the
    way it does the records of any other source. Packing behind the shard,
    as this loader did before, made a window a fact about one process's own
    documents and its saved position a shard offset.

    `documents` is any dataset of per-token fields whose lengths are
    `lengths`, and `described` names it the way a saved position needs
    (`describe`), since which chunks share a window is part of the order the
    position counts into.
    """

    def __init__(self, documents: pygrain.MapDataset[Batch], lengths, window: int,
                 bins: int, described: str):
        super().__init__(DocumentChunks(documents, lengths, window))
        self._window = window
        self._described = described
        self._order, self._starts = first_fit(chunk_lengths(lengths, window), window, bins)

    def __repr__(self) -> str:
        # A saved position names the order it counts into (`dew.position`),
        # and which chunks share a window is part of that order.
        return (f"PackedWindows({self._described}, window={self._window}, "
                f"windows={len(self)})")

    def __len__(self) -> int:
        return len(self._starts) - 1

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset[Batch]: ...

    @overload
    def __getitem__(self, index: int) -> Batch: ...

    def __getitem__(self, index):
        # grain's slice is the sharding and windowing API (ds[shard::count]),
        # and an index past the end wraps, so `repeat` is a length change.
        if isinstance(index, slice):
            return self.slice(index)
        index = index % len(self)
        chunks = []
        for position in self._order[self._starts[index]:self._starts[index + 1]]:
            chunk = self._parent[int(position)]
            if chunk is None:
                raise ValueError(f"chunk {int(position)} of the packed corpus is missing")
            chunks.append(chunk)
        fields = {key: np.zeros(self._window, value.dtype)
                  for key, value in chunks[0].items()}
        segment_ids = np.zeros(self._window, np.int32)
        positions = np.zeros(self._window, np.int32)
        filled = 0
        for segment, chunk in enumerate(chunks, start=1):
            length = len(next(iter(chunk.values())))
            for key, value in chunk.items():
                fields[key][filled:filled + length] = value
            segment_ids[filled:filled + length] = segment
            positions[filled:filled + length] = np.arange(length, dtype=np.int32)
            filled += length
        return {**fields,
                **{f"{key}_segment_ids": segment_ids for key in fields},
                **{f"{key}_positions": positions for key in fields}}


@datasets("packed_tokens")
@dataclasses.dataclass(frozen=True)
class PackedTokens(DatasetSpec):
    """Whole documents packed into `seq_len + 1` windows.

    Documents come from `TokenDocumentSource`, which cuts the token stream at
    the eos ids the tokenize tool writes between files (`--pack`). Each
    document (in chunks, when it outgrows the window) is one element the
    packer adds to the first window with room, and every window carries
    `text_segment_ids` (which document each token is from, 0 for padding) and
    `text_positions` (the token's position inside its document), so the model
    can stop attention and the loss at document boundaries.

    The packing is planned over the whole corpus in file order, ahead of the
    shard (`PackedWindows`), so a window is a fact about the corpus rather
    than about one process's documents: the training stream shuffles and
    shards windows the way it does any other record, and its saved position
    is a global window count that resumes on any process count. Which
    documents share a window is then the same in every run over that corpus,
    and the seed decides the order the windows come in.

    `records` counts the windows a pass over the split holds, exactly, so
    `steps_per_epoch` is that pass. `val_batches` bounds a validation pass;
    None scores all of val.bin.
    """

    path: str | None = None
    seq_len: int = 256
    val_batches: int | None = 4
    seed: int = 0
    loading: Loading = Loading()
    packing_bins: int = 8
    """Windows the plan keeps open at once. More of them leave less padding
    in a window and let documents further apart in the file share one."""

    def load(self, *, batch: int) -> Dataset:
        from .sources.text import TokenDocumentSource

        train_bin, val_bin = token_files(self.path, "PackedTokens")
        rows, window = local_batch(batch), self.seq_len + 1

        # One source per split, and one plan over it. Finding the boundaries
        # reads the whole file, so rebuilding either per epoch would read a
        # multi-gigabyte train.bin again for a table the run already has.
        def packed(path: str) -> PackedWindows:
            source = TokenDocumentSource(path)
            return PackedWindows(pygrain.MapDataset.source(source), source.lengths, window,
                                 self.packing_bins, describe(source))

        train, val = packed(train_bin), packed(val_bin)

        return Dataset(
            train=train_stream(train, [], batch=rows, seed=self.seed, loading=self.loading),
            val=bounded(validation_pass(val, [], batch=rows, seed=self.seed,
                                        loading=self.loading), self.val_batches),
            records=len(train),
            batch=batch,
        )
