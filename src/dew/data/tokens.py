"""Reads token datasets off a tokenized corpus directory.

The corpus is the `train.bin`, `val.bin` and `meta.json` that
`dew tokenize` writes, or the same splits as ArrayRecord shards
(`dew.data.sources.text`).

`TokenWindows` reads `seq_len + 1` windows off the token stream, at a fixed
stride or packed with whole documents and the segment ids and positions the
backbone's mask needs. Train shuffles from `seed`, reshuffles per epoch and
runs forever; val reads `val.bin` once, in file order, in whole batches, so
every validation pass scores the same windows. Both shard by JAX process.

The sharding comes last, so a run resumes from a global record count: the
strided windows are fixed spans of the stream, and packing is planned over
the whole corpus in file order. A step is then the same windows at any
process count, and a saved position is a place in one order rather than one
process's offset into its shard.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator, Mapping
from typing import Annotated, overload

import grain.python as pygrain
import numpy as np

from .dataset import (
    Batch,
    Corpus,
    DataPartition,
    DataPhase,
    Dataset,
    DatasetSpec,
    Forwarding,
    Reader,
    Records,
    Tokenize,
    describe,
    record_argument,
)
from .sources.text import HubText, _Documents, _Windows, same_tokenizer, token_corpus


class _BoundedIterator(Forwarding):
    """Yields the first `batches` batches of `source` and closes `source`.

    `Forwarding.close` reaches the source, so stopping early still releases
    the grain workers the source owns.
    """

    def __init__(self, source: Iterator[Batch], batches: int):
        self._source: Iterator[Batch] | None = source
        self._iterator: Iterator[Batch] = itertools.islice(source, batches)

    @property
    def endless(self) -> bool:
        return False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        try:
            super().close()
        finally:
            self._iterator = iter(())
            self._source = None


def bounded(stream: Reader, batches: int | None) -> Reader:
    """`stream` stopped after `batches` batches, or `stream` itself when `batches` is None.

    The wrapper forwards close to the source, so a caller that stops early
    still releases the grain workers the stream owns.
    """
    if batches is None:
        return stream
    if batches < 0:
        raise ValueError("validation batch limit must be nonnegative")
    if batches == 0:
        return lambda partition: iter(())

    def first(partition: DataPartition) -> Iterator[Batch]:
        return _BoundedIterator(stream(partition), batches)

    return first


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
    (`grain/_src/python/dataset/transformations/packing.py:341-355`). Run
    over the lengths alone it is a plan rather than a pass: which window
    every chunk belongs to, without reading a token. That plan is what lets
    the packing run ahead of the shard, so a window holds the same chunks at
    any process count.

    Returns the chunk indices grouped by window, in the order the packer
    added them, and where each window starts in that grouping. A window the
    packer never added anything to is dropped, since it would be a batch row
    of pure padding. Only the last set of bins can hold one, because a record
    goes to the first window with room and an empty window has room for
    anything.
    """
    if bins < 1:
        raise ValueError(f"a packer fills at least one window at a time, got {bins}")
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


class _WrappingDataset(pygrain.MapDataset[Batch]):
    """A dataset read by index, whose index wraps into its own length.

    grain's slice is the sharding and windowing API (`ds[shard::count]`), and
    an index past the end wraps, so `repeat` is a length change. Both are
    answered here and a subclass reads one record through `record`.
    """

    def record(self, index: int) -> Batch:
        """The record at `index`, which is inside this dataset's length."""
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset[Batch]: ...

    @overload
    def __getitem__(self, index: int) -> Batch: ...

    def __getitem__(self, index: int | slice) -> Batch | pygrain.MapDataset[Batch]:
        if isinstance(index, slice):
            return self.slice(index)
        return self.record(index % len(self))


class DocumentChunks(_WrappingDataset):
    """Cuts documents into consecutive chunks of at most `chunk_len` tokens.

    A window holds nothing longer than itself, so a document that outgrows
    the window is cut first. Each chunk becomes its own segment in the packed
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

    def record(self, index: int) -> Batch:
        document = self._parent[int(self._document[index])]
        if document is None:
            raise ValueError(f"document {index} of the packed corpus is missing")
        start = int(self._offset[index])
        # Every per-token field is cut the same way, so ids and roles stay
        # aligned inside the chunk.
        return {key: value[start:start + self._chunk_len] for key, value in document.items()}


class PackedWindows(_WrappingDataset):
    """Packs documents into windows of `window` tokens, by one plan over the
    whole corpus.

    Every window carries its per-token fields and, named after the first of
    them (the ids), `<field>_segment_ids` (which chunk of the window each
    token came from, counted from 1, and 0 for the padding at the end) and
    `<field>_positions` (the token's place inside its chunk). A block-diagonal
    mask and per-document RoPE read that pair. grain's packer writes one
    pair per packed feature
    (`grain/_src/python/dataset/transformations/packing_packed_batch.py:116-117`);
    chunks are cut the same way in every field here, so one pair describes
    them all and nothing transfers a second copy to the device.

    A window is read by index. `first_fit` plans the packing from the
    document lengths alone, in file order, before any sharding. Window w
    therefore holds the same chunks at every process count, and the training
    stream shuffles, shards and counts windows as it does any other record.

    `documents` is any dataset of per-token fields whose lengths are
    `lengths`. `described` names it the way a saved position needs
    (`describe`), since which chunks share a window is part of the order the
    position counts into.
    """

    def __init__(self, documents: pygrain.MapDataset[Batch], lengths, window: int,
                 bins: int, described: str):
        super().__init__(DocumentChunks(documents, lengths, window))
        self._window, self._bins = window, bins
        self._described = described
        self._order, self._starts = first_fit(chunk_lengths(lengths, window), window, bins)

    def __repr__(self) -> str:
        # A saved position names the order it counts into (`dew.position`),
        # and which chunks share a window is part of that order.
        return (f"PackedWindows({self._described}, window={self._window}, "
                f"bins={self._bins}, windows={len(self)})")

    def __len__(self) -> int:
        return len(self._starts) - 1

    def record(self, index: int) -> Batch:
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
        ids = next(iter(fields))
        return {**fields, f"{ids}_segment_ids": segment_ids, f"{ids}_positions": positions}


@dataclasses.dataclass(frozen=True)
class TokenWindows(DatasetSpec):
    """Reads windows of `seq_len + 1` ids from tokenized corpora, as spans of
    the stream or packed with whole documents.

    `path` is the directory that `dew tokenize` or `TokenCorpus.write` wrote.
    `hub` reads a Hugging Face text split instead, tokenized once into dew's
    cache (`HubText`), which is what `load("hf/<name>", tokenizer=,
    seq_len=)` builds. A batch is `{"text": int32 [batch, seq_len + 1]}`.

    A window is a contiguous span of the stream by default. Training windows
    start `stride` ids apart, `seq_len` by default; a stride of one lets the
    shuffled training stream read every contiguous window. Validation always
    starts windows `seq_len` ids apart, so it counts each target once.

    With `pack`, whole documents fill the windows instead. They are cut at the
    eos ids the tokenize tool writes after each document (`--pack`), and the
    packer adds each to the first of `packing_bins` open windows with room,
    split into chunks when it is longer than a window. Every window then also
    has `text_segment_ids` (which document each token is from, 0 for padding)
    and `text_positions` (the token's position inside its document), so the
    model can stop attention and the loss at document boundaries. The packing
    is planned over the whole corpus in file order, before the data is
    sharded (`PackedWindows`), so every run over the same corpus packs the
    same documents together and the seed decides only the order the windows
    come in.

    `path` may also map several directories to the share of a step each
    fills, like MaxText's weighted `grain_train_files`. Each corpus is read by
    its own order (and packed by its own plan, so a window holds one
    corpus's documents), and `mixture` interleaves them at their weights
    before the data is sharded. The weights are shares of the windows a step
    reads, and so of its tokens. The corpora must come from one tokenizer,
    as their `meta.json` records.

    `phases` replaces `path` for a run that switches its data at step
    boundaries. Each `DataPhase` names a corpus or mixture and the step it
    ends before, and the last phase runs to the end. A phase that reads a
    corpus again continues that corpus's shuffled order where the earlier
    phases left it, so no window repeats before its corpus's epoch ends
    (`dew.data.providers.phased_dataset`). On resume, the run checks the
    phases it has already read, and you may append phases or move a boundary
    it has not reached. Validation reads the first phase's held-out split.

    The training stream's saved position is a global window count, so a run
    can resume with any number of processes that divides the global batch.
    `records` is exactly the windows in one pass over the split, so
    `steps_per_epoch` is that pass; for a mixture, a pass is the windows in
    which every corpus has been read at least once (`mixed_records`), and for
    a phased run it is the first phase's pass. `val_batches` caps the
    batches in a validation pass; None scores the whole split, a mixture's
    held-out splits mixed at the same weights.
    """

    path: str | Mapping[str, float] | None = None
    seq_len: int = 256
    val_batches: int | None = 4
    field: str | None = None
    """The arrayrecord field that holds the ids, for a corpus stored as ArrayRecord shards of dict
    records; None reads each record's bytes as the ids. A `.bin` corpus ignores it."""
    stride: int | None = dataclasses.field(default=None, kw_only=True)
    pack: bool = dataclasses.field(default=False, kw_only=True)
    packing_bins: int = dataclasses.field(default=8, kw_only=True)
    """The windows packing keeps open at once. More of them leave less padding in a window and let
    documents further apart in the file share one."""
    phases: Annotated[tuple[DataPhase, ...], record_argument(tuple[DataPhase, ...])] = dataclasses.field(
        default=(), kw_only=True)
    hub: HubText | None = dataclasses.field(default=None, kw_only=True)

    @property
    def corpora(self) -> list[str]:
        """Every tokenized directory the run reads, across `path` and `phases`, in name order."""
        from .providers import name_ordered
        named = {name for phase in self.phases for name in name_ordered(phase.path)}
        return sorted(named | set(name_ordered(self.path)))

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        from .providers import corpora_dataset, name_ordered, phased_dataset

        self.uncaptioned(tokenize)
        if self.hub is not None and (self.path or self.phases):
            raise ValueError("TokenWindows reads hub= alone, without path= or phases=")
        if self.phases and self.path:
            raise ValueError("TokenWindows reads path= or phases=, not both")
        if self.pack and self.stride is not None:
            raise ValueError("a packed window holds whole documents, so packing takes no stride")
        weighted = {self.hub.tokenized(): 1.0} if self.hub is not None else name_ordered(self.path)
        if not weighted and not self.phases:
            raise ValueError("TokenWindows needs path= set to the directory `dew tokenize` "
                             "wrote, several with weights, phases= or hub=")
        same_tokenizer(self.corpora)
        # One pair of sources per corpus, read once. Packing finds the
        # document boundaries by reading the whole file, so rebuilding it per
        # phase would read a multi-gigabyte train.bin again for a table the
        # run already has.
        splits: dict[str, tuple[Records, Records]] = {}

        def corpora(named: Mapping[str, float]) -> tuple[list[Corpus], list[Corpus]]:
            train, held = [], []
            for path, weight in named.items():
                if path not in splits:
                    splits[path] = self._windows(*token_corpus(path, "TokenWindows", field=self.field))
                train.append(Corpus(path, splits[path][0], weight))
                held.append(Corpus(path, splits[path][1], weight))
            return train, held

        if not self.phases:
            train, held = corpora(weighted)
            return corpora_dataset(train, held, [], batch=batch, seed=self.seed,
                                   loading=self.loading, val_batches=self.val_batches)
        phases = [(corpora(name_ordered(phase.path)), phase.until_step) for phase in self.phases]
        return phased_dataset([(train, until) for (train, _), until in phases], phases[0][0][1], [],
                              batch=batch, seed=self.seed, loading=self.loading,
                              val_batches=self.val_batches)

    def _windows(self, corpus, held_out) -> tuple[Records, Records]:
        """The training and validation windows of one corpus's two splits."""
        if not self.pack:
            return _Windows(corpus, self.seq_len, stride=self.stride), _Windows(held_out, self.seq_len)
        window = self.seq_len + 1

        def packed(tokens) -> PackedWindows:
            # Cut where the packer would, so a chunk of a long document reads
            # its own span instead of the whole document.
            source = _Documents(tokens, chunk_len=window)
            return PackedWindows(pygrain.MapDataset.source(source), source.lengths, window,
                                 self.packing_bins, describe(source))

        return packed(corpus), packed(held_out)
