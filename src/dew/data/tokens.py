"""Reads token datasets off a tokenized corpus directory.

The corpus is the `train.bin`, `val.bin` and `meta.json` that
`tools/tokenize_text.py` writes, or the same splits as ArrayRecord shards or
parquet (`dew.data.sources.text`).

`TokenWindows` reads fixed `seq_len + 1` windows off the token stream.
`PackedTokens` packs whole documents into windows of that size and carries
the segment ids and positions the backbone's mask needs. Train shuffles from
`seed`, reshuffles per epoch and runs forever; val reads `val.bin` once, in
file order, in whole batches, so every validation pass scores the same
windows. Both shard by JAX process.

Both put the sharding last, so both resume from a global record count.
`TokenWindows` reads windows off the stream at a fixed stride, and
`PackedTokens` plans its packing over the whole corpus in file order. A step
is then the same windows at any process count, and a saved position is a
place in one order rather than one process's offset into its shard.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Annotated, Callable, Iterator, Mapping, overload

import grain.python as pygrain
import numpy as np

from dew.registry import datasets

from .dataset import (
    Batch,
    Corpus,
    DataPhase,
    Dataset,
    DatasetSpec,
    Forwarding,
    Tokenize,
    describe,
    json_list_argument,
    local_batch,
    train_stream,
    validation_pass,
)


class _BoundedIterator(Forwarding):
    """Yields the first `batches` batches of `source` and closes `source`.

    `Forwarding.close` reaches the source, so stopping early still releases
    the grain workers the source owns.
    """

    def __init__(self, source: Iterator[Batch], batches: int):
        self._source: Iterator[Batch] | None = source
        self._iterator: Iterator[Batch] = itertools.islice(source, batches)

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


def bounded(stream: Callable[[], Iterator[Batch]], batches: int | None) -> Callable[[], Iterator[Batch]]:
    """`stream` stopped after `batches` batches, or `stream` itself when `batches` is None.

    The wrapper forwards close to the source, so a caller that stops early
    still releases the grain workers the stream owns.
    """
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
    """Reads fixed windows of `seq_len + 1` ids off the token stream.

    Each window starts `seq_len` ids after the last, so record i's last token
    is record i + 1's first and the model sees every transition once. A batch
    is `{"text": int32 [batch, seq_len + 1]}`. `val_batches` bounds a
    validation pass; None scores the whole split.

    The training stream's saved position is a global window count, so a run
    resumes on any process count the global batch divides over.
    """

    path: str | None = None
    seq_len: int = 256
    val_batches: int | None = 4
    field: str | None = None
    """Which arrayrecord field or parquet column the ids are in, for a corpus
    held in one of those; a `.bin` corpus is the stream itself."""

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        from .sources.text import TokenWindowSource, token_corpus

        self.uncaptioned(tokenize)
        corpus, held_out = token_corpus(self.path, "TokenWindows", field=self.field)
        train = TokenWindowSource(corpus, self.seq_len)
        validation = TokenWindowSource(held_out, self.seq_len)
        return Dataset(
            train=train_stream(train, [], batch=local_batch(batch),
                               seed=self.seed, loading=self.loading),
            val=bounded(validation_pass(validation, [], batch=local_batch(batch), seed=self.seed, loading=self.loading), self.val_batches),
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

    Every window carries, beside each per-token field, `<field>_segment_ids`
    (which chunk of the window each token came from, counted from 1, and 0
    for the padding at the end) and `<field>_positions` (the token's place
    inside its chunk). A block-diagonal mask and per-document RoPE read that
    pair, which is what grain's packer writes per packed feature
    (`grain/_src/python/dataset/transformations/packing_packed_batch.py:116-117`).
    Chunks are cut the same way in every field, so one pair describes them
    all and the arrays are shared rather than copied per field.

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
        return {**fields,
                **{f"{key}_segment_ids": segment_ids for key in fields},
                **{f"{key}_positions": positions for key in fields}}


@datasets("packed_tokens")
@dataclasses.dataclass(frozen=True)
class PackedTokens(DatasetSpec):
    """Packs whole documents into `seq_len + 1` windows.

    Documents come from `TokenDocumentSource`, which cuts the token stream at
    the eos ids the tokenize tool writes between files (`--pack`). Each
    document, in chunks when it outgrows the window, is one element the
    packer adds to the first window with room. Every window carries
    `text_segment_ids` (which document each token is from, 0 for padding) and
    `text_positions` (the token's position inside its document), so the model
    can stop attention and the loss at document boundaries.

    `PackedWindows` plans the packing over the whole corpus in file order,
    ahead of the shard, so a window is a fact about the corpus rather than
    about one process's documents. The training stream shuffles and shards
    windows as it does any other record, and its saved position is a global
    window count that resumes on any process count. Which documents share a
    window is the same in every run over that corpus; the seed decides only
    the order the windows come in.

    `path` names one tokenized directory, or several with the share of a
    step each fills, MaxText's weighted `grain_train_files`. Each corpus is
    packed by its own plan, so a window holds one corpus's documents, and
    `mixture` interleaves the windows at their weights ahead of the shard:
    the weights are shares of the windows, and so of the tokens, a step
    reads, and a position is still one global window count. The corpora
    have to come from one tokenizer, which their `meta.json` records.

    `phases` switches what a run reads at step boundaries instead: each
    `DataPhase` names a corpus or mixture and the step it ends before, the
    last running on, and each phase's mixture continues every corpus's
    shuffled order where the earlier phases left it, so no window repeats
    before its corpus's epoch ends (`dew.data.providers.phased_dataset`). A resume checks the phases the
    run has read and accepts phases appended or moved past its step, so a run
    of one mixture continues into a phase list that begins with it. `path`
    is then unset, and validation reads the first phase's held-out split.

    `records` counts the windows a pass over the split holds, exactly, so
    `steps_per_epoch` is that pass; a mixture's pass is the windows in which
    every corpus has been read at least once (`mixed_records`), and a phased
    run's is its first phase's.
    `val_batches` bounds a validation pass; None scores the whole split, a
    mixture's split mixed at the same weights, each corpus in its own order.
    """

    path: str | Mapping[str, float] | None = None
    phases: Annotated[tuple[DataPhase, ...], json_list_argument(DataPhase)] = ()
    seq_len: int = 256
    val_batches: int | None = 4
    field: str | None = None
    """Which arrayrecord field or parquet column the ids are in, for a
    corpus held in one of those; a `.bin` corpus is the stream itself."""
    packing_bins: int = 8
    """Windows the plan keeps open at once. More of them leave less padding
    in a window and let documents further apart in the file share one."""

    @property
    def corpora(self) -> list[str]:
        """Every tokenized directory the run reads, in name order."""
        from .providers import name_ordered
        named = {name for phase in self.phases for name in name_ordered(phase.path)}
        return sorted(named | set(name_ordered(self.path)))

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        from .providers import corpora_dataset, name_ordered, phased_dataset
        from .sources.text import TokenDocumentSource, same_tokenizer, token_corpus

        self.uncaptioned(tokenize)
        if self.phases and self.path:
            raise ValueError("PackedTokens reads path= or phases=, not both")
        weighted = name_ordered(self.path)
        if not weighted and not self.phases:
            raise ValueError("PackedTokens needs path= set to the directory "
                             "tools/tokenize_text.py wrote, or several with weights")
        same_tokenizer(self.corpora)
        window = self.seq_len + 1

        # One source per split, and one plan over it. Finding the boundaries
        # reads the whole file, so rebuilding either per epoch would read a
        # multi-gigabyte train.bin again for a table the run already has.
        def packed(tokens) -> PackedWindows:
            source = TokenDocumentSource(tokens)
            return PackedWindows(pygrain.MapDataset.source(source), source.lengths, window,
                                 self.packing_bins, describe(source))

        splits: dict[str, tuple[PackedWindows, PackedWindows]] = {}

        def corpora(named: Mapping[str, float]) -> tuple[list[Corpus], list[Corpus]]:
            train, held = [], []
            for path, weight in named.items():
                if path not in splits:
                    corpus, held_out = token_corpus(path, "PackedTokens", field=self.field)
                    splits[path] = packed(corpus), packed(held_out)
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
