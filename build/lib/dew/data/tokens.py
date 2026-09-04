"""Token datasets over a directory that `tools/tokenize_text.py` wrote:
`train.bin`, `val.bin` and `meta.json`.

`TokenWindows` reads fixed `seq_len + 1` windows off the token stream;
`PackedTokens` packs whole documents into windows of that size and carries
the segment ids and positions the backbone's mask needs. Train shuffles from
`seed`, reshuffled per epoch, and runs forever; val reads `val.bin` once, in
file order, in whole batches, so every validation pass scores the same
windows. Both shard by JAX process.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Callable, Iterator

import grain.python as pygrain
import jax
import numpy as np

from dew.registry import datasets

from .dataset import (Batch, Dataset, DatasetSpec, Loading, local_batch, train_stream,
                      validation_pass)


def token_files(path: str | None, name: str) -> tuple[str, str]:
    """`(train.bin, val.bin)` of a tokenized directory, both required.

    Reading train.bin in val.bin's place scored the validation pass on the
    windows the model trains on.
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


def bounded(stream: Callable[[], Iterator[Batch]], batches: int | None) -> Callable[[], Iterator[Batch]]:
    """`stream`, ending after `batches` batches when that is set."""
    if batches is None:
        return stream
    return lambda: itertools.islice(stream(), batches)


@datasets("token_windows")
@dataclasses.dataclass(frozen=True)
class TokenWindows(DatasetSpec):
    """Fixed windows of `seq_len + 1` ids, each starting `seq_len` after the
    last, so record i's last token is record i + 1's first and the model sees
    every transition once. A batch is `{"text": int32 [batch, seq_len + 1]}`.
    `val_batches` bounds a validation pass; None scores all of val.bin."""

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
        knobs = dict(batch=local_batch(batch), seed=self.seed, loading=self.loading)
        return Dataset(
            train=train_stream(train, [], **knobs),
            val=bounded(validation_pass(val, [], **knobs), self.val_batches),
            records=len(train),
            batch=batch,
        )


def chunk_counts(lengths, chunk_len: int):
    """Chunks of at most `chunk_len` tokens each of `lengths` is cut into."""
    return -(-np.asarray(lengths, np.int64) // chunk_len)


class DocumentChunks(pygrain.MapDataset):
    """Documents cut into consecutive chunks of at most `chunk_len` tokens.

    Grain's packer refuses an element longer than the bin it packs into, so a
    document that outgrows the window is cut first; each chunk becomes its own
    segment in the packed row, which keeps attention inside the chunk and RoPE
    running from the chunk's own 0.

    The chunk table is built once from the document lengths, so a record costs
    one memmap slice rather than a walk over the documents before it.
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

    def __getitem__(self, index):
        # grain's conventions: a slice is the sharding and windowing API
        # (ds[shard::count]), and an index past the end wraps, which is what
        # makes `repeat` a length change rather than a copy.
        if isinstance(index, slice):
            return self.slice(index)
        index = index % len(self)
        text = self._parent[int(self._document[index])]["text"]
        start = int(self._offset[index])
        return {"text": text[start:start + self._chunk_len]}


@datasets("packed_tokens")
@dataclasses.dataclass(frozen=True)
class PackedTokens(DatasetSpec):
    """Whole documents packed into `seq_len + 1` windows.

    Documents come from `TokenDocumentSource`, which cuts the token stream at
    the eos ids the tokenize tool writes between files (`--pack`). Each
    document (in chunks, when it outgrows the window) is one element the
    packer adds to the first bin with room, and every emitted window carries
    `text_segment_ids` (which document each token is from, 0 for padding) and
    `text_positions` (the token's position inside its document), so the model
    can stop attention and the loss at document boundaries. This is grain's
    `Dataset` API rather than `DataLoader` for the reason grain gives for
    switching: packing. Documents are sliced per process before packing,
    since sharding after it would have every process pack the same ones.

    `records` counts window-sized chunks, the upper bound on the windows a
    pass over the split yields and the count a run has before it packs
    anything: every emitted window holds at least one chunk, the bound is
    tight once documents reach the window, and which chunks share a window
    depends on the shuffle. Counting documents instead reported zero steps
    for a corpus of fewer documents than a batch. `val_batches` bounds a
    validation pass; None scores all of val.bin.
    """

    path: str | None = None
    seq_len: int = 256
    val_batches: int | None = 4
    seed: int = 0
    loading: Loading = Loading()
    """The packer reads through grain's Dataset API, which takes no read
    options, so `workers` and `worker_buffer` are the two that reach it."""
    packing_bins: int = 8

    def load(self, *, batch: int) -> Dataset:
        from grain.experimental import FirstFitPackIterDataset

        from .sources.text import TokenDocumentSource

        train_bin, val_bin = token_files(self.path, "PackedTokens")
        per_process = local_batch(batch)
        window = self.seq_len + 1

        # One source per split, reused by its loader: finding the boundaries
        # reads the whole file, and a run rebuilding it per epoch would read a
        # multi-gigabyte train.bin again for a table it already has.
        train_source = TokenDocumentSource(train_bin)
        val_source = TokenDocumentSource(val_bin)

        def stream(source, shuffle, epochs):
            documents = DocumentChunks(
                pygrain.MapDataset.source(source), source.lengths, window)
            documents = documents[jax.process_index()::jax.process_count()]
            if shuffle:
                documents = documents.shuffle(self.seed)
            documents = documents.repeat(epochs)
            reads = documents.to_iter_dataset()
            if self.loading.workers:
                # The workers read documents, and the packer stays behind them
                # in this process: grain runs a whole pipeline per worker, so
                # packing inside them would fill bins from one worker's slice
                # of the documents and make the windows depend on worker_count.
                reads = reads.mp_prefetch(pygrain.MultiprocessingOptions(
                    num_workers=self.loading.workers,
                    per_worker_buffer_size=self.loading.worker_buffer))
            packed = FirstFitPackIterDataset(
                reads,
                length_struct={"text": window},
                num_packing_bins=self.packing_bins,
                seed=self.seed,
                # Bins come out in packing order for val, so a validation pass
                # is the same batches every time.
                shuffle_bins=shuffle,
                padding_struct={"text": 0},
            )
            return iter(packed.batch(per_process, drop_remainder=True))

        return Dataset(
            train=lambda: stream(train_source, True, None),
            val=bounded(lambda: stream(val_source, False, 1), self.val_batches),
            records=int(chunk_counts(train_source.lengths, window).sum()),
            batch=batch,
        )
