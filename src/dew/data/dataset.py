"""The value a run trains on, and the grain plumbing every dataset shares.

A `DatasetSpec` is a frozen dataclass behind `@datasets(name)` that says what
a dataset is and how it is read; `load(batch=)` turns it into a `Dataset`,
the value a recipe hands the trainer. Everything here is what the image,
video and token specs have in common: the per-process batch, the shuffled
training stream, the ordered validation pass and the slice that keeps the two
disjoint.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Mapping, Sequence

import grain.python as pygrain
import jax
from absl import flags

# grain's worker processes read absl flags; a script that never runs absl.app
# would crash on any worker_count > 0 with UnparsedFlagAccessError.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()

Batch = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Loading:
    """How fast records are read: grain's worker processes, the threads each
    of them reads with, and the two buffers held ahead of the step.

    None of the four changes which records a run sees or what is in them, so
    a host tuning them for its disk leaves the batches identical. The shuffle
    seed is not one of them for that reason: it decides the order records
    arrive in and keys the per-record rng that augments and captions them.
    """

    workers: int = 32
    threads: int = 64
    read_buffer: int = 128
    worker_buffer: int = 20


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Batches for a run.

    `train()` opens an endless shuffled stream; `val()` opens one pass over
    the held-out records in a fixed order that ends by itself, and is None
    when nothing is held out. `batch` is the global batch, `records` the
    training records behind it, so `steps_per_epoch` is one pass over them.

    Image and video fields are uint8 in [0, 255], text is the tokenized
    `{"input_ids", "attention_mask"}` dict under "text", and a token window
    is int32 ids under "text". Grain-backed iterators carry `get_state` and
    `set_state`, which is what a checkpoint records the run's position with;
    a stream without them cannot resume mid-epoch, and the trainer refuses
    to checkpoint one.
    """

    train: Callable[[], Iterator[Batch]]
    val: Callable[[], Iterator[Batch]] | None
    records: int | None
    batch: int

    @property
    def steps_per_epoch(self) -> int | None:
        return None if self.records is None else self.records // self.batch


class DatasetSpec(ABC):
    """What a dataset is and how it is read; a frozen dataclass per kind.

    A dataset that captions its records takes `tokenize` as well: the
    captions are the dataset's own product and which encoder reads them, at
    which context length, belongs to the run's condition, not to the source
    of the pictures.
    """

    @abstractmethod
    def load(self, *, batch: int) -> Dataset:
        """The dataset's batches, `batch` records a step across every process."""


CAPTION = "caption"
"""The batch field a captioning dataset writes its text in, before a run's
conditions read it."""


def tokenized(stream: Callable[[], Iterator[Batch]],
              tokenize: Callable[[Sequence[str]], Mapping[str, Any]] | None
              ) -> Callable[[], Iterator[Batch]]:
    """`stream` with each batch's captions replaced by what `tokenize` reads
    out of them.

    `tokenize` takes the batch's captions and returns the batch fields a
    run's conditions want, so an encoder's context length is the encoder's
    business and `--text.encoder char_table` and `--text.encoder clip_text`
    read the same dataset. It runs here, on the host, once per batch and
    outside the grain workers, so no encoder's weights are pickled into
    them.

    The captions never survive the stage: they are strings and a device
    takes numbers. None reads nothing out of them, which is what an
    unconditional run wants; a caller that wants the words keeps them with
    a reader that hands them back.
    """
    if tokenize is None:
        def tokenize(captions):
            return {}

    class Tokenizing:
        """The stream's iterator with the caption stage on its end."""

        def __init__(self, source):
            self.source = source

        def __iter__(self):
            return self

        def __next__(self) -> Batch:
            batch = dict(next(self.source))
            captions = [str(caption) for caption in batch.pop(CAPTION)]
            batch.update(tokenize(captions))
            return batch

        def __getattr__(self, name):
            # get_state/set_state belong to the grain iterator underneath,
            # and a stream without them stays a stream without them.
            if name in ("get_state", "set_state"):
                return getattr(self.source, name)
            raise AttributeError(name)

    return lambda: Tokenizing(iter(stream()))


def local_batch(batch: int) -> int:
    """The share of a global batch each JAX process batches for itself.

    Every process batches its own shard and the run reports `batch` as the
    global batch, so a remainder would train on fewer records a step than
    the run reports, and a batch below the process count would give every
    process a batch of nothing.
    """
    processes = jax.process_count()
    if batch % processes:
        raise ValueError(
            f"batch {batch} does not split over {processes} JAX processes")
    return batch // processes


class SourceSlice:
    """Random-access view over `source[start:stop]`.

    Gives the train and validation loaders disjoint index ranges while
    sharding and epoch handling stay grain's, the sampler's on the train side
    and the Dataset API's on the validation side. Plain attributes only:
    grain pickles the source to its workers.
    """

    def __init__(self, source: Any, start: int, stop: int):
        self.source = source
        self.start = start
        self.length = stop - start

    def __repr__(self) -> str:
        # grain writes repr(source) into a DataLoader iterator's checkpoint and
        # refuses to restore a state whose repr no longer matches, so a resumed
        # run needs a description that survives the process that wrote it. The
        # wrapped source is named by type: an arrayrecord source's own repr is
        # its address, and asking a hub source for its length would download
        # the table. Which half of the split this is, and how long, is what the
        # index stream depends on.
        return (f"SourceSlice({type(self.source).__name__}, "
                f"start={self.start}, length={self.length})")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source[self.start + index]


def hold_out(source: Any, records: int, held_out: int, name: str):
    """`(train_source, val_source)`: the first `held_out` of `records` records,
    in canonical order, as the validation split, and the rest as training.

    A held-out slice off the head keeps the two disjoint, so FID and CLIP are
    not measured on records the model trained on. `held_out` of zero holds
    nothing out and validates nothing.
    """
    if not held_out:
        return SourceSlice(source, 0, records), None
    if not held_out < records:
        raise ValueError(
            f"{name} holds out {held_out} validation records, which leaves "
            f"nothing of its {records} records to train on")
    return (SourceSlice(source, held_out, records),
            SourceSlice(source, 0, held_out))


def train_stream(source: Any, operations: Sequence[pygrain.Transformation], *,
                 batch: int, seed: int,
                 loading: Loading) -> Callable[[], Iterator[Batch]]:
    """An endless shuffled stream over `source`, batched per process.

    The sampler reshuffles every epoch from `seed` and shards by JAX process,
    so process p of n reads a disjoint 1/n of every epoch. `operations` run
    inside the workers, ahead of the batch.
    """
    sampler = pygrain.IndexSampler(
        num_records=len(source), shuffle=True, seed=seed, num_epochs=None,
        shard_options=pygrain.ShardByJaxProcess())

    def stream():
        return iter(pygrain.DataLoader(
            data_source=source, sampler=sampler,
            operations=[*operations, pygrain.Batch(batch, drop_remainder=True)],
            worker_count=loading.workers,
            read_options=pygrain.ReadOptions(loading.threads, loading.read_buffer),
            worker_buffer_size=loading.worker_buffer))

    return stream


def validation_pass(source: Any, transformations: Sequence[pygrain.Transformation], *,
                    batch: int, seed: int,
                    loading: Loading) -> Callable[[], Iterator[Batch]]:
    """One pass over `source` in record order, batched in this process.

    grain's DataLoader applies its operations inside the worker processes, so
    each worker had to fill a whole batch out of its own slice of the split.
    At the default eight workers a 512-record split gave batches of 64
    records read four times over, and with the sampler unbounded the pass
    never ended. Here the workers read and transform records and the batch is
    formed behind them, which is what the packed loader does with its packer,
    and it leaves the batches independent of worker_count.

    Sharding is grain's slice convention, so process p of n reads records
    p, p + n, ... of the split. The transforms are applied before that slice
    because grain keys a record's rng by its index in the dataset the random
    map sits on: applied after, record k was keyed by its position in the
    slice, and the same seed augmented and captioned it differently on one
    host than on a pod. A pass is whole batches only, because a part-full
    batch cannot be sharded over a device mesh.
    """
    def stream():
        records = pygrain.MapDataset.source(source).seed(seed).apply(list(transformations))
        reads = records[jax.process_index()::jax.process_count()].to_iter_dataset(
            pygrain.ReadOptions(loading.threads, loading.read_buffer))
        if loading.workers:
            reads = reads.mp_prefetch(pygrain.MultiprocessingOptions(
                num_workers=loading.workers,
                per_worker_buffer_size=loading.worker_buffer))
        return iter(reads.batch(batch, drop_remainder=True))

    return stream
