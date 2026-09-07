"""The value a run trains on, and the grain plumbing every dataset shares.

A `DatasetSpec` is a frozen dataclass behind `@datasets(name)` that says what
a dataset is and how it is read; `load(batch=)` turns it into a `Dataset`,
the value a recipe hands the trainer. Everything here is what the image,
video and token specs have in common: the per-process batch, the shuffled
training stream, the ordered validation pass and the slice that keeps the two
disjoint.

A training stream's position is one global record count rather than a shard
offset, so a run saved on one process count resumes on another;
`GlobalStream` here and `dew.position` are that contract.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import grain.python as pygrain
import jax
from absl import flags
from grain.experimental import ElasticIterator

from dew import position

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
    seed is not one of them; it decides the order records arrive in and keys
    the per-record rng that augments and captions them.

    `read_buffer` counts records held ahead of the read and `worker_buffer`
    counts batches held ahead of the step, per worker.
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

    Each factory call returns a fresh iterator owned by its caller. Close it
    after use when it exposes close; never close the shared dataset/backing
    store. A source's optional request_stop is a separate thread-safe signal,
    not permission to call final close concurrently with iteration.

    Image and video fields are uint8 in [0, 255], text is the tokenized
    `{"input_ids", "attention_mask"}` dict under "text", and a token window
    is int32 ids under "text".

    Whether a run can checkpoint its position depends on the iterator.
    Grain-backed iterators carry `get_state` and `set_state` and a
    fetch-as-you-go stream carries neither; `tokenized` forwards the pair.
    A run over a stream without them trains with `checkpoint_every=None` and
    is refused otherwise. A `train_stream` position is global and resumes on
    any process count; a packed stream's is its own shard's and resumes on
    the count that wrote it.
    """

    train: Callable[[], Iterator[Batch]]
    val: Callable[[], Iterator[Batch]] | None
    records: int | None
    batch: int

    @property
    def steps_per_epoch(self) -> int | None:
        return None if self.records is None else self.records // self.batch

    def epoch_steps(self, epochs: int = 1) -> int:
        """Steps in `epochs` passes over the records. A stream without a
        record count has no epoch, so a run over it gives its length in steps."""
        if self.steps_per_epoch is None:
            raise ValueError(
                "epochs need a dataset with a record count; this one streams "
                "without one, so give the run length in steps")
        return epochs * self.steps_per_epoch


class DatasetSpec(ABC):
    """What a dataset is and how it is read; a frozen dataclass per kind.

    A dataset that captions its records takes `tokenize` as well. The
    captions are the dataset's own product, and the run's condition decides
    which encoder reads them and at which context length.
    """

    @abstractmethod
    def load(self, *, batch: int) -> Dataset:
        """The dataset's batches, `batch` records a step across every process."""


CAPTION = "caption"
"""The batch field a captioning dataset writes its text in, before a run's
conditions read it."""


@runtime_checkable
class Checkpointable(Protocol):
    """A data stream that can say where it stopped and be put back there.

    Grain's iterators satisfy this, and so does `GlobalStream`. `Trainer.fit`
    refuses a run that asks for checkpoints over a stream without them. The
    state itself is opaque here; `dew.position` says which of its two kinds
    a checkpoint holds.
    """

    def get_state(self) -> Any: ...

    def set_state(self, state: Any) -> None: ...


def _no_captions(captions: Sequence[str]) -> Mapping[str, Any]:
    """Nothing out of a batch's captions; an unconditional run passes this as `tokenize`."""
    return {}


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

    The captions never survive the stage. They are strings and a device
    takes numbers. Pass None for an unconditional run, or a reader that hands
    the words back to keep them.
    """
    read = tokenize if tokenize is not None else _no_captions

    class Tokenizing:
        """The stream's iterator with each batch's captions tokenized."""

        def __init__(self, source: Iterator[Batch]):
            self.source: Iterator[Batch] | None = source

        def __iter__(self):
            return self

        def __next__(self) -> Batch:
            if self.source is None:
                raise StopIteration
            batch = dict(next(self.source))
            captions = [str(caption) for caption in batch.pop(CAPTION)]
            batch.update(read(captions))
            return batch

        def request_stop(self) -> None:
            """Forward only the source's thread-safe cancellation signal."""
            request_stop = getattr(self.source, "request_stop", None)
            if request_stop is not None:
                request_stop()

        def close(self) -> None:
            close = getattr(self.source, "close", None)
            try:
                if close is not None:
                    close()
            finally:
                # request_stop must still reach a source waiting inside close.
                self.source = None

    class CheckpointableTokenizing(Tokenizing):
        """The same stage over a stream that can report and restore its
        position, forwarding both. The methods are explicit because the
        protocol reads attributes statically, where a forwarding `__getattr__`
        would only satisfy `hasattr`."""

        def get_state(self) -> Any:
            source = self.source
            if not isinstance(source, Checkpointable):
                raise RuntimeError("the tokenized iterator is closed")
            return source.get_state()

        def set_state(self, state: Any) -> None:
            source = self.source
            if not isinstance(source, Checkpointable):
                raise RuntimeError("the tokenized iterator is closed")
            source.set_state(state)

    def start() -> Iterator[Batch]:
        source = iter(stream())
        if isinstance(source, Checkpointable):
            return CheckpointableTokenizing(source)
        return Tokenizing(source)

    return start


def local_batch(batch: int) -> int:
    """The share of a global batch each JAX process reads for itself.

    Every process batches its own shard and the run reports `batch` as the
    global batch, so a remainder would train on fewer records a step than
    the run reports, and a batch below the process count would leave some
    process with no records.
    """
    processes = jax.process_count()
    if batch % processes:
        raise ValueError(
            f"batch {batch} does not split over {processes} JAX processes")
    return batch // processes


class SourceSlice:
    """Random-access view over `source[start:stop]`.

    Gives the train and validation loaders disjoint index ranges while the
    shuffle, the epochs and the sharding stay grain's. Attributes stay plain
    because grain pickles the source to its workers.
    """

    def __init__(self, source: Any, start: int, stop: int):
        self.source = source
        self.start = start
        self.length = stop - start

    def __repr__(self) -> str:
        # A saved position names the order it counts into, and the order is
        # named by the source; a resumed run needs a description that survives
        # the process that wrote it. The wrapped source is named by type: an
        # arrayrecord source's own repr is its address, and asking a hub source
        # for its length would download the table. Which half of the split this
        # is, and how long, is what the record order depends on.
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


GLOBAL_INDEX = "global_next_index"
"""The field grain's elastic iterator keeps its global record offset in
(grain/_src/python/dataset/elastic_iterator.py:25,84-91). Its own name for
the key is private, the persisted shape is not, and reading it wrong fails
loudly in `tests/test_data.py`."""


def describe(source: object) -> str:
    """`source`'s own description, or its type when it has none.

    Grain compared repr(source) between a saved DataLoader state and the run
    restoring it, which is why dew's sources describe themselves without an
    address (`dew/data/sources/text.py`). A source that does not is named by
    its type: the default repr is this process's address for the object, and
    a resume would compare two addresses and refuse every time.
    """
    described = type(source).__repr__ is not object.__repr__
    return repr(source) if described else type(source).__name__


class GlobalStream:
    """A training stream whose saved position is one global record count.

    Grain's `ElasticIterator` owns the sharding and the batching together, so
    global batch k is records [k * batch, (k + 1) * batch) of one order and
    process p of n takes every nth of them starting at p
    (grain/_src/python/dataset/elastic_iterator.py:63-77). Its state is that
    offset in records: one number, the same on every process, naming no
    shard, so the position two processes wrote is where one process or four
    resume, on the same records in the same steps.

    The offset alone would resume that many records into whatever order the
    run now has, so `order` rides with it and a restore that disagrees is
    refused. `dew.position` is the shape both this and `dew.checkpoints`
    read.
    """

    def __init__(self, iterator: pygrain.DatasetIterator[Batch], order: str):
        self._iterator = iterator
        self._order = order

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        return next(self._iterator)

    def get_state(self) -> bytes:
        return position.encode(position.Global(
            records=int(self._iterator.get_state()[GLOBAL_INDEX]), order=self._order))

    def set_state(self, state: bytes) -> None:
        saved = position.decode(state)
        if saved is None:
            raise ValueError(
                "this training stream resumes from a global record count, and the "
                "saved position is one process's own offset into its shard; a run "
                "that wrote one cannot resume a stream that reads the records "
                "globally")
        if saved.order != self._order:
            raise ValueError(
                f"the saved data position is {saved.records} records into "
                f"{saved.order}, and this run reads {self._order}; a record count "
                f"is a place in one order and another place in another, so resume "
                f"the corpus, record count and seed the checkpoint was written with")
        self._iterator.set_state({GLOBAL_INDEX: saved.records})

    def close(self) -> None:
        self._iterator.close()


def train_stream(source: Any, operations: Sequence[pygrain.Transformation], *,
                 batch: int, seed: int,
                 loading: Loading) -> Callable[[], Iterator[Batch]]:
    """An endless shuffled stream over `source`, batched per process.

    `batch` is this process's share of a step, so the global batch behind it
    is that share times the process count. Grain's elastic iterator takes the
    global batch and cuts it up itself, which is what makes global batch k the
    same records at every process count: process p of n reads every nth record
    of the window the whole run is on. A `GlobalStream` position is therefore
    a record count and not a shard offset.

    The order is the corpus reshuffled from `seed` every epoch, endlessly.
    `operations` run behind that order and ahead of the per-process slice, so
    they run inside grain's workers, a record's rng is keyed by its place in
    the endless stream, and what a record becomes depends on neither the
    process count nor the worker count.

    The elastic iterator batches ahead of its read, so a read thread hands
    back a whole batch and `read_buffer` records held ahead is
    `read_buffer // batch` of them. Left in elements, the default would hold
    128 batches of images per worker instead of 128 records.
    """
    order = f"{describe(source)}, {len(source)} records reshuffled from seed {seed}"
    reads = pygrain.ReadOptions(loading.threads, max(1, loading.read_buffer // batch))
    workers = pygrain.MultiprocessingOptions(
        num_workers=loading.workers,
        per_worker_buffer_size=loading.worker_buffer) if loading.workers else None

    def stream() -> GlobalStream:
        records = pygrain.MapDataset.source(source).seed(seed)
        records = records.shuffle(seed).repeat(None).apply(list(operations))
        return GlobalStream(iter(ElasticIterator(
            records, batch * jax.process_count(), pygrain.ShardByJaxProcess(),
            read_options=reads,
            multiprocessing_options=workers)), order)

    return stream


def validation_pass(source: Any, transformations: Sequence[pygrain.Transformation], *,
                    batch: int, seed: int,
                    loading: Loading) -> Callable[[], Iterator[Batch]]:
    """One pass over `source` in record order, batched in this process.

    Grain's `DataLoader` would apply its operations inside the worker
    processes, where each worker fills a whole batch out of its own slice of
    the split: at the default eight workers a 512-record split becomes
    batches of 64 read four times over. Here the workers read and transform
    records and the batch is formed behind them, as the packed loader does
    with its packer, which leaves the batches independent of worker_count.

    Sharding is grain's slice convention, so process p of n reads records
    p, p + n, ... of the split. The transforms are applied before that slice.
    Grain keys a record's rng by its index in the dataset the random map sits
    on, so applied after the slice, record k takes its key from its position
    in the slice and the same seed augments and captions it differently on
    one host than on a pod. A pass is whole batches only, because a part-full
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
