"""The value a run trains on, and the grain plumbing every dataset shares.

A `DatasetSpec` is a frozen dataclass behind `@datasets(name)` that says what
a dataset is and how it is read; `load(batch=)` turns it into a `Dataset`,
the value a recipe hands the trainer. Everything here is what the image,
video and token specs have in common: the per-process batch, the shuffled
training stream, the ordered validation pass and the slice that keeps the two
disjoint.

A training stream's position is one global record count rather than a shard
offset, so a run saved on one process count resumes on another;
`GlobalStream` here and `dew.position` are that contract. A weighted
`mixture` of corpora and a `Ramp` of the batch are both cut out of that one
order, so neither adds anything to what a checkpoint holds.
"""

from __future__ import annotations

import bisect
import dataclasses
import math
import sys
from abc import ABC, abstractmethod
from typing import (Any, Callable, Iterator, Mapping, Protocol, Sequence, overload,
                    runtime_checkable)

import grain.python as pygrain
import jax
import numpy as np
from absl import flags

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

    Each counts something of its own: `workers` is processes, `threads` is
    the record reads one worker keeps in flight, `read_buffer` is the records
    one worker reads ahead, and `worker_buffer` is the batches one worker
    holds ready for the process that trains. Only the last counts batches,
    because a worker stacks the records it read and hands whole batches back.
    """

    workers: int = 32
    threads: int = 64
    read_buffer: int = 128
    worker_buffer: int = 2


@dataclasses.dataclass(frozen=True)
class Stage:
    """One step of a batch ramp: the global batch a step reads while the run
    is in this stage, and the record the stage starts at."""

    batch: int
    records: int


@dataclasses.dataclass(frozen=True)
class Ramp:
    """A global batch that grows over the run's first records.

    MaxText's batch ramp-up (`configs/base.yml:755-765`,
    `utils/rampup_batch.py:53-101`): a run starts at `start`, adds
    `increment` once the records read since the last increment reach
    `samples` divided by the number of increments, and stops at the batch
    the dataset was loaded with. That difference has to be a whole number of
    increments, MaxText's own requirement (`utils/rampup_batch.py:38-50`).

    MaxText counts a batch per device; dew counts the global batch
    everywhere, so `start` and `increment` are records a step, and
    `per_device_batch_size_start` times the device count is what `start`
    means here.

    The stage a run is in is a function of the records it has read, and that
    count is what a checkpoint already holds as the data position, so a
    resumed run continues the ramp with nothing else saved. A stage lasts
    `ceil(samples / increments / batch)` steps, computed as one integer
    ratio rather than MaxText's two floating-point divisions, so the
    boundaries are exact.
    """

    start: int
    increment: int
    samples: int

    def stages(self, final: int) -> tuple[Stage, ...]:
        """Every stage of the ramp up to `final`, the run's own global batch.

        Each stage's batch has to split over the processes, since a step is
        read in per-process shares; the mesh has its own divisor, which the
        trainer checks against the shardings before the run starts.
        """
        if min(self.start, self.increment, self.samples) < 1:
            raise ValueError(
                f"a batch ramp counts records: start={self.start}, "
                f"increment={self.increment} and samples={self.samples} are all "
                f"positive")
        if final <= self.start:
            raise ValueError(
                f"a batch ramp grows into the run's batch: it starts at "
                f"{self.start} and the dataset's batch is {final}, so there is "
                f"nothing to ramp")
        if (final - self.start) % self.increment:
            raise ValueError(
                f"a batch ramp reaches the run's batch in whole increments: "
                f"{final} - {self.start} is not a multiple of {self.increment}")
        increments = (final - self.start) // self.increment
        stages, batch, records = [], self.start, 0
        while batch < final:
            local_batch(batch)
            stages.append(Stage(batch=batch, records=records))
            records += -(-self.samples // (increments * batch)) * batch
            batch += self.increment
        local_batch(final)
        stages.append(Stage(batch=final, records=records))
        return tuple(stages)

    def steps_for(self, records: int, final: int) -> int:
        """The steps a run reads `records` records in, under this ramp.

        Fewer records a step early means more steps for the same records, so
        a pass over a corpus is longer than the flat batch would make it.
        """
        stages = self.stages(final)
        steps = 0
        for stage, next_stage in zip(stages, stages[1:]):
            if records < next_stage.records:
                return steps + (records - stage.records) // stage.batch
            steps += (next_stage.records - stage.records) // stage.batch
        return steps + max(records - stages[-1].records, 0) // final


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Batches for a run.

    `train()` opens an endless shuffled stream; `val()` opens one pass over
    the held-out records in a fixed order that ends by itself, and is None
    when nothing is held out. `batch` is the global batch, `records` the
    training records behind it, so `steps_per_epoch` is one pass over them.
    `ramp` is set by `ramped` when the run grows its batch over its first
    records, and `batch` is then the batch the ramp ends at.

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
    any process count, whether it reads one corpus or a weighted mixture of
    them; a stream that batches its own records reports whatever position it
    has, and `dew.checkpoints` refuses a process count that did not write
    one of those.
    """

    train: Callable[[], Iterator[Batch]]
    val: Callable[[], Iterator[Batch]] | None
    records: int | None
    batch: int
    ramp: Ramp | None = None

    @property
    def steps_per_epoch(self) -> int | None:
        if self.records is None:
            return None
        if self.ramp is None:
            return self.records // self.batch
        return self.ramp.steps_for(self.records, self.batch)

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
            if tokenize is not None:
                batch.update(tokenize(captions))
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


def describe(source: object) -> str:
    """`source`'s own description, or its type when it has none.

    A saved position names the order it counts into, and the order is named
    by the source, so a resumed run compares two descriptions and needs one
    that survives the process that wrote it (`dew/data/sources/text.py`
    writes such a repr). A source without its own is named by type: the
    default repr is this process's address for the object, and two addresses
    would refuse every resume.
    """
    described = type(source).__repr__ is not object.__repr__
    return repr(source) if described else type(source).__name__


@dataclasses.dataclass(frozen=True)
class Corpus:
    """One corpus of a weighted mixture: what it is called, what it reads and
    how much of a step it fills.

    `weight` is a share of the step and not a record count, so the mixture
    holds the proportions whatever the corpora's lengths are: a small corpus
    comes round again while a large one is still on its first pass.
    """

    name: str
    source: pygrain.RandomAccessDataSource[object]
    weight: float


def _shares(corpora: Sequence[Corpus]) -> tuple[float, ...]:
    """What each corpus of `corpora` fills of a step, as fractions of one.

    Normalised here, as MaxText normalises a mixture's weights before it
    hands them to grain (`input_pipeline/grain_data_processing.py:180-184`),
    so weights are read as ratios and 7/3 is the mixture 0.7/0.3 is.
    """
    if len(corpora) < 2:
        raise ValueError(
            f"a mixture reads two or more corpora, and this one names "
            f"{len(corpora)}; one corpus is that corpus")
    weights = [corpus.weight for corpus in corpora]
    if min(weights) <= 0:
        raise ValueError(
            f"every corpus of a mixture fills a positive share of a step, and "
            f"{ {corpus.name: corpus.weight for corpus in corpora} } does not; a "
            f"corpus a run reads none of is a corpus the run does not read")
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def mixture(corpora: Sequence[Corpus], seed: int | None) -> pygrain.MapDataset[object]:
    """`corpora` read together at their weights, as one order over records.

    Grain's own proportional interleave decides which corpus record k comes
    from: the one whose share of the first k + 1 records is short by one, so
    every prefix of the order (every global batch) holds each corpus's share
    to within one record, and nothing is drawn at random
    (`grain/_src/python/dataset/transformations/mix.py:314-347`). Weights
    reach grain as ratios and it scales them to integers against the
    smallest, so a share is exact to a hundredth of the smallest one
    (`mix.py:305-311`).

    With `seed`, every corpus is reshuffled per epoch and repeated before
    the mixing, which is grain's own instruction for keeping a mixture's
    proportions under a shuffle and what MaxText's pipeline does
    (`input_pipeline/grain_data_processing.py:110-112,169-184`): the mixture
    is then endless and each corpus cycles its own records at its own rate.
    Without a seed the corpora are read in their own order and the mixture
    stops before any of them would come round again, grain's own length rule
    (`mix.py:52-68`), which is the pass a validation split wants.

    Mixing here, ahead of the shard, is what keeps a mixture's position one
    global record count: which corpus record k is from, and where it sits in
    that corpus, are functions of k. The shares hold over the global batch.
    One process's rows of it are every nth record of the interleave, so two
    corpora at equal weights, which alternate, put only the first on one of
    two processes and only the second on the other; the step's gradient is
    the same sum either way. MaxText mixes iterators after the
    shard and keeps a state per corpus per host, which lets it change the
    mixture on resume (`grain_data_processing.py:208-212`); dew refuses a
    changed order instead, as it already does for a changed corpus.
    """
    shares = _shares(corpora)

    def order(corpus: Corpus) -> pygrain.MapDataset[object]:
        records = pygrain.MapDataset.source(corpus.source)
        return records if seed is None else records.shuffle(seed).repeat(None)

    return pygrain.MapDataset.mix([order(corpus) for corpus in corpora], list(shares))


def mixed_records(corpora: Sequence[Corpus]) -> int:
    """One pass over a mixture: the records in which every corpus of it has
    been read at least once.

    A mixture has no pass of its own, since a corpus that fills a tenth of a
    step comes round ten times while one that fills the rest is read once, so
    the pass is the longest of the corpora's own. That is `len(source)` for
    one corpus and the records of both for two equal corpora at equal
    weights, which is what `steps_per_epoch` counts everywhere else.
    """
    return max(math.ceil(len(corpus.source) / share)
               for corpus, share in zip(corpora, _shares(corpora)))


class _WorkerBatches(pygrain.MapDataset[object]):
    """`parent` re-indexed so that grain's per-worker stride slice (index i
    goes to worker i % W) hands each worker whole, contiguous batches:
    index `q*W*B + p*W + w` is record `p` of batch `b = q*W + w`. Worker w
    then batches inside its own process and the round-robin interleave in the
    training process restores batch order 0, 1, 2, ... The length is padded
    to whole rounds of one batch per worker; an index past the parent's last
    whole batch is None, which grain's reader skips.
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(self, parent: pygrain.MapDataset[object], batch: int, workers: int):
        super().__init__(parent)
        self._batch, self._workers = batch, workers
        self._whole = len(parent) // batch
        self._length = min(math.ceil(self._whole / workers) * workers * batch, sys.maxsize)

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset[object]: ...
    @overload
    def __getitem__(self, index: int) -> object | None: ...

    def __getitem__(self, index: int | slice) -> object | None:
        if isinstance(index, slice):
            return self.slice(index)
        worker, within = index % self._workers, index // self._workers
        round_, row = divmod(within, self._batch)
        which = round_ * self._workers + worker
        return None if which >= self._whole else self._parent[which * self._batch + row]


def _batches(records: pygrain.MapDataset[object], *, batch: int, loading: Loading,
             offset: int = 0) -> pygrain.DatasetIterator[Batch]:
    """This process's share of `records`, in batches of `batch` records.

    The slice is `offset + process_index :: process_count`, so global batch k
    is the same records at every process count, and an offset is a slice
    bound rather than a replay.

    Reads are records: the threads behind `to_iter_dataset` each fetch one,
    so no read waits on a whole batch (grain's `ElasticIterator` batches
    ahead of its read and is an order of magnitude slower on a slow source).
    The batch is stacked inside the worker so each transfer to this process
    is a whole batch; `_WorkerBatches` permutes the slice so a worker's share
    is whole batches, and the permutation depends only on the batch and the
    worker count, so neither count changes which records a batch holds.
    """
    mine = records[offset + jax.process_index()::jax.process_count()]
    if loading.workers:
        mine = _WorkerBatches(mine, batch, loading.workers)
    stream = mine.to_iter_dataset(pygrain.ReadOptions(loading.threads, loading.read_buffer))
    stream = stream.batch(batch, drop_remainder=True)
    if loading.workers:
        stream = stream.mp_prefetch(pygrain.MultiprocessingOptions(
            num_workers=loading.workers,
            per_worker_buffer_size=loading.worker_buffer))
    return iter(stream)


def rows_of(batch: Mapping[str, Any]) -> int:
    """The records `batch` holds, read off its first field that has rows.

    A batch may carry a field that is one value for the whole step rather
    than one per record, and that field says nothing about how many records
    there are. Any mapping of fields answers, so the trainer's own batch
    type reaches this as it is.
    """
    for leaf in jax.tree.leaves(batch):
        shape = np.shape(leaf)
        if shape:
            return shape[0]
    raise ValueError("a batch of scalars holds no records")


class GlobalStream:
    """A training stream whose saved position is one global record count.

    The stream is a shuffled order over the whole corpus, endlessly, and the
    step is cut out of it here: global batch k is records
    [k * batch, (k + 1) * batch) of that order, and process p of n reads
    every nth of them starting at p. The position is then how many records
    the run has consumed -- one number, the same on every process, naming no
    shard -- so the position two processes wrote is where one process or four
    resume, on the same records in the same steps.

    `open(offset)` starts the per-process read at a record offset, which is
    what a restore does instead of replaying: an offset is a slice bound, so
    a resume reads nothing it has already read. It is called on the first
    batch and again after `set_state`, so a stream that is restored before it
    is read starts no worker twice.

    The offset alone would resume that many records into whatever order the
    run now has, so `order` rides with it and a restore that disagrees is
    refused. `dew.position` is the shape both this and `dew.checkpoints`
    read.
    """

    def __init__(self, open: Callable[[int], pygrain.DatasetIterator[Batch]],
                 batch: int, order: str):
        self._open = open
        self._batch = batch
        self._order = order
        self._records = 0
        self._reads: pygrain.DatasetIterator[Batch] | None = None

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        if self._reads is None:
            self._reads = self._open(self._records)
        batch = next(self._reads)
        # Counted after the batch: a step the stream did not deliver is not
        # a step a resume may skip.
        self._records += self._batch
        return batch

    def get_state(self) -> bytes:
        return position.encode(position.Global(records=self._records, order=self._order))

    def set_state(self, state: bytes) -> None:
        saved = position.read(state)
        if saved.order != self._order:
            raise ValueError(
                f"the saved data position is {saved.records} records into "
                f"{saved.order}, and this run reads {self._order}; a record count "
                f"is a place in one order and another place in another, so resume "
                f"the corpus, record count and seed the checkpoint was written with")
        self.close()
        self._records = saved.records

    def close(self) -> None:
        reads, self._reads = self._reads, None
        if reads is not None:
            reads.close()


@runtime_checkable
class Resumable(Checkpointable, Protocol):
    """A stream that is read and can be put back where it stopped, which is
    what a batch ramp wraps."""

    def __next__(self) -> Batch: ...


class RampedStream:
    """A training stream whose step grows over the run's first records.

    Wraps a stream whose position is a global record count, a `GlobalStream`
    or `tokenized` over one, and decides only how many of its records a step
    reads. MaxText reads a whole final-size batch and cuts the current one
    out of a rolling buffer (`common/data_loader.py:122-176`), so the records
    a step reads are the records the run would have read without the ramp,
    in the same order, and a stage change neither skips a record nor reads
    one twice. What is left in the buffer has not been trained on and is not
    in the position, which is the source's with its count replaced by the
    records handed out; a restore hands the source that count, so it reopens
    its read there.
    """

    def __init__(self, source: Resumable, stages: Sequence[Stage]):
        self._source = source
        self._stages = tuple(stages)
        self._starts = tuple(stage.records for stage in stages)
        self._processes = jax.process_count()
        self._records = 0
        self._held: Batch | None = None

    def __iter__(self) -> Iterator[Batch]:
        return self

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The batch this stream reads at each stage, and where each begins.

        The trainer reads them off the stream it opened: a compiled step per
        stage is the whole set of shapes a ramped run runs, and the mesh has
        to divide every one of them.
        """
        return self._stages

    def _stage(self) -> Stage:
        """The stage the records read so far leave the run in."""
        return self._stages[bisect.bisect_right(self._starts, self._records) - 1]

    def __next__(self) -> Batch:
        stage = self._stage()
        rows = stage.batch // self._processes
        while self._held is None or rows_of(self._held) < rows:
            read = next(self._source)
            self._held = read if self._held is None else jax.tree.map(
                lambda kept, new: np.concatenate([kept, new]), self._held, read)
        step = jax.tree.map(lambda field: field[:rows], self._held)
        self._held = jax.tree.map(lambda field: field[rows:], self._held)
        self._records += stage.batch
        return step

    def get_state(self) -> bytes:
        place = position.read(self._source.get_state())
        return position.encode(dataclasses.replace(place, records=self._records))

    def set_state(self, state: bytes) -> None:
        self._source.set_state(state)  # Refuses a shard offset and another order.
        self._held = None
        self._records = position.read(state).records
        stage = self._stage()
        if (self._records - stage.records) % stage.batch:
            raise ValueError(
                f"the saved data position is {self._records} records in, and this "
                f"ramp reads {stage.batch} records a step from record "
                f"{stage.records} on, so no step of it ends there: the checkpoint "
                f"was written by a run that ramped differently")

    def request_stop(self) -> None:
        """Forward only the source's thread-safe cancellation signal."""
        request_stop = getattr(self._source, "request_stop", None)
        if request_stop is not None:
            request_stop()

    def close(self) -> None:
        self._held = None
        close = getattr(self._source, "close", None)
        if close is not None:
            close()


def ramped(data: Dataset, ramp: Ramp) -> Dataset:
    """`data` with the training batch growing to `data.batch` over `ramp`.

    A validation pass keeps the whole batch: a score over a growing number
    of records is a score of a different thing each time.

    Only a stream whose position is a global record count can ramp, because
    the ramp cuts that order into other steps and the count it saves has to
    be the records it handed over. A stream that batches its own records
    reports a shard offset and is refused, by name.
    """
    stages = ramp.stages(data.batch)  # An impossible schedule fails here, not mid-run.

    def train() -> Iterator[Batch]:
        stream = data.train()
        if isinstance(stream, Resumable):
            state = stream.get_state()
            if isinstance(state, bytes) and position.translates(state):
                return RampedStream(stream, stages)
        close = getattr(stream, "close", None)
        if close is not None:
            close()
        raise TypeError(
            f"a batch ramp cuts the step out of a global record order, and "
            f"{type(stream).__name__} hands over batches it has cut itself; ramp "
            f"a dataset read through train_stream or mixed_stream")

    return dataclasses.replace(data, train=train, ramp=ramp)


def train_stream(source: pygrain.RandomAccessDataSource[object], operations: Sequence[pygrain.Transformation], *,
                 batch: int, seed: int,
                 loading: Loading) -> Callable[[], Iterator[Batch]]:
    """An endless shuffled stream over `source`, batched per process.

    `batch` is this process's share of a step, so the global batch behind it
    is that share times the process count. The order is the corpus reshuffled
    from `seed` every epoch, endlessly, and `_batches` cuts this process's
    share off it, so global batch k is the same records at every process
    count and a `GlobalStream` position is a record count rather than a shard
    offset.

    `operations` run behind the order and ahead of the slice, so they run
    inside the workers, a record's rng is keyed by its place in the endless
    stream, and what a record becomes depends on neither count either.
    """
    order = f"{describe(source)}, {len(source)} records reshuffled from seed {seed}"

    def open(offset: int) -> pygrain.DatasetIterator[Batch]:
        records = pygrain.MapDataset.source(source).seed(seed)
        records = records.shuffle(seed).repeat(None).apply(list(operations))
        return _batches(records, batch=batch, loading=loading, offset=offset)

    def stream() -> GlobalStream:
        return GlobalStream(open, batch * jax.process_count(), order)

    return stream


def mixed_stream(corpora: Sequence[Corpus], operations: Sequence[pygrain.Transformation], *,
                 batch: int, seed: int,
                 loading: Loading) -> Callable[[], Iterator[Batch]]:
    """An endless stream over `corpora` at their weights, batched per process.

    The order is `mixture(corpora, seed)` and everything after it is
    `train_stream`'s: the same per-process slice, the same batch behind the
    reads, and the same position, which is why a mixture's place in its
    corpora is one record count and resumes on any process count. The
    `operations` sit above the mixture, so a record's rng is keyed by its
    place in the mixed stream rather than in the corpus it came from.
    """
    shares = _shares(corpora)
    order = "mixture reshuffled from seed {} of [{}]".format(seed, ", ".join(
        f"{corpus.name} at {share:.6g}: {describe(corpus.source)}, "
        f"{len(corpus.source)} records"
        for corpus, share in zip(corpora, shares)))

    def open(offset: int) -> pygrain.DatasetIterator[Batch]:
        records = mixture(corpora, seed).seed(seed).apply(list(operations))
        return _batches(records, batch=batch, loading=loading, offset=offset)

    def stream() -> GlobalStream:
        return GlobalStream(open, batch * jax.process_count(), order)

    return stream


def validation_pass(source: pygrain.RandomAccessDataSource[object], transformations: Sequence[pygrain.Transformation], *,
                    batch: int, seed: int,
                    loading: Loading) -> Callable[[], Iterator[Batch]]:
    """One pass over `source` in record order, in batches of `batch`.

    Grain's DataLoader gives each worker its own slice of the split and lets
    it fill a whole batch out of that, so which records a batch holds moves
    with worker_count. `_batches` hands a worker the records of one batch
    instead, so the split is cut into the same batches at every count.

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
        return _batches(records, batch=batch, loading=loading)

    return stream
