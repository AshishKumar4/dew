"""The value a run trains on, and the grain plumbing every dataset shares.

A `DatasetSpec` is a frozen dataclass behind `@datasets(name)` that says what
a dataset is and how it is read. `load(batch=)` turns it into a `Dataset`,
the value a recipe hands the trainer. Everything here is what the image,
video and token specs have in common: the share of a batch a reader reads,
the shuffled training stream, the ordered validation pass, and the slice
that keeps the two disjoint.

Every stream is opened for a `DataPartition`, the share of each global batch
its reader reads. The trainer asks the mesh for it
(`dew.training.distributed.data_partition`), since the processes a pipeline
or a split sequence spans between them hold the same rows and read the same
share; a loader reads the share it is handed and nothing else.

A training stream's position is one global record count rather than a shard
offset, so a run saved on one process count resumes on another. `GlobalStream`
here and `dew.position` are that contract. A weighted `mixture` of corpora
and a `Ramp` of the batch are cut out of that one order, so neither adds
anything to what a checkpoint holds.
"""

from __future__ import annotations

import bisect
import dataclasses
import functools
import itertools
import json
import math
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterator, Mapping, Protocol, Sequence, overload, runtime_checkable

import grain.python as pygrain
import jax
import numpy as np
import tyro
from absl import flags

from dew import position

# `Batch` lives in dew.objectives.base. The data layer imports it from here
# so a dataset module needs one import for the value and its shape.
from dew.objectives.base import Batch

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


# grain's worker processes read absl flags; a script that never runs absl.app
# would crash on any worker_count > 0 with UnparsedFlagAccessError.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()


type Tokenize = Callable[[Sequence[str]], Batch]
"""A run's caption reader: the batch's captions in, the batch fields its
conditions want out. The dataset carries the text, the encoder behind this
decides what tokens it becomes."""

@dataclasses.dataclass(frozen=True)
class DataPartition:
    """Which share of every global batch a reader reads: the `index`th of
    `count` equal, disjoint shares.

    A loader cuts its record order `index :: count`, so the shares of global
    batch k together hold the same records at every count, and a record
    count is a place in the stream whatever the count. The trainer asks the
    mesh which share a process reads (`dew.training.distributed.
    data_partition`): processes whose devices hold the same rows read the
    same share, since the axes between them split a sequence or hold a
    pipeline's stages rather than rows. `DataPartition()` is one reader of
    every row, what a single process reads.
    """

    index: int = 0
    count: int = 1
    readers: int = 1
    """The processes that read this share. Each reads the same records, so a
    source whose rows are not the same on every read of one share, as a
    fetch over the network that drops what failed is not, refuses more than
    one, or reads on the first reader alone and hands its batch to the
    others (`dew.training.distributed.first_reader_batch`)."""
    reader: int = 0
    """Which of the share's readers this process is, in process order: 0
    for the first."""

    def __post_init__(self):
        if not 0 <= self.index < self.count or not 0 <= self.reader < self.readers:
            raise ValueError(
                f"a data partition is share index of count, 0 <= index < count, read "
                f"by one or more readers, of which this is reader 0 <= reader < readers; "
                f"got index {self.index} of {self.count} read by {self.readers}, reader "
                f"{self.reader}")

    def rows(self, batch: int) -> int:
        """The rows of a `batch`-row global batch one share holds.

        A remainder would train on fewer records a step than the run reports,
        and a batch below the share count would leave a share with nothing.
        """
        if batch % self.count:
            raise ValueError(
                f"batch {batch} does not split into {self.count} equal shares, "
                f"one for each group of processes that reads its own rows")
        return batch // self.count


type Reader = Callable[[DataPartition], Iterator[Batch]]
"""Opens a fresh iterator over one share of every global batch, the share the
partition names. The iterator is its caller's to close."""

type GrainPipeline = pygrain.MapDataset[Batch] | Callable[[DataPartition], pygrain.IterDataset[Batch]]
"""A grain pipeline a caller built, for `Dataset.from_grain`: a `MapDataset`,
read by index, which a share is cut from; or a function of the partition that
builds the `IterDataset` one share reads, since a pipeline read as it comes is
sharded by whoever builds it. Which of the two it is decides what a saved
position can be. Elements are one example's fields, the shape `Batch` names,
since grain stacks them into a batch of those fields."""

@runtime_checkable
class Records(Protocol):
    """Answers records by index, which is how the loaders read a source.

    A record is one example's fields, the shape `Batch` names, or the packed
    bytes an arrayrecord holds, which the spec's own transform unpacks before
    the fields exist. A grain dataset answers None where its padding covers
    an index, and grain's reader skips those rather than batching them.

    Grain's own `RandomAccessDataSource` says the same two methods with the
    record as a type parameter. That parameter is invariant, so a source
    declared through it cannot be handed to a loader that named a different
    record. This one names what the loaders actually read.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Batch | bytes | None: ...


type Indexed = Records | Sequence[Batch]
"""Records read by index: a source that answers grain's two methods, or a
plain sequence of them. A spec that lists its records in memory hands over
the sequence, the way the video specs list their clips."""


def json_argument[Options: DataclassInstance](
        options: type[Options]) -> tyro.constructors.PrimitiveConstructorSpec[Options]:
    """`options` written as one JSON object on the command line.

    A spec holds a provider's options as one frozen value, and some of that
    value's fields are the library's own objects: a `datasets.Features`, a
    tfds decoder tree. Their types are imported on use, so a flag per field
    would need annotations this process has not resolved and has no spelling
    for the objects anyway. The whole value is one argument instead, the way
    `--model.config` is one JSON object.
    """
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda given: options(**json.loads(given[0])),
        is_instance=lambda given: isinstance(given, options),
        # Only for the help text, where a library object is best shown as
        # itself; nothing reads this back.
        str_from_instance=lambda given: [json.dumps(
            {field.name: getattr(given, field.name)
             for field in dataclasses.fields(given)}, default=repr)],
    )


def json_list_argument[Entry: DataclassInstance](
        entry: type[Entry]) -> tyro.constructors.PrimitiveConstructorSpec[tuple[Entry, ...]]:
    """A tuple of `entry` records written as one JSON list on the command
    line, `[{"field": ...}, ...]`, since a flag per field cannot spell a list
    of records whose length the command line decides."""
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda given: tuple(entry(**record) for record in json.loads(given[0])),
        is_instance=lambda given: isinstance(given, tuple) and all(
            isinstance(value, entry) for value in given),
        str_from_instance=lambda given: [json.dumps([dataclasses.asdict(value) for value in given])],
    )


@dataclasses.dataclass(frozen=True)
class DataPhase:
    """One phase of a run's data: what it reads and the step it ends at.

    `path` names one corpus or a weighted mixture the way the spec's own
    `path` does. `until_step` is the step the phase ends before, counted from
    the run's start in steps of the full batch; None for the last phase,
    which runs to the end. `PhasedStream` says how a resume treats a changed
    list.
    """

    path: str | Mapping[str, float]
    until_step: int | None = None


@runtime_checkable
class Closeable(Protocol):
    """A stream that holds something a stopped run has to give back: worker
    processes, file handles, a shared memory block."""

    def close(self) -> None: ...


@runtime_checkable
class Stoppable(Protocol):
    """A stream that can be asked to stop before it is drained."""

    def request_stop(self) -> None: ...


@runtime_checkable
class Budgeted(Protocol):
    """Says how long stopping this stream may take.

    Separate from `Stoppable` because a stream can report a budget without
    taking a stop request. The wrapper forwards each on its own.
    """

    @property
    def stop_seconds(self) -> float | None: ...


class Forwarding:
    """Forwards a stream wrapper's stop signal, stop budget and close to its
    source. Subclasses keep the source at `_source` and override `close` for
    their own cleanup around `super().close()`."""

    def _forwarded(self) -> Iterator[Batch] | None:
        """The wrapped source, or None before a subclass sets one.

        This lookup is the boundary between the wrapper and whatever it
        wraps. A subclass declares `_source` with its own stream type,
        narrower than anything this base could state, and a wrapper may be
        constructed before it has a source. The three hooks below ask the
        Stoppable and Closeable protocols of what it returns.
        """
        return getattr(self, "_source", None)

    def request_stop(self) -> None:
        source = self._forwarded()
        if isinstance(source, Stoppable):
            source.request_stop()

    @property
    def stop_seconds(self) -> float | None:
        source = self._forwarded()
        seconds = source.stop_seconds if isinstance(source, Budgeted) else None
        return None if seconds is None else float(seconds)

    def close(self) -> None:
        source = self._forwarded()
        if isinstance(source, Closeable):
            source.close()


@dataclasses.dataclass(frozen=True)
class Loading:
    """Says how fast records are read, in grain's four throughput knobs.

    None of the four changes which records a run sees or what is in them. A
    host tuning them for its disk leaves the batches identical. The shuffle
    seed is not one of them; it decides the order records arrive in and keys
    the per-record rng that augments and captions them.

    Each counts something of its own. `workers` is processes, `threads` is
    the record reads one worker keeps in flight, and `read_buffer` is the
    records one worker reads ahead. `worker_buffer` alone counts batches, the
    batches one worker holds ready for the process that trains, because a
    worker stacks the records it read and hands whole batches back.
    """

    workers: int = 32
    threads: int = 64
    read_buffer: int = 128
    worker_buffer: int = 2

    @property
    def stop_seconds(self) -> float:
        """How long a stop of this many workers is allowed to take.

        Grain joins its worker processes one at a time, ~0.5 s each idle and
        up to a batch's work each busy.
        """
        return 2.0 + 1.0 * self.workers


@dataclasses.dataclass(frozen=True)
class Stage:
    """Holds one step of a batch ramp: the global batch a step reads while
    the run is in this stage, and the record the stage starts at."""

    batch: int
    records: int


@dataclasses.dataclass(frozen=True)
class Ramp:
    """Grows the global batch over the run's first records.

    A run starts at `start` and adds `increment` once the records read since
    the last increment reach `samples` divided by the number of increments.
    It stops at the batch the dataset was loaded with, and that difference
    has to be a whole number of increments (MaxText's `configs/base.yml:755-765`,
    `utils/rampup_batch.py:38-50,53-101`).

    MaxText counts a batch per device and dew counts the global batch
    everywhere, so `start` and `increment` are records a step:
    `per_device_batch_size_start` times the device count is `start` here.

    The stage a run is in is a function of the records it has read, which a
    checkpoint already holds as the data position, so a resumed run continues
    the ramp with nothing else saved. A stage lasts
    `ceil(samples / increments / batch)` steps, computed as one integer ratio
    rather than MaxText's two floating-point divisions, so the boundaries are
    exact.
    """

    start: int
    increment: int
    samples: int

    def stages(self, final: int) -> tuple[Stage, ...]:
        """Every stage of the ramp up to `final`, the run's own global batch.

        Each stage's batch has to split into the mesh's rows, and so into the
        shares its readers read; the trainer checks every stage against the
        mesh before the run starts.
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
            stages.append(Stage(batch=batch, records=records))
            records += -(-self.samples // (increments * batch)) * batch
            batch += self.increment
        stages.append(Stage(batch=final, records=records))
        return tuple(stages)

    def steps_for(self, records: int, final: int) -> int:
        """The steps a run reads `records` records in, under this ramp.

        Fewer records a step early means more steps for the same records, so
        a pass over a corpus is longer than the flat batch would make it.
        """
        stages = self.stages(final)
        steps = 0
        for stage, next_stage in itertools.pairwise(stages):
            if records < next_stage.records:
                return steps + (records - stage.records) // stage.batch
            steps += (next_stage.records - stage.records) // stage.batch
        return steps + max(records - stages[-1].records, 0) // final


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Opens the batches a run trains and validates on.

    `train(partition)` opens an endless shuffled stream. `val(partition)`
    opens one pass over the held-out records in a fixed order that ends by
    itself, and is None when nothing is held out. Either reads the share of
    each global batch `partition` names (`DataPartition`). `batch` is the
    global batch and `records` the training records behind it, so
    `steps_per_epoch` is one pass over them. `ramped` sets `ramp` when the
    run grows its batch over its first records, and `batch` is then the batch
    the ramp ends at.

    Each factory call returns a fresh iterator owned by its caller. Close it
    after use when it exposes close; never close the shared dataset or
    backing store. A source's optional request_stop is a separate thread-safe
    signal, not permission to call final close concurrently with iteration.

    Image and video fields are uint8 in [0, 255], text is the tokenized
    `{"input_ids", "attention_mask"}` dict under "text", and a token window
    is int32 ids under "text".

    Whether a run can checkpoint its position depends on the iterator.
    Grain-backed iterators carry `get_state` and `set_state`, a
    fetch-as-you-go stream carries neither, and `tokenized` forwards the
    pair. A run over a stream without them trains with
    `checkpoint_every=None` and is refused otherwise. A `train_stream`
    position is global and resumes on any partition, over one corpus or a
    weighted mixture. A stream that batches its own records reports whatever
    position its share has, and `dew.checkpoints` resumes it only on a
    reader of that share.
    """

    train: Reader
    val: Reader | None
    records: int | None
    batch: int
    ramp: Ramp | None = None

    @classmethod
    def from_grain(cls, train: GrainPipeline, *, batch: int,
                   validation: GrainPipeline | None = None,
                   records: int | None = None,
                   loading: Loading = Loading()) -> Dataset:
        """Builds a run over grain pipelines a caller built themselves.

        The order, the shuffle and what a record becomes are the caller's.
        This adds what every spec's `load` adds, through the same helpers:
        the reader's share of the batch, whole batches only, and the state
        pair a checkpoint saves.

        A `MapDataset` is read by index, so it gets the training stream every
        spec gets: endlessly repeated, cut into the reader's share, and saved
        as one global record count. A pipeline read as it comes is sharded by
        whoever builds it, so it arrives as a function of the partition that
        builds the `IterDataset` of that share. It is batched where it is and
        reports grain's own iterator state, which `dew.checkpoints` restores
        only into a reader of the same share.

        `records` is the records of one pass, which `steps_per_epoch`
        divides. It defaults to a MapDataset's own length, so a caller who
        repeated their dataset before handing it over gives the length of one
        pass instead. A grain pipeline has no description of its own, so the
        saved position names the pipeline's type and length rather than the
        corpus under it. Swapping the corpus under one pipeline is the
        caller's to keep straight.
        """
        mapped = train if isinstance(train, pygrain.MapDataset) else None
        if mapped is not None:
            endless = mapped.repeat(None)
            order = f"{describe(mapped)}, {len(mapped)} records"

            def training(partition: DataPartition) -> Iterator[Batch]:
                rows = partition.rows(batch)
                return GlobalStream(
                    lambda offset: _shared(endless, rows=rows, partition=partition,
                                           loading=loading, offset=offset),
                    batch, order, loading.stop_seconds)
        else:
            def training(partition: DataPartition) -> Iterator[Batch]:
                return _shared(train, rows=partition.rows(batch), partition=partition,
                               loading=loading)

        def validating(partition: DataPartition) -> Iterator[Batch]:
            assert validation is not None
            return _shared(validation, rows=partition.rows(batch), partition=partition,
                           loading=loading)

        return cls(
            train=training,
            val=None if validation is None else validating,
            records=len(mapped) if records is None and mapped is not None else records,
            batch=batch,
        )

    @property
    def steps_per_epoch(self) -> int | None:
        if self.records is None:
            return None
        if self.ramp is None:
            return self.records // self.batch
        return self.ramp.steps_for(self.records, self.batch)

    def epoch_steps(self, epochs: int = 1) -> int:
        """The steps `epochs` passes over the records take.

        A stream without a record count has no epoch, so a run over it gives
        its length in steps instead.
        """
        if self.steps_per_epoch is None:
            raise ValueError(
                "epochs need a dataset with a record count; this one streams "
                "without one, so give the run length in steps")
        return epochs * self.steps_per_epoch


@dataclasses.dataclass(frozen=True)
class DatasetSpec(ABC):
    """Says what a dataset is and how it is read, one frozen dataclass per kind.

    `seed` and `loading` belong to every kind, so they are declared once
    here. The seed decides the record order and keys the per-record rng;
    `loading` is how fast the records are read, which changes no record and
    no order. Both are keyword-only, so a kind can still declare a field of
    its own without a default.

    `load` takes `tokenize` on every kind, so a caller holding a
    `DatasetSpec` can load any of them. A dataset that captions its records
    hands it the captions and writes back what it returns. The captions are
    the dataset's own product; the run's condition decides which encoder
    reads them and at which context length. A dataset that carries no
    captions has nothing for a reader to read and says so (`uncaptioned`).
    """

    seed: int = dataclasses.field(default=0, kw_only=True)
    loading: Loading = dataclasses.field(default=Loading(), kw_only=True)

    @abstractmethod
    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        """The dataset's batches, `batch` records a step across every process."""

    def uncaptioned(self, tokenize: Tokenize | None) -> None:
        """Refuses a caption reader handed to a dataset that writes no captions.

        The parameter is on every `load` so one caller can load any spec.
        Silently dropping it would train a conditional run on nothing and
        report no reason.
        """
        if tokenize is not None:
            raise TypeError(
                f"{type(self).__name__} writes no captions, so tokenize= has "
                f"nothing to read; an image or video dataset takes one")


CAPTION = "caption"
"""The batch field a captioning dataset writes its text in, before a run's
conditions read it."""


type Position = bytes | Mapping[str, object]
"""Where a stream stopped, as it reports it: dew's own envelope is bytes and
grain reports a JSON object, which `dew.training.distributed` encodes before
a checkpoint holds it. `dew.position` says which of the two kinds a saved
position is."""


@runtime_checkable
class Checkpointable(Protocol):
    """A data stream that can say where it stopped and be put back there.

    Grain's iterators satisfy this, and so does `GlobalStream`. `Trainer.fit`
    refuses a run that asks for checkpoints over a stream without them. The
    state itself is opaque here; `dew.position` says which of its two kinds
    a checkpoint holds.
    """

    def get_state(self) -> Position: ...

    def set_state(self, state: Position) -> None: ...


def tokenized(stream: Reader, tokenize: Tokenize | None) -> Reader:
    """`stream` with each batch's captions replaced by what `tokenize` reads
    out of them.

    `tokenize` takes the batch's captions and returns the batch fields a
    run's conditions want. An encoder's context length is then the encoder's
    business, and `--text.encoder char_table` and `--text.encoder clip_text`
    read the same dataset. It runs here, on the host, once per batch and
    outside the grain workers, so no encoder's weights are pickled into them.

    The captions never survive the stage, since they are strings and a device
    takes numbers. Pass None for an unconditional run, or a reader that hands
    the words back to keep them.
    """
    def stage(batch: Batch) -> Batch:
        fields = dict(batch)
        captions = [str(caption) for caption in fields.pop(CAPTION)]
        if tokenize is not None:
            fields.update(tokenize(captions))
        return fields

    return mapped(stream, stage)


def tapped(stream: Reader, on_batch: Callable[[Batch], None]) -> Reader:
    """`stream` with `on_batch` called on every batch as it is read, the batch unchanged.

    It runs on whatever thread reads the stream, the trainer's prefetch
    worker included, so a consumer learns of each batch the moment it is read.
    """
    def stage(batch: Batch) -> Batch:
        on_batch(batch)
        return batch

    return mapped(stream, stage)


def mapped(stream: Reader, stage: Callable[[Batch], Batch]) -> Reader:
    """`stream` with `stage` applied to each batch, forwarding stop, close and position."""
    def start(partition: DataPartition) -> Iterator[Batch]:
        source = iter(stream(partition))
        if isinstance(source, Checkpointable):
            return _CheckpointableMapping(source, stage)
        return _Mapping(source, stage)

    return start


class _Mapping(Forwarding):
    """The stream's iterator with one stage applied to each batch."""

    def __init__(self, source: Iterator[Batch], stage: Callable[[Batch], Batch]):
        self._source: Iterator[Batch] | None = source
        self._stage = stage

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        if self._source is None:
            raise StopIteration
        return self._stage(next(self._source))

    def close(self) -> None:
        try:
            super().close()
        finally:
            # request_stop must still reach a source waiting inside close.
            self._source = None


class _CheckpointableMapping(_Mapping):
    """Runs the same stage over a stream that reports and restores its
    position, forwarding both. The methods are written out because the
    protocol reads attributes statically, where a forwarding
    `__getattr__` would only satisfy `hasattr`."""

    def get_state(self) -> Position:
        source = self._source
        if not isinstance(source, Checkpointable):
            raise RuntimeError("the mapped iterator is closed")
        return source.get_state()

    def set_state(self, state: Position) -> None:
        source = self._source
        if not isinstance(source, Checkpointable):
            raise RuntimeError("the mapped iterator is closed")
        source.set_state(state)


class SourceSlice:
    """Reads `source[start:stop]` by index.

    Gives the train and validation loaders disjoint index ranges while the
    shuffle, the epochs and the sharding stay grain's. Attributes stay plain
    because grain pickles the source to its workers.
    """

    def __init__(self, source: Indexed, start: int, stop: int):
        self.source = source
        self.start = start
        self.length = stop - start

    def __repr__(self) -> str:
        # The description a saved position compares against (`describe`).
        # The wrapped source is named by type rather than by its own repr,
        # which for an arrayrecord source is this process's address and for a
        # hub source would download the table to answer a length.
        return (f"SourceSlice({type(self.source).__name__}, "
                f"start={self.start}, length={self.length})")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source[self.start + index]


def hold_out(source: Indexed, records: int, held_out: int,
             name: str) -> tuple[SourceSlice, SourceSlice | None]:
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


def checked_count(count: int, length: int, name: str) -> int:
    """`count` records off the head of `name`'s `length`, refused when there
    are fewer of them than that."""
    if count > length:
        raise ValueError(
            f"count {count} is more than the {length} records of {name}")
    return count


def describe(source: Indexed | pygrain.MapDataset[Batch]) -> str:
    """`source`'s own description, or its type when it has none.

    A saved position names the order it counts into, and the source names
    that order. A resumed run compares two descriptions, so it needs one that
    survives the process that wrote it (`dew/data/sources/text.py` writes
    such a repr). A source without its own is named by type, since the
    default repr is this process's address and two addresses would refuse
    every resume.
    """
    described = type(source).__repr__ is not object.__repr__
    return repr(source) if described else type(source).__name__


@dataclasses.dataclass(frozen=True)
class Corpus:
    """Names one corpus of a weighted mixture, what it reads and how much of
    a step it fills.

    `weight` is a share of the step and not a record count, so the mixture
    holds its proportions whatever the corpora's lengths are. A small corpus
    comes round again while a large one is still on its first pass.

    `offset` is how many records of the corpus's endless training order
    earlier phases of the run already read (`PhasedStream`); the corpus
    starts past them, so a phase continues the order rather than replaying
    its head. Zero for a run of one order.
    """

    name: str
    source: Records
    weight: float
    offset: int = 0


def _endless(source: Records, seed: int, offset: int) -> pygrain.MapDataset[Batch]:
    """`source` reshuffled from `seed` every epoch, endlessly, past its first
    `offset` records."""
    order = pygrain.MapDataset.source(source).seed(seed).shuffle(seed).repeat(None)
    return order[offset:] if offset else order


def _described(corpus: Corpus) -> str:
    """`corpus` as a saved position names it: its source, and where in its
    order the run starts it."""
    start = f", from record {corpus.offset}" if corpus.offset else ""
    return f"{describe(corpus.source)}, {len(corpus.source)} records{start}"


def mixed_counts(corpora: Sequence[Corpus], records: int) -> tuple[int, ...]:
    """How many of the first `records` records of `mixture(corpora, seed)`
    each corpus supplies, for any seed.

    This is grain's own selection read in closed form: weights scaled to
    integers against the smallest (`_float_to_int_proportions`) and the count
    of dataset i in the first k + 1 elements peeled off one proportion at a
    time (`_dataset_and_key_of_next_element`, mix.py at grain 0.2).
    """
    shares = _shares(corpora)
    scale = 100 / min(shares)
    proportions = [int(share * scale) for share in shares]
    counts, remaining, left = [], sum(proportions), records
    for proportion in proportions:
        rest = left * (remaining - proportion) // remaining
        counts.append(left - rest)
        left, remaining = rest, remaining - proportion
    return tuple(counts)


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


def mixture(corpora: Sequence[Corpus], seed: int | None) -> pygrain.MapDataset[Batch]:
    """`corpora` read together at their weights, as one order over records.

    `MapDataset.mix` decides which corpus record k comes from: the one whose
    share of the first k + 1 records is short by one. Every prefix of the
    order, and so every global batch, holds each corpus's share to within one
    record, and nothing is drawn at random
    (`grain/_src/python/dataset/transformations/mix.py:314-347`). Weights
    reach grain as ratios and it scales them to integers against the
    smallest, so a share is exact to a hundredth of the smallest one
    (`mix.py:305-311`).

    With `seed`, every corpus is reshuffled per epoch and repeated before the
    mixing, so the mixture is endless and each corpus cycles its own records
    at its own rate. That is grain's own instruction for keeping a mixture's
    proportions under a shuffle, and what MaxText's pipeline does
    (`input_pipeline/grain_data_processing.py:110-112,169-184`). Without a
    seed the corpora are read in their own order and the mixture stops before
    any of them would come round again (grain's own length rule,
    `mix.py:52-68`), which is the pass a validation split wants.

    The mixing happens before the shard, so a mixture's position is one
    global record count: which corpus record k comes from is a function of k.
    The shares hold over the global batch, not over one process's rows. Two
    corpora at equal weights alternate, so of two processes one reads only
    the first and the other only the second; the step's gradient is the same
    sum either way. A resume onto a changed mixture is refused, as a resume
    onto a changed corpus is.
    """
    shares = _shares(corpora)

    def order(corpus: Corpus) -> pygrain.MapDataset[Batch]:
        if seed is not None:
            return _endless(corpus.source, seed, corpus.offset)
        if corpus.offset:
            raise ValueError("an ordered pass reads each corpus from its first record")
        return pygrain.MapDataset.source(corpus.source)

    return pygrain.MapDataset.mix([order(corpus) for corpus in corpora], list(shares))


def mixed_records(corpora: Sequence[Corpus]) -> int:
    """One pass over a mixture: the records in which every corpus of it has
    been read at least once.

    A mixture has no pass of its own: a corpus filling a tenth of a step
    comes round ten times while one filling the rest is read once. The pass
    is the longest of the corpora's own, which is `len(source)` for one
    corpus and the records of both for two equal corpora at equal weights.
    That is the count `steps_per_epoch` divides everywhere else.
    """
    return max(math.ceil(len(corpus.source) / share)
               for corpus, share in zip(corpora, _shares(corpora), strict=True))


class _WorkerBatches[Record](pygrain.MapDataset[Record]):
    """`parent` re-indexed so each grain worker reads whole, contiguous batches.

    Grain gives worker i every index where i == index % W. Index
    `q*W*B + p*W + w` is record p of batch `q*W + w`, so worker w batches
    inside its own process. The training process interleaves the workers
    round-robin, which restores batch order 0, 1, 2, ...

    The length is padded to whole rounds of one batch per worker. An index
    past the parent's last whole batch answers None, which grain's reader
    skips.
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(self, parent: pygrain.MapDataset[Record], batch: int, workers: int):
        super().__init__(parent)
        self._batch, self._workers = batch, workers
        self._whole = len(parent) // batch
        self._length = min(math.ceil(self._whole / workers) * workers * batch, sys.maxsize)

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: slice) -> pygrain.MapDataset[Record]: ...
    @overload
    def __getitem__(self, index: int) -> Record | None: ...

    def __getitem__(self, index: int | slice) -> Record | pygrain.MapDataset[Record] | None:
        if isinstance(index, slice):
            return self.slice(index)
        worker, within = index % self._workers, index // self._workers
        round_, row = divmod(within, self._batch)
        which = round_ * self._workers + worker
        return None if which >= self._whole else self._parent[which * self._batch + row]


def _batches[Record](records: pygrain.MapDataset[Record], *, rows: int,
                     partition: DataPartition, loading: Loading, offset: int = 0
                     ) -> pygrain.DatasetIterator[Batch]:
    """The partition's share of `records`, in batches of `rows` records.

    The slice is `offset + index :: count`. Global batch k is then the same
    records at every count, and an offset is a slice bound rather than a
    replay.

    Reads are records: the threads behind `to_iter_dataset` each fetch one,
    so no read waits on a whole batch. Grain's `ElasticIterator` batches
    ahead of its read and is an order of magnitude slower on a slow source.
    The batch is stacked inside the worker, so each transfer to this process
    is a whole batch. `_WorkerBatches` permutes the slice to make a worker's
    share whole batches, and that permutation depends only on the batch and
    worker counts, so neither changes which records a batch holds.
    """
    mine = records[offset + partition.index::partition.count]
    if loading.workers:
        mine = _WorkerBatches(mine, rows, loading.workers)
    stream = mine.to_iter_dataset(pygrain.ReadOptions(loading.threads, loading.read_buffer))
    stream = stream.batch(rows, drop_remainder=True)
    if loading.workers:
        stream = stream.mp_prefetch(pygrain.MultiprocessingOptions(
            num_workers=loading.workers,
            per_worker_buffer_size=loading.worker_buffer))
    return iter(stream)


def _shared(source: GrainPipeline, *, rows: int, partition: DataPartition, loading: Loading,
            offset: int = 0) -> pygrain.DatasetIterator[Batch]:
    """The partition's share of `source`, in batches of `rows` records.

    A `MapDataset` is read by index, which is what `_batches` needs to cut a
    share and to start it at a record offset. A pipeline read as it comes has
    neither, so the caller's function builds the share's own and it is
    batched where it is. Only the indexed branch is opened at an offset,
    since only it has a position to resume.
    """
    if isinstance(source, pygrain.MapDataset):
        return _batches(source, rows=rows, partition=partition, loading=loading, offset=offset)
    return iter(source(partition).batch(rows, drop_remainder=True))


def rows_of(batch: Mapping[str, object]) -> int:
    """The records `batch` holds, read off its first field that has rows.

    A batch may carry a field that is one value for the whole step rather
    than one per record, and that field says nothing about how many records
    there are. Any mapping of fields answers, so the trainer's own batch type
    reaches this as it is.
    """
    for leaf in jax.tree.leaves(batch):
        shape = np.shape(leaf)
        if shape:
            return shape[0]
    raise ValueError("a batch of scalars holds no records")


class GlobalStream:
    """Reads a training stream whose saved position is one global record count.

    The stream is a shuffled order over the whole corpus, endlessly, and the
    step is cut out of it here. Global batch k is records
    [k * batch, (k + 1) * batch) of that order, and share p of n reads every
    nth of them starting at p. The position is then how many records the run
    has consumed: one number, the same on every reader, naming no share. What
    two processes wrote is where one process or four resume, on the same
    records in the same steps.

    `open_at(offset)` starts the share's read at a record offset, which is
    what a restore does instead of replaying, since an offset is a slice
    bound. It is called on the first batch and again after `set_state`, so a
    stream restored before it is read starts no worker twice.

    The offset alone would resume that many records into whatever order the
    run now has, so `order` rides with it and a restore that disagrees is
    refused. `dew.position` is the shape both this and `dew.checkpoints`
    read.
    """

    def __init__(self, open_at: Callable[[int], pygrain.DatasetIterator[Batch]],
                 batch: int, order: str, stop_seconds: float):
        self._open = open_at
        self._batch = batch
        self._order = order
        self._records = 0
        self._reads: pygrain.DatasetIterator[Batch] | None = None
        self.stop_seconds = stop_seconds

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

    @property
    def order(self) -> str:
        """The description a saved position is compared against."""
        return self._order

    def set_state(self, state: bytes) -> None:
        saved = position.read(state)
        if saved.completed:
            raise ValueError(
                f"the saved data position is {saved.records} records into a phased run "
                f"past {len(saved.completed)} phase(s), and this run reads one order "
                f"({self._order}); resume it with the phases that wrote it")
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


class PhasedStream:
    """Reads one global order per phase, switching at step boundaries.

    Each phase is a `GlobalStream` factory (`train_stream`, `mixed_stream`)
    and the global record count it ends at, None for the last, which runs
    on. Phase k starts its own order at its own record zero when the run's
    count reaches phase k - 1's end, so which records a step reads is a
    function of the step and the phase list alone, at any process count.

    The saved position is the run's record count, the current phase's order,
    and each completed phase's order with its end (`position.Global`). A
    restore checks the phases the run already read against this list, and
    the current phase's order, and nothing after it: a run may append
    phases, or move a boundary it has not reached, and resume where it
    stopped. A one-order run's position is phase 0 with none completed, so a
    run that trained on one mixture resumes into a phase list starting with
    it. Changing a phase the run has read, or ending the current one before
    the records the run already read in it, is refused.
    """

    def __init__(self, phases: Sequence[tuple[Callable[[], GlobalStream], int | None]],
                 stop_seconds: float):
        ends = [end for _, end in phases]
        bounded = [end for end in ends[:-1] if end is not None]
        if not phases or ends[-1] is not None or len(bounded) != len(ends) - 1:
            raise ValueError("every phase but the last ends at a record count; the last runs on")
        if any(later <= earlier for earlier, later in itertools.pairwise([0, *bounded])):
            raise ValueError(f"phase ends must increase from above zero, got {bounded}")
        self._streams = [open_stream() for open_stream, _ in phases]
        self._ends: tuple[int, ...] = tuple(bounded)
        self._records = 0
        self._current: int | None = None
        self.stop_seconds = stop_seconds

    def __iter__(self) -> Iterator[Batch]:
        return self

    def _phase(self, records: int) -> int:
        return bisect.bisect_right(self._ends, records)

    def _start(self, phase: int) -> int:
        return 0 if phase == 0 else self._ends[phase - 1]

    def __next__(self) -> Batch:
        phase = self._phase(self._records)
        if phase != self._current:
            if self._current is not None:
                self._streams[self._current].close()
            stream = self._streams[phase]
            stream.set_state(position.encode(position.Global(
                records=self._records - self._start(phase), order=stream.order)))
            self._current = phase
        stream = self._streams[phase]
        before = stream.get_state()
        batch = next(stream)
        read = position.read(stream.get_state()).records - position.read(before).records
        self._records += read
        if phase < len(self._ends) and self._records > self._ends[phase]:
            raise ValueError(
                f"a step of {read} records crossed the phase boundary at record "
                f"{self._ends[phase]}; phases end on step boundaries")
        return batch

    def get_state(self) -> bytes:
        # The phase of the last record read: at a boundary nothing of the next
        # phase is read yet, so the finished one is still current and a
        # resume may change what follows it or extend it.
        phase = bisect.bisect_left(self._ends, self._records)
        return position.encode(position.Global(
            records=self._records, order=self._streams[phase].order,
            completed=tuple((self._streams[index].order, self._ends[index])
                            for index in range(phase))))

    def set_state(self, state: bytes) -> None:
        saved = position.read(state)
        for index, (order, end) in enumerate(saved.completed):
            if index >= len(self._ends) or (self._streams[index].order, self._ends[index]) != (order, end):
                raise ValueError(
                    f"the saved run finished phase {index} reading {order} to record "
                    f"{end}, and this run's phase {index} is not that; a phase the "
                    f"run has read cannot change under it")
        phase = len(saved.completed)
        if phase >= len(self._streams) or self._streams[phase].order != saved.order:
            raise ValueError(
                f"the saved position is {saved.records} records into phase {phase}, "
                f"reading {saved.order}, and this run's phase {phase} reads something "
                f"else; resume with the order the checkpoint was written in")
        if phase < len(self._ends) and saved.records > self._ends[phase]:
            raise ValueError(
                f"the saved run has read {saved.records} records, past this run's end "
                f"of phase {phase} at {self._ends[phase]}")
        self.close()
        self._records = saved.records

    def close(self) -> None:
        if self._current is not None:
            self._streams[self._current].close()
        self._current = None


def phased(phases: Sequence[tuple[Callable[[DataPartition], GlobalStream], int | None]], *,
           loading: Loading) -> Callable[[DataPartition], PhasedStream]:
    """A `PhasedStream` factory over `phases`, each a stream factory and the
    global record count it ends at (None for the last); every phase reads
    the share its reader is handed."""
    return lambda partition: PhasedStream(
        [(functools.partial(open_stream, partition), end) for open_stream, end in phases],
        loading.stop_seconds)


@runtime_checkable
class Resumable(Checkpointable, Protocol):
    """Reads batches and can be put back where it stopped, which is what a
    batch ramp wraps."""

    def __next__(self) -> Batch: ...


def _global_position(state: Position) -> bytes:
    """`state`'s own bytes, or the refusal that it is no global position.

    A ramp cuts its step out of a global record order, so it reads the
    source's position. Grain's own state counts one process's shard and holds
    no global count.
    """
    if not isinstance(state, bytes):
        raise TypeError(
            f"a batch ramp reads a global record position, and this stream "
            f"reports {type(state).__name__}, which counts one process's shard")
    return state


class RampedStream(Forwarding):
    """Grows a training stream's step over the run's first records.

    Wraps a stream whose position is a global record count, a `GlobalStream`
    or `tokenized` over one, and decides only how many of its records a step
    reads. Reads run a whole final-size batch ahead and each step is cut out
    of a rolling buffer, as MaxText's do (`common/data_loader.py:122-176`).
    A step therefore reads the records the run would have read without the
    ramp, in the same order, and a stage change neither skips a record nor
    reads one twice.

    What is left in the buffer has not been trained on and is not in the
    position, which is the source's count replaced by the records handed out.
    A restore hands the source that count, so it reopens its read there.
    """

    def __init__(self, source: Resumable, stages: Sequence[Stage], partition: DataPartition):
        self._source = source
        self._stages = tuple(stages)
        self._starts = tuple(stage.records for stage in stages)
        self._partition = partition
        self._records = 0
        self._held: Batch | None = None

    def __iter__(self) -> Iterator[Batch]:
        return self

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The batch this stream reads at each stage, and where each begins.

        The trainer reads them off the stream it opened. A compiled step per
        stage is the whole set of shapes a ramped run runs, and the mesh has
        to divide every one of them.
        """
        return self._stages

    def _stage(self) -> Stage:
        """The stage the records read so far leave the run in."""
        return self._stages[bisect.bisect_right(self._starts, self._records) - 1]

    def __next__(self) -> Batch:
        stage = self._stage()
        rows = self._partition.rows(stage.batch)
        while self._held is None or rows_of(self._held) < rows:
            read = next(self._source)
            self._held = read if self._held is None else jax.tree.map(
                lambda kept, new: np.concatenate([kept, new]), self._held, read)
        step = jax.tree.map(lambda field: field[:rows], self._held)
        self._held = jax.tree.map(lambda field: field[rows:], self._held)
        self._records += stage.batch
        return step

    def get_state(self) -> bytes:
        place = position.read(_global_position(self._source.get_state()))
        return position.encode(dataclasses.replace(place, records=self._records))

    def set_state(self, state: bytes) -> None:
        self._source.set_state(state)
        self._held = None
        self._records = position.read(state).records
        stage = self._stage()
        if (self._records - stage.records) % stage.batch:
            raise ValueError(
                f"the saved data position is {self._records} records in, and this "
                f"ramp reads {stage.batch} records a step from record "
                f"{stage.records} on, so no step of it ends there: the checkpoint "
                f"was written by a run that ramped differently")

    def close(self) -> None:
        self._held = None
        super().close()


def ramped(dataset: Dataset, ramp: Ramp) -> Dataset:
    """`dataset` with the training batch growing to `dataset.batch` over `ramp`.

    A validation pass keeps the whole batch, since a score over a growing
    number of records is a score of a different thing each time.

    Only a stream whose position is a global record count can ramp. The ramp
    cuts that order into other steps, and the count it saves has to be the
    records it handed over. A stream that batches its own records reports a
    shard offset and is refused, by name.
    """
    stages = ramp.stages(dataset.batch)  # An impossible schedule fails here, not mid-run.

    def train(partition: DataPartition) -> Iterator[Batch]:
        stream = dataset.train(partition)
        if isinstance(stream, PhasedStream):
            stream.close()
            raise TypeError(
                "phases end at step boundaries of the full batch, and a batch ramp "
                "cuts other steps; ramp a run of one order")
        if isinstance(stream, Resumable):
            state = stream.get_state()
            if isinstance(state, bytes) and position.translates(state):
                return RampedStream(stream, stages, partition)
        if isinstance(stream, Closeable):
            stream.close()
        raise TypeError(
            f"a batch ramp cuts the step out of a global record order, and "
            f"{type(stream).__name__} hands over batches it has cut itself; ramp "
            f"a dataset read through train_stream or mixed_stream")

    return dataclasses.replace(dataset, train=train, ramp=ramp)


def train_stream(source: Records, operations: Sequence[pygrain.Transformation], *,
                 batch: int, seed: int, loading: Loading,
                 offset: int = 0) -> Callable[[DataPartition], GlobalStream]:
    """An endless shuffled stream over `source`, `batch` records a global step.

    The order is the corpus reshuffled from `seed` every epoch, endlessly,
    and `_batches` cuts the reader's share off it. Global batch k is then the
    same records at every partition, and a `GlobalStream` position is a
    record count rather than a shard offset.

    `operations` run behind the order and ahead of the slice. They therefore
    run inside the workers, a record's rng is keyed by its place in the
    endless stream, and what a record becomes depends on neither count.

    `offset` starts the order past the records earlier phases read.
    """
    start = f", from record {offset}" if offset else ""
    order = f"{describe(source)}, {len(source)} records reshuffled from seed {seed}{start}"

    def records() -> pygrain.MapDataset[Batch]:
        return _endless(source, seed, offset).apply(list(operations))

    return _global_stream(records, order, batch=batch, loading=loading)


def mixed_stream(corpora: Sequence[Corpus], operations: Sequence[pygrain.Transformation], *,
                 batch: int, seed: int, loading: Loading) -> Callable[[DataPartition], GlobalStream]:
    """An endless stream over `corpora` at their weights, `batch` records a global step.

    The order is `mixture(corpora, seed)` and everything after it is
    `train_stream`'s: the same share's slice, the same batch behind the
    reads, and the same position. A mixture's place in its corpora is
    therefore one record count and resumes on any partition. The
    `operations` sit above the mixture, so a record's rng is keyed by its
    place in the mixed stream rather than in the corpus it came from.
    """
    shares = _shares(corpora)
    order = "mixture reshuffled from seed {} of [{}]".format(seed, ", ".join(
        f"{corpus.name} at {share:.6g}: {_described(corpus)}"
        for corpus, share in zip(corpora, shares, strict=True)))

    def records() -> pygrain.MapDataset[Batch]:
        return mixture(corpora, seed).seed(seed).apply(list(operations))

    return _global_stream(records, order, batch=batch, loading=loading)


def _global_stream(records: Callable[[], pygrain.MapDataset[Batch]], order: str, *,
                   batch: int, loading: Loading) -> Callable[[DataPartition], GlobalStream]:
    """A `GlobalStream` factory over the endless order `records` builds.

    Each reader gets its own pipeline over its share, opened at whatever
    record offset a restore hands it, and `order` is the description a saved
    position is compared against.
    """
    def stream(partition: DataPartition) -> GlobalStream:
        rows = partition.rows(batch)

        def open_at(offset: int) -> pygrain.DatasetIterator[Batch]:
            return _batches(records(), rows=rows, partition=partition, loading=loading,
                            offset=offset)

        return GlobalStream(open_at, batch, order, loading.stop_seconds)

    return stream


def validation_pass(source: Records, transformations: Sequence[pygrain.Transformation], *,
                    batch: int, seed: int, loading: Loading) -> Reader:
    """One pass over `source` in record order, `batch` records a global step.

    Grain's DataLoader gives each worker its own slice of the split to fill a
    whole batch out of, so which records a batch holds moves with
    worker_count. `_batches` hands a worker the records of one batch instead,
    so the split is cut into the same batches at every count.

    Sharding is grain's slice convention, so share p of n reads records
    p, p + n, ... of the split. The transforms are applied before that slice
    because grain keys a record's rng by its index in the dataset the random
    map sits on. Applied after the slice, record k would take its key from
    its place in the slice, and one seed would augment it differently on one
    host than on a pod. A pass is whole batches only, because a part-full
    batch cannot be sharded over a device mesh.
    """
    def stream(partition: DataPartition) -> Iterator[Batch]:
        records = pygrain.MapDataset.source(source).seed(seed).apply(list(transformations))
        return _batches(records, rows=partition.rows(batch), partition=partition, loading=loading)

    return stream
