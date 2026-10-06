"""The `Dataset` a run trains on, and the Grain plumbing every dataset shares.

A `DatasetSpec` is a frozen dataclass registered with `@datasets(name)` that
says what a dataset is and how to read it. `load(batch=)` turns it into a
`Dataset`, which a recipe passes to the trainer. This module holds what the
image, video and token specs have in common: the share of each batch a reader
reads, the shuffled training stream, the ordered validation pass, and the
slice that keeps the two disjoint.

Every stream is opened for a `DataPartition`, the share of each global batch
its reader reads. The trainer gets it from the mesh (`DataPartition.of`),
because processes that a pipeline or a split sequence spans hold the same rows
and so read the same share. A loader reads the share it is given and nothing
else.

A training stream's position is one global record count, not a shard offset,
so a run saved on one process count resumes on another. `GlobalStream` here
and `dew.position` define that contract. A weighted `mixture` of corpora and a
`Ramp` of the batch are both cut from that one order, so neither adds anything
to what a checkpoint holds.
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import functools
import itertools
import json
import logging
import math
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

import grain.python as pygrain
import jax
import numpy as np
import tyro
from absl import flags
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from numpy.typing import ArrayLike

from dew import position
from dew.nn.sharding import BATCH_AXES, LayoutRefused

# `Batch` lives in dew.objectives.base. The data layer imports it from here
# so a dataset module needs one import for the value and its shape.
from dew.objectives.base import VALID_ROWS, Batch

_log = logging.getLogger(__name__)

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
    """The share of every global batch a reader reads: the `index`th of `count` equal, disjoint shares.

    A loader cuts its record order as `index :: count`, so the shares of
    global batch k together hold the same records at every count, and a record
    count marks the same place in the stream whatever the count. The trainer
    gets the share a process reads from the mesh (`DataPartition.of`).
    Processes whose devices hold the same rows read the same share, because
    the axes between them split a sequence or hold a pipeline's stages, not
    rows. `DataPartition()` is a single reader of every row, which is what a
    single process reads.
    """

    index: int = 0
    count: int = 1
    readers: int = 1
    """The number of processes that read this share.

    Each reads the same records. A source whose rows can differ between reads
    of one share, such as a network fetch that drops failed records, must
    either refuse more than one reader or read on the first reader alone and
    send its batch to the others (`dew.training.distributed.first_reader_batch`).
    """
    reader: int = 0
    """This process's place among the share's readers, in process order (0 for the first)."""

    def __post_init__(self):
        if not 0 <= self.index < self.count or not 0 <= self.reader < self.readers:
            raise ValueError(
                f"a data partition is share index of count, 0 <= index < count, read "
                f"by one or more readers, of which this is reader 0 <= reader < readers; "
                f"got index {self.index} of {self.count} read by {self.readers}, reader "
                f"{self.reader}")

    @classmethod
    def of(cls, mesh: Mesh) -> DataPartition:
        """Return the share of every global batch this process reads on `mesh`.

        A batch's rows split over the batch axes and no others (`BATCH_SPEC`).
        The sequence axis splits positions, the tensor axis splits widths, and
        the stage axis holds a pipeline's stages, so processes whose devices
        hold the same row shards need the same rows. The processes therefore
        fall into groups by the rows they hold. Each group reads one share,
        numbered by the first row shard it holds, and every process in the
        group reads it (`readers`); `reader` is this process's place among
        them, in process order.

        A hand-built device order can produce groups whose rows overlap without
        being the same rows. No share can then be read whole by each group, so
        such a mesh is refused.
        """
        return _partition(mesh)

    def rows(self, batch: int) -> int:
        """Return how many rows of a `batch`-row global batch one share holds.

        `batch` must split evenly into the shares: a remainder would train on
        fewer records a step than the run reports, and a batch smaller than
        the share count would leave a share with nothing.
        """
        if batch % self.count:
            raise ValueError(
                f"batch {batch} does not split into {self.count} equal shares, "
                f"one for each group of processes that reads its own rows")
        return batch // self.count


@functools.cache
def _partition(mesh: Mesh) -> DataPartition:
    shards = math.prod(mesh.shape[axis] for axis in BATCH_AXES)
    held: dict[int, set[int]] = {}
    placement = NamedSharding(mesh, P(BATCH_AXES)).devices_indices_map((shards,))
    for device, index in placement.items():
        held.setdefault(device.process_index, set()).add(index[0].start or 0)
    groups = sorted({frozenset(rows) for rows in held.values()}, key=min)
    if (sum(len(group) for group in groups) != shards
            or len({len(group) for group in groups}) != 1):
        raise LayoutRefused(
            f"the processes of this mesh hold the row shards "
            f"{ {process: sorted(rows) for process, rows in sorted(held.items())} }, "
            f"which overlap without being the same; each group of processes has to "
            f"hold rows no other group holds, so it can read them as its own share")
    mine = frozenset(held[jax.process_index()])
    readers = sorted(process for process, rows in held.items() if frozenset(rows) == mine)
    return DataPartition(index=groups.index(mine), count=len(groups), readers=len(readers),
                         reader=readers.index(jax.process_index()))


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
    """Returns records by index, which is how the loaders read a source.

    A record is one example's fields, in the shape `Batch` names, or the
    packed bytes an ArrayRecord file holds, which the spec's own transform
    unpacks into fields. A Grain dataset returns None for an index its padding
    covers, and Grain's reader skips those instead of batching them.

    Grain's own `RandomAccessDataSource` declares the same two methods with the
    record as a type parameter. That parameter is invariant, so a source
    declared through it cannot be passed to a loader that names a different
    record type. This protocol names what the loaders actually read.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Batch | bytes | None: ...


type Indexed = Records | Sequence[Batch]
"""Records read by index: a source that answers grain's two methods, or a
plain sequence of them. A spec that lists its records in memory hands over
the sequence, the way the video specs list their clips."""

type InMemory = Mapping[str, ArrayLike] | Indexed
"""Records a caller holds, for `Dataset.from_records`: columns whose first
axis is the record, or anything `Indexed`."""


class Columns:
    """Reads records from equal-length columns, one row of each column per record.

    The description that a saved position is compared against names the
    fields, their per-record shapes and dtypes, and the record count. It cannot
    tell apart two tables with the same layout; no source description can
    without reading the data.
    """

    def __init__(self, columns: Mapping[str, ArrayLike]):
        held = {name: np.asarray(column) for name, column in columns.items()}
        if not held:
            raise ValueError("in-memory records need at least one column")
        first, *_ = held
        for name, column in held.items():
            if column.ndim == 0:
                raise ValueError(
                    f"column {name!r} is a single value; a column holds one value per "
                    f"record along its first axis")
            if len(column) != len(held[first]):
                raise ValueError(
                    f"a column holds one row per record, and {name!r} holds "
                    f"{len(column)} records and {first!r} holds {len(held[first])}")
        self._columns = held

    def __len__(self) -> int:
        return len(next(iter(self._columns.values())))

    def __getitem__(self, index: int) -> Batch:
        return {name: column[index] for name, column in self._columns.items()}

    def __repr__(self) -> str:
        fields = ", ".join(f"{name} {column.dtype}{list(column.shape[1:])}"
                           for name, column in self._columns.items())
        return f"Columns({fields}; {len(self)} records)"


def in_memory(records: InMemory) -> Indexed:
    """`records` as a source read by index: columns wrapped, the rest as given."""
    return Columns(records) if isinstance(records, Mapping) else records


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
    of records whose length the command line decides.

    Each entry is read and written as a run record holds it, so a field that
    holds a registered value, such as a `ParamGroup`'s schedule, is the record
    that names it, `{"name": ..., "fields": {...}}`."""
    from dew.registry import from_record, to_record

    def written(given: tuple[Entry, ...]) -> list[str]:
        return [json.dumps([to_record(value, entry) for value in given])]

    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda given: tuple(
            from_record(entry, record, dtypes=False) for record in json.loads(given[0])),
        is_instance=lambda given: isinstance(given, tuple) and all(
            isinstance(value, entry) for value in given),
        str_from_instance=written,
    )


@dataclasses.dataclass(frozen=True)
class DataPhase:
    """One phase of a run's data: what it reads and the step it ends at.

    `path` names one corpus or a weighted mixture, the same way the spec's own
    `path` does. `until_step` is the step the phase ends before, counted from
    the run's start in steps of the full batch, or None for the last phase,
    which runs to the end. `PhasedStream` describes how a resume treats a
    changed phase list.
    """

    path: str | Mapping[str, float]
    until_step: int | None = None


@runtime_checkable
class Closeable(Protocol):
    """A stream that holds resources a stopped run must release, such as worker processes,
    file handles or a shared memory block."""

    def close(self) -> None: ...


@runtime_checkable
class Stoppable(Protocol):
    """A stream that can be asked to stop before it is drained."""

    def request_stop(self) -> None: ...


@runtime_checkable
class Budgeted(Protocol):
    """A stream that reports how long stopping it may take.

    It is separate from `Stoppable` because a stream can report a budget
    without accepting a stop request. A wrapper forwards each one separately.
    """

    @property
    def stop_seconds(self) -> float | None: ...


class Forwarding:
    """Forwards a stream wrapper's stop signal, stop budget and close to its source.

    Subclasses keep the source at `_source` and override `close` for their own
    cleanup around `super().close()`.
    """

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


_QUIET_STOP_SECONDS = 5.0
"""How long a stop of grain's workers runs before it says what it waits for
(`Loading.announced_stop`)."""


def _grain_kill_seconds() -> int:
    """How long grain waits for a stopped worker process to exit before it
    kills it: its own `_PROCESS_KILL_TIMEOUT_S`. The name is private, so it is
    read where a grain-backed stream needs it, and a grain release that moves
    it breaks only grain-backed loading."""
    from grain._src.python.dataset.transformations import process_prefetch

    return process_prefetch._PROCESS_KILL_TIMEOUT_S


@dataclasses.dataclass(frozen=True)
class Loading:
    """Read throughput settings, as Grain's four knobs.

    None of the four changes which records a run sees or what they contain, so
    a host can tune them for its disk and still produce identical batches. The
    shuffle seed is not one of them: it sets the order records arrive in and
    keys the per-record rng that augments and captions them.

    Each knob counts something different:

    - `workers`: worker processes;
    - `threads`: record reads one worker keeps in flight;
    - `read_buffer`: records one worker reads ahead;
    - `worker_buffer`: whole batches one worker holds ready for the training
      process, since a worker stacks the records it read into batches.

    Zero workers is Grain's own default: threads in the training process read
    the records. Each worker process imports the program again, so it costs
    seconds and a process's memory before the first batch, and pays off only
    once decoding or augmentation is slower than the threads. On two cores of
    a shared workstation, the first batch of a 100-record pipeline took 17.8 s
    and 7.07 GiB resident across 33 processes with 32 workers, against under
    0.1 s and 0.23 GiB with none.
    """

    workers: int = 0
    threads: int = 64
    read_buffer: int = 128
    worker_buffer: int = 2

    @property
    def stop_seconds(self) -> float:
        """How long stopping this many workers may take before it counts as a hang.

        This is Grain's own bound. A stop first waits for the batch being read.
        Grain then stops the worker processes one after another, each finishing
        its current batch before it exits, and kills a worker that has not
        exited within Grain's kill timeout; the read gets the same time. Slow
        stops are normal: four workers took 5.1 to 7.4 s to stop on 12 vCPUs.
        """
        # Grain's kill timeout is read from Grain (`_grain_kill_seconds`), so an
        # upgrade that changes it moves this budget too. The 5.1 to 7.4 s stop
        # was examples/sft_gemma4.py's four workers.
        return float(_grain_kill_seconds() * (1 + self.workers))

    @contextlib.contextmanager
    def announced_stop(self) -> Iterator[None]:
        """Stop the workers inside this context, explaining on stderr if the stop takes long.

        If the stop runs past 5 seconds (`_QUIET_STOP_SECONDS`), it prints once
        what it is waiting for, so a long stop does not look like a hang.
        """
        announcer = self._announcer()
        try:
            yield
        finally:
            if announcer is not None:
                announcer.cancel()
                announcer.join()

    def _announcer(self) -> threading.Timer | None:
        """A started timer that prints what a stop of these workers waits for
        once `_QUIET_STOP_SECONDS` pass. None with no workers, and None where
        no thread can start (at interpreter shutdown, or at the process's
        thread limit), since the stop must run with or without its line."""
        if not self.workers:
            return None
        line = (f"waiting for {self.workers} grain workers to stop, "
                f"up to {self.workers * _grain_kill_seconds()} s")
        timer = threading.Timer(_QUIET_STOP_SECONDS, _log.warning, (line,))
        timer.daemon = True
        try:
            timer.start()
        except RuntimeError as refused:
            _log.warning("stopping %s grain workers without a progress line: %s", self.workers, refused)
            return None
        return timer


_DEFAULT_LOADING = Loading()


@dataclasses.dataclass(frozen=True)
class Stage:
    """One stage of a batch ramp: the global batch a step reads in it, and the record it starts at."""

    batch: int
    records: int


@dataclasses.dataclass(frozen=True)
class Ramp:
    """Grows the global batch over the run's first records.

    A run starts at `start` records a step and adds `increment` each time the
    records read since the last increment reach `samples` divided by the
    number of increments. It stops at the batch the dataset was loaded with,
    and the difference must be a whole number of increments, as in MaxText's
    batch rampup.

    MaxText counts the batch per device, while Dew counts the global batch
    everywhere, so `start` and `increment` are records a step: MaxText's
    `per_device_batch_size_start` times the device count is `start` here.

    The stage a run is in depends only on the records it has read, which a
    checkpoint already holds as the data position, so a resumed run continues
    the ramp with nothing else saved. A stage lasts
    `ceil(samples / increments / batch)` steps, computed as one integer ratio
    instead of MaxText's two floating-point divisions, so the boundaries are
    exact. Where the ramp ends, MaxText's loader drops what is left of its
    buffered batch while Dew keeps reading, so from then on MaxText reads
    later records than Dew in each step, although the batch sizes match.
    """
    # MaxText reference: configs/base.yml:755-765 and
    # utils/rampup_batch.py:38-50,53-101.

    start: int
    increment: int
    samples: int

    def stages(self, final: int) -> tuple[Stage, ...]:
        """Return every stage of the ramp up to `final`, the run's own global batch.

        Each stage's batch has to split evenly into the mesh's rows, and so
        into the shares its readers read; the trainer checks every stage
        against the mesh before the run starts.
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
        """Return the number of steps a run takes to read `records` records under this ramp.

        Early steps read fewer records, so a pass over a corpus takes more
        steps than it would with the final batch from the start.
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
    opens one pass over the held-out records in a fixed order, which ends by
    itself; `val` is None when nothing is held out. Both read the share of
    each global batch that `partition` names (`DataPartition`). `batch` is the
    global batch and `records` the number of training records behind it, so
    `steps_per_epoch` is one pass over them. `ramped` sets `ramp` when the run
    grows its batch over its first records, and `batch` is then the batch the
    ramp ends at. `held_out` is how many records of the training split a spec
    kept back for validation, which `fit` reports when the run starts; it is 0
    when validation is a separate split or there is none.

    Each factory call returns a fresh iterator that the caller owns. Close it
    after use if it has `close`, but never close the shared dataset or its
    backing store. A source's optional `request_stop` is a separate,
    thread-safe signal; it does not make it safe to call the final `close`
    while another thread is still iterating.

    Image and video fields are uint8 in [0, 255], text is the tokenized
    `{"input_ids", "attention_mask"}` dict under "text", and a token window
    is int32 ids under "text".

    Whether a run can checkpoint its position depends on the iterator.
    Grain-backed iterators have `get_state` and `set_state`, a fetch-as-you-go
    stream has neither, and `tokenized` forwards the pair. A run over a stream
    without them must train with `checkpoint_every=None` and is refused
    otherwise. A `train_stream` position is global and resumes on any
    partition, over one corpus or a weighted mixture. A stream that batches
    its own records reports its share's own position, and `dew.checkpoints`
    resumes it only on a reader of that share.
    """

    train: Reader
    val: Reader | None
    records: int | None
    batch: int
    ramp: Ramp | None = None
    held_out: int = 0

    @classmethod
    def from_grain(cls, train: GrainPipeline, *, batch: int,
                   validation: GrainPipeline | None = None,
                   records: int | None = None,
                   loading: Loading = _DEFAULT_LOADING) -> Dataset:
        """Build a dataset from Grain pipelines the caller built.

        The caller decides the order, the shuffle and what each record turns into.
        This adds what every spec's `load` adds, with the same helpers: the reader's
        share of the batch, whole training batches, a validation pass over every
        record with its last batch padded (`VALID_ROWS`), and the state pair a
        checkpoint saves.

        A `MapDataset` is read by index, so it gets the same training stream as
        every spec: repeated endlessly, cut into the reader's share, and saved as one
        global record count. A pipeline read in sequence is sharded by whoever builds
        it, so pass it as a function that takes the partition and builds that
        share's `IterDataset`. It is batched as it is and reports Grain's own
        iterator state, which `dew.checkpoints` restores only into a reader of the
        same share.

        `records` is the number of records in one pass, which `steps_per_epoch`
        divides. It defaults to a `MapDataset`'s own length, so if you repeated your
        dataset before passing it in, give the length of one pass instead. A Grain
        pipeline has no description of its own, so the saved position names the
        pipeline's type and length, not the corpus under it. If you swap the corpus
        under one pipeline, Dew cannot detect it.
        """
        mapped = train if isinstance(train, pygrain.MapDataset) else None
        for pipeline in (train, validation):
            if isinstance(pipeline, pygrain.MapDataset):
                _refuse_filters(pipeline)
        if mapped is not None:
            endless = mapped.repeat(None)
            order = f"{describe(mapped)}, {len(mapped)} records"

            def training(partition: DataPartition) -> Iterator[Batch]:
                rows = partition.rows(batch)
                return GlobalStream(
                    lambda offset: _shared(endless, rows=rows, partition=partition,
                                           loading=loading, offset=offset),
                    batch, order, loading)
        else:
            def training(partition: DataPartition) -> Iterator[Batch]:
                return _shared(train, rows=partition.rows(batch), partition=partition,
                               loading=loading)

        def validating(partition: DataPartition) -> Iterator[Batch]:
            assert validation is not None
            return _shared(validation, rows=partition.rows(batch), partition=partition,
                           loading=loading, remainder=True)

        return cls(
            train=training,
            val=None if validation is None else validating,
            records=len(mapped) if records is None and mapped is not None else records,
            batch=batch,
        )

    @classmethod
    def from_records(cls, records: InMemory, *, batch: int, seed: int = 0,
                     validation: InMemory | None = None,
                     loading: Loading = _DEFAULT_LOADING) -> Dataset:
        """Build a dataset from records the caller holds: columns, rows or a source.

        `records` is a mapping of columns whose first axis is the record, such as
        `{"x": x, "y": y}`, a sequence of per-record mappings, or any source read by
        index. Training reshuffles the records from `seed` every epoch, with the same
        stream every spec reads (`train_stream`), so the position a checkpoint saves
        is a global record count and each process reads its own share of every
        batch. `validation` is read once, in order, and every record is scored, with
        the last batch padded (`VALID_ROWS`).
        """
        held = None if validation is None else in_memory(validation)
        source = in_memory(records)
        if len(source) < batch:
            raise ValueError(
                f"{len(source)} training records, fewer than one batch of {batch}: a "
                f"batch would hold a record twice and an epoch would take no steps")
        return cls(
            train=train_stream(source, [], batch=batch, seed=seed, loading=loading),
            val=None if held is None else validation_pass(held, [], batch=batch, seed=seed,
                                                          loading=loading),
            records=len(source),
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
        """Return the number of steps that `epochs` passes over the records take.

        A stream without a record count has no epoch, so a run over it must
        give its length in steps instead.
        """
        if self.steps_per_epoch is None:
            raise ValueError(
                "epochs need a dataset with a record count; this one streams "
                "without one, so give the run length in steps")
        return epochs * self.steps_per_epoch


@dataclasses.dataclass(frozen=True)
class DatasetSpec(ABC):
    """Describes a dataset and how to read it, with one frozen dataclass per kind.

    `seed` and `loading` apply to every kind, so they are declared once here.
    The seed sets the record order and keys the per-record rng; `loading` sets
    how fast records are read, which changes neither the records nor their
    order. Both are keyword-only, so a kind can still declare its own fields
    without defaults.

    Every kind's `load` takes `tokenize`, so a caller holding any
    `DatasetSpec` can load it. A dataset that captions its records passes the
    captions to `tokenize` and writes back what it returns. The dataset only
    produces the captions; the run's condition decides which encoder reads
    them and at which context length. A dataset without captions has nothing
    for a reader and raises if given one (`uncaptioned`).
    """

    seed: int = dataclasses.field(default=0, kw_only=True)
    loading: Loading = dataclasses.field(default=Loading(), kw_only=True)

    @abstractmethod
    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        """Return the dataset's batches, `batch` records a step across all processes."""

    def uncaptioned(self, tokenize: Tokenize | None) -> None:
        """Raise if a caption reader is given to a dataset that writes no captions.

        The parameter is on every `load` so that one caller can load any spec.
        Dropping it silently would train a conditional run on nothing without
        saying why.
        """
        if tokenize is not None:
            raise TypeError(
                f"{type(self).__name__} writes no captions, so tokenize= has "
                f"nothing to read; an image or video dataset takes one")


CAPTION = "caption"
"""The batch field a captioning dataset writes its text in, before a run's
conditions read it."""


type Position = bytes | Mapping[str, object]
"""Where a stream stopped, as the stream reports it.

Dew's own envelope is bytes, and Grain reports a JSON object, which
`dew.training.distributed` encodes before a checkpoint stores it.
`dew.position` tells which of the two kinds a saved position is."""


@runtime_checkable
class Checkpointable(Protocol):
    """A data stream that can report where it stopped and resume from there.

    Grain's iterators satisfy this, and so does `GlobalStream`. `Trainer.fit`
    refuses a run that asks for checkpoints over a stream without these
    methods. The state itself is opaque here; `dew.position` tells which of its
    two kinds a checkpoint holds.
    """

    def get_state(self) -> Position: ...

    def set_state(self, state: Position) -> None: ...


def tokenized(stream: Reader, tokenize: Tokenize | None) -> Reader:
    """Return `stream` with each batch's captions replaced by the fields `tokenize` makes from them.

    `tokenize` takes the batch's captions and returns the batch fields the
    run's conditions need. The encoder then decides its own context length,
    so `--text.encoder char_table` and `--text.encoder clip_text` read the
    same dataset. It runs on the host, once per batch and outside the Grain
    workers, so no encoder weights are pickled into the workers.

    The captions themselves are dropped, since they are strings and a device
    takes numbers. Pass None for an unconditional run, or a `tokenize` that
    returns the words to keep them.
    """
    def stage(batch: Batch) -> Batch:
        fields = dict(batch)
        captions = [str(caption) for caption in fields.pop(CAPTION)]
        if tokenize is not None:
            fields.update(tokenize(captions))
        return fields

    return mapped(stream, stage)


def tapped(stream: Reader, on_batch: Callable[[Batch], None]) -> Reader:
    """Return `stream` with `on_batch` called on every batch as it is read, leaving the batch unchanged.

    The callback runs on whatever thread reads the stream, including the
    trainer's prefetch worker, so a consumer sees each batch as soon as it is
    read.
    """
    def stage(batch: Batch) -> Batch:
        on_batch(batch)
        return batch

    return mapped(stream, stage)


def mapped(stream: Reader, stage: Callable[[Batch], Batch]) -> Reader:
    """Return `stream` with `stage` applied to each batch, forwarding stop, close and position."""
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

    It gives the train and validation loaders disjoint index ranges while
    Grain still handles the shuffle, the epochs and the sharding. The
    attributes stay plain because Grain pickles the source to its workers.
    """

    def __init__(self, source: Indexed, start: int, stop: int):
        self.source = source
        self.start = start
        self.length = stop - start

    def __repr__(self) -> str:
        # The description a saved position compares against, so it carries
        # the wrapped corpus's own (`describe`).
        return (f"SourceSlice({describe(self.source)}, "
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


def _refuse_filters(pipeline: pygrain.MapDataset[Batch]) -> None:
    """Refuse a pipeline holding a `filter`, as grain's `ElasticIterator` does.

    A filter answers None at the indices it drops, so an index range yields
    fewer records than it spans: a global record count then no longer says
    where a resumed run starts, and process shares cut by index fill uneven
    batches. grain names no public type for the filter, so it is read from
    where grain's own iterator reads it.
    """
    from grain._src.python.dataset.transformations.filter import FilterMapDataset

    pending: list[pygrain.MapDataset[Batch]] = [pipeline]
    while pending:
        node = pending.pop()
        if isinstance(node, FilterMapDataset):
            raise ValueError(
                "this MapDataset filters its records, and a filter yields fewer records than "
                "the indices it reads, so a record count cannot say where a resumed run "
                "starts; filter the records before the source, or pass a function of the "
                "partition that builds an IterDataset, whose position is grain's own")
        pending.extend(node.parents)


def describe(source: Indexed | pygrain.MapDataset[Batch]) -> str:
    """`source`'s own description, or its type when it has none.

    A saved position names the order it counts into, and the source names
    that order. A resumed run compares two descriptions, so it needs one that
    survives the process that wrote it (`dew/data/sources/text.py` writes
    such a repr). A source without its own is named by type, since the
    default repr is this process's address and two addresses would refuse
    every resume. So is a list or tuple of records, whose repr is every
    record it holds.
    """
    described = type(source).__repr__ is not object.__repr__ and not isinstance(source, (list, tuple))
    return repr(source) if described else type(source).__name__


@dataclasses.dataclass(frozen=True)
class Corpus:
    """One corpus of a weighted mixture: what it reads and how much of a step it fills.

    `weight` is a share of the step, not a record count, so the mixture keeps
    its proportions whatever the corpora's lengths. A small corpus comes round
    again while a large one is still on its first pass.

    `offset` is how many records of the corpus's endless training order
    earlier phases of the run already read (`PhasedStream`). The corpus starts
    after them, so a phase continues the order instead of replaying its start.
    It is zero for a run with one order.
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
    past the parent's last whole batch, or with `remainder` past its last
    record, answers None, which grain's reader skips.
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(self, parent: pygrain.MapDataset[Record], batch: int, workers: int, *,
                 remainder: bool = False):
        super().__init__(parent)
        self._batch, self._workers = batch, workers
        self._whole = -(-len(parent) // batch) if remainder else len(parent) // batch
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
        record = which * self._batch + row
        return None if which >= self._whole or record >= len(self._parent) else self._parent[record]


class _Filled(pygrain.MapTransform):
    """A batch of fewer than `rows` records filled out with repeats of its
    own rows (`RowPlan.pad`), and `VALID_ROWS` marking the real ones. Without
    `real` every row is a repeat."""

    def __init__(self, rows: int, *, real: bool = True):
        self._rows, self._real = rows, real

    def map(self, element: Batch) -> Batch:
        from dew.nn.inputs import RowPlan

        held = rows_of(element)
        if held == self._rows and self._real:
            return element
        plan = RowPlan(None, held, self._rows, 0, 1)
        return {**plan.pad(element), VALID_ROWS: ~plan.padding & self._real}


def _batches[Record](records: pygrain.MapDataset[Record], *, rows: int,
                     partition: DataPartition, loading: Loading, offset: int = 0,
                     remainder: bool = False) -> pygrain.DatasetIterator[Batch]:
    """The partition's share of `records`, in batches of `rows` records.

    A training stream is endless and takes whole batches. An evaluation pass
    (`remainder`) takes every record: its last batch is filled out to `rows`
    (`_Filled`).

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
    # A share a pass's split holds no record for, as process 1's of one record
    # on two, reads one batch of the split's first record with every row a
    # repeat: it meets its peers' first batch and scores nothing.
    empty = remainder and not len(mine) and len(records)
    if empty:
        mine = records[:1]
    if loading.workers:
        mine = _WorkerBatches(mine, rows, loading.workers, remainder=remainder)
    stream = mine.to_iter_dataset(pygrain.ReadOptions(loading.threads, loading.read_buffer))
    stream = stream.batch(rows, drop_remainder=not remainder)
    if remainder:
        stream = stream.map(_Filled(rows, real=not empty))
    if loading.workers:
        stream = stream.mp_prefetch(pygrain.MultiprocessingOptions(
            num_workers=loading.workers,
            per_worker_buffer_size=loading.worker_buffer))
    return iter(stream)


def _shared(source: GrainPipeline, *, rows: int, partition: DataPartition, loading: Loading,
            offset: int = 0, remainder: bool = False) -> pygrain.DatasetIterator[Batch]:
    """The partition's share of `source`, in batches of `rows` records.

    A `MapDataset` is read by index, which is what `_batches` needs to cut a
    share and to start it at a record offset. A pipeline read as it comes has
    neither, so the caller's function builds the share's own and it is
    batched where it is. Only the indexed branch is opened at an offset,
    since only it has a position to resume.
    """
    if isinstance(source, pygrain.MapDataset):
        return _batches(source, rows=rows, partition=partition, loading=loading, offset=offset,
                        remainder=remainder)
    batches = source(partition).batch(rows, drop_remainder=not remainder)
    return iter(batches.map(_Filled(rows)) if remainder else batches)


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


_UNSTACKED = "Expected all input elements to have the same structure"
"""How grain's batching starts the error it raises when records' fields do
not stack. A grain that words it otherwise raises its own error unchanged."""


def stacked(reads: Iterator[Batch]) -> Batch:
    """The next batch of `reads`, with grain's failure to stack its records
    said in terms of what to change.

    The usual cause is a field of varying length, as token ids are before
    anything cuts or packs them, and grain's message names the batch's
    structure rather than the field or the remedy.
    """
    try:
        return next(reads)
    except ValueError as failed:
        if not str(failed).startswith(_UNSTACKED):
            raise
        raise ValueError(
            "the records of one batch hold a field in different shapes, and a batch "
            "stacks each field into one array: cut or pad a variable-length field, such "
            "as token ids, to one length, or pack documents into fixed windows (Training "
            "data, Packing). Grain's report of the shapes is the cause below") from failed


class GlobalStream:
    """Reads a training stream whose saved position is one global record count.

    The stream is an endless shuffled order over the whole corpus, and each
    step is cut from it here. Global batch k is records
    [k * batch, (k + 1) * batch) of that order, and share p of n reads every
    nth of them starting at p. The position is then the number of records the
    run has consumed: one number, the same on every reader, tied to no share.
    A checkpoint written by two processes resumes on one process or four, on
    the same records in the same steps.

    `open_at(offset)` starts the share's read at a record offset. A restore
    uses it instead of replaying the stream, since an offset is just a slice
    bound. It is called on the first batch and again after `set_state`, so a
    stream restored before it is read does not start its workers twice.

    The offset alone would resume that many records into whatever order the
    run now has, so `order` is saved with it and a restore whose order
    disagrees is refused. `dew.position` defines the format that both this
    class and `dew.checkpoints` read.
    """

    def __init__(self, open_at: Callable[[int], pygrain.DatasetIterator[Batch]],
                 batch: int, order: str, loading: Loading):
        self._open = open_at
        self._batch = batch
        self._order = order
        self._records = 0
        self._reads: pygrain.DatasetIterator[Batch] | None = None
        self._loading = loading
        self.stop_seconds = loading.stop_seconds

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        if self._reads is None:
            self._reads = self._open(self._records)
        batch = stacked(self._reads)
        # Counted after the batch: a step the stream did not deliver is not
        # a step a resume may skip.
        self._records += self._batch
        return batch

    def get_state(self) -> bytes:
        return position.encode(position.Global(records=self._records, order=self._order))

    @property
    def order(self) -> str:
        """The order's description, which a saved position is compared against."""
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
            with self._loading.announced_stop():
                reads.close()


class PhasedStream:
    """Reads one global order per phase, switching at step boundaries.

    Each phase is a `GlobalStream` factory (`train_stream`, `mixed_stream`)
    with the global record count it ends at, or None for the last phase, which
    runs on. Phase k starts its own order at its own record zero when the
    run's count reaches the end of phase k - 1, so the records a step reads
    depend only on the step and the phase list, at any process count.

    The saved position is the run's record count, the current phase's order,
    and each completed phase's order with its end (`position.Global`). A
    restore checks the completed phases and the current phase's order against
    this list, and nothing after them. So a run may append phases, or move a
    boundary it has not reached yet, and still resume where it stopped. A
    one-order run's position is phase 0 with no completed phases, so a run that
    trained on one mixture resumes into a phase list that starts with it.
    Changing a phase the run has already read, or ending the current phase
    before the records the run already read in it, is refused.
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
        # The stream's own count, which it advances only for a batch it
        # delivered; its checkpoint envelope is for checkpoints.
        before = stream._records
        batch = next(stream)
        read = stream._records - before
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
    """A batch stream that can resume where it stopped; this is what a batch ramp wraps."""

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

    It wraps a stream whose position is a global record count (a
    `GlobalStream`, or `tokenized` over one) and decides only how many of its
    records each step reads. Reads run one whole final-size batch ahead, and
    each step is cut from a rolling buffer, as in MaxText's loader. A step
    therefore reads the records the run would have read without the ramp, in
    the same order, and a stage change neither skips a record nor reads one
    twice.

    Records left in the buffer have not been trained on and are not counted in
    the position, which is the number of records handed out so far. A restore
    gives the source that count, so it reopens its read there.
    """
    # MaxText reference: common/data_loader.py:122-176.

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
        """The batch this stream reads at each stage, and where each stage begins.

        The trainer reads them from the stream it opened. One compiled step per
        stage covers every shape a ramped run uses, and the mesh has to divide
        each of them.
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
    """Return an endless shuffled stream over `source`, `batch` records a global step.

    The order is the corpus reshuffled from `seed` every epoch, endlessly, and
    the reader's share is cut from it. Global batch k is then the same records
    at every partition, and a `GlobalStream` position is a record count rather
    than a shard offset.

    `operations` run after the order and before the slice. So they run inside
    the workers, a record's rng is keyed by its place in the endless stream,
    and what a record turns into depends on neither the share count nor the worker
    count.

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

        return GlobalStream(open_at, batch, order, loading)

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
    host than on a pod. A part-full batch cannot be sharded over a device
    mesh, so the last one is filled out to the batch's rows with repeats
    that `VALID_ROWS` marks, and the pass still counts every record once.
    """
    def stream(partition: DataPartition) -> Iterator[Batch]:
        records = pygrain.MapDataset.source(source).seed(seed).apply(list(transformations))
        return _batches(records, rows=partition.rows(batch), partition=partition, loading=loading,
                        remainder=True)

    return stream


__all__ = [
    "Budgeted",
    "Checkpointable",
    "Closeable",
    "Columns",
    "Corpus",
    "DataPartition",
    "DataPhase",
    "Dataset",
    "DatasetSpec",
    "Forwarding",
    "GlobalStream",
    "Loading",
    "PhasedStream",
    "Ramp",
    "RampedStream",
    "Records",
    "Resumable",
    "SourceSlice",
    "Stage",
    "Stoppable",
    "mapped",
    "tokenized",
    "train_stream",
]
