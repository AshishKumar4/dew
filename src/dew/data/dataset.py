"""The compiled `Loader`, shared reader partitions and checkpoint contracts.

Every stream is opened for a `DataPartition`, the share of each global batch
its reader reads. The trainer gets it from the mesh (`DataPartition.of`),
because processes that a pipeline or a split sequence spans hold the same rows
and so read the same share. A loader reads the share it is given and nothing
else.

An indexed global order resumes from a record count across process counts.
Unknown-length, rejecting and token-budget orders instead save Grain's
per-share state and refuse changed topology before a reader opens.
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
import math
import os
import threading
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import grain.python as pygrain
import jax
import numpy as np
import tyro
from absl import flags
from grain._src.python.dataset.transformations.batch import make_batch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew import position
from dew.nn.sharding import BATCH_AXES, LayoutRefused

# `Batch` lives in dew.objectives.base. The data layer imports it from here
# so a dataset module needs one import for the value and its shape.
from dew.objectives.base import VALID_ROWS, Batch

_log = logging.getLogger(__name__)

# Grain's worker processes read absl flags, which a script without absl.app never
# parsed; importing Dew's data layer marks them parsed, as `absl.app` would.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()

if TYPE_CHECKING:
    from dew.data.mixing import CorpusCounts
    from dew.data.pipeline import DeviceStep
    from dew.data.plan import Plan




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
    row_multiple: int = 1
    """Number of distinct row shards this reader feeds within its host."""

    def __post_init__(self):
        if not 0 <= self.index < self.count or not 0 <= self.reader < self.readers:
            raise ValueError(
                f"a data partition is share index of count, 0 <= index < count, read "
                f"by one or more readers, of which this is reader 0 <= reader < readers; "
                f"got index {self.index} of {self.count} read by {self.readers}, reader "
                f"{self.reader}"
            )

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
                f"one for each group of processes that reads its own rows"
            )
        rows = batch // self.count
        if rows % self.row_multiple:
            raise ValueError(f"batch {batch} must divide across {self.count * self.row_multiple} row shards")
        return rows


@functools.cache
def _partition(mesh: Mesh) -> DataPartition:
    shards = math.prod(mesh.shape[axis] for axis in BATCH_AXES)
    held: dict[int, set[int]] = {}
    placement = NamedSharding(mesh, P(BATCH_AXES)).devices_indices_map((shards,))
    for device, index in placement.items():
        held.setdefault(device.process_index, set()).add(index[0].start or 0)
    groups = sorted({frozenset(rows) for rows in held.values()}, key=min)
    if sum(len(group) for group in groups) != shards or len({len(group) for group in groups}) != 1:
        raise LayoutRefused(
            f"the processes of this mesh hold the row shards "
            f"{ {process: sorted(rows) for process, rows in sorted(held.items())} }, "
            f"which overlap without being the same; each group of processes has to "
            f"hold rows no other group holds, so it can read them as its own share"
        )
    mine = frozenset(held[jax.process_index()])
    readers = sorted(process for process, rows in held.items() if frozenset(rows) == mine)
    return DataPartition(
        index=groups.index(mine),
        count=len(groups),
        readers=len(readers),
        reader=readers.index(jax.process_index()),
        row_multiple=len(mine),
    )


type Reader = Callable[[DataPartition], Iterator[Batch]]
"""Opens a fresh iterator over one share of every global batch, the share the
partition names. The iterator is its caller's to close."""


@runtime_checkable
class Records(Protocol):
    """Returns records by index, which is how the loaders read a source.

    A record is one example's fields, in the shape `Batch` names, or the
    packed bytes an ArrayRecord file holds, which the spec's own transform
    unpacks into fields. A record may be missing (None). A memory read and a
    cache leave a missing record out and count it, records held in memory
    leave it out of an order batched by index, and a stream skips it; any
    other source read in an order batched by index must hold every record.

    Grain's own `RandomAccessDataSource` declares the same two methods with the
    record as a type parameter. That parameter is invariant, so a source
    declared through it cannot be passed to a loader that names a different
    record type. This protocol names what the loaders actually read.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> Batch | bytes | None: ...


type Indexed = Records | Sequence[Batch]
"""Records read by index: a source that answers grain's two methods, or a
plain sequence of them. A spec that lists its records in memory hands over
the sequence, the way the video specs list their clips."""


def record_argument[Value](annotation: type[Value]) -> tyro.constructors.PrimitiveConstructorSpec[Value]:
    """A field typed `annotation` written as one JSON value on the command
    line, the record a run holds: a tuple of records as a list,
    `[{"field": ...}, ...]`, and a mapping of class records as an
    object, `{"sigma": {"class": "linear", "fields": {...}}}`, since a flag per
    field cannot spell a collection whose length or keys the command line
    decides.

    The value is read with `from_record` and written with `to_record`, so a
    flag, a default and `run.json` hold the same record. Use it as the
    field's metadata: `Annotated[Mapping[str, ScheduleSpec],
    record_argument(Mapping[str, ScheduleSpec])]`."""
    from dew.registry import from_record, to_record

    container = typing.get_origin(annotation) or annotation
    if container is Mapping:
        container = dict
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda given: from_record(annotation, json.loads(given[0]), dtypes=False),
        is_instance=lambda given: isinstance(given, container),
        str_from_instance=lambda given: [json.dumps(to_record(given, annotation))],
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
    """Forwards a stream wrapper's stop signal, stop budget, close, identity and endlessness to its source.

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

    def identity(self) -> str:
        return stream_identity(self._forwarded())

    @property
    def endless(self) -> bool:
        return stream_endless(self._forwarded())


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
    - `threads`: reads one worker keeps in flight;
    - `read_buffer`: reads one worker holds ahead;
    - `worker_buffer`: whole batches one worker holds ready for the training
      process.

    `threads` and `read_buffer` count the elements a read produces: a whole
    batch where the order is indexed, which is cut into batches by index
    before any reader starts, as Grain's own indexed pipelines are, and a
    record where the order streams and is batched as it arrives.

    The defaults, 16 reads in flight and 16 held ahead, keep enough reads in
    flight for a remote store's latency while holding few batches. On an
    8-core armada host with two workers, Dew read 128-pixel images at 0.91 to
    1.04 times plain Grain under the same settings, within the run-to-run
    noise of parity, and held about 32 batches, 50 MB for 32 such images, per
    worker. 64 and 128 buffered 300 MB per worker and matched Grain only
    because Grain itself slowed down there.

    Zero workers is Grain's own default: threads in the training process read
    the records. Each worker process imports the program again, so it costs
    seconds and a process's memory before the first batch, and pays off only
    once decoding or augmentation is slower than the threads. On two cores of
    a shared workstation, the first batch of a 100-record pipeline took 17.8 s
    and 7.07 GiB resident across 33 processes with 32 workers, against under
    0.1 s and 0.23 GiB with none.
    """

    workers: int | Literal["auto"] = 0
    threads: int = 16
    read_buffer: int = 16
    worker_buffer: int = 2
    ram_budget_mb: int | None = None
    max_workers: int | None = None

    def resolved(self, records: pygrain.IterDataset) -> Loading:
        """Ask Grain to size workers and buffers within the host's RAM limit."""
        if self.workers != "auto":
            return self
        import grain.experimental as gx
        import psutil

        available = psutil.virtual_memory().available
        maximum, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
        if maximum.is_file() and current.is_file():
            limit = maximum.read_text().strip()
            if limit != "max":
                available = min(available, max(0, int(limit) - int(current.read_text())))
        budget = self.ram_budget_mb if self.ram_budget_mb is not None else max(1, available // (2 * 1024**2))
        try:
            picked = gx.pick_performance_config(
                records,
                ram_budget_mb=budget,
                max_workers=self.max_workers or os.cpu_count() or 1,
                max_buffer_size=self.read_buffer,
            )
        except StopIteration:
            _log.warning("Grain autotune found no element; using explicit Loading fallback (0 workers)")
            return dataclasses.replace(self, workers=0)
        workers = 0 if picked.multiprocessing_options is None else picked.multiprocessing_options.num_workers
        buffers = (
            self.read_buffer if picked.read_options is None else picked.read_options.prefetch_buffer_size
        )
        return dataclasses.replace(self, workers=workers, read_buffer=buffers)

    @property
    def worker_count(self) -> int:
        if self.workers == "auto":
            raise ValueError("automatic Loading must be resolved on its Grain pipeline")
        return self.workers

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
        return float(
            _grain_kill_seconds()
            * (1 + (self.max_workers or os.cpu_count() or 1) if self.workers == "auto" else 1 + self.workers)
        )

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
        if not self.workers or self.workers == "auto":
            return None
        line = (
            f"waiting for {self.workers} grain workers to stop, "
            f"up to {self.workers * _grain_kill_seconds()} s"
        )
        timer = threading.Timer(_QUIET_STOP_SECONDS, _log.warning, (line,))
        timer.daemon = True
        try:
            timer.start()
        except RuntimeError as refused:
            _log.warning("stopping %s grain workers without a progress line: %s", self.workers, refused)
            return None
        return timer


def _data_statistics() -> Mapping[str, float]:
    return {}


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
                f"positive"
            )
        if final <= self.start:
            raise ValueError(
                f"a batch ramp grows into the run's batch: it starts at "
                f"{self.start} and the dataset's batch is {final}, so there is "
                f"nothing to ramp"
            )
        if (final - self.start) % self.increment:
            raise ValueError(
                f"a batch ramp reaches the run's batch in whole increments: "
                f"{final} - {self.start} is not a multiple of {self.increment}"
            )
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
class Loader:
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

    Each factory call returns a fresh iterator, and the code that opened it
    owns closing it. Close it after use when it is `Closeable`; every wrapper
    around a stream (host steps, ramps, phases, the readers) forwards `close`
    to the stream it wraps. Never close the shared dataset or its backing
    store. A batch an iterator yields stays valid until its consumer asks for
    the next one. A consumer that keeps batches longer copies them, since a
    batch made by worker processes lives in shared memory and holds a
    descriptor for as long as it is kept. A source's optional `request_stop`
    is a separate, thread-safe signal; it does not make it safe to call the
    final `close` while another thread is still iterating.

    Image and video fields are uint8 in [0, 255], text is the tokenized
    `{"input_ids", "attention_mask"}` dict under "text", and a token window
    is int32 ids under "text".

    Whether a run resumes its data position depends on the iterator.
    Grain-backed iterators have `get_state` and `set_state`, and host
    transforms forward the pair. A stream with neither trains, and a resume
    of its checkpoint is refused, unless its reader declares that it reads
    the same batches each time it opens (`replayed`), which positions it by
    replay. An indexed global-order position resumes on any
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
    counts: Mapping[str, CorpusCounts] = dataclasses.field(default_factory=dict)
    validation_counts: Mapping[str, CorpusCounts] = dataclasses.field(default_factory=dict)
    plan: Plan | None = None
    statistics: Callable[[], Mapping[str, float]] = dataclasses.field(default=_data_statistics, repr=False)
    batch_tokens: int | None = None
    device: tuple[DeviceStep, ...] = ()
    """The training batch's steps the compiled training step runs, on the
    accelerator, after the reader delivers it (`DeviceStep`)."""

    @property
    def steps_per_epoch(self) -> int | None:
        if self.records is None:
            return None
        if self.ramp is None:
            return self.records // self.batch
        return self.ramp.steps_for(self.records, self.batch)

    def epoch_steps(self, epochs: int = 1) -> int:
        """Return the number of steps that `epochs` passes over the records take.

        A batch ramp grows the batch over the run's first records once, so its
        smaller early steps count once across all the passes, not once a pass.
        A stream without a record count has no epoch, so a run over it must
        give its length in steps instead.
        """
        if self.records is None:
            raise ValueError(
                "epochs need a dataset with a record count; this one streams "
                "without one, so give the run length in steps"
            )
        records = epochs * self.records
        return records // self.batch if self.ramp is None else self.ramp.steps_for(records, self.batch)


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

    Grain's iterators satisfy this, and so does `GlobalStream`, and so does
    a stream a reader declares to replay (`replayed`). The state itself is
    opaque here; `dew.position` tells which of its two kinds a checkpoint
    holds.
    """

    def get_state(self) -> Position: ...

    def set_state(self, state: Position) -> None: ...


@runtime_checkable
class Identified(Protocol):
    """A training stream that names the records it reads and says whether it
    ever runs out.

    Each process of a pool opens its own stream, over its own copy of the
    corpus, and `Trainer.fit` refuses a pool whose streams name different
    records before any process reads one. It agrees every read of a stream
    that may run out, so a process whose stream ran out stops its peers
    before they step without it; an endless one is agreed at its first read.
    """

    endless: bool

    def identity(self) -> str: ...


type _Held = Mapping[str, _Held] | Sequence[_Held] | str | bytes | np.ndarray | np.generic | float | None
"""What a batch's fields hold, as `batch_digest` reads them."""


def batch_digest(batch: Batch) -> str:
    """A digest of what `batch` holds: its fields by name, their values'
    dtypes, shapes and every element, its strings, bytes and scalars, and
    the type of any other object."""
    digest = hashlib.blake2b(digest_size=16)

    def visit(value: _Held) -> None:
        if isinstance(value, Mapping):
            digest.update(b"{%d" % len(value))
            for key in sorted(value, key=repr):
                digest.update(repr(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"[%d" % len(value))
            for element in value:
                visit(element)
        elif isinstance(value, str):
            digest.update(b"s" + value.encode())
        elif isinstance(value, bytes):
            digest.update(b"b" + value)
        elif value is None or isinstance(value, (bool, int, float, complex)):
            digest.update(b"v" + repr(value).encode())
        else:
            array = np.asarray(value)
            digest.update(f"{array.dtype}{array.shape}".encode())
            if array.dtype != object:
                digest.update(np.ascontiguousarray(array).data)
            elif array.ndim:
                for element in array.flat:
                    visit(element)
            else:
                # An object NumPy holds whole, whose text may name its address: its type alone.
                digest.update(type(value).__qualname__.encode())

    visit(batch)
    return digest.hexdigest()


class Replayed(Forwarding):
    """The batches of a stream declared to read the same batches each time
    `reopen()` opens it, positioned by how many it has read and a digest of
    the last.

    A resume reopens the stream and reads that many again. The digest of the
    batch at the position is a guard, not a proof: a stream whose batch there
    differs is refused, and one that agrees there and differs elsewhere, one
    drawing on a process-wide RNG or the clock after that batch, passes and
    resumes on other data. Hashing every batch costs the reader about a
    millisecond a megabyte."""

    def __init__(self, reopen: Callable[[], Iterator[Batch]], source: Iterator[Batch] | None = None):
        self._reopen = reopen
        self._source: Iterator[Batch] | None = iter(reopen()) if source is None else source
        self._read, self._last = 0, None

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        if self._source is None:
            raise StopIteration
        batch = next(self._source)
        self._read += 1
        self._last = batch_digest(batch)
        return batch

    def get_state(self) -> dict[str, int | str | None]:
        return {"read": self._read, "last": self._last}

    def set_state(self, state: Mapping[str, object]) -> None:
        read, last = state.get("read"), state.get("last")
        if not isinstance(read, int) or read < 0 or (read and not isinstance(last, str)):
            raise ValueError(f"a replayed stream's position is the count it read and the last one's "
                             f"digest, not {dict(state)!r}")
        self.close()
        self._source, self._read, self._last = iter(self._reopen()), 0, None
        if read:
            _log.info("replaying %d batches of %s to resume", read, stream_identity(self._source))
        for skipped in range(read - 1):
            try:
                next(self._source)
            except StopIteration:
                raise ValueError(f"the stream ran out after {skipped} batches, short of the {read} "
                                 f"a resume reads again") from None
            self._read += 1
        if read:
            next(self)
            if self._last != last:
                raise ValueError(
                    f"this stream does not replay deterministically: its batch {read} reads back "
                    f"other data than before the checkpoint. Make it Checkpointable "
                    f"(get_state and set_state) or read it through dew.data")


def replayed(reader: Reader) -> Reader:
    """`reader`, declared to read the same batches each time it opens, as a
    script restarted from the top reads them: its streams are positioned by
    replay (`Replayed`), so a run over it resumes where it stopped."""

    def opened(partition: DataPartition) -> Iterator[Batch]:
        return Replayed(lambda: reader(partition))

    return opened


def stream_identity(stream: Iterator[Batch] | None) -> str:
    """What `stream` reads, as `Identified` names it, or else its type."""
    return stream.identity() if isinstance(stream, Identified) else type(stream).__name__


def stream_endless(stream: Iterator[Batch] | None) -> bool:
    """Whether `stream` says it never runs out (`Identified`), as Dew's
    global streams do."""
    return isinstance(stream, Identified) and stream.endless


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
        return f"SourceSlice({describe(self.source)}, start={self.start}, length={self.length})"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source[self.start + index]


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


def weight_counts(shares: Sequence[float], records: int) -> tuple[int, ...]:
    """Grain's proportional selection counts over a prefix of an indexed mixture."""
    scale = 100 / min(shares)
    proportions = [int(share * scale) for share in shares]
    counts, remaining, left = [], sum(proportions), records
    for proportion in proportions:
        rest = left * (remaining - proportion) // remaining
        counts.append(left - rest)
        left, remaining = rest, remaining - proportion
    return tuple(counts)


class FilledBatch(pygrain.MapTransform):
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
        indices = plan.pad(np.arange(held))
        return {**take_batch(element, indices), VALID_ROWS: ~plan.padding & self._real}


def rows_of(batch: Mapping[str, object]) -> int:
    """The records `batch` holds, read off its first field that has rows.

    A batch may carry a field that is one value for the whole step rather
    than one per record, and that field says nothing about how many records
    there are; a batch of such fields alone holds none. Any mapping of
    fields answers, so the trainer's own batch type reaches this as it is.
    """
    leaves = jax.tree.leaves(batch, is_leaf=lambda column: isinstance(column, list))
    return next((len(leaf) if isinstance(leaf, list) else np.shape(leaf)[0]
                 for leaf in leaves if isinstance(leaf, list) or np.shape(leaf)), 0)


def take_batch(batch: Batch, indices: int | slice | np.ndarray) -> Batch:
    """Rows selected along array axes and opaque lists of object records."""
    return jax.tree.map(
        lambda column: ([column[int(index)] for index in indices]
                        if isinstance(column, list) and isinstance(indices, np.ndarray)
                        else column[indices]),
        batch, is_leaf=lambda column: isinstance(column, list),
    )


_UNSTACKED = "Expected all input elements to have the same structure"
"""How grain's batching starts the error it raises when records' fields do
not stack. A grain that words it otherwise raises its own error unchanged."""


def _stacked(value) -> bool:
    """Whether Grain stacks `value` into one array: numbers, strings and bytes,
    and mappings of them; not lists, object arrays or other records."""
    if isinstance(value, Mapping):
        return bool(value) and all(_stacked(field) for field in value.values())
    return (isinstance(value, (np.ndarray, np.generic, jax.Array, int, float, complex, str, bytes))
            and np.asarray(value).dtype.kind in "biufcSU")


def record_batch(rows: Sequence[Batch], *, shared: bool = False) -> Batch:
    """Numeric and text columns stacked by Grain, with object columns (a
    conversation's messages, a record's tools) kept as record lists."""
    if not rows:
        raise ValueError("a batch needs at least one record")
    if any(row.keys() != rows[0].keys() for row in rows[1:]):
        raise ValueError("the records of one batch must hold the same fields")
    batched: Batch = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        batched[name] = (make_batch(values, output_to_shared_memory=shared)
                         if all(_stacked(value) for value in values) else values)
    return batched


def stacked(rows: Sequence[Batch], stack: Callable[[Sequence[Batch]], Batch]) -> Batch:
    """`rows` stacked into one batch by `stack`, with grain's failure to
    stack them said in terms of what to change. Every batch Dew makes of
    rows is stacked here, so a reader and the loading tuner that samples it
    refuse alike.

    The usual cause is a field of varying length, as token ids are before
    anything cuts or packs them, and grain's message names the batch's
    structure rather than the field or the remedy.
    """
    try:
        return stack(rows)
    except ValueError as failed:
        if not str(failed).startswith(_UNSTACKED):
            raise
        raise ValueError(
            "the records of one batch hold a field in different shapes, and a batch "
            "stacks each field into one array: cut or pad a variable-length field, such "
            "as token ids, to one length, or pack documents into fixed windows (Training "
            "data, Packing). Grain's report of the shapes is the cause below"
        ) from failed


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
    run now has, so the plan's order (`order`) and the plan are saved with
    it and a restore whose plan disagrees is refused (`Plan.check`).
    `dew.position` defines the format that both this class and
    `dew.checkpoints` read.
    """

    endless = True

    def __init__(
        self,
        open_at: Callable[[int], pygrain.DatasetIterator[Batch]],
        batch: int,
        plan: Plan,
        loading: Loading,
        *,
        resume: Literal["exact", "non-exact"] = "exact",
        endless: bool = True,
    ):
        self._open = open_at
        self.endless = endless
        self._batch = batch
        self._order = json.dumps(plan.order, sort_keys=True)
        self._records = 0
        self._reads: pygrain.DatasetIterator[Batch] | None = None
        self._loading = loading
        self._plan = plan
        self._resume: Literal["exact", "non-exact"] = resume
        self.manifest_changes: tuple[str, ...] = ()
        self.stop_seconds = loading.stop_seconds

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
        return position.encode(
            position.Global(
                records=self._records,
                order=self._order,
                plan=self._plan.record(),
            )
        )

    @property
    def records(self) -> int:
        return self._records

    @property
    def order(self) -> str:
        """The order's description, which a saved position is compared against."""
        return self._order

    def identity(self) -> str:
        """The plan's identity (`Plan.identity`), which a pool agrees before it
        reads: hosts whose copies of a corpus hold other records, or whose
        steps make other values, are refused however the sources describe
        themselves."""
        return self._plan.identity()

    def set_state(self, state: bytes) -> None:
        saved = position.read(state)
        if saved.plan is not None:
            self.manifest_changes = self._plan.check(saved.plan, resume=self._resume)
        if saved.completed:
            raise ValueError(
                f"the saved data position is {saved.records} records into a phased run "
                f"past {len(saved.completed)} phase(s), and this run reads one order "
                f"({self._order}); resume it with the phases that wrote it"
            )
        from dew.data.plan import same_order

        if not same_order(saved.order, self._order):
            raise ValueError(
                f"the saved data position is {saved.records} records into "
                f"{saved.order}, and this run reads {self._order}; a record count "
                f"is a place in one order and another place in another, so resume "
                f"the corpus, record count and seed the checkpoint was written with"
            )
        self.close()
        self._records = saved.records

    def close(self) -> None:
        reads, self._reads = self._reads, None
        if reads is not None:
            with self._loading.announced_stop():
                reads.close()


@runtime_checkable
class GlobalReader(Protocol):
    @property
    def records(self) -> int: ...
    @property
    def order(self) -> str: ...
    def __next__(self) -> Batch: ...
    def get_state(self) -> Position: ...
    def set_state(self, state: bytes) -> None: ...
    def close(self) -> None: ...
    def identity(self) -> str: ...


class PhasedStream:
    """Reads one global order per phase, switching at step boundaries.

    Each phase opens a `GlobalReader`, such as a compiled global order, and
    has the global record count it ends at, or None for the last phase, which
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

    endless = True

    def __init__(self, phases: Sequence[tuple[Callable[[], GlobalReader], int | None]], stop_seconds: float):
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
            stream.set_state(
                position.encode(
                    position.Global(records=self._records - self._start(phase), order=stream.order)
                )
            )
            self._current = phase
        stream = self._streams[phase]
        # The stream's own count, which it advances only for a batch it
        # delivered; its checkpoint envelope is for checkpoints.
        before = stream.records
        batch = next(stream)
        read = stream.records - before
        self._records += read
        if phase < len(self._ends) and self._records > self._ends[phase]:
            raise ValueError(
                f"a step of {read} records crossed the phase boundary at record "
                f"{self._ends[phase]}; phases end on step boundaries"
            )
        return batch

    def get_state(self) -> bytes:
        # The phase of the last record read: at a boundary nothing of the next
        # phase is read yet, so the finished one is still current and a
        # resume may change what follows it or extend it.
        phase = bisect.bisect_left(self._ends, self._records)
        plans = [
            position.read(_global_position(stream.get_state())).plan for stream in self._streams[: phase + 1]
        ]
        return position.encode(
            position.Global(
                records=self._records,
                order=self._streams[phase].order,
                completed=tuple((self._streams[index].order, self._ends[index]) for index in range(phase)),
                plan={"phases": plans} if any(plan is not None for plan in plans) else None,
            )
        )

    def set_state(self, state: bytes) -> None:
        saved = position.read(state)
        if saved.plan is not None:
            plans = saved.plan.get("phases") if "phases" in saved.plan else [saved.plan]
            if not isinstance(plans, list) or len(plans) != len(saved.completed) + 1:
                raise ValueError("saved phase plans do not cover the completed and current phases")
            for index, plan in enumerate(plans):
                if index >= len(self._streams) or (plan is not None and not isinstance(plan, Mapping)):
                    raise ValueError("saved phase plan is not part of this run")
                end = saved.completed[index][1] if index < len(saved.completed) else saved.records
                self._streams[index].set_state(
                    position.encode(
                        position.Global(
                            records=end - self._start(index), order=self._streams[index].order, plan=plan
                        )
                    )
                )
        from dew.data.plan import same_order

        for index, (order, end) in enumerate(saved.completed):
            if (index >= len(self._ends) or not same_order(order, self._streams[index].order)
                    or self._ends[index] != end):
                raise ValueError(
                    f"the saved run finished phase {index} reading {order} to record "
                    f"{end}, and this run's phase {index} is not that; a phase the "
                    f"run has read cannot change under it"
                )
        phase = len(saved.completed)
        if phase >= len(self._streams) or not same_order(saved.order, self._streams[phase].order):
            raise ValueError(
                f"the saved position is {saved.records} records into phase {phase}, "
                f"reading {saved.order}, and this run's phase {phase} reads something "
                f"else; resume with the order the checkpoint was written in"
            )
        if phase < len(self._ends) and saved.records > self._ends[phase]:
            raise ValueError(
                f"the saved run has read {saved.records} records, past this run's end "
                f"of phase {phase} at {self._ends[phase]}"
            )
        self.close()
        self._records = saved.records

    def identity(self) -> str:
        return "; ".join(
            [*(stream.identity() for stream in self._streams), f"phases ending at records {list(self._ends)}"]
        )

    def close(self) -> None:
        if self._current is not None:
            self._streams[self._current].close()
        self._current = None


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
            f"reports {type(state).__name__}, which counts one process's shard"
        )
    return state


class RampedStream(Forwarding):
    """Grows a training stream's step over the run's first records.

    It wraps a stream whose position is a global record count (a
    `GlobalStream` or a host transform over one) and decides only how many of its
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
            self._held = (
                read
                if self._held is None
                else jax.tree.map(lambda kept, new: kept + new if isinstance(kept, list)
                                  else np.concatenate([kept, new]), self._held, read,
                                  is_leaf=lambda column: isinstance(column, list))
            )
        step = take_batch(self._held, slice(rows))
        self._held = take_batch(self._held, slice(rows, None))
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
                f"was written by a run that ramped differently"
            )

    def close(self) -> None:
        self._held = None
        super().close()


def ramped(dataset: Loader, ramp: Ramp) -> Loader:
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
                "cuts other steps; ramp a run of one order"
            )
        if isinstance(stream, Resumable):
            state = stream.get_state()
            if isinstance(state, bytes) and position.translates(state):
                return RampedStream(stream, stages, partition)
        if isinstance(stream, Closeable):
            stream.close()
        raise TypeError(
            f"a batch ramp cuts the step out of a global record order, and "
            f"{type(stream).__name__} hands over batches it has cut itself; ramp "
            f"a dataset compiled to a global record order"
        )

    return dataclasses.replace(dataset, train=train, ramp=ramp)


__all__ = [
    "Budgeted",
    "Checkpointable",
    "Closeable",
    "DataPartition",
    "DataPhase",
    "Forwarding",
    "GlobalStream",
    "Identified",
    "Loader",
    "Loading",
    "PhasedStream",
    "Ramp",
    "RampedStream",
    "Reader",
    "Records",
    "Resumable",
    "SourceSlice",
    "Stage",
    "Stoppable",
    "mapped",
    "record_argument",
]
