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
import itertools
import json
import logging
import math
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import grain.python as pygrain
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew import position
from dew.batch import VALID_ROWS, Batch, padded_rows
from dew.mesh import LayoutRefused, batch_axes

_log = logging.getLogger(__name__)

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
    send its batch to the others (`dew.distributed.batches.first_reader_batch`).
    """
    reader: int = 0
    """This process's place among the share's readers, in process order (0 for the first)."""
    row_multiple: int = 1
    """Number of distinct row shards this reader feeds within its host."""
    window: int = 1
    """The steps an update pools (`Trainer`'s accumulation): a batch ramp changes stage only where an
    update begins (`Ramp.stages`). It cuts the stream into steps and names no data, so it is no part
    of what a share reads."""

    def __post_init__(self):
        if self.window < 1:
            raise ValueError(f"a data partition's window is the steps an update pools, at least 1; got "
                             f"{self.window}")
        if not 0 <= self.index < self.count or not 0 <= self.reader < self.readers:
            raise ValueError(
                f"a data partition is share index of count, 0 <= index < count, read "
                f"by one or more readers, of which this is reader 0 <= reader < readers; "
                f"got index {self.index} of {self.count} read by {self.readers}, reader "
                f"{self.reader}"
            )

    @classmethod
    def of(cls, mesh: Mesh, window: int = 1) -> DataPartition:
        """Return the share of every global batch this process reads on `mesh`.

        A batch's rows split over the row axes and no others: those the rules
        in context give `activation_batch` (`batch_axes`). The sequence axis
        splits positions, the tensor axis splits widths, and the stage axis
        holds a pipeline's stages, so processes whose devices
        hold the same row shards need the same rows. The processes therefore
        fall into groups by the rows they hold. Each group reads one share,
        numbered by the first row shard it holds, and every process in the
        group reads it (`readers`); `reader` is this process's place among
        them, in process order.

        A hand-built device order can produce groups whose rows overlap without
        being the same rows. No share can then be read whole by each group, so
        such a mesh is refused.
        """
        return dataclasses.replace(_partition(mesh, batch_axes(mesh)), window=window)

    def held(self, rows: int, offset: int = 0) -> tuple[slice, bool]:
        """Which of `rows` rows a validation read of this share holds, and whether they are real:
        every count-th from its index, `offset` places on as a resume's count inside a batch leaves
        it (`GlobalStream`); where that holds none of rows that hold some, the first in their place,
        not real (`FilledBatch` marks them so), so every share reads a batch where any does and
        none scores another's row."""
        mine = self.cut(offset)
        if rows and not range(rows)[mine]:
            return slice(1), False
        return mine, True

    def cut(self, offset: int = 0) -> slice:
        """This share's rows of an order or a batch: every count-th from its first (`first`)."""
        return slice(self.first(offset), None, self.count)

    def first(self, offset: int = 0) -> int:
        """This share's first row of an order or a batch: its index, `offset` places on as a resume's
        count inside a batch leaves it (`GlobalStream`)."""
        return (self.index + offset) % self.count

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
def _partition(mesh: Mesh, rows: tuple[str, ...]) -> DataPartition:
    shards = math.prod(mesh.shape[axis] for axis in rows)
    held: dict[int, set[int]] = {}
    placement = NamedSharding(mesh, P(rows or None)).devices_indices_map((shards,))
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


GRAIN_KILL_SECONDS = 25
"""How long Grain waits for a stopped worker process to exit before it kills it, which it publishes
no setting or reading of: 25 s in grain 0.2.18 (process_prefetch.py), the release tested."""


WORKER_BUFFER = 2
"""The batches each worker process reads ahead where none is asked for (`Loading`, `Pipeline.workers`)."""


WORKER_PROCESS_BYTES = 219 * 2**20
"""The memory a loading worker's process takes before it holds a batch: each imports the program
again. 32 workers of a 100-record pipeline held 7.07 GiB resident against 0.23 GiB with none, on
two cores of a shared workstation (`Loading`), 219 MiB each."""


SAMPLED_BATCHES = 5
"""The batches automatic loading measures to size its buffers (`Loading.resolved`), as many as
Grain's own pick samples."""


@dataclasses.dataclass(frozen=True)
class Host:
    """What automatic loading sizes its workers and buffers by (`Loading.resolved`): the CPUs this
    process may run on, its CPU affinity bounded by its cgroup's quota, and the bytes of memory it
    may still take, the host's available memory bounded by what its cgroup's limit leaves, so a
    container's limits size it, not the machine's."""

    cpus: int
    memory: int

    @classmethod
    def current(cls) -> Host:
        import psutil

        cpus = len(os.sched_getaffinity(0))
        quota = Path("/sys/fs/cgroup/cpu.max")
        if quota.is_file():
            limit, period = quota.read_text().split()
            if limit != "max":
                cpus = min(cpus, math.ceil(int(limit) / int(period)))
        memory = psutil.virtual_memory().available
        maximum, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
        if maximum.is_file() and current.is_file():
            limit = maximum.read_text().strip()
            if limit != "max":
                memory = min(memory, max(0, int(limit) - int(current.read_text())))
        return cls(cpus, memory)


def batch_bytes(batch: Batch) -> int:
    """The bytes of a batch's leaves, as arrays hold them."""
    return sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(batch))


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
    worker_buffer: int = WORKER_BUFFER
    ram_budget_mb: int | None = None
    max_workers: int | None = None
    cache: bool = True
    """Whether a run reads its chains' leading deterministic steps from the records an earlier run
    made of the same records (`dew.data.cache.automatic_root`), which changes none of them."""

    def resolved(self, records: pygrain.IterDataset, *, reads: int, host: Host | None = None) -> Loading:
        """These settings with automatic workers and read buffer sized for `host` (this process's
        by default) from its CPUs and its memory together, as tf.data's autotuning budgets both.
        Grain's own pick (`pick_performance_config`) reads memory alone, takes the machine's CPU
        count where a container may run on a few of them, and counts one batch for each worker.

        A worker for each CPU, as tf.data's parallelism takes each schedulable one, where there is
        more than one: on one CPU a worker adds a process and no parallelism, so the reading process
        reads. On four CPUs of an eight-core host an image read in four workers read as many images
        a second as in eight, within 2%, and in three 11% fewer (armada, 2026-10-26). At most
        `max_workers` of them, and no more than half the memory the process may still take holds
        (`ram_budget_mb` where given), each a process of its own (`WORKER_PROCESS_BYTES`) holding
        its `worker_buffer` batches and `read_buffer` reads, the read buffer itself no more reads
        than that half holds. `records` are the batches the read makes without workers,
        `SAMPLED_BATCHES` of which are measured; `reads` is how many reads make one: one where the
        order is read by batch, its rows where it streams records. A read that makes no batch reads
        without workers."""
        if self.workers != "auto":
            return self
        sampled = list(itertools.islice(records, SAMPLED_BATCHES))
        if not sampled:
            return dataclasses.replace(self, workers=0)
        host = Host.current() if host is None else host
        batch = max(1, sum(batch_bytes(value) for value in sampled) // len(sampled))
        read = max(1, batch // reads)
        budget = host.memory // 2 if self.ram_budget_mb is None else self.ram_budget_mb * 2**20
        cpus = host.cpus if host.cpus > 1 else 0
        if self.max_workers is not None:
            cpus = min(cpus, self.max_workers)
        read_buffer = min(self.read_buffer, budget // read)
        held = WORKER_PROCESS_BYTES + self.worker_buffer * batch + read_buffer * read
        workers = max(0, min(cpus, budget // held))
        _log.info("automatic loading reads in %d workers holding %d reads ahead, sized for %d CPUs and "
                  "%.2f GiB", workers, read_buffer, host.cpus, budget / 2**30)
        return dataclasses.replace(self, workers=workers, read_buffer=read_buffer)

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
        # The 5.1 to 7.4 s stop was examples/sft_gemma4.py's four workers.
        return float(
            GRAIN_KILL_SECONDS
            * (1 + (self.max_workers or Host.current().cpus) if self.workers == "auto" else 1 + self.workers)
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
            f"up to {self.workers * GRAIN_KILL_SECONDS} s"
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


def _update_aligned(stages: tuple[Stage, ...], window: int) -> tuple[Stage, ...]:
    """`stages` each beginning at the first multiple of `window` steps at or after the step it begins
    at, counted from the run's first: every start is taken from the given schedule, not from the
    stage before it moved, so no stage drifts past its own end, and a stage left with no step is
    dropped. Each start's record is what the stages before it read up to it. A window is at least one
    step."""
    if window < 1:
        raise ValueError(f"a ramp aligns its stages to updates of at least one step; got a window of "
                         f"{window}")
    starts, step = [], 0
    for stage, after in itertools.pairwise(stages):
        starts.append(step)
        step += (after.records - stage.records) // stage.batch
    starts.append(step)
    moved = [-(-start // window) * window for start in starts]
    aligned, records = [], 0
    for index, (stage, start) in enumerate(zip(stages, moved, strict=True)):
        end = moved[index + 1] if index + 1 < len(stages) else None
        if end == start:
            continue
        if aligned:
            records += (start - aligned[-1][1]) * aligned[-1][0].batch
        aligned.append((Stage(batch=stage.batch, records=records), start))
    return tuple(stage for stage, _ in aligned)


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
    exact. A run whose updates pool several steps (`DataPartition.window`)
    begins each stage where an update does, at the first update boundary at or
    after the step it would begin at, as MaxText and torchtitan apply a batch
    schedule per optimizer step: no update pools two stages' batches, and a
    boundary moves by less than one update. The records stay the position, so a
    run resumed under another window cuts its stages again. Where the ramp
    ends, MaxText's loader drops what is left of its buffered batch while Dew
    keeps reading, so from then on MaxText reads later records than Dew in each
    step, although the batch sizes match.
    """

    # MaxText reference: configs/base.yml:755-765 and
    # utils/rampup_batch.py:38-50,53-101.

    start: int
    increment: int
    samples: int

    def stages(self, final: int, window: int = 1) -> tuple[Stage, ...]:
        """Return every stage of the ramp up to `final`, the run's own global batch, each beginning
        where an update of `window` steps does.

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
        return _update_aligned(tuple(stages), window)

    def steps_for(self, records: int, final: int, window: int = 1, tail: Tail = "dropped") -> int:
        """Return the number of steps a run of updates of `window` steps takes to read `records`
        records under this ramp, a last step that holds fewer than a batch counted where it is read
        (`tail`).

        Early steps read fewer records, so a pass over a corpus takes more
        steps than it would with the final batch from the start.
        """
        def whole(count: int, batch: int) -> int:
            return -(-count // batch) if tail == "read" else count // batch

        stages = self.stages(final, window)
        steps = 0
        for stage, next_stage in itertools.pairwise(stages):
            if records < next_stage.records:
                return steps + whole(records - stage.records, stage.batch)
            steps += (next_stage.records - stage.records) // stage.batch
        return steps + whole(max(records - stages[-1].records, 0), final)


LONG_RUN = 1024
"""The run of rejected records or empty shards in a row a read first logs (`logged_run`): longer
than an ordinary filter's runs or a sharded store's empty files, so a filter that keeps nothing or a
store of empty files is named from the first line. An endless read reads on through any run; none
proves the store holds nothing to keep."""


def logged_run(run: int, first: int = 1) -> bool:
    """Whether a read logs a run of `run` empty reads in a row: at `first`, then at each doubling, so a
    run that goes on is named again ever more rarely."""
    return run >= first and not run & (run - 1)


type Tail = Literal["dropped", "read"]
"""What becomes of the records an epoch leaves short of a whole batch (`Loader.tail`): `read`, where
an order that reads on carries them into the next epoch's first batch or one that keeps its last
short batch fills it; `dropped`, where a training order that ends drops them."""


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
    tail: Tail = "dropped"
    """What becomes of the records an epoch leaves short of a whole batch: read, carried into the
    next epoch by an order that reads on or kept and filled by one that keeps them, so an epoch's
    steps count their batch; or dropped, so they count none (`Tail`)."""
    order_records: int | None = None
    """The records a training read of a global order that ends hands out over all its passes, the
    same on every share, by which a batch ramp ends its last step (`RampedStream`); None where the
    order never ends or its read counts none ahead."""
    examples: int | None = None
    """The records the training read is bounded to before they are packed (`compile`'s
    `examples`), whose steps a run's `examples=` budget takes (`example_steps`); None where it is
    not bounded."""
    token_ends: tuple[int | None, ...] | None = None
    """The committed tokens each phase of a schedule ends at (`dew.data.pipeline.schedule`), the last
    where the data ends or None where it runs on, which the trainer reads its commits against and ends
    the phase at (`PhasedStream.ending`); None for a read on no token clock."""

    @property
    def steps_per_epoch(self) -> int | None:
        return None if self.records is None else self._steps(self.records)

    def _steps(self, records: int, window: int = 1) -> int:
        """The steps that read `records` records, a last short batch counted where it is read."""
        if self.ramp is None:
            return -(-records // self.batch) if self.tail == "read" else records // self.batch
        return self.ramp.steps_for(records, self.batch, window, self.tail)

    def epoch_steps(self, epochs: int = 1, window: int = 1) -> int:
        """Return the number of steps that `epochs` passes over the records take, in updates of
        `window` steps (`Ramp.stages`), refusing passes too short to fill a batch.

        A batch ramp grows the batch over the run's first records once, so its
        smaller early steps count once across all the passes, not once a pass.
        A stream without a record count has no epoch, so a run over it must
        give its length in steps instead.
        """
        if self.records is None:
            raise ValueError(
                "epochs need a dataset with a record count, and this one streams without one: online "
                "packing or a filter over an indexed or streamed read cuts its order per share. Read it "
                "whole first (.read('memory')), read a published packing generation by index, or give "
                "the run length in steps or tokens"
            )
        records = epochs * self.records
        steps = self._steps(records, window)
        if steps == 0:
            raise ValueError(f"{records} records do not fill one batch of {self.batch}, so {epochs} epochs "
                             "are no steps at all: read more data, lower the batch, or count the run in "
                             "steps")
        return steps

    def example_steps(self, examples: int, window: int = 1) -> int:
        """The steps that read the `examples` records this loader's training read is bounded to
        (`examples`), its last batch short where they end inside one, in whole updates of `window`
        steps. A global order's plan counts them ahead (`order_records`); one read in its own
        order, as a pack that runs as rows arrive makes, is counted by reading it once to its end, on
        the host. Refused where the read is bounded to another count, in a pool, whose processes
        split the short batch the examples end in unevenly, so each would end at a step of its own,
        and where the last steps would make no whole update, so some examples would train nothing."""
        if self.examples != examples:
            held = "not bounded" if self.examples is None else f"bounded to {self.examples}"
            raise ValueError(f"examples={examples} reads a training chain bounded to that many records "
                             f"before packing, and this loader's is {held}; give fit the Dataset chain, "
                             "which it bounds and compiles")
        if jax.process_count() > 1:
            raise ValueError("examples end in a short batch, which a pool's processes would split unevenly "
                             "and end at steps of their own; train them on one process or count the run "
                             "in steps")
        if self.order_records is not None:
            steps = self._steps(self.order_records, window)
        else:
            stream = self.train(DataPartition())
            try:
                steps = sum(1 for _ in stream)
            finally:
                if isinstance(stream, Closeable):
                    stream.close()
        if steps % window:
            raise ValueError(f"{examples} examples fill {steps} batches, which are not whole updates of "
                             f"{window} steps: the last {steps % window} would update nothing; change the "
                             "examples, the batch or the accumulation")
        return steps


CAPTION = "caption"
"""The batch field a captioning dataset writes its text in, before a run's
conditions read it."""


type Position = bytes | Mapping[str, object]
"""Where a stream stopped, as the stream reports it.

Dew's own envelope is bytes, and Grain reports a JSON object, which
`dew.distributed.batches` encodes before a checkpoint stores it.
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


def batch_digest(batch: Batch) -> str:
    """A digest of what `batch` holds, as the codec reads content (`content_fingerprint`): its fields
    by name, their values' dtypes, shapes and every element, a Dew dataclass's fields, its strings,
    bytes and scalars, and the type of any object the codec has no schema for."""
    # The sources read this module's batches, so their package imports it first.
    from dew.data.sources.codec import content_fingerprint

    return content_fingerprint(batch, opaque=True).hex()


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


class FilledBatch(pygrain.MapTransform):
    """A batch of fewer than `rows` records filled out with repeats of its
    own rows (`dew.batch.padded_rows`), and `VALID_ROWS` marking the real ones: those of
    its own a batch that marks them already marks real. Without `real` every
    row is a repeat."""

    def __init__(self, rows: int, *, real: bool = True):
        self._rows, self._real = rows, real

    def map(self, element: Batch) -> Batch:
        held = rows_of(element)
        if held == self._rows and self._real:
            return element
        indices, repeats = padded_rows(held, self._rows)
        valid = ~repeats & self._real
        if VALID_ROWS in element:
            valid = valid & np.asarray(element[VALID_ROWS], bool)[indices]
        return {**take_batch(element, indices), VALID_ROWS: valid}


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


def take_rows(column, indices: np.ndarray):
    """`column`'s rows at `indices`: a list or tuple of per-row records as its own type, an array
    along its leading axis."""
    if isinstance(column, list | tuple):
        return type(column)(column[int(index)] for index in indices)
    return column[indices]


def take_batch(batch: Batch, indices: int | slice | np.ndarray) -> Batch:
    """Rows selected along array axes and opaque lists of object records (`take_rows`)."""
    return jax.tree.map(
        lambda column: take_rows(column, indices) if isinstance(indices, np.ndarray) else column[indices],
        batch, is_leaf=lambda column: isinstance(column, list),
    )


class Unstacked(ValueError):
    """The records of one batch hold a field in different shapes or structures, which no array
    stacks (`record_batch`)."""


def _stacked(value) -> bool:
    """Whether Grain stacks `value` into one array: numbers, strings and bytes,
    and mappings of them; not lists, object arrays or other records."""
    if isinstance(value, Mapping):
        return bool(value) and all(_stacked(field) for field in value.values())
    return (isinstance(value, (np.ndarray, np.generic, jax.Array, int, float, complex, str, bytes))
            and np.asarray(value).dtype.kind in "biufcSU")


SHARED_BYTES = 1 << 16
"""A batch column smaller than this crosses from a loading worker pickled, not in a shared-memory
segment of its own: making, mapping, closing and unlinking a segment costs the reader more than
copying a few KiB through the worker's queue (`DatasetOptions.min_shm_size`). A batch's 32
captions, a shared segment each step, cost an image read 5 to 7% of its throughput."""


def _shareable(values: Sequence[object]) -> bool:
    """Whether a column of `values` stacks into an array of at least `SHARED_BYTES` numbers: a string
    column's width is only known once stacked, and a column of strings is seldom large."""
    leaves = jax.tree.leaves(values[0])
    return (all(np.asarray(leaf).dtype.kind not in "SU" for leaf in leaves)
            and len(values) * sum(np.asarray(leaf).nbytes for leaf in leaves) >= SHARED_BYTES)


def record_batch(rows: Sequence[Batch], *, shared: bool = False) -> Batch:
    """Numeric and text columns stacked by Grain, with object columns (a
    conversation's messages, a record's tools) kept as record lists. With
    `shared`, a column large enough to be worth it (`SHARED_BYTES`) is stacked
    into shared memory, for a loading worker to hand over without a copy."""
    if not rows:
        raise ValueError("a batch needs at least one record")
    if any(row.keys() != rows[0].keys() for row in rows[1:]):
        raise ValueError("the records of one batch must hold the same fields")
    batched: Batch = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        batched[name] = (_column(values, shared=shared and _shareable(values))
                         if all(_stacked(value) for value in values) else values)
    return batched


def _column(values: Sequence[object], *, shared: bool):
    """`values`, one field's across a batch's records, stacked along a new leading axis, a mapping
    of them field by field, as Grain batches them: into Grain's shared memory with `shared`, which
    `_shareable` gives only a numeric column."""
    if len(values) == 1:
        return jax.tree.map(lambda leaf: np.expand_dims(leaf, 0), values[0])

    from grain.multiprocessing import SharedMemoryArray

    def stack(*leaves):
        if not shared:
            return np.stack(leaves)
        first = np.asanyarray(leaves[0])
        return np.stack(leaves, out=SharedMemoryArray((len(leaves), *first.shape), dtype=first.dtype))

    try:
        return jax.tree.map(stack, *values)
    except ValueError as error:
        shapes = sorted({str(jax.tree.map(np.shape, value)) for value in values})
        raise Unstacked(f"the field's records do not stack: {'; '.join(shapes[:6])}") from error


def stacked(rows: Sequence[Batch], stack: Callable[[Sequence[Batch]], Batch]) -> Batch:
    """`rows` stacked into one batch by `stack`, with its failure to
    stack them said in terms of what to change. Every batch Dew makes of
    rows is stacked here, so a reader and the loading tuner that samples it
    refuse alike.

    The usual cause is a field of varying length, as token ids are before
    anything cuts or packs them. A batch function of the caller's own raises
    its own errors unchanged.
    """
    try:
        return stack(rows)
    except Unstacked as failed:
        raise ValueError(
            "the records of one batch hold a field in different shapes, and a batch "
            "stacks each field into one array: cut or pad a variable-length field, such "
            "as token ids, to one length, or pack documents into fixed windows (Training "
            "data, Packing). The shapes are in the cause below"
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


@dataclasses.dataclass(frozen=True)
class Ranges:
    """Consecutive ranges of `counts` items each, as an indexed store's shards hold their records,
    a run's phases its records and a ramp's stages its batches: which range holds an item
    (`locate`, or `holding` where a range that runs on follows them), where a range starts
    (`start`) and the items of them all. A range of no items holds none, so an item is never
    located in one."""

    counts: tuple[int, ...]

    @classmethod
    def ending(cls, ends: Sequence[int]) -> Ranges:
        """The ranges that end at each of `ends`, the first starting at 0."""
        return cls(tuple(end - start for start, end in itertools.pairwise((0, *ends))))

    @functools.cached_property
    def ends(self) -> tuple[int, ...]:
        return tuple(itertools.accumulate(self.counts))

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def start(self, index: int) -> int:
        """Where range `index` starts: past the last, where the range that runs on after them does."""
        return self.ends[index - 1] if index else 0

    def locate(self, position: int) -> tuple[int, int]:
        """The range that holds the item at `position` of them all, and its place in that range."""
        if not 0 <= position < len(self):
            raise IndexError(position)
        index = self.holding(position)
        return index, position - self.start(index)

    def holding(self, position: int) -> int:
        """The range that holds the item at `position`, one past the last for an item past them all:
        the range that runs on after them, as a run's last phase or a ramp's final batch does."""
        return bisect.bisect_right(self.ends, position)


def phase_ends(ends: Sequence[int | None], unit: Literal["step", "token", "record"]) -> tuple[int, ...]:
    """The ends of every phase but the last, which runs on (`None`), each past the one before it and
    above zero, counted in `unit`s: steps or committed tokens where a run writes its phases (`Phases`),
    records where it reads them (`PhasedStream`)."""
    if not ends or ends[-1] is not None:
        raise ValueError(f"phases need one end per phase, in {unit}s, the last None, which runs on")
    bounded = tuple(end for end in ends[:-1] if end is not None)
    rising = all(later > earlier for earlier, later in itertools.pairwise((0, *bounded)))
    if len(bounded) != len(ends) - 1 or not rising:
        raise ValueError(f"every phase but the last ends at a {unit}, past the one before it and above "
                         f"zero, not {list(ends)}")
    return bounded


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

    A run on the committed-token clock (`tokens`, the committed tokens each
    phase ends at, the last's where the data ends or None, `dew.data.pipeline.schedule`) knows no phase's
    end in records ahead: the trainer ends the phase it stands in at the step
    whose commit reaches the phase's tokens (`ending`), and the saved position
    carries that end, as the end of a phase the run completed. Each phase opens
    with the record ends of the phases before it (a corpus they read continues
    in it from where they left it), so one after an end the run has yet to
    reach opens once the run reaches it. A phase the run's commits passed
    within one step ends where it starts, holding no record.
    """

    endless = True

    def __init__(self, phases: Sequence[tuple[Callable[[tuple[int, ...]], GlobalReader], int | None]],
                 stop_seconds: float, *, tokens: Sequence[int | None] | None = None):
        ends = [end for _, end in phases]
        if tokens is None:
            phase_ends(ends, "record")
        elif any(end is not None for end in ends) or len(tokens) != len(ends):
            raise ValueError("a phase on the token clock ends where the run's commits reach its tokens, "
                             "so it has no end in records until the run reaches it")
        self._tokens = None if tokens is None else tuple(tokens)
        # Each ending phase's end in records, None for one on the token clock the run has yet to reach.
        self._ends: list[int | None] = ends[:-1]
        self._open = [open_phase for open_phase, _ in phases]
        self._streams: dict[int, GlobalReader] = {}
        for index in range(len(phases) if tokens is None else 1):
            self._stream(index)
        self._records = 0
        self._current: int | None = None
        self.stop_seconds = stop_seconds

    def __iter__(self) -> Iterator[Batch]:
        return self

    def _stream(self, phase: int) -> GlobalReader:
        """Phase `phase`'s reader, opened with the ends of the phases before it."""
        if phase not in self._streams:
            before = self._ends[:phase]
            if any(end is None for end in before):
                raise ValueError(f"phase {phase} opens where the phases before it end, which the run has "
                                 "not reached")
            self._streams[phase] = self._open[phase](tuple(end for end in before if end is not None))
        return self._streams[phase]

    def _phase(self, records: int) -> int:
        """The phase that reads the record at `records`: the first whose end, known, is past it."""
        return next((index for index, end in enumerate(self._ends) if end is None or records < end),
                    len(self._ends))

    def _start(self, phase: int) -> int:
        start = self._ends[phase - 1] if phase else 0
        if start is None:
            raise ValueError(f"phase {phase} starts where the run has not reached")
        return start

    def _resolve(self, phase: int, end: int | None) -> None:
        """Phase `phase` of the token clock ends at `end` records, or at none the run has reached; the
        phases after it, opened at its old end, open again where they now start."""
        if self._ends[phase] == end:
            return
        if end is not None and end < self._start(phase):
            raise ValueError(f"phase {phase} cannot end at record {end}, before it starts")
        self._ends[phase] = end
        for later in [index for index in self._streams if index > phase]:
            if later == self._current:
                self._current = None
            self._streams.pop(later).close()

    def __next__(self) -> Batch:
        phase = self._phase(self._records)
        if phase != self._current:
            if self._current is not None:
                self._streams[self._current].close()
            stream = self._stream(phase)
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
        end = self._ends[phase] if phase < len(self._ends) else None
        if end is not None and self._records > end:
            raise ValueError(
                f"a step of {read} records crossed the phase boundary at record "
                f"{end}; phases end on step boundaries"
            )
        return batch

    def get_state(self) -> bytes:
        # The phase of the last record read: at a boundary nothing of the next
        # phase is read yet, so the finished one is still current and a
        # resume may change what follows it or extend it. A phase the token
        # clock ended is completed where it ended, as no list of phases
        # says where that is.
        if self._tokens is not None:
            phase = self._phase(self._records)
        else:
            phase = self._phase(self._records - 1) if self._records else 0
        plans = [
            position.read(_global_position(self._stream(index).get_state())).plan
            for index in range(phase + 1)
        ]
        completed = tuple((self._stream(index).order, self._start(index + 1)) for index in range(phase))
        return position.encode(
            position.Global(
                records=self._records,
                order=self._stream(phase).order,
                completed=completed,
                plan={"phases": plans} if any(plan is not None for plan in plans) else None,
            )
        )

    def set_state(self, state: bytes) -> None:
        saved = position.read(state)
        from dew.data.plan import same_order

        if self._tokens is not None:
            if len(saved.completed) > len(self._ends):
                raise ValueError(f"the saved run finished {len(saved.completed)} phases, and this run has "
                                 f"{len(self._ends) + 1}")
            # The ends the saved run reached, and none past them, which this run reaches as it reads.
            for index in range(len(self._ends)):
                self._resolve(index, saved.completed[index][1] if index < len(saved.completed) else None)
        if saved.plan is not None:
            plans = saved.plan.get("phases") if "phases" in saved.plan else [saved.plan]
            if not isinstance(plans, list) or len(plans) != len(saved.completed) + 1:
                raise ValueError("saved phase plans do not cover the completed and current phases")
            for index, plan in enumerate(plans):
                if index >= len(self._open) or (plan is not None and not isinstance(plan, Mapping)):
                    raise ValueError("saved phase plan is not part of this run")
                end = saved.completed[index][1] if index < len(saved.completed) else saved.records
                stream = self._stream(index)
                stream.set_state(
                    position.encode(
                        position.Global(records=end - self._start(index), order=stream.order, plan=plan)
                    )
                )

        for index, (order, end) in enumerate(saved.completed):
            if (index >= len(self._ends) or not same_order(order, self._stream(index).order)
                    or self._ends[index] != end):
                raise ValueError(
                    f"the saved run finished phase {index} reading {order} to record "
                    f"{end}, and this run's phase {index} is not that; a phase the "
                    f"run has read cannot change under it"
                )
        phase = len(saved.completed)
        if phase >= len(self._open) or not same_order(saved.order, self._stream(phase).order):
            raise ValueError(
                f"the saved position is {saved.records} records into phase {phase}, "
                f"reading {saved.order}, and this run's phase {phase} reads something "
                f"else; resume with the order the checkpoint was written in"
            )
        end = self._ends[phase] if phase < len(self._ends) else None
        if end is not None and saved.records > end:
            raise ValueError(
                f"the saved run has read {saved.records} records, past this run's end "
                f"of phase {phase} at {end}"
            )
        self.close()
        self._records = saved.records

    def ending(self, state: bytes) -> bytes:
        """`state`, a position of this run on the committed-token clock, with the phase it stands in
        ended where it stands and the next opened there: where the run's commits reached the phase's
        tokens, which the trainer reads (`dew.training.Trainer.fit`)."""
        if self._tokens is None:
            raise ValueError("phases on the step clock end at the steps they were written with")
        self.set_state(state)
        phase = self._phase(self._records)
        if phase >= len(self._ends):
            raise ValueError("the last phase runs on; it has no end to reach")
        self._resolve(phase, self._records)
        return self.get_state()

    def identity(self) -> str:
        # A phase on the token clock opens where the run reaches it: the pool agrees on those open (the
        # trainer agrees again as it opens a later one) and on where every phase ends.
        ends = (f"records {list(self._ends)}" if self._tokens is None
                else f"committed tokens {list(self._tokens)}")
        return "; ".join([*(self._streams[index].identity() for index in sorted(self._streams)),
                          f"phases ending at {ends}"])

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


def _joined(kept: Batch, new: Batch) -> Batch:
    """The rows of `kept` then `new`, a ramp's buffer as a read adds to it: a batch that marks its
    real rows (`VALID_ROWS`), a kept short one, beside one that does not, all of whose rows are."""
    if (VALID_ROWS in kept) != (VALID_ROWS in new):
        kept, new = (batch if VALID_ROWS in batch else {**batch, VALID_ROWS: np.ones(rows_of(batch), bool)}
                     for batch in (kept, new))

    def join(held: list | np.ndarray, read: list | np.ndarray) -> list | np.ndarray:
        return held + read if isinstance(held, list) else np.concatenate([held, read])

    return jax.tree.map(join, kept, new, is_leaf=lambda column: isinstance(column, list))


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

    def __init__(self, source: Resumable, stages: Sequence[Stage], partition: DataPartition, *,
                 tail: Tail = "dropped", order_records: int | None = None):
        self._source = source
        self._stages = tuple(stages)
        # Where a global order ends, every share ends its last step at the same record: whole where
        # its short last step is dropped, holding the order's last records, the rest filled, where
        # it is read (`Loader.tail`).
        self._tail, self._order_records = tail, order_records
        # The stages that end, the final batch running on after them.
        self._ramp = Ranges.ending(tuple(stage.records for stage in stages[1:]))
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
        return self._stages[self._ramp.holding(self._records)]

    def __next__(self) -> Batch:
        stage = self._stage()
        rows = self._partition.rows(stage.batch)
        left = None if self._order_records is None else self._order_records - self._records
        if left is not None and (left <= 0 or (left < stage.batch and self._tail == "dropped")):
            raise StopIteration
        while self._held is None or rows_of(self._held) < rows:
            read = next(self._source, None)
            if read is None:
                break
            self._held = read if self._held is None else _joined(self._held, read)
        if self._held is None or not rows_of(self._held):
            raise StopIteration
        if left is not None and left < stage.batch:
            # The order's last step: its records the first `left` of the step, which this share's
            # rows hold every count-th of from its own place on, the rest repeats.
            step = FilledBatch(rows).map(take_batch(self._held, slice(min(rows, rows_of(self._held)))))
            real = np.arange(rows) * self._partition.count + self._partition.index < left
            self._held, self._records = None, self._records + stage.batch
            return {**step, VALID_ROWS: np.asarray(step.get(VALID_ROWS, real), bool) & real}
        if rows_of(self._held) < rows:
            # The source ended short of a whole step, in an order whose end no read counted ahead.
            raise StopIteration
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
    # An impossible schedule fails here, not mid-run. Moving its stages to update boundaries
    # (`Ramp.stages`' window) cannot make one impossible, so this checks the schedule as given.
    ramp.stages(dataset.batch)

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
                return RampedStream(stream, ramp.stages(dataset.batch, partition.window), partition,
                                    tail=dataset.tail, order_records=dataset.order_records)
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
]
