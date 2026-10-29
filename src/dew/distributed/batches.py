"""A run's batches on the devices: the fields a batch carries to the host apart (`split_fields`,
`with_host`), the shape a batch's compiled step is keyed by (`Signature`), its shardings and its
placement on a mesh (`batch_shardings`, `shard_batch`, `first_reader_batch`), its host fields
gathered across processes by the rows' origins (`row_origins`, `shared_host_fields`), and the
prefetch that keeps the next batches in flight within a memory budget (`DevicePrefetchIterator`).
Where the state's arrays live is `dew.distributed.placement`'s, whose placed-tree helpers this
reads."""

from __future__ import annotations

import base64
import collections
import contextlib
import json
import logging
import pickle
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental import multihost_utils
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew._coordination import agreed, broadcast_from_process_zero, from_every_process, stop_at_exit
from dew.batch import Batch
from dew.cleanup import attempted
from dew.data.dataset import Budgeted, Checkpointable, Closeable, DataPartition, Stoppable
from dew.distributed.placement import Placement, placeable, placed_bytes, placed_shapes
from dew.mesh import SEQUENCE_AXIS, batch_axes
from dew.nn.inputs import filled_validity
from dew.telemetry.profile import region

_log = logging.getLogger(__name__)


@overload
def split_fields(batch: Batch) -> tuple[Batch, dict]: ...


@overload
def split_fields(batch: np.ndarray | jax.Array) -> tuple[np.ndarray | jax.Array, dict]: ...


def split_fields(batch: Batch | np.ndarray | jax.Array) -> tuple[Batch | np.ndarray | jax.Array, dict]:
    """`batch` split in one walk into the fields a device can hold
    (`placeable`), the batch a step and an objective's programs are called
    on, and those none can, which host code reads and a compiled step cannot
    take, each in the mappings that hold them (`with_host` joins them back).
    A batch with no host field, a bare array among them, is its own device
    part."""
    if not isinstance(batch, Mapping):
        return batch, {}
    device: dict = {}
    host: dict = {}
    for name, value in batch.items():
        if isinstance(value, Mapping):
            device[name], nested = split_fields(value)
            if nested:
                host[name] = nested
        elif placeable(value):
            device[name] = value
        else:
            host[name] = value
    return (device if host else batch), host


def with_host(placed: Batch, host: Batch) -> Batch:
    """`placed` with the host fields put back where they were."""
    return {**placed, **{name: with_host(placed[name], value) if isinstance(value, Mapping) else value
                         for name, value in host.items()}}


Signature = tuple[jax.tree_util.PyTreeDef, tuple[jax.ShapeDtypeStruct, ...]]
"""A batch's abstract signature, the key a compiled step is held under: its
tree, which holds each node's static fields, and each leaf's shape and
dtype. A fixed batch has one; a ramped batch has one per stage of its ramp,
each compiled on the first step that reads it."""


def batch_signature(batch: Batch) -> Signature:
    """The abstract signature of a batch of arrays or their shapes (`jax.ShapeDtypeStruct`)."""
    leaves, tree = jax.tree.flatten(batch)
    return tree, tuple(jax.ShapeDtypeStruct(np.shape(leaf), jnp.result_type(leaf)) for leaf in leaves)


def batch_layout(batch: Batch) -> str:
    """A batch's fields, each with its dtype and global shape; the batch may be
    arrays or their abstract shapes (`jax.ShapeDtypeStruct`)."""
    return ", ".join(f"{jax.tree_util.keystr(path)} {jnp.result_type(leaf)}{list(np.shape(leaf))}"
                     for path, leaf in jax.tree_util.tree_leaves_with_path(batch))


def batch_shardings(mesh: Mesh | AbstractMesh, batch: Batch) -> Placement[Batch]:
    """Return a sharding for each leaf of `batch`, chosen from the leaf's shape.

    Rows split over the row axes, those the rules in context give
    `activation_batch` (`batch_axes`): data, expert and fsdp by default. A
    leaf of rank 2 or 3 holds a sequence per row: token ids, segment ids,
    positions, encoded tokens. Its second dimension splits over the sequence
    axis when the mesh has one that divides it, and otherwise stays whole,
    the way `logical_spec` drops a name no dimension can split. An image or
    a video is not a sequence, so only its rows split.

    This only decides placement and never changes values. Until a model
    constrains its attention to the sequence axis, a sequence split here is
    gathered again inside the model. Only the shape is read, so a leaf that is
    already a global array costs no transfer.
    """
    rows = batch_axes(mesh) or None
    sequence = SEQUENCE_AXIS if SEQUENCE_AXIS in mesh.axis_names else None
    sequence_size = mesh.shape.get(SEQUENCE_AXIS, 1)

    def leaf_sharding(leaf):
        shape = np.shape(leaf)
        if not shape:
            return NamedSharding(mesh, P())
        if sequence is not None and len(shape) in (2, 3) and shape[1] % sequence_size == 0:
            return NamedSharding(mesh, P(rows, sequence))
        return NamedSharding(mesh, P(rows))

    return jax.tree.map(leaf_sharding, batch)


def _placed_rows(mesh: Mesh) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
    """The shape a share's leaf is placed at on `mesh`: its rows those of every share
    (`DataPartition.of`), each share whole rows, and every other dimension whole."""
    count = DataPartition.of(mesh).count

    def placed(shape: tuple[int, ...]) -> tuple[int, ...]:
        return (shape[0] * count, *shape[1:]) if shape else ()

    return placed


def placed_abstract(mesh: Mesh, batch: Batch) -> Batch:
    """The batch `shard_batch` would place from this process's share `batch`, as shapes with their
    shardings and nothing transferred: what a run reads a step's plan and its whole state from before
    any array of it is on a device (`Trainer.fit`)."""
    device, host = split_fields(batch)
    if host:
        return with_host(placed_abstract(mesh, device), host)
    batch = filled_validity(batch) if jax.process_count() > 1 else batch
    placed = _placed_rows(mesh)

    def shape(leaf, sharding: NamedSharding) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(placed(np.shape(leaf)), jnp.result_type(leaf), sharding=sharding)

    return jax.tree.map(shape, batch, batch_shardings(mesh, batch))


def shard_batch(mesh: Mesh, batch: Batch, reads: Mapping[Signature, tuple[bool, ...]] | None = None) -> Batch:
    """Assemble this process's share of each array into a globally sharded one.

    The placed batch owns its memory because reader buffers are borrowed until
    the next batch. CPU placement copies each borrowed host leaf once, since
    JAX 0.11.2 drops copy_semantics in pxla._shard_np_array and ignores
    may_alias=False for NumPy inputs. Accelerator transfers copy by construction
    and release the borrowed host buffer when the transfer completes.

    The share is the one `DataPartition.of(mesh)` names: that share's rows, each
    whole in every other dimension. A pool assembles one leaf per process,
    so every process has to hand this
    the same tree. Validity is the one optional token field, and whether a
    process's own rows needed padding is rank-local. So a pool materializes
    it at every `ModelInputs` of the batch that lacks it, before the
    traversal below reads the leaves. The schema then depends on the process
    count and the batch's structure and never on which rows this process
    drew. One process changes nothing and keeps the omission the fused
    attention kernel wants.

    `reads` holds, by the signature of a batch as placed (`batch_signature`
    of what this returns), which of its leaves the step compiled for it reads
    (`step_reads`); one it does not read stays on the host, no transfer
    made, the reader's buffer still released at the next read: a copy of its
    own where this process holds every row, else a stand-in of the placed
    shape, which the executable checks and drops, its values unread. A
    batch whose placed signature it lacks is placed whole. The lookup is
    here, where the placed shapes are made, so the step that publishes a
    mask and the placement that reads it name one batch alike.

    The rule is deliberately local. Placement runs on
    `DevicePrefetchIterator`'s worker thread while the step's collectives run
    on the caller's, and a collective issued from here would have to be
    ordered against those across every process. Agreeing which sites
    actually carry validity, the way `agreed_validity` does for a generation
    request, belongs where the caller's own collectives are issued.
    """
    device, host = split_fields(batch)
    if host:
        # A field no device holds (`placeable`) stays on the host, where host code reads it.
        return with_host(shard_batch(mesh, device, reads), host)
    batch = filled_validity(batch) if jax.process_count() > 1 else batch
    cpu = all(device.platform == "cpu" for device in mesh.devices.flat)
    leaves, tree = jax.tree.flatten(batch)
    placed = _placed_rows(mesh)
    signature = tree, tuple(jax.ShapeDtypeStruct(placed(np.shape(leaf)), jnp.result_type(leaf))
                            for leaf in leaves)
    read = (reads or {}).get(signature, (True,) * len(leaves))
    # An unread leaf is still an argument the executable checks against the placed shapes before it
    # drops it, so in a pool, whose process holds a share of the rows, it is a zero-cost stand-in
    # of the placed shape; a process that holds every row keeps its own copy.
    held = {index: np.array(leaf, copy=True) if placed(np.shape(leaf)) == np.shape(leaf)
            else np.broadcast_to(np.zeros((), np.asarray(leaf).dtype), placed(np.shape(leaf)))
            for index, (leaf, kept) in enumerate(zip(leaves, read, strict=True))
            if not kept and not isinstance(leaf, jax.Array)}
    moving = [leaf if isinstance(leaf, jax.Array) else np.array(leaf, copy=True) if cpu else np.asarray(leaf)
              for index, leaf in enumerate(leaves) if index not in held]
    every: list[NamedSharding] = jax.tree.leaves(batch_shardings(mesh, batch))
    shardings = [sharding for index, sharding in enumerate(every) if index not in held]
    if jax.process_count() == 1:
        # The whole batch is this process's share, placed by one device_put
        # for the tree: 74 us of host time a batch on an RTX 4080, and 174 on
        # 8 CPU devices, where a call per leaf took 105 and 222.
        moved = iter(jax.device_put(moving, shardings))
    else:
        def place(leaf, sharding: NamedSharding) -> jax.Array:
            # A device of a sequence or a stage that spans processes picks its own slice out of it.
            return jax.make_array_from_process_local_data(sharding, leaf, placed(np.shape(leaf)))

        moved = iter([place(leaf, sharding) for leaf, sharding in zip(moving, shardings, strict=True)])
    return tree.unflatten([held[index] if index in held else next(moved) for index in range(len(leaves))])


def first_reader_batch(mesh: Mesh, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The batch the first reader of this process's share read, on every reader of it.

    The processes that read one share (`DataPartition.of(mesh).readers`) hand
    `shard_batch` rows the same devices hold, so their batches must be the
    same. A source whose reads differ between them, as independent draws
    from an engine do, reads on the share's first reader alone
    (`DataPartition.reader` 0). Every process of the pool then calls this
    on the thread its other collectives run on, as a rollout runs: each
    first reader's batch, laid out as process 0's is, reaches every
    process, and each takes its share's. A later reader's `batch` is not
    read. On a mesh whose shares have one reader each this is `batch`.
    """
    partition = DataPartition.of(mesh)
    if partition.readers == 1:
        return dict(batch)
    layout = broadcast_from_process_zero(
        {name: [list(np.shape(leaf)), str(np.asarray(leaf).dtype)] for name, leaf in batch.items()})

    def held() -> dict[str, np.ndarray]:
        if partition.reader:
            return {name: np.zeros(shape, np.dtype(dtype)) for name, (shape, dtype) in layout.items()}
        mine = {name: np.asarray(leaf) for name, leaf in batch.items()}
        if {name: [list(leaf.shape), str(leaf.dtype)] for name, leaf in mine.items()} != layout:
            raise ValueError("the first readers' batches are not laid out alike; every process of a pool "
                             "hands shard_batch the same tree")
        return mine

    rows = agreed("first reader batch", held)
    gathered = multihost_utils.process_allgather(
        {"share": np.asarray([partition.index, partition.reader], np.int32), **rows})
    source = next(process for process, (index, reader) in enumerate(gathered["share"].tolist())
                  if index == partition.index and reader == 0)
    return {name: np.asarray(gathered[name][source]) for name in layout}


def row_origins(mesh: Mesh, rows: int) -> jax.Array:
    """Each row of the batch placed on `mesh`, numbered by where it came from, `share * rows + row`:
    this process's `rows` rows of its share (`DataPartition.of`), placed as the batch's are
    (`shard_batch`), so the gathered numbers say which share's row each row of the batch is."""
    share = DataPartition.of(mesh).index
    return shard_batch(mesh, {"origin": share * rows + np.arange(rows, dtype=np.int32)})["origin"]


def shared_host_fields(mesh: Mesh, host: Batch, origins: np.ndarray, rows: int) -> Batch:
    """The host fields of a batch spread over processes, each row where `origins` places it: every
    share's own rows, from its first reader, which no device carries, so they meet on the host
    (`from_every_process`), numbered as `row_origins` numbers them. Every process takes part; the
    fields are whole on each, a list or array column a leaf, as a validation pass and a rollout's
    reward read them."""
    partition = DataPartition.of(mesh)
    # Agreed, so a record no reader can serialize fails every process rather than leaving its peers
    # waiting for its fields.
    told = agreed("host fields serialization", lambda: base64.b64encode(pickle.dumps(host)).decode()
                  if partition.reader == 0 else "")
    shares = {index: pickle.loads(base64.b64decode(word))
              for index, word in from_every_process([partition.index, told]) if word}
    ordered = [shares[index] for index in range(len(shares))]
    share, row = np.divmod(origins, rows)

    def gathered(*columns):
        first = columns[0]
        # A field of one value for the whole batch, not one a row (`rows_of`), is every share's alike.
        rowed = len(first) == rows if isinstance(first, list | tuple) else np.shape(first)[:1] == (rows,)
        if not rowed:
            return first
        values = [columns[position][place] for position, place in zip(share, row, strict=True)]
        if not isinstance(first, np.ndarray):
            return values
        return np.asarray(values, dtype=object) if first.dtype == object else np.asarray(values)

    return jax.tree.map(gathered, *ordered, is_leaf=lambda column: not isinstance(column, Mapping))


# The batches a `DevicePrefetchIterator` queues ahead of the step by default.
PREFETCH_DEPTH = 2


def prefetched_bytes(batch: Batch, shardings: Placement[Batch], depth: int = PREFETCH_DEPTH) -> int:
    """Return the bytes a device holds of the batches `fit` places while a step runs.

    These are the batches next to the one the step reads: the `depth` it queues
    (`fit(device_prefetch=)`) and the one it is placing, each laid out as
    `shardings` places `batch`.
    """
    return (depth + 1) * placed_bytes(placed_shapes(batch, shardings))


class DevicePrefetchIterator:
    """Reads batches on a worker thread and places them on the mesh ahead of the step.

    The iterator closes its source when the source runs out or the iterator is
    closed, so use it as a context manager or call `close`, even after a loop
    that ended early. At most `depth` batches are queued, plus the one being
    read and placed.

    Every call into the source runs on the worker thread: `next`, reading the
    checkpoint position, and the final close. The exception is the source's
    optional `request_stop` hook, which runs on the thread that closes the
    iterator, so it must be thread-safe and must not block.

    `source_state`, when given, is a saved position the worker restores before
    its first read; if the source is not checkpointable, the first `next`
    raises TypeError. After each batch the iterator returns, `source_state`
    holds the source's position after that batch.

    `prepare`, when supplied, transforms the source's host batch before
    placement, on the same worker and after its checkpoint position is read.
    `peek` returns the first batch so prepared, on the host, before the worker
    places anything; the worker places it, and reads on, from the first `next`.
    `reads` holds, by the placed batch's signature, which leaves a compiled
    step reads (`step_reads`); a batch of a signature it lacks, its step not
    yet compiled, is placed whole (`shard_batch`).
    Analytical per-row telemetry therefore travels with its own batch and
    never requires a synchronous device-to-host read in the step loop.

    If the worker thread cannot be created, the source is finalized on the
    caller's thread before the error is re-raised; no worker has touched it at
    that point. That path runs the source's close synchronously, so the close
    has to return on its own. When a running worker is stuck in source code
    that cannot be stopped, `close` raises TimeoutError.
    """

    def __init__(self, iterator: Iterator, mesh: Mesh, depth: int = PREFETCH_DEPTH,
                 source_state: bytes | None = None, *, prepare: Callable[[Batch], Batch] | None = None,
                 reads: Mapping[Signature, tuple[bool, ...]] | None = None):
        if depth <= 0:
            raise ValueError("prefetch depth must be positive")
        self._iterator: Iterator | None = iter(iterator)
        self._source_name = type(self._iterator).__name__
        self._mesh: Mesh | None = mesh
        # The placed batches waiting for the consumer, at most `depth`; `_ready` wakes whoever waits on
        # them, on an enqueue, a dequeue, a stop and the worker's end, so neither side polls.
        self._queued: collections.deque[tuple[Batch, bytes | None]] = collections.deque()
        self._depth = depth
        self._ready = threading.Condition()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._start_lock = threading.Lock()
        # Held while a stop request reaches the source and while the worker lets the source go:
        # a request reaches the source it read (during its close too, which may wait on it), and
        # the worker releases the source only after that request returns.
        self._handover = threading.Lock()
        self._error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self.source_state = source_state
        self._prepare = prepare
        # The worker places batches by the rules its creator reads, which flax
        # keeps per thread (`batch_shardings`).
        self._rules = nn.get_logical_axis_rules()
        self._reads = {} if reads is None else reads
        # The first batch, read and prepared and held on the host while `_placing` is unset, which the
        # first `next` sets (`peek`).
        self._first: Batch | None = None
        self._placing = False
        # The callers holding placements (`held`) and whether a batch is being placed now, both under
        # `_ready`, which a stop notifies too, so a worker waiting out a hold still stops.
        self._holds = 0
        self._placing_now = False
        self._thread = threading.Thread(target=self._prefetch, name="dew-prefetch", daemon=True)
        self._withdraw_exit: Callable[[], None] | None = None
        # No source work may start until the caller owns this object. In
        # particular, an interrupt during restoration must unwind through
        # this iterator's close, never the caller's untransferred-source path.

    def _start(self) -> None:
        """Start the worker on first use, finalizing here if it cannot start."""
        with self._start_lock:
            if self._done.is_set():
                return
            if self._thread.ident is None:
                try:
                    self._thread.start()
                except RuntimeError as error:
                    if self._thread.ident is None:
                        # Thread creation failed before any source operation.
                        self._stop.set()
                        try:
                            self._stop_source()
                        except BaseException as failure:
                            error.add_note(f"Source cancellation failed: {failure!r}")
                        self._prefetch()
                    raise
                # A worker still placing a batch when Python finalizes would abort the process.
                self._withdraw_exit = stop_at_exit(self._thread, self._cancel, timeout=self._budget())

    def _budget(self) -> float | None:
        """The seconds the source gives itself to stop (`Budgeted.stop_seconds`), None for a
        source that sets no deadline of its own."""
        source = self._iterator
        seconds = source.stop_seconds if isinstance(source, Budgeted) else None
        return None if seconds is None else float(seconds)

    def _cancel(self) -> None:
        """Ask the worker to stop between batches, and a stoppable source to stop its own work."""
        first = not self._stop.is_set()
        self._stop.set()
        with self._ready:
            self._ready.notify_all()
        if first:
            self._stop_source()

    def _stop_source(self) -> None:
        """Ask the source to stop its own work, unless the worker has already let it go."""
        with self._handover:
            source = self._iterator
            if isinstance(source, Stoppable):
                source.request_stop()

    def _prefetch(self):
        """Read, place and enqueue batches until the source drains or a stop.

        This is the worker thread's whole body, and it owns the source's
        finalization too, so the source is touched from one thread only. A
        drained iterator ends the loop through `StopIteration`, any other
        failure is published to the consumer, and the `finally` closes the
        source either way.

        `_start` calls this on the caller's thread when the worker could not
        be created. The stop flag is already set there, so the loop does
        nothing but finalize.
        """
        iterator, mesh = self._iterator, self._mesh
        assert iterator is not None and mesh is not None
        source = iterator if isinstance(iterator, Checkpointable) else None
        batch = state = read = placed = None
        try:
            if not self._stop.is_set() and self.source_state is not None:
                if source is None:
                    raise TypeError(f"{self._source_name} cannot resume from a saved position")
                saved = self.source_state
                source.set_state(saved if isinstance(source.get_state(), bytes)
                                 else json.loads(saved))
            while not self._stop.is_set():
                with region("data.read"):
                    batch = next(iterator)
                if self._stop.is_set():
                    break
                with region("data.position"):
                    read = source.get_state() if source is not None else None
                    state = read if read is None or isinstance(read, bytes) else json.dumps(read).encode()
                if self._prepare is not None:
                    with region("data.place"), nn.logical_axis_rules(self._rules):
                        batch = self._prepare(batch)
                if not self._placing:
                    with self._ready:
                        self._first = batch
                        self._ready.notify_all()
                        self._ready.wait_for(lambda: self._stop.is_set() or self._placing)
                        self._first = None
                    if self._stop.is_set():
                        break
                with self._ready:
                    self._ready.wait_for(lambda: self._stop.is_set() or not self._holds)
                    if self._stop.is_set():
                        break
                    self._placing_now = True
                try:
                    with region("data.place"), nn.logical_axis_rules(self._rules):
                        placed = shard_batch(mesh, batch, self._reads)
                finally:
                    with self._ready:
                        self._placing_now = False
                        self._ready.notify_all()
                batch = None
                with region("data.enqueue"), self._ready:
                    self._ready.wait_for(lambda: self._stop.is_set() or len(self._queued) < self._depth)
                    if not self._stop.is_set():
                        self._queued.append((placed, state))
                        self._ready.notify_all()
                placed = state = read = None
        except StopIteration:
            # A drained source is how a prefetch thread ends; the finally
            # below publishes the end, and no error goes with it.
            _log.debug("%s drained; the prefetch thread is done", self._source_name)
        except BaseException as error:
            self._error = error
        finally:
            batch = placed = state = read = None
            try:
                close = iterator.close if isinstance(iterator, Closeable) else None
                if close is not None:
                    close()
            except BaseException as error:
                self._cleanup_error = error
            finally:
                with self._handover:
                    self._iterator = None
                self._mesh = self._prepare = None
                # No bound close method or checkpointable alias may retain a
                # Grain pipeline after the worker exits.
                iterator = source = mesh = close = None
                if self._stop.is_set():
                    self._discard()
                with self._ready:
                    self._done.set()
                    self._ready.notify_all()

    @contextlib.contextmanager
    def held(self) -> Iterator[None]:
        """Place no batch while the caller's block runs, a placement under way finishing first; the
        worker reads on meanwhile, and its next placement waits for the block to end.

        A run holds its prefetcher while it does device work beside its steps (a compile, an
        evaluation, a checkpoint): XLA's GPU pool gives an allocation the lowest block it fits, so a
        batch placed while that work holds memory lands past it, inside the free block the next step's
        temporaries take whole, and stays there until a later step reads it (row 2088: a 4B's step that
        fit its 50.35 GiB block before its first validation pass found it split after the pass, by a
        batch the stream read ahead of the pass). The block must not read this iterator, whose next
        batch may wait on the placement it holds; closing it stops the worker as ever."""
        with self._ready:
            self._holds += 1
            self._ready.wait_for(lambda: not self._placing_now)
        try:
            yield
        finally:
            with self._ready:
                self._holds -= 1
                self._ready.notify_all()

    def _discard(self) -> None:
        """Drop every batch already queued, without waiting for more."""
        with self._ready:
            self._queued.clear()
            self._ready.notify_all()

    def close(self, *, timeout: float | None = None) -> None:
        """Stop the worker and wait for it, discarding unread batches but keeping `source_state`.

        The wait lasts `timeout` seconds, else the source's own `stop_seconds`
        (a grain pipeline joins one worker process after another), else until
        the worker ends: the source, the external reader, owns its deadline. If
        the worker is still alive after a deadline, this raises TimeoutError.
        Cancellation stays requested and a later close may wait again, but the
        source's in-flight work and buffers may not have been released. An
        error from the source's own close is raised here too.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("a prefetch worker cannot close itself")
        if timeout is None:
            timeout = self._budget()
        error = None
        try:
            self._cancel()
        except BaseException as failure:
            error = failure
        self._discard()
        self._start()  # Even an unused iterator finalizes on its worker.
        if self._thread.ident is not None:
            self._thread.join(timeout)
        self._discard()
        self._error = None  # Unconsumed speculative errors are discarded too.
        if not self._thread.is_alive() and self._withdraw_exit is not None:
            self._withdraw_exit()
            self._withdraw_exit = None
        failure = None
        if self._thread.is_alive():
            failure = TimeoutError(
                f"{self._source_name} did not stop within {timeout}s: its next, "
                "placement or finalization is uncooperative; the worker is still alive")
        elif self._cleanup_error is not None:
            failure, self._cleanup_error = self._cleanup_error, None
        if error is not None:
            if failure is not None:
                error.add_note(f"Prefetch cleanup also failed: {failure!r}")
            raise error
        if failure is not None:
            raise failure

    def __enter__(self) -> DevicePrefetchIterator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        attempted((("Prefetch cleanup", self.close),), primary=exc)

    def __iter__(self):
        return self

    def peek(self) -> Batch | None:
        """The first batch as the source and `prepare` give it, on the host, before the worker places
        any; None where the source holds none. A run checks what it will train against its shapes
        before it places its state (`dew.distributed.batches.placed_abstract`); the first `next`
        places this batch and reads on, after which the peeked batch's buffers are the reader's
        again."""
        self._start()
        with self._ready:
            self._ready.wait_for(lambda: self._first is not None or self._placing or self._queued
                                 or self._done.is_set() or self._stop.is_set())
            if self._first is not None:
                return self._first
            if self._placing or self._queued:
                raise RuntimeError("peek reads the first batch, before any is placed")
        self._raise_ended()
        return None

    def __next__(self):
        if self._stop.is_set():
            raise StopIteration
        with self._ready:
            self._placing = True
            self._ready.notify_all()
        self._start()
        with self._ready:
            self._ready.wait_for(lambda: self._queued or self._done.is_set() or self._stop.is_set())
            if self._stop.is_set():
                raise StopIteration
            if self._queued:
                batch, position = self._queued.popleft()
                self._ready.notify_all()
                self.source_state = position
                return batch
        self._raise_ended()
        raise StopIteration

    def _raise_ended(self) -> None:
        """Raise the worker's error, now it has ended with every batch it published read, if it had
        one, or its source's cleanup failure."""
        error, self._error = self._error, None
        cleanup, self._cleanup_error = self._cleanup_error, None
        if error is not None:
            if cleanup is not None:
                error.add_note(f"Source cleanup failed: {cleanup!r}")
            raise error from None
        if cleanup is not None:
            raise cleanup from None


__all__ = ["DevicePrefetchIterator", "batch_shardings", "placed_abstract"]
