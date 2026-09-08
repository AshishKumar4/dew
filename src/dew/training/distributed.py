"""Device mesh, parameter layout, and host-to-device prefetch."""

from __future__ import annotations

import dataclasses
import json
import math
import queue
import threading
from collections.abc import Mapping
from typing import Any, Iterator, Optional, TypeAlias

import jax
import numpy as np
from flax import linen as nn
from flax.linen import spmd
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P
from dew.data.dataset import Checkpointable
from dew.nn.inputs import BATCH_AXES, filled_validity
from dew.nn.sharding import (
    DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, SEQUENCE_AXIS, STAGE_AXIS, TENSOR_AXIS, LogicalAxes,
    declared_axes,
)
from dew.objectives.base import Batch, Variables

# The axes a parameter can be split over. A dimension named 'exp' takes the
# expert axis, a width the rules redirect takes tensor, everything else
# takes fsdp. The data and sequence axes split the batch, and the stage axis
# holds the pipeline's stages of the layer stack; none of the three ever
# places a parameter, which `Layout` refuses a rule for.
PARAMETER_AXES = (EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS)

# The batch's rows split across every axis but sequence and stage, whichever
# they sit on; the sequence dimension splits over the sequence axis. Every
# stage sees the whole batch, since the pipeline hands its microbatches from
# stage to stage itself. Only parameters distinguish the axes further.
BATCH_SPEC = P(BATCH_AXES, SEQUENCE_AXIS)

MeshAxes: TypeAlias = str | tuple[str, ...] | None
Placement: TypeAlias = Any
"""A pytree shaped like what it places, with a `NamedSharding` at every leaf.
Python has no way to say "this tree's structure with those leaves", so the
name carries what the annotation cannot."""

LogicalAxisRules: TypeAlias = tuple[tuple[str, MeshAxes], ...]

# Rule order is precedence when two logical dimensions target the one fsdp axis.
# It reproduces the largest-axis choice for the declared model shapes while
# giving a config one place to redirect a width onto the tensor axis.
DEFAULT_RULES: LogicalAxisRules = (
    ("vocab", FSDP_AXIS),
    ("mlp", FSDP_AXIS),
    ("modulation", FSDP_AXIS),
    ("attention", FSDP_AXIS),
    # The gated delta net's projected width (keys, values and their gate),
    # placed like the attention's: the width over the model dimension.
    ("linear", FSDP_AXIS),
    ("embed", FSDP_AXIS),
    ("head_dim", FSDP_AXIS),
    ("heads", FSDP_AXIS),
    ("kv", FSDP_AXIS),
    # The latent widths multi-head latent attention compresses through and
    # the sparse indexer's head dim: model-width-like, so they ride fsdp.
    # There is no tensor axis today; when one lands, these keep fsdp until
    # a rule moves them.
    ("index", FSDP_AXIS),
    ("kvlora", FSDP_AXIS),
    ("qlora", FSDP_AXIS),
    ("output", FSDP_AXIS),
    ("exp", EXPERT_AXIS),
)


@dataclasses.dataclass(frozen=True)
class MeshSpec:
    """How many devices each sharding axis takes; data parallelism fills the rest."""
    fsdp: int = 1
    expert: int = 1
    """Devices the expert dimension of an MoE layer is split over."""
    tensor: int = 1
    """Devices a redirected width is split over; 1 keeps every width on fsdp."""
    sequence: int = 1
    """Devices the batch's sequence dimension is split over; 1 keeps whole sequences."""
    stage: int = 1
    """Pipeline stages the layer stack is split into, each on its own devices; 1
    runs the stack whole on every device."""
    microbatches: Optional[int] = None
    """Microbatches a step feeds through the stages, a multiple of `stage`; None
    is one per stage, the smallest schedule. A stage runs one microbatch while
    the next runs the one before it, so more microbatches shrink the idle
    time at either end of the step and make each iteration's matmuls smaller."""

    def __post_init__(self):
        if self.stage < 1:
            raise ValueError(f"stage counts pipeline stages, got {self.stage}")
        if self.microbatches is None:
            return
        if self.stage == 1:
            raise ValueError(
                f"microbatches ({self.microbatches}) feed the stage axis, and "
                "stage is 1; set stage above 1 or leave microbatches unset")
        if self.microbatches < self.stage or self.microbatches % self.stage:
            raise ValueError(
                f"microbatches must be a positive multiple of stage "
                f"({self.stage}), got {self.microbatches}")


def _mesh_axes(assignment: MeshAxes) -> tuple[str, ...]:
    """One entry of a spec or a rule as the mesh axes it names."""
    if assignment is None:
        return ()
    return (assignment,) if isinstance(assignment, str) else tuple(assignment)


def _rule_table(rules: LogicalAxisRules | Mapping[str, MeshAxes]) -> LogicalAxisRules:
    """`rules` as the tuple of pairs flax reads, in precedence order. They are
    written as a mapping in code, as pairs on the command line, and arrive as
    lists from a JSON record."""
    items = rules.items() if isinstance(rules, Mapping) else rules
    return tuple((name, axes if axes is None or isinstance(axes, str) else tuple(axes))
                 for name, axes in items)



def build_mesh(spec: MeshSpec = MeshSpec(), devices: Optional[list] = None) -> Mesh:
    """Six-axis device mesh: parameters shard over 'fsdp', 'expert' and
    'tensor', batches over the first four with their sequence dimension on
    'sequence', and the layer stack over 'stage'.

    An MoE layer's expert dimension is the one dimension no dense model has,
    and splitting it is what expert parallelism is, so it gets its own axis
    and leaves 'fsdp' to the model's widths. The tensor axis is where a run's
    rules redirect a width when one card cannot hold it; the sequence axis is
    where long sequences split; the stage axis is where a decoder's layers
    split into pipeline stages, each stage on its own devices. Sizes of 1
    degenerate to plain data parallelism, so the same code path serves every
    topology without a flag. Axes are Auto so GSPMD infers the collectives.
    """
    devices = list(devices) if devices is not None else jax.devices()
    sharded = spec.fsdp * spec.expert * spec.tensor * spec.sequence * spec.stage
    if (spec.fsdp < 1 or spec.expert < 1 or spec.tensor < 1
            or spec.sequence < 1 or len(devices) % sharded):
        raise ValueError(
            f"fsdp {spec.fsdp} times expert {spec.expert} times tensor "
            f"{spec.tensor} times sequence {spec.sequence} times stage "
            f"{spec.stage} must be a positive divisor of device count {len(devices)}")
    return jax.make_mesh(
        (len(devices) // sharded, spec.expert, spec.fsdp, spec.tensor, spec.sequence,
         spec.stage),
        (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS, SEQUENCE_AXIS, STAGE_AXIS),
        devices=devices,
        axis_types=(AxisType.Auto,) * 6,
    )


def parameter_spec(shape: tuple, fsdp_size: int, min_shard_size: int) -> P:
    """Shard the largest evenly-divisible axis over 'fsdp', else replicate.

    Applied to every leaf of the train state, params, optimizer moments and
    EMA copies alike: the moments and the copies have the same shapes as the
    params they track, so they pick up the same spec without anyone having
    to describe the optimizer's layout.
    """
    if fsdp_size == 1 or int(np.prod(shape, dtype=np.int64)) < min_shard_size:
        return P()
    for axis in sorted(range(len(shape)), key=lambda i: -shape[i]):
        if shape[axis] % fsdp_size == 0:
            return P(*([None] * axis), FSDP_AXIS)
    return P()


def _mesh_spec(shape: tuple, axes: LogicalAxes, rules: LogicalAxisRules, mesh: Mesh) -> P:
    """The spec these logical axes ask for, reduced to one the shape can take.

    A mesh axis of size 1 shards nothing, so it is dropped from the spec,
    where it would only obscure what is replicated. A dimension its assigned
    axes do not divide evenly cannot be split at all, so its name is dropped
    and the rules hand the axis to the next dimension that names it: an odd
    vocabulary shards the embedding on its width and keeps the table in the
    layout. Only a parameter no named dimension can split stays whole, which
    the tolerance check turns into an error when it matters.
    """
    names: list[str | None] = list(axes)
    while True:
        mapped = spmd.logical_to_mesh_axes(tuple(names), rules)
        assert mapped is not None, "flax answers None for array_dim_names=None only"
        assigned = [
            tuple(axis for axis in _mesh_axes(assignment) if mesh.shape[axis] > 1)
            for assignment in mapped]
        blocked = [
            dimension for dimension, mesh_axes in enumerate(assigned)
            if shape[dimension] % math.prod(mesh.shape[axis] for axis in mesh_axes)]
        if not blocked:
            break
        for dimension in blocked:
            names[dimension] = None
    entries = [mesh_axes[0] if len(mesh_axes) == 1 else mesh_axes or None
               for mesh_axes in assigned]
    while entries and entries[-1] is None:
        entries.pop()
    return P(*entries)


HOST_RESIDENT = ("opt_state", "ema")
"""The train-state fields a layout may keep in pinned host memory between
steps. The parameters are not among them: a host copy of the weights saves
device memory only if each layer fetches its own just in time inside the
stack, which the decoder does not do."""


@dataclasses.dataclass(frozen=True)
class Layout:
    """How a train state is placed on a mesh.

    `rules` map the logical axes the modules declare (`dew.nn.sharding`) onto
    the parameter axes of the mesh, in precedence order; an axis of size 1
    shards nothing, so the same table serves every topology. A rule onto the
    data, sequence or stage axis is refused: the first two split the batch,
    and a parameter placed on either would be gathered on every use; the
    stage axis holds the layer stack's pipeline stages, which the decoder
    places itself from the stored tree. Below `min_shard` elements a
    parameter costs more in collectives than it saves in memory, so it stays
    replicated. `tolerance` is the fraction of shardable parameter elements
    a layout may leave replicated before `check` refuses it. `host` names
    the fields of `HOST_RESIDENT` kept in pinned host memory between steps;
    the step fetches them to the device, updates them as it would have, and
    writes them back, so what they hold is the same and only where changes.
    """
    rules: LogicalAxisRules | Mapping[str, MeshAxes] = DEFAULT_RULES
    min_shard: int = 2 ** 16
    tolerance: float = 0.02
    host: tuple[str, ...] = ()

    def __post_init__(self):
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(
                f"sharding tolerance must be between 0 and 1, got {self.tolerance}")
        rules = _rule_table(self.rules)
        for name, axes in rules:
            outside = [axis for axis in _mesh_axes(axes) if axis not in PARAMETER_AXES]
            if outside:
                raise ValueError(
                    f"rule {name!r} places a parameter on {outside}; parameters "
                    f"split over {list(PARAMETER_AXES)}, the data and sequence "
                    f"axes split the batch, and the stage axis holds the pipeline")
        object.__setattr__(self, "rules", rules)
        host = tuple(self.host)
        unknown = sorted(set(host) - set(HOST_RESIDENT))
        if unknown:
            raise ValueError(
                f"host names the train-state fields kept in pinned host memory, "
                f"{list(HOST_RESIDENT)}, got {unknown}")
        object.__setattr__(self, "host", host)

    def shardings(self, mesh: Mesh, tree: Any) -> Placement:
        """A NamedSharding per leaf of `tree`, from the declared parameter axes.

        A leaf whose path no module declares takes the largest-divisible-axis
        heuristic, so a model family can be declared at a time. Flax metadata,
        if a caller's own module attached any, is removed here, because the
        state the trainer materialises against this tree carries plain arrays.
        """
        rules = _rule_table(self.rules)
        fsdp_size = mesh.shape[FSDP_AXIS]
        sharded_devices = math.prod(mesh.shape[axis] for axis in PARAMETER_AXES)

        def leaf_sharding(path, value):
            axes = declared_axes(path, value.ndim)
            size = int(np.prod(value.shape, dtype=np.int64))
            if axes is None:
                spec = parameter_spec(value.shape, fsdp_size, self.min_shard)
            elif sharded_devices == 1 or size < self.min_shard:
                spec = P()
            else:
                spec = _mesh_spec(value.shape, axes, rules, mesh)
            return NamedSharding(mesh, spec)

        return jax.tree_util.tree_map_with_path(leaf_sharding, nn.unbox(tree))

    def check(self, params: Variables, shardings: Placement, mesh: Mesh) -> None:
        """Reject a layout that left too much of the model replicated.

        MaxText's guardrail (base.yml sharding_tolerance) against a mesh whose
        parameter axes divide none of the model's dimensions, which the shape
        heuristic otherwise absorbs by replicating everything.

        MaxText measures excess per-chip memory over perfect sharding across
        every parameter. Here the same ratio is taken over the parameters the
        threshold policy meant to shard: anything below min_shard is
        replicated on purpose, so counting it would fire on models that are
        merely small.
        """
        if all(mesh.shape[axis] == 1 for axis in PARAMETER_AXES):
            return

        path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
        shardable_elements = 0
        replicated = []
        for (path, param), sharding in zip(
                path_leaves, jax.tree.leaves(shardings), strict=True):
            elements = int(np.prod(param.shape, dtype=np.int64))
            if elements < self.min_shard:
                continue
            shardable_elements += elements
            if any(axis in _mesh_axes(assignment)
                   for assignment in sharding.spec for axis in PARAMETER_AXES):
                continue
            replicated.append((elements, jax.tree_util.keystr(path), param.shape))

        if not shardable_elements:
            return
        fraction = sum(elements for elements, _, _ in replicated) / shardable_elements
        if fraction <= self.tolerance:
            return

        details = "\n".join(
            f"  {name}: shape={tuple(shape)}, elements={elements}"
            for elements, name, shape in sorted(replicated, reverse=True)[:5])
        raise ValueError(
            f"{fraction:.2%} of shardable parameter elements are replicated, over "
            f"the sharding tolerance of {self.tolerance:.2%}.\n"
            f"Largest replicated parameters:\n{details}")


def batch_shardings(mesh: Mesh | AbstractMesh, batch: Batch) -> Any:
    """A sharding per leaf of `batch`, from the batch spec and the leaf's shape.

    Rows split over the data, expert, fsdp, and tensor axes. A leaf of rank 2 or 3 is a
    sequence per row (token ids, segment ids, positions, encoded tokens), and
    its second dimension splits over the sequence axis when the axis divides
    it; otherwise that dimension stays whole, the way `_mesh_spec` drops a
    name no dimension can split. An image or a video is not a sequence, so
    only its rows split. Placement is all this decides: values never change,
    and until a model constrains its attention to the axis, the sequence
    splits here and gathers there. Only the shape is read, so a leaf that is
    already a global array costs no transfer.
    """
    rows, sequence = BATCH_SPEC
    sequence_size = mesh.shape[SEQUENCE_AXIS]

    def leaf_sharding(leaf):
        shape = np.shape(leaf)
        if not shape:
            return NamedSharding(mesh, P())
        if len(shape) in (2, 3) and shape[1] % sequence_size == 0:
            return NamedSharding(mesh, P(rows, sequence))
        return NamedSharding(mesh, P(rows))

    return jax.tree.map(leaf_sharding, batch)


def shard_batch(mesh: Mesh, batch: Batch) -> Batch:
    """Assemble this process's slice of each array into a globally sharded one.

    A pool assembles one leaf per process, so every process has to hand this
    the same tree. Validity is the one optional token field, and whether a
    process's own rows needed padding is rank-local, so a pool materializes
    it at every `ModelInputs` of the batch that lacks it before the traversal
    below reads the leaves. The schema then depends on the process count and
    the batch's structure and never on which rows this process drew. One
    process changes nothing and keeps the omission the fused attention kernel
    wants.

    The rule is deliberately local. Placement runs on
    `DevicePrefetchIterator`'s worker thread while the step's collectives run
    on the caller's, and a collective issued from here would have to be
    ordered against those across every process. Agreeing which sites actually
    carry validity, the way `agreed_validity` does for a generation request,
    belongs where the caller's own collectives are issued.
    """
    batch = filled_validity(batch) if jax.process_count() > 1 else batch
    return jax.tree.map(
        lambda leaf, sharding: jax.make_array_from_process_local_data(
            sharding, np.asarray(leaf)),
        batch, batch_shardings(mesh, batch))


class DevicePrefetchIterator:
    """Bounded host-to-device read-ahead, exclusively owning its source.

    Use as a context manager or call close, including after a bounded loop.
    At most depth batches are queued plus one being read/placed. Source next,
    checkpoint operations and final close run on the worker. An optional
    thread-safe, nonblocking request_stop hook runs on the closing thread.
    If thread creation itself fails, caller-thread finalization runs before
    re-raising; no worker has touched the source. That synchronous error path
    requires a cooperative finalizer. Active-worker close reports a timeout
    when arbitrary source code cannot be stopped.
    """

    def __init__(self, iterator: Iterator, mesh: Mesh, depth: int = 2,
                 source_state: Optional[bytes] = None):
        if depth <= 0:
            raise ValueError("prefetch depth must be positive")
        self._iterator: Iterator | None = iter(iterator)
        self._source_name = type(self._iterator).__name__
        self._mesh: Mesh | None = mesh
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._start_lock = threading.Lock()
        self._error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self.source_state = source_state
        self._thread = threading.Thread(target=self._prefetch, name="dew-prefetch", daemon=True)
        # No source work may start until the caller owns this object. In
        # particular, an interrupt during restoration must unwind through
        # this iterator's close, never the caller's untransferred-source path.

    def _start(self) -> None:
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
                            request_stop = getattr(self._iterator, "request_stop", None)
                            if request_stop is not None:
                                request_stop()
                        except BaseException as failure:
                            error.add_note(f"Source cancellation failed: {failure!r}")
                        self._prefetch()
                    raise

    def _prefetch(self):
        iterator, mesh = self._iterator, self._mesh
        assert iterator is not None and mesh is not None
        source = iterator if isinstance(iterator, Checkpointable) else None
        batch = state = placed = None
        try:
            if not self._stop.is_set() and self.source_state is not None:
                if source is None:
                    raise TypeError(f"{self._source_name} cannot resume from a saved position")
                saved = self.source_state
                source.set_state(saved if isinstance(source.get_state(), bytes)
                                 else json.loads(saved))
            while not self._stop.is_set():
                batch = next(iterator)
                if self._stop.is_set():
                    break
                state = source.get_state() if source is not None else None
                if source is not None and not isinstance(state, bytes):
                    state = json.dumps(state).encode()
                placed = shard_batch(mesh, batch)
                batch = None
                while not self._stop.is_set():
                    try:
                        self._queue.put((placed, state), timeout=0.05)
                        break
                    except queue.Full:
                        pass
                placed = state = None
        except StopIteration:
            pass
        except BaseException as error:
            self._error = error
        finally:
            batch = placed = state = None
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            except BaseException as error:
                self._cleanup_error = error
            finally:
                self._iterator = self._mesh = None
                # No bound close method or checkpointable alias may retain a
                # Grain pipeline after the worker exits.
                iterator = source = mesh = close = None
                if self._stop.is_set():
                    self._discard()
                self._done.set()

    def _discard(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def close(self, *, timeout: float = 5.0) -> None:
        """Cancel and join, discarding unread batches, not consumed position.

        A timeout leaves cancellation requested; a later close may join again.
        It does not mean the source's in-flight work or buffers were released.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("a prefetch worker cannot close itself")
        if timeout < 0:
            raise ValueError("close timeout must be nonnegative")
        first_stop = not self._stop.is_set()
        self._stop.set()
        error = None
        try:
            request_stop = getattr(self._iterator, "request_stop", None)
            if first_stop and request_stop is not None:
                request_stop()
        except BaseException as failure:
            error = failure
        self._discard()
        self._start()  # Even an unused iterator finalizes on its worker.
        if self._thread.ident is not None:
            self._thread.join(timeout)
        self._discard()
        self._error = None  # Unconsumed speculative errors are discarded too.
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
        try:
            self.close()
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"Prefetch cleanup failed: {error!r}")

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop.is_set():
            raise StopIteration
        self._start()
        while not self._stop.is_set():
            try:
                batch, position = self._queue.get(timeout=0.05)
            except queue.Empty:
                if not self._done.is_set():
                    continue
                # The final publication may race the timed-out get.
                try:
                    batch, position = self._queue.get_nowait()
                except queue.Empty:
                    error, self._error = self._error, None
                    cleanup, self._cleanup_error = self._cleanup_error, None
                    if error is not None:
                        if cleanup is not None:
                            error.add_note(f"Source cleanup failed: {cleanup!r}")
                        raise error
                    if cleanup is not None:
                        raise cleanup
                    raise StopIteration
            if self._stop.is_set():
                break
            self.source_state = position
            return batch
        raise StopIteration
