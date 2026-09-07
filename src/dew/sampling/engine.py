"""A persistent generation engine over the family kernels.

The engine owns what a batch ``generate`` call does not: request admission,
copied and versioned weights, deterministic prefix reuse, row placement in
shared decode states, streaming, cancellation and shutdown. The science stays
in the family: prefill, one advance step and the typed result.
"""

from __future__ import annotations

import functools
import hashlib
import threading
import uuid
from collections import OrderedDict, deque
from collections.abc import Hashable, Iterator, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.typing import ArrayLike

from dew.nn.inputs import ModelInputs
from dew.objectives.base import Variables

P = TypeVar("P", bound="GenerationPlan")
S = TypeVar("S")
O = TypeVar("O")
R = TypeVar("R")
E = TypeVar("E")
S_contra = TypeVar("S_contra", contravariant=True)
O_contra = TypeVar("O_contra", contravariant=True)
R_co = TypeVar("R_co", covariant=True)
E_co = TypeVar("E_co", covariant=True)


class GenerationPlan(Protocol):
    """Static values of one request."""

    @property
    def controls(self) -> Hashable:
        """Stepping controls; requests sharing them may share a decode state."""
        ...

    @property
    def geometry(self) -> Hashable:
        """What the deterministic prefill depends on besides the inputs."""
        ...

    @property
    def steps(self) -> int:
        """Advance calls that exhaust the token budget."""
        ...


class GenerationProgress(Protocol[O_contra, S_contra, R_co, E_co]):
    """Host record of one request between advances."""

    @property
    def bytes(self) -> int: ...

    @property
    def done(self) -> bool: ...

    @property
    def count(self) -> int: ...

    def record(self, outputs: O_contra) -> None: ...

    def events(self, start: int) -> Sequence[E_co]: ...

    def result(self, state: S_contra | None) -> R_co: ...


class GenerationFamily(Protocol[P, S, O, R_co, E_co]):
    """One algorithm's kernels, from host validation to the typed result.

    ``shared_rows`` declares that rows of different requests may live in one
    state because ``advance`` treats rows independently. ``begin`` is
    deterministic: the engine may retain and copy its result.
    """

    shared_rows: ClassVar[bool]

    def prepare(self, model: nn.Module, inputs: ModelInputs, max_new_tokens: int,
                generation: object | None) -> tuple[ModelInputs, P]: ...

    def begin(self, model: nn.Module, variables: Variables, inputs: ModelInputs,
              geometry: Any) -> S: ...

    def advance(self, model: nn.Module, variables: Variables, state: S, keys: jax.Array,
                controls: Any, steps: int) -> tuple[S, O]: ...

    def keys(self, key: jax.Array, rows: int) -> jax.Array: ...

    def progress(self, inputs: ModelInputs, plan: P) -> GenerationProgress[O, S, R_co, E_co]: ...

    def generate(self, model: nn.Module, variables: Variables,
                 inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
                 *, key: jax.Array, generation: object | None = None) -> R_co: ...


def generation_family(model: nn.Module) -> GenerationFamily[Any, Any, Any, Any, Any]:
    """The family a native model generates with, without a source's policy."""
    from dew.nn.diffusion_gemma import DiffusionGemma

    if isinstance(model, DiffusionGemma):
        from dew.diffusion.block import BlockProcess, CanvasFamily

        return CanvasFamily(BlockProcess(canvas_length=model.canvas_length, vocab_size=model.vocab_size))
    from dew.sampling.text import AutoregressiveFamily

    return AutoregressiveFamily()


class CapacityError(RuntimeError):
    """Admission would exceed a configured bound; nothing was accepted."""


@dataclass(frozen=True)
class WeightVersion:
    """An engine-scoped identity of one published, copied weight snapshot."""

    engine: str
    serial: int


@dataclass(frozen=True)
class EngineStats:
    pending: int
    active: int
    allocated_rows: int
    versions: int
    prefix_entries: int
    prefix_bytes: int
    prefix_hits: int
    prefix_misses: int


_TERMINAL = ("done", "cancelled", "error")


class GenerationJob(Generic[R, E]):
    """One accepted request. ``result`` blocks; ``stream`` yields typed events."""

    def __init__(self, engine: Engine[Any, Any], inputs: ModelInputs, plan: GenerationPlan,
                 key: jax.Array, version: WeightVersion, digest: bytes,
                 progress: GenerationProgress[Any, Any, R, E], reservation: int) -> None:
        self._engine = engine
        self.inputs = inputs
        self.plan = plan
        self.key = key
        self.version = version
        self.rows = int(inputs.tokens.shape[0])
        self._digest = digest
        self._progress = progress
        self._reservation = reservation
        self._remaining = plan.steps
        self._changed = threading.Condition(engine._lock)
        self._status = "waiting"
        self._error: BaseException | None = None
        self._result: R | None = None
        self._cancel_requested = False
        self._streaming = False
        # Placement, owned by the controller thread.
        self._cohort: _Cohort | None = None
        self._slots: np.ndarray | None = None
        self._state: Any = None
        self._keys: jax.Array | None = None

    @property
    def status(self) -> str:
        """waiting, active, done, cancelled or error."""
        return self._status

    @property
    def done(self) -> bool:
        return self._status in _TERMINAL

    def cancel(self) -> bool:
        """Stop the request; false once it is terminal.

        A request cancelled before its terminal transition ends with
        CancelledError, including one whose final device step was in flight.
        Events recorded before then stay readable.
        """
        with self._engine._condition:
            if self._status in _TERMINAL:
                return False
            if self._status == "waiting":
                self._engine._waiting.remove(self)
                self._engine._finish(self, "cancelled")
            else:
                self._cancel_requested = True
                self._engine._condition.notify_all()
            return True

    def _outcome(self) -> R:
        if self._error is not None:
            raise self._error
        if self._status == "cancelled":
            raise CancelledError("the generation was cancelled")
        assert self._result is not None
        return self._result

    def result(self, timeout: float | None = None) -> R:
        """The typed result; a timeout leaves the request running."""
        with self._changed:
            if not self._changed.wait_for(lambda: self._status in _TERMINAL, timeout):
                raise TimeoutError("the generation is still active")
            return self._outcome()

    def stream(self) -> Iterator[E]:
        """Events as they are recorded; one consumer per request."""
        with self._changed:
            if self._streaming:
                raise RuntimeError("a request has one stream consumer")
            self._streaming = True
        cursor = 0
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._progress.count > cursor or self._status in _TERMINAL)
                events = list(self._progress.events(cursor))
                cursor += len(events)
                if not events:
                    self._outcome()
                    return
            yield from events


class _Version:
    def __init__(self, token: WeightVersion, variables: Variables) -> None:
        self.token = token
        self.variables = variables
        self.pins = 0


class _Cohort:
    """One shared decode state; each slot holds one row of one request."""

    def __init__(self, key: Hashable, capacity: int, axes: tuple[int, ...], keys: jax.Array) -> None:
        self.key = key
        self.capacity = capacity
        self.axes = axes
        self.state: Any = None
        self.keys = keys
        self.owners: list[GenerationJob[Any, Any] | None] = [None] * capacity
        self.jobs: list[GenerationJob[Any, Any]] = []

    @property
    def used(self) -> int:
        return sum(job.rows for job in self.jobs)


def _bucket(rows: int, limit: int) -> int:
    return min(1 << max(0, (rows - 1).bit_length()), limit)


def _row_index(axis: int, rows: Any) -> tuple[Any, ...]:
    return (slice(None),) * axis + (rows,)


@functools.lru_cache(maxsize=None)
def _placer(axes: tuple[int, ...]):
    def place(stacked, item, rows):
        leaves, treedef = jax.tree.flatten(stacked)
        items = jax.tree.leaves(item)
        return treedef.unflatten([leaf.at[_row_index(axis, rows)].set(value)
                                  for leaf, value, axis in zip(leaves, items, axes)])

    return jax.jit(place, donate_argnums=(0,))


@functools.lru_cache(maxsize=None)
def _expander(axes: tuple[int, ...], capacity: int):
    def expand(item, rows):
        leaves, treedef = jax.tree.flatten(item)
        out = []
        for leaf, axis in zip(leaves, axes):
            shape = list(leaf.shape)
            shape[axis] = capacity
            out.append(jnp.zeros(shape, leaf.dtype).at[_row_index(axis, rows)].set(leaf))
        return treedef.unflatten(out)

    return jax.jit(expand)


@functools.lru_cache(maxsize=None)
def _compactor(axes: tuple[int, ...]):
    """Gather rows into a new state; the index length is the new capacity."""
    def compact(stacked, rows):
        leaves, treedef = jax.tree.flatten(stacked)
        return treedef.unflatten([jnp.take(leaf, rows, axis=axis) for leaf, axis in zip(leaves, axes)])

    return jax.jit(compact)


def _copy(tree):
    return jax.tree.map(lambda leaf: jnp.array(leaf, copy=True), tree)


def _nbytes(tree) -> int:
    return sum(int(leaf.nbytes) for leaf in jax.tree.leaves(tree))


def _signature(tree) -> tuple[Any, ...]:
    return (jax.tree.structure(tree),
            tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in jax.tree.leaves(tree)))


class Engine(Generic[R, E]):
    """Concurrent generation over one native model and its family.

    ``max_batch_size`` bounds the rows allocated to decode states at once;
    a request's rows must fit it. ``max_pending_requests`` bounds accepted
    requests that are not yet placed. ``max_request_bytes`` bounds the
    numeric input and host output arrays of every accepted request.
    ``prefix_cache_bytes`` bounds retained deterministic prefill states.
    ``max_weight_versions`` bounds snapshots kept alive by the current
    version and by requests pinned to older ones. ``steps_per_dispatch``
    advances each state that many steps per device call.
    """

    def __init__(self, source: Any, variables: Variables | None = None, *,
                 family: GenerationFamily[Any, Any, Any, R, E] | None = None,
                 max_batch_size: int, max_pending_requests: int = 64,
                 max_request_bytes: int = 256 * 1024 ** 2, prefix_cache_bytes: int = 0,
                 max_weight_versions: int = 2, steps_per_dispatch: int = 1) -> None:
        from dew.interop.pretrained import Pretrained

        if isinstance(source, Pretrained):
            if variables is not None or family is not None:
                raise ValueError("a loaded source supplies its own variables and family")
            if source.generation_adapter is None:
                raise ValueError("this source has no native generation family")
            model, variables, family = source.model, source.variables, source.generation_adapter
        else:
            model = source
            if not isinstance(model, nn.Module) or variables is None:
                raise TypeError("pass a loaded source, or a native model with its variables")
            if family is None:
                family = generation_family(model)
        if jax.process_count() > 1:
            raise NotImplementedError("the engine schedules one process; pooled processes use generate")
        for name, value in (("max_batch_size", max_batch_size), ("max_weight_versions", max_weight_versions),
                            ("steps_per_dispatch", steps_per_dispatch)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("max_pending_requests", max_pending_requests),
                            ("max_request_bytes", max_request_bytes), ("prefix_cache_bytes", prefix_cache_bytes)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self.model = model
        self.family: GenerationFamily[Any, Any, Any, R, E] = family
        self.max_batch_size = max_batch_size
        self.max_pending_requests = max_pending_requests
        self.max_request_bytes = max_request_bytes
        self.prefix_cache_bytes = prefix_cache_bytes
        self.max_weight_versions = max_weight_versions
        self.steps_per_dispatch = steps_per_dispatch
        self._identity = uuid.uuid4().hex
        self._initial: Variables | None = variables
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._waiting: deque[GenerationJob[R, E]] = deque()
        self._active: list[GenerationJob[R, E]] = []
        self._versions: dict[WeightVersion, _Version] = {}
        self._current: WeightVersion | None = None
        self._serial = 0
        self._layout: Any = None
        self._reserved = 0
        self._prefixes: OrderedDict[tuple[Any, ...], tuple[Any, int]] = OrderedDict()
        self._prefix_bytes = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        self._layouts: dict[tuple[Any, ...], tuple[tuple[int, ...], tuple[Any, ...]]] = {}
        self._cohorts: dict[Hashable, _Cohort] = {}
        self._exclusive: list[GenerationJob[R, E]] = []
        self._allocated = 0
        self._closing = False
        self._closed = False
        self._controller: threading.Thread | None = None
        self._begin = jax.jit(family.begin, static_argnames=("model", "geometry"))
        self._advance = jax.jit(family.advance, static_argnames=("model", "controls", "steps"),
                                donate_argnums=(2,))
        self._keys = jax.jit(family.keys, static_argnames=("rows",))

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> Engine[R, E]:
        """Snapshot the initial weights and start the controller."""
        with self._condition:
            if self._controller is not None:
                raise RuntimeError("the engine was already started")
            assert self._initial is not None
            initial, self._initial = self._initial, None
            self._layout = _signature(initial)
            self._publish(initial)
            self._controller = threading.Thread(target=self._run, name="dew-engine", daemon=True)
            self._controller.start()
        return self

    def __enter__(self) -> Engine[R, E]:
        return self.start()

    def __exit__(self, kind, error, traceback) -> None:
        self.close(cancel=error is not None)

    def close(self, *, cancel: bool = False, timeout: float | None = None) -> None:
        """Stop admission, finish or cancel accepted work, then release resources."""
        with self._condition:
            self._closing = True
            if cancel:
                for job in list(self._waiting):
                    self._waiting.remove(job)
                    self._finish(job, "cancelled")
                for job in self._active:
                    job._cancel_requested = True
            self._condition.notify_all()
            controller = self._controller
        if controller is not None:
            controller.join(timeout)
            if controller.is_alive():
                raise TimeoutError("the controller still owns device work")
        with self._condition:
            for version in self._versions.values():
                for leaf in jax.tree.leaves(version.variables):
                    if isinstance(leaf, jax.Array):
                        leaf.delete()
            self._versions.clear()
            self._prefixes.clear()
            self._prefix_bytes = 0
            self._cohorts.clear()
            self._exclusive.clear()
            self._closed = True

    # -- weights -------------------------------------------------------------------

    @property
    def version(self) -> WeightVersion:
        """The version new requests pin by default."""
        with self._condition:
            if self._current is None:
                raise RuntimeError("the engine has not been started")
            return self._current

    def publish(self, variables: Variables) -> WeightVersion:
        """Copy a new weight snapshot; it becomes the default for later requests.

        The caller may donate or delete its arrays once this returns. Requests
        already accepted keep their pinned version.
        """
        with self._condition:
            if self._closing or self._controller is None:
                raise RuntimeError("the engine is not open for publication")
            if _signature(variables) != self._layout:
                raise ValueError("the variables tree does not match the engine's model")
            self._retire()
            if len(self._versions) >= self.max_weight_versions:
                raise CapacityError("every retained weight version is pinned by accepted requests")
            return self._publish(variables)

    def _publish(self, variables: Variables) -> WeightVersion:
        snapshot = jax.tree.map(
            lambda leaf: jax.device_put(leaf, leaf.sharding if isinstance(leaf, jax.Array) else None,
                                        may_alias=False), variables)
        jax.block_until_ready(snapshot)
        token = WeightVersion(self._identity, self._serial)
        self._serial += 1
        self._versions[token] = _Version(token, snapshot)
        self._current = token
        return token

    def _retire(self) -> None:
        retired = [token for token, version in self._versions.items()
                   if token != self._current and version.pins == 0]
        for token in retired:
            for leaf in jax.tree.leaves(self._versions.pop(token).variables):
                if isinstance(leaf, jax.Array):
                    leaf.delete()
        if retired:
            serials = {token.serial for token in retired}
            for identity in [key for key in self._prefixes if key[0] in serials]:
                self._prefix_bytes -= self._prefixes.pop(identity)[1]

    # -- admission -----------------------------------------------------------------

    def submit(self, inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
               *, key: jax.Array, generation: object | None = None,
               version: WeightVersion | None = None) -> GenerationJob[R, E]:
        """Validate and accept one request; the controller runs it."""
        canonical = ModelInputs.from_value(inputs)
        random_key = jax.random.wrap_key_data(jax.random.key_data(key), impl=jax.random.key_impl(key))
        if random_key.shape != ():
            raise ValueError("key must be a single JAX PRNG key")
        prepared, plan = self.family.prepare(self.model, canonical, max_new_tokens, generation)
        rows = int(prepared.tokens.shape[0])
        if rows > self.max_batch_size:
            raise CapacityError(f"{rows} rows exceed max_batch_size {self.max_batch_size}")
        digest = hashlib.sha256(repr(plan.geometry).encode())
        for path, leaf in jax.tree_util.tree_leaves_with_path(prepared):
            value = np.asarray(leaf)
            digest.update(repr((jax.tree_util.keystr(path), value.shape, str(value.dtype))).encode())
            digest.update(value.tobytes())
        progress = self.family.progress(prepared, plan)
        reservation = _nbytes(prepared) + progress.bytes
        with self._condition:
            if self._closing or self._controller is None:
                raise RuntimeError("the engine is not accepting requests")
            if len(self._waiting) >= self.max_pending_requests:
                raise CapacityError("the pending request bound is reached")
            if self._reserved + reservation > self.max_request_bytes:
                raise CapacityError("the request byte bound is reached")
            token = self._current if version is None else version
            if token not in self._versions:
                raise ValueError("the weight version is unknown, retired, or from another engine")
            job = GenerationJob[R, E](self, prepared, plan, random_key, token, digest.digest(),
                                      progress, reservation)
            self._versions[token].pins += 1
            self._reserved += reservation
            self._waiting.append(job)
            self._condition.notify_all()
            return job

    def stats(self) -> EngineStats:
        with self._condition:
            return EngineStats(len(self._waiting), len(self._active), self._allocated, len(self._versions),
                               len(self._prefixes), self._prefix_bytes, self._prefix_hits, self._prefix_misses)

    # -- controller ----------------------------------------------------------------

    def _finish(self, job: GenerationJob[R, E], status: str, error: BaseException | None = None,
                result: R | None = None) -> None:
        """Terminal transition; the caller holds the condition."""
        assert job._status not in _TERMINAL
        job._status, job._error, job._result = status, error, result
        job._state, job._keys, job._cohort, job._slots = None, None, None, None
        if job in self._active:
            self._active.remove(job)
        self._reserved -= job._reservation
        self._versions[job.version].pins -= 1
        self._retire()
        job._changed.notify_all()
        self._condition.notify_all()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._waiting or self._active or self._closing)
                    if self._closing and not self._waiting and not self._active:
                        return
                    head = self._waiting[0] if self._waiting else None
                if head is not None and self.family.shared_rows and head.plan.steps > 0:
                    try:
                        self._axes(head)
                    except Exception as error:
                        with self._condition:
                            if head in self._waiting:
                                self._waiting.remove(head)
                                self._finish(head, "error", error)
                        continue
                with self._condition:
                    starting = self._admissible()
                for job in starting:
                    self._start(job)
                self._dispatch()
        except BaseException as error:
            with self._condition:
                for job in list(self._waiting) + list(self._active):
                    if job in self._waiting:
                        self._waiting.remove(job)
                    self._finish(job, "error", error)
                self._closing = True
                self._condition.notify_all()

    def _admissible(self) -> list[GenerationJob[R, E]]:
        """Pop requests in order while the row bound admits them; caller holds the condition."""
        starting: list[GenerationJob[R, E]] = []
        allocated = self._allocated
        projected: dict[Hashable, tuple[int, int]] = {}
        while self._waiting:
            job = self._waiting[0]
            if job.plan.steps == 0:
                self._waiting.popleft()
                self._finish(job, "done", result=job._progress.result(None))
                continue
            if self.family.shared_rows:
                if self._layout_key(job) not in self._layouts:
                    break
                key = self._cohort_key(job)
                cohort = self._cohorts.get(key)
                capacity, used = projected.get(key, (cohort.capacity, cohort.used) if cohort else (0, 0))
                if used + job.rows > self.max_batch_size:
                    break
                needed = _bucket(used + job.rows, self.max_batch_size)
                growth = max(needed - capacity, 0)
                projected[key] = (needed, used + job.rows)
            else:
                growth = job.rows
            if allocated + growth > self.max_batch_size:
                break
            allocated += growth
            self._waiting.popleft()
            job._status = "active"
            self._active.append(job)
            starting.append(job)
        return starting

    def _layout_key(self, job: GenerationJob[R, E]) -> tuple[Any, ...]:
        return (_signature(job.inputs), repr(job.plan.geometry))

    def _cohort_key(self, job: GenerationJob[R, E]) -> Hashable:
        return (job.version, job.plan.controls, str(jax.random.key_impl(job.key)),
                self._layouts[self._layout_key(job)][1])

    def _prefix(self, job: GenerationJob[R, E]) -> tuple[Any, bool]:
        """The deterministic prefill state and whether the cache also holds it."""
        identity = (job.version.serial, job._digest)
        with self._condition:
            entry = self._prefixes.get(identity)
            if entry is not None:
                self._prefixes.move_to_end(identity)
                self._prefix_hits += 1
                return entry[0], True
            self._prefix_misses += 1
        variables = self._versions[job.version].variables
        state = self._begin(self.model, variables, job.inputs, job.plan.geometry)
        jax.block_until_ready(state)
        size = _nbytes(state)
        retained = False
        with self._condition:
            if 0 < size <= self.prefix_cache_bytes:
                while self._prefixes and self._prefix_bytes + size > self.prefix_cache_bytes:
                    self._prefix_bytes -= self._prefixes.popitem(last=False)[1][1]
                self._prefixes[identity] = state, size
                self._prefix_bytes += size
                retained = True
        return state, retained

    def _start(self, job: GenerationJob[R, E]) -> None:
        try:
            state, shared = self._prefix(job)
            if self.family.shared_rows:
                self._place(job, state)
            else:
                job._state = _copy(state) if shared else state
                job._keys = self._keys(job.key, job.rows)
                self._exclusive.append(job)
                self._allocated += job.rows
        except Exception as error:
            with self._condition:
                self._finish(job, "error", error)

    def _place(self, job: GenerationJob[R, E], state: Any) -> None:
        axes, _ = self._axes(job)
        key = self._cohort_key(job)
        cohort = self._cohorts.get(key)
        keys = self._keys(job.key, job.rows)
        if cohort is None:
            capacity = _bucket(job.rows, self.max_batch_size)
            slots = np.arange(job.rows)
            cohort = _Cohort(key, capacity, axes, _expander((0,), capacity)(keys, jnp.asarray(slots)))
            cohort.state = _expander(axes, capacity)(state, jnp.asarray(slots))
            self._cohorts[key] = cohort
            self._allocated += capacity
        else:
            if cohort.used + job.rows > cohort.capacity:
                self._resize(cohort, _bucket(cohort.used + job.rows, self.max_batch_size))
            slots = np.flatnonzero([owner is None for owner in cohort.owners])[:job.rows]
            assert len(slots) == job.rows
            cohort.state = _placer(cohort.axes)(cohort.state, state, jnp.asarray(slots))
            cohort.keys = cohort.keys.at[jnp.asarray(slots)].set(keys)
        for slot in slots:
            cohort.owners[slot] = job
        cohort.jobs.append(job)
        job._cohort, job._slots = cohort, slots

    def _axes(self, job: GenerationJob[R, E]) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        """Row axis of every state leaf, by widening the batch in abstract evaluation."""
        signature = self._layout_key(job)
        cached = self._layouts.get(signature)
        if cached is not None:
            return cached
        variables = self._versions[job.version].variables

        def widened(extra: int):
            wide = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct((leaf.shape[0] + extra,) + leaf.shape[1:], leaf.dtype),
                                job.inputs)
            return jax.eval_shape(lambda v, x: self.family.begin(self.model, v, x, job.plan.geometry),
                                  variables, wide)

        narrow, wide = widened(1), widened(2)
        axes = []
        shapes = []
        for (path, small), large in zip(jax.tree_util.tree_leaves_with_path(narrow), jax.tree.leaves(wide)):
            candidates = [axis for axis, (a, b) in enumerate(zip(small.shape, large.shape)) if b == a + 1]
            if len(candidates) != 1 or len(small.shape) != len(large.shape):
                raise TypeError(f"state leaf {jax.tree_util.keystr(path)} has no unique row axis; "
                                "rows of separate requests cannot share this state")
            axis = candidates[0]
            axes.append(axis)
            shapes.append((tuple(small.shape[:axis] + small.shape[axis + 1:]), str(small.dtype)))
        result = tuple(axes), (jax.tree.structure(narrow), tuple(shapes))
        self._layouts[signature] = result
        return result

    def _resize(self, cohort: _Cohort, capacity: int) -> None:
        live = [slot for slot, owner in enumerate(cohort.owners) if owner is not None]
        # Unowned slots copy a live row; nothing reads them before placement.
        rows = np.asarray((live + [live[0]] * capacity)[:capacity], np.int32)
        cohort.state = _compactor(cohort.axes)(cohort.state, jnp.asarray(rows))
        cohort.keys = cohort.keys[jnp.asarray(rows)]
        owners: list[GenerationJob[Any, Any] | None] = [None] * capacity
        for new, old in enumerate(live):
            owners[new] = cohort.owners[old]
        for job in cohort.jobs:
            job._slots = np.asarray([new for new, old in enumerate(live) if cohort.owners[old] is job])
        cohort.owners = owners
        self._allocated += capacity - cohort.capacity
        cohort.capacity = capacity

    def _release(self, job: GenerationJob[R, E]) -> None:
        """Free the rows of a request that left; controller-private bookkeeping."""
        cohort = job._cohort
        if cohort is not None:
            assert job._slots is not None
            for slot in job._slots:
                cohort.owners[slot] = None
            cohort.jobs.remove(job)
            if not cohort.jobs and self._cohorts.pop(cohort.key, None) is not None:
                self._allocated -= cohort.capacity
        elif job in self._exclusive:
            self._exclusive.remove(job)
            self._allocated -= job.rows

    def _dispatch(self) -> None:
        for cohort in list(self._cohorts.values()):
            with self._condition:
                for job in list(cohort.jobs):
                    if job._cancel_requested:
                        self._release(job)
                        self._finish(job, "cancelled")
            if not cohort.jobs:
                if self._cohorts.pop(cohort.key, None) is not None:
                    self._allocated -= cohort.capacity
                continue
            wanted = _bucket(cohort.used, self.max_batch_size)
            if wanted < cohort.capacity:
                self._resize(cohort, wanted)
            steps = min([self.steps_per_dispatch] + [job._remaining for job in cohort.jobs])
            variables = self._versions[cohort.jobs[0].version].variables
            try:
                cohort.state, outputs = self._advance(self.model, variables, cohort.state, cohort.keys,
                                                      cohort.jobs[0].plan.controls, steps)
                outputs = jax.device_get(outputs)
            except Exception as error:
                with self._condition:
                    for job in list(cohort.jobs):
                        self._release(job)
                        self._finish(job, "error", error)
                continue
            with self._condition:
                for job in list(cohort.jobs):
                    assert job._slots is not None
                    job._progress.record(jax.tree.map(lambda leaf: leaf[:, job._slots], outputs))
                    job._remaining -= steps
                    if job._cancel_requested:
                        self._release(job)
                        self._finish(job, "cancelled")
                    elif job._progress.done or job._remaining == 0:
                        self._release(job)
                        self._finish(job, "done", result=job._progress.result(None))
                    else:
                        job._changed.notify_all()
        for job in list(self._exclusive):
            with self._condition:
                if job._cancel_requested:
                    self._release(job)
                    self._finish(job, "cancelled")
                    continue
            steps = min(self.steps_per_dispatch, job._remaining)
            variables = self._versions[job.version].variables
            try:
                job._state, outputs = self._advance(self.model, variables, job._state, job._keys,
                                                    job.plan.controls, steps)
                outputs = jax.device_get(outputs)
            except Exception as error:
                with self._condition:
                    self._release(job)
                    self._finish(job, "error", error)
                continue
            with self._condition:
                job._progress.record(outputs)
                job._remaining -= steps
                if job._cancel_requested:
                    self._release(job)
                    self._finish(job, "cancelled")
                elif job._progress.done or job._remaining == 0:
                    result = job._progress.result(job._state)
                    self._release(job)
                    self._finish(job, "done", result=result)
                else:
                    job._changed.notify_all()
