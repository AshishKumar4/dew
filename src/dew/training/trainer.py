"""The trainer: mesh, compiled step, EMA, checkpoints and logging.

What is learned is up to the objective (`dew.objectives.base`). The trainer
creates the objective's variables on the mesh, compiles one step over the
global batch, and updates the EMA copy once per optimizer update. Side effects
go to the objects it was given: a `Checkpoints` for disk and a `Tracker` for
numbers and artifacts.

Constructing a trainer opens nothing; the mesh, the compiled step and those
objects' resources are created in `fit`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import functools
import logging
import math
import sys
import time
import types
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from typing_extensions import TypeVar as DefaultTypeVar

from dew.checkpoints import Checkpoints, Ranking
from dew.coordination import agree_process_phase, agreed
from dew.data.dataset import Checkpointable, Closeable, DataPartition, Dataset, RampedStream, Reader, rows_of
from dew.nn.kernels.generation import measured_kernel
from dew.nn.sharding import (
    BATCH_AXES,
    SEQUENCE_AXIS,
    STAGE_AXIS,
    TENSOR_AXIS,
    LayoutRefused,
    Link,
    LogicalAxes,
    Schedule,
    boxed,
    boxed_axes,
    measured_links,
    pipeline_microbatches,
)
from dew.objectives.base import (
    FROZEN,
    Aux,
    Batch,
    Initializer,
    Metric,
    Objective,
    Ratio,
    Shown,
    Step,
    TrainingScalar,
    Variables,
    select,
)
from dew.records import JSON, boolean, record
from dew.telemetry import profile as telemetry_profile
from dew.telemetry.instrumentation import compiled_flops, model_flops_utilization, peak_flops
from dew.telemetry.profile import region
from dew.telemetry.records import (
    AxisLink,
    CheckpointRequested,
    FitEnded,
    FitStarted,
    ProfileWindow as ProfileWindowRecord,
    Record,
    StepCompiled,
)
from dew.training.display import TrainingDisplay
from dew.training.distributed import (
    DevicePrefetchIterator,
    Layout,
    MeshSpec,
    Placement,
    batch_divisor,
    batch_shardings,
    link_bandwidth,
    prefetched_bytes,
    shard_batch,
)
from dew.training.evaluation import EvalSuite, Evaluation
from dew.training.memory import (
    climb_to,
    compiled_if_it_fits,
    device_tokens,
    fitting_default,
    head_tile_of,
    recompute_more,
    recompute_record,
    refuse_wide_floats,
    split_share,
    step_compiler_options,
    step_fits,
)
from dew.training.narrow import narrowed, narrowed_paths
from dew.training.rungs import keep_rung, recorded_rung
from dew.training.runtime import Preempted, PreemptionNotice
from dew.training.selection import Best
from dew.training.state import Accumulation, TrainState
from dew.training.tracker import Tracker
from dew.training.transaction import Transaction, compact_qk, with_ema

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dew.config import OptimConfig
    from dew.telemetry.profile import Profiler

# Consecutive non-finite losses that stop a run.
BAD_LOSS_STEPS = 5

StepFn = Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]
"""A compiled step's body: the state and the global batch in, the new state,
the loss and the objective's report out."""

CompiledStep = Callable[
    [TrainState, Batch],
    tuple[TrainState, jax.Array, dict[str, jax.Array], jax.Array, jax.Array]]
"""A call returns the stepped state, the scalar loss, the objective's whole
report (`Aux.metrics`), whether that loss was finite, and whether the
microbatch was accepted."""

Loss = DefaultTypeVar("Loss", default=Ratio | jax.Array | float)
Effects = DefaultTypeVar("Effects", default=None)

Shapes = tuple[tuple[int, ...], ...]
"""A batch's leaf shapes in tree order, the key a compiled step is held
under. A fixed batch has one; a ramped batch has one per stage of its ramp,
each compiled on the first step that reads it."""


def batch_shapes(batch: Batch) -> Shapes:
    """List a batch's leaf shapes in tree order."""
    return tuple(np.shape(leaf) for leaf in jax.tree.leaves(batch))


class Rollout(Protocol):
    """Produces a batch on the host, before the compiled step reads it.

    Sampling has side effects and cannot be traced, so it runs outside `jit`. The
    trainer calls the rollout with the state, the prefetched batch and a key
    folded from the run key and the step, then reshards the result with
    `shard_batch`. The returned batch must hold arrays of fixed shapes, so the
    step still compiles once per run.

    A rollout may have `metrics`, a mapping of names to floats about its latest
    call (rewards, lag, truncation); each logging interval sends them to the
    tracker and the display as `rollout/<name>`. It may declare how the display
    shows them in `shown`, as an objective does.
    """

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...


@dataclasses.dataclass(frozen=True)
class ProfileWindow:
    """A request for one profiler window per fit.

    It traces `steps` steps into `directory` after `warmup` steps have run, so the
    trace shows the loop and not the compile.

    `dew.Profiler` is the other way to capture a trace, as a context manager
    around any code; a fit refuses to schedule a window inside one. The loop
    reports the window it wrote as the `ProfileWindow` record of
    `dew.telemetry.records`.
    """
    directory: str
    steps: int
    warmup: int = 2


Book = tuple[jax.Array, jax.Array, jax.Array]
"""The loop's device-side counters: the interval's summed loss, the current
streak of non-finite losses, and the longest streak since the last check."""


def fresh_book(loss: jax.Array) -> Book:
    """Return the counters a fresh logging interval starts from, on the device the step returns `loss` on.

    Counters on any other device would move to the step's devices at every count,
    and `bookkeep` would compile again.
    """
    dtype = jnp.promote_types(loss.dtype, jnp.float32)
    zeros = (np.zeros((), dtype), np.zeros((), np.int32), np.zeros((), np.int32))
    return jax.device_put(zeros, loss.sharding)


@jax.jit
def bookkeep(book: Book, loss: jax.Array, finite: jax.Array) -> Book:
    """Advance the loop's counters for one step in a single dispatch.

    Five eager ops would cost more: 176 against 37 us a step on a CPU.
    """
    interval_loss, bad_run, worst_bad_run = book
    bad_run = jnp.where(finite, 0, bad_run + 1)
    dtype = jnp.promote_types(loss.dtype, jnp.float32)
    return interval_loss + loss.astype(dtype), bad_run, jnp.maximum(worst_bad_run, bad_run)


def learning_rate(opt_state: optax.OptState) -> jax.typing.ArrayLike | None:
    """Return the learning rate stored in `opt_state`, if the optimizer stores one.

    Optax exposes a schedule's value through `optax.inject_hyperparams`, which
    keeps it in the state as a scalar the caller can read. The result is None when
    the optimizer closes over the rate, or when parameter groups hold several
    rates.
    """
    injected = (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState)
    rates = [node.hyperparams["learning_rate"]
             for node in jax.tree.leaves(opt_state, is_leaf=lambda node: isinstance(node, injected))
             if isinstance(node, injected) and "learning_rate" in node.hyperparams]
    return rates[0] if len(rates) == 1 else None


# How the display shows the metrics the trainer itself logs; an objective,
# a rollout and a validation metric declare their own (`Shown`).
TRAINER_SHOWN = {"loss": Shown(better="lower"),
                 "learning_rate": Shown(group="optimizer"), "loss_scale": Shown(group="optimizer"),
                 "accepted": Shown(percent=True, group="optimizer"),
                 "step_time_ms": Shown(better="lower", group="throughput"),
                 "samples_per_sec": Shown(better="higher", group="throughput"),
                 "mfu": Shown(better="higher", percent=True, group="throughput"),
                 "rollout_seconds": Shown(better="lower", group="throughput")}


def goodput(wall: float, first_step: float | None, other: float) -> dict[str, float]:
    """Compute the two goodput numbers from MaxText's report that need no cluster telemetry.

    `first_step` is the time from the start of `fit` to the first step's result:
    the placement or restore, the first batch, the compile and the step itself.
    It is None when no step ran. `other` is the time spent outside steps after
    that: evaluations, checkpoint writes and the wait for them at the end.

    The step fraction is what is left of `wall`. A step's own data stall therefore
    counts as step time, as it does in MaxText's start-to-start step time.
    """
    numbers = {}
    if first_step is not None:
        numbers["goodput/time_to_first_step_s"] = first_step
    steps = wall - (wall if first_step is None else first_step) - other
    numbers["goodput/step_fraction"] = max(steps, 0.0) / wall if wall > 0 else 0.0
    return numbers


def _rollout_metrics(rollout: Rollout) -> Mapping[str, float]:
    """The rollout's report of its latest call. A rollout is any callable,
    and a plain function reports nothing, so the trainer reads `metrics` at
    this boundary."""
    return getattr(rollout, 'metrics', {})


def _reported(rollout: Rollout | None, metrics: Sequence[Metric]) -> dict[str, Shown]:
    """How the display shows what the rollout and the validation metrics
    report. Both may declare `shown`, which their protocols leave optional,
    so the trainer reads it at this boundary."""
    return {
        **getattr(rollout, "shown", {}),
        **{metric.name: shown for metric in metrics if (shown := getattr(metric, "shown", None)) is not None},
    }


@dataclasses.dataclass(frozen=True)
class Plateau:
    """Stops training after `evals` eligible evaluations without an improvement larger than `min_delta`."""
    metric: Metric | TrainingScalar | types.MethodType
    evals: int = 5
    min_delta: float = 0.0
    mode: Literal['min', 'max'] | None = None
    split: str | None = None

    def __post_init__(self):
        if self.evals < 1 or self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError("Plateau needs evals >= 1 and min_delta >= 0")


class _MetricValues[Statistics, Additions](Mapping):
    def __init__(
        self,
        metrics: Sequence[Metric],
        scores: Mapping[str, float],
        objective: Objective[Statistics, Additions],
    ):
        self.objective = objective
        self.metrics = tuple(metrics) + tuple(objective.scalars[name] for name in ('loss', *objective.shown)
                                              if name == 'loss' or f'train/{name}' in scores)
        self.scores = scores

    def __getitem__(self, metric):
        split = 'val'
        if isinstance(metric, tuple):
            split, metric = metric
        if isinstance(metric, TrainingScalar):
            if metric.owner is not self.objective:
                raise KeyError("training score belongs to another objective")
            return self.scores[f'train/{metric.name}']
        if (
            isinstance(metric, types.MethodType)
            and metric.__name__ == "loss"
            and metric.__self__ is self.objective
        ):
            return self.scores['train/loss']
        if isinstance(metric, str):
            if any(metric == f'{name}/{declared.name}' for name in {key.split('/')[0] for key in self.scores}
                   for declared in self.metrics):
                raise KeyError("index declared evaluation scores by metric object")
            return self.scores[metric]
        for declared in self.metrics:
            if declared is metric:
                return self.scores[f'{split}/{declared.name}']
        raise KeyError("the score refers to a metric not passed to fit")

    def __iter__(self):
        for name in self.scores:
            split, _, label = name.partition('/')
            metric = next((metric for metric in self.metrics if metric.name == label), None)
            yield (
                name
                if metric is None
                else metric
                if split == "val" or isinstance(metric, TrainingScalar)
                else (split, metric)
            )

    def __len__(self):
        return len(self.scores)


def _suites(dataset: Dataset, validation: Mapping[str, Reader | EvalSuite] | None,
            metrics: Sequence[Metric], eval_every: int | None) -> dict[str, EvalSuite]:
    """Every split `fit` validates, by name, with its metrics and cadence
    resolved: `dataset.val` alone is the one suite `val`."""
    if validation is None:
        # `val` is read only when a pass is scheduled; a dataset need not hold one.
        validation = {"val": dataset.val} if (eval_every or metrics) and dataset.val is not None else {}
    elif not validation:
        raise ValueError("validation must contain a split")
    suites = {}
    for name, entry in validation.items():
        suite = entry if isinstance(entry, EvalSuite) else EvalSuite(entry)
        suites[name] = EvalSuite(suite.data, tuple(suite.metrics) or tuple(metrics),
                                 eval_every if suite.every is None else suite.every)
    return suites


@dataclasses.dataclass(frozen=True)
class _FitPlan:
    """What one `Trainer.fit` call was asked for."""

    dataset: Dataset
    steps: int
    log_every: int
    eval_every: int | None
    checkpoint_every: int | datetime.timedelta | None
    local_every: int | None
    metrics: Sequence[Metric]
    preview: bool
    best: tuple[Best, ...] = ()
    stop: Plateau | None = None
    suites: Mapping[str, EvalSuite] = dataclasses.field(default_factory=dict)
    """Every validation split by name, its metrics and cadence resolved."""

    @property
    def validation(self) -> bool:
        return bool(self.suites)

    @property
    def scored(self) -> tuple[Metric, ...]:
        """Fit's metrics and every suite's, each once."""
        every = (*self.metrics, *(metric for suite in self.suites.values() for metric in suite.metrics))
        return tuple({metric.name: metric for metric in every}.values())

    @property
    def cadenced(self) -> bool:
        """Whether any evaluation runs before the end."""
        return bool(self.eval_every or any(suite.every for suite in self.suites.values()))

    def due(self, step: int) -> tuple[str, ...]:
        """The suites whose own cadence falls on `step`."""
        return tuple(name for name, suite in self.suites.items() if suite.every and step % suite.every == 0)


@dataclasses.dataclass
class _FitRun:
    """One `Trainer.fit` call's progress: what its loop carries from step
    to step and what its cleanup reads however the run ends."""

    started: float
    current: int = 0
    loss: jax.Array | None = None
    evaluation: Evaluation | None = None
    last_checkpoint: float = 0.0
    stop_control: dict = dataclasses.field(default_factory=dict)
    stopped: bool = False
    training: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    source: Iterator[Batch] | None = None
    train: DevicePrefetchIterator | None = None
    tracing: bool = False
    traced: int = 0
    other: float = 0.0
    first_step: float | None = None
    preempted: int | None = None
    notice: PreemptionNotice | None = None


@dataclasses.dataclass
class _Interval:
    """What `fit` sums between logs and between checkpoints.

    `book` is the loss summed since the last checkpoint and both bad-loss
    counters, on device, so the loop never blocks on a result, and they move
    together in one dispatch; the host reads them at the logging cadence.
    The first step places them where its loss is, and `fresh` keeps them at
    zero there, for a counter's restart to reuse rather than build again.
    `steps` counts since the last checkpoint, the rest since the last log.
    The records and FLOPs are summed per step rather than taken off the
    dataset's batch, so a ramped interval reports the records it read and
    their FLOPs. `rollout_seconds` is the time spent sampling, logged under
    train/rollout_seconds when a rollout is set."""

    last_log_time: float
    last_saved: int | None
    book: Book | None = None
    fresh: Book | None = None
    steps: int = 0
    since_log: int = 0
    samples: int = 0
    flops: float | None = 0.0
    rollout_seconds: float = 0.0

    def count(self, batch: Batch, flops: float | None, loss: jax.Array, finite: jax.Array) -> None:
        """Add one step: its records, its FLOPs and its loss."""
        self.since_log += 1
        self.steps += 1
        self.samples += rows_of(batch)
        self.flops = None if self.flops is None or flops is None else self.flops + flops
        if self.book is None:
            self.book = self.fresh = fresh_book(loss)
        self.book = bookkeep(self.book, loss, finite)

    def check_finite(self, step: int, display: TrainingDisplay) -> None:
        """Raise RuntimeError once the loss has been non-finite for
        BAD_LOSS_STEPS steps, and start the longest streak over.

        Deferred to the logging cadence so the step loop never synchronises;
        detection is late by at most that many steps, never missed.
        """
        if self.book is None or self.fresh is None:
            return
        loss, bad_run, worst_bad_run = self.book
        streak = int(jax.device_get(worst_bad_run))
        if streak >= BAD_LOSS_STEPS:
            raise RuntimeError(
                f"Loss has been non-finite for {streak} consecutive steps "
                f"ending near step {step}, stopping")
        if streak:
            display.note(f"Non-finite loss for {streak} step(s) before {step}", style="red")
        self.book = (loss, bad_run, self.fresh[2])

    def logged(self, now: float) -> None:
        """Start the next logging interval at `now`."""
        self.last_log_time, self.since_log, self.samples = now, 0, 0
        self.flops, self.rollout_seconds = 0.0, 0.0

    def saved(self, step: int) -> None:
        """Start the next checkpoint interval after the one saved at `step`."""
        self.last_saved, self.steps = step, 0
        if self.book is not None and self.fresh is not None:
            _, bad_run, worst_bad_run = self.book
            self.book = (self.fresh[0], bad_run, worst_bad_run)


_DEFAULT_MESH = MeshSpec()
_DEFAULT_LAYOUT = Layout()


class Trainer(Generic[Loss, Effects]):
    """Runs an `Objective`: gradients, sharding, EMA, checkpoints and logging."""

    def __init__(
        self,
        objective: Objective[Loss, Effects],
        optimizer: optax.GradientTransformation | OptimConfig,
        *,
        key: int | jax.Array,
        mesh: MeshSpec = _DEFAULT_MESH,
        layout: Layout = _DEFAULT_LAYOUT,
        accumulation: int = 1,
        dynamic_scale: bool = False,
        checkpoints: Checkpoints | None = None,
        tracker: Tracker | None = None,
        step: Callable[[Objective[Loss, Effects], optax.GradientTransformation], StepFn] | None = None,
        rollout: Rollout | None = None,
        profile: ProfileWindow | None = None,
    ):
        """Hold everything a run needs, without opening any of it.

        The mesh, the compiled step and the capabilities' resources come into
        being in `fit`, so constructing a Trainer allocates nothing.

        `accumulation` is how many microbatches pool into one optimizer commit. `optimizer`
        is an optax transformation, or an `OptimConfig` each `fit` builds over its `steps //
        accumulation` updates; the run steps the objective's (`Objective.optimizer`), made from
        it. `step` replaces the built-in transaction. A custom step then owns the clocks, the
        scaler, the EMA and the mutable writes, and the compiled wrapper owns only the
        attempted-step counter. `rollout` runs once per batch read, before the step and outside
        replay. `layout` and `mesh` say where the state lives.
        """
        if accumulation < 1:
            raise ValueError(f"accumulation must be at least 1, got {accumulation}")
        if step is not None and "variables" in layout.host:
            raise ValueError(
                "Parameter-streamed training requires the trainer objective transaction; "
                "custom steps own their execution"
            )
        self.objective = objective
        built = isinstance(optimizer, optax.GradientTransformation)
        self._config = None if built else optimizer
        self._optimizer = objective.optimizer(optimizer, accumulation=accumulation) if built else None
        from dew.nn.inputs import request_key
        self.seed = int(key) if isinstance(key, (int, np.integer)) and not isinstance(key, bool) else None
        self.key = request_key(key)
        self.mesh = mesh
        self.layout = layout
        self.accumulation = accumulation
        self.dynamic_scale = dynamic_scale
        self.checkpoints = checkpoints
        self.tracker = tracker
        self.step = step
        self.rollout = rollout
        self.profile = profile
        # Set by `compile`, for the batch shape it was called with. A ramped
        # run has one value per stage; `fit` keeps them beside each step. The
        # program is the step as handed to XLA, before GSPMD partitions it,
        # and the executable the step as compiled, whose memory analysis a
        # benchmark reads.
        self.flops_per_step = None
        self.program: jax.stages.Lowered | None = None
        self.executable: jax.stages.Compiled | None = None
        # The links the last compiled step decided its projections by, which
        # say whether one spread over each axis (`dew.nn.sharding.Link`), and
        # each mesh axis's measured bandwidth.
        self.links: dict[str, Link] = {}
        self._bandwidths: dict[tuple[Mesh, str], float | None] = {}
        self._display = TrainingDisplay()
        # The fit ladder's rung beyond the objective's own head and remat: a
        # step that fit only under XLA's default options keeps them for later
        # compiles. A resumed run starts at its checkpoint's rung, and says so
        # if its first compile has to climb past it (`_climb_to`).
        self._xla_defaults = False
        # Why this run's checkpoints describe no model to load, said once.
        self._unrecorded: str | None = None
        # The axes the objective's modules box on their own parameters, as the
        # last trace of `initial_state` saw them (`_boxed_axes`).
        self._boxed: dict[tuple[str, ...], LogicalAxes] | None = None
        self._resumed_rung: JSON = None

    @property
    def optimizer(self) -> optax.GradientTransformation:
        if self._optimizer is None:
            raise ValueError("an OptimConfig is built over the run's length, which only fit(steps=...) gives")
        return self._optimizer

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def initial_state(self, initializer: Initializer | None = None,
                      key: int | jax.Array | None = None) -> TrainState:
        """Build the state a fresh run starts from.

        It is pure, so `fit` traces it once for its shapes and once, sharded, for its
        values.

        Both inputs default to the run's own, and `place` passes them explicitly so
        that what it compiles takes them as arguments. A held checkpoint then reaches
        the device as an argument instead of a constant embedded in the executable.
        Passing None means "use the configured input", which is what a call without
        arguments does. This is the only implementation of the state, so a subclass
        overrides it here and every path sees the override.
        """
        initializer = self.objective.initializer if initializer is None else initializer
        from dew.nn.inputs import request_key
        key = self.key if key is None else request_key(key)
        init_key, run_key = jax.random.split(key)
        initialized = initializer(init_key)
        # Static metadata, the same on every trace of the state: the
        # placement reads it from here rather than tracing init again.
        self._boxed = boxed_axes(initialized)
        params = nn.unbox(initialized)
        if "params" not in params:
            raise ValueError(
                f"the objective's tree has no params collection, only {sorted(params)}; "
                "the optimizer moves params and treats every other collection as state")
        ema = self.objective.ema
        return TrainState(
            step=jnp.zeros((), jnp.int32),
            microstep=jnp.zeros((), jnp.int32),
            updates=jnp.zeros((), jnp.int32),
            scale=(jax.tree.map(jnp.asarray, dynamic_scale_lib.DynamicScale())
                   if self.dynamic_scale else None),
            window_size=jnp.asarray(self.accumulation, jnp.int32),
            variables=params,
            opt_state=self._boxed_init(initialized, params["params"]),
            # The average starts equal to the parameters but as its own
            # buffers: the step donates the state, and a buffer can be
            # donated once.
            ema=None if ema is None else jax.tree.map(jnp.copy, select(params, ema.select)),
            key=run_key,
        )

    @functools.cached_property
    def device_mesh(self) -> Mesh:
        """Build the mesh `MeshSpec` describes over this process pool's devices, on first use."""
        return self.mesh.build()

    @property
    def host_master(self) -> bool:
        return "variables" in self.layout.host

    @functools.cached_property
    def state_mesh(self) -> Mesh:
        if not self.host_master:
            return self.device_mesh
        from dew.training.host import companion_mesh
        return companion_mesh(self.device_mesh)

    def _boxed_init(self, initialized: Variables, params: Variables) -> optax.OptState:
        """The optimizer's state of `params`, Muon's grouping reading the axes
        modules boxed on them in `initialized`."""
        with boxed(boxed_axes(initialized)):
            return self.optimizer.init(params)

    def _boxed_axes(self) -> dict[tuple[str, ...], LogicalAxes]:
        """The axes the objective's modules box on their own parameters
        (`nn.with_logical_partitioning`), which its init returns and the state
        unboxes: what `initial_state` last traced, else one abstract init, for
        a state the trainer did not build."""
        if self._boxed is None:
            self._boxed = boxed_axes(jax.eval_shape(self.objective.initializer, jax.random.key(0)))
        return self._boxed

    def shardings(self, state: TrainState) -> Placement[TrainState]:
        """Return where each field of `state` is placed, on the axes that suit its kind.

        Parameter gradients follow parameters, replay records follow batches, and the
        layout's host-resident fields go in pinned host memory. Under a CPU-owned
        state, the frozen collection is the exception: it stays where the realization
        reads it (`execution.resident`) for the whole run.
        """
        with boxed(self._boxed_axes()):
            return self._shardings(state)

    def _shardings(self, state: TrainState) -> Placement[TrainState]:
        mesh = self.state_mesh
        params = dict(state.variables)
        frozen = params.pop(FROZEN, None) if self.host_master else None
        placed = self.layout.shardings(mesh, dataclasses.replace(state, variables=params, accumulation=None))
        placed = dataclasses.replace(placed, **{
            field: jax.tree.map(lambda s: s.with_memory_kind("pinned_host"), getattr(placed, field))
            for field in (() if self.host_master else self.layout.host)})
        if frozen is not None:
            placed = dataclasses.replace(
                placed, variables={**placed.variables, FROZEN: self._frozen_shardings(state, frozen)}
            )
        accumulation = state.accumulation
        if accumulation is None:
            return placed
        replicated = NamedSharding(mesh, P())
        pending = jax.tree.map(lambda _: replicated, accumulation)
        def buffered_shardings(tree, batches):
            if tree is None:
                return None
            sample = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), tree)
            layout = batch_shardings(mesh, sample) if batches else self.layout.shardings(mesh, sample)
            return jax.tree.map(lambda s: NamedSharding(mesh, P(None, *s.spec)), layout)
        pending = dataclasses.replace(pending,
            gradient=None if accumulation.gradient is None else placed.variables["params"],
            batches=buffered_shardings(accumulation.batches, batches=True),
            variables=buffered_shardings(accumulation.variables, batches=False))
        return dataclasses.replace(placed, accumulation=pending)

    def _fetched(self, state: TrainState, shardings: Placement[TrainState]) -> TrainState:
        """Bring the layout's host-resident fields to the device, where a step
        or an evaluation reads them."""
        if self.host_master:
            return state
        return dataclasses.replace(state, **{
            field: jax.device_put(getattr(state, field), jax.tree.map(
                lambda s: s.with_memory_kind("device"), getattr(shardings, field)))
            for field in self.layout.host})

    def place(self) -> tuple[TrainState, Placement[TrainState], bytes | None]:
        """Put the state on the mesh, fresh or restored.

        It returns the state with its shardings and the data position a resume
        continues from.
        """
        # Resolved once, so that the shapes and the values are the same
        # inputs through the same overridable method, and the objective is
        # asked for what it holds exactly once.
        initializer, key = self.objective.initializer, self.key
        if self.host_master:
            from dew.training.host import transfer
            key = transfer(key, NamedSharding(self.state_mesh, P()))
        abstract = jax.eval_shape(self.initial_state, initializer, key)
        refuse_wide_floats(abstract, self.device_mesh)
        shardings = self.shardings(abstract)
        self.layout.check(abstract.variables, shardings.variables, self.device_mesh)
        checkpoints = self.checkpoints
        resume = None if checkpoints is None else checkpoints.latest
        if checkpoints is None or resume is None:
            if self.host_master:
                return self._placed_host(initializer, key, shardings), shardings, None
            held = self._placed_held(initializer, abstract, shardings)
            with warnings.catch_warnings():
                # A held array no output can take over is freed as the JIT returns.
                warnings.filterwarnings('ignore', 'Some donated buffers were not usable')
                state = jax.jit(self.initial_state, out_shardings=shardings, donate_argnums=0)(held, key)
            return state, shardings, None
        abstract = dataclasses.replace(abstract, accumulation=checkpoints.accumulation_template(resume))
        shardings = self.shardings(abstract)
        if self.host_master and FROZEN in abstract.variables:
            abstract = dataclasses.replace(abstract, variables={
                **abstract.variables, FROZEN: self._banked_frozen(
                    abstract.variables[FROZEN],
                    lambda rows, path: jax.ShapeDtypeStruct((len(rows), *rows[0].shape), rows[0].dtype))})
        template = jax.tree.map(
            lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
            abstract, shardings)
        state, position = checkpoints.restore(template, resume,
                                              share=DataPartition.of(self.device_mesh))
        self._climb_to(checkpoints.rung(resume))
        from dew.nn.inputs import request_key
        state = dataclasses.replace(state, key=jax.device_put(request_key(state.key), shardings.key))
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        self._display.note(f"Resumed from step {resume} in {checkpoints.source(resume)}")
        return state, shardings, position

    def _artifact(self) -> JSON:
        """The objective's inference record, which a checkpoint carries for
        loaders. Training and resuming never read it, so a model or component
        no record can describe still checkpoints: the record says why, a
        loader raises that, and the first checkpoint warns it, as it does a
        model defined where no import path reaches it (`__main__`, a function)."""
        artifact: JSON
        reason = None
        try:
            artifact = self.objective.inference_record()
        except (KeyError, TypeError, ValueError) as error:
            reason = f"{type(self.objective).__name__}: {error}"
            artifact = {'unrecorded': reason}
        if reason is not None and reason != self._unrecorded:
            self._unrecorded = reason
            _log.warning("this run's checkpoints resume, but no loader can rebuild their model "
                         "(TextGeneration.from_run, dew.pipeline) until it is fixed: %s", reason)
        return artifact

    def _rung(self) -> JSON:
        """The fit ladder's rung this trainer's step compiles at: the
        objective's head tile, its trained modules' remat (`recompute_record`) and whether
        the step keeps XLA's default options (`fitting_default`). A
        checkpoint records it, since each decides the program a step runs."""
        tile = head_tile_of(self.objective)
        return {'head_tile': None if tile is None else list(tile),
                'remat': recompute_record(self.objective),
                'xla_defaults': self._xla_defaults}

    def _climb_to(self, rung: JSON) -> None:
        """Move the objective up to `rung`, a checkpoint's (`_rung`), where
        it is above where the objective stands, never below it.

        The fit check reads the free memory its own process finds, and one
        that restores a state can find more than the one that built it and
        pick a lighter rung, another program. So the resumed run compiles the
        rung its checkpoint trained on, and climbs further only where that
        does not fit."""
        if rung is None:
            return
        fields = record(rung, 'rung')
        climb_to(self.objective, fields)
        self._xla_defaults = self._xla_defaults or boolean(
            fields.get("xla_defaults", False), "rung xla_defaults"
        )
        self._resumed_rung = self._rung()

    def _placed_held(self, initializer: Initializer, abstract: TrainState,
                     shardings: Placement[TrainState]) -> Initializer:
        """`initializer` with a copy of each array it holds placed where the
        state keeps the variable of its path and shape
        (`Objective.held_variables`), for the state's JIT to take over, the
        rest replicated.

        Handed to the JIT as they are, the arrays land below the state it
        builds and leave a hole the checkpoint's size once freed, which splits
        the free memory a step's temporaries need in one block
        (`step_headroom`). Placed where the state keeps them and donated, they
        become the state's buffers. The copy leaves the objective's own
        arrays alone."""
        variables = {path: (sharding, leaf.shape) for (path, leaf), sharding in zip(
            jax.tree_util.tree_leaves_with_path(abstract.variables), jax.tree.leaves(shardings.variables),
            strict=True)}
        replicated = NamedSharding(self.device_mesh, P())

        def placement(path, leaf):
            # An initializer binds the checkpoint as `variables`
            # (`Objective.initializer`), so a leaf's path below that keyword
            # is its variable's.
            keys = [key.key if isinstance(key, jax.tree_util.DictKey) else None for key in path]
            sharding, shape = variables.get(path[keys.index('variables') + 1:] if 'variables' in keys else (),
                                            (replicated, None))
            return sharding if shape == np.shape(leaf) else replicated

        # A copy of an array already on the devices: device_put can reuse a
        # buffer the target shares with the source, which donation would free.
        owned = jax.tree.map(lambda leaf: leaf.copy() if isinstance(leaf, jax.Array) else leaf, initializer)
        return jax.device_put(owned, jax.tree_util.tree_map_with_path(placement, initializer))

    def _frozen_shardings(self, state: TrainState, frozen):
        """Return where the frozen collection sits for a CPU-owned run.

        Before placement the layout names each row's shards, and `resident`
        moves the stack's rows to bank memory, a scanned run's shared rows as
        one bank. Once placed, the collection holds those banks, whose
        leading layer axis the layout's rules do not name, and its arrays say
        where they sit.
        """
        leaves = jax.tree.leaves(frozen)
        if leaves and all(isinstance(leaf, jax.Array) for leaf in leaves):
            return jax.tree.map(lambda leaf: leaf.sharding, frozen)
        from dew.training.execution import resident
        rows = self.layout.shardings(
            self.state_mesh, dataclasses.replace(state, variables={FROZEN: frozen}, accumulation=None))
        return resident(rows.variables[FROZEN], self.bank_sites, self.device_mesh)

    @functools.cached_property
    def bank_sites(self):
        """List the objective's declared layer stacks, which a host layout streams as banks."""
        from dew.inference.banks import bank_sites
        return bank_sites(self.objective) if self.objective.bank_sites else ()

    def _banked_frozen(self, tree, stack, release=None):
        """Stack each scanned run's shared leaves of `tree` into one bank
        (`execution.banked`).

        That is the shape a placement and a checkpoint template take, and the
        arrays the state holds."""
        from dew.training.execution import banked
        return banked(tree, self.bank_sites, stack, release)

    def _placed_host(self, initializer, key, shardings: Placement[TrainState]) -> TrainState:
        """A fresh CPU-owned state, its leaves streamed into place one at a time.

        The tree is built eagerly on the CPU, where the held leaves the
        objective hands over are read by reference and nothing of size is
        computed. Each leaf is then moved to its placement and the source let
        go as it lands (`host.stream`): a moving leaf to the companion, a
        frozen one to where it stays resident (`execution.resident`). The
        transient is one leaf, not the tree, and one JIT could not have
        returned to the two device sets anyway.
        """
        from dew.training.execution import bank_bytes, check_bank_pool
        from dew.training.host import evict, place_leaf, stream, transfer
        held = self.objective.held_variables()
        with jax.default_device(self.state_mesh.local_devices[0]):
            state = self.initial_state(initializer, key)
            if FROZEN in state.variables:
                check_bank_pool(bank_bytes(state.variables[FROZEN], self.bank_sites), self.device_mesh)
                # A scanned run's frozen rows become its bank here and the
                # bank lands where it stays resident before the next is
                # stacked, so the host holds one bank in transit, not the
                # stack; the rows' pages are let go as it lands and the held
                # tree holds the placed bank where each row was. An
                # objective placed this way holds banks at its rows afterwards
                # and does not seed a second trainer.
                targets = shardings.variables[FROZEN]

                def stack(rows, path):
                    target = targets
                    for component in path:
                        target = target[component]
                    bank = place_leaf(jnp.stack(rows), target)
                    for row in rows:
                        evict(np.asarray(row))
                    return bank

                def release(namespace, index, keys, bank):
                    # The rows sit under the held tree's own frozen split
                    # when it came split, or under its params when init split it.
                    node = None if held is None else held.get(FROZEN, held.get("params"))
                    for component in (*namespace, f"layers_{index}", *keys[:-1]):
                        node = node.get(component) if isinstance(node, dict) else None
                    if isinstance(node, dict) and keys[-1] in node:
                        node[keys[-1]] = bank
                frozen = self._banked_frozen(state.variables[FROZEN], stack, release)
                state = dataclasses.replace(state, variables={**state.variables, FROZEN: frozen})
        params = stream(state.variables, shardings.variables, held)
        ema = None if state.ema is None else stream(state.ema, shardings.ema)
        rest = dataclasses.replace(state, variables=None, ema=None)
        placed = transfer(rest, dataclasses.replace(shardings, variables=None, ema=None))
        return dataclasses.replace(placed, variables=params, ema=ema)

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def _loss_shape(self, state: TrainState, batch: Batch):
        # Shapes and dtypes only: a resident frozen leaf sits in another
        # memory space than the moving ones, and the loss the realization
        # runs reads a snapshot in one space. The step's key is drawn inside
        # the trace, so an abstract state compiles a step as a placed one does.
        params = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

        def loss(params, batch, microstep, key, step, ema):
            return self.objective._loss(
                params, batch, Step(microstep, jax.random.fold_in(key, step), with_ema(params, ema)))

        return jax.eval_shape(loss, params, batch, state.microstep, state.key, state.step, state.ema)

    def _initialize_accumulation(self, state: TrainState, batch: Batch, shapes, *, shape_only=False):
        if self.accumulation == 1 or self.step is not None or state.accumulation is not None:
            return state
        stats, aux = shapes
        shared = isinstance(stats, (Ratio, jax.ShapeDtypeStruct))
        mean_dtype = jnp.result_type(jnp.float32, *(x.dtype for x in jax.tree.leaves(stats)))
        slots = self.accumulation - 1
        def shape(leaf: jax.Array) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)

        trainable = jax.tree.map(shape, state.variables["params"])
        records = jax.tree.map(shape, batch)
        mutable = (None if shared or aux.variables is None else
                   jax.tree.map(shape, {name: state.variables[name] for name in aux.variables}))

        def zeros(leaf):
            dtype = (
                jnp.promote_types(leaf.dtype, jnp.float32)
                if jnp.issubdtype(leaf.dtype, jnp.inexact)
                else leaf.dtype
            )
            return jnp.zeros(leaf.shape, dtype)

        def buffer(tree):
            return jax.tree.map(lambda x: jnp.zeros((slots, *x.shape), x.dtype), tree)

        def allocate():
            return Accumulation(
                gradient=jax.tree.map(zeros, trainable) if shared else None,
                mass=jnp.zeros((), mean_dtype) if shared else None,
                statistics=((jnp.zeros((), mean_dtype),) if shared else
                            tuple(zeros(x) for x in jax.tree.leaves(stats))),
                effects=tuple(zeros(x) for x in jax.tree.leaves(aux.effects)),
                qk_stats=jax.tree.map(
                    lambda x: jnp.full(x.shape, -jnp.inf, jnp.promote_types(x.dtype, jnp.float32))
                    if jnp.issubdtype(x.dtype, jnp.inexact) else zeros(x),
                    compact_qk(jax.tree.map(zeros, aux.qk_stats))),
                batches=None if shared else buffer(records),
                variables=None if mutable is None else buffer(mutable),
                attempts=None if shared else jnp.zeros((slots,), jnp.int32))

        pending = jax.eval_shape(allocate)
        if not shape_only:
            placement = self.shardings(dataclasses.replace(state, accumulation=pending)).accumulation
            # Allocate on the final shards, without a replicated window-sized temporary.
            pending = jax.jit(allocate, out_shardings=placement)()
        return dataclasses.replace(state, accumulation=pending)

    def _links(self, mesh: Mesh) -> dict[str, Link]:
        """The links of `mesh`'s tensor and sequence axes, those it splits,
        for one trace. Every process lists the mesh's devices alike and
        `link_bandwidth` agrees one figure across the pool, so every process
        decides alike and compiles the same program. The peak is the fastest
        device's, on which the FLOPs a spread saves take the least time. Each
        axis's bandwidth is measured once per mesh, and not at all on a CPU
        mesh or for a device the peak table does not name, where it decides
        nothing."""
        devices = list(mesh.devices.flat)
        peaks = [peak for device in devices if (peak := peak_flops(device.device_kind)) is not None]
        peak = max(peaks) if len(peaks) == len(devices) else None
        platform = devices[0].platform
        links = {}
        for axis in (TENSOR_AXIS, SEQUENCE_AXIS):
            if mesh.shape[axis] == 1:
                continue
            if (mesh, axis) not in self._bandwidths:
                self._bandwidths[mesh, axis] = (
                    None if platform == 'cpu' or peak is None else link_bandwidth(mesh, axis))
            links[axis] = Link(self._bandwidths[mesh, axis], peak, platform)
        return links

    @contextlib.contextmanager
    def _traced_on(self, mesh: Mesh, links: Mapping[str, Link] | None = None) -> Iterator[Schedule]:
        """What a traced step reads from context: the mesh, the pipeline's
        microbatch count, the layout's rules, which place the activations
        the model constrains (`dew.nn.sharding.constrain`) as they place the
        parameters, and the links of the axes the step measured, none by
        default."""
        with (jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches) as schedule,
              nn.logical_axis_rules(self.layout.axis_rules), measured_links({} if links is None else links)):
            yield schedule

    def compile(self, state: TrainState, batch: Batch) -> CompiledStep:
        """Compile a transaction over the state and one already-produced global batch.

        The step consumes the state it is given. The returned state takes over its
        buffers, so the update runs in place and peak memory holds one copy of the
        parameters and optimizer state, not two. Keep no reference to a state after
        stepping it; `new = step(old, batch)` is the whole contract.

        A checkpoint saved before the step is safe: Orbax copies every array to the
        host before `save` returns, as long as `Checkpoints` names no prioritized keys
        and no concurrent transfer limit. The batch is not donated, because the loader
        owns it.
        """
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        if (state.scale is not None) != self.dynamic_scale:
            raise ValueError("checkpoint dynamic-scaler configuration differs from this trainer")
        if self.host_master:
            return self._compile_host(state, batch)
        mesh = self.device_mesh
        links = self._links(mesh)
        resumed, self._resumed_rung = self._resumed_rung, None
        with self._traced_on(mesh, links) as schedule:
            shapes = None if self.step is not None else self._loss_shape(state, batch)
            if shapes is not None and mesh.shape[STAGE_AXIS] > 1 and not schedule.pipelined:
                raise LayoutRefused(
                    f"the stage axis of {mesh.shape[STAGE_AXIS]} holds a pipeline's stages of "
                    f"a decoder's layer stack, and {type(self.objective).__name__}'s model runs "
                    f"no pipeline, so every stage would compute the whole step; give those "
                    f"devices to the data or fsdp axis")
            prepared = self._initialize_accumulation(state, batch, shapes, shape_only=True)
            copies = self._narrow_copies(state, batch)
            if copies:
                prepared = dataclasses.replace(prepared, compute=jax.eval_shape(
                    lambda variables: narrowed(variables, copies), prepared.variables))
            shardings = self.shardings(prepared)
            replicated = NamedSharding(mesh, P())
            placement = batch_shardings(mesh, batch)
            held = prefetched_bytes(batch, placement)
            prepared = jax.tree.map(
                lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), prepared, shardings)
            refusals: list[RuntimeError] = []
            consulted, record_path, key = False, None, {}
            while True:
                body = (self.step(self.objective, self.optimizer) if self.step is not None else
                        Transaction(self.objective, self.optimizer, self.accumulation, shapes,
                                    copies).step())

                def step(current, batch, body=body):
                    # The body sees every field on the device; the out shardings
                    # return the host-resident ones to pinned host memory.
                    advanced, loss, aux = body(self._fetched(current, shardings), batch)
                    return (dataclasses.replace(advanced, step=current.step + 1), loss,
                            aux.metrics, jnp.isfinite(loss), advanced.microstep > current.microstep)

                jitted = jax.jit(step, in_shardings=(shardings, placement),
                                 out_shardings=(shardings, replicated, replicated, replicated,
                                                replicated),
                                 donate_argnums=0)
                self.program = jitted.lower(prepared, batch)
                if not consulted:
                    consulted, before = True, self._rung()
                    record_path, key, recorded = recorded_rung(self.program, mesh)
                    self._climb_to(recorded)
                    self._resumed_rung = None
                    if self._rung() != before:
                        _log.warning("taking an earlier run's rung; delete %s to decide again", record_path)
                        continue
                shards = math.prod(mesh.shape[axis] for axis in BATCH_AXES)
                options = None if self._xla_defaults else step_compiler_options(
                    self.objective, device_tokens(self.objective, batch, shards),
                    FROZEN in prepared.variables)
                self.executable = compiled_if_it_fits(self.program, options, refusals)
                fits = step_fits(self.executable, mesh, held)
                if not fits and options is not None:
                    self.executable, fits = fitting_default(self.program, self.executable, mesh, held,
                                                            refusals)
                    self._xla_defaults = fits
                if not fits and resumed is not None:
                    _log.warning(
                        "the step does not fit the devices at the rung its checkpoint trained on "
                        "(%s); from here the resumed run computes otherwise than the run it continues",
                        resumed,
                    )
                    resumed = None
                if fits or not recompute_more(self.objective):
                    break
            if self.executable is None:
                # XLA refused the last rung too, and its refusal is the run's error.
                raise refusals[-1]
            if record_path is not None:
                keep_rung(record_path, key, self._rung())
            self.flops_per_step = compiled_flops(self.executable)
            self.links = links
        # The program compiled above, not the jit: a call through the jit
        # traces and compiles its own, without the step's compiler options,
        # which was 25 s of an A100's cold start.
        executable = self.executable

        def run(current, batch):
            with self._traced_on(mesh):
                if current.accumulation is None and self.accumulation > 1 and self.step is None:
                    current = self._initialize_accumulation(current, batch, shapes)
                    current = jax.device_put(current, shardings)
                if copies and current.compute is None:
                    current = dataclasses.replace(current, compute=jax.jit(
                        lambda variables: narrowed(variables, copies), out_shardings=shardings.compute)(
                            current.variables))
                return executable(current, batch)
        return run

    def _narrow_copies(self, state: TrainState, batch: Batch) -> dict:
        """The parameters the step reads through narrow copies
        (`dew.training.narrow`), by path under `params`: on the CUDA
        generations measured to gain, for a step this trainer composes that
        commits every microbatch without a loss scale."""
        if (measured_kernel('narrow_copies', 'off') == 'off' or self.step is not None
                or self.accumulation != 1 or self.dynamic_scale or FROZEN in state.variables):
            return {}
        variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

        def loss(params, rest, batch, microstep, key, step, ema):
            # As `_loss_shape` traces it, the averaged weights the step hands
            # the objective included: a parameter read there is used twice.
            variables = {**rest, "params": params}
            return self.objective._loss(
                variables, batch, Step(microstep, jax.random.fold_in(key, step), with_ema(variables, ema)))

        rest = {name: tree for name, tree in variables.items() if name != "params"}
        return narrowed_paths(loss, variables["params"], rest, batch, state.microstep, state.key, state.step,
                              state.ema)

    def _compile_host(self, state: TrainState, batch: Batch) -> CompiledStep:
        from dew.training.execution import HostExecution
        from dew.training.host import transfer
        cpu = self.state_mesh
        execution = HostExecution(self.objective, self.layout, self.device_mesh, cpu)
        cpu_batch = transfer(batch, batch_shardings(cpu, batch))
        with self._traced_on(cpu):
            shapes = self._loss_shape(state, cpu_batch)
            prepared = self._initialize_accumulation(state, cpu_batch, shapes, shape_only=True)
            placement = self.shardings(prepared)
            transaction = Transaction(self.objective, self.optimizer, self.accumulation, shapes)
            body = transaction.step(realize=execution.realize, host=True)

        def run(current, batch):
            batch = transfer(batch, batch_shardings(cpu, batch))
            with self._traced_on(cpu):
                if current.accumulation is None and self.accumulation > 1:
                    current = self._initialize_accumulation(current, batch, shapes)
                advanced, loss, aux = body(current, batch)
                advanced = dataclasses.replace(advanced, step=current.step + 1)
                # A leaf already where its placement says stays the array it
                # is: the resident frozen collection crosses no boundary.
                advanced = transfer(advanced, placement)
                return (advanced, loss, aux.metrics, jnp.isfinite(loss),
                        advanced.microstep > current.microstep)
        return run

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: Dataset,
        *,
        steps: int,
        log_every: int = 100,
        eval_every: int | None = None,
        checkpoint_every: int | datetime.timedelta | None = None,
        metrics: Sequence[Metric] = (),
        preview: bool = False,
        best: str | Metric | TrainingScalar | types.MethodType | Best | Sequence[Best] | None = None,
        stop: Plateau | None = None,
        validation: Mapping[str, Reader | EvalSuite] | None = None,
        restore_best: bool = False,
        state: TrainState | None = None,
    ) -> TrainState:
        """Train to `steps` total steps, resuming from the latest checkpoint if the directory has one.

        An explicit `state` takes precedence over initialization and checkpoint
        restoration; its input reader starts at the position `dataset` supplies, not
        at a checkpoint's data position.

        Every `log_every` steps, the tracker receives the loss, the objective's
        metrics and the throughput.

        Every `eval_every` steps, and at the end, the validation split is scored. The
        objective's artifacts go to the tracker and to `metrics`, whose reductions are
        logged as `val/<name>`. `validation` names several splits instead, each a
        reader scored as `dataset.val` would be or an `EvalSuite` with its own
        metrics and cadence, logged as `<name>/<metric>`.

        Every `checkpoint_every` steps, and at the end, the state and the data
        position are written; a duration is checked at each `log_every` step, so
        they are written at the first log step after that much time. Every
        `checkpoints.local_every` steps they are also written to the local
        directory.

        A preemption notice (a scheduler's SIGTERM; `PreemptionNotice`) stops the run
        at the next step every process agrees on. That step's state and data position
        are written, the final validation is skipped, and fit raises `Preempted`,
        which ends the program with status 143 unless caught. Run it again and fit
        resumes from there.

        Previews are generated only when `preview=True` and a tracker receives them;
        scalar reporting never triggers preview work.
        """
        if self._config is not None:
            built = self._config.build(steps // self.accumulation)
            self._optimizer = self.objective.optimizer(built, accumulation=self.accumulation)
        suites = _suites(dataset, validation, metrics, eval_every)
        selection, stop = self._fit_policies(best, stop, metrics, suites, checkpoint_every, restore_best)
        if metrics and eval_every is None and validation is None:
            raise ValueError("metrics need eval_every to schedule their validation pass")
        self._preflight(dataset, stop, suites, preview=preview)
        preview = preview and self.tracker is not None
        run = _FitRun(time.perf_counter())
        # A display of this fit's own: one before it may have run other steps.
        self._display = TrainingDisplay()
        profile, checkpoints = self.profile, self.checkpoints
        profiler = self._own_profile_window()
        # One boundary lookup at setup: the prefetch worker and the step
        # scopes share whichever profiler owns the capture.
        tracer = profiler if profiler is not None else telemetry_profile.active_profile()
        try:
            plan = _FitPlan(
                dataset,
                steps,
                log_every,
                eval_every,
                checkpoint_every,
                None if checkpoints is None else checkpoints.local_every,
                metrics,
                preview,
                best=selection,
                stop=stop,
                suites=suites,
            )
            trained, complete = self._training_loop(plan, run, profiler, tracer, profile, checkpoints, state)
            # The state a checkpoint would hold: the step rebuilds its narrow copies.
            trained = dataclasses.replace(trained, compute=None)
        finally:
            primary = sys.exception()
            error = self._closed(run, primary, profiler)
            if primary is None and error is not None:
                raise error
        if complete:
            return trained
        if run.preempted is not None:
            raise Preempted(run.preempted)
        if restore_best and checkpoints is not None:
            trained, _ = checkpoints.restore(
                jax.tree.map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), trained
                ),
                step="best",
            )
        return trained

    def _training_loop(self, plan: _FitPlan, run: _FitRun, profiler: Profiler | None,
                       tracer: Profiler | None, profile: ProfileWindow | None,
                       checkpoints: Checkpoints | None,
                       initial: TrainState | None = None) -> tuple[TrainState, bool]:
        """Place the state, dispatch the numerical steps and finish their loop
        before resource cleanup. Returns the final state and whether the run
        was already complete, its last checkpoint written, so nothing ran.

        The loop is the state's one holder: each step donates the state it is
        handed, and a reference kept past that, as fit's own once was, holds
        an array whose buffers the step consumed."""
        if initial is None:
            state, shardings, position = self.place()
        else:
            shardings = self.shardings(initial)
            self.layout.check(initial.variables, shardings.variables, self.device_mesh)
            state = jax.device_put(initial, shardings)
            position = None
        run.last_checkpoint = time.perf_counter()
        if checkpoints is not None and checkpoints.latest is not None:
            run.stop_control = checkpoints.control(checkpoints.latest)
        if self._opened(plan, run, state, position):
            return state, True
        compiled: dict[Shapes, tuple[CompiledStep, float | None]] = {}
        interval = _Interval(time.time(), last_saved=(
            run.current if checkpoints is not None and checkpoints.latest is not None else None))
        seen = 0
        run.notice = PreemptionNotice()
        while run.current < plan.steps:
            # Read through `run` each time: a local alias would keep the
            # closed iterator reachable from a failed run's traceback.
            assert run.train is not None
            # The window's capture opens before this iteration's first
            # read, so the step row records the read it waits on rather
            # than a compile that ran before capture began.
            if (profiler is not None and profile is not None
                    and not run.tracing and run.traced == 0
                    and seen >= profile.warmup):
                self._start_window(profiler)
                run.tracing = True
            capturing = tracer is not None and tracer.running
            step_scope = (jax.profiler.StepTraceAnnotation("train", step_num=run.current)
                          if capturing else contextlib.nullcontext())
            with step_scope:
                with region("input.wait"):
                    batch = next(run.train)
                if self.rollout is not None:
                    batch, sampled = self._rolled_out(state, batch)
                    interval.rollout_seconds += sampled
                inputs = self.objective.inputs
                if not compiled and self.rollout is None and inputs is not None:
                    # A rollout writes the batch its loss reads, from prompts it reads itself.
                    agreed("training input declaration", functools.partial(inputs.check, batch))
                first_compile = not compiled
                train_step, measured_flops = self._compiled_for(compiled, state, batch)
                if first_compile:
                    # Rebound once the step is compiled, so the first tick
                    # measures steps, not the compile.
                    interval.last_log_time = time.time()
                with region("train.step"):
                    state, loss, aux, finite, accepted = train_step(state, batch)
                run.loss = loss
                position = run.train.source_state
                run.current += 1
                self._display.step(run.current)
                seen += 1
                interval.count(batch, measured_flops, loss, finite)
                if run.first_step is None:
                    loss.block_until_ready()
                    run.first_step = time.perf_counter() - run.started
                if self._between_steps(plan, run, interval, state, shardings, position,
                                       loss, aux, accepted):
                    break
            # The step row is complete once its scope exits; closing the
            # window here keeps the last iteration inside the capture.
            if run.tracing and profile is not None:
                run.traced += 1
                if run.traced == profile.steps:
                    run.tracing = False
                    assert profiler is not None
                    self._stop_trace(run, profile, profiler)
        self._wind_down(plan, run, interval, state, shardings, position, profiler)
        return state, False

    def _closed(self, run: _FitRun, primary: BaseException | None,
                profiler: Profiler | None) -> BaseException | None:
        """`fit`'s cleanup however the run ended: the preemption notice, the
        stream, a trace window and the display closed, the checkpoints
        waited on, and the outcome reported. Returns the error to raise when
        the run itself raised none.

        Each teardown step runs even when an earlier one failed, and a later
        failure becomes a note on the first, so one broken sink cannot hide
        the error that ended the run."""
        paused = time.perf_counter()
        profile, checkpoints = self.profile, self.checkpoints
        if run.notice is not None:
            run.notice.close()
        close = (run.train.close if run.train is not None else
                 run.source.close if isinstance(run.source, Closeable) else None)
        stop_trace = None
        if run.tracing and profile is not None:
            run.tracing = False
            assert profiler is not None
            stop_trace = functools.partial(self._stop_trace, run, profile, profiler)
        error = primary
        for label, cleanup in (
            ("Training iterator", close),
            ("Profiler", stop_trace),
            ("Checkpoint wait", None if checkpoints is None else checkpoints.wait),
            ("Display", self._display.close),
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f"{label} cleanup failed: {failure!r}")
        # The traceback of a failed run holds fit's frame and this one, and
        # with them whatever these names still point at.
        run.source = run.train = None
        close = stop_trace = cleanup = None
        run.other += time.perf_counter() - paused
        return self._reported_outcome(error, run)

    def _opened(self, plan: _FitPlan, run: _FitRun, state: TrainState, position) -> bool:
        """Start `fit` from the placed state: report it, refuse a run already
        past its steps or a checkpoint cadence with no checkpointer, and open
        the training stream on its prefetch worker. Returns whether the run
        is already complete, a checkpoint at its last step having been
        written, so there is nothing to do."""
        mesh, checkpoints, steps = self.device_mesh, self.checkpoints, plan.steps
        run.current = current = int(state.step)
        started = FitStarted(current, steps,
            checkpoints.source(current) if checkpoints is not None and position is not None else None,
            sum(leaf.size for leaf in jax.tree.leaves(state.variables["params"])), mesh.devices.size,
            jax.devices()[0].device_kind, jax.process_count(), dict(mesh.shape),
            split_share(state.variables),
            seed=self.seed)
        self._report(started, current)
        # Read through the type that declares it: fit takes any object with a
        # Dataset's readers, and a held-out count is not one of them.
        if isinstance(plan.dataset, Dataset) and plan.dataset.held_out:
            self._display.note(f"validation: {plan.dataset.held_out} records held out of train")
        if current > steps:
            raise ValueError(f"the run is at step {current}, past the {steps} asked for")

        if plan.checkpoint_every and checkpoints is None:
            raise ValueError(
                "checkpoint_every asks for checkpoints and this trainer has no "
                "checkpointer; pass Checkpoints(directory) to write any")
        if current == steps and checkpoints is not None and checkpoints.latest is not None:
            return True
        if current < steps:
            run.source = plan.dataset.train(DataPartition.of(mesh))
            self._check_stream(run.source, mesh, plan.dataset.batch,
                               checkpointing=bool(plan.checkpoint_every or plan.local_every or (
                                   checkpoints is not None and plan.cadenced and plan.best and
                                   any(not choice.weights_only for choice in plan.best))))
            run.train = DevicePrefetchIterator(run.source, mesh, source_state=position)
            run.source = None  # Lifetime transferred to the prefetch worker.

        trained = [entry.module for entry in self.objective.program_key() if entry.trained]
        stored = sorted({str(leaf.dtype) for leaf in jax.tree.leaves(state.variables["params"])})
        compute = getattr(trained[0], "dtype", None) if trained else None
        precision = "/".join(stored)
        if compute is not None and [str(jnp.dtype(compute))] != stored:
            precision += f" parameters, {jnp.dtype(compute)} compute"
        shown = {**TRAINER_SHOWN, **self.objective.shown, **_reported(self.rollout, plan.scored)}
        agreed("training announcement", functools.partial(
            self._display.start, started, model=type(trained[0] if trained else self.objective).__name__,
            batch=plan.dataset.batch, precision=precision, shown=shown,
            averaged=self.objective.ema is not None and not self.objective._ema_is_reference))
        return False

    def _between_steps(self, plan: _FitPlan, run: _FitRun, interval: _Interval, state: TrainState,
                       shardings: Placement[TrainState], position, loss: jax.Array,
                       aux: dict[str, jax.Array], accepted: jax.Array) -> bool:
        """The cadenced work after one step of `fit`: the log, the validation,
        the checkpoints, and the preemption notice. Returns whether the
        notice stops the run at this step, whose checkpoint is then written."""
        current, steps, checkpoints = run.current, plan.steps, self.checkpoints
        run.training = {'train/loss': loss, **{f'train/{name}': value for name, value in aux.items()}}
        if current % plan.log_every == 0:
            interval.check_finite(current, self._display)
            self._log_interval(current, loss, aux, accepted, state, interval)

        if isinstance(plan.checkpoint_every, datetime.timedelta):
            # The clock is read at the log interval, where the step is already
            # waited on, so a pool's processes agree on it there rather than
            # gathering behind every step.
            checkpoint_due = False
            if current % plan.log_every == 0:
                elapsed = time.perf_counter() - run.last_checkpoint
                due = np.asarray(elapsed >= plan.checkpoint_every.total_seconds())
                checkpoint_due = bool(due) if jax.process_count() == 1 else bool(
                    np.asarray(multihost_utils.process_allgather(due)).any())
        else:
            checkpoint_due = bool(plan.checkpoint_every and current % plan.checkpoint_every == 0)
        due = plan.due(current)
        evaluation_due = bool(due) or bool(plan.eval_every and current % plan.eval_every == 0)
        ranked = bool((plan.validation or plan.best or plan.stop) and checkpoint_due)
        if current < steps and (evaluation_due or ranked):
            with region("evaluate"):
                # A checkpoint that ranks the steps reads every suite's score.
                names = None if ranked else due
                run.evaluation = self._evaluation(plan, state, shardings, run.training, names)
                run.other += run.evaluation.elapsed_seconds
        scores = (
            {} if run.evaluation is None or run.evaluation.step != current else dict(run.evaluation.scores)
        )
        ranking = self._ranking(plan, scores)
        winners = checkpoints._candidates(ranking) if (
            evaluation_due and checkpoints is not None and (plan.checkpoint_every or plan.best)) else ()
        candidate = bool(winners)
        if scores and self._plateau(plan, run, scores):
            run.stopped = True
            run.stop_control['stop_reason'] = 'validation plateau'
            self._display.note("Stopped: validation plateau")
            candidate = checkpoints is not None

        # On its own clock, not the logging one, so that a cadence
        # which does not divide log_every still fires.
        if (plan.checkpoint_every and checkpoints is not None
                and checkpoint_due and current < steps) or (candidate and current < steps):
            assert checkpoints is not None
            run.other += self._saved_checkpoint(
                checkpoints,
                current,
                state,
                position,
                interval,
                scores=scores,
                ranking=ranking,
                training_best=not plan.validation and not plan.best,
                control=run.stop_control,
                weights_only=not checkpoint_due
                and not run.stopped
                and bool(winners)
                and all(rank.weights_only for rank in winners),
            )
            run.last_checkpoint = time.perf_counter()
        if (plan.local_every and checkpoints is not None
                and current % plan.local_every == 0 and current < steps):
            run.other += self._saved_local_checkpoint(
                checkpoints, current, state, position, control=run.stop_control)
        # Asked at every step on every process, as JAX's agreement
        # needs; the last step ends the run on its own.
        if run.stopped:
            return True
        assert run.notice is not None
        if current < steps and run.notice.reached(current):
            if checkpoints is not None and interval.last_saved != current:
                run.other += self._saved_checkpoint(
                    checkpoints,
                    current,
                    state,
                    position,
                    interval,
                    training_best=not plan.validation and not plan.best,
                    control=run.stop_control,
                )
            run.preempted = current
            return True
        return False

    def _wind_down(self, plan: _FitPlan, run: _FitRun, interval: _Interval, state: TrainState,
                   shardings: Placement[TrainState], position, profiler: Profiler | None) -> None:
        """`fit` after its last step: the stream closed, a preemption
        announced, the trace window stopped, the final validation (skipped on
        a preemption) and the checkpoint of the state the run ends on."""
        profile, checkpoints, current, loss = self.profile, self.checkpoints, run.current, run.loss
        if run.notice is not None:
            run.notice.close()
        paused = time.perf_counter()
        if run.train is not None:
            run.train.close()
            run.train = None
        run.other += time.perf_counter() - paused
        if run.preempted is not None:
            stopped_at = run.preempted

            def announce_preemption() -> None:
                self._display.note(f"Preempted at step {stopped_at}: " + (
                    f"its checkpoint and data position go to {checkpoints.directory}, "
                    "where the next run resumes" if checkpoints is not None
                    else "the trainer has no checkpointer, so nothing is written"))

            agreed("preemption announcement", announce_preemption)
        if run.tracing and profile is not None:
            run.tracing = False
            # The window outlived the run, and a trace left running takes the
            # next one down with it.
            assert profiler is not None
            self._stop_trace(run, profile, profiler)
        interval.check_finite(current, self._display)
        if loss is not None:
            # The last step has to land before the wall time is read.
            loss.block_until_ready()
        if (
            (plan.validation or plan.eval_every or plan.best or plan.stop)
            and run.preempted is None
            and not run.stopped
        ):
            run.evaluation = self._evaluation(plan, state, shardings, run.training)
            run.other += run.evaluation.elapsed_seconds
            if self._plateau(plan, run, run.evaluation.scores):
                run.stopped = True
                run.stop_control['stop_reason'] = 'validation plateau'
                self._display.note("Stopped: validation plateau")
        if checkpoints is not None and interval.last_saved != current:
            # The in-loop saves are conditional, so the state the run ends
            # on may never have been written. It goes out under its real
            # step: a step-0 checkpoint holding the final weights would
            # make a resume restart the schedule from the beginning.
            scores = {} if run.evaluation is None or run.evaluation.step != current else run.evaluation.scores
            run.other += self._saved_checkpoint(
                checkpoints,
                current,
                state,
                position,
                interval,
                scores=scores,
                ranking=self._ranking(plan, scores),
                training_best=not plan.validation and not plan.best,
                control=run.stop_control,
            )
        if checkpoints is not None:
            checkpoints.wait()

    def _fit_policies(self, best, stop, metrics, suites, checkpoint_every, restore_best):
        if isinstance(checkpoint_every, datetime.timedelta) and checkpoint_every.total_seconds() <= 0:
            raise ValueError("checkpoint_every duration must be positive")
        splits = tuple(suites) or ('val',)
        metrics = tuple({metric.name: metric for metric in (*metrics, *(
            metric for suite in suites.values() for metric in suite.metrics))}.values())
        raw = (
            ()
            if best is None
            else tuple(best)
            if isinstance(best, Sequence) and not isinstance(best, str)
            else (best,)
        )

        def owned(choice):
            if isinstance(choice, types.MethodType):
                if choice.__name__ == 'loss' and choice.__self__ is self.objective:
                    return self.objective.scalars.loss
                raise TypeError("a training selector must be this objective's loss or declared scalar")
            if isinstance(choice, TrainingScalar) and choice.owner is not self.objective:
                raise ValueError("training scalar belongs to a different objective")
            if isinstance(choice, Best):
                source = choice._source
                if source is None and (choice.metric.startswith('train/') or choice.split == 'train'):
                    source = self.objective.scalars[choice.metric.removeprefix('train/')]
                return dataclasses.replace(choice, _source=owned(source) if source is not None else None)
            return choice

        selection = tuple(
            self._best_selection(owned(Best(choice) if isinstance(choice, str) else choice), metrics, splits)
            for choice in raw
        )
        labels = [(choice.split, choice.metric) for choice in selection
                  if isinstance(choice._source, (Metric, TrainingScalar))]
        if len(labels) != len(set(labels)):
            raise ValueError("select each metric once; one Best.top already keeps all its winners")
        if stop is not None:
            metric = owned(stop.metric)
            if not isinstance(metric, (Metric, TrainingScalar)):
                raise TypeError(
                    "Plateau selects a declared metric object, not a Best policy or score function"
                )
            stopping = self._best_selection(Best(metric, mode=stop.mode, split=stop.split), metrics, splits)
            stop = dataclasses.replace(stop, metric=metric, mode=stopping.mode, split=stopping.split)
        if restore_best and any(choice.weights_only for choice in selection):
            raise ValueError("restore_best requires full checkpoints, not weights_only snapshots")
        return selection, stop

    @staticmethod
    def _best_selection(
        best: str | Metric | TrainingScalar | Best,
        metrics: Sequence[Metric],
        splits: Sequence[str] = ("val",),
    ) -> Best:
        selection = best if isinstance(best, Best) else Best(best)
        metric = selection._source
        if metric is None:
            name = selection.metric.removeprefix((selection.split or 'val') + '/')
            metric = next((declared for declared in metrics if declared.name == name), None)
            if metric is None:
                raise ValueError(f"best metric {selection.metric!r} is not among fit's metrics")
            selection = dataclasses.replace(selection, _source=metric)
        if isinstance(metric, TrainingScalar):
            if selection.mode is None:
                if metric.shown.better is None:
                    raise ValueError("training scalar has no declared direction; use Best(scalar, mode=...)")
                selection = dataclasses.replace(
                    selection, mode="max" if metric.shown.better == "higher" else "min"
                )
            return dataclasses.replace(selection, split='train')
        if isinstance(metric, (Metric, TrainingScalar)):
            if not any(metric is declared for declared in metrics):
                raise ValueError("best must be a metric object passed in metrics")
            declaration = _reported(None, metrics).get(metric.name)
            better = declaration.better if isinstance(declaration, Shown) else None
            if selection.mode is None:
                if better not in ('higher', 'lower'):
                    raise ValueError(
                        "best metric has no declared direction; use Best(metric, mode='min' or 'max')"
                    )
                selection = dataclasses.replace(selection, mode='max' if better == 'higher' else 'min')
        if selection.split is None:
            if len(splits) != 1 and isinstance(metric, Metric):
                raise ValueError("Best(metric, split=...) is required with several validation splits")
            selection = dataclasses.replace(selection, split=splits[0] if len(splits) == 1 else None)
        elif selection.split not in splits:
            raise ValueError(f"unknown validation split {selection.split!r}")
        return selection

    def _ranking(self, plan: _FitPlan, scores: Mapping[str, float]) -> tuple[Ranking, ...]:
        if not plan.best:
            name = (
                next(iter(plan.suites)) + "/loss"
                if plan.validation
                else "train/loss"
            )
            return (Ranking(name, scores[name]),) if name in scores else ()
        ranks = []
        for index, selection in enumerate(plan.best):
            metric = selection._source
            assert metric is not None
            if isinstance(metric, (Metric, TrainingScalar)):
                name = f'{selection.split}/{metric.name}'
                value = scores.get(name, float('nan'))
            else:
                name = f'aggregate:{index}'
                try:
                    value = float(metric(_MetricValues(plan.scored, scores, self.objective)))
                except KeyError:
                    value = float('nan')
            mode = selection.mode or 'min'
            threshold = selection.threshold
            if threshold is not None and (value >= threshold if mode == 'min' else value <= threshold):
                value = float('nan')
            ranks.append(Ranking(name, value, mode, selection.top, selection.weights_only))
        return tuple(ranks)

    @staticmethod
    def _plateau(plan: _FitPlan, run: _FitRun, scores: Mapping[str, float]) -> bool:
        stop = plan.stop
        if stop is None:
            return False
        assert isinstance(stop.metric, (Metric, TrainingScalar)), (
            "fit binds the stopping selector before stepping"
        )
        name = f'{stop.split}/{stop.metric.name}'
        if name not in scores:
            return False
        value = scores[name] if stop.mode == 'min' else -scores[name]
        if not math.isfinite(value):
            return False
        key = f'plateau:{name}'
        held = run.stop_control.get(key)
        rule = {'mode': stop.mode, 'evals': stop.evals, 'min_delta': stop.min_delta}
        if held is not None and held.get('rule') != rule:
            raise ValueError(
                "Plateau policy differs from the resumed checkpoint; resume with the same stopping rule"
            )
        if held is not None and held.get('step') == run.current:
            return held['bad'] >= stop.evals
        if held is None or value < held['best'] - stop.min_delta:
            run.stop_control[key] = {'best': value, 'bad': 0, 'step': run.current, 'rule': rule}
            return False
        held['bad'] += 1
        held['step'] = run.current
        return held['bad'] >= stop.evals

    def _evaluation(self, plan: _FitPlan, state: TrainState, shardings: Placement[TrainState],
                    training: Mapping[str, jax.Array], names: Sequence[str] | None = None) -> Evaluation:
        """Score the suites in `names`, every suite when None, as one evaluation.

        With no suite at all, the evaluation carries the training values a
        ranking or a stopping rule reads."""
        reports = training if self.checkpoints is not None or plan.best or plan.stop else None
        chosen = list(plan.suites if names is None else names)
        if not chosen:
            return self._evaluate(state, shardings, None, (), plan.preview, split="val", training=reports)
        evaluations = []
        for split in chosen:
            suite = plan.suites[split]
            # The loss is scored where it ranks the run's checkpoints.
            loss = (self.checkpoints is not None and not plan.best
                    and not any(metric.name == 'loss' for metric in suite.metrics))
            evaluations.append(self._evaluate(state, shardings, suite.data, suite.metrics, plan.preview,
                                              loss=loss, split=split, training=reports))
        return dataclasses.replace(
            evaluations[0],
            scores={name: value for report in evaluations for name, value in report.scores.items()},
            elapsed_seconds=sum(report.elapsed_seconds for report in evaluations),
        )

    def _preflight(self, dataset: Dataset, stop, suites: Mapping[str, EvalSuite], *, preview: bool) -> None:
        """The refusals `fit` makes before it places anything: validation
        nothing reads, and a pipeline's batch, whose placement could take
        minutes or run out of memory before the error. Every other mesh's
        batch is checked with its stream (`_check_stream`), a ramp's at each
        of its stages."""
        for name, suite in suites.items():
            if suite.metrics and suite.every is None:
                raise ValueError(f"{name}'s metrics need eval_every or EvalSuite.every to schedule its pass")
            if (stop is None or not isinstance(stop.metric, TrainingScalar)) and suite.every:
                self._check_validation_is_read(name, suite, preview=preview)
        if self.mesh.stage > 1:
            self._check_batch(dataset.batch, self.device_mesh)

    def _check_validation_is_read(self, name: str, suite: EvalSuite, *, preview: bool) -> None:
        """Refuse a validation cadence whose batches nothing would read.

        A validation pass hands its batches to metrics, to the preview and,
        under a checkpointer, to the loss that ranks the run's steps. Scheduled
        with none of them, it would open nothing and report nothing, so the
        contradiction is refused before the run does any work.
        """
        if not suite.metrics and self.checkpoints is None and not (preview and self.tracker is not None):
            raise ValueError(
                f"{name} is scored every {suite.every} steps and nothing consumes it: "
                + ("preview=True needs a tracker to receive the samples; "
                   if preview else "")
                + "pass metrics to score it, preview=True with a tracker to sample from it, "
                "or leave eval_every unset")

    def _own_profile_window(self) -> Profiler | None:
        """Take ownership of the capture a configured window will write.

        A configured window must own the capture. When another profiler
        already holds it, refuse before the dataset or the mesh do any work,
        so neither trace is silently dropped or cut short. The window also
        resolves its optional dependency here, rather than after warmup steps
        have run.

        Every rank either leaves owning a stopped Profiler or raises
        together; a rank that proceeded alone would deadlock its peers at the
        first collective of training.
        """
        profile = self.profile
        if profile is None:
            return None

        def own_window() -> Profiler:
            if telemetry_profile.active_profile() is not None:
                raise ValueError(
                    "Trainer.profile cannot schedule a window while an explicit "
                    "dew.Profiler capture is active; drop one or stop the outer "
                    "profiler before fitting")
            telemetry_profile.require_profile_support()
            return telemetry_profile.Profiler(profile.directory)

        return agreed("profiling window setup", own_window)

    def _check_batch(self, batch: int, mesh: Mesh) -> None:
        """Refuse, before anything is placed, a global batch the mesh cannot
        split (`batch_divisor`): its rows shard over the batch axes as whole
        rows, and a pipeline cuts each device's rows again into M
        microbatches, where a device that holds none of a microbatch's rows
        computes another's again. The message names the row shards, and the
        batches and the microbatch counts that fit. A rollout's rows are its
        own, so a run with one is checked where its step traces."""
        if self.rollout is not None:
            return
        divisor = batch_divisor(mesh, self.mesh)
        if batch % divisor == 0:
            return
        count = self.mesh.microbatches or self.mesh.stage
        shards = divisor // count
        suggestions = [f"a batch that is a multiple of {divisor} rows, {-(-batch // divisor) * divisor} "
                       f"the nearest above {batch}"]
        axes = " x ".join(f"{axis} {mesh.shape[axis]}" for axis in BATCH_AXES if mesh.shape[axis] > 1)
        if count == 1:
            raise LayoutRefused(
                f"a global batch of {batch} rows over the {shards} row shards of {axes} leaves some "
                f"device without whole rows; use {suggestions[0]}, or a mesh whose batch axes "
                f"({' x '.join(BATCH_AXES)}) divide {batch}")
        if batch % shards == 0:
            fitting = [m for m in range(self.mesh.stage, batch // shards + 1, self.mesh.stage)
                       if (batch // shards) % m == 0]
            if fitting:
                suggestions.append(f"microbatches={fitting[-1]}, the most that divide the "
                                   f"{batch // shards} rows a device holds")
        raise LayoutRefused(
            f"a global batch of {batch} rows over {shards} row shards and {count} microbatches "
            f"leaves some microbatch without rows on some device, which then computes another "
            f"microbatch's again; use {' or '.join(suggestions)}")

    def _check_stream(self, source, mesh: Mesh, batch: int, *, checkpointing: bool) -> None:
        """Refuse a training stream this run cannot checkpoint or cannot shard.

        A checkpoint written without the data position would replay the data
        on resume. A `batch` the mesh cannot divide fails where it is placed
        or traced (`_check_batch`), and a ramp stage's, for a later stage,
        an hour in.
        """
        if checkpointing and not isinstance(source, Checkpointable):
            raise ValueError(
                f"checkpoint_every needs a training stream with get_state and "
                f"set_state, and {type(source).__name__} lacks one; a checkpoint "
                f"written without the data position would replay the data on "
                f"resume. Train it with checkpoint_every=None "
                f"(--trainer.checkpoint-every None)")
        if not isinstance(source, RampedStream):
            self._check_batch(batch, mesh)
            return
        if self.accumulation > 1:
            raise ValueError(
                f"a batch ramp grows the records a step reads, and an "
                f"accumulation window of {self.accumulation} pools "
                f"microbatches of one shape into one update; ramp the "
                f"batch or accumulate, not both")
        divisor = batch_divisor(mesh, self.mesh)
        refused = [stage.batch for stage in source.stages if stage.batch % divisor]
        if refused:
            raise ValueError(
                f"the batch ramp reads {refused} records a step at some of "
                f"its stages, and {dict(mesh.shape)} with "
                f"{self.mesh.microbatches or self.mesh.stage} microbatch(es) "
                f"holds a batch that is a multiple of {divisor}")

    def _start_window(self, profiler: Profiler) -> None:
        """Open the profiler's capture on every rank.

        A peer that failed to start raises here. A capture this rank did
        start is closed first, so no live trace outlives the aborted run.
        """
        capturing = False

        def start() -> None:
            nonlocal capturing
            profiler.start()
            capturing = True

        try:
            agreed("profiling window start", start)
        except BaseException as primary:
            if capturing:
                try:
                    profiler.stop()
                except BaseException as failure:
                    primary.add_note(f"Profiler stop failed: {failure!r}")
            raise

    def _rolled_out(self, state: TrainState, batch: Batch) -> tuple[Batch, float]:
        """Sample one batch through the rollout, with the seconds it took.

        Host-side and untraceable: sampling, scoring, advantages. The key
        folds the step key once more, keeping the rollout's draws off the
        step's stream; both are checkpointed, so a resumed run samples
        forward. Fixed shapes mean the step still compiles once.
        """
        began = time.perf_counter()
        assert self.rollout is not None
        key = jax.random.fold_in(jax.random.fold_in(state.key, state.step), 1)
        batch = shard_batch(self.device_mesh, self.rollout(state, batch, key))
        return batch, time.perf_counter() - began

    def _compiled_for(self, compiled: dict[Shapes, tuple[CompiledStep, float | None]],
                      state: TrainState, batch: Batch) -> tuple[CompiledStep, float | None]:
        """Return the step compiled for this batch's shapes, compiling on first sight.

        A ramped run reads a new shape at every stage, so `compiled` keeps
        one step per stage with the FLOPs measured for it.
        """
        shapes = batch_shapes(batch)
        if shapes not in compiled:
            began = time.perf_counter()
            with region("compile"):
                compiled[shapes] = (self.compile(state, batch), self.flops_per_step)
            seconds = time.perf_counter() - began
            links = {axis: AxisLink(link.bytes_per_second, link.spread)
                     for axis, link in self.links.items()}
            self._report(StepCompiled(seconds, recompute_record(self.objective), links), int(state.step))
        return compiled[shapes]

    def _saved_checkpoint(
        self,
        checkpoints: Checkpoints,
        step: int,
        state: TrainState,
        position: bytes | None,
        interval: _Interval,
        *,
        scores: Mapping[str, float] | None = None,
        ranking: Sequence[Ranking] = (),
        training_best: bool = True,
        control: dict | None = None,
        weights_only: bool = False,
    ) -> float:
        """Write one checkpoint with the interval's mean loss, report it and
        start the next interval.

        Returns the seconds it took, the reading of the interval's loss
        included: that read waits on the device, and the wait is time the
        steps did not have."""
        paused = time.perf_counter()
        self._display.status("writing a checkpoint")
        metadata = dict(scores or {})
        if interval.steps and interval.book is not None:
            metadata.setdefault('train/loss', float(jax.device_get(interval.book[0]) / interval.steps))
            if not ranking and training_best:
                ranking = (Ranking('train/loss', metadata['train/loss']),)
        checkpoints.save(step, state, position, metadata, share=DataPartition.of(self.device_mesh),
                         ranking=ranking, control=control, weights_only=weights_only,
                         rung=self._rung(), artifact=self._artifact())
        self._report(CheckpointRequested(checkpoints.directory), step)
        interval.saved(step)
        self._display.status("")
        return time.perf_counter() - paused

    def _saved_local_checkpoint(
        self,
        checkpoints: Checkpoints,
        step: int,
        state: TrainState,
        position: bytes | None,
        *,
        control: dict | None = None,
    ) -> float:
        """Write one checkpoint to the local directory, and report it.

        Returns the seconds it took. The local copy carries no metadata; it
        is the one a restarted node reads back, not the run's record."""
        paused = time.perf_counter()
        self._display.status("writing a local checkpoint")
        checkpoints.save_local(step, state, position, share=DataPartition.of(self.device_mesh),
                               control=control, rung=self._rung(), artifact=self._artifact())
        self._report(CheckpointRequested(str(checkpoints.local_directory), local=True), step)
        self._display.status("")
        return time.perf_counter() - paused

    def _log_interval(self, step: int, loss: jax.Array, aux: dict[str, jax.Array],
                      accepted: jax.Array, state: TrainState, interval: _Interval) -> None:
        """Report one logging interval and start the next.

        Rank zero builds the row; every other rank has nothing to report.
        The report is agreed, so a tracker that failed on rank zero stops its
        peers here rather than at their next collective.
        """
        def report() -> None:
            if jax.process_index() != 0:
                return
            # The interval's numbers need the loss on the host, so this is
            # where the loop waits on the device.
            loss.block_until_ready()
            now = time.time()
            # One read for every number, since each float() of a device
            # array is a copy of its own.
            read = jax.device_get({
                "loss": loss, "aux": aux, "accepted": accepted, "rate": learning_rate(state.opt_state),
                "scale": None if state.scale is None else state.scale.scale,
                "rollout": {} if self.rollout is None else dict(_rollout_metrics(self.rollout))})
            scalars = {"train/loss": float(read["loss"]),
                       **{f"train/{k}": float(v) for k, v in read["aux"].items()},
                       **self._throughput(now - interval.last_log_time, interval.since_log,
                                          interval.samples, interval.flops)}
            scalars["train/accepted"] = float(read["accepted"])
            if read["rate"] is not None:
                scalars["train/learning_rate"] = float(read["rate"])
            if read["scale"] is not None:
                scalars["train/loss_scale"] = float(read["scale"])
            if self.rollout is not None:
                scalars["train/rollout_seconds"] = interval.rollout_seconds
                scalars.update({f"rollout/{name}": float(value) for name, value in read["rollout"].items()})
            self._display.interval(step, scalars)
            if self.tracker is not None:
                self.tracker.log(scalars, step)
            interval.logged(now)

        with region("log"):
            agreed("training reporting", report)

    def _reported_outcome(self, error: BaseException | None, run: _FitRun) -> BaseException | None:
        """Agree the run's cleanup, report its goodput and its outcome.

        Every rank reaches the cleanup agreement, so a rank that failed alone
        is heard here. Goodput is only reported for a run that reached this
        point without an error, since its numbers describe a run that ran.
        """
        try:
            agree_process_phase(error, phase="fit cleanup")
        except BaseException as failure:
            if error is None:
                error = failure
        if error is None:
            def report_goodput() -> None:
                if jax.process_index() == 0:
                    wall = time.perf_counter() - run.started
                    scalars = goodput(wall, run.first_step, run.other)
                    self._display.summary(run.current, wall, scalars,
                                          None if run.loss is None else float(jax.device_get(run.loss)))
                    if self.tracker is not None:
                        self.tracker.log(scalars, run.current)

            try:
                agreed("goodput reporting", report_goodput)
            except BaseException as failure:
                error = failure
        try:
            outcome = FitEnded.outcome(time.perf_counter() - run.started, error,
                                       preempted=run.preempted is not None)
            self._report(outcome, run.current)
        except BaseException as failure:
            if error is None:
                error = failure
            else:
                error.add_note(f'Reporting fit outcome failed: {failure!r}')
        return error

    def _report(self, value: Record, step: int) -> None:
        """Hand one record to the tracker on rank zero, then agree with the pool.

        The record's type names the phase, so a rank that failed to report is
        heard about at the report the pool was making, not at the next
        collective."""
        def send() -> None:
            if jax.process_index() == 0 and self.tracker is not None:
                self.tracker.artifact(value, step)

        agreed(type(value).__name__, send)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        state: TrainState,
        shardings: Placement[TrainState],
        reader: Reader | None,
        metrics: Sequence[Metric],
        preview: bool,
        *,
        loss: bool = False,
        split: str = "val",
        training: Mapping[str, jax.Array] | None = None,
    ) -> Evaluation:
        """Score the validation split with this state's variables and report
        it, returning the seconds it took: the run's, but not its steps',
        which is what the goodput fraction is measured against.

        A CPU-owned run evaluates on the accelerator, over the same snapshot
        a step realizes, so validation reads the weights where the loss
        does."""
        paused = time.perf_counter()
        self._display.status("evaluating")
        mesh, params = self.device_mesh, state.variables
        averaged = with_ema(state.variables, self._fetched(state, shardings).ema)
        key = state.key
        if self.host_master:
            from dew.training.execution import HostExecution
            execution = HostExecution(self.objective, self.layout, mesh, self.state_mesh)
            with jax.set_mesh(self.state_mesh):
                params, averaged = execution.snapshot(params), execution.snapshot(averaged)
            key = execution.on_accelerator(key)
        assert params is not None, "evaluation always has model variables"
        # The rules and the microbatch count; `Evaluation.run` scores under the
        # mesh itself, and previews decode outside it.
        with (pipeline_microbatches(self.mesh.microbatches),
              nn.logical_axis_rules(self.layout.axis_rules), measured_links(self._links(mesh))):
            evaluation = Evaluation.run(
                self.objective, params, reader, metrics=metrics, key=key,
                step=state.step, schedule_step=state.microstep,
                averaged=averaged, preview=preview, mesh=mesh, loss=loss, split=split, training=training)
        # The training values ride along for ranking (`_ranking`); the log
        # interval reports them, so the evaluation's own report leaves them out.
        self._report_evaluation(dataclasses.replace(evaluation, scores={
            name: value for name, value in evaluation.scores.items() if name not in (training or {})}))
        return dataclasses.replace(evaluation, elapsed_seconds=time.perf_counter() - paused)

    def _report_evaluation(self, advanced: Evaluation) -> None:
        """Print one evaluation on rank zero and log its previews and scores."""
        def report() -> None:
            if jax.process_index() != 0:
                return
            self._display.evaluation(advanced)
            if self.tracker is not None:
                for artifact in advanced.previews:
                    self.tracker.artifact(artifact, advanced.step)
                scalars = advanced.scalars
                if scalars:
                    self.tracker.log(scalars, advanced.step)

        agreed("evaluation reporting", report)


    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _stop_trace(self, run: _FitRun, profile: ProfileWindow, profiler: Profiler) -> None:
        """Stop the window's owned capture, then report it on process zero.

        The core Profiler drains the backend and exports the native reports
        on `stop`. The run's last loss is blocked on only to order the
        primary's failure ahead of the profiler's own drain."""
        loss, traced = run.loss, run.traced

        def stop() -> None:
            try:
                if loss is not None:
                    loss.block_until_ready()
            finally:
                primary = sys.exception()
                try:
                    profiler.stop()
                except BaseException as failure:
                    if primary is None:
                        raise
                    primary.add_note(f"Profiler stop failed: {failure!r}")

        def announce() -> None:
            self._display.note(f"Wrote profile for {traced} steps to {profile.directory}")

        agreed("profile stop", stop)
        self._report(ProfileWindowRecord(profile.directory, traced), run.current)
        agreed("profile announcement", announce)

    def _throughput(self, elapsed: float, steps: int, samples: int,
                    flops: float | None) -> dict[str, float]:
        """Compute the interval's rates from the records it read and its FLOPs.

        The FLOPs are the ones the compiler measured for each step's own
        shape; `flops` is None when a shape's measurement was unavailable.

        An interval that spans a stage boundary carries that stage's compile
        in its wall time. MaxText instead hides the performance metrics
        during a ramp (`common/metric_logger.py:166-194`).
        """
        if elapsed <= 0 or steps <= 0:
            return {}
        step_time = elapsed / steps
        scalars = {"train/step_time_ms": step_time * 1000,
                   "train/samples_per_sec": samples / elapsed}
        mfu = None if flops is None else model_flops_utilization(flops / steps, step_time)
        if mfu is not None:
            scalars["train/mfu"] = mfu
        return scalars
