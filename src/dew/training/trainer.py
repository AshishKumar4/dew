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
import os
import sys
import time
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Generic

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P
from typing_extensions import TypeVar as DefaultTypeVar

from dew._coordination import (
    agree_process_phase,
    agreed,
    agreed_next,
    agreed_same,
    broadcast_from_process_zero,
)
from dew._records import JSON, boolean, json_value, record
from dew.batch import FROZEN, Batch, Variables, owned_dicts, select
from dew.checkpoints import Checkpoints, Ranking, Resume
from dew.cleanup import attempted
from dew.data.dataset import (
    Closeable,
    DataPartition,
    Loader,
    PhasedStream,
    RampedStream,
    Reader,
    Stage,
    rows_of,
    stream_endless,
    stream_identity,
)
from dew.data.pipeline import Dataset, DeviceStep
from dew.distributed.batches import (
    PREFETCH_DEPTH,
    DevicePrefetchIterator,
    Signature,
    batch_layout,
    batch_shardings,
    batch_signature,
    placed_abstract,
    prefetched_bytes,
    split_fields,
)
from dew.distributed.host import snapshot
from dew.distributed.placement import (
    Layout,
    MeshSpec,
    Placement,
    Validation,
    device_mesh,
    link_topology,
    placed_bytes,
    placed_shapes,
)
from dew.inputs import Field, InputSpec
from dew.mesh import SEQUENCE_AXIS, STAGE_AXIS, TENSOR_AXIS, LayoutRefused, LogicalAxes
from dew.nn.flops import MissingFlopsCount
from dew.nn.kernels import tuning
from dew.nn.kernels.generation import assumed, backend, gpu_mesh, gpu_target, measured_kernel
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import (
    Link,
    Schedule,
    boxed,
    boxed_axes,
    measured_links,
    pipeline_microbatches,
    ruled_boxes,
)
from dew.objectives.base import (
    Aux,
    Initializer,
    Metric,
    Objective,
    Ratio,
    Recorded,
    Step,
    TrainingScalar,
    taken_record,
)
from dew.optim import clips_qk, held_rates, learning_rate, scaled_rates, settable_rates, with_rates
from dew.state import MODEL_STATE, Accumulation, TrainState, apart, kept_window, together
from dew.telemetry import profile as telemetry_profile
from dew.telemetry.instrumentation import StepFlops, compilation_cache_counts
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
from dew.training._fit import (
    FitPlan,
    FitRun,
    Start,
    declarations,
    first_batch,
    fit_policies,
    stage_reached,
    unrecorded_parts,
    waited_read,
)
from dew.training._lowering import Lowered, Routed, routed
from dew.training.budget import Pause, Plateau
from dew.training.cooldown import Cooldowns, Pending, check_branches, decay, resolved, schedule_ended
from dew.training.display import TRAINER_SHOWN, TrainingDisplay, goodput
from dew.training.distributed import (
    check_batch,
    check_stream,
    link_bandwidth,
    link_figures,
    marked,
    microbatch,
    microbatch_shapes,
    placed_start,
    resumed_link_figures,
    ruled,
    stage_microbatches,
    step_reads,
    traced_start,
)
from dew.training.evaluation import EvalSuite, Evaluation, Summary
from dew.training.interval import Interval, TokenClock, held, throughput
from dew.training.memory import (
    climb_to,
    first_run,
    head_tile_of,
    local_devices,
    memory_record,
    peak_memory_gib,
    recompute_record,
    refuse_wide_floats,
    rung_record,
    split_share,
    step_options,
)
from dew.training.model_flops import (
    TOKENS,
    TRAINING_FLOPS,
    _TokenCount,
    batch_tokens,
    model_batch,
    prepare_counts,
    uncounted,
)
from dew.training.narrow import narrowed, narrowed_paths
from dew.training.observe import Observe
from dew.training.planning import (
    Candidate,
    FitSearch,
    Planned,
    Trials,
    faster,
    read_whole,
    settled,
    splits,
    untimed_frontier,
    whole_step,
)
from dew.training.profiling import ProfileWindow, owned_capture, pgle_runs, start_capture, stop_capture
from dew.training.rollout import Rollout, _branched, _reported, _rollout_metrics, rolled_out
from dew.training.rungs import keep_rung, plan_flops, recorded_launches, recorded_rung
from dew.training.runtime import Preempted, PreemptionNotice
from dew.training.selection import Best, rankings
from dew.training.startup import Startup, run_data
from dew.training.tracker import Tracker
from dew.training.transaction import (
    COMMIT_FIELDS,
    Committed,
    Transaction,
    compact_qk,
    loss_scale,
    outcome,
    window_dropped,
    with_ema,
)

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dew.optim import OptimConfig
    from dew.telemetry.profile import Profiler

StepFn = Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]
"""A compiled step's body: the state and the global batch in, the new state,
the loss and the objective's report out."""

CompiledStep = Callable[
    [TrainState, Batch],
    tuple[TrainState, jax.Array, dict[str, jax.Array], jax.Array, jax.Array, Committed | None]]
"""A call returns the stepped state, the scalar loss, the objective's whole
report (`Aux.metrics`), whether that loss was finite, and whether the
step's microbatches were accepted, and the transaction's commit telemetry (None for a custom step)."""

Loss = DefaultTypeVar("Loss", default=Ratio | jax.Array | float)
Effects = DefaultTypeVar("Effects", default=None)

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
        mesh: MeshSpec | Mesh = _DEFAULT_MESH,
        layout: Layout = _DEFAULT_LAYOUT,
        microbatches: int | None = None,
        accumulation: int = 1,
        split: int | None = None,
        dynamic_scale: bool | dynamic_scale_lib.DynamicScale = False,
        checkpoints: Checkpoints | None = None,
        tracker: Tracker | None = None,
        step: Callable[[Objective[Loss, Effects], optax.GradientTransformation], StepFn] | None = None,
        rollout: Rollout | None = None,
        profile: ProfileWindow | None = None,
        lower_for: str | None = None,
        tune_kernels: bool = False,
        record: bool = False,
        links: Mapping[str, Link] | None = None,
    ):
        """Hold everything a run needs, without opening any of it.

        The mesh, the compiled step and the capabilities' resources come into
        being in `fit`, so constructing a Trainer allocates nothing.

        `accumulation` is how many steps pool into one optimizer commit, and `split` how many
        microbatches each step's batch is read in, its rows split evenly; None plans it, the
        fewest that fit the devices before any recomputation (`compile`). `optimizer`
        is an optax transformation, or an `OptimConfig` each `fit` builds over its `steps //
        accumulation` updates; the run steps the objective's (`Objective.optimizer`), made from
        it. `step` replaces the built-in transaction. A custom step then owns the clocks, the
        scaler, the EMA and the mutable writes, and the compiled wrapper owns only the
        attempted-step counter. `rollout` runs once per batch read, before the step and outside
        replay. `layout` and `mesh` say where the state lives: `mesh` a `MeshSpec` or a native
        `jax.sharding.Mesh` of any axes, which `layout`'s rules name (`Layout.data_axes` its rows').
        `microbatches` is how many a pipelined step feeds through the mesh's stage axis, a
        multiple of its size; None feeds one a stage. `lower_for`, a CUDA generation, lowers
        the first step for it in place of compiling it (`dew.training._lowering.Lowered`).
        `tune_kernels` opts in to the tuner (`dew.nn.kernels.tuning.tune`), else each launch takes its
        derived config. `record` makes each step return in its metrics what each evaluation, replays
        too, sampled or passed as a boundary (`taken_record`).
        `links` states an axis's `Link` by name in place of measuring it (`_links`).
        """
        whole = read_whole(accumulation, split, step, layout, rollout, mesh)
        self.objective = objective
        built = isinstance(optimizer, optax.GradientTransformation)
        self._config = None if built else optimizer
        self._optimizer = objective.optimizer(optimizer) if built else None
        from dew.nn.inputs import request_key
        self.seed = int(key) if isinstance(key, (int, np.integer)) and not isinstance(key, bool) else None
        self.key = request_key(key)
        self.mesh = mesh
        self.layout = layout
        self._microbatches = microbatches
        self.accumulation = accumulation
        self.split = split
        self._loss_scale = loss_scale(dynamic_scale)
        self.dynamic_scale = self._loss_scale is not None
        self.checkpoints = checkpoints
        self.tracker, self.step, self.rollout = tracker, step, rollout
        if record and step is not None:
            raise ValueError("a custom step builds its own Steps, which Trainer(record=True) cannot reach: "
                             "build them with Step(record=True)")
        self.record = record
        self.records: list[Recorded] = []
        self.profile = profile
        self.lower_for, self.tune_kernels = lower_for, tune_kernels
        # Set by `compile`, for the batch shape it was called with. A ramped run has one value per stage;
        # `fit` keeps them beside each step. The executable is the step as compiled, whose memory analysis a
        # benchmark reads; the program it was lowered from is released with its plan (`Planned.released`).
        self.step_flops: StepFlops | None = None
        self.executable: jax.stages.Compiled | None = None
        # The plan of the last compiled step: its split, the rows of its
        # microbatch and every compile its search made (`Probe`).
        self.plan: JSON = None
        # The check that the last compiled step fits (`memory_record`), and whether a step compiled
        # since has yet to run once.
        self.memory: JSON = None
        self._unproven = False
        # The last validation of the state between steps (`validate`): as placed, then as each step holds it.
        self.validation: Validation | None = None
        # The links the last compiled step decided its projections by, which say whether one spread over
        # each axis (`dew.nn.sharding.Link`), and each mesh axis's measured bandwidth.
        self.links: dict[str, Link] = {}
        self._stated_links = dict(links or {})
        self._bandwidths: dict[tuple[Mesh, str], float | None] = {}
        self._display = TrainingDisplay()
        self._refusal: str | None = None
        # The ladder's rungs past the objective's head and remat that a step fit at (no narrow copies, XLA's
        # defaults) stay for later compiles; a resumed run starts at its checkpoint's rung (`_climb_to`).
        self._narrow, self._xla_defaults = True, False
        # Why this run's checkpoints describe no model to load, said once.
        self._unrecorded: str | None = None
        # The axes the objective's modules box on their own parameters, as the
        # last trace of `initial_state` saw them (`_boxed_axes`).
        self._boxed: dict[tuple[str, ...], LogicalAxes] | None = None
        self._resumed_rung: JSON = None
        # The rung each newly met shape starts from, as the first compile found the objective: its modules,
        # head tile, XLA option choice and record (`_at_base_rung`); then the rung each shape compiled at, by
        # layout, and the rungs a resumed checkpoint recorded for its shapes.
        self._base_rung: tuple[tuple[nn.Module, ...], tuple[int, int] | None, bool, dict[str, JSON]] | None
        self._base_rung = None
        self._shape_rungs: dict[str, JSON] = {}
        self._resumed_shapes: dict[str, JSON] = {}
        self._resumed_links: dict[str, float] = {}
        # The microbatches a step's batch is read in, once decided (`split`,
        # 1 where `whole`, a pooling state's window, or the first plan), which
        # cut every batch the run reads (`_rows`, read off `fit`'s data).
        self._split: int | None = 1 if whole is not None else split
        self._rows: tuple[int, ...] | None = None
        # The stages of the batch ramp `fit` reads, which change the batch's shape; none without one.
        self._ramp: tuple[Stage, ...] = ()
        self._stage_starts: frozenset[int] = frozenset()
        # The training batch's device steps, which the compiled step runs (`Loader.device`).
        self._device: tuple[DeviceStep, ...] = ()
        self._prefetch = PREFETCH_DEPTH

    @property
    def optimizer(self) -> optax.GradientTransformation:
        if self._optimizer is None:
            raise ValueError("an OptimConfig is built over the run's length, which only fit(steps=...) gives")
        return self._optimizer

    # ------------------------------------------------------------------ The state
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
        # The state holds its trees as dicts of its own, whichever Flax container (a FrozenDict, under a
        # mapping or a list) the objective's tree came in, so its placement, its updates, the
        # collections a step writes back (`write_back`) and its restores share one structure.
        params = owned_dicts(nn.unbox(initialized))
        # A held tree's initializer returns the objective's own arrays (`Objective.held_variables`),
        # which the step's donation would free: the moving state takes buffers of its own, as a
        # placed one does (`distributed.placed_held`). The frozen tree no step donates (`apart`), so
        # it shares the held arrays as they are, one copy for the run. Traced, no leaf is a held array,
        # so nothing is copied; a state held on the host is never donated, and takes the arrays over
        # leaf by leaf as it is placed (`execution.placed_host`), so it takes them as they are.
        if not self.host_master:
            held = {id(leaf) for leaf in jax.tree.leaves(self.objective.held_variables())
                    if isinstance(leaf, jax.Array)}
            params = {collection: tree if collection == FROZEN else jax.tree.map(
                lambda leaf: leaf.copy() if id(leaf) in held else leaf, tree)
                for collection, tree in params.items()}
        if "params" not in params:
            raise ValueError(
                f"the objective's tree has no params collection, only {sorted(params)}; "
                "the optimizer moves params and treats every other collection as state")
        ema = self.objective.ema
        return TrainState(
            step=jnp.zeros((), jnp.int32),
            microstep=jnp.zeros((), jnp.int32),
            updates=jnp.zeros((), jnp.int32),
            # A state of its own, the given scale's leaves copied: the step donates the state.
            scale=None if self._loss_scale is None else jax.tree.map(
                lambda leaf: jnp.array(leaf, copy=True), self._loss_scale),
            window_size=jnp.asarray(self.accumulation, jnp.int32),
            variables=params,
            opt_state=self._boxed_init(initialized, params["params"]),
            # The average starts equal to the parameters but as its own buffers: the step donates
            # the state, and a buffer can be donated once.
            ema=None if ema is None else jax.tree.map(jnp.copy, select(params, ema.select)),
            key=run_key,
        )

    @functools.cached_property
    def device_mesh(self) -> Mesh:
        """The device mesh (`device_mesh`), built on first use."""
        return device_mesh(self.mesh)

    @property
    def microbatches(self) -> int:
        """The microbatches a pipelined step feeds through the stage axis (`stage_microbatches`)."""
        return stage_microbatches(self.device_mesh, self._microbatches)

    @property
    def host_master(self) -> bool:
        return "variables" in self.layout.host

    @functools.cached_property
    def state_mesh(self) -> Mesh:
        if not self.host_master:
            return self.device_mesh
        from dew.distributed.host import companion_mesh
        return companion_mesh(self.device_mesh)

    def _boxed_init(self, initialized: Variables, params: Variables) -> optax.OptState:
        """The optimizer's state of `params`, Muon's grouping reading the axes
        modules boxed on them in `initialized`."""
        with boxed(ruled_boxes(boxed_axes(initialized), self.layout.axis_rules)):
            return self.optimizer.init(params)

    def _boxed_axes(self) -> dict[tuple[str, ...], LogicalAxes]:
        """The axes the objective's modules box on their own parameters
        (`nn.with_logical_partitioning`), which its init returns and the state
        unboxes: what `initial_state` last traced, else one abstract init, for
        a state the trainer did not build. A checkpoint records them for its
        restore."""
        if self._boxed is None:
            self._boxed = boxed_axes(jax.eval_shape(self.objective.initializer, jax.random.key(0)))
        return self._boxed

    @ruled
    def shardings(self, state: TrainState) -> Placement[TrainState]:
        """Return where each field of `state` is placed, on the axes that suit its kind.

        Parameter gradients follow parameters, replay records follow batches, and the
        layout's host-resident fields go in pinned host memory. Under a CPU-owned
        state, the frozen collection is the exception: it stays where the realization
        reads it (`execution.resident`) for the whole run.
        """
        with self._declared():
            return self._shardings(state)

    def _declared(self):
        """The context in which the layout reads the axes the objective's
        modules box, as its rules place them."""
        return boxed(ruled_boxes(self._boxed_axes(), self.layout.axis_rules))

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

    def _fetched(self, state: TrainState, shardings: Placement[TrainState],
                 fields: Sequence[str] | None = None) -> TrainState:
        """Bring the layout's host-resident fields, or those of them among `fields`, to the device,
        where a step or an evaluation reads them."""
        if self.host_master:
            return state
        return dataclasses.replace(state, **{
            field: jax.device_put(getattr(state, field), jax.tree.map(
                lambda s: s.with_memory_kind("device"), getattr(shardings, field)))
            for field in self.layout.host if fields is None or field in fields})

    @ruled
    def validate(self, batch: Batch | None = None, *, steps: int | None = None,
                 capacity: int | None = None) -> Validation:
        """What `fit` checks of the state before it places it (`Layout.validate`), read without
        placing anything, so a run is checked on a CPU host before a costly launch.

        The state is a fresh run's: its parameters, optimizer state and EMA, and, given one
        global `batch` as the run reads it, the accumulation its steps pool where `accumulation`
        is above 1, read in the first of the splits its plan may take whose devices hold it, the
        trainer's own where it fixes one (`splits`). It leaves out the narrow copies a step may read
        (`TrainState.compute`), which the plan drops first where the step does not fit with them.
        An `OptimConfig` is built over `steps`, as `fit(steps=...)` builds it. `capacity` is the
        bytes each device has, for a mesh whose devices report none: a CPU mesh of
        `--xla_force_host_platform_device_count` devices standing in for GPUs.

        Raises what `fit` raises before placing the state: `LayoutRefused` for a state over a
        device's memory or a layout that replicates too much of the model, and ValueError for
        rules that name an axis the mesh lacks.
        """
        self._schedule(None if steps is None else steps // self.accumulation)
        state = jax.eval_shape(self.initial_state)
        if batch is None:
            return self._validated(state, capacity)[1]
        return self._fitting(state, marked(DataPartition(), self._counted_batch)(batch), capacity)

    def _validated(self, state: TrainState, capacity: int | None = None
                   ) -> tuple[Placement[TrainState], Validation]:
        """`state`'s placement and what `Layout.validate` reads of the state it places, as the
        layout placed it, once every refusal before a placement passed: a float the devices
        cannot hold, a layout that replicates too much and a state over a device's memory."""
        refuse_wide_floats(state, self.device_mesh)
        shardings = self.shardings(state)
        variables, placement = state.variables, shardings.variables
        if self.host_master and FROZEN in variables:
            # A CPU-owned run's frozen banks, read as the layers the layout placed.
            from dew.inference.banks import row_sharding, shape_rows, spread
            variables = {**variables, FROZEN: spread(variables[FROZEN], shape_rows)}
            placement = {**placement, FROZEN: spread(
                placement[FROZEN], lambda sharding, count: [row_sharding(sharding)] * count)}
        with self._declared():
            self.layout.check(variables, placement, self.device_mesh)
        self.validation = self._validation(placed_shapes(dataclasses.replace(state, variables=variables),
                                                         dataclasses.replace(shardings, variables=placement)),
                                           capacity)
        _log.info("the state holds %.2f GiB on its fullest device between steps",
                  max(self.validation.device_bytes.values()) / 2 ** 30)
        return shardings, self.validation

    def _validation(self, placed: TrainState, capacity: int | None = None) -> Validation:
        """`Layout.validate` of `placed`, read in every axis the modules box, which every process
        agrees, each reading its own devices' memory."""
        with boxed(self._boxed_axes()):
            return agreed("state placement", functools.partial(
                self.layout.validate, self.device_mesh, placed, capacity=capacity))

    @ruled
    def place(self) -> tuple[TrainState, Placement[TrainState], bytes | None]:
        """Put the state on the mesh, fresh or restored, once `validate`'s checks pass.

        It returns the state with its shardings and the data position a resume
        continues from.
        """
        return self._placed(self._started(), positioned=True)

    def _started(self, resume: Resume = "auto") -> Start:
        """What placing the state reads before it places anything: the objective's held tree and
        the key, resolved once, so that the shapes and the values are the same inputs through the
        same overridable method and the objective is asked for what it holds exactly once; the
        state's shapes; the checkpoint a run resumes from as `resume` says (`Checkpoints.resumed`), the
        one every process agrees; and the placement of the state, once its checks pass (`_validated`)."""
        initializer, key = self.objective.initializer, self.key
        if self.host_master:
            from dew.distributed.host import transfer
            key = transfer(key, NamedSharding(self.state_mesh, P()))
        abstract, through = traced_start(self.initial_state, initializer, key)
        checkpoints = self.checkpoints
        # Each process lists the checkpoints itself, and on a shared filesystem one listing can lag
        # another's: every process resumes from one step or none does.
        step = (None if checkpoints is None
                else agreed_same("newest checkpoint", lambda: checkpoints.resumed(resume)))
        if checkpoints is not None and step is not None:
            abstract = dataclasses.replace(abstract, accumulation=checkpoints.accumulation_template(step))
        shardings, _ = self._validated(abstract)
        return Start(initializer, key, abstract, through, step, shardings)

    def _placed(self, start: Start, *, positioned: bool
                ) -> tuple[TrainState, Placement[TrainState], bytes | None]:
        """The state `start` describes on the mesh, fresh or restored, with its shardings and, where
        `positioned`, the data position a resume continues from; `fit` reads that alone, before it
        places the state (`Checkpoints.position`), and restores the state alone."""
        initializer, key, abstract, shardings, resume = (start.initializer, start.key, start.abstract,
                                                         start.shardings, start.resume)
        checkpoints = self.checkpoints
        if checkpoints is None or resume is None:
            if self.host_master:
                from dew.training.execution import placed_host
                return (placed_host(self.objective, lambda: self.initial_state(initializer, key),
                                    self.state_mesh, self.device_mesh, shardings), shardings, None)
            return (placed_start(self.initial_state, initializer, key, abstract, shardings, start.through,
                                 self.device_mesh), shardings, None)
        if self.host_master and FROZEN in abstract.variables:
            from dew.training.execution import banked_frozen
            abstract = dataclasses.replace(abstract, variables={
                **abstract.variables, FROZEN: banked_frozen(
                    self.objective, abstract.variables,
                    lambda rows, path: jax.ShapeDtypeStruct((len(rows), *rows[0].shape), rows[0].dtype))})
        share = DataPartition.of(self.device_mesh) if positioned else None
        state, position = checkpoints.restore(placed_shapes(abstract, shardings), resume, share=share)
        if self._config is not None:
            # The rate this run configures, not the one its checkpoint holds.
            state = dataclasses.replace(state, opt_state=with_rates(state.opt_state, self._config.rates()))
        self._resume_rung(checkpoints.rung(resume))
        from dew.nn.inputs import request_key
        state = dataclasses.replace(state, key=jax.device_put(request_key(state.key), shardings.key))
        if int(state.window_size) % self.accumulation:
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

    def _rung(self) -> dict[str, JSON]:
        """The fit ladder's rung this trainer's step compiles at: the
        objective's head tile, its trained modules' remat (`recompute_record`) and whether
        the step keeps XLA's default options, the half rung `compile` climbs
        to before the next. A checkpoint records it, since each decides the
        program a step runs."""
        return rung_record(self.objective, self._xla_defaults, narrow=self._narrow)

    def _checkpoint_rung(self) -> JSON:
        """The rung a checkpoint records: the run's own, which every newly met
        shape starts from (`_rung`'s fields as the first compile found them),
        and under `shapes` the rung each shape compiled at, by its layout
        (`batch_layout`). The split is the state's (`window_size`), and its
        link figures by placement (`link_figures`), a resume's own there."""
        base = self._rung() if self._base_rung is None else self._base_rung[3]
        links = link_figures(self._bandwidths)
        return {**base, 'shapes': dict(self._shape_rungs), **({'links': links} if links else {})}

    def _resume_rung(self, rung: JSON) -> None:
        """Take a checkpoint's rung (`_checkpoint_rung`): the objective climbs
        to the run's own rung, and each shape the checkpoint names climbs to
        its own when it is compiled (`_at_base_rung`). A rung recorded before
        shapes had their own names none, so every shape starts at it."""
        self._climb_to(rung)
        self._resumed_links = {} if rung is None else resumed_link_figures(rung)
        if rung is not None:
            shapes = record(record(rung, 'rung').get('shapes', {}), 'rung shapes')
            self._resumed_shapes = {name: json_value(shape, f'rung of {name}')
                                    for name, shape in shapes.items()}

    def _at_base_rung(self, batch: Batch) -> JSON:
        """Put the objective at the rung a newly met shape starts from, and
        return the rung a resumed checkpoint recorded for this shape, or None.

        Each shape fits on its own. A run bucketed by length meets a long
        bucket that climbs, and when that rung stayed, every later, shorter
        bucket compiled at it too, though it fitted with less recomputation
        (in one real mix, six shapes ran at 'minimal' that fitted with none,
        15% slower a token). So each new shape starts where the first compile
        found the objective, its configured remat and head or the rung its
        checkpoint trained on, and climbs only as far as it must. A shape the
        checkpoint names climbs to that shape's own rung, so a resumed run
        compiles each shape as the run it continues did."""
        if self._base_rung is None:
            self._base_rung = (tuple(entry.module for entry in self.objective.program_key()),
                               head_tile_of(self.objective), self._xla_defaults, self._rung())
        else:
            modules, tile, defaults, rung = self._base_rung
            self.objective.substitute(list(modules))
            self.objective.restore_head(tile, fused=bool(rung['head_fused']))
            self._xla_defaults, self._narrow = defaults, bool(rung['narrow'])
        recorded = self._resumed_shapes.get(batch_layout(batch))
        if recorded is None:
            return None
        self._climb_to(recorded)
        return self._rung()

    def _climb_to(self, rung: JSON) -> None:
        """Move the objective up to `rung`, a checkpoint's (`_rung`), where
        it is above where the objective stands, never below it.

        The fit check reads the free memory its own process finds, and one
        that restores a state can find more than the one that built it and
        pick a lighter rung, another program. So the resumed run compiles the
        rung its checkpoint trained on, and climbs further only where that
        does not fit; a ladder rung the record does not name it never climbed."""
        if rung is None:
            return
        fields = record(rung, 'rung')
        climb_to(self.objective, fields)
        self._xla_defaults = self._xla_defaults or boolean(
            fields.get("xla_defaults", False), "rung xla_defaults"
        )
        self._narrow = self._narrow and boolean(fields.get("narrow", True), "rung narrow")
        self._resumed_rung = self._rung()


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
        from dew.training.execution import resident, stack_sites
        rows = self.layout.shardings(
            self.state_mesh, dataclasses.replace(state, variables={FROZEN: frozen}, accumulation=None))
        sites = stack_sites(self.objective, {**state.variables, FROZEN: frozen})
        return resident(rows.variables[FROZEN], sites, self.device_mesh)

    # ------------------------------------------------------------------ The step
    # ------------------------------------------------------------------

    def _model_batch(self, batch: Batch) -> Batch:
        """The objective's inputs: the batch after the dataset's device steps,
        traced into the compiled step, without the trainer's accounting columns."""
        for step in self._device:
            batch = step(batch)
        return model_batch(batch)

    def _model_shapes(self, batch: Batch) -> Batch:
        """`_model_batch`'s output, traced only where device steps change it,
        each leaf of an Explicit axis typed as the step reads it (which
        `jax.eval_shape` drops on a mesh whose other axes are Auto)."""
        if not self._device:
            return model_batch(batch)
        return jax.tree.map(lambda aval: jax.ShapeDtypeStruct(
            aval.shape, aval.dtype, sharding=aval.sharding if aval.sharding.mesh.explicit_axes else None,
            weak_type=aval.weak_type), jax.jit(self._model_batch).trace(batch).out_avals)

    def _loss_shape(self, state: TrainState, batch: Batch, window: int):
        batch = self._model_shapes(batch)
        # Shapes and dtypes only: a resident frozen leaf sits in another memory space than the moving ones,
        # and the loss the realization runs reads a snapshot in one space. The step's key is drawn inside the
        # trace, so an abstract state compiles a step as a placed one does.
        params = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

        def loss(params, batch, microstep, key, step, ema):
            return self.objective._loss(params, batch, Step(
                microstep, jax.random.fold_in(key, step), with_ema(params, ema),
                qk_stats=clips_qk(state.opt_state), example_count=window * rows_of(batch) or None))

        return jax.eval_shape(loss, params, batch, state.microstep, state.key, state.step, state.ema)

    def _initialize_accumulation(self, state: TrainState, batch: Batch, shapes, *, window: int,
                                 shape_only=False):
        """The state with the accumulation a `window` of microbatches of `batch`'s shapes pools in,
        empty; unchanged under a window of one, a custom step or a held one that replays such batches
        (`Accumulation.replays`): a ramp's stage begins with a window (`Ramp.stages`)."""
        if window == 1 or self.step is not None:
            return state
        stats, aux = shapes
        shared = isinstance(stats, (Ratio, jax.ShapeDtypeStruct))
        mean_dtype = jnp.result_type(jnp.float32, *(x.dtype for x in jax.tree.leaves(stats)))
        slots = window - 1
        def shape(leaf: jax.Array) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)

        trainable = jax.tree.map(shape, state.variables["params"])
        records = jax.tree.map(shape, self._model_shapes(batch))
        mutable = (None if shared or aux.variables is None else
                   jax.tree.map(shape, {name: state.variables[name] for name in aux.variables}))

        def zeros(leaf):
            wide = at_least_fp32(leaf.dtype)
            return jnp.zeros(leaf.shape, wide if jnp.issubdtype(leaf.dtype, jnp.inexact) else leaf.dtype)

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
                    lambda x: jnp.full(x.shape, -jnp.inf, at_least_fp32(x.dtype))
                    if jnp.issubdtype(x.dtype, jnp.inexact) else zeros(x),
                    compact_qk(jax.tree.map(zeros, aux.qk_stats))),
                batches=None if shared else buffer(records),
                variables=None if mutable is None else buffer(mutable),
                attempts=None if shared else jnp.zeros((slots,), jnp.int32),
                tokens=jnp.zeros((2,), jnp.uint32) if TOKENS in batch else None)

        pending = jax.eval_shape(allocate)
        if kept_window(state.accumulation, pending, None if shape_only else int(state.microstep) % window):
            return state
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
        axis's bandwidth is measured once per mesh, and not at all for a
        device the peak table does not name (a CPU's among them), where it
        decides nothing; a link the run states (`links`) is taken as stated."""
        devices = list(mesh.devices.flat)
        peaks = [peak for device in devices if (peak := gpu_target(device.device_kind).bf16_peak) is not None]
        peak = max(peaks) if len(peaks) == len(devices) else None
        links = {}
        for axis in (TENSOR_AXIS, SEQUENCE_AXIS):
            if mesh.shape.get(axis, 1) == 1:
                continue
            if axis in self._stated_links:
                stated = self._stated_links[axis]
                links[axis] = Link(stated.bytes_per_second, stated.flops_per_second)
                continue
            if (mesh, axis) not in self._bandwidths:
                # A resumed run takes the figure its checkpoint traced with, on this very placement.
                self._bandwidths[mesh, axis] = None if peak is None else (
                    self._resumed_links.get(link_topology(mesh, axis)) or link_bandwidth(mesh, axis))
            links[axis] = Link(self._bandwidths[mesh, axis], peak)
        return links

    @contextlib.contextmanager
    def _traced_on(self, mesh: Mesh | AbstractMesh,
                   links: Mapping[str, Link] | None = None) -> Iterator[Schedule]:
        """What a traced step reads from context: the mesh, the pipeline's
        microbatch count, the layout's rules, which place the activations
        the model constrains (`dew.nn.sharding.constrain`) as they place the
        parameters, and the links of the axes the step measured, none by
        default. An abstract mesh is a GPU's this process lacks (`_lowered_for`)."""
        on = jax.sharding.use_abstract_mesh(mesh) if isinstance(mesh, AbstractMesh) else jax.set_mesh(mesh)
        with (on, pipeline_microbatches(self.microbatches) as schedule,
              nn.logical_axis_rules(self.layout.axis_rules), measured_links({} if links is None else links)):
            yield schedule

    def _step_body(self, shapes, copies, split: int, fetch: Callable[[TrainState], TrainState] | None = None):
        """The objective transaction over a step's batch read in `split` microbatches, which brings the
        state's fields `fetch` brings to the device only once its gradient is ready, or a custom step
        with no invented commit telemetry, which reads it whole."""
        if self.step is None:
            return Transaction(self.objective, self.optimizer, split * self.accumulation, shapes, copies,
                               split=split, record=self.record, fetch=fetch).step()
        custom = self.step(self.objective, self.optimizer)

        def run(current, batch, part, tokens):
            advanced, loss, aux = custom(current, batch)
            return advanced, loss, aux, None
        return run

    @ruled
    def compile(self, state: TrainState, batch: Batch, trials: Trials | None = None) -> CompiledStep:
        """Compile a transaction over the state and one already-produced global batch.

        The step consumes the state it is given. The returned state takes over its
        buffers, so the update runs in place and peak memory holds one copy of the
        parameters and optimizer state, not two. Keep no reference to a state after
        stepping it; `new = step(old, batch)` is the whole contract.

        A checkpoint saved before the step is safe: Orbax copies every array to the
        host before `save` returns, as long as `Checkpoints` names no prioritized keys
        and no concurrent transfer limit. The batch is not donated, because the loader
        owns it.

        The step is planned to fit (`dew.training.planning`): the fewest
        microbatches at the least recomputation, each an attempt of the
        transaction, a window pooling `split` x `accumulation`. Every batch of
        a run is split alike, as its first compile planned or its pooling
        state's window holds; later shapes climb the ladder alone. A plan that
        splits or spans devices, left open and unrecorded, is timed on `trials` (`fit`'s, `Trials`)
        against the batch whole a rung up where they ask for it, and the faster kept. Where JAX's AutoPGLE
        is on (`jax_enable_pgle`), the plan admitted runs its first runs on a trial, `trials`' or, where
        none are given, a copy of the state beside it, refused where the devices hold no room for it
        (`Trials.profiled`), and its program compiles again guided by their profile (`Planned.guided`),
        the executable it replaces released with its program (`Planned.released`).
        """
        if (target := self.lower_for) is not None:
            # Shapes are all a lowering reads, so the state may be abstract (`jax.eval_shape`).
            raise Lowered(target, *routed(lambda: self._lowered_for(state, batch, target)[0]))
        split, unaligned = divmod(int(state.window_size), self.accumulation)
        if unaligned or (state.accumulation is not None and self._split not in (None, split)):
            raise ValueError(f"the state pools windows of {int(state.window_size)} microbatches, and this "
                             f"run updates every {self.accumulation} steps of {self._split} microbatches")
        if state.accumulation is not None:
            self._split = split
        if (state.scale is not None) != self.dynamic_scale:
            raise ValueError("checkpoint dynamic-scaler configuration differs from this trainer")
        if self.host_master:
            return self._compile_host(state, batch)
        mesh = self.device_mesh
        links = self._links(mesh)
        shaped = self._at_base_rung(batch)
        resumed, self._resumed_rung = self._resumed_rung if shaped is None else shaped, None
        with self._traced_on(mesh, links) as schedule:
            planned, flops = self._planned(state, batch, mesh, schedule, resumed, trials)
            split, candidate, self.executable = planned.split, planned.candidate, planned.executable
            self._split = split
            self._shape_rungs[batch_layout(batch)] = self._rung()
            self.step_flops = flops and split * flops
            devices = local_devices(self.device_mesh)
            self.memory = memory_record(self.executable, devices, candidate.held, fits=True)
            self.links = links
        return self._runner(planned, mesh)

    def _staged(self, state: TrainState, step: int) -> TrainState:
        """`state` to run step `step`: its window of `window_size` microbatches dropped where the step, of
        the attempt count a resumed run restores, begins a ramp's stage after the first and a rejection
        left the window part filled, so an update pools one stage's batches. Only the window's fields
        are compiled (`window_dropped`), so a host-master state's resident weights stay where they are."""
        if step not in self._stage_starts or state.accumulation is None:
            return state
        fields = (state.microstep, state.accumulation)
        microstep, window = jax.jit(functools.partial(window_dropped, size=int(state.window_size)),
                                    out_shardings=jax.tree.map(lambda x: x.sharding, fields))(*fields)
        return dataclasses.replace(state, microstep=microstep, accumulation=window)

    def _runner(self, planned: Planned, mesh: Mesh) -> CompiledStep:
        """The step `planned` compiled, which reads a batch in its microbatches."""
        # The program compiled above, not the jit: a call through the jit traces and compiles its own,
        # without the step's compiler options, which was 25 s of an A100's cold start.
        split, candidate, executable = planned.split, planned.candidate, planned.executable
        parts = planned.parts(mesh)
        window = split * self.accumulation

        def run(current, batch):
            with self._traced_on(mesh):
                # Each shape plans its copies (`_at_base_rung`), and a step reads the state with its own.
                if not candidate.copies:
                    current = dataclasses.replace(current, compute=None)
                elif current.compute is None:
                    current = dataclasses.replace(current, compute=jax.jit(
                        lambda variables: narrowed(variables, candidate.copies),
                        out_shardings=candidate.shardings.compute)(current.variables))
                if (window > 1 and self.step is None and (current.accumulation is None or (
                        (self._ramp or self._stage_starts)
                        and not current.accumulation.replays(candidate.prepared.accumulation)))):
                    # The window, and with it the split, is the state's from here on.
                    current = dataclasses.replace(self._initialize_accumulation(
                        current, candidate.micro, candidate.shapes, window=window),
                        window_size=jnp.asarray(window, jnp.int32))
                    # The moving state alone: the frozen tree stays the arrays the run holds.
                    moving, frozen = apart(current)
                    current = together(jax.device_put(moving, apart(candidate.shardings)[0]), frozen)
                outputs = []
                moving, frozen = apart(current)
                for part in parts:
                    moving, *output = executable(moving, frozen, batch, part)
                    outputs.append(tuple(output))
                return (together(moving, frozen), *(outputs[0] if split == 1 else whole_step(outputs)))
        return run

    def _planned(self, state: TrainState, batch: Batch, mesh: Mesh, schedule: Schedule, resumed: JSON,
                 trials: Trials | None) -> tuple[Planned, StepFlops | None]:
        """The plan `compile` reads `batch` in (`FitSearch`), from the run's splits (`splits`), the
        faster on `trials` of a frontier of two, guided under AutoPGLE (`Planned.guided`) and released
        (`Planned.released`), with `plan` its record and the objective at its rung, and a microbatch's
        FLOPs (`plan_flops`). A plan recorded beside the compilation cache (`recorded_rung`) is the first
        candidate, kept untimed; `resumed` is a checkpoint's rung."""
        replicated = NamedSharding(mesh, P())
        placement = batch_shardings(mesh, batch)
        candidates = splits(self._rows, batch, mesh, self.microbatches, self._split)

        def lower(split: int, narrow: bool) -> Candidate:
            rung = {**self._rung(), 'narrow': narrow, 'split': split}
            return self._candidate(state, batch, schedule, placement, replicated, split, narrow,
                                   recorded_launches(recorded, rung))

        # Read before anything traces, so a recorded rung is the first one lowered.
        record_path, key, recorded = recorded_rung(self.objective.program_key(), (state, batch), candidates,
                                                   mesh, self.step)
        before = self._rung()
        self._climb_to(recorded)
        self._resumed_rung = None
        if self._rung() != before:
            _log.warning("taking an earlier run's plan; delete %s to decide again", record_path)
        first = None if recorded is None else record(recorded, 'recorded plan').get('split', 1)
        search = FitSearch(
            lower, lambda candidate: step_options(self.objective, candidate.micro,
                                                  candidate.prepared.variables, mesh),
            self.objective, mesh, self._xla_defaults, self._narrow,
            None if self._base_rung is None else self._base_rung[1], resumed)
        plans, probes = search.plans(
            candidates, candidates.index(first) if isinstance(first, int) and first in candidates else 0,
            # A recorded plan, and a resumed run's, is the plan it continues: neither is timed again.
            frontier=trials is not None and trials.frontier and recorded is None and resumed is None)
        seconds = ([] if trials is None or len(plans) == 1
                   else trials.seconds([self._runner(plan, mesh) for plan in plans]))
        # Every process runs the plan process zero timed clearly faster (`faster`).
        planned = plans[broadcast_from_process_zero(faster(seconds) if seconds else 0)]
        if runs := pgle_runs():
            trials = Trials(state, batch, owned=False, frontier=False) if trials is None else trials
            planned = planned.guided(trials.profiled(self._runner(planned, mesh), planned, mesh, runs), runs,
                                     mesh)
        # The state the chosen plan holds between steps, its accumulator and copies included; the
        # plan's fit check counted them before any is built (`Candidate.held`).
        self.validation = self._validation(planned.candidate.prepared)
        self.objective.substitute(list(planned.modules))
        self.objective.restore_head(planned.tile, fused=planned.fused)
        self._xla_defaults, self._narrow = planned.xla_defaults, planned.narrow
        chosen = {**self._rung(), 'split': planned.split}
        flops, counted = plan_flops(recorded, planned.executable)
        if record_path is not None:
            keep_rung(record_path, key, {**chosen, 'flops': counted, 'launches': planned.candidate.launches})
        self.plan = settled(planned, plans, seconds, probes)
        if planned.split != candidates[0] or seconds:
            _log.info("the step fits the devices in the plan %s", self.plan)
        return planned.released(), flops

    def _candidate(self, state: TrainState, batch: Batch, schedule: Schedule, placement, replicated,
                   split: int, narrow: bool, launches: str | None) -> Candidate:
        """The step reading `batch` in `split` microbatches, lowered at the
        objective's current rung, with what the fit check holds beside it.
        `launches` are its kernel launches where a record holds them, which
        settle their configs before it lowers; otherwise, or where the run
        measures configs (`tune_kernels`), its lowering records them (`tune`)."""
        micro = microbatch_shapes(batch, split)
        shapes, copies, prepared = self._prepared(state, micro, schedule, split * self.accumulation, narrow)
        shardings = self.shardings(prepared)
        prepared = placed_shapes(prepared, shardings)
        # The check reads the state's arguments as already on the devices. The copies and accumulators
        # `run` builds on the first call are not yet, so they count with the prefetched batches beside
        # the step, a ramp stage's window too, which `run` builds beside the last, already resident.
        last = state.accumulation
        rebuilt = last is not None and not last.replays(prepared.accumulation)
        built = [(prepared.compute, state.compute), (prepared.accumulation, None if rebuilt else last)]
        held = prefetched_bytes(batch, placement, self._prefetch) + placed_bytes(
            [field for field, kept in built if kept is None and field is not None])
        jitted = self._jitted(shapes, copies, shardings, placement, replicated, split)
        part = jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated)
        lower = Routed(functools.partial(jitted.lower, *apart(prepared), batch, part))
        if self.tune_kernels:
            lowered, launches = tuning.tune(lower, measure=True)
        else:
            # The tuner is off (bible §6): each launch takes its derived config, nothing settled or read.
            lowered, launches = lower(), launches or "[]"
        return Candidate(micro, shapes, copies, shardings, prepared, held, lowered, launches, lower.report())

    def _prepared(self, state: TrainState, batch: Batch, schedule: Schedule, window: int,
                  narrow: bool | None = None):
        """The loss's shapes (None under a custom step), the narrow copies and the state's shapes
        with the accumulation of a `window` of microbatches and the copies the step builds, as the
        step traces them over microbatches of `batch`'s shapes, none where `narrow` (the trainer's) is off."""
        shapes = None if self.step is not None else self._loss_shape(state, batch, window)
        if shapes is not None:
            self._refuse_unpipelined(schedule)
        prepared = self._initialize_accumulation(state, batch, shapes, window=window, shape_only=True)
        copies = self._narrow_copies(state, batch) if (self._narrow if narrow is None else narrow) else {}
        prepared = dataclasses.replace(prepared, compute=jax.eval_shape(
            lambda variables: narrowed(variables, copies), prepared.variables) if copies else None)
        return shapes, copies, prepared

    def _jitted(self, shapes, copies: dict, shardings: Placement[TrainState], placement,
                replicated: NamedSharding, split: int):
        """The step over one of the `split` microbatches of a batch
        (`microbatch`), which counts the step at its last microbatch. It takes the state apart
        (`apart`): the moving state, which it donates and returns advanced, and the frozen tree, which
        it reads and neither donates nor returns, so the frozen arrays stay the ones the run (and the
        objective's held tree) holds, and no output copies them."""
        # The transaction brings the optimizer state and the EMA the layout keeps on the host to the device
        # once its gradient is ready (`Transaction.past`); a custom step, and every other field, reads them
        # from its start.
        late = () if self.step is not None else tuple(set(COMMIT_FIELDS) & set(self.layout.host))
        body = self._step_body(shapes, copies, split, (lambda state: self._fetched(state, shardings, late))
                               if late else None)
        moving, frozen = apart(shardings)

        def step(current, held, batch, part):
            # The body sees every field on the device; the out shardings
            # return the host-resident ones to pinned host memory.
            current = together(current, held)
            micro = microbatch(batch, part, split)
            early = [field for field in self.layout.host if field not in late]
            advanced, loss, aux, committed = body(self._fetched(current, shardings, early),
                                                  self._model_batch(micro), part, batch_tokens(micro))
            stepped, *rest = outcome(current, advanced, loss, aux, committed, batch, part == split - 1)
            return (apart(stepped)[0], *rest)

        return jax.jit(step, in_shardings=(moving, frozen, placement, replicated),
                       out_shardings=(moving, replicated, replicated, replicated, replicated, replicated),
                       donate_argnums=0)

    def _lowered_for(self, state: TrainState, batch: Batch, generation: str, devices: Mesh | None = None
                     ) -> tuple[jax.stages.Lowered, jax.stages.CompilerOptions | None]:
        """The step `compile` builds first, traced as on a GPU of the CUDA `generation` and lowered for
        one (`dew.training._lowering`) on this process's devices, and, lowered over `devices`, the
        options its compile there takes (`step_options`). It traces on an abstract mesh of that GPU's
        devices, from which JAX's Pallas lowering reads the GPU, or on `devices`, compile-only GPUs
        XLA compiles it for (`dew.training.ahead`), at the base rung and the plan's first split: a
        run that does not fit splits its batch or recomputes more, no other kernel. A CPU-owned
        state's step realizes its parameters on the host as it runs, and is not lowered."""
        if self.host_master:
            raise ValueError("lower_for lowers a step whose state the devices hold; this layout keeps "
                             "the variables on the host")
        mesh = gpu_mesh(self.device_mesh, generation) if devices is None else devices
        self._at_base_rung(batch)
        with assumed(generation), self._traced_on(mesh) as schedule:
            state, batch = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), (state, batch))
            split = splits(self._rows, batch, mesh, self.microbatches, self._split)[0]
            micro = microbatch_shapes(batch, split)
            shapes, copies, prepared = self._prepared(state, micro, schedule, split * self.accumulation)
            shardings = jax.tree.map(lambda s: NamedSharding(mesh, s.spec, memory_kind=s.memory_kind),
                                     self.shardings(prepared))
            placement = batch_shardings(mesh, batch)
            replicated = NamedSharding(mesh, P())
            jitted = self._jitted(shapes, copies, shardings, placement, replicated, split)
            traced = jitted.trace(*apart(placed_shapes(prepared, shardings)), placed_shapes(batch, placement),
                                  jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated))
            if devices is None:
                return traced.lower(lowering_platforms=('cuda',)), None
            return traced.lower(), step_options(self.objective, micro, prepared.variables, devices)

    def _narrow_copies(self, state: TrainState, batch: Batch) -> dict:
        """The parameters the step reads through narrow copies
        (`dew.training.narrow`), by path under `params`: on every GPU but
        one whose row measured them off, for a step this trainer composes,
        without a loss scale."""
        if (measured_kernel('narrow_copies', 'on' if backend() == 'gpu' else 'off') == 'off'
                or self.step is not None or self.dynamic_scale or FROZEN in state.variables):
            return {}
        variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

        def loss(params, rest, batch, microstep, key, step, ema):
            # As `_loss_shape` traces it, the averaged weights the step hands
            # the objective included: a parameter read there is used twice.
            variables = {**rest, "params": params}
            return self.objective._loss(
                variables, self._model_batch(batch),
                Step(microstep, jax.random.fold_in(key, step), with_ema(variables, ema),
                     qk_stats=clips_qk(state.opt_state)))

        rest = {name: tree for name, tree in variables.items() if name != "params"}
        return narrowed_paths(loss, variables["params"], rest, batch, state.microstep, state.key, state.step,
                              state.ema)

    def _refuse_unpipelined(self, schedule: Schedule) -> None:
        """Refuse a stage axis under a model whose traced step ran no pipeline."""
        stages = self.device_mesh.shape.get(STAGE_AXIS, 1)
        if stages > 1 and not schedule.pipelined:
            raise LayoutRefused(
                f"the stage axis of {stages} holds a pipeline's stages of a stack of like layers (a "
                f"decoder's, a DiT's or a ViT's blocks, a UNet level's transformers) that it splits "
                f"evenly, and {type(self.objective).__name__}'s model runs none, so every stage would "
                f"compute the whole step; give those devices to the data or fsdp axis")

    def _compile_host(self, state: TrainState, batch: Batch) -> CompiledStep:
        from dew.distributed.host import transfer
        from dew.training.execution import HostExecution, stack_sites
        cpu = self.state_mesh
        execution = HostExecution(self.objective, self.layout, self.device_mesh, cpu,
                                  stack_sites(self.objective, state.variables), device=self._device)
        cpu_batch = transfer(batch, batch_shardings(cpu, batch))
        with self._traced_on(cpu) as schedule:
            shapes = self._loss_shape(state, cpu_batch, self.accumulation)
            self._refuse_unpipelined(schedule)
            prepared = self._initialize_accumulation(state, cpu_batch, shapes, window=self.accumulation,
                                                     shape_only=True)
            placement = self.shardings(prepared)
            transaction = Transaction(self.objective, self.optimizer, self.accumulation, shapes,
                                      record=self.record)
            body = transaction.step(realize=execution.realize, host=True)

        def run(current, batch):
            with self._traced_on(cpu):
                batch = transfer(batch, batch_shardings(cpu, batch))
                if self.accumulation > 1 and (current.accumulation is None or (
                        (self._ramp or self._stage_starts)
                        and not current.accumulation.replays(prepared.accumulation))):
                    current = self._initialize_accumulation(current, batch, shapes, window=self.accumulation)
                # The device steps run in the accelerator's forward (`HostExecution`).
                advanced, loss, aux, committed = body(current, model_batch(batch), 0, batch_tokens(batch))
                advanced, *stepped = outcome(current, advanced, loss, aux, committed, batch, 1)
                # A leaf already where its placement says stays the array it
                # is: the resident frozen collection crosses no boundary.
                return transfer(advanced, placement), *stepped
        return run

    # ------------------------------------------------------------------ The loop
    # ------------------------------------------------------------------

    @ruled
    def fit(
        self,
        dataset: Dataset | Loader,
        *,
        steps: int | None = None,
        epochs: int | None = None,
        tokens: int | None = None,
        examples: int | None = None,
        log_every: int | str = 100,
        eval_every: int | str | None = None,
        checkpoint_every: int | str | datetime.timedelta | None = None,
        checkpoint_final: bool = True,
        metrics: Sequence[Metric] = (),
        preview: bool = False,
        scored_to: str | os.PathLike[str] | None = None,
        best: str | Metric | TrainingScalar | types.MethodType | Best | Sequence[Best] | None = None,
        stop: Plateau | Pause | None = None,
        validation: Dataset | Mapping[str, Dataset | Reader | EvalSuite] | None = None,
        summaries: Sequence[Summary] = (),
        restore_best: bool = False,
        state: TrainState | None = None,
        observe: Observe | None = None,
        device_prefetch: int = PREFETCH_DEPTH,
        resume: Resume = "auto",
        cooldowns: Cooldowns | None = None,
    ) -> TrainState:
        """Train until the step or real-token cap is reached, resuming the latest checkpoint as `resume` says.

        `resume="auto"` continues the checkpointer's latest checkpoint where it holds one and otherwise
        starts afresh; "require" refuses to start afresh, for a job that must continue, and "never" refuses
        a directory that already holds a run (`Checkpoints.resumed`). An explicit `state` is the run's
        initial state, in place of initializing one and of restoring a checkpoint (a float64 truth's widened
        copy of another run's, say), not a continuation, which is a checkpoint's alone: its input reader
        starts at the position `dataset` supplies.

        Every `log_every` steps, the tracker receives the loss, the objective's metrics and the throughput.
        `tokens` stops at the first microbatch that reaches this real-token budget, from the same counter
        as `train/tokens`, persisted in checkpoints. Padding never counts. With both caps the first reached
        ends the run; with `steps=None`, a decaying schedule needs its own step horizon. `examples` trains
        on the chain's first source examples, each once with all its rows (`Loader.example_steps`).

        Every `eval_every` steps, and at the end, the validation split is scored. The objective's
        artifacts go to the tracker and to `metrics`, whose reductions are logged as `val/<name>`.
        `validation` is that split as a Dataset, or names several, each a Dataset, a reader scored as
        `dataset.val` would be or an `EvalSuite` with its own metrics and cadence, logged as
        `<name>/<metric>`. Each of `summaries` weighs those scores into one, `evaluation/summary/<name>`,
        formed at a step that scored every split it weighs, which `best` and `Plateau` rank by.

        `restore_best=True` returns the model the run ranked best: the final state with the counters,
        variables and EMA of the best step (`MODEL_STATE`), read from its best checkpoint, a weights-only
        one included, or without `Checkpoints` from the copy it held (`HeldBest`). Its optimizer state
        and the rest stay the final step's, so it is the best model, not a point to resume training
        from, which the run's checkpoints are. `cooldowns` branch the run into cooldowns, each scored by
        every suite as `cooldown_<split>/<metric>` and discarded (`dew.training.cooldown.Cooldowns`).

        Every `checkpoint_every` steps (or "5%", `FitPlan.cadences`), the state and data position are written;
        `checkpoint_final=True` also writes the final state, independently of that cadence. A duration is
        checked at each `log_every` step, so they are written at the first log step after that much time, and
        every `checkpoints.local_every` steps to the local directory. A preemption notice (a scheduler's
        SIGTERM; `PreemptionNotice`) stops the run at the next step every process agrees on. That step's state
        and data position are written, the final validation is skipped, and fit raises `Preempted`, which ends
        the program with status 143 unless caught. Run it again and fit resumes from there.

        Previews are generated only when `preview=True` and a tracker receives them; scalar reporting never
        triggers preview work. `scored_to`, a directory, takes every validation batch's scoring artifacts at
        each evaluation, `<scored_to>/step-<step>/<split>-<batch>.npz` (`Evaluation.run`). `observe` takes
        values of the state at a cadence of its own while the run trains (`Observe`). `device_prefetch`
        training batches are placed on the devices ahead of the step, which the step's memory plan holds
        room for. The process is readied first
        (`dew.training.runtime.ready_process`), as `prepare_process` readies it: a program that never called
        that gets Dew's log handler, and one whose JAX opened on the CPU beside an NVIDIA GPU stops.
        """
        self._startup = Startup()
        loader, held, tokens, steps = run_data(
            self.objective, dataset, validation, steps=steps, epochs=epochs, tokens=tokens, examples=examples,
            accumulation=self.accumulation, startup=self._startup)
        self._device, self._prefetch = loader.device, device_prefetch
        self._schedule(None if steps is None else steps // self.accumulation)
        plan = self._plan(loader, held, dataset, steps=steps, tokens=tokens,
                          written=(log_every, eval_every, checkpoint_every),
                          checkpoint_final=checkpoint_final, metrics=metrics, preview=preview,
                          scored_to=scored_to, best=best, stop=stop, summaries=summaries,
                          restore_best=restore_best, observe=observe, cooldowns=cooldowns)
        run = FitRun(time.perf_counter())
        # A display of this fit's own: one before it may have run other steps.
        self._display = TrainingDisplay()
        profile, checkpoints = self.profile, self.checkpoints
        profiler = owned_capture(self.profile)
        # One boundary lookup at setup: the prefetch worker and the step scopes share whichever profiler
        # owns the capture.
        tracer = profiler if profiler is not None else telemetry_profile.active_profile()
        try:
            trained = self._training_loop(plan, run, profiler, tracer, profile, checkpoints, state, resume)
            # The state a checkpoint would hold: the step rebuilds its narrow copies.
            trained = dataclasses.replace(trained, compute=None)
            # Whether this fit trained or found the run complete, and inside
            # the lifecycle, so a selection that fails is the run's outcome.
            if restore_best and run.preempted is None:
                trained = self._best_model(trained, run)
        finally:
            primary = sys.exception()
            error = self._closed(run, primary, profiler)
            if primary is None and error is not None:
                raise error
        if run.preempted is not None:
            raise Preempted(run.preempted)
        return trained

    def _plan(self, loader: Loader, held: Mapping[str, Reader | EvalSuite] | None,
              dataset: Dataset | Loader, *, steps: int | None, tokens: int | None,
              written: tuple[int | str, int | str | None, int | str | datetime.timedelta | None],
              checkpoint_final: bool, metrics: Sequence[Metric], preview: bool,
              scored_to: str | os.PathLike[str] | None,
              best: str | Metric | TrainingScalar | types.MethodType | Best | Sequence[Best] | None,
              stop: Plateau | Pause | None, summaries: Sequence[Summary], restore_best: bool,
              observe: Observe | None, cooldowns: Cooldowns | None) -> FitPlan:
        """What `fit` was asked for over its compiled data: its cadences resolved against the run's
        length (`FitPlan.cadences`), its scores declared, its best and stop policies, and its refusals
        made before anything is placed (`_preflight`)."""
        logged, evaluated, saved, local, suites = FitPlan.cadences(
            loader, held, metrics, steps, tokens, self.accumulation, written, self.checkpoints)
        declared = declarations(suites, metrics, summaries, loss=self.objective.defines_validation_loss)
        selection, plateau, pause = fit_policies(self.objective, best, stop, declared, saved, steps=steps,
                                                 tokens=tokens, accumulation=self.accumulation,
                                                 checkpointed=self.checkpoints is not None)
        self._preflight(loader, plateau, suites, preview=preview, stated=dataset,
                        loss=(self.checkpoints is not None or restore_best) and not selection)
        return FitPlan(
            dataset=loader, steps=steps, log_every=logged, eval_every=evaluated, tokens=tokens,
            checkpoint_every=saved, checkpoint_final=checkpoint_final, local_every=local,
            metrics=metrics, preview=preview and self.tracker is not None, best=selection, stop=plateau,
            suites=suites, pause=pause, scored_to=None if scored_to is None else os.fspath(scored_to),
            held_best=restore_best and self.checkpoints is None, observe=observe, summaries=tuple(summaries),
            declared=declared,
            branches=() if cooldowns is None else cooldowns.branches(steps, tokens, self.accumulation),
        )

    def _schedule(self, updates: int | None) -> None:
        """The configured optimizer, built for `updates` updates; its state holds the rate it logs."""
        if self._config is None:
            return
        built = self._config.build(updates)
        self._optimizer = self.objective.optimizer(built)

    def _training_loop(self, plan: FitPlan, run: FitRun, profiler: Profiler | None,
                       tracer: Profiler | None, profile: ProfileWindow | None,
                       checkpoints: Checkpoints | None,
                       initial: TrainState | None = None, resume: Resume = "auto") -> TrainState:
        """Place the state, dispatch the numerical steps and finish their loop
        before resource cleanup. Returns the final state, the placed one for a
        run already complete, its last checkpoint written, so nothing ran.

        The loop is the state's one holder: each step donates the state it is
        handed, and a reference kept past that, as fit's own once was, holds
        an array whose buffers the step consumed."""
        current, position, placed, abstract = self._starting(run, plan.dataset, initial, resume)
        run.last_checkpoint = time.perf_counter()
        interval = Interval.resumed(run.stop_control, last_saved=(
            current if checkpoints is not None and checkpoints.latest is not None else None))
        state, shardings, complete = self._opened(plan, run, interval, current, position, placed, abstract,
                                                  resumed=initial is None and interval.last_saved is not None)
        if complete:
            return state
        compiled: dict[Signature, tuple[CompiledStep, StepFlops | None]] = run.compiled
        seen = 0
        run.notice = PreemptionNotice()
        run.clock = TokenClock(*interval.clocked(state.accumulation))
        if run.cooling is not None and not plan.ended(run.current, run.clock.read):
            # A branch a preemption cut short runs again before the run reads on from where it stood.
            state, cut = self._cooled(plan, run, interval, state, shardings, position, run.cooling)
            if cut:
                run.preempted = run.current
                return self._wind_down(plan, run, interval, state, shardings, position, profiler)
        while not plan.ended(run.current, run.clock.read):
            # A schedule's stage ends where the commits reach its tokens.
            self._scheduled(plan, run)
            self._open_window(run, profile, profiler, seen, compiled)
            capturing = tracer is not None and tracer.running
            step_scope = (jax.profiler.StepTraceAnnotation("train", step_num=run.current)
                          if capturing else contextlib.nullcontext())
            with step_scope:
                # The first read restores the stream's position and opens its reader on the prefetch
                # worker, and a stream that may run out can on any read: a process whose read failed stops
                # its peers before they compile or step without it.
                batch = waited_read(run, interval, agree=not (seen and run.endless))
                if batch is None:
                    self._display.note(f"The training data ran out at step {run.current}")
                    break
                batch = rolled_out(self.rollout, state, batch, self.device_mesh, self._marked, interval)
                device, _ = split_fields(batch)
                inputs = self.objective.inputs
                if not compiled and self.rollout is None and inputs is not None:
                    # A rollout writes the batch its loss reads, from prompts it reads itself.
                    # The objective reads the batch its device steps make (`_model_batch`).
                    with jax.set_mesh(self.device_mesh):
                        checked = self._model_shapes(device) if self._device else device
                    agreed("training input declaration", functools.partial(inputs.check, checked))
                known, began = len(compiled), time.perf_counter()
                with self._startup.phase("compile", "compiling the step", when=not known):
                    train_step, measured_flops, state = self._compiled_for(compiled, state, device, run)
                if len(compiled) > known and self.executable is not None and not self.host_master:
                    signature = batch_signature(device)
                    run.reads[signature] = step_reads(self.executable, signature)
                interval.compiles = len(compiled)
                if len(compiled) > known:
                    with self._startup.phase("warm", "warming the evaluations", when=not known):
                        self._warm_evaluations(plan, state, shardings)
                if not known:
                    # Rebound once the step is compiled, so the first tick measures steps, not the compile.
                    interval.last_log_time = time.time()
                    interval.waited = 0.0
                elif len(compiled) > known:
                    # A shape met later compiles inside the interval, which
                    # reports that time apart from its steps'.
                    interval.compiling += time.perf_counter() - began
                with region("train.step"):
                    state, loss, aux, finite, accepted, committed = train_step(state, device)
                # Read with this step queued behind the one it describes, so the devices go on meanwhile.
                self._read_taken(run)
                if self._unproven:
                    # A step's failure on the devices surfaces at the next program that reads its
                    # outputs, the loss counters, so a newly compiled step's first run is awaited here,
                    # where its failure can be named.
                    self._unproven = False
                    first_run(loss, run.current, self.memory, self.device_mesh)
                run.loss = loss
                assert run.train is not None
                position = run.train.source_state
                run.current, seen = run.current + 1, seen + 1
                self._display.step(run.current)
                interval.count(batch, measured_flops, loss, finite, committed, aux)
                run.clock.stepped(self.objective.row_tokens(device), rows_of(device))
                # The step's batch is no cooldown branch's to hold beside its own (`_cooled`).
                batch = device = None
                if run.first_step is None:
                    loss.block_until_ready()
                    run.first_step = time.perf_counter() - run.started
                stopped, state = self._between_steps(plan, run, interval, state, shardings, position,
                                                     loss, aux, accepted)
                if stopped:
                    break
            # The step row is complete once its scope exits; closing the
            # window here keeps the last iteration inside the capture.
            if run.tracing and profile is not None:
                run.traced += 1
                if run.traced == profile.steps:
                    run.tracing = False
                    assert profiler is not None
                    self._stop_trace(run, profile, profiler)
        return self._wind_down(plan, run, interval, state, shardings, position, profiler)

    def _closed(self, run: FitRun, primary: BaseException | None,
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
        # A failed run's peers may be gone: its last log is read only where no read waits on them.
        unread = (None if run.taken is None or not run.taken.ready()
                  else functools.partial(self._read_taken, run, agree=False))
        cleanups = (("Training log", unread), ("Training iterator", close), ("Profiler", stop_trace),
                    ("Checkpoint wait", None if checkpoints is None else checkpoints.wait),
                    ("Display", self._display.close))
        try:
            attempted([(f"{label} cleanup", cleanup) for label, cleanup in cleanups if cleanup is not None],
                      primary=error)
        except BaseException as failure:
            # The first cleanup to fail, where the run itself did not, is the run's outcome.
            error = failure
        # The traceback of a failed run holds fit's frame and this one, and what these names point at.
        run.source, run.train, run.taken = None, None, None
        close = stop_trace = cleanups = unread = None
        run.other += time.perf_counter() - paused
        return self._reported_outcome(error, run)

    def _starting(self, run: FitRun, dataset: Loader, initial: TrainState | None, resume: Resume
                  ) -> tuple[int, bytes | None, Callable[[], tuple[TrainState, Placement[TrainState]]],
                             TrainState | None]:
        """Where `fit` starts, before it places anything: the step, the data position a resume reads
        alone (`Checkpoints.position`), so the data opens where the run left it, the placement still
        to make, and the state's shapes it checks first. A supplied `initial` state is placed as it
        is given, its shapes already placed: no continuation, which `resume="require"` refuses, and
        under "never" into a directory that holds no run either. An objective that names its input field
        alone declares it first (`_declare_input`)."""
        checkpoints = self.checkpoints
        if self.objective.named_inputs is not None and self.objective.inputs is None:
            self._declare_input(self.objective.named_inputs, dataset)
        if resume == "require" and (checkpoints is None or initial is not None):
            raise ValueError("resume='require' continues the checkpointer's latest checkpoint, which a "
                             "trainer with no checkpointer, or a run given its state=, starts without")
        if initial is not None and checkpoints is not None and resume == "never":
            agreed_same("newest checkpoint", lambda: checkpoints.resumed(resume))
        if initial is not None:
            # A supplied state's trees are held as dicts too (`initial_state`), its optimizer's and
            # accumulation's mirrors of the parameters with them.
            initial = owned_dicts(dataclasses.replace(initial, variables=dict(initial.variables)))
            shardings, _ = self._validated(initial)
            supplied = jax.device_put(initial, shardings)
            return int(supplied.step), None, lambda: (supplied, shardings), None
        start = self._started(resume)
        if checkpoints is not None and checkpoints.latest is not None:
            run.stop_control = checkpoints.control(checkpoints.latest)
        if checkpoints is None or start.resume is None:
            return 0, None, lambda: self._placed(start, positioned=False)[:2], start.abstract
        return (start.resume, checkpoints.position(start.resume, DataPartition.of(self.device_mesh)),
                lambda: self._placed(start, positioned=False)[:2], start.abstract)

    def _declare_input(self, named: str, dataset: Loader) -> None:
        """Declare the objective's input `named` (`Objective.named_inputs`) at the per-example shape its
        model reads (`Field.of`): of `dataset`'s first batch, placed as the run places it and through the
        loader's device steps (`_model_shapes`), which may crop or resize it, agreed across the pool
        before anything initializes."""
        mesh = self.device_mesh
        first = agreed("training input batch", lambda: first_batch(dataset, mesh))
        with jax.set_mesh(mesh):
            device, _ = split_fields(placed_abstract(mesh, first))
            shape = Field.of(named, self._model_shapes(device)).shape
        held = agreed_same("training input shape", lambda: list(shape))
        self.objective.inputs = InputSpec(Field(named, tuple(held)))

    def _opened(self, plan: FitPlan, run: FitRun, interval: Interval, current: int, position,
                placed: Callable[[], tuple[TrainState, Placement[TrainState]]],
                abstract: TrainState | None, *, resumed: bool,
                ) -> tuple[TrainState, Placement[TrainState], bool]:
        """Start `fit` at step `current`: refuse a run already past its steps, a checkpoint cadence
        with no checkpointer or a resume its stream cannot continue; open the training stream on its
        prefetch worker at `position` and read its first batch, against which the whole state a
        fresh or resumed run holds between steps (`abstract`'s, its accumulation included) is checked
        before anything is placed (`_whole_validated`); then place the state (`placed`) and report
        it. Returns the placed state, its shardings and whether the run is already complete, a
        checkpoint at its last step having been written, so there is nothing to do."""
        mesh, checkpoints, steps = self.device_mesh, self.checkpoints, plan.steps
        if steps is not None and current > steps:
            raise ValueError(f"the run is at step {current}, past the {steps} asked for")
        if plan.checkpoint_every and checkpoints is None:
            raise ValueError(
                "checkpoint_every asks for checkpoints and this trainer has no "
                "checkpointer; pass Checkpoints(directory) to write any")
        # A cooldown a preemption cut short, which a resume from its step runs first, as every process
        # agrees it, its last step's too (`_cooled`).
        kept = None if checkpoints is None else agreed(
            "cooldown record", lambda: checkpoints.cooldown(current) if jax.process_index() == 0 else None)
        run.cooling = broadcast_from_process_zero(kept)
        pending = run.cooling is not None
        complete = (plan.ended(current, interval.real_tokens) and checkpoints is not None
                    and checkpoints.latest is not None and not pending)
        # The step the run reads from, as the pre-read's empty-stream check reads it, and the FLOP
        # count's refusal cleared before the prefetch worker counts its first batch.
        run.current = current
        self._refusal = None
        if not complete and (pending or not plan.ended(current, interval.real_tokens)):
            # A ramp's stages begin where updates do (`Ramp.stages`).
            partition = DataPartition.of(mesh, window=self.accumulation)
            run.source = plan.dataset.train(partition)
            check_stream(run.source, mesh, self.microbatches, plan.dataset.batch, rows=self.rollout is None,
                         resumed=current if resumed and position is None else None)
            self._ramp = run.source.stages if isinstance(run.source, RampedStream) else ()
            # The steps a stage after the first begins at, each where an update does (`Ramp.stages`): a
            # ramp's here, a schedule's as each begins (`_scheduled`); the step lays its window out there.
            self._stage_starts = frozenset(accumulate((after.records - stage.records) // stage.batch
                                                      for stage, after in pairwise(self._ramp)))
            if self._ramp:
                # Each stage's batch is read whole: no plan splits a ramp's.
                self._split = 1
            self._rows = (plan.dataset.batch,)
            run.partition = partition
            run.train = self._prefetched(plan, run, run.source, position)
            run.source = None  # Lifetime transferred to the prefetch worker.
            if abstract is not None and self.rollout is None:
                # Read on the host before the state is placed, and agreed: a process whose read failed
                # stops its peers before any of them places one. The prefetcher places it at the first step.
                first = waited_read(run, interval, agree=True, peek=True)
                if first is not None:
                    placed_first = placed_abstract(mesh, first)
                    # A process whose reader changed a field or a shape is refused by every process, as
                    # a compile refuses it, before the state it would check is traced from that batch.
                    agreed_same("training batch layout", lambda: batch_layout(split_fields(placed_first)[0]))
                    self._fitting(abstract, placed_first)
        state, shardings = placed()
        if int(state.step) != current:
            raise ValueError(f"the checkpoint of step {current} holds a state at step {int(state.step)}")
        check_branches(bool(plan.branches) or pending, plan.suites, settable=settable_rates(state.opt_state))
        started = FitStarted(current, steps,
            checkpoints.source(current) if checkpoints is not None and position is not None else None,
            sum(leaf.size for leaf in jax.tree.leaves(state.variables["params"])), mesh.devices.size,
            jax.devices()[0].device_kind, jax.process_count(), dict(mesh.shape),
            split_share(state.variables),
            seed=self.seed, tokens=plan.tokens, cadences=plan.recorded(self._config, self.accumulation))
        self._report(started, current)
        if plan.dataset.held_out:
            self._display.note(f"validation: {plan.dataset.held_out} records held out of train")
        if plan.observe:
            plan.observe.opened(current, state, complete=complete)
        if complete:
            return state, shardings, True

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
            averaged=self.objective.ema is not None and not self.objective.ema_is_reference))
        return state, shardings, False

    def _whole_validated(self, state: TrainState, batch: Batch, split: int, capacity: int | None
                         ) -> Validation:
        """`Layout.validate` of the whole state the run's steps keep between them reading `batch`'s
        shapes (a marked batch, as the stream gives it) in `split` microbatches: `state` with the
        accumulation its window pools. The narrow copies are left out: the plan's first rung drops
        them where the step does not fit with them (`dew.training.planning.FitSearch`)."""
        device, _ = split_fields(batch)
        with self._traced_on(self.device_mesh) as schedule:
            _, _, whole = self._prepared(state, microbatch_shapes(device, split), schedule,
                                         split * self.accumulation, narrow=False)
        return self._validated(whole, capacity)[1]

    def _fitting(self, state: TrainState, batch: Batch, capacity: int | None = None) -> Validation:
        """The report of the whole state the run's steps keep reading `batch` (`_whole_validated`) in
        the first of the splits its plan may take (`splits`) whose devices hold it, refused, before
        anything is placed, where none does: the plan picks among them once the state is placed."""
        refusals = []
        for split in splits(self._rows, batch, self.device_mesh, self.microbatches, self._split):
            try:
                return self._whole_validated(state, batch, split, capacity)
            except LayoutRefused as refused:
                refusals.append(refused)
        raise refusals[-1]

    def _prefetched(self, plan: FitPlan, run: FitRun, source: Iterator[Batch],
                    position: bytes | None) -> DevicePrefetchIterator:
        """`source`'s batches placed ahead of the step by the prefetch worker, which restores it to
        `position` first, once a pool's processes agree their streams read the same records."""
        assert run.partition is not None
        if position is not None and isinstance(source, PhasedStream):
            # A schedule's stage opens where the position stands: the pool agrees on the one it reads.
            source.set_state(position)
        if jax.process_count() > 1:
            # Each process opened its own stream over its own copy of the
            # corpus, and none reads a record before all agree it is one.
            agreed_same("training data", lambda: (
                f"{plan.dataset.batch} records a step of {stream_identity(source)}"))
        run.endless = stream_endless(source)
        self._marked = marked(run.partition, self._counted_batch)
        prepare = self._marked if self.rollout is None else None
        return DevicePrefetchIterator(source, self.device_mesh, self._prefetch, source_state=position,
                                      reads=run.reads, prepare=prepare)

    def _scheduled(self, plan: FitPlan, run: FitRun) -> None:
        """Open the next stage of a schedule where the run's commits reached the one it stands in
        (`stage_reached`), the prefetch worker reading it from the step that ended the last."""
        reached = stage_reached(plan, run)
        if reached is None:
            return
        source, state, stage = reached
        self._display.note(f"Stage {stage} of the schedule begins at step {run.current}, after "
                           f"{run.clock.read} committed tokens")
        # An update pools one stage's batches: a window a rejection left part filled at the commit the
        # stage begins after is dropped as a ramp's stage drops it (`_staged`), and laid out for the
        # stage's batches, which may be shaped as no earlier stage's were (`_runner`).
        self._stage_starts = self._stage_starts | {run.current}
        run.train = self._prefetched(plan, run, source, state)
        run.source = None  # Lifetime transferred to the prefetch worker.

    def _between_steps(self, plan: FitPlan, run: FitRun, interval: Interval, state: TrainState,
                       shardings: Placement[TrainState], position, loss: jax.Array,
                       aux: dict[str, jax.Array], accepted: jax.Array) -> tuple[bool, TrainState]:
        """The cadenced work after one step of `fit`: the log, the validation,
        the checkpoints, and the preemption notice. Returns whether the
        notice stops the run at this step, whose checkpoint is then written."""
        current, checkpoints = run.current, self.checkpoints
        before = run.clock.read
        real = plan.committed(interval, run.clock, state)
        record, aux = taken_record(aux)
        if record is not None:
            self.records.append(record)
        aux = {name: value for name, value in aux.items() if name not in (TRAINING_FLOPS, TOKENS)}
        run.training = {'train/loss': loss, **{f'train/{name}': value for name, value in aux.items()}}
        run.stepped = (loss, aux, accepted)
        logged = plan.falls(plan.log_every, current, real, before)
        if logged:
            self._take(plan, run, interval, state)

        if isinstance(plan.checkpoint_every, datetime.timedelta):
            # The clock is read at the log interval, where the step is already waited on, so a pool's
            # processes agree on it there rather than gathering behind every step.
            run.checkpoint_due = False
            if logged:
                elapsed = time.perf_counter() - run.last_checkpoint
                due = elapsed >= plan.checkpoint_every.total_seconds()
                run.checkpoint_due = agree_process_phase(None, phase="checkpoint clock", available=due) > 0
        else:
            run.checkpoint_due = plan.falls(plan.checkpoint_every, current, real, before)
        run.local_due = plan.falls(plan.local_every, current, real, before)
        due = plan.due(current, real, before)
        evaluation_due = bool(due) or plan.falls(plan.eval_every, current, real, before)
        # A run's last step, its budget's or the step after which its stream runs out, is scored and saved
        # once, by `_wind_down`, with every suite: never here before a rarer suite has. Only a step that
        # evaluates or pauses could save first (a ranked save scores every suite), so only there is the
        # stream read ahead (`FitRun.ends_here`): no other step's save waits on a batch a stream has yet to
        # make.
        ending = plan.finished(current, real) or (
            (evaluation_due or plan.paused(current, real)) and run.ends_here(interval))
        paused = not ending and plan.paused(current, real)
        ranked = bool((plan.validation or plan.best or plan.stop) and run.checkpoint_due)
        if not ending and (evaluation_due or ranked):
            self._read_taken(run)
            with region("evaluate"):
                # A checkpoint that ranks the steps reads every suite's score.
                names = None if ranked else due
                run.evaluation = self._evaluation(plan, state, shardings, run.training, names)
                run.scored = frozenset(plan.suites if names is None else names)
                run.other += run.evaluation.elapsed_seconds
        scores = (
            {} if run.evaluation is None or run.evaluation.step != current else dict(run.evaluation.scores)
        )
        # Before the step's save, so a preemption inside the branch saves this step to run it again; at
        # the run's last step, after its final save (`_wind_down`). The points share one length and
        # decay, so those due together are one branch.
        due = next((branch for branch in plan.branches if branch.due(current, real, before)), None)
        branch = None if due is None else due.pending(current, self.accumulation)
        if ending and branch is not None and checkpoints is not None:
            # Kept before the step's final save makes the run look complete (`Checkpoints.cooldown`).
            checkpoints.cooling(branch)
        if ending:
            run.cooling, branch = branch, None
        state, cut = (state, False) if branch is None else self._cooled(plan, run, interval, state, shardings,
                                                                         position, branch)
        # Asked at every step on every process, as JAX's agreement needs, before the step's save, so a
        # stop's save is its one save, whole; the last step ends the run. A pause falls on the same step
        # on every process, so where it is reached none asks.
        assert run.notice is not None
        stopping = paused or cut or (not ending and run.notice.reached())
        if not ending:
            self._saves(plan, run, interval, state, position, scores, evaluation_due=evaluation_due,
                        full=stopping)
        if plan.observe:
            plan.observe.at(current, state)
        if run.stopped:
            return True, state
        if stopping:
            self._take(plan, run, interval, state, unlogged=True)
            if checkpoints is not None and interval.resumable != current:
                self._read_taken(run)
                run.other += self._saved_checkpoint(
                    checkpoints, current, state, position, interval,
                    training_best=not plan.validation and not plan.best, control=run.stop_control)
            if paused:
                run.paused = current
            else:
                run.preempted = current
            return True, state
        return False, state

    def _cooled(self, plan: FitPlan, run: FitRun, interval: Interval, state: TrainState,
                shardings: Placement[TrainState], position: bytes | None,
                pending: Pending) -> tuple[TrainState, bool]:
        """Train the cooldown branch `pending` from `state`, on the run's batches from its data `position`,
        by the run's own compiled steps at the rates its state holds times the branch's decay
        (`scaled_rates`); score every suite on it as `cooldown_<split>` at the run's step, and discard
        it. The run's moving state waits on the host and its stream is closed meanwhile, so the devices
        hold one state and one prefetcher, as the run's plan sized them (`Trials`). Returns the run's
        state, and whether a preemption cut the branch short: then it stays pending beside the
        checkpoints (`Checkpoints.cooldown`), and a resume from this step runs it first."""
        (steps, shape), checkpoints = resolved(pending, self.accumulation), self.checkpoints
        if position is None:
            raise ValueError("a cooldown branch reads on from the run's data position, and this stream "
                             "keeps none")
        assert run.partition is not None and run.notice is not None
        began, at, held = time.perf_counter(), state.step, held_rates(state.opt_state)
        self._display.note(f"Cooldown branch from step {run.current}: {steps} steps")
        if checkpoints is not None:
            checkpoints.cooling(pending)
        moving, frozen = apart(state)
        restore, step_at = snapshot(moving), jax.device_put(np.asarray(int(at), at.dtype), at.sharding)
        reading, run.ahead = run.train is not None, None
        if run.train is not None:
            run.train.close()
        branched = FitRun(run.started, current=run.current, partition=run.partition, reads=run.reads,
                          compiled=run.compiled)
        # The branch's own clock of committed tokens, from the run's, which a schedule's stages end at
        # as the run's do (`_scheduled`), read off the devices where its steps may have reached what it
        # reads next (`FitPlan.committed`); the stages it begins are its own, the run's kept as they were.
        counted = Interval(time.time(), None, tokens_before_log=interval.real_tokens)
        branched.clock = TokenClock(*counted.clocked(state.accumulation))
        starts = self._stage_starts
        cooled, done, cut = state, 0, False
        # A rollout run's branch draws its batches from the branch's own states (`rollout._branched`),
        # named by the step it branches at, which a resume that runs it again branches at too.
        with _branched(self.rollout, f"cooldown-{run.current}"):
            branched.train = self._prefetched(plan, branched, plan.dataset.train(run.partition), position)
            try:
                ends = plan.dataset.token_ends
                while done < steps:
                    # Only a schedule's stages and its end read the branch's committed tokens (`_scheduled`).
                    if ends is not None:
                        plan.committed(counted, branched.clock, cooled)
                    if schedule_ended(ends, lambda: branched.clock.read) or (cut := run.notice.reached()):
                        break
                    self._scheduled(plan, branched)
                    assert branched.train is not None
                    batch = agreed_next("cooldown read", branched.train)
                    if batch is None:
                        break
                    batch = rolled_out(self.rollout, cooled, batch, self.device_mesh, self._marked, counted)
                    device, _ = split_fields(batch)
                    step, _, cooled = self._compiled_for(run.compiled, cooled, device, branched,
                                                         reported=run.current)
                    factor = decay(shape, done // self.accumulation, steps // self.accumulation)
                    scaled = scaled_rates(cooled.opt_state, held, factor)
                    cooled, loss, aux, finite, _, committed = step(
                        dataclasses.replace(cooled, opt_state=scaled), device)
                    counted.count(batch, None, loss, finite, committed, aux)
                    branched.clock.stepped(self.objective.row_tokens(device), rows_of(device))
                    done, branched.current = done + 1, branched.current + 1
            finally:
                self._stage_starts = starts
                if branched.train is not None:
                    branched.train.close()
        run.cooling = pending if cut else None
        if cut:
            self._display.note(f"Preempted inside the cooldown from step {run.current}, which a resume runs")
        elif done < steps:
            self._display.note(f"The training data ran out {done} steps into the cooldown from step "
                               f"{run.current}, which records no score")
        else:
            suites = {f"cooldown_{name}": suite for name, suite in plan.suites.items()}
            self._evaluation(dataclasses.replace(plan, suites=suites, summaries=()),
                             dataclasses.replace(cooled, step=step_at), shardings, {})
        if checkpoints is not None and not cut:
            checkpoints.cooling(None)
        for leaf in (leaf for leaf in jax.tree.leaves(apart(cooled)[0]) if not leaf.is_deleted()):
            leaf.delete()
        restored = together(restore(), frozen)
        if reading:
            run.train = self._prefetched(plan, run, plan.dataset.train(run.partition), position)
        run.other += time.perf_counter() - began
        return restored, cut

    def _best_model(self, final: TrainState, run: FitRun) -> TrainState:
        """`final` with the model state of the step the run ranked best (`MODEL_STATE`): read from
        its best checkpoint, a weights-only one included, or without `Checkpoints` from the copy the
        run held (`HeldBest`)."""
        checkpoints = self.checkpoints
        if checkpoints is None:
            return run.best.state(final)
        checkpoints.wait()  # the best may be the last step's save, still in flight
        template = {name: jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), value)
            for name in MODEL_STATE if (value := getattr(final, name)) is not None}
        model, _ = checkpoints.restore(template, step="best")
        return dataclasses.replace(final, **{name: model[name] for name in template})

    def _take(self, plan: FitPlan, run: FitRun, interval: Interval, state: TrainState, *,
              unlogged: bool = False) -> None:
        """Take the log of the interval ending at the current step (`Interval.take`); with
        `unlogged`, only where it holds steps no log took, as the step a run ends on between its
        logging steps does."""
        if run.stepped is None or (unlogged and not interval.since_log):
            return
        if unlogged:
            # The latest logging step's log, which the loop reads a step late, goes first.
            self._read_taken(run)
        loss, aux, accepted = run.stepped
        run.taken = interval.take(run.current, loss, aux, accepted,
                                  held(learning_rate(state.opt_state),
                                       None if state.scale is None else state.scale.scale),
                                  {} if self.rollout is None else dict(_rollout_metrics(self.rollout)),
                                  plan.dataset.statistics())

    def _saves(self, plan: FitPlan, run: FitRun, interval: Interval, state: TrainState, position,
               scores: Mapping[str, float], *, evaluation_due: bool, final: bool = False,
               full: bool = False) -> None:
        """Resolve cadence, ranking, stopping and unconditional final saves in one place; with `full`,
        the step's save holds the whole state and the data position, as a pause's must."""
        checkpoints, current = self.checkpoints, run.current
        ranking = rankings(plan.best, scores, validation=next(iter(plan.suites)) if plan.validation else None,
                           declared=plan.declared, objective=self.objective)
        if plan.held_best and evaluation_due:
            run.best.offer(ranking, dataclasses.replace(state, compute=None))
        winners = checkpoints._candidates(ranking) if (
            evaluation_due and checkpoints is not None and (plan.checkpoint_every or plan.best)) else ()
        if not run.stopped and scores and plan.plateaued(run, scores):
            self._take(plan, run, interval, state, unlogged=True)
            run.stopped = True
            run.stop_control['stop_reason'] = 'validation plateau'
            self._display.note("Stopped: validation plateau")
            if not final:
                # The step the plateau stops at is the run's last: `_wind_down` scores every suite there
                # and saves it (`final`), as it does a budget's last step.
                return
        candidate = bool(winners) or (run.stopped and checkpoints is not None)
        if (checkpoints is not None and interval.last_saved != current
                and (run.checkpoint_due or candidate or (final and plan.checkpoint_final) or full)):
            self._read_taken(run)
            # A save of weights alone is no step to resume from, so not the run's last (`FitRun.ends_here`).
            run.other += self._saved_checkpoint(
                checkpoints, current, state, position, interval, scores=scores, ranking=ranking,
                training_best=not plan.validation and not plan.best, control=run.stop_control,
                weights_only=not run.checkpoint_due and not run.stopped and not full
                and not (final and plan.checkpoint_final) and bool(winners)
                and all(rank.weights_only for rank in winners) and not run.ends_here(interval))
            run.last_checkpoint = time.perf_counter()
        if run.local_due and checkpoints is not None and run.last_local_saved != current:
            self._read_taken(run)
            run.other += self._saved_local_checkpoint(
                checkpoints, current, state, position, interval, control=run.stop_control)
            run.last_local_saved = current

    def _wind_down(self, plan: FitPlan, run: FitRun, interval: Interval, state: TrainState,
                   shardings: Placement[TrainState], position, profiler: Profiler | None) -> TrainState:
        """`fit` after its last step: the stream closed, a preemption
        announced, the trace window stopped, the final validation of every
        suite the step has not scored (skipped on a preemption or a pause,
        which a resume continues) and the checkpoint of the state the run ends
        on, ranked by them: a budget's last step, a stream's that ran out and a
        plateau's (`_between_steps`)."""
        profile, checkpoints, current, loss = self.profile, self.checkpoints, run.current, run.loss
        self._take(plan, run, interval, state, unlogged=True)
        self._read_taken(run)
        if run.notice is not None and run.cooling is None:
            # A last step's cooldown asks the notice between its steps; `_closed` closes it then.
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
        if run.paused is not None and checkpoints is not None:
            self._display.note(f"Paused at step {run.paused}: its checkpoint and data position are in "
                               f"{checkpoints.directory}, where the run resumed without the pause goes on")
        if run.tracing and profile is not None:
            run.tracing = False
            # The window outlived the run, and a trace left running takes the next one down with it.
            assert profiler is not None
            self._stop_trace(run, profile, profiler)
        interval.check_finite(current, self._display)
        if loss is not None:
            # The last step has to land before the wall time is read.
            loss.block_until_ready()
        scores: Mapping[str, float] = {}
        # A plateau or a stream that ran out ends a run whose last step may have evaluated some suites
        # already: only the others are scored now, joined to the scores that step reported.
        joined = run.evaluation if run.evaluation is not None and run.evaluation.step == current else None
        scored = run.scored if joined is not None else frozenset()
        if (
            (plan.validation or plan.eval_every or plan.best or plan.stop or plan.held_best)
            and run.preempted is None
            and run.paused is None
        ):
            missing = [name for name in plan.suites if name not in scored]
            if not plan.suites or missing:
                run.evaluation = self._evaluation(plan, state, shardings, run.training, missing,
                                                  joined=None if joined is None else joined.scores)
                run.other += run.evaluation.elapsed_seconds
            assert run.evaluation is not None
            scores = run.evaluation.scores
        # A token update may cross the pause and the budget at once, which still saves the whole state.
        self._saves(plan, run, interval, state, position, scores, evaluation_due=bool(scores), final=True,
                    full=plan.paused(current, run.clock.read))
        if run.cooling is not None and run.preempted is None and run.paused is None:
            # The last step's branch, its main model scored and saved first, which a resume from the
            # step runs again where a preemption cuts it short.
            state, cut = self._cooled(plan, run, interval, state, shardings, position, run.cooling)
            if cut and checkpoints is not None and interval.resumable != current:
                # Whatever the final-save policy, the step the branch resumes from is written whole.
                run.other += self._saved_checkpoint(checkpoints, current, state, position, interval,
                                                    training_best=False, control=run.stop_control)
            run.preempted = current if cut else None
        if plan.observe:
            plan.observe.last(current, state)
        if checkpoints is not None:
            checkpoints.wait()
        return state

    def _warm_evaluations(self, plan: FitPlan, state: TrainState, shardings: Placement[TrainState]) -> None:
        """Compile the programs the run's evaluations run beside a step's compile, where any runs
        between steps (`FitPlan.evaluates_midway`): every suite warms (`Evaluation.run`'s `warm`),
        its preview included, and nothing is reported.

        An evaluation's programs compile at their first call, which would otherwise be an
        evaluation's inside the run, its seconds holding the compile (7.5 s at the first of 45
        records), counted against the run's goodput. A new training shape can change the modules
        the programs run (the fit ladder's rung), so each step compiled warms them again. The
        training values a ranking reads ride on the first evaluation that reads them, whose gather
        of them compiles there."""
        if plan.evaluates_midway:
            with region("evaluate"):
                self._evaluation(plan, state, shardings, {}, warm=True)

    def _evaluation(self, plan: FitPlan, state: TrainState, shardings: Placement[TrainState],
                    training: Mapping[str, jax.Array], names: Sequence[str] | None = None, *,
                    warm: bool = False, joined: Mapping[str, float] | None = None) -> Evaluation:
        """Score the suites in `names`, every suite when None, as one evaluation, joined to the scores
        the step's earlier evaluation of the others recorded (`joined`, `_wind_down`).

        With no suite at all, the evaluation carries the training values a
        ranking or a stopping rule reads. `warm` warms every suite instead and
        reports nothing (`_warm_evaluations`)."""
        ranks = self.checkpoints is not None or plan.best or plan.stop or plan.held_best
        reports = training if ranks and not warm else None
        chosen = list(plan.suites if names is None else names)
        if not chosen:
            return self._evaluate(state, shardings, None, (), plan.preview, split="val", training=reports,
                                  warm=warm)
        evaluations = []
        for split in chosen:
            suite = plan.suites[split]
            # The loss is recorded wherever the objective defines one, and is scored where it ranks
            # the run's checkpoints or the state it holds, which preflight checks it can.
            ranks = (self.checkpoints is not None or plan.held_best) and not plan.best
            loss = ((ranks or self.objective.defines_validation_loss)
                    and not any(metric.name == 'loss' for metric in suite.metrics))
            evaluations.append(self._evaluate(state, shardings, suite.data, suite.metrics, plan.preview,
                                              loss=loss, split=split, training=reports, warm=warm,
                                              scored_to=plan.scored_to))
        scores = {**(joined or {}),
                  **{name: value for report in evaluations for name, value in report.scores.items()}}
        # Each summary is formed once a step, here, by the evaluation that completes the splits it weighs.
        formed = {} if warm else {name: value for summary in plan.summaries
                                  for name, value in summary.formed(scores).items() if name not in scores}
        if formed:
            self._report_evaluation(dataclasses.replace(evaluations[0], split=Summary.SPLIT, scores=formed,
                                                        previews=(), coordinated_batches=0))
        return dataclasses.replace(evaluations[0], scores={**scores, **formed},
                                   elapsed_seconds=sum(report.elapsed_seconds for report in evaluations))

    def _preflight(self, dataset: Loader, stop, suites: Mapping[str, EvalSuite], *,
                   preview: bool, stated: Dataset | Loader, loss: bool) -> None:
        """The refusals `fit` makes before it places anything: validation
        nothing reads, and a pipeline's batch, whose placement could take
        minutes or run out of memory before the error. Every other mesh's
        batch is checked with its stream (`check_stream`), a ramp's at each
        of its stages. The checkpoints learn why no record rebuilds a run of
        this trainer on the data it was `stated` (`Checkpoints.python_run`)."""
        if self.checkpoints is not None:
            given = {"a custom step (step=)": self.step, "a rollout (rollout=)": self.rollout,
                     "a tracker (tracker=)": self.tracker, "recording each step's draws (record=True)":
                     self.record or None}
            self.checkpoints.python_run(unrecorded_parts(
                type(self.objective).__name__, stated, optimized=self._config is not None,
                seeded=self.seed is not None, mesh=self.mesh, given=given))
        for name, suite in suites.items():
            if loss and not any(metric.name == 'loss' for metric in suite.metrics):
                self.objective._check_validation_loss()
            if (stop is None or not isinstance(stop.metric, TrainingScalar)) and suite.every:
                self._check_validation_is_read(name, suite, preview=preview, loss=loss)
        if self.device_mesh.shape.get(STAGE_AXIS, 1) > 1 and self.rollout is None:
            check_batch(dataset.batch, self.device_mesh, self.microbatches)

    def _check_validation_is_read(self, name: str, suite: EvalSuite, *, preview: bool, loss: bool) -> None:
        """Refuse a validation cadence whose batches nothing would read.

        A validation pass records the objective's validation loss where it
        defines one, and hands its batches to metrics, to the preview and,
        under a checkpointer, or with `loss` where its loss ranks the state a
        run holds (`FitPlan.held_best`), to the loss that ranks the run's
        steps. Only an objective with no validation loss (Flow-GRPO's) and
        none of the rest would open nothing and report nothing, and that
        contradiction is refused before the run does any work.
        """
        read = self.objective.defines_validation_loss or self.checkpoints is not None or loss
        if not suite.metrics and not read and not (preview and self.tracker is not None):
            raise ValueError(
                f"{name} is scored every {suite.every} steps and nothing consumes it: "
                + ("preview=True needs a tracker to receive the samples; "
                   if preview else "")
                + "pass metrics to score it, preview=True with a tracker to sample from it, "
                "or leave eval_every unset")

    def _counted_batch(self, batch: Batch) -> Batch:
        """Prepare host counts before placement, never fetch a device batch to count it.

        The count is the capability: from the first batch whose work the
        model or objective cannot declare (`MissingFlopsCount`, as a media
        row's tower), this process counts no more, says why in the debug
        log, and trains on. Its batches keep their work
        columns, as unknown work (`uncounted`), so every process runs one
        step program and the step's global sum leaves train/mfu unreported
        pool-wide. One thread prepares batches, the prefetch worker or a
        rollout's caller, and only it writes the reason. A count asked of the
        objective directly stays strict.
        """
        if self._refusal is None:
            try:
                return prepare_counts(self.objective, batch, count_model=True)
            except MissingFlopsCount as missing:
                self._refusal = str(missing)
                _log.debug("train/mfu is not reported for %s: %s", type(self.objective).__name__, missing)
        counted = prepare_counts(self.objective, batch, count_model=False)
        return counted if self._refusal is None else uncounted(counted)

    def _compiled_for(self, compiled: dict[Signature, tuple[CompiledStep, StepFlops | None]],
                      state: TrainState, batch: Batch, run: FitRun,
                      reported: int | None = None) -> tuple[CompiledStep, StepFlops | None, TrainState]:
        """Return the step compiled for this batch's abstract signature (`batch_signature`),
        compiling on first sight, and the state to step, a fresh copy where the plan timed steps
        on it (`Trials`). A compile reads `run`'s pending log first: it holds the devices idle
        anyway, and its seconds are the next interval's.

        A ramped run reads a new shape at every stage, so `compiled` keeps
        one step per stage with the FLOPs measured for it. A batch whose
        dtype, fields or a node's static field change at the same shapes is
        another program, compiled as one: an executable refuses it. A step
        that begins a ramp's or a schedule's stage gets its state with the
        window a rejection left part filled dropped (`_staged`), the state
        a timed trial steps on too. A compile reports at `reported`, a
        cooldown branch's at the run's step, else at the state's.
        """
        state = self._staged(state, run.current)
        # Dew's tests keep a frontier's first plan (`UNTIMED_FRONTIER`), a test setting.
        trials = Trials(state, batch, frontier=not untimed_frontier())
        signature = batch_signature(batch)
        if signature not in compiled:
            self._read_taken(run)
            # A process whose reader changed a field or a shape would compile
            # a program whose collectives its peers' never meet.
            agreed_same("training batch layout", lambda: batch_layout(batch))
            began = time.perf_counter()
            with region("compile"):
                compiled[signature] = (self.compile(state, batch, trials), self.step_flops)
            seconds = time.perf_counter() - began
            links = {axis: AxisLink(link.bytes_per_second, link.spread)
                     for axis, link in self.links.items()}
            self._report(StepCompiled(seconds, recompute_record(self.objective), links, self.memory,
                                      self.plan), int(trials.state.step) if reported is None else reported)
            self._unproven = True
        return (*compiled[signature], trials.state)

    def _saved_checkpoint(
        self,
        checkpoints: Checkpoints,
        step: int,
        state: TrainState,
        position: bytes | None,
        interval: Interval,
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
            metadata.setdefault('train/loss', float(jax.device_get(interval.book.loss) / interval.steps))
            if not ranking and training_best:
                ranking = (Ranking('train/loss', metadata['train/loss']),)
        write = functools.partial(
            checkpoints.save, step, state, position, metadata, share=DataPartition.of(self.device_mesh),
            ranking=ranking, control={**(control or {}), **interval.token_control},
            weights_only=weights_only, rung=self._checkpoint_rung(), artifact=self._artifact(),
            boxed=self._boxed_axes())
        self._report(CheckpointRequested(checkpoints.directory), step, write)
        interval.saved(step, resumable=not weights_only)
        self._display.status("")
        return time.perf_counter() - paused

    def _saved_local_checkpoint(
        self,
        checkpoints: Checkpoints,
        step: int,
        state: TrainState,
        position: bytes | None,
        interval: Interval,
        *,
        control: dict | None = None,
    ) -> float:
        """Write one checkpoint to the local directory, and report it.

        Returns the seconds it took. The local copy carries no metadata; it
        is the one a restarted node reads back, not the run's record."""
        paused = time.perf_counter()
        self._display.status("writing a local checkpoint")
        write = functools.partial(
            checkpoints.save_local, step, state, position, share=DataPartition.of(self.device_mesh),
            control={**(control or {}), **interval.token_control}, rung=self._checkpoint_rung(),
            artifact=self._artifact(), boxed=self._boxed_axes())
        self._report(CheckpointRequested(str(checkpoints.local_directory), local=True), step, write)
        self._display.status("")
        return time.perf_counter() - paused

    def _read_taken(self, run: FitRun, *, agree: bool = True) -> None:
        """Read the log the loop took (`Taken`), check its losses and report
        one logging interval, its clock ending when its step was done.

        Rank zero builds the row; every other rank has nothing to report.
        The report is agreed, so a tracker that failed on rank zero stops its
        peers here rather than at their next collective; a failed run's
        cleanup reads without agreeing, since its peers may never agree.

        Loss and objective aux metrics are the logged step's values.
        train/grad_norm is instead the mean over admitted optimizer commits
        since the last log: bookkeep sums each window's native gradient norm
        and its commit count on device, independently of the checkpoint loss
        sum. A tick with no commit omits it. train/compiles counts this fit's
        StepCompiled records; persistent-cache counts span all jitted programs
        in this process, as compilation_cache_counts describes.

        The optimizer state supplies the last applied update's rate
        (`learning_rate`): a rejected attempt leaves it, and an accumulation
        window moves it only when its update commits. It joins the same host
        read as the other interval metrics.
        """
        taken, run.taken = run.taken, None
        if taken is None:
            return
        counts = taken.counts

        def report() -> None:
            if jax.process_index() != 0:
                return
            # A work count a process refused (`_counted_batch`) is NaN here, and train/mfu absent.
            work = None if read.training is None else float(read.training)
            scalars = {"train/loss": float(read.loss),
                       **{f"train/{k}": float(v) for k, v in read.aux.items()},
                       **throughput(elapsed, counts.since_log, counts.samples, counts.flops,
                                    self.device_mesh.devices.size, waited=counts.waited,
                                    compiling=counts.compiling, training_flops=work,
                                    tokens=None if read.tokens is None else _TokenCount.host(read.tokens))}
            scalars["train/accepted"] = float(read.accepted)
            if read.committed is not None:
                norms, commits = read.committed
                if commits:
                    scalars['train/grad_norm'] = float(norms / commits)
            scalars['train/compiles'] = float(counts.compiles)
            scalars.update(compilation_cache_counts())
            peak = peak_memory_gib()
            if peak is not None:
                scalars['train/peak_memory_gib'] = peak
            if read.rate is not None:
                scalars["train/learning_rate"] = float(read.rate)
            if read.scale is not None:
                scalars["train/loss_scale"] = float(read.scale)
            if self.rollout is not None:
                scalars["train/rollout_seconds"] = counts.rollout_seconds
                scalars.update({f"rollout/{name}": float(value) for name, value in read.rollout.items()})
            scalars.update(read.statistics)
            scalars.update(self._startup.scalars())
            self._display.interval(taken.step, scalars)
            if self.tracker is not None:
                self.tracker.log(scalars, taken.step)

        with region("log"):
            # One read for every number, since each float() of a device array is a copy of its own.
            read, elapsed = taken.read(self._display)
            if agree:
                agreed("training reporting", report)
            else:
                report()

    def _reported_outcome(self, error: BaseException | None, run: FitRun) -> BaseException | None:
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
                    # A run that logged no step reports its start here (`Startup.scalars`).
                    scalars = {**goodput(wall, run.first_step, run.other), **self._startup.scalars()}
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
                                       preempted=run.preempted is not None,
                                       lowered=isinstance(error, Lowered))
            self._report(outcome, run.current)
        except BaseException as failure:
            if error is None:
                error = failure
            else:
                error.add_note(f'Reporting fit outcome failed: {failure!r}')
        return error

    def _report(self, value: Record, step: int, after: Callable[[], None] | None = None) -> None:
        """Hand one record to the tracker on rank zero, then agree with the pool.

        The record's type names the phase, so a rank that failed to report is
        heard about at the report the pool was making, not at the next
        collective. `after` is the work the record reports, run first inside
        the same agreement: a rank whose work fails after its own coordination
        rounds (a checkpoint's position gather, then its write) meets its
        peers at this report, rather than skip it while they wait there."""
        def send() -> None:
            if after is not None:
                after()
            if jax.process_index() == 0 and self.tracker is not None:
                self.tracker.artifact(value, step)

        agreed(type(value).__name__, send)

    # ------------------------------------------------------------------ Validation
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
        warm: bool = False,
        scored_to: str | None = None,
    ) -> Evaluation:
        """Score the validation split with this state's variables and report
        it, returning the seconds it took: the run's, but not its steps',
        which is what the goodput fraction is measured against. `warm` warms
        the split (`Evaluation.run`'s `warm`) and reports nothing.

        A CPU-owned run evaluates on the accelerator, over the same snapshot
        a step realizes, so validation reads the weights where the loss does."""
        paused = time.perf_counter()
        if not warm:
            self._display.status("evaluating")
        mesh, params = self.device_mesh, state.variables
        averaged = with_ema(state.variables, self._fetched(state, shardings).ema)
        key = state.key
        if self.host_master:
            from dew.training.execution import HostExecution, stack_sites
            execution = HostExecution(self.objective, self.layout, mesh, self.state_mesh,
                                      stack_sites(self.objective, state.variables))
            with jax.set_mesh(self.state_mesh):
                params, averaged = execution.snapshot(params), execution.snapshot(averaged)
            key = execution.on_accelerator(key)
        assert params is not None, "evaluation always has model variables"
        # The rules and the microbatch count; `Evaluation.run` scores under the mesh itself, and
        # previews decode outside it. The objective's schedules read steps, as training does
        # (`Transaction`): a window of window_size microbatches is `accumulation` steps.
        with (pipeline_microbatches(self.microbatches),
              nn.logical_axis_rules(self.layout.axis_rules), measured_links(self._links(mesh))):
            evaluation = Evaluation.run(
                self.objective, params, reader, metrics=metrics, key=key,
                step=state.step, schedule_step=state.microstep * self.accumulation // state.window_size,
                averaged=averaged, preview=preview, mesh=mesh, loss=loss, split=split, training=training,
                warm=warm, updates=state.updates,
                scored_to=None if scored_to is None else os.path.join(scored_to, f"step-{int(state.step)}"))
        if not warm:
            # The training values ride along for ranking (`_ranking`); the log interval reports
            # them, so the evaluation's own report leaves them out.
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


    # ------------------------------------------------------------------ Telemetry
    # ------------------------------------------------------------------

    @staticmethod
    def _open_window(run: FitRun, profile: ProfileWindow | None, profiler: Profiler | None, seen: int,
                     compiled: Mapping[Signature, object]) -> None:
        """Open the profile window's capture before an iteration's first read, so the step row records
        the read it waits on, once its warmup steps have run and the run's first step has compiled: the
        window holds steps, not the compile."""
        if (profiler is not None and profile is not None and not run.tracing and run.traced == 0
                and seen >= profile.warmup and compiled):
            start_capture(profiler)
            run.tracing = True

    def _stop_trace(self, run: FitRun, profile: ProfileWindow, profiler: Profiler) -> None:
        """Stop the window's owned capture (`stop_capture`), then report it on process zero."""
        stop_capture(profiler, run.loss)
        self._report(ProfileWindowRecord(profile.directory, run.traced), run.current)
        agreed("profile announcement", lambda: self._display.note(
            f"Wrote profile for {run.traced} steps to {profile.directory}"))
