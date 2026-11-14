"""The trainer: mesh, compiled step, EMA, checkpoints and logging.

What is learned is up to the objective (`dew.objectives.base`). The trainer creates the objective's variables
on the mesh, compiles one step over the global batch, and updates the EMA copy once per optimizer update. Side
effects go to the objects it was given: a `Checkpoints` for disk and a `Tracker` for numbers and artifacts.

Constructing a trainer opens nothing; the mesh, the compiled step and those objects' resources are created in
`fit`.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import datetime
import functools
import logging
import os
import sys
import time
import types
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Generic, Unpack

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P
from typing_extensions import TypeVar as DefaultTypeVar

from dew._cleanup import attempted
from dew._coordination import agree_process_phase, agreed, agreed_same, broadcast_from_process_zero
from dew._records import JSON
from dew.batch import FROZEN, Batch, at_path, owned_dicts, placed_at, select, unboxed
from dew.checkpoints import Checkpoints, Envelope, Ranking, Resume
from dew.data.compile import Handed
from dew.data.dataset import (
    Checkpointable,
    Closeable,
    DataPartition,
    Loader,
    PhasedStream,
    RampedStream,
    ReadsInProcess,
    StepPhases,
    stream_identity,
)
from dew.data.pipeline import Dataset
from dew.data.plan import DataResume, PlanChanged, environment_changes
from dew.distributed.batches import (
    PREFETCH_DEPTH,
    DevicePrefetchIterator,
    batch_layout,
    batch_shardings,
    batch_signature,
    placed_abstract,
    shard_batch,
    split_fields,
)
from dew.distributed.host import snapshot
from dew.distributed.placement import Layout, MeshSpec, Placement, Validation, device_mesh, placed_shapes
from dew.held_updates import waiting
from dew.inputs import Field, InputSpec
from dew.mesh import STAGE_AXIS, LayoutRefused, LogicalAxes
from dew.nn.flops import MissingFlopsCount
from dew.nn.kernels.options import Compiling, attempting, owned_program
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import Link, Schedule, boxed, measured_links, pipeline_microbatches, placing
from dew.objectives.base import (
    Aux,
    Initializer,
    Metric,
    Objective,
    Ratio,
    Recorded,
    TrainingScalar,
    taken_record,
)
from dew.optim import STACKED_OPTIMIZERS, declaring, with_bounds
from dew.optim_schedules import (
    ScheduleBase,
    held_rates,
    scaled_rates,
    schedule_states,
    settable_rates,
    with_horizons,
    with_rates,
)
from dew.registry import Unrecordable
from dew.scores import Summary
from dew.state import (
    MODEL_STATE,
    Accumulation,
    TrainState,
    apart,
    drawn_in,
    kept_window,
    occupied,
    paired,
    taken,
    together,
)
from dew.telemetry import profile as telemetry_profile
from dew.telemetry.instrumentation import StepFlops
from dew.telemetry.profile import region
from dew.telemetry.records import (
    AxisLink,
    CheckpointRequested,
    DataRestarted,
    FitEnded,
    FitStarted,
    Record,
    StepCompiled,
)
from dew.training._candidates import RunShape, Stepping, StepSettings, prepared_for
from dew.training._fit import (
    Emptied,
    FitArguments,
    FitPlan,
    FitRun,
    HandedSplits,
    Start,
    _rollout_admitted,
    _rollout_resumed,
    _rollout_saved,
    afresh,
    check_inputs,
    clocked,
    closes_window,
    consumption_counted,
    counted_step,
    declarations,
    first_batch,
    fit_policies,
    handed_splits,
    pooled_first,
    read_batch,
    restarted,
    restored,
    restored_or_restarted,
    resumed_on,
    stage_reached,
    unrecorded_parts,
    validation_read,
    waited_read,
)
from dew.training._lowering import Lowered
from dew.training.budget import Pause, Plateau
from dew.training.cadence import Counted
from dew.training.compiler import Compiled, CompiledStep, StepCompiler, freshly_held
from dew.training.cooldown import Cooldowns, Pending, check_branches, decay, resolved, schedule_ended
from dew.training.diloco import DiLoCo, dropped, grouped, rated, rates_set, windowed
from dew.training.display import TRAINER_SHOWN, TrainingDisplay, goodput
from dew.training.distributed import (
    Produced,
    check_batch,
    check_stream,
    marked,
    microbatch_shapes,
    placed_start,
    ruled,
    stage_microbatches,
    start_compiling,
    step_reads,
    traced_start,
)
from dew.training.evaluation import EvalSuite, Evaluator, episodic
from dew.training.inspection import Inspection
from dew.training.interval import Interval, Reading, Taken, TokenClock, copied, logged_scalars
from dew.training.memory import first_run, local_devices, recompute_record, refuse_wide_floats, split_share
from dew.training.model_flops import TOKENS, TRAINING_FLOPS, model_batch, prepare_counts, uncounted
from dew.training.observe import Observe
from dew.training.phase_passes import PhasePasses
from dew.training.planning import Candidate, Planned, Trials, read_whole, released, splits
from dew.training.profiling import ProfileWindow, owned_capture, pgle_runs, reported_window, waited_export
from dew.training.rollout import Rollout, _branched, _reported, _rollout_metrics
from dew.training.runtime import Preempted, PreemptionNotice, ready_process, signal_wake
from dew.training.selection import Best, rankings
from dew.training.stages import StageLocal
from dew.training.staging import fetched_state
from dew.training.startup import Startup, computation_differs, run_data
from dew.training.tracker import Tracker
from dew.training.transaction import (
    PassesProgress,
    Transaction,
    batch_statistics_shapes,
    compact_qk,
    loss_scale,
    pooled_by_mean,
    refuse_unpassed_progress,
    traced_loss,
)

_log = logging.getLogger(__name__)
_input_log = logging.getLogger("dew.training.input")
"""Where a run says what its input costs its steps, once (`_written`)."""

if TYPE_CHECKING:
    from dew.optim import OptimConfig
    from dew.telemetry.profile import Profiler
    from dew.training.ahead import AheadOfTime, LaterPlans

StepFn = Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]
"""A compiled step's body: the state and the global batch in, the new state,
the loss and the objective's report out."""

Loss = DefaultTypeVar("Loss", default=Ratio | jax.Array | float)
Effects = DefaultTypeVar("Effects", default=None)

_DEFAULT_MESH = MeshSpec()
_DEFAULT_LAYOUT = Layout()


class Trainer(Generic[Loss, Effects]):
    """Runs an `Objective`: gradients, sharding, EMA, checkpoints and logging."""

    _examples: int | None = None  # the bound of a run on the examples clock (`TrainState.consumed`)

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
        ahead: AheadOfTime | None = None,
        tune_kernels: bool = False,
        record: bool = False,
        links: Mapping[str, Link] | None = None,
        inspection: Inspection | None = None,
        param_dtype: str | None = None,
        diloco: DiLoCo | None = None,
        phase_passes: Sequence[int] | None = None,
        data_resume: DataResume = "relaxed",
    ):
        """Hold everything a run needs, without opening any of it.

        The mesh, the compiled step and the capabilities' resources come into being in `fit`, so constructing
        a Trainer allocates nothing.

        `accumulation` is how many steps pool into one optimizer commit, and `split` how many microbatches
        each step's batch is read in, its rows split evenly; None plans it, the fewest that fit the devices
        before any recomputation (`compile`). `optimizer` is an optax transformation, or an `OptimConfig` each
        `fit` builds over its `steps // accumulation` updates; the run steps the objective's
        (`Objective.optimizer`), made from it. `step` replaces the built-in transaction. A custom step then
        owns the clocks, the scaler, the EMA and the mutable writes, and the compiled wrapper owns only the
        attempted-step counter; it passes a model that reads the run's progress it (`PassesProgress`).
        `rollout` runs once per batch read, before the step and outside replay. `layout` and `mesh` say where
        the state lives: `mesh` a `MeshSpec` or a native `jax.sharding.Mesh` of any axes, which `layout`'s
        rules name (`Layout.data_axes` its rows'). `microbatches` is how many a pipelined step feeds through
        the mesh's stage axis, a multiple of its size; None feeds one a stage. `lower_for`, a CUDA generation,
        lowers the first step for it in place of compiling it (`dew.training._lowering.Lowered`); `ahead`, a
        step compiled ahead (`dew.training.ahead`), runs as the first step where it is that program, and
        one compiled without its GPU kind's tuned choices (`AheadOfTime.tuned`) is not run: the run compiles
        its step as without it, said once (`AheadOfTime.runnable`).
        `tune_kernels` opts in to the tuner (`dew.nn.kernels.tuning.tune`), else each launch takes its derived
        config. It measures each of Dew's own launches that declares variants to time (`KernelOp`: the
        kernel templates, `attention_impl="template"`, the norms, SwiGLU and decode attention); tokamax's
        flash kernel (`attention_impl="triton"`) runs at tokamax's heuristic config, which it does not
        measure. `record` makes each step return in its metrics what each
        evaluation, replays too, sampled or passed as a boundary (`taken_record`). `links` states an axis's
        `Link` by name in place of measuring it (`_links`). `param_dtype`, a dtype's name, stores the
        parameters the run draws fresh, as Flax's `param_dtype` (`dew.state.drawn_in`); None keeps each
        module's own. `inspection` names what the run watches at its sampled steps
        (`dew.training.inspection`). `diloco` trains the mesh's diloco groups apart, every few updates
        averaged (`dew.training.diloco`). `phase_passes` runs a looped model a count of passes a phase of
        the run's data (`dew.training.phase_passes`).
        """
        layout = objective.placement(layout)  # as its models place their trees (a tier-3 plan, ledger 2777)
        whole = read_whole(accumulation, split, step, layout, rollout, mesh)
        self.objective = objective
        built = isinstance(optimizer, optax.GradientTransformation)
        self._config, self._optimizer = ((None, declaring(objective.optimizer(optimizer), self._boxed_axes))
                                         if built else (optimizer, None))
        from dew.nn.inputs import seed_of
        # A seed's key is made when first read, after `fit` readies the process (`key`).
        self.seed, self._given_key, self._key = seed_of(key), key, None
        self.mesh = mesh
        self.layout = layout
        self.accumulation = accumulation
        self.split = split
        # How a resume treats its data (`DataResume`): relaxed starts it again where its plan differs.
        self.data_resume: DataResume = data_resume
        self._microbatches, self._loss_scale = microbatches, loss_scale(dynamic_scale)
        self.dynamic_scale = self._loss_scale is not None
        # A loss scale stated by a flag alone is a value a step's trace can be kept by (`_step_key`).
        self._scaled = dynamic_scale if isinstance(dynamic_scale, bool) else None
        self.checkpoints = checkpoints
        self.tracker, self.step, self.rollout = tracker, step, rollout
        if record and step is not None:
            raise ValueError("a custom step builds its own Steps, which Trainer(record=True) cannot reach: "
                             "build them with Step(record=True)")
        if inspection is not None and step is not None:
            raise ValueError("a custom step commits no update an inspection reads (Trainer(inspection=...)); "
                             "summarize its arrays in the step's own report")
        self.record = record
        self.records: list[Recorded] = []
        self.profile = profile
        self.lower_for, self.tune_kernels = lower_for, tune_kernels
        self.inspection, self.param_dtype, self.diloco = inspection, param_dtype, diloco
        self.phase_passes = PhasePasses.of(objective, phase_passes)
        # The later data phases' steps of the fit in progress, planned off its loop (`LaterPlans`).
        self._later: LaterPlans | None = None
        if diloco is not None:
            diloco.admits(self.layout.host, record)
        # Where the fit ladder stands and the links its traces read (`StepCompiler`), and what the last
        # compile found (`Compiled`), which `plan`, `memory`, `executable`, `step_flops` and `links` read.
        self._compiler = StepCompiler(objective, links, None if ahead is None else ahead.runnable(),
                                      self.phase_passes)
        self._compiled: Compiled | None = None
        # Whether a step has run since the last compile proved its plan fits (`memory`).
        self._unproven = False
        # The last validation of the state between steps (`validate`): as placed, then as each step holds it.
        self.validation: Validation | None = None
        self._display = TrainingDisplay()
        self._refusal: str | None = None
        # Why this run's checkpoints describe no model to load, said once.
        self._unrecorded: str | None = None
        # How the run's batches are cut and counted (`RunShape`): the microbatches a step's batch is read in
        # start 1 where `whole`, else as stated, until a pooling state's window or the first plan decides.
        self._shape = RunShape(split=1 if whole is not None else split)
        # The split the run was given, which a phase planned afresh at an update's end plans under
        # (`_compiled_for`); None where the planner chooses.
        self._given_split = self._shape.split

    @property
    def optimizer(self) -> optax.GradientTransformation:
        """The optimizer the run's steps commit with, once built over its length
        (`StepSettings.optimizer`)."""
        return self._settings().optimizer

    def _settings(self, objective: Objective[Loss, Effects] | None = None) -> StepSettings[Loss, Effects]:
        """What this trainer states of the step it compiles, as it stands (`StepSettings`): the one place a
        compile reads them from."""
        return StepSettings(self.objective if objective is None else objective, self._optimizer, self.layout,
                            self.device_mesh, self.accumulation,
                            self.step, self.dynamic_scale, self.diloco, self.tune_kernels, self.inspection,
                            self.microbatches, self.record, self.lower_for, self.state_mesh)

    # ------------------------------------------------------------------ The state
    # ------------------------------------------------------------------

    def initial_state(self, initializer: Initializer | None = None,
                      key: int | jax.Array | None = None) -> TrainState:
        """Build the state a fresh run starts from, a DiLoCo run's with its groups' copies (`DiLoCo.started`),
        every layer whole, which a stage-local step stacks by stage (`_staged_start`). It is pure, so `fit`
        traces it once for its shapes and once, sharded, for its values.

        Both inputs default to the run's own, and `place` passes them explicitly so that what it compiles
        takes them as arguments. A held checkpoint then reaches the device as an argument instead of a
        constant embedded in the executable. Passing None means "use the configured input", which is what a
        call without arguments does. This is the only implementation of the state, so a subclass overrides it
        here and every path sees the override.
        """
        initializer = self.objective.initializer if initializer is None else initializer
        from dew.nn.inputs import request_key
        ready_process()  # A key given here opens the backend as the run's does (`key`).
        key = self.key if key is None else request_key(key)
        init_key, run_key = jax.random.split(key)
        initialized = initializer(init_key)
        # The state holds its trees as dicts of its own, whichever Flax container (a FrozenDict, under a
        # mapping or a list) the objective's tree came in, so its placement, its updates, the
        # collections a step writes back (`write_back`) and its restores share one structure.
        params = drawn_in(owned_dicts(unboxed(initialized)), self.param_dtype, self.objective)
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
        # The optimizer reads the axes the tree boxes (`optim.declaring`), Muon's grouping among them, each
        # rule a stage-local run's leaves as its layers (`StageLocal.held`).
        with self._staging.held():
            opt_state = self.optimizer.init(params["params"])
        ema = self.objective.ema
        state = TrainState(
            step=jnp.zeros((), jnp.int32),
            microstep=jnp.zeros((), jnp.int32),
            updates=jnp.zeros((), jnp.int32),
            # A state of its own, the given scale's leaves copied: the step donates the state.
            scale=None if self._loss_scale is None else jax.tree.map(
                lambda leaf: jnp.array(leaf, copy=True), self._loss_scale),
            window_size=jnp.asarray(self.accumulation, jnp.int32),
            variables=params,
            opt_state=opt_state,
            # The average starts equal to the parameters but as its own buffers: the step donates
            # the state, and a buffer can be donated once.
            ema=None if ema is None else jax.tree.map(jnp.copy, select(params, ema.select)),
            key=run_key,
            consumed=None if self._shape.examples is None else jnp.zeros((), jnp.int32),
            deferred=waiting(opt_state))
        return state if not self.diloco else self.diloco.started(state, self.device_mesh)

    def staged(self, state: TrainState) -> TrainState:
        """`state` (`initial_state`'s, `stored`'s) held and placed as the step holds it, as `fit` starts."""
        return self._staging.supplied(self._staging.received(self.objective, state), self.shardings)[0]

    def stored(self, state: TrainState) -> TrainState:
        """`state` with a stage-local run's layers back on `layers_N`, whole on every stage, every mirror
        alike: for a reader by layer name (an export), at the cost of every layer on every device."""
        return self._staging.stored(state)

    def _staged_start(self, initializer: Initializer | None = None, key: int | jax.Array | None = None
                      ) -> TrainState:
        """`initial_state` as the step holds it, a stage-local run's layers stacked by stage (`staged`): the
        state built first, so the layout reads the shapes its draw recorded (`Objective.drawn_shapes`)."""
        state = self.initial_state(initializer, key)
        return self._staging.state(state, self._staging.staged)

    @property
    def key(self) -> jax.Array:
        """The run's PRNG key: the one it was given, or its seed's, made when first read, which every
        path that builds state does, after readying the process (`ready_process`) and outside any
        trace: constructing a trainer opens no backend before its flags and cache are set."""
        if self._key is None:
            from dew.nn.inputs import request_key
            ready_process()
            with jax.ensure_compile_time_eval():
                self._key = request_key(self._given_key)
        return self._key

    def reading(self, loader: Loader, prefetch: int = PREFETCH_DEPTH, *, examples=None, steps=None) -> None:
        """Take `loader`'s device steps, prefetch and clock (`clocked`), and the batches its steps read, each
        phase's (`Loader.batches`), which every split a plan takes cuts (`splits`), as fit and compile-ahead
        do."""
        progress = self._config is not None and self._config.reads_progress()
        self._shape = dataclasses.replace(
            self._shape, device=loader.device, prefetch=prefetch, rows=loader.batches or (loader.batch,),
            examples=clocked(examples, steps, bound_ends=loader.bound_ends,
                             custom_progress=self.step is not None and progress))

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

    def _step_key(self, objective: Objective[Loss, Effects], compiler: StepCompiler) -> Hashable | None:
        """A value of everything this trainer's step traces from beside its arguments, equal for two trainers
        whose steps trace alike, so a refit or a resume in the same process that builds its trainer and
        objective anew reads the step its equal traced (`dew.training._candidates.candidate_for`; row 3873):
        the objective's own (`Objective.trace_key`, its EMA among it) at its current rung, the optimizer's
        config and the updates or examples it was built for, the reader's device steps, and every setting the
        step reads. The links it traces under are the candidate's key's (`trace_context`, their measured
        figures), its spreads replayed where it is kept (`Candidate.spread`), so the links the last compile
        decided key nothing (row 6022: every refit on a tensor- or sequence-split mesh traced its step again).
        None where a value states none of it: an objective that states no trace, an optimizer given built, a
        custom step, a device step or a setting compared by identity (`frozen_value`), a loss scale given
        built, stated links, or a run that measures its kernels' configs as it lowers. `objective` and
        `compiler` are those the step is built from and climbs on (`_stepping`)."""
        from dew.objectives.base import UNKEYED, frozen_value, stated_trace

        stated = stated_trace(objective)
        if (stated is None or self._config is None or self.step is not None or self._scaled is None
                or compiler.stated_links or self.tune_kernels):
            return None
        settings = frozen_value((self._config, self._shape.device, self.mesh, self.layout, self.inspection,
                                 self.param_dtype, self.diloco))
        if settings is UNKEYED:
            return None
        return (type(objective), stated, settings, self._shape.updates, self._shape.examples,
                self._microbatches,
                self.accumulation, self.split, self._scaled, self.lower_for, compiler.narrow,
                self.record)

    def _stepping(self, objective: Objective[Loss, Effects] | None = None,
                  compiler: StepCompiler | None = None) -> Stepping[Loss, Effects]:
        """What a fit search's candidates read of this trainer beside its public fields, as it stands
        (`dew.training._candidates.Stepping`): its step built from `objective` and climbed on `compiler`,
        the trainer's own where not given (a later phase's off the loop, `dew.training.ahead.LaterPlans`)."""
        objective = self.objective if objective is None else objective
        compiler = self._compiler if compiler is None else compiler
        bound = functools.partial
        return Stepping(self._settings(objective), bound(self._step_key, objective, compiler), self.shardings,
                        bound(self._loss_shape, objective=objective), self._staging,
                        bound(self._initialize_accumulation, objective=objective), self._model_batch,
                        self._model_shapes, bound(self._step_body, objective=objective),
                        bound(fetched_state, layout=self.layout), self._traced_on, self._shape,
                        self._validation, self._builds_window, bound(self._window_of, objective=objective))

    @functools.cached_property
    def _staging(self) -> StageLocal:
        """Where the step keeps each pipelined run stacked by stage (`StageLocal.of`); none for a CPU-owned
        state, which streams its stack's banks, nor a DiLoCo run, whose groups' copies stack apart."""
        built = self._config is not None and self._config.optimizer in STACKED_OPTIMIZERS
        return StageLocal.of(self.objective, self.device_mesh, optimizer=None if built else self.optimizer,
                             local=self.layout.stage_local and self.diloco is None and not self.host_master)

    def _groups(self, state: TrainState) -> dict[str, str] | None:
        """Where the optimizer's split sent each parameter (`OptimConfig.groups`), read under the axes the
        modules boxed, as the optimizer's state was initialized; None for an optimizer given built."""
        if self._config is None:
            return None
        with self._staging.held(boxed(self._boxed_axes())):
            return self._config.groups(state.variables["params"])

    def _boxed_axes(self) -> dict[tuple[str, ...], LogicalAxes]:
        """The axes the objective's modules declare on their parameters (`Objective.parameter_axes`), a
        drawn tree's and a held one's alike. A checkpoint records them for its restore."""
        return self.objective.parameter_axes

    @ruled
    def shardings(self, state: TrainState) -> Placement[TrainState]:
        """Return where each field of `state` is placed, on the axes that suit its kind.

        Parameter gradients follow parameters, replay records follow batches, and the layout's host-resident
        fields go in pinned host memory. Under a CPU-owned state, the frozen collection is the exception: it
        stays where the realization reads it (`execution.resident`) for the whole run.
        """
        with self._declared():
            return self._staging.placed(state, self._shardings)

    def _declared(self):
        """The context in which the layout, the optimizer and the step read the axes the objective's
        modules box: every one, which placement filters by its rules (`Layout._declared`), each rule a
        stage-local run's leaves as its layers (`StageLocal.held`)."""
        return self._staging.held(boxed(self._boxed_axes()))

    def _shardings(self, state: TrainState) -> Placement[TrainState]:
        mesh = self.state_mesh
        if state.inner is not None:  # DiLoCo's groups' copies, each on its own devices (`grouped`)
            return grouped(self._shardings, state)
        params, frozen = dict(state.variables), ()
        if self.host_master:
            from dew.training.execution import frozen_trees, stack_sites
            frozen = tuple(frozen_trees(params, stack_sites(self.objective, params)))
            for path, _, _ in frozen:
                params = placed_at(params, path, {})
        placed = self.layout.shardings(mesh, dataclasses.replace(state, variables=params, accumulation=None))
        placed = dataclasses.replace(placed, **{
            field: jax.tree.map(lambda s: s.with_memory_kind("pinned_host"), getattr(placed, field))
            for field in (() if self.host_master else self.layout.host)})
        for path, tree, sites in frozen:
            placed = dataclasses.replace(placed, variables=placed_at(
                placed.variables, path, self._frozen_shardings(state, path, tree, sites)))
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

    @ruled
    def validate(self, batch: Batch | None = None, *, steps: int | None = None) -> Validation:
        """What `fit` checks of the state before it places it (`Layout.validate`), read without
        placing anything, so a run is checked on a CPU host before a costly launch.

        The state is a fresh run's: its parameters, optimizer state and EMA, and, given one global `batch` as
        the run reads it, the accumulation its steps pool where `accumulation` is above 1, read in the first
        of the splits its plan may take whose devices hold it, the trainer's own where it fixes one
        (`splits`). It leaves out the narrow copies a step may read (`TrainState.compute`), which the plan
        drops first where the step does not fit with them. An `OptimConfig` is built over `steps`, as
        `fit(steps=...)` builds it. A trainer on compile-only GPUs
        (`dew.nn.kernels.generation.compile_only_gpus`) checks it against the memory stated for them, on a
        host that holds none.

        Raises what `fit` raises before placing the state: `LayoutRefused` for a state over a device's memory
        or a layout that replicates too much of the model, and ValueError for rules that name an axis the
        mesh lacks.
        """
        self.scheduled(None if steps is None else steps // self.accumulation)
        state = jax.eval_shape(self._staged_start)
        if batch is None:
            return self._validated(state)[1]
        return self._fitting(state, marked(DataPartition(), self._counted_batch)(batch))

    def _validated(self, state: TrainState) -> tuple[Placement[TrainState], Validation]:
        """`state`'s placement and what `Layout.validate` reads of the state it places, as the
        layout placed it, once every refusal before a placement passed: a float the devices
        cannot hold, a layout that replicates too much and a state over a device's memory."""
        refuse_wide_floats(state, self.device_mesh)
        shardings = self.shardings(state)
        variables, placement = state.variables, shardings.variables
        if self.host_master:
            # A CPU-owned run's frozen banks, read as the layers the layout placed.
            from dew.inference.banks import row_sharding, shape_rows, spread
            from dew.training.execution import frozen_trees, stack_sites
            for path, tree, _ in tuple(frozen_trees(variables, stack_sites(self.objective, variables))):
                variables = placed_at(variables, path, spread(tree, shape_rows))
                placed = at_path(placement, path)
                assert isinstance(placed, Mapping), path
                placement = placed_at(placement, path, spread(
                    placed, lambda sharding, count: [row_sharding(sharding)] * count))
        with self._declared():
            self._staging.check(self.layout, variables, placement, self.device_mesh)
        self.validation = self._validation(placed_shapes(dataclasses.replace(state, variables=variables),
                                                         dataclasses.replace(shardings, variables=placement)))
        _log.info("the state holds %.2f GiB on its fullest device between steps",
                  max(self.validation.device_bytes.values()) / 2 ** 30)
        return shardings, self.validation

    def _validation(self, placed: TrainState) -> Validation:
        """`Layout.validate` of `placed`, read in every axis the modules box, which every process
        agrees, each reading its own devices' memory."""
        with self._staging.held(boxed(self._boxed_axes())):
            return agreed("state placement", functools.partial(
                self.layout.validate, self.device_mesh, placed))

    @ruled
    def place(self) -> tuple[TrainState, Placement[TrainState], bytes | None]:
        """Put the state on the mesh, fresh or restored, once `validate`'s checks pass.

        It returns the state with its shardings and the data position a resume continues from, the state as
        the step holds it, which `compile` takes: a pipelined run's layers stacked by stage, and a one-stage
        scan's runs of parameters stacked as the scan reads them; `stored` reads it on its stored layers.
        """
        return self._placed(self._started(), positioned=True)

    def _taking(self) -> contextlib.AbstractContextManager[None]:
        """The block a run builds its start and its first step's copies in: a run handed a step compiled
        ahead runs the programs compiled with it as compiled (`AheadOfTime.taking`), compiling none."""
        return contextlib.nullcontext() if self._compiler.ahead is None else self._compiler.ahead.taking()

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
        abstract, through, build = traced_start(self._staged_start, initializer, key)
        checkpoints = self.checkpoints
        # Each process lists the checkpoints itself, and on a shared filesystem one listing can lag
        # another's: every process resumes from one step or none does.
        step = (None if checkpoints is None
                else agreed_same("newest checkpoint", lambda: checkpoints.resumed(resume)))
        if checkpoints is not None and step is not None:
            abstract = self._staging.state(abstract.with_window(checkpoints.accumulation_template(step)),
                                          self._staging.staged_shapes)
        shardings, validation = self._validated(abstract)
        return Start(initializer, key, abstract, through, step, shardings, build, validation.device_bytes)

    def _placed(self, start: Start, *, positioned: bool
                ) -> tuple[TrainState, Placement[TrainState], bytes | None]:
        """The state `start` describes on the mesh, fresh or restored, with its shardings and, where
        `positioned`, the data position a resume continues from; `fit` reads that alone, before it
        places the state (`Checkpoints.position`), and restores the state alone. A run handed a step
        compiled ahead builds it with the programs compiled with the step (`_taking`)."""
        with self._taking():
            state, shardings, position = self._placing(start, positioned=positioned)
        return self._stated(state), shardings, position

    def _stated(self, state: TrainState) -> TrainState:
        """`state` with its scheduled solvers' step horizons this run's (`OptimConfig.horizons`), which a
        configured run's optimizer holds as operands it sets on every state it steps (`stated`,
        `dew.optim_schedules.with_horizons`): its step and its start are one program at any length of run, a
        rehearsal's or a restart's served from the persistent cache. A restore's template (`_placing`)
        holds them made, so a checkpoint saved without them restores. A trainer given an Optax
        transformation keeps its own."""
        config = self._config
        if config is None:
            return state
        steps = self._shape.updates
        horizons = config.horizons(steps, clocked=self._shape.examples is not None)
        schedules = {group: rate for group, (rate, _) in config.grouped_rates().items()
                     if isinstance(rate, ScheduleBase)}
        return rates_set(state, lambda held: with_horizons(held, horizons, schedules, steps))

    def _placing(self, start: Start, *, positioned: bool
                 ) -> tuple[TrainState, Placement[TrainState], bytes | None]:
        initializer, key, abstract, shardings, resume = (start.initializer, start.key, start.abstract,
                                                         start.shardings, start.resume)
        checkpoints = self.checkpoints
        if checkpoints is None or resume is None:
            if self.host_master:
                from dew.training.execution import placed_host
                return (placed_host(self.objective, lambda: self._staged_start(initializer, key),
                                    self.state_mesh, self.device_mesh, shardings), shardings, None)
            return (placed_start(start.build, initializer, key, abstract, shardings, start.through,
                                 self.device_mesh), shardings, None)
        if self.host_master:
            from dew.training.execution import banked_frozen
            abstract = dataclasses.replace(abstract, variables=banked_frozen(
                self.objective, abstract.variables,
                lambda rows, path: jax.ShapeDtypeStruct((len(rows), *rows[0].shape), rows[0].dtype)))
        share = DataPartition.of(self.device_mesh) if positioned else None
        state, position = self._staging.restore(checkpoints, self._stated(placed_shapes(abstract, shardings)),
                                               resume, share)
        state = restored(state, resume, self._shape.examples)
        _rollout_resumed(self.rollout, checkpoints, resume, state.key.sharding)
        if (config := self._config) is not None:
            # The rate and bounds this run configures, not the ones its checkpoint holds.
            state = rates_set(state, lambda held: with_bounds(with_rates(held, config.rates()),
                                                              config.bounds()))
        self._compiler.resume_rung(checkpoints.rung(resume))
        # At the rung it trained on, the run's computation is the checkpoint's or the resume is refused.
        differs = computation_differs(checkpoints.artifact(resume), self._artifact())
        if differs:
            raise ValueError(f"the checkpoint at step {resume} in {checkpoints.directory} trained another "
                             f"computation ({', '.join(differs)}); resume it with the objective and model "
                             "it recorded, or train in another directory")
        # The code and machine that wrote the step's data position, which decide none of its rows: warned of.
        wrote = checkpoints.environment(resume)
        if wrote is not None:
            environment_changes(wrote)
        # A step an earlier Dew wrote records no key's implementation (`key_record`): its key data is the
        # run's own key's.
        key = state.key if jnp.issubdtype(state.key.dtype, jax.dtypes.prng_key) else (
            jax.random.wrap_key_data(state.key, impl=jax.random.key_impl(self.key)))
        state = dataclasses.replace(state, key=jax.device_put(key, shardings.key))
        if int(state.window_size) % self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        self._display.note(f"Resumed from step {resume} in {checkpoints.source(resume)}")
        return state, shardings, position

    def _artifact(self) -> JSON:
        """The objective's inference record, which a checkpoint carries for
        loaders. Training and resuming never read it, so a model or component
        no record can describe (`dew.registry.Unrecordable`: one defined where no import path reaches it,
        `__main__` or a function, or holding what no record carries) still checkpoints: the record says
        why, a loader raises that, and the first checkpoint warns it. Any other error is the record's own
        bug, and raises."""
        artifact: JSON
        reason = None
        try:
            artifact = self.objective.inference_record()
        except Unrecordable as error:
            reason = f"{type(self.objective).__name__}: {error}"
            artifact = {'unrecorded': reason}
        if reason is not None and reason != self._unrecorded:
            self._unrecorded = reason
            _log.warning("this run's checkpoints resume, but no loader can rebuild their model "
                         "(TextGeneration.from_run, dew.pipeline) until it is fixed: %s", reason)
        return artifact

    def _frozen_shardings(self, state: TrainState, path: tuple[str, ...], frozen, sites):
        """Return where a frozen tree at `path` of the variables (`execution.frozen_trees`), with the
        stacks `sites` over it, sits for a CPU-owned run.

        Before placement the layout names each row's shards, and `resident` moves the stack's rows to bank
        memory, a scanned run's shared rows as one bank. Once placed, the tree holds those banks,
        whose leading layer axis the layout's rules do not name, and its arrays say where they sit.
        """
        leaves = jax.tree.leaves(frozen)
        if leaves and all(isinstance(leaf, jax.Array) for leaf in leaves):
            return jax.tree.map(lambda leaf: leaf.sharding, frozen)
        from dew.training.execution import resident
        rows = self.layout.shardings(
            self.state_mesh,
            dataclasses.replace(state, variables=placed_at({}, path, frozen), accumulation=None))
        return resident(at_path(rows.variables, path), sites, self.device_mesh)

    # ------------------------------------------------------------------ The step
    # ------------------------------------------------------------------

    def _model_batch(self, batch: Batch) -> Batch:
        """The objective's inputs: the batch after the dataset's device steps,
        traced into the compiled step, without the trainer's accounting columns."""
        for step in self._shape.device:
            batch = step(batch)
        return model_batch(batch)

    def _model_shapes(self, batch: Batch) -> Batch:
        """`_model_batch`'s output, traced only where device steps change it,
        each leaf of an Explicit axis typed as the step reads it (which
        `jax.eval_shape` drops on a mesh whose other axes are Auto)."""
        if not self._shape.device:
            return model_batch(batch)
        traced, shapes = jax.make_jaxpr(self._model_batch, return_shape=True)(batch)
        return jax.tree.map(lambda aval: jax.ShapeDtypeStruct(
            aval.shape, aval.dtype, sharding=aval.sharding if aval.sharding.mesh.explicit_axes else None,
            weak_type=aval.weak_type), jax.tree.unflatten(jax.tree.structure(shapes), traced.out_avals))

    def _loss_shape(self, state: TrainState, batch: Batch, window: int, split: int = 1, *,
                    objective: Objective[Loss, Effects]):
        """The loss's program (`dew.training.transaction.traced_loss`, which narrow copies read) and its
        shapes."""
        loss, (variables, *rest) = traced_loss(objective, state, window, split)
        return jax.make_jaxpr(loss, return_shape=True)(variables, self._model_shapes(batch), *rest)

    def _initialize_accumulation(self, state: TrainState, batch: Batch, shapes, *, window: int,
                                 split: int = 1, shape_only=False, donor: Accumulation | None = None,
                                 objective: Objective[Loss, Effects]):
        """The state with the accumulation a `window` of microbatches of `batch`'s shapes pools in, empty;
        unchanged under a custom step or a held one that replays such batches (`Accumulation.replays`): a
        ramp's stage begins with a window (`Ramp.stages`). Under a window of one, the state holds none: an
        empty one it holds of another split (a phase planned afresh, `StepCompiler.afresh`) is the one its
        step lets go as it is handed the state (`laid_out`), so the step is planned over the state without
        it. Built over `donor`, the empty window it replaces, where given: each of its leaves of a new
        leaf's shape and dtype donated to it, its buffer the new one's, zeroed, the rest released
        (`taken`)."""
        if self.step is not None:
            return state
        if window == 1:
            return state if state.accumulation is None else dataclasses.replace(state, accumulation=None)
        stats, aux = shapes
        shared = pooled_by_mean(stats)
        mean_dtype = jnp.result_type(jnp.float32, *(x.dtype for x in jax.tree.leaves(stats)))
        slots = window - 1
        def shape(leaf: jax.Array) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)

        trainable = jax.tree.map(shape, state.variables["params"])
        records = jax.tree.map(shape, self._model_shapes(batch))
        mutable = (None if shared or aux.variables is None else
                   jax.tree.map(shape, {name: state.variables[name] for name in aux.variables}))
        batch_stats = batch_statistics_shapes(objective, state, records, window, split)

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
                tokens=jnp.zeros((2,), jnp.uint32) if TOKENS in batch else None,
                batch_statistics=None if batch_stats is None else tuple(zeros(x) for x in batch_stats),
                replayed_statistics=(None if shared or batch_stats is None
                                     else tuple(buffer(zeros(x)) for x in batch_stats)),
                statistics_stamp=None if batch_stats is None else jnp.full((2,), -1, jnp.int32),
                fill=jnp.zeros((), jnp.int32))

        pending = jax.eval_shape(allocate)
        # The microbatches the state's window holds are read only where it is replaced (`kept_window`).
        if kept_window(state.accumulation, pending, read=not shape_only):
            return state
        if not shape_only:
            placement = self.shardings(dataclasses.replace(state, accumulation=pending)).accumulation
            # Allocate on the final shards, without a replicated window-sized temporary, over the donor's
            # buffers where the old window's leaves have the new ones' shapes: none allocated twice.
            # The window it replaces: one laid out for another split (`donor`), or the state's own, a ramp's
            # stage's of other shapes.
            donor = state.accumulation if donor is None else donor
            released(taken(donor, pending)[1])
            pairs = paired(donor, pending)
            given = jax.tree.leaves(donor)
            donated = [given[index] for index in sorted(pairs.values())]
            place = {index: sorted(pairs.values()).index(held) for index, held in pairs.items()}

            def built(held):
                # Each leaf a donated one takes the place of is written over it: the donated buffer is an
                # operand, so XLA writes the new window's zeros into it.
                leaves, structure = jax.tree.flatten(allocate())
                for index, at in place.items():
                    leaves[index] = held[at].at[...].set(leaves[index])
                return jax.tree.unflatten(structure, leaves)

            pending = owned_program(built, out_shardings=placement, donate_argnums=0)(donated)
        return dataclasses.replace(state, accumulation=pending)

    def _builds_window(self, state: TrainState, candidate: Candidate, window: int) -> bool:
        """Whether a step of `window` microbatches over `state` builds its window first (`_window_of`): a
        state that holds none, or a ramp's stage whose window replays other shapes."""
        held, staged = state.window, self._shape.ramp or self._shape.stage_starts
        return window > 1 and self.step is None and (
            held is None or (bool(staged) and not held.replays(candidate.prepared.window)))

    def _window_of(self, state: TrainState, candidate: Candidate, window: int,
                   donor: Accumulation | None = None, *,
                   objective: Objective[Loss, Effects] | None = None) -> TrainState:
        """`state` with the accumulation `window` microbatches of `candidate`'s pool in, empty, on its shards
        (`windowed`), built over `donor`, the empty window it replaces, where given: what a run's first step
        of a window builds, and a run compiled ahead compiles (`dew.training.ahead.started`)."""
        return windowed(lambda state, rows, *, shape_only: self._initialize_accumulation(
            state, rows, candidate.shapes, window=window, split=window // self.accumulation,
            shape_only=shape_only, donor=donor,
            objective=self.objective if objective is None else objective), state, candidate.micro,
            self.shardings)

    @contextlib.contextmanager
    def _traced_on(self, mesh: Mesh | AbstractMesh,
                   links: Mapping[str, Link] | None = None) -> Iterator[Schedule]:
        """What a traced step reads from context: the mesh, the pipeline's
        microbatch count, the layout's rules, which place the activations
        the model constrains (`dew.nn.sharding.constrain`) as they place the
        parameters, the links of the axes the step measured, none by
        default, and the layout itself with the axes the modules declared
        (`placing`, `_declared`), where a projection mapped per shard reads
        its weight's placement (`Layout.leaf_spec`). An abstract mesh is a
        GPU's this process lacks (`_candidates.lowered_for`)."""
        on = jax.sharding.use_abstract_mesh(mesh) if isinstance(mesh, AbstractMesh) else jax.set_mesh(mesh)
        with (on, pipeline_microbatches(self.microbatches) as schedule, self._declared(),
              nn.logical_axis_rules(self.layout.axis_rules), measured_links({} if links is None else links),
              placing(self.layout)):
            yield schedule

    def _step_body(self, shapes, copies, split: int, resting: Mapping[str, object] | None = None,
                   sampled: bool = False, placed=None, *, objective: Objective[Loss, Effects]):
        """The objective transaction over a step's batch read in `split` microbatches, which brings the
        state's fields `resting` places in host memory to the device only once its gradient is ready, a
        chunk at a time (`Transaction.past`), and, `sampled`, summarizes what the run inspects at its
        commit; or a custom step with no invented commit telemetry, which reads it whole."""
        if self.step is None:
            return Transaction(objective, self.optimizer, split * self.accumulation, shapes, copies,
                               split=split, record=self.record, resting=resting,
                               examples=self._shape.examples,
                               inspection=self.inspection if sampled else None, placed=placed).step()
        # A custom step need not run the objective's loss, which passes a model the run's progress.
        refuse_unpassed_progress(objective, passes=isinstance(self.step, PassesProgress))
        custom = self.step(objective, self.optimizer)

        def run(current, batch, part, tokens, closes=None, whole=None):
            advanced, loss, aux = custom(current, batch)
            return advanced, loss, aux, None
        return run

    def _warmed(self, plan: FitPlan, run: FitRun, state: TrainState, shardings: Placement[TrainState], *,
                first: bool) -> None:
        """The evaluations' programs warmed beside a step's compile (`Evaluator.warm`): the run's first, and
        one at another rung than they were last warmed at (`FitRun.warmed`)."""
        if not first and self._compiler.rung() == run.warmed:
            return
        with run.startup.phase("warm", "warming the evaluations", when=first), run.held():
            self._evaluator().warm(plan, state, shardings, self._compiler.links(self.device_mesh),
                                   first=first)
        run.warmed = self._compiler.rung()

    def _planned_afresh(self) -> None:
        """The step planned afresh, as where a phase of other passes begins at an update's end: at the split
        the run was given, or the planner's for its own passes."""
        self._shape = dataclasses.replace(self._shape, split=self._given_split)

    @ruled
    def compile(self, state: TrainState, batch: Batch, trials: Trials | None = None) -> CompiledStep:
        """Compile a transaction over the state and one already-produced global batch
        (`StepCompiler.compile`).

        The step consumes its state: `new = step(old, batch)` donates the old buffers, keeping one
        copy of parameters and optimizer state. Keep no reference to the old state after stepping.
        Orbax copies a checkpoint to host before `save` returns, provided `Checkpoints` names neither
        prioritized keys nor a concurrent transfer limit. The loader's batch is not donated. The plan takes
        the fewest microbatches at least recomputation, a compile-only GPU holding the memory stated for it.
        Built at the pass count of the phase its batch was read in (`PhasePasses.scoped`).
        """
        with self.phase_passes.scoped(self, self._compiler):
            compiled = self._took(self._compiler.compile(self._stepping(), state, batch, trials))
        assert compiled.step is not None
        return compiled.step

    @ruled
    def _planned_unplaced(self, start: Start, batch: Batch) -> None:
        """A fresh run's first step planned and compiled over its state's shapes (`start`'s) before the
        state is placed, held for the run's first compile over `batch` (`StepCompiler.planned_unplaced`),
        which takes it where the devices hold it once the state is placed and otherwise plans afresh, at the
        split the run was given or the planner's own, not the held plan's:
        built at the pass count of the phase its batch was read in (`PhasePasses.scoped`), as `compile`
        builds it, its admission counting the bytes the start will place in a device's own memory beside it:
        the state's as the layout validates it (`Start.held`), none of a field kept in the host's, but for
        the frozen held arrays it keeps as they are, placed where it keeps them already (`placed_start`'s
        `through`), each device's own shards of them; every other held array it copies. The most any of
        this process's devices takes."""
        inputs = jax.tree.leaves(start.initializer)
        where = jax.tree.leaves(apart(start.shardings)[1])
        kept: dict = collections.Counter()
        for index, given in start.through.items():
            leaf = inputs[given]
            if (isinstance(leaf, jax.Array) and leaf.sharding == where[index]
                    and (leaf.sharding.memory_kind or "device") == "device"):
                for shard in leaf.addressable_shards:
                    kept[shard.device.id] += shard.data.nbytes
        unplaced = max(start.held[device.id] - kept[device.id] for device in local_devices(self.device_mesh))
        with self.phase_passes.scoped(self, self._compiler):
            self._compiler.planned_unplaced(self._stepping(), start.abstract, batch, unplaced)

    @ruled
    def _compiled_plan(self, state: TrainState, batch: Batch, trials: Trials | None,
                       frontier: bool = False) -> Planned:
        """The plan the run's step compiles at (`StepCompiler.plan`), taken as `compile` takes its compile;
        the caller reads its lowering before releasing it or serializing its executable (`compile_ahead`)."""
        planned, compiled = self._compiler.plan(self._stepping(), state, batch, trials, frontier)
        self._took(compiled)
        return planned

    def _took(self, compiled: Compiled) -> Compiled:
        """Take what a compile found: its result the trainer's last (`plan`, `memory`, `executable`,
        `step_flops`, `links`), the microbatches it decided the run's from here on, and the state its plan
        holds between steps its validation where it planned one."""
        self._compiled = compiled
        self._shape = dataclasses.replace(self._shape, split=compiled.split)
        if compiled.validation is not None:
            self.validation = compiled.validation
        return compiled

    @property
    def plan(self) -> JSON:
        """The plan of the last compiled step: its split, the rows of its microbatch and every compile its
        search made (`Probe`)."""
        return None if self._compiled is None else self._compiled.plan

    @property
    def memory(self) -> JSON:
        """The check that the last compiled step fits (`memory_record`)."""
        return None if self._compiled is None else self._compiled.memory

    @property
    def executable(self) -> jax.stages.Compiled | None:
        """The last step as compiled, whose memory analysis a benchmark reads; the program it was lowered
        from is released with its plan (`Planned.released`)."""
        return None if self._compiled is None else self._compiled.executable

    @property
    def step_flops(self) -> StepFlops | None:
        """The FLOPs of the last compiled step; a ramped run has one value per stage, which `fit` keeps
        beside each step."""
        return None if self._compiled is None else self._compiled.step_flops

    @property
    def links(self) -> Mapping[str, Link]:
        """The links the last compiled step decided its projections by, which say whether one spread over
        each axis (`dew.nn.sharding.Link`)."""
        return {} if self._compiled is None else self._compiled.links

    def _staged(self, state: TrainState, step: int) -> TrainState:
        """`state` to run step `step`: its window of `window_size` microbatches dropped where the step, of
        the attempt count a resumed run restores, begins a ramp's stage after the first and a rejection
        left the window part filled, so an update pools one stage's batches. Only the window's fields
        are compiled (`window_dropped`), so a host-master state's resident weights stay where they are."""
        if step not in self._shape.stage_starts or state.window is None:
            return state
        return dropped(state)

    @ruled
    def fit(
        self,
        dataset: Dataset | Loader | Handed,
        *,
        state: TrainState | None = None,
        resume: Resume = "auto",
        **asked: Unpack[FitArguments],
    ) -> TrainState:
        """Train until the step or real-token cap is reached, resuming the latest checkpoint as `resume` says;
        return the state as the run holds it, which over a pipeline's stages its model reads off the mesh
        (`off_stages`), and on one stage with every layer on its stored layers (`stored`).

        `resume="auto"` continues the checkpointer's latest checkpoint where it holds one and otherwise
        starts afresh; "require" refuses to start afresh, for a job that must continue, and "never" refuses
        a directory that already holds a run (`Checkpoints.resumed`). An explicit `state` is the run's
        initial state, in place of initializing one and of restoring a checkpoint (a float64 truth's widened
        copy of another run's, say), not a continuation, which is a checkpoint's alone: its input reader
        starts at the position `dataset` supplies.

        Every `log_every` steps, the tracker receives the loss, the objective's metrics and the throughput.
        `tokens` stops at the first microbatch that reaches this real-token budget, the tokens committed
        updates trained on (`train/committed_tokens`), persisted in checkpoints. Padding never counts.
        With both caps the first reached ends the run; with `steps=None`, a decaying schedule needs its own
        step horizon. `examples` trains on the chain's first source examples, each once with all its rows
        (`Loader.example_steps`).

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
        every suite as `cooldown/<split>/<metric>` and discarded (`dew.training.cooldown.Cooldowns`).

        Every `checkpoint_every` steps (or "5%", `FitPlan.cadences`), the state and data position are written;
        `checkpoint_final=True` also writes the final state, independently of that cadence. A duration is
        checked at each `log_every` step, so they are written at the first log step after that much time, and
        every `checkpoints.local_every` steps to the local directory. A preemption notice (a scheduler's
        SIGTERM; `PreemptionNotice`) stops the run at the next step every process agrees on. That step's state
        and data position are written, the final validation is skipped, and fit raises `Preempted`, which ends
        the program with status 143 unless caught. Run it again and fit resumes from there.

        Previews are generated only when `preview=True` and a tracker receives them; scalar reporting never
        triggers preview work. `scored_to`, a directory, takes every validation batch's scoring artifacts at
        each evaluation, `<scored_to>/step-<step>/<split>-<batch>.npz` (`Evaluation.run`), a cooldown
        branch's under `cooldown/` there. `observe` takes values of the state at a cadence of its own while
        the run trains (`Observe`). `device_prefetch` training batches are placed on the devices ahead of
        the step, which the step's memory plan holds room for. The process is readied first
        (`dew.training.runtime.ready_process`), as `prepare_process` readies it: a program that never called
        that gets Dew's log handler, and one whose JAX opened on the CPU beside an NVIDIA GPU stops.
        """
        # The main thread's waits on the data's workers take a signal however it lands (`signal_wake`),
        # from the plan's first read of the data on.
        with signal_wake():
            plan = self._fit_plan(dataset, **asked)
            run = FitRun(time.perf_counter(), plan.startup)
            # Each fit plans its step as the trainer's first did, at the split it was given, so a refit's
            # first step is keyed as that fit's was and reads the step it traced (row 6022); a state that
            # pools a window binds its own split as it compiles (`StepCompiler.compile`).
            self._planned_afresh()
            # A display of this fit's own: one before it may have run other steps.
            self._display = TrainingDisplay()
            profile, checkpoints, profiler = self.profile, self.checkpoints, owned_capture(self.profile)
            # One boundary lookup at setup: the prefetch worker and the step scopes share whichever profiler
            # owns the capture.
            tracer = profiler if profiler is not None else telemetry_profile.active_profile()
            try:
                # A run handed a step compiled ahead runs every program compiled with it as compiled
                # (`_taking`).
                with self._taking():
                    trained = self._training_loop(plan, run, profiler, tracer, profile, checkpoints, state,
                                                  resume)
                # The state a checkpoint would hold: the step rebuilds its narrow copies.
                trained = dataclasses.replace(trained, compute=None)
                # Whether this fit trained or found the run complete, inside the lifecycle, so a selection
                # that fails is the run's outcome.
                if plan.restore_best and run.preempted is None:
                    trained = self._best_model(trained, run)
            finally:
                primary = sys.exception()
                error = self._closed(run, primary, profiler)
                # After the run's log, tracker and outcome closed: a plan still running off the loop is
                # waited for, which a run that failed or stopped need not wait on to close.
                if self._later is not None:
                    self._later.stopped()
                    self._later = None
                if primary is None and error is not None:
                    raise error
        if run.preempted is not None:
            raise Preempted(run.preempted)
        # One stage's scan holds its runs stacked for its step alone: the state a run returns holds them on
        # their stored layers, as every reader by layer name reads them, at no device's cost but the copy.
        return trained if self._staging.stages > 1 else self._staging.stored(trained, consumed=True)

    def _fit_plan(
        self,
        dataset: Dataset | Loader | Handed,
        *,
        steps: int | None = None,
        epochs: int | None = None,
        tokens: int | None = None,
        examples: int | None = None,
        log_every: int | str | Counted = 100,
        eval_every: int | str | Counted | None = None,
        checkpoint_every: int | str | Counted | datetime.timedelta | None = None,
        checkpoint_final: bool = True,
        metrics: Sequence[Metric] = (),
        preview: bool = False,
        scored_to: str | os.PathLike[str] | None = None,
        best: str | Metric | TrainingScalar | types.MethodType | Best | Sequence[Best] | None = None,
        stop: Plateau | Pause | None = None,
        validation: HandedSplits | None = None,
        summaries: Sequence[Summary] = (),
        restore_best: bool = False,
        observe: Observe | None = None,
        device_prefetch: int = PREFETCH_DEPTH,
        cooldowns: Cooldowns | None = None,
    ) -> FitPlan:
        """What `fit` over these arguments (`FitArguments`) runs, before it places anything: its data compiled
        and read (`run_data`, `reading`), its optimizer built for its updates (`scheduled`), and its
        cadences, scores and policies (`FitPlan`). Its defaults are `fit`'s, stated here alone, which a run
        compiled ahead plans its evaluations by too (`dew.training.ahead.fit_plan`)."""
        startup = Startup()
        loader, held, tokens, steps, examples = run_data(
            self.objective, dataset, handed_splits(validation), steps=steps, epochs=epochs, tokens=tokens,
            examples=examples, accumulation=self.accumulation, startup=startup, resume=self.data_resume)
        self.reading(loader, device_prefetch, examples=examples, steps=steps)
        self.scheduled(None if steps is None else steps // self.accumulation)
        # Its cadences resolved against the run's length, its scores declared, its best and stop policies,
        # and its refusals made before anything is placed (`_preflight`).
        logged, evaluated, saved, local, suites = FitPlan.cadences(
            loader, held, metrics, steps, tokens, self.accumulation,
            (log_every, eval_every, checkpoint_every), self.checkpoints, self._shape.examples)
        declared = declarations(suites, metrics, summaries, loss=self.objective.defines_validation_loss)
        selection, plateau, pause = fit_policies(self.objective, best, stop, declared, saved, steps=steps,
                                                 tokens=tokens, accumulation=self.accumulation,
                                                 checkpointed=self.checkpoints is not None,
                                                 examples=self._shape.examples)
        self._preflight(loader, plateau, suites, preview=preview, stated=dataset,
                        loss=(self.checkpoints is not None or restore_best) and not selection)
        return FitPlan(
            dataset=loader, steps=steps, log_every=logged, eval_every=evaluated, tokens=tokens,
            checkpoint_every=saved, checkpoint_final=checkpoint_final, local_every=local,
            metrics=metrics, preview=preview and self.tracker is not None, best=selection, stop=plateau,
            suites=suites, pause=pause, scored_to=None if scored_to is None else os.fspath(scored_to),
            held_best=restore_best and self.checkpoints is None, restore_best=restore_best, observe=observe,
            summaries=tuple(summaries), startup=startup,
            declared=declared, branches=() if cooldowns is None else cooldowns.branches(
                steps, tokens, self.accumulation, self._shape.examples),
        )

    def scheduled(self, updates: int | None) -> None:
        """Build the configured optimizer (an `OptimConfig`) for `updates` updates, as `fit` and
        `validate` do for their run's length; its state holds the rate it logs. A trainer given an Optax
        transformation keeps it."""
        if self._config is None:
            return
        built = self._config.build(updates, clocked=self._shape.examples is not None, stated=True)
        self._shape = dataclasses.replace(self._shape, updates=updates)
        self._optimizer = declaring(self.objective.optimizer(built), self._boxed_axes)

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
        current, position, placed, abstract, fresh = self._starting(run, plan.dataset, initial, resume)
        run.last_checkpoint = time.perf_counter()
        interval = Interval.resumed(run.stop_control, last_saved=(
            current if checkpoints is not None and checkpoints.latest is not None else None))
        state, shardings, complete = self._opened(plan, run, interval, current, position, placed, abstract,
                                                  resumed=initial is None and interval.last_saved is not None,
                                                  fresh=fresh)
        if complete:
            return state
        # The update's work beyond the model's, of an optimizer the objective steps its params with whole: one
        # that phases it apart (`Objective.optimizer`, a consistency distillation's two networks) states none.
        if self._config is not None and type(self.objective).optimizer is Objective.optimizer:
            # Read under the axes the modules boxed, a stage-local run's stacked by stage, as `_groups` reads.
            with self._staging.held(boxed(self._boxed_axes())):
                run.update_flops = self._config.update_flops(state.variables["params"])
        compiled: dict[Hashable, tuple[CompiledStep, StepFlops | None]] = run.compiled
        seen = 0
        run.notice = PreemptionNotice()
        run.clock = TokenClock(*interval.clocked(state.pending_tokens))
        if run.cooling is not None and not plan.ended(run.current, run.clock.read, interval.consumed):
            # A branch a preemption cut short runs again before the run reads on from where it stood.
            state, cut = self._cooled(plan, run, interval, state, shardings, position, run.cooling)
            if cut:
                run.preempted = run.current
                return self._wind_down(plan, run, interval, state, shardings, position, profiler)
        while not plan.ended(run.current, run.clock.read, interval.consumed):
            # A schedule's stage ends where the commits reach its tokens.
            self._scheduled(plan, run)
            run.open_window(profile, profiler, seen, bool(compiled))
            capturing = tracer is not None and tracer.running
            step_scope = (jax.profiler.StepTraceAnnotation("train", step_num=run.current)
                          if capturing else contextlib.nullcontext())
            with step_scope:
                # The first read restores the stream's position and opens its reader on the prefetch
                # worker, and a stream that may run out can on any read: a process whose read failed stops
                # its peers before they compile or step without it.
                read = read_batch(run, interval, state, self.rollout, self.device_mesh, run.marked,
                                  self._shape.examples, agree=run.agrees(first=not seen),
                                  objective=self._step_objective)
                if read is None:
                    self._display.note(f"The training data ran out at step {run.current}")
                    break
                (batch, device, ahead), read = read, None  # the batch's buffers held by `batch` alone
                # The step's pass count, the phase its batch was read in (`phase_passes`).
                self.phase_passes.at(batch)
                if self.phase_passes.key(batch_signature(device)) not in compiled and self.rollout is None:
                    # Each batch of a shape the run has not stepped is checked against what the objective
                    # declares, a later ramp's or schedule's stage as the first; a rollout writes its own.
                    check_inputs(self, device)
                known, began = len(compiled), time.perf_counter()
                with run.startup.phase("compile", "compiling the step", when=not known):
                    train_step, measured_flops, state = self._compiled_for(compiled, state, device, run)
                if len(compiled) > known and self.executable is not None and not self.host_master:
                    signature = batch_signature(device)
                    run.reads[signature] = step_reads(self.executable, signature)
                interval.compiles = len(compiled)
                if len(compiled) > known:
                    self._warmed(plan, run, state, shardings, first=not known)
                if not known:
                    # Rebound once the step is compiled, so the first tick measures steps, not the compile.
                    interval.last_log_time = time.time()
                    interval.waited = 0.0
                elif len(compiled) > known:
                    # A shape met later compiles inside the interval, which
                    # reports that time apart from its steps'.
                    interval.compiling += time.perf_counter() - began
                with region("train.step"):
                    state, loss, aux, finite, accepted, committed = train_step(
                        state, device, sampled=self.inspection is not None and self.inspection.samples(
                            run.current + 1, self.accumulation))
                if not seen and self.phase_passes.counts:
                    # Once the first step is dispatched, its later phases' steps are planned off the loop.
                    from dew.training.ahead import LaterPlans

                    self._later = LaterPlans.started(self, batch, state)
                # Written off the loop, which never waits on it (row 2313), its clock the step it sees done.
                interval.seen()
                self._read_taken(run, wait=False)
                if self._unproven:
                    # A step's failure on the devices surfaces at the next program that reads its
                    # outputs, the loss counters, so a newly compiled step's first run is awaited here,
                    # where its failure can be named.
                    self._unproven = False
                    first_run(loss, run.current, self.memory, self.device_mesh)
                run.loss = loss
                assert run.train is not None
                position = run.train.source_state if ahead is None else ahead
                run.current, seen = run.current + 1, seen + 1
                self._display.step(run.current)
                counted_step(interval, run, self.objective.row_tokens, batch, device, measured_flops, loss,
                             finite, committed, aux)
                # The step's batch is no cooldown branch's to hold beside its own (`_cooled`).
                batch = device = None
                if run.first_step is None:
                    loss.block_until_ready()
                    run.first_step = time.perf_counter() - run.started
            # The step row is complete once its scope exits; the window's last step closes it before the work
            # after it, whose evaluations, saves and cooldowns run outside it (`FitRun.outside_window`).
            if run.tracing and profile is not None:
                run.traced += 1
                if run.traced == profile.steps:
                    run.tracing = False
                    assert profiler is not None
                    run.close_window(profiler)
            stopped, state = self._between_steps(plan, run, interval, state, shardings, position, loss, aux,
                                                 accepted)
            if stopped:
                break
        return self._wind_down(plan, run, interval, state, shardings, position, profiler)

    def _closed(self, run: FitRun, primary: BaseException | None,
                profiler: Profiler | None) -> BaseException | None:
        """`fit`'s cleanup however the run ended: the preemption notice, the
        stream, a trace window and the display closed, the checkpoints waited on, and the outcome reported.
        Returns the error to raise when the run itself raised none.

        Each teardown step runs even when an earlier one failed, and a later
        failure becomes a note on the first, so one broken sink cannot hide the error that ended the run."""
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
            stop_trace = functools.partial(run.close_window, profiler)
        error = primary
        # A run's last log is written at its cleanup, a failed run's only where no write waits on a step
        # its peers may never finish (`LogWriter.raise_failure`).
        unread = functools.partial(self._read_taken, run, agree=False)
        exported = waited_export(profile, profiler)
        cleanups = (("Training log", unread), ("Log writer", run.writer.close), ("Training iterator", close),
                    ("Profiler", stop_trace), ("Profile export", exported),
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
                  ) -> tuple[int, bytes | None,
                             Callable[[Compiling | None], tuple[TrainState, Placement[TrainState]]],
                             TrainState | None, Start | None]:
        """Where `fit` starts, before it places anything: the step, the data position a resume reads
        alone (`Checkpoints.position`), so the data opens where the run left it, the placement still
        to make (handed the start's compile where one began, `attempting`), the state's shapes it checks
        first and, for a fresh state, its start. A supplied `initial` state is placed as it
        is given, once its shapes are checked as a fresh one's are: no continuation, which
        `resume="require"` refuses, and under "never" into a directory that holds no run either. An
        objective that names its input field alone declares it first (`_declare_input`)."""
        checkpoints = self.checkpoints
        self._declare_input(lambda: first_batch(dataset, self._stream_partition()))
        if resume == "require" and (checkpoints is None or initial is not None):
            raise ValueError("resume='require' continues the checkpointer's latest checkpoint, which a "
                             "trainer with no checkpointer, or a run given its state=, starts without")
        if initial is not None and checkpoints is not None and resume == "never":
            agreed_same("newest checkpoint", lambda: checkpoints.resumed(resume))
        if initial is not None:
            # A supplied state's trees are held as dicts too (`initial_state`), its optimizer's and
            # accumulation's mirrors of the parameters with them, each pipelined run stacked by stage. Its
            # run reads its data afresh: a clock of examples starts at none consumed, a run on another clock
            # keeps none, and its schedules take the run's own form (`reclocked`).
            initial = owned_dicts(self._staging.received(self.objective, dataclasses.replace(
                initial, variables=dict(initial.variables))))
            with self._staging.held(boxed(self._boxed_axes())):
                initial = self._stated(afresh(initial, schedule_states(self.optimizer,
                                                                       initial.variables["params"]),
                                              self._shape.examples))
            # Placed once the first batch's whole state, its accumulation among it, is checked against the
            # devices (`_fitting`), as a fresh or resumed one is.
            return (int(initial.step), None,
                    lambda begun=None: self._staging.supplied(
                        initial, lambda staged: self._validated(staged)[0]),
                    self._staging.stacked_shapes(initial), None)
        start = self._started(resume)
        if checkpoints is not None and checkpoints.latest is not None:
            run.stop_control = checkpoints.control(checkpoints.latest)
        if checkpoints is None or start.resume is None:
            def placed(begun: Compiling | None = None) -> tuple[TrainState, Placement[TrainState]]:
                with attempting(begun):
                    return self._placed(start, positioned=False)[:2]
            return 0, None, placed, start.abstract, start
        # Its saved position, which the stream the run opens restores, or starts again where its plan
        # changed (`_prefetched`); a position saved for other shares is none of this run's to read.
        try:
            position = checkpoints.position(start.resume, DataPartition.of(self.device_mesh))
        except PlanChanged as changed:
            if self.data_resume == "exact":
                raise
            position = None
            self._restarting(run, tuple(changed.differs))
        return (start.resume, position, lambda begun=None: self._placed(start, positioned=False)[:2],
                start.abstract, None)

    def _stream_partition(self) -> DataPartition:
        """The share of the run's data this process's training stream reads: the mesh's, its ramp's stages
        beginning where updates do (`window`, `Ramp.stages`)."""
        return DataPartition.of(self.device_mesh, window=self.accumulation)

    def _declare_input(self, first: Callable[[], Batch]) -> None:
        """Declare the objective's input where it names its field alone (`Objective.named_inputs`), at the
        per-example shape its model reads (`Field.of`) of the run's first batch, `first`'s, placed as the
        run places it and through the loader's device steps (`_model_shapes`), which may crop or resize
        it, agreed across the pool before anything initializes."""
        if (named := self.objective.named_inputs) is None or self.objective.inputs is not None:
            return
        batch = agreed("training input batch", first)
        with jax.set_mesh(self.device_mesh):
            device, _ = split_fields(placed_abstract(self.device_mesh, batch))
            shape = Field.of(named, self._model_shapes(device)).shape
        self.objective.inputs = InputSpec(Field(named, tuple(agreed_same("training input shape",
                                                                       lambda: list(shape)))))

    def _opened(self, plan: FitPlan, run: FitRun, interval: Interval, current: int, position,
                placed: Callable[[Compiling | None], tuple[TrainState, Placement[TrainState]]],
                abstract: TrainState | None, *, resumed: bool, fresh: Start | None = None,
                ) -> tuple[TrainState, Placement[TrainState], bool]:
        """Start `fit` at step `current`: refuse a run already past its steps, a checkpoint cadence
        with no checkpointer or a resume its stream cannot continue; open the training stream on its
        prefetch worker at `position` and read its first batch, against which the whole state a
        fresh or resumed run holds between steps (`abstract`'s, its accumulation included) is checked
        before anything is placed (`_whole_validated`); a `fresh` start's first step is planned and compiled
        there, over its state's shapes (`_planned_unplaced`), while its start's program compiles beside it;
        then place the state (`placed`) and report
        it. Returns the placed state, its shardings and whether the run is already complete, a
        checkpoint at its last step having been written, so there is nothing to do."""
        mesh, checkpoints, steps = self.device_mesh, self.checkpoints, plan.steps
        begun: Compiling | None = None
        if steps is not None and current > steps:
            raise ValueError(f"the run is at step {current}, past the {steps} asked for")
        bound = self._shape.examples
        if bound is not None and interval.consumed > bound:
            raise ValueError(f"the run consumed {interval.consumed} examples, past the {bound} asked for")
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
        complete = (plan.ended(current, interval.real_tokens, interval.consumed) and checkpoints is not None
                    and checkpoints.latest is not None and not pending)
        # The step the run reads from, as the pre-read's empty-stream check reads it, and the FLOP
        # count's refusal cleared before the prefetch worker counts its first batch.
        run.current = current
        self._refusal = None
        if not complete and (pending or not plan.ended(current, interval.real_tokens, interval.consumed)):
            partition = self._stream_partition()
            run.source = plan.dataset.train(partition)
            run.unworked = isinstance(run.source, ReadsInProcess) and run.source.in_process
            check_stream(run.source, mesh, self.microbatches, plan.dataset.batch, rows=self.rollout is None,
                         resumed=current if resumed and position is None and run.restarted is None else None)
            source = run.source
            stages = source.stages if isinstance(source, RampedStream | PhasedStream | StepPhases) else ()
            ramp = stages if isinstance(source, RampedStream) else ()
            # The steps a stage after the first begins at, each where an update does (`Ramp.stages`): a
            # ramp's and phases' of other batches here (`PhasedStream.stages`), a schedule's as each begins
            # (`_scheduled`); the step lays its window out there. Each stage's batch of a ramp is read whole:
            # no plan splits a ramp's.
            self._shape = dataclasses.replace(
                self._shape, ramp=ramp, split=1 if ramp else self._shape.split,
                stage_starts=frozenset(accumulate((after.records - stage.records) // stage.batch
                                                  for stage, after in pairwise(stages))))

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
                    if fresh is not None:
                        begun = self._compiled_fresh(run, fresh, first, placed_first)
            elif position is not None:
                # Its first batch read, so its worker has restored the stream or started it again
                # (`_prefetched`), which the run reports as it starts (`DataRestarted`).
                waited_read(run, interval, agree=True, peek=True)
        state, shardings = placed(begun)
        if int(state.step) != current:
            raise ValueError(f"the checkpoint of step {current} holds a state at step {int(state.step)}")
        check_branches(bool(plan.branches) or pending, plan.suites, settable=settable_rates(rated(state)))
        started = FitStarted(current, steps,
            checkpoints.source(current) if checkpoints is not None and position is not None else None,
            sum(leaf.size for leaf in jax.tree.leaves(state.variables["params"])), mesh.devices.size,
            jax.devices()[0].device_kind, jax.process_count(), dict(mesh.shape),
            split_share(state.variables),
            seed=self.seed, tokens=plan.tokens, examples=self._shape.examples,
            cadences=plan.recorded(self._config, self.accumulation, clocked=self._shape.examples is not None),
            groups=self._groups(state))
        self._report(started, current)
        if run.restarted is not None:
            self._report(DataRestarted(current, run.restarted), current)
            self._display.note(f"Data starts again at step {current}: its saved plan differs in "
                               f"{', '.join(run.restarted)}")
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

    def _compiled_fresh(self, run: FitRun, fresh: Start, first: Batch, placed: Batch) -> Compiling | None:
        """A fresh start's first step compiled over `placed`, its first batch's shapes, and its start's
        program begun beside it in a thread of its own (`start_compiling`), the key lowered unplaced as the
        start compiled ahead takes it: both once the whole state is checked, so a refused run compiles
        neither. None where the step is not compiled here (one compiled ahead, a host-master's, a profile
        guided run's), the start then compiling as it is placed."""
        if self._compiler.ahead is not None or self.host_master or pgle_runs():
            return None
        key = jax.ShapeDtypeStruct(fresh.key.shape, fresh.key.dtype)
        begun = start_compiling(fresh.build, fresh.initializer, key, fresh.abstract, fresh.shardings,
                                fresh.through, self.device_mesh)
        self.phase_passes.at(first)
        with run.startup.phase("compile", "compiling the step"), run.held():
            # As the loop hands its step the batch (`read_batch`): on the examples clock, marked whether
            # it closes its window.
            self._planned_unplaced(fresh, closes_window(split_fields(placed)[0], self._shape.examples))
        return begun

    def _whole_validated(self, state: TrainState, batch: Batch, split: int) -> Validation:
        """`Layout.validate` of the whole state the run's steps keep between them reading `batch`'s
        shapes (a marked batch, as the stream gives it) in `split` microbatches: `state` with the
        accumulation its window pools. The narrow copies are left out: the plan's first rung drops
        them where the step does not fit with them (`dew.training.planning.FitSearch`)."""
        micro = microbatch_shapes(split_fields(batch)[0], split)
        with self._traced_on(self.device_mesh) as schedule:
            _, _, whole, _ = prepared_for(self._stepping(), state, micro, schedule, split, narrow=False)
        return self._validated(whole)[1]

    def _fitting(self, state: TrainState, batch: Batch) -> Validation:
        """The report of the whole state the run's steps keep reading `batch` (`_whole_validated`) in
        the first of the splits its plan may take (`splits`) whose devices hold it, refused, before
        anything is placed, where none does: the plan picks among them once the state is placed."""
        refusals = []
        for split in splits(self._shape.rows, batch, self.device_mesh, self.microbatches, self._shape.split):
            try:
                return self._whole_validated(state, batch, split)
            except LayoutRefused as refused:
                refusals.append(refused)
        raise refusals[-1]

    def _prefetched(self, plan: FitPlan, run: FitRun, source: Iterator[Batch],
                    position: bytes | None) -> DevicePrefetchIterator:
        """`source`'s batches placed ahead of the step by the prefetch worker, which restores it to
        `position` first, once a pool's processes agree their streams read the same records."""
        assert run.partition is not None
        if position is None and run.restarted is not None and isinstance(
                source, StepPhases | RampedStream | PhasedStream):
            # Started again where its saved position was none of this run's: at the stage its step is in.
            position = restarted(source, None, None, run.current, run.restarted)
        elif position is not None and isinstance(source, PhasedStream):
            # A schedule's stage opens where the position stands, or where it starts again: the pool
            # agrees on the one it reads.
            position, differs = resumed_on(source, lambda: position, resume=self.data_resume,
                                           step=run.current)
            self._restarting(run, differs)

        def restore(stream: Checkpointable, saved: bytes) -> None:
            # On the worker that reads the stream, which it opens once (row 5577).
            self._restarting(run, restored_or_restarted(stream, saved, resume=self.data_resume,
                                                        step=run.current))
        if jax.process_count() > 1:
            # Each process opened its own stream over its own copy of the
            # corpus, and none reads a record before all agree it is one.
            agreed_same("training data", lambda: (
                f"{plan.dataset.batch} records a step of {stream_identity(source)}"))
        run.opened(source, self._shape.examples)
        run.marked = marked(run.partition, self._counted_batch)
        counted = run.marked if self.rollout is None else functools.partial(
            consumption_counted, clocked=self._shape.examples is not None, marks_rows=False)
        inputs = self.objective.inputs
        # What the stream states it hands each step, a rollout's batches aside, which make their own rows.
        produced = None if self.rollout is not None else Produced.of(plan.dataset, run.partition)

        def prepare(batch: Batch) -> Batch:
            # Every batch as its loader states it, before it is placed (`Produced`), and its token ids within
            # its model's table (`InputSpec.bounded`): an id past the table reads no row, and its gather
            # fills NaN.
            if produced is not None:
                produced.checked(batch)
            if inputs is not None:
                inputs.bounded(batch)
            return counted(batch)
        if run.pooled:
            run.emptied = Emptied(functools.partial(pooled_first, plan.dataset, run.partition), run.marked,
                                  functools.partial(shard_batch, self.device_mesh, reads=run.reads))
        return DevicePrefetchIterator(source, self.device_mesh, self._shape.prefetch, source_state=position,
                                      reads=run.reads, prepare=prepare, restore=restore)

    @staticmethod
    def _restarting(run: FitRun, differs: tuple[str, ...]) -> None:
        """Note a resume whose data starts again, its saved plan differing in `differs` (`DataRestarted`)."""
        if not differs:
            return
        run.restarted = differs
        _log.warning("the saved data position's plan differs from this run's in %s, so its data starts "
                     "again from the beginning of its order while its state resumes (data_resume='relaxed'; "
                     "'exact' refuses it)", ", ".join(differs))

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
        self._shape = dataclasses.replace(self._shape, stage_starts=self._shape.stage_starts | {run.current})
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
        counts = {"tokens": (real, before), "examples": (interval.consumed, interval.consumed_before)}
        logged = plan.falls(plan.log_every, current, counts)
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
            run.checkpoint_due = plan.falls(plan.checkpoint_every, current, counts)
        run.local_due = plan.falls(plan.local_every, current, counts)
        due = plan.due(current, counts)
        evaluation_due = bool(due) or plan.falls(plan.eval_every, current, counts)
        # A run's last step, its budget's or the step after which its stream runs out, is scored and saved
        # once, by `_wind_down`, with every suite: never here before a rarer suite has. Only a step that
        # evaluates or pauses could save first (a ranked save scores every suite), so only there is the stream
        # read ahead (`FitRun.ends_here`): no other step's save waits on a batch a stream has yet to make.
        ending = plan.finished(current, real, interval.consumed) or (
            (evaluation_due or plan.paused(current, real, interval.consumed)) and run.ends_here(interval))
        paused = not ending and plan.paused(current, real, interval.consumed)
        ranked = bool((plan.validation or plan.best or plan.stop) and run.checkpoint_due)
        if not ending and (evaluation_due or ranked):
            self._read_taken(run)
            with run.outside_window(), run.held(), region("evaluate"):
                # A checkpoint that ranks the steps reads every suite's score.
                names = None if ranked else due
                run.evaluation = self._evaluator().evaluation(
                    plan, state, shardings, self._compiler.links(self.device_mesh), run.training, names,
                    examples=self._consumed(interval))
                run.scored = frozenset(plan.suites if names is None else names)
                run.other += run.evaluation.elapsed_seconds
        scores = (
            {} if run.evaluation is None or run.evaluation.step != current else dict(run.evaluation.scores)
        )
        # Before the step's save, so a preemption inside the branch saves this step to run it again; at
        # the run's last step, after its final save (`_wind_down`). The points share one length and
        # decay, so those due together are one branch.
        crossed = run.committed_examples(interval, plan.example_points, state,
                                         (self._shape.split or 1) * self.accumulation)
        due = next((b for b in plan.branches if b.due(current, {**counts, "examples": crossed})), None)
        branch = None if due is None else due.pending(current, self.accumulation)
        if ending and branch is not None and checkpoints is not None:
            # Kept before the step's final save makes the run look complete (`Checkpoints.cooldown`).
            checkpoints.cooling(branch)
        if ending:
            run.cooling, branch = branch, None
        with contextlib.nullcontext() if branch is None else run.outside_window():
            state, cut = (state, False) if branch is None else self._cooled(plan, run, interval, state,
                                                                             shardings, position, branch)
        # Asked at every step on every process, as JAX's agreement needs, before the step's save, so a
        # stop's save is its one save, whole; the last step ends the run. A pause falls on the same step
        # on every process, so where it is reached none asks.
        assert run.notice is not None
        stopping = paused or cut or (not ending and run.notice.reached())
        if not ending:
            self._saves(plan, run, interval, state, position, scores, evaluation_due=evaluation_due,
                        full=stopping)
        if plan.observe:
            with run.held():
                plan.observe.at(current, state)
        if run.stopped:
            return True, state
        if stopping:
            self._take(plan, run, interval, state, unlogged=True)
            if checkpoints is not None and interval.resumable != current:
                self._read_taken(run)
                run.other += self._saved_checkpoint(
                    checkpoints, current, state, position, interval,
                    training_best=not plan.validation and not plan.best, run=run)
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
        (`scaled_rates`); score every suite on it as `cooldown/<split>` at the run's step, and discard
        it. The run's moving state waits on the host and its stream is closed meanwhile, so the devices
        hold one state and one prefetcher, as the run's plan sized them (`Trials`). Returns the run's
        state, and whether a preemption cut the branch short: then it stays pending beside the
        checkpoints (`Checkpoints.cooldown`), and a resume from this step runs it first."""
        (steps, shape), checkpoints = resolved(pending, self.accumulation), self.checkpoints
        if position is None:
            raise ValueError("a cooldown branch reads on from the run's data position, and this stream "
                             "keeps none")
        assert run.partition is not None and run.notice is not None
        began, at, held = time.perf_counter(), state.step, held_rates(rated(state))
        self._display.note(f"Cooldown branch from step {run.current}: {steps} steps")
        if checkpoints is not None:
            checkpoints.cooling(pending)
        moving, frozen = apart(state)
        restore, step_at = snapshot(moving), jax.device_put(np.asarray(int(at), at.dtype), at.sharding)
        reading, run.ahead = run.train is not None, None
        if run.train is not None:
            run.train.close()
        branched = FitRun(run.started, run.startup, current=run.current, partition=run.partition,
                          reads=run.reads, compiled=run.compiled, marked=run.marked)
        # The branch's own clock of committed tokens, from the run's, which a schedule's stages end at
        # as the run's do (`_scheduled`), read off the devices where its steps may have reached what it
        # reads next (`FitPlan.committed`); the stages it begins are its own, the run's kept as they were.
        counted = Interval(time.time(), None, tokens_before_log=interval.real_tokens)
        branched.clock = TokenClock(*counted.clocked(state.pending_tokens))
        starts = self._shape.stage_starts
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
                    read = read_batch(branched, counted, cooled, self.rollout, self.device_mesh, run.marked,
                                      self._shape.examples, agree=branched.agrees(first=not done),
                                      phase="cooldown read", branch=True, objective=self._step_objective)
                    if read is None:
                        break
                    (batch, device, _), read = read, None
                    self.phase_passes.at(batch)
                    step, _, cooled = self._compiled_for(run.compiled, cooled, device, branched,
                                                         reported=run.current)
                    factor = decay(shape, done // self.accumulation, steps // self.accumulation)
                    cooled, loss, aux, finite, _, committed = step(
                        rates_set(cooled, functools.partial(scaled_rates, held=held, factor=factor)), device)
                    counted_step(counted, branched, self.objective.row_tokens, batch, device, None, loss,
                                 finite, committed, aux)
                    done, branched.current = done + 1, branched.current + 1
            finally:
                self._shape = dataclasses.replace(self._shape, stage_starts=starts)
                if branched.train is not None:
                    branched.train.close()
        run.cooling = pending if cut else None
        if cut:
            self._display.note(f"Preempted inside the cooldown from step {run.current}, which a resume runs")
        elif done < steps:
            self._display.note(f"The training data ran out {done} steps into the cooldown from step "
                               f"{run.current}, which records no score")
        else:
            self._evaluator().evaluation(
                dataclasses.replace(plan, summaries=()), dataclasses.replace(cooled, step=step_at), shardings,
                self._compiler.links(self.device_mesh), {}, examples=self._consumed(interval, counted),
                cooled=True)
        if checkpoints is not None and not cut:
            checkpoints.cooling(None)
        released(apart(cooled)[0])
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
        model, _ = self._staging.restore(checkpoints, template, "best")
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
        run.taken = interval.take(run.current, loss, aux, accepted, copied(state),
                                  {} if self.rollout is None else dict(_rollout_metrics(self.rollout)),
                                  {**plan.dataset.statistics(), **interval.consumption(self._shape.examples)})

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
                training_best=not plan.validation and not plan.best, run=run,
                weights_only=not run.checkpoint_due and not run.stopped and not full
                and not (final and plan.checkpoint_final) and bool(winners)
                and all(rank.weights_only for rank in winners) and not run.ends_here(interval))
            run.last_checkpoint = time.perf_counter()
        if run.local_due and checkpoints is not None and run.last_local_saved != current:
            self._read_taken(run)
            run.other += self._saved_local_checkpoint(checkpoints, current, state, position, interval, run)
            run.last_local_saved = current

    def _wind_down(self, plan: FitPlan, run: FitRun, interval: Interval, state: TrainState,
                   shardings: Placement[TrainState], position, profiler: Profiler | None) -> TrainState:
        """`fit` after its last step: the stream closed, a preemption
        announced, the trace window stopped, the final validation of every suite the step has not scored
        (skipped on a preemption or a pause, which a resume continues) and the checkpoint of the state the run
        ends on, ranked by them: a budget's last step, a stream's that ran out and a
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
            run.close_window(profiler)
        interval.check_finite(current, self._display)
        if loss is not None:
            # The last step has to land before the wall time is read.
            loss.block_until_ready()
        scores: Mapping[str, float] = {}
        # A plateau or a stream that ran out ends a run whose last step may have evaluated some suites
        # already: only the others are scored now, joined to the scores that step reported.
        joined = run.evaluation if run.evaluation is not None and run.evaluation.step == current else None
        scored = run.scored if joined is not None else frozenset()
        if plan.evaluates and run.preempted is None and run.paused is None:
            missing = [name for name in plan.suites if name not in scored]
            if not plan.suites or missing:
                run.evaluation = self._evaluator().evaluation(
                    plan, state, shardings, self._compiler.links(self.device_mesh), run.training, missing,
                    joined=None if joined is None else joined.scores, examples=self._consumed(interval))
                run.other += run.evaluation.elapsed_seconds
            assert run.evaluation is not None
            scores = run.evaluation.scores
        # A token update may cross the pause and the budget at once, which still saves the whole state.
        self._saves(plan, run, interval, state, position, scores, evaluation_due=bool(scores), final=True,
                    full=plan.paused(current, run.clock.read, interval.consumed))
        if run.cooling is not None and run.preempted is None and run.paused is None:
            # The last step's branch, its main model scored and saved first, which a resume from the
            # step runs again where a preemption cuts it short.
            state, cut = self._cooled(plan, run, interval, state, shardings, position, run.cooling)
            if cut and checkpoints is not None and interval.resumable != current:
                # Whatever the final-save policy, the step the branch resumes from is written whole.
                run.other += self._saved_checkpoint(checkpoints, current, state, position, interval,
                                                    training_best=False, run=run)
            run.preempted = current if cut else None
        if plan.observe:
            plan.observe.last(current, state)
        if checkpoints is not None:
            checkpoints.wait()
        return state

    def _evaluator(self) -> Evaluator:
        """The run's evaluations (`Evaluator`), over what this trainer states of them."""
        return Evaluator(self.objective, self.layout, self.device_mesh, self.state_mesh, self.microbatches,
                         self.accumulation, self._display, self.tracker, self.checkpoints is not None)

    def _consumed(self, *intervals: Interval) -> int | None:
        """The examples a run on the examples clock had consumed, as its host counted them over `intervals`
        (`Interval.consumed`: the run's, and a cooldown branch's beyond it), the count its log records;
        None for a run counted in steps."""
        return None if self._shape.examples is None else sum(interval.consumed for interval in intervals)

    def _preflight(self, dataset: Loader, stop, suites: Mapping[str, EvalSuite], *,
                   preview: bool, stated: Dataset | Loader | Handed, loss: bool) -> None:
        """The refusals `fit` makes before it places anything: validation nothing reads, a rollout that
        checkpoints its state in a pool (`_rollout_admitted`), and a pipeline's batch, whose placement could
        take minutes or run out of memory before the error. Every other mesh's batch is checked with its
        stream (`check_stream`), a ramp's at each of its stages. The checkpoints learn why no record rebuilds
        a run of this trainer on the data it was `stated` (`Checkpoints.python_run`)."""
        if self.checkpoints is not None:
            given = {"a rollout (rollout=)": _rollout_admitted(self.rollout),
                     "a custom step (step=)": self.step, "a tracker (tracker=)": self.tracker,
                     "recording each step's draws (record=True)": self.record or None}
            self.checkpoints.python_run(unrecorded_parts(
                type(self.objective).__name__, stated, optimized=self._config is not None,
                seeded=self.seed is not None, mesh=self.mesh, given=given))
        for name, suite in suites.items():
            if loss and not episodic(suite.data) and all(metric.name != 'loss' for metric in suite.metrics):
                self.objective._check_validation_loss()
            if (stop is None or not isinstance(stop.metric, TrainingScalar)) and suite.every:
                read = self.objective.defines_validation_loss or self.checkpoints is not None or loss
                validation_read(name, suite, read=read, previewed=preview and self.tracker is not None)
        if self.device_mesh.shape.get(STAGE_AXIS, 1) > 1 and self.rollout is None:
            check_batch(dataset.batch, self.device_mesh, self.microbatches)

    def _step_objective(self, batch: Batch) -> Objective[Loss, Effects]:
        """The objective the step reading `batch` trains: the run's own, or its copy at the pass count of the
        phase `batch` was read in (`phase_passes`), which the step is compiled at, before a rollout draws
        for it (`Rollout`)."""
        return self.phase_passes.objective_for(self.phase_passes.count(batch))

    def _counted_batch(self, batch: Batch) -> Batch:
        """Prepare host counts before placement, never fetch a device batch to count it.

        The count is the capability: from the first batch whose work the model or objective cannot declare
        (`MissingFlopsCount`, as a media row's tower), this process counts no more, says why in the debug
        log, and trains on. Its batches keep their work columns, as unknown work (`uncounted`), so every
        process runs one step program and the step's global sum leaves train/mfu unreported pool-wide. One
        thread prepares batches, the prefetch worker or a rollout's caller, and only it writes the reason.
        A count asked of the objective directly stays strict. A phased run's batch is counted at its
        phase's model (`phase_passes`).
        """
        batch = consumption_counted(batch, clocked=self._shape.examples is not None)
        if self._refusal is None:
            # The model the batch's step runs: its phase's pass count (`phase_passes`), read off the batch,
            # since this thread prepares it ahead of the step that takes the count.
            objective = self._step_objective(batch)
            try:
                return prepare_counts(objective, batch, count_model=True)
            except MissingFlopsCount as missing:
                self._refusal = str(missing)
                _log.debug("train/mfu is not reported for %s: %s", type(self.objective).__name__, missing)
        counted = prepare_counts(self.objective, batch, count_model=False)
        return counted if self._refusal is None else uncounted(counted)

    def _compiled_for(self, compiled: dict[Hashable, tuple[CompiledStep, StepFlops | None]],
                      state: TrainState, batch: Batch, run: FitRun,
                      reported: int | None = None) -> tuple[CompiledStep, StepFlops | None, TrainState]:
        """Return the step compiled for this batch's abstract signature (`batch_signature`),
        compiling on first sight, and the state to step, a fresh copy where AutoPGLE profiled it
        on trials (`Trials`). A compile reads `run`'s pending log first: it holds the devices idle
        anyway, and its seconds are the next interval's.

        A ramped run reads a new shape at every stage, so `compiled` keeps one step per stage with the FLOPs
        measured for it. A batch whose dtype, fields or a node's static field change at the same shapes is
        another program, compiled as one: an executable refuses it. A step that begins a ramp's or a
        schedule's stage gets its state with the window a rejection left part filled dropped (`_staged`), the
        state an AutoPGLE trial steps on too. A compile reports at `reported`, a cooldown branch's at the
        run's step, else at the state's.
        """
        state = self._staged(state, run.current)
        signature = self.phase_passes.key(batch_signature(batch))
        if signature not in compiled and self.phase_passes.begins(compiled) and (
                not occupied(state.window)):
            # A phase of other passes begins at an update's end: its step is planned afresh, at the split
            # the run was given or the planner's for its own passes, over the state as a fresh state holds
            # it, no window laid out; one that begins inside an update keeps the run's, the one its open
            # window is laid out for.
            self._planned_afresh()
            if state.inner is None:
                state = freshly_held(state)
                # Its empty window binds no split: the new step's window is built over it.
                self._compiler.afresh = True
            if self._later is not None:
                # What the phase's plan off the loop compiled, which its plan here takes (`LaterPlans`).
                self._compiler.taken_executables(self._later.taken(self.phase_passes.current))
        # Trials here serve only explicitly enabled AutoPGLE, never frontier timing.
        trials = Trials(state, batch)
        if signature not in compiled:
            self._read_taken(run)
            # A process whose reader changed a field or a shape would compile
            # a program whose collectives its peers' never meet.
            agreed_same("training batch layout", lambda: batch_layout(batch))
            began = time.perf_counter()
            with region("compile"), run.held():
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
        run: FitRun,
        weights_only: bool = False,
    ) -> float:
        """Write one checkpoint with the interval's mean loss and `run`'s stop control, its stream
        placing nothing meanwhile (`FitRun.held`), report it and start the next interval.

        Returns the seconds it took, the reading of the interval's loss included: that read waits
        on the device, and the wait is time the steps did not have."""
        paused = time.perf_counter()
        self._display.status("writing a checkpoint")
        metadata = dict(scores or {})
        if interval.steps and interval.book is not None:
            metadata.setdefault('train/loss', float(jax.device_get(interval.book.loss) / interval.steps))
            if not ranking and training_best:
                ranking = (Ranking('train/loss', metadata['train/loss']),)
        envelope = self._checkpoint_envelope(state, position, interval, run, weights_only=weights_only)
        write = functools.partial(checkpoints.save, step, metrics=metadata, ranking=ranking,
                                  weights_only=weights_only, **envelope)
        with run.outside_window(), run.held():
            self._report(CheckpointRequested(checkpoints.directory), step, write)
        interval.saved(step, resumable=not weights_only)
        self._display.status("")
        return time.perf_counter() - paused

    def _checkpoint_envelope(self, state: TrainState, position: bytes | None, interval: Interval, run: FitRun,
                             *, weights_only: bool = False) -> Envelope:
        """What a checkpoint of `state` holds on every tier, each tier's save adding its own metadata: one
        envelope a persistent and a local save submit (row 4126)."""
        control, share = {**run.stop_control, **interval.token_control}, DataPartition.of(self.device_mesh)
        return {"state": self._staging.saved(state), "saved": position, "share": share, "control": control,
                "rung": self._compiler.checkpoint_rung(), "artifact": self._artifact(),
                "boxed": self._boxed_axes(),
                "rollout": None if weights_only else _rollout_saved(self.rollout)}

    def _saved_local_checkpoint(self, checkpoints: Checkpoints, step: int, state: TrainState,
                                position: bytes | None, interval: Interval, run: FitRun) -> float:
        """Write one checkpoint to the local directory with `run`'s stop control, its stream placing
        nothing meanwhile (`FitRun.held`), and report it.

        Returns the seconds it took. The local copy carries no metadata; it is the one a restarted
        node reads back, not the run's record."""
        paused = time.perf_counter()
        self._display.status("writing a local checkpoint")
        write = functools.partial(checkpoints.save_local, step,
                                  **self._checkpoint_envelope(state, position, interval, run))
        with run.outside_window(), run.held():
            self._report(CheckpointRequested(str(checkpoints.local_directory), local=True), step, write)
        self._display.status("")
        return time.perf_counter() - paused

    def _read_taken(self, run: FitRun, *, agree: bool = True, wait: bool = True) -> None:
        """Hand the log the loop took (`Taken`) to the run's writer, which reads it on the host and reports
        one logging interval off the step loop (`LogWriter`, `_written`), and raise a write's failure: of
        every write, waiting for each, where `wait`, as before a compile, an evaluation, a save or the end,
        which read the run's state anyway; else of those done, as each logging tick does, which never waits.

        The failures are agreed here, on the loop, at a point every process reaches alike, so a tracker that
        failed on rank zero, or a losses' streak, stops its peers here rather than at their next collective;
        a failed run's cleanup raises without agreeing, waiting only on writes that wait on no device, since
        its peers may never agree and its steps never end."""
        taken, run.taken = run.taken, None
        if taken is not None:
            run.writer.submit(functools.partial(self._read_log, run, taken),
                              functools.partial(self._written, run, taken), taken.ready)
        elif not (wait and run.writer.unchecked):
            return
        check = functools.partial(run.writer.raise_failure, wait=wait, ready_only=not agree)
        with region("log"):
            if agree:
                agreed("training reporting", check)
            else:
                check()

    def _read_log(self, run: FitRun, taken: Taken) -> tuple[Reading, float]:
        """`taken` on the host, its losses checked, and its interval's seconds, on the writer's reader
        thread (`LogWriter`), whose clock it moves."""
        read, elapsed, run.writer.since = taken.read(self._display, run.writer.since)
        return read, elapsed

    def _written(self, run: FitRun, taken: Taken, reading: tuple[Reading, float]) -> None:
        """Report `taken`'s interval's scalars (`logged_scalars`) on rank zero, on the writer's reporter
        thread (`_read_taken`); every other rank has nothing to report."""
        if jax.process_index() != 0:
            return
        read, elapsed = reading
        scalars = {**logged_scalars(read, taken.counts, elapsed, self.device_mesh.devices.size,
                                    rollout=self.rollout is not None, update_flops=run.update_flops),
                   **run.startup.scalars()}
        if run.unworked and elapsed > 0 and taken.counts.waited > elapsed / 2:
            # Said once: a per-share stream read in the process starves the step.
            run.unworked = False
            _input_log.warning("the step waits on its input for more than half its time, read by a per-share "
                               "stream in the training process: state its loading workers (.workers(n)), or "
                               "read it .elastic(), whose order any count of workers keeps")
        self._display.interval(taken.step, scalars)
        if self.tracker is not None:
            self.tracker.log(scalars, taken.step)

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
        error = reported_window(self._report, self._display.note, run.profiled, error)
        if error is None:
            def report_goodput() -> None:
                if jax.process_index() == 0:
                    wall = time.perf_counter() - run.started
                    # A run that logged no step reports its start here (`Startup.scalars`).
                    scalars = {**goodput(wall, run.first_step, run.other), **run.startup.scalars()}
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

        The record's type names the phase, so a rank that failed to report is heard about at the report the
        pool was making, not at the next collective. `after` is the work the record reports, run first inside
        the same agreement: a rank whose work fails after its own coordination rounds (a checkpoint's position
        gather, then its write) meets its
        peers at this report, rather than skip it while they wait there."""
        def send() -> None:
            if after is not None:
                after()
            if jax.process_index() == 0 and self.tracker is not None:
                self.tracker.artifact(value, step)

        agreed(type(value).__name__, send)

    # ------------------------------------------------------------------ Validation
    # ------------------------------------------------------------------
