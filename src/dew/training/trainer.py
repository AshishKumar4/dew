"""The trainer: mesh, compiled step, EMA, checkpoints, logging.

What is learned is the objective's business (`dew.objectives.base`). The
trainer materialises the objective's tree on the mesh, compiles one step over
the global batch, keeps the EMA copy on the optimizer's clock, and hands
effects to the capabilities it was given: a `Checkpoints` for disk, a
`Tracker` for numbers and artifacts. Constructing one opens nothing; the mesh,
the compiled step and the capabilities' resources come into being in `fit`.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from termcolor import colored

from dew.artifacts import agree_process_phase
from dew.checkpoints import Checkpoints
from dew.data.dataset import Checkpointable, RampedStream, rows_of
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Aux, Batch, Effects, Initializer, Loss, Mean, Metric, Objective, Step, select
from dew.telemetry import profile as telemetry_profile
from dew.telemetry.instrumentation import model_flops_utilization, step_flops
from dew.telemetry.records import CheckpointRequested, FitEnded, FitStarted, ProfileWindow, Record
from dew.training.distributed import (
    DevicePrefetchIterator,
    Layout,
    MeshSpec,
    Placement,
    batch_divisor,
    batch_shardings,
    build_mesh,
    shard_batch,
)
from dew.training.evaluation import Evaluation, evaluate
from dew.training.state import Accumulation, TrainState
from dew.training.tracker import Tracker
from dew.training.transaction import Transaction, compact_qk, with_ema

if TYPE_CHECKING:
    from dew.data import Dataset
    from dew.telemetry.profile import Profiler

# Consecutive non-finite losses that stop a run.
BAD_LOSS_STEPS = 5

StepFn = Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]
"""A compiled step's body: the state and the global batch in, the new state,
the loss and the objective's report out."""

CompiledStep = Callable[
    [TrainState, Batch],
    tuple[TrainState, jax.Array, Mapping[str, jax.Array], jax.Array, jax.Array]]
"""A call returns state, scalar loss, metrics, loss_finite, and accepted."""

Shapes = tuple[tuple[int, ...], ...]
"""A batch's leaf shapes in tree order, the key a compiled step is held
under. A fixed batch has one; a ramped batch has one per stage of its ramp,
each compiled on the first step that reads it."""


def batch_shapes(batch: Batch) -> Shapes:
    return tuple(np.shape(leaf) for leaf in jax.tree.leaves(batch))


class Rollout(Protocol):
    """A host-side batch producer the trainer runs before the compiled step.

    Sampling is effectful and untraceable, so it lives outside `jit`: the
    trainer calls the rollout with the state, the prefetched batch and a key
    folded from the run key and the step, then reshards what comes back with
    `shard_batch`. The returned batch must hold arrays in fixed shapes, so
    the step still compiles once per run."""

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...


@dataclasses.dataclass(frozen=True)
class Profile:
    """One profiler window per fit: `steps` steps traced into `directory`
    after `warmup` steps have run, so the trace holds the loop and not the
    compile."""
    directory: str
    steps: int
    warmup: int = 2


Book = tuple[jax.Array, jax.Array, jax.Array]
"""The loop's device-side counters: the interval's summed loss, the current
streak of non-finite losses, and the longest streak since the last check."""


def fresh_book() -> Book:
    return jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)


@jax.jit
def bookkeep(book: Book, loss: jax.Array, finite: jax.Array) -> Book:
    """The counters after one step, in one dispatch.

    As five eager ops (the cast, the add, the where, the add, the maximum)
    each dispatched an executable of its own, 176 us a step on an i9-12900K
    against 37 us for this one call, measured over 2000 steps on the CPU
    backend with the result blocked on at the end.
    """
    interval_loss, bad_run, worst_bad_run = book
    bad_run = jnp.where(finite, 0, bad_run + 1)
    dtype = jnp.promote_types(loss.dtype, jnp.float32)
    return interval_loss + loss.astype(dtype), bad_run, jnp.maximum(worst_bad_run, bad_run)


def goodput(wall: float, first_step: float | None, other: float) -> dict[str, float]:
    """The two goodput numbers of MaxText's report dew can compute locally.

    `first_step` is the time from the start of `fit` to the first step's
    result: the placement or restore, the first batch, the compile and the
    step itself, or None when no step ran. `other` is the time spent
    outside steps after that (evaluations, checkpoint writes and the wait
    for them at the end). The step fraction is what is left of `wall`,
    which counts a step's own data stall as step time, as MaxText's
    start-to-start step time does.
    """
    numbers = {}
    if first_step is not None:
        numbers["goodput/time_to_first_step_s"] = first_step
    steps = wall - (wall if first_step is None else first_step) - other
    numbers["goodput/step_fraction"] = max(steps, 0.0) / wall if wall > 0 else 0.0
    return numbers


class Trainer(Generic[Loss, Effects]):
    """Runs an `Objective`: gradients, sharding, EMA, checkpoints, logging."""

    def __init__(
        self,
        objective: Objective[Loss, Effects],
        optimizer: optax.GradientTransformation,
        *,
        key: jax.Array,
        mesh: MeshSpec = MeshSpec(),
        layout: Layout = Layout(),
        accumulation: int = 1,
        dynamic_scale: bool = False,
        checkpoints: Checkpoints | None = None,
        tracker: Tracker | None = None,
        step: Callable[[Objective[Loss, Effects], optax.GradientTransformation], StepFn] | None = None,
        rollout: Rollout | None = None,
        profile: Profile | None = None,
    ):
        """Accumulate accepted microbatches before an optimizer commit.

        A custom step owns accepted/update clocks, scaler, EMA and mutable
        writes. The compiled wrapper owns attempted-work advancement. Host
        rollout collection runs once per consumed batch, outside replay.
        """
        if accumulation < 1:
            raise ValueError(f"accumulation must be at least 1, got {accumulation}")
        if step is not None and "params" in layout.host:
            raise ValueError("Parameter-streamed training requires the trainer objective transaction; custom steps own their execution")
        self.objective = objective
        self.optimizer = optimizer
        self.key = key
        self.mesh = mesh
        self.layout = layout
        self.accumulation = accumulation
        self.dynamic_scale = dynamic_scale
        self.checkpoints = checkpoints
        self.tracker = tracker
        self.step = step
        self.rollout = rollout
        self.profile = profile
        # Measured off the step `compile` last compiled, which for a ramped
        # run is the stage it was called for; `fit` keeps one per stage.
        self.flops_per_step = None

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def initial_state(self, initializer: Initializer | None = None,
                      key: jax.Array | None = None) -> TrainState:
        """The state a fresh run starts from. Pure, so `fit` traces it once
        for its shapes and once, sharded, for its values.

        Both inputs are the run's own by default, and `place` passes them
        explicitly so what it compiles takes them as arguments: a held
        checkpoint then reaches the device as an argument instead of as a
        constant embedded in the executable. Passing None means resolve the
        configured input, which is what a no-argument call does. This is the
        one state implementation, so a subclass overrides it here and every
        path sees the override.
        """
        initializer = self.objective.initializer if initializer is None else initializer
        key = self.key if key is None else key
        init_key, run_key = jax.random.split(key)
        params = nn.unbox(initializer(init_key))
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
            params=params,
            opt_state=self.optimizer.init(params["params"]),
            ema=None if ema is None else select(params, ema.select),
            key=run_key,
        )

    @functools.cached_property
    def device_mesh(self) -> Mesh:
        """The mesh `MeshSpec` describes over this process pool's devices,
        built on first use."""
        return build_mesh(self.mesh)

    @property
    def host_master(self) -> bool:
        return "params" in self.layout.host

    @functools.cached_property
    def state_mesh(self) -> Mesh:
        if not self.host_master:
            return self.device_mesh
        from dew.training.host import companion_mesh
        return companion_mesh(self.device_mesh)

    def shardings(self, state: TrainState) -> Placement:
        """Parameter gradients follow parameters; replay records follow batches;
        the layout's host-resident fields sit in pinned host memory."""
        mesh = self.state_mesh
        placed = self.layout.shardings(mesh, dataclasses.replace(state, accumulation=None))
        placed = dataclasses.replace(placed, **{
            field: jax.tree.map(lambda s: s.with_memory_kind("pinned_host"), getattr(placed, field))
            for field in (() if self.host_master else self.layout.host)})
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
            gradient=None if accumulation.gradient is None else placed.params["params"],
            batches=buffered_shardings(accumulation.batches, True),
            variables=buffered_shardings(accumulation.variables, False))
        return dataclasses.replace(placed, accumulation=pending)

    def _fetched(self, state: TrainState, shardings: Placement) -> TrainState:
        """`state` with the layout's host-resident fields brought to the
        device, where a step or an evaluation reads them."""
        if self.host_master:
            return state
        return dataclasses.replace(state, **{
            field: jax.device_put(getattr(state, field), jax.tree.map(
                lambda s: s.with_memory_kind("device"), getattr(shardings, field)))
            for field in self.layout.host})

    def place(self) -> tuple[TrainState, Placement, bytes | None]:
        """The state itself, fresh or restored, on the mesh, with its shardings
        and the data position a resume continues from."""
        # Resolved once: the shapes and the values are then the same inputs
        # through the same overridable method, and the objective is asked for
        # what it holds exactly once.
        initializer, key = self.objective.initializer, self.key
        if self.host_master:
            from dew.training.host import transfer
            replicated = NamedSharding(self.state_mesh, P())
            initializer, key = transfer(
                (initializer, key), jax.tree.map(lambda _: replicated, (initializer, key)))
        abstract = jax.eval_shape(self.initial_state, initializer, key)
        shardings = self.shardings(abstract)
        self.layout.check(abstract.params, shardings.params, self.device_mesh)
        checkpoints = self.checkpoints
        resume = None if checkpoints is None else checkpoints.latest
        if checkpoints is None or resume is None:
            state = jax.jit(self.initial_state, out_shardings=shardings)(initializer, key)
            return state, shardings, None
        abstract = dataclasses.replace(abstract, accumulation=checkpoints.accumulation_template(resume))
        shardings = self.shardings(abstract)
        template = jax.tree.map(
            lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
            abstract, shardings)
        state, position = checkpoints.restore(template, resume)
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        print(f"Resumed from step {resume} in {checkpoints.source(resume)}")
        return state, shardings, position

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def _loss_shape(self, state: TrainState, batch: Batch):
        info = Step(state.microstep, jax.random.fold_in(state.key, state.step),
                    with_ema(state.params, state.ema))
        return jax.eval_shape(self.objective.loss, state.params, batch, info)

    def _initialize_accumulation(self, state: TrainState, batch: Batch, shapes, *, shape_only=False):
        if self.accumulation == 1 or self.step is not None or state.accumulation is not None:
            return state
        stats, aux = shapes
        shared = isinstance(stats, (Mean, jax.ShapeDtypeStruct))
        mean_dtype = jnp.result_type(jnp.float32, *(x.dtype for x in jax.tree.leaves(stats)))
        slots = self.accumulation - 1
        shape = lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
        trainable = jax.tree.map(shape, state.params["params"])
        records = jax.tree.map(shape, batch)
        mutable = (None if shared or aux.variables is None else
                   jax.tree.map(shape, {name: state.params[name] for name in aux.variables}))

        def zeros(leaf):
            dtype = jnp.promote_types(leaf.dtype, jnp.float32) if jnp.issubdtype(leaf.dtype, jnp.inexact) else leaf.dtype
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


    def _default_step(self, shapes):
        return Transaction(self.objective, self.optimizer, self.accumulation, shapes).step()

    def compile(self, state: TrainState, batch: Batch) -> CompiledStep:
        """Compile a transaction over state and one already-produced global batch.

        Retained records and asynchronous checkpoints can own old array leaves.
        The call therefore does not donate input state or batch buffers.
        """
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        if (state.scale is not None) != self.dynamic_scale:
            raise ValueError("checkpoint dynamic-scaler configuration differs from this trainer")
        if self.host_master:
            return self._compile_host(state, batch)
        mesh = self.device_mesh
        with jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches):
            shapes = None if self.step is not None else self._loss_shape(state, batch)
            prepared = self._initialize_accumulation(state, batch, shapes, shape_only=True)
            body = self.step(self.objective, self.optimizer) if self.step is not None else self._default_step(shapes)
            shardings = self.shardings(prepared)
            replicated = NamedSharding(mesh, P())

            def step(current, batch):
                # The body sees every field on the device; the out shardings
                # return the host-resident ones to pinned host memory.
                result, loss, aux = body(self._fetched(current, shardings), batch)
                return (dataclasses.replace(result, step=current.step + 1), loss, aux.metrics,
                        jnp.isfinite(loss), result.microstep > current.microstep)

            jitted = jax.jit(step, in_shardings=(shardings, batch_shardings(mesh, batch)),
                             out_shardings=(shardings, replicated, replicated, replicated, replicated))
            prepared = jax.tree.map(
                lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), prepared, shardings)
            self.flops_per_step = step_flops(jitted, prepared, batch)

        def run(current, batch):
            with jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches):
                if current.accumulation is None and self.accumulation > 1 and self.step is None:
                    current = self._initialize_accumulation(current, batch, shapes)
                    current = jax.device_put(current, shardings)
                return jitted(current, batch)
        return run

    def _compile_host(self, state: TrainState, batch: Batch) -> CompiledStep:
        from dew.training.execution import HostExecution
        from dew.training.host import transfer
        cpu = self.state_mesh
        execution = HostExecution(self.objective, self.layout, self.device_mesh, cpu)
        cpu_batch = transfer(batch, batch_shardings(cpu, batch))
        with jax.set_mesh(cpu), pipeline_microbatches(self.mesh.microbatches):
            shapes = self._loss_shape(state, cpu_batch)
            prepared = self._initialize_accumulation(state, cpu_batch, shapes, shape_only=True)
            placement = self.shardings(prepared)
            transaction = Transaction(self.objective, self.optimizer, self.accumulation, shapes)
            body = transaction.step(realize=execution.realize, host=True)

        def run(current, batch):
            batch = transfer(batch, batch_shardings(cpu, batch))
            with jax.set_mesh(cpu), pipeline_microbatches(self.mesh.microbatches):
                if current.accumulation is None and self.accumulation > 1:
                    current = self._initialize_accumulation(current, batch, shapes)
                result, loss, aux = body(current, batch)
                result = dataclasses.replace(result, step=current.step + 1)
                result = jax.device_put(result, placement)
                return (result, loss, aux.metrics, jnp.isfinite(loss),
                        result.microstep > current.microstep)
        return run

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def fit(self, data: Dataset, *, steps: int, log_every: int = 100,
            eval_every: int | None = None, checkpoint_every: int | None = None,
            metrics: Sequence[Metric] = (), preview: bool = False) -> TrainState:
        """Train to `steps` total steps, resuming from the checkpoints' latest
        step when the directory holds one.

        Every `log_every` steps the tracker receives the loss, the objective's
        metrics and the throughput. Every `eval_every` steps, and at the end,
        the validation split is scored: the objective's artifacts go to the
        tracker and to `metrics`, whose reductions are logged as `val/<name>`.
        Every `checkpoint_every` steps, and at the end, the state and the data
        position are written; every `checkpoints.local_every` steps they are
        written to the local directory as well.
        Previews are generated only when `preview=True` and a tracker receives
        them; scalar reporting never triggers preview work.
        """
        preview = preview and self.tracker is not None
        started = time.perf_counter()
        profile, checkpoints = self.profile, self.checkpoints
        source = train = None
        tracing, traced = False, 0
        loss = None
        other = 0.0
        current = 0
        first_step = None
        process_zero = jax.process_index() == 0
        # A configured window must own the capture: refuse before the dataset
        # or the mesh do any work when another profiler already holds it, so
        # neither trace is silently dropped or cut short. A configured window
        # also resolves its optional dependency up front rather than after
        # warmup steps have run.
        profiler = None
        if profile is not None:
            # Every rank either leaves this block owning a stopped Profiler or
            # raises together; a rank that proceeds alone would deadlock its
            # peers at the first collective of training.
            error = None
            try:
                if telemetry_profile.active_profile() is not None:
                    raise ValueError(
                        "Trainer.profile cannot schedule a window while an explicit "
                        "dew.profile capture is active; drop one or stop the outer "
                        "profiler before fitting")
                telemetry_profile.require_profile_support()
                profiler = telemetry_profile.profile(profile.directory)
            except BaseException as failure:
                error = failure
            agree_process_phase(error, phase="profiling window setup")
        outer = telemetry_profile.active_profile()
        # One boundary lookup at setup: the prefetch worker and the step
        # scopes share whichever profiler owns the capture.
        tracer = profiler if profiler is not None else outer
        try:
            mesh = self.device_mesh

            state, shardings, position = self.place()
            current = int(state.step)
            self._report(FitStarted(current, steps,
                checkpoints.source(current) if checkpoints is not None and position is not None else None,
                sum(leaf.size for leaf in jax.tree.leaves(state.params["params"])), mesh.devices.size,
                jax.devices()[0].device_kind, jax.process_count(), dict(mesh.shape)), current)
            if current > steps:
                raise ValueError(f"the run is at step {current}, past the {steps} asked for")

            if checkpoint_every and checkpoints is None:
                raise ValueError(
                    "checkpoint_every asks for checkpoints and this trainer has no "
                    "checkpointer; pass Checkpoints(directory) to write any")
            if current == steps and checkpoints is not None and checkpoints.latest is not None:
                return state
            local_every = None if checkpoints is None else checkpoints.local_every
            compiled: dict[Shapes, tuple[CompiledStep, float | None]] = {}
            # Rebound once the step is compiled, so the first tick measures steps,
            # not the compile.
            last_log_time = time.time()
            last_saved = current if checkpoints is not None and checkpoints.latest is not None else None
            interval_steps = 0
            steps_since_log = 0
            # Summed per step rather than taken off the dataset's batch, so a
            # ramped interval reports the records it read and their FLOPs.
            interval_samples = 0
            interval_flops: float | None = 0.0
            # Seconds spent sampling this interval, logged under
            # train/rollout_seconds when a rollout is set.
            rollout_seconds = 0.0
            # The interval's loss and both bad-loss counters live on device, so the
            # loop never blocks on a result, and move together in one dispatch.
            # `worst_bad_run` remembers the longest streak of non-finite losses
            # seen since the last host check; the host check reads it to decide
            # whether to stop.
            book = fresh_book()
            seen = 0
            first_step = None

            if current < steps:
                source = data.train()
                if (checkpoint_every or local_every) and not isinstance(source, Checkpointable):
                    raise ValueError(
                        f"checkpoint_every needs a training stream with get_state and "
                        f"set_state, and {type(source).__name__} lacks one; a checkpoint "
                        f"written without the data position would replay the data on "
                        f"resume. Train it with checkpoint_every=None "
                        f"(--trainer.checkpoint-every None)")
                if isinstance(source, RampedStream):
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
                train = DevicePrefetchIterator(source, mesh, source_state=position,
                                               profiler=tracer)
                source = None  # Lifetime transferred to the prefetch worker.

            error = None
            try:
                if process_zero:
                    print(f"Training from step {current} to {steps} on "
                          f"{dict(mesh.shape)} ({jax.process_count()} process(es))")
            except BaseException as failure:
                error = failure
            agree_process_phase(error, phase="training announcement")
            while current < steps:
                assert train is not None
                # The window's capture opens before this iteration's first
                # read, so the step row records the read it waits on rather
                # than a compile that ran before capture began.
                if (profiler is not None and profile is not None
                        and not tracing and traced == 0
                        and seen >= profile.warmup):
                    error = None
                    try:
                        profiler.start()
                    except BaseException as failure:
                        error = failure
                    else:
                        tracing = True
                    try:
                        agree_process_phase(error, phase="profiling window start")
                    except BaseException as primary:
                        # A peer failed while this capture did start: closing it
                        # here keeps a live trace from outliving the aborted run.
                        if tracing:
                            tracing = False
                            try:
                                profiler.stop()
                            except BaseException as failure:
                                primary.add_note(f"Profiler stop failed: {failure!r}")
                        raise
                capturing = tracer is not None and tracer.running
                step_scope = (jax.profiler.StepTraceAnnotation("train", step_num=current)
                              if capturing else None)
                if step_scope is not None:
                    step_scope.__enter__()
                try:
                    annotation = None
                    if capturing:
                        annotation = jax.profiler.TraceAnnotation("input.wait")
                        annotation.__enter__()
                    try:
                        batch = next(train)
                    finally:
                        if annotation is not None:
                            annotation.__exit__(None, None, None)
                    if self.rollout is not None:
                        # Host-side and untraceable: sampling, scoring, advantages.
                        # The key folds the step key once more, keeping the
                        # rollout's draws off the step's stream; both are
                        # checkpointed, so a resumed run samples forward. Fixed
                        # shapes mean the compile below traces once.
                        began = time.perf_counter()
                        key = jax.random.fold_in(
                            jax.random.fold_in(state.key, state.step), 1)
                        batch = shard_batch(mesh, self.rollout(state, batch, key))
                        rollout_seconds += time.perf_counter() - began
                    shapes = batch_shapes(batch)
                    if shapes not in compiled:
                        annotation = None
                        if capturing:
                            annotation = jax.profiler.TraceAnnotation("compile")
                            annotation.__enter__()
                        try:
                            compiled[shapes] = (self.compile(state, batch), self.flops_per_step)
                        finally:
                            if annotation is not None:
                                annotation.__exit__(None, None, None)
                        if len(compiled) == 1:
                            last_log_time = time.time()
                    train_step, measured_flops = compiled[shapes]

                    annotation = None
                    if capturing:
                        annotation = jax.profiler.TraceAnnotation("train.step")
                        annotation.__enter__()
                    try:
                        state, loss, aux, finite, accepted = train_step(state, batch)
                    finally:
                        if annotation is not None:
                            annotation.__exit__(None, None, None)
                    position = train.source_state
                    current += 1
                    seen += 1
                    steps_since_log += 1
                    interval_steps += 1
                    interval_samples += rows_of(batch)
                    interval_flops = (None if interval_flops is None or measured_flops is None
                                      else interval_flops + measured_flops)
                    book = bookkeep(book, loss, finite)
                    if first_step is None:
                        loss.block_until_ready()
                        first_step = time.perf_counter() - started

                    if current % log_every == 0:
                        interval_loss, _, worst_bad_run = book
                        self._check_finite(worst_bad_run, current)
                        book = (interval_loss, book[1], jnp.zeros((), jnp.int32))
                        error = None
                        annotation = None
                        if capturing:
                            annotation = jax.profiler.TraceAnnotation("log")
                            annotation.__enter__()
                        try:
                            try:
                                if process_zero:
                                    # The interval's numbers need the loss on the host, so
                                    # this is where the loop waits on the device.
                                    loss.block_until_ready()
                                    now = time.time()
                                    scalars = {"train/loss": float(loss),
                                               **{f"train/{k}": float(v) for k, v in aux.items()},
                                               **self._throughput(now - last_log_time, steps_since_log,
                                                                  interval_samples, interval_flops)}
                                    scalars["train/accepted"] = float(accepted)
                                    if state.scale is not None:
                                        scalars["train/loss_scale"] = float(state.scale.scale)
                                    if self.rollout is not None:
                                        scalars["train/rollout_seconds"] = rollout_seconds
                                    print(f"step {current}: loss {scalars['train/loss']:.4f}")
                                    if self.tracker is not None:
                                        self.tracker.log(scalars, current)
                                    last_log_time, steps_since_log, rollout_seconds = now, 0, 0.0
                                    interval_samples, interval_flops = 0, 0.0

                            except BaseException as failure:
                                error = failure
                            agree_process_phase(error, phase="training reporting")
                        finally:
                            if annotation is not None:
                                annotation.__exit__(None, None, None)

                    if eval_every and current % eval_every == 0 and current < steps:
                        paused = time.perf_counter()
                        annotation = None
                        if capturing:
                            annotation = jax.profiler.TraceAnnotation("evaluate")
                            annotation.__enter__()
                        try:
                            self._evaluate(state, shardings, data, metrics, preview, mesh)
                        finally:
                            if annotation is not None:
                                annotation.__exit__(None, None, None)
                        other += time.perf_counter() - paused

                    # On its own clock, not the logging one: nested inside the log
                    # tick, a cadence that did not divide log_every never fired at all.
                    if (checkpoint_every and checkpoints is not None
                            and current % checkpoint_every == 0 and current < steps):
                        paused = time.perf_counter()
                        checkpoints.save(current, state, position,
                                         {"loss": float(book[0] / interval_steps)})
                        self._report(CheckpointRequested(checkpoints.directory), current)
                        other += time.perf_counter() - paused
                        last_saved = current
                        book = (jnp.zeros_like(book[0]), book[1], book[2])
                        interval_steps = 0
                    if (local_every and checkpoints is not None
                            and current % local_every == 0 and current < steps):
                        paused = time.perf_counter()
                        checkpoints.save_local(current, state, position)
                        self._report(CheckpointRequested(str(checkpoints.local_directory), local=True), current)
                        other += time.perf_counter() - paused
                finally:
                    if step_scope is not None:
                        step_scope.__exit__(None, None, None)
                # The step row is complete once its scope exits; closing the
                # window here keeps the last iteration inside the capture.
                if tracing and profile is not None:
                    traced += 1
                    if traced == profile.steps:
                        tracing = False
                        assert profiler is not None
                        self._stop_trace(traced, loss, profile, profiler, step=current)


            paused = time.perf_counter()
            if train is not None:
                train.close()
                train = None
            other += time.perf_counter() - paused
            if tracing and profile is not None:
                tracing = False
                # The window outlived the run, and a trace left running takes the
                # next one down with it.
                assert profiler is not None
                self._stop_trace(traced, loss, profile, profiler, step=current)
            interval_loss, _, worst_bad_run = book
            self._check_finite(worst_bad_run, current)
            if loss is not None:
                # The last step has to land before the wall time is read.
                loss.block_until_ready()
            paused = time.perf_counter()
            if eval_every:
                self._evaluate(state, shardings, data, metrics, preview, mesh)
            if checkpoints is not None and last_saved != current:
                # The in-loop saves are conditional, so the state the run ends on
                # may never have been written. It goes out under its real step,
                # because a step-0 checkpoint holding the final weights would make
                # a resume restart the schedule from the beginning.
                checkpoints.save(
                    current, state, position,
                    {"loss": float(interval_loss / interval_steps)} if interval_steps else None)
                self._report(CheckpointRequested(checkpoints.directory), current)
            other += time.perf_counter() - paused
        finally:
            paused = time.perf_counter()
            primary = sys.exception()
            error = primary
            close = train.close if train is not None else getattr(source, "close", None)
            stop_trace = None
            if tracing and profile is not None:
                tracing = False
                assert profiler is not None
                stop_trace = lambda: self._stop_trace(traced, loss, profile, profiler, step=current)
            for label, cleanup in (
                ("Training iterator", close),
                ("Profiler", stop_trace),
                ("Checkpoint wait", None if checkpoints is None else checkpoints.wait),
            ):
                if cleanup is not None:
                    try:
                        cleanup()
                    except BaseException as failure:
                        if error is None:
                            error = failure
                        else:
                            error.add_note(f"{label} cleanup failed: {failure!r}")
            source = train = close = cleanup = None
            other += time.perf_counter() - paused
            try:
                agree_process_phase(error, phase="fit cleanup")
            except BaseException as failure:
                if error is None:
                    error = failure
            if error is None:
                try:
                    if process_zero:
                        scalars = goodput(time.perf_counter() - started, first_step, other)
                        print(f"Goodput: first step after {scalars.get('goodput/time_to_first_step_s', 0.0):.2f} s, "
                              f"{scalars['goodput/step_fraction']:.1%} of the wall time in steps")
                        if self.tracker is not None:
                            self.tracker.log(scalars, current)
                except BaseException as failure:
                    error = failure
                try:
                    agree_process_phase(error, phase="goodput reporting")
                except BaseException as failure:
                    error = failure
            try:
                self._report(FitEnded.outcome(time.perf_counter() - started, error), current)
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f'Reporting fit outcome failed: {failure!r}')
            if primary is None and error is not None:
                raise error
        return state

    def _report(self, value: Record, step: int) -> None:
        error = None
        if jax.process_index() == 0 and self.tracker is not None:
            try:
                self.tracker.artifact(value, step)
            except BaseException as failure:
                error = failure
        agree_process_phase(error, phase=type(value).__name__)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _evaluate(self, state: TrainState, shardings: Placement, data: Dataset,
                  metrics: Sequence[Metric], preview: bool, mesh) -> None:
        params = state.params
        averaged = with_ema(state.params, self._fetched(state, shardings).ema)
        key = state.key
        if self.host_master:
            from dew.training.execution import HostExecution
            execution = HostExecution(self.objective, self.layout, self.device_mesh, self.state_mesh)
            with jax.set_mesh(self.state_mesh):
                params, averaged = execution.snapshot(params), execution.snapshot(averaged)
            key = execution.on_accelerator(key)
        assert params is not None, "evaluation always has model variables"
        self._report_evaluation(evaluate(
            self.objective, params, data.val, metrics=metrics, key=key,
            step=state.step, schedule_step=state.microstep,
            averaged=averaged,
            preview=preview, mesh=mesh))

    def _report_evaluation(self, result: Evaluation) -> None:
        error = None
        if jax.process_index() == 0:
            try:
                print(f"Evaluation {result.split} at step {result.step}: "
                      f"{result.coordinated_batches} coordinated batches, {result.records} records, "
                      f"uneven_shards={result.uneven_shards}, event_key={result.event_key}: {result.scores}")
                if self.tracker is not None:
                    for artifact in result.previews:
                        self.tracker.artifact(artifact, result.step)
                    scalars = result.scalars
                    if scalars:
                        self.tracker.log(scalars, result.step)
            except BaseException as failure:
                error = failure
        agree_process_phase(error, phase="evaluation reporting")


    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _stop_trace(self, traced: int, loss, profile: Profile,
                    profiler: Profiler, *, step: int) -> None:
        """Stop the window's owned capture before reporting it on process zero.

        The core Profiler drains the backend and exports the native reports on
        `stop`; the loss's block only orders the primary's failure ahead of
        the profiler's own drain."""
        error = None
        try:
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
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="profile stop")
        self._report(ProfileWindow(profile.directory, traced), step)
        error = None
        try:
            if jax.process_index() == 0:
                print(f"Wrote profile for {traced} steps to {profile.directory}")
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="profile announcement")

    def _check_finite(self, worst_bad_run, step: int):
        """Raise RuntimeError once the loss has been non-finite for BAD_LOSS_STEPS steps.

        Deferred to the logging cadence so the step loop never synchronises;
        detection is late by at most that many steps, never missed.
        """
        streak = int(worst_bad_run)
        if streak >= BAD_LOSS_STEPS:
            raise RuntimeError(
                f"Loss has been non-finite for {streak} consecutive steps "
                f"ending near step {step}, stopping")
        if streak:
            print(colored(f"Non-finite loss for {streak} step(s) before {step}", 'red'))

    def _throughput(self, elapsed: float, steps: int, samples: int,
                    flops: float | None) -> dict[str, float]:
        """The interval's rates from the records it read and the FLOPs the
        compiler measured for each step's own shape; `flops` is None when a
        shape's measurement was unavailable. An interval that spans a stage
        boundary carries that stage's compile in its wall time, where MaxText
        hides the performance metrics during a ramp
        (`common/metric_logger.py:166-194`).
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
