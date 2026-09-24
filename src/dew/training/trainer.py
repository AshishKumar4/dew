"""The trainer: mesh, compiled step, EMA, checkpoints, logging.

What is learned is the objective's business (`dew.objectives.base`). The
trainer materialises the objective's tree on the mesh, compiles one step over
the global batch, and keeps the EMA copy on the optimizer's clock. Effects go
to the capabilities it was given: a `Checkpoints` for disk, a `Tracker` for
numbers and artifacts.

Constructing one opens nothing; the mesh, the compiled step and the
capabilities' resources come into being in `fit`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from termcolor import colored

from dew.artifacts import agree_process_phase, agreed
from dew.checkpoints import Checkpoints
from dew.data.dataset import Checkpointable, Closeable, RampedStream, rows_of
from dew.nn.kernels.generation import device_generation
from dew.nn.backbones.causal_transformer import REMAT_POLICIES, CausalTransformer
from dew.nn.sharding import STAGE_AXIS, Schedule, pipeline_microbatches
from dew.objectives.base import (
    FROZEN,
    Aux,
    Batch,
    Effects,
    Initializer,
    Loss,
    Mean,
    Metric,
    Objective,
    Step,
    select,
)
from dew.telemetry import profile as telemetry_profile
from dew.telemetry.devices import TRITON_GEMM_OFF_GENERATIONS, xla_flag
from dew.telemetry.instrumentation import compiled_flops, model_flops_utilization
from dew.telemetry.profile import region
from dew.telemetry.records import (
    CheckpointRequested,
    FitEnded,
    FitStarted,
    ProfileWindow as ProfileWindowRecord,
    Record,
)
from dew.training.distributed import (
    DevicePrefetchIterator,
    Layout,
    MeshSpec,
    Placement,
    batch_divisor,
    batch_shardings,
    build_mesh,
    data_partition,
    shard_batch,
)
from dew.training.evaluation import Evaluation, evaluate
from dew.training.state import Accumulation, TrainState
from dew.training.tracker import Tracker
from dew.training.transaction import Transaction, compact_qk, with_ema

if TYPE_CHECKING:
    from dew.config import TrainerConfig
    from dew.data import Dataset
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

ObjectiveLoss = TypeVar("ObjectiveLoss")
ObjectiveEffects = TypeVar("ObjectiveEffects")
"""The two parameters of the objective `Trainer.from_config` is handed.

A classmethod cannot solve the class's own `Loss` and `Effects` from an
argument, since an unparameterized `Trainer.from_config` binds them to their
defaults. So the factory carries its own pair and names the class it builds."""

Shapes = tuple[tuple[int, ...], ...]
"""A batch's leaf shapes in tree order, the key a compiled step is held
under. A fixed batch has one; a ramped batch has one per stage of its ramp,
each compiled on the first step that reads it."""


def batch_shapes(batch: Batch) -> Shapes:
    """List a batch's leaf shapes in tree order."""
    return tuple(np.shape(leaf) for leaf in jax.tree.leaves(batch))


class Rollout(Protocol):
    """Produces a batch on the host, before the compiled step reads it.

    Sampling is effectful and untraceable, so it lives outside `jit`. The
    trainer calls the rollout with the state, the prefetched batch and a key
    folded from the run key and the step, then reshards what comes back with
    `shard_batch`. The returned batch must hold arrays in fixed shapes, so
    the step still compiles once per run."""

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...


@dataclasses.dataclass(frozen=True)
class ProfileWindow:
    """Asks for one profiler window per fit: `steps` steps traced into
    `directory` after `warmup` steps have run, so the trace holds the loop
    and not the compile.

    `dew.profile` is the other way to capture one, a context manager around
    any code at all; a fit refuses to schedule a window inside one. The
    window the loop wrote is reported as the `ProfileWindow` record of
    `dew.telemetry.records`."""
    directory: str
    steps: int
    warmup: int = 2


Book = tuple[jax.Array, jax.Array, jax.Array]
"""The loop's device-side counters: the interval's summed loss, the current
streak of non-finite losses, and the longest streak since the last check."""


def fresh_book() -> Book:
    """Return the counters a fresh logging interval starts from."""
    return jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)


@jax.jit
def bookkeep(book: Book, loss: jax.Array, finite: jax.Array) -> Book:
    """Advance the loop's counters for one step, in one dispatch.

    The same work as five eager ops, the cast, the add, the where, the add
    and the maximum, each dispatching an executable of its own. Those cost
    176 us a step on an i9-12900K against 37 us for this one call, measured
    over 2000 steps on the CPU backend with the result blocked on at the end.
    """
    interval_loss, bad_run, worst_bad_run = book
    bad_run = jnp.where(finite, 0, bad_run + 1)
    dtype = jnp.promote_types(loss.dtype, jnp.float32)
    return interval_loss + loss.astype(dtype), bad_run, jnp.maximum(worst_bad_run, bad_run)


def goodput(wall: float, first_step: float | None, other: float) -> dict[str, float]:
    """Compute the two goodput numbers from MaxText's report that need no
    cluster telemetry.

    `first_step` is the time from the start of `fit` to the first step's
    result: the placement or restore, the first batch, the compile and the
    step itself. It is None when no step ran. `other` is the time spent
    outside steps after that: evaluations, checkpoint writes and the wait for
    them at the end.

    The step fraction is what is left of `wall`. That counts a step's own
    data stall as step time, as MaxText's start-to-start step time does.
    """
    numbers = {}
    if first_step is not None:
        numbers["goodput/time_to_first_step_s"] = first_step
    steps = wall - (wall if first_step is None else first_step) - other
    numbers["goodput/step_fraction"] = max(steps, 0.0) / wall if wall > 0 else 0.0
    return numbers


def step_compiler_options(objective) -> jax.stages.CompilerOptions | None:
    """XLA options for this objective's training step on this device: Triton
    GEMM fusions off where `TRITON_GEMM_OFF_GENERATIONS` measured a win and
    no mixer of the model keeps them, unless the run set the flag itself."""
    if (device_generation() not in TRITON_GEMM_OFF_GENERATIONS
            or xla_flag('xla_gpu_enable_triton_gemm') is not None):
        return None
    model = getattr(objective, 'model', None)
    if model is None:
        return None
    mixers = [getattr(model, 'mixer', None)]
    mixers += [kind.mixer for kind in (getattr(model, 'kinds', None) or {}).values()]
    if any(getattr(mixer, 'keeps_triton_gemm', False) for mixer in mixers):
        return None
    return {'xla_gpu_enable_triton_gemm': False}
# What a model recomputes in its backward pass when its step does not fit,
# weakest first. Each rung is slower and holds less: on an NVIDIA L4 (24 GB,
# jax 0.11.2, bf16) the 359.8M-parameter decoder at 4 x 1024 tokens took
# 334.7, 356.4 and 401.3 ms at 9.93, 8.00 and 6.29 GiB, and at 16 x 1024
# only 'full' fits; DiT-L/2 on 64x64 inputs at 16 fits only under 'full'
# (docs/performance.md). A model's own remat is where it starts: the
# trainer moves it up one rung at a time until the compiled step fits the
# devices, and never past a policy the ladder does not name.
DECODER_REMAT = (None, REMAT_POLICIES['minimal'], REMAT_POLICIES['full'])
DIFFUSION_REMAT = (False, 'dots', 'full')


def step_fits(executable: jax.stages.Compiled, mesh: Mesh) -> bool:
    """Whether the compiled step's temporaries and new outputs fit the free
    memory of each of `mesh`'s devices, where the backend reports it. The
    arguments, the state and the batch, are already resident and counted in
    use; the donated state's buffers are reused for the outputs that alias
    them."""
    stats = executable.memory_analysis()
    memory = [device.memory_stats() or {} for device in mesh.devices.flat]
    if stats is None or not all('bytes_limit' in m and 'bytes_in_use' in m for m in memory):
        return True
    needed = stats.output_size_in_bytes - stats.alias_size_in_bytes + stats.temp_size_in_bytes
    return needed <= min(m['bytes_limit'] - m['bytes_in_use'] for m in memory)


def recompute_more(objective) -> bool:
    """Move the objective's model one rung up its remat ladder, and say
    whether there was a rung to move to."""
    model = getattr(objective, 'model', None)
    current = getattr(model, 'remat', ...)
    ladder = (DECODER_REMAT if isinstance(model, CausalTransformer)
              else DIFFUSION_REMAT if isinstance(current, bool | str) else ())
    current = 'dots' if current is True else current
    if current not in ladder or current == ladder[-1]:
        return False
    stronger = ladder[ladder.index(current) + 1]
    print(colored(f"the step does not fit the devices under remat {current!r}; "
                  f"compiling it again under {stronger!r}", "yellow"), file=sys.stderr)
    objective.model = model.clone(remat=stronger)
    return True


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
        profile: ProfileWindow | None = None,
    ):
        """Hold everything a run needs, without opening any of it.

        The mesh, the compiled step and the capabilities' resources come into
        being in `fit`, so constructing a Trainer allocates nothing.

        `accumulation` is how many microbatches pool into one optimizer
        commit. `step` replaces the built-in transaction. A custom step then
        owns the clocks, the scaler, the EMA and the mutable writes, and the
        compiled wrapper owns only the attempted-step counter. `rollout` runs
        once per batch read, before the step and outside replay. `layout` and
        `mesh` say where the state lives.
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
        # Set by `compile`, for the batch shape it was called with. A ramped
        # run has one value per stage; `fit` keeps them beside each step. The
        # program is the step as handed to XLA, before GSPMD partitions it,
        # and the executable the step as compiled, whose memory analysis a
        # benchmark reads.
        self.flops_per_step = None
        self.program: jax.stages.Lowered | None = None
        self.executable: jax.stages.Compiled | None = None

    @classmethod
    def from_config(
        cls, config: TrainerConfig, objective: Objective[ObjectiveLoss, ObjectiveEffects],
        optimizer: optax.GradientTransformation, *, key: jax.Array,
        checkpoints: Checkpoints | None = None, tracker: Tracker | None = None,
        step: Callable[[Objective[ObjectiveLoss, ObjectiveEffects],
                        optax.GradientTransformation], StepFn] | None = None,
        rollout: Rollout | None = None,
    ) -> Trainer[ObjectiveLoss, ObjectiveEffects]:
        """Build the trainer a `TrainerConfig` describes.

        The mapping from the config's field names to this constructor's is
        written once, here. `mesh`, `layout`, `accumulation`,
        `dynamic_scale` and `profile` are the config fields a trainer holds.
        `key` is the run key, which `RunConfig.train` draws from
        `config.seed`.

        The rest of the config belongs to the capabilities and to the loop,
        and reaches them from their own owners. `checkpoint_dir` and `keep`
        build the `Checkpoints` passed in here, and `wandb` the tracker.
        `xla_flags`, `multi_host` and `compilation_cache_dir` are read by
        `prepare_process` before JAX opens a backend. `batch_ramp` wraps the
        dataset with `dew.data.ramped`. `steps`, `epochs`, `log_every`,
        `eval_every` and `checkpoint_every` are arguments of `fit`. `step`
        and `rollout` are not configurable: they are code a caller hands
        over.

        It builds a `Trainer`, whatever it is called on. The objective's two
        parameters are the factory's own, so a subclass that wants one of
        itself constructs it.
        """
        return Trainer(
            objective, optimizer,
            key=key,
            mesh=config.mesh,
            layout=config.layout,
            accumulation=config.accumulation,
            dynamic_scale=config.dynamic_scale,
            checkpoints=checkpoints,
            tracker=tracker,
            step=step,
            rollout=rollout,
            profile=config.profile,
        )

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def initial_state(self, initializer: Initializer | None = None,
                      key: jax.Array | None = None) -> TrainState:
        """Build the state a fresh run starts from.

        It is pure, so `fit` traces it once for its shapes and once, sharded,
        for its values.

        Both inputs are the run's own by default, and `place` passes them
        explicitly so that what it compiles takes them as arguments. A held
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
            # The average starts equal to the parameters but as its own
            # buffers: the step donates the state, and a buffer can be
            # donated once.
            ema=None if ema is None else jax.tree.map(jnp.copy, select(params, ema.select)),
            key=run_key,
        )

    @functools.cached_property
    def device_mesh(self) -> Mesh:
        """Build the mesh `MeshSpec` describes over this process pool's devices,
        on first use."""
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

    def shardings(self, state: TrainState) -> Placement[TrainState]:
        """Place every field of `state`, each on the axes its own kind takes.

        Parameter gradients follow parameters, replay records follow batches,
        and the layout's host-resident fields sit in pinned host memory.
        Under a CPU-owned state the frozen collection is the exception: it
        sits where the realization reads it (`execution.resident`) for the
        whole run."""
        mesh = self.state_mesh
        params = dict(state.params)
        frozen = params.pop(FROZEN, None) if self.host_master else None
        placed = self.layout.shardings(mesh, dataclasses.replace(state, params=params, accumulation=None))
        placed = dataclasses.replace(placed, **{
            field: jax.tree.map(lambda s: s.with_memory_kind("pinned_host"), getattr(placed, field))
            for field in (() if self.host_master else self.layout.host)})
        if frozen is not None:
            placed = dataclasses.replace(placed, params={**placed.params, FROZEN: self._frozen_shardings(state, frozen)})
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

        Returns it with its shardings and the data position a resume
        continues from."""
        # Resolved once, so that the shapes and the values are the same
        # inputs through the same overridable method, and the objective is
        # asked for what it holds exactly once.
        initializer, key = self.objective.initializer, self.key
        if self.host_master:
            from dew.training.host import transfer
            key = transfer(key, NamedSharding(self.state_mesh, P()))
        abstract = jax.eval_shape(self.initial_state, initializer, key)
        shardings = self.shardings(abstract)
        self.layout.check(abstract.params, shardings.params, self.device_mesh)
        checkpoints = self.checkpoints
        resume = None if checkpoints is None else checkpoints.latest
        if checkpoints is None or resume is None:
            if self.host_master:
                return self._placed_host(initializer, key, shardings), shardings, None
            state = jax.jit(self.initial_state, out_shardings=shardings)(initializer, key)
            return state, shardings, None
        abstract = dataclasses.replace(abstract, accumulation=checkpoints.accumulation_template(resume))
        shardings = self.shardings(abstract)
        if self.host_master and FROZEN in abstract.params:
            abstract = dataclasses.replace(abstract, params={**abstract.params, FROZEN: self._banked_frozen(
                abstract.params[FROZEN],
                lambda rows, path: jax.ShapeDtypeStruct((len(rows), *rows[0].shape), rows[0].dtype))})
        template = jax.tree.map(
            lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
            abstract, shardings)
        state, position = checkpoints.restore(template, resume,
                                              share=data_partition(self.device_mesh))
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        print(f"Resumed from step {resume} in {checkpoints.source(resume)}")
        return state, shardings, position

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
            self.state_mesh, dataclasses.replace(state, params={FROZEN: frozen}, accumulation=None))
        return resident(rows.params[FROZEN], self.bank_sites, self.device_mesh)

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
            if FROZEN in state.params:
                check_bank_pool(bank_bytes(state.params[FROZEN], self.bank_sites), self.device_mesh)
                # A scanned run's frozen rows become its bank here and the
                # bank lands where it stays resident before the next is
                # stacked, so the host holds one bank in transit, not the
                # stack; the rows' pages are let go as it lands and the held
                # tree holds the placed bank where each row was. An
                # objective placed this way holds banks at its rows afterwards
                # and does not seed a second trainer.
                targets = shardings.params[FROZEN]

                def stack(rows, path):
                    target = targets
                    for component in path:
                        target = target[component]
                    bank = place_leaf(jnp.stack(rows), target)
                    for row in rows:
                        evict(np.asarray(row))
                    return bank

                def release(namespace, index, keys, bank):
                    node = held["params"] if held is not None else None
                    for component in (*namespace, f"layers_{index}", *keys[:-1]):
                        node = node.get(component) if isinstance(node, dict) else None
                    if isinstance(node, dict) and keys[-1] in node:
                        node[keys[-1]] = bank
                frozen = self._banked_frozen(state.params[FROZEN], stack, release)
                state = dataclasses.replace(state, params={**state.params, FROZEN: frozen})
        params = stream(state.params, shardings.params, held)
        ema = None if state.ema is None else stream(state.ema, shardings.ema)
        rest = dataclasses.replace(state, params=None, ema=None)
        placed = transfer(rest, dataclasses.replace(shardings, params=None, ema=None))
        return dataclasses.replace(placed, params=params, ema=ema)

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def _loss_shape(self, state: TrainState, batch: Batch):
        # Shapes and dtypes only: a resident frozen leaf sits in another
        # memory space than the moving ones, and the loss the realization
        # runs reads a snapshot in one space. The step's key is drawn inside
        # the trace, so an abstract state compiles a step as a placed one does.
        params = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.params)

        def loss(params, batch, microstep, key, step, ema):
            return self.objective.loss(
                params, batch, Step(microstep, jax.random.fold_in(key, step), with_ema(params, ema)))

        return jax.eval_shape(loss, params, batch, state.microstep, state.key, state.step, state.ema)

    def _initialize_accumulation(self, state: TrainState, batch: Batch, shapes, *, shape_only=False):
        if self.accumulation == 1 or self.step is not None or state.accumulation is not None:
            return state
        stats, aux = shapes
        shared = isinstance(stats, (Mean, jax.ShapeDtypeStruct))
        mean_dtype = jnp.result_type(jnp.float32, *(x.dtype for x in jax.tree.leaves(stats)))
        slots = self.accumulation - 1
        def shape(leaf: jax.Array) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)

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

    @contextlib.contextmanager
    def _traced_on(self, mesh: Mesh) -> Iterator[Schedule]:
        """What a traced step reads from context: the mesh, the pipeline's
        microbatch count, and the layout's rules, which place the activations
        the model constrains (`dew.nn.sharding.constrain`) as they place the
        parameters."""
        with (jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches) as schedule,
              nn.logical_axis_rules(self.layout.axis_rules)):
            yield schedule

    def compile(self, state: TrainState, batch: Batch) -> CompiledStep:
        """Compile a transaction over state and one already-produced global batch.

        The step consumes the state it is given. The returned state takes
        over its buffers, so the update runs in place and peak memory holds
        one copy of the parameters and optimizer state, not two. Keep no
        reference to a state after stepping it; `new = step(old, batch)` is
        the whole contract.

        A checkpoint saved before the step is safe. Orbax copies every array
        to the host before `save` returns, as long as `Checkpoints` names no
        prioritized keys and no concurrent transfer limit. The batch is not
        donated; the loader owns it.
        """
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        if (state.scale is not None) != self.dynamic_scale:
            raise ValueError("checkpoint dynamic-scaler configuration differs from this trainer")
        if self.host_master:
            return self._compile_host(state, batch)
        mesh = self.device_mesh
        with self._traced_on(mesh) as schedule:
            shapes = None if self.step is not None else self._loss_shape(state, batch)
            if shapes is not None and mesh.shape[STAGE_AXIS] > 1 and not schedule.pipelined:
                raise ValueError(
                    f"the stage axis of {mesh.shape[STAGE_AXIS]} holds a pipeline's stages of "
                    f"a decoder's layer stack, and {type(self.objective).__name__}'s model runs "
                    f"no pipeline, so every stage would compute the whole step; give those "
                    f"devices to the data or fsdp axis")
            prepared = self._initialize_accumulation(state, batch, shapes, shape_only=True)
            shardings = self.shardings(prepared)
            replicated = NamedSharding(mesh, P())
            prepared = jax.tree.map(
                lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), prepared, shardings)
            while True:
                body = (self.step(self.objective, self.optimizer) if self.step is not None
                        else self._default_step(shapes))

                def step(current, batch, body=body):
                    # The body sees every field on the device; the out shardings
                    # return the host-resident ones to pinned host memory.
                    advanced, loss, aux = body(self._fetched(current, shardings), batch)
                    return (dataclasses.replace(advanced, step=current.step + 1), loss,
                            aux.metrics, jnp.isfinite(loss), advanced.microstep > current.microstep)

                jitted = jax.jit(step, in_shardings=(shardings, batch_shardings(mesh, batch)),
                                 out_shardings=(shardings, replicated, replicated, replicated,
                                                replicated),
                                 donate_argnums=0)
                self.program = jitted.lower(prepared, batch)
                self.executable = self.program.compile(step_compiler_options(self.objective))
                if step_fits(self.executable, mesh) or not recompute_more(self.objective):
                    break
            self.flops_per_step = compiled_flops(self.executable)

        def run(current, batch):
            with self._traced_on(mesh):
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

    def fit(self, dataset: Dataset, *, steps: int, log_every: int = 100,
            eval_every: int | None = None, checkpoint_every: int | None = None,
            metrics: Sequence[Metric] = (), preview: bool = False) -> TrainState:
        """Train to `steps` total steps, resuming from the checkpoints' latest
        step when the directory holds one.

        Every `log_every` steps the tracker receives the loss, the objective's
        metrics and the throughput.

        Every `eval_every` steps, and at the end, the validation split is
        scored. The objective's artifacts go to the tracker and to `metrics`,
        whose reductions are logged as `val/<name>`.

        Every `checkpoint_every` steps, and at the end, the state and the
        data position are written. Every `checkpoints.local_every` steps they
        are written to the local directory as well.

        Previews are generated only when `preview=True` and a tracker
        receives them; scalar reporting never triggers preview work.
        """
        self._check_validation_is_read(eval_every, metrics, preview=preview)
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
        profiler = self._own_profile_window()
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
            # The interval's loss and both bad-loss counters live on device,
            # so the loop never blocks on a result, and they move together in
            # one dispatch. The host reads them at the logging cadence.
            book = fresh_book()
            seen = 0
            first_step = None

            if current < steps:
                source = dataset.train(data_partition(mesh))
                self._check_stream(source, mesh,
                                   checkpointing=bool(checkpoint_every or local_every))
                train = DevicePrefetchIterator(source, mesh, source_state=position)
                source = None  # Lifetime transferred to the prefetch worker.

            def announce() -> None:
                if process_zero:
                    print(f"Training from step {current} to {steps} on "
                          f"{dict(mesh.shape)} ({jax.process_count()} process(es))")

            agreed("training announcement", announce)
            while current < steps:
                assert train is not None
                # The window's capture opens before this iteration's first
                # read, so the step row records the read it waits on rather
                # than a compile that ran before capture began.
                if (profiler is not None and profile is not None
                        and not tracing and traced == 0
                        and seen >= profile.warmup):
                    tracing = self._start_window(profiler)
                capturing = tracer is not None and tracer.running
                step_scope = (jax.profiler.StepTraceAnnotation("train", step_num=current)
                              if capturing else contextlib.nullcontext())
                with step_scope:
                    with region("input.wait"):
                        batch = next(train)
                    if self.rollout is not None:
                        batch, sampled = self._rolled_out(state, batch, mesh)
                        rollout_seconds += sampled
                    first_compile = not compiled
                    train_step, measured_flops = self._compiled_for(compiled, state, batch)
                    if first_compile:
                        # Rebound once the step is compiled, so the first tick
                        # measures steps, not the compile.
                        last_log_time = time.time()
                    with region("train.step"):
                        state, loss, aux, finite, accepted = train_step(state, batch)
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
                        now = self._log_interval(
                            current, loss, aux, accepted, state, since=last_log_time,
                            steps=steps_since_log, samples=interval_samples,
                            flops=interval_flops, rollout_seconds=rollout_seconds,
                            process_zero=process_zero)
                        if now is not None:
                            last_log_time, steps_since_log, rollout_seconds = now, 0, 0.0
                            interval_samples, interval_flops = 0, 0.0

                    if eval_every and current % eval_every == 0 and current < steps:
                        other += self._timed_evaluation(
                            state, shardings, dataset, metrics, preview, mesh)

                    # On its own clock, not the logging one, so that a cadence
                    # which does not divide log_every still fires.
                    if (checkpoint_every and checkpoints is not None
                            and current % checkpoint_every == 0 and current < steps):
                        other += self._saved_checkpoint(checkpoints, current, state, position,
                                                        book, interval_steps)
                        last_saved = current
                        book = (jnp.zeros_like(book[0]), book[1], book[2])
                        interval_steps = 0
                    if (local_every and checkpoints is not None
                            and current % local_every == 0 and current < steps):
                        other += self._saved_local_checkpoint(
                            checkpoints, current, state, position)
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
                self._evaluate(state, shardings, dataset, metrics, preview, mesh)
            if checkpoints is not None and last_saved != current:
                # The in-loop saves are conditional, so the state the run ends
                # on may never have been written. It goes out under its real
                # step: a step-0 checkpoint holding the final weights would
                # make a resume restart the schedule from the beginning.
                checkpoints.save(
                    current, state, position,
                    {"loss": float(interval_loss / interval_steps)} if interval_steps else None,
                    share=data_partition(mesh))
                self._report(CheckpointRequested(checkpoints.directory), current)
            other += time.perf_counter() - paused
        finally:
            paused = time.perf_counter()
            primary = sys.exception()
            close = (train.close if train is not None else
                     source.close if isinstance(source, Closeable) else None)
            stop_trace: Callable[[], None] | None = None
            if tracing and profile is not None:
                tracing = False
                assert profiler is not None
                stop_trace = functools.partial(
                    self._stop_trace, traced, loss, profile, profiler, step=current)
            error = self._cleaned_up(primary, close, stop_trace, checkpoints)
            # The traceback of a failed run holds this frame, and with it
            # whatever these names still point at.
            source = train = close = stop_trace = None
            other += time.perf_counter() - paused
            error = self._reported_outcome(error, started, first_step, other, current,
                                           process_zero=process_zero)
            if primary is None and error is not None:
                raise error
        return state

    def _check_validation_is_read(self, eval_every: int | None,
                                  metrics: Sequence[Metric], *, preview: bool) -> None:
        """Refuse a validation cadence whose batches nothing would read.

        A validation pass hands its batches to metrics and to the preview.
        Scheduled with neither, it would open nothing and report nothing, so
        the contradiction is refused before the run does any work.
        """
        if eval_every and not metrics and not (preview and self.tracker is not None):
            raise ValueError(
                f"eval_every={eval_every} schedules a validation pass that nothing consumes: "
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
                    "dew.profile capture is active; drop one or stop the outer "
                    "profiler before fitting")
            telemetry_profile.require_profile_support()
            return telemetry_profile.profile(profile.directory)

        return agreed("profiling window setup", own_window)

    def _check_stream(self, source, mesh: Mesh, *, checkpointing: bool) -> None:
        """Refuse a training stream this run cannot checkpoint or cannot shard.

        A checkpoint written without the data position would replay the data
        on resume. A ramp stage whose batch the mesh cannot divide fails
        where it is placed or traced, which for a later stage is an hour in.
        """
        if checkpointing and not isinstance(source, Checkpointable):
            raise ValueError(
                f"checkpoint_every needs a training stream with get_state and "
                f"set_state, and {type(source).__name__} lacks one; a checkpoint "
                f"written without the data position would replay the data on "
                f"resume. Train it with checkpoint_every=None "
                f"(--trainer.checkpoint-every None)")
        if not isinstance(source, RampedStream):
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

    def _start_window(self, profiler: Profiler) -> bool:
        """Open the profiler's capture on every rank, and say it is capturing.

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
        return capturing

    def _rolled_out(self, state: TrainState, batch: Batch, mesh: Mesh) -> tuple[Batch, float]:
        """Sample one batch through the rollout, with the seconds it took.

        Host-side and untraceable: sampling, scoring, advantages. The key
        folds the step key once more, keeping the rollout's draws off the
        step's stream; both are checkpointed, so a resumed run samples
        forward. Fixed shapes mean the step still compiles once.
        """
        began = time.perf_counter()
        assert self.rollout is not None
        key = jax.random.fold_in(jax.random.fold_in(state.key, state.step), 1)
        batch = shard_batch(mesh, self.rollout(state, batch, key))
        return batch, time.perf_counter() - began

    def _compiled_for(self, compiled: dict[Shapes, tuple[CompiledStep, float | None]],
                      state: TrainState, batch: Batch) -> tuple[CompiledStep, float | None]:
        """Return the step compiled for this batch's shapes, compiling on first sight.

        A ramped run reads a new shape at every stage, so `compiled` keeps
        one step per stage with the FLOPs measured for it.
        """
        shapes = batch_shapes(batch)
        if shapes not in compiled:
            with region("compile"):
                compiled[shapes] = (self.compile(state, batch), self.flops_per_step)
        return compiled[shapes]

    def _timed_evaluation(self, state: TrainState, shardings: Placement[TrainState],
                          dataset: Dataset, metrics: Sequence[Metric], preview: bool,
                          mesh) -> float:
        """Score the validation split, returning the seconds it took.

        Those seconds are the run's, but not its steps', which is what the
        goodput fraction is measured against."""
        paused = time.perf_counter()
        with region("evaluate"):
            self._evaluate(state, shardings, dataset, metrics, preview, mesh)
        return time.perf_counter() - paused

    def _saved_checkpoint(self, checkpoints: Checkpoints, step: int, state: TrainState,
                          position: bytes | None, book: Book, interval_steps: int) -> float:
        """Write one checkpoint with the interval's mean loss, and report it.

        Returns the seconds it took, the reading of the interval's loss
        included: that read waits on the device, and the wait is time the
        steps did not have."""
        paused = time.perf_counter()
        checkpoints.save(step, state, position, {"loss": float(book[0] / interval_steps)},
                         share=data_partition(self.device_mesh))
        self._report(CheckpointRequested(checkpoints.directory), step)
        return time.perf_counter() - paused

    def _saved_local_checkpoint(self, checkpoints: Checkpoints, step: int,
                                state: TrainState, position: bytes | None) -> float:
        """Write one checkpoint to the local directory, and report it.

        Returns the seconds it took. The local copy carries no metadata; it
        is the one a restarted node reads back, not the run's record."""
        paused = time.perf_counter()
        checkpoints.save_local(step, state, position, share=data_partition(self.device_mesh))
        self._report(CheckpointRequested(str(checkpoints.local_directory), local=True), step)
        return time.perf_counter() - paused

    def _log_interval(self, step: int, loss: jax.Array, aux: dict[str, jax.Array],
                      accepted: jax.Array, state: TrainState, *, since: float, steps: int,
                      samples: int, flops: float | None, rollout_seconds: float,
                      process_zero: bool) -> float | None:
        """Report one logging interval, returning the clock the next one starts from.

        Rank zero builds the row and returns the time it read; every other
        rank has nothing to report and returns None. The report is agreed, so
        a tracker that failed on rank zero stops its peers here rather than
        at their next collective.
        """
        def report() -> float | None:
            if not process_zero:
                return None
            # The interval's numbers need the loss on the host, so this is
            # where the loop waits on the device.
            loss.block_until_ready()
            now = time.time()
            scalars = {"train/loss": float(loss),
                       **{f"train/{k}": float(v) for k, v in aux.items()},
                       **self._throughput(now - since, steps, samples, flops)}
            scalars["train/accepted"] = float(accepted)
            if state.scale is not None:
                scalars["train/loss_scale"] = float(state.scale.scale)
            if self.rollout is not None:
                scalars["train/rollout_seconds"] = rollout_seconds
            print(f"step {step}: loss {scalars['train/loss']:.4f}")
            if self.tracker is not None:
                self.tracker.log(scalars, step)
            return now

        with region("log"):
            return agreed("training reporting", report)

    def _cleaned_up(self, primary: BaseException | None, close: Callable[[], None] | None,
                    stop_trace: Callable[[], None] | None,
                    checkpoints: Checkpoints | None) -> BaseException | None:
        """Run every teardown step, returning the failure the run ends on.

        Each step runs even when an earlier one failed, and a later failure
        becomes a note on the first, so one broken sink cannot hide the error
        that ended the run.
        """
        error = primary
        for label, cleanup in (
            ("Training iterator", close),
            ("Profiler", stop_trace),
            ("Checkpoint wait", None if checkpoints is None else checkpoints.wait),
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
        return error

    def _reported_outcome(self, error: BaseException | None, started: float,
                          first_step: float | None, other: float, step: int, *,
                          process_zero: bool) -> BaseException | None:
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
                if process_zero:
                    scalars = goodput(time.perf_counter() - started, first_step, other)
                    print(f"Goodput: first step after {scalars.get('goodput/time_to_first_step_s', 0.0):.2f} s, "
                          f"{scalars['goodput/step_fraction']:.1%} of the wall time in steps")
                    if self.tracker is not None:
                        self.tracker.log(scalars, step)

            try:
                agreed("goodput reporting", report_goodput)
            except BaseException as failure:
                error = failure
        try:
            self._report(FitEnded.outcome(time.perf_counter() - started, error), step)
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

    def _evaluate(self, state: TrainState, shardings: Placement[TrainState], dataset: Dataset,
                  metrics: Sequence[Metric], preview: bool, mesh) -> None:
        """Score the validation split with this state's variables and report it.

        A CPU-owned run evaluates on the accelerator, over the same snapshot
        a step realizes, so validation reads the weights where the loss
        does."""
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
        # The rules and the microbatch count; `evaluate` scores under the mesh
        # itself, and previews decode outside it.
        with (pipeline_microbatches(self.mesh.microbatches),
              nn.logical_axis_rules(self.layout.axis_rules)):
            evaluation = evaluate(
                self.objective, params, dataset.val, metrics=metrics, key=key,
                step=state.step, schedule_step=state.microstep,
                averaged=averaged, preview=preview, mesh=mesh)
        self._report_evaluation(evaluation)

    def _report_evaluation(self, advanced: Evaluation) -> None:
        """Print one evaluation on rank zero and log its previews and scores."""
        def report() -> None:
            if jax.process_index() != 0:
                return
            print(f"Evaluation {advanced.split} at step {advanced.step}: "
                  f"{advanced.coordinated_batches} coordinated batches, {advanced.records} records, "
                  f"uneven_shards={advanced.uneven_shards}, event_key={advanced.event_key}: {advanced.scores}")
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

    def _stop_trace(self, traced: int, loss, profile: ProfileWindow,
                    profiler: Profiler, *, step: int) -> None:
        """Stop the window's owned capture, then report it on process zero.

        The core Profiler drains the backend and exports the native reports
        on `stop`. The loss's block only orders the primary's failure ahead
        of the profiler's own drain."""
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
            if jax.process_index() == 0:
                print(f"Wrote profile for {traced} steps to {profile.directory}")

        agreed("profile stop", stop)
        self._report(ProfileWindowRecord(profile.directory, traced), step)
        agreed("profile announcement", announce)

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
