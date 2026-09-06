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
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from termcolor import colored

from dew.artifacts import host
from dew.checkpoints import Checkpoints
from dew.data.dataset import Checkpointable
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Aux, Batch, Metric, Objective, Step, Variables, merge, select
from dew.telemetry.instrumentation import model_flops_utilization, step_flops
from dew.training.distributed import (
    DevicePrefetchIterator, Layout, MeshSpec, Placement, batch_shardings, build_mesh,
    minimum_across_processes, shard_batch,
)
from dew.training.state import TrainState
from dew.training.tracker import Tracker

if TYPE_CHECKING:
    from dew.data import Dataset

# Consecutive non-finite losses that stop a run.
BAD_LOSS_STEPS = 5

StepFn = Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]
"""A compiled step's body: the state and the global batch in, the new state,
the loss and the objective's report out."""

CompiledStep = Callable[
    [TrainState, dynamic_scale_lib.DynamicScale | None, Batch],
    tuple[TrainState, dynamic_scale_lib.DynamicScale | None, jax.Array,
          Mapping[str, jax.Array], jax.Array]]
"""What `Trainer.compile` returns: `(state, scale, batch)` in, `(state, scale,
loss, metrics, finite)` out."""


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
    return interval_loss + loss.astype(jnp.float32), bad_run, jnp.maximum(worst_bad_run, bad_run)


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


def with_ema(params: Variables, ema: Variables | None) -> Variables | None:
    """The variables tree with the averaged leaves in place of the live ones."""
    return None if ema is None else merge(params, ema)


def _project(tree: Variables, like: Variables) -> Variables:
    """The leaves of `tree` at the paths `like` holds, in `like`'s nesting."""
    return {name: _project(tree[name], child) if isinstance(child, Mapping) else tree[name]
            for name, child in like.items()}


def ema_update(ema: Variables, params: Variables,
               decay: jax.typing.ArrayLike) -> Variables:
    """One EMA step over the selected leaves; `decay` is the schedule's value.

    At unit decay the average passes through unchanged. The arithmetic form
    multiplies the live tree by zero, and zero times a non-finite parameter
    is NaN, so one bad parameter would poison a frozen reference on the step
    it appears. The select runs per leaf."""
    def step(average, live):
        updated = decay * average + (1 - decay) * live
        return jnp.where(jnp.asarray(decay) >= 1.0, average, updated)

    return jax.tree.map(step, ema, _project(params, ema))


def write_back(params: Variables, variables: Variables | None) -> Variables:
    """`params` with the collections in `variables` replaced whole."""
    if variables is None:
        return params
    if "params" in variables:
        raise ValueError("Aux.variables cannot carry the params collection; "
                         "the optimizer owns it")
    for name, collection in variables.items():
        if name not in params:
            raise ValueError(f"Aux.variables names {name!r}, which is not a collection "
                             f"of the objective's tree {sorted(params)}")
        if jax.tree.structure(collection) != jax.tree.structure(params[name]):
            raise ValueError(f"Aux.variables[{name!r}] does not have the collection's "
                             f"structure")
    return {**params, **variables}


def _pick(artifacts: tuple, reads: type):
    matching = [artifact for artifact in artifacts if isinstance(artifact, reads)]
    if len(matching) != 1:
        raise ValueError(
            f"a metric reads {reads.__name__}, and the objective's evaluation produced "
            f"{[type(a).__name__ for a in artifacts]}")
    return matching[0]


class Trainer:
    """Runs an `Objective`: gradients, sharding, EMA, checkpoints, logging."""

    def __init__(
        self,
        objective: Objective,
        optimizer: optax.GradientTransformation,
        *,
        key: jax.Array,
        mesh: MeshSpec = MeshSpec(),
        layout: Layout = Layout(),
        accumulation: int = 1,
        dynamic_scale: bool = False,
        checkpoints: Checkpoints | None = None,
        tracker: Tracker | None = None,
        step: Callable[[Objective, optax.GradientTransformation], StepFn] | None = None,
        rollout: Rollout | None = None,
        profile: Profile | None = None,
    ):
        """`accumulation` wraps the optimizer in `optax.MultiSteps` here, and
        the EMA runs on the update clock that wrapper defines. `step` replaces
        the compiled step's body with `step(objective, optimizer)`, for an
        update that is not one loss (a GAN's alternating optimizers); it then
        owns the step counter, the EMA and the `Aux.variables` write-back,
        with `ema_update` and `write_back` at hand. `rollout` produces the
        batch the step consumes; None trains the prefetched batch untouched."""
        if accumulation < 1:
            raise ValueError(f"accumulation must be at least 1, got {accumulation}")
        self.objective = objective
        self.optimizer: optax.GradientTransformation = (
            optax.MultiSteps(optimizer, every_k_schedule=accumulation).gradient_transformation()
            if accumulation > 1 else optimizer)
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
        # Measured off the compiled step, once per fit.
        self.flops_per_step = None

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def initial_state(self) -> TrainState:
        """The state a fresh run starts from. Pure, so `fit` traces it once
        for its shapes and once, sharded, for its values."""
        init_key, run_key = jax.random.split(self.key)
        params = nn.unbox(self.objective.init(init_key))
        if "params" not in params:
            raise ValueError(
                f"the objective's tree has no params collection, only {sorted(params)}; "
                "the optimizer moves params and treats every other collection as state")
        ema = self.objective.ema
        return TrainState(
            step=jnp.zeros((), jnp.int32),
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

    def shardings(self, state: TrainState) -> Placement:
        """The layout's placement of `state`, leaf for leaf."""
        return self.layout.shardings(self.device_mesh, state)

    def place(self) -> tuple[TrainState, Placement, bytes | None]:
        """The state itself, fresh or restored, on the mesh, with its shardings
        and the data position a resume continues from."""
        abstract = jax.eval_shape(self.initial_state)
        shardings = self.shardings(abstract)
        self.layout.check(abstract.params, shardings.params, self.device_mesh)
        checkpoints = self.checkpoints
        resume = None if checkpoints is None else checkpoints.latest
        if checkpoints is None or resume is None:
            state = jax.jit(self.initial_state, out_shardings=shardings)()
            return state, shardings, None
        template = jax.tree.map(
            lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
            abstract, shardings)
        state, position = checkpoints.restore(template, resume)
        print(f"Resumed from step {resume} in {checkpoints.source(resume)}")
        return state, shardings, position

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def _default_step(self):
        """The compiled step's body over the global batch; GSPMD partitions it.

        The loss is a mean over the batch-sharded axis, so its gradient
        carries the cross-device all-reduce on its own. One key per step:
        threefry is partitionable, so every device draws its own slice of the
        same stream without folding in a device index.
        """
        objective = self.objective
        optimizer = self.optimizer
        ema_spec = objective.ema
        accumulation = self.accumulation

        def step(state: TrainState, scale, batch):
            info = Step(step=state.step, key=jax.random.fold_in(state.key, state.step),
                        ema=with_ema(state.params, state.ema))

            def loss_fn(trainable):
                return objective.loss({**state.params, "params": trainable}, batch, info)

            if scale is not None:
                grad_fn = scale.value_and_grad(loss_fn, has_aux=True)
                scale, finite, (loss, aux), grads = grad_fn(state.params["params"])
            else:
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    state.params["params"])
                finite = None

            # The QK-Clip's per-head maxima ride in as an extra arg where the
            # optimizer declares it takes them. Plain optax transforms take
            # the update as always.
            if isinstance(optimizer, optax.GradientTransformationExtraArgs):
                updates, opt_state = optimizer.update(
                    grads, state.opt_state, state.params["params"],
                    qk_stats=aux.qk_stats)
            else:
                updates, opt_state = optimizer.update(
                    grads, state.opt_state, state.params["params"])
            params = write_back(
                {**state.params, "params": optax.apply_updates(state.params["params"], updates)},
                aux.variables)
            new_state = dataclasses.replace(
                state, step=state.step + 1, params=params, opt_state=opt_state)

            if finite is not None:
                # Overflowed gradients mean the update did not happen, so the
                # step counter, which every schedule reads, stays with the
                # params and the optimizer state.
                keep = functools.partial(jnp.where, finite)
                new_state = dataclasses.replace(
                    new_state,
                    step=keep(new_state.step, state.step),
                    params=jax.tree.map(keep, new_state.params, state.params),
                    opt_state=jax.tree.map(keep, new_state.opt_state, state.opt_state))

            if ema_spec is not None:
                # `state.step` counts micro-batches, and under MultiSteps the
                # params only move on every accumulation-th one. The EMA runs
                # on that same clock: the schedule is indexed by completed
                # updates and the average happens on the micro-step whose
                # update lands. A rejected mixed-precision step is not an
                # update either.
                decay = ema_spec.decay(state.step // accumulation)
                due = True if finite is None else finite
                if accumulation > 1:
                    due = due & ((state.step + 1) % accumulation == 0)
                if state.ema is None:
                    raise ValueError(
                        "the objective declares an EMA and the state carries none; "
                        "a state built by this trainer always holds one")
                averaged = ema_update(state.ema, new_state.params, decay)
                if due is not True:
                    averaged = jax.tree.map(functools.partial(jnp.where, due),
                                            averaged, state.ema)
                new_state = dataclasses.replace(new_state, ema=averaged)
            return new_state, scale, loss, aux

        return step

    def _step_body(self):
        if self.step is None:
            return self._default_step()
        custom = self.step(self.objective, self.optimizer)

        def step(state, scale, batch):
            state, loss, aux = custom(state, batch)
            return state, scale, loss, aux

        return step

    def compile(self, state: TrainState, batch: Batch,
                scale: dynamic_scale_lib.DynamicScale | None = None) -> CompiledStep:
        """The training step `fit` runs, compiled ahead of time over `state`
        and one global `batch`.

        The step takes `(state, scale, batch)` and returns `(state, scale,
        loss, metrics, finite)`; `scale` is the `DynamicScale` of a
        mixed-precision run and None otherwise. Its shardings are the layout's
        and it donates the state, so a benchmark that runs it measures the
        step a real run runs.

        What comes back calls the jitted function, not the executable.
        Calling an executable from Python re-checks every leaf's shape and
        sharding on every call, about 4.5 us a leaf (0.4 ms a step for a
        99-leaf state, 4.9 ms for 1067 leaves, one CPU core of an i9-12900K),
        while jit dispatches through its C++ cache. The FLOP count still comes
        off the ahead-of-time executable: jit lowers through the same cache,
        so the first call finds that compilation and compiles nothing.

        Every call runs under `jax.set_mesh` and the pipeline's microbatch
        count, which put the mesh and the schedule in context while the step
        traces: the attention seam reads the sequence axis off the mesh
        (`dew.nn.sharding.sequence_shards`), a decoder reads the stage axis
        and the count (`pipeline_stages` and `microbatches`), and the mesh
        context is part of jit's cache key, so a call outside it would trace
        and compile the step a second time. Entering it costs nothing
        measurable beside the dispatch (32 us either way on the CPU above).
        """
        body = self._step_body()
        mesh = self.device_mesh
        shardings = self.shardings(state)
        replicated = NamedSharding(mesh, P())

        def step(state, scale, batch):
            state, scale, loss, aux = body(state, scale, batch)
            return state, scale, loss, aux.metrics, jnp.isfinite(loss)

        jitted = jax.jit(
            step,
            in_shardings=(shardings, jax.tree.map(lambda _: replicated, scale),
                          batch_shardings(mesh, batch)),
            out_shardings=(shardings, jax.tree.map(lambda _: replicated, scale),
                           replicated, replicated, replicated),
            donate_argnums=(0,),
        )
        with jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches):
            self.flops_per_step = step_flops(jitted, state, scale, batch)

        def run(state, scale, batch):
            with jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches):
                return jitted(state, scale, batch)

        return run

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def fit(self, data: "Dataset", *, steps: int, log_every: int = 100,
            eval_every: int | None = None, checkpoint_every: int | None = None,
            metrics: Sequence[Metric] = ()) -> TrainState:
        """Train to `steps` total steps, resuming from the checkpoints' latest
        step when the directory holds one.

        Every `log_every` steps the tracker receives the loss, the objective's
        metrics and the throughput. Every `eval_every` steps, and at the end,
        the validation split is scored: the objective's artifacts go to the
        tracker and to `metrics`, whose reductions are logged as `val/<name>`.
        Every `checkpoint_every` steps, and at the end, the state and the data
        position are written; every `checkpoints.local_every` steps they are
        written to the local directory as well.
        """
        started = time.perf_counter()
        profile, checkpoints = self.profile, self.checkpoints
        source = train = None
        tracing, traced = False, 0
        loss = None
        other = 0.0
        try:
            mesh = self.device_mesh
            process_zero = jax.process_index() == 0
            state, shardings, position = self.place()
            current = int(state.step)
            if current > steps:
                raise ValueError(f"the run is at step {current}, past the {steps} asked for")

            if checkpoint_every and checkpoints is None:
                raise ValueError(
                    "checkpoint_every asks for checkpoints and this trainer has no "
                    "checkpointer; pass Checkpoints(directory) to write any")
            local_every = None if checkpoints is None else checkpoints.local_every
            scale = dynamic_scale_lib.DynamicScale() if self.dynamic_scale else None
            train_step = None
            # Rebound once the step is compiled, so the first tick measures steps,
            # not the compile.
            last_log_time = time.time()
            last_saved = current if checkpoints is not None and current else None
            interval_steps = 0
            steps_since_log = 0
            # Seconds spent sampling this interval, logged under
            # train/rollout_seconds when a rollout is set.
            rollout_seconds = 0.0
            # The interval's loss and both bad-loss counters live on device, so the
            # loop never blocks on a result, and move together in one dispatch.
            # `worst_bad_run` remembers the longest streak of non-finite losses
            # seen since the last host check; the host check reads it to decide
            # whether to stop.
            book = fresh_book()
            tracing, traced, seen = False, 0, 0
            loss = None
            first_step = None
            other = 0.0

            if current < steps:
                source = data.train()
                if (checkpoint_every or local_every) and not isinstance(source, Checkpointable):
                    raise ValueError(
                        f"checkpoint_every needs a training stream with get_state and "
                        f"set_state, and {type(source).__name__} lacks one; a checkpoint "
                        f"written without the data position would replay the data on "
                        f"resume. Train it with checkpoint_every=None "
                        f"(--trainer.checkpoint-every None)")
                train = DevicePrefetchIterator(source, mesh, source_state=position)
                source = None  # Lifetime transferred to the prefetch worker.

            if process_zero:
                print(f"Training from step {current} to {steps} on "
                      f"{dict(mesh.shape)} ({jax.process_count()} process(es))")
            while current < steps:
                assert train is not None
                batch = next(train)
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
                if train_step is None:
                    train_step = self.compile(state, batch, scale)
                    last_log_time = time.time()
                if (profile is not None and not tracing and traced == 0
                        and seen >= profile.warmup):
                    jax.profiler.start_trace(profile.directory)
                    tracing = True

                state, scale, loss, aux, finite = train_step(state, scale, batch)
                position = train.source_state
                current += 1
                seen += 1
                steps_since_log += 1
                interval_steps += 1
                book = bookkeep(book, loss, finite)
                if first_step is None:
                    loss.block_until_ready()
                    first_step = time.perf_counter() - started

                if tracing and profile is not None:
                    traced += 1
                    if traced == profile.steps:
                        tracing = False
                        self._stop_trace(traced, loss, profile)

                if current % log_every == 0:
                    interval_loss, _, worst_bad_run = book
                    self._check_finite(worst_bad_run, current)
                    book = (interval_loss, book[1], jnp.zeros((), jnp.int32))
                    if process_zero:
                        # The interval's numbers need the loss on the host, so
                        # this is where the loop waits on the device.
                        loss.block_until_ready()
                        now = time.time()
                        scalars = {"train/step": current, "train/loss": float(loss),
                                   **{f"train/{k}": float(v) for k, v in aux.items()},
                                   **self._throughput(now - last_log_time, steps_since_log,
                                                      data.batch)}
                        if self.rollout is not None:
                            scalars["train/rollout_seconds"] = rollout_seconds
                        print(f"step {current}: loss {scalars['train/loss']:.4f}")
                        if self.tracker is not None:
                            self.tracker.log(scalars, current)
                        last_log_time, steps_since_log, rollout_seconds = now, 0, 0.0

                if eval_every and current % eval_every == 0 and current < steps:
                    paused = time.perf_counter()
                    self._evaluate(state, data, metrics, mesh, current)
                    other += time.perf_counter() - paused

                # On its own clock, not the logging one: nested inside the log
                # tick, a cadence that did not divide log_every never fired at all.
                if (checkpoint_every and checkpoints is not None
                        and current % checkpoint_every == 0 and current < steps):
                    paused = time.perf_counter()
                    checkpoints.save(current, state, position,
                                     {"loss": float(book[0] / interval_steps)})
                    other += time.perf_counter() - paused
                    last_saved = current
                    book = (jnp.zeros((), jnp.float32), book[1], book[2])
                    interval_steps = 0
                if (local_every and checkpoints is not None
                        and current % local_every == 0 and current < steps):
                    paused = time.perf_counter()
                    checkpoints.save_local(current, state, position)
                    other += time.perf_counter() - paused

            paused = time.perf_counter()
            if train is not None:
                train.close()
                train = None
            other += time.perf_counter() - paused
            if tracing and profile is not None:
                tracing = False
                # The window outlived the run, and a trace left running takes the
                # next one down with it.
                self._stop_trace(traced, loss, profile)
            interval_loss, _, worst_bad_run = book
            self._check_finite(worst_bad_run, current)
            if loss is not None:
                # The last step has to land before the wall time is read.
                loss.block_until_ready()
            paused = time.perf_counter()
            if eval_every:
                self._evaluate(state, data, metrics, mesh, current)
            if checkpoints is not None and last_saved != current:
                # The in-loop saves are conditional, so the state the run ends on
                # may never have been written. It goes out under its real step,
                # because a step-0 checkpoint holding the final weights would make
                # a resume restart the schedule from the beginning.
                checkpoints.save(
                    current, state, position,
                    {"loss": float(interval_loss / interval_steps)} if interval_steps else None)
            other += time.perf_counter() - paused
        finally:
            paused = time.perf_counter()
            primary = sys.exception()
            error = primary
            close = train.close if train is not None else getattr(source, "close", None)
            stop_trace = None
            if tracing and profile is not None:
                tracing = False
                stop_trace = lambda: self._stop_trace(traced, loss, profile)
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
            if primary is None and error is not None:
                raise error
        if process_zero:
            scalars = goodput(time.perf_counter() - started, first_step, other)
            print(f"Goodput: first step after {scalars.get('goodput/time_to_first_step_s', 0.0):.2f} s, "
                  f"{scalars['goodput/step_fraction']:.1%} of the wall time in steps")
            if self.tracker is not None:
                self.tracker.log(scalars, current)
        return state

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _evaluate(self, state: TrainState, data: "Dataset", metrics: Sequence[Metric],
                  mesh, step: int) -> dict[str, float]:
        """Score the validation split: the objective's artifacts through the
        metrics and, on the first batch, to the tracker.

        Every process scores the batch count all of them have. The held-out
        split runs out at different points on different processes (the token
        and packed splits are whole files strided per process), and a process
        that left the pass while the others waited in its collectives would
        wedge the pool, so each batch is agreed before it is scored. A metric
        or a loader that raises takes the pass down with it.

        An artifact and the batch beside it come home before they are scored or
        drawn: a metric reads them with numpy and the tracker draws on process
        zero, neither of which can touch a shard of a global array. The gathers
        are collectives, so they happen here, on every process, and not inside
        the one metric or the one process that consumes them. Scoring the whole
        batch is also the only way the number means what it says: a process
        scoring its own shard would log its slice of the split as the metric.
        """
        if data.val is None:
            return {}
        process_zero = jax.process_index() == 0
        info = Step(step=state.step, key=jax.random.fold_in(state.key, state.step),
                    ema=with_ema(state.params, state.ema))
        values: dict[str, list] = {metric.name: [] for metric in metrics}
        iterator = iter(data.val())
        try:
            scored = 0
            while True:
                batch = next(iterator, None)
                if not minimum_across_processes(int(batch is not None)):
                    break
                if batch is None:
                    # Every process agreed one was available, so this cannot
                    # happen; leaving the loop here instead would strand the
                    # others in the next collective.
                    raise RuntimeError(
                        "the validation pass agreed a batch was available and this "
                        "process has none")
                batch = shard_batch(mesh, batch)
                produced = self.objective.evaluate(state.params, batch, info)
                produced = (() if produced is None
                            else produced if isinstance(produced, tuple) else (produced,))
                produced = tuple(host(artifact) for artifact in produced)
                if metrics:
                    # One gather for every metric, whatever fields they read.
                    home = host(batch)
                    for metric in metrics:
                        values[metric.name].append(metric(_pick(produced, metric.reads), home))
                if scored == 0 and process_zero and self.tracker is not None:
                    for artifact in produced:
                        self.tracker.artifact(artifact, step)
                scored += 1
            scores = {f"val/{name}": float(metric.reduce(values[name]))
                      for metric in metrics for name in [metric.name] if values[name]}
            if process_zero:
                print(f"Validation at step {step} over {scored} batches: {scores}")
                if self.tracker is not None and scores:
                    self.tracker.log(scores, step)
            return scores
        finally:
            primary = sys.exception()
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"Validation iterator cleanup failed: {error!r}")


    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _stop_trace(self, traced: int, loss, profile: Profile) -> None:
        """Close the profiler window once its last step has actually landed."""
        try:
            if loss is not None:
                loss.block_until_ready()
        finally:
            primary = sys.exception()
            try:
                jax.profiler.stop_trace()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"Profiler stop failed: {error!r}")
        print(f"Wrote profile for {traced} steps to {profile.directory}")

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

    def _throughput(self, elapsed: float, steps: int, batch: int) -> dict[str, float]:
        if elapsed <= 0 or steps <= 0:
            return {}
        step_time = elapsed / steps
        scalars = {"train/step_time_ms": step_time * 1000,
                   "train/samples_per_sec": batch / step_time}
        mfu = model_flops_utilization(self.flops_per_step, step_time)
        if mfu is not None:
            scalars["train/mfu"] = mfu
        return scalars
