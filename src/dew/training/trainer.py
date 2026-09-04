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
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

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
from dew.objectives.base import Aux, Batch, Metric, Objective, Step, Variables, merge, select
from dew.telemetry.instrumentation import compiled_flops, model_flops_utilization
from dew.training.distributed import (
    Checkpointable, DevicePrefetchIterator, Layout, MeshSpec, Placement, batch_sharding, build_mesh,
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


@dataclasses.dataclass(frozen=True)
class Profile:
    """One profiler window per fit: `steps` steps traced into `directory`
    after `warmup` steps have run, so the trace holds the loop and not the
    compile."""
    directory: str
    steps: int
    warmup: int = 2


def with_ema(params: Variables, ema: Variables | None) -> Variables | None:
    """The variables tree with the averaged leaves in place of the live ones."""
    return None if ema is None else merge(params, ema)


def _project(tree: Variables, like: Variables) -> Variables:
    """The leaves of `tree` at the paths `like` holds, in `like`'s nesting."""
    return {name: _project(tree[name], child) if isinstance(child, Mapping) else tree[name]
            for name, child in like.items()}


def ema_update(ema: Variables, params: Variables,
               decay: jax.typing.ArrayLike) -> Variables:
    """One EMA step over the selected leaves; `decay` is the schedule's value."""
    return jax.tree.map(lambda average, live: decay * average + (1 - decay) * live,
                        ema, _project(params, ema))


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
        profile: Profile | None = None,
    ):
        """`accumulation` is the one owner of gradient accumulation: the
        optimizer is wrapped in `optax.MultiSteps` here, and the EMA runs on
        the update clock that wrapper defines. `step` replaces the compiled
        step's body with `step(objective, optimizer)`, the one place for an
        update that is not one loss (a GAN's alternating optimizers); it then
        owns the step counter, the EMA and the `Aux.variables` write-back,
        with `ema_update` and `write_back` at hand."""
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

    @property
    def batch_sharding(self) -> NamedSharding:
        """How a global batch is placed: split over every device."""
        return batch_sharding(self.device_mesh)

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
        print(f"Resumed from step {resume} in {checkpoints.directory}")
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

            updates, opt_state = optimizer.update(grads, state.opt_state, state.params["params"])
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
                scale: dynamic_scale_lib.DynamicScale | None = None) -> jax.stages.Compiled:
        """The training step's executable, the one `fit` runs, compiled ahead
        of time over `state` and one global `batch`.

        The executable takes `(state, scale, batch)` and returns `(state,
        scale, loss, metrics, finite)`; `scale` is the `DynamicScale` of a
        mixed-precision run and None otherwise. Its shardings are the layout's
        and it donates the state, so a benchmark that runs it measures the
        step a real run runs.

        Calling a jitted function compiles it, and asking the compiler for its
        cost analysis compiles it a second time: `lower(...).compile()` builds
        an executable of its own that the jit cache knows nothing about. Running
        the loop on that executable pays for one compilation and reads the FLOP
        count off it.
        """
        body = self._step_body()
        shardings = self.shardings(state)
        replicated = NamedSharding(self.device_mesh, P())

        def step(state, scale, batch):
            state, scale, loss, aux = body(state, scale, batch)
            return state, scale, loss, aux.metrics, jnp.isfinite(loss)

        jitted = jax.jit(
            step,
            in_shardings=(shardings, jax.tree.map(lambda _: replicated, scale),
                          self.batch_sharding),
            out_shardings=(shardings, jax.tree.map(lambda _: replicated, scale),
                           replicated, replicated, replicated),
            donate_argnums=(0,),
        )
        compiled = jitted.lower(state, scale, batch).compile()
        self.flops_per_step = compiled_flops(compiled)
        return compiled

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
        position are written.
        """
        mesh = self.device_mesh
        sharding = self.batch_sharding
        process_zero = jax.process_index() == 0
        state, shardings, position = self.place()
        current = int(state.step)
        if current > steps:
            raise ValueError(f"the run is at step {current}, past the {steps} asked for")

        profile, checkpoints = self.profile, self.checkpoints
        if checkpoint_every and checkpoints is None:
            raise ValueError(
                "checkpoint_every asks for checkpoints and this trainer has no "
                "checkpointer; pass Checkpoints(directory) to write any")
        source = data.train()
        if checkpoint_every and not isinstance(source, Checkpointable):
            raise ValueError(
                f"checkpoint_every needs a training stream with get_state and "
                f"set_state, and {type(source).__name__} lacks one; a checkpoint "
                f"written without the data position would replay the data on "
                f"resume. Train it with checkpoint_every=None "
                f"(--trainer.checkpoint-every None)")
        train = DevicePrefetchIterator(source, sharding, source_state=position)
        scale = dynamic_scale_lib.DynamicScale() if self.dynamic_scale else None
        compiled = None
        # Rebound the moment the executable exists, so the first tick measures
        # steps rather than the compile.
        last_log_time = time.time()
        last_saved = current if self.checkpoints is not None and current else None
        interval_loss, interval_steps = jnp.zeros((), jnp.float32), 0
        steps_since_log = 0
        # Both counters live on device so the loop never blocks on a result.
        # `worst_bad_run` remembers the longest streak of non-finite losses
        # seen since the last host check, which is what decides whether to stop.
        bad_run = jnp.zeros((), jnp.int32)
        worst_bad_run = jnp.zeros((), jnp.int32)
        tracing, traced, seen = False, 0, 0
        loss = None

        if process_zero:
            print(f"Training from step {current} to {steps} on "
                  f"{dict(mesh.shape)} ({jax.process_count()} process(es))")
        while current < steps:
            batch = next(train)
            if compiled is None:
                compiled = self.compile(state, batch, scale)
                # The interval clock starts once the executable exists, so the
                # first tick reports step time rather than compile time.
                last_log_time = time.time()
            if (self.profile is not None and not tracing and traced == 0
                    and seen >= self.profile.warmup):
                jax.profiler.start_trace(self.profile.directory)
                tracing = True

            state, scale, loss, aux, finite = compiled(state, scale, batch)
            position = train.source_state
            current += 1
            seen += 1
            steps_since_log += 1
            interval_loss = interval_loss + loss.astype(jnp.float32)
            interval_steps += 1
            bad_run = jnp.where(finite, 0, bad_run + 1)
            worst_bad_run = jnp.maximum(worst_bad_run, bad_run)

            if tracing and profile is not None:
                traced += 1
                if traced == profile.steps:
                    tracing = False
                    self._stop_trace(traced, loss, profile)

            if current % log_every == 0:
                self._check_finite(worst_bad_run, current)
                worst_bad_run = jnp.zeros((), jnp.int32)
                if process_zero:
                    # The one place per interval where waiting on the device
                    # is justified: the numbers below are meaningless without it.
                    loss.block_until_ready()
                    now = time.time()
                    scalars = {"train/step": current, "train/loss": float(loss),
                               **{f"train/{k}": float(v) for k, v in aux.items()},
                               **self._throughput(now - last_log_time, steps_since_log,
                                                  data.batch)}
                    print(f"step {current}: loss {float(loss):.4f}")
                    if self.tracker is not None:
                        self.tracker.log(scalars, current)
                    last_log_time, steps_since_log = now, 0

            if eval_every and current % eval_every == 0 and current < steps:
                self._evaluate(state, data, metrics, mesh, current)

            # On its own clock, not the logging one: nested inside the log
            # tick, a cadence that did not divide log_every never fired at all.
            if (checkpoint_every and checkpoints is not None
                    and current % checkpoint_every == 0 and current < steps):
                checkpoints.save(current, state, position,
                                 {"loss": float(interval_loss / interval_steps)})
                last_saved = current
                interval_loss, interval_steps = jnp.zeros((), jnp.float32), 0

        if tracing and profile is not None:
            # The window outlived the run, and a trace left running takes the
            # next one down with it.
            self._stop_trace(traced, loss, profile)
        self._check_finite(worst_bad_run, current)
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
        if checkpoints is not None:
            checkpoints.wait()
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
        sharding = self.batch_sharding
        process_zero = jax.process_index() == 0
        info = Step(step=state.step, key=jax.random.fold_in(state.key, state.step),
                    ema=with_ema(state.params, state.ema))
        values: dict[str, list] = {metric.name: [] for metric in metrics}
        iterator = iter(data.val())
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
            batch = shard_batch(sharding, batch)
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

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _stop_trace(self, traced: int, loss, profile: Profile) -> None:
        """Close the profiler window once its last step has actually landed."""
        loss.block_until_ready()
        jax.profiler.stop_trace()
        print(f"Wrote profile for {traced} steps to {profile.directory}")

    def _check_finite(self, worst_bad_run, step: int):
        """Fail a diverged run loudly rather than papering over it.

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
