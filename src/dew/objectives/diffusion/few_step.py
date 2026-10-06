"""Few-step generators trained from scratch: MeanFlow and shortcut models.

MeanFlow (Geng et al. 2025, "Mean Flows for One-step Generative Modeling")
trains a model of the average velocity u(z_t, r, t) over [r, t] through the
MeanFlow identity u = v - (t - r) du/dt, where one JVP takes the total
derivative along the flow. One step of the average velocity then crosses the
whole interval. The official code is Gsunshine/meanflow's `MeanFlow`, in
JAX, which `tools/meanflow_reference.py` runs.

Shortcut models (Frans et al. 2025, "One Step Diffusion via Shortcut
Models") train a velocity conditioned on its step size d. They train flow
matching at the smallest step, and on a fraction of the batch they train
self-consistency: one step of 2d equals two steps of d. The official code is
kvfrans/shortcut-models' `get_targets`, in JAX, which
`tools/shortcut_reference.py` runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.diffusion.presets import MeanFlow, Shortcut
from dew.diffusion.process import Process
from dew.diffusion.schedules import FlowMatchingScheduler, expand
from dew.diffusion.transforms import FlowMatchPredictionTransform, broadcast_rates
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.objectives.base import Aux, Step, Variables
from dew.registry import objectives, trainings
from dew.sampling.solvers import Euler

from .objective import DiffusionObjective, Training, _own_loss

if TYPE_CHECKING:
    from .config import DiffusionRunConfig

Velocity = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
"""An average velocity u(z, t, r) over [r, t]."""


def intervals(t: jax.Array, r: jax.Array, instantaneous: float) -> tuple[jax.Array, jax.Array]:
    """Return MeanFlow's (t, r) interval draw from two independent times.

    The later time is t and the earlier is r. The first `instantaneous`
    fraction of the rows takes r = t, the plain flow-matching case (the
    reference's `sample_tr`).
    """
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)
    count = t.shape[0]
    return t, jnp.where(jnp.arange(count) < int(count * instantaneous), t, r)


def guided_velocity(v, unconditional, conditional, omega, kappa):
    """Return MeanFlow's training-time guided target velocity (the reference's `guidance_fn`).

    It mixes the sample's v with the model's own unconditional and
    conditional velocities: omega v + (1 - omega - kappa) v_u + kappa v_c.
    With kappa 0 this is v_u + omega (v - v_u).
    """
    return omega * v + (1 - omega - kappa) * unconditional + kappa * conditional


def mean_flow_target(velocity: Velocity, z, t, r, v) -> tuple[jax.Array, jax.Array]:
    """Return the model's average velocity at (z, t, r) and its regression target v - (t - r) du/dt.

    One JVP takes the derivative along (dz/dt, dt/dt, dr/dt) = (v, 1, 0). No
    gradient flows through the target.
    """
    u, derivative = jax.jvp(velocity, (z, t, r), (v, jnp.ones_like(t), jnp.zeros_like(r)))
    target = v - expand(jnp.clip(t - r, 0.0, 1.0), derivative) * derivative
    return u, jax.lax.stop_gradient(target)


def adaptive_loss(u, target, power: float, epsilon: float) -> jax.Array:
    """Return MeanFlow's adaptively weighted loss for each row.

    Each row's squared error is summed over its entries and divided by the
    stopped (error + epsilon)^power. Power 0 gives the plain error, and
    power 1 gives a loss near one per row.
    """
    error = jnp.sum(jnp.square(u - target), axis=tuple(range(1, u.ndim)))
    return error / jax.lax.stop_gradient((error + epsilon) ** power)


def shortcut_levels(rows: int, sections: int) -> jax.Array:
    """Return the step levels of the self-consistency rows, where level l is a step of 2^-l.

    Each level gets `rows // log2(sections)` rows, starting with the finest
    level, log2(sections) - 1, and ending with level 0. The remaining rows are
    also at level 0, one step across the whole path (the reference's
    `get_targets`).
    """
    count = int(np.log2(sections))
    levels = jnp.repeat(count - 1 - jnp.arange(count), rows // count)
    return jnp.concatenate([levels, jnp.zeros(rows - levels.shape[0], levels.dtype)])


def shortcut_target(velocity: Velocity, x, sigma, step) -> jax.Array:
    """Return the self-consistency target for one step of size `step` from `x` at `sigma`.

    The target is the mean velocity of two steps of half that size, with each
    state clipped to [-4, 4] as the reference clips it.
    `velocity(x, sigma, sigma - step)` is the model's velocity over the
    interval to `sigma - step`.
    """
    half = step / 2
    first = velocity(x, sigma, sigma - half)
    midway = jnp.clip(x - expand(half, x) * first, -4, 4)
    second = velocity(midway, sigma - half, sigma - step)
    return jax.lax.stop_gradient(jnp.clip((first + second) / 2, -4, 4))


SMOOTH_TIME_SCALE = 0.002
"""The Fourier time scale a model trained through a derivative in time takes
when its config names none, and the fastest an sCM teacher's may turn
(`ConsistencyDistillation.check_teacher`). On a 2-D two-class toy (RTX
4080), one-step class accuracy at simple_dit's default 16 against 0.002 was
MeanFlow 23% against 99%, an sCM student 11% against 98.6%."""


@trainings("mean_flow")
@dataclasses.dataclass(frozen=True)
class MeanFlowTraining(Training):
    """MeanFlow training under the `MeanFlow` preset, for a model that samples in one step.

    `MeanFlowObjective` documents the fields. Sampling is unguided, because the
    guidance is trained in.
    """

    preset_class = MeanFlow
    guided = False

    instantaneous: float = 0.75
    omega: float = 1.0
    kappa: float = 0.0
    guidance_interval: tuple[float, float] = (0.0, 1.0)
    norm_p: float = 1.0
    norm_eps: float = 0.01

    def __post_init__(self) -> None:
        # A record carries the interval as a JSON list.
        start, stop = (float(edge) for edge in self.guidance_interval)
        object.__setattr__(self, "guidance_interval", (start, stop))

    def objective(self, run: DiffusionRunConfig, model: nn.Module, process: Process, inputs: InputSpec, *,
                  base: nn.Module, autoencoder: AutoEncoder | None,
                  variables: Variables | None) -> MeanFlowObjective:
        return MeanFlowObjective(model, process, inputs, self, autoencoder=autoencoder, variables=variables,
                                 unconditional_prob=run.unconditional_prob, ema_decay=run.ema_decay,
                                 solver=run.solver, guidance=None, steps=run.sampling_steps)


@trainings("shortcut")
@dataclasses.dataclass(frozen=True)
class ShortcutTraining(Training):
    """Shortcut-model training under the `Shortcut` preset.

    `ShortcutObjective` documents the fields, and `sections` must be a power of two,
    at least 2. Sampling is unguided.
    """

    preset_class = Shortcut
    guided = False

    sections: int = 128
    bootstrap_every: int = 8

    def __post_init__(self) -> None:
        if self.sections < 2 or self.sections & (self.sections - 1):
            raise ValueError(f"sections is a power of two, not {self.sections}")

    def objective(self, run: DiffusionRunConfig, model: nn.Module, process: Process, inputs: InputSpec, *,
                  base: nn.Module, autoencoder: AutoEncoder | None,
                  variables: Variables | None) -> ShortcutObjective:
        return ShortcutObjective(model, process, inputs, self, autoencoder=autoencoder, variables=variables,
                                 unconditional_prob=run.unconditional_prob, ema_decay=run.ema_decay,
                                 solver=run.solver, guidance=None, steps=run.sampling_steps)


@objectives("mean_flow")
class MeanFlowObjective(DiffusionObjective):
    """Trains MeanFlow on an interval process (`presets.MeanFlow`).

    `instantaneous` is the fraction of rows trained at r = t (the reference's
    `data_proportion`, 0.75). `omega` and `kappa` set the training-time
    guidance, applied where t lies in `guidance_interval`; omega 1 and kappa
    0 train without guidance. `norm_p` and `norm_eps` are the adaptive
    weighting's power and epsilon.

    The condition is dropped on `unconditional_prob` of the rows, and the
    target of a dropped row is the unguided v. As in the reference's
    `cond_drop`, the dropped rows are the first ones, as many as a draw at
    that rate counts. They therefore fall on the instantaneous rows, which is
    where the training-time guidance reads the unconditional velocity.
    Sampling takes `steps - 1` Euler steps of the average velocity, one by
    default. It uses no guidance, because the guidance is trained into the
    model.

    The loss differentiates the model with respect to time, so the model's
    time embedding must be smooth in time. A run config sets
    `simple_dit(time_scale=0.002)` for this, in place of the default 16.
    """

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec, mean_flow: MeanFlowTraining,
                 **kwargs):
        schedule = process.schedule
        if not (process.interval and isinstance(schedule, FlowMatchingScheduler) and schedule.shift == 1.0
                and isinstance(process.prediction, FlowMatchPredictionTransform)):
            raise ValueError("MeanFlow trains an interval model of velocity on the unshifted linear "
                             "path; build the process with presets.MeanFlow")
        _own_loss("MeanFlow", kwargs)
        kwargs.setdefault("guidance", None)
        kwargs.setdefault("solver", Euler())
        kwargs.setdefault("steps", 2)
        super().__init__(model, process, inputs, **kwargs)
        self.mean_flow = mean_flow

    def _draws(self, key, count: int, shape) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """The loss's randomness: two training times per row, the noise, and a
        uniform per row, whose count under `unconditional_prob` is how many
        rows drop their condition."""
        later, earlier, noise, dropping = jax.random.split(key, 4)
        schedule = self.process.schedule
        return (schedule.sample_t(later, count), schedule.sample_t(earlier, count),
                jax.random.normal(noise, shape, dtype=jnp.float32), jax.random.uniform(dropping, (count,)))

    def loss(self, variables, batch, step: Step):
        encode_key, condition_key, draw_key, dropout_key = jax.random.split(step.key, 4)
        samples = self.clean_samples(variables, batch, encode_key)
        count = samples.shape[0]
        schedule = self.process.schedule
        given, blank = self._conditions(variables, batch, condition_key, dropout=False)
        first, second, noise, uniform = self._draws(draw_key, count, samples.shape)
        t, r = intervals(first, second, self.mean_flow.instantaneous)
        z, _, v = self.process.prediction.forward_diffusion(
            samples, noise, broadcast_rates(schedule, t, samples)
        )
        variables = self.model_variables(variables)

        def velocity(conditions, *, train: bool) -> Velocity:
            def average(z, t, r) -> jax.Array:
                output = self.model.apply(variables, z, schedule.model_time(t), **conditions,
                                          duration=schedule.model_time(t) - schedule.model_time(r),
                                          train=train, rngs={"dropout": dropout_key})
                assert isinstance(output, jax.Array)
                return output
            return average

        if self.mean_flow.omega != 1.0 or self.mean_flow.kappa != 0.0:
            start, stop = self.mean_flow.guidance_interval
            inside = (t >= start) & (t <= stop)
            omega = expand(jnp.where(inside, self.mean_flow.omega, 1.0), v)
            kappa = expand(jnp.where(inside, self.mean_flow.kappa, 0.0), v)
            guided = jax.lax.stop_gradient(
                guided_velocity(
                    v,
                    velocity(blank, train=False)(z, t, t),
                    velocity(given, train=False)(z, t, t),
                    omega,
                    kappa,
                )
            )
        else:
            guided = v
        dropped = jnp.arange(count) < jnp.sum(uniform < self.unconditional_prob)
        conditions = jax.tree.map(lambda value, null: jnp.where(expand(dropped, value), null, value),
                                  given, blank)
        guided = jnp.where(expand(dropped, v), v, guided)
        u, target = mean_flow_target(velocity(conditions, train=True), z, t, r, guided)
        losses = adaptive_loss(u, target, self.mean_flow.norm_p, self.mean_flow.norm_eps)
        return self.row_mean(losses, batch), Aux(metrics={})


# The noise a shortcut model's path keeps at the data end, the reference's
# (1 - 1e-5) in x_t = (1 - (1 - 1e-5) t) x_0 + t x_1.
_NOISE_FLOOR = 1e-5


@objectives("shortcut")
class ShortcutObjective(DiffusionObjective):
    """Trains a shortcut model on an interval process (`presets.Shortcut`).

    `sections` is the finest grid, the reference's `denoise_timesteps` (128).
    Flow-matching rows train at one step of 1 / sections, on times of that
    grid. One row in `bootstrap_every` (8) trains self-consistency at a level
    from `shortcut_levels`, on times of that level's grid. Its target is two
    half steps of the EMA weights when the run keeps them. The condition is
    dropped on `unconditional_prob` of the flow-matching rows.

    The path is the reference's, whose noise never quite vanishes. The state
    at t is (1 - (1 - 1e-5) t) noise + t data, where Dew's sigma is 1 - t,
    and the velocity toward the noise is (1 - 1e-5) noise - data. Sampling
    takes `steps - 1` equal Euler steps with no guidance. Step counts that
    are powers of two up to `sections` are the ones the model trained at.
    """

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec, shortcut: ShortcutTraining,
                 **kwargs):
        schedule = process.schedule
        if not (process.interval and isinstance(schedule, FlowMatchingScheduler) and schedule.shift == 1.0
                and isinstance(process.prediction, FlowMatchPredictionTransform)):
            raise ValueError("a shortcut model is an interval model of velocity on the unshifted "
                             "linear path; build the process with presets.Shortcut")
        _own_loss("a shortcut model", kwargs)
        kwargs.setdefault("guidance", None)
        kwargs.setdefault("solver", Euler())
        kwargs.setdefault("steps", 2)
        super().__init__(model, process, inputs, **kwargs)
        self.shortcut = shortcut

    def _draws(self, key, count: int, grid: jax.Array, shape) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The loss's randomness: each row's time index below its `grid`, the
        noise, and whether a flow-matching row drops its condition."""
        times, noise, dropping = jax.random.split(key, 3)
        return (jax.random.randint(times, (count,), 0, grid.astype(jnp.int32)),
                jax.random.normal(noise, shape, dtype=jnp.float32),
                jax.random.bernoulli(dropping, self.unconditional_prob, (count,)))

    def loss(self, variables, batch, step: Step):
        encode_key, condition_key, draw_key, dropout_key = jax.random.split(step.key, 4)
        samples = self.clean_samples(variables, batch, encode_key)
        count = samples.shape[0]
        rows = count // self.shortcut.bootstrap_every
        schedule = self.process.schedule
        given, blank = self._conditions(variables, batch, condition_key, dropout=False)

        levels = shortcut_levels(rows, self.shortcut.sections)
        grid = jnp.concatenate([2.0 ** levels, jnp.full((count - rows,), float(self.shortcut.sections))])
        index, noise, dropping = self._draws(draw_key, count, grid, samples.shape)
        # Data at t = index / grid in the reference's time, on its path; Dew's sigma is 1 - t.
        t = index / grid
        sigma = 1 - t
        x = (1 - (1 - _NOISE_FLOOR) * expand(t, samples)) * noise + expand(t, samples) * samples
        v = (1 - _NOISE_FLOOR) * noise - samples
        step_size = jnp.concatenate([2.0 ** -levels, jnp.full((count - rows,), 1 / self.shortcut.sections)])
        dropped = (jnp.arange(count) >= rows) & dropping
        conditions = jax.tree.map(lambda value, null: jnp.where(expand(dropped, value), null, value),
                                  given, blank)

        def velocity(variables, conditions, *, train: bool) -> Velocity:
            def over(x, sigma, following) -> jax.Array:
                output = self.model.apply(
                    variables,
                    x,
                    schedule.model_time(sigma),
                    **conditions,
                    duration=schedule.model_time(sigma) - schedule.model_time(following),
                    train=train,
                    rngs={"dropout": dropout_key},
                )
                assert isinstance(output, jax.Array)
                return output
            return over

        teacher = self.model_variables(jax.lax.stop_gradient(variables if step.ema is None else step.ema))
        leading = jax.tree.map(lambda value: value[:rows], given)
        bootstrapped = shortcut_target(velocity(teacher, leading, train=False), x[:rows], sigma[:rows],
                                       step_size[:rows])
        target = jnp.concatenate([bootstrapped, v[rows:]])
        u = velocity(self.model_variables(variables), conditions, train=True)(x, sigma, sigma - step_size)
        losses = jnp.square(u - target)
        return self.row_mean(losses, batch), Aux(metrics={})


__all__ = ["MeanFlowObjective", "ShortcutObjective", "adaptive_loss", "guided_velocity", "intervals",
           "mean_flow_target", "shortcut_levels", "shortcut_target"]
