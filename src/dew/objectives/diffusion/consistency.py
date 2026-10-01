"""Continuous-time consistency distillation (sCM) with distribution matching
(DMD2), together rCM.

sCM (Lu & Song 2025, "Simplifying, Stabilizing and Scaling Continuous-Time
Consistency Models") trains a student F on TrigFlow, x_t = cos(t) x_0 +
sin(t) z, toward the tangent of the teacher's probability-flow ODE: the
student's time derivative along the ODE comes from one JVP, and the loss
pulls F toward F plus the normalized tangent g. DMD2 (Yin et al. 2024,
"Improved Distribution Matching Distillation for Fast Image Synthesis")
moves the student's few-step samples along the difference of a fake score,
trained on those samples, and the teacher's. rCM (Zheng et al. 2025,
"Large Scale Diffusion Distillation via Score-Regularized Continuous-Time
Consistency") combines the two. The official code is NVlabs/rcm's
`T2VDistillModel_rCM`, which `tools/rcm_reference.py` runs.

Every network here is a rectified-flow velocity model on Dew's `Flow`
process; `trig_prediction` reads one on TrigFlow as rCM's
`RectifiedFlow_TrigFlowWrapper` does, at rf time sin t / (cos t + sin t).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.process import Process, aligned_conditions
from dew.diffusion.schedules import FlowMatchingScheduler, expand
from dew.diffusion.transforms import FlowMatchPredictionTransform
from dew.inputs import InputSpec, unit_range
from dew.nn.attention import forward_mode_attention
from dew.objectives.base import Aux, Mean, Step, Variables
from dew.registry import objectives
from dew.sampling.solvers import Consistency

from .objective import FAKE_SCORE, TEACHER, DiffusionObjective

Velocity = Callable[[jax.Array, jax.Array], jax.Array]
"""A rectified-flow velocity v(x, rf) at rf time."""



def trig_time(rf: jax.Array) -> jax.Array:
    """TrigFlow time of an rf time: arctan of sigma = rf / (1 - rf)."""
    return jnp.arctan(rf / (1 - rf))


def trig_prediction(velocity: Velocity, x: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(x_0, F) of a velocity model at TrigFlow `(x, t)`: the input and time
    scaled to rf, x_0 = x / (cos + sin) - rf v, and F = (cos x - x_0) / sin."""
    cos, sin = expand(jnp.cos(t), x), expand(jnp.sin(t), x)
    rf = jnp.sin(t) / (jnp.cos(t) + jnp.sin(t))
    scaled = x / (cos + sin)
    clean = scaled - expand(rf, x) * velocity(scaled, rf)
    return clean, (cos * x - clean) / sin


def guided(unconditional: jax.Array, conditional: jax.Array, scale: float) -> jax.Array:
    """The teacher's guided prediction, applied above a scale of one only."""
    return conditional if scale <= 1.0 else unconditional + scale * (conditional - unconditional)


def rows(value: jax.Array) -> jax.Array:
    """The sum over every axis but the batch."""
    return jnp.sum(value, axis=tuple(range(1, value.ndim)))


def consistency_loss(
    student: Callable[[jax.Array, jax.Array], jax.Array],
    x,
    t,
    teacher_F,
    warmup: float | jax.Array,
    scale: float,
) -> jax.Array:
    """sCM's per-row loss (`_student_scm_step`), scaled: the student's
    F-derivative along the teacher's ODE by one JVP, with tangents
    (cos t sin t F_teacher, cos t sin t), the tangent
    g = -cos t sqrt(1 - r^2 sin^2 t) (F_sg - F_teacher) - (r cos t sin t x + dF/dt)
    at warmup ratio r, normalized by its norm plus 0.1, and the loss
    ||F - F_sg - g||^2. `student(x, t)` is the student's F."""
    cos, sin = expand(jnp.cos(t), x), expand(jnp.sin(t), x)
    with forward_mode_attention():
        _, derivative = jax.jvp(student, (x, t), (cos * sin * teacher_F, jnp.cos(t) * jnp.sin(t)))
    derivative = jax.lax.stop_gradient(derivative)
    F = student(x, t)
    held = jax.lax.stop_gradient(F)
    g = (-cos * jnp.sqrt(1 - warmup ** 2 * sin ** 2) * (held - teacher_F)
         - (warmup * cos * sin * x + derivative))
    g = g / expand(jnp.sqrt(rows(jnp.square(g))) + 0.1, g)
    return scale * rows(jnp.square(F - held - g))


def discrete_consistency_loss(student: Callable[[jax.Array, jax.Array], jax.Array],
                              teacher: Callable[[jax.Array, jax.Array], jax.Array], x0, noise, u,
                              steps: int, skip: int, shift: float, scale: float) -> jax.Array:
    """rCM's discrete consistency (dCM) per-row loss (`_student_dcm_step`):
    on a grid of `steps` in shifted rf time, the student's x_0 at a point u
    against its stopped x_0 `skip` teacher Euler steps of F later.
    `student(x, t)` is the student's x_0 and `teacher(x, t)` the guided
    teacher's F, both at TrigFlow time."""
    def trig(k):
        s = 1.0 - (u + k / steps)
        rf = shift * s / (1 + (shift - 1) * s)
        rf = jnp.clip(rf, 0.0, 1.0 - jnp.finfo(rf.dtype).eps)
        return jnp.arctan(rf / (1 - rf))
    times = [trig(k) for k in range(skip + 1)]
    x = expand(jnp.cos(times[0]), x0) * x0 + expand(jnp.sin(times[0]), x0) * noise
    predicted = student(x, times[0])
    walked = x
    for current, following in itertools.pairwise(times):
        walked = walked - expand(current - following, walked) * teacher(walked, current)
    target = jax.lax.stop_gradient(student(jax.lax.stop_gradient(walked), times[-1]))
    return scale * rows(jnp.square(predicted - target))


def backward_simulation(student: Callable[[jax.Array, jax.Array], jax.Array], x_T, times, noises,
                        live: jax.Array | None = None) -> jax.Array:
    """The student's few-step sample from `x_T` at t = pi/2
    (`backward_simulation`): at each of `times`, a row's minimum with the
    time before, the clean prediction is noised again with the next of
    `noises`, and the last clean prediction, which alone keeps its
    gradient, is the sample. `student(x, t)` is the student's x_0. `live`
    marks the steps walked; the rest leave the state and the time as they
    are, so a traced count of steps runs over a fixed number of draws."""
    current = jnp.full(x_T.shape[:1], math.pi / 2, jnp.float32)
    x = x_T
    live = jnp.ones((len(times),), bool) if live is None else live
    for following, noise, active in zip(times, noises, live, strict=True):
        following = jnp.where(active, jnp.minimum(following, current), current)
        clean = jax.lax.stop_gradient(student(x, current))
        renoised = expand(jnp.cos(following), clean) * clean + expand(jnp.sin(following), clean) * noise
        x = jnp.where(active, renoised, x)
        current = following
    return student(x, current)


def distribution_matching_loss(generated, fake, teacher, scale: float) -> jax.Array:
    """DMD2's per-row generator loss (`_student_dmd_step`): the sample moved
    along fake - teacher, normalized by the mean distance to the teacher's
    prediction (at least 1e-5), as a regression onto the stopped target."""
    weight = jnp.maximum(jnp.mean(jnp.abs(generated - teacher), axis=tuple(range(1, generated.ndim)),
                                  keepdims=True), 1e-5)
    target = jax.lax.stop_gradient(generated - (fake - teacher) / weight)
    return scale * rows(jnp.square(generated - target))


def critic_loss(generated, fake, t) -> jax.Array:
    """The fake score's per-row denoising loss on the student's samples at
    TrigFlow `t`, over sin^2 t (`training_step_critic`)."""
    return rows(jnp.square(generated - fake) / expand(jnp.sin(t) ** 2, generated))


@objectives("rcm")
class ConsistencyDistillationObjective(DiffusionObjective):
    """rCM: sCM distillation of a flow teacher, regularized by DMD2.

    `teacher` is the teacher model's variables; the student and the fake
    score start from them. `consistency_weight` is sCM's loss scale (100,
    rCM's; 0 leaves DMD2 alone), `dmd_weight` DMD2's (1; 0 leaves sCM
    alone), and `teacher_guidance` the teacher's classifier-free scale in
    both. For the first `tangent_warmup` steps the tangent's warmup ratio
    rises from 0 to 1 and only the student trains; after them one step in
    `student_update_freq` trains the student and the rest the fake score, as
    rCM alternates its two optimizers. The student's DMD2 sample takes 1 to
    `max_simulation_steps` steps, cycling with the step count. Training
    times are rCM's log-normals in rf time, `student_times` for sCM and
    `critic_times` for DMD2 and the critic.

    `consistency` "discrete" trains rCM's discrete consistency (dCM, Song
    et al. 2023's consistency distillation) in place of sCM: the student's
    x_0 at a point of a `discrete_steps` grid in rf time shifted by
    `discrete_shift`, against its own stopped x_0 `discrete_skip` teacher
    Euler steps later.

    Where rCM steps one optimizer and leaves the other network's state as
    it was, here the idle network's gradient is zero for that step: an
    optimizer whose update moves on a zero gradient, such as Adam's momentum,
    still moves it. Sampling walks the student's multistep consistency
    sampler, `Consistency`.

    sCM's loss differentiates the student in time, so its time embedding
    must be smooth in it: `simple_dit(time_scale=0.002)`, which a run config
    sets for it, rather than the default 16.
    """

    def __init__(
        self,
        model: nn.Module,
        process: Process,
        inputs: InputSpec,
        *,
        teacher: Variables,
        consistency_weight: float = 100.0,
        dmd_weight: float = 1.0,
        teacher_guidance: float = 1.0,
        tangent_warmup: int = 0,
        student_update_freq: int = 5,
        max_simulation_steps: int = 4,
        student_times: tuple[float, float] = (-0.8, 1.6),
        critic_times: tuple[float, float] = (0.0, 1.6),
        consistency: Literal["continuous", "discrete"] = "continuous",
        discrete_steps: int = 48,
        discrete_skip: int = 1,
        discrete_shift: float = 5.0,
        **kwargs,
    ):
        schedule = process.schedule
        if not (isinstance(schedule, FlowMatchingScheduler) and schedule.shift == 1.0 and not process.interval
                and isinstance(process.prediction, FlowMatchPredictionTransform)):
            raise ValueError("rCM distills a velocity model on the unshifted linear path; build the "
                             "process with presets.Flow()")
        unused = sorted(
            key for key in ("uncertainty", "alignment", "end_to_end") if kwargs.get(key) is not None
        )
        if unused:
            raise ValueError(f"rCM trains on its own losses, which read none of {unused}")
        if consistency_weight <= 0 and dmd_weight <= 0:
            raise ValueError("rCM needs a consistency or a distribution-matching loss")
        kwargs.setdefault("guidance", None)
        kwargs.setdefault("sampler", Consistency())
        kwargs.setdefault("steps", 3)
        super().__init__(model, process, inputs, **kwargs)
        self.teacher = teacher
        self.consistency_weight = consistency_weight
        self.dmd_weight = dmd_weight
        self.teacher_guidance = teacher_guidance
        self.tangent_warmup = tangent_warmup
        self.student_update_freq = student_update_freq
        self.max_simulation_steps = max_simulation_steps
        self.student_times = student_times
        self.critic_times = critic_times
        if consistency not in ("continuous", "discrete"):
            raise ValueError(f"consistency is continuous (sCM) or discrete (dCM), not {consistency!r}")
        self.consistency = consistency
        self.discrete_steps = discrete_steps
        self.discrete_skip = discrete_skip
        self.discrete_shift = discrete_shift

    def held_variables(self) -> Variables:
        return {**super().held_variables(), TEACHER: self.teacher}

    def init(self, key, variables: Variables | None = None) -> Variables:
        state = dict(super().init(key, variables))
        if FAKE_SCORE in state["params"]:
            return state
        teacher = self.teacher
        # The student and the fake score start as the teacher, each in its own
        # buffers, since the step donates them.
        for collection, tree in teacher.items():
            state[collection] = {**state.get(collection, {}), **jax.tree.map(jnp.copy, tree),
                                 FAKE_SCORE: jax.tree.map(jnp.copy, tree)}
        state[TEACHER] = teacher
        return state

    def _network(self, variables: Variables, conditions) -> Velocity:
        schedule = self.process.schedule

        def velocity(x, rf):
            output = self.model.apply(variables, x, schedule.model_time(rf), **conditions)
            assert isinstance(output, jax.Array)
            return output
        return velocity

    def _fake(self, params) -> Variables:
        return {collection: tree[FAKE_SCORE] for collection, tree in params.items()
                if isinstance(tree, dict) and FAKE_SCORE in tree}

    def _teacher(self, params, given, blank, x, t) -> tuple[jax.Array, jax.Array]:
        teacher = jax.lax.stop_gradient(params[TEACHER])
        clean, F = trig_prediction(self._network(teacher, given), x, t)
        if self.teacher_guidance <= 1.0:
            return clean, F
        clean_u, F_u = trig_prediction(self._network(teacher, blank), x, t)
        return guided(clean_u, clean, self.teacher_guidance), guided(F_u, F, self.teacher_guidance)

    def _times(self, key, count, moments) -> jax.Array:
        mean, std = moments
        return trig_time(jnp.clip(jax.nn.sigmoid(mean + std * jax.random.normal(key, (count,))), 0.0, 1.0))

    def _generated(self, student_params, given, x_T, key, iteration) -> jax.Array:
        """The student's DMD2 sample: `max_simulation_steps` noisings drawn,
        the first `steps - 1` walked, where steps cycles with `iteration`."""
        steps = iteration % self.max_simulation_steps + 1
        time_key, noise_key = jax.random.split(key)
        count = x_T.shape[0]
        times = self._times(time_key, count * (self.max_simulation_steps - 1), self.critic_times)
        times = times.reshape(self.max_simulation_steps - 1, count)
        noises = jax.random.normal(noise_key, (self.max_simulation_steps - 1, *x_T.shape))
        # Steps past the drawn count leave the time where it is: from t = 0
        # nothing is noised again.
        live = jnp.arange(self.max_simulation_steps - 1) < steps - 1
        network = self._network(student_params, given)
        return backward_simulation(lambda x, t: trig_prediction(network, x, t)[0], x_T, times, noises, live)

    def loss(self, params, batch, step: Step):
        samples = unit_range(batch[self.inputs.sample.key])
        encode_key, drop_key, time_key, noise_key, generate_key = jax.random.split(step.key, 5)
        if self.autoencoder is not None:
            samples = self.autoencoder.encode(params["autoencoder"], samples, encode_key)
        count = samples.shape[0]
        given, unconditional = self._conditions(params, batch, drop_key, dropout=False)
        blank = jax.tree.map(lambda value, null: jnp.broadcast_to(null, value.shape),
                             given, aligned_conditions(given, unconditional))
        iteration = step.step
        warm = iteration < self.tangent_warmup
        student_phase = (self.dmd_weight <= 0) | warm | (
            (iteration - self.tangent_warmup) % self.student_update_freq == 0)
        effective = jnp.where(warm, iteration, self.tangent_warmup
                              + (iteration - self.tangent_warmup) // self.student_update_freq)
        def student_losses(params):
            student_params = self.trainable(params)
            total = jnp.zeros((count,), jnp.float32)
            if self.consistency_weight > 0 and self.consistency == "discrete":
                network = self._network(student_params, given)
                u = jax.random.uniform(time_key, (count,)) * (1 - self.discrete_skip / self.discrete_steps)
                total = total + discrete_consistency_loss(
                    lambda x, t: trig_prediction(network, x, t)[0],
                    lambda x, t: self._teacher(params, given, blank, x, t)[1],
                    samples, jax.random.normal(noise_key, samples.shape), u, self.discrete_steps,
                    self.discrete_skip, self.discrete_shift, self.consistency_weight)
            elif self.consistency_weight > 0:
                t = self._times(time_key, count, self.student_times)
                noise = jax.random.normal(noise_key, samples.shape)
                x = expand(jnp.cos(t), samples) * samples + expand(jnp.sin(t), samples) * noise
                _, teacher_F = self._teacher(params, given, blank, x, t)
                ratio = 1.0 if self.tangent_warmup == 0 else jnp.minimum(1.0, iteration / self.tangent_warmup)
                network = self._network(student_params, given)
                total = total + consistency_loss(lambda x, t: trig_prediction(network, x, t)[1], x, t,
                                                 teacher_F, ratio, self.consistency_weight)
            if self.dmd_weight > 0:
                distribution = self._distribution_matching(params, student_params, given, blank, generate_key,
                                                           effective, count, samples.shape)
                total = total + jnp.where(warm, 0.0, distribution)
            return total

        def critic_losses(params):
            generate, time_key, noise_key = jax.random.split(generate_key, 3)
            x_T = jax.random.normal(noise_key, samples.shape)
            generated = jax.lax.stop_gradient(self._generated(
                self.trainable(params), given, x_T, generate, iteration - effective - 1))
            t = self._times(time_key, count, self.critic_times)
            noise = jax.random.normal(jax.random.fold_in(noise_key, 1), samples.shape)
            x = expand(jnp.cos(t), generated) * generated + expand(jnp.sin(t), generated) * noise
            fake, _ = trig_prediction(self._network(self._fake(params), given), x, t)
            return critic_loss(generated, fake, t)

        losses = jax.lax.cond(student_phase, student_losses, critic_losses, params)
        return Mean(jnp.sum(losses), jnp.asarray(count, jnp.float32)), Aux(metrics={})

    def _distribution_matching(self, params, student_params, given, blank, key, iteration, count, shape):
        generate, time_key, noise_key = jax.random.split(key, 3)
        x_T = jax.random.normal(noise_key, shape)
        generated = self._generated(student_params, given, x_T, generate, iteration)
        t = self._times(time_key, count, self.critic_times)
        noise = jax.random.normal(jax.random.fold_in(noise_key, 1), shape)
        x = expand(jnp.cos(t), generated) * generated + expand(jnp.sin(t), generated) * noise
        fake, _ = trig_prediction(self._network(jax.lax.stop_gradient(self._fake(params)), given), x, t)
        teacher, _ = self._teacher(params, given, blank, x, t)
        return distribution_matching_loss(generated, jax.lax.stop_gradient(fake), teacher, self.dmd_weight)


__all__ = [
    "FAKE_SCORE",
    "TEACHER",
    "ConsistencyDistillationObjective",
    "backward_simulation",
    "consistency_loss",
    "critic_loss",
    "discrete_consistency_loss",
    "distribution_matching_loss",
    "guided",
    "trig_prediction",
    "trig_time",
]
