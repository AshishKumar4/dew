"""Continuous-time consistency distillation (sCM) combined with distribution
matching (DMD2), which together make rCM.

sCM (Lu & Song 2025, "Simplifying, Stabilizing and Scaling Continuous-Time
Consistency Models") trains a student F on TrigFlow, x_t = cos(t) x_0 +
sin(t) z, toward the tangent of the teacher's probability-flow ODE. One JVP
gives the student's time derivative along the ODE, and the loss pulls F
toward F plus the normalized tangent g. DMD2 (Yin et al. 2024, "Improved
Distribution Matching Distillation for Fast Image Synthesis") moves the
student's few-step samples along the difference between a fake score,
trained on those samples, and the teacher's score. rCM (Zheng et al. 2025,
"Large Scale Diffusion Distillation via Score-Regularized Continuous-Time
Consistency") combines the two. The official code is NVlabs/rcm's
`T2VDistillModel_rCM`, which `tools/rcm_reference.py` runs.

Every network here is a rectified-flow velocity model on Dew's `Flow`
process. `trig_prediction` reads one on TrigFlow the way rCM's
`RectifiedFlow_TrigFlowWrapper` does, at rf time sin t / (cos t + sin t).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.diffusion.presets import Flow
from dew.diffusion.process import Process
from dew.diffusion.schedules import FlowMatchingScheduler, expand
from dew.inputs import InputSpec
from dew.nn.attention import forward_mode_attention
from dew.nn.protocols import TimeScaled
from dew.objectives.base import Aux, EMASpec, ProgramModule, Step, Variables
from dew.registry import models

from .few_step import SMOOTH_TIME_SCALE
from .objective import FAKE_SCORE, TEACHER, DiffusionObjective, Distillation, FlowDistillationObjective

Velocity = Callable[[jax.Array, jax.Array], jax.Array]
"""A rectified-flow velocity v(x, rf) at rf time."""


def trig_prediction(velocity: Velocity, x: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return (x_0, F) of a velocity model at TrigFlow `(x, t)`.

    The input and the time are first scaled to rf. Then
    x_0 = x / (cos + sin) - rf v, and F = (cos x - x_0) / sin.
    """
    cos, sin = expand(jnp.cos(t), x), expand(jnp.sin(t), x)
    rf = jnp.sin(t) / (jnp.cos(t) + jnp.sin(t))
    scaled = x / (cos + sin)
    clean = scaled - expand(rf, x) * velocity(scaled, rf)
    return clean, (cos * x - clean) / sin


def guided(unconditional: jax.Array, conditional: jax.Array, scale: float) -> jax.Array:
    """Return the teacher's guided prediction. Guidance applies only at a scale above one."""
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
    """Return sCM's per-row loss times `scale` (rCM's `_student_scm_step`).

    One JVP gives the student's F-derivative along the teacher's ODE, with
    tangents (cos t sin t F_teacher, cos t sin t). At warmup ratio r the
    tangent is

        g = -cos t sqrt(1 - r^2 sin^2 t) (F_sg - F_teacher) - (r cos t sin t x + dF/dt)

    normalized by its norm plus 0.1, and the loss is ||F - F_sg - g||^2.
    `student(x, t)` returns the student's F.
    """
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
    """Return rCM's discrete consistency (dCM) per-row loss (rCM's `_student_dcm_step`).

    On a grid of `steps` points in shifted rf time, the loss compares the
    student's x_0 at a point u with its own stopped x_0 after `skip` teacher
    Euler steps of F. `student(x, t)` returns the student's x_0 and
    `teacher(x, t)` the guided teacher's F, both at TrigFlow time.
    """
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
    """Return the student's few-step sample from `x_T`, starting at t = pi/2 (rCM's `backward_simulation`).

    At each time in `times`, capped per row at the time before it, the clean
    prediction is noised again with the next entry of `noises`. The last
    clean prediction is the sample, and only it keeps its gradient.
    `student(x, t)` returns the student's x_0. `live` marks the steps that
    run; the other steps leave the state and the time unchanged, so a traced
    step count can run over a fixed number of draws.
    """
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
    """Return DMD2's per-row generator loss (rCM's `_student_dmd_step`).

    The target is the sample moved along fake - teacher, normalized by the
    mean distance to the teacher's prediction (at least 1e-5). The loss
    regresses the sample onto that stopped target.
    """
    weight = jnp.maximum(jnp.mean(jnp.abs(generated - teacher), axis=tuple(range(1, generated.ndim)),
                                  keepdims=True), 1e-5)
    target = jax.lax.stop_gradient(generated - (fake - teacher) / weight)
    return scale * rows(jnp.square(generated - target))


def critic_loss(generated, fake, t) -> jax.Array:
    """Return the fake score's per-row denoising loss on the student's samples
    at TrigFlow `t`, divided by sin^2 t (rCM's `training_step_critic`)."""
    return rows(jnp.square(generated - fake) / expand(jnp.sin(t) ** 2, generated))


class _Draws(NamedTuple):
    """Every random value one rCM step reads. A student step reads the
    consistency draws and, past the warmup, the DMD2 ones; a critic step the
    DMD2 ones. Times are standard normals the log-normal time maps."""

    consistency_time: jax.Array
    """sCM's training time, a normal per row, or dCM's grid point, a uniform."""
    consistency_noise: jax.Array
    start: jax.Array
    """The student's sample's starting noise, x_T."""
    simulation_times: jax.Array
    """`max_simulation_steps - 1` rows of time normals, those past a step's count unread."""
    simulation_noises: jax.Array
    critic_time: jax.Array
    critic_noise: jax.Array


@dataclasses.dataclass(frozen=True)
class ConsistencyDistillation(Distillation):
    """rCM distillation of a saved flow run into a few-step student.

    rCM is sCM's consistency loss regularized by DMD2's, or either loss alone when
    the other's weight is 0. It trains under the `Flow` preset and samples unguided.
    `ConsistencyDistillationObjective` documents the other fields, and the student
    and the fake score start from the teacher's weights.
    """

    preset_class = Flow
    guided = False

    consistency_weight: float = 100.0
    dmd_weight: float = 1.0
    teacher_guidance: float = 1.0
    tangent_warmup: int = 0
    student_update_freq: int = 5
    max_simulation_steps: int = 4
    student_times: tuple[float, float] = (-0.8, 1.6)
    critic_times: tuple[float, float] = (0.0, 1.6)
    consistency: Literal["continuous", "discrete"] = "continuous"
    discrete_steps: int = 48
    discrete_skip: int = 1
    discrete_shift: float = 5.0

    def __post_init__(self) -> None:
        if self.consistency_weight <= 0 and self.dmd_weight <= 0:
            raise ValueError("rCM needs a consistency or a distribution-matching loss")
        if self.consistency not in ("continuous", "discrete"):
            raise ValueError(f"consistency is continuous (sCM) or discrete (dCM), not {self.consistency!r}")
        for name in ("student_times", "critic_times"):
            mean, std = (float(value) for value in getattr(self, name))
            object.__setattr__(self, name, (mean, std))

    def objective(self, base: nn.Module, variables: Variables | None, **run) -> DiffusionObjective:
        self.check_teacher()
        return ConsistencyDistillationObjective(distillation=self, teacher=base,
                                                teacher_variables=self.teacher_variables(variables),
                                                variables=variables, **run)

    def check_teacher(self) -> None:
        """Refuse sCM from a teacher whose time features turn too fast to differentiate in time.

        The student starts from the teacher's variables, its Fourier table included, and
        a `TimeScaled` model's derivative in time grows with its `time_scale`. For
        continuous consistency with a positive `consistency_weight`, this raises
        ValueError when the teacher run's model turns faster than `SMOOTH_TIME_SCALE`.
        """
        from .config import DiffusionRunConfig

        if self.consistency != "continuous" or self.consistency_weight <= 0:
            return
        teacher = DiffusionRunConfig.load(self.teacher)
        if teacher.pretrained is not None:  # a pipeline's denoiser is its source's, not `model`'s
            return
        model = models.build(teacher.model.name, teacher.model_fields(None))
        if isinstance(model, TimeScaled) and abs(model.time_scale) > SMOOTH_TIME_SCALE:
            raise ValueError(f"sCM differentiates the student in time, and its teacher's time features turn "
                             f"at time_scale={model.time_scale}, faster than {SMOOTH_TIME_SCALE}; train the "
                             f"teacher at time_scale={SMOOTH_TIME_SCALE}, or distill with dmd only "
                             "(consistency_weight=0)")


class ConsistencyDistillationObjective(FlowDistillationObjective):
    """Trains rCM: sCM distillation of a flow teacher, regularized by DMD2.

    `teacher` and `teacher_variables` are the teacher's model, which the fake
    score runs too, and its variables; the student and the fake score start
    from them. `consistency_weight` is sCM's loss scale (100, as
    in rCM; 0 trains DMD2 alone), `dmd_weight` is DMD2's (1; 0 trains sCM
    alone), and `teacher_guidance` is the teacher's classifier-free guidance
    scale in both losses.

    For the first `tangent_warmup` steps, the tangent's warmup ratio rises
    from 0 to 1 and only the student trains. After that, one step in
    `student_update_freq` trains the student and the others train the fake
    score, the way rCM alternates its two optimizers. Each network gets its
    own copy of the optimizer given to the trainer and steps only on its own
    updates (`optimizer`). The EMA averages only the student's updates, and
    its decay reads the student's own update count (`averages`); with
    `ema_decay=dew.training.posthoc.power_decay(0.1)` it is rCM's power EMA at
    rate 0.1. The student's DMD2 sample takes 1 to `max_simulation_steps`
    steps, cycling with the step count. Training times are rCM's log-normals
    in rf time: `student_times` for sCM, and `critic_times` for DMD2 and the
    critic.

    With `consistency="discrete"`, the objective trains rCM's discrete
    consistency (dCM, the consistency distillation of Song et al. 2023) in
    place of sCM. dCM compares the student's x_0 at a point of a
    `discrete_steps` grid in rf time, shifted by `discrete_shift`, with its
    own stopped x_0 `discrete_skip` teacher Euler steps later.

    The loss is the mean over rows, while rCM's trainer backpropagates their
    sum. Under Adam the two take the same steps when the epsilon is rCM's
    divided by the batch size. The phases alternate from one update to the
    next, so a trainer that accumulates more than one microbatch per update
    is refused. Sampling runs the student's multistep consistency solver,
    `Consistency`.

    sCM's loss differentiates the student with respect to time, so the
    student's time embedding must be smooth in time. A run config sets
    `simple_dit(time_scale=0.002)` for this, in place of the default 16.
    """

    network = FAKE_SCORE

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec,
                 distillation: ConsistencyDistillation, **kwargs):
        if isinstance(process.schedule, FlowMatchingScheduler) and process.schedule.shift != 1.0:
            raise ValueError("rCM distills a velocity model on the unshifted linear path; build the "
                             "process with presets.Flow()")
        kwargs.setdefault("steps", 3)
        super().__init__(model, process, inputs, **kwargs)
        self.fake_score = self.teacher
        self.distillation = distillation
        if self.ema is not None:
            decay = self.ema.decay
            self.ema = EMASpec(decay=lambda count: decay(self._effective(count)),
                               select=lambda path: path[0] == "params" and path[1:2] != (FAKE_SCORE,))

    def _student(self, iteration) -> jax.Array:
        """Whether update `iteration` trains the student (`is_student_phase`)."""
        options = self.distillation
        return ((options.dmd_weight <= 0) | (iteration < options.tangent_warmup)
                | ((iteration - options.tangent_warmup) % options.student_update_freq == 0))

    def _effective(self, iteration) -> jax.Array:
        """The student updates before update `iteration` (`get_effective_iteration`)."""
        if self.distillation.dmd_weight <= 0:
            return jnp.asarray(iteration)
        options = self.distillation
        return jnp.where(iteration < options.tangent_warmup, iteration, options.tangent_warmup
                         + (iteration - options.tangent_warmup) // options.student_update_freq)

    def optimizer(self, tx: optax.GradientTransformation, *,
                  accumulation: int) -> optax.GradientTransformation:
        """Return an optimizer that gives the student and the fake score each their own copy of `tx`.

        Each copy steps only on its own network's updates, as rCM's two
        optimizers do. With DMD2 on, an `accumulation` above one raises
        `ValueError`.
        """
        if accumulation > 1 and self.distillation.dmd_weight > 0:
            raise ValueError(
                "rCM alternates its student and fake-score updates update by update, so an update "
                "takes one microbatch: train with accumulation=1 and a larger batch")

        def network(params):
            return {name: FAKE_SCORE if name == FAKE_SCORE else "student" for name in params}

        def student(step, **_):
            return self._student(step)

        def critic(step, **_):
            return ~self._student(step)

        return optax.multi_transform({"student": optax.conditionally_mask(tx, student),
                                      FAKE_SCORE: optax.conditionally_mask(tx, critic)}, network)

    def averages(self, update: jax.Array) -> jax.Array:
        return self._student(update)

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The student, the frozen teacher, then the fake score, which trains."""
        return (*super().program_key(), ProgramModule(self.fake_score, None, trained=True))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        *student_and_teacher, self.fake_score = modules
        super().substitute(student_and_teacher)

    def network_variables(self, key: jax.Array, state: Variables) -> Variables:
        """The fake score, started as the teacher too."""
        return {collection: {FAKE_SCORE: jax.tree.map(jnp.copy, held)}
                for collection, held in state[TEACHER].items()}

    def _network(self, model: nn.Module, variables: Variables, conditions) -> Velocity:
        schedule = self.process.schedule

        def velocity(x, rf):
            output = model.apply(variables, x, schedule.model_time(rf), **conditions)
            assert isinstance(output, jax.Array)
            return output
        return velocity

    def _fake(self, params) -> Variables:
        return {collection: tree[FAKE_SCORE] for collection, tree in params.items()
                if isinstance(tree, dict) and FAKE_SCORE in tree}

    def _teacher(self, params, given, blank, x, t) -> tuple[jax.Array, jax.Array]:
        teacher = jax.lax.stop_gradient(params[TEACHER])
        clean, F = trig_prediction(self._network(self.teacher, teacher, given), x, t)
        if self.distillation.teacher_guidance <= 1.0:
            return clean, F
        clean_u, F_u = trig_prediction(self._network(self.teacher, teacher, blank), x, t)
        scale = self.distillation.teacher_guidance
        return guided(clean_u, clean, scale), guided(F_u, F, scale)

    @staticmethod
    def _times(normal, moments) -> jax.Array:
        """rCM's log-normal training time in rf time, as TrigFlow time, the arctan of rf / (1 - rf)."""
        mean, std = moments
        rf = jnp.clip(jax.nn.sigmoid(mean + std * normal), 0.0, 1.0)
        return jnp.arctan(rf / (1 - rf))

    def _draws(self, step: Step, count: int, shape) -> _Draws:
        """Every random value the step at `step` reads, from its key."""
        keys = jax.random.split(step.key, 7)
        simulated = self.distillation.max_simulation_steps - 1
        return _Draws(
            consistency_time=(jax.random.uniform(keys[0], (count,))
                              if self.distillation.consistency == "discrete"
                              else jax.random.normal(keys[0], (count,))),
            consistency_noise=jax.random.normal(keys[1], shape),
            start=jax.random.normal(keys[2], shape),
            simulation_times=jax.random.normal(keys[3], (simulated, count)),
            simulation_noises=jax.random.normal(keys[4], (simulated, *shape)),
            critic_time=jax.random.normal(keys[5], (count,)),
            critic_noise=jax.random.normal(keys[6], shape))

    def _generated(self, student_params, given, draws: _Draws, iteration) -> jax.Array:
        """The student's DMD2 sample: `max_simulation_steps` noisings drawn,
        the first `steps - 1` walked, where steps cycles with `iteration`."""
        steps = iteration % self.distillation.max_simulation_steps + 1
        times = self._times(draws.simulation_times, self.distillation.critic_times)
        # Steps past the drawn count leave the time where it is: from t = 0
        # nothing is noised again.
        live = jnp.arange(self.distillation.max_simulation_steps - 1) < steps - 1
        network = self._network(self.model, student_params, given)
        return backward_simulation(lambda x, t: trig_prediction(network, x, t)[0], draws.start, times,
                                   draws.simulation_noises, live)

    def loss(self, variables, batch, step: Step):
        options = self.distillation
        encode_key, drop_key = jax.random.split(jax.random.fold_in(step.key, 1))
        samples = self.clean_samples(variables, batch, encode_key)
        count = samples.shape[0]
        given, blank = self._conditions(variables, batch, drop_key, dropout=False)
        draws = self._draws(step, count, samples.shape)
        iteration = step.step
        warm = iteration < options.tangent_warmup
        student_phase = self._student(iteration)
        effective = self._effective(iteration)

        def student_losses(params):
            student_params = self.model_variables(params)
            total = jnp.zeros((count,), jnp.float32)
            if options.consistency_weight > 0 and options.consistency == "discrete":
                network = self._network(self.model, student_params, given)
                u = draws.consistency_time * (1 - options.discrete_skip / options.discrete_steps)
                total = total + discrete_consistency_loss(
                    lambda x, t: trig_prediction(network, x, t)[0],
                    lambda x, t: self._teacher(params, given, blank, x, t)[1],
                    samples, draws.consistency_noise, u, options.discrete_steps,
                    options.discrete_skip, options.discrete_shift, options.consistency_weight)
            elif options.consistency_weight > 0:
                t = self._times(draws.consistency_time, options.student_times)
                noise = draws.consistency_noise
                x = expand(jnp.cos(t), samples) * samples + expand(jnp.sin(t), samples) * noise
                _, teacher_F = self._teacher(params, given, blank, x, t)
                warmup = options.tangent_warmup
                ratio = 1.0 if warmup == 0 else jnp.minimum(1.0, iteration / warmup)
                network = self._network(self.model, student_params, given)
                total = total + consistency_loss(lambda x, t: trig_prediction(network, x, t)[1], x, t,
                                                 teacher_F, ratio, options.consistency_weight)
            if options.dmd_weight > 0:
                distribution = self._distribution_matching(params, student_params, given, blank, draws,
                                                           effective)
                total = total + jnp.where(warm, 0.0, distribution)
            return total

        def critic_losses(params):
            generated = jax.lax.stop_gradient(self._generated(
                self.model_variables(params), given, draws, iteration - effective - 1))
            t = self._times(draws.critic_time, options.critic_times)
            noise = draws.critic_noise
            x = expand(jnp.cos(t), generated) * generated + expand(jnp.sin(t), generated) * noise
            fake, _ = trig_prediction(self._network(self.fake_score, self._fake(params), given), x, t)
            return critic_loss(generated, fake, t)

        losses = jax.lax.cond(student_phase, student_losses, critic_losses, variables)
        return self.row_mean(losses, batch), Aux(metrics={})

    def _distribution_matching(self, params, student_params, given, blank, draws: _Draws, iteration):
        options = self.distillation
        generated = self._generated(student_params, given, draws, iteration)
        t = self._times(draws.critic_time, options.critic_times)
        noise = draws.critic_noise
        x = expand(jnp.cos(t), generated) * generated + expand(jnp.sin(t), generated) * noise
        frozen = jax.lax.stop_gradient(self._fake(params))
        fake, _ = trig_prediction(self._network(self.fake_score, frozen, given), x, t)
        teacher, _ = self._teacher(params, given, blank, x, t)
        return distribution_matching_loss(generated, jax.lax.stop_gradient(fake), teacher, options.dmd_weight)


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
]
