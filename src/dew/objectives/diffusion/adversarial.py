"""Adversarial diffusion distillation: LADD, with ADD's distillation term.

LADD (Sauer et al. 2024, "Fast High-Resolution Image Synthesis with Latent
Adversarial Diffusion Distillation") trains a few-step student against a
projected discriminator built on the frozen teacher. The student's clean
prediction and a real sample are both noised again, and the teacher reads
each. Independent heads on the teacher's token sequences after chosen
blocks score them with the hinge loss. The heads are StyleGAN-T's (Sauer et
al. 2023, autonomousvision/stylegan-t `networks/discriminator.py`
`DiscHead`), with LADD's change from 1D to 2D convolutions over the token
grid, and they are conditioned by projection on the noise level and the
pooled text. ADD (Sauer et al. 2023, "Adversarial Diffusion Distillation")
adds the R1 penalty on each head's input, with gamma 1e-5, and a
distillation term that pulls the student's prediction toward the teacher's
denoising of it, with lambda 2.5 and the exponential weighting alpha_t on
the squared L2 distance. Neither paper publishes its training code. The
heads follow StyleGAN-T's, which `tools/stylegan_t_reference.py` runs, and
the losses follow the papers' equations.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.presets import Flow
from dew.diffusion.process import Process
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.diffusion.transforms import FlowMatchPredictionTransform, broadcast_rates
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.nn.dit import TextContext, masked_mean
from dew.objectives.base import Aux, Batch, Objective, ProgramModule, Step, Variables
from dew.registry import objectives, trainings
from dew.sampling.solvers import Consistency

from .objective import (
    DISCRIMINATOR,
    SPECTRAL,
    TEACHER,
    DiffusionObjective,
    Training,
    _from_teacher,
    _own_loss,
    teacher_variables,
)

if TYPE_CHECKING:
    from .config import DiffusionRunConfig


def timestep_embedding(time: jax.Array, features: int) -> jax.Array:
    """DiT's sinusoidal embedding of a model time: cosines then sines at
    geometric frequencies from 1 to 1/10000."""
    half = features // 2
    frequencies = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    angles = jnp.asarray(time, jnp.float32)[:, None] * frequencies[None, :]
    return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)


class SpectralConv(nn.Module):
    """A convolution whose kernel is divided by its largest singular value,
    estimated by one step of power iteration as torch's `spectral_norm` does
    (`dim=0`: the kernel as an `[out, in * taps]` matrix). The iteration's
    `u` lives in the `spectral` collection and advances when `update`."""

    features: int
    kernel_size: tuple[int, int] = (1, 1)
    circular: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, update: bool) -> jax.Array:
        shape = (*self.kernel_size, x.shape[-1], self.features)
        fan_in = math.prod(shape[:-1])
        bound = 1 / math.sqrt(fan_in)
        kernel = self.param("kernel", lambda key: jax.random.uniform(key, shape, minval=-bound, maxval=bound))
        bias = self.param("bias", lambda key: jax.random.uniform(
            key, (self.features,), minval=-bound, maxval=bound))
        u = self.variable(SPECTRAL, "u", lambda: _normalized(jax.random.normal(self.make_rng("params"),
                                                                                (self.features,))))
        # torch's [out, in, kh, kw] flattened per output channel.
        matrix = jnp.transpose(kernel, (3, 2, 0, 1)).reshape(self.features, -1)
        held = jax.lax.stop_gradient(matrix)
        v = _normalized(held.T @ u.value)
        following = _normalized(held @ v)
        sigma = following @ (matrix @ v)
        if update and not self.is_initializing():
            u.value = following
        pad = [(size // 2, size // 2) for size in self.kernel_size]
        if self.circular:
            x = jnp.pad(x, [(0, 0), *pad, (0, 0)], mode="wrap")
            pad = [(0, 0), (0, 0)]
        y = jax.lax.conv_general_dilated(x, (kernel / sigma).astype(x.dtype), (1, 1), pad,
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
        return y + bias


def _normalized(x: jax.Array, eps: float = 1e-12) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x), eps)


class BatchNormLocal(nn.Module):
    """StyleGAN-T's `BatchNormLocal`: each channel normalized over the
    positions and the samples of groups of `virtual_batch`, with an affine
    map and no running statistics. A group's samples are `batch`'s rows
    (`Objective.row_mean`)."""

    virtual_batch: int = 8
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jax.Array, batch: Batch) -> jax.Array:
        rows, height, width, channels = x.shape
        groups = math.ceil(rows / self.virtual_batch)
        grouped = x.reshape(groups, -1, height * width, channels)

        def group_mean(values: jax.Array) -> jax.Array:
            mean, _ = Objective.row_mean(values, batch, (1, 2), rows=(0, 1)).mean()
            return jnp.expand_dims(mean, (1, 2))

        mean = group_mean(grouped)
        var = group_mean(jnp.square(grouped - mean))
        normalized = ((grouped - mean) / jnp.sqrt(var + self.eps)).reshape(x.shape)
        weight = self.param("weight", nn.initializers.ones, (channels,))
        bias = self.param("bias", nn.initializers.zeros, (channels,))
        return normalized * weight + bias


class Block(nn.Module):
    """StyleGAN-T's `make_block`: a spectral convolution with circular
    padding, the local batch norm and a leaky ReLU."""

    features: int
    kernel_size: tuple[int, int]

    @nn.compact
    def __call__(self, x: jax.Array, update: bool, batch: Batch) -> jax.Array:
        y = SpectralConv(self.features, self.kernel_size, name="conv")(x, update)
        return nn.leaky_relu(BatchNormLocal(name="norm")(y, batch), 0.2)


class Head(nn.Module):
    """StyleGAN-T's `DiscHead` on a `[B, H, W, C]` token grid: a 1x1 block,
    a residual block of `kernel_size` scaled by 1/sqrt(2), a 1x1 spectral
    convolution to `cmap_dim` channels, and the projection onto the
    condition's map, summed over channels over sqrt(cmap_dim): one logit per
    position. The condition's map is StyleGAN-T's `FullyConnectedLayer`, a
    dense layer whose weight is scaled by 1/sqrt(its input width)."""

    cmap_dim: int = 64
    kernel_size: tuple[int, int] = (9, 9)

    @nn.compact
    def __call__(self, x: jax.Array, condition: jax.Array, update: bool, batch: Batch) -> jax.Array:
        channels = x.shape[-1]
        h = Block(channels, (1, 1), name="block_0")(x, update, batch)
        h = (Block(channels, self.kernel_size, name="block_1")(h, update, batch) + h) / math.sqrt(2)
        out = SpectralConv(self.cmap_dim, (1, 1), circular=False, name="cls")(
            h, update)
        weight = self.param("cmapper_weight", nn.initializers.normal(1.0),
                            (condition.shape[-1], self.cmap_dim))
        cmap_bias = self.param("cmapper_bias", nn.initializers.zeros, (self.cmap_dim,))
        cmap = condition @ (weight / math.sqrt(condition.shape[-1])) + cmap_bias
        return jnp.sum(out * cmap[:, None, None, :], axis=-1) / math.sqrt(self.cmap_dim)


class Heads(nn.Module):
    """One `Head` per teacher feature layer, conditioned on the noise level's
    embedding and the pooled text."""

    layers: int
    cmap_dim: int = 64
    kernel_size: tuple[int, int] = (9, 9)

    @nn.compact
    def __call__(self, features: Sequence[jax.Array], condition: jax.Array, update: bool,
                 batch: Batch) -> list[jax.Array]:
        return [Head(self.cmap_dim, self.kernel_size, name=f"head_{index}")(grid, condition, update, batch)
                for index, grid in enumerate(features)]


def hinge_discriminator(real: Sequence[jax.Array], fake: Sequence[jax.Array]) -> jax.Array:
    """Each row's hinge loss, relu(1 - D(real)) + relu(1 + D(fake)), meaned
    over every head's every logit as StyleGAN-T's concatenated logits are."""
    return (jnp.mean(nn.relu(1 - _logits(real)), axis=-1) + jnp.mean(nn.relu(1 + _logits(fake)), axis=-1))


def hinge_generator(fake: Sequence[jax.Array]) -> jax.Array:
    """Each row's generator loss, -D(fake), meaned over every logit."""
    return -jnp.mean(_logits(fake), axis=-1)


def _logits(scores: Sequence[jax.Array]) -> jax.Array:
    return jnp.concatenate([score.reshape(score.shape[0], -1) for score in scores], axis=-1)


def r1_penalty(score, features: Sequence[jax.Array], batch: Batch) -> jax.Array:
    """ADD's R1, computed on each head's input: per row, the summed squared
    gradient of that head's mean logit with respect to its input features.
    `score(features)` is the list of each head's logits, summed over
    `batch`'s rows (`Objective.row_mean`)."""
    def total(features):
        return sum(Objective.row_mean(jnp.mean(head.reshape(head.shape[0], -1), axis=-1), batch).total
                   for head in score(features))
    gradients = jax.grad(total)(list(features))
    return jnp.sum(jnp.stack([
        jnp.sum(jnp.square(g).reshape(g.shape[0], -1), axis=-1) for g in gradients]), axis=0)


@trainings("ladd")
@dataclasses.dataclass(frozen=True)
class AdversarialDistillation(Training):
    """Adversarial distillation of a saved flow run into a few-step student.

    This is LADD with ADD's R1 penalty and distillation term. It trains under the
    `Flow` preset and samples unguided. `AdversarialDistillationObjective` documents
    the fields, and `feature_layers` must name at least one layer.

    `teacher` is the teacher run's directory, which a run loads; a run without one
    is refused. The teacher's model is this run's `model` without the run's adapter.
    An objective built in code is given the teacher's model and weights directly.
    """

    preset_class = Flow
    guided = False

    teacher: str = ""
    feature_layers: tuple[str, ...] = ()
    student_times: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    renoise_times: tuple[float, float] = (1.0, 1.0)
    distillation_weight: float = 2.5
    r1_weight: float = 1e-5
    cmap_dim: int = 64
    kernel_size: tuple[int, int] = (9, 9)

    def __post_init__(self) -> None:
        if not self.feature_layers:
            raise ValueError("the discriminator reads the teacher's tokens after at least one layer")
        object.__setattr__(self, "feature_layers", tuple(self.feature_layers))
        object.__setattr__(self, "student_times", tuple(float(time) for time in self.student_times))
        mean, std = (float(value) for value in self.renoise_times)
        object.__setattr__(self, "renoise_times", (mean, std))
        height, width = (int(size) for size in self.kernel_size)
        object.__setattr__(self, "kernel_size", (height, width))

    def check(self, run: DiffusionRunConfig) -> None:
        super().check(run)
        if not self.teacher:
            raise ValueError("adversarial distillation distills a teacher; name its run directory")

    def objective(self, run: DiffusionRunConfig, model: nn.Module, process: Process, inputs: InputSpec, *,
                  base: nn.Module, autoencoder: AutoEncoder | None,
                  variables: Variables | None) -> AdversarialDistillationObjective:
        return AdversarialDistillationObjective(
            model, process, inputs, self, teacher=base,
            teacher_variables=teacher_variables(self.teacher, variables), autoencoder=autoencoder,
            variables=variables, unconditional_prob=run.unconditional_prob, ema_decay=run.ema_decay,
            solver=run.solver, guidance=None, steps=run.sampling_steps)


@objectives("ladd")
class AdversarialDistillationObjective(DiffusionObjective):
    """Trains LADD, with ADD's R1 penalty and its distillation term.

    `teacher` is the teacher's model and `teacher_variables` its variables,
    frozen under `TEACHER`. The student starts from them, and a `Head` reads
    the teacher's token grid after each layer in `feature_layers`. The
    student's time is drawn from `student_times`, and the renoising level
    from LADD's logit-normal with `renoise_times` (mean 1, std 1, the
    high-noise setting the paper uses for images).

    The discriminator and the student train in the same step, each through
    its own loss with the other stopped. StyleGAN-T, whose heads these are,
    alternates a generator step and a discriminator step, and neither LADD
    nor ADD says which it does. The heads' spectral norms take one power
    iteration per step. The real pass iterates and keeps its `u`; the held,
    fake and R1 passes iterate from that `u` and keep nothing, while
    StyleGAN-T keeps the `u` of each training pass.

    The discriminator adds `r1_weight` (ADD's gamma, 1e-5) times R1 on its
    real inputs. The student adds `distillation_weight` (ADD's lambda, 2.5)
    times alpha_t ||x_0 - sg(teacher's x_0 of the renoised x_0)||^2, summed
    over the sample. LADD itself drops that term when it trains on synthetic
    data, and on CIFAR-10 at 32 pixels the term at 2.5 dominated and the
    student did better without it. Sampling runs `Consistency`.

    The teacher never runs through the student's model, so a LoRA-adapted
    student trains its factors alone.
    """

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec,
                 distillation: AdversarialDistillation, *, teacher: nn.Module, teacher_variables: Variables,
                 time_features: int = 256, **kwargs):
        schedule = process.schedule
        if not (isinstance(schedule, FlowMatchingScheduler) and not process.interval
                and isinstance(process.prediction, FlowMatchPredictionTransform)):
            raise ValueError("LADD distills a velocity model on the linear path; build the process with "
                             "presets.Flow()")
        _own_loss("LADD", kwargs)
        kwargs.setdefault("solver", Consistency())
        kwargs.setdefault("steps", 2)
        super().__init__(model, process, inputs, **kwargs)
        self.teacher = teacher
        self.teacher_variables = teacher_variables
        self.distillation = distillation
        self.time_features = time_features
        self.heads = Heads(len(distillation.feature_layers), distillation.cmap_dim, distillation.kernel_size)

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The student, then the frozen teacher the discriminator reads."""
        return (*super().program_key(), ProgramModule(self.teacher, None, trained=False))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        self.model, self.teacher = modules

    def held_variables(self) -> Variables:
        return {**super().held_variables(), TEACHER: self.teacher_variables}

    def complete_variables(self, key: jax.Array, tree: Variables) -> Variables:
        """Start the student as the tree's teacher and draw the heads, unless it holds them."""
        state = super().complete_variables(key, tree)
        if DISCRIMINATOR in state["params"]:
            return state
        teacher = state[TEACHER]
        state = _from_teacher(state, teacher)
        given = jax.tree.map(lambda value: value[:1], self.unconditional_conditions)
        x = jnp.zeros((1, *self.latent_shape))
        features = jax.eval_shape(lambda: self._features(teacher, x, jnp.ones((1,)), given))
        condition = self._condition(jnp.ones((1,)), given)
        heads = self.heads.init(jax.random.fold_in(key, 3), [jnp.zeros(f.shape, f.dtype) for f in features],
                                condition, update=False, batch={})
        state["params"] = {**state["params"], DISCRIMINATOR: heads["params"]}
        state[SPECTRAL] = heads[SPECTRAL]
        return state

    def _condition(self, time: jax.Array, conditions) -> jax.Array:
        """The heads' condition: the noise level's sinusoidal embedding and,
        where the model reads text, its pooled states."""
        parts = [timestep_embedding(time, self.time_features)]
        parts += [masked_mean(value.hidden, value.mask).astype(jnp.float32)
                  for value in conditions.values() if isinstance(value, TextContext)]
        return jnp.concatenate(parts, axis=-1)

    def _features(self, teacher: Variables, x, t, conditions) -> list[jax.Array]:
        """The teacher's token grid after each feature layer at `(x, t)`."""
        schedule = self.process.schedule
        layers = self.distillation.feature_layers
        _, captured = self.teacher.apply(teacher, x, schedule.model_time(t), **conditions,
                                         capture_intermediates=lambda module, method: (
                                             method == "__call__" and module.name in layers),
                                         mutable=["intermediates"])
        kept = captured["intermediates"]
        missing = [name for name in self.distillation.feature_layers if name not in kept]
        if missing:
            raise ValueError(f"the teacher has no layers {missing} to read features after")
        grids = []
        for name in self.distillation.feature_layers:
            tokens = jnp.asarray(kept[name]["__call__"][0])
            side = math.isqrt(tokens.shape[1])
            if side * side != tokens.shape[1]:
                raise ValueError(f"LADD's 2D heads read a square token grid, not {tokens.shape[1]} tokens")
            grids.append(tokens.reshape(tokens.shape[0], side, side, tokens.shape[-1]).astype(jnp.float32))
        return grids

    def loss(self, variables, batch, step: Step):
        encode_key, drop_key, time_key, noise_key, renoise_key = jax.random.split(step.key, 5)
        samples = self.clean_samples(variables, batch, encode_key)
        count = samples.shape[0]
        schedule = self.process.schedule
        assert self.process.prediction is not None
        conditions, _ = self._conditions(variables, batch, drop_key, dropout=True)
        indices = jax.random.randint(time_key, (count,), 0, len(self.distillation.student_times))
        t = jnp.asarray(self.distillation.student_times)[indices]
        noise = jax.random.normal(noise_key, samples.shape)
        noisy, _, _ = self.process.prediction.forward_diffusion(samples, noise,
                                                                 broadcast_rates(schedule, t, samples))
        student = self.process.denoiser(self.model, self.model_variables(variables), conditions)
        clean, _ = student(noisy, t)

        level_key, renoise_noise = jax.random.split(renoise_key)
        mean, std = self.distillation.renoise_times
        level = jax.nn.sigmoid(mean + std * jax.random.normal(level_key, (count,)))
        rates = broadcast_rates(schedule, level, samples)
        renoise = jax.random.normal(renoise_noise, samples.shape)
        teacher = jax.lax.stop_gradient(variables[TEACHER])

        def renoised(x):
            return rates[0] * x + rates[1] * renoise

        real = self._features(teacher, renoised(samples), level, conditions)
        fake = self._features(teacher, renoised(clean), level, conditions)
        held = self._features(teacher, renoised(jax.lax.stop_gradient(clean)), level, conditions)
        condition = jax.lax.stop_gradient(self._condition(schedule.model_time(level), conditions))
        heads = variables["params"][DISCRIMINATOR]

        def score(head_params, spectral, features, *, update=False):
            scores, updated = self.heads.apply(
                {"params": head_params, SPECTRAL: spectral}, features, condition,
                update=update, batch=batch, mutable=[SPECTRAL])
            return scores, updated[SPECTRAL]

        real_scores, spectral = score(heads, variables[SPECTRAL], real, update=True)
        held_scores, _ = score(heads, spectral, held)
        fake_scores, _ = score(jax.lax.stop_gradient(heads), spectral, fake)
        discriminator = hinge_discriminator(real_scores, held_scores)
        generator = hinge_generator(fake_scores)
        total = discriminator + generator
        metrics = {"discriminator": jnp.mean(discriminator), "generator": jnp.mean(generator)}
        if self.distillation.r1_weight > 0:
            r1 = r1_penalty(lambda features: score(heads, spectral, features)[0],
                            [jax.lax.stop_gradient(f) for f in real], batch)
            metrics["r1"] = jnp.mean(r1)
            total = total + self.distillation.r1_weight * r1
        if self.distillation.distillation_weight > 0:
            target, _ = self.process.denoiser(self.teacher, teacher, conditions)(
                renoised(jax.lax.stop_gradient(clean)), level)
            distance = jnp.sum(jnp.square(clean - jax.lax.stop_gradient(target)),
                               axis=tuple(range(1, clean.ndim)))
            distillation = self.distillation.distillation_weight * schedule.rates(level)[0] * distance
            metrics["distillation"] = jnp.mean(distillation)
            total = total + distillation
        return (self.row_mean(total, batch),
                Aux(metrics=metrics, variables={SPECTRAL: jax.lax.stop_gradient(spectral)}))


__all__ = ["AdversarialDistillationObjective"]
