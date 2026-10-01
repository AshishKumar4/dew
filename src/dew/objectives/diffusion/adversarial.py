"""Adversarial diffusion distillation: LADD, with ADD's distillation term.

LADD (Sauer et al. 2024, "Fast High-Resolution Image Synthesis with Latent
Adversarial Diffusion Distillation") trains a few-step student against a
discriminator built on the frozen teacher's own features: the student's
one-step clean prediction and a real sample are both noised again to a
high noise level, the teacher reads each, and small heads on its hidden
tokens score them with the hinge loss. ADD (Sauer et al. 2023,
"Adversarial Diffusion Distillation") adds a distillation term that pulls
the student's prediction toward the teacher's denoising of it. Neither
paper publishes training code, so this follows their equations.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.process import Process
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.diffusion.transforms import FlowMatchPredictionTransform, broadcast_rates
from dew.inputs import InputSpec, unit_range
from dew.nn.blocks import FourierEmbedding
from dew.objectives.base import Aux, Mean, Step, Variables
from dew.registry import objectives
from dew.sampling.solvers import Consistency

from .objective import DISCRIMINATOR, TEACHER, DiffusionObjective


class Heads(nn.Module):
    """One discriminator head per teacher feature layer: each token's
    features through a hidden layer modulated by the noise level's
    embedding, to one logit per token."""

    layers: int
    width: int = 256

    @nn.compact
    def __call__(self, features: Sequence[jax.Array], time: jax.Array) -> list[jax.Array]:
        condition = FourierEmbedding(features=self.width)(time)
        logits = []
        for index, tokens in enumerate(features):
            hidden = nn.Dense(self.width, name=f"hidden_{index}")(tokens.astype(jnp.float32))
            hidden = hidden + nn.Dense(self.width, name=f"condition_{index}")(condition)[:, None, :]
            logits.append(nn.Dense(1, name=f"logit_{index}")(nn.leaky_relu(hidden, 0.2))[..., 0])
        return logits


def hinge_discriminator(real: Sequence[jax.Array], fake: Sequence[jax.Array]) -> jax.Array:
    """Each row's hinge loss, relu(1 - D(real)) + relu(1 + D(fake)), meaned
    over tokens and summed over the heads."""
    return jnp.sum(jnp.stack([jnp.mean(nn.relu(1 - r), axis=-1) + jnp.mean(nn.relu(1 + f), axis=-1)
                              for r, f in zip(real, fake, strict=True)]), axis=0)


def hinge_generator(fake: Sequence[jax.Array]) -> jax.Array:
    """Each row's generator loss, -D(fake), meaned over tokens and summed
    over the heads."""
    return -jnp.sum(jnp.stack([jnp.mean(f, axis=-1) for f in fake]), axis=0)


@objectives("ladd")
class AdversarialDistillationObjective(DiffusionObjective):
    """LADD, and ADD's distillation term at `distillation_weight` > 0.

    `teacher` is the teacher model's variables, frozen under `TEACHER`; the
    student starts from them, and the discriminator's heads read the
    teacher's hidden tokens after each of `feature_layers`. The student's
    time is drawn from `student_times` and the renoising level from
    LADD's logit-normal at `renoise_times` (mean 1, std 1: high noise). The
    discriminator and the student train in the same step, each through a
    loss whose other side is stopped. ADD's distillation term is
    weight * alpha_t ||x_0 - sg(teacher's x_0 of the renoised x_0)||^2,
    its exponential weighting. Sampling walks `Consistency`: one step from
    noise, or more renoised.
    """

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec, *, teacher: Variables,
                 feature_layers: Sequence[str], student_times: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
                 renoise_times: tuple[float, float] = (1.0, 1.0), distillation_weight: float = 0.0,
                 head_width: int = 256, **kwargs):
        schedule = process.schedule
        if not (isinstance(schedule, FlowMatchingScheduler) and not process.interval
                and isinstance(process.prediction, FlowMatchPredictionTransform)):
            raise ValueError("LADD distills a velocity model on the linear path; build the process with "
                             "presets.Flow()")
        unused = sorted(key for key in ("uncertainty", "alignment", "end_to_end") if kwargs.get(key) is not None)
        if unused:
            raise ValueError(f"LADD trains on its own losses, which read none of {unused}")
        if not feature_layers:
            raise ValueError("the discriminator reads the teacher's tokens after at least one layer")
        kwargs.setdefault("guidance", None)
        kwargs.setdefault("sampler", Consistency())
        kwargs.setdefault("steps", 2)
        super().__init__(model, process, inputs, **kwargs)
        self.teacher = teacher
        self.feature_layers = tuple(feature_layers)
        self.student_times = tuple(float(time) for time in student_times)
        self.renoise_times = renoise_times
        self.distillation_weight = distillation_weight
        self.heads = Heads(len(self.feature_layers), head_width)

    def held_variables(self) -> Variables:
        return {**super().held_variables(), TEACHER: self.teacher}

    def init(self, key, variables: Variables | None = None) -> Variables:
        state = dict(super().init(key, variables))
        if DISCRIMINATOR in state["params"]:
            return state
        # The student starts as the teacher, in buffers of its own, since the
        # step donates them.
        for collection, tree in self.teacher.items():
            state[collection] = {**state.get(collection, {}), **jax.tree.map(jnp.copy, tree)}
        state[TEACHER] = self.teacher
        given = jax.tree.map(lambda value: value[:1], self.unconditional_conditions)
        features = jax.eval_shape(lambda: self._features(self.teacher, jnp.zeros((1, *self.latent_shape)),
                                                         jnp.ones((1,)), given))
        heads = self.heads.init(jax.random.fold_in(key, 3), [jnp.zeros(f.shape, f.dtype) for f in features],
                                jnp.ones((1,)))
        for collection, value in heads.items():
            state[collection] = {**state.get(collection, {}), DISCRIMINATOR: value}
        return state

    def _features(self, teacher: Variables, x, t, conditions) -> list[jax.Array]:
        """The teacher's hidden tokens after each feature layer at `(x, t)`."""
        schedule = self.process.schedule
        _, captured = self.model.apply(teacher, x, schedule.model_time(t), **conditions,
                                       capture_intermediates=lambda module, method: (
                                           method == "__call__" and module.name in self.feature_layers),
                                       mutable=["intermediates"])
        kept = captured["intermediates"]
        missing = [name for name in self.feature_layers if name not in kept]
        if missing:
            raise ValueError(f"the teacher has no layers {missing} to read features after")
        return [jnp.asarray(kept[name]["__call__"][0]) for name in self.feature_layers]

    def _scores(self, heads: Variables, features: list[jax.Array], time: jax.Array) -> list[jax.Array]:
        scores = self.heads.apply(heads, features, time)
        assert isinstance(scores, list)
        return scores

    def loss(self, params, batch, step: Step):
        samples = unit_range(batch[self.inputs.sample.key])
        encode_key, drop_key, time_key, noise_key, renoise_key = jax.random.split(step.key, 5)
        if self.autoencoder is not None:
            samples = self.autoencoder.encode(params["autoencoder"], samples, encode_key)
        count = samples.shape[0]
        schedule = self.process.schedule
        conditions, _ = self._conditions(params, batch, drop_key, dropout=True)
        t = jnp.asarray(self.student_times)[jax.random.randint(time_key, (count,), 0, len(self.student_times))]
        noise = jax.random.normal(noise_key, samples.shape)
        noisy, _, _ = self.process.prediction.forward_diffusion(samples, noise,
                                                                 broadcast_rates(schedule, t, samples))
        student = self.process.denoiser(self.model, self.trainable(params), conditions)
        clean, _ = student(noisy, t)

        level_key, renoise_noise = jax.random.split(renoise_key)
        mean, std = self.renoise_times
        level = jax.nn.sigmoid(mean + std * jax.random.normal(level_key, (count,)))
        rates = broadcast_rates(schedule, level, samples)
        renoise = jax.random.normal(renoise_noise, samples.shape)
        teacher = jax.lax.stop_gradient(params[TEACHER])

        def renoised(x):
            return rates[0] * x + rates[1] * renoise

        real = self._features(teacher, renoised(samples), level, conditions)
        fake = self._features(teacher, renoised(clean), level, conditions)
        held = self._features(teacher, renoised(jax.lax.stop_gradient(clean)), level, conditions)
        heads = {collection: params[collection][DISCRIMINATOR] for collection in ("params", "constants")}
        frozen = jax.lax.stop_gradient(heads)
        time = schedule.model_time(level)
        discriminator = hinge_discriminator(self._scores(heads, real, time), self._scores(heads, held, time))
        generator = hinge_generator(self._scores(frozen, fake, time))
        total = discriminator + generator
        metrics = {"discriminator": jnp.mean(discriminator), "generator": jnp.mean(generator)}
        if self.distillation_weight > 0:
            target, _ = self.process.denoiser(self.model, self.trainable(teacher), conditions)(
                renoised(jax.lax.stop_gradient(clean)), level)
            distance = jnp.mean(jnp.square(clean - jax.lax.stop_gradient(target)),
                                axis=tuple(range(1, clean.ndim)))
            distillation = self.distillation_weight * schedule.rates(level)[0] * distance
            metrics["distillation"] = jnp.mean(distillation)
            total = total + distillation
        return Mean(jnp.sum(total), jnp.asarray(count, jnp.float32)), Aux(metrics=metrics)


__all__ = ["AdversarialDistillationObjective", "Heads", "hinge_discriminator", "hinge_generator"]
