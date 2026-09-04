"""The denoising diffusion objective.

Sample a noise level, corrupt, predict, weight. The convention (schedule,
parameterization, weighting) is the `Process`; the sample field and the
conditions are the `InputSpec`; every draw comes from the step's key. The
frozen encoders' weights live in the tree's `encoders` collection, so they
reach the compiled step as arguments and the optimizer never sees them.

Evaluation samples a few images from the validation batch's conditions with
the averaged weights, through the same `sample` inference uses.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.artifacts import ImageGrid, VideoGrid
from dew.diffusion.process import Process
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.inputs import InputSpec, unit_range
from dew.nn.autoencoders.api import AutoEncoder
from dew.objectives.base import Aux, EMASpec, Objective, Step, under
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM

# Samples a validation batch draws, conditioned or not.
VALIDATION_SAMPLES = 4


class DiffusionObjective(Objective):
    """Denoising diffusion: sample a noise level, corrupt, predict, weight."""

    def __init__(
        self,
        model,
        process: Process,
        inputs: InputSpec,
        *,
        autoencoder: Optional[AutoEncoder] = None,
        unconditional_prob: float = 0.12,
        loss_fn: Callable = optax.l2_loss,
        ema_decay: float = 0.999,
        sampler=DDIM(),
        guidance: Optional[CFG] = CFG(3.0),
        steps: int = 200,
    ):
        """`sampler`, `guidance` and `steps` are how evaluation samples;
        `guidance` None is the plain conditional prediction."""
        self.model = model
        self.process = process
        self.inputs = inputs
        self.autoencoder = autoencoder
        self.unconditional_prob = unconditional_prob
        self.loss_fn = loss_fn
        self.sampler = sampler
        self.guidance = guidance
        self.steps = steps
        self.ema = EMASpec(decay=optax.constant_schedule(ema_decay), select=under("params"))
        self.artifact = VideoGrid if len(inputs.sample.shape) == 4 else ImageGrid
        # The unconditional datum, tokenized once on the host; encoded on
        # device wherever a branch needs it.
        self.unconditional_tokens = {
            keyword: condition.encoder.tokenize([condition.unconditional])
            for keyword, condition in inputs.conditions.items()}
        self._sample = jax.jit(self._sample_impl, static_argnames=("count",))

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """The per-example shape the model denoises: the sample field's, or
        its latent when an autoencoder sits in front of the model."""
        shape = self.inputs.sample.shape
        if self.autoencoder is None:
            return shape
        *lead, height, width, _ = shape
        factor = self.autoencoder.downscale_factor
        return (*lead, height // factor, width // factor, self.autoencoder.latent_channels)

    def encoder_params(self) -> dict:
        return {keyword: condition.encoder.params
                for keyword, condition in self.inputs.conditions.items()}

    def encode(self, encoders, tokens: dict) -> dict:
        """Every condition's value from its tokens, under the tree's encoder
        parameters."""
        return {keyword: condition.encoder.encode(encoders[keyword], tokens[keyword])
                for keyword, condition in self.inputs.conditions.items()}

    def init(self, key):
        encoders = self.encoder_params()
        conditions = self.encode(encoders, self.unconditional_tokens)
        variables = self.model.init(
            key, jnp.ones((1, *self.latent_shape)), jnp.ones((1,)), **conditions)
        return {**variables, "encoders": encoders}

    def loss(self, params, batch, step: Step):
        data = unit_range(batch[self.inputs.sample.key])
        encode_key, drop_key, time_key, noise_key, dropout_key = jax.random.split(step.key, 5)
        if self.autoencoder is not None:
            data = self.autoencoder.encode(data, encode_key)
        count = data.shape[0]

        # Conditioning dropout: a row drawn for the unconditional branch reads
        # the unconditional value in every one of its conditions.
        dropped = jax.random.bernoulli(drop_key, self.unconditional_prob, (count,))
        tokens = {keyword: batch[condition.field]
                  for keyword, condition in self.inputs.conditions.items()}
        given = self.encode(params["encoders"], tokens)
        null = self.encode(params["encoders"], self.unconditional_tokens)
        conditions = jax.tree.map(
            lambda value, blank: jnp.where(
                expand(dropped, value), jnp.broadcast_to(blank, value.shape), value),
            given, null)

        schedule = self.process.schedule
        t = schedule.sample_t(time_key, count)
        noise = jax.random.normal(noise_key, data.shape, dtype=jnp.float32)
        rates = broadcast_rates(schedule, t, data)
        noisy, c_in, target = self.process.prediction.forward_diffusion(data, noise, rates)

        variables = {name: value for name, value in params.items() if name != "encoders"}
        preds = self.model.apply(
            variables, noisy * c_in, schedule.model_time(t), **conditions,
            train=True, rngs={"dropout": dropout_key})
        preds = self.process.prediction.pred_transform(noisy, preds, rates)
        losses = self.loss_fn(preds, target)
        weights = expand(self.process.weight(t), losses)
        return jnp.mean(losses * weights), Aux(metrics={})

    def _sample_impl(self, params, tokens, key, *, count: int):
        given = self.encode(params["encoders"], tokens)
        null = self.encode(params["encoders"], self.unconditional_tokens)
        variables = {name: value for name, value in params.items() if name != "encoders"}
        denoise = self.process.denoiser(
            self.model, variables, given, None if self.guidance is None else null)
        noise_key, sample_key = jax.random.split(key)
        x_T = self.process.noise(noise_key, (count, *self.latent_shape))
        samples = sample(denoise, x_T, self.steps, solver=self.sampler,
                         guidance=self.guidance, key=sample_key)
        if self.autoencoder is not None:
            samples = self.autoencoder.decode(samples)
        return jnp.clip(samples, -1.0, 1.0)

    def evaluate(self, params, batch, step: Step):
        """`VALIDATION_SAMPLES` samples from the batch's conditions, with the
        averaged weights when the run keeps them, seeded by the step's key."""
        params = params if step.ema is None else step.ema
        count = min(VALIDATION_SAMPLES, batch[self.inputs.sample.key].shape[0])
        tokens = {keyword: jax.tree.map(lambda value: value[:count], batch[condition.field])
                  for keyword, condition in self.inputs.conditions.items()}
        captions = ()
        for keyword, condition in self.inputs.conditions.items():
            captions = condition.encoder.captions(jax.tree.map(np.asarray, tokens[keyword]))
            if captions:
                break
        samples = self._sample(params, tokens, step.key, count=count)
        return self.artifact(samples, captions)
