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

from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.artifacts import ImageGrid, VideoGrid, agree_process_phase, collective_host
from dew.diffusion.process import Process
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.inputs import InputSpec, unit_range
from dew.nn.autoencoders import AutoEncoder
from dew.objectives.base import Aux, EMASpec, Mean, Objective, Step, under
from dew.registry import objectives
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM, Solver

# Samples a validation batch draws, conditioned or not.
VALIDATION_SAMPLES = 4


def check_solver(process, sampler, steps: int) -> None:
    """Trace a solver step on the objective's actual sampling grid."""
    x = jnp.zeros((1, 1), jnp.float32)
    with jax.ensure_compile_time_eval():
        times = process.times(steps)
    state = sampler.init(x, times, process)
    if times.shape[0] < 2:
        return
    t, t_next = times[:1], times[1:2]
    key = jax.ShapeDtypeStruct((2,), jnp.uint32)
    jax.eval_shape(
        lambda x, key: sampler.step(x, t, t_next, x, x, state, key, process,
                                    lambda x_t, t_: (x_t, x_t)),
        x, key)


@objectives("diffusion")
class DiffusionObjective(Objective[Mean]):
    """Denoising diffusion: sample a noise level, corrupt, predict, weight."""

    def __init__(
        self,
        model: nn.Module,
        process: Process,
        inputs: InputSpec,
        *,
        autoencoder: Optional[AutoEncoder] = None,
        unconditional_prob: float = 0.12,
        ema_decay: float | None = 0.999,
        sampler: Solver[Any] = DDIM(),
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
        self.sampler = sampler
        self.guidance = guidance
        self.steps = steps
        self.ema = (None if ema_decay is None else
                    EMASpec(decay=optax.constant_schedule(ema_decay), select=under("params")))
        self.artifact = VideoGrid if len(inputs.sample.shape) == 4 else ImageGrid
        check_solver(process, sampler, steps)
        # The unconditional datum's value: the encoders are frozen, so one
        # pass here serves every step and every sample.
        self.unconditional = self.encode(self.encoder_params(), {
            keyword: condition.encoder.tokenize([condition.unconditional])
            for keyword, condition in inputs.conditions.items()})
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

    def text_to_image(self, variables):
        """The inference task paired with this objective and variables snapshot."""
        from dew.sampling.pipelines import TextToImage
        return TextToImage(self.model, self.process, self.inputs, variables, self.autoencoder)

    def init(self, key):
        encoders = self.encoder_params()
        variables = self.model.init(
            key, jnp.ones((1, *self.latent_shape)), jnp.ones((1,)), **self.unconditional)
        state = {**variables, "encoders": encoders}
        if self.autoencoder is not None:
            # The frozen weights are state, like the encoders'. They ride in
            # as an argument to the compiled step for the layout to place.
            state["autoencoder"] = self.autoencoder.params
        return state

    def trainable(self, params) -> dict:
        """The model's own collections: what is left once the frozen towers
        are taken out."""
        return {name: value for name, value in params.items()
                if name not in ("encoders", "autoencoder")}

    def _conditions(self, params, batch, key, *, dropout):
        tokens = {keyword: batch[condition.field]
                  for keyword, condition in self.inputs.conditions.items()}
        given = self.encode(params["encoders"], tokens)
        if dropout:
            count = batch[self.inputs.sample.key].shape[0]
            dropped = jax.random.bernoulli(key, self.unconditional_prob, (count,))
            given = jax.tree.map(
                lambda value, blank: jnp.where(
                    expand(dropped, value), jnp.broadcast_to(blank, value.shape), value),
                given, self.unconditional)
        return given, self.unconditional

    def _sampling_batch(self, batch):
        return {name: batch[name] for name in (self.inputs.sample.key,
                *(condition.field for condition in self.inputs.conditions.values()))}

    def loss(self, params, batch, step: Step):
        data = unit_range(batch[self.inputs.sample.key])
        encode_key, drop_key, time_key, noise_key, dropout_key = jax.random.split(step.key, 5)
        if self.autoencoder is not None:
            data = self.autoencoder.encode(params["autoencoder"], data, encode_key)
        count = data.shape[0]

        conditions, _ = self._conditions(params, batch, drop_key, dropout=True)

        schedule = self.process.schedule
        t = schedule.sample_t(time_key, count)
        noise = jax.random.normal(noise_key, data.shape, dtype=jnp.float32)
        rates = broadcast_rates(schedule, t, data)
        noisy, c_in, target = self.process.prediction.forward_diffusion(data, noise, rates)

        variables = self.trainable(params)
        preds = self.model.apply(
            variables, noisy * c_in, schedule.model_time(t), **conditions,
            train=True, rngs={"dropout": dropout_key})
        preds = self.process.prediction.pred_transform(noisy, preds, rates, t)
        losses = optax.l2_loss(preds, target)
        weights = expand(self.process.weight(t), losses)
        return Mean(jnp.sum(losses * weights),
                    jnp.asarray(losses.size, jnp.promote_types(losses.dtype, jnp.float32))), Aux(metrics={})

    def _sample_impl(self, params, batch, key, *, count: int):
        given, unconditional = self._conditions(params, batch, key, dropout=False)
        variables = self.trainable(params)
        denoise = self.process.denoiser(
            self.model, variables, given, None if self.guidance is None else unconditional)
        noise_key, sample_key = jax.random.split(key)
        x_T = self.process.noise(noise_key, (count, *self.latent_shape))
        samples = sample(denoise, x_T, self.steps, solver=self.sampler,
                         guidance=self.guidance, key=sample_key)
        if self.autoencoder is not None:
            samples = self.autoencoder.decode(params["autoencoder"], samples)
        return jnp.clip(samples, -1.0, 1.0)

    def evaluate(self, params, batch, step: Step):
        """One generated sample for every real row, without display decoding."""
        params = params if step.ema is None else step.ema
        count = batch[self.inputs.sample.key].shape[0]
        samples = self._sample(params, self._sampling_batch(batch), step.key, count=count)
        assert self.artifact is not None
        return self.artifact(samples)

    def preview(self, params, batch, step: Step, *, scored=None):
        """A separate small draw for display, with root-only caption decoding."""
        error = None
        prepared = None
        try:
            params = params if step.ema is None else step.ema
            count = min(VALIDATION_SAMPLES, batch[self.inputs.sample.key].shape[0])
            prepared = (self._sample, count, self._sampling_batch(batch))
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="diffusion preview setup")
        error = None
        samples = tokens = None
        try:
            assert prepared is not None
            sample, count, raw_batch = prepared
            selected = jax.tree.map(lambda value: value[:count], raw_batch)
            samples = sample(params, selected, step.key, count=count)
            tokens = {keyword: selected[condition.field]
                      for keyword, condition in self.inputs.conditions.items()}
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="diffusion preview generation")
        samples, tokens = collective_host((samples, tokens), phase="diffusion preview")
        if jax.process_index() != 0:
            return None
        assert samples is not None and tokens is not None
        captions = ()
        for keyword, condition in self.inputs.conditions.items():
            captions = condition.encoder.captions(tokens[keyword])
            if captions:
                break
        assert self.artifact is not None
        return self.artifact(samples, captions)
