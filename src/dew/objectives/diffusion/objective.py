"""The denoising diffusion objective.

Sample a noise level, corrupt, predict, weight. The convention (schedule,
parameterization, weighting) is the `Process`; the sample field and the
conditions are the `InputSpec`; every draw comes from the step's key. The
frozen encoders' weights live in the tree's `encoders` collection, so they
reach the compiled step as arguments and the optimizer never sees them. The
unconditional branch is a pure function of those frozen weights and a fixed
prompt, so the objective encodes it once, when it is built, and the step
reads that: the tower runs over the batch and nothing else, once a step.

Evaluation samples a few images from the validation batch's conditions with
the averaged weights, through the same `sample` inference uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.artifacts import ImageGrid, VideoGrid, agreed, collective_host
from dew.diffusion.process import Process, aligned_conditions
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.inputs import InputSpec, unit_range
from dew.nn.autoencoders import AutoEncoder
from dew.objectives.base import Aux, EMASpec, Mean, Objective, Step, Variables, under
from dew.registry import objectives
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM, Solver

if TYPE_CHECKING:
    from dew.sampling.pipelines import TextToImage
    from dew.training.state import TrainState

# Samples a validation batch draws, conditioned or not.
VALIDATION_SAMPLES = 4


def check_solver(process, sampler, steps: int) -> None:
    """Trace a solver step on the objective's actual sampling grid."""
    x = jnp.zeros((1, 1), jnp.float32)
    with jax.ensure_compile_time_eval():
        times = process.times(steps)
    state = sampler.init(x, times, process, key=jax.random.PRNGKey(0))
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
        autoencoder: AutoEncoder | None = None,
        unconditional_prob: float = 0.12,
        ema_decay: float | None = 0.999,
        sampler: Solver = DDIM(),
        guidance: CFG | None = CFG(3.0),
        steps: int = 200,
        pretrained: Variables | None = None,
    ):
        """Build a denoising objective over `model` for the `inputs` field.

        `sampler`, `guidance` and `steps` are how evaluation samples;
        `guidance` None is the plain conditional prediction.
        """
        self.model = model
        self.process = process
        self.inputs = inputs
        self.autoencoder = autoencoder
        self.pretrained = pretrained
        if inputs.mask is not None and autoencoder is None:
            raise ValueError("Masked-image conditioning requires an autoencoder")
        # The unconditional branch is a pure function of the frozen towers
        # and each condition's fixed datum, so it is encoded here, once, and
        # not inside every step and every sample. A few hundred kilobytes of
        # host arrays, which a compiled step takes as a constant; the towers
        # themselves stay in the state, for the reason `held_variables` gives.
        self.unconditional_conditions = jax.tree.map(
            np.asarray, self.encode(self.encoder_params()))
        self.unconditional_prob = unconditional_prob
        self.sampler = sampler
        self.guidance = guidance
        self.steps = steps
        self.ema = (None if ema_decay is None else
                    EMASpec(decay=optax.constant_schedule(ema_decay), select=under("params")))
        self.artifact = VideoGrid if len(inputs.sample.shape) == 4 else ImageGrid
        check_solver(process, sampler, steps)
        self._sample = jax.jit(self._sample_impl, static_argnames=("count",))

    def pipeline(self, state: TrainState, *, ema: bool = True) -> TextToImage:
        """The model over the state's published weights as a `TextToImage`
        task, sampling the way this objective's evaluation does."""
        from dew.sampling.pipelines import TextToImage

        return TextToImage.from_objective(self, self._pipeline_weights(state, ema))

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
        if self.pretrained is not None:
            return dict(self.pretrained["encoders"])
        return {keyword: condition.encoder.params
                for keyword, condition in self.inputs.conditions.items()}

    def encode(self, encoders, tokens: dict | None = None) -> dict:
        """Encode conditions under the supplied parameters; omitted tokens
        select each condition's configured unconditional datum."""
        if tokens is None:
            tokens = {keyword: condition.encoder.tokenize([condition.unconditional])
                      for keyword, condition in self.inputs.conditions.items()}
        return {keyword: condition.encoder.encode(encoders[keyword], tokens[keyword])
                for keyword, condition in self.inputs.conditions.items()}

    def blank_conditions(self, like: dict) -> dict:
        """Cast the stored unconditional conditions to the conditional branch's dtypes.

        `like` is the conditional branch. The values themselves were
        encoded once at construction, so this only changes dtype.
        """
        return jax.tree.map(lambda blank, value: jnp.asarray(blank, value.dtype),
                            self.unconditional_conditions, like)

    def held_variables(self) -> Variables:
        """Every array `init` starts from rather than draws: a whole pretrained
        tree, or the frozen towers.

        A text tower and a VAE are released weights: hundreds of megabytes
        that a nullary trace would compile into the state executable as
        constants. One mapping, so an objective that starts from more than
        the towers extends this and `init` together.
        """
        if self.pretrained is not None:
            return self.pretrained
        held: dict[str, Any] = {"encoders": self.encoder_params()}
        if self.autoencoder is not None:
            held["autoencoder"] = self.autoencoder.params
        return held

    def init(self, key, variables: Variables | None = None) -> Variables:
        held = self.held_variables() if variables is None else variables
        if "params" in held:
            return held
        conditions = self.encode(held["encoders"])
        if self.inputs.mask is not None:
            conditions = {**conditions, "mask": jnp.zeros((1, *self.latent_shape[:-1], 1)),
                          "masked_image": jnp.zeros((1, *self.latent_shape))}
        drawn = self.model.init(key, jnp.ones((1, *self.latent_shape)), jnp.ones((1,)), **conditions)
        state: dict[str, Any] = {**drawn, "encoders": held["encoders"]}
        if "autoencoder" in held:
            # The frozen weights are state, like the encoders'. They ride in
            # as an argument to the compiled step for the layout to place.
            state["autoencoder"] = held["autoencoder"]
        return state

    def trainable(self, params) -> dict:
        """Return the model's own collections, without the frozen towers."""
        return {name: value for name, value in params.items()
                if name not in ("encoders", "autoencoder")}

    def encoded_conditions(self, params, batch) -> dict:
        """Encode each condition's own batch field under the tree's frozen towers."""
        tokens = {keyword: batch[condition.field]
                  for keyword, condition in self.inputs.conditions.items()}
        return self.encode(params["encoders"], tokens)

    def denoiser(self, params, given, unconditional):
        """Build the process's denoiser over the model's own collections.

        The unconditional branch is passed only when this objective is
        guided; without guidance the sampler never evaluates it.
        """
        return self.process.denoiser(self.model, self.trainable(params), given,
                                     None if self.guidance is None else unconditional)

    def _conditions(self, params, batch, key, *, dropout):
        given = self.encoded_conditions(params, batch)
        # The unconditional prompt is fixed, so its encoding is a constant
        # and the text tower runs over the batch alone, once per step.
        unconditional = self.blank_conditions(given)
        if dropout:
            count = batch[self.inputs.sample.key].shape[0]
            dropped = jax.random.bernoulli(key, self.unconditional_prob, (count,))
            given = jax.tree.map(
                lambda value, blank: jnp.where(
                    expand(dropped, value), jnp.broadcast_to(blank, value.shape), value),
                given, aligned_conditions(given, unconditional))
        if self.inputs.mask is not None:
            from dew.inputs.diffusion import latent_image_conditions
            spatial = latent_image_conditions(self.autoencoder, params["autoencoder"],
                unit_range(batch[self.inputs.sample.key]), batch[self.inputs.mask.key], jax.random.fold_in(key, 1))
            return {**given, **spatial}, {**unconditional, **spatial}
        return given, unconditional

    def _sampling_batch(self, batch):
        fields = [condition.field for condition in self.inputs.conditions.values()]
        if self.inputs.mask is not None:
            fields.extend((self.inputs.sample.key, self.inputs.mask.key))
        return {name: batch[name] for name in fields}

    def loss(self, params, batch, step: Step):
        samples = unit_range(batch[self.inputs.sample.key])
        encode_key, drop_key, time_key, noise_key, dropout_key = jax.random.split(step.key, 5)
        if self.autoencoder is not None:
            samples = self.autoencoder.encode(params["autoencoder"], samples, encode_key)
        count = samples.shape[0]

        conditions, _ = self._conditions(params, batch, drop_key, dropout=True)

        schedule = self.process.schedule
        t = schedule.sample_t(time_key, count)
        noise = jax.random.normal(noise_key, samples.shape, dtype=jnp.float32)
        rates = broadcast_rates(schedule, t, samples)
        noisy, c_in, target = self.process.prediction.forward_diffusion(samples, noise, rates)

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
        denoise = self.denoiser(params, given, unconditional)
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
        def setup():
            weights = params if step.ema is None else step.ema
            count = min(VALIDATION_SAMPLES, batch[self.inputs.sample.key].shape[0])
            return weights, count, self._sampling_batch(batch)

        def generate():
            selected = jax.tree.map(lambda value: value[:count], raw_batch)
            return (self._sample(weights, selected, step.key, count=count),
                    {keyword: selected[condition.field]
                     for keyword, condition in self.inputs.conditions.items()})

        weights, count, raw_batch = agreed("diffusion preview setup", setup)
        samples, tokens = agreed("diffusion preview generation", generate)
        samples, tokens = collective_host((samples, tokens), phase="diffusion preview")
        if jax.process_index() != 0:
            return None
        captions = ()
        for keyword, condition in self.inputs.conditions.items():
            captions = condition.encoder.captions(tokens[keyword])
            if captions:
                break
        assert self.artifact is not None
        return self.artifact(samples, captions)
