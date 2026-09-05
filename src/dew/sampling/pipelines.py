"""A trained model, rebuilt from its run, that turns prompts into images."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.process import Process
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.objectives.base import Variables
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM


@dataclass(frozen=True, eq=False)
class TextToImage:
    """`pipe(prompts, steps=40, guidance=4.0, sampler=samplers.Heun(), key=key)`.

    `params` is the objective's whole tree, the EMA copy merged over the live
    weights when the run kept one, so a sample comes from the weights a run
    publishes.
    """

    model: nn.Module
    process: Process
    inputs: InputSpec
    params: Variables
    autoencoder: Optional[AutoEncoder] = None

    @classmethod
    def from_run(cls, directory: str, *, ema: bool = True,
                 step: Optional[int] = None) -> "TextToImage":
        """The run in `directory`: its `run.json` built the way the recipe
        built it, and the weights of its latest checkpoint (or `step`)."""
        from dew.checkpoints import Checkpoints
        from dew.objectives.base import merge, select
        from dew.objectives.diffusion import DiffusionRunConfig

        objective = DiffusionRunConfig.load(directory).build()
        # Only shapes are traced, so the key is abstract too. Inference runs
        # on one process, so every leaf lands on the first device and the
        # sampling jit moves it from there.
        device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        abstract = jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=device),
            jax.eval_shape(objective.init, jax.ShapeDtypeStruct((2,), jnp.uint32)))
        template = {"params": abstract}
        if ema:
            if objective.ema is None:
                raise ValueError(
                    "this run's objective keeps no EMA; load its weights with ema=False")
            template["ema"] = select(abstract, objective.ema.select)
        values, _ = Checkpoints(directory).restore(template, step=step)
        params = values["params"]
        if ema:
            params = merge(params, values["ema"])
        return cls(model=objective.model, process=objective.process, inputs=objective.inputs,
                   params=params, autoencoder=objective.autoencoder)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, ema: bool = True) -> "TextToImage":
        """A run directory published to the Hugging Face Hub, as
        `dew.interop.hub.push_to_hub` writes it."""
        from dew.interop.hub import pull_from_hub

        return cls.from_run(os.fspath(pull_from_hub(repo_id)), ema=ema)

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

    def conditions(self, prompts: Sequence[str]) -> tuple[dict, dict]:
        """The encoded prompts and the encoded unconditional datum, keyed by
        model keyword, under the pipeline's parameters."""
        given, null = {}, {}
        for keyword, condition in self.inputs.conditions.items():
            encoder_params = self.params["encoders"][keyword]
            given[keyword] = condition.encoder.encode(
                encoder_params, condition.encoder.tokenize(list(prompts)))
            null[keyword] = condition.encoder.encode(
                encoder_params, condition.encoder.tokenize([condition.unconditional]))
        return given, null

    def __call__(self, prompts: Sequence[str], *, steps: int = 50,
                 guidance: CFG | float | None = None, sampler=DDIM(), key) -> jax.Array:
        """Images in [-1, 1], `[len(prompts), H, W, C]`. `guidance` is a
        classifier-free guidance scale, or a `CFG` with its interval, or None
        for the plain conditional prediction."""
        if isinstance(guidance, (int, float)):
            guidance = CFG(float(guidance))
        given, null = self.conditions(prompts)
        x_T = self.process.noise(key, (len(prompts), *self.latent_shape))
        return _run(self, self.params, given, null, x_T, jax.random.fold_in(key, 1),
                    steps=steps, sampler=sampler, guidance=guidance)


# The pipeline (by identity), the step count, the solver and the guidance
# are compile-time constants, so a second call with the same settings runs
# the compiled loop instead of tracing it again.
@functools.partial(jax.jit, static_argnames=("pipe", "steps", "sampler", "guidance"))
def _run(pipe: TextToImage, params, given, null, x_T, key, *, steps, sampler, guidance):
    denoise = pipe.process.denoiser(pipe.model, params, given,
                                    None if guidance is None else null)
    samples = sample(denoise, x_T, steps, solver=sampler, guidance=guidance, key=key)
    if pipe.autoencoder is not None:
        samples = pipe.autoencoder.decode(params["autoencoder"], samples)
    return jnp.clip(samples, -1.0, 1.0)
