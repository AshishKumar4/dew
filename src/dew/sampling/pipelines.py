"""A trained model, rebuilt from its run, that turns prompts into images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp

from dew.diffusion.process import Process
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder, StableDiffusionVAE
from dew.registry import models, presets
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM


AUTOENCODERS = {"stable_diffusion": StableDiffusionVAE}


def load_autoencoder(entry: Optional[dict]) -> Optional[AutoEncoder]:
    """The autoencoder a manifest names, for the one kind that has a loader."""
    if entry is None:
        return None
    try:
        return AUTOENCODERS[entry["name"]](**entry["fields"])
    except KeyError:
        raise ValueError(
            f"no loader rebuilds the {entry['name']!r} autoencoder from a manifest; "
            f"known: {sorted(AUTOENCODERS)}") from None


@dataclass(frozen=True, eq=False)
class TextToImage:
    """`pipe(prompts, steps=40, guidance=4.0, sampler=samplers.Heun(), key=key)`.

    `params` is the objective's whole tree, the EMA copy merged over the live
    weights when the run kept one, so a sample comes from the weights a run
    publishes.
    """

    model: Any
    process: Process
    inputs: InputSpec
    params: Any
    autoencoder: Optional[AutoEncoder] = None

    @classmethod
    def from_run(cls, directory: str, *, ema: bool = True,
                 step: Optional[int] = None) -> "TextToImage":
        """The model, process, inputs and weights of the run in `directory`,
        from its manifest and its latest checkpoint (or `step`)."""
        from dew.checkpoints import Checkpoints
        from dew.interop.manifest import Manifest
        from dew.objectives.base import merge, select
        from dew.objectives.diffusion import DiffusionObjective

        manifest = Manifest.read(directory)
        model = models.build(manifest.model["name"], **manifest.model["fields"])
        process = presets.build(manifest.preset["name"], **manifest.preset["fields"])()
        inputs = InputSpec.from_json(manifest.inputs)
        autoencoder = load_autoencoder(manifest.autoencoder)
        objective = DiffusionObjective(model, process, inputs, autoencoder=autoencoder)

        # Only shapes are traced, so the key is abstract too.
        abstract = jax.eval_shape(objective.init, jax.ShapeDtypeStruct((2,), jnp.uint32))
        template = {"params": abstract}
        if ema:
            template["ema"] = select(abstract, objective.ema.select)
        values, _ = Checkpoints(directory).restore(template, step=step)
        params = values["params"]
        if ema:
            params = merge(params, values["ema"])
        return cls(model=model, process=process, inputs=inputs, params=params,
                   autoencoder=autoencoder)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, ema: bool = True) -> "TextToImage":
        """A run directory published to the Hugging Face Hub, as
        `dew.interop.hub.push_to_hub` writes it."""
        from dew.interop.hub import pull_from_hub

        return cls.from_run(pull_from_hub(repo_id), ema=ema)

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

        @jax.jit
        def run(params, given, null, x_T, key):
            denoise = self.process.denoiser(self.model, params, given,
                                            None if guidance is None else null)
            samples = sample(denoise, x_T, steps, solver=sampler, guidance=guidance, key=key)
            if self.autoencoder is not None:
                samples = self.autoencoder.decode(samples)
            return jnp.clip(samples, -1.0, 1.0)

        return run(self.params, given, null, x_T, jax.random.fold_in(key, 1))
