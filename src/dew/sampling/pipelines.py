"""A trained model, rebuilt from its run, that turns prompts into images."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.core import freeze

from dew.diffusion.process import Process
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.objectives.base import Variables
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM, Solver


if TYPE_CHECKING:
    from dew.objectives.diffusion import DiffusionObjective
    from dew.interop.diffusers import DiffusersTextToImage


@struct.dataclass
class DenoisingInputs:
    """Initial noise and already encoded conditioning for one denoising call."""

    noise: jax.Array
    conditions: Mapping[str, object] = struct.field(default_factory=dict)
    unconditional: Mapping[str, object] = struct.field(default_factory=dict)


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
    autoencoder: AutoEncoder | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", freeze(dict(self.params)))

    def bind(self, variables: Variables) -> TextToImage:
        """Bind another variables snapshot without rebuilding the model or encoders."""
        return replace(self, params=variables)

    @classmethod
    def from_objective(cls, objective: DiffusionObjective, variables: Variables) -> TextToImage:
        """The objective's own inference counterpart, bound to these variables."""
        return objective.text_to_image(variables)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool = True,
                 step: int | None = None) -> "TextToImage":
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
        return cls.from_objective(objective, params)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, ema: bool = True) -> "TextToImage":
        """A run directory published to the Hugging Face Hub, as
        `dew.interop.hub.push_to_hub` writes it."""
        from dew.interop.hub import pull_from_hub

        return cls.from_run(os.fspath(pull_from_hub(repo_id)), ema=ema)

    @classmethod
    def from_diffusers(cls, directory: str, *, revision: str | None = None, dtype=jnp.float32,
                       height: int | None = None, width: int | None = None,
                       local_files_only: bool = False, from_pt: bool = False,
                       task: str | None = None) -> DiffusersTextToImage:
        """Load a Diffusers saved pipeline through its source-specific adapter."""
        from dew.interop.diffusers import load_diffusers_pipeline
        return load_diffusers_pipeline(directory, revision=revision, dtype=dtype, height=height, width=width,
                             local_files_only=local_files_only, from_pt=from_pt, task=task)

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

    def prepare(self, prompts: str | Sequence[str], *, key: jax.Array) -> DenoisingInputs:
        """Encode prompts once; the returned arrays can be reused with other solvers."""
        rows = [prompts] if isinstance(prompts, str) else list(prompts)
        if not rows or not all(isinstance(prompt, str) for prompt in rows):
            raise ValueError("prompts must be a non-empty string sequence")
        given, null = self.conditions(rows)
        noise = self.process.noise(key, (len(rows), *self.latent_shape))
        return DenoisingInputs(noise, given, null)

    def __call__(self, prompts: str | Sequence[str] | DenoisingInputs, *, steps: int = 50,
                 guidance: CFG | float | None = None, sampler: Solver[object] = DDIM(),
                 key: jax.Array) -> jax.Array:
        """Images in [-1, 1], `[len(prompts), H, W, C]`. `guidance` is a
        classifier-free guidance scale, or a `CFG` with its interval, or None
        for the plain conditional prediction."""
        if isinstance(guidance, (int, float)):
            guidance = CFG(float(guidance))
        prepared = prompts if isinstance(prompts, DenoisingInputs) else self.prepare(prompts, key=key)
        if prepared.noise.ndim != len(self.latent_shape) + 1 or prepared.noise.shape[1:] != self.latent_shape:
            raise ValueError(f"initial noise must have shape [batch, {self.latent_shape}]")
        return _run(self.model, self.process, self.autoencoder, self.params, prepared.conditions,
                    prepared.unconditional, prepared.noise, jax.random.fold_in(key, 1),
                    steps=steps, sampler=sampler, guidance=guidance)


# Rebinding weights must not change the static compilation identity.
@functools.partial(jax.jit, static_argnames=("model", "process", "autoencoder", "steps", "sampler", "guidance"))
def _run(model: nn.Module, process: Process, autoencoder: AutoEncoder | None,
         params, given, null, x_T, key, *, steps, sampler, guidance):
    variables = {name: value for name, value in params.items() if name not in ("encoders", "autoencoder")}
    denoise = process.denoiser(model, variables, given, None if guidance is None else null)
    samples = sample(denoise, x_T, steps, solver=sampler, guidance=guidance, key=key)
    if autoencoder is not None:
        samples = autoencoder.decode(params["autoencoder"], samples)
    return jnp.clip(samples, -1.0, 1.0)
