"""Official LMS schedule preparation with Dew's existing JAX LMS update.

Diffusers 0.34 Flax LMS flattens derivative history and truncates model times.
Its PyTorch scheduler supplies only the small timestep/sigma table here; model
outputs and latent history remain JAX arrays throughout denoising.
"""
from __future__ import annotations

import json
from pathlib import Path

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_utils_flax import CommonSchedulerState

from dew.diffusion.process import Process
from dew.diffusion.schedules.common import GeneralizedNoiseScheduler
from dew.diffusion.transforms import EpsilonPredictionTransform
from dew.sampling.solvers import LMS


class _SigmaCoordinate(GeneralizedNoiseScheduler):
    def sigmas(self, t):
        return jnp.asarray(t)

    def t_of_sigma(self, sigma):
        return jnp.asarray(sigma)


@struct.dataclass
class LMSState:
    common: CommonSchedulerState
    timesteps: jnp.ndarray
    sigmas: jnp.ndarray
    init_noise_sigma: jnp.ndarray
    history: tuple


class SourceLMS:
    def __init__(self, config: dict):
        self.config = config
        self.solver = LMS(order=4)
        self.process = Process(_SigmaCoordinate(), EpsilonPredictionTransform())

    def set_timesteps(self, state, num_inference_steps: int, shape: tuple):
        source = LMSDiscreteScheduler.from_config(self.config)
        assert isinstance(source, LMSDiscreteScheduler)
        source.set_timesteps(num_inference_steps)
        times = jnp.asarray(source.timesteps.numpy())
        sigmas = jnp.asarray(source.sigmas.numpy())
        history = self.solver.init(jax.ShapeDtypeStruct(shape, jnp.float32), sigmas, self.process)
        return LMSState(state.common, times, sigmas, jnp.asarray(np.asarray(source.init_noise_sigma)), history)

    def scale_model_input(self, state, sample, timestep):
        index = jnp.argmax(state.timesteps == timestep)
        return sample / jnp.sqrt(1 + state.sigmas[index] ** 2)

    def add_noise(self, state, original_samples, noise, timesteps):
        indices = jnp.argmax(state.timesteps[None, :] == timesteps[:, None], axis=-1)
        sigma = state.sigmas[indices].reshape((-1,) + (1,) * (noise.ndim - 1))
        return original_samples + sigma * noise

    @partial(jax.jit, static_argnums=(0,), static_argnames=("return_dict",))
    def step(self, state, model_output, timestep, sample, *, return_dict=False):
        index = jnp.argmax(state.timesteps == timestep)
        sigma, next_sigma = state.sigmas[index], state.sigmas[index + 1]
        if self.config.get("prediction_type", "epsilon") == "v_prediction":
            clean = sample / (1 + sigma ** 2) - sigma * model_output / jnp.sqrt(1 + sigma ** 2)
        else:
            clean = sample - sigma * model_output
        batch = sample.shape[0]
        value, history = self.solver.step(sample, jnp.full((batch,), sigma), jnp.full((batch,), next_sigma),
                                          clean, model_output, state.history, None, self.process, None)
        return value, state.replace(history=history)

    def save_pretrained(self, save_directory):
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scheduler_config.json").write_text(json.dumps(self.config, indent=2))
