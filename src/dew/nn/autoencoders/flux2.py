"""FLUX.2's autoencoder, diffusers' `AutoencoderKLFlux2`, as its pipeline uses it.

The network is a Stable Diffusion `AutoencoderKL` with 32 latent channels
(and, in the small decoder, narrower decoder levels). What differs is the
latent the transformer reads: `Flux2Pipeline` folds each 2x2 block of the
VAE's latent into the channels, 32 to 128 at a sixteenth of the image's side,
and normalizes each of the 128 channels by the running statistics of the
VAE's affine-free batch norm, `(z - mean) / sqrt(var + eps)`. Both belong to
the latent here, so a diffusion run sees what the pipeline's transformer
sees.
"""
from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

from dew.batch import Variables
from dew.nn.kernels.options import program

from ..scan_orders import pixel_shuffle, pixel_unshuffle
from .api import LatentNormalization, ModuleAutoEncoder
from .kl import AutoencoderKL


def _folded(model: AutoencoderKL, params, image, key=None, noise=None):
    """One batch of frames encoded and folded 2x2 into the latent's channels, a posterior's `noise`
    unfolded to the layout it is drawn in."""
    return pixel_unshuffle(jnp.asarray(model.apply({"params": params}, image, key,
                                                   None if noise is None else pixel_shuffle(noise),
                                                   method=model.encode)))


def _unfolded(model: AutoencoderKL, params, latent):
    """One batch of folded latents unfolded and decoded."""
    return model.apply({"params": params}, pixel_shuffle(latent), method=model.decode)


class Flux2Autoencoder(ModuleAutoEncoder[AutoencoderKL]):
    """Native FLUX.2 VAE weights, the 2x2 fold and the batch-norm statistics
    as the latent normalization, per folded channel."""

    def __init__(self, *, model: AutoencoderKL, params: Variables, mean, variance, epsilon: float):
        super().__init__(model, params)
        # `Flux2Pipeline._patchify_latents` and `_unpatchify_latents`, channels last. A posterior's
        # `noise` comes in the folded latent's layout and is unfolded to the one it is drawn in.
        self.encode_single_frame = functools.partial(program(_folded, model, static_argnums=0), model)
        self.decode_single_frame = functools.partial(program(_unfolded, model, static_argnums=0), model)
        mean, variance = np.asarray(mean, np.float32), np.asarray(variance, np.float32)
        scale = 1.0 / np.sqrt(variance + np.float32(epsilon))
        if mean.shape != (4 * model.latent_channels,) or scale.shape != mean.shape:
            raise ValueError(
                f"batch-norm statistics {mean.shape} do not hold one value per folded "
                f"channel ({4 * model.latent_channels},)"
            )
        self.normalization = LatentNormalization({'mean': mean, 'variance': variance, 'epsilon': epsilon},
                                                 mean, scale)

    def record_arguments(self) -> dict:
        return dict(self.normalization.record)

    @property
    def downscale_factor(self) -> int:
        return 2 * self.model.downscale_factor

    @property
    def latent_channels(self) -> int:
        return 4 * self.model.latent_channels


