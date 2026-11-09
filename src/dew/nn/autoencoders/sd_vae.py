"""Pretrained variational image autoencoders behind the AutoEncoder seam."""
import jax.numpy as jnp

from .api import LatentNormalization, ModuleAutoEncoder
from .kl import AutoencoderKL


class StableDiffusionVAE(ModuleAutoEncoder[AutoencoderKL]):
    """Encodes and decodes images with a frozen native `AutoencoderKL` and its latent normalization."""

    def __init__(self, model: AutoencoderKL, params, dtype=jnp.bfloat16, latent_shift=None,
                 latent_scale=None):
        """The autoencoder around `params` as given: `dtype` sets the computation dtype and never casts
        them, and `latent_shift` and `latent_scale` default to 0.0 and 0.18215 (SD1's). A release's, its
        config's shift and scale with it, is `dew.interop.autoencoders.pretrained.load_autoencoder`'s."""
        self.dtype = dtype
        if params is None:
            raise ValueError("A native autoencoder requires explicit parameters")
        super().__init__(model, params)
        shift = 0.0 if latent_shift is None else latent_shift
        scale = 0.18215 if latent_scale is None else latent_scale
        self.normalization = LatentNormalization({'latent_shift': shift, 'latent_scale': scale}, shift, scale)

    def record_arguments(self) -> dict:
        from dew.registry import dtype_name
        return {'dtype': dtype_name(self.dtype), **self.normalization.record}

    def moments(self, params, images):
        """Return `AutoencoderKL.moments` of the frames, which `encode_batch` draws its latent from."""
        return jnp.asarray(self.model.apply({"params": params}, images, method=self.model.moments))

    def decode_raw(self, params, latents):
        """Decode a posterior draw as `decode_batch` does, traced in the caller's step, not jitted apart."""
        return jnp.asarray(self.model.apply({"params": params}, latents, method=self.model.decode))
