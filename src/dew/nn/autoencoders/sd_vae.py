"""Pretrained variational image autoencoders behind the AutoEncoder seam."""
import jax.numpy as jnp

from .api import LatentNormalization, ModuleAutoEncoder
from .kl import AutoencoderKL


def published(config: dict, dtype) -> tuple[AutoencoderKL, float, float]:
    """An AutoencoderKL config's model, computing in `dtype`, and its latent shift and scale: its
    `shift_factor` (0 where absent or null) and `scaling_factor` (SD1's 0.18215 where absent), built
    alike by a standalone load and a pipeline's (`load_kl_vae`)."""
    return (AutoencoderKL.from_diffusers(config, dtype), config.get("shift_factor") or 0.0,
            config.get("scaling_factor", 0.18215))


class StableDiffusionVAE(ModuleAutoEncoder[AutoencoderKL]):
    """Encodes and decodes images with a frozen native `AutoencoderKL` and its latent normalization."""

    def __init__(self, modelname="CompVis/stable-diffusion-v1-4", revision="bf16",
                 dtype=jnp.bfloat16, latent_shift=None, latent_scale=None, params=None,
                 model: AutoencoderKL | None = None):
        """Build the autoencoder around `params` as given, loading only the model metadata that is missing.

        Without `params`, it loads the source weights of `modelname` at
        `revision`. Passing `model` as well avoids reading any metadata, and
        passing `model` without `params` raises `ValueError`. `dtype` sets the
        computation dtype and never casts supplied params. When the config is
        read from the checkpoint, `latent_shift` and `latent_scale` default to
        its `shift_factor` and `scaling_factor`; otherwise they default to 0.0
        and 0.18215.
        """
        self.modelname = modelname
        self.revision = revision
        self.dtype = dtype
        if model is None:
            from .pretrained import load_kl_vae

            _, params, _, config = load_kl_vae(modelname, dtype, revision=revision, subfolder=None,
                                               params=params)
            model, shift, scale = published(config, dtype)
            latent_shift = shift if latent_shift is None else latent_shift
            latent_scale = scale if latent_scale is None else latent_scale
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
