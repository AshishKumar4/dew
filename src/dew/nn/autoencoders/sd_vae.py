"""Pretrained variational image autoencoders behind the AutoEncoder seam."""
import jax.numpy as jnp

from .api import ModuleAutoEncoder
from .kl import AutoencoderKL
from .vae import load_pretrained_vae


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
            pretrained = load_pretrained_vae(modelname, revision=revision, params=params)
            config = pretrained["config"]
            params = pretrained["params"]
            model = AutoencoderKL.from_diffusers(config, dtype)
            latent_shift = (config.get("shift_factor") or 0.0) if latent_shift is None else latent_shift
            latent_scale = config.get("scaling_factor", 0.18215) if latent_scale is None else latent_scale
        if params is None:
            raise ValueError("A native autoencoder requires explicit parameters")
        super().__init__(model, params)
        self.latent_shift = 0.0 if latent_shift is None else latent_shift
        self.latent_scale = 0.18215 if latent_scale is None else latent_scale

    def to_json(self) -> dict:
        import numpy as np

        from dew.registry import dtype_name, record_fields
        return {'name': 'sd_vae', 'fields': {
            'model': record_fields(self.model, AutoencoderKL), 'dtype': dtype_name(self.dtype),
            'latent_shift': np.asarray(self.latent_shift).tolist(),
            'latent_scale': np.asarray(self.latent_scale).tolist()}}

    def moments(self, params, images):
        """Return `AutoencoderKL.moments` of the frames, which `encode_batch` draws its latent from."""
        return jnp.asarray(self.model.apply({"params": params}, images, method=self.model.moments))

    def decode_raw(self, params, latents):
        """Decode a posterior draw as `decode_batch` does, traced in the caller's step, not jitted apart."""
        return jnp.asarray(self.model.apply({"params": params}, latents, method=self.model.decode))

    @property
    def downscale_factor(self) -> int:
        return self.model.downscale_factor

    @property
    def latent_channels(self) -> int:
        return self.model.latent_channels
