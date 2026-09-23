"""Pretrained variational image autoencoders behind the AutoEncoder seam."""
import jax
import jax.numpy as jnp

from .api import ModuleAutoEncoder
from .kl import AutoencoderKL
from .vae import load_pretrained_vae


class StableDiffusionVAE(ModuleAutoEncoder[AutoencoderKL]):
    """Frozen native AutoencoderKL variables and their latent normalization."""

    def __init__(self, modelname="CompVis/stable-diffusion-v1-4", revision="bf16",
                 dtype=jnp.bfloat16, latent_shift=None, latent_scale=None, params=None,
                 model: AutoencoderKL | None = None):
        """Bind supplied params unchanged, reading only missing model metadata.

        Without params, load source weights. Supplying model as well avoids
        metadata I/O; dtype controls computation and never casts supplied params.
        """
        self.modelname = modelname
        self.revision = revision
        self.dtype = dtype
        if model is None:
            pretrained = load_pretrained_vae(modelname, revision=revision, params=params)
            config = pretrained["config"]
            params = pretrained["params"]
            model = AutoencoderKL(
                channels=tuple(config["block_out_channels"]), latent_channels=config["latent_channels"],
                image_channels=config["in_channels"], blocks_per_level=config["layers_per_block"],
                norm_groups=config["norm_num_groups"], quantize=config.get("use_quant_conv", True),
                post_quantize=config.get("use_post_quant_conv", True), dtype=dtype)
            latent_shift = (config.get("shift_factor") or 0.0) if latent_shift is None else latent_shift
            latent_scale = config.get("scaling_factor", 0.18215) if latent_scale is None else latent_scale
        if params is None:
            raise ValueError("A native autoencoder requires explicit parameters")
        super().__init__(model, params)
        self.latent_shift = 0.0 if latent_shift is None else latent_shift
        self.latent_scale = 0.18215 if latent_scale is None else latent_scale
        frame = jax.ShapeDtypeStruct((1, 128, 128, model.image_channels), dtype)
        latent = jax.eval_shape(self.encode_single_frame, self.params, frame)
        self._downscale_factor = frame.shape[1] // latent.shape[1]
        self._latent_channels = latent.shape[-1]

    @property
    def downscale_factor(self) -> int:
        return self._downscale_factor

    @property
    def latent_channels(self) -> int:
        return self._latent_channels
