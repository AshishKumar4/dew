"""A convolutional variational autoencoder with explicit latent sampling."""
from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from ..conv import Conv
from .vae import FlaxDecoder, FlaxEncoder


def posterior_moments(moments: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The posterior's mean and log-variance: the moments split on the channel
    axis, the log-variance clamped to [-30, 20] as diffusers'
    `DiagonalGaussianDistribution` clamps it."""
    mean, log_variance = jnp.split(moments, 2, axis=-1)
    return mean, jnp.clip(log_variance, -30.0, 20.0)


def posterior_latent(moments: jnp.ndarray, key: jax.Array | None,
                     noise: jnp.ndarray | None = None) -> jnp.ndarray:
    """The posterior mean when `key` and `noise` are None, else one draw from it
    (`posterior_moments`): `mean + deviation * noise`, a standard normal draw of
    the mean's shape, drawn from `key` where `noise` is None. Raises
    `ValueError` for a noise of another shape than the mean's."""
    mean, log_variance = posterior_moments(moments)
    if noise is None:
        if key is None:
            return mean
        # Drawn in float32, as the rest of a step's draws are, whatever the
        # moments' precision, so a float64 run draws the same numbers.
        noise = jax.random.normal(key, mean.shape, dtype=jnp.float32)
    elif noise.shape != mean.shape:
        raise ValueError(f"the posterior's noise is {noise.shape}, and its mean {mean.shape}")
    return mean + jnp.exp(0.5 * log_variance) * noise.astype(mean.dtype)


class AutoencoderKL(nn.Module):
    """Encodes NHWC images to latents with a convolutional VAE, and decodes latents back to images.

    `encode(image, key=None)` returns the posterior mean, and passing a key
    samples its diagonal Gaussian. `decode(latents)` returns unnormalized
    image pixels. The latent scale and shift belong to `AutoEncoder`.
    """
    channels: tuple[int, ...] = (128, 256, 512, 512)
    latent_channels: int = 4
    image_channels: int = 3
    blocks_per_level: int = 2
    norm_groups: int = 32
    quantize: bool = True
    post_quantize: bool = True
    decoder_channels: tuple[int, ...] | None = None
    """The decoder's widths where they differ from the encoder's, as a
    distilled decoder's do; None uses `channels`."""
    dtype: Dtype = jnp.float32

    @classmethod
    def from_diffusers(cls, config: Mapping[str, object], dtype: Dtype) -> AutoencoderKL:
        """The model a diffusers `AutoencoderKL` config describes, computing in `dtype`. The stack
        computes SiLU, attention in the bottleneck, a decoder writing the image's channels and the
        plain encoder and decoder levels; a config asking for anything else is refused by name."""
        from dew import _records
        from dew.interop.diffusion import flag

        channels = _records.integers(config["block_out_channels"], "block_out_channels")
        refused = [name for name, holds in (
            ("act_fn", config.get("act_fn", "silu") == "silu"),
            ("mid_block_add_attention", flag(config, "mid_block_add_attention", default=True)),
            ("out_channels", config.get("out_channels", 3) == config["in_channels"]),
            *((field, _records.strings(config.get(field) or [block] * len(channels), field)
               == (block,) * len(channels))
              for field, block in (("down_block_types", "DownEncoderBlock2D"),
                                   ("up_block_types", "UpDecoderBlock2D"))),
        ) if not holds]
        if refused:
            named = ", ".join(f"{name}={config.get(name)!r}" for name in refused)
            raise ValueError(f"the KL autoencoder does not compute {named}")
        decoder = config.get("decoder_block_out_channels")
        return cls(channels=channels,
                   latent_channels=_records.integer(config["latent_channels"], "latent_channels"),
                   image_channels=_records.integer(config["in_channels"], "in_channels"),
                   blocks_per_level=_records.integer(config["layers_per_block"], "layers_per_block"),
                   norm_groups=_records.integer(config["norm_num_groups"], "norm_num_groups"),
                   quantize=flag(config, "use_quant_conv", default=True),
                   post_quantize=flag(config, "use_post_quant_conv", default=True),
                   decoder_channels=(None if not decoder
                                     else _records.integers(decoder, "decoder_block_out_channels")),
                   dtype=dtype)

    @property
    def downscale_factor(self) -> int:
        """The spatial factor between an image and its latent.

        Every encoder level except the last halves each spatial axis, so the
        factor is 2 ** (len(channels) - 1).
        """
        return 2 ** (len(self.channels) - 1)

    def setup(self):
        self.encoder = FlaxEncoder(
            out_channels=self.latent_channels,
            block_out_channels=self.channels,
            layers_per_block=self.blocks_per_level,
            norm_num_groups=self.norm_groups,
            dtype=jnp.dtype(self.dtype),
        )
        self.decoder = FlaxDecoder(
            out_channels=self.image_channels,
            block_out_channels=self.decoder_channels or self.channels,
            layers_per_block=self.blocks_per_level,
            norm_num_groups=self.norm_groups,
            dtype=jnp.dtype(self.dtype),
        )
        if self.quantize:
            self.quant_conv = Conv(2 * self.latent_channels, (1, 1), padding="VALID", dtype=self.dtype)
        if self.post_quantize:
            self.post_quant_conv = Conv(self.latent_channels, (1, 1), padding="VALID", dtype=self.dtype)

    def moments(self, image):
        """Return the posterior's mean and log-variance, stacked on the channel axis."""
        moments = self.encoder(image)
        return self.quant_conv(moments) if self.quantize else moments

    def encode(self, image, key=None, noise=None):
        return posterior_latent(self.moments(image), key, noise)

    def decode(self, latents):
        if self.post_quantize:
            latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def __call__(self, image, key=None):
        return self.decode(self.encode(image, key))
