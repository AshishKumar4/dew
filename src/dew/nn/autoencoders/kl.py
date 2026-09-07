"""A convolutional variational autoencoder with explicit latent sampling."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from .vae import FlaxEncoder, FlaxDecoder


class AutoencoderKL(nn.Module):
    """NHWC image/latent arrays; scaling and shifts belong to AutoEncoder.

    encode(image, key=None) returns the posterior mean. Passing a key samples
    its diagonal Gaussian; decode(latents) returns unnormalized image pixels.
    """
    channels: tuple[int, ...] = (128, 256, 512, 512)
    latent_channels: int = 4
    image_channels: int = 3
    blocks_per_level: int = 2
    norm_groups: int = 32
    quantize: bool = True
    post_quantize: bool = True
    dtype: Dtype = jnp.float32

    def setup(self):
        levels = len(self.channels)
        self.encoder = FlaxEncoder(in_channels=self.image_channels, out_channels=self.latent_channels,
            down_block_types=("DownEncoderBlock2D",) * levels, block_out_channels=self.channels,
            layers_per_block=self.blocks_per_level, norm_num_groups=self.norm_groups, double_z=True, dtype=jnp.dtype(self.dtype))
        self.decoder = FlaxDecoder(in_channels=self.latent_channels, out_channels=self.image_channels,
            up_block_types=("UpDecoderBlock2D",) * levels, block_out_channels=self.channels,
            layers_per_block=self.blocks_per_level, norm_num_groups=self.norm_groups, dtype=jnp.dtype(self.dtype))
        if self.quantize:
            self.quant_conv = nn.Conv(2 * self.latent_channels, (1, 1), padding="VALID", dtype=self.dtype)
        if self.post_quantize:
            self.post_quant_conv = nn.Conv(self.latent_channels, (1, 1), padding="VALID", dtype=self.dtype)

    def encode(self, image, key=None):
        moments = self.encoder(image)
        if self.quantize:
            moments = self.quant_conv(moments)
        mean, log_variance = jnp.split(moments, 2, axis=-1)
        if key is None:
            return mean
        deviation = jnp.exp(0.5 * jnp.clip(log_variance, -30.0, 20.0))
        return mean + deviation * jax.random.normal(key, mean.shape, dtype=mean.dtype)

    def decode(self, latents):
        if self.post_quantize:
            latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def __call__(self, image, key=None):
        return self.decode(self.encode(image, key))
