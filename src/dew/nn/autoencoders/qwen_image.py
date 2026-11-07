"""Qwen-Image 2.1's autoencoder, `AutoencoderKLQwenImage21`, on one frame.

An independent linen port of diffusers 6256aa76
autoencoder_kl_qwenimage21.py (Apache-2.0): Wan's causal video VAE for
images, every 3D convolution squeezed to 2D, NHWC, on the first-chunk path a
single frame takes; its norm, residual, attention and mid blocks are Wan's
own on a frame. The temporal machinery leaves three facts: the downsampling
`AvgDown3D` shortcut pads one zero frame in front before averaging; the
upsampling `DupUp3D` shortcut keeps the second duplicated frame; and
`time_conv` runs only from a second frame on, its weights held so an export
writes them back unchanged (zero gradient here). The decoder clamps pixels
to [-1, 1], and RMS gammas keep the stored `[C, 1, 1(, 1)]` shape.
"""
from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew.batch import Variables

from ..activations import silu
from ..blocks import Upsample
from .api import LatentNormalization, ModuleAutoEncoder
from .kl import posterior_latent
from .vae import FlaxDownsample2D
from .wan import WanMidBlock, WanResidualBlock, WanRMSNorm, causal_conv


def _conv(features: int, kernel: int, dtype: Dtype, name: str | None) -> nn.Module:
    """The source's padded "causal" convolution, which is a 2D one on a frame."""
    return causal_conv(features, kernel, 2, dtype, name)


class _TimeConv(nn.Module):
    """A `time_conv` 1x1 convolution the single-frame path never runs: its
    kernel and bias exist so they load and export, and are not applied."""

    in_features: int
    features: int

    @nn.compact
    def __call__(self) -> None:
        self.param("kernel", nn.initializers.lecun_normal(), (1, 1, self.in_features, self.features))
        self.param("bias", nn.initializers.zeros, (self.features,))


class _Downsampler(nn.Module):
    features: int
    temporal: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.temporal:
            _TimeConv(self.features, self.features, name="time_conv")()
        return FlaxDownsample2D(self.features, dtype=jnp.dtype(self.dtype), name="resample")(x)


class _Upsampler(nn.Module):
    features: int
    temporal: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.temporal:
            _TimeConv(self.features, 2 * self.features, name="time_conv")()
        return Upsample(self.features, 2, dtype=jnp.dtype(self.dtype), conv="conv", name="resample")(x)


def _average_down(x, features: int, temporal: bool, spatial: int):
    """`AvgDown3D` on one frame: each output channel is the mean of a run of
    the (channel, frame, row, column) phases, the frame axis a zero frame
    and the image when the block also halves time."""
    batch, height, width, channels = x.shape
    x = x.reshape(batch, height // spatial, spatial, width // spatial, spatial, channels)
    x = x.transpose(0, 1, 3, 5, 2, 4)[..., None, :, :]
    if temporal:
        x = jnp.concatenate([jnp.zeros_like(x), x], axis=-3)
    return x.reshape(*x.shape[:3], features, -1).mean(axis=-1)


def _duplicate_up(x, features: int, temporal: bool):
    """`DupUp3D` on a first chunk: every channel repeated in place, read as
    (channel, frame, row, column) phases, keeping the last frame."""
    batch, height, width, channels = x.shape
    frames = 2 if temporal else 1
    x = jnp.repeat(x, features * frames * 4 // channels, axis=-1)
    x = x.reshape(batch, height, width, features, frames, 2, 2)[..., -1, :, :]
    return x.transpose(0, 1, 4, 2, 5, 3).reshape(batch, 2 * height, 2 * width, features)


class _DownBlock(nn.Module):
    in_features: int
    features: int
    num_res_blocks: int
    temporal: bool
    down: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        shortcut = _average_down(x, self.features, self.temporal, 2 if self.down else 1)
        for index in range(self.num_res_blocks):
            x = WanResidualBlock(self.in_features if index == 0 else self.features, self.features,
                               self.dtype, name=f"resnets_{index}")(x)
        if self.down:
            x = _Downsampler(self.features, self.temporal, self.dtype, name="downsampler")(x)
        return x + shortcut


class _UpBlock(nn.Module):
    in_features: int
    features: int
    num_res_blocks: int
    temporal: bool
    up: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        source = x
        for index in range(self.num_res_blocks + 1):
            x = WanResidualBlock(self.in_features if index == 0 else self.features, self.features,
                               self.dtype, name=f"resnets_{index}")(x)
        if not self.up:
            return x
        x = _Upsampler(self.features, self.temporal, self.dtype, name="upsampler")(x)
        return x + _duplicate_up(source, self.features, self.temporal)


class _Encoder(nn.Module):
    base_dim: int
    moments: int
    dim_mult: tuple[int, ...]
    num_res_blocks: int
    temporal_downsample: tuple[bool, ...]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        widths = [self.base_dim * multiple for multiple in (1, *self.dim_mult)]
        x = _conv(widths[0], 3, self.dtype, "conv_in")(x)
        last = len(self.dim_mult) - 1
        for index in range(len(self.dim_mult)):
            x = _DownBlock(widths[index], widths[index + 1], self.num_res_blocks,
                           index != last and self.temporal_downsample[index], index != last,
                           self.dtype, name=f"down_blocks_{index}")(x)
        x = WanMidBlock(widths[-1], self.dtype, name="mid_block")(x)
        x = silu(WanRMSNorm(widths[-1], 4, self.dtype, name="norm_out")(x))
        return _conv(self.moments, 3, self.dtype, "conv_out")(x)


class _Decoder(nn.Module):
    base_dim: int
    image_channels: int
    dim_mult: tuple[int, ...]
    num_res_blocks: int
    temporal_upsample: tuple[bool, ...]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, z):
        widths = [self.base_dim * multiple for multiple in (self.dim_mult[-1], *self.dim_mult[::-1])]
        x = _conv(widths[0], 3, self.dtype, "conv_in")(z)
        x = WanMidBlock(widths[0], self.dtype, name="mid_block")(x)
        last = len(self.dim_mult) - 1
        for index in range(len(self.dim_mult)):
            x = _UpBlock(widths[index], widths[index + 1], self.num_res_blocks,
                         index != last and self.temporal_upsample[index], index != last,
                         self.dtype, name=f"up_blocks_{index}")(x)
        x = silu(WanRMSNorm(widths[-1], 4, self.dtype, name="norm_out")(x))
        return _conv(self.image_channels, 3, self.dtype, "conv_out")(x)


class QwenImageVAE(nn.Module):
    """`AutoencoderKLQwenImage21` on single frames, NHWC, pixels in [-1, 1].

    The defaults are Qwen/Qwen-Image-2.1's published config. `encode` gives
    the raw latent `[B, H / 16, W / 16, latent_channels]` (the factor is
    `downscale_factor`), `decode` the clamped pixels.
    """

    base_dim: int = 96
    decoder_base_dim: int = 144
    latent_channels: int = 64
    dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8)
    num_res_blocks: int = 2
    temporal_downsample: tuple[bool, ...] = (False, True, True, True)
    image_channels: int = 4
    dtype: Dtype = jnp.float32

    @property
    def downscale_factor(self) -> int:
        """Every level but the last halves each spatial axis."""
        return 2 ** (len(self.dim_mult) - 1)

    def setup(self):
        self.encoder = _Encoder(self.base_dim, 2 * self.latent_channels, self.dim_mult, self.num_res_blocks,
                                self.temporal_downsample, self.dtype)
        self.quant_conv = _conv(2 * self.latent_channels, 1, self.dtype, None)
        self.post_quant_conv = _conv(self.latent_channels, 1, self.dtype, None)
        self.decoder = _Decoder(
            self.decoder_base_dim,
            self.image_channels,
            self.dim_mult,
            self.num_res_blocks,
            self.temporal_downsample[::-1],
            self.dtype,
        )

    def encode(self, x, key=None, noise=None):
        """The posterior mean, or a draw from it when `key` or `noise` is given."""
        return posterior_latent(self.quant_conv(self.encoder(x)), key, noise)

    def decode(self, z):
        return jnp.clip(self.decoder(self.post_quant_conv(z)), -1.0, 1.0)

    def __call__(self, x, key=None):
        return self.decode(self.encode(x, key))


class QwenImageAutoencoder(ModuleAutoEncoder[QwenImageVAE]):
    """Native Qwen-Image 2.1 VAE weights and the pipeline's per-channel
    latent normalization, `(z - latents_mean) / latents_std` on the way out
    and `z * latents_std + latents_mean` on the way in."""

    def __init__(self, *, model: QwenImageVAE, params: Variables,
                 latents_mean: Sequence[float], latents_std: Sequence[float]):
        super().__init__(model, params)
        mean, std = np.asarray(latents_mean, np.float32), np.asarray(latents_std, np.float32)
        expected = (model.latent_channels,)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                f"latents_mean {mean.shape} and latents_std {std.shape} "
                f"must both hold one value per latent channel {expected}"
            )
        # The pipeline divides by the standard deviation and multiplies back.
        self.normalization = LatentNormalization({'latents_mean': mean, 'latents_std': std}, mean, std,
                                                 divides=True)

    def record_arguments(self) -> dict:
        return dict(self.normalization.record)


