"""Wan 2.1's causal video autoencoder, `AutoencoderKLWan`.

An independent linen port of diffusers 0.34.0 autoencoder_kl_wan.py
(Apache-2.0), channels last: videos are `[B, T, H, W, C]`. Every 3-D
convolution is causal in time, two zero frames in front. The first frame
compresses alone and every four after it into one latent frame, so 1 + 4k
frames make 1 + k latent frames and an image is the one-frame video.

The source walks a video in chunks and carries each convolution's last two
input frames across them, which is the causal convolution over the whole
video. `WanVAE.encode` and `WanVAE.decode` compute it in one pass;
`chunked_moments` and `chunked_decode` walk it as the source does, the first
frame alone and then four frames to one latent frame at a time, with each
convolution's frames carried in the "cache" collection, so the activations
held are one chunk's, not the clip's. `WanAutoencoder` encodes and decodes
in chunks. The resampling blocks keep the first frame out of their temporal
convolution, as the source's first chunk does: a downsampling block
convolves with stride 2 and no padding, so output frame j > 0 reads frames
2j - 2 to 2j; an upsampling block turns every later frame into two by a
causal convolution that doubles the channels. The decoder clamps pixels to
[-1, 1] (`_decode`), and RMS gammas keep the stored `[C, 1, 1(, 1)]` shape.
"""
from __future__ import annotations

import functools
import math
from itertools import pairwise

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew.batch import Variables
from dew.nn.kernels.options import program

from ..activations import silu
from ..attention import scaled_dot_product_attention
from ..blocks import Upsample
from ..conv import Conv
from ..norms import l2_normalized
from ..precision import at_least_fp32
from .api import AutoEncoder, LatentNormalization
from .kl import posterior_latent


class WanRMSNorm(nn.Module):
    """`WanRMS_norm`, which Qwen-Image's VAE keeps: `F.normalize` over the
    channels times sqrt(C) and a gamma stored `[C]` followed by `rank - 1`
    unit axes. Half precision computes in float32 and rounds once."""

    features: int
    rank: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        gamma = self.param("gamma", nn.initializers.ones, (self.features, *(1,) * (self.rank - 1)))
        wide = x.astype(at_least_fp32(x.dtype))
        return (l2_normalized(wide) * math.sqrt(self.features) * gamma.reshape(-1)).astype(self.dtype)


def causal_conv(features: int, kernel: int, spatial: int, dtype: Dtype, name: str | None) -> nn.Module:
    """`WanCausalConv3d` over a video's three axes, every frame's padding in
    front of it, or over a frame's two, the 2-D convolution a single frame
    reduces it to. The source's parameters sit directly under `name`, so the
    module is the Flax convolution itself."""
    pad = kernel // 2
    time = ((kernel - 1, 0),) if spatial == 3 else ()
    return Conv(features, (kernel,) * spatial, padding=(*time, (pad, pad), (pad, pad)), dtype=dtype,
                name=name)


def _chunked_call(module: nn.Module) -> bool:
    """Whether `module` runs one chunk of a longer video: it may write the
    "cache" collection, outside `init`, where every collection is writable."""
    return module.is_mutable_collection("cache") and not module.is_initializing()


def _causal(parent: nn.Module, features: int, kernel: int, spatial: int, name: str, x):
    """`causal_conv` as `parent`'s child `name`, applied to `x`.

    Where `x` is one chunk of a longer video (`_chunked_call`), the last
    `kernel - 1` input frames of the chunks before stand in for the
    convolution's zero frames, zeros before the first chunk, and the chunk's
    own last frames replace them."""
    if spatial != 3 or kernel == 1 or not _chunked_call(parent):
        return causal_conv(features, kernel, spatial, parent.dtype, name)(x)
    held = parent.variable("cache", f"{name}_frames", jnp.zeros, (x.shape[0], kernel - 1, *x.shape[2:]),
                           x.dtype)
    x = jnp.concatenate([held.value, x], axis=1)
    held.value = x[:, 1 - kernel:]
    pad = kernel // 2
    return Conv(features, (kernel,) * 3, padding=((0, 0), (pad, pad), (pad, pad)), dtype=parent.dtype,
                name=name)(x)


def _frames(module: nn.Module, x):
    """Apply a 2-D module to every frame of `[B, T, H, W, C]`."""
    batch, frames = x.shape[:2]
    y = module(x.reshape(batch * frames, *x.shape[2:]))
    return y.reshape(batch, frames, *y.shape[1:])


class WanResidualBlock(nn.Module):
    """`WanResidualBlock` over a video `[B, T, H, W, C]` or a frame
    `[B, H, W, C]`: norm, SiLU and a 3-wide convolution twice, the input
    added back, through a 1-wide convolution where the width changes."""

    in_features: int
    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        spatial = x.ndim - 2
        shortcut = x
        if self.in_features != self.features:
            shortcut = causal_conv(self.features, 1, spatial, self.dtype, "conv_shortcut")(x)
        x = silu(WanRMSNorm(self.in_features, 4, self.dtype, name="norm1")(x))
        x = _causal(self, self.features, 3, spatial, "conv1", x)
        x = silu(WanRMSNorm(self.features, 4, self.dtype, name="norm2")(x))
        return _causal(self, self.features, 3, spatial, "conv2", x) + shortcut


class WanAttention(nn.Module):
    """Single-head self-attention over each frame's pixels, the projections
    1x1 convolutions, over a video or a frame."""

    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        *_, height, width, channels = x.shape
        normalized = WanRMSNorm(channels, 3, self.dtype, name="norm")(x).reshape(-1, height, width, channels)
        qkv = causal_conv(3 * channels, 1, 2, self.dtype, "to_qkv")(normalized)
        frames = qkv.shape[0]
        query, key, value = jnp.split(qkv.reshape(frames, height * width, 1, 3 * channels), 3, axis=-1)
        attended = scaled_dot_product_attention(query, key, value, dtype=self.dtype).reshape(
            frames, height, width, channels)
        return causal_conv(channels, 1, 2, self.dtype, "attention_out")(attended).reshape(x.shape) + x


class WanMidBlock(nn.Module):
    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = WanResidualBlock(self.features, self.features, self.dtype, name="resnets_0")(x)
        x = WanAttention(self.dtype, name="attentions_0")(x)
        return WanResidualBlock(self.features, self.features, self.dtype, name="resnets_1")(x)


class _Downsample(nn.Module):
    """`WanResample` down: halve each frame, padded one pixel at the bottom
    and right; with `temporal`, halve time after the first frame. Chunked,
    a later chunk's time convolution reads the last frame before it first,
    as the source's does."""

    features: int
    temporal: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        conv = Conv(
            self.features, (3, 3), strides=2, padding=((0, 1), (0, 1)), dtype=self.dtype, name="resample_1"
        )
        x = _frames(conv, x)
        if not self.temporal:
            return x
        time_conv = Conv(self.features, (3, 1, 1), strides=(2, 1, 1), padding="VALID", dtype=self.dtype,
                         name="time_conv")
        if _chunked_call(self):
            # No frame held yet: this chunk opens the video.
            started = self.has_variable("cache", "time_frames")
            held = self.variable("cache", "time_frames", jnp.zeros, (x.shape[0], 1, *x.shape[2:]), x.dtype)
            previous, held.value = held.value, x[:, -1:]
            if started:
                return time_conv(jnp.concatenate([previous, x], axis=1))
        return jnp.concatenate([x[:, :1], time_conv(x)], axis=1)


class _Upsample(nn.Module):
    """`WanResample` up: with `temporal`, every frame after the first becomes
    two; then each frame doubles in size and halves its channels. Chunked,
    the time convolution carries the last two frames after the first."""

    features: int
    temporal: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.temporal:
            cached = _chunked_call(self)
            time_conv = Conv(2 * self.features, (3, 1, 1), padding=((0 if cached else 2, 0), (0, 0), (0, 0)),
                             dtype=self.dtype, name="time_conv")
            # The first frame passes through, so the time convolution reads
            # the frames after it; chunked, every frame of a later chunk.
            started = cached and self.has_variable("cache", "time_frames")
            first, later = (x[:, :0], x) if started else (x[:, :1], x[:, 1:])
            if cached:
                held = self.variable("cache", "time_frames", jnp.zeros, (x.shape[0], 2, *x.shape[2:]),
                                     x.dtype)
                later = jnp.concatenate([held.value, later], axis=1)
                held.value = later[:, -2:]
            if later.shape[1] > (2 if cached else 0):
                batch, _, height, width, channels = later.shape
                later = time_conv(later)
                frames = later.shape[1]
                later = later.reshape(batch, frames, height, width, 2, channels)
                later = later.transpose(0, 1, 4, 2, 3, 5).reshape(batch, 2 * frames, height, width, channels)
                x = jnp.concatenate([first, later], axis=1)
        # Each frame doubled by nearest resize and convolved to half the channels (`blocks.Upsample`), its
        # convolution this module's `resample_1`, as the published Wan VAE names it.
        upsample = Upsample(self.features // 2, 2, dtype=self.dtype, conv="resample_1")
        nn.share_scope(upsample, self)
        return _frames(upsample, x)


class _Encoder(nn.Module):
    base: int
    latent: int
    multipliers: tuple[int, ...]
    blocks: int
    temporal: tuple[bool, ...]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dims = [self.base * m for m in (1, *self.multipliers)]
        x = _causal(self, dims[0], 3, 3, "conv_in", x)
        index = 0
        for level, (in_features, features) in enumerate(pairwise(dims)):
            for _ in range(self.blocks):
                x = WanResidualBlock(in_features, features, self.dtype, name=f"down_blocks_{index}")(x)
                in_features, index = features, index + 1
            if level != len(self.multipliers) - 1:
                x = _Downsample(features, self.temporal[level], self.dtype, name=f"down_blocks_{index}")(x)
                index += 1
        x = WanMidBlock(dims[-1], self.dtype, name="mid_block")(x)
        x = silu(WanRMSNorm(dims[-1], 4, self.dtype, name="norm_out")(x))
        return _causal(self, 2 * self.latent, 3, 3, "conv_out", x)


class _UpBlock(nn.Module):
    in_features: int
    features: int
    blocks: int
    upsample: bool
    temporal: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        in_features = self.in_features
        for index in range(self.blocks + 1):
            x = WanResidualBlock(in_features, self.features, self.dtype, name=f"resnets_{index}")(x)
            in_features = self.features
        if self.upsample:
            x = _Upsample(self.features, self.temporal, self.dtype, name="upsamplers_0")(x)
        return x


class _Decoder(nn.Module):
    base: int
    multipliers: tuple[int, ...]
    blocks: int
    temporal: tuple[bool, ...]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, z):
        dims = [self.base * m for m in (self.multipliers[-1], *self.multipliers[::-1])]
        x = _causal(self, dims[0], 3, 3, "conv_in", z)
        x = WanMidBlock(dims[0], self.dtype, name="mid_block")(x)
        upsampling = self.temporal[::-1]
        for level, (in_features, features) in enumerate(pairwise(dims)):
            upsample = level != len(self.multipliers) - 1
            x = _UpBlock(in_features // 2 if level > 0 else in_features, features, self.blocks, upsample,
                         upsample and upsampling[level], self.dtype, name=f"up_blocks_{level}")(x)
        x = silu(WanRMSNorm(dims[-1], 4, self.dtype, name="norm_out")(x))
        return _causal(self, 3, 3, 3, "conv_out", x)


def _posterior(model: WanVAE, params, video, key=None, noise=None):
    """A video's posterior draw: the whole video's moments, its chunks assembled, and its noise whole."""
    return posterior_latent(chunked_moments(model, {"params": params}, video), key, noise)


def _chunk_decoded(model: WanVAE, params, latents):
    """A video's latents decoded chunk by chunk (`chunked_decode`)."""
    return chunked_decode(model, {"params": params}, latents)


class WanVAE(nn.Module):
    """`AutoencoderKLWan` on `[B, T, H, W, 3]` videos of 1 + 4k frames;
    normalization belongs to the AutoEncoder."""

    base: int = 96
    latent: int = 16
    multipliers: tuple[int, ...] = (1, 2, 4, 4)
    blocks: int = 2
    temporal: tuple[bool, ...] = (False, True, True)
    dtype: Dtype = jnp.float32

    @property
    def downscale_factor(self) -> int:
        return 2 ** len(self.temporal)

    @property
    def temporal_factor(self) -> int:
        return 2 ** sum(self.temporal)

    def setup(self):
        self.encoder = _Encoder(self.base, self.latent, self.multipliers, self.blocks, self.temporal,
                                self.dtype)
        self.quant_conv = causal_conv(2 * self.latent, 1, 3, self.dtype, "quant_conv")
        self.post_quant_conv = causal_conv(self.latent, 1, 3, self.dtype, "post_quant_conv")
        self.decoder = _Decoder(self.base, self.multipliers, self.blocks, self.temporal, self.dtype)

    def moments(self, video):
        """The posterior's mean and log-variance, stacked on the channel axis.
        A chunk (`chunked_moments`) is any run of frames after the first."""
        if (video.shape[1] - 1) % self.temporal_factor and not _chunked_call(self):
            raise ValueError(f"a Wan VAE encodes 1 + {self.temporal_factor}k frames, not {video.shape[1]}")
        return self.quant_conv(self.encoder(video))

    def encode(self, video, key=None, noise=None):
        return posterior_latent(self.moments(video), key, noise)

    def decode(self, latents):
        return jnp.clip(self.decoder(self.post_quant_conv(latents)), -1.0, 1.0)

    def __call__(self, video):
        return self.decode(self.encode(video))


def _chunked(model: WanVAE, variables: Variables, sequence, step: int, method):
    """`method` over `sequence` `[B, T, ...]` as the source walks it: the
    first frame alone, then `step` frames at a time in one `lax.scan`, each
    convolution's last frames carried between chunks in the "cache"
    collection, the outputs joined in time."""
    head, cache = model.apply(variables, sequence[:, :1], method=method, mutable=["cache"])
    if sequence.shape[1] == 1:
        return head
    batch, frames = sequence.shape[:2]
    chunks = sequence[:, 1:].reshape(batch, (frames - 1) // step, step, *sequence.shape[2:])
    chunks = jnp.moveaxis(chunks, 1, 0)

    def walk(cache, chunk):
        out, cache = model.apply({**variables, **cache}, chunk, method=method, mutable=["cache"])
        return cache, out

    _, tail = jax.lax.scan(walk, cache, chunks)
    tail = jnp.moveaxis(tail, 0, 1)
    return jnp.concatenate([head, tail.reshape(batch, -1, *tail.shape[3:])], axis=1)


def chunked_moments(model: WanVAE, variables: Variables, video):
    """`WanVAE.moments` as the source encodes: the first frame, then four
    frames to a latent frame at a time, holding one chunk's activations."""
    if (video.shape[1] - 1) % model.temporal_factor:
        raise ValueError(f"a Wan VAE encodes 1 + {model.temporal_factor}k frames, not {video.shape[1]}")
    return _chunked(model, variables, video, model.temporal_factor, WanVAE.moments)


def chunked_decode(model: WanVAE, variables: Variables, latents):
    """`WanVAE.decode` as the source decodes: one latent frame at a time,
    holding one chunk's activations rather than the clip's."""
    return _chunked(model, variables, latents, 1, WanVAE.decode)


class WanAutoencoder(AutoEncoder):
    """Native Wan 2.1 VAE weights and the pipelines' per-channel latent
    normalization, `(z - latents_mean) * (1 / latents_std)` on the way out
    and `z / (1 / latents_std) + latents_mean` on the way in.

    A video `[B, T, H, W, C]` is encoded as one causal sequence, 1 + 4k
    frames to 1 + k latent frames, and an image `[B, H, W, C]` as a
    one-frame video, both in the source's chunks (`chunked_moments`,
    `chunked_decode`), so a clip's length does not set the memory they hold.
    """

    def __init__(self, *, model: WanVAE, params: Variables, latents_mean, latents_std):
        self.model = model
        self.params = params
        mean, std = np.asarray(latents_mean, np.float32), np.asarray(latents_std, np.float32)
        expected = (model.latent,)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                f"latents_mean {mean.shape} and latents_std {std.shape} "
                f"must both hold one value per latent channel {expected}"
            )
        self.normalization = LatentNormalization({'latents_mean': mean, 'latents_std': std}, mean, 1.0 / std)
        self._encode = functools.partial(program(_posterior, model, static_argnums=0), model)
        self._decode = functools.partial(program(_chunk_decoded, model, static_argnums=0), model)

    def record_arguments(self) -> dict:
        return dict(self.normalization.record)

    @property
    def downscale_factor(self) -> int:
        return self.model.downscale_factor

    @property
    def latent_channels(self) -> int:
        return self.model.latent

    def latent_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(shape) == 3:
            return super().latent_shape(shape)
        frames, *image = shape
        if frames < 1 or (frames - 1) % self.model.temporal_factor:
            raise ValueError(f"a Wan VAE encodes 1 + {self.model.temporal_factor}k frames, not {frames}")
        return ((frames - 1) // self.model.temporal_factor + 1, *super().latent_shape(tuple(image)))

    def encode_batch(self, params, x, key=None, noise=None):
        return self._encode(params, x[:, None], key, None if noise is None else noise[:, None])[:, 0]

    def decode_batch(self, params, z):
        return self._decode(params, z[:, None])[:, 0]

    def encode_video(self, params, x, key=None, noise=None):
        return self._encode(params, x, key, noise)

    def decode_video(self, params, z):
        return self._decode(params, z)


