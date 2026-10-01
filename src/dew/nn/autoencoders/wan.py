"""Wan 2.1's causal video autoencoder, `AutoencoderKLWan`.

An independent linen port of diffusers 0.34.0
src/diffusers/models/autoencoders/autoencoder_kl_wan.py (Apache-2.0),
channels last: videos are `[B, T, H, W, C]`. Every 3-D convolution is causal
in time, padded with two zero frames in front, so frame t reads only frames
up to t. Four frames compress into one latent frame after the first, which is
compressed alone: a video of 1 + 4k frames has 1 + k latent frames, and an
image is the one-frame video.

The source walks a video in chunks (the first frame, then four at a time;
one latent frame at a time to decode) and carries each convolution's last
two input frames across chunks. That is the causal convolution over the
whole video, which is what this module computes in one pass. The resampling
blocks keep the first frame out of their temporal convolution, as the
source's first chunk does:

- a temporally downsampling block passes the first frame through and
  convolves the whole sequence with stride 2 and no padding, so output
  frame j > 0 reads frames 2j - 2 to 2j;
- a temporally upsampling block passes the first frame through and turns
  every later frame into two, by a causal convolution over the frames after
  the first that doubles the channels, each half one frame.

The decoder clamps its pixels to [-1, 1], as the source's `_decode` does.
RMS gammas keep the source's stored `[C, 1, 1(, 1)]` shape.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew import records
from dew.nn.text_encoders import check_tree
from dew.objectives.base import Variables

from ..conv import Conv
from .api import AutoEncoder
from .kl import posterior_latent

if TYPE_CHECKING:
    from dew.interop.pretrained import WeightLayout

TEMPORAL = 4
"""Frames per latent frame after the first."""


class _RMSNorm(nn.Module):
    """`WanRMS_norm`: the channel vector scaled to unit L2 norm, times sqrt(C)
    and a gamma stored `[C]` followed by `rank - 1` unit axes."""

    features: int
    rank: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        gamma = self.param("gamma", nn.initializers.ones, (self.features, *(1,) * (self.rank - 1)))
        norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))
        return (x / jnp.maximum(norm, 1e-12) * math.sqrt(self.features) * gamma.reshape(-1)).astype(self.dtype)


def _causal(features: int, kernel: int, dtype: Dtype, name: str) -> nn.Module:
    """A cubic causal convolution; the source's parameters sit directly under
    `name`, so the module is the Flax convolution itself."""
    pad = kernel // 2
    return Conv(features, (kernel,) * 3, padding=((kernel - 1, 0), (pad, pad), (pad, pad)),
                dtype=dtype, name=name)


def _frames(module: nn.Module, x):
    """Apply a 2-D module to every frame of `[B, T, H, W, C]`."""
    batch, frames = x.shape[:2]
    y = module(x.reshape(batch * frames, *x.shape[2:]))
    return y.reshape(batch, frames, *y.shape[1:])


class _ResidualBlock(nn.Module):
    in_features: int
    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        shortcut = x
        if self.in_features != self.features:
            shortcut = _causal(self.features, 1, self.dtype, "conv_shortcut")(x)
        x = nn.silu(_RMSNorm(self.in_features, 4, self.dtype, name="norm1")(x))
        x = _causal(self.features, 3, self.dtype, "conv1")(x)
        x = nn.silu(_RMSNorm(self.features, 4, self.dtype, name="norm2")(x))
        return _causal(self.features, 3, self.dtype, "conv2")(x) + shortcut


class _Attention(nn.Module):
    """Single-head self-attention over each frame's pixels, the projections
    1x1 convolutions."""

    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        batch, frames, height, width, channels = x.shape
        normalized = _RMSNorm(channels, 3, self.dtype, name="norm")(x)
        qkv = Conv(3 * channels, (1, 1), dtype=self.dtype, name="to_qkv")(
            normalized.reshape(batch * frames, height, width, channels))
        query, key, value = jnp.split(qkv.reshape(batch * frames, height * width, 1, 3 * channels), 3, axis=-1)
        attended = jax.nn.dot_product_attention(query, key, value).reshape(batch * frames, height, width, channels)
        out = Conv(channels, (1, 1), dtype=self.dtype, name="attention_out")(attended)
        return out.reshape(x.shape) + x


class _MidBlock(nn.Module):
    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = _ResidualBlock(self.features, self.features, self.dtype, name="resnets_0")(x)
        x = _Attention(self.features, self.dtype, name="attentions_0")(x)
        return _ResidualBlock(self.features, self.features, self.dtype, name="resnets_1")(x)


class _Downsample(nn.Module):
    """`WanResample` down: halve each frame, padded one pixel at the bottom
    and right; with `temporal`, halve time after the first frame."""

    features: int
    temporal: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        conv = Conv(self.features, (3, 3), strides=2, padding=((0, 1), (0, 1)), dtype=self.dtype, name="resample_1")
        x = _frames(conv, x)
        if not self.temporal:
            return x
        later = Conv(self.features, (3, 1, 1), strides=(2, 1, 1), padding="VALID", dtype=self.dtype,
                     name="time_conv")(x)
        return jnp.concatenate([x[:, :1], later], axis=1)


class _Upsample(nn.Module):
    """`WanResample` up: with `temporal`, every frame after the first becomes
    two; then each frame doubles in size and halves its channels."""

    features: int
    temporal: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.temporal:
            time_conv = Conv(2 * self.features, (3, 1, 1), padding=((2, 0), (0, 0), (0, 0)), dtype=self.dtype,
                             name="time_conv")
            if x.shape[1] > 1:
                batch, frames, height, width, channels = x.shape
                later = time_conv(x[:, 1:]).reshape(batch, frames - 1, height, width, 2, channels)
                later = later.transpose(0, 1, 4, 2, 3, 5).reshape(batch, 2 * (frames - 1), height, width, channels)
                x = jnp.concatenate([x[:, :1], later], axis=1)
        x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
        conv = Conv(self.features // 2, (3, 3), padding=((1, 1), (1, 1)), dtype=self.dtype, name="resample_1")
        return _frames(conv, x)


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
        x = _causal(dims[0], 3, self.dtype, "conv_in")(x)
        index = 0
        for level, (in_features, features) in enumerate(pairwise(dims)):
            for _ in range(self.blocks):
                x = _ResidualBlock(in_features, features, self.dtype, name=f"down_blocks_{index}")(x)
                in_features, index = features, index + 1
            if level != len(self.multipliers) - 1:
                x = _Downsample(features, self.temporal[level], self.dtype, name=f"down_blocks_{index}")(x)
                index += 1
        x = _MidBlock(dims[-1], self.dtype, name="mid_block")(x)
        x = nn.silu(_RMSNorm(dims[-1], 4, self.dtype, name="norm_out")(x))
        return _causal(2 * self.latent, 3, self.dtype, "conv_out")(x)


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
            x = _ResidualBlock(in_features, self.features, self.dtype, name=f"resnets_{index}")(x)
            in_features = self.features
        if self.upsample:
            x = _Upsample(self.features, self.temporal, self.dtype, name="upsamplers_0")(x)
        return x


class _Decoder(nn.Module):
    base: int
    latent: int
    multipliers: tuple[int, ...]
    blocks: int
    temporal: tuple[bool, ...]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, z):
        dims = [self.base * m for m in (self.multipliers[-1], *self.multipliers[::-1])]
        x = _causal(dims[0], 3, self.dtype, "conv_in")(z)
        x = _MidBlock(dims[0], self.dtype, name="mid_block")(x)
        upsampling = self.temporal[::-1]
        for level, (in_features, features) in enumerate(pairwise(dims)):
            upsample = level != len(self.multipliers) - 1
            x = _UpBlock(in_features // 2 if level > 0 else in_features, features, self.blocks, upsample,
                         upsample and upsampling[level], self.dtype, name=f"up_blocks_{level}")(x)
        x = nn.silu(_RMSNorm(dims[-1], 4, self.dtype, name="norm_out")(x))
        return _causal(3, 3, self.dtype, "conv_out")(x)


class WanVAEFields(TypedDict):
    base: int
    latent: int
    multipliers: tuple[int, ...]
    blocks: int
    temporal: tuple[bool, ...]


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
        fields = (self.base, self.latent, self.multipliers, self.blocks, self.temporal, self.dtype)
        self.encoder = _Encoder(*fields)
        self.quant_conv = _causal(2 * self.latent, 1, self.dtype, "quant_conv")
        self.post_quant_conv = _causal(self.latent, 1, self.dtype, "post_quant_conv")
        self.decoder = _Decoder(*fields)

    def moments(self, video):
        """The posterior's mean and log-variance, stacked on the channel axis."""
        if (video.shape[1] - 1) % self.temporal_factor:
            raise ValueError(f"a Wan VAE encodes 1 + {self.temporal_factor}k frames, not {video.shape[1]}")
        return self.quant_conv(self.encoder(video))

    def encode(self, video, key=None):
        return posterior_latent(self.moments(video), key)

    def decode(self, latents):
        return jnp.clip(self.decoder(self.post_quant_conv(latents)), -1.0, 1.0)

    def __call__(self, video):
        return self.decode(self.encode(video))


def wan_vae_fields(config: Mapping[str, object]) -> WanVAEFields:
    """Read an `AutoencoderKLWan` config into the fields `WanVAE` takes,
    refusing what the port does not compute."""
    multipliers = records.integers(config["dim_mult"], "dim_mult")
    halvings = config["temperal_downsample"]
    if not isinstance(halvings, (list, tuple)):
        raise ValueError(f"temperal_downsample={halvings!r}: this field is one flag per downsampling")
    temporal = tuple(records.boolean(halves, "temperal_downsample") for halves in halvings)
    if len(temporal) != len(multipliers) - 1:
        raise ValueError(f"temperal_downsample has {len(temporal)} entries for {len(multipliers)} levels")
    if config.get("attn_scales"):
        raise ValueError("attention inside the levels (attn_scales) is not ported; the published VAEs have none")
    if config.get("dropout", 0.0):
        raise ValueError("a frozen autoencoder runs without dropout")
    for name in ("is_residual", "patch_size"):
        if config.get(name):
            raise ValueError(f"{name} belongs to Wan 2.2's VAE, which this port does not compute")
    return WanVAEFields(base=records.integer(config["base_dim"], "base_dim"),
                        latent=records.integer(config["z_dim"], "z_dim"), multipliers=multipliers,
                        blocks=records.integer(config["num_res_blocks"], "num_res_blocks"), temporal=temporal)


_LEAF_RANK = {"weight": (4, 5), "bias": (1,), "gamma": (3, 4)}


def wan_vae_path(name: str, ndim: int) -> tuple[str, ...]:
    """The parameter path of one `AutoencoderKLWan` tensor. Digits join the
    name before them; a 4-D or 5-D weight is a convolution kernel, and a gamma
    keeps its stored shape. The attention's output `proj` is `attention_out`:
    a bare `proj` names a sharded projection elsewhere in Dew."""
    parts = name.split(".")
    leaf = parts.pop()
    if parts[0] not in ("encoder", "decoder", "quant_conv", "post_quant_conv") or leaf not in _LEAF_RANK:
        raise ValueError(f"{name!r} is not an AutoencoderKLWan tensor")
    if ndim not in _LEAF_RANK[leaf]:
        raise ValueError(f"{name!r} has {ndim} axes, not {' or '.join(map(str, _LEAF_RANK[leaf]))}")
    path: list[str] = []
    for part in parts:
        if part.isdigit() and path:
            path[-1] = f"{path[-1]}_{part}"
        else:
            path.append("attention_out" if part == "proj" else part)
    return (*path, "kernel" if leaf == "weight" else leaf)


class WanAutoencoder(AutoEncoder):
    """Native Wan 2.1 VAE weights and the pipelines' per-channel latent
    normalization, `(z - latents_mean) * (1 / latents_std)` on the way out
    and `z / (1 / latents_std) + latents_mean` on the way in.

    A video `[B, T, H, W, C]` is encoded as one causal sequence, 1 + 4k
    frames to 1 + k latent frames, and an image `[B, H, W, C]` as a
    one-frame video.
    """

    def __init__(self, *, model: WanVAE, params: Variables, latents_mean, latents_std):
        self.model = model
        self.params = params
        self.latent_shift = np.asarray(latents_mean, np.float32)
        self.latent_scale = 1.0 / np.asarray(latents_std, np.float32)
        expected = (model.latent,)
        if self.latent_shift.shape != expected or self.latent_scale.shape != expected:
            raise ValueError(f"latents_mean {self.latent_shift.shape} and latents_std {self.latent_scale.shape} "
                             f"must both hold one value per latent channel {expected}")
        self._encode = jax.jit(lambda params, video, key=None: model.apply(
            {"params": params}, video, key, method=model.encode))
        self._decode = jax.jit(lambda params, latents: model.apply(
            {"params": params}, latents, method=model.decode))

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
        return ((frames - 1) // self.model.temporal_factor + 1, *super().latent_shape(tuple(image)))

    def encode_batch(self, params, x, key=None):
        return self._encode(params, x[:, None], key)[:, 0]

    def decode_batch(self, params, z):
        return self._decode(params, z[:, None])[:, 0]

    def encode_video(self, params, x, key=None):
        return self._encode(params, x, key)

    def decode_video(self, params, z):
        return self._decode(params, z)


def load_wan_vae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
                 subfolder: str = "vae", param_dtype: str = "float32", params: Variables | None = None
                 ) -> tuple[WanAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published Wan 2.1 VAE, its parameters and their source layouts
    from `subfolder` of a pipeline directory or Hub repo, such as
    `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`; only that component downloads. Every
    tensor the module declares must be published, in the shape it declares.

    Supplied `params` are bound unchanged: only the config is read, and no
    source layouts are returned."""
    from dew.interop import diffusion, sources

    directory = sources.snapshot(str(name_or_dir), revision, weights=(subfolder,) if params is None else False)
    config = json.loads((directory / subfolder / "config.json").read_text())
    if config.get("_class_name") != "AutoencoderKLWan":
        raise ValueError(f"{directory / subfolder} holds a {config.get('_class_name')}, not an AutoencoderKLWan")
    model = WanVAE(**wan_vae_fields(config), dtype=compute)
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tensors = diffusion.component_tensors(directory, subfolder)
        params, layouts = diffusion.record_layouts(
            "vae", tensors, lambda name: wan_vae_path(name, np.ndim(tensors[name])), ("autoencoder",),
            param_dtype=param_dtype)
    video = jax.ShapeDtypeStruct((1, 1 + model.temporal_factor, model.downscale_factor, model.downscale_factor, 3),
                                 jnp.float32)
    check_tree({"params": params}, model, video)
    autoencoder = WanAutoencoder(model=model, params=params, latents_mean=config["latents_mean"],
                                 latents_std=config["latents_std"])
    return autoencoder, params, layouts, config
