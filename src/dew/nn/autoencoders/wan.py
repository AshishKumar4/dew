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
from dew.interop.components import bind_component, component_source
from dew.objectives.base import Variables

from ..attention import scaled_dot_product_attention
from ..conv import Conv
from .api import AutoEncoder
from .kl import posterior_latent

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


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
        wide = x.astype(jnp.promote_types(x.dtype, jnp.float32))
        # F.normalize clamps the norm at 1e-12. Clamping its square first
        # keeps a zero row's gradient finite (sqrt's derivative at zero would
        # meet the clamp's zero); the second clamp, the same value, keeps the
        # division off a bare square root, which XLA rewrites to rsqrt.
        norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(wide), axis=-1, keepdims=True), 1e-24))
        return (wide / jnp.maximum(norm, 1e-12) * math.sqrt(self.features) * gamma.reshape(-1)).astype(
            self.dtype
        )


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
        x = nn.silu(WanRMSNorm(self.in_features, 4, self.dtype, name="norm1")(x))
        x = _causal(self, self.features, 3, spatial, "conv1", x)
        x = nn.silu(WanRMSNorm(self.features, 4, self.dtype, name="norm2")(x))
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
        x = nn.silu(WanRMSNorm(dims[-1], 4, self.dtype, name="norm_out")(x))
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
        x = nn.silu(WanRMSNorm(dims[-1], 4, self.dtype, name="norm_out")(x))
        return _causal(self, 3, 3, 3, "conv_out", x)


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

    def encode(self, video, key=None):
        return posterior_latent(self.moments(video), key)

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
        raise ValueError(
            "attention inside the levels (attn_scales) is not ported; the published VAEs have none"
        )
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
    one-frame video, both in the source's chunks (`chunked_moments`,
    `chunked_decode`), so a clip's length does not set the memory they hold.
    """

    def __init__(self, *, model: WanVAE, params: Variables, latents_mean, latents_std):
        self.model = model
        self.params = params
        self.latent_shift = np.asarray(latents_mean, np.float32)
        self.latent_scale = 1.0 / np.asarray(latents_std, np.float32)
        expected = (model.latent,)
        if self.latent_shift.shape != expected or self.latent_scale.shape != expected:
            raise ValueError(
                f"latents_mean {self.latent_shift.shape} and latents_std {self.latent_scale.shape} "
                f"must both hold one value per latent channel {expected}"
            )
        self._encode = jax.jit(lambda params, video, key=None: posterior_latent(
            chunked_moments(model, {"params": params}, video), key))
        self._decode = jax.jit(lambda params, latents: chunked_decode(model, {"params": params}, latents))

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

    def encode_batch(self, params, x, key=None):
        return self._encode(params, x[:, None], key)[:, 0]

    def decode_batch(self, params, z):
        return self._decode(params, z[:, None])[:, 0]

    def encode_video(self, params, x, key=None):
        return self._encode(params, x, key)

    def decode_video(self, params, z):
        return self._decode(params, z)


def load_wan_vae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
                 subfolder: str = "vae", param_dtype: str = "float32", params: Variables | None = None,
                 lazy: bool = False
                 ) -> tuple[WanAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published Wan 2.1 VAE, its parameters and their source layouts
    from `subfolder` of a pipeline directory or Hub repo, such as
    `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`; only that component downloads. Every
    tensor the module declares must be published, in the shape it declares.

    Supplied `params` are bound unchanged: only the config is read, and no
    source layouts are returned. `lazy` leaves read ones `SourceLeaf`s for a
    placement to read (`dew.interop.weights.record_layouts`)."""
    directory, config = component_source(name_or_dir, revision, subfolder, weights=params is None)
    if config.get("_class_name") != "AutoencoderKLWan":
        raise ValueError(
            f"{directory} holds a {config.get('_class_name')}, not an AutoencoderKLWan"
        )
    model = WanVAE(**wan_vae_fields(config), dtype=compute)
    video = jax.ShapeDtypeStruct(
        (1, 1 + model.temporal_factor, model.downscale_factor, model.downscale_factor, 3), jnp.float32
    )
    return bind_component(
        directory, "vae", config, model, wan_vae_path,
        lambda bound: WanAutoencoder(model=model, params=bound, latents_mean=config["latents_mean"],
                                     latents_std=config["latents_std"]),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, inputs=(video,))
