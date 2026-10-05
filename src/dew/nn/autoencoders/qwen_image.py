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

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew import records
from dew.interop.components import bind_component
from dew.objectives.base import Variables

from .api import ModuleAutoEncoder
from .kl import posterior_latent
from .vae import FlaxDownsample2D, FlaxUpsample2D
from .wan import WanMidBlock, WanResidualBlock, WanRMSNorm, causal_conv

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


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
        return FlaxUpsample2D(self.features, dtype=jnp.dtype(self.dtype), name="resample")(x)


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
        x = nn.silu(WanRMSNorm(widths[-1], 4, self.dtype, name="norm_out")(x))
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
        x = nn.silu(WanRMSNorm(widths[-1], 4, self.dtype, name="norm_out")(x))
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

    def encode(self, x, key=None):
        """The posterior mean, or a draw from it when `key` is given."""
        return posterior_latent(self.quant_conv(self.encoder(x)), key)

    def decode(self, z):
        return jnp.clip(self.decoder(self.post_quant_conv(z)), -1.0, 1.0)

    def __call__(self, x, key=None):
        return self.decode(self.encode(x, key))


class QwenImageVAEFields(TypedDict):
    base_dim: int
    decoder_base_dim: int
    latent_channels: int
    dim_mult: tuple[int, ...]
    num_res_blocks: int
    temporal_downsample: tuple[bool, ...]
    image_channels: int


def qwen_image_vae_fields(config: Mapping[str, object]) -> QwenImageVAEFields:
    """Read an `AutoencoderKLQwenImage21` config into `QwenImageVAE` fields.

    Refused: `is_residual: false`, whose stack is laid out differently (flat
    down blocks, attention at `attn_scales`); a non-zero `dropout`; a
    `patch_size`, whose patchify this module does not run; different input
    and output channels; a `scale_factor_spatial` the levels do not produce;
    and widths the average and duplicate shortcuts cannot group. The
    residual stack reads no `attn_scales`, so it is accepted as the source
    accepts it.
    """
    if config.get("is_residual", True) is not True:
        raise ValueError("Only the residual Qwen-Image VAE stack (is_residual: true) is implemented")
    if config.get("dropout", 0.0) != 0.0:
        raise ValueError(f"The Qwen-Image VAE runs without dropout, got {config.get('dropout')!r}")
    if config.get("patch_size") is not None:
        raise ValueError(f"The Qwen-Image VAE's patchified layout is not implemented, got "
                         f"patch_size {config.get('patch_size')!r}")
    base_dim = records.integer(config.get("base_dim"), "base_dim")
    decoder_base_dim = (base_dim if config.get("decoder_base_dim") is None
                        else records.integer(config["decoder_base_dim"], "decoder_base_dim"))
    dim_mult = records.integers(config.get("dim_mult"), "dim_mult")
    halving = config.get("temperal_downsample")
    if not isinstance(halving, (list, tuple)):
        raise ValueError(f"temperal_downsample={halving!r}: this field is a list of flags")
    temporal = tuple(records.boolean(halves_time, "temperal_downsample") for halves_time in halving)
    if not dim_mult or len(temporal) != len(dim_mult) - 1:
        raise ValueError(f"temperal_downsample needs one flag per downsampling level of dim_mult {dim_mult}, "
                         f"got {temporal}")
    image_channels = records.integer(config.get("in_channels"), "in_channels")
    if records.integer(config.get("out_channels"), "out_channels") != image_channels:
        raise ValueError("The Qwen-Image VAE must reconstruct its input channels: in_channels "
                         f"{image_channels} against out_channels {config.get('out_channels')}")
    spatial = config.get("scale_factor_spatial")
    if spatial is not None and spatial != 2 ** (len(dim_mult) - 1):
        raise ValueError(f"scale_factor_spatial {spatial} is not the {2 ** (len(dim_mult) - 1)} "
                         f"the {len(dim_mult)} levels of dim_mult produce")
    encoder = [base_dim * multiple for multiple in (1, *dim_mult)]
    decoder = [decoder_base_dim * multiple for multiple in (dim_mult[-1], *dim_mult[::-1])]
    for index, halves_time in enumerate((*temporal, False)):
        down = index < len(temporal)
        factor = (2 if halves_time else 1) * (4 if down else 1)
        if encoder[index] * factor % encoder[index + 1]:
            raise ValueError(f"Encoder level {index} cannot average {encoder[index]} x {factor} phases "
                             f"into {encoder[index + 1]} channels")
    for index, doubles_time in enumerate(temporal[::-1]):
        if decoder[index + 1] * (2 if doubles_time else 1) * 4 % decoder[index]:
            raise ValueError(f"Decoder level {index} cannot duplicate {decoder[index]} channels into "
                             f"{decoder[index + 1]}")
    return QwenImageVAEFields(base_dim=base_dim, decoder_base_dim=decoder_base_dim,
                              latent_channels=records.integer(config.get("z_dim"), "z_dim"),
                              dim_mult=dim_mult,
                              num_res_blocks=records.integer(config.get("num_res_blocks"), "num_res_blocks"),
                              temporal_downsample=temporal, image_channels=image_channels)


_PARAMETER = r"\.(?:weight|bias)"
_RESNET = rf"resnets\.\d+\.(?:norm[12]\.gamma|(?:conv[12]|conv_shortcut){_PARAMETER})"
_ATTENTION = rf"attentions\.\d+\.(?:norm\.gamma|(?:to_qkv|proj){_PARAMETER})"
_RESAMPLE = rf"(?:downsampler|upsampler)\.(?:resample\.1|time_conv){_PARAMETER}"
_NAME = re.compile(
    rf"(?:quant_conv|post_quant_conv){_PARAMETER}"
    rf"|(?:encoder|decoder)\.(?:(?:conv_in|conv_out){_PARAMETER}|norm_out\.gamma"
    rf"|mid_block\.(?:{_RESNET}|{_ATTENTION})"
    rf"|(?:down_blocks|up_blocks)\.\d+\.(?:{_RESNET}|{_RESAMPLE}))")
_INDEXED = {"down_blocks", "up_blocks", "resnets", "attentions"}
_LEAF_RANK = {"weight": (4,), "bias": (1,), "gamma": (3, 4)}


def qwen_image_vae_path(name: str, ndim: int) -> tuple[str, ...]:
    """The `QwenImageVAE` parameter path of a published tensor.

    A weight is a 4-D convolution kernel, which `record_layouts` transposes
    to HWIO; a gamma keeps its stored `[C, 1, 1(, 1)]` shape.
    """
    if _NAME.fullmatch(name) is None:
        raise ValueError(f"{name!r} is not an AutoencoderKLQwenImage21 tensor")
    # The attention's output conv is `proj` in the source; a bare `proj`
    # names the sharded feed-forward projections elsewhere in Dew.
    parts = name.replace("resample.1", "resample.conv").replace(".proj.", ".attention_out.").split(".")
    leaf = parts.pop()
    if ndim not in _LEAF_RANK[leaf]:
        raise ValueError(f"{name!r} has {ndim} axes, not {' or '.join(map(str, _LEAF_RANK[leaf]))}")
    path: list[str] = []
    for part in parts:
        if part.isdigit() and path and path[-1] in _INDEXED:
            path[-1] = f"{path[-1]}_{part}"
        else:
            path.append(part)
    return (*path, "kernel" if leaf == "weight" else leaf)


class QwenImageAutoencoder(ModuleAutoEncoder[QwenImageVAE]):
    """Native Qwen-Image 2.1 VAE weights and the pipeline's per-channel
    latent normalization, `(z - latents_mean) / latents_std` on the way out
    and `z * latents_std + latents_mean` on the way in."""

    def __init__(self, *, model: QwenImageVAE, params: Variables,
                 latents_mean: Sequence[float], latents_std: Sequence[float]):
        super().__init__(model, params)
        self.latents_mean = np.asarray(latents_mean, np.float32)
        self.latents_std = np.asarray(latents_std, np.float32)
        expected = (model.latent_channels,)
        if self.latents_mean.shape != expected or self.latents_std.shape != expected:
            raise ValueError(
                f"latents_mean {self.latents_mean.shape} and latents_std {self.latents_std.shape} "
                f"must both hold one value per latent channel {expected}"
            )
    @property
    def downscale_factor(self) -> int:
        return self.model.downscale_factor

    @property
    def latent_channels(self) -> int:
        return self.model.latent_channels

    # The base class's shift and scale stay 0 and 1, so its encode and decode
    # only flatten video around the batch calls here.
    def encode(self, params, x, key=None):
        latent = super().encode(params, x, key=key)
        return (latent - self.latents_mean.astype(latent.dtype)) / self.latents_std.astype(latent.dtype)

    def decode(self, params, z):
        return super().decode(
            params, z * self.latents_std.astype(z.dtype) + self.latents_mean.astype(z.dtype)
        )


def load_qwen_image_vae(directory: Path, compute, *, param_dtype: str = "float32",
                        params: Variables | None = None, lazy: bool = False
                        ) -> tuple[QwenImageAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build the published autoencoder, its parameters and their source layouts
    from `directory/vae`. Every tensor the module declares must be published,
    in the shape it declares. Supplied `params` are bound without a weight read;
    `lazy` leaves read ones `SourceLeaf`s for a placement to read
    (`dew.interop.weights.record_layouts`)."""
    config = json.loads((directory / "vae" / "config.json").read_text())
    model = QwenImageVAE(**qwen_image_vae_fields(config), dtype=compute)
    frame = jax.ShapeDtypeStruct((1, model.downscale_factor, model.downscale_factor, model.image_channels),
                                 jnp.float32)
    return bind_component(
        directory / "vae", "vae", config, model, qwen_image_vae_path,
        lambda bound: QwenImageAutoencoder(model=model, params=bound, latents_mean=config["latents_mean"],
                                           latents_std=config["latents_std"]),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, inputs=(frame,))
