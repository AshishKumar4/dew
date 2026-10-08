"""The deep compression autoencoder (DC-AE) of SANA, `AutoencoderDC`.

An independent linen port of diffusers 0.34.0 autoencoder_dc.py
(Apache-2.0), NHWC; `encode` is deterministic and ignores a key. Levels hold
`ResBlock`s or EfficientViT blocks (ReLU linear attention with an optional
depthwise multiscale branch, then a `GLUMBConv`), joined by resampling
blocks whose shortcuts average channel groups down and repeat channels up
around a pixel (un)shuffle. Kept as the source has them: the attention reads
its heads from the concatenated `[q | k | v]` channels in runs of
`3 * head_dim`, and attends quadratically over at most `head_dim` positions,
linearly in float32 over more. A batch norm (the `in-1.0` decoder's) uses
its running statistics as eval mode does, held as parameters; the frozen
autoencoder never reads `num_batches_tracked`.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype

from dew import records
from dew.interop.components import bind_component, component_source
from dew.objectives.base import Variables

from ..conv import Conv
from ..scan_orders import pixel_shuffle, pixel_unshuffle
from .api import ModuleAutoEncoder

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

_EPS = 1e-5
_ATTENTION_EPS = 1e-15
_ACTIVATIONS = {"relu": nn.relu, "relu6": nn.relu6, "silu": nn.silu}


class _RMSNorm(nn.Module):
    """diffusers' `RMSNorm` with a bias, over the channel axis."""

    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.features,))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x * jax.lax.rsqrt(variance + _EPS) * scale + bias).astype(self.dtype)


class _BatchNorm(nn.Module):
    """`BatchNorm2d` in eval mode: the running statistics normalize."""

    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.features,))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        mean = self.param("running_mean", nn.initializers.zeros, (self.features,))
        variance = self.param("running_var", nn.initializers.ones, (self.features,))
        return ((x - mean) * jax.lax.rsqrt(variance + _EPS) * scale + bias).astype(self.dtype)


def _norm(kind: str, features: int, dtype: Dtype, name: str) -> nn.Module:
    return (_RMSNorm if kind == "rms_norm" else _BatchNorm)(features, dtype, name=name)


def _conv(features: int, kernel: int, dtype: Dtype, name: str, *, stride: int = 1, groups: int = 1,
          bias: bool = True) -> Conv:
    pad = kernel // 2
    return Conv(features, (kernel, kernel), strides=stride, padding=((pad, pad), (pad, pad)),
                feature_group_count=groups, use_bias=bias, dtype=dtype, name=name)


def _group_mean(x, features: int):
    """Each of `features` outputs is the mean of a run of adjacent channels."""
    return x.reshape(*x.shape[:-1], features, -1).mean(axis=-1)


class _ResBlock(nn.Module):
    features: int
    norm: str
    activation: str
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        y = _ACTIVATIONS[self.activation](_conv(self.features, 3, self.dtype, "conv1")(x))
        y = _conv(self.features, 3, self.dtype, "conv2", bias=False)(y)
        return _norm(self.norm, self.features, self.dtype, "norm")(y) + x


class _MultiscaleProjection(nn.Module):
    """A depthwise `kernel` convolution of q, k and v, then a 1x1 one per head."""

    features: int
    heads: int
    kernel: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = _conv(self.features, self.kernel, self.dtype, "proj_in", groups=self.features, bias=False)(x)
        return _conv(self.features, 1, self.dtype, "per_head", groups=3 * self.heads, bias=False)(x)


class _LinearAttention(nn.Module):
    """`SanaMultiscaleLinearAttention` with its residual."""

    features: int
    head_dim: int
    scales: tuple[int, ...]
    norm: str
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        batch, height, width, _ = x.shape
        heads = self.features // self.head_dim
        inner = heads * self.head_dim
        qkv = jnp.concatenate([nn.Dense(inner, use_bias=False, dtype=self.dtype, name=name)(x)
                               for name in ("to_q", "to_k", "to_v")], axis=-1)
        qkv = jnp.concatenate(
            [
                qkv,
                *(
                    _MultiscaleProjection(
                        3 * inner, heads, kernel, self.dtype, name=f"to_qkv_multiscale_{index}"
                    )(qkv)
                    for index, kernel in enumerate(self.scales)
                ),
            ],
            axis=-1,
        )
        positions = height * width
        quadratic = positions <= self.head_dim
        if not quadratic:
            qkv = qkv.astype(jnp.float32)
        qkv = qkv.reshape(batch, positions, -1, 3, self.head_dim)
        query, key, value = nn.relu(qkv[..., 0, :]), nn.relu(qkv[..., 1, :]), qkv[..., 2, :]
        if quadratic:
            scores = jnp.einsum("bmgd,bngd->bgmn", key, query).astype(jnp.float32)
            scores = scores / (jnp.sum(scores, axis=2, keepdims=True) + _ATTENTION_EPS)
            attended = jnp.einsum("bmgd,bgmn->bngd", value, scores.astype(value.dtype))
        else:
            value = jnp.concatenate([value, jnp.ones_like(value[..., :1])], axis=-1)
            scores = jnp.einsum("bngd,bnge->bgde", value, key)
            attended = jnp.einsum("bgde,bnge->bngd", scores, query)
            attended = (attended[..., :-1] / (attended[..., -1:] + _ATTENTION_EPS)).astype(x.dtype)
        attended = attended.reshape(batch, height, width, -1)
        out = nn.Dense(self.features, use_bias=False, dtype=self.dtype, name="to_out")(attended)
        return _norm(self.norm, self.features, self.dtype, "norm_out")(out) + x


class _GLUMBConv(nn.Module):
    """An inverted bottleneck four times wide, gated by its own second half."""

    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        hidden = 4 * self.features
        y = nn.silu(_conv(2 * hidden, 1, self.dtype, "conv_inverted")(x))
        y = _conv(2 * hidden, 3, self.dtype, "conv_depth", groups=2 * hidden)(y)
        y, gate = jnp.split(y, 2, axis=-1)
        y = _conv(self.features, 1, self.dtype, "conv_point", bias=False)(y * nn.silu(gate))
        return _RMSNorm(self.features, self.dtype, name="norm")(y) + x


class _EfficientViTBlock(nn.Module):
    features: int
    head_dim: int
    scales: tuple[int, ...]
    norm: str
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = _LinearAttention(self.features, self.head_dim, self.scales, self.norm, self.dtype, name="attn")(x)
        return _GLUMBConv(self.features, self.dtype, name="conv_out")(x)


def _block(kind: str, features: int, head_dim: int, scales: tuple[int, ...], norm: str, activation: str,
           dtype: Dtype, name: str) -> nn.Module:
    if kind == "ResBlock":
        return _ResBlock(features, norm, activation, dtype, name=name)
    return _EfficientViTBlock(features, head_dim, scales, norm, dtype, name=name)


class _Down(nn.Module):
    """`DCDownBlock2d`: halve each spatial axis, by a strided convolution or a
    convolution then a pixel unshuffle; the shortcut averages the unshuffled
    input's channel groups."""

    features: int
    unshuffle: bool
    shortcut: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.unshuffle:
            y = pixel_unshuffle(_conv(self.features // 4, 3, self.dtype, "conv")(x))
        else:
            y = _conv(self.features, 3, self.dtype, "conv", stride=2)(x)
        return y + _group_mean(pixel_unshuffle(x), self.features) if self.shortcut else y


class _Up(nn.Module):
    """`DCUpBlock2d`: double each spatial axis, by nearest-neighbour
    interpolation then a convolution or a convolution then a pixel shuffle;
    the shortcut repeats each channel in place and shuffles."""

    in_features: int
    features: int
    interpolate: bool
    shortcut: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.interpolate:
            y = _conv(self.features, 3, self.dtype, "conv")(jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2))
        else:
            y = pixel_shuffle(_conv(4 * self.features, 3, self.dtype, "conv")(x))
        if not self.shortcut:
            return y
        return y + pixel_shuffle(jnp.repeat(x, 4 * self.features // self.in_features, axis=-1))


class _Encoder(nn.Module):
    latent_channels: int
    head_dim: int
    blocks: tuple[str, ...]
    channels: tuple[int, ...]
    layers: tuple[int, ...]
    scales: tuple[tuple[int, ...], ...]
    unshuffle: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, image):
        channels, layers = self.channels, self.layers
        if layers[0] > 0:
            x = _conv(channels[0], 3, self.dtype, "conv_in")(image)
        else:
            x = _Down(channels[1], self.unshuffle, shortcut=False, dtype=self.dtype, name="conv_in")(image)
        for level, (features, count) in enumerate(zip(channels, layers, strict=True)):
            for index in range(count):
                x = _block(
                    self.blocks[level],
                    features,
                    self.head_dim,
                    self.scales[level],
                    "rms_norm",
                    "silu",
                    self.dtype,
                    f"down_blocks_{level}_{index}",
                )(x)
            if level < len(channels) - 1 and count > 0:
                x = _Down(channels[level + 1], self.unshuffle, shortcut=True, dtype=self.dtype,
                          name=f"down_blocks_{level}_{count}")(x)
        return _conv(self.latent_channels, 3, self.dtype, "conv_out")(x) + _group_mean(
            x, self.latent_channels
        )


class _Decoder(nn.Module):
    image_channels: int
    latent_channels: int
    head_dim: int
    blocks: tuple[str, ...]
    channels: tuple[int, ...]
    layers: tuple[int, ...]
    scales: tuple[tuple[int, ...], ...]
    norms: tuple[str, ...]
    activations: tuple[str, ...]
    interpolate: bool
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, latents):
        channels, layers = self.channels, self.layers
        x = _conv(channels[-1], 3, self.dtype, "conv_in")(latents)
        x = x + jnp.repeat(latents, channels[-1] // self.latent_channels, axis=-1)
        for level in reversed(range(len(channels))):
            count = layers[level]
            upsampled = level < len(channels) - 1 and count > 0
            if upsampled:
                x = _Up(
                    channels[level + 1],
                    channels[level],
                    self.interpolate,
                    shortcut=True,
                    dtype=self.dtype,
                    name=f"up_blocks_{level}_0",
                )(x)
            for index in range(count):
                x = _block(self.blocks[level], channels[level], self.head_dim, self.scales[level],
                           self.norms[level], self.activations[level], self.dtype,
                           f"up_blocks_{level}_{index + upsampled}")(x)
        features = channels[0] if layers[0] > 0 else channels[1]
        x = nn.relu(_RMSNorm(features, self.dtype, name="norm_out")(x))
        if layers[0] > 0:
            return _conv(self.image_channels, 3, self.dtype, "conv_out")(x)
        return _Up(features, self.image_channels, self.interpolate, shortcut=False, dtype=self.dtype,
                   name="conv_out")(x)


class DCAEFields(TypedDict):
    image_channels: int
    latent_channels: int
    head_dim: int
    encoder_blocks: tuple[str, ...]
    decoder_blocks: tuple[str, ...]
    encoder_channels: tuple[int, ...]
    decoder_channels: tuple[int, ...]
    encoder_layers: tuple[int, ...]
    decoder_layers: tuple[int, ...]
    encoder_scales: tuple[tuple[int, ...], ...]
    decoder_scales: tuple[tuple[int, ...], ...]
    decoder_norms: tuple[str, ...]
    decoder_activations: tuple[str, ...]
    unshuffle: bool
    interpolate: bool


class DCAE(nn.Module):
    """`AutoencoderDC` on NHWC frames; scaling belongs to the AutoEncoder."""

    image_channels: int = 3
    latent_channels: int = 32
    head_dim: int = 32
    encoder_blocks: tuple[str, ...] = ("ResBlock",) * 3 + ("EfficientViTBlock",) * 3
    decoder_blocks: tuple[str, ...] = ("ResBlock",) * 3 + ("EfficientViTBlock",) * 3
    encoder_channels: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    decoder_channels: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    encoder_layers: tuple[int, ...] = (2, 2, 2, 3, 3, 3)
    decoder_layers: tuple[int, ...] = (3, 3, 3, 3, 3, 3)
    encoder_scales: tuple[tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,))
    decoder_scales: tuple[tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,))
    decoder_norms: tuple[str, ...] = ("rms_norm",) * 6
    decoder_activations: tuple[str, ...] = ("silu",) * 6
    unshuffle: bool = False
    interpolate: bool = True
    dtype: Dtype = jnp.float32

    @property
    def downscale_factor(self) -> int:
        return 2 ** (len(self.encoder_channels) - 1)

    def setup(self):
        self.encoder = _Encoder(
            self.latent_channels,
            self.head_dim,
            self.encoder_blocks,
            self.encoder_channels,
            self.encoder_layers,
            self.encoder_scales,
            self.unshuffle,
            self.dtype,
        )
        self.decoder = _Decoder(self.image_channels, self.latent_channels, self.head_dim, self.decoder_blocks,
                                self.decoder_channels, self.decoder_layers, self.decoder_scales,
                                self.decoder_norms, self.decoder_activations, self.interpolate, self.dtype)

    def encode(self, image, key=None):
        """The latent of `image`; the encoder is deterministic, so `key` is unused."""
        return self.encoder(image)

    def decode(self, latents):
        return self.decoder(latents)

    def __call__(self, image):
        return self.decode(self.encode(image))


def _per_level(config: Mapping[str, object], name: str, levels: int) -> tuple:
    """A config entry the source takes as one value or one per level."""
    value = config[name]
    if isinstance(value, str):
        values = (value,) * levels
    elif isinstance(value, (list, tuple)):
        values = tuple(value)
    else:
        raise ValueError(f"{name}={value!r}: this field is one name or one entry per level")
    if len(values) != levels:
        raise ValueError(f"{name} has {len(values)} entries for {levels} levels")
    return values


def dc_ae_fields(config: Mapping[str, object]) -> DCAEFields:
    """Read an `AutoencoderDC` config into the fields `DCAE` takes, refusing
    what the port does not compute."""
    encoder_channels = records.integers(config["encoder_block_out_channels"], "encoder_block_out_channels")
    decoder_channels = records.integers(config["decoder_block_out_channels"], "decoder_block_out_channels")
    levels = len(encoder_channels)
    if len(decoder_channels) != levels:
        raise ValueError("the encoder and decoder have different numbers of levels")
    fields = DCAEFields(
        image_channels=records.integer(config["in_channels"], "in_channels"),
        latent_channels=records.integer(config["latent_channels"], "latent_channels"),
        head_dim=records.integer(config["attention_head_dim"], "attention_head_dim"),
        encoder_blocks=_per_level(config, "encoder_block_types", levels),
        decoder_blocks=_per_level(config, "decoder_block_types", levels),
        encoder_channels=encoder_channels, decoder_channels=decoder_channels,
        encoder_layers=_per_level(config, "encoder_layers_per_block", levels),
        decoder_layers=_per_level(config, "decoder_layers_per_block", levels),
        encoder_scales=tuple(map(tuple, _per_level(config, "encoder_qkv_multiscales", levels))),
        decoder_scales=tuple(map(tuple, _per_level(config, "decoder_qkv_multiscales", levels))),
        decoder_norms=_per_level(config, "decoder_norm_types", levels),
        decoder_activations=_per_level(config, "decoder_act_fns", levels),
        unshuffle=config["downsample_block_type"] == "pixel_unshuffle",
        interpolate=config["upsample_block_type"] == "interpolate")
    checks = [
        (
            "block types",
            {*fields["encoder_blocks"], *fields["decoder_blocks"]},
            {"ResBlock", "EfficientViTBlock"},
        ),
        ("decoder norm types", set(fields["decoder_norms"]), {"rms_norm", "batch_norm"}),
        ("decoder activations", set(fields["decoder_activations"]), set(_ACTIVATIONS)),
        ("downsample_block_type", {config["downsample_block_type"]}, {"pixel_unshuffle", "Conv"}),
        ("upsample_block_type", {config["upsample_block_type"]}, {"pixel_shuffle", "interpolate"}),
    ]
    for what, found, known in checks:
        if not found <= known:
            raise ValueError(
                f"{what} {sorted(found - known)} are not ones AutoencoderDC builds: {sorted(known)}"
            )
    return fields


_NAME = re.compile(r"(encoder|decoder)(\.\d+|\.[a-z][a-z0-9_]*)+\.(weight|bias|running_mean|running_var)")
_NORMS = ("norm", "norm_out")


def dc_ae_path(name: str, ndim: int) -> tuple[str, ...] | None:
    """The parameter path of one `AutoencoderDC` tensor, or None for a batch
    norm's step count. Digits join the name before them; a norm's weight is
    its 1-D scale, and every other weight a 2-D or 4-D kernel. The multiscale
    branch's `proj_out` is `per_head`: a bare `proj_out` names a sharded
    projection elsewhere in Dew."""
    if name.endswith(".num_batches_tracked"):
        return None
    if _NAME.fullmatch(name) is None:
        raise ValueError(f"{name!r} is not an AutoencoderDC tensor")
    parts = name.split(".")
    leaf = parts.pop()
    norm = parts[-1] in _NORMS
    ranks = (2, 4) if leaf == "weight" and not norm else (1,)
    if ndim not in ranks:
        raise ValueError(f"{name!r} has {ndim} axes, not {' or '.join(map(str, ranks))}")
    path: list[str] = []
    for part in parts:
        if part.isdigit():
            path[-1] = f"{path[-1]}_{part}"
        else:
            path.append("per_head" if part == "proj_out" else part)
    if leaf == "weight":
        leaf = "scale" if norm else "kernel"
    return (*path, leaf)


class DCAutoencoder(ModuleAutoEncoder[DCAE]):
    """Native DC-AE weights and the pipeline's latent scaling,
    `z * scaling_factor` on the way out and `z / scaling_factor` on the way in."""

    def __init__(self, *, model: DCAE, params: Variables, latent_scale: float, latent_shift: float = 0.0):
        super().__init__(model, params)
        self.latent_scale = latent_scale
        self.latent_shift = latent_shift

    @property
    def downscale_factor(self) -> int:
        return self.model.downscale_factor

    @property
    def latent_channels(self) -> int:
        return self.model.latent_channels


def load_dc_ae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
               subfolder: str = "", param_dtype: str = "float32", params: Variables | None = None
               ) -> tuple[DCAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published DC-AE, its parameters and their source layouts from
    `subfolder` of a directory or Hub repo holding its config.json and
    weights, such as `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` (the root)
    or a SANA pipeline (`vae`); only that component downloads. Every tensor
    the module declares must be published, in the shape it declares.

    Supplied `params` are bound unchanged: only the config is read, and no
    source layouts are returned."""
    directory, config = component_source(name_or_dir, revision, subfolder, weights=params is None)
    if config.get("_class_name") != "AutoencoderDC":
        raise ValueError(f"{directory} holds a {config.get('_class_name')}, not an AutoencoderDC")
    model = DCAE(**dc_ae_fields(config), dtype=compute)
    frame = jax.ShapeDtypeStruct((1, model.downscale_factor, model.downscale_factor, model.image_channels),
                                 jnp.float32)
    return bind_component(
        directory, "vae", config, model, dc_ae_path,
        lambda bound: DCAutoencoder(model=model, params=bound,
                                    latent_scale=config.get("scaling_factor", 1.0)),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, inputs=(frame,))
