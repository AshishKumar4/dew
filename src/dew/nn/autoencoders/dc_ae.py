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

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype

from dew.batch import Variables
from dew.nn.dense import Dense

from ..activations import gated_product, silu
from ..conv import Conv
from ..norms import rms_normalized
from ..precision import at_least_fp32, float32_dot_general, float32_einsum
from ..scan_orders import pixel_shuffle, pixel_unshuffle
from .api import LatentNormalization, ModuleAutoEncoder

_EPS = 1e-5
_ATTENTION_EPS = 1e-15
ACTIVATIONS = {"relu": nn.relu, "relu6": nn.relu6, "silu": silu}


class _RMSNorm(nn.Module):
    """diffusers' `RMSNorm` with a bias, over the channel axis: the bias joins
    the scaled activation before its one cast."""

    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.features,))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        wide = at_least_fp32(x.dtype)
        normed = rms_normalized(x, scale, _EPS, wide, scale_offset=False, scale_after_cast=False,
                                fp32_statistics=True)
        return (normed + bias.astype(wide)).astype(self.dtype)


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
        y = ACTIVATIONS[self.activation](_conv(self.features, 3, self.dtype, "conv1")(x))
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
        qkv = jnp.concatenate([Dense(inner, use_bias=False, dtype=self.dtype, name=name,
                                     dot_general=float32_dot_general)(x)
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
            scores = float32_einsum("bmgd,bngd->bgmn", key, query).astype(jnp.float32)
            scores = scores / (jnp.sum(scores, axis=2, keepdims=True) + _ATTENTION_EPS)
            attended = float32_einsum("bmgd,bgmn->bngd", value, scores.astype(value.dtype))
        else:
            value = jnp.concatenate([value, jnp.ones_like(value[..., :1])], axis=-1)
            scores = float32_einsum("bngd,bnge->bgde", value, key)
            attended = float32_einsum("bgde,bnge->bngd", scores, query)
            attended = (attended[..., :-1] / (attended[..., -1:] + _ATTENTION_EPS)).astype(x.dtype)
        attended = attended.reshape(batch, height, width, -1)
        out = Dense(self.features, use_bias=False, dtype=self.dtype, name="to_out",
                    dot_general=float32_dot_general)(attended)
        return _norm(self.norm, self.features, self.dtype, "norm_out")(out) + x


class _GLUMBConv(nn.Module):
    """An inverted bottleneck four times wide, gated by its own second half."""

    features: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        hidden = 4 * self.features
        y = silu(_conv(2 * hidden, 1, self.dtype, "conv_inverted")(x))
        y = _conv(2 * hidden, 3, self.dtype, "conv_depth", groups=2 * hidden)(y)
        y, gate = jnp.split(y, 2, axis=-1)
        y = _conv(self.features, 1, self.dtype, "conv_point", bias=False)(gated_product('swiglu')(gate, y))
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


def _resamples(layers: tuple[int, ...], level: int) -> bool:
    """Whether the encoder downsamples after `level` and the decoder upsamples into it: a level with
    blocks below the last. A first level with none resamples at the image instead (`conv_in` and
    `conv_out`), so the levels' resamplings, plus that one, are the autoencoder's factor of 2."""
    return level < len(layers) - 1 and layers[level] > 0


class _Encoder(nn.Module):
    latent_channels: int
    head_dim: int
    blocks: tuple[str, ...]
    channels: tuple[int, ...]
    layers: tuple[int, ...]
    scales: tuple[tuple[int, ...], ...]
    unshuffle: bool
    shortcut: bool = True
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
            if _resamples(layers, level):
                x = _Down(channels[level + 1], self.unshuffle, shortcut=True, dtype=self.dtype,
                          name=f"down_blocks_{level}_{count}")(x)
        latents = _conv(self.latent_channels, 3, self.dtype, "conv_out")(x)
        return latents + _group_mean(x, self.latent_channels) if self.shortcut else latents


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
    shortcut: bool = True
    out_activation: str = "relu"
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, latents):
        channels, layers = self.channels, self.layers
        x = _conv(channels[-1], 3, self.dtype, "conv_in")(latents)
        if self.shortcut:
            x = x + jnp.repeat(latents, channels[-1] // self.latent_channels, axis=-1)
        for level in reversed(range(len(channels))):
            count = layers[level]
            upsampled = _resamples(layers, level)
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
        x = ACTIVATIONS[self.out_activation](_RMSNorm(features, self.dtype, name="norm_out")(x))
        if layers[0] > 0:
            return _conv(self.image_channels, 3, self.dtype, "conv_out")(x)
        return _Up(features, self.image_channels, self.interpolate, shortcut=False, dtype=self.dtype,
                   name="conv_out")(x)


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
    encoder_shortcut: bool = True
    """Whether the encoder's last convolution adds its input's channel groups averaged down
    (`AutoencoderDC`'s `encoder_out_shortcut`)."""
    decoder_shortcut: bool = True
    """Whether the decoder's first convolution adds its latents' channels repeated up
    (`decoder_in_shortcut`)."""
    decoder_out_activation: str = "relu"
    """The activation before the decoder's last convolution (`decoder_conv_act_fn`), one of `ACTIVATIONS`."""
    dtype: Dtype = jnp.float32

    @property
    def downscale_factor(self) -> int:
        """2 to the encoder's resamplings: one a level with blocks below the last, and one at the
        image where the first level has none (`_resamples`)."""
        layers = self.encoder_layers
        return 2 ** ((layers[0] == 0) + sum(_resamples(layers, level) for level in range(len(layers))))

    def setup(self):
        self.encoder = _Encoder(
            self.latent_channels,
            self.head_dim,
            self.encoder_blocks,
            self.encoder_channels,
            self.encoder_layers,
            self.encoder_scales,
            self.unshuffle,
            self.encoder_shortcut,
            self.dtype,
        )
        self.decoder = _Decoder(self.image_channels, self.latent_channels, self.head_dim, self.decoder_blocks,
                                self.decoder_channels, self.decoder_layers, self.decoder_scales,
                                self.decoder_norms, self.decoder_activations, self.interpolate,
                                self.decoder_shortcut, self.decoder_out_activation, self.dtype)

    def encode(self, image, key=None, noise=None):
        """The latent of `image`. The encoder is deterministic: `key` is unused, and a posterior's
        `noise` raises TypeError, since it has none to draw."""
        if noise is not None:
            raise TypeError(f"{type(self).__name__} encodes deterministically and draws no posterior noise")
        return self.encoder(image)

    def decode(self, latents):
        return self.decoder(latents)

    def __call__(self, image):
        return self.decode(self.encode(image))


class DCAutoencoder(ModuleAutoEncoder[DCAE]):
    """Native DC-AE weights and the pipeline's latent scaling,
    `z * scaling_factor` on the way out and `z / scaling_factor` on the way in."""

    def __init__(self, *, model: DCAE, params: Variables, latent_scale: float, latent_shift: float = 0.0):
        super().__init__(model, params)
        self.normalization = LatentNormalization({'latent_scale': latent_scale, 'latent_shift': latent_shift},
                                                 latent_shift, latent_scale)

    @property
    def stochastic(self) -> bool:
        return False

    def record_arguments(self) -> dict:
        return dict(self.normalization.record)


