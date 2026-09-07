"""Conditional convolutional UNet for SD1/2 and SDXL latent denoising."""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import FlaxFeedForward, scaled_dot_product_attention
from dew.nn.blocks import ResidualBlock
from dew.registry import models


@struct.dataclass
class DenoisingCondition:
    context: jax.Array
    pooled: jax.Array | None = None
    time_ids: jax.Array | None = None


@dataclass(frozen=True)
class UNetStage:
    features: int
    heads: int
    depth: int = 1
    cross_attention: bool = True
    cross_only: bool = False


def sinusoidal_time(time, features: int, *, shift: float = 0, cosine_first: bool = True):
    half = features // 2
    if features % 2 or half <= shift:
        raise ValueError("Time embedding width must be even and exceed twice the frequency shift")
    frequencies = jnp.exp(jnp.arange(half, dtype=jnp.float32) * (-math.log(10000.0) / (half - shift)))
    phase = jnp.asarray(time, jnp.float32).reshape(-1, 1) * frequencies[None]
    first, second = (jnp.cos(phase), jnp.sin(phase)) if cosine_first else (jnp.sin(phase), jnp.cos(phase))
    return jnp.concatenate([first, second], axis=-1)


class _TimeMLP(nn.Module):
    features: int
    dtype: Dtype
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features, dtype=self.dtype, precision=self.precision, name="in_proj")(x)
        return nn.Dense(self.features, dtype=self.dtype, precision=self.precision, name="out_proj")(nn.silu(x))


class _Attention(nn.Module):
    features: int
    heads: int
    dropout: float
    dtype: Dtype
    precision: PrecisionLike = None
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x, context=None, *, train=False):
        context = x if context is None else context
        depth = self.features // self.heads
        def project(value, name):
            return nn.DenseGeneral((self.heads, depth), use_bias=False, dtype=self.dtype,
                                   precision=self.precision, name=name)(value)
        attended = scaled_dot_product_attention(project(x, "q"), project(context, "k"), project(context, "v"),
            dtype=self.dtype, precision=self.precision, implementation=self.attention_impl)
        output = nn.DenseGeneral(self.features, axis=(-2, -1), dtype=self.dtype,
                                 precision=self.precision, name="output")(attended)
        return nn.Dropout(self.dropout)(output, deterministic=not train) if self.dropout else output


class _Transformer(nn.Module):
    stage: UNetStage
    dropout: float
    dtype: Dtype
    precision: PrecisionLike = None
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x, context, *, train=False):
        def norm(name):
            return nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name=name)
        def attention(name):
            return _Attention(self.stage.features, self.stage.heads, self.dropout, self.dtype,
                              self.precision, self.attention_impl, name=name)
        x = x + attention("self_attention")(norm("self_norm")(x), context if self.stage.cross_only else None, train=train)
        x = x + attention("cross_attention")(norm("cross_norm")(x), context, train=train)
        x = x + FlaxFeedForward(self.stage.features, dtype=self.dtype, precision=self.precision,
                               dropout=self.dropout, name="feed_forward")(norm("ff_norm")(x), train=train)
        return nn.Dropout(self.dropout)(x, deterministic=not train) if self.dropout else x


class _SpatialAttention(nn.Module):
    stage: UNetStage
    linear_projection: bool
    dropout: float
    dtype: Dtype
    precision: PrecisionLike = None
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x, context, *, train=False):
        residual = x
        x = nn.GroupNorm(32, epsilon=1e-5, name="norm")(x)
        batch, height, width, channels = x.shape
        if self.linear_projection:
            x = x.reshape(batch, height * width, channels)
            x = nn.Dense(channels, dtype=self.dtype, precision=self.precision, name="input")(x)
        else:
            x = nn.Conv(channels, (1, 1), padding="VALID", dtype=self.dtype,
                        precision=self.precision, name="input")(x).reshape(batch, height * width, channels)
        for index in range(self.stage.depth):
            x = _Transformer(self.stage, self.dropout, self.dtype, self.precision,
                             self.attention_impl, name=f"layer_{index}")(x, context, train=train)
        if self.linear_projection:
            x = nn.Dense(channels, dtype=self.dtype, precision=self.precision, name="output")(x)
            x = x.reshape(batch, height, width, channels)
        else:
            x = nn.Conv(channels, (1, 1), padding="VALID", dtype=self.dtype, precision=self.precision,
                        name="output")(x.reshape(batch, height, width, channels))
        x = residual + x
        return nn.Dropout(self.dropout)(x, deterministic=not train) if self.dropout else x


class _Level(nn.Module):
    stage: UNetStage
    blocks: int
    direction: str
    resize: bool
    linear_projection: bool
    dropout: float
    dtype: Dtype
    precision: PrecisionLike = None
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x, time, context, skips=(), *, train=False):
        outputs = []
        for index in range(self.blocks):
            if self.direction == "up":
                x = jnp.concatenate([x, skips[-(index + 1)]], axis=-1)
            x = ResidualBlock(self.stage.features, norm_groups=32, norm_epsilon=1e-5,
                              dropout=self.dropout, dtype=self.dtype, precision=self.precision,
                              name=f"residual_{index}")(x, nn.silu(time), train=train)
            if self.stage.cross_attention:
                x = _SpatialAttention(self.stage, self.linear_projection, self.dropout, self.dtype,
                                      self.precision, self.attention_impl, name=f"attention_{index}")(x, context, train=train)
            outputs.append(x)
        if self.resize:
            if self.direction == "up":
                x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), "nearest")
            stride = (2, 2) if self.direction == "down" else (1, 1)
            x = nn.Conv(self.stage.features, (3, 3), strides=stride, padding=((1, 1), (1, 1)),
                        dtype=self.dtype, precision=self.precision, name="resize")(x)
            outputs.append(x)
        return x, tuple(outputs)


@models("unet_2d_condition")
class UNet2DCondition(nn.Module):
    """NHWC noisy latents plus text, pooled/size and optional inpaint conditions.

    Stages specify spatial width, attention heads and transformer depth. The
    encoder saves each residual output and each downsample; the decoder consumes
    those skips in reverse order with one extra residual block per level.
    """
    stages: tuple[UNetStage, ...]
    in_channels: int = 4
    out_channels: int = 4
    blocks_per_level: int = 2
    linear_projection: bool = False
    additional_time_features: int = 0
    middle_attention: bool = True
    frequency_shift: float = 0
    cosine_first: bool = True
    dropout: float = 0.0
    dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x, time, *, conditioning: DenoisingCondition, mask=None, masked_image=None, train=False):
        if mask is not None:
            if masked_image is None:
                raise ValueError("An inpaint mask needs its masked-image latents")
            x = jnp.concatenate([x, mask, masked_image], axis=-1)
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"UNet input has {x.shape[-1]} channels; expected {self.in_channels}")
        first = self.stages[0].features
        time = sinusoidal_time(time, first, shift=self.frequency_shift, cosine_first=self.cosine_first)
        time = _TimeMLP(first * 4, self.dtype, self.precision, name="time")(time)
        if self.additional_time_features:
            if conditioning.pooled is None or conditioning.time_ids is None:
                raise ValueError("This UNet needs pooled text and size/aesthetic conditioning")
            ids = sinusoidal_time(conditioning.time_ids.reshape(-1), self.additional_time_features,
                                  shift=self.frequency_shift, cosine_first=self.cosine_first)
            extra = jnp.concatenate([conditioning.pooled, ids.reshape(x.shape[0], -1)], axis=-1)
            time = time + _TimeMLP(first * 4, self.dtype, self.precision, name="additional_time")(extra)
        x = nn.Conv(first, (3, 3), dtype=self.dtype, precision=self.precision, name="input")(x)
        skips = [x]
        for index, stage in enumerate(self.stages):
            x, outputs = _Level(stage, self.blocks_per_level, "down", index + 1 < len(self.stages),
                self.linear_projection, self.dropout, self.dtype, self.precision, self.attention_impl,
                name=f"down_{index}")(x, time, conditioning.context, train=train)
            skips.extend(outputs)
        if self.middle_attention:
            stage = self.stages[-1]
            x = ResidualBlock(stage.features, norm_groups=32, norm_epsilon=1e-5, dropout=self.dropout,
                              dtype=self.dtype, precision=self.precision, name="middle_in")(x, nn.silu(time), train=train)
            x = _SpatialAttention(stage, self.linear_projection, self.dropout, self.dtype, self.precision,
                                  self.attention_impl, name="middle_attention")(x, conditioning.context, train=train)
            x = ResidualBlock(stage.features, norm_groups=32, norm_epsilon=1e-5, dropout=self.dropout,
                              dtype=self.dtype, precision=self.precision, name="middle_out")(x, nn.silu(time), train=train)
        for index, stage in enumerate(reversed(self.stages)):
            count = self.blocks_per_level + 1
            inputs, skips = tuple(skips[-count:]), skips[:-count]
            x, _ = _Level(stage, count, "up", index + 1 < len(self.stages), self.linear_projection,
                          self.dropout, self.dtype, self.precision, self.attention_impl,
                          name=f"up_{index}")(x, time, conditioning.context, inputs, train=train)
        x = nn.silu(nn.GroupNorm(32, epsilon=1e-5, name="output_norm")(x))
        return nn.Conv(self.out_channels, (3, 3), dtype=self.dtype, precision=self.precision, name="output")(x)
