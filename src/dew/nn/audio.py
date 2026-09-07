"""Gemma 3n and Gemma 4 audio encoders, from Transformers 5.16.1.

Sources: transformers/models/gemma3n/modeling_gemma3n.py:155-916,1448-1526
and transformers/models/gemma4/modeling_gemma4.py:218-573,1912-1995
(Apache 2.0). Both use full-audio prefills, without an encoder cache.
The public mask is True for valid frames for both families. Gemma 3n's
reference uses the inverse mask inside its encoder.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike

from dew.registry import from_record, towers
from .attention import RMSNorm
from .sharding import logical_axes
from .vision import TowerBase


@struct.dataclass
class AudioEncoding:
    features: jax.Array
    mask: jax.Array
    """True for valid encoded frames, including after temporal subsampling."""


@towers("gemma3n_audio")
@dataclasses.dataclass(frozen=True)
class Gemma3nAudio(TowerBase):
    input_feat_size: int = 128
    hidden_size: int = 1536
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_logit_cap: float = 50.0
    conf_num_attention_heads: int = 8
    conf_num_hidden_layers: int = 12
    conf_conv_kernel_size: int = 5
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    sscp_conv_channel_size: tuple[int, int] = (128, 32)
    sscp_conv_group_norm_eps: float = 1e-3
    sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = ((3, 3), (3, 3))
    sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = ((2, 2), (2, 2))
    initializer_range: float = 0.02

    def build(self) -> nn.Module:
        return Gemma3nAudioEncoder(self)


@towers("gemma4_audio")
@dataclasses.dataclass(frozen=True)
class Gemma4Audio(TowerBase):
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    subsampling_conv_channels: tuple[int, int] = (128, 32)
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1e9
    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536
    initializer_range: float = 0.02

    def build(self) -> nn.Module:
        return Gemma4AudioEncoder(self)


def _check_geometry(width: int, heads: int, layers: int, chunk: int, left: int,
                    right: int, kernel: int, cap: float) -> None:
    if width < 2 or width % 2 or heads < 1 or width % heads:
        raise ValueError("audio hidden_size must be even and divisible by its positive head count")
    if layers < 1 or chunk < 1 or left < 1 or right < 0 or kernel < 1 or cap <= 0:
        raise ValueError("audio layers, chunk, left context, kernel and logit cap must be positive; right context is nonnegative")


def _inputs(features, mask, width: int) -> tuple[jax.Array, jax.Array]:
    features, mask = jnp.asarray(features), jnp.asarray(mask)
    if features.ndim != 3 or features.shape[-1] != width or min(features.shape) < 1:
        raise ValueError(f"input_features must be nonempty [B, T, {width}], got {features.shape}")
    if not jnp.issubdtype(features.dtype, jnp.floating):
        raise ValueError("input_features must be floating point mel features")
    if mask.shape != features.shape[:2] or mask.dtype != jnp.bool_:
        raise ValueError("input_features_mask must be bool [B, T], True for valid frames")
    return features, mask


def _activation(x, name: str = "silu"):
    # Torch evaluates these elementwise activations in its fp32 opmath dtype.
    dtype = x.dtype
    x = x.astype(jnp.promote_types(dtype, jnp.float32))
    if name == "silu":
        y = jax.nn.silu(x)
    elif name in ("gelu", "gelu_pytorch_tanh"):
        y = jax.nn.gelu(x, approximate=name == "gelu_pytorch_tanh")
    elif name == "relu":
        y = jax.nn.relu(x)
    else:
        raise ValueError(f"audio hidden_act {name!r} has no implementation")
    return y.astype(dtype)


def _clip(x, bound: float):
    bound = min(bound, float(jnp.finfo(x.dtype).max))
    return jnp.clip(x, -bound, bound)


@logical_axes({("linear",): ("embed", "mlp")})
class AudioLinear(nn.Module):
    features: int
    clipped: bool = False
    use_bias: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    initializer_range: float = 0.02

    @nn.compact
    def __call__(self, x):
        if self.clipped:
            lo = self.variable("constants", "input_min", lambda: jnp.array(-jnp.inf)).value
            hi = self.variable("constants", "input_max", lambda: jnp.array(jnp.inf)).value
            x = jnp.clip(x, lo, hi)
        x = nn.Dense(self.features, use_bias=self.use_bias, dtype=self.dtype,
                     precision=self.precision, kernel_init=nn.initializers.normal(self.initializer_range),
                     name="linear")(x)
        if self.clipped:
            lo = self.variable("constants", "output_min", lambda: jnp.array(-jnp.inf)).value
            hi = self.variable("constants", "output_max", lambda: jnp.array(jnp.inf)).value
            x = jnp.clip(x, lo, hi)
        return x


class CumulativeGroupNorm(nn.Module):
    epsilon: float = 1e-3

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        fp32 = x.astype(jnp.float32)
        axes = tuple(range(2, x.ndim))
        count = jnp.arange(1, x.shape[1] + 1, dtype=jnp.float32) * math.prod(x.shape[2:])
        count = count.reshape((1, x.shape[1]) + (1,) * (x.ndim - 2))
        mean = jnp.cumsum(jnp.sum(fp32, axis=axes, keepdims=True), axis=1) / count
        # The reference accumulates each frame's squared difference from its
        # own prefix mean. It is not the usual E[x^2] - E[x]^2 variance.
        variance = jnp.cumsum(jnp.sum(jnp.square(fp32 - mean), axis=axes, keepdims=True), axis=1) / count
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],), jnp.float32)
        return ((fp32 - mean) * jax.lax.rsqrt(variance + self.epsilon) * scale).astype(dtype)


class AudioSubsampleLayer(nn.Module):
    features: int
    kernel: tuple[int, int]
    stride: tuple[int, int]
    cumulative: bool
    norm_eps: float
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    initializer_range: float = 0.02

    @nn.compact
    def __call__(self, x, mask):
        if not self.cumulative:
            x = x * mask[:, :, None, None]
        padding = ((0, self.kernel[0] - 1), (1, 1)) if self.cumulative else ((1, 1), (1, 1))
        x = nn.Conv(self.features, self.kernel, strides=self.stride, padding=padding,
                    use_bias=False, dtype=self.dtype, precision=self.precision,
                    kernel_init=nn.initializers.normal(self.initializer_range), name="conv")(x)
        if self.cumulative:
            x = CumulativeGroupNorm(self.norm_eps, name="norm")(x)
        else:
            x = nn.LayerNorm(epsilon=self.norm_eps, use_bias=False, dtype=self.dtype,
                             name="norm")(x)
        return jax.nn.relu(x), mask[:, ::self.stride[0]][:, :x.shape[1]]


class AudioSubsample(nn.Module):
    config: Gemma3nAudio | Gemma4Audio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, mask):
        cfg = self.config
        old = isinstance(cfg, Gemma3nAudio)
        channels = cfg.sscp_conv_channel_size if old else cfg.subsampling_conv_channels
        kernels = cfg.sscp_conv_kernel_size if old else ((3, 3), (3, 3))
        strides = cfg.sscp_conv_stride_size if old else ((2, 2), (2, 2))
        eps = cfg.sscp_conv_group_norm_eps if old else cfg.rms_norm_eps
        x = x[..., None]
        for i in range(2):
            x, mask = AudioSubsampleLayer(
                channels[i], kernels[i], strides[i], old, eps, dtype=self.dtype,
                precision=self.precision, initializer_range=cfg.initializer_range,
                name=f"conv_{i}" if old else f"layer{i}")(x, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return nn.Dense(cfg.hidden_size, use_bias=False, dtype=self.dtype,
                        precision=self.precision, kernel_init=nn.initializers.normal(cfg.initializer_range),
                        name="input_proj_linear")(x), mask


def _block_context(x, chunk: int, left: int, right: int):
    x = jnp.pad(x, ((0, 0), (left, right + chunk - 1)) + ((0, 0),) * (x.ndim - 2))
    length = chunk + left + right
    blocks = (x.shape[1] - length) // chunk + 1
    indices = jnp.arange(blocks)[:, None] * chunk + jnp.arange(length)[None, :]
    return x[:, indices]


def _queries(x, chunk: int):
    blocks = -(-x.shape[1] // chunk)
    x = jnp.pad(x, ((0, 0), (0, blocks * chunk - x.shape[1])) + ((0, 0),) * (x.ndim - 2))
    return x.reshape(x.shape[0], blocks, chunk, *x.shape[2:])


def _position_signal(width: int, positions, dtype):
    half = width // 2
    inv = jnp.exp(jnp.arange(half, dtype=jnp.float32) * (-math.log(10000.0) / max(half - 1, 1)))
    angles = positions.astype(jnp.float32)[:, None] * inv[None, :]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1).astype(dtype)


def _relative_logits(query, key, positional, precision: PrecisionLike = None):
    batch, blocks, chunk, heads, dim = query.shape
    context = key.shape[2]
    q = query.transpose(0, 3, 1, 2, 4)
    content = jnp.matmul(q, key.transpose(0, 3, 1, 4, 2), precision=precision)
    positions = positional.reshape(-1, heads, dim).transpose(1, 2, 0)
    relative = jnp.matmul(q.reshape(batch, heads, blocks * chunk, dim), positions, precision=precision)
    relative = relative.reshape(batch, heads, blocks, chunk, -1)
    relative = jnp.pad(relative, ((0, 0),) * 4 + ((0, context + 1 - relative.shape[-1]),))
    relative = relative.reshape(batch, heads, blocks, chunk * (context + 1))
    return content + relative[..., :chunk * context].reshape(batch, heads, blocks, chunk, context)


class Gemma3nRelativePosition(nn.Module):
    config: Gemma3nAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, queries, keys):
        cfg = self.config
        positions = jnp.arange(cfg.conf_attention_context_left - 1,
                               -cfg.conf_attention_context_right - 1, -1)
        signal = _position_signal(cfg.hidden_size, positions, queries.dtype)
        projected = nn.Dense(cfg.hidden_size, use_bias=False, dtype=self.dtype,
                             precision=self.precision, kernel_init=nn.initializers.normal(cfg.initializer_range), name="pos_proj")(signal)
        return _relative_logits(queries, keys, projected, self.precision)


class Gemma3nAttention(nn.Module):
    config: Gemma3nAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        cfg = self.config
        heads, dim = cfg.conf_num_attention_heads, cfg.hidden_size // cfg.conf_num_attention_heads
        projected = [nn.Dense(cfg.hidden_size, use_bias=False, dtype=self.dtype,
                              precision=self.precision, kernel_init=nn.initializers.normal(cfg.initializer_range), name=name)(x).reshape(*x.shape[:2], heads, dim)
                     for name in ("q_proj", "k_proj", "v_proj")]
        query, key, value = projected
        per_dim = self.param("per_dim_scale", nn.initializers.zeros, (dim,), jnp.float32)
        query = query * jnp.asarray(dim ** -0.5 / math.log(2), query.dtype)
        query = query * jax.nn.softplus(per_dim).astype(query.dtype)
        chunk, left, right = (cfg.conf_attention_chunk_size, cfg.conf_attention_context_left - 1,
                              cfg.conf_attention_context_right)
        query = _queries(query, chunk)
        key, value = (_block_context(z, chunk, left, right) for z in (key, value))
        logits = Gemma3nRelativePosition(cfg, dtype=self.dtype, precision=self.precision,
                                         name="relative_position_embedding")(query, key)
        cap = jnp.asarray(cfg.conf_attention_logit_cap, logits.dtype)
        logits = jnp.tanh(logits / cap) * cap
        indices = jnp.arange(chunk + left + right)[None, :] - jnp.arange(chunk)[:, None]
        local = (indices >= 0) & (indices <= left + right)
        allowed = _block_context(valid, chunk, left, right)[:, None, :, None, :] & local[None, None, None]
        logits = jnp.where(allowed, logits, jnp.finfo(logits.dtype).min)
        probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(value.dtype)
        out = jnp.matmul(probabilities, value.transpose(0, 3, 1, 2, 4), precision=self.precision)
        return out.transpose(0, 2, 3, 1, 4).reshape(x.shape[0], -1, heads, dim)[:, :x.shape[1]]


class Gemma4Attention(nn.Module):
    config: Gemma4Audio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        cfg = self.config
        heads, dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        projected = [AudioLinear(cfg.hidden_size, cfg.use_clipped_linears,
                                 dtype=self.dtype, precision=self.precision, initializer_range=cfg.initializer_range, name=name)(x)
                     .astype(jnp.float32).reshape(*x.shape[:2], heads, dim)
                     for name in ("q_proj", "k_proj", "v_proj")]
        query, key, value = projected
        per_dim = self.param("per_dim_scale", nn.initializers.zeros, (dim,), jnp.float32)
        query = query * (dim ** -0.5 / math.log(2)) * jax.nn.softplus(per_dim)
        key = key * (math.log(1 + math.e) / math.log(2))
        chunk, left, right = cfg.attention_chunk_size, cfg.attention_context_left - 1, cfg.attention_context_right
        context = chunk + left + right
        position = _position_signal(cfg.hidden_size, jnp.arange(context // 2, -1, -1), x.dtype)
        position = nn.Dense(cfg.hidden_size, use_bias=False, dtype=self.dtype,
                            precision=self.precision, kernel_init=nn.initializers.normal(cfg.initializer_range), name="relative_k_proj")(position).astype(jnp.float32)
        query = _queries(query, chunk)
        key, value = (_block_context(z, chunk, left, right) for z in (key, value))
        logits = _relative_logits(query, key, position, self.precision)
        logits = jnp.tanh(logits / cfg.attention_logit_cap) * cfg.attention_logit_cap
        indices = jnp.arange(context)[None, :] - jnp.arange(chunk)[:, None]
        allowed = _block_context(valid, chunk, left, right)[:, None, :, None, :]
        # Gemma 4 excludes both window boundaries; Gemma 3n includes them.
        distance = left - indices
        local = ((distance >= 0) & (distance < left)) | ((distance < 0) & (-distance < right))
        allowed = allowed & local[None, None, None]
        query_real = _queries(jnp.ones(valid.shape, jnp.bool_), chunk)
        allowed = allowed & query_real[:, None, :, :, None]
        probabilities = jax.nn.softmax(jnp.where(allowed, logits, cfg.attention_invalid_logits_value), axis=-1)
        out = jnp.matmul(probabilities, value.transpose(0, 3, 1, 2, 4), precision=self.precision)
        out = out.transpose(0, 2, 3, 1, 4).reshape(x.shape[0], -1, cfg.hidden_size)[:, :x.shape[1]]
        return AudioLinear(cfg.hidden_size, cfg.use_clipped_linears, dtype=self.dtype,
                           precision=self.precision, initializer_range=cfg.initializer_range, name="post")(out.astype(x.dtype))


class AudioFeedForward(nn.Module):
    width: int
    residual_weight: float
    clipping: float
    clipped: bool = False
    activation: str = "silu"
    wrapped: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    initializer_range: float = 0.02

    @nn.compact
    def __call__(self, x):
        residual = x
        x = RMSNorm(dtype=self.dtype, epsilon=1e-6, name="pre_layer_norm")(_clip(x, self.clipping))
        for name, width in (("ffw_layer_1", self.width * 4), ("ffw_layer_2", self.width)):
            if self.wrapped:
                x = AudioLinear(width, self.clipped, dtype=self.dtype, precision=self.precision, initializer_range=self.initializer_range, name=name)(x)
            else:
                x = nn.Dense(width, use_bias=False, dtype=self.dtype, precision=self.precision, kernel_init=nn.initializers.normal(self.initializer_range), name=name)(x)
            if name == "ffw_layer_1":
                x = _activation(x, self.activation)
        x = RMSNorm(dtype=self.dtype, epsilon=1e-6, name="post_layer_norm")(_clip(x, self.clipping))
        return residual + x * self.residual_weight


class AudioLightConv(nn.Module):
    width: int
    kernel_size: int
    norm_eps: float
    clipping: float
    clipped: bool = False
    activation: str = "silu"
    wrapped: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    initializer_range: float = 0.02

    @nn.compact
    def __call__(self, x):
        residual = x
        x = RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name="pre_layer_norm")(x)
        def linear(features, name, value):
            if self.wrapped:
                return AudioLinear(features, self.clipped, dtype=self.dtype, precision=self.precision, initializer_range=self.initializer_range, name=name)(value)
            return nn.Dense(features, use_bias=False, dtype=self.dtype, precision=self.precision, kernel_init=nn.initializers.normal(self.initializer_range), name=name)(value)
        x = linear(self.width * 2, "linear_start", x)
        first, gate = jnp.split(x, 2, axis=-1)
        x = first * jax.nn.sigmoid(gate)
        x = nn.Conv(self.width, (self.kernel_size,), padding=((self.kernel_size - 1, 0),),
                    feature_group_count=self.width, use_bias=False, dtype=self.dtype,
                    precision=self.precision, kernel_init=nn.initializers.normal(self.initializer_range), name="depthwise_conv1d")(x)
        x = RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name="conv_norm")(_clip(x, self.clipping))
        return residual + linear(self.width, "linear_end", _activation(x, self.activation))


class Gemma3nConformerAttention(nn.Module):
    config: Gemma3nAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        residual, cfg = x, self.config
        x = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="pre_attn_norm")(_clip(x, cfg.gradient_clipping))
        x = Gemma3nAttention(cfg, dtype=self.dtype, precision=self.precision, name="attn")(x, valid)
        x = nn.Dense(cfg.hidden_size, use_bias=False, dtype=self.dtype, precision=self.precision,
                     kernel_init=nn.initializers.normal(cfg.initializer_range), name="post")(x.reshape(x.shape[0], x.shape[1], cfg.hidden_size))
        return residual + RMSNorm(epsilon=1e-6, dtype=self.dtype, name="post_norm")(_clip(x, cfg.gradient_clipping))


class Gemma3nConformer(nn.Module):
    config: Gemma3nAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        cfg = self.config
        x = AudioFeedForward(cfg.hidden_size, cfg.conf_residual_weight, cfg.gradient_clipping,
                             dtype=self.dtype, precision=self.precision, initializer_range=cfg.initializer_range, name="ffw_layer_start")(x)
        x = Gemma3nConformerAttention(cfg, dtype=self.dtype, precision=self.precision, name="attention")(x, valid)
        x = AudioLightConv(cfg.hidden_size, cfg.conf_conv_kernel_size, cfg.rms_norm_eps,
                           cfg.gradient_clipping, dtype=self.dtype, precision=self.precision,
                           initializer_range=cfg.initializer_range, name="lconv1d")(x * valid[..., None])
        x = AudioFeedForward(cfg.hidden_size, cfg.conf_residual_weight, cfg.gradient_clipping,
                             dtype=self.dtype, precision=self.precision, initializer_range=cfg.initializer_range, name="ffw_layer_end")(x)
        return RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(_clip(x, cfg.gradient_clipping))


class Gemma4Conformer(nn.Module):
    config: Gemma4Audio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        cfg = self.config
        x = AudioFeedForward(cfg.hidden_size, cfg.residual_weight, cfg.gradient_clipping,
                             cfg.use_clipped_linears, cfg.hidden_act, True, dtype=self.dtype,
                             precision=self.precision, initializer_range=cfg.initializer_range, name="feed_forward1")(x)
        residual = x
        x = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_pre_attn")(_clip(x, cfg.gradient_clipping))
        x = Gemma4Attention(cfg, dtype=self.dtype, precision=self.precision, name="self_attn")(x, valid)
        x = residual + RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_post_attn")(_clip(x, cfg.gradient_clipping))
        x = AudioLightConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.rms_norm_eps, cfg.gradient_clipping,
                           cfg.use_clipped_linears, cfg.hidden_act, True, dtype=self.dtype,
                           precision=self.precision, initializer_range=cfg.initializer_range, name="lconv1d")(x)
        x = AudioFeedForward(cfg.hidden_size, cfg.residual_weight, cfg.gradient_clipping,
                             cfg.use_clipped_linears, cfg.hidden_act, True, dtype=self.dtype,
                             precision=self.precision, initializer_range=cfg.initializer_range, name="feed_forward2")(x)
        return RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_out")(_clip(x, cfg.gradient_clipping))


class Gemma3nAudioEncoder(nn.Module):
    config: Gemma3nAudio = Gemma3nAudio()
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, input_features, input_features_mask) -> AudioEncoding:
        cfg = self.config
        _check_geometry(cfg.hidden_size, cfg.conf_num_attention_heads, cfg.conf_num_hidden_layers,
                        cfg.conf_attention_chunk_size, cfg.conf_attention_context_left,
                        cfg.conf_attention_context_right, cfg.conf_conv_kernel_size, cfg.conf_attention_logit_cap)
        if cfg.conf_reduction_factor < 1:
            raise ValueError("conf_reduction_factor must be positive")
        x, mask = _inputs(input_features, input_features_mask, cfg.input_feat_size)
        x, mask = AudioSubsample(cfg, dtype=self.dtype, precision=self.precision,
                                 name="subsample_conv_projection")(x, mask)
        for index in range(cfg.conf_num_hidden_layers):
            x = Gemma3nConformer(cfg, dtype=self.dtype, precision=self.precision,
                                 name=f"conformer_{index}")(x, mask)
        x, mask = x[:, ::cfg.conf_reduction_factor], mask[:, ::cfg.conf_reduction_factor]
        return AudioEncoding(jnp.where(mask[..., None], x, 0), mask)


class Gemma4AudioEncoder(nn.Module):
    config: Gemma4Audio = Gemma4Audio()
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, input_features, input_features_mask) -> AudioEncoding:
        cfg = self.config
        _check_geometry(cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers,
                        cfg.attention_chunk_size, cfg.attention_context_left, cfg.attention_context_right,
                        cfg.conv_kernel_size, cfg.attention_logit_cap)
        x, mask = _inputs(input_features, input_features_mask, cfg.subsampling_conv_channels[0])
        if cfg.subsampling_conv_channels[0] % 4:
            raise ValueError("Gemma4 subsampling_conv_channels[0] must be divisible by four")
        x, mask = AudioSubsample(cfg, dtype=self.dtype, precision=self.precision,
                                 name="subsample_conv_projection")(x, mask)
        for index in range(cfg.num_hidden_layers):
            x = Gemma4Conformer(cfg, dtype=self.dtype, precision=self.precision, name=f"layers_{index}")(x, mask)
        x = nn.Dense(cfg.output_proj_dims, use_bias=True, dtype=self.dtype, precision=self.precision,
                     kernel_init=nn.initializers.normal(cfg.initializer_range), name="output_proj")(x)
        return AudioEncoding(x, mask)


_METADATA = frozenset({"model_type", "architectures", "transformers_version", "torch_dtype", "dtype",
                       "output_hidden_states", "output_attentions", "return_dict", "is_encoder_decoder",
                       "id2label", "label2id", "problem_type", "chunk_size_feed_forward"})


def audio_config(record: Mapping[str, object]) -> Gemma3nAudio | Gemma4Audio:
    """Validate an HF audio record and retain every encoder computation field."""
    model_type = record.get("model_type")
    if model_type == "gemma3n_audio":
        cls = Gemma3nAudio
        other = {"vocab_size", "vocab_offset"}  # consumed by the shared multimodal embedder
    elif model_type == "gemma4_audio":
        cls = Gemma4Audio
        other = set()
    else:
        raise ValueError(f"audio model_type {model_type!r} is not supported")
    fields = {field.name for field in dataclasses.fields(cls)}
    unknown = set(record) - fields - _METADATA - other - {key for key in record if key.startswith("_")}
    if unknown:
        raise ValueError(f"audio config fields {sorted(unknown)} have no counterpart")
    return from_record(cls, {key: value for key, value in record.items() if key in fields})


@functools.cache
def _audio_template(config: Gemma3nAudio | Gemma4Audio):
    features = config.input_feat_size if isinstance(config, Gemma3nAudio) else config.subsampling_conv_channels[0]
    template = jax.eval_shape(config.build().init, jax.random.key(0),
                              jax.ShapeDtypeStruct((1, 16, features), jnp.float32),
                              jax.ShapeDtypeStruct((1, 16), jnp.bool_))
    return {tuple(str(k.key) for k in path): value
            for path, value in jax.tree_util.tree_leaves_with_path(template)}


def audio_weight_path(name: str, config: Gemma3nAudio | Gemma4Audio) -> tuple[str, ...]:
    """One bare HF audio tensor name into its complete native collection path."""
    parts = name.split(".")
    if len(parts) > 1 and parts[0] in ("conformer", "layers") and parts[1].isdigit():
        parts = [f"{parts[0]}_{parts[1]}", *parts[2:]]
    collection = "constants" if parts[-1] in ("input_min", "input_max", "output_min", "output_max") else "params"
    if parts[-1] == "weight":
        norms = {"norm", "pre_attn_norm", "post_norm", "pre_layer_norm", "post_layer_norm",
                 "norm_pre_attn", "norm_post_attn", "norm_out", "conv_norm"}
        parts[-1] = "scale" if len(parts) > 1 and parts[-2] in norms else "kernel"
    path = (collection, *parts)
    if path not in _audio_template(config):
        raise ValueError(f"audio tensor {name!r} does not match the encoder")
    return path



def audio_weights(tensors: Mapping[str, np.ndarray], config: Gemma3nAudio | Gemma4Audio) -> dict:
    """Translate reference weights into params and frozen clipping constants.

    Prefix stripping belongs to the shared wrapper loader. This accepts only
    the encoder's own tensor names and verifies them against its abstract tree.
    """
    expected = _audio_template(config)
    result = {}
    seen = set()
    for name, array in tensors.items():
        path = audio_weight_path(name, config)
        value = np.asarray(array, np.float32)
        if path[-1] == "kernel":
            value = np.ascontiguousarray(value.transpose(*range(2, value.ndim), 1, 0))
        if path not in expected or value.shape != expected[path].shape:
            raise ValueError(f"audio tensor {name!r} with shape {value.shape} does not match the encoder")
        seen.add(path)
        target = result
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target[path[-1]] = value
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"audio checkpoint is missing {sorted('/'.join(path) for path in missing)}")
    return result
