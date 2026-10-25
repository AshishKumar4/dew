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
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike

from dew import _records
from dew.interop.weights import ParamTree, check_shapes, native_shape, placed
from dew.nn.dense import Dense
from dew.nn.inputs import sequence_blocks
from dew.nn.precision import accumulated, capped_logits, float32_dot_general, float32_products, products_dtype
from dew.nn.rope import sinusoid_frequencies, sinusoids
from dew.registry import from_record

from .activations import activation
from .attention import LayerNorm, RMSNorm
from .conv import Conv
from .sharding import logical_axes
from .vision import TowerBase, TowerGeometry
from .vision.gemma4 import clipped_linear

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


@struct.dataclass
class AudioEncoding:
    features: jax.Array
    mask: jax.Array
    """True for valid encoded frames, including after temporal subsampling."""


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

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(mel_features=self.input_feat_size)


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

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(mel_features=self.subsampling_conv_channels[0])


def _check_geometry(width: int, heads: int, layers: int, chunk: int, left: int,
                    right: int, kernel: int, cap: float) -> None:
    if width < 2 or width % 2 or heads < 1 or width % heads:
        raise ValueError("audio hidden_size must be even and divisible by its positive head count")
    if layers < 1 or chunk < 1 or left < 1 or right < 0 or kernel < 1 or cap <= 0:
        raise ValueError(
            "audio layers, chunk, left context, kernel and logit cap must be positive; "
            "right context is nonnegative"
        )


def _inputs(features, mask, width: int) -> tuple[jax.Array, jax.Array]:
    features, mask = jnp.asarray(features), jnp.asarray(mask)
    if features.ndim != 3 or features.shape[-1] != width or min(features.shape) < 1:
        raise ValueError(f"input_features must be nonempty [B, T, {width}], got {features.shape}")
    if not jnp.issubdtype(features.dtype, jnp.floating):
        raise ValueError("input_features must be floating point mel features")
    if mask.shape != features.shape[:2] or mask.dtype != jnp.bool_:
        raise ValueError("input_features_mask must be bool [B, T], True for valid frames")
    return features, mask


def _clip(x, bound: float):
    bound = min(bound, float(jnp.finfo(x.dtype).max))
    return jnp.clip(x, -bound, bound)


def _dense(config: Gemma3nAudio | Gemma4Audio, dtype: Dtype | None, precision: PrecisionLike):
    """The encoders' bias-free linear map, its kernel drawn at `initializer_range`."""
    return functools.partial(Dense, use_bias=False, dtype=dtype, precision=precision,
                             kernel_init=nn.initializers.normal(config.initializer_range),
                             dot_general=float32_dot_general)


@logical_axes({("linear",): ("embed", "mlp")})
class AudioLinear(nn.Module):
    """Gemma 4's clippable linear in the audio tower's layout: the `dense` map
    under `linear` and its bounds beside it, as the reference nests them
    (`dew.nn.vision.gemma4.clipped_linear`)."""

    features: int
    dense: Callable[..., Dense]
    use_clipped_linears: bool = False

    @nn.compact
    def __call__(self, x):
        return clipped_linear(self, self.dense(self.features, name="linear"), x, self.use_clipped_linears)


def _clippable(config: Gemma4Audio, dtype: Dtype | None, precision: PrecisionLike):
    """Gemma 4's linear map: each one an `AudioLinear`."""
    dense = _dense(config, dtype, precision)

    def linear(features, *, name=None, **fields):
        return AudioLinear(features, dense=functools.partial(dense, **fields),
                           use_clipped_linears=config.use_clipped_linears, name=name)

    return linear


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


def _every_nth_frame(z, stride: int):
    """Keep every `stride`-th frame, as a gather rather than a strided slice.

    A strided slice feeding a dot crashes XLA's GPU tiling pass
    (`triton_tiling_propagation.cc:698`, jaxlib 0.11.1). A gather is not
    fused into a GEMM operand, so it stops the propagation instead.
    """
    return z if stride == 1 else z[:, jnp.arange(-(-z.shape[1] // stride)) * stride]


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
        x = Conv(self.features, self.kernel, strides=self.stride, padding=padding,
                 use_bias=False, dtype=self.dtype, precision=self.precision,
                 kernel_init=nn.initializers.normal(self.initializer_range), name="conv")(x)
        if self.cumulative:
            x = CumulativeGroupNorm(self.norm_eps, name="norm")(x)
        else:
            x = LayerNorm(epsilon=self.norm_eps, use_bias=False, dtype=self.dtype,
                             name="norm")(x)
        return jax.nn.relu(x), _every_nth_frame(mask, self.stride[0])[:, :x.shape[1]]


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
        return _dense(cfg, self.dtype, self.precision)(cfg.hidden_size, name="input_proj_linear")(x), mask


def _block_context(x, chunk: int, left: int, right: int):
    """Gather each chunk's keys with `left` before it and `right` after it.

    Returns `[B, blocks, left + chunk + right, ...]`. The padding makes
    every block whole, so the trailing block reads zeros past the end.
    """
    x = jnp.pad(x, ((0, 0), (left, right + chunk - 1)) + ((0, 0),) * (x.ndim - 2))
    length = chunk + left + right
    blocks = (x.shape[1] - length) // chunk + 1
    indices = jnp.arange(blocks)[:, None] * chunk + jnp.arange(length)[None, :]
    return x[:, indices]


def _position_signal(width: int, positions, dtype):
    """Embed each relative position as `width` sinusoids, sines then cosines."""
    half = width // 2
    # `exp(-ln(10000) i / (half - 1))`, which is `10000 ** (-i / (half - 1))`, rounded once from float64.
    inv = jnp.asarray(sinusoid_frequencies(half, max(half - 1, 1), dtype=np.float32))
    return sinusoids(positions.astype(jnp.float32)[:, None] * inv[None, :]).astype(dtype)


def _relative_logits(query, key, positional, precision: PrecisionLike = None):
    """Add content and relative-position logits: `[B, H, blocks, chunk, context]`.

    The content term is the usual query-key product per block. The relative
    term is the query against `positional`, one row per relative offset,
    then shifted so row i reads the offsets its own position spans. The
    shift is the pad-and-reshape trick of the reference (Shaw et al. 2018):
    padding each row by one and reading the flat result back as
    `chunk * context` moves every row one step left of the one above it.
    """
    batch, blocks, chunk, heads, dim = query.shape
    context = key.shape[2]
    q = query.transpose(0, 3, 1, 2, 4)
    products = float32_products(products_dtype(q, key, positional), precision)
    matmul = functools.partial(accumulated, functools.partial(jnp.matmul, precision=products))
    content = matmul(q, key.transpose(0, 3, 1, 4, 2))
    positions = positional.reshape(-1, heads, dim).transpose(1, 2, 0)
    relative = matmul(q.reshape(batch, heads, blocks * chunk, dim), positions)
    relative = relative.reshape(batch, heads, blocks, chunk, -1)
    relative = jnp.pad(relative, ((0, 0),) * 4 + ((0, context + 1 - relative.shape[-1]),))
    relative = relative.reshape(batch, heads, blocks, chunk * (context + 1))
    return content + relative[..., :chunk * context].reshape(batch, heads, blocks, chunk, context)


def _chunked_attention(query, key, value, positional, valid, chunk: int, left: int, right: int, *,
                       cap: float, window: Callable[[jax.Array], jax.Array], fill: float,
                       precision: PrecisionLike = None):
    """Chunked local attention with relative positions: `[B, T, H * D]`.

    Each chunk of the `[B, T, H, D]` queries scores its block of context
    (`_block_context`) and the relative positions `positional` holds
    (`_relative_logits`), soft-capped at `cap`. A key counts where it is
    valid and `window` holds for its query-minus-key distance; the rest take
    `fill`. Here the families differ in those two alone: Gemma 3n's window
    includes both boundaries and fills with the logits' lowest value, Gemma
    4's excludes them and fills with `attention_invalid_logits_value`. Both
    references take the softmax in float32 and hand it on in the values'
    dtype (modeling_gemma3n.py:523, modeling_gemma4.py:348).
    """
    batch, length, heads, dim = query.shape
    query = sequence_blocks(query, chunk)
    key, value = (_block_context(z, chunk, left, right) for z in (key, value))
    logits = _relative_logits(query, key, positional, precision)
    logits = capped_logits(logits, cap)
    distance = left + jnp.arange(chunk)[:, None] - jnp.arange(chunk + left + right)[None, :]
    allowed = _block_context(valid, chunk, left, right)[:, None, :, None, :] & window(distance)
    logits = jnp.where(allowed, logits, fill)
    probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(value.dtype)
    out = accumulated(functools.partial(jnp.matmul, precision=float32_products(value.dtype, precision)),
                      probabilities, value.transpose(0, 3, 1, 2, 4))
    return out.transpose(0, 2, 3, 1, 4).reshape(batch, -1, heads * dim)[:, :length]


class Gemma3nRelativePosition(nn.Module):
    config: Gemma3nAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, queries):
        cfg = self.config
        positions = jnp.arange(cfg.conf_attention_context_left - 1,
                               -cfg.conf_attention_context_right - 1, -1)
        signal = _position_signal(cfg.hidden_size, positions, queries.dtype)
        return _dense(cfg, self.dtype, self.precision)(cfg.hidden_size, name="pos_proj")(signal)


class Gemma3nAttention(nn.Module):
    config: Gemma3nAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        cfg = self.config
        heads, dim = cfg.conf_num_attention_heads, cfg.hidden_size // cfg.conf_num_attention_heads
        dense = _dense(cfg, self.dtype, self.precision)
        projected = [dense(cfg.hidden_size, name=name)(x).reshape(*x.shape[:2], heads, dim)
                     for name in ("q_proj", "k_proj", "v_proj")]
        query, key, value = projected
        per_dim = self.param("per_dim_scale", nn.initializers.zeros, (dim,), jnp.float32)
        query = query * jnp.asarray(dim ** -0.5 / math.log(2), query.dtype)
        query = query * jax.nn.softplus(per_dim).astype(query.dtype)
        left, right = cfg.conf_attention_context_left - 1, cfg.conf_attention_context_right
        positional = Gemma3nRelativePosition(cfg, dtype=self.dtype, precision=self.precision,
                                             name="relative_position_embedding")(query)
        return _chunked_attention(
            query, key, value, positional, valid, cfg.conf_attention_chunk_size, left, right,
            cap=cfg.conf_attention_logit_cap,
            window=lambda distance: (distance >= -right) & (distance <= left),
            fill=float(jnp.finfo(query.dtype).min), precision=self.precision)


class Gemma4Attention(nn.Module):
    config: Gemma4Audio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        cfg = self.config
        heads, dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        linear = _clippable(cfg, self.dtype, self.precision)
        projected = [
            linear(cfg.hidden_size, name=name)(x).astype(jnp.float32).reshape(*x.shape[:2], heads, dim)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        query, key, value = projected
        per_dim = self.param("per_dim_scale", nn.initializers.zeros, (dim,), jnp.float32)
        query = query * (dim ** -0.5 / math.log(2)) * jax.nn.softplus(per_dim)
        key = key * (math.log(1 + math.e) / math.log(2))
        chunk, left, right = (cfg.attention_chunk_size, cfg.attention_context_left - 1,
                              cfg.attention_context_right)
        signal = _position_signal(cfg.hidden_size, jnp.arange((chunk + left + right) // 2, -1, -1), x.dtype)
        positional = _dense(cfg, self.dtype, self.precision)(cfg.hidden_size, name="relative_k_proj")(signal)
        out = _chunked_attention(
            query, key, value, positional.astype(jnp.float32), valid, chunk, left, right,
            cap=cfg.attention_logit_cap,
            window=lambda distance: (((distance >= 0) & (distance < left))
                                     | ((distance < 0) & (distance > -right))),
            fill=cfg.attention_invalid_logits_value, precision=self.precision)
        return linear(cfg.hidden_size, name="post")(out.astype(x.dtype))


class AudioFeedForward(nn.Module):
    width: int
    residual_weight: float
    clipping: float
    linear: Callable[..., nn.Module]
    activation: str = "silu"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        residual = x
        x = RMSNorm(dtype=self.dtype, epsilon=1e-6, name="pre_layer_norm")(_clip(x, self.clipping))
        from .mlp import GatedMLP

        mlp = GatedMLP(self.width * 4, self.width, activation=activation(self.activation),
                        names=("ffw_layer_1", "ffw_layer_2"), linear=self.linear, dtype=self.dtype,
                        precision=self.precision)
        nn.share_scope(self, mlp)
        x = mlp(x)
        x = RMSNorm(dtype=self.dtype, epsilon=1e-6, name="post_layer_norm")(_clip(x, self.clipping))
        return residual + x * self.residual_weight


class AudioLightConv(nn.Module):
    width: int
    kernel_size: int
    norm_eps: float
    clipping: float
    linear: Callable[..., nn.Module]
    activation: str = "silu"
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    initializer_range: float = 0.02

    @nn.compact
    def __call__(self, x):
        residual = x
        x = RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name="pre_layer_norm")(x)
        x = self.linear(self.width * 2, name="linear_start")(x)
        first, gate = jnp.split(x, 2, axis=-1)
        x = first * jax.nn.sigmoid(gate)
        x = Conv(
            self.width,
            (self.kernel_size,),
            padding=((self.kernel_size - 1, 0),),
            feature_group_count=self.width,
            use_bias=False,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(self.initializer_range),
            name="depthwise_conv1d",
        )(x)
        x = RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name="conv_norm")(_clip(x, self.clipping))
        return residual + self.linear(self.width, name="linear_end")(activation(self.activation)(x))


class Gemma3nConformerAttention(nn.Module):
    config: Gemma3nAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        residual, cfg = x, self.config
        x = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="pre_attn_norm")(_clip(x, cfg.gradient_clipping))
        x = Gemma3nAttention(cfg, dtype=self.dtype, precision=self.precision, name="attn")(x, valid)
        x = _dense(cfg, self.dtype, self.precision)(cfg.hidden_size, name="post")(x)
        return residual + RMSNorm(epsilon=1e-6, dtype=self.dtype, name="post_norm")(
            _clip(x, cfg.gradient_clipping)
        )


class Gemma3nConformer(nn.Module):
    config: Gemma3nAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        cfg = self.config
        dense = _dense(cfg, self.dtype, self.precision)
        feed_forward = functools.partial(AudioFeedForward, cfg.hidden_size, cfg.conf_residual_weight,
                                         cfg.gradient_clipping, dense, dtype=self.dtype,
                                         precision=self.precision)
        x = feed_forward(name="ffw_layer_start")(x)
        x = Gemma3nConformerAttention(cfg, dtype=self.dtype, precision=self.precision, name="attention")(
            x, valid
        )
        x = AudioLightConv(cfg.hidden_size, cfg.conf_conv_kernel_size, cfg.rms_norm_eps,
                           cfg.gradient_clipping, dense, dtype=self.dtype, precision=self.precision,
                           initializer_range=cfg.initializer_range, name="lconv1d")(x * valid[..., None])
        x = feed_forward(name="ffw_layer_end")(x)
        return RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(_clip(x, cfg.gradient_clipping))


class Gemma4Conformer(nn.Module):
    config: Gemma4Audio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, valid):
        cfg = self.config
        linear = _clippable(cfg, self.dtype, self.precision)
        feed_forward = functools.partial(AudioFeedForward, cfg.hidden_size, cfg.residual_weight,
                                         cfg.gradient_clipping, linear, cfg.hidden_act, dtype=self.dtype,
                                         precision=self.precision)
        x = feed_forward(name="feed_forward1")(x)
        residual = x
        x = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_pre_attn")(_clip(x, cfg.gradient_clipping))
        x = Gemma4Attention(cfg, dtype=self.dtype, precision=self.precision, name="self_attn")(x, valid)
        x = residual + RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_post_attn")(
            _clip(x, cfg.gradient_clipping)
        )
        x = AudioLightConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.rms_norm_eps, cfg.gradient_clipping,
                           linear, cfg.hidden_act, dtype=self.dtype, precision=self.precision,
                           initializer_range=cfg.initializer_range, name="lconv1d")(x)
        x = feed_forward(name="feed_forward2")(x)
        return RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm_out")(_clip(x, cfg.gradient_clipping))


class Gemma3nAudioEncoder(nn.Module):
    config: Gemma3nAudio = Gemma3nAudio()
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, input_features, input_features_mask) -> AudioEncoding:
        cfg = self.config
        _check_geometry(
            cfg.hidden_size,
            cfg.conf_num_attention_heads,
            cfg.conf_num_hidden_layers,
            cfg.conf_attention_chunk_size,
            cfg.conf_attention_context_left,
            cfg.conf_attention_context_right,
            cfg.conf_conv_kernel_size,
            cfg.conf_attention_logit_cap,
        )
        if cfg.conf_reduction_factor < 1:
            raise ValueError("conf_reduction_factor must be positive")
        x, mask = _inputs(input_features, input_features_mask, cfg.input_feat_size)
        x, mask = AudioSubsample(cfg, dtype=self.dtype, precision=self.precision,
                                 name="subsample_conv_projection")(x, mask)
        for index in range(cfg.conf_num_hidden_layers):
            x = Gemma3nConformer(cfg, dtype=self.dtype, precision=self.precision,
                                 name=f"conformer_{index}")(x, mask)
        x, mask = (_every_nth_frame(z, cfg.conf_reduction_factor) for z in (x, mask))
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
            x = Gemma4Conformer(cfg, dtype=self.dtype, precision=self.precision, name=f"layers_{index}")(
                x, mask
            )
        x = Dense(cfg.output_proj_dims, use_bias=True, dtype=self.dtype, precision=self.precision,
                  kernel_init=nn.initializers.normal(cfg.initializer_range), name="output_proj",
                  dot_general=float32_dot_general)(x)
        return AudioEncoding(x, mask)


_METADATA = _records.HF_METADATA | {"model_type"}


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


def encoded_frame_stride(config: Gemma3nAudio | Gemma4Audio) -> int:
    """Input frames per encoded frame: the subsampling strides and any reduction.

    The encoders keep every stride-th frame of the input mask, so a host can
    count a clip's encoded frames as mask[::stride] without running them.
    """
    if isinstance(config, Gemma3nAudio):
        stride = config.conf_reduction_factor
        for time_stride, _ in config.sscp_conv_stride_size:
            stride *= time_stride
        return stride
    return 4



@functools.cache
def _audio_template(config: Gemma3nAudio | Gemma4Audio):
    features = (
        config.input_feat_size if isinstance(config, Gemma3nAudio) else config.subsampling_conv_channels[0]
    )
    template = jax.eval_shape(config.build().init, jax.random.key(0),
                              jax.ShapeDtypeStruct((1, 16, features), jnp.float32),
                              jax.ShapeDtypeStruct((1, 16), jnp.bool_))
    return {tuple(str(k.key) for k in path): value.shape
            for path, value in jax.tree_util.tree_leaves_with_path(template)}


def audio_weight_path(name: str, config: Gemma3nAudio | Gemma4Audio) -> tuple[str, ...]:
    """One bare HF audio tensor name into its complete native collection path."""
    parts = name.split(".")
    if len(parts) > 1 and parts[0] in ("conformer", "layers") and parts[1].isdigit():
        parts = [f"{parts[0]}_{parts[1]}", *parts[2:]]
    collection = (
        "constants" if parts[-1] in ("input_min", "input_max", "output_min", "output_max") else "params"
    )
    if parts[-1] == "weight":
        norms = {"norm", "pre_attn_norm", "post_norm", "pre_layer_norm", "post_layer_norm",
                 "norm_pre_attn", "norm_post_attn", "norm_out", "conv_norm"}
        parts[-1] = "scale" if len(parts) > 1 and parts[-2] in norms else "kernel"
    path = (collection, *parts)
    if path not in _audio_template(config):
        raise ValueError(f"audio tensor {name!r} does not match the encoder")
    return path



def audio_weights(
    tensors: Mapping[str, np.ndarray],
    config: Gemma3nAudio | Gemma4Audio,
    *,
    param_dtype: str = "float32",
) -> tuple[ParamTree, tuple[WeightLayout, ...]]:
    """Translate reference weights into params and frozen clipping constants, with the layouts
    that write each tensor back from them (`placed`).

    Prefix stripping belongs to the shared wrapper loader. Parameters use the
    requested storage precision; clipping state stays FP32. Integer tensors
    retain their native dtype, and every shape is checked against the encoder.
    """
    paths = {name: audio_weight_path(name, config) for name in tensors}
    check_shapes(((path, native_shape(path, tensors[name].shape)) for name, path in paths.items()),
                 _audio_template(config))
    return placed(tensors, paths.__getitem__, param_dtype)
