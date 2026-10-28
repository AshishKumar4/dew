"""The gated short convolution mixer kind of Liquid's LFM2 and LFM3, from the reference's own field names."""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.nn.dense import Dense
from dew.nn.flops import conv_flops, projection_flops
from dew.nn.inputs import PLAIN_CALL, MixerCall, coordinates_of
from dew.nn.linear import DepthwiseConv1d, held_conv1d
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.precision import float32_dot_general


class GatedShortConv(nn.Module):
    """Liquid's gated short convolution: `in_proj` to B, C and x, the depthwise causal convolution
    of B * x over `kernel` taps with no activation, gated by C, and `out_proj` back
    (modeling_lfm3.py, Lfm3ShortConv; modeling_lfm2.py, Lfm2ShortConv). `use_bias` is the
    reference's `conv_bias`, which biases the convolution and both projections.

    The decode cache holds the last K-1 convolved inputs, `conv_state` `[B, K-1, D]` in the
    projections' dtype; padded slots read and advance none, and packed documents convolve apart
    (`held_conv1d`). Parameter names are the checkpoint's (`conv/weight` the `[D, 1, K]` taps)."""

    reads: ClassVar[frozenset[str]] = frozenset()
    """None of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`): a convolution reads the K-1
    tokens before each of its document, with no mask or bias to narrow it."""

    emb_features: int
    kernel: int = 3
    use_bias: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Its two projections and its depthwise convolution."""
        width = self.emb_features
        return (projection_flops(lengths, width, 3 * width) + projection_flops(lengths, width, width)
                + conv_flops(lengths, width, width, self.kernel, groups=width))

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """All of it is learned maps."""
        return self.model_flops(lengths)

    def setup(self):
        dense = functools.partial(Dense, use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
                                  dot_general=float32_dot_general)
        self.in_proj = dense(3 * self.emb_features, name='in_proj')
        self.out_proj = dense(self.emb_features, name='out_proj')
        self.conv = DepthwiseConv1d(features=self.emb_features, kernel=self.kernel, use_bias=self.use_bias,
                                    name='conv')

    def __call__(self, x: jax.Array, call: MixerCall = PLAIN_CALL) -> jax.Array:
        batch, length, width = x.shape
        tokens = coordinates_of(call.metadata, batch, length)
        gate_in, gate_out, value = jnp.split(self.in_proj(x), 3, axis=-1)
        mixed = gate_in * value
        taps, bias = self.conv()
        held = None
        if call.decode:
            allocated = self.has_variable('cache', 'conv_state')
            held = self.variable('cache', 'conv_state', jnp.zeros, (batch, self.kernel - 1, width),
                                 mixed.dtype)
            if not allocated:
                # The allocation-only call (`open_kv_cache`'s dummy token) consumes nothing.
                return self.out_proj(jnp.zeros_like(mixed))
        convolved, history = held_conv1d(mixed, taps, tokens.valid, None if held is None else held.value,
                                         bias=bias, segments=tokens.segments,
                                         right_padded=tokens.right_padded, dtype=mixed.dtype,
                                         activation=False)
        if held is not None:
            held.value = history.astype(held.value.dtype)
        return self.out_proj(gate_out * checkpoint_name(convolved, 'conv'))


@dataclasses.dataclass(frozen=True)
class ShortConvMixer(MixerBase):
    """Liquid's gated short convolution layer kind (`GatedShortConv`), by the config's field names:
    `conv_L_cache` taps and `conv_bias`. The context's attention geometry is not read: the layer has
    no keys to cache, no rope and no window."""

    residuals = ('conv',)

    conv_L_cache: int = 3
    conv_bias: bool = False

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("short_conv requires causal=True; its convolution reads only earlier tokens")
        return functools.partial(GatedShortConv, emb_features=ctx.emb_features, kernel=self.conv_L_cache,
                                 use_bias=self.conv_bias, dtype=ctx.dtype, precision=ctx.precision)
