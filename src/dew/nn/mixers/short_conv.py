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
    (`held_conv1d`), as LFM3's code zeroes a padded slot's convolved input. Under `padding_holds_bias`
    a biased layer's leading padding (a left-padded prompt's, in a row's first call, `conv_started`
    `[B]` marking a row that has consumed a real token) holds what LFM2's code does: the projections'
    bias product, its code zeroing a padded slot's input before them alone. Parameter names are the
    checkpoint's (`conv/weight` the `[D, 1, K]` taps)."""

    reads: ClassVar[frozenset[str]] = frozenset()
    """None of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`): a convolution reads the K-1
    tokens before each of its document, with no mask or bias to narrow it."""

    emb_features: int
    kernel: int = 3
    use_bias: bool = False
    padding_holds_bias: bool = False
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

    @nn.compact
    def __call__(self, x: jax.Array, call: MixerCall = PLAIN_CALL) -> jax.Array:
        batch, length, width = x.shape
        tokens = coordinates_of(call.metadata, batch, length)
        valid = tokens.valid
        started = None
        biased_padding = self.use_bias and self.padding_holds_bias
        if call.decode and biased_padding:
            # Whether each row has consumed a real token: only a row's first call can open with a
            # prompt's padding.
            started = self.variable('cache', 'conv_started', jnp.zeros, (batch,), bool)
        if biased_padding and valid is not None and tokens.segments is None:
            # A prompt's leading padding, the slots before the first real token of a row that has
            # consumed none (not a paused slot, a row's trailing slots or a continuation's): the source
            # zeroes its input before the projections (LFM2's `apply_mask_to_padding_states`), so a
            # biased layer's slots there hold the projections' bias product, which the real tokens
            # after them convolve (ledger 3165).
            held_valid = jnp.asarray(valid, bool)
            fresh = jnp.ones((batch, 1), bool) if started is None else ~started.value[:, None]
            leading = (~held_valid & (jnp.cumsum(held_valid, axis=1) == 0)
                       & held_valid.any(axis=1, keepdims=True) & fresh)
            x = jnp.where(leading[..., None], jnp.zeros_like(x), x)
            valid = held_valid | leading
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
        if started is not None:
            started.value = started.value | (jnp.ones((batch,), bool) if valid is None
                                             else jnp.asarray(valid, bool).any(axis=1))
        convolved, history = held_conv1d(mixed, taps, valid, None if held is None else held.value,
                                         bias=bias, segments=tokens.segments,
                                         right_padded=tokens.right_padded, dtype=mixed.dtype,
                                         activation=False)
        if held is not None:
            held.value = history.astype(held.value.dtype)
        return self.out_proj(gate_out * checkpoint_name(convolved, 'conv'))


@dataclasses.dataclass(frozen=True)
class ShortConvMixer(MixerBase):
    """Liquid's gated short convolution layer kind (`GatedShortConv`), by the config's field names:
    `conv_L_cache` taps and `conv_bias`, and whether a biased layer's leading padding holds its bias
    product (`GatedShortConv.padding_holds_bias`, LFM2's). The context's attention geometry is not read:
    the layer has no keys to cache, no rope and no window."""

    residuals = ('conv',)

    conv_L_cache: int = 3
    conv_bias: bool = False
    padding_holds_bias: bool = False

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("short_conv requires causal=True; its convolution reads only earlier tokens")
        return functools.partial(GatedShortConv, emb_features=ctx.emb_features, kernel=self.conv_L_cache,
                                 use_bias=self.conv_bias, padding_holds_bias=self.padding_holds_bias,
                                 dtype=ctx.dtype, precision=ctx.precision)
