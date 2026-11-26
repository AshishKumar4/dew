"""A causal depthwise short convolution added back to its own input at one site of a layer (`ShortConvSite`):
`(conv(x) + x) / sqrt(k + 1)` over the last `k` tokens, its taps static or, with a dynamic input,
static plus per-token taps a projection of that input gives a head (`/ sqrt(2k + 1)`), each site its own
decode history (Inkling 2's attention's keys, values and output and its MoE's output,
modeling_inkling2_loop.py:266-442; theme M71). The static convolution is the linear layers' own
(`dew.nn.linear.held_conv1d`, the short-conv kernels where they run); the dynamic one is the portable
arithmetic, a window of the last `k` inputs times each token's taps, masked as the static one is.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.dense import Dense
from dew.nn.inputs import PLAIN_CALL, MixerCall, coordinates_of
from dew.nn.kv_cache import Opened, cache_variables, compute_dtype, kept
from dew.nn.linear import DepthwiseConv1d, held_conv1d
from dew.nn.precision import at_least_fp32, float32_dot_general


@dataclasses.dataclass(frozen=True)
class ShortConvs:
    """An attention layer's causal short convolutions (`ShortConvSite`) of its projected keys and values,
    before their head clip and norm, each `kernel` taps, with per-token dynamic taps a key head where
    `dynamic` (Inkling 2's global layers' `k_sconv`, `v_sconv`)."""

    kernel: int = 4
    dynamic: bool = False

    def __post_init__(self):
        if self.kernel < 2:
            raise ValueError(f"a short convolution reads the token before its own too, got {self.kernel}")


class ShortConvSite(nn.Module):
    """`(conv(x) + x) / sqrt(kernel + 1)` of `x` `[B, T, features]`, its causal depthwise taps `conv` (no
    bias, `conv/weight` `[features, 1, kernel]`), in at least fp32 and returned in `x`'s dtype. With
    `dynamic_heads`, each token's taps add a projection of the call's dynamic input (`dynamic`, bias-free,
    `[heads * kernel]` a token) shared by each head's `features // heads` channels, and the sum divides by
    `sqrt(2 * kernel + 1)` instead. A padding token's input convolves as zero and packed documents apart;
    a cached call convolves behind the site's history, `conv_state` `[B, kernel - 1, features]`. Every tap
    is in convolution order, the last the token itself (`causal_conv1d`'s), the dynamic projection's too;
    a source holding them by lag, the first the token itself (Inkling 2's dynamic sites), reverses them on
    import."""

    features: int
    kernel: int = 4
    dynamic_heads: int | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def cache_spec(self, rows: int) -> dict[str, Opened]:
        """The site's decode history for `rows` rows: its last `kernel - 1` inputs."""
        return {"conv_state": Opened((rows, self.kernel - 1, self.features), compute_dtype(self, self.dtype))}

    @nn.compact
    def __call__(self, x: jax.Array, call: MixerCall = PLAIN_CALL, dynamic_input: jax.Array | None = None
                 ) -> jax.Array:
        batch, length, _ = x.shape
        if (dynamic_input is None) != (self.dynamic_heads is None):
            raise ValueError("a site with dynamic taps reads their input, and one without reads none")
        if self.dynamic_heads is not None and self.features % self.dynamic_heads:
            raise ValueError(f"a site's dynamic taps are shared by each of its {self.dynamic_heads} heads' "
                             f"channels, which divide its {self.features}")
        conv = DepthwiseConv1d(features=self.features, kernel=self.kernel, use_bias=False, name='conv')
        tokens = coordinates_of(call.metadata, batch, length)
        if tokens.forks is not None:
            raise ValueError("a short convolution reads each document's own inputs, and a forked document's "
                             "first tokens would read none of its parent's: run each fork as its own row")
        if call.metadata is not None and call.metadata.admitted is not None:
            raise ValueError("a short convolution's history is its row's, and a server's mixed step packs "
                             "several requests' tokens in one: serve this layer by the dense cache")
        held = cache_variables(self, self.cache_spec(batch))["conv_state"] if call.decode else None
        taps, _ = conv()
        wide = at_least_fp32(x.dtype)
        if dynamic_input is None or self.dynamic_heads is None:
            convolved, history = held_conv1d(x, taps, tokens.valid, None if held is None else held.value,
                                             segments=tokens.segments, right_padded=tokens.right_padded,
                                             dtype=wide, activation=False)
            scale = math.sqrt(self.kernel + 1)
        else:
            dynamic = Dense(self.dynamic_heads * self.kernel, use_bias=False, dtype=self.dtype,
                            precision=self.precision, dot_general=float32_dot_general,
                            kernel_axes=('embed', 'linear'), name='dynamic')
            convolved, history = self._dynamic(x, taps, dynamic(dynamic_input), tokens,
                                               None if held is None else held.value)
            scale = math.sqrt(2 * self.kernel + 1)
        if held is not None:
            kept(held, jnp.asarray(history, held.value.dtype), "conv_state")
        return ((jnp.asarray(convolved, wide) + x.astype(wide)) / scale).astype(x.dtype)

    @nn.nowrap
    def _dynamic(self, x, taps, projected, tokens, history):
        """`x` convolved by `taps` `[features, kernel]` plus each token's own (`projected`, the `dynamic`
        projection of the dynamic input, a head's shared by its channels), in at least fp32: the window of
        each token's last `kernel` inputs (behind `history` where held, zeros else), a padding token's
        input zero and another document's tokens read as zero; and the history the next call holds."""
        batch, length, features = x.shape
        heads, k = self.dynamic_heads, self.kernel
        assert heads is not None, "dynamic taps are a dynamic site's"
        if tokens.valid is not None and not tokens.right_padded:
            raise ValueError("a short convolution with dynamic taps convolves rows whose real tokens come "
                             "first (right-padded): its windows read no gap past a padding slot")
        wide = at_least_fp32(x.dtype)
        inputs = x.astype(wide)
        if tokens.valid is not None:
            inputs = jnp.where(jnp.asarray(tokens.valid, bool)[..., None], inputs, 0)
        before = (jnp.zeros((batch, k - 1, features), wide) if history is None else history.astype(wide))
        padded = jnp.concatenate([before, inputs], axis=1)
        # Window position j of token t reads input t - (k - 1) + j: oldest first, the token itself last.
        windows = jnp.stack([padded[:, j:j + length] for j in range(k)], axis=-1)
        documents = None
        if tokens.segments is not None:
            # The held inputs are the first document's, which the call continues (`held_conv1d`'s rule).
            segments = jnp.asarray(tokens.segments, jnp.int32)
            documents = jnp.concatenate([jnp.broadcast_to(segments[:, :1], (batch, k - 1)), segments], axis=1)
            same = jnp.stack([documents[:, j:j + length] for j in range(k)], axis=-1) == documents[
                :, k - 1:, None]
            windows = jnp.where(same[:, :, None, :], windows, 0)
        own = projected.astype(wide).reshape(batch, length, heads, 1, k)
        weights = taps.astype(wide).reshape(heads, features // heads, k)[None, None] + own
        convolved = jnp.sum(windows.reshape(batch, length, heads, features // heads, k) * weights, axis=-1)
        # The row's last real inputs, as the static convolution keeps them (`held_conv1d`): a right-padded
        # row's padding enters no history, and of packed documents only the last real token's does.
        ends = (jnp.full((batch,), length, jnp.int32) if tokens.valid is None
                else jnp.sum(jnp.asarray(tokens.valid, jnp.int32), axis=1))
        rows = ends[:, None] + jnp.arange(k - 1)[None]
        history = jnp.take_along_axis(padded, rows[..., None], axis=1)
        if documents is not None:
            last = jnp.take_along_axis(documents, ends[:, None] + k - 2, axis=1)
            history = jnp.where((jnp.take_along_axis(documents, rows, axis=1) == last)[..., None], history, 0)
        return convolved.reshape(batch, length, features), history.astype(x.dtype)
