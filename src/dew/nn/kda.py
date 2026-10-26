"""Kimi Delta Attention: the linear-attention layer of GLM-5.3-Flash.

KDA is the gated delta rule of `dew.nn.linear` with a different recurrence
decay and parameterisation (`Glm5NextTextLinearAttention`,
modeling_glm5_next.py:584-733):

- The decay is a vector per head over the key dimensions, `S <- S *
  exp(g_t)[:, None]` (modeling_glm5_next.py:468-471, 529-532).
- `g = f_b_proj(f_a_proj(x)) + dt_bias` over `[H, Dk]`, and with
  `linear_lower_bound` the decay is `lower_bound * sigmoid(exp(A_log) * g)`
  instead of `-exp(A_log) * softplus(g)` (:319-335).
- `beta = sigmoid(b_proj(x))` per head (:697).
- q, k and v have their own projections and depthwise convs, stored apart
  as the release stores them and concatenated at call time in the
  reference's order (:607-615, 642-649).
- The output gate `g_b_proj(g_a_proj(x))` drives a sigmoid-gated weighted
  RMSNorm, `o_norm` (:339-358, 729-730), then `o_proj`.
- q and k are l2-normalised inside the rule as `x / sqrt(sum x^2 + eps)`
  (:416-424).

`chunk_kimi_delta_attention` (:482-578) is the gated delta rule's chunked form
with a decay per key dimension, `dew.nn.kernels.recurrence_chunk.chunk_gated_delta_rule`: on a
GPU fla's chunked KDA ported to Pallas (`dew.nn.kernels.fla_kda`), elsewhere
the portable math in fp32; the recurrent form is
`dew.nn.kernels.recurrence_chunk.recurrent_delta_rule` (:428-478). tests/test_kda.py holds the
math to a float64 oracle of the reference, and tests/test_fla_kda.py the port.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.dense import Dense

from .flops import conv_flops, delta_flops, projection_flops
from .inputs import PLAIN_CALL, MixerCall, coordinates_of
from .kernels.recurrence_chunk import CHUNK_SIZE, chunk_gated_delta_rule, kernel_routed, recurrent_delta_rule
from .linear import DepthwiseConv1d, RMSNormGated, held_scan
from .mixer_base import DELTA_RULE_RESIDUALS, MixerBase, MixerContext
from .precision import at_least_fp32, float32_dot_general
from .sharding import logical_axes


# q/k/v_proj and o_proj carry the attention mixer's declarations under the
# same names; the gates' low-rank pairs and the beta projection are KDA's own.
@logical_axes({
    ("b_proj",): ("embed", "kv"),
    ("f_a_proj",): ("embed", None),
    ("f_b_proj",): (None, "heads"),
    ("g_a_proj",): ("embed", None),
    ("g_b_proj",): (None, "heads"),
    ("g_proj",): ("embed", "heads"),
})
class KimiDeltaAttention(nn.Module):
    """The token mixer of a GLM-5.3-Flash `linear_attention` layer.

    Parameter names are the release's: `q_proj`, `k_proj`, `v_proj`,
    `{q,k,v}_conv1d/weight` `[H Dk, 1, K]`, `f_a_proj`, `f_b_proj`, `dt_bias`
    `[H Dk]` and `A_log` `[H]` directly under the layer (transformers renames
    them under `forget_gate`), `b_proj`, `g_a_proj`, `g_b_proj`, `o_norm/weight`
    `[Dk]`, `o_proj`. `lower_bound` is `linear_lower_bound`. The decode state is
    `GatedDeltaNet`'s pair of `cache` leaves.

    `full_rank_gate` is Kimi K3's output gate, one `g_proj` from the model width
    to the heads (`use_full_rank_gate`, modeling_kimi_linear.py:531-537, 651-656
    of moonshotai/Kimi-K3 at f831ab6); the rest of K3's layer is this one (fla
    `chunk_kda`, fla-core 0.5.2, fla/ops/kda/gate.py:57-70).
    """

    reads: ClassVar[frozenset[str]] = frozenset()
    """None of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`): a recurrence reads every earlier token
    of its document, with no mask or bias to narrow it."""

    emb_features: int
    num_heads: int
    head_dim: int
    conv_kernel: int = 4
    lower_bound: float | None = -5.0
    full_rank_gate: bool = False
    chunk_size: int = CHUNK_SIZE
    norm_eps: float = 1e-5
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def qkv_features(self) -> int:
        return self.num_heads * self.head_dim

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Q/K/V/output, low-rank decay and gate, convs, and chunked delta work."""
        inner = self.qkv_features
        direct = 4 * inner + self.head_dim + self.num_heads
        low_rank = self.head_dim * inner
        if self.full_rank_gate:
            direct += inner
        else:
            direct += self.head_dim
            low_rank += self.head_dim * inner
        return (projection_flops(lengths, self.emb_features, direct)
                + projection_flops(lengths, 1, low_rank)
                + 3 * conv_flops(lengths, inner, inner, self.conv_kernel, groups=inner)
                + delta_flops(lengths, self.num_heads, self.head_dim, self.head_dim))

    def setup(self):
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision,
                                  dot_general=float32_dot_general)
        self.q_proj = dense(self.qkv_features, name='q_proj')
        self.k_proj = dense(self.qkv_features, name='k_proj')
        self.v_proj = dense(self.qkv_features, name='v_proj')
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (
            DepthwiseConv1d(features=self.qkv_features, kernel=self.conv_kernel, name=f'{name}_conv1d')
            for name in 'qkv')
        self.f_a_proj = dense(self.head_dim, name='f_a_proj')
        self.f_b_proj = dense(self.qkv_features, name='f_b_proj')
        self.dt_bias = self.param('dt_bias', nn.initializers.zeros, (self.qkv_features,), jnp.float32)
        self.A_log = self.param('A_log', nn.initializers.zeros, (self.num_heads,), jnp.float32)
        self.b_proj = dense(self.num_heads, name='b_proj')
        if self.full_rank_gate:
            self.g_proj = dense(self.qkv_features, name='g_proj')
        else:
            self.g_a_proj = dense(self.head_dim, name='g_a_proj')
            self.g_b_proj = dense(self.qkv_features, name='g_b_proj')
        # The reference's fp32 norm with its weight, then the sigmoid of the
        # gate (Glm5NextTextRMSNormGated, modeling_glm5_next.py:346-358).
        self.o_norm = RMSNormGated(
            epsilon=self.norm_eps, activation="sigmoid", dtype=self.dtype, name="o_norm"
        )
        self.o_proj = dense(self.emb_features, name='o_proj')

    def _decay(self, x):
        """`Glm5NextTextForgetGate.forward` (modeling_glm5_next.py:319-335): `[B, S, H, Dk]` in fp32
        (`at_least_fp32`)."""
        B, S, _ = x.shape
        gate = self.f_b_proj(self.f_a_proj(x))
        gate = gate.astype(at_least_fp32(gate.dtype)) + self.dt_bias
        gate = gate.reshape(B, S, self.num_heads, self.head_dim)
        rate = jnp.exp(self.A_log)[None, None, :, None]
        if self.lower_bound is not None:
            return self.lower_bound * nn.sigmoid(rate * gate)
        return -rate * jnp.where(gate > 20.0, gate, jnp.log1p(jnp.exp(jnp.minimum(gate, 20.0))))

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL):
        decode, attention_metadata = call.decode, call.metadata
        B, S, _ = x.shape
        tokens = coordinates_of(attention_metadata, B, S)
        valid = tokens.valid
        if valid is not None:
            # The reference zeroes padded rows before projecting them
            # (apply_mask_to_padding_states, modeling_glm5_next.py:361-370).
            x = jnp.where(valid[:, :, None], x, 0)
        projections = (self.q_proj(x), self.k_proj(x), self.v_proj(x))
        if decode:
            # At one row XLA lowers each projection to a reduction and fuses the three into their
            # concatenate, which ran 34 of an H100's 151 us a step (`held_conv1d`'s barrier, behind it).
            projections = jax.lax.optimization_barrier(projections)
        projected = jnp.concatenate(projections, axis=-1)
        wide = at_least_fp32(projected.dtype)
        taps = jnp.concatenate([conv()[0] for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)])
        g = self._decay(x)
        beta = nn.sigmoid(self.b_proj(x).astype(wide))
        if valid is not None:
            # exp(0) = 1 and beta = 0 preserve the memory across a padded slot.
            g = jnp.where(valid[:, :, None, None], g, 0.0)
            beta = jnp.where(valid[:, :, None], beta, 0.0)

        def split(part):
            return part.reshape(B, S, self.num_heads, self.head_dim)

        def scan(convolved, *, starts, valid, held):
            del valid
            query, key, value = (split(part) for part in convolved)
            if S == 1:
                return recurrent_delta_rule(query, key, value, g, beta, held, starts=starts)
            # A held history can make the convolved operands vary where the projections did not, so the
            # route is asked again of them.
            return chunk_gated_delta_rule(query, key, value, g, beta, held, self.chunk_size,
                                          starts=starts, max_segments=tokens.max_segments,
                                          floor=self.lower_bound)

        # The decode state is GatedDeltaNet's pair, here in fp32; the
        # allocation-only call returns before any state is written. The conv
        # normalises the queries and keys (`l2norm`) and rounds once, to the
        # dtype the rule takes: the projections' where the port runs it, as
        # fla's layer hands them over, the wide one for the math.
        rows, keys, values = (split(part) for part in jnp.split(projected, 3, axis=-1))
        ported = S > 1 and kernel_routed(rows, keys, values, g, beta, floor=self.lower_bound)
        heads = ((self.qkv_features, self.head_dim),) * 2 + ((self.qkv_features, None),)
        out = held_scan(self, projected, taps, None, scan,
                        (self.num_heads, self.head_dim, self.head_dim), decode=decode, coordinates=tokens,
                        state='recurrent_state', state_dtype=wide, conv_dtype=wide, parts=heads,
                        dtype=projected.dtype if ported else wide)
        if out is None:
            return self.o_proj(jnp.zeros((B, S, self.qkv_features), self.dtype))
        gate = self.g_proj(x) if self.full_rank_gate else self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, S, self.num_heads, self.head_dim)
        out = self.o_norm(out, gate).reshape(B, S, self.qkv_features)
        return self.o_proj(out)


@dataclasses.dataclass(frozen=True)
class KimiDeltaAttentionMixer(MixerBase):
    """The `kimi_delta_attention` kind, by GLM-5.3-Flash's config fields:
    `linear_num_heads` heads of `linear_head_dim`, the depthwise conv's
    window and the forget gate's lower bound (configuration_glm5_next.py:143-146),
    and Kimi K3's `use_full_rank_gate` output gate."""

    residuals = DELTA_RULE_RESIDUALS

    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
    use_full_rank_gate: bool = False

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError(
                "kimi_delta_attention requires causal=True; its recurrence has no bidirectional mode"
            )
        return functools.partial(
            KimiDeltaAttention,
            emb_features=ctx.emb_features,
            num_heads=self.linear_num_heads,
            head_dim=self.linear_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
            lower_bound=self.linear_lower_bound,
            full_rank_gate=self.use_full_rank_gate,
            norm_eps=ctx.norm_eps,
            dtype=ctx.dtype,
            precision=ctx.precision)
