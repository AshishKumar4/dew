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

`chunk_kimi_delta_rule` is `chunk_kimi_delta_attention` (:482-578) in fp32 on
the gated delta rule's chunked form, `dew.nn.linear.chunk_delta_rule`, and
the recurrent form `dew.nn.linear.recurrent_delta_rule` (:428-478).
tests/test_kda.py holds both to a float64 oracle of the reference.
"""

from __future__ import annotations

import dataclasses
import functools

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .inputs import AttentionMetadata
from .linear import (
    CHUNK_SIZE,
    DepthwiseConv1d,
    RMSNormGated,
    chunk_delta_rule,
    held_scan,
    l2norm,
    recurrent_delta_rule,
)
from .mixer_base import MixerBase, MixerContext
from .precision import at_least_fp32
from .sharding import logical_axes


def chunk_kimi_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE):
    """`chunk_kimi_delta_attention` (modeling_glm5_next.py:482-578): the
    chunked rule (`dew.nn.linear.chunk_delta_rule`) with `g` `[B, S, H, Dk]`,
    a log decay per key dimension, on the XLA recurrence, as the Pallas
    kernels decay per head."""
    return chunk_delta_rule(query, key, value, g, beta, state, chunk_size, implementation='xla')


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

    def setup(self):
        if self.conv_kernel < 2:
            raise ValueError(
                f"the causal conv needs a history, so a kernel of at least 2, got {self.conv_kernel}"
            )
        dense = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
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
    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None):
        del positions, segment_ids, kv_store
        B, S, _ = x.shape
        valid = None if attention_metadata is None else attention_metadata.valid
        if valid is not None and valid.shape != (B, S):
            raise ValueError(f"row validity must be {(B, S)}, got {valid.shape}")
        if valid is not None:
            # The reference zeroes padded rows before projecting them
            # (apply_mask_to_padding_states, modeling_glm5_next.py:361-370).
            x = jnp.where(valid[:, :, None], x, 0)
        projected = jnp.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1)
        wide = at_least_fp32(projected.dtype)
        taps = jnp.concatenate([conv()[0] for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)])
        g = self._decay(x)
        beta = nn.sigmoid(self.b_proj(x).astype(wide))
        if valid is not None:
            # exp(0) = 1 and beta = 0 preserve the memory across a padded slot.
            g = jnp.where(valid[:, :, None, None], g, 0.0)
            beta = jnp.where(valid[:, :, None], beta, 0.0)

        def scan(convolved, *, starts, valid, held):
            del starts, valid
            query, key, value = (part.reshape(B, S, self.num_heads, self.head_dim)
                                 for part in jnp.split(jnp.moveaxis(convolved, 2, 1), 3, axis=-1))
            rule = recurrent_delta_rule if S == 1 else functools.partial(
                chunk_kimi_delta_rule, chunk_size=self.chunk_size)
            return rule(l2norm(query), l2norm(key), value, g, beta, held)

        # The decode state is GatedDeltaNet's pair, here in fp32; the
        # allocation-only call returns before any state is written.
        out = held_scan(self, projected.astype(wide), taps, None, scan,
                        (self.num_heads, self.head_dim, self.head_dim), decode=decode, valid=valid,
                        state='recurrent_state')
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
