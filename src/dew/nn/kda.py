"""Kimi Delta Attention: the linear-attention layer of GLM-5.3-Flash.

KDA is the gated delta rule of `dew.nn.linear` with one change in the
recurrence and a different parameterisation around it
(`Glm5NextTextLinearAttention`, modeling_glm5_next.py:584-733):

- The decay is a vector per head, one entry per key dimension, not a
  scalar per head: `S <- S * exp(g_t)[:, None]` scales the rows of the
  `[Dk, Dv]` memory (modeling_glm5_next.py:468-471, 529-532). In the chunked
  form that puts `g` in every decay product where GDN broadcasts one value.
- `g` comes from a two-layer forget gate, `f_b_proj(f_a_proj(x)) + dt_bias`
  over `[H, Dk]`, and with the released `linear_lower_bound` it is
  `lower_bound * sigmoid(exp(A_log) * g)` rather than `-exp(A_log) *
  softplus(g)` (`Glm5NextTextForgetGate`, modeling_glm5_next.py:319-335).
- `beta = sigmoid(b_proj(x))` per head (modeling_glm5_next.py:697).
- q, k and v have their own projections and their own depthwise convs,
  which the reference concatenates and runs as one conv over `3 * H * Dk`
  channels (modeling_glm5_next.py:607-615, 642-649); the release stores the
  three convs apart as `q_conv1d`, `k_conv1d` and `v_conv1d` (transformers'
  conversion concatenates them on load), and so does this module, whose
  taps concatenate in the same order at call time.
- The output gate is `g_b_proj(g_a_proj(x))` through a sigmoid-gated RMSNorm
  with a weight, `o_norm` (modeling_glm5_next.py:339-358, 729-730), then
  `o_proj`.
- q and k are l2-normalised inside the rule as `x / sqrt(sum x^2 + eps)`
  (modeling_glm5_next.py:416-424).

Both rule forms live here beside their GDN counterparts' shapes:
`chunk_kimi_delta_rule` is `chunk_kimi_delta_attention`
(modeling_glm5_next.py:482-578) and `recurrent_kimi_delta_rule` is
`recurrent_kimi_delta_attention` (modeling_glm5_next.py:428-478), both in
fp32. tests/test_kda.py holds them to a float64 oracle of the reference.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .inputs import AttentionMetadata
from .linear import CHUNK_SIZE, DepthwiseConv1d, RMSNormGated, _masked_conv1d, causal_conv1d, l2norm
from .mixers import MixerBase, MixerContext, mixers
from .sharding import logical_axes


def chunk_kimi_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE):
    """The chunked KDA rule over `[B, S, H, D]` operands with `g` `[B, S, H, Dk]`
    and `beta` `[B, S, H]`: `(output [B, S, H, Dv], state [B, H, Dk, Dv])`.

    `chunk_kimi_delta_attention` (modeling_glm5_next.py:482-578) line for
    line in fp32. The reference's row correction loop inverts `I - A` for a
    strictly lower triangular `A`; here the series is summed by doubling, as
    `dew.nn.linear.chunk_gated_delta_rule` does and explains.
    """
    dtype = query.dtype
    query, key, value, g, beta = (x.astype(jnp.float32) for x in (query, key, value, g, beta))
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    pad = (chunk_size - S % chunk_size) % chunk_size
    query, key, value, g = (jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))) for x in (query, key, value, g))
    beta = jnp.pad(beta, ((0, 0), (0, pad), (0, 0)))
    T = S + pad
    query = query * (Dk ** -0.5)

    def chunks(x):  # [B, T, H, ...] -> [NC, B, H, C, ...] for the scan
        moved = jnp.moveaxis(x, 2, 1)
        blocked = moved.reshape(B, H, T // chunk_size, chunk_size, *moved.shape[3:])
        return jnp.moveaxis(blocked, 2, 0)

    q_c, k_c, g_c = chunks(query), chunks(key), chunks(g)
    kb_c, vb_c = chunks(key * beta[..., None]), chunks(value * beta[..., None])
    # Per key dimension: g [NC, B, H, C, Dk] cumulated within the chunk
    # (modeling_glm5_next.py:530), the decay between positions s >= t of
    # `exp(gc[s] - gc[t])` per dimension (modeling_glm5_next.py:532), zero
    # above the diagonal.
    gc = jnp.cumsum(g_c, axis=-2)
    inclusive = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_))[..., None]
    strict = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_), -1)
    diff = gc[..., :, None, :] - gc[..., None, :, :]  # [NC, B, H, C, C, Dk]
    # Masked before exp: the unused positive differences overflow otherwise,
    # and a where outside alone leaves 0 * inf in the gradient.
    decay = jnp.where(inclusive, jnp.exp(jnp.where(inclusive, diff, 0.0)), 0.0)
    # attn[s, t] = -sum_d k_beta[s, d] k[t, d] decay[s, t, d], strictly lower
    # (the reference's `masked_fill(triu(0), 0)`, modeling_glm5_next.py:533).
    attn = jnp.where(strict, -jnp.einsum('...sd,...td,...std->...st', kb_c, k_c, decay), 0.0)
    inv = jnp.broadcast_to(jnp.eye(chunk_size, dtype=attn.dtype), attn.shape)
    power = attn
    for _ in range(max(1, math.ceil(math.log2(chunk_size)))):
        inv = inv + power @ inv
        power = power @ power
    out_vals = inv @ vb_c
    k_cumdecay = inv @ (kb_c * jnp.exp(gc))

    state = (jnp.zeros((B, H, Dk, Dv), jnp.float32) if state is None else state.astype(jnp.float32))

    def one_chunk(s, step):
        q_i, k_i, v_i, decay_i, kd_i, gc_i = (step[name] for name in ('q', 'k', 'v', 'decay', 'kd', 'gc'))
        attn_inter = (q_i * jnp.exp(gc_i)) @ s
        # decay is zero above the diagonal, so this is the reference's
        # inclusive-lower `masked_fill(triu(1), 0)` (modeling_glm5_next.py:560).
        attn_intra = jnp.einsum('...sd,...td,...std->...st', q_i, k_i, decay_i)
        v_corrected = v_i - kd_i @ s
        out = attn_inter + attn_intra @ v_corrected
        last = gc_i[..., -1:, :]
        s = s * jnp.exp(last)[..., 0, :, None] + jnp.swapaxes(k_i * jnp.exp(last - gc_i), -1, -2) @ v_corrected
        return s, out

    state, core = jax.lax.scan(
        one_chunk, state,
        {'q': q_c, 'k': k_c, 'v': out_vals, 'decay': decay, 'kd': k_cumdecay, 'gc': gc})
    core = jnp.moveaxis(core, 0, 2).reshape(B, H, T, Dv)
    core = jnp.moveaxis(core, 1, 2)[:, :S]
    return core.astype(dtype), state.astype(dtype)


def recurrent_kimi_delta_rule(query, key, value, g, beta, state=None):
    """One token at a time, `recurrent_kimi_delta_attention`
    (modeling_glm5_next.py:428-478) in fp32 as a scan over time."""
    dtype = query.dtype
    query, key, value, g, beta = (x.astype(jnp.float32) for x in (query, key, value, g, beta))
    query = query * (key.shape[-1] ** -0.5)

    def one_token(s, step):
        q_t, k_t, v_t, g_t, beta_t = (step[name] for name in ('q', 'k', 'v', 'g', 'beta'))
        s = s * jnp.exp(g_t)[..., :, None]                  # rows decay per key dimension
        kv_mem = jnp.sum(s * k_t[..., :, None], axis=-2)   # [B, H, Dv]
        delta = (v_t - kv_mem) * beta_t[..., None]
        s = s + k_t[..., :, None] * delta[..., None, :]
        return s, jnp.sum(s * q_t[..., :, None], axis=-2)

    if state is None:
        state = jnp.zeros((query.shape[0], query.shape[2], key.shape[-1], value.shape[-1]), jnp.float32)
    state, out = jax.lax.scan(
        one_token, state.astype(jnp.float32),
        {name: jnp.moveaxis(x, 1, 0) for name, x in
         (('q', query), ('k', key), ('v', value), ('g', g), ('beta', beta))})
    return jnp.moveaxis(out, 0, 1).astype(dtype), state.astype(dtype)


# q/k/v_proj and o_proj carry the attention mixer's declarations under the
# same names; the gates' low-rank pairs and the beta projection are KDA's own.
@logical_axes({
    ("b_proj",): ("embed", "kv"),
    ("f_a_proj",): ("embed", None),
    ("f_b_proj",): (None, "heads"),
    ("g_a_proj",): ("embed", None),
    ("g_b_proj",): (None, "heads"),
}, heuristic=(("q_conv1d",), ("k_conv1d",), ("v_conv1d",)))
class KimiDeltaAttention(nn.Module):
    """The token mixer of a GLM-5.3-Flash `linear_attention` layer.

    Parameter names are the checkpoint's under the reference's module:
    `q_proj`, `k_proj`, `v_proj`, `{q,k,v}_conv1d/weight` `[H Dk, 1, K]`, the
    forget gate's `f_a_proj`, `f_b_proj`, `dt_bias` `[H Dk]` and `A_log` `[H]`
    (which the release stores directly under the layer and transformers'
    conversion renames under `forget_gate`; the loader keeps the released
    names), `b_proj`, `g_a_proj`, `g_b_proj`, `o_norm/weight` `[Dk]`, `o_proj`.

    `lower_bound` is the config's `linear_lower_bound`: with it the decay is
    `lower_bound * sigmoid(exp(A_log) * g)`, without it the softplus form
    (modeling_glm5_next.py:327-335). The decode state is the same pair of
    `cache` leaves `GatedDeltaNet` keeps, allocated on the first decode call.
    """

    emb_features: int
    num_heads: int
    head_dim: int
    conv_kernel: int = 4
    lower_bound: float | None = -5.0
    chunk_size: int = CHUNK_SIZE
    norm_eps: float = 1e-5
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def qkv_features(self) -> int:
        return self.num_heads * self.head_dim

    def setup(self):
        if self.conv_kernel < 2:
            raise ValueError(f"the causal conv needs a history, so a kernel of at least 2, got {self.conv_kernel}")
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
        self.g_a_proj = dense(self.head_dim, name='g_a_proj')
        self.g_b_proj = dense(self.qkv_features, name='g_b_proj')
        # The reference's fp32 norm with its weight, then the sigmoid of the
        # gate (Glm5NextTextRMSNormGated, modeling_glm5_next.py:346-358).
        self.o_norm = RMSNormGated(epsilon=self.norm_eps, activation='sigmoid', dtype=self.dtype, name='o_norm')
        self.o_proj = dense(self.emb_features, name='o_proj')

    def _decay(self, x):
        """`Glm5NextTextForgetGate.forward` (modeling_glm5_next.py:319-335): `[B, S, H, Dk]` in fp32."""
        B, S, _ = x.shape
        gate = self.f_b_proj(self.f_a_proj(x)).astype(jnp.float32) + self.dt_bias
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
        conv_input = jnp.moveaxis(
            jnp.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1).astype(jnp.float32), 2, 1)
        taps = jnp.concatenate([conv()[0] for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)])
        recurrent = None
        if decode:
            allocated = self.has_variable('cache', 'recurrent_state')
            conv_state = self.variable('cache', 'conv_state', jnp.zeros,
                                       (B, 3 * self.qkv_features, self.conv_kernel - 1), jnp.float32)
            recurrent = self.variable('cache', 'recurrent_state', jnp.zeros,
                                      (B, self.num_heads, self.head_dim, self.head_dim), jnp.float32)
            if not allocated:
                # Allocation only, as GatedDeltaNet: init_cache's dummy token
                # must not consume a position or leave state behind.
                return self.o_proj(jnp.zeros((B, S, self.qkv_features), self.dtype))
            if valid is not None:
                mixed, history = _masked_conv1d(conv_input, taps, valid, conv_state.value)
                conv_state.value = history
            else:
                history = jnp.concatenate([conv_state.value, conv_input], axis=2)
                conv_state.value = history[:, :, -(self.conv_kernel - 1):]
                mixed = causal_conv1d(history, taps)[..., -S:]
        elif valid is not None:
            mixed, _ = _masked_conv1d(conv_input, taps, valid)
        else:
            mixed = causal_conv1d(conv_input, taps)
        mixed = jnp.moveaxis(mixed, 2, 1)
        query, key, value = (part.reshape(B, S, self.num_heads, self.head_dim)
                             for part in jnp.split(mixed, 3, axis=-1))
        query, key = l2norm(query), l2norm(key)
        g = self._decay(x)
        beta = nn.sigmoid(self.b_proj(x).astype(jnp.float32))
        if valid is not None:
            # exp(0) = 1 and beta = 0 preserve the memory across a padded slot.
            g = jnp.where(valid[:, :, None, None], g, 0.0)
            beta = jnp.where(valid[:, :, None], beta, 0.0)
        rule = recurrent_kimi_delta_rule if S == 1 else functools.partial(
            chunk_kimi_delta_rule, chunk_size=self.chunk_size)
        out, final = rule(query, key, value, g, beta, None if recurrent is None else recurrent.value)
        if recurrent is not None:
            recurrent.value = final
        gate = self.g_b_proj(self.g_a_proj(x)).reshape(B, S, self.num_heads, self.head_dim)
        out = self.o_norm(out, gate).reshape(B, S, self.qkv_features)
        return self.o_proj(out)


@mixers("kimi_delta_attention")
@dataclasses.dataclass(frozen=True)
class KimiDeltaAttentionMixer(MixerBase):
    """The `kimi_delta_attention` kind, by GLM-5.3-Flash's config fields:
    `linear_num_heads` heads of `linear_head_dim`, the depthwise conv's
    window and the forget gate's lower bound (configuration_glm5_next.py:143-146)."""

    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("kimi_delta_attention requires causal=True; its recurrence has no bidirectional mode")
        return functools.partial(
            KimiDeltaAttention,
            emb_features=ctx.emb_features,
            num_heads=self.linear_num_heads,
            head_dim=self.linear_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
            lower_bound=self.linear_lower_bound,
            norm_eps=ctx.norm_eps,
            dtype=ctx.dtype,
            precision=ctx.precision)
