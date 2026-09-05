"""Gated DeltaNet: the linear-attention mixer of the Qwen3.5 family.

The delta rule keeps an outer-product memory `S = sum_t k_t v_t^T` and
corrects it toward the value each new key predicts, rather than only
accumulating onto it the way attention accumulates its keys. Two gates make
it trainable at scale: a decay `g` that shrinks the memory before each
write (Mamba's selectivity, spelled as a log-space cumulative product in
the chunked form) and a beta that scales how far one write moves the memory
toward its own value.

dew computes the same chunked formulation transformers 5.16.1 computes
(`modeling_qwen3_next.py:374-453`, identical in qwen3_5 and qwen4_exp):
per chunk of 64 tokens the intra-chunk attention `(I+A)^-1 @ v_beta`
resolves the sequential dependencies of the delta rule inside the chunk in
matrix form, and the memory crosses chunk boundaries through
`k_cumdecay @ S`. A sequence never materialises anything quadratic in the
sequence length; the largest new tensor is `[C, C]` per head per chunk.

The recurrent form (one token at a time) is what a decode step runs
(`modeling_qwen3_next.py:456-506`):

    S <- S * exp(g_t)
    delta <- (v_t - S k_t) * beta_t
    S <- S + k_t delta^T
    o_t <- S q_t

Both forms live here because a model decodes with the second and trains
with the first; the parallel-scored test in tests/test_linear_attention.py
holds them to the same numbers, which is what catches a wrong recurrent
state.

The short mixer is the depthwise causal conv1d the reference applies to the
projected qkv before the rule (`modeling_qwen3_next.py:325-365`): kernel 4,
zero padding to the left so a token sees itself and its three predecessors,
silu after. Decode keeps the last `kernel - 1` columns as a state, exactly
as the reference's `update_conv_state` does.

Parameter names are the checkpoint's: in_proj_qkv, in_proj_z, in_proj_b,
in_proj_a, conv1d, A_log, dt_bias, norm, out_proj. A_log and dt_bias are
separate leaves because the reference materialises them as parameters, and
`g = -exp(A_log) * softplus(a + dt_bias)` reads them per value head.
"""

import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .sharding import logical_axes

CHUNK_SIZE = 64
"""Tokens per chunk of the chunked form, the reference's default
(`torch_chunk_gated_delta_rule(..., chunk_size=64)`). The chunked and the
recurrent form agree at any size; the size only trades memory for
parallelism, and 64 is what the reference's kernels assume."""


def l2norm(x, eps: float = 1e-6):
    """The FLA library's normalisation of q and k, which the reference
    applies inside the rule (`l2norm`, modeling_qwen3_next.py:368-371):
    unit keys and queries per head, so `q k^T` is a cosine and the
    `1/sqrt(d)` scale keeps it bounded."""
    inv = jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * inv


def causal_conv1d(x, kernel, activation: bool = True):
    """Depthwise causal conv over [B, D, S] with the [D, K] taps.

    The reference pads `K - 1` zeros to the left
    (`F.conv1d(padding=kernel_size - 1)`, modeling_qwen3_next.py:345-365)
    so position s convolves s-K+1..s, then applies silu. `kernel` is
    `conv1d.weight[:, 0, :]`, the checkpoint's [D, K] depthwise taps.

    Depthwise in lax terms: the input's channel axis is the feature axis of
    a grouped conv with one channel per group, so the taps land as
    [D, 1, K, 1, 1] (lhs feature, window, rhs feature), matching
    `feature_group_count=D` on a one-in-one-out grouping.
    """
    D, K = kernel.shape
    padded = jnp.pad(x, ((0, 0), (0, 0), (K - 1, 0)))
    taps = kernel[:, None, :]  # [D(out), 1(in per group), K] in OIH terms
    windows = jax.lax.conv_general_dilated(
        padded, taps,
        window_strides=(1,), padding='VALID',
        dimension_numbers=('NCH', 'OIH', 'NCH'),
        feature_group_count=D)
    if activation:
        windows = nn.silu(windows)
    return windows


def chunk_gated_delta_rule(query, key, value, g, beta, state=None,
                           chunk_size: int = CHUNK_SIZE):
    """The chunked form of the gated delta rule, the reference's math.

    Operands are [B, S, H, D] (the mixer's head layout); the computation
    matches `torch_chunk_gated_delta_rule` (modeling_qwen3_next.py:374-453)
    line for line, in fp32, and returns
    `(output [B, S, H, Dv], final_state [B, H, Dk, Dv])` in the input dtype.

    The reference's sequential correction (`for i in range(1, chunk_size)`)
    is the forward substitution that inverts `I - A` for a strictly lower
    triangular A, so `looped + I` is `I + A + A^2 + ...`, which terminates
    at C-1 powers because A is nilpotent. The series is summed by doubling
    (`S <- S + A^(2^k) S; A <- A^2`), log2(C) matmuls instead of C row
    updates; see the comment where it happens for the verification.
    """
    dtype = query.dtype
    query, key, value, g, beta = (
        x.astype(jnp.float32) for x in (query, key, value, g, beta))
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    pad = (chunk_size - S % chunk_size) % chunk_size
    query = jnp.pad(query, ((0, 0), (0, pad), (0, 0), (0, 0)))
    key = jnp.pad(key, ((0, 0), (0, pad), (0, 0), (0, 0)))
    value = jnp.pad(value, ((0, 0), (0, pad), (0, 0), (0, 0)))
    beta = jnp.pad(beta, ((0, 0), (0, pad), (0, 0)))
    g = jnp.pad(g, ((0, 0), (0, pad), (0, 0)))
    T = S + pad
    query = query * (Dk ** -0.5)

    def chunks(x):  # [B, S, H, ...] -> [NC, B, H, C, ...] for the scan
        moved = jnp.moveaxis(x, 2, 1)  # [B, H, S, ...]
        blocked = moved.reshape(B, H, T // chunk_size, chunk_size, *moved.shape[3:])
        return jnp.moveaxis(blocked, 2, 0)  # [NC, B, H, C, ...]

    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]
    q_c, k_c, v_c = chunks(query), chunks(key), chunks(value)
    kb_c, vb_c, g_c = chunks(k_beta), chunks(v_beta), chunks(g)

    # Cumulative log decay within each chunk, the reference's
    # `g = g.cumsum(dim=-1)` (modeling_qwen3_next.py:417).
    gc = jnp.cumsum(g_c, axis=-1)  # [B, H, NC, C]

    # decay[i, s, t] = exp(gc[s] - gc[t]) for s >= t else 0, the reference's
    # `((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()`.
    inclusive = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_))
    diff = gc[..., :, None] - gc[..., None, :]  # [B, H, NC, C, C]
    decay = jnp.where(inclusive, jnp.exp(diff), 0.0)
    # The strictly-lower operator the reference inverts row by row: its loop
    # `attn[i, :i] += sum_k attn[i, k] attn[k, :i]`, iterated to the last row,
    # computes exactly (I - A)^-1 = I + A + A^2 + ... for a nilpotent A
    # (verified against the loop at C=4 and C=64: they agree to 4e-15). The
    # series is summed by doubling: S <- S + A^(2^k) S, A <- A^2, which is
    # log2(C) matmuls instead of C row updates. The mask is strictly lower
    # (tril, -1): the reference's masked_fill zeroes the diagonal too.
    strict = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_), -1)
    attn = jnp.where(strict, -(kb_c @ jnp.swapaxes(k_c, -1, -2)) * decay, 0.0)
    inv = jnp.broadcast_to(jnp.eye(chunk_size, dtype=attn.dtype), attn.shape)
    power = attn
    for _ in range(max(1, int(math.ceil(math.log2(chunk_size))))):
        inv = inv + power @ inv
        power = power @ power
    # inv is (I + A)^-1 where A = attn, the reference's `attn + I` operator
    # after its row correction loop.
    out_vals = inv @ vb_c  # the reference's `value = attn @ v_beta`
    k_cumdecay = inv @ (kb_c * jnp.exp(gc)[..., None])

    state = (jnp.zeros((B, H, Dk, Dv), jnp.float32) if state is None
             else state.astype(jnp.float32))

    def one_chunk(carry, step):
        s = carry
        q_i, k_i, v_i = step['q'], step['k'], step['v']
        decay_i, kd_i, gc_i = step['decay'], step['kd'], step['gc']
        attn_i = q_i @ jnp.swapaxes(k_i, -1, -2) * decay_i
        v_prime = kd_i @ s
        v_new = v_i - v_prime
        attn_inter = (q_i * jnp.exp(gc_i)[..., None]) @ s
        out = attn_inter + attn_i @ v_new
        # The chunk's write into the memory, the reference's
        # `s * exp(gc[-1]) + (k * exp(gc[-1] - gc))^T @ v_new`
        # (modeling_qwen3_next.py:443-446).
        s = (s * jnp.exp(gc_i[..., -1])[..., None, None]
             + jnp.swapaxes(
                 k_i * jnp.exp(gc_i[..., -1][..., None] - gc_i)[..., None],
                 -1, -2) @ v_new)
        return s, out
    state, core = jax.lax.scan(
        one_chunk, state,
        {'q': q_c, 'k': k_c, 'v': out_vals, 'decay': decay,
         'kd': k_cumdecay, 'gc': gc})
    # core: [NC, B, H, C, Dv] -> [B, H, NC*C, Dv] -> [B, S, H, Dv]
    core = jnp.moveaxis(core, 0, 2).reshape(B, H, T, Dv)
    core = jnp.moveaxis(core, 1, 2)[:, :S]
    return core.astype(dtype), state.astype(dtype)


def recurrent_gated_delta_rule(query, key, value, g, beta, state=None):
    """One-token-at-a-time form, the decode path.

    `torch_recurrent_gated_delta_rule` (modeling_qwen3_next.py:456-506)
    verbatim, in fp32, as a lax.scan over the time axis so the state rides
    the scan the same way it rides the decode cache.
    """
    dtype = query.dtype
    query, key, value, g, beta = (
        x.astype(jnp.float32) for x in (query, key, value, g, beta))
    query = query * (key.shape[-1] ** -0.5)

    def one_token(carry, step):
        s = carry
        q_t, k_t, v_t = step['q'], step['k'], step['v']
        g_t = jnp.exp(step['g'])                    # [B, H]
        beta_t = step['beta']                        # [B, H]
        s = s * g_t[..., None, None]                 # [B, H, Dk, Dv]
        kv_mem = jnp.sum(s * k_t[..., :, None], axis=-2)   # [B, H, Dv]
        delta = (v_t - kv_mem) * beta_t[..., None]        # [B, H, Dv]
        s = s + k_t[..., :, None] * delta[..., None, :]    # [B, H, Dk, Dv]
        out = jnp.sum(s * q_t[..., :, None], axis=-2)      # [B, H, Dv]
        return s, out
    # The scan stacks along the first axis, so the operands go time-major.
    if state is None:
        state = jnp.zeros((query.shape[0], query.shape[-2], key.shape[-1],
                           value.shape[-1]), jnp.float32)
    state, out = jax.lax.scan(
        one_token, state.astype(jnp.float32),
        {name: jnp.moveaxis(x, 1, 0) for name, x in
         (('q', query), ('k', key), ('v', value), ('g', g), ('beta', beta))})
    return jnp.moveaxis(out, 0, 1).astype(dtype), state.astype(dtype)


@logical_axes({
    ("in_proj_qkv",): ("embed", "linear"),
    ("in_proj_z",): ("embed", "linear"),
    ("in_proj_b",): ("embed", "kv"),
    ("in_proj_a",): ("embed", "kv"),
    ("out_proj",): ("linear", "embed"),
}, heuristic=(("conv1d",),))
class GatedDeltaNet(nn.Module):
    """The token mixer of a linear_attention layer.

    Projects the hidden state into q, k, v, z, b and a; runs the depthwise
    causal conv over qkv; applies the gated delta rule chunked in training
    and recurrently when decoding; and gates the output with the
    RMSNormGated the reference applies: norm first, then silu(z) (or
    sigmoid(z), which qwen4_exp's output_gate_type names).

    `num_v_heads // num_k_heads` value heads share one key head, which is
    what the reference's `repeat_interleave` says; q and k are broadcast to
    the value head count before the rule, so a key's memory serves every
    value head it covers.

    The decode state is two leaves in the flax `cache` collection:
    `recurrent_state` [B, H, Dk, Dv] and `conv_state` [B, D, K-1], both
    allocated at the batch the first decode-mode call sees, the way
    open_kv_cache allocates its slots. Decode is one code path with
    prefill: the conv state crosses the prefill/decode boundary because a
    continuation must see the last K-1 real columns rather than the zeros a
    fresh sequence pads with.

    Parameter names are the checkpoint's, so translation moves weights and
    does not synthesise them: `conv1d/weight` is the depthwise taps
    `[D, 1, K]`, and `A_log`/`dt_bias` are the `[Hv]` leaves the reference
    materialises as parameters.
    """

    emb_features: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int = 4
    max_seq_len: Optional[int] = None
    chunk_size: int = CHUNK_SIZE
    norm_eps: float = 1e-6
    gate_activation: str = 'silu'  # the norm's gate: 'silu' | 'sigmoid', qwen4_exp's output_gate_type
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @property
    def key_features(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_features(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_features(self) -> int:
        return 2 * self.key_features + self.value_features

    @property
    def per_v(self) -> int:
        return self.num_v_heads // self.num_k_heads

    def setup(self):
        if self.num_v_heads % self.num_k_heads:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be a multiple of "
                f"num_k_heads ({self.num_k_heads})")
        if self.conv_kernel < 2:
            raise ValueError(
                "the causal conv needs a history, so a kernel of at least 2, "
                f"got {self.conv_kernel}")
        if self.gate_activation not in ('silu', 'sigmoid'):
            raise ValueError(
                "gate_activation must be 'silu' or 'sigmoid' (the reference's "
                f"output_gate_type), got {self.gate_activation!r}")
        dense = functools.partial(
            nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.in_proj_qkv = dense(self.key_features * 2 + self.value_features,
                                 name='in_proj_qkv')
        self.in_proj_z = dense(self.value_features, name='in_proj_z')
        self.in_proj_b = dense(self.num_v_heads, name='in_proj_b')
        self.in_proj_a = dense(self.num_v_heads, name='in_proj_a')
        self.conv1d = DepthwiseConv1d(features=self.conv_features,
                                       kernel=self.conv_kernel, name='conv1d')
        self.A_log = self.param('A_log', nn.initializers.constant(0.0),
                                (self.num_v_heads,), jnp.float32)
        self.dt_bias = self.param('dt_bias', nn.initializers.ones,
                                  (self.num_v_heads,), jnp.float32)
        self.out_proj = dense(self.emb_features, name='out_proj')
        self.norm = RMSNormGated(epsilon=self.norm_eps, activation=self.gate_activation,
                                 dtype=self.dtype, name='norm')

    def _split_heads(self, mixed, last_dim: int):
        return mixed.reshape(*mixed.shape[:-1], -1, last_dim)  # [B, S, H, D]

    def _expand_kv(self, x):
        """One key head serves `per_v` value heads, the reference's
        `repeat_interleave(num_v_heads // num_k_heads, dim=2)`."""
        if self.per_v == 1:
            return x
        B, S, H, D = x.shape
        return jnp.repeat(x.reshape(B, S, H, 1, D), self.per_v, axis=3).reshape(
            B, S, H * self.per_v, D)

    def _conv_taps(self):
        """The depthwise taps [D, K]; calling the conv initialises its param."""
        return jnp.asarray(self.conv1d()[:, 0, :], jnp.float32)

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None):
        del positions, segment_ids, kv_store
        B, S, _ = x.shape
        projected = self.in_proj_qkv(x)
        key_dim, value_dim = self.key_features, self.value_features
        query, key, value = jnp.split(projected, [key_dim, 2 * key_dim], axis=-1)
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        # fp32 on purpose, as the reference notes: an fp16 A can make exp
        # underflow to -inf (modeling_qwen3_next.py:652-653).
        conv_input = jnp.moveaxis(
            jnp.concatenate([query, key, value], axis=-1).astype(jnp.float32),
            2, 1)  # [B, D, S], the conv's channel-major layout
        recurrent = None
        if decode:
            # The first decode-mode call only allocates, the way
            # open_kv_cache's is: init_cache's dummy token must not consume
            # a position or leave state behind.
            allocated = self.has_variable('cache', 'recurrent_state')
            conv_state = self.variable(
                'cache', 'conv_state', jnp.zeros,
                (B, self.conv_features, self.conv_kernel - 1), jnp.float32)
            recurrent = self.variable(
                'cache', 'recurrent_state', jnp.zeros,
                (B, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                jnp.float32)
            if not allocated:
                # Allocation only: the caller's first real forward, not this
                # call, starts the state.
                mixed = causal_conv1d(conv_input, self._conv_taps())
                out = jnp.zeros((B, S, self.value_features), self.dtype)
                return self.out_proj(out)
            history = jnp.concatenate([conv_state.value, conv_input], axis=2)
            # The next step sees the last K-1 columns of this one.
            conv_state.value = history[:, :, -(self.conv_kernel - 1):]
            # The whole history goes through the same conv, so a multi-token
            # prefill against a live cache and a single-token step land in
            # one code path; only the new columns' outputs are kept.
            mixed = causal_conv1d(history, self._conv_taps())[..., -S:]
        else:
            mixed = causal_conv1d(conv_input, self._conv_taps())
        mixed = jnp.moveaxis(mixed, 2, 1)  # back to [B, S, D]

        query, key, value = jnp.split(mixed, [key_dim, 2 * key_dim], axis=-1)
        query = self._expand_kv(self._split_heads(query, self.head_k_dim))
        key = self._expand_kv(self._split_heads(key, self.head_k_dim))
        value = self._split_heads(value, self.head_v_dim)

        beta = nn.sigmoid(b.astype(jnp.float32))
        g = -jnp.exp(self.A_log.astype(jnp.float32)) * nn.softplus(
            a.astype(jnp.float32) + self.dt_bias.astype(jnp.float32))

        query = l2norm(query)
        key = l2norm(key)

        out, final = (recurrent_gated_delta_rule(
                          query, key, value, g, beta,
                          None if recurrent is None else recurrent.value)
                      if S == 1 else
                      chunk_gated_delta_rule(
                          query, key, value, g, beta,
                          None if recurrent is None else recurrent.value,
                          self.chunk_size))
        if recurrent is not None:
            recurrent.value = final

        gate = z.reshape(B, S, -1, self.head_v_dim)
        out = self.norm(out, gate)  # the norm scales, then the activation gates
        out = out.reshape(B, S, self.value_features)
        return self.out_proj(out)


class DepthwiseConv1d(nn.Module):
    """The conv's taps as a raw parameter, in the checkpoint's [D, 1, K] layout.

    A raw parameter rather than flax's Conv, because the checkpoint stores
    `conv1d.weight` exactly this way and nothing else about flax's conv
    (its [K, D, 1] kernel order, its channel-last input) matches the
    reference's [B, D, S] depthwise conv1d call. The leaf keeps the
    checkpoint's name, `weight`, because `kernel` is what a translation
    transposes as a Linear's [out, in]; the caller reads `weight[:, 0, :]`
    for the [D, K] taps. The depthwise taps have no matrix axis worth a
    name, so they take the shape heuristic.
    """

    features: int
    kernel: int = 4

    @nn.compact
    def __call__(self):
        return self.param('weight', nn.initializers.lecun_normal(),
                         (self.features, 1, self.kernel))


class RMSNormGated(nn.Module):
    """The reference's Qwen3NextRMSNormGated: RMSNorm in fp32, then the
    gate, then the cast back (modeling_qwen3_next.py:57-74). The gate is
    silu in the qwen3_5/qwen3_next references and sigmoid where qwen4_exp's
    output_gate_type says so; the caller picks, because the activation is
    a config field there, not a property of the norm.
    """

    epsilon: float = 1e-6
    activation: str = 'silu'
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, x, gate):
        dtype = self.dtype if self.dtype is not None else x.dtype
        y = x.astype(jnp.float32)
        y = y * jax.lax.rsqrt(
            jnp.mean(jnp.square(y), axis=-1, keepdims=True) + self.epsilon)
        scale = self.param('weight', nn.initializers.ones,
                           (x.shape[-1],), jnp.float32)
        y = y * scale.astype(jnp.float32)
        gate = gate.astype(jnp.float32)
        gated = (nn.silu(gate) if self.activation == 'silu'
                 else nn.sigmoid(gate))
        return (y * gated).astype(dtype)
