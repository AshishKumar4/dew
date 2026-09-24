"""Gated DeltaNet: the linear-attention mixer of the Qwen3.5 family.

The delta rule keeps an outer-product memory `S = sum_t k_t v_t^T` and
corrects it toward the value each new key predicts, where attention only
accumulates its keys. Two gates make it trainable at scale: a decay `g`
that shrinks the memory before each write (Mamba's selectivity, spelled as
a log-space cumulative product in the chunked form) and a beta that scales
how far one write moves the memory toward its own value.

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
with the first; tests/test_linear_attention.py holds them to the same
numbers.

The short mixer is the depthwise causal conv1d the reference applies to the
projected qkv before the rule (`modeling_qwen3_next.py:325-365`): kernel 4,
zero padding to the left so a token sees itself and its three predecessors,
silu after. Decode keeps the last `kernel - 1` columns as a state, exactly
as the reference's `update_conv_state` does.

Parameter names are the checkpoint's: in_proj_qkv, in_proj_z, in_proj_b,
in_proj_a, conv1d, A_log, dt_bias, norm, out_proj. A_log and dt_bias are
separate leaves because the reference materialises them as parameters, and
`g = -exp(A_log) * softplus(a + dt_bias)` reads them per value head.
Qwen3-Next fuses the four input projections into two, `in_proj_qkvz` and
`in_proj_ba`, whose rows are grouped by key head
(`modeling_qwen3_next.py:540-586`); `fused_in_proj` keeps those two leaves
and splits them the way the reference does.
"""

import functools
import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.scatter import DROPPED

from .blocks import normal_kernel
from .inputs import AttentionMetadata
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


def causal_conv1d(x, kernel, activation: bool = True, bias=None):
    """Depthwise causal conv over [B, D, S] with the [D, K] taps.

    The reference pads `K - 1` zeros to the left
    (`F.conv1d(padding=kernel_size - 1)`, modeling_qwen3_next.py:345-365)
    so position s convolves s-K+1..s, then applies silu. `kernel` is
    `conv1d.weight[:, 0, :]`, the checkpoint's [D, K] depthwise taps, and
    `bias` the `[D]` per-channel bias a conv with one adds before the
    activation (Mamba-2's `use_conv_bias`).

    The taps apply as K shifted products summed in fp32, not as a grouped
    `conv_general_dilated`: XLA's SPMD partitioner sums that conv's filter
    gradient over every device of the mesh, the ones that hold the same rows
    included, so a batch split over part of the mesh (fsdp beside a tensor
    axis the conv's input does not use) doubled the taps' gradient (jax
    0.11.2). The products are what a depthwise conv computes anyway.
    """
    _, K = kernel.shape
    length = x.shape[-1]
    padded = jnp.pad(x, ((0, 0), (0, 0), (K - 1, 0))).astype(jnp.promote_types(x.dtype, jnp.float32))
    taps = kernel.astype(padded.dtype)
    windows = padded[..., :length] * taps[:, :1]
    for tap in range(1, K):
        windows = windows + padded[..., tap:tap + length] * taps[:, tap:tap + 1]
    windows = windows.astype(x.dtype)
    if bias is not None:
        windows = windows + bias.astype(windows.dtype)[None, :, None]
    if activation:
        windows = nn.silu(windows)
    return windows


def document_starts(segments, before=None, valid=None):
    """`[B, S]`: whether each token's segment differs from the one before it.
    `before` `[B]` is the segment of the token ahead of the first; None
    takes the first token's own, a row that continues whatever preceded it.
    With `valid` `[B, S]`, "before" is the previous real token, across any
    padding slots between, and a padding slot starts nothing."""
    if valid is None:
        before = segments[:, 0] if before is None else before
        previous = jnp.concatenate([before[:, None], segments[:, :-1]], axis=1)
        return segments != previous
    rank, source = _stream_order(valid)
    compact = jnp.take_along_axis(segments, source, axis=1, mode='fill', fill_value=-1)
    starts = document_starts(compact, before)
    return valid & jnp.take_along_axis(starts, jnp.maximum(rank, 0), axis=1)


def document_conv1d(history, x, history_segments, segments, taps, bias=None):
    """The causal depthwise conv of `causal_conv1d` over `x` `[B, D, S]`
    behind `history` `[B, D, K-1]`, where a tap reads a token only if it
    shares the output token's segment: each packed document convolves as if
    zeros preceded it, as mamba_ssm's `causal_conv1d_fn(seq_idx=...)` does.
    `taps` `[D, K]`, `history_segments` `[B, K-1]`, `segments` `[B, S]`;
    silu after the bias."""
    width = taps.shape[1] - 1
    length = x.shape[-1]
    stream = jnp.concatenate([history, x], axis=2)
    stream_segments = jnp.concatenate([history_segments, segments], axis=1)
    out = jnp.zeros_like(x) if bias is None else jnp.broadcast_to(bias[None, :, None], x.shape)
    for tap in range(width + 1):
        same = stream_segments[:, tap:tap + length] == segments
        out = out + taps[None, :, tap, None] * jnp.where(same[:, None], stream[..., tap:tap + length], 0)
    return nn.silu(out)


def _stream_order(valid):
    """Each slot's index among its row's real tokens (`cumsum(valid) - 1`)
    and, per compact column, the physical slot it holds. A column past the
    row's real tokens holds `length`, the zero column `_masked_conv1d` appends,
    so it gathers a zero and its cotangent lands there, never on a real slot;
    every real token writes its own column, so no two writers meet."""
    batch, length = valid.shape
    valid = jnp.asarray(valid, bool)
    rank = jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1
    source = jnp.full((batch, length), length, jnp.int32).at[
        jnp.arange(batch)[:, None], jnp.where(valid, rank, DROPPED)].set(
            jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length)), mode='drop')
    return rank, source


def _masked_conv1d(x, kernel, valid, state=None, bias=None, segments=None):
    """Convolve real tokens without advancing a paused row's history.

    A row's real tokens keep their order and their history: the j-th of them
    reads the K-1 real tokens before it, whatever padding sits between them,
    and an invalid slot neither reads a window nor advances the history. That
    is a statement about the row's stream of real tokens, so the stream is
    what the convolution reads. `cumsum(valid) - 1` is each real token's index
    in it; gathering by that index hands `causal_conv1d` the same ordered
    stream a token-by-token scan fed it, behind the row's existing history,
    in one convolution instead of S sequential windows.

    `segments` `[B, S]` makes each packed document convolve alone: a tap
    reads a real token only if it shares the output token's segment
    (`document_conv1d` over the compact stream). The held `state` counts as
    the first real token's document, which a decode call continues.

    Masking the input of an ordinary convolution over the physical slots
    instead is a different function: a token after an interior gap would read
    the zeros in the gap rather than the real tokens before it. On nine slots
    with holes at 2, 5 and 6 that moves the outputs by 5.0 in fp32, which
    test_the_masked_conv_reads_across_a_gap measures.
    """
    batch, channels, _ = x.shape
    width = kernel.shape[1] - 1
    if state is None:
        state = jnp.zeros((batch, channels, width), x.dtype)
    valid = jnp.asarray(valid, bool)
    rank, source = _stream_order(valid)
    # Gathered in bounds from one appended zero column: an out-of-range 'fill'
    # gather differentiates into a scatter-add whose dropped index XLA's
    # deterministic GPU scatter writes onto the next channel (openxla/xla#49380).
    padded = jnp.pad(x, ((0, 0), (0, 0), (0, 1)))
    compact = jnp.take_along_axis(padded, source[:, None, :], axis=2, mode='promise_in_bounds')
    stream = jnp.concatenate([state, compact], axis=2)
    if segments is None:
        # The history in front carries the K-1 taps the first real token
        # reads, so column width + j of the convolution is the output of real
        # token j.
        convolved = causal_conv1d(stream, kernel, bias=bias)[..., width:]
    else:
        ordered = jnp.take_along_axis(segments, source, axis=1, mode='fill', fill_value=-1)
        held = jnp.broadcast_to(ordered[:, :1], (batch, width))
        convolved = document_conv1d(state, compact, held, ordered, kernel, bias)
    output = jnp.take_along_axis(convolved, jnp.maximum(rank, 0)[:, None, :], axis=2)
    # The K-1 stream columns from the row's token count on are the history it leaves.
    history = jnp.take_along_axis(
        stream, (rank[:, -1] + 1)[:, None, None] + jnp.arange(width)[None, None, :], axis=2)
    return jnp.where(valid[:, None, :], output, 0), history


def chunk_decay(g):
    """Cumulate per-chunk log decays and build the pairwise decay between positions.

    `g` is `[..., C, F]`: the C positions of a chunk and F decay channels.
    Returns the inclusive cumulative sum `gc` over C, `[..., C, F]`, and
    `decay[..., s, t, f] = exp(gc[s, f] - gc[t, f])` for s >= t, zero above
    the diagonal, `[..., C, C, F]`.
    """
    chunk_size = g.shape[-2]
    gc = jnp.cumsum(g, axis=-2)
    inclusive = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_))[..., None]
    diff = gc[..., :, None, :] - gc[..., None, :, :]
    # Masked before exp, as the references do. The unused positive
    # differences can overflow, and an outer where alone leaves 0 * inf in
    # the decay gradient.
    diff = jnp.where(inclusive, diff, 0.0)
    return gc, jnp.where(inclusive, jnp.exp(diff), 0.0)


def strictly_lower_inverse(a):
    """Return (I - A)^-1 = I + A + A^2 + ... for strictly lower triangular `a` `[..., C, C]`.

    The chunked delta rules' reference loop `attn[i, :i] += sum_k attn[i, k]
    attn[k, :i]`, iterated to the last row, computes exactly this for the
    nilpotent A (verified against the loop at C=4 and C=64: they agree to
    4e-15). The series is summed by doubling, S <- S + A^(2^k) S and
    A <- A^2, which is log2(C) matmuls instead of C row updates.
    """
    chunk_size = a.shape[-1]
    inv = jnp.broadcast_to(jnp.eye(chunk_size, dtype=a.dtype), a.shape)
    power = a
    for _ in range(max(1, math.ceil(math.log2(chunk_size)))):
        inv = inv + power @ inv
        power = power @ power
    return inv


def chunk_gated_delta_rule(query, key, value, g, beta, state=None,
                           chunk_size: int = CHUNK_SIZE):
    """The chunked form of the gated delta rule, the reference's math.

    Operands are [B, S, H, D] (the mixer's head layout); the computation
    matches `torch_chunk_gated_delta_rule` (modeling_qwen3_next.py:374-453)
    line for line, in fp32, and returns
    `(output [B, S, H, Dv], final_state [B, H, Dk, Dv])` in the input dtype.

    The reference's sequential correction (`for i in range(1, chunk_size)`)
    is the forward substitution that inverts `I - A` for a strictly lower
    triangular A, which `strictly_lower_inverse` sums as a series.
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
    q_c, k_c = chunks(query), chunks(key)
    kb_c, vb_c, g_c = chunks(k_beta), chunks(v_beta), chunks(g)

    # Cumulative log decay within each chunk, the reference's
    # `g = g.cumsum(dim=-1)` (modeling_qwen3_next.py:417), and the reference's
    # `((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()`, one decay
    # channel per head.
    gc, decay = chunk_decay(g_c[..., None])
    gc, decay = gc[..., 0], decay[..., 0]  # [NC, B, H, C], [NC, B, H, C, C]
    # The mask is strictly lower (tril, -1): the reference's masked_fill
    # zeroes the diagonal too.
    strict = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_), -1)
    attn = jnp.where(strict, -(kb_c @ jnp.swapaxes(k_c, -1, -2)) * decay, 0.0)
    # The reference's `attn + I` operator after its row correction loop.
    inv = strictly_lower_inverse(attn)
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
        v_corrected = v_i - v_prime
        attn_inter = (q_i * jnp.exp(gc_i)[..., None]) @ s
        out = attn_inter + attn_i @ v_corrected
        # The chunk's write into the memory, the reference's
        # `s * exp(gc[-1]) + (k * exp(gc[-1] - gc))^T @ v_new`
        # (modeling_qwen3_next.py:443-446).
        s = (s * jnp.exp(gc_i[..., -1])[..., None, None]
             + jnp.swapaxes(
                 k_i * jnp.exp(gc_i[..., -1][..., None] - gc_i)[..., None],
                 -1, -2) @ v_corrected)
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


# The projected width of the keys, values and the gate is the delta net's own
# ("linear"); the output projection is the width every mixer projects back
# into the model, which the table already calls "attention".
@logical_axes({
    ("in_proj_qkv",): ("embed", "linear"),
    ("in_proj_z",): ("embed", "linear"),
    ("in_proj_b",): ("embed", "kv"),
    ("in_proj_a",): ("embed", "kv"),
    ("out_proj",): ("attention", "embed"),
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
    open_kv_cache allocates its slots. Prefill and decode share one code
    path, and the conv state crosses the boundary between them because a
    continuation must see the last K-1 real columns, not the zeros a fresh
    sequence pads with.

    Parameter names are the checkpoint's, so a translation only moves
    weights: `conv1d/weight` is the depthwise taps `[D, 1, K]`, and
    `A_log`/`dt_bias` are the `[Hv]` leaves the reference materialises as
    parameters. With `fused_in_proj` the input projections are the two
    leaves Qwen3-Next stores, `in_proj_qkvz` and `in_proj_ba`
    (`Qwen3NextGatedDeltaNet.__init__`, modeling_qwen3_next.py:540-543);
    `fix_query_key_value_ordering` (modeling_qwen3_next.py:558-586) splits
    each row group of one key head into its q, k and the value heads' v and
    z (b and a), which `_project` mirrors.
    """

    emb_features: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int = 4
    chunk_size: int = CHUNK_SIZE
    norm_eps: float = 1e-6
    gate_activation: str = 'silu'  # the norm's gate: 'silu' | 'sigmoid', qwen4_exp's output_gate_type
    fused_in_proj: bool = False
    dtype: Dtype | None = None
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
        if self.fused_in_proj:
            self.in_proj_qkvz = dense(2 * self.key_features + 2 * self.value_features,
                                      name='in_proj_qkvz')
            self.in_proj_ba = dense(2 * self.num_v_heads, name='in_proj_ba')
        else:
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

    def _project(self, x):
        """(query, key, value, z, b, a) of `x`, flat over heads: `[B, S, H*D]`
        for the four wide ones and `[B, S, Hv]` for b and a."""
        B, S, _ = x.shape
        if not self.fused_in_proj:
            key_dim = self.key_features
            query, key, value = jnp.split(self.in_proj_qkv(x), [key_dim, 2 * key_dim], axis=-1)
            return query, key, value, self.in_proj_z(x), self.in_proj_b(x), self.in_proj_a(x)
        # One row group per key head: its q, its k, then the v and z of the
        # value heads it serves (modeling_qwen3_next.py:558-586).
        per_key = self.per_v * self.head_v_dim
        qkvz = self.in_proj_qkvz(x).reshape(B, S, self.num_k_heads, 2 * self.head_k_dim + 2 * per_key)
        query, key, value, z = jnp.split(
            qkvz, [self.head_k_dim, 2 * self.head_k_dim, 2 * self.head_k_dim + per_key], axis=-1)
        ba = self.in_proj_ba(x).reshape(B, S, self.num_k_heads, 2 * self.per_v)
        b, a = jnp.split(ba, 2, axis=-1)
        return (query.reshape(B, S, -1), key.reshape(B, S, -1), value.reshape(B, S, -1),
                z.reshape(B, S, -1), b.reshape(B, S, -1), a.reshape(B, S, -1))

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None):
        del positions, segment_ids, kv_store
        B, S, _ = x.shape
        valid = None if attention_metadata is None else attention_metadata.valid
        if valid is not None and valid.shape != (B, S):
            raise ValueError(f"row validity must be {(B, S)}, got {valid.shape}")
        query, key, value, z, b, a = self._project(x)
        key_dim = self.key_features

        # fp32 on purpose, as the reference notes: an fp16 A can make exp
        # underflow to -inf (modeling_qwen3_next.py:652-653).
        conv_input = jnp.moveaxis(
            jnp.concatenate([query, key, value], axis=-1).astype(jnp.float32),
            2, 1)  # [B, D, S], the conv's channel-major layout
        taps, _ = self.conv1d()
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
                out = jnp.zeros((B, S, self.value_features), self.dtype)
                return self.out_proj(out)
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
        if valid is not None:
            # exp(0)=1 and beta=0 preserve recurrent memory for invalid input.
            beta = jnp.where(valid[:, :, None], beta, 0.0)
            g = jnp.where(valid[:, :, None], g, 0.0)

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

    The checkpoint stores `conv1d.weight` this way, and flax's Conv matches
    neither its [K, D, 1] kernel order nor the reference's channel-major
    [B, D, S] input. The leaf keeps the checkpoint's name, `weight`, because
    a translation transposes a `kernel` as a Linear's [out, in]. The taps
    have no matrix axis worth a name, so they take the shape heuristic.
    `use_bias` adds the checkpoint's `conv1d.bias` [D], as Mamba 2's
    `nn.Conv1d(groups=conv_dim)` has one (modeling_mamba2.py:392-399).

    Returns the [D, K] taps and the bias (None without one), both fp32, the
    dtype the convolution runs in.
    """

    features: int
    kernel: int = 4
    use_bias: bool = False
    init_std: float | None = None  # None: lecun normal

    @nn.compact
    def __call__(self) -> tuple[jax.Array, jax.Array | None]:
        weight = self.param('weight', normal_kernel(self.init_std, nn.initializers.lecun_normal())[
            'kernel_init'], (self.features, 1, self.kernel))
        bias = (self.param('bias', nn.initializers.zeros, (self.features,), jnp.float32)
                if self.use_bias else None)
        return (jnp.asarray(weight[:, 0, :], jnp.float32),
                None if bias is None else jnp.asarray(bias, jnp.float32))


class RMSNormGated(nn.Module):
    """The reference's Qwen3NextRMSNormGated: RMSNorm in fp32, then the
    gate, then the cast back (modeling_qwen3_next.py:57-74). The gate is
    silu in the qwen3_5/qwen3_next references and sigmoid where qwen4_exp's
    output_gate_type says so; the caller picks, since the reference makes
    the activation a config field.
    """

    epsilon: float = 1e-6
    activation: str = 'silu'
    dtype: Dtype | None = None

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
