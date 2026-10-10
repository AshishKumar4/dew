"""Gated DeltaNet: the linear-attention mixer of the Qwen3.5 family.

The delta rule keeps an outer-product memory `S = sum_t k_t v_t^T` and
corrects it toward the value each new key predicts. A decay `g` shrinks the
memory before each write and a beta scales how far one write moves it.

The chunked form is transformers 5.16.1's (`modeling_qwen3_next.py:374-453`,
identical in qwen3_5 and qwen4_exp): per chunk of 64 tokens `(I+A)^-1 @
v_beta` resolves the delta rule's dependencies in matrix form and the memory
crosses chunks through `k_cumdecay @ S`, so the largest new tensor is
`[C, C]` per head per chunk. Decoding runs the recurrent form
(`modeling_qwen3_next.py:456-506`):

    S <- S * exp(g_t)
    delta <- (v_t - S k_t) * beta_t
    S <- S + k_t delta^T
    o_t <- S q_t

tests/test_linear_attention.py holds the two forms to the same numbers. The
short mixer is the reference's depthwise causal conv1d on the projected qkv
(`modeling_qwen3_next.py:325-365`): kernel 4, left zero padding, silu after,
the last `kernel - 1` columns kept as decode state (`update_conv_state`).
`g = -exp(A_log) * softplus(a + dt_bias)` reads `A_log` and `dt_bias` per
value head.
"""

import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.kernels import delta_chunks, delta_rule
from dew.nn.scatter import DROPPED

from .attention import FORWARD_MODE, unweighted_rmsnorm
from .blocks import normal_kernel
from .inputs import AttentionMetadata
from .kernels.generation import first_refusal, measured_kernel, ran_kernel, triton_runs
from .precision import asks_default_precision, at_least_fp32
from .sharding import logical_axes, rows_like

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
    (`F.conv1d(padding=kernel_size - 1)`, modeling_qwen3_next.py:345-365) and
    applies silu; `bias` is the `[D]` per-channel bias (Mamba-2's
    `use_conv_bias`). The taps apply as K shifted products summed in fp32, not
    a grouped `conv_general_dilated`, whose filter gradient XLA's SPMD
    partitioner sums over every device, replicas included, doubling it under
    fsdp beside an unused tensor axis (jax 0.11.2,
    https://github.com/openxla/xla/issues/49382).
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


def held_conv1d(x, kernel, valid, held=None, *, bias=None, segments=None):
    """`[B, D, S]` convolved causally after the `held` history, `[B, D, K - 1]`
    or None for none, and the history the next call holds: a linear-attention
    or state-space layer's conv over its decode cache, padded slots read and
    advanced by none (`_masked_conv1d`). `bias` is added before the silu, and
    packed `segments` convolve each document alone."""
    if valid is not None or segments is not None:
        valid = jnp.ones(x.shape[::2], bool) if valid is None else valid
        return _masked_conv1d(x, kernel, valid, held, bias=bias, segments=segments)
    history = x if held is None else jnp.concatenate([held, x], axis=2)
    return (causal_conv1d(history, kernel, bias=bias)[..., -x.shape[2]:],
            history[:, :, -(kernel.shape[-1] - 1):])


def _masked_conv1d(x, kernel, valid, state=None, bias=None, segments=None):
    """Convolve real tokens without advancing a paused row's history.

    The j-th real token of a row reads the K-1 real tokens before it, whatever
    padding sits between, and an invalid slot neither reads nor advances the
    history. `cumsum(valid) - 1` indexes each real token in the row's stream,
    so one convolution over the gathered stream replaces S sequential windows.
    `segments` `[B, S]` makes each packed document convolve alone
    (`document_conv1d`); the held `state` belongs to the first real token's
    document. Masking an ordinary convolution's input instead reads the zeros in
    a gap (5.0 off in fp32 on nine slots with holes at 2, 5 and 6,
    test_the_masked_conv_reads_across_a_gap).
    """
    batch, channels, _ = x.shape
    width = kernel.shape[1] - 1
    if state is None:
        state = rows_like(jnp.zeros((batch, channels, width), x.dtype), x)
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


def _compensated_add(left, right):
    """Add two double-float values, `(hi, lo)` pairs whose sum is the value:
    `hi` is the rounded sum and `lo` carries what its rounding dropped
    (Knuth's TwoSum), so a running sum keeps fp32's precision squared.
    XLA reassociates no float arithmetic unless fast math is on; with it,
    `lo` cancels to zero and the sum is the plain one."""
    (left_hi, left_lo), (right_hi, right_lo) = left, right
    total = left_hi + right_hi
    part = total - left_hi
    dropped = (left_hi - (total - part)) + (right_hi - part)
    return total, dropped + left_lo + right_lo


def chunk_decay(g):
    """Cumulate per-chunk log decays and build the pairwise decay between positions.

    `g` is `[..., C, F]`: the C positions of a chunk and F decay channels.
    Returns the inclusive cumulative sum `gc` over C, `[..., C, F]`, and
    `decay[..., s, t, f] = exp(gc[s, f] - gc[t, f])` for s >= t, zero above
    the diagonal, `[..., C, C, F]`.

    The references subtract two fp32 cumulative sums, so every exponent
    carries the rounding of the chunk's whole accumulated magnitude, which
    exp turns into a relative error on every decay. Here the cumulative sum
    is compensated (`_compensated_add`) and the difference takes both
    halves, so an exponent is as precise as its own range. On
    tests/fixtures/hf/kimi-linear-tiny that error dominated the gradient:
    over 16 orderings of the residual stream (an exact symmetry, so only the
    rounding moves) the updated logits sat at an RMS 2.04 times the
    reference's distance from float64 with plain sums on CPU and 1.83 on an
    RTX 4080, and 0.92 and 0.72 with these, against the reference's own
    0.98. Summing each pair's range instead (Mamba-2's `segment_sum`) is as
    exact and made a KDA layer's forward and backward 25% slower; this
    costs about 1% (on an RTX 4080, Kimi Linear's KDA layer over 1024 tokens
    47.6 against 48.0 ms, Qwen3.5-0.8B's gated delta net over 4096 16.3
    against 16.5).
    """
    chunk_size = g.shape[-2]
    hi, lo = jax.lax.associative_scan(_compensated_add, (g, jnp.zeros_like(g)), axis=g.ndim - 2)
    inclusive = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_))[..., None]
    diff = ((hi[..., :, None, :] - hi[..., None, :, :]) + (lo[..., :, None, :] - lo[..., None, :, :]))
    # Masked before exp, as the references do. The unused positive
    # differences can overflow, and an outer where alone leaves 0 * inf in
    # the decay gradient.
    diff = jnp.where(inclusive, diff, 0.0)
    return hi, jnp.where(inclusive, jnp.exp(diff), 0.0)


def strictly_lower_inverse(a):
    """Return (I - A)^-1 for strictly lower triangular `a` `[..., C, C]`, the
    chunked delta rules' `attn[i, :i] += sum_k attn[i, k] attn[k, :i]` loop.

    Built from its diagonal blocks, doubling their size: with X1 and X2 the
    inverses of two neighbouring blocks, their union's is X1 and X2 on the
    diagonal and X2 A21 X1 below. Each block is a diagonal block of the
    whole inverse, so every product stays the size of its entries. The
    series I + A + A^2 + ..., summed by doubling powers of A, did not: on
    Qwen3.5-0.8B's aligned keys the powers grew as binomials (1.8e17 at C=64)
    and cancelled to an inverse whose entries are at most 1, 1e24 off in
    fp32. The first level is written out rather than as products with
    one-wide identity blocks: XLA TPU rewrites a matmul by a broadcast
    identity into a dilated convolution whose fusion fails register
    allocation on a v6e (`live_range_finder.cc:57 RET_CHECK`, jax 0.11.2).
    """
    chunk_size = a.shape[-1]
    if chunk_size == 1:
        return jnp.ones_like(a)
    width = 1 << (chunk_size - 1).bit_length()
    if width != chunk_size:  # zero rows and columns extend it by the identity
        pad = [(0, 0)] * (a.ndim - 2) + [(0, width - chunk_size)] * 2
        return strictly_lower_inverse(jnp.pad(a, pad))[..., :chunk_size, :chunk_size]
    lead = a.shape[:-2]

    def diagonal_blocks(size):
        """`a`'s `[..., width // size, size, size]` blocks on the diagonal."""
        count = width // size
        blocked = a.reshape(*lead, count, size, count, size)
        return jnp.moveaxis(jnp.diagonal(blocked, axis1=-4, axis2=-2), -1, -3)

    # Two-wide blocks: [[1, 0], [a, 1]].
    below = diagonal_blocks(2)[..., 1, 0]
    one, zero = jnp.ones_like(below), jnp.zeros_like(below)
    inverse = jnp.stack([jnp.stack([one, zero], -1), jnp.stack([below, one], -1)], -2)
    size = 2
    while size < width:
        pairs = inverse.reshape(*lead, width // (2 * size), 2, size, size)
        first, second = pairs[..., 0, :, :], pairs[..., 1, :, :]
        joined = second @ diagonal_blocks(2 * size)[..., size:, :size] @ first
        top = jnp.concatenate([first, jnp.zeros_like(first)], -1)
        inverse = jnp.concatenate([top, jnp.concatenate([joined, second], -1)], -2)
        size *= 2
    return inverse.reshape(*lead, width, width)


def xla_chunk_states(key, k_cumdecay, out_vals, gc, state):
    """The memory crossing chunks, the one sequential part of the chunked
    rule, as a `lax.scan`: per chunk the reference's corrected values
    `v_new = value - k_cumdecay @ S` and its write `S * exp(gc[-1]) +
    (k * exp(gc[-1] - gc))^T @ v_new` (modeling_qwen3_next.py:439-446).

    Operands are per chunk in the reference's layout, `[B, H, NC, C, ...]`,
    `gc` `[B, H, NC, C]` being the log decay cumulated within each chunk,
    and `state` `[B, H, Dk, Dv]` enters the first chunk. Returns the state
    each chunk entered `[B, H, NC, Dk, Dv]`, each chunk's corrected values
    `[B, H, NC, C, Dv]` and the state leaving the last chunk.
    """
    def one_chunk(s, step):
        k_i, kd_i, v_i, gc_i = step
        v_corrected = v_i - kd_i @ s
        last = gc_i[..., -1]
        written = k_i * jnp.exp(last[..., None] - gc_i)[..., None]
        return (s * jnp.exp(last)[..., None, None] + jnp.swapaxes(written, -1, -2) @ v_corrected,
                (s, v_corrected))
    final, (entered, corrected) = jax.lax.scan(
        one_chunk, state, tuple(jnp.moveaxis(x, 2, 0) for x in (key, k_cumdecay, out_vals, gc)))
    return jnp.moveaxis(entered, 0, 2), jnp.moveaxis(corrected, 0, 2), final


GATED_DELTA_RULES = ('auto', 'xla', 'pallas')
"""The recurrences `chunk_gated_delta_rule` runs its memory across chunks on."""


def chunk_states_kernel(implementation: str, key, out_vals) -> str:
    """The recurrence a chunked rule over these per-chunk keys and corrected
    values runs: 'xla' (`xla_chunk_states`) or 'pallas'
    (`dew.nn.kernels.delta_chunks`).

    'auto' takes the generation's measured one (`KERNELS['gated_delta_rule']`)
    and 'xla' elsewhere. 'pallas' needs a GPU the kernels compile for,
    operands they take (`delta_chunks.refusal`) and reverse mode, since under
    `forward_mode_attention`, which traces for `jax.jvp`, the XLA scan gives
    the tangent. A call it turns down runs 'xla', which `ran_kernel` logs
    where 'pallas' was measured fastest. An explicit 'pallas' on a host
    without a GPU interprets the kernels, as the CPU suite runs them.
    """
    if implementation not in GATED_DELTA_RULES:
        raise ValueError(f"implementation must be one of {list(GATED_DELTA_RULES)}, got {implementation!r}")
    chosen = measured_kernel('gated_delta_rule', 'xla') if implementation == 'auto' else implementation
    if chosen != 'pallas':
        return chosen
    placed = triton_runs() or (implementation == 'pallas' and jax.default_backend() == 'cpu')
    refusal = first_refusal(
        (placed, "the Pallas kernels need a GPU of sm80 or later"),
        (not FORWARD_MODE.get(), "forward_mode_attention traces for jax.jvp, which the kernels lack"),
    ) or delta_chunks.refusal(key, out_vals)
    return 'pallas' if refusal is None else ran_kernel('gated_delta_rule', 'xla', refusal)


def chunk_gated_delta_rule(query, key, value, g, beta, state=None,
                           chunk_size: int = CHUNK_SIZE, implementation: str = 'auto'):
    """The chunked form of the gated delta rule, `torch_chunk_gated_delta_rule`
    (modeling_qwen3_next.py:374-453) in fp32, over [B, S, H, D] operands;
    returns `(output [B, S, H, Dv], final_state [B, H, Dk, Dv])` in the input
    dtype. The reference's sequential correction is the inverse
    `strictly_lower_inverse` builds from its diagonal blocks.

    Only the memory crosses chunks, so only its recurrence runs chunk after
    chunk, on the recurrence `implementation` names (`chunk_states_kernel`).
    Each chunk's output, the reference's `attn_inter + attn @ v_new`, reads
    the state that chunk entered and its corrected values, so it is computed
    for every chunk at once after the recurrence. Those are the same sums in
    the same order, and the scan no longer launches the three products of
    the output at every chunk.
    """
    dtype, work = query.dtype, at_least_fp32(query.dtype)
    query, key, value, g, beta = (
        x.astype(work) for x in (query, key, value, g, beta))
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

    def chunks(x):  # [B, S, H, ...] -> [B, H, NC, C, ...], the reference's layout
        moved = jnp.moveaxis(x, 2, 1)  # [B, H, S, ...]
        return moved.reshape(B, H, T // chunk_size, chunk_size, *moved.shape[3:])

    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]
    q_c, k_c = chunks(query), chunks(key)
    kb_c, vb_c, g_c = chunks(k_beta), chunks(v_beta), chunks(g)

    # Cumulative log decay within each chunk, the reference's
    # `g = g.cumsum(dim=-1)` (modeling_qwen3_next.py:417), and the reference's
    # `((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()`, one decay
    # channel per head.
    gc, decay = chunk_decay(g_c[..., None])
    gc, decay = gc[..., 0], decay[..., 0]  # [B, H, NC, C], [B, H, NC, C, C]
    # The mask is strictly lower (tril, -1): the reference's masked_fill
    # zeroes the diagonal too.
    strict = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_), -1)
    attn = jnp.where(strict, -(kb_c @ jnp.swapaxes(k_c, -1, -2)) * decay, 0.0)
    # The reference's `attn + I` operator after its row correction loop.
    inv = strictly_lower_inverse(attn)
    out_vals = inv @ vb_c  # the reference's `value = attn @ v_beta`
    k_cumdecay = inv @ (kb_c * jnp.exp(gc)[..., None])

    state = (jnp.zeros((B, H, Dk, Dv), work) if state is None
             else state.astype(work))
    if chunk_states_kernel(implementation, k_c, out_vals) == 'pallas':
        entered, v_corrected, state = delta_chunks.chunk_states(
            k_c, k_cumdecay, out_vals, gc, state, not triton_runs())
    else:
        entered, v_corrected, state = xla_chunk_states(k_c, k_cumdecay, out_vals, gc, state)
    attn = q_c @ jnp.swapaxes(k_c, -1, -2) * decay
    attn_inter = (q_c * jnp.exp(gc)[..., None]) @ entered
    core = attn_inter + attn @ v_corrected
    # core: [B, H, NC, C, Dv] -> [B, H, NC*C, Dv] -> [B, S, H, Dv]
    core = jnp.moveaxis(core.reshape(B, H, T, Dv), 1, 2)[:, :S]
    return core.astype(dtype), state.astype(dtype)


def recurrent_delta_rule(query, key, value, g, beta, state=None):
    """One token at a time, the decode path, in fp32 as a lax.scan over the
    time axis so the state rides the scan the way it rides the decode cache.

    `g` is the log decay per key dimension, `[B, S, H, Dk]`, which is
    `recurrent_kimi_delta_attention` (modeling_glm5_next.py:428-478). The
    gated delta rule decays per head, one decay over every key dimension.
    """
    dtype, work = query.dtype, at_least_fp32(query.dtype)
    query, key, value, g, beta = (
        x.astype(work) for x in (query, key, value, g, beta))
    query = query * (key.shape[-1] ** -0.5)

    def one_token(s, step):
        q_t, k_t, v_t, g_t, beta_t = (step[name] for name in ('q', 'k', 'v', 'g', 'beta'))
        s = s * jnp.exp(g_t)[..., :, None]                 # rows decay per key dimension
        kv_mem = jnp.sum(s * k_t[..., :, None], axis=-2)   # [B, H, Dv]
        delta = (v_t - kv_mem) * beta_t[..., None]        # [B, H, Dv]
        s = s + k_t[..., :, None] * delta[..., None, :]    # [B, H, Dk, Dv]
        return s, jnp.sum(s * q_t[..., :, None], axis=-2)  # [B, H, Dv]
    # The scan stacks along the first axis, so the operands go time-major.
    if state is None:
        state = jnp.zeros((query.shape[0], query.shape[-2], key.shape[-1],
                           value.shape[-1]), work)
    state, out = jax.lax.scan(
        one_token, state.astype(work),
        {name: jnp.moveaxis(x, 1, 0) for name, x in
         (('q', query), ('k', key), ('v', value), ('g', g), ('beta', beta))})
    return jnp.moveaxis(out, 0, 1).astype(dtype), state.astype(dtype)


def decode_gated_delta_rule(query, key, value, g, beta, state, active=None):
    """`recurrent_gated_delta_rule` for one decode token `[B, 1, H, D]`
    through `dew.nn.kernels.delta_rule.step`, which reads and writes the
    state once (CUDA, where `delta_rule.fits` the state). A row `active`
    `[B]` marks False keeps its state untouched and outputs zeros."""
    work = jnp.float32
    query, key, value = (x[:, 0].astype(work) for x in (query, key, value))
    decay = jnp.broadcast_to(jnp.exp(g[:, 0].astype(work))[..., None], key.shape)
    final, out = delta_rule.step(state, query * key.shape[-1] ** -0.5, key, value, decay,
                                 beta[:, 0].astype(work), active)
    return out[:, None], final


def recurrent_gated_delta_rule(query, key, value, g, beta, state=None):
    """`torch_recurrent_gated_delta_rule` (modeling_qwen3_next.py:456-506):
    the delta rule with one log decay per head, `g` `[B, S, H]`."""
    return recurrent_delta_rule(query, key, value, g[..., None], beta, state)


# The projected width of the keys, values and the gate is the delta net's own
# ("linear"); the output projection is the width every mixer projects back
# into the model, which the table already calls "attention".
@logical_axes({
    ("in_proj_qkv",): ("embed", "linear"),
    ("in_proj_z",): ("embed", "linear"),
    ("in_proj_b",): ("embed", "kv"),
    ("in_proj_a",): ("embed", "kv"),
    ("out_proj",): ("attention", "embed"),
})
class GatedDeltaNet(nn.Module):
    """The token mixer of a linear_attention layer.

    Projects into q, k, v, z, b and a, runs the causal conv over qkv, applies
    the gated delta rule (chunked in training, recurrent when decoding) and
    gates the output with the reference's RMSNormGated: norm, then silu(z), or
    sigmoid(z) where qwen4_exp's output_gate_type says. `num_v_heads //
    num_k_heads` value heads share a key head (`repeat_interleave`).

    The decode state is `recurrent_state` [B, H, Dk, Dv] and `conv_state`
    [B, D, K-1] in the `cache` collection. A bfloat16 model holds its
    recurrent state in bfloat16, as vLLM does, rounded stochastically after
    each decode token (`delta_rule.round_to_bf16`): rounding to nearest
    holds a slowly decaying state where it is, and its error grew tenfold
    over 1024 tokens. Any other model holds fp32, as does one asked for more
    than the default matmul precision (`matmul_precision` "highest", as its
    vocabulary head stays fp32), and the rule always runs in fp32
    (docs/performance.md, "The hybrid's bf16 state"). The conv state
    crosses from prefill to decode so a continuation sees the last K-1 real
    columns. Parameter names are the checkpoint's (`conv1d/weight` the `[D, 1, K]` taps, `A_log` and
    `dt_bias` `[Hv]`). `fused_in_proj` keeps Qwen3-Next's two fused leaves,
    `in_proj_qkvz` and `in_proj_ba`, split per key-head row group as
    `fix_query_key_value_ordering` does (modeling_qwen3_next.py:540-586).
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
        # underflow to -inf (modeling_qwen3_next.py:652-653); at least fp32,
        # so a float64 model's stays float64 (`at_least_fp32`).
        wide = at_least_fp32(query.dtype)
        conv_input = jnp.moveaxis(
            jnp.concatenate([query, key, value], axis=-1).astype(wide),
            2, 1)  # [B, D, S], the conv's channel-major layout
        taps, _ = self.conv1d()
        recurrent = None
        if decode:
            # The first decode-mode call only allocates, the way
            # open_kv_cache's is: init_cache's dummy token must not consume
            # a position or leave state behind. The conv state holds the
            # projections' own values, so it is held in their dtype at no
            # cost; the recurrent state is held as the docstring says.
            allocated = self.has_variable('cache', 'recurrent_state')
            held = (jnp.promote_types(self.dtype, jnp.bfloat16)
                    if self.dtype is not None and asks_default_precision(self.precision, configured=True)
                    else wide)
            conv_state = self.variable(
                'cache', 'conv_state', jnp.zeros,
                (B, self.conv_features, self.conv_kernel - 1), jnp.promote_types(query.dtype, held))
            recurrent = self.variable(
                'cache', 'recurrent_state', jnp.zeros,
                (B, self.num_v_heads, self.head_k_dim, self.head_v_dim), held)
            if not allocated:
                # Allocation only: the caller's first real forward, not this
                # call, starts the state.
                out = jnp.zeros((B, S, self.value_features), self.dtype)
                return self.out_proj(out)
            mixed, history = held_conv1d(conv_input, taps, valid, conv_state.value.astype(wide))
            conv_state.value = history.astype(conv_state.value.dtype)
        else:
            mixed, _ = held_conv1d(conv_input, taps, valid)
        mixed = jnp.moveaxis(mixed, 2, 1)  # back to [B, S, D]

        query, key, value = jnp.split(mixed, [key_dim, 2 * key_dim], axis=-1)
        query = self._expand_kv(self._split_heads(query, self.head_k_dim))
        key = self._expand_kv(self._split_heads(key, self.head_k_dim))
        value = self._split_heads(value, self.head_v_dim)

        beta = nn.sigmoid(b.astype(wide))
        g = -jnp.exp(self.A_log.astype(wide)) * nn.softplus(
            a.astype(wide) + self.dt_bias.astype(wide))

        query = l2norm(query)
        key = l2norm(key)
        if valid is not None:
            # exp(0)=1 and beta=0 preserve recurrent memory for invalid input.
            beta = jnp.where(valid[:, :, None], beta, 0.0)
            g = jnp.where(valid[:, :, None], g, 0.0)

        held = None if recurrent is None else recurrent.value
        state = None if held is None else held.astype(wide)
        if S == 1 and held is not None and delta_rule.fits(held):
            out, final = decode_gated_delta_rule(query, key, value, g, beta, held,
                                                 None if valid is None else valid[:, 0])
        elif S == 1:
            out, final = recurrent_gated_delta_rule(query, key, value, g, beta, state)
            if held is not None and held.dtype == jnp.bfloat16:
                final = delta_rule.round_to_bf16(final, key[:, 0].astype(wide), value[:, 0].astype(wide))
        else:
            out, final = chunk_gated_delta_rule(query, key, value, g, beta, state, self.chunk_size)
        if recurrent is not None:
            recurrent.value = final.astype(recurrent.value.dtype)

        gate = z.reshape(B, S, -1, self.head_v_dim)
        out = self.norm(out, gate)  # the norm scales, then the activation gates
        out = out.reshape(B, S, self.value_features)
        return self.out_proj(out)


class DepthwiseConv1d(nn.Module):
    """The conv's taps as a raw parameter, in the checkpoint's [D, 1, K] layout.

    flax's Conv matches neither the checkpoint's taps nor the reference's
    channel-major input, and the leaf is `weight` so a translation does not
    transpose it as a Linear's kernel. `use_bias` adds `conv1d.bias` [D], as
    Mamba 2's depthwise conv has (modeling_mamba2.py:392-399). Returns the
    [D, K] taps and the bias (or None), fp32.
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
        wide = at_least_fp32(weight.dtype)
        return (jnp.asarray(weight[:, 0, :], wide),
                None if bias is None else jnp.asarray(bias, wide))


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
        wide = at_least_fp32(x.dtype)
        y = unweighted_rmsnorm(x.astype(wide), self.epsilon)
        scale = self.param('weight', nn.initializers.ones,
                           (x.shape[-1],), jnp.float32)
        y = y * scale.astype(wide)
        gate = gate.astype(wide)
        gated = (nn.silu(gate) if self.activation == 'silu'
                 else nn.sigmoid(gate))
        return (y * gated).astype(dtype)
