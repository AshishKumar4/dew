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
The conv runs over Dew's activation layout, time before channels, and so
does the history it caches, `[B, K-1, D]`.
`g = -exp(A_log) * softplus(a + dt_bias)` reads `A_log` and `dt_bias` per
value head.
"""

import functools
import math
from collections.abc import Sequence
from typing import overload

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.interpreters.ad import JVPTracer

from dew.nn.kernels import delta_rule, fla_gdn, short_conv

from .attention import FORWARD_MODE, gated_rms_normalized
from .blocks import normal_kernel
from .flops import conv_flops, delta_flops, projection_flops
from .inputs import AttentionMetadata, valid_order
from .kernels import generation
from .kernels.generation import first_refusal, ran_kernel, triton_runs
from .precision import asks_default_precision, at_least_fp32, compensated_add
from .sharding import (HEADS, RESIDUAL, constrain, logical_axes, logical_spec, manual_map, mesh_axes, rows_like,
                       seen_whole)

CHUNK_SIZE = 64
"""Tokens per chunk of the chunked form, the reference's default
(`torch_chunk_gated_delta_rule(..., chunk_size=64)`). The chunked and the
recurrent form agree at any size; the size only trades memory for
parallelism, and 64 is what the reference's kernels assume."""


L2NORM_EPS = 1e-6
"""`l2norm`'s epsilon under the root, the FLA library's."""


def l2norm(x, eps: float = L2NORM_EPS):
    """The FLA library's normalisation of q and k, which the reference
    applies inside the rule (`l2norm`, modeling_qwen3_next.py:368-371):
    unit keys and queries per head, so `q k^T` is a cosine and the
    `1/sqrt(d)` scale keeps it bounded."""
    inv = jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * inv


Pieces = Sequence[tuple[int, int | None]]
"""`causal_conv1d`'s `parts`: each piece's width and the head its l2 norm runs over, or None."""


@overload
def causal_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, parts: None = None,
                  dtype=None) -> jax.Array: ...


@overload
def causal_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, parts: Pieces,
                  dtype=None) -> tuple[jax.Array, ...]: ...


def causal_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, parts: Pieces | None = None,
                  dtype=None) -> jax.Array | tuple[jax.Array, ...]:
    """Depthwise causal conv over `[B, S, D]` with the `[D, K]` taps.

    The reference pads `K - 1` zeros to the left
    (`F.conv1d(padding=kernel_size - 1)`, modeling_qwen3_next.py:345-365) and
    applies silu; `bias` is the `[D]` per-channel bias (Mamba-2's
    `use_conv_bias`). The taps apply as K shifted products summed in fp32, not
    a grouped `conv_general_dilated`, whose filter gradient XLA's SPMD
    partitioner sums over every device, replicas included, doubling it under
    fsdp beside an unused tensor axis (jax 0.11.2,
    https://github.com/openxla/xla/issues/49382). Time-major, as the
    projections give it and the scans read it: the reference's channel-major
    `[B, D, S]` cost each mixer a transpose of its widest activation in and
    out, forward and backward.

    The sum, bias and silu run in at least fp32 and round once, to `dtype`
    (x's where None), so a mixer hands over its projection as it comes and
    asks for the width its scan runs in. `valid` `[B, S]` zeroes the outputs
    of padding slots. `parts` `((width, head), ...)` returns the channels as
    those consecutive pieces, a piece with a `head` l2-normalised over each
    `head` channels (`l2norm`), the queries and keys of a delta rule: each
    piece is the array its scan reads, not a slice of one to copy out.

    On a GPU that runs Dew's Triton kernels all of it is one kernel forward
    and one backward (`short_conv`), which save only `x`; elsewhere, and for
    a conv they turn down (`conv_refusal`), `xla_conv1d`.
    """
    dtype = x.dtype if dtype is None else jnp.dtype(dtype)
    pieces = ((x.shape[-1], None),) if parts is None else tuple(parts)
    if conv_refusal(x, kernel, pieces, dtype) is None:
        out = short_conv.conv(x, kernel, bias, None if valid is None else jnp.asarray(valid, jnp.int8),
                              pieces, activation=activation, eps=L2NORM_EPS, dtype=dtype,
                              interpret=not generation.triton_runs())
        return out[0] if parts is None else out
    return xla_conv1d(x, kernel, activation, bias, valid=valid, parts=parts, dtype=dtype)


def xla_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, parts: Pieces | None = None,
               dtype=None) -> jax.Array | tuple[jax.Array, ...]:
    """`causal_conv1d`'s arithmetic in XLA: a CPU's and a TPU's, and the
    oracle of the GPU's kernels."""
    dtype = x.dtype if dtype is None else jnp.dtype(dtype)
    _, K = kernel.shape
    length = x.shape[1]
    wide = jnp.promote_types(at_least_fp32(x.dtype), dtype)
    padded = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0))).astype(wide)
    taps = kernel.astype(wide)
    windows = padded[:, :length] * taps[:, 0]
    for tap in range(1, K):
        windows = windows + padded[:, tap:tap + length] * taps[:, tap]
    if bias is not None:
        windows = windows + bias.astype(wide)
    if activation:
        windows = nn.silu(windows)
    if valid is not None:
        windows = jnp.where(valid[:, :, None], windows, 0)
    return conv_pieces(windows, parts, dtype)


def conv_pieces(convolved, parts, dtype):
    """`convolved` `[B, S, D]` as `causal_conv1d` returns it for `parts`
    (the whole, where None), in `dtype`: each piece split off in order and
    the `head`-wide groups of a normalised one l2-normalised."""
    if parts is None:
        return convolved.astype(dtype)
    pieces, first = [], 0
    for width, head in parts:
        piece = convolved[..., first:first + width]
        first += width
        if head is not None:
            piece = l2norm(piece.reshape(*piece.shape[:-1], -1, head)).reshape(piece.shape)
        pieces.append(piece.astype(dtype))
    return tuple(pieces)


def conv_refusal(x, kernel, parts, dtype) -> str | None:
    """Why `short_conv`'s kernels do not take `causal_conv1d` of `x` by
    `kernel` into `parts` in `dtype` here, or None where they do: a GPU they
    compile for, reverse mode (a caller differentiating forward sets
    `FORWARD_MODE`), an operand no mesh axis splits, and the shapes and
    dtypes they take (`short_conv.refusal`)."""
    return first_refusal(
        (triton_runs(), "the kernels need a GPU of sm80 or later"),
        (not FORWARD_MODE.get() and not isinstance(x, JVPTracer),
         "a JVP trace needs forward-mode derivatives; the kernels implement a VJP"),
        (seen_whole(x), "a mesh axis splits the conv's operands, and the kernels see whole arrays"),
    ) or short_conv.refusal(x, kernel, parts, dtype)


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
    """The causal depthwise conv of `causal_conv1d` over `x` `[B, S, D]`
    behind `history` `[B, K-1, D]`, where a tap reads a token only if it
    shares the output token's segment: each packed document convolves as if
    zeros preceded it, as mamba_ssm's `causal_conv1d_fn(seq_idx=...)` does.
    `taps` `[D, K]`, `history_segments` `[B, K-1]`, `segments` `[B, S]`;
    silu after the bias, in at least fp32."""
    width = taps.shape[1] - 1
    length = x.shape[1]
    stream = jnp.concatenate([history, x], axis=1)
    stream = stream.astype(at_least_fp32(stream.dtype))
    stream_segments = jnp.concatenate([history_segments, segments], axis=1)
    out = (jnp.zeros(x.shape, stream.dtype) if bias is None
           else jnp.broadcast_to(bias.astype(stream.dtype), x.shape))
    for tap in range(width + 1):
        same = stream_segments[:, tap:tap + length] == segments
        out = out + taps[:, tap] * jnp.where(same[..., None], stream[:, tap:tap + length], 0)
    return nn.silu(out)


def _stream_order(valid):
    """Each slot's index among its row's real tokens (`cumsum(valid) - 1`)
    and, per compact column, the physical slot it holds. A column past the
    row's real tokens holds `length`, the zero column `_masked_conv1d` appends,
    so it gathers a zero and its cotangent lands there, never on a real slot;
    every real token writes its own column, so no two writers meet."""
    length = valid.shape[1]
    order, counts = valid_order(valid)
    rank = jnp.cumsum(jnp.asarray(valid, bool), axis=1, dtype=jnp.int32) - 1
    return rank, jnp.where(jnp.arange(length)[None, :] < counts[:, None], order, length).astype(jnp.int32)


def held_conv1d(x, kernel, valid, held=None, *, bias=None, segments=None, right_padded: bool = False,
                parts=None, dtype=None):
    """`[B, S, D]` convolved causally after the `held` history, `[B, K - 1, D]`
    or None for none, and the history the next call holds: a linear-attention
    or state-space layer's conv over its decode cache, padded slots read and
    advanced by none (`_masked_conv1d`). `bias` is added before the silu, and
    packed `segments` convolve each document alone; `parts` and `dtype` are
    `causal_conv1d`'s.

    `right_padded` says each row's `valid` slots are its first
    (`AttentionMetadata.right_padded`). Its real tokens then follow the held
    ones with no gap, so the row convolves as it stands and its padding is
    zeroed after: the masked conv's values, without its two gathers over
    every slot and their scattered gradients."""
    width = kernel.shape[-1] - 1
    conv = functools.partial(_behind, kernel=kernel, held=held, bias=bias, parts=parts, dtype=dtype)
    if valid is not None and right_padded and segments is None:
        empty = jnp.zeros((x.shape[0], width, x.shape[2]), x.dtype)
        stream = jnp.concatenate([rows_like(empty, x) if held is None else held, x], axis=1)
        valid = jnp.asarray(valid, bool)
        # The K-1 rows that end at the row's last real token.
        ends = jnp.sum(valid, axis=1, dtype=jnp.int32)
        rows = (ends[:, None] + jnp.arange(width)[None, :])[:, :, None]
        return conv(x, valid=valid), jnp.take_along_axis(stream, rows, axis=1)
    if valid is not None or segments is not None:
        valid = jnp.ones(x.shape[:2], bool) if valid is None else valid
        return _masked_conv1d(x, kernel, valid, held, bias=bias, segments=segments, parts=parts, dtype=dtype)
    history = x if held is None else jnp.concatenate([held, x], axis=1)
    return conv(x), history[:, -width:]


def _behind(x, *, kernel, held, bias, valid=None, parts=None, dtype=None):
    """`causal_conv1d` of `x` after the `held` history `[B, K - 1, D]`, or
    after the conv's own zeros where None."""
    if held is None:
        return causal_conv1d(x, kernel, bias=bias, valid=valid, parts=parts, dtype=dtype)
    width = held.shape[1]
    if valid is not None:
        valid = jnp.pad(valid, ((0, 0), (width, 0)), constant_values=True)
    out = causal_conv1d(jnp.concatenate([held, x], axis=1), kernel, bias=bias, valid=valid, parts=parts,
                        dtype=dtype)
    return jax.tree.map(lambda piece: piece[:, width:], out)


def held_scan(module: nn.Module, mixed, taps, bias, scan, state_shape: tuple[int, ...], *, decode: bool,
              valid, segments=None, state: str = 'ssm_state', state_dtype=None, conv_dtype=None,
              right_padded: bool = False, parts=None, dtype=None):
    """The conv and scan of a mixer that holds per-row state: a decode step or
    prefill advancing `module`'s flax cache, `conv_state` `[B, K-1, D]` held
    in `conv_dtype` and `state` `[B, *state_shape]` held in `state_dtype`
    (both `mixed`'s dtype where None), or rows with padding slots. `mixed`
    `[B, S, D]` is the conv's input in its own dtype and layout, and
    `scan(convolved, *, starts, valid, held)` the rest, given the convolved
    `[B, S, D]` in `dtype` (`mixed`'s where None), or its `parts` where given
    (`causal_conv1d`), and the held state as it is stored and returning
    `(output, final state)`. None for the allocation-only decode call. Packed
    `segments` reset the conv and the state at each document's first real
    token; the held state counts as the call's first document, which it
    continues. `right_padded` is `held_conv1d`'s."""
    batch = mixed.shape[0]
    held = conv_state = None
    if decode:
        allocated = module.has_variable('cache', state)
        conv_state = module.variable('cache', 'conv_state', jnp.zeros,
                                     (batch, taps.shape[1] - 1, mixed.shape[2]), conv_dtype or mixed.dtype)
        held = module.variable('cache', state, jnp.zeros, (batch, *state_shape), state_dtype or mixed.dtype)
        if not allocated:
            return None
    mixed, history = held_conv1d(mixed, taps, valid,
                                 None if conv_state is None else conv_state.value.astype(mixed.dtype),
                                 bias=bias, segments=segments, right_padded=right_padded, parts=parts,
                                 dtype=dtype)
    if conv_state is not None:
        conv_state.value = history.astype(conv_state.value.dtype)
    starts = None if segments is None else document_starts(segments, valid=valid)
    out, final = scan(mixed, starts=starts, valid=valid, held=None if held is None else held.value)
    if held is not None:
        held.value = final.astype(held.value.dtype)
    return out


def _masked_conv1d(x, kernel, valid, state=None, bias=None, segments=None, parts=None, dtype=None):
    """Convolve real tokens without advancing a paused row's history.

    The j-th real token of a row reads the K-1 real tokens before it, whatever
    padding sits between, and an invalid slot neither reads nor advances the
    history. `cumsum(valid) - 1` indexes each real token in the row's stream,
    so one convolution over the gathered stream replaces S sequential windows.
    `segments` `[B, S]` makes each packed document convolve alone
    (`document_conv1d`); the held `state` belongs to the first real token's
    document. Masking an ordinary convolution's input instead reads the zeros in
    a gap (5.0 off in fp32 on nine slots with holes at 2, 5 and 6,
    test_the_masked_conv_reads_across_a_gap). `parts` and `dtype` are
    `causal_conv1d`'s, applied to the real tokens' outputs.
    """
    batch, _, channels = x.shape
    width = kernel.shape[1] - 1
    if state is None:
        state = rows_like(jnp.zeros((batch, width, channels), x.dtype), x)
    valid = jnp.asarray(valid, bool)
    rank, source = _stream_order(valid)
    # Gathered in bounds from one appended zero row: an out-of-range 'fill'
    # gather differentiates into a scatter-add whose dropped index XLA's
    # deterministic GPU scatter writes onto the next channel (openxla/xla#49380).
    padded = jnp.pad(x, ((0, 0), (0, 1), (0, 0)))
    compact = jnp.take_along_axis(padded, source[:, :, None], axis=1, mode='promise_in_bounds')
    stream = jnp.concatenate([state, compact], axis=1)
    if segments is None:
        # The history in front carries the K-1 taps the first real token
        # reads, so row width + j of the convolution is the output of real
        # token j.
        convolved = causal_conv1d(stream, kernel, bias=bias, dtype=at_least_fp32(x.dtype))[:, width:]
    else:
        ordered = jnp.take_along_axis(segments, source, axis=1, mode='fill', fill_value=-1)
        held = jnp.broadcast_to(ordered[:, :1], (batch, width))
        convolved = document_conv1d(state, compact, held, ordered, kernel, bias)
    output = jnp.take_along_axis(convolved, jnp.maximum(rank, 0)[:, :, None], axis=1)
    # The K-1 stream rows from the row's token count on are the history it leaves.
    history = jnp.take_along_axis(
        stream, ((rank[:, -1] + 1)[:, None] + jnp.arange(width)[None, :])[:, :, None], axis=1)
    output = jnp.where(valid[:, :, None], output, 0)
    return conv_pieces(output, parts, x.dtype if dtype is None else dtype), history


def chunk_decay(g):
    """Cumulate per-chunk log decays and build the pairwise decay between positions.

    `g` is `[..., C, F]`: the C positions of a chunk and F decay channels.
    Returns the inclusive cumulative sum `gc` over C, `[..., C, F]`, and
    `decay[..., s, t, f] = exp(gc[s, f] - gc[t, f])` for s >= t, zero above
    the diagonal, `[..., C, C, F]`.

    The references subtract two fp32 cumulative sums, so every exponent
    carries the rounding of the chunk's whole accumulated magnitude, which
    exp turns into a relative error on every decay. Here the cumulative sum
    is compensated (`compensated_add`) and the difference takes both
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
    hi, lo = jax.lax.associative_scan(compensated_add, (g, jnp.zeros_like(g)), axis=g.ndim - 2)
    chunk_size = hi.shape[-2]
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
    `gc` `[B, H, NC, C, D]` being the log decay cumulated within each chunk,
    D one for a decay per head or Dk for one per key dimension (KDA's, whose
    rows of S decay apart, modeling_glm5_next.py:529-532), and `state`
    `[B, H, Dk, Dv]` enters the first chunk. Returns the state each chunk
    entered `[B, H, NC, Dk, Dv]`, each chunk's corrected values
    `[B, H, NC, C, Dv]` and the state leaving the last chunk.
    """
    def one_chunk(s, step):
        k_i, kd_i, v_i, gc_i = step
        v_corrected = v_i - kd_i @ s
        last = gc_i[..., -1:, :]
        written = k_i * jnp.exp(last - gc_i)
        return (s * jnp.swapaxes(jnp.exp(last), -1, -2) + jnp.swapaxes(written, -1, -2) @ v_corrected,
                (s, v_corrected))
    final, (entered, corrected) = jax.lax.scan(
        one_chunk, state, tuple(jnp.moveaxis(x, 2, 0) for x in (key, k_cumdecay, out_vals, gc)))
    return jnp.moveaxis(entered, 0, 2), jnp.moveaxis(corrected, 0, 2), final


def _paired(left, right, decay):
    """`sum_d left[s, d] right[t, d] decay[s, t, d]` within each chunk,
    `[..., C, C]`: one decay channel scales the product after it, one per key
    dimension weighs each term of it."""
    if decay.shape[-1] == 1:
        return (left @ jnp.swapaxes(right, -1, -2)) * decay[..., 0]
    return jnp.einsum('...sd,...td,...std->...st', left, right, decay)


def chunk_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE):
    """The chunked delta rule over [B, S, H, D] operands with the log decay
    `g` `[B, S, H, D]`, D one for a decay per head (the gated delta rule) or
    Dk for one per key dimension (Kimi Delta Attention); returns
    `(output [B, S, H, Dv], final_state [B, H, Dk, Dv])` in the input dtype,
    computed in fp32. The references' sequential correction is the inverse
    `strictly_lower_inverse` builds from its diagonal blocks.

    Only the memory crosses chunks, so only its recurrence runs chunk after
    chunk (`xla_chunk_states`). Each chunk's output, the references'
    `attn_inter + attn @ v_new`, reads the state that chunk entered and its
    corrected values, so it is computed for every chunk at once after the
    recurrence. The two decays differ only in how a pair of positions is
    weighed (`_paired`). It is the portable math: a GPU runs the gated rule on
    fla's kernels (`chunk_gated_delta_rule`), and this is their test oracle.
    """
    dtype, work = query.dtype, at_least_fp32(query.dtype)
    query, key, value, g, beta = (x.astype(work) for x in (query, key, value, g, beta))
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    pad = (chunk_size - S % chunk_size) % chunk_size
    query, key, value, g = (jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))) for x in (query, key, value, g))
    beta = jnp.pad(beta, ((0, 0), (0, pad), (0, 0)))
    T = S + pad
    query = query * (Dk ** -0.5)

    def chunks(x):  # [B, S, H, ...] -> [B, H, NC, C, ...], the reference's layout
        moved = jnp.moveaxis(x, 2, 1)  # [B, H, S, ...]
        return moved.reshape(B, H, T // chunk_size, chunk_size, *moved.shape[3:])

    q_c, k_c = chunks(query), chunks(key)
    kb_c, vb_c = chunks(key * beta[..., None]), chunks(value * beta[..., None])
    # The log decay cumulated within each chunk, the reference's
    # `g = g.cumsum(dim=-1)` (modeling_qwen3_next.py:417), and the decay
    # `exp(gc[s] - gc[t])` between positions s >= t, zero above the diagonal:
    # [B, H, NC, C, D], [B, H, NC, C, C, D].
    gc, decay = chunk_decay(chunks(g))
    # The mask is strictly lower (tril, -1): the reference's masked_fill
    # zeroes the diagonal too.
    strict = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_), -1)
    attn = jnp.where(strict, -_paired(kb_c, k_c, decay), 0.0)
    # The reference's `attn + I` operator after its row correction loop.
    inv = strictly_lower_inverse(attn)
    out_vals = inv @ vb_c  # the reference's `value = attn @ v_beta`
    k_cumdecay = inv @ (kb_c * jnp.exp(gc))

    state = jnp.zeros((B, H, Dk, Dv), work) if state is None else state.astype(work)
    # XLA:CPU workaround: under a jitted scan over the layers the chunk
    # loop's zero initial state came back with other values in most
    # processes (GLM-5-Next off by 8.7, 10.1 or NaN, byte-identical modules).
    # The barrier materializes the zero. Remove it when XLA:CPU no longer
    # returns an uninitialized scan carry here.
    state = jax.lax.optimization_barrier(state)
    entered, v_corrected, state = xla_chunk_states(k_c, k_cumdecay, out_vals, gc, state)
    # decay is zero above the diagonal, so this is the reference's inclusive
    # lower `masked_fill(triu(1), 0)`.
    core = (q_c * jnp.exp(gc)) @ entered + _paired(q_c, k_c, decay) @ v_corrected
    # core: [B, H, NC, C, Dv] -> [B, H, NC*C, Dv] -> [B, S, H, Dv]
    core = jnp.moveaxis(core.reshape(B, H, T, Dv), 1, 2)[:, :S]
    return core.astype(dtype), state.astype(dtype)


def chunk_gated_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE, *,
                           projected=None):
    """`torch_chunk_gated_delta_rule` (modeling_qwen3_next.py:374-453) with one
    log decay per head, `g` `[B, S, H]`. Its queries and keys may sit on fewer
    heads than the values, each serving `Hv // Hk` of them, as fla takes them.

    Where the port takes it (`port_refusal`), it is fla 0.5.2's chunked rule
    on Dew's port of its kernels (`fla_rule`), on fla's numbers, its operands
    and beta in `projected` where given, the dtype a layer's projections
    hand fla's rule; elsewhere the portable math (`chunk_delta_rule`) on the
    operands as they come, in `chunk_size` chunks, which the port's 64 need
    not match: the rule's value does not depend on them.
    """
    if value.shape[2] % key.shape[2]:
        raise ValueError(f"{value.shape[2]} value heads over {key.shape[2]} key heads: each key head serves "
                         "a whole number of value heads")
    rows, keys, values, gates = ((query, key, value, beta) if projected is None
                                 else (x.astype(projected) for x in (query, key, value, beta)))
    if on_the_port(rows, keys, values):
        return fla_rule(rows, keys, values, g, gates, state)
    group = value.shape[2] // key.shape[2]
    if group > 1:
        query, key = (jnp.repeat(x, group, axis=2) for x in (query, key))
    return chunk_delta_rule(query, key, value, g[..., None], beta, state, chunk_size)


def port_refusal(query, key, value) -> str | None:
    """Why a rule over these operands runs the portable math rather than the
    port (`fla_rule`), or None where the port takes it: every GPU Triton runs
    (`triton_runs`) and every shape, dtype, chunk size and mesh, except
    inside `jax.jvp`'s traces (`FORWARD_MODE`), which the port's custom VJP
    gives no tangent; inside a caller's `shard_map` that checks varying axes
    over operands that vary, where a Pallas call, which states no varying
    type for its results, cannot run; float64 operands, whose rule is the
    reference math itself; and keys wider than the port takes in their
    dtype."""
    widest = fla_gdn.WIDEST_KEYS if jnp.result_type(query, key, value).itemsize == 2 else fla_gdn.WIDEST
    return first_refusal(
        (triton_runs(), "the port's kernels need a GPU of sm80 or later"),
        (not FORWARD_MODE.get(), "forward_mode_attention traces for jax.jvp, which the port's custom VJP "
                                 "gives no tangent"),
        (not jax.typeof(value).mat.varying,
         "a shard_map that checks varying axes holds the operands, and a Pallas call states no "
         "varying type for its results"),
        (all(jnp.dtype(x.dtype) != jnp.float64 for x in (query, key, value)),
         "float64 operands, whose rule is the reference math itself"),
        (key.shape[-1] <= widest,
         f"keys {key.shape[-1]} wide; the port takes at most {widest} in "
         f"{jnp.result_type(query, key, value).name}"),
    )


def on_the_port(query, key, value) -> bool:
    """Whether the rule runs on the port; a GPU that runs it elsewhere logs
    why the first time (`ran_kernel`)."""
    refusal = port_refusal(query, key, value)
    if refusal is not None and triton_runs():
        ran_kernel('gated_delta_rule', 'xla', refusal, fastest='fla')
    return refusal is None


def fla_rule(query, key, value, g, beta, state=None):
    """`fla_gdn.chunk_gated_delta_rule` for any call the portable math takes:
    `[B, S, Hk, Dk]` queries and keys, `[B, S, Hv, Dv]` values, the log decay
    `g` and `beta` `[B, S, Hv]`, the memory `[B, Hv, Dk, Dv]` or None.

    The operands go to their common dtype, beta with them, as fla's layer
    holds them. On a mesh the rule runs per shard (`manual_map`): rows where
    the rules split them, and value heads with the key heads they read where
    the head split divides the key heads, the time axis whole; a split that
    does not divide them keeps the heads whole, so no key's gradient is
    summed across shards after rounding. Each shard's call is shaped to the
    kernels' blocks (`_blocked_rule`) and cut back. Off a GPU its kernels
    interpret, as the CPU suite runs them.
    """
    dtype = jnp.result_type(query, key, value)
    query, key, value, beta = (x.astype(dtype) for x in (query, key, value, beta))
    g = g.astype(jnp.float32)
    B, _, Hv, Dv = value.shape
    state = (jnp.zeros((B, Hv, key.shape[-1], Dv), jnp.float32) if state is None
             else state.astype(jnp.float32))
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or set(mesh.axis_names) <= set(mesh.manual_axes):
        return _blocked_rule(query, key, value, g, beta, state)
    spec = logical_spec(HEADS, value.shape)
    rows, heads = (spec[i] if i < len(spec) else None for i in (0, 2))
    if key.shape[2] % math.prod(mesh.shape[axis] for axis in mesh_axes(heads)):
        heads = None
    P = jax.sharding.PartitionSpec
    four, three, memory = P(rows, None, heads, None), P(rows, None, heads), P(rows, heads, None, None)
    return manual_map(_blocked_rule, (four, four, four, three, three, memory), (four, memory))(
        query, key, value, g, beta, state)


def _blocked_rule(query, key, value, g, beta, state):
    """The port's rule on one shard, shaped to its kernels' blocks: values
    wider than the port's widest split into more value heads of their key
    head; value heads padded to a power of two per key head, which dqkwg's
    block holds, with heads that neither write nor read (zero value, beta
    and decay); keys and values padded with zero columns to 64, 128 or 256,
    widths every block of the kernels' configs divides; the sequence padded
    to whole chunks with tokens that neither decay nor write. None of them
    changes a real output or gradient: every padded product is an exact
    zero. The outputs are cut back."""
    B, S, Hk, Dk = key.shape
    Hv, Dv = value.shape[2:]
    scale = Dk ** -0.5
    split = -(-Dv // fla_gdn.WIDEST)
    width = -(-Dv // split)
    if split > 1:
        value = _padded(value, 3, split * width).reshape(B, S, Hv * split, width)
        g, beta = (jnp.repeat(x, split, axis=2) for x in (g, beta))
        state = jnp.moveaxis(_padded(state, 3, split * width).reshape(B, Hv, Dk, split, width), 3, 2)
        state = state.reshape(B, Hv * split, Dk, width)
    served = value.shape[2] // Hk
    group = 1 << (served - 1).bit_length()

    def grouped(x, axis):
        x = _padded(x.reshape(*x.shape[:axis], Hk, served, *x.shape[axis + 1:]), axis + 1, group)
        return x.reshape(*x.shape[:axis], Hk * group, *x.shape[axis + 2:])

    value, g, beta, state = grouped(value, 2), grouped(g, 2), grouped(beta, 2), grouped(state, 1)
    blocks = (max(64, 1 << (Dk - 1).bit_length()), max(64, 1 << (value.shape[3] - 1).bit_length()))
    query, key = (_padded(x, 3, blocks[0]) for x in (query, key))
    value = _padded(value, 3, blocks[1])
    state = _padded(_padded(state, 2, blocks[0]), 3, blocks[1])
    length = -(-S // CHUNK_SIZE) * CHUNK_SIZE
    query, key, value, g, beta = (_padded(x, 1, length) for x in (query, key, value, g, beta))
    out, final = fla_gdn.chunk_gated_delta_rule(query, key, value, g, beta, state, scale, CHUNK_SIZE,
                                                not generation.triton_runs())
    out = out[:, :S, :, :width].reshape(B, S, Hk, group, width)[:, :, :, :served]
    out = out.reshape(B, S, Hv, split * width)[..., :Dv]
    final = final[:, :, :Dk, :width].reshape(B, Hk, group, Dk, width)[:, :, :served]
    final = jnp.moveaxis(final.reshape(B, Hv, split, Dk, width), 2, 3).reshape(B, Hv, Dk, split * width)
    return out, final[..., :Dv]


def _padded(x, axis: int, size: int):
    """`x` padded with zeros on the end of `axis` to `size`."""
    if x.shape[axis] == size:
        return x
    return jnp.pad(x, [(0, size - x.shape[axis]) if i == axis else (0, 0) for i in range(x.ndim)])


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
    [B, K-1, D] in the `cache` collection. A bfloat16 model holds its
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

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Learned maps, conv and algorithm-independent recurrent rule (`delta_flops`)."""
        return (sum(projection_flops(lengths, *shape) for shape in self.projections().values())
                + conv_flops(lengths, self.conv_features, self.conv_features,
                             self.conv_kernel, groups=self.conv_features)
                + delta_flops(lengths, self.num_v_heads, self.head_k_dim, self.head_v_dim))

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """Frozen learned maps/conv, excluding the nonparameter state contractions."""
        return self.model_flops(lengths) - delta_flops(
            lengths, self.num_v_heads, self.head_k_dim, self.head_v_dim)

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        """The delta net's maps as `{name: (inputs, outputs)}`: the
        reference's four input projections or Qwen3-Next's two fused ones,
        and the map back to the residual. Setup builds them and the count
        reads them."""
        if self.fused_in_proj:
            widths = {'in_proj_qkvz': 2 * self.key_features + 2 * self.value_features,
                      'in_proj_ba': 2 * self.num_v_heads}
        else:
            widths = {'in_proj_qkv': 2 * self.key_features + self.value_features,
                      'in_proj_z': self.value_features, 'in_proj_b': self.num_v_heads,
                      'in_proj_a': self.num_v_heads}
        return {**{name: (self.emb_features, width) for name, width in widths.items()},
                'out_proj': (self.value_features, self.emb_features)}

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
        for name, (_, outputs) in self.projections().items():
            setattr(self, name, dense(outputs, name=name))
        self.conv1d = DepthwiseConv1d(features=self.conv_features,
                                       kernel=self.conv_kernel, name='conv1d')
        self.A_log = self.param('A_log', nn.initializers.constant(0.0),
                                (self.num_v_heads,), jnp.float32)
        self.dt_bias = self.param('dt_bias', nn.initializers.ones,
                                  (self.num_v_heads,), jnp.float32)
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
        """(mixed, z, b, a) of `x`: the conv's input `[B, S, 2K + V]`, the
        queries, keys and values flat over heads in that order, z
        `[B, S, Hv*Dv]`, and b and a `[B, S, Hv]`.

        Each projection is placed as the residual is, its rows where the
        batch axes put them and its width whole: the rules split no delta
        net width over the tensor axis, and the rule's `manual_map` takes
        rows split. Unplaced, an fsdp mesh ran the layer head-split with
        every row gathered to every device, its all-to-alls 8.4 GiB of a
        Qwen3.5-9B step over FSDP 2 (docs/performance.md)."""
        B, S, _ = x.shape

        def projected(dense):
            return constrain(dense(x), RESIDUAL)

        if not self.fused_in_proj:
            return (projected(self.in_proj_qkv), projected(self.in_proj_z), projected(self.in_proj_b),
                    projected(self.in_proj_a))
        # One row group per key head: its q, its k, then the v and z of the
        # value heads it serves (modeling_qwen3_next.py:558-586).
        per_key = self.per_v * self.head_v_dim
        qkvz = projected(self.in_proj_qkvz).reshape(B, S, self.num_k_heads, 2 * self.head_k_dim + 2 * per_key)
        query, key, value, z = jnp.split(
            qkvz, [self.head_k_dim, 2 * self.head_k_dim, 2 * self.head_k_dim + per_key], axis=-1)
        ba = projected(self.in_proj_ba).reshape(B, S, self.num_k_heads, 2 * self.per_v)
        b, a = jnp.split(ba, 2, axis=-1)
        mixed = jnp.concatenate([query.reshape(B, S, -1), key.reshape(B, S, -1), value.reshape(B, S, -1)],
                                axis=-1)
        return mixed, z.reshape(B, S, -1), b.reshape(B, S, -1), a.reshape(B, S, -1)

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None):
        del positions, segment_ids, kv_store
        B, S, _ = x.shape
        valid = None if attention_metadata is None else attention_metadata.valid
        if valid is not None and valid.shape != (B, S):
            raise ValueError(f"row validity must be {(B, S)}, got {valid.shape}")
        mixed, z, b, a = self._project(x)
        key_dim = self.key_features
        operand = mixed.dtype

        # fp32 on purpose, as the reference notes: an fp16 A can make exp
        # underflow to -inf (modeling_qwen3_next.py:652-653); at least fp32,
        # so a float64 model's stays float64 (`at_least_fp32`).
        wide = at_least_fp32(operand)
        beta = nn.sigmoid(b.astype(wide))
        g = -jnp.exp(self.A_log.astype(wide)) * nn.softplus(a.astype(wide) + self.dt_bias.astype(wide))
        if valid is not None:
            # exp(0)=1 and beta=0 preserve recurrent memory for invalid input.
            beta = jnp.where(valid[:, :, None], beta, 0.0)
            g = jnp.where(valid[:, :, None], g, 0.0)
        # The conv rounds once, to the dtype the rule takes: the projections'
        # where the port runs it (`chunk_gated_delta_rule`), the wide one for
        # the math.
        widths = (self.head_k_dim, self.head_k_dim, self.head_v_dim)
        rows, keys, values = (self._split_heads(part, width) for part, width in
                              zip(jnp.split(mixed, [key_dim, 2 * key_dim], axis=-1), widths, strict=True))
        ported = S > 1 and port_refusal(rows, keys, values) is None

        def scan(convolved, *, starts, valid, held):
            del starts
            query, key, value = (self._split_heads(part, width)
                                 for part, width in zip(convolved, widths, strict=True))
            state = None if held is None else held.astype(wide)
            if S > 1:
                # fla's layer hands its rule the projections' dtype, the
                # queries and keys on their own heads and beta beside them.
                return chunk_gated_delta_rule(query, key, value, g, beta, state, self.chunk_size,
                                              projected=operand)
            query, key = self._expand_kv(query), self._expand_kv(key)
            if S == 1 and held is not None and delta_rule.fits(held):
                return decode_gated_delta_rule(query, key, value, g, beta, held,
                                               None if valid is None else valid[:, 0])
            out, final = recurrent_gated_delta_rule(query, key, value, g, beta, state)
            if held is not None and held.dtype == jnp.bfloat16:
                final = delta_rule.round_to_bf16(final, key[:, 0].astype(wide), value[:, 0].astype(wide))
            return out, final

        # The conv state holds the projections' own values, so it is held in
        # their dtype at no cost; the recurrent state is held as the
        # docstring says. The first decode-mode call only allocates, the way
        # open_kv_cache's is: init_cache's dummy token must not consume a
        # position or leave state behind. The conv normalises the queries
        # and keys (`l2norm`).
        stored = (jnp.promote_types(self.dtype, jnp.bfloat16)
                  if self.dtype is not None and asks_default_precision(self.precision, configured=True)
                  else wide)
        heads = ((key_dim, self.head_k_dim), (key_dim, self.head_k_dim), (self.value_features, None))
        out = held_scan(self, mixed, self.conv1d()[0], None, scan,
                        (self.num_v_heads, self.head_k_dim, self.head_v_dim), decode=decode,
                        valid=valid, state='recurrent_state', state_dtype=stored,
                        conv_dtype=jnp.promote_types(operand, stored),
                        right_padded=attention_metadata is not None and attention_metadata.right_padded,
                        parts=heads, dtype=operand if ported else wide)
        if out is None:
            return self.out_proj(jnp.zeros((B, S, self.value_features), self.dtype))
        # The rule's output, attention's 'context' (a remat policy's `RESIDUALS`).
        out = checkpoint_name(out, 'context')

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
    the activation a config field. It takes per-head `[batch, length, heads,
    head_dim]` outputs and gates (`gated_rms_normalized`).
    """

    epsilon: float = 1e-6
    activation: str = 'silu'
    dtype: Dtype | None = None
    implementation: str = 'auto'

    @nn.compact
    def __call__(self, x, gate):
        dtype = self.dtype if self.dtype is not None else x.dtype
        scale = self.param('weight', nn.initializers.ones,
                           (x.shape[-1],), jnp.float32)
        return gated_rms_normalized(x, gate, scale, self.epsilon, dtype, self.activation,
                                    implementation=self.implementation, axes=HEADS)
