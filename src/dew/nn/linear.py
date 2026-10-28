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

import dataclasses
import functools
import math
from collections.abc import Sequence
from typing import ClassVar, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.interpreters.ad import JVPTracer
from jax.sharding import PartitionSpec as P

from dew.batch import Coordinates, valid_order
from dew.mesh import SEQUENCE_AXIS, LayoutRefused, MeshAxes, logical_spec, mesh_axes
from dew.nn.dense import Dense
from dew.nn.kernels import recurrence, short_conv

from .attention import FORWARD_MODE, gated_rms_normalized, repeat_kv_heads
from .blocks import normal_kernel
from .flops import conv_flops, delta_flops, projection_flops
from .inputs import PLAIN_CALL, MixerCall, coordinates_of
from .joined import joined_dense, joins
from .kernels import generation
from .kernels.generation import INHERENT, Refusal, triton_runs
from .kernels.recurrence_chunk import (
    CHUNK_SIZE,
    chunk_gated_delta_rule,
    entering_memory,
    kernel_routed,
    recurrent_delta_rule,
)
from .kernels.selection import Kernel, Op, Requirement, forced, select
from .precision import L2NORM_EPS, asks_default_precision, at_least_fp32, float32_dot_general, l2norm
from .sharding import (
    HEADS,
    MLP_HIDDEN,
    RESIDUAL,
    automatic_axes,
    constrain,
    grouped_heads,
    logical_axes,
    manual_map,
    rows_like,
    seen_whole,
    sequence_shards,
)

Pieces = Sequence[tuple[int, int | None]]
"""`causal_conv1d`'s `parts`: each piece's width and the head its l2 norm runs over, or None."""


@overload
def causal_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, segments=None,
                  parts: None = None, dtype=None) -> jax.Array: ...


@overload
def causal_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, segments=None,
                  parts: Pieces, dtype=None) -> tuple[jax.Array, ...]: ...


def causal_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, segments=None,
                  parts: Pieces | None = None, dtype=None) -> jax.Array | tuple[jax.Array, ...]:
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
    of padding slots. `segments` `[B, S]` numbers packed documents, and a tap
    reads a token only in its output token's: each document convolves as if
    zeros preceded it, as mamba_ssm's `causal_conv1d_fn(seq_idx=...)` does.
    `parts` `((width, head), ...)` returns the channels as
    those consecutive pieces, a piece with a `head` l2-normalised over each
    `head` channels (`l2norm`), the queries and keys of a delta rule: each
    piece is the array its scan reads, not a slice of one to copy out.

    The pieces are named `conv` for a remat policy to keep.

    On a GPU that runs Dew's Triton kernels all of it is one kernel forward
    and one backward (`short_conv`), which save only `x`; elsewhere, and for
    a conv they turn down (`short_conv.CONV`), `xla_conv1d`. On a mesh whose
    axes split only the rows of the conv's operands, as data and fsdp axes
    do, each shard convolves its own rows (`manual_map`), so the kernels
    take an fsdp run's convs too; a mesh splitting the time or channel axis
    leaves the conv to XLA's partitioner.
    """
    dtype = x.dtype if dtype is None else jnp.dtype(dtype)
    rows = _row_split(x)
    if rows is not None:
        out = _per_row_shard(x, kernel, activation, bias, valid, segments, parts, dtype, rows)
    else:
        pieces = ((x.shape[-1], None),) if parts is None else tuple(parts)
        case = short_conv.ConvCase(placed=triton_runs(),
                                   reverse=not FORWARD_MODE.get() and not isinstance(x, JVPTracer),
                                   whole=seen_whole(x), shape=x.shape, dtype=jnp.dtype(x.dtype), out=dtype,
                                   taps=kernel.shape[-1], parts=pieces)
        if select(short_conv.CONV, case).kernel == 'pallas':
            out = short_conv.conv(x, kernel, bias, None if valid is None else jnp.asarray(valid, jnp.int8),
                                  None if segments is None else jnp.asarray(segments, jnp.int32),
                                  pieces, activation=activation, eps=L2NORM_EPS, dtype=dtype,
                                  interpret=not generation.triton_runs())
            out = out[0] if parts is None else out
        else:
            out = xla_conv1d(x, kernel, activation, bias, valid=valid, segments=segments, parts=parts,
                             dtype=dtype)
    # The conv's output, which a remat policy may keep (`dew.nn.mixer_base.RESIDUALS`).
    return jax.tree.map(lambda piece: checkpoint_name(piece, 'conv'), out)


def _row_split(x) -> MeshAxes | None:
    """The mesh axes that split the rows of a conv's `[B, S, D]` operand where they are every axis
    left to the partitioner that is larger than one, and the context's rules split nothing else of
    it, as a residual or a feed-forward activation; None otherwise. An axis that left the operand
    whole on its devices would sum the taps' gradient over copies of the same rows."""
    automatic = set(automatic_axes())
    if not automatic:
        return None
    residual, hidden = (tuple(logical_spec(axes, x.shape)) for axes in (RESIDUAL, MLP_HIDDEN))
    if any(entry is not None for spec in (residual, hidden) for entry in spec[1:]):
        return None
    rows = residual[0] if residual else None
    return rows if automatic <= set(mesh_axes(rows)) else None


def _per_row_shard(x, kernel, activation, bias, valid, segments, parts, dtype, rows: MeshAxes):
    """`causal_conv1d` with each shard of `rows` convolving its own rows (`manual_map`); the taps and
    bias are whole on every shard, and their gradients sum over the shards' rows."""
    row = P(rows, None, None)
    pieces = 1 if parts is None else len(tuple(parts))

    def local(x, kernel, bias, valid, segments):
        return causal_conv1d(x, kernel, activation, bias, valid=valid, segments=segments, parts=parts,
                             dtype=dtype)

    tokens = [None if given is None else P(rows, None) for given in (valid, segments)]
    in_specs = (row, P(), None if bias is None else P(), *tokens)
    return manual_map(local, in_specs, row if parts is None else (row,) * pieces)(x, kernel, bias, valid,
                                                                                 segments)


def xla_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, segments=None,
               parts: Pieces | None = None, dtype=None) -> jax.Array | tuple[jax.Array, ...]:
    """`causal_conv1d`'s arithmetic in XLA: a CPU's and a TPU's, and the
    oracle of the GPU's kernels."""
    dtype = x.dtype if dtype is None else jnp.dtype(dtype)
    _, K = kernel.shape
    length = x.shape[1]
    wide = jnp.promote_types(at_least_fp32(x.dtype), dtype)
    padded = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0))).astype(wide)
    taps = kernel.astype(wide)

    def read(tap):
        """The tokens `K - 1 - tap` before each, those of its own document."""
        window = padded[:, tap:tap + length]
        if segments is None:
            return window
        before = jnp.pad(segments, ((0, 0), (K - 1, 0)))[:, tap:tap + length]
        return jnp.where((before == segments)[..., None], window, 0)

    windows = read(0) * taps[:, 0]
    for tap in range(1, K):
        windows = windows + read(tap) * taps[:, tap]
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


def real_tail(x, valid, segments, width: int):
    """The last `width` real tokens of each row of `x` `[B, S, D]` (`valid` `[B, S]`), zeros in front
    where the row holds fewer, with their documents (`segments` `[B, S]`, -1 in front; all 0 where
    None) and the row's real count `[B]`: what a later slice of the row's sequence reads as the history
    behind its first real token (`held_conv1d`'s `held`)."""
    length = x.shape[1]
    order, count = valid_order(jnp.asarray(valid, bool))
    # Each tail column's place among the row's real tokens, negative in front of the first, and the
    # slot it reads: the real token's own, or the appended zero (document -1) in front.
    places = count[:, None] - width + jnp.arange(width)[None, :]
    slots = jnp.where(places >= 0, jnp.take_along_axis(order, jnp.maximum(places, 0), axis=1), length)
    padded = jnp.pad(x, ((0, 0), (0, 1), (0, 0)))
    tail = jnp.take_along_axis(padded, slots[:, :, None], axis=1, mode='promise_in_bounds')
    owners = (jnp.zeros(valid.shape, jnp.int32) if segments is None else jnp.asarray(segments, jnp.int32))
    documents = jnp.take_along_axis(jnp.pad(owners, ((0, 0), (0, 1)), constant_values=-1), slots, axis=1)
    return tail, documents, count


def held_conv1d(x, kernel, valid, held=None, *, bias=None, segments=None, right_padded: bool = False,
                parts=None, dtype=None):
    """`[B, S, D]` convolved causally after the `held` history, `[B, K - 1, D]`
    or None for none, and the history the next call holds: a linear-attention
    or state-space layer's conv over its decode cache, padded slots read and
    advanced by none (`_masked_conv1d`). `bias` is added before the silu, and
    packed `segments` convolve each document alone; `parts` and `dtype` are
    `causal_conv1d`'s.

    `right_padded` says each row's `valid` slots are its first
    (`AttentionMetadata.right_padded`), as they all are where `valid` is
    None. Its real tokens then follow the held ones with no gap, so the row
    convolves as it stands, its packed documents apart, and its padding is
    zeroed after: the masked conv's values, without its two gathers over
    every slot and their scattered gradients.

    Behind a held history `x` passes an optimization barrier: at one row XLA
    lowers the projection that made `x` to a reduction and fused it into the
    history's concatenate, which then ran at about 90 GB/s. On an H100 a
    Qwen3.5-0.8B gated delta net's decode step took 338 against 57 us, 170
    against 41 with bf16 parameters (CUPTI); at 8 rows the projection is a
    GEMM and the barrier changes nothing."""
    if held is not None:
        x = jax.lax.optimization_barrier(x)
    width = kernel.shape[-1] - 1
    if valid is not None and not right_padded:
        return _masked_conv1d(x, kernel, valid, held, bias=bias, segments=segments, parts=parts, dtype=dtype)
    batch, length, channels = x.shape
    empty = jnp.zeros((batch, width, channels), x.dtype)
    stream = jnp.concatenate([rows_like(empty, x) if held is None else held, x], axis=1)
    valid = None if valid is None else jnp.asarray(valid, bool)
    # The K-1 stream rows that end at the row's last real token.
    ends = (rows_like(jnp.full((batch,), length, jnp.int32), x) if valid is None
            else jnp.sum(valid, axis=1, dtype=jnp.int32))
    rows = ends[:, None] + jnp.arange(width)[None, :]
    history = jnp.take_along_axis(stream, rows[:, :, None], axis=1)
    if segments is not None:
        # The next call continues the last real token's document, so the history keeps only its tokens,
        # the held ones counting as the first document's, as `_masked_conv1d`'s does.
        owners = _held_first(segments, width)
        last = jnp.take_along_axis(owners, ends[:, None] + width - 1, axis=1)
        history = jnp.where((jnp.take_along_axis(owners, rows, axis=1) == last)[:, :, None], history, 0)
    return _behind(x, kernel=kernel, held=held, bias=bias, valid=valid, segments=segments, parts=parts,
                   dtype=dtype), history


def _held_first(segments, width: int):
    """`segments` `[B, S]` behind `width` held tokens of the first one's
    document. Not `jnp.pad`'s 'edge' mode, which slices every axis's edge, the
    rows an Explicit axis splits too."""
    return jnp.concatenate([jnp.broadcast_to(segments[:, :1], (segments.shape[0], width)), segments], axis=1)


def _behind(x, *, kernel, held, bias, valid=None, segments=None, parts=None, dtype=None):
    """`causal_conv1d` of `x` after the `held` history `[B, K - 1, D]`, or
    after the conv's own zeros where None; the history is the first of
    packed `segments`' documents."""
    if held is None:
        return causal_conv1d(x, kernel, bias=bias, valid=valid, segments=segments, parts=parts, dtype=dtype)
    width = held.shape[1]
    if valid is not None:
        valid = jnp.pad(valid, ((0, 0), (width, 0)), constant_values=True)
    if segments is not None:
        segments = _held_first(segments, width)
    out = causal_conv1d(jnp.concatenate([held, x], axis=1), kernel, bias=bias, valid=valid, segments=segments,
                        parts=parts, dtype=dtype)
    return jax.tree.map(lambda piece: piece[:, width:], out)


def held_scan(module: nn.Module, mixed, taps, bias, scan, state_shape: tuple[int, ...], *, decode: bool,
              coordinates: Coordinates, state: str = 'ssm_state', state_dtype=None, conv_dtype=None,
              parts=None, dtype=None, token=None):
    """The conv and scan of a mixer that holds per-row state: a decode step or
    prefill advancing `module`'s flax cache, `conv_state` `[B, K-1, D]` held
    in `conv_dtype` and `state` `[B, *state_shape]` held in `state_dtype`
    (both `mixed`'s dtype where None), or rows with padding slots. `mixed`
    `[B, S, D]` is the conv's input in its own dtype and layout, and
    `scan(convolved, *, starts, valid, held)` the rest, given the convolved
    `[B, S, D]` in `dtype` (`mixed`'s where None), or its `parts` where given
    (`causal_conv1d`), and the held state as it is stored and returning
    `(output, final state)`. None for the allocation-only decode call. The
    tokens' `coordinates` say which are real and which start a packed
    document, where the conv and the state reset; the held state counts as
    the call's first document, which it continues. `token(mixed, history,
    held)`, where given, runs a decode token's conv and scan in one, from the
    history and state as they are stored, returning `(output, history',
    final state)`; the cache stays this function's either way."""
    batch = mixed.shape[0]
    held = conv_state = None
    if decode:
        allocated = module.has_variable('cache', state)
        conv_state = module.variable('cache', 'conv_state', jnp.zeros,
                                     (batch, taps.shape[1] - 1, mixed.shape[2]), conv_dtype or mixed.dtype)
        held = module.variable('cache', state, jnp.zeros, (batch, *state_shape), state_dtype or mixed.dtype)
        if not allocated:
            return None
        if token is not None:
            out, history, final = token(mixed, conv_state.value, held.value)
            conv_state.value = history.astype(conv_state.value.dtype)
            held.value = final.astype(held.value.dtype)
            return checkpoint_name(checkpoint_name(out, 'scan'), 'context')
    valid = coordinates.valid
    mixed, history = held_conv1d(mixed, taps, valid,
                                 None if conv_state is None else conv_state.value.astype(mixed.dtype),
                                 bias=bias, segments=coordinates.segments,
                                 right_padded=coordinates.right_padded, parts=parts, dtype=dtype)
    if conv_state is not None:
        conv_state.value = history.astype(conv_state.value.dtype)
    out, final = scan(mixed, starts=coordinates.starts, valid=valid,
                      held=None if held is None else held.value)
    if held is not None:
        held.value = final.astype(held.value.dtype)
    # The scan's output under both names a remat policy reads it by (`RESIDUALS`): a mixer's
    # 'context', as MaxText's recipes keep it, and the 'scan' only a recurrence marks.
    return checkpoint_name(checkpoint_name(out, 'scan'), 'context')


def over_sequence(local, shards: int, tokens: tuple, whole: tuple):
    """`local(*tokens, *whole)` in a `manual_map` that splits the rows and the sequence of each of `tokens`
    (`[B, S, ...]` or None) as the rule table places a `[batch, length, ...]` activation, `whole` on every
    shard, its output placed as the tokens are: a recurrent mixer's conv and scan over its shard's slice
    of whole training sequences (`sequence_conv`), its projections and norms token by token outside with
    GSPMD."""
    length = tokens[0].shape[1]
    if length % shards:
        raise LayoutRefused(f"the sequence of {length} tokens splits {shards} ways over the mesh's sequence "
                            f"axis, and it does not divide")
    spec = logical_spec(("activation_batch", "activation_length"), tokens[0].shape[:2])
    return manual_map(local, (*(spec,) * len(tokens), *(P(),) * len(whole)), spec)(*tokens, *whole)


def sequence_conv(mixed, taps, bias, segments, valid, *, parts=None, dtype=None):
    """This shard's slice of `mixed` `[B, S, D]` convolved causally, inside a `shard_map` that holds the
    sequence axis manual (`over_sequence`): each real token's conv reads the `K-1` real tokens before it in
    its document, wherever they lie, from earlier shards too, and a segment change resets it wherever it
    falls. With `valid` the halo is gathered from every earlier shard (`sequence_halo`), so a shard of
    fewer real tokens than the conv reads passes on the ones before it; without, every slot is real and
    the previous shard's last `K-1` tokens and segments come by one `ppermute` (zeros on the first).
    `parts` and `dtype` are `causal_conv1d`'s."""
    width = taps.shape[1] - 1
    length = mixed.shape[1]
    if length < width:
        raise LayoutRefused(
            f"each sequence shard holds {length} tokens, fewer than the {width} the "
            f"conv reads behind its first; use fewer sequence shards or longer sequences")
    if valid is not None:
        halo = sequence_halo(mixed, valid, segments, width)
        convolved, _ = held_conv1d(mixed, taps, valid, halo, bias=bias, segments=segments, parts=parts,
                                   dtype=dtype)
        return convolved
    forward = [(shard, shard + 1) for shard in range(jax.lax.axis_size(SEQUENCE_AXIS) - 1)]
    history = jax.lax.ppermute(mixed[:, length - width:], SEQUENCE_AXIS, forward)
    owners = None
    if segments is not None:
        before = jax.lax.ppermute(segments[:, length - width:], SEQUENCE_AXIS, forward)
        owners = jnp.concatenate([before, segments], axis=1)
    stream = jnp.concatenate([history, mixed], axis=1)
    convolved = causal_conv1d(stream, taps, bias=bias, segments=owners, parts=parts, dtype=dtype)
    return (convolved[:, width:] if parts is None
            else tuple(piece[:, width:] for piece in convolved))


def sequence_halo(mixed, valid, segments, width: int):
    """The `width` real tokens before this shard's first, `[B, width, conv_dim]`, from every earlier
    shard of the sequence axis: each shard's last real tokens (`real_tail`) gathered, then taken in
    order, so a shard holding fewer than `width` real tokens passes on the ones before it too. Rows of
    another document than this shard's first real token's are zeros, where its conv starts afresh."""
    tail, documents, count = real_tail(mixed, valid, segments, width)
    tails, all_documents, counts = (jax.lax.all_gather(value, SEQUENCE_AXIS)
                                    for value in (tail, documents, count))
    here = jax.lax.axis_index(SEQUENCE_AXIS)
    halo = jnp.zeros_like(tail)
    halo_documents = jnp.full_like(documents, -1)
    slots = jnp.arange(width)[None, :]
    for shard in range(tails.shape[0]):
        # A shard's tail holds its last `kept` real tokens at its end, zeros before them; behind them
        # come the halo's last `width - kept`.
        kept = jnp.minimum(counts[shard], width)[:, None]
        behind = slots < width - kept
        rows = jnp.minimum(kept + slots, width - 1)
        joined = jnp.where(behind[:, :, None], jnp.take_along_axis(halo, rows[:, :, None], axis=1),
                           tails[shard])
        joined_documents = jnp.where(behind, jnp.take_along_axis(halo_documents, rows, axis=1),
                                     all_documents[shard])
        earlier = shard < here
        halo = jnp.where(earlier, joined, halo)
        halo_documents = jnp.where(earlier, joined_documents, halo_documents)
    if segments is None:
        return halo
    first_real = jnp.argmax(jnp.asarray(valid, bool), axis=1)[:, None]
    first = jnp.take_along_axis(jnp.asarray(segments, jnp.int32), first_real, axis=1)
    return jnp.where((halo_documents == first)[:, :, None], halo, 0)


def split_delta_rule(mixed, g, beta, segments, starts, valid, taps, *, widths, parts, dtype, rule):
    """A delta-rule mixer's conv and rule over this shard's slice of whole training sequences, inside a
    `shard_map` that holds the sequence axis manual (`over_sequence`): the conv with its halo
    (`sequence_conv`, its `parts` the queries', keys' and values' widths, the first two l2-normalised), the
    pieces in heads of their `widths`, and `rule(query, key, value, g, beta, state, starts)` from the
    memory this shard's slice enters with (`entering_memory`), its output named as `held_scan`'s."""
    query, key, value = (piece.reshape(*piece.shape[:2], -1, width) for piece, width in
                         zip(sequence_conv(mixed, taps, None, segments, valid, parts=parts, dtype=dtype),
                             widths, strict=True))
    state = entering_memory(rule, query, key, value, g, beta, starts, axis=SEQUENCE_AXIS)
    out, _ = rule(query, key, value, g, beta, state, starts)
    return checkpoint_name(checkpoint_name(out, 'scan'), 'context')


def _masked_conv1d(x, kernel, valid, state=None, bias=None, segments=None, parts=None, dtype=None):
    """Convolve real tokens without advancing a paused row's history.

    The j-th real token of a row reads the K-1 real tokens before it, whatever
    padding sits between, and an invalid slot neither reads nor advances the
    history. `cumsum(valid) - 1` indexes each real token in the row's stream,
    so one convolution over the gathered stream replaces S sequential windows.
    `segments` `[B, S]` makes each packed document convolve alone
    (`causal_conv1d`); the held `state` belongs to the first real token's
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
    # The K-1 stream rows from the row's token count on are the history it leaves.
    rows = (rank[:, -1] + 1)[:, None] + jnp.arange(width)[None, :]
    history = jnp.take_along_axis(stream, rows[:, :, None], axis=1)
    owners = None
    if segments is not None:
        # The stream's documents, the history in front counting as the first real token's.
        ordered = jnp.take_along_axis(segments, source, axis=1, mode='fill', fill_value=-1)
        owners = _held_first(ordered, width)
        # The next call continues the last real token's document, so the history keeps only its
        # tokens; an earlier document's, where the last is shorter than K-1, read as its start's zeros.
        kept = jnp.take_along_axis(owners, rows, axis=1)
        last = jnp.take_along_axis(ordered, jnp.maximum(rank[:, -1:], 0), axis=1)
        history = jnp.where(((kept == last) | (rank[:, -1:] < 0))[:, :, None], history, 0)
    # The history in front carries the K-1 taps the first real token reads, so row width + j of the
    # convolution is the output of real token j.
    convolved = causal_conv1d(stream, kernel, bias=bias, segments=owners,
                              dtype=at_least_fp32(x.dtype))[:, width:]
    output = jnp.take_along_axis(convolved, jnp.maximum(rank, 0)[:, :, None], axis=1)
    output = jnp.where(valid[:, :, None], output, 0)
    return conv_pieces(output, parts, x.dtype if dtype is None else dtype), history


def decode_gated_delta_rule(query, key, value, g, beta, state, active=None):
    """`recurrent_gated_delta_rule` for one decode token `[B, 1, H, D]`
    through the recurrence's decode step (`dew.nn.kernels.recurrence.step`), which reads and writes
    the state once (CUDA, where no `recurrence.refusal` turns the state down). A row `active`
    `[B]` marks False keeps its state untouched and outputs zeros. On a mesh
    the step runs per shard (`manual_map`), each shard's rows and heads with
    their state, as `fla_rule` runs: the kernel has no partitioning rule, and
    the state is most of a decode step's bytes."""
    work = jnp.float32
    query, key, value = (x[:, 0].astype(work) for x in (query, key, value))
    out, final = stepped_per_shard(STEP, state, query * key.shape[-1] ** -0.5, key, value,
                                   g[:, 0].astype(work), beta[:, 0].astype(work), active)
    return out[:, None], final


def stepped_per_shard(hooks: recurrence.Hooks, state: jax.Array, query: jax.Array, key: jax.Array,
                      value: jax.Array, log_decay: jax.Array, beta: jax.Array, active: jax.Array | None
                      ) -> tuple[jax.Array, jax.Array]:
    """`recurrence.step`'s `(output [B, Hv, Dv], state')` of a decode token of `query`, `key`,
    `value` `[B, H, D]`, `log_decay` and `beta` `[B, Hv]` over its `state` `[B, Hv, Dk, Dv]`, run per
    shard (`manual_map`), each shard's rows and heads with their state: the launch has no partitioning
    rule, and the state is most of a decode step's bytes. A row `active` `[B]` marks False keeps its
    state."""
    active = jnp.ones(state.shape[:1], bool) if active is None else jnp.asarray(active, bool)
    rows, heads = grouped_heads((state.shape[0], 1, *value.shape[1:]), key.shape[1])
    three, two, memory = P(rows, heads, None), P(rows, heads), P(rows, heads, None, None)

    def local(query, key, value, log_decay, beta, held, active):
        out, _, final = recurrence.step(hooks, recurrence.Step(
            held, query=query, key=key, value=value, log_decay=log_decay, beta=beta, active=active))
        return out, final

    return manual_map(local, (three, three, three, two, two, memory, P(rows)), (three, memory))(
        query, key, value, log_decay, beta, state, active)


def decode_kernel(state: jax.Array | jax.ShapeDtypeStruct, *, tangent: bool, fuses: bool) -> str:
    """The kernel a decode token over `state` `[B, Hv, Dk, Dv]` runs on the target, as the decode selector
    reports it (`DELTA_DECODE`): a caller that has a conv, gates and a gated norm for the fused step to
    take (`fuses`), and a token traced for `jax.jvp` (`tangent`)."""
    return select(DELTA_DECODE, DecodeCase(recurrence.refusal(state, tangent=tangent), fuses)).kernel


def decode_step(hooks: recurrence.Hooks, state: jax.Array, query: jax.Array, key: jax.Array,
                value: jax.Array, log_decay: jax.Array, beta: jax.Array, active: jax.Array | None, *,
                tangent: bool) -> tuple[jax.Array, jax.Array]:
    """A decode token's `(output, state')` through the shared recurrence, its operands as
    `stepped_per_shard` takes them: the step kernel per shard where `decode_kernel` picks it, else the
    portable step (`recurrence.step_reference`) in the operands' own precision, which the compiler
    partitions."""
    if decode_kernel(state, tangent=tangent, fuses=False) == 'step':
        return stepped_per_shard(hooks, state, query, key, value, log_decay, beta, active)
    out, _, final = recurrence.step_reference(hooks, recurrence.Step(
        state, query=query, key=key, value=value, log_decay=log_decay, beta=beta, active=active))
    return out, final


STEP = recurrence.Hooks(decay='scalar')
"""The decode step between the layer's own conv and norm: the delta rule with one decay a head."""


@dataclasses.dataclass(frozen=True)
class DecodeCase:
    """A gated delta net's decode token as the decode selector reads it (`DELTA_DECODE`): why the launch
    turns its state down (`refusal`, `recurrence.refusal`: a token traced for `jax.jvp`, a GPU, the
    state's precision and widths), None where it takes it, and whether the caller has a conv, gates and a
    gated norm for the fused step to take (`fuses`)."""

    refusal: Refusal | None
    fuses: bool


def _launches(code: str, row: str) -> Requirement[DecodeCase]:
    """The launch's requirement `code` (`recurrence.refusal`'s), which the fused step and the step share."""
    return Requirement[DecodeCase](code, lambda case: case.refusal is None or case.refusal.code != code,
                                   lambda case: case.refusal.detail if case.refusal is not None else code,
                                   row=row)


_LAUNCH = tuple(_launches(code, row) for code, row in (
    ('forward_mode', INHERENT), ('capability', INHERENT), ('precision', INHERENT),
    ('widths', 'roadmap: recurrence step widths')))


DELTA_DECODE: Op[DecodeCase] = Op('delta_decode', (
    Kernel('fused', (*_LAUNCH, Requirement('unfused', lambda case: case.fuses,
                                           "the caller has no conv, gates or gated norm for the fused step "
                                           "to take; its token runs the step alone", row=INHERENT,
                                           kind='handoff'))),
    Kernel('step', _LAUNCH),
    Kernel('xla', (), generic=True)), lambda case: 'xla' if forced('gated_delta_rule') == 'xla' else 'step')
"""The decode selector of both recurrence callers (a gated delta net's layer, `decode_step`'s), by what each
kernel takes (no table of GPUs): the step (`recurrence.step` alone, `decode_gated_delta_rule`) between a
caller's own conv and norm on any GPU Dew's Triton kernels run on, its preference (the fused step won at
one row alone, 0.157 ms against 0.165 on an H100, and lost at 32 and 128 rows, SE-decode-h100-d5e0ebadb),
or the portable recurrence where a caller names the rule's math (`generation.chosen`), so a reference
decodes unfused;
the fused step (`recurrence.step` with the layer's conv, gates and norm) where a call names it; else the
portable recurrence ('xla'), a kernel the token does not take handing it to the next. Both kernels update
the state in place, which JAX's Pallas gives no tangent."""


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
    # Qwen3-Next's checkpoint fuses the same projections (`fused_in_proj`).
    ("in_proj_qkvz",): ("embed", "linear"),
    ("in_proj_ba",): ("embed", "kv"),
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
    each decode token (`recurrence.round_to_bf16`): rounding to nearest
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

    reads: ClassVar[frozenset[str]] = frozenset()
    """None of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`): a recurrence reads every earlier token
    of its document, with no mask or bias to narrow it."""

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
        if self.gate_activation not in ('silu', 'sigmoid'):
            raise ValueError(
                "gate_activation must be 'silu' or 'sigmoid' (the reference's "
                f"output_gate_type), got {self.gate_activation!r}")
        dense = functools.partial(
            Dense, use_bias=False, dtype=self.dtype, precision=self.precision,
            dot_general=float32_dot_general)
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

    def _project(self, x):
        """(mixed, z, b, a) of `x`: the conv's input `[B, S, 2K + V]`, the
        queries, keys and values flat over heads in that order, z
        `[B, S, Hv*Dv]`, and b and a `[B, S, Hv]`.

        Each projection keeps its rows where the batch axes put them, as the
        rule's `manual_map` takes them: q, k, v and z with their width whole,
        which no rule splits, and b and a with their heads where the tensor
        axis splits the kernels'. Unplaced, an fsdp mesh ran the layer
        head-split with every row gathered to every device, its all-to-alls
        8.4 GiB of a Qwen3.5-9B step over FSDP 2 (docs/performance.md).
        Where the generation's row joins b and a (`dew.nn.joined`, 'delta_gates'),
        they are one GEMM, each head's two gates side by side so a tensor split
        of the heads keeps both: two narrow GEMMs of a head's width run at a few
        percent of the device (24,576 tokens of Qwen3.5-0.8B's layer: 2.006 ->
        1.945 ms fwd+bwd on an H100, 5.44 -> 5.29 on an A100 40GB)."""
        B, S, _ = x.shape

        def projected(dense, axes=RESIDUAL):
            return constrain(dense(x), axes)

        if not self.fused_in_proj:
            gates = (self.in_proj_b, self.in_proj_a)
            if joins('delta_gates', gates):
                b, a = jnp.moveaxis(constrain(joined_dense(x, gates, axis=-1), HEADS), -1, 0)
            else:
                b, a = (projected(dense, HEADS[:3]) for dense in gates)
            return projected(self.in_proj_qkv), projected(self.in_proj_z), b, a
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

    def _stored(self, operand):
        """The decode caches' dtypes: the recurrent state's as the docstring says, the conv history's
        the projections' or wider (`held_scan`)."""
        wide = at_least_fp32(operand)
        stored = (jnp.promote_types(self.dtype, jnp.bfloat16)
                  if self.dtype is not None and asks_default_precision(self.precision, configured=True)
                  else wide)
        return stored, jnp.promote_types(operand, stored)

    def _decode_kernel(self, x, operand) -> str:
        """The kernel a decode token runs (`decode_kernel`): the fused step with the layer's conv,
        gates and norm (`_fused_token`), the step alone between them (`decode_gated_delta_rule`), or
        the portable recurrence ('xla')."""
        state = jax.ShapeDtypeStruct((x.shape[0], self.num_v_heads, self.head_k_dim, self.head_v_dim),
                                     self._stored(operand)[0])
        return decode_kernel(state, tangent=FORWARD_MODE.get() or isinstance(x, JVPTracer), fuses=True)

    def _fused_hooks(self) -> recurrence.Hooks:
        """The decode token whole in the recurrence's step: the conv and its silu, the queries' and
        keys' l2 norm, the gates from their pre-activations, the delta rule and the gated norm."""
        return recurrence.Hooks(conv=recurrence.Conv('silu'), l2norm=True, decay='scalar',
                                gates=recurrence.Gates(),
                                output=recurrence.OutputNorm(self.norm_eps, self.gate_activation))

    def _fused_token(self, z, b, a, valid):
        """A decode token's conv, gates, update and gated norm in one launch (`recurrence.step`),
        `held_scan`'s `token`, each shard's rows and heads on their own (`manual_map`), as
        `decode_gated_delta_rule` places them. The conv's channels are queries, keys and values in
        turn, which a mesh splitting the heads holds by shard: there they are reordered so each
        shard's slice of them is its heads' queries, keys and values in turn, and back after."""
        B = z.shape[0]
        widths = (self.head_k_dim, self.head_k_dim, self.head_v_dim)
        heads = (self.num_k_heads, self.num_k_heads, self.num_v_heads)
        rows, split = grouped_heads((B, 1, self.num_v_heads, self.head_v_dim), self.num_k_heads)
        mesh = jax.sharding.get_abstract_mesh()
        shards = math.prod(mesh.shape[axis] for axis in mesh_axes(split))
        # Each channel's place in the shards' order: by shard, then piece, then the piece's heads there.
        first = np.cumsum([0, *(h * w for h, w in zip(heads, widths, strict=True))])
        order = np.concatenate([first[i] + np.arange(shard * h * w // shards, (shard + 1) * h * w // shards)
                                for shard in range(shards)
                                for i, (h, w) in enumerate(zip(heads, widths, strict=True))])
        inverse = np.argsort(order)
        active = jnp.ones((B,), bool) if valid is None else jnp.asarray(valid, bool)[:, 0]

        def token(mixed, history, held):
            taps = self.conv1d()[0]
            weight = self.norm.variables['params']['weight']
            dtype = self.norm.dtype if self.norm.dtype is not None else at_least_fp32(mixed.dtype)

            def local(token, history, taps, gate, beta, log_a, bias, z, weight, state, active):
                out, moved, state = recurrence.step(self._fused_hooks(), recurrence.Step(
                    state, token=token, held=history, taps=taps, gate=gate, beta=beta, log_a=log_a, bias=bias,
                    z=z, weight=weight, active=active), dtype=dtype)
                assert moved is not None, 'a step with a conv moves its history'
                return out, moved, state

            def ordered(x, axis):
                return x if shards == 1 else jnp.take(x, order, axis=axis)

            specs = (P(rows, split), P(rows, None, split), P(split, None), P(rows, split), P(rows, split),
                     P(split), P(split), P(rows, split, None), P(), P(rows, split, None, None), P(rows))
            out, history, state = manual_map(local, specs, (specs[7], specs[1], specs[9]))(
                ordered(mixed[:, 0], 1), ordered(history, 2), ordered(taps, 0), a[:, 0], b[:, 0], self.A_log,
                self.dt_bias, z[:, 0].reshape(B, self.num_v_heads, -1), weight, held, active)
            history = history if shards == 1 else jnp.take(history, inverse, axis=2)
            return out.reshape(B, 1, -1), history, state

        return token

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL):
        decode, attention_metadata = call.decode, call.metadata
        B, S, _ = x.shape
        tokens = coordinates_of(attention_metadata, B, S)
        valid = tokens.valid
        mixed, z, b, a = self._project(x)
        key_dim = self.key_features
        operand = mixed.dtype
        token, stepped = None, False
        if decode and S == 1:
            kernel = self._decode_kernel(x, operand)
            token = self._fused_token(z, b, a, valid) if kernel == 'fused' else None
            stepped = kernel == 'step'

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
        ported = S > 1 and kernel_routed(rows, keys, values, g, beta)

        def scan(convolved, *, starts, valid, held):
            query, key, value = (self._split_heads(part, width)
                                 for part, width in zip(convolved, widths, strict=True))
            state = None if held is None else held.astype(wide)
            if S > 1:
                # fla's layer hands its rule the projections' dtype, the
                # queries and keys on their own heads and beta beside them. A
                # held history can make the convolved operands vary where the
                # projections did not, so the route is asked again of them.
                # The batch's documents' firsts, built once with it (`Coordinates.firsts`).
                return chunk_gated_delta_rule(query, key, value, g, beta, state, self.chunk_size,
                                              starts=starts, max_segments=tokens.max_segments,
                                              firsts=tokens.firsts)
            # One key head serves `per_v` value heads, the reference's `repeat_interleave`.
            query, key = (repeat_kv_heads(x, self.num_v_heads) for x in (query, key))
            if stepped and held is not None:
                return decode_gated_delta_rule(query, key, value, g, beta, held,
                                               None if valid is None else valid[:, 0])
            out, final = recurrent_gated_delta_rule(query, key, value, g, beta, state)
            if held is not None and held.dtype == jnp.bfloat16:
                final = recurrence.round_to_bf16(final, key[:, 0].astype(wide), value[:, 0].astype(wide))
            return out, final

        # The conv state holds the projections' own values, so it is held in
        # their dtype at no cost; the recurrent state is held as the
        # docstring says. The first decode-mode call only allocates, the way
        # open_kv_cache's is: init_cache's dummy token must not consume a
        # position or leave state behind. The conv normalises the queries
        # and keys (`l2norm`).
        stored, conv = self._stored(operand)
        heads = ((key_dim, self.head_k_dim), (key_dim, self.head_k_dim), (self.value_features, None))
        shards = sequence_shards()
        if shards > 1:
            if decode:
                raise LayoutRefused(f"a gated delta net under a sequence axis of {shards} runs whole "
                                    f"training sequences; a decode step holds one state per row. Decode on "
                                    f"a mesh with sequence=1.")
            rule = functools.partial(chunk_gated_delta_rule, chunk_size=self.chunk_size,
                                     max_segments=tokens.max_segments)
            out = over_sequence(functools.partial(split_delta_rule, widths=widths, parts=heads,
                                                  dtype=operand if ported else wide, rule=rule),
                                shards, (mixed, g, beta, tokens.segments, tokens.starts, valid),
                                (self.conv1d()[0],))
        else:
            out = held_scan(self, mixed, self.conv1d()[0], None, scan,
                            (self.num_v_heads, self.head_k_dim, self.head_v_dim), decode=decode,
                            coordinates=tokens, state='recurrent_state', state_dtype=stored, conv_dtype=conv,
                            parts=heads, dtype=operand if ported else wide, token=token)
        if out is None:
            return self.out_proj(jnp.zeros((B, S, self.value_features), self.dtype))
        if token is not None:
            # The fused step gated and normalized its output (`_fused_hooks`).
            return self.out_proj(out)

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
    [D, K] taps and the bias (or None) as stored: every conv widens them to
    at least fp32 where it sums (the kernels in registers), so parameters
    held in bf16 are not converted in memory on each decoding step.
    """

    features: int
    kernel: int = 4
    use_bias: bool = False
    init_std: float | None = None  # None: lecun normal

    @nn.compact
    def __call__(self) -> tuple[jax.Array, jax.Array | None]:
        if self.kernel < 1:
            raise ValueError(f"a causal conv takes at least one tap, got {self.kernel}")
        weight = self.param('weight', normal_kernel(self.init_std, nn.initializers.lecun_normal())[
            'kernel_init'], (self.features, 1, self.kernel))
        bias = (self.param('bias', nn.initializers.zeros, (self.features,), jnp.float32)
                if self.use_bias else None)
        return weight[:, 0, :], bias


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
