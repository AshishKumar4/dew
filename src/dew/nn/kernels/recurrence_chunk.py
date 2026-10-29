"""The recurrence template's chunked op: a sequence's tokens through the linear recurrence of
`dew.nn.kernels.recurrence` a chunk at a time, for training and prefill (`chunk`).

Per head, over a state `S` `[Dk, Dv]`, a token updates and reads

    S' = (I - delta beta k k^T) Gamma S + iota k v^T,    o = S'^T q (+ D v)

as the decode step does (`recurrence.step`), its form the same `recurrence.Hooks`. Within a chunk of
`C` tokens the dependencies resolve in matrix form: the decay between two positions is one
exponential of their prefixes' difference (`recurrence_algebra.decays`), the delta rule's corrections
one strictly lower inverse (`recurrence_algebra.factor`, fla's WY form), and only the memory crosses chunks
(`xla_chunk_states`). Without the delta rule (`delta=False`, R = 0, Mamba2's SSD and the gated
linear attentions) the inverse is the identity and nothing is corrected.

The portable math in XLA (`chunk_delta_rule`) computes every hook: Dew's float32 reference and the
CPU path. Kernels take the rest, by what each takes and in order, no GPU named: the delta rule with one
decay a head or one per key dimension runs on the recurrence template (`mosaic_rule`) where its
launches take the call and the cell is not a declared port cell (`PORT_CELLS`), else on the port of fla
0.5.2's kernels (`fla_rule`, `fla_gdn` and `fla_kda`) on any GPU Dew's Triton kernels run on
(`GATED_DELTA`); without the delta rule, a rated decay (Mamba2's SSD) runs on the port of mamba_ssm's
kernels (`dew.nn.kernels.mamba_ssd`) on such a GPU, or on a TPU's Pallas scan (`dew.nn.kernels.ssd`)
(`SCAN`). `chunk` picks one per call and applies the hooks around it.
"""

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P

from dew.nn.attention import FORWARD_MODE, repeat_kv_heads
from dew.nn.inputs import cu_seqlens
from dew.nn.precision import at_least_fp32, l2norm
from dew.nn.sharding import grouped_heads, manual_map, rows_like

from . import fla_gdn, fla_kda, generation, mamba_ssd, recurrence, recurrence_algebra as algebra
from .generation import INHERENT, triton_runs
from .selection import Kernel, KernelRefused, Op, Requirement, select, selected
from .ssd import ssd_chunk_scan, ssd_kernel_platform
from .templates import base

CHUNK_SIZE = 64
"""Tokens per chunk of the chunked form, the reference's default
(`torch_chunk_gated_delta_rule(..., chunk_size=64)`). The chunked and the
recurrent form agree at any size; the size only trades memory for
parallelism, and 64 is what the reference's kernels assume."""

ADJOINT_CHUNKS = 16
"""Chunks replayed at once in the portable backward: bounded groups keep a token span's memories
from growing with the sequence, while every group's rows and heads run in parallel."""


def held_chunk(chunk_size: int, key_width: int) -> int:
    """The delta rule's chunk at `key_width`: `chunk_size`, at most twice the keys' width. A measured
    sizing policy, not a conditioning bound: over a long sequence (1024 tokens, the decay near one),
    64-token chunks of 16-wide keys put the final memory 1.7 to 1.8 times the token-by-token
    recurrence's error from float64 under every delta family, where chunks of 32 and 16 held it, and so
    did 64-token chunks of 32-, 64- and 128-wide keys (tests/test_recurrence_stability.py, its long
    sequence)."""
    return min(chunk_size, 2 * key_width)


def xla_chunk_states(key, k_cumdecay, out_vals, grown, last_decay, state, kept, *, last_token=None,
                     delta: bool = True):
    """The memory crossing chunks, the one sequential part of the chunked
    rule, as a `lax.scan` over the algebra's `carry`: per chunk the reference's corrected values
    `v_new = value - k_cumdecay @ S` and its write `S * exp(gc[-1]) +
    (k * exp(gc[-1] - gc))^T @ v_new` (modeling_qwen3_next.py:439-446).

    Operands are per chunk in the reference's layout, `[B, H, NC, C, ...]`,
    `grown` `[B, H, NC, C, D]` being the decay from each chunk's start, `exp(gc)` of the log decay
    cumulated within it (`recurrence_algebra.decays`),
    D one for a decay per head or Dk for one per key dimension (KDA's, whose
    rows of S decay apart, modeling_glm5_next.py:529-532), and `state`
    `[B, H, Dk, Dv]` enters the first chunk. Returns the state each chunk
    entered `[B, H, NC, Dk, Dv]`, each chunk's corrected values
    `[B, H, NC, C, Dv]` and the state leaving the last chunk.

    The write weighs each token by `last_decay` `[B, H, NC, C, D]`, the last
    row of the chunk's pairwise decay (`recurrence_algebra.decays`' `exp(gc[-1] - gc)`):
    its exponent compensated as every pair's is, where the prefix's rounded
    half alone put a long chunk's write off by its whole sum's rounding, and
    zero for every token of a packed chunk's earlier documents, which the
    write does not keep. `kept` `[B, 1, NC, 1, 1]` marks the chunks no
    document starts in, the only ones the memory they entered passes through
    (`Chunked`).

    With `last_token`, these operands describe the prefix through the penultimate real token
    (`_peeled_chunk_inputs`); the last real token runs through `peeled_carry` afterwards. The
    corrected values still cover every token, so the chunk's readout is unchanged.
    """
    carry = algebra.carry if last_token is None else functools.partial(algebra.peeled_carry, delta=delta)
    operands = key, k_cumdecay, out_vals, grown, last_decay, kept, *(last_token or ())
    final, (entered, corrected) = jax.lax.scan(
        carry, state, tuple(jnp.moveaxis(x, 2, 0) for x in operands))
    return jnp.moveaxis(entered, 0, 2), jnp.moveaxis(corrected, 0, 2), final


class Chunked(NamedTuple):
    """A sequence's chunks as the algebra takes them, `[B, H, NC, C, ...]` in the working precision: the
    scaled queries, the keys, the decay from each chunk's start `grown` (`recurrence_algebra.decays`), the
    pairs' decay, the queries' and the write's decay from the memory entered `entering` (`grown`, zero
    outside the entered document), the
    write's products `W` and `U` (`recurrence_algebra.apply`), the decay's last row `last` (zero outside
    the chunk's last document) and `kept` `[B, 1, NC, 1, 1]` (one where the memory entered passes
    through), with `padding` the tokens the last chunk was padded with."""
    query: jax.Array
    key: jax.Array
    grown: jax.Array
    pairs: jax.Array
    entering: jax.Array
    w: jax.Array
    u: jax.Array
    last: jax.Array
    kept: jax.Array
    padding: int


def chunked(query, key, value, g, beta, chunk_size: int = CHUNK_SIZE, *, starts=None, delta: bool = True,
            scale: float | None = None) -> Chunked:
    """`chunk_delta_rule`'s operands as each chunk's algebra (`Chunked`): what `carry` and `readout` read,
    and what `transition` composes. Within a chunk each document's decays are its own, `entering` is zero
    outside the document the memory entering belongs to and `last` outside the chunk's last document, so
    a chunk's transition keeps the memory entering only where `kept`."""
    work = at_least_fp32(jnp.result_type(query, key, value))
    query, key, value, g, beta = (x.astype(work) for x in (query, key, value, g, beta))
    B, S, H, Dk = key.shape
    pad = (chunk_size - S % chunk_size) % chunk_size
    query, key, value, g = (jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))) for x in (query, key, value, g))
    beta = jnp.pad(beta, ((0, 0), (0, pad), (0, 0)))
    T = S + pad
    query = query * (Dk ** -0.5 if scale is None else scale)

    def chunks(x):  # [B, S, H, ...] -> [B, H, NC, C, ...], the reference's layout
        moved = jnp.moveaxis(x, 2, 1)  # [B, H, S, ...]
        return moved.reshape(B, H, T // chunk_size, chunk_size, *moved.shape[3:])

    q_c, k_c = chunks(query), chunks(key)
    kb_c, vb_c = chunks(key * beta[..., None]), chunks(value * beta[..., None])
    document = first = None
    kept = jnp.ones((1, 1, T // chunk_size, 1, 1), work)
    if starts is not None:
        # Each token's document within its row, per chunk, [B, 1, NC, C, 1], and the document the
        # memory each chunk enters belongs to: its first token's, unless that token starts one.
        begins = jnp.pad(jnp.asarray(starts, jnp.int32), ((0, 0), (0, pad)))
        document = jnp.cumsum(begins, axis=1).reshape(B, 1, T // chunk_size, chunk_size, 1)
        entered_document = document[..., :1, :] - begins.reshape(document.shape)[..., :1, :]
        first = document == entered_document
        kept = (document[..., -1:, :] == entered_document).astype(work)
    # The decay from each chunk's start, exp of the reference's `g = g.cumsum(dim=-1)`
    # (modeling_qwen3_next.py:417), and the decay `exp(gc[s] - gc[t])` between positions s >= t, zero
    # above the diagonal: [B, H, NC, C, D], [B, H, NC, C, C, D].
    grown, pairs = algebra.decays(chunks(g), document)
    entering = grown if first is None else grown * first
    # The reference's `attn + I` operator after its row correction loop; without the delta rule nothing
    # is corrected, and each token writes its gated value.
    inverse = algebra.factor(algebra.intra(k_c, kb_c, pairs)) if delta else None
    w, u = algebra.apply(inverse, kb_c, vb_c, entering)
    return Chunked(q_c, k_c, grown, pairs, entering, w, u, pairs[..., -1, :, :], kept, pad)


def chunk_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE, *, starts=None,
                     delta: bool = True, scale: float | None = None, dtype=None):
    """The chunked delta rule over [B, S, H, D] operands with the log decay
    `g` `[B, S, H, D]`, D one for a decay per head (the gated delta rule) or
    Dk for one per key dimension (Kimi Delta Attention); returns
    `(output [B, S, H, Dv], final_state [B, H, Dk, Dv])`, the output in the
    queries', keys' and values' common dtype and the memory in at least fp32,
    as the port returns them (`fla_rule`), computed in at least fp32. The
    references' sequential correction is the inverse `recurrence_algebra.factor`
    builds from its diagonal blocks.

    Only the memory crosses chunks, so only its recurrence runs chunk after
    chunk (`xla_chunk_states`). Each chunk's output, the references'
    `attn_inter + attn @ v_new`, reads the state that chunk entered and its
    corrected values, so it is computed for every chunk at once after the
    recurrence. The two decays differ only in how a pair of positions is
    weighed (`recurrence_algebra.paired`). It is the portable math: a GPU runs the gated rule on
    fla's kernels (`chunk_gated_delta_rule`), and this is their test oracle.

    Without the `delta` rule (R = 0: Mamba2's SSD, the gated linear
    attentions) nothing is corrected: the inverse is the identity and each
    token writes `beta k v^T`, beta its write gate. The queries are scaled by
    `scale` in the working precision, `Dk^-1/2` where None, and the output is
    returned in `dtype` where one is given.

    `starts` `[B, S]` marks packed documents' first tokens, each from a zero
    memory: within a chunk each document's decays are its own (`recurrence_algebra.decays`),
    a token reads the memory the chunk entered only in the document that
    memory belongs to, and the chunk's write keeps only its last document's
    tokens (the decay's last row is zero for the others), the memory it
    entered passing through only where no document starts in it. A document
    computes as it would alone.

    The reverse rule erases before differentiating a token's decay, within independently replayed chunks
    (`_chunk_delta_backward`). Differentiating the WY products instead cancelled independently
    rounded paths through the inverse, pairwise decays and readout: under aligned keys their RMS
    magnitudes summed to 16.2 times the resulting decay gradient on one draw. Over 40 draws its
    fp32 error was 1.883 times the token recurrence's, and 2.941 at 128-wide keys with 64-token
    chunks; regrouped it is 1.068 and 0.989. The vector decay's is 1.021, and the formed gates'
    gate, log_a and bias 1.053, 1.038 and 1.023 over 320 draws. All arithmetic remains fp32.
    Only the chunk boundaries couple across the sequence; bounded groups schedule each chunk's C-step
    token scans, not one scan over the sequence's T. Forward mode takes the chunked math's own tangent
    (`_portable_rule`), in any context.
    """
    dtype = jnp.result_type(query, key, value) if dtype is None else jnp.dtype(dtype)
    work = at_least_fp32(jnp.result_type(query, key, value))
    B, _, H, Dk = key.shape
    # A zero state like the values: inside a shard_map that checks varying axes the chunk loop's
    # carry varies as the operands it takes in do, and under an Explicit axis its rows are placed as
    # theirs are.
    state = (rows_like(jnp.zeros_like(value, work, shape=(B, H, Dk, value.shape[-1])), value) if state is None
             else state.astype(work))
    query, key, value, g, beta = (x.astype(work) for x in (query, key, value, g, beta))
    scale = Dk ** -0.5 if scale is None else scale
    if query.shape[1] == 0:
        return _chunk_delta_rule(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype)
    rule = _portable_rule(chunk_size, delta, scale, jnp.dtype(dtype))
    return rule(query, key, value, g, beta, state, starts)


def _chunk_delta_forward(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype):
    """The chunked readout and peeled memory write, and the operands their reverse rule keeps."""
    B, S, H, _ = key.shape
    Dv = value.shape[-1]
    steps = chunked(query, key, value, g, beta, chunk_size, starts=starts, delta=delta, scale=scale)
    T = S + steps.padding
    # XLA:CPU workaround, on a CPU alone (row 1701), kept because a CPU user's
    # results are wrong without it (bible §5): under a jitted scan over the
    # layers the chunk loop's zero initial state came back with other values in
    # most processes (GLM-5-Next off by 8.7, 10.1 or NaN, byte-identical
    # modules). Without it a scanned GLM-5-Next's whole output is NaN at base
    # 0b28dd7d7 (test_layer_stack, probe a079ff009). The barrier materializes
    # the zero. Remove it when XLA:CPU no longer returns an uninitialized scan
    # carry here.
    if generation.placed_platform() == 'cpu':
        state = jax.lax.optimization_barrier(state)
    grown, last, kept, last_token = _peeled_chunk_inputs(steps, value, g, beta, starts, chunk_size)
    entered, v_corrected, state = xla_chunk_states(
        steps.key, steps.w, steps.u, grown, last, state, kept, last_token=last_token, delta=delta)
    # The pairs are zero above the diagonal, so this is the reference's inclusive
    # lower `masked_fill(triu(1), 0)`.
    core = algebra.readout(steps.query, steps.entering, entered, steps.key, steps.pairs, v_corrected)
    # core: [B, H, NC, C, Dv] -> [B, H, NC*C, Dv] -> [B, S, H, Dv]
    core = jnp.moveaxis(core.reshape(B, H, T, Dv), 1, 2)[:, :S]
    return (core.astype(dtype), state), (steps, entered)


def _chunk_delta_rule(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype):
    """The chunked primal with its ordinary tangent, used by forward-mode callers."""
    return _chunk_delta_forward(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype)[0]


def _chunk_rows(x, chunk_size: int):
    """`[B, S, H, ...]` as `[B, H, NC, C, ...]`, padded with exact zero tokens."""
    B, S, H = x.shape[:3]
    pad = (chunk_size - S % chunk_size) % chunk_size
    x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
    return jnp.moveaxis(x, 2, 1).reshape(B, H, -1, chunk_size, *x.shape[3:])


def _chunk_sequence(x, length: int):
    """`[B, H, NC, C, ...]` back to `[B, S, H, ...]`, removing the padding."""
    B, H, chunks, width = x.shape[:4]
    return jnp.moveaxis(x.reshape(B, H, chunks * width, *x.shape[4:]), 1, 2)[:, :length]


def _chunk_token(x, positions):
    """One token per chunk of `[B, H, NC, C, ...]`, its local position `[NC]`, without its C axis."""
    index = positions.reshape(1, 1, -1, 1, *((1,) * (x.ndim - 4)))
    index = jnp.broadcast_to(index, (*x.shape[:3], 1, *x.shape[4:]))
    return jnp.take_along_axis(x, index, axis=3)[:, :, :, 0]


def _peeled_chunk_inputs(steps, value, g, beta, starts, chunk_size):
    """The prefix's decay row and kept flag, and the last real token of every chunk.

    The ragged last chunk peels its last unpadded token. An empty prefix has unit decay, no
    writes and passes its entered memory on; a reset on the peeled token then drops it there.
    Prefix `kept` sees only starts before the peeled token, while its pairwise decay row
    already excludes previous documents. Thus a document starting on the boundary belongs to
    the final token, not the WY prefix."""
    count = steps.key.shape[2]
    final = jnp.minimum(chunk_size, value.shape[1] - jnp.arange(count) * chunk_size) - 1
    penultimate = jnp.maximum(final - 1, 0)
    nonempty = final[None, None, :, None] > 0
    grown = jnp.where(nonempty, _chunk_token(steps.grown, penultimate), 1.0)[..., None, :]
    last = jnp.where(nonempty[..., None], _chunk_token(steps.pairs, penultimate), 0.0)
    if starts is None:
        kept = jnp.ones((1, 1, count, 1, 1), value.dtype)
        reset = jnp.zeros(steps.key.shape[:3], bool)
    else:
        begins = jnp.broadcast_to(jnp.asarray(starts, bool)[..., None], value.shape[:3])
        begins = _chunk_rows(begins, chunk_size)
        prefix = jnp.arange(chunk_size)[None, None, None, :] < final[None, None, :, None]
        kept = (~jnp.any(begins & prefix, axis=-1))[..., None, None].astype(value.dtype)
        reset = _chunk_token(begins, final)
    key = _chunk_token(steps.key, final)
    value, g, beta = (_chunk_token(_chunk_rows(x, chunk_size), final) for x in (value, g, beta))
    return grown, last, kept, (key, value, g, beta, reset)


def _chunk_delta_residual(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype):
    """`_kept_residual`'s, or where the block's remat rung recomputes the delta rule's memory (`RECOMPUTED`,
    'delta_memory') the operands alone, the backward computing the memory and what it derives again, as the
    port and the template drop theirs: `(out, (kept, operands))`, one of them None."""
    if "delta_memory" in RECOMPUTED.get():
        out = _chunk_delta_rule(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype)
        return out, (None, (query, key, value, g, beta, state, starts))
    out, kept = _kept_residual(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype)
    return out, (kept, None)


def _kept_residual(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype):
    """Keep chunk boundaries and the readout's memory-gradient operands, not per-token memories."""
    out, (steps, entered) = _chunk_delta_forward(
        query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype)
    grown, last, kept, last_token = _peeled_chunk_inputs(steps, value, g, beta, starts, chunk_size)
    transition, _ = algebra.peeled_transition(
        steps.key, steps.w, steps.u, grown, last, kept, last_token, delta=delta)
    scores = algebra.paired("scores", steps.query, steps.key, steps.pairs)
    resets = (jnp.zeros(value.shape[:2], bool) if starts is None else jnp.asarray(starts, bool))
    resets = _chunk_rows(jnp.broadcast_to(resets[..., None], value.shape[:3]), chunk_size)
    residual = (steps.query, steps.key, *(_chunk_rows(x, chunk_size) for x in (value, g, beta)),
                resets, entered, transition, steps.query * steps.entering, steps.w, scores)
    return out, residual


def _chunk_delta_backward(chunk_size, delta, scale, dtype, residual, cotangent):
    """The token adjoint within independently replayed chunks; only their memory cotangents cross
    boundaries. Replay C tokens from each entered memory, derive their terminal cotangents from
    `readout_entered_gradient` and the transitions, then reverse C tokens. Replaying keeps a token's
    erasure beside its decay, rather than cancelling separately rounded WY derivatives afterwards.
    Checkpoint every SUB tokens, then replay one span at a time inside the reverse scan. Groups of
    ADJOINT_CHUNKS bound the live span and checkpoints independently of the sequence's length.
    Replays and adjoints preserve the token recurrence's arithmetic and order. Where the forward kept
    its operands alone (`_chunk_delta_residual`), the rest is computed again first."""
    kept, operands = residual
    if kept is None:
        query, key, value, g, beta, state, starts = operands
        _, kept = _kept_residual(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype)
    del chunk_size, dtype
    query, key, value, g, beta, resets, entered, transition, decayed_query, weights, scores = kept
    cotangent, final_cotangent = cotangent
    length = cotangent.shape[1]
    cotangent = _chunk_rows(cotangent.astype(query.dtype), query.shape[3])
    tokens = query, key, value, g, beta, resets

    width = query.shape[3]
    span_width = min(algebra.SUB, width)
    spans = (width + span_width - 1) // span_width

    def span_major(x):
        x = jnp.moveaxis(x, 3, 0)
        x = jnp.pad(x, [(0, spans * span_width - width)] + [(0, 0)] * (x.ndim - 1))
        return x.reshape(spans, span_width, *x.shape[1:])

    def token(state, query, key, value, g, beta, reset):
        state = jnp.where(reset[..., None, None], 0.0, state)
        # A decay a head broadcast over the keys before the update, as the token-by-token recurrence's
        # (`recurrence._gates`): its gradient then sums each key's paths first, where a `[..., 1]` decay's
        # took one reduction over every key and value, 3.3 times the recurrence's distance from float64
        # (test_recurrence_mosaic_gradient's ragged fp32 case, after the gate compiled both arms without
        # YNNPACK's fusions).
        decay = jnp.broadcast_to(jnp.exp(g), key.shape)
        return recurrence.update(state, query, key, value, decay, beta[..., None], delta=delta)

    def replay(state, inputs):
        _, final = token(state, *inputs)
        return final, state

    def checkpoint(state, inputs):
        final, _ = jax.lax.scan(replay, state, inputs)
        return final, state

    source = algebra.readout_entered_gradient(decayed_query, weights, scores, cotangent)

    def across(state_gradient, inputs):
        transition, source = inputs
        initial = algebra.portable("compose", jnp.swapaxes(transition, -1, -2), state_gradient) + source
        return initial, state_gradient

    _, terminal = jax.lax.scan(across, final_cotangent, (jnp.moveaxis(transition, 2, 0),
                                                        jnp.moveaxis(source, 2, 0)), reverse=True)
    terminal = jnp.moveaxis(terminal, 0, 2)

    def adjoint(state_gradient, inputs):
        state, out_gradient, query, key, value, g, beta, reset = inputs
        _, pull = jax.vjp(lambda state, q, k, v, g, beta: token(state, q, k, v, g, beta, reset),
                          state, query, key, value, g, beta)
        state_gradient, *gradients = pull((out_gradient, state_gradient))
        return state_gradient, tuple(gradients)

    def reverse_span(state_gradient, inputs):
        checkpoint, out_gradient, *tokens = inputs
        _, before = jax.lax.scan(replay, checkpoint, tuple(tokens))
        return jax.lax.scan(adjoint, state_gradient, (before, out_gradient, *tokens), reverse=True)

    def within(inputs):
        entered, terminal, cotangent, *tokens = inputs
        span_tokens = tuple(map(span_major, tokens))
        _, checkpoints = jax.lax.scan(checkpoint, entered, span_tokens)
        initial, gradients = jax.lax.scan(
            reverse_span, terminal, (checkpoints, span_major(cotangent), *span_tokens), reverse=True)

        def rows(x):
            return jnp.moveaxis(x.reshape(spans * span_width, *x.shape[2:])[:width], 0, 3)

        return initial, tuple(map(rows, gradients))

    count = entered.shape[2]
    group_width = min(ADJOINT_CHUNKS, count)
    groups = (count + group_width - 1) // group_width

    def grouped(x):
        padding = [(0, 0)] * x.ndim
        padding[2] = (0, groups * group_width - count)
        x = jnp.pad(x, padding)
        return jnp.moveaxis(x.reshape(*x.shape[:2], groups, group_width, *x.shape[3:]), 2, 0)

    def ungrouped(x):
        x = jnp.moveaxis(x, 0, 2)
        return x.reshape(*x.shape[:2], groups * group_width, *x.shape[4:])[:, :, :count]

    initial, gradients = jax.lax.map(
        within, tuple(map(grouped, (entered, terminal, cotangent, *tokens))))
    dq, dk, dv, dg, dbeta = (_chunk_sequence(ungrouped(x), length) for x in gradients)
    return dq * scale, dk, dv, dg, dbeta, ungrouped(initial)[:, :, 0], None


@functools.cache
def _portable_rule(chunk_size: int, delta: bool, scale: float, dtype: jnp.dtype) -> Callable:
    """The portable rule over `(query, key, value, g, beta, state, starts)` differentiated at every order
    (`generation.linear_tangent`): reverse mode runs its reverse rule (`_chunk_delta_residual`,
    `_chunk_delta_backward`), forward mode (`jax.jvp`, `jax.linearize`) the chunked math's own tangent,
    in any context: a custom VJP alone refused a plain `jax.jvp` outside the attention's forward-mode
    context (row 1834)."""
    def forward(query, key, value, g, beta, state, starts):
        return _chunk_delta_residual(query, key, value, g, beta, state, chunk_size, starts, delta, scale,
                                     dtype)

    def backward(saved, operands, cotangent):
        # Every operand's cotangent, as the portable rule's pullback gives them (`linear_tangent` checks the
        # trees agree): `starts` has no tangent space, its cotangent float0's zero, or none without it.
        starts = operands[6]
        cotangents = _chunk_delta_backward(chunk_size, delta, scale, dtype, saved, cotangent)[:6]
        return (*cotangents, None if starts is None else np.zeros(starts.shape, jax.dtypes.float0))

    def portable(query, key, value, g, beta, state, starts):
        return _chunk_delta_rule(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype)

    return generation.linear_tangent(forward, backward, portable)


def chunk_gated_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE, *,
                           starts=None, max_segments: int | None = None, floor: float | None = None,
                           firsts=None):
    """`torch_chunk_gated_delta_rule` (modeling_qwen3_next.py:374-453) with one
    log decay per head, `g` `[B, S, H]`, or `chunk_kimi_delta_attention`
    (modeling_glm5_next.py:482-578) with one per key dimension, `[B, S, H,
    Dk]`: the delta rule's row of the chunked op (`chunk`), its queries
    scaled by `Dk^-1/2`. Its queries and keys may sit on fewer heads than the
    values, each serving `Hv // Hk` of them, as fla takes them.

    Its kernels (`GATED_DELTA`: the recurrence template, or fla 0.5.2's
    chunked rule on Dew's port of its kernels, `fla_rule`, on fla's numbers)
    take it where they take these operands; elsewhere the portable math
    (`chunk_delta_rule`), in `chunk_size` chunks, which the kernels' 64 need
    not match: the rule's value does not depend on them. Each takes the
    operands as they come, which a layer rounds for the route it asks of
    them first (`kernel_routed`: the projections' dtype for a kernel, as
    fla's layer hands them over, the wide one for the math).

    `starts` `[B, S]` marks packed documents' first tokens, each starting
    from a zero memory (`state` continues a row's first document), at most
    `max_segments` a row (`Coordinates.max_segments`): the port runs each
    document as a sequence of its own (`fla_rule`), its kernels' grids sized
    by that bound, and the portable math resets its recurrence at each start,
    which needs none. `floor` is the least log decay a token the layer declares (`recurrence.Hooks.floor`:
    Kimi delta attention's lower bound), which the recurrence template's factored form of a decay a key
    dimension reads.
    """
    hooks = recurrence.Hooks(decay='vector' if g.ndim == 4 else 'scalar', scale=key.shape[-1] ** -0.5,
                             floor=floor)
    return chunk(hooks, Chunk(query, key, value, log_decay=g, beta=beta, state=state, starts=starts,
                              firsts=firsts), chunk_size=chunk_size, max_segments=max_segments)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Chunk(recurrence.Gated):
    """A sequence's operands for the chunked op (`chunk`), each `[B, S, ...]`; a hook's are None where
    it is off, as a decode token's are (`recurrence.Step`).

    `query` and `key` `[B, S, Hk, Dk]`, `value` `[B, S, Hv, Dv]`, each key head serving `Hv // Hk`
    value heads. `log_decay` is `[B, S, Hv]` (scalar) or `[B, S, Hv, Dk]` (vector), or with the gates
    `gate` `[B, S, Hv]`, `log_a` and `bias` `[Hv]`. `beta` `[B, S, Hv]` is the write gate (a raw beta
    with the gates), `skip` `[Hv]` the skip's `D`, `z` `[B, S, Hv, Dv]` and `weight` `[Dv]` the
    epilogue's, `rate` `[Hv]` a rated decay's, at most zero (`recurrence.Hooks.rated`). `state`
    `[B, Hv, Dk, Dv]` is the memory entering, zero where None, and `starts` `[B, S]` marks packed
    documents' first tokens, each from a zero memory, `firsts` their rows' documents' first slots as the
    batch built them (`dew.batch.Coordinates.firsts`; the kernels sort them from `starts` where None)."""
    query: jax.Array
    key: jax.Array
    value: jax.Array
    state: jax.Array | None = None
    starts: jax.Array | None = None
    firsts: jax.Array | None = None


def chunk(hooks: recurrence.Hooks, operands: Chunk, *, chunk_size: int = CHUNK_SIZE,
          max_segments: int | None = None, dtype=None, mainloop: str | None = None):
    """`(output [B, S, Hv, Dv], final memory [B, Hv, Dk, Dv])` of a sequence through the recurrence of
    `hooks` (`recurrence.Hooks`, the decode step's form), the output in `dtype` (the queries', keys'
    and values' common dtype where None) and the memory in at least fp32.

    The prologue (the l2 norm, the queries' scale, the gates' activations) and the epilogue (the
    skip, the gated RMS norm) are the decode step's arithmetic (`recurrence.step_reference`) on every
    token, in at least fp32. The recurrence between them runs, for the delta rule with a decay, on the
    kernel the delta rule's selector gives each product (`GATED_DELTA`, `port_kernel`: the recurrence
    template, or the port of fla's kernels, `fla_rule`); without the delta rule, a rated decay (Mamba2's
    SSD) runs on the kernel the scan's selector gives each product (`SCAN`: the recurrence template, the port
    of mamba_ssm's kernels, or a TPU's Pallas scan where its tiling covers the geometry); else on the
    portable math (`chunk_delta_rule`, in `chunk_size` chunks, the delta rule's no longer than `held_chunk`,
    which a kernel's need not match), which `generation.chosen` names for either selector ('xla').
    Packed documents (`starts`, at most `max_segments` a row) each compute as they would alone. A conv
    runs before the op: it takes the conv's output. The operands the hooks read are checked as the
    decode step's (`recurrence.read`), and a rated decay is taken on its declared domain, a contraction.
    `mainloop` 'mosaic' runs the rule on the recurrence template's Mosaic GPU mainloop (`mosaic_rule`),
    every decay with or without the delta rule."""
    if hooks.conv is not None:
        raise ValueError("the chunked op takes the conv's output: its queries, keys and values")
    given = recurrence.read(hooks, operands, optional=('state',))
    query, key, value = given['query'], given['key'], given['value']
    if value.shape[2] % key.shape[2]:
        raise ValueError(f"{value.shape[2]} value heads over {key.shape[2]} key heads: each key head serves "
                         "a whole number of value heads")
    ranks = {'scalar': 3, 'vector': 4}
    if 'log_decay' in given and given['log_decay'].ndim != ranks[hooks.decay]:
        raise ValueError(f"a {hooks.decay} decay is {ranks[hooks.decay]}-dimensional, [B, S, Hv"
                         f"{', Dk' * (hooks.decay == 'vector')}], not {given['log_decay'].shape}")
    dtype = jnp.result_type(query, key, value) if dtype is None else jnp.dtype(dtype)
    work = at_least_fp32(jnp.result_type(query, key, value))
    if hooks.l2norm:
        query, key = (l2norm(x.astype(work)) for x in (query, key))
    scale = (key.shape[-1] ** -0.5 if hooks.l2norm else 1.0) if hooks.scale is None else hooks.scale
    g, beta = recurrence.log_gates(hooks, given, work)
    g = jnp.zeros(value.shape[:3], work) if g is None else g
    beta = jnp.ones(value.shape[:3], work) if beta is None else beta
    state, starts, firsts = operands.state, operands.starts, operands.firsts
    epilogue = hooks.skip or hooks.output is not None
    if mainloop is not None:
        if mainloop != 'mosaic':
            raise ValueError(f"no {mainloop!r} mainloop: the recurrence template's is 'mosaic'")
        # Unrounded where an epilogue follows, as on the portable math.
        out, final = template_rule(query, key, value, g, beta, state, starts, firsts,
                                   max_segments=max_segments, scale=scale, delta=hooks.delta, wide=epilogue,
                                   floor=hooks.floor)
        out = recurrence.skipped(hooks, out, value, given, work)
    elif hooks.rated and not hooks.delta and not hooks.l2norm and hooks.scale is None:
        # Each of its routes adds the skip, the port in its kernels.
        out, final = _rated_scan(hooks, query, key, value, g, beta, given, state, starts, firsts, chunk_size,
                                 work, max_segments=max_segments)
    else:
        # The delta rule's selector decides every form (`GATED_DELTA`): the forms its kernels do not run hand
        # over to the portable math there, reported as the call's route.
        def product_kernel(product: str) -> str:
            return port_kernel(query, key, value, g, beta, state, starts, max_segments=max_segments,
                               scale=scale, wide=epilogue, floor=hooks.floor, product=product,
                               delta=hooks.delta, decayed=hooks.decay != 'none')

        def run(kernel, query, key, value, g, beta, state, starts, firsts):
            if kernel == 'mosaic_gpu':
                return template_rule(query, key, value, g, beta, state, starts, firsts,
                                     max_segments=max_segments, scale=scale, delta=hooks.delta, wide=epilogue,
                                     floor=hooks.floor)
            if kernel == 'fla':
                # Unrounded where an epilogue follows, as on the portable math.
                return fla_rule(query, key, value, g, beta, state, starts=starts, max_segments=max_segments,
                                scale=scale, wide=epilogue, firsts=firsts)
            group = value.shape[2] // key.shape[2]
            if group > 1:
                query, key = (jnp.repeat(x, group, axis=2) for x in (query, key))
            size = held_chunk(chunk_size, key.shape[-1]) if hooks.delta else chunk_size
            return chunk_delta_rule(query, key, value, g.reshape(*value.shape[:3], -1), beta, state, size,
                                    starts=starts, delta=hooks.delta, scale=scale, dtype=work)

        out, final = _per_product(product_kernel, run, query, key, value, g, beta, state, starts, firsts)
        out = recurrence.skipped(hooks, out.astype(work), value, given, work)
    # The epilogue in the working precision, and the output rounded once.
    return recurrence.normed(hooks, out, given, work).astype(dtype), final


@dataclasses.dataclass(frozen=True)
class PortCase:
    """A delta rule's call as the port's selector reads it (`GATED_DELTA`): whether it is traced for
    reverse mode alone (`reverse`; `FORWARD_MODE` marks `jax.jvp`'s traces), whether a caller's
    `shard_map` that checks varying axes holds operands that vary (`varying`), the operands' common
    `dtype`, the keys' width (`key_width`), the recurrence template's implementations bound to the call
    (`call`, `templates.base.Call`; None where the caller asks of the port alone), the
    instruction family of the device it runs on (`family`, `generation.family`; None off a GPU), and the
    product it is selected for (`product`: 'forward' where no gradient is taken, 'training' where one is,
    `_by_product`), and the recurrence's form: whether it is the delta rule (`delta`) and decays
    (`decayed`)."""

    reverse: bool
    varying: bool
    dtype: np.dtype
    key_width: int
    call: base.Call | None = None
    family: str | None = None
    product: str = 'forward'
    delta: bool = True
    decayed: bool = True

    @property
    def cell(self) -> tuple[str | None, str]:
        """The call's cell, its device's instruction family and its product (`selection.Celled`,
        `PORT_CELLS`)."""
        return self.family, self.product

    @property
    def widest(self) -> int:
        """The widest keys the port takes in the call's dtype."""
        return fla_gdn.WIDEST_KEYS if self.dtype.itemsize == 2 else fla_gdn.WIDEST


_VARYING = ("a shard_map that checks varying axes holds the operands, and a Pallas call states no varying "
            "type for its results")

PORT_SUPPORTS: tuple[Requirement[PortCase], ...] = (
    Requirement('gated', lambda case: case.delta and case.decayed,
                "a recurrence other than the gated delta rule, which the rule's kernels run (fla's "
                "chunk_gated_delta_rule and chunk_kda, the template's delta-rule mainloop): no delta rule (a "
                "gated linear attention, a rated scan with its keys normed or scaled) or no decay (the plain "
                "delta rule), which the portable math runs", row=INHERENT, kind='handoff'),
    Requirement('forward_mode', lambda case: case.reverse,
                "forward_mode_attention traces for jax.jvp, which the port's custom VJP gives no tangent",
                row=INHERENT),
    Requirement('varying', lambda case: not case.varying, _VARYING, row='ledger 822'),
    Requirement('float64', lambda case: case.dtype != jnp.float64,
                "float64 operands, whose rule is the reference math itself", row=INHERENT),
    Requirement('key_width', lambda case: case.key_width <= case.widest,
                lambda case: f"keys {case.key_width} wide; the port takes at most {case.widest} in "
                             f"{case.dtype.name}", row=INHERENT),
    Requirement('triton', lambda case: triton_runs(),
                "no GPU Dew's Triton kernels run on (sm80 or later): off one the port's kernels interpret, "
                "as the CPU suite runs them where a call names the port", row=INHERENT, kind='measured'),
)
"""What the port (`fla_rule`) needs of a rule's call: the gated delta rule, a decay with the delta rule (the
other forms hand over to the portable math, the call's route reporting it); a GPU Dew's Triton kernels run on,
any of them (no target's row names it: a GPU no row named took the portable math, 140 to 155 ms a training
step where the port took 3.1, template review 2026-10-26 P1-1), and every shape, dtype, chunk size and mesh,
except inside `jax.jvp`'s traces, which the port's custom VJP gives no tangent; inside a caller's `shard_map`
that checks varying axes over operands that vary, where a Pallas call, which states no varying type for its
results, cannot run; float64 operands, whose rule is the reference math itself; and keys wider than the port
takes in their dtype. Packed documents take the port at any bound: it stores the memories by where they pass,
as many as the tokens' chunks, a row's and one, whatever the bound (`chunked_launch.Sequences`)."""

PORT_CELLS: dict[tuple[str | None, str], tuple[str, str]] = {
    ('wgmma', 'forward'): ('fla', "on an H100 80GB HBM3, 16-bit, the port's forward 0.440 ms against the "
                                  "template's 0.559 on Qwen3.5-27B's gated delta net and 0.276 against 0.489 "
                                  "on Qwen3-Next's (SE-family-h100-b0980aaff; on an H100 PCIe 0.602 against "
                                  "0.776, SE-family-h100-e0de17766)"),
    ('mma.sync', 'forward'): ('fla', "on an A100 SXM4 40GB, 16-bit, the port's forward 0.892 ms against the "
                                     "template's 1.213 on Qwen3.5-27B's gated delta net and 0.690 against "
                                     "0.942 on Qwen3-Next's (SE-family-a100-1a31d279a)"),
}
"""The cells, each an instruction family and a product (`PortCase.cell`), where a receipt shows the fla
port faster than the recurrence template, each its old route's kernel and that receipt: the delta rule's
selector runs the port on them where it takes the call (`Op.declared`, owner, bible da64ae5); the list
shrinks as the template wins its cells, and no cell is picked or ranked at run time. Training is the
template's: on an H100 its fast backward took 1.851 ms against the port's 2.400 (SE-family-h100-b0980aaff;
2.56 against 3.14 on an H100 PCIe, SE-family-h100-e0de17766); on an A100 SXM4 40GB its forward and gradient
4.56 ms against the port's 4.83 on Qwen3.5-27B's gated delta net and 3.44 against 4.69 on Qwen3-Next's, its
keys launch a key block a program and its query/key launch's memory sum a product
(JS-train80-a100-db2c4aca0)."""

def _trains(call: base.Call | None) -> bool:
    """Whether the recurrence template's gradient runs the call (`Mosaic.trains`)."""
    from .templates.recurrence import template

    if call is None or not call.eligible('mosaic_gpu'):
        return False
    implementation, bound = call.bound['mosaic_gpu']
    assert isinstance(implementation, template.Mosaic)
    return implementation.trains(bound, call.target)


class Templated(Protocol):
    """A case the recurrence template's requirements read (`TEMPLATE_SUPPORTS`): the template's
    implementations bound to the call (`call`), the instruction family of the device it runs on (`family`,
    None off a GPU) and the product it is selected for (`product`)."""

    @property
    def call(self) -> base.Call | None: ...

    @property
    def family(self) -> str | None: ...

    @property
    def product(self) -> str: ...


TEMPLATE_SUPPORTS: tuple[Requirement[Templated], ...] = (
    Requirement('launch', lambda case: case.call is not None and case.call.eligible('mosaic_gpu'),
                "the recurrence template's launches take no call of these dtypes or widths, or no config "
                "fits this device, or no Mosaic bundle claims its instruction family", row='ledger 1294'),
    Requirement('device', lambda case: case.family is not None,
                "no GPU runs the call: off one, the mainloop runs interpreted where a call names it",
                row=INHERENT, kind='measured'),
    Requirement('backward', lambda case: case.product != 'training' or _trains(case.call),
                "a gradient the template's fast backward does not take: fp32 operands, no delta rule, a "
                "decay a key dimension, or launches past the device's registers or shared memory at the "
                "call's tiles (`mainloop.backward_holds`)", row='ledger 1294'),
)
"""What the recurrence template's Mosaic GPU mainloop (`templates.recurrence.template.Mosaic`) needs of a
call under either selector (`GATED_DELTA`, `SCAN`): its own launches' call (`Template.eligible`: dtypes,
widths, a decay a key dimension's floor, a bundle for the device's family, a config that fits), a GPU, and for
a training product its fast backward (`Mosaic.trains`; the rule's gradient takes no other, row 1926)."""

MOSAIC_SUPPORTS: tuple[Requirement[PortCase], ...] = (*PORT_SUPPORTS[1:3], *TEMPLATE_SUPPORTS)
"""What the recurrence template needs of a delta rule's call: the port's trace and placement (its custom VJP,
a Pallas call's results) and `TEMPLATE_SUPPORTS`, in every form its mainloop implements, the delta rule with
a decay or without one and the forms without the delta rule alike (row 1963: the port's 'gated' form was the
template's too). A declared port cell runs the port first where the port takes the call (`PORT_CELLS`,
`GATED_DELTA`'s declared cells)."""

MOSAIC = Kernel('mosaic_gpu', MOSAIC_SUPPORTS)


GATED_DELTA: Op[PortCase] = Op('gated_delta_rule',
                               (MOSAIC, Kernel('fla', PORT_SUPPORTS), Kernel('xla', (), generic=True)),
                               declared=PORT_CELLS)
"""The delta rule's selector: on a declared port cell (`PORT_CELLS`) the port where it takes the call;
otherwise the first of its kernels that takes the call, by what each takes (no table of GPUs): the
recurrence template at its derived tiles, else the port on any GPU Dew's Triton kernels run on, else the
portable math."""


def port_kernel(query, key, value, g, beta, state=None, starts=None, *, max_segments: int | None = None,
                scale: float | None = None, wide: bool = False, floor: float | None = None,
                product: str = 'forward', report: bool = True, delta: bool = True,
                decayed: bool = True) -> str:
    """The kernel the delta rule's selector (`GATED_DELTA`) gives a call over these operands for `product`
    ('forward', 'training'), of the form `delta` and `decayed` state (`PortCase`), the template's
    implementations bound to it, on the target (`generation`'s, an assumed one's within `assumed`):
    reported as its route, or not where not `report` (`selected`). A kernel a call names
    (`generation.chosen`) insists on it (`selection.KernelRefused`: the port named over varying operands),
    but for a form the rule's kernels do not run, which the portable math takes."""
    from .templates.recurrence import template

    call = base.call((template.MOSAIC,), query, key, value, g, beta, state, starts,
                     max_segments=max_segments, scale=scale, delta=delta, wide=wide, floor=floor)
    family = generation.family(generation.device_generation()) if generation.backend() == 'gpu' else None
    case = PortCase(reverse=not FORWARD_MODE.get(), varying=bool(jax.typeof(value).mat.varying),
                    dtype=jnp.dtype(jnp.result_type(query, key, value)), key_width=key.shape[-1], call=call,
                    family=family, product=product, delta=delta, decayed=decayed)
    return select(GATED_DELTA, case).kernel if report else selected(GATED_DELTA, case).name


def _per_product(kernel_of: Callable[[str], str], run, *operands):
    """`run(kernel, *operands)` on the kernel a selector gives each product (`kernel_of('forward')`,
    `kernel_of('training')`): one run where both take the same kernel, else each product its own
    (`_by_product`: a cell whose forward and training differ, `PORT_CELLS`, `SCAN_CELLS`, or a template that
    trains no such call). A kernel named for the call that runs its forward but not its gradient refuses
    where a gradient is traced, not a forward alone."""
    forward = kernel_of('forward')
    try:
        training = kernel_of('training')
    except KernelRefused as refusal:
        return _by_product(functools.partial(run, forward), functools.partial(_refusing, refusal), *operands)
    if forward == training:
        return run(forward, *operands)
    return _by_product(functools.partial(run, forward), functools.partial(run, training), *operands)


def _refusing(refusal: KernelRefused, *operands):
    """A product whose kernel refused the call (`_per_product`'s training): raises `refusal` where it is
    traced."""
    del operands
    raise refusal


def _by_product(forward, training, *operands):
    """`forward(*operands)` where no gradient is taken, and `training(*operands)`'s forward and pullback
    where one is (each kernel's own VJP): the delta rule's cells per product (`PORT_CELLS`), which a trace
    cannot tell apart before its gradient is asked."""

    @jax.custom_vjp
    def rule(*operands):
        return forward(*operands)

    def ahead(*operands):
        return jax.vjp(training, *operands)

    def back(pullback, cotangent):
        return pullback(cotangent)

    rule.defvjp(ahead, back)
    return rule(*operands)


def entering_memory(rule, query, key, value, g, beta, starts, *, axis: str, state=None):
    """The memory this shard of a sequence the mesh axis `axis` splits enters its slice with, `[B, Hv, Dk,
    Dv]` in at least fp32, inside a `shard_map` that holds `axis` manual: fla's context-parallel
    summary-then-states (fla/ops/cp/chunk_delta_h.py, fla-core 0.5.2). A delta rule's slice is affine in
    the memory it enters, `S = F S_0 + Z` (each token's transition on the left, `recurrence.update`), so
    each shard's summary is `rule`'s final memory over its slice from `[0 | I]` with values `[v | 0]`, in
    at least fp32 (rounding the identity block's products to the operands' dtype would round the
    transfer): `Z` the write its last document leaves from zero, `F` its transfer, zero where that document
    starts inside the slice. The summaries are each shard's span (`recurrence_algebra.Span`), whose exclusive
    prefix over `axis` (`recurrence_algebra.entering`) from the row's `state` (zero where None) on the first
    shard is the memory each enters, `E_r = F_{r-1} E_{r-1} + Z_{r-1}`; a slice that starts a document (its
    first token's `starts`) enters from zero. `rule(query, key, value, g, beta, state, *, starts)` returns
    `(output, final memory)`."""
    batch, _, heads, columns = value.shape
    width = key.shape[-1]
    wide = at_least_fp32(value.dtype)
    shard = jax.lax.axis_index(axis)
    begins = (jnp.zeros((batch,), bool) if starts is None else jnp.asarray(starts[:, 0], bool)) & (shard > 0)
    zero = rows_like(jnp.zeros((batch, heads, width, columns), wide), value)
    identity = jnp.where(begins[:, None, None, None], 0.0,
                         jnp.broadcast_to(jnp.eye(width, dtype=wide), (batch, heads, width, width)))
    widened = jnp.concatenate([value.astype(wide), jnp.zeros((*value.shape[:-1], width), wide)], axis=-1)
    _, summary = rule(query.astype(wide), key.astype(wide), widened, g, beta.astype(wide),
                      jnp.concatenate([zero, identity], axis=-1), starts=starts)
    span = algebra.Span(summary[..., columns:], summary[..., :columns])
    memory = algebra.entering(span, axis, None if state is None else state.astype(wide))
    return jnp.where(begins[:, None, None, None], 0.0, memory)


def kernel_routed(query, key, value, g, beta, *, floor: float | None = None) -> bool:
    """Whether the delta rule's selector gives a call over these operands a kernel, its forward's
    (`port_kernel`, unreported: the call reports its own): what a layer rounds its operands for, the
    projections' dtype for a kernel, as fla's layer hands them over, the wide one for the portable math."""
    return port_kernel(query, key, value, g, beta, floor=floor, report=False) != 'xla'


def fla_rule(query, key, value, g, beta, state=None, *, starts=None, max_segments: int | None = None,
             scale: float | None = None, wide: bool = False, firsts=None):
    """The port's chunked rule for any call the portable math takes:
    `[B, S, Hk, Dk]` queries and keys, `[B, S, Hv, Dv]` values, the log decay
    `g` `[B, S, Hv]` per head (`fla_gdn.chunk_gated_delta_rule`) or `[B, S,
    Hv, Dk]` per key dimension (Kimi's delta attention, `fla_kda.chunk_kda`),
    `beta` `[B, S, Hv]`, the memory `[B, Hv, Dk, Dv]` or None, and packed
    documents' first tokens `starts` `[B, S]`, at most `max_segments` a row
    (the row length where None), the queries scaled by `scale` in the
    kernels (`Dk^-1/2` where None), the output in the operands' common dtype,
    or fp32 where `wide`, for an epilogue to read it unrounded. Each document
    runs as a sequence of its own (fla's varlen `cu_seqlens`), its chunks
    from its first token and its memory from zero, as it would alone; the
    row's first takes `state`, unless its first token begins a document of
    its own. The kernels' grids are sized by that bound, a chunk more for
    each document and the row's padding; the memories they keep are not
    (`chunked_launch.Sequences`).

    A known defect (ledger row 1328, its measured ratios and reproduction):
    fla's arithmetic, which the port is, holds the recurrence less exactly
    than the portable math, check 1 against the token-by-token reference,
    under saturated write gates (5.8 to 5.9 times the reference's error from
    float64, every family), aligned keys under a decay a key dimension
    (5.4), documents resetting on chunk edges (1.6, every family) and every
    token at the decay's bound (a row 8e-7 from float64 over its size, a
    decay a head and a key dimension). Its error grows with the chunk's
    length, its prefix difference `exp2(gc_s - gc_t)` uncompensated.

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
    documents = None if starts is None else query.shape[1] if max_segments is None else max_segments
    starts = jnp.zeros(query.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    if documents is not None:
        # The kernels start each row from its memory; a first token that begins a document drops it.
        state = jnp.where(starts[:, 0, None, None, None], 0.0, state)
    four, three, memory, decays, resets = _shard_specs(value.shape, key.shape[2], g.ndim)
    if firsts is None:
        return manual_map(functools.partial(_blocked_rule, documents=documents, scale=scale, wide=wide),
                          (four, four, four, decays, three, memory, resets),
                          (four, memory))(query, key, value, g, beta, state, starts)
    return manual_map(functools.partial(_blocked_rule, documents=documents, scale=scale, wide=wide),
                      (four, four, four, decays, three, memory, resets, resets),
                      (four, memory))(query, key, value, g, beta, state, starts, firsts)


def _shard_specs(shape: tuple[int, ...], key_heads: int, decay_ndim: int) -> tuple:
    """A delta rule's per-shard placement (`manual_map`) over values of `shape` reading `key_heads`: the
    `[B, T, H, D]` operands', the write gate's `[B, T, H]`, the memory's `[B, H, Dk, Dv]`, the decay's (a
    head's or a key dimension's), and the documents' starts' `[B, T]`: rows where the rules split them, and
    value heads with the key heads they read where the head split divides the key heads (`grouped_heads`),
    the time axis whole."""
    rows, heads = grouped_heads(shape, key_heads)
    P = jax.sharding.PartitionSpec
    four = P(rows, None, heads, None)
    return four, P(rows, None, heads), P(rows, heads, None, None), P(*four[:decay_ndim]), P(rows, None)


def template_rule(query, key, value, g, beta, state=None, starts=None, firsts=None, **options):
    """The recurrence template's op (`templates.recurrence.template.MOSAIC`, `mosaic_rule`) on each shard
    (`manual_map`), placed as `fla_rule`: its tiles derived from each shard's shapes, its forward and its
    gradient on each device's own rows. Called on the whole batch, every device of a 2xH100 FSDP run ran
    its launches on the global batch, layer 0's solve over 4 rows for its 2, beside 40 GB of gathers a
    microbatch (row 1897). `options` are the op's (`max_segments`, `scale`, `delta`, `wide`, `floor`)."""
    from .templates.recurrence import template

    four, three, memory, decays, resets = _shard_specs(value.shape, key.shape[2], g.ndim)
    optional = [(state, memory), (starts, resets), (firsts, resets)]
    operands = (query, key, value, g, beta, *(x for x, _ in optional if x is not None))
    specs = (four, four, four, decays, three, *(spec for x, spec in optional if x is not None))

    def local(query, key, value, g, beta, *given):
        given = iter(given)
        state, starts, firsts = (None if x is None else next(given) for x, _ in optional)
        return template.MOSAIC(query, key, value, g, beta, state, starts, firsts=firsts, **options)

    return manual_map(local, specs, (four, memory))(*operands)


def mosaic_rule(query, key, value, g, beta, state=None, *, tiles, starts=None,
                max_segments: int | None = None, scale: float | None = None, delta: bool = True,
                wide: bool = False, family=None, floor: float | None = None, firsts=None):
    """The chunked rule, a decay a head (`g` `[B, S, Hv]`) or a key dimension (`[B, S, Hv, Dk]`), with or
    without the `delta` rule, on the recurrence template's Mosaic GPU mainloop
    (`templates.recurrence.mainloop.forward`): `fla_rule`'s operands and results, the operands in their
    common dtype and beta with them, the decay and the memory fp32, on one device, by the bundle of its
    instruction family (`mainloop.family_of`, `bundles.of`), interpreted off a GPU (Mosaic GPU's
    interpreter, `mainloop.INTERPRETED`'s). Its gradient: the gated delta rule's in a 16-bit call on the
    mainloop (`mainloop.gradients`, fla's DAG, fla's rounding) where its launches hold on the device
    (`mainloop.backward_holds`); it takes no other call's, which raises NotImplementedError as the gradient
    is traced: the delta rule's selector gives a training call it refuses another kernel
    (`MOSAIC_SUPPORTS`' 'backward', `Mosaic.trains`), and a call that names the mainloop insists on it
    (ledger 1926: no kernel falls back inside itself). Its states and output launches take `tiles`
    (`mainloop.Tiles`, the template's config, `templates.recurrence.template`); a `family` given is the bundle
    it issues by, interpreted off a GPU of that family (a CPU's). A decay a key dimension takes the layer's
    declared `floor`, which the mainloop's intra launch reads (`mainloop.per_key_refusal`)."""
    from .templates.recurrence import mainloop, template

    dtype = jnp.result_type(query, key, value)
    query, key, value = (x.astype(dtype) for x in (query, key, value))
    # The delta rule's write gate in the operands' dtype, as fla's layer holds it; without the delta rule the
    # call's own, as the portable math takes it (Mamba-2's fp32 `dt`).
    beta = beta.astype(dtype) if delta else beta
    g = g.astype(jnp.float32)
    # Interpreted off a GPU (an assumed one's calls lower for it); a GPU no bundle claims runs no launch.
    device, interpret = mainloop.family_of(dtype), generation.backend() != 'gpu'
    if family is None and device is None and not interpret:
        raise NotImplementedError(f"the recurrence template has no bundle for the "
                                  f"{generation.family(generation.device_generation())!r} family")
    run = {"family": family or device or mainloop.INTERPRETED, "interpret": interpret, "tiles": tiles,
           "floor": floor}
    # The fast backward where its launches hold on the device (a derived launch spills nothing), else none.
    fast = (delta and g.ndim == 3 and dtype != jnp.float32
            and mainloop.backward_holds(query, key, value, g, beta, state, tiles=tiles, family=run["family"],
                                        shared=generation.shared_budget(template.Mosaic.launched()),
                                        starts=starts, max_segments=max_segments))
    run_back = {name: value for name, value in run.items() if name != "floor"}

    # The documents' starts and firsts operands, not closed over (a custom VJP's function closes over no
    # tracer).
    @jax.custom_vjp
    def rule(query, key, value, g, beta, state, starts, firsts):
        # A forward without a gradient at its own tiles, no fast backward narrowing them (`Tiles.forward`).
        alone = {**run, "tiles": tiles.forward or tiles}
        return mainloop.forward(query, key, value, g, beta, state, starts=starts, firsts=firsts,
                                max_segments=max_segments, scale=scale, delta=delta, wide=wide, **alone)

    def forward(*operands):
        if not fast:
            return rule(*operands), (operands, (None, None))
        # The chunks' inverses kept for the fast backward, as fla's keeps its `A`, and the launches' memories,
        # weighted keys and corrected values, as the port keeps its memory, but where the block's remat rung
        # recomputes them (`RECOMPUTED`). Each is named as the port names its own, so a rung that saves names
        # keeps them rather than replaying the forward's launches for them (`mixer_base.DELTA_RESIDUALS`).
        out, final, (inverse, kept) = mainloop.kept_forward(*operands[:6], starts=operands[6],
                                                            firsts=operands[7], max_segments=max_segments,
                                                            scale=scale, delta=delta, wide=wide, **run)
        kept = None if "delta_memory" in RECOMPUTED.get() else checkpoint_name(kept, 'delta_memory')
        return (out, final), (operands, (checkpoint_name(inverse, 'delta_inverse'), kept))

    def backward(saved, grads):
        operands, (inverse, kept) = saved
        out_grad, final_grad = grads
        query, key, value, g, beta, state, starts, firsts = operands
        if not fast:
            form = (f"{dtype.name}, {'the delta rule' if delta else 'no delta rule'}, a decay a "
                    f"{'head' if g.ndim == 3 else 'key dimension'}")
            raise NotImplementedError(
                f"the recurrence template's fast backward takes no gradient of this call ({form}, or its "
                "launches past the device's at these tiles): the delta rule's selector gives a training call "
                "it refuses the port or the portable math (`MOSAIC_SUPPORTS`' 'backward')")
        grads = mainloop.gradients(query, key, value, g, beta, state, out_grad, final_grad, starts=starts,
                                   firsts=firsts, max_segments=max_segments, scale=scale, inverse=inverse,
                                   kept=kept, **run_back)
        return (*grads[:5], None if state is None else grads[5], None, None)

    rule.defvjp(forward, backward)
    return rule(query, key, value, g, beta, state, starts, firsts)


RECOMPUTED: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("recomputed", default=())
"""The residuals a decoder block's remat rung has the layers that compute them recompute, the block
keeping every other (`dew.nn.backbones.decoder_block.RematPolicy.recompute`): a delta rule traced
inside recomputes its memory in its backward where this names 'delta_memory' (`_blocked_rule`),
and keeps it otherwise. A block sets it as it traces its layers (`recomputing`)."""


@contextlib.contextmanager
def recomputing(names: tuple[str, ...]):
    """Trace the layers inside with `names` recomputed by the layers that compute them (`RECOMPUTED`)."""
    token = RECOMPUTED.set(names)
    try:
        yield
    finally:
        RECOMPUTED.reset(token)


def _blocked_rule(query, key, value, g, beta, state, starts, firsts=None, *, documents: int | None,
                  scale: float | None = None, wide: bool = False):
    """The port's rule on one shard, shaped to its kernels' blocks: values
    wider than the port's widest split into more value heads of their key
    head, each key head's value heads as many as it serves (dqkwg's block
    holds them, loaded a head at a time, so no count need be a power of
    two); keys and values padded with zero columns to 64, 128 or 256,
    widths every block of the kernels' configs divides, a decay per key
    dimension with them. None of them changes a real output or gradient:
    every padded product is an exact zero. The outputs are cut back. Where
    `documents` bounds a row's packed documents, each begun at `starts`, each
    is a sequence of its own (fla's varlen `cu_seqlens`). The decay's shape picks
    the port: per head fla's gated delta rule, per key dimension its Kimi
    delta attention. It keeps its memory for the backward
    but where the block's remat rung recomputes it (`RECOMPUTED`)."""
    B, S, _, Dk = key.shape
    Hv, Dv = value.shape[2:]
    scale = Dk ** -0.5 if scale is None else scale
    split = -(-Dv // fla_gdn.WIDEST)
    width = -(-Dv // split)
    if split > 1:
        value = _padded(value, 3, split * width).reshape(B, S, Hv * split, width)
        g, beta = (jnp.repeat(x, split, axis=2) for x in (g, beta))
        state = jnp.moveaxis(_padded(state, 3, split * width).reshape(B, Hv, Dk, split, width), 3, 2)
        state = state.reshape(B, Hv * split, Dk, width)
    blocks = (max(64, 1 << (Dk - 1).bit_length()), max(64, 1 << (value.shape[3] - 1).bit_length()))
    query, key = (_padded(x, 3, blocks[0]) for x in (query, key))
    per_key = g.ndim == 4
    g = _padded(g, 3, blocks[0]) if per_key else g
    value = _padded(value, 3, blocks[1])
    state = _padded(_padded(state, 2, blocks[0]), 3, blocks[1])
    # A row's padding after its documents runs as one more sequence of its own.
    spans = None if documents is None else cu_seqlens(starts, B * (documents + 1), firsts)
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule
    out, final = port(query, key, value, g, beta, state, spans, scale, CHUNK_SIZE,
                      not generation.triton_runs(), "delta_memory" not in RECOMPUTED.get(), wide)
    out = out[..., :width].reshape(B, S, Hv, split * width)[..., :Dv]
    final = final[:, :, :Dk, :width]
    final = jnp.moveaxis(final.reshape(B, Hv, split, Dk, width), 2, 3).reshape(B, Hv, Dk, split * width)
    return out, final[..., :Dv]


def _padded(x, axis: int, size: int):
    """`x` padded with zeros on the end of `axis` to `size`."""
    if x.shape[axis] == size:
        return x
    return jnp.pad(x, [(0, size - x.shape[axis]) if i == axis else (0, 0) for i in range(x.ndim)])


def recurrent_delta_rule(query, key, value, g, beta, state=None, *, starts=None):
    """One token at a time, the decode path, in at least fp32 as a lax.scan over
    the time axis so the state rides the scan the way it rides the decode cache,
    returned as the chunked rule returns them (`chunk_delta_rule`).

    `g` is the log decay per key dimension, `[B, S, H, Dk]`, which is
    `recurrent_kimi_delta_attention` (modeling_glm5_next.py:428-478). The
    gated delta rule decays per head, one decay over every key dimension.
    A token `starts` `[B, S]` marks begins a packed document from a zero memory. Each token is the decode
    step's update (`recurrence.update`), the scan only its orchestration.
    """
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    query, key, value, g, beta = (
        x.astype(work) for x in (query, key, value, g, beta))
    query = query * (key.shape[-1] ** -0.5)

    def one_token(s, step):
        s = jnp.where(step['start'][:, None, None, None], 0.0, s)
        out, s = recurrence.update(s, step['q'], step['k'], step['v'], jnp.exp(step['g']),
                                   step['beta'][..., None], delta=True)
        return s, out
    # The scan stacks along the first axis, so the operands go time-major.
    if state is None:
        state = jnp.zeros((query.shape[0], query.shape[-2], key.shape[-1],
                           value.shape[-1]), work)
    starts = jnp.zeros(query.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    state, out = jax.lax.scan(
        one_token, state.astype(work),
        {name: jnp.moveaxis(x, 1, 0) for name, x in
         (('q', query), ('k', key), ('v', value), ('g', g), ('beta', beta), ('start', starts))})
    return jnp.moveaxis(out, 0, 1).astype(dtype), state


def _rated_scan(hooks: recurrence.Hooks, query, key, value, g, beta, given: dict, state, starts, firsts,
                chunk_size: int, work, *, max_segments: int | None = None):
    """`chunk` without the delta rule over a rated decay (Mamba2's SSD: `x` the values, `B` the keys,
    `C` the queries, `dt` the write gate and `A` the rate) on the kernel the scan's selector gives each
    product (`SCAN`, `scan_kernel`): the recurrence template; the port of mamba_ssm's kernels, which adds
    the skip itself and takes the rated decay's declared domain, a contraction (`recurrence.Hooks`), as its
    kernels clamp each pairwise decay to at most one; a TPU's Pallas scan where its tiling covers the
    geometry; or the portable math, the skip added after. `g` is the log decay the rate and the write gate
    `beta` form (`recurrence.log_gates`), which the port's kernels form themselves. The port and the TPU's
    scan take head and state widths padded with zero columns to powers of two (`_scan_widths`), which change
    no real output or gradient, the outputs cut back: Nemotron-H 4B's 80-wide heads took the portable math,
    6.89 ms a forward against mamba_ssm's 0.89 (SE-ssd-mamba-h100-0428aad7b). Returns the output in `work`,
    unrounded, and the memory `[B, Hv, Dk, Dv]`."""
    rate = given['rate'].astype(work)
    skip = given['skip'].astype(work) if hooks.skip else jnp.zeros(rate.shape, work)

    def product_kernel(product: str) -> str:
        return scan_kernel(query, key, value, g, beta, state, starts, chunk_size=chunk_size,
                           max_segments=max_segments, product=product)

    def run(kernel, query, key, value, g, beta, rate, skip, state, starts, firsts):
        if kernel == 'mosaic_gpu':
            out, final = template_rule(query, key, value, g, beta, state, starts, firsts,
                                       max_segments=max_segments, scale=1.0, delta=False, wide=True)
            return recurrence.skipped(hooks, out, value, {'skip': skip}, work), final
        operand = jnp.result_type(query, key, value)
        x, B, C = (jnp.asarray(t, operand) for t in (value, key, query))
        batch, _, heads, head_dim = x.shape
        state_size = B.shape[-1]
        if kernel in ('pallas', 'tpu'):
            # The SSD's memory is `[P, N]`, values by keys: the template's `[N, P]` transposed.
            carried = (rows_like(jnp.zeros((batch, heads, head_dim, state_size), work), x) if state is None
                       else jnp.swapaxes(state, -1, -2).astype(work))
            columns, rows = _scan_widths(head_dim), _scan_widths(state_size)
            wide_x, wide_B, wide_C = _padded(x, 3, columns), _padded(B, 3, rows), _padded(C, 3, rows)
            wide_state = _padded(_padded(carried, 2, columns), 3, rows)
            if kernel == 'pallas':
                out, final = _ssd_ported(wide_x, beta, rate, wide_B, wide_C, skip, wide_state, chunk_size,
                                         starts)
                out, final = out.astype(work), final.astype(work)
            else:
                out, final = _ssd_tpu(wide_x, beta, rate, wide_B, wide_C, skip, wide_state, chunk_size,
                                      starts, scan_case(x, B, chunk_size, padded=True).tpu, work)
            return out[..., :head_dim], jnp.swapaxes(final[..., :head_dim, :state_size], -1, -2)
        group = heads // key.shape[2]
        if group > 1:
            query, key = (jnp.repeat(t, group, axis=2) for t in (query, key))
        out, final = chunk_delta_rule(query, key, value, g[..., None], beta, state, chunk_size, starts=starts,
                                      delta=False, scale=1.0, dtype=work)
        return recurrence.skipped(hooks, out, value, {'skip': skip}, work), final

    return _per_product(product_kernel, run, query, key, value, g, beta, rate, skip, state, starts, firsts)


def _scan_widths(width: int) -> int:
    """A scan kernel's width for a head or state `width` (`_rated_scan`): the power of two at or past it, at
    least 16, the smallest block Triton's dot takes (`SSD_SUPPORTS`)."""
    return max(16, 1 << (width - 1).bit_length())


RESET_DECAY = -1e4
"""The log decay a document's first token takes in place of `A dt` in the TPU's Pallas scan: the
state entering it is multiplied by `exp(-1e4)`, which is 0 in fp32. A step's own `A dt` is never
positive, so every span containing a reset sums to at most this and decays to exactly 0, and a span
without one never sees it. It stays finite because the kernel forms its segment sums as a matmul
against a 0/1 triangle, where an infinity times 0 is a NaN; and it stays small enough that thousands
of resets in one shard's total decay are nowhere near fp32's range."""


def ssd_operands(x, dt, A, B, C, chunk_size: int, starts=None):
    """A layer's SSD operands as the chunk scan takes them (`dew.nn.kernels.ssd.ssd_chunk_scan`):
    `(x dt, B, C, A dt)`, the groups of `B` and `C` expanded over the heads, the sequence padded to whole
    chunks of `chunk_size`, `x` scaled by its step and the log decay `A dt` a step, `RESET_DECAY` at each
    packed document's first token (`starts`)."""
    heads, length = x.shape[2], x.shape[1]
    # The `H // G` heads of one group read the same B and C, the reference's `repeat_interleave`.
    B, C = repeat_kv_heads(B, heads), repeat_kv_heads(C, heads)
    pad = (chunk_size - length % chunk_size) % chunk_size
    x, B, C = (jnp.pad(t, ((0, 0), (0, pad), (0, 0), (0, 0))) for t in (x, B, C))
    dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
    a = A[None, None, :] * dt
    if starts is not None:
        a = jnp.where(jnp.pad(starts, ((0, 0), (0, pad)))[..., None], RESET_DECAY, a)
    return x * dt[..., None], B, C, a


def _ssd_tpu(x, dt, A, B, C, D, carried, chunk_size: int, starts, platform, work):
    """The rated scan on a TPU's Pallas kernel (`dew.nn.kernels.ssd`), on its operands in `work`
    (`ssd_operands`), per shard of rows and heads (`manual_map`). The skip `D x` is added after. Returns
    the output and the memory `[B, H, P, N]`."""
    length = x.shape[1]
    x, B, C = (jnp.asarray(t, work) for t in (x, B, C))
    skip = D[None, None, :, None] * x
    scaled, B, C, a = ssd_operands(x, dt, A, B, C, chunk_size, starts)

    def kernel(x, B, C, a, carried):
        return ssd_chunk_scan(x, B, C, a, carried, chunk_size, platform)

    # The groups are expanded, so any split of the heads divides them.
    rows, split = grouped_heads(scaled.shape, scaled.shape[2])
    four, held = P(rows, None, split, None), P(rows, split, None, None)
    scanned, final = manual_map(kernel, (four, four, four, P(rows, None, split), held),
                                (four, held))(scaled, B, C, a, carried)
    return scanned[:, :length] + skip, final


@dataclasses.dataclass(frozen=True)
class ScanCase:
    """A rated decay's scan as the scan selector reads it (`SCAN`): whether it is traced for reverse mode
    alone (`reverse`), whether a caller's `shard_map` that checks varying axes holds operands that vary
    (`varying`), the operands' `dtype`, the port's head width, state width and chunk (`widths`, padded,
    `_scan_widths`), the platform the TPU's Pallas scan builds for, None where it does not take the scan
    (`ssd_kernel_platform`), and as the delta rule's case reads them (`PortCase`) the recurrence template's
    implementations bound to the call (`call`), the device's instruction family (`family`) and the product
    (`product`)."""

    reverse: bool
    varying: bool
    dtype: np.dtype
    widths: tuple[int, int, int]
    tpu: str | None
    call: base.Call | None = None
    family: str | None = None
    product: str = 'forward'

    @property
    def cell(self) -> tuple[str | None, str]:
        """The call's cell, its device's instruction family and its product (`SCAN_CELLS`)."""
        return self.family, self.product


SSD_SUPPORTS: tuple[Requirement[ScanCase], ...] = (
    Requirement('forward_mode', lambda case: case.reverse,
                "forward_mode_attention traces for jax.jvp, which the port's custom VJP gives no tangent",
                row=INHERENT),
    Requirement('varying', lambda case: not case.varying, _VARYING, row='ledger 822'),
    Requirement('float64', lambda case: case.dtype != jnp.float64,
                "float64 operands, whose scan is the reference math itself", row=INHERENT),
    Requirement('widths', lambda case: all(width >= 16 and width & (width - 1) == 0 for width in case.widths),
                lambda case: "head width {}, state {} and chunk {}; the port takes powers of two of at least "
                             "16".format(*case.widths), row=INHERENT),
    Requirement('triton', lambda case: triton_runs(),
                "no GPU Dew's Triton kernels run on (sm80 or later): off one the port's kernels interpret, "
                "as the CPU suite runs them where a call names the port", row=INHERENT, kind='measured'),
)
"""What mamba_ssm's kernels (`mamba_ssd.ssd_scan`) need of a scan: reverse mode, which the port's
custom VJP alone gives; no caller's `shard_map` that checks varying axes over operands that vary,
where a Pallas call, which states no varying type for its results, cannot run; no float64 operands,
whose scan is the reference math itself; head widths, state widths and chunks that are powers of
two of at least 16, the smallest block Triton's dot takes (a call's head and state widths padded to them,
`_rated_scan`); and a GPU Dew's Triton kernels run on, any of them."""

TPU_SCAN_SUPPORTS: tuple[Requirement[ScanCase], ...] = (
    Requirement('forward_mode', lambda case: case.reverse,
                "forward_mode_attention traces for jax.jvp, which the TPU scan's custom VJP gives no "
                "tangent: the portable math's scan takes it (row 1869, audit K2)", row=INHERENT),
    Requirement('geometry', lambda case: case.tpu is not None,
                lambda case: "the TPU's Pallas scan takes a float32 scan on a TPU over power-of-two widths "
                             "and a chunk its program holds, not head width {}, state {} and chunk {}"
                             .format(*case.widths), row=INHERENT),
)
"""What the TPU's Pallas scan (`dew.nn.kernels.ssd`) needs: reverse mode alone (its custom VJP gives no
tangent) and its tiling over the geometry (`ssd_kernel_runs`)."""


def scan_case(x, B, chunk_size: int, *, tpu: bool = True, padded: bool = False, call: base.Call | None = None,
              product: str = 'forward') -> ScanCase:
    """The `ScanCase` of a scan over values `x` `[B, S, H, P]` and keys `B` `[B, S, G, N]` in
    `chunk_size` chunks, their widths as the port and the TPU's scan take them where `padded`
    (`_scan_widths`, `_rated_scan`); on a TPU where `tpu` asks, the platform its Pallas scan builds for
    (`ssd_kernel_platform`); the template's `call` and the `product`."""
    widths = (x.shape[-1], B.shape[-1], chunk_size)
    if padded:
        widths = (_scan_widths(widths[0]), _scan_widths(widths[1]), chunk_size)
    family = generation.family(generation.device_generation()) if generation.backend() == 'gpu' else None
    return ScanCase(reverse=not FORWARD_MODE.get(), varying=bool(jax.typeof(x).mat.varying),
                    dtype=jnp.dtype(x.dtype), widths=widths,
                    tpu=(ssd_kernel_platform(chunk_size, widths[0], widths[1], dtype=at_least_fp32(x.dtype))
                         if tpu and generation.backend() == 'tpu' else None),
                    call=call, family=family, product=product)


def scan_kernel(query, key, value, g, beta, state=None, starts=None, *, chunk_size: int,
                max_segments: int | None = None, product: str = 'forward', report: bool = True) -> str:
    """The kernel the scan's selector (`SCAN`) gives a rated scan over these operands (`_rated_scan`'s:
    `value` the SSD's `x`, `key` its `B`, `query` its `C`) for `product` ('forward', 'training'), the
    template's implementations bound to it, on the target (`generation`'s, an assumed one's within
    `assumed`): reported as its route, or not where not `report` (`selected`)."""
    from .templates.recurrence import template

    call = base.call((template.MOSAIC,), query, key, value, g, beta, state, starts,
                     max_segments=max_segments, scale=1.0, delta=False, wide=True)
    case = scan_case(value, key, chunk_size, padded=True, call=call, product=product)
    return select(SCAN, case).kernel if report else selected(SCAN, case).name


SCAN_CELLS: dict[tuple[str | None, str], tuple[str, str]] = {
    ('wgmma', 'forward'): (
        'pallas', "on an H100 PCIe, the port's forward 0.840 ms against the template's 0.954 on "
                  "Nemotron-H 4B's SSD (chunk 256) and 0.329 against 0.377 on Nemotron-H 30B's (chunk 128), "
                  "each the same `chunk(SSD, ...)` call (SE-1964-h100-5e6e356e0)"),
    ('mma.sync', 'forward'): (
        'pallas', "on an A100 SXM4 40GB, the port's forward 1.486 ms against the template's 1.821 on "
                  "Nemotron-H 4B's SSD and 0.529 against 0.769 on 30B's, each the same `chunk(SSD, ...)` "
                  "call (SE-1964-a100-5e6e356e0)"),
}
"""The cells, each an instruction family and a product (`ScanCase.cell`), where a receipt timing the same
product (`chunk(SSD, ...)` with its skip, output width, chunk and geometry, row 1964) shows the port of
mamba_ssm's kernels faster than the recurrence template, each its old route's kernel and that receipt: the
scan's selector runs the port on them where it takes the call (`Op.declared`); the list shrinks as the
template wins its cells. Training is the port's by the template's refusal (it trains no rated scan, row
1926), not a cell."""

SCAN: Op[ScanCase] = Op('ssd', (Kernel('mosaic_gpu', (*SSD_SUPPORTS[:3], *TEMPLATE_SUPPORTS)),
                               Kernel('pallas', SSD_SUPPORTS), Kernel('tpu', TPU_SCAN_SUPPORTS),
                               Kernel('xla', (), generic=True)), declared=SCAN_CELLS)
"""The rated scan's selector: on a declared port cell (`SCAN_CELLS`) the port where it takes the call;
otherwise the first of its kernels that takes the call, by what each takes (no table of GPUs): the
recurrence template at its derived tiles (`TEMPLATE_SUPPORTS`, under the port's trace and placement), the
port of mamba_ssm's kernels on any GPU Dew's Triton kernels run on, the TPU's Pallas scan where its tiling
covers the geometry, else the portable math (a GPU no row named took it, 103 ms a training step of
Nemotron-H 30B's where the port took 2.58, template review 2026-10-26 P1-1)."""


def _ssd_ported(x, dt, A, B, C, D, state, chunk_size: int, starts):
    """The scan on mamba_ssm's kernels, each token's document counted from
    `starts` (mamba_ssm's `seq_idx`). On a mesh it runs per shard
    (`manual_map`): rows where the rules split them, and heads with their
    groups where the head split divides the groups, the time axis whole;
    off a GPU its kernels interpret, as the CPU suite runs them."""
    interpret = not triton_runs()
    seq = None if starts is None else jnp.cumsum(starts, axis=1, dtype=jnp.int32)

    def scan(x, dt, A, B, C, D, state, *seq):
        return mamba_ssd.ssd_scan(x, dt, A, B, C, D, state, *(seq or (None,)), chunk_size, interpret)

    operands = (x, dt, A, B, C, D, state, *([] if seq is None else [seq]))
    rows, heads = grouped_heads(x.shape, B.shape[2])
    four, held = P(rows, None, heads, None), P(rows, heads, None, None)
    specs = (four, P(rows, None, heads), P(heads), four, four, P(heads), held, P(rows, None))
    return manual_map(scan, specs[:len(operands)], (four, held))(*operands)
