"""The recurrence template's chunked op: a sequence's tokens through the linear recurrence of
`dew.nn.kernels.recurrence` a chunk at a time, for training and prefill (`chunk`).

Per head, over a state `S` `[Dk, Dv]`, a token updates and reads

    S' = (I - delta beta k k^T) Gamma S + iota k v^T,    o = S'^T q (+ D v)

as the decode step does (`recurrence.step`), its form the same `recurrence.Hooks`. Within a chunk of
`C` tokens the dependencies resolve in matrix form: the decay between two positions is one
exponential of their prefixes' difference (`chunk_decay`), the delta rule's corrections one strictly
lower inverse (`strictly_lower_inverse`, fla's WY form), and only the memory crosses chunks
(`xla_chunk_states`). Without the delta rule (`delta=False`, R = 0, Mamba2's SSD and the gated
linear attentions) the inverse is the identity and nothing is corrected.

Two implementations compute it: the portable math in XLA (`chunk_delta_rule`), every hook, Dew's
float32 reference and the CPU path, and on a GPU whose row names it the port of fla 0.5.2's kernels
(`fla_rule`, `fla_gdn` and `fla_kda`), which takes the delta rule with one decay a head or one per
key dimension. `chunk` picks one per call (`on_the_port`) and applies the hooks around it.
"""

import contextlib
import contextvars
import functools

import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

from dew.nn.attention import FORWARD_MODE
from dew.nn.inputs import cu_seqlens
from dew.nn.precision import at_least_fp32, compensated_add, compensated_difference
from dew.nn.sharding import grouped_heads, manual_map

from . import fla_gdn, fla_kda, generation
from .generation import Refusal, measured_kernel, refused, route

CHUNK_SIZE = 64
"""Tokens per chunk of the chunked form, the reference's default
(`torch_chunk_gated_delta_rule(..., chunk_size=64)`). The chunked and the
recurrent form agree at any size; the size only trades memory for
parallelism, and 64 is what the reference's kernels assume."""


def _cumulated(g, documents=None):
    """`g` `[..., C, F]` cumulated over its C positions as a compensated pair `(hi, lo)`
    (`compensated_add`), restarting at each of packed `documents`' first positions."""
    zeros = jnp.zeros_like(g)
    if documents is None:
        return jax.lax.associative_scan(compensated_add, (g, zeros), axis=g.ndim - 2)

    def segmented(left, right):
        (left_start, *left_sum), (right_start, *right_sum) = left, right
        total = compensated_add(left_sum, right_sum)
        restarted = (jnp.where(right_start, kept, summed)
                     for kept, summed in zip(right_sum, total, strict=True))
        return left_start | right_start, *restarted

    changed = jnp.concatenate([jnp.ones_like(documents[..., :1, :], bool),
                               documents[..., 1:, :] != documents[..., :-1, :]], axis=-2)
    _, hi, lo = jax.lax.associative_scan(segmented, (jnp.broadcast_to(changed, g.shape), g, zeros),
                                         axis=g.ndim - 2)
    return hi, lo


@jax.custom_jvp
def chunk_decay(g, documents=None):
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

    The gradient is the pairs' own (`_chunk_decay_tangent`): differentiating
    the difference of two prefixes sums a pair's cotangent into its two ends
    with opposite signs, and the decay gradient, a row sum less a column sum
    of the chunk's pairs, cancelled to fp32 noise wherever the prefixes ran
    long.

    With packed `documents` `[..., C, 1]` (each position's document within
    its chunk), the sum restarts at each document's first position and pairs
    of positions in different documents weigh zero, masked before exp: a
    document's decays are its own tokens' alone, the ones it computes run by
    itself.
    """
    hi, lo = _cumulated(g, documents)
    chunk_size = hi.shape[-2]
    inclusive = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_))[..., None]
    if documents is not None:
        inclusive = inclusive & (documents[..., :, None, :] == documents[..., None, :, :])
    diff = compensated_difference((hi[..., :, None, :], lo[..., :, None, :]),
                                  (hi[..., None, :, :], lo[..., None, :, :]))
    # Masked before exp, as the references do. The unused positive
    # differences can overflow, and an outer where alone leaves 0 * inf in
    # the decay gradient.
    diff = jnp.where(inclusive, diff, 0.0)
    return hi, jnp.where(inclusive, jnp.exp(diff), 0.0)


def _ranged(decay, dg):
    """The decay's tangent: each pair's decay times its range's tangent, `dg` summed over `(t, s]`,
    pair by pair; a pair across documents weighs zero."""
    width = dg.shape[-2]
    strict = jnp.tril(jnp.ones((width, width), bool), -1)[..., None]
    return decay * jnp.cumsum(jnp.where(strict, dg[..., :, None, :], 0.0), axis=-3)


def _ranged_transpose(decay, cotangent):
    """`_ranged`'s transpose: position i's gradient is the weighted cotangent of every pair whose
    range holds it, `sum_{s >= i} sum_{t < i} cotangent[s, t] decay[s, t]`, two cumulative sums
    of the chunk's pairs, no pair entering any sum it does not span."""
    width = decay.shape[-3]
    strict = jnp.tril(jnp.ones((width, width), bool), -1)[..., None]
    weighted = jnp.cumsum(jnp.where(strict, cotangent * decay, 0.0), axis=-2)
    before = jnp.concatenate([jnp.zeros_like(weighted[..., :, :1, :]), weighted[..., :, :-1, :]], axis=-2)
    after = jnp.flip(jnp.cumsum(jnp.flip(before, axis=-3), axis=-3), axis=-3)
    return jnp.moveaxis(jnp.diagonal(after, axis1=-3, axis2=-2), -1, -2)


@chunk_decay.defjvp
def _chunk_decay_tangent(primals, tangents):
    """The pairs' own tangent, in forward and reverse mode alike: the decay's is each pair's range
    (`_ranged`), and its transpose sums only the pairs spanning a position (`_ranged_transpose`).
    On Qwen3.5-tiny's packed decision rows (tests/test_decision_packing.py, joint) the prefix
    difference's gradient put the gated delta net's A_log 3.58 and dt_bias 4.49 times as far from
    float64 as the rows run alone, against 2 allowed; with this and the memory write's decay row
    (`xla_chunk_states`) every backbone leaf sat at 0.77-0.93, on six CPUs of two models. It costs
    the chunked rule's forward and backward 4% with a decay per head and 26% with one per key
    dimension on CPU (2 x 4096 x 16 heads of 128: 1.508 against 1.564 s; 2 x 1024 x 8 heads,
    1.936 against 2.440 s)."""
    g, documents = primals
    dg, _ = tangents
    gc, decay = chunk_decay(g, documents)
    return (gc, decay), (_cumulated(dg, documents)[0], linear_call(_ranged, _ranged_transpose, decay, dg))


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


def xla_chunk_states(key, k_cumdecay, out_vals, gc, last_decay, state, kept=None):
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

    The write weighs each token by `last_decay` `[B, H, NC, C, D]`, the last
    row of the chunk's pairwise decay (`chunk_decay`'s `exp(gc[-1] - gc)`):
    its exponent compensated as every pair's is, where the prefix's rounded
    half alone put a long chunk's write off by its whole sum's rounding, and
    zero for every token of a packed chunk's earlier documents, which the
    write does not keep. With packed documents, `kept` `[B, 1, NC, 1, 1]`
    marks the chunks no document starts in, the only ones the memory they
    entered passes through.
    """
    def one_chunk(s, step):
        k_i, kd_i, v_i, gc_i, last_i, kept_i = step
        v_corrected = v_i - kd_i @ s
        return (s * jnp.swapaxes(jnp.exp(gc_i[..., -1:, :]), -1, -2) * kept_i
                + jnp.swapaxes(k_i * last_i, -1, -2) @ v_corrected, (s, v_corrected))
    kept = jnp.ones((1, 1, gc.shape[2], 1, 1), gc.dtype) if kept is None else kept.astype(gc.dtype)
    final, (entered, corrected) = jax.lax.scan(
        one_chunk, state,
        tuple(jnp.moveaxis(x, 2, 0) for x in (key, k_cumdecay, out_vals, gc, last_decay, kept)))
    return jnp.moveaxis(entered, 0, 2), jnp.moveaxis(corrected, 0, 2), final


def _paired(left, right, decay):
    """`sum_d left[s, d] right[t, d] decay[s, t, d]` within each chunk,
    `[..., C, C]`: one decay channel scales the product after it, one per key
    dimension weighs each term of it."""
    if decay.shape[-1] == 1:
        return (left @ jnp.swapaxes(right, -1, -2)) * decay[..., 0]
    return jnp.einsum('...sd,...td,...std->...st', left, right, decay)


def chunk_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE, *, starts=None):
    """The chunked delta rule over [B, S, H, D] operands with the log decay
    `g` `[B, S, H, D]`, D one for a decay per head (the gated delta rule) or
    Dk for one per key dimension (Kimi Delta Attention); returns
    `(output [B, S, H, Dv], final_state [B, H, Dk, Dv])`, the output in the
    queries', keys' and values' common dtype and the memory in at least fp32,
    as the port returns them (`fla_rule`), computed in at least fp32. The
    references' sequential correction is the inverse `strictly_lower_inverse`
    builds from its diagonal blocks.

    Only the memory crosses chunks, so only its recurrence runs chunk after
    chunk (`xla_chunk_states`). Each chunk's output, the references'
    `attn_inter + attn @ v_new`, reads the state that chunk entered and its
    corrected values, so it is computed for every chunk at once after the
    recurrence. The two decays differ only in how a pair of positions is
    weighed (`_paired`). It is the portable math: a GPU runs the gated rule on
    fla's kernels (`chunk_gated_delta_rule`), and this is their test oracle.

    `starts` `[B, S]` marks packed documents' first tokens, each from a zero
    memory: within a chunk each document's decays are its own (`chunk_decay`),
    a token reads the memory the chunk entered only in the document that
    memory belongs to, and the chunk's write keeps only its last document's
    tokens (the decay's last row is zero for the others), the memory it
    entered passing through only where no document starts in it. A document
    computes as it would alone.
    """
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
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
    document = first = kept = None
    if starts is not None:
        # Each token's document within its row, per chunk, [B, 1, NC, C, 1], and the document the
        # memory each chunk enters belongs to: its first token's, unless that token starts one.
        begins = jnp.pad(jnp.asarray(starts, jnp.int32), ((0, 0), (0, pad)))
        document = jnp.cumsum(begins, axis=1).reshape(B, 1, T // chunk_size, chunk_size, 1)
        entered_document = document[..., :1, :] - begins.reshape(document.shape)[..., :1, :]
        first = document == entered_document
        kept = document[..., -1:, :] == entered_document
    # The log decay cumulated within each chunk, the reference's
    # `g = g.cumsum(dim=-1)` (modeling_qwen3_next.py:417), and the decay
    # `exp(gc[s] - gc[t])` between positions s >= t, zero above the diagonal:
    # [B, H, NC, C, D], [B, H, NC, C, C, D].
    gc, decay = chunk_decay(chunks(g), document)
    # The mask is strictly lower (tril, -1): the reference's masked_fill
    # zeroes the diagonal too.
    strict = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_), -1)
    attn = jnp.where(strict, -_paired(kb_c, k_c, decay), 0.0)
    # The reference's `attn + I` operator after its row correction loop.
    inv = strictly_lower_inverse(attn)
    out_vals = inv @ vb_c  # the reference's `value = attn @ v_beta`
    entering = jnp.exp(gc) if first is None else jnp.exp(gc) * first
    k_cumdecay = inv @ (kb_c * entering)

    # A zero state like the values: inside a shard_map that checks varying axes the chunk loop's
    # carry varies as the operands it takes in do.
    state = jnp.zeros_like(value, work, shape=(B, H, Dk, Dv)) if state is None else state.astype(work)
    # XLA:CPU workaround: under a jitted scan over the layers the chunk
    # loop's zero initial state came back with other values in most
    # processes (GLM-5-Next off by 8.7, 10.1 or NaN, byte-identical modules).
    # The barrier materializes the zero. Remove it when XLA:CPU no longer
    # returns an uninitialized scan carry here.
    state = jax.lax.optimization_barrier(state)
    entered, v_corrected, state = xla_chunk_states(k_c, k_cumdecay, out_vals, gc, decay[..., -1, :, :], state,
                                                   kept)
    # decay is zero above the diagonal, so this is the reference's inclusive
    # lower `masked_fill(triu(1), 0)`.
    core = (q_c * entering) @ entered + _paired(q_c, k_c, decay) @ v_corrected
    # core: [B, H, NC, C, Dv] -> [B, H, NC*C, Dv] -> [B, S, H, Dv]
    core = jnp.moveaxis(core.reshape(B, H, T, Dv), 1, 2)[:, :S]
    return core.astype(dtype), state


def chunk_gated_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE, *,
                           ported: bool | None = None, starts=None, max_segments: int | None = None):
    """`torch_chunk_gated_delta_rule` (modeling_qwen3_next.py:374-453) with one
    log decay per head, `g` `[B, S, H]`, or `chunk_kimi_delta_attention`
    (modeling_glm5_next.py:482-578) with one per key dimension, `[B, S, H,
    Dk]`. Its queries and keys may sit on fewer heads than the values, each
    serving `Hv // Hk` of them, as fla takes them.

    On the port it is fla 0.5.2's chunked rule on Dew's port of its kernels
    (`fla_rule`), on fla's numbers; elsewhere the portable math
    (`chunk_delta_rule`), in `chunk_size` chunks, which the port's 64 need
    not match: the rule's value does not depend on them. Both take the
    operands as they come. `ported` is the caller's route (`on_the_port`),
    which a layer decides once, before it rounds the operands for that route
    (the projections' dtype for the port, as fla's layer hands them over, the
    wide one for the math); None decides it here, where the port takes these
    operands.

    `starts` `[B, S]` marks packed documents' first tokens, each starting
    from a zero memory (`state` continues a row's first document), at most
    `max_segments` a row (`Coordinates.max_segments`): the port runs each
    document as a sequence of its own (`fla_rule`), its kernels' grids sized
    by that bound, and the portable math resets its recurrence at each start,
    which needs none.
    """
    if value.shape[2] % key.shape[2]:
        raise ValueError(f"{value.shape[2]} value heads over {key.shape[2]} key heads: each key head serves "
                         "a whole number of value heads")
    if ported is None:
        ported = on_the_port(query, key, value)
    if ported:
        return fla_rule(query, key, value, g, beta, state, starts=starts, max_segments=max_segments)
    group = value.shape[2] // key.shape[2]
    if group > 1:
        query, key = (jnp.repeat(x, group, axis=2) for x in (query, key))
    return chunk_delta_rule(query, key, value, g.reshape(*value.shape[:3], -1), beta, state, chunk_size,
                            starts=starts)


def port_refusal(query, key, value) -> Refusal | None:
    """Why a rule over these operands runs the portable math rather than the
    port (`fla_rule`) on a target whose row names the port, or None where the
    port takes it: every shape, dtype, chunk size and mesh, except
    inside `jax.jvp`'s traces (`FORWARD_MODE`), which the port's custom VJP
    gives no tangent; inside a caller's `shard_map` that checks varying axes
    over operands that vary, where a Pallas call, which states no varying
    type for its results, cannot run; float64 operands, whose rule is the
    reference math itself; and keys wider than the port takes in their
    dtype. Packed documents take the port at any bound: it stores the
    memories by where they pass, as many as the tokens' chunks, a row's and
    one, whatever the bound (`chunked_launch.Sequences`)."""
    widest = fla_gdn.WIDEST_KEYS if jnp.result_type(query, key, value).itemsize == 2 else fla_gdn.WIDEST
    return refused(
        (not FORWARD_MODE.get(), 'forward_mode',
         "forward_mode_attention traces for jax.jvp, which the port's custom VJP gives no tangent"),
        (not jax.typeof(value).mat.varying, 'varying',
         "a shard_map that checks varying axes holds the operands, and a Pallas call states no "
         "varying type for its results"),
        (all(jnp.dtype(x.dtype) != jnp.float64 for x in (query, key, value)), 'float64',
         "float64 operands, whose rule is the reference math itself"),
        (key.shape[-1] <= widest, 'key_width',
         f"keys {key.shape[-1]} wide; the port takes at most {widest} in "
         f"{jnp.result_type(query, key, value).name}"),
    )


def on_the_port(query, key, value) -> bool:
    """Whether the rule runs on the port: where the target's row names it (`measured_kernel`, the table
    qualification reads) and the call has no refusal (`port_refusal`), reported as the target's route
    (`generation.route`), which logs a call the row's port turns down the first time."""
    preferred = measured_kernel('gated_delta_rule', 'xla')
    refusal = port_refusal(query, key, value) if preferred == 'fla' else None
    kernel = 'fla' if preferred == 'fla' and refusal is None else 'xla'
    return route('gated_delta_rule', kernel=kernel, refusal=refusal).kernel == 'fla'


def fla_rule(query, key, value, g, beta, state=None, *, starts=None, max_segments: int | None = None):
    """The port's chunked rule for any call the portable math takes:
    `[B, S, Hk, Dk]` queries and keys, `[B, S, Hv, Dv]` values, the log decay
    `g` `[B, S, Hv]` per head (`fla_gdn.chunk_gated_delta_rule`) or `[B, S,
    Hv, Dk]` per key dimension (Kimi's delta attention, `fla_kda.chunk_kda`),
    `beta` `[B, S, Hv]`, the memory `[B, Hv, Dk, Dv]` or None, and packed
    documents' first tokens `starts` `[B, S]`, at most `max_segments` a row
    (the row length where None). Each document runs as a sequence of its own
    (fla's varlen `cu_seqlens`), its chunks from its first token and its
    memory from zero, as it would alone; the row's first takes `state`, unless
    its first token begins a document of its own. The kernels' grids are
    sized by that bound, a chunk more for each document and the row's
    padding; the memories they keep are not (`chunked_launch.Sequences`).

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
    rows, heads = grouped_heads(value.shape, key.shape[2])
    P = jax.sharding.PartitionSpec
    four, three, memory = P(rows, None, heads, None), P(rows, None, heads), P(rows, heads, None, None)
    return manual_map(functools.partial(_blocked_rule, documents=documents),
                      (four, four, four, P(*four[:g.ndim]), three, memory, P(rows, None)),
                      (four, memory))(query, key, value, g, beta, state, starts)


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


def _blocked_rule(query, key, value, g, beta, state, starts, *, documents: int | None):
    """The port's rule on one shard, shaped to its kernels' blocks: values
    wider than the port's widest split into more value heads of their key
    head; value heads padded to a power of two per key head, which dqkwg's
    block holds, with heads that neither write nor read (zero value, beta
    and decay); keys and values padded with zero columns to 64, 128 or 256,
    widths every block of the kernels' configs divides, a decay per key
    dimension with them. None of them changes a real output or gradient:
    every padded product is an exact zero. The outputs are cut back. Where
    `documents` bounds a row's packed documents, each begun at `starts`, each
    is a sequence of its own (fla's varlen `cu_seqlens`). The decay's shape picks
    the port: per head fla's gated delta rule, per key dimension its Kimi
    delta attention. It keeps its memory for the backward
    but where the block's remat rung recomputes it (`RECOMPUTED`)."""
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
    per_key = g.ndim == 4
    g = _padded(g, 3, blocks[0]) if per_key else g
    value = _padded(value, 3, blocks[1])
    state = _padded(_padded(state, 2, blocks[0]), 3, blocks[1])
    # A row's padding after its documents runs as one more sequence of its own.
    spans = None if documents is None else cu_seqlens(starts, B * (documents + 1))
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule
    out, final = port(query, key, value, g, beta, state, spans, scale, CHUNK_SIZE,
                      not generation.triton_runs(), "delta_memory" not in RECOMPUTED.get())
    out = out[..., :width].reshape(B, S, Hk, group, width)[:, :, :, :served]
    out = out.reshape(B, S, Hv, split * width)[..., :Dv]
    final = final[:, :, :Dk, :width].reshape(B, Hk, group, Dk, width)[:, :, :served]
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
    A token `starts` `[B, S]` marks begins a packed document from a zero memory.
    """
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    query, key, value, g, beta = (
        x.astype(work) for x in (query, key, value, g, beta))
    query = query * (key.shape[-1] ** -0.5)

    def one_token(s, step):
        q_t, k_t, v_t, g_t, beta_t = (step[name] for name in ('q', 'k', 'v', 'g', 'beta'))
        s = jnp.where(step['start'][:, None, None, None], 0.0, s)
        s = s * jnp.exp(g_t)[..., :, None]                 # rows decay per key dimension
        kv_mem = jnp.sum(s * k_t[..., :, None], axis=-2)   # [B, H, Dv]
        delta = (v_t - kv_mem) * beta_t[..., None]        # [B, H, Dv]
        s = s + k_t[..., :, None] * delta[..., None, :]    # [B, H, Dk, Dv]
        return s, jnp.sum(s * q_t[..., :, None], axis=-2)  # [B, H, Dv]
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
