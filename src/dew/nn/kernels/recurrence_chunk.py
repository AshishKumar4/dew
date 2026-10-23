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
CPU path. Kernels take the rest: on a GPU whose row names it, the port of fla 0.5.2's kernels
(`fla_rule`, `fla_gdn` and `fla_kda`) takes the delta rule with one decay a head or one per key
dimension; without the delta rule, a rated decay (Mamba2's SSD) runs on the port of mamba_ssm's
kernels (`dew.nn.kernels.mamba_ssd`) on such a GPU, or on a TPU's Pallas scan (`dew.nn.kernels.ssd`).
`chunk` picks one per call and applies the hooks around it.
"""

import contextlib
import contextvars
import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from dew.nn.attention import FORWARD_MODE, repeat_kv_heads
from dew.nn.inputs import cu_seqlens
from dew.nn.precision import at_least_fp32, l2norm
from dew.nn.sharding import grouped_heads, manual_map, rows_like

from . import fla_gdn, fla_kda, generation, mamba_ssd, recurrence, recurrence_algebra as algebra
from .generation import Refusal, first_refusal, measured_kernel, ran_kernel, refused, route, triton_runs
from .ssd import ssd_chunk_scan, ssd_kernel_platform

CHUNK_SIZE = 64
"""Tokens per chunk of the chunked form, the reference's default
(`torch_chunk_gated_delta_rule(..., chunk_size=64)`). The chunked and the
recurrent form agree at any size; the size only trades memory for
parallelism, and 64 is what the reference's kernels assume."""


def held_chunk(chunk_size: int, key_width: int) -> int:
    """The delta rule's chunk at `key_width`: `chunk_size`, at most twice the keys' width. A measured
    sizing policy, not a conditioning bound: over a long sequence (1024 tokens, the decay near one),
    64-token chunks of 16-wide keys put the final memory 1.7 to 1.8 times the token-by-token
    recurrence's error from float64 under every delta family, where chunks of 32 and 16 held it, and so
    did 64-token chunks of 32-, 64- and 128-wide keys (tests/test_recurrence_stability.py, its long
    sequence)."""
    return min(chunk_size, 2 * key_width)


def xla_chunk_states(key, k_cumdecay, out_vals, grown, last_decay, state, kept):
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
    """
    final, (entered, corrected) = jax.lax.scan(
        algebra.carry, state,
        tuple(jnp.moveaxis(x, 2, 0) for x in (key, k_cumdecay, out_vals, grown, last_decay, kept)))
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
    """
    dtype = jnp.result_type(query, key, value) if dtype is None else jnp.dtype(dtype)
    work = at_least_fp32(jnp.result_type(query, key, value))
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    steps = chunked(query, key, value, g, beta, chunk_size, starts=starts, delta=delta, scale=scale)
    T = S + steps.padding

    # A zero state like the values: inside a shard_map that checks varying axes the chunk loop's
    # carry varies as the operands it takes in do, and under an Explicit axis its rows are placed as
    # theirs are.
    state = (rows_like(jnp.zeros_like(value, work, shape=(B, H, Dk, Dv)), value) if state is None
             else state.astype(work))
    # XLA:CPU workaround: under a jitted scan over the layers the chunk
    # loop's zero initial state came back with other values in most
    # processes (GLM-5-Next off by 8.7, 10.1 or NaN, byte-identical modules).
    # The barrier materializes the zero. Remove it when XLA:CPU no longer
    # returns an uninitialized scan carry here.
    state = jax.lax.optimization_barrier(state)
    entered, v_corrected, state = xla_chunk_states(steps.key, steps.w, steps.u, steps.grown, steps.last,
                                                   state, steps.kept)
    # The pairs are zero above the diagonal, so this is the reference's inclusive
    # lower `masked_fill(triu(1), 0)`.
    core = algebra.readout(steps.query, steps.entering, entered, steps.key, steps.pairs, v_corrected)
    # core: [B, H, NC, C, Dv] -> [B, H, NC*C, Dv] -> [B, S, H, Dv]
    core = jnp.moveaxis(core.reshape(B, H, T, Dv), 1, 2)[:, :S]
    return core.astype(dtype), state


def chunk_gated_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE, *,
                           ported: bool | None = None, starts=None, max_segments: int | None = None):
    """`torch_chunk_gated_delta_rule` (modeling_qwen3_next.py:374-453) with one
    log decay per head, `g` `[B, S, H]`, or `chunk_kimi_delta_attention`
    (modeling_glm5_next.py:482-578) with one per key dimension, `[B, S, H,
    Dk]`: the delta rule's row of the chunked op (`chunk`), its queries
    scaled by `Dk^-1/2`. Its queries and keys may sit on fewer heads than the
    values, each serving `Hv // Hk` of them, as fla takes them.

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
    hooks = recurrence.Hooks(decay='vector' if g.ndim == 4 else 'scalar', scale=key.shape[-1] ** -0.5)
    return chunk(hooks, Chunk(query, key, value, log_decay=g, beta=beta, state=state, starts=starts),
                 chunk_size=chunk_size, ported=ported, max_segments=max_segments)


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
    documents' first tokens, each from a zero memory."""
    query: jax.Array
    key: jax.Array
    value: jax.Array
    state: jax.Array | None = None
    starts: jax.Array | None = None


def chunk(hooks: recurrence.Hooks, operands: Chunk, *, chunk_size: int = CHUNK_SIZE,
          ported: bool | None = None, max_segments: int | None = None, dtype=None):
    """`(output [B, S, Hv, Dv], final memory [B, Hv, Dk, Dv])` of a sequence through the recurrence of
    `hooks` (`recurrence.Hooks`, the decode step's form), the output in `dtype` (the queries', keys'
    and values' common dtype where None) and the memory in at least fp32.

    The prologue (the l2 norm, the queries' scale, the gates' activations) and the epilogue (the
    skip, the gated RMS norm) are the decode step's arithmetic (`recurrence.step_reference`) on every
    token, in at least fp32. The recurrence between them runs on the port of fla's kernels
    (`fla_rule`) for the delta rule with a decay where `ported` (the caller's route, `on_the_port`,
    decided here where None); without the delta rule, a rated decay (Mamba2's SSD) runs on the port
    of mamba_ssm's kernels where it takes the call (`ssd_port_refusal`), or on a TPU's Pallas scan
    where its tiling covers the geometry (`ssd_kernel_platform`), unless `ported` is False; else the
    portable math
    (`chunk_delta_rule`, in `chunk_size` chunks, the delta rule's no longer than `held_chunk`, which a
    port's need not match). Packed documents
    (`starts`, at most `max_segments` a row) each compute as they would alone. A conv runs before the
    op: it takes the conv's output. The operands the hooks read are checked as the decode step's
    (`recurrence.read`), and a rated decay is taken on its declared domain, a contraction."""
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
    state, starts = operands.state, operands.starts
    epilogue = hooks.skip or hooks.output is not None
    if hooks.rated and not hooks.delta and not hooks.l2norm and hooks.scale is None:
        # Each of its routes adds the skip, the port in its kernels.
        out, final = _rated_scan(hooks, query, key, value, g, beta, given, state, starts, chunk_size, work,
                                 kernels=ported is not False)
    else:
        if hooks.delta and hooks.decay != 'none' and (on_the_port(query, key, value) if ported is None
                                                      else ported):
            # Unrounded where an epilogue follows, as on the portable math.
            out, final = fla_rule(query, key, value, g, beta, state, starts=starts, max_segments=max_segments,
                                  scale=scale, wide=epilogue)
        else:
            group = value.shape[2] // key.shape[2]
            if group > 1:
                query, key = (jnp.repeat(x, group, axis=2) for x in (query, key))
            size = held_chunk(chunk_size, key.shape[-1]) if hooks.delta else chunk_size
            out, final = chunk_delta_rule(query, key, value, g.reshape(*value.shape[:3], -1), beta, state,
                                          size, starts=starts, delta=hooks.delta, scale=scale,
                                          dtype=work)
        out = recurrence.skipped(hooks, out.astype(work), value, given, work)
    # The epilogue in the working precision, and the output rounded once.
    return recurrence.normed(hooks, out, given, work).astype(dtype), final


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


def fla_rule(query, key, value, g, beta, state=None, *, starts=None, max_segments: int | None = None,
             scale: float | None = None, wide: bool = False):
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
    rows, heads = grouped_heads(value.shape, key.shape[2])
    P = jax.sharding.PartitionSpec
    four, three, memory = P(rows, None, heads, None), P(rows, None, heads), P(rows, heads, None, None)
    return manual_map(functools.partial(_blocked_rule, documents=documents, scale=scale, wide=wide),
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


def _blocked_rule(query, key, value, g, beta, state, starts, *, documents: int | None,
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
    spans = None if documents is None else cu_seqlens(starts, B * (documents + 1))
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


def _rated_scan(hooks: recurrence.Hooks, query, key, value, g, beta, given: dict, state, starts,
                chunk_size: int, work, *, kernels: bool = True):
    """`chunk` without the delta rule over a rated decay (Mamba2's SSD: `x` the values, `B` the keys,
    `C` the queries, `dt` the write gate and `A` the rate): on the port of mamba_ssm's kernels where
    it takes the call, which adds the skip itself and takes the rated decay's declared domain, a
    contraction (`recurrence.Hooks`), as its kernels clamp each pairwise decay to at most one; else
    on a TPU's Pallas scan where its tiling covers the geometry, or the portable math, the skip added
    after; only the portable math without `kernels`. `g` is the log decay the rate and the write gate
    `beta` form (`recurrence.log_gates`), which the kernels form themselves. Returns the output in
    `work`, unrounded, and the memory `[B, Hv, Dk, Dv]`."""
    rate = given['rate'].astype(work)
    skip = given['skip'].astype(work) if hooks.skip else jnp.zeros(rate.shape, work)
    operand = jnp.result_type(query, key, value)
    x, B, C = (jnp.asarray(t, operand) for t in (value, key, query))
    batch, _, heads, head_dim = x.shape
    state_size = B.shape[-1]
    # The SSD's memory is `[P, N]`, values by keys: the template's `[N, P]` transposed.
    carried = (rows_like(jnp.zeros((batch, heads, head_dim, state_size), work), x) if state is None
               else jnp.swapaxes(state, -1, -2).astype(work))
    refusal = ssd_port_refusal(x, B, chunk_size) if kernels else "the caller's route is the portable math"
    if refusal is None:
        out, final = _ssd_ported(x, beta, rate, B, C, skip, carried, chunk_size, starts)
        return out.astype(work), jnp.swapaxes(final.astype(work), -1, -2)
    ran_kernel('ssd', 'xla', refusal)
    platform = ssd_kernel_platform(chunk_size, head_dim, state_size, dtype=work) if kernels else None
    if platform is not None:
        out, final = _ssd_tpu(x, beta, rate, B, C, skip, carried, chunk_size, starts, platform, work)
        return out, jnp.swapaxes(final, -1, -2)
    group = heads // key.shape[2]
    if group > 1:
        query, key = (jnp.repeat(t, group, axis=2) for t in (query, key))
    out, final = chunk_delta_rule(query, key, value, g[..., None], beta, state, chunk_size, starts=starts,
                                  delta=False, scale=1.0, dtype=work)
    return recurrence.skipped(hooks, out, value, given, work), final


RESET_DECAY = -1e4
"""The log decay a document's first token takes in place of `A dt` in the TPU's Pallas scan: the
state entering it is multiplied by `exp(-1e4)`, which is 0 in fp32. A step's own `A dt` is never
positive, so every span containing a reset sums to at most this and decays to exactly 0, and a span
without one never sees it. It stays finite because the kernel forms its segment sums as a matmul
against a 0/1 triangle, where an infinity times 0 is a NaN; and it stays small enough that thousands
of resets in one shard's total decay are nowhere near fp32's range."""


def _ssd_tpu(x, dt, A, B, C, D, carried, chunk_size: int, starts, platform, work):
    """The rated scan on a TPU's Pallas kernel (`dew.nn.kernels.ssd`), on its operands: `x`, `B` and
    `C` in `work`, the groups expanded, the length padded to whole chunks, `x` scaled by `dt`, the
    log decay `A dt` with `RESET_DECAY` at each document's first token; per shard of rows and heads
    (`manual_map`). The skip `D x` is added after. Returns the output and the memory `[B, H, P, N]`."""
    heads, length = x.shape[2], x.shape[1]
    x, B, C = (jnp.asarray(t, work) for t in (x, B, C))
    # The `H // G` heads of one group read the same B and C, the reference's `repeat_interleave`.
    B, C = repeat_kv_heads(B, heads), repeat_kv_heads(C, heads)
    pad = (chunk_size - length % chunk_size) % chunk_size
    x, B, C = (jnp.pad(t, ((0, 0), (0, pad), (0, 0), (0, 0))) for t in (x, B, C))
    dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
    skip = D[None, None, :, None] * x
    a = A[None, None, :] * dt
    if starts is not None:
        a = jnp.where(jnp.pad(starts, ((0, 0), (0, pad)))[..., None], RESET_DECAY, a)

    def kernel(x, B, C, a, carried):
        return ssd_chunk_scan(x, B, C, a, carried, chunk_size, platform)

    # The groups are expanded, so any split of the heads divides them.
    rows, split = grouped_heads(x.shape, heads)
    four, held = P(rows, None, split, None), P(rows, split, None, None)
    scanned, final = manual_map(kernel, (four, four, four, P(rows, None, split), held),
                                (four, held))(x * dt[..., None], B, C, a, carried)
    return (scanned + skip)[:, :length], final


def ssd_port_refusal(x, B, chunk_size: int) -> str | None:
    """Why a chunked scan over these operands runs the XLA scan rather than
    mamba_ssm's kernels (`mamba_ssd.ssd_scan`), or None where the port takes
    it: on every GPU whose target row names it (`measured_kernel`), except
    inside `jax.jvp`'s traces (`FORWARD_MODE`), which the port's custom VJP
    gives no tangent; inside a caller's `shard_map` that checks varying axes
    over operands that vary, where a Pallas call, which states no varying
    type for its results, cannot run; float64 operands, whose scan is the
    reference math itself; and head widths, state widths or chunks that are
    not powers of two of at least 16, the smallest block Triton's dot takes."""
    widths = (x.shape[-1], B.shape[-1], chunk_size)
    return first_refusal(
        (measured_kernel('ssd', 'xla') == 'pallas',
         "the device's target row (generation.TARGETS) does not name the port"),
        (not FORWARD_MODE.get(), "forward_mode_attention traces for jax.jvp, which the port's custom VJP "
                                 "gives no tangent"),
        (not jax.typeof(x).mat.varying,
         "a shard_map that checks varying axes holds the operands, and a Pallas call states no "
         "varying type for its results"),
        (jnp.dtype(x.dtype) != jnp.float64, "float64 operands, whose scan is the reference math itself"),
        (all(width >= 16 and width & (width - 1) == 0 for width in widths),
         f"head width {widths[0]}, state {widths[1]} and chunk {widths[2]}; the port takes powers of "
         "two of at least 16"),
    )


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
