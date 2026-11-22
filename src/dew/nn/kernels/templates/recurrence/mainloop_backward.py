"""The recurrence mainloop's training pass on its machinery (`mainloop`): the fast backward's launches
(`gradients`, fla's DAG of the delta rule and, without it, mamba_ssm's rounding of Mamba2's SSD) and the
forms they take (`fast_backward_form`). The plan of a call's launches, forward and backward, is the
template's (`template.planned`, `template.backward_holds`)."""

import dataclasses
import functools
from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, recurrence_algebra as algebra, tables
from dew.nn.kernels.chunked_launch import zero_padded
from dew.nn.kernels.templates import mosaic, product, resources, swizzling
from dew.nn.kernels.templates.operands import Issuer, Parts
from dew.nn.kernels.templates.recurrence import key_backward
from dew.nn.kernels.templates.recurrence.chunk_forward import ForkedTable, prepared
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Operand,
    Tiles,
    Walked,
    addressed,
    block_stored,
    chunk_liveness,
    column_tile,
    diagonal,
    f32,
    final_memory_landing,
    fitting_block,
    held_tile,
    launch_kernel,
    lowered_once,
    masks,
    memory_landing,
    prefix_reader,
    product_fragments,
    product_results,
    role_tiles,
    sequence_live,
    sequence_walk,
    shared_allocation,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
    summed_issuer,
    summing_staging,
    table_held,
    vector_blocks,
    vector_views,
    walk_stores,
    whole_blocks,
)
from dew.nn.kernels.templates.recurrence.table import Table, returned_by, table_landing, table_returned
from dew.nn.kernels.templates.resources import largest_live_set
from dew.nn.precision import compensated_sum


def fast_backward_form(delta: bool, rounded: bool, dtype, *, per_key: bool = False,
                       wide: bool = False) -> bool:
    """Whether the fast backward takes a call's form: the delta rule, fla's DAG at fla's rounding, a decay a
    head's or a key dimension's (fla_kda's, `key_backward`); or a bf16 call without it and a decay a head or
    none, the same DAG without the delta rule's: `rounded` (the SSD's under the run's fast forms, its route's)
    at mamba_ssm's rounding (`prepared`), held to recorded mamba_ssm's bf16 case, else its intermediates in
    fp32's three parts as its forward holds them (lightning attention's, a write gate's, the SSD's under the
    accurate forms), held to the portable math; either in fp32 under the run's fast forms (`rounded`), its
    operands in two bf16 parts (`mainloop.FAST_PARTS`). An fp16 call without the delta rule, which no
    recording holds, a decay a key dimension without it, an fp32 call's three parts (the accurate forms, or a
    form without the delta rule on its route, `recurrence_chunk.chunk_rounding`), and a bf16 call's held in
    three parts whose output is fp32 (`wide`: an epilogue reads it, Mamba2's gated norm), its gradient fp32
    where the launches read it in bf16 (rounded so, Mamba2's step's gradient under its norm was 2.2e-3 from
    float64 against the portable math's 3.1e-7), take none. A decay a key dimension's form takes each
    sub-chunk's own pairs each their own decay (`algebra.key_diagonal_pairs_back`)."""
    dtype = jnp.dtype(dtype)
    if per_key and dtype == jnp.float32 and not rounded:
        return False
    if delta:
        return True
    if per_key or dtype == jnp.float16:
        return False
    return rounded or dtype == jnp.float32 or not wide


class Kept(NamedTuple):
    """Each launch of the fast backward's outputs as it stores them (`kept_backward`), nothing computed again
    for them: what check 2's witness holds each launch to (tests/recurrence_witness.py). Without the delta
    rule the system's, values' and keys' launches take none (their fields None)."""

    run: object
    """The prepared call (`chunk_forward.prepared`): its laid operands, decay halves, inverse, corrected
    values and stored memories, which the launches read."""
    local: jax.Array
    """`_local_back`'s corrected values' gradient from each chunk's own output."""
    dh: jax.Array
    """`_reverse`'s gradient of the memory each chunk leaves."""
    dv: jax.Array
    """`_reverse`'s corrected values' gradient."""
    dh0: jax.Array
    """`_reverse`'s gradient of the initial memory."""
    ds: jax.Array
    """The scores' gradient in `_score_grads`' form: `_scores_back`'s, a key head's group summed after it
    (`_grouped_scores`)."""
    decay_pairs: jax.Array
    """`_scores_back`'s decay pair terms."""
    dq: jax.Array
    dk: jax.Array
    dw: jax.Array | None
    after: jax.Array
    leaving: jax.Array
    memory: jax.Array
    """`_query_key`'s queries', keys' and weights' gradients and its decay terms."""
    d_l: jax.Array | None
    """`_system_back`'s system's gradient."""
    values: jax.Array | None
    values_beta: jax.Array | None
    """`_values_back`'s values' gradient and beta's through `u`."""
    keys: jax.Array | None
    keys_beta: jax.Array | None
    keys_after: jax.Array | None
    keys_decay: jax.Array | None
    """`_keys_back`'s keys' gradient and its beta, range and pair terms, a key block's each, unsummed."""


def gradients(q, k, v, g, beta, h0, do, dfinal, **options):
    """The fast backward's gradients (`kept_backward`'s first)."""
    return kept_backward(q, k, v, g, beta, h0, do, dfinal, **options)[0]


def kept_backward(q, k, v, g, beta, h0, do, dfinal, *, tiles: Tiles, starts=None, firsts=None,
                  max_segments: int | None = None, scale: float | None = None,
                  family: mosaic.Family | None = None, interpret: bool = False,
                  inverse=None, kept=None, delta: bool = True,
                  rounded: bool = False, apart: int = 0, values_gated: bool = False,
                  unit_keys: bool = False, ranged: bool = False,
                  skip=None,
                  forked: ForkedTable | None = None) -> tuple[tuple, "Kept | key_backward.KeyKept"]:
    """The fast backward of the gated delta rule, a decay a head (`forward`'s, `g` `[B, T, HV]`), in a 16-bit
    call, or of a call without it whose form it takes (`fast_backward_form`):
    `(dq, dk, dv, dg, dbeta, dh0, dskip, dentries)` for the output's and final memory's gradients `do`,
    `dfinal`, fla's
    DAG (recomputing the forward's values, as fla does) through the algebra's backward tile functions:
    `_local_back`, `_reverse`, `_scores_back`, `_query_key`, `_system_back`, `_values_back`, `_keys_back`, the
    decay's gradient summed token by token (`algebra.summed_pairs`, `algebra.summed_ranges`) and the memory's
    decay's term added, a key head's value heads summed; the forward's `inverse` and `kept` values where given
    (`kept_forward`'s) rather than computed again, its states launch at the forward's `tiles` where they are
    not. Its rounding is fla's (mamba_ssm's for a `rounded` call without the delta rule; another holds its
    intermediates in fp32's three parts, as its forward holds them). Without the delta rule there is no
    system: the corrected values are the values gated, `u = v beta`, whose gradient `du` (the reverse walk's)
    gives `dv = du beta` and `dbeta = sum(v du)`, and the memory walks back with no `w^T dv`; a `skip` `[HV]`
    (Mamba2's `D`,
    `chunk_forward.forward`'s) adds `D do` to `dv`, its own gradient `sum(v do)` (`dskip`, None without a
    skip). Returns them with each launch's outputs as stored (`Kept`; a decay a key dimension's,
    `key_backward.KeyKept`), the forward's `kept_forward` beside it. A `forked` call's
    (`chunk_forward.ForkedTable`) `dh0` is its given memories' and `dentries` its entry write's
    (`_tabled_gradients`, `_entered_gradient`), None for another."""
    dtype = jnp.result_type(q, k, v)
    if not fast_backward_form(delta, rounded, dtype, per_key=g.ndim == 4, wide=jnp.dtype(do.dtype) != dtype):
        raise NotImplementedError("the fast backward takes the delta rule (under a decay a key dimension "
                                  "not in fp32 three-part form), or a bf16 or fp32 call without it and a "
                                  "decay a head (a bf16 call's output gradient bf16 where not rounded)")
    run = prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=delta,
                   tiles=tiles, family=family, interpret=interpret, inverse=inverse, kept=kept,
                   firsts=firsts, rounded=rounded, apart=apart, values_gated=values_gated,
                   unit_keys=unit_keys, ranged=ranged, forked=forked,
                   skipped=skip is not None)
    call = run.call
    rows, heads, width, columns = call.rows, call.heads, call.width, call.value_width
    laid_do = call.given.split(run.laid(zero_padded(do.astype(run.dtype), 3, columns)).reshape(rows, -1))
    dht = zero_padded(zero_padded(dfinal.astype(jnp.float32), 2, width), 3, columns).reshape(-1, columns)
    if call.per_key:
        (dq, dk, dv, dg, dbeta, dh0, taken), key_kept = key_backward.gradients(run, laid_do, dht, tiles)
        dg = run.unlaid(dg.reshape(rows, heads, width))[..., :run.key_width]
        grads = _unlaid(run, (q, k, v, g, beta), dq, dk, dv, dg, run.unlaid(dbeta), dh0, starts)
        return (*_tabled_gradients(run, grads, taken), None, _entered_gradient(run, taken)), key_kept
    local = _local_back(run.q, run.k, laid_do, run.hi, run.lo, run.taken, call, scale=run.scale)
    # Without the delta rule the walk gates the values where their columns are the call's own. Padded, it
    # read a padded copy of them (0.061 ms of Nemotron-H 4B's SSD training on an H100) and its gate took
    # 0.115, where XLA's reduction over the walk's fp32 gradient fused the padding in 0.119; unpadded, the
    # walk's gate took 0.038 where XLA's took 0.137 (30B; window-h100-c70aaa8-b8109ee).
    gates = not delta and run.value_width == columns
    dh, dv2, dh0, *gated = _reverse(run.q, run.k, run.w, laid_do, local, run.hi, run.lo, dht, run.sequences,
                                    run.v if gates else None, run.beta if gates else None, call, tiles,
                                    scale=run.scale, skip=skip if gates else None, table=run.table)
    taken = gated.pop() if run.table is not None else None
    ds, decay_pairs = _scores_back(run.q, run.k, run.corrected, laid_do, run.hi, run.lo, run.taken, run.beta,
                                   call, scale=run.scale)
    ds = _grouped_scores(ds, call)
    dq, dk, dw, after, leaving, memory = _query_key(run.q, run.k, run.corrected, laid_do, dv2, run.h, dh, ds,
                                                    run.hi, run.lo, run.taken, run.beta, call,
                                                    scale=run.scale)
    kept_ = Kept(run, local, dh, dv2, dh0, ds, decay_pairs, dq, dk, dw, after, leaving, memory,
                 *[None] * 7)
    chunks = call.chunks
    dskip = None
    if delta:
        d_l = _system_back(run.a, run.k, run.v, dw, dv2, run.hi, run.lo, run.beta, run.taken, call)
        dv, dbeta_u = _values_back(run.a, run.v, dv2, run.beta, run.taken, call)
        dk_w, dbeta_k, after_w, decay_w = _keys_back(run.a, run.k, dw, d_l, run.hi, run.lo, run.beta,
                                                     run.taken, call)
        kept_ = kept_._replace(d_l=d_l, values=dv, values_beta=dbeta_u, keys=dk_w, keys_beta=dbeta_k,
                               keys_after=after_w, keys_decay=decay_w)
        # The keys launch's terms a key block's, summed (each linear in them).
        dbeta_k, after_w, decay_w = (x.sum(0) for x in (dbeta_k, after_w, decay_w))
        dbeta = dbeta_u + dbeta_k
        after_w = after_w.reshape(heads, chunks, call.chunk)
    else:
        dv, dbeta, dskip = _gate_back(run, call, dv2, gated, laid_do, skip)
        dk_w, after_w, decay_w = None, 0.0, 0.0
    # Each token's decay's gradient from the launches' terms (`algebra.decay_gradient`, linear in them): the
    # query/key launch's ranges, `after` from each token to the chunk's end and `leaving` from its start to
    # before it, the keys launch's beside them, and the memory's decay across each chunk.
    memory = memory.reshape(-1, heads, chunks, call.block).sum((0, 3))
    ending = (run.hi + run.lo).reshape(heads, chunks, call.chunk)[..., -1]
    after, leaving = (x.sum(0).reshape(heads, chunks, call.chunk) for x in (after, leaving))
    dg = algebra.decay_gradient((decay_pairs + decay_w).reshape(heads, chunks, call.chunk), after + after_w,
                                leaving, memory, ending)
    dg, dbeta = dg.reshape(heads, rows).T, dbeta.T if delta else dbeta

    def headed(x, size):
        """`[R, HV * size]` laid back as `[B, T, HV, size]`."""
        return run.unlaid(x.reshape(rows, heads, size))

    def keyed(x):
        """A value head's gradient `[B, T, HV, Dk]` summed onto its key head's."""
        return x.reshape(*x.shape[:2], heads // call.group, call.group, width).sum(3)[..., :run.key_width]

    # The launches' gradients in the operands' dtype (fla's), summed in fp32, the held operand's parts joined.
    dv = call.held.joined(dv) if delta else dv
    dk_w = None if dk_w is None else call.held.joined(dk_w)
    extra = 0.0 if dk_w is None else keyed(headed(dk_w, width))
    grads = _unlaid(run, (q, k, v, g, beta), dq, dk, dv, run.unlaid(dg), run.unlaid(dbeta), dh0, starts,
                    extra_dk=extra)
    return (*_tabled_gradients(run, grads, taken), dskip, _entered_gradient(run, taken)), kept_


def _taken_up(run, taken):
    """The gradient of the memory each chunk takes up (`taken`, the reverse walk's, `[chunks * HV * Dk, Dv]`),
    `[NC, HV, Dk, Dv]`, zero at a chunk that takes nothing up (which the walk wrote nothing to)."""
    call, table = run.call, run.table
    taken = taken[table.pad:taken.shape[0] - table.pad]
    taken = taken.reshape(-1, call.heads, call.width, call.value_width)
    return jnp.where((table.loads >= 0)[:, None, None, None], taken, 0.0)


def _turned_back(run, taken):
    """`taken` (`_taken_up`'s) turned back by each chunk's load rotation (`table.Table.turning`), its slots'
    own frames; as it is where the call turns none."""
    table = run.table
    if table.turning is None:
        return taken
    rows, after, before = (part.reshape(*taken.shape[:2], -1)[..., None] for part in table.turning)
    shifted = jnp.pad(taken, ((0, 0),) * 2 + ((1, 1), (0, 0)))
    return rows * taken - after * shifted[..., 2:, :] - before * shifted[..., :-2, :]


def _tabled_gradients(run, grads: tuple, taken) -> tuple:
    """`grads` (`_unlaid`'s) with a forked call's entering memories' gradient its given memories', `[B, S',
    HV, Dk, Dv]`: each the sum of what the chunks that took it up gave it (`taken`); `grads` as they are
    for another call."""
    if taken is None:
        return grads
    table = run.table
    # Each chunk taking a given slot up adds to its row's slot (a stored slot's, its storing chunk's walk),
    # its gradient turned back into the slot's frame where the call turns it (`_turned_back`); a slot a chunk
    # adds (`Table.adds`) takes the same gradient as the one it loads.
    returned = _turned_back(run, _taken_up(run, taken))
    given = jnp.zeros((run.batch, table.count, *returned.shape[1:]), jnp.float32)
    for slots in (table.loads, *(() if table.adds is None else (table.adds,))):
        given_slot = (slots > 0) & (slots <= table.count)
        given = given.at[table.rows, jnp.clip(slots - 1, 0, table.count - 1)].add(
            jnp.where(given_slot[:, None, None, None], returned, 0.0), mode="drop")
    return (*grads[:5], given[..., :run.key_width, :run.value_width])


def _entered_gradient(run, taken):
    """A forked call's entry write's gradient `[NC, HV, Dk, Dv]` (`chunk_forward.ForkedTable.entries`): the
    memory's each chunk takes up, which its entry write adds to; None where the call gives none."""
    if taken is None or run.table.entries is None:
        return None
    return _taken_up(run, taken)[..., :run.key_width, :run.value_width]

def _unlaid(run, operands: tuple, dq, dk, dv, dg, dbeta, dh0, starts, *, extra_dk=0.0) -> tuple:
    """The backward's gradients laid back as the call's operands (`operands`, `(q, k, v, g, beta)`) and in
    their dtypes: the queries' and keys' `[R, H * Dk]` a key head's, `extra_dk` `[B, T, H, Dk]` added to the
    keys', the values' `[R, HV * Dv]`, the log decay's and write gates' as the call's (laid back already), and
    the entering memories' `[B * HV * Dk, Dv]` (`chunk_forward._Run.shaped`), zero for a row a document
    starts."""
    call = run.call
    q, k, v, g, beta = operands
    rows, heads, width, columns = call.rows, call.heads, call.width, call.value_width

    def key_headed(x):
        """`[R, H * Dk]`, a key head's, laid back as `[B, T, H, Dk]`."""
        laid = run.unlaid(x.astype(jnp.float32).reshape(rows, heads // call.group, width))
        return laid[..., :run.key_width]

    dv = run.unlaid(dv.reshape(rows, heads, columns))[..., :run.value_width]
    dh0 = run.shaped(dh0)
    if starts is not None:
        dh0 = jnp.where(jnp.asarray(starts, bool)[:, 0, None, None, None], 0.0, dh0)
    return (key_headed(dq).astype(q.dtype), (key_headed(dk) + extra_dk).astype(k.dtype), dv.astype(v.dtype),
            dg.astype(g.dtype), dbeta.astype(beta.dtype), dh0)


# The fast backward of a decay a head under the delta rule (fla's DAG through the algebra's backward tile
# functions, `algebra.local_gradient` ... `algebra.summed_ranges`), a 16-bit call's: the memory's reverse
# recurrence a sequence at a time (`_reverse`), each chunk's scores' gradient (`_scores_back`), the queries',
# keys' and weights' gradients a key block of a chunk at a time (`_query_key`), and the WY representation's
# (`_system_back`, `_values_back`, `_keys_back`); each launch writes its decay's terms a chunk and value
# head, its pairs' summed token by token (`summed_pairs`) and its ranges' by XLA's cumulative sums, which
# XLA adds.



def _local_back_held(call: Call, vb: int) -> Held:
    """`_local_back`'s consumer thread at value block `vb`: the local scores, their decays and masks; then
    their narrow form's product with a block of the output's gradient and its cast (`algebra.local_gradient`:
    the scores are narrowed before the products): as one phase, an H100's model stated 248 registers where
    ptxas needed 136, an A100's 280 against 160."""
    chunk = call.chunk
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    # The keys against the queries read transposed (`algebra.local_gradient`'s 'local_t'); then the narrow
    # scores against a block of the output's gradient ('local_dv').
    return Held(phases=(
        [f32(chunk, chunk)] * 2 + masks(chunk, chunk)
        + product_fragments(call, (given, chunk, call.block), (given, call.block, chunk),
                            rhs_transposed=True),
        [*held_tile(held, chunk, chunk), f32(chunk, vb), *held_tile(_value_grads(call), chunk, vb),
         *product_fragments(call, (held, chunk, chunk), (given, chunk, vb))]))


def _local_back_staged(call: Call, vb: int) -> tuple:
    """`_local_back`'s staging tiles and landings at value block `vb`: the scores' narrow form and a value
    block's gradient's landing."""
    chunk = call.chunk
    held = dataclasses.replace(call.held, rows=call.rows)
    return staging_tiles(held, (chunk, chunk)), {"row": _value_grads(call).landing(chunk, vb)}


@lowered_once
def _local_back(q, k, do, hi, lo, taken, call: Call, *, scale: float):
    """`[R, HV * Dv]` in `_value_grads`' form: the corrected values' gradient from each chunk's own output
    (`algebra.local_gradient`, fla's chunk_bwd_dv_local), its scores in the held operand's form (fla's
    rounding under the delta rule and mamba_ssm's for a rounded call; fp32's three parts for an unrounded call
    without the delta rule, as its forward's output holds them: rounded to the operands' 16 bits, its write
    gate's gradient `sum(v du)` was 1.4e-3 from float64 against the portable math's 9e-8), a program a chunk
    and value head (persistent): apart from the memory's reverse walk (`_reverse`), whose tiles beside these
    spilled 80 bytes a thread on an H100's."""
    chunk = call.chunk
    width, value_width, family = call.width, call.value_width, call.family
    given, grads = call.given, _value_grads(call)
    # A value row a block at a time, from the whole width down to whole value tiles (`Call.value_block`).
    vb = fitting_block(call, functools.partial(_local_back_held, call),
                       functools.partial(_local_back_staged, call), call.value_width, call.value_block)

    def program(at, q_ref, k_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref, dv_ref, *,
                streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        views = stream_views(queries=given.views(q_ref, row, chunk, keyed, width),
                             keys=given.views(k_ref, row, chunk, keyed, width),
                             out=given.views(do_ref, row, chunk, h * value_width, value_width))
        vectors = {"gate": hi_ref, "low": lo_ref, "column_gate": column_hi_ref, "column_low": column_lo_ref}
        streamed = family.copies_vectors
        if streamed:
            views |= vector_views(n, h, **vectors)

        read = prefix_reader(family, streamed, vectors, h, span)

        def step(index, slots, carry, accumulators, loaded=None):
            del index
            found = read(slots) if streamed else loaded
            assert found is not None, "a family that streams no vectors loads them in the prologue"
            (gate, low), laid = found
            (local,) = staging

            def block(name, c):
                return stream_tile(slots, name, given, columns=pl.ds(c * call.block, call.block))

            def stored(j, dv):
                grads.store(dv_ref, dv, row, chunk, h * value_width + j * vb, vb, family=family,
                            smem=landing["row"])

            # The local scores once, staged for every value block's product with the output's gradient.
            issue = Issuer(call, {"local_dv": [local]})
            algebra.local_gradient(
                functools.partial(block, "keys"), functools.partial(block, "queries"), width // call.block,
                gate, low, lambda j: stream_tile(slots, "out", given, columns=pl.ds(j * vb, vb)),
                                   value_width // vb,
                scale, call.held.dtype, issue, issue.summed, emit=stored, laid=laid, apart=call.apart)
            return carry, accumulators

        return mosaic.Walk(views, step, prologue=None if streamed else lambda: read(None))

    tiles = stream_blocks(queries=((chunk, width), given), keys=((chunk, width), given),
                          out=((chunk, value_width), given))
    tiles |= vector_blocks(call, "gate", "low", "column_gate", "column_low")
    # The local scores, then their product with the output's gradient: the model's.
    model = 2 * chunk * chunk * (width + value_width)
    staging, landing = _local_back_staged(call, vb)
    return launch_kernel(program, call, out_type=grads.shape(call.heads * value_width),
                         grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=tiles,
                         stages=1,
                         live=chunk_liveness(7), steps=1, model=model, held=_local_back_held(call, vb),
                         staging=staging,
                         landing=landing, persistent=True)(
        q, k, do, hi, lo, hi, lo, taken)


def _gate_back(run, call: Call, dv2, gated: list, laid_do, skip) -> tuple:
    """`(dv, dbeta, dskip)` without the delta rule, where the values gated, `u = v beta`, are the corrected
    ones: the reverse walk's gradients through the gate and the skip and its value blocks' sums of beta's and
    the skip's, where it formed them (`gated`); else the same from its gradient of the values gated `dv2`,
    elementwise in fp32, beta's sum over the value columns in the rows' own layout, and the skip's from the
    output's gradient `laid_do`. `dskip` None without a `skip`."""
    if gated:
        dbeta, *skipped = gated
        # The skip's terms summed over the call's tokens alone: the walk visits the chunks of the rows'
        # sequences and leaves a laid chunk no sequence takes unwritten (non-finite where a packed call's
        # table holds one, the SSD's skip gradient on every draw, M-JS-63).
        dskip = compensated_sum(run.unlaid(jnp.moveaxis(skipped[0], 2, 0)), (0, 1, 2)) if skipped else None
        return dv2, dbeta.sum(0).T, dskip
    rows, heads, columns = call.rows, call.heads, call.value_width
    du = dv2.astype(jnp.float32).reshape(rows, heads, columns)
    values = call.given.joined(run.v).reshape(rows, heads, columns)
    (dv, dbeta), dskip = algebra.gated_values_back(du, values, run.beta.T), None
    if skip is not None:
        read = call.given.joined(laid_do).reshape(rows, heads, columns)
        skipped, dskip = algebra.skipped_back(read, values, skip.astype(jnp.float32)[:, None])
        dv, dskip = dv + skipped, compensated_sum(dskip, (0,))
    return dv.reshape(rows, heads * columns), dbeta, dskip


def _reverse_held(call: Call, bv: int, *, gates: bool, skips: bool = False) -> Held:
    """What `_reverse`'s consumer thread holds at value block `bv`: the memory's gradient carried, its key
    slices in fp32, and the one it gives the chunk before as it is summed; a step's two fp32 tiles at once,
    the corrected values' gradient beside the local one it adds or the decayed output's gradient read after
    it (`algebra.reverse_carried`: the local one folds into it before the read forms), beside the output's
    gradient, a memory slice as an operand, and the stored tiles' casts; where the walk `gates` the values,
    the values beside the gradient for beta's sum, and where it `skips`, the output's gradient widened for
    the skip's; under a decay a key dimension a key slice of the memory entered, widened for the memory
    decay's sum."""
    chunk = call.chunk
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    return Held(carried=[f32(call.width, bv)] * 2, phases=(
        [f32(chunk, bv)] * (2 + gates + skips) + [f32(call.block, bv)] * call.per_key
        + held_tile(given, chunk, bv) * (1 + gates)
        + held_tile(held, call.block, bv) + held_tile(call.memories, call.block, bv)
        + held_tile(_reverse_values(call, gates=gates), chunk, bv)
        + _reverse_products(call, bv),
        [f32(chunk, bv)] * (2 + gates + skips) + [f32(call.block, bv)] * _reverse_sums(call)))


def _reverse_sums(call: Call) -> int:
    """The fresh sums `_reverse`'s step holds at once beside its two fp32 tiles, its operands staged in shared
    memory: where its products' operands take three parts (`Issuer.apart`: a write held whole, lightning
    attention's, its memory's gradient and the decayed output's in fp32's three parts) and a family's
    products overlap, every key slice's terms of both its products ('values_dv' and 'reads',
    `algebra.reverse_carried`), each term its own accumulator, as the states and output walks state theirs;
    else none. Unstated, lightning attention's reverse walk on an H100 (four heads of 128) stated 105
    registers at a 16-wide value block where its body held 133 at its peak, twelve such sums, under the 128 a
    thread launches with, so no register moved to its consumers and ptxas spilled 168 bytes at every rung
    of its ladder; at 32 wide 199 against 255, which ptxas spilled at every rung and a program a
    multiprocessor (row 6816)."""
    keyed, held, read = _reverse_operands(call)
    if not call.family.products_overlap or max(keyed.parts, held.parts, read.parts) <= 2:
        return 0
    return (call.width // call.block) * (len(product.terms(keyed.parts, held.parts))
                                         + len(product.terms(keyed.parts, read.parts)))


def _reverse_operands(call: Call) -> tuple[Operand, Operand, Operand]:
    """`_reverse`'s step's product operands (`algebra.reverse_carried`): the keys (the held memory's rows
    under a decay a key dimension, the keys as given else), the held memory's gradient, and the output's
    gradient read (`reads`: the gradient as given under a decay a key dimension, else decayed in fp32 and
    staged in two parts under a decay a head, a value tile's own parts where it holds more)."""
    held = dataclasses.replace(call.held, rows=call.rows)
    if call.per_key:
        return held, held, call.given
    return call.given, held, Operand(call.rows, 2 if held.parts == 1 else held.parts, jnp.dtype(jnp.float32))


def _reverse_products(call: Call, bv: int) -> list:
    """The fragments of `_reverse`'s step's products, issued in turn (`algebra.reverse_carried`): the keys
    against the memory's gradient staged ('values_dv'); the queries read transposed against the output's
    gradient read (`reads`: decayed in fp32 and staged in two parts under a decay a head, a value
    tile's own parts where it holds more, the gradient as given under a decay a key dimension); and with the
    delta rule the weights read transposed against the corrected values' gradient ('weights_dv')."""
    chunk, (keyed, held, read) = call.chunk, _reverse_operands(call)
    return largest_live_set(
        product_fragments(call, (keyed, chunk, call.block), (held, call.block, bv)),
        product_fragments(call, (keyed, call.block, chunk), (read, chunk, bv)),
        *([product_fragments(call, (held, call.block, chunk), (held, chunk, bv))] if call.delta else []))


def _reverse_views(call: Call, q_ref, k_ref, w_ref, do_ref, local_ref, v_ref, h, walked, walked_width: int
                   ) -> dict:
    """`_reverse`'s walked streams' views for value head `h`'s `walked_width` columns from `walked`: step `f`
    the chunk `chunks - 1 - f`, a sequence's chunks from its last, walked forward; the values' where the walk
    gates them (`v_ref` given)."""
    width, value_width, given = call.width, call.value_width, call.given
    keyed, key_columns = _reverse_keyed(call, h)
    value_columns = h * value_width + walked
    held = dataclasses.replace(call.held, rows=call.rows)
    return stream_views(lambda f: (call.chunks - 1 - f, 0),
                        queries=keyed.views(q_ref, 0, call.rows, key_columns, width),
                        keys=keyed.views(k_ref, 0, call.rows, key_columns, width),
                        **({"w": held.views(w_ref, 0, call.rows, h * width, width)} if call.delta else {}),
                        out=given.views(do_ref, 0, call.rows, value_columns, walked_width),
                        local=_value_grads(call).views(local_ref, 0, call.rows, value_columns, walked_width),
                        **({"values": given.views(v_ref, 0, call.rows, value_columns, walked_width)}
                           if v_ref is not None else {}))


def _entered_views(call: Call, entered_ref, h, walked, walked_width: int) -> dict:
    """The per-key reverse walk's streams of the memory each chunk entered (`memory`, the forward's, in
    `call.memories`' form), a key slice `entered{i}` each, value head `h`'s `walked_width` columns from
    `walked`: step `f` the chunk `chunks - 1 - f`'s slice, as `_reverse_views`' steps, its ring prefetching it
    for the memory decay's sum. Read from global memory in the step, its load stalled the walk's consumer a
    quarter of Kimi Linear's reverse walk on an emulated B200 (754k cycles, a load of the accumulator's layout
    at the walk's every key slice, HS's tip2 ranking, nvgpu-lean run 20261114203056-aebe0292)."""
    slices = call.width // call.block
    whole = call.memories.views(entered_ref, 0, call.chunks * call.heads * call.width, walked, walked_width)
    views: dict = {}
    for i in range(slices):
        # The slice's block of rows: chunk-major, then head, then key slice (`memory`'s rows).
        views |= stream_views(lambda f, i=i: (((call.chunks - 1 - f) * call.heads + h) * slices + i, 0),
                              **{f"entered{i}": whole})
    return views


def _reverse_keyed(call: Call, h) -> tuple[Operand, object]:
    """The form and columns of the queries and keys `_reverse` takes for value head `h`: the key head's as
    given under a decay a head; under one a key dimension each value head's own, gated (`key_gated`'s
    queries, `key_decayed_write`'s keys) and held."""
    if call.per_key:
        return dataclasses.replace(call.held, rows=call.rows), h * call.width
    return call.given, h // call.group * call.width


def _reverse_launched(program, call: Call, tiles: Tiles, batch: int, sequences: int, *, gates: bool,
                      skips: bool, table: Table | None):
    """`_reverse`'s `program` launched (`launch_kernel`): its outputs, its walked streams and its final
    memories' streams, a program walking `tiles.walkers` value blocks; where the walk `gates` the values,
    their stream and beta's sums, where it `skips`, the skip's sums, under a decay a key dimension the
    memory decay's, and a forked call's `table`'s (`table_returned`); its staging and landing tiles, its
    walk over the laid sequences' mean of chunks and its register model (`_reverse_held`, `table_held`)."""
    chunk, bv = call.chunk, tiles.block_v
    walked_width = bv * tiles.walkers
    width, value_width, given = call.width, call.value_width, call.given
    final = Operand(batch * call.heads * width, 1, jnp.dtype(jnp.float32))
    values = _reverse_values(call, gates=gates)
    out_type = (call.memories.shape(value_width), values.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((batch * call.heads * width, value_width), jnp.float32),
                *((jax.ShapeDtypeStruct((value_width // bv, call.heads, call.rows), jnp.float32),)
                  * (gates + skips)),
                *((jax.ShapeDtypeStruct((value_width // bv, call.chunks * call.heads * width), jnp.float32),)
                  * call.per_key))
    keyed = _reverse_keyed(call, 0)[0]
    blocks = stream_blocks(queries=((chunk, width), keyed), keys=((chunk, width), keyed),
                           **({"w": ((chunk, width), dataclasses.replace(call.held, rows=call.rows))}
                              if call.delta else {}),
                           out=((chunk, walked_width), given),
                           local=((chunk, walked_width), _value_grads(call)),
                           **({"values": ((chunk, walked_width), given)} if gates else {}))
    if call.per_key:
        # The memory each chunk entered, a key slice a stream (`_entered_views`).
        blocks |= stream_blocks(**{f"entered{i}": ((call.block, walked_width), call.memories)
                                   for i in range(width // call.block)})
    finals = stream_blocks(**{f"final{i}": ((call.block, walked_width), final)
                              for i in range(width // call.block)})
    out_type = (*out_type, *(() if table is None else table_returned(call, table, batch)))
    # The memory's carried gradient: three memory products a chunk, two without the delta rule's weights.
    steps = call.chunks / sequences
    held = dataclasses.replace(call.held, rows=call.rows)
    final_landing, after = final_memory_landing(call, bv)
    return launch_kernel(program, call, out_type=out_type, grid=(value_width // walked_width, sequences,
                                                                 call.heads),
                         # Two deep where the block holds both, else one: fp32's two parts and 256-wide keys
                         # took 237920 and 282848 bytes, past an H100's 232448 (fp32-plan-sm90, cdcd620ef).
                         grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=(1, 2),
                         steps=steps, model=2 * (2 + call.delta) * chunk * width * walked_width * steps,
                         held=table_held(_reverse_held(call, bv, gates=gates, skips=skips), call, table, bv),
                         consumers=tiles.walkers,
                         live=sequence_live(8),
                         staging=(*staging_tiles(held, *dict.fromkeys(((call.block, bv), (chunk, bv)))),
                                  (plgpu.SMEM((chunk, bv), held.stored),)),
                         once=finals,
                         landing={**memory_landing(call, bv), "corrected": values.landing(chunk, bv),
                                  **final_landing, **table_landing(call, table is not None, bv)},
                         after=after)


@lowered_once
def _reverse(q, k, w, do, local, hi, lo, dht, sequences, v, beta, call: Call, tiles: Tiles, *, scale: float,
             skip=None, memory=None, table: Table | None = None):
    """`(dh [chunks * HV * Dk, Dv], dv [R, HV * Dv], dh0 [B * HV * Dk, Dv])`: the gradient of the memory each
    chunk leaves as `call.memories`, the corrected values' gradient in `_reverse_values`' form (their own
    output's, `local`, `_local_back`'s, and the memory's, `algebra.reverse_carried`) and each row's initial
    memory's gradient, fp32, each sequence's chunks from its last over a value block, the states launch's
    (`tiles.block_v`; fla's dhu). `q`, `k`, `w`, `do`, `local` and `v` are laid as the forward's, `beta`
    `[HV, R]`; `dht` `[B * HV * Dk, Dv]` fp32 each row's final memory's gradient. A program walks
    `tiles.walkers` value blocks of its sequence and head, one a consumer role, as the states launch does.
    Without the delta rule `w` is None and the memory's gradient takes no `w^T dv`
    (`algebra.reverse_carried`); nothing was corrected but the values gated, `u = v beta`, and where `v` and
    `beta` are given the walk gates them: `dv` is the values' own, `du beta`, and a fourth output `[Dv / BV,
    HV, R]` fp32 each value block's part of beta's, `sum(v du)` over its columns, from the tile the walk holds
    (mamba_ssm's `_chunk_scan_chunk_state_bwd_dx` forms its `ddt` so; `gradients` says where); with a `skip`
    `[HV]` fp32 there, `dv` takes `D do` too and a fifth output the skip's, `sum(v do)`, as that kernel forms
    `dD`. Under a decay a key dimension `memory` is the memories each chunk enters (`call.memories`), and a
    fourth output `[Dv / BV, chunks * HV * Dk]` fp32 each value block's part of the memory decay's gradient
    `sum(h dh)` (`algebra.key_memory_decay`) from the memory's gradient the walk holds, its key rows a
    tile's (kda-lower5-sm90 at 6e03eb57e: where the keys' columns met them, no layout held). A forked call's
    `table` returns its slots' gradients (`Returning`): the gradient of the memory each chunk takes up, an
    output after the rest (`table_returned`'s first; its stored slots' sums no caller reads)."""
    gates, skips = v is not None, skip is not None
    chunk, width, value_width, family = call.chunk, call.width, call.value_width, call.family
    grads = _reverse_values(call, gates=gates)
    bv, slices, walkers = tiles.block_v, width // call.block, tiles.walkers
    walked_width = bv * walkers
    memories, given = call.memories, call.given
    batch, chunks = dht.shape[0] // (call.heads * width), call.chunks
    final = Operand(dht.shape[0], 1, jnp.dtype(jnp.float32))
    assert gates or not skips, "the walk adds a skip where it gates the values"
    producer = call.producer(_reverse)
    assert call.per_key == (memory is not None), "the walk takes the memories under a decay a key dimension"
    returns = 0 if table is None else 1 + (table.held > 0)

    def program(at, q_ref, k_ref, w_ref, do_ref, local_ref, hi_ref, lo_ref, dht_ref, seq_ref, v_ref, beta_ref,
                *refs, streams, staging, landing):
        skip_ref, refs = (refs[0], refs[1:]) if skips else (None, refs)
        entered_ref, refs = (refs[0], refs[1:]) if call.per_key else (None, refs)
        taken = 0 if table is None else len(table.returning)
        table_refs, refs = refs[:taken], refs[taken:]
        refs, returned_refs = refs[:len(refs) - returns], refs[len(refs) - returns:]
        dh_ref, dv_ref, dh0_ref, *gated_ref = refs
        j, s, h = at["blocks"], at["sequences"], at["heads"]
        sequence = tables.unpacked(Walked, lambda i: seq_ref[s, i])
        row = sequence.row
        roled = role_tiles(staging, landing, walkers)
        walked = j * walked_width

        def memory(base, i):
            return (base * call.heads + h) * width + i * call.block

        views = _reverse_views(call, q_ref, k_ref, w_ref, do_ref, local_ref, v_ref if gates else None, h,
                               walked, walked_width)
        if entered_ref is not None:
            views |= _entered_views(call, entered_ref, h, walked, walked_width)
        final_views = stream_views(**{f"final{i}": final.views(dht_ref, memory(row, i), call.block, walked,
                                                               walked_width) for i in range(slices)})
        walked_streams = family.opened(streams["walked"], views)
        leaving = family.opened(streams["once"], final_views)

        # Each key slice of the memory's gradient and the corrected values', a store each in flight.
        before_store = walk_stores(family, width // call.block + 1)

        def step(role):
            (*staged, (second,)), lands = roled(role)
            sliced, chunked = staged[0], staged[-1]  # a key slice's operand, a chunk's where wider
            # The decayed output gradient fp32 as two parts against the queries (fla's split product, 16 of
            # its significand's bits): one part put the initial memory's gradient 5 to 7 times the port's
            # distance from float64. Its first part reuses the memory products' tile, each consumed before
            # the next stages; an fp32 call's two parts are the memory products' own (a third took the
            # product's parts apart, each its own accumulator: 277 values live at the walk's peak against a
            # 16-bit call's 133, 100 bytes spilled on an H100, back-need at 5cec06bd8).
            issue = Issuer(call, {"values_dv": sliced, "weights_dv": chunked,
                                   "reads": chunked if len(chunked) > 1 else (*chunked, second)})
            return functools.partial(_reverse_step, lands, issue, walked + role * bv, pl.ds(role * bv, bv))

        def _reverse_step(lands, issue, column, columns, f, slots, grad, accumulators):
            n = chunks - 1 - f
            span = pl.ds(n * chunk, chunk)
            for i, part in enumerate(grad):
                before_store()
                memories.store(dh_ref, part, memory(n, i), call.block, column, bv, family=family,
                               smem=lands[f"memory{i}"], wait=False)
                if entered_ref is not None:
                    entered = stream_tile(slots, f"entered{i}", call.memories, columns=columns)
                    gated_ref[0].at[column // bv, pl.ds(memory(n, i), call.block)][...] = (
                        algebra.key_memory_decay(entered.value(), part))

            out = stream_tile(slots, "out", given, columns=columns)
            grad, dv = _reverse_carried(call, slots, grad, out, columns, issue, (hi_ref, lo_ref), h, n,
                                        scale=scale)
            if gates:
                dv = _gated_values_stored(call, slots, columns, out, dv, (beta_ref, skip_ref, gated_ref), h,
                                          span, column // bv)
            before_store()
            grads.store(dv_ref, dv, n * chunk, chunk, h * value_width + column, bv, family=family,
                        smem=lands["corrected"], wait=False)
            return grad, accumulators

        # The sequence walked from its last chunk: its gradient the final's where it gives one, the initial
        # memory's where it takes one.
        backward = sequence._replace(first=chunks - sequence.first - sequence.chunks, takes=sequence.gives,
                                     gives=sequence.takes)
        sequence_walk(call, final, "final", (leaving, final_views), (walked_streams, views), step, backward,
                      lambda role, i: dh0_ref.at[pl.ds(memory(row, i), call.block),
                                                 pl.ds(walked + role * bv, bv)], bv,
                      landing=lambda role: roled(role)[1].get("final"), walkers=walkers, producer=producer,
                      table=returned_by(table, table_refs, returned_refs, call, h,
                                        lambda role: walked + role * bv, bv, row,
                                        lambda role: roled(role)[1]["table"]))

    # The keys take the place of weights without the delta rule, and the keys and the prefix that of values
    # a walk gates none of: operands no step reads.
    launched = _reverse_launched(program, call, tiles, batch, sequences.shape[0], gates=gates, skips=skips,
                                 table=table)(
        q, k, k if w is None else w, do, local, hi, lo, dht, sequences, k if v is None else v,
        hi if beta is None else beta, *(() if skip is None else (skip.astype(jnp.float32),)),
        *(() if memory is None else (memory,)), *(() if table is None else table.returning))
    # A forked call's stored slots' gathered gradients the walk alone reads.
    return launched[:-1] if returns == 2 else launched


def _reverse_carried(call: Call, slots, grad: list, out: Parts, columns, issue: Issuer, prefix: tuple, h,
                     n, *, scale: float) -> tuple:
    """`_reverse`'s step's algebra on chunk `n` of head `h`: the memory's gradient carried through the chunk
    and the corrected values' (`algebra.reverse_carried`; under a decay a key dimension
    `algebra.key_reverse_carried`, each key slice's rows decayed across the chunk by `exp2(g_last)` a
    column's), from the walk's tiles (`slots`, the value block's `columns`) and the prefix's halves'
    refs."""
    chunk = call.chunk
    family, held, width = call.family, dataclasses.replace(call.held, rows=call.rows), call.width
    span, last = pl.ds(n * chunk, chunk), n * chunk + chunk - 1

    def block(name, operand, i):
        return stream_tile(slots, name, operand, columns=pl.ds(i * call.block, call.block))

    local = stream_tile(slots, "local", _value_grads(call), columns=columns).value().astype(jnp.float32)
    keyed = _reverse_keyed(call, h)[0]
    keys, queries = (functools.partial(block, name, keyed) for name in ("keys", "queries"))
    weights = functools.partial(block, "w", held) if call.delta else None
    if call.per_key:
        kept = [algebra.raised(*(family.rows_of(ref.at[last, pl.ds(h * width + i * call.block, call.block)])
                                 for ref in prefix))[:, None] for i in range(width // call.block)]
        return algebra.key_reverse_carried(grad, keys, weights, queries, out, local, kept, held.dtype, issue)
    gate, low = (family.rows_of(ref.at[h, span]) for ref in prefix)
    ending, ending_low = (jnp.full((chunk,), ref[h, last]) for ref in prefix)
    kept = [algebra.raised(*(jnp.full((call.block,), ref[h, last]) for ref in prefix))[:, None]] * len(grad)
    return algebra.reverse_carried(grad, keys, weights, queries, out.value(), local, gate, low, ending,
                                   ending_low, kept, scale, held.dtype, issue)


def _values_gated_back(call: Call, slots, columns, out: Parts, dv, betas, skip) -> tuple:
    """`_reverse`'s gate on the values its walk holds (`algebra.gated_values_back`) and, with a `skip`, the
    skip's (`algebra.skipped_back`) on the output's gradient `out`: `(dv, beta's terms)`, then the skip's."""
    values = stream_tile(slots, "values", call.given, columns=columns).value().astype(jnp.float32)
    dv, beta_grad = algebra.gated_values_back(dv, values, betas)
    if skip is None:
        return dv, beta_grad
    skipped, skip_grad = algebra.skipped_back(out.value().astype(jnp.float32), values, skip)
    return dv + skipped, beta_grad, skip_grad


def _query_key_views(call: Call, keyed: tuple, served: tuple, memory: tuple, *, row, key_columns, first,
                     chunk, block, scores: tuple, split=0, vq: int | None = None) -> dict:
    """`_query_key`'s streams' views (`stream_views`) for a program's key block `block` of chunk `chunk`, its
    key head's first value head `first`: the queries' and keys' block (`keyed`, their refs), and each step `i`
    the group's `i`-th value head's values, output gradient and corrected values' gradient (`served`) at
    its columns, its entered memory and leaving memory's gradient (`memory`) a head's rows apart, and its
    scores' gradient (`scores`: the ref and the key head; without the delta rule each step the key head's,
    summed over its value heads, `_grouped_scores`); of each value head's columns the `split`-th `vq` (all of
    them where None, `_query_key_split`)."""
    width, value_width, group = call.width, call.value_width, call.group
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    q_ref, k_ref = keyed
    operands = dict(zip(("values", "out", "value_grads"), ((held, served[0]), (given, served[1]),
                                                           (held, served[2])), strict=True))
    if not call.delta:
        # Without the delta rule no weights, so no stream of the corrected values' gradient.
        del operands["value_grads"]
    memory_row = (chunk * call.heads + first) * width + block * call.block
    vq = value_width if vq is None else vq
    # Step `i` the `i`-th value head's columns from the split's, a value head's columns apart: the view from
    # the first head's split to the last head's, `vq` blocks.
    columns, apart, rows = first * value_width + split * vq, value_width // vq, call.chunk
    ds_ref, keyed_head = scores
    scored = (first, group) if call.delta else (keyed_head, 1)
    return {**stream_views(lambda i: (0, i if call.delta else 0), ds=_score_grads(call).views(
                ds_ref, row, rows, scored[0] * rows, scored[1] * rows)),
            **stream_views(lambda i: (0, 0), queries=given.views(q_ref, row, rows, key_columns, call.block),
                           keys=given.views(k_ref, row, rows, key_columns, call.block)),
            **stream_views(lambda i: (0, i * apart),
                           **{name: operand.views(ref, row, rows, columns, (group - 1) * value_width + vq)
                              for name, (operand, ref) in operands.items()}),
            **stream_views(lambda i: (i * (width // call.block), 0),
                           **{name: memories.views(ref, memory_row, group * width, split * vq, vq)
                              for name, ref in zip(("entered", "leaving"), memory, strict=True)})}


def _gradient_stored(call: Call, kind: str, block: int, grad, *, weights: Sequence, sums: dict, first,
                     landing: dict, held_sums: dict | None, split: bool = False) -> None:
    """`algebra.memory_gradients`' emit in `_query_key`: the weights' gradient (`kind` 'dw') in the
    operands' dtype to `weights` (its parts' views) as it is final, as fla's dqkwg writes it; the queries'
    and keys' ('dq', 'dk') summed over the key head's value heads in fp32, the first value head's (`first`)
    taken as it is and the rest added, in the order the steps take them: in their own shared tiles where
    the launch holds them (`held_sums[kind]`, `_query_key_holds`), which the step stores once at the last;
    else into `sums[kind]` in turn (`Family.added`: a TMA reduction, or an A100's element-wise global
    atomics), each through the launch's landing tile. A value split's (`split`) weights' gradient is its
    split's own part of it, fp32, which XLA adds, through its own landing tile."""
    del block
    family = call.family
    if kind == "dw":
        if split:
            family.stored([grad], weights, smem=landing["weights"])
            return
        pieces = product.formed(grad.astype(jnp.float32), call.held.parts) if call.held.parts > 1 else (grad,)
        family.stored([piece.astype(call.held.stored) for piece in pieces], weights, smem=landing["weights"])
        return

    if held_sums is None:
        @pl.when(first)
        def _():
            family.stored([grad], [sums[kind]], smem=landing["sums"])

        @pl.when(jnp.logical_not(first))
        def _():
            family.added(grad, sums[kind], call.interpret, smem=landing["sums"])
        return

    @pl.when(first)
    def _():
        plgpu.store(held_sums[kind], grad, optimized=False)

    @pl.when(jnp.logical_not(first))
    def _():
        _summed_into(call, held_sums[kind], grad)


def _scores_summed(call: Call, issue, ds, keys, queries, *, sums: dict, landing: dict,
                   held_sums: dict | None) -> None:
    """A split's scores' gradient's products (`ds @ k` the queries', `ds^T @ q` the keys') added to its
    queries' and keys' sums, after its memory products: into their held tiles (`_gradient_stored`), or
    each into the output (`Family.added`)."""
    family = call.family
    for kind, lhs, rhs in (("dq", ds, keys), ("dk", ds.T, queries)):
        product_sum = issue("scores_dq", lhs, rhs)
        if held_sums is None:
            family.store_wait(0)
            family.added(product_sum, sums[kind], call.interpret, smem=landing["sums"])
        else:
            _summed_into(call, held_sums[kind], product_sum)


def _sums_stored(call: Call, held_sums: dict, sums: dict, landing, last) -> None:
    """The key head's queries' and keys' sums (`_gradient_stored`'s shared tiles) stored to `sums` once, at
    its last value head's step (`last`), each in turn through the launch's `landing` tile."""
    family = call.family

    @pl.when(last)
    def _():
        for kind in ("dq", "dk"):
            family.store_wait(0)
            family.stored([family.tile_of(held_sums[kind])], [sums[kind]], smem=landing)


def _summed_into(call: Call, tile, value: jax.Array) -> None:
    """`value` added into the fp32 shared `tile` in the family's accumulator layout, which each thread reads
    and writes its own words of (`Family.tile_of`): no thread reads another's, so none waits on them."""
    plgpu.store(tile, call.family.tile_of(tile) + value, optimized=False)


def _memory_sum(call: Call, issue: Issuer, entered: Parts, leaving: Parts) -> jax.Array:
    """`sum(S dS)` over the value columns of a block of the memories entered and left, `[rows]`: the
    diagonal of their product `S dS^T` where the family's products read registers, else their elementwise
    product's rows (`_query_key`)."""
    if call.family.operands_in_registers:
        traced = issue("memory_trace", entered, leaving.T)
        return jnp.sum(jnp.where(diagonal(traced.shape[0], call.family), traced, 0.0), axis=1)
    return jnp.sum(entered.value().astype(jnp.float32) * leaving.value().astype(jnp.float32), axis=1)


def _query_key_split(call: Call) -> int:
    """`_query_key`'s value columns a program takes of each value head: the widest whole number of value
    tiles (`Call.value_block`) dividing the value width whose streams (each step's values, output gradient
    and corrected values' gradient `[C, vq]`, its memories `[block, vq]`, the queries', keys' and scores'
    gradient's blocks) fit the block's shared memory (`generation.shared_budget`) and their live registers
    (`_query_key_held`), a tile where none does:
    an fp32 call's two parts at Qwen3.5-27B's widths took 237824 bytes and a 256-wide value row's 215040 on
    an A100, past its 166912 (fp32-plan-sm90 at cdcd620ef, wide-backward). Halved instead, 192 columns took
    96, a block and a half, and the split's programs read no column past their first block's. Each split's
    bytes are the launch's own (`shared_allocation`, as `launch_kernel` lays it: its streams one deep, its
    landing). Compiled as interpreted: the weights' gradient a split once landed zero natively, `-dw` read
    from its accumulator, because CUDA 12.9's ptxas folded the read's sign flip (an xor) so that the
    product's HGMMA wrote the zero register, not the split (row 4346: the JAX fork's float-sign-ops
    lowering, post66; M-JS-22q native as interpreted)."""
    return next((vq for vq in whole_blocks(call.value_width, call.value_block)
                 if _query_key_fits(call, vq, holds=False)), call.value_block)


def _query_key_fits(call: Call, vq: int, *, holds: bool) -> bool:
    """Whether `_query_key` at value split `vq` fits the block's shared memory (`generation.shared_budget`)
    and a thread's registers (`_query_key_held`), its streams one deep, its landing and, where it `holds` its
    sums, their staging (`_query_key_staging`): each split's bytes the launch's own (`shared_allocation`, as
    `launch_kernel` lays it)."""
    tiles, _ = _query_key_launched(call, vq)
    landing = _query_key_landing(call, call.value_width // vq)
    staging = _query_key_staging(call) if holds else ()
    _, scope, _, _ = shared_allocation(call, tiles, staging, landing)
    registers = _query_key_held(call, vq, holds=holds).registers(call.family, addressed(staging, landing))
    return (resources.shared_bytes(call.family, scope(1)) <= generation.shared_budget(call.target)
            and registers <= call.family.registers(1))


def _query_key_held(call: Call, vq: int, *, holds: bool) -> Held:
    """What `_query_key`'s consumer thread holds (`algebra.memory_gradients`): a block's gradient's fp32 sum
    (one accumulator over the value blocks), that sum gated, its scores' product, and a block's queries or
    keys in fp32 for the range terms; then `sum(S dS)` (`_memory_sum`): the memories' product and its
    diagonal's mask where products read registers, with a block of each operand, else the memories in fp32.
    An accumulator a value block, the SSD's spilled 168 bytes a thread on an A100, where one needed 249
    registers (gateqk at c2808ebbe). Where it `holds` its sums (`_query_key_holds`), a sum's tile read back
    beside the gradient it adds (`_summed_into`): unstated, the scalar write's training at 16-wide value
    tiles spilled at 255 registers on an A100 against a stated 232 (M-JS-191)."""
    chunk, vb = call.chunk, call.value_block
    held, memories, given = call.held, call.memories, call.given

    def read(rows: int, lhs, rhs_parts: int, *, retained: bool = False) -> list:
        # A product of `lhs` `[rows, vb]` and the memories read transposed, their fragments `rhs_parts` times.
        return resources.operand_fragments(
            call.family, rows, vb, call.block, resources.OperandRead.of(lhs, retained=retained),
            resources.OperandRead(memories.stored, rhs_parts, retained=retained, transposed=True))

    # The output's gradient and the corrected values against the memories entered and leaving, read
    # transposed ('memory_dq', 'memory_dk'); then the memories against themselves (`_memory_sum`).
    # The unscored head segment reads each memory in its gradients and again in its trace, with no
    # weights store between them: its staged operands remain live across those products. Each value block
    # has its own operands. A scored head or a weights store ends that phase.
    retained = not call.delta and call.group > 1
    trace_operands = [tile for _ in range(vq // vb if retained else 1)
                      for tile in read(call.block, memories, memories.parts, retained=retained)]
    summed = ([f32(call.block, call.block), *masks(call.block, call.block), *trace_operands]
              if call.family.operands_in_registers else [f32(call.block, vb)] * 2)
    # An fp32 call's products overlapping (warpgroup MMA's) keep its row results `after` and `leaving`
    # beside them: unstated, Qwen3.5's fp32 training claimed 3 programs an H100 at 136 registers, whose cap
    # spilled 36 bytes (jsf-owncap).
    rows = [f32(chunk, 4)] * 2 if call.family.products_overlap and held.parts > 1 else []
    return Held(phases=(
        [f32(chunk, call.block)] * (4 + holds) + rows + read(chunk, given, memories.parts)
        + read(chunk, held, 0), summed))


def _query_key_landing(call: Call, splits: int) -> dict:
    """`_query_key`'s stores' tiles at `splits` value splits: the weights' gradient with the delta rule
    (fp32 where there are two splits or more) and the queries' and keys' sums, stored once each in turn
    (`_sums_stored`)."""
    chunk = call.chunk
    held = dataclasses.replace(call.held, rows=call.rows)
    weights = held.landing(chunk, call.block) if splits == 1 else ((chunk, call.block), jnp.float32)
    return {**({"weights": weights} if call.delta else {}), "sums": ((chunk, call.block), jnp.float32)}


def _query_key_staging(call: Call) -> tuple:
    """`_query_key`'s queries' and keys' sums over a key head's value heads, fp32, a plain shared tile each,
    held across its steps (`_gradient_stored`): staging of their own, as a launch's landing tiles are its
    stores' in turn, which a family whose stores finish when they return keys by shape (`_landing_keys`:
    the two sums one tile there, each value head's 'dk' landing on its 'dq')."""
    return (swizzling.plain((call.chunk, call.block), jnp.float32),) * 2


def _query_key_holds(call: Call, vq: int) -> bool:
    """Whether `_query_key` at value split `vq` holds its sums in shared memory (`_query_key_staging`): where
    its block's shared memory and its registers fit them beside its streams, as `_query_key_split` fits
    those. Else each value head's added into the output in turn (`Family.added`), the same fp32 sums in the
    same order: held, an A100's element-wise global atomics each value head (26% of query_key's stall
    samples, its tensor pipe under 10% busy, M-JS-183) are one store a key head; an sm86's 100 KB block held
    no Qwen3.5-27B's two more tiles. Where its family's stores and adds go through registers alone
    (`Family.stores_through_registers`): a family's TMA reduction adds a tile from shared memory in one copy,
    which held sums only take residency from (an H100's small Mamba-2 query-key walk certified at its default
    residency where main's stepped a program fewer, row 6718's test). A key head of one value head at one
    split adds nothing to hold: its sums stored as they come."""
    if not call.family.stores_through_registers or (call.group == 1 and vq == call.value_width):
        return False
    return _query_key_fits(call, vq, holds=True)


def _query_key_launched(call: Call, vq: int) -> tuple[dict, tuple]:
    """`_query_key`'s streams at value split `vq` (`_query_key_split`) and its outputs' types, each split's
    sums its own (the weights' gradient, the delta rule's alone, in fp32 where there are two splits or
    more)."""
    chunk = call.chunk
    width, held, memories, given = call.width, call.held, call.memories, call.given
    splits = call.value_width // vq
    tiles = stream_blocks(queries=((chunk, call.block), given), keys=((chunk, call.block), given),
                          values=((chunk, vq), held), out=((chunk, vq), given),
                          **({"value_grads": ((chunk, vq), held)} if call.delta else {}),
                          entered=((call.block, vq), memories),
                          leaving=((call.block, vq), memories), ds=((chunk, chunk), _score_grads(call)))
    keyed_out = jax.ShapeDtypeStruct((call.rows, splits * call.heads // call.group * width), jnp.float32)
    terms = jax.ShapeDtypeStruct((splits * width // call.block, call.heads, call.rows), jnp.float32)
    weights = (dataclasses.replace(held, rows=call.rows).shape(call.heads * width) if splits == 1
               else jax.ShapeDtypeStruct((call.rows, splits * call.heads * width), jnp.float32))
    return tiles, (keyed_out, keyed_out, *((weights,) if call.delta else ()), terms, terms,
                   jax.ShapeDtypeStruct((splits * width // call.block, call.heads, call.chunks * call.block),
                                        jnp.float32))


@lowered_once
def _query_key(q, k, corrected, do, dv, h, dh, ds, hi, lo, taken, beta, call: Call, *, scale: float):
    """`(dq, dk [R, H * Dk] fp32, dw [R, HV * Dk] as call.held (None without the delta rule, which forms no
    weights), after, leaving [blocks, HV, R], memory
    [blocks, HV, chunks * block] fp32)`: each chunk's and key block's queries', keys' and weights' gradients
    (fla's dqkwg's memory products, `algebra.memory_gradients`, given the scores' gradient `ds` in
    `_score_grads`' form, `_grouped_scores`': `[R, HV * C]` with the delta rule, each step's value head's
    product; without it `[R, H * C]`, the key head's, its products the first step's alone), a program a key
    head's and a step each value head it serves,
    the queries' and keys' summed over them in fp32 where they leave (each program's own tiles: a register
    carry across the steps spilled), as the port's sum; the decay's range terms over the block's key columns,
    each token's (`algebra.memory_gradients`' `(after, leaving)`, which `gradients` sums over their ranges:
    summed here, their `[C, C]` tiles spilled 376 bytes a thread on an H100's), and the block's rows of `sum(S
    dS)`, the memory's decay's gradient across the chunk. `h` and `dh` the memories each chunk enters and its
    leaving memory's gradient (`call.memories`). Where a value row's streams are past the block's shared
    memory, a program takes a split of each value head's columns (`_query_key_split`), each split's sums its
    own, which XLA adds, the scores' gradient's products the first split's alone."""
    chunk = call.chunk
    width, value_width, family = call.width, call.value_width, call.family
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    grads = _score_grads(call)
    vq = _query_key_split(call)
    holds = _query_key_holds(call, vq)
    splits, blocks, group = value_width // vq, vq // call.value_block, call.group
    keyed_width, key_blocks = call.heads // group * width, width // call.block

    def program(at, q_ref, k_ref, v_ref, do_ref, dv_ref, h_ref, dh_ref, ds_ref, hi_ref, lo_ref, taken_ref,
                beta_ref, *outs, streams, staging, landing):
        held_sums: dict[str, Any] | None = {"dq": staging[0], "dk": staging[1]} if holds else None
        dq_ref, dk_ref, *weights_ref, after_ref, leaving_ref, memory_ref = outs
        c, n, keyed = at["blocks"], at["chunks"], at["heads"]
        split = at["values"]
        row, span, last = n * chunk, pl.ds(n * chunk, chunk), n * chunk + chunk - 1
        key_columns = keyed * width + c * call.block
        first = keyed * group
        views, once = _walked_once(call, _query_key_views(
            call, (q_ref, k_ref), (v_ref, do_ref, dv_ref), (h_ref, dh_ref), row=row, key_columns=key_columns,
            first=first, chunk=n, block=c, split=split, vq=vq, scores=(ds_ref, keyed)))

        def step(index, slots, carry, accumulators, *, scores: bool = True):
            h = first + index

            column = functools.partial(column_tile, call, slots, width=call.value_block)

            def whole(name, operand, i):
                del i
                return stream_tile(slots, name, operand)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            ending, ending_low = (jnp.full((chunk,), ref[h, last]) for ref in (hi_ref, lo_ref))
            first_head = index == 0

            # Each split's sums its own, a split's columns after another's.
            columns = pl.ds(split * keyed_width + key_columns, call.block)
            sums = {"dq": dq_ref.at[span, columns], "dk": dk_ref.at[span, columns]}
            weights_at = h * width + c * call.block
            weights = (() if not weights_ref else
                       held.views(weights_ref[0], row, chunk, weights_at, call.block) if splits == 1 else
                       [weights_ref[0].at[span, pl.ds(split * call.heads * width + weights_at, call.block)]])
            stored = functools.partial(_gradient_stored, call, weights=weights, sums=sums, first=first_head,
                                       landing=landing, held_sums=held_sums, split=splits > 1)

            issue = Issuer(call)
            ds = stream_tile(slots, "ds", grads) if scores else None
            after, leaving = algebra.memory_gradients(
                functools.partial(whole, "queries", given), functools.partial(whole, "keys", given), 1,
                functools.partial(column, "values", held), functools.partial(column, "out", given),
                functools.partial(column, "value_grads", held) if call.delta else None,
                lambda i, j: column("entered", memories, j), lambda i, j: column("leaving", memories, j),
                blocks, ds if splits == 1 else None, gate, low, ending, ending_low, None, scale, issue.summed,
                emit=stored, row_scale=family.rows_of(beta_ref.at[h, span]) if call.gated_keys else None,
                summed_all=issue.summed_together)
            if splits > 1 and ds is not None:
                # The scores' gradient's products the first split's alone, added after its memory products.
                pl.when(split == 0)(lambda: _scores_summed(
                    call, issue, ds, whole("keys", given, 0), whole("queries", given, 0), sums=sums,
                    landing=landing, held_sums=held_sums))
            after_ref.at[split * key_blocks + c, h, span][...] = after
            leaving_ref.at[split * key_blocks + c, h, span][...] = leaving
            # `sum(S dS)` over the block's rows, read here rather than both memories again: where products
            # read registers (mma.sync's), each value block's `S^T dS` diagonal, a product of the memories
            # as they lie in shared memory (their elementwise product held two fp32 tiles beside the step's
            # fragments, past an A100's 255 registers); where they read shared memory, that elementwise
            # product (the product's accumulator spilled 44 bytes a thread on an H100's).
            traced = functools.reduce(jnp.add, (
                _memory_sum(call, issue, column("entered", memories, j), column("leaving", memories, j))
                for j in range(blocks)))
            memory_ref.at[split * key_blocks + c, h, pl.ds(n * call.block, call.block)][...] = traced

            if held_sums is not None:
                _sums_stored(call, held_sums, sums, landing["sums"], index == group - 1)
            # The step's stores and adds waited on by design: Mosaic GPU waits on no copy at a kernel's exit
            # (numerics review 10-26, P3-4).
            family.store_wait(0)
            return carry, accumulators

        # Without the delta rule the scores' products the first step's alone.
        return mosaic.Walk(views, step, steps=group, segments=() if call.delta else (
            (step, 0, 1), (functools.partial(step, scores=False), 1, group)), once=once)

    tiles, out_type = _query_key_launched(call, vq)
    tiles, once = _walked_once(call, tiles)
    # Persistent under the delta rule, whose tile walks one step function (without it, the first step's
    # scores products are a segment of their own, which a persistent tile walks none of): a program a tile
    # paid each tile's setup, its chunk's table entries' dependent loads and its ring's and done barriers,
    # most of a modelled B200's CTA (44.3k cycles for three tiles against a persistent one's 11.6k a tile).
    # One or two deep as its plan hides most latency: 30B's A100 walk 560 us one deep, 516 two (M-JS-204).
    dq, dk, *weights, after, leaving, memory = launch_kernel(
        program, call, out_type=out_type, stages=(1, 2), live=chunk_liveness(10),
        grid=(splits, width // call.block, call.chunks, call.heads // group),
        grid_names=("values", "blocks", "chunks", "heads"), blocks=tiles, once=once or None,
        steps=call.group, model=_query_key_model(call), held=_query_key_held(call, vq, holds=holds),
        landing=_query_key_landing(call, splits), staging=_query_key_staging(call) if holds else (),
        persistent=call.delta, together=(call.block,) * (2 + call.delta + 2 * (splits == 1)))(
        q, k, corrected, do, dv, h, dh, ds, hi, lo, taken, beta)
    # The splits' own sums added (one split's, as they are, where a program takes every value column).
    if splits > 1:
        dq, dk, *weights = (x.reshape(call.rows, splits, -1).sum(1) for x in (dq, dk, *weights))
    after, leaving, memory = (x.reshape(splits, -1, *x.shape[1:]).sum(0) for x in (after, leaving, memory))
    dw = None if not weights else weights[0] if splits == 1 else held.split(weights[0])
    return dq, dk, dw, after, leaving, memory


def _walked_once(call: Call, streams: dict) -> tuple[dict, dict]:
    """`_query_key`'s `streams` (its views or its blocks, by part name) as those its steps stream and those
    its walk loads once (`mosaic.Walk.once`): without the delta rule the key head's queries, keys and scores'
    gradient, which every value head's step reads alike (a plain walk's, on a family whose walks load them,
    `Family.walks_once`; under the delta rule its tiles persist, whose once streams the recurrence's bundles
    do not load, each step its value head's scores' gradient): streamed, a step each of a group's value heads
    loaded them again, the 4B SSD's twelve where one serves."""
    once = () if call.delta or not call.family.walks_once else ("queries", "keys", "ds")
    held = {name: value for name, value in streams.items() if name.rstrip("0123456789") in once}
    return {name: value for name, value in streams.items() if name not in held}, held


def _query_key_model(call: Call) -> float:
    """`_query_key`'s tile's model FLOPs, a step a value head of its group: each value head's three memory
    products (two without the delta rule's weights) and the scores' gradient's two, a key head's two without
    the delta rule."""
    chunk = call.chunk
    scores = 2 * chunk * chunk * call.block
    each = (2 + call.delta) * chunk * call.value_width * call.block + scores * call.delta
    return 2 * (each * call.group + scores * (not call.delta))


def _value_grads(call: Call) -> Operand:
    """The form the corrected values' gradient `du` passes from `_local_back` through `_reverse` in: the held
    operand's for the delta rule (fla rounds its `dv` so); without it fp32, as mamba_ssm accumulates `dx` and
    sums `ddt` from it in fp32 (`_chunk_scan_chunk_state_bwd_dx`): rounded, a token's write gate's gradient
    `sum(v du)` took the rounding of every `du` it sums."""
    return dataclasses.replace(call.held, rows=call.rows) if call.delta else Operand(call.rows, 1,
                                                                                       jnp.dtype(jnp.float32))


def _gated_values_stored(call: Call, slots, columns, out, dv, refs, h, span, slice_index):
    """A chunk's values' gradient through the walk's gate (`_values_gated_back`, at the head's write gate
    and skip), its gate's and skip's terms stored at the chunk's span of their outputs: the gated
    gradient."""
    beta_ref, skip_ref, gated_ref = refs
    skip = None if skip_ref is None else skip_ref[h]
    dv, *terms = _values_gated_back(call, slots, columns, out, dv, call.family.rows_of(beta_ref.at[h, span]),
                                    skip)
    for ref, term in zip(gated_ref, terms, strict=False):
        ref.at[slice_index, h, span][...] = term
    return dv


def _reverse_values(call: Call, *, gates: bool) -> Operand:
    """The form `_reverse` leaves the corrected values' gradient in: where the walk gates the values without
    the delta rule, their gradient leaves it whole, the gate's and the skip's terms in it, its one reader
    the call's own result (`_query_key` streams none): the operands' form, rounded once in the walk where an
    XLA pass narrowed the fp32 tile after it (30 us of Nemotron-H 30B's SSD training on an H100, FS-WHY3).
    Otherwise `_value_grads`', which later launches read."""
    whole = gates and not call.delta and call.given.parts == 1
    return call.given if whole else _value_grads(call)


def _score_grads(call: Call) -> Operand:
    """The form the scores' gradient passes from `_scores_back` to `_query_key` in: the held operand's, each
    value head's, for the delta rule (fla's dqkwg rounds it so); without it the operands', a key head's summed
    over its value heads in fp32 first, as mamba_ssm sums its `dCB` over a group's heads before its products
    round it to bf16 (`_chunk_scan_bwd_dcb`, `_bmm_chunk_bwd`'s `dot_dtype`). Each value head's rounded on
    its own, even in two bf16 parts, a one-token document's C row, its size 1.3e-6 where the rest sat at 0.1
    to 0.5, was 0.242 of it from float64 against mamba_ssm's allowed 0.223 (ssd-c-rows-h100-b9d82d9b7)."""
    held = dataclasses.replace(call.held, rows=call.rows)
    return held if call.delta else dataclasses.replace(call.given, rows=call.rows)


def _head_grads(call: Call) -> Operand:
    """The form `_scores_back` stores each value head's scores' gradient in: `_score_grads`' where a value
    head is its key head's alone (the delta rule's, or a group of one), else fp32, the group's sum's terms
    (`_grouped_scores`)."""
    grouped = not call.delta and call.group > 1
    return Operand(call.rows, 1, jnp.dtype(jnp.float32)) if grouped else _score_grads(call)


def _grouped_scores(ds: jax.Array, call: Call) -> jax.Array:
    """`_scores_back`'s value heads' scores' gradients `[R, HV * C]` fp32 summed over each key head's group
    in `_score_grads`' form, compensated and rounded once (`precision.compensated_sum`): near-zero decays
    leave a token's own ds dominant, its 64 heads cancelling to 4e-4 from terms of order 100 (seed 117), so
    a plain sum lost a quarter of dq/dk (row 2066). Summed in the launch, a program a chunk and key head
    stepping its value heads, Kimi's log-linear levels' 64 heads ran 64 programs a level, 330 us of a GH200
    (M-JS-146ll), and its running pair held sm120's consumer-fed thread past 255 registers (row 10522)."""
    if call.delta or call.group == 1:
        return ds
    summed = compensated_sum(ds.reshape(ds.shape[0], call.heads // call.group, call.group, call.chunk), (2,))
    return _score_grads(call).split(summed.reshape(ds.shape[0], -1))


def _scores_back_held(call: Call, grads: Operand) -> Held:
    """What `_scores_back`'s consumer thread holds: the scores' gradient fp32 and narrow, the pairs'
    decays, the scores and the pair terms, the masks, and the pairs' sum's product: three fp32 tiles at
    once, the decays taken before the scores' product and each block sum into one accumulator
    (`algebra.scores_gradient`; an accumulator a block, four, and an H100's need was 232 in a bf16 call, 192
    in one, back-need-one-32 at 9482e4d26). Where the scores' gradient leaves in parts (an fp32 call's), the
    remainder each part after the first is formed from (`product.formed`): without it the model stated 232
    registers where ptxas spilled 32 bytes past 255 on an H100 (back-need2 at 170dd3a22). Stored fp32 (a key
    head's group's terms, `_head_grads`), the fp32 tile is stored as it is, no narrow copy (sm120's 208
    registers to ptxas at Kimi's log-linear level, jsf-llc120d)."""
    chunk, given, held = call.chunk, call.given, dataclasses.replace(call.held, rows=call.rows)
    narrow = [] if grads.stored == jnp.float32 else held_tile(grads, chunk, chunk)
    phases = ([f32(chunk, chunk)] * (3 + (grads.parts > 1))
              + narrow
              + masks(chunk, chunk)
              # In turn: the queries against the keys read transposed ('local_t'); the output's gradient
              # against the corrected values read transposed ('values_ds'); the pairs' sum's three fp32
              # parts against the exact bf16 triangle (`algebra.scores_gradient`).
              + largest_live_set(
                  product_fragments(call, (given, chunk, call.block), (given, call.block, chunk),
                                    rhs_transposed=True),
                  product_fragments(call, (given, chunk, call.value_block), (held, call.value_block, chunk),
                                    rhs_transposed=True),
                  product_fragments(call, (Operand(call.rows, 3, jnp.dtype(jnp.float32)), chunk, chunk),
                                    (Operand(call.rows, 1, jnp.dtype(jnp.bfloat16)), chunk, chunk))),)
    return Held(phases=phases)


@lowered_once
def _scores_back(q, k, corrected, do, hi, lo, taken, beta, call: Call, *, scale: float):
    """`(ds in `_head_grads`' form, decay [HV, R] fp32)`: each chunk and value head's scores' gradient and
    the decay's gradient from its pair terms (`algebra.scores_gradient`, `algebra.summed_pairs`), fla's
    dqkwg's part the memory's products do not take, a program a chunk and value head; without the delta rule
    a key head's value heads' each fp32, which `_grouped_scores` sums. Persistent: apart from `_query_key`,
    each launch's tiles a thread's registers hold (one launch spilled 1.5 KB a thread on an H100's, SE
    bwd-probe at a49c8da50)."""
    chunk = call.chunk
    width, value_width, family = call.width, call.value_width, call.family
    grads = _head_grads(call)
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    blocks = value_width // call.value_block

    def program(at, q_ref, k_ref, v_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref,
                beta_ref, ds_ref, decay_ref, *, streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        views = stream_views(lambda i: (0, 0), queries=given.views(q_ref, row, chunk, keyed, width),
                             keys=given.views(k_ref, row, chunk, keyed, width))
        views |= stream_views(lambda i: (0, 0), values=held.views(v_ref, row, chunk, h * value_width,
                                                                  value_width),
                              out=given.views(do_ref, row, chunk, h * value_width, value_width))
        vectors = {"gate": hi_ref, "low": lo_ref, "column_gate": column_hi_ref, "column_low": column_lo_ref}
        # The head's vectors streamed where the family copies them, else read in its prologue.
        streamed = family.copies_vectors
        if streamed:
            views |= vector_views(n, h, **vectors)

        read = prefix_reader(family, streamed, vectors, h, span)

        def step(index, slots, carry, accumulators, loaded=None):
            del index
            found = read(slots) if streamed else loaded
            assert found is not None, "a family that streams no vectors loads them in the prologue"
            (gate, low), laid = found

            column = functools.partial(column_tile, call, slots)
            value_column = functools.partial(column, width=call.value_block)

            summed, column_sums = summed_issuer(tuple(staging), call)
            ds, pairs = algebra.scores_gradient(
                functools.partial(column, "queries", given), functools.partial(column, "keys", given),
                width // call.block, functools.partial(value_column, "values", held),
                functools.partial(value_column, "out", given), blocks, gate, low, scale, grads.dtype,
                Issuer(call).summed, laid=laid,
                column_scale=family.columns_of(beta_ref.at[h, span]) if call.gated_keys else None,
                apart=call.apart)
            # The pairs' sum before the scores' gradient's store, so an fp32 call's parts form after the
            # pairs are gone.
            decay_ref.at[h, span][...] = algebra.summed_pairs(pairs, summed, column_sums)
            grads.store(ds_ref, ds, row, chunk, h * chunk, chunk, family=family, smem=landing["square"])
            return carry, accumulators

        return mosaic.Walk(views, step, steps=1, prologue=None if streamed else lambda: read(None))

    tiles = stream_blocks(queries=((chunk, width), given), keys=((chunk, width), given),
                          values=((chunk, value_width), held), out=((chunk, value_width), given))
    tiles |= vector_blocks(call, "gate", "low", "column_gate", "column_low")
    out_type = (grads.shape(call.heads * chunk), jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    # The scores' gradient, the scores and the pairs' sums, each value head's: the model's.
    model = 2 * chunk * chunk * (value_width + width + chunk)
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(8),
                         steps=1, model=model, held=_scores_back_held(call, grads),
                         staging=summing_staging(call), landing={"square": grads.landing(chunk, chunk)},
                         persistent=True)(q, k, corrected, do, hi, lo, hi, lo, taken, beta)


@lowered_once
def _system_back(a, k, v, dw, du, hi, lo, beta, taken, call: Call):
    """`dL [R, HV * C]` as `call.held`: the WY representation's backward's system part a chunk and value head
    at a time (fla's prepare_wy_repr_bwd's, `algebra.system_gradient`) from `w`'s and `u`'s gradients `dw`,
    `du` (`call.held`) and the inverse `a` less its unit diagonal (`_solved`'s), which `_keys_back` takes.
    Apart from `_keys_back` and `_values_back`, each launch's tiles a thread's registers hold (in one they
    spilled 360 bytes a thread on an H100's)."""
    chunk, vb = call.chunk, call.value_block
    width, value_width, family = call.width, call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    # The square's operand, a key block's and a value tile's, each shape's tile one (a chunk as wide as a
    # key block, a value tile as wide as one).
    shapes = (chunk, chunk), (chunk, call.block), (chunk, vb)
    staged = list(dict.fromkeys(shapes))

    def program(at, a_ref, k_ref, v_ref, dw_ref, du_ref, hi_ref, lo_ref, beta_ref, column_hi_ref,
                column_lo_ref, taken_ref, dl_ref, *, streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        square, keyed, valued = (staging[staged.index(shape)] for shape in shapes)
        views = stream_views(lower=held.views(a_ref, row, chunk, h * chunk, chunk),
                             keys=given.views(k_ref, row, chunk, h // call.group * width, width),
                             values=given.views(v_ref, row, chunk, h * value_width, value_width),
                             w_grads=held.views(dw_ref, row, chunk, h * width, width),
                             u_grads=held.views(du_ref, row, chunk, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(column_tile, call, slots)
            value_column = functools.partial(column, width=vb)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            beta = family.rows_of(beta_ref.at[h, span])
            issue = Issuer(call, {"inverse_t": square, "inverse_l": square, "inverse_dx": keyed,
                                  "inverse_du": valued})
            d_l = algebra.system_gradient(
                stream_tile(slots, "lower", held), functools.partial(column, "keys", given),
                width // call.block, functools.partial(value_column, "values", given), value_width // vb,
                functools.partial(column, "w_grads", held), functools.partial(value_column, "u_grads", held),
                beta, gate, low, held.dtype, issue, issue.summed,
                laid=(family.columns_of(column_hi_ref.at[h, span]),
                      family.columns_of(column_lo_ref.at[h, span])))
            held.store(dl_ref, d_l, row, chunk, h * chunk, chunk, family=family, smem=landing["square"])
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(lower=((chunk, chunk), held), keys=((chunk, width), given),
                          values=((chunk, value_width), given), w_grads=((chunk, width), held),
                          u_grads=((chunk, value_width), held))
    # The weighted keys' and values' products into the inverse's gradient, and its products through it: the
    # model's.
    model = 2 * chunk * chunk * (width + value_width + 2 * chunk)
    # A block's keys or values scaled, narrow, beside the inverse's gradient; then it masked, its products
    # through the inverse and the pairs' decays with their masks (`algebra.system_gradient`).
    held_tiles = Held(phases=(
        # In turn: the weights' and the values' gradients against the scaled keys and values read transposed
        # ('inverse_dx', 'inverse_du'); then the masked gradient against the inverse read transposed and the
        # inverse read transposed against its product ('inverse_l', 'inverse_t'; `algebra.system_gradient`).
        [f32(chunk, chunk), f32(chunk, call.block),
         *largest_live_set(
             product_fragments(call, (held, chunk, call.block), (held, call.block, chunk),
                               rhs_transposed=True, rhs_staged=True),
             product_fragments(call, (held, chunk, call.value_block), (held, call.value_block, chunk),
                               rhs_transposed=True, rhs_staged=True))],
        [f32(chunk, chunk)] * 4 + masks(chunk, chunk)
        + largest_live_set(
            product_fragments(call, (held, chunk, chunk), (held, chunk, chunk), rhs_transposed=True,
                              lhs_staged=True),
            product_fragments(call, (held, chunk, chunk), (held, chunk, chunk), rhs_staged=True))))
    return launch_kernel(program, call, out_type=held.shape(call.heads * chunk),
                         grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(10),
                         steps=1, model=model,
                         held=held_tiles,
                         staging=staging_tiles(held, *staged),
                         landing={"square": held.landing(chunk, chunk)})(a, k, v, dw, du, hi, lo, beta, hi,
                                                                         lo,
                                                                         taken)


@lowered_once
def _values_back(a, v, du, beta, taken, call: Call):
    """`dv [R, HV * Dv]` as `call.held` and `dbeta [HV, R]` fp32: the WY representation's backward's values'
    part a chunk and value head at a time (`algebra.values_gradient`): the values' gradient and beta's
    through `u`."""
    chunk, vb = call.chunk, call.value_block
    value_width, family = call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(at, a_ref, v_ref, du_ref, beta_ref, taken_ref, dv_ref, dbeta_ref, *, streams, staging,
                landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        (staged,) = staging
        views = stream_views(lower=held.views(a_ref, row, chunk, h * chunk, chunk),
                             values=given.views(v_ref, row, chunk, h * value_width, value_width),
                             u_grads=held.views(du_ref, row, chunk, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(column_tile, call, slots, width=vb)
            stored = block_stored(call, dv_ref, row, h * value_width, landing["block"], width=vb)

            dbeta_ref.at[h, span][...] = algebra.values_gradient(
                stream_tile(slots, "lower", held), functools.partial(column, "values", given),
                value_width // vb, functools.partial(column, "u_grads", held),
                family.rows_of(beta_ref.at[h, span]), held.dtype, Issuer(call, {"inverse_t": staged}),
                emit=stored)
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(lower=((chunk, chunk), held), values=((chunk, value_width), given),
                          u_grads=((chunk, value_width), held))
    out_type = (held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    model = 2 * chunk * chunk * value_width
    # A value block's gradient in fp32 and narrow, its product through the inverse, and the values in fp32;
    # where products overlap (warpgroup MMA's), the write gates' and their gradient's rows, and past two value
    # blocks the next block's values loaded beside them (unstated, 192-wide values claimed 3 programs an H100
    # at 120 registers, whose cap spilled 32 bytes, jsf-owncap).
    overlapping = ((held_tile(call.given, chunk, vb) if value_width // vb > 2 else []) + [f32(chunk, 4)] * 2
                   if call.family.products_overlap else [])
    held_tiles = Held(phases=(
        [f32(chunk, vb)] * 3 + held_tile(held, chunk, vb) + overlapping
        # The inverse read transposed against a value block's gradient ('inverse_t',
        # `algebra.values_gradient`).
        + product_fragments(call, (held, chunk, chunk), (held, chunk, vb)),))
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(4),
                         steps=1, model=model,
                         held=held_tiles, staging=staging_tiles(held, (chunk, vb)),
                         landing={"block": held.landing(chunk, vb)})(a, v, du, beta, taken)


@lowered_once
def _keys_back(a, k, dw, dl, hi, lo, beta, taken, call: Call):
    """`dk [R, HV * Dk]` as `call.held` and `(dbeta, after, decay [blocks, HV, R])` fp32: the WY
    representation's backward's keys' part a key block of a chunk and value head at a time
    (`algebra.keys_gradient`), given the system's gradient `dl` (`_system_back`'s): the keys' gradient through
    `w` and through the system, beta's through both, the decay's range terms through `w`, and its gradient
    from the pair terms (`algebra.summed_pairs`, in the launch), each a key block's, which `gradients` sums
    (each linear in its block's terms). A program takes one key block: a program walking a chunk's key blocks
    held each block's operand fragments across the walk, 64 to 200 bytes a thread past an A100's 255
    registers."""
    chunk = call.chunk
    width, family = call.width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(at, a_ref, k_ref, dw_ref, dl_ref, hi_ref, lo_ref, beta_ref, taken_ref, dk_ref, dbeta_ref,
                after_ref, decay_ref, *, streams, staging, landing):
        c, n, h = at["blocks"], at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        # The weighted system's tile and, where a chunk is wider than a key block, a block's apart: the
        # products' operands in registers (the rest are streamed).
        square, *rest = staging
        blocked, summing = (rest[0], rest[1:]) if chunk != call.block else (square, rest)
        column = c * call.block
        views = stream_views(lower=held.views(a_ref, row, chunk, h * chunk, chunk),
                             keys=given.views(k_ref, row, chunk, h // call.group * width + column,
                                              call.block),
                             w_grads=held.views(dw_ref, row, chunk, h * width + column, call.block),
                             system=held.views(dl_ref, row, chunk, h * chunk, chunk))

        def step(index, slots, carry, accumulators):
            del index

            def whole(name, operand, block):
                del block
                return stream_tile(slots, name, operand)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            beta = family.rows_of(beta_ref.at[h, span])
            stored = block_stored(call, dk_ref, row, h * width + column, landing["block"])

            d_beta, after, pairs = algebra.keys_gradient(
                stream_tile(slots, "lower", held), functools.partial(whole, "keys", given), 1,
                functools.partial(whole, "w_grads", held), stream_tile(slots, "system", held), beta, gate,
                                                         low,
                held.dtype, Issuer(call, {"inverse_t": blocked, "keys_dlt": square}),
                emit=stored)
            summed, columns = summed_issuer(tuple(summing), call)
            decay_ref.at[c, h, span][...] = algebra.summed_pairs(pairs, summed, columns)
            dbeta_ref.at[c, h, span][...] = d_beta
            after_ref.at[c, h, span][...] = after
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(lower=((chunk, chunk), held), keys=((chunk, call.block), given),
                          w_grads=((chunk, call.block), held), system=((chunk, chunk), held))
    blocks = width // call.block
    terms = jax.ShapeDtypeStruct((blocks, call.heads, call.rows), jnp.float32)
    out_type = (held.shape(call.heads * width), terms, terms, terms)
    # The key block's products through `w` and the system, and its keys' own pairs: the model's.
    model = 2 * chunk * chunk * (4 * call.block)
    # The system's gradient narrow and scaled by beta beside the key block's terms: through `w`, the keys in
    # fp32, the product being issued and the sum it joins (`algebra.keys_gradient`: each term dies as the next
    # forms); then its keys' own pairs, the pair terms, and the pairs' sums' masks and product.
    # In turn (`algebra.keys_gradient`): the inverse read transposed against the weights' gradient
    # ('inverse_t'), the system's gradient and its transpose against the keys ('keys_dl', 'keys_dlt').
    keyed = largest_live_set(
        product_fragments(call, (held, chunk, chunk), (held, chunk, call.block)),
        product_fragments(call, (held, chunk, chunk), (given, chunk, call.block)))
    held_tiles = Held(phases=(
        held_tile(held, chunk, chunk) * 2 + [f32(chunk, call.block)] * 4
        + keyed,
        held_tile(held, chunk, chunk) + [f32(chunk, chunk)] * 4 + masks(chunk, chunk)))
    if held.parts > 1 and product_results(call) == "registers":
        # An fp32 call's on warpgroup MMA (jsf-three-back: 211 registers against an H100's need of 248): its
        # row outputs kept beside the gradient's phase, and the pair sums' three-part operand, its exact bf16
        # triangle and their partial sums.
        rows = f32(chunk, 4)
        held_tiles = Held(phases=(
            held_tile(held, chunk, chunk) * 2 + [f32(chunk, call.block)] * 4 + [rows] * 4
            + keyed,
            held_tile(Operand(call.rows, 3, jnp.dtype(jnp.float32)), chunk, chunk)
            + held_tile(Operand(call.rows, 1, jnp.dtype(jnp.bfloat16)), chunk, chunk)
            + [f32(chunk, chunk)] * 4 + masks(chunk, chunk) + [rows] * 2))
    return launch_kernel(program, call, out_type=out_type, grid=(blocks, call.chunks, call.heads),
                         grid_names=("blocks", "chunks", "heads"), blocks=tiles, stages=1,
                         live=chunk_liveness(7),
                         steps=1, model=model, held=held_tiles,
                         staging=(staging_tiles(held, *dict.fromkeys(((chunk, chunk), (chunk, call.block))))
                                  + summing_staging(call)),
                         landing={"block": held.landing(chunk, call.block)})(a, k, dw, dl, hi, lo, beta,
                                                                             taken)
