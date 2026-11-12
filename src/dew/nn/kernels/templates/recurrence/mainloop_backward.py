"""The recurrence mainloop's training pass on its machinery (`mainloop`): the fast backward's launches
(`gradients`, fla's DAG of the delta rule and, without it, mamba_ssm's rounding of Mamba2's SSD) and the
forms they take (`fast_backward_form`). The plan of a call's launches, forward and backward, is the
template's (`template.planned`, `template.backward_holds`)."""

import dataclasses
import functools
import os
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, recurrence_algebra as algebra, tables
from dew.nn.kernels.chunked_launch import zero_padded
from dew.nn.kernels.templates import mosaic, product, resources
from dew.nn.kernels.templates.cost import Work
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
    stated_producer,
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


def fast_backward_form(delta: bool, rounded: bool, dtype, *, per_key: bool = False) -> bool:
    """Whether the fast backward takes a call's form: the delta rule, fla's DAG at fla's rounding, a decay a
    head's or a key dimension's (fla_kda's, `key_backward`); or a bf16 `rounded` call without it and a decay a
    head (the SSD's under the run's fast forms, its route's), the same DAG without the delta rule's at
    mamba_ssm's rounding (`prepared`), held to recorded mamba_ssm's bf16 case; either in fp32 under the run's
    fast forms (`rounded`), its operands in two bf16 parts (`mainloop.FAST_PARTS`). An fp16 call without the
    delta rule, which no recording holds, a decay a key dimension without it, and an fp32 call's three parts
    (the accurate forms, or a form without the delta rule on its route, `recurrence_chunk.chunk_rounding`)
    take none. A decay a key dimension's form takes each sub-chunk's own pairs each their own decay
    (`algebra.key_diagonal_pairs_back`)."""
    if jnp.dtype(dtype) == jnp.float32 and not rounded:
        return False
    if per_key:
        return delta
    return delta or (rounded and jnp.dtype(dtype) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)))


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
    """`_scores_back`'s scores' gradient."""
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
                  rounded: bool = False, ranged: bool = False,
                  skip=None,
                  forked: ForkedTable | None = None) -> tuple[tuple, "Kept | key_backward.KeyKept"]:
    """The fast backward of the gated delta rule, a decay a head (`forward`'s, `g` `[B, T, HV]`), in a 16-bit
    call, or of a `rounded` call without it (the SSD's under the run's fast forms, `fast_backward_form`):
    `(dq, dk, dv, dg, dbeta, dh0, dskip, dentries)` for the output's and final memory's gradients `do`,
    `dfinal`, fla's
    DAG (recomputing the forward's values, as fla does) through the algebra's backward tile functions:
    `_local_back`, `_reverse`, `_scores_back`, `_query_key`, `_system_back`, `_values_back`, `_keys_back`, the
    decay's gradient summed token by token (`algebra.summed_pairs`, `algebra.summed_ranges`) and the memory's
    decay's term added, a key head's value heads summed; the forward's `inverse` and `kept` values where given
    (`kept_forward`'s) rather than computed again, its states launch at the forward's `tiles` where they are
    not. Its rounding is fla's (mamba_ssm's without the delta rule); the accurate backward is the portable
    math's token adjoint. Without the delta rule there is no system: the corrected values are the values
    gated, `u = v beta`, whose gradient `du` (the reverse walk's) gives `dv = du beta` and `dbeta = sum(v
    du)`, and the memory walks back with no `w^T dv`; a `skip` `[HV]` (Mamba2's `D`,
    `chunk_forward.forward`'s) adds `D do` to `dv`, its own gradient `sum(v do)` (`dskip`, None without a
    skip). Returns them with each launch's outputs as stored (`Kept`; a decay a key dimension's,
    `key_backward.KeyKept`), the forward's `kept_forward` beside it. A `forked` call's
    (`chunk_forward.ForkedTable`) `dh0` is its given memories' and `dentries` its entry write's
    (`_tabled_gradients`, `_entered_gradient`), None for another."""
    if not fast_backward_form(delta, rounded, jnp.result_type(q, k, v), per_key=g.ndim == 4):
        raise NotImplementedError("the fast backward takes the delta rule, or a rounded call without it "
                                  "under the run's fast forms, an fp32 call's under them alone")
    run = prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=delta,
                   tiles=tiles, family=family, interpret=interpret, inverse=inverse, kept=kept,
                   firsts=firsts, rounded=rounded, ranged=ranged, forked=forked)
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
                                   call,
                                   scale=run.scale)
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
    `[B, NC, HV, Dk, Dv]`, zero at a chunk that takes nothing up (which the walk wrote nothing to)."""
    call, table = run.call, run.table
    taking = (table.loads >= 0).reshape(run.batch, -1)
    taken = taken.reshape(*taking.shape, call.heads, call.width, call.value_width)
    return jnp.where(taking[..., None, None, None], taken, 0.0)


def _tabled_gradients(run, grads: tuple, taken) -> tuple:
    """`grads` (`_unlaid`'s) with a forked call's entering memories' gradient its given memories', `[B, S',
    HV, Dk, Dv]`: each the sum of what the chunks that took it up gave it (`taken`); `grads` as they are
    for another call."""
    if taken is None:
        return grads
    table = run.table
    loads = table.loads.reshape(run.batch, -1)
    slots = jnp.arange(1, table.count + 1)
    given = jnp.einsum("bcs,bchkv->bshkv", (loads[..., None] == slots).astype(jnp.float32),
                       _taken_up(run, taken))
    return (*grads[:5], given[..., :run.key_width, :run.value_width])


def _entered_gradient(run, taken):
    """A forked call's entry write's gradient `[B, NC, HV, Dk, Dv]` (`chunk_forward.ForkedTable.entries`): the
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
    (`algebra.local_gradient`, fla's chunk_bwd_dv_local), rounded as fla's, a program a chunk and value head
    (persistent): apart from the memory's reverse walk (`_reverse`), whose tiles beside these spilled 80
    bytes a thread on an H100's."""
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
                scale, given.dtype, issue, issue.summed, emit=stored, laid=laid)
            return carry, accumulators

        return mosaic.Walk(views, step, prologue=None if streamed else lambda: read(None))

    tiles = stream_blocks(queries=((chunk, width), given), keys=((chunk, width), given),
                          out=((chunk, value_width), given))
    tiles |= vector_blocks(call, "gate", "low", "column_gate", "column_low")
    # The local scores, then their product with the output's gradient.
    work = Work(flops=2 * chunk * chunk * (width + value_width), span=1, exponentials=chunk * chunk)
    staging, landing = _local_back_staged(call, vb)
    return launch_kernel(program, call, out_type=grads.shape(call.heads * value_width),
                         grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=tiles,
                         stages=1,
                         live=chunk_liveness(7), work=work, held=_local_back_held(call, vb), staging=staging,
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
        + _reverse_products(call, bv),))


def _reverse_products(call: Call, bv: int) -> list:
    """The fragments of `_reverse`'s step's products, issued in turn (`algebra.reverse_carried`): the keys
    against the memory's gradient staged ('values_dv'); the queries read transposed against the output's
    gradient read (`reads`: decayed in fp32 and staged in two parts under a decay a head, a value
    tile's own parts where it holds more, the gradient as given under a decay a key dimension); and with the
    delta rule the weights read transposed against the corrected values' gradient ('weights_dv')."""
    chunk, held = call.chunk, dataclasses.replace(call.held, rows=call.rows)
    keyed = held if call.per_key else call.given
    read = (call.given if call.per_key
            else Operand(call.rows, 2 if held.parts == 1 else held.parts, jnp.dtype(jnp.float32)))
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


def _reverse_keyed(call: Call, h) -> tuple[Operand, object]:
    """The form and columns of the queries and keys `_reverse` takes for value head `h`: the key head's as
    given under a decay a head; under one a key dimension each value head's own, gated (`key_gated`'s
    queries, `key_decayed_write`'s keys) and held."""
    if call.per_key:
        return dataclasses.replace(call.held, rows=call.rows), h * call.width
    return call.given, h // call.group * call.width


def _reverse_launched(program, call: Call, tiles: Tiles, batch: int, sequences: int, *, gates: bool,
                      skips: bool, table: Table | None, producer: int | None):
    """`_reverse`'s `program` launched (`launch_kernel`): its outputs, its walked streams and its final
    memories' streams, a program walking `tiles.walkers` value blocks; where the walk `gates` the values,
    their stream and beta's sums, where it `skips`, the skip's sums, under a decay a key dimension the
    memory decay's, and a forked call's `table`'s (`table_returned`); its staging and landing tiles, its
    `Work` over the laid sequences' mean of chunks and its register model (`_reverse_held`, `table_held`)."""
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
                         work=Work(flops=2 * (2 + call.delta) * chunk * width * walked_width * steps,
                                   span=steps),
                         held=table_held(_reverse_held(call, bv, gates=gates, skips=skips), call, table, bv),
                         consumers=tiles.walkers, producer=producer,
                         live=sequence_live(8),
                         staging=(*staging_tiles(held, *dict.fromkeys(((call.block, bv), (chunk, bv)))),
                                  (plgpu.SMEM((chunk, bv), held.stored),)),
                         once=finals,
                         landing={**memory_landing(call, bv), "corrected": values.landing(chunk, bv),
                                  **final_landing, **table_landing(call, table is not None, bv)},
                         after=after)


def _reverse_producer(call: Call) -> int | None:
    """`_reverse`'s producer floor, the one value its plan and its walk's schedule read (`stated_producer`):
    80, GDN's bf16 reverse walk's producer spilling 12 to 40 bytes at 24 to 64, none from 72 (row 5357,
    jsf-pfloor-ab3 on the promotion's candidate; at 64 on its base before, jsf-pfloor-ab2), and a forked
    call's 12 bytes at 72, none at 80 (on a B200, its PTX's decrement alone changed, armada
    20261110174422-11220d9e)."""
    return stated_producer(call, 80)


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
    producer = _reverse_producer(call)
    assert call.per_key == (memory is not None), "the walk takes the memories under a decay a key dimension"
    returns = 0 if table is None else 1 + (table.held > 0)

    def program(at, q_ref, k_ref, w_ref, do_ref, local_ref, hi_ref, lo_ref, dht_ref, seq_ref, v_ref, beta_ref,
                *refs, streams, staging, landing):
        skip_ref, refs = (refs[0], refs[1:]) if skips else (None, refs)
        entered_ref, refs = (refs[0], refs[1:]) if call.per_key else (None, refs)
        table_refs, refs = refs[:2 * (table is not None)], refs[2 * (table is not None):]
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
                    rows = pl.ds(memory(n, i), call.block)
                    gated_ref[0].at[column // bv, rows][...] = algebra.key_memory_decay(
                        family.tile_of(entered_ref.at[rows, pl.ds(column, bv)]), part)

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
                                 table=table, producer=producer)(
        q, k, k if w is None else w, do, local, hi, lo, dht, sequences, k if v is None else v,
        hi if beta is None else beta, *(() if skip is None else (skip.astype(jnp.float32),)),
        *(() if memory is None else (memory,)), *(() if table is None else (table.loads, table.stores)))
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
                     chunk, block, split=0, vq: int | None = None) -> dict:
    """`_query_key`'s streams' views (`stream_views`) for a program's key block `block` of chunk `chunk`, its
    key head's first value head `first`: the queries' and keys' block (`keyed`, their refs), and each step `i`
    the group's `i`-th value head's values, output gradient and corrected values' gradient (`served`) at
    its columns, and its entered memory and leaving memory's gradient (`memory`) a head's rows apart; of
    each value head's columns the `split`-th `vq` (all of them where None, `_query_key_split`)."""
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
    return {**stream_views(lambda i: (0, 0), queries=given.views(q_ref, row, rows, key_columns, call.block),
                           keys=given.views(k_ref, row, rows, key_columns, call.block)),
            **stream_views(lambda i: (0, i * apart),
                           **{name: operand.views(ref, row, rows, columns, (group - 1) * value_width + vq)
                              for name, (operand, ref) in operands.items()}),
            **stream_views(lambda i: (i * (width // call.block), 0),
                           **{name: memories.views(ref, memory_row, group * width, split * vq, vq)
                              for name, ref in zip(("entered", "leaving"), memory, strict=True)})}


def _gradient_stored(call: Call, kind: str, block: int, grad, *, weights: Sequence, sums: dict, first,
                     landing: dict, apart: bool = False) -> None:
    """`algebra.memory_gradients`' emit in `_query_key`: the weights' gradient (`kind` 'dw') in the
    operands' dtype to `weights` (its parts' views) as it is final, as fla's dqkwg writes it; the queries'
    and keys' ('dq', 'dk') to `sums[kind]` summed over the key head's value heads in fp32, stored at the
    first (`first`) and added at the rest (`Family.added`: a TMA reduction, or whole rows through
    registers), each through its launch's landing tiles. A value split's (`apart`) weights' gradient is its
    split's own part of it, fp32, which XLA adds, through its own landing tile."""
    del block
    family = call.family
    if kind == "dw":
        if apart:
            family.stored([grad], weights, smem=landing["weights"])
            return
        pieces = product.formed(grad.astype(jnp.float32), call.held.parts) if call.held.parts > 1 else (grad,)
        family.stored([piece.astype(call.held.stored) for piece in pieces], weights, smem=landing["weights"])
        return

    @pl.when(first)
    def _():
        family.stored([grad], [sums[kind]], smem=landing["sums"])

    @pl.when(jnp.logical_not(first))
    def _():
        family.added(grad, sums[kind], call.interpret, smem=landing["sums"])


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
    def fits(vq: int) -> bool:
        tiles, _ = _query_key_launched(call, vq)
        landing = _query_key_landing(call, call.value_width // vq)
        _, scope, _, _ = shared_allocation(call, tiles, (), landing)
        registers = _query_key_held(call, vq).registers(call.family, addressed((), landing))
        return (resources.shared_bytes(call.family, scope(1)) <= generation.shared_budget(call.target)
                and registers <= call.family.registers(1))

    return next((vq for vq in whole_blocks(call.value_width, call.value_block) if fits(vq)),
                call.value_block)


def _query_key_held(call: Call, vq: int) -> Held:
    """What `_query_key`'s consumer thread holds (`algebra.memory_gradients`): a block's gradient's fp32 sum
    (one accumulator over the value blocks), that sum gated, its scores' product, and a block's queries or
    keys in fp32 for the range terms; then `sum(S dS)` (`_memory_sum`): the memories' product and its
    diagonal's mask where products read registers, with a block of each operand, else the memories in fp32.
    An accumulator a value block, the SSD's spilled 168 bytes a thread on an A100, where one needed 249
    registers (gateqk at c2808ebbe)."""
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
        [f32(chunk, call.block)] * 4 + rows + read(chunk, given, memories.parts) + read(chunk, held, 0),
        summed))


def _query_key_landing(call: Call, splits: int) -> dict:
    """`_query_key`'s stores' tiles at `splits` value splits: the weights' gradient with the delta rule
    (fp32 where there are two splits or more) and the queries' and keys' sums."""
    chunk = call.chunk
    held = dataclasses.replace(call.held, rows=call.rows)
    weights = held.landing(chunk, call.block) if splits == 1 else ((chunk, call.block), jnp.float32)
    return {**({"weights": weights} if call.delta else {}), "sums": ((chunk, call.block), jnp.float32)}


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
    `_score_grads`' form, `_scores_back`'s: `[R, HV * C]` with the delta rule, each step's value head's
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
    splits, blocks, group = value_width // vq, vq // call.value_block, call.group
    keyed_width, key_blocks = call.heads // group * width, width // call.block

    def program(at, q_ref, k_ref, v_ref, do_ref, dv_ref, h_ref, dh_ref, ds_ref, hi_ref, lo_ref, taken_ref,
                beta_ref, *outs, streams, staging, landing):
        del staging
        dq_ref, dk_ref, *weights_ref, after_ref, leaving_ref, memory_ref = outs
        c, n, keyed = at["blocks"], at["chunks"], at["heads"]
        split = at["values"]
        row, span, last = n * chunk, pl.ds(n * chunk, chunk), n * chunk + chunk - 1
        key_columns = keyed * width + c * call.block
        first = keyed * group
        views = _query_key_views(call, (q_ref, k_ref), (v_ref, do_ref, dv_ref), (h_ref, dh_ref), row=row,
                                 key_columns=key_columns, first=first, chunk=n, block=c, split=split, vq=vq)
        # Step `i` the group's `i`-th value head's scores' gradient; without the delta rule each step the key
        # head's, summed over its value heads (`_scores_back`).
        scored = (first, group) if call.delta else (keyed, 1)
        views |= stream_views(lambda i: (0, i if call.delta else 0),
                              ds=grads.views(ds_ref, row, chunk, scored[0] * chunk, scored[1] * chunk))

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
                                       landing=landing, apart=splits > 1)

            issue = Issuer(call)
            ds = stream_tile(slots, "ds", grads) if scores else None
            after, leaving = algebra.memory_gradients(
                functools.partial(whole, "queries", given), functools.partial(whole, "keys", given), 1,
                functools.partial(column, "values", held), functools.partial(column, "out", given),
                functools.partial(column, "value_grads", held) if call.delta else None,
                lambda i, j: column("entered", memories, j), lambda i, j: column("leaving", memories, j),
                blocks, ds if splits == 1 else None, gate, low, ending, ending_low, None, scale, issue.summed,
                emit=stored, row_scale=family.rows_of(beta_ref.at[h, span]) if call.gated_keys else None)
            if splits > 1 and ds is not None:
                # The scores' gradient's products the first split's alone, added after its memory products.
                @pl.when(split == 0)
                def _():
                    keys, queries = whole("keys", given, 0), whole("queries", given, 0)
                    for kind, lhs, rhs in (("dq", ds, keys), ("dk", ds.T, queries)):
                        family.store_wait(0)
                        family.added(issue("scores_dq", lhs, rhs), sums[kind], call.interpret,
                                     smem=landing["sums"])
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
            # The step's reduce-adds ('dq', 'dk', `Family.added`) waited on by design, not by 'dw''s store
            # after them: Mosaic GPU waits on no copy at a kernel's exit (numerics review 10-26, P3-4).
            family.store_wait(0)
            return carry, accumulators

        # Without the delta rule the scores' products the first step's alone.
        return mosaic.Walk(views, step, steps=group, segments=() if call.delta else (
            (step, 0, 1), (functools.partial(step, scores=False), 1, group)))

    tiles, out_type = _query_key_launched(call, vq)
    # Each value head of the group: three memory products (two without the delta rule's weights) and the
    # scores' gradient's two, a key head's two without the delta rule.
    scores = 2 * chunk * chunk * call.block
    each = (2 + call.delta) * chunk * value_width * call.block + scores * call.delta
    work = Work(flops=2 * (each * group + scores * (not call.delta)), span=group)
    dq, dk, *weights, after, leaving, memory = launch_kernel(
        program, call, out_type=out_type,
        grid=(splits, width // call.block, call.chunks, call.heads // group),
        grid_names=("values", "blocks", "chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(10),
        work=work, held=_query_key_held(call, vq), landing=_query_key_landing(call, splits))(
        q, k, corrected, do, dv, h, dh, ds, hi, lo, taken, beta)
    # The splits' own sums added (one split's, as they are, where a program takes every value column).
    if splits > 1:
        dq, dk, *weights = (x.reshape(call.rows, splits, -1).sum(1) for x in (dq, dk, *weights))
    after, leaving, memory = (x.reshape(splits, -1, *x.shape[1:]).sum(0) for x in (after, leaving, memory))
    dw = None if not weights else weights[0] if splits == 1 else held.split(weights[0])
    return dq, dk, dw, after, leaving, memory


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


def _scores_back_held(call: Call, grads: Operand, group: int) -> Held:
    """What `_scores_back`'s consumer thread holds: the scores' gradient fp32 and narrow, the pairs'
    decays, the scores and the pair terms, the masks, and the pairs' sum's product: three fp32 tiles at
    once, the decays taken before the scores' product and each block sum into one accumulator
    (`algebra.scores_gradient`; an accumulator a block, four, and an H100's need was 232 in a bf16 call, 192
    in one, back-need-one-32 at 9482e4d26). Where the scores' gradient leaves in parts (an fp32 call's), the
    remainder each part after the first is formed from (`product.formed`): without it the model stated 232
    registers where ptxas spilled 32 bytes past 255 on an H100 (back-need2 at 170dd3a22). Where a key head's
    value heads are summed (`group`), a phase after their pair reduction."""
    chunk, given, held = call.chunk, call.given, dataclasses.replace(call.held, rows=call.rows)
    phases = ([f32(chunk, chunk)] * (3 + (grads.parts > 1))
              + held_tile(grads, chunk, chunk)
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
    if group > 1:
        # After the pair reduction, no scores, masks or operand fragments remain: the incoming ds,
        # the running pair and TwoSum's temporaries peak at six fp32 tiles.
        phases += ([f32(chunk, chunk)] * 6,)
    return Held(phases=phases)


@lowered_once
def _scores_back(q, k, corrected, do, hi, lo, taken, beta, call: Call, *, scale: float):
    """`(ds in `_score_grads`' form, decay [HV, R] fp32)`: each chunk and value head's scores' gradient and
    the decay's gradient from its pair terms (`algebra.scores_gradient`, `algebra.summed_pairs`), fla's
    dqkwg's part the memory's products do not take, a program a chunk and value head with the delta rule;
    without it a program a chunk and key head, a step each of its value heads, their scores' gradients summed
    in fp32 in shared memory and the key head's stored at the last (`_score_grads`). Persistent: apart from
    `_query_key`, each launch's tiles a thread's registers hold (one launch spilled 1.5 KB a thread on an
    H100's, SE bwd-probe at a49c8da50)."""
    chunk = call.chunk
    width, value_width, family = call.width, call.value_width, call.family
    grads = _score_grads(call)
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    blocks = value_width // call.value_block
    group = 1 if call.delta else call.group

    def program(at, q_ref, k_ref, v_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref,
                beta_ref, ds_ref, decay_ref, *, streams, staging, landing):
        n, head = at["chunks"], at["heads"]
        first = head * group
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = first // call.group * width
        views = stream_views(lambda i: (0, 0), queries=given.views(q_ref, row, chunk, keyed, width),
                             keys=given.views(k_ref, row, chunk, keyed, width))
        # Step `i` the `i`-th value head's.
        columns = first * value_width, group * value_width
        views |= stream_views(lambda i: (0, i), values=held.views(v_ref, row, chunk, *columns),
                              out=given.views(do_ref, row, chunk, *columns))
        vectors = {"gate": hi_ref, "low": lo_ref, "column_gate": column_hi_ref, "column_low": column_lo_ref}
        # A program a value head streams its vectors where the family copies them; a key head's steps read
        # theirs.
        streamed = family.copies_vectors and group == 1
        if streamed:
            views |= vector_views(n, first, **vectors)
        summing, accumulated = (staging, None) if group == 1 else (staging[:-2], staging[-2:])

        def step(index, slots, carry, accumulators, loaded=None):
            h = first + index
            found = prefix_reader(family, streamed, vectors, h, span)(slots) if loaded is None else loaded
            (gate, low), laid = found

            column = functools.partial(column_tile, call, slots)
            value_column = functools.partial(column, width=call.value_block)

            summed, column_sums = summed_issuer(tuple(summing), call)
            ds, pairs = algebra.scores_gradient(
                functools.partial(column, "queries", given), functools.partial(column, "keys", given),
                width // call.block, functools.partial(value_column, "values", held),
                functools.partial(value_column, "out", given), blocks, gate, low, scale,
                grads.dtype if accumulated is None else jnp.float32, Issuer(call).summed, laid=laid,
                column_scale=family.columns_of(beta_ref.at[h, span]) if call.gated_keys else None)
            # The pairs' sum before the scores' gradient's store, so an fp32 call's parts form after the
            # pairs are gone.
            decay_ref.at[h, span][...] = algebra.summed_pairs(pairs, summed, column_sums)
            if accumulated is None:
                grads.store(ds_ref, ds, row, chunk, h * chunk, chunk, family=family, smem=landing["square"])
            else:
                # Keep the group's head-sum roundings, as mamba_ssd's dCB does (row 2066).
                # Near-zero decays leave a token's own ds dominant; its 64 heads can cancel to
                # 4e-4 from terms of order 100 (seed 117), so a plain sum loses a quarter of dq/dk.
                total, correction = accumulated

                @pl.when(index == 0)
                def _():
                    total[...] = jnp.zeros(total.shape, jnp.float32)
                    correction[...] = jnp.zeros(correction.shape, jnp.float32)

                ds, tail = algebra.compensated_add(
                    (family.tile_of(total), family.tile_of(correction)), (ds, jnp.zeros_like(ds)))
                total[...] = ds
                correction[...] = tail

                @pl.when(index == group - 1)
                def _():
                    grads.store(ds_ref, ds.astype(grads.dtype), row, chunk, head * chunk, chunk,
                                family=family, smem=landing["square"])

            return carry, accumulators

        read = prefix_reader(family, streamed, vectors, first, span)
        prologue = None if streamed or group > 1 else lambda: read(None)
        if os.environ.get("DBG_NO_PROLOGUE"):
            prologue = None
        return mosaic.Walk(views, step, steps=group, prologue=prologue)

    tiles = stream_blocks(queries=((chunk, width), given), keys=((chunk, width), given),
                          values=((chunk, value_width), held), out=((chunk, value_width), given))
    if group == 1:
        tiles |= vector_blocks(call, "gate", "low", "column_gate", "column_low")
    out_type = (grads.shape((call.heads // group) * chunk),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    # The scores' gradient, the scores and the pairs' sums, each value head's.
    work = Work(flops=2 * chunk * chunk * (value_width + width + chunk) * group, span=group,
                exponentials=chunk * chunk * group)
    held_tiles = _scores_back_held(call, grads, group)
    summing = summing_staging(call) + ((plgpu.SMEM((chunk, chunk), jnp.float32),) * 2 if group > 1 else ())
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads // group),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(8),
                         work=work,
                         held=held_tiles, staging=summing, landing={"square": grads.landing(chunk, chunk)},
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
    # The weighted keys' and values' products into the inverse's gradient, and its products through it.
    work = Work(flops=2 * chunk * chunk * (width + value_width + 2 * chunk), span=1)
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
                         work=work,
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
    work = Work(flops=2 * chunk * chunk * value_width, span=1)
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
                         work=work,
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
    # The key block's products through `w` and the system, and its keys' own pairs.
    work = Work(flops=2 * chunk * chunk * (4 * call.block), span=1)
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
                         work=work, held=held_tiles,
                         staging=(staging_tiles(held, *dict.fromkeys(((chunk, chunk), (chunk, call.block))))
                                  + summing_staging(call)),
                         landing={"block": held.landing(chunk, call.block)})(a, k, dw, dl, hi, lo, beta,
                                                                             taken)
