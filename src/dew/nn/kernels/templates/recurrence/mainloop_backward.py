"""The recurrence mainloop's training pass on its machinery (`mainloop`): the fast backward's launches
(`gradients`, fla's DAG of the delta rule and, without it, mamba_ssm's rounding of Mamba2's SSD) and the
forms they take (`fast_backward_form`). The plan of a call's launches, forward and backward, is the
template's (`template.planned`, `template.backward_holds`)."""

import dataclasses
import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, product, resources
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.operands import Issuer, Parts
from dew.nn.kernels.templates.recurrence import key_backward
from dew.nn.kernels.templates.recurrence.mainloop import (
    CHUNK,
    Call,
    Held,
    Operand,
    Tiles,
    block_stored,
    chunk_liveness,
    column_tile,
    diagonal,
    f32,
    final_memory_landing,
    fitting_block,
    fragments,
    held_tile,
    launch_kernel,
    lowered_once,
    masks,
    memory_landing,
    prefix_reader,
    prepared,
    role_tiles,
    sequence_walk,
    shared_allocation,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
    summed_issuer,
    summing_staging,
    vector_blocks,
    vector_views,
    walk_stores,
    whole_blocks,
    zero_padded,
)
from dew.nn.precision import compensated_sum


def fast_backward_form(delta: bool, rounded: bool, dtype, *, per_key: bool = False) -> bool:
    """Whether the fast backward takes a call's form: the delta rule, fla's DAG at fla's rounding, a decay a
    head's or a key dimension's (fla_kda's, `key_backward`); or a bf16 `rounded` call without it and a decay a
    head (the SSD's under the run's fast forms, its route's), the same DAG without the delta rule's at
    mamba_ssm's rounding (`prepared`), held to recorded mamba_ssm's bf16 case; either in fp32 under the run's
    fast forms (`rounded`), its operands in two bf16 parts (`mainloop.FAST_PARTS`). An fp16 call without the
    delta rule, which no recording holds, a decay a key dimension without it, and an fp32 call's three parts
    (the accurate forms) take none. A decay a key dimension's form takes either pair factoring its layer's
    floor gives (`algebra.factored_range`, `algebra.key_diagonal_pairs_back`)."""
    if jnp.dtype(dtype) == jnp.float32 and not rounded:
        return False
    if per_key:
        return delta
    return delta or (rounded and jnp.dtype(dtype) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)))


def gradients(q, k, v, g, beta, h0, do, dfinal, *, tiles: Tiles, starts=None, firsts=None,
              max_segments: int | None = None, scale: float | None = None,
              family: mosaic.Family | None = None, interpret: bool = False,
              inverse=None, kept=None, delta: bool = True, floor: float | None = None, rounded: bool = False,
              ranged: bool = False, skip=None):
    """The fast backward of the gated delta rule, a decay a head (`forward`'s, `g` `[B, T, HV]`), in a 16-bit
    call, or of a `rounded` call without it (the SSD's under the run's fast forms, `fast_backward_form`):
    `(dq, dk, dv, dg, dbeta, dh0, dskip)` for the output's and final memory's gradients `do`, `dfinal`, fla's
    DAG (recomputing the forward's values, as fla does) through the algebra's backward tile functions:
    `_local_back`, `_reverse`, `_scores_back`, `_query_key`, `_system_back`, `_values_back`, `_keys_back`, the
    decay's gradient summed token by token (`algebra.summed_pairs`, `algebra.summed_ranges`) and the memory's
    decay's term added, a key head's value heads summed; the forward's `inverse` and `kept` values where given
    (`kept_forward`'s) rather than computed again, its states launch at the forward's `tiles` where they are
    not. Its rounding is fla's (mamba_ssm's without the delta rule); the accurate backward is the portable
    math's token adjoint. Without the delta rule there is no system: the corrected values are the values
    gated, `u = v beta`, whose gradient `du` (the reverse walk's) gives `dv = du beta` and `dbeta = sum(v
    du)`, and the memory walks back with no `w^T dv`; a `skip` `[HV]` (Mamba2's `D`, `mainloop.forward`'s)
    adds `D do` to `dv`, its own gradient `sum(v do)` (`dskip`, None without a skip)."""
    if not fast_backward_form(delta, rounded, jnp.result_type(q, k, v), per_key=g.ndim == 4):
        raise NotImplementedError("the fast backward takes the delta rule, or a rounded call without it "
                                  "under the run's fast forms, an fp32 call's under them alone")
    run = prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=delta,
                   tiles=tiles, family=family, interpret=interpret, inverse=inverse, kept=kept,
                   firsts=firsts, floor=floor, rounded=rounded, ranged=ranged)
    call = run.call
    rows, heads, width, columns = call.rows, call.heads, call.width, call.value_width
    laid_do = call.given.split(run.laid(zero_padded(do.astype(run.dtype), 3, columns)).reshape(rows, -1))
    dht = zero_padded(zero_padded(dfinal.astype(jnp.float32), 2, width), 3, columns).reshape(-1, columns)
    if call.per_key:
        dq, dk, dv, dg, dbeta, dh0 = key_backward.gradients(run, laid_do, dht, tiles)
        dg = run.unlaid(dg.reshape(rows, heads, width))[..., :run.key_width]
        return (*_unlaid(run, (q, k, v, g, beta), dq, dk, dv, dg, run.unlaid(dbeta), dh0, starts), None)
    local = _local_back(run.q, run.k, laid_do, run.hi, run.lo, run.taken, call, scale=run.scale)
    # Without the delta rule the walk gates the values where their columns are the call's own. Padded, it
    # read a padded copy of them (0.061 ms of Nemotron-H 4B's SSD training on an H100) and its gate took
    # 0.115, where XLA's reduction over the walk's fp32 gradient fused the padding in 0.119; unpadded, the
    # walk's gate took 0.038 where XLA's took 0.137 (30B; window-h100-c70aaa8-b8109ee).
    gates = not delta and run.value_width == columns
    dh, dv2, dh0, *gated = _reverse(run.q, run.k, run.w, laid_do, local, run.hi, run.lo, dht, run.sequences,
                                    run.v if gates else None, run.beta if gates else None, call, tiles,
                                    scale=run.scale, skip=skip if gates else None)
    ds, decay_pairs = _scores_back(run.q, run.k, run.corrected, laid_do, run.hi, run.lo, run.taken, run.beta,
                                   call,
                                   scale=run.scale)
    dq, dk, dw, after, leaving, memory = _query_key(run.q, run.k, run.corrected, laid_do, dv2, run.h, dh, ds,
                                                    run.hi, run.lo, run.taken, run.beta, call,
                                                    scale=run.scale)
    chunks = call.chunks
    dskip = None
    if delta:
        d_l = _system_back(run.a, run.k, run.v, dw, dv2, run.hi, run.lo, run.beta, run.taken, call)
        dv, dbeta_u = _values_back(run.a, run.v, dv2, run.beta, run.taken, call)
        dk_w, dbeta_k, after_w, decay_w = _keys_back(run.a, run.k, dw, d_l, run.hi, run.lo, run.beta,
                                                     run.taken, call)
        # The keys launch's terms a key block's, summed (each linear in them).
        dbeta_k, after_w, decay_w = (x.sum(0) for x in (dbeta_k, after_w, decay_w))
        dbeta = dbeta_u + dbeta_k
        after_w = after_w.reshape(heads, chunks, CHUNK)
    else:
        dv, dbeta, dskip = _gate_back(run, call, dv2, gated, laid_do, skip)
        dk_w, after_w, decay_w = None, 0.0, 0.0
    # Each token's decay's gradient from the launches' terms (`algebra.decay_gradient`, linear in them): the
    # query/key launch's ranges, `after` from each token to the chunk's end and `leaving` from its start to
    # before it, the keys launch's beside them, and the memory's decay across each chunk.
    memory = memory.reshape(-1, heads, chunks, call.block).sum((0, 3))
    ending = (run.hi + run.lo).reshape(heads, chunks, CHUNK)[..., -1]
    after, leaving = (x.sum(0).reshape(heads, chunks, CHUNK) for x in (after, leaving))
    dg = algebra.decay_gradient((decay_pairs + decay_w).reshape(heads, chunks, CHUNK), after + after_w,
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
    return (*grads, dskip)


def _unlaid(run, operands: tuple, dq, dk, dv, dg, dbeta, dh0, starts, *, extra_dk=0.0) -> tuple:
    """The backward's gradients laid back as the call's operands (`operands`, `(q, k, v, g, beta)`) and in
    their dtypes: the queries' and keys' `[R, H * Dk]` a key head's, `extra_dk` `[B, T, H, Dk]` added to the
    keys', the values' `[R, HV * Dv]`, the log decay's and write gates' as the call's (laid back already), and
    the initial memory's `[B * HV * Dk, Dv]`, zero for a row a document starts."""
    call = run.call
    q, k, v, g, beta = operands
    rows, heads, width, columns = call.rows, call.heads, call.width, call.value_width

    def key_headed(x):
        """`[R, H * Dk]`, a key head's, laid back as `[B, T, H, Dk]`."""
        laid = run.unlaid(x.astype(jnp.float32).reshape(rows, heads // call.group, width))
        return laid[..., :run.key_width]

    dv = run.unlaid(dv.reshape(rows, heads, columns))[..., :run.value_width]
    dh0 = dh0.reshape(run.batch, heads, width, columns)[:, :, :run.key_width, :run.value_width]
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
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    return Held(phases=(
        [f32(CHUNK, CHUNK)] * 2 + masks(CHUNK, CHUNK)
        + fragments(call, (given, CHUNK, call.block), (given, CHUNK, call.block)),
        [*held_tile(held, CHUNK, CHUNK), f32(CHUNK, vb), *held_tile(_value_grads(call), CHUNK, vb),
         *fragments(call, (held, CHUNK, CHUNK), (given, CHUNK, vb))]))


def _local_back_staged(call: Call, vb: int) -> tuple:
    """`_local_back`'s staging tiles and landings at value block `vb`: the scores' narrow form and a value
    block's gradient's landing."""
    held = dataclasses.replace(call.held, rows=call.rows)
    return staging_tiles(held, (CHUNK, CHUNK)), {"row": _value_grads(call).landing(CHUNK, vb)}


@lowered_once
def _local_back(q, k, do, hi, lo, taken, call: Call, *, scale: float):
    """`[R, HV * Dv]` in `_value_grads`' form: the corrected values' gradient from each chunk's own output
    (`algebra.local_gradient`, fla's chunk_bwd_dv_local), rounded as fla's, a program a chunk and value head
    (persistent): apart from the memory's reverse walk (`_reverse`), whose tiles beside these spilled 80
    bytes a thread on an H100's."""
    width, value_width, family = call.width, call.value_width, call.family
    given, grads = call.given, _value_grads(call)
    # A value row a block at a time, from the whole width down to whole value tiles (`Call.value_block`).
    vb = fitting_block(call, functools.partial(_local_back_held, call),
                       functools.partial(_local_back_staged, call), call.value_width, call.value_block)

    def program(at, q_ref, k_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref, dv_ref, *,
                streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = h // call.group * width
        views = stream_views(queries=given.views(q_ref, row, CHUNK, keyed, width),
                             keys=given.views(k_ref, row, CHUNK, keyed, width),
                             out=given.views(do_ref, row, CHUNK, h * value_width, value_width))
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
                grads.store(dv_ref, dv, row, CHUNK, h * value_width + j * vb, vb, family=family,
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

    tiles = stream_blocks(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                          out=((CHUNK, value_width), given))
    tiles |= vector_blocks(call, "gate", "low", "column_gate", "column_low")
    # The local scores, then their product with the output's gradient.
    work = Work(flops=2 * CHUNK * CHUNK * (width + value_width), span=1, exponentials=CHUNK * CHUNK)
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
        return dv2, dbeta.sum(0).T, compensated_sum(skipped[0], (0, 2)) if skipped else None
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
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    return Held(carried=[f32(call.width, bv)] * 2, phases=(
        [f32(CHUNK, bv)] * (2 + gates + skips) + [f32(call.block, bv)] * call.per_key
        + held_tile(given, CHUNK, bv) * (1 + gates)
        + held_tile(held, call.block, bv) + held_tile(call.memories, call.block, bv)
        + held_tile(_reverse_values(call, gates=gates), CHUNK, bv)
        + fragments(call, (given, CHUNK, call.block), (held, CHUNK, call.block)),))


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


def _reverse_launched(call: Call, batch: int, bv: int, walked_width: int, *, gates: bool,
                      skips: bool = False) -> tuple:
    """`_reverse`'s outputs' types, its walked streams and its final memories' streams, a program walking
    `walked_width` value columns; where the walk `gates` the values, their stream and beta's sums, where it
    `skips`, the skip's sums, and under a decay a key dimension the memory decay's."""
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
    blocks = stream_blocks(queries=((CHUNK, width), keyed), keys=((CHUNK, width), keyed),
                           **({"w": ((CHUNK, width), dataclasses.replace(call.held, rows=call.rows))}
                              if call.delta else {}),
                           out=((CHUNK, walked_width), given),
                           local=((CHUNK, walked_width), _value_grads(call)),
                           **({"values": ((CHUNK, walked_width), given)} if gates else {}))
    finals = stream_blocks(**{f"final{i}": ((call.block, walked_width), final)
                              for i in range(width // call.block)})
    return out_type, blocks, finals


@lowered_once
def _reverse(q, k, w, do, local, hi, lo, dht, sequences, v, beta, call: Call, tiles: Tiles, *, scale: float,
             skip=None, memory=None):
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
    tile's (kda-lower5-sm90 at 6e03eb57e: where the keys' columns met them, no layout held)."""
    gates, skips = v is not None, skip is not None
    width, value_width, family = call.width, call.value_width, call.family
    grads = _reverse_values(call, gates=gates)
    bv, slices, walkers = tiles.block_v, width // call.block, tiles.walkers
    walked_width = bv * walkers
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    batch = dht.shape[0] // (call.heads * width)
    final = Operand(dht.shape[0], 1, jnp.dtype(jnp.float32))
    chunks = call.chunks
    assert gates or not skips, "the walk adds a skip where it gates the values"
    assert call.per_key == (memory is not None), "the walk takes the memories under a decay a key dimension"

    def program(at, q_ref, k_ref, w_ref, do_ref, local_ref, hi_ref, lo_ref, dht_ref, seq_ref, v_ref, beta_ref,
                *refs, streams, staging, landing):
        skip_ref, refs = (refs[0], refs[1:]) if skips else (None, refs)
        entered_ref, (dh_ref, dv_ref, dh0_ref, *gated_ref) = (refs[0], refs[1:]) if call.per_key else (None,
                                                                                                         refs)
        j, s, h = at["blocks"], at["sequences"], at["heads"]
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
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
            (staged, (second,)), lands = roled(role)
            # The decayed output gradient fp32 as two parts against the queries (fla's split product, 16 of
            # its significand's bits): one part put the initial memory's gradient 5 to 7 times the port's
            # distance from float64. Its first part reuses the memory products' tile, each consumed before
            # the next stages; an fp32 call's two parts are the memory products' own (a third took the
            # product's parts apart, each its own accumulator: 277 values live at the walk's peak against a
            # 16-bit call's 133, 100 bytes spilled on an H100, back-need at 5cec06bd8).
            issue = Issuer(call, {**dict.fromkeys(("values_dv", "weights_dv"), staged),
                                   "reads": staged if len(staged) > 1 else (*staged, second)})
            return functools.partial(_reverse_step, lands, issue, walked + role * bv, pl.ds(role * bv, bv))

        def _reverse_step(lands, issue, column, columns, f, slots, grad, accumulators):
            n = chunks - 1 - f
            span = pl.ds(n * CHUNK, CHUNK)
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
            grads.store(dv_ref, dv, n * CHUNK, CHUNK, h * value_width + column, bv, family=family,
                        smem=lands["corrected"], wait=False)
            return grad, accumulators

        sequence_walk(call, final, "final", (leaving, final_views), (walked_streams, views), step,
                      chunks - first - count, count, ends, begins,
                      lambda role, i: dh0_ref.at[pl.ds(memory(row, i), call.block),
                                                 pl.ds(walked + role * bv, bv)], bv,
                      landing=lambda role: roled(role)[1].get("final"), walkers=walkers)

    out_type, blocks, finals = _reverse_launched(call, batch, bv, walked_width, gates=gates, skips=skips)
    # The memory's carried gradient: three memory products a chunk, two without the delta rule's weights.
    steps = chunks / sequences.shape[0]
    work = Work(flops=2 * (2 + call.delta) * CHUNK * width * walked_width * steps, span=steps)
    final_landing, after = final_memory_landing(call, bv)
    return launch_kernel(program, call, out_type=out_type,
                         grid=(value_width // walked_width, sequences.shape[0], call.heads),
                         # Two deep where the block holds both stages, else one: an fp32 call's two parts and
                         # keys 256 wide took 237920 and 282848 bytes two deep, past an H100's 232448
                         # (fp32-plan-sm90 at cdcd620ef, wide-backward).
                         grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=(1, 2), work=work,
                         held=_reverse_held(call, bv, gates=gates, skips=skips), consumers=walkers,
                         live=lambda at, refs: refs[8][at["sequences"], 1] > 0,
                         staging=(*staging_tiles(held, (call.block, bv)),
                                  (plgpu.SMEM((CHUNK, bv), held.stored),)),
                         once=finals,
                         landing={**memory_landing(call, bv), "corrected": grads.landing(CHUNK, bv),
                                  **final_landing},
                         # The keys take the place of weights without the delta rule, and the keys and the
                         # prefix that of values a walk gates none of: operands no step reads.
                         after=after)(q, k, k if w is None else w, do, local, hi, lo, dht, sequences,
                                      k if v is None else v, hi if beta is None else beta,
                                      *(() if skip is None else (skip.astype(jnp.float32),)),
                                      *(() if memory is None else (memory,)))


def _reverse_carried(call: Call, slots, grad: list, out: Parts, columns, issue: Issuer, prefix: tuple, h,
                     n, *, scale: float) -> tuple:
    """`_reverse`'s step's algebra on chunk `n` of head `h`: the memory's gradient carried through the chunk
    and the corrected values' (`algebra.reverse_carried`; under a decay a key dimension
    `algebra.key_reverse_carried`, each key slice's rows decayed across the chunk by `exp2(g_last)` a
    column's), from the walk's tiles (`slots`, the value block's `columns`) and the prefix's halves'
    refs."""
    family, held, width = call.family, dataclasses.replace(call.held, rows=call.rows), call.width
    span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1

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
    ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in prefix)
    return algebra.reverse_carried(grad, keys, weights, queries, out.value(), local, gate, low, ending,
                                   ending_low, scale, held.dtype, issue)


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
    columns, apart = first * value_width + split * vq, value_width // vq
    return {**stream_views(lambda i: (0, 0), queries=given.views(q_ref, row, CHUNK, key_columns, call.block),
                           keys=given.views(k_ref, row, CHUNK, key_columns, call.block)),
            **stream_views(lambda i: (0, i * apart),
                           **{name: operand.views(ref, row, CHUNK, columns, (group - 1) * value_width + vq)
                              for name, (operand, ref) in operands.items()}),
            **stream_views(lambda i: (i * (width // call.block), 0),
                           **{name: memories.views(ref, memory_row, group * width, split * vq, vq)
                              for name, ref in zip(("entered", "leaving"), memory, strict=True)})}


def _query_key_split(call: Call) -> int:
    """`_query_key`'s value columns a program takes of each value head: the widest whole number of blocks
    (`Call.block`) dividing the value width whose streams (each step's values, output gradient and
    corrected values' gradient `[C, vq]`, its memories `[block, vq]`, the queries', keys' and scores'
    gradient's blocks) fit the block's shared memory (`generation.shared_budget`), a block where none does:
    an fp32 call's two parts at Qwen3.5-27B's widths took 237824 bytes and a 256-wide value row's 215040 on
    an A100, past its 166912 (fp32-plan-sm90 at cdcd620ef, wide-backward). Halved instead, 192 columns took
    96, a block and a half, and the split's programs read no column past their first block's. Each split's
    bytes are the launch's own (`shared_allocation`, as `launch_kernel` lays it: its streams one deep, its
    landing)."""
    def streamed(vq: int) -> int:
        tiles, _ = _query_key_launched(call, vq)
        _, scope, _, _ = shared_allocation(call, tiles, (), _query_key_landing(call, call.value_width // vq))
        return resources.shared_bytes(call.family, scope(1))

    budget = generation.shared_budget(call.target)
    return next((vq for vq in whole_blocks(call.value_width, call.value_block) if streamed(vq) <= budget),
                call.block)


def _gradient_stored(call: Call, kind: str, block: int, grad, *, weights: Sequence, sums: dict, first,
                     landing: dict, apart: bool = False) -> None:
    """`algebra.memory_gradients`' emit in `_query_key`: the weights' gradient (`kind` 'dw') in the
    operands' dtype to `weights` (its parts' views) as it is final, as fla's dqkwg writes it; the queries'
    and keys' ('dq', 'dk') to `sums[kind]` summed over the key head's value heads in fp32, stored at the
    first (`first`) and added at the rest (`Family.added`: a TMA reduction, or whole rows through
    registers), each through its launch's landing tiles. A value split's (`apart`) weights' gradient is its
    split's own part of it, fp32, which XLA adds."""
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
    diagonal of their product `S^T dS` where the family's products read registers, else their elementwise
    product's rows (`_query_key`)."""
    if call.family.operands_in_registers:
        traced = issue("memory_trace", entered.T, leaving)
        return jnp.sum(jnp.where(diagonal(traced.shape[0], call.family), traced, 0.0), axis=1)
    return jnp.sum(entered.value().astype(jnp.float32) * leaving.value().astype(jnp.float32), axis=1)


def _query_key_held(call: Call) -> Held:
    """What `_query_key`'s consumer thread holds (`algebra.memory_gradients`): a block's gradient's fp32 sum
    (one accumulator over the value blocks), that sum gated, its scores' product, and a block's queries or
    keys in fp32 for the range terms; then `sum(S dS)` (`_memory_sum`): the memories' product and its
    diagonal's mask where products read registers, with a block of each operand, else the memories in fp32.
    An accumulator a value block, the SSD's spilled 168 bytes a thread on an A100, where one needed 249
    registers (gateqk at c2808ebbe)."""
    held, memories, given = call.held, call.memories, call.given
    block = (memories, call.block, call.block)
    summed = ([f32(call.block, call.block), *masks(call.block, call.block), *fragments(call, block, block)]
              if call.family.operands_in_registers else [f32(call.block, call.block)] * 2)
    return Held(phases=(
        [f32(CHUNK, call.block)] * 4
        + fragments(call, (given, CHUNK, call.block), (held, CHUNK, call.block), (memories, call.block,
                                                                                      call.block)),
        summed))


def _query_key_landing(call: Call, splits: int) -> dict:
    """`_query_key`'s stores' tiles at `splits` value splits: the weights' gradient with the delta rule (fp32
    where there are two splits or more) and the queries' and keys' sums."""
    held = dataclasses.replace(call.held, rows=call.rows)
    weights = held.landing(CHUNK, call.block) if splits == 1 else ((CHUNK, call.block), jnp.float32)
    return {**({"weights": weights} if call.delta else {}), "sums": ((CHUNK, call.block), jnp.float32)}


def _query_key_launched(call: Call, vq: int) -> tuple[dict, tuple]:
    """`_query_key`'s streams at value split `vq` (`_query_key_split`) and its outputs' types, each split's
    sums its own (the weights' gradient, the delta rule's alone, in fp32 where there are two splits or
    more)."""
    width, held, memories, given = call.width, call.held, call.memories, call.given
    splits = call.value_width // vq
    tiles = stream_blocks(queries=((CHUNK, call.block), given), keys=((CHUNK, call.block), given),
                          values=((CHUNK, vq), held), out=((CHUNK, vq), given),
                          **({"value_grads": ((CHUNK, vq), held)} if call.delta else {}),
                          entered=((call.block, vq), memories),
                          leaving=((call.block, vq), memories), ds=((CHUNK, CHUNK), _score_grads(call)))
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
    width, value_width, family = call.width, call.value_width, call.family
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    grads = _score_grads(call)
    vq = _query_key_split(call)
    splits, blocks, group = value_width // vq, vq // call.block, call.group
    keyed_width, key_blocks = call.heads // group * width, width // call.block

    def program(at, q_ref, k_ref, v_ref, do_ref, dv_ref, h_ref, dh_ref, ds_ref, hi_ref, lo_ref, taken_ref,
                beta_ref, *outs, streams, staging, landing):
        del staging
        dq_ref, dk_ref, *weights_ref, after_ref, leaving_ref, memory_ref = outs
        c, n, keyed = at["blocks"], at["chunks"], at["heads"]
        split = at["values"]
        row, span, last = n * CHUNK, pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
        key_columns = keyed * width + c * call.block
        first = keyed * group
        views = _query_key_views(call, (q_ref, k_ref), (v_ref, do_ref, dv_ref), (h_ref, dh_ref), row=row,
                                 key_columns=key_columns, first=first, chunk=n, block=c, split=split, vq=vq)
        # Step `i` the group's `i`-th value head's scores' gradient; without the delta rule each step the key
        # head's, summed over its value heads (`_scores_back`).
        scored = (first, group) if call.delta else (keyed, 1)
        views |= stream_views(lambda i: (0, i if call.delta else 0),
                              ds=grads.views(ds_ref, row, CHUNK, scored[0] * CHUNK, scored[1] * CHUNK))

        def step(index, slots, carry, accumulators, *, scores: bool = True):
            h = first + index

            column = functools.partial(column_tile, call, slots)

            def whole(name, operand, i):
                del i
                return stream_tile(slots, name, operand)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
            first_head = index == 0

            # Each split's sums its own, a split's columns after another's.
            columns = pl.ds(split * keyed_width + key_columns, call.block)
            sums = {"dq": dq_ref.at[span, columns], "dk": dk_ref.at[span, columns]}
            weights_at = h * width + c * call.block
            weights = (() if not weights_ref else
                       held.views(weights_ref[0], row, CHUNK, weights_at, call.block) if splits == 1 else
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
            # being emitted after them: Mosaic GPU waits on no copy at a kernel's exit (template numerics
            # review 2026-10-26, P3-4).
            family.store_wait(0)
            return carry, accumulators

        # Without the delta rule the scores' products the first step's alone.
        return mosaic.Walk(views, step, steps=group, segments=() if call.delta else (
            (step, 0, 1), (functools.partial(step, scores=False), 1, group)))

    tiles, out_type = _query_key_launched(call, vq)
    # Each value head of the group: three memory products (two without the delta rule's weights) and the
    # scores' gradient's two, a key head's two without the delta rule.
    scores = 2 * CHUNK * CHUNK * call.block
    each = (2 + call.delta) * CHUNK * value_width * call.block + scores * call.delta
    work = Work(flops=2 * (each * group + scores * (not call.delta)), span=group)
    dq, dk, *weights, after, leaving, memory = launch_kernel(
        program, call, out_type=out_type,
        grid=(splits, width // call.block, call.chunks, call.heads // group),
        grid_names=("values", "blocks", "chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(10),
        work=work, held=_query_key_held(call), landing=_query_key_landing(call, splits))(
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


@lowered_once
def _scores_back(q, k, corrected, do, hi, lo, taken, beta, call: Call, *, scale: float):
    """`(ds in `_score_grads`' form, decay [HV, R] fp32)`: each chunk and value head's scores' gradient and
    the decay's gradient from its pair terms (`algebra.scores_gradient`, `algebra.summed_pairs`), fla's
    dqkwg's part the memory's products do not take, a program a chunk and value head with the delta rule;
    without it a program a chunk and key head, a step each of its value heads, their scores' gradients summed
    in fp32 in shared memory and the key head's stored at the last (`_score_grads`). Persistent: apart from
    `_query_key`, each launch's tiles a thread's registers hold (one launch spilled 1.5 KB a thread on an
    H100's, SE bwd-probe at a49c8da50)."""
    width, value_width, family = call.width, call.value_width, call.family
    grads = _score_grads(call)
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    blocks = value_width // call.block
    group = 1 if call.delta else call.group

    def program(at, q_ref, k_ref, v_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref,
                beta_ref, ds_ref, decay_ref, *, streams, staging, landing):
        n, head = at["chunks"], at["heads"]
        first = head * group
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = first // call.group * width
        views = stream_views(lambda i: (0, 0), queries=given.views(q_ref, row, CHUNK, keyed, width),
                             keys=given.views(k_ref, row, CHUNK, keyed, width))
        # Step `i` the `i`-th value head's.
        columns = first * value_width, group * value_width
        views |= stream_views(lambda i: (0, i), values=held.views(v_ref, row, CHUNK, *columns),
                              out=given.views(do_ref, row, CHUNK, *columns))
        vectors = {"gate": hi_ref, "low": lo_ref, "column_gate": column_hi_ref, "column_low": column_lo_ref}
        # A program a value head streams its vectors where the family copies them; a key head's steps read
        # theirs.
        streamed = family.copies_vectors and group == 1
        if streamed:
            views |= vector_views(n, first, **vectors)
        summing, accumulated = (staging, None) if group == 1 else (staging[:-1], staging[-1])

        def step(index, slots, carry, accumulators, loaded=None):
            h = first + index
            found = prefix_reader(family, streamed, vectors, h, span)(slots) if loaded is None else loaded
            (gate, low), laid = found

            column = functools.partial(column_tile, call, slots)

            summed, column_sums = summed_issuer(tuple(summing), call)
            ds, pairs = algebra.scores_gradient(
                functools.partial(column, "queries", given), functools.partial(column, "keys", given),
                width // call.block, functools.partial(column, "values", held),
                functools.partial(column, "out", given), blocks, gate, low, scale,
                grads.dtype if accumulated is None else jnp.float32, Issuer(call).summed, laid=laid,
                column_scale=family.columns_of(beta_ref.at[h, span]) if call.gated_keys else None)
            # The pairs' sum before the scores' gradient's store, so an fp32 call's parts form after the
            # pairs are gone.
            decay_ref.at[h, span][...] = algebra.summed_pairs(pairs, summed, column_sums)
            if accumulated is None:
                grads.store(ds_ref, ds, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            else:
                # The key head's sum so far, in shared memory from step to step, zero at its first.
                total = accumulated

                @pl.when(index == 0)
                def _():
                    total[...] = jnp.zeros(total.shape, jnp.float32)

                ds = family.tile_of(total) + ds
                total[...] = ds

                @pl.when(index == group - 1)
                def _():
                    grads.store(ds_ref, ds.astype(grads.dtype), row, CHUNK, head * CHUNK, CHUNK,
                                family=family, smem=landing["square"])

            return carry, accumulators

        read = prefix_reader(family, streamed, vectors, first, span)
        prologue = None if streamed or group > 1 else lambda: read(None)
        return mosaic.Walk(views, step, steps=group, prologue=prologue)

    tiles = stream_blocks(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                          values=((CHUNK, value_width), held), out=((CHUNK, value_width), given))
    if group == 1:
        tiles |= vector_blocks(call, "gate", "low", "column_gate", "column_low")
    out_type = (grads.shape((call.heads // group) * CHUNK),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    # The scores' gradient, the scores and the pairs' sums, each value head's.
    work = Work(flops=2 * CHUNK * CHUNK * (value_width + width + CHUNK) * group, span=group,
                exponentials=CHUNK * CHUNK * group)
    # The scores' gradient fp32 and narrow, the pairs' decays, the scores and the pair terms, the masks, and
    # the pairs' sum's product: three fp32 tiles at once, the decays taken before the scores' product and
    # each block sum into one accumulator (`algebra.scores_gradient`; an accumulator a block, four, and an
    # H100's need was 232 in a bf16 call, 192 in one, back-need-one-32 at 9482e4d26). A key head's sum read
    # back beside them; and where the scores' gradient leaves in parts (an fp32 call's), the remainder each
    # part after the first is formed from (`product.formed`): without it the model stated 232 registers
    # where ptxas spilled 32 bytes past 255 on an H100 (back-need2 at 170dd3a22).
    held_tiles = Held(phases=([f32(CHUNK, CHUNK)] * (3 + (group > 1) + (grads.parts > 1))
                              + held_tile(grads, CHUNK, CHUNK)
                              + masks(CHUNK, CHUNK)
                              + fragments(call, (given, CHUNK, call.block), (given, CHUNK, call.block)),))
    summing = summing_staging(call) + ((plgpu.SMEM((CHUNK, CHUNK), jnp.float32),) if group > 1 else ())
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads // group),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(8),
                         work=work,
                         held=held_tiles, staging=summing, landing={"square": grads.landing(CHUNK, CHUNK)},
                         persistent=True)(q, k, corrected, do, hi, lo, hi, lo, taken, beta)


@lowered_once
def _system_back(a, k, v, dw, du, hi, lo, beta, taken, call: Call):
    """`dL [R, HV * C]` as `call.held`: the WY representation's backward's system part a chunk and value head
    at a time (fla's prepare_wy_repr_bwd's, `algebra.system_gradient`) from `w`'s and `u`'s gradients `dw`,
    `du` (`call.held`) and the inverse `a` less its unit diagonal (`_solved`'s), which `_keys_back` takes.
    Apart from `_keys_back` and `_values_back`, each launch's tiles a thread's registers hold (in one they
    spilled 360 bytes a thread on an H100's)."""
    width, value_width, family = call.width, call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(at, a_ref, k_ref, v_ref, dw_ref, du_ref, hi_ref, lo_ref, beta_ref, column_hi_ref,
                column_lo_ref, taken_ref, dl_ref, *, streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        views = stream_views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                             keys=given.views(k_ref, row, CHUNK, h // call.group * width, width),
                             values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                             w_grads=held.views(dw_ref, row, CHUNK, h * width, width),
                             u_grads=held.views(du_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(column_tile, call, slots)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            beta = family.rows_of(beta_ref.at[h, span])
            issue = Issuer(call, dict.fromkeys(("inverse_t", "inverse_dx", "inverse_l"), staged))
            d_l = algebra.system_gradient(
                stream_tile(slots, "lower", held), functools.partial(column, "keys", given),
                                          width // call.block,
                functools.partial(column, "values", given), value_width // call.block,
                functools.partial(column, "w_grads", held), functools.partial(column, "u_grads", held), beta,
                gate, low, held.dtype, issue, issue.summed,
                laid=(family.columns_of(column_hi_ref.at[h, span]),
                      family.columns_of(column_lo_ref.at[h, span])))
            held.store(dl_ref, d_l, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(lower=((CHUNK, CHUNK), held), keys=((CHUNK, width), given),
                          values=((CHUNK, value_width), given), w_grads=((CHUNK, width), held),
                          u_grads=((CHUNK, value_width), held))
    # The weighted keys' and values' products into the inverse's gradient, and its products through it.
    work = Work(flops=2 * CHUNK * CHUNK * (width + value_width + 2 * CHUNK), span=1)
    # A block's keys or values scaled, narrow, beside the inverse's gradient; then it masked, its products
    # through the inverse and the pairs' decays with their masks (`algebra.system_gradient`).
    held_tiles = Held(phases=(
        [f32(CHUNK, CHUNK), f32(CHUNK, call.block), *held_tile(held, CHUNK, call.block),
         *fragments(call, (held, CHUNK, call.block), (held, CHUNK, call.block))],
        [f32(CHUNK, CHUNK)] * 4 + held_tile(held, CHUNK, CHUNK) + masks(CHUNK, CHUNK)
        + fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, CHUNK))))
    return launch_kernel(program, call, out_type=held.shape(call.heads * CHUNK),
                         grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(10),
                         work=work,
                         held=held_tiles, staging=staging_tiles(held, (CHUNK, CHUNK)),
                         landing={"square": held.landing(CHUNK, CHUNK)})(a, k, v, dw, du, hi, lo, beta, hi,
                                                                         lo,
                                                                         taken)


@lowered_once
def _values_back(a, v, du, beta, taken, call: Call):
    """`dv [R, HV * Dv]` as `call.held` and `dbeta [HV, R]` fp32: the WY representation's backward's values'
    part a chunk and value head at a time (`algebra.values_gradient`): the values' gradient and beta's
    through `u`."""
    value_width, family = call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(at, a_ref, v_ref, du_ref, beta_ref, taken_ref, dv_ref, dbeta_ref, *, streams, staging,
                landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        views = stream_views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                             values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                             u_grads=held.views(du_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(column_tile, call, slots)
            stored = block_stored(call, dv_ref, row, h * value_width, landing["block"])

            dbeta_ref.at[h, span][...] = algebra.values_gradient(
                stream_tile(slots, "lower", held), functools.partial(column, "values", given),
                value_width // call.block, functools.partial(column, "u_grads", held),
                family.rows_of(beta_ref.at[h, span]), held.dtype, Issuer(call, {"inverse_t": staged}),
                emit=stored)
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(lower=((CHUNK, CHUNK), held), values=((CHUNK, value_width), given),
                          u_grads=((CHUNK, value_width), held))
    out_type = (held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    work = Work(flops=2 * CHUNK * CHUNK * value_width, span=1)
    # A value block's gradient in fp32 and narrow, its product through the inverse, and the values in fp32.
    held_tiles = Held(phases=(
        [f32(CHUNK, call.block)] * 3 + held_tile(held, CHUNK, call.block)
        + fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, call.block)),))
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(4),
                         work=work,
                         held=held_tiles, staging=staging_tiles(held, (CHUNK, call.block)),
                         landing={"block": held.landing(CHUNK, call.block)})(a, v, du, beta, taken)


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
    width, family = call.width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(at, a_ref, k_ref, dw_ref, dl_ref, hi_ref, lo_ref, beta_ref, taken_ref, dk_ref, dbeta_ref,
                after_ref, decay_ref, *, streams, staging, landing):
        c, n, h = at["blocks"], at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        staged, *summing = staging
        column = c * call.block
        views = stream_views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                             keys=given.views(k_ref, row, CHUNK, h // call.group * width + column,
                                              call.block),
                             w_grads=held.views(dw_ref, row, CHUNK, h * width + column, call.block),
                             system=held.views(dl_ref, row, CHUNK, h * CHUNK, CHUNK))

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
                held.dtype, Issuer(call, dict.fromkeys(("inverse_t", "keys_dl", "keys_dlt"), staged)),
                emit=stored)
            summed, columns = summed_issuer(tuple(summing), call)
            decay_ref.at[c, h, span][...] = algebra.summed_pairs(pairs, summed, columns)
            dbeta_ref.at[c, h, span][...] = d_beta
            after_ref.at[c, h, span][...] = after
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(lower=((CHUNK, CHUNK), held), keys=((CHUNK, call.block), given),
                          w_grads=((CHUNK, call.block), held), system=((CHUNK, CHUNK), held))
    blocks = width // call.block
    terms = jax.ShapeDtypeStruct((blocks, call.heads, call.rows), jnp.float32)
    out_type = (held.shape(call.heads * width), terms, terms, terms)
    # The key block's products through `w` and the system, and its keys' own pairs.
    work = Work(flops=2 * CHUNK * CHUNK * (4 * call.block), span=1)
    # The system's gradient narrow and scaled by beta beside the key block's terms: through `w`, the keys in
    # fp32, the product being issued and the sum it joins (`algebra.keys_gradient`: each term dies as the next
    # forms); then its keys' own pairs, the pair terms, and the pairs' sums' masks and product.
    held_tiles = Held(phases=(
        held_tile(held, CHUNK, CHUNK) * 2 + [f32(CHUNK, call.block)] * 4
        + fragments(call, (held, CHUNK, CHUNK), (given, CHUNK, call.block), (held, CHUNK, call.block)),
        held_tile(held, CHUNK, CHUNK) + [f32(CHUNK, CHUNK)] * 4 + masks(CHUNK, CHUNK)))
    return launch_kernel(program, call, out_type=out_type, grid=(blocks, call.chunks, call.heads),
                         grid_names=("blocks", "chunks", "heads"), blocks=tiles, stages=1,
                         live=chunk_liveness(7),
                         work=work, held=held_tiles,
                         staging=staging_tiles(held, (CHUNK, CHUNK)) + summing_staging(call),
                         landing={"block": held.landing(CHUNK, call.block)})(a, k, dw, dl, hi, lo, beta,
                                                                             taken)
