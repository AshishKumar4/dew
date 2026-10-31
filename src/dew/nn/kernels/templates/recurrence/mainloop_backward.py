"""The recurrence mainloop's training pass on its machinery (`mainloop`): the fast backward's launches
(`gradients`, fla's DAG of the delta rule and, without it, mamba_ssm's rounding of Mamba2's SSD), the forms
and devices they take (`fast_backward_form`, `backward_holds`), and the plan of a call's launches, forward
and backward (`planned`, `holds`), from which the template derives its tiles."""

import contextlib
import dataclasses
import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, product
from dew.nn.kernels.templates.recurrence.issuing import Issuer, Parts
from dew.nn.kernels.templates.recurrence.mainloop import (
    _PLANNED,
    CHUNK,
    Launch,
    Tiles,
    Work,
    _axis,
    _block_stored,
    _Call,
    _column,
    _diagonal,
    _f32,
    _final_landing,
    _fragments,
    _Held,
    _held,
    _kernel,
    _lowered_once,
    _masks,
    _memory_landing,
    _Operand,
    _padded,
    _prefix_reader,
    _prepared,
    _roled,
    _sequence_walk,
    _staging,
    _streamed,
    _summed,
    _summing,
    _taken,
    _tile,
    _vector_blocks,
    _vector_views,
    _views,
    _walk,
    _walk_stores,
    kept_forward,
)


def planned(q, k, v, g, beta, h0=None, *, tiles: Tiles, family: mosaic.Family, starts=None,
            max_segments: int | None = None, delta: bool = True, backward: bool = False,
            floor: float | None = None, target: generation.GpuTarget | None = None,
            rounded: bool = False) -> tuple[Launch, ...]:
    """The launches a call over operands of these shapes (`jax.ShapeDtypeStruct`s, `forward`'s) runs at
    `tiles` by `family`'s bundle on `target` (the device's, or the assumed one's, where None), in order, as
    planned (`Launch`): its arrays abstract (`jax.eval_shape`), its kernels declared and not built
    (`_kernel`); with `backward`, the fast backward's too (`gradients`)."""
    found: list[Launch] = []

    def run(q, k, v, g, beta, h0, starts):
        token = _PLANNED.set(found)
        try:
            out, final, (inverse, kept) = kept_forward(q, k, v, g, beta, h0, tiles=tiles, starts=starts,
                                                       max_segments=max_segments, delta=delta, family=family,
                                                       floor=floor, rounded=rounded)
            if backward:
                gradients(q, k, v, g, beta, h0, out, final, tiles=tiles, starts=starts,
                          max_segments=max_segments, family=family, inverse=inverse, kept=kept, delta=delta,
                          rounded=rounded)
        finally:
            _PLANNED.reset(token)

    with contextlib.nullcontext() if target is None else generation.assumed(target):
        jax.eval_shape(run, q, k, v, g, beta, h0, starts)
    return tuple(found)


def holds(launches: tuple[Launch, ...], family: mosaic.Family, shared: int) -> bool:
    """Whether each of `launches` fits a block of `shared` bytes and its consumer threads hold their registers
    (`Launch.registers` within `Family.registers` at its consumer roles, of the bundle it issues by,
    `Launch.family`, `family` where it names none): a generated launch spills nothing."""
    return all(launch.shared <= shared
               and launch.registers <= (launch.family or family).registers(launch.consumers)
               for launch in launches)


def fast_backward_form(delta: bool, rounded: bool, dtype) -> bool:
    """Whether the fast backward takes a call's form (a decay a head in a 16-bit call besides): the delta
    rule, fla's DAG at fla's rounding, or a bf16 `rounded` call without it under the run's fast forms, the
    same DAG without the delta rule's at mamba_ssm's rounding (`_prepared`), held to recorded mamba_ssm's
    bf16 case; an fp16 call's, which no recording holds, the port's."""
    fast = rounded and jnp.dtype(dtype) == jnp.bfloat16 and generation.numerics().forms == "fast"
    return delta or fast


def backward_holds(q, k, v, g, beta, h0=None, *, tiles: Tiles, family: mosaic.Family, shared: int,
                   starts=None, max_segments: int | None = None, delta: bool = True,
                   rounded: bool = False) -> bool:
    """Whether the fast backward's launches (`gradients`'s, past the forward's) hold on `family`'s bundle at
    `tiles` within a block of `shared` bytes (`holds`); where they do not, the rule's gradient is the
    portable math's."""
    shapes = [None if x is None else jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, k, v, g, beta, h0,
                                                                                    starts)]
    *operands, starts = shapes
    forward = planned(*operands, tiles=tiles, family=family, starts=starts, max_segments=max_segments,
                      delta=delta, rounded=rounded)
    both = planned(*operands, tiles=tiles, family=family, starts=starts, max_segments=max_segments,
                   backward=True, delta=delta, rounded=rounded)
    return holds(both[len(forward):], family, shared)


def gradients(q, k, v, g, beta, h0, do, dfinal, *, tiles: Tiles, starts=None, firsts=None,
              max_segments: int | None = None, scale: float | None = None,
              family: mosaic.Family | None = None, interpret: bool = False,
              inverse=None, kept=None, delta: bool = True, rounded: bool = False):
    """The fast backward of the gated delta rule, a decay a head (`forward`'s, `g` `[B, T, HV]`), in a 16-bit
    call, or of a `rounded` call without it under the run's fast forms (`fast_backward_form`): `(dq, dk,
    dv, dg, dbeta, dh0)` for the output's and final memory's gradients `do`, `dfinal`, fla's
    DAG (recomputing the forward's values, as fla does) through the algebra's backward tile functions:
    `_local_back`, `_reverse`, `_scores_back`, `_query_key`, `_system_back`, `_values_back`, `_keys_back`,
    the decay's gradient summed token by token
    (`algebra.summed_back`) and the memory's decay's term added, a key head's value heads summed; the
    forward's `inverse` and `kept` values where given (`kept_forward`'s) rather than computed again, its
    states launch at the forward's `tiles` where they are not. Its rounding is fla's (mamba_ssm's without the
    delta rule); the accurate backward is the portable math's token adjoint. Without the delta rule there is
    no system: the corrected values are the values gated, `u = v beta`, whose gradient `du` (the reverse
    walk's) gives `dv = du beta` and `dbeta = sum(v du)`, and the memory walks back with no `w^T dv`."""
    if not fast_backward_form(delta, rounded, jnp.result_type(q, k, v)):
        raise NotImplementedError("the fast backward takes the delta rule, or a rounded call without it "
                                  "under the run's fast forms")
    run = _prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=delta,
                    tiles=tiles, family=family, interpret=interpret, inverse=inverse, kept=kept,
                    firsts=firsts, rounded=rounded)
    call = run.call
    if call.per_key or call.held.parts > 1:
        raise NotImplementedError("the fast backward takes a decay a head in a 16-bit call")
    rows, heads, width, columns = call.rows, call.heads, call.width, call.value_width
    batch = run.batch
    laid_do = call.given.split(run.laid(_padded(do.astype(run.dtype), 3, columns)).reshape(rows, -1))
    dht = _padded(_padded(dfinal.astype(jnp.float32), 2, width), 3, columns).reshape(-1, columns)
    local = _local_back(run.q, run.k, laid_do, run.hi, run.lo, run.taken, call, scale=run.scale)
    dh, dv2, dh0 = _reverse(run.q, run.k, run.w, laid_do, local, run.hi, run.lo, dht, run.sequences, call,
                            tiles, scale=run.scale)
    ds, decay_pairs = _scores_back(run.q, run.k, run.corrected, laid_do, run.hi, run.lo, run.taken, run.beta,
                                   call,
                                   scale=run.scale)
    dq, dk, dw, after, leaving, memory = _query_key(run.q, run.k, run.corrected, laid_do, dv2, run.h, dh, ds,
                                                    run.hi, run.lo, run.taken, run.beta, call,
                                                    scale=run.scale)
    chunks = call.chunks
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
        # The values gated, `u = v beta`: their gradient through the gate, elementwise, in fp32.
        # The write gate's sum over the value columns, the rows' minor axis, in the rows' own layout: as the
        # heads' `[HV, R]` it fused with the decay gradient's transpose into a 0.137 ms reduction on an H100
        # (Nemotron-H 30B's SSD, ssd-hlo-h100-fdab2424c).
        du = dv2.astype(jnp.float32).reshape(rows, heads, columns)
        dv = (du * run.beta.T[:, :, None]).reshape(rows, heads * columns)
        dbeta = jnp.sum(du * run.v.astype(jnp.float32).reshape(rows, heads, columns), -1)
        dk_w, after_w, decay_w = None, 0.0, 0.0
    # Each token's decay's gradient, the launches' (`algebra.summed_back` is linear in its terms) and the
    # memory's decay across each chunk: its gradient `sum(S dS)` through the decay `exp(g_last)` from the
    # chunk's start, every token's.
    memory = memory.reshape(-1, heads, chunks, call.block).sum((0, 3))
    ending = (run.hi + run.lo).reshape(heads, chunks, CHUNK)[..., -1]
    # The query/key launch's range terms each token's through them (`algebra.summed_back`'s ranges' part):
    # `after` from each token to the chunk's end, `leaving` from its start to before it.
    after, leaving = (x.sum(0).reshape(heads, chunks, CHUNK) for x in (after, leaving))
    after = after + after_w
    # `leaving`'s sum before each token, never its own term added and taken away: a token's own `leaving` is
    # order one where the terms before it are `exp(-30)`'s under steep decays, which the sum rounded away
    # (an H100's decay gradient under decays near zero 140 times the port's distance from float64, whose
    # `leaving` enters its strictly lower pairs, fla `chunk_bwd_dqkwg`; SE-numerics-h100-fd61e11eeb).
    before = jnp.concatenate((jnp.zeros_like(leaving[..., :1]), jnp.cumsum(leaving[..., :-1], -1)), -1)
    ranges = jnp.flip(jnp.cumsum(jnp.flip(after, -1), -1), -1) + before
    dg = ((decay_pairs + decay_w).reshape(heads, chunks, CHUNK) + ranges
          + (memory * jnp.exp2(ending))[..., None])
    dg, dbeta = dg.reshape(heads, rows).T, dbeta.T if delta else dbeta

    def headed(x, size):
        """`[R, HV * size]` laid back as `[B, T, HV, size]`."""
        return run.unlaid(x.reshape(rows, heads, size))

    def keyed(x):
        """A value head's gradient `[B, T, HV, Dk]` summed onto its key head's."""
        return x.reshape(*x.shape[:2], heads // call.group, call.group, width).sum(3)[..., :run.key_width]

    # The launches' gradients in the operands' dtype (fla's), summed in fp32.
    dq, dk, dv = (x.astype(jnp.float32) for x in (dq, dk, dv))
    key_heads = heads // call.group
    def key_headed(x):
        """`[R, H * Dk]`, a key head's, laid back as `[B, T, H, Dk]`."""
        return run.unlaid(x.reshape(rows, key_heads, width))[..., :run.key_width]

    dq = key_headed(dq)
    dk = key_headed(dk) + (0.0 if dk_w is None else keyed(headed(dk_w.astype(jnp.float32), width)))
    dv = headed(dv, columns)[..., :run.value_width]
    dh0 = dh0.reshape(batch, heads, width, columns)[:, :, :run.key_width, :run.value_width]
    if starts is not None:
        dh0 = jnp.where(jnp.asarray(starts, bool)[:, 0, None, None, None], 0.0, dh0)
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), run.unlaid(dg).astype(g.dtype),
            run.unlaid(dbeta).astype(beta.dtype), dh0)


# The fast backward of a decay a head under the delta rule (fla's DAG through the algebra's backward tile
# functions, `algebra.local_gradient` ... `algebra.summed_back`), a 16-bit call's: the memory's reverse
# recurrence a sequence at a time (`_reverse`), each chunk's scores' gradient (`_scores_back`), the queries',
# keys' and weights' gradients a key block of a chunk at a time (`_query_key`), and the WY representation's
# (`_system_back`, `_values_back`, `_keys_back`); each launch writes its decay's terms a chunk and value
# head, its pairs' summed token by token (`summed_pairs`) and its ranges' by XLA's cumulative sums, which
# XLA adds.


@_lowered_once
def _local_back(q, k, do, hi, lo, taken, call: _Call, *, scale: float):
    """`[R, HV * Dv]` in `_value_grads`' form: the corrected values' gradient from each chunk's own output
    (`algebra.local_gradient`, fla's chunk_bwd_dv_local), rounded as fla's, a program a chunk and value head
    (persistent): apart from the memory's reverse walk (`_reverse`), whose tiles beside these spilled 80
    bytes a thread on an H100's."""
    width, value_width, family = call.width, call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    grads = _value_grads(call)

    def program(q_ref, k_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref, dv_ref, *,
                streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = h // call.group * width
        views = _views(queries=given.views(q_ref, row, CHUNK, keyed, width),
                       keys=given.views(k_ref, row, CHUNK, keyed, width),
                       out=given.views(do_ref, row, CHUNK, h * value_width, value_width))
        vectors = {"gate": hi_ref, "low": lo_ref, "column_gate": column_hi_ref, "column_low": column_lo_ref}
        streamed = family.copies_vectors
        if streamed:
            views |= _vector_views(n, h, **vectors)

        read = _prefix_reader(family, streamed, vectors, h, span)

        def step(index, slots, carry, accumulators, loaded=None):
            del index
            found = read(slots) if streamed else loaded
            assert found is not None, "a family that streams no vectors loads them in the prologue"
            (gate, low), laid = found
            (local,) = staging

            def block(name, c):
                return _tile(slots, name, given, columns=pl.ds(c * call.block, call.block))

            # The local scores once, against the output's gradient's whole row.
            dv = algebra.local_gradient(
                functools.partial(block, "keys"), functools.partial(block, "queries"), width // call.block,
                gate, low, _tile(slots, "out", given), scale, Issuer(call, {"local_dv": local}), laid=laid)
            grads.store(dv_ref, dv, row, CHUNK, h * value_width, value_width, family=family,
                       smem=landing["row"])
            return carry, accumulators

        _walk(streams, views, step, call, prologue=None if streamed else lambda: read(None))

    tiles = _streamed(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                      out=((CHUNK, value_width), given))
    tiles |= _vector_blocks(call, "gate", "low", "column_gate", "column_low")
    # The local scores, then their product with the output's gradient.
    work = Work(2 * CHUNK * CHUNK * (width + value_width), exponentials=CHUNK * CHUNK)
    # The local scores, their decays and masks; then their narrow form's product with the output's gradient
    # and its cast (`algebra.local_gradient`: the scores are narrowed before the product): as one phase, an
    # H100's model stated 248 registers where ptxas needed 136, an A100's 280 against 160.
    held_tiles = _Held(phases=(
        [_f32(CHUNK, CHUNK)] * 2 + _masks(CHUNK, CHUNK)
        + _fragments(call, (given, CHUNK, call.block), (given, CHUNK, call.block)),
        [*_held(held, CHUNK, CHUNK), _f32(CHUNK, value_width), *_held(grads, CHUNK, value_width),
         *_fragments(call, (held, CHUNK, CHUNK), (given, CHUNK, value_width))]))
    return _kernel(program, call, out_type=grads.shape(call.heads * value_width),
                   grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=tiles, stages=1,
                   live=_taken(7), work=work, held=held_tiles, staging=_staging(held, (CHUNK, CHUNK)),
                   landing={"row": grads.landing(CHUNK, value_width)}, persistent=True)(
        q, k, do, hi, lo, hi, lo, taken)


@_lowered_once
def _reverse(q, k, w, do, local, hi, lo, dht, sequences, call: _Call, tiles: Tiles, *, scale: float):
    """`(dh [chunks * HV * Dk, Dv], dv [R, HV * Dv], dh0 [B * HV * Dk, Dv])`: the gradient of the memory each
    chunk leaves as `call.memories`, the corrected values' gradient in `_value_grads`' form (their own
    output's, `local`, `_local_back`'s, and the memory's, `algebra.reverse_carried`) and each row's initial
    memory's gradient, fp32, each sequence's chunks from its last over a value block, the states launch's
    (`tiles.block_v`; fla's dhu). `q`, `k`, `w`, `do` and `local` are laid as the forward's; `dht` `[B * HV *
    Dk, Dv]` fp32 each row's final memory's gradient. A program walks `tiles.walkers` value blocks of its
    sequence and head, one a consumer role, as the states launch does. Without the delta rule `w` is None and
    the memory's gradient takes no `w^T dv` (`algebra.reverse_carried`)."""
    width, value_width, family = call.width, call.value_width, call.family
    bv, slices, walkers = tiles.block_v, width // call.block, tiles.walkers
    walked_width = bv * walkers
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    grads = _value_grads(call)
    batch = dht.shape[0] // (call.heads * width)
    final = _Operand(dht.shape[0], 1, jnp.dtype(jnp.float32))
    chunks = call.chunks

    def program(q_ref, k_ref, w_ref, do_ref, local_ref, hi_ref, lo_ref, dht_ref, seq_ref, dh_ref, dv_ref,
                dh0_ref, *, streams, staging, landing):
        j, s, h = lax.axis_index("blocks"), lax.axis_index("sequences"), lax.axis_index("heads")
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
        roled = _roled(staging, landing, walkers)
        walked = j * walked_width

        def memory(base, i):
            return (base * call.heads + h) * width + i * call.block

        key_columns = h // call.group * width
        # Step `f` the chunk `chunks - 1 - f`: a sequence's chunks from its last, walked forward.
        views = _views(lambda f: (chunks - 1 - f, 0),
                       queries=given.views(q_ref, 0, call.rows, key_columns, width),
                       keys=given.views(k_ref, 0, call.rows, key_columns, width),
                       **({"w": held.views(w_ref, 0, call.rows, h * width, width)} if call.delta else {}),
                       out=given.views(do_ref, 0, call.rows, h * value_width + walked, walked_width),
                       local=grads.views(local_ref, 0, call.rows, h * value_width + walked, walked_width))
        final_views = _views(**{f"final{i}": final.views(dht_ref, memory(row, i), call.block, walked,
                                                         walked_width) for i in range(slices)})
        walked_streams = family.opened(streams["walked"], views)
        leaving = family.opened(streams["once"], final_views)

        # Each key slice of the memory's gradient and the corrected values', a store each in flight.
        before_store = _walk_stores(family, width // call.block + 1)

        def step(role):
            (staged, (second,)), lands = roled(role)
            # The decayed output gradient fp32 as two parts against the queries (fla's split product, 16 of
            # its significand's bits): one part put the initial memory's gradient 5 to 7 times the port's
            # distance from float64. Its first part reuses the memory products' tile, each consumed before
            # the next stages.
            issue = Issuer(call, {**dict.fromkeys(("values_dv", "weights_dv"), staged),
                                   "reads": (*staged, second)})
            return functools.partial(_reverse_step, lands, issue, walked + role * bv, pl.ds(role * bv, bv))

        def _reverse_step(lands, issue, column, columns, f, slots, grad, accumulators):
            n = chunks - 1 - f
            span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
            for i, part in enumerate(grad):
                before_store()
                memories.store(dh_ref, part, memory(n, i), call.block, column, bv, family=family,
                               smem=lands[f"memory{i}"], wait=False)

            def block(name, operand, i):
                return _tile(slots, name, operand, columns=pl.ds(i * call.block, call.block))

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            out = _tile(slots, "out", given, columns=columns)
            local = _tile(slots, "local", grads, columns=columns).value().astype(jnp.float32)
            ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
            grad, dv = algebra.reverse_carried(
                grad, functools.partial(block, "keys", given),
                functools.partial(block, "w", held) if call.delta else None,
                functools.partial(block, "queries", given), out.value(), local, gate, low, ending,
                ending_low, scale, held.dtype, issue)
            before_store()
            grads.store(dv_ref, dv, n * CHUNK, CHUNK, h * value_width + column, bv, family=family,
                        smem=lands["corrected"], wait=False)
            return grad, accumulators

        _sequence_walk(call, final, "final", (leaving, final_views), (walked_streams, views), step,
                       chunks - first - count, count, ends, begins,
                       lambda role, i: dh0_ref.at[pl.ds(memory(row, i), call.block),
                                                  pl.ds(walked + role * bv, bv)], bv,
                       landing=lambda role: roled(role)[1].get("final"), walkers=walkers)

    out_type = (memories.shape(value_width), grads.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((batch * call.heads * width, value_width), jnp.float32))
    blocks = _streamed(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                       **({"w": ((CHUNK, width), held)} if call.delta else {}),
                       out=((CHUNK, walked_width), given), local=((CHUNK, walked_width), grads))
    finals = _streamed(**{f"final{i}": ((call.block, walked_width), final) for i in range(slices)})
    # The memory's carried gradient: three memory products a chunk, two without the delta rule's weights.
    steps = chunks / sequences.shape[0]
    work = Work(2 * (2 + call.delta) * CHUNK * width * walked_width * steps, steps)
    # The memory's gradient carried, its key slices in fp32, and the one it gives the chunk before as it is
    # summed; a step's two fp32 tiles at once, the corrected values' gradient beside the local one it adds
    # or the decayed output's gradient read after it (`algebra.reverse_carried`: the local one folds into it
    # before the read forms), beside the output's gradient, a memory slice as an operand, and the stored
    # tiles' casts.
    held_tiles = _Held(carried=[_f32(width, bv)] * 2, phases=(
        [_f32(CHUNK, bv)] * 2 + _held(given, CHUNK, bv)
        + _held(held, call.block, bv) + _held(memories, call.block, bv) + _held(grads, CHUNK, bv)
        + _fragments(call, (given, CHUNK, call.block), (held, CHUNK, call.block)),))
    final_landing, after = _final_landing(call, bv)
    return _kernel(program, call, out_type=out_type,
                   grid=(value_width // walked_width, sequences.shape[0], call.heads),
                   grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=2, work=work,
                   held=held_tiles, consumers=walkers,
                   live=lambda refs: refs[8][lax.axis_index("sequences"), 1] > 0,
                   staging=(*_staging(held, (call.block, bv)), (plgpu.SMEM((CHUNK, bv), held.stored),)),
                   once=finals,
                   landing={**_memory_landing(call, bv), "corrected": grads.landing(CHUNK, bv),
                            **final_landing},
                   # Without the delta rule the weights' place takes the keys, an operand no step reads.
                   after=after)(q, k, k if w is None else w, do, local, hi, lo, dht, sequences)


def _query_key_views(call: _Call, keyed: tuple, served: tuple, memory: tuple, *, row, key_columns, first,
                     chunk, block) -> dict:
    """`_query_key`'s streams' views (`_views`) for a program's key block `block` of chunk `chunk`, its key
    head's first value head `first`: the queries' and keys' block (`keyed`, their refs), and each step `i`
    the group's `i`-th value head's values, output gradient and corrected values' gradient (`served`) at
    its columns, and its entered memory and leaving memory's gradient (`memory`) a head's rows apart."""
    width, value_width, group = call.width, call.value_width, call.group
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    q_ref, k_ref = keyed
    operands = dict(zip(("values", "out", "value_grads"), ((held, served[0]), (given, served[1]),
                                                           (held, served[2])), strict=True))
    if not call.delta:
        # Without the delta rule no weights, so no stream of the corrected values' gradient.
        del operands["value_grads"]
    memory_row = (chunk * call.heads + first) * width + block * call.block
    return {**_views(lambda i: (0, 0), queries=given.views(q_ref, row, CHUNK, key_columns, call.block),
                     keys=given.views(k_ref, row, CHUNK, key_columns, call.block)),
            **_views(lambda i: (0, i), **{name: operand.views(ref, row, CHUNK, first * value_width,
                                                              group * value_width)
                                          for name, (operand, ref) in operands.items()}),
            **_views(lambda i: (i * (width // call.block), 0),
                     **{name: memories.views(ref, memory_row, group * width, 0, value_width)
                        for name, ref in zip(("entered", "leaving"), memory, strict=True)})}


def _gradient_stored(call: _Call, kind: str, block: int, grad, *, weights: Sequence, sums: dict, first,
                     landing: dict) -> None:
    """`algebra.memory_gradients`' emit in `_query_key`: the weights' gradient (`kind` 'dw') in the
    operands' dtype to `weights` (its parts' views) as it is final, as fla's dqkwg writes it; the queries'
    and keys' ('dq', 'dk') to `sums[kind]` summed over the key head's value heads in fp32, stored at the
    first (`first`) and added at the rest (`Family.added`: a TMA reduction, or whole rows through
    registers), each through its launch's landing tiles."""
    del block
    family = call.family
    if kind == "dw":
        pieces = product.formed(grad.astype(jnp.float32), call.held.parts) if call.held.parts > 1 else (grad,)
        family.stored([piece.astype(call.held.stored) for piece in pieces], weights, smem=landing["weights"])
        return

    @pl.when(first)
    def _():
        family.stored([grad], [sums[kind]], smem=landing["sums"])

    @pl.when(jnp.logical_not(first))
    def _():
        family.added(grad, sums[kind], call.interpret, smem=landing["sums"])


def _memory_sum(call: _Call, issue: Issuer, entered: Parts, leaving: Parts) -> jax.Array:
    """`sum(S dS)` over the value columns of a block of the memories entered and left, `[rows]`: the
    diagonal of their product `S^T dS` where the family's products read registers, else their elementwise
    product's rows (`_query_key`)."""
    if call.family.operands_in_registers:
        traced = issue("memory_trace", entered.T, leaving)
        return jnp.sum(jnp.where(_diagonal(traced.shape[0], call.family), traced, 0.0), axis=1)
    return jnp.sum(entered.value().astype(jnp.float32) * leaving.value().astype(jnp.float32), axis=1)


def _query_key_held(call: _Call) -> _Held:
    """What `_query_key`'s consumer thread holds (`algebra.memory_gradients`): a block's gradient's fp32 sum,
    its memory product and its scores' product, and a block's queries or keys in fp32 for the range terms;
    then `sum(S dS)` (`_memory_sum`): the memories' product and its diagonal's mask where products read
    registers, with a block of each operand, else the memories in fp32."""
    held, memories, given = call.held, call.memories, call.given
    block = (memories, call.block, call.block)
    summed = ([_f32(call.block, call.block), *_masks(call.block, call.block), *_fragments(call, block, block)]
              if call.family.operands_in_registers else [_f32(call.block, call.block)] * 2)
    return _Held(phases=(
        [_f32(CHUNK, call.block)] * 4
        + _fragments(call, (given, CHUNK, call.block), (held, CHUNK, call.block), (memories, call.block,
                                                                                      call.block)),
        summed))


@_lowered_once
def _query_key(q, k, corrected, do, dv, h, dh, ds, hi, lo, taken, beta, call: _Call, *, scale: float):
    """`(dq, dk [R, H * Dk] fp32, dw [R, HV * Dk] as call.held, after, leaving [blocks, HV, R], memory
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
    leaving memory's gradient (`call.memories`)."""
    width, value_width, family = call.width, call.value_width, call.family
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    grads = _score_grads(call)
    blocks, group = value_width // call.block, call.group

    def program(q_ref, k_ref, v_ref, do_ref, dv_ref, h_ref, dh_ref, ds_ref, hi_ref, lo_ref, taken_ref,
                beta_ref, dq_ref, dk_ref, dw_ref, after_ref, leaving_ref, memory_ref, *, streams, staging,
                landing):
        del staging
        c, n, keyed = lax.axis_index("blocks"), lax.axis_index("chunks"), lax.axis_index("heads")
        row, span, last = n * CHUNK, pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
        key_columns = keyed * width + c * call.block
        first = keyed * group
        views = _query_key_views(call, (q_ref, k_ref), (v_ref, do_ref, dv_ref), (h_ref, dh_ref), row=row,
                                 key_columns=key_columns, first=first, chunk=n, block=c)
        # Step `i` the group's `i`-th value head's scores' gradient; without the delta rule each step the key
        # head's, summed over its value heads (`_scores_back`).
        scored = (first, group) if call.delta else (keyed, 1)
        views |= _views(lambda i: (0, i if call.delta else 0),
                        ds=grads.views(ds_ref, row, CHUNK, scored[0] * CHUNK, scored[1] * CHUNK))

        def step(index, slots, carry, accumulators, *, scores: bool = True):
            h = first + index

            column = functools.partial(_column, call, slots)

            def whole(name, operand, i):
                del i
                return _tile(slots, name, operand)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
            first_head = index == 0

            stored = functools.partial(
                _gradient_stored, call, weights=held.views(dw_ref, row, CHUNK, h * width + c * call.block,
                                                           call.block),
                sums={"dq": dq_ref.at[span, pl.ds(key_columns, call.block)],
                      "dk": dk_ref.at[span, pl.ds(key_columns, call.block)]},
                first=first_head, landing=landing)

            issue = Issuer(call)
            after, leaving = algebra.memory_gradients(
                functools.partial(whole, "queries", given), functools.partial(whole, "keys", given), 1,
                functools.partial(column, "values", held), functools.partial(column, "out", given),
                functools.partial(column, "value_grads", held) if call.delta else None,
                lambda i, j: column("entered", memories, j), lambda i, j: column("leaving", memories, j),
                blocks, _tile(slots, "ds", grads) if scores else None, gate, low, ending, ending_low, None,
                scale, issue,
                emit=stored, row_scale=family.rows_of(beta_ref.at[h, span]) if call.gated_keys else None)
            after_ref.at[c, h, span][...] = after
            leaving_ref.at[c, h, span][...] = leaving
            # `sum(S dS)` over the block's rows, read here rather than both memories again: where products
            # read registers (mma.sync's), each value block's `S^T dS` diagonal, a product of the memories
            # as they lie in shared memory (their elementwise product held two fp32 tiles beside the step's
            # fragments, past an A100's 255 registers); where they read shared memory, that elementwise
            # product (the product's accumulator spilled 44 bytes a thread on an H100's).
            memory_ref.at[c, h, pl.ds(n * call.block, call.block)][...] = functools.reduce(jnp.add, (
                _memory_sum(call, issue, column("entered", memories, j), column("leaving", memories, j))
                for j in range(blocks)))
            # The step's reduce-adds ('dq', 'dk', `Family.added`) waited on by design, not by 'dw''s store
            # being emitted after them: Mosaic GPU waits on no copy at a kernel's exit (template numerics
            # review 2026-10-26, P3-4).
            family.store_wait(0)
            return carry, accumulators

        # Without the delta rule the scores' products the first step's alone.
        _walk(streams, views, step, call, steps=group, segments=() if call.delta else (
            (step, 0, 1), (functools.partial(step, scores=False), 1, group)))

    tiles = _streamed(queries=((CHUNK, call.block), given), keys=((CHUNK, call.block), given),
                      values=((CHUNK, value_width), held), out=((CHUNK, value_width), given),
                      **({"value_grads": ((CHUNK, value_width), held)} if call.delta else {}),
                      entered=((call.block, value_width), memories),
                      leaving=((call.block, value_width), memories), ds=((CHUNK, CHUNK), grads))
    keyed_out = jax.ShapeDtypeStruct((call.rows, call.heads // group * width), jnp.float32)
    terms = jax.ShapeDtypeStruct((width // call.block, call.heads, call.rows), jnp.float32)
    out_type = (keyed_out, keyed_out, held.shape(call.heads * width), terms, terms,
                jax.ShapeDtypeStruct((width // call.block, call.heads, call.chunks * call.block),
                                     jnp.float32))
    # Each value head of the group: three memory products (two without the delta rule's weights) and the
    # scores' gradient's two, a key head's two without the delta rule.
    scores = 2 * CHUNK * CHUNK * call.block
    each = (2 + call.delta) * CHUNK * value_width * call.block + scores * call.delta
    work = Work(2 * (each * group + scores * (not call.delta)), group)
    return _kernel(program, call, out_type=out_type,
                   grid=(width // call.block, call.chunks, call.heads // group),
                   grid_names=("blocks", "chunks", "heads"), blocks=tiles, stages=1, live=_taken(10),
                   work=work, held=_query_key_held(call),
                   landing={"weights": held.landing(CHUNK, call.block),
                            "sums": ((CHUNK, call.block), jnp.float32)})(
        q, k, corrected, do, dv, h, dh, ds, hi, lo, taken, beta)


def _value_grads(call: _Call) -> _Operand:
    """The form the corrected values' gradient `du` passes from `_local_back` through `_reverse` in: the held
    operand's for the delta rule (fla rounds its `dv` so); without it fp32, as mamba_ssm accumulates `dx` and
    sums `ddt` from it in fp32 (`_chunk_scan_chunk_state_bwd_dx`): rounded, a token's write gate's gradient
    `sum(v du)` took the rounding of every `du` it sums."""
    return dataclasses.replace(call.held, rows=call.rows) if call.delta else _Operand(call.rows, 1,
                                                                                       jnp.dtype(jnp.float32))


def _score_grads(call: _Call) -> _Operand:
    """The form the scores' gradient passes from `_scores_back` to `_query_key` in: the held operand's, each
    value head's, for the delta rule (fla's dqkwg rounds it so); without it the operands', a key head's summed
    over its value heads in fp32 first, as mamba_ssm sums its `dCB` over a group's heads before its products
    round it to bf16 (`_chunk_scan_bwd_dcb`, `_bmm_chunk_bwd`'s `dot_dtype`). Each value head's rounded on
    its own, even in two bf16 parts, a one-token document's C row, its size 1.3e-6 where the rest sat at 0.1
    to 0.5, was 0.242 of it from float64 against mamba_ssm's allowed 0.223 (ssd-c-rows-h100-b9d82d9b7)."""
    held = dataclasses.replace(call.held, rows=call.rows)
    return held if call.delta else dataclasses.replace(call.given, rows=call.rows)


@_lowered_once
def _scores_back(q, k, corrected, do, hi, lo, taken, beta, call: _Call, *, scale: float):
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

    def program(q_ref, k_ref, v_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref,
                beta_ref, ds_ref, decay_ref, *, streams, staging, landing):
        n, head = _axis("chunks"), _axis("heads")
        first = head * group
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = first // call.group * width
        views = _views(lambda i: (0, 0), queries=given.views(q_ref, row, CHUNK, keyed, width),
                       keys=given.views(k_ref, row, CHUNK, keyed, width))
        # Step `i` the `i`-th value head's.
        columns = first * value_width, group * value_width
        views |= _views(lambda i: (0, i), values=held.views(v_ref, row, CHUNK, *columns),
                        out=given.views(do_ref, row, CHUNK, *columns))
        vectors = {"gate": hi_ref, "low": lo_ref, "column_gate": column_hi_ref, "column_low": column_lo_ref}
        # A program a value head streams its vectors where the family copies them; a key head's steps read
        # theirs.
        streamed = family.copies_vectors and group == 1
        if streamed:
            views |= _vector_views(n, first, **vectors)
        summing, accumulated = (staging, None) if group == 1 else (staging[:-1], staging[-1])

        def step(index, slots, carry, accumulators, loaded=None):
            h = first + index
            found = _prefix_reader(family, streamed, vectors, h, span)(slots) if loaded is None else loaded
            (gate, low), laid = found

            column = functools.partial(_column, call, slots)

            summed, column_sums = _summed(tuple(summing), call)
            ds, pairs = algebra.scores_gradient(
                functools.partial(column, "queries", given), functools.partial(column, "keys", given),
                width // call.block, functools.partial(column, "values", held),
                functools.partial(column, "out", given), blocks, gate, low, scale,
                grads.dtype if accumulated is None else jnp.float32, Issuer(call), laid=laid,
                column_scale=family.columns_of(beta_ref.at[h, span]) if call.gated_keys else None)
            if accumulated is None:
                grads.store(ds_ref, ds, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            else:
                # The key head's sum so far, in shared memory from step to step, zero at its first.
                @pl.when(index == 0)
                def _():
                    accumulated[...] = jnp.zeros(accumulated.shape, jnp.float32)

                ds = family.tile_of(accumulated) + ds
                accumulated[...] = ds

                @pl.when(index == group - 1)
                def _():
                    grads.store(ds_ref, ds.astype(grads.dtype), row, CHUNK, head * CHUNK, CHUNK,
                                family=family, smem=landing["square"])

            decay_ref.at[h, span][...] = algebra.summed_pairs(pairs, summed, column_sums)
            return carry, accumulators

        read = _prefix_reader(family, streamed, vectors, first, span)
        prologue = None if streamed or group > 1 else lambda: read(None)
        _walk(streams, views, step, call, steps=group, prologue=prologue)

    tiles = _streamed(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                      values=((CHUNK, value_width), held), out=((CHUNK, value_width), given))
    if group == 1:
        tiles |= _vector_blocks(call, "gate", "low", "column_gate", "column_low")
    out_type = (grads.shape((call.heads // group) * CHUNK),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    # The scores' gradient, the scores and the pairs' sums, each value head's.
    work = Work(2 * CHUNK * CHUNK * (value_width + width + CHUNK) * group, group,
                exponentials=CHUNK * CHUNK * group)
    # The scores' gradient fp32 and narrow, the pairs' decays, the scores and the pair terms, the masks, and
    # the pairs' sum's product: four fp32 tiles at once, the decays taken before the scores' product
    # (`algebra.scores_gradient`); five stated an A100's 280 registers where ptxas needed 240. A key head's
    # sum read back beside them.
    held_tiles = _Held(phases=([_f32(CHUNK, CHUNK)] * (4 + (group > 1)) + _held(grads, CHUNK, CHUNK)
                               + _masks(CHUNK, CHUNK)
                               + _fragments(call, (given, CHUNK, call.block), (given, CHUNK, call.block)),))
    summing = _summing(call) + ((plgpu.SMEM((CHUNK, CHUNK), jnp.float32),) if group > 1 else ())
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads // group),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(8), work=work,
                   held=held_tiles, staging=summing, landing={"square": grads.landing(CHUNK, CHUNK)},
                   persistent=True)(q, k, corrected, do, hi, lo, hi, lo, taken, beta)


@_lowered_once
def _system_back(a, k, v, dw, du, hi, lo, beta, taken, call: _Call):
    """`dL [R, HV * C]` as `call.held`: the WY representation's backward's system part a chunk and value head
    at a time (fla's prepare_wy_repr_bwd's, `algebra.system_gradient`) from `w`'s and `u`'s gradients `dw`,
    `du` (`call.held`) and the inverse `a` less its unit diagonal (`_solved`'s), which `_keys_back` takes.
    Apart from `_keys_back` and `_values_back`, each launch's tiles a thread's registers hold (in one they
    spilled 360 bytes a thread on an H100's)."""
    width, value_width, family = call.width, call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(a_ref, k_ref, v_ref, dw_ref, du_ref, hi_ref, lo_ref, beta_ref, column_hi_ref, column_lo_ref,
                taken_ref, dl_ref, *, streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        views = _views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                       keys=given.views(k_ref, row, CHUNK, h // call.group * width, width),
                       values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                       w_grads=held.views(dw_ref, row, CHUNK, h * width, width),
                       u_grads=held.views(du_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(_column, call, slots)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            beta = family.rows_of(beta_ref.at[h, span])
            d_l = algebra.system_gradient(
                _tile(slots, "lower", held), functools.partial(column, "keys", given), width // call.block,
                functools.partial(column, "values", given), value_width // call.block,
                functools.partial(column, "w_grads", held), functools.partial(column, "u_grads", held), beta,
                gate, low, held.dtype, Issuer(call, dict.fromkeys(("inverse_t", "inverse_dx", "inverse_l"),
                                                                   staged)),
                laid=(family.columns_of(column_hi_ref.at[h, span]),
                      family.columns_of(column_lo_ref.at[h, span])))
            held.store(dl_ref, d_l, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(lower=((CHUNK, CHUNK), held), keys=((CHUNK, width), given),
                      values=((CHUNK, value_width), given), w_grads=((CHUNK, width), held),
                      u_grads=((CHUNK, value_width), held))
    # The weighted keys' and values' products into the inverse's gradient, and its products through it.
    work = Work(2 * CHUNK * CHUNK * (width + value_width + 2 * CHUNK))
    # A block's keys or values scaled, narrow, beside the inverse's gradient; then it masked, its products
    # through the inverse and the pairs' decays with their masks (`algebra.system_gradient`).
    held_tiles = _Held(phases=(
        [_f32(CHUNK, CHUNK), _f32(CHUNK, call.block), *_held(held, CHUNK, call.block),
         *_fragments(call, (held, CHUNK, call.block), (held, CHUNK, call.block))],
        [_f32(CHUNK, CHUNK)] * 4 + _held(held, CHUNK, CHUNK) + _masks(CHUNK, CHUNK)
        + _fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, CHUNK))))
    return _kernel(program, call, out_type=held.shape(call.heads * CHUNK), grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(10), work=work,
                   held=held_tiles, staging=_staging(held, (CHUNK, CHUNK)),
                   landing={"square": held.landing(CHUNK, CHUNK)})(a, k, v, dw, du, hi, lo, beta, hi, lo,
                                                                   taken)


@_lowered_once
def _values_back(a, v, du, beta, taken, call: _Call):
    """`dv [R, HV * Dv]` as `call.held` and `dbeta [HV, R]` fp32: the WY representation's backward's values'
    part a chunk and value head at a time (`algebra.values_gradient`): the values' gradient and beta's
    through `u`."""
    value_width, family = call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(a_ref, v_ref, du_ref, beta_ref, taken_ref, dv_ref, dbeta_ref, *, streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        views = _views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                       values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                       u_grads=held.views(du_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(_column, call, slots)
            stored = _block_stored(call, dv_ref, row, h * value_width, landing["block"])

            dbeta_ref.at[h, span][...] = algebra.values_gradient(
                _tile(slots, "lower", held), functools.partial(column, "values", given),
                value_width // call.block, functools.partial(column, "u_grads", held),
                family.rows_of(beta_ref.at[h, span]), held.dtype, Issuer(call, {"inverse_t": staged}),
                emit=stored)
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(lower=((CHUNK, CHUNK), held), values=((CHUNK, value_width), given),
                      u_grads=((CHUNK, value_width), held))
    out_type = (held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    work = Work(2 * CHUNK * CHUNK * value_width)
    # A value block's gradient in fp32 and narrow, its product through the inverse, and the values in fp32.
    held_tiles = _Held(phases=(
        [_f32(CHUNK, call.block)] * 3 + _held(held, CHUNK, call.block)
        + _fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, call.block)),))
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(4), work=work,
                   held=held_tiles, staging=_staging(held, (CHUNK, call.block)),
                   landing={"block": held.landing(CHUNK, call.block)})(a, v, du, beta, taken)


@_lowered_once
def _keys_back(a, k, dw, dl, hi, lo, beta, taken, call: _Call):
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

    def program(a_ref, k_ref, dw_ref, dl_ref, hi_ref, lo_ref, beta_ref, taken_ref, dk_ref, dbeta_ref,
                after_ref, decay_ref, *, streams, staging, landing):
        c, n, h = _axis("blocks"), _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        staged, *summing = staging
        column = c * call.block
        views = _views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                       keys=given.views(k_ref, row, CHUNK, h // call.group * width + column, call.block),
                       w_grads=held.views(dw_ref, row, CHUNK, h * width + column, call.block),
                       system=held.views(dl_ref, row, CHUNK, h * CHUNK, CHUNK))

        def step(index, slots, carry, accumulators):
            del index

            def whole(name, operand, block):
                del block
                return _tile(slots, name, operand)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            beta = family.rows_of(beta_ref.at[h, span])
            stored = _block_stored(call, dk_ref, row, h * width + column, landing["block"])

            d_beta, after, pairs = algebra.keys_gradient(
                _tile(slots, "lower", held), functools.partial(whole, "keys", given), 1,
                functools.partial(whole, "w_grads", held), _tile(slots, "system", held), beta, gate, low,
                held.dtype, Issuer(call, dict.fromkeys(("inverse_t", "keys_dl", "keys_dlt"), staged)),
                emit=stored)
            summed, columns = _summed(tuple(summing), call)
            decay_ref.at[c, h, span][...] = algebra.summed_pairs(pairs, summed, columns)
            dbeta_ref.at[c, h, span][...] = d_beta
            after_ref.at[c, h, span][...] = after
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(lower=((CHUNK, CHUNK), held), keys=((CHUNK, call.block), given),
                      w_grads=((CHUNK, call.block), held), system=((CHUNK, CHUNK), held))
    blocks = width // call.block
    terms = jax.ShapeDtypeStruct((blocks, call.heads, call.rows), jnp.float32)
    out_type = (held.shape(call.heads * width), terms, terms, terms)
    # The key block's products through `w` and the system, and its keys' own pairs.
    work = Work(2 * CHUNK * CHUNK * (4 * call.block))
    # The system's gradient narrow and scaled by beta beside the key block's terms: through `w`, the keys in
    # fp32, the product being issued and the sum it joins (`algebra.keys_gradient`: each term dies as the next
    # forms); then its keys' own pairs, the pair terms, and the pairs' sums' masks and product.
    held_tiles = _Held(phases=(
        _held(held, CHUNK, CHUNK) * 2 + [_f32(CHUNK, call.block)] * 4
        + _fragments(call, (held, CHUNK, CHUNK), (given, CHUNK, call.block), (held, CHUNK, call.block)),
        _held(held, CHUNK, CHUNK) + [_f32(CHUNK, CHUNK)] * 4 + _masks(CHUNK, CHUNK)))
    return _kernel(program, call, out_type=out_type, grid=(blocks, call.chunks, call.heads),
                   grid_names=("blocks", "chunks", "heads"), blocks=tiles, stages=1, live=_taken(7),
                   work=work, held=held_tiles, staging=_staging(held, (CHUNK, CHUNK)) + _summing(call),
                   landing={"block": held.landing(CHUNK, call.block)})(a, k, dw, dl, hi, lo, beta, taken)
