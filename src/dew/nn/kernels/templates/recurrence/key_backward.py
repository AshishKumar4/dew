"""The fast backward of the gated delta rule under a decay a key dimension (Kimi delta attention's), fla_kda's
chunk_kda_bwd DAG through the algebra's per-key backward tile functions: each chunk's local part
(`_key_scores_back`, fla_kda's dAv), the memory's reverse walk under the per-key decay
(`mainloop_backward._reverse`, its gated queries and keys), the memory's and the WY representation's part
(`_key_memory_back`, fla_kda's wy_dqkg) and the chunk's pairs' part (`_key_pairs_back`, fla_kda's intra),
which XLA sums over a key head's value heads and cumulates over each chunk (`gradients`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import product
from dew.nn.kernels.templates.recurrence.issuing import Issuer, Parts
from dew.nn.kernels.templates.recurrence.mainloop import (
    CHUNK,
    FAST_PARTS,
    Work,
    _axis,
    _Call,
    _column,
    _f32,
    _fragments,
    _Held,
    _held,
    _kernel,
    _key_block,
    _lowered_once,
    _masks,
    _Operand,
    _staging,
    _streamed,
    _taken,
    _tile,
    _views,
    _walk,
)


def _fp32(call: _Call) -> _Operand:
    """The form of the launches' fp32 gradients between them: one fp32 part, a row a token."""
    return _Operand(call.rows, 1, jnp.dtype(jnp.float32))


def _parted(call: _Call) -> _Operand:
    """The form a register operand of the pairs' gradients' products takes: fp32 in `FAST_PARTS` bf16 parts
    (a transposed fp32 operand is no warpgroup MMA's, and 16 bits of each, past the TF32 fla_kda's intra
    takes them in)."""
    return _Operand(call.rows, FAST_PARTS, jnp.dtype(jnp.float32))


@_lowered_once
def _key_scores_back(scores, do, corrected, taken, call: _Call):
    """`(dAqk [R, HV * C] fp32, dv [R, HV * Dv])`: each chunk and value head's scores' gradient and the
    corrected values' gradient from the chunk's own output (`algebra.key_scores_back`, fla_kda's dAv), the
    latter in `_value_grads`' form, from the intra launch's scores (`call.held`), a program a chunk and value
    head (persistent)."""
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    value_width, family = call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    grads, square = _value_grads(call), _fp32(call)
    blocks = value_width // call.block

    def program(s_ref, do_ref, v_ref, taken_ref, ds_ref, dv_ref, *, streams, staging, landing):
        del staging
        n, h = _axis("chunks"), _axis("heads")
        row = n * CHUNK
        views = _views(scores=held.views(s_ref, row, CHUNK, h * CHUNK, CHUNK),
                       out=given.views(do_ref, row, CHUNK, h * value_width, value_width),
                       corrected=held.views(v_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            def stored(j, dv):
                grads.store(dv_ref, dv, row, CHUNK, h * value_width + j * call.block, call.block,
                            family=family, smem=landing["row"])

            d_scores = algebra.key_scores_back(
                _tile(slots, "scores", held), functools.partial(_column, call, slots, "out", given),
                functools.partial(_column, call, slots, "corrected", held), blocks, Issuer(call), emit=stored)
            square.store(ds_ref, d_scores, row, CHUNK, h * CHUNK, CHUNK, family=family,
                         smem=landing["square"])
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(scores=((CHUNK, CHUNK), held), out=((CHUNK, value_width), given),
                      corrected=((CHUNK, value_width), held))
    # The scores' gradient summed over the value blocks, then held beside each block's local product.
    held_tiles = _Held(carried=[_f32(CHUNK, CHUNK)], phases=(
        _masks(CHUNK, CHUNK) + _fragments(call, (given, CHUNK, call.block), (held, CHUNK, call.block)),
        [_f32(CHUNK, call.block), *_held(grads, CHUNK, call.block),
         *_fragments(call, (held, CHUNK, CHUNK), (given, CHUNK, call.block))]))
    return _kernel(program, call,
                   out_type=(square.shape(call.heads * CHUNK), grads.shape(call.heads * value_width)),
                   grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=tiles, stages=1,
                   live=_taken(3), work=Work(4 * CHUNK * CHUNK * value_width), held=held_tiles,
                   landing={"square": square.landing(CHUNK, CHUNK), "row": grads.landing(CHUNK, call.block)},
                   persistent=True)(scores, do, corrected, taken)


@_lowered_once
def _key_memory_back(q, k, hi, lo, do, corrected, dv, v, memory, dh, a, beta, taken, call: _Call, *,
                     scale: float):
    """`(dq, dk, dg [R, HV * Dk], dv [R, HV * Dv], dA [R, HV * C] fp32, dbeta [HV, R] fp32)`: each chunk and
    value head's gradients from the memory and the WY representation (`algebra.key_memory_block` a key block
    a step, its system's and write gates' terms carried across them, then `algebra.key_values_back` and
    `algebra.key_system_back` on the last; fla_kda's wy_dqkg), the queries' and keys' each value head's, which
    `gradients` sums over its key head's. `memory` and `dh` the memories each chunk enters and the gradient
    of the one it leaves (`call.memories`), `dv` the corrected values' whole gradient (`_reverse`'s), `a` the
    inverse less its unit diagonal (`_intra_solve`'s)."""
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    width, value_width, family = call.width, call.value_width, call.family
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    gates, grads, square = call.gates, _value_grads(call), _fp32(call)
    keys = _key_block(call, functools.partial(_key_memory_held, call))
    steps, blocks = width // keys, value_width // call.block
    landing = _landings(call, keys)

    def program(q_ref, k_ref, hi_ref, lo_ref, do_ref, v_new_ref, dv_ref, v_ref, h_ref, dh_ref, a_ref,
                beta_ref, taken_ref, dq_ref, dk_ref, dg_ref, dv_out_ref, da_ref, db_ref, *, streams, staging,
                landing):
        squared = landing[_square_landing(keys)]
        weights, values, system = staging
        n, h = lax.axis_index("chunks"), lax.axis_index("heads")
        row, span, last = n * CHUNK, pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
        keyed, memory_row = h // call.group * width, (n * call.heads + h) * width
        views = {**_views(lambda i: (0, 0),
                          out=given.views(do_ref, row, CHUNK, h * value_width, value_width),
                          corrected=held.views(v_new_ref, row, CHUNK, h * value_width, value_width),
                          value_grads=grads.views(dv_ref, row, CHUNK, h * value_width, value_width),
                          values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                          lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK)),
                 **_views(lambda i: (0, i),
                          queries=given.views(q_ref, row, CHUNK, keyed, width),
                          keys=given.views(k_ref, row, CHUNK, keyed, width),
                          gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                          lows=gates.views(lo_ref, row, CHUNK, h * width, width)),
                 **_views(lambda i: (i, 0),
                          entered=memories.views(h_ref, memory_row, width, 0, value_width),
                          leaving=memories.views(dh_ref, memory_row, width, 0, value_width))}
        issue = Issuer(call, {"weights_merge": list(weights), "values_merge": values, "system_merge": system})

        def step(index, slots, carry, accumulators):
            d_system, d_betas = carry
            # Every slot acquired in the step's own trace: the values' is read on the last step alone, and
            # first acquired under its `pl.when` the other steps' waits went untaken (the interpreter:
            # "Barrier completed phase 1, but no threads observed phase 0", kda-cpu5 at b9450c237).
            slots.acquire_all()
            column = functools.partial(_column, call, slots)
            betas = family.rows_of(beta_ref.at[h, span])
            at = pl.ds(h * width + index * keys, keys)
            last_prefix = tuple(family.columns_of(ref.at[last, at]) for ref in (hi_ref, lo_ref))

            def stored(kind, grad):
                ref = {"dq": dq_ref, "dk": dk_ref, "dg": dg_ref}[kind]
                square.store(ref, grad, row, CHUNK, h * width + index * keys, keys, family=family,
                             smem=landing["block"])

            lower = _tile(slots, "lower", held)
            loads = {name: functools.partial(column, name, operand) for name, operand in (
                ("out", given), ("corrected", held), ("value_grads", grads), ("values", given),
                ("entered", memories), ("leaving", memories))}
            system, beta_terms = algebra.key_memory_block(
                _tile(slots, "queries", given), _tile(slots, "keys", given),
                (_tile(slots, "gates", gates).value(), _tile(slots, "lows", gates).value()), last_prefix,
                loads["out"], loads["corrected"], loads["value_grads"], loads["entered"], loads["leaving"],
                blocks, lower, betas, scale, held.dtype, issue, emit=stored)
            d_system, d_betas = d_system + system, d_betas + beta_terms

            # The values' part and the system's gradient once every key block's terms are in.
            @pl.when(index == steps - 1)
            def _():
                def values_stored(j, grad):
                    square.store(dv_out_ref, grad, row, CHUNK, h * value_width + j * call.block, call.block,
                                 family=family, smem=landing["values"])

                summed, beta_terms = algebra.key_values_back(
                    loads["value_grads"], loads["values"], blocks, lower, betas, held.dtype, issue,
                    emit=values_stored)
                d_system_back = algebra.key_system_back(d_system + summed, lower, held.dtype, issue)
                square.store(da_ref, d_system_back, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=squared)
                db_ref.at[h, span][...] = d_betas + beta_terms

            return (d_system, d_betas), accumulators

        # The carries laid out as their sums are: splats, Mosaic GPU's layout inference ran out of its
        # search's fuel (kda-lower4-sm90 at 739b21575).
        _walk(streams, views, step, call, steps=steps,
              carry=(family.as_tile(family.zeros((CHUNK, CHUNK))), family.as_row(family.zeros((CHUNK,)))))

    tiles = _streamed(out=((CHUNK, value_width), given), corrected=((CHUNK, value_width), held),
                      value_grads=((CHUNK, value_width), grads), values=((CHUNK, value_width), given),
                      lower=((CHUNK, CHUNK), held), queries=((CHUNK, keys), given),
                      keys=((CHUNK, keys), given), gates=((CHUNK, keys), gates),
                      lows=((CHUNK, keys), gates), entered=((keys, value_width), memories),
                      leaving=((keys, value_width), memories))
    staging = ([_staging(held, (CHUNK, keys))[0] for _ in range(2)],
               _staging(held, (CHUNK, call.block))[0], _staging(held, (CHUNK, CHUNK))[0])
    fp32_rows = square.shape(call.heads * width)
    out_type = (fp32_rows, fp32_rows, fp32_rows, square.shape(call.heads * value_width),
                square.shape(call.heads * CHUNK), jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    products = 2 * CHUNK * keys * value_width * 3 * steps + 2 * CHUNK * CHUNK * value_width * 2
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(12),
                   work=Work(products, steps), held=_key_memory_held(call, keys), staging=staging,
                   landing={**landing, "values": square.landing(CHUNK, call.block)})(
        q, k, hi, lo, do, corrected, dv, v, memory, dh, a, beta, taken)


def _key_memory_held(call: _Call, keys: int) -> _Held:
    """`_key_memory_back`'s consumer thread at key block `keys`: the system's gradient and the write gates'
    carried; a key block's three memory products' sums and the memories' product (`algebra.key_memory_block`'s
    'memory_trace') beside a value block's fragments; then the key block's decays, gated keys and their WY
    gradient."""
    held, given, memories = dataclasses.replace(call.held, rows=call.rows), call.given, call.memories
    return _Held(carried=[_f32(CHUNK, CHUNK), _f32(CHUNK, 1)], phases=(
        [_f32(CHUNK, keys)] * 3 + [_f32(keys, keys)]
        + _fragments(call, (given, CHUNK, call.block), (memories, keys, call.block)),
        [_f32(CHUNK, keys)] * 6 + _held(held, CHUNK, keys) * 2))


def _square_landing(keys: int) -> str:
    """The landing a launch stores a chunk's `[C, C]` fp32 tile through: its key blocks' where they are as
    wide as a chunk, else its own (`_landings`)."""
    return "block" if keys == CHUNK else "square"


def _landings(call: _Call, keys: int) -> dict:
    """A launch's landing tiles for its fp32 stores: a key block's `[C, keys]`, and a chunk's `[C, C]` where
    it is not one (`_square_landing`)."""
    square = _fp32(call)
    return {"block": square.landing(CHUNK, keys),
            **({} if keys == CHUNK else {"square": square.landing(CHUNK, CHUNK)})}


@_lowered_once
def _key_pairs_back(q, k, hi, lo, beta, d_scores, d_system, taken, call: _Call, *, scale: float):
    """`(dq, dk, dg [R, HV * Dk], dbeta [HV, R])` fp32: each chunk and value head's gradients through its
    pairs, the scores' `scale (q . k) exp2(g_q - g_k)` and the keys' system's `beta (k . k) exp2(..)`
    (`algebra.key_decayed_pairs_back` a key block a step, the write gates' terms carried; fla_kda's intra),
    given their gradients `d_scores` (`_key_scores_back`'s) and `d_system` (`_key_memory_back`'s), staged
    once a step in `_parted`'s form. The queries' and keys' each value head's, which `gradients` sums."""
    width, family = call.width, call.family
    given, gates, square, parted = call.given, call.gates, _fp32(call), _parted(call)
    keys = _key_block(call, functools.partial(_key_pairs_held, call))
    steps = width // keys

    def program(q_ref, k_ref, hi_ref, lo_ref, beta_ref, ds_ref, da_ref, taken_ref, dq_ref, dk_ref, dg_ref,
                db_ref, *, streams, staging, landing):
        pairs, squares = staging
        n, h = lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = h // call.group * width
        views = {**_views(lambda i: (0, 0), scores=square.views(ds_ref, row, CHUNK, h * CHUNK, CHUNK),
                          system=square.views(da_ref, row, CHUNK, h * CHUNK, CHUNK)),
                 **_views(lambda i: (0, i),
                          queries=given.views(q_ref, row, CHUNK, keyed, width),
                          keys=given.views(k_ref, row, CHUNK, keyed, width),
                          gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                          lows=gates.views(lo_ref, row, CHUNK, h * width, width))}

        def step(index, slots, d_betas, accumulators):
            betas = family.rows_of(beta_ref.at[h, span])
            at = pl.ds(h * width + index * keys, keys)

            def reference(sub, c):
                del c
                return tuple(family.columns_of(ref.at[row + sub * algebra.SUB, at])
                             for ref in (hi_ref, lo_ref))

            # Each pairs' gradient staged once in the step, its parts read by every product as they are and
            # transposed.
            d_pairs = []
            for name, buffers in zip(("scores", "system"), squares, strict=True):
                value = _tile(slots, name, square).value()
                d_pairs.append(Parts([family.staged(part, buffer)
                                      for part, buffer in zip(product.formed(value, parted.parts), buffers,
                                                              strict=True)], jnp.float32))
            query = _tile(slots, "queries", given).value().astype(jnp.float32)
            key = _tile(slots, "keys", given).value().astype(jnp.float32)
            found = {}
            algebra.key_decayed_pairs_back(
                d_pairs, [lambda c: query, lambda c: key * betas[:, None]], lambda c: key, 1,
                lambda c: (_tile(slots, "gates", gates).value(), _tile(slots, "lows", gates).value()),
                reference, Issuer(call, {"pairs": list(pairs)}), chunk=CHUNK,
                emit=lambda kind, p, c, grad: found.__setitem__((kind, p), grad), scales=(scale, 1.0))
            d_query, d_keyed, d_key = found["rows", 0], found["rows", 1], found["keys", 0]
            for ref, grad in ((dq_ref, d_query), (dk_ref, d_keyed * betas[:, None] + d_key),
                              (dg_ref, query * d_query + key * (d_keyed * betas[:, None] - d_key))):
                square.store(ref, grad, row, CHUNK, h * width + index * keys, keys, family=family,
                             smem=landing["block"])
            d_betas = d_betas + jnp.sum(d_keyed * key, axis=1)

            @pl.when(index == steps - 1)
            def _():
                db_ref.at[h, span][...] = d_betas

            return d_betas, accumulators

        _walk(streams, views, step, call, steps=steps, carry=family.as_row(family.zeros((CHUNK,))))

    tiles = _streamed(scores=((CHUNK, CHUNK), square), system=((CHUNK, CHUNK), square),
                      queries=((CHUNK, keys), given), keys=((CHUNK, keys), given),
                      gates=((CHUNK, keys), gates), lows=((CHUNK, keys), gates))
    staging = ([_staging(parted, (CHUNK, keys))[0] for _ in range(2)],
               [_staging(parted, (CHUNK, CHUNK))[0] for _ in range(2)])
    fp32_rows = square.shape(call.heads * width)
    out_type = (fp32_rows, fp32_rows, fp32_rows, jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    products = 2 * 2 * 2 * CHUNK * CHUNK * keys * (CHUNK // algebra.SUB) * steps
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(7),
                   work=Work(products, steps, exponentials=(CHUNK // algebra.SUB + 1) * CHUNK * width),
                   held=_key_pairs_held(call, keys), staging=staging,
                   landing={"block": square.landing(CHUNK, keys)})(
        q, k, hi, lo, beta, d_scores, d_system, taken)


def _key_pairs_held(call: _Call, keys: int) -> _Held:
    """`_key_pairs_back`'s consumer thread at key block `keys`: the write gates' terms carried; a key block's
    prefix halves, its rows' common points' and decays, its queries, keys and their gated rows, each pairs'
    gradients' sums and the keys' beside a sub-chunk's raised keys and a product's fragments."""
    parted = _parted(call)
    return _Held(carried=[_f32(CHUNK, 1)], phases=(
        [_f32(CHUNK, keys)] * 10 + _fragments(call, (parted, CHUNK, CHUNK), (parted, CHUNK, keys)),))


def gradients(run, laid_do, dht, tiles) -> tuple:
    """`(dq [R, H * Dk], dk [R, H * Dk], dv [R, HV * Dv], dg [R, HV * Dk], dbeta [R, HV], dh0 [B * HV * Dk,
    Dv])` fp32 of the gated delta rule under a decay a key dimension, laid as the prepared run's
    (`mainloop._prepared`'s, its intra launch's scores and inverse, its gated keys), fla_kda's DAG: the local
    part, the memory's reverse walk on the queries gated by their decays to the chunk's start and scaled
    (XLA's, `algebra.key_gated`'s), the memory's and WY part and the pairs' part, summed; the log decay's
    gradient each token's terms cumulated from the chunk's end (its prefix is the chunk's own), a key head's
    value heads' queries' and keys' gradients summed."""
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _reverse

    call = run.call
    rows, heads, width, group = call.rows, call.heads, call.width, call.group
    held = dataclasses.replace(call.held, rows=rows)
    query = call.given.joined(run.q) if call.given.parts > 1 else run.q
    query = jnp.repeat(query.reshape(rows, heads // group, width), group, axis=1).reshape(rows, heads * width)
    gated = held.split(algebra.key_gated(query, (run.hi, run.lo), run.scale))
    d_scores, local = _key_scores_back(run.scores, laid_do, run.corrected, run.taken, call)
    dh, dv2, dh0 = _reverse(gated, run.keys, run.w, laid_do, local, run.hi, run.lo, dht, run.sequences, None,
                            None, call, tiles, scale=run.scale)
    dq_m, dk_m, dg_m, dv, d_system, dbeta_m = _key_memory_back(
        run.q, run.k, run.hi, run.lo, laid_do, run.corrected, dv2, run.v, run.h, dh, run.a, run.beta,
        run.taken, call, scale=run.scale)
    dq_p, dk_p, dg_p, dbeta_p = _key_pairs_back(run.q, run.k, run.hi, run.lo, run.beta, d_scores, d_system,
                                                run.taken, call, scale=run.scale)
    terms = (dg_m + dg_p).reshape(call.chunks, CHUNK, heads * width)
    dg = jnp.flip(jnp.cumsum(jnp.flip(terms, 1), 1), 1).reshape(rows, heads * width)

    def keyed(x):
        """A value head's gradient `[R, HV * Dk]` summed onto its key head's `[R, H * Dk]`."""
        return x.reshape(rows, heads // group, group, width).sum(2).reshape(rows, -1)

    return keyed(dq_m + dq_p), keyed(dk_m + dk_p), dv, dg, (dbeta_m + dbeta_p).T, dh0
