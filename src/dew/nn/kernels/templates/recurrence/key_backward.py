"""The fast backward of the gated delta rule under a decay a key dimension (Kimi delta attention's), fla_kda's
chunk_kda_bwd DAG through the algebra's per-key backward tile functions: each chunk's local part
(`_key_scores_back`, fla_kda's dAv), the memory's reverse walk under the per-key decay
(`mainloop_backward._reverse`, its gated queries and keys), the memory's and the WY representation's part
(`_key_memory_back`, fla_kda's wy_dqkg) and the chunk's pairs' part (`_key_pairs_back`, fla_kda's intra),
which XLA sums over a key head's value heads and cumulates over each chunk (`gradients`)."""

import dataclasses
import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, product
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.operands import Issuer, Parts
from dew.nn.kernels.templates.recurrence.mainloop import (
    CHUNK,
    FAST_PARTS,
    Call,
    Held,
    Operand,
    chunk_liveness,
    column_sums,
    column_sums_scratch,
    column_tile,
    f32,
    fragments,
    held_tile,
    key_block,
    launch_kernel,
    lowered_once,
    masks,
    staged_parts,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
)


def _fp32(call: Call) -> Operand:
    """The form of the launches' fp32 gradients between them: one fp32 part, a row a token."""
    return Operand(call.rows, 1, jnp.dtype(jnp.float32))


def _parted(call: Call) -> Operand:
    """The form a register operand of the pairs' gradients' products takes: fp32 in `FAST_PARTS` bf16 parts
    (a transposed fp32 operand is no warpgroup MMA's, and 16 bits of each, past the TF32 fla_kda's intra
    takes them in)."""
    return Operand(call.rows, FAST_PARTS, jnp.dtype(jnp.float32))


def _value_blocks(form: Operand, ref, row, column, call: Call, smem) -> Callable[[int, jax.Array], None]:
    """A store of a chunk's value block `j` `[C, block]` into `ref` in `form` at `row`, its columns from
    `column` on (`algebra.key_scores_back`'s and `key_values_back`'s `emit`)."""

    def stored(j, value):
        form.store(ref, value, row, CHUNK, column + j * call.block, call.block, family=call.family, smem=smem)

    return stored


@lowered_once
def _key_scores_back(scores, do, corrected, taken, call: Call):
    """`(dAqk [R, HV * C] fp32, dv [R, HV * Dv])`: each chunk and value head's scores' gradient and the
    corrected values' gradient from the chunk's own output (`algebra.key_scores_back`, fla_kda's dAv), the
    latter in `_value_grads`' form, from the intra launch's scores (`call.held`), a program a chunk and value
    head (persistent)."""
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    value_width, family = call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    grads, square = _value_grads(call), _fp32(call)
    blocks = value_width // call.block

    def program(at, s_ref, do_ref, v_ref, taken_ref, ds_ref, dv_ref, *, streams, staging, landing):
        del staging
        n, h = at["chunks"], at["heads"]
        row = n * CHUNK
        views = stream_views(scores=held.views(s_ref, row, CHUNK, h * CHUNK, CHUNK),
                             out=given.views(do_ref, row, CHUNK, h * value_width, value_width),
                             corrected=held.views(v_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            stored = _value_blocks(grads, dv_ref, row, h * value_width, call, landing["row"])
            d_scores = algebra.key_scores_back(
                stream_tile(slots, "scores", held), functools.partial(column_tile, call, slots, "out", given),
                functools.partial(column_tile, call, slots, "corrected", held), blocks, Issuer(call),
                emit=stored)
            square.store(ds_ref, d_scores, row, CHUNK, h * CHUNK, CHUNK, family=family,
                         smem=landing["square"])
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(scores=((CHUNK, CHUNK), held), out=((CHUNK, value_width), given),
                          corrected=((CHUNK, value_width), held))
    # The scores' gradient summed over the value blocks, then held beside each block's local product.
    held_tiles = Held(carried=[f32(CHUNK, CHUNK)], phases=(
        masks(CHUNK, CHUNK) + fragments(call, (given, CHUNK, call.block), (held, CHUNK, call.block)),
        [f32(CHUNK, call.block), *held_tile(grads, CHUNK, call.block),
         *fragments(call, (held, CHUNK, CHUNK), (given, CHUNK, call.block))]))
    return launch_kernel(program, call,
                         out_type=(square.shape(call.heads * CHUNK), grads.shape(call.heads * value_width)),
                         grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=tiles,
                         stages=1,
                         live=chunk_liveness(3), work=Work(flops=4 * CHUNK * CHUNK * value_width, span=1),
                         held=held_tiles,
                         landing={"square": square.landing(CHUNK, CHUNK),
                                  "row": grads.landing(CHUNK, call.block)},
                         persistent=True)(scores, do, corrected, taken)


@lowered_once
def _key_memory_back(q, k, hi, lo, do, corrected, dv, memory, dh, a, beta, taken, call: Call, *,
                     scale: float):
    """`(dq, dk, dg [R, HV * Dk], dA [R, HV * C], dbeta [HV, R])` fp32: each chunk and value head's gradients
    from the memory and the weights' part of the WY representation (`algebra.key_memory_block` a key block a
    step, its system's and write gates' terms carried across them and stored on the last; fla_kda's
    wy_dqkg's), the system's gradient before the values' part and the inverse (`_key_values_back`'s), the
    queries' and keys' each value head's, which `gradients` sums over its key head's. `memory` and `dh` the
    memories each chunk enters and the gradient of the one it leaves (`call.memories`), `dv` the corrected
    values' whole gradient (`_reverse`'s), `a` the inverse less its unit diagonal (`_intra_solve`'s)."""
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    width, value_width, family = call.width, call.value_width, call.family
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    gates, grads, square = call.gates, _value_grads(call), _fp32(call)
    keys = key_block(call, functools.partial(_key_memory_held, call),
                     functools.partial(_key_memory_staged, call))
    steps, blocks = width // keys, value_width // call.block
    staging, landing = _key_memory_staged(call, keys)

    def program(at, q_ref, k_ref, hi_ref, lo_ref, do_ref, v_new_ref, dv_ref, h_ref, dh_ref, a_ref, beta_ref,
                taken_ref, dq_ref, dk_ref, dg_ref, da_ref, db_ref, *, streams, staging, landing):
        squared = landing[_square_landing(keys)]
        weights, scratch = staging
        n, h = at["chunks"], at["heads"]
        row, span, last = n * CHUNK, pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
        keyed, memory_row = h // call.group * width, (n * call.heads + h) * width
        views = {**stream_views(lambda i: (0, 0),
                                out=given.views(do_ref, row, CHUNK, h * value_width, value_width),
                                corrected=held.views(v_new_ref, row, CHUNK, h * value_width, value_width),
                                value_grads=grads.views(dv_ref, row, CHUNK, h * value_width, value_width),
                                lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK)),
                 **stream_views(lambda i: (0, i),
                                queries=given.views(q_ref, row, CHUNK, keyed, width),
                                keys=given.views(k_ref, row, CHUNK, keyed, width),
                                gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                                lows=gates.views(lo_ref, row, CHUNK, h * width, width)),
                 **stream_views(lambda i: (i, 0),
                                entered=memories.views(h_ref, memory_row, width, 0, value_width),
                                leaving=memories.views(dh_ref, memory_row, width, 0, value_width))}
        issue = Issuer(call, {"weights_merge": list(weights)})

        def step(index, slots, carry, accumulators):
            d_system, d_betas = carry
            column = functools.partial(column_tile, call, slots)
            betas = family.rows_of(beta_ref.at[h, span])
            at = pl.ds(h * width + index * keys, keys)
            last_prefix = tuple(family.columns_of(ref.at[last, at]) for ref in (hi_ref, lo_ref))

            def stored(kind, grad):
                ref = {"dq": dq_ref, "dk": dk_ref, "dg": dg_ref}[kind]
                square.store(ref, grad, row, CHUNK, h * width + index * keys, keys, family=family,
                             smem=landing["block"])

            loads = {name: functools.partial(column, name, operand) for name, operand in (
                ("out", given), ("corrected", held), ("value_grads", grads), ("entered", memories),
                ("leaving", memories))}
            d_system, beta_terms = algebra.key_memory_block(
                stream_tile(slots, "queries", given), stream_tile(slots, "keys", given),
                (stream_tile(slots, "gates", gates).value(), stream_tile(slots, "lows", gates).value()),
                last_prefix,
                loads["out"], loads["corrected"], loads["value_grads"], loads["entered"], loads["leaving"],
                blocks, stream_tile(slots, "lower", held), betas, d_system, scale, held.dtype, issue,
                issue.summed, emit=stored, columns=column_sums(call, scratch))
            d_betas = d_betas + beta_terms

            # The weights' terms stored once every key block's are in.
            @pl.when(index == steps - 1)
            def _():
                square.store(da_ref, d_system, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=squared)
                db_ref.at[h, span][...] = d_betas

            return (d_system, d_betas), accumulators

        # The carries laid out as their sums are: splats, Mosaic GPU's layout inference ran out of its
        # search's fuel (kda-lower4-sm90 at 739b21575).
        return mosaic.Walk(views, step, steps=steps,
              carry=(family.as_tile(family.zeros((CHUNK, CHUNK))), family.as_row(family.zeros((CHUNK,)))))

    tiles = stream_blocks(out=((CHUNK, value_width), given), corrected=((CHUNK, value_width), held),
                          value_grads=((CHUNK, value_width), grads), lower=((CHUNK, CHUNK), held),
                          queries=((CHUNK, keys), given), keys=((CHUNK, keys), given),
                          gates=((CHUNK, keys), gates), lows=((CHUNK, keys), gates),
                          entered=((keys, value_width), memories), leaving=((keys, value_width), memories))
    fp32_rows = square.shape(call.heads * width)
    out_type = (fp32_rows, fp32_rows, fp32_rows, square.shape(call.heads * CHUNK),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    products = 2 * CHUNK * keys * value_width * 3 * steps + 2 * 2 * CHUNK * CHUNK * width
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(11),
                         work=Work(flops=products, span=steps), held=_key_memory_held(call, keys),
                         staging=staging,
                         landing=landing)(
        q, k, hi, lo, do, corrected, dv, memory, dh, a, beta, taken)


def _key_memory_staged(call: Call, keys: int) -> tuple:
    """`_key_memory_back`'s staging tiles and landings at key block `keys`: the weights' merge's two sets of
    a key block's, the column sums' scratch where they take the consumer's own
    (`mainloop.column_sums_scratch`); a key block's landing and the system's square."""
    held = dataclasses.replace(call.held, rows=call.rows)
    staging = ([staging_tiles(held, (CHUNK, keys))[0] for _ in range(2)], column_sums_scratch(call))
    return staging, _landings(call, keys)


@lowered_once
def _key_values_back(dv, v, a, beta, d_system, taken, call: Call):
    """`(dv [R, HV * Dv], dA [R, HV * C], dbeta [HV, R])` fp32: each chunk and value head's values' part of
    the WY representation's gradient and its system's through the inverse (`algebra.key_values_back`,
    `algebra.key_system_back`; fla_kda's wy_dqkg's last), from the corrected values' whole gradient `dv`
    (`_reverse`'s), the inverse less its unit diagonal `a` and the weights' part of the system's gradient
    `d_system` (`_key_memory_back`'s), a program a chunk and value head (persistent). Apart from the memory
    launch's walk: its last step's, beside the walk's carries, spilled 516 of its 720 bytes a thread on an
    H100 (kda-ablate at 7d5cd5447)."""
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    value_width, family = call.value_width, call.family
    held, given, grads, square = dataclasses.replace(call.held, rows=call.rows), call.given, _value_grads(
        call), _fp32(call)
    blocks = value_width // call.block

    def program(at, dv_ref, v_ref, a_ref, beta_ref, system_ref, taken_ref, dv_out_ref, da_ref, db_ref, *,
                streams, staging, landing):
        values, system = staging
        n, h = at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        views = stream_views(value_grads=grads.views(dv_ref, row, CHUNK, h * value_width, value_width),
                             values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                             lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                             system=square.views(system_ref, row, CHUNK, h * CHUNK, CHUNK))

        def step(index, slots, carry, accumulators):
            del index
            column = functools.partial(column_tile, call, slots)
            lower = stream_tile(slots, "lower", held)
            issue = Issuer(call, {"values_merge": values, "system_merge": system})

            stored = _value_blocks(square, dv_out_ref, row, h * value_width, call, landing["values"])
            summed, beta_terms = algebra.key_values_back(
                functools.partial(column, "value_grads", grads), functools.partial(column, "values", given),
                blocks, lower, family.rows_of(beta_ref.at[h, span]), held.dtype, issue, issue.summed,
                emit=stored)
            d_system = stream_tile(slots, "system", square).value() + summed
            square.store(da_ref, algebra.key_system_back(d_system, lower, held.dtype, issue), row, CHUNK,
                         h * CHUNK, CHUNK, family=family, smem=landing["square"])
            db_ref.at[h, span][...] = beta_terms
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(value_grads=((CHUNK, value_width), grads), values=((CHUNK, value_width), given),
                          lower=((CHUNK, CHUNK), held), system=((CHUNK, CHUNK), square))
    out_type = (square.shape(call.heads * value_width), square.shape(call.heads * CHUNK),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    # The values' two products a value block, then the system's two.
    work = Work(flops=2 * 2 * CHUNK * CHUNK * value_width + 2 * 2 * CHUNK ** 3, span=1)
    landing = {"values": square.landing(CHUNK, call.block), "square": square.landing(CHUNK, CHUNK)}
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(5),
                         work=work,
                         held=_key_values_held(call),
                         staging=(staging_tiles(held, (CHUNK, call.block))[0],
                                  staging_tiles(held, (CHUNK, CHUNK))[0]),
                         landing=landing, persistent=True)(
        dv, v, a, beta, d_system, taken)


def _key_values_held(call: Call) -> Held:
    """`_key_values_back`'s consumer thread: the system's gradient summed over the value blocks beside a
    value block's values, their gradient and the gated values' (`algebra.key_values_back`); then the system's
    gradient and its two products through the inverse, their mask and a product's narrow operand
    (`algebra.key_system_back`)."""
    held = dataclasses.replace(call.held, rows=call.rows)
    return Held(phases=(
        [f32(CHUNK, CHUNK)] * 2 + [f32(CHUNK, call.block)] * 3 + held_tile(held, CHUNK, call.block)
        + fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, call.block)),
        [f32(CHUNK, CHUNK)] * 3 + masks(CHUNK, CHUNK) + held_tile(held, CHUNK, CHUNK)
        + fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, CHUNK))))


def _key_memory_held(call: Call, keys: int) -> Held:
    """`_key_memory_back`'s consumer thread at key block `keys`: the system's gradient and the write gates'
    carried; a key block's three memory products' sums beside a value block's fragments; then the key block's
    decays, gated keys and their WY gradient beside the log decay's terms gathered so far, the system's
    product (`algebra.key_memory_block`'s 'weights_merge') and its operands' fragments."""
    held, given, memories = dataclasses.replace(call.held, rows=call.rows), call.given, call.memories
    return Held(carried=[f32(CHUNK, CHUNK), f32(CHUNK, 1)], phases=(
        [f32(CHUNK, keys)] * 3
        + fragments(call, (given, CHUNK, call.block), (memories, keys, call.block)),
        [f32(CHUNK, keys)] * 7 + held_tile(held, CHUNK, keys) * 2 + [f32(CHUNK, CHUNK)]
        + fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, keys))))


def _square_landing(keys: int) -> str:
    """The landing a launch stores a chunk's `[C, C]` fp32 tile through: its key blocks' where they are as
    wide as a chunk, else its own (`_landings`)."""
    return "block" if keys == CHUNK else "square"


def _landings(call: Call, keys: int) -> dict:
    """A launch's landing tiles for its fp32 stores: a key block's `[C, keys]`, and a chunk's `[C, C]` where
    it is not one (`_square_landing`)."""
    square = _fp32(call)
    return {"block": square.landing(CHUNK, keys),
            **({} if keys == CHUNK else {"square": square.landing(CHUNK, CHUNK)})}


@lowered_once
def _key_pairs_back(q, k, hi, lo, beta, d_scores, d_system, taken, call: Call, *, scale: float):
    """`(dq, dk, dg [R, HV * Dk], dbeta [HV, R])` fp32: each chunk and value head's gradients through its
    pairs, the scores' `scale (q . k) exp2(g_q - g_k)` and the keys' system's `beta (k . k) exp2(..)`
    (`algebra.key_decayed_pairs_back` a key block a step, the write gates' terms carried; fla_kda's intra),
    given their gradients `d_scores` (`_key_scores_back`'s) and `d_system` (`_key_memory_back`'s), staged
    once a step in `_parted`'s form. The queries' and keys' each value head's, which `gradients` sums."""
    width, family = call.width, call.family
    given, gates, square, parted = call.given, call.gates, _fp32(call), _parted(call)
    keys = key_block(call, functools.partial(_key_pairs_held, call),
                     functools.partial(_key_pairs_staged, call))
    steps = width // keys

    def program(at, q_ref, k_ref, hi_ref, lo_ref, beta_ref, ds_ref, da_ref, taken_ref, dq_ref, dk_ref, dg_ref,
                db_ref, *, streams, staging, landing):
        pairs, squares, gated_rows, referenced, *padded = staging
        n, h = at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = h // call.group * width
        views = {**stream_views(lambda i: (0, 0), scores=square.views(ds_ref, row, CHUNK, h * CHUNK, CHUNK),
                                system=square.views(da_ref, row, CHUNK, h * CHUNK, CHUNK)),
                 **stream_views(lambda i: (0, i),
                                queries=given.views(q_ref, row, CHUNK, keyed, width),
                                keys=given.views(k_ref, row, CHUNK, keyed, width),
                                gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                                lows=gates.views(lo_ref, row, CHUNK, h * width, width))}

        def step(index, slots, d_betas, accumulators):
            # Every slot acquired in the step's own trace: the rolled sub-chunks read them.
            slots.acquire_all()
            betas = family.rows_of(beta_ref.at[h, span])
            at = pl.ds(h * width + index * keys, keys)
            # The sub-chunks' first tokens' prefix halves read once a step into a plain tile, a row each, as
            # the intra launch reads its own (`mainloop._intra`).
            for half, ref in enumerate((hi_ref, lo_ref)):
                for sub in range(CHUNK // algebra.SUB):
                    referenced[half, sub, :] = family.columns_of(ref.at[row + sub * algebra.SUB, at])

            def reference(sub, c):
                del c
                return tuple(family.columns_of(referenced.at[half, sub]) for half in (0, 1))

            # Each pairs' gradient staged once in the step, its parts read by every product as they are and
            # transposed; the scores' scaled as their pairs were.
            d_pairs, d_loads = [], []
            for (name, scaled), buffers in zip((("scores", scale), ("system", 1.0)), squares, strict=True):
                value = stream_tile(slots, name, square).value() * scaled
                d_loads.append(functools.partial(
                    lambda name, scaled: stream_tile(slots, name, square).value() * scaled, name, scaled))
                d_pairs.append(Parts([family.staged(part, buffer)
                                      for part, buffer in zip(product.formed(value, parted.parts), buffers,
                                                              strict=True)], jnp.float32))
            query = stream_tile(slots, "queries", given).value().astype(jnp.float32)
            key = stream_tile(slots, "keys", given).value().astype(jnp.float32)

            def gated(c):
                return stream_tile(slots, "gates", gates).value(), stream_tile(slots, "lows", gates).value()

            def shifted(c):
                """The key block's keys and separated prefix staged below `SUB` rows of any values, each
                offset's reads its loads shifted by it (`algebra.Shifted`)."""
                behind, _ = padded
                for i, value in enumerate((key, *algebra.separated(*gated(c)))):
                    behind.at[i, pl.ds(algebra.SUB, CHUNK)][...] = value
                return lambda e: tuple(family.tile_of(behind.at[i, pl.ds(algebra.SUB - e, CHUNK)])
                                       for i in range(3))

            def lower(e, value):
                """`value`'s row `i` at row `i - e`: staged above `SUB` rows of any values, read shifted."""
                _, ahead = padded
                ahead.at[pl.ds(0, CHUNK)][...] = value
                return family.tile_of(ahead.at[pl.ds(e, CHUNK)])

            staged_rows = iter(gated_rows)

            def staged(value):
                """A pairs' rows gated down staged once a key block in `_parted`'s form, its own set."""
                return staged_parts(family, value, next(staged_rows))

            found = {}
            algebra.key_decayed_pairs_back(
                d_pairs, [lambda c: query, lambda c: key * betas[:, None]], lambda c: key, 1, gated,
                reference, Issuer(call, {"pairs": list(pairs)}).summed, staged, chunk=CHUNK,
                emit=lambda kind, p, c, grad: found.__setitem__((kind, p), grad),
                **({} if call.factored else {"shifted": shifted, "lower": lower, "d_loads": d_loads}))
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

        return mosaic.Walk(views, step, steps=steps, carry=family.as_row(family.zeros((CHUNK,))))

    tiles = stream_blocks(scores=((CHUNK, CHUNK), square), system=((CHUNK, CHUNK), square),
                          queries=((CHUNK, keys), given), keys=((CHUNK, keys), given),
                          gates=((CHUNK, keys), gates), lows=((CHUNK, keys), gates))
    staging, landing = _key_pairs_staged(call, keys)
    fp32_rows = square.shape(call.heads * width)
    out_type = (fp32_rows, fp32_rows, fp32_rows, jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    products = 2 * 2 * 2 * CHUNK * CHUNK * keys * (CHUNK // algebra.SUB) * steps
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(7),
                         work=Work(flops=products, span=steps,
                                   exponentials=(CHUNK // algebra.SUB + 1) * CHUNK * width),
                         held=_key_pairs_held(call, keys), staging=staging, landing=landing)(
        q, k, hi, lo, beta, d_scores, d_system, taken)


def _key_pairs_staged(call: Call, keys: int) -> tuple:
    """`_key_pairs_back`'s staging tiles and landings at key block `keys`: two sets of a key block's (the
    products' own), two of a chunk's square (the pairs' gradients) and two of a key block's (the pairs' rows
    gated down, staged once a key block) in `_parted`'s form, and the sub-chunks' references' plain tile; a
    key block's landing."""
    parted = _parted(call)
    # Where the call's pairs do not factor (`Call.factored`), a key block's keys and separated prefix below a
    # sub-chunk's rows and a tile above them (`algebra.key_diagonal_pairs_back`'s shifted reads).
    padded = () if call.factored else (plgpu.SMEM((3, algebra.SUB + CHUNK, keys), jnp.float32),
                                       plgpu.SMEM((CHUNK + algebra.SUB, keys), jnp.float32))
    staging = ([staging_tiles(parted, (CHUNK, keys))[0] for _ in range(2)],
               [staging_tiles(parted, (CHUNK, CHUNK))[0] for _ in range(2)],
               [staging_tiles(parted, (CHUNK, keys))[0] for _ in range(2)],
               plgpu.SMEM((2, CHUNK // algebra.SUB, keys), jnp.float32), *padded)
    return staging, {"block": _fp32(call).landing(CHUNK, keys)}


def _key_pairs_held(call: Call, keys: int) -> Held:
    """`_key_pairs_back`'s consumer thread at key block `keys`: the write gates' terms carried; across the
    rolled sub-chunks (`algebra.key_decayed_pairs_back`) a key block's prefix halves, its rows' decays to
    their common points, its queries and keys, the rows' and keys' gradients carried and the rows' indices;
    in a sub-chunk's turn its keys' gate up, the keys raised, a product's sum and the operand its product
    forms, its parts (`product.formed`) and their remainder, their mask, beside a product's fragments."""
    parted = _parted(call)
    return Held(carried=[f32(CHUNK, 1)], phases=(
        [f32(CHUNK, keys)] * 8 + masks(CHUNK, keys) + [f32(CHUNK, keys)] * 5 + masks(CHUNK, keys)
        + held_tile(parted, CHUNK, keys) + fragments(call, (parted, CHUNK, CHUNK), (parted, CHUNK, keys)),))


class KeyKept(NamedTuple):
    """Each launch of a decay a key dimension's backward's outputs as it stores them (`gradients`), nothing
    computed again for them: what check 2's witness holds each launch to (tests/recurrence_witness.py)."""

    run: object
    """The prepared call (`chunk_forward.prepared`): its laid operands, intra scores, inverse, gated keys,
    corrected values and stored memories."""
    d_scores: jax.Array
    local: jax.Array
    """`_key_scores_back`'s scores' gradient and corrected values' gradient from each chunk's own output."""
    dh: jax.Array
    dv: jax.Array
    dh0: jax.Array
    decayed: jax.Array
    """`_reverse`'s gradient of the memory each chunk leaves, corrected values' and initial memory's
    gradients, and its memory decay's terms."""
    dq_memory: jax.Array
    dk_memory: jax.Array
    dg_memory: jax.Array
    d_weights: jax.Array
    dbeta_memory: jax.Array
    """`_key_memory_back`'s queries', keys' and log decay's gradients, the weights' and beta's."""
    values: jax.Array
    d_system: jax.Array
    dbeta_values: jax.Array
    """`_key_values_back`'s values' gradient, the system's and beta's."""
    dq_pairs: jax.Array
    dk_pairs: jax.Array
    dg_pairs: jax.Array
    dbeta_pairs: jax.Array
    """`_key_pairs_back`'s queries', keys', log decay's and beta's gradients through the pairs."""


def gradients(run, laid_do, dht, tiles) -> tuple[tuple, KeyKept]:
    """`(dq [R, H * Dk], dk [R, H * Dk], dv [R, HV * Dv], dg [R, HV * Dk], dbeta [R, HV], dh0 [B * HV * Dk,
    Dv])` fp32 of the gated delta rule under a decay a key dimension, laid as the prepared run's
    (`chunk_forward.prepared`'s, its intra launch's scores and inverse, its gated keys), fla_kda's DAG: the
    local part, the memory's reverse walk on the queries gated by their decays to the chunk's start and scaled
    (XLA's, `algebra.key_gated`'s), the memory's and WY part and the pairs' part, summed; the log decay's
    gradient each token's terms, the memory decay's on each chunk's last token (`_reverse`'s fourth),
    cumulated from the chunk's end (its prefix is the chunk's own), a key head's value heads' queries' and
    keys' gradients summed. Returns them with each launch's outputs as stored (`KeyKept`)."""
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _reverse

    call = run.call
    rows, heads, width, group = call.rows, call.heads, call.width, call.group
    held = dataclasses.replace(call.held, rows=rows)
    query = call.given.joined(run.q) if call.given.parts > 1 else run.q
    query = jnp.repeat(query.reshape(rows, heads // group, width), group, axis=1).reshape(rows, heads * width)
    gated = held.split(algebra.key_gated(query, (run.hi, run.lo), run.scale))
    d_scores, local = _key_scores_back(run.scores, laid_do, run.corrected, run.taken, call)
    dh, dv2, dh0, decayed = _reverse(gated, run.keys, run.w, laid_do, local, run.hi, run.lo, dht,
                                     run.sequences, None, None, call, tiles, scale=run.scale, memory=run.h)
    dq_m, dk_m, dg_m, d_weights, dbeta_m = _key_memory_back(
        run.q, run.k, run.hi, run.lo, laid_do, run.corrected, dv2, run.h, dh, run.a, run.beta, run.taken,
        call, scale=run.scale)
    dv, d_system, dbeta_v = _key_values_back(dv2, run.v, run.a, run.beta, d_weights, run.taken, call)
    dq_p, dk_p, dg_p, dbeta_p = _key_pairs_back(run.q, run.k, run.hi, run.lo, run.beta, d_scores, d_system,
                                                run.taken, call, scale=run.scale)
    terms = (dg_m + dg_p).reshape(call.chunks, CHUNK, heads * width)
    # The memory decay's term on each chunk's last token, `sum(h dh) exp2(g_last)`, the reverse walk's value
    # blocks' sums added.
    last = tuple(x.reshape(call.chunks, CHUNK, heads * width)[:, -1] for x in (run.hi, run.lo))
    terms = terms.at[:, -1].add(decayed.sum(0).reshape(call.chunks, -1) * algebra.raised(*last))
    dg = jnp.flip(jnp.cumsum(jnp.flip(terms, 1), 1), 1).reshape(rows, heads * width)

    def keyed(x):
        """A value head's gradient `[R, HV * Dk]` summed onto its key head's `[R, H * Dk]`."""
        return x.reshape(rows, heads // group, group, width).sum(2).reshape(rows, -1)

    kept = KeyKept(run, d_scores, local, dh, dv2, dh0, decayed, dq_m, dk_m, dg_m, d_weights, dbeta_m, dv,
                   d_system, dbeta_v, dq_p, dk_p, dg_p, dbeta_p)
    return (keyed(dq_m + dq_p), keyed(dk_m + dk_p), dv, dg, (dbeta_m + dbeta_v + dbeta_p).T, dh0), kept
