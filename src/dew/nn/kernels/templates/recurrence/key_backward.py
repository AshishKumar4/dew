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

from dew.nn.kernels import recurrence_algebra as algebra, tables
from dew.nn.kernels.templates import mosaic, product, resources, swizzling
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.operands import Issuer, Parts
from dew.nn.kernels.templates.recurrence import own_pairs
from dew.nn.kernels.templates.recurrence.mainloop import (
    FAST_PARTS,
    Call,
    Held,
    Operand,
    Walked,
    chunk_liveness,
    column_sums,
    column_sums_scratch,
    column_tile,
    f32,
    fp32_operand,
    held_tile,
    key_block,
    launch_kernel,
    lowered_once,
    masks,
    product_fragments,
    product_results,
    staged_parts,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
)
from dew.nn.kernels.templates.resources import largest_live_set


def _parted(call: Call) -> Operand:
    """The form a register operand of the pairs' gradients' products takes: fp32 in `FAST_PARTS` bf16 parts
    (a transposed fp32 operand is no warpgroup MMA's, and 16 bits of each, past the TF32 fla_kda's intra
    takes them in)."""
    return Operand(call.rows, FAST_PARTS, jnp.dtype(jnp.float32))


def _value_blocks(form: Operand, ref, row, column, call: Call, smem) -> Callable[[int, jax.Array], None]:
    """A store of a chunk's value tile `j` `[C, Call.value_block]` into `ref` in `form` at `row`, its columns
    from `column` on (`algebra.key_scores_back`'s and `key_values_back`'s `emit`)."""
    vb = call.value_block

    def stored(j, value):
        form.store(ref, value, row, call.chunk, column + j * vb, vb, family=call.family, smem=smem)

    return stored


@lowered_once
def _key_scores_back(scores, do, corrected, taken, call: Call):
    """`(dAqk [R, HV * C] fp32, dv [R, HV * Dv])`: each chunk and value head's scores' gradient and the
    corrected values' gradient from the chunk's own output (`algebra.key_scores_back`, fla_kda's dAv), the
    latter in `_value_grads`' form, from the intra launch's scores (`call.held`), a program a chunk and value
    head (persistent)."""
    chunk = call.chunk
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    value_width, family = call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    grads, square = _value_grads(call), fp32_operand(call)
    vb = call.value_block
    blocks = value_width // vb

    def program(at, s_ref, do_ref, v_ref, taken_ref, ds_ref, dv_ref, *, streams, staging, landing):
        del staging
        n, h = at["chunks"], at["heads"]
        row = n * chunk
        views = stream_views(scores=held.views(s_ref, row, chunk, h * chunk, chunk),
                             out=given.views(do_ref, row, chunk, h * value_width, value_width),
                             corrected=held.views(v_ref, row, chunk, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            stored = _value_blocks(grads, dv_ref, row, h * value_width, call, landing["row"])
            d_scores = algebra.key_scores_back(
                stream_tile(slots, "scores", held),
                functools.partial(column_tile, call, slots, "out", given, width=vb),
                functools.partial(column_tile, call, slots, "corrected", held, width=vb), blocks,
                Issuer(call), emit=stored)
            square.store(ds_ref, d_scores, row, chunk, h * chunk, chunk, family=family,
                         smem=landing["square"])
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(scores=((chunk, chunk), held), out=((chunk, value_width), given),
                          corrected=((chunk, value_width), held))
    # The scores' gradient summed over the value blocks, then held beside each block's local product.
    # The local products' results beside the scores' sum: three `[C, C]` warpgroup accumulators live at once
    # (jsf-three-back: 90 registers against an H100's need of 112), a tensor memory read's beside them, and
    # mma.sync's two (one stated, an A100 claimed 3 programs at 152 registers, whose cap spilled, jsf-owncap).
    accumulators = {"fragments": 2, "registers": 2, "tensor": 3}[product_results(call)]
    # The output's gradient against the corrected values read transposed ('values_ds'); then the scores
    # (read transposed) against it ('local_dv', `algebra.key_scores_back`).
    held_tiles = Held(carried=[f32(chunk, chunk)], phases=(
        masks(chunk, chunk)
        + product_fragments(call, (given, chunk, vb), (held, vb, chunk), rhs_transposed=True),
        [f32(chunk, vb)] * accumulators + held_tile(grads, chunk, vb)
        + product_fragments(call, (held, chunk, chunk), (given, chunk, vb))))
    return launch_kernel(program, call,
                         out_type=(square.shape(call.heads * chunk), grads.shape(call.heads * value_width)),
                         grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=tiles,
                         stages=1,
                         live=chunk_liveness(3), work=Work(flops=4 * chunk * chunk * value_width, span=1),
                         held=held_tiles,
                         landing={"square": square.landing(chunk, chunk),
                                  "row": grads.landing(chunk, vb)},
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
    chunk = call.chunk
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    width, value_width, family = call.width, call.value_width, call.family
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    gates, grads, square = call.gates, _value_grads(call), fp32_operand(call)
    keys = key_block(call, functools.partial(_key_memory_held, call),
                     functools.partial(_key_memory_staged, call))
    steps, blocks = width // keys, value_width // call.value_block
    staging, landing = _key_memory_staged(call, keys)

    def program(at, q_ref, k_ref, hi_ref, lo_ref, do_ref, v_new_ref, dv_ref, h_ref, dh_ref, a_ref, beta_ref,
                taken_ref, dq_ref, dk_ref, dg_ref, da_ref, db_ref, *, streams, staging, landing):
        squared = landing[_square_landing(call, keys)]
        weights, scratch = staging
        n, h = at["chunks"], at["heads"]
        row, span, last = n * chunk, pl.ds(n * chunk, chunk), n * chunk + chunk - 1
        keyed, memory_row = h // call.group * width, (n * call.heads + h) * width
        views = {**stream_views(lambda i: (0, 0),
                                out=given.views(do_ref, row, chunk, h * value_width, value_width),
                                corrected=held.views(v_new_ref, row, chunk, h * value_width, value_width),
                                value_grads=grads.views(dv_ref, row, chunk, h * value_width, value_width),
                                lower=held.views(a_ref, row, chunk, h * chunk, chunk)),
                 **stream_views(lambda i: (0, i),
                                queries=given.views(q_ref, row, chunk, keyed, width),
                                keys=given.views(k_ref, row, chunk, keyed, width),
                                gates=gates.views(hi_ref, row, chunk, h * width, width),
                                lows=gates.views(lo_ref, row, chunk, h * width, width)),
                 **stream_views(lambda i: (i, 0),
                                entered=memories.views(h_ref, memory_row, width, 0, value_width),
                                leaving=memories.views(dh_ref, memory_row, width, 0, value_width))}
        issue = Issuer(call, {"weights_merge": list(weights)})

        def step(index, slots, carry, accumulators):
            d_system, d_betas = carry
            column = functools.partial(column_tile, call, slots, width=call.value_block)
            betas = family.rows_of(beta_ref.at[h, span])
            at = pl.ds(h * width + index * keys, keys)
            last_prefix = tuple(family.columns_of(ref.at[last, at]) for ref in (hi_ref, lo_ref))

            def stored(kind, grad):
                ref = {"dq": dq_ref, "dk": dk_ref, "dg": dg_ref}[kind]
                square.store(ref, grad, row, chunk, h * width + index * keys, keys, family=family,
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
                square.store(da_ref, d_system, row, chunk, h * chunk, chunk, family=family, smem=squared)
                db_ref.at[h, span][...] = d_betas

            return (d_system, d_betas), accumulators

        # The carries laid out as their sums are: splats, Mosaic GPU's layout inference ran out of its
        # search's fuel (kda-lower4-sm90 at 739b21575).
        return mosaic.Walk(views, step, steps=steps,
              carry=(family.as_tile(family.zeros((chunk, chunk))), family.as_row(family.zeros((chunk,)))))

    tiles = stream_blocks(out=((chunk, value_width), given), corrected=((chunk, value_width), held),
                          value_grads=((chunk, value_width), grads), lower=((chunk, chunk), held),
                          queries=((chunk, keys), given), keys=((chunk, keys), given),
                          gates=((chunk, keys), gates), lows=((chunk, keys), gates),
                          entered=((keys, value_width), memories), leaving=((keys, value_width), memories))
    fp32_rows = square.shape(call.heads * width)
    out_type = (fp32_rows, fp32_rows, fp32_rows, square.shape(call.heads * chunk),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    products = 2 * chunk * keys * value_width * 3 * steps + 2 * 2 * chunk * chunk * width
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
    staging = ([staging_tiles(held, (call.chunk, keys))[0] for _ in range(2)], column_sums_scratch(call))
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
    chunk = call.chunk
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    value_width, family = call.value_width, call.family
    held, given, grads, square = dataclasses.replace(call.held, rows=call.rows), call.given, _value_grads(
        call), fp32_operand(call)
    vb = call.value_block
    blocks = value_width // vb

    def program(at, dv_ref, v_ref, a_ref, beta_ref, system_ref, taken_ref, dv_out_ref, da_ref, db_ref, *,
                streams, staging, landing):
        values, system = staging
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        views = stream_views(value_grads=grads.views(dv_ref, row, chunk, h * value_width, value_width),
                             values=given.views(v_ref, row, chunk, h * value_width, value_width),
                             lower=held.views(a_ref, row, chunk, h * chunk, chunk),
                             system=square.views(system_ref, row, chunk, h * chunk, chunk))

        def step(index, slots, carry, accumulators):
            del index
            column = functools.partial(column_tile, call, slots, width=vb)
            lower = stream_tile(slots, "lower", held)
            issue = Issuer(call, {"values_merge": values, "system_merge": system})

            stored = _value_blocks(square, dv_out_ref, row, h * value_width, call, landing["values"])
            summed, beta_terms = algebra.key_values_back(
                functools.partial(column, "value_grads", grads), functools.partial(column, "values", given),
                blocks, lower, family.rows_of(beta_ref.at[h, span]), held.dtype, issue, issue.summed,
                emit=stored)
            d_system = stream_tile(slots, "system", square).value() + summed
            square.store(da_ref, algebra.key_system_back(d_system, lower, held.dtype, issue), row, chunk,
                         h * chunk, chunk, family=family, smem=landing["square"])
            db_ref.at[h, span][...] = beta_terms
            return carry, accumulators

        return mosaic.Walk(views, step)

    tiles = stream_blocks(value_grads=((chunk, value_width), grads), values=((chunk, value_width), given),
                          lower=((chunk, chunk), held), system=((chunk, chunk), square))
    out_type = (square.shape(call.heads * value_width), square.shape(call.heads * chunk),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    # The values' two products a value block, then the system's two.
    work = Work(flops=2 * 2 * chunk * chunk * value_width + 2 * 2 * chunk ** 3, span=1)
    landing = {"values": square.landing(chunk, vb), "square": square.landing(chunk, chunk)}
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(5),
                         work=work,
                         held=_key_values_held(call),
                         staging=(staging_tiles(held, (chunk, vb))[0],
                                  staging_tiles(held, (chunk, chunk))[0]),
                         landing=landing, persistent=True)(
        dv, v, a, beta, d_system, taken)


def _key_values_held(call: Call) -> Held:
    """`_key_values_back`'s consumer thread: the system's gradient summed over the value blocks beside a
    value block's values, their gradient and the gated values' (`algebra.key_values_back`); then the system's
    gradient and its two products through the inverse, their mask and a product's narrow operand
    (`algebra.key_system_back`)."""
    chunk, vb = call.chunk, call.value_block
    held = dataclasses.replace(call.held, rows=call.rows)
    # In turn: the inverse (read transposed) against a value block's gradient, that gradient against the
    # gated values read transposed ('values_merge'); then the system's gradient against the inverse read
    # transposed, the inverse (read transposed) against their product ('system_merge').
    return Held(phases=(
        [f32(chunk, chunk)] * 2 + [f32(chunk, vb)] * 3
        + largest_live_set(
            product_fragments(call, (held, chunk, chunk), (held, chunk, vb)),
            product_fragments(call, (held, chunk, vb), (held, vb, chunk), rhs_transposed=True,
                              rhs_staged=True)),
        [f32(chunk, chunk)] * 3 + masks(chunk, chunk)
        + largest_live_set(
            product_fragments(call, (held, chunk, chunk), (held, chunk, chunk), rhs_transposed=True,
                              lhs_staged=True),
            product_fragments(call, (held, chunk, chunk), (held, chunk, chunk), rhs_staged=True))))


def _key_memory_held(call: Call, keys: int) -> Held:
    """`_key_memory_back`'s consumer thread at key block `keys`: the system's gradient and the write gates'
    carried; a key block's three memory products' sums beside a value block's fragments; then the key block's
    decays, gated keys and their WY gradient beside the log decay's terms gathered so far, the system's
    product (`algebra.key_memory_block`'s 'weights_merge') and its operands' fragments."""
    chunk = call.chunk
    held, given, memories = dataclasses.replace(call.held, rows=call.rows), call.given, call.memories
    # The system's update a second square around the merge where warpgroup accumulators hold it
    # (jsf-three-back: 208 registers against an H100's need of 240).
    merged = 2 if product_results(call) == "registers" else 1
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _value_grads

    vb = call.value_block
    # In turn: the output's gradient, the corrected values and their gradient against the memories entered
    # and leaving, read transposed ('memory_dq', 'memory_dk', 'memory_dw'); then the inverse (read
    # transposed) against the weights' gradient, that against the gated keys read transposed
    # ('weights_merge', `algebra.key_memory_block`).
    memory = largest_live_set(
        *(product_fragments(call, (operand, chunk, vb), (memories, vb, keys), rhs_transposed=True)
          for operand in (given, held, _value_grads(call))))
    merging = largest_live_set(
        product_fragments(call, (held, chunk, chunk), (held, chunk, keys)),
        product_fragments(call, (held, chunk, keys), (held, keys, chunk), rhs_transposed=True))
    return Held(carried=[f32(chunk, chunk), f32(chunk, 1)], phases=(
        [f32(chunk, keys)] * 3 + memory,
        [f32(chunk, keys)] * 7 + held_tile(held, chunk, keys) * 2 + [f32(chunk, chunk)] * merged + merging))


def _square_landing(call: Call, keys: int) -> str:
    """The landing a launch stores a chunk's `[C, C]` fp32 tile through: its key blocks' where they are as
    wide as a chunk, else its own (`_landings`)."""
    return "block" if keys == call.chunk else "square"


def _landings(call: Call, keys: int) -> dict:
    """A launch's landing tiles for its fp32 stores: a key block's `[C, keys]`, and a chunk's `[C, C]` where
    it is not one (`_square_landing`)."""
    chunk = call.chunk
    square = fp32_operand(call)
    return {"block": square.landing(chunk, keys),
            **({} if keys == chunk else {"square": square.landing(chunk, chunk)})}


@lowered_once
def _key_pairs_back(q, k, hi, lo, beta, d_scores, d_system, taken, call: Call, *, scale: float):
    """`(dq, dk, dg [R, HV * Dk], dbeta [HV, R])` fp32: each chunk and value head's gradients through its
    pairs, the scores' `scale (q . k) exp2(g_q - g_k)` and the keys' system's `beta (k . k) exp2(..)`
    (`algebra.key_decayed_pairs_back` a key block a step, the write gates' terms carried; fla_kda's intra),
    given their gradients `d_scores` (`_key_scores_back`'s) and `d_system` (`_key_memory_back`'s), staged
    once a step in `_parted`'s form. The queries' and keys' each value head's, which `gradients` sums."""
    chunk = call.chunk
    width, family = call.width, call.family
    given, gates, square, parted = call.given, call.gates, fp32_operand(call), _parted(call)
    keys = key_block(call, functools.partial(_key_pairs_held, call),
                     functools.partial(_key_pairs_staged, call))
    steps = width // keys

    def program(at, q_ref, k_ref, hi_ref, lo_ref, beta_ref, ds_ref, da_ref, taken_ref, dq_ref, dk_ref, dg_ref,
                db_ref, *, streams, staging, landing):
        pairs, squares, gated_rows, referenced = staging
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        views = own_pairs.pairs_back_views(call, row, h, q_ref, k_ref, hi_ref, lo_ref, ds_ref, da_ref)

        def step(index, slots, d_betas, accumulators):
            # Every slot acquired in the step's own trace: the rolled sub-chunks read them.
            slots.acquire_all()
            betas = family.rows_of(beta_ref.at[h, span])
            at = pl.ds(h * width + index * keys, keys)
            # The sub-chunks' first tokens' prefix halves read once a step into a plain tile, a row each, as
            # the intra launch reads its own (`mainloop._intra`).
            for half, ref in enumerate((hi_ref, lo_ref)):
                for sub in range(chunk // algebra.SUB):
                    referenced[half, sub, :] = family.columns_of(ref.at[row + sub * algebra.SUB, at])

            def reference(sub, c):
                del c
                return tuple(family.columns_of(referenced.at[half, sub]) for half in (0, 1))

            # Each pairs' gradient staged once in the step, its parts read by every product as they are and
            # transposed; the scores' scaled as their pairs were.
            d_pairs = []
            for (name, scaled), buffers in zip((("scores", scale), ("system", 1.0)), squares, strict=True):
                value = stream_tile(slots, name, square).value() * scaled
                d_pairs.append(Parts([family.staged(part, buffer)
                                      for part, buffer in zip(product.formed(value, parted.parts), buffers,
                                                              strict=True)], jnp.float32))
            query = stream_tile(slots, "queries", given).value().astype(jnp.float32)
            key = stream_tile(slots, "keys", given).value().astype(jnp.float32)

            def gated(c):
                return stream_tile(slots, "gates", gates).value(), stream_tile(slots, "lows", gates).value()

            staged_rows = iter(gated_rows)

            def staged(value):
                """A pairs' rows gated down staged once a key block in `_parted`'s form, its own set."""
                return staged_parts(family, value, next(staged_rows))

            found = {}
            algebra.key_decayed_pairs_back(
                d_pairs, [lambda c: query, lambda c: key * betas[:, None]], lambda c: key, 1, gated,
                reference, Issuer(call, {"pairs": list(pairs)}).summed, staged, chunk=chunk,
                emit=lambda kind, p, c, grad: found.__setitem__((kind, p), grad))
            d_query, d_keyed, d_key = found["rows", 0], found["rows", 1], found["keys", 0]
            d_betas = d_betas + own_pairs.pairs_back_stored(
                call, (dq_ref, dk_ref, dg_ref), (row, h * width + index * keys), keys, landing["block"],
                (query, key, betas), (d_query, d_keyed, d_key))

            @pl.when(index == steps - 1)
            def _():
                db_ref.at[h, span][...] = d_betas

            return d_betas, accumulators

        return mosaic.Walk(views, step, steps=steps, carry=family.as_row(family.zeros((chunk,))))

    tiles = stream_blocks(scores=((chunk, chunk), square), system=((chunk, chunk), square),
                          queries=((chunk, keys), given), keys=((chunk, keys), given),
                          gates=((chunk, keys), gates), lows=((chunk, keys), gates))
    staging, landing = _key_pairs_staged(call, keys)
    fp32_rows = square.shape(call.heads * width)
    out_type = (fp32_rows, fp32_rows, fp32_rows, jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    products = 2 * 2 * 2 * chunk * chunk * keys * (chunk // algebra.SUB) * steps
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=chunk_liveness(7),
                         work=Work(flops=products, span=steps,
                                   exponentials=(chunk // algebra.SUB + 1) * chunk * width),
                         held=_key_pairs_held(call, keys), staging=staging, landing=landing)(
        q, k, hi, lo, beta, d_scores, d_system, taken)


def _key_pairs_staged(call: Call, keys: int) -> tuple:
    """`_key_pairs_back`'s staging tiles and landings at key block `keys`: two sets of a key block's (the
    products' own), two of a chunk's square (the pairs' gradients) and two of a key block's (the pairs' rows
    gated down, staged once a key block) in `_parted`'s form, and the sub-chunks' references' plain tile; a
    key block's landing."""
    chunk = call.chunk
    parted = _parted(call)
    # The squares' and gated rows' sub-chunk products contract `algebra.SUB` rows, which tcgen05 issues in
    # whole swizzles: their swizzle fits that depth, where inference's 128 bytes refused the launch on an
    # sm100 ('K must be a multiple of 64, got: 16', jsf-gate100g).
    transforms = swizzling.swizzle_transforms(parted.stored,
                                              swizzling.swizzle_for(parted.stored, algebra.SUB))

    def sub_parts(columns):
        return tuple(plgpu.SMEM((chunk, columns), parted.stored, transforms=transforms)
                     for _ in range(parted.parts))

    staging = ([staging_tiles(parted, (chunk, keys))[0] for _ in range(2)],
               [sub_parts(chunk) for _ in range(2)],
               [sub_parts(keys) for _ in range(2)],
               plgpu.SMEM((2, chunk // algebra.SUB, keys), jnp.float32))
    return staging, {"block": fp32_operand(call).landing(chunk, keys)}


def _key_pairs_held(call: Call, keys: int) -> Held:
    """`_key_pairs_back`'s consumer thread at key block `keys`: the write gates' terms carried; across the
    rolled sub-chunks (`algebra.key_decayed_pairs_back`) a key block's prefix halves, its rows' decays to
    their common points, its queries and keys, the rows' and keys' gradients carried and the rows' indices;
    in a sub-chunk's turn its keys' gate up, the keys raised, a product's sum and the operand its product
    forms, its parts (`product.formed`) and their remainder, their mask, beside a product's fragments."""
    chunk = call.chunk
    parted = _parted(call)
    # As an sm100's lowered peak counts it (row 5298, jsf-sm100-backward): nine tiles a block, their masks,
    # the rows' parts, and a product's fragments with mma.sync's right operand loaded ahead.
    # In turn: a pair's gradient against the keys raised; a sub-chunk's (read transposed) against its gated
    # rows (`algebra.key_decayed_pairs_back`); with the first's right operand loaded ahead.
    ahead = resources.operand_fragments(call.family, chunk, chunk, keys, parted.stored, parted.stored,
                                        lhs_parts=0, rhs_parts=parted.parts)
    return Held(carried=[f32(chunk, 1)], phases=(
        [f32(chunk, keys)] * 9 + masks(chunk, keys) * 2 + held_tile(parted, chunk, keys)
        + largest_live_set(
            product_fragments(call, (parted, chunk, chunk), (parted, chunk, keys)),
            product_fragments(call, (parted, chunk, algebra.SUB), (parted, algebra.SUB, keys)))
        + ahead,))


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
    """`_key_pairs_back`'s and `own_pairs.key_pairs_own`'s queries', keys', log decay's and beta's gradients
    through the pairs, summed."""


def _first_decays(dg, run, dh0, taken=None):
    """`dg` `[R, HV * Dk]` with each sequence's first token's taken from the memory its decay scales
    (`algebra.first_decays`, as fla_kda's port takes it): `<h0, dh0>` for a sequence that takes its memory
    (begins its row), at its first chunk's first row. A forked call's documents each take theirs up at a
    chunk's first token (`table.Table.loads`): `<h, taken>` there, the memory the chunk took up in fp32
    (`_taken_memories`, as the walk took it, not its 16-bit copy the states launch stored) and its gradient
    (`taken`, the reverse walk's)."""
    call = run.call
    walked = tables.unpacked(Walked, lambda i: run.sequences[:, i])
    dg = algebra.first_decays(dg, run.h0, dh0, walked.first * call.chunk, walked.row, walked.takes == 1,
                              walked.count > 0)
    if taken is None:
        return dg
    chunks = jnp.arange(call.chunks)
    took = run.table.loads >= 0
    return algebra.first_decays(dg, _taken_memories(run), taken, chunks * call.chunk, chunks, took, took)


def _taken_memories(run) -> jax.Array:
    """The memory each chunk of a forked call took up `[chunks * HV * Dk, Dv]` fp32, as its walk formed it
    (`table.Taking`): its given or stored slot's (`table.Table.loads`; the forward's stored finals,
    `run.stored`), its entry write added; zero at a chunk that took none up."""
    call, table = run.call, run.table
    slots = call.heads * call.width
    per_row = call.chunks // run.batch

    def by_row(memories):
        return memories.reshape(run.batch, -1, slots, memories.shape[-1])

    held = [] if run.stored is None else [by_row(run.stored)]
    given = by_row(table.given)
    # Slot zero a zero memory, then the given slots and the stored ones, one-based as `loads` counts them.
    slotted = jnp.concatenate([jnp.zeros_like(given[:, :1]), given, *held], axis=1)
    loads = table.loads.reshape(run.batch, per_row)
    memories = jnp.take_along_axis(slotted, jnp.maximum(loads, 0)[..., None, None], axis=1)
    memories = memories.reshape(-1, memories.shape[-1])
    if table.entries is not None:
        memories = memories + table.entries
    return jnp.where((table.loads >= 0).repeat(slots)[:, None], memories, 0.0)


def gradients(run, laid_do, dht, tiles) -> tuple[tuple, KeyKept]:
    """`(dq [R, H * Dk], dk [R, H * Dk], dv [R, HV * Dv], dg [R, HV * Dk], dbeta [R, HV], dh0 [B * HV * Dk,
    Dv])` fp32 of the gated delta rule under a decay a key dimension, laid as the prepared run's
    (`chunk_forward.prepared`'s, its intra launch's scores and inverse, its gated keys), fla_kda's DAG: the
    local part, the memory's reverse walk on the queries gated by their decays to the chunk's start and scaled
    (XLA's, `algebra.key_gated`'s), the memory's and WY part and the pairs' part, summed; the log decay's
    gradient each token's terms, the memory decay's on each chunk's last token (`_reverse`'s fourth),
    cumulated from the chunk's end (its prefix is the chunk's own), a key head's value heads' queries' and
    keys' gradients summed. Returns them, and a forked call's gradient of the memory each chunk takes up
    (`_reverse`'s table output, None for another), with each launch's outputs as stored (`KeyKept`)."""
    from dew.nn.kernels.templates.recurrence.mainloop_backward import _reverse

    call = run.call
    rows, heads, width, group = call.rows, call.heads, call.width, call.group
    held = dataclasses.replace(call.held, rows=rows)
    query = call.given.joined(run.q) if call.given.parts > 1 else run.q
    query = jnp.repeat(query.reshape(rows, heads // group, width), group, axis=1).reshape(rows, heads * width)
    gated = held.split(algebra.key_gated(query, (run.hi, run.lo), run.scale))
    d_scores, local = _key_scores_back(run.scores, laid_do, run.corrected, run.taken, call)
    dh, dv2, dh0, decayed, *taken = _reverse(gated, run.keys, run.w, laid_do, local, run.hi, run.lo, dht,
                                             run.sequences, None, None, call, tiles, scale=run.scale,
                                             memory=run.h, table=run.table)
    taken = taken[0] if taken else None
    dq_m, dk_m, dg_m, d_weights, dbeta_m = _key_memory_back(
        run.q, run.k, run.hi, run.lo, laid_do, run.corrected, dv2, run.h, dh, run.a, run.beta, run.taken,
        call, scale=run.scale)
    dv, d_system, dbeta_v = _key_values_back(dv2, run.v, run.a, run.beta, d_weights, run.taken, call)
    pairs = (run.q, run.k, run.hi, run.lo, run.beta, d_scores, d_system, run.taken, call)
    # The sub-chunks' factored pairs' gradients and their own pairs', two launches (`own_pairs`).
    dq_p, dk_p, dg_p, dbeta_p = (factored + own for factored, own in zip(
        _key_pairs_back(*pairs, scale=run.scale), own_pairs.key_pairs_own(*pairs, scale=run.scale),
        strict=True))
    terms = (dg_m + dg_p).reshape(call.chunks, call.chunk, heads * width)
    # The memory decay's term on each chunk's last token, `sum(h dh) exp2(g_last)`, the reverse walk's value
    # blocks' sums added.
    last = tuple(x.reshape(call.chunks, call.chunk, heads * width)[:, -1] for x in (run.hi, run.lo))
    terms = terms.at[:, -1].add(decayed.sum(0).reshape(call.chunks, -1) * algebra.raised(*last))
    dg = jnp.flip(jnp.cumsum(jnp.flip(terms, 1), 1), 1).reshape(rows, heads * width)
    dg = _first_decays(dg, run, dh0, taken)

    def keyed(x):
        """A value head's gradient `[R, HV * Dk]` summed onto its key head's `[R, H * Dk]`."""
        return x.reshape(rows, heads // group, group, width).sum(2).reshape(rows, -1)

    kept = KeyKept(run, d_scores, local, dh, dv2, dh0, decayed, dq_m, dk_m, dg_m, d_weights, dbeta_m, dv,
                   d_system, dbeta_v, dq_p, dk_p, dg_p, dbeta_p)
    return (keyed(dq_m + dq_p), keyed(dk_m + dk_p), dv, dg, (dbeta_m + dbeta_v + dbeta_p).T, dh0, taken), kept
