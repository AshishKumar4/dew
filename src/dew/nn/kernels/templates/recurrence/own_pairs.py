"""Each sub-chunk's own pairs under a decay a key dimension, launches of their own beside the factored walks
(`intra._intra_walk`, `key_backward._key_pairs_back`): the forward's bands (`algebra.key_diagonal_bands`)
and the backward's transpose (`algebra.key_diagonal_pairs_back`), no products on the tensor cores, as
fla_kda's token-parallel diagonal (chunk_kda_fwd_intra_token_parallel) and its gradient. In the walks, an
H100's Kimi Linear intra launch took 1982 us against 492 and its pairs' backward 2268 against 672, each at 248
registers (M-JS-50d, row 5281)."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, resources, swizzling
from dew.nn.kernels.templates.recurrence.intra import intra_rows, systems
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    chunk_liveness,
    f32,
    fp32_operand,
    held_tile,
    key_block,
    launch_kernel,
    masks,
    stream_blocks,
    stream_tile,
    stream_views,
)


def own_staging(call: Call, keys: int) -> tuple:
    """The plain tile a key block's keys and separated prefix are staged in for its shifted reads
    (`shifted_keys`): `[3, SUB + C, keys]` fp32, each's block below `SUB` rows a shifted read takes in place
    of the tokens before the chunk (masked). Its rows are the block's own width apart: the accumulator's
    layout reads a plain tile only at strides whole tiles of 16 columns divide (a stride of 24 does not lower,
    jsf-own-smem). At a 16-column block a warp's rows alternate two runs of banks, two of its four rows a
    wavefront on one; at 32 or 64 columns every row starts on the same bank, its four rows a wavefront's
    conflict, which `key_block`'s choice (by registers and residency) does not weigh."""
    return (swizzling.plain((3, algebra.SUB + call.chunk, keys), jnp.float32),)


def shifted_keys(call: Call, staged, keys: jax.Array, prefix: tuple):
    """A key block's keys and separated prefix (`algebra.separated` of its halves `prefix`), each offset's
    reads of them shifted by it (`algebra.Shifted`, a sub-chunk a group: the pairs a sub-chunk's own), any
    value where a row's source precedes its sub-chunk (masked): stored once in `staged` (`own_staging`) and
    read at the offset's row. Shifted in registers (`Family.row_shifted`, a lane's two shuffles and a select
    a register for each of the three at each offset), the B200's own pairs ran 350 M warp instructions
    against fla's whole intra's 532 M (M-JS-118)."""
    chunk, family = call.chunk, call.family
    width = keys.shape[1]
    # The last block's shifted reads done before its rows are written again, and the writes seen before
    # this block's reads (`mosaic.synced`).
    mosaic.synced(call.interpret)
    for index, value in enumerate((keys, *algebra.separated(*prefix))):
        plgpu.store(staged.at[index, pl.ds(algebra.SUB, chunk), pl.ds(0, width)], value, optimized=False)
    mosaic.synced(call.interpret)
    return lambda e: tuple(family.tile_of(staged.at[index, pl.ds(algebra.SUB - e, chunk), pl.ds(0, width)])
                           for index in range(3))


def shifted_up(call: Call, e, value: jax.Array) -> jax.Array:
    """`value`'s row `i` at row `i - e` (`algebra.key_diagonal_pairs_back`'s `lower`), in registers as
    `Family.row_shifted`, any value where a row's source leaves its sub-chunk (masked)."""
    return call.family.row_shifted(value, -e, period=algebra.SUB, interpret=call.interpret)


def intra_own(q, k, hi, lo, beta, taken, call: Call, *, scale: float, products: tuple[str, ...]) -> tuple:
    """Each output of the intra walk's (`products`) sub-chunks' own pairs, each its own decay, as their bands
    `[R, HV * SUB]` in the output's form (`algebra.key_diagonal_bands`' form: a row's column `e` its pair `e`
    tokens back, masked as the output takes it, `algebra.band_masked`), which the caller places
    (`algebra.placed_bands`) on the sub-chunks' diagonal blocks the walk's sums leave zero
    (`algebra.key_block_pairs` takes the keys before a sub-chunk alone), so each entry is rounded once and
    their sum is exact (`chunk_forward.prepared`): a launch of their own, no products on the tensor cores, as
    fla_kda's token-parallel diagonal is (chunk_kda_fwd_intra_token_parallel). It walks each chunk's key
    blocks, the keys, queries and the walk's stored prefix halves `hi`, `lo` a block at a time, summing each
    output's bands and storing them at its last step. In the walk, they held an H100's Kimi Linear intra
    launch at 248 registers, 1982 us against 492 (M-JS-50d); placed `[C, C]` at its last step (`banded`'s
    selects an offset at a time), each output wrote its chunk's square for its bands' `SUB` columns."""
    chunk, width, gates, family = call.chunk, call.width, call.gates, call.family
    scored = "scores" in products
    keys = key_block(call, functools.partial(_own_held, call, outputs=len(products)),
                     functools.partial(_own_staged, call, products=products),
                     functools.partial(_own_streamed, call, queries=scored), contracted=True)
    steps = width // keys

    def program(at, q_ref, k_ref, hi_ref, lo_ref, beta_ref, taken_ref, *sums, streams, staging, landing):
        del taken_ref
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        keyed = h // call.group * width
        halves = {name: gates.views(ref, row, chunk, h * width, width)
                  for name, ref in (("highs", hi_ref), ("lows", lo_ref))}
        views = stream_views(lambda i: (0, i), keys=call.given.views(k_ref, row, chunk, keyed, width),
                             **halves, **({"queries": call.given.views(q_ref, row, chunk, keyed, width)}
                                          if scored else {}))
        shifts, = staging

        def step(index, slots, bands, accumulators):
            slots.acquire_all()

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            prefix = tuple(stream_tile(slots, name, gates).value() for name in ("highs", "lows"))
            found = algebra.key_diagonal_bands(
                [intra_rows(call, name, operand, beta_ref.at[h, span]) for name in products],
                shifted_keys(call, shifts, operand("keys"), prefix), algebra.separated(*prefix))
            bands = tuple(band + more for band, more in zip(bands, found, strict=True))

            @pl.when(index == steps - 1)
            def _():
                for name, band, ref in zip(products, bands, sums, strict=True):
                    _band_form(call, name).store(
                        ref, algebra.band_masked(band, strict=name == "system", scale=scale), row, chunk,
                        h * algebra.SUB, algebra.SUB, family=family, smem=landing[name])

            return bands, accumulators

        carry = tuple(family.as_tile(family.zeros((chunk, algebra.SUB))) for _ in products)
        return mosaic.Walk(views, step, steps=steps, carry=carry)

    blocks = _own_streamed(call, keys, queries=scored)
    out_type = tuple(_band_form(call, name).shape(call.heads * algebra.SUB) for name in products)
    # Each output's pairs a key a term, a sub-chunk's offsets: the model's.
    model = len(products) * 3 * chunk * algebra.SUB * width
    staging, landing = _own_staged(call, keys, products=products)
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=blocks, stages=(1, 2),
                         live=chunk_liveness(5), steps=steps, model=model,
                         held=_own_held(call, keys, len(products)),
                         staging=staging, landing=landing, name="recurrence_intra_own", issues=False,
                         swizzles={name: resources.widest_swizzle(dtype, keys)
                                   for name, (_, dtype) in blocks.items()})(q, k, hi, lo, beta, taken)


def _own_streamed(call: Call, keys: int, *, queries: bool) -> dict:
    """`intra_own`'s walked streams at key block `keys`: the keys and the prefix halves a block at a time,
    and the queries where it forms the scores."""
    chunk = call.chunk
    return stream_blocks(keys=((chunk, keys), call.given), highs=((chunk, keys), call.gates),
                         lows=((chunk, keys), call.gates),
                         **({"queries": ((chunk, keys), call.given)} if queries else {}))


def _band_form(call: Call, name: str):
    """The form an output's bands are stored in (`intra_own`'s): the scores' the held operand's, the system's
    `systems`', as the intra walk stores their sums."""
    return dataclasses.replace(call.held, rows=call.rows) if name == "scores" else systems(call)


def _own_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`intra_own`'s staging at key block `keys`: its shifted reads' tile (`own_staging`), and each output's
    bands' landing."""
    return own_staging(call, keys), {name: _band_form(call, name).landing(call.chunk, algebra.SUB)
                                     for name in products}


def _own_held(call: Call, block: int, outputs: int = 1) -> Held:
    """`intra_own`'s consumer thread at key block `block` forming `outputs` outputs' bands, carried across
    the key blocks: an offset's rows, keys and separated prefix as read from their staged tile
    (`shifted_keys`), the row's prefix, the decay and a term, nothing held across the offsets (ptxas's least
    spill-free cap at Kimi Linear's training on sm100, sm90 and sm80 104, 96 and 104, jsf-own-smem2; shifted
    in registers, each offset's sources held across them, 184 on an H100, jsf-ownneed); at the last step one
    output's bands masked and the lanes' index."""
    chunk = call.chunk
    return Held(carried=[f32(chunk, algebra.SUB)] * outputs, phases=(
        [f32(chunk, block)] * (outputs + 8),
        [f32(chunk, algebra.SUB)] * 2))


def pairs_back_views(call: Call, row: int, h, q_ref, k_ref, hi_ref, lo_ref) -> dict:
    """The key-block streams of a key pairs backward's program at chunk `row` and value head `h`
    (`key_pairs_own`'s, `key_backward._key_pairs_back`'s): the queries, keys and the separated prefix's
    halves a key block a step."""
    chunk, width = call.chunk, call.width
    given, gates = call.given, call.gates
    keyed = h // call.group * width
    return stream_views(lambda i: (0, i),
                        queries=given.views(q_ref, row, chunk, keyed, width),
                        keys=given.views(k_ref, row, chunk, keyed, width),
                        gates=gates.views(hi_ref, row, chunk, h * width, width),
                        lows=gates.views(lo_ref, row, chunk, h * width, width))


def pairs_back_stored(call: Call, refs: tuple, at: tuple, keys: int, smem, operands: tuple, grads: tuple):
    """A key block's gradients through a chunk's pairs stored at `at` (its row and column) into `refs`
    (`dq`, `dk`, `dg`): the queries', the keys' through the gated keys (`beta k`) and their own, and the log
    decay's (`q dq + k (beta dk' - dk)`); returns the write gates' terms, each row's `sum(dk' k)`."""
    (query, key, betas), (d_query, d_keyed, d_key), (row, column) = operands, grads, at
    for ref, grad in zip(refs, (d_query, d_keyed * betas[:, None] + d_key,
                                query * d_query + key * (d_keyed * betas[:, None] - d_key)), strict=True):
        fp32_operand(call).store(ref, grad, row, call.chunk, column, keys, family=call.family, smem=smem)
    return jnp.sum(d_keyed * key, axis=1)


def key_pairs_own(q, k, hi, lo, beta, d_scores, d_system, taken, call: Call, *, scale: float):
    """`(dq, dk, dg [R, HV * Dk], dbeta [HV, R])` fp32: each chunk and value head's gradients through its
    sub-chunks' own pairs, each its own decay (`algebra.key_diagonal_pairs_back`), given the pairs'
    gradients `d_scores` and `d_system` `[R, HV * C]`: a launch of their own, no products on the tensor
    cores, as the forward's (`intra_own`), each key block's offsets reading the pairs' gradients' own bands
    (`algebra.own_bands`, taken once from the squares, a column an offset, `algebra.band_strip`) and the keys
    and separated prefix shifted (`shifted_keys`). In `_key_pairs_back`'s walk, they held an H100's Kimi
    Linear launch at 248 registers, 2268 us against 672 (M-JS-50d)."""
    chunk, width, family = call.chunk, call.width, call.family
    given, gates, square = call.given, call.gates, fp32_operand(call)
    # The block a multiprocessor holds most programs of, at its registers and its shared memory: by its
    # registers alone, an sm86's 99 KB held none of its 32-key block's 106 KB (row 5282). It issues no
    # products, so a product's depth steps it (`key_block`'s `contracted`), as the forward's.
    keys = key_block(call, functools.partial(_key_own_held, call), functools.partial(_key_own_staged, call),
                     functools.partial(_key_own_streamed, call), contracted=True)
    steps = width // keys

    def program(at, q_ref, k_ref, hi_ref, lo_ref, beta_ref, ds_ref, da_ref, taken_ref, dq_ref, dk_ref, dg_ref,
                db_ref, *, streams, staging, landing):
        del taken_ref
        shifts, = staging
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        # The pairs' own bands' gradients once, the same tile every step.
        views = {**stream_views(lambda i: (0, 0),
                                scores=square.views(ds_ref, row, chunk, h * algebra.SUB, algebra.SUB),
                                system=square.views(da_ref, row, chunk, h * algebra.SUB, algebra.SUB)),
                 **pairs_back_views(call, row, h, q_ref, k_ref, hi_ref, lo_ref)}

        def step(index, slots, d_betas, accumulators):
            slots.acquire_all()
            betas = family.rows_of(beta_ref.at[h, span])
            query = stream_tile(slots, "queries", given).value().astype(jnp.float32)
            key = stream_tile(slots, "keys", given).value().astype(jnp.float32)
            prefix = stream_tile(slots, "gates", gates).value(), stream_tile(slots, "lows", gates).value()
            # Each pairs' gradient's band read at each offset, the scores' scaled as their pairs were.
            def strip(name, scaled, e):
                return algebra.band_strip(stream_tile(slots, name, square).value(), e) * scaled

            strips = [functools.partial(strip, name, scaled)
                      for name, scaled in (("scores", scale), ("system", 1.0))]
            zero = jnp.zeros(key.shape, jnp.float32)
            def rows(name):
                return stream_tile(slots, name, given).value().astype(jnp.float32)

            (d_query, d_keyed), d_key = algebra.key_diagonal_pairs_back(
                strips,
                [functools.partial(rows, "queries"), lambda: algebra.write_gated(rows("keys"), betas)],
                algebra.separated(*prefix),
                shifted_keys(call, shifts, key, prefix), functools.partial(shifted_up, call),
                (zero, zero), zero)
            d_betas = d_betas + pairs_back_stored(
                call, (dq_ref, dk_ref, dg_ref), (row, h * width + index * keys), keys, landing["block"],
                (query, key, betas), (d_query, d_keyed, d_key))

            @pl.when(index == steps - 1)
            def _():
                db_ref.at[h, span][...] = d_betas

            return d_betas, accumulators

        return mosaic.Walk(views, step, steps=steps, carry=family.as_row(family.zeros((chunk,))))

    tiles = _key_own_streamed(call, keys)
    staging, landing = _key_own_staged(call, keys)
    fp32_rows = square.shape(call.heads * width)
    out_type = (fp32_rows, fp32_rows, fp32_rows, jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    # Each offset's two pairs' terms a key, forward and moved up: the model's.
    terms = 2 * 2 * 3 * chunk * algebra.SUB * width
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=tiles, stages=(1, 2), live=chunk_liveness(7),
                         steps=steps, model=terms,
                         held=_key_own_held(call, keys), staging=staging, landing=landing,
                         name="recurrence_key_pairs_own", issues=False)(
        q, k, hi, lo, beta, *(algebra.own_bands(d, chunk=chunk) for d in (d_scores, d_system)), taken)


def _key_own_streamed(call: Call, keys: int) -> dict:
    """`key_pairs_own`'s streams at key block `keys`: the pairs' gradients' own bands once, the queries,
    keys and prefix halves a block at a time."""
    chunk, given, gates, square = call.chunk, call.given, call.gates, fp32_operand(call)
    bands = ((chunk, algebra.SUB), square)
    return stream_blocks(scores=bands, system=bands,
                         queries=((chunk, keys), given), keys=((chunk, keys), given),
                         gates=((chunk, keys), gates), lows=((chunk, keys), gates))


def _key_own_staged(call: Call, keys: int) -> tuple:
    """`key_pairs_own`'s staging at key block `keys`: its shifted reads' tile (`own_staging`; its moved
    gradients shifted in registers, `shifted_up`), and a key block's landing."""
    return own_staging(call, keys), {"block": fp32_operand(call).landing(call.chunk, keys)}


def _key_own_held(call: Call, keys: int) -> Held:
    """`key_pairs_own`'s consumer thread at key block `keys`, as an sm100's lowered peak counts it (row 5298,
    jsf-sm100-backward): the write gates' terms carried, a row's vector; at an offset's shifted loads the
    three gradients carried, the separated prefix's two tiles, the three shifted reads (`shifted_keys`, from
    their staged tile) and the block's keys and queries kept for the gradients' stores, the queries and keys
    as read, the masks and a row's vector; at its strips' reads eight of those tiles, a pairs' gradient's
    band `[C, SUB]` and its lanes' mask, the queries and keys and three row vectors (ptxas's least
    spill-free cap at Kimi Linear's training 112 on sm100, sm90 and sm80, jsf-own-smem2)."""
    chunk = call.chunk
    tile, given = f32(chunk, keys), held_tile(call.given, chunk, keys)
    row = f32(chunk, 4)
    banded = masks(chunk, keys)
    band = [f32(chunk, algebra.SUB), *masks(chunk, algebra.SUB)]
    return Held(carried=[row], phases=([tile] * 9 + given * 2 + banded + [row],
                                       [tile] * 8 + band + given * 2 + banded + [row] * 3))
