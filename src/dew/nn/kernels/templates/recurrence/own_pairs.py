"""Each sub-chunk's own pairs under a decay a key dimension, launches of their own beside the factored walks
(`intra._sub_chunk_walk`, `key_backward._key_pairs_back`): the forward's bands (`algebra.key_diagonal_bands`)
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
from jax.experimental.mosaic.gpu import utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith, memref

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import emission, mosaic, resources, scan, swizzling
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.recurrence.intra import _named, intra_rows, systems
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
    """The plain tile a key block's keys and its tokens' own decays are staged in for their shifted reads
    (`shifted_keys`): `[2, SUB + C, keys]` fp32, each's block below `SUB` rows a shifted read takes in place
    of the tokens before the chunk (masked). Its rows are the block's own width apart: the accumulator's
    layout reads a plain tile only at strides whole tiles of 16 columns divide (a stride of 24 does not lower,
    jsf-own-smem). At a 16-column block a warp's rows alternate two runs of banks, two of its four rows a
    wavefront on one; at 32 or 64 columns every row starts on the same bank, its four rows a wavefront's
    conflict, which `key_block`'s choice (by registers and residency) does not weigh."""
    return (swizzling.plain((2, algebra.SUB + call.chunk, keys), jnp.float32),)


def own_zeroed(call: Call, staged) -> None:
    """`staged`'s (`own_staging`'s) rows below the chunk zeroed, which a shifted read takes for the tokens
    before it and nothing writes: once a program, before its walk, so a row's reads before its sub-chunk are
    finite (`algebra.key_diagonal_pairs_back` masks its strips, not its terms)."""
    for index in range(2):
        below = staged.at[index, pl.ds(0, algebra.SUB)]
        plgpu.store(below, call.family.zeros(below.shape), optimized=False)


def shifted_keys(call: Call, staged, keys: jax.Array, prefix: tuple):
    """A key block's keys and its tokens' own decays (`algebra.token_decays` of its separated prefix, the
    prefix halves `prefix`, each row against the row before, shifted in registers once a block), each
    offset's reads of them shifted by it (a sub-chunk a group: the pairs a sub-chunk's own), any value where a
    row's source precedes its sub-chunk (masked): stored once in `staged` (`own_staging`) and read at the
    offset's row. Shifted in registers at every offset (`Family.row_shifted`, a lane's two shuffles and a
    select a register for each), the B200's own pairs ran 350 M warp instructions against fla's whole
    intra's 532 M (M-JS-118)."""
    chunk, family = call.chunk, call.family
    width = keys.shape[1]
    whole = algebra.separated(*prefix)
    earlier = tuple(family.row_shifted(half, 1, period=algebra.SUB, interpret=call.interpret)
                    for half in whole)
    # The last block's shifted reads done before its rows are written again, and the writes seen before
    # this block's reads (`mosaic.synced`).
    mosaic.synced(call.interpret)
    for index, value in enumerate((keys, algebra.token_decays(whole, earlier))):
        plgpu.store(staged.at[index, pl.ds(algebra.SUB, chunk), pl.ds(0, width)], value, optimized=False)
    mosaic.synced(call.interpret)
    return lambda e: tuple(family.tile_of(staged.at[index, pl.ds(algebra.SUB - e, chunk), pl.ds(0, width)])
                           for index in range(2))


QUARTERS = 4
"""The threads sharing a pair of rows in `own_bands`, each a run of the key block's columns."""


def own_bands(rows, staged, bands, *, unit_keys: bool) -> None:
    """Each output's sub-chunks' own pairs of one key block added into `bands`, a plain fp32 shared tile
    `[outputs, C, SUB]` (`algebra.key_diagonal_bands`' form, a row's column `e` its pair `e` tokens back, an
    entry no thread writes zero), from the outputs' rows `rows` `[outputs, C, keys]` and the keys and the
    tokens' own decays `staged` (`own_staging`'s, a chunk's row `r` at `SUB + r`), a thread's own words:
    each quad of the warpgroup's threads a sub-chunk's rows `p` and `SUB - 1 - p` and a quarter of the
    columns each, one loop of `SUB + 1` steps over both rows' pairs at or before them alone (`p + 1` and
    `SUB - p`, so every thread steps alike), its pair's decay the product of its tokens' own formed back from
    its own token as the token recurrence forms it, each term the row times the key, then times the decay
    (or, `unit_keys`, the row times the decay, then times the key), summed over its columns and then the
    quad's (`utils.shfl_bfly`), the warpgroup met once when done. fla_kda's token-parallel diagonal
    (chunk_kda_fwd_intra_token_parallel) walks the pairs at or before a token alone; the bands a tile in the
    accumulator's layout took every offset of every row, its shifted reads whole tiles and its sums over a
    key block's columns a warp's shuffles, 70% of an emulated GH200's intra_own issues (M-JS-145h)."""
    outputs, chunk, keys = rows.shape
    pairs = outputs * chunk * (algebra.SUB + 1) // 2 * keys
    # Each pair's terms (a multiply and an add an output, a decay's multiply) and its words read.
    body = emission.stating(lambda *refs: Work(operations=(2 * outputs + 1) * pairs,
                                               shared_bytes=4 * 2 * pairs))(
        lambda ctx, *refs: _own_bands(*refs, unit_keys=unit_keys))
    plgpu.inline_mgpu(arg_types=(plgpu.RefType(),) * 3, return_type=())(body)(rows, staged, bands)


def _own_bands(rows_ref, staged_ref, bands_ref, *, unit_keys: bool) -> None:
    """`own_bands` in Mosaic GPU's terms."""
    index, i32, f32_type = ir.IndexType.get(), ir.IntegerType.get_signless(32), ir.F32Type.get()
    outputs, _, keys = ir.MemRefType(rows_ref.type).shape
    sub, run = algebra.SUB, keys // QUARTERS

    def const(value):
        return arith.constant(i32, value)

    def at(*values):
        def indexed(value):
            return (arith.index_cast(index, value) if isinstance(value, ir.Value)
                    else arith.constant(index, value))
        return [indexed(value) for value in values]

    thread = arith.remui(utils.thread_idx(), const(scan.THREADS))
    quad, quarter = arith.divui(thread, const(QUARTERS)), arith.remui(thread, const(QUARTERS))
    pair_rows = sub // 2
    first_row = arith.muli(arith.divui(quad, const(pair_rows)), const(sub))
    p = arith.remui(quad, const(pair_rows))
    short, long = arith.addi(first_row, p), arith.subi(arith.addi(first_row, const(sub - 1)), p)
    columns = [arith.addi(arith.muli(quarter, const(run)), const(m)) for m in range(run)]
    loaded = [[[memref.load(rows_ref, at(o, row, column)) for column in columns] for row in (short, long)]
              for o in range(outputs)]
    one = arith.constant(f32_type, 1.0)
    decays = [one] * run
    decayed = [[one] * run for _ in range(outputs)]
    leader = arith.cmpi(arith.CmpIPredicate.eq, quarter, const(0))
    for step in range(sub + 1):
        on_short = arith.cmpi(arith.CmpIPredicate.sle, const(step), p)
        row = arith.select(on_short, short, long)
        e = arith.select(on_short, const(step), arith.subi(const(step - 1), p))
        first = arith.cmpi(arith.CmpIPredicate.eq, e, const(0))
        # Token `i - e`'s keys, and token `i - e + 1`'s own decay (any of the chunk's at a row's own pair).
        source = arith.subi(arith.addi(row, const(sub)), e)
        own = arith.select(first, source, arith.addi(source, const(1)))
        key = [memref.load(staged_ref, at(0, source, column)) for column in columns]
        factor = [memref.load(staged_ref, at(1, own, column)) for column in columns]
        if not unit_keys:
            # The pair's decay, one for every output, restarting at a row's own pair.
            decays = [arith.select(first, one, arith.mulf(before, a)) for before, a in zip(decays, factor,
                                                                                            strict=True)]
        for o in range(outputs):
            values = [arith.select(on_short, short_value, long_value)
                      for short_value, long_value in zip(*loaded[o], strict=True)]
            if unit_keys:
                decayed[o] = [arith.select(first, value, arith.mulf(before, a))
                              for value, before, a in zip(values, decayed[o], factor, strict=True)]
                terms = [arith.mulf(d, k) for d, k in zip(decayed[o], key, strict=True)]
            else:
                terms = [arith.mulf(arith.mulf(v, k), d) for v, k, d in zip(values, key, decays, strict=True)]
            total = terms[0]
            for term in terms[1:]:
                total = arith.addf(total, term)
            for distance in (1, 2):
                total = arith.addf(total, utils.shfl_bfly(total, distance))
            with utils.when(leader):
                place = at(o, row, e)
                memref.store(arith.addf(memref.load(bands_ref, place), total), bands_ref, place)
    utils.warpgroup_barrier()


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
        shifts, rows, summed = staging

        def step(index, slots, bands, accumulators):
            slots.acquire_all()

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            prefix = tuple(stream_tile(slots, name, gates).value() for name in ("highs", "lows"))
            formed = [intra_rows(call, name, operand, beta_ref.at[h, span]) for name in products]
            shifted = shifted_keys(call, shifts, operand("keys"), prefix)
            if call.interpret:
                found = algebra.key_diagonal_bands(formed, shifted, unit_keys=call.unit_keys)
                bands = tuple(band + more for band, more in zip(bands, found, strict=True))
            else:
                # Natively, each thread its own pairs' words (`own_bands`), the bands summed in shared memory.
                for o, value in enumerate(formed):
                    plgpu.store(rows.at[o], value, optimized=False)
                mosaic.synced(call.interpret)
                own_bands(rows, shifts, summed, unit_keys=call.unit_keys)

            @pl.when(index == steps - 1)
            def _():
                for o, (name, ref) in enumerate(zip(products, sums, strict=True)):
                    band = bands[o] if call.interpret else family.tile_of(summed.at[o])
                    _band_form(call, name).store(
                        ref, algebra.band_masked(band, strict=name == "system", scale=scale), row, chunk,
                        h * algebra.SUB, algebra.SUB, family=family, smem=landing[name])

            return bands, accumulators

        def prologue():
            own_zeroed(call, shifts)
            if not call.interpret:
                for o in range(len(products)):
                    plgpu.store(summed.at[o], family.as_tile(family.zeros((chunk, algebra.SUB))),
                                optimized=False)

        carry = (tuple(family.as_tile(family.zeros((chunk, algebra.SUB))) for _ in products) if call.interpret
                 else ())
        return mosaic.Walk(views, step, steps=steps, carry=carry, prologue=prologue)

    blocks = _own_streamed(call, keys, queries=scored)
    out_type = tuple(_band_form(call, name).shape(call.heads * algebra.SUB) for name in products)
    # Each output's pairs a key a term, a sub-chunk's offsets: the model's.
    model = len(products) * 3 * chunk * algebra.SUB * width
    staging, landing = _own_staged(call, keys, products=products)
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=blocks, stages=(1, 2),
                         live=chunk_liveness(5), steps=steps, model=model,
                         held=_own_held(call, keys, len(products)),
                         staging=staging, landing=landing, name=_named("recurrence_intra_own", products),
                         issues=False,
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
    """`intra_own`'s staging at key block `keys`: its shifted reads' tile (`own_staging`), the outputs' rows
    and their bands summed (`own_bands`'), plain fp32, and each output's bands' landing."""
    outputs, chunk = len(products), call.chunk
    staging = (*own_staging(call, keys), swizzling.plain((outputs, chunk, keys), jnp.float32),
               swizzling.plain((outputs, chunk, algebra.SUB), jnp.float32))
    return staging, {name: _band_form(call, name).landing(chunk, algebra.SUB) for name in products}


def _own_held(call: Call, block: int, outputs: int = 1) -> Held:
    """`intra_own`'s consumer thread at key block `block` forming `outputs` outputs' bands, carried across
    the key blocks: as its staged reads are formed (`shifted_keys`), the separated prefix, its rows shifted,
    the tokens' own decays and the keys; at an offset the rows, the keys and decays read from their staged
    tile, the pair's decay and the decay following it carried across the offsets, and a term's two products
    (ptxas's least spill-free cap at Kimi Linear's training on sm100, sm90 and sm80 104, 96 and 104 where each
    offset read the separated prefix and held the row's, jsf-own-smem2; shifted in registers, each offset's
    sources held across them, 184 on an H100, jsf-ownneed); at the last step one output's bands masked and
    the lanes' index."""
    chunk = call.chunk
    if not call.interpret:
        # Natively the bands summed in shared memory (`own_bands`), a thread holding its two rows' runs of
        # each output, the pair's decay and a step's keys and factors.
        return Held(phases=([f32(chunk, block)] * 6, [f32(chunk, block)] * (outputs + 2),
                            [f32(chunk, algebra.SUB)] * 2))
    return Held(carried=[f32(chunk, algebra.SUB)] * outputs, phases=(
        [f32(chunk, block)] * 6,
        [f32(chunk, block)] * (outputs + 6),
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


def pairs_back_stored(call: Call, refs: tuple, at: tuple, keys: int, smem, operands: tuple, grads: tuple,
                      diagonal=None):
    """A key block's gradients through a chunk's pairs stored at `at` (its row and column) into `refs`
    (`dq`, `dk`, `dg`): the queries', the keys' through the gated keys (`beta k`) and their own, and the log
    decay's (`q dq + k (beta dk' - dk)`); returns the write gates' terms, each row's `sum(dk' k)`. Where
    `diagonal` gives a token's own scores pair's gradient `[C, 1]` (scaled), `grads` were summed without
    it and it joins the queries' and keys' gradients alone: the log decay's terms never take it, a token's
    write read by its own query undecayed, its two halves `q (dS k)` and `k (dS q)` equal, each the size of
    `q k dS` where their difference is none, which summed in left their rounding on every earlier token's
    suffix sum, 4e-10 at Kimi Linear's last token where float64's was 2e-17 (row 7906)."""
    (query, key, betas), (d_query, d_keyed, d_key), (row, column) = operands, grads, at
    own_query, own_key = (0.0, 0.0) if diagonal is None else (diagonal * key, diagonal * query)
    for ref, grad in zip(refs, (d_query + own_query, d_keyed * betas[:, None] + d_key + own_key,
                                query * d_query + key * (d_keyed * betas[:, None] - d_key)), strict=True):
        fp32_operand(call).store(ref, grad, row, call.chunk, column, keys, family=call.family, smem=smem)
    return jnp.sum(d_keyed * key, axis=1)


def key_pairs_own(q, k, hi, lo, beta, d_scores, d_system, taken, call: Call, *, scale: float):
    """`(dq, dk, dg [R, HV * Dk], dbeta [HV, R])` fp32: each chunk and value head's gradients through its
    sub-chunks' own pairs, each its own decay (`algebra.key_diagonal_pairs_back`), given the pairs'
    gradients `d_scores` and `d_system` `[R, HV * C]`: a launch of their own, no products on the tensor
    cores, as the forward's (`intra_own`), each key block's offsets reading the pairs' gradients' own
    bands (`algebra.own_bands`, taken once from the squares, laid an offset a row `[HV * SUB, R]`: each
    offset's strip a row read as a row's vector) and the keys and the tokens' own decays shifted
    (`shifted_keys`). In `_key_pairs_back`'s walk, they held an H100's Kimi Linear launch at 248
    registers, 2268 us against 672 (M-JS-50d)."""
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
        # The pairs' own bands' gradients once, the same tile every step, an offset a row
        # (`algebra.own_bands`).
        views = {**stream_views(lambda i: (0, 0),
                                scores=square.views(ds_ref, h * algebra.SUB, algebra.SUB, row, chunk),
                                system=square.views(da_ref, h * algebra.SUB, algebra.SUB, row, chunk)),
                 **pairs_back_views(call, row, h, q_ref, k_ref, hi_ref, lo_ref)}

        def step(index, slots, d_betas, accumulators):
            slots.acquire_all()
            betas = family.rows_of(beta_ref.at[h, span])
            query = stream_tile(slots, "queries", given).value().astype(jnp.float32)
            key = stream_tile(slots, "keys", given).value().astype(jnp.float32)
            prefix = stream_tile(slots, "gates", gates).value(), stream_tile(slots, "lows", gates).value()
            # Each pairs' gradient's band read at each offset, its row (every token's pair `e` back) as a
            # row's vector, the scores' scaled as their pairs were.
            def strip(name, scaled, e):
                return functools.reduce(jnp.add, (family.rows_of(slots[f"{name}{p}"].at[e])
                                                  for p in range(square.parts))) * scaled

            # The scores' own pair (offset zero) summed apart (`pairs_back_stored`'s `diagonal`): its strip
            # scaled to zero there, a scalar's product (a select against a scalar lowers no broadcast).
            def scores(e):
                return strip("scores", scale, e) * (e != 0).astype(jnp.float32)

            strips = [scores, functools.partial(strip, "system", 1.0)]
            zero = jnp.zeros(key.shape, jnp.float32)
            def rows(name):
                return stream_tile(slots, name, given).value().astype(jnp.float32)

            (d_query, d_keyed), d_key = algebra.key_diagonal_pairs_back(
                strips,
                [functools.partial(rows, "queries"), lambda: algebra.write_gated(rows("keys"), betas)],
                shifted_keys(call, shifts, key, prefix), functools.partial(shifted_up, call),
                (zero, zero), zero)
            d_betas = d_betas + pairs_back_stored(
                call, (dq_ref, dk_ref, dg_ref), (row, h * width + index * keys), keys, landing["block"],
                (query, key, betas), (d_query, d_keyed, d_key), diagonal=strip("scores", scale, 0)[:, None])

            @pl.when(index == steps - 1)
            def _():
                db_ref.at[h, span][...] = d_betas

            return d_betas, accumulators

        return mosaic.Walk(views, step, steps=steps, carry=family.as_row(family.zeros((chunk,))),
                           prologue=lambda: own_zeroed(call, shifts))

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
                         name="recurrence_key_pairs_own", issues=False,
                         # The bands read a row at a time, at a traced offset: staged plain.
                         swizzles={f"{name}{p}": swizzling.UNSWIZZLED for name in ("scores", "system")
                                   for p in range(fp32_operand(call).parts)})(
        q, k, hi, lo, beta, *(algebra.own_bands(d, chunk=chunk) for d in (d_scores, d_system)), taken)


def _key_own_streamed(call: Call, keys: int) -> dict:
    """`key_pairs_own`'s streams at key block `keys`: the pairs' gradients' own bands once, an offset a row
    (`algebra.own_bands`), the queries, keys and prefix halves a block at a time."""
    chunk, given, gates, square = call.chunk, call.given, call.gates, fp32_operand(call)
    bands = ((algebra.SUB, chunk), square)
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
    three gradients carried, the pair's decay and the decay following it, the two shifted reads
    (`shifted_keys`, from their staged tile) and the block's keys and queries kept for the gradients'
    stores, the queries and keys as read, the masks and a row's vector; at its strips' reads seven of those
    tiles, the queries and keys and three row vectors, each strip read from its band's row in shared memory
    (a band `[C, SUB]` and its lanes' mask held where it was taken out of a token's band; ptxas's least
    spill-free cap at Kimi Linear's training 112 on sm100, sm90 and sm80 where each offset read the
    separated prefix and held it, jsf-own-smem2)."""
    chunk = call.chunk
    tile, given = f32(chunk, keys), held_tile(call.given, chunk, keys)
    row = f32(chunk, 4)
    banded = masks(chunk, keys)
    return Held(carried=[row], phases=([tile] * 8 + given * 2 + banded + [row],
                                       [tile] * 7 + given * 2 + banded + [row] * 3))
