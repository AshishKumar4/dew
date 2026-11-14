"""The recurrence template's output launch (`mainloop`'s list: output): each chunk's output from the memory
it entered and its own pairs (`algebra.chunk_output`, `key_decayed_output`), the memories and corrected
values the states walk wrote read back, on the mainloop's machinery."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, operands, product, resources, swizzling
from dew.nn.kernels.templates.operands import Issuer, Parts
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Tiles,
    addressed,
    chunk_liveness,
    f32,
    held_tile,
    launch_kernel,
    lowered_once,
    masks,
    product_fragments,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
)
from dew.nn.kernels.templates.resources import largest_live_set


def _output_streams(call: Call, bv: int) -> tuple[dict, tuple]:
    """`output_launch`'s streamed blocks and its staging tiles at value blocks `bv` wide: a value block's
    memory entered and values; under a decay a head the queries and keys, the scores staged once for every
    value block's product; under one a key dimension the intra launch's scores, and each key block's gated
    queries staged, their queries and prefix halves read from global memory at the first step alone
    (`_gated_queries`), no stream's slots held for the walk: streamed, the queries' and the two fp32
    prefixes' slots staged Kimi delta attention's output at 147,456 bytes at every value block, past a 99 KB
    part's block (an sm86's, sm89's and sm120's: its call the port's there, jsf-kda-derive)."""
    chunk = call.chunk
    width, held, memories = call.width, call.held, call.memories
    queries = {} if call.per_key else {"queries": ((chunk, width), call.given)}
    if call.per_key:
        given = {"scores": ((chunk, chunk), held)}
        # The local product's staged scores, then each key block's gated queries.
        staging = (staging_tiles(held, (chunk, chunk)) + staging_tiles(call.given, *[(chunk, call.block)]
                                                                       * (width // call.block)),)
    else:
        given = {"keys": ((chunk, width), call.given)}
        staging = staging_tiles(held, (chunk, chunk))
    blocks = stream_blocks(**queries, entered=((width, bv), memories), values=((chunk, bv), held), **given)
    return blocks, staging


def _gated_queries(call: Call, refs: tuple, row, h, scale: float, staged) -> None:
    """Each key block's queries gated by their decays to the chunk's start and scaled (`algebra.key_gated`)
    into its staging tile of `staged`, a decay a key dimension's at the first step: the queries and the
    prefix's halves (`refs`, the launch's queries and its high and low prefixes) read from global memory a key
    block at a time in the accumulator's layout (`Family.tile_of`), each operand's parts summed as its tiles'
    (`Parts.value`), no slot held past the step, and staged as the parts the products read (`operands.staged`:
    an fp32 call's two under the fast forms, whose first alone staged left the second's buffer unwritten)."""
    family, width, chunk = call.family, call.width, call.chunk
    q_ref, hi_ref, lo_ref = refs

    def read(operand, ref, column):
        tiles = [family.tile_of(view) for view in operand.views(ref, row, chunk, column, call.block)]
        if len(tiles) == 1:
            return tiles[0]
        return functools.reduce(jnp.add, (tile.astype(jnp.float32) for tile in tiles))

    for c, tile in enumerate(staged):
        column = c * call.block
        query = read(call.given, q_ref, h // call.group * width + column)
        decays = tuple(read(call.gates, ref, h * width + column) for ref in (hi_ref, lo_ref))
        operands.staged(family, algebra.key_gated(query, decays, scale), tile)


def _output_swizzle(call: Call, bv: int, staging: tuple, landing: dict) -> int:
    """The swizzle `output_launch` stages its keys at under a decay a head (`Kᵀ` read transposed, each load a
    whole swizzle of its depth, `Family.rhs_fragment_depth`): the widest its key rows fill, and where products
    read their operands from shared memory the memory's value-block rows too, whose consumer thread's
    registers fit (`_output_held`, beside its addresses into its `staging` and `landing` tiles), halved down
    to the narrowest Mosaic stages; the narrowest where none fits, which the launch then refuses.
    Nemotron-H 4B's SSD output held a bf16 key block's whole 64-column depth at 128 bytes, 255 registers
    and a spill on an A100; at 64 bytes, 32 columns, it compiled within them (row 2943). Warpgroup MMA reads
    a product's two operands at one transform, and the queries are the left operand of the scores' product
    and the memory's (`Q Kᵀ`, `Q H`): the keys' swizzle is the memory's, which a 32-wide bf16 block's rows
    fill at 64 bytes, and at the keys' 128 Mosaic GPU found no layout for the launch."""
    family = call.family
    swizzle = resources.widest_swizzle(call.given.stored, call.width)
    if not family.operands_in_registers:
        swizzle = min(swizzle, family.right_swizzle(call.memories.stored, bv))
    while swizzle > swizzling.NARROWEST_SWIZZLE_BYTES and (
            _output_held(call, bv, swizzle).registers(family, addressed(staging, landing))
            > family.registers(1)):
        swizzle //= 2
    return swizzle


def _output_held(call: Call, bv: int, swizzle: int | None = None) -> Held:
    """What `output_launch`'s consumer thread holds. Under a decay a head: a value block's phase, its output's
    fp32 sum (the memory's and the local product's, each its own accumulator where products overlap,
    `Family.products_overlap`), its values and one product's fragments (`product_fragments`); and the first
    step's scores' phase, their sum, their pairs' decays and their masks (an fp32 call's split into parts) and
    their product's fragments. The two phases hold at once where the walk steps over value blocks (the scores'
    branch inside its loop); apart where it takes one value block, the scores staged before its products
    issue. So an A100's need was 232, 240, None and 216 registers at value blocks 16, 32, 64 and 128 wide of
    128 values, and None at 128 of 256; an H100's 222 at 64 and within 255 at 128 (Mosaic GPU forming each
    warpgroup product's descriptors at its use, JAX 0.12.0.1.post33; post32's hoisted descriptors held the
    128-wide block past 255, where both phases were stated at once). Under a decay a key dimension the value
    block's phase, and apart from it the first step's: a key block's prefix halves, its decay raised from them
    and its queries widened, read from global memory and staged for every step (`_gated_queries`)."""
    chunk = call.chunk
    held, memories, given, family = call.held, call.memories, call.given, call.family
    products = largest_live_set(
        product_fragments(call, (given, chunk, call.block), (memories, call.block, bv)),
        product_fragments(call, (held, chunk, chunk), (held, chunk, bv)))
    if call.per_key:
        # One accumulator for every product (`Issuer.summed`) beside the values, the intra launch's scores
        # in fp32 and their masked narrow form; apart from it, at the first step, the key blocks' gated
        # queries, staged before any product (an A100's held 64- and 128-wide value blocks at 247 registers,
        # an H100's 128-wide at 248).
        values = [f32(chunk, bv), *held_tile(held, chunk, bv), f32(chunk, chunk),
                  *held_tile(held, chunk, chunk),
                  *products]
        return Held(phases=(values, [f32(chunk, call.block)] * 4))
    # Accurate three-part products each take a fresh accumulator (`Issuer.apart`), the memory's output
    # beside the local product's sums: two stated, an unrounded fp32 write gate's output was admitted where
    # ptxas serialized its warpgroup MMAs (C7512, sm90; armada output-values-fresh-sums). Two parts issue
    # into one accumulator (`Issuer.__call__`).
    fresh = (max(len(product.terms(given.parts, memories.parts)),
                 len(product.terms(held.parts, held.parts)))
             if family.products_overlap and max(given.parts, memories.parts, held.parts) > 2 else 1)
    values = ([f32(chunk, bv)] * (1 + fresh if family.products_overlap else 1)
              + held_tile(held, chunk, bv))
    # An fp32 call's prologue's high and low prefixes, a row's and a column's, held across the first step's
    # envelope where products overlap: unstated, Qwen3.5's fp32 forward claimed 2 programs an H100 at 204
    # registers, whose cap spilled 16 bytes (jsf-owncap).
    split = family.products_overlap and held.parts > 1
    prefixes = [f32(chunk, 4)] * 2 + [f32(32, chunk)] * 2 if split else []
    # A write gate on the keys' side scales the scores' columns (`chunk_scores`' `column_scale`): its vector
    # across a thread's columns, 16 fragments on a 64-wide tile (the lowered liveness of Nemotron-H 4B's SSD
    # output, ssd-evidence5 at 97a10a66b; row 2943).
    gated = [f32(chunk, chunk // 2)] if call.gated_keys else []
    parts = held_tile(held, chunk, chunk) if held.parts > 1 else []
    if family.products_overlap and given.parts > 2:
        # Accurate three-part products (`Issuer.apart`) overlap every term's fresh sum, then the decay and
        # mask apply, then the scores split into parts (`algebra.chunk_scores`), each stage's tiles dead
        # before the next's: one stage at a time. One sum priced accepted an unrounded fp32 write gate whose
        # output ptxas serialized (C7512); every stage at once stated 384 registers where ptxas held an fp32
        # rated scan's output at 248 spilling none, least 240 (the six sums alone; armada
        # recurrence-disjoint-score-proof).
        sums = len(product.terms(given.parts, given.parts))
        stages = ([f32(chunk, chunk)] * sums + gated, [f32(chunk, chunk)] * 3 + masks(chunk, chunk) + gated,
                  [f32(chunk, chunk)] * 2 + parts + gated)
    else:
        stages = ([f32(chunk, chunk)] * 3 + masks(chunk, chunk) + parts + gated,)
    # The keys staged as they arrive, a token a row: `Kᵀ`'s depth its stage's columns, a whole swizzle of it
    # a load (`swizzle`, `_output_swizzle`'s; a 128-byte one's 64 bf16 columns held Nemotron-H 4B's SSD
    # output 48 registers past a model of 16, 255 and a spill on an A100, where it stated 216; row 2943).
    scored = product_fragments(call, (given, chunk, call.block), (given, call.block, chunk),
                               rhs_transposed=True, rhs_swizzle=swizzle)
    if call.value_width // bv == 1:
        return Held(carried=prefixes, phases=(values + products, *(stage + scored for stage in stages)))
    # One product's fragments at a time where both phases hold at once, the products issuing in turn: each
    # phase's own beside the other stated an A100's fp32 call at 288 registers a 16-wide output block, where
    # ptxas needed 248 (tf32-fp32c at 9f82d62c4).
    return Held(carried=prefixes,
                phases=tuple(values + stage + largest_live_set(products, scored) for stage in stages))


@lowered_once
def output_launch(q, k, h, corrected, hi, lo, taken, beta, call: Call, tiles: Tiles, *, scale: float,
                  out_dtype, scores=None, skip=None):
    """Each chunk's output `[R, HV * Dv]` in `out_dtype`, a program a chunk and value head and a step each
    value block: under a decay a head its own pairs' scores taken once, at the first step, and staged for
    every block's (`algebra.chunk_scores`, `chunk_output`), under one a key dimension the intra launch's
    (`scores`, `_intra`; `algebra.key_decayed_output`). Where the write gate takes the keys' side
    (`Call.gated_keys`) the scores' columns take `beta` `[HV, R]` before they are rounded, and a `skip`
    `[HV]` adds `D v` from the values the step holds (there the corrected ones)."""
    chunk = call.chunk
    assert skip is None or call.gated_keys, "a skip reads the values a call whose keys take the gate holds"
    width, value_width, bv = call.width, call.value_width, tiles.output_block
    held, memories, family = dataclasses.replace(call.held, rows=call.rows), call.memories, call.family
    steps = value_width // bv

    def program(at, q_ref, k_ref, h_ref, v_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref,
                beta_ref, *refs, streams, staging, landing):
        *skip_ref, o_ref = refs
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        (staged,) = staging
        if call.per_key:
            given = {"scores": held.views(k_ref, row, chunk, h * chunk, chunk)}
        else:
            given = {name: call.given.views(ref, row, chunk, h // call.group * width, width)
                     for name, ref in (("queries", q_ref), ("keys", k_ref))}
        # Step `i` the `i`-th value block: its columns of the memory entered and of the values.
        entering = memories.views(h_ref, (n * call.heads + h) * width, width, 0, value_width)
        views = {**stream_views(lambda i: (0, 0), **given),
                 **stream_views(lambda i: (0, i), entered=entering,
                                values=held.views(v_ref, row, chunk, h * value_width, value_width))}

        def prologue():
            """A decay a head's prefix as a row's vectors and a column's, from global memory before the walk
            waits on its tiles. Not streamed: four more rings' stream bookkeeping spilled a value block 64
            wide's output (388 bytes on an H100's, 0a261aae0)."""
            return (tuple(family.rows_of(ref.at[h, span]) for ref in (hi_ref, lo_ref)),
                    tuple(family.columns_of(ref.at[h, span]) for ref in (column_hi_ref, column_lo_ref)))

        def step(index, slots, carry, accumulators, loaded=None):
            def block(name, operand, c, *, along_rows=False):
                place = pl.ds(c * call.block, call.block)
                return stream_tile(slots, name, operand,
                                   **({"rows": place} if along_rows else {"columns": place}))

            entered = functools.partial(block, "entered", memories, along_rows=True)
            value = stream_tile(slots, "values", held)
            if call.per_key:
                local, *gated = staged

                # Each key block's gated queries staged once, at the first step, for every value block's
                # product: computed each step, they were live across the walk's steps beside each one's sum.
                @pl.when(index == 0)
                def _():
                    _gated_queries(call, (q_ref, hi_ref, lo_ref), row, h, scale, gated)

                out = algebra.key_decayed_output(
                    lambda c: Parts(gated[c], call.given.dtype), entered, width // call.block,
                    stream_tile(slots, "scores", held).value().astype(jnp.float32), value,
                    Issuer(call, {"local": local}).summed)
            else:
                assert loaded is not None, "a decay a head's prefix loads in the prologue"
                (gate, low), laid = loaded
                # The tiles taken before the first step's branch: a slot is the stream's, every step's.
                queries, keys = ([block(name, call.given, c) for c in range(width // call.block)]
                                 for name in ("queries", "keys"))

                @pl.when(index == 0)
                def _():
                    gated = family.columns_of(beta_ref.at[h, span]) if call.gated_keys else None
                    scored = algebra.chunk_scores(
                        queries.__getitem__, keys.__getitem__, width // call.block, gate, low, held.dtype,
                        Issuer(call), laid=laid, column_scale=gated)
                    operands.staged(family, scored, staged)

                out = algebra.chunk_output(queries.__getitem__, entered, width // call.block, gate, low,
                                           Parts(staged, held.dtype), value, scale, Issuer(call))
            if skip_ref:
                out = out + skip_ref[0][h] * value.value().astype(jnp.float32)
            view = o_ref.at[span, pl.ds(h * value_width + index * bv, bv)]
            family.stored([out.astype(o_ref.dtype)], [view], smem=landing["out"])
            return carry, accumulators

        return mosaic.Walk(views, step, steps=steps, prologue=None if call.per_key else prologue)

    blocks, staging = _output_streams(call, bv)
    out_type = jax.ShapeDtypeStruct((call.rows, call.heads * value_width), out_dtype)
    # The prefix twice, as the solve's.
    # The chunk's scores once (the intra launch's under a decay a key dimension), then each value block's
    # memory and local products: the model's.
    scored = 0 if call.per_key else 2 * chunk * chunk * width
    model = scored + 2 * chunk * (width + chunk) * value_width
    landing = {"out": ((chunk, bv), out_dtype)}
    swizzle = None if call.per_key else _output_swizzle(call, bv, staging, landing)
    keys = None if swizzle is None else {f"keys{p}": swizzle for p in range(call.given.parts)}
    # Its kernel named `recurrence_output` as the profiles and the gate know it (`launch_kernel` names a
    # launch for the function its program is defined in, here `output_launch`).
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         name="recurrence_output", grid_names=("chunks", "heads"), blocks=blocks, stages=1,
                         live=chunk_liveness(8), steps=steps, model=model,
                         held=_output_held(call, bv, swizzle), staging=staging, landing=landing,
                         # A 16-bit call's takes no reduction (an A100's asked for its tiles' 49152 bytes
                         # alone).
                         persistent=True, reduces=held.parts > 1, swizzles=keys)(
        q, k if scores is None else scores, h, corrected, hi, lo, hi, lo, taken, beta,
        *(() if skip is None else (skip.astype(jnp.float32),)))
