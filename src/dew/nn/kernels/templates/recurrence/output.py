"""The recurrence template's output launch (`mainloop`'s list: output): each chunk's output from the memory
it entered and its own pairs (`algebra.chunk_output`, `key_decayed_output`), the memories and corrected
values the states walk wrote read back, on the mainloop's machinery."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, operands, product, resources, swizzling
from dew.nn.kernels.templates.operands import Issuer, Parts
from dew.nn.kernels.templates.recurrence.mainloop import (
    Call,
    Held,
    Tiles,
    addressed,
    building,
    chunk_liveness,
    f32,
    held_tile,
    launch_kernel,
    masks,
    product_fragments,
    staging_tiles,
    stream_blocks,
    stream_tile,
    stream_views,
)
from dew.nn.kernels.templates.resources import largest_live_set
from dew.nn.precision import activation_and_slope, rms_of_total, rms_weighted


@dataclasses.dataclass(frozen=True)
class Normed:
    """The output launch's epilogue (`recurrence.Hooks.output`'s `OutputNorm`): each token's value row of a
    head read back in `read` (its `read_in`, else fp32) through its gated RMS norm, `rms_norm(o) weight
    act(z)` with `epsilon`, `activation` the gate's, the mean square over the head's `width` real values (its
    padding none); with `kept`, the read-back output stored beside it, the norm's backward's residual (a
    training forward's)."""

    epsilon: float
    activation: str
    read: str
    width: int
    kept: bool = False


def _output_streams(call: Call, bv: int) -> tuple[dict, tuple]:
    """`output_launch`'s streamed blocks and its staging tiles at value blocks `bv` wide: a value block's
    memory entered and values; under a decay a head the queries and keys, the scores staged once for every
    value block's product; under one a key dimension the intra launch's scores, and each key block's gated
    queries staged, their queries and prefix read from global memory at the first step alone
    (`_gated_queries`), no stream's slots held for the walk: streamed, the queries' and two fp32 prefixes'
    slots (a compensated prefix's halves) staged Kimi delta attention's output at 147,456 bytes at every
    value block, past a 99 KB part's block (an sm86's, sm89's and sm120's: its call the port's there,
    jsf-kda-derive)."""
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
    prefix (`refs`, the launch's queries and its prefix, `intra._prefix`'s) read from global memory in the
    accumulator's layout (`Family.tile_of`), every key block's before any is staged, each operand's parts
    summed as its tiles' (`Parts.value`), no slot held past the step, and staged as the parts the products
    read (`operands.staged`: an fp32 call's two under the fast forms, whose first alone staged left the
    second's buffer unwritten)."""
    family, width, chunk = call.family, call.width, call.chunk
    q_ref, *prefix_refs = refs

    def read(operand, ref, column):
        tiles = [family.tile_of(view) for view in operand.views(ref, row, chunk, column, call.block)]
        return tiles[0] if len(tiles) == 1 else product.whole(tiles)

    # Every key block's reads issued before any is gated and staged, so their latencies overlap: a block at a
    # time, each waited on its own, 15% of a GH200's Kimi Linear output launch's samples (M-JS-272).
    read_blocks = [(read(call.given, q_ref, h // call.group * width + c * call.block),
                    tuple(read(call.gates, ref, h * width + c * call.block) for ref in prefix_refs))
                   for c in range(len(staged))]
    for (query, decays), tile in zip(read_blocks, staged, strict=True):
        operands.staged(call, algebra.key_gated(query, decays, scale), tile)


def _output_swizzle(call: Call, bv: int, staging: tuple, landing: dict, *, normed: bool = False) -> int:
    """The swizzle `output_launch` stages its keys at under a decay a head (`Kᵀ` read transposed, each load a
    whole swizzle of its depth, `Family.rhs_fragment_depth`): the widest its key rows fill, and where products
    read their operands from shared memory the memory's value-block rows too, whose consumer thread's
    registers fit (`_output_held`, an epilogue's (`normed`) among them, beside its addresses into its
    `staging` and `landing` tiles), halved down
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
            _output_held(call, bv, swizzle, normed=normed).registers(family, addressed(staging, landing))
            > family.registers(1)):
        swizzle //= 2
    return swizzle


def _normed_carried(chunk: int, bv: int, steps: int, normed: bool) -> list:
    """An epilogue's carried tiles (`_normed_step`): its rows' sums of squares, a row's vector, and where the
    row is one value block, the block read back itself."""
    if not normed:
        return []
    return [(chunk, 1, jnp.float32), *([f32(chunk, bv)] if steps == 1 else [])]


def _normed_phases(chunk: int, bv: int, normed: bool) -> tuple:
    """An epilogue's read (`_normed_read`): a block's read-back row, its gate and its normalized values, with
    the weight's column vector."""
    return ([f32(chunk, bv)] * 3 + [(1, bv, jnp.float32)],) if normed else ()


def _output_held(call: Call, bv: int, swizzle: int | None = None, *, normed: bool = False) -> Held:
    """What `output_launch`'s consumer thread holds. Under a decay a head: a value block's phase, its output's
    fp32 sum (the memory's and the local product's, each its own accumulator where products overlap,
    `Family.products_overlap`), its values and one product's fragments (`product_fragments`); and the first
    step's scores' phase, their sum, their pairs' decays and their masks (an fp32 call's split into parts) and
    their product's fragments. The two phases apart: the scores staged in the first step's branch before the
    value block's products issue, whether the walk takes one value block or steps over several. Under a decay
    a key dimension the value block's phase, and apart from it the first step's: a key block's prefix, its
    decay raised from it and its queries widened, read from global memory and staged for every step
    (`_gated_queries`). An epilogue's (`normed`) rows' sums carried across the steps, and at the last its
    phase of a block's read-back row, gate and normalized values (`_normed_step`)."""
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
        # Every key block's queries (their parts joined in fp32) and prefix read before any is gated
        # (`_gated_queries`), one block's gating at a time beside them.
        query = held_tile(given, chunk, call.block) if given.parts == 1 else [f32(chunk, call.block)]
        read = (query + [f32(chunk, call.block)] * call.prefix_limbs) * (call.width // call.block - 1)
        return Held(carried=_normed_carried(chunk, bv, call.value_width // bv, normed),
                    phases=(values, [f32(chunk, call.block)] * (2 + call.prefix_limbs) + read,
                            *_normed_phases(chunk, bv, normed)))
    # Accurate three-part products each take a fresh accumulator (`Issuer.apart`), the memory's output
    # beside the local product's sums: two stated, an unrounded fp32 write gate's output was admitted where
    # ptxas serialized its warpgroup MMAs (C7512, sm90; armada output-values-fresh-sums). Two parts issue
    # into one accumulator a product (`Issuer.__call__`), each key block's memory product its own beside
    # the local product's, all in flight at once (`algebra.chunk_output`): two stated, a 192-wide value
    # head's output stated 248 registers where its three accumulators alone held 288 and ptxas serialized
    # them (C7512, row 7151).
    fresh = (max(len(product.terms(given.parts, memories.parts)),
                 len(product.terms(held.parts, held.parts)))
             if family.products_overlap and max(given.parts, memories.parts, held.parts) > 2 else 1)
    memory = call.width // call.block if fresh == 1 else 1
    values = ([f32(chunk, bv)] * (memory + fresh if family.products_overlap else 1)
              + held_tile(held, chunk, bv))
    # The prologue's high and low prefixes as a row's vectors, read by the first step's scores and by every
    # value block's output (`algebra.chunk_output`'s decay), held across the walk: unstated, a 192-wide value
    # head's output stated 248 registers, its budget, where ptxas took 254 and serialized its warpgroup MMAs
    # (C7512; the resource gate's training forward, FS-GATEFWD-908471231). An fp32 call's, a column's too,
    # across the first step's envelope where products overlap: unstated, Qwen3.5's fp32 forward claimed 2
    # programs an H100 at 204 registers, whose cap spilled 16 bytes (jsf-owncap).
    split = family.products_overlap and held.parts > 1
    prefixes = [f32(chunk, 4)] * 2 + ([f32(32, chunk)] * 2 if split else [])
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
    # The phases apart at every output width: the first step's scores staged (its branch) before the value
    # block's products issue, the step's order whether it walks one value block or many. Joined for a walk
    # of several, an A100's bf16 output with no delta rule was stated 304 registers at a 32-wide block and
    # 272 at 16 and taken at neither, where ptxas held them at 255 and 238 spilling none (row 6233); its
    # compile certifies what this states (`tuning`'s spill refusal).
    return Held(carried=prefixes + _normed_carried(chunk, bv, call.value_width // bv, normed),
                phases=(values + products, *(stage + scored for stage in stages),
                        *_normed_phases(chunk, bv, normed)))


def _normed_step(normed: Normed, family, out, index, carry, *, steps: int, bv: int, columns, refs: tuple,
                 rows, landing) -> tuple:
    """The epilogue's step (`Normed`): the value block's output `out` read back and its squares summed into
    each row's sums (`carry`'s first), the block held for the walk's read (`_normed_read`): carried itself
    where the row is one block, else in the rows' own shared tile (`rows`, a head's whole value rows, staged
    apart from every store's landing); where the norm's residual is kept, the read-back block stored
    (`refs`' last, `_output_refs`). `columns(c)` is block `c`'s columns of the outputs' `[R, HV * Dv]`
    rows. Returns the carry the next step takes."""
    kept_ref = refs[3]
    read = out.astype(normed.read)
    wide = read.astype(jnp.float32)
    sums = carry[0] + jnp.sum(wide * wide, axis=1)
    if kept_ref is not None:
        family.stored([read], [kept_ref.at[columns(index)]], smem=landing["kept"])
    if steps == 1:
        return sums, wide
    # Each block its own `[chunk, bv]` tile staged as layout inference stages a landing's (`swizzling.staged`
    # unstated): a product layout's store into a plain tile has no transfer plan on a Hopper (M-JS-324c2's
    # GH200 lowering).
    rows.at[index][...] = read
    return (sums,)


def _normed_read(normed: Normed, family, carry, *, steps: int, bv: int, columns, refs: tuple, rows,
                 landing) -> None:
    """The epilogue's read past the walk (`mosaic.Walk.read`): each row's inverse RMS from its sums
    (`precision.rms_of_total` over the head's `width` real values) and each block of the row through it, the
    weight and the activated gate (`precision.rms_weighted`), stored to the output (`refs`' first): the one
    block carried, or each read back from the rows' shared tile, each thread the entries it wrote."""
    out_ref, gate_ref, weight_ref, _ = refs
    _, inverse = rms_of_total(carry[0], normed.width, normed.epsilon)
    for block in range(steps):
        row = carry[1] if steps == 1 else family.tile_of(rows.at[block])
        row = row.astype(jnp.float32)
        gate = plgpu.load(gate_ref.at[columns(block)], layout=family.layout, optimized=False)
        weight = family.columns_of(weight_ref.at[pl.ds(block * bv, bv)]).astype(jnp.float32)
        normalized = rms_weighted(row, inverse[:, None], weight=weight[None, :],
                                  gate=activation_and_slope(gate.astype(jnp.float32), normed.activation)[0])
        family.stored([normalized.astype(out_ref.dtype)], [out_ref.at[columns(block)]], smem=landing["out"])


def _output_walk(normed: Normed | None, family, views: dict, produced, *, prologue, chunk: int, steps: int,
                 bv: int, refs: tuple, rows, columns, landing) -> mosaic.Walk:
    """`output_launch`'s walk of its `steps` value blocks, each step's output `produced(index, slots,
    loaded)` stored at its block's columns (`columns`, `refs`' first, `_output_refs`); under an epilogue
    (`normed`) its rows' sums of squares carried across the steps, zero as the chunk starts, with a row of
    one block itself (`_normed_step`), and the rows normalized past the walk (`_normed_read`,
    `mosaic.Walk.read`)."""
    if normed is None:
        def step(index, slots, carry, accumulators, loaded=None):
            out = produced(index, slots, loaded)
            family.stored([out.astype(refs[0].dtype)], [refs[0].at[columns(index)]], smem=landing["out"])
            return carry, accumulators

        return mosaic.Walk(views, step, steps=steps, prologue=prologue)
    norm = normed
    epilogue = {"steps": steps, "bv": bv, "refs": refs, "rows": rows, "columns": columns, "landing": landing}

    def normed_step(index, slots, carry, accumulators, loaded=None):
        out = produced(index, slots, loaded)
        return _normed_step(norm, family, out, index, carry, **epilogue), accumulators

    carry = (family.as_row(family.zeros((chunk,))), *((family.zeros((chunk, bv)),) if steps == 1 else ()))
    return mosaic.Walk(views, normed_step, steps=steps, prologue=prologue, carry=carry,
                       read=lambda carry, sums: _normed_read(norm, family, carry, **epilogue))


def _columns(span, first: int, bv: int, block):
    """Value block `block`'s columns of a head's `[R, HV * Dv]` rows from its first column `first`, over the
    chunk's rows `span`."""
    return span, pl.ds(first + block * bv, bv)


def _output_refs(refs, normed: Normed | None) -> tuple:
    """The output launch's refs past its write gate's, `(skip's, outputs)`: the skip's (none or one), then
    the output's and, under an epilogue (`normed`), the gate's, the weight's and the kept residual's (None
    unless kept), as `output_launch` passes and returns them (`_normed_step`'s `refs`)."""
    if normed is None:
        return refs[:-1], (refs[-1],)
    taken = 3 + normed.kept
    skip_ref, (gate_ref, weight_ref, out_ref, *kept_ref) = refs[:-taken], refs[-taken:]
    return skip_ref, (out_ref, gate_ref, weight_ref, kept_ref[0] if kept_ref else None)


def _normed_layout(normed: Normed | None, chunk: int, bv: int, value_width: int, staging: tuple,
                   landing: dict, out_type, gate, weight) -> tuple:
    """`output_launch`'s staging, landing, output type and inputs past its own with its epilogue's
    (`normed`): the head's whole value rows read back across the steps a staged tile of their own (more than
    one block), and where
    its residual is kept, a second output of the read-back rows, stored through a landing of its own; its
    `gate` and its `weight` in fp32 inputs. A row of one block is carried, no tile staged for it."""
    if normed is None:
        return staging, landing, out_type, ()
    assert gate is not None and weight is not None, "an epilogue reads its gate and weight"
    read = jnp.dtype(normed.read)
    if bv < value_width:
        staging = (*staging, swizzling.staged((value_width // bv, chunk, bv), read))
    if normed.kept:
        out_type = (out_type, jax.ShapeDtypeStruct(out_type.shape, read))
        landing = {**landing, "kept": ((chunk, bv), read)}
    return staging, landing, out_type, (gate, weight.astype(jnp.float32))


@building(lambda launch: ("recurrence_output",))
def output_launch(q, k, h, corrected, prefix: tuple, taken, beta, call: Call, tiles: Tiles, *, scale: float,
                  out_dtype, scores=None, skip=None, normed: Normed | None = None, gate=None, weight=None):
    """Each chunk's output `[R, HV * Dv]` in `out_dtype`, a program a chunk and value head and a step each
    value block, under the decay's `prefix` (`chunk_forward._Run`'s): under a decay a head its own pairs'
    scores taken once, at the first step, and staged for every block's (`algebra.chunk_scores`,
    `chunk_output`), under one a key dimension the intra launch's (`scores`, `_intra`;
    `algebra.key_decayed_output`). Where the write gate takes the keys' side (`Call.gated_keys`) the scores'
    columns take `beta` `[HV, R]` before they are rounded, and a `skip` `[HV]` adds `D v` from the values the
    step holds (there the corrected ones). A `normed` call's rows leave through its gated norm, its
    `gate` `[R, HV * Dv]` and `weight` `[Dv]` (`_normed_step`, `_normed_layout`)."""
    chunk = call.chunk
    assert skip is None or call.gated_keys, "a skip reads the values a call whose keys take the gate holds"
    width, value_width, bv = call.width, call.value_width, tiles.output_block
    held, memories, family = dataclasses.replace(call.held, rows=call.rows), call.memories, call.family
    steps = value_width // bv
    # A decay a head's prefix read as a row's vectors and as a column's, each from its own operand; a key
    # dimension's a key block at a time (`_gated_queries`).
    prefixes = prefix if call.per_key else (*prefix, *prefix)

    def program(at, q_ref, k_ref, h_ref, v_ref, *refs, streams, staging, landing):
        prefix_refs, (_taken_ref, beta_ref, *refs) = refs[:len(prefixes)], refs[len(prefixes):]
        skip_ref, outputs = _output_refs(refs, normed)
        n, h = at["chunks"], at["heads"]
        row, span = n * chunk, pl.ds(n * chunk, chunk)
        staged, *rows = staging
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
            rows, columns = prefix_refs[:len(prefix)], prefix_refs[len(prefix):]
            return (tuple(family.rows_of(ref.at[h, span]) for ref in rows),
                    tuple(family.columns_of(ref.at[h, span]) for ref in columns))

        def produced(index, slots, loaded=None):
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
                    _gated_queries(call, (q_ref, *prefix_refs), row, h, scale, gated)

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
                        Issuer(call), laid=laid, column_scale=gated, apart=call.apart)
                    operands.staged(call, scored, staged)

                out = algebra.chunk_output(queries.__getitem__, entered, width // call.block, gate, low,
                                           Parts(staged, held.dtype), value, scale, Issuer(call))
            if skip_ref:
                out = out + skip_ref[0][h] * value.value().astype(jnp.float32)
            return out

        return _output_walk(normed, family, views, produced, prologue=None if call.per_key else prologue,
                            chunk=chunk, steps=steps, bv=bv, refs=outputs, rows=rows[0] if rows else None,
                            columns=functools.partial(_columns, span, h * value_width, bv), landing=landing)

    blocks, staging = _output_streams(call, bv)
    out_type = jax.ShapeDtypeStruct((call.rows, call.heads * value_width), out_dtype)
    # The chunk's scores once (the intra launch's under a decay a key dimension), then each value block's
    # memory and local products: the model's.
    scored = 0 if call.per_key else 2 * chunk * chunk * width
    model = scored + 2 * chunk * (width + chunk) * value_width
    staging, landing, out_type, sides = _normed_layout(
        normed, chunk, bv, value_width, staging, {"out": ((chunk, bv), out_dtype)}, out_type, gate, weight)
    swizzle = None if call.per_key else _output_swizzle(call, bv, staging, landing, normed=normed is not None)
    keys = None if swizzle is None else {f"keys{p}": swizzle for p in range(call.given.parts)}
    # Its kernel named `recurrence_output` as the profiles and the gate know it (`launch_kernel` names a
    # launch for the function its program is defined in, here `output_launch`).
    inputs = (q, k if scores is None else scores, h, corrected, *prefixes, taken)
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         name="recurrence_output", grid_names=("chunks", "heads"), blocks=blocks, stages=1,
                         live=chunk_liveness(len(inputs) - 1), steps=steps, model=model,
                         held=_output_held(call, bv, swizzle, normed=normed is not None), staging=staging,
                         landing=landing,
                         # A 16-bit call's takes no reduction (an A100's asked for its tiles' 49152 bytes
                         # alone), but its epilogue's rows' sums of squares. A launch with an epilogue a
                         # program a tile, its rows normalized past its walk (`mosaic.Walk.read`), which a
                         # persistent tile's walk takes none of.
                         persistent=normed is None, reduces=held.parts > 1 or normed is not None,
                         swizzles=keys)(
        *inputs, beta, *(() if skip is None else (skip.astype(jnp.float32),)), *sides)
